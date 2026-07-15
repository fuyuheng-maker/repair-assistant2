import os
import hashlib
import logging
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

import asyncio

os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"

import uuid
import json
import shutil
import secrets
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import pdfplumber
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
from io import BytesIO
import base64
import bcrypt
import time
import threading
import re
from collections import defaultdict

# ---------- 导入 features 和 reviewer ----------
from features import router as features_router, init_features
from reviewer import reviewer_router, init_reviewer

# ========== 配置区（从环境变量读取，确保安全）==========
def get_env_or_default(key: str, default: str = None) -> str:
    """从环境变量读取配置，若不存在则使用默认值或报错"""
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"环境变量 {key} 未设置，请在 .env 文件中配置")
    return value

# API 密钥（必须设置）
OPENAI_API_KEY = get_env_or_default("OPENAI_API_KEY", "")
OPENAI_BASE_URL = get_env_or_default("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MULTIMODAL_MODEL = get_env_or_default("MULTIMODAL_MODEL", "qwen3.5-omni-plus")
TEXT_MODEL = get_env_or_default("TEXT_MODEL", "qwen-plus-0112")

# 模型配置
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB 上传大小限制

# JWT 密钥（优先从环境变量读取，否则生成并持久化）
_jwt_key_file = os.path.join(CHROMA_PERSIST_DIR, ".jwt_secret")
def _get_or_create_jwt_secret() -> str:
    env_key = os.environ.get("JWT_SECRET_KEY")
    if env_key:
        return env_key
    # 持久化：首次生成后写入文件，避免重启后密钥变化导致全员登出
    if os.path.exists(_jwt_key_file):
        with open(_jwt_key_file, "r") as f:
            return f.read().strip()
    new_key = secrets.token_urlsafe(32)
    os.makedirs(os.path.dirname(_jwt_key_file), exist_ok=True)
    with open(_jwt_key_file, "w") as f:
        f.write(new_key)
    return new_key

SECRET_KEY = _get_or_create_jwt_secret()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

USERS_FILE = "users.json"
FILE_REGISTRY = "file_registry.json"
PROJECTS_FILE = "projects.json"
SCHEDULE_FILE = "schedule.json"
_file_registry_lock = threading.Lock()
_users_lock = threading.RLock()
_projects_lock = threading.Lock()
_schedule_lock = threading.Lock()
_users_cache: Optional[List[Dict]] = None
_users_cache_time: float = 0

def load_users() -> List[Dict]:
    global _users_cache, _users_cache_time
    with _users_lock:
        # 使用缓存，5秒内直接返回
        if _users_cache is not None and time.time() - _users_cache_time < 5:
            return _users_cache
        if not os.path.exists(USERS_FILE):
            return []
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                _users_cache = json.load(f)
                _users_cache_time = time.time()
                return _users_cache
        except json.JSONDecodeError:
            # JSON 损坏时返回空列表，避免崩溃
            logger.warning(f"{USERS_FILE} is corrupted, returning empty list")
            return []

def save_users(users: List[Dict]):
    global _users_cache, _users_cache_time
    with _users_lock:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        _users_cache = users
        _users_cache_time = time.time()

def get_user(username: str) -> Optional[Dict]:
    for u in load_users():
        if u["username"] == username:
            return u
    return None

def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

app = FastAPI()

# CORS 配置
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False)
)

def get_user_collection(username: str):
    """获取用户专属的知识库 collection"""
    import hashlib
    name_hash = hashlib.md5(username.encode('utf-8')).hexdigest()[:16]
    collection_name = f"u_{name_hash}"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 128, "hnsw:search_ef": 128}
    )
    return collection

def get_global_collection():
    """获取全局共享知识库（用于管理员发布的公共知识）"""
    collection = chroma_client.get_or_create_collection(
        name="global_knowledge",
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 128, "hnsw:search_ef": 128}
    )
    return collection

os.makedirs(UPLOAD_DIR, exist_ok=True)

security = HTTPBearer(auto_error=False)

# ---------- 异常处理中间件 ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP错误: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"answer": f"请求错误: {exc.detail}", "references": [], "answer_id": str(uuid.uuid4()), "retrieved_snippets": [], "suggested_questions": []}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"服务器错误: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"answer": f"服务器错误: {str(exc)}", "references": [], "answer_id": str(uuid.uuid4()), "retrieved_snippets": [], "suggested_questions": []}
    )

# ---------- 速率限制（中间件方式） ----------
_rate_limit_store = defaultdict(list)  # {ip: [timestamp, ...]}
RATE_LIMIT_MAX_REQUESTS = 30  # 每分钟最大请求数
RATE_LIMIT_WINDOW = 60  # 秒
# 不限速的路径前缀
RATE_LIMIT_EXEMPT_PREFIXES = ("/static", "/login")

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """基于 IP 的简单速率限制中间件"""
    path = request.url.path
    # 静态资源和登录页面不限速
    if not any(path.startswith(p) for p in RATE_LIMIT_EXEMPT_PREFIXES):
        client_ip = request.client.host
        now = time.time()
        _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]
        if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(status_code=429, content={"detail": "请求过于频繁，请稍后再试"})
        _rate_limit_store[client_ip].append(now)
        # 定期清理过期 IP（每 100 次请求清理一次）
        if len(_rate_limit_store) > 100 and hash(now) % 100 == 0:
            for ip in list(_rate_limit_store.keys()):
                if not _rate_limit_store[ip]:
                    del _rate_limit_store[ip]
    return await call_next(request)

# ---------- JWT 工具 ----------
def create_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data = {"sub": username, "role": role, "exp": expire}
    token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"为用户 {username} 创建了token")
    return token

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, str]:
    logger.info(f"认证检查 - 收到的credentials: {credentials}")
    if credentials is None:
        logger.warning("没有找到Authorization header")
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials
    logger.info(f"收到的token: {token[:50]}...")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role", "worker")
        logger.info(f"从token中解析出用户: {username}, 角色: {role}")
        user = get_user(username)
        if not user:
            logger.warning(f"用户 {username} 不存在")
            raise HTTPException(status_code=401, detail="User not found")
        return {"username": username, "role": user["role"]}
    except JWTError as e:
        logger.error(f"JWT解析错误: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(current_user: Dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user

# ---------- PDF 解析 ----------
def semantic_chunk_text(text: str, page_num: int, max_chunk_size: int = 800, overlap: int = 100):
    """按段落语义切分文本，保持上下文连贯"""
    # 按段落分割（支持多种换行格式）
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n(?=\d+\.)', text) if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        # 如果当前段落太长，需要进一步切分
        if para_size > max_chunk_size:
            # 先保存当前积累的块
            if current_chunk:
                chunks.append({
                    "page": page_num,
                    "type": "text",
                    "content": "\n\n".join(current_chunk)
                })
                # 保留最后一段作为重叠
                overlap_text = current_chunk[-1][-overlap:] if len(current_chunk[-1]) > overlap else current_chunk[-1]
                current_chunk = [overlap_text]
                current_size = len(overlap_text)
            
            # 长段落按句子切分
            sentences = re.split(r'(?<=[。！？；.!?;])\s*', para)
            for sent in sentences:
                if current_size + len(sent) > max_chunk_size and current_chunk:
                    chunks.append({
                        "page": page_num,
                        "type": "text",
                        "content": "\n\n".join(current_chunk)
                    })
                    # 保留重叠
                    overlap_text = current_chunk[-1][-overlap:] if len(current_chunk[-1]) > overlap else current_chunk[-1]
                    current_chunk = [overlap_text]
                    current_size = len(overlap_text)
                current_chunk.append(sent)
                current_size += len(sent)
        else:
            # 正常段落
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append({
                    "page": page_num,
                    "type": "text",
                    "content": "\n\n".join(current_chunk)
                })
                # 保留重叠
                overlap_text = current_chunk[-1][-overlap:] if len(current_chunk[-1]) > overlap else current_chunk[-1]
                current_chunk = [overlap_text, para]
                current_size = len(overlap_text) + para_size
            else:
                current_chunk.append(para)
                current_size += para_size
    
    # 保存最后一块
    if current_chunk:
        chunks.append({
            "page": page_num,
            "type": "text",
            "content": "\n\n".join(current_chunk)
        })
    
    return chunks

def extract_text_and_images(pdf_path: str):
    """提取 PDF 文本和图片，文本按语义切分，图片直接保存信息（不调用AI）"""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"开始解析PDF，共 {total_pages} 页")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # 每10页输出一次进度
            if page_num % 10 == 0 or page_num == 1:
                logger.info(f"正在解析第 {page_num}/{total_pages} 页...")
            
            text = page.extract_text()
            if text:
                # 使用语义切分代替整页存储
                text_chunks = semantic_chunk_text(text, page_num)
                chunks.extend(text_chunks)
            
            # 提取页面中的图片基本信息（不调用AI，省时）
            if page.images:
                for img_idx, img_info in enumerate(page.images):
                    try:
                        # 只提取图片的基本信息，不做AI描述
                        width = int(img_info.get("x1", 0) - img_info.get("x0", 0))
                        height = int(img_info.get("bottom", 0) - img_info.get("top", 0))
                        chunks.append({
                            "page": page_num, "type": "image",
                            "content": f"[图片 {img_idx+1}] 位置: 第{page_num}页，尺寸: {width}x{height}px"
                        })
                    except Exception as e:
                        logger.warning(f"PDF第{page_num}页图片{img_idx+1}提取失败: {e}")
        
        logger.info(f"PDF解析完成，共提取 {len(chunks)} 个文本块")
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    """批量嵌入文本以提升性能"""
    if not texts:
        return []
    
    logger.info(f"开始生成嵌入向量，共 {len(texts)} 个文本块...")
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True).tolist()
    logger.info(f"嵌入向量生成完成")
    
    return embeddings

# ---------- 核心 API ----------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user: Dict = Depends(get_current_user)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "请上传PDF文件")
    # 读取文件内容并检查大小
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "文件为空")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"文件过大，最大允许 {MAX_UPLOAD_SIZE // (1024*1024)}MB")
    # 校验 PDF magic bytes，防止伪装文件
    if not content.startswith(b'%PDF-'):
        raise HTTPException(400, "无效的PDF文件，请上传真正的PDF格式文件")
    file_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(content)

    try:
        # 使用 asyncio.to_thread 避免阻塞事件循环，添加超时保护（300秒，大文件需要更长时间）
        try:
            chunks = await asyncio.wait_for(
                asyncio.to_thread(extract_text_and_images, pdf_path),
                timeout=300.0
            )
        except asyncio.TimeoutError:
            logger.error(f"PDF解析超时: {file.filename}")
            raise HTTPException(400, "PDF文件过大或解析超时，请尝试更小的文件或等待后重试")
        
        if not chunks:
            os.unlink(pdf_path)
            raise HTTPException(500, "PDF解析未获得任何内容")

        # 获取用户专属的知识库
        user_collection = get_user_collection(user["username"])
        
        texts = [c["content"] for c in chunks]
        # 嵌入生成也添加超时保护
        try:
            embeddings = await asyncio.wait_for(
                asyncio.to_thread(embed_texts, texts),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.error(f"嵌入生成超时: {file.filename}")
            raise HTTPException(400, "文本嵌入超时，请尝试更小的PDF文件")
        user_collection.add(
            ids=[f"{file_id}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            metadatas=[{"page": c["page"], "type": c["type"], "file_id": file_id} for c in chunks],
            documents=texts
        )
    except HTTPException:
        raise
    except Exception as e:
        # 清理残留文件
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        raise HTTPException(500, f"PDF处理失败: {str(e)}")
    
    # 检查是否已存在同名文件，避免重复上传
    registry = _load_registry()
    existing = next((f for f in registry if f["username"] == user["username"] and f["filename"] == file.filename), None)
    if existing:
        raise HTTPException(400, f"已存在同名文件「{file.filename}」，如需更新请先删除旧文件")
    
    # 记录文件到注册表
    _register_file(file_id, file.filename, user["username"], len(chunks))
    
    return {"file_id": file_id, "chunk_count": len(chunks), "user": user["username"]}

# ---------- 文件注册表 ----------
def _load_registry() -> List[Dict]:
    with _file_registry_lock:
        abs_path = os.path.abspath(FILE_REGISTRY)
        if not os.path.exists(abs_path):
            logger.warning(f"文件注册表不存在: {abs_path}")
            return []
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"加载文件注册表成功，共 {len(data)} 个文件")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"文件注册表解析失败: {e}")
            return []

def _save_registry(registry: List[Dict]):
    with _file_registry_lock:
        with open(FILE_REGISTRY, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

def _register_file(file_id: str, filename: str, username: str, chunk_count: int):
    registry = _load_registry()
    registry.append({
        "file_id": file_id,
        "filename": filename,
        "username": username,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.now().isoformat()
    })
    _save_registry(registry)

@app.get("/api/files")
async def list_files(user: Dict = Depends(get_current_user)):
    """列出所有文件（共享知识库）"""
    registry = _load_registry()
    registry.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return {"files": registry}

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str, user: Dict = Depends(get_current_user)):
    """删除指定文件及其知识库条目"""
    logger.info(f"删除请求 - 用户: {user['username']}, 文件ID: {file_id}")
    
    registry = _load_registry()
    file_record = next((f for f in registry if f["file_id"] == file_id), None)
    
    if not file_record:
        logger.warning(f"文件不存在 - 文件ID: {file_id}")
        raise HTTPException(404, "文件不存在")
    
    # 检查权限
    if file_record["username"] != user["username"] and user["role"] != "admin":
        logger.warning(f"权限不足 - 用户: {user['username']}, 文件ID: {file_id}, 文件所有者: {file_record['username']}")
        raise HTTPException(403, "无权删除此文件")
    
    # 删除 PDF 文件
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if os.path.exists(pdf_path):
        try:
            os.unlink(pdf_path)
            logger.info(f"已删除PDF文件: {pdf_path}")
        except Exception as e:
            logger.error(f"删除PDF文件失败: {e}")
            raise HTTPException(500, f"删除文件失败: {str(e)}")
    else:
        logger.warning(f"PDF文件不存在: {pdf_path}")
    
    # 删除知识库中的条目（包括用户个人知识库和全局知识库）
    try:
        # 删除用户个人知识库中的条目
        user_collection = get_user_collection(user["username"])
        all_data = user_collection.get()
        if all_data and all_data.get("ids") and len(all_data["ids"]) > 0:
            ids_to_delete = []
            for i, mid in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i] if all_data["metadatas"] and i < len(all_data["metadatas"]) else {}
                if meta.get("file_id") == file_id:
                    ids_to_delete.append(mid)
            
            if ids_to_delete:
                user_collection.delete(ids=ids_to_delete)
                logger.info(f"删除用户知识库中的 {len(ids_to_delete)} 个条目")
            else:
                logger.warning(f"未找到文件对应的用户知识库条目")
        
        # 删除全局知识库中的条目
        global_collection = get_global_collection()
        all_global_data = global_collection.get()
        if all_global_data and all_global_data.get("ids") and len(all_global_data["ids"]) > 0:
            global_ids_to_delete = []
            for i, mid in enumerate(all_global_data["ids"]):
                meta = all_global_data["metadatas"][i] if all_global_data["metadatas"] and i < len(all_global_data["metadatas"]) else {}
                if meta.get("file_id") == file_id or meta.get("username") == user["username"]:
                    global_ids_to_delete.append(mid)
            
            if global_ids_to_delete:
                global_collection.delete(ids=global_ids_to_delete)
                logger.info(f"删除全局知识库中的 {len(global_ids_to_delete)} 个条目")
            else:
                logger.warning(f"未找到文件对应的全局知识库条目")
    except Exception as e:
        logger.error(f"删除知识库条目失败: {e}")
        # 继续执行，不中断
    
    # 从注册表移除
    registry = [f for f in registry if f["file_id"] != file_id]
    _save_registry(registry)
    
    logger.info(f"删除成功 - 用户: {user['username']}, 文件: {file_record['filename']}")
    return {"message": f"已删除文件「{file_record['filename']}」"}

@app.get("/api/files/{file_id}/download")
async def download_file(file_id: str, user: Dict = Depends(get_current_user), preview: bool = False):
    """下载或预览指定文件（所有已登录用户均可预览）"""
    logger.info(f"========== 文件请求开始 ==========")
    logger.info(f"用户: {user['username']}, 角色: {user['role']}")
    logger.info(f"文件ID: {file_id}, 预览模式: {preview}")
    
    registry = _load_registry()
    logger.info(f"注册表文件数量: {len(registry)}")
    
    file_record = next((f for f in registry if f["file_id"] == file_id), None)
    logger.info(f"找到文件记录: {file_record is not None}")
    
    if file_record:
        logger.info(f"文件名: {file_record['filename']}")
        logger.info(f"文件所有者: {file_record['username']}")
    
    if not file_record:
        logger.warning(f"文件不存在 - 文件ID: {file_id}")
        raise HTTPException(404, "文件不存在")
    
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(pdf_path):
        logger.error(f"文件已丢失 - {pdf_path}")
        raise HTTPException(404, "文件已丢失")
    
    logger.info(f"文件请求成功 - 用户: {user['username']}, 文件: {file_record['filename']}, 预览: {preview}")
    from fastapi.responses import FileResponse
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=file_record["filename"],
        content_disposition_type='inline' if preview else 'attachment'
    )

@app.get("/download-deploy")
async def download_deploy_package():
    """下载部署包（用于部署到龙芯服务器）"""
    deploy_path = "./repair-assistant-deploy.zip"
    if not os.path.exists(deploy_path):
        raise HTTPException(404, "部署包不存在")
    return FileResponse(deploy_path, media_type="application/zip", filename="repair-assistant-deploy.zip")

@app.post("/ask")
async def ask_question(
    question: str = Form(""),
    image: UploadFile = File(None),
    user: Dict = Depends(get_current_user)
):
    try:
        if not question and image:
            question = "请分析这张图片中的设备状态或故障"
        
        if not question and not image:
            return JSONResponse(
                status_code=400,
                content={
                    "answer": "请输入问题或上传图片",
                    "references": [],
                    "answer_id": str(uuid.uuid4()),
                    "retrieved_snippets": [],
                    "suggested_questions": []
                }
            )
            
        image_desc = ""
        if image:
            try:
                img_bytes = await image.read()
                if len(img_bytes) > 10 * 1024 * 1024:
                    raise HTTPException(400, "图片过大，最大允许10MB")
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                resp = client.chat.completions.create(
                    model=MULTIMODAL_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请描述这张图片中的设备异常或检修相关细节："},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }]
                )
                image_desc = resp.choices[0].message.content
                question = question + "（附带图片描述：" + image_desc + "）"
            except Exception as e:
                logger.error(f"图片分析失败: {e}")
                question += "（附带图片，但分析失败）"

        q_emb = embed_texts([question])[0]
    
        user_collection = get_user_collection(user["username"])
    
        total_count = user_collection.count()
        if total_count == 0:
            return {
                "answer": "⚠️ 您的知识库为空，请先上传检修手册（PDF）后再提问。\n\n操作步骤：\n1. 点击左侧菜单「上传手册」\n2. 选择 PDF 格式的检修手册文件\n3. 点击「上传并建立知识库」\n4. 上传成功后即可开始提问",
                "references": [],
                "answer_id": str(uuid.uuid4()),
                "retrieved_snippets": [],
                "suggested_questions": []
            }
        
        text_results = user_collection.query(
            query_embeddings=[q_emb], 
            n_results=15,
            where={"type": "text"}
        )
        
        all_docs = list(text_results["documents"][0])
        all_metas = list(text_results["metadatas"][0])
        
        valid_items = []
        seen_content_hashes = set()
        
        for doc, meta in zip(all_docs, all_metas):
            if "暂未启用多模态分析" in doc or "图片提取失败" in doc or "图片坐标无效" in doc:
                continue
            if len(doc.strip()) < 20:
                continue
            content_hash = hashlib.md5(doc[:100].encode('utf-8')).hexdigest()
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            valid_items.append((doc, meta))
        
        if not valid_items:
            return {
                "answer": "⚠️ 未在知识库中找到与您问题相关的内容。\n\n建议：\n1. 检查是否已上传相关的检修手册\n2. 尝试使用更具体的关键词提问\n3. 如果手册内容较少，可尝试上传更完整的资料",
                "references": [],
                "answer_id": str(uuid.uuid4()),
                "retrieved_snippets": [],
                "suggested_questions": []
            }
        
        valid_items = valid_items[:10]
        valid_contexts = [item[0] for item in valid_items]
        valid_refs = [(item[1], item[0]) for item in valid_items]
        
        contexts = "\n\n---\n\n".join(valid_contexts)
        
        retrieved_snippets = []
        for i, (doc, meta) in enumerate(valid_items[:5]):
            page = meta.get("page", "?")
            source = meta.get("file_id", "")[:8] + "..."
            snippet = doc[:100] + "..." if len(doc) > 100 else doc
            retrieved_snippets.append({
                "index": i + 1,
                "page": page,
                "source": source,
                "snippet": snippet
            })
        
        format_instruction = ""
        if "表格" in question or "清单" in question or "列表" in question:
            format_instruction = """
【重要格式要求】
用户明确要求表格格式，请必须使用 Markdown 表格输出！
表格格式示例：
| 序号 | 项目 | 规格/参数 | 备注 |
|------|------|-----------|------|
| 1    | xxx  | xxx       | xxx  |
"""
        elif "步骤" in question or "流程" in question or "怎么" in question:
            format_instruction = """
【重要格式要求】
用户询问操作流程，请必须使用步骤列表格式：
1. 第一步：xxx
2. 第二步：xxx
...
"""
        
        system_prompt = """你是一名经验丰富的设备检修专家助手。你的职责是：
1. 根据上传的检修手册内容，准确回答用户的检修相关问题
2. 如果用户的问题与检修无关，礼貌地引导用户回到检修话题
3. 始终保持专业、简洁的回答风格"""
        
        prompt = f"""{system_prompt}

以下是从检修手册中检索到的相关内容：

【检修手册内容】
{contexts}
{format_instruction}
【回答要求】
1. 严格依据上述检修手册内容回答，不要编造信息
2. 如果用户要求特定格式（如表格），必须严格按照该格式输出
3. 如果手册中没有明确答案，请诚实说明"手册中未找到相关内容"
4. 回答要简洁专业，避免冗余重复
5. 引用手册内容时，请注明页码

【用户问题】
{question}

【专家回答】
（回答完成后，请另起一行，用【推荐问题】标记，给出3个与当前话题相关的后续问题，格式如下：
【推荐问题】
1. xxx？
2. xxx？
3. xxx？）"""

        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            full_response = resp.choices[0].message.content
        except Exception as ai_err:
            logger.error(f"AI服务调用失败: {ai_err}")
            full_response = "⚠️ AI服务暂时不可用，已切换到本地模式。\n\n根据知识库内容，为您整理以下信息：\n\n"
            for i, (doc, meta) in enumerate(valid_items[:5]):
                page = meta.get("page", "?")
                source = meta.get("file_id", "")[:8] + "..."
                full_response += f"【第{page}页】\n{doc[:200]}...\n\n"
            full_response += "【推荐问题】\n1. 请描述具体的故障现象？\n2. 需要哪些检修工具？\n3. 相关设备型号是什么？"
        
        answer = full_response
        suggested_questions = []
        
        if "【推荐问题】" in full_response:
            parts = full_response.split("【推荐问题】")
            answer = parts[0].strip()
            if len(parts) > 1:
                questions_text = parts[1]
                questions = re.findall(r'\d+\.\s*(.+？)', questions_text)
                suggested_questions = questions[:3]

        references = []
        for meta, doc in valid_refs:
            page_info = meta.get("page", "?")
            if meta.get("source") == "user_case":
                page_info = f"案例《{meta.get('title', '')}》"
            elif meta.get("source") == "user_correction":
                page_info = "用户修正"
            references.append({
                "page": page_info,
                "content": doc[:200] + "..." if len(doc) > 200 else doc
            })

        return {
            "answer": answer, 
            "references": references, 
            "answer_id": str(uuid.uuid4()),
            "retrieved_snippets": retrieved_snippets,
            "suggested_questions": suggested_questions
        }
    except Exception as e:
        logger.error(f"Ask API error: {str(e)}", exc_info=True)
        return {
            "answer": "⚠️ 服务暂时不可用，请稍后重试。",
            "references": [],
            "answer_id": str(uuid.uuid4()),
            "retrieved_snippets": [],
            "suggested_questions": []
        }

# ---------- 注册 ----------
@app.post("/api/register")
async def api_register(username: str = Form(...), password: str = Form(...)):
    if not username.strip() or not password.strip():
        raise HTTPException(400, "用户名和密码不能为空")
    if len(username) < 3:
        raise HTTPException(400, "用户名至少3个字符")
    if len(password) < 6:
        raise HTTPException(400, "密码至少6个字符")
    # 在同一把锁内完成检查和写入，避免竞态条件
    with _users_lock:
        users = load_users()
        if any(u["username"] == username for u in users):
            raise HTTPException(400, "用户名已存在")
        users.append({
            "username": username,
            "password_hash": hash_password(password),
            "role": "worker"
        })
        save_users(users)
    return {"message": "注册成功，请登录"}

# ---------- 登录 ----------
@app.post("/api/login")
async def api_login(username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(400, "用户名或密码错误")
    token = create_token(username, user["role"])
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

@app.post("/api/register")
async def api_register(username: str = Form(...), password: str = Form(...)):
    if get_user(username):
        raise HTTPException(400, "用户名已存在")
    if len(username) < 3:
        raise HTTPException(400, "用户名至少需要3个字符")
    if len(password) < 6:
        raise HTTPException(400, "密码至少需要6个字符")
    
    users = load_users()
    users.append({
        "username": username,
        "password_hash": hash_password(password),
        "role": "worker"
    })
    save_users(users)
    logger.info(f"新用户注册: {username}")
    return {"message": "注册成功，请登录"}

@app.get("/login")
async def login_page():
    return HTMLResponse(open("static/login.html", encoding="utf-8").read())

# ---------- 健康检查 ----------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "repair-assistant"}

# ---------- 主页 ----------
@app.get("/")
async def main_page():
    html = open("static/index.html", encoding="utf-8").read()
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})

# ---------- 注入依赖并挂载 features 路由 ----------
init_features(embed_model, get_user_collection, client, TEXT_MODEL, get_current_user, require_admin)
app.include_router(features_router, prefix="/api")

# ---------- 注入依赖并挂载 reviewer 路由 ----------
init_reviewer(create_token, load_users, verify_password)
app.include_router(reviewer_router, prefix="/api")

# ---------- 运维项目管理 API ----------
def _load_projects() -> List[Dict]:
    with _projects_lock:
        if not os.path.exists(PROJECTS_FILE):
            return []
        try:
            with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

def _save_projects(projects: List[Dict]):
    with _projects_lock:
        with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)

@app.get("/api/projects")
async def list_projects(user: Dict = Depends(get_current_user)):
    projects = _load_projects()
    user_projects = [p for p in projects if p["username"] == user["username"] or user["role"] == "admin"]
    user_projects.sort(key=lambda x: x["created_at"], reverse=True)
    return {"projects": user_projects}

@app.post("/api/projects")
async def create_project(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    status = data.get("status", "pending")
    assignee = data.get("assignee", "").strip()
    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")
    
    if not name:
        raise HTTPException(400, "项目名称不能为空")
    
    projects = _load_projects()
    project = {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": description,
        "username": user["username"],
        "created_at": datetime.now().isoformat(),
        "status": status,
        "assignee": assignee,
        "start_date": start_date,
        "end_date": end_date,
        "progress": 0,
        "tasks": [],
        "documents": [],
        "comments": []
    }
    projects.append(project)
    _save_projects(projects)
    logger.info(f"用户 {user['username']} 创建项目: {name}")
    
    return {"message": "项目创建成功", "project": project}

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, user: Dict = Depends(get_current_user)):
    projects = _load_projects()
    project = next((p for p in projects if p["id"] == project_id), None)
    
    if not project:
        raise HTTPException(404, "项目不存在")
    
    if project["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权访问此项目")
    
    return project

@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    projects = _load_projects()
    project = next((p for p in projects if p["id"] == project_id), None)
    
    if not project:
        raise HTTPException(404, "项目不存在")
    
    if project["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权修改此项目")
    
    if "name" in data:
        project["name"] = data["name"].strip()
    if "description" in data:
        project["description"] = data["description"].strip()
    if "status" in data:
        project["status"] = data["status"]
    if "assignee" in data:
        project["assignee"] = data["assignee"].strip()
    if "start_date" in data:
        project["start_date"] = data["start_date"]
    if "end_date" in data:
        project["end_date"] = data["end_date"]
    if "progress" in data:
        project["progress"] = min(100, max(0, data["progress"]))
    
    project["updated_at"] = datetime.now().isoformat()
    _save_projects(projects)
    logger.info(f"用户 {user['username']} 更新项目: {project['name']}")
    
    return {"message": "项目更新成功", "project": project}

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str, user: Dict = Depends(get_current_user)):
    projects = _load_projects()
    project = next((p for p in projects if p["id"] == project_id), None)
    
    if not project:
        raise HTTPException(404, "项目不存在")
    
    if project["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权删除此项目")
    
    projects = [p for p in projects if p["id"] != project_id]
    _save_projects(projects)
    logger.info(f"用户 {user['username']} 删除项目: {project['name']}")
    
    return {"message": "项目删除成功"}

@app.post("/api/projects/{project_id}/tasks")
async def add_project_task(project_id: str, request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    title = data.get("title", "").strip()
    description = data.get("description", "").strip()
    
    if not title:
        raise HTTPException(400, "任务标题不能为空")
    
    projects = _load_projects()
    project = next((p for p in projects if p["id"] == project_id), None)
    
    if not project:
        raise HTTPException(404, "项目不存在")
    
    if project["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权访问此项目")
    
    task = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "created_by": user["username"]
    }
    
    if "tasks" not in project:
        project["tasks"] = []
    project["tasks"].append(task)
    project["updated_at"] = datetime.now().isoformat()
    _save_projects(projects)
    
    return {"message": "任务添加成功", "task": task}

@app.put("/api/projects/{project_id}/tasks/{task_id}")
async def update_project_task(project_id: str, task_id: str, request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    
    projects = _load_projects()
    project = next((p for p in projects if p["id"] == project_id), None)
    
    if not project:
        raise HTTPException(404, "项目不存在")
    
    if project["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权访问此项目")
    
    task = next((t for t in project.get("tasks", []) if t["id"] == task_id), None)
    if not task:
        raise HTTPException(404, "任务不存在")
    
    if "title" in data:
        task["title"] = data["title"].strip()
    if "description" in data:
        task["description"] = data["description"].strip()
    if "status" in data:
        task["status"] = data["status"]
    
    task["updated_at"] = datetime.now().isoformat()
    project["updated_at"] = datetime.now().isoformat()
    _save_projects(projects)
    
    return {"message": "任务更新成功", "task": task}

@app.delete("/api/projects/{project_id}/tasks/{task_id}")
async def delete_project_task(project_id: str, task_id: str, user: Dict = Depends(get_current_user)):
    projects = _load_projects()
    project = next((p for p in projects if p["id"] == project_id), None)
    
    if not project:
        raise HTTPException(404, "项目不存在")
    
    if project["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权访问此项目")
    
    project["tasks"] = [t for t in project.get("tasks", []) if t["id"] != task_id]
    project["updated_at"] = datetime.now().isoformat()
    _save_projects(projects)
    
    return {"message": "任务删除成功"}

@app.post("/api/projects/{project_id}/comments")
async def add_project_comment(project_id: str, request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    content = data.get("content", "").strip()
    
    if not content:
        raise HTTPException(400, "评论内容不能为空")
    
    projects = _load_projects()
    project = next((p for p in projects if p["id"] == project_id), None)
    
    if not project:
        raise HTTPException(404, "项目不存在")
    
    if project["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权访问此项目")
    
    comment = {
        "id": str(uuid.uuid4()),
        "content": content,
        "created_at": datetime.now().isoformat(),
        "created_by": user["username"]
    }
    
    if "comments" not in project:
        project["comments"] = []
    project["comments"].append(comment)
    project["updated_at"] = datetime.now().isoformat()
    _save_projects(projects)
    
    return {"message": "评论添加成功", "comment": comment}

# ---------- 定时任务管理 API ----------
def _load_schedule() -> List[Dict]:
    with _schedule_lock:
        if not os.path.exists(SCHEDULE_FILE):
            return []
        try:
            with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

def _save_schedule(tasks: List[Dict]):
    with _schedule_lock:
        with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

@app.get("/api/schedule")
async def list_schedule(user: Dict = Depends(get_current_user)):
    tasks = _load_schedule()
    user_tasks = [t for t in tasks if t["username"] == user["username"] or user["role"] == "admin"]
    user_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    return {"tasks": user_tasks}

@app.post("/api/schedule")
async def create_schedule(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    name = data.get("name", "").strip()
    cron_expression = data.get("cron_expression", "").strip()
    description = data.get("description", "").strip()
    
    if not name or not cron_expression:
        raise HTTPException(400, "任务名称和Cron表达式不能为空")
    
    tasks = _load_schedule()
    tasks.append({
        "id": str(uuid.uuid4()),
        "name": name,
        "cron_expression": cron_expression,
        "description": description,
        "username": user["username"],
        "active": True,
        "created_at": datetime.now().isoformat()
    })
    _save_schedule(tasks)
    
    return {"message": "定时任务创建成功"}

# ---------- 知识库统计 API ----------
@app.get("/api/stats")
async def get_stats(user: Dict = Depends(get_current_user)):
    user_collection = get_user_collection(user["username"])
    total_chunks = user_collection.count()
    
    registry = _load_registry()
    user_files = [f for f in registry if f["username"] == user["username"]]
    
    return {
        "total_files": len(user_files),
        "total_chunks": total_chunks,
        "storage_usage": sum(f["chunk_count"] * 1024 for f in user_files) // 1024,
        "last_upload": user_files[0]["uploaded_at"] if user_files else None
    }

# ---------- 通用路由（必须放在最后）----------
@app.get("/{file_name}")
async def serve_html(file_name: str):
    """直接访问HTML文件"""
    if file_name.endswith('.html'):
        file_path = os.path.join("static", file_name)
        if os.path.exists(file_path):
            return FileResponse(file_path)
    raise HTTPException(404, "Not Found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)