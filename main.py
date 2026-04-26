import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import uuid
import shutil
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import pdfplumber
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
from io import BytesIO
import base64

# ---------- 导入 features 和 reviewer ----------
from features import router as features_router, init_features
from reviewer import reviewer_router, init_reviewer

# ========== 配置区 ==========
OPENAI_API_KEY = "你的有效API-Key"   # 替换为真实 Key
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MULTIMODAL_MODEL = "qwen3.5-omni-plus"
TEXT_MODEL = "qwen-plus-0112"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"

SECRET_KEY = "repair-assistant-secret-key-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "worker": {"password": "worker123", "role": "worker"},
}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(name="repair_knowledge")
os.makedirs(UPLOAD_DIR, exist_ok=True)

security = HTTPBearer(auto_error=False)

# ---------- JWT 工具 ----------
def create_token(username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, str]:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role", "worker")
        if username not in USERS:
            raise HTTPException(status_code=401, detail="User not found")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(current_user: Dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user

# ---------- PDF 解析 ----------
def extract_text_and_images(pdf_path: str):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                chunks.append({"page": page_num, "type": "text", "content": text.strip()})
            for img_idx, _ in enumerate(page.images):
                chunks.append({
                    "page": page_num, "type": "image",
                    "content": f"[图片 {img_idx+1}] （PDF图片暂未分析）"
                })
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    return embed_model.encode(texts).tolist()

# ---------- 核心 API ----------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user: Dict = Depends(get_current_user)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "请上传PDF文件")
    file_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = extract_text_and_images(pdf_path)
    if not chunks:
        raise HTTPException(500, "PDF解析未获得任何内容")

    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)
    collection.add(
        ids=[f"{file_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        metadatas=[{"page": c["page"], "type": c["type"], "file_id": file_id} for c in chunks],
        documents=texts
    )
    return {"file_id": file_id, "chunk_count": len(chunks)}

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    image: UploadFile = File(None),
    user: Dict = Depends(get_current_user)
):
    if image:
        try:
            img_bytes = await image.read()
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
            print(f"图片分析失败: {e}")
            question += "（附带图片，但分析失败）"

    q_emb = embed_texts([question])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=5)

    contexts = "\n---\n".join(results["documents"][0])
    prompt = f"""你是一名经验丰富的设备检修专家，请严格依据以下检修手册内容回答用户问题。
如果手册中没有明确答案，请如实说明，切勿编造。

检修手册片段：
{contexts}

用户问题：{question}

专家回答："""

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    answer = resp.choices[0].message.content

    references = []
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        references.append({
            "page": meta["page"],
            "content": doc[:200] + "..." if len(doc) > 200 else doc
        })

    return {"answer": answer, "references": references, "answer_id": str(uuid.uuid4())}

# ---------- 登录 ----------
@app.post("/api/login")
async def api_login(username: str = Form(...), password: str = Form(...)):
    user = USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(400, "用户名或密码错误")
    token = create_token(username, user["role"])
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

@app.get("/login")
async def login_page():
    return HTMLResponse(open("static/login.html", encoding="utf-8").read())


# ---------- 主页 ----------
@app.get("/")
async def main_page():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

# ---------- 注入依赖并挂载 features 路由 ----------
init_features(embed_model, collection, client, TEXT_MODEL, get_current_user, require_admin)
app.include_router(features_router, prefix="/api")

# ---------- 注入依赖并挂载 reviewer 路由 ----------
init_reviewer(SECRET_KEY, ALGORITHM, USERS)
app.include_router(reviewer_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)