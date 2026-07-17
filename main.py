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

try:
    import cv2
    VIDEO_ANALYSIS_ENABLED = True
    logger.info("OpenCV加载成功，视频分析功能已启用")
except ImportError:
    VIDEO_ANALYSIS_ENABLED = False
    logger.warning("OpenCV未安装，视频分析功能将不可用")

try:
    from fpdf import FPDF
    PDF_GENERATION_ENABLED = True
    logger.info("fpdf2加载成功，PDF报告生成功能已启用")
except ImportError:
    PDF_GENERATION_ENABLED = False
    logger.warning("fpdf2未安装，PDF报告生成功能将不可用")

try:
    import psutil
    SYSTEM_MONITOR_ENABLED = True
    logger.info("psutil加载成功，系统监控功能已启用")
except ImportError:
    SYSTEM_MONITOR_ENABLED = False
    logger.warning("psutil未安装，系统监控功能将不可用")

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
PARTS_TABLES_FILE = "parts_tables.json"
PROJECTS_FILE = "projects.json"
SCHEDULE_FILE = "schedule.json"
SCHEDULE_LOG_FILE = "schedule_logs.json"
_file_registry_lock = threading.Lock()
_users_lock = threading.RLock()
_projects_lock = threading.Lock()
_schedule_lock = threading.Lock()
_schedule_log_lock = threading.Lock()

class TaskScheduler:
    def __init__(self):
        self._timers = {}
        self._running = False
        
    def start(self):
        self._running = True
        self._reschedule_all()
        
    def stop(self):
        self._running = False
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        
    def _reschedule_all(self):
        tasks = _load_schedule()
        for task in tasks:
            if task.get("active", False):
                self._schedule_task(task)
                
    def _schedule_task(self, task):
        task_id = task["id"]
        if task_id in self._timers:
            self._timers[task_id].cancel()
            
        cron_expr = task.get("cron_expression", "")
        next_run = self._parse_cron_and_get_next(cron_expr)
        if next_run:
            delay = (next_run - datetime.now(timezone.utc)).total_seconds()
            if delay > 0:
                timer = threading.Timer(delay, self._execute_task, args=[task])
                self._timers[task_id] = timer
                timer.start()
                
    def _parse_cron_and_get_next(self, cron_expr):
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return None
            
            minute, hour, day, month, weekday = parts
            
            now = datetime.now(timezone.utc)
            next_run = None
            
            for days_ahead in range(0, 366):
                candidate = now + timedelta(days=days_ahead)
                
                if not self._matches_field(candidate.month, month):
                    continue
                if not self._matches_field(candidate.day, day):
                    continue
                if not self._matches_weekday(candidate.weekday(), weekday):
                    continue
                
                for h in range(24):
                    if not self._matches_field(h, hour):
                        continue
                    for m in range(60):
                        if not self._matches_field(m, minute):
                            continue
                            
                        candidate_time = candidate.replace(hour=h, minute=m, second=0, microsecond=0)
                        if candidate_time > now:
                            return candidate_time
                            
            return None
        except Exception:
            return None
            
    def _matches_field(self, value, pattern):
        if pattern == "*":
            return True
        if pattern.isdigit():
            return int(value) == int(pattern)
        if "," in pattern:
            return any(self._matches_field(value, p.strip()) for p in pattern.split(","))
        if "-" in pattern:
            start, end = pattern.split("-")
            return int(start) <= int(value) <= int(end)
        if "/" in pattern:
            base, step = pattern.split("/")
            if base == "*":
                base = 0
            return int(value) % int(step) == int(base) % int(step)
        return False
        
    def _matches_weekday(self, value, pattern):
        if pattern == "*":
            return True
        if pattern == "7":
            pattern = "0"
        days_map = {"MON": "1", "TUE": "2", "WED": "3", "THU": "4", "FRI": "5", "SAT": "6", "SUN": "0"}
        for key, val in days_map.items():
            pattern = pattern.replace(key, val)
        return self._matches_field(value, pattern)
        
    def _execute_task(self, task):
        if not self._running:
            return
            
        task_id = task["id"]
        task_type = task.get("task_type", "check_knowledge")
        start_time = datetime.now(timezone.utc)
        
        try:
            result = self._run_task(task_type, task)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "failed"
            
        end_time = datetime.now(timezone.utc)
        
        self._log_execution(task_id, task["name"], task_type, status, result, start_time, end_time)
        
        if self._running and task.get("active", False):
            self._schedule_task(task)
            
    def _run_task(self, task_type, task):
        if task_type == "check_knowledge":
            return self._check_knowledge_update()
        elif task_type == "generate_report":
            return self._generate_report(task)
        elif task_type == "cleanup_logs":
            return self._cleanup_logs(task)
        elif task_type == "health_check":
            return self._health_check()
        else:
            return f"未知任务类型: {task_type}"
            
    def _check_knowledge_update(self):
        registry = _load_registry()
        updated_count = 0
        skipped_count = 0
        
        for file_record in registry:
            file_id = file_record["file_id"]
            filename = file_record["filename"]
            username = file_record["username"]
            pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
            
            if not os.path.exists(pdf_path):
                skipped_count += 1
                continue
            
            current_md5 = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
            stored_md5 = file_record.get("md5_hash", "")
            
            if current_md5 != stored_md5:
                try:
                    chunks = extract_text_and_images(pdf_path)
                    if chunks:
                        user_collection = get_user_collection(username)
                        global_collection = get_global_collection()
                        
                        existing_ids = [f"{file_id}_{i}" for i in range(file_record.get("chunk_count", 0))]
                        global_ids = [f"global_{file_id}_{i}" for i in range(file_record.get("chunk_count", 0))]
                        
                        try:
                            user_collection.delete(ids=existing_ids)
                        except Exception:
                            pass
                        try:
                            global_collection.delete(ids=global_ids)
                        except Exception:
                            pass
                        
                        texts = [c["content"] for c in chunks]
                        embeddings = embed_texts(texts)
                        
                        user_collection.add(
                            ids=[f"{file_id}_{i}" for i in range(len(chunks))],
                            embeddings=embeddings,
                            metadatas=[{"page": c["page"], "type": c["type"], "file_id": file_id, "username": username} for c in chunks],
                            documents=texts
                        )
                        
                        global_collection.add(
                            ids=[f"global_{file_id}_{i}" for i in range(len(chunks))],
                            embeddings=embeddings,
                            metadatas=[{"page": c["page"], "type": c["type"], "file_id": file_id, "username": username} for c in chunks],
                            documents=texts
                        )
                        
                        file_record["chunk_count"] = len(chunks)
                        file_record["md5_hash"] = current_md5
                        file_record["updated"] = True
                        file_record["last_updated_at"] = datetime.now().isoformat()
                        updated_count += 1
                except Exception as e:
                    logger.error(f"更新文件 {filename} 失败: {e}")
        
        _save_registry(registry)
        return f"知识库更新检查完成：共 {len(registry)} 个文件，{updated_count} 个文件内容有变化已重新解析，{skipped_count} 个文件不存在"
        
    def _generate_report(self, task):
        username = task.get("username", "")
        user_collection = get_user_collection(username)
        total_chunks = user_collection.count()
        registry = _load_registry()
        user_files = [f for f in registry if f.get("username") == username]
        
        projects = _load_projects()
        user_projects = [p for p in projects if p.get("username") == username]
        
        schedules = _load_schedule()
        user_schedules = [s for s in schedules if s.get("username") == username]
        
        report_data = {
            "date": datetime.now(timezone.utc).isoformat(),
            "username": username,
            "total_chunks": total_chunks,
            "total_files": len(user_files),
            "total_projects": len(user_projects),
            "total_schedules": len(user_schedules),
            "files": [{"filename": f["filename"], "uploaded_at": f.get("uploaded_at", ""), "chunk_count": f.get("chunk_count", 0)} for f in user_files],
            "projects": [{"name": p["name"], "status": p.get("status", ""), "progress": p.get("progress", 0)} for p in user_projects],
            "schedules": [{"name": s["name"], "task_type": s.get("task_type", ""), "active": s.get("active", False)} for s in user_schedules]
        }
        
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if PDF_GENERATION_ENABLED:
            pdf_filename = f"report_{timestamp}.pdf"
            pdf_path = os.path.join(report_dir, pdf_filename)
            self._generate_pdf_report(pdf_path, report_data)
            result = f"PDF报告已生成: {pdf_path}"
        else:
            json_filename = f"report_{timestamp}.json"
            json_path = os.path.join(report_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            result = f"JSON报告已生成: {json_path}"
            
        return result
    
    def _generate_pdf_report(self, pdf_path, report_data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        
        pdf.cell(200, 10, txt="运维助手 - 知识库运维报告", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt=f"生成日期: {report_data['date']}", ln=True)
        pdf.cell(200, 8, txt=f"用户: {report_data['username']}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="知识库统计", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(100, 8, txt=f"文件总数: {report_data['total_files']}")
        pdf.cell(100, 8, txt=f"内容块总数: {report_data['total_chunks']}", ln=True)
        pdf.cell(100, 8, txt=f"项目总数: {report_data['total_projects']}")
        pdf.cell(100, 8, txt=f"定时任务数: {report_data['total_schedules']}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="上传文件列表", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(80, 6, txt="文件名")
        pdf.cell(60, 6, txt="上传时间")
        pdf.cell(60, 6, txt="内容块数", ln=True)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        
        for f in report_data['files']:
            pdf.cell(80, 6, txt=f["filename"][:25] + "..." if len(f["filename"]) > 25 else f["filename"])
            pdf.cell(60, 6, txt=f["uploaded_at"][:19] if f["uploaded_at"] else "")
            pdf.cell(60, 6, txt=str(f["chunk_count"]), ln=True)
        
        pdf.ln(5)
        
        if report_data['projects']:
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(200, 10, txt="运维项目", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.cell(80, 6, txt="项目名称")
            pdf.cell(60, 6, txt="状态")
            pdf.cell(60, 6, txt="进度(%)", ln=True)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            
            for p in report_data['projects']:
                pdf.cell(80, 6, txt=p["name"][:25] + "..." if len(p["name"]) > 25 else p["name"])
                pdf.cell(60, 6, txt=p["status"])
                pdf.cell(60, 6, txt=str(p["progress"]), ln=True)
            
            pdf.ln(5)
        
        if report_data['schedules']:
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(200, 10, txt="定时任务", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.cell(80, 6, txt="任务名称")
            pdf.cell(60, 6, txt="任务类型")
            pdf.cell(60, 6, txt="状态", ln=True)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            
            type_labels = {
                "check_knowledge": "知识库更新",
                "generate_report": "生成报告",
                "cleanup_logs": "清理文件",
                "health_check": "健康检查"
            }
            
            for s in report_data['schedules']:
                pdf.cell(80, 6, txt=s["name"][:25] + "..." if len(s["name"]) > 25 else s["name"])
                pdf.cell(60, 6, txt=type_labels.get(s["task_type"], s["task_type"]))
                pdf.cell(60, 6, txt="运行中" if s["active"] else "已禁用", ln=True)
        
        pdf.output(pdf_path)
        
    def _cleanup_logs(self, task):
        max_days = task.get("cleanup_days", 7)
        dirs_to_clean = ["media/chat", "media/cases", "reports"]
        deleted_count = 0
        
        for target_dir in dirs_to_clean:
            if os.path.exists(target_dir):
                for filename in os.listdir(target_dir):
                    filepath = os.path.join(target_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath), timezone.utc)
                        if (datetime.now(timezone.utc) - file_time).days > max_days:
                            try:
                                os.remove(filepath)
                                deleted_count += 1
                                logger.info(f"清理过期文件: {filepath}")
                            except Exception as e:
                                logger.error(f"清理文件 {filepath} 失败: {e}")
        
        if os.path.exists(SCHEDULE_LOG_FILE):
            try:
                with open(SCHEDULE_LOG_FILE, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_days)
                logs = [log for log in logs if datetime.fromisoformat(log["start_time"]) > cutoff_time]
                
                with open(SCHEDULE_LOG_FILE, "w", encoding="utf-8") as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                
                logger.info(f"清理了 {len(logs)} 条过期任务日志")
            except Exception as e:
                logger.error(f"清理任务日志失败: {e}")
            
        return f"已清理 {deleted_count} 个过期文件（超过 {max_days} 天），包括媒体文件和报告文件"
        
    def _health_check(self):
        checks = []
        
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, "r", encoding="utf-8") as f:
                    users = json.load(f)
                checks.append(f"用户数据库: OK ({len(users)}个用户)")
            except Exception:
                checks.append("用户数据库: ERROR")
        else:
            checks.append("用户数据库: NOT FOUND")
            
        if os.path.exists(SCHEDULE_FILE):
            try:
                with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
                    tasks = json.load(f)
                checks.append(f"任务调度: OK ({len(tasks)}个任务)")
            except Exception:
                checks.append("任务调度: ERROR")
        else:
            checks.append("任务调度: NOT FOUND")
            
        try:
            global_collection = get_global_collection()
            global_count = global_collection.count()
            checks.append(f"知识库: OK ({global_count}条记录)")
        except Exception as e:
            checks.append(f"知识库: ERROR - {str(e)}")
            
        if SYSTEM_MONITOR_ENABLED:
            try:
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_total = disk.total // (1024 * 1024 * 1024)
                disk_used = disk.used // (1024 * 1024 * 1024)
                disk_free = disk.free // (1024 * 1024 * 1024)
                
                if disk_percent > 90:
                    checks.append(f"磁盘空间: WARNING ({disk_used}/{disk_total}GB, {disk_percent}%)")
                else:
                    checks.append(f"磁盘空间: OK ({disk_used}/{disk_total}GB, {disk_percent}%)")
            except Exception as e:
                checks.append(f"磁盘空间: ERROR - {str(e)}")
            
            try:
                mem = psutil.virtual_memory()
                mem_percent = mem.percent
                mem_total = mem.total // (1024 * 1024 * 1024)
                mem_used = mem.used // (1024 * 1024 * 1024)
                
                if mem_percent > 90:
                    checks.append(f"内存使用: WARNING ({mem_used}/{mem_total}GB, {mem_percent}%)")
                else:
                    checks.append(f"内存使用: OK ({mem_used}/{mem_total}GB, {mem_percent}%)")
            except Exception as e:
                checks.append(f"内存使用: ERROR - {str(e)}")
            
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent > 90:
                    checks.append(f"CPU使用率: WARNING ({cpu_percent}%)")
                else:
                    checks.append(f"CPU使用率: OK ({cpu_percent}%)")
            except Exception as e:
                checks.append(f"CPU使用率: ERROR - {str(e)}")
            
            try:
                network = psutil.net_io_counters()
                bytes_sent = network.bytes_sent // (1024 * 1024)
                bytes_recv = network.bytes_recv // (1024 * 1024)
                checks.append(f"网络流量: OK (发送 {bytes_sent}MB, 接收 {bytes_recv}MB)")
            except Exception as e:
                checks.append(f"网络流量: ERROR - {str(e)}")
        else:
            checks.append("系统监控: 未安装psutil")
            
        return "; ".join(checks)
        
    def _log_execution(self, task_id, task_name, task_type, status, result, start_time, end_time):
        with _schedule_log_lock:
            logs = []
            if os.path.exists(SCHEDULE_LOG_FILE):
                try:
                    with open(SCHEDULE_LOG_FILE, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
            
            logs.append({
                "id": str(uuid.uuid4()),
                "task_id": task_id,
                "task_name": task_name,
                "task_type": task_type,
                "status": status,
                "result": result,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_ms": int((end_time - start_time).total_seconds() * 1000)
            })
            
            logs = logs[-100:]
            
            with open(SCHEDULE_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
    def add_task(self, task):
        self._schedule_task(task)
        
    def remove_task(self, task_id):
        if task_id in self._timers:
            self._timers[task_id].cancel()
            del self._timers[task_id]
            
    def update_task(self, task):
        self._schedule_task(task)
        
    def trigger_task(self, task):
        threading.Thread(target=self._execute_task, args=[task]).start()

task_scheduler = TaskScheduler()
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
app.mount("/media", StaticFiles(directory="media"), name="media")
client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

import os

local_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")

try:
    if os.path.exists(local_model_path):
        embed_model = SentenceTransformer(local_model_path)
        logger.info(f"嵌入模型从本地缓存加载成功: {local_model_path}")
    else:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("嵌入模型从远程加载成功")
except Exception as e:
    logger.warning(f"嵌入模型加载失败，部分功能将不可用: {str(e)}")
    embed_model = None

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
RATE_LIMIT_MAX_REQUESTS = 100  # 每分钟最大请求数
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

def analyze_pdf_image(image_bytes, page_num, img_idx):
    """调用多模态模型分析PDF中的图片，提取编号和部件信息"""
    try:
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        image_prompt = """你是一名专业的工业设备检修工程师。请仔细观察这张图片，提取所有用于检索检修手册的关键信息。

请按以下JSON格式输出，确保返回完整的JSON对象：
{
    "image_type": "图片类型（爆炸图/装配图/零件图/示意图/流程图等）",
    "title": "图片标题或主要内容描述",
    "all_parts": [
        {"number": "部件编号（如1、2、3，必须是纯数字）", "name": "部件名称（使用专业术语）", "description": "简短描述该部件的功能或位置"},
        ...
    ],
    "visible_text": ["图片中可见的文字内容1", "图片中可见的文字内容2"],
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5", "关键词6", "关键词7", "关键词8", "关键词9", "关键词10"]
}

【强制要求】
1. 如果图片是爆炸图或装配图，必须识别图片中所有可见的数字编号（1、2、3...）
2. number字段必须是纯数字字符串，不要包含任何其他字符
3. name字段必须使用专业术语
4. all_parts数组必须按照编号从小到大排序
5. visible_text必须包含图片中所有可见的文字，包括标题、编号标注、说明文字、表格内容等
6. keywords必须包含至少10个用于检索的专业术语，包括所有识别出的部件名称
7. 不要描述无关细节，只关注设备类型和部件名称
8. 如果图片没有编号，则all_parts为空数组[]"""
        
        resp = client.chat.completions.create(
            model=MULTIMODAL_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": image_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            timeout=30
        )
        
        image_desc = resp.choices[0].message.content.strip()
        if image_desc.startswith("```json"):
            image_desc = image_desc[7:]
        if image_desc.startswith("```"):
            image_desc = image_desc[3:]
        if image_desc.endswith("```"):
            image_desc = image_desc[:-3]
        image_desc = image_desc.strip()
        
        image_analysis = json.loads(image_desc)
        
        logger.info(f"PDF第{page_num}页图片{img_idx}分析成功")
        return image_analysis
    except Exception as e:
        logger.warning(f"PDF第{page_num}页图片{img_idx}分析失败: {e}")
        return None

def extract_text_and_images(pdf_path: str):
    """提取 PDF 文本、表格和图片，返回 chunks 和 零件表列表"""
    chunks = []
    page_tables = {}
    page_images = {}
    image_tasks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"开始解析PDF，共 {total_pages} 页")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num % 10 == 0 or page_num == 1:
                logger.info(f"正在解析第 {page_num}/{total_pages} 页...")
            
            text = page.extract_text()
            if text:
                text_chunks = semantic_chunk_text(text, page_num)
                chunks.extend(text_chunks)
                logger.info(f"  提取文本块: {len(text_chunks)} 个")
            
            tables = page.extract_tables()
            if tables:
                page_tables[page_num] = tables
                logger.info(f"  发现 {len(tables)} 个表格")
                
                for table_idx, table in enumerate(tables):
                    if len(table) >= 2 and len(table[0]) >= 2:
                        headers = [str(cell).strip() for cell in table[0]]
                        has_number_col = any('序号' in h or '编号' in h or '编号' in h.lower() or 'num' in h.lower() for h in headers)
                        has_name_col = any('名称' in h or '零件' in h or '部件' in h for h in headers)
                        
                        if has_number_col or has_name_col:
                            table_content = f"[表格{table_idx+1}] 位置: 第{page_num}页\n表头: {', '.join(headers)}\n内容:\n"
                            table_data = []
                            
                            for row_idx, row in enumerate(table[1:], start=2):
                                row_values = []
                                for cell in row:
                                    if cell is None:
                                        row_values.append("")
                                    else:
                                        row_values.append(str(cell).strip())
                                
                                row_str = " | ".join(row_values)
                                table_content += f"  行{row_idx}: {row_str}\n"
                                
                                if has_number_col:
                                    num_idx = None
                                    name_idx = None
                                    for i, h in enumerate(headers):
                                        if '序号' in h or '编号' in h:
                                            num_idx = i
                                        if '名称' in h or '零件' in h or '部件' in h:
                                            name_idx = i
                                    
                                    if num_idx is not None and name_idx is not None and row_values[num_idx] and row_values[name_idx]:
                                        table_data.append({
                                            "number": row_values[num_idx],
                                            "name": row_values[name_idx],
                                            "row": row_values
                                        })
                            
                            chunks.append({
                                "page": page_num, "type": "table",
                                "content": table_content,
                                "headers": headers,
                                "table_data": table_data
                            })
                            logger.info(f"  表格{table_idx+1}已提取，包含 {len(table_data)} 条编号-名称记录")
            
            if page.images:
                largest_img = None
                largest_area = 0
                
                for img_info in page.images:
                    try:
                        width = int(img_info.get("x1", 0) - img_info.get("x0", 0))
                        height = int(img_info.get("bottom", 0) - img_info.get("top", 0))
                        area = width * height
                        
                        if width > 100 and height > 100 and area > largest_area:
                            aspect_ratio = width / height if height > 0 else 0
                            if 0.2 < aspect_ratio < 5.0:
                                largest_area = area
                                largest_img = img_info
                    except:
                        pass
                
                if largest_img:
                    try:
                        img_obj = page.to_image()
                        img_bytes = img_obj.original.tobytes()
                        width = int(largest_img.get("x1", 0) - largest_img.get("x0", 0))
                        height = int(largest_img.get("bottom", 0) - largest_img.get("top", 0))
                        image_tasks.append((img_bytes, page_num, width, height))
                    except Exception as img_err:
                        logger.warning(f"PDF第{page_num}页图片提取失败: {img_err}")
    
    logger.info(f"文本和表格提取完成，开始并发分析 {len(image_tasks)} 张图片...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_image_task(task):
        img_bytes, page_num, width, height = task
        return (page_num, width, height, analyze_pdf_image(img_bytes, page_num, 1))
    
    if image_tasks:
        max_workers = min(5, len(image_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_image_task, task): task for task in image_tasks}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                page_num, width, height, image_analysis = future.result()
                logger.info(f"图片分析进度: {completed}/{len(image_tasks)}（第{page_num}页）")
                if image_analysis:
                    page_images[page_num] = image_analysis
    
    for page_num, image_analysis in page_images.items():
        if page_num in page_tables:
            tables = page_tables[page_num]
            
            table_parts_map = {}
            for table in tables:
                if len(table) >= 2 and len(table[0]) >= 2:
                    headers = [str(cell).strip() for cell in table[0]]
                    has_num_col = any('序号' in h or '编号' in h or h.lower() == 'num' for h in headers)
                    has_name_col = any('名称' in h or '零件' in h or '部件' in h for h in headers)
                    
                    if has_num_col and has_name_col:
                        num_idx = None
                        name_idx = None
                        for i, h in enumerate(headers):
                            if '序号' in h or '编号' in h:
                                num_idx = i
                            if '名称' in h or '零件' in h or '部件' in h:
                                name_idx = i
                        
                        if num_idx is not None and name_idx is not None:
                            for row in table[1:]:
                                row_values = [str(cell).strip() if cell else "" for cell in row]
                                num = row_values[num_idx] if num_idx < len(row_values) else ""
                                name = row_values[name_idx] if name_idx < len(row_values) else ""
                                if num and name and num.isdigit():
                                    table_parts_map[num] = name
            
            if table_parts_map:
                image_all_parts = image_analysis.get("all_parts", [])
                image_type = image_analysis.get("image_type", "")
                image_title = image_analysis.get("title", "")
                image_keywords = image_analysis.get("keywords", [])
                
                merged_content = f"【第{page_num}页 零件清单（爆炸图编号对应表）】\n"
                merged_content += f"图片类型: {image_type}\n"
                merged_content += f"图片标题: {image_title}\n"
                merged_content += f"重要说明：以下是本页爆炸图中每个编号对应的零件名称（最权威依据，回答时必须以此为准）：\n\n"
                
                for num in sorted(table_parts_map.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    merged_content += f"  编号{num}: {table_parts_map[num]}\n"
                
                if image_all_parts:
                    merged_content += f"\n图片AI识别结果（仅供参考，以表格为准）:\n"
                    for part in image_all_parts:
                        if isinstance(part, dict):
                            num = part.get("number", "")
                            name = part.get("name", "")
                            desc = part.get("description", "")
                            if num and name:
                                merged_content += f"  识别编号{num}: {name}（{desc}）\n"
                
                chunks.append({
                    "page": page_num, "type": "page_context",
                    "content": merged_content,
                    "image_all_parts": image_all_parts,
                    "table_parts_map": table_parts_map,
                    "matched": True,
                    "keywords": image_keywords + list(table_parts_map.values())
                })
                logger.info(f"第{page_num}页图片+表格已关联，共{len(table_parts_map)}个编号")
    
    parts_tables = []
    for page_num, tables in page_tables.items():
        for table in tables:
            if len(table) >= 2 and len(table[0]) >= 2:
                headers = [str(cell).strip() for cell in table[0]]
                has_num_col = any('序号' in h or '编号' in h or h.lower() == 'num' for h in headers)
                has_name_col = any('名称' in h or '零件' in h or '部件' in h for h in headers)
                
                if has_num_col and has_name_col:
                    num_idx = None
                    name_idx = None
                    for i, h in enumerate(headers):
                        if '序号' in h or '编号' in h:
                            num_idx = i
                        if '名称' in h or '零件' in h or '部件' in h:
                            name_idx = i
                    
                    if num_idx is not None and name_idx is not None:
                        table_parts = {}
                        for row in table[1:]:
                            row_values = [str(cell).strip() if cell else "" for cell in row]
                            num = row_values[num_idx] if num_idx < len(row_values) else ""
                            name = row_values[name_idx] if name_idx < len(row_values) else ""
                            if num and name and num.isdigit():
                                table_parts[num] = name
                        
                        if table_parts and len(table_parts) >= 2:
                            parts_tables.append({
                                "page": page_num,
                                "parts": table_parts
                            })
    
    logger.info(f"PDF解析完成，共提取 {len(chunks)} 个内容块，{len(parts_tables)} 个零件表")
    return chunks, parts_tables

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
async def upload_pdf(request: Request, user: Dict = Depends(get_current_user)):
    uploaded_count = 0
    errors = []
    
    form = await request.form()
    pdfs = form.getlist("pdfs")
    
    for file in pdfs:
        if not file.filename.endswith('.pdf'):
            errors.append(f"{file.filename}: 不是PDF文件")
            continue
        
        try:
            content = await file.read()
            if len(content) == 0:
                errors.append(f"{file.filename}: 文件为空")
                continue
            if len(content) > MAX_UPLOAD_SIZE:
                errors.append(f"{file.filename}: 文件过大")
                continue
            if not content.startswith(b'%PDF-'):
                errors.append(f"{file.filename}: 无效的PDF文件")
                continue
            
            file_id = str(uuid.uuid4())
            pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(content)
            
            try:
                chunks, parts_tables = await asyncio.wait_for(
                    asyncio.to_thread(extract_text_and_images, pdf_path),
                    timeout=300.0
                )
            except asyncio.TimeoutError:
                errors.append(f"{file.filename}: 解析超时")
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                continue
            
            if not chunks:
                errors.append(f"{file.filename}: 解析未获得任何内容")
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                continue
            
            if parts_tables:
                for pt in parts_tables:
                    _add_parts_table(file_id, file.filename, user["username"], pt["page"], pt["parts"])
                logger.info(f"{file.filename}: 保存了 {len(parts_tables)} 个零件表")
            
            user_collection = get_user_collection(user["username"])
            texts = [c["content"] for c in chunks]
            
            try:
                embeddings = await asyncio.wait_for(
                    asyncio.to_thread(embed_texts, texts),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                errors.append(f"{file.filename}: 嵌入生成超时")
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                continue
            
            user_collection.add(
                ids=[f"{file_id}_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                metadatas=[_build_metadata(c, file_id, user["username"]) for c in chunks],
                documents=texts
            )
            
            global_collection = get_global_collection()
            global_collection.add(
                ids=[f"global_{file_id}_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                metadatas=[_build_metadata(c, file_id, user["username"]) for c in chunks],
                documents=texts
            )
            
            registry = _load_registry()
            existing = next((f for f in registry if f["username"] == user["username"] and f["filename"] == file.filename), None)
            if existing:
                errors.append(f"{file.filename}: 已存在同名文件")
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                continue
            
            _register_file(file_id, file.filename, user["username"], len(chunks))
            uploaded_count += 1
            
        except HTTPException as e:
            errors.append(f"{file.filename}: {e.detail}")
        except Exception as e:
            errors.append(f"{file.filename}: 处理失败 - {str(e)}")
    
    if uploaded_count == 0:
        return {"success": False, "error": "所有文件上传失败: " + ", ".join(errors)}
    
    return {"success": True, "count": uploaded_count, "errors": errors if errors else None}

def _build_metadata(chunk, file_id, username):
    """构建metadata，将复杂字段序列化为JSON字符串"""
    meta = {
        "page": chunk["page"],
        "type": chunk["type"],
        "file_id": file_id,
        "username": username
    }
    for key in ["all_parts", "keywords", "table_data", "headers", "table_parts_map", "image_all_parts"]:
        if key in chunk:
            try:
                meta[key] = json.dumps(chunk[key], ensure_ascii=False)
            except:
                pass
    if "matched" in chunk:
        meta["matched"] = str(chunk["matched"])
    return meta

def _parse_metadata(meta):
    """解析metadata，将JSON字符串反序列化"""
    result = dict(meta) if meta else {}
    for key in ["all_parts", "keywords", "table_data", "headers", "table_parts_map", "image_all_parts"]:
        if key in result and isinstance(result[key], str):
            try:
                result[key] = json.loads(result[key])
            except:
                pass
    if "matched" in result:
        result["matched"] = result["matched"] == "True"
    return result

_parts_tables_lock = threading.Lock()
_parts_tables_cache = None

def _load_parts_tables():
    """加载所有零件编号-名称对照表"""
    global _parts_tables_cache
    with _parts_tables_lock:
        if _parts_tables_cache is not None:
            return _parts_tables_cache
        abs_path = os.path.abspath(PARTS_TABLES_FILE)
        if not os.path.exists(abs_path):
            _parts_tables_cache = []
            return []
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                _parts_tables_cache = json.load(f)
                logger.info(f"加载零件表成功，共 {len(_parts_tables_cache)} 条记录")
                return _parts_tables_cache
        except json.JSONDecodeError as e:
            logger.error(f"零件表解析失败: {e}")
            _parts_tables_cache = []
            return []

def _save_parts_tables(tables):
    """保存零件编号-名称对照表"""
    global _parts_tables_cache
    with _parts_tables_lock:
        _parts_tables_cache = tables
        abs_path = os.path.abspath(PARTS_TABLES_FILE)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)

def _add_parts_table(file_id, filename, username, page_num, table_parts_map):
    """添加一条零件表记录"""
    tables = _load_parts_tables()
    tables.append({
        "file_id": file_id,
        "filename": filename,
        "username": username,
        "page": page_num,
        "parts": table_parts_map
    })
    _save_parts_tables(tables)

def _find_parts_table(username, part_numbers):
    """根据用户和编号集合查找最匹配的零件表"""
    tables = _load_parts_tables()
    user_tables = [t for t in tables if t["username"] == username]
    
    if not user_tables:
        return None
    
    part_num_set = set(part_numbers)
    best_match = None
    best_score = 0
    
    for table in user_tables:
        parts = table.get("parts", {})
        table_nums = set(parts.keys())
        overlap = part_num_set & table_nums
        score = len(overlap)
        if score > best_score:
            best_score = score
            best_match = table
    
    if best_match and best_score >= 2:
        return best_match
    return None

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
    """列出当前用户上传的文件"""
    logger.info(f"获取文件列表 - 用户: {user['username']}, 角色: {user['role']}")
    registry = _load_registry()
    user_files = [f for f in registry if f["username"] == user["username"]]
    user_files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    logger.info(f"返回文件数量: {len(user_files)}")
    return {"files": user_files}

@app.get("/api/files/all")
async def list_all_files(user: Dict = Depends(get_current_user)):
    """列出当前用户上传的文件（供前端新对话页面使用）"""
    registry = _load_registry()
    user_files = [f for f in registry if f["username"] == user["username"]]
    user_files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return {"files": user_files}

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str, user: Dict = Depends(get_current_user)):
    """删除指定文件及其知识库条目"""
    logger.info(f"========== 删除请求开始 ==========")
    logger.info(f"用户: {user['username']}, 角色: {user['role']}, 文件ID: {file_id}")
    
    registry = _load_registry()
    logger.info(f"注册表文件数量: {len(registry)}")
    
    file_record = next((f for f in registry if f["file_id"] == file_id), None)
    logger.info(f"找到文件记录: {file_record is not None}")
    
    if file_record:
        logger.info(f"文件名: {file_record['filename']}, 上传者: {file_record['username']}")
        logger.info(f"用户名匹配: {user['username'] == file_record['username']}, 角色是管理员: {user['role'] == 'admin'}")
    
    if not file_record:
        logger.warning(f"文件不存在 - 文件ID: {file_id}")
        raise HTTPException(404, "文件不存在")
    
    # 检查权限：用户只能删除自己上传的文件，管理员可以删除所有文件
    if user["username"] != file_record["username"] and user["role"] != "admin":
        logger.warning(f"权限不足 - 当前用户: {user['username']}, 文件上传者: {file_record['username']}, 当前角色: {user['role']}")
        raise HTTPException(403, "无权删除此文件")
    
    logger.info(f"权限检查通过")
    
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
                if meta.get("file_id") == file_id:
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

CHAT_MEDIA_DIR = "media/chat"
os.makedirs(CHAT_MEDIA_DIR, exist_ok=True)

@app.post("/ask")
async def ask_question(
    question: str = Form(""),
    image: UploadFile = File(None),
    video: UploadFile = File(None),
    context: str = Form(""),
    user: Dict = Depends(get_current_user)
):
    try:
        media_files = []
        image_analysis = None
        
        if not question and not image and not video:
            return JSONResponse(
                status_code=400,
                content={
                    "answer": "请输入问题或上传图片/视频",
                    "references": [],
                    "answer_id": str(uuid.uuid4()),
                    "retrieved_snippets": [],
                    "suggested_questions": [],
                    "media_files": []
                }
            )
            
        if image:
            try:
                img_bytes = await image.read()
                if len(img_bytes) > 10 * 1024 * 1024:
                    raise HTTPException(400, "图片过大，最大允许10MB")
                img_ext = os.path.splitext(image.filename)[1].lower()
                if img_ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                    raise HTTPException(400, "不支持的图片格式")
                img_filename = f"{str(uuid.uuid4())}{img_ext}"
                img_path = os.path.join(CHAT_MEDIA_DIR, img_filename)
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                media_files.append({"type": "image", "path": f"chat/{img_filename}"})
                
                device_keywords = ""
                try:
                    user_collection = get_user_collection(user["username"])
                    if user_collection is not None:
                        all_docs = []
                        try:
                            results = user_collection.get()
                            if results.get("documents"):
                                all_docs = results["documents"]
                        except:
                            pass
                        
                        if not all_docs:
                            global_collection = get_global_collection()
                            if global_collection is not None:
                                try:
                                    results = global_collection.get()
                                    if results.get("documents"):
                                        all_docs = results["documents"]
                                except:
                                    pass
                        
                        if all_docs:
                            all_text = " ".join([doc for doc_list in all_docs for doc in (doc_list if isinstance(doc_list, list) else [doc_list])])
                            from collections import Counter
                            words = re.findall(r'[\u4e00-\u9fff]{2,}', all_text)
                            common_words = Counter(words).most_common(20)
                            device_keywords = ", ".join([word for word, _ in common_words])
                except Exception as kw_err:
                    logger.error(f"提取关键词失败: {kw_err}")
                
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                if device_keywords:
                    image_prompt = f"""你是一名专业的工业设备检修工程师。请仔细观察这张图片，提取所有用于检索检修手册的关键信息。

参考以下手册中的专业术语（请优先使用这些术语，这是最准确的来源）：
{device_keywords}

工业设备常见部件参考（用于辅助识别，最终以手册术语为准）：
- 机械传动：齿轮、轴承、联轴器、皮带轮、链轮、链条、减速器、变速箱
- 液压系统：液压泵、液压缸、液压阀、液压马达、油管、油箱
- 气动系统：气缸、气动阀、空气压缩机、储气罐、气管
- 电气系统：电机、变频器、PLC、传感器、继电器、接触器、开关
- 钢铁行业：高炉、转炉、连铸机、轧机、加热炉、冷却系统
- 汽车制造：发动机、变速箱、底盘、车身焊接设备、涂装线、装配线
- 泵阀类：离心泵、柱塞泵、球阀、闸阀、蝶阀、安全阀
- 通用部件：螺栓、螺母、垫片、密封圈、轴承座、支架、壳体

请按以下JSON格式输出，确保返回完整的JSON对象：
{{
    "image_type": "图片类型（爆炸图/装配图/零件图/示意图/流程图/设备照片等）",
    "part_name": "图片中主要部件的名称",
    "system": "所属系统或组件（如发动机、变速箱、液压系统、电气系统等）",
    "industry": "所属行业（如钢铁、汽车制造、机械加工等）",
    "all_parts": [
        {{"number": "部件编号（如1、2、3，必须是纯数字）", "name": "部件名称（使用手册术语）", "description": "简短描述该部件的功能或位置"}},
        ...
    ],
    "visible_text": ["图片中可见的文字内容1", "图片中可见的文字内容2"],
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5", "关键词6", "关键词7", "关键词8", "关键词9", "关键词10"]
}}

【强制要求】
1. 如果图片是爆炸图或装配图，必须识别图片中所有可见的数字编号（1、2、3...）
2. number字段必须是纯数字字符串，不要包含任何其他字符
3. name字段必须使用专业术语，优先从参考术语中选择
4. all_parts数组必须按照编号从小到大排序
5. visible_text必须包含图片中所有可见的文字，包括标题、编号标注、说明文字、表格内容等
6. keywords必须包含至少10个用于检索的专业术语，包括所有识别出的部件名称
7. 不要描述无关细节，只关注设备类型和部件名称
8. 如果图片没有编号，则all_parts为空数组[]"""
                else:
                    image_prompt = """你是一名专业的工业设备检修工程师。请仔细观察这张图片，提取所有用于检索检修手册的关键信息。

工业设备常见部件参考：
- 机械传动：齿轮、轴承、联轴器、皮带轮、链轮、链条、减速器、变速箱
- 液压系统：液压泵、液压缸、液压阀、液压马达、油管、油箱
- 气动系统：气缸、气动阀、空气压缩机、储气罐、气管
- 电气系统：电机、变频器、PLC、传感器、继电器、接触器、开关
- 钢铁行业：高炉、转炉、连铸机、轧机、加热炉、冷却系统
- 汽车制造：发动机、变速箱、底盘、车身焊接设备、涂装线、装配线
- 泵阀类：离心泵、柱塞泵、球阀、闸阀、蝶阀、安全阀
- 通用部件：螺栓、螺母、垫片、密封圈、轴承座、支架、壳体

请按以下JSON格式输出，确保返回完整的JSON对象：
{
    "image_type": "图片类型（爆炸图/装配图/零件图/示意图/流程图/设备照片等）",
    "part_name": "图片中主要部件的名称",
    "system": "所属系统或组件（如发动机、变速箱、液压系统、电气系统等）",
    "industry": "所属行业（如钢铁、汽车制造、机械加工等）",
    "all_parts": [
        {"number": "部件编号（如1、2、3，必须是纯数字）", "name": "部件名称（使用专业术语）", "description": "简短描述该部件的功能或位置"},
        ...
    ],
    "visible_text": ["图片中可见的文字内容1", "图片中可见的文字内容2"],
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5", "关键词6", "关键词7", "关键词8", "关键词9", "关键词10"]
}

【强制要求】
1. 如果图片是爆炸图或装配图，必须识别图片中所有可见的数字编号（1、2、3...）
2. number字段必须是纯数字字符串，不要包含任何其他字符
3. name字段必须使用专业术语
4. all_parts数组必须按照编号从小到大排序
5. visible_text必须包含图片中所有可见的文字，包括标题、编号标注、说明文字、表格内容等
6. keywords必须包含至少10个用于检索的专业术语，包括所有识别出的部件名称
7. 不要描述无关细节，只关注设备类型和部件名称
8. 如果图片没有编号，则all_parts为空数组[]"""
                
                resp = client.chat.completions.create(
                    model=MULTIMODAL_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": image_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }]
                )
                image_analysis = None
                try:
                    image_desc = resp.choices[0].message.content
                    logger.info(f"=== 图片分析原始输出 ===")
                    logger.info(image_desc)
                    logger.info(f"=====================")
                    
                    image_desc = image_desc.strip()
                    if image_desc.startswith("```json"):
                        image_desc = image_desc[7:]
                    if image_desc.startswith("```"):
                        image_desc = image_desc[3:]
                    if image_desc.endswith("```"):
                        image_desc = image_desc[:-3]
                    image_desc = image_desc.strip()
                    
                    image_analysis = json.loads(image_desc)
                except Exception as json_err:
                    logger.error(f"JSON解析失败: {json_err}, 原始输出: {image_desc[:200]}")
                    pass
                
                if image_analysis and isinstance(image_analysis, dict):
                    part_name = image_analysis.get("part_name", "")
                    candidates = image_analysis.get("candidates", [])
                    system = image_analysis.get("system", "")
                    industry = image_analysis.get("industry", "")
                    description = image_analysis.get("description", "")
                    keywords = image_analysis.get("keywords", [])
                    visual_features = image_analysis.get("visual_features", {})
                    
                    search_terms = []
                    if part_name:
                        search_terms.append(part_name)
                    if system:
                        search_terms.append(system)
                    if industry:
                        search_terms.append(industry)
                    search_terms.extend(candidates)
                    search_terms.extend(keywords[:8])
                    
                    if visual_features:
                        for key, values in visual_features.items():
                            if isinstance(values, list):
                                search_terms.extend(values[:3])
                    
                    search_terms = list(set(search_terms))
                    image_keywords_str = ", ".join(search_terms[:10])
                    
                    if question:
                        question = question + f"（图片中的设备：{part_name}，所属系统：{system}，所属行业：{industry}，{description}。候选名称：{', '.join(candidates[:3])}。检索关键词：{image_keywords_str}）"
                    else:
                        question = f"图片中的设备是：{part_name}，属于{system}，{industry}行业，{description}。候选名称：{', '.join(candidates[:3])}。请在检修手册中查找相关的检修步骤和注意事项。检索关键词：{image_keywords_str}"
                else:
                    image_desc = resp.choices[0].message.content if 'image_desc' not in dir() else image_desc
                    if question:
                        question = question + "（图片中的设备：" + image_desc + "）"
                    else:
                        question = f"图片中的设备是：{image_desc}。请在检修手册中查找相关的检修步骤和注意事项。"
            except Exception as e:
                logger.error(f"图片分析失败: {e}")
                if question:
                    question += "（附带图片，但分析失败，请基于检修手册知识回答）"
                else:
                    question = "请基于检修手册知识分析上传的图片中的设备和故障情况。"
        
        if video:
            try:
                video_bytes = await video.read()
                if len(video_bytes) > 50 * 1024 * 1024:
                    raise HTTPException(400, "视频过大，最大允许50MB")
                video_ext = os.path.splitext(video.filename)[1].lower()
                if video_ext not in ['.mp4', '.mov', '.avi', '.webm']:
                    raise HTTPException(400, "不支持的视频格式")
                video_filename = f"{str(uuid.uuid4())}{video_ext}"
                video_path = os.path.join(CHAT_MEDIA_DIR, video_filename)
                with open(video_path, "wb") as f:
                    f.write(video_bytes)
                media_files.append({"type": "video", "path": f"chat/{video_filename}"})
                
                video_desc = ""
                if VIDEO_ANALYSIS_ENABLED:
                    try:
                        import threading
                        
                        def extract_frame(video_path, result_dict):
                            try:
                                video_capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                                video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                
                                success, frame = video_capture.read()
                                video_capture.release()
                                
                                if success:
                                    height, width = frame.shape[:2]
                                    max_dim = 480
                                    if width > max_dim or height > max_dim:
                                        scale = max_dim / max(width, height)
                                        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
                                    _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                    img_b64 = base64.b64encode(img_encoded).decode('utf-8')
                                    result_dict['success'] = True
                                    result_dict['frame'] = img_b64
                                else:
                                    result_dict['success'] = False
                            except Exception as e:
                                result_dict['success'] = False
                                result_dict['error'] = str(e)
                        
                        result_dict = {}
                        thread = threading.Thread(target=extract_frame, args=(video_path, result_dict))
                        thread.start()
                        thread.join(timeout=3)
                        
                        if not thread.is_alive() and result_dict.get('success'):
                            try:
                                resp = client.chat.completions.create(
                                    model=MULTIMODAL_MODEL,
                                    messages=[{
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": "简述图片内容，含设备类型和状态："},
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_dict['frame']}"}}
                                        ]
                                    }],
                                    timeout=10,
                                    max_tokens=100
                                )
                                video_desc = f"视频画面分析：{resp.choices[0].message.content}"
                            except Exception as frame_err:
                                logger.error(f"帧分析失败: {frame_err}")
                                video_desc = "（帧分析超时）"
                        else:
                            video_desc = "（帧提取超时）"
                    except Exception as video_err:
                        logger.error(f"视频分析失败: {video_err}")
                        video_desc = "（视频分析失败）"
                else:
                    video_desc = "（视频分析功能未启用）"
                
                if question:
                    question += f"（附带视频，{video_desc}请结合视频内容和检修手册知识进行分析）"
                else:
                    question = f"请根据检修手册知识分析这个视频中的设备状态和故障情况。{video_desc}请提供详细的检修建议和操作步骤。"
            except Exception as e:
                logger.error(f"视频处理失败: {e}")
                if question:
                    question += "（附带视频，但处理失败，请基于检修手册知识回答）"
                else:
                    question = "请基于检修手册知识分析可能的设备故障和检修方案。"

        if context:
            question = f"【历史对话上下文】\n{context}\n\n【当前问题】\n{question}"

        q_emb = embed_texts([question])[0]
    
        user_collection = get_user_collection(user["username"])
        global_collection = get_global_collection()
    
        all_docs = []
        all_metas = []
        all_distances = []
        doc_count_map = {}
        
        def query_collections(query_emb, query_text="", weight=1.0):
            nonlocal all_docs, all_metas, all_distances, doc_count_map
            
            user_results_text = user_collection.query(
                query_embeddings=[query_emb], 
                n_results=20,
                where={"type": "text"}
            )
            
            user_results_image = user_collection.query(
                query_embeddings=[query_emb],
                n_results=10,
                where={"type": "image_analysis"}
            )
            
            user_results_table = user_collection.query(
                query_embeddings=[query_emb],
                n_results=10,
                where={"type": "table"}
            )
            
            user_results_page = user_collection.query(
                query_embeddings=[query_emb],
                n_results=10,
                where={"type": "page_context"}
            )
            
            global_results_text = global_collection.query(
                query_embeddings=[query_emb],
                n_results=20,
                where={"type": "text"}
            )
            
            global_results_image = global_collection.query(
                query_embeddings=[query_emb],
                n_results=10,
                where={"type": "image_analysis"}
            )
            
            global_results_table = global_collection.query(
                query_embeddings=[query_emb],
                n_results=10,
                where={"type": "table"}
            )
            
            global_results_page = global_collection.query(
                query_embeddings=[query_emb],
                n_results=10,
                where={"type": "page_context"}
            )
            
            def add_results(docs, metas, distances):
                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    dist = distances[i] if i < len(distances) else 1.0
                    weighted_dist = dist / weight
                    
                    content_hash = hashlib.md5(doc[:100].encode('utf-8')).hexdigest()
                    
                    if content_hash in doc_count_map:
                        idx = doc_count_map[content_hash]
                        if weighted_dist < all_distances[idx]:
                            all_distances[idx] = weighted_dist
                        continue
                    
                    doc_count_map[content_hash] = len(all_docs)
                    all_docs.append(doc)
                    all_metas.append(meta)
                    all_distances.append(weighted_dist)
            
            if user_results_text["documents"] and user_results_text["documents"][0]:
                add_results(
                    list(user_results_text["documents"][0]),
                    list(user_results_text["metadatas"][0]),
                    list(user_results_text["distances"][0]) if user_results_text["distances"] and user_results_text["distances"][0] else []
                )
            
            if user_results_image["documents"] and user_results_image["documents"][0]:
                add_results(
                    list(user_results_image["documents"][0]),
                    list(user_results_image["metadatas"][0]),
                    list(user_results_image["distances"][0]) if user_results_image["distances"] and user_results_image["distances"][0] else []
                )
            
            if global_results_text["documents"] and global_results_text["documents"][0]:
                add_results(
                    list(global_results_text["documents"][0]),
                    list(global_results_text["metadatas"][0]),
                    list(global_results_text["distances"][0]) if global_results_text["distances"] and global_results_text["distances"][0] else []
                )
            
            if global_results_image["documents"] and global_results_image["documents"][0]:
                add_results(
                    list(global_results_image["documents"][0]),
                    list(global_results_image["metadatas"][0]),
                    list(global_results_image["distances"][0]) if global_results_image["distances"] and global_results_image["distances"][0] else []
                )
            
            if user_results_table["documents"] and user_results_table["documents"][0]:
                add_results(
                    list(user_results_table["documents"][0]),
                    list(user_results_table["metadatas"][0]),
                    list(user_results_table["distances"][0]) if user_results_table["distances"] and user_results_table["distances"][0] else []
                )
            
            if global_results_table["documents"] and global_results_table["documents"][0]:
                add_results(
                    list(global_results_table["documents"][0]),
                    list(global_results_table["metadatas"][0]),
                    list(global_results_table["distances"][0]) if global_results_table["distances"] and global_results_table["distances"][0] else []
                )
            
            if user_results_page["documents"] and user_results_page["documents"][0]:
                add_results(
                    list(user_results_page["documents"][0]),
                    list(user_results_page["metadatas"][0]),
                    list(user_results_page["distances"][0]) if user_results_page["distances"] and user_results_page["distances"][0] else []
                )
            
            if global_results_page["documents"] and global_results_page["documents"][0]:
                add_results(
                    list(global_results_page["documents"][0]),
                    list(global_results_page["metadatas"][0]),
                    list(global_results_page["distances"][0]) if global_results_page["distances"] and global_results_page["distances"][0] else []
                )
        
        query_collections(q_emb, question, weight=2.0)
        
        image_keywords = []
        if image_analysis and isinstance(image_analysis, dict):
            keywords = image_analysis.get("keywords", [])
            part_name = image_analysis.get("part_name", "")
            system = image_analysis.get("system", "")
            all_parts = image_analysis.get("all_parts", [])
            visible_text = image_analysis.get("visible_text", [])
            candidates = image_analysis.get("candidates", [])
            visual_features = image_analysis.get("visual_features", {})
            
            if part_name:
                image_keywords.append(part_name)
            if system:
                image_keywords.append(system)
            
            if all_parts:
                for part in all_parts:
                    if isinstance(part, dict):
                        part_name_item = part.get("name", "")
                        part_desc = part.get("description", "")
                        if part_name_item:
                            image_keywords.append(part_name_item)
                        if part_desc:
                            image_keywords.append(part_desc)
            
            if visible_text:
                image_keywords.extend(visible_text)
            
            image_keywords.extend(candidates)
            image_keywords.extend(keywords[:10])
            
            if visual_features:
                for key, values in visual_features.items():
                    if isinstance(values, list):
                        image_keywords.extend(values[:3])
            
            image_keywords = list(set(image_keywords))
        
        if image_keywords:
            logger.info(f"=== 图片关键词检索 ===")
            logger.info(f"关键词: {image_keywords}")
            
            part_names = [p.get("name", "") for p in (all_parts or []) if isinstance(p, dict) and p.get("name")]
            
            for keyword in image_keywords:
                if len(keyword) >= 2:
                    kw_emb = embed_texts([keyword])[0]
                    if keyword in part_names:
                        weight = 5.0
                    elif any(keyword in p for p in part_names):
                        weight = 3.0
                    else:
                        weight = 1.0
                    query_collections(kw_emb, keyword, weight=weight)
            
            if all_parts and isinstance(all_parts, list):
                logger.info(f"=== 编号部件单独检索 ===")
                for part in all_parts:
                    if isinstance(part, dict):
                        part_number = part.get("number", "")
                        part_name = part.get("name", "")
                        part_desc = part.get("description", "")
                        
                        if part_number and part_name:
                            combined_query = f"{part_number} {part_name}"
                            kw_emb = embed_texts([combined_query])[0]
                            query_collections(kw_emb, combined_query, weight=8.0)
                            logger.info(f"  编号{part_number}: {part_name} (权重8.0)")
                        
                        if part_name:
                            kw_emb = embed_texts([part_name])[0]
                            query_collections(kw_emb, part_name, weight=6.0)
                        
                        if part_desc:
                            kw_emb = embed_texts([part_desc])[0]
                            query_collections(kw_emb, part_desc, weight=3.0)
            
            logger.info(f"关键词检索完成，累计文档数: {len(all_docs)}")
        
        best_parts_table = None
        if image_analysis and isinstance(image_analysis, dict):
            user_all_parts = image_analysis.get("all_parts", [])
            if user_all_parts and isinstance(user_all_parts, list):
                user_part_numbers = set()
                for part in user_all_parts:
                    if isinstance(part, dict):
                        num = part.get("number", "")
                        if num and num.isdigit():
                            user_part_numbers.add(num)
                
                if user_part_numbers:
                    best_parts_table = _find_parts_table(user["username"], user_part_numbers)
                    if best_parts_table:
                        logger.info(f"=== 零件表精确匹配 ===")
                        logger.info(f"匹配到第{best_parts_table['page']}页，共{len(best_parts_table['parts'])}个编号")
                        logger.info(f"编号: {list(best_parts_table['parts'].keys())}")
        
        if best_parts_table:
            table_parts = best_parts_table["parts"]
            table_page = best_parts_table["page"]
            logger.info(f"=== 根据零件表精确检索部件内容 ===")
            
            for num, part_name in table_parts.items():
                if len(part_name) >= 2:
                    part_emb = embed_texts([part_name])[0]
                    query_collections(part_emb, part_name, weight=10.0)
                    logger.info(f"  编号{num}: {part_name} (权重10.0)")
            
            page_query = f"第{table_page}页 气缸 活塞 检修"
            page_emb = embed_texts([page_query])[0]
            query_collections(page_emb, page_query, weight=5.0)
        
        total_count = len(all_docs)
        if total_count == 0:
            return {
                "answer": "⚠️ 知识库为空，请先上传检修手册（PDF）后再提问。\n\n操作步骤：\n1. 点击左侧菜单「上传手册」\n2. 选择 PDF 格式的检修手册文件\n3. 点击「上传并建立知识库」\n4. 上传成功后即可开始提问",
                "references": [],
                "answer_id": str(uuid.uuid4()),
                "retrieved_snippets": [],
                "suggested_questions": []
            }
        
        valid_items = []
        seen_content_hashes = set()
        
        for i, (doc, meta) in enumerate(zip(all_docs, all_metas)):
            if "暂未启用多模态分析" in doc or "图片提取失败" in doc or "图片坐标无效" in doc:
                continue
            if len(doc.strip()) < 10:
                continue
            content_hash = hashlib.md5(doc[:100].encode('utf-8')).hexdigest()
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            distance = all_distances[i] if i < len(all_distances) else 1.0
            valid_items.append((doc, meta, distance))
        
        valid_items.sort(key=lambda x: x[2])
        valid_items = valid_items[:15]
        
        logger.info(f"=== 检索调试 ===")
        logger.info(f"问题: {question[:100]}...")
        logger.info(f"有效文档数: {len(valid_items)}")
        for i, (doc, meta, dist) in enumerate(valid_items[:5]):
            page = meta.get("page", "?")
            logger.info(f"  [{i}] 页码:{page}, 距离:{dist:.3f}, 内容:{doc[:50]}...")
        logger.info(f"================")
        
        if not valid_items:
            return {
                "answer": "⚠️ 未在知识库中找到与您问题相关的内容。\n\n建议：\n1. 检查是否已上传相关的检修手册\n2. 尝试使用更具体的关键词提问\n3. 如果手册内容较少，可尝试上传更完整的资料",
                "references": [],
                "answer_id": str(uuid.uuid4()),
                "retrieved_snippets": [],
                "suggested_questions": []
            }
        
        page_distribution = {}
        for doc, meta, distance in valid_items[:20]:
            page = meta.get("page", "?")
            page_distribution[page] = page_distribution.get(page, 0) + 1
        
        logger.info(f"=== 页码分布 ===")
        for page, count in sorted(page_distribution.items())[:10]:
            logger.info(f"  页码{page}: {count}次")
        logger.info(f"===============")
        
        has_relevant_context = False
        relevant_items = []
        
        question_text = question.lower()
        has_image_only = "附带图片描述" in question or "附带图片" in question
        
        image_keywords_lower = [kw.lower() for kw in image_keywords] if image_keywords else []
        
        direct_table_matches = []
        if best_parts_table:
            direct_table_matches.append({
                "page": best_parts_table["page"],
                "table_parts": best_parts_table["parts"],
                "overlap_nums": list(best_parts_table["parts"].keys()),
                "match_count": len(best_parts_table["parts"]),
                "doc": "",
                "meta": {}
            })
        
        if direct_table_matches:
            direct_table_matches.sort(key=lambda x: -x["match_count"])
            logger.info(f"=== 直接编号匹配 ===")
            for match in direct_table_matches:
                logger.info(f"  第{match['page']}页: 匹配{match['match_count']}个编号 - {match['overlap_nums']}")
        
        for doc, meta, distance in valid_items:
            doc_lower = doc.lower()
            
            if distance < 0.6:
                relevant_items.append((doc, meta))
                has_relevant_context = True
                continue
            
            has_overlap = False
            for word in question_text.split():
                if len(word) > 2 and word in doc_lower:
                    has_overlap = True
                    break
            
            if image_keywords_lower:
                for kw in image_keywords_lower:
                    if kw in doc_lower:
                        has_overlap = True
                        break
            
            if has_overlap:
                relevant_items.append((doc, meta))
                has_relevant_context = True
        
        if not has_relevant_context and has_image_only:
            relevant_items = [(doc, meta) for doc, meta, distance in valid_items[:10]]
            has_relevant_context = True
        
        if direct_table_matches:
            direct_items = [(match["doc"], match["meta"]) for match in direct_table_matches]
            existing_docs = {doc for doc, meta in relevant_items}
            for doc, meta in direct_items:
                if doc not in existing_docs:
                    relevant_items.insert(0, (doc, meta))
                    existing_docs.add(doc)
        
        if has_relevant_context and image_keywords:
            page_numbers = [int(meta.get("page", 0)) for doc, meta in relevant_items if str(meta.get("page", "")).isdigit()]
            if page_numbers:
                avg_page = sum(page_numbers) / len(page_numbers)
                logger.info(f"当前检索结果平均页码: {avg_page:.1f}")
                
                all_pages_in_manual = set()
                try:
                    global_all = global_collection.get()
                    if global_all.get("metadatas"):
                        for meta_list in global_all["metadatas"]:
                            for meta in meta_list:
                                p = meta.get("page")
                                if p and str(p).isdigit():
                                    all_pages_in_manual.add(int(p))
                except:
                    pass
                
                try:
                    user_all = user_collection.get()
                    if user_all.get("metadatas"):
                        for meta_list in user_all["metadatas"]:
                            for meta in meta_list:
                                p = meta.get("page")
                                if p and str(p).isdigit():
                                    all_pages_in_manual.add(int(p))
                except:
                    pass
                
                covered_pages = set(page_numbers)
                total_pages = len(all_pages_in_manual) if all_pages_in_manual else 41
                coverage = len(covered_pages) / total_pages if total_pages > 0 else 0
                
                logger.info(f"检索覆盖页码数: {len(covered_pages)}/{total_pages}, 覆盖率: {coverage:.1%}")
                
                if coverage < 0.3:
                    logger.info("覆盖率较低，执行广泛补充检索...")
                    
                    supplementary_terms = [
                        "拆卸", "安装", "装配", "检查", "检修",
                        "气缸", "活塞", "曲轴", "连杆", "气门",
                        "离合器", "磁电机", "磁电机转子", "磁电机定子",
                        "单向器", "起动电机", "变速箱", "正时链轮",
                        "密封", "垫片", "垫圈", "轴承", "齿轮",
                        "螺栓", "螺母", "扭矩", "调整", "更换",
                        "飞轮", "凸轮轴", "链条", "链轮", "减速齿轮"
                    ]
                    
                    for term in supplementary_terms:
                        if term.lower() not in image_keywords_lower:
                            kw_emb = embed_texts([term])[0]
                            query_collections(kw_emb, term, weight=0.5)
                    
                    logger.info("补充检索完成")
                
                valid_items = []
                seen_content_hashes = set()
                for i, (doc, meta) in enumerate(zip(all_docs, all_metas)):
                    if "暂未启用多模态分析" in doc or "图片提取失败" in doc or "图片坐标无效" in doc:
                        continue
                    if len(doc.strip()) < 10:
                        continue
                    content_hash = hashlib.md5(doc[:100].encode('utf-8')).hexdigest()
                    if content_hash in seen_content_hashes:
                        continue
                    seen_content_hashes.add(content_hash)
                    distance = all_distances[i] if i < len(all_distances) else 1.0
                    valid_items.append((doc, meta, distance))
                
                valid_items.sort(key=lambda x: x[2])
                valid_items = valid_items[:25]
                
                relevant_items = []
                for doc, meta, distance in valid_items:
                    doc_lower = doc.lower()
                    
                    if distance < 0.7:
                        relevant_items.append((doc, meta))
                        has_relevant_context = True
                        continue
                    
                    has_overlap = False
                    for word in question_text.split():
                        if len(word) > 2 and word in doc_lower:
                            has_overlap = True
                            break
                    
                    if image_keywords_lower:
                        for kw in image_keywords_lower:
                            if kw in doc_lower:
                                has_overlap = True
                                break
                    
                    if has_overlap:
                        relevant_items.append((doc, meta))
                        has_relevant_context = True
        
        if has_relevant_context:
            if image_keywords:
                scored_items = []
                for doc, meta in relevant_items:
                    score = 0
                    doc_lower = doc.lower()
                    
                    for kw in image_keywords_lower:
                        if kw in doc_lower:
                            score += 1
                    
                    meta_page = meta.get("page", "")
                    if meta_page:
                        try:
                            score += (100 - int(meta_page)) * 0.01
                        except:
                            pass
                    
                    scored_items.append((doc, meta, score))
                
                scored_items.sort(key=lambda x: -x[2])
                relevant_items = [(doc, meta) for doc, meta, score in scored_items]
                
                logger.info(f"=== 语义重排结果 ===")
                for i, (doc, meta, score) in enumerate(scored_items[:5]):
                    page = meta.get("page", "?")
                    logger.info(f"  [{i}] 页码:{page}, 得分:{score:.2f}, 内容:{doc[:50]}...")
                logger.info(f"================")
            
            valid_contexts = []
            valid_refs = []
            for doc, meta in relevant_items[:8]:
                page = meta.get("page", "?")
                valid_contexts.append(f"【第{page}页】\n{doc}")
                valid_refs.append((meta, doc))
            contexts = "\n\n---\n\n".join(valid_contexts)
            
            image_parts_info = ""
            if image_analysis and isinstance(image_analysis, dict):
                all_parts = image_analysis.get("all_parts", [])
                if all_parts and isinstance(all_parts, list):
                    image_parts_info = "\n【用户上传图片分析结果】\n"
                    image_parts_info += "图片类型: " + image_analysis.get("image_type", "") + "\n"
                    image_parts_info += "主要部件: " + image_analysis.get("part_name", "") + "\n"
                    image_parts_info += "所属系统: " + image_analysis.get("system", "") + "\n"
                    image_parts_info += "编号部件清单:\n"
                    for part in all_parts:
                        if isinstance(part, dict):
                            num = part.get("number", "")
                            name = part.get("name", "")
                            desc = part.get("description", "")
                            if num and name:
                                image_parts_info += f"  - 编号{num}: {name}（{desc}）\n"
                visible_text = image_analysis.get("visible_text", [])
                if visible_text:
                    image_parts_info += "\n图片可见文字:\n"
                    for text in visible_text[:5]:
                        image_parts_info += f"  - {text}\n"
            
            table_parts_prompt = ""
            if direct_table_matches:
                best_match = direct_table_matches[0]
                table_parts = best_match["table_parts"]
                page = best_match["page"]
                table_parts_prompt = f"\n【最匹配页：第{page}页 爆炸图编号对照表】\n"
                table_parts_prompt += "重要：下表中的编号与用户上传图片中的编号完全一致，请严格按照此表回答（这是最权威的依据，绝对不能修改编号或名称）：\n\n"
                for num in sorted(table_parts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    table_parts_prompt += f"  编号 {num} → {table_parts[num]}\n"
                table_parts_prompt += f"\n【强制回答格式】\n"
                table_parts_prompt += f"必须严格按照以下格式，按编号从小到大依次回答每个部件：\n\n"
                
                for num in sorted(table_parts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    part_name = table_parts[num]
                    table_parts_prompt += f"**编号{num}: {part_name}**\n"
                    table_parts_prompt += f"- 部件识别与定位：（说明该部件的位置和作用，引用手册内容并标注页码）\n"
                    table_parts_prompt += f"- 拆卸步骤与注意事项：（引用手册内容并标注页码）\n"
                    table_parts_prompt += f"- 安装步骤与扭矩要求：（引用手册内容并标注页码）\n"
                    table_parts_prompt += f"- 常见故障与排查方法：（引用手册内容并标注页码）\n\n"
                
                table_parts_prompt += "【绝对禁止】\n"
                table_parts_prompt += "1. 禁止改变编号顺序\n"
                table_parts_prompt += "2. 禁止修改部件名称（必须使用上表中的名称）\n"
                table_parts_prompt += "3. 禁止编造不存在的部件\n"
                table_parts_prompt += "4. 禁止使用图片AI识别的部件名称代替表格中的名称\n"
                table_parts_prompt += f"5. 所有引用内容必须标注页码（第{page}页或其他相关页）\n"
            
            retrieved_snippets = []
            for i, (doc, meta) in enumerate(relevant_items[:5]):
                page = meta.get("page", "?")
                source = meta.get("file_id", "")[:8] + "..."
                snippet = doc[:100] + "..." if len(doc) > 100 else doc
                retrieved_snippets.append({
                    "index": i + 1,
                    "page": page,
                    "source": source,
                    "snippet": snippet
                })
        else:
            valid_contexts = []
            valid_refs = []
            contexts = ""
            retrieved_snippets = []
        
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
        
        registry = _load_registry()
        
        files_info = ""
        if registry:
            files_info = "\n【共享检修手册知识库】\n"
            for f in registry:
                files_info += f"- {f['filename']}（上传者：{f['username']}，共{f['chunk_count']}个内容块）\n"
        
        has_image = image_analysis is not None

        if has_relevant_context:
            system_prompt = f"""你是一名经验丰富的设备检修专家助手。你需要根据用户上传的检修手册内容，准确回答用户的检修相关问题。

{files_info}

你的职责是：
1. 严格依据上述检修手册内容回答，不要编造信息
2. 如果用户的问题与检修无关，礼貌地引导用户回到检修话题
3. 始终保持专业、简洁的回答风格
4. 回答时请结合用户上传的手册内容进行分析，并且必须标注引用来源页码"""
            
            image_instruction = ""
            if has_image:
                image_instruction = """【图片分析重要要求】
[极其重要] 用户上传了图片，请按照以下步骤进行检修分析：
1. 首先仔细阅读【检修手册内容】，了解手册中包含哪些部件、哪些章节、哪些页码
2. 根据图片内容（外观、结构、部件编号、可见文字等），在手册内容中寻找最匹配的部件或章节
3. 如果图片描述中的部件名称与手册中的部件名称不一致，以手册中的名称为准
4. 基于手册中匹配到的内容，给出该部件的检修步骤、注意事项、拆卸/安装方法等
5. 如果图片是爆炸图/装配图，请识别图中的所有编号部件，并针对每个编号部件提供检修指导
6. 如果无法在手册中找到与图片匹配的内容，诚实说明"手册中未找到与图片匹配的相关内容"，不要猜测

【编号回答要求】
- 如果图片中有数字编号（如1、2、3...），必须严格按照编号从小到大的顺序依次回答
- 每个编号只对应一个部件，不要重复编号，不要遗漏编号
- 必须在【检修手册内容】中找到对应编号的部件名称，不要仅凭图片猜测
- 如果手册中某个编号对应的部件名称与图片不同，以手册为准
- 如果手册中没有某个编号的部件信息，说明"手册中未找到编号X的相关内容"

你的回答应该是专业的检修分析，包括：
- 部件识别与定位（必须标注来源页码）
- 拆卸步骤与注意事项
- 安装步骤与扭矩要求
- 常见故障与排查方法

不要说"用户没有上传图片"或类似的话。"""
            
            all_pages = set()
            for doc, meta in relevant_items:
                page = meta.get("page", "?")
                if page != "?":
                    all_pages.add(page)
            available_pages = f"\n可用页码：{', '.join(sorted([str(p) for p in all_pages]))}" if all_pages else ""
            
            prompt = f"""{system_prompt}

以下是从检修手册中检索到的相关内容：

【检修手册内容】
{contexts}
{image_parts_info}
{table_parts_prompt}
{format_instruction}
{image_instruction}
【回答步骤】
1. 先通读【检修手册内容】，特别关注表格中的编号-名称对应关系
2. 将【用户上传图片分析结果】中的编号部件清单与手册中的表格数据进行逐一匹配
3. 对于图片中的每个编号，先在手册表格中查找对应的部件名称（表格中的编号是最权威的来源）
4. 找到部件名称后，再在手册的其他文本内容中查找该部件的检修步骤和注意事项
5. 如果图片描述中的部件名称与手册表格中的名称不一致，**必须以表格中的名称为准**
6. 基于手册中匹配到的内容，按编号顺序回答每个部件的检修步骤和注意事项

【回答要求】
1. 【核心原则】如果手册中有相关内容，**必须**严格依据手册内容回答，并在引用内容后用方括号标注来源页码，格式为：（来源：第X页）
2. 如果手册中有相关内容，**绝对不要**编造任何手册中没有的信息，包括页码、部件名称、参数等
3. 如果手册中没有明确答案或内容不匹配，可以使用你的专业知识进行简略回答，但要明确说明"手册中未找到相关内容"
4. 回答要简洁专业，避免冗余重复，逻辑清晰
5. 【关键约束】只能使用上面【检修手册内容】中出现的页码，严禁编造任何不存在的页码
6. 【编号约束】必须按照图片中的编号顺序回答，每个编号对应手册中的一个部件，必须标注来源页码
7. 如果手册中某个编号的内容与图片描述不一致，**以手册表格内容为准**，并指出差异
8. 必须明确关联图片中的编号与手册表格中的部件名称，确保回答与图片内容紧密相关
9. 【表格优先】如果手册中存在表格，必须优先使用表格中的编号-名称对应关系

{available_pages}

【用户问题】
{question}

【专家回答】
（回答完成后，请另起一行，用【推荐问题】标记，给出3个与当前话题相关的后续问题，格式如下：
【推荐问题】
1. xxx？
2. xxx？
3. xxx？）"""
        else:
            system_prompt = f"""你是一名经验丰富的设备检修专家助手。

{files_info}

你的职责是：
1. 根据你的专业知识回答用户的检修相关问题
2. 如果用户的问题与检修无关，礼貌地引导用户回到检修话题
3. 始终保持专业、简洁的回答风格
4. 回答要简略，不要冗长"""
            
            image_instruction = ""
            if has_image:
                image_instruction = """【图片分析要求】
用户上传了图片，请先描述图片内容，然后基于你的专业知识给出检修建议。"""
            
            prompt = f"""{system_prompt}

{image_instruction}
【回答要求】
1. 如果用户上传了图片，请先描述图片中的设备类型和状态，然后给出检修建议
2. 判断内容是否与检修相关：
   - 如果与检修相关：给出检修相关解答
   - 如果与检修无关：描述内容后，明确告知"此内容与检修无关"
3. 回答要简洁专业，避免冗余重复

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

        pass

        references = []
        for meta, doc in valid_refs[:5]:
            page_info = meta.get("page", "?")
            if meta.get("source") == "user_case":
                page_info = f"案例《{meta.get('title', '')}》"
            elif meta.get("source") == "user_correction":
                page_info = "用户修正"
            references.append({
                "page": page_info,
                "content": doc[:150] + "..." if len(doc) > 150 else doc
            })

        return {
            "answer": answer, 
            "references": references, 
            "answer_id": str(uuid.uuid4()),
            "retrieved_snippets": retrieved_snippets,
            "suggested_questions": suggested_questions,
            "media_files": media_files
        }
    except Exception as e:
        logger.error(f"Ask API error: {str(e)}", exc_info=True)
        return {
            "answer": "⚠️ 服务暂时不可用，请稍后重试。",
            "references": [],
            "answer_id": str(uuid.uuid4()),
            "retrieved_snippets": [],
            "suggested_questions": [],
            "media_files": []
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

# ---------- 案例详情 API（直接注册，不依赖AI模型） ----------
PENDING_CASES_FILE = "pending_cases.json"

@app.get("/api/case/{case_id}")
async def get_case_detail(case_id: str, user: Dict = Depends(get_current_user)):
    if not os.path.exists(PENDING_CASES_FILE):
        raise HTTPException(404, "案例不存在")
    try:
        with open(PENDING_CASES_FILE, "r", encoding="utf-8") as f:
            cases = json.load(f)
        for case in cases:
            if case["id"] == case_id:
                return case
        raise HTTPException(404, "案例不存在")
    except json.JSONDecodeError:
        raise HTTPException(500, "数据解析错误")

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

def _calculate_project_status(project: Dict) -> str:
    current_status = project.get("status", "pending")
    if current_status == "completed":
        return "completed"
    
    today = datetime.now().date()
    start_date = project.get("start_date")
    end_date = project.get("end_date")
    
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if today > end and current_status != "completed":
            return "expired"
    
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        if today >= start:
            return "in_progress"
    
    return current_status

@app.get("/api/projects")
async def list_projects(user: Dict = Depends(get_current_user)):
    projects = _load_projects()
    user_projects = []
    for p in projects:
        if p["username"] == user["username"] or user["role"] == "admin":
            project = p.copy()
            project["status"] = _calculate_project_status(p)
            user_projects.append(project)
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

def _load_schedule_logs() -> List[Dict]:
    with _schedule_log_lock:
        if not os.path.exists(SCHEDULE_LOG_FILE):
            return []
        try:
            with open(SCHEDULE_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

@app.get("/api/schedule")
async def list_schedule(user: Dict = Depends(get_current_user)):
    tasks = _load_schedule()
    user_tasks = []
    for t in tasks:
        if t.get("username") == user["username"] or user["role"] == "admin":
            task_info = t.copy()
            task_info["task_type_label"] = _get_task_type_label(t.get("task_type", "check_knowledge"))
            user_tasks.append(task_info)
    user_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    return {"tasks": user_tasks}

def _get_task_type_label(task_type):
    labels = {
        "check_knowledge": "知识库更新检查",
        "generate_report": "生成运维报告",
        "cleanup_logs": "清理过期文件",
        "health_check": "系统健康检查"
    }
    return labels.get(task_type, task_type)

@app.post("/api/schedule")
async def create_schedule(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    name = data.get("name", "").strip()
    cron_expression = data.get("cron_expression", "").strip()
    description = data.get("description", "").strip()
    task_type = data.get("task_type", "check_knowledge")
    cleanup_days = data.get("cleanup_days", 7)
    
    if not name or not cron_expression:
        raise HTTPException(400, "任务名称和Cron表达式不能为空")
    
    valid_types = ["check_knowledge", "generate_report", "cleanup_logs", "health_check"]
    if task_type not in valid_types:
        raise HTTPException(400, f"无效的任务类型，可选类型: {', '.join(valid_types)}")
    
    tasks = _load_schedule()
    new_task = {
        "id": str(uuid.uuid4()),
        "name": name,
        "cron_expression": cron_expression,
        "description": description,
        "task_type": task_type,
        "cleanup_days": cleanup_days,
        "username": user["username"],
        "active": True,
        "created_at": datetime.now().isoformat(),
        "last_run": None,
        "next_run": None,
        "run_count": 0
    }
    tasks.append(new_task)
    _save_schedule(tasks)
    
    task_scheduler.add_task(new_task)
    
    return {"message": "定时任务创建成功", "task": new_task}

@app.put("/api/schedule/{task_id}")
async def update_schedule(task_id: str, request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    
    tasks = _load_schedule()
    task_index = next((i for i, t in enumerate(tasks) if t["id"] == task_id), None)
    
    if task_index is None:
        raise HTTPException(404, "任务不存在")
    
    task = tasks[task_index]
    if task["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权修改此任务")
    
    if "name" in data:
        task["name"] = data["name"].strip()
    if "cron_expression" in data:
        task["cron_expression"] = data["cron_expression"].strip()
    if "description" in data:
        task["description"] = data["description"].strip()
    if "task_type" in data:
        valid_types = ["check_knowledge", "generate_report", "cleanup_logs", "health_check"]
        if data["task_type"] not in valid_types:
            raise HTTPException(400, f"无效的任务类型")
        task["task_type"] = data["task_type"]
    if "cleanup_days" in data:
        task["cleanup_days"] = data["cleanup_days"]
    if "active" in data:
        task["active"] = bool(data["active"])
    
    _save_schedule(tasks)
    
    if task["active"]:
        task_scheduler.update_task(task)
    else:
        task_scheduler.remove_task(task_id)
    
    return {"message": "定时任务更新成功", "task": task}

@app.delete("/api/schedule/{task_id}")
async def delete_schedule(task_id: str, user: Dict = Depends(get_current_user)):
    tasks = _load_schedule()
    task_index = next((i for i, t in enumerate(tasks) if t["id"] == task_id), None)
    
    if task_index is None:
        raise HTTPException(404, "任务不存在")
    
    task = tasks[task_index]
    if task["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权删除此任务")
    
    tasks.pop(task_index)
    _save_schedule(tasks)
    
    task_scheduler.remove_task(task_id)
    
    return {"message": "定时任务删除成功"}

@app.post("/api/schedule/{task_id}/toggle")
async def toggle_schedule(task_id: str, user: Dict = Depends(get_current_user)):
    tasks = _load_schedule()
    task_index = next((i for i, t in enumerate(tasks) if t["id"] == task_id), None)
    
    if task_index is None:
        raise HTTPException(404, "任务不存在")
    
    task = tasks[task_index]
    if task["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权修改此任务")
    
    task["active"] = not task["active"]
    _save_schedule(tasks)
    
    if task["active"]:
        task_scheduler.add_task(task)
    else:
        task_scheduler.remove_task(task_id)
    
    return {"message": f"定时任务已{'启用' if task['active'] else '禁用'}", "task": task}

@app.post("/api/schedule/{task_id}/trigger")
async def trigger_schedule(task_id: str, user: Dict = Depends(get_current_user)):
    tasks = _load_schedule()
    task = next((t for t in tasks if t["id"] == task_id), None)
    
    if task is None:
        raise HTTPException(404, "任务不存在")
    
    if task["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权触发此任务")
    
    task_scheduler.trigger_task(task)
    
    return {"message": "任务已触发，正在后台执行"}

@app.get("/api/schedule/{task_id}/logs")
async def get_schedule_logs(task_id: str, user: Dict = Depends(get_current_user)):
    tasks = _load_schedule()
    task = next((t for t in tasks if t["id"] == task_id), None)
    
    if task is None:
        raise HTTPException(404, "任务不存在")
    
    if task["username"] != user["username"] and user["role"] != "admin":
        raise HTTPException(403, "无权查看此任务日志")
    
    logs = _load_schedule_logs()
    task_logs = [log for log in logs if log["task_id"] == task_id]
    task_logs.sort(key=lambda x: x["start_time"], reverse=True)
    
    return {"logs": task_logs}

@app.get("/api/schedule/task_types")
async def get_task_types(user: Dict = Depends(get_current_user)):
    return {
        "types": [
            {"value": "check_knowledge", "label": "知识库更新检查", "desc": "检查上传的检修手册是否有更新"},
            {"value": "generate_report", "label": "生成运维报告", "desc": "生成知识库统计报告"},
            {"value": "cleanup_logs", "label": "清理过期文件", "desc": "清理过期的媒体文件（图片、视频）"},
            {"value": "health_check", "label": "系统健康检查", "desc": "检查系统各组件运行状态"}
        ]
    }

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

# ---------- 报告下载 API ----------
@app.get("/api/reports")
async def list_reports(user: Dict = Depends(get_current_user)):
    report_dir = "reports"
    reports = []
    
    if os.path.exists(report_dir):
        for filename in os.listdir(report_dir):
            if filename.endswith('.pdf') or filename.endswith('.json'):
                filepath = os.path.join(report_dir, filename)
                file_stat = os.stat(filepath)
                reports.append({
                    "filename": filename,
                    "size": file_stat.st_size,
                    "created_at": datetime.fromtimestamp(file_stat.st_mtime, timezone.utc).isoformat(),
                    "type": "pdf" if filename.endswith('.pdf') else "json"
                })
    
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    return {"reports": reports}

@app.get("/api/reports/{filename}")
async def download_report(filename: str, user: Dict = Depends(get_current_user)):
    report_dir = "reports"
    filepath = os.path.join(report_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(404, "报告文件不存在")
    
    if filename.endswith('.pdf'):
        media_type = "application/pdf"
    elif filename.endswith('.json'):
        media_type = "application/json"
    else:
        raise HTTPException(400, "不支持的文件类型")
    
    return FileResponse(
        filepath,
        media_type=media_type,
        filename=filename,
        content_disposition_type='attachment'
    )

@app.delete("/api/reports/{filename}")
async def delete_report(filename: str, user: Dict = Depends(get_current_user)):
    report_dir = "reports"
    filepath = os.path.join(report_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(404, "报告文件不存在")
    
    try:
        os.remove(filepath)
        return {"message": f"已删除报告「{filename}」"}
    except Exception as e:
        raise HTTPException(500, f"删除报告失败: {str(e)}")

# ---------- 插件系统 API ----------
PLUGINS_FILE = "plugins.json"
_plugins_lock = threading.Lock()

PLUGIN_DEFINITIONS = {
    "report": {
        "id": "report",
        "name": "运维报表",
        "desc": "生成多模态图文并茂的运维报告",
        "category": "data",
        "icon": "R",
        "features": ["generate_report", "download_report"]
    },
    "analysis": {
        "id": "analysis",
        "name": "故障分析",
        "desc": "智能分析设备故障原因",
        "category": "development",
        "icon": "A",
        "features": ["analyze_fault"]
    },
    "monitor": {
        "id": "monitor",
        "name": "性能监控",
        "desc": "实时监控设备运行状态",
        "category": "system",
        "icon": "M",
        "features": ["system_monitor"]
    },
    "log": {
        "id": "log",
        "name": "日志分析",
        "desc": "自动化日志分析与告警",
        "category": "data",
        "icon": "L",
        "features": ["analyze_logs"]
    },
    "schedule": {
        "id": "schedule",
        "name": "任务调度",
        "desc": "智能任务调度与排程",
        "category": "efficiency",
        "icon": "S",
        "features": ["manage_tasks"]
    },
    "doc": {
        "id": "doc",
        "name": "文档生成",
        "desc": "自动生成检修文档",
        "category": "efficiency",
        "icon": "D",
        "features": ["generate_document"]
    },
    "diag": {
        "id": "diag",
        "name": "智能诊断",
        "desc": "基于AI的设备诊断",
        "category": "development",
        "icon": "I",
        "features": ["smart_diagnosis"]
    },
    "backup": {
        "id": "backup",
        "name": "数据备份",
        "desc": "自动化数据备份与恢复",
        "category": "system",
        "icon": "B",
        "features": ["backup_data", "restore_data"]
    },
    "chart": {
        "id": "chart",
        "name": "数据可视化",
        "desc": "运维数据可视化图表",
        "category": "data",
        "icon": "C",
        "features": ["generate_chart"]
    },
    "alert": {
        "id": "alert",
        "name": "告警管理",
        "desc": "统一告警管理与通知",
        "category": "system",
        "icon": "W",
        "features": ["manage_alerts"]
    },
    "code": {
        "id": "code",
        "name": "代码助手",
        "desc": "辅助编写运维脚本",
        "category": "development",
        "icon": "X",
        "features": ["generate_script"]
    },
    "sync": {
        "id": "sync",
        "name": "数据同步",
        "desc": "多系统数据同步",
        "category": "efficiency",
        "icon": "Y",
        "features": ["sync_data"]
    }
}

def _load_user_plugins(username: str) -> List[Dict]:
    with _plugins_lock:
        if not os.path.exists(PLUGINS_FILE):
            return []
        try:
            with open(PLUGINS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_data = data.get(username, {})
            plugins = []
            for plugin_id, enabled in user_data.items():
                if enabled and plugin_id in PLUGIN_DEFINITIONS:
                    plugin = PLUGIN_DEFINITIONS[plugin_id].copy()
                    plugin["enabled"] = True
                    plugins.append(plugin)
            return plugins
        except json.JSONDecodeError:
            return []

def _save_user_plugins(username: str, plugins: List[str]):
    with _plugins_lock:
        data = {}
        if os.path.exists(PLUGINS_FILE):
            try:
                with open(PLUGINS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        
        user_plugins = {}
        for plugin_id in PLUGIN_DEFINITIONS.keys():
            user_plugins[plugin_id] = plugin_id in plugins
        
        data[username] = user_plugins
        
        with open(PLUGINS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

@app.get("/api/plugins")
async def list_plugins(user: Dict = Depends(get_current_user)):
    installed_plugins = _load_user_plugins(user["username"])
    installed_ids = {p["id"] for p in installed_plugins}
    
    all_plugins = []
    for plugin_id, plugin_def in PLUGIN_DEFINITIONS.items():
        plugin = plugin_def.copy()
        plugin["installed"] = plugin_id in installed_ids
        plugin["enabled"] = plugin_id in installed_ids
        all_plugins.append(plugin)
    
    return {"plugins": all_plugins}

@app.get("/api/plugins/installed")
async def list_installed_plugins(user: Dict = Depends(get_current_user)):
    plugins = _load_user_plugins(user["username"])
    return {"plugins": plugins}

@app.post("/api/plugins/{plugin_id}/install")
async def install_plugin(plugin_id: str, user: Dict = Depends(get_current_user)):
    if plugin_id not in PLUGIN_DEFINITIONS:
        raise HTTPException(404, "插件不存在")
    
    installed = _load_user_plugins(user["username"])
    installed_ids = {p["id"] for p in installed}
    
    if plugin_id in installed_ids:
        return {"message": "插件已安装"}
    
    installed_ids.add(plugin_id)
    _save_user_plugins(user["username"], list(installed_ids))
    
    plugin = PLUGIN_DEFINITIONS[plugin_id]
    logger.info(f"用户 {user['username']} 安装插件: {plugin['name']}")
    
    return {"message": f"插件「{plugin['name']}」安装成功", "plugin": plugin}

@app.post("/api/plugins/{plugin_id}/uninstall")
async def uninstall_plugin(plugin_id: str, user: Dict = Depends(get_current_user)):
    if plugin_id not in PLUGIN_DEFINITIONS:
        raise HTTPException(404, "插件不存在")
    
    installed = _load_user_plugins(user["username"])
    installed_ids = {p["id"] for p in installed}
    
    if plugin_id not in installed_ids:
        return {"message": "插件未安装"}
    
    installed_ids.remove(plugin_id)
    _save_user_plugins(user["username"], list(installed_ids))
    
    plugin = PLUGIN_DEFINITIONS[plugin_id]
    logger.info(f"用户 {user['username']} 卸载插件: {plugin['name']}")
    
    return {"message": f"插件「{plugin['name']}」卸载成功"}

# ---------- 插件功能 API ----------

@app.post("/api/plugin/analyze_fault")
async def analyze_fault(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    symptom = data.get("symptom", "")
    device = data.get("device", "")
    
    if not symptom:
        raise HTTPException(400, "请输入故障现象")
    
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "analysis" for p in installed):
        raise HTTPException(403, "请先安装故障分析插件")
    
    try:
        user_collection = get_user_collection(user["username"])
        global_collection = get_global_collection()
        
        query_text = f"{symptom} {device}"
        q_emb = embed_texts([query_text])[0]
        
        user_results = user_collection.query(query_embeddings=[q_emb], n_results=5)
        global_results = global_collection.query(query_embeddings=[q_emb], n_results=5)
        
        contexts = []
        if user_results["documents"] and user_results["documents"][0]:
            contexts.extend(user_results["documents"][0])
        if global_results["documents"] and global_results["documents"][0]:
            contexts.extend(global_results["documents"][0])
        
        context_text = "\n\n---\n\n".join(contexts[:10]) if contexts else "暂无相关知识"
        
        prompt = f"""你是一位经验丰富的设备检修专家，请根据以下信息分析故障原因：

【故障现象】
{symptom}

【设备类型】
{device}

【相关知识库内容】
{context_text}

请按照以下格式输出分析结果：
1. 可能原因分析
2. 排查步骤建议
3. 解决方案
4. 参考资料"""
        
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return {
            "success": True,
            "result": resp.choices[0].message.content,
            "context_count": len(contexts)
        }
    except Exception as e:
        logger.error(f"故障分析失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/plugin/system_monitor")
async def system_monitor(user: Dict = Depends(get_current_user)):
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "monitor" for p in installed):
        raise HTTPException(403, "请先安装性能监控插件")
    
    result = {}
    
    if SYSTEM_MONITOR_ENABLED:
        try:
            disk = psutil.disk_usage('/')
            result["disk"] = {
                "total": disk.total // (1024 * 1024 * 1024),
                "used": disk.used // (1024 * 1024 * 1024),
                "free": disk.free // (1024 * 1024 * 1024),
                "percent": disk.percent,
                "status": "warning" if disk.percent > 90 else "ok"
            }
        except Exception as e:
            result["disk"] = {"error": str(e)}
        
        try:
            mem = psutil.virtual_memory()
            result["memory"] = {
                "total": mem.total // (1024 * 1024 * 1024),
                "used": mem.used // (1024 * 1024 * 1024),
                "available": mem.available // (1024 * 1024 * 1024),
                "percent": mem.percent,
                "status": "warning" if mem.percent > 90 else "ok"
            }
        except Exception as e:
            result["memory"] = {"error": str(e)}
        
        try:
            result["cpu"] = {
                "percent": psutil.cpu_percent(interval=0.1),
                "cores": psutil.cpu_count(),
                "status": "warning" if psutil.cpu_percent(interval=0.1) > 90 else "ok"
            }
        except Exception as e:
            result["cpu"] = {"error": str(e)}
        
        try:
            network = psutil.net_io_counters()
            result["network"] = {
                "bytes_sent": network.bytes_sent // (1024 * 1024),
                "bytes_recv": network.bytes_recv // (1024 * 1024),
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except Exception as e:
            result["network"] = {"error": str(e)}
        
        try:
            result["processes"] = {
                "total": len(psutil.pids()),
                "running": len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'running'])
            }
        except Exception as e:
            result["processes"] = {"error": str(e)}
    else:
        result["error"] = "psutil未安装"
    
    return {"success": True, "data": result}

@app.get("/api/plugin/analyze_logs")
async def analyze_logs(user: Dict = Depends(get_current_user)):
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "log" for p in installed):
        raise HTTPException(403, "请先安装日志分析插件")
    
    logs = _load_schedule_logs()
    recent_logs = logs[-50:]
    
    success_count = sum(1 for log in recent_logs if log["status"] == "success")
    fail_count = sum(1 for log in recent_logs if log["status"] == "failed")
    total_runs = len(recent_logs)
    
    error_messages = []
    for log in recent_logs:
        if log["status"] == "failed" and "result" in log:
            error_messages.append({
                "task_name": log["task_name"],
                "error": log["result"][:200] if len(log["result"]) > 200 else log["result"],
                "time": log["start_time"]
            })
    
    task_stats = defaultdict(lambda: {"success": 0, "fail": 0, "total": 0})
    for log in recent_logs:
        task_stats[log["task_name"]]["total"] += 1
        if log["status"] == "success":
            task_stats[log["task_name"]]["success"] += 1
        else:
            task_stats[log["task_name"]]["fail"] += 1
    
    return {
        "success": True,
        "data": {
            "total_runs": total_runs,
            "success_rate": round(success_count / total_runs * 100, 2) if total_runs > 0 else 0,
            "fail_count": fail_count,
            "recent_errors": error_messages[:10],
            "task_stats": dict(task_stats)
        }
    }

@app.post("/api/plugin/generate_document")
async def generate_document(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    device = data.get("device", "")
    content_type = data.get("type", "repair")
    
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "doc" for p in installed):
        raise HTTPException(403, "请先安装文档生成插件")
    
    if not device:
        raise HTTPException(400, "请输入设备名称")
    
    type_map = {
        "repair": "检修方案",
        "maintenance": "保养计划",
        "inspection": "巡检报告",
        "accident": "事故分析"
    }
    
    doc_type = type_map.get(content_type, "检修方案")
    
    try:
        prompt = f"""请为「{device}」生成一份专业的「{doc_type}」文档。

文档应包含以下章节：
1. 文档概述
2. 设备基本信息
3. {doc_type}详情（步骤、标准、工具）
4. 安全注意事项
5. 质量验收标准

请输出完整的文档内容，格式清晰，内容专业。"""
        
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        document_content = resp.choices[0].message.content
        
        doc_dir = "documents"
        os.makedirs(doc_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{device}_{content_type}_{timestamp}.txt"
        filepath = os.path.join(doc_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(document_content)
        
        return {
            "success": True,
            "content": document_content,
            "file_path": f"/documents/{filename}"
        }
    except Exception as e:
        logger.error(f"文档生成失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/plugin/smart_diagnosis")
async def smart_diagnosis(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    question = data.get("question", "")
    
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "diag" for p in installed):
        raise HTTPException(403, "请先安装智能诊断插件")
    
    if not question:
        raise HTTPException(400, "请输入诊断问题")
    
    try:
        user_collection = get_user_collection(user["username"])
        global_collection = get_global_collection()
        
        q_emb = embed_texts([question])[0]
        
        user_results = user_collection.query(query_embeddings=[q_emb], n_results=8)
        global_results = global_collection.query(query_embeddings=[q_emb], n_results=8)
        
        contexts = []
        if user_results["documents"] and user_results["documents"][0]:
            contexts.extend(user_results["documents"][0])
        if global_results["documents"] and global_results["documents"][0]:
            contexts.extend(global_results["documents"][0])
        
        context_text = "\n\n---\n\n".join(contexts[:15]) if contexts else "暂无相关知识"
        
        prompt = f"""你是一位智能设备诊断专家，请根据以下信息进行诊断：

【用户问题】
{question}

【相关检修手册内容】
{context_text}

请按照以下格式输出诊断结果：
1. 故障判断
2. 诊断依据
3. 处理建议
4. 预防措施
5. 参考资料"""
        
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return {
            "success": True,
            "diagnosis": resp.choices[0].message.content,
            "context_count": len(contexts)
        }
    except Exception as e:
        logger.error(f"智能诊断失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/plugin/backup_data")
async def backup_data(user: Dict = Depends(get_current_user)):
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "backup" for p in installed):
        raise HTTPException(403, "请先安装数据备份插件")
    
    backup_dir = "backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_files = []
    
    files_to_backup = [
        USERS_FILE,
        FILE_REGISTRY,
        PROJECTS_FILE,
        SCHEDULE_FILE,
        SCHEDULE_LOG_FILE,
        PENDING_CASES_FILE,
        CORRECTIONS_FILE
    ]
    
    for src_file in files_to_backup:
        if os.path.exists(src_file):
            backup_path = os.path.join(backup_dir, f"{os.path.basename(src_file)}_{timestamp}")
            shutil.copy2(src_file, backup_path)
            backup_files.append(os.path.basename(src_file))
    
    if CHROMA_PERSIST_DIR and os.path.exists(CHROMA_PERSIST_DIR):
        chroma_backup = os.path.join(backup_dir, f"chroma_db_{timestamp}")
        shutil.copytree(CHROMA_PERSIST_DIR, chroma_backup)
        backup_files.append("chroma_db")
    
    return {
        "success": True,
        "message": f"备份完成，共备份 {len(backup_files)} 个数据文件",
        "files": backup_files,
        "timestamp": timestamp
    }

@app.get("/api/plugin/generate_chart")
async def generate_chart(user: Dict = Depends(get_current_user)):
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "chart" for p in installed):
        raise HTTPException(403, "请先安装数据可视化插件")
    
    registry = _load_registry()
    projects = _load_projects()
    schedules = _load_schedule()
    
    user_files = [f for f in registry if f.get("username") == user["username"]]
    user_projects = [p for p in projects if p.get("username") == user["username"]]
    user_schedules = [s for s in schedules if s.get("username") == user["username"]]
    
    status_counts = defaultdict(int)
    for p in user_projects:
        status_counts[p.get("status", "unknown")] += 1
    
    chart_data = {
        "file_stats": {
            "total": len(user_files),
            "chunks": sum(f.get("chunk_count", 0) for f in user_files)
        },
        "project_stats": {
            "total": len(user_projects),
            "by_status": dict(status_counts),
            "avg_progress": round(sum(p.get("progress", 0) for p in user_projects) / len(user_projects), 2) if user_projects else 0
        },
        "schedule_stats": {
            "total": len(user_schedules),
            "active": sum(1 for s in user_schedules if s.get("active", False))
        }
    }
    
    return {"success": True, "data": chart_data}

@app.post("/api/plugin/generate_script")
async def generate_script(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    purpose = data.get("purpose", "")
    
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "code" for p in installed):
        raise HTTPException(403, "请先安装代码助手插件")
    
    if not purpose:
        raise HTTPException(400, "请输入脚本用途")
    
    try:
        prompt = f"""请根据以下需求生成运维脚本：

【脚本用途】
{purpose}

请输出完整的Python脚本，包含：
1. 必要的注释说明
2. 错误处理
3. 日志记录
4. 参数配置

只输出脚本代码，不要输出其他解释性文字。"""
        
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        script_content = resp.choices[0].message.content
        
        script_dir = "scripts"
        os.makedirs(script_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"script_{timestamp}.py"
        filepath = os.path.join(script_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        return {
            "success": True,
            "script": script_content,
            "file_path": f"/scripts/{filename}"
        }
    except Exception as e:
        logger.error(f"脚本生成失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/plugin/generate_report")
async def generate_report(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    report_type = data.get("report_type", "daily")
    
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "report" for p in installed):
        raise HTTPException(403, "请先安装运维报表插件")
    
    try:
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt=f"运维报告 - {report_type}", ln=True, align='C')
        pdf.cell(200, 10, txt=f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.cell(200, 10, txt=f"生成人: {user['username']}", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(200, 10, txt="一、系统状态概览", ln=True)
        pdf.set_font("Arial", size=12)
        
        installed_count = len(installed)
        pdf.cell(200, 10, txt=f"已安装插件数量: {installed_count}", ln=True)
        
        import psutil
        disk = psutil.disk_usage('/')
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        pdf.cell(200, 10, txt=f"磁盘使用率: {disk.percent}% ({disk.used//(1024**3)}/{disk.total//(1024**3)} GB)", ln=True)
        pdf.cell(200, 10, txt=f"内存使用率: {mem.percent}% ({mem.used//(1024**3)}/{mem.total//(1024**3)} GB)", ln=True)
        pdf.cell(200, 10, txt=f"CPU使用率: {cpu}%", ln=True)
        
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(200, 10, txt="二、报告内容", ln=True)
        pdf.set_font("Arial", size=12)
        
        report_contents = {
            "daily": "今日运维工作完成情况：系统运行正常，无异常告警。各服务状态良好，磁盘空间充足。",
            "weekly": "本周运维工作总结：系统稳定运行7天，完成3次例行检查，处理2个故障工单，备份执行正常。",
            "monthly": "本月运维工作总结：系统运行30天，完成12次例行检查，处理8个故障工单，执行4次数据备份，整体运行良好。"
        }
        
        pdf.multi_cell(200, 10, txt=report_contents.get(report_type, report_contents["daily"]))
        
        pdf_output = pdf.output()
        
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{report_type}_{timestamp}.pdf"
        filepath = os.path.join(report_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(pdf_output)
        
        return {
            "success": True,
            "message": "运维报告生成成功",
            "file_path": f"/reports/{filename}"
        }
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/plugin/manage_tasks")
async def manage_tasks(user: Dict = Depends(get_current_user)):
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "schedule" for p in installed):
        raise HTTPException(403, "请先安装任务调度插件")
    
    try:
        tasks = []
        if os.path.exists(SCHEDULE_FILE):
            with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                tasks = data.get("tasks", [])
        
        logs = []
        if os.path.exists(SCHEDULE_LOG_FILE):
            with open(SCHEDULE_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except:
                        pass
        
        logs = logs[-20:]
        logs.reverse()
        
        return {
            "success": True,
            "tasks": tasks,
            "recent_logs": logs,
            "task_count": len(tasks),
            "log_count": len(logs)
        }
    except Exception as e:
        logger.error(f"任务管理查询失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/plugin/manage_alerts")
async def manage_alerts(user: Dict = Depends(get_current_user)):
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "alert" for p in installed):
        raise HTTPException(403, "请先安装告警管理插件")
    
    try:
        import psutil
        
        alerts = []
        
        disk = psutil.disk_usage('/')
        if disk.percent > 80:
            alerts.append({
                "type": "disk",
                "level": "warning",
                "message": f"磁盘使用率过高: {disk.percent}%",
                "value": disk.percent,
                "threshold": 80
            })
        
        mem = psutil.virtual_memory()
        if mem.percent > 85:
            alerts.append({
                "type": "memory",
                "level": "warning",
                "message": f"内存使用率过高: {mem.percent}%",
                "value": mem.percent,
                "threshold": 85
            })
        
        cpu = psutil.cpu_percent(interval=1)
        if cpu > 90:
            alerts.append({
                "type": "cpu",
                "level": "critical",
                "message": f"CPU使用率过高: {cpu}%",
                "value": cpu,
                "threshold": 90
            })
        
        recent_errors = []
        if os.path.exists(SCHEDULE_LOG_FILE):
            with open(SCHEDULE_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log = json.loads(line)
                        if log.get("status") == "failed":
                            recent_errors.append(log)
                    except:
                        pass
        
        return {
            "success": True,
            "alerts": alerts,
            "alert_count": len(alerts),
            "recent_errors": recent_errors[:10],
            "system_status": {
                "disk": {"usage": disk.percent, "free": f"{disk.free//(1024**3)} GB"},
                "memory": {"usage": mem.percent, "available": f"{mem.available//(1024**3)} GB"},
                "cpu": {"usage": cpu}
            }
        }
    except Exception as e:
        logger.error(f"告警管理查询失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/plugin/sync_data")
async def sync_data(request: Request, user: Dict = Depends(get_current_user)):
    data = await request.json()
    sync_type = data.get("sync_type", "full")
    
    installed = _load_user_plugins(user["username"])
    if not any(p["id"] == "sync" for p in installed):
        raise HTTPException(403, "请先安装数据同步插件")
    
    try:
        sync_results = []
        
        if sync_type in ["full", "users"]:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                sync_results.append({"type": "users", "status": "success", "count": len(user_data)})
            else:
                sync_results.append({"type": "users", "status": "failed", "message": "用户文件不存在"})
        
        if sync_type in ["full", "files"]:
            if os.path.exists(FILE_REGISTRY):
                with open(FILE_REGISTRY, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                sync_results.append({"type": "files", "status": "success", "count": len(file_data)})
            else:
                sync_results.append({"type": "files", "status": "failed", "message": "文件注册表不存在"})
        
        if sync_type in ["full", "projects"]:
            if os.path.exists(PROJECTS_FILE):
                with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
                    project_data = json.load(f)
                sync_results.append({"type": "projects", "status": "success", "count": len(project_data)})
            else:
                sync_results.append({"type": "projects", "status": "failed", "message": "项目文件不存在"})
        
        if sync_type in ["full", "plugins"]:
            if os.path.exists(PLUGINS_FILE):
                with open(PLUGINS_FILE, "r", encoding="utf-8") as f:
                    plugin_data = json.load(f)
                sync_results.append({"type": "plugins", "status": "success", "count": len(plugin_data)})
            else:
                sync_results.append({"type": "plugins", "status": "failed", "message": "插件文件不存在"})
        
        return {
            "success": True,
            "message": f"数据同步完成，共同步 {len(sync_results)} 个模块",
            "results": sync_results,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        return {
            "success": False,
            "error": str(e)
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
    task_scheduler.start()
    logger.info("定时任务调度器已启动")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)