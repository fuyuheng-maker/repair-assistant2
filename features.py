import json
import os
import uuid
import re
import threading
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException, Depends, Query, Form

router = APIRouter()

# 全局变量占位，由 init_features 注入
_embed_model = None
_get_user_collection = None  # 函数，用于获取用户 collection
_client = None
_TEXT_MODEL = None
_get_current_user = None
_require_admin = None

PROCEDURE_TEMPLATES = {
    ("发动机", "日常检查"): [
        {"step": 1, "desc": "检查机油液位", "compliance": "冷车时油位应在上下标记之间", "tool": "机油尺"},
        {"step": 2, "desc": "检查冷却液液位", "compliance": "液面不低于 LOW 线", "tool": "目视"},
        {"step": 3, "desc": "检查皮带松紧度", "compliance": "按压10mm无异常，无裂纹", "tool": "手压"},
    ],
    ("发动机", "小修"): [
        {"step": 1, "desc": "更换机油机滤", "compliance": "扭矩25N·m，密封垫圈", "tool": "扭矩扳手"},
        {"step": 2, "desc": "清洗空滤", "compliance": "吹尘方向从内向外", "tool": "压缩空气"},
        {"step": 3, "desc": "检查火花塞", "compliance": "电极间隙0.8-0.9mm", "tool": "塞尺"},
    ],
    ("发动机", "大修"): [
        {"step": 1, "desc": "拆卸发动机总成", "compliance": "使用吊具，保持垂直", "tool": "吊车"},
        {"step": 2, "desc": "解体气缸盖", "compliance": "对角线顺序分2-3次拧松", "tool": "扭矩扳手"},
        {"step": 3, "desc": "测量气缸内径", "compliance": "圆度<0.05mm，圆柱度<0.10mm", "tool": "量缸表"},
    ],
}

PENDING_CASES_FILE = "pending_cases.json"
CORRECTIONS_FILE = "corrections.json"
_json_locks = {PENDING_CASES_FILE: threading.Lock(), CORRECTIONS_FILE: threading.Lock()}

for f in [PENDING_CASES_FILE, CORRECTIONS_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            json.dump([], fp)

def load_json(path):
    lock = _json_locks.get(path, threading.Lock())
    with lock:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def save_json(path, data):
    lock = _json_locks.get(path, threading.Lock())
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- 功能函数（无依赖） ----------
def get_procedure_handler(device, level, username):
    # 输入长度限制，防止 Prompt 注入
    device = device[:50].strip()
    level = level[:20].strip()
    
    template = PROCEDURE_TEMPLATES.get((device, level))
    if template:
        return {"device": device, "level": level, "steps": template, "source": "template"}
    try:
        prompt = f"请为设备“{device}”的“{level}”检修生成详细步骤清单（JSON数组，每项含step、desc、compliance、tool）。只输出JSON，不要输出其他内容。"
        resp = _client.chat.completions.create(
            model=_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = resp.choices[0].message.content
        # 尝试直接解析整个响应为 JSON
        try:
            steps = json.loads(raw)
            if isinstance(steps, list):
                return {"device": device, "level": level, "steps": steps, "source": "ai_generated"}
        except json.JSONDecodeError:
            pass
        # 如果直接解析失败，尝试提取第一个完整的 JSON 数组
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            try:
                steps = json.loads(match.group())
                return {"device": device, "level": level, "steps": steps, "source": "ai_generated"}
            except json.JSONDecodeError:
                pass
        return {"device": device, "level": level, "steps_text": raw, "source": "ai_generated_raw"}
    except Exception as e:
        raise HTTPException(500, f"生成指引失败：{e}")

def submit_case_logic(title, content, type, username):
    cases = load_json(PENDING_CASES_FILE)
    cases.append({
        "id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "type": type,
        "submitter": username,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    })
    save_json(PENDING_CASES_FILE, cases)

def get_pending_cases_logic():
    return [c for c in load_json(PENDING_CASES_FILE) if c["status"] == "pending"]

def review_case_logic(case_id, action):
    cases = load_json(PENDING_CASES_FILE)
    for case in cases:
        if case["id"] == case_id:
            if action == "approve":
                texts = [case["content"]]
                emb = _embed_model.encode(texts).tolist()
                # 审核通过的案例同时加入提交者知识库和全局知识库
                user_col = _get_user_collection(case["submitter"])
                user_col.add(
                    ids=[f"case_{case_id}"],
                    embeddings=emb,
                    metadatas=[{"source": "user_case", "title": case["title"], "type": case["type"]}],
                    documents=texts
                )
                # 加入全局知识库供所有用户检索
                global_col = _get_user_collection("global")
                global_col.add(
                    ids=[f"global_case_{case_id}"],
                    embeddings=emb,
                    metadatas=[{"source": "global_case", "title": case["title"], "type": case["type"], "submitter": case["submitter"]}],
                    documents=texts
                )
                case["status"] = "approved"
            else:
                case["status"] = "rejected"
            save_json(PENDING_CASES_FILE, cases)
            return f"案例已{action}"
    raise HTTPException(404, "案例不存在")

def submit_correction_logic(question, original_answer, correct_answer, username):
    corrections = load_json(CORRECTIONS_FILE)
    corrections.append({
        "id": str(uuid.uuid4()),
        "question": question,
        "original_answer": original_answer,
        "correct_answer": correct_answer,
        "submitter": username,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    })
    save_json(CORRECTIONS_FILE, corrections)

def get_pending_corrections_logic():
    return [c for c in load_json(CORRECTIONS_FILE) if c["status"] == "pending"]

def review_correction_logic(correction_id, action):
    corrections = load_json(CORRECTIONS_FILE)
    for corr in corrections:
        if corr["id"] == correction_id:
            if action == "approve":
                texts = [f"问题：{corr['question']}\n正确答案：{corr['correct_answer']}"]
                emb = _embed_model.encode(texts).tolist()
                # 审核通过的修正加入提交者的个人知识库
                user_col = _get_user_collection(corr["submitter"])
                user_col.add(
                    ids=[f"correction_{correction_id}"],
                    embeddings=emb,
                    metadatas=[{"source": "user_correction"}],
                    documents=texts
                )
                corr["status"] = "approved"
            else:
                corr["status"] = "rejected"
            save_json(CORRECTIONS_FILE, corrections)
            return f"修正已{action}"
    raise HTTPException(404, "修正记录不存在")

# ---------- 动态注册路由 ----------
def init_features(embed_model, get_user_collection, client, text_model, get_current_user, require_admin):
    global _embed_model, _get_user_collection, _client, _TEXT_MODEL, _get_current_user, _require_admin
    _embed_model = embed_model
    _get_user_collection = get_user_collection
    _client = client
    _TEXT_MODEL = text_model
    _get_current_user = get_current_user
    _require_admin = require_admin

    @router.get("/procedure")
    async def procedure(device: str = Query(...), level: str = Query(...), user: Dict = Depends(_get_current_user)):
        return get_procedure_handler(device, level, user["username"])

    @router.post("/submit-case")
    async def submit_case(title: str = Form(...), content: str = Form(...), type: str = Form("经验"), user: Dict = Depends(_get_current_user)):
        submit_case_logic(title, content, type, user["username"])
        return {"message": "案例已提交，等待审核"}

    @router.get("/pending-cases")
    async def pending_cases(admin: Dict = Depends(_require_admin)):
        return {"cases": get_pending_cases_logic()}

    @router.post("/review-case/{case_id}")
    async def review_case(case_id: str, action: str = Query(..., pattern="^(approve|reject)$"), admin: Dict = Depends(_require_admin)):
        msg = review_case_logic(case_id, action)
        return {"message": msg}

    @router.post("/submit-correction")
    async def submit_correction(question: str = Form(...), original_answer: str = Form(...), correct_answer: str = Form(...), user: Dict = Depends(_get_current_user)):
        submit_correction_logic(question, original_answer, correct_answer, user["username"])
        return {"message": "修正建议已提交"}

    @router.get("/pending-corrections")
    async def pending_corrections(admin: Dict = Depends(_require_admin)):
        return {"corrections": get_pending_corrections_logic()}

    @router.post("/review-correction/{correction_id}")
    async def review_correction(correction_id: str, action: str = Query(..., pattern="^(approve|reject)$"), admin: Dict = Depends(_require_admin)):
        msg = review_correction_logic(correction_id, action)
        return {"message": msg}