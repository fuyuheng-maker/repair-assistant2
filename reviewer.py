# reviewer.py
from fastapi import APIRouter, HTTPException, Form

_create_token = None
_load_users = None
_verify_password = None

reviewer_router = APIRouter()

def init_reviewer(create_token_fn, load_users_fn, verify_password_fn):
    global _create_token, _load_users, _verify_password
    _create_token = create_token_fn
    _load_users = load_users_fn
    _verify_password = verify_password_fn

def _get_user(username: str):
    for u in _load_users():
        if u["username"] == username:
            return u
    return None

@reviewer_router.post("/reviewer/login")
async def reviewer_login(username: str = Form(...), password: str = Form(...)):
    user = _get_user(username)
    if not user or not _verify_password(password, user["password_hash"]):
        raise HTTPException(400, detail="用户名或密码错误")
    if user["role"] != "admin":
        raise HTTPException(403, detail="您没有审核员权限")
    # 复用 main.py 的 create_token 函数
    token = _create_token(username, "admin")
    return {"access_token": token, "token_type": "bearer", "role": "admin"}
