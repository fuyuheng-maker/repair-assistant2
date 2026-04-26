# reviewer.py
from fastapi import APIRouter, HTTPException, Form
from datetime import datetime, timedelta
from jose import jwt

_SECRET_KEY = None
_ALGORITHM = None
_USERS = None

reviewer_router = APIRouter()

def init_reviewer(secret_key, algorithm, users):
    global _SECRET_KEY, _ALGORITHM, _USERS
    _SECRET_KEY = secret_key
    _ALGORITHM = algorithm
    _USERS = users

@reviewer_router.post("/reviewer/login")
async def reviewer_login(username: str = Form(...), password: str = Form(...)):
    user = _USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(400, detail="用户名或密码错误")
    if user["role"] != "admin":
        raise HTTPException(403, detail="您没有审核员权限")
    expire = datetime.utcnow() + timedelta(minutes=120)
    data = {"sub": username, "role": "admin", "exp": expire}
    token = jwt.encode(data, _SECRET_KEY, algorithm=_ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "role": "admin"}