"""
认证模块 —— 用户登录/登出/权限校验/用户管理/审计日志
"""
import json
import hashlib
import secrets
import logging
import time
import base64
import os as _os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

USERS_FILE = Path(__file__).resolve().parent.parent / "config" / "users.json"
USERS_EXAMPLE_FILE = Path(__file__).resolve().parent.parent / "config" / "users.example.json"
AUDIT_FILE = Path(__file__).resolve().parent.parent / "data" / "logs" / "audit.jsonl"

# ---- 用户数据库加密 ----
def _get_encrypt_key() -> bytes:
    from app.config import settings
    raw = settings.user_db_key
    if not raw:
        raw = "kb_system_default_fallback_key"
    return hashlib.sha256(raw.encode()).digest()


def _encrypt(plaintext: str) -> str:
    key = _get_encrypt_key()
    salt = _os.urandom(16)
    plain_bytes = plaintext.encode("utf-8")
    ks = b""
    c = 0
    while len(ks) < len(plain_bytes):
        ks += hashlib.sha256(key + salt + c.to_bytes(4, "big")).digest()
        c += 1
    encrypted = bytes(a ^ b for a, b in zip(plain_bytes, ks[:len(plain_bytes)]))
    return base64.b64encode(salt + encrypted).decode()


def _decrypt(ciphertext: str) -> str:
    key = _get_encrypt_key()
    raw = base64.b64decode(ciphertext)
    salt, data = raw[:16], raw[16:]
    ks = b""
    c = 0
    while len(ks) < len(data):
        ks += hashlib.sha256(key + salt + c.to_bytes(4, "big")).digest()
        c += 1
    decrypted = bytes(a ^ b for a, b in zip(data, ks[:len(data)]))
    return decrypted.decode("utf-8")

# 角色信息：标签名 + 默认人数配额（实际配额从 users.json roles 中读取）
ROLE_INFO = {
    "management":    {"label": "管理层",   "max": 2},
    "hr":            {"label": "人事行政", "max": 1},
    "manager":       {"label": "业务经理", "max": 1},
    "sales":         {"label": "业务员",   "max": 4},
    "telemarketing": {"label": "电销部",   "max": 4},
}


def get_role_max(role: str) -> int:
    """从 users.json 读取角色人数上限（配置文件可调），fallback 到 ROLE_INFO。"""
    try:
        data = _load_users()
        return data.get("roles", {}).get(role, {}).get("max", ROLE_INFO[role]["max"])
    except Exception:
        return ROLE_INFO.get(role, {}).get("max", 999)

# 各角色的默认 can_see 权限
ROLE_CAN_SEE = {
    "management":    ["shared", "reports", "progress", "salary"],
    "hr":            ["shared", "salary"],
    "manager":       ["shared", "reports", "progress"],
    "sales":         ["shared", "reports", "progress"],
    "telemarketing": ["shared", "reports", "progress"],
}

# 内存中的会话表: {token: {"username": str, "role": str, "expires": float}}
_sessions: dict[str, dict] = {}


def _load_users() -> dict:
    if not USERS_FILE.exists():
        if USERS_EXAMPLE_FILE.exists():
            with open(USERS_EXAMPLE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"users": [], "roles": {}}
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _save_users(data)
        return data

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {"users": [], "roles": {}}
    # 先尝试直接解析 JSON（兼容旧版明文），失败再解密
    try:
        data = json.loads(raw)
        # 自动迁移：如果是明文 JSON，立即加密保存
        _save_users(data)
        return data
    except json.JSONDecodeError:
        return json.loads(_decrypt(raw))


def _save_users(data: dict):
    plain = json.dumps(data, ensure_ascii=False, indent=2)
    encrypted = _encrypt(plain)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        f.write(encrypted)


def get_user_by_username(username: str) -> Optional[dict]:
    """根据用户名查找用户。"""
    data = _load_users()
    for u in data["users"]:
        if u["username"] == username:
            return u
    return None


def verify_password(username: str, password: str) -> Optional[dict]:
    """验证用户名密码，成功返回用户信息。"""
    user = get_user_by_username(username)
    if user and user["password"] == password:
        return user
    return None


def create_session(user: dict) -> str:
    """创建会话，返回 token。"""
    token = hashlib.sha256(f"{user['username']}{time.time()}".encode()).hexdigest()[:32]
    _sessions[token] = {
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "can_see": user["can_see"],
        "expires": time.time() + 86400,  # 24小时过期
    }
    return token


def validate_token(request: Request) -> dict:
    """
    从请求头中验证 token，返回用户信息。
    公开接口（登录/health/静态文件）跳过校验。
    """
    path = request.url.path

    # 无需登录的路径
    public_paths = ["/login", "/health", "/api/v1/login", "/favicon.ico"]
    if path in public_paths or path.startswith("/static") or path == "/":
        return None  # None 表示公开访问

    # API 调用需要 token
    if path.startswith("/api/"):
        token = request.headers.get("X-Auth-Token", "")
        if not token:
            token = request.cookies.get("auth_token", "")

        if token and token in _sessions:
            session = _sessions[token]
            if session["expires"] > time.time():
                return session
            else:
                del _sessions[token]

        raise HTTPException(status_code=401, detail="请先登录")

    return None  # 静态文件等


def check_token(request: Request) -> Optional[dict]:
    """静默校验 token，失败返回 None（不抛异常），供文件下载等场景使用。"""
    token = request.headers.get("X-Auth-Token", "")
    if not token:
        token = request.cookies.get("auth_token", "")
    if not token:
        token = request.query_params.get("token", "")
    if token and token in _sessions:
        session = _sessions[token]
        if session["expires"] > time.time():
            return session
        else:
            del _sessions[token]
    return None


def can_access(user: dict, doc_type: str) -> bool:
    """检查用户是否有权访问某类文档（匹配 can_see 列表 或 用户角色名）。"""
    if not user:
        return doc_type == "shared"
    if doc_type in user.get("can_see", []):
        return True
    # 也支持按角色名匹配（如 access_roles="management" → role="management" 可看）
    if doc_type == user.get("role", ""):
        return True
    return False


def can_upload_to(user: dict, doc_type: str) -> bool:
    """检查用户是否可以上传到某类文档。共享库所有人都能上传，内部文档按 access_roles 细分（路由层校验具体标签）。"""
    if doc_type == "shared":
        return True
    # 管理层和经理可上传内部文档，其他角色也可上传（具体 access_roles 标签在路由中校验）
    if user["role"] in ("management", "manager", "hr", "sales", "telemarketing"):
        return True
    return doc_type in user.get("can_see", [])


def list_all_users() -> list:
    """列出所有用户（不含密码）。"""
    data = _load_users()
    return [{"username": u["username"], "name": u["name"], "role": u["role"]}
            for u in data["users"]]


def get_all_users_full() -> list:
    """返回所有用户完整信息（含密码），仅供管理层二次验证后查看。"""
    data = _load_users()
    return data["users"]


def get_role_counts() -> dict:
    """返回各角色当前人数."""
    data = _load_users()
    counts = {r: 0 for r in ROLE_INFO}
    for u in data["users"]:
        r = u.get("role", "")
        if r in counts:
            counts[r] += 1
    return counts


def get_role_label(role: str) -> str:
    """返回角色中文标签."""
    return ROLE_INFO.get(role, {}).get("label", role)


def update_role_quotas(quotas: dict) -> dict:
    """
    管理端更新角色人数上限。
    quotas 格式: {"management": 3, "hr": 2, ...}
    返回更新后的 role_info。
    """
    data = _load_users()
    for role, max_val in quotas.items():
        if role not in data.get("roles", {}):
            raise ValueError(f"无效角色: {role}")
        if not isinstance(max_val, int) or max_val < 1:
            raise ValueError(f"「{get_role_label(role)}」上限必须为正整数")
        current = sum(1 for u in data["users"] if u.get("role") == role)
        if max_val < current:
            raise ValueError(f"「{get_role_label(role)}」上限({max_val})不能低于当前人数({current})")
        data["roles"][role]["max"] = max_val
    _save_users(data)
    logger.info("角色配额已更新: %s", quotas)
    result = {}
    for r, info in data["roles"].items():
        result[r] = {"label": info["label"], "max": info.get("max", 999),
                      "current": sum(1 for u in data["users"] if u.get("role") == r)}
    return result


def add_user(name: str, role: str, password: str = None) -> dict:
    """
    管理端添加用户。校验角色配额，写入 users.json。
    返回完整用户信息（含自动生成的 username 和密码）。
    """
    if role not in ROLE_INFO:
        raise ValueError(f"无效角色: {role}")

    data = _load_users()
    counts = {r: 0 for r in ROLE_INFO}
    for u in data["users"]:
        r = u.get("role", "")
        if r in counts:
            counts[r] += 1

    max_quota = get_role_max(role)
    if counts[role] >= max_quota:
        raise ValueError(f"「{get_role_label(role)}」已达人数上限（{max_quota}人）")

    # 生成用户名: 角色前缀 + 递增编号
    existing = [u["username"] for u in data["users"]]
    counter = counts[role] + 1
    while True:
        username = f"{role}{counter:02d}"
        if username not in existing:
            break
        counter += 1

    if password is None:
        password = secrets.token_hex(6)

    can_see = ROLE_CAN_SEE.get(role, ["shared"])

    new_user = {
        "username": username,
        "password": password,
        "name": f"{ROLE_INFO[role]['label']}-{name}",
        "role": role,
        "can_see": can_see,
    }
    data["users"].append(new_user)
    _save_users(data)
    logger.info("新增用户: %s (%s)", username, new_user["name"])
    return new_user


def edit_user(username: str, name: str = None, password: str = None,
              role: str = None, can_see: list = None, new_username: str = None) -> dict:
    """管理端编辑用户（账号、姓名、密码、角色、权限）。返回更新后的用户信息。"""
    data = _load_users()
    for u in data["users"]:
        if u["username"] == username:
            if new_username is not None and new_username != username:
                if any(x["username"] == new_username for x in data["users"]):
                    raise ValueError("账号已存在")
                u["username"] = new_username
            if name is not None:
                if not name or len(name.strip()) < 1:
                    raise ValueError("姓名不能为空")
                role_label = get_role_label(u["role"] if role is None else role)
                u["name"] = f"{role_label}-{name.strip()}"
            if password is not None:
                if len(password) < 6:
                    raise ValueError("密码至少6位")
                u["password"] = password
            if role is not None:
                if role not in ROLE_INFO:
                    raise ValueError(f"无效角色: {role}")
                u["role"] = role
            if can_see is not None:
                u["can_see"] = can_see
            _save_users(data)
            logger.info("管理员编辑用户 %s -> %s", username, u["username"])
            # 清除该用户的所有会话
            expired = [t for t, s in _sessions.items() if s.get("username") in (username, u["username"])]
            for t in expired:
                del _sessions[t]
            return u
    raise ValueError("用户不存在")


def delete_user(username: str) -> dict:
    """管理端删除用户。返回被删除用户的基本信息。"""
    data = _load_users()
    for i, u in enumerate(data["users"]):
        if u["username"] == username:
            deleted = data["users"].pop(i)
            _save_users(data)
            logger.info("管理员删除用户: %s (%s)", username, deleted["name"])
            expired = [t for t, s in _sessions.items() if s.get("username") == username]
            for t in expired:
                del _sessions[t]
            return {"username": deleted["username"], "name": deleted["name"], "role": deleted["role"]}
    raise ValueError("用户不存在")


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """修改用户密码，需验证旧密码。"""
    if len(new_password) < 6:
        raise ValueError("新密码至少6位")
    data = _load_users()
    for u in data["users"]:
        if u["username"] == username:
            if u["password"] != old_password:
                raise ValueError("旧密码不正确")
            u["password"] = new_password
            _save_users(data)
            logger.info("用户 %s 修改了密码", username)
            # 清除该用户所有旧会话，强制重新登录
            expired = [t for t, s in _sessions.items() if s.get("username") == username]
            for t in expired:
                del _sessions[t]
            return True
    raise ValueError("用户不存在")


def change_name(username: str, new_name: str) -> dict:
    """修改用户显示名称。只接受个人姓名，角色标签由系统自动拼接。"""
    if not new_name or len(new_name.strip()) < 1:
        raise ValueError("姓名不能为空")
    personal_name = new_name.strip()
    data = _load_users()
    for u in data["users"]:
        if u["username"] == username:
            role_label = get_role_label(u["role"])
            full_name = f"{role_label}-{personal_name}"
            old_name = u["name"]
            u["name"] = full_name
            _save_users(data)
            logger.info("用户 %s 改名: %s -> %s", username, old_name, full_name)
            for s in _sessions.values():
                if s.get("username") == username:
                    s["name"] = full_name
            return {"username": username, "name": full_name, "role": u["role"]}
    raise ValueError("用户不存在")


def audit_log(user_name: str, role: str, action: str, file_name: str = "", ip: str = ""):
    """写入审计日志（JSONL 追加）。"""
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user": user_name,
        "role": role,
        "action": action,
        "file": file_name,
        "ip": ip,
    }
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_audit_log(limit: int = 100) -> list:
    """读取最近 N 条审计日志。"""
    if not AUDIT_FILE.exists():
        return []
    lines = []
    with open(AUDIT_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return lines[-limit:][::-1]  # 最新在前
