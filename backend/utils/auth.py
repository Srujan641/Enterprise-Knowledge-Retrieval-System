import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from utils.config import settings

ROLES = ("general", "HR", "Admin")
TOKEN_HOURS = 12

# Without a configured secret, tokens simply stop working on restart.
_secret = settings.JWT_SECRET or secrets.token_urlsafe(48)
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class User:
    username: str
    role: str


def _load_users() -> Dict[str, Tuple[str, str]]:
    users = {}
    for entry in filter(None, (e.strip() for e in settings.AUTH_USERS.split(","))):
        try:
            username, rest = entry.split(":", 1)
            password, role = rest.rsplit(":", 1)
        except ValueError:
            raise ValueError("AUTH_USERS entries must look like username:password:role")
        if role not in ROLES:
            raise ValueError(f"Unknown role '{role}' in AUTH_USERS. Use one of {ROLES}.")
        users[username] = (password, role)
    return users


_users = _load_users()

MAX_FAILED_LOGINS = 10
LOGIN_WINDOW_SECONDS = 300
_failed_logins: Dict[str, List[float]] = {}
_failures_lock = threading.Lock()


def ensure_login_allowed(client: str) -> None:
    cutoff = time.monotonic() - LOGIN_WINDOW_SECONDS
    with _failures_lock:
        recent = [t for t in _failed_logins.get(client, []) if t > cutoff]
        if recent:
            _failed_logins[client] = recent
        else:
            _failed_logins.pop(client, None)
    if len(recent) >= MAX_FAILED_LOGINS:
        raise HTTPException(status_code=429, detail="Too many failed logins. Try again in a few minutes.")


def record_failed_login(client: str) -> None:
    with _failures_lock:
        _failed_logins.setdefault(client, []).append(time.monotonic())


def can_read(user_role: str, document_role: str) -> bool:
    return user_role == "Admin" or document_role in ("general", user_role)


def authenticate(username: str, password: str) -> Optional[User]:
    expected, role = _users.get(username, (secrets.token_hex(16), ""))
    if not hmac.compare_digest(expected.encode(), password.encode()) or not role:
        return None
    return User(username, role)


def create_token(user: User) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": user.username, "role": user.role, "exp": expires}, _secret, algorithm="HS256")


def current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Login required")
    try:
        claims = jwt.decode(credentials.credentials, _secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    return User(claims["sub"], claims["role"])
