from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

import main
from utils import auth

client = TestClient(main.app)
SECRET = "test-secret-test-secret-test-secret-1234"


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def claims(**overrides):
    return {"sub": "x", "role": "Admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1), **overrides}


def login(username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


@pytest.mark.parametrize(
    "token",
    [
        jwt.encode(claims(), "wrong-secret-wrong-secret-wrong-secret-12", algorithm="HS256"),
        jwt.encode(claims(), None, algorithm="none"),
        jwt.encode(claims(exp=datetime.now(timezone.utc) - timedelta(hours=1)), SECRET, algorithm="HS256"),
        "not-a-token",
    ],
)
def test_forged_or_expired_tokens_are_rejected(token):
    assert client.get("/api/usage", headers=bearer(token)).status_code == 401


@pytest.mark.parametrize(
    "method,path",
    [("post", "/api/query"), ("post", "/api/ingest"), ("get", "/api/usage")],
)
def test_private_endpoints_require_login(method, path):
    assert client.request(method, path, json={"query": "x"}).status_code in (401, 411)


def test_ingest_without_token_is_refused():
    response = client.post("/api/ingest", files={"file": ("a.txt", b"hello")})
    assert response.status_code == 401


def test_api_docs_are_off_by_default():
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_health_reveals_nothing_sensitive():
    assert client.get("/api/health").json() == {"status": "healthy"}


def test_usage_is_admin_only():
    staff = login("staff", "s3").json()["token"]
    admin = login("admin", "a3").json()["token"]
    assert client.get("/api/usage", headers=bearer(staff)).status_code == 403
    assert client.get("/api/usage", headers=bearer(admin)).status_code == 200


def test_oversized_upload_is_refused_before_parsing():
    response = client.post("/api/ingest", content=b"x", headers={"content-length": str(10**10)})
    assert response.status_code == 413


def test_repeated_failed_logins_are_blocked():
    auth._failed_logins.clear()
    try:
        for _ in range(auth.MAX_FAILED_LOGINS):
            assert login("admin", "wrong").status_code == 401
        assert login("admin", "a3").status_code == 429
    finally:
        auth._failed_logins.clear()
