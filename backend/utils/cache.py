import hashlib
import json
import sqlite3
from contextlib import closing

from utils.config import settings

settings.data_path.mkdir(parents=True, exist_ok=True)
CACHE_FILE = str(settings.data_path / "query_cache.sqlite")


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(CACHE_FILE, timeout=30, isolation_level=None)
    connection.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return connection


def get_cache_key(query: str, role: str) -> str:
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(f"{role}\0{normalized}".encode()).hexdigest()


def get_cached_response(query: str, role: str):
    with closing(_connect()) as connection:
        row = connection.execute("SELECT value FROM answers WHERE key = ?", (get_cache_key(query, role),)).fetchone()
    return json.loads(row[0]) if row else None


def set_cached_response(query: str, role: str, response: dict):
    with closing(_connect()) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO answers (key, value) VALUES (?, ?)",
            (get_cache_key(query, role), json.dumps(response)),
        )


def clear_cache():
    with closing(_connect()) as connection:
        connection.execute("DELETE FROM answers")
