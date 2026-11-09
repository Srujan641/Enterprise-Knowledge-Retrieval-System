import logging
import os
import tempfile
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from agents.rag_agent import get_answer
from ingestion.document_processor import SUPPORTED_EXTENSIONS, process_document
from retrieval.vector_store import vector_store_manager
from utils import budget
from utils.auth import (
    ROLES,
    User,
    authenticate,
    create_token,
    current_user,
    ensure_login_allowed,
    record_failed_login,
)
from utils.budget import BudgetExceeded
from utils.cache import clear_cache, get_cached_response, set_cached_response
from utils.config import settings
from utils.llm import LLMNotConfigured, active_provider

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 1500


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=50)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cached: bool = False


def _clean_filename(raw: Optional[str]) -> str:
    name = Path((raw or "").replace("\\", "/")).name
    ext = Path(name).suffix.lower()
    if not name or name.startswith(".") or ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file. Allowed types: {allowed}")
    return name


def _document_role(requested: Optional[str], user: User) -> str:
    role = requested or user.role
    if role not in ROLES:
        raise HTTPException(status_code=400, detail=f"Unknown role: {role}")
    if user.role != "Admin" and role not in ("general", user.role):
        raise HTTPException(status_code=403, detail=f"You cannot publish documents to the {role} role")
    return role


def _save_upload(upload: UploadFile, destination) -> None:
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    while block := upload.file.read(1024 * 1024):
        written += len(block)
        if written > limit:
            raise HTTPException(status_code=413, detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit")
        destination.write(block)


@router.post("/auth/login")
def login(req: LoginRequest, request: Request):
    client = request.client.host if request.client else "unknown"
    ensure_login_allowed(client)
    user = authenticate(req.username, req.password)
    if user is None:
        record_failed_login(client)
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return {"token": create_token(user), "username": user.username, "role": user.role}


@router.post("/ingest")
def ingest_document(
    file: UploadFile = File(...),
    role: Optional[str] = Form(None),
    user: User = Depends(current_user),
):
    name = _clean_filename(file.filename)
    ext = Path(name).suffix.lower()
    document_role = _document_role(role, user)
    upload_dir = settings.data_path / "uploads" / document_role
    upload_dir.mkdir(parents=True, exist_ok=True)
    doc_path = f"{document_role}/{name}"

    with tempfile.NamedTemporaryFile(dir=upload_dir, suffix=ext, delete=False) as temp:
        temp_path = temp.name
        try:
            _save_upload(file, temp)
        except HTTPException:
            temp.close()
            os.unlink(temp_path)
            raise

    try:
        try:
            chunks = process_document(temp_path, ext, name, doc_path, document_role)
        except Exception:
            logger.exception("Could not parse %s", doc_path)
            raise HTTPException(status_code=400, detail="The file could not be read. It may be corrupt.")
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text found in this file. Scanned PDFs need OCR before upload.",
            )

        if vector_store_manager.is_indexed(doc_path, chunks[0].metadata["doc_hash"]):
            return {"message": "Document already indexed", "chunks": 0, "role": document_role, "duplicate": True}

        vector_store_manager.replace_document(doc_path, chunks)
        os.replace(temp_path, upload_dir / name)
        clear_cache()
        return {"message": "Document ingested successfully", "chunks": len(chunks), "role": document_role}
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/query", response_model=QueryResponse)
def query_system(req: QueryRequest, user: User = Depends(current_user)):
    # A follow-up's answer depends on the conversation, so only standalone questions are cached.
    standalone = not req.history
    cached = get_cached_response(req.query, user.role) if standalone else None
    if cached:
        return QueryResponse(answer=cached["answer"], sources=cached["sources"], cached=True)

    history = [{"role": t.role, "content": t.content[:MAX_HISTORY_CHARS]} for t in req.history[-MAX_HISTORY_TURNS:]]
    try:
        result = get_answer(req.query, role=user.role, history=history)
    except BudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except LLMNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=502, detail="The AI provider request failed. Check your API key and model.")

    if standalone and result["sources"]:
        set_cached_response(req.query, user.role, result)
    return QueryResponse(answer=result["answer"], sources=result["sources"], cached=False)


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.get("/usage")
def usage(user: User = Depends(current_user)):
    if user.role != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return {
        "llm_provider": active_provider(),
        "llm_spent_usd": round(budget.spent(), 6),
        "llm_budget_usd": settings.LLM_BUDGET_USD,
    }
