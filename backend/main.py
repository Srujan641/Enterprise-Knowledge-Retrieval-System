from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from utils.config import settings

docs_enabled = settings.ENABLE_API_DOCS
app = FastAPI(
    title="Enterprise RAG System",
    docs_url="/docs" if docs_enabled else None,
    redoc_url=None,
    openapi_url="/openapi.json" if docs_enabled else None,
)


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    # Form bodies are parsed before authentication, so size must be checked first.
    if request.method == "POST" and request.url.path == "/api/ingest":
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return JSONResponse({"detail": "Content-Length header is required"}, status_code=411)
        if int(length) > (settings.MAX_UPLOAD_MB + 1) * 1024 * 1024:
            return JSONResponse({"detail": f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit"}, status_code=413)
    return await call_next(request)


# Added last so that it wraps the size check and error responses stay readable by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
