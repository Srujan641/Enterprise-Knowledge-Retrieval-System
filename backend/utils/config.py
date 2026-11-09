from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
    )

    LLM_PROVIDER: str = ""
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    XAI_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    OPENAI_MODEL: str = "gpt-5-nano"
    XAI_MODEL: str = "grok-4.3"
    LLM_BUDGET_USD: float = 0.50
    LLM_LEDGER_PATH: str = "data/llm_spend.sqlite"

    AUTH_USERS: str = ""
    JWT_SECRET: str = ""

    HUGGINGFACE_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    FAISS_INDEX_PATH: str = "vector_store/faiss_index"
    DATA_DIR: str = "data"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_UPLOAD_MB: int = 25
    ENABLE_API_DOCS: bool = False
    CORS_ORIGINS: str = "http://localhost:5173"

    @property
    def data_path(self) -> Path:
        return BACKEND_DIR / self.DATA_DIR

    @property
    def ledger_path(self) -> Path:
        return BACKEND_DIR / self.LLM_LEDGER_PATH

    @property
    def index_path(self) -> Path:
        return BACKEND_DIR / self.FAISS_INDEX_PATH


settings = Settings()
