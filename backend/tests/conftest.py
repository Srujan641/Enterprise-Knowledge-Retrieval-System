import os
import tempfile

_root = tempfile.mkdtemp(prefix="rag-tests-")
os.environ.update(
    DATA_DIR=os.path.join(_root, "data"),
    FAISS_INDEX_PATH=os.path.join(_root, "index"),
    AUTH_USERS="staff:s3:general,hr:h3:HR,admin:a3:Admin",
    JWT_SECRET="test-secret-test-secret-test-secret-1234",
    LLM_LEDGER_PATH=os.path.join(_root, "spend.sqlite"),
    LLM_PROVIDER="",
    GROQ_API_KEY="",
    OPENAI_API_KEY="",
    XAI_API_KEY="",
)
