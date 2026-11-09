import io
from pathlib import Path

import docx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agents.rag_agent as rag_agent
import main
from api import routes
from retrieval.vector_store import vector_store_manager as store
from utils import llm
from utils.config import settings

client = TestClient(main.app)
PASSWORDS = {"staff": "s3", "hr": "h3", "admin": "a3"}


def auth(username):
    response = client.post("/api/auth/login", json={"username": username, "password": PASSWORDS[username]})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def upload(name, body, user="admin", role=None):
    data = {"role": role} if role else {}
    return client.post("/api/ingest", files={"file": (name, body)}, data=data, headers=auth(user))


def retrieved_files(query, role):
    return {doc.metadata["source"] for doc in store.search(query, role=role)}


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def stub_answer(monkeypatch):
    calls = []

    def answer(query, role="general", history=None):
        calls.append((query, role))
        docs = store.search(query, role=role)
        return {"answer": "stub", "sources": sorted({d.metadata["source"] for d in docs})}

    monkeypatch.setattr(routes, "get_answer", answer)
    return calls


def test_requires_login():
    assert client.post("/api/query", json={"query": "hi"}).status_code == 401


def test_rejects_bad_password():
    assert client.post("/api/auth/login", json={"username": "hr", "password": "no"}).status_code == 401


def test_role_comes_from_token_not_request(stub_answer):
    assert upload("pay.txt", b"The executive salary is 900000 dollars.", role="HR").status_code == 200
    body = {"query": "What is the executive salary?", "role": "HR"}
    response = client.post("/api/query", json=body, headers=auth("staff"))
    assert "pay.txt" not in response.json()["sources"]
    assert stub_answer[-1][1] == "general"
    assert "pay.txt" in retrieved_files("executive salary", "HR")
    assert "pay.txt" in retrieved_files("executive salary", "Admin")


def test_user_cannot_publish_to_another_role():
    assert upload("x.txt", b"hello", user="staff", role="HR").status_code == 403


def test_path_traversal_stays_in_uploads():
    assert upload("../../escaped.txt", b"Traversal content here.").status_code == 200
    assert not (settings.data_path.parent / "escaped.txt").exists()
    assert (settings.data_path / "uploads" / "Admin" / "escaped.txt").exists()


def test_rejected_type_is_not_saved():
    assert upload("bad.exe", b"x").status_code == 400
    assert not list(settings.data_path.rglob("bad.exe"))


def test_docx_ingests():
    buffer = io.BytesIO()
    document = docx.Document()
    document.add_paragraph("The cafeteria opens at nine.")
    document.save(buffer)
    response = upload("cafe.docx", buffer.getvalue(), role="general")
    assert response.status_code == 200 and response.json()["chunks"] == 1


def test_empty_file_is_reported():
    assert upload("blank.txt", b"   \n ").status_code == 422


def test_duplicate_and_new_version():
    upload("dup.txt", b"Version one of the travel rules.", role="general")
    count = store.vector_store.index.ntotal
    assert upload("dup.txt", b"Version one of the travel rules.", role="general").json()["duplicate"]
    assert store.vector_store.index.ntotal == count
    upload("dup.txt", b"Version two of the travel rules.", role="general")
    assert store.vector_store.index.ntotal == count
    contents = [d.page_content for d in store.search("travel rules", role="general")]
    assert "Version two of the travel rules." in contents
    assert "Version one of the travel rules." not in contents


def test_other_roles_do_not_crowd_out_readable_documents():
    for i in range(20):
        upload(f"hr{i}.txt", f"Vacation policy: employees receive vacation days, version {i}.".encode(), role="HR")
    upload("vacation.txt", b"Vacation policy for all staff is in the general summary.", role="general")
    assert "vacation.txt" in retrieved_files("vacation policy days for employees", "general")


def test_cache_hits_and_clears_on_upload(stub_answer):
    headers = auth("staff")
    upload("gym.txt", b"The gym closes at eight.", role="general")
    query = {"query": "When does the gym close?"}
    assert client.post("/api/query", json=query, headers=headers).json()["cached"] is False
    assert client.post("/api/query", json=query, headers=headers).json()["cached"] is True
    upload("other.txt", b"Unrelated new document.", role="general")
    assert client.post("/api/query", json=query, headers=headers).json()["cached"] is False


def test_missing_key_gives_clear_error():
    response = client.post("/api/query", json={"query": "A question with no key?"}, headers=auth("staff"))
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_agent_searches_and_cites(monkeypatch):
    upload("parking.txt", b"The parking fee is 12 dollars per day.", role="general")
    replies = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_knowledge_base", "args": {"search_query": "parking fee"}, "id": "1"}],
            ),
            AIMessage(content="The fee is 12 dollars (parking.txt)."),
        ]
    )
    monkeypatch.setattr(rag_agent, "get_llm", lambda: FakeToolModel(messages=replies))
    response = client.post("/api/query", json={"query": "Parking cost?"}, headers=auth("staff"))
    assert response.status_code == 200
    assert response.json()["answer"] == "The fee is 12 dollars (parking.txt)."
    assert response.json()["sources"] == ["parking.txt"]


@pytest.mark.parametrize(
    "key,expected",
    [("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"), ("XAI_API_KEY", "xai")],
)
def test_provider_follows_available_key(monkeypatch, key, expected):
    monkeypatch.setattr(llm.settings, key, "k")
    assert llm.active_provider() == expected
    assert llm.get_llm().model_name == getattr(llm.settings, llm.PROVIDERS[expected][1])


def test_upload_folder_has_no_temp_leftovers():
    assert not [p for p in Path(settings.data_path, "uploads").rglob("tmp*")]
