import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

import agents.rag_agent as rag_agent
import main
from utils import budget, llm
from utils.config import settings


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(llm.settings, "OPENAI_API_KEY", "k")
    return llm.get_llm()


def fake_result(input_tokens=1000, output_tokens=500):
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok", usage_metadata=usage))])


def test_default_is_cheapest_model(model):
    assert model.model_name == "gpt-5-nano"


def test_settles_to_actual_cost(model, monkeypatch):
    monkeypatch.setattr(ChatOpenAI, "_generate", lambda self, *a, **k: fake_result())
    before = budget.spent()
    model._generate([HumanMessage(content="hi")])
    assert budget.spent() - before == pytest.approx((1000 * 0.05 + 500 * 0.40) / 1_000_000)


def test_call_is_never_sent_when_it_could_pass_the_cap(model, monkeypatch):
    def must_not_run(self, *args, **kwargs):
        raise AssertionError("request was sent")

    monkeypatch.setattr(ChatOpenAI, "_generate", must_not_run)
    monkeypatch.setattr(settings, "LLM_BUDGET_USD", budget.spent() + 0.0000001)
    before = budget.spent()
    with pytest.raises(budget.BudgetExceeded):
        model._generate([HumanMessage(content="hi")])
    assert budget.spent() == before


def test_unpriced_model_is_refused(monkeypatch):
    monkeypatch.setattr(llm.settings, "OPENAI_API_KEY", "k")
    monkeypatch.setattr(llm.settings, "OPENAI_MODEL", "gpt-6-astra")
    with pytest.raises(llm.LLMNotConfigured):
        llm.get_llm()


def test_api_reports_limit_clearly(monkeypatch):
    def over_budget():
        raise budget.BudgetExceeded("limit reached")

    monkeypatch.setattr(rag_agent, "get_llm", over_budget)
    client = TestClient(main.app)
    token = client.post("/api/auth/login", json={"username": "staff", "password": "s3"}).json()["token"]
    response = client.post(
        "/api/query",
        json={"query": "Anything at all?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 429
    assert response.json()["detail"] == "limit reached"
