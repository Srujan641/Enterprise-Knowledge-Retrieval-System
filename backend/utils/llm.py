import json
from typing import Optional

import openai
from langchain_openai import ChatOpenAI

from utils import budget
from utils.config import settings

# All three expose an OpenAI-compatible API. Order sets the auto-detect priority.
PROVIDERS = {
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "https://api.groq.com/openai/v1"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", None),
    "xai": ("XAI_API_KEY", "XAI_MODEL", "https://api.x.ai/v1"),
}

# USD per 1M tokens (input, output). A model missing here is refused, so spend is always known.
PRICES = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-6-luna": (0.10, 0.50),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "grok-4.3": (1.25, 2.50),
}

MAX_OUTPUT_TOKENS = 2000


class LLMNotConfigured(RuntimeError):
    pass


class BudgetedChatOpenAI(ChatOpenAI):
    """Charges every model call against the spending cap before it is sent."""

    input_price: float
    output_price: float

    def _worst_case_cost(self, messages, kwargs) -> float:
        characters = sum(len(str(m.content)) + len(str(getattr(m, "tool_calls", ""))) for m in messages)
        characters += len(json.dumps(kwargs.get("tools", []), default=str))
        # Real text averages about 4 characters per token, so 2 safely overestimates.
        input_tokens = characters // 2 + 200
        return (input_tokens * self.input_price + MAX_OUTPUT_TOKENS * self.output_price) / 1_000_000

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reserved = self._worst_case_cost(messages, kwargs)
        budget.reserve(reserved)
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except openai.APIStatusError as e:
            # A rejected request isn't billed. Anything else keeps its reservation, to stay safe.
            if e.status_code < 500:
                budget.settle(reserved, 0)
            raise

        usage = result.generations[0].message.usage_metadata
        if usage:
            actual = (usage["input_tokens"] * self.input_price + usage["output_tokens"] * self.output_price) / 1_000_000
            budget.settle(reserved, actual)
        return result

    async def _agenerate(self, *args, **kwargs):
        raise NotImplementedError("Async calls would bypass the spending cap.")


def _key(provider: str) -> str:
    return getattr(settings, PROVIDERS[provider][0]).strip()


def active_provider() -> Optional[str]:
    forced = settings.LLM_PROVIDER.strip().lower()
    if forced:
        return forced if forced in PROVIDERS and _key(forced) else None
    return next((name for name in PROVIDERS if _key(name)), None)


def get_llm() -> ChatOpenAI:
    provider = active_provider()
    if provider is None:
        raise LLMNotConfigured(
            "No LLM is configured. Add GROQ_API_KEY, OPENAI_API_KEY or XAI_API_KEY to .env "
            "and restart the backend."
        )

    _, model_setting, base_url = PROVIDERS[provider]
    model = getattr(settings, model_setting)
    if model not in PRICES:
        raise LLMNotConfigured(f"No price is known for model '{model}', so its cost can't be capped.")

    options = {}
    if provider == "openai":
        # Reasoning counts as output, so keep it short to protect cost and leave room for the answer.
        options["reasoning_effort"] = "low"
    else:
        options["temperature"] = 0
    return BudgetedChatOpenAI(
        model=model,
        api_key=_key(provider),
        base_url=base_url,
        timeout=60,
        max_retries=0,  # A retried call could be billed without being counted.
        max_tokens=MAX_OUTPUT_TOKENS,
        disable_streaming=True,
        input_price=PRICES[model][0],
        output_price=PRICES[model][1],
        **options,
    )
