"""Offline checks for RcForecastBot's changes to the template (no network, no keys).

Run from the repo root:  python tests/check_rc_bot.py
Covers model routing by environment, the binary trimmed mean, and research that
survives one failing source. Fake key values only; nothing is sent anywhere.
"""

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "PERPLEXITY_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
    "ASKNEWS_API_KEY",
    "EXA_API_KEY",
    "RESEARCHER",
    "FORECASTER_MODEL",
    "PARSER_MODEL",
)


@contextmanager
def env(**values: str):
    saved = {k: os.environ.pop(k) for k in KEYS if k in os.environ}
    os.environ.update(values)
    try:
        yield
    finally:
        for k in KEYS:
            os.environ.pop(k, None)
        os.environ.update(saved)


with env():
    import main  # noqa: E402  (import after clearing keys: dotenv must not leak real ones)
    from forecasting_tools import BinaryQuestion, GeneralLlm  # noqa: E402


def model_of(llm) -> str:
    return llm.model if isinstance(llm, GeneralLlm) else str(llm)


def check_routing() -> None:
    with env(ANTHROPIC_API_KEY="x"):
        d = main.RcForecastBot._llm_config_defaults()
        assert model_of(d["default"]) == "anthropic/claude-sonnet-5", d["default"]
        assert d["default"].litellm_kwargs["temperature"] is None
        assert model_of(d["parser"]) == "anthropic/claude-haiku-4-5"
        assert d["parser"].litellm_kwargs["temperature"] == 0
        # no search provider: the forecaster researches from its own knowledge
        assert model_of(d["researcher"]) == "anthropic/claude-sonnet-5"
        assert d["researcher_2"] is None

    with env(OPENROUTER_API_KEY="x", ASKNEWS_CLIENT_ID="x", ASKNEWS_SECRET="y"):
        d = main.RcForecastBot._llm_config_defaults()
        assert model_of(d["default"]) == "openrouter/anthropic/claude-sonnet-5"
        assert model_of(d["parser"]) == "openrouter/anthropic/claude-haiku-4.5"
        assert d["researcher"] == main.ASKNEWS_RESEARCHER == "asknews/latest"
        # Metaculus's OpenRouter credits cover Anthropic but not Perplexity
        assert model_of(d["researcher_2"]) == "openrouter/anthropic/claude-sonnet-5:online"
        assert d["researcher_2"].litellm_kwargs["temperature"] is None

    with env(ANTHROPIC_API_KEY="x", PERPLEXITY_API_KEY="x", FORECASTER_MODEL="anthropic/claude-opus-5"):
        d = main.RcForecastBot._llm_config_defaults()
        assert model_of(d["default"]) == "anthropic/claude-opus-5"
        assert model_of(d["researcher"]) == "perplexity/sonar-pro"
        assert d["researcher_2"] is None

    with env(ANTHROPIC_API_KEY="x", PARSER_MODEL="anthropic/claude-sonnet-5"):
        d = main.RcForecastBot._llm_config_defaults()
        # a model that rejects sampling parameters gets none
        assert d["parser"].litellm_kwargs["temperature"] is None

    with env():  # no LLM key at all: forecasting-tools' own defaults stay in charge
        d = main.RcForecastBot._llm_config_defaults()
        assert d["researcher_2"] is None
        assert "claude-sonnet-5" not in model_of(d["default"])
    print("routing: ok")


def binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it happen?",
        page_url="https://example.invalid/q/1",
        resolution_criteria="Resolves Yes if it happens.",
        fine_print="",
        background_info="",
    )


def check_aggregation() -> None:
    with env(ANTHROPIC_API_KEY="x"):
        bot = main.RcForecastBot(publish_reports_to_metaculus=False)
    q = binary_question()
    six = [0.10, 0.30, 0.32, 0.34, 0.36, 0.90]
    got = asyncio.run(bot._aggregate_predictions(six, q))
    assert abs(got - 0.33) < 1e-12, got  # mean of the middle four
    three = [0.2, 0.5, 0.9]
    got = asyncio.run(bot._aggregate_predictions(three, q))
    assert abs(got - 0.5) < 1e-12, got  # fewer than four: the package median
    print("aggregation: ok")


def check_research_survives_one_failure() -> None:
    with env(ANTHROPIC_API_KEY="x", PERPLEXITY_API_KEY="x", ASKNEWS_CLIENT_ID="x", ASKNEWS_SECRET="y"):
        bot = main.RcForecastBot(publish_reports_to_metaculus=False)

    async def fake(source, prompt, question):
        if source == main.ASKNEWS_RESEARCHER:
            raise RuntimeError("asknews down")
        assert "Include dates for every fact" in prompt
        assert question.question_text == "Will it happen?"
        return "Sonar says: nothing has happened yet (2026-09-20)."

    bot._run_one_research_source = fake  # type: ignore[method-assign]
    research = asyncio.run(bot.run_research(binary_question()))
    assert "Research from perplexity/sonar-pro" in research, research
    assert "asknews" not in research.lower(), research

    async def both(source, prompt, question):
        return f"notes from {bot._source_name(source)}"

    bot._run_one_research_source = both  # type: ignore[method-assign]
    research = asyncio.run(bot.run_research(binary_question()))
    assert research.count("## Research from") == 2, research
    print("research: ok")


def check_asknews_uses_one_call() -> None:
    """Latest-news only: one AskNews search per research (the free tier is 1,000 a month)."""
    import asknews_sdk

    calls: list[dict] = []

    class FakeResponse:
        as_dicts: list = []

    class FakeNews:
        async def search_news(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

    class FakeSDK:
        def __init__(self, **kwargs):
            self.news = FakeNews()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    real = asknews_sdk.AsyncAskNewsSDK
    asknews_sdk.AsyncAskNewsSDK = FakeSDK  # type: ignore[misc]
    try:
        with env(ASKNEWS_CLIENT_ID="x", ASKNEWS_SECRET="y"):
            text = asyncio.run(
                main.AskNewsLatestSearcher().get_formatted_news_async("Will it happen?")
            )
    finally:
        asknews_sdk.AsyncAskNewsSDK = real  # type: ignore[misc]
    assert len(calls) == 1, calls
    assert calls[0]["strategy"] == "latest news", calls
    assert calls[0]["query"] == "Will it happen?"
    assert "No articles were found" in text
    print("asknews: ok")


if __name__ == "__main__":
    check_routing()
    check_aggregation()
    check_research_survives_one_failure()
    check_asknews_uses_one_call()
    print("all checks passed")
