"""Offline checks for RcForecastBot's changes to the template (no network, no keys).

Run from the repo root:  python tests/check_rc_bot.py
Covers model routing by environment, credit pacing (budget.py and the run loop in
main.main), the binary trimmed mean, and research that survives one failing source.
Fake key values only; nothing is sent anywhere.
"""

import asyncio
import logging
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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
    "SEARCH_MODEL",
    "METACULUS_TOKEN",
    "BUDGET_TIER",
    "TEST_QUESTIONS",
    "PUBLISH",
    "AIB_TOURNAMENT_ID",
    "MINIBENCH_ID",
    "METACULUS_CUP_ID",
    "PREDICTIONS_PER_REPORT",
    "RESEARCH_REPORTS",
    "GITHUB_STEP_SUMMARY",
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
    import budget  # noqa: E402
    import main  # noqa: E402  (import after clearing keys: dotenv must not leak real ones)
    from forecasting_tools import (  # noqa: E402
        BinaryQuestion,
        GeneralLlm,
        MultipleChoiceQuestion,
        NumericQuestion,
    )


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
        # donated credits run the stronger model
        assert model_of(d["default"]) == "openrouter/anthropic/claude-opus-5.5"
        assert d["default"].litellm_kwargs["temperature"] is None
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

    # both keys: Metaculus's donated credits pay before the personal key
    with env(ANTHROPIC_API_KEY="x", OPENROUTER_API_KEY="y", ASKNEWS_API_KEY="z"):
        assert main.llm_route() == "openrouter"
        d = main.RcForecastBot._llm_config_defaults()
        assert model_of(d["default"]) == "openrouter/anthropic/claude-opus-5.5"
        # the personal route (Metaculus Cup) never touches the OpenRouter key
        d = main.build_llms("anthropic")
        assert model_of(d["default"]) == "anthropic/claude-sonnet-5"
        assert model_of(d["parser"]) == "anthropic/claude-haiku-4-5"
        assert d["researcher"] == main.ASKNEWS_RESEARCHER
        assert d["researcher_2"] is None

    # a pacing tier picks the forecaster on the donated credits
    with env(OPENROUTER_API_KEY="x", FORECASTER_MODEL="openrouter/some/other-model"):
        d = main.build_llms("openrouter", budget.TIERS["sonnet-3"])
        assert model_of(d["default"]) == "openrouter/anthropic/claude-sonnet-5"
        # FORECASTER_MODEL applies to a personal key only
        assert model_of(main.build_llms("openrouter")["default"]) == budget.OPUS
    print("routing: ok")


def check_plan() -> None:
    tiers = budget.TIERS
    plan = budget.make_plan
    assert (plan(100).futureeval, plan(100).minibench) == (tiers["sonnet-3"], tiers["opus-3"])
    assert (plan(120).futureeval, plan(120).minibench) == (tiers["opus-3"], tiers["opus-3"])
    assert (plan(150).futureeval, plan(150).minibench) == (tiers["opus-3"], tiers["opus-5"])
    assert (plan(512).futureeval, plan(512).minibench) == (tiers["opus-5"], tiers["opus-5"])
    assert plan(math.inf).futureeval == tiers["opus-5"]  # a key without a limit
    # too little for the horizon: the cheapest tiers, down to the floor
    low = plan(40)
    assert (low.futureeval, low.minibench) == (tiers["sonnet-3"], tiers["sonnet-3"])
    assert "less than" in low.note
    assert plan(1.99).futureeval == plan(1.99).minibench == tiers["pause"]
    # an unreadable balance: cheapest tiers, uncapped
    unknown = plan(None)
    assert unknown.futureeval == tiers["sonnet-3"] and "unreadable" in unknown.note
    # MiniBench (which decides further funding) never runs below FutureEval, and each
    # step down the ladder costs less
    costs = []
    for fe, mb in budget.LADDER:
        assert tiers[mb].cost >= tiers[fe].cost, (fe, mb)
        costs.append(budget.horizon_cost(tiers[fe], tiers[mb]))
    assert costs == sorted(costs, reverse=True), costs

    with env(BUDGET_TIER="opus-5"):
        pinned = plan(3, budget.pinned_tier())
        assert pinned.futureeval == pinned.minibench == tiers["opus-5"]
        assert plan(1, budget.pinned_tier()).futureeval == tiers["pause"]  # floor still holds
    with env(BUDGET_TIER="Pause"):
        assert budget.pinned_tier() == tiers["pause"]
    with env(BUDGET_TIER="opus-9"):
        try:
            budget.pinned_tier()
        except SystemExit as e:
            assert "opus-9" in str(e)
        else:
            raise AssertionError("an unknown BUDGET_TIER must stop the run")

    assert budget.affordable(10, tiers["sonnet-3"]) == 26  # (10 - 2) / 0.30
    assert budget.affordable(1, tiers["sonnet-3"]) == 0
    assert budget.affordable(None, tiers["opus-5"]) is None
    assert budget.affordable(math.inf, tiers["opus-5"]) is None
    assert budget.affordable(500, tiers["pause"]) == 0
    print("plan: ok")


def check_key_status() -> None:
    """Reads limit_remaining and usage; a failure returns None and logs no request details."""
    secret = "sk-or-v1-FAKE-never-logged"
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"limit": 100, "limit_remaining": 87.5, "usage": 12.5}}

    def fake_get(url, headers, timeout):
        calls.append((url, headers))
        return FakeResponse()

    def failing_get(url, headers, timeout):
        raise RuntimeError(f"connection refused while sending {headers}")

    real = budget.requests.get
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    budget.logger.addHandler(handler)
    try:
        budget.requests.get = fake_get  # type: ignore[assignment]
        status = budget.key_status(secret)
        assert calls == [(budget.KEY_URL, {"Authorization": f"Bearer {secret}"})]
        assert budget.remaining_of(status) == 87.5 and budget.usage_of(status) == 12.5
        budget.requests.get = failing_get  # type: ignore[assignment]
        assert budget.key_status(secret) is None
    finally:
        budget.requests.get = real  # type: ignore[assignment]
        budget.logger.removeHandler(handler)
    assert records and all(secret not in r.getMessage() for r in records)
    assert budget.remaining_of({"limit": None, "limit_remaining": None}) == math.inf
    assert budget.remaining_of(None) is None
    print("key status: ok")


NOW = datetime.now(timezone.utc)


def open_question(n: int, hours: float, kind: str = "binary", forecasted: bool = False):
    fields = dict(
        question_text=f"Q{n}?",
        page_url=f"https://example.invalid/q/{n}",
        close_time=NOW + timedelta(hours=hours),
        already_forecasted=forecasted,
    )
    if kind == "mc":
        return MultipleChoiceQuestion(options=["a", "b"], **fields)
    if kind == "numeric":
        return NumericQuestion(
            upper_bound=10, lower_bound=0, open_upper_bound=False, open_lower_bound=False, **fields
        )
    return BinaryQuestion(**fields)


def run_main(mode: str, balances: list, questions: dict, fail_urls=(), **env_values):
    """Runs main.main() offline: fake Metaculus client, fake key balances (one per read),
    fake forecasting. Returns (forecast calls, step summary, exit message or None)."""
    calls: list[tuple] = []
    reads = iter(balances)

    class FakeClient:
        CURRENT_MINIBENCH_ID = "minibench"

        def __init__(self, *args, **kwargs):
            pass

        def get_all_open_questions_from_tournament(self, tournament_id):
            return list(questions.get(tournament_id, []))

    async def fake_forecast(bot, qs, return_exceptions=False):
        calls.append(
            (
                model_of(bot._llms["default"]),
                bot.predictions_per_research_report,
                [q.page_url.rsplit("/", 1)[1] for q in qs],
                bot.skip_previously_forecasted_questions,
            )
        )
        return [RuntimeError("boom") if q.page_url in fail_urls else "report" for q in qs]

    def fake_status(api_key, timeout=15):
        assert api_key == "sk-fake"
        balance = next(reads)
        if balance is None:
            return None
        remaining, usage = balance
        return {"limit": 100.0, "limit_remaining": remaining, "usage": usage}

    saved = (
        main.MetaculusClient,
        main.RcForecastBot.forecast_questions,
        main.RcForecastBot.log_report_summary,
        main.print_run_summary_banner,
        budget.key_status,
        sys.argv,
    )
    summary_path = Path(tempfile.mkdtemp()) / "summary.md"
    exit_message = None
    try:
        main.MetaculusClient = FakeClient  # type: ignore[misc]
        main.RcForecastBot.forecast_questions = fake_forecast  # type: ignore[method-assign]
        main.RcForecastBot.log_report_summary = classmethod(lambda cls, *a, **k: None)  # type: ignore[method-assign]
        main.print_run_summary_banner = lambda *a, **k: None
        budget.key_status = fake_status
        sys.argv = ["main.py", "--mode", mode]
        with env(METACULUS_TOKEN="x", GITHUB_STEP_SUMMARY=str(summary_path), **env_values):
            try:
                main.main()
            except SystemExit as e:
                exit_message = str(e.code)
    finally:
        (
            main.MetaculusClient,
            main.RcForecastBot.forecast_questions,
            main.RcForecastBot.log_report_summary,
            main.print_run_summary_banner,
            budget.key_status,
            sys.argv,
        ) = saved
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    return calls, summary, exit_message


def check_run_loop() -> None:
    opus, sonnet = budget.OPUS, budget.SONNET
    fe_id = main.FALL_2026_FUTUREEVAL_ID

    # $100: FutureEval on sonnet-3, MiniBench on opus-3; soonest-closing first; the
    # already-forecast question skipped; each pass's spend measured
    questions = {
        fe_id: [open_question(1, 2), open_question(2, 1), open_question(3, 0.5, forecasted=True)],
        "minibench": [open_question(4, 50), open_question(5, 40)],
    }
    calls, summary, exited = run_main(
        "tournament",
        [(100.0, 0.0), (99.4, 0.6), (98.0, 2.0)],
        questions,
        OPENROUTER_API_KEY="sk-fake",
    )
    assert exited is None, exited
    assert calls == [
        (sonnet, 3, ["2", "1"], True),
        (opus, 3, ["5", "4"], True),
    ], calls
    assert "| FutureEval | sonnet-3 | 2 | 0 | $0.60 | $0.30 |" in summary, summary
    assert "| MiniBench | opus-3 | 2 | 0 | $1.40 | $0.70 |" in summary, summary
    assert "$100.00 of $100.00" in summary and "Now: $98.00" in summary, summary

    # $5: cheapest tiers, and only what the balance pays for above the $2 floor
    many = {"minibench": [open_question(n, 100 - n) for n in range(10, 40)]}
    calls, _, exited = run_main(
        "tournament", [(5.0, 95.0), (2.0, 98.0)], many, OPENROUTER_API_KEY="sk-fake"
    )
    assert exited is None
    assert len(calls) == 1 and calls[0][:2] == (sonnet, 3), calls
    assert calls[0][2] == [str(n) for n in range(39, 29, -1)], calls  # the 10 closing soonest

    # under the floor: nothing runs
    calls, summary, _ = run_main(
        "tournament", [(1.5, 98.5)], questions, OPENROUTER_API_KEY="sk-fake"
    )
    assert calls == [] and "pause" in summary, (calls, summary)

    # a big balance: top tier, but no more than MAX_QUESTIONS_PER_RUN in one run
    burst = {
        fe_id: [open_question(n, n) for n in range(100, 120)],
        "minibench": [open_question(n, n) for n in range(200, 220)],
    }
    calls, _, _ = run_main(
        "tournament", [(900.0, 0.0), (890.0, 10.0), (880.0, 20.0)], burst,
        OPENROUTER_API_KEY="sk-fake",
    )
    assert [(c[0], c[1], len(c[2])) for c in calls] == [
        (opus, 5, 20),
        (opus, 5, main.MAX_QUESTIONS_PER_RUN - 20),
    ], calls

    # an unreadable balance: cheapest tiers, no credit cap, spend "unknown"
    calls, summary, _ = run_main(
        "tournament", [None, None, None], questions, OPENROUTER_API_KEY="sk-fake"
    )
    assert [c[:2] for c in calls] == [(sonnet, 3), (sonnet, 3)], calls
    assert "unknown" in summary

    # a failed question turns the run red, after the spend is reported
    calls, summary, exited = run_main(
        "tournament",
        [(100.0, 0.0), (99.4, 0.6), (98.0, 2.0)],
        questions,
        fail_urls=("https://example.invalid/q/4",),
        OPENROUTER_API_KEY="sk-fake",
    )
    assert exited and "1 question(s) failed" in exited, exited
    assert "| MiniBench | opus-3 | 1 | 1 |" in summary, summary

    # the smoke test: one question of each type, on MiniBench's tier, re-forecast
    testing = {
        "bot-testing-area": [
            open_question(1, 5),
            open_question(2, 5),
            open_question(3, 5, "mc"),
            open_question(4, 5, "numeric"),
        ]
    }
    calls, _, _ = run_main(
        "test_questions", [(100.0, 0.0), (98.5, 1.5)], testing, OPENROUTER_API_KEY="sk-fake"
    )
    assert calls == [(opus, 3, ["1", "3", "4"], False)], calls
    calls, _, _ = run_main(
        "test_questions", [(100.0, 0.0), (98.5, 1.5)], testing,
        OPENROUTER_API_KEY="sk-fake", BUDGET_TIER="sonnet-3", TEST_QUESTIONS="1",
    )
    assert calls == [(sonnet, 3, ["1"], False)], calls

    # the Metaculus Cup: never on Metaculus's credits
    cup = {main.FALL_2026_METACULUS_CUP_ID: [open_question(7, 30, forecasted=True)]}
    calls, _, exited = run_main("metaculus_cup", [], cup, OPENROUTER_API_KEY="sk-fake")
    assert calls == [] and exited and "ANTHROPIC_API_KEY" in exited, (calls, exited)
    calls, summary, exited = run_main(
        "metaculus_cup", [], cup, OPENROUTER_API_KEY="sk-fake", ANTHROPIC_API_KEY="x"
    )
    assert exited is None and summary == "", (exited, summary)  # no balance read at all
    assert calls == [("anthropic/claude-sonnet-5", 5, ["7"], False)], calls
    print("run loop: ok")


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
    check_plan()
    check_key_status()
    check_run_loop()
    check_aggregation()
    check_research_survives_one_failure()
    check_asknews_uses_one_call()
    print("all checks passed")
