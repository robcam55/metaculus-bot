"""Offline checks for RcForecastBot's changes to the template (no network, no keys).

Run from the repo root:  python tests/check_rc_bot.py
Covers model routing by environment, credit pacing (budget.py and the run loop in
main.main), the binary trimmed mean, research that survives one failing source and waits
for the next run when every source fails, the dates in every prompt, the run page's
report, and the AskNews settings. Fake key values only; nothing is sent anywhere.
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
    "ASKNEWS_STRATEGY",
    "ASKNEWS_ARTICLES",
    "ASKNEWS_CACHE_MODE",
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
        assert d["researcher"] == main.ASKNEWS_RESEARCHER == "asknews"
        assert main.source_name(d["researcher"]) == "asknews/latest"  # the default strategy
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
    assert (plan(72).futureeval, plan(72).minibench) == (tiers["sonnet-3"], tiers["opus-3"])
    assert (plan(85).futureeval, plan(85).minibench) == (tiers["opus-3"], tiers["opus-3"])
    assert (plan(100).futureeval, plan(100).minibench) == (tiers["opus-3"], tiers["opus-5"])
    assert (plan(110).futureeval, plan(110).minibench) == (tiers["opus-5"], tiers["opus-5"])
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

    assert budget.affordable(10, tiers["sonnet-3"]) == 34  # (10 - 2) / 0.23
    assert budget.affordable(1, tiers["sonnet-3"]) == 0
    assert budget.affordable(None, tiers["opus-5"]) is None
    assert budget.affordable(math.inf, tiers["opus-5"]) is None
    assert budget.affordable(500, tiers["pause"]) == 0
    print("plan: ok")


def check_key_status() -> None:
    """Reads limit_remaining; a failure returns None and logs no request details."""
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
        assert budget.remaining_of(status) == 87.5
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


def research_deferral(url: str) -> BaseException:
    """What forecasting-tools hands back for a question whose research raised
    ResearchUnavailable: the error wrapped in an ExceptionGroup."""
    return ExceptionGroup(
        "1 sub-exceptions -> Error while processing question url",
        [main.ResearchUnavailable(f"No research for {url}")],
    )


def run_main(
    mode: str,
    balance,
    questions: dict,
    fail_urls=(),
    cost_each=0.25,
    deferred_urls=(),
    **env_values,
):
    """Runs main.main() offline: fake Metaculus client, a fake key balance (None when
    unreadable), and fake forecasting that reports `cost_each` per question the way
    OpenRouter's responses do. Questions in `deferred_urls` come back as research
    deferrals. Returns (forecast calls, step summary, exit message or None, number of
    balance reads)."""
    calls: list[tuple] = []
    reads: list[str] = []

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
        results = []
        for q in qs:  # a failed question still costs what its calls cost
            main.MonetaryCostManager.increase_current_usage_in_parent_managers(cost_each)
            if q.page_url in fail_urls:
                results.append(RuntimeError("boom"))
            elif q.page_url in deferred_urls:
                results.append(research_deferral(q.page_url))
            else:
                results.append("report")
        return results

    def fake_status(api_key, timeout=15):
        assert api_key == "sk-fake"
        reads.append(api_key)
        if balance is None:
            return None
        return {"limit": 100.0, "limit_remaining": balance, "usage": 100.0 - balance}

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
    return calls, summary, exit_message, len(reads)


def check_run_loop() -> None:
    opus, sonnet = budget.OPUS, budget.SONNET
    fe_id = main.FALL_2026_FUTUREEVAL_ID

    # $100: FutureEval on opus-3, MiniBench on opus-5; soonest-closing first; the
    # already-forecast question skipped; the balance read once; each pass's spend summed
    # from the costs its responses report
    questions = {
        fe_id: [open_question(1, 2), open_question(2, 1), open_question(3, 0.5, forecasted=True)],
        "minibench": [open_question(4, 50), open_question(5, 40)],
    }
    calls, summary, exited, reads = run_main(
        "tournament", 100.0, questions, OPENROUTER_API_KEY="sk-fake"
    )
    assert exited is None and reads == 1, (exited, reads)
    assert calls == [
        (opus, 3, ["2", "1"], True),
        (opus, 5, ["5", "4"], True),
    ], calls
    assert "| FutureEval | opus-3 | 2 | 0 | $0.50 | $0.25 |" in summary, summary
    assert "| MiniBench | opus-5 | 2 | 0 | $0.50 | $0.25 |" in summary, summary
    assert "$100.00 of $100.00. Spent: $1.00. Left now: about $99.00." in summary, summary
    # the run page says what each tournament had and which research sources are set up
    assert "- FutureEval (`33121`): 3 open, 2 not yet forecast, 2 taken this run." in summary
    assert "- MiniBench (`minibench`): 2 open, 2 not yet forecast, 2 taken this run." in summary
    assert "- Research sources: openrouter/anthropic/claude-sonnet-5:online." in summary, summary

    # nothing open: said apart from "nothing new", and an idle run still names the sources,
    # so newly added AskNews keys show up before any question arrives
    calls, summary, exited, _ = run_main(
        "tournament", 100.0, {"minibench": [open_question(9, 5, forecasted=True)]},
        OPENROUTER_API_KEY="sk-fake", ASKNEWS_API_KEY="x", ASKNEWS_STRATEGY="both",
    )
    assert calls == [] and exited is None, (calls, exited)
    assert "- FutureEval (`33121`): no open questions." in summary, summary
    assert "- MiniBench (`minibench`): 1 open, 0 not yet forecast, 0 taken this run." in summary
    assert (
        "- Research sources: asknews/latest+archive,"
        " openrouter/anthropic/claude-sonnet-5:online." in summary
    ), summary
    assert "| Research source |" not in summary  # no searches, no table

    # $5: cheapest tiers, and only what the balance pays for above the $2 floor
    many = {"minibench": [open_question(n, 100 - n) for n in range(10, 40)]}
    calls, summary, exited, _ = run_main("tournament", 5.0, many, OPENROUTER_API_KEY="sk-fake")
    assert exited is None
    assert (
        "- MiniBench (`minibench`): 30 open, 30 not yet forecast, 13 taken this run"
        " (run size or credit; later runs take the rest)." in summary
    ), summary
    assert len(calls) == 1 and calls[0][:2] == (sonnet, 3), calls
    assert calls[0][2] == [str(n) for n in range(39, 26, -1)], calls  # the 13 closing soonest

    # spend earlier in a run shrinks the next pass's cap: $6 pays for 17 sonnet-3
    # questions above the floor; FutureEval's 5 at $0.50 leave $3.50, room for 6 more
    split = {
        fe_id: [open_question(n, n) for n in range(1, 6)],
        "minibench": [open_question(n, n) for n in range(100, 130)],
    }
    calls, _, _, _ = run_main(
        "tournament", 6.0, split, cost_each=0.5, OPENROUTER_API_KEY="sk-fake"
    )
    assert [len(c[2]) for c in calls] == [5, 6], calls

    # under the floor: nothing runs
    calls, summary, _, _ = run_main("tournament", 1.5, questions, OPENROUTER_API_KEY="sk-fake")
    assert calls == [] and "pause" in summary, (calls, summary)
    assert "- FutureEval (`33121`): 3 open, 2 not yet forecast, 0 taken this run (paused)." in summary

    # a big balance: top tier, but no more than MAX_QUESTIONS_PER_RUN in one run
    burst = {
        fe_id: [open_question(n, n) for n in range(100, 120)],
        "minibench": [open_question(n, n) for n in range(200, 220)],
    }
    calls, _, _, _ = run_main("tournament", 900.0, burst, OPENROUTER_API_KEY="sk-fake")
    assert [(c[0], c[1], len(c[2])) for c in calls] == [
        (opus, 5, 20),
        (opus, 5, main.MAX_QUESTIONS_PER_RUN - 20),
    ], calls

    # an unreadable balance: cheapest tiers, no credit cap; the spend is still measured
    calls, summary, _, _ = run_main("tournament", None, questions, OPENROUTER_API_KEY="sk-fake")
    assert [c[:2] for c in calls] == [(sonnet, 3), (sonnet, 3)], calls
    assert "start of this run: unknown. Spent: $1.00. Left now: unknown." in summary, summary

    # a failed question turns the run red, after the spend is reported
    calls, summary, exited, _ = run_main(
        "tournament",
        100.0,
        questions,
        fail_urls=("https://example.invalid/q/4",),
        OPENROUTER_API_KEY="sk-fake",
    )
    assert exited and "1 question(s) failed" in exited, exited
    assert "lack of research" not in exited, exited
    assert "| MiniBench | opus-5 | 1 | 1 | $0.50 | $0.50 |" in summary, summary

    # a question left without research also turns the run red, says why, and is counted
    # on the run page (the next run retries it)
    calls, summary, exited, _ = run_main(
        "tournament",
        100.0,
        questions,
        fail_urls=("https://example.invalid/q/5",),
        deferred_urls=("https://example.invalid/q/4",),
        OPENROUTER_API_KEY="sk-fake",
    )
    assert exited == (
        "2 question(s) failed (1 for lack of research; the next run retries them);"
        " see the log above"
    ), exited
    assert "- 1 question(s) not forecast: every research source failed." in summary, summary

    # the smoke test: one question of each type, on MiniBench's tier, re-forecast
    testing = {
        "bot-testing-area": [
            open_question(1, 5),
            open_question(2, 5),
            open_question(3, 5, "mc"),
            open_question(4, 5, "numeric"),
        ]
    }
    calls, summary, _, _ = run_main("test_questions", 100.0, testing, OPENROUTER_API_KEY="sk-fake")
    assert calls == [(opus, 5, ["1", "3", "4"], False)], calls
    assert "- bot-testing-area (`bot-testing-area`): 4 open, 3 taken this run." in summary, summary
    calls, _, _, _ = run_main(
        "test_questions", 100.0, testing,
        OPENROUTER_API_KEY="sk-fake", BUDGET_TIER="sonnet-3", TEST_QUESTIONS="1",
    )
    assert calls == [(sonnet, 3, ["1"], False)], calls

    # the Metaculus Cup: never on Metaculus's credits, and no balance read at all
    cup = {main.FALL_2026_METACULUS_CUP_ID: [open_question(7, 30, forecasted=True)]}
    calls, _, exited, reads = run_main("metaculus_cup", 100.0, cup, OPENROUTER_API_KEY="sk-fake")
    assert calls == [] and reads == 0, (calls, reads)
    assert exited and "ANTHROPIC_API_KEY" in exited, exited
    calls, summary, exited, reads = run_main(
        "metaculus_cup", 100.0, cup, OPENROUTER_API_KEY="sk-fake", ANTHROPIC_API_KEY="x"
    )
    # no credits section on a personal key, but the run page still reports the run
    assert exited is None and reads == 0, (exited, reads)
    assert "OpenRouter credits" not in summary, summary
    assert "- Metaculus Cup (`33108`): 1 open, 1 taken this run (re-forecast)." in summary
    assert calls == [("anthropic/claude-sonnet-5", 5, ["7"], False)], calls

    # a mistyped AskNews setting stops the run before anything is read or spent
    for bad in ({"ASKNEWS_STRATEGY": "archives"}, {"ASKNEWS_ARTICLES": "eight"}):
        calls, summary, exited, reads = run_main(
            "tournament", 100.0, questions, OPENROUTER_API_KEY="sk-fake", **bad
        )
        assert calls == [] and reads == 0 and summary == "", (calls, reads, summary)
        assert exited and next(iter(bad)) in exited, exited
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
    # the failure is counted for the run page, by error type only
    tallies = bot.research_health.sources
    assert (tallies["asknews/latest"].searches, tallies["asknews/latest"].worked) == (1, 0)
    assert tallies["asknews/latest"].failures == {"RuntimeError": 1}, tallies
    assert (tallies["perplexity/sonar-pro"].searches, tallies["perplexity/sonar-pro"].worked) == (1, 1)

    async def both(source, prompt, question):
        return f"notes from {bot._source_name(source)}"

    bot._run_one_research_source = both  # type: ignore[method-assign]
    research = asyncio.run(bot.run_research(binary_question()))
    assert research.count("## Research from") == 2, research
    print("research: ok")


def timed_question(kind: str = "binary", closes_in: timedelta | None = timedelta(hours=2)):
    fields = dict(
        question_text="Will it happen?",
        page_url="https://example.invalid/q/77",
        resolution_criteria="Resolves Yes if it happens.",
        fine_print="",
        background_info="The resolution source is the agency's monthly bulletin.",
        close_time=NOW + closes_in if closes_in is not None else None,
        scheduled_resolution_time=datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc),
    )
    if kind == "mc":
        return MultipleChoiceQuestion(options=["a", "b"], **fields)
    if kind == "numeric":
        return NumericQuestion(
            upper_bound=10, lower_bound=0, open_upper_bound=True, open_lower_bound=False, **fields
        )
    if kind == "date":
        return main.DateQuestion(
            upper_bound=datetime(2027, 6, 1, tzinfo=timezone.utc),
            lower_bound=datetime(2026, 10, 1, tzinfo=timezone.utc),
            open_upper_bound=True,
            open_lower_bound=False,
            **fields,
        )
    return BinaryQuestion(**fields)


def check_prompts() -> None:
    """Today's date and the question's close and resolution times reach the researcher
    and every forecast prompt; the researcher also gets the question's background."""
    q = timed_question()
    now = datetime(2026, 10, 4, 13, 5, tzinfo=timezone.utc)
    line = main.timeline(q, now)
    assert line.startswith("Today is 2026-10-04 (13:05 UTC)."), line
    assert f"closes {(NOW + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')} UTC." in line, line
    assert line.endswith("It is scheduled to resolve 2026-12-31 23:59 UTC."), line
    bare = BinaryQuestion(question_text="Q?")
    assert main.timeline(bare, now) == "Today is 2026-10-04 (13:05 UTC).", main.timeline(bare, now)
    # an aware time in another zone is shown in UTC
    eastern = timezone(timedelta(hours=-4))
    late = BinaryQuestion(question_text="Q?", close_time=datetime(2026, 10, 4, 20, 0, tzinfo=eastern))
    assert "closes 2026-10-05 00:00 UTC." in main.timeline(late, now), main.timeline(late, now)

    with env(ANTHROPIC_API_KEY="x"):
        bot = main.RcForecastBot(publish_reports_to_metaculus=False)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    research_prompt = bot._research_prompt(q)
    for expected in (
        f"Today is {today}",
        "scheduled to resolve 2026-12-31 23:59 UTC",
        "The resolution source is the agency's monthly bulletin.",
        "Resolves Yes if it happens.",
    ):
        assert expected in research_prompt, (expected, research_prompt)
    assert "None given." in bot._research_prompt(bare)  # no background: no "None"

    prompts: dict[str, str] = {}

    def capture(kind):
        async def fake(question, prompt):
            prompts[kind] = prompt
            return "captured"

        return fake

    bot._binary_prompt_to_forecast = capture("binary")  # type: ignore[method-assign]
    bot._multiple_choice_prompt_to_forecast = capture("mc")  # type: ignore[method-assign]
    bot._numeric_prompt_to_forecast = capture("numeric")  # type: ignore[method-assign]
    bot._date_prompt_to_forecast = capture("date")  # type: ignore[method-assign]
    asyncio.run(bot._run_forecast_on_binary(timed_question("binary"), "notes"))
    asyncio.run(bot._run_forecast_on_multiple_choice(timed_question("mc"), "notes"))
    asyncio.run(bot._run_forecast_on_numeric(timed_question("numeric"), "notes"))
    asyncio.run(bot._run_forecast_on_date(timed_question("date"), "notes"))
    assert sorted(prompts) == ["binary", "date", "mc", "numeric"], prompts
    for kind, prompt in prompts.items():
        assert f"Today is {today}" in prompt, (kind, prompt)
        assert "scheduled to resolve 2026-12-31 23:59 UTC" in prompt, (kind, prompt)
        assert "Forecasting on this question closes" in prompt, (kind, prompt)
    print("prompts: ok")


def check_research_policy() -> None:
    """Every source failing: a question a later run can reach waits for it; one closing
    sooner is forecast without research, and the forecaster is told so."""
    with env(OPENROUTER_API_KEY="x", ASKNEWS_API_KEY="y"):
        bot = main.RcForecastBot(publish_reports_to_metaculus=False)

    async def all_fail(source, prompt, question):
        if source == main.ASKNEWS_RESEARCHER:
            raise RuntimeError("quota exceeded")
        return "   "  # a model that answers with nothing counts as a failure

    bot._run_one_research_source = all_fail  # type: ignore[method-assign]

    # closes in two hours: later runs can still reach it, so it waits for them
    try:
        asyncio.run(bot.run_research(timed_question(closes_in=timedelta(hours=2))))
    except main.ResearchUnavailable as e:
        assert "next run" in str(e), e
    else:
        raise AssertionError("research that failed everywhere must not feed a forecast")
    # an unknown close time waits too
    try:
        asyncio.run(bot.run_research(timed_question(closes_in=None)))
    except main.ResearchUnavailable:
        pass
    else:
        raise AssertionError("a question without a close time must wait")

    # closes in 30 minutes, before another run can reach it: forecast, but told why
    research = asyncio.run(bot.run_research(timed_question(closes_in=timedelta(minutes=30))))
    assert research == main.NO_RESEARCH_NOTE, research
    health = bot.research_health
    assert health.without_research == ["https://example.invalid/q/77"], health
    asknews, online = health.sources["asknews/latest"], health.sources[
        "openrouter/anthropic/claude-sonnet-5:online"
    ]
    assert (asknews.searches, asknews.worked, dict(asknews.failures)) == (
        3, 0, {"RuntimeError": 3}
    ), asknews
    assert (online.searches, online.worked, dict(online.failures)) == (
        3, 0, {"returned nothing": 3}
    ), online
    # the boundary: just past the margin waits, just inside it doesn't
    margin = main.RESEARCH_RETRY_MARGIN
    assert main.RcForecastBot._can_retry(timed_question(closes_in=margin + timedelta(minutes=1)))
    assert not main.RcForecastBot._can_retry(timed_question(closes_in=margin - timedelta(minutes=1)))

    # research turned off on purpose is not a failure: no error, no note
    with env(OPENROUTER_API_KEY="x", RESEARCHER="no_research"):
        quiet = main.RcForecastBot(publish_reports_to_metaculus=False)
    assert main.research_sources(quiet._llms) == []
    assert asyncio.run(quiet.run_research(timed_question())) == ""
    assert quiet.research_health.sources == {} and quiet.research_health.without_research == []

    # end to end through forecasting-tools: no forecast is made, nothing is posted, and
    # the question comes back as a research deferral the run loop recognises
    made: list[str] = []

    async def no_forecast(question, research):
        made.append(research)
        raise AssertionError("forecast made without research")

    bot._make_prediction = no_forecast  # type: ignore[method-assign]
    results = asyncio.run(
        bot.forecast_questions([timed_question(closes_in=timedelta(hours=3))], return_exceptions=True)
    )
    assert made == [] and len(results) == 1, (made, results)
    assert isinstance(results[0], BaseException) and main.deferred_for_research(results[0]), results
    # other failures, and a mix, are not deferrals
    assert not main.deferred_for_research(RuntimeError("boom"))
    assert not main.deferred_for_research(
        ExceptionGroup("mixed", [main.ResearchUnavailable("x"), ValueError("parse")])
    )
    assert main.deferred_for_research(main.ResearchUnavailable("x"))
    assert not main.deferred_for_research("report")
    print("research policy: ok")


def check_run_report() -> None:
    health = main.ResearchHealth()
    for _ in range(3):
        health.record("asknews/latest")
    health.record("asknews/latest", "RateLimitError")
    health.record("asknews/latest", "RateLimitError")
    health.record("asknews/latest", "AuthenticationError")
    health.record("openrouter/anthropic/claude-sonnet-5:online")
    health.without_research.append("https://example.invalid/q/9")
    lines = main.run_report(
        ["- FutureEval (`33121`): no open questions."],
        ["asknews/latest", "openrouter/anthropic/claude-sonnet-5:online"],
        health,
        ["report", research_deferral("https://example.invalid/q/8"), RuntimeError("boom")],
    )
    text = "\n".join(lines)
    assert lines[0] == "### This run", lines
    assert "| asknews/latest | 6 | 3 | 3 (RateLimitError ×2, AuthenticationError) |" in text, text
    assert "| openrouter/anthropic/claude-sonnet-5:online | 1 | 1 | 0 |" in text, text
    assert "- 1 question(s) not forecast: every research source failed." in text, text
    assert "closing before another try: https://example.invalid/q/9." in text, text
    idle = "\n".join(main.run_report([], [], main.ResearchHealth(), []))
    assert "- Research sources: none." in idle and "|" not in idle and "not forecast" not in idle
    print("run report: ok")


def check_asknews() -> None:
    """The strategy and article count come from ASKNEWS_STRATEGY and ASKNEWS_ARTICLES;
    latest news only, one call, by default (the free tier is 1,000 calls a month)."""
    import asknews_sdk

    calls: list[dict] = []
    sleeps: list[float] = []

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

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    def search(**settings) -> str:
        calls.clear()
        sleeps.clear()
        with env(ASKNEWS_CLIENT_ID="x", ASKNEWS_SECRET="y", **settings):
            return asyncio.run(
                main.ConfiguredAskNewsSearcher().get_formatted_news_async("Will it happen?")
            )

    real_sdk, real_sleep = asknews_sdk.AsyncAskNewsSDK, main.asyncio.sleep
    asknews_sdk.AsyncAskNewsSDK = FakeSDK  # type: ignore[misc]
    main.asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        text = search()
        assert [(c["strategy"], c["n_articles"]) for c in calls] == [("latest news", 8)], calls
        assert calls[0]["query"] == "Will it happen?"
        assert sleeps == [], sleeps
        assert text.startswith("Latest news") and "No articles were found" in text, text

        text = search(ASKNEWS_STRATEGY="archive", ASKNEWS_ARTICLES="12")
        assert [(c["strategy"], c["n_articles"]) for c in calls] == [("news knowledge", 12)], calls
        assert text.startswith("Earlier news from the AskNews archive"), text

        # both: two searches, spaced for the free tier's one call every 10 seconds
        text = search(ASKNEWS_STRATEGY=" Both ")
        assert [c["strategy"] for c in calls] == ["latest news", "news knowledge"], calls
        assert len(sleeps) == 1 and sleeps[0] >= 10, sleeps
        assert text.index("Latest news") < text.index("Earlier news"), text
    finally:
        asknews_sdk.AsyncAskNewsSDK = real_sdk  # type: ignore[misc]
        main.asyncio.sleep = real_sleep  # type: ignore[assignment]

    with env(ASKNEWS_STRATEGY="both"):
        assert main.source_name(main.ASKNEWS_RESEARCHER) == "asknews/latest+archive"
    with env(ASKNEWS_STRATEGY="archive"):
        assert main.source_name(main.ASKNEWS_RESEARCHER) == "asknews/archive"
    for name, value in (
        ("ASKNEWS_STRATEGY", "historical"),
        ("ASKNEWS_ARTICLES", "0"),
        ("ASKNEWS_ARTICLES", "-3"),
        ("ASKNEWS_ARTICLES", "8.5"),
    ):
        with env(**{name: value}):
            try:
                main.asknews_strategy()
                main.asknews_articles()
            except SystemExit as e:
                assert name in str(e) and repr(value) in str(e), e
            else:
                raise AssertionError(f"{name}={value!r} must stop the run")
    print("asknews: ok")


if __name__ == "__main__":
    check_routing()
    check_plan()
    check_key_status()
    check_run_loop()
    check_aggregation()
    check_research_survives_one_failure()
    check_prompts()
    check_research_policy()
    check_run_report()
    check_asknews()
    print("all checks passed")
