"""Offline checks for RcForecastBot's changes to the template (no network, no keys).

Run from the repo root:  python tests/check_rc_bot.py
Covers model routing by environment, credit pacing (budget.py and the run loop in
main.main), Rob's own key taking over under the floor, the alert issues (alerts.py), the
binary trimmed mean, research that survives one failing source and waits for the next
run when every source fails, the dates in every prompt, the run page's report, the
posted comment's sections, the AskNews settings, the research tool settings and the
research watch (research_watch.py), and the spend measured per call (spend.py). Fake key
values only; nothing is sent anywhere.
"""

import asyncio
import contextlib
import io
import json
import logging
import math
import os
import re
import statistics
import sys
import tempfile
import types
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
    "FORECASTER_EFFORT",
    "FORECASTER_TIMEOUT_S",
    "PARSER_MODEL",
    "SEARCH_MODEL",
    "RESEARCH_WEB_FETCH",
    "RESEARCH_TOOLS",
    "RESEARCH_TOOL_NOTE",
    "RESEARCH_WATCH_FILE",
    "TRIAL_RESEARCH_TOOLS",
    "TRIAL_RAW_TRACE",
    "TRIAL_FORECASTER_MODEL",
    "TRIAL_RESEARCHER_MODEL",
    "TRIAL_PARSER_MODEL",
    "TRIAL_EXTRA_FORECASTERS",
    "EXTRA_FORECASTERS",
    "OPENROUTER_OWN_API_KEY",
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
    "PAY_WITH",
    "GITHUB_TOKEN",
    "GITHUB_REPOSITORY",
    "GITHUB_REPOSITORY_OWNER",
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
    import alerts  # noqa: E402
    import budget  # noqa: E402

    budget.KEY_RETRY_WAIT = 0.0  # the second balance read, without the wait
    alerts.REREAD_WAIT = 0.0  # the second issue-list read, without the wait
    import main  # noqa: E402  (import after clearing keys: dotenv must not leak real ones)
    import research_watch  # noqa: E402
    import spend  # noqa: E402
    from forecasting_tools import (  # noqa: E402
        BinaryPrediction,
        BinaryQuestion,
        GeneralLlm,
        MultipleChoiceQuestion,
        NumericQuestion,
        PredictedOptionList,
        ReasonedPrediction,
    )

# Generation look-ups the offline run loop attempted (it must attempt none: every fake
# response carries its cost)
LOOKUPS: list[str] = []


class FakeUsage:
    """The usage OpenRouter returns, as litellm keeps it on a response."""

    def __init__(self, cost=None, upstream=None, byok=None, searches=0, tokens=(1000, 200)):
        self.prompt_tokens, self.completion_tokens = tokens
        if cost is not None:
            self.cost = cost
        if byok is not None:
            self.is_byok = byok
        if upstream is not None:
            self.cost_details = {"upstream_inference_cost": upstream}
        self.server_tool_use = {"web_search_requests": searches}


class FakeResponse:
    def __init__(self, generation="gen-fake", **usage):
        self.id = generation
        self.usage = FakeUsage(**usage)
        self._hidden_params = {"response_cost": 0.0}


@contextmanager
def fixed_anthropic_prices():
    """Anthropic's list prices, fixed in litellm's table for the spend checks. litellm
    downloads its table at import and falls back to a bundled copy, and litellm 1.80.10's
    copy lacks these models, so a failed download must not fail an unrelated change."""
    import litellm

    # input, output, cache read, cache write (5 minutes), per token
    rows = {
        "claude-opus-5-5": (4e-06, 2e-05, 2e-07, 5e-06),
        "claude-sonnet-5": (2e-06, 1e-05, 2e-07, 2.5e-06),
        "claude-sonnet-5-5": (2e-06, 1e-05, 2e-07, 2.5e-06),
        "claude-haiku-4-5": (1e-06, 5e-06, 1e-07, 1.25e-06),
        "claude-haiku-5-5": (1e-07, 5e-07, 1e-08, 1.25e-07),
    }  # fmt: skip
    saved = {name: litellm.model_cost.get(name) for name in rows}
    litellm.register_model({
        name: {"input_cost_per_token": i, "output_cost_per_token": o,
               "cache_read_input_token_cost": r, "cache_creation_input_token_cost": w,
               "litellm_provider": "anthropic", "mode": "chat"}
        for name, (i, o, r, w) in rows.items()
    })  # fmt: skip
    try:
        yield
    finally:
        for name, row in saved.items():
            if row is None:
                litellm.model_cost.pop(name, None)
            else:
                litellm.model_cost[name] = row


def model_of(llm) -> str:
    return llm.model if isinstance(llm, GeneralLlm) else str(llm)


def check_routing() -> None:
    with env(ANTHROPIC_API_KEY="x"):
        d = main.RcForecastBot._llm_config_defaults()
        assert model_of(d["default"]) == "anthropic/claude-sonnet-5-5", d["default"]
        assert d["default"].litellm_kwargs["temperature"] is None
        # litellm's 4,096-token default for Anthropic would leave no room to think, and
        # the forecasters think at high effort, so they get the larger cap and timeout
        assert d["default"].litellm_kwargs["max_tokens"] == main.ANTHROPIC_FORECAST_MAX_TOKENS
        assert d["default"].litellm_kwargs["output_config"] == {"effort": "high"}
        assert d["default"].litellm_kwargs["timeout"] == 480
        # Haiku 5.5 parses: no temperature (it rejects any but the default), low effort,
        # and a reply cap with room for its thinking
        assert model_of(d["parser"]) == "anthropic/claude-haiku-5-5"
        assert d["parser"].litellm_kwargs["temperature"] is None
        assert d["parser"].litellm_kwargs["output_config"] == {"effort": "low"}
        assert d["parser"].litellm_kwargs["max_tokens"] == main.ANTHROPIC_PARSER_MAX_TOKENS
        assert d["summarizer"] is d["parser"]
        # no AskNews: Sonnet 5.5 with Anthropic's own web search and fetch is the one source
        assert main.source_name(d["researcher"]) == "anthropic/claude-sonnet-5-5 + web search and fetch"
        assert isinstance(d["researcher"], main.AnthropicSearchLlm), d["researcher"]
        # the direct tools, passed as tools: web_search_options would send the older
        # web_search_20250305
        assert d["researcher"].litellm_kwargs["tools"] == [main.ANTHROPIC_SEARCH_TOOL, main.ANTHROPIC_FETCH_TOOL]
        assert d["researcher"].litellm_kwargs["tools"][0]["type"] == "web_search_20260209"
        assert main.ANTHROPIC_FETCH_TOOL["type"] == "web_fetch_20260209"
        assert main.ANTHROPIC_SEARCH_TOOL["allowed_callers"] == ["direct"]
        assert main.ANTHROPIC_FETCH_TOOL["allowed_callers"] == ["direct"]
        assert main.ANTHROPIC_SEARCH_TOOL["max_uses"] == 5
        assert main.ANTHROPIC_FETCH_TOOL["max_uses"] == 2
        # searches, fetches and the final answer stay inside the server's 10 rounds, at the
        # highest setting RESEARCH_TOOLS allows
        assert main.MAX_RESEARCH_SEARCHES + main.ANTHROPIC_FETCH_TOOL["max_uses"] <= 8
        assert "web_search_options" not in d["researcher"].litellm_kwargs
        assert d["researcher"].litellm_kwargs["max_tokens"] == main.ANTHROPIC_MAX_TOKENS
        assert d["researcher_2"] is None
    # RESEARCH_WEB_FETCH=false: search only, without a code change
    with env(ANTHROPIC_API_KEY="x", RESEARCH_WEB_FETCH="false"):
        d = main.build_llms("anthropic")
        assert d["researcher"].litellm_kwargs["tools"] == [main.ANTHROPIC_SEARCH_TOOL]
        assert main.source_name(d["researcher"]) == "anthropic/claude-sonnet-5-5 + web search"

    with env(OPENROUTER_API_KEY="x", ASKNEWS_CLIENT_ID="x", ASKNEWS_SECRET="y"):
        d = main.RcForecastBot._llm_config_defaults()
        # donated credits run the stronger model
        assert model_of(d["default"]) == "openrouter/anthropic/claude-opus-5.5"
        assert d["default"].litellm_kwargs["temperature"] is None
        assert model_of(d["parser"]) == "openrouter/anthropic/claude-haiku-5.5"
        assert d["parser"].litellm_kwargs["temperature"] is None
        assert d["researcher"] == main.ASKNEWS_RESEARCHER == "asknews"
        assert main.source_name(d["researcher"]) == "asknews/latest"  # the default strategy
        # Metaculus's OpenRouter credits cover Anthropic but not Perplexity
        assert model_of(d["researcher_2"]) == "openrouter/anthropic/claude-sonnet-5.5:online"
        assert d["researcher_2"].litellm_kwargs["temperature"] is None

    with env(ANTHROPIC_API_KEY="x", PERPLEXITY_API_KEY="x", FORECASTER_MODEL="anthropic/claude-opus-5"):
        d = main.RcForecastBot._llm_config_defaults()
        assert model_of(d["default"]) == "anthropic/claude-opus-5"
        assert model_of(d["researcher"]) == "perplexity/sonar-pro"
        assert d["researcher_2"] is None

    with env(ANTHROPIC_API_KEY="x", PARSER_MODEL="anthropic/claude-sonnet-5-5"):
        d = main.RcForecastBot._llm_config_defaults()
        # a model that rejects sampling parameters gets none
        assert d["parser"].litellm_kwargs["temperature"] is None
    with env(ANTHROPIC_API_KEY="x", PARSER_MODEL="anthropic/claude-haiku-4-5"):
        d = main.RcForecastBot._llm_config_defaults()
        # Haiku 4.5, still selectable: temperature 0 as before, and no effort (it errors)
        assert d["parser"].litellm_kwargs["temperature"] == 0
        assert "output_config" not in d["parser"].litellm_kwargs

    with env():  # no LLM key at all: forecasting-tools' own defaults stay in charge
        d = main.RcForecastBot._llm_config_defaults()
        assert d["researcher_2"] is None
        assert "claude-sonnet-5" not in model_of(d["default"])

    # both keys: Metaculus's donated credits pay before the personal key
    with env(ANTHROPIC_API_KEY="x", OPENROUTER_API_KEY="y", ASKNEWS_API_KEY="z"):
        assert main.llm_route() == "openrouter"
        d = main.RcForecastBot._llm_config_defaults()
        assert model_of(d["default"]) == "openrouter/anthropic/claude-opus-5.5"
        # the personal route (Metaculus Cup, and Rob's key once the credits run out) never
        # touches the OpenRouter key: its searcher calls Anthropic's web search itself
        d = main.build_llms("anthropic")
        assert model_of(d["default"]) == "anthropic/claude-sonnet-5-5"
        assert model_of(d["parser"]) == "anthropic/claude-haiku-5-5"
        assert d["researcher"] == main.ASKNEWS_RESEARCHER
        assert main.source_name(d["researcher_2"]) == "anthropic/claude-sonnet-5-5 + web search and fetch"
        # the OpenRouter route sends no reply cap and no search tool (:online searches)
        d = main.build_llms("openrouter")
        assert "max_tokens" not in d["default"].litellm_kwargs
        assert "tools" not in d["researcher_2"].litellm_kwargs
        assert not isinstance(d["researcher_2"], main.AnthropicSearchLlm)

    # a pacing tier picks the forecaster on the donated credits
    with env(OPENROUTER_API_KEY="x", FORECASTER_MODEL="openrouter/some/other-model"):
        d = main.build_llms("openrouter", budget.TIERS["sonnet-3"])
        assert model_of(d["default"]) == "openrouter/anthropic/claude-sonnet-5.5"
        # FORECASTER_MODEL applies to a personal key only
        assert model_of(main.build_llms("openrouter")["default"]) == budget.OPUS

    # and on Rob's own key, in Anthropic's ids; FORECASTER_MODEL doesn't override a tier
    with env(ANTHROPIC_API_KEY="x", FORECASTER_MODEL="anthropic/some-other-model"):
        d = main.build_llms("anthropic", budget.TIERS["opus-3"])
        assert model_of(d["default"]) == "anthropic/claude-opus-5-5", d["default"]
        assert d["default"].litellm_kwargs["max_tokens"] == main.ANTHROPIC_FORECAST_MAX_TOKENS
        assert d["default"].litellm_kwargs["output_config"] == {"effort": "high"}
        d = main.build_llms("anthropic", budget.TIERS["sonnet-1"])
        assert model_of(d["default"]) == "anthropic/claude-sonnet-5-5"
    assert budget.on_route(budget.OPUS, "openrouter") == budget.OPUS

    # the Test Bot's trial models replace the bot's own in their role, tier or not
    trial = {
        "TRIAL_FORECASTER_MODEL": "anthropic/claude-opus-9",
        "TRIAL_RESEARCHER_MODEL": " anthropic/claude-sonnet-9 ",
        "TRIAL_PARSER_MODEL": "anthropic/claude-haiku-9",
    }
    with env(ANTHROPIC_API_KEY="x", PARSER_MODEL="anthropic/claude-haiku-4-5", **trial):
        d = main.build_llms("anthropic", budget.TIERS["opus-3"])
        assert model_of(d["default"]) == "anthropic/claude-opus-9", d["default"]
        assert d["default"].litellm_kwargs["output_config"] == {"effort": "high"}
        assert model_of(d["researcher"]) == "anthropic/claude-sonnet-9"  # no AskNews here
        assert isinstance(d["researcher"], main.AnthropicSearchLlm)
        assert model_of(d["parser"]) == "anthropic/claude-haiku-9"
    with env(OPENROUTER_API_KEY="x", TRIAL_RESEARCHER_MODEL="openrouter/anthropic/claude-sonnet-9:online"):
        d = main.build_llms("openrouter", budget.TIERS["opus-3"])
        assert model_of(d["researcher"]) == "openrouter/anthropic/claude-sonnet-9:online"
        assert model_of(d["default"]) == budget.OPUS
    # one source for the ids: model_watch.py reads budget.py's
    assert main.SONNET_MODEL["anthropic"] == budget.ANTHROPIC_IDS[budget.SONNET]
    assert main.HAIKU_MODEL == {"openrouter": budget.HAIKU, "anthropic": "anthropic/claude-haiku-5-5"}

    # FORECASTER_EFFORT: another level, or "default" for the model's own; a mistyped one
    # stops the run; never sent on OpenRouter, to the searcher, or to a Haiku 4.5 forecaster
    with env(ANTHROPIC_API_KEY="x", FORECASTER_EFFORT=" Medium "):
        assert main.build_llms("anthropic")["default"].litellm_kwargs["output_config"] == {"effort": "medium"}  # fmt: skip
    with env(ANTHROPIC_API_KEY="x", FORECASTER_EFFORT="default"):
        assert "output_config" not in main.build_llms("anthropic")["default"].litellm_kwargs
    with env(ANTHROPIC_API_KEY="x", FORECASTER_EFFORT="xhigh"):
        try:
            main.forecaster_effort()
        except SystemExit as e:
            assert "FORECASTER_EFFORT" in str(e) and "'xhigh'" in str(e), e
        else:
            raise AssertionError("an effort litellm 1.80.10 rejects must stop the run")
    with env(ANTHROPIC_API_KEY="x", OPENROUTER_API_KEY="y", FORECASTER_MODEL="anthropic/claude-haiku-4-5"):
        assert "output_config" not in main.build_llms("openrouter")["default"].litellm_kwargs
        d = main.build_llms("anthropic")
        assert "output_config" not in d["default"].litellm_kwargs  # Haiku 4.5
        # the parser keeps its own low effort, whatever FORECASTER_EFFORT says
        assert d["parser"].litellm_kwargs["output_config"] == {"effort": "low"}
        assert "output_config" not in d["researcher"].litellm_kwargs  # the searcher
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
    # under a week of credit at the bottom of the ladder (about $17): the last resort, one
    # Sonnet forecast a question, cheaper than sonnet-3
    week = budget.horizon_cost(tiers["sonnet-3"], tiers["sonnet-3"], budget.LAST_RESORT_WEEKS)
    assert plan(budget.FLOOR + week + 0.01).minibench == tiers["sonnet-3"]
    last = plan(budget.FLOOR + week - 0.01)
    assert last.futureeval == last.minibench == tiers["sonnet-1"], last
    assert "last resort" in last.note, last.note
    assert tiers["sonnet-1"].predictions == 1
    assert tiers["sonnet-1"].cost < tiers["sonnet-3"].cost
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
    with env(BUDGET_TIER="sonnet-1"):  # a pin skips the ladder and the last-resort rule
        pinned = plan(80, budget.pinned_tier())
        assert pinned.futureeval == pinned.minibench == tiers["sonnet-1"]
    with env(BUDGET_TIER="Pause"):
        assert budget.pinned_tier() == tiers["pause"]
    with env(BUDGET_TIER="opus-9"):
        try:
            budget.pinned_tier()
        except SystemExit as e:
            assert "opus-9" in str(e)
        else:
            raise AssertionError("an unknown BUDGET_TIER must stop the run")

    # the run cap prices a question at the measured cost, not the pacing estimate
    assert budget.affordable(10, tiers["sonnet-3"]) == 14  # (10 - 2) / 0.55
    assert budget.cap_cost(tiers["opus-3"]) == 0.65 and tiers["opus-3"].cost == 0.30
    assert budget.affordable(2.64, tiers["opus-3"]) == 0 and budget.affordable(2.66, tiers["opus-3"]) == 1
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


# What the last run_main call alerted (issue titles) and how often it checked Rob's key
LAST: dict = {}

SPEND_LIMIT_ERROR = (
    'litellm.BadRequestError: AnthropicException - {"type":"error","error":{"type":'
    '"invalid_request_error","message":"You have reached your specified workspace API'
    ' usage limits. You will regain access on 2026-11-01 at 00:00 UTC."}}'
)


def run_main(
    mode: str,
    balance,
    questions: dict,
    fail_urls=(),
    cost_each=0.25,
    deferred_urls=(),
    limit_urls=(),
    key_check=None,
    own_research=False,
    **env_values,
):
    """Runs main.main() offline: fake Metaculus client, a fake key balance (None when
    unreadable), and fake forecasting whose one model call per question OpenRouter bills
    at `cost_each`. Questions in `deferred_urls` come back as research deferrals, those in
    `limit_urls` as Anthropic's spend-limit error. Rob's own key's check returns
    `key_check`; alerts are recorded in LAST, not opened. With `own_research`, each question
    also makes one research call on Rob's key whose searches found nothing. Returns (forecast
    calls, step summary, exit message or None, number of balance reads)."""
    calls: list[tuple] = []
    reads: list[str] = []
    LAST.clear()
    LAST.update(alerts=[], key_checks=0)

    def fake_alert(title, body, timeout=15, open_only=False):
        LAST["alerts"].append(title)
        return True

    def fake_key_check():
        LAST["key_checks"] += 1
        return key_check

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
            with main.spend.tagged(question=main.question_key(q), part="forecast 1"):
                main.spend.record_response("fake-model", FakeResponse(cost=cost_each))
            if own_research:
                with main.spend.tagged(question=main.question_key(q), part="research: anthropic/x"):
                    main.spend.record_response("anthropic/claude-sonnet-5-5", FakeResponse(cost=0.01))
            # one search per question, through the bot, as run_research records it
            failed = q.page_url in deferred_urls
            bot.research_health.record("fake-search", "RuntimeError" if failed else None)
            if q.page_url in fail_urls:
                results.append(RuntimeError("boom"))
            elif q.page_url in limit_urls:
                results.append(ExceptionGroup("question failed", [RuntimeError(SPEND_LIMIT_ERROR)]))
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

    def fake_generation(api_key, generation_id, timeout=10):
        LOOKUPS.append(generation_id)
        return None

    saved = (
        main.MetaculusClient,
        main.RcForecastBot.forecast_questions,
        main.RcForecastBot.log_report_summary,
        main.print_run_summary_banner,
        budget.key_status,
        spend.generation_record,
        sys.argv,
        alerts.raise_alert,
        main.own_key_check,
    )
    summary_path = Path(tempfile.mkdtemp()) / "summary.md"
    exit_message = None
    try:
        main.MetaculusClient = FakeClient  # type: ignore[misc]
        main.RcForecastBot.forecast_questions = fake_forecast  # type: ignore[method-assign]
        main.RcForecastBot.log_report_summary = classmethod(lambda cls, *a, **k: None)  # type: ignore[method-assign]
        main.print_run_summary_banner = lambda *a, **k: None
        budget.key_status = fake_status
        spend.generation_record = fake_generation
        alerts.raise_alert = fake_alert
        main.own_key_check = fake_key_check
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
            spend.generation_record,
            sys.argv,
            alerts.raise_alert,
            main.own_key_check,
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
    assert "- Research sources: openrouter/anthropic/claude-sonnet-5.5:online." in summary, summary
    # both passes' searches reach the run page: one tally shared across the run
    assert "| fake-search | 4 | 4 | 0 |" in summary, summary
    # and each pass's spend by question and part of the work, with where the figures came from
    assert "### FutureEval: spend by question" in summary, summary
    assert "| Question | Status | Forecast 1 | Total |" in summary, summary
    assert "| [2](https://example.invalid/q/2) | posted | $0.250 | $0.250 |" in summary, summary
    assert "| All 2 | | $0.500 | $0.500 |" in summary, summary
    assert "| Share | | 100% | 100% |" in summary, summary
    assert "- 2 model calls, 0 web searches. Cost figures: 2 billed by OpenRouter." in summary
    assert summary.index("Metaculus's OpenRouter credits") < summary.index("spend by question")

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
        " openrouter/anthropic/claude-sonnet-5.5:online." in summary
    ), summary
    assert "| Research source |" not in summary  # no searches, no table

    # $5, less than a week of credit: the last resort (one Sonnet forecast a question),
    # and only what the balance pays for above the $2 floor at $0.50 a question
    many = {"minibench": [open_question(n, 100 - n) for n in range(10, 40)]}
    calls, summary, exited, _ = run_main("tournament", 5.0, many, OPENROUTER_API_KEY="sk-fake")
    assert exited is None
    assert (
        "- MiniBench (`minibench`): 30 open, 30 not yet forecast, 6 taken this run"
        " (run size or credit; later runs take the rest)." in summary
    ), summary
    assert len(calls) == 1 and calls[0][:2] == (sonnet, 1), calls
    assert calls[0][2] == [str(n) for n in range(39, 33, -1)], calls  # the 6 closing soonest

    # spend earlier in a run shrinks the next pass's cap: $6 pays for 8 sonnet-1
    # questions above the floor; FutureEval's 5 at $0.50 leave $3.50, room for 3 more
    split = {
        fe_id: [open_question(n, n) for n in range(1, 6)],
        "minibench": [open_question(n, n) for n in range(100, 130)],
    }
    calls, _, _, _ = run_main(
        "tournament", 6.0, split, cost_each=0.5, OPENROUTER_API_KEY="sk-fake"
    )
    assert [len(c[2]) for c in calls] == [5, 3], calls

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

    # an unreadable balance, tried twice: cheapest tiers, no credit cap; the spend is
    # still measured
    calls, summary, _, reads = run_main("tournament", None, questions, OPENROUTER_API_KEY="sk-fake")
    assert reads == 2, reads
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
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        calls, summary, exited, _ = run_main(
            "tournament",
            100.0,
            questions,
            fail_urls=("https://example.invalid/q/5",),
            deferred_urls=("https://example.invalid/q/4",),
            OPENROUTER_API_KEY="sk-fake",
        )
    # and the failed source is annotated, since it alone doesn't turn a run red
    assert "::warning title=Research source failed::fake-search failed 1 of 4 searches (RuntimeError x1)" in printed.getvalue()
    assert exited == (
        "2 question(s) failed (1 for lack of research; the next run retries them);"
        " see the log above"
    ), exited
    assert "- 1 question(s) not forecast: every research source failed." in summary, summary
    assert "| fake-search | 4 | 3 | 1 (RuntimeError) |" in summary, summary
    # what a question that waits or fails spent still shows, against its status
    assert "| [4](https://example.invalid/q/4) | waits for research | $0.250 |" in summary, summary
    assert "| [5](https://example.invalid/q/5) | failed | $0.250 |" in summary, summary

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
    assert calls == [("anthropic/claude-sonnet-5-5", 5, ["7"], False)], calls

    # a mistyped AskNews setting stops the run before anything is read or spent
    for bad in ({"ASKNEWS_STRATEGY": "archives"}, {"ASKNEWS_ARTICLES": "eight"}):
        calls, summary, exited, reads = run_main(
            "tournament", 100.0, questions, OPENROUTER_API_KEY="sk-fake", **bad
        )
        assert calls == [] and reads == 0 and summary == "", (calls, reads, summary)
        assert exited and next(iter(bad)) in exited, exited
    assert LOOKUPS == [], LOOKUPS  # every response carried its cost: nothing looked up
    print("run loop: ok")


def check_own_key() -> None:
    """Rob's own Anthropic key takes over under the floor (his choice of 2026-10-11)."""
    fe_id = main.FALL_2026_FUTUREEVAL_ID
    opus_own = "anthropic/claude-opus-5-5"
    questions = {
        fe_id: [open_question(1, 2), open_question(2, 1)],
        "minibench": [open_question(4, 50)],
    }
    keys = dict(OPENROUTER_API_KEY="sk-fake", ANTHROPIC_API_KEY="x")

    # under the floor with the key set: opus-3 on Rob's key for both tournaments, checked
    # once, uncapped by the credits; the run page says whose money paid
    calls, summary, exited, reads = run_main("tournament", 1.5, questions, **keys)
    assert exited is None and reads == 1, (exited, reads)
    assert calls == [(opus_own, 3, ["2", "1"], True), (opus_own, 3, ["4"], True)], calls
    assert LAST["key_checks"] == 1, LAST
    assert "### Your Anthropic key" in summary and "OpenRouter credits" not in summary, summary
    assert "Metaculus's credits: $1.50 left. Spent on your key this run: $0.75." in summary
    assert "(Metaculus's credits are under the $2 floor)" in summary, summary
    assert "| FutureEval | opus-3 | 2 | 0 | $0.50 | $0.25 |" in summary, summary
    assert "- Research sources: anthropic/claude-sonnet-5-5 + web search and fetch." in summary, summary
    assert LAST["alerts"] == ["Metaculus's credits ran out (of the $100 limit)"], LAST
    assert "- Opened an issue to tell Rob about the credits." in summary, summary

    # the key's workspace at its spend limit: nothing forecast, the run green, one alert
    calls, summary, exited, _ = run_main("tournament", 1.5, questions, key_check="limit", **keys)
    assert calls == [] and exited is None, (calls, exited)
    assert LAST["key_checks"] == 1, LAST  # checked once, not once per tournament
    assert (
        "- FutureEval (`33121`): 2 open, 2 not yet forecast, 0 taken this run"
        " (paused: your Anthropic key reached its spend limit)." in summary
    ), summary
    assert any("reached its spend limit" in t for t in LAST["alerts"]), LAST

    # a rejected key: the same, with its own alert
    calls, summary, exited, _ = run_main(
        "tournament", 1.5, questions, key_check="rejected: AuthenticationError", **keys
    )
    assert calls == [] and exited is None, (calls, exited)
    assert "(paused: Anthropic rejected your key (AuthenticationError))." in summary, summary
    assert any("Anthropic rejected the bot's key" in t for t in LAST["alerts"]), LAST

    # nothing new to forecast: the key isn't checked at all
    idle = {"minibench": [open_question(9, 5, forecasted=True)]}
    calls, _, _, _ = run_main("tournament", 1.5, idle, **keys)
    assert calls == [] and LAST["key_checks"] == 0, (calls, LAST)

    # the research watch reads each pass's replies on Rob's key: a line on the run page,
    # an annotation, and the run stays green
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        calls, summary, exited, _ = run_main("tournament", 1.5, questions, own_research=True, **keys)
    assert exited is None and len(calls) == 2, (exited, calls)
    assert (
        "- Web research on your key: 3 reports; 0 of 0 searches brought back pages; 0 fetches"
        " tried; 0 code runs; 3 degraded (2: searches failed, 1: searches failed, 4: searches"
        " failed). Last 3 reports (window not full): 3 degraded." in summary
    ), summary
    assert "::warning title=Research degraded::3 of 3 research reports" in printed.getvalue()
    # nothing on Rob's key: no line
    _, summary, _, _ = run_main("tournament", 1.5, questions, **keys)
    assert "Web research on your key" not in summary, summary

    # a mistyped RESEARCH_TOOLS stops the run at startup; the Test Bot's input does too on
    # the Test Bot, and any other run drops it, like the model trials
    calls, _, exited, _ = run_main("tournament", 1.5, questions, RESEARCH_TOOLS="fast", **keys)
    assert calls == [] and "RESEARCH_TOOLS must be" in (exited or ""), (calls, exited)
    calls, _, exited, _ = run_main("tournament", 1.5, questions, TRIAL_RESEARCH_TOOLS="fast", **keys)
    assert exited is None and calls, (calls, exited)
    _, _, exited, _ = run_main("test_questions", 1.5, questions, TRIAL_RESEARCH_TOOLS="fast", **keys)
    assert "RESEARCH_TOOLS must be" in (exited or ""), exited

    # no key set: paused under the floor, and the alert says so
    calls, summary, _, _ = run_main("tournament", 1.5, questions, OPENROUTER_API_KEY="sk-fake")
    assert calls == [] and LAST["key_checks"] == 0, (calls, LAST)
    assert "(paused)." in summary and "### Metaculus's OpenRouter credits" in summary, summary

    # BUDGET_TIER=pause stops Rob's key too
    calls, _, _, _ = run_main("tournament", 1.5, questions, BUDGET_TIER="pause", **keys)
    assert calls == [] and LAST["key_checks"] == 0, (calls, LAST)

    # low but above the floor: still on the credits, one alert
    calls, _, _, _ = run_main("tournament", 10.0, questions, **keys)
    assert calls and all(c[0].startswith("openrouter/") for c in calls), calls
    assert LAST["alerts"] == ["Metaculus's credits are below $15 (of the $100 limit)"], LAST
    calls, _, _, _ = run_main("tournament", 100.0, questions, **keys)
    assert LAST["alerts"] == [], LAST

    # the spend limit reached mid-run: those questions don't turn the run red, others do
    calls, summary, exited, _ = run_main(
        "tournament", 1.5, questions, limit_urls=("https://example.invalid/q/4",), **keys
    )
    assert exited is None, exited
    assert "- 1 question(s) not forecast: your Anthropic key reached its spend limit." in summary
    assert any("reached its spend limit" in t for t in LAST["alerts"]), LAST
    _, _, exited, _ = run_main(
        "tournament", 1.5, questions,
        limit_urls=("https://example.invalid/q/4",), fail_urls=("https://example.invalid/q/1",),
        **keys,
    )
    assert exited and exited.startswith("1 question(s) failed"), exited

    # PAY_WITH=own-key (the Test Bot's input) makes Rob's key pay while credits last
    testing = {"bot-testing-area": [open_question(1, 5), open_question(3, 5, "mc")]}
    calls, summary, _, _ = run_main("test_questions", 80.0, testing, PAY_WITH="own-key", **keys)
    assert calls == [(opus_own, 3, ["1", "3"], False)], calls
    assert "(PAY_WITH=own-key)" in summary, summary
    # ... and stops the run before anything is read without a key, or when mistyped
    for bad, needle in (
        (dict(PAY_WITH="own-key", OPENROUTER_API_KEY="sk-fake"), "ANTHROPIC_API_KEY"),
        (dict(PAY_WITH="own_key", **keys), "PAY_WITH"),
    ):
        calls, summary, exited, reads = run_main("test_questions", 80.0, testing, **bad)
        assert calls == [] and reads == 0 and summary == "", (calls, reads, summary)
        assert exited and needle in exited, exited

    # Anthropic's spend-limit errors, found inside forecasting-tools' error groups
    assert main.spend_limit_reached(RuntimeError(SPEND_LIMIT_ERROR))
    assert main.spend_limit_reached(
        ExceptionGroup("q", [ExceptionGroup("r", [RuntimeError("enforced_spend_limit_reached")])])
    )
    assert main.spend_limit_reached(RuntimeError("Your credit balance is too low to access"))
    assert not main.spend_limit_reached(RuntimeError("rate_limit_error: retry after 20s"))
    assert not main.spend_limit_reached(research_deferral("https://example.invalid/q/1"))

    # The band above the floor: the cap lets a run spend only while the balance above $2
    # covers one $0.30 opus-3 estimate, so the credits come to rest between $2.00 and
    # $2.30. That counts as spent: Rob's key takes over, and "ran out" opens
    calls, summary, exited, _ = run_main("tournament", 2.16, questions, BUDGET_TIER="opus-3", **keys)
    assert exited is None and calls == [
        (opus_own, 3, ["2", "1"], True), (opus_own, 3, ["4"], True)
    ], calls
    assert "can't pay for another question above the $2 reserve" in summary, summary
    assert LAST["alerts"] == ["Metaculus's credits ran out (of the $100 limit)"], LAST
    # unpinned, the same band under the last resort ($0.18 a question)
    calls, _, _, _ = run_main("tournament", 2.10, questions, **keys)
    assert [c[0] for c in calls] == [opus_own, opus_own], calls
    # without a key: paused there, said plainly, not "later runs take the rest"
    calls, summary, exited, _ = run_main(
        "tournament", 2.16, questions, BUDGET_TIER="opus-3", OPENROUTER_API_KEY="sk-fake"
    )
    assert calls == [] and exited is None, (calls, exited)
    assert "(paused)." in summary and "later runs take the rest" not in summary, summary
    assert LAST["alerts"] == ["Metaculus's credits ran out (of the $100 limit)"], LAST
    # one question's worth above the reserve ($0.65 at opus-3): still on the credits,
    # capped at one; $2.40 is already spent
    calls, _, _, _ = run_main("tournament", 2.70, questions, BUDGET_TIER="opus-3", **keys)
    assert [(c[0], len(c[2])) for c in calls] == [(budget.OPUS, 1)], calls
    assert LAST["key_checks"] == 0, LAST
    calls, _, _, _ = run_main("tournament", 2.40, questions, BUDGET_TIER="opus-3", **keys)
    assert [c[0] for c in calls] == [opus_own, opus_own], calls
    # BUDGET_TIER=pause in the band is a deliberate stop, not spent credits
    calls, _, _, _ = run_main("tournament", 2.16, questions, BUDGET_TIER="pause", **keys)
    assert calls == [] and LAST["key_checks"] == 0, (calls, LAST)
    assert not any("ran out" in t for t in LAST["alerts"]), LAST

    # Rob's key hits its limit in the FutureEval pass: the MiniBench pass pauses quietly
    # instead of spending AskNews calls and refused requests on every question
    calls, summary, exited, _ = run_main(
        "tournament", 1.5, questions, limit_urls=("https://example.invalid/q/2",), **keys
    )
    assert exited is None and [c[2] for c in calls] == [["2", "1"]], (exited, calls)
    assert "(paused: your Anthropic key reached its spend limit)." in summary, summary
    # on Metaculus's credits the same words are someone else's limit: an ordinary failure
    _, _, exited, _ = run_main(
        "tournament", 100.0, questions, limit_urls=("https://example.invalid/q/4",), **keys
    )
    assert exited and exited.startswith("1 question(s) failed"), exited
    assert not any("spend limit" in t for t in LAST["alerts"]), LAST

    # The real key check, against a stand-in litellm with nothing but completion(): a
    # class one litellm version lacks (1.80.10, which the runs install, has no top-level
    # PermissionDeniedError) can't crash it, and every outcome is classified
    class Refused(Exception):
        def __init__(self, text, status_code=None):
            super().__init__(text)
            self.status_code = status_code

    outcomes = [
        (None, None),
        (Refused(SPEND_LIMIT_ERROR, 400), "limit"),
        (Refused('AnthropicException - {"error":{"type":"billing_error"}}', 402), "limit"),
        (Refused("invalid x-api-key", 401), "rejected: Refused"),
        (Refused('AnthropicException - {"error":{"type":"permission_error"}}', 500), "rejected: Refused"),
        (Refused("Overloaded", 529), None),  # passing: the run goes ahead
        (TimeoutError("timed out"), None),
    ]
    real_litellm = main.litellm
    try:
        for error, expected in outcomes:
            def completion(error=error, **kwargs):
                assert kwargs["model"] == "anthropic/claude-haiku-5-5" and kwargs["max_tokens"] == 64
                assert kwargs["output_config"] == {"effort": "low"} and "temperature" not in kwargs
                if error is not None:
                    raise error

            main.litellm = types.SimpleNamespace(completion=completion)
            assert main.own_key_check() == expected, (error, expected)
    finally:
        main.litellm = real_litellm

    # the credits' balance can't be read (twice): Rob's key pays, rather than the run going
    # back onto credits that ran out, uncapped
    calls, summary, exited, reads = run_main("tournament", None, questions, **keys)
    assert reads == 2 and exited is None, (reads, exited)
    assert calls == [(opus_own, 3, ["2", "1"], True), (opus_own, 3, ["4"], True)], calls
    assert LAST["key_checks"] == 1, LAST
    assert "(Metaculus's credits couldn't be read)" in summary, summary
    assert "Metaculus's credits: not read" in summary, summary

    # no OpenRouter key at all: Rob's key at the pinned tier, checked, and pause still stops
    # it (it used to fall through to the Metaculus Cup's unpaced Sonnet 5 x 5)
    calls, summary, exited, reads = run_main("tournament", 100.0, questions, ANTHROPIC_API_KEY="x")
    assert reads == 0 and exited is None, (reads, exited)
    assert calls == [(opus_own, 3, ["2", "1"], True), (opus_own, 3, ["4"], True)], calls
    assert LAST["key_checks"] == 1 and "(no OpenRouter key)" in summary, (LAST, summary)
    calls, _, _, _ = run_main("tournament", 100.0, questions, ANTHROPIC_API_KEY="x", BUDGET_TIER="pause")
    assert calls == [] and LAST["key_checks"] == 0, (calls, LAST)
    calls, _, _, _ = run_main("tournament", 100.0, questions, ANTHROPIC_API_KEY="x", key_check="limit")
    assert calls == [] and any("spend limit" in t for t in LAST["alerts"]), (calls, LAST)

    plan = budget.own_key_plan(None, "why")
    assert plan.futureeval == plan.minibench == budget.TIERS["opus-3"] and plan.remaining is None
    assert budget.own_key_plan(budget.TIERS["sonnet-3"], "why").minibench.name == "sonnet-3"
    assert budget.under_floor(1.99) and not budget.under_floor(2.0)
    assert not budget.under_floor(None)
    print("own key: ok")


def check_alerts() -> None:
    """One issue per alert title, mentioning the owner; nothing leaks into the log."""
    secret = "ghs_FAKE-never-logged"
    issues: list[dict] = [{"title": "Already said"}]
    sent: list[tuple] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_get(url, headers, params, timeout):
        sent.append(("GET", url, params))
        assert headers["Authorization"] == f"Bearer {secret}"
        return FakeResponse(issues)

    def fake_post(url, headers, json, timeout):
        sent.append(("POST", url, json))
        return FakeResponse({})

    def failing_get(url, headers, params, timeout):
        raise RuntimeError(f"connection refused while sending {headers}")

    real_get, real_post = alerts.requests.get, alerts.requests.post
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    alerts.logger.addHandler(handler)
    try:
        alerts.requests.get, alerts.requests.post = fake_get, fake_post  # type: ignore[assignment]
        with env():  # outside the workflow: only logged
            assert alerts.raise_alert("Title", "Body") is False and sent == []
        repo = dict(GITHUB_TOKEN=secret, GITHUB_REPOSITORY="o/r", GITHUB_REPOSITORY_OWNER="o")
        with env(**repo):
            assert alerts.raise_alert("Already said", "Body") is False
            assert [s[0] for s in sent] == ["GET"], sent
            assert sent[0][1] == "https://api.github.com/repos/o/r/issues"
            assert sent[0][2]["state"] == "all"  # a closed issue still counts as said
            sent.clear()
            assert alerts.raise_alert("New", "Body") is True
            # read twice before opening: one read came back without #9 and #10 duplicated it
            assert [s[0] for s in sent] == ["GET", "GET", "POST"], sent
            assert sent[2] == (
                "POST", "https://api.github.com/repos/o/r/issues", {"title": "New", "body": "@o Body"}
            ), sent
            # the first read misses the issue, the second finds it: nothing opens
            sent.clear()
            reads = iter([[], issues])
            alerts.requests.get = lambda url, headers, params, timeout: (  # type: ignore[assignment]
                sent.append(("GET", url, params)) or FakeResponse(next(reads))
            )
            assert alerts.raise_alert("Already said", "Body") is False
            assert [s[0] for s in sent] == ["GET", "GET"], sent
            alerts.requests.get = fake_get  # type: ignore[assignment]
            # Rob's key's alerts count an issue only while it's open
            sent.clear()
            assert alerts.raise_alert("Already said", "Body", open_only=True) is False
            assert sent[0][2]["state"] == "open", sent
            alerts.requests.get = failing_get  # type: ignore[assignment]
            assert alerts.raise_alert("New", "Body") is False
    finally:
        alerts.requests.get, alerts.requests.post = real_get, real_post  # type: ignore[assignment]
        alerts.logger.removeHandler(handler)
    assert records and all(secret not in r.getMessage() for r in records)

    # titles change only with what makes the event new: the limit, or the month
    titles: list[str] = []
    open_only: list[bool] = []
    real = alerts.raise_alert
    alerts.raise_alert = lambda title, body, timeout=15, open_only_=False, **kw: (  # type: ignore[assignment]
        titles.append(title) or open_only.append(kw.get("open_only", open_only_)) or True
    )
    try:
        alerts.credits_low(14.2, 100.0, own_key=True)
        alerts.credits_low(9.0, 100.0, own_key=False)
        alerts.credits_low(14.2, 250.0, own_key=True)
        alerts.credits_out(1.5, None, own_key=True)
        alerts.own_key_limit()
    finally:
        alerts.raise_alert = real  # type: ignore[assignment]
    assert titles[0] == titles[1] == "Metaculus's credits are below $15 (of the $100 limit)"
    assert titles[2] == "Metaculus's credits are below $15 (of the $250 limit)"
    assert titles[3] == "Metaculus's credits ran out"
    assert titles[4].startswith("Your Anthropic key reached its spend limit (20"), titles
    # credit alerts count closed issues too (one per limit); Rob's key's, open ones only
    assert open_only == [False, False, False, False, True], open_only

    # degraded research: titled by the ISO week, so a second call that week opens nothing,
    # whether or not Rob closed the first
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    filed: list[dict] = []
    alerts.requests.get = lambda url, headers, params, timeout: FakeResponse(filed)  # type: ignore[assignment]
    alerts.requests.post = lambda url, headers, json, timeout: (  # type: ignore[assignment]
        filed.append(json) or FakeResponse({})
    )
    try:
        with env(GITHUB_TOKEN=secret, GITHUB_REPOSITORY="o/r"):
            assert alerts.research_degraded(6, 12, "searches failed") is True
            assert alerts.research_degraded(7, 12, "stopped early") is False
    finally:
        alerts.requests.get, alerts.requests.post = real_get, real_post  # type: ignore[assignment]
    assert [issue["title"] for issue in filed] == [f"Research is degraded ({year}-W{week:02d})"], filed
    assert filed[0]["body"].startswith(
        "6 of the last 12 research reports on your Anthropic key came back degraded"
        " (searches failed)."
    ), filed
    assert "RESEARCH_TOOLS changes the tool setting" in filed[0]["body"], filed
    print("alerts: ok")


def check_watchdog() -> None:
    """clock_watchdog.py: the backup clock's check that the Cloudflare clock still ticks."""
    import clock_watchdog as watchdog

    now = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
    runs: list[dict] = []
    raised: list[tuple] = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"workflow_runs": runs}

    def fake_get(url, headers, params, timeout):
        assert url.endswith("/actions/workflows/run_bot_on_tournament.yaml/runs"), url
        assert params["event"] == "workflow_dispatch", params
        return FakeResponse()

    def run(minutes_ago: int, source: str | None, actor: str = "robcam55") -> dict:
        created = (now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        title = "Forecast on new AI tournament questions" + (f" ({source})" if source else "")
        return {"created_at": created, "display_title": title, "triggering_actor": {"login": actor}}

    real_get, real_alert = watchdog.requests.get, watchdog.alerts.raise_alert
    try:
        watchdog.requests.get = fake_get  # type: ignore[assignment]
        watchdog.alerts.raise_alert = lambda title, body, **kw: raised.append((title, kw)) or True  # type: ignore[assignment]
        with env(GITHUB_TOKEN="ghs_fake", GITHUB_REPOSITORY="o/r"):
            # ticking: the Worker's run 15 minutes ago, after the backup clock's own
            runs[:] = [run(2, "backup", "github-actions[bot]"), run(15, "cloudflare")]
            assert watchdog.check(now).startswith("Cloudflare clock: ticking"), raised
            assert raised == []
            # stopped: only the backup clock's runs and Rob's own for three hours (a run by
            # hand doesn't hide a stopped Worker, though Rob's login starts both)
            runs[:] = [run(5, "backup", "github-actions[bot]"), run(30, "manual"), run(200, "cloudflare")]
            line = watchdog.check(now)
            assert "stopped (its last run was 2026-11-20 08:40 UTC); opened an issue" in line, line
            assert raised == [("The Cloudflare clock stopped", {"open_only": True})], raised
            # runs from before the run-name: told apart by who started them
            runs[:] = [run(5, None, "github-actions[bot]"), run(15, None)]
            assert watchdog.check(now).startswith("Cloudflare clock: ticking")
            runs[:] = []
            assert "no run it started is listed" in watchdog.check(now)
            # GitHub unreachable: noted, never raised, so the tick itself goes on
            watchdog.requests.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))  # type: ignore[assignment]
            assert watchdog.check(now) == "Cloudflare clock: not checked (RuntimeError)."
        with env():
            assert "no GitHub token" in watchdog.check(now)
    finally:
        watchdog.requests.get, watchdog.alerts.raise_alert = real_get, real_alert  # type: ignore[assignment]
    print("watchdog: ok")


def check_model_watch() -> None:
    """model_watch.py: the daily check for new Claude models and retired ones."""
    import model_watch

    pages: list[dict] = []
    asked: list[dict] = []
    raised: list[tuple] = []

    class FakeResponse:
        def __init__(self, page):
            self.page = page

        def raise_for_status(self):
            pass

        def json(self):
            return self.page

    def fake_get(url, headers, params, timeout):
        assert url == "https://api.anthropic.com/v1/models", url
        assert headers["x-api-key"] == "sk-fake" and headers["anthropic-version"], headers
        asked.append(dict(params))
        return FakeResponse(pages[len(asked) - 1])

    def model(model_id: str, released: str, name: str = "") -> dict:
        return {"type": "model", "id": model_id, "display_name": name or model_id,
                "created_at": f"{released}T00:00:00Z"}  # fmt: skip

    in_use = [
        model("claude-opus-5-5", "2026-08-12"),
        model("claude-sonnet-5-5", "2026-09-30"),
        model("claude-haiku-5-5", "2026-10-14"),
        model("claude-fable-5-1", "2026-10-02"),
        model("claude-opus-4-8", "1970-01-01"),  # an unknown release date reads as the epoch
    ]
    real_get, real_alert = model_watch.requests.get, model_watch.alerts.raise_alert
    try:
        model_watch.requests.get = fake_get  # type: ignore[assignment]
        model_watch.alerts.raise_alert = lambda title, body, **kw: raised.append((title, body, kw)) or True  # type: ignore[assignment]
        with env(ANTHROPIC_API_KEY="sk-fake"):
            # nothing released since the review: no issue
            pages[:] = [{"data": in_use, "has_more": False, "last_id": "claude-opus-4-8"}]
            lines = model_watch.check()
            assert lines == ["Claude models: nothing new since 2026-10-28 (5 listed)."], lines
            assert raised == [] and asked == [{"limit": 1000}], (raised, asked)
            # a new Opus on the second page, and a new line: one issue each, by role
            asked.clear()
            pages[:] = [
                {"data": in_use[:3], "has_more": True, "last_id": "claude-haiku-5-5"},
                {"data": [*in_use[3:], model("claude-opus-5-6", "2026-11-19", "Claude Opus 5.6"),
                          model("claude-vellum-1", "2026-12-02")], "has_more": False},
            ]  # fmt: skip
            lines = model_watch.check()
            assert asked[1] == {"limit": 1000, "after_id": "claude-haiku-5-5"}, asked
            titles = [t for t, _, _ in raised]
            assert titles == [
                "New Claude model to evaluate: Claude Opus 5.6 (claude-opus-5-6)",
                "New Claude model to evaluate: claude-vellum-1 (claude-vellum-1)",
            ], titles
            opus_body, other_body = raised[0][1], raised[1][1]
            assert "runs for forecasts" in opus_body, opus_body
            assert "`anthropic/claude-opus-5-6` as the forecaster model" in opus_body
            assert "own-key" in opus_body and "never tournament questions" in opus_body
            assert "weigh it against Opus 5.5" in other_body, other_body
            assert all(kw == {} for _, _, kw in raised), raised  # once ever, closed or not
            assert lines[0] == "Claude models: new, `claude-opus-5-6` (2026-11-19); opened an issue.", lines
            # a model the bot runs drops off the list: an issue, before any new-model one
            asked.clear()
            raised.clear()
            pages[:] = [{"data": [m for m in in_use if m["id"] != "claude-haiku-5-5"]}]
            lines = model_watch.check()
            assert [t for t, _, _ in raised] == ["Claude model no longer offered: claude-haiku-5-5"]
            assert "parsing" in raised[0][1] and "fails" in raised[0][1], raised
            assert lines == ["Claude models: `anthropic/claude-haiku-5-5` is no longer listed; opened an issue."], lines
            # an empty or failed list is never read as every model gone
            raised.clear()
            asked.clear()
            pages[:] = [{"data": []}]
            assert model_watch.check() == ["Claude models: not checked (the list came back empty)."]
            model_watch.requests.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sk-fake"))  # type: ignore[assignment]
            assert model_watch.check() == ["Claude models: not checked (RuntimeError)."]
            assert raised == []
        with env():
            assert "no ANTHROPIC_API_KEY" in model_watch.check()[0]
    finally:
        model_watch.requests.get, model_watch.alerts.raise_alert = real_get, real_alert  # type: ignore[assignment]
    # the roles cover every model the bot runs by default
    assert set(model_watch.ROLES) == set(budget.ANTHROPIC_IDS.values()), model_watch.ROLES
    print("model watch: ok")


def check_model_tags() -> None:
    """Each forecast names its model in its reasoning and on its line of the posted
    summary, in a form bot-review keeps and scorecard.py reads."""
    import scorecard
    from forecasting_tools.data_models.binary_report import BinaryReport
    from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
    from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

    opus = "anthropic/claude-opus-5-5"
    tagged = main.with_model(opus, "Step (a)...\nProbability: 34%")
    assert tagged.startswith(f"Model: {opus}\n\n") and main.model_of_reasoning(tagged) == opus
    assert main.model_of_reasoning("Step (a)... Model: x") is None

    with env(ANTHROPIC_API_KEY="x"):
        bot = main.RcForecastBot(publish_reports_to_metaculus=False)
    bot.get_llm = lambda name, kind="llm": FakeLlm("Probability: 34%", model=opus)  # type: ignore[method-assign]

    async def fake_parse(*args, **kwargs):
        return BinaryPrediction(prediction_in_decimal=0.34)

    real_parse = main.parse
    try:
        main.parse = fake_parse
        got = asyncio.run(bot._binary_prompt_to_forecast(binary_question(), "prompt"))
    finally:
        main.parse = real_parse
    assert main.model_of_reasoning(got.reasoning) == opus, got.reasoning

    # the summary: bot-review 0.1.2's own pattern still finds every forecaster, and the
    # text it keeps for each carries the model
    forecaster_pattern = re.compile(r"^\*Forecaster (\d+)(?: \([^)]*\))?\*:", re.MULTILINE)
    collection = ResearchWithPredictions(
        research_report="r", summary_report="s",
        predictions=[got, ReasonedPrediction(prediction_value=0.5, reasoning="untagged")],
    )  # fmt: skip
    summary = main.RcForecastBot._format_and_expand_research_summary(1, BinaryReport, collection)
    marks = list(forecaster_pattern.finditer(summary))
    assert [m.group(1) for m in marks] == ["1", "2"], summary
    first = summary[marks[0].end() : marks[1].start()].strip()
    assert first == f"34.0%\nModel: {opus}", first
    assert scorecard._MODEL.search(first).group(1) == opus
    assert "Model:" not in summary[marks[1].end() :].split("###")[0], summary
    assert "## Report 1 Summary" in summary and "### Research Summary" in summary, summary
    # multiple choice: the option lines stay parseable above the model line
    options = PredictedOptionList(predicted_options=[
        {"option_name": "a", "probability": 0.7}, {"option_name": "b", "probability": 0.3}])  # fmt: skip
    mc_collection = ResearchWithPredictions(
        research_report="r", summary_report="s",
        predictions=[ReasonedPrediction(prediction_value=options, reasoning=main.with_model(opus, "x"))],
    )  # fmt: skip
    mc_summary = main.RcForecastBot._format_and_expand_research_summary(1, MultipleChoiceReport, mc_collection)  # fmt: skip
    body = mc_summary[forecaster_pattern.search(mc_summary).end() :].split("###")[0].strip()
    q = {"question_type": "multiple_choice", "resolution": "b"}
    assert scorecard._MODEL.search(body).group(1) == opus, body
    assert abs(scorecard._member_points(q, scorecard._MODEL.sub("", body)) - 100 * math.log(0.3)) < 1e-9, body
    print("model tags: ok")


def check_extra_forecasters() -> None:
    """A second model family on Rob's own OpenRouter key: off by default, the last forecasts
    of each report, OpenRouter's own reasoning setting, and its spend kept apart."""
    sol, argon, opus = "openrouter/openai/gpt-6.1-sol", "openrouter/google/gemini-4-argon", "anthropic/claude-opus-5-5"
    own = {"OPENROUTER_OWN_API_KEY": "sk-or-own-fake"}
    with env(ANTHROPIC_API_KEY="x"):
        assert main.extra_forecaster_models() == [] and main.build_extra_forecasters() == []
    # a count per model; a variant such as :online is part of the id; commas, semicolons,
    # spaces or new lines between items
    with env(EXTRA_FORECASTERS=f" {sol}:2, {argon} ;openrouter/x/y:online\nopenrouter/x/z:online:1", **own):
        assert main.extra_forecaster_models() == [sol, sol, argon, "openrouter/x/y:online", "openrouter/x/z:online"]
    # without the key: none (main() then notes and annotates it, or stops a Test Bot trial)
    with env(EXTRA_FORECASTERS=f"{sol}:2"):
        assert main.extra_forecaster_models() == [] and main.requested_extras() == f"{sol}:2"
    # mistyped: the run stops at startup
    for bad in ("anthropic/claude-opus-5-5:2", f"{sol}:0", f"{sol}:6", f"{sol}: 2", f"{sol} x2", "openrouter/sol"):
        with env(EXTRA_FORECASTERS=bad, **own):
            try:
                main.extra_forecaster_models()
            except SystemExit as e:
                assert "EXTRA_FORECASTERS" in str(e), e
            else:
                raise AssertionError(f"{bad!r} must stop the run")
    # the Test Bot's input wins over the variable
    with env(EXTRA_FORECASTERS=f"{sol}:2", TRIAL_EXTRA_FORECASTERS=argon, **own):
        assert main.extra_forecaster_models() == [argon]

    # built on Rob's own key (never the credits' key in the environment), with OpenRouter's
    # reasoning setting, which litellm passes for Gemini 4 Argon where reasoning_effort fails
    with env(ANTHROPIC_API_KEY="x", OPENROUTER_API_KEY="sk-or-credits", EXTRA_FORECASTERS=f"{sol}:2,{argon}", **own):
        built = main.build_extra_forecasters()
        assert [b.model for b in built] == [sol, sol, argon] and built[0] is built[1], built
        assert built[0].litellm_kwargs["api_key"] == "sk-or-own-fake"
        assert built[2].litellm_kwargs["reasoning"] == {"effort": "high"}
        assert built[0].litellm_kwargs["timeout"] == 480
        assert built[0].litellm_kwargs["max_tokens"] == main.ANTHROPIC_FORECAST_MAX_TOKENS
        # three Opus forecasts plus three extras
        bot = main.make_bot("anthropic", budget.TIERS["opus-3"], publish=False)
        assert bot.predictions_per_research_report == 6 and len(bot.extra_forecasters) == 3
        # never more extras than the bot's own forecasts: one Sonnet, one extra
        bot = main.make_bot("anthropic", budget.TIERS["sonnet-1"], publish=False)
        assert bot.predictions_per_research_report == 2 and [b.model for b in bot.extra_forecasters] == [sol]
    with env(EXTRA_FORECASTERS=sol, FORECASTER_EFFORT="default", **own):
        assert "reasoning" not in main.build_extra_forecasters()[0].litellm_kwargs

    # the forecasts of one question run together; the last two go to the extra model, each
    # names its model, and the run page counts the extras' forecasts
    with env(ANTHROPIC_API_KEY="x"):
        bot = main.RcForecastBot(
            publish_reports_to_metaculus=False, predictions_per_research_report=5,
            extra_forecasters=[FakeLlm("Probability: 60%", model=sol)] * 2,
        )  # fmt: skip
    bot.get_llm = lambda name, kind="llm": FakeLlm("Probability: 30%", model=opus)  # type: ignore[method-assign]

    async def fake_parse(text, *args, **kwargs):
        return BinaryPrediction(prediction_in_decimal=float(text.split()[-1].rstrip("%")) / 100)

    async def predictions(question, n, **kwargs):
        bot._note_pads.append(await bot._initialize_notepad(question))  # as forecast_questions does
        return await asyncio.gather(*(bot._make_prediction(question, "notes") for _ in range(n)), **kwargs)  # fmt: skip

    real_parse = main.parse
    try:
        main.parse = fake_parse
        got = asyncio.run(predictions(binary_question(), 5))
        assert [main.model_of_reasoning(g.reasoning) for g in got] == [opus] * 3 + [sol] * 2, got
        assert [g.prediction_value for g in got] == [0.3] * 3 + [0.6] * 2, got
        assert bot.research_health.forecasters[sol].worked == 2, bot.research_health.forecasters
        # an extra that fails thins the members and is counted; the others go on
        broken = FakeLlm(model=argon)

        async def refuse(prompt, system_prompt=None):
            raise RuntimeError("402 insufficient credits")

        broken.invoke = refuse  # type: ignore[method-assign]
        bot.extra_forecasters = [broken]
        bot.predictions_per_research_report = 4
        q = binary_question().model_copy(update={"page_url": "https://example.invalid/q/9"})
        results = asyncio.run(predictions(q, 4, return_exceptions=True))
        assert [type(r).__name__ for r in results] == ["ReasonedPrediction"] * 3 + ["RuntimeError"], results
        assert dict(bot.research_health.forecasters[argon].failures) == {"RuntimeError": 1}
        lines = main.run_report([], [], bot.research_health, [])
        assert f"| {sol} | 2 | 2 | 0 |" in lines and f"| {argon} | 1 | 0 | 1 (RuntimeError) |" in lines, lines
        # and the failure is annotated, since a thinned question stays green
        assert f"::warning title=Extra forecaster failed::{argon} failed 1 of 1 forecasts (RuntimeError x1)" in main.health_warnings(bot.research_health)
    finally:
        main.parse = real_parse

    # whole questions: the success bar counts the bot's own forecasts only. Extras that all
    # fail (no OpenRouter credit) thin a question but don't fail it, even with one Opus
    # forecast failing too; extras alone never carry a question
    def question_bot(own_replies, extras):
        with env(ANTHROPIC_API_KEY="x"):
            b = main.RcForecastBot(
                publish_reports_to_metaculus=False, skip_previously_forecasted_questions=False,
                predictions_per_research_report=len(own_replies) + len(extras), extra_forecasters=extras,
            )  # fmt: skip
        replies = iter(own_replies)

        def default(name, kind="llm"):
            if name != "default":
                return FakeLlm("parsed", model="anthropic/claude-haiku-5-5")
            llm = FakeLlm(model=opus)

            async def answer(prompt, system_prompt=None):
                reply = next(replies)
                if reply is None:
                    raise TimeoutError("opus timed out")
                return reply

            llm.invoke = answer  # type: ignore[method-assign]
            return llm

        async def research(question):
            return "notes"

        async def summary(question, research):
            return "summary"

        b.get_llm = default  # type: ignore[method-assign]
        b.run_research = research  # type: ignore[method-assign]
        b.summarize_research = summary  # type: ignore[method-assign]
        return b

    def dead(model):
        llm = FakeLlm(model=model)

        async def refuse(prompt, system_prompt=None):
            raise RuntimeError("402 This request requires more credits")

        llm.invoke = refuse  # type: ignore[method-assign]
        return llm

    real_parse = main.parse
    try:
        main.parse = fake_parse
        q = binary_question().model_copy(update={"page_url": "https://example.invalid/q/21"})
        for own_replies, extras in (
            (["Probability: 30%"] * 3, [dead(sol)] * 4),  # four dead extras: 3 of 7 succeed
            (["Probability: 30%", "Probability: 30%", None], [dead(sol)] * 2),  # and an Opus timeout
            (["Probability: 30%"], [dead(sol)]),  # one forecast and its one extra
        ):
            b = question_bot(own_replies, extras)
            [report] = asyncio.run(b.forecast_questions([q], return_exceptions=True))
            assert not isinstance(report, BaseException) and abs(report.prediction - 0.3) < 1e-9, report
        # the bot's own model fails (its key at its spend limit): no forecast from extras alone
        alive = FakeLlm("Probability: 60%", model=sol)
        b = question_bot([None, None, None], [alive, alive])
        [report] = asyncio.run(b.forecast_questions([q], return_exceptions=True))
        assert isinstance(report, BaseException) and "opus timed out" in str(report), report
        # two research reports: each report's last forecasts go to the extras
        b = question_bot(["x"] * 3, [alive, alive])
        assert [b._extra_for(slot) is alive for slot in range(1, 11)] == [False] * 3 + [True] * 2 + [False] * 3 + [True] * 2
        # a leg of a conditional question keeps its member's slot: no new slot, no tally
        leg = binary_question().model_copy(update={"conditional_type": "parent"})
        b = question_bot(["Probability: 30%"], [])
        b._note_pads.append(asyncio.run(b._initialize_notepad(leg)))
        asyncio.run(b._make_prediction(leg, "notes"))
        assert not b._forecasts_started and not b._own_succeeded, b._forecasts_started
    finally:
        main.parse = real_parse

    # the request: OpenRouter, Rob's own key, reasoning effort; OpenRouter's billed cost is
    # the call's cost, and spend_on keeps it apart from the Anthropic key's spend
    import httpx

    sent: list[dict] = []

    async def fake_send(self, request, *args, **kwargs):
        sent.append({"url": str(request.url), "auth": request.headers.get("authorization"), "body": json.loads(request.content)})  # fmt: skip
        return httpx.Response(200, request=request, json={
            "id": "gen-1", "object": "chat.completion", "created": 1, "model": "openai/gpt-6.1-sol",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Probability: 40%"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0012}})  # fmt: skip

    real_send = httpx.AsyncClient.send
    try:
        httpx.AsyncClient.send = fake_send  # type: ignore[method-assign]
        with env(OPENROUTER_API_KEY="sk-or-credits", EXTRA_FORECASTERS=argon, **own):
            llm = main.build_extra_forecasters()[0]
            with spend.collecting() as ledger:
                assert asyncio.run(llm.invoke("Forecast this.")) == "Probability: 40%"
    finally:
        httpx.AsyncClient.send = real_send  # type: ignore[method-assign]
    assert sent[0]["url"] == "https://openrouter.ai/api/v1/chat/completions", sent
    assert sent[0]["auth"] == "Bearer sk-or-own-fake", sent[0]["auth"]
    assert sent[0]["body"]["reasoning"] == {"effort": "high"} and "reasoning_effort" not in sent[0]["body"], sent
    assert sent[0]["body"]["max_tokens"] == main.ANTHROPIC_FORECAST_MAX_TOKENS, sent
    assert sent[0]["body"]["model"] == "google/gemini-4-argon", sent
    assert [(c.model, c.cost) for c in ledger.calls] == [(argon, 0.0012)], ledger.calls
    assert main.spend_on(ledger.calls, {argon}) == 0.0012 and main.spend_on(ledger.calls, {sol}) == 0.0
    print("extra forecasters: ok")


def check_scorecard() -> None:
    """scorecard.py over a review.json shaped like bot-review's."""
    import scorecard

    def binary(p, outcome, score):
        return {"question_type": "binary", "project_name": "MiniBench", "resolution": outcome,
                "forecast": {"probability_yes": p}, "scores": {"spot_peer_score": score}}  # fmt: skip

    numeric = {
        "question_type": "numeric", "project_name": "MiniBench", "resolution": "42",
        "forecast": {"median": 40}, "scores": {"spot_peer_score": -3.0},
        "traces": [{"forecasters": [
            {"prediction": "Probability distribution:\n- 10.00% chance of value below 30.0\n- 90.00% chance of value below 41.0"},
            {"prediction": "Probability distribution:\n- 10.00% chance of value below 35.0\n- 90.00% chance of value below 50.0"},
        ]}],
    }  # fmt: skip
    mc = {
        "question_type": "multiple_choice", "project_name": "FutureEval", "resolution": "b",
        "forecast": {"probabilities": {"a": 0.6, "b": 0.4}}, "scores": {"spot_peer_score": -10.0},
    }  # fmt: skip
    review = {"questions": [
        binary(0.05, "no", 5.0), binary(0.8, "yes", 12.0), binary(0.3, "yes", -20.0),
        binary(0.5, "annulled", None), numeric, mc,
    ]}  # fmt: skip
    text = "\n".join(scorecard.scorecard(review))
    for expected in (
        "| MiniBench | all types | 4 | -1.5 ± 6.9 | -6.0 |",
        "| MiniBench | binary | 3 | -1.0 ± 9.7 | -3.0 |",
        "| FutureEval | multiple_choice | 1 | -10.0 | -10.0 |",
        "3 resolved: mean forecast 38%, resolved Yes 67% (± 27%), Brier 0.177.",
        "| 0%–10% | 1 | 5% | 0% |",
        "2 forecaster ranges on 1 questions: outcome below the 10th percentile 0%, between the 10th and 90th 50%, above the 90th 50%",
        "1 resolved: mean probability on the winner 40%; the winner was the favourite 0% of the time.",
    ):
        assert expected in text, (expected, text)
    # values in scientific notation, outcomes beyond an open bound, and the run that was
    # scored rather than a later re-post
    tiny = "Probability distribution:\n- 10.00% chance of value below 1e-05\n- 90.00% chance of value below 2.5e-05"
    beyond = {**numeric, "resolution": "above_upper_bound", "spot_scoring_time": "2026-10-20T15:00:00Z",
              "traces": [
                  {"run_time": "2026-10-20T12:00:00Z", "forecasters": [{"prediction": tiny}]},
                  {"run_time": "2026-10-21T12:00:00Z", "forecasters": [{"prediction": "unparseable 1e"}]},
              ]}  # fmt: skip
    text = "\n".join(scorecard.scorecard({"questions": [beyond]}))
    assert "1 forecaster ranges on 1 questions: outcome below the 10th percentile 0%, between the 10th and 90th 0%, above the 90th 100%" in text, text
    empty = "\n".join(scorecard.scorecard({"questions": []}))
    assert "No resolved binary forecasts yet." in empty, empty
    assert "No scored questions yet." in empty and "No resolved yes/no or multiple-choice forecasts with their members yet." in empty, empty  # fmt: skip

    # kinds of question, by title (first match wins; status only for yes/no)
    def titled(title, kind="binary"):
        return {"title": title, "question_type": kind}
    for title, kind, expected in (
        ("Will the Seahawks beat the Broncos on Thursday Night Football?", "binary", "Sports"),
        ("Will Delta Air Lines (DAL) stock close above $85.00 on October 15?", "binary", "Prices and markets"),
        ("Will the U.S. naval blockade of Iran's crude oil exports still be in effect on October 15?", "binary", "Status on a date"),
        ("Will Zaporizhzhia be connected to offsite grid power on October 17?", "binary", "Status on a date"),
        ("What will U.S. spot Bitcoin ETFs' cumulative net inflow be as of October 15?", "numeric", "Prices and markets"),
        ("How many measles cases will Peru's CDC report as of mid-October?", "numeric", "Reported counts"),
        ("What will the Google Trends score for 'flu' be?", "numeric", "Google Trends"),
        ("Will the FAA type-certify the Boeing 737 MAX 10 before 2027?", "binary", "Other"),
        # words that mean something else outside a market or a game
        ("Will Iran close the Strait of Hormuz before December 1, 2026?", "binary", "Other"),
        ("Will Nintendo release a new Zelda game before 2027?", "binary", "Other"),
        ("Will the Fed's December decision match market expectations?", "binary", "Other"),
        ("Will Russia and Ukraine sign a peace settlement before 2027?", "binary", "Other"),
        ("How many gold medals will the US win at the 2028 Olympics?", "numeric", "Sports"),
        ("Will Bitcoin still be above $100,000 on November 30?", "binary", "Prices and markets"),
        ("When will the CDC report the first case?", "date", "Other"),
    ):  # fmt: skip
        assert scorecard.kind(titled(title, kind)) == expected, (title, scorecard.kind(titled(title, kind)))

    # members and models, from the "Model:" line main.py adds to each forecaster's entry;
    # the aggregate against the average member; each model against the most common one
    def tagged(p, model):
        return {"prediction": f"{p}%\nModel: {model}"}

    def run(forecasters, at="2026-10-31T12:00:05.0Z", posted="2026-10-31T12:00:01.0Z"):
        return {"forecast_time": posted, "traces": [{"run_time": at, "forecasters": forecasters}]}
    opus, gpt = "anthropic/claude-opus-5-5", "openrouter/openai/gpt-6.1-sol"
    q1 = {**binary(0.7, "yes", 4.0), **run([tagged(60.0, opus), tagged(70.0, opus), tagged(80.0, gpt)])}  # fmt: skip
    q2 = {**binary(0.2, "no", 2.0), **run([tagged(10.0, opus), tagged(30.0, opus), tagged(20.0, gpt)])}  # fmt: skip
    old = {**binary(0.5, "yes", 0.0), **run([{"prediction": "40.0%"}, {"prediction": "60.0%"}])}  # fmt: skip
    mc_members = {**mc, **run([
        {"prediction": "- a: 50.0%\n- b: 50.0%\n\nModel: " + opus},
        {"prediction": "- a: 70.0%\n- b: 30.0%\n\nModel: " + gpt},
    ])}  # fmt: skip
    lines = scorecard.members([q1, q2, old, mc_members])
    text = "\n".join(lines)
    ln = math.log
    gains = [
        100 * ln(0.7) - statistics.mean(100 * ln(x) for x in (0.6, 0.7, 0.8)),
        100 * ln(0.8) - statistics.mean(100 * ln(x) for x in (0.9, 0.7, 0.8)),
        100 * ln(0.5) - statistics.mean(100 * ln(x) for x in (0.4, 0.6)),
        100 * ln(0.4) - statistics.mean(100 * ln(x) for x in (0.5, 0.3)),
    ]
    assert f"| Aggregate − average member | 4 | {scorecard._mean_se(gains)} |" in text, (text, gains)
    diffs = [
        100 * ln(0.8) - statistics.mean(100 * ln(x) for x in (0.6, 0.7)),
        100 * ln(0.8) - statistics.mean(100 * ln(x) for x in (0.9, 0.7)),
        100 * ln(0.3) - 100 * ln(0.5),
    ]
    # one name per model on either route (openrouter/ dropped, claude-opus-5-5 as 5.5)
    assert f"| openai/gpt-6.1-sol − anthropic/claude-opus-5.5 | 3 | {scorecard._mean_se(diffs)} |" in text, (text, diffs)
    # one family so far: said so, no comparison row; Opus on either route is one model
    assert "One model so far (anthropic/claude-opus-5.5)" in "\n".join(scorecard.members([
        {**q1, **run([tagged(60.0, opus), tagged(70.0, "openrouter/anthropic/claude-opus-5.5")])}]))  # fmt: skip
    # two models that never share a question: said so
    assert "no questions in common" in "\n".join(scorecard.members([
        {**q1, **run([tagged(60.0, opus)])}, {**q2, **run([tagged(20.0, gpt)])}]))  # fmt: skip
    # a later run's forecast (the Cup, the Test Bot) isn't scored against an earlier run's
    # members; the members still count for the models
    later = {**q1, **run([tagged(30.0, opus), tagged(30.0, opus)], posted="2026-11-02T09:00:00Z")}
    assert "Aggregate" not in "\n".join(scorecard.members([later])), scorecard.members([later])
    # setups: the "Model:" line marks the 2026-10-30 rules; untagged runs are older
    assert scorecard.setup(q1) == "since 2026-10-30" and scorecard.setup(old) == "before 2026-10-30"
    text = "\n".join(scorecard.scorecard({"questions": [q1, q2, old]}))
    assert "| since 2026-10-30 | binary | 2 | +3.0 ± 1.0 |" in text and "| before 2026-10-30 | binary | 1 | +0.0 |" in text, text
    assert "- since 2026-10-30: 2 resolved: mean forecast 45%" in text, text
    print("scorecard: ok")


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


class FakeLlm:
    """A forecaster whose reply is fixed, for the prompt-to-forecast methods."""

    def __init__(self, reply: str = "Reasoning...", model: str = "fake/forecaster") -> None:
        self.reply = reply
        self.model = model

    async def invoke(self, prompt, system_prompt=None):
        return self.reply


def check_guards() -> None:
    """forecast_guards.py: the forecasting-tools 0.2.92 answers that sank a whole question
    (reproduced offline on 2026-10-28), and the numeric mixture."""
    import faulthandler

    import forecast_guards as fg
    from forecasting_tools import NumericDistribution, Percentile

    heights = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    q = NumericQuestion(
        question_text="How many?", page_url="https://example.invalid/q/5", lower_bound=0,
        upper_bound=100, open_lower_bound=True, open_upper_bound=True, zero_point=None,
        cdf_size=201,
    )  # fmt: skip

    def pcts(values):
        return [Percentile(percentile=h, value=v) for h, v in zip(heights, values)]

    def dist(values, question=q):
        return NumericDistribution.from_question(pcts(values), question)

    # P10 = P20 at an open lower bound with a concentrated middle: 0.2.92's CDF loop never
    # ended and froze the run. Bounded, it raises (a watchdog kills the check if not)
    faulthandler.dump_traceback_later(60, exit=True)
    try:
        dist([0, 0, 5, 5.2, 20, 30]).get_cdf()
    except ValueError as e:
        assert "Could not standardize CDF" in str(e), e
    else:
        raise AssertionError("a CDF that isn't increasing must raise")
    finally:
        faulthandler.cancel_dump_traceback_later()

    # spreading the repeat keeps the CDF increasing, so that answer forecasts normally
    spread = fg.spread_repeats(pcts([0, 0, 5, 5.2, 20, 30]), q)
    assert all(close(p.value, v) for p, v in zip(spread, [0, 1e-4, 5, 5.2, 20, 30])), spread
    cdf = NumericDistribution.from_question(spread, q).get_cdf()
    assert all(b.percentile > a.percentile for a, b in zip(cdf, cdf[1:]))
    below = fg.spread_repeats(pcts([-5, -5, -5, 3, 9, 12]), q)
    assert all(close(p.value, v) for p, v in zip(below, [-5, -5 + 1e-4, -5 + 2e-4])), below
    # repeats inside the range, and repeats at a closed bound, are the package's to handle
    # log-scaled, 1 to 1e9: a step in log space, so the repeat stays below the next value
    log = NumericQuestion(**{**q.model_dump(), "lower_bound": 1, "upper_bound": 1e9, "zero_point": 0})
    logged = fg.spread_repeats(pcts([1, 1, 3, 10, 100, 1000]), log)
    assert 1 < logged[1].value < 1.0001 and logged[2].value == 3, logged
    assert NumericDistribution.from_question(logged, log).get_cdf()
    inside = pcts([10, 10, 20, 30, 40, 50])
    assert fg.spread_repeats(inside, q) == inside
    closed = q.model_copy(update={"open_lower_bound": False})
    assert fg.spread_repeats(pcts([0, 0, 5, 9, 20, 30]), closed) == pcts([0, 0, 5, 9, 20, 30])

    # through the bot: the answer that used to fail now forecasts, and an answer that still
    # can't make a CDF fails only its own forecast (the CDF is built before aggregation)
    with env(ANTHROPIC_API_KEY="x"):
        bot = main.RcForecastBot(publish_reports_to_metaculus=False)
    bot.get_llm = lambda name, kind="llm": FakeLlm()  # type: ignore[method-assign]
    parsed: list = []

    async def fake_parse(*args, **kwargs):
        return parsed[0]

    real_parse = main.parse
    try:
        main.parse = fake_parse
        parsed[:] = [pcts([0, 0, 5, 5.2, 20, 30])]
        got = asyncio.run(bot._numeric_prompt_to_forecast(q, "prompt"))
        assert got.prediction_value.get_cdf(), got
        assert main.model_of_reasoning(got.reasoning) == "fake/forecaster", got.reasoning
        # a date forecast runs end to end too, and names its model
        when = timed_question("date")
        parsed[:] = [[main.DatePercentile(percentile=p, value=datetime(2026, 11, d, tzinfo=timezone.utc))
                      for p, d in zip([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], [2, 5, 9, 14, 20, 27])]]  # fmt: skip
        got_date = asyncio.run(bot._date_prompt_to_forecast(when, "prompt"))
        assert got_date.prediction_value.get_cdf() and main.model_of_reasoning(got_date.reasoning) == "fake/forecaster"

        # three forecasts: the mixture is their mean CDF, wider than the median where they
        # disagree; a misread outlier is left out
        wide = NumericQuestion(**{**q.model_dump(), "upper_bound": 1000})
        a, b, c = (dist(v, wide) for v in ([10, 20, 30, 40, 50, 60], [20, 30, 40, 50, 60, 70], [30, 40, 50, 60, 70, 80]))  # fmt: skip
        mixed = asyncio.run(bot._aggregate_predictions([a, b, c], wide))
        mean = [sum(t) / 3 for t in zip(*([p.percentile for p in d.get_cdf()] for d in (a, b, c)))]
        # (the declared CDF; posting standardises it again, as it does the package's median)
        assert all(abs(m.percentile - x) < 1e-6 for m, x in zip(mixed.declared_percentiles, mean))
        # ordinary disagreement, ranges that overlap, stays in
        d = dist([45, 60, 70, 80, 90, 100], wide)
        mixed = asyncio.run(bot._aggregate_predictions([a, b, d], wide))
        three = [sum(t) / 3 for t in zip(*([p.percentile for p in x.get_cdf()] for x in (a, b, d)))]
        assert all(abs(m.percentile - x) < 1e-6 for m, x in zip(mixed.declared_percentiles, three))
        far = dist([600, 700, 750, 800, 850, 900], wide)  # the same forecast in the wrong unit
        mixed = asyncio.run(bot._aggregate_predictions([a, b, far], wide))
        both = [sum(t) / 2 for t in zip(*([p.percentile for p in d.get_cdf()] for d in (a, b)))]
        assert all(abs(m.percentile - x) < 1e-6 for m, x in zip(mixed.declared_percentiles, both))
        # two forecasts: nothing to judge an outlier by, so both count
        mixed = asyncio.run(bot._aggregate_predictions([a, far], wide))
        assert mixed.declared_percentiles[100].percentile < a.get_cdf()[100].percentile

        # multiple choice: a confident answer with six options at 0%, which 0.2.92's
        # validator rejected on every retry, now posts with each long shot at 1%
        options = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]
        mc = MultipleChoiceQuestion(question_text="Which?", options=options)
        sharp = [0.92, 0.08] + [0.0] * 6
        parsed[:] = [[fg.OptionGuess(option_name=o, probability=p) for o, p in zip(options, sharp)]]
        got = asyncio.run(bot._multiple_choice_prompt_to_forecast(mc, "prompt"))
        assert main.model_of_reasoning(got.reasoning) == "fake/forecaster", got.reasoning
        probs = got.prediction_value.to_dict()
        assert list(probs) == options and abs(sum(probs.values()) - 1) < 1e-9, probs
        assert all(abs(probs[o] - 0.01) < 1e-9 for o in options[2:]), probs
        assert abs(probs["Alpha"] - 0.92 * 0.94) < 1e-9, probs
    finally:
        main.parse = real_parse

    def guesses(pairs):
        return [fg.OptionGuess(option_name=n, probability=p) for n, p in pairs]

    two = ["Yes, before May", "No"]
    # names as forecasters write them; percentages; a sum a little off
    got = fg.option_list(guesses([(" yes,  before MAY ", 61), ("Option: No.", 41)]), two).to_dict()
    assert abs(got["Yes, before May"] - 61 / 102) < 1e-9 and abs(got["No"] - 41 / 102) < 1e-9
    for bad, needle in (
        (guesses([("No", 1.0)]), "no probability for option"),
        (guesses([("Yes, before May", 0.9), ("No", 0.6)]), "sum to 150%"),
    ):
        try:
            fg.option_list(bad, two)
        except ValueError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"{bad} must fail its forecast")
    print("guards: ok")


def anthropic_search_reply(stop_reason: str = "end_turn") -> dict:
    """A web-search reply as Anthropic's API sends it: narration, a search, cited text."""
    url = "https://www.reuters.com/world/example-2026-10-27/"
    return {
        "id": "msg_fake", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [
            {"type": "text", "text": "I'll search for this."},
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": url, "title": "Reuters", "encrypted_content": "e"}]},
            {"type": "text", "text": " The ministry announced the delay on 2026-10-27.", "citations": [
                {"type": "web_search_result_location", "url": url, "title": "Reuters",
                 "encrypted_index": "i", "cited_text": "delay"}]},
        ],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 3000, "cache_read_input_tokens": 40000,
                  "cache_creation_input_tokens": 20000, "output_tokens": 2000,
                  "server_tool_use": {"web_search_requests": 4}},
    }  # fmt: skip


def cut_short_stops(*stops: str) -> list[str]:
    """The stop reasons a cut-short check can run with. litellm 1.80.10, which the runs and
    CI install, keeps Anthropic's own; 1.102.1 maps pause_turn and refusal to others, so
    off CI, on a virtualenv drifted from the lock, those two wait for checks.yaml. On CI none
    is skipped: a lock that moved to such a version would fail here, as it should."""
    from litellm.litellm_core_utils.core_helpers import map_finish_reason

    return [stop for stop in stops if os.getenv("CI") or map_finish_reason(stop) in main.CUT_SHORT]


def anthropic_limit_reply() -> dict:
    """A research reply as Anthropic's API sends it when lookups run into their limits: one
    search that found pages, one past its max_uses, a fetch, and code that printed the
    limit. Its secrets are the ones a trace must never log."""
    url = "https://www.reuters.com/world/example-2026-10-27/"
    reply = anthropic_search_reply()
    reply["content"] = [
        {"type": "thinking", "thinking": "THINKING-TEXT", "signature": "s"},
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "polymarket"}},
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "url": url, "title": "Reuters", "encrypted_content": "ENCRYPTED-TEXT"}]},
        {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "kalshi"}},
        {"type": "web_search_tool_result", "tool_use_id": "s2", "content": {
            "type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
        {"type": "server_tool_use", "id": "f1", "name": "web_fetch", "input": {"url": url}},
        {"type": "web_fetch_tool_result", "tool_use_id": "f1", "content": {
            "type": "web_fetch_result", "url": url, "retrieved_at": "2026-10-30T12:00:00Z",
            "content": {"type": "document", "title": "Reuters", "citations": {"enabled": True},
                        "source": {"type": "text", "media_type": "text/plain", "data": "DOCUMENT-TEXT"}}}},
        {"type": "server_tool_use", "id": "c1", "name": "code_execution", "input": {"code": "print(1)"}},
        {"type": "code_execution_tool_result", "tool_use_id": "c1", "content": {
            "type": "code_execution_result", "stdout": "Server tool use limit exceeded",
            "stderr": "", "return_code": 1, "content": []}},
        {"type": "text", "text": "REPORT-TEXT, sources thin.\n\nLookups: 1 worked, 2 failed."},
    ]  # fmt: skip
    return reply


def check_tool_census() -> None:
    """What the web tools did, through real litellm with only its HTTP layer faked: this
    is what proves the tool_calls, the web search results and the logger_fn route on
    1.80.10, in checks.yaml. The trace shows the blocks litellm drops, and nothing private."""
    import httpx

    key = "sk-ant-FAKE-never-logged"
    sent: list[dict] = []

    async def fake_send(self, request, *args, **kwargs):
        sent.append(json.loads(request.content))
        return httpx.Response(200, request=request, json=anthropic_limit_reply())

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    level = spend.logger.level
    real_send = httpx.AsyncClient.send
    spend.logger.addHandler(handler)
    spend.logger.setLevel(logging.INFO)
    try:
        httpx.AsyncClient.send = fake_send  # type: ignore[method-assign]
        with env(ANTHROPIC_API_KEY=key):
            llm = main.build_llms("anthropic")["researcher"]
            llm.allowed_tries = 1
            with spend.collecting() as ledger:
                asyncio.run(llm.invoke("Research this question."))
            call = ledger.calls[0]
            assert (call.search_tries, call.search_pages, call.search_errors) == (2, 1, ("max_uses_exceeded",)), call
            assert (call.fetch_tries, call.code_runs, call.tally) == (1, 1, (1, 2)), call
            assert call.answered and call.searches == 4, call
            assert not any("Research trace:" in r.getMessage() for r in records), "trace is off here"
            spend_line = next(r.getMessage() for r in records if r.getMessage().startswith("Spend: "))
            assert (
                " | tools: 2 searches tried, 1 with pages, 1 fetches tried, 1 code runs, errors"
                " max_uses_exceeded, tally (1, 2), stop " in spend_line
            ), spend_line

        records.clear()
        with env(ANTHROPIC_API_KEY=key, TRIAL_RAW_TRACE="true"):
            llm = main.build_llms("anthropic")["researcher"]
            llm.allowed_tries = 1
            with spend.collecting():
                asyncio.run(llm.invoke("Research this question."))
    finally:
        httpx.AsyncClient.send = real_send  # type: ignore[method-assign]
        spend.logger.setLevel(level)
        spend.logger.removeHandler(handler)
    assert "logger_fn" not in sent[-1], sent[-1]  # stays on our side of the request
    logged = [r.getMessage() for r in records]
    traces = [line for line in logged if line.startswith("Research trace: ")]
    assert len(traces) == 1, logged
    assert all(
        secret not in line
        for line in logged
        for secret in (key, "ENCRYPTED-TEXT", "DOCUMENT-TEXT", "THINKING-TEXT", "REPORT-TEXT")
    ), logged
    trace = json.loads(traces[0].removeprefix("Research trace: "))
    assert trace["stop_reason"] == "end_turn" and trace["container"] is False, trace
    assert trace["usage"]["server_tool_use"] == {"web_search_requests": 4}, trace
    by_type: dict[str, list[dict]] = {}
    for block in trace["blocks"]:
        by_type.setdefault(block["type"], []).append(block)
    assert [b["name"] for b in by_type["server_tool_use"]] == ["web_search", "web_search", "web_fetch", "code_execution"]
    assert "polymarket" in by_type["server_tool_use"][0]["input"], trace
    assert [b.get("results") for b in by_type["web_search_tool_result"]] == [1, None], trace
    assert by_type["web_search_tool_result"][1]["error_code"] == "max_uses_exceeded", trace
    assert by_type["web_fetch_tool_result"][0]["result_type"] == "web_fetch_result", trace
    assert by_type["code_execution_tool_result"] == [{
        "type": "code_execution_tool_result", "result_type": "code_execution_result", "return_code": 1,
        "stdout": "Server tool use limit exceeded", "stderr": ""}], trace  # fmt: skip
    # a long print is cut, and a reply the trace can't read costs only the trace
    records.clear()
    spend.logger.addHandler(handler)
    spend.logger.setLevel(logging.INFO)
    try:
        spend.log_trace(json.dumps({"content": [
            {"type": "code_execution_tool_result", "content": {"stdout": "x" * 900, "stderr": "y" * 900}},
            {"type": "server_tool_use", "name": "web_search", "caller": {"type": "direct"}, "input": {"query": "z" * 900}},
        ]}))  # fmt: skip

        async def reply(**kwargs):
            kwargs["logger_fn"]({"original_response": "not json", "api_key": key})
            return FakeResponse()

        with env(TRIAL_RAW_TRACE="true"):
            asyncio.run(spend.metered(reply)(model="m", tools=[{"name": "web_search"}]))
    finally:
        spend.logger.setLevel(level)
        spend.logger.removeHandler(handler)
    blocks = json.loads(records[0].getMessage().removeprefix("Research trace: "))["blocks"]
    assert (len(blocks[0]["stdout"]), len(blocks[0]["stderr"]), len(blocks[1]["input"])) == (400, 400, 400), blocks
    assert blocks[1]["caller"] == {"type": "direct"}, blocks
    assert records[-1].getMessage() == "Spend: could not trace a reply (JSONDecodeError)", records[-1]
    print("tool census: ok")


def check_own_key_research() -> None:
    """Research on Rob's key through litellm 1.80.10 with only its HTTP layer faked: the
    request carries the direct search and fetch tools, the note and one cache marker; the
    reply's sources reach the research; a reply cut short fails; cached tokens are priced
    as such."""
    import json

    import httpx

    sent: list[dict] = []
    replies: list[dict] = []

    async def fake_send(self, request, *args, **kwargs):
        assert "api.anthropic.com" in str(request.url), request.url
        sent.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx.Response(200, request=request, json=replies.pop(0))

    real_send = httpx.AsyncClient.send
    try:
        httpx.AsyncClient.send = fake_send  # type: ignore[method-assign]
        with env(ANTHROPIC_API_KEY="sk-ant-fake"):
            llm = main.build_llms("anthropic")["researcher"]
            llm.allowed_tries = 1  # no retry backoff in the check
            replies.append(anthropic_search_reply())
            with spend.collecting() as ledger:
                text = asyncio.run(llm.invoke("Research this question."))
            body = sent[0]["body"]
            assert body["tools"] == [
                {"type": "web_search_20260209", "name": "web_search", "max_uses": 5,
                 "allowed_callers": ["direct"]},
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 2,
                 "max_content_tokens": 8000, "citations": {"enabled": True},
                 "allowed_callers": ["direct"]},
            ], body["tools"]  # fmt: skip
            # litellm 1.80.10 names the fetch beta in place of any other; the Test Bot
            # confirms Anthropic accepts it (RESEARCH_WEB_FETCH=false turns fetch off)
            assert sent[0]["headers"].get("anthropic-beta") == "web-fetch-2025-09-10", sent[0]["headers"]
            assert body["messages"] == [{"role": "user", "content": [
                {"type": "text", "text": "Research this question.", "cache_control": {"type": "ephemeral"}}]}], body["messages"]  # fmt: skip
            assert "5 web searches and 2 page fetches" in json.dumps(body["system"]), body
            assert "temperature" not in body and "thinking" not in body, body
            # the cited source survives as a Sources footer (it used to be dropped)
            assert "The ministry announced the delay on 2026-10-27." in text, text
            assert "https://www.reuters.com/world/example-2026-10-27/" in text, text
            # tokens at their cached prices plus $0.01 a search
            call = ledger.calls[0]
            assert (call.cache_reads, call.cache_writes, call.searches) == (40000, 20000, 4), call
            expected = 3000 * 2e-6 + 40000 * 2e-7 + 20000 * 2.5e-6 + 2000 * 1e-5 + 4 * 0.01
            assert abs(call.cost - expected) < 1e-9, (call.cost, expected)

            # the way back: filtered:4 is the request as it ran before 2026-11-04
            with env(ANTHROPIC_API_KEY="sk-ant-fake", RESEARCH_TOOLS="filtered:4"):
                old_tools = main.build_llms("anthropic")["researcher"]
                old_tools.allowed_tries = 1
                replies.append(anthropic_search_reply())
                asyncio.run(old_tools.invoke("Research this question."))
            body = sent[-1]["body"]
            assert body["tools"] == [
                {"type": "web_search_20260209", "name": "web_search", "max_uses": 4},
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 2,
                 "max_content_tokens": 8000, "citations": {"enabled": True}},
            ], body["tools"]  # fmt: skip
            assert "system" not in body, body

            # a reply whose server-side search loop paused, ran out of tokens or was refused
            # holds only narration: it comes back empty (run_research counts that as no
            # research) and is not retried, since a retry reruns the whole search
            llm.allowed_tries = 2
            for stop in cut_short_stops("pause_turn", "max_tokens", "refusal"):
                sent.clear()
                replies[:] = [anthropic_search_reply(stop), anthropic_search_reply()]
                assert asyncio.run(llm.invoke("Research this question.")) == "", stop
                assert len(sent) == 1, (stop, len(sent))
            replies.clear()
            llm.allowed_tries = 1

            # nothing cited (a reply can cite none): the pages the searches returned stand
            # in as the sources
            uncited = anthropic_search_reply()
            uncited["content"][-1].pop("citations")
            replies.append(uncited)
            text = asyncio.run(llm.invoke("Research this question."))
            assert "https://www.reuters.com/world/example-2026-10-27/" in text, text

            # a page opened with web fetch: litellm drops the fetch result and a citation of
            # the page has no URL, so the fetched URL comes from the fetch call. It is
            # listed first, and an uncited search result is not passed off as the source
            page = "https://www.iaea.org/newscenter/update-2026-10-29"
            fetch_blocks = [
                {"type": "server_tool_use", "id": "f1", "name": "web_fetch", "input": {"url": page}},
                {"type": "web_fetch_tool_result", "tool_use_id": "f1", "content": {
                    "type": "web_fetch_result", "url": page, "retrieved_at": "2026-10-30T12:00:00Z",
                    "content": {"type": "document", "title": "IAEA update", "citations": {"enabled": True},
                                "source": {"type": "text", "media_type": "text/plain", "data": "Offsite power restored."}}}},
                {"type": "text", "text": " Offsite power was restored on 2026-10-29.", "citations": [
                    {"type": "char_location", "cited_text": "Offsite power restored.", "document_index": 0,
                     "document_title": "IAEA update", "start_char_index": 0, "end_char_index": 23}]},
            ]  # fmt: skip
            searched_and_fetched = anthropic_search_reply()
            searched_and_fetched["content"][-1].pop("citations")  # the search result isn't cited
            searched_and_fetched["content"] += fetch_blocks
            fetch_only = anthropic_search_reply()
            fetch_only["content"] = fetch_only["content"][:1] + fetch_blocks
            for reply in (searched_and_fetched, fetch_only):
                replies.append(reply)
                text = asyncio.run(llm.invoke("Research this question."))
                assert page in text and "Offsite power was restored on 2026-10-29." in text, text
                assert "reuters.com" not in text.split("2026-10-29.")[-1], text

            # a reply cut short with no text at all (thinking and tool blocks only) fails
            # GeneralLlm's own check before the stop reason is read: empty, not retried
            llm.allowed_tries = 2
            for stop in cut_short_stops("pause_turn", "max_tokens"):
                sent.clear()
                silent = anthropic_search_reply(stop)
                silent["content"] = [{"type": "thinking", "thinking": "", "signature": "s"}] + silent["content"][1:3]
                replies[:] = [silent, anthropic_search_reply()]
                assert asyncio.run(llm.invoke("Research this question.")) == "", stop
                assert len(sent) == 1, (stop, len(sent))
            replies.clear()
            llm.allowed_tries = 1

            # an Opus 5.5 forecast: effort high in output_config, the larger reply cap, no
            # thinking budget or sampling parameters (Opus 5.5 rejects both)
            sent.clear()
            replies.append({
                "id": "msg_fake", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "thinking", "thinking": "", "signature": "sig"},
                            {"type": "text", "text": "Probability: 35%"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 5000, "output_tokens": 3000},
            })  # fmt: skip
            forecaster = main.build_llms("anthropic", budget.TIERS["opus-3"])["default"]
            assert asyncio.run(forecaster.invoke("Forecast this.")) == "Probability: 35%"
            body = sent[0]["body"]
            assert body["model"] == "claude-opus-5-5" and body["output_config"] == {"effort": "high"}, body  # fmt: skip
            assert body["max_tokens"] == main.ANTHROPIC_FORECAST_MAX_TOKENS, body
            for absent in ("thinking", "temperature", "top_p", "reasoning_effort"):
                assert absent not in body, (absent, body)
    finally:
        httpx.AsyncClient.send = real_send  # type: ignore[method-assign]
    print("own-key research: ok")


def check_research_tools() -> None:
    """RESEARCH_TOOLS, and the Test Bot's input for it. The workflows pass both variables,
    and only the tournament keeps the watch's window between runs."""
    search = {"type": "web_search_20260209", "name": "web_search", "max_uses": 4}
    fetch = {
        "type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 2,
        "max_content_tokens": 8000, "citations": {"enabled": True},
    }  # fmt: skip
    with env():
        assert main.research_tools() == ("direct", 5)
        assert main.anthropic_research_tools() == [main.ANTHROPIC_SEARCH_TOOL, main.ANTHROPIC_FETCH_TOOL]
    with env(RESEARCH_TOOLS="basic:5"):
        tools = main.anthropic_research_tools()
        assert [t["type"] for t in tools] == ["web_search_20250305", "web_fetch_20250910"], tools
        assert tools[0]["max_uses"] == 5 and all("allowed_callers" not in t for t in tools), tools
    with env(RESEARCH_TOOLS="filtered:4"):
        assert main.anthropic_research_tools() == [search, fetch]
    with env(RESEARCH_TOOLS="direct:4"):
        tools = main.anthropic_research_tools()
        assert tools[0]["max_uses"] == 4 and tools[0]["allowed_callers"] == ["direct"], tools
        assert tools[1] == main.ANTHROPIC_FETCH_TOOL, tools
    with env(RESEARCH_TOOLS="filtered:4", RESEARCH_WEB_FETCH="false"):
        assert main.anthropic_research_tools() == [search]
    with env(RESEARCH_TOOLS=" Direct:5 "):
        assert main.research_tools() == ("direct", 5)
    assert main.ANTHROPIC_SEARCH_TOOL["max_uses"] == 5, "a mode must not change the constants"
    assert main.ANTHROPIC_SEARCH_TOOL["allowed_callers"] == ["direct"]
    for bad in ("direct:0", "direct:7", "direct", "fast", "fast:5", "direct:five", "direct:²"):
        with env(RESEARCH_TOOLS=bad):
            try:
                main.research_tools()
            except SystemExit as e:
                assert "RESEARCH_TOOLS must be direct, basic or filtered" in str(e), e
                assert repr(bad) in str(e), e
            else:
                raise AssertionError(f"{bad} must stop the run")
    # the Test Bot's input wins, and main() drops it on any other run, like the model trials
    with env(RESEARCH_TOOLS="basic:5", TRIAL_RESEARCH_TOOLS="filtered:4"):
        assert main.research_tools() == ("filtered", 4)
    assert main.TRIAL_MODELS["tools"] == "TRIAL_RESEARCH_TOOLS"

    workflows = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    tournament, test_bot, cup = (
        (workflows / name).read_text(encoding="utf-8")
        for name in ("run_bot_on_tournament.yaml", "test_bot.yaml", "run_bot_on_metaculus_cup.yaml")
    )
    for workflow in (tournament, test_bot, cup):
        assert "RESEARCH_TOOLS: ${{ vars.RESEARCH_TOOLS }}" in workflow
        assert "RESEARCH_TOOL_NOTE: ${{ vars.RESEARCH_TOOL_NOTE }}" in workflow
    for name in ("RESEARCH_WATCH_FILE", "actions/cache/restore", "actions/cache/save"):
        assert name in tournament and name not in test_bot and name not in cup, name
    assert "continue-on-error: true" in tournament
    for name in (spend.TRACE_VARIABLE, "TRIAL_RESEARCH_TOOLS"):
        assert name in test_bot and name not in tournament and name not in cup, name
    print("research tools: ok")


def check_tool_rules() -> None:
    """The system note: built from the tools' own numbers, on this route alone, and never
    in the way of the cache marker."""
    marker = {"type": "ephemeral"}

    def research_messages():
        llm = main.build_llms("anthropic")["researcher"]
        messages = llm.model_input_to_message("Research this question.")
        assert messages[-1]["content"] == [
            {"type": "text", "text": "Research this question.", "cache_control": marker}
        ], messages
        return messages

    with env(ANTHROPIC_API_KEY="x"):
        system, _ = research_messages()
        assert system["role"] == "system", system
        assert system["content"] == (
            "You have 5 web searches and 2 page fetches for this question, and no more. A fetch"
            " that fails still counts. Make one call at a time and read each result before the next.\n\n"
            "Spend them in this order:\n"
            "1. If the resolution criteria or fine print name a source or link to one, open it"
            " first: fetch its link, or spend one search finding it.\n"
            "2. Forecasts elsewhere: at most two searches, one naming Polymarket and Kalshi, one"
            " naming Manifold and Metaculus, not one per venue.\n"
            "3. The base rate or past values the brief asks for.\n"
            "4. With what is left, the latest value and volatility of a data series the question"
            " turns on, then other news.\n\n"
            "If a tool says it has reached its limit, stop calling it and write the report from"
            " what you have. Write nothing between calls: everything goes in one final report.\n\n"
            'A lookup that returned an error or was refused is not a finding: write "lookup failed"'
            ' for it, never "not found". End the report with one line in exactly this form,'
            " counting every search and fetch you tried: Lookups: N worked, M failed."
        ), system["content"]
        # a line the note asks for is a line spend.py reads
        assert spend.TALLY.findall(system["content"].replace("N", "1").replace("M", "2"))[-1] == ("1", "2")
    # the numbers are the tools' own
    with env(ANTHROPIC_API_KEY="x", RESEARCH_TOOLS="basic:3"):
        assert "You have 3 web searches and 2 page fetches for" in research_messages()[0]["content"]
    with env(ANTHROPIC_API_KEY="x", RESEARCH_TOOLS="direct:1"):
        assert "You have 1 web search and 2 page fetches for" in research_messages()[0]["content"]
    assert "You have 9 web searches and 7 page fetches for" in main.tool_rules(
        [{"name": "web_search", "max_uses": 9}, {"name": "web_fetch", "max_uses": 7}]
    )
    assert main.tool_rules([]) is None
    # fetch off: search alone, in every line
    with env(ANTHROPIC_API_KEY="x", RESEARCH_WEB_FETCH="false"):
        note = research_messages()[0]["content"]
        assert "fetch" not in note.lower(), note
        assert note.startswith(
            "You have 5 web searches for this question, and no more. Make one call at a time"
            " and read each result before the next.\n\n"
        ), note
        assert (
            "1. If the resolution criteria or fine print name a source or link to one, spend"
            " one search finding its latest reading first.\n" in note
        ), note
        assert "counting every search you tried: Lookups: N worked, M failed." in note, note
    with env(ANTHROPIC_API_KEY="x", RESEARCH_WEB_FETCH="false", RESEARCH_TOOLS="direct:1"):
        assert research_messages()[0]["content"].startswith("You have 1 web search for this")
    # no note at all: one user message, as before
    for setting in ({"RESEARCH_TOOL_NOTE": "false"}, {"RESEARCH_TOOLS": "filtered:4"}):
        with env(ANTHROPIC_API_KEY="x", **setting):
            messages = research_messages()
            assert [m["role"] for m in messages] == ["user"], (setting, messages)
    # a list of messages keeps its own shape, and the other researcher gets no note
    with env(ANTHROPIC_API_KEY="x"):
        llm = main.build_llms("anthropic")["researcher"]
        given = [{"role": "user", "content": "Hello"}]
        assert [m["role"] for m in llm.model_input_to_message(given)] == ["user"]
    with env(OPENROUTER_API_KEY="x"):
        llm = main.build_llms("openrouter")["researcher"]
        assert llm.model_input_to_message("Research this question.") == [
            {"role": "user", "content": "Research this question."}
        ]
    print("tool rules: ok")


def research_call(**fields) -> spend.Call:
    """A healthy research reply on Rob's key, but for the fields given."""
    values = dict(
        question="https://www.metaculus.com/questions/46392/",
        part="research: anthropic/claude-sonnet-5-5 + web search and fetch",
        model="anthropic/claude-sonnet-5-5", cost=0.2, basis="litellm", generation=None,
        prompt_tokens=40000, cache_reads=20000, searches=5, search_tries=5, search_pages=5,
        fetch_tries=1, stop="stop",
    )  # fmt: skip
    return spend.Call(**{**values, **fields})


def check_research_watch() -> None:
    """Which reports count as degraded, the window kept between runs, and what the run page
    and the alert get. No question text in the file; a failure here can't touch the run."""
    reason = research_watch.reason
    assert reason(research_call()) is None
    # each reason in turn, the first that applies winning
    for stop in research_watch.CUT_SHORT:
        assert reason(research_call(stop=stop)) == "stopped early", stop
    assert reason(research_call(answered=False)) == "stopped early"
    assert reason(research_call(answered=False, search_pages=0)) == "stopped early"
    assert reason(research_call(search_pages=0)) == "searches failed"
    errors = ("too_many_requests",) * 3
    assert reason(research_call(search_pages=2, search_errors=errors)) == "searches failed"
    assert reason(research_call(search_pages=3, search_errors=errors)) is None
    # billed searches that left under 15,000 fresh tokens never reached the reply
    thin = dict(prompt_tokens=34999, cache_reads=20000)
    assert reason(research_call(**thin)) == "results never reached the reply"
    assert reason(research_call(prompt_tokens=35000, cache_reads=20000)) is None
    assert reason(research_call(searches=3, **thin)) == "results never reached the reply"
    assert reason(research_call(searches=2, **thin)) is None  # too few billed to tell
    assert reason(research_call(tally=(2, 3))) == "most lookups failed"
    assert reason(research_call(tally=(0, 0))) == "most lookups failed"
    assert reason(research_call(tally=(3, 3))) is None
    assert reason(research_call(tally=(5, 0))) is None

    # only research on Rob's key
    picked = research_call()
    assert research_watch.replies([
        picked,
        research_call(part="forecast 1"),
        research_call(model="openrouter/anthropic/claude-sonnet-5.5:online"),
    ]) == [picked]  # fmt: skip

    bad = research_call(search_pages=0, search_errors=("too_many_requests",))
    good = research_call()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "watch.json")
        marker = path + ".updated"
        # nothing to report: None, and the file and its marker stay away
        assert research_watch.update([], 0, path) is None
        assert not os.path.exists(path) and not os.path.exists(marker)

        # one run: a window not yet full
        watch = research_watch.update([good, bad, good], 0, path)
        assert (watch.reports, watch.degraded, watch.bad, watch.window) == (3, ["searches failed"], 1, 3), watch
        assert not watch.tripped
        assert watch.line == (
            "- Web research on your key: 3 reports; 10 of 15 searches brought back pages"
            " (too_many_requests ×1); 3 fetches tried; 0 code runs; 1 degraded (46392:"
            " searches failed). Last 3 reports (window not full): 1 degraded."
        ), watch.line
        assert watch.warning == (
            "::warning title=Research degraded::1 of 3 research reports this run were degraded"
            " (searches failed)"
        ), watch.warning
        assert json.loads(Path(path).read_text(encoding="utf-8")) == {"window": [0, 1, 0]}
        assert os.path.getsize(marker) == 0
        assert "46392" not in Path(path).read_text(encoding="utf-8")  # no question text

        # the next run reads it; a clean run gives no warning
        os.remove(marker)
        watch = research_watch.update([good], 0, path)
        assert (watch.reports, watch.degraded, watch.warning) == (1, [], ""), watch
        assert (watch.bad, watch.window) == (1, 4), watch
        assert os.path.exists(marker)
        assert watch.line == (
            "- Web research on your key: 1 report; 5 of 5 searches brought back pages; 1 fetch"
            " tried; 0 code runs; 0 degraded. Last 4 reports (window not full): 1 degraded."
        ), watch.line

        # an idle run leaves the file and the marker as they were, so nothing is saved
        os.remove(marker)
        before = Path(path).read_text(encoding="utf-8")
        assert research_watch.update([], 0, path) is None
        assert not os.path.exists(marker) and Path(path).read_text(encoding="utf-8") == before

        # raised calls count as degraded reports, and say so
        watch = research_watch.update([], 2, path)
        assert (watch.reports, watch.degraded, watch.bad, watch.window) == (2, ["call raised"] * 2, 3, 6), watch
        assert watch.line.endswith(
            "0 degraded; 2 raised. Last 6 reports (window not full): 3 degraded."
        ), watch.line
        assert watch.warning.endswith("2 of 2 research reports this run were degraded (call raised)")

        # a full window, the oldest falling off
        Path(path).write_text(json.dumps({"window": [1] + [0] * 11}), encoding="utf-8")
        watch = research_watch.update([good], 0, path)
        assert (watch.bad, watch.window) == (0, 12), watch
        assert watch.line.endswith(" Last 12 reports: 0 degraded."), watch.line

        # corrupt files start empty
        for corrupt in ("not json", "[1, 0]", '"x"', "{}", '{"window": "0"}', '{"window": [0, 2]}',
                        '{"window": [0, true]}', '{"window": [0, 0.5]}', "\u0000"):  # fmt: skip
            Path(path).write_text(corrupt, encoding="utf-8")
            assert research_watch.update([bad], 0, path).window == 1, corrupt
        Path(path).write_bytes(b"\xff\xfe\x00")
        assert research_watch.update([bad], 0, path).window == 1

        # five of twelve stay quiet; the sixth trips
        for count, tripped in ((5, False), (6, True), (12, True)):
            calls = [bad] * count + [good] * (12 - count)
            watch = research_watch.update(calls, 0, os.path.join(folder, f"{count}.json"))
            assert (watch.bad, watch.window, watch.tripped) == (count, 12, tripped), watch
        Path(path).write_text(json.dumps({"window": [0] + [1] * 5 + [0] * 6}), encoding="utf-8")
        assert research_watch.update([], 1, path).tripped

        # a file that can't be written leaves no marker
        blocked = os.path.join(folder, "folder.json")
        os.mkdir(blocked)
        assert research_watch.update([good], 0, blocked).window == 1
        assert not os.path.exists(blocked + ".updated")

    # without a file (the Test Bot and the Cup) the window is this run alone
    assert research_watch.update([good, good], 0).window == 2

    # through main: a cut-short reply is a call and "returned nothing", and counts once
    health = main.ResearchHealth()
    name = "anthropic/claude-sonnet-5-5 + web search and fetch"
    health.record(name, "returned nothing")
    health.record(name, "APIConnectionError")
    health.record("asknews/latest", "KeyError")  # another source's failure isn't the tools'
    health.record("openrouter/anthropic/claude-sonnet-5.5:online", "TimeoutError")
    opened: list[tuple] = []
    real_alert = alerts.research_degraded
    alerts.research_degraded = lambda *args: opened.append(args) or True  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as folder, env(RESEARCH_WATCH_FILE=os.path.join(folder, "w.json")):
            notes: list[str] = []
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                main.watch_research([research_call(stop="pause_turn")], health, notes)
            saved = Path(os.environ["RESEARCH_WATCH_FILE"])
            assert json.loads(saved.read_text(encoding="utf-8")) == {"window": [1, 1]}
            assert len(notes) == 1 and notes[0].endswith("; 1 raised. Last 2 reports (window not full): 2 degraded."), notes
            assert "::warning title=Research degraded::2 of 2 research reports" in printed.getvalue()
            assert opened == []

            # tripped: one alert with this run's reasons, and the run page's line
            saved.write_text('{"window": [1, 1, 1, 1, 1]}', encoding="utf-8")
            with contextlib.redirect_stdout(printed):
                main.watch_research([bad, bad], main.ResearchHealth(), [])
            assert opened == [(7, 7, "searches failed")], opened

            # tripped by earlier runs alone, this run clean
            opened.clear()
            main.watch_research([good], main.ResearchHealth(), [])
            assert opened == [(7, 8, "none of them this run")], opened

            # an idle run: no line, no file touched
            opened.clear()
            notes = []
            main.watch_research([], main.ResearchHealth(), notes)
            assert notes == [] and opened == []

        # a failure here can't touch the run
        real_update = research_watch.update
        research_watch.update = lambda *args: 1 / 0  # type: ignore[assignment]
        try:
            main.watch_research([good], main.ResearchHealth(), [])
        finally:
            research_watch.update = real_update  # type: ignore[assignment]
    finally:
        alerts.research_degraded = real_alert  # type: ignore[assignment]
    print("research watch: ok")


def check_research_survives_one_failure() -> None:
    # Source labels such as "asknews/latest" are read from the environment as research
    # runs, so the whole check runs inside env(): an exported ASKNEWS_STRATEGY can't leak in
    with env(ANTHROPIC_API_KEY="x", PERPLEXITY_API_KEY="x", ASKNEWS_CLIENT_ID="x", ASKNEWS_SECRET="y"):
        _research_survives_one_failure()
    print("research: ok")


def _research_survives_one_failure() -> None:
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

    # every source failed and one because Rob's key hit its spend limit: that error comes
    # through (counted as stopped by the limit), not "no research" (a red failure)
    async def limited(source, prompt, question):
        if source == main.ASKNEWS_RESEARCHER:
            raise RuntimeError("asknews down")
        raise RuntimeError(SPEND_LIMIT_ERROR)

    bot._run_one_research_source = limited  # type: ignore[method-assign]
    try:
        asyncio.run(bot.run_research(binary_question()))
    except main.ResearchUnavailable as e:
        raise AssertionError(f"a spend limit must not read as missing research: {e}")
    except RuntimeError as e:
        assert main.spend_limit_reached(e), e
    else:
        raise AssertionError("research with every source failed must raise")


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
    # the source the criteria name, and forecasts elsewhere; the outcome by question type
    for expected in ("Give its latest value or status", "Polymarket, Kalshi, Manifold", "whether the question would resolve Yes or No"):  # fmt: skip
        assert expected in research_prompt, (expected, research_prompt)
    mc_prompt = bot._research_prompt(timed_question("mc"))
    assert "which option the question would resolve to" in mc_prompt, mc_prompt
    assert "The options are: ['a', 'b']" in mc_prompt and "Yes or No" not in mc_prompt, mc_prompt
    assert "what the question would resolve to" in bot._research_prompt(timed_question("numeric"))
    assert not research_prompt.startswith(" "), research_prompt  # indents still cleaned

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
    # yes/no: restate the criteria, trajectory, soft clock, price thresholds, markets, and a
    # check both ways in place of the one-sided "over-predict Yes" step
    for expected in (
        "(a) The question: in one sentence, what exactly must happen",
        "1 - (1 - p)^(days left / 365)",
        "1 - e^(-r x days left / 365)",
        "needs no rescaling",
        "a process already under way reaching its scheduled end",
        "An announced target date shows a plan exists, not that it will be met",
        "A search that finds nothing is not evidence",
        "(f) Markets and prices: a recent forecast on this question or a close match",
        "If only daily closes count it is less: about the same with 1 trading day left",
        "the standard deviation of its daily % changes",
        "A thin or stale market is weak evidence",
        "Would a Yes surprise you at this probability? Would a No?",
        "(h) Reconcile these into one probability",
    ):
        assert expected in prompts["binary"], (expected, prompts["binary"])
    assert "over-predict" not in prompts["binary"], prompts["binary"]
    assert "if the favourite fails, the most likely way it happens" in prompts["mc"], prompts["mc"]
    assert "a thin or stale market is weak evidence" in prompts["mc"], prompts["mc"]
    # numeric and date: width by forecastability, not "set wide intervals"; traded prices
    # keep a random-walk floor
    for kind in ("numeric", "date"):
        assert main.WIDTH_RULE in prompts[kind], (kind, prompts[kind])
        assert "humble" not in prompts[kind] and "wide 90/10" not in prompts[kind], prompts[kind]
    assert "1.3 x that daily volatility x the square root of the days it trades before the deadline" in prompts["numeric"]
    assert "go wider than this" in prompts["numeric"] and main.DAILY_VOLATILITY in prompts["numeric"]
    assert "daily volatility" not in prompts["date"], prompts["date"]
    # research: open the named source, a searched base rate, a series' move, each venue
    for expected in (
        "open it first if you can",
        "as a count that says what was counted",
        "the standard deviation of its daily % changes",
        'For each of Polymarket, Kalshi, Manifold and Metaculus',
        'or write "not found" (searched) or "not searched"',
    ):
        assert expected in research_prompt, (expected, research_prompt)
    # numeric questions get past values of the quantity, not an event count
    numeric_research = bot._research_prompt(timed_question("numeric"))
    assert "past values of the same quantity" in numeric_research and "Search once for the base rate" not in numeric_research
    print("prompts: ok")


def check_research_policy() -> None:
    """Every source failing: a question a later run can reach waits for it; one closing
    sooner is forecast without research, and the forecaster is told so."""
    with env(OPENROUTER_API_KEY="x", ASKNEWS_API_KEY="y"):  # see check_research_survives_one_failure
        _research_policy()
        _research_policy_with_several_reports()
    print("research policy: ok")


def _research_policy() -> None:
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
        "openrouter/anthropic/claude-sonnet-5.5:online"
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


def _research_policy_with_several_reports() -> None:
    """RESEARCH_REPORTS=2 near close: the no-research fallback is decided per question.
    A report whose sources all fail is dropped while another report has (or may still
    find) research; only the last report, when none found any, is forecast without it."""

    def bot_whose_reports(*outcomes: bool):
        bot = main.RcForecastBot(
            publish_reports_to_metaculus=False,
            research_reports_per_question=2,
            predictions_per_research_report=1,
        )
        calls = iter(outcomes)

        async def research(source, prompt, question):
            ok = next(calls) if source == main.ASKNEWS_RESEARCHER else False
            if not ok:
                raise RuntimeError("down")
            return "found something"

        bot._run_one_research_source = research  # type: ignore[method-assign]
        return bot

    soon = timed_question(closes_in=timedelta(minutes=30))
    for outcomes in ((True, False), (False, True)):
        bot = bot_whose_reports(*outcomes)
        got = []
        for _ in outcomes:
            try:
                got.append(asyncio.run(bot.run_research(soon)))
            except main.ResearchUnavailable as e:
                assert "other reports go ahead" in str(e), e
                got.append(None)
        assert main.NO_RESEARCH_NOTE not in got, (outcomes, got)
        assert sum(g is not None for g in got) == 1, (outcomes, got)
        assert bot.research_health.without_research == [], outcomes
    # both reports fail: the first is dropped, the last is forecast without research
    bot = bot_whose_reports(False, False)
    try:
        asyncio.run(bot.run_research(soon))
    except main.ResearchUnavailable:
        pass
    else:
        raise AssertionError("a report must not go blind while another may find research")
    assert asyncio.run(bot.run_research(soon)) == main.NO_RESEARCH_NOTE
    assert bot.research_health.without_research == ["https://example.invalid/q/77"]

    # through forecasting-tools: only researched predictions are aggregated
    seen: list[str] = []

    async def record(question, research):
        seen.append(research)
        return main.ReasonedPrediction(prediction_value=0.4, reasoning="r")

    bot = bot_whose_reports(False, True)
    bot._make_prediction = record  # type: ignore[method-assign]
    bot.summarize_research = lambda question, research: asyncio.sleep(0, "summary")  # type: ignore[method-assign]
    bot._create_unified_explanation = lambda *a, **k: "# SUMMARY\n# RESEARCH\n# FORECASTS\n"  # type: ignore[method-assign]
    results = asyncio.run(bot.forecast_questions([soon], return_exceptions=True))
    assert not isinstance(results[0], BaseException), results
    assert seen == ["## Research from asknews/latest\nfound something"], seen


def check_run_report() -> None:
    health = main.ResearchHealth()
    for _ in range(3):
        health.record("asknews/latest")
    health.record("asknews/latest", "RateLimitError")
    health.record("asknews/latest", "RateLimitError")
    health.record("asknews/latest", "AuthenticationError")
    health.record("openrouter/anthropic/claude-sonnet-5:online")
    # listed once each, and only when the forecast then went through (q/10 failed)
    health.without_research += [
        "https://example.invalid/q/9",
        "https://example.invalid/q/9",
        "https://example.invalid/q/10",
    ]

    class Report:  # stands in for a ForecastReport, which carries its question
        def __init__(self, n):
            self.question = BinaryQuestion(question_text="Q?", page_url=f"https://example.invalid/q/{n}")

    lines = main.run_report(
        ["- FutureEval (`33121`): no open questions."],
        ["asknews/latest", "openrouter/anthropic/claude-sonnet-5:online"],
        health,
        [Report(9), research_deferral("https://example.invalid/q/8"), RuntimeError("boom")],
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


def check_health_warnings() -> None:
    """A research source that fails leaves the run green; it is annotated instead."""
    health = main.ResearchHealth()
    health.record("asknews/latest")
    health.record("anthropic/claude-sonnet-5-5 + web search and fetch", "BadRequestError")
    health.record("anthropic/claude-sonnet-5-5 + web search and fetch", "BadRequestError")
    health.record("anthropic/claude-sonnet-5-5 + web search and fetch")
    assert main.health_warnings(health) == [
        "::warning title=Research source failed::anthropic/claude-sonnet-5-5 + web search and fetch"
        " failed 2 of 3 searches (BadRequestError x2)"
    ], main.health_warnings(health)
    assert main.health_warnings(main.ResearchHealth()) == []
    print("health warnings: ok")


def check_comment_sections() -> None:
    """The posted comment keeps exactly its three top-level sections when a research source
    or the summary brings its own "# " headings (both happened on 2026-10-05)."""
    # Only what forecasting-tools reads as a heading changes: "#hashtag" and a bare "#" don't
    assert main.flatten_headings("# Key facts\ntext\n### Dates ##\n#hashtag\n# **Bold**\n#\n# \n") == (
        "**Key facts**\ntext\n**Dates**\n#hashtag\n**Bold**\n#\n\n"
    )
    assert main.flatten_headings("## Tips for C#") == "**Tips for C#**"
    with env(ANTHROPIC_API_KEY="x", PERPLEXITY_API_KEY="x"):
        _comment_sections()
    print("comment sections: ok")


def _comment_sections() -> None:
    from unittest import mock

    from forecasting_tools import BinaryReport, ForecastBot, ReasonedPrediction
    from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

    bot = main.RcForecastBot(publish_reports_to_metaculus=False)
    q = binary_question()

    async def source(source, prompt, question):
        # The shape of Sonnet's research on Q45912
        return "# Brent Crude Front-Month Futures — Briefing\n## Current price\n$71 (2026-10-04)"

    async def summary(self, question, research):
        return "# Research Summary\n\nBrent sits near $71.\n\n## Sources\n- example.invalid"

    bot._run_one_research_source = source  # type: ignore[method-assign]
    research = asyncio.run(bot.run_research(q))
    with mock.patch.object(ForecastBot, "summarize_research", summary):
        summarized = asyncio.run(bot.summarize_research(q, research))
    assert "**Research Summary**" in summarized, summarized
    reasoning = "**(a) Time left:** ten days.\n\nProbability: 55%"
    collection = ResearchWithPredictions(
        research_report=research,
        summary_report=summarized,
        errors=[],
        predictions=[ReasonedPrediction(prediction_value=0.55, reasoning=reasoning)] * 3,
    )
    explanation = bot._create_unified_explanation(q, [collection], 0.55, 0.4, 1.0)
    report = BinaryReport(question=q, prediction=0.55, explanation=explanation)
    titles = [s.text_of_section_and_subsections.split("\n")[0] for s in report.report_sections]
    assert titles == ["# SUMMARY", "# RESEARCH", "# FORECASTS"], titles
    assert "Brent sits near $71." in report.summary
    assert "**Brent Crude Front-Month Futures — Briefing**" in report.research
    assert "[Hashtag]" not in explanation  # forecasting-tools' fallback for mixed levels
    assert "Probability: 55%" in report.first_rationale


def check_asknews() -> None:
    """The strategy and article count come from ASKNEWS_STRATEGY and ASKNEWS_ARTICLES;
    latest news only, one call, by default (the free tier is 1,000 calls a month)."""
    import asknews_sdk

    calls: list[dict] = []
    sleeps: list[float] = []
    # what each search strategy finds: one article by default
    found: dict[str, list] = {}
    article = types.SimpleNamespace(
        eng_title="Ministry delays the vote", summary="The vote moves to November.",
        language="en", pub_date=datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc),
        source_id="Reuters", article_url="https://www.reuters.com/x",
    )  # fmt: skip

    class FakeResponse:
        def __init__(self, strategy):
            self.as_dicts = found.get(strategy, [article])

    class FakeNews:
        async def search_news(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse(kwargs["strategy"])

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
        assert text.startswith("Latest news") and "Ministry delays the vote" in text, text
        # nothing found is no research: empty, so it reads as "returned nothing"
        found["latest news"] = []
        assert search() == "", search()

        text = search(ASKNEWS_STRATEGY="archive", ASKNEWS_ARTICLES="12")
        assert [(c["strategy"], c["n_articles"]) for c in calls] == [("news knowledge", 12)], calls
        assert text.startswith("Earlier news from the AskNews archive"), text

        # both: two searches, spaced for the free tier's one call every 10 seconds
        text = search(ASKNEWS_STRATEGY=" Both ")
        assert [c["strategy"] for c in calls] == ["latest news", "news knowledge"], calls
        assert len(sleeps) == 1 and sleeps[0] >= 10, sleeps
        assert text.index("Latest news") < text.index("Earlier news"), text
        # one of the two searches empty: the other's articles still count
        assert "No articles were found" in text and "Ministry delays the vote" in text, text
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


def close(a, b) -> bool:
    return a is not None and abs(a - b) < 1e-9


def check_spend() -> None:
    """What every model call cost, by OpenRouter's own figures, tagged with its question
    and the part of the work it was for. litellm's price list stopped pricing OpenRouter's
    responses on 2026-10-09, and OpenRouter's billed `cost` read $0.00 on every logged call
    (bring-your-own-key: the provider's bill is a separate figure)."""
    figure = spend.openrouter_figure
    assert figure(0.12, None, False) == (0.12, "billed")
    # BYOK: the provider's bill plus OpenRouter's fee, which is $0 on these keys so far
    assert figure(0.0, 0.40, True) == (0.40, "byok")
    got = figure(0.02, 0.40, True)
    assert close(got[0], 0.42) and got[1] == "byok", got
    assert figure(0.0, 0.30, None) == (0.30, "upstream")  # no BYOK flag, but a provider bill
    # a call that reads $0, or a BYOK call without the provider's figure, has no figure:
    # no model call on these keys is free, so $0 would be the old undercount again
    for args in ((0.0, None, None), (None, None, None), (0.0, None, True), (0.0, 0.0, False)):
        assert figure(*args) is None, args

    call = spend.call_from_response(
        "openrouter/anthropic/claude-sonnet-5:online",
        FakeResponse(generation="gen-1", cost=0.0, upstream=0.4, byok=True, searches=5),
        "https://example.invalid/q/1",
        "research: x",
    )
    assert (call.cost, call.basis, call.searches, call.generation) == (0.4, "byok", 5, "gen-1"), call
    assert (call.prompt_tokens, call.completion_tokens) == (1000, 200), call
    # OpenRouter gave no figure: kept for a look-up by its generation id
    call = spend.call_from_response("m", FakeResponse(generation="gen-2", cost=0.0))
    assert (call.cost, call.basis, call.generation) == (None, None, "gen-2"), call
    # not an OpenRouter response (Anthropic's API on Rob's key): the tokens at litellm's
    # prices (Sonnet 5: $2/$10 per M) plus $0.01 a web search, which litellm 1.80.10
    # leaves out of its own figure; litellm's figure only for a model it can't price
    other = FakeResponse(generation="msg_01", searches=4)
    other._hidden_params = {"response_cost": 0.03}
    call = spend.call_from_response("anthropic/claude-sonnet-5", other)
    assert call.basis == "litellm" and call.generation is None, call
    assert math.isclose(call.cost, 1000 * 2e-6 + 200 * 1e-5 + 4 * 0.01), call
    call = spend.call_from_response("anthropic/no-such-model", other)
    assert (call.cost, call.basis) == (0.03, "litellm"), call
    assert spend.label("https://www.metaculus.com/questions/45958/") == "45958"
    assert spend.label(None) == "outside any question"
    # list prices are OpenRouter's, by its ids, for Anthropic's own API names too
    oid = spend.openrouter_id
    assert oid("openrouter/anthropic/claude-sonnet-5:online") == "anthropic/claude-sonnet-5"
    assert oid("openrouter/anthropic/claude-opus-5.5") == "anthropic/claude-opus-5.5"
    assert oid("anthropic/claude-opus-5-5") == "anthropic/claude-opus-5.5"
    assert oid("anthropic/claude-haiku-4-5") == "anthropic/claude-haiku-4.5"
    assert oid("claude-haiku-4-5-20251001") == "anthropic/claude-haiku-4.5"
    assert oid("anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"

    _spend_look_up()
    _spend_metering()
    with env(OPENROUTER_API_KEY="x"):  # one search-backed source; no AskNews keys, no network
        _spend_through_the_bot()
    print("spend: ok")


def _spend_look_up() -> None:
    """A call without a figure is looked up from OpenRouter's record of the generation,
    which can take a few seconds to appear; the key never reaches the log."""
    secret = "sk-or-v1-FAKE-never-logged"
    answers = iter([404, 200])
    seen: list[tuple] = []
    record = {"data": {"total_cost": 0.0, "upstream_inference_cost": 0.31, "is_byok": True}}
    prices = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "pricing": {"prompt": "0.000002", "completion": "0.00001", "web_search": "0.01"},
            },
            {
                "id": "anthropic/claude-opus-5.5",
                "pricing": {"prompt": "0.000004", "completion": "0.00002", "web_search": "0.01"},
            },
        ]
    }

    class Answer:
        def __init__(self, status, body):
            self.status_code, self.body = status, body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append((url, params, headers))
        if url == spend.MODELS_URL:
            return Answer(200, prices)
        return Answer(next(answers), record)

    def failing_get(url, params=None, headers=None, timeout=None):
        raise RuntimeError(f"connection refused while sending {headers}")

    ledger = spend.Ledger()
    ledger.calls.append(spend.call_from_response("m", FakeResponse(generation="gen-late", cost=0.0)))
    sleeps: list[float] = []
    real = spend.requests.get
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    spend.logger.addHandler(handler)
    try:
        spend.requests.get = fake_get  # type: ignore[assignment]
        ledger.look_up(secret, sleep=sleeps.append)
        assert seen == [(spend.GENERATION_URL, {"id": "gen-late"}, {"Authorization": f"Bearer {secret}"})] * 2
        assert sleeps == [5.0], sleeps  # one wait, before the second try
        assert (ledger.calls[0].cost, ledger.calls[0].basis) == (0.31, "lookup"), ledger.calls
        # unreadable, the record and the price list both: the call stays without a figure
        online = "openrouter/anthropic/claude-sonnet-5:online"
        ledger.calls.append(
            spend.call_from_response(online, FakeResponse(generation="gen-x", cost=0.0, searches=2))
        )
        spend.requests.get = failing_get  # type: ignore[assignment]
        ledger.look_up(secret, tries=2, sleep=sleeps.append)
        assert [c.cost for c in ledger.calls] == [0.31, None], ledger.calls
        # out of time for records: no more look-ups, and the list price stands in (1,000
        # prompt and 200 completion tokens, two web searches)
        seen.clear()
        spend.requests.get = fake_get  # type: ignore[assignment]
        ticks = iter([0.0, 500.0, 500.0, 500.0])
        ledger.look_up(secret, sleep=sleeps.append, clock=lambda: next(ticks, 500.0))
        assert seen == [(spend.MODELS_URL, None, None)], seen
        assert close(ledger.calls[1].cost, 0.002 + 0.002 + 0.02), ledger.calls
        assert ledger.calls[1].basis == "list price"
        # not OpenRouter's, and no price anywhere: counted as without a figure, not as $0
        ledger.calls.append(spend.call_from_response("m", FakeResponse(generation="msg_1", cost=0.0)))
        ledger.look_up(secret, sleep=sleeps.append)
        assert ledger.calls[2].cost is None, ledger.calls
        # Anthropic's own API with no OpenRouter key (1,000 prompt and 200 completion
        # tokens, one search): litellm's token prices plus the search fee, nothing fetched
        own = spend.Ledger()
        own_response = FakeResponse(generation="msg_2", searches=1)
        own.calls.append(spend.call_from_response("anthropic/claude-opus-5-5", own_response))
        seen.clear()
        own.look_up("", sleep=sleeps.append)
        assert seen == [], seen
        assert close(own.calls[0].cost, 0.004 + 0.004 + 0.01), own.calls
        assert own.calls[0].basis == "litellm"
        # ... and when litellm can't price the model: no generation look-up, and the list
        # price under OpenRouter's id, the same total
        import litellm

        real_cost_per_token = litellm.cost_per_token
        litellm.cost_per_token = lambda **kwargs: (_ for _ in ()).throw(ValueError("unpriced"))
        try:
            own = spend.Ledger()
            own.calls.append(spend.call_from_response("anthropic/claude-opus-5-5", own_response))
        finally:
            litellm.cost_per_token = real_cost_per_token
        seen.clear()
        own.look_up("", sleep=sleeps.append)
        assert seen == [(spend.MODELS_URL, None, None)], seen
        assert close(own.calls[0].cost, 0.004 + 0.004 + 0.01), own.calls
        assert own.calls[0].basis == "list price"
    finally:
        spend.requests.get = real  # type: ignore[assignment]
        spend.logger.removeHandler(handler)
    assert records and all(secret not in r.getMessage() for r in records)
    text = "\n".join(spend.breakdown("FutureEval", ledger))
    assert "| outside any question | | $0.334+? | $0.334+? |" in text, text
    assert (
        "- 3 model calls, 2 web searches. Cost figures: 1 looked up from OpenRouter"
        " afterwards, 1 estimated at OpenRouter's list prices; 1 without any figure"
        " (not in the totals)." in text
    ), text


def _spend_metering() -> None:
    """The wrapper records each response and each call that raised, only inside a pass,
    never changes what the call returns, and never breaks a call by failing to measure."""

    async def fine(**kwargs):
        return FakeResponse(cost=0.01)

    async def times_out(**kwargs):
        raise TimeoutError("slow")

    class Broken:
        @property
        def usage(self):
            raise RuntimeError("odd response")

    async def odd(**kwargs):
        return Broken()

    metered = spend.metered(fine)
    assert spend.metered(metered) is metered  # wrapping twice is wrapping once
    response = asyncio.run(metered(model="m", messages=[]))  # outside a pass: not recorded
    assert response.usage.cost == 0.01
    with spend.collecting() as ledger:
        with spend.tagged(question="https://example.invalid/q/3", part="forecast 2"):
            asyncio.run(metered(model="m", messages=[]))
            try:
                asyncio.run(spend.metered(times_out)(model="m", messages=[]))
            except TimeoutError:
                pass
            else:
                raise AssertionError("the call's own error must reach the caller")
            assert isinstance(asyncio.run(spend.metered(odd)(model="m")), Broken)
        asyncio.run(metered(model="m", messages=[]))  # untagged: "other"
    assert [(c.question, c.part, c.cost) for c in ledger.calls] == [
        ("https://example.invalid/q/3", "forecast 2", 0.01),
        (None, "other", 0.01),
    ], ledger.calls
    assert ledger.raised == {"TimeoutError": 1}, ledger.raised
    assert "Calls that raised: TimeoutError ×1." in "\n".join(spend.breakdown("X", ledger))
    # a response with no choices (these fakes, and any odd one) records with no tool figures
    assert spend.tool_use(FakeResponse()) == {} and spend.tool_use(Broken()) == {}
    assert (ledger.calls[0].search_tries, ledger.calls[0].stop, ledger.calls[0].answered) == (0, "", True)
    assert ledger.calls[0].tally is None

    async def slow(**kwargs):
        await asyncio.sleep(0.05)
        return FakeResponse(cost=0.01)

    with spend.collecting() as ledger:
        asyncio.run(spend.metered(slow)(model="m", messages=[]))
    assert ledger.calls[0].seconds >= 0.04, ledger.calls
    # main meters forecasting-tools' one gateway to litellm when it is imported
    from forecasting_tools.ai_models import general_llm

    assert getattr(general_llm.acompletion, "_spend_metered", False)


def _spend_through_the_bot() -> None:
    """One opus-3 question through forecasting-tools' real flow, with a fake litellm:
    each call lands in the right column, and the numbers add up."""
    import litellm
    from litellm.types.utils import ServerToolUse
    from forecasting_tools.ai_models import general_llm

    def reply(text, **usage):
        return litellm.ModelResponse(
            id="gen-fake",
            usage=litellm.Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200, **usage),
            choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop", "index": 0}],
        )

    models: dict[str, set] = {}

    async def fake_acompletion(**kwargs):
        prompt = str(kwargs["messages"])
        if "convert text into structured data" in prompt:
            kind, response = "parsing", reply('{"prediction_in_decimal": 0.4}', cost=0.002)
        elif "summarize the following research" in prompt:
            kind, response = "summary", reply("The research finds nothing yet.", cost=0.004)
        elif "assistant to a superforecaster" in prompt:
            kind, response = "research", reply(
                "Nothing has happened yet (2026-10-10).",
                cost=0.0,
                is_byok=True,
                cost_details={"upstream_inference_cost": 0.40},
                server_tool_use=ServerToolUse(web_search_requests=5),
            )
        elif "superforecaster with an excellent calibration record" in prompt:
            kind, response = "forecast", reply(
                "The base rate is low.\n\nProbability: 40%",
                cost=0.0,
                is_byok=True,
                cost_details={"upstream_inference_cost": 0.05},
            )
        else:
            raise AssertionError(f"unexpected prompt: {prompt[:300]}")
        models.setdefault(kind, set()).add(kwargs["model"])
        return response

    q = timed_question()
    bot = main.make_bot("openrouter", budget.TIERS["opus-3"], publish=False)
    real = general_llm.acompletion
    general_llm.acompletion = spend.metered(fake_acompletion)
    try:
        with spend.collecting() as ledger:
            results = asyncio.run(bot.forecast_questions([q], return_exceptions=True))
    finally:
        general_llm.acompletion = real
    assert not isinstance(results[0], BaseException), results
    assert results[0].prediction == 0.4, results[0].prediction
    # the calls the template and the bot make, each where it belongs (two parsing calls a
    # forecast: forecasting-tools checks the parse against a second sample)
    from collections import Counter

    assert Counter(c.part for c in ledger.calls) == {
        "research: openrouter/anthropic/claude-sonnet-5.5:online": 1,
        "summary": 1,
        "forecast 1": 1,
        "forecast 2": 1,
        "forecast 3": 1,
        "parsing": 6,
    }, [c.part for c in ledger.calls]
    assert {c.question for c in ledger.calls} == {"https://example.invalid/q/77"}
    assert models == {
        "research": {"openrouter/anthropic/claude-sonnet-5.5:online"},
        "forecast": {budget.OPUS},
        "parsing": {"openrouter/anthropic/claude-haiku-5.5"},
        "summary": {"openrouter/anthropic/claude-haiku-5.5"},
    }, models
    assert close(ledger.total(), 0.40 + 3 * 0.05 + 6 * 0.002 + 0.004), ledger.total()

    text = "\n".join(
        spend.breakdown("FutureEval", ledger, main.question_statuses([q], results, publish=False))
    )
    for expected in (
        "| Question | Status | Research: openrouter/anthropic/claude-sonnet-5.5:online"
        " | Forecast 1 | Forecast 2 | Forecast 3 | Parsing and summary | Total |",
        "| [77](https://example.invalid/q/77) | dry run | $0.400 | $0.050 | $0.050 | $0.050"
        " | $0.016 | $0.566 |",
        "| Share | | 71% | 9% | 9% | 9% | 3% | 100% |",
        "- 11 model calls, 5 web searches. Cost figures: 7 billed by OpenRouter,"
        " 4 provider's bill plus OpenRouter's fee (BYOK).",
    ):
        assert expected in text, (expected, text)


if __name__ == "__main__":
    check_routing()
    check_plan()
    check_key_status()
    check_run_loop()
    check_own_key()
    check_alerts()
    check_watchdog()
    check_model_watch()
    check_model_tags()
    check_extra_forecasters()
    check_scorecard()
    check_aggregation()
    check_guards()
    check_research_survives_one_failure()
    check_prompts()
    check_research_policy()
    check_run_report()
    check_health_warnings()
    check_comment_sections()
    check_asknews()
    check_research_tools()
    check_tool_rules()
    check_research_watch()
    with fixed_anthropic_prices():
        check_spend()
        check_own_key_research()
        check_tool_census()
    print("all checks passed")
