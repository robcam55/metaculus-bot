import argparse
import asyncio
import json
import logging
import math
import os
import re
import statistics
import sys
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import dotenv

# Pacing for Metaculus's incremental OpenRouter credits.
import budget

# What each run spends, call by call, by question and part of the work.
import spend

# GitHub issues that tell Rob when the credits run low or his own key stops.
import alerts

# Numeric and multiple-choice answers that would otherwise sink a whole question.
import forecast_guards

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    ForecastReport,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)
from forecasting_tools.ai_models import general_llm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
import litellm

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Every model call, forecasting-tools' own included, goes through general_llm's acompletion:
# metering it records what each call cost (see spend.py). It changes no request.
spend.install(general_llm)
# 0.2.92's CDF standardisation can loop forever on a CDF that isn't increasing, freezing
# the whole run; this bounds it as forecasting-tools 0.3.5 does (see forecast_guards.py)
forecast_guards.install()

# AskNews's free tournament tier allows 1,000 calls a month. The template's
# "news-summaries" makes 6 per question (latest news = 1, archive = 5); a season plus
# MiniBench is roughly 280 questions a month, so by default only the latest-news call is
# used. ASKNEWS_STRATEGY (latest, archive or both) and ASKNEWS_ARTICLES change that without
# a code change once AskNews confirms the quota and how it counts archive searches.
ASKNEWS_RESEARCHER = "asknews"
ASKNEWS_STRATEGIES = {
    "latest": ("latest news",),
    "archive": ("news knowledge",),
    "both": ("latest news", "news knowledge"),
}
ASKNEWS_DEFAULT_ARTICLES = 8

# Research settings that mean "no research" on purpose, as opposed to research that failed
NO_RESEARCH_SETTINGS = ("None", "no_research")

# A question whose research fails waits for the next run instead of being forecast blind:
# a missed question adds nothing to the score, while a blind forecast on a news-driven
# question can subtract. Runs come every 20 minutes and one can wait behind a busy run (25
# questions at about 35 seconds each), so a question closing sooner than this gets no
# further try and is forecast without research instead, with the forecaster told so.
RESEARCH_RETRY_MARGIN = timedelta(minutes=45)
NO_RESEARCH_NOTE = (
    "No research is available: every research source failed, and the question closes"
    " before the bot can try again. Forecast from your own knowledge and base rates, say"
    " what you could not check, and keep your uncertainty wide."
)

# How wide a numeric or date range should be. The template said "set wide 90/10 confidence
# intervals". The bot's evidence that its ranges ran too wide is thin: its 9 resolved
# forecaster ranges (3 questions, one model, before the mixture) all held the outcome
# inside the 10th to 90th percentile, which a calibrated forecaster would manage about
# half the time on so few. nostreambot's 43-question check found its own non-financial
# ranges 22% too wide. So this asks for width by forecastability, with no push either way,
# and keeps room for revised reports. The scorecard's 10th-90th coverage row is the test:
# under about 75% inside over 20 or more new-prompt questions means put a widening line
# back. Traded prices get a random-walk floor in the numeric prompt's step (g).
WIDTH_RULE = (
    "Set the width of your range by how forecastable the quantity is: narrow when"
    " published data or a short horizon nearly fixes it, wide when it depends on events"
    " that have not happened yet. For counts and statistics still being reported, leave"
    " room for revisions, backlogs and late reports. Do not widen or narrow out of habit."
)
# How the price rules measure a series' daily moves. The 1.3 in the numeric floor is the
# 10th/90th percentile z (1.28) for a standard deviation; a median or average absolute move
# runs 20-50% smaller for fat-tailed prices, which would make the floor that much too narrow
DAILY_VOLATILITY = (
    "daily volatility (the standard deviation of its daily % changes: annualized"
    " volatility / 16, or / 19 for crypto, which trades every day; an average absolute"
    " daily move x 1.25)"
)

# forecasting-tools 0.2.x (the template's pin) still points at the Summer 2026 season;
# its 0.3.1 release carries these Fall 2026 ids. AIB_TOURNAMENT_ID and METACULUS_CUP_ID
# (numeric id or slug) override them for later seasons.
FALL_2026_FUTUREEVAL_ID = 33121  # https://www.metaculus.com/tournament/fall-futureeval-2026/
FALL_2026_METACULUS_CUP_ID = 33108  # https://www.metaculus.com/tournament/metaculus-cup-fall-2026/

# Research runs one question at a time (a minute or so each), so a burst such as a new
# MiniBench round would outlast the workflow's 60-minute timeout. Runs come every 20
# minutes and pick up the rest.
MAX_QUESTIONS_PER_RUN = 25

# litellm caps an Anthropic reply at 4,096 tokens unless told otherwise, and Opus 5.5,
# Sonnet 5.5 and Haiku 5.5 think on every request, inside that cap. OpenRouter sets its own.
ANTHROPIC_MAX_TOKENS = 16000
# The forecasters on Rob's key think at "high" effort (FORECASTER_EFFORT), so they get
# more room: a reply that spends the whole cap thinking comes back empty and is retried
ANTHROPIC_FORECAST_MAX_TOKENS = 32000
# Opus 5.5 thinks at "medium" when no effort is sent, a level below Opus 5's default. In
# Metaculus's Spring 2026 analysis, high-effort versions beat their standard-effort twins
# in 8 of 8 model pairs, though most of those pairs compared no thinking with thinking.
# The forecasts are about $0.05 of a question's $0.60. litellm 1.80.10 passes Anthropic's
# output_config through (and accepts only low, medium and high); its reasoning_effort would
# send a thinking budget, which Opus 5.5 rejects.
FORECASTER_EFFORTS = ("low", "medium", "high")
DEFAULT_FORECASTER_EFFORT = "high"
# The models on each route, from budget.py: the newest Sonnet and Haiku (2026-10), with
# Opus 5.5 from the pacing tiers. Haiku 5.5 parses at a tenth of Haiku 4.5's price, at low
# effort: thinking counts against its reply cap, and copying numbers out of a forecast
# needs little of it. It rejects a temperature other than the default, which Haiku 4.5
# (PARSER_MODEL can still pick it) ran at 0.
SONNET_MODEL = {"openrouter": budget.SONNET, "anthropic": budget.ANTHROPIC_IDS[budget.SONNET]}
HAIKU_MODEL = {"openrouter": budget.HAIKU, "anthropic": budget.ANTHROPIC_IDS[budget.HAIKU]}
PARSER_EFFORT = "low"
ANTHROPIC_PARSER_MAX_TOKENS = 8000
# Models that take no effort setting and accept a temperature: Haiku 4.5 and older
_OLDER_HAIKU = re.compile(r"haiku-[34]")
# Anthropic's own web search for the search-backed model on Rob's key, up to four searches
# a question at $10 per 1,000. It is passed as a tool, not litellm's web_search_options,
# which litellm 1.80.10 always sends as the older web_search_20250305: this version
# filters results before they enter the context (dynamic filtering, no extra fee), so
# fewer of them are billed as input.
ANTHROPIC_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 4}
# And Anthropic's web fetch, so research can open the page the resolution criteria name
# and read its current state: the Spring 2026 winner read its resolution links, and
# several losses here turned on a source's state (a grid connection, a tide gauge). No
# fee; a fetched page is billed as input, capped here at 8,000 tokens, twice a question.
# Four searches and two fetches keep the server's tool loop well under its 10 rounds.
# litellm 1.80.10 sends the beta header web-fetch-2025-09-10 with it, in place of any
# other. RESEARCH_WEB_FETCH=false turns it off without a code change.
ANTHROPIC_FETCH_TOOL = {
    "type": "web_fetch_20260209",
    "name": "web_fetch",
    "max_uses": 2,
    "max_content_tokens": 8000,
    "citations": {"enabled": True},
}
# Stop reasons (as litellm reports them) for a research reply that ended before its
# answer: a server-side search loop paused, the reply ran out of tokens, or a safety
# classifier stopped it
CUT_SHORT = ("pause_turn", "length", "refusal")

# What Anthropic says when a key can't spend: a workspace or organization spend limit
# (400 invalid_request_error, or 429 enforced_spend_limit_reached), prepaid credit gone,
# or a 402 billing_error. The text survives litellm's wrapping on any version.
SPEND_LIMIT_MARKERS = (
    "api usage limits",
    "enforced_spend_limit_reached",
    "credit balance is too low",
    "billing_error",
)
# Anthropic's error types for a key it turns away (401 and 403). litellm 1.80.10, which
# the runs install, maps a 403 to APIConnectionError and has no top-level
# PermissionDeniedError, so the status code and these words decide, not litellm's classes.
KEY_REJECTED_MARKERS = ("authentication_error", "permission_error")


def spend_limit_reached(error: BaseException) -> bool:
    """Whether an error, or any error grouped inside it, is Anthropic refusing to spend."""
    if any(marker in str(error).lower() for marker in SPEND_LIMIT_MARKERS):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(spend_limit_reached(e) for e in error.exceptions)
    return False


def key_rejected(error: BaseException) -> bool:
    """Whether Anthropic turned the key itself away: a 401 or 403."""
    if getattr(error, "status_code", None) in (401, 403):
        return True
    return any(marker in str(error).lower() for marker in KEY_REJECTED_MARKERS)


def own_key_check() -> str | None:
    """One tiny request on Rob's own key (a few millionths of a dollar) before a run spends
    AskNews calls and research on questions the key can't pay for. None when the key works
    or the failure looks passing (an overload, a timeout); otherwise "limit" (a spend limit
    or no credit) or "rejected: " and the error type (a bad or revoked key)."""
    try:
        litellm.completion(
            model=HAIKU_MODEL["anthropic"],
            messages=[{"role": "user", "content": "Reply with OK."}],
            # room for a little thinking, which Haiku 5.5 does by default
            max_tokens=64,
            output_config={"effort": "low"},
            timeout=60,
        )
    except Exception as e:  # never log the message: keep request details out of logs
        if spend_limit_reached(e):
            return "limit"
        if key_rejected(e):
            return f"rejected: {type(e).__name__}"
        logger.warning(f"Own-key check failed ({type(e).__name__}); forecasting anyway")
    return None


def forecaster_effort() -> str | None:
    """FORECASTER_EFFORT (a repository variable): the forecasters' effort on Anthropic's
    own API and the extra forecasters' (OpenRouter's reasoning effort), high by default;
    "default" sends none, leaving the model's own."""
    value = os.getenv("FORECASTER_EFFORT", "").strip().lower() or DEFAULT_FORECASTER_EFFORT
    if value == "default":
        return None
    if value not in FORECASTER_EFFORTS:
        raise SystemExit(
            f"FORECASTER_EFFORT must be one of {', '.join(FORECASTER_EFFORTS)} or default;"
            f" got {value!r}"
        )
    return value


def asknews_strategy() -> str:
    """ASKNEWS_STRATEGY (a repository variable): latest (the default), archive or both."""
    name = os.getenv("ASKNEWS_STRATEGY", "").strip().lower() or "latest"
    if name not in ASKNEWS_STRATEGIES:
        raise SystemExit(
            f"ASKNEWS_STRATEGY must be one of {', '.join(ASKNEWS_STRATEGIES)}; got {name!r}"
        )
    return name


def asknews_articles() -> int:
    """ASKNEWS_ARTICLES (a repository variable): articles per AskNews search, default 8."""
    value = os.getenv("ASKNEWS_ARTICLES", "").strip()
    if not value:
        return ASKNEWS_DEFAULT_ARTICLES
    if not value.isdigit() or int(value) < 1:
        raise SystemExit(f"ASKNEWS_ARTICLES must be a whole number above 0; got {value!r}")
    return int(value)


def asknews_label(strategy: str | None = None) -> str:
    """How research from AskNews is labelled in reports and on the run page."""
    strategy = strategy or asknews_strategy()
    return "asknews/latest+archive" if strategy == "both" else f"asknews/{strategy}"


class ConfiguredAskNewsSearcher(AskNewsSearcher):
    """AskNews news search as ASKNEWS_STRATEGY and ASKNEWS_ARTICLES set it: one latest-news
    search by default, instead of the template's latest-news plus archive searches."""

    HEADINGS = {
        "latest news": "Latest news (the past day or two)",
        "news knowledge": "Earlier news from the AskNews archive (the past two months or so)",
    }

    def __init__(
        self, strategy: str | None = None, n_articles: int | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.strategy = strategy or asknews_strategy()
        self.n_articles = n_articles or asknews_articles()

    async def get_formatted_news_async(self, query: str) -> str:
        cache_key = f"{self.strategy} {self.n_articles} {query}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        from asknews_sdk import AsyncAskNewsSDK

        sections: list[str] = []
        any_found = False
        async with AsyncAskNewsSDK(
            client_id=self.client_id,
            client_secret=self.client_secret,
            api_key=self.api_key,
            scopes=set(["news"]),
        ) as ask:
            for i, search in enumerate(ASKNEWS_STRATEGIES[self.strategy]):
                if i:
                    # The free tier allows one call every 10 seconds
                    await asyncio.sleep(self._default_rate_limit)
                response = await ask.news.search_news(
                    query=query,
                    n_articles=self.n_articles,
                    return_type="both",
                    strategy=search,
                    try_cache=self._default_try_cache,
                )
                articles = response.as_dicts
                found = (
                    self._format_articles(articles)
                    if articles
                    else "No articles were found.\n\n"
                )
                sections.append(f"{self.HEADINGS[search]}:\n\n{found}")
                any_found = any_found or bool(articles)
        # No articles from any search is no research: returned empty, it counts as such
        # (the run page's "returned nothing"), so when the other source fails too the
        # question waits for the next run instead of being forecast blind
        formatted = "".join(sections) if any_found else ""
        self.cache.set(cache_key, formatted)
        return formatted


class ResearchUnavailable(RuntimeError):
    """Every research source failed for a question that a later run can still forecast."""


def deferred_for_research(result: object) -> bool:
    """Whether a question's result is a failure caused only by missing research."""
    if isinstance(result, ResearchUnavailable):
        return True
    if isinstance(result, BaseExceptionGroup):
        matched, rest = result.split(ResearchUnavailable)
        return matched is not None and rest is None
    return False


@dataclass
class SourceTally:
    searches: int = 0
    worked: int = 0
    failures: Counter = field(default_factory=Counter)  # error type -> count


@dataclass
class ResearchHealth:
    """What each research source returned during a run, for the Actions run page. A failed
    source leaves the run green; it shows in this table and as a warning annotation."""

    sources: dict[str, SourceTally] = field(default_factory=dict)
    # Questions (by question_key) whose forecast went ahead with no research
    without_research: list[str] = field(default_factory=list)
    # The extra forecasters' forecasts, by model: a failed one only thins the question's
    # members, so it would otherwise show nowhere but the log
    forecasters: dict[str, SourceTally] = field(default_factory=dict)

    def record(self, source: str, error: str | None = None) -> None:
        self._count(self.sources, source, error)

    def record_forecast(self, model: str, error: str | None = None) -> None:
        self._count(self.forecasters, model, error)

    @staticmethod
    def _count(tallies: dict[str, SourceTally], name: str, error: str | None) -> None:
        tally = tallies.setdefault(name, SourceTally())
        tally.searches += 1
        if error is None:
            tally.worked += 1
        else:
            tally.failures[error] += 1


# forecasting-tools posts each forecast as one comment with three top-level sections,
# "# SUMMARY", "# RESEARCH" and "# FORECASTS", and finds them again by position. A heading
# inside text the bot embeds there (a research source's own "# Key facts", a summary that
# opens with "# Research Summary") adds sections and shifts the rest: the log then shows
# "Failed to get first rationale", a research section can have every "#" turned into
# "[Hashtag]", and bot-review's --section reads the wrong part. Embedded text keeps its
# headings as bold lines instead. The pattern matches what forecasting-tools counts as a
# heading (MarkdownTree: one to eight "#" and a space at the start of a line).
_EMBEDDED_HEADING = re.compile(r"^#{1,8} +(.*?)(?: +#+)? *$", re.MULTILINE)


def flatten_headings(text: str) -> str:
    """Markdown headings in `text` as bold lines, so it can't add comment sections."""

    def bold(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        if not title or (title.startswith("**") and title.endswith("**")):
            return title
        return f"**{title}**"

    return _EMBEDDED_HEADING.sub(bold, text)


def question_key(question: MetaculusQuestion) -> str:
    return question.page_url or question.question_text


async def parse(*args, **kwargs):
    """forecasting-tools' structure_output, unchanged, with its model calls counted as
    parsing in the run page's spend breakdown."""
    with spend.tagged(part="parsing"):
        return await structure_output(*args, **kwargs)


def source_name(source: GeneralLlm | str) -> str:
    if isinstance(source, GeneralLlm):
        tools = source.litellm_kwargs.get("tools") or []
        if source.litellm_kwargs.get("web_search_options") or any(
            isinstance(t, dict) and t.get("name") == "web_search" for t in tools
        ):
            fetch = any(isinstance(t, dict) and t.get("name") == "web_fetch" for t in tools)
            return f"{source.model} + web search" + (" and fetch" if fetch else "")
        return source.model
    if source == ASKNEWS_RESEARCHER:
        return asknews_label()
    return str(source)


def research_sources(llms: dict[str, str | GeneralLlm | None]) -> list[GeneralLlm | str]:
    """The research sources in an llms dict. A "no research" setting counts as none."""
    return [
        source
        for source in (llms.get("researcher"), llms.get("researcher_2"))
        if source and not (isinstance(source, str) and source in NO_RESEARCH_SETTINGS)
    ]


def _utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def timeline(question: MetaculusQuestion, now: datetime | None = None) -> str:
    """Today's date and the question's close and scheduled resolution, for every prompt:
    without them a model can't tell what is still to come or how much time is left."""
    now = now or datetime.now(timezone.utc)
    parts = [f"Today is {now:%Y-%m-%d} ({now:%H:%M} UTC)."]
    if question.close_time:
        parts.append(f"Forecasting on this question closes {_utc(question.close_time)}.")
    if question.scheduled_resolution_time:
        parts.append(
            f"It is scheduled to resolve {_utc(question.scheduled_resolution_time)}."
        )
    return " ".join(parts)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _has_asknews() -> bool:
    return bool(
        (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
        or os.getenv("ASKNEWS_API_KEY")
    )


class ResearchCutShort(RuntimeError):
    """A research reply that stopped before its answer: what came back is narration."""


class AnthropicSearchLlm(GeneralLlm):
    """Sonnet 5.5 with Anthropic's web search, on Rob's own key. Three things GeneralLlm
    doesn't do on this route:
    - Keep the sources. litellm 1.80.10 keeps Anthropic's citations in the message's
      provider_specific_fields, which GeneralLlm never reads, so until 2026-10-28 the
      research reached the forecasters (and the posted comment) with every URL dropped.
    - Cache the search loop. One cache marker on the prompt is enough for Anthropic to
      cache each round of search results for the next search in the same request, read
      at a tenth of the input price, instead of billing every round again in full. The
      prompt itself is too short to cache.
    - Turn away a reply that stopped early (CUT_SHORT). GeneralLlm accepts any text, so
      a reply holding only "I'll search for this." counted as research. It comes back
      empty instead, which run_research counts as "returned nothing", and isn't retried:
      a fresh request reruns the whole search and usually stops the same way. A reply
      with no text at all (only thinking and tool blocks, as a cut-short loop can be)
      fails GeneralLlm's own check before the stop reason is read; it is treated the same.
    - List the pages it opened. litellm drops web fetch results, and a citation of a
      fetched page carries no URL, so the fetched URLs come from the fetch calls.
    The researcher is always invoked with a plain prompt, which model_input_to_message
    turns into one user message."""

    async def _mockable_direct_call_to_model(self, prompt):
        try:
            return await super()._mockable_direct_call_to_model(prompt)
        except (ResearchCutShort, AssertionError) as e:  # inside GeneralLlm's retries
            if isinstance(e, AssertionError) and not str(e).startswith(NO_TEXT_ASSERTION):
                raise
            logger.warning(f"{e}: no research from {self.model} this time")
            return TextTokenCostResponse(
                data="",
                prompt_tokens_used=0,
                completion_tokens_used=0,
                total_tokens_used=0,
                model=self.model,
                cost=0.0,
            )

    def model_input_to_message(self, user_input, system_prompt=None):
        messages = super().model_input_to_message(user_input, system_prompt)
        last = messages[-1]
        if last.get("role") == "user" and isinstance(last.get("content"), str):
            block = {"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}
            messages[-1] = {**last, "content": [block]}  # type: ignore[dict-item]
        return messages

    @staticmethod
    def _extract_citations(response, choices) -> list[str]:
        # The one hook GeneralLlm gives on the raw response, so the stop reason is
        # checked here too, before the text is taken as research
        reason = getattr(choices[0], "finish_reason", None)
        if reason in CUT_SHORT:
            raise ResearchCutShort(f"The research reply stopped early ({reason})")
        message = choices[0].message
        fields = getattr(message, "provider_specific_fields", None) or {}
        urls: list[str] = []
        cited_page = False  # a citation of a fetched page (it carries no URL)
        cited_any = False
        for group in fields.get("citations") or []:
            for citation in group or []:
                if not isinstance(citation, dict):
                    continue
                cited_any = True
                cited_page = cited_page or citation.get("type") in FETCH_CITATIONS
                url = citation.get("url")
                if url and url not in urls:
                    urls.append(url)
        if cited_page:
            # The pages it opened come first: the resolution source, when it found one.
            # Only when a fetched page is cited, since a failed fetch can't be told apart
            urls = [u for u in _fetched_urls(message) if u not in urls] + urls
        if not cited_any:
            # Nothing cited (with dynamic filtering the searches run inside code, and
            # the answer may cite none): the pages the searches returned, instead
            for result in fields.get("web_search_results") or []:
                found = result.get("content") if isinstance(result, dict) else None
                for item in found if isinstance(found, list) else []:
                    url = item.get("url") if isinstance(item, dict) else None
                    if url and url not in urls:
                        urls.append(url)
        return urls or GeneralLlm._extract_citations(response, choices)


# GeneralLlm's assertion for a reply with no text block
NO_TEXT_ASSERTION = "Answer is not a string"
# Citation types that point into a fetched page (a document) rather than a search result
FETCH_CITATIONS = ("char_location", "page_location", "content_block_location")


def _fetched_urls(message) -> list[str]:
    """The URLs the reply's web fetch calls opened, in order (litellm keeps the server's
    tool calls as tool_calls, with their input as JSON arguments)."""
    urls: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        if getattr(function, "name", None) != "web_fetch":
            continue
        try:
            url = json.loads(function.arguments or "{}").get("url")
        except (TypeError, ValueError, AttributeError):
            continue
        if url and url not in urls:
            urls.append(url)
    return urls


# Each forecast's reasoning starts with the model that wrote it, and each forecaster's line
# in the posted summary ends with it. bot-review keeps that line's text, so the weekly
# scorecard can compare models on the same questions (scorecard.py, members)
MODEL_LINE = "Model: {}"
_MODEL_PREFIX = re.compile(r"\AModel: (\S+)\n")


def with_model(model: str, reasoning: str) -> str:
    return f"{MODEL_LINE.format(model)}\n\n{reasoning}"


def model_of_reasoning(reasoning: str) -> str | None:
    found = _MODEL_PREFIX.match(reasoning)
    return found.group(1) if found else None


# The Test Bot's inputs for trying another model in one role (test_bot.yaml sets these
# from its own inputs; no other workflow does, and main() drops them outside the Test Bot):
# how a newly released model is run before a switch (model_watch.py)
TRIAL_MODELS = {
    "forecaster": "TRIAL_FORECASTER_MODEL",
    "researcher": "TRIAL_RESEARCHER_MODEL",
    "parser": "TRIAL_PARSER_MODEL",
    "extra": "TRIAL_EXTRA_FORECASTERS",
}


def trial_model(role: str) -> str:
    """The model the Test Bot is trying for `role`, or "" for the bot's own."""
    return os.getenv(TRIAL_MODELS[role], "").strip()


# A second model family beside the Opus forecasts, paid by Rob's own OpenRouter key. In
# Metaculus's same-setup house bots OpenAI's models led most seasons, while no Claude model
# has; a second family hedges that choice and makes the members' errors less alike
# (2026-10-30 research). EXTRA_FORECASTERS lists OpenRouter models, each with how many
# forecasts it adds: "openrouter/openai/gpt-6.1-sol:2". They read the same research and
# prompts, and the usual aggregation takes every member: with three Opus forecasts and two
# others, yes/no questions use the trimmed mean. The key goes in OPENROUTER_OWN_API_KEY,
# never OPENROUTER_API_KEY, which llm_route reads as Metaculus's credits. Off unless both
# are set; the Test Bot's "extra" input tries a list for one run.
OWN_OPENROUTER_KEY = "OPENROUTER_OWN_API_KEY"
MAX_EXTRA_FORECASTS = 5
# An OpenRouter id, with an optional variant (":online") and an optional count (":2")
_EXTRA_ITEM = re.compile(
    r"^(?P<model>openrouter/[\w.-]+/[\w.-]+(?::[a-z]+)?)(?::(?P<count>\d+))?$"
)
# The extra forecaster for the forecast being made, set per forecast in _make_prediction
_EXTRA_FORECASTER: ContextVar[GeneralLlm | None] = ContextVar("extra_forecaster", default=None)
# The question being forecast, for the success bar (_handle_errors_in__run_individual_question)
_QUESTION: ContextVar[str | None] = ContextVar("question", default=None)


def requested_extras() -> str:
    """The extra forecasters asked for: the Test Bot's input, else EXTRA_FORECASTERS."""
    return trial_model("extra") or os.getenv("EXTRA_FORECASTERS", "").strip()


def extra_forecaster_models() -> list[str]:
    """One model id per extra forecast, in order; [] when off or without the key. A
    mistyped list stops the run at startup. Items are separated by commas, semicolons,
    spaces or new lines."""
    spec = requested_extras()
    if not spec:
        return []
    models: list[str] = []
    for item in (part for part in re.split(r"[,;\s]+", spec) if part):
        found = _EXTRA_ITEM.match(item)
        count = int(found.group("count") or 1) if found else 0
        if not found or count < 1:
            raise SystemExit(
                "EXTRA_FORECASTERS lists OpenRouter models, comma-separated, each with an"
                f" optional count, like openrouter/openai/gpt-6.1-sol:2; got {item!r}"
            )
        models += [found.group("model")] * count
    if len(models) > MAX_EXTRA_FORECASTS:
        raise SystemExit(f"EXTRA_FORECASTERS adds {len(models)} forecasts; at most {MAX_EXTRA_FORECASTS}")
    return models if os.getenv(OWN_OPENROUTER_KEY) else []


def build_extra_forecasters() -> list[GeneralLlm]:
    """The extra forecasters, one per forecast, on Rob's own OpenRouter key. Effort goes as
    OpenRouter's own reasoning setting: litellm 1.80.10 refuses reasoning_effort for models
    it doesn't know to reason, Gemini 4 Argon among them."""
    models = extra_forecaster_models()
    effort = forecaster_effort()
    built: dict[str, GeneralLlm] = {}
    for model in dict.fromkeys(models):
        built[model] = GeneralLlm(
            model=model,
            api_key=os.getenv(OWN_OPENROUTER_KEY),  # never the credits' key from the environment
            temperature=None,
            timeout=_env_int("FORECASTER_TIMEOUT_S", 480),
            allowed_tries=2,
            # caps one call's cost, and what OpenRouter reserves against the key's balance
            max_tokens=ANTHROPIC_FORECAST_MAX_TOKENS,
            **({"reasoning": {"effort": effort}} if effort else {}),
        )
    return [built[model] for model in models]


def spend_on(calls: list, models: set[str]) -> float:
    """What the calls to these models cost."""
    return sum(c.cost or 0.0 for c in calls if c.model in models)


def llm_route() -> str | None:
    """Which key pays: Metaculus's donated OpenRouter credits before a personal
    Anthropic key. None leaves forecasting-tools' own defaults in charge."""
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def build_llms(
    route: str | None, tier: budget.Tier | None = None
) -> dict[str, str | GeneralLlm | None]:
    """The bot's models on one route. A pacing tier picks the forecaster on either route:
    on Metaculus's OpenRouter credits (Opus 5.5 without one), and on Rob's own Anthropic
    key once the credits run out. Without a tier (the Metaculus Cup) a personal key runs
    Sonnet 5.5, or FORECASTER_MODEL. Haiku 5.5 parses on both; PARSER_MODEL overrides it."""
    defaults = dict(ForecastBot._llm_config_defaults())
    if route is None:
        defaults["researcher_2"] = None
        return defaults
    # Anthropic's API through litellm needs a reply cap that leaves room for thinking
    room = {"max_tokens": ANTHROPIC_MAX_TOKENS} if route == "anthropic" else {}
    if trial_model("forecaster"):
        forecaster = trial_model("forecaster")
    elif route == "openrouter":
        forecaster = tier.forecaster if tier and tier.forecaster else budget.OPUS
    else:
        if tier and tier.forecaster:
            forecaster = budget.on_route(tier.forecaster, route)
        else:
            forecaster = os.getenv("FORECASTER_MODEL") or SONNET_MODEL[route]
    parser = trial_model("parser") or os.getenv("PARSER_MODEL") or HAIKU_MODEL[route]
    older_parser = bool(_OLDER_HAIKU.search(parser))
    thinking: dict = {}
    if route == "anthropic":
        thinking["max_tokens"] = ANTHROPIC_FORECAST_MAX_TOKENS
        effort = forecaster_effort()
        if effort and not _OLDER_HAIKU.search(forecaster):  # Haiku 4.5 takes no effort
            thinking["output_config"] = {"effort": effort}
    defaults["default"] = GeneralLlm(
        model=forecaster,
        # Opus 5.5, Sonnet 5.5 and Haiku 5.5 reject sampling parameters: send none
        temperature=None,
        # more thinking takes longer: up to 8 minutes on Anthropic's API, not streamed.
        # Kept short of 10 so a dropped connection and its one retry stay well inside the
        # job's 60 minutes
        timeout=_env_int("FORECASTER_TIMEOUT_S", 480 if route == "anthropic" else 300),
        allowed_tries=2,
        **thinking,
    )
    parser_extra: dict = {}
    if route == "anthropic" and not older_parser:
        parser_extra = {
            "max_tokens": ANTHROPIC_PARSER_MAX_TOKENS,
            "output_config": {"effort": PARSER_EFFORT},
        }
    parser_llm = GeneralLlm(
        model=parser,
        temperature=0 if older_parser else None,
        timeout=120,
        allowed_tries=3,
        **parser_extra,
    )
    defaults["parser"] = parser_llm
    defaults["summarizer"] = parser_llm

    # Two independent research sources: AskNews (free for tournament bots) and a
    # search-backed model. Either may be missing; research then uses what exists.
    # Metaculus's OpenRouter credits cover only OpenAI, Anthropic and Google models,
    # so the searcher there is Sonnet 5.5 with ":online" (the provider's own web
    # search; OpenRouter's Exa fallback is not covered). A personal-key run never
    # touches the OpenRouter key: there Sonnet 5.5 calls Anthropic's web search tool itself.
    search_llm: GeneralLlm | None = None
    if os.getenv("PERPLEXITY_API_KEY"):
        search_llm = GeneralLlm(model="perplexity/sonar-pro", temperature=0.1)
    elif route == "openrouter":
        search_llm = GeneralLlm(
            model=trial_model("researcher")
            or os.getenv("SEARCH_MODEL")
            or f"{SONNET_MODEL['openrouter']}:online",
            temperature=None,
            timeout=300,
            allowed_tries=2,
        )
    else:
        search_llm = AnthropicSearchLlm(
            model=trial_model("researcher") or SONNET_MODEL["anthropic"],
            temperature=None,
            timeout=300,
            allowed_tries=2,
            tools=[dict(ANTHROPIC_SEARCH_TOOL)]
            + ([dict(ANTHROPIC_FETCH_TOOL)] if _env_flag("RESEARCH_WEB_FETCH", True) else []),
            **room,
        )
    if os.getenv("RESEARCHER"):
        defaults["researcher"] = os.getenv("RESEARCHER")
        defaults["researcher_2"] = None
    elif _has_asknews():
        defaults["researcher"] = ASKNEWS_RESEARCHER
        defaults["researcher_2"] = search_llm
    elif search_llm is not None:
        defaults["researcher"] = search_llm
        defaults["researcher_2"] = None
    else:
        # No search provider at all: research from the forecaster's own knowledge
        # (weak for current events; set ASKNEWS_* or a search key)
        defaults["researcher"] = defaults["default"]
        defaults["researcher_2"] = None
    return defaults


class RcForecastBot(ForecastBot):
    """
    rc_trader's entry in the Metaculus AI forecasting tournament (FutureEval), built on
    Metaculus's template bot. Changes from the template:
    - Claude models, paid by Metaculus's donated credits (OPENROUTER_API_KEY) before a
      personal key (ANTHROPIC_API_KEY). On the donated credits, budget.py paces spending:
      each run reads the key's balance and picks Opus 5.5 or Sonnet 5.5 and the number of
      predictions per tournament, MiniBench never below FutureEval. Rob's own key runs
      the same tiers once the credits can't pay, or with no OpenRouter key; only the
      Metaculus Cup's runs Sonnet 5.5 unpaced. Haiku 5.5 parses; PARSER_MODEL overrides it.
    - Two independent research sources per report: AskNews news search (one latest-news
      call by default, to stay inside the free tier; ASKNEWS_STRATEGY and ASKNEWS_ARTICLES
      change it) and a search-backed model (Sonnet 5.5 with ":online" through OpenRouter, or
      Perplexity with its own key). A single search's luck was the largest source of
      night-to-night noise in rc_trader's own forecasts. When every source fails, the
      question waits for the next run unless it closes first; each run's page on GitHub
      counts what every source returned.
    - Every prompt gets today's date and the question's close and resolution times.
    - Each run's page shows what every model call cost, by OpenRouter's own figures, per
      question and per part of the work: each research source, each forecast, and parsing
      with the research summary (spend.py). Measuring changes no request.
    - Binary questions: an outside-view-first prompt that restates the criteria, reads the
      status quo from the named source, treats liquid matched markets as strong evidence
      and checks its number in both directions; a trimmed mean (drop the highest and
      lowest) once there are four or more predictions.
    - Tournament ids overridable by env (AIB_TOURNAMENT_ID, MINIBENCH_ID) so a new season
      doesn't wait on a forecasting-tools release.

    The template's original notes follow.

    This is the template bot for Summer 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Summer AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(
        self,
        *args,
        research_health: ResearchHealth | None = None,
        extra_forecasters: list[GeneralLlm] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Shared across a run's passes, so the run page counts every search
        self.research_health = (
            research_health if research_health is not None else ResearchHealth()
        )
        # The last forecasts of each research report go to these, in order (make_bot adds
        # their number to predictions_per_research_report)
        self.extra_forecasters = list(extra_forecasters or [])
        # Per question: forecasts by the bot's own model that succeeded (the success bar)
        self._own_succeeded: Counter[str] = Counter()
        # Per question: research reports tried so far, and questions any report found
        # research for (RESEARCH_REPORTS can run several reports for one question)
        self._reports_tried: Counter[str] = Counter()
        self._research_found: set[str] = set()
        # Per question: forecasts started, to number them in the spend breakdown
        self._forecasts_started: Counter[str] = Counter()

    ##################################### MODELS #####################################

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        """Claude by default (see build_llms); forecasting-tools' own defaults when
        neither OPENROUTER_API_KEY nor ANTHROPIC_API_KEY is set."""
        return build_llms(llm_route())

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            self._reports_tried[question_key(question)] += 1
            prompt = self._research_prompt(question)
            sources = research_sources(self._llms)
            parts: list[str] = []
            refused: BaseException | None = None  # a source Anthropic wouldn't pay for
            for source in sources:
                name = self._source_name(source)
                try:
                    with spend.tagged(part=f"research: {name}"):
                        text = await self._run_one_research_source(source, prompt, question)
                except Exception as e:  # one failed source must not sink the question
                    logger.warning(
                        f"Research source {name} failed for {question.page_url}:"
                        f" {type(e).__name__}: {e}"
                    )
                    self.research_health.record(name, type(e).__name__)
                    if spend_limit_reached(e):
                        refused = e
                    continue
                if not (text and text.strip()):
                    logger.warning(
                        f"Research source {name} returned nothing for {question.page_url}"
                    )
                    self.research_health.record(name, "returned nothing")
                    continue
                self.research_health.record(name)
                parts.append(f"## Research from {name}\n{flatten_headings(text)}")
            if parts:
                self._research_found.add(question_key(question))
            if sources and not parts and refused is not None:
                # Rob's key at its spend limit, not research gone missing: the run counts
                # the question as stopped by the limit (quiet pause), not as a red failure
                raise refused
            if sources and not parts:
                research = self._without_research(question)
            else:
                research = "\n\n".join(parts)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    def _without_research(self, question: MetaculusQuestion) -> str:
        """Every source failed for this research report. A question a later run can still
        reach waits for it (the error leaves it unforecast, so the next run picks it up).
        One closing sooner is forecast without research, and the forecaster is told so,
        but only from its last report and only if no report found any: otherwise this
        report alone is dropped and the question is forecast from the research found."""
        key = question_key(question)
        if self._can_retry(question):
            raise ResearchUnavailable(
                f"No research for {question.page_url}: every source failed. Not forecast"
                " now; the next run tries again."
            )
        if (
            key in self._research_found
            or self._reports_tried[key] < self.research_reports_per_question
        ):
            raise ResearchUnavailable(
                f"No research for {question.page_url} in this report; its other"
                " reports go ahead"
            )
        logger.warning(
            f"No research for {question.page_url}, which closes before another try:"
            " forecasting without it"
        )
        self.research_health.without_research.append(key)
        return NO_RESEARCH_NOTE

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        """The template's summary for the comment, its headings kept as bold lines: on
        2026-10-05 Haiku opened every summary with a "# " heading (see flatten_headings)."""
        with spend.tagged(part="summary"):
            summary = await super().summarize_research(question, research)
        return flatten_headings(summary)

    ##################################### SPEND #####################################

    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """The template's, unchanged; every model call it makes counts toward this
        question in the run page's spend breakdown."""
        token = _QUESTION.set(question_key(question))
        try:
            with spend.tagged(question=question_key(question)):
                return await super()._run_individual_question(question)
        finally:
            _QUESTION.reset(token)

    async def _handle_errors_in__run_individual_question(
        self, all_predictions, research_errors, all_errors, exception_group
    ) -> None:
        """The package's check that enough forecasts succeeded (half by default), counting
        the bot's own model only once there are extra forecasters: an extra that fails
        (no OpenRouter credit, say) only thins the question, and the extras can't carry a
        question on their own when the bot's own model fails (its key at its spend limit)."""
        if not self.extra_forecasters:
            return await super()._handle_errors_in__run_individual_question(
                all_predictions, research_errors, all_errors, exception_group
            )
        if research_errors:
            logger.warning(f"Encountered errors while researching: {research_errors}")
        own = self._own_succeeded[_QUESTION.get() or ""]
        own_expected = self.research_reports_per_question * (
            self.predictions_per_research_report - len(self.extra_forecasters)
        )
        needed = own_expected * self.required_successful_predictions
        if own == 0 and exception_group is not None:
            self._reraise_exception_with_prepended_message(
                exception_group, "Every forecast by the bot's own model failed"
            )
        if own < needed:
            raise ValueError(
                f"Expected at least {needed} successful predictions from the bot's own model,"
                f" but only got {own}. Errors encountered: {all_errors}"
            )

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[PredictionTypes]:
        """The template's, made by the forecast's own model (see _forecaster); its calls
        count as forecast 1, 2, 3... of the question in the spend breakdown (its parsing
        calls as parsing, see parse)."""
        if getattr(question, "conditional_type", None) is not None:
            # a leg of a conditional question: its member's slot and model, set by the
            # outer call, carry over, so all four legs of one member use one model
            return await super()._make_prediction(question, research)
        key = question_key(question)
        self._forecasts_started[key] += 1
        slot = self._forecasts_started[key]
        extra = self._extra_for(slot)
        token = _EXTRA_FORECASTER.set(extra)
        try:
            with spend.tagged(part=f"forecast {slot}"):
                prediction = await super()._make_prediction(question, research)
        except Exception as e:
            if extra is not None:
                self.research_health.record_forecast(extra.model, type(e).__name__)
            raise
        finally:
            _EXTRA_FORECASTER.reset(token)
        if extra is not None:
            self.research_health.record_forecast(extra.model)
        else:
            self._own_succeeded[key] += 1
        return prediction

    def _extra_for(self, slot: int) -> GeneralLlm | None:
        """The extra forecaster for a question's forecast number `slot`, or None for the
        default model: in each research report the last ones go to the extras, in order."""
        if not self.extra_forecasters:
            return None
        per_report = self.predictions_per_research_report
        index = (slot - 1) % per_report
        first = per_report - len(self.extra_forecasters)
        return self.extra_forecasters[index - first] if first >= 0 and index >= first else None

    def _forecaster(self) -> GeneralLlm:
        """The model for the forecast being made: its extra forecaster, else the default."""
        return _EXTRA_FORECASTER.get() or self.get_llm("default", "llm")

    @classmethod
    def _format_and_expand_research_summary(
        cls,
        report_number: int,
        report_type: type[ForecastReport],
        predicted_research: ResearchWithPredictions,
    ) -> str:
        """The package's summary of one research report, with each forecaster's model on
        the line after its prediction (see MODEL_LINE)."""
        bullets = ""
        for j, forecast in enumerate(predicted_research.predictions):
            readable = report_type.make_readable_prediction(forecast.prediction_value)
            model = model_of_reasoning(forecast.reasoning)
            # after a list (multiple choice, numeric) the model line is its own paragraph,
            # and each forecaster ends with a blank line, as in the package's original
            gap = "\n\n" if readable.endswith("\n") else "\n"
            tag = f"{gap}{MODEL_LINE.format(model)}" if model else ""
            bullets += f"*Forecaster {j + 1}*: {readable.rstrip()}{tag}\n\n"
        return clean_indents(
            f"""
            ## Report {report_number} Summary
            ### Forecasts
            {bullets}

            ### Research Summary
            {predicted_research.summary_report}
            """
        )

    @staticmethod
    def _can_retry(question: MetaculusQuestion, now: datetime | None = None) -> bool:
        if question.close_time is None:
            return True
        now = now or datetime.now(timezone.utc)
        return question.close_time - now > RESEARCH_RETRY_MARGIN

    @staticmethod
    def _source_name(source: GeneralLlm | str) -> str:
        return source_name(source)

    def _research_prompt(self, question: MetaculusQuestion) -> str:
        """The research brief. Besides the news, it asks for the two things the
        Spring 2026 bot survey found most associated with better scores: the current
        reading of the source the criteria name (opened directly where the web fetch tool
        is on), and forecasts on the question or close ones elsewhere, venue by venue so a
        missing one reads "not found" rather than an invented number. Using published
        forecasts, Metaculus's own included, is within the tournament's rules. It also asks
        for a searched base rate as a count, and for a price series' latest value and
        typical move, which the forecast prompts use."""
        event_base_rate = (
            "Search once for the base rate: how often this kind of event happened, as a count"
            " that says what was counted: years with at least one such event (for example"
            ' "in 3 of the last 20 years"), windows like this question\'s (for example "2 of'
            ' the last 15 Novembers"), or events per year. If an exact figure is missing or'
            " paywalled, give the closest proxy and say it is one."
        )
        if isinstance(question, MultipleChoiceQuestion):
            outcome = "which option the question would resolve to"
            options = f"\nThe options are: {question.options}\n"
            base_rate = event_base_rate
        elif isinstance(question, (NumericQuestion, DateQuestion)):
            outcome = "what the question would resolve to"
            options = ""
            base_rate = (
                "Unless it is a traded price, give past values of the same quantity in"
                " comparable periods, with their dates."
            )
        else:
            outcome = "whether the question would resolve Yes or No"
            options = ""
            base_rate = event_base_rate
        return clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including {outcome} based on current information.
            Include dates for every fact, and any scheduled events before the question resolves.
            If the resolution criteria or fine print name a source (an official page, a data series, a report) or link to one, open it first if you can. Give its latest value or status and when it was published, any schedule or prediction it publishes for the period (a court calendar, a release date, tide predictions), and whether the outcome already looks settled.
            {base_rate}
            For a price, index or other data series the question turns on, give its latest value with its date and how volatile it is: the standard deviation of its daily % changes, or failing that its annualized or option-implied volatility, saying which figure you give.
            For each of Polymarket, Kalshi, Manifold and Metaculus, say whether you found a forecast on this question or a close one: give it with its date and a link, and say how closely that question matches this one's criteria, or write "not found" (searched) or "not searched".
            You do not produce forecasts yourself.

            {timeline(question)}

            Question:
            {question.question_text}
            {options}
            Background:
            {question.background_info or "None given."}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )

    async def _run_one_research_source(
        self, researcher: GeneralLlm | str, prompt: str, question: MetaculusQuestion
    ) -> str:
        """The template's researcher dispatch for one source, plus AskNews as
        ASKNEWS_STRATEGY sets it (queried with the question itself, which searches better
        than the prompt)."""
        if isinstance(researcher, GeneralLlm):
            return await researcher.invoke(prompt)
        if researcher == ASKNEWS_RESEARCHER:
            return await ConfiguredAskNewsSearcher().get_formatted_news_async(
                question.question_text
            )
        if researcher in (
            "asknews/news-summaries",
            "asknews/deep-research/low-depth",
            "asknews/deep-research/medium-depth",
            "asknews/deep-research/high-depth",
        ):
            return await AskNewsSearcher().call_preconfigured_version(researcher, prompt)
        if researcher.startswith("smart-searcher"):
            model_name = researcher.removeprefix("smart-searcher/")
            searcher = SmartSearcher(
                model=model_name,
                temperature=0,
                num_searches_to_run=2,
                num_sites_per_search=10,
                use_advanced_filters=False,
            )
            return await searcher.invoke(prompt)
        if not researcher or researcher in NO_RESEARCH_SETTINGS:
            return ""
        return await GeneralLlm(model=researcher).invoke(prompt)

    ##################################### AGGREGATION #####################################

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        """Binary: trimmed mean (drop the single highest and lowest) once there are at
        least four predictions; it beat the median in Halawi et al. (2024). Numeric,
        discrete and date: the mixture of the forecasts (their mean CDF), less any far
        from the others, in place of the package's pointwise median (forecast_guards.py).
        Multiple choice keeps forecasting-tools' mean per option."""
        if isinstance(question, BinaryQuestion) and len(predictions) >= 4:
            values = sorted(float(p) for p in predictions)  # type: ignore[arg-type]
            return float(statistics.mean(values[1:-1]))  # type: ignore[return-value]
        if isinstance(question, (NumericQuestion, DateQuestion)) and len(predictions) >= 2:
            return forecast_guards.mixture(predictions, question)  # type: ignore[arg-type,return-value]
        return await super()._aggregate_predictions(predictions, question)

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a superforecaster with an excellent calibration record. Your forecast is scored with a log score, so confident errors are punished hard and hedging when you have real evidence also costs you.

            Question:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Research gathered today from independent sources (it may be incomplete, outdated or wrong; weigh each item by its source and date):
            {research}

            {timeline(question)}

            Work through these steps briefly before answering:
            (a) The question: in one sentence, what exactly must happen, by when, and according to which source. Forecast that question, not a related one.
            (b) Time left: how long until the outcome is known, and how much can realistically change in that time?
            (c) Outside view: name the most relevant reference class and its base rate. If several apply, say which you trust most and why. For an event that could happen on any day, convert the base rate to the time left: from the share of years with at least one such event, p, about 1 - (1 - p)^(days left / 365); from a number of events per year, r, about 1 - e^(-r x days left / 365). A base rate already counted over windows like this question's (the same months in past years) needs no rescaling.
            (d) Status quo: start from the current state the named source shows. Decide whether the status quo means nothing changes, or a process already under way reaching its scheduled end (a deadline arriving, a trend continuing). The world changes slowly most of the time, so give the status quo extra weight, especially when little time is left.
            (e) Inside view: the specific, dated evidence that should move you away from the base rate, and roughly how far. Separate facts from speculation, and note what you could not find out. An announced target date shows a plan exists, not that it will be met; official timelines often slip. A search that finds nothing is not evidence that something did not happen, unless the source would certainly have reported it.
            (f) Markets and prices: a recent forecast on this question or a close match from a liquid market or Metaculus is strong evidence: move away from it only for a reason you can name, such as different criteria or dates. A thin or stale market is weak evidence. If the question asks whether a traded price or index will be past a level, also estimate where it will be (its latest value, and its {DAILY_VOLATILITY} times the square root of the days it trades before the deadline), then read off the chance. Reaching a level not yet reached at any time before the deadline is more likely than being past it on the final day: about twice as likely if intraday prices count and several weeks are left. If only daily closes count it is less: about the same with 1 trading day left, 1.2 times with 2-3, 1.4 times at a week, 1.5 times at two weeks, and somewhat less again for levels far from today's price.
            (g) The strongest case for the outcome you currently think less likely. Then check your number both ways: out of 100 questions like this one, about how many resolve Yes? Would a Yes surprise you at this probability? Would a No?
            (h) Reconcile these into one probability. Avoid 0% and 100%, and go below 3% or above 97% only when the outcome is effectively already determined.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        forecaster = self._forecaster()
        reasoning = await forecaster.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await parse(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(
            prediction_value=decimal_pred, reasoning=with_model(forecaster.model, reasoning)
        )

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            {timeline(question)}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) For a contest or ranking: how often the favourite or current leader actually wins in comparable cases and, if the favourite fails, the most likely way it happens.
            (d) Any recent forecast from a liquid market or Metaculus on this question or a close match. Treat it as strong evidence and move away from it only for a reason you can name, such as different criteria or dates; a thin or stale market is weak evidence.
            (e) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            Your probabilities must add up to 100%.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        forecaster = self._forecaster()
        reasoning = await forecaster.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        # Parsed without the package's validator, which rejects a confident answer with
        # several 0% options and would do so again on every retry; option_list floors
        # and renormalises instead (forecast_guards.py)
        guesses: list[forecast_guards.OptionGuess] = await parse(
            text_to_structure=reasoning,
            output_type=list[forecast_guards.OptionGuess],
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        predicted_option_list = forecast_guards.option_list(guesses, question.options)

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list,
            reasoning=with_model(forecaster.model, reasoning),
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            {timeline(question)}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            (g) If the question is about a traded price, index or rate: its latest value and its {DAILY_VOLATILITY}. Unless the value cannot move before resolution (a policy rate with no decision in the window) or a hard limit holds it, keep your 10th to 90th percentile range at least as wide as about 1.3 x that daily volatility x the square root of the days it trades before the deadline, on each side of the latest value, in percent terms. If earnings, a central-bank meeting or another scheduled release that can move it falls inside the window, go wider than this.

            {self._get_conditional_disclaimer_if_necessary(question)}
            {WIDTH_RULE}

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        forecaster = self._forecaster()
        reasoning = await forecaster.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await parse(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(
            forecast_guards.spread_repeats(percentile_list, question), question
        )
        # Build the CDF now, so an answer that can't make one fails only this forecast,
        # not the whole question at aggregation after every forecast is paid for
        prediction.get_cdf()
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=with_model(forecaster.model, reasoning)
        )

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            {timeline(question)}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            {WIDTH_RULE}

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        forecaster = self._forecaster()
        reasoning = await forecaster.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await parse(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(
            forecast_guards.spread_repeats(percentile_list, question), question
        )
        prediction.get_cdf()  # see _numeric_prompt_to_forecast
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=with_model(forecaster.model, reasoning)
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


def _tournament_id(value: str | None, default: int | str) -> int | str:
    """An env override (numeric id or slug), or the default."""
    value = (value or "").strip()
    if not value:
        return default
    return int(value) if value.isdigit() else value


def soonest_closing(
    questions: list[MetaculusQuestion], limit: int | None
) -> list[MetaculusQuestion]:
    """Up to `limit` questions (None = all), those closing soonest first."""
    ordered = sorted(
        questions,
        key=lambda q: q.close_time.timestamp() if q.close_time else math.inf,
    )
    return ordered if limit is None else ordered[: max(limit, 0)]


def one_of_each_type(
    questions: list[MetaculusQuestion], limit: int
) -> list[MetaculusQuestion]:
    """Up to `limit` questions, covering as many question types as possible."""
    firsts: list[MetaculusQuestion] = []
    rest: list[MetaculusQuestion] = []
    seen: set[type] = set()
    for question in questions:
        (rest if type(question) in seen else firsts).append(question)
        seen.add(type(question))
    return (firsts + rest)[:limit]


def make_bot(
    route: str | None,
    tier: budget.Tier | None,
    publish: bool,
    skip_previous: bool = True,
    research_health: ResearchHealth | None = None,
) -> RcForecastBot:
    """One research report (two independent sources) per question. A pacing tier (on
    Metaculus's credits, or on Rob's own key once they run out) sets the forecaster and
    the number of predictions; without one, on a personal key, RESEARCH_REPORTS and
    PREDICTIONS_PER_REPORT do (1 and 5 by default)."""
    if tier is not None:
        reports, predictions = 1, tier.predictions
    else:
        reports = _env_int("RESEARCH_REPORTS", 1)
        predictions = _env_int("PREDICTIONS_PER_REPORT", 5)
    # never more extras than the bot's own forecasts, so with fewer than four members the
    # median can't leave the bot's own model out (one own forecast: one extra, a mean)
    extras = build_extra_forecasters()[:predictions]
    return RcForecastBot(
        research_reports_per_question=reports,
        predictions_per_research_report=predictions + len(extras),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previous,
        extra_metadata_in_explanation=True,
        llms=build_llms(route, tier),
        research_health=research_health,
        extra_forecasters=extras,
    )


def describe_pass(
    label: str,
    tournament: int | str,
    open_count: int,
    taken: int,
    new_count: int | None = None,
    note: str = "",
) -> str:
    """One line of the run page per tournament. "No open questions" and "nothing new"
    read the same in the log otherwise, and only the first means the bot sees nothing."""
    if not open_count:
        return f"- {label} (`{tournament}`): no open questions."
    counts = [f"{open_count} open"]
    if new_count is not None:
        counts.append(f"{new_count} not yet forecast")
    counts.append(f"{taken} taken this run")
    return f"- {label} (`{tournament}`): {', '.join(counts)}" + (f" ({note})." if note else ".")


def health_warnings(health: ResearchHealth) -> list[str]:
    """A GitHub warning annotation for each research source that failed this run. A source
    that fails while the other still feeds the question leaves the run green, so without
    this it shows only in the run page's table; an annotation shows on the run's summary
    and in GitHub's API."""
    warnings = []
    kinds = (
        (health.sources, "Research source failed", "searches"),
        (health.forecasters, "Extra forecaster failed", "forecasts"),
    )
    for tallies, title, unit in kinds:
        for name, tally in tallies.items():
            failed = tally.searches - tally.worked
            if failed:
                reasons = ", ".join(f"{r} x{n}" for r, n in tally.failures.most_common())
                warnings.append(
                    f"::warning title={title}::{name} failed {failed} of"
                    f" {tally.searches} {unit} ({reasons})"
                )
    return warnings


def _tally_rows(tallies: dict[str, SourceTally]) -> list[str]:
    rows = []
    for name, tally in tallies.items():
        reasons = ", ".join(
            f"{reason} ×{count}" if count > 1 else reason
            for reason, count in tally.failures.most_common()
        )
        failed = f"{tally.searches - tally.worked}" + (f" ({reasons})" if reasons else "")
        rows.append(f"| {name} | {tally.searches} | {tally.worked} | {failed} |")
    return rows


def run_report(
    passes: list[str],
    configured: list[str],
    health: ResearchHealth,
    results: list[ForecastReport | BaseException],
) -> list[str]:
    """Markdown for the log and the Actions run page: what each tournament had, which
    research sources are set up and what each returned. An idle run still names the
    sources, so a newly added key (AskNews's, say) shows on the next run."""
    lines = ["### This run", "", *passes]
    lines.append(f"- Research sources: {', '.join(configured) if configured else 'none'}.")
    if health.sources:
        lines += ["", "| Research source | Searches | Worked | Failed |", "|---|---|---|---|"]
        lines += _tally_rows(health.sources)
    if health.forecasters:
        lines += ["", "| Extra forecaster | Forecasts | Worked | Failed |", "|---|---|---|---|"]
        lines += _tally_rows(health.forecasters)
    notes: list[str] = []
    deferred = sum(deferred_for_research(r) for r in results)
    if deferred:
        notes.append(
            f"- {deferred} question(s) not forecast: every research source failed. The next"
            " run tries again."
        )
    # Only questions whose forecast then went through, each once
    forecast = {
        question_key(r.question)
        for r in results
        if not isinstance(r, BaseException) and hasattr(r, "question")
    }
    blind = [key for key in dict.fromkeys(health.without_research) if key in forecast]
    if blind:
        notes.append(
            f"- Forecast without research, closing before another try: {', '.join(blind)}."
        )
    if notes:
        lines += ["", *notes]
    return lines


def question_statuses(
    questions: list[MetaculusQuestion],
    results: list[ForecastReport | BaseException],
    publish: bool,
) -> dict[str, str]:
    """What became of each question, for the spend breakdown: research a question paid
    for still shows when it waits for the next run or fails."""
    statuses: dict[str, str] = {}
    for question, result in zip(questions, results):
        if deferred_for_research(result):
            status = "waits for research"
        elif isinstance(result, BaseException):
            status = "failed"
        else:
            status = "posted" if publish else "dry run"
        statuses[question_key(question)] = status
    return statuses


def pay_with() -> str:
    """PAY_WITH (the Test Bot's input): credits, the default, or own-key, which makes Rob's
    own Anthropic key pay even while Metaculus's credits last."""
    value = os.getenv("PAY_WITH", "").strip().lower() or "credits"
    if value not in ("credits", "own-key"):
        raise SystemExit(f"PAY_WITH must be credits or own-key; got {value!r}")
    if value == "own-key" and not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("PAY_WITH=own-key needs the ANTHROPIC_API_KEY secret")
    return value


def credits_spent(credits_left: float | None, plan: budget.Plan) -> str | None:
    """Why Metaculus's credits can't pay for this run, or None (budget.exhausted)."""
    if budget.under_floor(credits_left):
        return f"Metaculus's credits are under the ${budget.FLOOR:.0f} floor"
    if budget.exhausted(credits_left, plan):
        return (
            f"Metaculus's credits ({budget.dollars(credits_left)}) can't pay for another"
            f" question above the ${budget.FLOOR:.0f} reserve"
        )
    return None


def own_key_reason(credits_left: float | None, plan: budget.Plan) -> str | None:
    """Why Rob's own Anthropic key pays this run instead of Metaculus's credits, or None.
    His key after the credits is Rob's choice of 2026-10-11; it takes over once they can't
    pay for another question (budget.exhausted, Claude's rule in #8), if ANTHROPIC_API_KEY
    is set. PAY_WITH=own-key asks for it sooner."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    if pay_with() == "own-key":
        return "PAY_WITH=own-key"
    if credits_left is None:
        # Since the credits ran out, a run that can't read their balance (an OpenRouter
        # outage, or a key Metaculus has disabled) would go back onto them, uncapped, and
        # fail every question there; PR #8's second read covers only a brief failure
        return "Metaculus's credits couldn't be read"
    return credits_spent(credits_left, plan)


def credit_alerts(meter: budget.SpendMeter, plan: budget.Plan) -> list[str]:
    """Opens an issue the first time the credits fall below LOW_CREDIT, and another when
    they can't pay for another question (each once per credit limit). Lines for the run
    page."""
    left = budget.remaining_of(meter.start)
    if left is None or math.isinf(left) or left >= budget.LOW_CREDIT:
        return []
    limit = (meter.start or {}).get("limit")
    limit = None if limit is None else float(limit)
    own_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if credits_spent(left, plan):
        opened = alerts.credits_out(left, limit, own_key)
    else:
        opened = alerts.credits_low(left, limit, own_key)
    return ["- Opened an issue to tell Rob about the credits."] if opened else []


class OwnKey:
    """Rob's own Anthropic key during one run. It is checked once, before the first
    question it would pay for, so a key that can't pay costs no AskNews calls or
    research, and the run stays green (one issue tells Rob instead)."""

    def __init__(self) -> None:
        self.checked = False
        self.problem: str | None = None

    def stopped(self) -> str | None:
        """Why the key can't pay this run, or None; checks it the first time."""
        if not self.checked:
            self.checked = True
            self.problem = own_key_check()
            if self.problem == "limit":
                alerts.own_key_limit()
            elif self.problem:
                alerts.own_key_rejected(self.problem.removeprefix("rejected: "))
        if self.problem == "limit":
            return "your Anthropic key reached its spend limit"
        if self.problem:
            return f"Anthropic rejected your key ({self.problem.removeprefix('rejected: ')})"
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    # A mistyped AskNews, effort or PAY_WITH setting stops the run here, not quietly later
    asknews_strategy()
    asknews_articles()
    forecaster_effort()
    pay_with()
    # A model on trial (the Test Bot's inputs) never reaches a tournament or the Cup
    trials = {role: trial_model(role) for role in TRIAL_MODELS}
    if run_mode != "test_questions":
        for variable in TRIAL_MODELS.values():
            os.environ.pop(variable, None)
    extras = extra_forecaster_models()  # a mistyped EXTRA_FORECASTERS stops the run here
    extras_without_key = bool(requested_extras()) and not extras
    if extras_without_key and trials["extra"]:
        sys.exit(f"The Test Bot's extra forecasters need the {OWN_OPENROUTER_KEY} secret (Setup, step 11)")
    # PUBLISH=false gives a dry run: full research and forecasts, nothing posted
    publish = _env_flag("PUBLISH", True)
    print_startup_banner(run_mode, will_publish=publish)

    route = llm_route()
    if run_mode == "metaculus_cup":
        # Metaculus's credits are for FutureEval and MiniBench only
        if not os.getenv("ANTHROPIC_API_KEY"):
            sys.exit(
                "The Metaculus Cup runs only on your own ANTHROPIC_API_KEY: Metaculus's"
                " OpenRouter credits are for FutureEval and MiniBench."
            )
        route = "anthropic"

    # Pacing applies to Metaculus's credits. Once they can't pay for another question,
    # Rob's own Anthropic key takes over, capped by its workspace's monthly spend limit;
    # the Metaculus Cup's personal key runs unpaced.
    meter: budget.SpendMeter | None = None
    plan: budget.Plan | None = None
    own_key: OwnKey | None = None
    notes: list[str] = []
    if route == "openrouter":
        meter = budget.SpendMeter(os.environ["OPENROUTER_API_KEY"])
        plan = budget.make_plan(meter.remaining(), budget.pinned_tier())
        logger.info(
            f"Credits left: {budget.dollars(meter.remaining())}. FutureEval runs"
            f" {plan.futureeval.name}, MiniBench {plan.minibench.name} ({plan.note})."
        )
        notes += credit_alerts(meter, plan)
        reason = own_key_reason(meter.remaining(), plan)
        spent = credits_spent(meter.remaining(), plan)
        if reason:
            route = "anthropic"
            own_key = OwnKey()
            meter = budget.OwnKeyMeter(meter.start)
            plan = budget.own_key_plan(budget.pinned_tier(), reason)
            logger.info(
                f"Your Anthropic key pays this run ({reason}). FutureEval runs"
                f" {plan.futureeval.name}, MiniBench {plan.minibench.name}."
            )
        elif spent:
            # No key to take over: say so on the run page, rather than "later runs take
            # the rest" while nothing can pay
            pause = budget.TIERS["pause"]
            plan = budget.Plan(pause, pause, meter.remaining(), f"{spent}: paused")
    elif route == "anthropic" and run_mode != "metaculus_cup":
        # No OpenRouter key at all (say Metaculus's is removed now that it's spent): Rob's
        # key pays as it does after the credits, at the pinned tier, with its key check
        # and BUDGET_TIER=pause, rather than the Metaculus Cup's unpaced Sonnet 5.5 x 5
        own_key = OwnKey()
        meter = budget.OwnKeyMeter(None)
        plan = budget.own_key_plan(budget.pinned_tier(), "no OpenRouter key")
        logger.info(
            f"Your Anthropic key pays this run (no OpenRouter key). FutureEval runs"
            f" {plan.futureeval.name}, MiniBench {plan.minibench.name}."
        )

    client = MetaculusClient()
    tournament_id = _tournament_id(os.getenv("AIB_TOURNAMENT_ID"), FALL_2026_FUTUREEVAL_ID)
    minibench_id = _tournament_id(os.getenv("MINIBENCH_ID"), client.CURRENT_MINIBENCH_ID)
    cup_id = _tournament_id(os.getenv("METACULUS_CUP_ID"), FALL_2026_METACULUS_CUP_ID)
    # Tournament URL shown in the summary banner footer.
    tournament_urls = {
        "tournament": f"https://www.metaculus.com/tournament/{tournament_id}/",
        "metaculus_cup": f"https://www.metaculus.com/tournament/{cup_id}/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    health = ResearchHealth()
    passes: list[str] = []
    breakdowns: list[str] = []
    own_openrouter_spent = [0.0]  # the extra forecasters' spend on Rob's OpenRouter key
    own_openrouter_raised = [0]  # and their calls that raised (OpenRouter may still bill them)

    def run_pass(
        label: str,
        questions: list[MetaculusQuestion],
        tier: budget.Tier | None,
        skip_previous: bool = True,
    ) -> list[ForecastReport | BaseException]:
        if not questions:
            logger.info(f"{label}: no new questions")
            return []
        bot = make_bot(route, tier, publish, skip_previous, research_health=health)
        # Records what every model call in the pass cost, by OpenRouter's own figures:
        # failed questions, retries and questions left waiting for research included
        with spend.collecting() as ledger:
            results = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
        if own_key is not None and any(
            isinstance(r, BaseException) and spend_limit_reached(r) for r in results
        ):
            # Rob's key reached its limit mid-pass: later passes this run pause quietly
            own_key.checked, own_key.problem = True, "limit"
        # OpenRouter's records only on its own route; list prices fill gaps on every route
        ledger.look_up(os.environ["OPENROUTER_API_KEY"] if route == "openrouter" else "")
        breakdowns.extend(
            spend.breakdown(label, ledger, question_statuses(questions, results, publish))
        )
        # The extra forecasters bill Rob's own OpenRouter key, not the key the meter tracks
        extra_models = {llm.model for llm in bot.extra_forecasters}
        extra_spent = spend_on(ledger.calls, extra_models)
        own_openrouter_spent[0] += extra_spent
        own_openrouter_raised[0] += sum(n for m, n in ledger.raised_models.items() if m in extra_models)
        if meter is not None and tier is not None:
            failed = sum(isinstance(r, BaseException) for r in results)
            meter.record(label, tier, len(results) - failed, failed, ledger.total() - extra_spent)
        return results

    def paused_by(tier: budget.Tier | None, has_work: bool) -> str | None:
        """Why a pass forecasts nothing ("paused", or why Rob's key can't pay), or None.
        Rob's key is checked only when a pass has work for it."""
        if tier is not None and tier.predictions == 0:
            return "paused"
        if has_work and own_key is not None:
            stop = own_key.stopped()
            if stop:
                return f"paused: {stop}"
        return None

    reports: list[ForecastReport | BaseException] = []
    if run_mode == "tournament":
        room = MAX_QUESTIONS_PER_RUN
        tournaments = (
            ("FutureEval", tournament_id, plan.futureeval if plan else None),
            ("MiniBench", minibench_id, plan.minibench if plan else None),
        )
        for label, tid, tier in tournaments:
            open_questions = client.get_all_open_questions_from_tournament(tid)
            new = [q for q in open_questions if not q.already_forecasted]
            why = paused_by(tier, bool(new))
            if why:
                if new:
                    logger.warning(f"{label}: {why}; {len(new)} new question(s) skipped")
                passes.append(describe_pass(label, tid, len(open_questions), 0, len(new), why))
                continue
            limit = room
            if meter is not None and tier is not None:
                can_pay = budget.affordable(meter.remaining(), tier)
                if can_pay is not None:
                    limit = min(limit, can_pay)
            chosen = soonest_closing(new, limit)
            note = ""
            if len(chosen) < len(new):
                logger.warning(
                    f"{label}: forecasting {len(chosen)} of {len(new)} new questions"
                    " (run size or credit); later runs take the rest"
                )
                note = "run size or credit; later runs take the rest"
            passes.append(
                describe_pass(label, tid, len(open_questions), len(chosen), len(new), note)
            )
            reports += run_pass(label, chosen, tier)
            room -= len(chosen)
    elif run_mode == "metaculus_cup":
        # Re-forecasts every open question. The Metaculus Cup may be uninitialized near
        # the start of a season (Jan/May/Sep).
        questions = client.get_all_open_questions_from_tournament(cup_id)
        passes.append(
            describe_pass(
                "Metaculus Cup", cup_id, len(questions), len(questions), note="re-forecast"
            )
        )
        reports += run_pass("Metaculus Cup", questions, None, skip_previous=False)
    elif run_mode == "test_questions":
        # The bot-testing-area tournament has every question type. TEST_QUESTIONS (3 by
        # default, one of each type where possible) keeps the smoke test cheap. It runs
        # MiniBench's tier, the richer of the two.
        # https://www.metaculus.com/tournament/bot-testing-area/
        tier = plan.minibench if plan else None
        why = paused_by(tier, True)
        if why:
            logger.warning(f"bot-testing-area: {why}; nothing forecast")
            passes.append(f"- bot-testing-area: {why}; nothing forecast.")
        else:
            testing = client.get_all_open_questions_from_tournament("bot-testing-area")
            questions = one_of_each_type(testing, _env_int("TEST_QUESTIONS", 3))
            passes.append(
                describe_pass("bot-testing-area", "bot-testing-area", len(testing), len(questions))
            )
            reports += run_pass("bot-testing-area", questions, tier, skip_previous=False)
        trying = [f"{role} `{model}`" for role, model in trials.items() if model and (role != "extra" or extras)]
        if trying:
            notes.append(f"- On trial (Test Bot inputs): {', '.join(trying)}.")

    failures = [r for r in reports if isinstance(r, BaseException)]
    # Questions Anthropic refused to pay for on Rob's key mid-run: no red run (the next
    # run's key check pauses quietly), but one alert. On Metaculus's credits the same words
    # would be about someone else's account, so there they stay ordinary failures.
    limited = sum(spend_limit_reached(r) for r in failures) if route == "anthropic" else 0
    if limited:
        alerts.own_key_limit()
        notes.append(
            f"- {limited} question(s) not forecast: your Anthropic key reached its spend"
            " limit. Runs stay paused until it resets or you raise it."
        )

    RcForecastBot.log_report_summary(reports, raise_errors=False)
    print_run_summary_banner(
        reports,
        will_publish=publish,
        tournament_url=tournament_urls.get(run_mode),
    )
    if meter is not None and plan is not None:
        lines = meter.summary(plan)
        print("\n".join(lines) + "\n")
        budget.write_step_summary(lines)
    if breakdowns:
        print("\n".join(breakdowns) + "\n")
        budget.write_step_summary(breakdowns)
    configured = [source_name(s) for s in research_sources(build_llms(route))]
    lines = run_report(passes, configured, health, reports)
    if extras:
        counts = Counter(extras)
        names = ", ".join(f"{m} ×{n}" if n > 1 else m for m, n in counts.items())
        raised = own_openrouter_raised[0]
        unbilled = (
            f" {raised} of their calls raised (a timeout, say); OpenRouter may still bill"
            " those, and their cost isn't counted."
            if raised
            else ""
        )
        notes.append(
            f"- Extra forecasters: {names}. Spent on your own OpenRouter key this run:"
            f" ${own_openrouter_spent[0]:.2f}.{unbilled}"
        )
    elif extras_without_key:
        notes.append(f"- Extra forecasters off: EXTRA_FORECASTERS is set but {OWN_OPENROUTER_KEY} isn't.")
        print(f"::warning title=Extra forecasters off::EXTRA_FORECASTERS is set but {OWN_OPENROUTER_KEY} isn't")
    if notes:
        lines += ["", *notes]
    print("\n".join(lines) + "\n")
    budget.write_step_summary(lines)
    for warning in health_warnings(health):
        print(warning)
    failed = len(failures) - limited
    if failed:
        # a failed question turns the Actions run red; one that waits for research too,
        # so a source that keeps failing doesn't go unnoticed
        deferred = sum(deferred_for_research(r) for r in failures)
        why = f" ({deferred} for lack of research; the next run retries them)" if deferred else ""
        sys.exit(f"{failed} question(s) failed{why}; see the log above")


if __name__ == "__main__":
    main()
