import argparse
import asyncio
import logging
import math
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import dotenv

# Pacing for Metaculus's incremental OpenRouter credits.
import budget

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
    MonetaryCostManager,
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

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

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

# forecasting-tools 0.2.x (the template's pin) still points at the Summer 2026 season;
# its 0.3.1 release carries these Fall 2026 ids. AIB_TOURNAMENT_ID and METACULUS_CUP_ID
# (numeric id or slug) override them for later seasons.
FALL_2026_FUTUREEVAL_ID = 33121  # https://www.metaculus.com/tournament/fall-futureeval-2026/
FALL_2026_METACULUS_CUP_ID = 33108  # https://www.metaculus.com/tournament/metaculus-cup-fall-2026/

# Research runs one question at a time (a minute or so each), so a burst such as a new
# MiniBench round would outlast the workflow's 60-minute timeout. Runs come every 20
# minutes and pick up the rest.
MAX_QUESTIONS_PER_RUN = 25


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
        formatted = "".join(sections)
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
    source is otherwise only a warning in the log, and the run stays green."""

    sources: dict[str, SourceTally] = field(default_factory=dict)
    # Questions (by question_key) whose forecast went ahead with no research
    without_research: list[str] = field(default_factory=list)

    def record(self, source: str, error: str | None = None) -> None:
        tally = self.sources.setdefault(source, SourceTally())
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


def source_name(source: GeneralLlm | str) -> str:
    if isinstance(source, GeneralLlm):
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
    """The bot's models on one route. On Metaculus's OpenRouter credits the pacing tier
    picks the forecaster (Opus 5.5 without one); a personal Anthropic key runs Sonnet 5,
    or FORECASTER_MODEL. Haiku 4.5 parses on both; PARSER_MODEL overrides it."""
    defaults = dict(ForecastBot._llm_config_defaults())
    if route is None:
        defaults["researcher_2"] = None
        return defaults
    if route == "openrouter":
        prefix, haiku = "openrouter/anthropic/", "claude-haiku-4.5"
        forecaster = tier.forecaster if tier and tier.forecaster else budget.OPUS
    else:
        prefix, haiku = "anthropic/", "claude-haiku-4-5"
        forecaster = os.getenv("FORECASTER_MODEL") or f"{prefix}claude-sonnet-5"
    parser = os.getenv("PARSER_MODEL") or f"{prefix}{haiku}"
    defaults["default"] = GeneralLlm(
        model=forecaster,
        # Sonnet 5, Opus 5 and newer reject sampling parameters: send none
        temperature=None,
        timeout=_env_int("FORECASTER_TIMEOUT_S", 300),
        allowed_tries=2,
    )
    parser_llm = GeneralLlm(
        model=parser,
        temperature=0 if "haiku" in parser else None,
        timeout=120,
        allowed_tries=3,
    )
    defaults["parser"] = parser_llm
    defaults["summarizer"] = parser_llm

    # Two independent research sources: AskNews (free for tournament bots) and a
    # search-backed model. Either may be missing; research then uses what exists.
    # Metaculus's OpenRouter credits cover only OpenAI, Anthropic and Google models,
    # so the searcher there is Sonnet 5 with ":online" (the provider's own web
    # search; OpenRouter's Exa fallback is not covered). A personal-key run never
    # touches the OpenRouter key.
    search_llm: GeneralLlm | None = None
    if os.getenv("PERPLEXITY_API_KEY"):
        search_llm = GeneralLlm(model="perplexity/sonar-pro", temperature=0.1)
    elif route == "openrouter":
        search_llm = GeneralLlm(
            model=os.getenv("SEARCH_MODEL") or "openrouter/anthropic/claude-sonnet-5:online",
            temperature=None,
            timeout=300,
            allowed_tries=2,
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
      each run reads the key's balance and picks Opus 5.5 or Sonnet 5 and the number of
      predictions per tournament, MiniBench never below FutureEval. A personal key runs
      Sonnet 5 unpaced. Haiku 4.5 parses; PARSER_MODEL overrides it.
    - Two independent research sources per report: AskNews news search (one latest-news
      call by default, to stay inside the free tier; ASKNEWS_STRATEGY and ASKNEWS_ARTICLES
      change it) and a search-backed model (Sonnet 5 with ":online" through OpenRouter, or
      Perplexity with its own key). A single search's luck was the largest source of
      night-to-night noise in rc_trader's own forecasts. When every source fails, the
      question waits for the next run unless it closes first; each run's page on GitHub
      counts what every source returned.
    - Every prompt gets today's date and the question's close and resolution times.
    - Binary questions: an outside-view-first prompt with an explicit check against the
      known "Yes" lean of language models, and a trimmed mean (drop the highest and
      lowest) over all predictions instead of the median.
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
        self, *args, research_health: ResearchHealth | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Shared across a run's passes, so the run page counts every search
        self.research_health = (
            research_health if research_health is not None else ResearchHealth()
        )
        # Per question: research reports tried so far, and questions any report found
        # research for (RESEARCH_REPORTS can run several reports for one question)
        self._reports_tried: Counter[str] = Counter()
        self._research_found: set[str] = set()

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
            for source in sources:
                name = self._source_name(source)
                try:
                    text = await self._run_one_research_source(source, prompt, question)
                except Exception as e:  # one failed source must not sink the question
                    logger.warning(
                        f"Research source {name} failed for {question.page_url}:"
                        f" {type(e).__name__}: {e}"
                    )
                    self.research_health.record(name, type(e).__name__)
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
        return flatten_headings(await super().summarize_research(question, research))

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
        return clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            Include dates for every fact, relevant base rates or historical frequencies if you know them, and any scheduled events before the question resolves.
            You do not produce forecasts yourself.

            {timeline(question)}

            Question:
            {question.question_text}

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
        least four predictions; it beat the median in Halawi et al. (2024). Other types
        keep forecasting-tools' aggregation."""
        if isinstance(question, BinaryQuestion) and len(predictions) >= 4:
            values = sorted(float(p) for p in predictions)  # type: ignore[arg-type]
            return float(statistics.mean(values[1:-1]))  # type: ignore[return-value]
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
            (a) Time left: how long until the outcome is known, and how much can realistically change in that time?
            (b) Outside view: name the most relevant reference class and its base rate. If several apply, say which you trust most and why.
            (c) Status quo: what happens if nothing changes? The world changes slowly most of the time, so give the status quo extra weight, especially when little time is left.
            (d) Inside view: the specific, dated evidence that should move you away from the base rate, and roughly how far. Separate facts from speculation, and note what you could not find out.
            (e) The strongest case for the outcome you currently think less likely. Language models tend to over-predict "Yes"; check you are not doing the same.
            (f) Reconcile these into one probability. Avoid 0% and 100%, and go below 3% or above 97% only when the outcome is effectively already determined.
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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

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
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
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

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
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
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

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
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
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
        date_percentile_list: list[DatePercentile] = await structure_output(
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
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

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
    """One research report (two independent sources) per question. On Metaculus's
    credits the pacing tier sets the forecaster and the number of predictions; on a
    personal key RESEARCH_REPORTS and PREDICTIONS_PER_REPORT do (1 and 5 by default)."""
    if route == "openrouter" and tier is not None:
        reports, predictions = 1, tier.predictions
    else:
        reports = _env_int("RESEARCH_REPORTS", 1)
        predictions = _env_int("PREDICTIONS_PER_REPORT", 5)
    return RcForecastBot(
        research_reports_per_question=reports,
        predictions_per_research_report=predictions,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previous,
        extra_metadata_in_explanation=True,
        llms=build_llms(route, tier),
        research_health=research_health,
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
        for name, tally in health.sources.items():
            reasons = ", ".join(
                f"{reason} ×{count}" if count > 1 else reason
                for reason, count in tally.failures.most_common()
            )
            failed = f"{tally.searches - tally.worked}" + (f" ({reasons})" if reasons else "")
            lines.append(f"| {name} | {tally.searches} | {tally.worked} | {failed} |")
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
    # A mistyped AskNews setting stops the run here, not quietly at every question
    asknews_strategy()
    asknews_articles()
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

    # Pacing applies to Metaculus's credits; a personal key runs unpaced.
    meter: budget.SpendMeter | None = None
    plan: budget.Plan | None = None
    if route == "openrouter":
        meter = budget.SpendMeter(os.environ["OPENROUTER_API_KEY"])
        plan = budget.make_plan(meter.remaining(), budget.pinned_tier())
        logger.info(
            f"Credits left: {budget.dollars(meter.remaining())}. FutureEval runs"
            f" {plan.futureeval.name}, MiniBench {plan.minibench.name} ({plan.note})."
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
        # Collects the cost OpenRouter returns with every response in the pass, failed
        # questions included
        with MonetaryCostManager() as cost:
            results = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
        if meter is not None and tier is not None:
            failed = sum(isinstance(r, BaseException) for r in results)
            meter.record(label, tier, len(results) - failed, failed, cost.current_usage)
        return results

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
            if tier is not None and tier.predictions == 0:
                if new:
                    logger.warning(f"{label}: paused; {len(new)} new question(s) skipped")
                passes.append(describe_pass(label, tid, len(open_questions), 0, len(new), "paused"))
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
        if tier is not None and tier.predictions == 0:
            logger.warning("bot-testing-area: paused; nothing forecast")
            passes.append("- bot-testing-area: paused; nothing forecast.")
        else:
            testing = client.get_all_open_questions_from_tournament("bot-testing-area")
            questions = one_of_each_type(testing, _env_int("TEST_QUESTIONS", 3))
            passes.append(
                describe_pass("bot-testing-area", "bot-testing-area", len(testing), len(questions))
            )
            reports += run_pass("bot-testing-area", questions, tier, skip_previous=False)

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
    configured = [source_name(s) for s in research_sources(build_llms(route))]
    lines = run_report(passes, configured, health, reports)
    print("\n".join(lines) + "\n")
    budget.write_step_summary(lines)
    failures = sum(isinstance(r, BaseException) for r in reports)
    if failures:
        # a failed question turns the Actions run red; one that waits for research too,
        # so a source that keeps failing doesn't go unnoticed
        deferred = sum(deferred_for_research(r) for r in reports)
        why = f" ({deferred} for lack of research; the next run retries them)" if deferred else ""
        sys.exit(f"{failures} question(s) failed{why}; see the log above")


if __name__ == "__main__":
    main()
