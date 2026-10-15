"""
What each run spends, call by call, by question and by part of the work.

Every model call the bot makes goes through forecasting-tools' GeneralLlm, which calls
litellm's `acompletion`. `install` wraps that one function, so each response is recorded
as it returns, in the task that made the call, tagged with its question and the part of
the work it was for: a research source, forecast 1, 2 or 3, parsing, or the research
summary. Retries and calls for questions that end up not posted count like any other.

The cost of a call is OpenRouter's own figure for it, from the usage it returns with every
response. Until 2026-10-09 the run page summed litellm's prices instead, through
forecasting-tools' cost managers, and that went wrong twice:
- The pinned litellm (1.80.10) prices OpenRouter responses from a price list it downloads
  at startup. On 2026-10-09 litellm removed every OpenRouter row from that list, and from
  the next run on every response counted $0.00.
- Before that, the list's token prices came to about 12% less than the balance fell,
  probably the web-search fees, which a token price list doesn't carry.
OpenRouter's billed figure, `usage.cost`, read $0.00 on all 99 calls the logs show it for
(2026-10-05 to 10-09, as litellm's "hidden param cost"), while the key's balance kept
falling: Metaculus's credits look like bring-your-own-key, where the
provider's bill is `cost_details.upstream_inference_cost` and OpenRouter's own fee is
`cost`. So a BYOK call costs both together. That reading is inferred from the logs; the run
page counts which figure each call's cost came from, so the first runs confirm it. A call
whose response gives no figure is looked up from OpenRouter's record of it after the pass,
and failing that estimated at OpenRouter's list prices, so it can't read as $0 again.

No forecasting-tools import here, so the offline checks can exercise it directly.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import requests

logger = logging.getLogger(__name__)

GENERATION_URL = "https://openrouter.ai/api/v1/generation"
MODELS_URL = "https://openrouter.ai/api/v1/models"  # public: no key

# Where a call's cost came from, best first
BASES = {
    "billed": "billed by OpenRouter",
    "byok": "provider's bill plus OpenRouter's fee (BYOK)",
    "upstream": "provider's bill only",
    "lookup": "looked up from OpenRouter afterwards",
    "list price": "estimated at OpenRouter's list prices",
    "litellm": "litellm's prices (on Anthropic's API: tokens plus $0.01 a search)",
}

# Anthropic's fee for a web search on its own API: $10 per 1,000
ANTHROPIC_SEARCH_FEE = 0.01

PARSING = "Parsing and summary"
OTHER = "Other"


@dataclass
class Call:
    question: str | None  # the question's key (its URL), None outside any question
    part: str  # "research: <source>", "forecast 2", "parsing", "summary" or "other"
    model: str
    cost: float | None  # dollars; None when no figure was found
    basis: str | None  # a key of BASES; None with no figure
    generation: str | None  # OpenRouter's generation id, for a later look-up
    prompt_tokens: int = 0
    completion_tokens: int = 0
    searches: int = 0  # web searches the provider ran for this call


class Ledger:
    """The calls of one pass (one tournament's questions in one run)."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.raised: Counter[str] = Counter()  # calls that raised, by error type

    def total(self) -> float:
        return sum(c.cost or 0.0 for c in self.calls)

    def without_figure(self) -> list[Call]:
        return [c for c in self.calls if c.cost is None]

    def look_up(
        self,
        api_key: str,
        tries: int = 3,
        wait: float = 5.0,
        limit: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Fills in calls whose response carried no cost from OpenRouter's record of the
        generation, which can take a few seconds to appear. Free, never logs the key, and
        gives up after `limit` seconds so a slow answer can't hold up the run. Only
        OpenRouter's responses carry a generation id, so a call on Anthropic's own API is
        never looked up; with no OpenRouter key ("") nothing is, and the list prices
        still stand in."""
        start = clock()
        missing = [c for c in self.without_figure() if c.generation] if api_key else []
        for attempt in range(tries):
            if not missing or clock() - start > limit:
                break
            if attempt:
                sleep(wait)
            still = []
            for call in missing:
                record = None if clock() - start > limit else generation_record(
                    api_key, call.generation or ""
                )
                if record is None:  # not recorded yet, unreadable, or out of time
                    still.append(call)
                    continue
                figure = openrouter_figure(
                    _number(_field(record, "total_cost")),
                    _number(_field(record, "upstream_inference_cost")),
                    _field(record, "is_byok"),
                )
                if figure is not None:  # a record without a figure: nothing more to learn
                    call.cost, call.basis = figure[0], "lookup"
            missing = still
        # Last, an estimate at OpenRouter's list prices, so a figure OpenRouter doesn't
        # give can't read as $0 again. Before 2026-10-09 litellm's copy of these prices
        # came to about 12% under the settled balance.
        rest = self.without_figure()
        prices = list_prices() if rest else {}
        for call in rest:
            estimate = list_price(call, prices)
            if estimate:
                call.cost, call.basis = estimate, "list price"
        if self.without_figure():
            logger.warning(
                f"Spend: {len(self.without_figure())} call(s) without a cost figure"
            )


_ledger: ContextVar[Ledger | None] = ContextVar("spend_ledger", default=None)
_question: ContextVar[str | None] = ContextVar("spend_question", default=None)
_part: ContextVar[str | None] = ContextVar("spend_part", default=None)


@contextmanager
def collecting() -> Iterator[Ledger]:
    """Records every call made inside the block (including inside asyncio.run, whose
    tasks inherit the context) into a fresh ledger."""
    ledger = Ledger()
    token = _ledger.set(ledger)
    try:
        yield ledger
    finally:
        _ledger.reset(token)


@contextmanager
def tagged(question: str | None = None, part: str | None = None) -> Iterator[None]:
    """Calls made inside the block count toward this question and part of the work."""
    tokens = []
    if question is not None:
        tokens.append((_question, _question.set(question)))
    if part is not None:
        tokens.append((_part, _part.set(part)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def openrouter_figure(
    cost: float | None, upstream: float | None, byok: Any
) -> tuple[float, str] | None:
    """A call's cost from OpenRouter's figures: what OpenRouter billed (`cost`) and, for
    a bring-your-own-key call, what the provider billed (`upstream_inference_cost`).
    None when they don't give one: a BYOK call without the provider's figure, or a call
    that reads $0 (no model call on these keys is free)."""
    if byok is True:
        return None if upstream is None else ((cost or 0.0) + upstream, "byok")
    if cost is not None and cost > 0:
        return cost, "billed"
    if upstream is not None and upstream > 0:
        return upstream, "upstream"
    return None


def _litellm_estimate(response: Any, model: str, searches: int) -> float | None:
    """A call that isn't OpenRouter's (Rob's own Anthropic key): its tokens at litellm's
    prices plus Anthropic's fee for each web search. litellm 1.80.10, which the runs
    install, leaves the searches out of its own figure for a response, and newer litellm
    counts them, so the tokens are priced here and the searches added: the same total on
    either version. Failing that, litellm's own figure for the response."""
    usage = getattr(response, "usage", None)
    prompt = int(_number(_field(usage, "prompt_tokens")) or 0)
    completion = int(_number(_field(usage, "completion_tokens")) or 0)
    # Only Anthropic's own models: another provider's search fees and extras (Perplexity's,
    # say) are litellm's own business, in its figure for the response below
    anthropic = model.startswith(("anthropic/", "claude-"))
    if anthropic and (prompt or completion):
        try:
            import litellm

            costs = litellm.cost_per_token(
                model=model, prompt_tokens=prompt, completion_tokens=completion
            )
            tokens = sum(_number(c) or 0.0 for c in costs)
        except Exception:  # a model litellm doesn't price
            tokens = 0.0
        if tokens > 0:
            return tokens + searches * ANTHROPIC_SEARCH_FEE
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = _number(_field(hidden, "response_cost"))
    if cost:
        return cost
    try:
        import litellm

        cost = _number(litellm.completion_cost(completion_response=response))
    except Exception:
        return None
    return cost or None


def call_from_response(
    model: str, response: Any, question: str | None = None, part: str = "other"
) -> Call:
    usage = getattr(response, "usage", None)
    figure = openrouter_figure(
        _number(_field(usage, "cost")),
        _number(_field(_field(usage, "cost_details"), "upstream_inference_cost")),
        _field(usage, "is_byok"),
    )
    generation = getattr(response, "id", None)
    if not (isinstance(generation, str) and generation.startswith("gen-")):
        generation = None
    searches = int(
        _number(_field(_field(usage, "server_tool_use"), "web_search_requests")) or 0
    )
    if figure is None and generation is None:  # not an OpenRouter response
        estimate = _litellm_estimate(response, model, searches)
        figure = None if estimate is None else (estimate, "litellm")
    return Call(
        question=question,
        part=part,
        model=model,
        cost=figure[0] if figure else None,
        basis=figure[1] if figure else None,
        generation=generation,
        prompt_tokens=int(_number(_field(usage, "prompt_tokens")) or 0),
        completion_tokens=int(_number(_field(usage, "completion_tokens")) or 0),
        searches=searches,
    )


def label(question: str | None) -> str:
    """A question's short name: its id from the URL, or the start of its text."""
    if not question:
        return "outside any question"
    if question.startswith("http"):
        return question.rstrip("/").rsplit("/", 1)[-1] or question
    return question[:40]


def _log(call: Call) -> None:
    cost = "no figure yet" if call.cost is None else f"${call.cost:.4f} ({call.basis})"
    logger.info(
        f"Spend: {label(call.question)} | {call.part} | {call.model} | {cost}"
        f" | {call.prompt_tokens:,} in, {call.completion_tokens:,} out"
        f" | {call.searches} web searches | {call.generation or '-'}"
    )


def record_response(model: str, response: Any) -> None:
    """Records one model response in the current pass's ledger. Never raises: measuring
    must not break forecasting."""
    ledger = _ledger.get()
    if ledger is None:
        return
    try:
        call = call_from_response(model, response, _question.get(), _part.get() or "other")
        ledger.calls.append(call)
        _log(call)
    except Exception as e:
        logger.warning(f"Spend: could not record a call ({type(e).__name__})")


def record_raised(model: str, error: BaseException) -> None:
    ledger = _ledger.get()
    if ledger is not None:
        ledger.raised[type(error).__name__] += 1
        logger.info(
            f"Spend: {label(_question.get())} | {_part.get() or 'other'} | {model}"
            f" | raised {type(error).__name__}; any cost is missing here"
        )


def metered(acompletion: Callable[..., Any]) -> Callable[..., Any]:
    """`acompletion`, recording every response, and every call that raised."""
    if getattr(acompletion, "_spend_metered", False):
        return acompletion

    @functools.wraps(acompletion)
    async def call(*args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "unknown model")
        try:
            response = await acompletion(*args, **kwargs)
        except BaseException as e:  # a timeout or cancellation may still have been billed
            record_raised(model, e)
            raise
        record_response(model, response)
        return response

    call._spend_metered = True  # type: ignore[attr-defined]
    return call


def install(module: Any) -> None:
    """Meters `module.acompletion`: forecasting-tools' general_llm imports litellm's by
    name, and every GeneralLlm call goes through it. Safe to call more than once."""
    module.acompletion = metered(module.acompletion)


def generation_record(
    api_key: str, generation_id: str, timeout: float = 10
) -> dict[str, Any] | None:
    """OpenRouter's record of one generation; None until it's there, or if unreadable."""
    try:
        response = requests.get(
            GENERATION_URL,
            params={"id": generation_id},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if response.status_code == 404:  # not recorded yet
            return None
        response.raise_for_status()
        data = response.json()["data"]
    except Exception as e:  # never log the message: keep request details out of logs
        logger.warning(f"Spend: could not look up a generation ({type(e).__name__})")
        return None
    return data if isinstance(data, dict) else None


def list_prices(timeout: float = 15) -> dict[str, dict[str, float]]:
    """OpenRouter's public price list: dollars per prompt token, per completion token
    (thinking included) and per web search, by model id. Empty if unreadable."""
    try:
        response = requests.get(MODELS_URL, timeout=timeout)
        response.raise_for_status()
        models = response.json()["data"]
    except Exception as e:
        logger.warning(f"Spend: could not read OpenRouter's price list ({type(e).__name__})")
        return {}
    prices = {}
    for model in models:
        pricing = _field(model, "pricing") or {}
        prices[str(_field(model, "id"))] = {
            name: _number(_field(pricing, name)) or 0.0
            for name in ("prompt", "completion", "web_search")
        }
    return prices


def openrouter_id(model: str) -> str:
    """OpenRouter's price-list id for a model as litellm names it. litellm's
    "openrouter/anthropic/claude-sonnet-5:online" is OpenRouter's "anthropic/claude-sonnet-5"
    with the web-search variant; Anthropic's own API (a personal key) names
    "anthropic/claude-opus-5-5" and "claude-haiku-4-5-20251001" what OpenRouter lists as
    "anthropic/claude-opus-5.5" and "anthropic/claude-haiku-4.5", at the same prices."""
    name = model.removeprefix("openrouter/").split(":", 1)[0]
    if name.startswith("claude-"):
        name = "anthropic/" + name
    if name.startswith("anthropic/claude-"):
        name = re.sub(r"-\d{8}$", "", name)  # a dated snapshot
        name = re.sub(r"-(\d)-(\d)$", r"-\1.\2", name)
    return name


def list_price(call: Call, prices: dict[str, dict[str, float]]) -> float | None:
    """A call's cost at list prices, from its tokens and web searches."""
    price = prices.get(openrouter_id(call.model))
    if not price or not (call.prompt_tokens or call.completion_tokens):
        return None
    return (
        call.prompt_tokens * price["prompt"]
        + call.completion_tokens * price["completion"]
        + call.searches * price["web_search"]
    )


def column(part: str) -> str:
    """The breakdown column a part of the work goes in."""
    if part.startswith("research: "):
        return "Research: " + part.removeprefix("research: ")
    if part.startswith("forecast "):
        return "Forecast " + part.removeprefix("forecast ")
    if part in ("parsing", "summary"):
        return PARSING
    return OTHER


def _order(name: str) -> tuple[int, int, str]:
    if name.startswith("Research: "):
        return (0, 0, "")  # research columns keep the order they were first seen in
    if name.startswith("Forecast "):
        number = name.removeprefix("Forecast ")
        return (1, int(number) if number.isdigit() else 0, number)
    return (2 if name == PARSING else 3, 0, name)


def _row(cells: list[str]) -> str:
    return "|" + "|".join(f" {cell} " if cell else " " for cell in cells) + "|"


def _cell(calls: list[Call]) -> str:
    if not calls:
        return "-"
    known = sum(c.cost or 0.0 for c in calls)
    if all(c.cost is None for c in calls):
        return "?"
    return f"${known:.3f}" + ("+?" if any(c.cost is None for c in calls) else "")


def breakdown(title: str, ledger: Ledger, statuses: dict[str, str] | None = None) -> list[str]:
    """Markdown for the log and the run page: one row per question, one column per
    research source, per forecast, and for parsing and the research summary."""
    if not ledger.calls and not ledger.raised:
        return []
    statuses = statuses or {}
    columns: list[str] = []
    for call in ledger.calls:
        name = column(call.part)
        if name not in columns:
            columns.append(name)
    columns.sort(key=_order)  # stable: research sources keep their first-seen order
    questions = list(dict.fromkeys(c.question for c in ledger.calls))
    lines = [
        "",
        f"### {title}: spend by question",
        "",
        _row(["Question", "Status", *columns, "Total"]),
        "|---|---|" + "---|" * (len(columns) + 1),
    ]

    def row(name: str, status: str, calls: list[Call]) -> str:
        cells = [_cell([c for c in calls if column(c.part) == col]) for col in columns]
        return _row([name, status, *cells, _cell(calls)])

    for question in questions:
        calls = [c for c in ledger.calls if c.question == question]
        name = label(question)
        if question and question.startswith("http"):
            name = f"[{name}]({question})"
        lines.append(row(name, statuses.get(question or "", ""), calls))
    if len(questions) > 1:
        lines.append(row(f"All {len(questions)}", "", ledger.calls))
    total = ledger.total()
    if total > 0:
        shares = [
            f"{sum(c.cost or 0.0 for c in ledger.calls if column(c.part) == col) / total:.0%}"
            for col in columns
        ]
        lines.append(_row(["Share", "", *shares, "100%"]))

    bases = Counter(c.basis for c in ledger.calls if c.basis)
    figures = ", ".join(f"{n} {BASES[b]}" for b, n in bases.most_common())
    missing = len(ledger.without_figure())
    notes = [
        f"- {len(ledger.calls)} model calls, {sum(c.searches for c in ledger.calls)} web"
        f" searches. Cost figures: {figures or 'none'}"
        + (f"; {missing} without any figure (not in the totals)." if missing else ".")
    ]
    if ledger.raised:
        raised = ", ".join(f"{name} ×{n}" for name, n in ledger.raised.most_common())
        notes.append(f"- Calls that raised: {raised}. Anything they cost is missing here.")
    return lines + ["", *notes, ""]
