"""
Whether research on Rob's key is reaching its reports, across runs.

A reply with text counted as research that worked, so from 10-31 to 11-03 most reports lost
their searches with every run green (README, Why). Each Sonnet 5.5 research reply is now
read for signs of that, and the last WINDOW outcomes are kept between runs: most runs carry
one question, so a rule within one run rarely sees enough. The same file remembers which
questions already waited once for web research that came back empty (main.py,
RcForecastBot._wait_for_web_research). It holds 0s and 1s and those questions' page URLs,
never a question's text, and losing it only shortens the window and allows a second wait.

No forecasting-tools import here, so the offline checks can exercise it directly.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass

import spend

logger = logging.getLogger(__name__)

# Stop reasons (as litellm reports them) for a research reply that ended before its
# answer: a server-side search loop paused, the reply ran out of tokens, a safety
# classifier stopped it, or it called a tool the bot doesn't run (tool_calls). Every tool
# the research reply is given runs on Anthropic's side and ends as "stop" (each research
# line on the run log through 11-08), so tool_calls means the reply asked the bot to run
# something: on 11-05 and 11-06 two replies did so and came back empty
CUT_SHORT = ("pause_turn", "length", "refusal", "tool_calls")

# Reports kept, and how many of them degraded open an issue
WINDOW = 12
TRIP = 6
# Questions remembered as having waited once for web research. A FutureEval question is
# open about 3 hours and MiniBench rounds bring a few dozen, so 200 covers several days
WAITED_KEPT = 200
# Fresh input tokens (the prompt less cache reads) under which billed searches can't have
# reached the reply. In 10-31 to 11-03's blackouts the most any reply wrote was 14,443
FRESH_FLOOR = 15000


@dataclass
class Watch:
    reports: int  # research reports this run, raised calls included
    degraded: list[str]  # why, one reason per degraded report this run
    bad: int  # degraded reports in the window
    window: int  # reports in the window
    tripped: bool
    line: str  # for the run page
    warning: str  # a GitHub warning annotation, "" with nothing degraded


def replies(calls: list[spend.Call]) -> list[spend.Call]:
    """The research replies on Rob's key: Anthropic's API, with its web tools."""
    return [c for c in calls if c.part.startswith("research: ") and c.model.startswith("anthropic/")]


def reason(call: spend.Call) -> str | None:
    """Why a research reply counts as degraded, or None."""
    if call.stop in CUT_SHORT or not call.answered:
        return "stopped early"
    if call.search_pages == 0 or len(call.search_errors) > call.search_pages:
        return "searches failed"
    if call.searches >= 3 and call.prompt_tokens - call.cache_reads < FRESH_FLOOR:
        return "results never reached the reply"
    if call.tally and (call.tally[0] == 0 or call.tally[1] > call.tally[0]) and call.search_pages < call.search_tries:
        return "most lookups failed"
    return None


def _load(path: str) -> dict:
    """The file's contents; a missing or unreadable file reads as empty."""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            kept = json.load(file)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Research watch: starting empty, unreadable file ({type(e).__name__})")
        return {}
    if not isinstance(kept, dict):
        logger.warning("Research watch: starting empty, unexpected values in the file")
        return {}
    return kept


def _read(path: str) -> list[int]:
    """The kept window; a missing or corrupt file starts empty."""
    kept = _load(path)
    if not kept:
        return []
    window = kept.get("window")
    if not isinstance(window, list) or any(type(v) is not int or v not in (0, 1) for v in window):
        logger.warning("Research watch: starting empty, unexpected values in the file")
        return []
    return window


def waited(path: str) -> list[str]:
    """The questions (page URLs) that already waited once for web research, oldest first.
    Without a file the bot remembers none, and RcForecastBot doesn't wait at all."""
    found = _load(path).get("waited")
    if not isinstance(found, list) or any(not isinstance(v, str) for v in found):
        return []
    return found


def mark_waited(path: str, question: str) -> None:
    """Remembers that a question waited once for web research. Saved at once, not at the
    end of the run, so a run that then fails still remembers it."""
    if not path:
        return
    kept = waited(path)
    if question not in kept:
        kept = (kept + [question])[-WAITED_KEPT:]
    _write(path, _read(path), kept)


def _write(path: str, window: list[int], waited_for: list[str] | None = None) -> None:
    """Saves the window, and the questions that waited (kept as they are when not given).
    The empty marker beside it tells the workflow there's something to save, so an idle
    run saves nothing."""
    if waited_for is None:
        waited_for = waited(path)
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"window": window, **({"waited": waited_for} if waited_for else {})}, file)
        with open(path + ".updated", "w", encoding="utf-8"):
            pass
    except OSError as e:
        logger.warning(f"Research watch: could not save the window ({type(e).__name__})")


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _line(calls: list[spend.Call], raised: int, reports: int, window: list[int]) -> str:
    errors = Counter(code for c in calls for code in c.search_errors)
    codes = ", ".join(f"{code} ×{n}" for code, n in errors.most_common())
    pages = f"{sum(c.search_pages for c in calls)} of {sum(c.search_tries for c in calls)} searches"
    flagged = [f"{spend.label(c.question)}: {reason(c)}" for c in calls if reason(c)]
    window_name = _count(len(window), "report", "reports")
    if len(window) < WINDOW:
        window_name += " (window not full)"
    return (
        f"- Web research on your key: {_count(reports, 'report', 'reports')};"
        f" {pages} brought back pages" + (f" ({codes})" if codes else "") + ";"
        f" {_count(sum(c.fetch_tries for c in calls), 'fetch', 'fetches')} tried;"
        f" {_count(sum(c.code_runs for c in calls), 'code run', 'code runs')};"
        f" {len(flagged)} degraded" + (f" ({', '.join(flagged)})" if flagged else "")
        + (f"; {raised} raised" if raised else "")
        + f". Last {window_name}: {sum(window)} degraded."
    )  # fmt: skip


def update(calls: list[spend.Call], raised: int, path: str = "") -> Watch | None:
    """Adds this run's reports to the window, a raised call counting as one degraded
    report. None for a run with no research on Rob's key, which leaves the file alone."""
    if not calls and not raised:
        return None
    window = (_read(path) + [1 if reason(c) else 0 for c in calls] + [1] * raised)[-WINDOW:]
    if path:
        _write(path, window)
    degraded = [why for why in map(reason, calls) if why] + ["call raised"] * raised
    reports = len(calls) + raised
    warning = ""
    if degraded:
        reasons = ", ".join(sorted(set(degraded)))
        warning = (
            f"::warning title=Research degraded::{len(degraded)} of {reports} research"
            f" reports this run were degraded ({reasons})"
        )
    return Watch(
        reports=reports,
        degraded=degraded,
        bad=sum(window),
        window=len(window),
        tripped=sum(window) >= TRIP,
        line=_line(calls, raised, reports, window),
        warning=warning,
    )
