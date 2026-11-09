"""
The tournament questions the bot never forecast, for the weekly review's run page.

A missed question scores nothing, so it costs the tournament points a forecast would have
had a fair chance of earning. One bot maker reported losing about 150 points to 10 missed
questions. That is as much as a whole rank band in the Fall 2026 standings. Neither of the
bot's watches notices a single miss. The clock watchdog looks for gaps of hours between runs,
and the research watch looks at research that ran. A question can still be missed in several
ways:
- It closes while it waits for research (main.py, RESEARCH_RETRY_MARGIN).
- It sits behind a full run (MAX_QUESTIONS_PER_RUN).
- Its forecast fails.
- It falls in a gap between runs.

Each week this lists every FutureEval and MiniBench question that closed in the last
MISSED_DAYS days without a forecast from the bot's account. It opens one issue that week if
there is any. It only reads from Metaculus, and a failure here never fails the review:
`python missed_check.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import alerts

logger = logging.getLogger(__name__)

# The window, in days back from the review: it runs every Monday, so a week, plus a day so
# a late run doesn't leave a gap (a question in two windows shows twice; harmless)
DAYS = 8
# How far back a question may have opened and still close in the window: MiniBench
# questions stay open for days, FutureEval's for hours
OPENED_WITHIN = timedelta(days=30)
# Questions read at most per tournament; a list this long says it may be incomplete
CAP = 1000


@dataclass
class Tally:
    label: str  # "FutureEval", "MiniBench"
    closed: int  # questions that closed in the window
    missed: list  # those the bot never forecast (MetaculusQuestion)
    unknown: int  # those whose forecast status didn't come back
    capped: bool  # the read stopped at CAP


def missed(questions: list, now: datetime, days: int = DAYS) -> tuple[list, list, int]:
    """(closed in the window, never forecast, status unknown), from questions as
    forecasting-tools reads them with the bot's token. A question with no close time, or
    one closing after `now`, isn't counted: it can still be forecast."""
    since = now - timedelta(days=days)
    closed = [q for q in questions if (at := _utc(q.close_time)) is not None and since <= at <= now]
    never = [q for q in closed if q.already_forecasted is False]
    unknown = sum(q.already_forecasted is None for q in closed)
    return closed, never, unknown


def _utc(moment: datetime | None) -> datetime | None:
    if moment is not None and moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _read(client, tid: int | str, now: datetime) -> list:
    from forecasting_tools.helpers.metaculus_client import ApiFilter

    api_filter = ApiFilter(
        allowed_statuses=["closed", "resolved"],
        allowed_tournaments=[tid],
        group_question_mode="unpack_subquestions",
        open_time_gt=now - OPENED_WITHIN,
    )
    return asyncio.run(
        client.get_questions_matching_filter(
            api_filter, num_questions=CAP, error_if_question_target_missed=False
        )
    )


def _link(q) -> str:
    title = (q.question_text or "").strip().replace("|", "/")
    title = title if len(title) <= 90 else title[:87] + "..."
    when = f"{q.close_time:%a %m-%d %H:%M} UTC" if q.close_time else "no close time"
    return f"[{title}]({q.page_url}) (closed {when})"


def lines(tallies: list[Tally], days: int = DAYS) -> list[str]:
    out = ["## Missed questions", ""]
    total = sum(len(t.missed) for t in tallies)
    out.append(
        f"Tournament questions that closed in the last {days} days without a forecast from"
        f" the bot: {total}. It should read 0."
    )
    out.append("")
    for t in tallies:
        notes = []
        if t.unknown:
            notes.append(f"{t.unknown} whose forecast status didn't come back")
        if t.capped:
            notes.append(f"read stopped at {CAP} questions, so this may be incomplete")
        extra = f" ({'; '.join(notes)})" if notes else ""
        out.append(f"- {t.label}: {t.closed} closed, {len(t.missed)} missed{extra}.")
        out += [f"  - {_link(q)}" for q in t.missed]
    if total:
        out += [
            "",
            "For each, find its runs in the tournament workflow's log around its close time:"
            " was it skipped as already forecast, waiting for research, behind a full run, or"
            " did its forecast fail?",
        ]
    return out


def check(client, tournaments: list[tuple[str, int | str]], now: datetime) -> list[Tally]:
    tallies = []
    for label, tid in tournaments:
        found = _read(client, tid, now)
        closed, never, unknown = missed(found, now)
        tallies.append(Tally(label, len(closed), never, unknown, len(found) >= CAP))
    return tallies


def _setup():
    """forecasting-tools' client with the bot's token, and the tournaments the bot runs."""
    import main as bot
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

    client = MetaculusClient()
    return client, [
        ("FutureEval", bot._tournament_id(os.getenv("AIB_TOURNAMENT_ID"), bot.FALL_2026_FUTUREEVAL_ID)),
        ("MiniBench", bot._tournament_id(os.getenv("MINIBENCH_ID"), client.CURRENT_MINIBENCH_ID)),
    ]


def main() -> int:
    logging.basicConfig(level=logging.INFO)  # to stderr: stdout is the review's page
    try:
        with contextlib.redirect_stdout(sys.stderr):
            tallies = check(*_setup(), datetime.now(timezone.utc))
    except Exception as e:  # the review's own summary matters more than this section
        print(f"## Missed questions\n\nThe check couldn't run ({type(e).__name__}: {e}).")
        return 0
    print("\n".join(lines(tallies)))
    found = [q for t in tallies for q in t.missed]
    if found:
        with contextlib.redirect_stdout(sys.stderr):
            alerts.missed_questions([_link(q) for q in found])
    return 0


if __name__ == "__main__":
    sys.exit(main())
