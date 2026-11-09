"""
The bot's setups, dated: which version of the bot made each forecast, for the weekly
scorecard's "By setup" rows, the run page, and the judging rules in JUDGING.md.

Add an entry with every change that alters forecasts (prompts, models, research,
aggregation), dated when it went live. For code that is the merge into main, since every
run checks main out afresh. For a repository variable it is the variable's update time,
which `gh variable list` shows. A change that doesn't touch forecasts, such as an alert or a
check, gets no entry. Entries are never edited after their rows have been read; a wrong
date gets a correction noted beside it.

A forecast's setup comes from its run's time. A forecast whose posted summary has no
"Model:" line (main.py, MODEL_LINE) is always s1, because that line arrived with s2.
Standard library only, so scorecard.py can import it alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Setup:
    name: str
    since: datetime | None  # when it went live (UTC); None for the first
    change: str


SETUPS = (
    Setup("s1", None, "the template-era prompts, Claude Opus only"),
    Setup(
        "s2",
        datetime(2026, 10, 31, 17, 53, 13, tzinfo=timezone.utc),
        "the 2026-10-30 prompt and research rules (#14); research lost most results until s4",
    ),
    Setup(
        "s3",
        datetime(2026, 11, 1, 13, 17, 37, tzinfo=timezone.utc),
        "GPT-6.1 Sol and Gemini 4 Argon join (EXTRA_FORECASTERS, #16)",
    ),
    Setup(
        "s4",
        datetime(2026, 11, 4, 17, 3, 41, tzinfo=timezone.utc),
        "research tools called directly (#17)",
    ),
    Setup(
        "s5",
        datetime(2026, 11, 9, 0, 10, 19, tzinfo=timezone.utc),
        "empty web research waits once for the next run (#20)",
    ),
)

# JUDGING.md rule A compares this setup and later ones against s1: s2 and s3 lost most of
# their research, which would blame the prompts for a research fault
COMPARED_FROM = "s4"
# And rules B and C, and the kinds table, read everything forecast under the 2026-10-30 rules
RULES_FROM = "s2"


def setup_at(when: datetime | None, tagged: bool = True) -> Setup:
    """The setup live at `when`. Untagged (no "Model:" line) is s1 whatever the time; a
    tagged forecast at an unknown time, or one stamped just before s2's merge, is s2."""
    if not tagged:
        return SETUPS[0]
    live = SETUPS[1]
    for setup in SETUPS[2:]:
        if when is not None and setup.since is not None and when >= setup.since:
            live = setup
    return live


def current(now: datetime | None = None) -> Setup:
    """The setup live now, for the run page."""
    return setup_at(now or datetime.now(timezone.utc))


def from_(name: str) -> list[str]:
    """The names of a setup and every later one."""
    names = [s.name for s in SETUPS]
    return names[names.index(name):]


def describe(setup: Setup) -> str:
    since = f"since {setup.since:%Y-%m-%d %H:%M} UTC" if setup.since else "before s2"
    return f"{setup.name} ({since}): {setup.change}"
