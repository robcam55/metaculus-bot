"""
Pacing for Metaculus's incremental OpenRouter credits.

Metaculus funds the bot's OpenRouter key in steps: about $100 to start, raised
automatically after above-average MiniBench results, plus a possible bonus for
open-source bots. Each run reads what the key has left and picks a
spending tier that the balance can pay for over the next few weeks of questions. A small
balance then can't run dry in the middle of an evaluation window, and a raised balance
buys stronger forecasts from the next run on.

No forecasting-tools import here, so the offline checks can exercise it directly.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

KEY_URL = "https://openrouter.ai/api/v1/key"

OPUS = "openrouter/anthropic/claude-opus-5.5"
SONNET = "openrouter/anthropic/claude-sonnet-5"
# The same models on Rob's own Anthropic key, in litellm's ids for Anthropic's API
ANTHROPIC_IDS = {OPUS: "anthropic/claude-opus-5-5", SONNET: "anthropic/claude-sonnet-5"}


@dataclass(frozen=True)
class Tier:
    name: str
    forecaster: str  # litellm model id; empty when paused
    predictions: int
    cost: float  # estimated dollars a question, research and parsing included


# Measured on the first live test (2026-10-02, three bot-testing-area questions at opus-3,
# web-search research only): $0.13, $0.15 and $0.35 a question, $0.21 on average. Research
# with web search costs the most and varies most ($0.24 of the $0.35). These estimates
# carry a margin of about 40% over that, for AskNews articles lengthening every prompt
# and for harder tournament questions. Recalibrate from the spend each run reports.
TIERS = {
    "opus-5": Tier("opus-5", OPUS, 5, 0.40),
    "opus-3": Tier("opus-3", OPUS, 3, 0.30),
    "sonnet-3": Tier("sonnet-3", SONNET, 3, 0.23),
    # The last resort (see make_plan). By the figures above an Opus forecast costs about
    # $0.05, research and parsing $0.15 and a Sonnet forecast under $0.03, so dropping two
    # Sonnet forecasts saves only about a fifth: research is most of the cost.
    "sonnet-1": Tier("sonnet-1", SONNET, 1, 0.18),
    "pause": Tier("pause", "", 0, 0.0),
}

# (FutureEval, MiniBench) pairs, richest first. MiniBench results decide whether
# Metaculus adds credits, so MiniBench never runs below FutureEval.
LADDER = (
    ("opus-5", "opus-5"),
    ("opus-3", "opus-5"),
    ("opus-3", "opus-3"),
    ("sonnet-3", "opus-3"),
    ("sonnet-3", "sonnet-3"),
)

# Expected volume, from Metaculus's announcement: 300-500 FutureEval questions between
# Sep 28 and a few weeks before Jan 1 (slow for the first one or two weeks), and MiniBench
# rounds of about 60 questions every two weeks.
FUTUREEVAL_PER_WEEK = 35
MINIBENCH_PER_WEEK = 30
# Plan four weeks (two MiniBench rounds) ahead: long enough to last through a performance
# evaluation, short enough that the balance gets spent rather than hoarded.
HORIZON_WEEKS = 4
# Below this the bot stops, rather than run out of credit halfway through a question.
FLOOR = 2.0
# When the bottom of the ladder can't pay for this many weeks, both tournaments drop to
# one forecast a question, so the last of the credit covers more questions before the
# floor stops the bot. Rob's choice of 2026-10-06, in place of lowering the tiers sooner.
LAST_RESORT = "sonnet-1"
LAST_RESORT_WEEKS = 1
# Below this the bot opens a GitHub issue (one email) saying the credits are nearly gone.
# About four days of questions at the measured costs (2026-10-11).
LOW_CREDIT = 15.0
# Rob's own key, once the credits are under the floor (Rob's choice of 2026-10-11), runs
# this tier unless BUDGET_TIER pins another: Opus, as D-21 keeps the credits on Opus.
OWN_KEY_TIER = "opus-3"


@dataclass(frozen=True)
class Plan:
    futureeval: Tier
    minibench: Tier
    remaining: float | None  # dollars left on the key: inf if uncapped, None if unknown
    note: str


def key_status(api_key: str, timeout: float = 15) -> dict[str, Any] | None:
    """The key's limit and usage (free to call, not rate limited); None if unreadable."""
    try:
        response = requests.get(
            KEY_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )
        response.raise_for_status()
        return response.json()["data"]
    except Exception as e:  # never log the message: keep request details out of logs
        logger.warning(f"Could not read the OpenRouter key's balance: {type(e).__name__}")
        return None


def remaining_of(status: dict[str, Any] | None) -> float | None:
    """Dollars left on the key: inf when the key has no limit, None when unknown."""
    if status is None:
        return None
    left = status.get("limit_remaining")
    return math.inf if left is None else float(left)


def pinned_tier() -> Tier | None:
    """BUDGET_TIER (a repository variable) pins one tier for both tournaments."""
    name = os.getenv("BUDGET_TIER", "").strip().lower()
    if not name:
        return None
    if name not in TIERS:
        raise SystemExit(f"BUDGET_TIER must be one of {', '.join(TIERS)}; got {name!r}")
    return TIERS[name]


def on_route(model: str, route: str | None) -> str:
    """A tier's model as the route names it: the tiers use OpenRouter's ids."""
    return ANTHROPIC_IDS.get(model, model) if route == "anthropic" else model


def under_floor(remaining: float | None) -> bool:
    return remaining is not None and remaining < FLOOR


def exhausted(remaining: float | None, plan: Plan) -> bool:
    """Whether the credits are spent: under the floor, or unable to pay for one more
    question at a tier the plan runs. affordable() lets a run spend only while the
    balance above the floor covers a question's estimate, so the balance comes to rest
    between the floor and the floor plus one estimate ($2.00 to $2.30 at opus-3) and stays
    there. Checking the floor alone left the bot idle in that band, on neither key, until
    a top-up. A plan that pauses on purpose (BUDGET_TIER=pause) is not exhausted."""
    if under_floor(remaining):
        return True
    running = [t for t in (plan.futureeval, plan.minibench) if t.predictions > 0]
    return any(affordable(remaining, t) == 0 for t in running)


def own_key_plan(pinned: Tier | None, note: str) -> Plan:
    """Both tournaments' tier on Rob's own Anthropic key: the pinned tier, so
    BUDGET_TIER=pause stops this too, or OWN_KEY_TIER. A plain Anthropic key can't read
    what its workspace has spent, so nothing here paces it: the workspace's monthly spend
    limit, set in the Claude Console, is the cap."""
    tier = pinned or TIERS[OWN_KEY_TIER]
    return Plan(tier, tier, None, note)


def horizon_cost(futureeval: Tier, minibench: Tier, weeks: float = HORIZON_WEEKS) -> float:
    return weeks * (
        FUTUREEVAL_PER_WEEK * futureeval.cost + MINIBENCH_PER_WEEK * minibench.cost
    )


def make_plan(remaining: float | None, pinned: Tier | None = None) -> Plan:
    if remaining is not None and remaining < FLOOR:  # even when pinned
        pause = TIERS["pause"]
        return Plan(pause, pause, remaining, f"under the ${FLOOR:.0f} floor: paused")
    if pinned is not None:
        return Plan(pinned, pinned, remaining, f"BUDGET_TIER pins {pinned.name}")
    if remaining is None:
        fe, mb = (TIERS[n] for n in LADDER[-1])
        return Plan(fe, mb, None, "balance unreadable: cheapest tiers")
    for fe_name, mb_name in LADDER:
        fe, mb = TIERS[fe_name], TIERS[mb_name]
        need = horizon_cost(fe, mb)
        if need <= remaining - FLOOR:
            return Plan(
                fe, mb, remaining, f"{HORIZON_WEEKS} weeks at these tiers cost about ${need:.0f}"
            )
    # Not enough for the horizon even at the bottom: keep going there until the last week
    # of credit, then at the last resort until the floor
    fe, mb = (TIERS[n] for n in LADDER[-1])
    if horizon_cost(fe, mb, LAST_RESORT_WEEKS) > remaining - FLOOR:
        last = TIERS[LAST_RESORT]
        return Plan(
            last, last, remaining, f"less than a week of credit left: {last.name}, the last resort"
        )
    return Plan(fe, mb, remaining, f"less than {HORIZON_WEEKS} weeks of credit left")


def affordable(remaining: float | None, tier: Tier) -> int | None:
    """Questions the balance pays for at this tier above the floor; None means no cap."""
    if tier.predictions == 0:
        return 0
    if remaining is None or math.isinf(remaining):
        return None
    return max(0, math.floor((remaining - FLOOR) / tier.cost))


def dollars(amount: float | None) -> str:
    if amount is None:
        return "unknown"
    if math.isinf(amount):
        return "no limit"
    return f"${amount:.2f}"


class SpendMeter:
    """The key's balance at the start of a run, and what each pass spent.

    OpenRouter's key figures lag the spend by minutes (the first live test re-read the
    key right after a pass: $0.32 of $0.63 showed, and usage had not moved), so the
    balance is read once, before the run, when the previous run's spend has settled.
    Each pass's spend is the sum of OpenRouter's own figures for every call in it
    (spend.py). Until 2026-10-09 it was litellm's price list instead, which ran about 12%
    below the settled balance and has priced OpenRouter's responses at $0 since."""

    def __init__(self, api_key: str) -> None:
        self.start = key_status(api_key)
        self.spent = 0.0
        self.rows: list[str] = []

    def remaining(self) -> float | None:
        """The balance at the start, less what this run has spent since."""
        left = remaining_of(self.start)
        return None if left is None else left - self.spent

    def record(self, label: str, tier: Tier, forecasts: int, failed: int, spent: float) -> None:
        self.spent += spent
        each = dollars(spent / forecasts) if forecasts else "-"
        self.rows.append(
            f"| {label} | {tier.name} | {forecasts} | {failed} | {dollars(spent)} | {each} |"
        )

    def summary(self, plan: Plan) -> list[str]:
        """Markdown for the log and the Actions run page."""
        limit = (self.start or {}).get("limit")
        of = f" of {dollars(float(limit))}" if limit is not None else ""
        now = self.remaining()
        lines = [
            "### Metaculus's OpenRouter credits",
            "",
            f"- Left at the start of this run: {dollars(remaining_of(self.start))}{of}."
            f" Spent: {dollars(self.spent)}."
            f" Left now: {'unknown' if now is None else 'about ' + dollars(now)}.",
            f"- Tiers: FutureEval {plan.futureeval.name}, MiniBench {plan.minibench.name}"
            f" ({plan.note}).",
        ]
        return lines + self.table()

    def table(self) -> list[str]:
        if not self.rows:
            return []
        return [
            "",
            "| Pass | Tier | Questions | Failed | Spent | Per question |",
            "|---|---|---|---|---|---|",
            *self.rows,
        ]


class OwnKeyMeter(SpendMeter):
    """What each pass spent on Rob's own Anthropic key: each response's tokens at litellm's
    prices plus $0.01 a web search (spend.py). A plain key can't read its
    workspace's spend, so there is no balance here: the workspace's monthly spend limit,
    set in the Claude Console, is the cap, and Anthropic enforces it."""

    def __init__(self, credits: dict[str, Any] | None) -> None:
        self.start = None  # no balance: remaining() is unknown, so runs aren't capped by it
        self.credits = credits  # the OpenRouter key's status, for the run page
        self.spent = 0.0
        self.rows = []

    def summary(self, plan: Plan) -> list[str]:
        lines = [
            "### Your Anthropic key",
            "",
            f"- Metaculus's credits: {dollars(remaining_of(self.credits))} left."
            f" Spent on your key this run: {dollars(self.spent)}. Its cap is the monthly"
            " spend limit on its workspace in the Claude Console.",
            f"- Tiers: FutureEval {plan.futureeval.name}, MiniBench {plan.minibench.name}"
            f" ({plan.note}).",
        ]
        return lines + self.table()


def write_step_summary(lines: list[str]) -> None:
    """Append Markdown lines to the GitHub Actions run page, when running there."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
