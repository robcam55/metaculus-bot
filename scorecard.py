"""
A calibration scorecard from bot-review's review.json, for the weekly review's run page.

A leaderboard rank after a few resolved questions is mostly noise: at 2 questions the 80%
range for a bot's true rank among 253 is about 14th to 200th. What the bot's own resolved
forecasts can show earlier is bias:
- Mean spot peer score per question, with its standard error, by tournament and type.
- Binary: the mean forecast against the share that resolved Yes, and by bin. This tests
  the binary prompt's "language models over-predict Yes" step before anyone tunes it.
- Numeric and discrete: how often the outcome fell below each forecaster's 10th
  percentile, between the 10th and 90th, or above the 90th; nominally 10/80/10. Fewer
  than 80% inside means the ranges run too narrow.
- Multiple choice: the mean probability given to the option that won, and how often it
  was the forecast's favourite.
Read-only, standard library only: `python scorecard.py review.json`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from typing import Any

BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001))
_PERCENTILE = re.compile(r"([\d.]+)% chance of value below (-?[\d.eE+]+)")


def _mean_se(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:+.1f}"
    return f"{mean:+.1f} ± {statistics.stdev(values) / math.sqrt(len(values)):.1f}"


def _scored(questions: list[dict]) -> list[dict]:
    return [
        q for q in questions
        if (q.get("scores") or {}).get("spot_peer_score") is not None
    ]  # fmt: skip


def _project(q: dict) -> str:
    return q.get("project_name") or q.get("project_slug") or "?"


def peer_scores(questions: list[dict]) -> list[str]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for q in _scored(questions):
        score = q["scores"]["spot_peer_score"]
        groups[(_project(q), "all types")].append(score)
        groups[(_project(q), q.get("question_type") or "?")].append(score)
    lines = [
        "| Tournament | Type | Scored | Mean spot peer ± SE | Sum |",
        "|---|---|---|---|---|",
    ]
    for (project, kind), scores in sorted(groups.items()):
        lines.append(f"| {project} | {kind} | {len(scores)} | {_mean_se(scores)} | {sum(scores):+.1f} |")
    return lines


def binary_calibration(questions: list[dict]) -> list[str]:
    pairs = [
        (float(q["forecast"]["probability_yes"]), 1.0 if q["resolution"] == "yes" else 0.0)
        for q in questions
        if q.get("question_type") == "binary"
        and q.get("resolution") in ("yes", "no")
        and (q.get("forecast") or {}).get("probability_yes") is not None
    ]
    if not pairs:
        return ["No resolved binary forecasts yet."]
    forecasts, outcomes = zip(*pairs)
    brier = statistics.mean((p - y) ** 2 for p, y in pairs)
    lines = [
        f"{len(pairs)} resolved: mean forecast {statistics.mean(forecasts):.0%}, resolved Yes"
        f" {statistics.mean(outcomes):.0%} (± {_rate_se(outcomes):.0%}), Brier {brier:.3f}.",
        "",
        "| Forecast | Questions | Mean forecast | Resolved Yes |",
        "|---|---|---|---|",
    ]
    for low, high in BINS:
        inside = [(p, y) for p, y in pairs if low <= p < high]
        if inside:
            lines.append(
                f"| {low:.0%}–{min(high, 1):.0%} | {len(inside)}"
                f" | {statistics.mean(p for p, _ in inside):.0%}"
                f" | {statistics.mean(y for _, y in inside):.0%} |"
            )
    return lines


def _rate_se(outcomes: tuple[float, ...]) -> float:
    rate = statistics.mean(outcomes)
    return math.sqrt(rate * (1 - rate) / len(outcomes))


def _forecasters(q: dict) -> list[str]:
    traces = [t for t in q.get("traces") or [] if t.get("forecasters")]
    return [f.get("prediction") or "" for f in traces[-1]["forecasters"]] if traces else []


def numeric_coverage(questions: list[dict]) -> list[str]:
    counts = {"below": 0, "inside": 0, "above": 0}
    for q in questions:
        if q.get("question_type") not in ("numeric", "discrete"):
            continue
        try:
            outcome = float(q.get("resolution"))
        except (TypeError, ValueError):
            continue  # annulled, or resolved beyond a bound
        for text in _forecasters(q):
            points = {round(float(p) / 100, 2): float(v) for p, v in _PERCENTILE.findall(text)}
            if 0.1 not in points or 0.9 not in points:
                continue
            if outcome < points[0.1]:
                counts["below"] += 1
            elif outcome > points[0.9]:
                counts["above"] += 1
            else:
                counts["inside"] += 1
    total = sum(counts.values())
    if not total:
        return ["No resolved numeric forecasts with their percentiles yet."]
    return [
        f"{total} forecaster ranges: outcome below the 10th percentile"
        f" {counts['below'] / total:.0%}, between the 10th and 90th"
        f" {counts['inside'] / total:.0%}, above the 90th {counts['above'] / total:.0%}"
        " (nominally 10%, 80%, 10%)."
    ]


def multiple_choice(questions: list[dict]) -> list[str]:
    given, favourite = [], []
    for q in questions:
        probs = (q.get("forecast") or {}).get("probabilities") or {}
        if q.get("question_type") != "multiple_choice" or q.get("resolution") not in probs:
            continue
        given.append(probs[q["resolution"]])
        favourite.append(1.0 if max(probs, key=probs.get) == q["resolution"] else 0.0)
    if not given:
        return ["No resolved multiple-choice forecasts yet."]
    return [
        f"{len(given)} resolved: mean probability on the winner {statistics.mean(given):.0%};"
        f" the winner was the favourite {statistics.mean(favourite):.0%} of the time."
    ]


def scorecard(review: dict[str, Any]) -> list[str]:
    questions = review.get("questions") or []
    return [
        "## Calibration scorecard",
        "",
        "Per question and with its standard error, because a rank after a few dozen"
        " questions mostly measures luck.",
        "",
        *peer_scores(questions),
        "",
        "**Binary**",
        "",
        *binary_calibration(questions),
        "",
        "**Numeric and discrete**",
        "",
        *numeric_coverage(questions),
        "",
        "**Multiple choice**",
        "",
        *multiple_choice(questions),
    ]


if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1 else "review.json", encoding="utf-8") as f:
        print("\n".join(scorecard(json.load(f))))
