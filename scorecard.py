"""
A calibration scorecard from bot-review's review.json, for the weekly review's run page.

A leaderboard rank after a few resolved questions is mostly noise: at 2 questions the 80%
range for a bot's true rank among 253 is about 14th to 200th. What the bot's own resolved
forecasts can show earlier is bias:
- Mean spot peer score per question, with its standard error, by tournament and type.
- Binary: the mean forecast against the share that resolved Yes, and by bin. This shows
  any lean either way; the binary prompt checks its number in both directions.
- Numeric and discrete: how often the outcome fell below each forecaster's 10th
  percentile, between the 10th and 90th, or above the 90th; nominally 10/80/10. Fewer
  than 80% inside means the ranges run too narrow.
- Multiple choice: the mean probability given to the option that won, and how often it
  was the forecast's favourite.
- By setup: forecasts from before the 2026-10-30 prompt and research rules against those
  after, told apart by the "Model:" line that arrived with them. The tables above mix
  both; judge the rules by these rows.
- By kind of question, from the title: prices and markets, status on a date, sports and
  so on, to show where the points are won and lost, per setup. Descriptive only: at a
  dozen questions a kind, a gap under about 12 points is noise.
- The aggregate against its own members, and each model against the most common one on
  the questions both forecast, in log score (100 x ln of the probability given to what
  happened, the peer score's units). Paired on the same questions, these have far less
  noise than comparing peer scores between two setups. The model comes from the
  "Model:" line main.py adds to each forecaster's line in the posted summary; older
  comments have none and count only in the aggregate's row. One model's ids on the two
  routes (OpenRouter's and Anthropic's) count as one.
Read-only, standard library only: `python scorecard.py review.json`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001))
# forecasting-tools prints values with str(round(v, 6)), so small ones come as "1e-05"
_PERCENTILE = re.compile(r"([\d.]+)% chance of value below (-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)")
# How Metaculus resolves an open-bound question whose outcome fell beyond its range
_BEYOND = {"above_upper_bound": math.inf, "below_lower_bound": -math.inf}
# The forecaster's model, on the last line of its summary entry (main.py, MODEL_LINE)
_MODEL = re.compile(r"^Model: (\S+)\s*$", re.MULTILINE)
_BINARY_PROBABILITY = re.compile(r"([\d.]+)%")
_OPTION = re.compile(r"^- (.+): ([\d.]+)%\s*$", re.MULTILINE)
# Kinds of question, by title; the first that matches wins. Status applies to yes/no
# questions only, and counts of official reports to numeric ones
SPORTS = re.compile(
    r"\b(nfl|nba|wnba|mlb|nhl|ncaa|ufc|nascar|bundesliga|premier league|la liga|serie a"
    r"|ligue 1|champions league|alcs|nlcs|world series|super bowl|stanley cup|world cup"
    r"|olympics?|wimbledon|(australian|french|us) open|ballon d'or|grand prix|formula 1"
    r"|matchday|night football|touchdowns?|passing yards|fixture|series score)\b",
    re.IGNORECASE,
)
# Market words only where they can't mean something else: "close" alone also closes
# straits and polls, "gold" wins medals, "settlement" ends wars
MARKETS = re.compile(
    r"\b(price|prices|clos\w* (price|level|high|low|above|below|at)|settle\w* (price|at"
    r"|above|below)|futures|stock|shares|etfs?|bitcoin|btc|ethereum|crypto\w*|zcash|zec"
    r"|s&p|nasdaq|dow jones|vix|nikkei|ftse|dax|bist|brent|wti|(gold|silver|oil|wheat)"
    r" (price|futures)|(bond|treasury|\d+-year) yield|exchange rate|market cap\w*)\b",
    re.IGNORECASE,
)
STATUS = re.compile(
    r"\bas of\b|\bstill\b|\bin effect\b|\bremain\b|\bhave (been )?(filed|announced|published"
    r"|released|declared|resumed|reported|confirmed|launched|signed)\b"
    r"|\bbe (held|connected|suspended|in place)\b",
    re.IGNORECASE,
)
REPORTED = re.compile(r"\breport\w*\b", re.IGNORECASE)
LOG_FLOOR = 0.001  # a probability below this scores as this, so one bad parse can't dominate


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


def binary_by_setup(questions: list[dict]) -> list[str]:
    """The binary calibration summary line, once per setup."""
    lines = []
    for name in ("before 2026-10-30", "since 2026-10-30"):
        group = [q for q in questions if setup(q) == name]
        line = binary_calibration(group)[0]
        lines.append(f"- {name}: {line}")
    return lines


def _rate_se(outcomes: tuple[float, ...]) -> float:
    rate = statistics.mean(outcomes)
    return math.sqrt(rate * (1 - rate) / len(outcomes))


def _scored_trace(q: dict) -> dict | None:
    """The run that was scored: the latest before the spot scoring time, as bot-review
    picks it (its traces come newest first), else the latest."""
    traces = sorted(
        (t for t in q.get("traces") or [] if t.get("forecasters")),
        key=lambda t: t.get("run_time") or "",
    )
    scored_at = q.get("spot_scoring_time")
    before = [t for t in traces if scored_at and (t.get("run_time") or "") <= scored_at]
    chosen = (before or traces)[-1:]
    return chosen[0] if chosen else None


def _forecasters(q: dict) -> list[str]:
    """The forecasters of the run that was scored."""
    trace = _scored_trace(q)
    return [f.get("prediction") or "" for f in trace["forecasters"]] if trace else []


def _time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _same_run(q: dict) -> bool:
    """Whether the posted forecast (bot-review's latest) came from the scored run. The bot
    posts its forecast seconds before its comment, so a later run's forecast is later than
    the scored run's comment; the Cup and the Test Bot forecast questions again."""
    trace = _scored_trace(q)
    forecast_at = _time(q.get("forecast_time"))
    run_at = _time(trace.get("run_time")) if trace else None
    return bool(forecast_at and run_at and forecast_at <= run_at)


def setup(q: dict) -> str:
    """Which setup forecast it: the "Model:" line arrived with the 2026-10-30 rules."""
    tagged = any(_MODEL.search(text) for text in _forecasters(q))
    return "since 2026-10-30" if tagged else "before 2026-10-30"


def _canonical(model: str) -> str:
    """One name for one model on either route: openrouter/anthropic/claude-opus-5.5 and
    anthropic/claude-opus-5-5 are both anthropic/claude-opus-5.5."""
    model = model.removeprefix("openrouter/")
    return re.sub(r"^(anthropic/claude-[a-z]+-\d+)-(\d+)$", r"\1.\2", model)


def numeric_coverage(questions: list[dict]) -> list[str]:
    counts = {"below": 0, "inside": 0, "above": 0}
    counted: set[int] = set()
    for i, q in enumerate(questions):
        if q.get("question_type") not in ("numeric", "discrete"):
            continue
        resolution = q.get("resolution")
        try:
            outcome = _BEYOND.get(resolution) or float(resolution)
        except (TypeError, ValueError):
            continue  # annulled or ambiguous
        for text in _forecasters(q):
            try:
                points = {round(float(p) / 100, 2): float(v) for p, v in _PERCENTILE.findall(text)}
            except ValueError:
                continue  # a forecast written some other way: skipped, not fatal
            if 0.1 not in points or 0.9 not in points:
                continue
            counted.add(i)
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
        f"{total} forecaster ranges on {len(counted)} questions: outcome below the 10th percentile"
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


def kind(q: dict) -> str:
    """The kind of question, from its title."""
    title = q.get("title") or ""
    if re.search(r"google trends", title, re.IGNORECASE):
        return "Google Trends"
    if re.search(r"community prediction", title, re.IGNORECASE):
        return "Community prediction"
    if SPORTS.search(title):
        return "Sports"
    if MARKETS.search(title):
        return "Prices and markets"
    if q.get("question_type") == "binary" and STATUS.search(title):
        return "Status on a date"
    if q.get("question_type") in ("numeric", "discrete") and REPORTED.search(title):
        return "Reported counts"
    return "Other"


def setups(questions: list[dict]) -> list[str]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for q in _scored(questions):
        score = q["scores"]["spot_peer_score"]
        groups[(setup(q), "all types")].append(score)
        groups[(setup(q), q.get("question_type") or "?")].append(score)
    if not groups:
        return ["No scored questions yet."]
    lines = ["| Setup | Type | Scored | Mean spot peer ± SE |", "|---|---|---|---|"]
    for (name, kind_), scores in sorted(groups.items()):
        lines.append(f"| {name} | {kind_} | {len(scores)} | {_mean_se(scores)} |")
    return lines


def kinds(questions: list[dict]) -> list[str]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for q in _scored(questions):
        groups[(kind(q), setup(q))].append(q["scores"]["spot_peer_score"])
    if not groups:
        return ["No scored questions yet."]
    lines = ["| Kind of question | Setup | Scored | Mean spot peer ± SE |", "|---|---|---|---|"]
    for (name, when), scores in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        lines.append(f"| {name} | {when} | {len(scores)} | {_mean_se(scores)} |")
    return lines


def _members(q: dict) -> list[tuple[str, str]]:
    """(model, prediction text) for each forecaster of the scored run; "untagged" for
    comments from before main.py named the model."""
    members = []
    for text in _forecasters(q):
        found = _MODEL.search(text)
        model = _canonical(found.group(1)) if found else "untagged"
        members.append((model, _MODEL.sub("", text).strip()))
    return members


def _points(probability: float) -> float:
    return 100 * math.log(max(probability, LOG_FLOOR))


def _member_points(q: dict, text: str) -> float | None:
    """100 x ln of the probability this forecaster gave the outcome; None if unreadable."""
    resolution = q.get("resolution")
    if q.get("question_type") == "binary" and resolution in ("yes", "no"):
        found = _BINARY_PROBABILITY.search(text)
        if not found:
            return None
        p = float(found.group(1)) / 100
        return _points(p if resolution == "yes" else 1 - p)
    if q.get("question_type") == "multiple_choice":
        options = {name.strip(): float(value) / 100 for name, value in _OPTION.findall(text)}
        if resolution in options:
            return _points(options[resolution])
    return None


def _aggregate_points(q: dict) -> float | None:
    forecast = q.get("forecast") or {}
    resolution = q.get("resolution")
    if q.get("question_type") == "binary" and resolution in ("yes", "no"):
        p = forecast.get("probability_yes")
        return None if p is None else _points(p if resolution == "yes" else 1 - p)
    probabilities = forecast.get("probabilities") or {}
    if q.get("question_type") == "multiple_choice" and resolution in probabilities:
        return _points(probabilities[resolution])
    return None


def members(questions: list[dict]) -> list[str]:
    """Yes/no and multiple choice: the posted aggregate against the average of its own
    members (only where the posted forecast came from the scored run), and each model
    against the most common one on shared questions."""
    gains: list[float] = []
    by_model: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for i, q in enumerate(questions):
        scored = [(model, _member_points(q, text)) for model, text in _members(q)]
        scored = [(model, points) for model, points in scored if points is not None]
        if not scored:
            continue
        aggregate = _aggregate_points(q) if _same_run(q) else None
        if aggregate is not None and len(scored) >= 2:
            gains.append(aggregate - statistics.mean(points for _, points in scored))
        for model, points in scored:
            by_model[model].setdefault(i, []).append(points)
    if not gains and not by_model:
        return ["No resolved yes/no or multiple-choice forecasts with their members yet."]
    lines = [
        "| Comparison (log score, 100 x ln) | Questions | Mean difference ± SE |",
        "|---|---|---|",
    ]
    if gains:
        lines.append(f"| Aggregate − average member | {len(gains)} | {_mean_se(gains)} |")
    tagged = {m: qs for m, qs in by_model.items() if m != "untagged"}
    if len(tagged) >= 2:
        reference = max(tagged, key=lambda m: len(tagged[m]))
        compared = 0
        for model, per_question in sorted(tagged.items()):
            if model == reference:
                continue
            shared = sorted(set(per_question) & set(tagged[reference]))
            diffs = [
                statistics.mean(per_question[i]) - statistics.mean(tagged[reference][i])
                for i in shared
            ]
            if diffs:
                compared += 1
                lines.append(f"| {model} − {reference} | {len(diffs)} | {_mean_se(diffs)} |")
        if not compared:
            lines.append(f"| Models so far ({', '.join(sorted(tagged))}): no questions in common | – | – |")
    elif tagged:
        lines.append(f"| One model so far ({', '.join(sorted(tagged))}) | – | – |")
    return lines


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
        "",
        "**By setup** (the 2026-10-30 prompt and research rules against the setup before them)",
        "",
        *setups(questions),
        "",
        *binary_by_setup(questions),
        "",
        "**By kind of question** (descriptive; at a dozen questions a kind, gaps under about 12 points are noise)",
        "",
        *kinds(questions),
        "",
        "**Members and models** (paired on the same questions; positive means the first did better)",
        "",
        *members(questions),
    ]


if __name__ == "__main__":
    # Never fails the review: a scorecard problem must not hide bot-review's own summary
    try:
        with open(sys.argv[1] if len(sys.argv) > 1 else "review.json", encoding="utf-8") as f:
            print("\n".join(scorecard(json.load(f))))
    except Exception as e:
        print(f"Calibration scorecard unavailable ({type(e).__name__}: {e}).")
