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
- By setup: each forecast under the setup live when its run posted it, from the dated
  list in setups.py (s1 for comments without the "Model:" line that arrived with s2).
  The tables above mix them all; JUDGING.md says which rows judge which change: s4 and
  later against a frozen s1 baseline, weighted by tournament (rule A); a calibration line
  for s2 and later (rule B); and number-question width counted once per question, for s2
  and later and for s4 and later (rule C). An outcome beyond a bound is left out of the
  width counts: the review holds no bounds to set a forecaster's range against.
- By kind of question, from the title: prices and markets, status on a date, sports and
  so on, to show where the points are won and lost, before and under the 2026-10-30
  rules. Descriptive only: at a dozen questions a kind, a gap under about 12 points is
  noise.
- The aggregate against its own members, and each model against the most common one on
  the questions both forecast, in log score (100 x ln of the probability given to what
  happened, the peer score's units). Paired on the same questions, these have far less
  noise than comparing peer scores between two setups. The model comes from the
  "Model:" line main.py adds to each forecaster's line in the posted summary; older
  comments have none and count only in the aggregate's row. One model's ids on the two
  routes (OpenRouter's and Anthropic's) count as one. Where other models forecast too,
  one row sets the posted aggregate against what the most common model's forecasts alone
  would have given (rule D), which is also the Claude-only reading of those questions.
Read-only, standard library only: `python scorecard.py review.json`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import setups as dated

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
    """The binary calibration summary line, once per setup, then for everything under the
    2026-10-30 rules (JUDGING.md rule B)."""
    lines = []
    for name in [s.name for s in dated.SETUPS]:
        group = [q for q in questions if setup(q) == name]
        lines.append(f"- {name}: {binary_calibration(group)[0]}")
    since = set(dated.from_(dated.RULES_FROM))
    group = [q for q in questions if setup(q) in since]
    lines.append(f"- {dated.RULES_FROM} on: {binary_calibration(group)[0]}")
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
    """Which setup forecast it (setups.py): the one live when the scored run posted it,
    and s1 for a comment without the "Model:" line that arrived with s2."""
    tagged = any(_MODEL.search(text) for text in _forecasters(q))
    trace = _scored_trace(q)
    at = _time(trace.get("run_time")) if trace else None
    if at is not None and at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return dated.setup_at(at, tagged).name


def era(q: dict) -> str:
    """Before or under the 2026-10-30 rules, for the kinds table: per setup it would split
    a dozen questions a kind into rows of two or three."""
    return f"{dated.RULES_FROM} on" if setup(q) in dated.from_(dated.RULES_FROM) else "s1"


def _canonical(model: str) -> str:
    """One name for one model on either route: openrouter/anthropic/claude-opus-5.5 and
    anthropic/claude-opus-5-5 are both anthropic/claude-opus-5.5."""
    model = model.removeprefix("openrouter/")
    return re.sub(r"^(anthropic/claude-[a-z]+-\d+)-(\d+)$", r"\1.\2", model)


def _ranges(questions: list[dict]) -> list[tuple[float, list[tuple[float, float]]]]:
    """For each resolved numeric or discrete question with readable forecasts: the outcome
    (infinite when it resolved beyond a bound) and each forecaster's 10th and 90th
    percentile."""
    out = []
    for q in questions:
        if q.get("question_type") not in ("numeric", "discrete"):
            continue
        resolution = q.get("resolution")
        try:
            outcome = _BEYOND.get(resolution) or float(resolution)
        except (TypeError, ValueError):
            continue  # annulled or ambiguous
        ranges = []
        for text in _forecasters(q):
            try:
                points = {round(float(p) / 100, 2): float(v) for p, v in _PERCENTILE.findall(text)}
            except ValueError:
                continue  # a forecast written some other way: skipped, not fatal
            if 0.1 in points and 0.9 in points:
                ranges.append((points[0.1], points[0.9]))
        if ranges:
            out.append((outcome, ranges))
    return out


def _beyond_note(beyond: int) -> str:
    """An outcome beyond a bound can't be placed against a range: the review holds no
    bounds, so a forecast with most of its probability out there would count as a miss
    (question 46181 scored +99.7 and counted as three). Left out and said so."""
    if not beyond:
        return ""
    return f" {beyond} more question(s) resolved beyond a bound and are left out: the review holds no bounds to set a range against."


def numeric_coverage(questions: list[dict]) -> list[str]:
    counts = {"below": 0, "inside": 0, "above": 0}
    found = _ranges(questions)
    inside_bounds = [(outcome, ranges) for outcome, ranges in found if math.isfinite(outcome)]
    for outcome, ranges in inside_bounds:
        for low, high in ranges:
            if outcome < low:
                counts["below"] += 1
            elif outcome > high:
                counts["above"] += 1
            else:
                counts["inside"] += 1
    total = sum(counts.values())
    beyond = _beyond_note(len(found) - len(inside_bounds))
    if not total:
        return ["No resolved numeric forecasts with their percentiles yet." + beyond]
    return [
        f"{total} forecaster ranges on {len(inside_bounds)} questions: outcome below the 10th percentile"
        f" {counts['below'] / total:.0%}, between the 10th and 90th"
        f" {counts['inside'] / total:.0%}, above the 90th {counts['above'] / total:.0%}"
        " (nominally 10%, 80%, 10%)." + beyond
    ]


def _width_line(questions: int) -> int:
    """The most questions inside at which JUDGING.md rule C recommends: so few that a true
    80% rate would give that or fewer less than 1 time in 20 (12 of 20, 34 of 50)."""
    cumulative, line = 0.0, -1
    for inside in range(questions + 1):
        cumulative += math.comb(questions, inside) * 0.8**inside * 0.2 ** (questions - inside)
        if cumulative > 0.05:
            break
        line = inside
    return line


def coverage_by_setup(questions: list[dict]) -> list[str]:
    """JUDGING.md rule C's rows: one outcome per question, inside when it lies between the
    mean of the forecasters' 10th percentiles and the mean of their 90th. From s2 on (every
    forecast under the 2026-10-30 rules) and from s4 on, which the rule judges."""
    lines = []
    for start in dict.fromkeys((dated.RULES_FROM, dated.COMPARED_FROM)):
        names = set(dated.from_(start))
        found = _ranges([q for q in questions if setup(q) in names])
        counted = [(outcome, ranges) for outcome, ranges in found if math.isfinite(outcome)]
        inside = sum(
            statistics.mean(low for low, _ in ranges) <= outcome <= statistics.mean(high for _, high in ranges)
            for outcome, ranges in counted
        )
        ranges_inside = sum(low <= outcome <= high for outcome, ranges in counted for low, high in ranges)
        ranges_total = sum(len(ranges) for _, ranges in counted)
        if not counted:
            lines.append(f"- {start} on: no resolved numeric forecasts inside their bounds yet." + _beyond_note(len(found)))
            continue
        judged = ""
        if start == dated.COMPARED_FROM:
            judged = (
                f" Rule C needs 20 questions; at {len(counted)} it would recommend at"
                f" {_width_line(len(counted))} or fewer inside."
                if len(counted) >= 20
                else " Rule C needs 20 questions: wait."
            )
        lines.append(
            f"- {start} on: outcome inside the 10th to 90th range on {inside} of {len(counted)}"
            f" questions ({inside / len(counted):.0%}; {ranges_inside} of {ranges_total} forecaster"
            f" ranges).{judged}" + _beyond_note(len(found) - len(counted))
        )
    return lines


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
    """Rows per setup, then for the setups that JUDGING.md rule A compares with s1."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    compared = set(dated.from_(dated.COMPARED_FROM))
    for q in _scored(questions):
        score = q["scores"]["spot_peer_score"]
        names = [setup(q)] + ([f"{dated.COMPARED_FROM} on"] if setup(q) in compared else [])
        for name in names:
            groups[(name, "all types")].append(score)
            groups[(name, q.get("question_type") or "?")].append(score)
    if not groups:
        return ["No scored questions yet."]
    lines = ["| Setup | Type | Scored | Mean spot peer ± SE |", "|---|---|---|---|"]
    for (name, kind_), scores in sorted(groups.items()):
        lines.append(f"| {name} | {kind_} | {len(scores)} | {_mean_se(scores)} |")
    return lines


# JUDGING.md rule A's baseline, frozen on 2026-11-23: every s1 question on the four weekly
# reviews saved by then (11-02, 11-09, 11-16, 11-23), by tournament, as (mean spot peer
# score, its standard error, questions). The weekly page's own s1 row covers only the
# review's window, so it drifts as questions leave it
RULE_A_BASELINE = {"FutureEval": (16.00, 4.75, 41), "MiniBench": (6.62, 2.09, 118)}
RULE_A_MINIMUM = 40


def _tournament(q: dict) -> str | None:
    name = _project(q).lower()
    return next((t for t in RULE_A_BASELINE if t.lower() in name), None)


def rule_a(questions: list[dict]) -> list[str]:
    """JUDGING.md rule A: s4 and later against the frozen s1 baseline, weighted to the
    tournaments the s4+ questions came from. FutureEval's s1 questions scored about 9
    points above MiniBench's, so an unweighted comparison would move with the mix."""
    start = f"{dated.COMPARED_FROM} on"
    compared = set(dated.from_(dated.COMPARED_FROM))
    groups: dict[str, list[float]] = defaultdict(list)
    for q in _scored(questions):
        tournament = _tournament(q)
        if tournament and setup(q) in compared:
            groups[tournament].append(q["scores"]["spot_peer_score"])
    scores = [score for group in groups.values() for score in group]
    frozen = ", ".join(f"{t} {m:+.1f} ± {se:.1f} over {n}" for t, (m, se, n) in RULE_A_BASELINE.items())
    lines = [f"- s1 baseline, frozen 2026-11-23: {frozen}."]
    if not scores:
        return lines + [f"- {start}: no scored questions yet."]
    lines.append(
        f"- {start}: " + "; ".join(f"{t} {len(g)} at {_mean_se(g)}" for t, g in sorted(groups.items())) + "."
    )
    if len(scores) < 2:
        return lines + [f"- Rule A needs {RULE_A_MINIMUM} questions: wait."]
    weights = {t: len(g) / len(scores) for t, g in groups.items()}
    baseline = sum(RULE_A_BASELINE[t][0] * w for t, w in weights.items())
    error = math.sqrt(
        statistics.variance(scores) / len(scores)
        + sum((RULE_A_BASELINE[t][1] * w) ** 2 for t, w in weights.items())
    )
    verdict = (
        f"Rule A needs {RULE_A_MINIMUM} questions: wait."
        if len(scores) < RULE_A_MINIMUM
        else f"Rule A calls it clearly worse below {-2 * error:+.1f}; otherwise no large harm seen."
    )
    lines.append(
        f"- Against s1 weighted to that mix ({baseline:+.1f}): {statistics.mean(scores) - baseline:+.1f}"
        f" ± {error:.1f} over {len(scores)} questions. {verdict}"
    )
    return lines


def setup_legend() -> list[str]:
    return [f"- {dated.describe(s)}" for s in dated.SETUPS]


def kinds(questions: list[dict]) -> list[str]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for q in _scored(questions):
        groups[(kind(q), era(q))].append(q["scores"]["spot_peer_score"])
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


def _member_probability(q: dict, text: str) -> float | None:
    """The probability this forecaster gave the outcome; None if unreadable."""
    resolution = q.get("resolution")
    if q.get("question_type") == "binary" and resolution in ("yes", "no"):
        found = _BINARY_PROBABILITY.search(text)
        if not found:
            return None
        p = float(found.group(1)) / 100
        return p if resolution == "yes" else 1 - p
    if q.get("question_type") == "multiple_choice":
        options = {name.strip(): float(value) / 100 for name, value in _OPTION.findall(text)}
        if resolution in options:
            return options[resolution]
    return None


def _member_points(q: dict, text: str) -> float | None:
    """100 x ln of the probability this forecaster gave the outcome; None if unreadable."""
    probability = _member_probability(q, text)
    return None if probability is None else _points(probability)


def _own_aggregate_points(q: dict, probabilities: list[float]) -> float | None:
    """What the bot would have scored from one model's forecasts alone, combined as main.py
    combines them: yes/no by the median (the trimmed mean from four forecasts on), and
    multiple choice by the mean. On the probability given to what happened, the median and
    the trimmed mean are the same as on the forecast itself."""
    if not probabilities:
        return None
    if q.get("question_type") == "binary":
        values = sorted(probabilities)
        combined = statistics.mean(values[1:-1]) if len(values) >= 4 else statistics.median(values)
    else:
        combined = statistics.mean(probabilities)
    return _points(combined)


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
    """Yes/no and multiple choice. The posted aggregate against the average of its own
    members (only where the posted forecast came from the scored run), apart for questions
    with one model's forecasts and with several; the posted aggregate against what the most
    common model's forecasts alone would have given (JUDGING.md rule D); and each model
    against the most common one on shared questions. Rows that cover several setups come
    again for s4 on, because s3's forecasts were made while research was failing."""
    found: list[tuple[int, dict, list[tuple[str, float]], float | None]] = []
    by_model: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for i, q in enumerate(questions):
        scored = [(model, _member_probability(q, text)) for model, text in _members(q)]
        scored = [(model, p) for model, p in scored if p is not None]
        if not scored:
            continue
        found.append((i, q, scored, _aggregate_points(q) if _same_run(q) else None))
        for model, p in scored:
            by_model[model].setdefault(i, []).append(_points(p))
    if not found:
        return ["No resolved yes/no or multiple-choice forecasts with their members yet."]
    tagged = {m: qs for m, qs in by_model.items() if m != "untagged"}
    # The bot's own model: on the most questions and, where another model is on as many,
    # with the most forecasts (it makes three a question, an extra one)
    reference = (
        max(tagged, key=lambda m: (len(tagged[m]), sum(len(v) for v in tagged[m].values())))
        if tagged
        else None
    )
    later = set(dated.from_(dated.COMPARED_FROM))  # s3's research was failing: shown apart
    single: list[float] = []  # aggregate − average member, one model's forecasts
    several: list[float] = []  # ... with another model's forecast among them
    alone: list[tuple[int, float]] = []  # aggregate − the reference model's forecasts alone
    for i, q, scored, aggregate in found:
        mixed = any(model not in (reference, "untagged") for model, _ in scored)
        if aggregate is None:
            continue
        if len(scored) >= 2:
            gain = aggregate - statistics.mean(_points(p) for _, p in scored)
            (several if mixed else single).append(gain)
        own = _own_aggregate_points(q, [p for model, p in scored if model == reference])
        if mixed and own is not None:
            alone.append((i, aggregate - own))
    lines = [
        "| Comparison (log score, 100 x ln) | Questions | Mean difference ± SE |",
        "|---|---|---|",
    ]
    if single and several:
        lines.append(f"| Aggregate − average member, one model's forecasts | {len(single)} | {_mean_se(single)} |")
        lines.append(f"| Aggregate − average member, several models | {len(several)} | {_mean_se(several)} |")
    elif single or several:
        lines.append(f"| Aggregate − average member | {len(single + several)} | {_mean_se(single + several)} |")

    def is_later(i: int) -> bool:
        return setup(questions[i]) in later

    def row(label: str, diffs: list[tuple[int, float]]) -> list[str]:
        """The row over every question, then again for s4 on when that leaves some out."""
        out = [f"| {label} | {len(diffs)} | {_mean_se([d for _, d in diffs])} |"]
        recent = [d for i, d in diffs if is_later(i)]
        if recent and len(recent) < len(diffs):
            out.append(f"| {label}, {dated.COMPARED_FROM} on | {len(recent)} | {_mean_se(recent)} |")
        return out

    if alone:
        # JUDGING.md rule D's third comparison, and the Claude-only reading of these questions
        lines += row(f"Posted aggregate − {reference}-only aggregate", alone)
    if len(tagged) >= 2:
        compared = 0
        for model, per_question in sorted(tagged.items()):
            if model == reference:
                continue
            shared = sorted(set(per_question) & set(tagged[reference]))
            diffs = [
                (i, statistics.mean(per_question[i]) - statistics.mean(tagged[reference][i]))
                for i in shared
            ]
            if diffs:
                compared += 1
                lines += row(f"{model} − {reference}", diffs)
        if not compared:
            lines.append(f"| Models so far ({', '.join(sorted(tagged))}): no questions in common | – | – |")
    elif tagged:
        lines.append(f"| One model so far ({', '.join(sorted(tagged))}) | – | – |")
    scores = [
        (questions[i]["scores"]["spot_peer_score"], diff)
        for i, diff in alone
        if (questions[i].get("scores") or {}).get("spot_peer_score") is not None
    ]
    if scores:
        official = statistics.mean(score for score, _ in scores)
        own_only = official - statistics.mean(diff for _, diff in scores)
        lines += [
            "",
            f"On the {len(scores)} scored questions with other models' forecasts, the official spot"
            f" peer score averaged {official:+.1f}. From {reference}'s forecasts alone it would have"
            f" averaged about {own_only:+.1f}: between two forecasts of one question the peer"
            " score differs by this log-score difference. Yes/no and multiple choice only.",
        ]
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
        "**By setup** (dated in setups.py; JUDGING.md says which rows judge which change)",
        "",
        *setup_legend(),
        "",
        *setups(questions),
        "",
        "Rule A, the 2026-10-30 rules: s4 and later against s1, by tournament.",
        "",
        *rule_a(questions),
        "",
        "Rule B, yes/no calibration.",
        "",
        *binary_by_setup(questions),
        "",
        "Rule C, number-question width: one outcome per question.",
        "",
        *coverage_by_setup(questions),
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
