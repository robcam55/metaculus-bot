"""
Notices new Claude models, and Claude models the bot runs that Anthropic stops offering.

A new model generation can make the bot better, but switching to one blind can cost
questions, and a failed question scores nothing. The last two releases each rejected a
setting the bot sent: Haiku 5.5 the parser's temperature, Sonnet 5.5 a way of turning
thinking off. A new model also needs litellm to know its price, or the run page's spend
goes blank. So nothing here changes a model. Once a day (.github/workflows/model_watch.yaml)
this lists the models Rob's key can use, from Anthropic's Models API (free), and opens one
issue for each released after RELEASED_AFTER. The issue names the role the model could take
and how to evaluate it on the Test Bot before switching (README, "When a new Claude model
comes out"). A closed issue stays closed. A model the bot runs that drops off the list
opens an issue too, because every call to it would fail.

No forecasting-tools import here, so the offline checks can exercise it directly.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timezone

import requests

import alerts
import budget

logger = logging.getLogger(__name__)

API = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
# Every model released by the 2026-10-28 review was weighed then. That review moved
# research to Sonnet 5.5 and parsing to Haiku 5.5, kept Opus 5.5 for forecasts, and left
# out Fable 5.1, at 2.5 times Opus 5.5's price
RELEASED_AFTER = date(2026, 10, 28)
# What each model the bot runs does, and the Test Bot input that tries a replacement
ROLES = {
    budget.ANTHROPIC_IDS[budget.OPUS]: ("forecasts", "forecaster"),
    budget.ANTHROPIC_IDS[budget.SONNET]: (
        "research with web search, and forecasts on the cheaper tiers and the Metaculus Cup",
        "researcher",
    ),
    budget.ANTHROPIC_IDS[budget.HAIKU]: ("parsing forecasts and summaries", "parser"),
}
DOCS = "https://platform.claude.com/docs/en"


def list_models(key: str, timeout: float = 15) -> list[dict]:
    """Every model the key can use, across pages."""
    models: list[dict] = []
    params: dict[str, str | int] = {"limit": 1000}
    while True:
        response = requests.get(
            API,
            headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION},
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        page = response.json()
        models += page.get("data") or []
        if not (page.get("has_more") and page.get("last_id")):
            return models
        params["after_id"] = page["last_id"]


def _family(model_id: str) -> str:
    """The model's line: "opus" for claude-opus-5-5, say."""
    found = re.match(r"(?:anthropic/)?claude-([a-z]+)", model_id)
    return found.group(1) if found else model_id


def _released(model: dict) -> date | None:
    try:
        return datetime.fromisoformat(str(model["created_at"]).replace("Z", "+00:00")).date()
    except (KeyError, ValueError):
        return None


def new_model_issue(model: dict) -> tuple[str, str]:
    """The issue for a model released after RELEASED_AFTER: its title and body."""
    model_id = model["id"]
    name = model.get("display_name") or model_id
    released = _released(model)
    same_line = [
        (in_use, role, field)
        for in_use, (role, field) in ROLES.items()
        if _family(in_use) == _family(model_id)
    ]
    if same_line:
        in_use, role, field = same_line[0]
        where = (
            f"It's in the line of `{in_use.removeprefix('anthropic/')}`, which the bot runs"
            f" for {role}."
        )
        trial = f"`anthropic/{model_id}` as the {field} model"
    else:
        where = (
            "The bot runs no model in its line, so weigh it against Opus 5.5 for forecasts,"
            " at its price."
        )
        trial = f"`anthropic/{model_id}` as the forecaster model"
    body = "\n".join(
        [
            f"Anthropic released **{name}** (`{model_id}`) on {released or 'an unlisted date'}."
            f" {where} Nothing changes until you switch. To evaluate it:",
            "",
            f"1. **What changed.** Read Anthropic's notes on moving to it ({DOCS}/about-claude/"
            f"models/migration-guide) and its price ({DOCS}/about-claude/pricing). Look for"
            " settings it rejects (the last two releases each rejected one the bot sent),"
            " new defaults such as its effort, and a new tokenizer, which changes what a"
            " question costs.",
            "2. **Does the bot run on it.** Actions → Test Bot → Run workflow, with"
            f" \"Who pays\" `own-key` and {trial}: three bot-testing-area questions, about"
            " $2, never tournament questions. The run should go green, and the run page's"
            " spend should show a figure for it (litellm has to know its price).",
            "3. **Is it better.** Not by re-forecasting resolved questions: a newer model's"
            " training data runs later, so it may already know how they came out. Look for"
            " evidence gathered before the questions resolved: Metaculus's own bots on it in"
            " FutureEval, if it runs any; forecasting benchmarks scored on live questions,"
            " such as ForecastBench; and its price against any gain.",
            "4. **Switch.** Change its id in `budget.py` (`OPUS`, `SONNET` or `HAIKU`, and"
            " `ANTHROPIC_IDS`), run the offline checks and the Test Bot, and merge, ideally"
            " between MiniBench rounds so each round runs on one model.",
            "",
            "Close this issue once you've decided either way; it won't open again.",
        ]
    )
    return f"New Claude model to evaluate: {name} ({model_id})", body


def gone_issue(model_id: str, role: str) -> tuple[str, str]:
    """The issue for a model the bot runs that Anthropic no longer lists."""
    short = model_id.removeprefix("anthropic/")
    body = (
        f"Anthropic's model list for your key no longer includes `{short}`, which the bot"
        f" runs for {role}, so every call to it on your key now fails. Switch to the newest"
        f" model in its line in `budget.py` and check the switch on the Test Bot (steps in"
        f" the README, \"When a new Claude model comes out\"). Anthropic's deprecation list:"
        f" {DOCS}/about-claude/model-deprecations"
    )
    return f"Claude model no longer offered: {short}", body


def check() -> list[str]:
    """Checks the list and opens any issues; lines for the log either way. Never raises."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return ["Claude models: not checked (no ANTHROPIC_API_KEY here)."]
    try:
        models = list_models(key)
    except Exception as e:  # never log the message: keep request details out of logs
        return [f"Claude models: not checked ({type(e).__name__})."]
    if not models:
        return ["Claude models: not checked (the list came back empty)."]
    lines: list[str] = []
    listed = {m.get("id") for m in models}
    for in_use, (role, _) in ROLES.items():
        if in_use.removeprefix("anthropic/") not in listed:
            opened = alerts.raise_alert(*gone_issue(in_use, role))
            lines.append(f"Claude models: `{in_use}` is no longer listed{_told(opened)}.")
    for model in models:
        released = _released(model)
        if released is not None and released > RELEASED_AFTER and model.get("id"):
            opened = alerts.raise_alert(*new_model_issue(model))
            lines.append(f"Claude models: new, `{model['id']}` ({released}){_told(opened)}.")
    return lines or [f"Claude models: nothing new since {RELEASED_AFTER} ({len(models)} listed)."]


def _told(opened: bool) -> str:
    return "; opened an issue" if opened else "; its issue exists, or couldn't be opened"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    lines = check()
    print("\n".join(lines))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("".join(f"- {line}\n" for line in lines))
