"""
Alerts that reach Rob as GitHub issues.

The workflow runs every 20 minutes, so a run that fails to get attention would email Rob
every 20 minutes for as long as the cause lasts. An alert instead opens one issue the first
time its event happens, mentioning the repository's owner. Later runs find an issue with
the same title, open or closed, and open nothing. Each title names what makes the event
new: a new credit limit after a Metaculus top-up, or a new month for Rob's own key.

No forecasting-tools import here, so the offline checks can exercise it directly.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

import budget

logger = logging.getLogger(__name__)

API = "https://api.github.com"
# Issues opened with the workflow's GITHUB_TOKEN are authored by this account
BOT_LOGIN = "github-actions[bot]"


def raise_alert(title: str, body: str, timeout: float = 15) -> bool:
    """Opens an issue titled `title` unless one exists; True if it opened one. Without
    GITHUB_TOKEN and GITHUB_REPOSITORY (any run outside the tournament workflow) the
    alert is only logged."""
    token, repo = os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPOSITORY")
    if not (token and repo):
        logger.warning(f"Alert, not opened as an issue (no GitHub token here): {title}")
        return False
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        response = requests.get(
            f"{API}/repos/{repo}/issues",
            headers=headers,
            params={"state": "all", "creator": BOT_LOGIN, "per_page": 100},
            timeout=timeout,
        )
        response.raise_for_status()
        if any(issue.get("title") == title for issue in response.json()):
            return False
        owner = os.getenv("GITHUB_REPOSITORY_OWNER")
        response = requests.post(
            f"{API}/repos/{repo}/issues",
            headers=headers,
            json={"title": title, "body": f"@{owner} {body}" if owner else body},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as e:  # never log the message: keep request details out of logs
        logger.warning(f"Could not open the alert issue {title!r}: {type(e).__name__}")
        return False
    logger.warning(f"Alert opened as an issue: {title}")
    return True


def _month() -> str:
    return f"{datetime.now(timezone.utc):%Y-%m}"


def _limit(limit: float | None) -> str:
    return f" (of the ${limit:.0f} limit)" if limit is not None else ""


def credits_low(left: float, limit: float | None, own_key: bool) -> bool:
    then = (
        "the bot switches to your Anthropic key (`ANTHROPIC_API_KEY`), capped by its"
        " workspace's monthly spend limit"
        if own_key
        else "the bot pauses: no `ANTHROPIC_API_KEY` is set to take over (README, Setup)"
    )
    return raise_alert(
        f"Metaculus's credits are below ${budget.LOW_CREDIT:.0f}{_limit(limit)}",
        f"Metaculus's OpenRouter credits are down to {budget.dollars(left)}. When they fall"
        f" below the ${budget.FLOOR:.0f} floor, {then}. A top-up from Metaculus shows in the"
        " next run. This issue opens once per credit limit; close it whenever.",
    )


def credits_out(left: float, limit: float | None, own_key: bool) -> bool:
    now = (
        "From this run on the bot forecasts on your Anthropic key, capped by its"
        " workspace's monthly spend limit."
        if own_key
        else "The bot is paused. To keep forecasting, add `ANTHROPIC_API_KEY` (README, Setup)."
    )
    return raise_alert(
        f"Metaculus's credits ran out{_limit(limit)}",
        f"Metaculus's OpenRouter credits are under the ${budget.FLOOR:.0f} floor"
        f" ({budget.dollars(left)} left). {now} A top-up from Metaculus moves the bot back"
        " onto the credits at the next run.",
    )


def own_key_limit() -> bool:
    return raise_alert(
        f"Your Anthropic key reached its spend limit ({_month()})",
        "Anthropic is refusing the bot's requests because the key's workspace reached its"
        " monthly spend limit. The bot pauses quietly until the limit resets on the 1st of"
        " next month, or until you raise it in the Claude Console (Settings, Workspaces).",
    )


def own_key_rejected(error: str) -> bool:
    return raise_alert(
        f"Anthropic rejected the bot's key ({_month()})",
        f"Anthropic rejected `ANTHROPIC_API_KEY` ({error}), so the bot is paused. Replace"
        " the repository secret with a working key from the Claude Console.",
    )
