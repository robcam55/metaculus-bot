"""
Alerts that reach Rob as GitHub issues.

The workflow runs every 20 minutes, so a run that fails to get attention would email Rob
every 20 minutes for as long as the cause lasts. An alert instead opens one issue the first
time its event happens, mentioning the repository's owner. Later runs find an issue with
the same title, open or closed (for Rob's own key, open), and open nothing. Each title
names what makes the event new: a new credit limit after a Metaculus top-up, or a new
month for Rob's own key.

No forecasting-tools import here, so the offline checks can exercise it directly.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import requests

import budget

logger = logging.getLogger(__name__)

API = "https://api.github.com"
# Issues opened with the workflow's GITHUB_TOKEN are authored by this account
BOT_LOGIN = "github-actions[bot]"
# Seconds before the issue list is read a second time. On 2026-10-22 one read in about 590
# came back without an existing issue (#9), and a duplicate (#10) opened
REREAD_WAIT = 3.0


def raise_alert(title: str, body: str, timeout: float = 15, open_only: bool = False) -> bool:
    """Opens an issue titled `title` unless one exists; True if it opened one. An issue
    counts whether open or closed, or with `open_only` only while open, so closing it
    lets the event alert again. The list is read twice before an issue opens. Without
    GITHUB_TOKEN and GITHUB_REPOSITORY (any run outside the tournament workflow) the
    alert is only logged."""
    token, repo = os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPOSITORY")
    if not (token and repo):
        logger.warning(f"Alert, not opened as an issue (no GitHub token here): {title}")
        return False
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        for read in range(2):
            if read:
                time.sleep(REREAD_WAIT)
            response = requests.get(
                f"{API}/repos/{repo}/issues",
                headers=headers,
                params={
                    "state": "open" if open_only else "all",
                    "creator": BOT_LOGIN,
                    "per_page": 100,
                },
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
        f"Metaculus's OpenRouter credits are down to {budget.dollars(left)}. When they can't"
        f" pay for another question above the ${budget.FLOOR:.0f} reserve, {then}. A top-up"
        " from Metaculus shows in the next run. This issue opens once per credit limit;"
        " close it whenever.",
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
        f"Metaculus's OpenRouter credits ({budget.dollars(left)} left) can't pay for another"
        f" question above the ${budget.FLOOR:.0f} reserve. {now} A top-up from Metaculus"
        " moves the bot back onto the credits at the next run.",
    )


# Rob's key's alerts count only while their issue is open: after he raises the limit or
# replaces the key and closes the issue, a second stop the same month alerts again
REOPEN_NOTE = (
    " Close this issue once that's done: if it happens again this month, a new issue opens."
)


def own_key_limit() -> bool:
    return raise_alert(
        f"Your Anthropic key reached its spend limit ({_month()})",
        "Anthropic is refusing the bot's requests on spending grounds. Usually the key's"
        " workspace reached its monthly spend limit: the bot pauses quietly until the limit"
        " resets on the 1st of next month, or until you raise it in the Claude Console"
        " (Settings, Workspaces). If your account is prepaid and out of credit instead, buy"
        " credit or turn on auto-reload (Settings, Billing); the bot resumes at the next run"
        " either way." + REOPEN_NOTE,
        open_only=True,
    )


def own_key_rejected(error: str) -> bool:
    return raise_alert(
        f"Anthropic rejected the bot's key ({_month()})",
        f"Anthropic rejected `ANTHROPIC_API_KEY` ({error}), so the bot is paused. Replace"
        " the repository secret with a working key from the Claude Console." + REOPEN_NOTE,
        open_only=True,
    )
