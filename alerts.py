"""
Alerts that reach Rob as GitHub issues.

The workflow runs every 20 minutes, so a run that fails to get attention would email Rob
every 20 minutes for as long as the cause lasts. An alert instead opens one issue the first
time its event happens, mentioning the repository's owner. Later runs find an issue with
the same title, open or closed (for Rob's own key, the extra forecasters and the
Cloudflare clock, open), and open nothing. Each title
names what makes the event new: a new credit limit after a Metaculus top-up, a new
month for Rob's own key or the extra forecasters, a new week for research that keeps
coming back degraded, or the question whose forecast posted without its comment.

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


def _week() -> str:
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    return f"{year}-W{week:02d}"


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
        " shows in the next run; with `PAY_WITH: own-key` set, as the tournament workflow"
        " has it, the bot stays on your key and one issue says the top-up arrived.",
    )


def credits_topped_up(left: float, limit: float | None) -> bool:
    """With the tournament runs pinned to Rob's key (PAY_WITH=own-key), a top-up no longer
    moves the bot by itself, so this tells him one arrived. Once per credit limit."""
    return raise_alert(
        f"Metaculus's credits were topped up{_limit(limit)}",
        f"Metaculus's OpenRouter credits now stand at {budget.dollars(left)}. The bot is still"
        " forecasting on your Anthropic key, because the tournament workflow sets"
        " `PAY_WITH: own-key`. The credits route researches and forecasts differently"
        " (README, \"Why it's built this way\"), so going back to it is your call: try it"
        " first with a Test Bot run with `pay_with` set to `credits`, then change that line"
        " by pull request and add a setup to `setups.py`. Credits you leave unused cost"
        " nothing, but your key keeps paying. This issue opens once per credit limit; close"
        " it whenever.",
    )


# Rob's key's alerts count only while their issue is open: after he raises the limit or
# replaces the key and closes the issue, a second stop the same month alerts again
REOPEN_NOTE = (
    " Leave this issue open while the bot is paused, and close it once it's forecasting"
    " again: if it stops again this month, a new issue opens."
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


def _run_link() -> str:
    """A sentence linking this run's page on GitHub, or "" outside GitHub Actions."""
    repo, run = os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_RUN_ID")
    return f" The run: https://github.com/{repo}/actions/runs/{run}" if repo and run else ""


def extras_silent(questions: list[str], failures: list[str]) -> bool:
    """Questions forecast on the bot's own model alone: every forecast asked of the extra
    forecasters (EXTRA_FORECASTERS, on Rob's own OpenRouter key) failed for them. `failures`
    names each model with its error types, never a key. Like Rob's own key's alerts it
    counts only while open: the cause is his to fix, and a repeat after that is news."""
    return raise_alert(
        f"The extra forecasters returned no forecasts ({_month()})",
        f"{len(questions)} question(s) in one run were forecast on the Claude forecasts alone:"
        " every forecast asked of the extra forecasters (`EXTRA_FORECASTERS`, on your own"
        " OpenRouter key) failed.\n\n"
        + "\n".join(f"- {line}" for line in failures) + "\n\n"
        f"Questions: {', '.join(questions)}.{_run_link()}\n\n"
        "The forecasts still posted, and the run stays green on purpose. While this lasts,"
        " questions go out on the Claude forecasts only, and the scorecard's paired rows for"
        " the other models (rule D in JUDGING.md) stop growing. The usual causes: OpenRouter"
        " refuses the key or it is out of credit (`OPENROUTER_OWN_API_KEY`), or a model id is"
        " no longer served. The run's log and the \"Extra forecaster\" table on its page show"
        " each failure. Leave this issue open until the extra forecasters answer again, then"
        " close it: if they stop again, a new issue opens.",
        open_only=True,
    )


def comment_missing(url: str, post: int, error: str) -> bool:
    """A forecast that posted while its comment did not (main.CommentWatchClient). No later
    run adds the comment, so the issue names the question for Rob to add it by hand. Once
    per question."""
    return raise_alert(
        f"A forecast posted without its comment (question {post})",
        f"The bot's forecast on {url} posted, but its comment did not ({error}, after the"
        " package's own retries). No later run adds it: the question now counts as forecast,"
        " so runs skip it. The run that posted it is red and lists the question as failed,"
        f" though the forecast stands.{_run_link()}\n\n"
        "Metaculus requires comments for prizes (how it treats one missing comment isn't"
        " known), and the weekly review reads each forecaster's numbers from the comment. To"
        " add it by hand: the comment's full text is in that run's log, after the line"
        f" \"Comment not posted on post {post}\". Post it on the question as a private"
        " comment from the bot's account. This issue opens once per question; close it"
        " when done.",
    )


def missed_questions(links: list[str], days: int, week: str | None = None) -> bool:
    """Tournament questions that closed without a forecast (missed_check.py). Titled by the
    ISO week they closed in (`week`; this week without one), so the daily run and the
    Monday review announce a miss once between them."""
    shown = links[:MISSED_SHOWN]
    more = f"\n- and {len(links) - len(shown)} more on the review's page" if len(links) > len(shown) else ""
    return raise_alert(
        f"Questions missed ({week or _week()})",
        f"{len(links)} tournament question(s) closed in the last {days} days without a forecast from"
        " the bot. A missed question scores nothing.\n\n"
        + "\n".join(f"- {link}" for link in shown) + more + "\n\n"
        "The weekly review's run page says how to trace each one. This issue opens once"
        " for each week's closes; close it whenever.",
    )


# Questions listed in the missed-questions issue; the review's page lists them all
MISSED_SHOWN = 20


def research_degraded(bad: int, window: int, reasons: str) -> bool:
    return raise_alert(
        f"Research is degraded ({_week()})",
        f"{bad} of the last {window} research reports on your Anthropic key came back degraded"
        f" ({reasons}). Forecasts still went out on AskNews and whatever search found, and the"
        " run stays green on purpose. Read the reports and the tools part of each Spend line"
        " in the run's log. RESEARCH_TOOLS changes the tool setting without a code change"
        " (README, the variables table). This issue opens at most once a week; close it"
        " whenever.",
    )
