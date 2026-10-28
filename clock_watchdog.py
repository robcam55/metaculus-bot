"""
Notices when the Cloudflare clock stops starting the tournament workflow.

The Worker in trigger/ starts the workflow every 20 minutes. When GitHub turns it away (an
expired or revoked token, say), the error shows only in Cloudflare's logs, and the bot
falls back to the backup clock (.github/workflows/backup_clock.yaml). GitHub fires that only
about every four hours (Oct 18-28: median gap 4.3 h), and a FutureEval question is open
for 1.5 to 3 hours, so on the backup alone a third to two thirds of them are missed while
every run stays green. The backup clock runs this on each tick: when the newest run the
Worker (or Rob, by hand) started is more than STALE_AFTER old, it opens one issue, which
counts while open.

No forecasting-tools import here, so the offline checks can exercise it directly.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone

import requests

import alerts

logger = logging.getLogger(__name__)

WORKFLOW = "run_bot_on_tournament.yaml"
# Longer than any gap the Worker has shown on its own (100 minutes on Oct 15, which fixed
# itself), shorter than a FutureEval question's window
STALE_AFTER = timedelta(hours=2)
TITLE = "The Cloudflare clock stopped"


def last_clock_run(repo: str, token: str, timeout: float = 15) -> datetime | None:
    """When the newest tournament run the Worker started was created; None if none is
    listed. The workflow's run-name ends in its source, "(cloudflare)", "(backup)" or
    "(manual)", so neither the backup clock's runs nor Rob's own hide a stopped Worker.
    Runs from before the run-name (which read only "Forecast on new AI tournament
    questions") count unless the backup clock's own token, github-actions[bot], started
    them."""
    response = requests.get(
        f"{alerts.API}/repos/{repo}/actions/workflows/{WORKFLOW}/runs",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        params={"event": "workflow_dispatch", "per_page": 50},
        timeout=timeout,
    )
    response.raise_for_status()
    for run in response.json().get("workflow_runs", []):
        source = re.search(r"\((\w+)\)\s*$", run.get("display_title") or "")
        if source:
            worker = source.group(1) == "cloudflare"
        else:
            worker = (run.get("triggering_actor") or {}).get("login") != alerts.BOT_LOGIN
        if worker:
            return datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
    return None


def check(now: datetime | None = None) -> str:
    """Checks the clock and alerts if it stopped; a line for the log either way. Never
    raises: the backup clock's own tick must not depend on it."""
    token, repo = os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPOSITORY")
    if not (token and repo):
        return "Cloudflare clock: not checked (no GitHub token here)."
    now = now or datetime.now(timezone.utc)
    try:
        last = last_clock_run(repo, token)
    except Exception as e:  # never log the message: keep request details out of logs
        return f"Cloudflare clock: not checked ({type(e).__name__})."
    if last is not None and now - last <= STALE_AFTER:
        return f"Cloudflare clock: ticking (last run {last:%Y-%m-%d %H:%M} UTC)."
    since = "no run it started is listed" if last is None else f"its last run was {last:%Y-%m-%d %H:%M} UTC"
    opened = alerts.raise_alert(
        TITLE,
        f"The Cloudflare Worker hasn't started the tournament workflow for over"
        f" {STALE_AFTER.total_seconds() / 3600:.0f} hours ({since}), so the bot is running"
        " only on the backup clock, about every four hours, and missing FutureEval"
        " questions. Its logs are in Cloudflare's dashboard (Workers, metaculus-bot-trigger)."
        " Usually its GitHub token expired: make a new one (README, Setup step 7) and run"
        " `npx wrangler secret put GITHUB_TOKEN --config trigger/wrangler.jsonc`. Leave this"
        " issue open until the runs are back every 20 minutes, then close it; if the clock"
        " stops again, a new issue opens.",
        open_only=True,
    )
    told = "opened an issue" if opened else "its issue is already open, or couldn't be opened"
    return f"Cloudflare clock: stopped ({since}); {told}."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    line = check()
    print(line)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"- {line}\n")
