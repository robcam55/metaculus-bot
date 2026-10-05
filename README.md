# rc_trader's Metaculus forecasting bot

A Claude-based bot for Metaculus's AI forecasting tournament (FutureEval). It is built on
Metaculus's template bot (`Metaculus/metac-bot-template` at commit `6dab04c`); the
template's own README follows below. This bot is the external accuracy lab for rc_trader,
a personal forecasting and paper-trading project. The tournament scores hundreds of
questions a season against professional forecasters, with no money at risk.

## What's different from the template

The changes are in `main.py` (`RcForecastBot`) and `budget.py`:
- **Claude models, on Metaculus's credits first.**
  - Metaculus's donated OpenRouter credits (`OPENROUTER_API_KEY`) pay before a personal
    `ANTHROPIC_API_KEY`.
  - They never pay for the Metaculus Cup, because Metaculus gives them for FutureEval and
    MiniBench only.
  - On the credits, the pacing tier below picks Opus 5.5 or Sonnet 5. A personal key runs
    Sonnet 5.

  Haiku 4.5 parses in both cases. These models reject sampling parameters, so none are sent.
- **Paced spending (`budget.py`).** Metaculus funds the key in steps:
  - about $100 to start;
  - more automatically after above-average MiniBench results;
  - possibly a bonus for open-source bots.

  Each run reads the key's balance from OpenRouter. For each tournament it picks the
  richest tier the balance can pay for over four weeks of expected questions (about 35
  FutureEval and 30 MiniBench a week):

  | Tier | Forecasts per question | Estimated cost per question |
  |---|---|---|
  | `opus-5` | 5 × Opus 5.5 | $0.40 |
  | `opus-3` | 3 × Opus 5.5 | $0.30 |
  | `sonnet-3` | 3 × Sonnet 5 | $0.23 |

  The estimates are about 40% above the first live test (2026-10-02): `opus-3` cost $0.13
  to $0.35 a question, $0.21 on average, and web-search research was most of it.
  - **MiniBench first.** MiniBench results decide further funding, so MiniBench never runs
    below FutureEval. At $100, FutureEval runs `opus-3` and MiniBench `opus-5`; from about
    $106, both run `opus-5`.
  - **Never runs dry.** Below $2 the bot pauses, and one run never forecasts more
    questions than the balance covers, so credit can't run out halfway through a question.
  - **Top-ups apply at once.** When Metaculus raises the key, the next run sees it.
  - **Measured spend.** Each run's page on GitHub shows the balance and what each tier
    cost per question. The costs are the ones OpenRouter returns with every response;
    its balance figure lags by minutes, so the bot reads it once, at the start of a run.
    The response costs run slightly low: the first test's added up to $0.63 while the
    balance fell $0.73, probably because web searches bill separately. Pacing works from
    the balance, so it isn't affected. Recalibrate the estimates in `budget.py` from
    those numbers after the first week.
- **Two independent research sources per research report.**
  - AskNews news search: by default one latest-news call per question, to fit the free
    tier's 1,000 calls a month. The template's version uses six. `ASKNEWS_STRATEGY` and
    `ASKNEWS_ARTICLES` change this without a code change.
  - A search-backed model: Sonnet 5 with OpenRouter's `:online` (Anthropic's own web search,
    which Metaculus's credits cover), or Perplexity with your own key. Its prompt carries
    the question's background.

  Every prompt, the search-backed model's and the forecasters', carries today's date and
  the question's close and resolution times.

  If one source fails, the other still feeds the forecast. If every source fails, the
  question isn't forecast blind:
  - It waits for the next run, 20 minutes later, and turns this run red.
  - A question closing within 45 minutes gets no further try. It is forecast without
    research, and the forecasters are told so.
  - With `RESEARCH_REPORTS` above 1, a report whose sources all fail is dropped, and the
    question is forecast from the reports that found research.
- **A report on each run's page.** For each tournament: how many questions were open, how
  many not yet forecast, and how many the run took. Then the research sources that are set
  up, and how many searches each returned, with the error types of any that failed.
  - A failing source is otherwise only a warning in the log, and the run stays green.
  - An idle run still lists the sources, so new AskNews keys show up within 20 minutes.
- **Several predictions per question** from one research report: three or five on
  Metaculus's credits (by tier), five on a personal key. With four or more, binary questions
  use a trimmed mean (dropping the highest and lowest), which beat the median in Halawi et
  al. (2024). With three they use the package's median.
- **An outside-view-first binary prompt.** It asks for a reference class and base rate,
  weights the status quo, and checks explicitly against language models' lean toward
  "Yes".
- **Operational changes.**
  - The Fall 2026 tournament ids are built in, because the pinned `forecasting-tools`
    still points at Summer. `AIB_TOURNAMENT_ID`, `MINIBENCH_ID` and `METACULUS_CUP_ID`
    override them for later seasons.
  - GitHub fires the 20-minute schedule only about five times a day, and FutureEval
    questions stay open for 1.5 hours (3 for now). So a Cloudflare Worker (`trigger/`)
    starts the workflow every 20 minutes, with GitHub's schedule kept as a backup. Both
    stay off until you set `BOT_ENABLED`.
  - One run forecasts at most 25 new questions, soonest-closing first, so a new MiniBench
    round can't outlast the 60-minute timeout. The rest wait 20 minutes for the next run.
  - The Metaculus Cup workflow is manual-only.

Offline checks (no network, no keys): `python tests/check_rc_bot.py`.

## Setup

1. **Metaculus token.** Sign up for a human account, then go to Settings → "My Forecasting
   Bots" → "Create a Bot". Add the bot's API key as the repository secret `METACULUS_TOKEN`.
2. **Participant form.** Fill in the Fall 2026 participant form. The first section is
   required; its second section applies for the free LLM credits, which arrive as an
   OpenRouter key.
3. **Model key.** Add the key Metaculus sends as the secret `OPENROUTER_API_KEY`. The
   safest way is below: it prompts for the value, so the key never lands in a file or your
   shell history.

   ```bash
   gh secret set OPENROUTER_API_KEY --repo robcam55/metaculus-bot
   ```

   Without it, `ANTHROPIC_API_KEY` (your own key, billed to you) runs the bot unpaced on
   Sonnet 5.
4. **Research (optional but recommended).** Get free AskNews access: make an AskNews account
   with the bot's email, then contact AskNews. Add `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET`,
   or `ASKNEWS_API_KEY`.
   - The next run's page lists `asknews/latest` under research sources. Its search counts
     show whether the keys work.
   - Ask AskNews for the monthly quota, how archive searches count against it, and the
     rate limit. Then set `ASKNEWS_STRATEGY` and `ASKNEWS_ARTICLES` (below) to use it.
5. **Test.** Run `Actions → Test Bot → Run workflow`.
   - It forecasts three bot-testing-area questions (one of each type) at MiniBench's tier,
     for about $1 of credit.
   - Check that the forecasts appear on the bot's Metaculus profile.
   - Read the run page's summary for the balance and the measured cost per question.
6. **Turn it on.** Set the repository variable `BOT_ENABLED` to `true`.
7. **Start the 20-minute clock.** GitHub's own schedule fires only about five times a day.
   The Worker in `trigger/` starts the workflow every 20 minutes instead.
   1. Create a fine-grained GitHub token: GitHub → Settings → Developer settings →
      Personal access tokens → Fine-grained tokens → Generate new token.
      - Name: `metaculus-bot-trigger`.
      - Expiration: after the season, e.g. 2027-01-31.
      - Repository access: only `robcam55/metaculus-bot`.
      - Permissions: Repository permissions → Actions → Read and write. Nothing else.
   2. Deploy the Worker. Wrangler must be signed in to Cloudflare (`npx wrangler login`).

      ```bash
      npx wrangler deploy --config trigger/wrangler.jsonc
      ```

   3. Store the token in the Worker. The command prompts for the value:

      ```bash
      npx wrangler secret put GITHUB_TOKEN --config trigger/wrangler.jsonc
      ```

   Within 20 minutes a "Forecast on new AI tournament questions" run appears in the
   Actions tab, started by `workflow_dispatch`. Cloudflare's dashboard shows each tick
   under the Worker's logs. When the token expires, the runs fall back to GitHub's
   schedule; make a new token and repeat step 3.
8. **Keep the schedule alive.** GitHub pauses scheduled workflows in a public repo after 60
   days without a commit, and the Fall 2026 season runs to Jan 6. Push any commit at least
   every 50 days: the next by Nov 16.

Other optional repository variables (empty means the default):

| Variable | Default |
|---|---|
| `BUDGET_TIER` | Empty: paced. `opus-5`, `opus-3` or `sonnet-3` pins that tier for both tournaments. `pause` stops all spending of Metaculus's credits. The $2 floor always applies. |
| `TEST_QUESTIONS` | 3 (Test Bot only) |
| `AIB_TOURNAMENT_ID` | `33121`, the Fall 2026 FutureEval |
| `MINIBENCH_ID` | The package's current MiniBench (`minibench`) |
| `METACULUS_CUP_ID` | `33108`, the Fall 2026 Cup (personal key only) |
| `PARSER_MODEL` | Claude Haiku 4.5 through the same route |
| `SEARCH_MODEL` | `openrouter/anthropic/claude-sonnet-5:online` (OpenRouter route only) |
| `FORECASTER_MODEL` | `anthropic/claude-sonnet-5` (personal key only; on Metaculus's credits the tier picks) |
| `PREDICTIONS_PER_REPORT` | 5 (personal key only) |
| `ASKNEWS_STRATEGY` | `latest`: one latest-news search per question, covering the past day or two. `archive` searches the AskNews archive (about two months) instead; `both` does both, 12 seconds apart. Archive searches may count as several calls against AskNews's quota: confirm with AskNews first. |
| `ASKNEWS_ARTICLES` | 8 articles per AskNews search |
| `RESEARCH_REPORTS` | 1 (personal key only). Setting it to 2 with `PREDICTIONS_PER_REPORT=3` gives six predictions over two independent searches, for about 35–45% more cost. |

## Why it's built this way

The main architectural choices and the reasons behind them. Each links to where it was decided; ones marked *inferred* were never written down, so the reason given is the likely one, not a recorded one. When a new architectural decision is made, add it here. Links into rc_trader go to a private repo.

- **Metaculus's template and its `forecasting-tools` package, in Python.** The package fetches questions, parses and posts forecasts, and picks up each season's fixes with a version bump. The bot's own code is then only research, prompts, aggregation and spending. — [rc_trader's 2026-09-23 audit](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-09-23-AUDIT.md) (§8: D-14, and Rob's choice of a repo built from the template, approved 2026-09-23); the reason is *inferred*
- **GitHub Actions every 20 minutes, started by a Cloudflare Worker, from a public repo, with no storage of its own.**
  - **Why 20 minutes.** FutureEval questions open at random hours, up to five at a time, and each stays open for 1.5 hours (3 for now). rc_trader's nightly job can't serve that.
  - **Why Cloudflare starts the runs.**
    - GitHub fired this repo's 20-minute schedule only about five times a day: 45 runs between Sep 23 and Oct 2, a median of 5 hours apart. That would miss roughly half the questions.
    - A Worker's cron trigger is a reliable clock that needs no server, and its token can touch only this repo's Actions.
    - Rob chose it over a workflow that re-triggers itself, which needs no token but keeps a GitHub runner busy around the clock.
  - **Why no storage.** Metaculus keeps the forecasts, and OpenRouter keeps the spend.
  - **Why off by default.** Scheduled runs, and runs the Worker starts, stay off until `BOT_ENABLED`, so they can't fail every 20 minutes before the secrets exist.

  — [rc_trader's 2026-09-23 audit](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-09-23-AUDIT.md) (§8, Rob's hosting choice, 2026-09-23). The Cloudflare clock is Rob's choice of 2026-10-02, built in [#3](https://github.com/robcam55/metaculus-bot/pull/3). The question windows are from [Metaculus's resources page](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/), and the `BOT_ENABLED` gate from [the workflow](.github/workflows/run_bot_on_tournament.yaml). No storage is *inferred*.
- **Claude models: Opus 5.5 on Metaculus's credits, Sonnet 5 on a personal key.** The bot is the external check on rc_trader's forecasting engine, which runs on Claude, and Metaculus's credits cover Anthropic models. — [rc_trader's 2026-09-23 audit](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-09-23-AUDIT.md) (§8, D-14: "upgrade it as the §4 engine lands"). The switch to Opus 5.5 was Claude's, in [f4dfc32](https://github.com/robcam55/metaculus-bot/commit/f4dfc32), and Rob's credit application of 2026-09-24 described it.
- **Metaculus's credits pay first, and never for the Metaculus Cup.**
  - **Cup on a personal key.** Metaculus gives the credits for FutureEval and MiniBench only, so the Cup, and any experiment run there, needs a personal key.
  - **Credits before a personal key.** The personal key was meant as a stopgap until the credits arrived, so when both are set the credits win.

  — Metaculus's funding email to Rob (2026-09-27, not public); proposed by Claude in [#1](https://github.com/robcam55/metaculus-bot/pull/1), not yet ratified
- **Paced spending of incremental credits (`budget.py`).**
  - **The problem.** Metaculus seeds about $100 and adds more only after above-average MiniBench results, which take weeks to resolve. A fixed setup either runs dry before more arrives, leaving the bot dark while its results are judged, or underspends once it does. Before any measurement, five Opus forecasts a question looked like two to three weeks of credit; at the first measured costs it is closer to five.
  - **The rule.** Each run buys the richest tier the balance covers for four weeks of expected volume, with MiniBench (which decides the funding) never below FutureEval.
  - **The promise it keeps.** The credit application promised a fallback to Sonnet 5 if credits ran short; pacing makes that fallback automatic.
  - **Awaiting Rob's ratification.** The tiers, the four-week horizon and MiniBench's priority are Claude's proposal.

  — Metaculus's funding email to Rob (2026-09-27, not public); proposed in [#1](https://github.com/robcam55/metaculus-bot/pull/1), not yet ratified
- **Two independent research sources, several forecasts, a trimmed mean.**
  - **Two sources.** Search luck drove most of the night-to-night noise in rc_trader's own forecasts. So each question gets AskNews latest news and Sonnet 5 with native web search, and if one source fails, the other still feeds the forecast.
  - **One AskNews call by default.** AskNews's free tier allows 1,000 calls a month, so it gets one latest-news call per question. `ASKNEWS_STRATEGY` and `ASKNEWS_ARTICLES` raise that without a code change once AskNews confirms its quota.
  - **A trimmed mean.** Several forecasts from one report are combined with a trimmed mean, which beat the median in Halawi et al. (2024).

  - **No blind forecasts.** When every research source fails, the question waits for the next run rather than being forecast from the model's memory. A missed question adds nothing to the score, while a blind forecast on a question about current events can take points away. A question closing within 45 minutes gets no further try, so it is forecast without research and the forecasters are told so. A question that waits turns the run red, so a source that keeps failing gets noticed.

  — the noise finding from [rc_trader's 2026-09-23 audit](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-09-23-AUDIT.md) (§2.4 and §4). The design was Claude's, in [de2f36f](https://github.com/robcam55/metaculus-bot/commit/de2f36f) and [e0fac54](https://github.com/robcam55/metaculus-bot/commit/e0fac54), and Rob's credit application of 2026-09-24 described it; not otherwise ratified. No blind forecasts, the AskNews settings and the run-page report were Claude's proposals, which Rob approved on 2026-10-04.

---

# Simple Metaculus forecasting bot (template README)
This repository contains a simple bot meant to get you started with creating your own bot for the AI Forecasting Tournament. Go to https://www.metaculus.com/futureeval/participate/ for more info and tournament rules (and then go to the  "Getting Started" section of our [resources](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#want-to-join-the-ai-forecasting-benchmark) page).

**Brand new to this?** You can get a working bot running in about 5 minutes without writing a single line of code — just fork this repo, paste two API keys into GitHub, and click "Run workflow". See **[Quick start](#quick-start--fork-and-use-github-actions)** below.

In this project are 2 files:
- **main.py**: Our recommended template option that uses the [forecasting-tools](https://github.com/Metaculus/forecasting-tools) package to handle a lot of stuff in the background for you (such as API calls). We will update the package, thus allowing you to gain new features with minimal changes to your code.
- **main_with_no_framework.py**: A copy of main.py but implemented with minimal dependencies. Useful if you want a more custom approach.


Join the conversation about bot creation, get support, and follow updates on the [Metaculus Discord](https://discord.com/invite/NJgCC2nDfh) 'build a forecasting bot' channel.

## 30min Video Tutorial
This tutorial shows you how to set up our template bot so you can start forecasting in the tournament.

[![Watch the tutorial](https://cdn.loom.com/sessions/thumbnails/fc3c1a643b984a15b510647d8f760685-42b452e1ab7d2afa-full-play.gif)](https://www.loom.com/share/fc3c1a643b984a15b510647d8f760685?sid=29b502e0-cf64-421e-82c0-3a78451159ed)

If you run into trouble, reach out to `ben [at] metaculus [.com]`


## Quick start -> Fork and use Github Actions
The easiest way to use this repo is to fork it, paste in two API keys, and click "Run workflow". After that, the bot will keep forecasting on new questions automatically every 20 minutes — no local setup needed.

1) **Fork the repository** — go to the [repository](https://github.com/Metaculus/metac-bot-template) and click **Fork** in the top right.
2) **Add your two API keys as repository secrets** — in your fork, go to `Settings → Secrets and variables → Actions → New repository secret`. Add these two (names must match exactly, all caps):
   - **`METACULUS_TOKEN`** — create one at https://www.metaculus.com/futureeval/participate/ (see the [resources page](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#creating-your-bot-account-and-metaculus-token) if you get stuck).
   - **`OPENROUTER_API_KEY`** — get free credits via [this form](https://forms.gle/aQdYMq9Pisrf1v7d8), or make your own key on [OpenRouter](https://openrouter.ai/). You can also use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PERPLEXITY_API_KEY`, `ASKNEWS_SECRET`, etc. — these all work out of the box if you set them.
3) **Enable Actions** — click the `Actions` tab, then click `I understand my workflows, go ahead and enable them`.
4) **Run the test workflow to confirm everything works** — go to `Actions → Test Bot → Run workflow → Run workflow` (green button). This forecasts on whatever's currently open in the [bot-testing-area tournament](https://www.metaculus.com/tournament/bot-testing-area/) so you can verify your setup posts forecasts to Metaculus end-to-end. Once the run finishes (~3–5 min), check your bot's profile on Metaculus to confirm the forecasts landed.
5) **You're done!** The `Forecast on new AI tournament questions` workflow is already enabled and will run every 20 minutes, picking up any new tournament questions and skipping ones it has already forecast on.

To pause your bot, go to `Actions → Forecast on new AI tournament questions → ... (top right) → Disable workflow`.

### Testing your changes against the GitHub Actions workflow
You can run any workflow against any branch — no need to merge to `main` first, and no need to fork if you have push access to this repo.

1. Push your branch to GitHub: `git push origin <your-branch>`.
2. In the repo's Actions tab, pick the workflow you want to run (e.g. `Test Bot`) and click **Run workflow** (top right).
3. Use the **"Use workflow from"** dropdown to select your branch instead of `main`, then click the green **Run workflow** button.

The runner checks out your branch and uses the repo's existing secrets — those are scoped to the repo, not the branch, so they work for any branch in the same repo. This works for all three workflows.

## API Keys
Instructions for getting your METACULUS_TOKEN, OPENROUTER_API_KEY, or optional search provider API keys (AskNews, Exa, Perplexity, etc) are listed on the "Getting Started" section of the [resources](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#want-to-join-the-ai-forecasting-benchmark) page.

## Changing the Github automation
To run a different script under the same workflows, edit the `poetry run python main.py` line in the appropriate file under `.github/workflows/` and replace `main.py` with your script. The workflows that exist:
- `test_bot.yaml` — manual-trigger smoke test against the bot-testing-area tournament.
- `run_bot_on_tournament.yaml` — every 20 min on the live AIB tournament + MiniBench.
- `run_bot_on_metaculus_cup.yaml` — every 2 days on the Metaculus Cup.

**To run `main_with_no_framework.py` via GitHub Actions instead of `main.py`:** open the workflow file you want and change `poetry run python main.py` to `poetry run python main_with_no_framework.py`. That's the only change required.

## Editing in GitHub UI
Remember that you can edit a bot non locally by clicking on a file in Github, and then clicking the 'Edit this file' button. Whether you develop locally or not, when making edits, attempt to do things that you think others have not tried, as this will help further innovation in the field more than doing something that has already been done. Feel free to ask about what has or has not been tried in the Discord, see [other bot's self-descriptions](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#what-are-other-bots-doing), or read bot's [open source code](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#open-source-bots).

## Run/Edit the bot locally
Local development is optional — most new users can run the bot entirely from GitHub Actions (see [Quick start](#quick-start--fork-and-use-github-actions)). Set up locally only if you want faster iteration on your prompts/code.

### 1. Clone the repository
```bash
git clone https://github.com/Metaculus/metac-bot-template.git
cd metac-bot-template
```
If you've already forked the repo, replace the URL with your fork's URL (copy it from your fork's page in the browser).

### 2. Install Python 3.11+ and Poetry
You need:
- **Python 3.11 or newer** — get it from [python.org](https://www.python.org/downloads/) (or your OS package manager / `pyenv` / whatever you prefer).
- **Poetry** — see Poetry's [install docs](https://python-poetry.org/docs/#installation). The `pipx install poetry` route works on macOS, Linux, and Windows.

Confirm both are on your `PATH`:
```bash
python --version    # 3.11.x or higher
poetry --version
```

(Optional, recommended) Keep the virtualenv inside the project directory so your editor picks it up automatically:
```bash
poetry config virtualenvs.in-project true
```

### 3. Install dependencies
From inside the cloned repository:
```bash
poetry install
```

### 4. Set your API keys
Copy the template and fill in your real keys:
```bash
cp .env.template .env
```
Then open `.env` in any text editor and replace each `REPLACE_ME` with your real key. At minimum you need `METACULUS_TOKEN` and one LLM key (`OPENROUTER_API_KEY` is recommended). See the comments inside `.env.template` for where to get each one.

### 5. Run the bot
**First run — smoke-test against the [bot-testing-area tournament](https://www.metaculus.com/tournament/bot-testing-area/):**
```bash
poetry run python main.py --mode test_questions
```
You'll see a one-line startup banner, forecasting progress logs, then a `🎉 Bot submitted N forecast(s)` banner with direct links to each forecast on Metaculus.

**Forecast on live AIB tournament + MiniBench:**
```bash
poetry run python main.py --mode tournament
```

**Forecast on the Metaculus Cup:**
```bash
poetry run python main.py --mode metaculus_cup
```

**Run the no-framework reference implementation instead:**
```bash
poetry run python main_with_no_framework.py
```
This file has no `--mode` flag; it's controlled by the constants at the top of the file (`SUBMIT_PREDICTION`, `USE_EXAMPLE_QUESTIONS`, `TOURNAMENT_ID`, etc.). Flip `USE_EXAMPLE_QUESTIONS = True` to point it at the bot-testing-area tournament instead of the live AIB.

To stop publishing forecasts (dry-run mode):
- `main.py`: set `publish_reports_to_metaculus=False` in the `SummerTemplateBot2026(...)` constructor near the bottom.
- `main_with_no_framework.py`: set `SUBMIT_PREDICTION = False` at the top.

## Reviewing how your bot did

Once your questions start resolving, the community-member-maintained optional
[bot-review](https://github.com/LouisP96/metaculus-bot-review) integration scores them and
helps to diagnose any reasoning errors.

```bash
poetry install --with integrations
poetry run bot-review review --resolved-since 30
```

A weekly workflow and a Claude Code skill come with it. See the
[integrations README](integrations/README.md#bot-review).

## Example usage of /news and /deepnews:
If you are using AskNews, here is some useful example code.
```python
from asknews_sdk import AsyncAskNewsSDK
import asyncio

"""
More information available here:
https://docs.asknews.app/en/news
https://docs.asknews.app/en/deepnews

Installation:
pip install asknews
"""

client_id = ""
client_secret = ""

ask = AsyncAskNewsSDK(
    client_id=client_id,
    client_secret=client_secret,
    scopes=["chat", "news", "stories", "analytics"],
)

# /news endpoint example
async def search_news(query):

  hot_response = await ask.news.search_news(
      query=query, # your natural language query
      n_articles=5, # control the number of articles to include in the context
      return_type="both",
      strategy="latest news" # enforces looking at the latest news only
  )

  print(hot_response.as_string)

  # get context from the "historical" database that contains a news archive going back to 2023
  historical_response = await ask.news.search_news(
      query=query,
      n_articles=10,
      return_type="both",
      strategy="news knowledge" # looks for relevant news within the past 60 days
  )

  print(historical_response.as_string)

# /deepnews endpoint example:
async def deep_research(
    query, sources, model, search_depth=2, max_depth=2
):

    response = await ask.chat.get_deep_news(
        messages=[{"role": "user", "content": query}],
        search_depth=search_depth,
        max_depth=max_depth,
        sources=sources,
        stream=False,
        return_sources=False,
        model=model,
        inline_citations="numbered"
    )

    print(response)


if __name__ == "__main__":
    query = "What is the TAM of the global market for electric vehicles in 2025? With your final report, please report the TAM in USD using the tags <TAM> ... </TAM>"

    sources = ["asknews"]
    model = "deepseek-basic"
    search_depth = 2
    max_depth = 2
    asyncio.run(
        deep_research(
            query, sources, model, search_depth, max_depth
        )
    )

    asyncio.run(search_news(query))
```

Some tips for DeepNews:

You will get tags in your response, including:

<think> </think>
<asknews_search> </asknews_search>
<final_response> </final_response>

These tags are likely useful for extracting the pieces that you need for your pipeline. For example, if you don't want to include all the thinking/searching, you could just extract <final_response> </final_response>


## Integrations

The **[integrations/](integrations/)** folder contains example scripts that integrate third-party tools with the bot template. 

See the [integrations README](integrations/README.md) for available integrations and how to add your own.

## Ideas for bot improvements
You can find some ideas of what you can do to improve this template by taking a look at what other bots have done [here](https://www.metaculus.com/notebooks/43497/what-are-other-bots-doing/). You can also look at research done by Metaculus and the field in the [research section](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#research-reports-and-overview-of-the-field) of the bot resources page. Asking an LLM to read through everything and give ideas may be a decent place to start. Please try to do something new, or something that is a spinoff (or better implementation) of what others have done. We don't want to test the same idea multiple times.
