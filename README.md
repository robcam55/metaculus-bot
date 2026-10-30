# rc_trader's Metaculus forecasting bot

A Claude-based bot for Metaculus's AI forecasting tournament (FutureEval). It is built on
Metaculus's template bot (`Metaculus/metac-bot-template` at commit `6dab04c`); the
template's own README follows below. This bot is the external accuracy lab for rc_trader,
a personal forecasting and paper-trading project. The tournament scores hundreds of
questions a season against professional forecasters, with no money at risk.

## What's different from the template

The changes are in `main.py` (`RcForecastBot`), `budget.py`, `spend.py`, `alerts.py`,
`forecast_guards.py`, `clock_watchdog.py`, `model_watch.py` and `scorecard.py`:
- **Claude models, on Metaculus's credits first.**
  - Metaculus's donated OpenRouter credits (`OPENROUTER_API_KEY`) pay before a personal
    `ANTHROPIC_API_KEY`.
  - They never pay for the Metaculus Cup, because Metaculus gives them for FutureEval and
    MiniBench only.
  - On the credits, the pacing tier below picks Opus 5.5 or Sonnet 5.5. The Cup's personal
    key runs Sonnet 5.5.
  - **Your own key takes over when the credits run out.** Once they can't pay for another
    question above the $2 reserve (at most $2.65 left at `opus-3`; at measured costs the
    last credits run usually ends a little under $2), and if `ANTHROPIC_API_KEY` is set,
    FutureEval and MiniBench run on your key at
    `opus-3` (or the tier `BUDGET_TIER` pins), with the same research. The cap is the
    monthly spend limit on the key's workspace in the Claude Console, which Anthropic
    enforces; at the limit the bot pauses quietly until the 1st. Each run checks the key
    with one tiny request before it spends anything else on it. A Metaculus top-up moves
    the bot back onto the credits at the next run. A run that can't read the credits'
    balance, or has no OpenRouter key at all, also runs on your key, the same way.

  Haiku 5.5 parses in every case, at low effort. These models reject sampling parameters,
  so none are sent. On Anthropic's own API the forecasters think at `high` effort
  (`FORECASTER_EFFORT`; Opus 5.5 defaults to `medium`), and each reply may run to 32,000
  tokens (research 16,000, parsing 8,000), where litellm's default of 4,096 would leave
  Opus 5.5, Sonnet 5.5 and Haiku 5.5, which always think, too little room.
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
  | `sonnet-3` | 3 × Sonnet 5.5 | $0.23 |
  | `sonnet-1` | 1 × Sonnet 5.5, the last resort | $0.18 |

  The estimates are about 40% above the first live test (2026-10-02): `opus-3` cost $0.13
  to $0.35 a question, $0.21 on average, and web-search research was most of it.
  - **MiniBench first.** MiniBench results decide further funding, so MiniBench never runs
    below FutureEval. At $100, FutureEval runs `opus-3` and MiniBench `opus-5`; from about
    $106, both run `opus-5`.
  - **The last week.** When even `sonnet-3` can't pay for a week of questions (below
    about $17), both tournaments drop to `sonnet-1`, so the last of the credit covers more
    questions. Research is most of a question's cost, so this stretches it by only about a
    quarter.
  - **A pin skips the ladder.** `BUDGET_TIER` pins one tier for both tournaments, and
    then neither the ladder nor the last week applies; the floor still does. It has
    pinned `opus-3` since 2026-10-07.
  - **Never runs dry.** One run takes no more questions than the balance above $2 covers
    at measured costs ($0.65 a question at `opus-3`, $0.75 at `opus-5`, $0.55 at
    `sonnet-3`, $0.50 at `sonnet-1`), not at the estimates above, and the $2 reserve
    absorbs a costly question. When the balance can't pay for one more question at the
    tier, the credits stop: your own key takes over if it is set, and otherwise the bot
    pauses and says so. A balance that can't be read is tried once more.
  - **Top-ups apply at once.** When Metaculus raises the key, the next run sees it.
  - **Measured spend.** Each run's page on GitHub shows the balance, what each tier cost
    per question, and a table per tournament of what each question cost: each research
    source, each forecast, and parsing with the research summary (`spend.py`).
    - The costs are OpenRouter's own figures for every model call, recorded as each
      response returns, retries and questions left waiting for research included. The log
      has one `Spend:` line per call.
    - OpenRouter's balance lags by minutes, so the bot reads it once, at the start of a
      run. The next run's starting balance is the check on this run's figures.
    - Until 2026-10-09 the costs came from litellm's price list, which ran about 12% below
      the balance. Then litellm dropped OpenRouter's models from the list, and every run
      page read $0.00. Pacing works from the balance, so it wasn't affected.
    - Recalibrate the estimates in `budget.py` from these numbers.
  - **Alerts as GitHub issues.** The bot opens an issue, which emails the repository's
    owner, the first time the credits fall below $15, when they run out, when your own
    key hits its spend limit or is rejected, and when the Cloudflare clock stops. Each
    event opens one issue, never again for the same credit limit, so there's no stream of
    red runs. Your key's alerts and the clock's count only while their issue is open: close
    it once fixed, and a repeat opens a new one.
- **Two independent research sources per research report.**
  - AskNews news search: by default one latest-news call per question, to fit the free
    tier's 1,000 calls a month. The template's version uses six. `ASKNEWS_STRATEGY` and
    `ASKNEWS_ARTICLES` change this without a code change.
  - A search-backed model: Sonnet 5.5 with OpenRouter's `:online` (Anthropic's own web search,
    which Metaculus's credits cover). On your own key, Sonnet 5.5 calls Anthropic's web
    search tool directly (`web_search_20260209`, which filters results before they enter
    the context), up to four searches a question, with each round of results cached for
    the next search and its sources kept. It can also open two pages a question with
    Anthropic's web fetch tool (`web_fetch_20260209`, up to 8,000 tokens a page;
    `RESEARCH_WEB_FETCH=false` turns it off). Or Perplexity, with its own key. Its prompt
    carries the question's background, and asks for:
    - the source the resolution criteria name, opened first where it can be: its latest
      value or status, any schedule or prediction it publishes for the period, and whether
      the outcome already looks settled;
    - a searched base rate, as a count with its period;
    - for a price or data series, its latest value and typical daily move;
    - for each of Polymarket, Kalshi, Manifold and Metaculus, a forecast on the question
      or a close one, or "not found" (the tournament's rules allow published forecasts).

  Every prompt, the search-backed model's and the forecasters', carries today's date and
  the question's close and resolution times.

  If one source fails, the other still feeds the forecast. If every source fails (an
  AskNews search that finds nothing, or a web search cut short, counts as failed), the
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
  - A red run whose job reads "cancelled" after about 15 minutes, with no steps and no log,
    never got a GitHub runner, so the bot never started. Nothing needs fixing: the next run
    20 minutes later takes the questions. A red run from the bot itself has a failed "Run
    bot" step and names the questions in its log.
- **A comment that keeps its sections.** Metaculus gets one comment per forecast, with
  `# SUMMARY`, `# RESEARCH` and `# FORECASTS` sections that `forecasting-tools` and
  `bot-review` find by position. Headings inside research and the research summary are kept
  as bold lines, so they can't add or shift a section.
- **Several predictions per question** from one research report: three or five by tier,
  on Metaculus's credits or on your own key after them; five on the Cup. With four or
  more, binary questions use a trimmed mean (dropping the highest and lowest), which beat
  the median in Halawi et al. (2024). With three they use the package's median. Numeric,
  discrete and date questions use the mixture of the forecasts (their mean CDF), less any
  whose range doesn't overlap the others' at all (a misread unit), which is wider where
  they disagree than the package's median.
- **Guards for forecasting-tools 0.2.92 (`forecast_guards.py`).** One forecaster's answer
  could sink a whole question there, after its research was paid for. Numeric
  percentiles repeated at an open lower bound made a CDF that failed the question or froze
  the run until its timeout (bounded now, as in 0.3.5); a confident multiple-choice answer
  with several 0% options, or a sum a little off 100%, failed on every retry (parsed
  leniently now, each option floored at 1%).
- **Forecast prompts with specific rules.**
  - Yes/no: restate exactly what must happen, by when and per which source; an outside
    view, with a yearly rate scaled to the days left; the status quo read from the named
    source, and whether it means nothing changes or a process reaching its scheduled end;
    announced target dates as plans, not promises; a search that finds nothing as no
    evidence; price thresholds read off a distribution of where the price will be
    (touching a level any time before the deadline is about twice as likely as being past
    it on the day); a liquid, closely matched market as strong evidence; and a check of
    the number in both directions, in place of the old one-sided "models over-predict
    Yes" step.
  - Multiple choice: how often the favourite actually wins, how it would fail, and the
    same market rule.
  - Numeric and date: the range's width set by how forecastable the quantity is, in
    place of the template's "set wide intervals"; for traded prices, a floor of about
    1.3 x the typical daily move x the square root of the trading days left on each side.
  - Each forecast's reasoning and its line in the posted summary name the model that made
    it, for the scorecard's per-model tables.
- **Operational changes.**
  - The Fall 2026 tournament ids are built in, because the pinned `forecasting-tools`
    still points at Summer. `AIB_TOURNAMENT_ID`, `MINIBENCH_ID` and `METACULUS_CUP_ID`
    override them for later seasons.
  - GitHub fires a 20-minute schedule only about five times a day, and FutureEval
    questions stay open for 1.5 hours (3 for now). So a Cloudflare Worker (`trigger/`)
    starts the workflow every 20 minutes. GitHub's schedule is kept as a backup, in a
    workflow of its own (`backup_clock.yaml`), so GitHub's 60-day pause can't stop the
    tournament workflow. Each backup tick also checks the Worker is still starting runs
    (`clock_watchdog.py`) and opens an issue if not. Both clocks stay off until you set
    `BOT_ENABLED`.
  - One run forecasts at most 25 new questions, soonest-closing first, so a new MiniBench
    round can't outlast the 60-minute timeout. The rest wait 20 minutes for the next run.
  - The Metaculus Cup workflow is manual-only.
  - **New Claude models are flagged, not switched to.** Once a day `model_watch.yaml`
    lists the models your key can use and opens one issue for each Claude model released
    since 2026-10-28, saying which role it could take and how to try it. It also opens an
    issue if a model the bot runs drops off the list. The Test Bot's model inputs try a new
    model on bot-testing-area questions without a code change (Setup, step 10).
- **A calibration scorecard in the weekly review (`scorecard.py`).** Mean spot peer score
  per question with its standard error, by tournament and type; binary calibration;
  where numeric outcomes fell against the forecasters' 10th and 90th percentiles; the
  probability given to the winning option on multiple choice; mean spot peer by kind of
  question (prices and markets, status on a date, sports, reported counts, other, from the
  title); and, paired on the same questions, the aggregate against its own members and
  each model against the most common one.

Offline checks (no keys): `python tests/check_rc_bot.py`. They also run on every pull
request and push to `main`, on the packages `poetry.lock` pins, which the forecasting runs
install (`.github/workflows/checks.yaml`). A local virtualenv can drift from the lock: a
newer local litellm once hid a call that crashes on the pinned one.

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

   Without it, `ANTHROPIC_API_KEY` (your own key, billed to you) runs FutureEval and
   MiniBench as it does after the credits (step 9): the pinned tier, `opus-3` without a
   pin, with its key check. With both, your key only takes over when the credits run out,
   or when a run can't read their balance.
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
   - Read the run page's summary for the balance and the measured cost per question,
     split by research, forecasts and parsing.
   - Setting "Who pays" to `own-key` runs it on your own key instead (step 9).
     "Questions" sets how many it forecasts (empty: `TEST_QUESTIONS`, or 3).
   - The forecaster, researcher and parser inputs try another model in that role for this
     run only (step 10). Empty runs the bot's own.
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
   under the Worker's logs. When the token expires, the runs fall back to the backup clock,
   about every four hours, and within a few hours an issue says the Cloudflare clock
   stopped; make a new token and repeat step 3.
8. **The backup clock's 60-day pause.** GitHub disables a scheduled workflow in a public
   repo after 60 days without a commit, and a disabled workflow refuses every trigger. Only
   the backup clock (`backup_clock.yaml`), the model watch (`model_watch.yaml`) and the
   weekly review have a schedule, so only they can lapse; the Worker keeps starting the tournament workflow either way. GitHub
   emails a week before. The Cloudflare watchdog runs inside the backup clock and lapses
   with it. To keep both, push any commit within 60 days of the last (for this season, one
   after about Nov 8 lasts to Jan 6), or re-enable it afterwards (`Actions → Backup clock →
   Enable workflow`).
9. **Your own key, for when the credits run out (optional).** The bot moves to it once
   Metaculus's credits can't pay for another question (around $2 left), and back to the
   credits when Metaculus tops them up.
   At `opus-3` a question costs about $0.65 on your key (range $0.55 to $0.85; research
   is most of it). This season's volume is about 4 FutureEval questions a day plus a
   MiniBench round of about 60 every two weeks, so a month with three rounds starting
   (November 2026) comes to about 300 questions and $195 (range $165 to $255), before
   whatever the forecasters' `high` effort adds: the Test Bot's and the first runs'
   `Spend:` lines will show it. A limit hit
   mid-month pauses the bot, and every question it misses scores zero.
   1. In the Claude Console, create a workspace for the bot (Settings → Workspaces). The
      default workspace can't take a spend limit.
   2. Open the new workspace's limits and set a monthly spend limit: about $350 leaves
      room for a heavy month. That is the cap: Anthropic enforces it, and it resets on the
      1st. At the limit the bot pauses quietly and opens one issue. The organization's own
      monthly cap (its usage tier) and, on a prepaid account, its credit balance can stop
      the key first: check both, or turn on auto-reload.
   3. Create an API key in that workspace (Settings → API keys), then store it. The command
      prompts for the value:

      ```bash
      gh secret set ANTHROPIC_API_KEY --repo robcam55/metaculus-bot
      ```

   4. Test it: `Actions → Test Bot → Run workflow` with "Who pays" set to `own-key`. It
      forecasts three bot-testing-area questions on your key, for about $2. The run
      page's "Your Anthropic key" section shows the spend, and the research table should
      show `anthropic/claude-sonnet-5-5 + web search and fetch` working. The log's `Spend:` lines show
      the research calls' cache reads.

   The Metaculus Cup workflow uses this key too, but only when you run it by hand.
10. **When a new Claude model comes out.** An issue titled "New Claude model to evaluate"
    opens within a day of the release, mentioning you. Nothing switches on its own: two
    releases in a row each rejected a setting the bot sent (Haiku 5.5 the parser's
    temperature, Sonnet 5.5 a way of turning thinking off), and every question that fails
    scores zero. To evaluate it:
    1. **What changed.** Read Anthropic's migration guide and pricing for it: settings it
       rejects, new defaults such as effort, and a new tokenizer, which changes the cost
       of a question.
    2. **Does the bot run on it.** `Actions → Test Bot → Run workflow` with "Who pays" set
       to `own-key` and the new model's id (`anthropic/claude-...`) in the forecaster,
       researcher or parser input. That's three bot-testing-area questions, about $2,
       never tournament questions. The run should go green, the run page should list the
       model under "On trial", and its spend should have a figure (litellm has to know
       the price).
    3. **Is it better.** Not by re-forecasting resolved questions: a newer model's
       training data runs later, so it may already know the answers. Use evidence
       gathered before the questions resolved: Metaculus's own bots on the new model in
       FutureEval, if it runs any; benchmarks scored on live questions, such as
       ForecastBench; and the price against any gain.
    4. **Switch.** Change the id in `budget.py` (`OPUS`, `SONNET` or `HAIKU`, and
       `ANTHROPIC_IDS`; `main.py` reads them from there), run the offline checks and the
       Test Bot, and merge, ideally between MiniBench rounds so each round runs on one
       model. Then close the issue; it won't open again.

    An issue titled "Claude model no longer offered" means a model the bot runs is gone
    from your key's list and every call to it fails: switch that one first. Anthropic
    also emails deprecation notices to the account well before a model is retired.

Other optional repository variables (empty means the default):

| Variable | Default |
|---|---|
| `BUDGET_TIER` | Empty: paced. `opus-5`, `opus-3`, `sonnet-3` or `sonnet-1` pins that tier for both tournaments, on the credits and on your own key after them. `pause` stops all spending, your key's included. The $2 floor always applies. |
| `TEST_QUESTIONS` | 3 (Test Bot only) |
| `AIB_TOURNAMENT_ID` | `33121`, the Fall 2026 FutureEval |
| `MINIBENCH_ID` | The package's current MiniBench (`minibench`) |
| `METACULUS_CUP_ID` | `33108`, the Fall 2026 Cup (personal key only) |
| `PARSER_MODEL` | Claude Haiku 5.5 through the same route. `anthropic/claude-haiku-4-5` brings back the old parser (at temperature 0). |
| `SEARCH_MODEL` | `openrouter/anthropic/claude-sonnet-5.5:online` (OpenRouter route only) |
| `FORECASTER_MODEL` | `anthropic/claude-sonnet-5-5` (Metaculus Cup only; on the credits, and on your own key after them, the tier picks) |
| `RESEARCH_WEB_FETCH` | On (Rob's own key only): research may open two pages a question. `false` turns it off, leaving web search alone. |
| `FORECASTER_EFFORT` | `high` (Anthropic's own API only). `low` or `medium` sets that level; `default` sends none, which is `medium` on Opus 5.5 and `high` on Sonnet 5.5. |
| `PREDICTIONS_PER_REPORT` | 5 (Metaculus Cup only) |
| `ASKNEWS_STRATEGY` | `latest`: one latest-news search per question, covering the past day or two. `archive` searches the AskNews archive (about two months) instead; `both` does both, 12 seconds apart. Archive searches may count as several calls against AskNews's quota: confirm with AskNews first. |
| `ASKNEWS_ARTICLES` | 8 articles per AskNews search |
| `RESEARCH_REPORTS` | 1 (Metaculus Cup only). Setting it to 2 with `PREDICTIONS_PER_REPORT=3` gives six predictions over two independent searches, for about 35–45% more cost. |

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
  - **Why off by default.** Runs the clocks start stay off until `BOT_ENABLED`, so they can't fail every 20 minutes before the secrets exist.
  - **Why the backup clock is a workflow of its own.** GitHub disables a scheduled workflow in a public repo after 60 days without a commit, and the disable refuses every trigger, `workflow_dispatch` included. With the backup schedule inside the tournament workflow, that pause (about Dec 23 from the Oct 24 commit, two weeks before the season ends) would have stopped the Worker's runs too. The tournament workflow now has no schedule; only the backup can lapse.
  - **Why a watchdog.** When GitHub turns the Worker away, the error shows only in Cloudflare's logs and the bot drops to the backup clock, every four hours or so, missing a third to two thirds of FutureEval questions with every run green. `clock_watchdog.py` opens an issue when no Worker run is under two hours old.

  — [rc_trader's 2026-09-23 audit](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-09-23-AUDIT.md) (§8, Rob's hosting choice, 2026-09-23). The Cloudflare clock is Rob's choice of 2026-10-02, built in [#3](https://github.com/robcam55/metaculus-bot/pull/3). The question windows are from [Metaculus's resources page](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/), and the `BOT_ENABLED` gate from [the workflow](.github/workflows/run_bot_on_tournament.yaml). No storage is *inferred*. The backup clock's own workflow and the watchdog came from Claude's health review of 2026-10-28, with Rob's go-ahead that day, in [#13](https://github.com/robcam55/metaculus-bot/pull/13).
- **Claude models: Opus 5.5 on Metaculus's credits and on Rob's key after them, Sonnet 5.5 on the Cup's personal key; Sonnet 5.5 searches and Haiku 5.5 parses.** The bot is the external check on rc_trader's forecasting engine, which runs on Claude, and Metaculus's credits cover Anthropic models. — [rc_trader's 2026-09-23 audit](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-09-23-AUDIT.md) (§8, D-14: "upgrade it as the §4 engine lands"). The switch to Opus 5.5 was Claude's, in [f4dfc32](https://github.com/robcam55/metaculus-bot/commit/f4dfc32), and Rob's credit application of 2026-09-24 described it. Sonnet 5.5 (same price as Sonnet 5) and Haiku 5.5 (a tenth of Haiku 4.5's price) replaced Sonnet 5 and Haiku 4.5 on 2026-10-28, at Rob's request to run the newest version of each model, in [#13](https://github.com/robcam55/metaculus-bot/pull/13). Opus 5.5 was already the newest Opus. Claude Fable 5.1, a tier above Opus at 2.5 times its price, was left as Rob's call.
- **Metaculus's credits pay first, and never for the Metaculus Cup.**
  - **Cup on a personal key.** Metaculus gives the credits for FutureEval and MiniBench only, so the Cup, and any experiment run there, needs a personal key.
  - **Credits before a personal key.** The personal key was meant as a stopgap until the credits arrived, so when both are set the credits win.

  — Metaculus's funding email to Rob (2026-09-27, not public); proposed by Claude in [#1](https://github.com/robcam55/metaculus-bot/pull/1), not yet ratified
- **Paced spending of incremental credits (`budget.py`).**
  - **The problem.** Metaculus seeds about $100 and adds more only after above-average MiniBench results, which take weeks to resolve. A fixed setup either runs dry before more arrives, leaving the bot dark while its results are judged, or underspends once it does. Before any measurement, five Opus forecasts a question looked like two to three weeks of credit; at the first measured costs it is closer to five.
  - **The rule.** Each run buys the richest tier the balance covers for four weeks of expected volume, with MiniBench (which decides the funding) never below FutureEval.
  - **The promise it keeps.** The credit application promised a fallback to Sonnet 5 if credits ran short; pacing makes that fallback automatic.
  - **Awaiting Rob's ratification.** The tiers, the four-week horizon and MiniBench's priority are Claude's proposal.
  - **Estimates kept below measured costs.** On 2026-10-05 MiniBench questions cost $0.68 at `opus-5` and $0.45 at `opus-3` (40 questions, by the settled balance), against estimates of $0.40 and $0.30, so the pacer holds richer tiers than four weeks of credit would buy. Rob chose on 2026-10-06 to keep the estimates and tiers: he expects Metaculus to add credit after the early MiniBench results, and would fund the bot himself otherwise. In place of lowering the tiers sooner, the last week of credit runs at `sonnet-1` when no tier is pinned.
  - **Pinned to Opus since 2026-10-07 (D-21).** `BUDGET_TIER=opus-3` holds both tournaments on three Opus 5.5 forecasts. Web search is about 70% of a question's cost, so Sonnet 5 would save only $0.03 to $0.06 a question, and MiniBench decides the top-ups. While pinned, neither the ladder nor the `sonnet-1` last week applies; the $2 floor does.

  — Metaculus's funding email to Rob (2026-09-27, not public); proposed in [#1](https://github.com/robcam55/metaculus-bot/pull/1), not yet ratified. The pin is [rc_trader's D-21](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-10-06-DECISIONS.md), approved by Rob on 2026-10-06 and set by him on 2026-10-07.
- **Spend measured call by call, from OpenRouter's own figures (`spend.py`).**
  - **Why not the package's cost tracking.** `forecasting-tools` prices each response from litellm's price list. That list never carried web-search fees, so it ran about 12% below the balance, and on 2026-10-09 it dropped OpenRouter's models: every run page since read $0.00 while the balance kept falling.
  - **Why both of OpenRouter's figures.** OpenRouter's billed `cost` read $0.00 on all 99 calls the logs show it for (2026-10-05 to 10-09), while the balance kept falling, so Metaculus's credits look like bring-your-own-key: the provider bills the call and OpenRouter adds its fee. A call's cost is both together. That reading is *inferred* from the logs; each run page counts which figure every call's cost came from, so the first runs confirm or refute it.
  - **Why per question and part of the work.** Research, forecasts and parsing cost very different amounts, and the decisions on tiers and research sources turn on the split.
  - **On Rob's own key.** Anthropic's API returns no cost, so each call is priced as its tokens at litellm's prices plus $0.01 a web search. litellm 1.80.10's own figure for a response leaves the searches out, about 10% of a question's cost. If litellm can't price the model, OpenRouter's list prices stand in. Tokens read from or written to the prompt cache are priced at their own rates (a tenth of the input price to read on Sonnet 5, 1.25 times to write), since the research call caches its search rounds (2026-10-28); pricing them as ordinary input had overstated a cached call by half.

  — Rob's go-ahead of 2026-10-12 for measurement only, built in [#7](https://github.com/robcam55/metaculus-bot/pull/7). Pricing Rob's key as tokens plus searches is Claude's fix in [#8](https://github.com/robcam55/metaculus-bot/pull/8), found by the 2026-10-15 review; not otherwise ratified.
- **Rob's own Anthropic key after the credits, capped by a workspace spend limit.**
  - **Why.** At the measured costs Metaculus's first $100 runs out around the end of October, and the season runs to Jan 6. Without more credit the bot would sit dark for about nine weeks. On 2026-10-06 Rob said he would fund the bot himself if Metaculus didn't.
  - **How.** Once the credits can't pay for another question above the $2 reserve and `ANTHROPIC_API_KEY` is set, both tournaments run on Rob's key at the pinned tier (`opus-3` without a pin), with the same two research sources: Sonnet 5 calls Anthropic's web search tool where the credits use OpenRouter's `:online`. A top-up moves the bot back onto the credits at the next run.
  - **Why "can't pay for another question", not "under $2".** The run cap spends only while the balance above $2 covers a question, so a balance can come to rest above $2 (up to $2.65 at `opus-3`) and stay there. The first version switched only under $2, so the bot would have sat idle in that band, on neither key, until a top-up. A pre-launch review caught this on 2026-10-15. The same review moved the run cap from the pacing estimates to measured costs, because capping at $0.30 a question let a burst plan about twice what lay above the reserve, so OpenRouter could refuse the key mid-question. The pacing estimates themselves are unchanged (Rob's choice of 2026-10-06).
  - **The cap lives at Anthropic.** A plain key can't read what its workspace has spent, and the bot keeps no storage, so the cap is the workspace's monthly spend limit in the Claude Console. Anthropic enforces it, and it can't drift the way a count the bot kept itself could. One tiny request before the first question of a run tells a key at its limit, or a rejected one, from a working one. The bot then pauses quietly and opens one issue, without spending AskNews calls or research.

  — Rob's choice of option (b) in chat on 2026-10-11, after Claude's status review that day. The design (the tier, Anthropic's web search, the cap as a workspace spend limit, the key check) is Claude's, in [#6](https://github.com/robcam55/metaculus-bot/pull/6). The "can't pay for another question" rule and the measured run cap are Claude's, in [#8](https://github.com/robcam55/metaculus-bot/pull/8). None of it is otherwise ratified.
- **Alerts as GitHub issues, one per event.** The bot runs every 20 minutes, so a run that fails to get attention would email Rob every 20 minutes until fixed. Instead `alerts.py` opens one issue, mentioning the owner, the first time the credits fall below $15, when they run out, and when Rob's key hits its spend limit or is rejected. An issue with the same title, open or closed, stops a repeat. Titles carry the credit limit, or the month for Rob's key, so a new limit or month alerts again. The workflow's token gets `issues: write` for this and nothing more.
  - **Rob's key's alerts, and the stopped clock's, count only while open.** After he raises a limit or replaces a key and closes the issue, a second stop the same month would otherwise stay silent, the bot paused with every run green.
  - **Read twice before opening.** On 2026-10-22 one list read in about 590 came back without the existing issue, and #10 duplicated #9.

  — Rob approved a low-credit warning in chat on 2026-10-11. Claude proposed failing the run; the issue form replaced that on finding the email stream it would cause, in [#6](https://github.com/robcam55/metaculus-bot/pull/6); not otherwise ratified. The open-only rule and the second read are from Claude's 2026-10-28 health review, with Rob's go-ahead that day, in [#13](https://github.com/robcam55/metaculus-bot/pull/13).
- **Two independent research sources, several forecasts, a trimmed mean.**
  - **Two sources.** Search luck drove most of the night-to-night noise in rc_trader's own forecasts. So each question gets AskNews latest news and Sonnet 5.5 with native web search, and if one source fails, the other still feeds the forecast.
  - **One AskNews call by default.** AskNews's free tier allows 1,000 calls a month, so it gets one latest-news call per question. `ASKNEWS_STRATEGY` and `ASKNEWS_ARTICLES` raise that without a code change once AskNews confirms its quota.
  - **A trimmed mean.** Several forecasts from one report are combined with a trimmed mean, which beat the median in Halawi et al. (2024).

  - **No blind forecasts.** When every research source fails, the question waits for the next run rather than being forecast from the model's memory. A missed question adds nothing to the score, while a blind forecast on a question about current events can take points away. A question closing within 45 minutes gets no further try, so it is forecast without research and the forecasters are told so. A question that waits turns the run red, so a source that keeps failing gets noticed.

  — the noise finding from [rc_trader's 2026-09-23 audit](https://github.com/robcam55/rc_trader/blob/main/docs/audit/2026-09-23-AUDIT.md) (§2.4 and §4). The design was Claude's, in [de2f36f](https://github.com/robcam55/metaculus-bot/commit/de2f36f) and [e0fac54](https://github.com/robcam55/metaculus-bot/commit/e0fac54), and Rob's credit application of 2026-09-24 described it; not otherwise ratified. No blind forecasts, the AskNews settings and the run-page report were Claude's proposals, which Rob approved on 2026-10-04.
- **Workflow actions pinned to commit SHAs, not version tags.** Every run carries the bot's API keys, including Rob's paid Anthropic, OpenAI and OpenRouter keys, and runs start every 20 minutes. A tag such as `@v1` can be moved to new code by the action's owner, or by anyone who takes over the owner's GitHub account. The OVHcloud CLI and LiteLLM 1.107.0 compromises of October 2026 both started that way. A commit SHA can't be moved. Each pin names its version in a comment, so an upgrade is a deliberate edit.

  — Rob, 2026-10-23 ("Yes — pin the third-party GitHub Actions to commit SHAs"), after Claude's Black October audit of 2026-10-22
- **Forecasts at high effort on Rob's key.** Opus 5.5 thinks at `medium` when no effort is sent, one level below Opus 5's default, and the bot sent none. In Metaculus's Spring 2026 analysis, high-effort versions beat their standard-effort twins in 8 of 8 model pairs, but most of those pairs compared no thinking with thinking, so the step from `medium` to `high` on a model that always thinks is smaller and unproven. The three forecasts were about $0.15 of a question's $0.60 to $0.65 at `medium`; the extra thinking's cost is unmeasured until the first own-key runs at `high` (their `Spend:` lines show it). `FORECASTER_EFFORT` sets it back without a code change. On the credits route (OpenRouter) no effort is sent yet.

  — Claude's proposal in the 2026-10-28 health review; Rob's go-ahead that day in chat ("move forward with all of your recommended enhancements that are within the rules of the competition"), built in [#13](https://github.com/robcam55/metaculus-bot/pull/13).
- **Guards for forecasting-tools 0.2.92, and a numeric mixture (`forecast_guards.py`).**
  - **Why guards.** Offline, one forecaster's answer could sink a whole question after its research was paid for, or freeze the run: numeric percentiles repeated at an open lower bound, and confident multiple-choice answers with several 0% options. Neither had happened live by 2026-10-28; each would have cost a question every time it did. 0.3.5 fixes the freeze but not the failed aggregation, and moving the lock mid-season was judged riskier than the bot-side guards.
  - **Why a mixture.** Language models' numeric ranges run too narrow: nominal 90% intervals cover about 79% of outcomes at best (QuantSightBench, 2026). The package's pointwise median keeps one forecast's width; the mean CDF widens exactly where the forecasts disagree, and under the log score the plain mean never scores below their average. On the 3 resolved numeric and discrete questions with every forecaster's percentiles on 2026-10-28, it beat the median on all 3, by 0.03 nats on average, too few to show anything. A forecast whose 10th-to-90th range doesn't overlap the others' at all is left out first, so a misread unit can't cost a third of the mass; ordinary disagreement stays in. Conditional questions' numeric legs still use the package's median (the bot tournaments ask none).

  — Claude's 2026-10-28 health review (each failure reproduced offline and confirmed by two skeptic reviewers); Rob's go-ahead that day, built in [#13](https://github.com/robcam55/metaculus-bot/pull/13).
- **Research asks for the resolution source and published forecasts.** In Metaculus's Spring 2026 bot survey, checking similar questions or markets (r=+0.34) and reading sources directly (r=+0.33) were the research habits most associated with better scores, while the choice of search provider made no clear difference. The tournament's rules allow "publicly available forecasts on questions found on other platforms or on Metaculus itself". The evidence is correlational.

  — Claude's 2026-10-28 health review; Rob's go-ahead that day, built in [#13](https://github.com/robcam55/metaculus-bot/pull/13).
- **Rob's key also when the credits' balance can't be read, or there's no OpenRouter key.** Once the credits ran out (2026-10-25), an unreadable balance sent a run back onto them, uncapped, where every question would fail. Paying with Rob's key for that run costs at most one run's questions while credits remain. Without any OpenRouter key, FutureEval and MiniBench used to fall through to the Metaculus Cup's unpaced Sonnet 5 × 5, ignoring `BUDGET_TIER=pause`.
- **Prompt and research rules from the leading bots (2026-10-30).** Rob asked for every avenue that might add a few points, ready to back out if it doesn't help. Two research workflows read Metaculus's analyses and nine open-source bots, then put each candidate past two skeptics, one on the evidence and one on the code. Five changes survived, all small and cheap:
  - **What the bot's own record showed.** Its 61 scored questions (one MiniBench round, made by the older setup) scored +1.0 a question on yes/no, against +9 to +18 on discrete and multiple choice. Status-on-a-date questions averaged about −1. Four of the five worst yes/no misses were Yes outcomes forecast at 13-42%. The binary forecasts averaged 36% against 41% resolved Yes, so no Yes lean showed for the one-sided step to correct.
  - **Yes/no rules.** The criteria restatement, the trajectory question and the soft clock come from nostreambot (10th in Spring 2026, about +12.7 a question in Summer 2026). Its 815-record audit, on its older models, found announced-target questions forecast at 44% for events that happened 8% of the time, about 19 points worse; the other rules each come from a single named miss. The market rule follows the Spring 2026 survey (checking markets r=+0.34, correlational) with nostreambot's caution that thin or stale prices don't anchor. Honest expectation: 0 to +0.5 a question, about $0.02-0.05 more.
  - **Numeric width and price floor.** The bot's 9 resolved numeric ranges all held the outcome inside their 10th to 90th percentiles (nominally 80%), and nostreambot found its non-financial ranges 22% too wide; its blanket widening moved calibration the wrong way in every segment it measured. So width now follows forecastability. The one place ranges ran too narrow was a traded price (XTND, −57.6, the worst question), hence the random-walk floor for prices only.
  - **Research.** Spring 2026's winner read its resolution links; several losses here turned on a source's state (a grid connection, a tide gauge). Web fetch lets the researcher open them. The searched base rate and the venue-by-venue market lines replace a base rate from memory and discourage invented market numbers. About $0.04-0.10 a question more.
  - **Measurement.** Per-question peer scores can't tell +3 from +6 this season (about 700 questions a setup would be needed). The model tags make paired, member-by-member comparisons possible once a second model family is added, and the kinds table shows where points go. bot-review moves to 0.1.2, which reads Metaculus's archived comments in full; with 0.1.1 the members of only 13 of 61 questions survived, likely for that reason.
  - **Backing out.** `RESEARCH_WEB_FETCH=false` stops the page fetches. The prompt rules revert with this change's commit. Judge them by the kinds table and the binary calibration table, not by rank.

  — Claude's 2026-10-30 research into the leading bots; Rob's go-ahead that day in chat ("Let's move forward with these changes ... If they don't seem to meaningfully enhance prediction accuracy, we can back down").
- **New Claude models flagged daily, never switched to automatically (`model_watch.py`).** Rob asked on 2026-10-28 to evaluate each new Claude release so the bot doesn't fall behind, by auto-update or at least a notice. A notice was chosen over an automatic switch for four reasons. Anthropic's model ids are fixed versions with no "latest" alias. Haiku 5.5 and Sonnet 5.5 each rejected a setting the bot sent, which would have failed every question until fixed. A new tier can cost far more (Fable 5.1 is 2.5 times Opus 5.5). And litellm must know a new model's price before the spend figures work. The check lists models through Anthropic's Models API, which costs nothing, and opens one issue per model released after 2026-10-28. The Test Bot's trial inputs let a new model be tried on bot-testing-area questions, which the rules allow, before a switch. Quality isn't judged by backtests, because a newer model may have seen the outcomes in training. In [#13](https://github.com/robcam55/metaculus-bot/pull/13).

  — Claude's 2026-10-28 health review; Rob's go-ahead that day, built in [#13](https://github.com/robcam55/metaculus-bot/pull/13).

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
- `model_watch.yaml` — daily check for new Claude models (opens an issue; switches nothing).
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
