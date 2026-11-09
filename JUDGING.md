# Judging the bot's changes

Written on 2026-11-08, before the weekly review of Monday 2026-11-09 (06:00 UTC) and before
anyone read a scorecard row for a forecast made after 2026-10-31. These rules say in advance
which numbers mean keep, revert or wait, so a lucky or unlucky first week can't drive a
change. They were Claude's proposal. Rob approved writing them on 2026-11-08 ("All recommended")
and ratified them in chat that evening ("Merged, all ratified"), the thresholds and rule A's
start at s4 included, before any review with the by-setup rows ran.

## Why most comparisons can only say "wait"

A question's spot peer score varies by about 25 to 35 points from one question to the next.
At 40 questions a setup's mean is known to about ±5. Telling two setups 5 points apart would
take about 300 questions each, close to the whole season. So peer-score rows can catch only a
large harm this season. Most of the signal is in paired comparisons (forecasters scored on
the same questions) and in process checks (did research arrive, were questions missed).

## The setups

The scorecard's "By setup" rows group each forecast by the setup live when its scored run posted it,
from the dated list in `setups.py`.

| Setup | Live from (UTC) | What changed |
|---|---|---|
| s1 | before s2 | The template-era prompts, Claude Opus only |
| s2 | 2026-10-31 17:53 | The 2026-10-30 prompt and research rules ([#14](https://github.com/robcam55/metaculus-bot/pull/14)). Research lost most of its results until s4 |
| s3 | 2026-11-01 13:17 | GPT-6.1 Sol and Gemini 4 Argon join as members (`EXTRA_FORECASTERS`, [#16](https://github.com/robcam55/metaculus-bot/pull/16)) |
| s4 | 2026-11-04 17:03 | Research tools called directly ([#17](https://github.com/robcam55/metaculus-bot/pull/17)) |
| s5 | 2026-11-09 00:10 | Empty web research waits once for the next run ([#20](https://github.com/robcam55/metaculus-bot/pull/20)) |

## Rules for every change

1. A rule that fires is a recommendation to Rob. Nothing changes automatically.
2. Below a rule's minimum sample the verdict is "wait", and the setup stays as it is.
3. "Clearly worse" means worse by more than 2 standard errors of the difference. Anything
   inside that is noise, and the verdict is "keep".
4. One forecasting change per area (research, prompts, members, aggregation) per week, so
   each change gets rows of its own.
5. Leaderboard rank and tournament totals never decide a change.
6. Bugs aren't judged by statistics. A missed question, a parsing or units error, or a
   forecast that contradicts its own members gets traced and fixed when it's found.

## Rules for each change

**A. The 2026-10-30 prompt and research rules.** Compare the mean spot peer score of s4 and
later against s1 (129 questions, +7.9 ± 2.2 on 2026-11-02). s2 and s3 are left out of this
comparison because their research lost most of its results, which would blame the prompts for
a research fault. Minimum: 40 resolved questions from s4 on. If s4+ is clearly worse (with 40
questions, about 10 points below s1), look at the by-type rows to find which rules to back
out. Otherwise keep.

**B. Yes/no calibration.** From s2 on, compare the mean forecast with the share that resolved
Yes. Minimum: 40 resolved yes/no questions. If the gap is beyond 2 standard errors of that
share (about 15 points at 40 questions), the "mildest qualifying event" step moves from Later
to Next. Otherwise it stays in Later.

**C. Numeric width.** This rule is already recorded in the README. With 20 or more numeric
or discrete questions from s2 on, fewer than 75% of outcomes inside the 10th to 90th
percentile means the widening line goes back into the prompt.

**D. The members (s3 on).** Use the scorecard's paired log score: each model against Opus on
the questions both forecast, and the aggregate against its members. Minimum: 30 paired
questions per model.
- A model clearly worse than Opus is dropped or replaced.
- A model level with Opus stays. Being different from Opus is what helps the average.
- If the aggregate is clearly worse than Opus alone, the aggregation gets re-examined.

**E. Adding Fable 5.1.** Metaculus's house-bot totals don't decide this (rule 5), and there
are no paired rows for a model the bot doesn't run. Wait until rule D has its 30 paired
questions for the current members, so two member changes don't overlap. Then the choice for
Rob is a trial setup in which Fable replaces one Opus forecast, judged by rule D against the
remaining Opus forecasts. It costs about $0.16 more a question.

**F. The research tools (s4 on).** Peer score doesn't judge research tools; the research
watch does. Keep direct tools while the watch stays below its alert (6 degraded in the last
12 reports). If it trips, follow the README's backing-out steps.

**G. The empty-report wait (s5 on).** Count, from the run pages, how many questions waited and
whether their second try found web research. Minimum: 6 waits.
- Keep it while the second try finds web research at least half the time and no question is
  missed because of a wait.
- If 4 or more of the first 6 second tries also come back empty, the wait isn't helping. The
  choice for Rob is then a trial of the older tool versions (`basic:5`) on bot-testing-area.
- A question missed because of a wait is a bug (rule 6).

**H. Kinds of question.** The kinds table is descriptive. At a dozen questions per kind, a
gap under about 12 points is noise. No change is made from one kind's row alone.

## Recording verdicts

Each Monday's review gives each rule a verdict: keep, revert or wait. Each verdict comes with
the numbers behind it and goes on Rob's check-in page. Changing a rule after its data has been
read requires saying so, and why, in this file's history.
