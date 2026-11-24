# Judging the bot's changes

Written on 2026-11-08, before the weekly review of Monday 2026-11-09 (06:00 UTC) and before
anyone read a scorecard row for a forecast made after 2026-10-31. These rules say in advance
which numbers mean keep, revert or wait, so a lucky or unlucky first week can't drive a
change. They were Claude's proposal. Rob approved writing them on 2026-11-08 ("All recommended")
and ratified them in chat at about 01:56 UTC on 2026-11-09 (8:56 PM ET on 11-08; "Merged, all
ratified"), the thresholds and rule A's start at s4 included, before any review run with the
by-setup rows.

**Amended on 2026-11-23, after data.** Rules A, C and D were amended as the
[design review](https://claude.ai/artifact/S74Ts4ARAnva1rCqqYopcZ) of 2026-11-12 proposed. That review was written from the 11-09
weekly review, before any row for a later setup had been read. Rob chose the amendments in
chat on 2026-11-23 ("All recommended"). By then the reviews of 11-16 and 11-23 had printed
rows for s3 (59 questions) and for four questions from s4 on, so under "Recording verdicts"
below these are changes made after data. They are written as proposed on 11-12 and were not
fitted to those rows, with one exception, stated in rule A: its baseline uses every s1
question on file on 11-23, not only those on file on 11-12. On the 11-23 rows the old and new
wording of each rule give the same verdict: A waits, C does not fire, D waits.

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
later against s1. The s1 baseline is frozen, by tournament, from every s1 question on the four
weekly reviews saved by 2026-11-23: FutureEval +16.0 ± 4.8 over 41 questions, MiniBench
+6.6 ± 2.1 over 118. It is weighted to the tournaments the s4+ questions came from, because
FutureEval's s1 questions scored about 9 points above MiniBench's and the mix would otherwise
move the comparison. The scorecard prints the comparison under "Rule A". The s1 row in its
by-setup table is not the baseline: it covers only the review's window. s2 and s3 are left out
of this comparison because their research lost most of its results, which would blame the
prompts for a research fault. Minimum: 40 resolved questions from s4 on. If s4+ is clearly
worse (with 40 questions, about 10 points below the baseline), look at the by-type rows to
find which rules to back out. Otherwise the verdict is "no large harm seen", not "keep": at 40
questions this comparison cannot see a harm smaller than about 9 points, so it is no evidence
that the rules help.

*Until 2026-11-23 the baseline read "s1 (129 questions, +7.9 ± 2.2 on 2026-11-02)", a 30-day
figure that the 11-09 review already printed as 143 questions at +10.1, and the last verdict
read "keep". On 11-12 the review proposed FutureEval +19.2 ± 5.8 over 30 and the same
MiniBench figure; 11 more FutureEval s1 questions have resolved since.*

**B. Yes/no calibration.** From s2 on, compare the mean forecast with the share that resolved
Yes. Minimum: 40 resolved yes/no questions. If the gap is beyond 2 standard errors of that
share (about 15 points at 40 questions), the "mildest qualifying event" step moves from Later
to Next. Otherwise it stays in Later.

**C. Numeric width.** This rule is also recorded in the README. It counts numeric and
discrete questions from s4 on, one outcome per question: inside when the outcome lies between
the mean of the forecasters' 10th percentiles and the mean of their 90th. A question that
resolved beyond a bound is left out and listed beside the count, because the review holds no
bounds to set a range against. With 20 or more questions, fewer than 75% inside is flagged on
the check-in page. A recommendation needs the count to be so low that a true rate of 80% would
give it less than 1 time in 20: 12 or fewer of 20, 19 of 30, 27 of 40, 34 of 50 (the scorecard
prints the line for the count at hand). The recommendation is then to score a widened curve
on resolved questions first, in the queued test of numeric methods, and put the result to
Rob. The widening line does not go back into the prompt on this rule alone.

*Until 2026-11-23: questions from s2 on, counted per forecaster range, and under 75% of 20
put the widening line back. That line fires about one time in five for a forecaster whose
true rate is 80%. The first setup, which had the widening line, sat at about 76% counted that
way, and the s2 and s3 forecasts were made while research was losing most of its results, the
reason rule A leaves them out. The scorecard still prints the count from s2 on beside it.*

**D. The members (s3 on).** Use the scorecard's paired log score: each model against Opus on
the questions both forecast, and the aggregate against its members. Minimum: 30 paired
questions per model.
- A model clearly worse than Opus is dropped or replaced.
- A model level with Opus stays. Being different from Opus is what helps the average.
- If the aggregate is clearly worse than Opus alone, the aggregation gets re-examined. "Opus
  alone" is what the bot would have posted from its Opus forecasts only: their median for
  yes/no, their mean per option for multiple choice. The scorecard prints it as "Posted
  aggregate − ...-only aggregate". (The row and this definition were added on 2026-11-23;
  before then no row showed this comparison.)

These rows cover yes/no and multiple-choice questions only. Number questions, about 45% of
the total, have no member rows yet. The s3 pairs come from the days when research was losing
most of its results, so the scorecard prints each row again for s4 on, and a verdict says
which rows it rests on. (Both sentences added on 2026-11-23.)

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
