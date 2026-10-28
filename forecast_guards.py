"""
Guards around forecasting-tools 0.2.92's numeric and multiple-choice forecasts.

In 0.2.92, one forecaster's answer can sink a whole question, after its research and
every forecast are paid for. The 2026-10-28 review reproduced each case offline:
- Numeric: when two of the lowest percentiles are equal at or below an open lower bound,
  the CDF stops increasing. Aggregation then raises for the whole question, or, when a bin
  is also concentrated, `_standardize_cdf` loops forever. It is synchronous, so it freezes
  every question in the run until the 60-minute timeout. forecasting-tools 0.3.5 (#374)
  bounds that loop; `install` backports the bound. `spread_repeats` keeps the trigger from
  arising in the first place.
- Multiple choice: `PredictedOptionList` raises when the stated probabilities don't sum
  to 100% within 1%. It also clamps every option to [1%, 99%], renormalises, and raises if
  any option moved by more than 5 points. So a confident answer with several 0% options
  fails, and the same parsing prompt fails the same way on every retry. `option_list`
  parses leniently instead and floors each option at 1%, which the validator then accepts
  unchanged.

Numeric forecasts are combined with `mixture`, the mean of the forecasters' CDFs, rather
than the package's pointwise median. The median keeps the spread of a single forecast,
and language models' numeric ranges run too narrow (FermiEval, QuantSightBench). Under
the log score a mixture never scores below the average of its members.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re

import numpy as np
from forecasting_tools import (
    DateQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    Percentile,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import NumericDefaults
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# The lowest probability any multiple-choice option gets: 0.2.92's validator clamps to
# 1% and rejects an answer that moves by more than 5 points, so at exactly 1% it accepts
# the answer unchanged
MIN_OPTION_PROBABILITY = 0.01
# A forecast whose probabilities sum this far from 100% is more likely misparsed than
# misadded, so it fails rather than being rescaled
MAX_SUM_ERROR = 0.25


def _standardize_cdf_bounded(self, cdf: list[float] | np.ndarray) -> list[float]:
    """
    See documentation: https://metaculus.com/api/#:~:text=CDF%20generation%20details in the
        "CDF generation details and examples" section

    Takes a cdf and returns a standardized version of it

    - assigns no mass outside of closed bounds (scales accordingly)
    - assigns at least a minimum amount of mass outside of open bounds
    - increasing by at least the minimum amount (0.01 / 200 = 0.0005)
    - caps the maximum growth to 0.2

    Note, thresholds change with different `inbound_outcome_count`s

    Boundary convention: cdf[0] is P(outcome < lower_bound) — strictly less than.
    For closed lower bounds this is forced to 0.0 after standardization; probability
    of landing exactly on lower_bound belongs in the first inbound bucket (cdf[1]).
    cdf[-1] is P(outcome <= upper_bound) — less than or equal to (standard convention).

    (Verbatim from forecasting-tools 0.3.5's NumericDistribution, #374: 0.2.92's has no
    max_scale_expansions, so a non-increasing CDF never leaves the first loop.)
    """

    lower_open = self.open_lower_bound
    upper_open = self.open_upper_bound

    # apply lower bound & enforce boundary values
    scale_lower_to = 0 if lower_open else cdf[0]
    scale_upper_to = 1.0 if upper_open else cdf[-1]
    rescaled_inbound_mass = scale_upper_to - scale_lower_to

    def apply_minimum(F: float, location: float) -> float:
        # `F` is the height of the cdf at `location` (in range [0, 1])
        # rescale
        rescaled_F = (F - scale_lower_to) / rescaled_inbound_mass
        # offset
        if lower_open and upper_open:
            return 0.988 * rescaled_F + 0.01 * location + 0.001
        elif lower_open:
            return 0.989 * rescaled_F + 0.01 * location + 0.001
        elif upper_open:
            return 0.989 * rescaled_F + 0.01 * location
        return 0.99 * rescaled_F + 0.01 * location

    for i, value in enumerate(cdf):
        cdf[i] = apply_minimum(value, i / (len(cdf) - 1))

    # apply upper bound
    # operate in PMF space
    pmf = np.diff(cdf, prepend=0, append=1)
    cap = NumericDefaults.get_max_pmf_value(len(cdf))

    def cap_pmf(scale: float) -> np.ndarray:
        return np.concatenate([pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]])

    def capped_sum(scale: float) -> float:
        return float(cap_pmf(scale).sum())

    # find the appropriate scale search space
    # An increasing CDF needs at most ~46 expansions (scale <= cap / min bin = 3800), so hitting
    # the limit means no scale works (e.g. negative bins from a non-increasing CDF)
    max_scale_expansions = 100
    lo = hi = scale = 1.0
    expansions = 0
    while capped_sum(hi) < 1.0:
        if expansions >= max_scale_expansions:
            raise ValueError(
                "Could not standardize CDF: no scale makes the capped probability mass sum to 1. "
                f"Smallest in-range bin probability is {pmf[1:-1].min():.3g} "
                "(negative means the CDF is not increasing)."
            )
        hi *= 1.2
        expansions += 1
    # hone in on scale value that makes capped sum 1
    for _ in range(100):
        scale = 0.5 * (lo + hi)
        s = capped_sum(scale)
        if s < 1.0:
            lo = scale
        else:
            hi = scale
        if s == 1.0 or (hi - lo) < 2e-5:
            break
    # apply scale and renormalize
    pmf = cap_pmf(scale)
    pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
    # back to CDF space
    cdf = np.cumsum(pmf)[:-1]

    # round to minimize floating point errors
    cdf = np.round(cdf, 10)
    return cdf.tolist()


def install() -> None:
    """Bounds 0.2.x's CDF standardisation, as 0.3.5 does. Later versions carry it already."""
    version = importlib.metadata.version("forecasting-tools")
    if version.startswith("0.2."):
        NumericDistribution._standardize_cdf = _standardize_cdf_bounded  # type: ignore[method-assign]


def spread_repeats(
    percentiles: list[Percentile], question: NumericQuestion | DateQuestion
) -> list[Percentile]:
    """The percentiles in order, with any value repeated at or below an open lower bound
    nudged up by a millionth of the range at each repeat. 0.2.92 moves such repeats a
    hair below the bound, and its CDF then isn't increasing (the module's first case).
    Repeats inside the range, at the upper bound or at a closed bound are left alone: the
    package handles those, and a discrete question's repeated integers stay in their bin."""
    ordered = sorted(percentiles, key=lambda p: p.percentile)
    if not question.open_lower_bound:
        return ordered
    if isinstance(question, DateQuestion):
        lower, upper = question.lower_bound.timestamp(), question.upper_bound.timestamp()
    else:
        lower, upper = question.lower_bound, question.upper_bound
    step = 1e-6 * (upper - lower)
    spread: list[Percentile] = []
    for p in ordered:
        value = p.value
        if spread and value <= lower and value <= spread[-1].value:
            value = spread[-1].value + step
        spread.append(Percentile(percentile=p.percentile, value=value))
    return spread


def _median_location(prediction: NumericDistribution) -> float:
    """Where a forecast's CDF crosses 50%."""
    cdf = prediction.get_cdf()
    return float(
        np.interp(0.5, [p.percentile for p in cdf], [p.value for p in cdf])
    )


def mixture(
    predictions: list[NumericDistribution], question: NumericQuestion | DateQuestion
) -> NumericDistribution:
    """The forecasts' mean CDF, the mixture of their distributions. A forecast whose
    median falls outside the middle 80% of the pointwise-median CDF (the same band as the
    forecasters' own 10th to 90th percentiles; with an open bound the package keeps 5% of
    the mass beyond it, so a 90% band would span the whole range) is left out first, so
    one misread unit can't put a third of the mass in the wrong place, as it would in a
    plain mixture (the median ignores it). With fewer than three forecasts nothing is left
    out, since there's no majority to judge by."""
    cdfs = [prediction.get_cdf() for prediction in predictions]
    x_axis = [p.value for p in cdfs[0]]
    heights = np.array([[p.percentile for p in cdf] for cdf in cdfs])
    kept = list(range(len(predictions)))
    if len(predictions) >= 3:
        median = np.median(heights, axis=0)
        low, high = np.interp([0.1, 0.9], median, x_axis)
        inside = [i for i, p in enumerate(predictions) if low <= _median_location(p) <= high]
        if len(inside) >= 2 and len(inside) < len(predictions):
            logger.info(
                f"Numeric mixture: left out {len(predictions) - len(inside)} forecast(s) far"
                f" from the others for {question.page_url}"
            )
            kept = inside
    mean = heights[kept].mean(axis=0)
    return NumericDistribution.from_question(
        [Percentile(value=x, percentile=h) for x, h in zip(x_axis, mean.tolist())], question
    )


class OptionGuess(BaseModel):
    """One option's probability as a forecaster wrote it, before any checks."""

    option_name: str
    probability: float


def _name_key(name: str) -> str:
    name = re.sub(r"^\s*option[\s_:-]*(?=\S)", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip(" :.*\"'").casefold()


def _normalize_with_minimum(probabilities: list[float], minimum: float) -> list[float]:
    """Sums to 1 with every option at least `minimum`: options under it are pinned there
    and the rest give up mass in proportion (forecasting-tools 0.3.5's rule)."""
    total = sum(probabilities)
    unadjusted = [p / total for p in probabilities]
    normalized, pinned = unadjusted, set()
    while True:
        newly = {i for i, p in enumerate(normalized) if p < minimum and i not in pinned}
        if not newly:
            break
        pinned |= newly
        free_mass = 1 - minimum * len(pinned)
        free_total = sum(p for i, p in enumerate(unadjusted) if i not in pinned)
        normalized = [
            minimum if i in pinned else p * free_mass / free_total
            for i, p in enumerate(unadjusted)
        ]
    largest = normalized.index(max(normalized))
    normalized[largest] = 1.0 - sum(p for i, p in enumerate(normalized) if i != largest)
    return normalized


def option_list(guesses: list[OptionGuess], options: list[str]) -> PredictedOptionList:
    """A forecaster's parsed answer as the package's PredictedOptionList, every option in
    the question's own order and wording. Names match exactly, or else ignoring case,
    spacing and an "Option" prefix. Percentages are read as such, and the sum is
    renormalised. A missing option, or a sum far from 100%, fails this one forecast: the
    question goes ahead on the others."""
    by_key: dict[str, float] = {}
    for guess in guesses:
        by_key.setdefault(guess.option_name, guess.probability)
        by_key.setdefault(_name_key(guess.option_name), guess.probability)
    values: list[float] = []
    for option in options:
        value = by_key.get(option, by_key.get(_name_key(option)))
        if value is None:
            raise ValueError(f"The forecast gives no probability for option {option!r}")
        values.append(max(float(value), 0.0))
    if max(values) > 1.0:  # written as percentages
        values = [v / 100 for v in values]
    total = sum(values)
    if abs(total - 1) > MAX_SUM_ERROR:
        raise ValueError(f"The forecast's probabilities sum to {total:.0%}, not 100%")
    if len(options) * MIN_OPTION_PROBABILITY >= 1:
        floored = [1 / len(options)] * len(options)
    else:
        floored = _normalize_with_minimum(values, MIN_OPTION_PROBABILITY)
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=option, probability=p)
            for option, p in zip(options, floored)
        ]
    )
