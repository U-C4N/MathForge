"""Lean value function of the discrete uniform distribution.

``discrete_uniform`` answers for a tuple or a list of 1 to ``SAMPLE_CAP`` distinct outcomes, each an exact int or a
canonical Rational (the bounds of ``stats_descriptive``); anything else is FALLBACK, and so is a call made while a
computation budget is active.

The producer scales the outcomes to their least common denominator, sorts the integers and builds the distribution
with ``probability._trusted_distribution``: the outcomes in increasing order, each with the weight 1/n. The check is
v0.7's criterion for the distribution, read off the value itself: its outcomes are strictly increasing and are exactly
the outcomes asked for (as pairs of integers, so that nothing of the producer's scaling is reused), and every weight
equals 1/n, which also makes the weights sum to one. A value is returned only when the check answers True.
"""

from __future__ import annotations

from operator import lt

from ..limits import current_budget
from ..model import Rational, trusted_rational
from ..probability import FiniteDistribution, _trusted_distribution
from . import FALLBACK
from .stats_descriptive import _load


def _check_uniform(value, nums, dens, size):
    """Whether value is the uniform distribution on the distinct outcomes nums[i]/dens[i]."""
    if type(value) is not FiniteDistribution:
        return False
    outcomes, weights = value.outcomes, value.probabilities
    if type(outcomes) is not tuple or type(weights) is not tuple or len(outcomes) != size or len(weights) != size:
        return False
    if weights.count(trusted_rational(1, size)) != size:  # Equal to 1/size, so positive; the sum is then one.
        return False
    if not all(map(lt, outcomes, outcomes[1:])):  # Strictly increasing: sorted and distinct.
        return False
    return {(outcome.numerator, outcome.denominator) for outcome in outcomes} == set(zip(nums, dens))


def discrete_uniform(outcomes):
    try:
        if current_budget() is not None:
            return FALLBACK
        loaded = _load(outcomes)
        if loaded is None:
            return FALLBACK
        items, nums, dens, common, scaled = loaded
        size = len(items)
        by_key = dict(zip(scaled, [item if type(item) is Rational else trusted_rational(item, 1) for item in items]))
        if len(by_key) != size:  # Two outcomes are equal: v0.7 refuses them.
            return FALLBACK
        value = _trusted_distribution(tuple([by_key[key] for key in sorted(by_key)]),
                                      (trusted_rational(1, size),) * size)
        return value if _check_uniform(value, nums, dens, size) is True else FALLBACK
    except Exception:
        return FALLBACK


LEAN = {"discrete_uniform": discrete_uniform}
