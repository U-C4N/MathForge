"""Lean value functions of the descriptive statistics: mean, median, variance, stddev and covariance.

Each function takes the public function's positional parameters and answers for the samples it handles, with the value
that ``mathforge.operations.<operation>(...).unwrap()`` returns; anything else is FALLBACK, for the structured path.

**What is handled.** A tuple or a list of 1 to ``SAMPLE_CAP`` entries, each an exact ``int`` or a canonical
``Rational`` whose numerator and denominator have at most ``ELEMENT_BITS`` bits and whose sample has a least common
denominator of at most ``COMMON_BITS`` bits; for the operations with a ddof, an exact int below the sample size. A
call made while a computation budget is active is FALLBACK, and so is anything else: a one-shot iterator (nothing
iterates it), a set or a range (the structured path's records refuse them), a float, a bool, a forged Rational, an
empty sample, a ddof that v0.7 rejects. These bounds sit far inside the default limits: with them no structured
intermediate (a deviation, a square, a product of two sums) gets near ``max_integer_bits``, and the records of the
structured path stay far below ``max_nodes``, ``max_evidence_items`` and ``max_work``. Where the lean path answers,
the structured path therefore answers too, with the same value (the median's structured check no longer costs
quadratic work either).

**The producer** works on integers: the sample is scaled to the least common denominator, so that every entry is an
integer over one denominator; sums, sorting and the squared deviations from the mean are integer arithmetic, and the
result is built with ``trusted_rational`` from a reduced pair.

**The check** is v0.7's criterion for the operation, never the producer's arithmetic: the mean is re-summed in checked
Rational arithmetic; the variance, the covariance and the standard deviation are compared with the one-pass formula
``(sum(a*b) - sum(a)*sum(b)/n) / (n - ddof)`` that v0.7's checker evaluates (the producer works with the centred sums
of squares); the median is accounted for by counting how many entries lie below, equal to and above it
(``statistics.median_certificate``), without trusting the order the producer sorted. A value is returned only when
its check answers True.
"""

from __future__ import annotations

from math import gcd, isqrt, lcm
from operator import mul

from ..limits import ComputationLimits, current_budget
from ..model import Rational, Sqrt, trusted_rational
from ..statistics import median_certificate
from . import FALLBACK

_LIMITS = ComputationLimits()
SAMPLE_CAP = min(4_096, _LIMITS.max_evidence_items // 20)  # Entries per sample: the records stay within the limits.
ELEMENT_BITS = min(8_192, _LIMITS.max_integer_bits // 8)  # Bits of an entry's numerator and denominator.
COMMON_BITS = ELEMENT_BITS  # Bits of the least common denominator of a sample.

_ZERO = trusted_rational(0, 1)
_EXACT_INT = {int}


def _prepare(data):
    """(items, nums, dens) of a sample that the lean path handles, else None.

    ``items`` is the sample as a tuple of its ints and Rationals; ``nums`` and ``dens`` are the numerators and the
    denominators, lists of exact ints with positive denominators in lowest terms, as ``probability._rational``
    requires of every entry.
    """
    kind = type(data)
    if kind is not tuple and kind is not list:
        return None
    items = tuple(data)
    if not 0 < len(items) <= SAMPLE_CAP:
        return None
    nums, dens = [], []
    push_num, push_den = nums.append, dens.append
    fraction = False
    for item in items:
        kind = type(item)
        if kind is Rational:
            push_num(item.numerator)
            push_den(item.denominator)
            fraction = True
        elif kind is int:
            push_num(item)
            push_den(1)
        else:
            return None
    if fraction and (set(map(type, nums)) != _EXACT_INT or set(map(type, dens)) != _EXACT_INT
                     or min(dens) < 1 or sum(map(gcd, nums, dens)) != len(items)):
        return None  # A Rational can be forged (trusted_rational, unpickling): v0.7 refuses a noncanonical one.
    if max(map(int.bit_length, nums)) > ELEMENT_BITS or max(dens).bit_length() > ELEMENT_BITS:
        return None
    return items, nums, dens


def _scale(nums, dens):
    """(common, scaled): the least common denominator of dens, and nums[i] * common // dens[i] for each entry.

    None when the common denominator has more than COMMON_BITS bits. ``scaled`` is ``nums`` itself when the common
    denominator is 1: it is never modified.
    """
    if sum(map(int.bit_length, dens)) <= COMMON_BITS:  # Their product, so their lcm, then fits.
        common = lcm(*dens)
    else:
        common = 1
        for denominator in dens:
            if common % denominator:
                common = common // gcd(common, denominator) * denominator
                if common.bit_length() > COMMON_BITS:
                    return None
    if common == 1:
        return 1, nums
    return common, [num * (common // den) for num, den in zip(nums, dens)]


def _load(data):
    """(items, nums, dens, common, scaled) of a sample that the lean path handles, else None."""
    sample = _prepare(data)
    if sample is None:
        return None
    items, nums, dens = sample
    scaled = _scale(nums, dens)
    if scaled is None:
        return None
    return items, nums, dens, scaled[0], scaled[1]


def _fraction(numerator, denominator):
    """numerator/denominator in lowest terms: a Rational, built without validation. The denominator is positive."""
    divisor = gcd(numerator, denominator)
    return trusted_rational(numerator // divisor, denominator // divisor)


def _centered(scaled, size):
    """The deviations from the mean of integers over one denominator, each times size: size * s - sum(scaled)."""
    total = sum(scaled)
    return [size * entry - total for entry in scaled]


# ---------------------------------------------------------------------------------------------------------------
# The checks: v0.7's criteria, on the sample's own Rationals and ints, in checked Rational arithmetic.

def _check_mean(items, size, value):
    """The mean is the sum of the sample divided by its size."""
    total = _ZERO
    for item in items:
        total = total + item
    return value == total / size


def _target(items, size, ddof):
    """v0.7's one-pass variance: (sum(a*a) - sum(a)**2 / size) / (size - ddof)."""
    total = products = _ZERO
    for item in items:
        total = total + item
        products = products + item * item
    return (products - total * total / size) / (size - ddof)


def _check_variance(items, size, ddof, value):
    return value == _target(items, size, ddof)


def _check_stddev(items, size, ddof, value):
    """A Rational is the nonnegative square root of the variance, a Sqrt has the variance as its radicand.

    One step beyond v0.7's check: a Sqrt must be in the form that ``model.sqrt`` gives, so its radicand is not the
    square of a rational (v0.7's check also accepts such a Sqrt, which no producer returns).
    """
    target = _target(items, size, ddof)
    if type(value) is Rational:
        return value >= _ZERO and value * value == target
    if type(value) is Sqrt:
        radicand = value.radicand
        if type(radicand) is not Rational or radicand != target or target < _ZERO:
            return False
        numerator, denominator = isqrt(radicand.numerator), isqrt(radicand.denominator)
        return numerator * numerator != radicand.numerator or denominator * denominator != radicand.denominator
    return False


def _check_covariance(items_x, items_y, size, ddof, value):
    """v0.7's one-pass covariance: (sum(a*b) - sum(a) * sum(b) / size) / (size - ddof)."""
    total_x = total_y = products = _ZERO
    for a, b in zip(items_x, items_y):
        total_x = total_x + a
        total_y = total_y + b
        products = products + a * b
    return value == (products - total_x * total_y / size) / (size - ddof)


# ---------------------------------------------------------------------------------------------------------------
# The lean functions.

def mean(data):
    try:
        if current_budget() is not None:
            return FALLBACK
        loaded = _load(data)
        if loaded is None:
            return FALLBACK
        items, nums, dens, common, scaled = loaded
        size = len(items)
        value = _fraction(sum(scaled), size * common)
        return value if _check_mean(items, size, value) is True else FALLBACK
    except Exception:
        return FALLBACK


def median(data):
    try:
        if current_budget() is not None:
            return FALLBACK
        loaded = _load(data)
        if loaded is None:
            return FALLBACK
        items, nums, dens, common, scaled = loaded
        size = len(items)
        ordered = sorted(scaled)
        middle = size // 2
        if size % 2:
            value = _fraction(ordered[middle], common)
        else:
            value = _fraction(ordered[middle - 1] + ordered[middle], 2 * common)
        accepted = median_certificate(nums, dens, value.numerator, value.denominator)
        return value if accepted is True else FALLBACK
    except Exception:
        return FALLBACK


def _variance_value(loaded, ddof):
    """The variance of a loaded sample as a Rational: the squared deviations, centred, over their common
    denominator."""
    items, nums, dens, common, scaled = loaded
    size = len(items)
    deviations = _centered(scaled, size)
    return _fraction(sum(map(mul, deviations, deviations)), size * size * common * common * (size - ddof))


def variance(data, ddof=0):
    try:
        if current_budget() is not None or type(ddof) is not int:
            return FALLBACK
        loaded = _load(data)
        if loaded is None or not 0 <= ddof < len(loaded[0]):
            return FALLBACK
        value = _variance_value(loaded, ddof)
        return value if _check_variance(loaded[0], len(loaded[0]), ddof, value) is True else FALLBACK
    except Exception:
        return FALLBACK


def stddev(data, ddof=0):
    try:
        if current_budget() is not None or type(ddof) is not int:
            return FALLBACK
        loaded = _load(data)
        if loaded is None or not 0 <= ddof < len(loaded[0]):
            return FALLBACK
        spread = _variance_value(loaded, ddof)
        numerator, denominator = isqrt(spread.numerator), isqrt(spread.denominator)
        if numerator * numerator == spread.numerator and denominator * denominator == spread.denominator:
            value = trusted_rational(numerator, denominator)  # Coprime, as the radicand's parts are.
        else:
            value = Sqrt(spread)
        return value if _check_stddev(loaded[0], len(loaded[0]), ddof, value) is True else FALLBACK
    except Exception:
        return FALLBACK


def covariance(x, y, ddof=0):
    try:
        if current_budget() is not None or type(ddof) is not int:
            return FALLBACK
        first, second = _load(x), _load(y)
        if first is None or second is None:
            return FALLBACK
        size = len(first[0])
        if len(second[0]) != size or not 0 <= ddof < size:
            return FALLBACK
        deviations_x, deviations_y = _centered(first[4], size), _centered(second[4], size)
        value = _fraction(sum(map(mul, deviations_x, deviations_y)),
                          size * size * first[3] * second[3] * (size - ddof))
        return value if _check_covariance(first[0], second[0], size, ddof, value) is True else FALLBACK
    except Exception:
        return FALLBACK


LEAN = {"mean": mean, "median": median, "variance": variance, "stddev": stddev, "covariance": covariance}
