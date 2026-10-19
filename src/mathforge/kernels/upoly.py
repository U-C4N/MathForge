"""Producers over dense univariate polynomials: calculus and division over Q, gcds, subresultants, square-free
decomposition, Sturm chains and real root isolation.

Rational coefficients come and go as tuples of canonical Rationals (``(Rational(0),)`` for zero); the work is done
on integer polynomials (mathforge.trusted.upoly). Nothing here is trusted: every lean function checks a result with
mathforge.certify before returning it. Work is charged to the active budget in batches (trusted.upoly.charge).
"""

from __future__ import annotations

from math import gcd

from ..limits import ResourceLimitError, current_budget
from ..model import Rational, trusted_rational
from ..trusted import caches
from ..trusted import upoly as tp

_ZERO = trusted_rational(0, 1)


def _max_bits() -> int:
    budget = current_budget()
    return 65_536 if budget is None else budget.limits.max_integer_bits


def _too_long() -> ResourceLimitError:
    return ResourceLimitError("Computation limit exceeded: integer bits")


# ---------------------------------------------------------------------------------------------------------------
# Calculus over Q

def derivative(coefficients, n: int, widest: int | None = None):
    """The coefficients of the n-th derivative (n >= 0): c_k k (k-1) ... (k-n+1), the falling factorial updated
    from one k to the next. ``widest`` bounds the bit length of the input numerators, when the caller knows it."""
    degree = len(coefficients) - 1
    if n == 0:
        return coefficients
    if n > degree or not coefficients[-1].numerator:
        return (_ZERO,)
    tp.charge(degree + n)
    limit = _max_bits()
    falling = 1
    for factor in range(2, n + 1):
        falling *= factor
    if widest is None:
        widest = max(value.numerator.bit_length() for value in coefficients)
    # No result is longer than the longest input numerator times the largest falling factorial, below degree**n.
    checked = widest + n * degree.bit_length() > limit
    result = []
    append = result.append
    new, set_numerator, set_denominator = tp.RATIONAL_SLOTS
    k = n
    for value in coefficients[n:]:
        if k > n:
            falling = falling * k // (k - n)
        k += 1
        numerator = value.numerator
        if not numerator:
            append(_ZERO)
            continue
        denominator = value.denominator
        if denominator == 1:
            numerator *= falling
        else:
            divisor = gcd(falling, denominator)
            numerator *= falling // divisor
            denominator //= divisor
        if checked and numerator.bit_length() > limit:
            raise _too_long()
        item = new(Rational)
        set_numerator(item, numerator)
        set_denominator(item, denominator)
        append(item)
    return tuple(result)


def antiderivative(coefficients):
    """The coefficients of the antiderivative with constant term 0: c_k / (k + 1) after c_0."""
    if not coefficients[-1].numerator:
        return (_ZERO,)
    tp.charge(len(coefficients))
    limit = _max_bits()
    result = [_ZERO]
    append = result.append
    for k, value in enumerate(coefficients, 1):
        numerator = value.numerator
        if not numerator:
            append(_ZERO)
            continue
        divisor = gcd(numerator, k)
        denominator = value.denominator * (k // divisor)
        if denominator.bit_length() > limit:
            raise _too_long()
        append(trusted_rational(numerator // divisor, denominator))
    return tuple(result)


def definite_integral(coefficients, lower, upper):
    """The integral of the polynomial from lower to upper (Rationals): A(upper) - A(lower) for the antiderivative A,
    each value by integer Horner on the homogenized form."""
    limit = _max_bits()
    widest = max(lower.numerator.bit_length(), lower.denominator.bit_length(),
                 upper.numerator.bit_length(), upper.denominator.bit_length())
    if (len(coefficients) + 1) * widest > limit:
        raise _too_long()  # Powers of the bounds alone would outgrow the integer limit.
    numerators, denominator = tp.from_rationals(antiderivative(coefficients))
    value = tp.value_at(numerators, upper) - tp.value_at(numerators, lower)
    if denominator != 1:
        value = value / denominator
    if value.numerator.bit_length() > limit or value.denominator.bit_length() > limit:
        raise _too_long()
    return value


# ---------------------------------------------------------------------------------------------------------------
# Division over Q

def pseudo_divmod(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """(Q, R, e) with lc(q)**e p = Q q + R over Z and deg R < deg q, e = deg p - deg q + 1 (q nonzero, p at
    least as long as q)."""
    m = len(q)
    lead = q[-1]
    exponent = len(p) - m + 1
    remainder = list(p)
    quotient = [0] * exponent
    lower = q[:-1]
    tp.charge(exponent * len(p))
    for shift in range(len(p) - m, -1, -1):
        top = remainder.pop()
        if lead != 1:
            quotient = [lead * value for value in quotient]
            remainder = [lead * value for value in remainder]
        quotient[shift] = top
        if top:
            for index, value in enumerate(lower, shift):
                remainder[index] -= top * value
    while remainder and not remainder[-1]:
        remainder.pop()
    return tuple(quotient), tuple(remainder), exponent


def divmod_rational(f, g):
    """(quotient, remainder) coefficients of f by g over Q (g nonzero): Rationals, zero as (0,)."""
    numerators_f, denominator_f = tp.from_rationals(f)
    numerators_g, denominator_g = tp.from_rationals(g)
    if len(numerators_f) < len(numerators_g):
        return (_ZERO,), f
    quotient, remainder, exponent = pseudo_divmod(numerators_f, numerators_g)
    lead = numerators_g[-1]
    # lead**e f = Q G' / dg ... with f = F/df and g = G/dg: q = Q dg / (lead**e df), r = R / (lead**e df).
    scale = lead ** exponent * denominator_f
    numerator_scale = denominator_g
    if scale < 0:
        scale, numerator_scale, sign = -scale, -numerator_scale, -1
    else:
        sign = 1
    if numerator_scale != 1:
        quotient = tuple([value * numerator_scale for value in quotient])
    if sign < 0:
        remainder = tuple([-value for value in remainder])
    return tp.rationals(quotient, scale), tp.rationals(remainder, scale)
