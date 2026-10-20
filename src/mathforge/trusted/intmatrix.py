"""Integer forms of rational matrices, shared by the linear algebra producers and their independent checkers.

A rational matrix A is handled as A_int = diag(s) A, each row scaled by the least common multiple s_i of its
denominators, so that A_int is an integer matrix with the same row space, the same rank and det(A) = det(A_int) / prod(s).
Rational results are rebuilt in lowest terms with ``ratio``. Nothing here eliminates or decides anything: the
algorithms live in mathforge.kernels.linalg (producers) and mathforge.certify.linalg (checkers).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd, lcm
from operator import attrgetter

from mathforge.limits import ResourceLimitError, current_budget
from mathforge.model import Rational, trusted_rational
from mathforge.trusted.upoly import charge_gcd

_numerator = attrgetter("numerator")
_denominator = attrgetter("denominator")
ZERO = trusted_rational(0, 1)
ONE = trusted_rational(1, 1)
_SMALL = {value: trusted_rational(value, 1) for value in range(-16, 17)}  # Shared instances of small integers.
_DEFAULT_BITS = 65_536  # ComputationLimits().max_integer_bits: the bound without an active budget.


def clear_row_denominators(rows: Sequence[Sequence[Rational]]) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """The integer rows and each row's positive scale: int_row = scale * row, scale the lcm of the row's denominators.

    The entries are trusted to be canonical Rationals (positive denominators, lowest terms). A scale past the active
    budget's max_integer_bits (65,536 without one) is refused with ResourceLimitError, where v0.7's determinant
    refused the same running lcm: a row whose denominators' lengths add up to more is scaled one denominator at a
    time, each step that grows the scale charged by the size of its gcd, and refused as soon as the scale passes the
    bound, before an lcm of millions of bits is formed."""
    budget = current_budget()
    limit = _DEFAULT_BITS if budget is None else budget.limits.max_integer_bits
    int_rows = []
    scales = []
    for row in rows:
        denominators = tuple(map(_denominator, row))
        if sum(map(int.bit_length, denominators)) <= limit:
            scale = lcm(*denominators) if denominators else 1  # At most the product: within the bound.
        else:
            scale = 1
            for denominator in denominators:
                if scale % denominator:
                    charge_gcd(max(scale.bit_length(), denominator.bit_length()))
                    scale = scale // gcd(scale, denominator) * denominator
                    if scale.bit_length() > limit:
                        raise ResourceLimitError("Computation limit exceeded: integer bits")
        if scale == 1:
            int_rows.append(tuple(map(_numerator, row)))
        else:
            int_rows.append(tuple([value.numerator * (scale // value.denominator) for value in row]))
        scales.append(scale)
    return tuple(int_rows), tuple(scales)


def ratio(numerator: int, denominator: int) -> Rational:
    """numerator / denominator in lowest terms, for exact ints with denominator != 0."""
    if denominator == 1:
        small = _SMALL.get(numerator)
        return small if small is not None else trusted_rational(numerator, 1)
    divisor = gcd(numerator, denominator)
    if denominator < 0:
        divisor = -divisor
    if divisor != 1:
        numerator //= divisor
        denominator //= divisor
    if denominator == 1:
        small = _SMALL.get(numerator)
        return small if small is not None else trusted_rational(numerator, 1)
    return trusted_rational(numerator, denominator)


def integer(value: int) -> Rational:
    """The Rational of an exact int, sharing the instances of small integers."""
    small = _SMALL.get(value)
    return small if small is not None else trusted_rational(value, 1)


def rational_rows(numerators: Sequence[Sequence[int]], denominator: int) -> tuple[tuple[Rational, ...], ...]:
    """The rows numerators / denominator in lowest terms (denominator != 0)."""
    if denominator == 1:
        return tuple(tuple([integer(value) for value in row]) for row in numerators)
    return tuple(tuple([ratio(value, denominator) for value in row]) for row in numerators)
