"""Input checks and value builders shared by the lean polynomial functions (not a lean module: no LEAN table).

A lean polynomial function takes a canonical input the way the structured path takes it, falls back on everything
else, runs a producer of mathforge.kernels in a fresh default budget and returns the value only when a checker of
mathforge.certify accepts it. ``coefficients`` decides what counts as canonical: an exact Polynomial whose
coefficients are Rationals in lowest terms, with integers within the default ``max_integer_bits`` (a larger input
integer makes the structured path raise resource_limit_exceeded, so the lean path falls back).
"""

from __future__ import annotations

from math import gcd

from ..model import Rational, Reals, Symbol
from ..polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial

MAX_BITS = 65_536  # ComputationLimits().max_integer_bits


def coefficients(value):
    """The coefficient tuple of a canonical Polynomial, or None."""
    checked = widths(value)
    return None if checked is None else checked[0]


def widths(value):
    """(coefficients, the bit length of the longest numerator) of a canonical Polynomial, or None."""
    if type(value) is not Polynomial:
        return None
    values = value.coefficients
    if type(values) is not tuple or not values or len(values) > MAX_POLYNOMIAL_DEGREE + 1:
        return None
    if type(value.variable) is not Symbol:
        return None
    widest = 0
    for item in values:
        if type(item) is not Rational:
            return None
        numerator, denominator = item.numerator, item.denominator
        if type(numerator) is not int or type(denominator) is not int:
            return None
        bits = numerator.bit_length()
        if bits > widest:
            if bits > MAX_BITS:
                return None
            widest = bits
        if denominator != 1 and (denominator < 1 or denominator.bit_length() > MAX_BITS
                                 or gcd(numerator, denominator) != 1):
            return None
    if len(values) > 1 and not values[-1].numerator:
        return None
    return values, widest


def real_variable(variable) -> bool:
    """An unconstrained real Symbol, as validate_variable and the record checks require."""
    return type(variable) is Symbol and variable.domain is Reals and not variable.assumptions


def same_variable(left, right) -> bool:
    return left is right or left == right


def small_int(value) -> bool:
    """An exact int (not bool) within the default max_integer_bits."""
    return type(value) is int and value.bit_length() <= MAX_BITS
