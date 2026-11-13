"""Numeric probes: decimal digits and rational approximations of exact numbers, and how exact input is read."""

from fractions import Fraction
from math import isqrt

from . import Probe, exact, record

GROUP = "numeric"


def _digits(sign, digits):
    """The check of a digits record (sign, d), where d = floor(|x| * 10^places)."""
    return lambda v: record(v)[:2] == (sign, digits)


PROBES = (
    Probe("numeric_sqrt2_digits", GROUP, "Write sqrt(2) with 30 decimal places.",
          lambda mf: mf.numeric.digits(mf.sqrt(2), 30), _digits(1, isqrt(2 * 10**60))),
    Probe("numeric_rational_digits", GROUP, "Write 1/7 with 20 decimal places.",
          lambda mf: mf.numeric.digits(mf.Rational(1, 7), 20), _digits(1, 10**20 // 7)),
    # floor(rho * 10^10) for the real root rho of t^3 - t - 1, by integer bisection: 13247179572.
    Probe("numeric_plastic_digits", GROUP, "Write the real root of t^3 - t - 1 with 10 decimal places.",
          lambda mf: mf.numeric.digits(mf.RealAlgebraicRoot((-1, -1, 0, 1), 0), 10), _digits(1, 13247179572)),
    # 140/99 and 99/70 are the neighbours of sqrt(2) in the Farey sequence of order 100; 140/99 is the closer.
    Probe("numeric_sqrt2_fraction", GROUP, "What is the closest fraction to sqrt(2) with denominator at most 100?",
          lambda mf: mf.numeric.rational_approximation(mf.sqrt(2), 100), lambda v: exact(v) == Fraction(140, 99)),
    Probe("numeric_decimal_input", GROUP, "Is the decimal 0.1 read as exactly 1/10?",
          lambda mf: mf.parse_expression("0.1", symbols={}), lambda v: exact(v) == Fraction(1, 10)),
    Probe("numeric_division_by_zero", GROUP, "Evaluate 1/0.",
          lambda mf: mf.operations.evaluate(mf.parse_expression("1/0", symbols={}))),
    Probe("numeric_deep_nesting", GROUP, "Evaluate 1 nested in 300 pairs of parentheses.",
          lambda mf: mf.parse_expression("(" * 300 + "1" + ")" * 300, symbols={}), lambda v: exact(v) == 1),
)
