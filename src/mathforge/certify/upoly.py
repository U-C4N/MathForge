"""Independent checkers over dense univariate polynomials.

They verify a producer's claim without its algorithm: a derivative against the falling factorials of math.perm, an
antiderivative by differentiating it, a division, a gcd or a square-free decomposition through product identities
decided by one Kronecker evaluation (trusted.upoly.kronecker), coprimality modulo a prime, a resultant by the
Euclidean algorithm modulo a prime larger than twice its Hadamard bound, and real root counts by their own Sturm
chains (subresultant remainders, normalized differently from the producer's). The irreducibility check reimplements
v0.7's ``polynomial_exact._check_irreducible`` and ``_rabin`` on trusted.upoly.

Rational coefficient tuples are canonical Rationals without trailing zeros, ``(Rational(0),)`` for zero. Every
function returns a bool and never trusts its arguments' types beyond what it checks.
"""

from __future__ import annotations

from math import gcd, perm

from ..model import Rational
from ..trusted import upoly as tp


def canonical(value) -> bool:
    """A Rational in lowest terms with a positive denominator."""
    if type(value) is not Rational:
        return False
    denominator = value.denominator
    return denominator == 1 or (denominator > 1 and gcd(value.numerator, denominator) == 1)


def _is_zero(coefficients) -> bool:
    return len(coefficients) == 1 and type(coefficients[0]) is Rational and not coefficients[0].numerator


def canonical_tuple(coefficients) -> bool:
    """A tuple of canonical Rationals without a trailing zero, or (0,)."""
    if type(coefficients) is not tuple or not coefficients:
        return False
    if len(coefficients) > 1 and (type(coefficients[-1]) is not Rational or not coefficients[-1].numerator):
        return False
    for value in coefficients:
        if type(value) is not Rational:
            return False
        denominator = value.denominator
        if denominator != 1 and (denominator < 1 or gcd(value.numerator, denominator) != 1):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# Calculus

def check_derivative(coefficients, n: int, result) -> bool:
    """result is the n-th derivative: result_j = c_(j+n) * (j+n)! / j!, the factor from math.perm, each a canonical
    Rational."""
    degree = len(coefficients) - 1
    if n == 0:
        return result == coefficients and canonical_tuple(result)
    if _is_zero(coefficients) or n > degree:
        return canonical_tuple(result) and _is_zero(result)
    if type(result) is not tuple or len(result) != degree - n + 1 or not result[-1].numerator:
        return False
    tp.charge(len(result))
    for value, source, k in zip(result, coefficients[n:], range(n, degree + 1)):
        if type(value) is not Rational:
            return False
        denominator = value.denominator
        if denominator == 1:
            if value.numerator * source.denominator != source.numerator * perm(k, n):
                return False
        elif (denominator < 1 or gcd(value.numerator, denominator) != 1
              or value.numerator * source.denominator != source.numerator * perm(k, n) * denominator):
            return False
    return True


def check_antiderivative(coefficients, result) -> bool:
    """result has constant term 0 and derivative coefficients: (k + 1) result_(k+1) = c_k."""
    if not canonical_tuple(result):
        return False
    if _is_zero(coefficients):
        return _is_zero(result)
    if len(result) != len(coefficients) + 1 or result[0].numerator:
        return False
    tp.charge(len(result))
    for k, source in enumerate(coefficients):
        value = result[k + 1]
        if value.numerator * (k + 1) * source.denominator != source.numerator * value.denominator:
            return False
    return True


def check_definite_integral(coefficients, lower, upper, value) -> bool:
    """value = sum of c_k (upper**(k+1) - lower**(k+1)) / (k + 1), v0.7's contributions, summed exactly."""
    if not canonical(value):
        return False
    tp.charge(2 * len(coefficients))
    total = Rational(0)
    low, high = Rational(1), Rational(1)
    for k, source in enumerate(coefficients, 1):
        low, high = low * lower, high * upper
        if source.numerator:
            total = total + source * (high - low) / k
    return total == value


# ---------------------------------------------------------------------------------------------------------------
# Product identities

def _integer(coefficients) -> tuple[tuple[int, ...], int]:
    return tp.from_rationals(coefficients)


def _degree(coefficients) -> int:
    return -1 if _is_zero(coefficients) else len(coefficients) - 1


def check_division(f, g, quotient, remainder) -> bool:
    """f = quotient * g + remainder over Q with deg remainder < deg g (g nonzero), by one Kronecker evaluation of
    F dq dg dr = Q G df dr + R df dq dg, the integer form with every denominator cleared."""
    if not (canonical_tuple(quotient) and canonical_tuple(remainder)) or _is_zero(g):
        return False
    if _degree(remainder) >= _degree(g):
        return False
    numerators_f, df = _integer(f)
    numerators_g, dg = _integer(g)
    numerators_q, dq = _integer(quotient)
    numerators_r, dr = _integer(remainder)
    left_scale, product_scale, remainder_scale = dq * dg * dr, df * dr, df * dq * dg
    bits = tp.kronecker_bits(
        tp.max_bits(numerators_f) + left_scale.bit_length(),
        tp.max_bits(numerators_q) + tp.max_bits(numerators_g) + min(len(numerators_q), len(numerators_g)).bit_length()
        + product_scale.bit_length(),
        tp.max_bits(numerators_r) + remainder_scale.bit_length())
    tp.charge(len(numerators_f) + len(numerators_q) + len(numerators_g) + len(numerators_r))
    kronecker = tp.kronecker
    return (kronecker(numerators_f, bits) * left_scale
            == kronecker(numerators_q, bits) * kronecker(numerators_g, bits) * product_scale
            + kronecker(numerators_r, bits) * remainder_scale)
