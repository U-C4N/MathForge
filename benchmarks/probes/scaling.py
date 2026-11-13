"""Scaling probes: requests just past a resource limit of v0.8, ported from the performance-cliff measurements.

Each one asks for a size that v0.8 refuses with resource_limit_exceeded (its work, integer-size, node or matrix-entry
budget) and none depends on timing: the budgets count steps. The inputs are fixed literals, and every check is cheap
Python arithmetic, so that a probe turns ok by itself once MathForge reaches the size.
"""

from fractions import Fraction
from math import comb, factorial

from . import Probe, coefficients, exact, exact_tuple, symbols
from .calculus import sin_times_exp_coefficients
from .discrete import fibonacci

GROUP = "scaling"
# 2^127 + 45 is prime (Pocklington): 2^127 + 44 = 2^2 * 191 * 719 * 1866439 * 269787574153 * 615108114994501, every
# factor proved prime by deterministic Miller-Rabin, and the base 2 passes the criterion for every prime factor.
PRIME_128_BITS = 2**127 + 45
SEMIPRIME_80_BITS = (2**40 + 15) * (2**40 - 87)  # Both factors are prime.
HIGHLY_COMPOSITE = 2**30 * 3**20 * 5**10 * 7**8  # 31 * 21 * 11 * 9 = 64449 divisors.


def _divide(dividend, divisor):
    """The exact quotient of two integer polynomials (coefficients, constant term first) by a monic divisor."""
    remainder, quotient = list(dividend), [0] * (len(dividend) - len(divisor) + 1)
    for i in range(len(quotient) - 1, -1, -1):
        quotient[i] = c = remainder[i + len(divisor) - 1]
        for j, coefficient in enumerate(divisor):
            remainder[i + j] -= c * coefficient
    if any(remainder):
        raise ArithmeticError("the division is not exact")
    return quotient


def _cyclotomic(n):
    """The cyclotomic polynomials Phi_d for the divisors d of n: Phi_d = (x^d - 1) / prod of Phi_e, e | d, e < d."""
    found = {}
    for d in range(1, n + 1):
        if n % d == 0:
            quotient = [-1] + [0] * (d - 1) + [1]
            for e, phi in found.items():
                if d % e == 0:
                    quotient = _divide(quotient, phi)
            found[d] = tuple(quotient)
    return found


def _cyclotomic_factors(value):
    """x^96 - 1 is the product of the cyclotomic polynomials Phi_d of the 12 divisors d of 96, each irreducible."""
    expected = sorted((phi, 1) for phi in _cyclotomic(96).values())
    return exact(value.coefficient) == 1 and sorted(
        (coefficients(item.polynomial), item.multiplicity) for item in value.factors) == expected


def _second_difference(mf, n):
    return mf.Matrix(tuple(tuple(2 if i == j else -1 if abs(i - j) == 1 else 0 for j in range(n)) for i in range(n)))


def _factor_x96_minus_1(mf):
    x, = symbols(mf, "x")
    return mf.factor(mf.Polynomial.from_expression(x**96 - 1, x))


def _taylor(mf):
    x, = symbols(mf, "x")
    return mf.taylor(mf.parse_function("sin(x)*exp(x)", symbols={"x": x}), x, 0, 12)


def _real_roots(mf):
    x, = symbols(mf, "x")
    return mf.real_roots(x**500 - x - 1, x)


def _two_real_roots(value):
    """x^500 - x - 1 has exactly two real roots: one sign change for x > 0 and one for x < 0 (Descartes)."""
    polynomial = (-1, -1) + (0,) * 498 + (1,)
    return sorted((tuple(item.root.integer_coefficients), item.root.real_index, item.multiplicity)
                  for item in value) == [(polynomial, 0, 1), (polynomial, 1, 1)]


def _all_divisors(value):
    """64449 increasing divisors of HIGHLY_COMPOSITE are all of them."""
    value = tuple(value)
    return (len(value) == 64449 and value[0] == 1 and value[-1] == HIGHLY_COMPOSITE
            and all(a < b for a, b in zip(value, value[1:])) and all(HIGHLY_COMPOSITE % d == 0 for d in value))


PROBES = (
    Probe("scaling_is_prime_128_bits", GROUP, "Is 2^127 + 45 prime?", lambda mf: mf.is_prime(PRIME_128_BITS),
          lambda v: v is True),
    Probe("scaling_is_prime_mersenne_521", GROUP, "Is the Mersenne number 2^521 - 1 prime?",
          lambda mf: mf.is_prime(2**521 - 1), lambda v: v is True),
    Probe("scaling_factor_integer_80_bits", GROUP, "Factor the 80-bit number (2^40 + 15)(2^40 - 87).",
          lambda mf: mf.factor_integer(SEMIPRIME_80_BITS),
          lambda v: v.sign == 1 and tuple(map(tuple, v.factors)) == ((2**40 - 87, 1), (2**40 + 15, 1))),
    Probe("scaling_factor_x96_minus_1", GROUP, "Factor x^96 - 1 over the rationals.", _factor_x96_minus_1,
          _cyclotomic_factors),
    Probe("scaling_taylor_order_12", GROUP, "What is the Taylor polynomial of sin(x) exp(x) at 0 of order 12?", _taylor,
          lambda v: exact_tuple(v.coefficients) == sin_times_exp_coefficients(12)),
    Probe("scaling_det_65x65", GROUP, "What is the determinant of the 65 x 65 second-difference matrix?",
          lambda mf: mf.det(_second_difference(mf, 65)), lambda v: exact(v) == 66),
    Probe("scaling_fibonacci_100000", GROUP, "What is the 100000th Fibonacci number?",
          lambda mf: mf.fibonacci(100000), lambda v: v == fibonacci(100000)),
    Probe("scaling_factorial_10000", GROUP, "What is 10000!?", lambda mf: mf.factorial(10000),
          lambda v: v == factorial(10000)),
    Probe("scaling_binomial_100000", GROUP, "What is the binomial coefficient C(100000, 50000)?",
          lambda mf: mf.binomial(100000, 50000), lambda v: v == comb(100000, 50000)),
    Probe("scaling_real_roots_degree_500", GROUP, "Isolate the real roots of x^500 - x - 1.", _real_roots,
          _two_real_roots),
    Probe("scaling_median_100000", GROUP, "What is the median of the integers 0 to 99999?",
          lambda mf: mf.median(tuple(range(100000))), lambda v: exact(v) == Fraction(99999, 2)),
    Probe("scaling_divisors_64449", GROUP, "List the divisors of 2^30 3^20 5^10 7^8.",
          lambda mf: mf.divisors(HIGHLY_COMPOSITE), _all_divisors),
    Probe("scaling_gcd_70000_bits", GROUP, "What is gcd(2^70000 + 1, 2^69999 + 3)?",
          lambda mf: mf.integer_gcd(2**70000 + 1, 2**69999 + 3), lambda v: v == 1),
)
