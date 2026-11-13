"""Number theory probes: gcds, factorizations, primes, modular arithmetic, symbols and continued fractions."""

from fractions import Fraction

from . import Probe

GROUP = "ntheory"
# 19 decimal digits of sqrt(2): the rational that v0.8's continued_fraction takes in place of sqrt(2) itself.
SQRT2_DECIMAL = Fraction(14142135623730950488, 10**19)


def _factorization(sign, *factors):
    return lambda v: v.sign == sign and tuple(map(tuple, v.factors)) == factors


def _folds_back(value):
    """Whether the continued fraction's terms evaluate back to SQRT2_DECIMAL (and start like sqrt(2)'s)."""
    terms = tuple(value.terms)
    total = Fraction(terms[-1])
    for term in reversed(terms[:-1]):
        total = term + 1 / total
    return terms[:4] == (1, 2, 2, 2) and total == SQRT2_DECIMAL


def _crt(value):
    return (value.residue, value.modulus) == (23, 105)


PROBES = (
    Probe("ntheory_gcd", GROUP, "What is gcd(1071, 462)?", lambda mf: mf.integer_gcd(1071, 462), lambda v: v == 21),
    Probe("ntheory_extended_gcd", GROUP, "Find g = gcd(240, 46) with Bezout coefficients s, t: 240 s + 46 t = g.",
          lambda mf: mf.integer_xgcd(240, 46), lambda v: v[0] == 2 and 240 * v[1] + 46 * v[2] == 2),
    Probe("ntheory_factor_mersenne_59", GROUP, "Factor 2^59 - 1.", lambda mf: mf.factor_integer(2**59 - 1),
          _factorization(1, (179951, 1), (3203431780337, 1))),
    Probe("ntheory_factor_two_primes", GROUP, "Factor 1000000007 * 998244353.",
          lambda mf: mf.factor_integer(1000000007 * 998244353),
          _factorization(1, (998244353, 1), (1000000007, 1))),
    Probe("ntheory_is_prime_mersenne_61", GROUP, "Is 2^61 - 1 prime?", lambda mf: mf.is_prime(2**61 - 1),
          lambda v: v is True),
    Probe("ntheory_mod_inverse", GROUP, "What is the inverse of 3 modulo 11?", lambda mf: mf.mod_inverse(3, 11),
          lambda v: v == 4),
    Probe("ntheory_mod_inverse_missing", GROUP, "What is the inverse of 6 modulo 9 (there is none)?",
          lambda mf: mf.mod_inverse(6, 9)),
    Probe("ntheory_powmod", GROUP, "What is 7^123456 modulo 1000003?",
          lambda mf: mf.ntheory.powmod(7, 123456, 1000003), lambda v: v == pow(7, 123456, 1000003)),
    Probe("ntheory_crt", GROUP, "Solve x = 2 (mod 3), x = 3 (mod 5), x = 2 (mod 7).",
          lambda mf: mf.crt(((2, 3), (3, 5), (2, 7))), _crt),
    Probe("ntheory_next_prime", GROUP, "What is the smallest prime above 10^9?",
          lambda mf: mf.ntheory.next_prime(10**9), lambda v: v == 1000000007),
    Probe("ntheory_prev_prime", GROUP, "What is the largest prime below 10^9?",
          lambda mf: mf.ntheory.prev_prime(10**9), lambda v: v == 999999937),
    Probe("ntheory_legendre_symbol", GROUP, "What is the Legendre symbol (2/7)?",
          lambda mf: mf.ntheory.legendre_symbol(2, 7), lambda v: v == 1),
    Probe("ntheory_jacobi_symbol", GROUP, "What is the Jacobi symbol (3/9)?",
          lambda mf: mf.ntheory.jacobi_symbol(3, 9), lambda v: v == 0),
    Probe("ntheory_multiplicative_order", GROUP, "What is the multiplicative order of 2 modulo 101?",
          lambda mf: mf.ntheory.multiplicative_order(2, 101), lambda v: v == 100),
    Probe("ntheory_primitive_root", GROUP, "What is the smallest primitive root modulo 101?",
          lambda mf: mf.ntheory.primitive_root(101), lambda v: v == 2),
    Probe("ntheory_sqrt_mod", GROUP, "Which x satisfy x^2 = 10 (mod 13)?",
          lambda mf: mf.ntheory.sqrt_mod(10, 13), lambda v: tuple(v) == (6, 7)),
    Probe("ntheory_linear_diophantine", GROUP, "Find all integer solutions of 6x + 9y = 15.",
          lambda mf: mf.ntheory.linear_diophantine(6, 9, 15),
          lambda v: (v.x, v.y, v.dx, v.dy) == (1, 1, 3, -2)),  # (1 + 3t, 1 - 2t), canonical: 0 <= x < dx.
    Probe("ntheory_continued_fraction_sqrt2", GROUP, "What is the continued fraction of sqrt(2)?",
          lambda mf: mf.ntheory.quadratic_continued_fraction(0, 1, 2, 1),
          lambda v: (tuple(v.preperiod), tuple(v.period)) == ((1,), (2,))),
    Probe("ntheory_continued_fraction_decimal", GROUP,
          "What is the continued fraction of 1.4142135623730950488, 19 digits of sqrt(2)?",
          lambda mf: mf.ntheory.continued_fraction(mf.Rational(SQRT2_DECIMAL.numerator, SQRT2_DECIMAL.denominator)),
          _folds_back),
    Probe("ntheory_continued_fraction_rational", GROUP, "What is the continued fraction of 415/93?",
          lambda mf: mf.ntheory.continued_fraction(mf.Rational(415, 93)),
          lambda v: tuple(v.terms) == (4, 2, 6, 7)),
    Probe("ntheory_divisors", GROUP, "List the divisors of 360.", lambda mf: mf.divisors(360),
          lambda v: tuple(v) == tuple(d for d in range(1, 361) if 360 % d == 0)),
    Probe("ntheory_totient", GROUP, "What is Euler's totient of 360?", lambda mf: mf.totient(360),
          lambda v: v == 96),
)
