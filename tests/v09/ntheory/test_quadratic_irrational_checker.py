"""Tests of the independent checker of mathforge.areas.ntheory.quadratic_irrational alone, written before the
family's producer exists: every claim is built here, and the checker answers it as the verifier asks it
(support.verdict, in a fresh budget of the default limits unless a test passes others).

The true claims come from short references of this module, never from a producer: the continued fraction of a
quadratic irrational by exact interval bisection (the floor of a + b*sqrt(d) found with exact sign tests, the complete
quotients kept in Q(sqrt(d)) as pairs of Fractions until one repeats), the convergents of a rational by folding each
prefix of its Euclidean terms from the right, and the fundamental solution of Pell's equation by brute force over y,
or from the period of sqrt(d) where y is out of reach, the brute force confirming that reference wherever it reaches.

A claim beyond the checker's caps (more than 99,000 terms, or term bit lengths adding up to more than
max_integer_bits) raises ResourceLimitError, as the card's caps allow; every other corrupted claim answers False.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal, localcontext
from fractions import Fraction
from math import isqrt
import random
import time

import mathforge as mf
from mathforge.areas.ntheory import quadratic_irrational as family
from mathforge.areas.ntheory.quadratic_irrational import PeriodicContinuedFraction as PCF
from mathforge.certify.ntheory.quadratic_irrational import MAX_TERMS
from mathforge.limits import ComputationLimits, ResourceLimitError

from .. import support

Q = mf.Rational
EXPAND, CONVERGENTS, PELL = "quadratic_continued_fraction", "convergents", "pell_fundamental"
FAST = 5.0  # Seconds: the bound for every oversized claim, far above what each one takes.
LONG_PERIOD_D = 1726  # sqrt(1726) has the longest period below 2,000: 88 terms.
# F(n+1)/F(n) has n - 1 convergents. The verifier's evidence limit admits about 16,600 at the default limits (each
# Rational counts three nodes, and the value is counted twice); building more than a few thousand Rationals of that
# size takes the test itself seconds, so the long cases stop here.
LONG_FIBONACCI = 4_000
NON_SQUARES = tuple(d for d in range(2, 200) if isqrt(d) * isqrt(d) != d)

# The known values of the card: (p, q, d, r) with (preperiod, period); convergents; d with (x, y).
KNOWN_EXPANSIONS = (
    ((0, 1, 2, 1), (1,), (2,)),
    ((0, 1, 3, 1), (1,), (1, 2)),
    ((1, 1, 5, 2), (), (1,)),
    ((0, 1, 7, 1), (2,), (1, 1, 1, 4)),
    ((3, 1, 2, 1), (4,), (2,)),
    ((0, -1, 2, 1), (-2, 1, 1), (2,)),
)
KNOWN_CONVERGENTS = (
    (Q(415, 93), (Q(4), Q(9, 2), Q(58, 13), Q(415, 93))),
    (5, (Q(5),)),
    (Q(-7, 3), (Q(-3), Q(-2), Q(-7, 3))),
)
KNOWN_PELL = ((2, (3, 2)), (3, (2, 1)), (5, (9, 4)), (7, (8, 3)), (13, (649, 180)), (61, (1766319049, 226153980)))


def expansion_args(p: int, q: int, d: int, r: int) -> dict[str, object]:
    return {"p": p, "q": q, "d": d, "r": r}


SQRT2 = expansion_args(0, 1, 2, 1)
SQRT2_VALUE = ((1,), (2,))  # sqrt(2) = [1; (2)]


def _sign(a: Fraction, b: Fraction, d: int) -> int:
    """The exact sign of a + b*sqrt(d), for a d that is no square."""
    if a >= 0 and b >= 0:
        return 0 if a == b == 0 else 1
    if a <= 0 and b <= 0:
        return -1
    # Opposite signs: the term with the larger square decides (a^2 = d*b^2 is impossible for nonzero a, b).
    return (1 if a > 0 else -1) if a * a > b * b * d else (1 if b > 0 else -1)


def _floor(a: Fraction, b: Fraction, d: int) -> int:
    """floor(a + b*sqrt(d)) for b != 0, by bisection on the integers with exact sign tests."""
    bound = abs(a) + abs(b) * (isqrt(d) + 1)
    low, high = -int(bound) - 2, int(bound) + 2  # low <= x < high throughout.
    while high - low > 1:
        middle = (low + high) // 2
        if _sign(a - middle, b, d) >= 0:
            low = middle
        else:
            high = middle
    return low


def reference_expansion(p: int, q: int, d: int, r: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The canonical (preperiod, period) of (p + q*sqrt(d))/r: the terms until a complete quotient repeats. A
    complete quotient decides every later term and the earlier ones are pairwise distinct, so the first repetition
    marks the end of the shortest preperiod and of the minimal period after it."""
    a, b = Fraction(p, r), Fraction(q, r)
    seen: dict[tuple[Fraction, Fraction], int] = {}
    terms: list[int] = []
    while (a, b) not in seen:
        seen[a, b] = len(terms)
        term = _floor(a, b, d)
        terms.append(term)
        a -= term
        norm = a * a - b * b * d  # 1/(a + b*sqrt(d)) = (a - b*sqrt(d))/(a^2 - d*b^2)
        a, b = a / norm, -b / norm
    start = seen[a, b]
    return tuple(terms[:start]), tuple(terms[start:])


def approximates(preperiod: tuple, period: tuple, p: int, q: int, d: int, r: int) -> bool:
    """Whether the expansion, its period repeated 200 times, agrees with (p + q*sqrt(d))/r to 40 decimal places: a
    numeric confirmation that a forged expansion still has the right value."""
    with localcontext() as context:
        context.prec = 80
        target = (Decimal(p) + Decimal(q) * Decimal(d).sqrt()) / Decimal(r)
        terms = list(preperiod) + list(period) * 200
        x = Decimal(terms[-1])
        for term in reversed(terms[:-1]):
            x = term + 1 / x
        return abs(x - target) < Decimal(10) ** -40


def _fraction(value: object) -> Fraction:
    if type(value) is int:
        return Fraction(value)
    assert type(value) is Q
    return Fraction(value.numerator, value.denominator)


def reference_convergents(value: object) -> tuple[mf.Rational, ...]:
    """The convergents of the canonical continued fraction of an int or Rational: its Euclidean terms, each prefix
    folded from the right with Fractions."""
    source = _fraction(value)
    numerator, denominator, terms = source.numerator, source.denominator, []
    while True:
        term, rest = divmod(numerator, denominator)
        terms.append(term)
        if not rest:
            break
        numerator, denominator = denominator, rest
    convergents = []
    for end in range(1, len(terms) + 1):
        x = Fraction(terms[end - 1])
        for term in reversed(terms[:end - 1]):
            x = term + 1 / x
        convergents.append(Q(x.numerator, x.denominator))
    return tuple(convergents)


def recurrence_convergents(terms: tuple[int, ...]) -> tuple[mf.Rational, ...]:
    """The convergents of any term list, canonical or not (zero terms included), by the forward recurrence."""
    p, p_before, q, q_before, convergents = 1, 0, 0, 1, []
    for term in terms:
        p, p_before = term * p + p_before, p
        q, q_before = term * q + q_before, q
        convergents.append(Q(p, q))
    return tuple(convergents)


def fibonacci_convergents(n: int) -> tuple[mf.Rational, ...]:
    """The convergents of F(n+1)/F(n) = [1; 1, ..., 1, 2] (n - 1 terms) in closed form: F(k+2)/F(k+1) for every
    prefix of ones, then F(n+1)/F(n)."""
    fib = [0, 1]
    while len(fib) <= n + 1:
        fib.append(fib[-1] + fib[-2])
    return tuple(Q(fib[k + 2], fib[k + 1]) for k in range(n - 2)) + (Q(fib[n + 1], fib[n]),)


def brute_pell(d: int, limit: int) -> tuple[int, int] | None:
    """The least (x, y) with x^2 - d*y^2 = 1 and 0 < y <= limit, by trying every y; None beyond the limit."""
    for y in range(1, limit + 1):
        x = isqrt(d * y * y + 1)
        if x * x == d * y * y + 1:
            return x, y
    return None


def sqrt_period(d: int) -> tuple[int, ...]:
    preperiod, period = reference_expansion(0, 1, d, 1)
    assert preperiod == (isqrt(d),)
    return period


def period_pell(d: int) -> tuple[int, int]:
    """The fundamental solution from the period of sqrt(d) by Lagrange's theorem: the convergent [a0; b0, ..., b(L-2)]
    folded from the right, squared in Z[sqrt(d)] for an odd period length L."""
    period = sqrt_period(d)
    terms = (isqrt(d),) + period[:-1]
    x = Fraction(terms[-1])
    for term in reversed(terms[:-1]):
        x = term + 1 / x
    h, k = x.numerator, x.denominator
    return (h, k) if len(period) % 2 == 0 else (h * h + d * k * k, 2 * h * k)


def closed_expansions(n: int) -> tuple[tuple[tuple[int, int, int, int], tuple[int, ...], tuple[int, ...]], ...]:
    """Classical expansions for n >= 2: sqrt(n^2 + 1) = [n; (2n)], sqrt(n^2 + 2) = [n; (n, 2n)] and
    sqrt(n^2 - 1) = [n - 1; (1, 2n - 2)]."""
    return (((0, 1, n * n + 1, 1), (n,), (2 * n,)),
            ((0, 1, n * n + 2, 1), (n,), (n, 2 * n)),
            ((0, 1, n * n - 1, 1), (n - 1,), (1, 2 * n - 2)))


def closed_pell(n: int) -> tuple[tuple[int, tuple[int, int], tuple[int, ...]], ...]:
    """(d, fundamental solution, period of sqrt(d)) for d = n^2 + 1 (L = 1), n^2 + 2 and n^2 - 1 (L = 2)."""
    return ((n * n + 1, (2 * n * n + 1, 2 * n), (2 * n,)),
            (n * n + 2, (n * n + 1, n), (n, 2 * n)),
            (n * n - 1, (n, 1), (1, 2 * n - 2)))


class QuadraticIrrationalCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertExhausts(self, operation: str, args: object, value: object, evidence: object = ()) -> None:
        """A claim beyond the checker's caps raises ResourceLimitError, the card's caps allow it, and quickly."""
        started = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(self.FAMILY, operation, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    def assertRejectsFast(self, operation: str, args: object, value: object, evidence: object = ()) -> None:
        started = time.perf_counter()
        self.assertRejects(operation, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    # True claims, each built by the test.

    def test_accepts_the_known_values_of_the_card(self):
        for (p, q, d, r), preperiod, period in KNOWN_EXPANSIONS:
            with self.subTest(p=p, q=q, d=d, r=r):
                self.assertEqual(reference_expansion(p, q, d, r), (preperiod, period))
                self.assertAccepts(EXPAND, expansion_args(p, q, d, r), PCF(preperiod, period))
        for value, convergents in KNOWN_CONVERGENTS:
            with self.subTest(value=value):
                self.assertEqual(reference_convergents(value), convergents)
                self.assertAccepts(CONVERGENTS, {"value": value}, convergents)
        for d, solution in KNOWN_PELL:
            with self.subTest(d=d):
                self.assertEqual(period_pell(d), solution)
                if d != 61:  # y = 226,153,980 is out of the brute force's reach.
                    self.assertEqual(brute_pell(d, 1_000), solution)
                self.assertAccepts(PELL, {"d": d}, solution, (sqrt_period(d),))
        self.assertEqual(sqrt_period(61), (1, 4, 3, 1, 2, 2, 1, 3, 4, 1, 14))

    def test_accepts_seeded_expansions(self):
        """300 seeded quadratic irrationals with their expansions from the bisection reference; the canonical
        expansion of x + 1, a true value of another number, is refused."""
        rng = random.Random(20261112)
        nonzero_q = [v for v in range(-20, 21) if v]
        nonzero_r = [v for v in range(-30, 31) if v]
        radicands = NON_SQUARES[:100]  # 2 to 111: periods of up to some hundreds of terms with these q and r.
        for _ in range(300):
            p, q, d, r = rng.randint(-60, 60), rng.choice(nonzero_q), rng.choice(radicands), rng.choice(nonzero_r)
            args = expansion_args(p, q, d, r)
            with self.subTest(args=args):
                self.assertAccepts(EXPAND, args, PCF(*reference_expansion(p, q, d, r)))
                self.assertRejects(EXPAND, args, PCF(*reference_expansion(p + r, q, d, r)))

    def test_accepts_special_expansions(self):
        """Empty preperiods, negative first terms, negative q or r, d with square factors, x far from 0, a long period,
        and huge terms from classical closed forms, which the reference confirms for small n."""
        for p, q, d, r in ((1, 1, 5, 2), (1, -1, 5, -2), (0, 1, 2, -1), (1, 1, 2, 1), (5, 1, 7, 2), (0, 1, 8, 1),
                           (0, 2, 2, 1), (1, 3, 12, 5), (2, -1, 18, 3), (-7, -1, 3, -4), (10**30 + 7, -3, 13, 1),
                           (-10**25, 7, 99, 3), (0, 1, LONG_PERIOD_D, 1), (1, 1, LONG_PERIOD_D, 3)):
            preperiod, period = reference_expansion(p, q, d, r)
            with self.subTest(p=p, q=q, d=d, r=r):
                self.assertAccepts(EXPAND, expansion_args(p, q, d, r), PCF(preperiod, period))
        self.assertEqual(reference_expansion(1, 1, 5, 2), ((), (1,)))
        self.assertEqual(len(sqrt_period(LONG_PERIOD_D)), 88)
        for n in range(2, 40):
            for args, preperiod, period in closed_expansions(n):
                self.assertEqual(reference_expansion(*args), (preperiod, period), args)
        for n in (10**40 + 3, 3**5000, 2**20000 + 12345):
            for args, preperiod, period in closed_expansions(n):
                with self.subTest(bits=n.bit_length(), d=args[2] - n * n):
                    self.assertAccepts(EXPAND, expansion_args(*args), PCF(preperiod, period))

    def test_accepts_convergents(self):
        """Ints, Rationals of denominator 1, zero, negative values, large values, 200 seeded rationals, and the long
        expansions of ratios of Fibonacci numbers."""
        rng = random.Random(415093)
        values = [0, 1, -1, 5, -5, Q(5), Q(0), Q(1, 2), Q(-1, 2), Q(2, 3), Q(-2, 3), Q(10**30 + 1, 10**29),
                  Q(-10**40, 7), Q(1, 10**25), 2**1000]
        values += [Q(rng.randint(-10**6, 10**6), rng.randint(1, 10**6)) for _ in range(200)]
        for value in values:
            with self.subTest(value=value):
                self.assertAccepts(CONVERGENTS, {"value": value}, reference_convergents(value))
        for n in (3, 4, 50):
            closed = fibonacci_convergents(n)
            self.assertEqual(closed, reference_convergents(closed[-1]))
        long_value = fibonacci_convergents(LONG_FIBONACCI)
        started = time.perf_counter()
        self.assertAccepts(CONVERGENTS, {"value": long_value[-1]}, long_value)
        self.assertLess(time.perf_counter() - started, FAST)

    def test_accepts_pell_for_every_small_d(self):
        """Every d < 200 that is no square: the brute force decides the solution wherever y <= 20,000 (and so
        confirms the period reference for both parities of L), the period reference elsewhere."""
        brute_forced, parities = 0, set()
        for d in NON_SQUARES:
            solution, period = period_pell(d), sqrt_period(d)
            brute = brute_pell(d, 20_000)
            with self.subTest(d=d):
                if brute is not None:
                    self.assertEqual(brute, solution)
                    brute_forced += 1
                    parities.add(len(period) % 2)
                self.assertAccepts(PELL, {"d": d}, solution, (period,))
        self.assertGreaterEqual(brute_forced, 150)
        self.assertEqual(parities, {0, 1})

    def test_accepts_large_pell_solutions(self):
        for n in (2, 3, 10):
            for d, solution, period in closed_pell(n):
                self.assertEqual((brute_pell(d, 1_000), sqrt_period(d)), (solution, period), d)
        for n in (10**40 + 3, 2**20000 + 12345):
            for d, solution, period in closed_pell(n):
                with self.subTest(bits=n.bit_length(), d=d - n * n):
                    self.assertAccepts(PELL, {"d": d}, solution, (period,))

    def test_caps(self):
        """The term bits of sqrt(n^2 + 1) = [n; (2n)] are 2*bitlen(n) + 1: at the default limits the cap admits
        n = 2**32766 (65,535 bits) and refuses n = 2**32767 (65,537 bits) with ResourceLimitError, a true claim beyond
        the budget, not a wrong one, for both operations; the same at max_integer_bits=200. The count of terms is
        refused beyond MAX_TERMS before the terms are read."""
        for n, limits in ((2**32766, None), (2**98, ComputationLimits(max_integer_bits=200))):
            for m, admitted in ((n, True), (2 * n, False)):
                expansion, value, period = PCF((m,), (2 * m,)), (2 * m * m + 1, 2 * m), (2 * m,)
                args, pell = expansion_args(0, 1, m * m + 1, 1), {"d": m * m + 1}
                with self.subTest(bits=m.bit_length(), limits=limits):
                    if admitted:
                        self.assertAccepts(EXPAND, args, expansion, limits=limits)
                        self.assertAccepts(PELL, pell, value, (period,), limits=limits)
                    else:
                        with self.assertRaises(ResourceLimitError):
                            support.verdict(family, EXPAND, args, expansion, limits=limits)
                        with self.assertRaises(ResourceLimitError):
                            support.verdict(family, PELL, pell, value, (period,), limits=limits)
        wide = ComputationLimits(max_integer_bits=200_000)  # So that only the count of terms can refuse.
        within = support.forged(PCF, preperiod=(1,), period=(1,) * (MAX_TERMS - 2) + (2,))
        beyond = support.forged(PCF, preperiod=(1,), period=(1,) * (MAX_TERMS - 1) + (2,))
        started = time.perf_counter()
        self.assertRejects(EXPAND, SQRT2, within, limits=wide)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, EXPAND, SQRT2, beyond, limits=wide)
        self.assertLess(time.perf_counter() - started, FAST)

    # Corrupted claims: each answers False, never raising.

    def test_rejects_wrong_values(self):
        for value in (PCF((1,), (3,)), PCF((), (2,)), PCF((0,), (2,)), PCF((1,), (1, 2)), PCF((-2, 1, 1), (2,)),
                      PCF((2,), (2, 1))):
            with self.subTest(value=value):
                self.assertRejects(EXPAND, SQRT2, value)
        self.assertRejects(EXPAND, expansion_args(0, 1, 3, 1), PCF(*SQRT2_VALUE))
        self.assertRejects(EXPAND, expansion_args(1, 1, 2, 1), PCF(*SQRT2_VALUE))
        self.assertAccepts(EXPAND, expansion_args(0, 2, 2, 2), PCF(*SQRT2_VALUE))  # Another form of sqrt(2).
        self.assertRejects(EXPAND, expansion_args(0, 2, 2, 2), PCF((1,), (3,)))
        right = (Q(4), Q(9, 2), Q(58, 13), Q(415, 93))
        for value in ((Q(4), Q(9, 2), Q(58, 13), Q(416, 93)), right[:3], right + right[-1:], (Q(5),) + right[1:],
                      (Q(4), Q(9, 2), Q(57, 13), Q(415, 93)), (Q(4), Q(58, 13), Q(415, 93)), right[1:]):
            with self.subTest(value=value):
                self.assertRejects(CONVERGENTS, {"value": Q(415, 93)}, value)
        self.assertRejects(CONVERGENTS, {"value": 6}, (Q(5),))
        self.assertRejects(CONVERGENTS, {"value": Q(415, 92)}, right)
        # (1, 1) and (18, 5) solve x^2 - d*y^2 = -1: the convergent p(L-1)/q(L-1) of an odd period, not squared.
        for d, value in ((2, (4, 2)), (2, (2, 1)), (2, (3, 3)), (2, (1, 1)), (7, (8, 4)), (13, (649, 181)),
                         (13, (18, 5))):
            with self.subTest(d=d, value=value):
                self.assertRejects(PELL, {"d": d}, value, (sqrt_period(d),))
        self.assertRejects(PELL, {"d": 3}, (3, 2), ((2,),))
        self.assertRejects(PELL, {"d": 8}, (3, 2), ((2,),))

    def test_rejects_noncanonical_values_that_are_mathematically_right(self):
        """Each forged value has the right value, confirmed numerically or exactly; only the canonical form that it
        breaks can refuse it."""
        forgeries = (
            ((0, 1, 2, 1), (1, 2), (2,)),  # The preperiod is not the shortest.
            ((0, 1, 2, 1), (1,), (2, 2)),  # The period is not primitive.
            ((0, 1, 2, 1), (1, 2, 2), (2, 2, 2)),
            ((0, 1, 2, 1), (0, 0, 1), (2,)),  # A zero inner term: [0; 0, t] = t.
            ((0, 1, 2, 1), (2, -1, -2, 1, 1), (2,)),  # Negative inner terms: [a; t] = [a + 1; -1, 1 - t].
            ((0, 1, 2, 1), (True,), (2,)),  # A bool term.
            ((1, 1, 5, 2), (1,), (1,)),
            ((1, 1, 5, 2), (), (1, 1)),
            ((0, 1, 7, 1), (2, 1), (1, 1, 4, 1)),  # The period rotated into the preperiod.
            ((0, 1, 7, 1), (2,), (1, 1, 1, 4, 1, 1, 1, 4)),
        )
        for args, preperiod, period in forgeries:
            with self.subTest(args=args, preperiod=preperiod, period=period):
                self.assertTrue(approximates(preperiod, period, *args))
                self.assertRejects(EXPAND, expansion_args(*args), support.forged(PCF, preperiod=preperiod,
                                                                                period=period))
        # Convergents of non-canonical expansions of 415/93: a last term 1, zero terms, a negative term.
        for terms in ((4, 2, 6, 6, 1), (4, 2, 0, 0, 6, 7), (5, -2, 7, 7)):
            convergents = recurrence_convergents(terms)
            with self.subTest(terms=terms):
                self.assertEqual(convergents[-1], Q(415, 93))
                self.assertRejects(CONVERGENTS, {"value": Q(415, 93)}, convergents)
        self.assertRejects(CONVERGENTS, {"value": Q(415, 93)}, (4, Q(9, 2), Q(58, 13), Q(415, 93)))  # An int.
        self.assertRejects(CONVERGENTS, {"value": 5}, (5,))
        self.assertRejects(CONVERGENTS, {"value": Q(5)}, (5,))
        # Pell: another solution, signs, and evidence that is not the minimal period (the value may even be right).
        for d, value, evidence in ((2, (17, 12), ((2,),)), (2, (3, 2), ((2, 2),)), (2, (17, 12), ((2, 2),)),
                                   (2, (99, 70), ((2, 2, 2),)), (2, (-3, 2), ((2,),)), (2, (3, -2), ((2,),)),
                                   (2, (-3, -2), ((2,),)), (3, (7, 4), ((1, 2),)), (3, (2, 1), ((1, 2, 1, 2),)),
                                   (7, (127, 48), ((1, 1, 1, 4),)), (7, (8, 3), ((1, 1, 4, 1),)),
                                   (7, (8, 3), ((2, 1, 1, 1, 4),)), (7, (8, 3), ((1, 1, 1, 4, 1, 1, 1, 4),))):
            with self.subTest(d=d, value=value, evidence=evidence):
                self.assertEqual(value[0] ** 2 - d * value[1] ** 2, 1)
                self.assertRejects(PELL, {"d": d}, value, evidence)

    def test_rejects_impostors_and_forged_records(self):
        true = PCF(*SQRT2_VALUE)
        self.assertAccepts(EXPAND, SQRT2, true)

        class Subclass(PCF):
            """A subclass is another class: its module and name differ."""

        class Integer(int):
            pass

        class Tuple(tuple):
            pass

        for value in (support.impostor(true), Subclass(*SQRT2_VALUE), SQRT2_VALUE, None, (true,),
                      support.forged(PCF, preperiod=(1,)), support.forged(PCF, period=(2,)), support.forged(PCF),
                      support.forged(PCF, preperiod=(1,), period=()), support.forged(PCF, preperiod=None, period=(2,)),
                      support.forged(PCF, preperiod=(Integer(1),), period=(2,)),
                      support.forged(PCF, preperiod=(1,), period=(Integer(2),)),
                      support.forged(PCF, preperiod=Tuple((1,)), period=(2,)),
                      support.forged(PCF, preperiod=(1,), period=Tuple((2,)))):
            with self.subTest(value=value):
                self.assertRejects(EXPAND, SQRT2, value)
        fake = dataclasses.make_dataclass("Rational", ["numerator", "denominator"], frozen=True)
        missing, zero = object.__new__(Q), object.__new__(Q)
        object.__setattr__(zero, "numerator", 5)
        object.__setattr__(zero, "denominator", 0)
        for source, value in ((5, (fake(5, 1),)), (Q(1, 2), (Q(0), fake(1, 2))), (5, Tuple((Q(5),))), (5, (missing,)),
                              (5, (zero,)), (5, (Q(5), missing)), (Q(1, 2), (Q(0), Integer(2)))):
            with self.subTest(source=source, value=value):
                self.assertRejects(CONVERGENTS, {"value": source}, value)
        self.assertRejects(CONVERGENTS, {"value": fake(5, 1)}, (Q(5),))
        self.assertRejects(CONVERGENTS, {"value": missing}, (Q(5),))
        for value, evidence in ((Tuple((3, 2)), ((2,),)), ((Integer(3), 2), ((2,),)), ((3, Integer(2)), ((2,),)),
                                ((3, 2), (Tuple((2,)),)), ((3, 2), ((Integer(2),),)), ((3, 2), Tuple(((2,),)))):
            with self.subTest(value=value, evidence=evidence):
                self.assertRejects(PELL, {"d": 2}, value, evidence)

    def test_rejects_inexact_numbers_and_lists(self):
        for preperiod, period in (((1.0,), (2,)), (("1",), (2,)), ((1,), (2.0,)), ((1,), ("2",)), ((1,), (True, 2)),
                                  ((Q(1),), (2,)), ((1,), (None,)), ([1], (2,)), ((1,), [2]), ([1], [2])):
            with self.subTest(preperiod=preperiod, period=period):
                self.assertRejects(EXPAND, SQRT2, support.forged(PCF, preperiod=preperiod, period=period))
        true = PCF(*SQRT2_VALUE)
        for name, bad in (("p", False), ("p", 0.0), ("p", Q(0)), ("q", True), ("q", 1.0), ("q", "1"), ("d", 2.0),
                          ("d", Q(2)), ("d", "2"), ("r", True), ("r", 1.0), ("r", None)):
            with self.subTest(name=name, bad=bad):
                self.assertRejects(EXPAND, dict(SQRT2, **{name: bad}), true)
        for source, value in ((True, (Q(1),)), (5.0, (Q(5),)), ("5", (Q(5),)), (None, (Q(5),)), ((5,), (Q(5),)),
                              (5, (5.0,)), (1, (True,)), (5, ("5",)), (5, [Q(5)]), (Q(1, 2), [Q(0), Q(1, 2)]),
                              (Q(1, 2), (Q(0), 0.5))):
            with self.subTest(source=source, value=value):
                self.assertRejects(CONVERGENTS, {"value": source}, value)
        for d, value, evidence in ((True, (3, 2), ((2,),)), (2.0, (3, 2), ((2,),)), ("2", (3, 2), ((2,),)),
                                   (Q(2), (3, 2), ((2,),)), (2, (3.0, 2), ((2,),)), (2, (3, 2.0), ((2,),)),
                                   (2, ("3", 2), ((2,),)), (2, [3, 2], ((2,),)), (2, (3, 2), ((2.0,),)),
                                   (2, (3, 2), (("2",),)), (2, (3, 2), ([2],)), (2, (3, 2), [(2,)]),
                                   (3, (2, 1), ((True, 2),)), (3, (True, 1), ((1, 2),))):
            with self.subTest(d=d, value=value, evidence=evidence):
                self.assertRejects(PELL, {"d": d}, value, evidence)

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        """This family proves no primality and needs no factorization: certificate-shaped evidence is refused like
        any other extra evidence (forged certificates have no place to go)."""
        from mathforge.kernels.certificates import factorization_evidence, prime_evidence
        certificate = prime_evidence(2**89 - 1)  # A Pocklington certificate: 2**89 - 1 is above psi_13.
        factors = factorization_evidence(12)
        true = PCF(*SQRT2_VALUE)
        for evidence in ((1,), ((),), ((2,),), (true,), (certificate,), certificate, (factors,), factors, [], None,
                         [()], {}, 0):
            with self.subTest(evidence=evidence):
                self.assertRejects(EXPAND, SQRT2, true, evidence)
                self.assertRejects(CONVERGENTS, {"value": Q(415, 93)}, reference_convergents(Q(415, 93)), evidence)
        for evidence in ((), ((2,), (2,)), ((2,), ()), (((2,),),), ((),), (None,), (2,), [(2,)], ((2,), certificate),
                         (certificate,), (factors,), factors, ((1, 2),), ((2, 1),), None):
            with self.subTest(evidence=evidence):
                self.assertRejects(PELL, {"d": 2}, (3, 2), evidence)
        self.assertAccepts(PELL, {"d": 2}, (3, 2), ((2,),))

    def test_rejects_invalid_arguments(self):
        """No claim is true on input that the producer rejects: d a square or below 2, q = 0 or r = 0 (with r = 0
        and q = 0 both identities of the checker would hold trivially), wrong argument names, or no dict."""
        true = PCF(*SQRT2_VALUE)
        for p, q, d, r in ((0, 1, 4, 1), (0, 1, 9, 1), (0, 1, 1, 1), (0, 1, 0, 1), (0, 1, -2, 1), (0, 0, 2, 1),
                           (0, 1, 2, 0), (0, 0, 2, 0), (1, 0, 2, 1)):
            for value in (true, PCF((2,), (1,)), PCF((), (1,)), PCF((1,), (1, 2))):
                with self.subTest(args=(p, q, d, r), value=value):
                    self.assertRejects(EXPAND, expansion_args(p, q, d, r), value)
        for args in ({"p": 0, "q": 1, "d": 2}, dict(SQRT2, s=0), {"P": 0, "q": 1, "d": 2, "r": 1}, list(SQRT2.items()),
                     None, {}):
            with self.subTest(args=args):
                self.assertRejects(EXPAND, args, true)
        for d in (4, 1, 0, -2, 9, 100):
            with self.subTest(d=d):
                self.assertRejects(PELL, {"d": d}, (3, 2), ((2,),))
        for args in ({"n": 2}, {"d": 2, "e": 1}, {}, [("d", 2)]):
            with self.subTest(args=args):
                self.assertRejects(PELL, args, (3, 2), ((2,),))
                self.assertRejects(CONVERGENTS, args, (Q(5),))
        self.assertRejects(CONVERGENTS, {"x": 5}, (Q(5),))
        self.assertRejects(CONVERGENTS, {"value": 5, "x": 1}, (Q(5),))

    def test_rejects_wrong_operation_names(self):
        true = PCF(*SQRT2_VALUE)
        for operation in ("continued_fraction", CONVERGENTS, PELL, "Quadratic_continued_fraction", "", None, 1):
            with self.subTest(operation=operation):
                self.assertRejects(operation, SQRT2, true)
        self.assertRejects(EXPAND, {"value": 5}, (Q(5),))
        self.assertRejects(PELL, {"value": 5}, (Q(5),))
        self.assertRejects(EXPAND, {"d": 2}, (3, 2), ((2,),))
        self.assertRejects(CONVERGENTS, {"d": 2}, (3, 2), ((2,),))
        self.assertRejects("continued_fraction", {"value": 5}, mf.ntheory.continued_fraction(5))

    def test_oversized_claims_fail_fast(self):
        """Beyond the caps of the card: ResourceLimitError before any matrix product. Within them the whole check
        runs, and still answers in time. convergents has no caps: its oversized values answer False at once."""
        huge = 2**70_000
        for preperiod, period in (((1,), (1,) * 999_999 + (2,)), ((1,), (huge,)), ((huge,), (2,)),
                                  ((1,), (3,) * 40_000 + (2,)), ((1,) * 1_000_000, (2,))):
            with self.subTest(terms=len(preperiod) + len(period)):
                self.assertExhausts(EXPAND, SQRT2, support.forged(PCF, preperiod=preperiod, period=period))
        for period in ((1,) * 999_999 + (2,), (huge,), (3,) * 40_000 + (2,)):
            with self.subTest(terms=len(period)):
                self.assertExhausts(PELL, {"d": 2}, (3, 2), (period,))
        self.assertRejectsFast(EXPAND, SQRT2, support.forged(PCF, preperiod=(1,), period=(1,) * 59_999 + (2,)))
        self.assertRejectsFast(PELL, {"d": 2}, (3, 2), ((1,) * 59_999 + (2,),))
        self.assertRejectsFast(PELL, {"d": 2}, (huge + 1, 1), ((2,),))
        self.assertRejectsFast(PELL, {"d": 2}, (3, huge), ((2,),))
        self.assertRejectsFast(PELL, {"d": 2}, (huge, huge), ((2,),))
        # A malformed claim answers False whatever the size of its terms.
        self.assertRejectsFast(EXPAND, SQRT2, support.forged(PCF, preperiod=(1,), period=(huge, 0)))
        self.assertRejectsFast(PELL, {"d": 2}, (3, 2), ((huge, 2.0),))
        source = Q(415, 93)
        for value in ((Q(4),) * 1_000_000, (Q(huge),), (Q(4), Q(9, 2), Q(58, 13), Q(huge + 1, huge)),
                      (Q(4), Q(1, huge), Q(58, 13), Q(415, 93)), (Q(4), Q(9, 2)) * 8, (Q(huge, 3),) * 10_000):
            with self.subTest(items=len(value)):
                self.assertRejectsFast(CONVERGENTS, {"value": source}, value)
        long_value = fibonacci_convergents(LONG_FIBONACCI)
        forged_value = long_value[:-1] + (Q(long_value[-1].numerator + 1, long_value[-1].denominator),)
        self.assertRejectsFast(CONVERGENTS, {"value": long_value[-1]}, forged_value)

    # The record and its presentation, which the checker tests rely on.

    def test_record_canonical_form(self):
        for preperiod, period in (((1,), (2,)), ((), (1,)), ((-2, 1, 1), (2,)), ((0,), (1, 2)), ((5,), (1, 1, 2)),
                                  ((7, 4), (1, 2, 1, 2, 1, 3))):
            with self.subTest(preperiod=preperiod, period=period):
                record = PCF(preperiod, period)
                self.assertEqual((record.preperiod, record.period), (preperiod, period))
                self.assertEqual(mf.from_data(mf.to_data(record)), record)
        for preperiod, period in (((1,), ()), ((1, 2), (2,)), ((1,), (2, 2)), ((1, 0), (2,)), ((1,), (0,)),
                                  ((1,), (-2,)), ((1, -1), (2,)), ((True,), (2,)), ((1,), (2.0,)), ([1], (2,)),
                                  ((1,), [2]), ((), (1, 1)), ((1,), (1,)), ((1,), (1, 2, 1, 2)), (None, (2,)),
                                  ((1,), None)):
            with self.subTest(preperiod=preperiod, period=period), self.assertRaises(mf.InvalidInput):
                PCF(preperiod, period)
        with self.assertRaises(ResourceLimitError):
            PCF((1,), (2**65537,))
        document = mf.to_data(PCF((1,), (2,)))
        document["data"]["period"]["items"].append(dict(document["data"]["period"]["items"][0]))
        with self.assertRaisesRegex(mf.SerializationError, "^Invalid mathematical document$"):
            mf.from_data(document)  # The period (2, 2) is not primitive.

    def test_rendering(self):
        for preperiod, period, text, latex in (
                ((1,), (2,), "[1; (2)]", r"[1; \overline{2}]"),
                ((2,), (1, 1, 1, 4), "[2; (1, 1, 1, 4)]", r"[2; \overline{1, 1, 1, 4}]"),
                ((), (1,), "[(1)]", r"[\overline{1}]"),
                ((-2, 1, 1), (2,), "[-2; 1, 1, (2)]", r"[-2; 1, 1, \overline{2}]")):
            with self.subTest(text=text):
                record = PCF(preperiod, period)
                self.assertEqual((mf.render_text(record), mf.to_latex(record)), (text, latex))
        with self.assertRaisesRegex(mf.ResourceLimitError, "characters"):
            mf.render_text(PCF((1,), (1,) * 30 + (2,)), limits=mf.RenderLimits(max_chars=20))
