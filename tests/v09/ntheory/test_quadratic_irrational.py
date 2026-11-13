"""The quadratic_irrational family of mathforge.ntheory: quadratic_continued_fraction, convergents, pell_fundamental,
from the producer to the MCP area tool.

These tests are the definition of done of the family; tests/v09/ntheory/test_quadratic_irrational_checker.py tests its
checker alone. tests/v09/support.py holds the shared helpers, and tests/v08/test_continued_fraction.py tests the pilot
family. A FamilyTestCase keeps FAMILY registered while its tests run, before and after

    python tools/new_family.py register ntheory quadratic_irrational

No expected value comes from the producer. Each one is a known value of the card (docs/v09/cards/m1-ntheory.md), one
worked by hand or a classical closed form, or the answer of a slow reference of this module that shares no code with
the producer: expansion() finds each term of (p + q*sqrt(d))/r by bisection over the integers with exact sign tests in
Q(sqrt(d)) and keeps the complete quotients as pairs of Fractions until one repeats (the producer keeps integer state
forms and stops at the first return to a reduced one); folded() folds every prefix of the Euclidean terms of a rational
from the right (the producer runs the forward recurrence); brute_pell() tries every y, and period_pell() reads the
fundamental solution off the period that expansion() finds (Lagrange's theorem); and quadratic() builds, backwards from
a chosen canonical expansion, the number that it represents, so that the claim exists before anything is expanded.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import gcd, isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean, operations_v6
from mathforge.areas.ntheory import quadratic_irrational as family
from mathforge.areas.ntheory.quadratic_irrational import PeriodicContinuedFraction as PCF
from mathforge.certify.ntheory.quadratic_irrational import MAX_TERMS
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import Budget, ComputationLimits, ResourceLimitError, computation

from .. import support

Q = mf.Rational
EXPAND, CONVERGENTS, PELL = "quadratic_continued_fraction", "convergents", "pell_fundamental"
FAST = 5.0  # Seconds: the bound for the heaviest inputs below, far above what each one takes.
NON_SQUARES = tuple(d for d in range(2, 340) if isqrt(d) ** 2 != d)  # 321 radicands.
LONG_D = 100_000_231  # sqrt(LONG_D) has a period of 21,032 terms whose bit lengths add up to 47,605, within the caps.
# Every inexact or misshapen item that an int argument rejects; the rational argument of convergents rejects the same
# but the Rationals, which it takes.
INEXACT = (0.5, True, False, "1", None, 2.0, Q(1, 2), Q(2), mf.Symbol("x"), mf.sqrt(2), (2,), [2], {"d": 2})

# ((p, q, d, r), preperiod, period): the card's known values, then values worked by hand (each confirmed by
# expansion() in test_known_values): r < 0, a purely periodic number below 1's conjugate, d with a square factor, the
# same number in another form, and a negative number above -1.
CARD_EXPANSIONS = (((0, 1, 2, 1), (1,), (2,)),
                   ((0, 1, 3, 1), (1,), (1, 2)),
                   ((1, 1, 5, 2), (), (1,)),
                   ((0, 1, 7, 1), (2,), (1, 1, 1, 4)),
                   ((3, 1, 2, 1), (4,), (2,)),
                   ((0, -1, 2, 1), (-2, 1, 1), (2,)))
BY_HAND_EXPANSIONS = (((0, 1, 2, -1), (-2, 1, 1), (2,)),  # -sqrt(2) through r < 0: the card's last value again.
                      ((1, -1, 5, -2), (0,), (1,)),  # (sqrt(5) - 1)/2 = 1/golden ratio.
                      ((1, 1, 2, 1), (), (2,)),  # 1 + sqrt(2), reduced: purely periodic.
                      ((-1, 1, 2, 1), (0,), (2,)),  # sqrt(2) - 1.
                      ((0, 1, 8, 1), (2,), (1, 4)),  # sqrt(8) = 2*sqrt(2).
                      ((0, 2, 2, 1), (2,), (1, 4)),  # The same number.
                      ((0, -1, 2, 2), (-1, 3), (2,)))  # -sqrt(2)/2 = -1 + 1/(2 + sqrt(2)).
# (value, convergents): the card's known values, then values worked by hand.
CARD_CONVERGENTS = ((Q(415, 93), (Q(4), Q(9, 2), Q(58, 13), Q(415, 93))),
                    (5, (Q(5),)),
                    (Q(-7, 3), (Q(-3), Q(-2), Q(-7, 3))))
BY_HAND_CONVERGENTS = ((0, (Q(0),)),
                       (Q(5), (Q(5),)),  # A Rational of denominator 1.
                       (Q(1, 2), (Q(0), Q(1, 2))),
                       (Q(-1, 2), (Q(-1), Q(-1, 2))),  # [-1; 2]
                       (Q(13, 8), (Q(1), Q(2), Q(3, 2), Q(5, 3), Q(13, 8))),  # [1; 1, 1, 1, 2]
                       (Q(-415, 93), (Q(-5), Q(-4), Q(-9, 2), Q(-58, 13), Q(-415, 93))))  # [-5; 1, 1, 6, 7]
# (d, (x, y), period of sqrt(d)): the card's known values, then d = 109, whose solution is a classical example.
CARD_PELL = ((2, (3, 2), (2,)), (3, (2, 1), (1, 2)), (5, (9, 4), (4,)), (7, (8, 3), (1, 1, 1, 4)),
             (13, (649, 180), (1, 1, 1, 1, 6)),
             (61, (1766319049, 226153980), (1, 4, 3, 1, 2, 2, 1, 3, 4, 1, 14)))
CLASSICAL_PELL = ((109, (158070671986249, 15140424455100), (2, 3, 1, 2, 4, 1, 6, 6, 1, 4, 2, 1, 3, 2, 20)),)


def expansion_args(p, q, d, r):
    """The arguments of a quadratic_continued_fraction request, as run() and the verifier hand them on."""
    return {"p": p, "q": q, "d": d, "r": r}


def produced(operation, /, *, limits=None, **arguments):
    """(value, evidence) from the producer itself, in a fresh budget."""
    with computation(limits, fresh=True):
        return family.produce(operation, arguments)


def _sign(a: Fraction, b: Fraction, d: int) -> int:
    """The exact sign of a + b*sqrt(d), for rationals a, b and an integer d >= 2 that is no square."""
    if a >= 0 and b >= 0:
        return 0 if a == 0 and b == 0 else 1
    if a <= 0 and b <= 0:
        return -1
    # Opposite signs, both nonzero: the part with the larger square decides; a^2 = d*b^2 would make d a square.
    return (1 if a > 0 else -1) if a * a > d * b * b else (1 if b > 0 else -1)


def _floor(a: Fraction, b: Fraction, d: int) -> int:
    """floor(a + b*sqrt(d)) for b != 0: an integer bracket found by doubling, then bisection."""
    low, high = -1, 1
    while _sign(a - low, b, d) < 0:
        low *= 2
    while _sign(a - high, b, d) >= 0:
        high *= 2
    while high - low > 1:  # a + b*sqrt(d) lies in [low, high).
        middle = (low + high) // 2
        if _sign(a - middle, b, d) >= 0:
            low = middle
        else:
            high = middle
    return low


def expansion(p, q, d, r):
    """The canonical (preperiod, period) of (p + q*sqrt(d))/r by definition: a term is the floor of the complete
    quotient a + b*sqrt(d), the next complete quotient 1/(a - term + b*sqrt(d)), until a complete quotient repeats. The
    complete quotient decides every later term, and the earlier ones are pairwise distinct, so the first repetition
    ends the shortest preperiod and the minimal period after it."""
    a, b = Fraction(p, r), Fraction(q, r)
    seen: dict[tuple[Fraction, Fraction], int] = {}
    terms: list[int] = []
    while (a, b) not in seen:
        seen[a, b] = len(terms)
        term = _floor(a, b, d)
        terms.append(term)
        a -= term
        norm = a * a - d * b * b  # (a + b*sqrt(d))*(a - b*sqrt(d)), nonzero.
        a, b = a / norm, -b / norm
    start = seen[a, b]
    return tuple(terms[:start]), tuple(terms[start:])


def _matrix(terms):
    """(A, B, C, D) of [[t0, 1], [1, 0]] ... [[t(n-1), 1], [1, 0]], the identity for no terms."""
    a, b, c, d = 1, 0, 0, 1
    for term in terms:
        a, b, c, d = a * term + b, a, c * term + d, c
    return a, b, c, d


def quadratic(preperiod, period):
    """(p, q, d, r) with (p + q*sqrt(d))/r = [preperiod, (period)], built backwards. With the period matrix
    [[A, B], [C, D]], the purely periodic tail is eta = (A - D + sqrt(Delta))/(2C), Delta = (A + D)^2 - 4*(-1)^L, and
    with the preperiod matrix [[P, P'], [Q, Q']] the number is (P*eta + P')/(Q*eta + Q') = (u + v*sqrt(Delta))/
    (w + z*sqrt(Delta)), which the conjugate of its denominator rationalizes. Delta is never a square, q = +-2C != 0."""
    big_a, _, big_c, big_d = _matrix(period)
    pre_p, pre_p1, pre_q, pre_q1 = _matrix(preperiod)
    delta = (big_a + big_d) ** 2 - (-4 if len(period) % 2 else 4)
    u, v = pre_p * (big_a - big_d) + 2 * big_c * pre_p1, pre_p
    w, z = pre_q * (big_a - big_d) + 2 * big_c * pre_q1, pre_q
    return u * w - v * z * delta, v * w - u * z, delta, w * w - z * z * delta


def primitive(word):
    """Whether a nonempty word is no repetition of a shorter block, by trying every block length."""
    length = len(word)
    return not any(length % size == 0 and word == word[:size] * (length // size) for size in range(1, length))


def canonical_expansion(rng):
    """A seeded canonical (preperiod, period): a primitive period, and a preperiod whose last term differs from the
    last period term."""
    while True:
        period = tuple(rng.randint(1, rng.choice((3, 30, 1000))) for _ in range(rng.randint(1, 6)))
        if not primitive(period):
            continue
        preperiod = tuple(rng.randint(1, 40) for _ in range(rng.randint(0, 4)))
        if preperiod:
            preperiod = (rng.randint(-50, 50),) + preperiod[1:]
            if preperiod[-1] == period[-1]:
                continue
        return preperiod, period


def _fraction(value) -> Fraction:
    return Fraction(value) if type(value) is int else Fraction(value.numerator, value.denominator)


def folded(value):
    """The convergents of the canonical continued fraction of an int or a Rational: its Euclidean terms, then every
    prefix of them folded from the right with Fractions."""
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


def brute_pell(d, limit):
    """The least (x, y) with x^2 - d*y^2 = 1 and 0 < y <= limit, by trying every y; None beyond the limit."""
    for y in range(1, limit + 1):
        x = isqrt(d * y * y + 1)
        if x * x == d * y * y + 1:
            return x, y
    return None


def period_pell(d):
    """((x, y), period) from the period of sqrt(d) that expansion() finds: the convergent [a0; b0, ..., b(L-2)] folded
    from the right, squared in Z[sqrt(d)] for an odd period length L (Lagrange)."""
    preperiod, period = expansion(0, 1, d, 1)
    assert preperiod == (isqrt(d),), (d, preperiod)
    terms = preperiod + period[:-1]
    x = Fraction(terms[-1])
    for term in reversed(terms[:-1]):
        x = term + 1 / x
    h, k = x.numerator, x.denominator
    return ((h, k) if len(period) % 2 == 0 else (h * h + d * k * k, 2 * h * k)), period


def closed_pell(n):
    """(d, solution, period of sqrt(d)) for the classical radicands n^2 + 1 (L = 1), n^2 + 2, n^2 - 1, n^2 + n
    (L = 2) and n^2 - 2 (L = 4), n >= 3."""
    return ((n * n + 1, (2 * n * n + 1, 2 * n), (2 * n,)),
            (n * n + 2, (n * n + 1, n), (n, 2 * n)),
            (n * n - 1, (n, 1), (1, 2 * n - 2)),
            (n * n + n, (2 * n + 1, 2), (2, 2 * n)),
            (n * n - 2, (n * n - 1, n), (1, n - 2, 1, 2 * n - 2)))


def seeded_expansion_args(seed, count):
    """count seeded arguments (p, q, d, r) whose periods stay within some hundreds of terms: small q, r and d (the
    discriminant of the number, 4*r^2*q^2*d over a square, bounds the period), p sometimes far from 0 (which moves
    only a0). Of every nine, quadratic() builds two: one from a long preperiod of short terms before a short period,
    a number of a small discriminant whose arguments are long, and one purely periodic, which random arguments
    rarely give."""
    rng = random.Random(seed)
    for case in range(count):
        if case % 9 in (0, 4):
            period = (1, 1)
            while not primitive(period):
                period = tuple(rng.randint(1, 9) for _ in range(rng.randint(1, 5)))
            if case % 9 == 4:
                yield quadratic((), period)
                continue
            preperiod = (rng.randint(-50, 50),) + tuple(rng.randint(1, 9) for _ in range(rng.randint(9, 30)))
            if preperiod[-1] == period[-1]:
                preperiod = preperiod[:-1] + (preperiod[-1] % 9 + 1,)  # Still in 1..9, and another term.
            yield quadratic(preperiod, period)
            continue
        p = rng.randint(-60, 60) if case % 5 else rng.randint(-10**30, 10**30)
        q = rng.choice((-1, 1)) * rng.randint(1, 20)
        r = rng.choice((-1, 1)) * rng.randint(1, 30)
        yield p, q, rng.choice(NON_SQUARES[:100]), r


def seeded_rationals(seed, count):
    """count seeded exact rationals, ints among them, of up to 400 bits and either sign."""
    rng = random.Random(seed)
    for case in range(count):
        bits = rng.choice((4, 16, 64, 400))
        numerator = rng.randint(-(1 << bits), 1 << bits)
        yield numerator if case % 6 == 0 else Q(numerator, rng.randint(1, 1 << bits))


class QuadraticIrrationalTests(support.FamilyTestCase):
    FAMILY = family

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        cases = [(EXPAND, expansion_args(*args), PCF(preperiod, period), ())
                 for args, preperiod, period in CARD_EXPANSIONS + BY_HAND_EXPANSIONS]
        cases += [(CONVERGENTS, {"value": value}, convergents, ())
                  for value, convergents in CARD_CONVERGENTS + BY_HAND_CONVERGENTS]
        cases += [(PELL, {"d": d}, solution, (period,)) for d, solution, period in CARD_PELL + CLASSICAL_PELL]
        for operation, arguments, value, evidence in cases:
            with self.subTest(operation=operation, arguments=arguments):
                if operation == EXPAND:  # The worked values, against the references.
                    self.assertEqual(expansion(**arguments), (value.preperiod, value.period))
                elif operation == CONVERGENTS:
                    self.assertEqual(folded(arguments["value"]), value)
                else:
                    self.assertEqual(period_pell(arguments["d"]), (value, evidence[0]))
                    self.assertEqual(value[0] ** 2 - arguments["d"] * value[1] ** 2, 1)
                result = support.run(operation, **arguments)
                self.assertEqual(result.value, value)
                self.assertEqual(result.request.version, 6)
                self.assertEqual(result.steps[0].rule, "v6." + operation)
                self.assertEqual(result.steps[0].outputs, (value, evidence))
                self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
                self.assertTrue(mf.verify(result).verified)
                self.roundtrip(result)
                self.assertEqual(support.value(operation, **arguments), value)
        # Every convergent is a Rational, those of denominator 1 included; Pell's solution is a pair of ints.
        self.assertTrue(all(type(item) is Q for item in support.run(CONVERGENTS, value=Q(415, 93)).value))
        self.assertEqual([type(item) for item in support.run(PELL, d=13).value], [int, int])
        for d in range(2, 30):  # Brute force confirms the small solutions (y = 1,820 for d = 29 is the largest).
            if isqrt(d) ** 2 != d:
                with self.subTest(d=d):
                    self.assertEqual(support.run(PELL, d=d).value, brute_pell(d, 2_000))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error."""
        # The card's invalid input: d < 2, d a perfect square, q = 0, r = 0, also all at once and next to huge values.
        square = (10**40 + 7) ** 2  # A perfect square of 266 bits.
        for p, q, d, r in ((0, 1, 4, 1), (0, 1, 9, 1), (0, 1, 1, 1), (0, 1, 0, 1), (0, 1, -2, 1), (5, 3, -7, 2),
                           (0, 1, square, 1), (0, 0, 2, 1), (3, 0, 7, -2), (0, 1, 2, 0), (0, 0, 2, 0), (0, 0, 4, 0),
                           (1 << 65_535, 0, 2, 1), (0, 1 << 65_535, 2, 0), (0, 1, (1 << 32_767) ** 2, 1)):
            with self.subTest(args=(p, q, d, r)):
                self.assertInvalidInput(support.run(EXPAND, p=p, q=q, d=d, r=r))
                with self.assertRaises(InvalidInput):
                    produced(EXPAND, p=p, q=q, d=d, r=r)
        for d in (4, 1, 0, -1, -5, 9, square, (1 << 32_767) ** 2):
            with self.subTest(d_bits=d.bit_length(), negative=d < 0):
                self.assertInvalidInput(support.run(PELL, d=d))
                with self.assertRaises(InvalidInput):
                    produced(PELL, d=d)
        # An inexact number, a str, None or a container in each argument in turn, the others valid.
        for name in ("p", "q", "d", "r"):
            for item in INEXACT:
                arguments = expansion_args(0, 1, 2, 1)
                arguments[name] = item
                with self.subTest(name=name, item=item):
                    self.assertInvalidInput(support.run(EXPAND, **arguments))
        for item in INEXACT:
            with self.subTest(item=item):
                self.assertInvalidInput(support.run(PELL, d=item))
                if type(item) is not Q:
                    self.assertInvalidInput(support.run(CONVERGENTS, value=item))
        for item in ("415/93", 1.5, mf.Symbol("x") / 2, (Q(1, 2),), [Q(1, 2)], float("nan")):
            with self.subTest(item=item):
                self.assertInvalidInput(support.run(CONVERGENTS, value=item))
        # The producer checks its inputs itself, as direct calls reach it without run()'s checks; it answers its own
        # operations only. A Rational made without its constructor, holding no exact integers, a zero denominator or
        # integers that are not in lowest terms, is invalid input too.
        for operation, arguments in ((EXPAND, expansion_args(0.5, 1, 2, 1)), (EXPAND, expansion_args(0, True, 2, 1)),
                                     (EXPAND, expansion_args(0, 1, "2", 1)), (EXPAND, expansion_args(0, 1, 2, None)),
                                     (EXPAND, expansion_args(0, 1, Q(2), 1)), (PELL, {"d": 2.0}), (PELL, {"d": True}),
                                     (CONVERGENTS, {"value": True}), (CONVERGENTS, {"value": 0.5}),
                                     (CONVERGENTS, {"value": "1/2"}), (CONVERGENTS, {"value": None})):
            with self.subTest(operation=operation, arguments=arguments), self.assertRaises(InvalidInput):
                produced(operation, **arguments)
        for numerator, denominator in ((2, 4), (0, 2), (1, 0), (1, -2), (1.0, 2), (1, True), (None, 1)):
            broken = support.forged(Q, numerator=numerator, denominator=denominator)
            with self.subTest(numerator=numerator, denominator=denominator), self.assertRaises(InvalidInput):
                produced(CONVERGENTS, value=broken)
        with self.assertRaises(InvalidInput):
            produced(CONVERGENTS, value=object.__new__(Q))  # No field at all.
        with self.assertRaises(UnsupportedOperation), computation(fresh=True):
            family.produce("continued_fraction", {"value": Q(415, 93)})
        # A wrong argument set, or arguments that are no mapping at all, as operations_v6.run receives them.
        for operation, arguments in ((EXPAND, {"p": 0, "q": 1, "d": 2}),
                                     (EXPAND, dict(expansion_args(0, 1, 2, 1), s=0)),
                                     (EXPAND, [("p", 0), ("q", 1), ("d", 2), ("r", 1)]), (PELL, {"n": 2}),
                                     (PELL, {"d": 2, "e": 1}), (CONVERGENTS, {}), (CONVERGENTS, (Q(1, 2),)),
                                     (CONVERGENTS, None)):
            with self.subTest(operation=operation, arguments=arguments):
                self.assertInvalidInput(operations_v6.run(operation, arguments))
        # The MCP area tool reports each at the stage that finds it.
        error = self.assertToolError(support.call(PELL, {"d": 4}), "invalid_input")
        self.assertEqual(error["stage"], "computation")
        error = self.assertToolError(support.call(EXPAND, {"p": 0, "q": 0, "d": 2, "r": 1}), "invalid_input")
        self.assertEqual(error["stage"], "computation")
        for raw in (0.5, 2.0, True, None, [2], "2.0"):
            with self.subTest(raw=raw):
                error = self.assertToolError(support.call(EXPAND, {"p": 0, "q": 1, "d": raw, "r": 1}), "invalid_input")
                self.assertEqual((error["stage"], error["field"]), ("decoding", "arguments.d"))
                error = self.assertToolError(support.call(PELL, {"d": raw}), "invalid_input")
                self.assertEqual((error["stage"], error["field"]), ("decoding", "arguments.d"))
        for raw in (0.5, True, None, [1, 2], "1/0", "x"):
            with self.subTest(raw=raw):
                error = self.assertToolError(support.call(CONVERGENTS, {"value": raw}), "invalid_input")
                self.assertEqual((error["stage"], error["field"]), ("decoding", "arguments.value"))

    def test_resource_limits(self):
        """An integer of 2**65536 and max_work=1 exceed the budget (this family takes no matrix)."""
        huge = support.HUGE
        for name in ("p", "q", "d", "r"):
            arguments = expansion_args(0, 1, 2, 1)
            arguments[name] = -huge if name == "r" else huge
            with self.subTest(name=name):
                self.assertResourceLimit(support.run(EXPAND, **arguments))
                with self.assertRaises(ResourceLimitError):
                    produced(EXPAND, **arguments)
        for value in (huge, -huge, Q(huge, 3), Q(3, huge)):
            with self.subTest(value_bits=max(_fraction(value).numerator.bit_length(),
                                             _fraction(value).denominator.bit_length())):
                self.assertResourceLimit(support.run(CONVERGENTS, value=value))
                with self.assertRaises(ResourceLimitError):
                    produced(CONVERGENTS, value=value)
        self.assertResourceLimit(support.run(PELL, d=huge + 1))
        with self.assertRaises(ResourceLimitError):
            produced(PELL, d=huge + 1)
        # Every operation charges at least two work units, so max_work=1 stops it before any answer.
        one = ComputationLimits(max_work=1)
        for operation, arguments in ((EXPAND, expansion_args(0, 1, 2, 1)), (EXPAND, expansion_args(1, 1, 5, 2)),
                                     (CONVERGENTS, {"value": 5}), (CONVERGENTS, {"value": Q(415, 93)}),
                                     (PELL, {"d": 2}), (PELL, {"d": 61})):
            with self.subTest(operation=operation, arguments=arguments):
                self.assertResourceLimit(support.run(operation, limits=one, **arguments))
                with self.assertRaises(ResourceLimitError):
                    produced(operation, limits=one, **arguments)

    def test_caps(self):
        """The producer refuses with ResourceLimitError exactly the expansions beyond the checker's caps, and admits
        those at them: term bit lengths adding up to max_integer_bits pass and two more bits fail, for
        quadratic_continued_fraction and for the period of pell_fundamental, through run() (the producer, then the
        verifier); the count of terms is checked against the checker's MAX_TERMS and against what the verifier
        admits, which counts the value twice."""
        self.assertEqual(family._MAX_TERMS, MAX_TERMS)
        # sqrt(n^2 - 1) = [n - 1; (1, 2n - 2)], whose term bits are 2*bitlen(n - 1) + 2, and Pell's solution (n, 1).
        for limit, low in ((65_536, 1 << 32_766), (200, 1 << 98)):
            limits = None if limit == 65_536 else ComputationLimits(max_integer_bits=limit)
            for n, admitted in ((low + 1, True), (2 * low + 1, False)):
                d = n * n - 1
                preperiod, period = (n - 1,), (1, 2 * n - 2)
                bits = sum(term.bit_length() for term in preperiod + period)
                self.assertEqual(bits, limit if admitted else limit + 2)
                with self.subTest(limit=limit, bits=bits):
                    expanded = support.run(EXPAND, limits=limits, **expansion_args(0, 1, d, 1))
                    pell = support.run(PELL, d=d, limits=limits)
                    if admitted:
                        self.assertEqual(expanded.value, PCF(preperiod, period))
                        self.assertEqual(pell.steps[0].outputs, ((n, 1), (period,)))
                        self.assertTrue(mf.verify(expanded, limits=limits).verified)
                        self.assertTrue(mf.verify(pell, limits=limits).verified)
                    else:
                        self.assertResourceLimit(expanded)
                        self.assertResourceLimit(pell)
                        with self.assertRaisesRegex(ResourceLimitError, "term bits"):
                            produced(EXPAND, limits=limits, **expansion_args(0, 1, d, 1))
                        with self.assertRaisesRegex(ResourceLimitError, "term bits"):
                            produced(PELL, d=d, limits=limits)
        # The count of terms, preperiod and period together: sqrt(7) has 5 and sqrt(13) has 6.
        with patch.object(family, "_MAX_TERMS", 5):
            self.assertEqual(produced(EXPAND, **expansion_args(0, 1, 7, 1))[0], PCF((2,), (1, 1, 1, 4)))
            self.assertEqual(produced(PELL, d=7), ((8, 3), ((1, 1, 1, 4),)))
            with self.assertRaisesRegex(ResourceLimitError, r"terms \(5\)"):
                produced(EXPAND, **expansion_args(0, 1, 13, 1))
            with self.assertRaisesRegex(ResourceLimitError, r"terms \(5\)"):
                produced(PELL, d=13)
        # sqrt(10**11 + 6) has a period of 371,174 terms. Beyond the default max_integer_bits, the 99,000 terms of the
        # checker's cap refuse it, or the 49,954 terms of a PeriodicContinuedFraction that the verifier admits under the
        # default max_evidence_items (Pell's period is evidence, counted once, so the cap decides for it).
        d = 10**11 + 6
        wide = ComputationLimits(max_integer_bits=400_000, max_evidence_items=400_000)
        bits = ComputationLimits(max_integer_bits=400_000)
        started = time.perf_counter()
        for operation, arguments, limits, message in (
                (EXPAND, expansion_args(0, 1, d, 1), wide, r"terms \(99000\)"),
                (PELL, {"d": d}, wide, r"terms \(99000\)"),
                (EXPAND, expansion_args(0, 1, d, 1), bits, r"max_evidence_items \(49954\)"),
                (PELL, {"d": d}, bits, r"terms \(99000\)"),
                (EXPAND, expansion_args(0, 1, d, 1), None, "term bits"),
                (PELL, {"d": d}, None, "term bits")):
            with self.subTest(operation=operation, limits=limits):
                with self.assertRaisesRegex(ResourceLimitError, message):
                    produced(operation, limits=limits, **arguments)
                self.assertResourceLimit(support.run(operation, limits=limits, **arguments))
        self.assertLess(time.perf_counter() - started, FAST)

    def test_verifier_evidence_boundaries(self):
        """The producer refuses exactly the values whose stored Result the verifier would not admit under the request's
        max_evidence_items, which bounds every item of the Result that the verifier walks (Budget.inspect): the value
        twice, the evidence, the request twice and the Result's own fields with its stored checks. With
        max_evidence_items equal to that count the Result is made, and it and its JSON copy verify under the same
        limits; with one item less the producer refuses, and the verifier leaves the Result made before UNKNOWN for its
        resources. The int and the Rational argument of convergents differ by the two integers of the Rational."""
        for operation, arguments in ((EXPAND, expansion_args(0, 1, 7, 1)), (EXPAND, expansion_args(1, 1, 5, 2)),
                                     (EXPAND, expansion_args(10**40, -1, 2, 3)), (CONVERGENTS, {"value": Q(415, 93)}),
                                     (CONVERGENTS, {"value": Q(5)}), (CONVERGENTS, {"value": -7}), (PELL, {"d": 13}),
                                     (PELL, {"d": 2})):
            with self.subTest(operation=operation, arguments=arguments):
                result = support.run(operation, **arguments)
                walk = Budget()
                walk.inspect(result, evidence=True)  # What verification_v6.verify_result counts.
                at = ComputationLimits(max_evidence_items=walk.evidence_count)
                below = ComputationLimits(max_evidence_items=walk.evidence_count - 1)
                bounded = support.run(operation, limits=at, **arguments)
                self.assertEqual(bounded, result)
                for candidate in (bounded, mf.loads(mf.dumps(bounded))):
                    self.assertTrue(mf.verify(candidate, limits=at).verified)
                    self.assertEqual(support.statuses(mf.verify(candidate, limits=below)),
                                     {"result.resources": CheckStatus.UNKNOWN})
                self.assertResourceLimit(support.run(operation, limits=below, **arguments))
                with self.assertRaisesRegex(ResourceLimitError, "max_evidence_items"):
                    produced(operation, limits=below, **arguments)
        # At the default limits: the 16,654 convergents of F(16656)/F(16655) = [1; 1, ..., 1, 2], which are F(k+1)/F(k)
        # for k = 1, ..., 16653 and the value itself (a Result of 99,997 items), but not the 16,655 of the next ratio.
        fibonacci = [0, 1]
        while len(fibonacci) < 16_658:
            fibonacci.append(fibonacci[-1] + fibonacci[-2])
        ratio = Q(fibonacci[16_656], fibonacci[16_655])
        expected = tuple(Q(fibonacci[k + 1], fibonacci[k]) for k in range(1, 16_654)) + (ratio,)
        started = time.perf_counter()
        admitted = support.run(CONVERGENTS, value=ratio)
        self.assertEqual(admitted.value, expected)
        self.assertTrue(mf.verify(admitted).verified)
        self.assertResourceLimit(support.run(CONVERGENTS, value=Q(fibonacci[16_657], fibonacci[16_656])))
        self.assertLess(time.perf_counter() - started, FAST)

    def test_long_expansions_within_the_caps(self):
        """sqrt(LONG_D): 21,033 terms of 47,605 bits, Pell's solution of 36,152 bits, and the expansion of another
        number of the same field, each equal to what the references give and verified within the default limits, and
        quickly; a budget of 50,000 work units refuses the producer, whose expansion takes some 85,000. Arguments of
        65,536 bits leave expansions far beyond the caps, which the producer refuses in time."""
        expected = expansion(0, 1, LONG_D, 1)
        self.assertEqual((len(expected[0]), len(expected[1])), (1, 21_032))
        other_expected = expansion(5, -3, LONG_D, 7)  # Of discriminant 42^2*LONG_D: another cycle of reduced forms.
        self.assertEqual((len(other_expected[0]), len(other_expected[1])), (3, 21_020))
        # Pell's solution read off the reference's period (Lagrange): the convergent [a0; b0, ..., b(L-2)], the first
        # column of the matrix product of this module, unsquared since L is even.
        x, _, y, _ = _matrix(expected[0] + expected[1][:-1])
        self.assertEqual((x.bit_length(), y.bit_length(), x * x - LONG_D * y * y), (36_152, 36_139, 1))
        started = time.perf_counter()
        expanded = support.run(EXPAND, **expansion_args(0, 1, LONG_D, 1))
        pell = support.run(PELL, d=LONG_D)
        other = support.run(EXPAND, **expansion_args(5, -3, LONG_D, 7))
        self.assertLess(time.perf_counter() - started, FAST)
        self.assertEqual((expanded.value.preperiod, expanded.value.period), expected)
        self.assertEqual(pell.steps[0].outputs, ((x, y), (expected[1],)))
        self.assertEqual((other.value.preperiod, other.value.period), other_expected)
        for result in (expanded, pell, other):
            self.assertTrue(mf.verify(result).verified)
        small = ComputationLimits(max_work=50_000)
        self.assertResourceLimit(support.run(EXPAND, limits=small, **expansion_args(0, 1, LONG_D, 1)))
        self.assertResourceLimit(support.run(PELL, d=LONG_D, limits=small))
        rng = random.Random(65_536)
        p, q, d, r = (rng.getrandbits(65_536) | 1 << 65_535 for _ in range(4))
        started = time.perf_counter()
        self.assertResourceLimit(support.run(EXPAND, **expansion_args(p, -q, d, r)))
        self.assertResourceLimit(support.run(PELL, d=d))
        self.assertLess(time.perf_counter() - started, FAST)

    def test_work_depends_on_the_number_only(self):
        """The producer's state form comes from the primitive equation of the number (module docstring), so neither a
        common factor of p, q and r nor a square factor m^2 of d that p and r absorb (p*m, q, d*m^2, r*m) changes the
        integers of the expansion: such a form costs only the gcds, products and divisions of its longer arguments
        more, at most about 140,000 work units for arguments of max_integer_bits bits (the test allows 200,000). With
        the least multiplier of the arguments, the state form that the producer used before, the period of 47,000 ones
        that quadratic() writes with the common factor 2*C (C the lower left entry of the period matrix), the
        heaviest input found within the caps, took 1.79 million work units of the default 2 million, against 0.96
        million in lowest terms; it now takes 1.05 million, and 1.00 million in lowest terms."""
        fixed = 200_000

        def work(p, q, d, r):
            with computation(ComputationLimits(max_work=10**9), fresh=True) as budget:
                value = family.produce(EXPAND, expansion_args(p, q, d, r))[0]
            return value, budget.work

        rng = random.Random(4_096)  # Small numbers in redundant forms, against the reference: signs, both parities.
        for _ in range(60):
            p, q, r = rng.randint(-40, 40), rng.choice((-3, -2, -1, 1, 2, 5)), rng.choice((-6, -2, -1, 1, 3, 4))
            d = rng.choice((2, 3, 5, 6, 7, 10, 13, 21))
            expected = PCF(*expansion(p, q, d, r))
            for m, f in ((1, 1), (2, 1), (3, -1), (6, 5), (1, -4), (12, 3)):
                with self.subTest(p=p, q=q, d=d, r=r, m=m, f=f):
                    self.assertEqual(produced(EXPAND, **expansion_args(p * m * f, q * f, d * m * m, r * m * f)),
                                     (expected, ()))
        for length in (20_000, 47_000):
            period = (1,) * length + (2,)
            built = quadratic((), period)  # p, q and r share the factor 2*C.
            common = gcd(gcd(built[0], built[1]), built[3])
            lowest = (built[0] // common, built[1] // common, built[2], built[3] // common)
            self.assertGreater(common.bit_length(), length // 2)
            value, lowest_work = work(*lowest)
            self.assertEqual(value, PCF((), period))
            forms = {"built": built}
            if length == 20_000:  # Room for more factors within max_integer_bits.
                f = (1 << 37_000) + 1
                m = (1 << 18_800) + 3
                forms["built times f"] = (built[0] * f, built[1] * f, built[2], built[3] * f)
                forms["square factor"] = (lowest[0] * m, lowest[1], lowest[2] * m * m, lowest[3] * m)
            for name, arguments in forms.items():
                with self.subTest(length=length, form=name):
                    self.assertLessEqual(max(abs(item).bit_length() for item in arguments), 65_536)
                    value, form_work = work(*arguments)
                    self.assertEqual(value, PCF((), period))
                    self.assertLessEqual(form_work, lowest_work + fixed)
        # The heaviest input found within the caps, through run() and the verifier, within 1.2 million work units.
        result = support.run(EXPAND, limits=ComputationLimits(max_work=1_200_000), **expansion_args(*built))
        self.assertEqual(result.value, PCF((), period))
        self.assertTrue(mf.verify(result).verified)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen. The
        value of each operation is canonical and its evidence (the period, or nothing) is determined by it, so the
        true claims here are built by hand, backwards from chosen canonical expansions and terms, and from classical
        closed forms; the producer then makes exactly the same claim."""
        rng = random.Random(20_261_112)
        for case in range(200):
            preperiod, period = canonical_expansion(rng)
            p, q, d, r = quadratic(preperiod, period)
            factor = rng.choice((1, 1, -1, 3, -7))  # The same number, in another form.
            arguments = expansion_args(p * factor, q * factor, d, r * factor)
            with self.subTest(case=case, preperiod=preperiod, period=period):
                if case < 40:
                    self.assertEqual(expansion(**arguments), (preperiod, period))
                self.assertAccepts(EXPAND, arguments, PCF(preperiod, period))
                self.assertEqual(produced(EXPAND, **arguments), (PCF(preperiod, period), ()))
        for n in (3, 10, 10**40 + 3, 2**20_000 + 12_345):  # sqrt(n^2 + 1), sqrt(n^2 + 2), sqrt(n^2 - 1).
            for arguments, value in ((expansion_args(0, 1, n * n + 1, 1), PCF((n,), (2 * n,))),
                                     (expansion_args(0, 1, n * n + 2, 1), PCF((n,), (n, 2 * n))),
                                     (expansion_args(0, 1, n * n - 1, 1), PCF((n - 1,), (1, 2 * n - 2)))):
                with self.subTest(bits=n.bit_length(), d=arguments["d"] - n * n):
                    if n < 100:
                        self.assertEqual(expansion(**arguments), (value.preperiod, value.period))
                    self.assertAccepts(EXPAND, arguments, value)
                    self.assertEqual(produced(EXPAND, **arguments), (value, ()))
        for case in range(100):  # Convergents of chosen canonical terms, by the forward recurrence of this test.
            terms = [rng.randint(-10**6, 10**6)] + [rng.randint(1, rng.choice((3, 100, 10**9)))
                                                    for _ in range(rng.randint(0, 30))]
            if len(terms) > 1 and terms[-1] == 1:
                terms[-1] = 2
            p, p_before, q, q_before, convergents = 1, 0, 0, 1, []
            for term in terms:
                p, p_before = term * p + p_before, p
                q, q_before = term * q + q_before, q
                convergents.append(Q(p, q))
            value = Q(p, q) if q > 1 or case % 2 else p  # An int when the expansion is a single term.
            with self.subTest(case=case, terms=len(terms)):
                self.assertAccepts(CONVERGENTS, {"value": value}, tuple(convergents))
                self.assertEqual(produced(CONVERGENTS, value=value), (tuple(convergents), ()))
        for n in (3, 4, 10, 10**40 + 3, 2**20_000 + 12_345):
            for d, solution, period in closed_pell(n):
                with self.subTest(bits=n.bit_length(), d=d - n * n):
                    if n <= 10:
                        self.assertEqual(brute_pell(d, 1_000), solution)
                        self.assertEqual(period_pell(d), (solution, period))
                    self.assertAccepts(PELL, {"d": d}, solution, (period,))
                    self.assertEqual(produced(PELL, d=d), (solution, (period,)))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        # Expansions: every produced claim, corrupted in one respect at a time. Doubling the period, or moving its
        # first term into the preperiod, keeps the number and breaks only the canonical form.
        for index, (p, q, d, r) in enumerate([args for args, _, _ in CARD_EXPANSIONS + BY_HAND_EXPANSIONS]
                                             + list(seeded_expansion_args(5, 30))):
            arguments = expansion_args(p, q, d, r)
            value, evidence = produced(EXPAND, **arguments)
            pre, per = value.preperiod, value.period
            with self.subTest(case=index):
                self.assertAccepts(EXPAND, arguments, value, evidence)
                for preperiod, period in ((pre, per[:-1] + (per[-1] + 1,)), (pre, per * 2),
                                          (pre + per[:1], per[1:] + per[:1]), (pre + per, ()),
                                          ((pre[0] + 1,) + pre[1:] if pre else (0,), per), (list(pre), per),
                                          (pre, list(per)), (pre, (float(per[0]),) + per[1:]),
                                          (pre, (str(per[0]),) + per[1:]), (None, per), (pre, None)):
                    self.assertRejects(EXPAND, arguments, support.forged(PCF, preperiod=preperiod, period=period))
                for other in ((0,), ((),), [], None, (value,), (per,)):
                    self.assertRejects(EXPAND, arguments, value, other)
        sqrt2, true = expansion_args(0, 1, 2, 1), PCF((1,), (2,))
        for value in (support.impostor(true), support.forged(PCF, period=(2,)), support.forged(PCF), ((1,), (2,)),
                      support.forged(PCF, preperiod=(True,), period=(2,)),  # True == 1: the right number, a bool.
                      support.forged(PCF, preperiod=(1,), period=(2, 2)), PCF((1, 2), (1,))):
            with self.subTest(value=value):
                self.assertRejects(EXPAND, sqrt2, value)
        for operation in ("continued_fraction", CONVERGENTS, PELL, "QUADRATIC_CONTINUED_FRACTION", "", None):
            with self.subTest(operation=operation):
                self.assertRejects(operation, sqrt2, true)
        for value in (support.forged(PCF, preperiod=(1,), period=(1,) * 1_000_000 + (2,)),
                      support.forged(PCF, preperiod=(1,), period=(2 ** 70_000,)),
                      support.forged(PCF, preperiod=(1,) * 200_000, period=(2,))):
            with self.subTest(terms=len(value.preperiod) + len(value.period)):
                self.assertRejects(EXPAND, sqrt2, value, may_exhaust=True)
        # Convergents: a missing, an extra or a wrong convergent, an int or a look-alike class for a Rational, a
        # list, evidence, another operation and oversized values.
        lookalike = support.impostor(Q(9, 2))  # A dataclass named Rational with the same fields.
        for source in [value for value, _ in CARD_CONVERGENTS + BY_HAND_CONVERGENTS] + list(seeded_rationals(9, 30)):
            value, evidence = produced(CONVERGENTS, value=source)
            arguments = {"value": source}
            with self.subTest(source=source):
                self.assertAccepts(CONVERGENTS, arguments, value, evidence)
                last = value[-1]
                for other in (value[:-1] or (Q(last.numerator + 1, last.denominator),), value + value[-1:],
                              value[:-1] + (Q(last.numerator + 1, last.denominator),), (last.numerator,) + value[1:],
                              (lookalike,) + value[1:], list(value), value[::-1] if len(value) > 1 else (Q(0), last)):
                    self.assertRejects(CONVERGENTS, arguments, other)
                for other in ((0,), ((),), [], None, (value,)):
                    self.assertRejects(CONVERGENTS, arguments, value, other)
        for operation in ("continued_fraction", EXPAND, PELL, ""):
            self.assertRejects(operation, {"value": Q(415, 93)}, folded(Q(415, 93)))
        for value in ((Q(4),) * 1_000_000, (Q(support.HUGE - 1, 3),), (Q(4), Q(9, 2)) * 10_000):
            with self.subTest(items=len(value)):
                self.assertRejects(CONVERGENTS, {"value": Q(415, 93)}, value, may_exhaust=True)
        # Pell: a wrong solution, the second solution (x^2 - d*y^2 = 1 holds), signs, a swapped pair, a bool, a list,
        # and evidence that is not the minimal period of sqrt(d), or not shaped (period,).
        for d in [d for d, _, _ in CARD_PELL] + [17, 94, 151]:
            (x, y), (period,) = produced(PELL, d=d)
            with self.subTest(d=d):
                self.assertAccepts(PELL, {"d": d}, (x, y), (period,))
                for value in ((x + 1, y), (x, y + 1), (x * x + d * y * y, 2 * x * y), (-x, -y), (x, -y), (y, x),
                              [x, y], (x, y, 1), (float(x), y), (x, str(y)), (x,)):
                    self.assertRejects(PELL, {"d": d}, value, (period,))
                for evidence in ((), (period * 2,), ((isqrt(d),) + period,), (period, period), (list(period),),
                                 [period], (period[:-1] + (period[-1] + 1,),), (period[1:] + period[:1],), None):
                    if evidence != (period,):  # Rotating a period of one term leaves it as it is.
                        self.assertRejects(PELL, {"d": d}, (x, y), evidence)
        self.assertRejects(PELL, {"d": 3}, (2, True), ((1, 2),))  # True == 1: the right pair, a bool.
        self.assertRejects(PELL, {"d": 2}, (3, 2), ((1,) * 1_000_000 + (2,),), may_exhaust=True)
        self.assertRejects(PELL, {"d": 2}, (3, 2), ((2 ** 70_000,),), may_exhaust=True)
        # mf.verify of a tampered Result: the trace holds, the mathematics fails.
        for operation, arguments, value, evidence in (
                (EXPAND, sqrt2, PCF((1,), (3,)), None), (EXPAND, sqrt2, PCF((), (1,)), None),
                (CONVERGENTS, {"value": Q(415, 93)}, (Q(4), Q(9, 2), Q(415, 93)), None),
                (CONVERGENTS, {"value": 5}, (Q(6),), None),
                (PELL, {"d": 2}, (17, 12), None), (PELL, {"d": 2}, (3, 2), ((2, 2),)),
                (PELL, {"d": 13}, (18, 5), None)):
            with self.subTest(operation=operation, value=value, evidence=evidence):
                tampered = support.tampered(support.run(operation, **arguments), value, evidence)
                report = support.statuses(mf.verify(tampered))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        cases = [(EXPAND, expansion_args(*args)) for args, _, _ in CARD_EXPANSIONS]
        cases += [(CONVERGENTS, {"value": value}) for value, _ in CARD_CONVERGENTS]
        cases += [(PELL, {"d": d}) for d, _, _ in CARD_PELL]
        results = [support.run(operation, **arguments) for operation, arguments in cases]
        decoded = [mf.loads(mf.dumps(result)) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, arguments), result, copy in zip(cases, results, decoded):
                with self.subTest(operation=operation, arguments=arguments):
                    self.assertTrue(mf.verify(result).verified)
                    self.assertTrue(mf.verify(copy).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            for operation, arguments in cases[:1] + cases[6:7] + cases[-1:]:
                failed = support.run(operation, **arguments)
                self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX."""
        for operation, arguments, text, latex in (
                (EXPAND, expansion_args(0, 1, 2, 1), "[1; (2)]", r"[1; \overline{2}]"),
                (EXPAND, expansion_args(0, 1, 7, 1), "[2; (1, 1, 1, 4)]", r"[2; \overline{1, 1, 1, 4}]"),
                (EXPAND, expansion_args(1, 1, 5, 2), "[(1)]", r"[\overline{1}]"),
                (EXPAND, expansion_args(0, -1, 2, 1), "[-2; 1, 1, (2)]", r"[-2; 1, 1, \overline{2}]"),
                (CONVERGENTS, {"value": Q(415, 93)}, "(4, 9/2, 58/13, 415/93)",
                 r"\left(4, \frac{9}{2}, \frac{58}{13}, \frac{415}{93}\right)"),
                (CONVERGENTS, {"value": Q(-7, 3)}, "(-3, -2, -7/3)", r"\left(-3, -2, -\frac{7}{3}\right)"),
                (CONVERGENTS, {"value": 5}, "(5)", r"\left(5\right)"),
                (PELL, {"d": 13}, "(649, 180)", r"\left(649, 180\right)")):
            with self.subTest(operation=operation, arguments=arguments):
                decoded = self.roundtrip(support.run(operation, **arguments))
                self.assertEqual(mf.render_text(decoded.value), text)
                self.assertEqual(mf.to_latex(decoded.value), latex)
                self.assertEqual(mf.render_text(decoded), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(PCF((1,), (2,)))["data"],
                         {"type": "v6_periodic_continued_fraction",
                          "preperiod": {"type": "tuple", "items": [{"type": "integer", "value": "1"}]},
                          "period": {"type": "tuple", "items": [{"type": "integer", "value": "2"}]}})
        convergents = mf.to_data(support.run(CONVERGENTS, value=5).value)["data"]
        self.assertEqual(convergents, {"type": "tuple", "items": [{"type": "rational", "numerator": "5",
                                                                   "denominator": "1"}]})
        # A document of a record that is not canonical does not decode.
        document = mf.to_data(PCF((1,), (2,)))
        document["data"]["preperiod"]["items"].append({"type": "integer", "value": "2"})  # [1, 2; (2)]
        with self.assertRaisesRegex(mf.SerializationError, "^Invalid mathematical document$"):
            mf.from_data(document)
        # The shared integer helper renders integers that str() refuses (over 4,300 digits).
        n = 10**5000
        long = self.roundtrip(support.run(EXPAND, **expansion_args(0, 1, n * n + 1, 1)))
        self.assertEqual(mf.render_text(long.value), "[1" + "0" * 5000 + "; (2" + "0" * 5000 + ")]")
        self.assertEqual(mf.to_latex(long.value), "[1" + "0" * 5000 + r"; \overline{2" + "0" * 5000 + "}]")
        pell = self.roundtrip(support.run(PELL, d=n * n + 1))
        self.assertEqual(mf.render_text(pell.value), "(2" + "0" * 9999 + "1, 2" + "0" * 5000 + ")")
        # The long period of sqrt(LONG_D) and Pell's solution of 36,152 bits round-trip as well.
        self.roundtrip(support.run(PELL, d=LONG_D))

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        cases = [(EXPAND, expansion_args(*args)) for args in seeded_expansion_args(7, 30)]
        cases += [(CONVERGENTS, {"value": value}) for value in seeded_rationals(7, 30)]
        cases += [(PELL, {"d": d}) for d in random.Random(7).sample(NON_SQUARES, 30)]
        for index, (operation, arguments) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                structured = support.run(operation, **arguments)
                self.assertTrue(mf.verify(structured).verified)
                self.assertEqual(support.lean_value(operation, **arguments), structured.value)
                self.assertEqual(support.value(operation, **arguments), structured.value)
        # Invalid input: the lean path leaves the call to the structured path, whose error the value function raises.
        for operation, arguments in ((EXPAND, expansion_args(0, 1, 4, 1)), (EXPAND, expansion_args(0, 0, 2, 1)),
                                     (PELL, {"d": 9}), (CONVERGENTS, {"value": 0.5})):
            with self.subTest(operation=operation, arguments=arguments):
                self.assertIs(support.lean_value(operation, **arguments), lean.FALLBACK)
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(operation, **arguments)
                self.assertEqual(raised.exception.code, "invalid_input")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, request, arguments, value in (
                (EXPAND, {"p": 0, "q": 1, "d": 7, "r": 1}, expansion_args(0, 1, 7, 1), PCF((2,), (1, 1, 1, 4))),
                (EXPAND, {"p": 0, "q": -1, "d": 2, "r": 1}, expansion_args(0, -1, 2, 1), PCF((-2, 1, 1), (2,))),
                (CONVERGENTS, {"value": "415/93"}, {"value": Q(415, 93)}, (Q(4), Q(9, 2), Q(58, 13), Q(415, 93))),
                (CONVERGENTS, {"value": 5}, {"value": 5}, (Q(5),)),
                (PELL, {"d": 61}, {"d": 61}, (1766319049, 226153980))):
            with self.subTest(operation=operation, request=request):
                result = self.answer(support.call(operation, request))
                self.assertEqual(result, support.run(operation, **arguments))
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)
        # Integers from 2**53 on travel as strings of decimal digits, as call_values encodes them.
        n = 3**100
        for operation, arguments, value in (
                (EXPAND, expansion_args(0, 1, n * n + 2, 1), PCF((n,), (n, 2 * n))),
                (CONVERGENTS, {"value": Q(3**80 + 1, 3**79)}, folded(Q(3**80 + 1, 3**79))),
                (PELL, {"d": n * n + 1}, (2 * n * n + 1, 2 * n))):
            with self.subTest(operation=operation):
                result = self.answer(support.call_values(operation, **arguments))
                self.assertEqual(result, support.run(operation, **arguments))
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)

    def test_seeded_agreement_with_slow_references(self):
        """At least 300 seeded inputs per operation: the producer succeeds, the checker accepts its claim, and the
        claim equals the slow reference's, evidence included: expansion() for 330 expansions, folded() for 330
        rationals, and for every d < 340 that is no square (321) and 40 seeded d below 20,000, period_pell(), which
        brute_pell() confirms wherever y <= 5,000 (both parities of the period length among them)."""
        shapes = Counter()
        for index, (p, q, d, r) in enumerate(seeded_expansion_args(20_261_113, 330)):
            arguments = expansion_args(p, q, d, r)
            value, evidence = produced(EXPAND, **arguments)
            shapes["empty preperiod" if not value.preperiod else "a0 < 0" if value.preperiod[0] < 0 else "a0 >= 0"] += 1
            shapes["preperiod of 10 or more"] += len(value.preperiod) >= 10
            with self.subTest(case=index, arguments=arguments):
                self.assertEqual(((value.preperiod, value.period), evidence), (expansion(p, q, d, r), ()))
                self.assertIs(support.verdict(family, EXPAND, arguments, value, evidence), True)
        for name in ("empty preperiod", "a0 < 0", "a0 >= 0", "preperiod of 10 or more"):
            self.assertGreaterEqual(shapes[name], 5, shapes)
        sizes = Counter()
        for index, source in enumerate(seeded_rationals(20_261_114, 330)):
            value, evidence = produced(CONVERGENTS, value=source)
            sizes["one" if len(value) == 1 else "2 to 99" if len(value) < 100 else "100 or more"] += 1
            with self.subTest(case=index, source=source):
                self.assertEqual((value, evidence), (folded(source), ()))
                self.assertIs(support.verdict(family, CONVERGENTS, {"value": source}, value, evidence), True)
        for name in ("one", "2 to 99", "100 or more"):  # From single terms to expansions of 100 terms and more.
            self.assertGreaterEqual(sizes[name], 20, sizes)
        radicands = list(NON_SQUARES) + sorted(d for d in random.Random(20_261_115).sample(range(340, 20_000), 45)
                                               if isqrt(d) ** 2 != d)[:40]
        brute_forced, parities = 0, set()
        for d in radicands:
            value, evidence = produced(PELL, d=d)
            solution, period = period_pell(d)
            with self.subTest(d=d):
                self.assertEqual((value, evidence), (solution, (period,)))
                self.assertIs(support.verdict(family, PELL, {"d": d}, value, evidence), True)
                if value[1] <= 5_000:
                    self.assertEqual(brute_pell(d, 5_000), value)
                    brute_forced += 1
                    parities.add(len(period) % 2)
        self.assertEqual(len(radicands), 361)
        self.assertGreaterEqual(brute_forced, 150)
        self.assertEqual(parities, {0, 1})
