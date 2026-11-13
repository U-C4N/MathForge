"""Tests of the independent checker of mathforge.areas.numeric.approximation (mathforge.certify.numeric.approximation)
alone, written before the family's producer exists: every claim is built here, and the checker answers it as the
verifier asks it (support.verdict, in a fresh budget of the default limits unless a test passes others).

The true claims come from references of this module, never from a producer. Each test value comes with an exact
comparison sign(x - t) of its own: Fractions for rationals, the sign of u + v*sqrt(n) by squaring for quadratic
numbers, and sqrt(2) + sqrt(3) = sqrt(5 + 2*sqrt(6)) by squaring once more. For N up to a few hundred, brute force over
every denominator q <= N gives the closest rational (each floor(x*q) by integer search, distances compared exactly, the
tie rule applied) and the Farey neighbours (the largest candidate below x and the smallest above it). For larger N, a
Stern-Brocot walk with batched steps gives the neighbours; a test compares it with brute force on random cases. The
evidence is the card's layout: () for a value that is a rational with denominator <= N, else (l, r).

The checker answers False to every corrupted claim, never with an exception, and refuses oversized forged values and
evidence within FAST seconds under the default budget. ResourceLimitError comes only where the checker's docstring
says so: from an argument beyond the budget, which run() and the verifier refuse before any checker runs.
"""

from __future__ import annotations

import ast
import dataclasses
from fractions import Fraction
import inspect
import random
import time
from collections.abc import Callable

import mathforge as mf
from mathforge.areas.numeric import approximation as family
from mathforge.certify.numeric import approximation as checker
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Sqrt
from mathforge.registry import Argument, OperationSpec

from .. import support

Q = mf.Rational
OP = "rational_approximation"
FAST = 3.0  # Seconds: an oversized forgery is refused, and a full-size claim verified, well within this.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets, "
         "decimal digits and best rational approximations")
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
BIG = 2 ** 64

Compare = Callable[[Fraction], int]  # sign(x - t)


# ---------------------------------------------------------------------------------------------------------------
# Exact references: every true claim of the tests comes from these.

def sign(number: Fraction | int) -> int:
    return (number > 0) - (number < 0)


def quadratic_sign(u: Fraction, v: Fraction, n: int) -> int:
    """The sign of u + v*sqrt(n), n >= 0, by squaring when u and v have opposite signs."""
    su, sv = sign(u), sign(v)
    if sv == 0 or n == 0:
        return su
    if su == 0 or su == sv:
        return sv
    return su * sign(u * u - v * v * n)


def rational_compare(x: Fraction) -> Compare:
    return lambda t: sign(x - t)


def quadratic_compare(a: Fraction, b: Fraction, n: int) -> Compare:
    """x = a + b*sqrt(n)."""
    return lambda t: quadratic_sign(Fraction(a) - t, Fraction(b), n)


def sqrt2_plus_sqrt3(t: Fraction) -> int:
    """sign(sqrt(2) + sqrt(3) - t): x > 0, and for t > 0, x > t iff x**2 = 5 + 2*sqrt(6) > t**2."""
    return 1 if t <= 0 else quadratic_sign(5 - t * t, Fraction(2), 6)


def floor_times(compare: Compare, q: int) -> int:
    """floor(x*q): the largest integer p with p/q <= x, by exponential and binary search."""
    if compare(Fraction(0)) >= 0:
        low, high = 0, 1
        while compare(Fraction(high, q)) >= 0:
            low, high = high, 2 * high
    else:
        low, high = -1, 0
        while compare(Fraction(low, q)) < 0:
            low, high = 2 * low, low
    while high - low > 1:
        middle = (low + high) // 2
        if compare(Fraction(middle, q)) >= 0:
            low = middle
        else:
            high = middle
    return low


def closer(compare: Compare, s: Fraction, t: Fraction) -> int:
    """sign(|x - s| - |x - t|), exactly."""
    a, b = compare(s), compare(t)
    if a == 0 or b == 0:
        return (a != 0) - (b != 0)
    if a == b:  # Both on the same side of x.
        return a * sign(t - s)
    return a * compare((s + t) / 2)  # x between them: |x - s| - |x - t| = +-(2x - s - t).


def better(compare: Compare, s: Fraction, t: Fraction) -> bool:
    """Whether s beats t by the card's rule: closer, then a smaller denominator, then smaller."""
    order = closer(compare, s, t)
    if order:
        return order < 0
    return (s.denominator, s) < (t.denominator, t)


def brute_force(compare: Compare, n: int) -> tuple[Fraction, tuple[Fraction, ...]]:
    """(the closest rational with denominator <= n, the card's evidence) over every denominator q <= n."""
    best, below, above, exact = None, None, None, None
    for q in range(1, n + 1):
        p = floor_times(compare, q)
        for s in (Fraction(p, q), Fraction(p + 1, q)):
            if best is None or better(compare, s, best):
                best = s
            side = compare(s)
            if side == 0:
                exact = s
            elif side > 0 and (below is None or s > below):
                below = s
            elif side < 0 and (above is None or s < above):
                above = s
        low = Fraction(p - 1, q)  # p/q may be x itself; then (p - 1)/q is the candidate below.
        if compare(low) > 0 and (below is None or low > below):
            below = low
    if exact is not None:
        return best, ()
    return best, (below, above)


def stern_brocot(compare: Compare, n: int) -> tuple[Fraction, tuple[Fraction, ...]]:
    """The same pair by a Stern-Brocot walk whose steps toward one side are taken in a batch (binary search on their
    number): for N far beyond brute force. The closest of the neighbours by the exact distance rule."""
    a = floor_times(compare, 1)
    if compare(Fraction(a)) == 0:
        return Fraction(a), ()
    low, high = (a, 1), (a + 1, 1)

    def walk(start: tuple[int, int], toward: tuple[int, int], wanted: int) -> tuple[tuple[int, int], Fraction | None]:
        """The farthest step start + k*toward (k >= 0, denominator <= n) on the wanted side of x, and x itself when
        the next step lands on it."""
        top = (n - start[1]) // toward[1]
        step = lambda k: Fraction(start[0] + k * toward[0], start[1] + k * toward[1])  # noqa: E731
        good, bad = 0, top + 1
        while bad - good > 1:
            middle = (good + bad) // 2
            if compare(step(middle)) == wanted:
                good = middle
            else:
                bad = middle
        hit = step(bad) if bad <= top and compare(step(bad)) == 0 else None
        return (start[0] + good * toward[0], start[1] + good * toward[1]), hit

    while True:
        new_low, hit = walk(low, high, 1)
        if hit is not None:
            return hit, ()
        new_high, hit = walk(high, new_low, -1)
        if hit is not None:
            return hit, ()
        if (new_low, new_high) == (low, high):
            break
        low, high = new_low, new_high
    left, right = Fraction(*low), Fraction(*high)
    return (left if better(compare, left, right) else right), (left, right)


def q(value: Fraction | int) -> Q:
    value = Fraction(value)
    return Q(value.numerator, value.denominator)


def claim(compare: Compare, n: int, reference=brute_force) -> tuple[Q, tuple[Q, ...]]:
    best, evidence = reference(compare, n)
    return q(best), tuple(q(item) for item in evidence)


def args_of(value: object, n: object) -> dict[str, object]:
    return {"value": value, "max_denominator": n}


def sqrt(value: object) -> Sqrt:
    """A Sqrt node of a rational, never simplified."""
    return Sqrt(q(value))


def forged_rational(numerator: object, denominator: object = 1, *, missing: str = "") -> Q:
    """A Rational made with object.__new__: its fields may break lowest terms, a positive denominator or exactness,
    and the field named by missing stays unset."""
    value = object.__new__(Q)
    for name, item in (("numerator", numerator), ("denominator", denominator)):
        if name != missing:
            object.__setattr__(value, name, item)
    return value


class SubRational(Q):
    """A Rational subclass: equal to the right number but no Rational of the model."""


FakeRational = dataclasses.make_dataclass("Rational", ["numerator", "denominator"], frozen=True)  # Another module.
SQRT2 = sqrt(2)
MINUS_SQRT2 = Mul((Q(-1), sqrt(2)))
ROOT2 = mf.RealAlgebraicRoot((-2, 0, 1), 1)
MINUS_ROOT2 = mf.RealAlgebraicRoot((-2, 0, 1), 0)

# (label, value, its exact comparison): irrational values of every kind of the card.
IRRATIONAL = (
    ("sqrt(2)", SQRT2, quadratic_compare(0, 1, 2)),
    ("-sqrt(2)", MINUS_SQRT2, quadratic_compare(0, -1, 2)),
    ("root sqrt(2)", ROOT2, quadratic_compare(0, 1, 2)),
    ("root -sqrt(2)", MINUS_ROOT2, quadratic_compare(0, -1, 2)),
    ("sqrt(2)**3", Pow(sqrt(2), 3), quadratic_compare(0, 2, 2)),
    ("golden ratio", Mul((Q(1, 2), Add((Q(1), sqrt(5))))), quadratic_compare(Fraction(1, 2), Fraction(1, 2), 5)),
    ("root golden ratio", mf.RealAlgebraicRoot((-1, -1, 1), 1), quadratic_compare(Fraction(1, 2), Fraction(1, 2), 5)),
    ("sqrt(2/9) - 3", Add((sqrt(Fraction(2, 9)), Q(-3))), quadratic_compare(-3, Fraction(1, 3), 2)),
    ("sqrt(2) + sqrt(3)", Add((sqrt(2), sqrt(3))), sqrt2_plus_sqrt3),
    ("root sqrt(2) + sqrt(3)", mf.RealAlgebraicRoot((1, 0, -10, 0, 1), 3), sqrt2_plus_sqrt3),
    ("sqrt(1/1000)", sqrt(Fraction(1, 1000)), quadratic_compare(0, Fraction(1, 100), 10)),
)


class ApproximationCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertFast(self, args: object, value: object, evidence: object, *, accept: bool = False) -> None:
        started = time.perf_counter()
        if accept:
            self.assertAccepts(OP, args, value, evidence)
        else:
            self.assertRejects(OP, args, value, evidence)
        self.assertLess(time.perf_counter() - started, FAST)

    def test_family_declares_the_card(self):
        expected = OperationSpec(
            OP, "numeric",
            (Argument("value", "Expression | int", None, "scalar"), Argument("max_denominator", "int", None, "int")),
            "Rational", "Find the closest rational number with a bounded denominator to an exact real number.",
            SCOPE, version=6)
        self.assertEqual(family.SPECS, (expected,))
        self.assertEqual(family.TYPES, {})
        self.assertEqual(family.CHECKER, "mathforge.certify.numeric.approximation")
        self.assertIs(support.checker(family), checker)
        self.assertEqual(checker.MAX_DENOMINATOR, family.MAX_DENOMINATOR)
        self.assertEqual(checker.MAX_DENOMINATOR, 2 ** 64)
        self.assertEqual(checker.ROOT, (mf.RealAlgebraicRoot.__module__, mf.RealAlgebraicRoot.__qualname__))

    # -----------------------------------------------------------------------------------------------------------
    # References

    def test_references_agree(self):
        """Brute force and the Stern-Brocot walk give the same claims on random rationals and quadratic numbers."""
        rng = random.Random(9)
        for _ in range(60):
            n = rng.randint(1, 40)
            x = Fraction(rng.randint(-500, 500), rng.randint(1, 80))
            with self.subTest(x=x, n=n):
                self.assertEqual(brute_force(rational_compare(x), n), stern_brocot(rational_compare(x), n))
        for _ in range(30):
            n = rng.randint(1, 40)
            compare = quadratic_compare(Fraction(rng.randint(-9, 9), rng.randint(1, 5)),
                                        Fraction(rng.choice((-3, -1, 1, 2)), rng.randint(1, 4)), rng.choice((2, 3, 7)))
            with self.subTest(n=n):
                self.assertEqual(brute_force(compare, n), stern_brocot(compare, n))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_accepts_the_known_values_of_the_card(self):
        known = (
            (SQRT2, 10, Q(7, 5), quadratic_compare(0, 1, 2)),
            (Q(314159265, 100000000), 1000, Q(355, 113), rational_compare(Fraction(314159265, 100000000))),
            (Q(1, 3), 2, Q(1, 2), rational_compare(Fraction(1, 3))),
            (Q(1, 2), 1, Q(0), rational_compare(Fraction(1, 2))),
            (Q(3, 7), 7, Q(3, 7), rational_compare(Fraction(3, 7))),
            (MINUS_SQRT2, 10, Q(-7, 5), quadratic_compare(0, -1, 2)),
        )
        for value, n, expected, compare in known:
            with self.subTest(value=value, n=n):
                best, evidence = claim(compare, n)
                self.assertEqual(best, expected)
                self.assertIs(type(best), Q)
                self.assertAccepts(OP, args_of(value, n), best, evidence)
        self.assertAccepts(OP, args_of(Q(3, 7), 7), Q(3, 7), ())
        self.assertAccepts(OP, args_of(SQRT2, 10), Q(7, 5), (Q(7, 5), Q(10, 7)))
        self.assertAccepts(OP, args_of(Q(1, 2), 1), Q(0), (Q(0), Q(1)))

    def test_accepts_irrational_values_of_every_kind(self):
        for label, value, compare in IRRATIONAL:
            for n in (1, 2, 3, 7, 10, 12, 50, 113, 400):
                with self.subTest(label, n=n):
                    best, evidence = claim(compare, n)
                    self.assertEqual(len(evidence), 2)
                    self.assertAccepts(OP, args_of(value, n), best, evidence)

    def test_accepts_random_rationals(self):
        """ints and Rationals, inside and outside the Farey sequence of order N, against brute force."""
        rng = random.Random(2026)
        for _ in range(150):
            n = rng.randint(1, 60)
            x = Fraction(rng.randint(-2000, 2000), rng.randint(1, 120))
            value = x.numerator if x.denominator == 1 and rng.random() < 0.5 else q(x)
            with self.subTest(x=x, n=n):
                best, evidence = claim(rational_compare(x), n)
                self.assertEqual(evidence == (), x.denominator <= n)
                self.assertAccepts(OP, args_of(value, n), best, evidence)

    def test_accepts_the_special_cases(self):
        cases = (
            # Values that are rationals with denominator <= N: evidence ().
            ("an int", 5, 3, Q(5), ()),
            ("a negative int", -4, 1, Q(-4), ()),
            ("zero", 0, 1, Q(0), ()),
            ("Rational zero", Q(0), 9, Q(0), ()),
            ("an integral Rational", Q(6), 1, Q(6), ()),
            ("a long int", 2 ** 60_000 + 1, 5, Q(2 ** 60_000 + 1), ()),
            ("a rational root", mf.RealAlgebraicRoot((-1, 2), 0), 2, Q(1, 2), ()),
            ("a rational root of a cubic", mf.RealAlgebraicRoot((0, -1, 0, 4), 2), 5, Q(1, 2), ()),
            ("Sqrt(4)", sqrt(4), 1, Q(2), ()),
            ("Sqrt(0)", sqrt(0), 1, Q(0), ()),
            ("sqrt(2)*sqrt(2)/3", Mul((sqrt(2), sqrt(2), Q(1, 3))), 3, Q(2, 3), ()),
            ("(sqrt(3) + 1)*(sqrt(3) - 1)", Mul((Add((sqrt(3), Q(1))), Add((sqrt(3), Q(-1))))), 1, Q(2), ()),
            # A rational expression outside the Farey sequence: (l, r).
            ("sqrt(2)*sqrt(2)/3 at N = 2", Mul((sqrt(2), sqrt(2), Q(1, 3))), 2, Q(1, 2), (Q(1, 2), Q(1))),
            # Ties: the smaller denominator, then the smaller rational.
            ("1/2 at N = 1", Q(1, 2), 1, Q(0), (Q(0), Q(1))),
            ("-1/2 at N = 1", Q(-1, 2), 1, Q(-1), (Q(-1), Q(0))),
            ("5/2 at N = 1", Q(5, 2), 1, Q(2), (Q(2), Q(3))),
            ("1/4 at N = 2", Q(1, 4), 2, Q(0), (Q(0), Q(1, 2))),
            ("3/4 at N = 2", Q(3, 4), 2, Q(1), (Q(1, 2), Q(1))),
            ("-3/4 at N = 2", Q(-3, 4), 2, Q(-1), (Q(-1), Q(-1, 2))),
            ("5/12 at N = 4", Q(5, 12), 4, Q(1, 2), (Q(1, 3), Q(1, 2))),  # The smaller denominator is on the right.
            ("root 3/4 at N = 2", mf.RealAlgebraicRoot((-3, 4), 0), 2, Q(1), (Q(1, 2), Q(1))),
            ("root 1/2 at N = 1", mf.RealAlgebraicRoot((-1, 2), 0), 1, Q(0), (Q(0), Q(1))),
            ("sqrt(1/16) at N = 2", sqrt(Fraction(1, 16)), 2, Q(0), (Q(0), Q(1, 2))),
            ("sqrt(9/16) at N = 2", sqrt(Fraction(9, 16)), 2, Q(1), (Q(1, 2), Q(1))),
            ("sqrt(1/4) at N = 1", sqrt(Fraction(1, 4)), 1, Q(0), (Q(0), Q(1))),
            # Near zero and below it.
            ("-1/1000 at N = 10", Q(-1, 1000), 10, Q(0), (Q(-1, 10), Q(0))),
            ("-sqrt(2) at N = 1", MINUS_SQRT2, 1, Q(-1), (Q(-2), Q(-1))),
        )
        for label, value, n, best, evidence in cases:
            with self.subTest(label):
                self.assertAccepts(OP, args_of(value, n), best, evidence)
        # The ties agree with brute force over every candidate.
        for x, n in ((Fraction(1, 2), 1), (Fraction(-1, 2), 1), (Fraction(1, 4), 2), (Fraction(3, 4), 2),
                     (Fraction(5, 12), 4), (Fraction(-3, 4), 2)):
            with self.subTest(x=x, n=n):
                best, evidence = claim(rational_compare(x), n)
                self.assertAccepts(OP, args_of(q(x), n), best, evidence)

    def test_accepts_the_largest_max_denominator(self):
        """N = 2**64 on every kind of value, each verified within FAST seconds."""
        x = Fraction(2 ** 90 + 1, 3 ** 45)  # Denominator above 2**64.
        cases = [("a rational", q(x), rational_compare(x)), ("an inside rational", Q(1, BIG - 1),
                                                             rational_compare(Fraction(1, BIG - 1)))]
        cases += [(label, value, compare) for label, value, compare in IRRATIONAL[:7]]
        for label, value, compare in cases:
            with self.subTest(label):
                best, evidence = claim(compare, BIG, stern_brocot)
                self.assertFast(args_of(value, BIG), best, evidence, accept=True)

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims, which the checker rejects

    def test_rejects_wrong_values(self):
        cases = (
            ("the farther neighbour", SQRT2, 10, Q(10, 7), (Q(7, 5), Q(10, 7))),
            ("not a neighbour", SQRT2, 10, Q(3, 2), (Q(7, 5), Q(10, 7))),
            ("the answer of a smaller N", SQRT2, 12, Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("tie to the larger denominator", Q(1, 4), 2, Q(1, 2), (Q(0), Q(1, 2))),
            ("tie to the larger denominator on the left", Q(5, 12), 4, Q(1, 3), (Q(1, 3), Q(1, 2))),
            ("tie to the larger of equal denominators", Q(1, 2), 1, Q(1), (Q(0), Q(1))),
            ("a rational claimed as another", Q(3, 7), 7, Q(2, 5), ()),
            ("an irrational claimed exact", SQRT2, 10, Q(7, 5), ()),
            ("a root claimed exact", ROOT2, 10, Q(7, 5), ()),
            ("a root at the wrong index", MINUS_ROOT2, 10, Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("-sqrt(2) as sqrt(2)", MINUS_SQRT2, 10, Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a rational expression claimed as another", Mul((sqrt(2), sqrt(2), Q(1, 3))), 3, Q(1, 3), ()),
            ("a denominator above N", SQRT2, 10, Q(17, 12), (Q(17, 12), Q(10, 7))),
            ("the right number from the wrong side", Q(3, 7), 6, Q(3, 7), (Q(2, 5), Q(1, 2))),
        )
        for label, value, n, best, evidence in cases:
            with self.subTest(label):
                self.assertRejects(OP, args_of(value, n), best, evidence)

    def test_rejects_noncanonical_values_that_are_mathematically_right(self):
        cases = (
            ("an int 0 for Rational(0)", Q(1, 2), 1, 0, (Q(0), Q(1))),
            ("an int 5 for Rational(5)", 5, 3, 5, ()),
            ("an int in the evidence", Q(1, 2), 1, Q(0), (0, Q(1))),
            ("ints in the evidence", Q(1, 2), 1, Q(0), (0, 1)),
            ("not in lowest terms", SQRT2, 10, forged_rational(14, 10), (Q(7, 5), Q(10, 7))),
            ("evidence not in lowest terms", SQRT2, 10, Q(7, 5), (forged_rational(14, 10), Q(10, 7))),
            ("a negative denominator", SQRT2, 10, forged_rational(-7, -5), (Q(7, 5), Q(10, 7))),
            ("a zero denominator", Q(0), 1, forged_rational(0, 0), ()),
            ("a subclass", SQRT2, 10, SubRational(7, 5), (Q(7, 5), Q(10, 7))),
            ("a subclass in the evidence", SQRT2, 10, Q(7, 5), (Q(7, 5), SubRational(10, 7))),
            ("a Fraction", SQRT2, 10, Fraction(7, 5), (Q(7, 5), Q(10, 7))),
            ("swapped evidence", SQRT2, 10, Q(7, 5), (Q(10, 7), Q(7, 5))),
        )
        for label, value, n, best, evidence in cases:
            with self.subTest(label):
                self.assertRejects(OP, args_of(value, n), best, evidence)

    def test_rejects_impostors_and_forged_records(self):
        root = ROOT2
        cases = (
            ("an impostor Rational value", args_of(SQRT2, 10), FakeRational(7, 5), (Q(7, 5), Q(10, 7))),
            ("an impostor Rational in the evidence", args_of(SQRT2, 10), Q(7, 5), (FakeRational(7, 5), Q(10, 7))),
            ("an impostor Rational argument", args_of(FakeRational(3, 7), 7), Q(3, 7), ()),
            ("an impostor root", args_of(support.impostor(root), 10), Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a forged Rational without a numerator", args_of(Q(3, 7), 7), forged_rational(3, 7, missing="numerator"),
             ()),
            ("a forged Rational without a denominator", args_of(Q(3, 7), 7),
             forged_rational(3, 7, missing="denominator"), ()),
            ("a forged argument not in lowest terms", args_of(forged_rational(6, 14), 7), Q(3, 7), ()),
            ("a forged argument with a bool", args_of(forged_rational(True, 2), 1), Q(0), (Q(0), Q(1))),
            ("a forged root, not primitive", args_of(support.forged(mf.RealAlgebraicRoot,
                                                                    integer_coefficients=(-4, 0, 2), real_index=1), 10),
             Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a forged root, not square-free", args_of(support.forged(mf.RealAlgebraicRoot,
                                                                      integer_coefficients=(4, 0, -4, 0, 1),
                                                                      real_index=1), 10),
             Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a forged root, index out of range", args_of(support.forged(mf.RealAlgebraicRoot,
                                                                         integer_coefficients=(-2, 0, 1),
                                                                         real_index=2), 10),
             Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a forged root, negative leading coefficient", args_of(support.forged(
                mf.RealAlgebraicRoot, integer_coefficients=(2, 0, -1), real_index=1), 10), Q(7, 5),
             (Q(7, 5), Q(10, 7))),
            ("a forged root without coefficients", args_of(support.forged(mf.RealAlgebraicRoot, real_index=1), 10),
             Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a forged Sqrt of a negative", args_of(support.forged(Sqrt, radicand=Q(-2)), 10), Q(7, 5),
             (Q(7, 5), Q(10, 7))),
            ("a forged Add of one term", args_of(support.forged(Add, terms=(SQRT2,)), 10), Q(7, 5),
             (Q(7, 5), Q(10, 7))),
            ("a Symbol inside", args_of(Add((mf.Symbol("x"), SQRT2)), 10), Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a root inside an expression", args_of(Add((ROOT2, Q(0, 1), Q(1))), 10), Q(12, 5), (Q(12, 5), Q(17, 7))),
        )
        for label, args, best, evidence in cases:
            with self.subTest(label):
                self.assertRejects(OP, args, best, evidence)

    def test_rejects_inexact_numbers_and_lists(self):
        cases = (
            ("a bool max_denominator", args_of(Q(1, 2), True), Q(0), (Q(0), Q(1))),
            ("a float max_denominator", args_of(SQRT2, 10.0), Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a str max_denominator", args_of(SQRT2, "10"), Q(7, 5), (Q(7, 5), Q(10, 7))),
            ("a bool value", args_of(True, 1), Q(1), ()),
            ("a float value", args_of(1.5, 1), Q(1), (Q(1), Q(2))),
            ("a str value", args_of("1/2", 1), Q(0), (Q(0), Q(1))),
            ("None as the value", args_of(None, 1), Q(0), ()),
            ("a bool claim", args_of(1, 1), True, ()),
            ("a float claim", args_of(SQRT2, 10), 1.4, (Q(7, 5), Q(10, 7))),
            ("a str claim", args_of(SQRT2, 10), "7/5", (Q(7, 5), Q(10, 7))),
            ("bools in the evidence", args_of(Q(1, 2), 1), Q(0), (False, True)),
            ("floats in the evidence", args_of(SQRT2, 10), Q(7, 5), (1.4, Q(10, 7))),
            ("a forged Rational with float fields", args_of(SQRT2, 10), forged_rational(7.0, 5.0),
             (Q(7, 5), Q(10, 7))),
            ("a forged Rational with bool fields", args_of(Q(1, 2), 1), forged_rational(True, True), (Q(0), Q(1))),
        )
        for label, args, best, evidence in cases:
            with self.subTest(label):
                self.assertRejects(OP, args, best, evidence)
        self.assertRejects(OP, args_of(SQRT2, 10), Q(7, 5), [Q(7, 5), Q(10, 7)])
        self.assertRejects(OP, args_of(Q(3, 7), 7), Q(3, 7), [])
        self.assertRejects(OP, args_of(SQRT2, 10), Q(7, 5), ([Q(7, 5), Q(10, 7)],))

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        args, best, good = args_of(SQRT2, 10), Q(7, 5), (Q(7, 5), Q(10, 7))
        self.assertAccepts(OP, args, best, good)
        for label, evidence in (("missing", ()), ("one", (Q(7, 5),)), ("extra", good + (Q(1),)),
                                ("nested", (good,)), ("a str", ("farey", Q(7, 5), Q(10, 7))),
                                ("None", (None, None)), ("strs", ("7/5", "10/7"))):
            with self.subTest(label):
                self.assertRejects(OP, args, best, evidence)
        for label, evidence in (("one", (Q(3, 7),)), ("two", (Q(3, 7), Q(3, 7))), ("neighbours", (Q(2, 5), Q(1, 2)))):
            with self.subTest("exact value with " + label):
                self.assertRejects(OP, args_of(Q(3, 7), 7), Q(3, 7), evidence)
        self.assertRejects(OP, args, best, None)

    def test_rejects_forged_evidence_that_is_consistent_but_wrong(self):
        cases = (
            # Farey neighbours of order 10 that do not hold sqrt(2).
            ("neighbours above the value", SQRT2, 10, Q(13, 9), (Q(13, 9), Q(3, 2))),
            ("neighbours below the value", SQRT2, 10, Q(4, 3), (Q(4, 3), Q(7, 5))),
            # Neighbours of a smaller order: a fraction with denominator <= N lies between them.
            ("neighbours of order 2", SQRT2, 10, Q(3, 2), (Q(1), Q(3, 2))),
            ("neighbours of order 11 at N = 12", SQRT2, 12, Q(7, 5), (Q(7, 5), Q(10, 7))),
            # Not neighbours at all: the determinant is not 1.
            ("an interval around the value", SQRT2, 10, Q(7, 5), (Q(7, 5), Q(3, 2))),
            ("a wide interval", SQRT2, 10, Q(1), (Q(1), Q(2))),
            ("determinant -1", SQRT2, 10, Q(7, 5), (Q(10, 7), Q(7, 5))),
            # Neighbours with a denominator above N.
            ("neighbours of order 12 at N = 10", SQRT2, 10, Q(17, 12), (Q(7, 5), Q(17, 12))),
            # The exact value given neighbours, and neighbours of the value at the same N for a root.
            ("an exact value between neighbours", Q(3, 7), 7, Q(3, 7), (Q(2, 5), Q(1, 2))),
            ("root neighbours above", ROOT2, 10, Q(13, 9), (Q(13, 9), Q(3, 2))),
            ("expression neighbours of order 2", Add((sqrt(2), sqrt(3))), 10, Q(3), (Q(3), Q(7, 2))),
        )
        for label, value, n, best, evidence in cases:
            with self.subTest(label):
                self.assertRejects(OP, args_of(value, n), best, evidence)

    def test_rejects_wrong_operations_and_arguments(self):
        args, best, evidence = args_of(SQRT2, 10), Q(7, 5), (Q(7, 5), Q(10, 7))
        for operation in ("digits", "rational_approximations", "Rational_Approximation", "", None, b"x", 1):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, best, evidence)
        for label, bad in (("an extra argument", dict(args, places=3)), ("a missing argument", {"value": SQRT2}),
                           ("a renamed argument", {"value": SQRT2, "N": 10}), ("a list", [SQRT2, 10]),
                           ("max_denominator 0", args_of(SQRT2, 0)), ("max_denominator -10", args_of(SQRT2, -10)),
                           ("max_denominator above 2**64", args_of(SQRT2, BIG + 1)),
                           ("a giant max_denominator", args_of(SQRT2, GIANT))):
            with self.subTest(label):
                self.assertRejects(OP, bad, best, evidence)
        self.assertRejects(OP, args_of(Q(1, 2), 0), Q(0), (Q(0), Q(1)))

    def test_oversized_forgeries_fail_fast(self):
        """Oversized forged values and evidence answer False within FAST seconds under the default budget."""
        for giant in (GIANT, 1 << 10_000_000):  # A shift, which builds the long integer in linear time.
            for label, args, best, evidence in (
                ("a giant claim", args_of(SQRT2, 10), Q(giant + 1, 1) if giant == GIANT else forged_rational(giant + 1),
                 (Q(7, 5), Q(10, 7))),
                ("a giant exact claim", args_of(3, 1), forged_rational(giant + 1), ()),
                ("a giant left neighbour", args_of(SQRT2, 10), Q(7, 5), (forged_rational(-giant), Q(10, 7))),
                ("a giant right neighbour", args_of(SQRT2, 10), Q(7, 5), (Q(7, 5), forged_rational(giant))),
                ("a giant denominator", args_of(SQRT2, 10), forged_rational(7, giant + 1), (Q(7, 5), Q(10, 7))),
            ):
                with self.subTest(label, bits=giant.bit_length()):
                    self.assertFast(args, best, evidence)
        # Long but admissible forged neighbours whose determinant is 1, against each kind of value.
        far = 2 ** 65_000
        for label, value in (("an int", 3), ("a Rational", Q(7, 2)), ("an expression", SQRT2), ("a root", ROOT2),
                             ("a negative root", MINUS_ROOT2), ("sqrt(2) + sqrt(3)", Add((sqrt(2), sqrt(3))))):
            for evidence in ((Q(far), Q(far + 1)), (Q(-far - 1), Q(-far))):
                for best in evidence:
                    with self.subTest(label, best=best.numerator.bit_length()):
                        self.assertFast(args_of(value, 1), best, evidence)
        # A long claim that the value would equal.
        self.assertFast(args_of(SQRT2, 1), Q(far), ())
        self.assertFast(args_of(ROOT2, 1), Q(far), ())
        self.assertFast(args_of(far + 1, 1), Q(far), ())
        # A long true claim at the size bound verifies within FAST seconds as well.
        self.assertFast(args_of(far, 1), Q(far), (), accept=True)
        self.assertFast(args_of(Add((Q(far), sqrt(2))), 1), Q(far + 1), (Q(far + 1), Q(far + 2)), accept=True)

    def test_arguments_beyond_max_integer_bits_exceed_the_budget(self):
        """An argument longer than max_integer_bits raises ResourceLimitError, as run() and the verifier refuse it
        before any checker runs (the checker's docstring): never a verified claim."""
        for label, value in (("an int", support.HUGE), ("a Rational", Q(1, support.HUGE)),
                             ("a Sqrt", Sqrt(Q(support.HUGE)))):
            with self.subTest(label):
                started = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, OP, args_of(value, 1), Q(0), ())
                self.assertLess(time.perf_counter() - started, FAST)

    def test_work_is_charged(self):
        """A check runs inside the budget: a budget far too small refuses an honest claim with ResourceLimitError,
        the default one accepts it."""
        args, best, evidence = args_of(Add((sqrt(2), sqrt(3))), BIG), *claim(sqrt2_plus_sqrt3, BIG, stern_brocot)
        self.assertAccepts(OP, args, best, evidence)
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation(OP, args, best, evidence), True)
        work = budget.work
        self.assertGreater(work, 10)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OP, args, best, evidence, limits=ComputationLimits(max_work=work // 2))

    def test_accepts_exactly_one_claim_in_a_small_universe(self):
        """Review addition: for small N, every claim built from the fractions of denominator <= N + 1 near the value,
        evidence () and every ordered pair (l, u) with each of l and u as the value, is offered to the checker, which
        accepts exactly the brute-force claim, on values of every kind, rational ones and ties included."""
        cases = (
            (Q(1, 2), rational_compare(Fraction(1, 2))), (Q(-3, 4), rational_compare(Fraction(-3, 4))),
            (Q(5, 12), rational_compare(Fraction(5, 12))), (Q(3, 7), rational_compare(Fraction(3, 7))), (2, None),
            (SQRT2, quadratic_compare(0, 1, 2)), (MINUS_ROOT2, quadratic_compare(0, -1, 2)),
            (mf.RealAlgebraicRoot((-1, -1, 1), 0), quadratic_compare(Fraction(1, 2), Fraction(-1, 2), 5)),
            (mf.RealAlgebraicRoot((2, -4, -1, 2), 1), rational_compare(Fraction(1, 2))),
            (Mul((sqrt(2), sqrt(2), Q(1, 4))), rational_compare(Fraction(1, 2))),
            (Add((sqrt(2), sqrt(3))), sqrt2_plus_sqrt3),
        )
        for value, compare in cases:
            compare = compare or rational_compare(Fraction(value))
            base = floor_times(compare, 1)
            for n in range(1, 5):
                expected = claim(compare, n)
                universe = sorted({Fraction(p, d) for d in range(1, n + 2) for p in range((base - 1) * d,
                                                                                          (base + 2) * d + 1)})
                offered = [(q(r), ()) for r in universe]
                offered += [(q(r), (q(l), q(u))) for i, l in enumerate(universe) for u in universe[i + 1:]
                            for r in (l, u)]
                offered += [(q(u), (q(u), q(l))) for i, l in enumerate(universe) for u in universe[i + 1:i + 3]]
                accepted = [item for item in offered if checker.check_operation(OP, args_of(value, n), *item)]
                with self.subTest(value=str(value), n=n):
                    self.assertEqual(accepted, [expected])

    def test_no_power_and_no_producer_code(self):
        """The checker forms no power, imports only the layers that a checker may use, and names no importlib,
        __import__, sys or builtins in its code."""
        tree = ast.parse(inspect.getsource(checker))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertEqual(names & {"pow", "importlib", "__import__", "sys", "builtins", "eval", "exec"}, set())
        code = ast.parse(inspect.getsource(checker).split('"""', 2)[2])
        powers = [node for node in ast.walk(code) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow) and not (isinstance(node, ast.BinOp)
                                                            and isinstance(node.left, ast.Constant))]
        self.assertEqual(powers, [])  # Only the constant 2 ** 64, folded when the module loads.
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imported, {"__future__", "collections.abc", "mathforge.certify.numeric.digits",
                                    "mathforge.certify.upoly", "mathforge.limits", "mathforge.model",
                                    "mathforge.trusted.upoly"})


if __name__ == "__main__":
    import unittest
    unittest.main()
