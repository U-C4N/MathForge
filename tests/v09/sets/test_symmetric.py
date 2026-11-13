"""The symmetric family of mathforge.sets: set_symmetric_difference.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register sets symmetric

Every expected value comes from the card or from the reference of this module, never from the producer. The reference
reads each end as a real of the form s * v^(1/p): a rational (p = 1), a signed square root of a positive rational that
is no square (p = 2) or the real cube root of a rational that is no cube (p = 3). It orders two such reals exactly, by
their signs and then by v1^p2 against v2^p1 (both sides raised to the power p1 * p2, which keeps the order of
nonnegative reals), and evaluates (x in a) != (x in b) at every end and on every open gap between neighbouring ends.
tests/v09/sets/test_symmetric_checker.py tests the checker on its own.

The card names no deferred (Unevaluated) case for this family, and its operation takes no int and no matrix: the
resource tests put an integer of 2**65536 into an end instead, and an oversized matrix in an IntervalSet argument is
invalid_input.

Run this module by name: python -m unittest tests.v09.sets.test_symmetric
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import functools
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean, operations_v6
from mathforge.algebraic import RealAlgebraicRoot, compare_real
from mathforge.areas.sets import symmetric as family
from mathforge.contracts import EmptySet, ErrorInfo, UniversalSet
from mathforge.errors import InvalidInput
from mathforge.intervals import Interval, IntervalSet, equal_interval_sets
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Rational, Reals

from .. import support

OPERATION = "set_symmetric_difference"
WIDE = ComputationLimits(max_work=10 ** 9, max_refinements=10 ** 8, max_nodes=10 ** 8, max_evidence_items=10 ** 8)
STAND_INS = (0.5, True, "1", None)
# Forgeries that operations_v6.run refuses before the producer runs, with execution_error: its budget.inspect of the
# arguments reads every field of a record, and these lack one or hold a float where an int belongs. No decoded
# document can hold them; produce() itself raises InvalidInput for each.
INSPECTED = frozenset({"no intervals field", "no lower field", "a Rational with a float numerator"})

Key = tuple  # ("q", Fraction) for a rational, or ("rad", sign, p, v) for sign * v^(1/p), p in (2, 3), v > 0.


# ---------------------------------------------------------------------------------------------------------------
# The reference: ends as signed radicals of rationals, ordered exactly.

def magnitude(key: Key) -> tuple[int, Fraction, int]:
    """(sign, v, p) of the real sign * v^(1/p) that a key names."""
    if key[0] == "q":
        x = key[1]
        return (x > 0) - (x < 0), abs(x), 1
    return key[1], key[3], key[2]


def compare(left: Key, right: Key) -> int:
    """The order of the reals of two keys, decided exactly."""
    (s, v, p), (t, w, r) = magnitude(left), magnitude(right)
    if s != t:
        return -1 if s < t else 1
    if s == 0:
        return 0
    order = (v ** r > w ** p) - (v ** r < w ** p)  # |x| against |y|, both raised to the power p * r.
    return order * s


def record(key: Key) -> Rational | RealAlgebraicRoot:
    """The end record of a key: k x^2 - m for sqrt(m/k) (index 1 for the positive root), k x^3 - s m for the real cube
    root of s m/k; primitive since m/k is in lowest terms."""
    if key[0] == "q":
        return Rational(key[1].numerator, key[1].denominator)
    _, s, p, v = key
    if p == 2:
        return RealAlgebraicRoot((-v.numerator, 0, v.denominator), 1 if s > 0 else 0)
    return RealAlgebraicRoot((-s * v.numerator, 0, 0, v.denominator), 0)


def key_of(end: object) -> Key:
    """The key of an end record of the reference's kinds."""
    if type(end) is Rational:
        return ("q", Fraction(end.numerator, end.denominator))
    c = end.integer_coefficients
    if len(c) == 2:
        return ("q", Fraction(-c[0], c[1]))
    if len(c) == 3 and c[1] == 0:
        return ("rad", 1 if end.real_index else -1, 2, Fraction(-c[0], c[2]))
    if len(c) == 4 and c[1] == c[2] == 0:
        x = Fraction(-c[0], c[3])
        return ("rad", 1 if x > 0 else -1, 3, abs(x))
    raise AssertionError(f"no key for {end!r}")


Written = list[tuple[Key | None, Key | None, bool, bool]]


def written(value: IntervalSet) -> Written:
    """An IntervalSet as (lower, upper, left_closed, right_closed) tuples of keys, None at an infinite end."""
    return [(None if item.lower is None else key_of(item.lower), None if item.upper is None else key_of(item.upper),
             item.left_closed, item.right_closed) for item in value.intervals]


def holds_point(intervals: Written, x: Key) -> bool:
    for lower, upper, left, right in intervals:
        if (lower is None or compare(lower, x) < (1 if left else 0)) and (
                upper is None or compare(x, upper) < (1 if right else 0)):
            return True
    return False


def holds_gap(intervals: Written, low: Key | None, high: Key | None) -> bool:
    """Whether the open gap (low, high) between two neighbouring ends lies in the set: an interval holds all of it or
    none of it."""
    for lower, upper, _, _ in intervals:
        if (lower is None or (low is not None and compare(lower, low) <= 0)) and (
                upper is None or (high is not None and compare(high, upper) <= 0)):
            return True
    return False


def reference(a: IntervalSet, b: IntervalSet) -> Written:
    """a ^ b as written() writes a set: the truth (x in a) != (x in b) on every end of the inputs and on every open
    gap between neighbouring ends, and the maximal runs of true members."""
    pieces = (written(a), written(b))
    ends: list[Key] = []
    for intervals in pieces:
        for lower, upper, _, _ in intervals:
            for end in (lower, upper):
                if end is not None and all(compare(end, other) for other in ends):
                    ends.append(end)
    ends.sort(key=functools.cmp_to_key(compare))
    bounds = [None, *ends, None]
    members: list[tuple[str, object]] = []  # ("gap", (low, high)) and ("point", key), in increasing order.
    for k in range(len(ends) + 1):
        members.append(("gap", (bounds[k], bounds[k + 1])))
        if k < len(ends):
            members.append(("point", ends[k]))
    truth = []
    for kind, where in members:
        inside = [holds_gap(intervals, *where) if kind == "gap" else holds_point(intervals, where)
                  for intervals in pieces]
        truth.append(inside[0] != inside[1])
    result: Written = []
    k = 0
    while k < len(members):
        if not truth[k]:
            k += 1
            continue
        stop = k
        while stop + 1 < len(members) and truth[stop + 1]:
            stop += 1
        first, last = members[k], members[stop]
        lower, left = (first[1][0], False) if first[0] == "gap" else (first[1], True)
        upper, right = (last[1][1], False) if last[0] == "gap" else (last[1], True)
        result.append((lower, upper, left, right))
        k = stop + 1
    return result


# ---------------------------------------------------------------------------------------------------------------
# Records

def iset(*intervals: Interval) -> IntervalSet:
    return IntervalSet(tuple(intervals))


def q(numerator: int, denominator: int = 1) -> Rational:
    return Rational(numerator, denominator)


def point(x: object) -> Interval:
    return Interval(x, x, True, True)


def forged_set(*intervals: object) -> IntervalSet:
    return support.forged(IntervalSet, intervals=tuple(intervals))


def forged_interval(lower: object, upper: object, left: object = False, right: object = False) -> Interval:
    return support.forged(Interval, lower=lower, upper=upper, left_closed=left, right_closed=right)


def forged_root(coefficients: object, index: object) -> RealAlgebraicRoot:
    return support.forged(RealAlgebraicRoot, integer_coefficients=coefficients, real_index=index)


SQRT2, MINUS_SQRT2 = RealAlgebraicRoot((-2, 0, 1), 1), RealAlgebraicRoot((-2, 0, 1), 0)
SQRT2_CUBIC = RealAlgebraicRoot((0, -2, 0, 1), 2)  # sqrt 2 as the largest root of x^3 - 2x.
CBRT2 = RealAlgebraicRoot((-2, 0, 0, 1), 0)
THREE = RealAlgebraicRoot((-3, 1), 0)  # 3 as the root of x - 3.
EMPTY, WHOLE = IntervalSet(()), IntervalSet((Interval(None, None),))

# The known values of the card (the first four), then special cases: (a, b, value).
KNOWN = (
    (iset(Interval(0, 2, True, True)), iset(Interval(1, 3, True, False)),
     iset(Interval(0, 1, True, False), Interval(2, 3))),
    (iset(Interval(0, 1, True, True)), iset(Interval(0, 1, True, True)), EMPTY),
    (WHOLE, iset(point(0)), iset(Interval(None, 0), Interval(0, None))),
    (iset(Interval(MINUS_SQRT2, SQRT2, True, True)), iset(Interval(0, 2, True, True)),
     iset(Interval(MINUS_SQRT2, 0, True, False), Interval(SQRT2, 2, False, True))),
    (EMPTY, EMPTY, EMPTY),
    (EMPTY, WHOLE, WHOLE),
    (WHOLE, WHOLE, EMPTY),
    (iset(Interval(0, 1)), EMPTY, iset(Interval(0, 1))),
    (iset(Interval(0, 1, True, True)), iset(Interval(1, 2, True, True)),
     iset(Interval(0, 1, True, False), Interval(1, 2, False, True))),  # Touching at a common closed end.
    (iset(Interval(0, 1, True, False)), iset(Interval(1, 2, True, True)), iset(Interval(0, 2, True, True))),
    (iset(Interval(0, 1)), iset(Interval(1, 2)), iset(Interval(0, 1), Interval(1, 2))),
    (iset(point(1)), iset(Interval(0, 2, True, True)), iset(Interval(0, 1, True, False), Interval(1, 2, False, True))),
    (iset(point(1)), iset(point(1)), EMPTY),
    (iset(point(1)), iset(point(2)), iset(point(1), point(2))),
    (iset(Interval(None, 0, False, True)), iset(Interval(0, None, True, False)),
     iset(Interval(None, 0), Interval(0, None))),  # Two rays sharing their closed end.
    (iset(Interval(None, 0)), iset(Interval(0, None)), iset(Interval(None, 0), Interval(0, None))),
    (iset(Interval(None, 1)), iset(Interval(0, None)), iset(Interval(None, 0, False, True),
                                                            Interval(1, None, True, False))),
    (iset(Interval(0, 4)), iset(Interval(1, 2), Interval(2, 3)),
     iset(Interval(0, 1, False, True), point(2), Interval(3, 4, True, False))),
    (iset(Interval(0, SQRT2_CUBIC, True, True)), iset(Interval(SQRT2, 3, True, True)),
     iset(Interval(0, SQRT2_CUBIC, True, False), Interval(SQRT2_CUBIC, 3, False, True))),  # Named as a names it.
    (iset(Interval(SQRT2, THREE, False, True)), iset(Interval(q(3), q(5), True, False)),
     iset(Interval(SQRT2, THREE), Interval(THREE, q(5)))),  # 3 named by a's linear root.
    (iset(Interval(None, CBRT2), point(q(5, 2))), iset(Interval(q(1), q(5, 2), True, True)),
     iset(Interval(None, 1), Interval(CBRT2, q(5, 2), True, False))),
)


def bound_sets(kind: str, n: int) -> tuple[IntervalSet, IntervalSet]:
    """Two sets of n intervals with 4n distinct interleaved ends: half-open ones with ends k/100 or (10^10000 + k)/7,
    closed ones with ends sqrt(10^24 + k) or the real cube roots of 2 + k (the families that the docstring of
    mathforge.areas.sets.interval_algebra measures)."""
    if kind in ("k/100", "long"):
        if kind == "k/100":
            ends = [q(k, 100) for k in range(-200 * n, 200 * n, 100)]
        else:
            with computation(WIDE, fresh=True):  # Building the sets compares their 33,000-bit ends.
                ends = [q(10 ** 10000 + k, 7) for k in range(4 * n)]
        with computation(WIDE, fresh=True):
            return (iset(*[Interval(ends[4 * i], ends[4 * i + 2], True, False) for i in range(n)]),
                    iset(*[Interval(ends[4 * i + 1], ends[4 * i + 3], False, True) for i in range(n)]))
    if kind == "sqrt":
        ends = [RealAlgebraicRoot((-(10 ** 24 + k), 0, 1), 1) for k in range(4 * n)]
    else:
        ends = [RealAlgebraicRoot((-(2 + k), 0, 0, 1), 0) for k in range(4 * n)]
    return (iset(*[Interval(ends[4 * i], ends[4 * i + 2], True, True) for i in range(n)]),
            iset(*[Interval(ends[4 * i + 1], ends[4 * i + 3], True, True) for i in range(n)]))


class SymmetricTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with its four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    # -----------------------------------------------------------------------------------------------------------
    # Values

    def test_known_values(self):
        """Known values of the card and special cases, each from a Result that mf.verify accepts, with the inputs'
        records as the ends of the value."""
        for a, b, value in KNOWN:
            with self.subTest(a=mf.render_text(a), b=mf.render_text(b)):
                if SQRT2_CUBIC not in [end for item in a.intervals for end in (item.lower, item.upper)]:
                    self.assertEqual(reference(a, b), written(value))  # The expectation, by the reference.
                result = support.run(OPERATION, a=a, b=b)
                self.assertVerified(result, value)
                self.assertIs(type(result.value), IntervalSet)
                inputs = [end for item in a.intervals + b.intervals for end in (item.lower, item.upper)]
                for item in result.value.intervals:  # Every finite end is a record of an input, of its class.
                    for end in (item.lower, item.upper):
                        if end is not None:
                            self.assertTrue(any(type(end) is type(x) and end == x for x in inputs), end)
                self.roundtrip(result)
        # The ends keep the inputs' classes: 3 as a's root of x - 3, sqrt 2 as a's root of x^3 - 2x.
        value = support.run(OPERATION, a=KNOWN[19][0], b=KNOWN[19][1]).value
        self.assertIs(type(value.intervals[0].upper), RealAlgebraicRoot)
        self.assertEqual(value.intervals[0].upper.integer_coefficients, (-3, 1))
        value = support.run(OPERATION, a=KNOWN[18][0], b=KNOWN[18][1]).value
        self.assertEqual(value.intervals[1].lower.integer_coefficients, (0, -2, 0, 1))
        # The evidence: the least-denominator rational below, between and above the points.
        result = support.run(OPERATION, a=KNOWN[0][0], b=KNOWN[0][1])
        self.assertEqual(result.steps[0].outputs[1], ((q(-1), q(1, 2), q(3, 2), q(5, 2), q(4)),))
        result = support.run(OPERATION, a=KNOWN[3][0], b=KNOWN[3][1])  # Points -sqrt 2, 0, sqrt 2, 2.
        self.assertEqual(result.steps[0].outputs[1], ((q(-2), q(-1), q(1), q(3, 2), q(3)),))
        result = support.run(OPERATION, a=EMPTY, b=WHOLE)
        self.assertEqual(result.steps[0].outputs[1], ((q(0),),))
        # A direct call of produce() opens a budget of its own.
        self.assertEqual(family.produce(OPERATION, dict(a=KNOWN[0][0], b=KNOWN[0][1]))[0], KNOWN[0][2])

    def test_seeded_agreement_with_the_reference(self):
        """produce() agrees with the reference of this module on 200 seeded inputs, rationals, square roots and cube
        roots shared between the two sets, and the checker accepts each claim."""
        rng = random.Random(20261113)
        universe: list[Key] = list(dict.fromkeys(("q", Fraction(n, d)) for n in range(-10, 11) for d in (1, 2, 3)))
        universe += [("rad", s, 2, Fraction(v)) for v in (2, 3, 5, 7, 8, 10, Fraction(3, 2), Fraction(5, 3))
                     for s in (-1, 1)]
        universe += [("rad", s, 3, Fraction(v)) for s, v in ((-1, 5), (-1, 2), (1, 2), (1, 3), (1, 4),
                                                             (1, Fraction(1, 2)))]
        order = functools.cmp_to_key(compare)

        def random_set(pool: Sequence[Key]) -> IntervalSet:
            ends = sorted(rng.sample(pool, rng.randint(0, min(6, len(pool)))), key=order)
            items, k = [], 0
            if ends and rng.random() < 0.2:
                items.append(Interval(None, record(ends[0]), False, rng.random() < 0.5))
                k = 1
            while k < len(ends):
                if rng.random() < 0.15:
                    items.append(point(record(ends[k])))
                    k += 1
                elif k + 1 < len(ends):
                    items.append(Interval(record(ends[k]), record(ends[k + 1]), rng.random() < 0.5,
                                          rng.random() < 0.5))
                    k += 2
                else:
                    items.append(Interval(record(ends[k]), None, rng.random() < 0.5, False))
                    k += 1
            if not ends and rng.random() < 0.3:
                items.append(Interval(None, None))
            return IntervalSet(tuple(items))

        nonempty = 0
        for trial in range(200):
            pool = rng.sample(universe, 8)
            a = random_set(pool)
            b = a if rng.random() < 0.05 else random_set(pool)
            with self.subTest(trial=trial):
                value, evidence = family.produce(OPERATION, dict(a=a, b=b))
                self.assertEqual(written(value), reference(a, b))
                self.assertLessEqual(len(value.intervals), len(a.intervals) + len(b.intervals))
                self.assertIs(support.verdict(family, OPERATION, dict(a=a, b=b), value, evidence), True)
                nonempty += bool(value.intervals)
        self.assertGreater(nonempty, 150)

    def test_seeded_agreement_on_shared_and_reducible_names(self):
        """produce() agrees with v0.7's IntervalSet.symmetric_difference (compare_real) on seeded inputs whose ends
        name one number in several ways: sqrt 2 as a root of x^2 - 2, x^3 - 2x, (x^2 - 2)(x^2 - 3), (x - 1)(x^2 - 2)
        and (x^2 - 2)(x^2 + 1), and 1 and -1/2 as Rationals and as roots of linear and reducible polynomials; the
        checker accepts each claim, and also v0.7's value with the same evidence."""
        rng = random.Random(20261116)
        polynomials = {(-2, 0, 1): 2, (0, -2, 0, 1): 3, (6, 0, -5, 0, 1): 4, (2, -2, -1, 1): 3, (-2, 0, -1, 0, 1): 2,
                       (-1, 1): 1, (1, 2): 1, (-3, 0, 1): 2}
        pool = [RealAlgebraicRoot(p, i) for p, count in polynomials.items() for i in range(count)]
        pool += [q(n, 2) for n in range(-5, 6)]
        order = functools.cmp_to_key(compare_real)

        def random_set() -> IntervalSet:
            ends = sorted(rng.sample(pool, rng.randint(0, 7)), key=order)
            items, k = [], 0
            if ends and rng.random() < 0.25:
                items.append(Interval(None, ends[0], False, rng.random() < 0.5))
                k = 1
            while k < len(ends):
                if rng.random() < 0.15 or k + 1 == len(ends):
                    items.append(point(ends[k]) if rng.random() < 0.5 else Interval(ends[k], None))
                    k += 1
                else:
                    items.append(Interval(ends[k], ends[k + 1], rng.random() < 0.5, rng.random() < 0.5))
                    k += 2
            return IntervalSet(tuple(items))

        for trial in range(60):
            a, b = random_set(), random_set()
            with self.subTest(trial=trial):
                value, evidence = family.produce(OPERATION, dict(a=a, b=b))
                expected = a.symmetric_difference(b)
                self.assertTrue(equal_interval_sets(value, expected), (value, expected))
                self.assertIs(support.verdict(family, OPERATION, dict(a=a, b=b), value, evidence), True)
                self.assertIs(support.verdict(family, OPERATION, dict(a=a, b=b), expected, evidence), True)

    def test_exact_points_survive_the_separation(self):
        """Regression (review of wave m7a): a root of a reducible polynomial that the separation of the line meets
        exactly at a bracket midpoint, next to a rational end of the same pass, used to remove that rational end from
        the line, and produce() failed with KeyError (internal_error). interval_algebra._Line._bisect now leaves an
        exact point as it is. -1 and 1 are the roots of x^2 - 1, -1/2 a root of
        2x^2 - x - 1."""
        minus_one, one = RealAlgebraicRoot((-1, 0, 1), 0), RealAlgebraicRoot((-1, 0, 1), 1)
        minus_half = RealAlgebraicRoot((-1, -1, 2), 0)
        b = iset(Interval(q(-2), q(-7, 5), True, False), Interval(minus_one, q(1, 2)))
        cases = (
            (EMPTY, b, b),
            (b, EMPTY, b),
            (iset(Interval(q(-3, 2), None, True, False)), iset(Interval(q(-2), one, True, False)),
             iset(Interval(q(-2), q(-3, 2), True, False), Interval(one, None, True, False))),
            (iset(Interval(q(-1), q(-2, 3), True, False), Interval(minus_half, q(7, 5), False, True)), EMPTY,
             iset(Interval(q(-1), q(-2, 3), True, False), Interval(minus_half, q(7, 5), False, True))),
            (b, iset(Interval(q(-3, 2), q(0), True, True)),
             iset(Interval(q(-2), q(-3, 2), True, False), Interval(q(-7, 5), minus_one, True, True),
                  Interval(q(0), q(1, 2), False, False))),
        )
        for a, b_, value in cases:
            with self.subTest(a=mf.render_text(a), b=mf.render_text(b_)):
                self.assertTrue(equal_interval_sets(value, a.symmetric_difference(b_)))
                result = support.run(OPERATION, a=a, b=b_)
                self.assertVerified(result, value)
                self.roundtrip(result)
        # The line keeps all five points -2, -7/5, -1, 1/2 and 1 (the other root of x^2 - 1): the samples are the
        # least-denominator rationals below, between and above them, worked out by hand.
        samples = family.produce(OPERATION, dict(a=EMPTY, b=b))[1][0]
        self.assertEqual(samples, (q(-3), q(-3, 2), q(-4, 3), q(0), q(2, 3), q(2)))

    # -----------------------------------------------------------------------------------------------------------
    # Inputs

    def test_invalid_inputs(self):
        """0.5, True, "1", None, a wrong container, an EmptySet, a UniversalSet or a matrix in every argument, and
        every record that no constructor made, are invalid_input, never the producer's internal_error."""
        a, b = iset(Interval(0, 1)), iset(Interval(2, 3, True, True))
        wrong = (*STAND_INS, 1, EmptySet(Reals), UniversalSet(Reals), Interval(0, 1), (Interval(0, 1),),
                 [Interval(0, 1)], q(1), mf.Matrix(((1, 2), (3, 4))),
                 type("Sub", (IntervalSet,), {})((Interval(0, 1),)))
        for name in ("a", "b"):
            for number, item in enumerate(wrong):
                with self.subTest(argument=name, item=number):
                    self.assertInvalidInput(support.run(OPERATION, **dict(dict(a=a, b=b), **{name: item})))
                    with self.assertRaises(InvalidInput):
                        family.produce(OPERATION, dict(dict(a=a, b=b), **{name: item}))
        with self.subTest(arguments="extra"):
            self.assertInvalidInput(operations_v6.run(OPERATION, dict(a=a, b=b, c=a)))
        with self.subTest(arguments="missing"):
            self.assertInvalidInput(operations_v6.run(OPERATION, {}))
            self.assertInvalidInput(operations_v6.run(OPERATION, dict(a=a)))
        q0, q1, q2 = q(0), q(1), q(2)
        forgeries = {
            "a list of intervals": support.forged(IntervalSet, intervals=[Interval(0, 1)]),
            "no intervals field": support.forged(IntervalSet),
            "a tuple for an interval": forged_set((q0, q1, False, False)),
            "unsorted": forged_set(Interval(2, 3), Interval(0, 1)),
            "overlapping": forged_set(Interval(0, 2), Interval(1, 3)),
            "mergeable": forged_set(Interval(0, 1, False, True), Interval(1, 2)),
            "touching open ends, mergeable": forged_set(Interval(0, 1, True, True), Interval(1, 2, False, False)),
            "an empty interval": forged_set(forged_interval(q1, q1)),
            "a half-open point": forged_set(forged_interval(q1, q1, True, False)),
            "reversed": forged_set(forged_interval(q2, q1)),
            "a closed infinite end": forged_set(forged_interval(None, q1, True, False)),
            "an int flag": forged_set(forged_interval(q0, q1, 1, 0)),
            "a None flag": forged_set(forged_interval(q0, q1, None, False)),
            "an int end": forged_set(forged_interval(0, q1)),
            "a float end": forged_set(forged_interval(0.5, q1)),
            "a str end": forged_set(forged_interval("0", q1)),
            "a bool end": forged_set(forged_interval(q0, True)),
            "a None upper end of a point": forged_set(forged_interval(q0, None, True, True)),
            "a Fraction end": forged_set(forged_interval(Fraction(1, 2), q1)),
            "no lower field": forged_set(support.forged(Interval, upper=q1, left_closed=False, right_closed=False)),
            "a Rational not in lowest terms": forged_set(forged_interval(support.forged(Rational, numerator=2,
                                                                                        denominator=4), q1)),
            "a Rational with denominator 0": forged_set(forged_interval(support.forged(Rational, numerator=1,
                                                                                       denominator=0), q1)),
            "a Rational with a negative denominator": forged_set(forged_interval(
                support.forged(Rational, numerator=1, denominator=-2), q1)),
            "a Rational with a float numerator": forged_set(forged_interval(
                support.forged(Rational, numerator=0.5, denominator=1), q1)),
            "a root of no real number": forged_set(forged_interval(forged_root((1, 0, 1), 0), q2)),
            "a root index beyond the real roots": forged_set(forged_interval(forged_root((-2, 0, 1), 2), q2)),
            "a cubic index beyond its one real root": forged_set(forged_interval(forged_root((-2, 0, 0, 1), 1), q2)),
            "a negative root index": forged_set(forged_interval(forged_root((-2, 0, 1), -1), q2)),
            "a bool root index": forged_set(forged_interval(forged_root((-2, 0, 1), True), q2)),
            "a square polynomial": forged_set(forged_interval(forged_root((4, 0, -4, 0, 1), 1), q2)),
            "a polynomial that is not primitive": forged_set(forged_interval(forged_root((-4, 0, 2), 1), q2)),
            "a negative leading coefficient": forged_set(forged_interval(forged_root((2, 0, -1), 1), q2)),
            "a constant polynomial": forged_set(forged_interval(forged_root((1,), 0), q2)),
            "a list of coefficients": forged_set(forged_interval(forged_root([-2, 0, 1], 1), q2)),
            "a float coefficient": forged_set(forged_interval(forged_root((-2.0, 0, 1), 1), q2)),
            "a bool coefficient": forged_set(forged_interval(forged_root((-2, 0, True), 1), q2)),
            "a linear root not in lowest terms": forged_set(forged_interval(forged_root((-2, 4), 0), q2)),
            "an impostor root": forged_set(forged_interval(support.impostor(SQRT2), q2)),
            "an impostor interval": forged_set(support.impostor(Interval(0, 1))),
            "an impostor rational": forged_set(forged_interval(support.impostor(q0), q1)),
        }
        for label, item in forgeries.items():
            for name in ("a", "b"):
                with self.subTest(case=label, argument=name):
                    args = dict(dict(a=a, b=b), **{name: item})
                    with self.assertRaises(InvalidInput):
                        family.produce(OPERATION, dict(args))
                    if label not in INSPECTED:
                        self.assertInvalidInput(support.run(OPERATION, **args))
        # Through produce() itself: other operations and argument sets other than the operation's.
        for operation in ("set_union", "symmetric_difference", None, 1):
            with self.subTest(operation=operation), self.assertRaises(mf.UnsupportedOperation):
                family.produce(operation, dict(a=a, b=b))
        for args in ({}, dict(a=a), dict(a=a, b=b, c=b), dict(a=a, c=b), [("a", a), ("b", b)], None):
            with self.subTest(args=args), self.assertRaises(InvalidInput):
                family.produce(OPERATION, args)

    def test_resource_limits(self):
        """An integer of 2**65536 in an end, and max_work=1, exceed the budget; an oversized matrix in an IntervalSet
        argument is no IntervalSet (invalid_input, before any of it is read); and a claim that the default limits
        could not verify or reload is refused, whatever the caller's limits."""
        with computation(ComputationLimits(max_integer_bits=70_000), fresh=True):
            huge = iset(Interval(Rational(support.HUGE), None))
            huge_root = iset(Interval(forged_root((-support.HUGE, 0, 1), 1), None))
        small = iset(Interval(0, 1))
        for name in ("a", "b"):
            for item in (huge, huge_root):
                with self.subTest(argument=name, root=item is huge_root):
                    self.assertResourceLimit(support.run(OPERATION, **dict(dict(a=small, b=small), **{name: item})))
            with self.subTest(argument=name, matrix=True):
                self.assertInvalidInput(support.run(OPERATION, **dict(dict(a=small, b=small),
                                                                      **{name: support.oversized_matrix()})))
        with self.subTest(max_work=1):
            self.assertResourceLimit(support.run(OPERATION, a=small, b=iset(Interval(SQRT2, None)),
                                                 limits=ComputationLimits(max_work=1)))
        # One beyond the largest accepted inputs (measured on 2026-11-13): the checker's work on 10 + 10 real cube
        # roots, a Result of 2 x 304 intervals with ends k/100 (606 + 608 records, twice), 2 x 13 intervals with ends
        # sqrt(10^24 + k), 2 x 6 intervals with ends (10^10000 + k)/7.
        for kind, n, reason in (("cube", 10, "the check"), ("k/100", 304, "reload"), ("sqrt", 13, "reload"),
                                ("long", 6, "the check")):
            a, b = bound_sets(kind, n)
            with self.subTest(kind=kind, n=n):
                start = time.perf_counter()
                result = support.run(OPERATION, a=a, b=b, limits=WIDE)
                self.assertResourceLimit(result)
                self.assertIn(reason, result.error.message)
                self.assertIn("default limits", result.error.message)
                self.assertLess(time.perf_counter() - start, 3.0)

    def test_largest_accepted_inputs_reload_and_verify(self):
        """The largest inputs accepted give Results that reload and verify under the default limits: 9 + 9 real cube
        roots (the checker needs about 885,000 units), 303 + 303 intervals with ends k/100, and 5 + 5 intervals with
        ends (10^10000 + k)/7 (about 1,118,000 units)."""
        for kind, n in (("cube", 9), ("k/100", 303), ("long", 5)):
            a, b = bound_sets(kind, n)
            with self.subTest(kind=kind, n=n):
                result = support.run(OPERATION, a=a, b=b, limits=WIDE)
                self.assertIsNone(result.error)
                self.assertEqual(len(result.value.intervals), 2 * n)  # Each interval of a split into two.
                decoded = mf.loads(mf.dumps(result))
                self.assertEqual(decoded, result)
                report = mf.verify(decoded)
                self.assertTrue(report.verified, support.statuses(report))

    # -----------------------------------------------------------------------------------------------------------
    # The checker

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: other
        samples, and ends named by the other input."""
        a, b = KNOWN[3][0], KNOWN[3][1]
        self.assertAccepts(OPERATION, dict(a=a, b=b), KNOWN[3][2], ((q(-7), q(-3, 4), q(1, 4), q(10, 7), q(100)),))
        a, b = KNOWN[18][0], KNOWN[18][1]
        named_by_b = iset(Interval(0, SQRT2, True, False), Interval(SQRT2, 3, False, True))
        self.assertAccepts(OPERATION, dict(a=a, b=b), named_by_b, family.produce(OPERATION, dict(a=a, b=b))[1])
        # v0.7's method on the card's values, with the producer's evidence.
        for a, b, value in KNOWN[:4]:
            with self.subTest(a=mf.render_text(a)):
                self.assertAccepts(OPERATION, dict(a=a, b=b), a ^ b, family.produce(OPERATION, dict(a=a, b=b))[1])

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where a Rational belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        args = dict(a=KNOWN[3][0], b=KNOWN[3][1])
        value, evidence = family.produce(OPERATION, dict(args))
        samples = evidence[0]
        union = iset(Interval(MINUS_SQRT2, 2, True, True))
        cases = (
            ("wrong value", OPERATION, args, iset(Interval(MINUS_SQRT2, 0, True, False)), evidence),
            ("the union", OPERATION, args, union, evidence),
            ("a closed end flipped", OPERATION, args,
             iset(Interval(MINUS_SQRT2, 0, True, True), Interval(SQRT2, 2, False, True)), evidence),
            ("wrong operation", "set_union", args, value, evidence),
            ("arguments swapped to other sets", OPERATION, dict(a=args["a"], b=args["a"]), value, evidence),
            ("not canonical", OPERATION, args, forged_set(*reversed(value.intervals)), evidence),
            ("an EmptySet", OPERATION, dict(a=KNOWN[1][0], b=KNOWN[1][1]), EmptySet(Reals),
             family.produce(OPERATION, dict(a=KNOWN[1][0], b=KNOWN[1][1]))[1]),
            ("a UniversalSet", OPERATION, dict(a=EMPTY, b=WHOLE), UniversalSet(Reals), ((q(0),),)),
            ("impostor value", OPERATION, args, support.impostor(value), evidence),
            ("forged interval", OPERATION, args,
             forged_set(value.intervals[0], forged_interval(SQRT2, q(2), 0, True)), evidence),
            ("a foreign name", OPERATION, args,
             iset(Interval(MINUS_SQRT2, 0, True, False), Interval(RealAlgebraicRoot((0, -2, 0, 1), 2), 2, False,
                                                                  True)), evidence),
            ("bool sample", OPERATION, args, value, ((True, *samples[1:]),)),
            ("float sample", OPERATION, args, value, ((-2.0, *samples[1:]),)),
            ("str sample", OPERATION, args, value, (("-2", *samples[1:]),)),
            ("list of samples", OPERATION, args, value, (list(samples),)),
            ("missing evidence", OPERATION, args, value, ()),
            ("extra evidence", OPERATION, args, value, (samples, samples)),
            ("a sample missing", OPERATION, args, value, (samples[1:],)),
            ("a sample at a point", OPERATION, args, value, ((q(-2), q(0), q(1), q(3, 2), q(3)),)),
            ("oversized evidence", OPERATION, args, value, (tuple(q(k) for k in range(100_000)),)),
            ("oversized value", OPERATION, args, forged_set(*[value.intervals[0]] * 100_000), evidence),
        )
        for label, operation, claim_args, claim, claim_evidence in cases:
            with self.subTest(case=label):
                self.assertRejects(operation, claim_args, claim, claim_evidence)
        result = support.run(OPERATION, **args)
        report = mf.verify(support.tampered(result, union))
        self.assertFalse(report.verified)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        results = [support.run(OPERATION, a=a, b=b) for a, b, _ in KNOWN]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for number, (result, text) in enumerate(zip(results, texts)):
                with self.subTest(case=number):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            failed = support.run(OPERATION, a=EMPTY, b=EMPTY)  # The patch is live: a new run reaches it and fails.
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    # -----------------------------------------------------------------------------------------------------------
    # Formats and paths

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: the empty set, the
        whole line, a point, unions of intervals, rays and algebraic ends."""
        rendered = (
            (KNOWN[0][0], KNOWN[0][1], "[0, 1) union (2, 3)", r"[0, 1) \cup (2, 3)"),
            (KNOWN[1][0], KNOWN[1][1], "{}", r"\varnothing"),
            (KNOWN[2][0], KNOWN[2][1], "(-inf, 0) union (0, inf)", r"(-\infty, 0) \cup (0, \infty)"),
            (EMPTY, WHOLE, "(-inf, inf)", r"(-\infty, \infty)"),
            (iset(point(q(1, 2))), EMPTY, "[1/2, 1/2]", r"[\frac{1}{2}, \frac{1}{2}]"),
            (KNOWN[3][0], KNOWN[3][1],
             "[RootOf(t^2 - 2; real_index=0), 0) union (RootOf(t^2 - 2; real_index=1), 2]",
             r"[\operatorname{RootOf}_{\mathbb{R}}\left(t^{2} - 2; 0\right), 0) \cup "
             r"(\operatorname{RootOf}_{\mathbb{R}}\left(t^{2} - 2; 1\right), 2]"),
            (iset(Interval(None, CBRT2, False, True)), WHOLE, "(RootOf(t^3 - 2; real_index=0), inf)",
             r"(\operatorname{RootOf}_{\mathbb{R}}\left(t^{3} - 2; 0\right), \infty)"),
        )
        for a, b, text, latex in rendered:
            with self.subTest(text=text):
                result = support.run(OPERATION, a=a, b=b)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run(OPERATION, a=WHOLE, b=WHOLE).value)["data"],
                         {"type": "interval_set", "intervals": []})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261117)
        ends = [q(n, d) for n in range(-6, 7) for d in (1, 2)] + [SQRT2, MINUS_SQRT2, CBRT2]
        ends = sorted(set(ends), key=functools.cmp_to_key(lambda x, y: compare(key_of(x), key_of(y))))
        for trial in range(16):
            chosen = sorted(rng.sample(range(len(ends)), 8))
            a = iset(*[Interval(ends[chosen[k]], ends[chosen[k + 1]], k % 4 == 0, True) for k in range(0, 4, 2)])
            b = iset(*[Interval(ends[chosen[k]], ends[chosen[k + 1]], True, k % 4 == 0)
                       for k in rng.choice(((1, 4), (3, 6), (0, 4)))])
            with self.subTest(trial=trial):
                expected = support.run(OPERATION, a=a, b=b).value
                self.assertIsNotNone(expected)
                self.assertEqual(written(expected), reference(a, b))
                found = support.lean_value(OPERATION, a=a, b=b)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(OPERATION, a=a, b=b), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        raw = {"a": [{"lower": 0, "upper": 2, "left_closed": True, "right_closed": True}],
               "b": [{"lower": 1, "upper": "3", "left_closed": True, "right_closed": False},
                     {"lower": "7/2", "upper": None, "left_closed": False, "right_closed": False}]}
        result = self.answer(support.call(OPERATION, raw))
        self.assertTrue(mf.verify(result).verified)
        expected = iset(Interval(0, 1, True, False), Interval(2, 3), Interval(q(7, 2), None))
        self.assertEqual(result.value, expected)
        self.assertEqual(result, support.run(OPERATION, a=KNOWN[0][0],
                                             b=iset(Interval(1, 3, True, False), Interval(q(7, 2), None))))
        for a, b, value in (KNOWN[3], (KNOWN[3][0], iset(Interval(0, 1), point(CBRT2)), None)):
            with self.subTest(a=mf.render_text(a), b=mf.render_text(b)):
                result = self.answer(support.call_values(OPERATION, a=a, b=b))
                self.assertEqual(result, support.run(OPERATION, a=a, b=b))
                self.assertTrue(mf.verify(result).verified)
                if value is not None:
                    self.assertEqual(result.value, value)
        self.assertToolError(support.call(OPERATION, {"a": [], "b": 5}), "invalid_input")


if __name__ == "__main__":
    import unittest
    unittest.main()
