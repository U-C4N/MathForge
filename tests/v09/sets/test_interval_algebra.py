"""The interval_algebra family of mathforge.sets: set_union, set_intersection, set_difference, set_complement.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register sets interval_algebra

Every expected value comes from the card or from the reference of this module, never from the producer: the reference
reads each end as a number of its own kind, a rational, a signed square root sqrt(m) of a non-square integer or the real
cube root of a non-cube integer, orders the ends by nested rational enclosures (integer square and cube roots of
scaled integers), and evaluates the operation on every end and every open gap between two neighbouring ends.
tests/v09/sets/test_interval_algebra_checker.py tests the checker on its own.

The card names no deferred (Unevaluated) case for this family, and no operation of it takes an int or a matrix: the
resource tests put an integer of 2**65536 into an end instead, and the oversized matrix has no argument to go to.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
import functools
from math import isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean, operations_v6
from mathforge.algebraic import RealAlgebraicRoot, compare_real
from mathforge.areas.sets import interval_algebra as family
from mathforge.contracts import EmptySet, ErrorInfo, UniversalSet
from mathforge.errors import InvalidInput
from mathforge.intervals import Interval, IntervalSet, equal_interval_sets
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Rational, Reals

from .. import support

OPERATIONS = ("set_union", "set_intersection", "set_difference", "set_complement")
WIDE = ComputationLimits(max_work=10 ** 9, max_refinements=10 ** 8, max_nodes=10 ** 8, max_evidence_items=10 ** 8)
STAND_INS = (0.5, True, "1", None)
# Forgeries that operations_v6.run refuses before the producer runs, with execution_error: its budget.inspect of the
# arguments reads every field of a record, and these lack one or hold a float where an int belongs. No decoded
# document can hold them; produce() itself raises InvalidInput for each.
INSPECTED = frozenset({"no intervals field", "no lower field", "a Rational with a float numerator"})

Key = tuple  # ("q", Fraction), ("sqrt", sign, m) for sign * sqrt(m), or ("cbrt", c) for the real cube root of c.


# ---------------------------------------------------------------------------------------------------------------
# The reference: ends as numbers of three kinds, ordered by rational enclosures.

def icbrt(n: int) -> int:
    """floor(n ** (1/3)) for an integer n, by bisection."""
    if n < 0:
        return -icbrt(-n) - (0 if icbrt(-n) ** 3 == -n else 1)
    low, high = 0, 1
    while high ** 3 <= n:
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if middle ** 3 <= n:
            low = middle
        else:
            high = middle
    return low


def enclosure(key: Key, bits: int) -> tuple[Fraction, Fraction]:
    """Rationals low <= x <= high around the number of a key, high - low <= 2**-bits."""
    if key[0] == "q":
        return key[1], key[1]
    scale = 1 << bits
    if key[0] == "sqrt":
        _, sign, m = key
        root = isqrt(m * scale * scale)
        low, high = Fraction(root, scale), Fraction(root + 1, scale)
        return (low, high) if sign > 0 else (-high, -low)
    root = icbrt(key[1] * scale ** 3)
    return Fraction(root, scale), Fraction(root + 1, scale)


def compare(left: Key, right: Key) -> int:
    """The order of the numbers of two keys. Two different keys of the reference's pools name different numbers: a
    rational is no square or cube root of a non-square or non-cube, and sqrt(m) = cbrt(c) would make m = (c / m)**2 a
    square."""
    if left == right:
        return 0
    bits = 8
    while True:
        (a, b), (c, d) = enclosure(left, bits), enclosure(right, bits)
        if b < c:
            return -1
        if d < a:
            return 1
        bits *= 2


def record(key: Key) -> Rational | RealAlgebraicRoot:
    """The end record of a key."""
    if key[0] == "q":
        return Rational(key[1].numerator, key[1].denominator)
    if key[0] == "sqrt":
        return RealAlgebraicRoot((-key[2], 0, 1), 1 if key[1] > 0 else 0)
    return RealAlgebraicRoot((-key[1], 0, 0, 1), 0)


def key_of(end: object) -> Key:
    """The key of an end record of the reference's kinds."""
    if type(end) is Rational:
        return ("q", Fraction(end.numerator, end.denominator))
    coefficients, index = end.integer_coefficients, end.real_index
    if len(coefficients) == 3 and coefficients[1] == 0 and coefficients[2] == 1:
        return ("sqrt", 1 if index else -1, -coefficients[0])
    if len(coefficients) == 4 and coefficients[1:] == (0, 0, 1):
        return ("cbrt", -coefficients[0])
    raise AssertionError(f"no key for {end!r}")


Written = list[tuple[Key | None, Key | None, bool, bool]]


def written(value: IntervalSet) -> Written:
    """An IntervalSet as (lower, upper, left_closed, right_closed) tuples of keys, None at an infinite end."""
    return [(None if item.lower is None else key_of(item.lower), None if item.upper is None else key_of(item.upper),
             item.left_closed, item.right_closed) for item in value.intervals]


def holds_point(intervals: Written, x: Key) -> bool:
    for lower, upper, left, right in intervals:
        above = lower is None or compare(lower, x) < (1 if left else 0)
        below = upper is None or compare(x, upper) < (1 if right else 0)
        if above and below:
            return True
    return False


def holds_gap(intervals: Written, low: Key | None, high: Key | None) -> bool:
    """Whether the open gap (low, high) between two neighbouring ends (None for an infinity) lies in the set: an
    interval holds all of it or none of it."""
    for lower, upper, _, _ in intervals:
        starts = lower is None or (low is not None and compare(lower, low) <= 0)
        ends = upper is None or (high is not None and compare(high, upper) <= 0)
        if starts and ends:
            return True
    return False


TRUTH: dict[str, Callable[..., bool]] = {
    "set_union": lambda x, y: x or y,
    "set_intersection": lambda x, y: x and y,
    "set_difference": lambda x, y: x and not y,
    "set_complement": lambda x: not x,
    "symmetric_difference": lambda x, y: x != y,  # For IntervalSet.symmetric_difference, no operation of the family.
}


def reference(operation: str, *sets: IntervalSet) -> Written:
    """The result of the operation, as written() writes a set, by the truth of the operation on every end of the
    inputs and on every open gap between neighbouring ends."""
    pieces = [written(value) for value in sets]
    ends: list[Key] = []
    for intervals in pieces:
        for lower, upper, _, _ in intervals:
            for end in (lower, upper):
                if end is not None and end not in ends:
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
        truth.append(TRUTH[operation](*inside))
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


def arguments(operation: str, a: IntervalSet, b: IntervalSet | None = None) -> dict[str, IntervalSet]:
    return dict(a=a) if operation == "set_complement" else dict(a=a, b=b)


SQRT2, MINUS_SQRT2 = RealAlgebraicRoot((-2, 0, 1), 1), RealAlgebraicRoot((-2, 0, 1), 0)
SQRT2_CUBIC = RealAlgebraicRoot((0, -2, 0, 1), 2)  # sqrt 2 as the largest root of x^3 - 2x.
CBRT2 = RealAlgebraicRoot((-2, 0, 0, 1), 0)
EMPTY, WHOLE = IntervalSet(()), IntervalSet((Interval(None, None),))

# The known values of the card, then special cases: (operation, a, b, value).
KNOWN = (
    ("set_union", iset(Interval(0, 2, True, True)), iset(Interval(1, 3, True, False)),
     iset(Interval(0, 3, True, False))),
    ("set_intersection", iset(Interval(0, 2, True, True)), iset(Interval(1, 3, True, False)),
     iset(Interval(1, 2, True, True))),
    ("set_complement", iset(Interval(0, 1, True, False)), None,
     iset(Interval(None, 0), Interval(1, None, True, False))),
    ("set_difference", WHOLE, iset(point(0)), iset(Interval(None, 0), Interval(0, None))),
    ("set_difference", iset(Interval(MINUS_SQRT2, SQRT2, True, True)), iset(Interval(0, 1)),
     iset(Interval(MINUS_SQRT2, 0, True, True), Interval(1, SQRT2, True, True))),
    ("set_union", EMPTY, EMPTY, EMPTY),
    ("set_union", iset(Interval(0, 1)), iset(Interval(1, 2, True, False)), iset(Interval(0, 2))),
    ("set_union", iset(Interval(None, 0, False, True)), iset(Interval(0, None)), WHOLE),
    ("set_intersection", iset(Interval(0, 1)), iset(Interval(1, 2)), EMPTY),
    ("set_intersection", iset(Interval(0, 1, False, True)), iset(Interval(1, 2, True, False)), iset(point(1))),
    ("set_intersection", iset(Interval(0, SQRT2_CUBIC, True, True)), iset(Interval(SQRT2, 3, True, True)),
     iset(point(SQRT2_CUBIC))),  # One number under two polynomials, named as a names it.
    ("set_difference", iset(Interval(0, 2, True, True)), iset(Interval(0, 1), Interval(1, 2)),
     iset(point(0), point(1), point(2))),
    ("set_difference", iset(Interval(0, 1)), WHOLE, EMPTY),
    ("set_complement", EMPTY, None, WHOLE),
    ("set_complement", WHOLE, None, EMPTY),
    ("set_complement", iset(Interval(None, CBRT2), point(q(5, 2))), None,
     iset(Interval(CBRT2, q(5, 2), True, False), Interval(q(5, 2), None))),
)


class IntervalAlgebraTests(support.FamilyTestCase):
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
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, a, b, value in KNOWN:
            args = arguments(operation, a, b)
            with self.subTest(operation=operation, args=mf.render_text(a) + " ; " + mf.render_text(b or EMPTY)):
                if SQRT2_CUBIC not in [end for item in a.intervals for end in (item.lower, item.upper)]:
                    self.assertEqual(reference(operation, *args.values()), written(value))  # By the reference.
                result = support.run(operation, **args)
                self.assertVerified(result, value)
                self.assertIs(type(result.value), IntervalSet)
                self.roundtrip(result)
        # The evidence: the least-denominator rational below, between and above the points.
        result = support.run("set_union", a=KNOWN[0][1], b=KNOWN[0][2])
        self.assertEqual(result.steps[0].outputs[1], ((q(-1), q(1, 2), q(3, 2), q(5, 2), q(4)),))
        result = support.run("set_difference", a=KNOWN[4][1], b=KNOWN[4][2])  # Points -sqrt 2, 0, 1, sqrt 2.
        self.assertEqual(result.steps[0].outputs[1], ((q(-2), q(-1), q(1, 2), q(5, 4), q(2)),))
        result = support.run("set_complement", a=EMPTY)
        self.assertEqual(result.steps[0].outputs[1], ((q(0),),))
        # The ends of the value are the inputs' records, the first that names each number: 3 as the root of x - 3.
        three = RealAlgebraicRoot((-3, 1), 0)
        result = support.run("set_intersection", a=iset(Interval(SQRT2, three, False, True)),
                             b=iset(Interval(q(3), q(5), True, False)))
        self.assertVerified(result, iset(point(three)))
        self.assertIs(type(result.value.intervals[0].lower), RealAlgebraicRoot)

    def test_exact_points_survive_the_separation(self):
        """Regression (review of wave m7a): a root of a reducible polynomial that the separation of the line meets
        exactly at a bracket midpoint, next to a rational end sorted in the same pass, used to drop that rational end
        from the line, and every operation failed with internal_error (KeyError in _members). -1 is root 0 of
        x^2 - 1, whose bracket meets -1 at a midpoint."""
        minus_one = RealAlgebraicRoot((-1, 0, 1), 0)
        b = iset(Interval(q(-2), q(-7, 5), True, False), Interval(minus_one, q(1, 2)))
        for operation, args, value in (
                ("set_union", dict(a=EMPTY, b=b), b),
                ("set_union", dict(a=b, b=EMPTY), b),
                ("set_intersection", dict(a=b, b=b), b),
                ("set_difference", dict(a=b, b=EMPTY), b),
                ("set_complement", dict(a=b), iset(Interval(None, q(-2), False, False),
                                                   Interval(q(-7, 5), minus_one, True, True),
                                                   Interval(q(1, 2), None, True, False)))):
            with self.subTest(operation=operation):
                result = support.run(operation, **args)
                self.assertVerified(result, value)
                self.roundtrip(result)

    def test_seeded_agreement_with_the_reference(self):
        """produce() agrees with the reference of this module on 200 seeded inputs per operation, rationals, square
        roots and cube roots shared between the two sets, and the checker accepts each claim."""
        rng = random.Random(20261113)
        universe: list[Key] = [("q", Fraction(n, d)) for n in range(-12, 13) for d in (1, 2, 3, 5)]
        universe = list(dict.fromkeys(universe))
        universe += [("sqrt", s, m) for m in (2, 3, 5, 6, 7, 8, 10, 12, 50) for s in (-1, 1)]
        universe += [("cbrt", c) for c in (-9, -3, 2, 3, 4, 20)]

        def random_set(pool: Sequence[Key]) -> IntervalSet:
            ends = sorted(rng.sample(pool, rng.randint(0, min(6, len(pool)))), key=functools.cmp_to_key(compare))
            items, k = [], 0
            if ends and rng.random() < 0.2:
                items.append(Interval(None, record(ends[0]), False, rng.random() < 0.5))
                k = 1
            while k < len(ends):
                if rng.random() < 0.15:  # A single point.
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

        for operation in OPERATIONS:
            for trial in range(200):
                pool = rng.sample(universe, 8) if rng.random() < 0.85 else [k for k in rng.sample(universe, 8)
                                                                             if k[0] != "cbrt"]
                a, b = random_set(pool), random_set(pool)
                args = arguments(operation, a, b)
                with self.subTest(operation=operation, trial=trial):
                    value, evidence = family.produce(operation, dict(args))
                    self.assertEqual(written(value), reference(operation, *args.values()))
                    self.assertIs(support.verdict(family, operation, args, value, evidence), True)

    def test_seeded_agreement_on_shared_and_reducible_names(self):
        """produce() agrees with the v0.7 IntervalSet methods (compare_real) on seeded inputs whose ends name one
        number in several ways: sqrt 2 as a root of x^2 - 2, x^3 - 2x, (x^2 - 2)(x^2 - 3), (x - 1)(x^2 - 2) and
        (x^2 - 2)(x^2 + 1), and 1 and -1/2 as Rationals and as roots of linear and reducible polynomials; the checker
        accepts each claim, and also the value that the methods write with other names of the inputs."""
        rng = random.Random(20261115)
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

        methods = {"set_union": IntervalSet.union, "set_intersection": IntervalSet.intersection,
                   "set_difference": IntervalSet.difference, "set_complement": lambda a: a.complement()}
        for operation in OPERATIONS:
            for trial in range(60):
                a, b = random_set(), random_set()
                args = arguments(operation, a, b)
                with self.subTest(operation=operation, trial=trial):
                    value, evidence = family.produce(operation, dict(args))
                    expected = methods[operation](*args.values())
                    self.assertTrue(equal_interval_sets(value, expected), (value, expected))
                    self.assertIs(support.verdict(family, operation, args, value, evidence), True)
                    self.assertIs(support.verdict(family, operation, args, expected, evidence), True)

    def test_interval_set_methods(self):
        """IntervalSet.complement, difference, symmetric_difference and the operators |, &, -, ^ and ~ (unchecked
        sugar) agree with the reference, also against EmptySet and UniversalSet; other operands are a TypeError."""
        a = iset(Interval(MINUS_SQRT2, q(1, 2), True, False), point(q(1)), Interval(SQRT2, None))
        b = iset(Interval(None, q(-1), False, True), Interval(q(0), SQRT2, True, True), point(q(3)))
        self.assertEqual(written(a | b), reference("set_union", a, b))
        self.assertEqual(written(a & b), reference("set_intersection", a, b))
        self.assertEqual(written(a - b), reference("set_difference", a, b))
        self.assertEqual(written(a.difference(b)), reference("set_difference", a, b))
        self.assertEqual(written(~a), reference("set_complement", a))
        self.assertEqual(written(a.complement()), reference("set_complement", a))
        self.assertEqual(written(a ^ b), reference("symmetric_difference", a, b))
        self.assertEqual(a.symmetric_difference(b), a ^ b)
        self.assertEqual(a ^ a, EMPTY)
        self.assertEqual(~EMPTY, WHOLE)
        self.assertEqual(~WHOLE, EMPTY)
        self.assertEqual(~iset(Interval(0, 1), Interval(1, 2)), iset(Interval(None, 0, False, True), point(q(1)),
                                                                     Interval(2, None, True, False)))
        self.assertEqual(a | EmptySet(Reals), a)
        self.assertEqual(EmptySet(Reals) | a, a)
        self.assertEqual(a & UniversalSet(Reals), a)
        self.assertEqual(UniversalSet(Reals) - a, ~a)
        self.assertEqual(a - UniversalSet(Reals), EMPTY)
        self.assertEqual(UniversalSet(Reals) ^ a, ~a)
        self.assertEqual(a.difference(EmptySet(Reals), limits=ComputationLimits()), a)
        for other in (1, q(1), "a", None, Interval(0, 1), (Interval(0, 1),)):
            with self.subTest(other=other):
                for apply in (lambda: a | other, lambda: other | a, lambda: a & other, lambda: a - other,
                              lambda: other - a, lambda: a ^ other, lambda: other ^ a):
                    with self.assertRaises(TypeError):
                        apply()
        # Unchecked, like union and intersection, but within the budget.
        with self.assertRaises(mf.ResourceLimitError):
            a.complement(limits=ComputationLimits(max_work=1))

    # -----------------------------------------------------------------------------------------------------------
    # Inputs

    def test_invalid_inputs(self):
        """0.5, True, "1", None, a wrong container, an EmptySet or a UniversalSet in every argument, and every record
        that no constructor made, are invalid_input, never the producer's internal_error."""
        a, b = iset(Interval(0, 1)), iset(Interval(2, 3, True, True))
        wrong = (*STAND_INS, EmptySet(Reals), UniversalSet(Reals), Interval(0, 1), (Interval(0, 1),),
                 [Interval(0, 1)], q(1), type("Sub", (IntervalSet,), {})((Interval(0, 1),)))
        for operation in OPERATIONS:
            for name in family._ARGUMENTS[operation]:
                for number, item in enumerate(wrong):
                    with self.subTest(operation=operation, argument=name, item=number):
                        args = dict(arguments(operation, a, b), **{name: item})
                        self.assertInvalidInput(support.run(operation, **args))
            with self.subTest(operation=operation, arguments="extra"):
                self.assertInvalidInput(operations_v6.run(operation, dict(a=a, b=b, c=a)))
            with self.subTest(operation=operation, arguments="missing"):
                self.assertInvalidInput(operations_v6.run(operation, {}))
        q0, q1, q2 = q(0), q(1), q(2)
        forgeries = {
            "a list of intervals": support.forged(IntervalSet, intervals=[Interval(0, 1)]),
            "no intervals field": support.forged(IntervalSet),
            "a tuple for an interval": forged_set((q0, q1, False, False)),
            "unsorted": forged_set(Interval(2, 3), Interval(0, 1)),
            "overlapping": forged_set(Interval(0, 2), Interval(1, 3)),
            "mergeable": forged_set(Interval(0, 1, False, True), Interval(1, 2)),
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
        }
        for label, item in forgeries.items():
            for operation in OPERATIONS:
                for name in family._ARGUMENTS[operation]:
                    with self.subTest(case=label, operation=operation, argument=name):
                        args = dict(arguments(operation, a, b), **{name: item})
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args))
                        if label not in INSPECTED:
                            self.assertInvalidInput(support.run(operation, **args))
        # Through produce() itself: an unknown operation and argument sets other than the operation's.
        with self.assertRaises(mf.UnsupportedOperation):
            family.produce("set_symmetric_difference", dict(a=a, b=b))
        for args in ({}, dict(a=a), dict(a=a, b=b, c=b), [("a", a), ("b", b)]):
            with self.subTest(args=args), self.assertRaises(InvalidInput):
                family.produce("set_union", args)

    def test_resource_limits(self):
        """An integer of 2**65536 in an end, and max_work=1, exceed the budget; and a claim that the default limits
        could not verify or reload is refused, whatever the caller's limits, while the largest accepted ones verify
        after a JSON round trip."""
        with computation(ComputationLimits(max_integer_bits=70_000), fresh=True):
            huge = iset(Interval(Rational(support.HUGE), None))
            huge_root = iset(Interval(forged_root((-support.HUGE, 0, 1), 1), None))
        small = iset(Interval(0, 1))
        for operation in OPERATIONS:
            for name in family._ARGUMENTS[operation]:
                for item in (huge, huge_root):
                    with self.subTest(operation=operation, argument=name, root=item is huge_root):
                        self.assertResourceLimit(support.run(operation, **dict(arguments(operation, small, small),
                                                                               **{name: item})))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1),
                                                     **arguments(operation, small, small)))
        # Beyond the bounds of the module: the checker's Sturm chain of 48 cubics (5,700,000 units), and 96 quadratics
        # sqrt(10^24 + k), whose evaluations at the samples need 4,130,000 units.
        # And two sets of 405 intervals with ends k/100, one more than the largest accepted: the Result would not
        # reload.
        # And two sets of 7 intervals with ends (10^10000 + k)/7, one more than the largest accepted: an earlier
        # estimate left out the comparisons and gcds of long Rationals, which are most of such a check, and accepted
        # n = 10 (2,190,000 units, estimated at 86,000), whose Result did not verify on reload.
        for kind, n, reason in (("cube", 12, "the check"), ("sqrt", 24, "the check"), ("k/100", 405, "reload"),
                                ("long", 7, "the check")):
            a, b = bound_sets(kind, n)
            with self.subTest(kind=kind, n=n):
                start = time.perf_counter()
                result = support.run("set_union", a=a, b=b, limits=WIDE)
                self.assertResourceLimit(result)
                self.assertIn(reason, result.error.message)
                self.assertIn("default limits", result.error.message)
                self.assertLess(time.perf_counter() - start, 3.0)

    def test_largest_accepted_inputs_reload_and_verify(self):
        """The largest inputs of the module's bounds give Results that reload and verify under the default limits:
        sqrt(10^24 + k) at n = 16, whose reload needs 91% of the refinements, real cube roots at n = 9, ends k/100
        at n = 404 and ends (10^10000 + k)/7 at n = 6."""
        for kind, n in (("sqrt", 16), ("cube", 9), ("k/100", 404), ("long", 6)):
            a, b = bound_sets(kind, n)
            with self.subTest(kind=kind, n=n):
                result = support.run("set_union", a=a, b=b, limits=WIDE)
                self.assertIsNone(result.error)
                decoded = mf.loads(mf.dumps(result))
                self.assertEqual(decoded, result)
                report = mf.verify(decoded)
                self.assertTrue(report.verified, support.statuses(report))

    # -----------------------------------------------------------------------------------------------------------
    # The checker

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: midpoint
        samples, and ends named by the other input."""
        a = iset(Interval(MINUS_SQRT2, SQRT2, True, True))
        b = iset(Interval(0, 1))
        self.assertAccepts("set_difference", dict(a=a, b=b),
                           iset(Interval(MINUS_SQRT2, 0, True, True), Interval(1, SQRT2, True, True)),
                           ((q(-2), q(-3, 4), q(1, 4), q(6, 5), q(7)),))
        a, b = iset(Interval(0, SQRT2_CUBIC, True, True)), iset(Interval(SQRT2, 3, True, True))
        self.assertAccepts("set_intersection", dict(a=a, b=b), iset(point(SQRT2)),  # Named as b names it.
                           family.produce("set_intersection", dict(a=a, b=b))[1])

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where a Rational belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        args = dict(a=KNOWN[4][1], b=KNOWN[4][2])
        value, evidence = family.produce("set_difference", dict(args))
        samples = evidence[0]
        cases = (
            ("wrong value", "set_difference", args, iset(Interval(MINUS_SQRT2, 0, True, True)), evidence),
            ("wrong operation", "set_union", args, value, evidence),
            ("unknown operation", "set_symmetric_difference", args, value, evidence),
            ("not canonical", "set_difference", args, forged_set(*reversed(value.intervals)), evidence),
            ("an EmptySet", "set_difference", args, EmptySet(Reals), evidence),
            ("impostor value", "set_difference", args, support.impostor(value), evidence),
            ("forged interval", "set_difference", args,
             forged_set(value.intervals[0], forged_interval(q(1), SQRT2, 1, True)), evidence),
            ("bool sample", "set_difference", args, value, ((True, *samples[1:]),)),
            ("float sample", "set_difference", args, value, ((-2.0, *samples[1:]),)),
            ("str sample", "set_difference", args, value, (("-2", *samples[1:]),)),
            ("list of samples", "set_difference", args, value, (list(samples),)),
            ("missing evidence", "set_difference", args, value, ()),
            ("extra evidence", "set_difference", args, value, (samples, samples)),
            ("a sample missing", "set_difference", args, value, (samples[1:],)),
            ("a sample at a point", "set_difference", args, value, ((q(-2), q(0), q(1, 2), q(5, 4), q(2)),)),
            ("oversized evidence", "set_difference", args, value, (tuple(q(k) for k in range(100_000)),)),
        )
        for label, operation, claim_args, claim, claim_evidence in cases:
            with self.subTest(case=label):
                self.assertRejects(operation, claim_args, claim, claim_evidence)
        result = support.run("set_difference", **args)
        report = mf.verify(support.tampered(result, iset(Interval(MINUS_SQRT2, 0, True, True))))
        self.assertFalse(report.verified)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        results = [support.run(operation, **arguments(operation, a, b)) for operation, a, b, _ in KNOWN]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for result, text in zip(results, texts):
                with self.subTest(operation=result.method):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            failed = support.run("set_complement", a=EMPTY)  # The patch is live: a new run reaches it and fails.
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    # -----------------------------------------------------------------------------------------------------------
    # Formats and paths

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: the empty set, the
        whole line, a point, unions of intervals and algebraic ends."""
        rendered = (
            ("set_union", KNOWN[0][1], KNOWN[0][2], "[0, 3)", "[0, 3)"),
            ("set_complement", KNOWN[2][1], None, "(-inf, 0) union [1, inf)", r"(-\infty, 0) \cup [1, \infty)"),
            ("set_complement", EMPTY, None, "(-inf, inf)", r"(-\infty, \infty)"),
            ("set_intersection", iset(Interval(0, 1)), iset(Interval(1, 2)), "{}", r"\varnothing"),
            ("set_intersection", iset(Interval(0, q(1, 2), False, True)), iset(Interval(q(1, 2), 1, True, False)),
             "[1/2, 1/2]", r"[\frac{1}{2}, \frac{1}{2}]"),
            ("set_difference", KNOWN[4][1], KNOWN[4][2],
             "[RootOf(t^2 - 2; real_index=0), 0] union [1, RootOf(t^2 - 2; real_index=1)]",
             r"[\operatorname{RootOf}_{\mathbb{R}}\left(t^{2} - 2; 0\right), 0] \cup "
             r"[1, \operatorname{RootOf}_{\mathbb{R}}\left(t^{2} - 2; 1\right)]"),
            ("set_complement", iset(Interval(None, CBRT2, False, True)), None,
             "(RootOf(t^3 - 2; real_index=0), inf)",
             r"(\operatorname{RootOf}_{\mathbb{R}}\left(t^{3} - 2; 0\right), \infty)"),
        )
        for operation, a, b, text, latex in rendered:
            with self.subTest(operation=operation, text=text):
                result = support.run(operation, **arguments(operation, a, b))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("set_complement", a=WHOLE).value)["data"],
                         {"type": "interval_set", "intervals": []})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261114)
        ends = [q(n, d) for n in range(-6, 7) for d in (1, 2)] + [SQRT2, MINUS_SQRT2, CBRT2]
        ends = sorted(set(ends), key=functools.cmp_to_key(lambda x, y: compare(key_of(x), key_of(y))))
        for trial in range(12):
            chosen = sorted(rng.sample(range(len(ends)), 8))
            a = iset(*[Interval(ends[chosen[k]], ends[chosen[k + 1]], k % 4 == 0, True) for k in range(0, 4, 2)])
            b = iset(*[Interval(ends[chosen[k]], ends[chosen[k + 1]], True, k % 4 == 0) for k in range(3, 7, 2)])
            for operation in OPERATIONS:
                args = arguments(operation, a, b)
                with self.subTest(trial=trial, operation=operation):
                    expected = support.run(operation, **args).value
                    self.assertIsNotNone(expected)
                    found = support.lean_value(operation, **args)
                    self.assertIsNot(found, lean.FALLBACK)
                    self.assertEqual(found, expected)
                    self.assertEqual(support.value(operation, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs each operation and returns the Result of the structured path."""
        friendly = {"set_union": {"a": [{"lower": 0, "upper": 2, "left_closed": True, "right_closed": True}],
                                  "b": [{"lower": 1, "upper": 3, "left_closed": True, "right_closed": False}]},
                    "set_complement": {"a": [{"lower": 0, "upper": "1/2", "left_closed": True,
                                              "right_closed": False}]}}
        for operation, raw in friendly.items():
            with self.subTest(operation=operation, form="friendly"):
                result = self.answer(support.call(operation, raw))
                self.assertTrue(mf.verify(result).verified)
                self.assertEqual(result, support.run(operation, **{name: support_set(raw[name]) for name in raw}))
        for operation in OPERATIONS:
            args = arguments(operation, KNOWN[4][1], iset(Interval(0, 1), point(CBRT2)))
            with self.subTest(operation=operation, form="values"):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        self.assertToolError(support.call("set_union", {"a": [], "b": 5}), "invalid_input")


def support_set(raw: list[dict]) -> IntervalSet:
    """The IntervalSet of a friendly JSON interval set of integer and fraction-text ends."""
    def end(value: object) -> object:
        return None if value is None else Rational(Fraction(value).numerator, Fraction(value).denominator)
    return iset(*[Interval(end(item["lower"]), end(item["upper"]), item["left_closed"], item["right_closed"])
                  for item in raw])


def bound_sets(kind: str, n: int) -> tuple[IntervalSet, IntervalSet]:
    """Two sets of n intervals with 4n distinct ends: closed ones with ends sqrt(10^24 + k) or the real cube roots of
    2 + k, or half-open ones with ends k/100 or (10^10000 + k)/7 (the measured families of the module docstring)."""
    if kind == "long":
        with computation(WIDE, fresh=True):  # Building the sets compares their 33,000-bit ends.
            firsts = [q(10 ** 10000 + 2 * k, 7) for k in range(2 * n)]
            seconds = [q(10 ** 10000 + 2 * k + 1, 7) for k in range(2 * n)]
            return (iset(*[Interval(firsts[2 * i], firsts[2 * i + 1], True, False) for i in range(n)]),
                    iset(*[Interval(seconds[2 * i], seconds[2 * i + 1], False, True) for i in range(n)]))
    if kind == "k/100":
        firsts = [q(k, 100) for k in range(-200 * n, 200 * n, 200)]
        seconds = [q(k, 100) for k in range(-200 * n + 100, 200 * n + 100, 200)]
        return (iset(*[Interval(firsts[2 * i], firsts[2 * i + 1], True, False) for i in range(n)]),
                iset(*[Interval(seconds[2 * i], seconds[2 * i + 1], False, True) for i in range(n)]))

    def end(k: int) -> RealAlgebraicRoot:
        if kind == "sqrt":
            return RealAlgebraicRoot((-(10 ** 24 + k), 0, 1), 1)
        return RealAlgebraicRoot((-(2 + k), 0, 0, 1), 0)
    first = [end(2 * k) for k in range(2 * n)]
    second = [end(2 * k + 1) for k in range(2 * n)]
    return (iset(*[Interval(first[2 * i], first[2 * i + 1], True, True) for i in range(n)]),
            iset(*[Interval(second[2 * i], second[2 * i + 1], True, True) for i in range(n)]))


if __name__ == "__main__":
    import unittest
    unittest.main()
