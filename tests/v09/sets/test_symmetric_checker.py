"""The independent checker of mathforge.areas.sets.symmetric (mathforge.certify.sets.symmetric), tested on its own
and independently of the family's producer.

Every true claim is computed here, never by the producer and never by IntervalSet.symmetric_difference (v0.7's method
serves only as a second opinion on the test's reference, on random sets). The test draws its own picture of the line,
Line: it isolates the real roots of every endpoint polynomial between rationals with a Sturm sequence over
fractions.Fraction and separates them from the rational endpoints and from each other. It writes each set as the
members of the line it holds (the open cells and the points), takes the members in exactly one of a and b, rebuilds the
canonical set from the maximal runs of true members and names each endpoint as an input names it. The samples are
rationals between the isolating intervals, their midpoints or, where a test says so, the rationals of least
denominator, in the layout of the family's docstring: (samples,). The checker answers False to every corrupted claim,
never with an exception, and refuses an oversized forgery within FAST seconds under the default budget.

Run this module by name with python -m unittest tests.v09.sets.test_symmetric_checker.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
import inspect
from math import ceil, floor
import random
import time

import mathforge as mf
from mathforge import contracts, registry
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.sets import symmetric as family
from mathforge.certify import cells
from mathforge.certify.sets import symmetric as checker
from mathforge.contracts import CheckStatus, EmptySet, UniversalSet
from mathforge.errors import UnsupportedOperation
from mathforge.intervals import Interval, IntervalSet
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational, Reals, trusted_rational
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
         "and decimal digits")
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
F = Fraction
OPERATION = "set_symmetric_difference"

# Endpoint polynomials, lowest degree first, primitive and square-free with a positive leading coefficient.
SQUARE_2 = (-2, 0, 1)  # x^2 - 2: -sqrt 2, sqrt 2.
SQUARE_3 = (-3, 0, 1)  # x^2 - 3: -sqrt 3, sqrt 3.
GOLDEN = (-1, -1, 1)  # x^2 - x - 1: (1 - sqrt 5)/2, (1 + sqrt 5)/2.
HALF_SQUARE = (-1, 0, 2)  # 2x^2 - 1: -1/sqrt 2, 1/sqrt 2.
CUBE_2 = (-2, 0, 0, 1)  # x^3 - 2: one real root and two complex ones, so the cut needs a Sturm chain.
TRIPLE = (1, -3, 0, 1)  # x^3 - 3x + 1: three real roots, about -1.879, 0.347 and 1.532.
SQRT2, MINUS_SQRT2 = RealAlgebraicRoot(SQUARE_2, 1), RealAlgebraicRoot(SQUARE_2, 0)
SQRT3, MINUS_SQRT3 = RealAlgebraicRoot(SQUARE_3, 1), RealAlgebraicRoot(SQUARE_3, 0)


# ---------------------------------------------------------------------------------------------------------------
# The test's reference: polynomials over Fraction, Sturm's theorem and the picture of the line.

def evaluate(p: Sequence[object], x: Fraction) -> Fraction:
    value = F(0)
    for coefficient in reversed(p):
        value = value * x + coefficient
    return value


def sign(x: Fraction) -> int:
    return (x > 0) - (x < 0)


def remainder(p: Sequence[object], q: Sequence[Fraction]) -> list[Fraction]:
    """The remainder of p by a nonzero q over the rationals, without trailing zeros."""
    rest = [F(c) for c in p]
    while rest and len(rest) >= len(q):
        factor, shift = rest[-1] / q[-1], len(rest) - len(q)
        for index, c in enumerate(q):
            rest[shift + index] -= factor * c
        while rest and rest[-1] == 0:
            rest.pop()
    return rest


def sturm(p: Sequence[int]) -> list[list[Fraction]]:
    """The classic Sturm sequence p, p', -rem(p, p'), ... of a square-free p."""
    chain = [[F(c) for c in p], [F(index * c) for index, c in enumerate(p)][1:]]
    while True:
        rest = remainder(chain[-2], chain[-1])
        if not rest:
            return chain
        chain.append([-c for c in rest])


def roots_between(chain: list[list[Fraction]], low: Fraction, high: Fraction) -> int:
    """Sturm's theorem: the distinct real roots of chain[0] in (low, high), neither end a root."""
    def variations(x: Fraction) -> int:
        signs = [s for s in (sign(evaluate(member, x)) for member in chain) if s]
        return sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    return variations(low) - variations(high)


def isolate(p: Sequence[int]) -> list[tuple[Fraction, Fraction]]:
    """Rational brackets of the distinct real roots of a square-free integer polynomial, in increasing order: (q, q)
    for a rational root q met exactly, else (low, high) with low < root < high and no other root in [low, high]."""
    chain = sturm(p)
    bound = 1 + max(abs(F(c, p[-1])) for c in p[:-1])  # Cauchy: every root lies in (-bound, bound).
    found, pending = [], [(-bound, bound)]
    while pending:
        low, high = pending.pop()
        count = roots_between(chain, low, high)
        if count == 0:
            continue
        middle = (low + high) / 2
        if evaluate(p, middle) == 0:
            found.append((middle, middle))
            width = (high - low) / 4
            while (evaluate(p, middle - width) == 0 or evaluate(p, middle + width) == 0
                   or roots_between(chain, middle - width, middle + width) != 1):
                width /= 2
            pending += [(low, middle - width), (middle + width, high)]
        elif count == 1:
            found.append((low, high))
        else:
            pending += [(low, middle), (middle, high)]
    return sorted(found)


def key_of(end: object) -> tuple:
    """The test's name of the number an endpoint stands for: ("q", value) or ("root", polynomial, index)."""
    if type(end) is Rational:
        return ("q", F(end.numerator, end.denominator))
    coefficients, index = end.integer_coefficients, end.real_index
    if len(coefficients) == 2:
        return ("q", F(-coefficients[0], coefficients[1]))
    return ("root", coefficients, index)


class Line:
    """The points of a cut as the test sees them: the rationals given and every real root of the polynomials given,
    in increasing order and each value once, with rational brackets that separate them (low == high for a rational
    point). number maps every name of a point to its number from 1."""

    def __init__(self, polynomials: Iterable[tuple[int, ...]] = (), rationals: Iterable[Fraction] = ()) -> None:
        self._exact: dict[Fraction, set] = {}
        self._brackets: list[list] = []  # [low, high, polynomial, {key}] of the roots not met exactly.
        for value in rationals:
            self._point(F(value), ("q", F(value)))
        for p in dict.fromkeys(polynomials):
            for index, (low, high) in enumerate(isolate(p)):
                if low == high:
                    self._point(low, ("root", p, index))
                else:
                    self._brackets.append([low, high, p, {("root", p, index)}])
        for _ in range(500):  # Each pass halves every bracket that meets a neighbour.
            items = self._items()
            clashes = [(left, right) for left, right in zip(items, items[1:]) if left[1] >= right[0]]
            if not clashes:
                break
            for left, right in clashes:
                self._separate(left, right)
        else:
            raise AssertionError("one irrational number under two polynomials: the test's data must avoid it")
        items = self._items()
        self.points = [(item[0], item[1]) for item in items]
        self.number = {key: k for k, item in enumerate(items, 1) for key in item[3]}

    def _point(self, value: Fraction, key: tuple) -> None:
        self._exact.setdefault(value, set()).update({key, ("q", value)})

    def _items(self) -> list[list]:
        exact = [[value, value, None, keys] for value, keys in self._exact.items()]
        return sorted(exact + self._brackets, key=lambda item: (item[0], item[1]))

    def _separate(self, left: list, right: list) -> None:
        for item, other in ((left, right), (right, left)):
            p = item[2]
            if p is None:
                continue
            low, high = item[0], item[1]
            at = other[0] if other[2] is None and low < other[0] < high else (low + high) / 2
            if evaluate(p, at) == 0:
                self._brackets = [bracket for bracket in self._brackets if bracket is not item]
                for key in item[3]:
                    self._point(at, key)
                item[2] = None
                return
            item[0], item[1] = (low, at) if sign(evaluate(p, low)) != sign(evaluate(p, at)) else (at, high)

    def samples(self, between: Callable[[Fraction, Fraction], Fraction] = lambda low, high: (low + high) / 2
                ) -> tuple[Rational, ...]:
        """One rational below the points, one between any two neighbours and one above: the evidence's samples."""
        if not self.points:
            return (Rational(0),)
        values = [F(floor(self.points[0][0]) - 1)]
        values += [between(high, low) for (_, high), (low, _) in zip(self.points, self.points[1:])]
        values.append(F(ceil(self.points[-1][1]) + 1))
        return tuple(Rational(value.numerator, value.denominator) for value in values)


def least_between(low: Fraction, high: Fraction) -> Fraction:
    """The rational of least denominator in the open interval (low, high), low < high (the Stern-Brocot tree)."""
    if low < 0 < high:
        return F(0)
    if high <= 0:
        return -least_between(-high, -low)
    n = floor(low)
    if n + 1 < high:
        return F(n + 1)
    if low == n:
        return n + F(1, floor(1 / (high - n)) + 1)
    return n + 1 / least_between(1 / (high - n), 1 / (low - n))


def line_of(*sets: IntervalSet) -> Line:
    polynomials, rationals = [], []
    for value in sets:
        for item in value.intervals:
            for end in (item.lower, item.upper):
                if end is not None:
                    key = key_of(end)
                    (rationals if key[0] == "q" else polynomials).append(key[1])
    return Line(polynomials, rationals)


def members(line: Line, value: IntervalSet) -> list[bool]:
    """Whether the set holds each member of the line: member 2j is the open cell number j, member 2k - 1 the point k."""
    r = len(line.points)
    truth = [False] * (2 * r + 1)
    for item in value.intervals:
        first = 0 if item.lower is None else 2 * line.number[key_of(item.lower)] - item.left_closed
        last = 2 * r if item.upper is None else 2 * line.number[key_of(item.upper)] - 2 + item.right_closed
        for member in range(first, last + 1):
            truth[member] = True
    return truth


def runs(truth: Sequence[bool]) -> list[tuple[int | None, int | None, bool, bool]]:
    """The maximal runs of true members as intervals over point numbers, None at an infinite end."""
    last, intervals, start = len(truth) - 1, [], 0
    while start <= last:
        if not truth[start]:
            start += 1
            continue
        stop = start
        while stop < last and truth[stop + 1]:
            stop += 1
        lower = (None, False) if start == 0 else ((start + 1) // 2, True) if start % 2 else (start // 2, False)
        upper = (None, False) if stop == last else ((stop + 1) // 2, True) if stop % 2 else (stop // 2 + 1, False)
        intervals.append((lower[0], upper[0], lower[1], upper[1]))
        start = stop + 1
    return intervals


def name_of(line: Line, sets: Sequence[IntervalSet], k: int) -> object:
    """An endpoint record for the point k: the first that an input writes, else the root of an input's polynomial."""
    for value in sets:
        for item in value.intervals:
            for end in (item.lower, item.upper):
                if end is not None and line.number[key_of(end)] == k:
                    return end
    for key, number in line.number.items():
        if number == k and key[0] == "root":
            return RealAlgebraicRoot(key[1], key[2])
    raise AssertionError(f"no name for the point {k}")


def assemble(line: Line, sets: Sequence[IntervalSet], truth: Sequence[bool]) -> IntervalSet:
    """The canonical IntervalSet of truth values over the members of the line, its ends named by the inputs."""
    records = tuple(Interval(None if lower is None else name_of(line, sets, lower),
                             None if upper is None else name_of(line, sets, upper), left, right)
                    for lower, upper, left, right in runs(truth))
    value = IntervalSet(records)
    assert value.intervals == records, "the runs of true members are canonical already"
    return value


def truth_of(line: Line, a: IntervalSet, b: IntervalSet) -> list[bool]:
    """The members in exactly one of a and b."""
    return [in_a != in_b for in_a, in_b in zip(members(line, a), members(line, b))]


def true_claim(a: IntervalSet, b: IntervalSet, between: Callable[[Fraction, Fraction], Fraction] | None = None
               ) -> tuple[dict, IntervalSet, tuple]:
    """(args, value, evidence) of the true claim, all computed by the test."""
    line = line_of(a, b)
    samples = line.samples() if between is None else line.samples(between)
    return dict(a=a, b=b), assemble(line, (a, b), truth_of(line, a, b)), (samples,)


def claimed(args: dict[str, object], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(OPERATION, tuple(args.items()), 6)
    step = contracts.Step("v6." + OPERATION, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=OPERATION, request=request,
                            steps=(step,))


# ---------------------------------------------------------------------------------------------------------------
# Records

def q(numerator: int, denominator: int = 1) -> Rational:
    return Rational(numerator, denominator)


def iset(*intervals: Interval) -> IntervalSet:
    return IntervalSet(tuple(intervals))


def point(x: object) -> Interval:
    return Interval(x, x, True, True)


def forged_set(*intervals: object) -> IntervalSet:
    return support.forged(IntervalSet, intervals=tuple(intervals))


def forged_interval(lower: object, upper: object, left: object = False, right: object = False) -> Interval:
    return support.forged(Interval, lower=lower, upper=upper, left_closed=left, right_closed=right)


def forged_root(coefficients: object, index: object) -> RealAlgebraicRoot:
    return support.forged(RealAlgebraicRoot, integer_coefficients=coefficients, real_index=index)


def replaced(value: IntervalSet, position: int, interval: object) -> IntervalSet:
    items = list(value.intervals)
    items[position] = interval
    return forged_set(*items)


EMPTY = IntervalSet(())
WHOLE = IntervalSet((Interval(None, None),))
Evidence = type("Evidence", (tuple,), {})  # A tuple subclass.
Arguments = type("Arguments", (dict,), {})  # A dict subclass.


class SubRational(Rational):
    """A Rational subclass, equal to its value but no exact Rational."""


# The base claim: a = [-sqrt 2, 1/2) | {1} | (sqrt 3, inf) and b = (-inf, -1] | [0, sqrt 2], whose cut has the points
# -sqrt 3, -sqrt 2, -1, 0, 1/2, 1, sqrt 2 and sqrt 3 (-sqrt 3 being no endpoint). a ^ b, member by member by hand:
# below -sqrt 2 in b alone, then in both up to -1, in a alone on (-1, 0), in both on [0, 1/2), in b alone on [1/2, 1),
# in both at 1, in b alone on (1, sqrt 2], in neither on (sqrt 2, sqrt 3], in a alone above sqrt 3.
BASE_A = iset(Interval(MINUS_SQRT2, q(1, 2), True, False), point(q(1)), Interval(SQRT3, None))
BASE_B = iset(Interval(None, q(-1), False, True), Interval(q(0), SQRT2, True, True))
BASE_VALUE = iset(Interval(None, MINUS_SQRT2), Interval(q(-1), q(0)), Interval(q(1, 2), q(1), True, False),
                  Interval(q(1), SQRT2, False, True), Interval(SQRT3, None))
BASE_SAMPLES = (q(-2), q(-3, 2), q(-6, 5), q(-1, 2), q(1, 4), q(3, 4), q(6, 5), q(3, 2), q(2))  # By hand.


def base_claim() -> tuple[dict, IntervalSet, tuple]:
    return true_claim(BASE_A, BASE_B)


class SymmetricCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, cases: Iterable[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case of set_symmetric_difference."""
        for label, args, value, evidence in cases:
            with self.subTest(case=label):
                self.assertRejects(OPERATION, args, value, evidence)

    def assertTrueClaim(self, a: IntervalSet, b: IntervalSet, expected: IntervalSet | None = None
                        ) -> tuple[dict, IntervalSet, tuple]:
        """The test's true claim is accepted (with midpoints and with samples of least denominator), stays within
        the bound of the checker's condition 1, and equals the expected value when one is given."""
        args, value, evidence = true_claim(a, b)
        if expected is not None:
            self.assertEqual(value, expected)
        self.assertLessEqual(len(value), 2 * (len(a) + len(b)))
        self.assertAccepts(OPERATION, args, value, evidence)
        self.assertAccepts(OPERATION, args, value, true_claim(a, b, least_between)[2])
        return args, value, evidence

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        a, b = Argument("a", "IntervalSet", None, "interval_set"), Argument("b", "IntervalSet", None, "interval_set")
        expected = (OperationSpec(OPERATION, "sets", (a, b), "IntervalSet",
                                  "Take the symmetric difference of two finite unions of real intervals.", SCOPE,
                                  version=6),)
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {})  # v0.7's IntervalSet records: no record class of its own.
        self.assertFalse(hasattr(family, "RENDER"))
        self.assertEqual(family.CHECKER, "mathforge.certify.sets.symmetric")
        self.assertEqual(registry.spec(OPERATION), expected[0])
        self.assertIs(registry.checker(OPERATION), checker)
        self.assertIs(support.checker(family), checker)
        with self.assertRaises(UnsupportedOperation):  # The producer serves only its own operation.
            family.produce("set_union", dict(a=BASE_A, b=BASE_B))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_accepts_the_known_values_of_the_card(self):
        cases = (
            (iset(Interval(0, 2, True, True)), iset(Interval(1, 3, True, False)),
             iset(Interval(0, 1, True, False), Interval(2, 3))),
            (iset(Interval(0, 1, True, True)), iset(Interval(0, 1, True, True)), EMPTY),
            (WHOLE, iset(point(0)), iset(Interval(None, 0), Interval(0, None))),
            (iset(Interval(MINUS_SQRT2, SQRT2, True, True)), iset(Interval(0, 2, True, True)),
             iset(Interval(MINUS_SQRT2, 0, True, False), Interval(SQRT2, 2, False, True))),
        )
        for a, b, expected in cases:
            with self.subTest(a=str(a), b=str(b)):
                self.assertTrueClaim(a, b, expected)
        # The evidence of the last one by hand: the points -sqrt 2, 0, sqrt 2 and 2.
        self.assertAccepts(OPERATION, dict(a=cases[-1][0], b=cases[-1][1]), cases[-1][2],
                           ((q(-2), q(-1), q(1), q(3, 2), q(3)),))
        # And of the first: the points 0, 1, 2, 3.
        self.assertAccepts(OPERATION, dict(a=cases[0][0], b=cases[0][1]), cases[0][2],
                           ((q(-1), q(1, 2), q(3, 2), q(5, 2), q(4)),))

    def test_accepts_the_special_cases(self):
        unit = iset(Interval(0, 1, True, True))
        cases = (
            (EMPTY, EMPTY, EMPTY),
            (EMPTY, unit, unit),
            (unit, EMPTY, unit),
            (WHOLE, WHOLE, EMPTY),
            (WHOLE, EMPTY, WHOLE),
            (EMPTY, WHOLE, WHOLE),
            (unit, WHOLE, iset(Interval(None, 0), Interval(1, None))),
            (iset(Interval(0, 1, True, False)), iset(Interval(1, 2, True, True)), iset(Interval(0, 2, True, True))),
            (iset(Interval(0, 1)), iset(Interval(1, 2)), iset(Interval(0, 1), Interval(1, 2))),
            (iset(Interval(0, 1)), iset(point(1)), iset(Interval(0, 1, False, True))),
            (iset(point(1)), iset(point(1)), EMPTY),
            (iset(point(1)), iset(point(2)), iset(point(1), point(2))),
            (unit, iset(Interval(0, 1)), iset(point(0), point(1))),
            (unit, iset(point(q(1, 2))), iset(Interval(0, q(1, 2), True, False), Interval(q(1, 2), 1, False, True))),
            (iset(Interval(0, 3, True, True)), iset(Interval(1, 2)),
             iset(Interval(0, 1, True, True), Interval(2, 3, True, True))),
            (iset(Interval(0, 1, True, True)), iset(Interval(2, 3, True, True)),
             iset(Interval(0, 1, True, True), Interval(2, 3, True, True))),
            (iset(Interval(None, 0, False, True)), iset(Interval(0, None, True, False)),
             iset(Interval(None, 0), Interval(0, None))),
            (iset(Interval(None, 1, False, True)), iset(Interval(0, None, True, False)),
             iset(Interval(None, 0), Interval(1, None))),
            (iset(Interval(None, 0)), iset(Interval(0, None, True, False)), WHOLE),
            (iset(Interval(None, 0), Interval(0, None)), iset(point(0)), WHOLE),
            (iset(Interval(MINUS_SQRT2, SQRT2)), iset(Interval(SQRT2, SQRT3)),
             iset(Interval(MINUS_SQRT2, SQRT2), Interval(SQRT2, SQRT3))),
            (iset(Interval(MINUS_SQRT2, SQRT2, True, True)), iset(Interval(SQRT2, 3, True, True)),
             iset(Interval(MINUS_SQRT2, SQRT2, True, False), Interval(SQRT2, 3, False, True))),
            (iset(point(SQRT2), point(SQRT3)), WHOLE,
             iset(Interval(None, SQRT2), Interval(SQRT2, SQRT3), Interval(SQRT3, None))),
        )
        for a, b, expected in cases:
            with self.subTest(a=str(a), b=str(b)):
                self.assertTrueClaim(a, b, expected)
        self.assertAccepts(OPERATION, dict(a=EMPTY, b=EMPTY), EMPTY, ((q(5, 7),),))  # No point: any one sample.
        self.assertAccepts(OPERATION, dict(a=WHOLE, b=EMPTY), WHOLE, ((q(-10**30),),))

    def test_accepts_the_base_claim(self):
        args, value, evidence = base_claim()
        self.assertEqual(value, BASE_VALUE)
        self.assertAccepts(OPERATION, args, value, evidence)
        self.assertAccepts(OPERATION, args, value, (BASE_SAMPLES,))
        self.assertAccepts(OPERATION, dict(a=BASE_B, b=BASE_A), BASE_VALUE, (BASE_SAMPLES,))  # Symmetric.
        # The samples certify the cut that the card names, cells.endpoint_cut(a, b, value), too.
        master, cuts = cells.endpoint_cut(BASE_A, BASE_B, value)
        self.assertTrue(cells.check_samples(master, BASE_SAMPLES, cuts))

    def test_accepts_algebraic_endpoints_of_every_kind(self):
        """Quadratic irrationals, roots of a cubic with three real roots, and the real cube root of 2, whose
        polynomial has two complex roots, so that the cut takes the Sturm chain."""
        golden = (RealAlgebraicRoot(GOLDEN, 0), RealAlgebraicRoot(GOLDEN, 1))
        half = (RealAlgebraicRoot(HALF_SQUARE, 0), RealAlgebraicRoot(HALF_SQUARE, 1))
        triple = tuple(RealAlgebraicRoot(TRIPLE, index) for index in range(3))
        cube = RealAlgebraicRoot(CUBE_2, 0)
        pairs = (
            (iset(Interval(golden[0], 0, True, True), Interval(1, golden[1], True, False)),
             iset(Interval(q(-1, 3), q(3, 2), True, True))),
            (iset(Interval(half[0], half[1])), iset(Interval(0, q(1, 2), True, True), point(golden[1]))),
            (iset(Interval(None, triple[0], False, True), Interval(triple[1], triple[2], False, True)),
             iset(Interval(q(-2), q(1, 3), True, False), Interval(SQRT2, None, True, False))),
            (iset(Interval(0, cube), Interval(cube, 2, False, True)), iset(Interval(1, q(3, 2), True, True))),
            (iset(Interval(cube, None, True, False)), iset(Interval(None, golden[1], False, True), point(3))),
        )
        for a, b in pairs:
            with self.subTest(a=str(a), b=str(b)):
                self.assertTrueClaim(a, b)
                self.assertTrueClaim(b, a)

    def test_accepts_every_name_the_inputs_give(self):
        """An endpoint may be written as an input writes it, a rational one also as a Rational or as the root of a
        linear polynomial whenever an input has it as such, and a root of an input's polynomial by its index."""
        # sqrt 2 under two polynomials, x^2 - 2 in a and x^3 - 2x in b: the points -sqrt 2, 0, sqrt 2 and 3.
        other = RealAlgebraicRoot((0, -2, 0, 1), 2)
        a, b = iset(Interval(0, SQRT2, True, True)), iset(Interval(other, 3, True, True))
        samples = ((q(-2), q(-1), q(1), q(2), q(4)),)
        for name in (SQRT2, other):
            for second in (SQRT2, other):
                value = iset(Interval(0, name, True, False), Interval(second, 3, False, True))
                with self.subTest(value=str(value)):
                    self.assertAccepts(OPERATION, dict(a=a, b=b), value, samples)
        # 1/2 as the root of 2x - 1 in an input: the claim may write it either way.
        linear = RealAlgebraicRoot((-1, 2), 0)
        for written in (linear, q(1, 2)):
            for named in (linear, q(1, 2)):
                with self.subTest(written=str(written), named=str(named)):
                    args, _, evidence = true_claim(iset(Interval(written, 1, True, True)), WHOLE)
                    self.assertAccepts(OPERATION, args, iset(Interval(None, named), Interval(q(1), None)), evidence)
        # -sqrt 2 as the root number 0 of x^2 - 2, b's polynomial, an index that no input uses (a writes it under
        # x^3 - 2x). The points are -sqrt 2, 0, 1 and sqrt 2; a ^ b = [-sqrt 2, 0) | (1, sqrt 2].
        a = iset(Interval(RealAlgebraicRoot((0, -2, 0, 1), 0), 1, True, True))
        b = iset(Interval(0, SQRT2, True, True))
        samples = ((q(-2), q(-1), q(1, 2), q(6, 5), q(2)),)
        for name in (MINUS_SQRT2, RealAlgebraicRoot((0, -2, 0, 1), 0)):
            with self.subTest(name=str(name)):
                self.assertAccepts(OPERATION, dict(a=a, b=b),
                                   iset(Interval(name, 0, True, False), Interval(1, SQRT2, False, True)), samples)
        # 1 as the root number 1 of (x - 1)(x^2 - 2), an input's polynomial of degree three.
        reducible = RealAlgebraicRoot((2, -2, -1, 1), 1)
        args, value, evidence = true_claim(iset(Interval(reducible, 2, True, True)), WHOLE)
        self.assertEqual(value, iset(Interval(None, reducible), Interval(q(2), None)))
        self.assertAccepts(OPERATION, args, value, evidence)

    def test_accepts_samples_the_producer_would_not_choose(self):
        args, value, evidence = base_claim()
        samples = list(evidence[0])
        line = line_of(BASE_A, BASE_B)
        far = [q(-10**40)] + samples[1:-1] + [Rational(10**40, 3)]
        tiny = [sample + Rational(1, 2**300) for sample in samples]
        lower = [Rational(*(low - F(1, 10**6)).as_integer_ratio()) for low, _ in line.points] + [q(10)]
        least = list(line.samples(least_between))
        for label, chosen in (("far", far), ("tiny offsets", tiny), ("by hand", list(BASE_SAMPLES)),
                              ("next to points", lower), ("least denominators", least)):
            with self.subTest(samples=label):
                self.assertAccepts(OPERATION, args, value, (tuple(chosen),))

    def test_accepts_random_claims_and_rejects_their_mutations(self):
        """Random sets over a pool of rationals and of roots of six polynomials: every true claim is accepted and
        within the bound 2(m + n), v0.7's symmetric_difference agrees with the test's reference, and the set of a
        truth assignment that differs in one member is refused."""
        polynomials = (SQUARE_2, SQUARE_3, GOLDEN, HALF_SQUARE, CUBE_2, TRIPLE)
        rationals = (q(-3), q(-5, 2), q(-1), q(-1, 3), q(0), q(1, 2), q(1), q(7, 5), q(2), q(3))
        pool = Line(polynomials, [F(r.numerator, r.denominator) for r in rationals])
        names = list(rationals) + [RealAlgebraicRoot(p, index) for p in polynomials
                                   for index in range(len(isolate(p)))]
        names.sort(key=lambda end: pool.number[key_of(end)])
        rng = random.Random(20261113)

        def random_set() -> IntervalSet:
            chosen = sorted(rng.sample(range(len(names)), rng.randint(0, 6)))
            truth = [rng.random() < 0.5 for _ in range(2 * len(chosen) + 1)]
            return IntervalSet(tuple(
                Interval(None if lower is None else names[chosen[lower - 1]],
                         None if upper is None else names[chosen[upper - 1]], left, right)
                for lower, upper, left, right in runs(truth)))

        accepted = refused = largest = 0
        for _ in range(100):
            a, b = random_set(), random_set()
            args, value, evidence = true_claim(a, b)
            with self.subTest(a=str(a), b=str(b)):
                self.assertAccepts(OPERATION, args, value, evidence)
                self.assertEqual(value, a.symmetric_difference(b))
                self.assertLessEqual(len(value), len(a) + len(b))  # The tight bound of the checker's docstring.
                largest = max(largest, len(value))
                accepted += 1
                line = line_of(a, b)
                truth = truth_of(line, a, b)
                for member in rng.sample(range(len(truth)), min(2, len(truth))):
                    flipped = list(truth)
                    flipped[member] = not flipped[member]
                    self.assertRejects(OPERATION, args, assemble(line, (a, b), flipped), evidence)
                    refused += 1
        self.assertEqual((accepted, refused >= accepted, largest > 3), (100, True, True))

    def test_the_bound_admits_the_largest_true_sets(self):
        """The largest true sets have len(a) + len(b) intervals (the checker's docstring proves that no true set has
        more), half the bound 2(m + n) of condition 1, which therefore never refuses a true claim: n points in the
        whole line, interleaved half-open intervals, points against points, and rays at both ends."""
        cases = [(iset(*[point(k) for k in range(n)]), WHOLE) for n in (1, 2, 5)]
        cases += [
            (iset(*[Interval(4 * k, 4 * k + 2, True, False) for k in range(4)]),
             iset(*[Interval(4 * k + 1, 4 * k + 3, False, True) for k in range(4)])),
            (iset(*[point(2 * k) for k in range(4)]), iset(*[point(2 * k + 1) for k in range(3)])),
            (iset(Interval(None, 0), Interval(2, 3, True, True)), iset(Interval(1, 4, True, True), Interval(5, None))),
            (iset(Interval(None, 0, False, True), Interval(1, None, True, False)), iset(Interval(0, 1, True, True))),
        ]
        for a, b in cases:
            with self.subTest(a=str(a), b=str(b)):
                args, value, evidence = self.assertTrueClaim(a, b)
                self.assertEqual(len(value), len(a) + len(b))

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims, which the checker refuses with False

    def test_rejects_wrong_values(self):
        args, value, evidence = base_claim()
        line = line_of(BASE_A, BASE_B)
        truth = truth_of(line, BASE_A, BASE_B)
        cases = []
        for member in range(len(truth)):  # Every set that differs from the true one in one member.
            flipped = list(truth)
            flipped[member] = not flipped[member]
            cases.append((f"member {member} flipped", args, assemble(line, (BASE_A, BASE_B), flipped), evidence))
        union = iset(Interval(None, SQRT2, False, True), Interval(SQRT3, None))
        cases += [
            ("the union", args, union, evidence),
            ("the empty set", args, EMPTY, evidence),
            ("the whole line", args, WHOLE, evidence),
            ("an interval dropped", args, iset(*value.intervals[1:]), evidence),
            ("a itself", args, BASE_A, evidence),
            ("b itself", args, BASE_B, evidence),
            ("sqrt 3 replaced by -sqrt 3", args, iset(*value.intervals[:-1], Interval(MINUS_SQRT3, None)), evidence),
            ("a closure flipped", args, replaced(value, 1, Interval(q(-1), q(0), True, False)), evidence),
            ("the point 1 added", args, iset(*value.intervals[:2], Interval(q(1, 2), SQRT2, True, True),
                                             value.intervals[4]), evidence),
            ("a ^ b claimed for a ^ a", dict(a=BASE_A, b=BASE_A), value, evidence),
        ]
        self.assertRejectsAll(cases)

    def test_rejects_noncanonical_values_that_are_mathematically_right(self):
        args, value, evidence = base_claim()  # (-inf, -sqrt 2) | (-1, 0) | [1/2, 1) | (1, sqrt 2] | (sqrt 3, inf).
        first, second, third, fourth, fifth = value.intervals
        written = {
            "split at an inner point": (Interval(None, q(-2)), Interval(q(-2), MINUS_SQRT2, True, False), second,
                                        third, fourth, fifth),
            "split at an inner point, closed twice": (first, Interval(q(-1), q(-1, 2), False, True),
                                                      Interval(q(-1, 2), q(0), True, False), third, fourth, fifth),
            "overlapping": (first, second, Interval(q(1, 2), q(1), True, False),
                            Interval(q(3, 4), q(1), True, False), fourth, fifth),
            "nested": (first, second, third, point(q(3, 4)), fourth, fifth),
            "unsorted": (second, first, third, fourth, fifth),
            "twice the same interval": (first, first, second, third, fourth, fifth),
            "an empty open interval": (first, second, forged_interval(q(1), q(1)), third, fourth, fifth),
            "a reversed interval": (first, second, third, forged_interval(SQRT2, q(1)), fifth),
            "minus infinity closed": (forged_interval(None, MINUS_SQRT2, True, False), second, third, fourth, fifth),
            "plus infinity closed": (first, second, third, fourth, forged_interval(SQRT3, None, False, True)),
        }
        cases = [(label, args, forged_set(*items), evidence) for label, items in written.items()]
        # Endpoints under names that no input gives: the checker's condition 2.
        foreign = {
            "sqrt 2 as a root of (x^2 - 2)(x^2 + 1)": RealAlgebraicRoot((-2, 0, -1, 0, 1), 1),
            "sqrt 2 as a root of x^3 - 2x": RealAlgebraicRoot((0, -2, 0, 1), 2),
        }
        for label, name in foreign.items():
            cases.append((label, args, replaced(value, 3, Interval(q(1), name, False, True)), evidence))
        cases.append(("1/2 as a root of (2x - 1)(x^2 - 2)", args,
                      replaced(value, 2, Interval(RealAlgebraicRoot((2, -4, -1, 2), 1), q(1), True, False)), evidence))
        cases.append(("1 as a root of x^2 - 1", args,
                      replaced(value, 2, Interval(q(1, 2), RealAlgebraicRoot((-1, 0, 1), 1), True, False)), evidence))
        # A Rational for the rational root of an input's polynomial of degree three.
        reducible = RealAlgebraicRoot((2, -2, -1, 1), 1)
        other_args, _, other_evidence = true_claim(iset(Interval(reducible, 2, True, True)), WHOLE)
        cases.append(("1 as a Rational where the input has a root of (x - 1)(x^2 - 2)", other_args,
                      iset(Interval(None, q(1)), Interval(q(2), None)), other_evidence))
        # The empty set and the whole line as the solution-set records of v0.7's solve.
        for a, b, record in ((BASE_A, BASE_A, EmptySet(Reals)), (EMPTY, EMPTY, EmptySet(Reals)),
                             (WHOLE, EMPTY, UniversalSet(Reals)), (iset(Interval(None, 0)),
                                                                    iset(Interval(0, None, True, False)),
                                                                    UniversalSet(Reals))):
            record_args, true_value, record_evidence = true_claim(a, b)
            self.assertAccepts(OPERATION, record_args, true_value, record_evidence)
            cases.append((f"{type(record).__name__} in place of {true_value}", record_args, record, record_evidence))
        self.assertRejectsAll(cases)

    def test_rejects_impostors_and_forged_records(self):
        args, value, evidence = base_claim()
        fifth = value.intervals[4]
        prefix = value.intervals[:3]

        def with_end(end: object, *, left: object = False, right: object = True) -> IntervalSet:
            return forged_set(*prefix, forged_interval(q(1), end, left, right), fifth)

        cases = [
            ("an impostor IntervalSet", args, support.impostor(value), evidence),
            ("an impostor Interval", args, replaced(value, 0, support.impostor(value.intervals[0])), evidence),
            ("impostor inputs", {name: support.impostor(record) for name, record in args.items()}, value, evidence),
            ("an impostor a", dict(args, a=support.impostor(BASE_A)), value, evidence),
            ("an impostor RealAlgebraicRoot", args, with_end(support.impostor(SQRT2)), evidence),
            ("an IntervalSet without intervals", args, support.forged(IntervalSet), evidence),
            ("an Interval without its upper end", args,
             forged_set(*prefix, support.forged(Interval, lower=q(1), left_closed=False, right_closed=True), fifth),
             evidence),
            ("a RealAlgebraicRoot without fields", args, with_end(support.forged(RealAlgebraicRoot)), evidence),
            ("an index past the degree", args, with_end(forged_root(SQUARE_2, 2)), evidence),
            ("a negative index", args, with_end(forged_root(SQUARE_2, -1)), evidence),
            ("a bool index", args, with_end(forged_root(SQUARE_2, True)), evidence),
            ("a float index", args, with_end(forged_root(SQUARE_2, 1.0)), evidence),
            ("the other root by index", args, with_end(forged_root(SQUARE_2, 0)), evidence),
            ("not primitive", args, with_end(forged_root((-4, 0, 2), 1)), evidence),
            ("a negative leading coefficient", args, with_end(forged_root((2, 0, -1), 0)), evidence),
            ("closure flags that are ints", args, with_end(SQRT2, left=0, right=1), evidence),
            ("a Rational subclass", args, replaced(value, 1, forged_interval(SubRational(-1), q(0))), evidence),
        ]
        self.assertRejectsAll(cases)

    def test_rejects_inexact_numbers_and_lists(self):
        args, value, evidence = base_claim()
        samples = evidence[0]
        cases = []
        for label, item in (("True", True), ("0.5", 0.5), ('"1"', "1"), ("None", None), ("an int", -1),
                            ("a Fraction", F(-1, 2)), ("a float Rational", trusted_rational(-0.5, 1))):
            changed = samples[:3] + (item,) + samples[4:]
            cases.append((f"a sample that is {label}", args, value, (changed,)))
        for label, item in (("True", True), ("False", False), ("0.0", 0.0), ('"0"', "0"), ("the int 0", 0),
                            ("a Fraction", F(0)), ("a float Rational", trusted_rational(0.0, 1))):
            cases.append((f"an endpoint that is {label}", args, replaced(value, 1, forged_interval(q(-1), item)),
                          evidence))
        cases += [
            ("a list of samples", args, value, (list(samples),)),
            ("a list of evidence", args, value, [samples]),
            ("a list of intervals", args, support.forged(IntervalSet, intervals=list(value.intervals)), evidence),
            ("a list of coefficients", args, replaced(value, 4, forged_interval(forged_root(list(SQUARE_3), 1), None)),
             evidence),
            ("a bool coefficient", args, replaced(value, 4, forged_interval(forged_root((-3, False, 1), 1), None)),
             evidence),
            ("a float coefficient", args, replaced(value, 4, forged_interval(forged_root((-3.0, 0, 1), 1), None)),
             evidence),
            ("a str coefficient", args, replaced(value, 4, forged_interval(forged_root((-3, "0", 1), 1), None)),
             evidence),
            ("a str index", args, replaced(value, 4, forged_interval(forged_root(SQUARE_3, "1"), None)), evidence),
            ("a str closure flag", args, replaced(value, 1, forged_interval(q(-1), q(0), "False", False)), evidence),
            ("a float closure flag", args, replaced(value, 1, forged_interval(q(-1), q(0), False, 0.0)), evidence),
        ]
        for label, item in (("None", None), ("a str", "[0, 1]"), ("an int", 1), ("a tuple of intervals",
                            BASE_B.intervals), ("a list of intervals", list(BASE_B.intervals)),
                            ("the empty set record", EmptySet(Reals)), ("the whole line record", UniversalSet(Reals))):
            cases.append((f"b as {label}", dict(args, b=item), value, evidence))
        self.assertRejectsAll(cases)

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        args, value, evidence = base_claim()
        samples = evidence[0]
        cases = [
            ("no evidence", ()),
            ("the samples twice", (samples, samples)),
            ("an extra item", (samples, ())),
            ("an extra int", (samples, 0)),
            ("the samples unwrapped", samples),
            ("no samples", ((),)),
            ("None", (None,)),
            ("None as evidence", None),
            ("wrapped twice", ((samples,),)),
            ("an extra sample", (samples + (q(100),),)),
            ("a sample missing", (samples[:-1],)),
            ("the first sample missing", (samples[1:],)),
            ("a tuple subclass", Evidence((samples,))),
            ("a tuple subclass of samples", (Evidence(samples),)),
            ("the samples reversed", (samples[::-1],)),
            ("a sample repeated", (samples[:4] + samples[3:-1],)),
        ]
        self.assertRejectsAll((label, args, value, case) for label, case in cases)

    def test_rejects_forged_evidence_that_is_consistent_but_wrong(self):
        """Strictly increasing canonical Rationals in the right layout that do not certify the cut."""
        args, value, evidence = base_claim()
        samples = list(evidence[0])
        line = line_of(BASE_A, BASE_B)
        two_in_one = list(samples)  # The sample of cell 2 moved into cell 3, beside its own: none left in cell 2.
        two_in_one[2] = Rational(*((line.points[2][1] + F(samples[3].numerator, samples[3].denominator)) / 2)
                                 .as_integer_ratio())
        endpoints_only = Line([], [F(-1), F(0), F(1, 2), F(1)])  # The cut without the roots of x^2 - 2, x^2 - 3.
        a_alone = line_of(BASE_A).samples()  # The cut of a alone.
        cases = [
            ("two samples in one cell", tuple(two_in_one)),
            ("a sample at the cut point 1/2", tuple(samples[:5] + [q(1, 2)] + samples[6:])),
            ("a sample at the cut point -1", tuple(samples[:2] + [q(-1)] + samples[3:])),
            ("samples of the endpoint values alone", endpoints_only.samples()),
            ("samples of the cut of a alone", a_alone),
            ("samples beyond every point", tuple(q(10 + k) for k in range(len(samples)))),
            ("a sample in lowest terms twice over", tuple(samples[:4] + [trusted_rational(-2, 4)] + samples[5:])),
            ("a sample with a negative denominator", tuple(samples[:4] + [trusted_rational(1, -2)] + samples[5:])),
        ]
        self.assertRejectsAll((label, args, value, (case,)) for label, case in cases)

    def test_rejects_wrong_operations_and_arguments(self):
        args, value, evidence = base_claim()
        cases = [(name, args) for name in ("set_symmetric_differences", "symmetric_difference", "", None, 3,
                                           "set_union", "set_difference", "SET_SYMMETRIC_DIFFERENCE",
                                           b"set_symmetric_difference", "v6.set_symmetric_difference",
                                           ("set_symmetric_difference",))]
        for name, case_args in cases:
            with self.subTest(operation=name):
                self.assertIs(support.verdict(family, name, case_args, value, evidence), False)
        self.assertRejectsAll([
            ("one argument", dict(a=BASE_A), value, evidence),
            ("an extra argument", dict(args, c=BASE_B), value, evidence),
            ("argument names in capitals", dict(A=BASE_A, B=BASE_B), value, evidence),
            ("a dict subclass", Arguments(args), value, evidence),
            ("pairs", tuple(args.items()), value, evidence),
            ("a list of arguments", [BASE_A, BASE_B], value, evidence),
        ])

    def test_rejects_claims_on_malformed_inputs(self):
        """Inputs that no IntervalSet constructor builds have no answer: every claim on them is refused."""
        args, value, evidence = base_claim()
        low, high = BASE_B.intervals
        square = (4, 0, -4, 0, 1)  # (x^2 - 2)^2, not square-free.
        cases = {
            "b unsorted": forged_set(high, low),
            "b with overlapping intervals": forged_set(low, Interval(q(-2), SQRT2, True, True)),
            "b with an empty interval": forged_set(low, forged_interval(q(1, 2), q(1, 2)), high),
            "b with a root of a polynomial that is not square-free": forged_set(low, forged_interval(q(0), forged_root(
                square, 1), True, True)),
            "b with a root of a polynomial that is not primitive": forged_set(low, forged_interval(q(0), forged_root(
                (-4, 0, 2), 1), True, True)),
            "b with a linear polynomial that is not primitive": forged_set(
                forged_interval(None, forged_root((-2, 4), 0)), high),
            "b with a giant rational end": forged_set(low, forged_interval(q(0), trusted_rational(GIANT, 1), True,
                                                                           True)),
            "b with an end that is an int": forged_set(forged_interval(None, -1, False, True), high),
            "b with a list of intervals": support.forged(IntervalSet, intervals=[low, high]),
        }
        self.assertRejectsAll((label, dict(args, b=forged), value, evidence) for label, forged in cases.items())

    def test_oversized_forgeries_fail_fast(self):
        args, value, evidence = base_claim()
        samples = evidence[0]
        first, *rest = value.intervals
        cases = [
            ("100,000 samples", value, (tuple(q(k) for k in range(100_000)),)),
            ("a giant sample", value, (samples[:-1] + (trusted_rational(GIANT, 1),),)),
            ("a long sample for a cut that has one sample more", value, (samples + (Rational(2 ** 60_000 + 1, 3),),)),
            ("100,000 intervals", forged_set(*[first] * 100_000), evidence),
            # One interval past the bound 2 * (3 + 2) = 10, of objects that are no records at all: refused unread.
            ("11 intervals", forged_set(*[None] * 11), evidence),
            ("a polynomial of a million coefficients", forged_set(
                forged_interval(None, forged_root((1,) * 1_000_000, 0), False, False), *rest), evidence),
            ("a giant coefficient", forged_set(forged_interval(None, forged_root((-GIANT, 0, 1), 0)), *rest),
             evidence),
            ("a long coefficient within max_integer_bits", forged_set(
                forged_interval(None, forged_root((-(2 ** 60_000), 0, 1), 0)), *rest), evidence),
            ("a giant rational end", forged_set(forged_interval(None, trusted_rational(GIANT, 3)), *rest), evidence),
            ("a long rational end within max_integer_bits", forged_set(
                forged_interval(None, Rational(2 ** 60_000 + 1, 3)), *rest), evidence),
        ]
        for label, claim, case_evidence in cases:
            with self.subTest(case=label):
                start = time.perf_counter()
                self.assertRejects(OPERATION, args, claim, case_evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # A long sample that is a valid one costs no more than its evaluations, which the budget pays for: accepted.
        start = time.perf_counter()
        self.assertAccepts(OPERATION, args, value, (samples[:-1] + (Rational(2 ** 60_000 + 1, 3),),))
        self.assertLess(time.perf_counter() - start, FAST)
        # Inputs belong to the request, not to the claim: 100,000 intervals in one of them are refused within the
        # same time, by the budget at worst.
        many = forged_set(*[point(q(k)) for k in range(100_000)])
        start = time.perf_counter()
        self.assertRejects(OPERATION, dict(args, b=many), value, evidence, may_exhaust=True)
        self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_deterministically(self):
        """A budget one unit short of the check runs out, its exact size passes, and a tiny budget runs out at once.
        The base cut has only real roots, so Cells builds no Sturm chain and no cache can change the charges."""
        args, value, evidence = base_claim()
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation(OPERATION, args, value, evidence), True)
        work = budget.work
        self.assertGreater(work, 20)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, args, value, evidence, limits=ComputationLimits(max_work=work - 1))
        self.assertAccepts(OPERATION, args, value, evidence, limits=ComputationLimits(max_work=work))
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, args, value, evidence, limits=ComputationLimits(max_work=5))

    def test_a_cut_beyond_the_budget_is_never_verified(self):
        """Two sets of 12 intervals whose ends are the real cube roots of 2 + k, 48 distinct cubics with two non-real
        roots each, so that Cells builds the Sturm chain of their product of degree 144: the true claim raises
        ResourceLimitError under the default budget, never True and never False, and a wider budget verifies it."""
        def built(ends: Sequence[object]) -> IntervalSet:
            return IntervalSet(tuple(Interval(ends[2 * i], ends[2 * i + 1], True, True) for i in range(len(ends) // 2)))

        a = built([RealAlgebraicRoot((-(2 + 2 * k), 0, 0, 1), 0) for k in range(24)])
        b = built([RealAlgebraicRoot((-(3 + 2 * k), 0, 0, 1), 0) for k in range(24)])
        args, value, least = true_claim(a, b, least_between)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, args, value, least)
        self.assertAccepts(OPERATION, args, value, least, limits=ComputationLimits(max_work=20_000_000))

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_every_power_is_charged(self):
        """No builtin pow and no ** in the checker: it forms no power at all."""
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])

    def test_checker_imports_no_producer_code(self):
        tree = ast.parse(inspect.getsource(checker))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
        allowed = ("mathforge.certify", "mathforge.trusted", "mathforge.errors", "mathforge.limits", "mathforge.model",
                   "..cells", ".interval_algebra")
        for module in imported:
            with self.subTest(module=module):
                self.assertTrue(module == "__future__" or module.startswith(allowed), module)

    def test_docstring_states_the_theorem_and_the_bound(self):
        text = " ".join(checker.__doc__.split())
        self.assertIn("The theorem that makes True imply the claim", text)
        self.assertIn(f"- {OPERATION}(", text)
        for layout in ("evidence is (samples,)", "IntervalSet(())", "IntervalSet((Interval(None, None),))",
                       "2 * (len(a) + len(b))"):
            self.assertIn(layout, text)

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layout
        survives the format, and a wrong value or wrong evidence fails the mathematics check."""
        args, value, evidence = base_claim()
        self.roundtrip(claimed(args, value, evidence))
        union = iset(Interval(None, SQRT2, False, True), Interval(SQRT3, None))
        for claim_value, claim_evidence in ((union, evidence), (value, ((q(0),),))):
            report = support.statuses(mf.verify(claimed(args, claim_value, claim_evidence)))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
