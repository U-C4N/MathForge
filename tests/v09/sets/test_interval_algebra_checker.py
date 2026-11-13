"""The independent checker of mathforge.areas.sets.interval_algebra (mathforge.certify.sets.interval_algebra), tested
on its own before the family's producer exists.

Every true claim is computed here, never by the producer and never by the IntervalSet methods (v0.7's union and
intersection serve only as a second opinion on the test's reference). The test draws its own picture of the line,
Line: it isolates the real roots of every endpoint polynomial between rationals with a Sturm sequence over
fractions.Fraction and separates them from the rational endpoints and from each other. It writes each set as the
members of the line it holds (the open cells and the points), evaluates the operation member by member, rebuilds the
canonical set from the maximal runs of true members and names each endpoint as an input names it. The samples are
rationals between the isolating intervals, their midpoints or, where a test says so, the rationals of least
denominator (least_between, the cheapest to check), in the layout of the family's docstring: (samples,). They also
certify the cut that the card names, cells.endpoint_cut(a, b, value). The checker answers False to every corrupted
claim, never with an exception, and refuses an oversized forgery within FAST seconds under the default budget.

Run this module by name with python -m unittest tests.v09.sets.test_interval_algebra_checker.
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
from mathforge.areas.sets import interval_algebra as family
from mathforge.certify import cells
from mathforge.certify.sets import interval_algebra as checker
from mathforge.contracts import CheckStatus, EmptySet, UniversalSet
from mathforge.intervals import Interval, IntervalSet
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.mcp.kinds import KINDS, Context
from mathforge.model import Rational, Reals, trusted_rational
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
         "and decimal digits")
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
F = Fraction
OPERATIONS = ("set_union", "set_intersection", "set_difference", "set_complement")

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
    """The classic Sturm sequence p, p', -rem(p, p'), ... of a square-free p: its last member is a nonzero constant."""
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
    """Rational brackets of the distinct real roots of a square-free integer polynomial, in increasing order:
    (q, q) for a rational root q met exactly, else (low, high) with low < root < high, no other root of p in
    [low, high], and p(low), p(high) of opposite signs."""
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
    """The test's name of the number an endpoint record stands for: ("q", value) for a rational, ("root", polynomial,
    index) for a root of a polynomial of degree two or more."""
    if type(end) is Rational:
        return ("q", F(end.numerator, end.denominator))
    coefficients, index = end.integer_coefficients, end.real_index
    if len(coefficients) == 2:
        return ("q", F(-coefficients[0], coefficients[1]))
    return ("root", coefficients, index)


class Line:
    """The points of a cut as the test sees them: the rationals given and every real root of the polynomials given,
    in increasing order and each value once, with rational brackets that separate them (low == high for a rational
    point). number maps every name of a point, ("q", value) or ("root", polynomial, index), to its number from 1."""

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
        """Narrow the brackets of two clashing items, at a rational point inside one when there is one: the root
        is that rational when its polynomial vanishes there."""
        for item, other in ((left, right), (right, left)):
            p = item[2]
            if p is None:  # A rational point, or a bracket already turned into one in this pass.
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
        """One rational below the points, one between any two neighbours and one above: the evidence's samples.
        between(low, high) picks a rational in (low, high) between two neighbouring brackets: their midpoint unless
        another rule is given, such as least_between."""
        if not self.points:
            return (Rational(0),)
        values = [F(floor(self.points[0][0]) - 1)]
        values += [between(high, low) for (_, high), (low, _) in zip(self.points, self.points[1:])]
        values.append(F(ceil(self.points[-1][1]) + 1))
        return tuple(Rational(value.numerator, value.denominator) for value in values)


def least_between(low: Fraction, high: Fraction) -> Fraction:
    """The rational of least denominator in the open interval (low, high), low < high (the Stern-Brocot tree): 0 when
    the interval holds it, else the integer nearest to 0 inside, else, below n + 1 for n = floor(low), n + 1/m with the
    least m such that 1/m < high - n when low = n, and otherwise n + 1/x for the simplest x in (1/(high - n),
    1/(low - n)). The checker charges each evaluation at a sample by its length, so these samples cost it least."""
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
    """The line of the endpoints of some IntervalSets: their rationals and every root of their polynomials."""
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
    """The maximal runs of true members as intervals over point numbers: (lower, upper, left_closed, right_closed),
    None at an infinite end."""
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
    """An endpoint record for the point k: one that an input writes, else the root of an input's polynomial."""
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


REFERENCE: dict[str, Callable[..., bool]] = {
    "set_union": lambda in_a, in_b: in_a or in_b,
    "set_intersection": lambda in_a, in_b: in_a and in_b,
    "set_difference": lambda in_a, in_b: in_a and not in_b,
    "set_complement": lambda in_a: not in_a,
}


def operands(operation: str, a: IntervalSet, b: IntervalSet | None) -> tuple[IntervalSet, ...]:
    return (a,) if operation == "set_complement" else (a, b)


def truth_of(operation: str, line: Line, sets: Sequence[IntervalSet]) -> list[bool]:
    return [REFERENCE[operation](*column) for column in zip(*(members(line, value) for value in sets))]


def true_claim(operation: str, a: IntervalSet, b: IntervalSet | None = None,
               between: Callable[[Fraction, Fraction], Fraction] | None = None) -> tuple[dict, IntervalSet, tuple]:
    """(args, value, evidence) of the true claim, all computed by the test; between picks the samples (Line.samples),
    the midpoints between the brackets by default."""
    sets = operands(operation, a, b)
    line = line_of(*sets)
    samples = line.samples() if between is None else line.samples(between)
    return dict(zip("ab", sets)), assemble(line, sets, truth_of(operation, line, sets)), (samples,)


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
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
    """An IntervalSet record that no constructor checked: the intervals exactly as given."""
    return support.forged(IntervalSet, intervals=tuple(intervals))


def forged_interval(lower: object, upper: object, left: object = False, right: object = False) -> Interval:
    return support.forged(Interval, lower=lower, upper=upper, left_closed=left, right_closed=right)


def forged_root(coefficients: object, index: object) -> RealAlgebraicRoot:
    return support.forged(RealAlgebraicRoot, integer_coefficients=coefficients, real_index=index)


def replaced(value: IntervalSet, position: int, interval: object) -> IntervalSet:
    """The value with the interval at position replaced, unchecked."""
    items = list(value.intervals)
    items[position] = interval
    return forged_set(*items)


EMPTY = IntervalSet(())
WHOLE = IntervalSet((Interval(None, None),))
Evidence = type("Evidence", (tuple,), {})  # A tuple subclass.
Arguments = type("Arguments", (dict,), {})  # A dict subclass, which could answer differently on each lookup.


class SubRational(Rational):
    """A Rational subclass, equal to the right endpoint but no exact Rational."""


# The base claims of the corruption tables: a = [-sqrt 2, 1/2) | {1} | (sqrt 3, inf) and b = (-inf, -1] | [0, sqrt 2],
# whose cut has the points -sqrt 3, -sqrt 2, -1, 0, 1/2, 1, sqrt 2 and sqrt 3 (-sqrt 3 being no endpoint).
BASE_A = iset(Interval(MINUS_SQRT2, q(1, 2), True, False), point(q(1)), Interval(SQRT3, None))
BASE_B = iset(Interval(None, q(-1), False, True), Interval(q(0), SQRT2, True, True))
BASE_VALUES = {
    "set_union": iset(Interval(None, SQRT2, False, True), Interval(SQRT3, None)),
    "set_intersection": iset(Interval(MINUS_SQRT2, q(-1), True, True), Interval(q(0), q(1, 2), True, False),
                             point(q(1))),
    "set_difference": iset(Interval(q(-1), q(0)), Interval(SQRT3, None)),
    "set_complement": iset(Interval(None, MINUS_SQRT2), Interval(q(1, 2), q(1), True, False),
                           Interval(q(1), SQRT3, False, True)),
}
BASE_SAMPLES = (q(-2), q(-3, 2), q(-6, 5), q(-1, 2), q(1, 4), q(3, 4), q(6, 5), q(3, 2), q(2))  # By hand.
COMPLEMENT_SAMPLES = (q(-2), q(-3, 2), q(0), q(3, 4), q(6, 5), q(3, 2), q(2))  # Without -1 and 0, endpoints of b.


def base_claim(operation: str) -> tuple[dict, IntervalSet, tuple]:
    return true_claim(operation, BASE_A, BASE_B)


class IntervalAlgebraCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, cases: Iterable[tuple[str, str, object, object, object]]) -> None:
        """The checker answers False to each (label, operation, args, value, evidence) case."""
        for label, operation, args, value, evidence in cases:
            with self.subTest(case=label, operation=operation):
                self.assertRejects(operation, args, value, evidence)

    def assertTrueClaim(self, operation: str, a: IntervalSet, b: IntervalSet | None = None,
                        expected: IntervalSet | None = None) -> tuple[dict, IntervalSet, tuple]:
        """The test's true claim is accepted, and equals the expected value when one is given."""
        args, value, evidence = true_claim(operation, a, b)
        if expected is not None:
            self.assertEqual(value, expected)
        self.assertAccepts(operation, args, value, evidence)
        return args, value, evidence

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        a, b = Argument("a", "IntervalSet", None, "interval_set"), Argument("b", "IntervalSet", None, "interval_set")
        expected = (
            OperationSpec("set_union", "sets", (a, b), "IntervalSet", "Unite two finite unions of real intervals.",
                          SCOPE, version=6),
            OperationSpec("set_intersection", "sets", (a, b), "IntervalSet",
                          "Intersect two finite unions of real intervals.", SCOPE, version=6),
            OperationSpec("set_difference", "sets", (a, b), "IntervalSet", "Remove the second set from the first.",
                          SCOPE, version=6),
            OperationSpec("set_complement", "sets", (a,), "IntervalSet",
                          "Complement a finite union of real intervals in the reals.", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {})  # v0.7's IntervalSet records: no record class of its own.
        self.assertFalse(hasattr(family, "RENDER"))
        self.assertEqual(family.CHECKER, "mathforge.certify.sets.interval_algebra")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_accepts_the_known_values_of_the_card(self):
        cases = (
            ("set_union", iset(Interval(0, 2, True, True)), iset(Interval(1, 3, True, False)),
             iset(Interval(0, 3, True, False))),
            ("set_intersection", iset(Interval(0, 2, True, True)), iset(Interval(1, 3, True, False)),
             iset(Interval(1, 2, True, True))),
            ("set_complement", iset(Interval(0, 1, True, False)), None,
             iset(Interval(None, 0), Interval(1, None, True, False))),
            ("set_difference", WHOLE, iset(point(0)), iset(Interval(None, 0), Interval(0, None))),
            ("set_difference", iset(Interval(MINUS_SQRT2, SQRT2, True, True)), iset(Interval(0, 1)),
             iset(Interval(MINUS_SQRT2, 0, True, True), Interval(1, SQRT2, True, True))),
        )
        for operation, a, b, expected in cases:
            with self.subTest(operation=operation, a=str(a), b=str(b)):
                self.assertTrueClaim(operation, a, b, expected)
        # The evidence of the last one by hand: the points -sqrt 2, 0, 1 and sqrt 2.
        self.assertAccepts("set_difference", dict(a=cases[-1][1], b=cases[-1][2]), cases[-1][3],
                           ((q(-2), q(-1), q(1, 2), q(5, 4), q(2)),))

    def test_accepts_the_special_cases(self):
        unit = iset(Interval(0, 1, True, True))
        cases = (
            ("set_union", EMPTY, EMPTY, EMPTY),
            ("set_intersection", EMPTY, unit, EMPTY),
            ("set_difference", unit, EMPTY, unit),
            ("set_difference", EMPTY, unit, EMPTY),
            ("set_complement", EMPTY, None, WHOLE),
            ("set_complement", WHOLE, None, EMPTY),
            ("set_union", WHOLE, unit, WHOLE),
            ("set_intersection", WHOLE, iset(Interval(0, SQRT2)), iset(Interval(0, SQRT2))),
            ("set_difference", unit, unit, EMPTY),
            ("set_difference", unit, WHOLE, EMPTY),
            ("set_difference", WHOLE, iset(Interval(0, 1, False, True)),
             iset(Interval(None, 0, False, True), Interval(1, None))),
            ("set_union", iset(Interval(0, 1, True, False)), iset(Interval(1, 2, True, True)),
             iset(Interval(0, 2, True, True))),
            ("set_union", iset(Interval(0, 1)), iset(Interval(1, 2)), iset(Interval(0, 1), Interval(1, 2))),
            ("set_union", iset(Interval(0, 1)), iset(point(1)), iset(Interval(0, 1, False, True))),
            ("set_union", iset(point(1)), iset(point(2)), iset(point(1), point(2))),
            ("set_intersection", iset(point(1)), iset(Interval(1, 2, True, True)), iset(point(1))),
            ("set_difference", iset(Interval(0, 2, True, True)), iset(point(1)),
             iset(Interval(0, 1, True, False), Interval(1, 2, False, True))),
            ("set_complement", iset(point(0)), None, iset(Interval(None, 0), Interval(0, None))),
            ("set_intersection", iset(Interval(None, 1, False, True)), iset(Interval(1, None, True, False)),
             iset(point(1))),
            ("set_complement", iset(Interval(None, 0, False, True), Interval(1, None, True, False)), None,
             iset(Interval(0, 1))),
            ("set_union", iset(Interval(None, 0)), iset(Interval(0, None, True, False)), WHOLE),
            ("set_intersection", iset(Interval(0, 1, True, True)), iset(Interval(2, 3, True, True)), EMPTY),
            ("set_intersection", iset(Interval(MINUS_SQRT2, SQRT2, True, True)), iset(Interval(SQRT2, 3, True, True)),
             iset(point(SQRT2))),
            ("set_union", iset(Interval(MINUS_SQRT2, SQRT2)), iset(Interval(SQRT2, SQRT3)),
             iset(Interval(MINUS_SQRT2, SQRT2), Interval(SQRT2, SQRT3))),
            ("set_complement", iset(point(SQRT2), point(SQRT3)), None,
             iset(Interval(None, SQRT2), Interval(SQRT2, SQRT3), Interval(SQRT3, None))),
        )
        for operation, a, b, expected in cases:
            with self.subTest(operation=operation, a=str(a), b=str(b)):
                self.assertTrueClaim(operation, a, b, expected)
        self.assertAccepts("set_union", dict(a=EMPTY, b=EMPTY), EMPTY, ((q(5, 7),),))  # No point: any one sample.
        self.assertAccepts("set_complement", dict(a=EMPTY), WHOLE, ((q(-10**30),),))

    def assertCardCut(self, args: dict, value: IntervalSet, evidence: tuple) -> None:
        """The samples certify the cut that the card names, endpoint_cut(a, b, value): the claim's endpoints, named as
        the inputs name them, add no point to the inputs' cut on which the checker builds its Cells."""
        master, cuts = cells.endpoint_cut(*args.values(), value)
        self.assertTrue(cells.check_samples(master, evidence[0], cuts))

    def test_accepts_the_base_claims(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                args, value, evidence = base_claim(operation)
                self.assertEqual(value, BASE_VALUES[operation])
                self.assertAccepts(operation, args, value, evidence)
                self.assertCardCut(args, value, evidence)
                samples = COMPLEMENT_SAMPLES if operation == "set_complement" else BASE_SAMPLES
                self.assertAccepts(operation, args, value, (samples,))
                self.assertCardCut(args, value, (samples,))

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
            for operation in OPERATIONS:
                with self.subTest(operation=operation, a=str(a), b=str(b)):
                    self.assertTrueClaim(operation, a, b)

    def test_accepts_every_name_the_inputs_give(self):
        """An endpoint may be written as an input writes it, a rational one also as a Rational or as the root of a
        linear polynomial whenever an input has it as such, and a root of an input's polynomial by its index."""
        # sqrt 2 under two polynomials, x^2 - 2 in a and x^3 - 2x in b: the points -sqrt 2, 0, sqrt 2 and 3.
        other = RealAlgebraicRoot((0, -2, 0, 1), 2)
        a, b = iset(Interval(0, SQRT2, True, True)), iset(Interval(other, 3, True, True))
        samples = ((q(-2), q(-1), q(1), q(2), q(4)),)
        for value in (iset(Interval(other, SQRT2, True, True)), iset(Interval(SQRT2, other, True, True)),
                      iset(point(SQRT2)), iset(point(other))):
            with self.subTest(value=str(value)):
                self.assertAccepts("set_intersection", dict(a=a, b=b), value, samples)
        self.assertAccepts("set_union", dict(a=a, b=b), iset(Interval(0, 3, True, True)), samples)
        self.assertAccepts("set_difference", dict(a=a, b=b), iset(Interval(0, SQRT2, True, False)), samples)
        self.assertAccepts("set_difference", dict(a=a, b=b), iset(Interval(0, other, True, False)), samples)
        self.assertAccepts("set_complement", dict(a=b), iset(Interval(None, other), Interval(3, None)), samples)
        # 1/2 as the root of 2x - 1 in the input: the claim may write it either way, and so when the input has 1/2.
        linear = RealAlgebraicRoot((-1, 2), 0)
        for written in (linear, q(1, 2)):
            for named in (linear, q(1, 2)):
                with self.subTest(written=str(written), named=str(named)):
                    args, value, evidence = true_claim("set_complement", iset(Interval(written, 1, True, True)))
                    self.assertAccepts("set_complement", args,
                                       iset(Interval(None, named), Interval(q(1), None)), evidence)
        # Another root of an input's polynomial: -sqrt 2 is an endpoint of a as the root number 0 of x^3 - 2x, and the
        # union may write it as the root number 0 of x^2 - 2, b's polynomial, an index that no input uses. The points
        # are -sqrt 2, 0, 1 and sqrt 2.
        a = iset(Interval(RealAlgebraicRoot((0, -2, 0, 1), 0), 1, True, True))
        b = iset(Interval(0, SQRT2, True, True))
        samples = ((q(-2), q(-1), q(1, 2), q(6, 5), q(2)),)
        for name in (MINUS_SQRT2, RealAlgebraicRoot((0, -2, 0, 1), 0)):
            with self.subTest(name=str(name)):
                self.assertAccepts("set_union", dict(a=a, b=b), iset(Interval(name, SQRT2, True, True)), samples)
        self.assertAccepts("set_intersection", dict(a=a, b=b), iset(Interval(0, 1, True, True)), samples)
        # 1 as the root number 1 of (x - 1)(x^2 - 2), an input's polynomial of degree three.
        reducible = RealAlgebraicRoot((2, -2, -1, 1), 1)
        args, value, evidence = true_claim("set_complement", iset(Interval(reducible, 2, True, True)))
        self.assertEqual(value, iset(Interval(None, reducible), Interval(q(2), None)))
        self.assertAccepts("set_complement", args, value, evidence)

    def test_accepts_samples_the_producer_would_not_choose(self):
        for operation in OPERATIONS:
            args, value, evidence = base_claim(operation)
            samples = list(evidence[0])
            line = line_of(*args.values())
            far = [q(-10**40)] + samples[1:-1] + [Rational(10**40, 3)]
            tiny = [sample + Rational(1, 2**300) for sample in samples]  # Inside the same cells: the gaps are wider.
            hand = list(COMPLEMENT_SAMPLES if operation == "set_complement" else BASE_SAMPLES)
            lower = [Rational(*(low - F(1, 10**6)).as_integer_ratio()) for low, _ in line.points] + [q(10)]
            least = list(line.samples(least_between))
            for label, chosen in (("far", far), ("tiny offsets", tiny), ("by hand", hand), ("next to points", lower),
                                  ("least denominators", least)):
                with self.subTest(operation=operation, samples=label):
                    self.assertAccepts(operation, args, value, (tuple(chosen),))

    def test_reference_least_between(self):
        """The test's least_between against a search over the denominators, on random intervals of both signs."""
        rng = random.Random(1112)
        for _ in range(300):
            low = F(rng.randint(-400, 400), rng.randint(1, 60))
            high = low + F(rng.randint(1, 50), rng.randint(1, 400))
            found = least_between(low, high)
            denominator = 1
            while not any(low < F(numerator, denominator) < high
                          for numerator in range(floor(low * denominator), ceil(high * denominator) + 1)):
                denominator += 1
            with self.subTest(low=low, high=high):
                self.assertTrue(low < found < high)
                self.assertEqual(found.denominator, denominator)

    def test_accepts_random_claims_and_rejects_their_mutations(self):
        """Random sets over a pool of rationals and of roots of six polynomials: every true claim is accepted, v0.7's
        union and intersection agree with the test's reference, and the set of every other truth assignment that
        differs in one member is refused."""
        polynomials = (SQUARE_2, SQUARE_3, GOLDEN, HALF_SQUARE, CUBE_2, TRIPLE)
        rationals = (q(-3), q(-5, 2), q(-1), q(-1, 3), q(0), q(1, 2), q(1), q(7, 5), q(2), q(3))
        pool = Line(polynomials, [F(r.numerator, r.denominator) for r in rationals])
        names = list(rationals) + [RealAlgebraicRoot(p, index) for p in polynomials
                                   for index in range(len(isolate(p)))]
        names.sort(key=lambda end: pool.number[key_of(end)])
        rng = random.Random(20261112)

        def random_set() -> IntervalSet:
            chosen = sorted(rng.sample(range(len(names)), rng.randint(0, 5)))
            truth = [rng.random() < 0.5 for _ in range(2 * len(chosen) + 1)]
            return IntervalSet(tuple(
                Interval(None if lower is None else names[chosen[lower - 1]],
                         None if upper is None else names[chosen[upper - 1]], left, right)
                for lower, upper, left, right in runs(truth)))

        accepted = refused = 0
        for _ in range(120):
            a, b = random_set(), random_set()
            operation = rng.choice(OPERATIONS)
            args, value, evidence = true_claim(operation, a, b)
            with self.subTest(operation=operation, a=str(a), b=str(b)):
                self.assertAccepts(operation, args, value, evidence)
                self.assertCardCut(args, value, evidence)
                accepted += 1
                if operation == "set_union":
                    self.assertEqual(value, a.union(b))
                elif operation == "set_intersection":
                    self.assertEqual(value, a.intersection(b))
                sets = operands(operation, a, b)
                line = line_of(*sets)
                truth = truth_of(operation, line, sets)
                for member in rng.sample(range(len(truth)), min(2, len(truth))):
                    flipped = list(truth)
                    flipped[member] = not flipped[member]
                    self.assertRejects(operation, args, assemble(line, sets, flipped), evidence)
                    refused += 1
        self.assertEqual((accepted, refused >= accepted), (120, True))

    def assertDecodable(self, *sets: IntervalSet) -> None:
        """Each set is an interval_set argument that the MCP tools accept: their decoder, which bounds the cost of
        rebuilding a set so that a Result holding it reloads, returns it from its JSON form."""
        kind = KINDS["interval_set"]
        for value in sets:
            self.assertEqual(kind.decode(kind.encode(value), Context(ComputationLimits())), value)

    def test_largest_friendly_inputs_reload_and_verify(self):
        """Two sets of 32 intervals, the most that the MCP tools decode in friendly form: with rational endpoints for
        every operation, and with 128 distinct endpoints 2k + sqrt 2 for the union and the complement, the Result of
        the claim survives the JSON round trip, and mf.verify accepts it under the default limits before and after.
        So does the costliest union of quadratic ends in friendly form that the decoder was found to accept, 128
        distinct ends sqrt(10^6 + k), with samples of least denominator (about 1,010,000 units of work; midpoints
        between the brackets need 1,970,000, the edge of the budget)."""
        def built(ends: Sequence[object], flags: Sequence[bool]) -> IntervalSet:
            return IntervalSet(tuple(Interval(ends[2 * i], ends[2 * i + 1], flags[2 * i], flags[2 * i + 1])
                                     for i in range(len(ends) // 2)))

        a = built([q(k, 100) for k in range(-6400, 6400, 200)], [True, False] * 32)
        b = built([q(k, 100) for k in range(-6300, 6500, 200)], [False, True] * 32)
        self.assertEqual((len(a), len(b)), (32, 32))
        self.assertDecodable(a, b)
        for operation in OPERATIONS:
            with self.subTest(operation=operation, ends="rational"):
                self.roundtrip(claimed(operation, *true_claim(operation, a, b)))

        def shifted(k: int) -> RealAlgebraicRoot:  # k + sqrt 2, a root of (x - k)^2 - 2.
            return RealAlgebraicRoot((k * k - 2, -2 * k, 1), 1)

        a = built([shifted(4 * k) for k in range(64)], [True, False] * 32)
        b = built([shifted(4 * k + 2) for k in range(64)], [False, True] * 32)
        self.assertDecodable(a, b)
        # The binary operations share the cut of a and b, and so their work: the union stands for all three.
        for operation in ("set_union", "set_complement"):
            with self.subTest(operation=operation, ends="2k + sqrt 2"):
                args, value, evidence = true_claim(operation, a, b)
                # Both roots of each of the 128 quadratics are points, or of the 64 of a alone.
                self.assertEqual(len(evidence[0]), 129 if operation == "set_complement" else 257)
                start = time.perf_counter()
                self.roundtrip(claimed(operation, args, value, evidence))
                self.assertLess(time.perf_counter() - start, 5 * FAST)

        def root(c: int) -> RealAlgebraicRoot:  # sqrt c.
            return RealAlgebraicRoot((-c, 0, 1), 1)

        a = built([root(10**6 + 2 * k) for k in range(64)], [True] * 64)
        b = built([root(10**6 + 2 * k + 1) for k in range(64)], [True] * 64)
        self.assertDecodable(a, b)
        start = time.perf_counter()
        self.roundtrip(claimed("set_union", *true_claim("set_union", a, b, least_between)))
        self.assertLess(time.perf_counter() - start, 5 * FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims, which the checker refuses with False

    def test_rejects_wrong_values(self):
        cases = []
        for operation in OPERATIONS:
            args, value, evidence = base_claim(operation)
            sets = tuple(args.values())
            line = line_of(*sets)
            truth = truth_of(operation, line, sets)
            for member in range(len(truth)):  # Every set that differs from the true one in one member.
                flipped = list(truth)
                flipped[member] = not flipped[member]
                cases.append((f"member {member} flipped", operation, args, assemble(line, sets, flipped), evidence))
            for other in OPERATIONS:
                if BASE_VALUES[other] != value:
                    cases.append((f"the value of {other}", operation, args, BASE_VALUES[other], evidence))
            cases += [("the empty set", operation, args, EMPTY, evidence),
                      ("the whole line", operation, args, WHOLE, evidence),
                      ("an interval dropped", operation, args, iset(*value.intervals[1:]), evidence),
                      ("the input itself", operation, args, args["a"], evidence)]
        args, value, evidence = base_claim("set_difference")
        cases += [
            ("sqrt 3 replaced by -sqrt 3", "set_difference", args,
             iset(Interval(q(-1), q(0)), Interval(MINUS_SQRT3, None)), evidence),
            ("a rational end moved to another cut point", "set_difference", args,
             iset(Interval(q(-1), q(1, 2)), Interval(SQRT3, None)), evidence),
            ("a closure flipped", "set_difference", args,
             iset(Interval(q(-1), q(0), True, False), Interval(SQRT3, None)), evidence),
            ("a point added", "set_difference", args,
             iset(Interval(q(-1), q(0)), point(q(1)), Interval(SQRT3, None)), evidence),
        ]
        self.assertRejectsAll(cases)

    def test_rejects_noncanonical_values_that_are_mathematically_right(self):
        args, union, evidence = base_claim("set_union")  # (-inf, sqrt 2] | (sqrt 3, inf).
        low, high = union.intervals
        written = {
            "split at an inner point, open then closed": (Interval(None, q(0)), Interval(q(0), SQRT2, True, True),
                                                          high),
            "split at an inner point, closed twice": (Interval(None, q(0), False, True),
                                                      Interval(q(0), SQRT2, True, True), high),
            "split at an inner point, closed then open": (Interval(None, q(0), False, True),
                                                          Interval(q(0), SQRT2, False, True), high),
            "overlapping": (Interval(None, q(1)), Interval(q(0), SQRT2, True, True), high),
            "nested": (low, point(q(1)), high),
            "unsorted": (high, low),
            "twice the same interval": (low, low, high),
            "an empty open interval": (low, forged_interval(SQRT2, SQRT2), high),
            "an empty half-open interval": (low, forged_interval(SQRT3, SQRT3, True, False), high),
            "a reversed interval": (low, forged_interval(SQRT3, q(1, 2)), high),
            "minus infinity closed": (forged_interval(None, SQRT2, True, True), high),
            "plus infinity closed": (low, forged_interval(SQRT3, None, False, True)),
        }
        cases = [(label, "set_union", args, forged_set(*items), evidence) for label, items in written.items()]
        # Endpoints under names that no input gives: the checker's condition 2.
        foreign = {
            "sqrt 2 as a root of (x^2 - 2)(x^2 + 1)": RealAlgebraicRoot((-2, 0, -1, 0, 1), 1),
            "sqrt 2 as a root of (x^2 - 2)(x^2 - 3)": RealAlgebraicRoot((6, 0, -5, 0, 1), 2),
            "sqrt 2 as a root of x^3 - 2x": RealAlgebraicRoot((0, -2, 0, 1), 2),
        }
        for label, name in foreign.items():
            cases.append((label, "set_union", args, iset(Interval(None, name, False, True), high), evidence))
        args, intersection, evidence = base_claim("set_intersection")
        cases += [
            ("1/2 as a root of (2x - 1)(x^2 - 2)", "set_intersection", args,
             replaced(intersection, 1, Interval(q(0), RealAlgebraicRoot((2, -4, -1, 2), 1), True, False)), evidence),
            ("1 as a root of x^2 - 1", "set_intersection", args,
             replaced(intersection, 2, point(RealAlgebraicRoot((-1, 0, 1), 1))), evidence),
        ]
        # A Rational for the root of an input's polynomial of degree three that is rational.
        reducible = RealAlgebraicRoot((2, -2, -1, 1), 1)
        args, value, evidence = true_claim("set_complement", iset(Interval(reducible, 2, True, True)))
        cases.append(("1 as a Rational where the input has a root of (x - 1)(x^2 - 2)", "set_complement", args,
                      iset(Interval(None, q(1)), Interval(q(2), None)), evidence))
        # The empty set and the whole line as the solution-set records of v0.7's solve.
        for operation, a, b, record in (("set_difference", BASE_A, BASE_A, EmptySet(Reals)),
                                        ("set_intersection", BASE_A, EMPTY, EmptySet(Reals)),
                                        ("set_complement", EMPTY, None, UniversalSet(Reals)),
                                        ("set_union", BASE_B, WHOLE, UniversalSet(Reals))):
            args, value, evidence = true_claim(operation, a, b)
            self.assertAccepts(operation, args, value, evidence)
            cases.append((f"{type(record).__name__} in place of {value}", operation, args, record, evidence))
        self.assertRejectsAll(cases)

    def test_rejects_impostors_and_forged_records(self):
        cases = []
        for operation in OPERATIONS:
            args, value, evidence = base_claim(operation)
            cases.append(("an impostor IntervalSet", operation, args, support.impostor(value), evidence))
            cases.append(("an impostor Interval", operation, args,
                          replaced(value, 0, support.impostor(value.intervals[0])), evidence))
            impostors = {name: support.impostor(record) for name, record in args.items()}
            cases.append(("impostor inputs", operation, impostors, value, evidence))
        args, union, evidence = base_claim("set_union")
        high = union.intervals[1]
        difference_args, _, difference_evidence = base_claim("set_difference")
        cases += [
            ("an impostor RealAlgebraicRoot", "set_union", args,
             forged_set(forged_interval(None, support.impostor(SQRT2), False, True), high), evidence),
            ("an IntervalSet without intervals", "set_union", args, support.forged(IntervalSet), evidence),
            ("an Interval without its upper end", "set_union", args,
             forged_set(support.forged(Interval, lower=None, left_closed=False, right_closed=True), high), evidence),
            ("a RealAlgebraicRoot without fields", "set_union", args,
             forged_set(forged_interval(None, support.forged(RealAlgebraicRoot), False, True), high), evidence),
            ("an index past the degree", "set_union", args,
             forged_set(forged_interval(None, forged_root(SQUARE_2, 2), False, True), high), evidence),
            ("a negative index", "set_union", args,
             forged_set(forged_interval(None, forged_root(SQUARE_2, -1), False, True), high), evidence),
            ("a bool index", "set_union", args,
             forged_set(forged_interval(None, forged_root(SQUARE_2, True), False, True), high), evidence),
            ("a float index", "set_union", args,
             forged_set(forged_interval(None, forged_root(SQUARE_2, 1.0), False, True), high), evidence),
            ("the other root by index", "set_union", args,
             forged_set(forged_interval(None, forged_root(SQUARE_2, 0), False, True), high), evidence),
            ("not primitive", "set_union", args,
             forged_set(forged_interval(None, forged_root((-4, 0, 2), 1), False, True), high), evidence),
            ("a negative leading coefficient", "set_union", args,
             forged_set(forged_interval(None, forged_root((2, 0, -1), 0), False, True), high), evidence),
            ("closure flags that are ints", "set_union", args,
             forged_set(forged_interval(None, SQRT2, 0, 1), high), evidence),
            ("a Rational subclass", "set_difference", difference_args,
             forged_set(forged_interval(SubRational(-1), q(0)), Interval(SQRT3, None)), difference_evidence),
        ]
        self.assertRejectsAll(cases)

    def test_rejects_inexact_numbers_and_lists(self):
        args, value, evidence = base_claim("set_difference")  # (-1, 0) | (sqrt 3, inf).
        samples = evidence[0]
        cases = []
        for label, item in (("True", True), ("0.5", 0.5), ('"1"', "1"), ("None", None), ("an int", -1),
                            ("a Fraction", F(-1, 2)), ("a float Rational", trusted_rational(-0.5, 1))):
            changed = samples[:3] + (item,) + samples[4:]
            cases.append((f"a sample that is {label}", "set_difference", args, value, (changed,)))
        for label, item in (("True", True), ("False", False), ("0.0", 0.0), ('"0"', "0"), ("the int 0", 0),
                            ("a Fraction", F(0)), ("a float Rational", trusted_rational(0.0, 1))):
            cases.append((f"an endpoint that is {label}", "set_difference", args,
                          replaced(value, 0, forged_interval(q(-1), item)), evidence))
        cases += [
            ("a list of samples", "set_difference", args, value, (list(samples),)),
            ("a list of evidence", "set_difference", args, value, [samples]),
            ("a list of intervals", "set_difference", args,
             support.forged(IntervalSet, intervals=list(value.intervals)), evidence),
            ("a list of coefficients", "set_difference", args,
             replaced(value, 1, forged_interval(forged_root(list(SQUARE_3), 1), None)), evidence),
            ("a bool coefficient", "set_difference", args,
             replaced(value, 1, forged_interval(forged_root((-3, False, 1), 1), None)), evidence),
            ("a float coefficient", "set_difference", args,
             replaced(value, 1, forged_interval(forged_root((-3.0, 0, 1), 1), None)), evidence),
            ("a str coefficient", "set_difference", args,
             replaced(value, 1, forged_interval(forged_root((-3, "0", 1), 1), None)), evidence),
            ("a str index", "set_difference", args,
             replaced(value, 1, forged_interval(forged_root(SQUARE_3, "1"), None)), evidence),
            ("a str closure flag", "set_difference", args,
             replaced(value, 0, forged_interval(q(-1), q(0), "False", False)), evidence),
            ("a float closure flag", "set_difference", args,
             replaced(value, 0, forged_interval(q(-1), q(0), False, 0.0)), evidence),
        ]
        for label, item in (("None", None), ("a str", "[0, 1]"), ("an int", 1), ("a tuple of intervals",
                            BASE_B.intervals), ("a list of intervals", list(BASE_B.intervals)),
                            ("the empty set record", EmptySet(Reals)), ("the whole line record", UniversalSet(Reals))):
            cases.append((f"b as {label}", "set_difference", dict(args, b=item), value, evidence))
        self.assertRejectsAll(cases)

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        args, value, evidence = base_claim("set_intersection")
        samples = evidence[0]
        cases = [
            ("no evidence", ()),
            ("the samples twice", (samples, samples)),
            ("an extra item", (samples, ())),
            ("an extra int", (samples, 0)),
            ("the samples unwrapped", samples),
            ("no samples", ((),)),
            ("None", (None,)),
            ("wrapped twice", ((samples,),)),
            ("an extra sample", (samples + (q(100),),)),
            ("a sample missing", (samples[:-1],)),
            ("the first sample missing", (samples[1:],)),
            ("a tuple subclass", Evidence((samples,))),
            ("a tuple subclass of samples", (Evidence(samples),)),
            ("the samples reversed", (samples[::-1],)),
            ("a sample repeated", (samples[:4] + samples[3:-1],)),
        ]
        self.assertRejectsAll((label, "set_intersection", args, value, case) for label, case in cases)

    def test_rejects_forged_evidence_that_is_consistent_but_wrong(self):
        """Strictly increasing canonical Rationals in the right layout that do not certify the cut."""
        args, value, evidence = base_claim("set_union")
        samples = list(evidence[0])
        line = line_of(*args.values())
        two_in_one = list(samples)  # The sample of cell 2 moved into cell 3, beside its own: none left in cell 2.
        two_in_one[2] = Rational(*((line.points[2][1] + F(samples[3].numerator, samples[3].denominator)) / 2)
                                 .as_integer_ratio())
        endpoints_only = Line([], [F(-1), F(0), F(1, 2), F(1)])  # The cut without the roots of x^2 - 2, x^2 - 3.
        cases = [
            ("two samples in one cell", tuple(two_in_one)),
            ("a sample at the cut point 1/2", tuple(samples[:5] + [q(1, 2)] + samples[6:])),
            ("a sample at the cut point -1", tuple(samples[:2] + [q(-1)] + samples[3:])),
            ("samples of the endpoint values alone", endpoints_only.samples()),
            ("samples of the complement's cut", base_claim("set_complement")[2][0]),
            ("samples beyond every point", tuple(q(10 + k) for k in range(len(samples)))),
            ("a sample in lowest terms twice over", tuple(samples[:4] + [trusted_rational(-2, 4)] + samples[5:])),
            ("a sample with a negative denominator", tuple(samples[:4] + [trusted_rational(1, -2)] + samples[5:])),
        ]
        self.assertRejectsAll((label, "set_union", args, value, (case,)) for label, case in cases)
        complement_args, complement, complement_evidence = base_claim("set_complement")
        self.assertRejects("set_complement", complement_args, complement, evidence)  # The binary cut's samples.

    def test_rejects_wrong_operations_and_arguments(self):
        args, value, evidence = base_claim("set_union")
        cases = [(f"operation {name!r}", name, args, value, evidence)
                 for name in ("set_unions", "union", "", None, 3, "set_symmetric_difference", "SET_UNION", b"set_union",
                              "v6.set_union", ("set_union",))]
        cases += [
            ("the union checked as an intersection", "set_intersection", args, value, evidence),
            ("the union checked as a complement", "set_complement", dict(a=BASE_A), value, evidence),
            ("two arguments for the complement", "set_complement", args, BASE_VALUES["set_complement"],
             base_claim("set_complement")[2]),
            ("one argument for the union", "set_union", dict(a=BASE_A), value, evidence),
            ("an extra argument", "set_union", dict(args, c=BASE_B), value, evidence),
            ("argument names in capitals", "set_union", dict(A=BASE_A, B=BASE_B), value, evidence),
            ("a dict subclass", "set_union", Arguments(args), value, evidence),
            ("pairs", "set_union", tuple(args.items()), value, evidence),
            ("the arguments of the difference swapped", "set_difference", dict(a=BASE_B, b=BASE_A),
             BASE_VALUES["set_difference"], evidence),
        ]
        self.assertRejectsAll(cases)

    def test_rejects_claims_on_malformed_inputs(self):
        """Inputs that no IntervalSet constructor builds have no answer: every claim on them is refused."""
        args, value, evidence = base_claim("set_union")
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
            "b with a linear polynomial with a negative leading coefficient": forged_set(
                forged_interval(None, forged_root((1, -1), 0)), high),
            "b with a giant rational end": forged_set(low, forged_interval(q(0), trusted_rational(GIANT, 1), True,
                                                                           True)),
            "b with an end that is an int": forged_set(forged_interval(None, -1, False, True), high),
            "b with a list of intervals": support.forged(IntervalSet, intervals=[low, high]),
        }
        self.assertRejectsAll((label, "set_union", dict(args, b=forged), value, evidence)
                              for label, forged in cases.items())

    def test_oversized_forgeries_fail_fast(self):
        args, value, evidence = base_claim("set_union")
        samples = evidence[0]
        low, high = value.intervals
        cases = [
            ("100,000 samples", value, (tuple(q(k) for k in range(100_000)),)),
            ("a giant sample", value, (samples[:-1] + (trusted_rational(GIANT, 1),),)),
            ("a long sample for a cut that has one sample more", value,
             (samples + (Rational(2 ** 60_000 + 1, 3),),)),
            ("100,000 intervals", forged_set(*[low] * 100_000), evidence),
            ("a polynomial of a million coefficients", forged_set(
                forged_interval(None, forged_root((1,) * 1_000_000, 0), False, True), high), evidence),
            ("a giant coefficient", forged_set(forged_interval(None, forged_root((-GIANT, 0, 1), 1), False, True),
                                               high), evidence),
            ("a long coefficient within max_integer_bits", forged_set(
                forged_interval(None, forged_root((-(2 ** 60_000), 0, 1), 1), False, True), high), evidence),
            ("a giant rational end", forged_set(forged_interval(None, trusted_rational(GIANT, 3), False, True), high),
             evidence),
            ("a long rational end within max_integer_bits", forged_set(
                forged_interval(None, Rational(2 ** 60_000 + 1, 3), False, True), high), evidence),
        ]
        for label, claim, case_evidence in cases:
            with self.subTest(case=label):
                start = time.perf_counter()
                self.assertRejects("set_union", args, claim, case_evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # A long sample that is a valid one costs no more than its evaluations, which the budget pays for: accepted.
        start = time.perf_counter()
        self.assertAccepts("set_union", args, value, (samples[:-1] + (Rational(2 ** 60_000 + 1, 3),),))
        self.assertLess(time.perf_counter() - start, FAST)
        # Inputs belong to the request, not to the claim: 100,000 intervals in one of them are refused within the
        # same time, by the budget at worst.
        many = forged_set(*[point(q(k)) for k in range(100_000)])
        start = time.perf_counter()
        self.assertRejects("set_union", dict(args, b=many), value, evidence, may_exhaust=True)
        self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_deterministically(self):
        """The charges depend on the claim alone: a budget one unit short of a check refuses it, its exact size
        passes, and a tiny budget runs out at once. The base cut has only real roots, so Cells builds no Sturm chain
        and no cache can change the charges."""
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                args, value, evidence = base_claim(operation)
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, evidence), True)
                work = budget.work
                self.assertGreater(work, 20)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, value, evidence, limits=ComputationLimits(max_work=work))
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence, limits=ComputationLimits(max_work=5))

    def test_a_cut_beyond_the_budget_is_never_verified(self):
        """Inputs that the MCP tools decode can need more than the default budget whatever the samples: two sets of
        12 intervals whose ends, sent as envelopes, are the real cube roots of 2 + k, 48 distinct cubics with two
        non-real roots each, so that Cells builds the Sturm chain of their product of degree 144 (about 5,700,000
        units for the chain alone). The true claim, with midpoints or with samples of least denominator, raises
        ResourceLimitError under the default budget, never True and never False (the producer refuses such inputs),
        and a wider budget verifies it."""
        def built(ends: Sequence[object]) -> IntervalSet:
            return IntervalSet(tuple(Interval(ends[2 * i], ends[2 * i + 1], True, True) for i in range(len(ends) // 2)))

        a = built([RealAlgebraicRoot((-(2 + 2 * k), 0, 0, 1), 0) for k in range(24)])
        b = built([RealAlgebraicRoot((-(3 + 2 * k), 0, 0, 1), 0) for k in range(24)])
        self.assertDecodable(a, b)
        line = line_of(a, b)  # true_claim by hand, so that both kinds of samples come from one line.
        args, value = dict(a=a, b=b), assemble(line, (a, b), truth_of("set_union", line, (a, b)))
        evidence, least = (line.samples(),), (line.samples(least_between),)
        for samples in (evidence, least):
            with self.assertRaises(ResourceLimitError):
                support.verdict(family, "set_union", args, value, samples)
        self.assertAccepts("set_union", args, value, least, limits=ComputationLimits(max_work=20_000_000))

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
                   "..cells")
        for module in imported:
            with self.subTest(module=module):
                self.assertTrue(module in ("__future__", "math", "collections.abc") or module.startswith(allowed),
                                module)

    def test_docstring_states_the_theorem_of_every_operation(self):
        text = checker.__doc__
        self.assertIn("The theorem that makes True imply the claim", text)
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                self.assertIn(f"- {operation}(", text)
        for layout in ("evidence is (samples,)", "IntervalSet(())", "IntervalSet((Interval(None, None),))"):
            self.assertIn(layout, " ".join(text.split()))

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layout
        survives the format, and a wrong value or wrong evidence fails the mathematics check."""
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                args, value, evidence = base_claim(operation)
                self.roundtrip(claimed(operation, args, value, evidence))
                wrong = BASE_VALUES["set_union" if operation != "set_union" else "set_intersection"]
                for claim_value, claim_evidence in ((wrong, evidence), (value, ((q(0),),))):
                    report = support.statuses(mf.verify(claimed(operation, args, claim_value, claim_evidence)))
                    self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                    self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
