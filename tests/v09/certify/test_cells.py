"""The cell engine of the checkers (mathforge.certify.cells).

Hand-built cuts with rational and irrational roots and with rational cut points; the sets of the five relations of
polynomials against 0, written out by hand; truth values rebuilt as canonical sets; claimed sets located, read and
compared, also against an oracle of the test's own; rational roots split off as cut points, signs read from
sign-preserving integer forms and relations decided by several signs; the cut of set endpoints with the set
operations it serves; and differential runs in which the engine, working from samples alone, accepts the solver's
solution sets and rejects every mutation of them, with and without cut points. Bad samples, bad cut points, forged
records and oversized inputs are rejected, the last fast under the default budget, and ordinary inputs with many
rational points fit it.

The engine imports no producer code. These tests do, to build its inputs and to have a second opinion: real_roots
and rational_between for the samples, mathforge.solve and IntervalSet for the sets, compare_real and sign_at for the
order and the signs of real numbers.
"""

from __future__ import annotations

from collections import Counter
from functools import cmp_to_key
import itertools
from math import comb
import random
import time
import unittest
from unittest import mock

import mathforge as mf
from mathforge import Eq, Inequality, Interval, IntervalSet, Polynomial, RealAlgebraicRoot
from mathforge.algebraic import as_endpoint, compare_real, rational_between, sign_at
from mathforge.certify import cells
from mathforge.certify import upoly as certify
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational, trusted_rational
from mathforge.root_isolation import count_real_roots, real_roots
from mathforge.trusted import upoly as tp

X = mf.Symbol("x", uid="00000000-0000-4000-8000-0000000009c1")
SOLVER_RELATIONS = {"=": None, "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}  # Eq, or the relation of an Inequality.
WHOLE_LINE = ((None, None, False, False),)
SECONDS = 1.0  # The time of a refusal that comes at once.
EXHAUSTED = 5.0  # The time of a refusal that comes when the default budget runs out, with a margin for a busy machine.
PRODUCER = ComputationLimits(max_work=10**12, max_nodes=10**9, max_refinements=10**9)  # The producer's unmeasured side.
T, F = True, False

# (x^2 - 2)(x - 1)(3x + 7): the roots -7/3, -sqrt 2, 1 and sqrt 2, with one sample in each of their five cells.
MASTER = (14, -8, -13, 4, 3)
SAMPLES = (Rational(-3), Rational(-2), Rational(0), Rational(5, 4), Rational(2))


# ---------------------------------------------------------------------------------------------------------------
# The producer's side: polynomials, roots, samples and sets

def integers(expression):
    """The integer polynomial of a polynomial expression in X: its coefficients times their common denominator,
    which keeps every sign."""
    return tp.from_rationals(Polynomial.from_expression(expression, X).coefficients)[0]


def roots_of(master):
    """The distinct real roots of an integer polynomial in increasing order, as the producer names them: Rationals
    and RealAlgebraicRoots; none for a constant."""
    if len(master) < 2:
        return ()
    polynomial = Polynomial(tuple(Rational(value) for value in master), X)
    return tuple(as_endpoint(record.root) for record in real_roots(polynomial))


def rational_roots(polynomial):
    """The rational roots of an integer polynomial in increasing order: the candidates a producer sends."""
    return tuple(point for point in roots_of(polynomial) if type(point) is Rational)


def samples_between(points):
    """One rational below the points, one between any two neighbours and one above: the producer's evidence."""
    bounds = (None, *points, None)
    return tuple(rational_between(lower, upper) for lower, upper in zip(bounds, bounds[1:]))


def grid_samples(master, cuts, grid):
    """The producer's samples by a walk over an increasing tuple of rationals that holds the cut points, for a master
    polynomial without a rational root whose real roots lie between neighbouring grid points at which its signs
    differ, one root at most between two of them (as for Chebyshev's T_60 on a grid of step 1/1000: its nearest roots
    are about 0.0027 apart). It spares the isolation of every root, which real_roots refines to rule out a rational
    one; the engine verifies the samples as it verifies any."""
    samples, needed, previous = [], True, None
    for point in grid:
        sign = tp.sign_at(master, point.numerator, point.denominator)
        assert sign, "a grid point is a root"
        if previous is not None and sign != previous:
            assert needed is False, "no grid point between a root and the point before it"
            needed = True  # A root lies between the grid point before and this one.
        previous = sign
        if point in cuts:
            assert needed is False, "no grid point between a cut point and the point before it"
            needed = True
        elif needed:
            samples.append(point)
            needed = False
    assert not needed, "no grid point above the last point"
    return tuple(samples)


def cut(master):
    """(cells, roots) of an integer polynomial, the samples being the producer's."""
    points = roots_of(master)
    return cells.Cells(master, samples_between(points)), points


def cut_points(master, cuts=()):
    """The points of a cut in increasing order: the real roots of the master polynomial and the cut points, a cut
    point at a root once."""
    points = []
    for point in sorted((*roots_of(master), *cuts), key=cmp_to_key(compare_real)):
        if not points or compare_real(points[-1], point):
            points.append(point)
    return tuple(points)


def cut_with(master, cuts=()):
    """(cells, points) of a master polynomial and cut points, the samples being the producer's."""
    points = cut_points(master, cuts)
    return cells.Cells(master, samples_between(points), cuts), points


def split_cut(*polynomials):
    """(master, cuts) of polynomials with their rational roots split off as cut points, as a checker builds them
    from a producer's candidates."""
    splits = [cells.split_rational_roots(polynomial, rational_roots(polynomial)) for polynomial in polynomials]
    return cells.master_polynomial(*(rest for _, rest in splits)), cells.merge_cuts(*(roots for roots, _ in splits))


def is_point(grid, q):
    """Whether a Rational is a point of the cut: a cut point or a root of the master polynomial."""
    return q in grid.cuts or (len(grid.master) > 1 and not tp.sign_at(grid.master, q.numerator, q.denominator))


def other_names(grid, points):
    """Another name for each point: a cut point as the root of its linear polynomial, any other point as a root of
    the master polynomial, counted among its real roots."""
    roots = roots_of(grid.master)
    names = []
    for point in points:
        if type(point) is Rational and point in grid.cuts:
            names.append(forged_root((-point.numerator, point.denominator), 0))
        else:
            names.append(forged_root(grid.master, number_of(point, roots) - 1))
    return names


def sign(value):
    return (value.numerator > 0) - (value.numerator < 0)


def random_factor(rng):
    """A random polynomial expression of degree 1 to 3 with rational or irrational real roots, or none."""
    shape = rng.randrange(5)
    if shape == 0:
        return X - Rational(rng.randint(-4, 4), rng.randint(1, 3))
    if shape == 1:
        return X**2 - rng.randint(-2, 7)
    if shape == 2:
        return X**2 + rng.randint(-4, 4) * X + rng.randint(-4, 4)
    if shape == 3:
        return X**3 + rng.randint(-3, 3) * X**2 + rng.randint(-3, 3) * X + rng.randint(-3, 3)
    return rng.randint(1, 3) * X - rng.randint(-5, 5)


def random_expression(rng):
    """A random nonzero polynomial expression with rational coefficients: dense, or a rational multiple of a
    product of up to four random factors with multiplicities up to 3."""
    if rng.randrange(4) == 0:
        coefficients = [Rational(rng.randint(-4, 4), rng.randint(1, 3)) for _ in range(rng.randint(1, 5))]
        coefficients.append(Rational(rng.choice((-3, -2, -1, 1, 2, 3)), rng.randint(1, 3)))
        return sum((coefficient * X**power for power, coefficient in enumerate(coefficients)), Rational(0))
    expression = Rational(rng.choice((-2, -1, 1, 3)), rng.randint(1, 2))
    for _ in range(rng.randint(1, 4)):
        expression = expression * random_factor(rng) ** rng.choice((1, 1, 1, 2, 3))
    return expression


def as_interval_set(value):
    """A solution set of the solver as an IntervalSet: a finite set becomes closed intervals of one point."""
    if type(value) is mf.FiniteSet:
        return IntervalSet(tuple(Interval(point, point, True, True) for point in map(as_endpoint, value.values)))
    return IntervalSet.from_solution_set(value)


def solve(left, relation):
    """The solver's set of the x with ``left relation 0``."""
    problem = Eq(left, 0) if relation == "=" else Inequality(left, SOLVER_RELATIONS[relation], 0)
    return as_interval_set(mf.solve(problem, for_=X).unwrap())


def relation_truth(grid, polynomial, relation):
    """(root_truth, cell_truth) of ``polynomial relation 0``, from the engine's signs one by one."""
    return (tuple(cells.holds(relation, value) for value in grid.root_signs(polynomial)),
            tuple(cells.holds(relation, value) for value in grid.sample_signs(polynomial)))


def chebyshev(n):
    """Chebyshev's T_n as an integer polynomial, lowest degree first, which has n distinct real roots in (-1, 1): the
    coefficient of x^(n - 2k) is (-1)^k 2^(n - 2k - 1) n / (n - k) C(n - k, k)."""
    coefficients = [0] * (n + 1)
    for k in range(n // 2 + 1):
        value = n * comb(n - k, k) * 2 ** (n - 2 * k) // (2 * (n - k))
        coefficients[n - 2 * k] = -value if k % 2 else value
    return tuple(coefficients)


def assembled(root_truth, cell_truth, points):
    """The set of truth values as the producer assembles it: an open interval per true cell and a point per true
    root, merged into canonical form by IntervalSet."""
    bounds = (None, *points, None)
    pieces = [Interval(bounds[k], bounds[k + 1]) for k, true in enumerate(cell_truth) if true]
    pieces += [Interval(point, point, True, True) for point, true in zip(points, root_truth) if true]
    return IntervalSet(tuple(pieces))


def complement(value):
    """The complement of a canonical IntervalSet, walked by the test itself."""
    pieces, end, closed = [], None, True
    for item in value.intervals:
        if item.lower is not None:
            pieces.append(Interval(end, item.lower, end is not None and not closed, not item.left_closed))
        end, closed = item.upper, item.right_closed
        if end is None:
            return IntervalSet(tuple(pieces))
    pieces.append(Interval(end, None, end is not None and not closed, False))
    return IntervalSet(tuple(pieces))


def set_check(combine, claim, sets, samples=None):
    """Whether the claim is the set of the x with combine(x in the first set, x in the second, ...), as the checker
    of a set operation decides it: every endpoint a point of the cut, a Rational one as a cut point, samples as
    evidence (the producer's when none are given)."""
    try:
        master, cuts = cells.endpoint_cut(*sets, claim)
        if samples is None:
            samples = samples_between(cut_points(master, cuts))
        grid = cells.Cells(master, samples, cuts)
        memberships = [grid.membership(value) for value in sets]
        return grid.matches(claim, tuple(map(combine, *(roots for roots, _ in memberships))),
                            tuple(map(combine, *(inside for _, inside in memberships))))
    except cells.Rejected:
        return False


def measured(call):
    """(result, work, seconds) of a call in a fresh default budget, the checker's Sturm chains not cached."""
    certify._STURM.clear()
    with computation(fresh=True) as budget:
        start = time.perf_counter()
        result = call()
        elapsed = time.perf_counter() - start
    return result, budget.work, elapsed


# ---------------------------------------------------------------------------------------------------------------
# Index form on the test's side, and records that no constructor validated

def number_of(endpoint, points):
    """The number k of the root that a producer endpoint equals, by compare_real; None for an infinite end."""
    if endpoint is None:
        return None
    found = [k for k, point in enumerate(points, 1) if compare_real(endpoint, point) == 0]
    assert len(found) == 1, endpoint
    return found[0]


def index_form(value, points):
    """A producer IntervalSet as intervals (lower, upper, left_closed, right_closed) over root numbers."""
    return tuple((number_of(item.lower, points), number_of(item.upper, points), item.left_closed, item.right_closed)
                 for item in value.intervals)


def members(intervals, r):
    """The truth of cell 0, root 1, cell 1, ..., root r, cell r for intervals in index form: an interval covers the
    members from its lower end (the root itself when closed, else the cell above it) to its upper end."""
    truth = [False] * (2 * r + 1)
    for lower, upper, left_closed, right_closed in intervals:
        first = 0 if lower is None else 2 * lower - left_closed
        last = 2 * r if upper is None else 2 * upper - 2 + right_closed
        for member in range(first, last + 1):
            truth[member] = True
    return truth


def truth_of(intervals, r):
    """(root_truth, cell_truth) of intervals in index form."""
    truth = members(intervals, r)
    return tuple(truth[1::2]), tuple(truth[0::2])


def proper(interval):
    """Whether an interval in index form is one: infinite ends open, ends in order, a single point closed."""
    lower, upper, left_closed, right_closed = interval
    if (lower is None and left_closed) or (upper is None and right_closed):
        return False
    return lower is None or upper is None or lower < upper or (lower == upper and left_closed and right_closed)


def canonical(intervals):
    """Whether intervals in index form are an IntervalSet's canonical form: proper, and each beginning after the end
    of the one before, or at that root with both open there."""
    if not all(map(proper, intervals)):
        return False
    for (_, end, _, closed), (start, _, opened, _) in zip(intervals, intervals[1:]):
        if end is None or start is None or end > start or (end == start and (closed or opened)):
            return False
    return True


def forged_interval(lower, upper, left_closed=False, right_closed=False):
    record = object.__new__(Interval)
    for name, value in (("lower", lower), ("upper", upper), ("left_closed", left_closed),
                        ("right_closed", right_closed)):
        object.__setattr__(record, name, value)
    return record


def forged_set(intervals):
    record = object.__new__(IntervalSet)
    object.__setattr__(record, "intervals", intervals)
    return record


def forged_root(coefficients, index):
    record = object.__new__(RealAlgebraicRoot)
    record.__dict__.update(integer_coefficients=coefficients, real_index=index)
    return record


def records(intervals, points):
    """The IntervalSet record of intervals in index form with the producer's endpoints, written as given: nothing
    is validated, sorted or merged."""
    def end(k):
        return None if k is None else points[k - 1]
    return forged_set(tuple(forged_interval(end(lower), end(upper), left_closed, right_closed)
                            for lower, upper, left_closed, right_closed in intervals))


def nearby_rational(root, low, high, steps=12):
    """A rational below an irrational root, within (high - low) / 2**steps of it, by bisection of [low, high]."""
    for _ in range(steps):
        middle = (low + high) / 2
        if compare_real(middle, root) < 0:
            low = middle
        else:
            high = middle
    return low


def mutations(true, points, grid, rng):
    """(kind, IntervalSet record) pairs that the engine must reject for the truth values of ``true``, the canonical
    set in index form over the points ``points`` of the cut ``grid``: other sets, and the same set written another
    way."""
    r = len(points)
    samples = grid.samples
    claims = []

    def claim(kind, intervals):
        intervals = tuple(intervals)
        assert intervals != true, kind
        claims.append((kind, records(intervals, points)))

    def replaced(position, interval):
        return true[:position] + (tuple(interval),) + true[position + 1:]

    def with_endpoint(kind, position, side, endpoint):
        items = list(records(true, points).intervals)
        ends = [items[position].lower, items[position].upper]
        ends[side] = endpoint
        items[position] = forged_interval(*ends, *true[position][2:])
        claims.append((kind, forged_set(tuple(items))))

    ends = [(position, side) for position in range(len(true)) for side in (0, 1)]
    finite = [(position, side) for position, side in ends if true[position][side] is not None]
    infinite = [(position, side) for position, side in ends if true[position][side] is None]
    if finite:
        position, side = rng.choice(finite)
        interval = list(true[position])
        neighbours = [k for k in (interval[side] - 1, interval[side] + 1) if 1 <= k <= r]
        if neighbours:
            interval[side] = rng.choice(neighbours)
            claim("an endpoint moved to a neighbouring root", replaced(position, interval))
        position, side = rng.choice(finite)
        interval = list(true[position])
        interval[2 + side] = not interval[2 + side]
        claim("a closure flag flipped", replaced(position, interval))
        interval = list(true[position])
        interval[side], interval[2 + side] = None, False
        claim("a finite end made infinite", replaced(position, interval))
    if infinite:
        position, side = rng.choice(infinite)
        interval = list(true[position])
        interval[2 + side] = True
        claim("an infinite end closed", replaced(position, interval))
        if r:
            interval = list(true[position])
            interval[side] = rng.randint(1, r)
            claim("an infinite end made finite", replaced(position, interval))
    if true:
        position = rng.randrange(len(true))
        claim("an interval dropped", true[:position] + true[position + 1:])
        claim("an interval written twice", true[:position + 1] + true[position:])
        claim("the empty set", ())
    if true != WHOLE_LINE:
        claim("the whole line", WHOLE_LINE)
    if len(true) > 1:
        position = rng.randrange(len(true) - 1)
        claim("two intervals swapped", true[:position] + (true[position + 1], true[position]) + true[position + 2:])
    inner = [(position, k) for position, (lower, upper, _, _) in enumerate(true)
             for k in range((lower or 0) + 1, r + 1 if upper is None else upper)]
    if inner:
        position, k = rng.choice(inner)
        lower, upper, left_closed, right_closed = true[position]
        left, right = rng.choice(((True, False), (False, True), (True, True)))
        claim("an interval split in two at an inner root",
              true[:position] + ((lower, k, left_closed, left), (k, upper, right, right_closed)) + true[position + 1:])
        claim("an inner root removed",
              true[:position] + ((lower, k, left_closed, False), (k, upper, False, right_closed)) + true[position + 1:])
    truth = members(true, r)
    false = [member for member, value in enumerate(truth) if not value]
    if false:
        member = rng.choice(false)
        if member % 2:
            extra = ((member + 1) // 2,) * 2 + (True, True)
        else:
            extra = (member // 2 or None, member // 2 + 1 if member < 2 * r else None, False, False)
        position = sum(1 for lower, _, left_closed, _ in true if lower is None or 2 * lower - left_closed < member)
        claim("an interval added", true[:position] + (extra,) + true[position:])

    algebraic = [(position, side) for position, side in finite
                 if type(points[true[position][side] - 1]) is RealAlgebraicRoot]
    rational = [(position, side) for position, side in finite if type(points[true[position][side] - 1]) is Rational]
    if algebraic:
        position, side = rng.choice(algebraic)
        k = true[position][side]
        root = points[k - 1]
        polynomial = Polynomial(tuple(map(Rational, root.integer_coefficients)), X)
        others = [index for index in (root.real_index - 1, root.real_index + 1)
                  if 0 <= index < count_real_roots(polynomial)]
        if others:
            with_endpoint("an algebraic endpoint replaced by a neighbouring root of its polynomial", position, side,
                          forged_root(root.integer_coefficients, rng.choice(others)))
        with_endpoint("an algebraic endpoint replaced by a nearby rational", position, side,
                      nearby_rational(root, samples[k - 1], samples[k]))
    if rational:
        position, side = rng.choice(rational)
        q = points[true[position][side] - 1]
        shifted = q + Rational(1, 1009)
        if not is_point(grid, shifted):
            with_endpoint("a rational endpoint shifted", position, side, shifted)
        if q.denominator == 1:
            with_endpoint("an int in place of a Rational endpoint", position, side, q.numerator)
    items = records(true, points).intervals
    claims.append(("a list of intervals", forged_set(list(items))))
    if true:
        position = rng.randrange(len(true))
        changed = list(items)
        changed[position] = forged_interval(items[position].lower, items[position].upper,
                                            int(true[position][2]), int(true[position][3]))
        claims.append(("closure flags that are ints", forged_set(tuple(changed))))
    return claims


MUTATION_KINDS = (
    "an endpoint moved to a neighbouring root", "a closure flag flipped", "a finite end made infinite",
    "an infinite end closed", "an infinite end made finite", "an interval dropped", "an interval written twice",
    "the empty set", "the whole line", "two intervals swapped", "an interval split in two at an inner root",
    "an inner root removed", "an interval added",
    "an algebraic endpoint replaced by a neighbouring root of its polynomial",
    "an algebraic endpoint replaced by a nearby rational", "a rational endpoint shifted",
    "an int in place of a Rational endpoint", "a list of intervals", "closure flags that are ints",
)


# ---------------------------------------------------------------------------------------------------------------

class SampleTests(unittest.TestCase):
    """The samples certify the real roots of the master polynomial."""

    def rejected(self, master, samples, label=None):
        self.assertIs(cells.check_samples(master, samples), False, label)
        with self.assertRaises(cells.Rejected, msg=label):
            cells.Cells(master, samples)

    def test_one_sample_in_every_cell_is_accepted(self):
        self.assertEqual(integers((X**2 - 2) * (X - 1) * (3 * X + 7)), MASTER)
        grid = cells.Cells(MASTER, SAMPLES)
        self.assertEqual((grid.master, grid.samples, grid.root_count), (MASTER, SAMPLES, 4))
        self.assertIs(cells.check_samples(MASTER, SAMPLES), True)
        for name, value in (("master", (-1, 1)), ("samples", SAMPLES[:2]), ("root_count", 1), ("other", 0)):
            with self.subTest(attribute=name), self.assertRaises(AttributeError):
                setattr(grid, name, value)
        self.assertEqual((grid.master, grid.samples, grid.root_count), (MASTER, SAMPLES, 4))
        # Any rationals of the same cells do: sqrt 2 lies between 141/100 and 142/100.
        others = (Rational(-10**9), Rational(-3, 2), Rational(-141, 100), Rational(141, 100), Rational(142, 100))
        self.assertIs(cells.check_samples(MASTER, others), True)
        self.assertIs(cells.check_samples(MASTER, samples_between(roots_of(MASTER))), True)  # The producer's.

    def test_the_master_polynomial_is_made_square_free(self):
        master = integers(-2 * (X - 1)**2 * (X + 2)**3)
        samples = (Rational(-3), Rational(0), Rational(2))
        grid = cells.Cells(master, samples)
        self.assertEqual((grid.master, grid.root_count), ((-2, 1, 1), 2))
        # The degree would allow six samples; the two distinct roots allow three.
        self.rejected(master, (Rational(-3), Rational(0), Rational(2), Rational(3)), "four samples")
        self.rejected(master, (Rational(-3), Rational(0), Rational(1, 2), Rational(2), Rational(3), Rational(4)), "six")

    def test_a_polynomial_without_real_roots_has_one_cell(self):
        for master in ((1, 0, 1), (5,), (-3,), integers((X**2 + 1) * (X**2 + X + 1)**2)):
            with self.subTest(master=master):
                self.assertEqual(cells.Cells(master, (Rational(7, 3),)).root_count, 0)
                self.rejected(master, ())
                self.rejected(master, (Rational(0), Rational(1)))
        # One real root among complex ones: two samples, not three.
        master = integers((X**2 + 1) * (X - 1))
        self.assertEqual(cells.Cells(master, (Rational(0), Rational(2))).root_count, 1)
        self.rejected(master, (Rational(0), Rational(2), Rational(3)))
        self.rejected(master, (Rational(0),))

    def test_bad_samples_are_rejected(self):
        a, b, c, d, e = SAMPLES
        cases = {
            "too few, the last missing": (a, b, c, d),
            "too few, the first missing": (b, c, d, e),
            "none": (),
            "too many": (a, b, c, d, e, Rational(3)),
            "two in one cell": (a, b, Rational(-1), c, d),
            "unordered": (a, c, b, d, e),
            "decreasing": (e, d, c, b, a),
            "equal": (a, b, c, c, e),
            "the same twice, in the right cells otherwise": (a, b, c, d, d),
            "a sample at a root, with the right count below it": (a, Rational(-7, 3), c, d, e),
            "the first sample at a root": (Rational(-7, 3), b, c, d, e),
            "a sample at the root 1": (a, b, Rational(1), d, e),
            "a sample that skips a root": (a, c, Rational(1, 2), d, e),
            "the first sample above a root": (b, Rational(-3, 2), c, d, e),
            "the last sample below a root": (a, b, c, Rational(11, 10), d),
        }
        for label, samples in cases.items():
            with self.subTest(label):
                self.rejected(MASTER, samples, label)

    def test_samples_of_the_wrong_type_are_rejected(self):
        class Fraction(Rational):
            pass

        a, b, _, d, e = SAMPLES
        forged = {
            "an int": 0, "a float": 0.0, "a bool": False, "a str": "0", "None": None, "a pair": (0, 1),
            "a float numerator": trusted_rational(0.0, 1), "a bool numerator": trusted_rational(False, 1),
            "a bool denominator": trusted_rational(0, True), "a zero denominator": trusted_rational(0, 0),
            "a negative denominator": trusted_rational(0, -1), "0/2": trusted_rational(0, 2),
            "2/4": trusted_rational(2, 4), "empty slots": object.__new__(Rational), "a subclass": Fraction(0),
        }
        for label, value in forged.items():
            with self.subTest(label):
                self.rejected(MASTER, (a, b, value, d, e), label)
        for label, samples in (("a list", list(SAMPLES)), ("None", None), ("a Rational", a), ("a dict", {0: a}),
                               ("a generator", iter(SAMPLES))):
            with self.subTest(label):
                self.rejected(MASTER, samples, label)

    def test_bad_master_polynomials_are_rejected(self):
        for master in ((), [14, -8, -13, 4, 3], (14.0, -8, -13, 4, 3), (True, 1), ("1", 1), (1, 0), (0,), None, 5,
                       (Rational(1), Rational(1)), "x"):
            with self.subTest(master=master):
                self.rejected(master, SAMPLES)

    def test_acceptance_is_the_stated_criterion(self):
        """Cells accepts exactly the samples of the catalogue's criterion, stated with certify.upoly.count_roots:
        one more than roots, increasing, none a root, the k-th with k roots up to it."""
        def criterion(master, samples):
            part = certify.squarefree_part(master)
            return (len(samples) == certify.count_roots(part, None, None) + 1
                    and all(lower < upper for lower, upper in zip(samples, samples[1:]))
                    and all(tp.sign_at(part, sample.numerator, sample.denominator) != 0
                            and certify.count_roots(part, None, sample) == k for k, sample in enumerate(samples)))

        rng = random.Random(909)
        verdicts = Counter()
        for _ in range(150):
            master = integers(random_expression(rng))
            points = roots_of(master)
            good = samples_between(points)
            self.assertIs(cells.check_samples(master, good), True)
            rational_roots = [point for point in points if type(point) is Rational]
            for _ in range(6):
                samples = list(good)
                change = rng.randrange(5)
                if change == 0:
                    samples[rng.randrange(len(samples))] = Rational(rng.randint(-40, 40), rng.randint(1, 8))
                elif change == 1:
                    del samples[rng.randrange(len(samples))]
                elif change == 2:
                    samples.append(Rational(rng.randint(-40, 40), rng.randint(1, 8)))
                elif change == 3 and rational_roots:
                    samples[rng.randrange(len(samples))] = rng.choice(rational_roots)
                else:  # Another point of the same cell, as often as not.
                    position = rng.randrange(len(samples))
                    samples[position] = samples[position] + Rational(rng.randint(-3, 3), 16)
                if rng.randrange(5):
                    samples.sort()
                samples = tuple(samples)
                expected = criterion(master, samples)
                self.assertIs(cells.check_samples(master, samples), expected, (master, samples))
                verdicts[expected] += 1
        self.assertGreater(verdicts[True], 100)
        self.assertGreater(verdicts[False], 400)


class SignTests(unittest.TestCase):
    """Signs at the samples and at the roots."""

    @classmethod
    def setUpClass(cls):
        cls.factors = [integers(expression) for expression in (X - 1, X**2 - 2, 3 * X + 7, X**2 + 1)]
        cls.grid = cells.Cells(cells.master_polynomial(*cls.factors), SAMPLES)

    def test_signs_of_the_factors(self):
        linear, quadratic, other, positive = self.factors
        grid = self.grid
        self.assertEqual(grid.root_count, 4)  # -7/3, -sqrt 2, 1 and sqrt 2: x^2 + 1 adds none.
        self.assertEqual(grid.root_signs(linear), (-1, -1, 0, 1))
        self.assertEqual(grid.root_signs(quadratic), (1, 0, -1, 0))
        self.assertEqual(grid.root_signs(other), (0, 1, 1, 1))
        self.assertEqual(grid.root_signs(positive), (1, 1, 1, 1))
        self.assertEqual(grid.sample_signs(linear), (-1, -1, -1, 1, 1))
        self.assertEqual(grid.sample_signs(quadratic), (1, 1, -1, -1, 1))
        self.assertEqual(grid.sample_signs(other), (-1, 1, 1, 1, 1))
        self.assertEqual(grid.sample_signs(positive), (1, 1, 1, 1, 1))

    def test_repeated_factors_constants_and_the_zero_polynomial(self):
        grid = self.grid
        product = integers(-(X - 1)**2 * (X**2 - 2)**3 * (X**2 + 1))  # The sign of 2 - x^2 away from the root 1.
        self.assertEqual(grid.root_signs(product), (-1, 0, 0, 0))
        self.assertEqual(grid.sample_signs(product), (-1, -1, 1, 1, -1))
        self.assertEqual(grid.root_signs(grid.master), (0, 0, 0, 0))
        self.assertEqual(grid.root_signs(tp.neg(MASTER)), (0, 0, 0, 0))
        self.assertEqual((grid.root_signs((-3,)), grid.sample_signs((-3,))), ((-1,) * 4, (-1,) * 5))
        self.assertEqual((grid.root_signs((7,)), grid.sample_signs((7,))), ((1,) * 4, (1,) * 5))
        self.assertEqual((grid.root_signs(()), grid.sample_signs(())), ((0,) * 4, (0,) * 5))

    def test_one_sign_at_a_time(self):
        linear = self.factors[0]
        grid = self.grid
        self.assertEqual([grid.sign_at_root(linear, k) for k in (1, 2, 3, 4)], [-1, -1, 0, 1])
        self.assertEqual([grid.sign_at_sample(linear, k) for k in (0, 1, 2, 3, 4)], [-1, -1, -1, 1, 1])
        for k in (0, 5, -1, True, 1.0, "1", None):
            with self.subTest(root=k), self.assertRaises(cells.Rejected):
                grid.sign_at_root(linear, k)
        for k in (-1, 5, True, 0.0, "0", None):
            with self.subTest(sample=k), self.assertRaises(cells.Rejected):
                grid.sign_at_sample(linear, k)

    def test_a_polynomial_with_a_root_outside_the_master_has_no_sign_at_the_roots(self):
        # The last one has only roots of the master on the real line, but its factor x^2 + 2 is not in the master:
        # the engine's condition is the exact division, and it fails closed.
        for expression in (X - 5, X**2 - 3, (X - 1) * (X - 5), X, 2 * X - 1, (X**2 - 2) * (X - 1) * (3 * X + 8),
                           (X - 1) * (X**2 + 2)):
            polynomial = integers(expression)
            with self.subTest(expression=str(expression)):
                with self.assertRaises(cells.Rejected):
                    self.grid.root_signs(polynomial)
                # A sample is a rational point: every polynomial has a sign there.
                self.assertEqual(self.grid.sample_signs(polynomial),
                                 tuple(tp.sign_at(polynomial, s.numerator, s.denominator) for s in SAMPLES))

    def test_polynomials_of_the_wrong_type_are_rejected(self):
        for polynomial in ([-1, 1], (-1.0, 1), (True, 1), (-1, 0), (0,), "x", None, (Rational(-1), Rational(1)), 1):
            with self.subTest(polynomial=polynomial):
                with self.assertRaises(cells.Rejected):
                    self.grid.root_signs(polynomial)
                with self.assertRaises(cells.Rejected):
                    self.grid.sample_signs(polynomial)

    def test_relations_against_zero(self):
        table = {"=": (F, T, F), "<": (T, F, F), "<=": (T, T, F), ">": (F, F, T), ">=": (F, T, T)}
        self.assertEqual(cells.RELATIONS, tuple(table))
        for relation, truth in table.items():
            self.assertEqual(tuple(cells.holds(relation, value) for value in (-1, 0, 1)), truth)
        for relation, value in (("==", 0), ("!=", 0), ("lt", 0), (None, 0), (0, 0), ("<", 2), ("<", -2), ("<", True),
                                ("<", 0.0), ("<", None), (("<",), 0)):
            with self.subTest(relation=relation, sign=value), self.assertRaises(cells.Rejected):
                cells.holds(relation, value)

    def test_signs_agree_with_the_producer(self):
        """root_signs against mathforge.algebraic.sign_at, which refines isolating intervals, and sample_signs
        against exact evaluation, for the factors of random products and for products of their powers."""
        rng = random.Random(4242)
        seen = Counter()
        for _ in range(120):
            factors = [random_factor(rng) for _ in range(rng.randint(1, 4))]
            product = factors[0]
            for factor in factors[1:]:
                product = product * factor
            grid, points = cut(cells.master_polynomial(*map(integers, factors)))
            for expression in (*factors, product, -3 * factors[0] * factors[-1]**2, Rational(2, 7) * product**2):
                polynomial = Polynomial.from_expression(expression, X)
                expected = tuple(sign_at(polynomial, point) for point in points)
                self.assertEqual(grid.root_signs(integers(expression)), expected, str(expression))
                self.assertEqual(grid.sample_signs(integers(expression)),
                                 tuple(sign(polynomial.evaluate(sample)) for sample in grid.samples))
                seen.update(expected)
        self.assertGreater(min(seen[-1], seen[0], seen[1]), 300, seen)


class RelationTests(unittest.TestCase):
    """Relations of a polynomial against 0: the truth values from the engine's signs, against the sets of the
    relations written out by hand."""

    SQRT2 = (RealAlgebraicRoot((-2, 0, 1), 0), RealAlgebraicRoot((-2, 0, 1), 1))
    EMPTY, LINE = IntervalSet(()), IntervalSet((Interval(None, None),))

    def check(self, grid, polynomial, expected):
        """expected maps each relation to the set of the x with ``polynomial relation 0``: the engine accepts that
        set for the truth values of the relation, and rejects the set of every other relation that differs."""
        self.assertEqual(tuple(expected), cells.RELATIONS)
        for relation, value in expected.items():
            with self.subTest(relation=relation):
                truth = grid.relation_truth(polynomial, relation)
                self.assertEqual(truth, relation_truth(grid, polynomial, relation))
                self.assertIs(grid.matches(value, *truth), True)
                for other, wrong in expected.items():
                    if wrong != value:
                        self.assertIs(grid.matches(wrong, *truth), False, other)

    def test_a_double_root_between_two_cells_of_one_sign(self):
        """p = (x - 1)^2 (x^2 - 2)(3x + 7) is its own master, repeated factor included: negative below -7/3,
        positive up to -sqrt 2, negative on both sides of the double root 1 and positive above sqrt 2. The intervals
        of p < 0 touch at the false root 1, those of p <= 0 join there, and p >= 0 holds at 1 alone."""
        p = integers((X - 1)**2 * (X**2 - 2) * (3 * X + 7))
        grid = cells.Cells(p, SAMPLES)
        self.assertEqual((grid.master, grid.root_count), (MASTER, 4))
        low, high = self.SQRT2
        a, one = Rational(-7, 3), Rational(1)
        self.check(grid, p, {
            "=": IntervalSet((Interval(a, a, True, True), Interval(low, low, True, True),
                              Interval(one, one, True, True), Interval(high, high, True, True))),
            "<": IntervalSet((Interval(None, a), Interval(low, one), Interval(one, high))),
            "<=": IntervalSet((Interval(None, a, False, True), Interval(low, high, True, True))),
            ">": IntervalSet((Interval(a, low), Interval(high, None))),
            ">=": IntervalSet((Interval(a, low, True, True), Interval(one, one, True, True),
                               Interval(high, None, True, False))),
        })

    def test_roots_of_the_master_where_the_polynomial_is_not_zero(self):
        """Factors of MASTER on its cut: each has the sign of its neighbouring cells at the other roots."""
        grid = cells.Cells(MASTER, SAMPLES)
        low, high = self.SQRT2
        self.check(grid, integers(X**2 - 2), {
            "=": IntervalSet((Interval(low, low, True, True), Interval(high, high, True, True))),
            "<": IntervalSet((Interval(low, high),)),
            "<=": IntervalSet((Interval(low, high, True, True),)),
            ">": IntervalSet((Interval(None, low), Interval(high, None))),
            ">=": IntervalSet((Interval(None, low, False, True), Interval(high, None, True, False))),
        })
        a = Rational(-7, 3)  # 3x + 7 > 0 holds at three roots of the master and on four of its cells.
        self.check(grid, integers(3 * X + 7), {
            "=": IntervalSet((Interval(a, a, True, True),)),
            "<": IntervalSet((Interval(None, a),)),
            "<=": IntervalSet((Interval(None, a, False, True),)),
            ">": IntervalSet((Interval(a, None),)),
            ">=": IntervalSet((Interval(a, None, True, False),)),
        })

    def test_the_empty_set_and_the_whole_line(self):
        empty, line = self.EMPTY, self.LINE
        # x^2 + 1 has no real root: one cell, the whole line.
        self.check(cells.Cells((1, 0, 1), (Rational(5),)), (1, 0, 1),
                   {"=": empty, "<": empty, "<=": empty, ">": line, ">=": line})
        # A negative constant, and the zero polynomial, on a cut with roots.
        grid = cells.Cells(MASTER, SAMPLES)
        self.check(grid, (-3,), {"=": empty, "<": line, "<=": line, ">": empty, ">=": empty})
        self.check(grid, (), {"=": line, "<": empty, "<=": line, ">": empty, ">=": line})
        # A square: (x - 1)^2 >= 0 everywhere and < 0 nowhere; = 0 and <= 0 at its root alone, and > 0 on two open
        # half-lines that touch at that false root.
        square = integers((X - 1)**2)
        point = IntervalSet((Interval(Rational(1), Rational(1), True, True),))
        self.check(cells.Cells(square, (Rational(0), Rational(2))), square, {
            "=": point, "<": empty, "<=": point,
            ">": IntervalSet((Interval(None, Rational(1)), Interval(Rational(1), None))), ">=": line})

    def test_bad_relations_and_polynomials_with_other_roots(self):
        grid = cells.Cells(MASTER, SAMPLES)
        self.assertEqual(grid.relation_truth((-1, 1), "<"), ((T, T, F, F), (T, T, T, F, F)))
        for relation in ("==", "!=", "lt", "", None, 0, True, ("<",), b"<"):
            with self.subTest(relation=relation), self.assertRaises(cells.Rejected):
                grid.relation_truth((-1, 1), relation)
        for polynomial in (integers(X - 5), integers((X - 1) * (X**2 + 2)), [-1, 1], (-1.0, 1), (True, 1), (-1, 0)):
            with self.subTest(polynomial=polynomial), self.assertRaises(cells.Rejected):
                grid.relation_truth(polynomial, "<")


class ReconstructionTests(unittest.TestCase):
    """Truth values to canonical sets, and back."""

    def test_maximal_runs_of_true_members(self):
        grid, _ = cut(integers((X**3 - 2) * (X - 3)))  # Two roots: the cube root of 2 and 3.
        table = [
            ((F, F), (F, F, F), ()),
            ((T, T), (T, T, T), WHOLE_LINE),
            ((F, F), (T, T, T), ((None, 1, F, F), (1, 2, F, F), (2, None, F, F))),  # False roots split the line.
            ((T, T), (F, F, F), ((1, 1, T, T), (2, 2, T, T))),  # Isolated points.
            ((T, F), (T, F, F), ((None, 1, F, T),)),
            ((T, T), (F, T, F), ((1, 2, T, T),)),
            ((F, F), (F, T, F), ((1, 2, F, F),)),
            ((F, T), (F, T, F), ((1, 2, F, T),)),
            ((F, F), (F, F, T), ((2, None, F, F),)),
            ((F, T), (F, F, T), ((2, None, T, F),)),
            ((T, F), (T, T, T), ((None, 2, F, F), (2, None, F, F))),  # A true root joins its cells, a false one not.
            ((F, T), (T, T, T), ((None, 1, F, F), (1, None, F, F))),
            ((T, T), (T, F, T), ((None, 1, F, T), (2, None, T, F))),
            ((T, F), (F, F, T), ((1, 1, T, T), (2, None, F, F))),
        ]
        for root_truth, cell_truth, expected in table:
            with self.subTest(root_truth=root_truth, cell_truth=cell_truth):
                self.assertEqual(grid.reconstruct(root_truth, cell_truth), expected)
                self.assertEqual(truth_of(expected, 2), (root_truth, cell_truth))

    def test_one_cell(self):
        grid = cells.Cells((1, 0, 1), (Rational(0),))
        self.assertEqual(grid.reconstruct((), (F,)), ())
        self.assertEqual(grid.reconstruct((), (T,)), WHOLE_LINE)
        self.assertIs(grid.matches(IntervalSet(()), (), (F,)), True)
        self.assertIs(grid.matches(IntervalSet((Interval(None, None),)), (), (T,)), True)
        self.assertIs(grid.matches(IntervalSet(()), (), (T,)), False)
        self.assertIs(grid.matches(IntervalSet((Interval(None, None),)), (), (F,)), False)
        self.assertEqual(grid.membership(IntervalSet((Interval(None, None),))), ((), (T,)))

    def test_truth_values_of_the_wrong_type_are_rejected(self):
        grid, points = cut(integers((X**3 - 2) * (X - 3)))
        claim = assembled((T, F), (F, T, F), points)
        self.assertIs(grid.matches(claim, (T, F), (F, T, F)), True)
        for root_truth, cell_truth in (([T, F], (F, T, F)), ((T, F), [F, T, F]), ((1, 0), (F, T, F)),
                                       ((T, F), (0, 1, 0)), ((T,), (F, T, F)), ((T, F), (F, T)), ((T, F, F), (F, T)),
                                       (None, (F, T, F)), ((T, F), None), ((T, None), (F, T, F)),
                                       ((T, F), (F, "T", F)), ((T, F), (F, 1.0, F))):
            with self.subTest(root_truth=root_truth, cell_truth=cell_truth):
                with self.assertRaises(cells.Rejected):
                    grid.reconstruct(root_truth, cell_truth)
                self.assertIs(grid.matches(claim, root_truth, cell_truth), False)

    def test_every_truth_assignment_round_trips(self):
        """With the three roots -sqrt 2, 0 and sqrt 2: the reconstruction is the set that the producer's IntervalSet
        merges from the true cells and roots; that set has those truth values as its membership, matches them and
        matches no assignment that differs from them in one value."""
        grid, points = cut(integers(X * (X**2 - 2)))
        self.assertEqual(grid.root_count, 3)
        seen = set()
        for bits in itertools.product((F, T), repeat=7):
            root_truth, cell_truth = bits[:3], bits[3:]
            value = assembled(root_truth, cell_truth, points)
            expected = index_form(value, points)
            seen.add(expected)
            self.assertEqual(grid.reconstruct(root_truth, cell_truth), expected)
            self.assertEqual(truth_of(expected, 3), (root_truth, cell_truth))
            self.assertEqual(grid.index_form(value), expected)
            self.assertEqual(grid.membership(value), (root_truth, cell_truth))
            self.assertIs(grid.matches(value, root_truth, cell_truth), True)
            for flip in range(7):
                other = bits[:flip] + (not bits[flip],) + bits[flip + 1:]
                self.assertIs(grid.matches(value, other[:3], other[3:]), False, (bits, flip))
        self.assertEqual(len(seen), 128)


class ClaimTests(unittest.TestCase):
    """Claimed sets: endpoints located among the roots, the canonical form, the comparison."""

    @classmethod
    def setUpClass(cls):
        # The roots -sqrt 2, 0, 1/2, sqrt 2 and 3, and a factor without a real root.
        cls.expression = X * (2 * X - 1) * (X**2 - 2) * (X - 3) * (X**2 + 1)
        cls.grid, cls.points = cut(integers(cls.expression))
        cls.low, cls.high = RealAlgebraicRoot((-2, 0, 1), 0), RealAlgebraicRoot((-2, 0, 1), 1)

    def test_endpoints_are_located_among_the_roots(self):
        grid = self.grid
        self.assertEqual(grid.root_count, 5)
        for k, point in enumerate(self.points, 1):
            self.assertEqual(grid.locate(point), k)
        for k, point in enumerate((self.low, Rational(0), Rational(1, 2), self.high, Rational(3)), 1):
            self.assertEqual(grid.locate(point), k)
        # Other names of the same numbers: roots of the master polynomial itself, of a product of its factors and
        # of linear polynomials, each counted from 0 among the real roots of its polynomial.
        for k in range(1, 6):
            self.assertEqual(grid.locate(forged_root(grid.master, k - 1)), k)
        cubic = (0, -2, 0, 1)  # x^3 - 2x: -sqrt 2, 0, sqrt 2.
        self.assertEqual([grid.locate(RealAlgebraicRoot(cubic, index)) for index in (0, 1, 2)], [1, 2, 4])
        with_complex = integers((X**2 - 2) * (X**2 + 1))
        self.assertEqual([grid.locate(RealAlgebraicRoot(with_complex, index)) for index in (0, 1)], [1, 4])
        self.assertEqual(grid.locate(RealAlgebraicRoot((-1, 2), 0)), 3)
        self.assertEqual(grid.locate(RealAlgebraicRoot((-3, 1), 0)), 5)

    def test_values_that_are_no_root_are_not_located(self):
        grid = self.grid
        longer = integers(self.expression * (X - 9))
        rejected = {
            "a rational that is no root": Rational(1),
            "a sample": grid.samples[2],
            "a rational beyond the samples": Rational(10**6),
            "an int": 0, "a float": 0.0, "a bool": False, "a str": "0", "None": None, "a pair": (0, 1),
            "0/2": trusted_rational(0, 2), "2/4": trusted_rational(2, 4), "3/-1": trusted_rational(-3, -1),
            "a float numerator": trusted_rational(0.0, 1), "empty slots": object.__new__(Rational),
            "a root of another polynomial": RealAlgebraicRoot((-3, 0, 1), 0),
            "-sqrt 2 as a root of a polynomial with another root": RealAlgebraicRoot(integers((X**2 - 2) * (X - 7)), 0),
            "an index past the degree": forged_root((-2, 0, 1), 2),
            "an index past the real roots": forged_root(integers((X**2 - 2) * (X**2 + 1)), 2),
            "a polynomial without a real root": forged_root((1, 0, 1), 0),
            "a negative index": forged_root((-2, 0, 1), -1),
            "a bool index": forged_root((-2, 0, 1), True),
            "a float index": forged_root((-2, 0, 1), 1.0),
            "not primitive": forged_root((-4, 0, 2), 1),
            "a negative leading coefficient": forged_root((2, 0, -1), 1),
            "not square-free": forged_root(integers((X**2 - 2)**2), 1),
            "a list of coefficients": forged_root([-2, 0, 1], 1),
            "a float coefficient": forged_root((-2.0, 0, 1), 1),
            "a bool coefficient": forged_root((-2, False, 1), 1),
            "a constant": forged_root((1,), 0),
            "a trailing zero": forged_root((-2, 0, 1, 0), 1),
            "no fields": object.__new__(RealAlgebraicRoot),
            "a polynomial longer than the master": forged_root(longer, 0),
        }
        for label, endpoint in rejected.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                grid.locate(endpoint)

    def test_a_canonical_claim_is_accepted_whatever_names_its_endpoints(self):
        grid, (a, b, c, d, e) = self.grid, self.points
        # (-inf, -sqrt 2] with [0, 1/2), the point sqrt 2 and (3, inf).
        expected = ((None, 1, F, T), (2, 3, T, F), (4, 4, T, T), (5, None, F, F))
        truth = ((T, T, F, T, F), (T, F, T, F, F, T))
        claims = (
            IntervalSet((Interval(None, a, False, True), Interval(b, c, True, False), Interval(d, d, True, True),
                         Interval(e, None))),
            IntervalSet((Interval(None, self.low, False, True), Interval(0, Rational(1, 2), True, False),
                         Interval(self.high, self.high, True, True), Interval(3, None))),
            records(expected, [forged_root(grid.master, index) for index in range(5)]),
            # The two ends of one point under two names.
            forged_set((forged_interval(None, a, False, True), forged_interval(b, RealAlgebraicRoot((-1, 2), 0), True),
                        forged_interval(self.high, forged_root(grid.master, 3), True, True),
                        forged_interval(RealAlgebraicRoot((-3, 1), 0), None))),
        )
        self.assertEqual(grid.reconstruct(*truth), expected)
        for claim in claims:
            self.assertEqual(grid.index_form(claim), expected)
            self.assertEqual(grid.membership(claim), truth)
            self.assertIs(grid.matches(claim, *truth), True)
        self.assertIs(grid.matches(IntervalSet(()), (F,) * 5, (F,) * 6), True)
        self.assertIs(grid.matches(IntervalSet((Interval(None, None),)), (T,) * 5, (T,) * 6), True)
        self.assertIs(grid.matches(IntervalSet(()), *truth), False)
        self.assertIs(grid.matches(IntervalSet((Interval(None, None),)), *truth), False)
        self.assertIs(grid.matches(claims[0], (F,) * 5, (F,) * 6), False)

    def test_another_way_of_writing_the_set_is_rejected(self):
        """Each claim denotes the set beside it, which is accepted in canonical form only."""
        cases = {
            "unsorted": ([(2, 3, T, F), (None, 1, F, T)], [(None, 1, F, T), (2, 3, T, F)]),
            "overlapping": ([(None, 3, F, F), (2, 4, F, F)], [(None, 4, F, F)]),
            "nested": ([(1, 5, F, F), (2, 3, T, T)], [(1, 5, F, F)]),
            "touching, closed then open": ([(1, 2, F, T), (2, 3, F, F)], [(1, 3, F, F)]),
            "touching, open then closed": ([(1, 2, F, F), (2, 3, T, F)], [(1, 3, F, F)]),
            "touching, closed twice": ([(1, 2, F, T), (2, 3, T, F)], [(1, 3, F, F)]),
            "a point before its interval": ([(2, 2, T, T), (2, 3, F, F)], [(2, 3, T, F)]),
            "a point after its interval": ([(1, 2, F, F), (2, 2, T, T)], [(1, 2, F, T)]),
            "a point inside an interval": ([(1, 3, F, F), (2, 2, T, T)], [(1, 3, F, F)]),
            "twice the same interval": ([(1, 2, F, F), (1, 2, F, F)], [(1, 2, F, F)]),
            "twice the same point": ([(2, 2, T, T), (2, 2, T, T)], [(2, 2, T, T)]),
            "an empty open interval": ([(1, 2, F, F), (3, 3, F, F)], [(1, 2, F, F)]),
            "an empty half-open interval": ([(3, 3, T, F)], []),
            "an empty interval alone": ([(3, 3, F, F)], []),
            "a reversed interval": ([(3, 2, F, F)], []),
            "a reversed closed interval": ([(1, 2, F, F), (5, 4, T, T)], [(1, 2, F, F)]),
            "minus infinity closed": ([(None, 1, T, F)], [(None, 1, F, F)]),
            "plus infinity closed": ([(5, None, T, T)], [(5, None, T, F)]),
            "the whole line closed": ([(None, None, T, T)], list(WHOLE_LINE)),
            "the whole line and more": ([(None, None, F, F), (1, 2, F, F)], list(WHOLE_LINE)),
            "the whole line twice": ([(None, None, F, F), (None, None, F, F)], list(WHOLE_LINE)),
            "two unbounded halves that overlap": ([(None, 3, F, F), (2, None, F, F)], list(WHOLE_LINE)),
        }
        for label, (written, canonical) in cases.items():
            with self.subTest(label):
                truth = truth_of(canonical, 5)
                self.assertIs(self.grid.matches(records(canonical, self.points), *truth), True)
                self.assertIs(self.grid.matches(records(written, self.points), *truth), False)
                with self.assertRaises(cells.Rejected):
                    self.grid.index_form(records(written, self.points))
                with self.assertRaises(cells.Rejected):
                    self.grid.membership(records(written, self.points))

    def test_random_claims_against_an_oracle(self):
        """Random claims in index form, canonical or not, with their own truth values or random ones: accepted
        exactly when the claim is canonical and holds the true members only, by the test's own reading of both."""
        rng = random.Random(31)
        grids = [cut(integers(expression)) for expression in (self.expression, (X**2 - 2) * (X**2 - 3), X**2 + 1)]
        verdicts = Counter()
        for trial in range(6_000):
            grid, points = grids[trial % 3]
            r = len(points)
            ends = [None, *range(1, r + 1)]
            claim = tuple((rng.choice(ends), rng.choice(ends), rng.random() < 0.5, rng.random() < 0.5)
                          for _ in range(rng.randint(0, 4)))
            if rng.random() < 0.3:
                claim = tuple(sorted(claim, key=lambda item: (-1 if item[0] is None else item[0], not item[2])))
            if rng.random() < 0.5 and all(map(proper, claim)):
                truth = truth_of(claim, r)
            else:
                bits = [rng.random() < 0.5 for _ in range(2 * r + 1)]
                truth = tuple(bits[1::2]), tuple(bits[0::2])
            expected = canonical(claim) and truth_of(claim, r) == truth
            names = points if trial % 2 else [forged_root(grid.master, index) for index in range(r)]
            self.assertIs(grid.matches(records(claim, names), *truth), expected, (claim, truth))
            verdicts[expected, canonical(claim)] += 1
        self.assertGreater(min(verdicts.values()), 500, verdicts)

    def test_random_algebraic_endpoints_against_the_producer(self):
        """Records (P, i) with products of factors of the master or of other polynomials, repeated factors, a
        content, a negative leading coefficient, any index: located exactly when they are what RealAlgebraicRoot
        stores and P divides the master, and then at the root that compare_real finds equal."""
        pool = [integers(expression) for expression in (X**2 - 2, 2 * X - 1, X, X - 3, X**3 - 2, X**2 + 1,
                                                          X**2 - X - 1, 3 * X + 7)]
        rng = random.Random(99)
        located = Counter()
        for _ in range(80):
            chosen = rng.sample(pool, rng.randint(1, 5))
            grid, points = cut(cells.master_polynomial(*chosen))
            for _ in range(20):
                source = chosen if rng.random() < 0.6 else pool
                factors = rng.sample(source, rng.randint(1, min(3, len(source))))
                polynomial = (1,)
                for factor in factors:
                    polynomial = tp.mul(polynomial, factor)
                change = rng.randrange(6)
                if change == 1:
                    polynomial = tp.mul(polynomial, factors[0])
                elif change == 2:
                    polynomial = tp.scale(polynomial, rng.choice((2, 3)))
                elif change == 3:
                    polynomial = tp.neg(polynomial)
                real = certify.count_roots(certify.squarefree_part(polynomial), None, None)
                index = rng.randrange(real) if real and rng.random() < 0.7 else rng.randint(-1, len(polynomial))
                try:
                    record = RealAlgebraicRoot(polynomial, index)
                    stored = (record.integer_coefficients, record.real_index) == (polynomial, index)
                except mf.MathForgeError:
                    record, stored = None, False
                division = tp.divmod_exact(grid.master, polynomial) if stored else None
                if division is not None and not division[1]:
                    expected = [k for k, point in enumerate(points, 1) if compare_real(record, point) == 0]
                    self.assertEqual(grid.locate(forged_root(polynomial, index)), expected[0])
                    located[True] += 1
                else:
                    with self.assertRaises(cells.Rejected, msg=(polynomial, index)):
                        grid.locate(forged_root(polynomial, index))
                    located[False] += 1
        self.assertGreater(min(located[True], located[False]), 300, located)

    def test_forged_records_are_rejected(self):
        """Each claim is the set [0, 1/2) with (3, inf), but for one value of the wrong type or place."""
        class Sub(IntervalSet):
            pass

        class Lookalike:
            def __init__(self, **fields):
                self.__dict__.update(fields)

        other_set = type("IntervalSet", (Lookalike,), {"__module__": "elsewhere"})
        other_interval = type("Interval", (Lookalike,), {"__module__": "elsewhere"})
        grid, (_, b, c, _, e) = self.grid, self.points
        canonical = [(2, 3, T, F), (5, None, F, F)]
        truth = truth_of(canonical, 5)
        first, second = forged_interval(b, c, True, False), forged_interval(e, None)
        good = forged_set((first, second))
        self.assertIs(grid.matches(good, *truth), True)
        forged = {
            "a list of intervals": forged_set([first, second]),
            "no intervals attribute": object.__new__(IntervalSet),
            "None for the intervals": forged_set(None),
            "a tuple in place of an Interval": forged_set((first, (e, None, False, False))),
            "None in place of an Interval": forged_set((first, None)),
            "an Interval without fields": forged_set((first, object.__new__(Interval))),
            "int flags": forged_set((forged_interval(b, c, 1, 0), second)),
            "a None flag": forged_set((forged_interval(b, c, True, None), second)),
            "a str flag": forged_set((first, forged_interval(e, None, "", False))),
            "an int endpoint": forged_set((forged_interval(0, c, True, False), second)),
            "a float endpoint": forged_set((forged_interval(b, 0.5, True, False), second)),
            "a str endpoint": forged_set((first, forged_interval("3", None))),
            "a False endpoint for infinity": forged_set((first, forged_interval(e, False))),
            "an endpoint that is no root": forged_set((forged_interval(b, Rational(1), True, False), second)),
            "a tuple of intervals": (first, second),
            "None": None,
            "a subclass of IntervalSet": Sub(good.intervals),
            "a class of the same name elsewhere": other_set(intervals=good.intervals),
            "an Interval of the same name elsewhere": forged_set(
                (first, other_interval(lower=e, upper=None, left_closed=False, right_closed=False))),
            "an Interval for the set": first,
            "a FiniteSet": mf.FiniteSet((b,)),
        }
        for label, claim in forged.items():
            with self.subTest(label):
                self.assertIs(grid.matches(claim, *truth), False)
                with self.assertRaises(cells.Rejected):
                    grid.index_form(claim)
        empty = truth_of([], 5)
        self.assertIs(grid.matches(IntervalSet(()), *empty), True)
        for label, claim in (("an EmptySet", mf.EmptySet()), ("a list", forged_set([])), ("a tuple", ()),
                             ("None", None)):
            with self.subTest(label):
                self.assertIs(grid.matches(claim, *empty), False)
        self.assertIs(grid.matches(mf.UniversalSet(), *truth_of(WHOLE_LINE, 5)), False)


class SetTests(unittest.TestCase):
    """The cut of set endpoints, and the set operations that it serves."""

    SQRT2 = RealAlgebraicRoot((-2, 0, 1), 1)
    # (-inf, -2] with [-sqrt 2, 1), the point 3/2 and (2, inf); (-3, -sqrt 2) with [0, sqrt 2] and [2, 5).
    A = IntervalSet((Interval(None, -2, False, True), Interval(RealAlgebraicRoot((-2, 0, 1), 0), 1, True, False),
                     Interval(Rational(3, 2), Rational(3, 2), True, True), Interval(2, None)))
    B = IntervalSet((Interval(-3, RealAlgebraicRoot((-2, 0, 1), 0)), Interval(0, SQRT2, True, True),
                     Interval(2, 5, True, False)))

    @staticmethod
    def decide(combine, claim, *sets):
        """set_check with the producer's samples."""
        return set_check(combine, claim, sets)

    def test_the_cut_of_endpoints(self):
        """Rational endpoints are cut points, in increasing order and each once; the polynomials of the algebraic
        ones make the master polynomial."""
        first = IntervalSet((Interval(Rational(1, 2), self.SQRT2, True, False), Interval(3, None)))
        self.assertEqual(cells.endpoint_cut(first), ((-2, 0, 1), (Rational(1, 2), Rational(3))))
        self.assertEqual(cells.endpoint_cut(), ((1,), ()))
        self.assertEqual(cells.endpoint_cut(IntervalSet(()), IntervalSet((Interval(None, None),))), ((1,), ()))
        # Equal endpoints under several names, and repeated sets, give each point once: 0 as a Rational and as a
        # root of x^3 - 2x, sqrt 2 as a root of x^3 - 2x and of x^2 - 2.
        second = IntervalSet((Interval(0, RealAlgebraicRoot((0, -2, 0, 1), 2)),))
        third = IntervalSet((Interval(RealAlgebraicRoot((0, -2, 0, 1), 1), self.SQRT2, True, True),))
        self.assertEqual(cells.endpoint_cut(second, third, second), ((0, -2, 0, 1), (Rational(0),)))
        self.assertEqual(cells.endpoint_cut(self.A, self.B),
                         ((-2, 0, 1), (*map(Rational, (-3, -2, 0, 1)), Rational(3, 2), Rational(2), Rational(5))))
        # A root of a linear polynomial is a rational, so it is a cut point; the golden ratio stays in the master.
        golden = RealAlgebraicRoot((-1, -1, 1), 1)
        linear = forged_set((forged_interval(forged_root((-1, 2), 0), golden),
                             forged_interval(forged_root((7, 3), 0), None)))
        self.assertEqual(cells.endpoint_cut(linear), ((-1, -1, 1), (Rational(-7, 3), Rational(1, 2))))
        # Cut points and master make the cut of the producer's points.
        master, cuts = cells.endpoint_cut(self.A, self.B)
        grid, points = cut_with(master, cuts)
        self.assertEqual(grid.root_count, 9)
        self.assertEqual([grid.locate(point) for point in points], list(range(1, 10)))

    def test_master_polynomials(self):
        self.assertEqual(cells.master_polynomial(), (1,))
        self.assertEqual(cells.master_polynomial((), (5,), (-7,)), (1,))
        self.assertEqual(cells.master_polynomial((-1, 1), (-1, 1), (1, -2, 1), (3, -3)), (-1, 1))
        self.assertEqual(cells.master_polynomial((2, -2), (0, 0, 3), ()), (0, -1, 1))
        self.assertEqual(cells.master_polynomial(integers((X**2 - 2)**2 * (X - 1)), integers((1 - X) * (X + 4))),
                         integers((X**2 - 2) * (X - 1) * (X + 4)))
        for polynomials in (([-1, 1],), ((-1.0, 1),), ((True, 1),), ((-1, 0),), (((-1, 1), (1, 1)),), (None,), ("x",)):
            with self.subTest(polynomials=polynomials), self.assertRaises(cells.Rejected):
                cells.master_polynomial(*polynomials)

    def test_malformed_sets_have_no_cut(self):
        good = forged_interval(Rational(0), self.SQRT2)
        for label, value in {
            "a tuple": (good,), "None": None, "an Interval": good, "a list of intervals": forged_set([good]),
            "an int endpoint": forged_set((forged_interval(0, self.SQRT2),)),
            "a float endpoint": forged_set((forged_interval(Rational(0), 1.5),)),
            "int flags": forged_set((forged_interval(Rational(0), self.SQRT2, 0, 1),)),
            "0/2": forged_set((forged_interval(trusted_rational(0, 2), self.SQRT2),)),
            "a polynomial that is not primitive": forged_set(
                (forged_interval(Rational(0), forged_root((-4, 0, 2), 1)),)),
            "a list of coefficients": forged_set((forged_interval(Rational(0), forged_root([-2, 0, 1], 1)),)),
            "a bool index": forged_set((forged_interval(Rational(0), forged_root((-2, 0, 1), True)),)),
            "a linear polynomial that is not primitive": forged_set((forged_interval(Rational(0),
                                                                                     forged_root((-2, 4), 0)),)),
            "an EmptySet": mf.EmptySet(),
        }.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                cells.endpoint_cut(self.A, value)

    def test_set_operations_on_hand_built_sets(self):
        a, b = self.A, self.B
        low = RealAlgebraicRoot((-2, 0, 1), 0)
        union = IntervalSet((Interval(None, self.SQRT2, False, True),
                             Interval(Rational(3, 2), Rational(3, 2), True, True), Interval(2, None, True, False)))
        intersection = IntervalSet((Interval(-3, -2, False, True), Interval(0, 1, True, False), Interval(2, 5)))
        complement = IntervalSet((Interval(-2, low), Interval(1, Rational(3, 2), True, False),
                                  Interval(Rational(3, 2), 2, False, True)))
        difference = IntervalSet((Interval(None, -3, False, True), Interval(low, 0, True, False),
                                  Interval(Rational(3, 2), Rational(3, 2), True, True), Interval(5, None, True, False)))
        self.assertEqual(a.union(b), union)
        self.assertEqual(a.intersection(b), intersection)
        operations = {
            "union": (lambda x, y: x or y, union, (a, b)),
            "intersection": (lambda x, y: x and y, intersection, (a, b)),
            "difference": (lambda x, y: x and not y, difference, (a, b)),
            "complement": (lambda x: not x, complement, (a,)),
        }
        for name, (combine, value, sets) in operations.items():
            with self.subTest(name):
                self.assertIs(self.decide(combine, value, *sets), True)
                for other, (_, wrong, _) in operations.items():
                    if other != name:
                        self.assertIs(self.decide(combine, wrong, *sets), False, other)
        # The complement of the complement, the empty set and the whole line.
        self.assertIs(self.decide(lambda x: not x, a, complement), True)
        self.assertIs(self.decide(lambda x: not x, IntervalSet(()), IntervalSet((Interval(None, None),))), True)
        self.assertIs(self.decide(lambda x: not x, IntervalSet((Interval(None, None),)), IntervalSet(())), True)
        self.assertIs(self.decide(lambda x, y: x and y, IntervalSet(()), a, complement), True)
        self.assertIs(self.decide(lambda x, y: x or y, IntervalSet((Interval(None, None),)), a, complement), True)
        self.assertIs(self.decide(lambda x, y: x or y, IntervalSet(()), a, complement), False)

    def test_random_unions_and_intersections(self):
        """Against IntervalSet.union and IntervalSet.intersection of the producer, with the membership of every root
        and sample against IntervalSet.contains."""
        pool = [Rational(-3), Rational(-2), Rational(-1, 2), Rational(0), Rational(1, 3), Rational(1), Rational(3, 2),
                Rational(2), Rational(5), RealAlgebraicRoot((-2, 0, 1), 0), self.SQRT2,
                RealAlgebraicRoot((-3, 0, 1), 1), RealAlgebraicRoot((-2, 0, 0, 1), 0),
                RealAlgebraicRoot((1, -3, 0, 1), 0), RealAlgebraicRoot((1, -3, 0, 1), 1),
                RealAlgebraicRoot((1, -3, 0, 1), 2)]
        pool.sort(key=cmp_to_key(compare_real))
        rng = random.Random(77)

        def random_set():
            intervals = []
            for _ in range(rng.randint(0, 4)):
                low, high = sorted(rng.choices(range(-1, len(pool) + 1), k=2))
                lower = None if low < 0 else pool[min(low, len(pool) - 1)]
                upper = None if high >= len(pool) else pool[max(high, 0)]
                intervals.append(Interval(lower, upper, lower is not None and rng.random() < 0.5,
                                          upper is not None and rng.random() < 0.5))
            return IntervalSet(tuple(intervals))

        differ = 0
        for _ in range(80):
            a, b = random_set(), random_set()
            union, intersection = a.union(b), a.intersection(b)
            self.assertIs(self.decide(lambda x, y: x or y, union, a, b), True, (a, b))
            self.assertIs(self.decide(lambda x, y: x and y, intersection, a, b), True, (a, b))
            if union != intersection:
                differ += 1
                self.assertIs(self.decide(lambda x, y: x or y, intersection, a, b), False, (a, b))
                self.assertIs(self.decide(lambda x, y: x and y, union, a, b), False, (a, b))
            grid, points = cut_with(*cells.endpoint_cut(a, b))
            for value in (a, b, union, intersection):
                self.assertEqual(grid.membership(value), (tuple(value.contains(point) for point in points),
                                                          tuple(value.contains(sample) for sample in grid.samples)))
        self.assertGreater(differ, 60)


class DifferentialTests(unittest.TestCase):
    """The engine against the solver: it accepts the solver's set and rejects every mutation of it."""

    def run_case(self, grid, points, value, root_truth, cell_truth, rng, kinds, context):
        """Check one solved problem; kinds counts the mutations that were rejected."""
        true = index_form(value, points)
        self.assertEqual(grid.reconstruct(root_truth, cell_truth), true, context)
        self.assertEqual(truth_of(true, len(points)), (root_truth, cell_truth), context)
        self.assertIs(grid.matches(value, root_truth, cell_truth), True, context)
        self.assertEqual(grid.membership(value), (root_truth, cell_truth), context)
        # The same set written with forged records, and with every endpoint as a root of the master polynomial or of
        # the linear polynomial of its cut point.
        self.assertIs(grid.matches(records(true, points), root_truth, cell_truth), True, context)
        renamed = records(true, other_names(grid, points))
        self.assertIs(grid.matches(renamed, root_truth, cell_truth), True, context)
        for kind, claim in mutations(true, points, grid, rng):
            self.assertIs(grid.matches(claim, root_truth, cell_truth), False, (context, kind))
            kinds[kind] += 1
        # The set is the set of no other truth values.
        bits = root_truth + cell_truth
        for flip in range(len(bits)):
            other = bits[:flip] + (not bits[flip],) + bits[flip + 1:]
            self.assertIs(grid.matches(value, other[:len(points)], other[len(points):]), False, (context, flip))
            kinds["a truth value flipped"] += 1

    def test_polynomial_relations(self):
        """400 seeded polynomials p with rational coefficients, each with one of the five relations: the solver's
        set of p relation 0, the truth values from the engine's signs of p. Every other case cuts the line at the
        roots of a second polynomial as well, so that p has roots of the master where it is not zero. Each case runs
        twice: with every root in the master polynomial, and with the rational roots split off as cut points, so that
        the master holds the irrational roots alone and p vanishes at cut points."""
        rng = random.Random(20261111)
        kinds, cut_kinds = Counter(), Counter()
        seen = Counter()
        for case in range(400):
            expression = random_expression(rng)
            relation = cells.RELATIONS[case % 5]
            polynomial = integers(expression)
            extra = () if case % 2 else integers(random_factor(rng))
            master = cells.master_polynomial(polynomial, extra) if extra else polynomial
            grid, points = cut(master)
            root_truth, cell_truth = grid.relation_truth(polynomial, relation)
            context = (case, relation, str(expression))
            source = Polynomial.from_expression(expression, X)
            value = solve(expression, relation)
            self.assertEqual(grid.root_signs(polynomial), tuple(sign_at(source, point) for point in points), context)
            self.run_case(grid, points, value, root_truth, cell_truth, rng, kinds, context)
            seen["roots"] += len(points)
            seen["algebraic"] += sum(type(point) is RealAlgebraicRoot for point in points)
            seen["nonzero at a root"] += sum(value != 0 for value in grid.root_signs(polynomial))
            # The same problem with rational cut points: the same points, the same truth values, the same verdicts.
            split, split_points = cut_with(*split_cut(polynomial, extra))
            self.assertEqual(len(split_points), len(points), context)
            self.assertTrue(all(compare_real(a, b) == 0 for a, b in zip(split_points, points)), context)
            self.assertEqual(split.root_count, grid.root_count, context)
            self.assertEqual(split.relation_truth(polynomial, relation), (root_truth, cell_truth), context)
            self.assertEqual(split.root_signs(polynomial), grid.root_signs(polynomial), context)
            self.run_case(split, split_points, value, root_truth, cell_truth, random.Random(case), cut_kinds,
                          ("cut points",) + context)
            seen["cut points"] += len(split.cuts)
            seen["zero at a cut point"] += sum(not tp.sign_at(polynomial, q.numerator, q.denominator)
                                               for q in split.cuts)
        self.assertGreater(seen["algebraic"], 300, seen)
        self.assertGreater(seen["roots"] - seen["algebraic"], 300, seen)
        self.assertGreater(seen["nonzero at a root"], 100, seen)
        self.assertEqual(seen["cut points"], seen["roots"] - seen["algebraic"], seen)
        self.assertGreater(seen["zero at a cut point"], 300, seen)
        for counter in (kinds, cut_kinds):
            for kind in (*MUTATION_KINDS, "a truth value flipped"):
                self.assertGreater(counter[kind], 40, kind)
            self.assertGreater(sum(counter.values()), 400 * 12)

    def test_rational_function_relations(self):
        """150 seeded quotients n/d: the solver's set of n/d relation 0 on the domain d != 0. The truth at a root
        and in a cell comes from two signs: false where d = 0, the relation of sign n * sign d elsewhere. Each case
        runs with every root in the master polynomial, and with the rational roots of n and d as cut points, where
        Cells.truth decides from the two signs at once."""
        rng = random.Random(1789)
        kinds, cut_kinds = Counter(), Counter()
        poles = 0
        for case in range(150):
            numerator = Rational(0) * X if case % 25 == 24 else random_expression(rng)
            denominator = random_factor(rng) * random_factor(rng) ** rng.choice((1, 2))
            relation = cells.RELATIONS[case % 5]
            n, d = integers(numerator), integers(denominator)
            grid, points = cut(cells.master_polynomial(n, d))

            def truth(numerator_signs, denominator_signs):
                return tuple(below != 0 and cells.holds(relation, above * below)
                             for above, below in zip(numerator_signs, denominator_signs))

            root_truth = truth(grid.root_signs(n), grid.root_signs(d))
            cell_truth = truth(grid.sample_signs(n), grid.sample_signs(d))
            poles += grid.root_signs(d).count(0)
            value = solve(mf.rational_function(numerator, denominator, variable=X), relation)
            context = (case, relation, str(numerator), str(denominator))
            self.run_case(grid, points, value, root_truth, cell_truth, rng, kinds, context)
            split, split_points = cut_with(*split_cut(n, d))
            self.assertEqual(split.root_count, grid.root_count, context)
            decided = split.truth(lambda signs: signs[1] != 0 and cells.holds(relation, signs[0] * signs[1]), n, d)
            self.assertEqual(decided, (root_truth, cell_truth), context)
            self.run_case(split, split_points, value, root_truth, cell_truth, random.Random(case), cut_kinds,
                          ("cut points",) + context)
        self.assertGreater(poles, 150)
        for counter in (kinds, cut_kinds):
            for kind in (*MUTATION_KINDS, "a truth value flipped"):
                self.assertGreater(counter[kind], 10, kind)


class CutTests(unittest.TestCase):
    """Rational cut points: compared with the samples directly, merged with the real roots of the master polynomial
    into one set of points, and kept out of it."""

    CUTS = (Rational(-7, 3), Rational(1))  # With x^2 - 2 as the master polynomial, the points of MASTER.

    def rejected(self, master, samples, cuts, label=None):
        self.assertIs(cells.check_samples(master, samples, cuts), False, label)
        with self.assertRaises(cells.Rejected, msg=label):
            cells.Cells(master, samples, cuts)

    def test_cut_points_and_roots_make_one_set_of_points(self):
        grid = cells.Cells((-2, 0, 1), SAMPLES, self.CUTS)
        self.assertEqual((grid.master, grid.cuts, grid.samples, grid.root_count), ((-2, 0, 1), self.CUTS, SAMPLES, 4))
        with self.assertRaises(AttributeError):
            grid.cuts = ()
        self.assertEqual([grid.locate(q) for q in self.CUTS], [1, 3])
        # Cut points alone, the master a constant; and nothing at all, one cell.
        self.assertEqual(cells.Cells((1,), (Rational(-3), Rational(0), Rational(2)), self.CUTS).root_count, 2)
        self.assertEqual(cells.Cells((-5,), (Rational(9),), ()).root_count, 0)
        # A cut point at a root of the master is that root, one point: (x - 1)(x^2 - 2) with the cut point 1, and
        # MASTER with both of its rational roots as cut points.
        three = (Rational(-2), Rational(0), Rational(5, 4), Rational(2))
        self.assertEqual(cells.Cells(integers((X - 1) * (X**2 - 2)), three, (Rational(1),)).root_count, 3)
        both = cells.Cells(MASTER, SAMPLES, self.CUTS)
        self.assertEqual((both.root_count, [both.locate(q) for q in self.CUTS]), (4, [1, 3]))
        self.assertIs(cells.check_samples(MASTER, SAMPLES, (Rational(1),)), True)
        self.assertIs(cells.check_samples(MASTER, SAMPLES, (Rational(-7, 3), Rational(1), Rational(3))), False)

    def test_bad_cut_points_are_rejected(self):
        master = (-2, 0, 1)
        for label, cuts in {
            "unordered": (Rational(1), Rational(-7, 3)),
            "repeated": (Rational(-7, 3), Rational(-7, 3)),
            "a list": [Rational(-7, 3), Rational(1)],
            "None": None,
            "a Rational": Rational(1),
            "an int": (Rational(-7, 3), 1),
            "a float": (Rational(-7, 3), 1.0),
            "a bool": (Rational(-7, 3), True),
            "not in lowest terms": (trusted_rational(-14, 6), Rational(1)),
            "a zero denominator": (Rational(-7, 3), trusted_rational(1, 0)),
            "one missing": (Rational(1),),
            "one too many": (Rational(-7, 3), Rational(1), Rational(3)),
            "one below the first sample": (Rational(-4), Rational(1)),
            "one above the last sample": (Rational(-7, 3), Rational(3)),
            "two between the same samples": (Rational(-7, 3), Rational(1, 4), Rational(1)),
            "one at a sample": (Rational(-7, 3), Rational(5, 4)),
        }.items():
            with self.subTest(label):
                self.rejected(master, SAMPLES, cuts, label)
        a, b, c, d, e = SAMPLES
        for label, samples in {
            "a sample at a cut point": (a, b, c, Rational(1), e),
            "the first sample at a cut point": (Rational(-7, 3), b, c, d, e),
            "two samples between the same points": (a, b, Rational(1, 2), Rational(3, 4), e),
            "a sample that skips a cut point": (a, b, Rational(9, 8), d, e),
            "a sample that skips a root": (a, b, c, Rational(3, 2), Rational(5, 2)),
        }.items():
            with self.subTest(label):
                self.rejected(master, samples, self.CUTS, label)

    def test_acceptance_is_the_stated_criterion_with_cut_points(self):
        """Cells accepts exactly the samples and cut points of item 1 of the module docstring, stated with
        certify.upoly.count_roots and comparisons: cut points increasing; one sample more than points; samples
        increasing, none a point; the k-th with k points below it."""
        def criterion(master, samples, cuts):
            part = certify.squarefree_part(master)

            def root(q):
                return len(part) > 1 and tp.sign_at(part, q.numerator, q.denominator) == 0

            def below(q):
                return certify.count_roots(part, None, q) if len(part) > 1 else 0

            new = [q for q in cuts if not root(q)]
            total = certify.count_roots(part, None, None) if len(part) > 1 else 0
            return (all(p < q for p, q in zip(cuts, cuts[1:])) and len(samples) == total + len(new) + 1
                    and all(p < q for p, q in zip(samples, samples[1:]))
                    and all(not root(s) and s not in cuts and below(s) + sum(q < s for q in new) == k
                            for k, s in enumerate(samples)))

        def rational(rng):
            return Rational(rng.randint(-40, 40), rng.randint(1, 8))

        rng = random.Random(1212)
        verdicts = Counter()
        for _ in range(150):
            master = integers(random_expression(rng))
            roots = rational_roots(master)
            chosen = set(rng.sample(roots, rng.randint(0, len(roots))))
            cuts = tuple(sorted(chosen | {rational(rng) for _ in range(rng.randint(0, 3))}))
            good = samples_between(cut_points(master, cuts))
            self.assertIs(cells.check_samples(master, good, cuts), True)
            for _ in range(6):
                samples, changed = list(good), list(cuts)
                change = rng.randrange(7)
                if change == 0:
                    samples[rng.randrange(len(samples))] = rational(rng)
                elif change == 1:
                    del samples[rng.randrange(len(samples))]
                elif change == 2:
                    samples.append(rational(rng))
                elif change == 3 and changed:
                    del changed[rng.randrange(len(changed))]
                elif change == 4:
                    changed.append(rng.choice(samples) if rng.randrange(2) else rational(rng))
                elif change == 5 and roots:
                    changed.append(rng.choice(roots))  # A root of the master, which may be a cut point already.
                else:  # Another point of the same cell, as often as not.
                    position = rng.randrange(len(samples))
                    samples[position] = samples[position] + Rational(rng.randint(-3, 3), 16)
                if rng.randrange(5):
                    samples.sort()
                if rng.randrange(5):
                    changed.sort()
                samples, changed = tuple(samples), tuple(changed)
                expected = criterion(master, samples, changed)
                self.assertIs(cells.check_samples(master, samples, changed), expected, (master, samples, changed))
                verdicts[expected] += 1
        self.assertGreater(verdicts[True], 80, verdicts)
        self.assertGreater(verdicts[False], 400, verdicts)


class CutSignTests(unittest.TestCase):
    """Signs, zeros and endpoints at cut points: zeros there are found by exact evaluation and divided out, so that
    the rest of a polynomial alone must divide the master polynomial."""

    LOW, HIGH = RealAlgebraicRoot((-2, 0, 1), 0), RealAlgebraicRoot((-2, 0, 1), 1)
    POINTS = (Rational(-7, 3), LOW, Rational(1), HIGH)

    @classmethod
    def setUpClass(cls):
        # The points -7/3, -sqrt 2, 1 and sqrt 2 of MASTER again, the rational ones as cut points: the master
        # polynomial holds x^2 - 2 and the rootless x^2 + 1.
        cls.grid = cells.Cells(cells.master_polynomial((-2, 0, 1), (1, 0, 1)), SAMPLES, (Rational(-7, 3), Rational(1)))

    def test_signs_of_factors_and_products(self):
        grid = self.grid
        self.assertEqual((grid.master, grid.root_count), (integers((X**2 - 2) * (X**2 + 1)), 4))
        for expression, signs in (
                (X - 1, (-1, -1, 0, 1)), (X**2 - 2, (1, 0, -1, 0)), (3 * X + 7, (0, 1, 1, 1)), (X**2 + 1, (1, 1, 1, 1)),
                ((3 * X + 7) * (X - 1), (0, -1, 0, 1)), ((X - 1)**3 * (3 * X + 7)**2, (0, -1, 0, 1)),
                (-(X - 1)**2 * (X**2 - 2)**3 * (X**2 + 1), (-1, 0, 0, 0)),
                ((X**2 - 2) * (X - 1) * (3 * X + 7), (0, 0, 0, 0)), (Rational(-5) + 0 * X, (-1, -1, -1, -1))):
            polynomial = integers(expression)
            with self.subTest(expression=str(expression)):
                self.assertEqual(grid.root_signs(polynomial), signs)
                source = Polynomial.from_expression(expression, X)
                self.assertEqual(signs, tuple(sign_at(source, point) for point in self.POINTS))
        self.assertEqual(grid.root_signs(()), (0, 0, 0, 0))

    def test_a_polynomial_with_a_root_that_is_no_point_has_no_sign_at_the_points(self):
        for expression in (X - 5, X**2 - 3, (X - 1) * (X - 5), X, 2 * X - 1, (3 * X + 8) * (X - 1),
                           (X - 1) * (X**2 + 2), (X**2 - 2)**2 * (X - 5)):
            polynomial = integers(expression)
            with self.subTest(expression=str(expression)):
                with self.assertRaises(cells.Rejected):
                    self.grid.root_signs(polynomial)
                self.assertEqual(self.grid.sample_signs(polynomial),
                                 tuple(tp.sign_at(polynomial, s.numerator, s.denominator) for s in SAMPLES))

    def test_a_rational_root_of_the_master_that_is_no_cut_point(self):
        # 1/2 a root of the master polynomial, 1 a cut point: the points -sqrt 2, 1/2, 1 and sqrt 2.
        grid, points = cut_with(integers((2 * X - 1) * (X**2 - 2)), (Rational(1),))
        self.assertEqual(grid.root_count, 4)
        self.assertEqual([grid.locate(point) for point in points], [1, 2, 3, 4])
        self.assertEqual(grid.locate(Rational(1, 2)), 2)
        self.assertEqual(grid.root_signs(integers(2 * X - 1)), (-1, 0, 1, 1))
        self.assertEqual(grid.root_signs(integers((2 * X - 1) * (X - 1))), (1, 0, 0, 1))
        self.assertEqual(grid.root_signs(integers(X - 1)), (-1, -1, 0, 1))
        for value in (Rational(3, 4), Rational(0), Rational(2)):
            with self.subTest(value=value), self.assertRaises(cells.Rejected):
                grid.locate(value)

    def test_endpoints_are_located_among_cut_points_and_roots(self):
        grid = self.grid
        self.assertEqual([grid.locate(point) for point in self.POINTS], [1, 2, 3, 4])
        # Names whose polynomials vanish at cut points: linear ones, and products with the master's factors.
        self.assertEqual(grid.locate(forged_root((7, 3), 0)), 1)
        self.assertEqual(grid.locate(RealAlgebraicRoot((-1, 1), 0)), 3)
        cubic = integers((X - 1) * (X**2 - 2))
        self.assertEqual([grid.locate(RealAlgebraicRoot(cubic, index)) for index in range(3)], [2, 3, 4])
        both = integers((3 * X + 7) * (X - 1))
        self.assertEqual([grid.locate(RealAlgebraicRoot(both, index)) for index in range(2)], [1, 3])
        everything = integers((3 * X + 7) * (X - 1) * (X**2 - 2) * (X**2 + 1))
        self.assertEqual([grid.locate(forged_root(everything, index)) for index in range(4)], [1, 2, 3, 4])
        self.assertEqual([grid.locate(forged_root(grid.master, index)) for index in range(2)], [2, 4])

    def test_values_that_are_no_point_are_not_located(self):
        grid = self.grid
        everything = integers((3 * X + 7) * (X - 1) * (X**2 - 2) * (X**2 + 1))
        for label, endpoint in {
            "a rational that is no point": Rational(0),
            "a sample": SAMPLES[3],
            "a rational root of a factor outside the cut": Rational(5),
            "a double zero at a cut point": forged_root(integers((X - 1)**2), 0),
            "a double zero at a cut point times a factor of the master": forged_root(integers((X - 1)**2 * (X**2 - 2)),
                                                                                   1),
            "a zero at a cut point times a factor outside the cut": forged_root(integers((X - 1) * (X - 5)), 0),
            "a root of another polynomial": RealAlgebraicRoot((-3, 0, 1), 1),
            "an index past the real roots": forged_root(everything, 4),
            "a polynomial longer than the master and the cut points": forged_root(tp.mul(everything, (-9, 1)), 0),
        }.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                grid.locate(endpoint)

    def test_claims_with_cut_points(self):
        grid = self.grid
        # (-inf, -7/3] with (-sqrt 2, 1] and the point sqrt 2.
        truth = ((T, F, T, T), (T, F, T, F, F))
        expected = ((None, 1, F, T), (2, 3, F, T), (4, 4, T, T))
        cubic = integers((X - 1) * (X**2 - 2))
        claims = (
            IntervalSet((Interval(None, Rational(-7, 3), False, True), Interval(self.LOW, Rational(1), False, True),
                         Interval(self.HIGH, self.HIGH, True, True))),
            forged_set((forged_interval(None, forged_root((7, 3), 0), False, True),
                        forged_interval(forged_root(grid.master, 0), RealAlgebraicRoot((-1, 1), 0), False, True),
                        forged_interval(self.HIGH, forged_root(cubic, 2), True, True))),
        )
        self.assertEqual(grid.reconstruct(*truth), expected)
        for claim in claims:
            self.assertEqual(grid.index_form(claim), expected)
            self.assertEqual(grid.membership(claim), truth)
            self.assertIs(grid.matches(claim, *truth), True)
            self.assertIs(grid.matches(claim, (T, F, F, T), (T, F, T, F, F)), False)


class SplitTests(unittest.TestCase):
    """Rational roots split off as cut points, cut points merged, and the sign-preserving integer form."""

    def test_rational_roots_are_split_off(self):
        expression = (2 * X - 1)**2 * (X + 3) * (X**2 - 2)
        p = integers(expression)
        self.assertEqual(cells.split_rational_roots(p, (Rational(-3), Rational(1, 2))),
                         ((Rational(-3), Rational(1, 2)), (-2, 0, 1)))
        self.assertEqual(cells.split_rational_roots(p, (Rational(-3),)),
                         ((Rational(-3),), integers((2 * X - 1)**2 * (X**2 - 2))))
        self.assertEqual(cells.split_rational_roots(p, ()), ((), p))
        # The rest is the exact quotient: what is left keeps its content and its leading sign.
        q = integers(-6 * (X - 1) * (X**2 + 1))
        self.assertEqual(cells.split_rational_roots(q, (Rational(1),)), ((Rational(1),), integers(-6 * (X**2 + 1))))
        self.assertEqual(cells.split_rational_roots((5,), ()), ((), (5,)))
        self.assertEqual(cells.split_rational_roots((), ()), ((), ()))
        # With the master polynomial of the rest and the roots as cut points, the polynomial has its signs at every
        # point, and its multiple roots at cut points are zeros like any other.
        roots, rest = cells.split_rational_roots(p, (Rational(-3), Rational(1, 2)))
        grid, points = cut_with(cells.master_polynomial(rest), roots)
        source = Polynomial.from_expression(expression, X)
        self.assertEqual(grid.root_signs(p), tuple(sign_at(source, point) for point in points))
        self.assertEqual(grid.sample_signs(p), tuple(sign(source.evaluate(sample)) for sample in grid.samples))
        # p vanishes at -3, -sqrt 2, 1/2 and sqrt 2; it is negative below -3 and between -sqrt 2 and sqrt 2, where
        # x^2 - 2 < 0, and positive elsewhere.
        self.assertEqual((grid.root_signs(p), grid.sample_signs(p)), ((0, 0, 0, 0), (-1, 1, -1, -1, 1)))

    def test_bad_candidates_are_rejected(self):
        p = integers((2 * X - 1) * (X + 3) * (X**2 - 2))
        for label, (polynomial, candidates) in {
            "a candidate that is no root": (p, (Rational(-3), Rational(2))),
            "a rational near an irrational root": (p, (Rational(141, 100),)),
            "unordered": (p, (Rational(1, 2), Rational(-3))),
            "repeated": (p, (Rational(-3), Rational(-3))),
            "more candidates than the degree": (integers(X - 1), (Rational(1), Rational(2))),
            "a candidate for a constant": ((5,), (Rational(0),)),
            "a candidate for the zero polynomial": ((), (Rational(0),)),
            "an int": (p, (-3,)),
            "a float": (p, (0.5,)),
            "a bool": (p, (False,)),
            "not in lowest terms": (p, (trusted_rational(2, 4),)),
            "a list": (p, [Rational(-3)]),
            "None": (p, None),
            "a list for the polynomial": ([1, 1], ()),
            "a trailing zero": ((1, 1, 0), ()),
            "a float coefficient": ((1.0, 1), ()),
        }.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                cells.split_rational_roots(polynomial, candidates)

    def test_cut_points_are_merged(self):
        self.assertEqual(cells.merge_cuts((Rational(1), Rational(-1)), (), (Rational(1, 2), Rational(1))),
                         (Rational(-1), Rational(1, 2), Rational(1)))
        self.assertEqual(cells.merge_cuts(), ())
        for label, groups in {"a list": ([Rational(1)],), "an int": ((1,),), "a float": ((0.5,),), "None": (None,),
                              "not in lowest terms": ((trusted_rational(2, 4),),),
                              "a Rational": (Rational(1),)}.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                cells.merge_cuts(*groups)
        rng = random.Random(7)
        for _ in range(60):
            groups = [tuple(Rational(rng.randint(-50, 50), rng.randint(1, 12)) for _ in range(rng.randint(0, 20)))
                      for _ in range(rng.randint(0, 4))]
            self.assertEqual(cells.merge_cuts(*groups), tuple(sorted(set(itertools.chain(*groups)))))

    def test_the_integer_form_keeps_the_sign(self):
        self.assertEqual(cells.integer_polynomial((Rational(2), Rational(0), Rational(-2))), (1, 0, -1))
        self.assertEqual(tp.primitive((2, 0, -2)), (-1, 0, 1))  # The normalization of master polynomials reverses it.
        self.assertEqual(cells.integer_polynomial((Rational(1, 2), Rational(-1, 3))), (3, -2))
        self.assertEqual(cells.integer_polynomial((Rational(-3, 4), 6, Rational(0))), (-1, 8))
        self.assertEqual(cells.integer_polynomial((-4, 0, -6)), (-2, 0, -3))
        self.assertEqual(cells.integer_polynomial((0, Rational(0))), ())
        self.assertEqual(cells.integer_polynomial(()), ())
        for label, coefficients in {
            "a list": [Rational(1)], "None": None, "a float": (Rational(1), 0.5), "a bool": (True,), "a str": ("1",),
            "not in lowest terms": (trusted_rational(2, 4),), "a long int": (1 << 70_000,),
        }.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                cells.integer_polynomial(coefficients)
        rng = random.Random(55)
        for _ in range(200):
            coefficients = tuple(Rational(rng.randint(-9, 9), rng.randint(1, 9)) for _ in range(rng.randint(1, 6)))
            form = cells.integer_polynomial(coefficients)
            for _ in range(5):
                point = Rational(rng.randint(-30, 30), rng.randint(1, 5))
                value = sum((c * point**power for power, c in enumerate(coefficients)), Rational(0))
                self.assertEqual(tp.sign_at(form, point.numerator, point.denominator), sign(value),
                                 (coefficients, point))


class TruthTests(unittest.TestCase):
    """Relations decided by the signs of several polynomials at once (Cells.truth)."""

    @staticmethod
    def sqrt_sign(signs):
        """The sign of A sqrt(P) + B from the signs of P, A, B and N = A^2 P - B^2 (catalogue 2.3); None outside
        the domain P >= 0."""
        sp, sa, sb, sn = signs
        if sp < 0:
            return None
        t = sa if sp > 0 else 0
        if t == 0:
            return sb
        if sb == 0 or sb == t:
            return t
        return t * sn

    def test_a_square_root_equation(self):
        """sqrt(x + 2) = x holds at 2 alone: -1 solves the squared equation x + 2 = x^2 but not the equation. Every
        critical point is rational (-2, -1, 0 and 2), so they are all cut points and the master is a constant."""
        P, A, B, N = (2, 1), (1,), (0, -1), (2, 1, -1)
        master, cuts = split_cut(P, A, B, N)
        self.assertEqual((master, cuts), ((1,), tuple(map(Rational, (-2, -1, 0, 2)))))
        grid, _ = cut_with(master, cuts)

        def truth(relation):
            return grid.truth(lambda signs: self.sqrt_sign(signs) is not None
                              and cells.holds(relation, self.sqrt_sign(signs)), P, A, B, N)

        two, minus_one = Rational(2), Rational(-1)
        self.assertIs(grid.matches(IntervalSet((Interval(two, two, True, True),)), *truth("=")), True)
        self.assertIs(grid.matches(IntervalSet((Interval(minus_one, minus_one, True, True),
                                                Interval(two, two, True, True))), *truth("=")), False)
        self.assertIs(grid.matches(IntervalSet((Interval(Rational(-2), two, True, False),)), *truth(">")), True)
        self.assertIs(grid.matches(IntervalSet((Interval(two, None),)), *truth("<")), True)
        self.assertIs(grid.matches(IntervalSet((Interval(two, None, True, False),)), *truth("<=")), True)

    def test_one_polynomial_gives_relation_truth(self):
        grid = cells.Cells(MASTER, SAMPLES, (Rational(1),))
        for polynomial in (MASTER, integers(X - 1), integers(X**2 - 2), (-3,), ()):
            for relation in cells.RELATIONS:
                with self.subTest(polynomial=polynomial, relation=relation):
                    self.assertEqual(grid.truth(lambda signs: cells.holds(relation, signs[0]), polynomial),
                                     grid.relation_truth(polynomial, relation))

    def test_no_polynomial_and_bad_decisions(self):
        grid = cells.Cells(MASTER, SAMPLES)
        self.assertEqual(grid.truth(lambda signs: signs == ()), ((T,) * 4, (T,) * 5))
        linear = integers(X - 1)
        for label, decide in {"an int": lambda signs: 1, "None": lambda signs: None, "a str": lambda signs: "yes",
                              "a tuple": lambda signs: signs}.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                grid.truth(decide, linear)
        for label, (decide, polynomials) in {
            "no function": (None, (linear,)),
            "a polynomial with a root that is no point": (lambda signs: True, (integers(X - 5),)),
            "a list for a polynomial": (lambda signs: True, ([-1, 1],)),
        }.items():
            with self.subTest(label), self.assertRaises(cells.Rejected):
                grid.truth(decide, *polynomials)


class ReviewFindingTests(unittest.TestCase):
    """Regression tests for the review of the first cell engine: each of them failed on it."""

    def test_a_square_free_master_gets_no_second_square_free_pass(self):
        """Cells takes the square-free part of its master polynomial from the samples, or from the last member of
        its Sturm chain: it no longer runs Euclid's square-free pass, which master_polynomial has already run."""
        with mock.patch.object(certify, "squarefree_part", side_effect=AssertionError("a second square-free pass")):
            self.assertEqual(cells.Cells(MASTER, SAMPLES).root_count, 4)  # All roots real: no chain either.
            self.assertEqual(cells.Cells(integers((X**2 + 1) * (X - 1)), (Rational(0), Rational(2))).root_count, 1)
            grid = cells.Cells(integers(-2 * (X - 1)**2 * (X + 2)**3), (Rational(-3), Rational(0), Rational(2)))
            self.assertEqual((grid.master, grid.root_count), ((-2, 1, 1), 2))  # The chain made it square-free.

    def test_evaluations_are_charged_by_the_size_of_the_coefficients(self):
        """An exact evaluation costs with the length of the coefficients as well as with that of the point: 81
        coefficients of 5,000 bits at a 62-bit sample and at two small ones. Charged by the point alone, as before,
        this was 324 units for about half a millisecond, six times the budget's calibration of about a quarter of a
        microsecond per unit."""
        big = tuple((-1) ** k * ((1 << 5000) + k) for k in range(80)) + (1,)
        grid = cells.Cells((-1, 0, 1), (Rational(-2), Rational((1 << 60) + 1, (1 << 61) + 3), Rational(2)))
        with computation(fresh=True) as budget:
            grid.sample_signs(big)
        self.assertGreater(budget.work, 1_500)

    def test_no_sturm_chain_is_evaluated_at_the_samples(self):
        """The samples of a master polynomial whose roots are all real are verified by its signs alone, and those of
        any other by its signs and the variations of its Sturm chain at infinity. Before, every member of the chain
        was evaluated at every sample: Cells on Chebyshev's T_80 with the producer's samples took 863,879 units,
        against 6,885 now."""
        master = chebyshev(12)
        points = roots_of(master)
        with mock.patch.object(certify, "sturm_chain", side_effect=AssertionError("a Sturm chain")):
            self.assertEqual(cells.Cells(master, samples_between(points)).root_count, 12)
        original = certify.sturm_chain
        chains = []
        with mock.patch.object(certify, "sturm_chain", side_effect=lambda p: chains.append(p) or original(p)):
            with mock.patch.object(cells.tp, "sign_at", side_effect=self.no_long_evaluation(cells.tp.sign_at)):
                with_complex = integers((X**2 + 1) * X * (X**2 - 2))
                self.assertEqual(cells.Cells(with_complex, samples_between(roots_of(with_complex))).root_count, 3)
        self.assertEqual(chains, [integers((X**2 + 1) * X * (X**2 - 2))])

    @staticmethod
    def no_long_evaluation(sign_at):
        """sign_at, refusing every polynomial of degree below 5: only the degree 5 polynomial itself may be evaluated,
        never its degree 4 derivative or a lower member of its Sturm chain."""
        def checked(polynomial, numerator, denominator):
            if len(polynomial) < 6:
                raise AssertionError("A member of a Sturm chain was evaluated")
            return sign_at(polynomial, numerator, denominator)
        return checked

    def test_the_module_docstring_says_what_endpoint_cut_does_not_bound(self):
        """It said that the lengths of claims are bounded before any work, while endpoint_cut (then
        endpoint_polynomial) bounds neither the number of intervals nor the length of an endpoint's polynomial."""
        text = " ".join(cells.__doc__.split())
        self.assertNotIn("The lengths and the integers of evidence and claims are bounded before any work on them",
                         text)
        self.assertIn("``endpoint_cut`` bounds neither the number of intervals it reads nor the length of an "
                      "endpoint's polynomial", text)

    def test_a_foreign_name_is_located_once_the_claim_joins_the_cut(self):
        """|x^2 - 2| >= 1 holds on (-inf, -sqrt 3] with [-1, 1] and [sqrt 3, inf): the pieces x^2 - 3 and 1 - x^2
        and the atom x^2 - 2 make the problem's cut, the rational roots -1 and 1 as cut points. sqrt 3 named as a
        root of (x^2 - 3)(x^2 + 1) is located only once the claim's endpoints join the cut, as the module docstring
        recommends to equation checkers; the name of the class, a root of x^2 - 3, is located either way."""
        atom, outer, inner = integers(X**2 - 2), integers(X**2 - 3), integers(1 - X**2)

        def decide(signs):  # |P| >= 1: P - 1 >= 0 where P >= 0, -P - 1 >= 0 where P < 0.
            return signs[1] >= 0 if signs[0] >= 0 else signs[2] >= 0

        foreign = integers((X**2 - 3) * (X**2 + 1))
        problem = split_cut(atom, outer, inner)
        for low, high, located_alone in ((RealAlgebraicRoot((-3, 0, 1), 0), RealAlgebraicRoot((-3, 0, 1), 1), True),
                                         (RealAlgebraicRoot(foreign, 0), RealAlgebraicRoot(foreign, 1), False)):
            claim = IntervalSet((Interval(None, low, False, True), Interval(Rational(-1), Rational(1), True, True),
                                 Interval(high, None, True, False)))
            alone, _ = cut_with(*problem)
            self.assertIs(alone.matches(claim, *alone.truth(decide, atom, outer, inner)), located_alone)
            claim_master, claim_cuts = cells.endpoint_cut(claim)
            joined, _ = cut_with(cells.master_polynomial(problem[0], claim_master),
                                 cells.merge_cuts(problem[1], claim_cuts))
            self.assertIs(joined.matches(claim, *joined.truth(decide, atom, outer, inner)), True)
            wrong = IntervalSet((Interval(None, low, False, True), Interval(high, None, True, False)))
            self.assertIs(joined.matches(wrong, *joined.truth(decide, atom, outer, inner)), False)

    def test_signs_come_from_the_sign_preserving_integer_form(self):
        """|x - 1| |x + 1| <= x^2 - 1 holds outside (-1, 1), where both sides agree; inside, the piece
        |x - 1| |x + 1| - x^2 + 1 is 2 - 2x^2 > 0. Made primitive as master polynomials are, that piece reverses
        its sign, and the whole line would be accepted."""
        grid = cells.Cells((1,), (Rational(-2), Rational(0), Rational(2)), (Rational(-1), Rational(1)))
        line = IntervalSet((Interval(None, None),))
        true = IntervalSet((Interval(None, Rational(-1), False, True), Interval(Rational(1), None, True, False)))

        def verdicts(piece):
            pieces = ((), piece, ())  # The pieces on the cells below -1, between -1 and 1, and above 1.
            cell_truth = tuple(cells.holds("<=", grid.sign_at_sample(pieces[k], k)) for k in range(3))
            root_truth = tuple(cells.holds("<=", grid.sign_at_root(pieces[k - 1], k)) for k in (1, 2))
            return grid.matches(true, root_truth, cell_truth), grid.matches(line, root_truth, cell_truth)

        inside = (Rational(2), Rational(0), Rational(-2))
        self.assertEqual(verdicts(cells.integer_polynomial(inside)), (True, False))
        self.assertEqual(verdicts(tp.primitive((2, 0, -2))), (False, True))  # The trap integer_polynomial avoids.


def absolute_sum_sublevel(values, c):
    """{x : sum of |x - a| over the values <= c}, for distinct rationals, by the convexity of the sum: a closed
    interval, a point or the empty set. The sum is linear between neighbouring values and has slope -m below them
    and m above them, m being their number."""
    values = sorted(values)
    m = len(values)

    def excess(x):
        return sum(((x - a) if x >= a else (a - x) for a in values), Rational(0)) - c

    at = [excess(a) for a in values]
    inside = [j for j in range(m) if at[j] <= 0]
    if not inside:
        return IntervalSet(())
    first, last = inside[0], inside[-1]

    def crossing(p, at_p, q, at_q):  # The zero of the linear piece through (p, at_p <= 0) and (q, at_q > 0).
        return p + (q - p) * (-at_p) / (at_q - at_p)

    lower = values[0] + at[0] / m if first == 0 else crossing(values[first], at[first], values[first - 1],
                                                               at[first - 1])
    upper = values[-1] - at[-1] / m if last == m - 1 else crossing(values[last], at[last], values[last + 1],
                                                                    at[last + 1])
    return IntervalSet((Interval(lower, upper, True, True),))


class CapacityTests(unittest.TestCase):
    """Ordinary inputs with many rational points fit the default budget (the review's major finding): rational
    endpoints and rational roots are cut points, no polynomial of them is formed, and a master polynomial whose roots
    are all real needs no Sturm chain. The bounds below are five to six times the work measured: at most 2,668 units
    for a set operation, 8,438 for the sum of absolute values and 13,156 for the endpoints at roots of T_60."""

    def test_set_operations_on_many_rational_endpoints(self):
        """The review's reproduction, two sets of 12 intervals [k/100, l/100) whose union needed about 4.6 million
        units, and two sets of 24 intervals each with endpoints k/100 (|k| <= 5000) and with random 16-bit
        rationals."""
        rng = random.Random(97531)

        def decimal_set(n):
            points = sorted({Rational(rng.randint(-5000, 5000), 100) for _ in range(2 * n)})
            return IntervalSet(tuple(Interval(points[2 * i], points[2 * i + 1], True, False)
                                     for i in range(len(points) // 2)))

        def random_set(n):
            points = set()
            while len(points) < 2 * n:
                points.add(Rational(rng.getrandbits(16) - (1 << 15), rng.getrandbits(16) | 1))
            points = sorted(points)
            return IntervalSet(tuple(Interval(points[2 * i], points[2 * i + 1], rng.random() < 0.5,
                                              rng.random() < 0.5) for i in range(n)))

        operations = {"union": (lambda x, y: x or y, IntervalSet.union),
                      "intersection": (lambda x, y: x and y, IntervalSet.intersection),
                      "difference": (lambda x, y: x and not y, lambda a, b: a.intersection(complement(b)))}
        pairs = [(decimal_set(12), decimal_set(12)), (decimal_set(24), decimal_set(24)),
                 (random_set(24), random_set(24))]
        for label, (a, b) in zip(("12 decimal intervals", "24 decimal intervals", "24 random 16-bit intervals"), pairs):
            for name, (combine, produce) in operations.items():
                claim = produce(a, b)
                wrong = complement(claim)
                with self.subTest(label, operation=name):
                    samples = samples_between(cut_points(*cells.endpoint_cut(a, b, claim)))
                    verdict, work, seconds = measured(lambda: set_check(combine, claim, (a, b), samples))
                    self.assertIs(verdict, True)
                    self.assertLess(work, 15_000)
                    self.assertLess(seconds, SECONDS)
                    samples = samples_between(cut_points(*cells.endpoint_cut(a, b, wrong)))
                    self.assertIs(measured(lambda: set_check(combine, wrong, (a, b), samples))[0], False)

    def test_a_sum_of_absolute_values(self):
        """sum_(i=1..20) |x - a_i| <= c with a_i = k/100: the atoms' roots and the roots of the linear pieces are
        cut points and the master polynomial is a constant, as the checker of solve_abs would build them. The claim
        comes from the convexity of the sum, independently of the engine."""
        rng = random.Random(4242)
        values = sorted({Rational(rng.randint(-5000, 5000), 100) for _ in range(20)})
        middle = values[len(values) // 2]
        c = sum(((middle - a) if middle >= a else (a - middle) for a in values), Rational(0)) + Rational(3_217, 100)
        atoms = tuple((-a.numerator, a.denominator) for a in values)  # den * x - num, so |x - a| = |atom| / den.

        def piece(pattern):  # sum of s_i (x - a_i) - c on the cells where the atoms have the signs s_i.
            slope = sum(pattern)
            return cells.integer_polynomial((-sum((s * a for s, a in zip(pattern, values)), Rational(0)) - c,
                                             Rational(slope)))

        def evidence(claim):
            """(samples of the atoms' cut, roots of each piece, samples of the whole cut): the producer's side."""
            cuts0 = tuple(values)
            samples0 = samples_between(cuts0)
            patterns = [tuple(tp.sign_at(atom, s.numerator, s.denominator) for atom in atoms) for s in samples0]
            roots = tuple(rational_roots(piece(pattern)) for pattern in patterns)
            claim_master, claim_cuts = cells.endpoint_cut(claim)
            cuts = cells.merge_cuts(cuts0, *roots, claim_cuts)
            return samples0, roots, samples_between(cut_points(claim_master, cuts))

        def check(claim, samples0, piece_roots, samples):
            """The solve_abs checker of catalogue 2.3 on this problem, with cut points."""
            try:
                splits = [cells.split_rational_roots(atom, (a,)) for atom, a in zip(atoms, values)]
                master0 = cells.master_polynomial(*(rest for _, rest in splits))
                cuts0 = cells.merge_cuts(*(roots for roots, _ in splits))
                grid0 = cells.Cells(master0, samples0, cuts0)
                patterns = list(zip(*(grid0.sample_signs(atom) for atom in atoms)))
                pieces = {pattern: piece(pattern) for pattern in patterns}
                piece_splits = [cells.split_rational_roots(pieces[pattern], roots)
                                for pattern, roots in zip(patterns, piece_roots)]
                claim_master, claim_cuts = cells.endpoint_cut(claim)
                master = cells.master_polynomial(master0, *(rest for _, rest in piece_splits), claim_master)
                cuts = cells.merge_cuts(cuts0, *(roots for roots, _ in piece_splits), claim_cuts)
                grid = cells.Cells(master, samples, cuts)
                cell_patterns = list(zip(*(grid.sample_signs(atom) for atom in atoms)))
                cell_truth = tuple(cells.holds("<=", grid.sign_at_sample(pieces[pattern], k))
                                   for k, pattern in enumerate(cell_patterns))
                root_truth = []
                for k in range(1, grid.root_count + 1):
                    left = grid.sign_at_root(pieces[cell_patterns[k - 1]], k)
                    if left != grid.sign_at_root(pieces[cell_patterns[k]], k):
                        return False
                    root_truth.append(cells.holds("<=", left))
                return grid.matches(claim, tuple(root_truth), cell_truth)
            except cells.Rejected:
                return False

        claim = absolute_sum_sublevel(values, c)
        self.assertEqual(len(claim.intervals), 1)
        proof = evidence(claim)
        verdict, work, seconds = measured(lambda: check(claim, *proof))
        self.assertIs(verdict, True)
        self.assertLess(work, 50_000)
        self.assertLess(seconds, SECONDS)
        lower, upper = claim.intervals[0].lower, claim.intervals[0].upper
        for wrong in (IntervalSet((Interval(lower, upper, False, True),)), IntervalSet(()),
                      IntervalSet((Interval(lower, upper + Rational(1, 100), True, True),))):
            with self.subTest(wrong=wrong):
                proof = evidence(wrong)
                self.assertIs(measured(lambda: check(wrong, *proof))[0], False)

    def test_chebyshev_endpoints(self):
        """Endpoints at roots of Chebyshev's T_60, all 60 of whose roots are real: the master polynomial is T_60, it
        changes sign between the samples at every one of them, and no Sturm chain is built."""
        t = chebyshev(60)
        cuts = (Rational(0), Rational(1))
        with computation(PRODUCER, fresh=True):  # The producer's records, beyond the default budget; not measured.
            a = IntervalSet((Interval(RealAlgebraicRoot(t, 0), RealAlgebraicRoot(t, 1), True, True),))
            b = IntervalSet((Interval(Rational(0), Rational(1)),))
            union = a.union(b)
            wrong = IntervalSet((Interval(RealAlgebraicRoot(t, 0), RealAlgebraicRoot(t, 2), True, True),
                                 Interval(Rational(0), Rational(1))))
        self.assertEqual(cells.endpoint_cut(a, b, union), (t, cuts))
        self.assertEqual(cells.endpoint_cut(a, b, wrong), (t, cuts))
        # Steps of 1/1000, and 4999/5000 between the largest root, about 0.99966, and the cut point 1.
        grid = tuple(sorted({Rational(k, 1000) for k in range(-1001, 1002)} | {Rational(4999, 5000)}))
        samples = grid_samples(t, cuts, grid)
        self.assertEqual(len(samples), 63)  # 60 roots and 2 cut points.

        def either(x, y):
            return x or y

        with mock.patch.object(certify, "sturm_chain", side_effect=AssertionError("a Sturm chain")):
            verdict, work, seconds = measured(lambda: set_check(either, union, (a, b), samples))
        self.assertIs(verdict, True)
        self.assertLess(work, 70_000)
        self.assertLess(seconds, SECONDS)
        self.assertIs(set_check(either, wrong, (a, b), samples), False)


class BudgetTests(unittest.TestCase):
    """The engine works in the computation budget, and forged oversized inputs fail fast: at once, before any work on
    them, or when they exhaust the default budget (the whole of its max_work is about a second of computation)."""

    def refused(self, call, *, work=1_000, seconds=SECONDS):
        """call() answers False or raises Rejected or ResourceLimitError, within the time and, in a fresh default
        budget, within the given work. The outcome is returned: False, or the class of the exception."""
        with computation(fresh=True) as budget:
            start = time.perf_counter()
            try:
                outcome = call()
            except (cells.Rejected, ResourceLimitError) as error:
                outcome = type(error)
            elapsed = time.perf_counter() - start
            self.assertIn(outcome, (False, cells.Rejected, ResourceLimitError))
            self.assertLessEqual(budget.work, work)
        self.assertLessEqual(elapsed, seconds)
        return outcome

    def test_work_is_charged_and_an_exhausted_budget_is_not_a_verdict(self):
        master = integers(X * (2 * X - 1) * (X**2 - 2) * (X - 3))
        points = roots_of(master)
        samples = samples_between(points)
        claim = assembled((T, F, T, F, T), (F, T, F, T, F, T), points)
        with computation(fresh=True) as budget:
            grid = cells.Cells(master, samples)
            built = budget.work
            self.assertGreater(built, 0)
            self.assertIs(grid.matches(claim, (T, F, T, F, T), (F, T, F, T, F, T)), True)
            self.assertGreater(budget.work, built)
        certify._STURM.clear()
        tight = ComputationLimits(max_work=built // 2)
        with computation(tight, fresh=True), self.assertRaises(ResourceLimitError):
            cells.Cells(master, samples)
        with computation(tight, fresh=True), self.assertRaises(ResourceLimitError):
            cells.check_samples(master, samples)
        with computation(ComputationLimits(max_work=3), fresh=True), self.assertRaises(ResourceLimitError):
            grid.matches(claim, (T, F, T, F, T), (F, T, F, T, F, T))

    def test_oversized_samples(self):
        many = tuple(map(Rational, range(200_000)))
        self.assertIs(self.refused(lambda: cells.check_samples(MASTER, many), work=10), False)
        self.assertIs(self.refused(lambda: cells.Cells(MASTER, many), work=10), cells.Rejected)
        a, b, c, d, _ = SAMPLES
        long = 1 << 1_000_000
        for label, sample in (("a long numerator", Rational(long)), ("a long denominator", trusted_rational(1, long)),
                              ("both long", trusted_rational(3 * long + 1, long))):
            with self.subTest(label):
                self.assertIs(self.refused(lambda: cells.check_samples(MASTER, (a, b, c, d, sample)), work=50), False)
        # One sample more than the master's 8 real roots, increasing and of the longest integers that the limits
        # admit: the first evaluation of the master, of degree 17, is charged before it runs, and that charge alone
        # passes max_work.
        master = integers((X**2 - 2) * (X - 1) * (3 * X + 7) * (X**3 - X + 5) * (X**4 + X + 3) * (X**5 - 3 * X + 1))
        self.assertEqual(cells.Cells(master, samples_between(roots_of(master))).root_count, 8)
        admitted = tuple(Rational(k * (1 << 65_000) + 1, 1 << 64_000) for k in range(-4, 5))
        self.assertLessEqual(max(max(s.numerator.bit_length(), s.denominator.bit_length()) for s in admitted),
                             ComputationLimits().max_integer_bits)
        self.assertIs(self.refused(lambda: cells.check_samples(master, admitted), work=2_000_000), ResourceLimitError)
        # Too few of them is rejected without an evaluation.
        self.assertIs(self.refused(lambda: cells.check_samples(master, admitted[1:]), work=1_000_000), False)

    def test_oversized_candidates(self):
        """More candidate roots than the degree allows are refused before any of them is read."""
        many = tuple(map(Rational, range(200_000)))
        self.assertIs(self.refused(lambda: cells.split_rational_roots((-1, 1), many), work=10), cells.Rejected)
        self.assertIs(self.refused(lambda: cells.split_rational_roots(MASTER, many[:5]), work=10), cells.Rejected)

    def test_oversized_polynomials(self):
        long = (1 << 70_000) + 1
        self.assertIs(self.refused(lambda: cells.Cells((long, 1), (Rational(0), Rational(1))), work=10),
                      ResourceLimitError)
        self.assertIs(self.refused(lambda: cells.master_polynomial((-1, 1), (long, 0, 1)), work=10), ResourceLimitError)
        grid = cells.Cells(MASTER, SAMPLES)
        self.assertIs(self.refused(lambda: grid.root_signs((long, 1)), work=10), ResourceLimitError)
        self.assertIs(self.refused(lambda: grid.sample_signs((long, 1)), work=10), ResourceLimitError)
        # A master polynomial of degree 3000 with all its roots real, Chebyshev's T_3000, and 3001 samples in (-1, 1):
        # their number is within the bound, and the budget stops the evaluations of the master at the samples, each
        # charged by the length of its coefficients as well as by that of the sample.
        master = chebyshev(3000)
        self.assertEqual(chebyshev(4), integers(8 * X**4 - 8 * X**2 + 1))
        samples = tuple(Rational(k - 1500, 1500) for k in range(3001))
        self.assertIs(self.refused(lambda: cells.check_samples(master, samples), work=2_000_000, seconds=EXHAUSTED),
                      ResourceLimitError)

    def test_oversized_claims(self):
        grid, points = cut(integers(X * (X**2 - 2)))
        truth = ((T, T, T), (F, T, T, F))
        claim = assembled(*truth, points)
        self.assertIs(grid.matches(claim, *truth), True)
        point = forged_interval(Rational(0), Rational(0), True, True)
        long = 1 << 1_000_000
        oversized = {
            "200,000 intervals": forged_set((point,) * 200_000),
            "a long rational endpoint": forged_set((forged_interval(points[0], Rational(long), True, True),)),
            "a long denominator": forged_set((forged_interval(points[0], trusted_rational(1, long), True, True),)),
            "a polynomial of 200,000 coefficients": forged_set(
                (forged_interval(points[0], forged_root((1,) * 200_000, 1), True, True),)),
            "a polynomial with a long coefficient": forged_set(
                (forged_interval(points[0], forged_root((-long - 1, 0, 1), 1), True, True),)),
        }
        for label, value in oversized.items():
            with self.subTest(label):
                self.assertIs(self.refused(lambda: grid.matches(value, *truth), work=100), False)
                self.assertIs(self.refused(lambda: grid.index_form(value), work=100), cells.Rejected)
        self.assertIs(self.refused(lambda: grid.reconstruct((T,) * 200_000, (T,) * 200_001), work=10), cells.Rejected)
        self.assertIs(self.refused(lambda: grid.matches(claim, (T,) * 200_000, (T,) * 200_001), work=10), False)

    def test_oversized_sets(self):
        long = 1 << 1_000_000
        # 30,000 distinct rational endpoints are 30,000 cut points: no polynomial is formed, and their sort is charged
        # before it runs. (When every rational endpoint was a linear factor of the master polynomial, they exhausted
        # the budget.)
        points = tuple(Rational(k, 7) for k in range(-15_000, 15_000))
        distinct = forged_set(tuple(forged_interval(point, point, True, True) for point in reversed(points)))
        with computation(fresh=True) as budget:
            start = time.perf_counter()
            self.assertEqual(cells.endpoint_cut(distinct), ((1,), points))
            elapsed = time.perf_counter() - start
        self.assertLess(budget.work, 1_000_000)
        self.assertLessEqual(elapsed, EXHAUSTED)
        # 200,000 distinct ones exhaust the default budget before their sort.
        many = forged_set(tuple(forged_interval(Rational(k), Rational(k), True, True) for k in range(200_000)))
        self.assertIs(self.refused(lambda: cells.endpoint_cut(many), work=2_000_000, seconds=EXHAUSTED),
                      ResourceLimitError)
        # 3,000 distinct quadratic endpoints: the product of their polynomials exhausts the budget.
        quadratics = forged_set(tuple(forged_interval(forged_root((-k, 0, 1), 1), None) for k in range(2, 3_002)))
        self.assertIs(self.refused(lambda: cells.endpoint_cut(quadratics), work=2_000_000, seconds=EXHAUSTED),
                      ResourceLimitError)
        # The same endpoint 100,000 times gives one cut point.
        same = forged_set((forged_interval(Rational(1), Rational(1), True, True),) * 50_000)
        self.assertEqual(cells.endpoint_cut(same), ((1,), (Rational(1),)))
        for label, endpoint in (("a long numerator", Rational(long)), ("a long denominator", trusted_rational(1, long)),
                                ("a long coefficient", forged_root((-long - 1, 0, 1), 1))):
            with self.subTest(label):
                value = forged_set((forged_interval(Rational(0), endpoint),))
                self.assertIs(self.refused(lambda: cells.endpoint_cut(value), work=100), cells.Rejected)


if __name__ == "__main__":
    unittest.main()
