"""Unions, intersections, differences and complements of finite unions of real intervals (mathforge.sets), with
rational or real algebraic endpoints.

    set_union(a, b)         a | b
    set_intersection(a, b)  a & b
    set_difference(a, b)    a - b, the reals in a and not in b
    set_complement(a)       the reals outside a

Arguments and value. Every argument is a canonical mathforge.intervals.IntervalSet, as its constructor builds it:
sorted, disjoint, not mergeable intervals, each nonempty, with open infinite ends and ends that are canonical
Rationals or RealAlgebraicRoots (a primitive square-free polynomial with a positive leading coefficient and the index
of a real root). The value is the canonical IntervalSet of the result: IntervalSet(()) when it is empty and
IntervalSet((Interval(None, None),)) for the whole line, never an EmptySet or a UniversalSet. Each finite end of the
value is a record of an input, the first that names its number (a before b, in the order of the intervals, lower
before upper), so the value brings no polynomial of its own, as the checker requires.

Evidence: (samples,), a strictly increasing tuple of canonical Rationals rho_0 < ... < rho_r with which the checker
(mathforge.certify.sets.interval_algebra) certifies the cut of the inputs' ends. The r points of that cut, in
increasing order and each value once, are the Rational ends of the inputs, the roots of their linear RealAlgebraicRoot
ends and every real root of the polynomial of each other RealAlgebraicRoot end, the roots that are no end included
(an end sqrt 2 makes -sqrt 2 a point too). rho_0 lies below the first point, rho_k strictly between the points k and
k + 1, rho_r above the last; without a point the evidence is ((0,),). For set_complement the points come from a alone.
Each sample is the rational of least denominator (then of least absolute value) in its gap: the checker evaluates its
master polynomial at every sample, charged by the sample's length, and these samples cost it least, about half of
what midpoints cost.

Algorithm. One line of points, without the IntervalSet methods:

1. Validation: every record is read once, with exact type tests (InvalidInput for anything else, ResourceLimitError
   for an integer longer than max_integer_bits). The Sturm chain of each distinct polynomial of degree two or more
   proves it square-free, and its isolation (kernels.upoly.isolate) gives disjoint rational brackets of its real
   roots, one per root (a root met at a bisection point is an exact rational); an index beyond them names no real
   root (InvalidInput).
2. Separation: the exact points and the brackets are sorted, and every pair of neighbours that leaves no rational
   between them is narrowed until it does: a bracket that holds or touches an exact point is cut at it or halved,
   and two overlapping brackets are both halved; after four clashes of the same two polynomials a common root is
   looked for once, by a Sturm count of their gcd in the overlap, and merges the two names into one point. Each
   bisection is one refinement of the budget. Each point keeps its names: ("q", value) for a rational, ("root",
   polynomial, index) for a root.
3. Members: the members of the line are cell 0, point 1, cell 1, ..., point r, cell r. Each input is written as the
   truth of each member, and InvalidInput is raised unless its intervals are exactly the maximal runs of true members,
   which proves it canonical (sorted, disjoint, not mergeable, nonempty, infinite ends open, ends in order).
4. Truth: the operation is evaluated member by member (or; and; and not; not), and the value is the list of maximal
   runs of true members, written with the inputs' records (above) and passed once through the IntervalSet
   constructor, which leaves it unchanged.
5. Samples: the rational of least denominator in each gap between neighbouring points, below the first point and
   above the last: the continued fraction that the ends of the gap share, ended by the least partial quotient that
   the gap admits, computed in a loop.
6. Bounds, below.

Bounds. operations_v6.run verifies the value under the caller's limits, but a Result has to reload and verify under
the default ComputationLimits too, whatever limits made it: mf.verify(mf.loads(mf.dumps(result))). So produce()
refuses with ResourceLimitError a claim that the default limits might not take, by two tests.

- The check: an estimate of the checker's work within 4/5 of the default max_work. The estimate restates the steps
  that dominate the check (mathforge.certify.cells): the product H of the inputs' distinct polynomials of degree two
  or more, formed here as the checker forms it; the evaluations of H at the cut points and at every sample, and of
  each end polynomial at the cut points and at the samples, with the checker's charges for an evaluation; the
  division of H by each end polynomial; and, unless the points hold deg H roots of H (no root of H is non-real and
  no root is shared by two polynomials), the Sturm chain that the checker then builds, run here as the subresultant
  PRS of H and H' (and of the square-free part of H when that differs), whose pseudo-remainders the kernels charge
  as the checker's chain does; the gcds and comparisons of the Rationals and the reading of the root records, which
  the cell engine charges by the square of their length (_reading_units: long rational ends such as
  (10^10000 + k)/7 cost the checker mostly these); plus 16 units per sample, cut point and interval. The steps that
  run here use a budget of the default limits, or of what remains of the caller's when that is less, and the caller
  is charged for them. Measured against the checker (on the sets below, on 300 seeded claims, and on adversarial
  families: long rational ends, one polynomial with many roots or long coefficients, many polynomials sharing the
  factor x^2 - 2 or x^2 + 1, reducible polynomials with rational roots), the checker's work was between 0.28 and
  1.09 times the estimate, so an accepted claim needs at most about 1,750,000 of the 2,000,000 units.
- The reload: the arguments and the value, rebuilt once from their JSON documents (mathforge.serialization), within
  12/25 of the default work and refinements and 2/5 of the nodes, evidence items and decoded bytes and nodes, in a
  fresh budget for which nobody is charged, as the MCP decoder tests an interval_set argument. A Result holds the
  request twice (in itself and in its step) and the value twice (as its value and in its step); rebuilding it cost
  twice this rebuilding in work and refinements and at most 2.4 times in nodes (measured on the sets below).

Measured on 2026-11-13 with CPython on one machine, for set_union of two sets of n intervals with distinct ends (the
families of the checker's docstring), whatever the caller's limits, the largest n accepted (and, in brackets, the
largest number of intervals of a set whose complement is accepted) were: rational ends k/100, 404 (606); ends
2k + sqrt 2, 53 (80); sqrt(10^6 + k), 36 (73); sqrt(10^12 + k), 26 (48); sqrt(10^24 + k), 16 (25); sqrt(10^100 + k),
4 (6); the real cube roots of 2 + k, 9 (18); the real fifth roots of 2 + k, 7 (12); rational ends (10^10000 + k)/7,
6. The Result of each of these
reloaded and verified under the default limits, and one interval more was refused. The MCP tools decode at most 32
intervals per set.

Work. produce() runs in the active budget (operations_v6.run opens the request's) or in one of the default limits
opened for a direct call. It charges one unit for the call and one per interval and coefficient read, the gcds of
the validation, one refinement per bisection, the sorting passes over the points, the members of each input, and
each step of a sample by the length of its ends; trusted.upoly and the kernels charge their own arithmetic, Sturm
chains and isolation.
"""

from __future__ import annotations

from fractions import Fraction
from math import floor, gcd

from ...algebraic import RealAlgebraicRoot
from ...errors import InvalidInput, UnsupportedOperation
from ...intervals import Interval, IntervalSet
from ...limits import Budget, ComputationLimits, DecodeLimits, ResourceLimitError, computation, current_budget
from ...model import Rational
from ...registry import Argument, OperationSpec

# The producer rules:
# - Imports at the top come only from modules that `import mathforge` already loads; kernels are imported inside
#   produce(). Never import mathforge.certify, mathforge.lean, mathforge.mcp, mathforge._generated or a generated
#   area module (mathforge.ntheory, ...).
# - produce() validates its inputs first, with tests like `type(x) is int` (a bool, a float or a str is never an
#   exact number), and raises InvalidInput. It builds records only from validated data. For a Matrix argument it
#   calls current_budget().matrix(rows, cols) before it validates the shape.
# - The value is never None. The evidence is a tuple of ints, Rationals, tuples, str and records: no list, dict or
#   float. Under the verifier, 23 + 2 * request + 2 * value + evidence nodes stay within max_evidence_items.
# - Every operation takes at least one argument. Only an Expression or a Matrix that Result.expression should hold
#   is named `expression` or `A`.
# - Operation names are new, and nothing is superseded (no SUPERSEDES): tools/new_family.py checked these names.
# - No speed-suite fixture and no hand-written lean module: the generic lean runner serves v6 operations.

# The scope text of every operation of wave M2 (docs/v09/cards/m2-algebra-equations-sets-numeric.md).
_SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
          "and decimal digits")
_SET = "IntervalSet"  # The annotation of the interval_set kind, and the return type of every operation.

SPECS = (
    OperationSpec("set_union", "sets",
                  (Argument("a", _SET, None, "interval_set"), Argument("b", _SET, None, "interval_set")),
                  _SET, "Unite two finite unions of real intervals.", _SCOPE, version=6),
    OperationSpec("set_intersection", "sets",
                  (Argument("a", _SET, None, "interval_set"), Argument("b", _SET, None, "interval_set")),
                  _SET, "Intersect two finite unions of real intervals.", _SCOPE, version=6),
    OperationSpec("set_difference", "sets",
                  (Argument("a", _SET, None, "interval_set"), Argument("b", _SET, None, "interval_set")),
                  _SET, "Remove the second set from the first.", _SCOPE, version=6),
    OperationSpec("set_complement", "sets",
                  (Argument("a", _SET, None, "interval_set"),),
                  _SET, "Complement a finite union of real intervals in the reals.", _SCOPE, version=6),
)

# No record classes: every value is a v0.7 IntervalSet, which the registry does not let a v6 family claim as its own
# (every TYPES class must be defined in the family module); it serializes and renders without one.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.sets.interval_algebra"

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
_MISSING = object()  # A field that a record made with object.__new__ lacks (None is an infinite end).
_NOT_A_SET = "{} requires an IntervalSet for {}"
_NOT_CANONICAL = "{} requires a canonical IntervalSet for {}, as the IntervalSet constructor builds it"
# The bounds of the module docstring: the estimated check within 4/5 of the default max_work; the arguments and the
# value rebuilt once within 12/25 of the default work and refinements and 2/5 of the nodes, evidence items and
# decoded bytes and nodes, since a Result rebuilds them twice.
_CHECK_SHARE = (4, 5)
_RELOAD_NODES = (2, 5)
_RELOAD_SHARES = {"max_work": (12, 25), "max_refinements": (12, 25), "max_nodes": _RELOAD_NODES,
                  "max_evidence_items": _RELOAD_NODES}
_PER_MEMBER = 16  # Work per sample, cut point and interval that the estimate of the check adds for its walks.
_GCD_AFTER = 4  # Clashes of two brackets of two polynomials before the line looks for a common root.


# Whether a real lies in the result, from whether it lies in each argument, in the order of _ARGUMENTS.
def _union(in_a: bool, in_b: bool) -> bool:
    return in_a or in_b


def _intersection(in_a: bool, in_b: bool) -> bool:
    return in_a and in_b


def _difference(in_a: bool, in_b: bool) -> bool:
    return in_a and not in_b


def _complement(in_a: bool) -> bool:
    return not in_a


_TRUTH = {"set_union": _union, "set_intersection": _intersection, "set_difference": _difference,
          "set_complement": _complement}


def produce(operation, args):
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name and
    kind; the module states the algorithm and the evidence. InvalidInput for an argument that is no canonical
    IntervalSet, ResourceLimitError past the budget or past the bound of the module."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the interval_algebra family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    sets = tuple(args[name] for name in names)
    for name, value in zip(names, sets):
        if type(value) is not IntervalSet:  # An EmptySet, a UniversalSet, a list, a subclass: no IntervalSet.
            raise InvalidInput(_NOT_A_SET.format(operation, name))
    with computation() as budget:
        budget.tick()  # The call itself.
        records = tuple(_records(budget, _NOT_CANONICAL.format(operation, name), value)
                        for name, value in zip(names, sets))
        line = _Line(budget, records, [_NOT_CANONICAL.format(operation, name) for name in names])
        truth = tuple(map(_TRUTH[operation], *line.members))
        value = line.assemble(truth)
        samples = line.samples()
        _bound(budget, sets, value, line, tuple(Fraction(x.numerator, x.denominator) for x in samples))
        return value, (samples,)


# ---------------------------------------------------------------------------------------------------------------
# The inputs, validated

def _records(budget: Budget, message: str, value: object) -> tuple[tuple[object, object, bool, bool], ...]:
    """(lower, upper, left_closed, right_closed) of every interval of an IntervalSet argument, each end None, a
    canonical Rational or a RealAlgebraicRoot with a primitive polynomial and an index below its degree, read once;
    InvalidInput(message) for a record that the constructors did not make. The line checks the rest: that every root
    is real, its polynomial square-free, and the intervals canonical."""
    intervals = getattr(value, "intervals", None)
    if type(intervals) is not tuple:
        raise InvalidInput(message)
    budget.tick(len(intervals))
    found = []
    for item in intervals:
        if type(item) is not Interval:
            raise InvalidInput(message)
        lower, upper = getattr(item, "lower", _MISSING), getattr(item, "upper", _MISSING)
        left_closed, right_closed = getattr(item, "left_closed", None), getattr(item, "right_closed", None)
        if type(left_closed) is not bool or type(right_closed) is not bool:
            raise InvalidInput(message)
        found.append((_end(budget, message, lower), _end(budget, message, upper), left_closed, right_closed))
    return tuple(found)


def _end(budget: Budget, message: str, end: object) -> object:
    """An end, validated as _records states it."""
    from ...trusted.upoly import charge_gcd, max_bits

    if end is None:  # An infinite end.
        return None
    if type(end) is Rational:
        numerator, denominator = getattr(end, "numerator", None), getattr(end, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            raise InvalidInput(message)
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator != 1:
            charge_gcd(max(numerator.bit_length(), denominator.bit_length()))
            if gcd(numerator, denominator) != 1:
                raise InvalidInput(message)
        return end
    if type(end) is RealAlgebraicRoot:
        coefficients, index = getattr(end, "integer_coefficients", None), getattr(end, "real_index", None)
        if (type(coefficients) is not tuple or len(coefficients) < 2 or type(index) is not int
                or not 0 <= index < len(coefficients) - 1):
            raise InvalidInput(message)
        budget.tick(len(coefficients))
        for coefficient in coefficients:
            if type(coefficient) is not int:
                raise InvalidInput(message)
            budget.check_int(coefficient)
        charge_gcd(max_bits(coefficients), len(coefficients))
        if coefficients[-1] < 1 or gcd(*coefficients) != 1:
            raise InvalidInput(message)
        return end
    raise InvalidInput(message)  # An int, a float, a str, a Fraction or a record of another class.


def _key(end: object) -> tuple:
    """The name of the number that a validated end stands for: ("q", value) for a Rational and for the root of a
    linear polynomial c0 + c1*x (primitive with c1 > 0, so -c0/c1 is in lowest terms), ("root", polynomial, index)
    for a root of a polynomial of degree two or more."""
    if type(end) is Rational:
        return ("q", Fraction(end.numerator, end.denominator))
    coefficients = end.integer_coefficients
    if len(coefficients) == 2:
        return ("q", Fraction(-coefficients[0], coefficients[1]))
    return ("root", coefficients, end.real_index)


# ---------------------------------------------------------------------------------------------------------------
# The line: every point of the cut, separated by rational brackets

class _Point:
    """A point of the line: an exact rational (polynomial None, low == high), or the one root of polynomial in the
    open interval (low, high), whose ends are no roots of it; sign is the sign of polynomial at low. names are the
    keys of the ends that stand for it."""

    __slots__ = ("low", "high", "polynomial", "sign", "names", "dead")

    def __init__(self, low: Fraction, high: Fraction, polynomial: tuple[int, ...] | None, sign: int,
                 names: set) -> None:
        self.low, self.high, self.polynomial, self.sign, self.names = low, high, polynomial, sign, names
        self.dead = False


class _Line:
    """The points of the inputs' cut in increasing order, each value once, with rational brackets that leave a
    rational between any two neighbours: the cut points (the inputs' rational ends and the roots of their linear
    polynomials) and every real root of the polynomial of each input end of degree two or more. members holds, for
    each input, whether it contains each member of the line: member 2k - 1 is the point number k (from 1), member
    2j the open cell number j (from 0)."""

    def __init__(self, budget: Budget, records: tuple, messages: list[str]) -> None:
        from ...kernels import upoly as kernels
        from ...trusted import upoly as tp

        self._budget, self._kernels, self._tp = budget, kernels, tp
        self.records = records  # The validated intervals of each input, as _records reads them.
        self._exact: dict[Fraction, _Point] = {}
        self._common: dict[tuple, tuple | None] = {}
        self._clashes: dict[tuple, int] = {}
        points: list[_Point] = []
        roots: dict[tuple[int, ...], list] = {}  # polynomial -> (message, indices named)
        self.cuts: dict[Fraction, None] = {}  # The checker's cut points: rational ends and roots of linear ends.
        self.linear: dict[tuple[int, ...], None] = {}  # The distinct polynomials of linear RealAlgebraicRoot ends.
        for message, intervals in zip(messages, records):
            for lower, upper, _, _ in intervals:
                for end in (lower, upper):
                    if end is None:
                        continue
                    key = _key(end)
                    if key[0] == "q":
                        points.append(self._exact_point(key[1], key))
                        self.cuts[key[1]] = None
                        if type(end) is RealAlgebraicRoot:
                            self.linear[end.integer_coefficients] = None
                    else:
                        roots.setdefault(key[1], [message]).append(key[2])
        self.polynomials = tuple(roots)  # In the order of the ends, as the checker multiplies them into H.
        for polynomial, (message, *indices) in roots.items():
            chain = kernels.sturm_chain(polynomial)  # Charged; a cache hit charges the same.
            if len(chain[-1]) > 1:  # The chain ends in the gcd of the polynomial and its derivative.
                raise InvalidInput(message)
            brackets = kernels.isolate(polynomial)
            if max(indices) >= len(brackets):  # The index names no real root.
                raise InvalidInput(message)
            for index, (low, high) in enumerate(brackets):
                key = ("root", polynomial, index)
                low, high = Fraction(*low), Fraction(*high)
                if low == high:  # A rational root, met at a midpoint of the isolation.
                    points.append(self._exact_point(low, key))
                else:
                    points.append(_Point(low, high, polynomial, self._sign(polynomial, low), {key}))
        self.points = self._separate(points)
        self.number = {key: k for k, point in enumerate(self.points, 1) for key in point.names}
        self.members = tuple(self._members(message, intervals) for message, intervals in zip(messages, records))

    # --- Construction ---------------------------------------------------------------------------------------------

    def _sign(self, polynomial: tuple[int, ...], x: Fraction) -> int:
        return self._tp.sign_at(polynomial, x.numerator, x.denominator)  # Charged by the size of the point.

    def _exact_point(self, value: Fraction, key: tuple) -> _Point:
        point = self._exact.get(value)
        if point is None:
            point = self._exact[value] = _Point(value, value, None, 0, set())
        point.names.update((key, ("q", value)))
        return point

    def _to_exact(self, point: _Point, value: Fraction) -> None:
        """The bracket's root is the rational value: the point becomes that rational, or joins the one there."""
        existing = self._exact.get(value)
        if existing is None:
            point.low = point.high = value
            point.polynomial, point.sign = None, 0
            point.names.add(("q", value))
            self._exact[value] = point
        else:
            existing.names.update(point.names)
            point.dead = True

    def _bisect(self, point: _Point, at: Fraction | None = None) -> None:
        """Halve the bracket of a root at its midpoint, or cut it at the rational at inside it. An exact point has no
        bracket and is left as it is: within one pass of _separate a bracket can become exact after the pass sorted
        the line, and bisecting that exact point would merge it into itself and drop it from the line (an internal
        error in _members); the pass has already set changed, so the next pass sorts again and two distinct exact
        points are apart once sorted."""
        if point.polynomial is None:
            return
        self._budget.refine()  # One refinement of an isolating interval, as the kernels count them.
        middle = (point.low + point.high) / 2 if at is None else at
        sign = self._sign(point.polynomial, middle)
        if not sign:
            self._to_exact(point, middle)
        elif sign == point.sign:
            point.low = middle
        else:
            point.high = middle

    def _same_root(self, left: _Point, right: _Point) -> bool:
        """Whether two overlapping brackets of different polynomials hold the same root: a root of the gcd of the two
        polynomials in the overlap is a root of each, so it is the one root of each bracket. The gcd is computed for
        a pair of polynomials whose brackets have clashed _GCD_AFTER times, once."""
        pair = (left.polynomial, right.polynomial)
        if pair[1] < pair[0]:
            pair = pair[1], pair[0]
        clashes = self._clashes.get(pair, 0) + 1
        self._clashes[pair] = clashes
        if clashes < _GCD_AFTER:
            return False
        if pair not in self._common:
            common = self._kernels.gcd_cofactors(*pair)[0]
            self._common[pair] = None if len(common) < 2 else (common, self._kernels.sturm_chain(common))
        found = self._common[pair]
        if found is None:
            return False
        low, high = max(left.low, right.low), min(left.high, right.high)
        if not low < high:
            return False
        count = self._kernels.count_open(found[1], (low.numerator, low.denominator),
                                         (high.numerator, high.denominator))
        if not count:
            return False
        left.low, left.high = low, high  # Inside the bracket of left: its one root is the common one.
        left.sign = self._sign(left.polynomial, low)
        left.names.update(right.names)
        right.dead = True
        return True

    def _separate(self, points: list[_Point]) -> list[_Point]:
        """The points, each value once, in increasing order, with brackets refined until any two neighbours leave a
        rational between them (samples). Each pass sorts the points and narrows every pair of neighbours that does not:
        a bracket that holds or touches an exact point is cut there or halved, and two overlapping brackets are halved
        both, unless they hold the same root of two polynomials, which become one point."""
        budget = self._budget
        points = [point for point in dict.fromkeys(points) if not point.dead]
        while True:
            budget.tick(len(points) * max(1, len(points).bit_length()))  # The sort and the walk.
            points.sort(key=lambda point: (point.low, point.high))
            changed = False
            for left, right in zip(points, points[1:]):
                if left.dead or right.dead or _apart(left, right):
                    continue
                changed = True
                if left.polynomial is None:  # An exact point at the lower end of a bracket.
                    self._bisect(right)
                elif right.polynomial is None:  # A bracket that holds the exact point or ends at it.
                    self._bisect(left, right.low if right.low < left.high else None)
                elif left.polynomial == right.polynomial:
                    raise ArithmeticError("Two isolating intervals of one polynomial overlap")
                elif not self._same_root(left, right):
                    self._bisect(left)
                    if not right.dead:
                        self._bisect(right)
            points = [point for point in points if not point.dead]
            if not changed:
                return points

    # --- Sets -------------------------------------------------------------------------------------------------------

    def _members(self, message: str, intervals: tuple) -> tuple[bool, ...]:
        """Whether an input holds each member of the line; InvalidInput(message) unless its intervals are the maximal
        runs of those members, which makes it canonical: nonempty, sorted, disjoint, not mergeable, infinite ends
        open."""
        r = len(self.points)
        truth = [False] * (2 * r + 1)
        written = []
        self._budget.tick(len(intervals))
        for lower, upper, left_closed, right_closed in intervals:
            low = None if lower is None else self.number[_key(lower)]
            high = None if upper is None else self.number[_key(upper)]
            written.append((low, high, left_closed, right_closed))
            first = 0 if low is None else 2 * low - left_closed
            last = 2 * r if high is None else 2 * high - 2 + right_closed
            self._budget.tick(max(0, last - first + 1))
            for member in range(first, last + 1):
                truth[member] = True
        if _runs(truth) != written:
            raise InvalidInput(message)
        return tuple(truth)

    def assemble(self, truth: tuple[bool, ...]) -> IntervalSet:
        """The canonical IntervalSet of the true members, each finite end written as the first input end at its
        point."""
        names: dict[int, object] = {}
        for intervals in self.records:
            for lower, upper, _, _ in intervals:
                for end in (lower, upper):
                    if end is not None:
                        names.setdefault(self.number[_key(end)], end)
        self._budget.tick(len(truth))
        intervals = []
        for low, high, left_closed, right_closed in _runs(truth):
            try:
                lower = None if low is None else names[low]
                upper = None if high is None else names[high]
            except KeyError as error:  # The boundary of the result lies among the inputs' ends.
                raise ArithmeticError("An end of the result is no end of an input") from error
            intervals.append(Interval(lower, upper, left_closed, right_closed))
        return IntervalSet(tuple(intervals))

    def samples(self) -> tuple[Rational, ...]:
        """rho_0 < ... < rho_r: the rational of least denominator below the first point, between any two neighbours
        and above the last; (0,) without a point."""
        points = self.points
        if not points:
            return (Rational(0),)
        self._budget.tick(len(points) + 1)
        values = [_simplest(None, False, points[0].low, points[0].polynomial is not None)]
        for left, right in zip(points, points[1:]):
            values.append(_simplest(left.high, left.polynomial is not None, right.low, right.polynomial is not None))
        values.append(_simplest(points[-1].high, points[-1].polynomial is not None, None, False))
        return tuple(Rational(value.numerator, value.denominator) for value in values)


# ---------------------------------------------------------------------------------------------------------------
# The bound: a Result that the default limits can reload and verify

def _evaluation_units(length: int, bits: int, point: Fraction) -> int:
    """The work that the checker's cell engine charges for one evaluation of a polynomial of length coefficients of
    at most bits bits at a rational point: the charges of mathforge.certify.cells._sign_at, restated here, and those
    of trusted.upoly.sign_at."""
    from ...trusted.upoly import evaluation_factor

    numerator, denominator = point.numerator, point.denominator
    point_bits = max(numerator.bit_length(), denominator.bit_length())
    units = length * (bits >> 14) + length * ((bits >> 6) * (point_bits >> 6) >> 8)
    if denominator != 1:
        units += length * ((bits >> 6) * (length * point_bits >> 6) >> 8)
    return units + length * evaluation_factor(length, numerator, denominator)


def _check_work(line: _Line, value: IntervalSet, samples: tuple[Fraction, ...]) -> int:
    """An estimate of the work of the family's checker on the claim, from the steps that dominate it (module
    docstring, Bounds), run or restated here in the active budget; ResourceLimitError when the steps that run here
    exceed it."""
    from ...kernels import upoly as kernels
    from ...trusted import upoly as tp

    budget = current_budget()
    start = budget.work
    master: tuple[int, ...] = (1,)
    for polynomial in line.polynomials:  # H, the product that the checker forms, in its order.
        master = tp.mul(master, polynomial)
    bits, cuts = tp.max_bits(master), tuple(line.cuts)
    units = (bits >> 6) * (bits >> 6) >> 6  # The content of H.
    units *= len(master)
    units += sum(_evaluation_units(len(master), bits, cut) for cut in cuts)
    real = sum(1 for point in line.points if any(name[0] == "root" for name in point.names))
    if len(master) > 1 and real != len(master) - 1:  # Not d real roots of H: the checker builds a Sturm chain.
        # The subresultant PRS of H and H', which the checker's chain is (certify.upoly.sturm_chain), as the kernels
        # compute it for a resultant: the same pseudo-remainders, charged alike. It is 0 when H is not square-free,
        # and the checker then builds the chain of the square-free part S as well.
        if not kernels.subresultant(master, tp.derivative(master)):
            master = kernels.squarefree_part(master)
            if len(master) > 1:
                kernels.subresultant(master, tp.derivative(master))
        bits = tp.max_bits(master)
    if len(master) > 1:
        units += sum(_evaluation_units(len(master), bits, sample) for sample in samples)
    for polynomial in line.polynomials + tuple(line.linear):  # Locating each polynomial of an end.
        length, own = len(polynomial), tp.max_bits(polynomial)
        units += sum(_evaluation_units(length, own, cut) for cut in cuts)
        if length > 2:
            if polynomial != master:  # The division of S by the polynomial.
                units += (len(master) - length + 1) * length * tp.size_factor(bits, own)
            units += sum(_evaluation_units(length, own, sample) for sample in samples)
    intervals = sum(len(records) for records in line.records) + len(value.intervals)
    units += _PER_MEMBER * (len(samples) + len(cuts) + intervals)
    units += _reading_units(line, value, samples)
    return budget.work - start + units


def _reading_units(line: _Line, value: IntervalSet, samples: tuple[Fraction, ...]) -> int:
    """The checker's work on the Rationals and the root records themselves, beyond evaluations: every gcd of a
    canonical test and every comparison of two Rationals, which the cell engine charges by the square of their
    length (trusted.upoly.charge_gcd and charge_product). Without this term the estimate ignored them, and two sets of
    10 intervals with ends (10^10000 + k)/7, which it put at 86,000 units, needed 2,190,000: a Result that the default
    limits could not verify. Every count is an upper bound, at the longest Rational B of the cut points and samples:
    a gcd for each Rational the checker validates (a Rational or linear end of an input twice, in its cut and where
    Cells.membership locates it or merge_cuts reads its root, a Rational end of the value once, each cut point and
    sample once); a comparison for each step of the sort of the cut points (n * bits(n)), for each neighbouring pair
    of cut points and of samples, for each step of the walk of the cut points against the samples, and for each step
    of the bisection that locates a Rational end among the cut points. A root record costs its length three times
    (read in the cut, in the check of its name and in its validation) and the gcd of its coefficients once wherever
    it is located."""
    from ...trusted.upoly import max_bits

    cuts = tuple(line.cuts)
    longest = max([max(x.numerator.bit_length(), x.denominator.bit_length()) for x in cuts + samples], default=0)
    gcd_units, product_units = (longest >> 6) ** 2 >> 6, (longest >> 6) ** 2 >> 8
    claimed = tuple((item.lower, item.upper) for item in value.intervals)
    ends = [(end, True) for intervals in line.records for lower, upper, _, _ in intervals for end in (lower, upper)]
    ends += [(end, False) for pair in claimed for end in pair]
    current_budget().tick(len(ends))  # The walk below.
    gcds, located, units = len(cuts) + len(samples), 0, 0
    for end, of_input in ends:
        if end is None:  # An infinite end.
            continue
        if type(end) is Rational:
            gcds += 2 if of_input else 1
            located += 1
            continue
        coefficients = end.integer_coefficients
        if len(coefficients) == 2 and of_input:
            gcds += 2
        words = max_bits(coefficients) >> 6
        units += 3 * len(coefficients) + len(coefficients) * (words * words >> 6)
    count = len(cuts)
    comparisons = count * count.bit_length() + 2 * (count + len(samples)) + located * (count.bit_length() + 1)
    return units + gcds * gcd_units + comparisons * (1 + 2 * product_units)


def _bound(budget: Budget, sets: tuple[IntervalSet, ...], value: IntervalSet, line: _Line,
           samples: tuple[Fraction, ...]) -> None:
    """ResourceLimitError unless the default limits can verify the claim and reload a Result that holds it (module
    docstring, Bounds), whatever limits the caller gave."""
    from dataclasses import replace

    from ...errors import SerializationError
    from ...serialization import from_data, to_data

    limits = ComputationLimits()
    # The estimate runs in a budget of the default limits, or of what remains of the caller's when that is less,
    # and the caller pays for the work done.
    room = max(1, min(limits.max_work, budget.limits.max_work - budget.work))
    work = None
    with computation(replace(limits, max_work=room), fresh=True) as model:
        try:
            work = _check_work(line, value, samples)
        except ResourceLimitError:
            pass
    budget.tick(model.work)
    if work is None or work * _CHECK_SHARE[1] > limits.max_work * _CHECK_SHARE[0]:
        estimate = f"more than {room:,}" if work is None else f"about {work:,}"
        raise ResourceLimitError("Computation limit exceeded: the check of this result would not fit the default "
                                 f"limits ({estimate} units of work)")
    # A Result holds each argument and the value twice; it reloads in one budget of the default limits.
    shares = {name: getattr(limits, name) * share[0] // share[1] for name, share in _RELOAD_SHARES.items()}
    decode = DecodeLimits()
    decode = replace(decode, **{name: getattr(decode, name) * _RELOAD_NODES[0] // _RELOAD_NODES[1]
                                for name in ("max_bytes", "max_nodes")})
    try:
        from_data(to_data(sets + (value,)), decode_limits=decode, computation_limits=replace(limits, **shares))
    except SerializationError as error:
        if error.code != "resource_limit_exceeded":
            raise
        raise ResourceLimitError("Computation limit exceeded: a Result holding these sets would not reload within "
                                 "the default limits") from error


def _apart(left: _Point, right: _Point) -> bool:
    """Whether a rational lies strictly between the points of two brackets or exact points, left.low <= right.low:
    between two brackets any rational from left.high to right.low, else one strictly beyond an exact point."""
    if left.polynomial is not None and right.polynomial is not None:
        return left.high <= right.low
    return left.high < right.low


def _runs(truth) -> list[tuple[int | None, int | None, bool, bool]]:
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
        if start == 0:
            lower, left_closed = None, False
        elif start % 2:
            lower, left_closed = (start + 1) // 2, True
        else:
            lower, left_closed = start // 2, False
        if stop == last:
            upper, right_closed = None, False
        elif stop % 2:
            upper, right_closed = (stop + 1) // 2, True
        else:
            upper, right_closed = stop // 2 + 1, False
        intervals.append((lower, upper, left_closed, right_closed))
        start = stop + 1
    return intervals


def _simplest(low: Fraction | None, low_closed: bool, high: Fraction | None, high_closed: bool) -> Fraction:
    """The rational of least denominator, and then of least absolute value, in the nonempty interval from low to
    high, each end included when closed and None when infinite: the continued fraction that the interval's ends
    share, ended by the least partial quotient that the interval admits (the Stern-Brocot tree). A loop, not a
    recursion: two ends such as consecutive ratios of Fibonacci numbers share thousands of partial quotients. Each
    step is charged by the length of the ends."""
    from ...trusted.upoly import charge, charge_product

    if (low is None or low < 0 or (low == 0 and low_closed)) and (high is None or high > 0 or (high == 0
                                                                                             and high_closed)):
        return Fraction(0)
    sign = 1
    if high is not None and high <= 0:  # Wholly at or below zero: mirror it.
        sign, low, low_closed, high, high_closed = -1, -high, high_closed, None if low is None else -low, low_closed
    # Now 0 <= low is finite, and 0 lies outside the interval.
    a, b, c, d = 1, 0, 0, 1  # x = (a*y + b) / (c*y + d) for the y of the current interval.
    while True:
        bits = max(low.numerator.bit_length(), low.denominator.bit_length(),
                   0 if high is None else max(high.numerator.bit_length(), high.denominator.bit_length()))
        charge(1)
        charge_product(bits, bits, 4)
        n = floor(low)
        if low == n and low_closed:
            y = n
            break
        if high is None or n + 1 < high or (n + 1 == high and high_closed):
            y = n + 1
            break
        # No integer inside: y = n + 1/z with z > 1 in the reciprocal interval, the simplest z giving the simplest y.
        a, b, c, d = a * n + b, a, c * n + d, c
        if low == n:  # Open at n: z from 1/(high - n) to infinity.
            low, low_closed, high, high_closed = 1 / (high - n), high_closed, None, False
        else:
            low, low_closed, high, high_closed = 1 / (high - n), high_closed, 1 / (low - n), low_closed
    return sign * Fraction(a * y + b, c * y + d)
