"""Independent checker of the family mathforge.areas.sets.interval_algebra: set_union, set_intersection, set_difference
and set_complement of finite unions of real intervals with rational or real algebraic endpoints.

The checker never reruns the producer's merge of sorted intervals. It decides every claim on a certified cut of the
cell engine, mathforge.certify.cells, whose module docstring proves each step used here (catalogue section 2.3 and
docs/v09/cards/m2-algebra-equations-sets-numeric.md). IntervalSet and Interval records (mathforge.intervals) and
RealAlgebraicRoot records (mathforge.algebraic) are recognized by the module and name of their class and read as data.

What it accepts. args holds exactly the operation's arguments: a and b, or a alone for set_complement, each an
IntervalSet record in canonical form. value is an IntervalSet record in canonical form (cells.Cells.index_form): a
tuple of Interval records with bool closure flags, each nonempty (a single point closed on both sides), infinite ends
open, sorted, disjoint and not mergeable. An EmptySet or a UniversalSet is never accepted: the empty set is
IntervalSet(()) and the whole line IntervalSet((Interval(None, None),)). evidence is (samples,), where samples is a
strictly increasing tuple of canonical Rationals rho_0 < ... < rho_r that certifies the cut of the inputs' endpoints:
one sample below the first point, one strictly between any two neighbouring points and one above the last (one
sample, any, when there is no point). The points, in increasing order and each value once, are

- the cut points Q: the inputs' Rational endpoints and the roots -c0/c1 of their RealAlgebraicRoot endpoints of
  linear polynomials c0 + c1*x;
- every real root of H, the product of the distinct polynomials of the inputs' other RealAlgebraicRoot endpoints, the
  roots that are no endpoint included (an endpoint sqrt 2, root number 1 of x^2 - 2, makes -sqrt 2 a point too).

Anything else answers False: a bool, an int, a float or a str where a Rational belongs, a list where a tuple belongs,
a record of another class, a missing or an extra item, samples that do not separate the points.

The cut. The card builds it with cells.endpoint_cut(a, b, value). This checker builds Cells(H, samples, Q) from the
inputs alone, with the same points: Cells works with the primitive square-free part of H, which is the master
polynomial of endpoint_cut(a, b), and condition 2 below makes every endpoint of the claim a point of that cut, so that
endpoint_cut(a, b, value) has these same points and the evidence is the card's. It does not call endpoint_cut, which
runs a square-free pass of its own (Euclid's algorithm on the product of the polynomials and its derivative) before any
sample is read. Cells needs no such pass when H changes sign between the samples as often as its degree, which proves
H square-free with all its roots real, as for the quadratic ends that the MCP tools decode, and otherwise takes the
square-free part from H's Sturm chain. That pass alone exhausts the default budget on two sets of 16 intervals with
distinct endpoints 2k + sqrt 2, which this check verifies with 32 intervals each (Work, below).

Two necessary conditions are tested before Cells runs; neither refuses a true claim whose endpoints are written as the
inputs write them:

1. The claim has at most len(a) + len(b) + 1 intervals (len(a) + 1 for set_complement), the card's bound, tested
   before anything is read from the claim. No true set has more: each interval of a union holds an interval of a or of
   b, and they are disjoint; the intervals of an intersection are the nonempty intersections of an interval of a with
   one of b, and the pairs of sorted disjoint intervals that overlap form a forest on len(a) + len(b) vertices, so
   there are at most len(a) + len(b) - 1 of them; the complement of a has at most len(a) + 1 intervals, the gaps
   around those of a; and a - b is the intersection of a with the complement of b.
2. Every finite endpoint of the claim is named by the inputs: a Rational that is a cut point of Q, the
   RealAlgebraicRoot of a linear polynomial whose root is a cut point of Q, or a RealAlgebraicRoot whose polynomial,
   of degree two or more, is the polynomial of a RealAlgebraicRoot endpoint of an input (with any index). A true set
   needs no other name: a finite endpoint of a canonical set is a boundary point of it, and the boundary of a union,
   an intersection, a difference or a complement lies in the union of the boundaries of the operands (near a point
   that is no endpoint of a or b, membership in a and in b, and so in the result, is constant), so every endpoint of
   the true set is an endpoint of an input, which the claim can write as that input writes it. The condition keeps
   the claim from bringing polynomials of its own into the master polynomial, whose evaluations at the samples, and
   whose Sturm chain when it has non-real roots, are the costly part of the check: an oversized or foreign claim
   answers False at once instead of exhausting the budget. A true set written with another name, such as sqrt 2 as a
   root of (x^2 - 2)(x^2 + 1) or of x^3 - 2x when no input uses that polynomial, is refused: the condition errs
   towards refusal, never towards acceptance. This is the one departure from the card: on the cut
   endpoint_cut(a, b, value), whose master polynomial holds the claim's polynomials too, Cells would locate such an
   endpoint and accept the claim.

The theorem that makes True imply the claim. Cells(H, samples, Q) accepts only samples that certify the points
alpha_1 < ... < alpha_r of the cut, which are all the real roots of H together with the cut points, and the open cells
between them (cells module docstring, item 1). Every finite endpoint of a and b is such a point: a Rational one and
the root of a linear polynomial are cut points, and any other is a root of its polynomial, which divides H.
Cells.membership locates every endpoint (item 4: a RealAlgebraicRoot is the root of its number among the points at
which its polynomial vanishes, once that polynomial is found square-free and dividing the square-free part of H),
refuses a and b unless they are canonical, and returns the truth of x in a at each point and on each whole open cell,
since a canonical set whose finite endpoints are points holds each open cell entirely or not at all; and so for b.
The truth of the claimed set at each point and on each open cell then follows from those of a and b by the
operation:

- set_union(a, b): x in a or x in b;
- set_intersection(a, b): x in a and x in b;
- set_difference(a, b): x in a and not x in b;
- set_complement(a): not x in a.

The true set is the union of the true points and the true open cells, and Cells.matches accepts the claim only when
every finite endpoint of it is a point, its intervals in index form are exactly the maximal runs of true members, and
it is written in the canonical form (items 3 and 4). Hence an accepted claim of set_union(a, b) is the union of a and
b in canonical form, and likewise for set_intersection, set_difference and set_complement.

Work. Every loop of this module charges the active budget before it runs, and the cell engine and trusted.upoly charge
their own work; the module calls no builtin pow. Beyond reading the claim, the work depends on the inputs and the
samples alone, because the claim names only points of the inputs' cut. Without an active budget the check opens one of
the default ComputationLimits; work past the budget raises ResourceLimitError, which leaves a claim unverified, never
verified. Cells bounds the samples by the degree of H and the number of cut points before reading them, so oversized
evidence answers False at once.

Measured on 2026-11-12 with CPython on one machine, under the default limits (2,000,000 units), for set_union of two
sets of n intervals with 2n distinct endpoints each (set_intersection and set_difference cost the same, set_complement
of one set far less), with samples at the midpoints between rational brackets of the points: rational endpoints k/100
need about 4,200 units for n = 32 (1 ms); endpoints 2k + sqrt 2, a distinct quadratic each, about 897,000 units for
n = 32 (0.2 s); endpoints sqrt(10^6 + k) about 1,970,000 units for n = 32, the edge of the budget; endpoints
sqrt(10^12 + k) about 496,000 units for n = 16 and sqrt(10^24 + k) about 1,820,000, while n = 24 exceeds the budget
for both (2,280,000 and 8,830,000 units). The evaluations of H at the samples dominate (about 70% of the work for
2k + sqrt 2, 95% for sqrt(10^24 + k)): H has degree 8n, both roots of each of the 4n quadratics being points, and
coefficients about 4n times as long as those of one endpoint's polynomial, and each evaluation is charged by the
length of its sample. The rational of least denominator in each cell therefore about halves the work: 402,000 units
for 2k + sqrt 2 and 1,010,000 for sqrt(10^6 + k) at n = 32, 1,110,000 for sqrt(10^12 + k) at n = 24, while
sqrt(10^12 + k) at n = 32 still needs 3,230,000 units and sqrt(10^24 + k) at n = 24 4,130,000. The MCP tools refuse
those sets: their decoder bounds the cost of rebuilding a set, which grows with the cost of comparing its ends. Of the
unions of quadratic ends in friendly form that they decode, the costliest measured, sqrt(10^6 + k) at n = 32, verifies
before and after a JSON round trip. Ends with complex conjugates are another matter: Cells then builds the Sturm chain
of H, whatever the samples, and the decoder accepts the real cube roots of 2 + k as envelopes. Two sets of 8 such
intervals need 417,000 units for the whole check, of 12 intervals 5,700,000 for the chain alone and of 16 intervals
37,900,000. The producer refuses the inputs whose check does not fit the default limits.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational, trusted_rational
from mathforge.trusted import upoly as tp

from ..cells import Cells, Rejected, merge_cuts

# The record classes read here, by module and qualified name: a checker never imports them.
_INTERVAL_SET = ("mathforge.intervals", "IntervalSet")
_INTERVAL = ("mathforge.intervals", "Interval")
_ALGEBRAIC_ROOT = ("mathforge.algebraic", "RealAlgebraicRoot")
_MISSING = object()  # An attribute that a forged record lacks (None is an infinite end).

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "set_union": ("a", "b"),
    "set_intersection": ("a", "b"),
    "set_difference": ("a", "b"),
    "set_complement": ("a",),
}


def _union(in_a: bool, in_b: bool) -> bool:
    return in_a or in_b


def _intersection(in_a: bool, in_b: bool) -> bool:
    return in_a and in_b


def _difference(in_a: bool, in_b: bool) -> bool:
    return in_a and not in_b


def _complement(in_a: bool) -> bool:
    return not in_a


# Whether a real lies in the claimed set, from whether it lies in each argument, in the order of _ARGUMENTS.
_TRUTH: dict[str, Callable[..., bool]] = {
    "set_union": _union,
    "set_intersection": _intersection,
    "set_difference": _difference,
    "set_complement": _complement,
}


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorem of the module.
    False for an unknown operation, for arguments other than the operation's IntervalSet records and for any
    malformed or non-canonical value or evidence; ResourceLimitError past the budget."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = _ARGUMENTS.get(operation)
    if names is None or len(args) != len(names) or any(name not in args for name in names):
        return False
    if type(evidence) is not tuple or len(evidence) != 1:
        return False
    with computation() as budget:  # One budget for every step: without an active budget, each would open its own.
        try:
            return _verified(budget, operation, tuple(args[name] for name in names), value, evidence[0])
        except Rejected:  # The cell engine's refusal of malformed evidence, records or claims, and this module's.
            return False


def _verified(budget: Budget, operation: str, sets: tuple, value: object, samples: object) -> bool:
    claimed = _intervals(value)
    # Condition 1 of the module docstring, before anything is read from the claim.
    if len(claimed) > sum([len(_intervals(item)) for item in sets]) + 1:
        return False
    master, cuts, polynomials = _cut(budget, sets)
    if not _named_by_inputs(budget, claimed, cuts, polynomials):  # Condition 2.
        return False
    grid = Cells(master, samples, cuts)
    memberships = [grid.membership(item) for item in sets]  # Locates every input endpoint; canonical inputs only.
    decide = _TRUTH[operation]
    budget.tick(2 * grid.root_count + 1)  # One decision per point and per open cell.
    root_truth = tuple(map(decide, *[roots for roots, _ in memberships]))
    cell_truth = tuple(map(decide, *[inside for _, inside in memberships]))
    return grid.matches(value, root_truth, cell_truth)


def _named(value: object) -> tuple[str, str]:
    kind = type(value)
    return kind.__module__, kind.__qualname__


def _intervals(value: object) -> tuple:
    """The entries of an IntervalSet record, read once; Rejected for anything else."""
    if _named(value) != _INTERVAL_SET:
        raise Rejected("An IntervalSet record is required")
    records = getattr(value, "intervals", None)
    if type(records) is not tuple:
        raise Rejected("An IntervalSet holds a tuple of intervals")
    return records


def _ends(record: object) -> tuple[object, object]:
    """(lower, upper) of an Interval record, each read once (_MISSING for an absent field); Rejected for anything
    else. Cells validates the ends and the flags where it locates them; this module only reads their names."""
    if _named(record) != _INTERVAL:
        raise Rejected("An IntervalSet holds Interval records")
    return getattr(record, "lower", _MISSING), getattr(record, "upper", _MISSING)


def _coefficients(budget: Budget, end: object, longest: int | None = None) -> tuple[int, ...] | None:
    """The polynomial of a RealAlgebraicRoot record: a tuple of at least two exact ints, and at most longest of them
    when longest is given, a bound tested before any coefficient is read; None for anything else."""
    if _named(end) != _ALGEBRAIC_ROOT:
        return None
    coefficients = getattr(end, "integer_coefficients", None)
    if (type(coefficients) is not tuple or len(coefficients) < 2
            or (longest is not None and len(coefficients) > longest)):
        return None
    budget.tick(len(coefficients))
    for coefficient in coefficients:
        if type(coefficient) is not int:  # Neither a bool nor a float is an exact integer.
            return None
    return coefficients


def _cut(budget: Budget, sets: tuple) -> tuple[tuple[int, ...], tuple[Rational, ...], frozenset]:
    """(H, Q, polynomials) of the inputs' finite endpoints (module docstring, The cut): Q their Rational endpoints and
    the roots -c0/c1 of their RealAlgebraicRoot endpoints of linear polynomials c0 + c1*x, in increasing order and
    each once (cells.merge_cuts); polynomials the set of the polynomials of their other RealAlgebraicRoot endpoints;
    and H the product of those polynomials, (1,) when there is none.

    Rejected for an endpoint that is no Rational, RealAlgebraicRoot or None, for an integer longer than
    max_integer_bits and for a linear polynomial that is not primitive with c1 > 0; ResourceLimitError when the
    product outgrows max_integer_bits. Cells.membership validates the rest of every endpoint where it locates it."""
    limit = budget.limits.max_integer_bits
    product: tuple[int, ...] = (1,)
    polynomials: set[tuple[int, ...]] = set()
    points: list[Rational] = []
    for item in sets:
        records = _intervals(item)
        budget.tick(len(records))
        for record in records:
            for end in _ends(record):
                if end is None:  # An infinite end.
                    continue
                if type(end) is Rational:  # merge_cuts validates it.
                    points.append(end)
                    continue
                coefficients = _coefficients(budget, end)
                if coefficients is None:
                    raise Rejected("An endpoint is a Rational, a RealAlgebraicRoot or None")
                bits = tp.max_bits(coefficients)
                if bits > limit:
                    raise Rejected("An integer is longer than max_integer_bits")
                if len(coefficients) == 2:
                    tp.charge_gcd(bits)
                    if coefficients[1] < 1 or gcd(*coefficients) != 1:
                        raise Rejected("A RealAlgebraicRoot holds a primitive polynomial with a positive leading "
                                       "coefficient")
                    points.append(trusted_rational(-coefficients[0], coefficients[1]))  # In lowest terms.
                elif coefficients not in polynomials:
                    polynomials.add(coefficients)
                    product = tp.mul(product, coefficients)  # Charged by trusted.upoly before it runs.
                    if tp.max_bits(product) > limit:
                        raise ResourceLimitError("Computation limit exceeded: integer bits")
    return product, merge_cuts(tuple(points)), frozenset(polynomials)


def _named_by_inputs(budget: Budget, claimed: tuple, cuts: tuple[Rational, ...], polynomials: frozenset) -> bool:
    """Condition 2 of the module docstring: every finite endpoint of the claimed intervals is a Rational cut point,
    the RealAlgebraicRoot of a linear polynomial whose root is a cut point, or a RealAlgebraicRoot whose polynomial,
    of degree two or more, is one of the inputs' polynomials. cuts are canonical Rationals (merge_cuts).

    The integers of the claim are bounded by max_integer_bits before they are hashed: no input holds a longer one,
    since _cut refuses it, so a longer one names no point."""
    limit = budget.limits.max_integer_bits
    budget.tick(len(cuts) + len(polynomials))
    points = {(cut.numerator, cut.denominator) for cut in cuts}  # Lowest terms with a positive denominator.
    longest = max([len(polynomial) for polynomial in polynomials], default=2)
    budget.tick(len(claimed))
    for record in claimed:
        for end in _ends(record):
            if end is None:  # An infinite end.
                continue
            if type(end) is Rational:
                key = getattr(end, "numerator", None), getattr(end, "denominator", None)
                if (type(key[0]) is not int or type(key[1]) is not int
                        or max(key[0].bit_length(), key[1].bit_length()) > limit or key not in points):
                    return False
                continue
            coefficients = _coefficients(budget, end, longest)
            if coefficients is None or tp.max_bits(coefficients) > limit:  # An int, a float, a str, a missing
                return False  # field, a record of another class or an integer that no input holds.
            if len(coefficients) == 2:  # c0 + c1*x, primitive with c1 > 0 when valid: its root is -c0/c1.
                if (-coefficients[0], coefficients[1]) not in points:
                    return False
            elif coefficients not in polynomials:
                return False
    return True
