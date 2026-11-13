"""Independent checker of the family mathforge.areas.sets.symmetric: set_symmetric_difference of two finite unions of
real intervals with rational or real algebraic endpoints.

The checker never reruns the producer's walk over sorted intervals and never calls IntervalSet.symmetric_difference.
It decides the claim on a certified cut of the cell engine, mathforge.certify.cells, whose module docstring proves each
step used here, with the same cut, conditions and helpers as the checker of the four operations of
mathforge.areas.sets.interval_algebra (mathforge.certify.sets.interval_algebra, whose docstring states them in full and
whose helpers _cut, _named_by_inputs and _intervals this module reuses: certify-to-certify imports are allowed).
IntervalSet and Interval records (mathforge.intervals) and RealAlgebraicRoot records (mathforge.algebraic) are
recognized by the module and name of their class and read as data.

What it accepts. args holds exactly the arguments a and b, each an IntervalSet record in canonical form. value is an
IntervalSet record in canonical form (cells.Cells.index_form): a tuple of Interval records with bool closure flags,
each nonempty (a single point closed on both sides), infinite ends open, sorted, disjoint and not mergeable. An
EmptySet or a UniversalSet is never accepted: the empty set is IntervalSet(()) and the whole line
IntervalSet((Interval(None, None),)). evidence is (samples,), where samples is a strictly increasing tuple of canonical
Rationals rho_0 < ... < rho_r that certifies the cut of the inputs' endpoints: one sample below the first point, one
strictly between any two neighbouring points and one above the last (one sample, any, when there is no point). The
points, in increasing order and each value once, are

- the cut points Q: the inputs' Rational endpoints and the roots -c0/c1 of their RealAlgebraicRoot endpoints of
  linear polynomials c0 + c1*x;
- every real root of H, the product of the distinct polynomials of the inputs' other RealAlgebraicRoot endpoints, the
  roots that are no endpoint included (an endpoint sqrt 2, root number 1 of x^2 - 2, makes -sqrt 2 a point too).

Anything else answers False: a bool, an int, a float or a str where a Rational belongs, a list where a tuple belongs,
a record of another class, a missing or an extra item, samples that do not separate the points.

Two necessary conditions are tested before Cells runs; neither refuses a true claim whose endpoints are written as the
inputs write them:

1. The claim has at most 2 * (len(a) + len(b)) intervals, the card's bound, tested before anything is read from the
   claim. No true set has more. Let E be the set of the distinct finite endpoint values of a and b; |E| is at most
   2 * (len(a) + len(b)) minus the number of infinite ends of the inputs, since every interval has two ends. Every
   finite endpoint of the true set lies in E (condition 2). Two intervals of a canonical set never share their lower
   end: both would hold the reals just above it, unless one is the single point {c} and the other starts open at c,
   and those two are mergeable. So the intervals of the true set have distinct lower ends, each in E or -infinity,
   and -infinity is a lower end only when the reals below every point of E lie in exactly one input, that is, when an
   input has an infinite lower end, which E does not count. Hence at most |E| + 1 <= 2 * (len(a) + len(b)) intervals
   when an input has an infinite end, and at most |E| <= 2 * (len(a) + len(b)) otherwise. (The true count is in fact
   at most len(a) + len(b): on the members of the cut below, a canonical set with k intervals has k = (c + f + l) / 2,
   where c counts the neighbouring members on which its truth changes and f, l its truth on the first and the last
   member; the truth of a ^ b changes only where that of a or of b does, and its f and l are at most the sums of
   those of a and b. The card's looser bound is kept: it only decides how early a claim is refused.)
2. Every finite endpoint of the claim is named by the inputs: a Rational that is a cut point of Q, the
   RealAlgebraicRoot of a linear polynomial whose root is a cut point of Q, or a RealAlgebraicRoot whose polynomial,
   of degree two or more, is the polynomial of a RealAlgebraicRoot endpoint of an input (with any index). A true set
   needs no other name: a finite endpoint of a canonical set is a boundary point of it, and the boundary of a ^ b lies
   in the union of the boundaries of a and b (near a point that is no endpoint of a or b, membership in a and in b,
   and so in a ^ b, is constant), so every endpoint of the true set is an endpoint of an input, which the claim can
   write as that input writes it. The condition keeps the claim from bringing polynomials of its own into the master
   polynomial: an oversized or foreign claim answers False at once instead of exhausting the budget. A true set
   written with another name, such as sqrt 2 as a root of (x^2 - 2)(x^2 + 1) or of x^3 - 2x when no input uses that
   polynomial, is refused: the condition errs towards refusal, never towards acceptance (the departure from the card
   of mathforge.certify.sets.interval_algebra, kept here).

The theorem that makes True imply the claim. Cells(H, samples, Q) accepts only samples that certify the points
alpha_1 < ... < alpha_r of the cut, which are all the real roots of H together with the cut points, and the open cells
between them (cells module docstring, item 1). Every finite endpoint of a and b is such a point: a Rational one and
the root of a linear polynomial are cut points, and any other is a root of its polynomial, which divides H.
Cells.membership locates every endpoint, refuses a and b unless they are canonical, and returns the truth of x in a at
each point and on each whole open cell, since a canonical set whose finite endpoints are points holds each open cell
entirely or not at all; and so for b. The truth of the claimed set at each point and on each open cell then follows:

- set_symmetric_difference(a, b): (x in a) != (x in b), x in exactly one of a and b.

The true set is the union of the true points and the true open cells, and Cells.matches accepts the claim only when
every finite endpoint of it is a point, its intervals in index form are exactly the maximal runs of true members, and
it is written in the canonical form. Hence an accepted claim of set_symmetric_difference(a, b) is the symmetric
difference of a and b in canonical form.

Work. Every loop of this module and of the reused helpers charges the active budget before it runs, and the cell
engine and trusted.upoly charge their own work; the module calls no builtin pow. The cut is the one that set_union of
the same a and b builds, so the check costs what the union's check costs (the measurements of
mathforge.certify.sets.interval_algebra apply). Without an active budget the check opens one of the default
ComputationLimits; work past the budget raises ResourceLimitError, which leaves a claim unverified, never verified.
Cells bounds the samples by the degree of H and the number of cut points before reading them, so oversized evidence
answers False at once.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from mathforge.limits import Budget, computation

from ..cells import Cells, Rejected
from .interval_algebra import _cut, _intervals, _named_by_inputs

_OPERATION = "set_symmetric_difference"
_ARGUMENTS = ("a", "b")  # In the order of the spec.


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of set_symmetric_difference on args, by the theorem of the
    module. False for an unknown operation, for arguments other than the operation's IntervalSet records and for any
    malformed or non-canonical value or evidence; ResourceLimitError past the budget."""
    if type(operation) is not str or operation != _OPERATION or type(args) is not dict:
        return False
    if len(args) != len(_ARGUMENTS) or any(name not in args for name in _ARGUMENTS):
        return False
    if type(evidence) is not tuple or len(evidence) != 1:
        return False
    with computation() as budget:  # One budget for every step: without an active budget, each would open its own.
        try:
            return _verified(budget, tuple(args[name] for name in _ARGUMENTS), value, evidence[0])
        except Rejected:  # The cell engine's refusal of malformed evidence, records or claims, and the helpers'.
            return False


def _symmetric(in_a: bool, in_b: bool) -> bool:
    return in_a != in_b


def _verified(budget: Budget, sets: tuple, value: object, samples: object) -> bool:
    claimed = _intervals(value)
    # Condition 1 of the module docstring, before anything is read from the claim.
    if len(claimed) > 2 * sum([len(_intervals(item)) for item in sets]):
        return False
    master, cuts, polynomials = _cut(budget, sets)
    if not _named_by_inputs(budget, claimed, cuts, polynomials):  # Condition 2.
        return False
    grid = Cells(master, samples, cuts)
    memberships = [grid.membership(item) for item in sets]  # Locates every input endpoint; canonical inputs only.
    budget.tick(2 * grid.root_count + 1)  # One decision per point and per open cell.
    root_truth = tuple(map(_symmetric, *[roots for roots, _ in memberships]))
    cell_truth = tuple(map(_symmetric, *[inside for _, inside in memberships]))
    return grid.matches(value, root_truth, cell_truth)
