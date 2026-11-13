"""The cell engine: a certified cut of the real line at rational cut points and at the real roots of one polynomial,
shared by the checkers of equations and of sets.

A checker gathers the points where the truth of its claim can change: rational cut points, which it compares
directly, and the real roots of a master polynomial H, which it counts with the signs of H at the samples and, unless
those show all of its roots, one Sturm chain (``master_polynomial``, ``split_rational_roots``, ``merge_cuts``,
``endpoint_cut``). Rational points stay out of H on purpose: each one folded into H as a linear factor would lengthen
H and every member of its Sturm chain, and the coefficients of a chain of many such factors explode. The only
evidence is rational samples rho_0 < ... < rho_r: one below the points, one between any two neighbours and one above
them. ``Cells(H, samples, cuts)`` verifies the samples and then answers every question about the points
alpha_1 < ... < alpha_r and the open cells between them with exact evaluations at rational points alone: no gcd with
the polynomial of a root, and no refinement of an isolating interval.

    Cells(H, samples, cuts)          the certified cut; Rejected when the samples do not certify the points
    check_samples(H, samples, cuts)  whether Cells(H, samples, cuts) accepts, as a bool
    cells.sample_signs(Q)            the signs of a polynomial Q at the samples, and
    cells.root_signs(Q)              at the points, when every real root of Q is a point as item 2 verifies it
    cells.relation_truth(Q, rel)     the truth values of ``Q rel 0`` at the points and on the cells (holds(rel, sign))
    cells.truth(decide, Q1, ...)     the truth values of decide(signs of Q1, ... at one point)
    cells.locate(endpoint)           the number k of the point alpha_k that a Rational or a RealAlgebraicRoot equals
    cells.reconstruct(...)           the canonical interval set of given truth values, in index form
    cells.index_form(set)            a canonical IntervalSet in that form, and
    cells.membership(set)            its truth values
    cells.matches(set, ...)          whether a claimed IntervalSet is exactly the set of given truth values

    master_polynomial(P1, ...)       the square-free part of a product, the master polynomial H
    split_rational_roots(P, qs)      verified rational roots qs of P, and what remains of P without them
    merge_cuts(qs1, ...)             cut points from several sources, in increasing order and each once
    endpoint_cut(set1, ...)          (H, cuts) of the finite endpoints of IntervalSets
    integer_polynomial(cs)           an integer polynomial with the sign of a polynomial of rational coefficients

A polynomial is an integer tuple of mathforge.trusted.upoly: lowest degree first, no trailing zero, ``()`` for zero.
Nothing here imports producer code: a Rational is the class of mathforge.model, and IntervalSet, Interval and
RealAlgebraicRoot records are recognized by the module and name of their class and read as plain data.

Signs come only from polynomials whose sign the checker formed itself, such as ``integer_polynomial`` gives. The
polynomials that ``master_polynomial``, ``endpoint_cut`` and ``Cells.master`` return have a positive leading
coefficient, which reverses the sign of a polynomial whose leading coefficient is negative, and the rest that
``split_rational_roots`` returns is a quotient by linear factors: they serve as master polynomials, never for signs.

Why acceptance proves the claim.

1. Samples. Let S be the square-free part of H and q_1 < ... < q_s the cut points. The points are the real roots of
   S together with the cut points; a cut point with S(q_j) = 0 is a root of S, one point. ``Cells`` requires
   S(rho_k) != 0 at every sample and counts the neighbouring samples between which S changes sign: between two such
   samples S has a root (intermediate values), so there are at most as many sign changes as real roots, and
   ``Cells`` requires exactly as many. When H made primitive changes sign as often as its degree d, it has d
   distinct real roots: it is square-free, it is S, and nothing more is needed. Otherwise the Sturm chain of H made
   primitive (certify.upoly.sturm_chain) ends in a constant multiple of gcd(H, H'): a constant last member proves it
   square-free, and otherwise S is H divided by that gcd, with a chain that must end in a constant; the sign
   variations of S's chain at minus infinity less those at plus infinity are the number of distinct real roots of S
   (Sturm). With as many sign changes as roots, each root of S lies alone between two neighbouring samples at which
   S has opposite signs, and S has no other real root: the roots of S below a sample are the sign changes up to it.
   For t no point let c(t) be the number of points <= t: the roots of S below t plus the cut points q_j < t with
   S(q_j) != 0, which one walk through the two sorted lists finds by comparisons. ``Cells`` accepts r + 1 samples,
   r being the number of points (the roots of S and the cut points that are no root of S), none of them a point
   (S(rho_k) != 0, and rho_k is no cut point), with c(rho_k) = k for every k. Then no point lies up to rho_0, exactly
   one, alpha_k, in each open interval (rho_(k-1), rho_k), and none from rho_r on. So alpha_1 < ... < alpha_r are
   all the points, each cut point q_j is the alpha_k with rho_(k-1) < q_j < rho_k, and rho_k lies in the open cell
   number k: cell 0 is below alpha_1, cell k lies between alpha_k and alpha_(k+1), cell r is above alpha_r, and with
   r = 0 the one cell is the whole line. No Sturm chain is ever evaluated at a sample.
2. Signs. Let Q be a nonzero polynomial. Its zeros at the cut points are found by exact evaluation and divided out
   as often as they divide: Q = C * R, where C is a product of powers of den * x - num over the cut points num/den
   at which Q vanishes and R vanishes at no cut point. ``root_signs`` requires the square-free part D of R (R itself,
   made primitive, when that divides S) to divide S exactly. Then every real root of Q is a point: those of C are cut
   points and those of R are roots of D, hence of S. So Q keeps one sign on each open cell, the sign of Q(rho_k). At
   a cut point alpha_k = q_j it vanishes exactly when Q(q_j) = 0. Any other alpha_k is a root of S at which C does
   not vanish, and D, square-free with its real roots among those of S, has no root in (rho_(k-1), rho_k) but
   possibly alpha_k, a simple one: Q(alpha_k) = 0 exactly when D(rho_(k-1)) and D(rho_k) differ in sign. Where Q
   does not vanish at alpha_k it has no zero on [alpha_k, rho_k], and its sign at alpha_k is its sign at rho_k.
3. Truth. The caller states whether its relation holds at each alpha_k and on each open cell. This is the caller's
   part of the proof: the relation must be constant on every open cell, as it is when the signs of polynomials of
   item 2 decide it, so that one evaluation at rho_k speaks for the cell (``relation_truth`` and ``truth`` give the
   values for such polynomials). The truth set T is then the union of the true cells and the true points, and
   ``reconstruct`` lists the maximal runs of true members in the order cell 0, alpha_1, cell 1, ..., alpha_r,
   cell r: intervals whose union is T, sorted, disjoint and not mergeable.
4. Claim. ``matches`` proves every finite endpoint of the claimed set equal to a point. A Rational q is alpha_k when
   it is the cut point at alpha_k, found by bisection among the cut points, or when S(q) = 0 and
   rho_(k-1) < q < rho_k: alpha_k is the only point in between. A RealAlgebraicRoot(P, i), P primitive with a
   positive leading coefficient, is read as Q is in item 2, with each zero at a cut point divided out once: the rest
   must vanish at no cut point and divide S exactly. Then P is square-free, as the class requires, its real roots
   are the points at which it vanishes, and its root number i is the (i + 1)-th of them in increasing order. With
   every endpoint a point number, the order and the equality of endpoints are those of integers, and the claimed set
   is accepted only when its intervals are exactly the intervals of item 3. Hence the claimed set is T, and it is
   written in the one canonical form.

So a RealAlgebraicRoot endpoint is located only when its polynomial, without its zeros at cut points, divides S: a
correct name with a factor outside the cut, such as sqrt 3 as a root of (x^2 - 3)(x^2 + 1), is rejected. The checker
of a set operation passes its inputs and the claimed set to ``endpoint_cut``, which makes every endpoint a point. An
equation checker builds H from the polynomials of its problem and the master polynomial of the claim's algebraic
endpoints, and takes the claim's Rational endpoints as cut points (``endpoint_cut`` of the claim, joined by
``master_polynomial`` and ``merge_cuts``). Points added so only refine the cut: the relation stays constant on every
new cell, and its truth at a new point comes from the signs there as at any other point, so a correct claim is
accepted on the finer cut and a wrong one is still rejected.

The index convention of mathforge.algebraic: RealAlgebraicRoot(integer_coefficients, real_index) holds the primitive
square-free polynomial with a positive leading coefficient that the class stores, lowest degree first, and the
number of the root among its distinct real roots in increasing order, counted from 0. Here points are numbered from
1, as alpha_k is, and cells from 0.

Capacity under the default ComputationLimits (2,000,000 work units), measured 2026-11-12 with CPython on one machine,
with the checkers of catalogue 2.3 written on this module; each figure is the check alone, without the producer's
work. A set operation on two sets of 24 intervals each, the grid holding the endpoints of the inputs and of the claim
(96 cut points), needs at most 3,000 units (1 ms) with decimal endpoints k/100, |k| <= 5000, and with random 16-bit
rationals alike: rational endpoints are cut points, and no polynomial is formed from them. (The first engine made each
of them a linear factor of H and ran out of the budget on two sets of 12 intervals.) A solve_abs check of
sum_(i=1..20) |x - a_i| <= c with a_i = k/100, the rational roots of the atoms and of the pieces split off as cut
points, needs about 17,600 units (5 ms); its work grows with the square of the number of atoms, to about 250,000
units for 80. Endpoints at roots of Chebyshev's T_60 need about 13,000 units (2 ms), and at roots of T_100 about
34,000: H changes sign between the samples at each of its roots, and no Sturm chain is built. A master polynomial with
non-real roots needs its chain, whose coefficients can grow fast: Cells on T_60 (x^2 + 1) needs about 134,000 units
(30 ms), almost all of them for the chain; the chain of T_80 alone costs about 575,000 units and that of T_100 does
not fit, while Cells on the product of x^3 - k for k = 2, ..., 21 (degree 60, 20 real roots) needs about 20,000.

Everything fails closed. Malformed evidence, a malformed or non-canonical claim, a value of the wrong exact type (a
bool, a float, a str or a list where an int, a Rational or a tuple belongs) and a step that cannot be verified raise
``Rejected``; ``matches`` and ``check_samples`` answer False instead, and a checker built on the other calls catches
``Rejected`` and answers False. Integers of evidence and claims are bounded by max_integer_bits before any work on
them, and every length is charged to the budget before the work it causes. ``Cells`` also bounds the number of
samples by the degree of H and the number of cut points, ``locate`` the length of an endpoint's polynomial by the
same, and ``index_form`` the number of claimed intervals by the number of cells. ``endpoint_cut`` bounds neither the
number of intervals it reads nor the length of an endpoint's polynomial: it charges them as it reads, so a set checker
bounds the claimed set's interval count by its inputs' before calling it. Every call runs in the active computation
budget, or in one of its own, and a computation that outgrows the budget raises ResourceLimitError, which the
verifier reports as exhausted resources.
"""

from __future__ import annotations

from functools import cmp_to_key
from math import gcd

from ..limits import ComputationLimits, ResourceLimitError, computation, current_budget
from ..model import Rational, trusted_rational
from ..trusted import upoly as tp
from . import upoly as up

__all__ = ["Cells", "RELATIONS", "Rejected", "check_samples", "endpoint_cut", "holds", "integer_polynomial",
           "master_polynomial", "merge_cuts", "split_rational_roots"]

RELATIONS = ("=", "<", "<=", ">", ">=")  # The values of the relation kind, each read against 0.
_HOLDS = {"=": (False, True, False), "<": (True, False, False), "<=": (True, True, False),
          ">": (False, False, True), ">=": (False, True, True)}  # The truth for the signs -1, 0 and 1.

# The record classes read here, by module and qualified name: this package never imports them.
_INTERVAL_SET = ("mathforge.intervals", "IntervalSet")
_INTERVAL = ("mathforge.intervals", "Interval")
_ALGEBRAIC_ROOT = ("mathforge.algebraic", "RealAlgebraicRoot")
_MISSING = object()  # An attribute that a forged record lacks (None is an infinite end).
_DEFAULT_BITS = ComputationLimits().max_integer_bits


class Rejected(Exception):
    """The evidence or the claim does not verify: a checker that meets it answers False."""


def holds(relation, sign) -> bool:
    """Whether a real number of the given sign (-1, 0 or 1) satisfies ``number relation 0``, for a relation of
    RELATIONS; Rejected for any other relation or sign. With the signs of ``root_signs`` and ``sample_signs`` it
    gives the truth values of a polynomial relation."""
    if type(relation) is not str or relation not in _HOLDS or type(sign) is not int or not -1 <= sign <= 1:
        raise Rejected("A relation of RELATIONS and a sign -1, 0 or 1 are required")
    return _HOLDS[relation][sign + 1]


# ---------------------------------------------------------------------------------------------------------------
# Plain data, validated

def _named(value) -> tuple[str, str]:
    kind = type(value)
    return kind.__module__, kind.__qualname__


def _limit() -> int:
    budget = current_budget()
    return _DEFAULT_BITS if budget is None else budget.limits.max_integer_bits


def _polynomial(value) -> tuple[int, ...]:
    """value, when it is a polynomial of the checker: a tuple of exact ints without a trailing zero. Coefficients
    longer than max_integer_bits are refused as the library refuses every such integer (ResourceLimitError)."""
    if type(value) is not tuple:
        raise Rejected("A polynomial is a tuple of ints")
    tp.charge(len(value))
    for coefficient in value:
        if type(coefficient) is not int:
            raise Rejected("A polynomial is a tuple of ints")
    if value and not value[-1]:
        raise Rejected("A polynomial has no trailing zero")
    if tp.max_bits(value) > _limit():
        raise ResourceLimitError("Computation limit exceeded: integer bits")
    return value


def _rational(value) -> Rational:
    """value, when it is a canonical Rational of evidence or of a claim: exact ints in lowest terms, a positive
    denominator, neither longer than max_integer_bits (tested before the gcd runs)."""
    if type(value) is not Rational:
        raise Rejected("An exact Rational is required")
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise Rejected("A Rational holds exact ints and a positive denominator")
    bits = max(numerator.bit_length(), denominator.bit_length())
    if bits > _limit():
        raise Rejected("An integer is longer than max_integer_bits")
    if denominator != 1:
        tp.charge_gcd(bits)
        if gcd(numerator, denominator) != 1:
            raise Rejected("A Rational is in lowest terms")
    return value


def _rationals(values, label: str) -> tuple[Rational, ...]:
    """values, when it is a strictly increasing tuple of canonical Rationals (samples, cut points, candidates)."""
    if type(values) is not tuple:
        raise Rejected(label + " are a tuple of Rationals")
    tp.charge(2 * len(values))
    for value in values:
        _rational(value)
    for lower, upper in zip(values, values[1:]):
        if not _less(lower, upper):
            raise Rejected(label + " are not strictly increasing")
    return values


def _bits(value: Rational) -> int:
    return max(value.numerator.bit_length(), value.denominator.bit_length())


def _less(a: Rational, b: Rational) -> bool:
    """a < b for canonical Rationals, on their integers; the two products are charged by their size."""
    tp.charge_product(_bits(a), _bits(b), 2)
    return a.numerator * b.denominator < b.numerator * a.denominator


def _same(a: Rational, b: Rational) -> bool:
    """a == b for canonical Rationals: their lowest terms are equal."""
    return a.numerator == b.numerator and a.denominator == b.denominator


def _order(a: Rational, b: Rational) -> int:
    """-1, 0 or 1 as a < b, a == b or a > b, for canonical Rationals; the caller charges the comparisons."""
    left, right = a.numerator * b.denominator, b.numerator * a.denominator
    return (left > right) - (left < right)


def _sorted_points(points: list) -> tuple[Rational, ...]:
    """The distinct values among canonical Rationals, in increasing order. A canonical Rational is its lowest terms,
    so equal values are equal pairs of integers; the sort, at most n * bits(n) comparisons, is charged before it
    runs."""
    distinct: dict = {}
    for point in points:
        distinct.setdefault((point.numerator, point.denominator), point)
    values = list(distinct.values())
    if len(values) > 1:
        comparisons = len(values) * len(values).bit_length()
        tp.charge(comparisons)
        longest = max(map(_bits, values))
        tp.charge_product(longest, longest, 2 * comparisons)
        values.sort(key=cmp_to_key(_order))
    return tuple(values)


def _root_fields(record, longest: int | None = None) -> tuple[tuple[int, ...], int]:
    """(polynomial, index) of a RealAlgebraicRoot record, as the class stores them: a nonconstant primitive
    polynomial with a positive leading coefficient, and an index from 0 below its degree. That the polynomial is
    square-free is left to the caller, which divides the square-free master polynomial by it. longest bounds the
    number of coefficients before any of them is read."""
    coefficients = getattr(record, "integer_coefficients", None)
    index = getattr(record, "real_index", None)
    if (type(coefficients) is not tuple or len(coefficients) < 2 or type(index) is not int
            or not 0 <= index < len(coefficients) - 1 or (longest is not None and len(coefficients) > longest)):
        raise Rejected("A RealAlgebraicRoot holds a nonconstant integer tuple and an index below its degree")
    tp.charge(len(coefficients))
    for coefficient in coefficients:
        if type(coefficient) is not int:
            raise Rejected("A RealAlgebraicRoot holds exact ints")
    bits = tp.max_bits(coefficients)
    if bits > _limit():
        raise Rejected("An integer is longer than max_integer_bits")
    tp.charge_gcd(bits, len(coefficients))
    if coefficients[-1] <= 0 or gcd(*coefficients) != 1:
        raise Rejected("A RealAlgebraicRoot holds a primitive polynomial with a positive leading coefficient")
    return coefficients, index


def _interval_records(value) -> tuple:
    """The entries of an IntervalSet record, read once."""
    if _named(value) != _INTERVAL_SET:
        raise Rejected("An IntervalSet record is required")
    records = getattr(value, "intervals", None)
    if type(records) is not tuple:
        raise Rejected("An IntervalSet holds a tuple of intervals")
    return records


def _interval_fields(record) -> tuple:
    """(lower, upper, left_closed, right_closed) of an Interval record, each read once; the flags are bools."""
    if _named(record) != _INTERVAL:
        raise Rejected("An IntervalSet holds Interval records")
    lower, upper = getattr(record, "lower", _MISSING), getattr(record, "upper", _MISSING)
    left_closed, right_closed = getattr(record, "left_closed", None), getattr(record, "right_closed", None)
    if type(left_closed) is not bool or type(right_closed) is not bool:
        raise Rejected("Closure flags are bools")
    return lower, upper, left_closed, right_closed


# ---------------------------------------------------------------------------------------------------------------
# The checker's own algebra (certify.upoly), exact evaluation and Sturm counts

def _primitive(polynomial: tuple[int, ...]) -> tuple[int, ...]:
    """The polynomial divided by its content, with a positive leading coefficient; the gcd is charged first."""
    tp.charge_gcd(tp.max_bits(polynomial), len(polynomial))
    return tp.primitive(polynomial)


def _squarefree(polynomial: tuple[int, ...]) -> tuple[int, ...]:
    """The primitive square-free part, with a positive leading coefficient, of a nonzero polynomial ((1,) for a
    constant), by Euclid."""
    try:
        return up.squarefree_part(polynomial)
    except ArithmeticError as error:  # A division of Euclid's algorithm that is not exact certifies nothing.
        raise Rejected("No square-free part") from error


def _chain(part: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """The checker's Sturm chain of a primitive polynomial of degree >= 1 (certify.upoly.sturm_chain): the
    subresultant remainders of the polynomial and its derivative, signed as a Sturm sequence, whose last member is a
    constant multiple of their gcd. Only the signs of its members at minus and plus infinity are read, never their
    values at a point."""
    try:
        return up.sturm_chain(part)
    except ArithmeticError as error:  # A chain that does not divide exactly certifies nothing.
        raise Rejected("No Sturm chain") from error


def _square_free_chain(part: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """(S, chain): the square-free part S of a primitive polynomial of degree >= 1 with a positive leading
    coefficient, and S's Sturm chain.

    No separate square-free pass runs (a master polynomial usually is square-free already): the chain of the
    polynomial ends in a constant multiple of the gcd of the polynomial and its derivative, so a constant last member
    proves the polynomial square-free. Otherwise S is the quotient by that gcd, an exact division over Z by a
    primitive divisor (Gauss), and S's own chain must end in a constant."""
    chain = _chain(part)
    if len(chain[-1]) > 1:
        division = tp.divmod_exact(part, _primitive(chain[-1]))
        if division is None or division[1] or len(division[0]) < 2:
            raise Rejected("No square-free part")
        part = division[0]
        chain = _chain(part)
        if len(chain[-1]) > 1:
            raise Rejected("No square-free part")
    return part, chain


def _separated(polynomial: tuple[int, ...], samples, cuts) -> tuple:
    """(S, signs, new, m) for a nonzero polynomial H, samples and cut points (item 1 of the module docstring): S the
    primitive square-free part of H with a positive leading coefficient, signs its signs at the samples, new whether
    each cut point is no root of it, and m its number of distinct real roots, verified to be the number of sign
    changes between neighbouring samples. Rejected when a sample is a root, when the samples are not one more than
    the points, or when the sign changes are not all the real roots. A constant has no root: for a constant H the
    result is ((1,), all signs 1, every cut point new, 0) whatever the samples, and the number of samples is left to
    Cells.__init__, which requires exactly one more than the cut points.

    A polynomial of degree d that changes sign d times between the samples has d distinct real roots: it is S and
    needs no Sturm chain. Otherwise the chain (``_square_free_chain``) gives S, and its variations at minus and plus
    infinity give m; the chain is never evaluated at a point. A wrong number of samples is rejected before any of
    them is evaluated, unless it is the number that the d roots of H would need."""
    part = _primitive(polynomial)
    if len(part) < 2:
        return (1,), (1,) * len(samples), (True,) * len(cuts), 0
    bits = tp.max_bits(part)
    tp.charge(len(cuts))
    new = tuple([_sign_at(part, cut, bits) != 0 for cut in cuts])  # H and S have the same roots.
    signs = None
    if len(samples) == len(part) + sum(new):  # One sample more than d roots and the new cut points.
        signs = _signs(part, samples, bits)
        if _changes(signs) == len(part) - 1:
            return part, signs, new, len(part) - 1
    square_free, chain = _square_free_chain(part)
    count = _variations_at_infinity(chain, False) - _variations_at_infinity(chain, True)
    if len(samples) != count + sum(new) + 1:
        raise Rejected("There is not exactly one sample more than points")
    if signs is None or square_free != part:
        signs = _signs(square_free, samples, tp.max_bits(square_free))
    if _changes(signs) != count:
        raise Rejected("The samples do not separate the real roots of the master polynomial")
    return square_free, signs, new, count


def _signs(polynomial: tuple[int, ...], samples, bits: int) -> tuple[int, ...]:
    """The signs of a master polynomial at the samples; Rejected when a sample is a root."""
    signs = tuple([_sign_at(polynomial, sample, bits) for sample in samples])
    if not all(signs):
        raise Rejected("A sample is a root of the master polynomial")
    return signs


def _sign_at(polynomial: tuple[int, ...], point: Rational, bits: int | None = None) -> int:
    """The sign of the polynomial at a rational point, by exact evaluation (trusted.upoly.sign_at).

    trusted.upoly.sign_at charges by the size of the point alone; here the size of the coefficients is charged as
    well, before the evaluation runs. Each Horner step multiplies the running value, at least as long as a
    coefficient, by the numerator, and at a fraction a coefficient by a power of the denominator, up to the length
    times as long as the point; a coefficient longer than 16,384 bits also costs a linear pass of its own. bits is
    the bit length of the largest coefficient, which a caller that evaluates one polynomial at many points passes."""
    if not polynomial:
        return 0
    length = len(polynomial)
    if bits is None:
        bits = tp.max_bits(polynomial)
    numerator, denominator = point.numerator, point.denominator
    point_bits = max(numerator.bit_length(), denominator.bit_length())
    tp.charge(length * (bits >> 14))
    tp.charge_product(bits, point_bits, length)
    if denominator != 1:
        tp.charge_product(bits, length * point_bits, length)
    return tp.sign_at(polynomial, numerator, denominator)


def _divide_out(polynomial: tuple[int, ...], root: Rational) -> tuple[int, ...]:
    """polynomial / (den * x - num) for a rational root num/den of it. The divisor is primitive with a positive
    leading coefficient, so the division, exact over Q, is exact over Z (Gauss); each step divides a coefficient by
    the denominator, a linear pass over a long coefficient that is charged here as well."""
    tp.charge(len(polynomial) * (tp.max_bits(polynomial) >> 13))
    division = tp.divmod_exact(polynomial, (-root.numerator, root.denominator))
    if division is None or division[1]:
        raise Rejected("A rational root does not divide out")  # Exact evaluation said it is a root.
    return division[0]


def _variations(signs) -> int:
    """The sign changes along a sequence of signs; zeros are skipped."""
    changes = previous = 0
    for sign in signs:
        if sign:
            if sign == -previous:
                changes += 1
            previous = sign
    return changes


def _variations_at_infinity(chain, positive: bool) -> int:
    """The sign variations of a chain at plus or minus infinity: each member has the sign of its leading
    coefficient, reversed at minus infinity when its degree is odd."""
    tp.charge(len(chain))
    signs = []
    for member in chain:
        sign = 1 if member[-1] > 0 else -1
        signs.append(sign if positive or len(member) % 2 else -sign)
    return _variations(signs)


def _changes(signs) -> int:
    """The number of neighbours with different signs in a sequence of nonzero signs."""
    tp.charge(len(signs))
    return sum([1 for left, right in zip(signs, signs[1:]) if left != right])


# ---------------------------------------------------------------------------------------------------------------
# Master polynomials and cut points

def _master(polynomials) -> tuple[int, ...]:
    """The square-free part of the product of the polynomials of an iterable. They are multiplied in as they
    arrive, so that a long forged list stops at the budget before it is read to its end, and equal ones once."""
    limit = _limit()
    product: tuple[int, ...] = (1,)
    seen = set()
    for polynomial in polynomials:
        polynomial = _polynomial(polynomial)
        tp.charge_gcd(tp.max_bits(polynomial), len(polynomial))
        factor = tp.primitive(polynomial)
        if len(factor) < 2 or factor in seen:
            continue
        seen.add(factor)
        product = tp.mul(product, factor)
        if tp.max_bits(product) > limit:
            raise ResourceLimitError("Computation limit exceeded: integer bits")
    return _squarefree(product)


def master_polynomial(*polynomials) -> tuple[int, ...]:
    """The primitive square-free polynomial, with a positive leading coefficient, whose roots are the roots of the
    given polynomials that are not zero: the square-free part of their product, (1,) when none of them has a root.

    The zero polynomial and constants add no root and are skipped; the zero polynomial has the sign 0 everywhere,
    which ``sample_signs`` and ``root_signs`` answer without it. The result is sign-normalized: it serves as a
    master polynomial and must never give a sign. Rational roots are better kept out of it as cut points
    (``split_rational_roots``): each one lengthens every member of the Sturm chain."""
    with computation():
        return _master(polynomials)


def split_rational_roots(polynomial, candidates) -> tuple[tuple[Rational, ...], tuple[int, ...]]:
    """(roots, rest): the candidates, verified to be roots of the polynomial, and what remains of the polynomial
    when each of them is divided out as often as it divides, so that rest vanishes at none of them.

    candidates is a strictly increasing tuple of canonical Rationals, typically from a producer's evidence: each must
    be a root, which exact evaluation decides, and anything else is Rejected, as are more candidates than the degree.
    The divisors den * x - num are primitive, so polynomial == rest * prod (den * x - num)**multiplicity holds over
    Z (Gauss). The roots become cut points (``merge_cuts``) and rest goes into the master polynomial, so that the
    Sturm chain is built for rest alone. rest is a quotient, not sign-preserving: it never gives a sign. The zero
    polynomial has no finite root set: ((), ()) without candidates, Rejected with any."""
    with computation():
        polynomial = _polynomial(polynomial)
        if type(candidates) is not tuple:
            raise Rejected("The candidates are a tuple of Rationals")
        if not polynomial:
            if candidates:
                raise Rejected("The zero polynomial has no finite root set")
            return (), ()
        if len(candidates) >= len(polynomial):  # A nonzero polynomial of degree d has at most d roots.
            raise Rejected("More candidates than the degree allows")
        roots = _rationals(candidates, "The candidates")
        rest = polynomial
        for root in roots:
            if len(rest) < 2 or _sign_at(rest, root):
                raise Rejected("A candidate is not a root of the polynomial")
            rest = _divide_out(rest, root)
            while len(rest) > 1 and not _sign_at(rest, root):
                rest = _divide_out(rest, root)
        return roots, rest


def merge_cuts(*groups) -> tuple[Rational, ...]:
    """The cut points of several groups, each a tuple of canonical Rationals in any order: their values in increasing
    order, each once, as Cells takes them. Rejected for anything else in place of a group or a point; the sort is
    charged before it runs."""
    with computation():
        points = []
        for group in groups:
            if type(group) is not tuple:
                raise Rejected("A group of cut points is a tuple of Rationals")
            tp.charge(len(group))
            for point in group:
                points.append(_rational(point))
        return _sorted_points(points)


def endpoint_cut(*sets) -> tuple[tuple[int, ...], tuple[Rational, ...]]:
    """(master, cuts) of the finite endpoints of IntervalSet records, for Cells(master, samples, cuts): master is the
    master polynomial (``master_polynomial``) of the polynomials of their RealAlgebraicRoot endpoints, (1,) when there
    is none, and cuts their Rational endpoints in increasing order, each once. A RealAlgebraicRoot of a linear
    polynomial a + b * x is the rational -a/b and joins the cut points.

    The checker of a set operation passes its input sets and the claimed set, so that every endpoint is a point. An
    equation checker passes the claimed set and joins the result to its own polynomials and cut points (module
    docstring). The records are read as data and need not be canonical here: ``index_form`` enforces that. Neither
    the number of intervals nor the length of an endpoint's polynomial is bounded here: both are charged as they are
    read, and a set checker bounds the claimed set's interval count by its inputs' before calling this. master is
    sign-normalized and never gives a sign."""
    with computation():
        polynomials = []
        points = []
        for value in sets:
            for record in _interval_records(value):
                tp.charge(1)
                for end in _interval_fields(record)[:2]:
                    if end is None:
                        continue
                    if type(end) is Rational:
                        points.append(_rational(end))
                    elif _named(end) == _ALGEBRAIC_ROOT:
                        coefficients = _root_fields(end)[0]
                        if len(coefficients) == 2:  # Primitive with b > 0: -a/b is in lowest terms.
                            points.append(trusted_rational(-coefficients[0], coefficients[1]))
                        else:
                            polynomials.append(coefficients)
                    else:
                        raise Rejected("An endpoint is a Rational, a RealAlgebraicRoot or None")
        return _master(polynomials), _sorted_points(points)


def integer_polynomial(coefficients) -> tuple[int, ...]:
    """The integer polynomial with the sign of the polynomial of the given coefficients, a tuple of exact ints and
    canonical Rationals, lowest degree first: their values times the least common denominator, divided by the
    POSITIVE content, so that the result is a positive multiple of the given polynomial and has its sign at every
    point; () for the zero polynomial, trailing zeros dropped.

    This is the form a checker reads signs from. Normalizing with trusted.upoly.primitive, as master_polynomial and
    Cells.master do, makes the leading coefficient positive and so reverses the sign of a polynomial whose leading
    coefficient is negative. An integer longer than max_integer_bits is Rejected; a common denominator that grows
    longer raises ResourceLimitError."""
    with computation():
        if type(coefficients) is not tuple:
            raise Rejected("Coefficients are a tuple of ints and Rationals")
        tp.charge(len(coefficients))
        limit = _limit()
        pairs = []
        for value in coefficients:
            if type(value) is int:
                if value.bit_length() > limit:
                    raise Rejected("An integer is longer than max_integer_bits")
                pairs.append((value, 1))
            else:
                value = _rational(value)
                pairs.append((value.numerator, value.denominator))
        denominator = 1
        for _, den in pairs:
            if den != 1 and denominator % den:
                tp.charge_gcd(max(denominator.bit_length(), den.bit_length()))
                denominator = denominator // gcd(denominator, den) * den
                if denominator.bit_length() > limit:
                    raise ResourceLimitError("Computation limit exceeded: integer bits")
        longest = max([numerator.bit_length() for numerator, _ in pairs], default=0)
        tp.charge_product(longest, denominator.bit_length(), len(pairs))
        values = [numerator * (denominator // den) for numerator, den in pairs]
        while values and not values[-1]:
            values.pop()
        if not values:
            return ()
        bits = tp.max_bits(values)
        tp.charge_gcd(bits, len(values))
        content = gcd(*values)  # Positive: dividing by it keeps every sign.
        values = [value // content for value in values]
        if bits > limit and tp.max_bits(values) > limit:
            raise ResourceLimitError("Computation limit exceeded: integer bits")
        return tuple(values)


# ---------------------------------------------------------------------------------------------------------------
# The cut

class Cells:
    """The real line cut at the points alpha_1 < ... < alpha_r, the real roots of a master polynomial together with
    rational cut points, certified by rational samples rho_0 < ... < rho_r (item 1 of the module docstring): the open
    cell number k holds rho_k, between alpha_k and alpha_(k+1), unbounded below for k = 0 and above for k = r.

    master is the primitive square-free part of the polynomial given, cuts the cut points, samples the samples and
    root_count the number r of points. Construction raises Rejected for a zero or malformed polynomial, for cut
    points that are not a strictly increasing tuple of canonical Rationals, and for samples that are not a tuple of
    exactly r + 1 canonical Rationals, strictly increasing, none of them a point, the k-th with exactly k points below
    it. A cut point at which the master polynomial vanishes is that root, one point.

    In the methods, a point is named by its number k from 1 to r and a cell or a sample by its number from 0 to r;
    a tuple over the points holds the entry of alpha_k at position k - 1. The four attributes are read-only: the
    answers rest on the samples having been verified for that master polynomial and those cut points."""

    __slots__ = ("_master", "_cuts", "_samples", "_root_count", "_numbers", "_cut_of", "_sample_signs",
                 "_root_signs", "_vanishing")

    def __init__(self, master, samples, cuts=()) -> None:
        with computation():
            polynomial = _polynomial(master)
            if not polynomial:
                raise Rejected("The zero polynomial has no finite root set")
            if type(cuts) is not tuple:
                raise Rejected("The cut points are a tuple of Rationals")
            # The evidence is bounded before any work on it: r + 1 <= degree + cut points + 1 samples.
            if type(samples) is not tuple or not 1 <= len(samples) <= len(polynomial) + len(cuts):
                raise Rejected("The samples are a tuple of at most degree + cut points + 1 Rationals")
            _rationals(cuts, "The cut points")
            _rationals(samples, "The samples")
            part, signs, new, count = _separated(polynomial, samples, cuts)
            root_count = count + sum(new)
            if len(samples) != root_count + 1:
                raise Rejected("There is not exactly one sample more than points")
            numbers = [0] * len(cuts)
            position = passed = roots = 0  # Below the sample: cut points, those of them that are no root, roots.
            for k, sample in enumerate(samples):
                while position < len(cuts) and _less(cuts[position], sample):
                    numbers[position] = k  # rho_(k-1) < q < rho_k: q is alpha_k when the counts hold.
                    passed += new[position]
                    position += 1
                if position < len(cuts) and _same(cuts[position], sample):
                    raise Rejected("A sample is a cut point")
                if k and signs[k - 1] != signs[k]:  # The one root of S between rho_(k-1) and rho_k (item 1).
                    roots += 1
                if roots + passed != k:
                    raise Rejected("A sample does not have exactly its number of points below it")
            cut_of: list = [None] * root_count
            for index, k in enumerate(numbers):  # Each cut point lies between two samples, alone (item 1).
                if not 1 <= k <= root_count or cut_of[k - 1] is not None:
                    raise Rejected("The cut points and the samples do not interleave")
                cut_of[k - 1] = index
            self._master = part
            self._cuts = cuts
            self._samples = samples
            self._root_count = root_count
            self._numbers = tuple(numbers)  # The number k of each cut point.
            self._cut_of = tuple(cut_of)  # The index of the cut point at alpha_k, None at a root that is no cut point.
            self._sample_signs: dict = {}
            self._root_signs: dict = {}
            self._vanishing: dict = {}

    @property
    def master(self) -> tuple[int, ...]:
        """The primitive square-free part of the polynomial given, with a positive leading coefficient: a master
        polynomial, sign-normalized, never a source of signs."""
        return self._master

    @property
    def cuts(self) -> tuple[Rational, ...]:
        """The verified cut points q_1 < ... < q_s."""
        return self._cuts

    @property
    def samples(self) -> tuple[Rational, ...]:
        """The verified samples rho_0, ..., rho_r."""
        return self._samples

    @property
    def root_count(self) -> int:
        """r, the number of points: the distinct real roots of the master polynomial and the cut points, a cut point
        at a root counted once."""
        return self._root_count

    # --- Signs ----------------------------------------------------------------------------------------------------

    def sample_signs(self, polynomial) -> tuple[int, ...]:
        """The signs (-1, 0 or 1) of the polynomial at rho_0, ..., rho_r, by exact evaluation. Any polynomial is
        accepted here: a sample is a rational point."""
        with computation():
            polynomial = _polynomial(polynomial)
            signs = self._sample_signs.get(polynomial)
            if signs is None:
                bits = tp.max_bits(polynomial)
                signs = tuple([_sign_at(polynomial, sample, bits) for sample in self._samples])
                self._sample_signs[polynomial] = signs
            return signs

    def root_signs(self, polynomial) -> tuple[int, ...]:
        """The signs of the polynomial at alpha_1, ..., alpha_r.

        The polynomial is zero (sign 0 everywhere), or every real root of it is a point as item 2 of the module
        docstring verifies it: with its zeros at the cut points divided out, the square-free part of the rest divides
        the master polynomial exactly. Rejected otherwise; a nonzero constant qualifies. The sign is 0 where the
        polynomial vanishes and its sign at rho_k elsewhere."""
        with computation():
            polynomial = _polynomial(polynomial)
            signs = self._root_signs.get(polynomial)
            if signs is None:
                signs = self._root_signs[polynomial] = self._signs_at_roots(polynomial)
            return signs

    def sign_at_sample(self, polynomial, k: int) -> int:
        """The sign of the polynomial at rho_k, 0 <= k <= r."""
        if type(k) is not int or not 0 <= k <= self.root_count:
            raise Rejected("No such sample")
        return self.sample_signs(polynomial)[k]

    def sign_at_root(self, polynomial, k: int) -> int:
        """The sign of the polynomial at alpha_k, 1 <= k <= r, under the condition of root_signs."""
        if type(k) is not int or not 1 <= k <= self.root_count:
            raise Rejected("No such point")
        return self.root_signs(polynomial)[k - 1]

    def relation_truth(self, polynomial, relation) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """(root_truth, cell_truth) of ``polynomial relation 0``, for a relation of RELATIONS, in the layout of
        reconstruct: holds() of the signs at alpha_1, ..., alpha_r and at rho_0, ..., rho_r.

        The polynomial meets the condition of root_signs, which is tested first; Rejected otherwise, and for any
        other relation. Its roots are then points, so it keeps on each open cell the sign it has at the cell's
        sample, and the truth at the sample is the truth on the whole cell."""
        with computation():
            if type(relation) is not str or relation not in _HOLDS:
                raise Rejected("A relation of RELATIONS is required")
            table = _HOLDS[relation]
            at_roots = self.root_signs(polynomial)
            at_samples = self.sample_signs(polynomial)
            return tuple([table[sign + 1] for sign in at_roots]), tuple([table[sign + 1] for sign in at_samples])

    def truth(self, decide, *polynomials) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """(root_truth, cell_truth) in the layout of reconstruct: decide(signs) at each alpha_k and at each sample
        rho_k, where signs is the tuple of the signs of the given polynomials at that point (the empty tuple when
        none is given) and decide answers a bool; Rejected for any other answer.

        Each polynomial meets the condition of root_signs, tested first. Each then keeps one sign on every open cell,
        so decide, a function of the signs alone, has one value on the cell: its value at the sample. This is how a
        relation decided by the signs of several polynomials is evaluated, such as A * sqrt(P) + B against 0 from
        the signs of P, A, B and A^2 P - B^2."""
        with computation():
            if not callable(decide):
                raise Rejected("decide is a function of the signs")
            at_roots = [self.root_signs(polynomial) for polynomial in polynomials]
            at_samples = [self.sample_signs(polynomial) for polynomial in polynomials]
            r = self.root_count
            tp.charge((2 * r + 1) * (1 + len(polynomials)))
            root_truth, cell_truth = [], []
            for signs, out, size in ((at_roots, root_truth, r), (at_samples, cell_truth, r + 1)):
                for k in range(size):
                    value = decide(tuple([column[k] for column in signs]))
                    if type(value) is not bool:
                        raise Rejected("decide answers a bool")
                    out.append(value)
            return tuple(root_truth), tuple(cell_truth)

    def _signs_at_roots(self, polynomial: tuple[int, ...]) -> tuple[int, ...]:
        if not polynomial:
            return (0,) * self.root_count
        at_samples = self.sample_signs(polynomial)
        zeros = self._zeros(polynomial)[0]
        tp.charge(self.root_count)
        signs = []
        for k in range(1, self.root_count + 1):
            if zeros[k - 1]:
                signs.append(0)
            elif at_samples[k] and at_samples[k - 1] == at_samples[k]:
                signs.append(at_samples[k])
            else:  # Without a zero in between, the polynomial has one nonzero sign from rho_(k-1) to rho_k.
                raise Rejected("The signs at two samples contradict the zeros between them")
        return tuple(signs)

    def _divides(self, divisor: tuple[int, ...]) -> bool:
        """Whether a primitive polynomial with a positive leading coefficient divides the master polynomial exactly:
        exact over Z precisely when exact over Q (Gauss)."""
        if divisor == self._master:
            return True
        if len(divisor) > len(self._master):
            return False
        division = tp.divmod_exact(self._master, divisor)
        return division is not None and not division[1]

    def _zeros(self, polynomial: tuple[int, ...]) -> tuple[tuple[bool, ...], tuple[int, ...], bool]:
        """(zeros, numbers, simple) of a nonzero polynomial whose real roots are all points, as item 2 of the module
        docstring verifies it; Rejected otherwise.

        zeros[k - 1] tells whether the polynomial vanishes at alpha_k, and numbers lists those k in increasing
        order. simple tells whether it has the form of a RealAlgebraicRoot (item 4): each zero at a cut point
        simple, and the rest, made primitive, dividing the master polynomial, so that the polynomial is square-free.
        The values are cached per polynomial once complete, so a failed check leaves nothing behind."""
        found = self._vanishing.get(polynomial)
        if found is not None:
            return found
        r = self.root_count
        zeros = [False] * r
        simple = True
        rest = polynomial
        for index, cut in enumerate(self._cuts):
            if len(rest) < 2:  # A nonzero constant vanishes nowhere.
                break
            if _sign_at(rest, cut):
                continue
            zeros[self._numbers[index] - 1] = True
            rest = _divide_out(rest, cut)
            if len(rest) > 1 and not _sign_at(rest, cut):
                simple = False
                while len(rest) > 1 and not _sign_at(rest, cut):
                    rest = _divide_out(rest, cut)
        if len(rest) > 1:
            divisor = _primitive(rest)
            if not self._divides(divisor):
                simple = False
                divisor = _squarefree(rest)
                if not self._divides(divisor):
                    raise Rejected("A real root of the polynomial is no point of the cut")
            bits = tp.max_bits(divisor)
            signs = [_sign_at(divisor, sample, bits) for sample in self._samples]
            tp.charge(r)
            if not all(signs):  # A divisor of the master polynomial vanishes at no sample.
                raise Rejected("A divisor of the master polynomial vanishes at a sample")
            for k in range(1, r + 1):
                changes = signs[k - 1] != signs[k]
                if self._cut_of[k - 1] is None:
                    zeros[k - 1] = changes  # alpha_k is a root of S, and the divisor's only possible root there.
                elif changes:  # The rest vanishes at no cut point, and alpha_k is the only point in between.
                    raise Rejected("A divisor of the master polynomial changes sign at a cut point")
        found = (tuple(zeros), tuple([k for k in range(1, r + 1) if zeros[k - 1]]), simple)
        self._vanishing[polynomial] = found
        return found

    # --- Endpoints ------------------------------------------------------------------------------------------------

    def locate(self, endpoint) -> int:
        """The number k, 1 <= k <= r, of the point alpha_k that the endpoint equals; Rejected when it equals none.

        A canonical Rational q equals alpha_k when it is the cut point at alpha_k, or when the master polynomial
        vanishes at q and rho_(k-1) < q < rho_k. A RealAlgebraicRoot record (P, i) equals alpha_k when P is primitive
        with a positive leading coefficient, P without its zeros at cut points (each divided out once, and then no
        zero there) divides the master polynomial exactly, and alpha_k is the (i + 1)-th point at which P vanishes
        (item 4 of the module docstring). So a correct name whose polynomial has a factor that is neither a cut point
        nor a factor of the master is rejected: an equation checker puts the claim's endpoints into its cut (module
        docstring). Any other value is rejected, an int and a float included."""
        with computation():
            if type(endpoint) is Rational:
                return self._locate_rational(_rational(endpoint))
            if _named(endpoint) == _ALGEBRAIC_ROOT:
                polynomial, index = _root_fields(endpoint, len(self._master) + len(self._cuts))
                _, numbers, simple = self._zeros(polynomial)
                if not simple:
                    raise Rejected("A RealAlgebraicRoot polynomial is square-free, its rest dividing the master")
                if index >= len(numbers):
                    raise Rejected("The index names no real root of the polynomial")
                return numbers[index]
            raise Rejected("An endpoint is a Rational or a RealAlgebraicRoot")

    def _locate_rational(self, q: Rational) -> int:
        cuts = self._cuts
        low, high = 0, len(cuts)  # The number of cut points below q, by bisection.
        while low < high:
            tp.charge(1)
            middle = (low + high) // 2
            if _less(cuts[middle], q):
                low = middle + 1
            else:
                high = middle
        if low < len(cuts) and _same(cuts[low], q):
            return self._numbers[low]
        if len(self._master) < 2 or _sign_at(self._master, q):
            raise Rejected("A rational endpoint is neither a cut point nor a root of the master polynomial")
        samples = self._samples
        low, high = 0, len(samples)  # The number of samples below q, by bisection.
        while low < high:
            tp.charge(1)
            middle = (low + high) // 2
            if _less(samples[middle], q):
                low = middle + 1
            else:
                high = middle
        if not 1 <= low <= self.root_count or not (_less(samples[low - 1], q) and _less(q, samples[low])):
            raise Rejected("A rational endpoint lies between no two samples")
        return low

    # --- Sets -----------------------------------------------------------------------------------------------------

    def reconstruct(self, root_truth, cell_truth) -> tuple[tuple[int | None, int | None, bool, bool], ...]:
        """The canonical interval set of the truth values, in index form: intervals (lower, upper, left_closed,
        right_closed) in increasing order, where an end is the number k of the point alpha_k, or None when infinite.

        root_truth is a tuple of r bools, the truth at alpha_1, ..., alpha_r; cell_truth a tuple of r + 1 bools, the
        truth on the open cells number 0, ..., r. The intervals are the maximal runs of true members in the order
        cell 0, alpha_1, cell 1, ..., alpha_r, cell r: a true cell joins a true neighbouring point, a true point
        between false cells is a closed interval of one point, a false point between true cells leaves two intervals
        open at it, and infinite ends are open."""
        with computation():
            r = self.root_count
            if (type(root_truth) is not tuple or type(cell_truth) is not tuple or len(root_truth) != r
                    or len(cell_truth) != r + 1):
                raise Rejected("Truth values are a tuple of r bools and a tuple of r + 1 bools")
            tp.charge(2 * r + 1)
            members = [cell_truth[0]]  # Member 2k is the cell number k, member 2k - 1 the point alpha_k.
            for k in range(1, r + 1):
                members.append(root_truth[k - 1])
                members.append(cell_truth[k])
            for member in members:
                if type(member) is not bool:
                    raise Rejected("Truth values are bools")
            last = 2 * r
            intervals = []
            start = 0
            while start <= last:
                if not members[start]:
                    start += 1
                    continue
                stop = start
                while stop < last and members[stop + 1]:
                    stop += 1
                # The run from start to stop is one interval: closed at an end that is a point of the run, open at
                # an end that bounds a cell of the run, which is the false point beyond it or infinity.
                if start % 2:
                    lower, left_closed = (start + 1) // 2, True
                else:
                    lower, left_closed = (start // 2 if start else None), False
                if stop % 2:
                    upper, right_closed = (stop + 1) // 2, True
                else:
                    upper, right_closed = (stop // 2 + 1 if stop < last else None), False
                intervals.append((lower, upper, left_closed, right_closed))
                start = stop + 1
            return tuple(intervals)

    def index_form(self, value) -> tuple[tuple[int | None, int | None, bool, bool], ...]:
        """The IntervalSet record in the index form of reconstruct; Rejected unless it is canonical.

        Canonical: a tuple of Interval records whose finite ends are points (``locate``) and whose flags are bools;
        an infinite end is open; no interval is reversed or empty (a single point is closed on both sides); and the
        intervals are sorted, disjoint and not mergeable: each begins after the end of the one before, or at that
        same point with both intervals open there."""
        with computation():
            records = _interval_records(value)
            if len(records) > self.root_count + 1:  # r points separate at most r + 1 intervals.
                raise Rejected("More intervals than cells")
            intervals: list = []
            for record in records:
                tp.charge(1)
                lower, upper, left_closed, right_closed = _interval_fields(record)
                low = None if lower is None else self.locate(lower)
                high = None if upper is None else self.locate(upper)
                if (low is None and left_closed) or (high is None and right_closed):
                    raise Rejected("An infinite end is open")
                if low is not None and high is not None and (
                        low > high or (low == high and not (left_closed and right_closed))):
                    raise Rejected("An interval is reversed or empty")
                if intervals:
                    end, closed = intervals[-1][1], intervals[-1][3]
                    if end is None or low is None or end > low or (end == low and (closed or left_closed)):
                        raise Rejected("The intervals are not sorted, disjoint and unmergeable")
                intervals.append((low, high, left_closed, right_closed))
            return tuple(intervals)

    def membership(self, value) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        """(root_truth, cell_truth) of a canonical IntervalSet record whose finite ends are points: whether it holds
        each alpha_k and each open cell, in the layout of reconstruct. Rejected as ``index_form`` rejects.

        Such a set holds a whole open cell or none of it, so these values determine it."""
        with computation():
            return self._membership(self.index_form(value))

    def _membership(self, intervals) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        roots = [False] * self.root_count
        cells = [False] * (self.root_count + 1)
        for low, high, left_closed, right_closed in intervals:
            first = 0 if low is None else low  # The cells first, ..., last lie inside, and the points between them.
            last = self.root_count if high is None else high - 1
            tp.charge(last - first + 2)
            for k in range(first, last + 1):
                cells[k] = True
            for k in range(first + 1, last + 1):
                roots[k - 1] = True
            if left_closed:
                roots[low - 1] = True
            if right_closed:
                roots[high - 1] = True
        return tuple(roots), tuple(cells)

    def matches(self, value, root_truth, cell_truth) -> bool:
        """Whether the claimed IntervalSet record is the set of the truth values, in canonical form.

        True only when its index form equals ``reconstruct(root_truth, cell_truth)``; its membership, computed the
        other way round, must also give the truth values back. Every rejection is False."""
        try:
            with computation():
                expected = self.reconstruct(root_truth, cell_truth)
                claimed = self.index_form(value)
                return claimed == expected and self._membership(claimed) == (root_truth, cell_truth)
        except Rejected:
            return False


def check_samples(master, samples, cuts=()) -> bool:
    """Whether the samples certify the points of the master polynomial and the cut points: ``Cells(master, samples,
    cuts)`` accepts."""
    try:
        Cells(master, samples, cuts)
    except Rejected:
        return False
    return True
