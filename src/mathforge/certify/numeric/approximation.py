"""Independent checker of the family mathforge.areas.numeric.approximation: rational_approximation.

It never searches for the approximation the way the producer does (no continued fraction, no convergent): it verifies
a Farey-neighbour certificate and decides every comparison of the value with a rational exactly, by integer
arithmetic, by Sturm counts, or by outward-rounded enclosures together with an exact evaluation in a multiquadratic
field, all from the machinery of the digits checker (mathforge.certify.numeric.digits), which it reuses.

rational_approximation(value, max_denominator) = r. Let x be the real number that value denotes and N the
max_denominator. The claim: r is the rational with denominator <= N that minimizes |x - r|; on a tie (x exactly
midway between two such rationals) the one with the smaller denominator, and of two with equal denominators the
smaller one. True requires:
  (0) the operation "rational_approximation"; args a dict of exactly value and max_denominator, N an exact int with
      1 <= N <= MAX_DENOMINATOR; r a Rational of the class mathforge.model.Rational (never an int, a subclass or an
      impostor) whose fields are exact ints in lowest terms with 1 <= denominator <= N and a numerator of at most
      max_integer_bits bits; the value one of the kinds below, in its canonical form (an int, a Rational in lowest
      terms, a RealAlgebraicRoot (S, i) as the digits checker reads it, or a closed expression of Add, Mul, Pow and
      Sqrt over Rationals as the digits checker parses it); the evidence a tuple in one of the two layouts:
  (A) evidence (): True requires x == r, decided exactly (below).
      Theorem. r has denominator <= N and |x - r| = 0, the least distance there is; no other rational has distance 0.
  (B) evidence (l, u), two Rationals of the class mathforge.model.Rational in the form of (0) (never ints), with
      u.numerator * l.denominator - l.numerator * u.denominator == 1, l.denominator <= N, u.denominator <= N and
      l.denominator + u.denominator > N; r is l or u; l < x < u decided exactly; and with m = (l + u) / 2, r is l
      when x < m, u when x > m, and when x == m the one of smaller denominator, l when the denominators are equal.
      Theorem. Write l = a/b and u = c/d, so cb - ad = 1 (hence l < u). A rational p/q with l < p/q < u has
      bp - aq >= 1 and cq - dp >= 1, so q = q(cb - ad) = b(cq - dp) + d(bp - aq) >= b + d > N: no rational with
      denominator <= N lies strictly between l and u (they are neighbours in the Farey sequence of order N). Every
      other candidate s with denominator <= N thus has s < l < x or x < u < s, and is strictly farther from x than l
      or u. The minimizers are therefore among l and u: x < m makes l the unique closest, x > m makes u, and x == m
      makes both closest, where the tie rule picks the one stated. The certificate determines r uniquely, so no other
      value passes. (A rational x with denominator <= N admits no such l and u, so (B) never applies to it.)

Exact comparisons sign(x - t) of x with a rational t = p/q in lowest terms (q >= 1), by the kind of the value:
  - int or Rational x = a/b: the sign of a*q - p*b.
  - RealAlgebraicRoot (S, i), S square-free with a positive leading coefficient, x = alpha_i the i-th of the real roots
    alpha_0 < alpha_1 < ... of S. With B = 2 + max_(k < deg S) |S_k| // S_(deg S): t >= B gives x < t and t <= -B
    gives x > t (Cauchy's bound: every root z of S has |z| <= 1 + max_k |S_k| / S_(deg S) < B). Otherwise, with
    c(t) = certify.upoly.count_roots(S, None, t), the number of roots <= t on the checker's own Sturm chain, and
    z(t) = 1 when S(t) = 0, else 0: the roots below t number c(t) - z(t), so x < t iff c(t) - z(t) >= i + 1, x == t
    iff z(t) = 1 and c(t) = i + 1, and x > t otherwise. Counts only, never a sign change of S, which shows that some
    root lies between two points, not that alpha_i does.
  - Closed expression: for b = 4 * bits(N) + 64, doubled up to max_integer_bits, the digits checker's outward-rounded
    enclosure (L, H) with L/2**b <= x <= H/2**b (its module docstring proves each rule): p*2**b < L*q proves x > t and
    H*q < p*2**b proves x < t. At the first precision that separates nothing, the checker computes once the formal
    value of x in the algebra A = Q[Y_1, ..., Y_s]/(Y_j**2 - b_j) over the natural coprime base of the radicands (the
    digits checker's _formal_value; Y_j -> sqrt(b_j) is a ring homomorphism phi onto the value); when the formal value
    is the constant t, x = phi(t) = t. Otherwise the enclosures go on. Soundness rests only on the enclosures and on
    phi being a homomorphism. Completeness: the members b_j are pairwise coprime non-squares, so the products of their
    square roots over the subsets are linearly independent over Q (Besicovitch; Kummer theory), phi is injective, and
    an x different from t is separated from it by a fine enough enclosure. No precision up to max_integer_bits
    separating x from t raises ResourceLimitError: beyond the budget, not wrong.

Sizes and failures. Every malformed claim answers False, never with an exception: another operation, other arguments,
an N outside its range, a value or an evidence member that is an int, a bool, a float, a str, a Rational subclass, an
impostor or a Rational forged with object.__new__ (a missing field, a zero or negative denominator, not in lowest
terms), a list for a tuple, evidence of another length, and a numerator longer than max_integer_bits in the value or
in the evidence (no claim of this family has one under the limits that verify it), which answers False before any
arithmetic on it. An argument beyond the budget raises ResourceLimitError, as run() and the verifier refuse it before
any checker runs: an integer longer than max_integer_bits, an expression of more than max_nodes distinct nodes or
deeper than max_depth, a polynomial whose Sturm chain passes the budget, an expression whose enclosure passes the
digits checker's limits or whose algebra holds more than MAX_RADICALS non-square members (the digits checker raises
both). Every loop is charged before it runs, and so is every product, division and gcd of long integers; no builtin
pow and no ** runs.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer: it compares (type(value).__module__, type(value).__qualname__) with ROOT, and the value and
the evidence must be of the class mathforge.model.Rational itself. tools/check_layering.py follows every import, and
tests/v08/test_layering.py rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable

from mathforge.certify.numeric.digits import (ROOT, _argument_fraction, _charge, _enclosure, _formal_value, _is_class,
                                              _lowest, _parse, _point, _rational_fields, _root, _times)
from mathforge.certify.upoly import count_roots
from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Rational, Sqrt, trusted_rational
from mathforge.trusted.upoly import sign_at

OPERATION = "rational_approximation"
ARGUMENTS = frozenset(("value", "max_denominator"))
MAX_DENOMINATOR = 2 ** 64  # The card's ceiling on max_denominator (mathforge.areas.numeric.approximation).
_GUARD_BITS = 64  # Bits of the first enclosure beyond 4 * bits(N).

Fraction = tuple[int, int]  # (p, q) in lowest terms with q >= 1.
Comparator = Callable[[Fraction], int]  # sign(x - t) for a rational t.


# ---------------------------------------------------------------------------------------------------------------
# Reading the claim

def _claimed(budget: Budget, value: object, limit: int) -> Fraction | None:
    """(p, q) of a Rational of the value or of the evidence: the class mathforge.model.Rational itself with exact int
    fields, 1 <= q <= limit (the max_denominator), a numerator of at most max_integer_bits bits and lowest terms;
    None for anything else, an int included (the family's values and evidence hold Rationals only). The lengths are
    tested before the gcd runs."""
    fraction = _rational_fields(value)
    if fraction is None:
        return None
    p, q = fraction
    if q > limit or p.bit_length() > budget.limits.max_integer_bits:
        return None
    return fraction if _lowest(budget, p, q) else None


# ---------------------------------------------------------------------------------------------------------------
# Exact comparisons of the value with rationals (module docstring)

def _rational_comparator(budget: Budget, a: int, b: int) -> Comparator:
    """sign(a/b - p/q) by the sign of a*q - p*b, b and q >= 1."""
    def compare(t: Fraction) -> int:
        p, q = t
        difference = _times(budget, a, q) - _times(budget, p, b)
        return (difference > 0) - (difference < 0)
    return compare


def _root_comparator(budget: Budget, polynomial: tuple[int, ...], index: int) -> Comparator:
    """sign(alpha_index - p/q) by Cauchy's bound and Sturm counts (module docstring)."""
    budget.tick(len(polynomial))
    leading = polynomial[-1]
    largest = max(abs(coefficient) for coefficient in polynomial[:-1])
    _charge(budget, largest.bit_length(), leading.bit_length())
    bound = 2 + largest // leading  # Every root z has |z| < bound.

    def compare(t: Fraction) -> int:
        p, q = t
        scaled = _times(budget, bound, q)
        if p >= scaled:
            return -1  # t >= bound > x.
        if p <= -scaled:
            return 1  # t <= -bound < x.
        below = count_roots(polynomial, None, trusted_rational(p, q))  # t is in lowest terms already.
        zero = 1 if sign_at(polynomial, p, q) == 0 else 0
        if below - zero >= index + 1:
            return -1
        if zero and below == index + 1:
            return 0
        return 1
    return compare


def _expression_comparator(budget: Budget, entries: list, denominator_limit: int) -> Comparator:
    """sign(x - p/q) for a parsed closed expression, by enclosures of rising precision and, where they separate
    nothing, the formal value in the multiquadratic algebra (module docstring)."""
    limit = budget.limits.max_integer_bits
    first = min(4 * denominator_limit.bit_length() + _GUARD_BITS, limit)
    formal: list[dict | None] = []  # The formal value, computed at most once.

    def equals(t: Fraction) -> bool:
        """Whether the formal value is the constant t, which proves x == t; False proves nothing, and the enclosures
        go on (a decomposition that fails its verified identity, which _formal_value never gives, leaves them alone
        to decide)."""
        if not formal:
            formal.append(_formal_value(budget, entries))
        element = formal[0]
        if element is None:
            return False
        budget.tick(len(element) + 1)
        if any(mask for mask in element):
            return False
        constant = element.get(0)
        if constant is None:
            return t[0] == 0
        return (constant.numerator, constant.denominator) == t

    def compare(t: Fraction) -> int:
        p, q = t
        bits, decided = first, False
        while True:
            budget.tick()
            low, high = _enclosure(budget, entries, bits)
            _charge(budget, p.bit_length() + bits, 1)
            target = p << bits
            if _times(budget, low, q) > target:
                return 1
            if _times(budget, high, q) < target:
                return -1
            if not decided:
                decided = True
                if equals(t):
                    return 0
            if bits >= limit:
                raise ResourceLimitError("Computation limit exceeded: no enclosure separates the value from a rational")
            bits = min(2 * bits, limit)
    return compare


def _comparator(budget: Budget, number: object, denominator_limit: int) -> Comparator | None:
    """The exact comparison for the value's kind; None for a value outside the family's scope or its canonical form.
    ResourceLimitError for an argument beyond the budget."""
    kind = type(number)
    if kind is int:
        budget.check_int(number)
        return _rational_comparator(budget, number, 1)
    if kind is Rational:
        fraction = _argument_fraction(budget, number)
        return None if fraction is None else _rational_comparator(budget, *fraction)
    if _is_class(number, ROOT):
        root = _root(budget, number)
        return None if root is None else _root_comparator(budget, *root)
    if kind is Add or kind is Mul or kind is Pow or kind is Sqrt:
        entries = _parse(budget, number)
        return None if entries is None else _expression_comparator(budget, entries, denominator_limit)
    return None


# ---------------------------------------------------------------------------------------------------------------
# The operation

def _check(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    """Condition (0), then layout (A) or (B); the cheap tests before any comparison."""
    if args.keys() != ARGUMENTS:
        return False
    limit = args["max_denominator"]
    if type(limit) is not int or not 1 <= limit <= MAX_DENOMINATOR:
        return False
    claim = _claimed(budget, value, limit)
    if claim is None:
        return False
    if not evidence:
        compare = _comparator(budget, args["value"], limit)
        return compare is not None and compare(claim) == 0
    if len(evidence) != 2:
        return False
    left, right = _claimed(budget, evidence[0], limit), _claimed(budget, evidence[1], limit)
    if left is None or right is None:
        return False
    (a, b), (c, d) = left, right
    if _times(budget, c, b) - _times(budget, a, d) != 1 or b + d <= limit or claim not in (left, right):
        return False
    compare = _comparator(budget, args["value"], limit)
    if compare is None or compare(left) != 1 or compare(right) != -1:
        return False
    # The midpoint (a*d + c*b) / (2*b*d), in lowest terms.
    midpoint = _point(budget, _times(budget, a, d) + _times(budget, c, b), 2 * _times(budget, b, d))
    side = compare((midpoint.numerator, midpoint.denominator))
    if side < 0:
        expected = left
    elif side > 0:
        expected = right
    else:  # A tie: the smaller denominator, and of equal denominators the smaller rational, l.
        expected = right if d < b else left
    return claim == expected


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of rational_approximation on args, by the theorems of the
    module. False for another operation, for arguments other than exactly value and max_denominator, and for any
    malformed value or evidence; ResourceLimitError for an argument beyond the budget or past the budget."""
    if type(operation) is not str or operation != OPERATION or type(args) is not dict or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        return _check(budget, args, value, evidence)
