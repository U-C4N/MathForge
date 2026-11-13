"""Independent checker of the family mathforge.areas.ntheory.quadratic_irrational: quadratic_continued_fraction,
convergents and pell_fundamental. It never expands a number itself: it verifies identities that the claimed terms
must satisfy.

quadratic_continued_fraction(p, q, d, r): value PeriodicContinuedFraction(preperiod, period), evidence ().
The claim, for exact integers with d >= 2 no perfect square, q != 0 and r != 0: the simple continued fraction of
x = (p + q*sqrt(d))/r is a0, ..., a(s-1) (the preperiod) followed by b0, ..., b(L-1) (the period) repeated forever,
in the canonical form. True requires
  (1) the shape: two tuples of exact ints, a nonempty period of terms >= 1, preperiod terms after the first >= 1;
  (6) a primitive period word (no repetition of a shorter block) and (7) an empty preperiod or one whose last term
      differs from the last period term;
  (2)-(5) with the period matrix M = [[b0, 1], [1, 0]] ... [[b(L-1), 1], [1, 0]] = [[A, B], [C, D]], the preperiod
      matrix N = [[a0, 1], [1, 0]] ... [[a(s-1), 1], [1, 0]] = [[P, P'], [Q, Q']] (the identity when s = 0) and
      Delta = (A + D)^2 - 4*(-1)^L: Delta*d is the square of an integer k >= 0, and
          r*E1 == p*E2 + q*Q*k  and  r*P*k == p*Q*k + q*d*E2,  with E1 = P*(A - D) + 2*C*P', E2 = Q*(A - D) + 2*C*Q'.
Theorem. For t > 0, [b0; b1, ..., b(L-1), t] = (A*t + B)/(C*t + D) (the matrix form of the convergents), with B >= 1,
C >= 1 and det M = (-1)^L. The purely periodic fraction eta = [b0; ..., b(L-1), b0, ...] converges to a number above
b0 >= 1 with eta = (A*eta + B)/(C*eta + D): eta is a root of C*t^2 + (D - A)*t - B, whose roots have the product
-B/C < 0, so it is the only positive root, eta = (A - D + sqrt(Delta))/(2*C), where (D - A)^2 + 4*B*C =
(A + D)^2 - 4*det M = Delta (conditions 2 and 3). When Delta*d = k^2, sqrt(Delta) = (k/d)*sqrt(d) (condition 4), so
eta = ((A - D)*d + k*sqrt(d))/(2*C*d) and (P*eta + P')/(Q*eta + Q') = (d*E1 + P*k*sqrt(d))/(d*E2 + Q*k*sqrt(d)),
whose denominator is 2*C*d*(Q*eta + Q') > 0 ((Q, Q') = (0, 1) for s = 0, Q >= 1 and Q' >= 0 for s >= 1, since the
preperiod terms after a0 are >= 1). Cross-multiplied with (p + q*sqrt(d))/r, r != 0, and compared on the basis
(1, sqrt(d)), which is linearly independent over Q because d is no square, the two identities say exactly
x = (P*eta + P')/(Q*eta + Q') (condition 5; the rational parts, both multiples of d, are compared divided by d). For
an integer a and a real t > 1, a + 1/t lies strictly between a and a + 1, so the continued fraction of a + 1/t is a
followed by that of t. Every complete quotient
[a(j); ..., a(s-1), eta] with j >= 1 exceeds 1 (a(j) >= 1 and eta > 1), so by induction the continued fraction of
x = [a0; a1, ..., a(s-1), eta] is a0, ..., a(s-1) followed by that of eta: the period repeated. An irrational
number has exactly one simple continued fraction, and conditions 6 and 7 write an eventually periodic sequence
exactly once (its minimal period divides every period, so the minimal period word is the primitive root of any
period word; the preperiod is the shortest exactly when its last term cannot move into the period, rotated). So
the value is the canonical expansion of x. Conversely, the true value passes every condition: its tail eta is a
complete quotient of x, so Q(sqrt(Delta)) = Q(eta) = Q(sqrt(d)), and two integers that are no squares generate the
same quadratic field exactly when their product is a square.

Caps. Before any matrix product, the check raises ResourceLimitError for an expansion with more than MAX_TERMS
(99,000) terms in preperiod and period together, or whose terms have bit lengths (of their absolute values) that
add up to more than max_integer_bits. That sum is the card's total of log2(|b| + 1), each term rounded up, since
bitlen(b) = ceil(log2(|b| + 1)). Such a claim is beyond the budget, not wrong, and the producer refuses the same
expansions; a malformed claim answers False first, whatever its size, except for the count of terms, which is read
before the terms. Within the caps every entry of M and N is at most 2**max_integer_bits in absolute value (an entry
of a product of matrices [[t, 1], [1, 0]] is at most the product of the |t| + 1), and at the default limits the
whole check of any claim costs well under max_work, so a true claim within the caps is always verified.

convergents(value): value a tuple of Rationals, evidence ().
The claim, for an exact rational v = p/q (an int, or a Rational in lowest terms with q >= 1): the value is
(p0/q0, ..., pn/qn), the convergents of the canonical finite continued fraction [a0; a1, ..., an] of v (a(k) >= 1
for k >= 1 and an >= 2 when n >= 1), each one a Rational, those with denominator 1 included. True requires a
nonempty tuple of Rationals, with p(k) and q(k) the numerator and denominator of the k-th; q0 == 1, and a0 = p0;
for k >= 1, a(k) = (q(k) - q(k-2))/q(k-1) is an exact integer >= 1 (q(-1) = 0, q(-2) = 1) and
p(k) == a(k)*p(k-1) + p(k-2) (p(-1) = 1, p(-2) = 0); an >= 2 when n >= 1; and pn/qn == v.
Theorem. Then q(k) = a(k)*q(k-1) + q(k-2) and p(k) = a(k)*p(k-1) + p(k-2) for every k >= 0 (q0 = 1 and p0 = a0 are
the case k = 0), so (p(k), q(k)) are the numerators and denominators that the forward recurrence gives for the
terms a0, ..., an: p(k)/q(k) = [a0; ..., a(k)]. These terms have the canonical form and [a0; ..., an] = pn/qn = v;
the canonical finite continued fraction of a rational is unique, so they are its terms and the value is its tuple
of convergents. Conversely, the recurrence gives coprime p(k), q(k) with q(k) >= 1 (p(k)*q(k-1) - p(k-1)*q(k) =
(-1)^(k-1)), so the reduced Rationals of the true tuple hold exactly those integers, and it passes.
Sizes. The true tuple has at most 2*bitlen(q) + 2 entries, because q(k) is at least the Fibonacci number F(k+1)
(every a(k) >= 1 for k >= 1), and its entries satisfy 1 <= q(k) <= q and |p(k)| <= |p| + q (the q(k) never
decrease, and |p(k)/q(k) - v| < 1). A longer tuple, or an entry beyond these bounds, is refused before any
arithmetic, so an oversized forged value answers False at once. The exact quotients that pass multiply to at most
q, so the divisions and products cost in all about a few products of the size of v: the check never exhausts the
default budget, and no cap applies.

pell_fundamental(d): value (x, y), evidence (period,).
The claim, for an exact integer d >= 2 that is no perfect square: (x, y) is the least solution of x^2 - d*y^2 = 1
in positive integers. True requires a tuple of two exact ints; the conditions of quadratic_continued_fraction for
(0, 1, d, 1), that is sqrt(d), with the preperiod (a0,), a0 = isqrt(d), and the given period, which prove that
period to be THE minimal period of the continued fraction of sqrt(d), of length L (the caps apply as above); with
M as above, (h, k) = (a0*B + D, B), the convergent [a0; b0, ..., b(L-2)] (the second column of
[[a0, 1], [1, 0]]*M): h^2 - d*k^2 == (-1)^L; (x, y) == (h, k) for even L and (h^2 + d*k^2, 2*h*k) for odd L; and
x^2 - d*y^2 == 1.
Theorem (Lagrange, on Pell's equation). When sqrt(d) = [a0; (b0, ..., b(L-1))] with the minimal period L and
p(j)/q(j) are its convergents, the solutions of x^2 - d*y^2 = +-1 in positive integers are exactly the
(p(mL-1), q(mL-1)), m >= 1, and p(mL-1)^2 - d*q(mL-1)^2 = (-1)^(mL). They increase with m, so the least solution of
x^2 - d*y^2 = 1 is the one of m = 1 for even L and that of m = 2 for odd L, where p(2L-1) + q(2L-1)*sqrt(d) =
(p(L-1) + q(L-1)*sqrt(d))^2 = h^2 + d*k^2 + 2*h*k*sqrt(d). The norm condition and the last identity follow from the
theorem; they are kept as cheap guards.

Common rules. The record class is identified by its module and name: a checker never imports producer code.
Anything malformed, a record forged without __init__ or with a missing field included, answers False. An argument
over max_integer_bits raises ResourceLimitError, as the producer's budget refuses it: such a claim is beyond the
budget, not wrong. No prime, factorization or modular power occurs, so the helpers of certify.ntheory.factorization
are not needed, and no builtin pow runs: squares are products. Every product, division and square root of
integers is charged to the budget before it runs (trusted.upoly.charge_product), and every loop charges the budget.
"""

from __future__ import annotations

from math import isqrt

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.trusted.upoly import charge_product

RECORD = ("mathforge.areas.ntheory.quadratic_irrational", "PeriodicContinuedFraction")  # Module and name.
MAX_TERMS = 99_000  # Preperiod and period together; the card's cap on the length of a verified expansion.
QUADRATIC_ARGUMENTS = frozenset(("p", "q", "d", "r"))
CONVERGENTS_ARGUMENTS = frozenset(("value",))
PELL_ARGUMENTS = frozenset(("d",))


def _charge(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, count products, divisions or square roots of integers of the given bit lengths: one
    unit each and what their lengths cost beyond it (trusted.upoly.charge_product)."""
    budget.tick(count)
    charge_product(left_bits, right_bits, count)


def _times(budget: Budget, *factors: int) -> int:
    """The product of the factors, each multiplication charged before it runs."""
    product = factors[0]
    for factor in factors[1:]:
        _charge(budget, product.bit_length(), factor.bit_length())
        product *= factor
    return product


def _square_root(budget: Budget, n: int) -> int:
    """isqrt(n) for an int n >= 0, charged before it runs like a product of n with itself."""
    _charge(budget, n.bit_length(), n.bit_length())
    return isqrt(n)


def _irrational_root(budget: Budget, d: object) -> int | None:
    """isqrt(d) when d is an exact int >= 2 that is no perfect square, so that sqrt(d) is irrational; None otherwise.
    ResourceLimitError for a d over max_integer_bits."""
    if type(d) is not int or d < 2:  # A bool is no exact integer.
        return None
    budget.check_int(d)
    root = _square_root(budget, d)
    return None if _times(budget, root, root) == d else root


def _fraction(value: object) -> tuple[object, object] | None:
    """(numerator, denominator) of a Rational, each read once; None for any other class. A Rational made with
    object.__new__ may lack a field, which reads as None and fails the caller's int tests."""
    if type(value) is not Rational:  # Not an int either, nor a class of another module with the same name.
        return None
    return getattr(value, "numerator", None), getattr(value, "denominator", None)


def _record_fields(value: object) -> tuple[object, object] | None:
    """(preperiod, period) of a value of the record class, each read once; None for any other class or for a record
    that lacks a field."""
    cls = type(value)
    module, name = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    if type(module) is not str or type(name) is not str or (module, name) != RECORD:
        return None
    try:
        return value.preperiod, value.period
    except AttributeError:  # A record made with object.__new__ may lack a field.
        return None


def _repeats(budget: Budget, word: tuple) -> bool:
    """Whether a nonempty word is a repetition of a shorter block (condition 6). Testing the prime divisors k of its
    length suffices: a word u^m with m >= 2 is also the k-th power of u^(m/k) for every prime k that divides m."""
    length = rest = len(word)
    factor = 2
    while factor * factor <= rest:
        budget.tick()
        if rest % factor == 0:
            budget.tick(1 + (length >> 6))  # Building and comparing the k-th power is linear in the length.
            if word == word[:length // factor] * factor:
                return True
            while rest % factor == 0:
                budget.tick()
                rest //= factor
        factor += 1
    if rest > 1:
        budget.tick(1 + (length >> 6))
        return word == word[:length // rest] * rest
    return False


def _matrix(budget: Budget, terms: tuple) -> tuple[int, int, int, int]:
    """(A, B, C, D) of [[A, B], [C, D]] = [[t0, 1], [1, 0]] ... [[t(n-1), 1], [1, 0]], the identity for no terms:
    A/C and B/D are the last two convergents of [t0; ..., t(n-1)]. The terms are exact ints within the caps."""
    a, b, c, d = 1, 0, 0, 1
    for term in terms:
        _charge(budget, max(a.bit_length(), c.bit_length()), term.bit_length(), 2)
        a, b, c, d = a * term + b, a, c * term + d, c
    return a, b, c, d


def _expansion(budget: Budget, p: int, q: int, d: int, r: int, preperiod: object,
               period: object) -> tuple[int, int, int, int] | None:
    """The period matrix (A, B, C, D) when [preperiod, (period)] is the canonical simple continued fraction of
    (p + q*sqrt(d))/r, None when it is not: conditions (1) to (7) of the module docstring, for validated arguments
    (d >= 2 no square, q != 0, r != 0). ResourceLimitError beyond the caps."""
    if type(preperiod) is not tuple or type(period) is not tuple or not period:
        return None
    if len(preperiod) + len(period) > MAX_TERMS:  # Read before the terms, so that a forged length costs nothing.
        raise ResourceLimitError(f"Computation limit exceeded: continued fraction terms ({MAX_TERMS})")
    bits = 0  # bitlen(t) = ceil(log2(|t| + 1)), also for a negative a0.
    for index, term in enumerate(preperiod):
        budget.tick()
        if type(term) is not int or (index and term < 1):  # A bool is no exact integer.
            return None
        bits += term.bit_length()
    for term in period:
        budget.tick()
        if type(term) is not int or term < 1:
            return None
        bits += term.bit_length()
    if preperiod and preperiod[-1] == period[-1]:
        return None  # Condition 7: the last preperiod term would start the period, rotated.
    if _repeats(budget, period):
        return None  # Condition 6.
    if bits > budget.limits.max_integer_bits:
        raise ResourceLimitError("Computation limit exceeded: continued fraction term bits")
    big_a, big_b, big_c, big_d = _matrix(budget, period)
    pre_p, pre_p1, pre_q, pre_q1 = _matrix(budget, preperiod)
    trace = big_a + big_d
    delta = _times(budget, trace, trace) - (-4 if len(period) & 1 else 4)  # det M = (-1)^L.
    square = _times(budget, delta, d)
    k = _square_root(budget, square)
    if _times(budget, k, k) != square:
        return None  # Condition 4: the tail lies in another quadratic field.
    gap, twice_c = big_a - big_d, 2 * big_c
    e1 = _times(budget, pre_p, gap) + _times(budget, twice_c, pre_p1)
    e2 = _times(budget, pre_q, gap) + _times(budget, twice_c, pre_q1)
    if _times(budget, r, e1) != _times(budget, p, e2) + _times(budget, q, pre_q, k):
        return None  # Condition 5, the rational parts divided by d.
    if _times(budget, r, pre_p, k) != _times(budget, p, pre_q, k) + _times(budget, q, d, e2):
        return None  # Condition 5, the parts of sqrt(d).
    return big_a, big_b, big_c, big_d


def _check_quadratic(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    if args.keys() != QUADRATIC_ARGUMENTS or evidence:
        return False
    p, q, d, r = args["p"], args["q"], args["d"], args["r"]
    if type(p) is not int or type(q) is not int or type(r) is not int or q == 0 or r == 0:
        return False  # Invalid input: no claim on it is true.
    budget.check_int(p)  # ResourceLimitError over max_integer_bits, as the producer's budget refuses it.
    budget.check_int(q)
    budget.check_int(r)
    if _irrational_root(budget, d) is None:
        return False
    fields = _record_fields(value)
    return fields is not None and _expansion(budget, p, q, d, r, *fields) is not None


def _check_convergents(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    if args.keys() != CONVERGENTS_ARGUMENTS or evidence:
        return False
    source = args["value"]
    fraction = (source, 1) if type(source) is int else _fraction(source)  # A bool is no exact rational.
    if fraction is None:
        return False
    numerator, denominator = fraction
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        return False
    budget.check_int(numerator)  # ResourceLimitError over max_integer_bits, as the producer's budget refuses it.
    budget.check_int(denominator)
    if type(value) is not tuple or not value or len(value) > 2 * denominator.bit_length() + 2:
        return False  # The true tuple has at most 2*bitlen(q) + 2 entries (module docstring).
    bound = abs(numerator) + denominator  # |p(k)| <= |p| + q for every true convergent.
    last = len(value) - 1
    p2, p1, q2, q1 = 0, 1, 1, 0  # p(k-2), p(k-1), q(k-2), q(k-1), from the seeds of k = 0.
    for index, item in enumerate(value):
        budget.tick()
        fraction = _fraction(item)  # Not an int: every convergent is a Rational, also with denominator 1.
        if fraction is None:
            return False
        p, q = fraction
        if type(p) is not int or type(q) is not int or not 1 <= q <= denominator or abs(p) > bound:
            return False
        if index == 0:
            if q != 1:
                return False
            term = p
        else:
            # The quotient has at most bitlen(q(k)) - bitlen(q(k-1)) + 1 bits: charge the long division first.
            _charge(budget, q.bit_length(), max(1, q.bit_length() - q1.bit_length() + 1))
            term, remainder = divmod(q - q2, q1)
            if remainder or term < 1 or (index == last and term < 2):
                return False
        if p != _times(budget, term, p1) + p2:
            return False
        p2, p1, q2, q1 = p1, p, q1, q
    return p1 == numerator and q1 == denominator


def _check_pell(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    if args.keys() != PELL_ARGUMENTS or len(evidence) != 1:
        return False
    d = args["d"]
    root = _irrational_root(budget, d)
    if root is None or type(value) is not tuple or len(value) != 2:
        return False
    x, y = value
    if type(x) is not int or type(y) is not int:
        return False
    period = evidence[0]
    matrix = _expansion(budget, 0, 1, d, 1, (root,), period)
    if matrix is None:
        return False
    big_b, big_d = matrix[1], matrix[3]
    h, k = _times(budget, root, big_b) + big_d, big_b  # The convergent p(L-1)/q(L-1) of sqrt(d).
    hh, dkk = _times(budget, h, h), _times(budget, d, k, k)
    odd = len(period) & 1
    if hh - dkk != (-1 if odd else 1):
        return False
    expected = (hh + dkk, _times(budget, 2, h, k)) if odd else (h, k)
    if x != expected[0] or y != expected[1]:
        return False
    return _times(budget, x, x) - _times(budget, d, y, y) == 1


def check_operation(operation, args, value, evidence):
    if type(args) is not dict or type(evidence) is not tuple:
        return False
    if operation == "quadratic_continued_fraction":
        check = _check_quadratic
    elif operation == "convergents":
        check = _check_convergents
    elif operation == "pell_fundamental":
        check = _check_pell
    else:
        return False
    with computation() as budget:
        return check(budget, args, value, evidence)
