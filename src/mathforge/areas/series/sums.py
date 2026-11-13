"""Closed forms of sums and series over the exact rationals (mathforge.series): the finite sum of a polynomial, finite
and infinite geometric sums, and the series of a polynomial times a geometric progression.

The argument polynomial is a Polynomial (MCP kind "polynomial"), first and ratio are exact rationals, a Rational or an
int (kind "rational"), and count is an exact int (kind "int"), so operations_v6.run already rejects a bool, a float, a
str and None as invalid input; no argument has a default. Every value is a Polynomial of mathforge.polynomial or a
Rational of mathforge.model: the family defines no record class, so TYPES is empty and there is no RENDER (the shared
renderer presents both). For each operation, its domain, its value and the evidence that its checker,
mathforge.certify.series.sums, verifies:

- sum_polynomial(polynomial): the closed form F(n) = p(1) + ... + p(n), the unique polynomial in the variable of p (the
  same Symbol) with F(0) = 0 and F(n) - F(n - 1) = p(n). deg F = deg p + 1, and p = 0 gives the zero polynomial. A p of
  degree MAX_POLYNOMIAL_DEGREE (1024) has no Polynomial value, since F would have degree 1025: produce() raises
  UnsupportedOperation for it, the error of every Polynomial beyond that degree. Evidence ().
- geometric_sum(first, ratio, count): first + first*ratio + ... + first*ratio**(count - 1), a Rational: an integral sum
  too is a Rational, Rational(1023), never an int. count = 0 gives Rational(0). count < 0 is invalid input, and
  count * bits(ratio) above max_integer_bits is a resource limit, bits(ratio) being the bit length of the longer of
  |numerator| and denominator: an a priori bound on the length of ratio**count. The ratios 1, 0 and -1 are exempt,
  since they form no power that grows (the card's limit bounds ratio**count; their sums are count * first, first and
  first or 0, and fit whenever their own integers do). Evidence ().
- geometric_series(first, ratio): first/(1 - ratio), a Rational, for |ratio| < 1; first = 0 gives Rational(0) for every
  ratio. first != 0 with |ratio| >= 1 is invalid input: the series diverges. Evidence ().
- polynomial_geometric_series(polynomial, ratio): the sum of p(k) * ratio**k over k = 0, 1, 2, ..., a Rational, for
  |ratio| < 1; p = 0 gives Rational(0) for every ratio, and a non-zero p with |ratio| >= 1 is invalid input. The value
  is -Q(0) for the unique polynomial Q with ratio*Q(x + 1) - Q(x) = p(x) and deg Q = deg p (the partial sums telescope
  to Q(n)*ratio**n - Q(0)). Evidence (Q,): the coefficients of Q, a tuple of Rationals (never ints), lowest degree
  first, without a trailing zero and exactly as many as p has: (Rational(0),) for p = 0.

How produce() computes (docs/v09/cards/m3-analysis.md). Nothing here is a proof: the checker verifies every value with
algorithms of its own (an exact Taylor shift and a comparison of coefficients for the two polynomial identities, binary
splitting without a division for the finite geometric sum, cross-multiplication for the infinite one).

- The inputs. produce() validates every argument itself with exact type tests, also those whose kind operations_v6.run
  checks: run() admits every object of the Polynomial and Rational classes, also one made with object.__new__ that
  breaks the canonical form. A polynomial must be a Polynomial whose variable is a Symbol with fields of their exact
  types (a str name and uid, a Domain, a tuple of at most three Assumptions) and whose coefficients are a nonempty tuple
  of at most 1,025 Rationals in lowest terms with positive denominators, without a trailing zero; first and ratio an
  int or such a Rational; count an int. Anything else is invalid input.
- Newton's divided differences. Write p = P/D with P in Z[x] of degree d and D >= 1 the least common denominator of the
  coefficients. Synthetic division of P by x - s, then of the quotient by x - s - 1, and so on, gives the coefficients
  of P in Newton's basis of the consecutive nodes s, s + 1, ..., s + d, P(x) = sum_i R_i (x - s)(x - s - 1)...(x - s -
  i + 1), where R_i = P[s, ..., s + i] is the divided difference: an integer, since dividing by x - k keeps every
  quotient in Z[x]. The nested form t_0 + x(t_1 + (x - 1)(t_2 + ...)) expands sum_k t_k x(x - 1)...(x - k + 1) back to
  powers of x with one multiplication by x - k per node, from the top.
- sum_polynomial is Newton's interpolation of the partial sums S(n) = p(1) + ... + p(n) at the nodes 0, 1, ..., d + 1.
  As S(n + 1) - S(n) = p(n + 1), their divided differences are S[0] = S(0) = 0 and S[0, ..., j] = p[1, ..., j]/j for
  j >= 1, so with the R_i of P at the nodes 1, ..., d + 1, D F(x) = sum_{j=1..d+1} (R_{j-1}/j) x(x - 1)...(x - j + 1).
  Times L = lcm(1, ..., d + 1) its coefficients are the integers R_{j-1} L/j, the nested form expands it, and the
  coefficients of F are the results over L D in lowest terms.
- geometric_sum is the card's closed form, first * (1 - ratio**count)/(1 - ratio) with first = an/ad and ratio =
  rn/rd: an U/(ad rd**(count - 1)) in lowest terms, where U = (rd**count - rn**count)/(rd - rn) is an exact integer
  quotient (U = 1 for ratio 0, count mod 2 for ratio -1); ratio 1 gives count * first. The powers come from a
  square-and-multiply loop of charged products; no builtin pow runs.
- geometric_series is first/(1 - ratio) = an rd/(ad (rd - rn)) in lowest terms.
- polynomial_geometric_series solves the identity in the falling factorials x^(k) = x(x - 1)...(x - k + 1), on which
  Q(x + 1) - Q(x) acts as x^(k) -> k x^(k-1). With p = sum a_k x^(k), the a_k = P[0, ..., k]/D by the divided
  differences at the nodes 0, 1, ..., d, the coefficients b_k of Q = sum b_k x^(k) satisfy (r - 1) b_k + r (k + 1)
  b_(k+1) = a_k, solved from b_d = a_d/(r - 1) down. With r = rn/rd and g = rd - rn > 0, b_k = beta_k/(D g^(d+1-k)) for
  the integers beta_d = -rd A_d and beta_k = rn (k + 1) beta_(k+1) - rd A_k g^(d-k) (A_k = D a_k); the nested form
  expands sum_k beta_k g^k x^(k) = D g^(d+1) Q, and Q's coefficients are its results over D g^(d+1) in lowest terms.
  The value is -Q(0).

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and one for each argument and coefficient that it
validates, refuses an integer longer than max_integer_bits with ResourceLimitError (also in a claim, which no Rational
could hold, and as the common denominator of a polynomial argument, which the checker refuses too), and charges every further step on the cost scale of mathforge.trusted.integers: a product its
product_work, a long division its division_work, a gcd the division of the longer operand by the shorter plus one unit
per 64 squared 64-bit words of the shorter (the checker's charge), and each Rational it builds that gcd again, as
Rational() reduces its integers. A pass of synthetic division or a step of the nested expansion costs half a unit per
coefficient and half a unit per 2,048 bits of all its integers together, charged before it runs at their length then
and afterwards for what they grew, which a pass cannot know before it runs. Measured on one machine with CPython,
those passes take 0.16 to 0.19 microseconds per unit, near the checker's 0.2, so the default max_work (2,000,000) pays
for about 0.4 seconds. The sum of x^1023 takes about 1.54 million units (0.27 seconds), the series of a dense
polynomial of degree 1024 with two-digit coefficients and ratio 1/2 about 1.36 million; the series of x^1024 with
ratio -2/5 needs 2.03 million, and a polynomial of high degree with long coefficients, or the series of one with a
ratio near 1 (whose Q has long coefficients), runs out of the default budget.

What the checker costs. It verifies F and Q by an exact Taylor shift of their integer form (the coefficients over their
common denominator): for n coefficients of at most b bits, 1 + n(n - 1)/8 * (1 + 2 floor((b + n)/4096)) work units,
plus a few units per coefficient. Under the default limits every degree up to 1024 verifies while b + n stays below
32,768 (about 1.98 million of the 2 million units, measured at 0.43 seconds); the sum of x^1023 takes about 410,000.
It refuses, before the shift, an F or Q beyond the size bounds that its docstring proves for the true one. The shift is
charged by the longest integer and the producer's passes by the length of all of them, so a claim can be cheap to make
and dear to check: the sum of 2^60000 + x^800 takes the producer 0.8 million units and its check 2.34 million. Before
it returns a claim, produce() therefore computes what the checker will charge for it, from the argument and the claim
alone, every charge in the checker's order: reading them, the common denominators, the shift and the comparison of the
coefficients, and for geometric_sum the products of the binary splitting, whose lengths the producer's own chain of
powers gives exactly. It raises ResourceLimitError when that exceeds max_work: the checker runs in a fresh budget of
the same limits (operations_v6.run's verifier, the lean runner), so both sides fail at the same point. geometric_sum's
check needs no intermediate beyond 4 * max_integer_bits for any sum that fits max_integer_bits.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, docs/v09/cards/m3-analysis.md is the specification, and tests/v09/series/test_sums.py is
the definition of done.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import islice
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Assumption, Domain, Rational, Symbol, Truth
from ...polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
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

# The scope text of every operation of wave M3 (docs/v09/cards/m3-analysis.md).
_SCOPE = "exact rationals and real functions; certified closed forms of sums, series and trigonometric integrals"
_POLYNOMIAL = Argument("polynomial", "Polynomial", None, "polynomial")
_FIRST = Argument("first", "Rational | int", None, "rational")
_RATIO = Argument("ratio", "Rational | int", None, "rational")

SPECS = (
    OperationSpec("sum_polynomial", "series", (_POLYNOMIAL,), "Polynomial",
                  "Find the closed form F(n) of the sum of p(k) for k = 1 .. n, a polynomial in the same variable.",
                  _SCOPE, version=6),
    OperationSpec("geometric_sum", "series", (_FIRST, _RATIO, Argument("count", "int", None, "int")), "Rational",
                  "Add the first count terms of a geometric progression with exact rational first term and ratio.",
                  _SCOPE, version=6),
    OperationSpec("geometric_series", "series", (_FIRST, _RATIO), "Rational",
                  "Sum an infinite geometric series with |ratio| < 1.", _SCOPE, version=6),
    OperationSpec("polynomial_geometric_series", "series", (_POLYNOMIAL, _RATIO), "Rational",
                  "Sum p(k) * ratio^k over k = 0, 1, 2, ... for a polynomial p and |ratio| < 1.", _SCOPE, version=6),
)

# No record classes: every value is a Polynomial or a Rational, which serialize and render without one.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.series.sums"

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
_MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.
_NOT_A_POLYNOMIAL = "{} requires a Polynomial"
_NOT_CANONICAL = ("{} requires a canonical Polynomial: a Symbol variable and a nonempty tuple of at most 1,025 "
                  "Rationals in lowest terms with positive denominators, without a trailing zero")
_NOT_RATIONAL = "{} requires {} to be an exact rational: an int or a Rational"
_NOT_CANONICAL_RATIONAL = "{} requires a Rational {} of two exact integers in lowest terms with a positive denominator"
_NOT_A_COUNT = "geometric_sum requires count to be an exact int"
_NEGATIVE_COUNT = "geometric_sum requires count >= 0"
_DIVERGES = "{} requires |ratio| < 1 unless {} is 0: the series diverges"
_TOO_HIGH = (f"sum_polynomial of a polynomial of degree {MAX_POLYNOMIAL_DEGREE} has degree "
             f"{MAX_POLYNOMIAL_DEGREE + 1}: dense polynomial degree exceeds {MAX_POLYNOMIAL_DEGREE}")
_POWER_BITS = "Computation limit exceeded: integer bits of ratio**count (count * bits(ratio) > max_integer_bits)"
_UNCHECKABLE = "Computation limit exceeded: work ({}) for the independent check of the claim"
_LONG_DENOMINATOR = "Computation limit exceeded: integer bits of the common denominator of the polynomial"

Pair = tuple[int, int]  # (numerator, denominator) of a canonical rational.


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name; the
    module states the algorithms and the evidence. InvalidInput for an argument of the wrong type or outside the
    operation's domain, UnsupportedOperation for the sum of a polynomial of degree 1024, ResourceLimitError past the
    budget, also for a claim whose independent check would need more than max_work."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the sums family")
    if type(args) is not dict or any(type(name) is not str for name in args) or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    with computation() as budget:
        budget.tick()  # The call itself; validating an argument charges at least one unit more.
        return _PRODUCERS[operation](budget, *(args[name] for name in names))


# ---------------------------------------------------------------------------------------------------------------
# The operations

def _sum_polynomial(budget: Budget, polynomial: object) -> tuple[Polynomial, tuple]:
    """F with F(0) = 0 and F(x) - F(x - 1) = p(x), by Newton's interpolation of the partial sums; the evidence ()."""
    from ...trusted.integers import product_work

    pairs, variable = _polynomial(budget, polynomial, "sum_polynomial")
    if pairs == [(0, 1)]:  # Every sum of zeros is 0.
        return Polynomial((Rational(0),), variable), ()
    degree = len(pairs) - 1
    if degree >= MAX_POLYNOMIAL_DEGREE:
        raise UnsupportedOperation(_TOO_HIGH)
    # A common denominator D beyond max_integer_bits is refused, as the checker refuses it before it reads F.
    integers, denominator, _, p_form = _common_form(budget, pairs, budget.limits.max_integer_bits)
    differences = _newton(budget, list(integers), 1)  # R_i = P[1, ..., i + 1].
    scale = _lcm_range(budget, degree + 1)
    # D F(x) = sum_j (R_(j-1)/j) x^(j), times L: the coefficient of x^(j) is R_(j-1) L/j, and of x^(0) zero (F(0) = 0).
    terms = [0] + [_times(budget, differences[j - 1], _quotient(budget, scale, j)) for j in range(1, degree + 2)]
    claim = _canonical_pairs(budget, _expand(budget, terms), _times(budget, scale, denominator))
    # The checker's charges for this claim (certify.series.sums._sum_polynomial), in its order.
    _, f_denominator, f_bits, f_form = _common_form(budget, claim)
    work = (_read_work(pairs) + p_form + degree + 1 + _read_work(claim) + f_form
            + _shift_work(degree + 2, f_bits))
    cofactor = _quotient(budget, f_denominator, denominator)  # D divides E: E p(x) = (E/D) P(x) is integral.
    for j in range(degree + 1):  # E F(x) - E F(x - 1) = (E/D) P(x), compared at x^j: a tick and two products.
        difference = _times(budget, cofactor, integers[j])
        work += (3 + product_work(denominator.bit_length(), difference.bit_length())
                 + product_work(f_denominator.bit_length(), integers[j].bit_length()))
    _refuse_unchecked(budget, work)
    return Polynomial(_rationals(budget, claim), variable), ()


def _geometric_sum(budget: Budget, first: object, ratio: object, count: object) -> tuple[Rational, tuple]:
    """first * (1 - ratio**count)/(1 - ratio), or count * first for ratio 1, and the evidence ()."""
    an, ad = _rational(budget, first, "geometric_sum", "first")
    rn, rd = _rational(budget, ratio, "geometric_sum", "ratio")
    count = _count(budget, count)
    # (1 - r^n)/(1 - r) = U/rd^(n-1) with U = (rd^n - rn^n)/(rd - rn) = sum_(k<n) rn^k rd^(n-1-k), an integer.
    lows = highs = None
    if not an or not count:  # The empty sum, or a sum of zeros.
        value, partial = (0, 1), 0
    elif rn == rd:  # ratio = 1, as both are canonical: count terms equal to first, and U = count.
        common = _gcd(budget, count, ad)
        value, partial = (_times(budget, an, _quotient(budget, count, common)), _quotient(budget, ad, common)), count
    elif rd == 1 and -1 <= rn <= 1:  # The ratios 0 and -1 form no power that grows: U = 1, or count mod 2.
        partial = count & 1 if rn else 1
        value = (an, ad) if partial else (0, 1)
    else:  # rn**count and rd**count, whose lengths count * bits(ratio) bounds a priori.
        if count * max(abs(rn).bit_length(), rd.bit_length()) > budget.limits.max_integer_bits:
            raise ResourceLimitError(_POWER_BITS)
        lows, highs = _chain(budget, rn, count), _chain(budget, rd, count)
        partial = _quotient(budget, highs[-1] - lows[-1], rd - rn)  # An exact quotient.
        numerator = _times(budget, an, partial)
        denominator = _times(budget, ad, _quotient(budget, highs[-1], rd))  # ad rd^(n-1) > 0.
        common = _gcd(budget, numerator, denominator)
        value = (_quotient(budget, numerator, common), _quotient(budget, denominator, common))
    _refuse_unchecked(budget, _geometric_sum_work(budget, (an, ad), (rn, rd), count, value, partial, lows, highs))
    return _rationals(budget, [value])[0], ()


def _geometric_series(budget: Budget, first: object, ratio: object) -> tuple[Rational, tuple]:
    """first/(1 - ratio) for |ratio| < 1 (0 for first = 0, whatever the ratio), and the evidence ()."""
    from ...trusted.integers import product_work

    an, ad = _rational(budget, first, "geometric_series", "first")
    rn, rd = _rational(budget, ratio, "geometric_series", "ratio")
    if not an:  # Every term is 0, whatever the ratio.
        value = (0, 1)
    elif not -rd < rn < rd:  # first != 0 and |ratio| >= 1: the terms do not tend to 0.
        raise InvalidInput(_DIVERGES.format("geometric_series", "first"))
    else:
        numerator, denominator = _times(budget, an, rd), _times(budget, ad, rd - rn)  # rd - rn > 0.
        common = _gcd(budget, numerator, denominator)
        value = (_quotient(budget, numerator, common), _quotient(budget, denominator, common))
    # The checker's charges (certify.series.sums._geometric_series): first, ratio and the value, then s (1 - r) = a
    # cross-multiplied as vn (rd - rn) ad == an vd rd when first != 0.
    work = _pair_work((an, ad)) + _pair_work((rn, rd)) + _pair_work(value)
    if an:
        vn, vd = value
        left, right = _times(budget, vn, rd - rn), _times(budget, an, vd)
        work += (4 + product_work(vn.bit_length(), (rd - rn).bit_length())
                 + product_work(left.bit_length(), ad.bit_length())
                 + product_work(an.bit_length(), vd.bit_length()) + product_work(right.bit_length(), rd.bit_length()))
    _refuse_unchecked(budget, work)
    return _rationals(budget, [value])[0], ()


def _polynomial_geometric_series(budget: Budget, polynomial: object, ratio: object) -> tuple[Rational, tuple]:
    """-Q(0) for the Q of degree deg p with ratio*Q(x + 1) - Q(x) = p(x), and the evidence (Q,)."""
    from ...trusted.integers import product_work

    pairs, _ = _polynomial(budget, polynomial, "polynomial_geometric_series")
    rn, rd = _rational(budget, ratio, "polynomial_geometric_series", "ratio")
    if pairs == [(0, 1)]:  # Every term is 0, whatever the ratio; Q = 0 telescopes to 0.
        zero = Rational(0)
        return zero, ((zero,),)
    if not -rd < rn < rd:  # p != 0 and |ratio| >= 1: the terms p(k) ratio^k do not tend to 0.
        raise InvalidInput(_DIVERGES.format("polynomial_geometric_series", "the polynomial"))
    degree = len(pairs) - 1
    # A common denominator D beyond max_integer_bits is refused, as the checker refuses it before it reads Q. Q may
    # still fit: for ratio 0, Q = -p, whose coefficients 1/q1 and 1/q2 fit while q1 q2 does not.
    integers, denominator, _, p_form = _common_form(budget, pairs, budget.limits.max_integer_bits)
    falling = _newton(budget, list(integers), 0)  # A_k = P[0, ..., k] = D a_k.
    gap = rd - rn  # g = rd (1 - r) > 0.
    betas = [0] * (degree + 1)
    betas[degree] = -_times(budget, rd, falling[degree])
    power = 1  # g^(d-k)
    for k in range(degree - 1, -1, -1):
        budget.tick()
        power = _times(budget, power, gap)
        betas[k] = (_times(budget, rn * (k + 1), betas[k + 1])
                    - _times(budget, _times(budget, rd, falling[k]), power))
    terms, power = [], 1  # beta_k g^k, then g^(d+1)
    for beta in betas:
        terms.append(_times(budget, beta, power))
        power = _times(budget, power, gap)
    claim = _canonical_pairs(budget, _expand(budget, terms), _times(budget, denominator, power))
    value = (-claim[0][0], claim[0][1])  # -Q(0)
    # The checker's charges for this claim (certify.series.sums._polynomial_geometric_series), in its order.
    q_integers, q_denominator, q_bits, q_form = _common_form(budget, claim)
    work = (_read_work(pairs) + _pair_work((rn, rd)) + _pair_work(value) + p_form + _read_work(claim) + q_form
            + _shift_work(degree + 1, q_bits) + 1 + product_work(rd.bit_length(), q_denominator.bit_length()))
    scaled = _times(budget, rd, q_denominator)  # The checker's scale, rd E.
    cofactor = _quotient(budget, scaled, denominator)  # D divides rd E: rd E p(x) is integral.
    for j in range(degree + 1):
        # rn E Q(x + 1) - rd E Q(x) = rd E p(x) = (rd E/D) P(x) at x^j: a tick and four products, the first by rn.
        difference = _times(budget, cofactor, integers[j])
        work += (4 + product_work(rd.bit_length(), q_integers[j].bit_length())
                 + product_work(denominator.bit_length(), difference.bit_length())
                 + product_work(scaled.bit_length(), integers[j].bit_length()))
        if rn:  # The coefficient of E Q(x + 1), from the identity: (difference + rd E Q_j)/rn.
            shifted = _quotient(budget, difference + _times(budget, rd, q_integers[j]), rn)
            work += 1 + product_work(rn.bit_length(), shifted.bit_length())
        else:  # A product by 0 costs its unit alone.
            work += 1
    _refuse_unchecked(budget, work)
    coefficients = _rationals(budget, claim)
    return -coefficients[0], (coefficients,)


_PRODUCERS: dict[str, Callable[..., tuple[object, tuple]]] = {
    "sum_polynomial": _sum_polynomial,
    "geometric_sum": _geometric_sum,
    "geometric_series": _geometric_series,
    "polynomial_geometric_series": _polynomial_geometric_series,
}


# ---------------------------------------------------------------------------------------------------------------
# Validation

def _integers(budget: Budget, value: object, message: str) -> Pair:
    """(numerator, denominator) of a Rational, which must hold two exact ints in lowest terms with a positive
    denominator: one made with object.__new__ may lack a field or break that form, InvalidInput(message) then.
    ResourceLimitError for an integer longer than max_integer_bits, before the gcd that its length bounds."""
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(message)
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator != 1 and _gcd(budget, numerator, denominator) != 1:
        raise InvalidInput(message)
    return numerator, denominator


def _rational(budget: Budget, value: object, operation: str, name: str) -> Pair:
    """(numerator, denominator) of an exact int or a canonical Rational argument; InvalidInput for anything else."""
    budget.tick()
    if type(value) is int:  # Neither a bool nor a float is an exact rational.
        budget.check_int(value)
        return value, 1
    if type(value) is not Rational:  # A float, a str, None, a Fraction, a Rational subclass, a container.
        raise InvalidInput(_NOT_RATIONAL.format(operation, name))
    return _integers(budget, value, _NOT_CANONICAL_RATIONAL.format(operation, name))


def _count(budget: Budget, value: object) -> int:
    """The count of geometric_sum: an exact int >= 0; InvalidInput otherwise."""
    budget.tick()
    if type(value) is not int:
        raise InvalidInput(_NOT_A_COUNT)
    budget.check_int(value)
    if value < 0:
        raise InvalidInput(_NEGATIVE_COUNT)
    return value


def _canonical_symbol(symbol: object) -> bool:
    """Whether a variable is a Symbol whose fields have their exact types, as the checker requires them."""
    if type(symbol) is not Symbol:
        return False
    name, domain = getattr(symbol, "name", None), getattr(symbol, "domain", None)
    assumptions, uid = getattr(symbol, "assumptions", None), getattr(symbol, "uid", None)
    if (type(name) is not str or type(domain) is not Domain or type(uid) is not str or type(assumptions) is not tuple
            or len(assumptions) > _MAX_ASSUMPTIONS):
        return False
    return all(type(item) is Assumption and type(getattr(item, "name", None)) is str
               and type(getattr(item, "truth", None)) is Truth for item in assumptions)


def _polynomial(budget: Budget, value: object, operation: str) -> tuple[list[Pair], Symbol]:
    """(pairs of the coefficients, variable) of a canonical Polynomial argument (module docstring); InvalidInput
    otherwise. One unit for each coefficient."""
    if type(value) is not Polynomial:
        raise InvalidInput(_NOT_A_POLYNOMIAL.format(operation))
    message = _NOT_CANONICAL.format(operation)
    coefficients, variable = getattr(value, "coefficients", None), getattr(value, "variable", None)
    if (type(coefficients) is not tuple or not 0 < len(coefficients) <= MAX_POLYNOMIAL_DEGREE + 1
            or not _canonical_symbol(variable)):
        raise InvalidInput(message)
    pairs = []
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not Rational:  # Neither an int nor a float nor a Rational subclass.
            raise InvalidInput(message)
        pairs.append(_integers(budget, coefficient, message))
    if len(pairs) > 1 and not pairs[-1][0]:  # A trailing zero.
        raise InvalidInput(message)
    return pairs, variable


# ---------------------------------------------------------------------------------------------------------------
# Charged integer arithmetic, on the cost scale of mathforge.trusted.integers

def _times(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs: one unit and what the lengths cost beyond it (product_work)."""
    from ...trusted.integers import product_work

    budget.tick(1 + product_work(left.bit_length(), right.bit_length()))
    return left * right


def _quotient(budget: Budget, dividend: int, divisor: int) -> int:
    """dividend // divisor for a divisor != 0, its long division charged before it runs (division_work)."""
    from ...trusted.integers import division_work

    budget.tick(1 + division_work(dividend.bit_length(), divisor.bit_length()))
    return dividend // divisor


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """What a gcd of integers of these lengths costs, the checker's charge: Lehmer's algorithm first divides the longer
    operand by the shorter (division_work of a quotient as long as their difference), then runs quadratically in the
    shorter one, one unit per 64 squared 64-bit words."""
    from ...trusted.integers import division_work

    longer, shorter = max(left_bits, right_bits), min(left_bits, right_bits)
    words = shorter >> 6
    return 1 + (words * words >> 6) + division_work(longer - shorter + 64, shorter)


def _gcd(budget: Budget, left: int, right: int) -> int:
    """math.gcd, charged before it runs (_gcd_work)."""
    budget.tick(_gcd_work(left.bit_length(), right.bit_length()))
    return gcd(left, right)


def _chain(budget: Budget, base: int, exponent: int) -> list[int]:
    """[base**m for each m of the left-to-right binary chain of the exponent >= 1]: m = 1, then for each bit after the
    leading one 2m and, for a set bit, 2m + 1; the last entry is base**exponent. Square-and-multiply of charged
    products, never the builtin pow; the caller keeps the exponent short (count * bits(ratio) <= max_integer_bits)."""
    powers = [base]
    for bit in bin(exponent)[3:]:  # The bits after the leading 1, from the top.
        budget.tick()
        powers.append(_times(budget, powers[-1], powers[-1]))
        if bit == "1":
            powers.append(_times(budget, powers[-1], base))
    return powers


def _lcm_range(budget: Budget, top: int) -> int:
    """lcm(1, 2, ..., top), every gcd, quotient and product charged."""
    value = 1
    for j in range(2, top + 1):
        value = _times(budget, _quotient(budget, value, _gcd(budget, value, j)), j)
    return value


def _canonical_pairs(budget: Budget, integers: list[int], scale: int) -> list[Pair]:
    """integers[i]/scale in lowest terms, for a scale >= 1: (0, 1) for a zero. ResourceLimitError when one of them
    has an integer longer than max_integer_bits: no Rational holds it, and the claim would be beyond the budget."""
    pairs = []
    for integer in integers:
        common = _gcd(budget, integer, scale)  # gcd(0, scale) = scale.
        numerator, denominator = _quotient(budget, integer, common), _quotient(budget, scale, common)
        budget.check_int(numerator)
        budget.check_int(denominator)
        pairs.append((numerator, denominator))
    return pairs


def _rationals(budget: Budget, pairs: list[Pair]) -> tuple[Rational, ...]:
    """The Rationals of canonical pairs. Rational() charges a unit for each and reduces its integers again, a gcd that
    is charged here first; it raises ResourceLimitError for an integer longer than max_integer_bits."""
    values = []
    for numerator, denominator in pairs:
        if denominator != 1:
            budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
        values.append(Rational(numerator, denominator))
    return tuple(values)


def _pass_work(steps: int, bits: int) -> int:
    """What a pass of steps of synthetic division or of the nested expansion costs on integers of bits bits together:
    half a unit for each step and half a unit for each 2,048 bits (the module states the measurements)."""
    return 1 + ((steps + (bits >> 11)) >> 1)


def _newton(budget: Budget, values: list[int], start: int) -> list[int]:
    """Replace the integer coefficients of P in values (lowest degree first) by its divided differences at the nodes
    start, start + 1, ..., start + d, and return the list: P(x) = sum_i values[i] (x - start) ... (x - start - i + 1).
    Pass i divides the quotient that is left by x - (start + i), whose remainder becomes final. A pass is charged
    before it runs at the length of its integers and afterwards for what they grew, which it cannot know before."""
    last = len(values) - 1
    bits = sum(map(int.bit_length, values))  # Of the entries still to divide, values[offset:].
    for offset in range(last):
        node = start + offset
        if not node:  # Dividing by x: the remainder is the constant term, the quotient the rest.
            budget.tick()
            bits -= values[offset].bit_length()
            continue
        steps = last - offset
        budget.tick(_pass_work(steps, bits))
        for i in range(last - 1, offset - 1, -1):
            values[i] += node * values[i + 1]
        grown = sum(map(int.bit_length, islice(values, offset, None)))
        if grown > bits:
            budget.tick(_pass_work(steps, grown) - _pass_work(steps, bits))
        bits = grown - values[offset].bit_length()  # The remainder is final.
    return values


def _expand(budget: Budget, terms: list[int]) -> list[int]:
    """The integer coefficients, lowest degree first, of sum_k terms[k] x (x - 1) ... (x - k + 1), by the nested form
    terms[0] + x (terms[1] + (x - 1) (terms[2] + ...)) from the top. Each step multiplies by x - k and adds terms[k];
    it is charged before it runs at the length of its integers and afterwards for what they grew."""
    result = [terms[-1]]
    bits = terms[-1].bit_length()  # Of the entries of result.
    for k in range(len(terms) - 2, -1, -1):
        if not k:  # Multiplying by x shifts the coefficients up.
            budget.tick()
            result.insert(0, terms[0])
            continue
        steps = len(result) + 1
        before = bits + terms[k].bit_length()
        budget.tick(_pass_work(steps, before))
        result.append(0)
        for i in range(len(result) - 1, 0, -1):
            result[i] = result[i - 1] - k * result[i]
        result[0] = terms[k] - k * result[0]
        bits = sum(map(int.bit_length, result))
        if bits > before:
            budget.tick(_pass_work(steps, bits) - _pass_work(steps, before))
    return result


# ---------------------------------------------------------------------------------------------------------------
# What the checker charges for a claim (mathforge.certify.series.sums states each charge)

def _common_form(budget: Budget, pairs: list[Pair], bound: int | None = None) -> tuple[list[int], int, int, int]:
    """(P, E, bits, work): the least common denominator E of canonical pairs, the integers P[i] = pairs[i] * E, their
    largest bit length, and the work that the checker charges for computing them, in its order (the running lcm of the
    denominators, then each integer); the same computation here charges this budget as much. With a bound,
    ResourceLimitError as soon as E has more than bound bits: the checker refuses a polynomial argument whose common
    denominator exceeds max_integer_bits (certify.series.sums._source) at the same step."""
    from ...trusted.integers import division_work, product_work

    work = 0

    def charge(units: int) -> None:
        nonlocal work
        budget.tick(units)
        work += units

    denominator = 1
    for _, part in pairs:
        charge(1)
        if part != 1:
            charge(_gcd_work(denominator.bit_length(), part.bit_length()))
            common = gcd(denominator, part)
            charge(1 + division_work(denominator.bit_length(), common.bit_length()))
            reduced = denominator // common
            charge(1 + product_work(reduced.bit_length(), part.bit_length()))
            denominator = reduced * part
            if bound is not None and denominator.bit_length() > bound:
                raise ResourceLimitError(_LONG_DENOMINATOR)
    integers, bits = [], 0
    for numerator, part in pairs:
        charge(1)
        if denominator == 1:
            integer = numerator
        else:
            charge(1 + division_work(denominator.bit_length(), part.bit_length()))
            cofactor = denominator // part
            charge(1 + product_work(numerator.bit_length(), cofactor.bit_length()))
            integer = numerator * cofactor
        integers.append(integer)
        bits = max(bits, integer.bit_length())
    return integers, denominator, bits, work


def _pair_work(pair: Pair) -> int:
    """What the checker charges to read a canonical rational: the gcd of its integers when the denominator is not 1."""
    numerator, denominator = pair
    return _gcd_work(numerator.bit_length(), denominator.bit_length()) if denominator != 1 else 0


def _read_work(pairs: list[Pair]) -> int:
    """What the checker charges to read a tuple of coefficients: a unit for each, and its gcd."""
    return sum(1 + _pair_work(pair) for pair in pairs)


def _shift_work(length: int, bits: int) -> int:
    """The checker's charge for the Taylor shift of length integer coefficients of at most bits bits."""
    steps = length * (length - 1) >> 1
    return 1 + (steps * (1 + 2 * ((bits + length) >> 12)) >> 2)


def _geometric_sum_work(budget: Budget, first: Pair, ratio: Pair, count: int, value: Pair, partial: int,
                        lows: list[int] | None, highs: list[int] | None) -> int:
    """What the checker charges for the claim value of geometric_sum (certify.series.sums._geometric_sum): reading the
    three rationals, then, along the bits of count after the leading one, a unit and the products U_m (rn^m + rd^m),
    (rn^m)^2 and (rd^m)^2, and for a set bit U_(2m) rd, rn^(2m) rn and rd^(2m) rd, where U_m = (rd^m - rn^m)/(rd - rn);
    then five products for the comparison, with U_count = partial. The chain of m is the one of the producer's powers,
    lows = [rn^m] and highs = [rd^m] (_chain), so every length is exact. None for the ratios 1, 0 and -1, whose
    powers have at most one bit: every product of the loop then costs its unit alone."""
    from ...trusted.integers import product_work

    def cost(left: int, right: int) -> int:
        return product_work(left.bit_length(), right.bit_length())

    (an, ad), (rn, rd), (vn, vd) = first, ratio, value
    work = _pair_work(first) + _pair_work(ratio) + _pair_work(value)
    if not an or not count:
        return work
    base = 1
    if lows is None or highs is None:
        work += 4 * (count.bit_length() - 1) + 3 * (count.bit_count() - 1)
    else:
        gap, index, power, base, u = rd - rn, 0, lows[0], highs[0], 1
        for bit in bin(count)[3:]:
            budget.tick()
            work += 4 + cost(u, power + base) + cost(power, power) + cost(base, base)
            index += 1
            power, base = lows[index], highs[index]
            if bit == "1":
                doubled = _quotient(budget, base - power, gap)  # U_(2m)
                work += 3 + cost(doubled, rd) + cost(power, rn) + cost(base, rd)
                index += 1
                power, base = lows[index], highs[index]
            u = _quotient(budget, base - power, gap)
    left, right, scaled = _times(budget, vn, ad), _times(budget, vd, an), _times(budget, partial, rd)
    return (work + 5 + cost(vn, ad) + cost(left, base) + cost(vd, an) + cost(partial, rd)
            + cost(right, scaled))


def _refuse_unchecked(budget: Budget, work: int) -> None:
    """ResourceLimitError when the checker would need more than max_work for the claim: it runs in a fresh budget of
    the same limits, never in what the production left of this one."""
    if work > budget.limits.max_work:
        raise ResourceLimitError(_UNCHECKABLE.format(budget.limits.max_work))
