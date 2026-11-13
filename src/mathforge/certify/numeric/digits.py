"""Independent checker of the family mathforge.areas.numeric.digits: digits.

It never computes the digits of a value the way the producer does (no mf.algebraic, no bisection): it verifies the
claimed record against the value with exact integer comparisons, Sturm counts of its own, its own rational enclosures
or an exact evaluation in a multiquadratic field.

digits(value, places) = TruncatedDecimal(sign, magnitude, places, exact). Let x be the real number that value denotes,
k = places and d = magnitude. The claim: sign is the exact sign of x, d = floor(|x| * 10**k), and exact says whether
|x| * 10**k is an integer. True requires, for every value:
  (0) the operation "digits"; args a dict of exactly value and places, places an exact int with
      0 <= places <= MAX_PLACES; a record of the class (module, qualname) == RECORD with all four fields: sign an exact
      int in {-1, 0, 1}, d an exact int >= 0 of at most max_integer_bits bits, places the very int of the argument,
      exact a bool, and (sign == 0) == (d == 0 and exact), the canonical form; the evidence a tuple in the layout of the
      value's kind below (a list is no tuple, a bool, a float or a str no number).
Then by the kind of the value:

Rational: value an exact int p (q = 1) or a Rational p/q, two exact ints in lowest terms with q >= 1; evidence ().
  True requires sign == sign(p), d*q <= |p|*10**k < (d + 1)*q and exact == (d*q == |p|*10**k).
  Theorem. Dividing by q*10**k > 0 gives d <= |x|*10**k < d + 1, so d = floor(|x|*10**k); |x|*10**k is an integer
  exactly when it equals its floor d, that is when d*q == |p|*10**k. The sign of x = p/q is the sign of p. The
  conditions determine every field, so no other record passes.

Real algebraic root: value a RealAlgebraicRoot (class (module, qualname) == ROOT, read as data) in the canonical
  form that its constructor makes: integer_coefficients S, lowest degree first, of degree >= 1, with a positive leading
  coefficient and S equal to its own primitive square-free part (certify.upoly.squarefree_part), and real_index i an
  exact int with 0 <= i < the number of real roots of S; it denotes x = alpha_i, the i-th of the real roots
  alpha_0 < alpha_1 < ... of S. Evidence ("root",). For a rational t let c(t) be the number of real roots of S at
  most t, certify.upoly.count_roots(S, None, t) on the checker's own Sturm chain, and z(t) = 1 if S(t) = 0, else 0;
  the roots below t number c(t) - z(t). Catalogue 2.2, by counts only (a sign change of S between two points shows
  that some root lies between them, not that alpha_i does: unsound):
  - the sign: -1 when c(0) - z(0) >= i + 1 (at least i + 1 roots below 0), 0 when z(0) = 1 and c(0) = i + 1
    (alpha_i = 0), and 1 otherwise; True requires sign to equal it;
  - sign 0: d == 0 and exact (the canonical form already says so);
  - sign 1, cell [lo, hi) with lo = d/10**k and hi = (d + 1)/10**k: c(lo) - z(lo) <= i, so alpha_i >= lo, and
    i < c(hi) - z(hi), so alpha_i < hi; exact == (z(lo) = 1 and c(lo) = i + 1), which says alpha_i = lo;
  - sign -1, cell (lo, hi] with lo = -(d + 1)/10**k and hi = -d/10**k: c(lo) <= i, so alpha_i > lo, and i < c(hi),
    so alpha_i <= hi; exact == (z(hi) = 1 and c(hi) = i + 1), which says alpha_i = hi.
  Theorem. With the sign established, x lies in the cell of d on its side of 0, so floor(|x|*10**k) = d, and
  |x|*10**k is an integer exactly when x is the cell's end that is d/10**k in absolute value. A root of S that happens
  to be rational is decided the same way (S need not be irreducible).
  The cell conditions at many places, through an isolating cell (k >= COARSE_PLACES). Evaluating the whole Sturm
  chain at points of 10**-k costs about deg S times one evaluation of S there, which put roots of degree 10 and more
  at 10,000 places beyond the default budget although the producer reaches them within it. So for j = 1, 2, 4, ...
  below k the checker first tries the decimal cell of precision j that holds the claimed cell: P = D/10**j and
  Q = (D + 1)/10**j with D = floor(d / 10**(k - j)) for sign 1, P = -(D + 1)/10**j and Q = -D/10**j for sign -1, so
  that lo and hi lie in [P, Q]. When the counts there give c(P) - z(P) = i and c(Q) = i + 1, alpha_i is the only
  root of S in [P, Q], and a simple one (S is square-free). If z(P) = 1, alpha_i = P, and if z(Q) = 1, alpha_i = Q:
  a rational, whose claim the Rational conditions decide. Otherwise S has a constant sign -s on [P, alpha_i), the
  sign s = sign S(Q) on (alpha_i, Q], and S(t) = 0 there only at t = alpha_i. Then for t in [P, Q], t < alpha_i exactly
  when sign S(t) = -s, and t = alpha_i exactly when S(t) = 0: for sign 1, alpha_i >= lo iff S(lo) = 0 or
  sign S(lo) = -s, alpha_i < hi iff sign S(hi) = s, and exact iff S(lo) = 0; for sign -1, alpha_i > lo iff
  sign S(lo) = -s, alpha_i <= hi iff S(hi) = 0 or sign S(hi) = s, and exact iff S(hi) = 0. These sign tests are
  sound because the counts have proved that alpha_i is the one root in [P, Q] (the unsound test is a sign change
  without that proof). When no such j isolates alpha_i, the counts at lo and hi above decide, as for k below
  COARSE_PLACES.

Expression: value a closed tree of the mathforge.model nodes Add, Mul, Pow and Sqrt over Rationals, in the form
  that their constructors make: a Rational in lowest terms, a Sqrt of a Rational >= 0, an Add or a Mul of a tuple of
  at least two nodes, a Pow of an exact int exponent >= 0 (Pow(b, 0) is 1, as power() makes it). No Symbol, no
  RealAlgebraicRoot leaf and no other class; shared nodes are evaluated once, and a node inside itself is refused.
  Two evidence layouts:
  ("bits", b), b an exact int with 1 <= b <= max_integer_bits. The checker's enclosure of x at precision b: integers
    (L, H) with L/2**b <= x <= H/2**b, computed bottom-up, every end rounded outward:
    - Rational p/q: L = floor(p*2**b/q), H = ceil(p*2**b/q);
    - Sqrt(p/q): L = isqrt(floor(p*4**b/q)) = floor(sqrt(p/q)*2**b) and H = L + 1;
    - Add: the sums of the ends;
    - Mul: the factors in order, left to right; the product of [a, c] and [g, h] is [floor(m/2**b), ceil(M/2**b)]
      for m and M the least and the greatest of a*g, a*h, c*g, c*h;
    - Pow(base, e): [2**b, 2**b] for e = 0; else R = the base's ends, then for each bit of e after its leading one,
      from the high end: R = the square of R, then R = R times the base's ends for a 1 bit; the square of [a, c] is
      [a*a, c*c] for a >= 0, [c*c, a*a] for c <= 0 and [0, max(a*a, c*c)] otherwise, rounded like a product.
    True requires exact False, sign != 0 and the enclosure strictly inside the open cell: for sign 1,
    d*2**b < L*10**k and H*10**k < (d + 1)*2**b; for sign -1, -(d + 1)*2**b < L*10**k and H*10**k < -d*2**b.
    Theorem. Each rule encloses the exact value of its node: floor(sqrt(floor(y))) = floor(sqrt(y)) for real y >= 0,
    interval sums and products contain the sums and products of their members, the square rule is the exact image of
    [a, c] under t -> t*t, and rounding outward only widens. So L/2**b <= x <= H/2**b, and for sign 1,
    d/10**k < x < (d + 1)/10**k: x > 0, floor(|x|*10**k) = d, and |x|*10**k is no integer; for sign -1 the mirror
    image. A number on a cell boundary (exact) has no such enclosure: it needs the next layout.
  ("rational", r), r a Rational in lowest terms (an int is refused: one canonical evidence). True requires the
    expression to equal r in the formal algebra below, and the record to pass the Rational conditions for r.
    The algebra. For every Sqrt(p/q), n = p*q: 0 gives 0, a perfect square n = s*s gives s/q; the non-square n are
    refined into their natural coprime base B by factor refinement (two numbers with g = gcd > 1 are replaced by g
    and what remains of each once every factor g is divided out, until the numbers are pairwise coprime; each n is
    then a product of powers of B, and B, the natural coprime base, does not depend on the order of the splits). The
    members of B that are no squares, b_1, ..., b_t with t <= MAX_RADICALS, span the algebra
    A = Q[Y_1, ..., Y_t]/(Y_j**2 - b_j), whose elements the checker adds coordinate by coordinate and multiplies by
    the rule of trusted.surd.Field.mul, e_s * e_u = weight[s & u] * e_(s xor u) with Field's table of the products
    of the b_j over each subset (rules that hold in A for any positive integers b_j), on the nonzero coordinates
    only (_formal_value states why). sqrt(p/q) maps to (c/q) * Y_S, Y_S the product of the Y_j over a set S,
    where n == c*c * prod_{j in S} b_j is verified exactly with an integer c >= 1; each other node maps by the same
    ring operation.
    Theorem. Y_j -> sqrt(b_j) defines a ring homomorphism phi from A to the reals, and the verified identity gives
    phi((c/q) * Y_S) = c*sqrt(prod b_j)/q = sqrt(n)/q = sqrt(p/q): phi of the formal value is x. So a formal value
    equal to the constant r gives x = r, whatever the members of B are; the Rational conditions then prove the record
    for x = r. Completeness: pairwise coprime non-squares b_j make no product of a nonempty subset of them a square,
    so Q(sqrt b_1, ..., sqrt b_t) has degree 2**t (Kummer theory) and phi is injective: a rational x always has a
    constant formal value. More than MAX_RADICALS such members raise ResourceLimitError (the algebra has 2**t
    coordinates); the producer refuses the same values.

Sizes and failures. Every malformed claim answers False, never with an exception: a record forged without __init__
or with a missing field, an impostor class, a bool, a float or a str where a number belongs, a list for a tuple, and
any other evidence. A claimed magnitude longer than max_integer_bits, a precision b above max_integer_bits and an r
longer than max_integer_bits answer False before any arithmetic: no claim of this family has them under the limits
that verify it. An argument beyond the budget raises ResourceLimitError, as run() and the verifier refuse it before
any checker runs: an integer longer than max_integer_bits, an expression of more than max_nodes distinct nodes or
deeper than max_depth, a polynomial whose Sturm chain passes the budget; so does an expression whose enclosure has an
end beyond 2**max_integer_bits in absolute value, or whose algebra passes the limits. Every loop is charged before it
runs, and so is every product, division, square root and gcd of long integers (trusted.upoly.charge_product and
charge_gcd; a gcd by both of its lengths, _gcd); no builtin pow and no ** runs, and a power of an integer is a chain
of charged products (_power), its multiplicity in another a chain of charged divisions (_strip).

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) with RECORD and ROOT instead. tools/check_layering.py follows every import, and
tests/v08/test_layering.py rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd, isqrt

from mathforge.certify.upoly import count_roots, squarefree_part
from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Rational, Sqrt, trusted_rational
from mathforge.trusted.surd import Field
from mathforge.trusted.upoly import charge_gcd, charge_product, sign_at

OPERATION = "digits"
RECORD = ("mathforge.areas.numeric.digits", "TruncatedDecimal")  # Module and name of the value's class.
ROOT = ("mathforge.algebraic", "RealAlgebraicRoot")  # Module and name of a real algebraic root argument.
ARGUMENTS = frozenset(("value", "places"))
MAX_PLACES = 10_000  # The card's ceiling on places (mathforge.areas.numeric.digits.MAX_PLACES).
MAX_RADICALS = 10  # The most coprime non-square radicands of an expression that the ("rational", r) layout evaluates.
COARSE_PLACES = 32  # From this many places on, a root is first decided through an isolating cell (_check_isolated).
_FIELDS = ("sign", "magnitude", "places", "exact")
_MISSING = object()  # A field that a record made with object.__new__ lacks.

Claim = tuple[int, int, bool]  # (sign, magnitude, exact), validated by _claim.
Entry = tuple  # A node of a parsed expression (_parse).


# ---------------------------------------------------------------------------------------------------------------
# Charged integer arithmetic

def _charge(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, count products, divisions or square roots of integers of the given bit lengths: one
    unit each and what their lengths cost beyond it (trusted.upoly.charge_product)."""
    budget.tick(count)
    charge_product(left_bits, right_bits, count)


def _times(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs."""
    _charge(budget, left.bit_length(), right.bit_length())
    return left * right


def _power(budget: Budget, base: int, exponent: int) -> int:
    """base**exponent for an exponent >= 0, by squaring and multiplying, each product charged before it runs (no
    builtin pow or **, which nothing could interrupt)."""
    result = 1
    while exponent:
        budget.tick()
        if exponent & 1:
            result = _times(budget, result, base)
        exponent >>= 1
        if exponent:
            base = _times(budget, base, base)
    return result


def _gcd(budget: Budget, left: int, right: int) -> int:
    """gcd(left, right), charged before it runs: one unit, Lehmer's algorithm on the shorter length (charge_gcd) and
    its first step, the long division of the longer by the shorter, (long words - short words + 1) * (short words + 1)
    word steps on charge_gcd's scale of 64 per unit. A long integer against a short one thus costs about its length,
    not the square of it."""
    _charge_gcd(budget, left.bit_length(), right.bit_length())
    return gcd(left, right)


def _charge_gcd(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, count gcds of integers of the given bit lengths, as _gcd states."""
    long_words, short_words = max(left_bits, right_bits) >> 6, min(left_bits, right_bits) >> 6
    budget.tick(count * (1 + ((long_words - short_words + 1) * (short_words + 1) >> 6)))
    charge_gcd(min(left_bits, right_bits), count)


def _strip(budget: Budget, number: int, factor: int) -> tuple[int, int]:
    """(rest, e) with number == factor**e * rest and factor no divisor of rest, for number >= 1 and factor >= 2.

    The powers factor**(2**j) are divided out in turn while each divides what is left, J of them; then each of them
    once more, from the highest down, where it divides. With rest = number / factor**e and factor no divisor of rest,
    factor**k divides factor**r * rest exactly when k <= r (else factor**(k - r) would divide rest): so the first pass
    stops with r < 2**J of the factor left, and the second takes the binary digits of r. About 2*log2(e) + 2 divisions
    and log2(e) + 1 squarings, each charged before it runs, where one division per factor would cost e divisions of
    long integers, a few microseconds each, for a single unit of work each."""
    powers: list[int] = []
    exponent, power = 0, factor
    while True:
        _charge(budget, number.bit_length(), power.bit_length())
        quotient, remainder = divmod(number, power)
        if remainder:
            break
        number = quotient
        exponent += 1 << len(powers)
        powers.append(power)
        power = _times(budget, power, power)
    for j in range(len(powers) - 1, -1, -1):
        _charge(budget, number.bit_length(), powers[j].bit_length())
        quotient, remainder = divmod(number, powers[j])
        if not remainder:
            number = quotient
            exponent += 1 << j
    return number, exponent


def _lowest(budget: Budget, numerator: int, denominator: int) -> bool:
    """Whether numerator/denominator, with denominator >= 1, is in lowest terms; the gcd is charged before it runs."""
    if denominator == 1:
        return True
    return _gcd(budget, numerator, denominator) == 1


def _point(budget: Budget, numerator: int, denominator: int) -> Rational:
    """numerator/denominator, denominator >= 1, as a Rational in lowest terms, the gcd and the divisions charged before
    they run."""
    divisor = _gcd(budget, numerator, denominator)
    _charge(budget, max(numerator.bit_length(), denominator.bit_length()), divisor.bit_length(), 2)
    return trusted_rational(numerator // divisor, denominator // divisor)


# ---------------------------------------------------------------------------------------------------------------
# Reading the claim and the arguments as data

def _is_class(value: object, identity: tuple[str, str]) -> bool:
    """Whether value's class has that module and qualified name, both read as exact strs."""
    cls = type(value)
    module, name = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    return type(module) is str and type(name) is str and (module, name) == identity


def _rational_fields(value: object) -> tuple[int, int] | None:
    """(numerator, denominator) of a model Rational whose fields are exact ints with a positive denominator, each read
    once; None for any other class (an int, a subclass, a Rational of another module included) and for a Rational made
    with object.__new__ that lacks a field or holds anything else."""
    if type(value) is not Rational:
        return None
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:  # A bool is no exact int.
        return None
    return numerator, denominator


def _argument_fraction(budget: Budget, value: object) -> tuple[int, int] | None:
    """(numerator, denominator) of a Rational inside an argument, in lowest terms; None for anything else.
    ResourceLimitError for an integer longer than max_integer_bits: beyond the budget, not wrong."""
    fraction = _rational_fields(value)
    if fraction is None:
        return None
    budget.check_int(fraction[0])
    budget.check_int(fraction[1])
    return fraction if _lowest(budget, *fraction) else None


def _claim(budget: Budget, value: object, places: int) -> Claim | None:
    """(sign, magnitude, exact) of a record of the class RECORD with all four fields in the canonical form and the
    argument's places; None otherwise. A magnitude longer than max_integer_bits answers None before any arithmetic."""
    if not _is_class(value, RECORD):
        return None
    sign, magnitude, claimed_places, exact = (getattr(value, name, _MISSING) for name in _FIELDS)
    if (type(sign) is not int or type(magnitude) is not int or type(claimed_places) is not int
            or type(exact) is not bool):  # Also every missing field, and a bool for an int.
        return None
    if (sign not in (-1, 0, 1) or magnitude < 0 or magnitude.bit_length() > budget.limits.max_integer_bits
            or claimed_places != places or (sign == 0) != (magnitude == 0 and exact)):
        return None
    return sign, magnitude, exact


def _root(budget: Budget, value: object) -> tuple[tuple[int, ...], int] | None:
    """(S, i) of a RealAlgebraicRoot argument in its constructor's canonical form (module docstring); None otherwise.
    ResourceLimitError for a coefficient longer than max_integer_bits or a Sturm chain beyond the budget."""
    coefficients = getattr(value, "integer_coefficients", _MISSING)
    index = getattr(value, "real_index", _MISSING)
    if type(coefficients) is not tuple or len(coefficients) < 2 or type(index) is not int or index < 0:
        return None
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not int:
            return None
        budget.check_int(coefficient)
    if coefficients[-1] <= 0 or squarefree_part(coefficients) != coefficients:
        return None  # Not the primitive square-free form with a positive leading coefficient.
    if index >= count_roots(coefficients, None, None):
        return None
    return coefficients, index


# ---------------------------------------------------------------------------------------------------------------
# Rationals and real algebraic roots

def _check_rational(budget: Budget, numerator: int, denominator: int, claim: Claim, scale: int) -> bool:
    """The Rational conditions for x = numerator/denominator (denominator >= 1) and scale = 10**k."""
    sign, magnitude, exact = claim
    if sign != (numerator > 0) - (numerator < 0):
        return False
    scaled = _times(budget, abs(numerator), scale)
    low = _times(budget, magnitude, denominator)
    return low <= scaled < low + denominator and exact == (low == scaled)


def _check_isolated(budget: Budget, polynomial: tuple[int, ...], index: int, claim: Claim, places: int,
                    scale: int) -> bool | None:
    """The cell conditions for a nonzero sign through an isolating decimal cell of precision j < places (module
    docstring); None when no j = 1, 2, 4, ... below places isolates alpha_index, and the counts at the claimed cell's
    ends have to decide."""
    sign, magnitude, exact = claim
    j = 1
    while j < places:
        budget.tick()
        factor = _power(budget, 10, places - j)
        _charge(budget, magnitude.bit_length(), factor.bit_length())
        coarse = magnitude // factor
        low, high = (coarse, coarse + 1) if sign > 0 else (-coarse - 1, -coarse)
        unit = _power(budget, 10, j)
        first, last = _point(budget, low, unit), _point(budget, high, unit)
        at_first = sign_at(polynomial, first.numerator, first.denominator)
        above = sign_at(polynomial, last.numerator, last.denominator)  # The sign of S above alpha_index, if no root.
        if (count_roots(polynomial, None, first) - (at_first == 0) == index
                and count_roots(polynomial, None, last) == index + 1):
            # alpha_index is the one root of the polynomial in [first, last], and a simple one.
            if not at_first:  # alpha_index = first, a rational: the Rational conditions decide.
                return _check_rational(budget, first.numerator, first.denominator, claim, scale)
            if not above:  # alpha_index = last.
                return _check_rational(budget, last.numerator, last.denominator, claim, scale)
            cell_low, cell_high = (magnitude, magnitude + 1) if sign > 0 else (-magnitude - 1, -magnitude)
            low_point, high_point = _point(budget, cell_low, scale), _point(budget, cell_high, scale)
            at_low = sign_at(polynomial, low_point.numerator, low_point.denominator)
            at_high = sign_at(polynomial, high_point.numerator, high_point.denominator)
            if sign > 0:
                return at_low != above and at_high == above and exact == (at_low == 0)
            return at_low == -above and at_high != -above and exact == (at_high == 0)
        j <<= 1
    return None


def _check_root(budget: Budget, polynomial: tuple[int, ...], index: int, claim: Claim, places: int,
                scale: int) -> bool:
    """The count conditions of the module docstring for x = alpha_index of the square-free polynomial."""
    sign, magnitude, exact = claim

    def at_most(numerator: int) -> tuple[int, int]:
        """(c(t), z(t)) at t = numerator/scale."""
        point = _point(budget, numerator, scale)
        below = count_roots(polynomial, None, point)
        return below, 1 if sign_at(polynomial, point.numerator, point.denominator) == 0 else 0

    below, zero = at_most(0)
    true_sign = -1 if below - zero >= index + 1 else 0 if zero and below == index + 1 else 1
    if sign != true_sign:
        return False
    if sign == 0:
        return magnitude == 0 and exact
    if places >= COARSE_PLACES:
        decided = _check_isolated(budget, polynomial, index, claim, places, scale)
        if decided is not None:
            return decided
    if sign > 0:
        (c_low, z_low), (c_high, z_high) = at_most(magnitude), at_most(magnitude + 1)
        return c_low - z_low <= index < c_high - z_high and exact == (z_low == 1 and c_low == index + 1)
    (c_low, z_low), (c_high, z_high) = at_most(-magnitude - 1), at_most(-magnitude)
    return c_low <= index < c_high and exact == (z_high == 1 and c_high == index + 1)


# ---------------------------------------------------------------------------------------------------------------
# Expressions

def _parse(budget: Budget, root: object) -> list[Entry] | None:
    """The distinct nodes of a closed expression, children before parents, as entries ("rational", p, q),
    ("sqrt", p, q), ("add", children), ("mul", children) and ("pow", child, exponent), children being positions in the
    list; the root's entry is the last. None for anything but the trees of the module docstring, and for a node inside
    itself. ResourceLimitError past max_nodes distinct nodes or max_depth levels, and for an integer longer than
    max_integer_bits."""
    entries: list[Entry] = []
    position: dict[int, int] = {}  # id(node) -> its entry, once all of its children have theirs.
    walking: set[int] = set()  # The ids of the nodes whose children are being walked: the path from the root.
    held: list[object] = []  # Every node walked, so that no id is reused while the walk lasts.
    stack: list[tuple[object, int, tuple | None]] = [(root, 1, None)]
    max_depth = budget.limits.max_depth
    while stack:
        node, depth, pending = stack.pop()
        key = id(node)
        if pending is not None:  # Every child of the node has its entry: the node's follows.
            walking.discard(key)
            kind, children, exponent = pending
            budget.tick(len(children))
            if kind is Pow:
                entries.append(("pow", position[id(children[0])], exponent))
            else:
                entries.append(("add" if kind is Add else "mul", tuple(position[id(child)] for child in children)))
            position[key] = len(entries) - 1
            continue
        if key in position:
            continue  # A shared node, evaluated once.
        if key in walking:
            return None  # A node inside itself: no finite expression.
        if depth > max_depth:
            raise ResourceLimitError("Computation limit exceeded: expression depth")
        budget.nodes()
        budget.tick()
        held.append(node)
        kind = type(node)
        if kind is Rational or kind is Sqrt:
            fraction = _argument_fraction(budget, node if kind is Rational else getattr(node, "radicand", _MISSING))
            if fraction is None or (kind is Sqrt and fraction[0] < 0):
                return None
            position[key] = len(entries)
            entries.append(("rational" if kind is Rational else "sqrt", *fraction))
            continue
        exponent = None
        if kind is Add or kind is Mul:
            children = getattr(node, "terms" if kind is Add else "factors", _MISSING)
            if type(children) is not tuple or len(children) < 2:
                return None
        elif kind is Pow:
            exponent = getattr(node, "exponent", _MISSING)
            if type(exponent) is not int or exponent < 0:
                return None
            budget.check_int(exponent)
            children = (getattr(node, "base", _MISSING),)
        else:
            return None  # A Symbol, a RealAlgebraicRoot, an int, a float, a missing field or any other class.
        walking.add(key)
        budget.tick(len(children))  # Every child pushed now is popped once, shared or not: paid before the push.
        stack.append((node, depth, (kind, children, exponent)))
        stack.extend((child, depth + 1, None) for child in reversed(children))
    return entries


def _bounded(low: int, high: int, limit: int) -> tuple[int, int]:
    """The ends, when both have at most limit bits; ResourceLimitError otherwise."""
    if max(-low, low, -high, high).bit_length() > limit:
        raise ResourceLimitError("Computation limit exceeded: integer bits")
    return low, high


def _product(budget: Budget, left: tuple[int, int], right: tuple[int, int], bits: int,
             limit: int) -> tuple[int, int]:
    """The enclosure of a product, rounded outward to the scale 2**bits."""
    (a, c), (g, h) = left, right
    left_size, right_size = max(-a, a, -c, c).bit_length(), max(-g, g, -h, h).bit_length()
    _charge(budget, left_size, right_size, 4)
    budget.tick((left_size + right_size) >> 9)  # The negations, comparisons and shifts, linear: a unit per 512 bits.
    products = (a * g, a * h, c * g, c * h)
    return _bounded(min(products) >> bits, -(-max(products) >> bits), limit)


def _square(budget: Budget, low: int, high: int, bits: int, limit: int) -> tuple[int, int]:
    """The enclosure of a square: the exact image of [low, high] under t -> t*t, rounded outward."""
    size = max(-low, low, -high, high).bit_length()
    _charge(budget, size, size, 2)
    budget.tick(size >> 8)  # The negations, comparisons and shifts, linear: a unit per 256 bits of the ends.
    if low >= 0:
        small, large = low * low, high * high
    elif high <= 0:
        small, large = high * high, low * low
    else:
        small, large = 0, max(low * low, high * high)
    return _bounded(small >> bits, -(-large >> bits), limit)


def _enclosure(budget: Budget, entries: list[Entry], bits: int) -> tuple[int, int]:
    """(L, H) with L/2**bits <= x <= H/2**bits for the expression whose last entry is the root, by the rules of the
    module docstring. ResourceLimitError when an end passes max_integer_bits + bits bits."""
    limit = budget.limits.max_integer_bits + bits
    one = 1 << bits
    ends: list[tuple[int, int]] = []
    for entry in entries:
        budget.tick()
        kind = entry[0]
        if kind == "rational":
            _, p, q = entry
            _charge(budget, p.bit_length() + bits, q.bit_length(), 2)
            shifted = p << bits
            low, high = shifted // q, -(-shifted // q)
        elif kind == "sqrt":
            _, p, q = entry
            _charge(budget, p.bit_length() + 2 * bits, q.bit_length())
            square = (p << 2 * bits) // q
            _charge(budget, square.bit_length(), square.bit_length())  # isqrt, charged like a product of its size.
            low = isqrt(square)
            high = low + 1
        elif kind == "add":
            low = high = 0
            for child in entry[1]:
                budget.tick()
                term_low, term_high = ends[child]
                low, high = low + term_low, high + term_high
        elif kind == "mul":
            children = entry[1]
            low, high = ends[children[0]]
            for child in children[1:]:
                low, high = _product(budget, (low, high), ends[child], bits, limit)
        else:
            _, child, exponent = entry
            if exponent == 0:
                low = high = one
            else:
                base = ends[child]
                low, high = base
                for digit in bin(exponent)[3:]:  # The bits after the leading one, from the high end.
                    low, high = _square(budget, low, high, bits, limit)
                    if digit == "1":
                        low, high = _product(budget, (low, high), base, bits, limit)
        ends.append(_bounded(low, high, limit))
    return ends[-1]


def _check_bits(budget: Budget, entries: list[Entry], bits: int, claim: Claim, scale: int) -> bool:
    """The enclosure at precision bits lies strictly inside the open cell of the claim (sign != 0, exact False)."""
    sign, magnitude, _ = claim
    low, high = _enclosure(budget, entries, bits)
    low_scaled, high_scaled = _times(budget, low, scale), _times(budget, high, scale)
    if sign > 0:
        return magnitude << bits < low_scaled and high_scaled < (magnitude + 1) << bits
    return -((magnitude + 1) << bits) < low_scaled and high_scaled < -(magnitude << bits)


def _square_root(budget: Budget, n: int) -> int | None:
    """The integer square root of n >= 0 when n is a perfect square; None otherwise."""
    _charge(budget, n.bit_length(), n.bit_length(), 2)
    root = isqrt(n)
    return root if root * root == n else None


def _coprime_base(budget: Budget, numbers: list[int]) -> tuple[list[int], dict[int, int | None]]:
    """The natural coprime base of non-square integers > 1, by factor refinement, and the square root of each member
    that is a square (None for the others). A split replaces m and a member c, with g = gcd(m, c) > 1, by g and the
    two numbers that remain of c and of m once every factor g is divided out (_strip), which divides the product of the
    numbers in play by g at least: the refinement terminates, its members stay pairwise coprime and every number is a
    product of powers of them. Stripping g whole refines a power such as 2**65001 against 2 in one split instead of
    one split per factor. Each gcd, division and square root is charged before it runs.

    Every number in play is a gcd or an exact quotient of the inputs, so the base reached is the unique natural coprime
    base of the inputs, whatever the order of the splits: the producer can compute the same members. Once a number is
    merged, every earlier member is a product of powers of the new members, distinct members of disjoint sets of
    them, and a member that is no square has a non-square among its own: the members that are no squares never
    become fewer. So the refinement stops with ResourceLimitError as soon as they pass MAX_RADICALS, where the finished
    base would pass it as well."""
    base: list[int] = []
    roots: dict[int, int | None] = {}
    for number in numbers:
        pending = [number]
        while pending:
            budget.tick()
            m = pending.pop()
            if m == 1:
                continue
            for j, c in enumerate(base):
                g = _gcd(budget, m, c)
                if g > 1:
                    del base[j]
                    pending.extend((g, _strip(budget, c, g)[0], _strip(budget, m, g)[0]))
                    break
            else:
                base.append(m)
        budget.tick(len(base))
        for c in base:
            if c not in roots:
                roots[c] = _square_root(budget, c)
        if sum(1 for c in base if roots[c] is None) > MAX_RADICALS:
            raise ResourceLimitError(f"Computation limit exceeded: more than {MAX_RADICALS} independent square roots")
    return base, roots


Element = dict  # An element of the algebra A: {subset mask: its coordinate, a nonzero Rational}; absent masks are 0.


def _gcd_extra(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, what count gcds of integers of the given bit lengths cost beyond the unit of the
    operation that forms them: the long division of the longer by the shorter, (long words - short words + 1) *
    (short words + 1) word steps on charge_gcd's scale of 64 per unit, and Lehmer's algorithm on the shorter
    (trusted.upoly.charge_gcd); nothing for integers of a few words."""
    long_words, short_words = max(left_bits, right_bits) >> 6, min(left_bits, right_bits) >> 6
    extra = count * ((long_words - short_words + 1) * (short_words + 1) >> 6)
    if extra:
        budget.tick(extra)
    charge_gcd(min(left_bits, right_bits), count)


def _lengths(budget: Budget, element: Element) -> tuple[int, int]:
    """(the longest numerator of the coordinates of an element, the length of the least common multiple of their
    denominators), in bits, the second at most max_integer_bits. Every denominator of a sum or product of
    coordinates of two elements divides the product of their two common multiples, which bounds the work of a whole
    operation of A in one batch (_charge_products); the gcds of the multiple are charged before they run."""
    budget.tick(len(element) + 1)
    limit = budget.limits.max_integer_bits
    numerator, common = 0, 1
    for c in element.values():
        numerator = max(numerator, c.numerator.bit_length())
        denominator = c.denominator
        if denominator != 1 and common.bit_length() <= limit:
            _gcd_extra(budget, common.bit_length(), denominator.bit_length())
            divisor = gcd(common, denominator)
            charge_product(common.bit_length(), denominator.bit_length(), 2)
            common = common // divisor * denominator
    return numerator, min(common.bit_length(), limit)


def _charge_sums(budget: Budget, count: int, numerator: int, denominator: int) -> None:
    """Charge, before they run, count sums of two Rationals whose numerators and denominators have at most those
    lengths, as model.Rational adds them: the cross products and the product of the denominators, the gcd of the
    denominators and that of the sum's numerator with their product, and the two divisions by it, each by the lengths
    of the integers it takes (_charge_gcd: a gcd with the denominator 1 costs a unit). One batch per operation of A,
    by the longest coordinates, which bounds every pair of them."""
    total, common = numerator + denominator + 1, 2 * denominator
    budget.tick(count)  # The loop; each Rational operation charges its own unit as well.
    charge_product(numerator, denominator, 3 * count)
    _gcd_extra(budget, denominator, denominator, count)
    _gcd_extra(budget, total, common, count)
    charge_product(total, common, 2 * count)


def _charge_products(budget: Budget, count: int, left: tuple[int, int], right: tuple[int, int],
                     weight_bits: int) -> None:
    """Charge, before they run, count terms a * b * weight of a product in A and their sums into it, for a and b with
    the lengths left and right (numerator, common denominator: _lengths) and a weight of weight_bits: as
    model.Rational multiplies, the gcds of each numerator with the other denominator and the divisions by them, the
    products of the numerators and of the denominators, then the product by the integer weight with its gcd against
    the denominator and the divisions by it; then the sum of the term into the result (_charge_sums)."""
    (na, da), (nb, db) = left, right
    budget.tick(count)  # The loop; each Rational operation charges its own unit as well.
    charge_product(na, nb, count)
    charge_product(da, db, count)
    charge_product(na, db, 2 * count)
    charge_product(nb, da, 2 * count)
    _gcd_extra(budget, na, db, count)
    _gcd_extra(budget, nb, da, count)
    if weight_bits > 1:
        charge_product(na + nb, weight_bits, count)
        _gcd_extra(budget, weight_bits, da + db, count)
        charge_product(na + nb + weight_bits, da + db, 2 * count)
    # The term sums into a coordinate whose denominator divides the product of the two common multiples, and whose
    # size is at most count times the largest term: its numerator has at most the bits below.
    limit = budget.limits.max_integer_bits
    _charge_sums(budget, count, min(na + nb + weight_bits + da + db + count.bit_length(), limit), min(da + db, limit))


def _accumulate(budget: Budget, total: Element, right: Element) -> None:
    """total += right in A, in place, coordinate by coordinate (the rule of trusted.surd.Field.add on the nonzero
    coordinates only: a zero coordinate adds nothing); each sum of two coordinates charged before it runs by the
    lengths of its own integers, so that a sum of many terms costs the sizes of the terms, not the size of the sum
    so far once per term."""
    budget.tick(len(right) + 1)
    for mask, b in right.items():
        a = total.get(mask)
        if a is None:
            total[mask] = b
            continue
        _charge_sums(budget, 1, max(a.numerator.bit_length(), b.numerator.bit_length()),
                     max(a.denominator.bit_length(), b.denominator.bit_length()))
        c = a + b
        if c.numerator:
            total[mask] = c
        else:
            del total[mask]


def _element_product(budget: Budget, weight: tuple[int, ...], weight_bits: int, left: Element,
                     right: Element) -> Element:
    """left * right in A by trusted.surd.Field.mul's rule, e_s * e_u = weight[s & u] * e_(s xor u), on the pairs of
    nonzero coordinates only (a pair with a zero coordinate contributes 0 there); every pair charged before the
    products run, by the longest coordinates and weight_bits, at least the length of every weight."""
    pairs = len(left) * len(right)
    _charge_products(budget, pairs, _lengths(budget, left), _lengths(budget, right), weight_bits)
    product: Element = {}
    for s, a in left.items():
        for u, b in right.items():
            w = weight[s & u]
            term = a * b if w == 1 else a * b * w
            mask = s ^ u
            c = product.get(mask)
            product[mask] = term if c is None else c + term
    budget.tick(len(product))
    return {mask: c for mask, c in product.items() if c.numerator}


def _formal_value(budget: Budget, entries: list[Entry]) -> Element | None:
    """The formal value of the expression in the algebra A of the module docstring, as an Element: the coordinates of
    the subsets over the members b_1, ..., b_t, those that are 0 left out; None when a radicand fails its verified
    identity (which the decomposition below never gives). ResourceLimitError for more than MAX_RADICALS coprime
    non-square radicands, or past the budget.

    The elements are sparse: A has 2**t coordinates, and an expression of a few thousand nodes over ten radicands
    would spend 1,024 units of work on every node in a dense tuple, beyond the default budget for values that the
    producer reaches in a few thousand units (tests: test_field_evaluation_is_sparse). Leaving out the zero
    coordinates computes the same element of A with the same ring operations: it changes the cost, never the answer."""
    radicands: dict[int, int | None] = {}  # n = p*q -> its square root, None for a non-square; in order of appearance.
    for entry in entries:
        budget.tick()
        if entry[0] == "sqrt":
            n = _times(budget, entry[1], entry[2])
            if n not in radicands:
                radicands[n] = _square_root(budget, n)
    base, roots = _coprime_base(budget, [n for n, root in radicands.items() if root is None])
    generators = [c for c in base if roots[c] is None]
    slot = {c: j for j, c in enumerate(generators)}
    size = 1 << len(generators)
    weight_bits = sum(c.bit_length() for c in generators)  # At least the length of every weight of a subset.
    budget.tick(4 * size)  # trusted.surd.Field builds the weights of every subset, its tuples and the fields below.
    charge_product(weight_bits, max((c.bit_length() for c in generators), default=0), size)
    field = Field(tuple(generators))
    weight = field.weight  # The multiplication table of A: e_s * e_u = weight[s & u] * e_(s xor u).
    # sqrt(n) = c * sqrt(prod b_j over the subset S), from n == c*c * prod b_j verified exactly.
    terms: dict[int, tuple[int, int]] = {}
    for n, root in radicands.items():
        if root is not None:
            continue
        rest, coefficient, mask = n, 1, 0
        for c in base:
            rest, exponent = _strip(budget, rest, c)
            if exponent:
                coefficient = _times(budget, coefficient, _power(budget, c, exponent >> 1))
                if exponent & 1:
                    member_root = roots[c]
                    if member_root is not None:
                        coefficient = _times(budget, coefficient, member_root)
                    else:
                        mask |= 1 << slot[c]
        if rest != 1 or _times(budget, _times(budget, coefficient, coefficient), field.weight[mask]) != n:
            return None
        terms[n] = (coefficient, mask)
    values: list[Element] = []
    for entry in entries:
        budget.tick()
        kind = entry[0]
        if kind == "rational":
            value: Element = {0: trusted_rational(entry[1], entry[2])} if entry[1] else {}
        elif kind == "sqrt":
            n = _times(budget, entry[1], entry[2])
            root = radicands[n]
            if root is not None:
                value = {0: Rational(root, entry[2])} if root else {}
            else:
                coefficient, mask = terms[n]
                value = {mask: Rational(coefficient, entry[2])}  # coefficient >= 1: a nonzero coordinate.
        elif kind == "add":
            value = dict(values[entry[1][0]])
            budget.tick(len(value))
            for child in entry[1][1:]:
                _accumulate(budget, value, values[child])
        elif kind == "mul":
            value = values[entry[1][0]]
            for child in entry[1][1:]:
                value = _element_product(budget, weight, weight_bits, value, values[child])
        else:
            _, child, exponent = entry
            value = {0: trusted_rational(1, 1)}
            if exponent:
                base_value = value = values[child]
                for digit in bin(exponent)[3:]:
                    budget.tick()
                    value = _element_product(budget, weight, weight_bits, value, value)
                    if digit == "1":
                        value = _element_product(budget, weight, weight_bits, value, base_value)
        values.append(value)
    return values[-1]


def _check_field(budget: Budget, entries: list[Entry], fraction: tuple[int, int], claim: Claim, scale: int) -> bool:
    """The expression equals r = fraction in the algebra, and the record passes the Rational conditions for r: the
    formal value has no nonzero coordinate but the constant one, and that one is r (absent, 0, when r is 0)."""
    value = _formal_value(budget, entries)
    if value is None:
        return False
    budget.tick(len(value))
    if any(mask for mask in value):
        return False
    constant = value.get(0)
    if (fraction[0] == 0) != (constant is None) or (constant is not None and constant != trusted_rational(*fraction)):
        return False
    return _check_rational(budget, *fraction, claim, scale)


# ---------------------------------------------------------------------------------------------------------------
# The operation

def _check(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    """Condition (0), then the conditions of the value's kind; cheap tests before any arithmetic."""
    if args.keys() != ARGUMENTS:
        return False
    places = args["places"]
    if type(places) is not int or not 0 <= places <= MAX_PLACES:
        return False
    claim = _claim(budget, value, places)
    if claim is None:
        return False
    number = args["value"]
    kind = type(number)
    if kind is int or kind is Rational:
        if evidence:
            return False
        if kind is int:
            budget.check_int(number)
            fraction: tuple[int, int] | None = (number, 1)
        else:
            fraction = _argument_fraction(budget, number)
        if fraction is None:
            return False
        return _check_rational(budget, *fraction, claim, _power(budget, 10, places))
    if _is_class(number, ROOT):
        if len(evidence) != 1 or type(evidence[0]) is not str or evidence[0] != "root":
            return False
        root = _root(budget, number)
        if root is None:
            return False
        return _check_root(budget, *root, claim, places, _power(budget, 10, places))
    if kind not in (Add, Mul, Pow, Sqrt) or len(evidence) != 2 or type(evidence[0]) is not str:
        return False
    tag, payload = evidence
    if tag == "bits":
        if type(payload) is not int or not 1 <= payload <= budget.limits.max_integer_bits:
            return False
        if claim[0] == 0 or claim[2]:
            return False  # An enclosure inside an open cell proves a nonzero number off every cell boundary.
        entries = _parse(budget, number)
        return entries is not None and _check_bits(budget, entries, payload, claim, _power(budget, 10, places))
    if tag == "rational":
        fraction = _rational_fields(payload)
        limit = budget.limits.max_integer_bits
        if (fraction is None or fraction[0].bit_length() > limit or fraction[1].bit_length() > limit
                or not _lowest(budget, *fraction)):
            return False
        entries = _parse(budget, number)
        return entries is not None and _check_field(budget, entries, fraction, claim, _power(budget, 10, places))
    return False


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of digits on args, by the theorems of the module. False
    for another operation, for arguments other than exactly value and places, and for any malformed value or
    evidence; ResourceLimitError for an argument beyond the budget or past the budget."""
    if type(operation) is not str or operation != OPERATION or type(args) is not dict or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        return _check(budget, args, value, evidence)
