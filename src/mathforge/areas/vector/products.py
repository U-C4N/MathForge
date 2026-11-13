"""Products of exact rational vectors (mathforge.vector): the dot product, the cross product, the orthogonal projection
and the scalar triple product.

Every argument is a vector of exact rationals: MCP kind "rational_list", so operations_v6.run freezes a list into a
tuple and already rejects a bool, a float, a str or None in it as invalid input. The arguments are unannotated, as
v0.7's rational_list arguments are (mean(data), covariance(x, y)), since a list is accepted as well as a tuple, and no
argument has a default. A value is canonical: a Rational, never an int (an integral value is Rational(n, 1)), and a
vector is a tuple of Rationals. The family defines no record class, since every value is a Rational or a tuple of
Rationals, which serialize and render without one: TYPES is empty and there is no RENDER. For each operation, its
domain, its value and the evidence that its checker, mathforge.certify.vector.products, verifies:

- dot(a, b): a.b = sum_i a_i b_i for vectors of the same length n >= 1; vectors of different lengths and empty
  vectors are invalid input. Value: a Rational. Evidence ().
- cross(a, b): a x b = (a_2 b_3 - a_3 b_2, a_3 b_1 - a_1 b_3, a_1 b_2 - a_2 b_1) for vectors of length 3; any other
  length is invalid input. Value: a tuple of three Rationals ((0, 0, 0) as Rationals for parallel vectors).
  Evidence ().
- projection(a, b): the orthogonal projection (a.b / b.b) b of a onto b, for vectors of the same length n >= 1 and
  b != 0; different lengths, empty vectors and the zero vector b are invalid input. Value: a tuple of n Rationals.
  Evidence ().
- triple_product(a, b, c): a.(b x c), the determinant of the 3 x 3 matrix with rows a, b and c, for vectors of
  length 3; any other length is invalid input. Value: a Rational. Evidence ().

The checker needs no evidence: it decides each value by the identities of the wave's card (the polarization identity
for dot; orthogonality, Lagrange's identity and the orientation for cross; parallelism and orthogonality for
projection; the rule of Sarrus for triple_product), docs/v09/cards/m4-linalg-geometry-stats-discrete.md.
mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/vector/test_products.py is the definition of done.

How produce() computes. Nothing here is a proof: the checker decides every value with identities of its own on
integer forms of the vectors (each vector times a common multiple of its denominators), and never forms the sums
below.

- Validation first: the operation's argument names; each vector a tuple (or a list, for a direct call); the lengths of
  the card's domains; then every entry, an exact int or a Rational of exact ints in lowest terms with a positive
  denominator (a Rational forged with object.__new__ may hold anything, and the checker refuses a request that holds
  one), an integer over max_integer_bits being a ResourceLimitError; last, for projection, b != 0.
- Every number is a pair (numerator, denominator) in lowest terms. A sum of products sum_i x_i y_i, with x_i = n_i/u_i
  and y_i = m_i/v_i, runs over one common denominator: each term n_i m_i / (u_i v_i) joins N / L unreduced, where L is
  the least common multiple of the denominators u_i v_i of the terms so far. A term whose denominator divides L adds
  n_i m_i (L / (u_i v_i)) to N; for another, with g = gcd(u_i v_i, L), N and L are first multiplied by (u_i v_i) / g.
  One gcd reduces N / L at the end. L divides K^2 for the least common multiple K of the denominators of the nonzero
  terms' entries, which the checker's polarization forms, so these integers never grow longer than the checker's: a sum
  reduced after every term paid two gcds of the long denominators per term, four times the checker's work for vectors
  with one shared denominator of 10,000 bits.
- The other steps are the exact rational arithmetic of Henrici and Knuth (The Art of Computer Programming, section
  4.5.1) that Python's fractions module uses, every result in lowest terms again. A product n/u * m/v cancels gcd(n, v)
  and gcd(m, u) before it multiplies. A sum n/u + m/v with g = gcd(u, v), u = g s and v = g w forms t = n w + m s over
  the least common multiple g s w and cancels h = gcd(t, g): t is prime to s and to w, so h is gcd(t, g s w). A sum
  that vanishes has u = v, so s = w = 1 and h = g, and it gives 0/1.
- dot: the sum of the products a_i b_i.
- cross: the cofactor formula above, each component a difference of two products.
- projection: t = (a.b) / (b.b), two sums of products, where b.b > 0 since b != 0; then p_j = t b_j.
- triple_product: a.(b x c), the cross product of b and c and then its sum of products with a, the expansion of
  det[a, b, c] along its first row, where the checker sums the six signed products of the rule of Sarrus.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges a unit for the call and one per entry read, so that every valid input
costs at least three, and charges every further step before it runs, on the scale of the cost model of
trusted.integers that the checker uses as well: three units for each arithmetic step, which a step's Python overhead
needs; product_work for a product and a unit per 64 words of its result; division_work by the length of the quotient
for an exact division; for a gcd, a long division and a pass over the longer operand, then Lehmer's algorithm on the
shorter one, one unit per 64 squared 64-bit words (the scale of trusted.upoly.charge_gcd); and a unit per 256 words of
a linear pass. A value whose integers exceed max_integer_bits is a ResourceLimitError, as the inspection of the value in
operations_v6.run would raise.

The producer does not predict the checker's work, and needs about as much work as the checker or less: a sum of
products forms one product per term where the polarization identity forms three squares, on integers no longer than
the checker's. Where the checker's budget does not suffice, operations_v6.run reports resource_limit_exceeded instead
of a value; so does the producer itself for a_i = 1/q_i and b_i = q_i with many long distinct q_i, whose terms cancel
to 1 but whose common denominator, like the checker's integer forms, carries every q_i. Measured with CPython on one
machine under the default limits: dot on 20,000 entries of 64-bit integers and 64-bit numerators over 2^0..2^7 takes
about 235,000 units (0.03 s; the checker about 450,000), dot on 1,000 entries over one shared 10,000-bit denominator
about 1,000,000 (the checker about 1,120,000), dot and cross on three 32,000-bit integers about 3,000 and 6,000 units,
triple_product on 21,000-bit integers about 5,000 and projection on 3,000 entries about 84,000. A unit takes about 0.1
microseconds on short numbers and up to about 0.45 where products of a few thousand bits prevail, which product_work
charges the least on the scale that the checker shares.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, computation
from ...model import Rational, trusted_rational
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

# The scope text of every spec of wave M4 (docs/v09/cards/m4-linalg-geometry-stats-discrete.md).
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

# a, b and c are unannotated, as v0.7's rational_list arguments are: a list is accepted as well as a tuple.
_A, _B, _C = (Argument(name, "", None, "rational_list") for name in ("a", "b", "c"))

SPECS = (
    OperationSpec("dot", "vector", (_A, _B), "Rational",
                  "Compute the dot product of two exact rational vectors.", SCOPE, version=6),
    OperationSpec("cross", "vector", (_A, _B), "tuple[Rational, Rational, Rational]",
                  "Compute the cross product of two exact rational vectors in three dimensions.", SCOPE, version=6),
    OperationSpec("projection", "vector", (_A, _B), "tuple[Rational, ...]",
                  "Compute the orthogonal projection of an exact rational vector onto a nonzero rational vector.",
                  SCOPE, version=6),
    OperationSpec("triple_product", "vector", (_A, _B, _C), "Rational",
                  "Compute the scalar triple product of three exact rational vectors in three dimensions.", SCOPE,
                  version=6),
)

# No record classes: every value is a Rational or a tuple of Rationals, which serialize and render without one.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.vector.products"

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
# The length that every vector of an operation must have; None: any common length of at least 1.
_DIMENSIONS: dict[str, int | None] = {"dot": None, "cross": 3, "projection": None, "triple_product": 3}
_STEP = 3  # The units of an arithmetic step beyond its integer work: its Python overhead, on the scale of a unit.

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational in lowest terms, denominator >= 1.
_ZERO: Pair = (0, 1)


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple[()]]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name and
    kind; the module states the domains, the algorithms and the evidence, always (). InvalidInput for an argument
    outside the operation's domain, UnsupportedOperation for another operation, ResourceLimitError past the budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the vector products family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    vectors = [args[name] for name in names]
    for name, vector in zip(names, vectors):
        if type(vector) is not tuple and type(vector) is not list:
            raise InvalidInput(f"{operation} requires {name} as a tuple or list of exact rationals")
    dimension, length = _DIMENSIONS[operation], len(vectors[0])
    if dimension is not None:
        if any(len(vector) != dimension for vector in vectors):
            raise InvalidInput(f"{operation} requires vectors of length {dimension}: {', '.join(names)}")
    elif any(len(vector) != length for vector in vectors):
        raise InvalidInput(f"{operation} requires a and b of the same length")
    elif not length:
        raise InvalidInput(f"{operation} requires nonempty vectors a and b")
    with computation() as budget:
        budget.tick()  # The call itself; every entry read costs a unit more.
        pairs = [_pairs(budget, operation, name, vector) for name, vector in zip(names, vectors)]
        return _PRODUCERS[operation](budget, *pairs), ()


def _pairs(budget: Budget, operation: str, name: str, vector: tuple[object, ...] | list[object]) -> list[Pair]:
    """The entries of a vector as pairs in lowest terms. InvalidInput for anything but an exact int or a canonical
    Rational, ResourceLimitError for an integer over max_integer_bits."""
    pairs = []
    for entry in vector:
        budget.tick()
        kind = type(entry)
        if kind is int:  # Neither a bool nor a float nor a str is an exact number.
            budget.check_int(entry)
            pairs.append((entry, 1))
            continue
        if kind is not Rational:
            raise InvalidInput(f"{operation} requires exact rationals in {name}: ints or Rationals")
        numerator = getattr(entry, "numerator", None)  # A Rational forged with object.__new__ may lack a slot.
        denominator = getattr(entry, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            raise InvalidInput(f"{operation} requires canonical Rationals in {name}")
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator != 1:
            budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
            if gcd(numerator, denominator) != 1:
                raise InvalidInput(f"{operation} requires canonical Rationals in {name}: lowest terms")
        pairs.append((numerator, denominator))
    return pairs


# ---------------------------------------------------------------------------------------------------------------
# The operations, on validated pairs of the operation's lengths

def _dot(budget: Budget, a: list[Pair], b: list[Pair]) -> Rational:
    """a.b, the sum of the products a_i b_i."""
    return _rational(budget, _inner(budget, a, b))


def _cross(budget: Budget, a: list[Pair], b: list[Pair]) -> tuple[Rational, ...]:
    """a x b by the cofactor formula."""
    return tuple(_rational(budget, component) for component in _cross_pairs(budget, a, b))


def _projection(budget: Budget, a: list[Pair], b: list[Pair]) -> tuple[Rational, ...]:
    """(a.b / b.b) b, for b != 0, which is validated here before anything is computed."""
    budget.tick(1 + (len(b) >> 6))
    if not any(numerator for numerator, _ in b):
        raise InvalidInput("projection requires a nonzero vector b: the zero vector spans no line")
    t = _quotient(budget, _inner(budget, a, b), _inner(budget, b, b))  # b.b > 0.
    return tuple(_rational(budget, _product(budget, t, y)) for y in b)


def _triple_product(budget: Budget, a: list[Pair], b: list[Pair], c: list[Pair]) -> Rational:
    """a.(b x c), the expansion of det[a, b, c] along its first row."""
    return _rational(budget, _inner(budget, a, _cross_pairs(budget, b, c)))


_PRODUCERS: dict[str, Callable[..., object]] = {
    "dot": _dot,
    "cross": _cross,
    "projection": _projection,
    "triple_product": _triple_product,
}


def _inner(budget: Budget, xs: list[Pair], ys: list[Pair]) -> Pair:
    """The sum of the products x_i y_i of vectors of the same length, in lowest terms: N / L over the least common
    multiple L of the denominators of the terms so far, each term n m / (u v) of x_i = n/u and y_i = m/v unreduced;
    one gcd reduces N / L at the end (module docstring)."""
    total, common = 0, 1
    for (n, u), (m, v) in zip(xs, ys):
        if not n or not m:  # A term 0 leaves N / L as it is.
            budget.tick()
            continue
        # The two products, and the sum of the term while N, L and the term are integers.
        budget.tick(_STEP + _product_work(n.bit_length(), m.bit_length())
                    + _product_work(u.bit_length(), v.bit_length()) + _linear(total))
        term, denominator = n * m, u * v
        if common == 1 and denominator == 1:
            total += term
            continue
        common_bits, denominator_bits = common.bit_length(), denominator.bit_length()
        budget.tick(_STEP + _quotient_work(common_bits, denominator_bits))
        scale, remainder = divmod(common, denominator)
        if remainder:  # L becomes L (q / g) for q = u v and g = gcd(q, L) = gcd(q, L mod q); N follows.
            budget.tick(_gcd_work(denominator_bits, remainder.bit_length()))
            divisor = gcd(denominator, remainder)
            divisor_bits = divisor.bit_length()
            budget.tick(_STEP + _quotient_work(denominator_bits, divisor_bits)
                        + _quotient_work(common_bits, divisor_bits))
            factor, scale = denominator // divisor, common // divisor
            factor_bits = factor.bit_length()
            budget.tick(_STEP + _product_work(total.bit_length(), factor_bits)
                        + _product_work(common_bits, factor_bits))
            total, common = total * factor, common * factor
        budget.tick(_STEP + _product_work(term.bit_length(), scale.bit_length()) + _linear(total))
        total += term * scale  # The term n m / (u v) is term * scale / L.
    if not total:
        return _ZERO
    if common == 1:
        return total, 1
    return _cancelled(budget, total, common)


def _cross_pairs(budget: Budget, xs: list[Pair], ys: list[Pair]) -> list[Pair]:
    """x x y = (x_2 y_3 - x_3 y_2, x_3 y_1 - x_1 y_3, x_1 y_2 - x_2 y_1) for vectors of length 3."""
    (x1, x2, x3), (y1, y2, y3) = xs, ys
    return [_difference(budget, _product(budget, x2, y3), _product(budget, x3, y2)),
            _difference(budget, _product(budget, x3, y1), _product(budget, x1, y3)),
            _difference(budget, _product(budget, x1, y2), _product(budget, x2, y1))]


def _rational(budget: Budget, pair: Pair) -> Rational:
    """The canonical Rational of a pair in lowest terms; ResourceLimitError for an integer over max_integer_bits."""
    numerator, denominator = pair
    budget.check_int(numerator)
    budget.check_int(denominator)
    return trusted_rational(numerator, denominator)  # Lowest terms and a positive denominator, by the arithmetic.


# ---------------------------------------------------------------------------------------------------------------
# Exact rational arithmetic on pairs in lowest terms (Henrici; Knuth, TAOCP 4.5.1), each step charged before it runs

def _product(budget: Budget, x: Pair, y: Pair) -> Pair:
    """x y in lowest terms: for x = n/u and y = m/v, gcd(n, v) and gcd(m, u) cancel before the integers multiply, and
    what remains is coprime, as n is prime to u and m to v."""
    (n, u), (m, v) = x, y
    if not n or not m:
        budget.tick()
        return _ZERO
    if v != 1:
        n, v = _cancelled(budget, n, v)
    if u != 1:
        m, u = _cancelled(budget, m, u)
    budget.tick(_STEP + _product_work(n.bit_length(), m.bit_length()) + _product_work(u.bit_length(), v.bit_length()))
    return n * m, u * v


def _sum(budget: Budget, x: Pair, y: Pair) -> Pair:
    """x + y in lowest terms (module docstring): t = n w + m s over g s w, then the cancellation of gcd(t, g)."""
    (n, u), (m, v) = x, y
    if u == 1 and v == 1:
        budget.tick(_STEP + _linear(n, m))
        return n + m, 1
    if u == 1 or v == 1:  # n + m/v = (n v + m)/v in lowest terms: gcd(n v + m, v) = gcd(m, v) = 1.
        if u != 1:
            (n, u), (m, v) = (m, v), (n, u)
        budget.tick(_STEP + _product_work(n.bit_length(), v.bit_length()) + _linear(m))
        return n * v + m, v
    n_bits, m_bits, u_bits, v_bits = n.bit_length(), m.bit_length(), u.bit_length(), v.bit_length()
    budget.tick(_gcd_work(u_bits, v_bits))
    g = gcd(u, v)
    if g == 1:
        budget.tick(_STEP + _product_work(n_bits, v_bits) + _product_work(m_bits, u_bits)
                    + _product_work(u_bits, v_bits))
        return n * v + m * u, u * v
    g_bits = g.bit_length()
    budget.tick(_STEP + _quotient_work(u_bits, g_bits) + _quotient_work(v_bits, g_bits))
    s, w = u // g, v // g
    s_bits = s.bit_length()
    budget.tick(_STEP + _product_work(n_bits, w.bit_length()) + _product_work(m_bits, s_bits))
    t = n * w + m * s
    t_bits = t.bit_length()
    budget.tick(_gcd_work(t_bits, g_bits))
    h = gcd(t, g)  # gcd(t, g s w): t is prime to s and to w.
    if h != 1:
        h_bits = h.bit_length()
        budget.tick(_STEP + _quotient_work(t_bits, h_bits) + _quotient_work(v_bits, h_bits))
        t, v = t // h, v // h
    budget.tick(_STEP + _product_work(s_bits, v.bit_length()))
    return t, s * v


def _difference(budget: Budget, x: Pair, y: Pair) -> Pair:
    """x - y in lowest terms: x + (-y)."""
    budget.tick(1 + _linear(y[0]))
    return _sum(budget, x, (-y[0], y[1]))


def _quotient(budget: Budget, x: Pair, y: Pair) -> Pair:
    """x / y for y > 0 (projection divides by b.b only): x times v/m for y = m/v, a pair in lowest terms as well."""
    m, v = y
    budget.tick()
    return _product(budget, x, (v, m))


def _cancelled(budget: Budget, numerator: int, denominator: int) -> tuple[int, int]:
    """numerator and denominator divided by their gcd."""
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    budget.tick(_gcd_work(numerator_bits, denominator_bits))
    divisor = gcd(numerator, denominator)
    if divisor == 1:
        return numerator, denominator
    divisor_bits = divisor.bit_length()
    budget.tick(_STEP + _quotient_work(numerator_bits, divisor_bits) + _quotient_work(denominator_bits, divisor_bits))
    return numerator // divisor, denominator // divisor


# ---------------------------------------------------------------------------------------------------------------
# Costs, on the scale of trusted.integers. Every arithmetic step calls these several times, and an import of
# trusted.integers inside them took more time than the arithmetic that they charge (about 0.2 microseconds per call,
# half of the producer's time on short rationals), so _product_work and _quotient_work restate the formulas of its
# product_work and division_work; tests/v09/vector/test_products.py pins them to those functions.

def _product_work(left_bits: int, right_bits: int) -> int:
    """The units of a product of integers of these lengths beyond one: trusted.integers.product_work, a unit per 256
    pairs of their 64-bit words, and a unit per 64 words of the result. product_work charges nothing for a factor
    shorter than a word, and a long integer times a short one takes about a microsecond per 400 words (measured: 8
    microseconds for 128,000 bits times 64 bits), which this pass charges as the pass of a gcd over its longer operand
    is charged."""
    return ((left_bits >> 6) * (right_bits >> 6) >> 8) + ((left_bits + right_bits) >> 12)


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient: trusted.integers.division_work of a
    dividend one word longer than the quotient, since division_work charges the words of the dividend and an exact
    quotient may be far shorter than the dividend."""
    return ((max(dividend_bits - divisor_bits, 0) + 64) >> 6) * (1 + (divisor_bits >> 6)) >> 5


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: one, a long division of the longer by the shorter and a pass
    over the longer, then Lehmer's algorithm on the shorter, one unit per 64 squared 64-bit words (the scale of
    trusted.upoly.charge_gcd)."""
    shorter, longer = min(left_bits, right_bits), max(left_bits, right_bits)
    words = shorter >> 6
    return 1 + _quotient_work(longer, shorter) + (longer >> 12) + (words * words >> 6)


def _linear(*numbers: int) -> int:
    """The units of a linear pass over these integers (a sum, a negation, a comparison): one per 256 64-bit words."""
    return sum(number.bit_length() for number in numbers) >> 14
