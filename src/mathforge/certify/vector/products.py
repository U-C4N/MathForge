"""Independent checker of the family mathforge.areas.vector.products: dot, cross, projection and triple_product.

The checker never reruns the producer's formulas. It decides each claim by the identities of the wave's card
(docs/v09/cards/m4-linalg-geometry-stats-discrete.md), exactly and with integers only. A vector x of exact rationals
x_i = n_i/d_i is read as X / D, with D the least common multiple of the d_i and the integers X_i = n_i (D / d_i). Below,
A, B and C are the integer forms of the request's vectors a, b and c, with the denominators D_a, D_b and D_c; X.Y is
the dot product of integer vectors, |X|^2 = X.X, and det[X, Y, Z] is the determinant of the 3 x 3 matrix with the rows
X, Y and Z, formed by the rule of Sarrus (the six signed products of the Leibniz formula). A rational has exactly one
canonical form, a Rational in lowest terms with a positive denominator, so a True below means that the claimed value
is the canonical value itself.

The theorem that makes True imply the claim, for each operation:

dot(a, b) = v, evidence ().
    a and b have the same length n >= 1, v = p/q is a canonical Rational, and
        q (|K a + K b|^2 - |K a|^2 - |K b|^2) = 2 p K^2,
    where the three sums of squares run over the coordinates J at which a_i and b_i are both nonzero and K is the
    least common multiple of the denominators of those a_i and b_i (K = 1 for an empty J), so that K a and K b are
    integer vectors on J. The polarization identity |X + Y|^2 - |X|^2 - |Y|^2 = 2 X.Y (expand the square: the dot
    product is bilinear and symmetric), applied to K a and K b on J, turns this into p/q = (K a).(K b) / K^2 = the sum
    of a_i b_i over J = a.b, since a coordinate outside J has a_i b_i = 0 (it would add (x + 0)^2 - x^2 - 0^2 = 0 to
    the left side). The squares are formed without squaring the long entries of K a: K a_i + K b_i = s (K / t) for
    a_i + b_i = s / t with t = lcm(d_i, e_i) or t = K (_polarization), and (K / t)^2 is formed as the square of K / t
    or as K^2 / t^2, whichever the cost model charges less (_Squares). Vectors of different lengths, and empty ones,
    have no dot product, and every claim on them answers False.

cross(a, b) = c, evidence ().
    a and b have length 3, c is a tuple of three canonical Rationals c_i = n_i/d_i, and every d_i divides M = D_a D_b,
    so that C = M c is an integer vector: a x b = (A x B) / M, so the true value passes this test, and a forged
    denominator fails it before anything is computed from it. The integer vectors meet the card's four conditions
        C.A = 0, C.B = 0, |C|^2 = |A|^2 |B|^2 - (A.B)^2 (Lagrange) and det[A, B, C] = |C|^2 (orientation),
    which force C = A x B =: N. When A and B are linearly dependent, |A|^2 |B|^2 = (A.B)^2 (the equality case of
    Cauchy and Schwarz), so |C|^2 = 0 and C = 0 = N. Otherwise the vectors orthogonal to both A and B form the line
    through N != 0, so C = s N for a rational s; Lagrange's identity |N|^2 = |A|^2 |B|^2 - (A.B)^2 makes the third
    condition s^2 = 1, and det[A, B, C] = C.(A x B) = s |N|^2, equal to |C|^2 = |N|^2, leaves s = 1. Hence
    c = (A x B) / M = a x b, by bilinearity. (Lagrange's condition and the orientation alone already force C = N:
    with |C|^2 = |N|^2 and C.N = |C|^2, |C - N|^2 = |C|^2 - 2 C.N + |N|^2 = 0. The orthogonality conditions are
    checked as the card states them; they cost two dot products.) Vectors of another length have no cross product.

projection(a, b) = p, evidence ().
    a and b have the same length n >= 1, b != 0, p is a tuple of n Rationals p_j = r_j/s_j, and b_j = m_j/e_j in lowest
    terms. With k the first index where b_k != 0, u / w the lowest terms of p_k / b_k = (r_k e_k) / (s_k m_k) with
    w > 0, J now the coordinates where b_i != 0 (a_i may be 0 there), K the least common multiple of the denominators
    of a_i and b_i for i in J, P = |K a + K b|^2 - |K a|^2 - |K b|^2 and R = |K b|^2, both sums over J:
        (r_j, s_j) is the canonical pair of (u / w) b_j for every j (_scaled_by), and w P = 2 u R.
    The first makes p = t b with t = u / w = p_k / b_k, and every p_j canonical, as a canonical pair is (for j = k it
    holds exactly when p_k is canonical); it is the card's condition p = t b, with t read from the claim, never formed
    from a.b and |b|^2 as the producer forms it. J holds every coordinate where a_i b_i != 0 and every one where
    b_i != 0, so P = 2 K^2 a.b, as for dot, and R = K^2 |b|^2 > 0. The second is then t |b|^2 = a.b, that is
    (a - p).b = 0 for p = t b, multiplied by 2 w K^2 > 0. Then |b|^2 > 0 gives t = a.b / |b|^2: p is the orthogonal
    projection of a onto b, which is unique. Different lengths, empty vectors and b = 0 have no projection.

triple_product(a, b, c) = v, evidence ().
    a, b and c have length 3, v = p/q is a canonical Rational, and q det[A, B, C] = p D_a D_b D_c. The determinant is
    linear in each row, so det[A, B, C] = D_a D_b D_c det[a, b, c], and det[a, b, c] = a.(b x c) is its expansion
    along the first row: p/q = a.(b x c). An N-version check (the rule of Sarrus on the integer forms, where the
    card's producer forms a.(b x c) with Rationals): no certificate is cheaper than the six products. Vectors of
    another length have no triple product.

Shapes and budget. The operation must be one of the four names; args a dict of exactly its argument names, each a
tuple (a request holds lists frozen into tuples) of exact ints and Rationals of exact ints with a positive denominator
in lowest terms; the value a Rational, or a tuple of Rationals, of exact ints in lowest terms with a positive
denominator (an int value is not canonical, and a list or a tuple subclass is no tuple); the evidence the empty tuple.
A bool, a float, a str, an int subclass or a Rational subclass is no exact number, and anything else answers False,
never an exception. A value integer over max_integer_bits answers False, as no computation under the budget forms it.
A request integer over max_integer_bits raises ResourceLimitError, as operations_v6.run and the verifier's inspection
raise before a checker runs, and so does work past max_work: a claim is then unverified, never verified. Every loop is
charged before it runs, a long integer operation on the scale of the cost model of trusted.integers (product_work for
a product, division_work for the length of a quotient, a gcd quadratic in its shorter operand as
trusted.upoly.charge_gcd charges it, a unit per 256 words of a linear pass), and no power is formed by the builtin pow
or by **: a square is a product. A forged value adds little work to the request's: dot and triple_product claim one
Rational, a claimed vector has the length of the request's vectors; for cross its denominators must divide D_a D_b
before anything else is computed from it, so that each of its scaled integers is at most max_integer_bits longer than
D_a D_b, and for projection each entry is compared with a pair formed from t = p_k / b_k and the request, whose
integers are as long as t and b_j together, before the request's sums of squares are formed.

Cost. An earlier form of this checker (wave M4 review) squared the n entries of the integer forms D_a a and D_b b,
each as long as D, which costs n products quadratic in D, and D grows with n for vectors of distinct denominators:
cubic in n. For projection it also cross multiplied the long scaled claim P with B and tested every p_j for lowest
terms with a gcd quadratic in the length of t. Formed that way, the dot claim of 800 entries of 64-bit integers and
64-bit rationals (a value of about 44,000 bits that the producer forms with about 57,000 units) and the projection
claim of 400 such entries (about 130,000 producer units) ran out of the default budget, and so did every claim whose a
holds many long denominators where b is 0. The squares (K / t)^2 of _Squares, the sums over J and the canonical pairs
of _scaled_by keep every product per entry a short integer times a long one: these claims now take about 160,000 and
110,000 units. The conditions decided are equivalent, as the theorems above show: the change is in how the integers are
formed, never in what is accepted. Measured under the same cost model: dot on 1,000 entries over one shared 10,000-bit
denominator about 1,120,000 units (the producer about 1,000,000), dot on 20,000 short entries about 450,000.

Limitation (in the safe direction). K carries every denominator of the coordinates in J. For vectors with many
distinct long denominators that cancel in the products a_i b_i (a_i = 1/q_i, b_i = q_i with long coprime q_i), K grows
with every new denominator, and a dot claim can run out of the budget (ResourceLimitError, unverified, never verified)
where a term-by-term sum stays cheaper: with 20,000-bit q_i, 8 entries take about 720,000 units and 12 about
1,870,000, where the producer takes about 210,000 and 480,000 and itself runs out near 20 entries, its common
denominator carrying every q_i as well. Projection on vectors whose entries all have distinct long denominators (60
entries of 3,000-bit rationals, b a multiple of a) needs about as much as the producer, which takes most of the default
budget there, and either may run out first.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer. The family defines no record class: every value is a mathforge.model.Rational or a tuple of
them, which the checker identifies by its exact class and reads as data (each field once; a Rational forged with
object.__new__ may lack one). tools/check_layering.py follows every import, and tests/v08/test_layering.py rejects
importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

from mathforge.limits import Budget, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

# The argument names of each operation, in the order of its spec.
ARGUMENTS: dict[str, tuple[str, ...]] = {
    "dot": ("a", "b"),
    "cross": ("a", "b"),
    "projection": ("a", "b"),
    "triple_product": ("a", "b", "c"),
}
# The length that every argument of an operation must have; None: any common length of at least 1.
DIMENSIONS: dict[str, int | None] = {"dot": None, "cross": 3, "projection": None, "triple_product": 3}
# The rule of Sarrus: det[X, Y, Z] = sum of sign * X_i * Y_j * Z_k over these (i, j, k, sign), the six permutations
# (i, j, k) of (0, 1, 2) with their signs (the Leibniz formula).
SARRUS = ((0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (2, 1, 0, -1), (1, 0, 2, -1), (0, 2, 1, -1))

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, malformed arguments, arguments outside the operation's domain, a malformed or
    non-canonical value and any evidence but (); ResourceLimitError for a request integer over max_integer_bits and
    for work past the budget."""
    if type(operation) is not str:
        return False
    names = ARGUMENTS.get(operation)
    if (names is None or type(args) is not dict or len(args) != len(names)
            or any(type(name) is not str for name in args) or any(name not in args for name in names)):
        return False
    vectors = [args[name] for name in names]
    if any(type(vector) is not tuple for vector in vectors) or type(evidence) is not tuple or evidence:
        return False
    length, dimension = len(vectors[0]), DIMENSIONS[operation]
    if length < 1 or any(len(vector) != length for vector in vectors) or dimension not in (None, length):
        return False
    with computation() as budget:  # One budget for every step: the active one, or one of the default limits.
        pairs = []
        for vector in vectors:
            read = _request_vector(budget, vector)
            if read is None:
                return False
            pairs.append(read)
        return _CHECKS[operation](budget, *pairs, value)


# ---------------------------------------------------------------------------------------------------------------
# Shapes

def _request_vector(budget: Budget, vector: tuple[object, ...]) -> list[Pair] | None:
    """The entries of a vector argument as pairs: each an exact int (over 1), or a Rational of exact ints with a
    positive denominator. None for anything else. An integer over max_integer_bits raises ResourceLimitError, as the
    request's inspection does. Lowest terms are checked later."""
    pairs = []
    for entry in vector:
        budget.tick()
        kind = type(entry)
        if kind is int:
            budget.check_int(entry)
            pairs.append((entry, 1))
            continue
        if kind is not Rational:
            return None
        numerator = getattr(entry, "numerator", None)  # A forged Rational may lack a slot.
        denominator = getattr(entry, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int:
            return None
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator < 1:
            return None
        pairs.append((numerator, denominator))
    return pairs


def _claimed(budget: Budget, number: object) -> Pair | None:
    """A Rational of the value as a pair: exact ints within max_integer_bits and a positive denominator. None for
    anything else, an int included: a value is a Rational, never an int. Lowest terms are checked later."""
    budget.tick()
    if type(number) is not Rational:
        return None
    numerator = getattr(number, "numerator", None)
    denominator = getattr(number, "denominator", None)
    limit = budget.limits.max_integer_bits
    if (type(numerator) is not int or type(denominator) is not int or denominator < 1
            or numerator.bit_length() > limit or denominator.bit_length() > limit):
        return None
    return numerator, denominator


def _claimed_vector(budget: Budget, value: object, length: int) -> list[Pair] | None:
    """The value of cross or projection as pairs: a tuple of exactly length Rationals, each as _claimed takes it;
    None otherwise. The length is tested before any entry is read."""
    if type(value) is not tuple or len(value) != length:
        return None
    pairs = []
    for number in value:
        pair = _claimed(budget, number)
        if pair is None:
            return None
        pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------------------------------------------
# Charged integer arithmetic

def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient (division_work charges the words of the
    dividend, right for a remainder after a product; here the quotient may be far shorter)."""
    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: one, a division of the longer by the shorter with a pass
    over the longer, then Lehmer's algorithm on the shorter, quadratic, one unit per 64 squared 64-bit words as
    trusted.upoly.charge_gcd charges it (16,384 at 65,536 bits; measured about 3 ms there)."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + _quotient_work(large, small) + (large >> 12) + (words * words >> 6)


def _linear(*numbers: int) -> int:
    """The units of a linear pass over these integers (a sum, a difference, a comparison): one per 256 64-bit words."""
    return sum(number.bit_length() for number in numbers) >> 14


def _lowest_terms(budget: Budget, pairs: list[Pair]) -> bool:
    """Whether every pair (with a positive denominator) is in lowest terms: 0 only as 0/1."""
    for numerator, denominator in pairs:
        if denominator == 1:
            budget.tick()
            continue
        budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
        if gcd(numerator, denominator) != 1:
            return False
    return True


def _common_denominator(budget: Budget, pairs: list[Pair]) -> int:
    """The least common multiple D of the denominators: D % d == 0 leaves D, else D * (d / gcd(d, D % d))."""
    common = 1
    for _, denominator in pairs:
        if denominator == 1:
            budget.tick()
            continue
        common_bits, bits = common.bit_length(), denominator.bit_length()
        budget.tick(1 + _quotient_work(common_bits, bits))
        remainder = common % denominator
        if remainder:
            budget.tick(_gcd_work(bits, remainder.bit_length()))
            divisor = gcd(denominator, remainder)  # gcd(d, D), at most d.
            divisor_bits = divisor.bit_length()
            factor_bits = bits - divisor_bits + 1
            budget.tick(1 + _quotient_work(bits, divisor_bits) + product_work(common_bits, factor_bits)
                        + ((common_bits + factor_bits) >> 14))
            common *= denominator // divisor
    return common


def _integers(budget: Budget, pairs: list[Pair]) -> tuple[list[int], int]:
    """(X, D) for a vector of the request: D the least common multiple of its denominators and X_i = n_i * (D / d_i),
    so that the vector is X / D."""
    common = _common_denominator(budget, pairs)
    common_bits = common.bit_length()
    integers = []
    for numerator, denominator in pairs:
        bits, numerator_bits = denominator.bit_length(), numerator.bit_length()
        budget.tick(1 + _quotient_work(common_bits, bits) + product_work(numerator_bits, common_bits - bits + 1)
                    + ((numerator_bits + common_bits) >> 14))
        integers.append(numerator * (common // denominator))
    return integers, common


def _scaled(budget: Budget, pairs: list[Pair], common: int) -> list[int] | None:
    """X_i = n_i * (common / d_i) for a claimed vector, the vector times common, when every denominator divides
    common; None otherwise. A denominator longer than common is refused before any division."""
    common_bits = common.bit_length()
    integers = []
    for numerator, denominator in pairs:
        budget.tick()
        bits = denominator.bit_length()
        if bits > common_bits:
            return None
        budget.tick(_quotient_work(common_bits, bits))
        quotient, remainder = divmod(common, denominator)
        if remainder:
            return None
        numerator_bits, quotient_bits = numerator.bit_length(), quotient.bit_length()
        budget.tick(1 + product_work(numerator_bits, quotient_bits) + ((numerator_bits + quotient_bits) >> 14))
        integers.append(numerator * quotient)
    return integers


def _product(budget: Budget, *factors: int) -> int:
    """The product of the factors, from the left, each multiplication charged before it runs."""
    result = 1
    for factor in factors:
        result_bits, factor_bits = result.bit_length(), factor.bit_length()
        budget.tick(1 + product_work(result_bits, factor_bits) + ((result_bits + factor_bits) >> 14))
        result *= factor
    return result


def _difference(budget: Budget, left: int, right: int) -> int:
    """left - right, charged before it runs."""
    budget.tick(1 + _linear(left, right))
    return left - right


def _equal(budget: Budget, left: int, right: int) -> bool:
    """Whether left == right, the comparison charged before it runs."""
    budget.tick(1 + _linear(left, right))
    return left == right


def _dot(budget: Budget, xs: list[int], ys: list[int]) -> int:
    """X.Y for integer vectors of the same length, each product and each sum charged before it runs."""
    total = 0
    for x, y in zip(xs, ys):
        x_bits, y_bits = x.bit_length(), y.bit_length()
        budget.tick(1 + product_work(x_bits, y_bits) + ((x_bits + y_bits + total.bit_length()) >> 14))
        total += x * y
    return total


class _Squares:
    """The squares (K / t)^2 of the quotients of one positive integer K by its divisors t, and K^2 itself, formed once.

    The square of a long quotient is the costly step of a sum of squares in an integer form: K x for a rational vector
    x has entries n_i (K / d_i) as long as K, and squaring each of them costs a product quadratic in K. (K / t)^2 is
    the same integer as K^2 / t^2, an exact division whose cost is linear in K for a short t, so each square is
    formed by whichever of the two routes the cost model charges less, and both give the same integer."""

    def __init__(self, common: int) -> None:
        self.common = common
        self.bits = common.bit_length()
        self._square: int | None = None
        self._last_divisor, self._last = 0, 0  # No divisor is 0.

    def square(self, budget: Budget) -> int:
        """K^2, charged and formed on the first call only."""
        if self._square is None:
            bits = self.bits
            budget.tick(1 + product_work(bits, bits) + (bits >> 13))
            self._square = self.common * self.common
        return self._square

    def cost(self, divisor_bits: int) -> tuple[int, bool]:
        """(units, direct) of (K / t)^2 for a divisor t of K of that length: direct for the square of K / t, else
        K^2 / t^2. K^2 is left out of the comparison: it is formed once for all the squares (and for dot's final
        equation)."""
        bits = self.bits
        quotient_bits = bits - divisor_bits + 1
        direct = 2 + _quotient_work(bits, divisor_bits) + product_work(quotient_bits, quotient_bits) + (
            quotient_bits >> 13)
        divided = 2 + product_work(divisor_bits, divisor_bits) + _quotient_work(2 * bits, 2 * divisor_bits - 1)
        return (direct, True) if direct <= divided else (divided, False)

    def of(self, budget: Budget, divisor: int) -> int:
        """(K / t)^2 for a divisor t of K. The last square is kept: a coordinate asks for (K / t)^2, (K / d_i)^2 and
        (K / e_i)^2 in that order, and t is d_i or e_i (or both) whenever one denominator divides the other."""
        if divisor == 1:
            budget.tick()
            return self.square(budget)
        bits = divisor.bit_length()
        budget.tick(1 + (bits >> 13))
        if divisor == self._last_divisor:
            return self._last
        units, direct = self.cost(bits)
        if direct:
            budget.tick(units)
            quotient = self.common // divisor
            result = quotient * quotient
        else:
            whole = self.square(budget)
            budget.tick(units)
            result = whole // (divisor * divisor)
        self._last_divisor, self._last = divisor, result
        return result


def _add_square(budget: Budget, total: int, number: int, scale: int) -> int:
    """total + number^2 scale, each product and the sum charged before it runs."""
    bits = number.bit_length()
    budget.tick(1 + product_work(bits, bits) + (bits >> 13))
    square = number * number
    square_bits, scale_bits = square.bit_length(), scale.bit_length()
    budget.tick(2 + product_work(square_bits, scale_bits) + ((square_bits + scale_bits + total.bit_length()) >> 14))
    return total + square * scale


def _polarization(budget: Budget, a: list[Pair], b: list[Pair], whole_b: bool) -> tuple[int, int, _Squares]:
    """(|K a + K b|^2 - |K a|^2 - |K b|^2, |K b|^2, K) over a set J of coordinates: those where a_i and b_i are both
    nonzero, or with whole_b those where b_i is nonzero; K is the least common multiple of the denominators of a_i
    and b_i for i in J (1 when J is empty).

    J holds every coordinate where a_i b_i != 0, and a coordinate where a_i or b_i is 0 adds (x + 0)^2 - x^2 - 0^2 = 0
    to the first sum, so the polarization identity makes the first result 2 (K a).(K b) = 2 K^2 a.b, a product that is
    never formed here. With whole_b, J holds every coordinate where b_i != 0, so the second result is K^2 |b|^2. For i
    in J, K a_i + K b_i = s (K / t) with a_i + b_i = s / t for any common multiple t of d_i and e_i that divides K, so
    that each of the three terms of a coordinate is a square s^2 times one of the squares (K / t)^2, (K / d_i)^2 and
    (K / e_i)^2 (_Squares). t is lcm(d_i, e_i), which keeps s short, or K itself, where s = n (K / d_i) + m (K / e_i)
    needs no gcd: for denominators as long as K (one shared denominator) the gcd of lcm(d_i, e_i) costs more than the
    square of the longer s, and the cost model picks the cheaper of the two for each coordinate."""
    joint = []
    for x, y in zip(a, b):
        budget.tick()
        if y[0] and (x[0] or whole_b):
            joint.append((x, y))
    squares = _Squares(_common_denominator(budget, [x for x, _ in joint] + [y for _, y in joint]))
    both = left = right = 0
    for (n, d), (m, e) in joint:
        if not n:  # a_i = 0 (whole_b only): K a_i + K b_i = K b_i, the same term in the first and the third sum.
            square = _add_square(budget, 0, m, squares.of(budget, e))
            budget.tick(2 + ((2 * square.bit_length() + both.bit_length() + right.bit_length()) >> 14))
            both, right = both + square, right + square
            continue
        # a_i + b_i = s / t with t = lcm(d, e): s = n (t / d) + m (t / e).
        n_bits, m_bits, d_bits, e_bits = n.bit_length(), m.bit_length(), d.bit_length(), e.bit_length()
        budget.tick(1 + ((d_bits + e_bits) >> 14))
        if d == e:
            budget.tick(1 + ((n_bits + m_bits) >> 14))
            s, t = n + m, d
        elif d == 1:
            budget.tick(1 + product_work(n_bits, e_bits) + ((n_bits + e_bits + m_bits) >> 14))
            s, t = n * e + m, e
        elif e == 1:
            budget.tick(1 + product_work(m_bits, d_bits) + ((m_bits + d_bits + n_bits) >> 14))
            s, t = n + m * d, d
        else:
            # The estimates choose a route; each step is charged by the lengths it really has when it runs.
            k_bits = squares.bits
            d_quotient_bits, e_quotient_bits = k_bits - d_bits + 1, k_bits - e_bits + 1
            long_bits = max(n_bits + d_quotient_bits, m_bits + e_quotient_bits) + 1
            through_common = (_quotient_work(k_bits, d_bits) + _quotient_work(k_bits, e_bits)
                              + product_work(n_bits, d_quotient_bits) + product_work(m_bits, e_quotient_bits)
                              + product_work(long_bits, long_bits))
            short_bits = max(n_bits + e_bits, m_bits + d_bits) + 1
            through_lcm = (_gcd_work(d_bits, e_bits) + product_work(n_bits, e_bits) + product_work(m_bits, d_bits)
                           + product_work(d_bits, e_bits) + product_work(short_bits, short_bits)
                           + squares.cost(min(d_bits + e_bits, k_bits))[0])
            if through_common <= through_lcm:
                budget.tick(2 + _quotient_work(k_bits, d_bits) + _quotient_work(k_bits, e_bits))
                d_scale, e_scale = squares.common // d, squares.common // e
                d_scale_bits, e_scale_bits = d_scale.bit_length(), e_scale.bit_length()
                budget.tick(3 + product_work(n_bits, d_scale_bits) + product_work(m_bits, e_scale_bits)
                            + ((2 * (n_bits + m_bits + d_scale_bits + e_scale_bits)) >> 14))
                s, t = n * d_scale + m * e_scale, squares.common
            else:
                budget.tick(_gcd_work(d_bits, e_bits))
                divisor = gcd(d, e)
                divisor_bits = divisor.bit_length()
                budget.tick(2 + _quotient_work(d_bits, divisor_bits) + _quotient_work(e_bits, divisor_bits))
                d_part, e_part = d // divisor, e // divisor
                d_part_bits, e_part_bits = d_part.bit_length(), e_part.bit_length()
                budget.tick(3 + product_work(n_bits, e_part_bits) + product_work(m_bits, d_part_bits)
                            + product_work(d_bits, e_part_bits)
                            + ((2 * (n_bits + m_bits + d_bits + e_part_bits + d_part_bits)) >> 14))
                s, t = n * e_part + m * d_part, d * e_part
        both = _add_square(budget, both, s, squares.of(budget, t))
        left = _add_square(budget, left, n, squares.of(budget, d))
        right = _add_square(budget, right, m, squares.of(budget, e))
    return _difference(budget, _difference(budget, both, left), right), right, squares


def _quotient_pair(budget: Budget, numerator: int, denominator: int, divisor: int) -> Pair:
    """(numerator / divisor, denominator / divisor) for a common divisor, the two exact divisions charged first."""
    if divisor == 1:
        budget.tick()
        return numerator, denominator
    bits = divisor.bit_length()
    budget.tick(2 + _quotient_work(numerator.bit_length(), bits) + _quotient_work(denominator.bit_length(), bits))
    return numerator // divisor, denominator // divisor


def _reduced(budget: Budget, numerator: int, denominator: int) -> Pair:
    """numerator / denominator (denominator != 0) in lowest terms with a positive denominator."""
    budget.tick(1 + _linear(numerator, denominator))
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
    return _quotient_pair(budget, numerator, denominator, gcd(numerator, denominator))


def _scaled_by(budget: Budget, u: int, w: int, m: int, e: int) -> Pair:
    """The canonical pair of (u / w)(m / e) for u / w and m / e in lowest terms with w, e >= 1: 0/1 when u or m is 0,
    else (u / g)(m / h) over (w / h)(e / g) with g = gcd(u, e) and h = gcd(m, w). That pair is in lowest terms: u / g
    is prime to e / g (g is the whole gcd) and to w / h (u is prime to w), and m / h is prime to w / h and to e / g
    likewise. Every product here is of one factor of t = u / w and one of an entry of b, so that for a long t and a
    short b_j the work is linear in the length of t (the cubic cost of cross multiplying two long factors per entry,
    module docstring)."""
    if not u or not m:
        budget.tick()
        return 0, 1
    budget.tick(_gcd_work(u.bit_length(), e.bit_length()) + _gcd_work(m.bit_length(), w.bit_length()))
    g, h = gcd(u, e), gcd(m, w)
    u, e = _quotient_pair(budget, u, e, g)
    m, w = _quotient_pair(budget, m, w, h)
    return _product(budget, u, m), _product(budget, w, e)


def _determinant(budget: Budget, xs: list[int], ys: list[int], zs: list[int]) -> int:
    """det[X, Y, Z] of the 3 x 3 integer matrix with the rows X, Y and Z, by the rule of Sarrus (SARRUS)."""
    total = 0
    for i, j, k, sign in SARRUS:
        term = _product(budget, xs[i], ys[j], zs[k])
        budget.tick(1 + _linear(term, total))
        total = total + term if sign > 0 else total - term
    return total


# ---------------------------------------------------------------------------------------------------------------
# The operations, each on the request's vectors as pairs (of the operation's lengths) and the claimed value

def _dot_check(budget: Budget, a: list[Pair], b: list[Pair], value: object) -> bool:
    """q (|K a + K b|^2 - |K a|^2 - |K b|^2) = 2 p K^2 for the canonical v = p/q (module docstring)."""
    claimed = _claimed(budget, value)
    if claimed is None or not _lowest_terms(budget, [claimed]):
        return False
    if not (_lowest_terms(budget, a) and _lowest_terms(budget, b)):
        return False
    twice, _, squares = _polarization(budget, a, b, False)
    p, q = claimed
    return _equal(budget, _product(budget, q, twice), _product(budget, 2, p, squares.square(budget)))


def _cross_check(budget: Budget, a: list[Pair], b: list[Pair], value: object) -> bool:
    """C = D_a D_b c integral, C.A = C.B = 0, |C|^2 = |A|^2 |B|^2 - (A.B)^2 and det[A, B, C] = |C|^2 (module
    docstring)."""
    claimed = _claimed_vector(budget, value, 3)
    if claimed is None or not (_lowest_terms(budget, a) and _lowest_terms(budget, b)):
        return False
    xs, a_denominator = _integers(budget, a)
    ys, b_denominator = _integers(budget, b)
    zs = _scaled(budget, claimed, _product(budget, a_denominator, b_denominator))
    if zs is None or not _lowest_terms(budget, claimed):
        return False
    if _dot(budget, zs, xs) or _dot(budget, zs, ys):  # C.A = 0 and C.B = 0.
        return False
    norm = _dot(budget, zs, zs)
    mixed = _dot(budget, xs, ys)
    lagrange = _difference(budget, _product(budget, _dot(budget, xs, xs), _dot(budget, ys, ys)),
                           _product(budget, mixed, mixed))
    return _equal(budget, norm, lagrange) and _equal(budget, _determinant(budget, xs, ys, zs), norm)


def _projection_check(budget: Budget, a: list[Pair], b: list[Pair], value: object) -> bool:
    """b != 0, every p_j is the canonical pair of t b_j for t = u / w = p_k / b_k in lowest terms, and w P = 2 u R
    (module docstring)."""
    claimed = _claimed_vector(budget, value, len(b))
    if claimed is None or not (_lowest_terms(budget, a) and _lowest_terms(budget, b)):
        return False
    pivot = None
    for index, (numerator, _) in enumerate(b):
        budget.tick()
        if numerator:
            pivot = index
            break
    if pivot is None:  # b = 0 has no projection.
        return False
    (r_k, s_k), (m_k, e_k) = claimed[pivot], b[pivot]
    u, w = _reduced(budget, _product(budget, r_k, e_k), _product(budget, s_k, m_k))  # t = p_k / b_k = u / w.
    for (r, s), (m, e) in zip(claimed, b):  # p = t b, each p_j canonical: the pair of t b_j, entry by entry.
        numerator, denominator = _scaled_by(budget, u, w, m, e)
        if not (_equal(budget, r, numerator) and _equal(budget, s, denominator)):
            return False
    twice, norm, _ = _polarization(budget, a, b, True)  # P = 2 K^2 a.b and R = K^2 |b|^2 > 0.
    return _equal(budget, _product(budget, w, twice), _product(budget, 2, u, norm))


def _triple_product_check(budget: Budget, a: list[Pair], b: list[Pair], c: list[Pair], value: object) -> bool:
    """q det[A, B, C] = p D_a D_b D_c for the canonical v = p/q (module docstring)."""
    claimed = _claimed(budget, value)
    if claimed is None or not _lowest_terms(budget, [claimed]):
        return False
    if not (_lowest_terms(budget, a) and _lowest_terms(budget, b) and _lowest_terms(budget, c)):
        return False
    xs, a_denominator = _integers(budget, a)
    ys, b_denominator = _integers(budget, b)
    zs, c_denominator = _integers(budget, c)
    p, q = claimed
    return _equal(budget, _product(budget, q, _determinant(budget, xs, ys, zs)),
                  _product(budget, p, a_denominator, b_denominator, c_denominator))


_CHECKS: dict[str, Callable[..., bool]] = {
    "dot": _dot_check,
    "cross": _cross_check,
    "projection": _projection_check,
    "triple_product": _triple_product_check,
}
