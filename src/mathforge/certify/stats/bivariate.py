"""Independent checker of the family mathforge.areas.stats.bivariate: correlation and linear_regression.

The checker never reruns the producer's raw moments. It scales the data to integers and verifies each claim by exact
integer identities: the correlation coefficient from centred sums, a fitted line by the normal equations that its
residuals satisfy.

The data, for both operations. The request holds xs and ys, tuples (a request holds lists frozen into tuples) of the
same length n >= 2, whose items are exact rationals: an int, or a Rational of two exact ints in lowest terms with a
positive denominator, as the Rational constructor makes it. With D and E the least common multiples of the
denominators of the xs and of the ys, X_i = D x_i and Y_i = E y_i are integers: the data scaled by positive factors.

The theorem that makes True imply the claim, for each operation:

correlation(xs, ys) = r, evidence ().
    With the centred integer deviations U_i = n X_i - sum_j X_j = n D (x_i - mean(xs)) and
    V_i = n Y_i - sum_j Y_j = n E (y_i - mean(ys)), the checker forms A = sum U_i^2 = (n D)^2 S_xx,
    B = sum V_i^2 = (n E)^2 S_yy and C = sum U_i V_i = n^2 D E S_xy, where S_xx, S_yy and S_xy are the centred sums of
    the data (the producer computes them from raw moments). A = 0 or B = 0 is a constant variable, which has no
    correlation coefficient: every claim answers False. Otherwise the Pearson coefficient
    r = S_xy / sqrt(S_xx S_yy) = C / sqrt(A B), the positive factors cancelling, is sign(C) sqrt(q) with
    q = C^2 / (A B) = S_xy^2 / (S_xx S_yy). The checker accepts exactly the canonical form that
    mathforge.model.mul(sign, mathforge.model.sqrt(q)) builds:
    - a Rational v = p/s with sign(p) = sign(C) and p^2 A B = s^2 C^2, that is v^2 = q with the sign of r: v = r.
      Some Rational passes exactly when q is the square of a rational (0 included, for C = 0);
    - Sqrt(t) when C > 0, or Mul((Rational(-1), Sqrt(t))) when C < 0, those two factors in this order, with t = a/b
      such that a A B = b C^2, that is t = q (b > 0 and A B > 0), and a or b no perfect square. As t is in lowest
      terms, that says that q is the square of no rational, so that no Rational is r and the radical is the
      canonical form; the expression denotes sign(C) sqrt(q) = r.
    Every number of the value is a canonical Rational (exact ints in lowest terms, a positive denominator), and every
    node has exactly the class Rational, Sqrt or Mul of mathforge.model. So each input has exactly one accepted value,
    r in its canonical form: an int, a factor taken out of the radical (5*sqrt(3/76) for sqrt(75/76)), another order
    of the factors and any other node answer False.

linear_regression(xs, ys) = LinearFit(slope, intercept), evidence ().
    The xs are not all equal (some X_i differs from X_1), else every claim answers False: the least-squares line is not
    unique. The value is a record of the class LinearFit of mathforge.areas.stats.bivariate, read as data, whose slope
    alpha/sigma and intercept beta/tau are canonical Rationals. Its residuals r_i = y_i - (slope x_i + intercept)
    satisfy sum r_i = 0 and sum x_i r_i = 0, the normal equations, which the checker evaluates exactly by linearity
    from the sums of the scaled data, multiplied by the positive integers E sigma tau D and D^2 E sigma tau:
        N1 = sigma tau D sum Y_i - E tau alpha sum X_i - n E sigma D beta = 0,
        N2 = sigma tau D sum X_i Y_i - E tau alpha sum X_i^2 - E sigma D beta sum X_i = 0
    (the claimed slope and intercept so enter a fixed number of products, however many points there are). They say
    that the gradient -2 (sum x_i r_i, sum r_i) of f(a, b) = sum (y_i - a x_i - b)^2 vanishes at the claim. Their
    matrix [[sum x_i^2, sum x_i], [sum x_i, n]], the Gram matrix of the vectors (x_i) and (1, ..., 1), has the
    determinant n sum x_i^2 - (sum x_i)^2 = n S_xx > 0 (Cauchy-Schwarz, strict as the xs are not all equal): it is
    positive definite, so f is strictly convex and the unique solution of the normal equations is its unique
    minimizer. The claim is the least-squares line, in its one canonical record.

Shapes and budget. The operation must be the string "correlation" or "linear_regression", args a dict of exactly xs
and ys, and the evidence the empty tuple; a bool, a float, a str, an int subclass or a Rational subclass is no exact
number, a list no tuple, and anything else answers False, never an exception. A value integer over max_integer_bits
answers False, as no computation under the budget forms it. A request integer over max_integer_bits raises
ResourceLimitError, as operations_v6.run and the verifier's inspection raise before a checker runs, and so does a
common denominator D or E over max_integer_bits (the producer's sums of the data carry that denominator; only data
built to cancel could keep them shorter), and work past max_work: a claim is then unverified, never verified. Every
loop is charged before it runs, each long integer operation on the scale of the cost model of trusted.integers
(product_work for a product, division_work for the length of a quotient, a gcd quadratic in its shorter operand, an
integer square root like a product of its length, one unit per 256 words of a linear pass), and no builtin pow is
used. The claimed value enters a fixed number of products: a forged value of any size is refused at once.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from math import gcd, isqrt

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Mul, Rational, Sqrt
from mathforge.trusted.integers import division_work, product_work

ARGUMENTS = frozenset(("xs", "ys"))  # The argument names of both specs.
LINEAR_FIT = ("mathforge.areas.stats.bivariate", "LinearFit")  # Module and name of linear_regression's record.

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1.
# A correlation claim read from its canonical form: ("rational", p, s) for p/s, or ("radical", sign, a, b) for
# sign * sqrt(a/b) with sign 1 (a Sqrt) or -1 (a Mul of Rational(-1) and a Sqrt).
Claim = tuple[object, ...]


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, malformed arguments, a malformed or non-canonical value and any evidence but ();
    ResourceLimitError for a request integer or a common denominator over max_integer_bits and for work past the
    budget."""
    if type(operation) is not str:
        return False
    check = _CHECKS.get(operation)
    if check is None:
        return False
    if (type(args) is not dict or len(args) != len(ARGUMENTS) or any(type(name) is not str for name in args)
            or not ARGUMENTS.issuperset(args)):
        return False
    xs, ys = args["xs"], args["ys"]
    if type(xs) is not tuple or type(ys) is not tuple or len(xs) < 2 or len(xs) != len(ys):
        return False
    if type(evidence) is not tuple or evidence:
        return False
    with computation() as budget:  # One budget for every step: the active one, or one of the default limits.
        return check(budget, xs, ys, value)


# ---------------------------------------------------------------------------------------------------------------
# The two operations

def _correlation(budget: Budget, xs: tuple[object, ...], ys: tuple[object, ...], value: object) -> bool:
    """The correlation theorem of the module: the centred sums A, B and C, then the claim against them."""
    claim = _correlation_claim(budget, value)
    if claim is None:
        return False
    data = _data(budget, xs, ys)
    if data is None:
        return False
    (x_pairs, x_common), (y_pairs, y_common) = data
    count = len(x_pairs)
    count_bits = count.bit_length()
    total_x = _total(budget, _scaled(budget, x_pairs, x_common))
    total_y = _total(budget, _scaled(budget, y_pairs, y_common))
    a = b = c = 0  # A, B and C of the module docstring.
    for x, y in zip(_scaled(budget, x_pairs, x_common), _scaled(budget, y_pairs, y_common)):
        x_bits, y_bits = x.bit_length(), y.bit_length()
        budget.tick(2 + product_work(count_bits, x_bits) + product_work(count_bits, y_bits)
                    + _linear(2 * (x_bits + y_bits + count_bits) + total_x.bit_length() + total_y.bit_length()))
        u = count * x - total_x  # U_i = n D (x_i - mean(xs)).
        v = count * y - total_y
        u_bits, v_bits = u.bit_length(), v.bit_length()
        budget.tick(3 + product_work(u_bits, u_bits) + product_work(v_bits, v_bits) + product_work(u_bits, v_bits)
                    + _linear(4 * (u_bits + v_bits) + a.bit_length() + b.bit_length() + c.bit_length()))
        a += u * u
        b += v * v
        c += u * v
    if not a or not b:  # A constant variable: no correlation coefficient exists.
        return False
    spread = _product(budget, a, b)  # A B > 0.
    if claim[0] == "rational":
        _, numerator, denominator = claim
        if (numerator > 0) != (c > 0) or (numerator < 0) != (c < 0):
            return False
        left = _product(budget, _product(budget, numerator, numerator), spread)
        right = _product(budget, _product(budget, denominator, denominator), _product(budget, c, c))
        return left == right  # v^2 = C^2 / (A B) = q.
    _, sign, radicand_numerator, radicand_denominator = claim
    if c == 0 or (c > 0) != (sign > 0):
        return False
    left = _product(budget, radicand_numerator, spread)
    right = _product(budget, radicand_denominator, _product(budget, c, c))
    if left != right:  # t = q.
        return False
    # t in lowest terms is the square of a rational exactly when its numerator and denominator are perfect squares.
    return not (_is_square(budget, radicand_numerator) and _is_square(budget, radicand_denominator))


def _linear_regression(budget: Budget, xs: tuple[object, ...], ys: tuple[object, ...], value: object) -> bool:
    """The regression theorem of the module: xs that vary, and the normal equations N1 = N2 = 0 at the claim."""
    if not _named(value, LINEAR_FIT):
        return False
    slope = _canonical_rational(budget, getattr(value, "slope", None))  # A forged record may lack a field.
    intercept = _canonical_rational(budget, getattr(value, "intercept", None))
    if slope is None or intercept is None:
        return False
    data = _data(budget, xs, ys)
    if data is None:
        return False
    (x_pairs, x_common), (y_pairs, y_common) = data
    count = len(x_pairs)
    first: int | None = None
    varies = False
    sum_x = sum_y = sum_xx = sum_xy = 0
    for x, y in zip(_scaled(budget, x_pairs, x_common), _scaled(budget, y_pairs, y_common)):
        x_bits, y_bits = x.bit_length(), y.bit_length()
        budget.tick(5 + product_work(x_bits, x_bits) + product_work(x_bits, y_bits)
                    + _linear(3 * (x_bits + y_bits) + sum_x.bit_length() + sum_y.bit_length() + sum_xx.bit_length()
                              + sum_xy.bit_length()))
        if first is None:
            first = x
        elif x != first:
            varies = True
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_xy += x * y
    if not varies:  # All xs equal: S_xx = 0, and the least-squares line is not unique.
        return False
    alpha, sigma = slope
    beta, tau = intercept
    # N1 = P sum Y - Q sum X - n R and N2 = P sum XY - Q sum XX - R sum X, with P = sigma tau D, Q = E tau alpha and
    # R = E sigma D beta.
    p = _product(budget, _product(budget, sigma, tau), x_common)
    q = _product(budget, _product(budget, y_common, tau), alpha)
    r = _product(budget, _product(budget, _product(budget, y_common, sigma), x_common), beta)
    if _difference(budget, _product(budget, p, sum_y), _product(budget, q, sum_x), _product(budget, count, r)):
        return False
    return not _difference(budget, _product(budget, p, sum_xy), _product(budget, q, sum_xx),
                           _product(budget, r, sum_x))


_CHECKS: dict[str, Callable[[Budget, tuple[object, ...], tuple[object, ...], object], bool]] = {
    "correlation": _correlation,
    "linear_regression": _linear_regression,
}


# ---------------------------------------------------------------------------------------------------------------
# Shapes of the value

def _named(value: object, name: tuple[str, str]) -> bool:
    """Whether the value's class has this module and name, both exact strings (a class may set __module__ to any
    object, whose comparison could answer anything)."""
    cls = type(value)
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


def _canonical_rational(budget: Budget, number: object) -> Pair | None:
    """(p, s) of a Rational of exact ints in lowest terms with s >= 1, both within max_integer_bits; None for
    anything else: an int, a Rational subclass, and a Rational made with object.__new__ that lacks a slot or breaks
    the form."""
    if type(number) is not Rational:
        return None
    numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        return None
    limit = budget.limits.max_integer_bits
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    if numerator_bits > limit or denominator_bits > limit:
        return None  # No computation under the budget forms it.
    budget.tick()
    if denominator > 1:
        budget.tick(_gcd_work(numerator_bits, denominator_bits))
        if gcd(numerator, denominator) != 1:
            return None
    return numerator, denominator


def _correlation_claim(budget: Budget, value: object) -> Claim | None:
    """The claim of a correlation value in one of the canonical shapes of the module docstring: a Rational, a Sqrt of
    a Rational, or a Mul of exactly Rational(-1) and such a Sqrt; None for any other value."""
    kind = type(value)
    if kind is Rational:
        pair = _canonical_rational(budget, value)
        return None if pair is None else ("rational", *pair)
    sign = 1
    if kind is Mul:
        factors = getattr(value, "factors", None)  # A forged node may lack it or hold a list.
        if type(factors) is not tuple or len(factors) != 2 or _canonical_rational(budget, factors[0]) != (-1, 1):
            return None
        sign, value = -1, factors[1]
    if type(value) is not Sqrt:
        return None
    radicand = _canonical_rational(budget, getattr(value, "radicand", None))
    return None if radicand is None else ("radical", sign, *radicand)


# ---------------------------------------------------------------------------------------------------------------
# The data

def _data(budget: Budget, xs: tuple[object, ...], ys: tuple[object, ...]) -> tuple[tuple[list[Pair], int],
                                                                                    tuple[list[Pair], int]] | None:
    """((pairs, D), (pairs, E)) of xs and ys, with the common denominators D and E; None when an item is no exact
    rational in canonical form. Linear steps first (the types and lengths of every item), then the gcds."""
    x_pairs = _pairs(budget, xs)
    if x_pairs is None:
        return None
    y_pairs = _pairs(budget, ys)
    if y_pairs is None or not _lowest_terms(budget, x_pairs) or not _lowest_terms(budget, y_pairs):
        return None
    return (x_pairs, _common_denominator(budget, x_pairs)), (y_pairs, _common_denominator(budget, y_pairs))


def _pairs(budget: Budget, numbers: tuple[object, ...]) -> list[Pair] | None:
    """The items of xs or ys as pairs: each an int, or a Rational of exact ints with a positive denominator; None for
    anything else. An integer over max_integer_bits raises ResourceLimitError, as the request's inspection does."""
    pairs = []
    for number in numbers:
        budget.tick()
        kind = type(number)
        if kind is int:
            budget.check_int(number)
            pairs.append((number, 1))
            continue
        if kind is not Rational:
            return None
        numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int:
            return None
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator < 1:
            return None
        pairs.append((numerator, denominator))
    return pairs


def _lowest_terms(budget: Budget, pairs: list[Pair]) -> bool:
    """Whether every pair (with a positive denominator) is in lowest terms: 0 only as 0/1."""
    for numerator, denominator in pairs:
        budget.tick()
        if denominator > 1:
            budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
            if gcd(numerator, denominator) != 1:
                return False
    return True


def _common_denominator(budget: Budget, pairs: list[Pair]) -> int:
    """The least common multiple of the denominators: D % d == 0 leaves D, else D * (d / gcd(d, D % d)).
    ResourceLimitError once it is longer than max_integer_bits."""
    limit = budget.limits.max_integer_bits
    common = 1
    for _, denominator in pairs:
        budget.tick()
        if denominator == 1:
            continue
        common_bits, bits = common.bit_length(), denominator.bit_length()
        budget.tick(_quotient_work(common_bits, bits))
        remainder = common % denominator
        if remainder:
            budget.tick(_gcd_work(bits, remainder.bit_length()))
            divisor = gcd(denominator, remainder)  # gcd(d, D), at most d.
            factor_bits = bits - divisor.bit_length() + 1
            budget.tick(_quotient_work(bits, divisor.bit_length()) + product_work(common_bits, factor_bits)
                        + _linear(common_bits + factor_bits))
            common *= denominator // divisor
            if common.bit_length() > limit:
                raise ResourceLimitError("Computation limit exceeded: common denominator bits")
    return common


def _scaled(budget: Budget, pairs: list[Pair], common: int) -> Iterator[int]:
    """The integers numerator * (common / denominator), one at a time and charged before each is formed, so that no
    list of long integers is kept."""
    common_bits = common.bit_length()
    for numerator, denominator in pairs:
        if denominator == common:
            budget.tick()
            yield numerator
            continue
        bits, numerator_bits = denominator.bit_length(), numerator.bit_length()
        budget.tick(1 + _quotient_work(common_bits, bits) + product_work(numerator_bits, common_bits - bits + 1)
                    + _linear(common_bits + numerator_bits))
        yield numerator * (common // denominator)


def _total(budget: Budget, numbers: Iterator[int]) -> int:
    """The sum of the integers, each addition charged by the length of its operands."""
    total = 0
    for number in numbers:
        budget.tick(1 + _linear(number.bit_length() + total.bit_length()))
        total += number
    return total


# ---------------------------------------------------------------------------------------------------------------
# Charged integer steps

def _linear(bits: int) -> int:
    """The units of a linear pass over integers of this many bits in all: one per 256 64-bit words."""
    return bits >> 14


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient (division_work charges the words of the
    dividend, right for a remainder after a product; here the quotient may be far shorter)."""
    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: one, a division of the longer by the shorter with a pass
    over the longer, then Lehmer's algorithm on the shorter, quadratic, one unit per 64 squared 64-bit words as
    trusted.upoly.charge_gcd charges it (16,384 at 65,536 bits)."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + _quotient_work(large, small) + (large >> 12) + (words * words >> 6)


def _product(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs."""
    left_bits, right_bits = left.bit_length(), right.bit_length()
    budget.tick(1 + product_work(left_bits, right_bits) + _linear(left_bits + right_bits))
    return left * right


def _difference(budget: Budget, first: int, second: int, third: int) -> int:
    """first - second - third, charged by the length of its operands."""
    budget.tick(1 + _linear(first.bit_length() + second.bit_length() + third.bit_length()))
    return first - second - third


def _is_square(budget: Budget, number: int) -> bool:
    """Whether the integer number >= 0 is a perfect square; math.isqrt costs about one product of its length
    (measured), charged twice."""
    bits = number.bit_length()
    budget.tick(1 + 2 * product_work(bits, bits) + _linear(bits))
    root = isqrt(number)
    budget.tick(1 + product_work(root.bit_length(), root.bit_length()) + _linear(bits))
    return root * root == number
