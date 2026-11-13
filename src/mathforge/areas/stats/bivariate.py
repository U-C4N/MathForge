"""Bivariate statistics of paired exact rational data (mathforge.stats): the Pearson correlation coefficient and the
least-squares line.

Both operations take xs and ys: exact rationals, ints or Rationals (MCP kind "rational_list": operations_v6.run
freezes a list into a tuple and already rejects a bool, a float, a str or None in it as invalid input), with the same
length n >= 2. The arguments are unannotated, as v0.7's rational_list arguments are (mean(data), covariance(x, y)): a
list is accepted as well as a tuple. With the means mx and my and the centred sums S_xx = sum (x_i - mx)^2,
S_yy = sum (y_i - my)^2 and S_xy = sum (x_i - mx)(y_i - my) (docs/v09/cards/m4-linalg-geometry-stats-discrete.md):

correlation(xs, ys) -> Expression
    Domain: S_xx > 0 and S_yy > 0. Lists of different lengths, fewer than 2 pairs and a constant variable (zero
    variance) are invalid input.
    Value: r = S_xy / sqrt(S_xx S_yy) = sign(S_xy) sqrt(q) with q = S_xy^2 / (S_xx S_yy), in the canonical form that
    mathforge.model.mul(sign(S_xy), mathforge.model.sqrt(q)) builds, as v0.7's stddev builds its radicals: the
    Rational r when q is the square of a rational (0 when S_xy = 0; 1 and -1 for exactly linear data; 4/5), else
    Sqrt(q) when S_xy > 0 and Mul((Rational(-1), Sqrt(q))) when S_xy < 0, with q the reduced Rational itself: no
    factor leaves the radical (sqrt(75/76), never 5*sqrt(3/76)). Never an int: 1 is Rational(1).
    Evidence: ().

linear_regression(xs, ys) -> LinearFit
    Domain: the xs are not all equal (S_xx != 0). Lists of different lengths, fewer than 2 pairs and xs that are all
    equal are invalid input.
    Value: LinearFit(slope, intercept), slope = S_xy / S_xx and intercept = my - slope * mx, the unique minimizer of
    sum (y_i - slope x_i - intercept)^2, as two Rationals (an integral one included, never an int).
    Evidence: ().

The checker, mathforge.certify.stats.bivariate, recomputes q and the sign of r from centred integer sums (the producer
uses raw moments) and requires the canonical form; for a fitted line it verifies that the residuals
r_i = y_i - (slope x_i + intercept) satisfy the normal equations sum r_i = 0 and sum x_i r_i = 0, whose solution is
unique when S_xx != 0.

How produce() computes. Nothing here is a proof: the checker verifies every value with identities of its own.

- Validation, before anything else: xs and ys are tuples or lists (operations_v6.run hands over tuples) whose items
  are exact ints or Rationals of exact ints in lowest terms with a positive denominator (a Rational made with
  object.__new__ may break that form); then the same length, at least 2 pairs. Everything else is invalid input.
- Integers: with D and E the least common multiples of the denominators of the xs and of the ys, X_i = D x_i and
  Y_i = E y_i are integers. One pass over the pairs forms the raw moments sum X, sum Y, sum X^2, sum X Y and, for
  correlation, sum Y^2. Then, by the raw-moment identities (the checker uses centred deviations instead),
      a = n sum X^2 - (sum X)^2 = n D^2 S_xx,   b = n sum Y^2 - (sum Y)^2 = n E^2 S_yy,
      c = n sum X Y - sum X sum Y = n D E S_xy,
  where a = sum_{i<j} (X_i - X_j)^2 is 0 exactly when the xs are all equal (b likewise for the ys): the zero-variance
  inputs, which are invalid input.
- correlation: q = c^2 / (a b) = S_xy^2 / (S_xx S_yy), reduced by one gcd, and the value
  mathforge.model.mul(sign(c), mathforge.model.sqrt(q)), the card's construction of the canonical form.
- linear_regression: the solution of the normal equations by Cramer's rule on the scaled sums,
  slope = c D / (a E) and intercept = (sum Y sum X^2 - sum X sum X Y) / (a E), each reduced by one gcd.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and one for each item it validates, refuses an
integer longer than max_integer_bits with ResourceLimitError, and charges every further step before it runs, on the
scale of the cost model of trusted.integers: product_work for a product, division_work by the length of a quotient, a
gcd quadratic in its shorter operand (one unit per 64 squared 64-bit words, as trusted.upoly.charge_gcd), a perfect-
square test (math.isqrt) about two products of its length, and a unit per 256 words of a linear pass. A common
denominator D or E longer than max_integer_bits raises ResourceLimitError, as the checker does (the scaled data carry
it), and so does a reduced integer of the value longer than max_integer_bits, which no Result may hold: for data of
random b-bit integers that happens from about b = 16,400 for correlation (a radical holds q, which has about four
times their bits; a rational coefficient r holds only the roots of q's numerator and denominator, so q itself is
never bounded) and b = 21,900 for linear_regression (the intercept's numerator has three times). Measured on one
machine with CPython: 20,000 pairs of 64-bit integers, about the most that a request and its Result may hold, take
about 15 ms and 120,000 units, 8,000 pairs of small rationals about 30 ms and 111,000 units (validation and common
denominators at about 0.3 microseconds a unit, the pass over the pairs at 0.2), and pairs of 65,536-bit integers
exhaust the default max_work of 2,000,000 after about 0.3 seconds. The checker spends between 0.2 and 2.9 times the
producer's work on the same input (most on many short items), so near the top of the budget a value may be computed
and yet its verification exceed the budget: operations_v6.run then reports resource_limit_exceeded, never a value.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/stats/test_bivariate.py is the definition of done.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import gcd
from typing import Any

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Rational, mul, sqrt, trusted_rational
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

# The scope text of every operation of wave M4 (docs/v09/cards/m4-linalg-geometry-stats-discrete.md).
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

SPECS = (
    OperationSpec("correlation", "stats",
                  (Argument("xs", "", None, "rational_list"), Argument("ys", "", None, "rational_list")),
                  "Expression", "Compute the exact Pearson correlation coefficient of paired rational data.", SCOPE,
                  version=6),
    OperationSpec("linear_regression", "stats",
                  (Argument("xs", "", None, "rational_list"), Argument("ys", "", None, "rational_list")),
                  "LinearFit", "Fit the exact least-squares line y = slope*x + intercept to paired rational data.",
                  SCOPE, version=6),
)


def _integers(budget: Budget, value: object, name: str) -> None:
    """Check that a field of a linear fit is a Rational of two exact ints in lowest terms with a positive denominator:
    a Rational made with object.__new__ may lack a slot or break that form. ResourceLimitError for an integer longer
    than max_integer_bits."""
    if type(value) is not Rational:
        raise InvalidInput(f"The {name} of a linear fit is a Rational: an int, a float or a bool is no Rational")
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"The {name} of a linear fit is a Rational of two exact integers with a positive "
                           "denominator")
    budget.check_int(numerator)
    budget.check_int(denominator)
    words = max(numerator.bit_length(), denominator.bit_length()) >> 6
    budget.tick(1 + (words * words >> 6))  # math.gcd, Lehmer's algorithm: quadratic in the length.
    if gcd(numerator, denominator) != 1:
        raise InvalidInput(f"The {name} of a linear fit is a Rational in lowest terms")


@dataclass(frozen=True)
class LinearFit:
    """The least-squares line y = slope*x + intercept of paired data.

    The canonical form, which each line has exactly once: slope and intercept are Rationals of exact ints in lowest
    terms with a positive denominator, an integral one included (an int, a bool or a float is no Rational here)."""
    slope: Rational
    intercept: Rational

    def __post_init__(self) -> None:
        with computation() as budget:
            budget.tick()
            _integers(budget, self.slope, "slope")
            _integers(budget, self.intercept, "intercept")


TYPES = {"linear_fit": LinearFit}
CHECKER = "mathforge.certify.stats.bivariate"

_ARGUMENTS = frozenset(("xs", "ys"))  # The argument names of both specs.
Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1, in lowest terms.


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of correlation or linear_regression, for arguments that operations_v6.run has checked by name
    and kind (a direct call is checked here as well); the module states the algorithm. InvalidInput for anything but
    two tuples or lists of exact rationals, for lists of different lengths, fewer than 2 pairs and zero variance;
    ResourceLimitError past the budget."""
    compute = _PRODUCERS.get(operation) if type(operation) is str else None
    if compute is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the bivariate family")
    if type(args) is not dict or set(args) != _ARGUMENTS:
        raise InvalidInput(f"{operation} takes exactly the arguments xs and ys")
    xs, ys = args["xs"], args["ys"]
    if (type(xs) is not tuple and type(xs) is not list) or (type(ys) is not tuple and type(ys) is not list):
        raise InvalidInput(f"{operation} requires xs and ys as tuples or lists of exact rationals")
    with computation() as budget:
        budget.tick()  # The call itself.
        x_pairs = _pairs(budget, operation, "xs", xs)
        y_pairs = _pairs(budget, operation, "ys", ys)
        if len(x_pairs) != len(y_pairs):
            raise InvalidInput(f"{operation} requires xs and ys of the same length")
        if len(x_pairs) < 2:
            raise InvalidInput(f"{operation} requires at least 2 pairs (x, y)")
        return compute(budget, x_pairs, y_pairs)


def _correlation(budget: Budget, x_pairs: list[Pair], y_pairs: list[Pair]) -> tuple[object, tuple]:
    """sign(c) sqrt(c^2 / (a b)) in the canonical form of the module, and the evidence ()."""
    count = len(x_pairs)
    x_common, y_common = _common_denominator(budget, x_pairs), _common_denominator(budget, y_pairs)
    sum_x, sum_y, sum_xx, sum_yy, sum_xy = _moments(budget, x_pairs, y_pairs, x_common, y_common, True)
    spread_x = _difference(budget, _product(budget, count, sum_xx), _product(budget, sum_x, sum_x))  # n D^2 S_xx.
    if not spread_x:
        raise InvalidInput("correlation requires xs that are not all equal: a constant variable has zero variance and "
                           "no correlation coefficient")
    spread_y = _difference(budget, _product(budget, count, sum_yy), _product(budget, sum_y, sum_y))  # n E^2 S_yy.
    if not spread_y:
        raise InvalidInput("correlation requires ys that are not all equal: a constant variable has zero variance and "
                           "no correlation coefficient")
    joint = _difference(budget, _product(budget, count, sum_xy), _product(budget, sum_x, sum_y))  # n D E S_xy.
    # q = c^2 / (a b) = S_xy^2 / (S_xx S_yy): the positive factors n^2 D^2 E^2 cancel. a b > 0 (Cauchy-Schwarz).
    # q is not bounded here: when it is the square of a rational r, the value holds r, whose integers have half the
    # length of q's (data permuted between xs and ys give S_yy = S_xx and r = S_xy / S_xx, about 34,000 bits for
    # 17,000-bit data, while q has 68,000), and only the value's integers must fit max_integer_bits.
    q = _reduced(budget, _product(budget, joint, joint), _product(budget, spread_x, spread_y), bounded=False)
    _charge_radical(budget, q)
    value = mul((joint > 0) - (joint < 0), sqrt(q))
    if type(value) is not Rational:  # A radical holds q itself (model.sqrt and mul check no length).
        budget.check_int(q.numerator)
        budget.check_int(q.denominator)
    return value, ()


def _linear_regression(budget: Budget, x_pairs: list[Pair], y_pairs: list[Pair]) -> tuple[object, tuple]:
    """LinearFit(slope, intercept) by Cramer's rule on the normal equations of the scaled data, and the evidence ()."""
    count = len(x_pairs)
    x_common, y_common = _common_denominator(budget, x_pairs), _common_denominator(budget, y_pairs)
    sum_x, sum_y, sum_xx, _, sum_xy = _moments(budget, x_pairs, y_pairs, x_common, y_common, False)
    spread_x = _difference(budget, _product(budget, count, sum_xx), _product(budget, sum_x, sum_x))  # n D^2 S_xx.
    if not spread_x:
        raise InvalidInput("linear_regression requires xs that are not all equal: every line through the mean point "
                           "then fits the data equally well")
    joint = _difference(budget, _product(budget, count, sum_xy), _product(budget, sum_x, sum_y))  # n D E S_xy.
    # With x = X / D and y = Y / E, Cramer's rule on [[sum x^2, sum x], [sum x, n]] (slope, intercept) =
    # (sum x y, sum y) gives slope = c D / (a E) and intercept = (sum Y sum X^2 - sum X sum X Y) / (a E).
    denominator = _product(budget, spread_x, y_common)  # a E > 0.
    slope = _reduced(budget, _product(budget, joint, x_common), denominator)
    intercept = _reduced(budget, _difference(budget, _product(budget, sum_y, sum_xx), _product(budget, sum_x, sum_xy)),
                         denominator)
    return LinearFit(slope, intercept), ()


_PRODUCERS: dict[str, Callable[[Budget, list[Pair], list[Pair]], tuple[object, tuple]]] = {
    "correlation": _correlation,
    "linear_regression": _linear_regression,
}


def _pairs(budget: Budget, operation: str, name: str, numbers: tuple[object, ...] | list[object]) -> list[Pair]:
    """(numerator, denominator) of each item of xs or ys: an exact int, or a Rational of exact ints in lowest terms
    with a positive denominator. InvalidInput for anything else; ResourceLimitError for an integer longer than
    max_integer_bits, as the request's inspection raises."""
    pairs = []
    for number in numbers:
        budget.tick()
        kind = type(number)
        if kind is int:
            budget.check_int(number)
            pairs.append((number, 1))
            continue
        if kind is not Rational:
            raise InvalidInput(f"{operation} requires exact rationals in {name}: ints or Rationals (a bool, a float or "
                               "a str is no exact number)")
        numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            raise InvalidInput(f"{operation} requires each Rational of {name} to hold exact integers with a positive "
                               "denominator")
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator > 1:
            budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
            if gcd(numerator, denominator) != 1:
                raise InvalidInput(f"{operation} requires each Rational of {name} in lowest terms")
        pairs.append((numerator, denominator))
    return pairs


def _common_denominator(budget: Budget, pairs: list[Pair]) -> int:
    """The least common multiple of the denominators, built up one denominator d at a time as
    common * (d / gcd(d, common)); ResourceLimitError once it is longer than max_integer_bits."""
    from ...trusted.integers import product_work

    limit = budget.limits.max_integer_bits
    common = 1
    for _, denominator in pairs:
        budget.tick()
        if denominator == 1:
            continue
        common_bits, bits = common.bit_length(), denominator.bit_length()
        budget.tick(_gcd_work(common_bits, bits))
        divisor = gcd(common, denominator)
        if divisor == denominator:  # d already divides the common denominator.
            continue
        factor_bits = bits - divisor.bit_length() + 1
        budget.tick(_quotient_work(bits, divisor.bit_length()) + product_work(common_bits, factor_bits)
                    + _linear(common_bits + factor_bits))
        common *= denominator // divisor
        if common.bit_length() > limit:
            raise ResourceLimitError("Computation limit exceeded: common denominator bits")
    return common


def _moments(budget: Budget, x_pairs: list[Pair], y_pairs: list[Pair], x_common: int, y_common: int,
             squares_y: bool) -> tuple[int, int, int, int, int]:
    """(sum X, sum Y, sum X^2, sum Y^2, sum X Y) of the data scaled to integers X_i = D x_i and Y_i = E y_i, by D and
    E the common denominators of the xs and of the ys, in one pass that forms each X_i and Y_i as it goes (no list of
    long integers is kept); sum Y^2 only with squares_y, else 0."""
    from ...trusted.integers import product_work

    sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0
    for x, y in zip(_scaled(budget, x_pairs, x_common), _scaled(budget, y_pairs, y_common)):
        x_bits, y_bits = x.bit_length(), y.bit_length()
        # Three products (two without squares_y) and five additions (four), each by the length of its operands.
        budget.tick(2 + product_work(x_bits, x_bits) + product_work(x_bits, y_bits)
                    + (product_work(y_bits, y_bits) if squares_y else 0)
                    + _linear(3 * (x_bits + y_bits) + sum_xx.bit_length() + sum_yy.bit_length() + sum_xy.bit_length()))
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_xy += x * y
        if squares_y:
            sum_yy += y * y
    return sum_x, sum_y, sum_xx, sum_yy, sum_xy


def _scaled(budget: Budget, pairs: list[Pair], common: int) -> Iterator[int]:
    """The integers numerator * (common / denominator) of the pairs, for a common multiple of their denominators,
    formed one at a time, each charged before it runs: a division by the length of its quotient, then a product."""
    from ...trusted.integers import division_work, product_work

    common_bits = common.bit_length()
    for numerator, denominator in pairs:
        if denominator == common:  # The pair's unit in _moments covers it.
            yield numerator
            continue
        bits, numerator_bits = denominator.bit_length(), numerator.bit_length()
        factor_bits = common_bits - bits + 1
        budget.tick(2 + division_work(factor_bits + 64, bits) + product_work(numerator_bits, factor_bits)
                    + _linear(common_bits + numerator_bits))
        yield numerator * (common // denominator)


def _reduced(budget: Budget, numerator: int, denominator: int, *, bounded: bool = True) -> Rational:
    """numerator / denominator in lowest terms, for a positive denominator: one gcd and two exact divisions, charged
    before they run. When bounded, ResourceLimitError when a reduced integer is longer than max_integer_bits: no
    Result holds such a value (operations_v6.run's inspection of the value refuses it, and so does the checker)."""
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    budget.tick(_gcd_work(numerator_bits, denominator_bits))
    divisor = gcd(numerator, denominator)  # Positive, as the denominator is.
    if divisor > 1:
        divisor_bits = divisor.bit_length()
        budget.tick(2 + _quotient_work(numerator_bits, divisor_bits) + _quotient_work(denominator_bits, divisor_bits))
        numerator //= divisor
        denominator //= divisor
    if bounded:
        budget.check_int(numerator)
        budget.check_int(denominator)
    return trusted_rational(numerator, denominator)  # Exact ints in lowest terms, the denominator positive.


def _charge_radical(budget: Budget, q: Rational) -> None:
    """Charge, before they run, mathforge.model.mul(sign, mathforge.model.sqrt(q)) for a reduced q >= 0: a perfect-
    square test (math.isqrt and the square of the root) of the numerator and of the denominator in each of the three
    calls of model.sqrt that a radical goes through (mul simplifies its factor twice), and the gcd of the two roots
    that the Rational of a rational square checks."""
    numerator_bits, denominator_bits = q.numerator.bit_length(), q.denominator.bit_length()
    budget.tick(3 * (_square_test_work(numerator_bits) + _square_test_work(denominator_bits))
                + _gcd_work((numerator_bits >> 1) + 1, (denominator_bits >> 1) + 1))


# ---------------------------------------------------------------------------------------------------------------
# Charged integer steps, on the scale of the cost model of trusted.integers

def _product(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs."""
    from ...trusted.integers import product_work

    left_bits, right_bits = left.bit_length(), right.bit_length()
    budget.tick(1 + product_work(left_bits, right_bits) + _linear(left_bits + right_bits))
    return left * right


def _difference(budget: Budget, left: int, right: int) -> int:
    """left - right, charged by the length of its operands."""
    budget.tick(1 + _linear(left.bit_length() + right.bit_length()))
    return left - right


def _linear(bits: int) -> int:
    """The units of a linear pass over integers of this many bits in all: one per 256 64-bit words."""
    return bits >> 14


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient (division_work by the whole dividend
    would charge a short quotient as a long one)."""
    from ...trusted.integers import division_work

    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: one, a division of the longer by the shorter, then
    Lehmer's algorithm on the shorter, quadratic, one unit per 64 squared 64-bit words as trusted.upoly.charge_gcd
    charges it (16,384 at 65,536 bits; measured 0.11 to 0.22 microseconds a unit from 8,192 to 524,288 bits on random
    operands, far less when the operands share a long factor and the algorithm ends early)."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + _quotient_work(large, small) + (large >> 12) + (words * words >> 6)


def _square_test_work(bits: int) -> int:
    """The units of a perfect-square test of an integer of this many bits: math.isqrt, about two products of its
    length (measured), and the square of the root, half as long."""
    from ...trusted.integers import product_work

    half = (bits >> 1) + 1
    return 2 + 2 * product_work(bits, bits) + product_work(half, half) + _linear(bits)


def _render(fit: LinearFit, renderer: Any, child: Callable[..., str]) -> str:
    r"""y = 2*x + 1: the slope before x, then the intercept with its sign, as a rendered Polynomial writes them
    (y = 3/2*x + 5/6, y = -2*x - 1/3, y = x, y = -x + 4); a zero intercept is left out (y = 2*x), and a zero slope
    leaves the intercept alone (y = 5, y = 0). LaTeX: y = 2\, x + 1 and y = \frac{3}{2}\, x - \frac{1}{3}."""
    slope, intercept = fit.slope, fit.intercept
    if not slope.numerator:
        return renderer.join(("y = ", child(intercept)))
    if slope.denominator == 1 and slope.numerator in (1, -1):
        term = "x" if slope.numerator == 1 else "-x"
    else:
        term = renderer.join((child(slope), r"\, " if renderer.latex else "*", "x"))
    if not intercept.numerator:
        return renderer.join(("y = ", term))
    constant = child(intercept)  # A negative intercept renders with a leading "-", which the sign takes over.
    sign, constant = (" - ", constant[1:]) if constant.startswith("-") else (" + ", constant)
    return renderer.join(("y = ", term, sign, constant))


RENDER = {LinearFit: _render}
