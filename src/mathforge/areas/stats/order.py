"""Order statistics of exact rational data (mathforge.stats): the type-7 quantile and every most frequent value.

Both operations take data: exact rationals, ints or Rationals (MCP kind "rational_list": operations_v6.run freezes a
list into a tuple and already rejects a bool, a float, a str or None in it as invalid input), at least one of them. The
argument is unannotated, as v0.7's rational_list arguments are (mean(data)): a list is accepted as well as a tuple.
With the sorted data x_(0) <= ... <= x_(n-1) (docs/v09/cards/m7-extended.md):

quantile(data, p) -> Rational
    Domain: nonempty data; p an exact rational (an int or a Rational) with 0 <= p <= 1. Empty data and any other p
    are invalid input.
    Value: with h = (n - 1) p and j = floor(h), x_(j) when h == j, else x_(j) + (h - j)(x_(j+1) - x_(j)), as a
    Rational in lowest terms (an integral one included, never an int).
    Evidence: the order statistics used, as (rank, value) pairs: ((j, x_(j)),) when h == j, else
    ((j, x_(j)), (j + 1, x_(j+1))); each rank an exact int, each value an exact int or a Rational in lowest terms.

mode(data) -> tuple[Rational, ...]
    Domain: nonempty data; empty data is invalid input.
    Value: every value of maximal multiplicity, strictly ascending, each a Rational in lowest terms (an integral one
    included, never an int). Data items are compared as numbers: 2 and Rational(2) are one value.
    Evidence: (multiplicity,), the maximal multiplicity as an exact int >= 1.

The checker, mathforge.certify.stats.order, never sorts the data: it proves each order statistic of the evidence by
counting, #{x < v} <= rank < #{x <= v}, and verifies the quantile by the formula; for mode it counts the
multiplicities in one pass and compares the value with the values of maximal count.

How produce() computes (the card: "the producer sorts and runs"). It validates every item of data to a reduced pair
(numerator, denominator), then sorts the pairs: plain ints by their own order when every denominator is 1, else by
cross-multiplication, each comparison charged by the length of its products before it runs. quantile forms
(n - 1) p = t / d, j = floor(t / d) and the remainder r, and builds x_(j) + (r / d)(x_(j+1) - x_(j)) as one reduced
fraction; mode walks the sorted pairs in runs of equal pairs (a reduced pair is the unique form of its number, so
equal numbers have equal pairs) and keeps the runs of maximal length, which the sort leaves ascending. No dict counts
the data: integers that share one hash cannot make the producer quadratic.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/stats/test_order.py is the definition of done.
"""

from __future__ import annotations

from functools import cmp_to_key
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

# The scope text of the stats area's v0.9 families (areas/stats/bivariate.py), which already names data and statistics.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

# data is unannotated, as v0.7's rational_list arguments are (mean(data)): a list is accepted as well as a tuple.
SPECS = (
    OperationSpec("quantile", "stats",
                  (Argument("data", "", None, "rational_list"), Argument("p", "Rational | int", None, "rational")),
                  "Rational", "Compute the exact type-7 quantile of rational data.", SCOPE, version=6),
    OperationSpec("mode", "stats",
                  (Argument("data", "", None, "rational_list"),),
                  "tuple[Rational, ...]", "Find every most frequent value of rational data.", SCOPE, version=6),
)

# Both values are v0.7 objects (a Rational, a tuple of Rationals): the family defines no record class and no RENDER.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.stats.order"


Pair = tuple[int, int]  # (numerator, denominator) of an exact rational, denominator >= 1, in lowest terms.
_ARGUMENTS = {"quantile": frozenset(("data", "p")), "mode": frozenset(("data",))}  # The argument names of each spec.


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of quantile or mode, for arguments that operations_v6.run has checked by name and kind (a
    direct call is checked here as well); the module states the contract and the algorithm. InvalidInput for anything
    but a nonempty tuple or list of exact rationals as data, and for a p that is no exact rational in [0, 1];
    ResourceLimitError for an integer longer than max_integer_bits and past the budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the order family")
    if type(args) is not dict or set(args) != names:
        raise InvalidInput(f"{operation} takes exactly the arguments {' and '.join(sorted(names))}")
    data = args["data"]
    if type(data) is not tuple and type(data) is not list:
        raise InvalidInput(f"{operation} requires data as a tuple or list of exact rationals")
    with computation() as budget:
        budget.tick()  # The call itself.
        pairs = _pairs(budget, operation, "data", data)
        p = _number(budget, operation, "p", args["p"]) if operation == "quantile" else None
        if not pairs:
            raise InvalidInput(f"{operation} requires nonempty data")
        if p is None:
            return _mode(budget, pairs)
        if p[0] < 0 or p[0] > p[1]:
            raise InvalidInput("quantile requires p with 0 <= p <= 1")
        return _quantile(budget, pairs, p)


def _quantile(budget: Budget, pairs: list[Pair], p: Pair) -> tuple[Rational, tuple]:
    """The type-7 quantile: (n - 1) p = t / d, j = floor(t / d), r = t - j d, and x_(j) + (r / d)(x_(j+1) - x_(j)),
    with the order statistics it used as evidence."""
    ordered = _sorted(budget, pairs)
    p_numerator, p_denominator = p
    t = _product(budget, len(ordered) - 1, p_numerator)
    budget.tick(1 + _quotient_work(t.bit_length(), p_denominator.bit_length()))
    j, remainder = divmod(t, p_denominator)
    a, b = ordered[j]
    if remainder == 0:
        return trusted_rational(a, b), ((j, _exact(a, b)),)
    c, e = ordered[j + 1]  # j < n - 1 here, as h = t / d < n - 1 whenever r != 0.
    # x_(j) + (r / d)(c / e - a / b) = (a d e + r (c b - a e)) / (d b e), then reduced.
    denominator = _product(budget, _product(budget, p_denominator, b), e)
    gap = _difference(budget, _product(budget, c, b), _product(budget, a, e))
    numerator = _sum(budget, _product(budget, _product(budget, a, p_denominator), e), _product(budget, remainder, gap))
    return _reduced(budget, numerator, denominator), ((j, _exact(a, b)), (j + 1, _exact(c, e)))


def _mode(budget: Budget, pairs: list[Pair]) -> tuple[tuple[Rational, ...], tuple[int]]:
    """Every value of maximal multiplicity, ascending, with the multiplicity: the runs of equal pairs of the sorted
    data, the longest kept."""
    ordered = _sorted(budget, pairs)
    best, winners = 0, []
    start = 0
    for index in range(1, len(ordered) + 1):
        if index < len(ordered):
            left, right = ordered[index - 1], ordered[index]
            budget.tick(1 + _linear(left[0].bit_length() + left[1].bit_length()))
            if left == right:  # Reduced pairs: equal exactly when the numbers are.
                continue
        length = index - start
        if length > best:
            best, winners = length, [ordered[start]]
        elif length == best:
            winners.append(ordered[start])
        start = index
    return tuple(trusted_rational(numerator, denominator) for numerator, denominator in winners), (best,)


def _sorted(budget: Budget, pairs: list[Pair]) -> list[Pair]:
    """The pairs in ascending order of their numbers. Plain integers sort by their own order, charged before the sort
    by its comparisons (about n log2 n, each linear in the length of its operands); fractions sort by
    cross-multiplication, each comparison charged by its two products before they run."""
    from ...trusted.integers import product_work

    count = len(pairs)
    if all(denominator == 1 for _, denominator in pairs):
        rounds = count.bit_length()  # At least log2(n): the comparisons of a merge sort per item, Python's sort.
        budget.tick(count * rounds + rounds * sum(_linear(numerator.bit_length()) for numerator, _ in pairs))
        return sorted(pairs)

    def compare(left: Pair, right: Pair) -> int:
        """-1, 0 or 1 as left <, == or > right, for reduced pairs with positive denominators."""
        left_bits, right_bits = left[0].bit_length(), right[0].bit_length()
        left_denominator_bits, right_denominator_bits = left[1].bit_length(), right[1].bit_length()
        budget.tick(2 + product_work(left_bits, right_denominator_bits) + product_work(right_bits,
                                                                                         left_denominator_bits)
                    + _linear(left_bits + right_bits + left_denominator_bits + right_denominator_bits))
        cross_left, cross_right = left[0] * right[1], right[0] * left[1]
        return (cross_left > cross_right) - (cross_left < cross_right)

    budget.tick(count)  # The decoration of the sort.
    return sorted(pairs, key=cmp_to_key(compare))


def _exact(numerator: int, denominator: int) -> int | Rational:
    """An order statistic as the evidence holds it: an int when it is integral, else a Rational in lowest terms."""
    return numerator if denominator == 1 else trusted_rational(numerator, denominator)


# ---------------------------------------------------------------------------------------------------------------
# Validation

def _pairs(budget: Budget, operation: str, name: str, numbers: tuple[object, ...] | list[object]) -> list[Pair]:
    """(numerator, denominator) of each item of data. InvalidInput for an item that is no exact rational;
    ResourceLimitError for an integer longer than max_integer_bits, as the request's inspection raises."""
    return [_number(budget, operation, name, number, item=True) for number in numbers]


def _number(budget: Budget, operation: str, name: str, number: object, *, item: bool = False) -> Pair:
    """(numerator, denominator) of an exact int, or of a Rational of exact ints in lowest terms with a positive
    denominator (a Rational made with object.__new__ may lack a slot or break that form). InvalidInput for anything
    else, a bool, a float, a str, an int subclass or a Fraction included; ResourceLimitError for an integer longer
    than max_integer_bits."""
    budget.tick()
    where = f"in {name}" if item else f"as {name}"
    kind = type(number)
    if kind is int:
        budget.check_int(number)
        return number, 1
    if kind is not Rational:
        raise InvalidInput(f"{operation} requires exact rationals {where}: ints or Rationals (a bool, a float or a str "
                           "is no exact number)")
    numerator, denominator = getattr(number, "numerator", None), getattr(number, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"{operation} requires each Rational {where} to hold exact integers with a positive "
                           "denominator")
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator > 1:
        budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
        if gcd(numerator, denominator) != 1:
            raise InvalidInput(f"{operation} requires each Rational {where} in lowest terms")
    return numerator, denominator


# ---------------------------------------------------------------------------------------------------------------
# Charged integer steps, on the scale of the cost model of trusted.integers (as areas/stats/bivariate.py charges them)

def _reduced(budget: Budget, numerator: int, denominator: int) -> Rational:
    """numerator / denominator in lowest terms, for a positive denominator: one gcd and two exact divisions, charged
    before they run. ResourceLimitError when a reduced integer is longer than max_integer_bits: no Result holds such a
    value (operations_v6.run's inspection of the value refuses it, and so does the checker)."""
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    budget.tick(_gcd_work(numerator_bits, denominator_bits))
    divisor = gcd(numerator, denominator)  # Positive, as the denominator is.
    if divisor > 1:
        divisor_bits = divisor.bit_length()
        budget.tick(2 + _quotient_work(numerator_bits, divisor_bits) + _quotient_work(denominator_bits, divisor_bits))
        numerator //= divisor
        denominator //= divisor
    budget.check_int(numerator)
    budget.check_int(denominator)
    return trusted_rational(numerator, denominator)  # Exact ints in lowest terms, the denominator positive.


def _product(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs."""
    from ...trusted.integers import product_work

    left_bits, right_bits = left.bit_length(), right.bit_length()
    budget.tick(1 + product_work(left_bits, right_bits) + _linear(left_bits + right_bits))
    return left * right


def _sum(budget: Budget, left: int, right: int) -> int:
    """left + right, charged by the length of its operands."""
    budget.tick(1 + _linear(left.bit_length() + right.bit_length()))
    return left + right


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
    charges it."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + _quotient_work(large, small) + (large >> 12) + (words * words >> 6)

