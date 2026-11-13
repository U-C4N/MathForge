"""Exact tests on tables of counts (mathforge.stats): the two-sided p-value of Fisher's exact test on a 2 x 2 table.

fisher_exact(table) -> Rational
    Domain: table is a 2 x 2 Matrix (MCP kind "matrix") of nonnegative integers [[a, b], [c, d]], not all zero
    (docs/v09/cards/m7-extended.md). A matrix of another shape, a non-integral or negative entry and the zero table are
    invalid input; a total a + b + c + d above 100,000 exceeds the resource limit.
    Value: with the row sums r1 = a + b, r2 = c + d, the column sums c1 = a + c, c2 = b + d and n = r1 + r2 fixed,
    the top-left count X is hypergeometric, P(X = k) = C(r1, k) C(r2, c1 - k) / C(n, c1) for
    max(0, c1 - r2) <= k <= min(r1, c1). The p-value is the sum of P(X = k) over every k with P(X = k) <= P(X = a),
    decided exactly (no tolerance), as a Rational in lowest terms with a positive denominator, 1 included (never an
    int): [[1, 9], [11, 3]] -> 41/14858, [[3, 1], [1, 3]] -> 17/35, [[0, 5], [5, 0]] -> 1/126, [[2, 2], [2, 2]] -> 1,
    [[8, 2], [1, 5]] -> 5/143.
    Evidence: ().

The checker, mathforge.certify.stats.exact_tests, recomputes the masses up to a common positive factor by the adjacent
ratio P(k + 1) / P(k) = (r1 - k)(c1 - k) / ((k + 1)(r2 - c1 + k + 1)) from the lowest k, without binomial
coefficients, and compares the claim with the selected sum over the total by one exact cross-multiplication.

How produce() computes (binomial coefficients, as the card prescribes). Nothing here is a proof: the checker verifies
the value with a recurrence of its own.

- It charges one unit for the call and max_matrix_entries for the table (current_budget().matrix) before it validates
  anything but the class of the table, then the shape (2 rows of 2 entries, ncols 2) and the entries: exact Rationals
  with int parts, each within max_integer_bits, a denominator of 1 and a numerator >= 0. Not all zero; a total above
  MAX_TOTAL raises ResourceLimitError.
- First the table is transposed and its columns swapped as needed so that c1 is its smallest margin (the p-value is
  invariant under both, see _oriented): the support is then 0 <= k <= c1 and every binomial below has at most
  c1 bits(n) bits, so a table with a small margin costs little however large its total.
- With N_k = C(r1, k) C(r2, c1 - k), the unnormalized mass, the p-value is S / T with S the sum of the N_k with
  N_k <= N_a over the support lo = max(0, c1 - r2) <= k <= hi = min(r1, c1), and T = C(n, c1), by Vandermonde's
  identity sum_k N_k = C(n, c1) (the checker sums its weights instead).
- N_a, N_lo and T come from math.comb directly. The N_k between lo and hi follow from the binomial recurrences
  C(r1, k + 1) = C(r1, k) (r1 - k) / (k + 1) and C(r2, c1 - k - 1) = C(r2, c1 - k) (c1 - k) / (r2 - c1 + k + 1), as
  one exact division of N_k (r1 - k)(c1 - k) by (k + 1)(r2 - c1 + k + 1) per step: a nonzero remainder, or a
  recurrence value at k = a other than the direct N_a, is an internal error that the true binomials never give.
- p/q = S / T reduced by one gcd, refused with ResourceLimitError when either integer is longer than max_integer_bits
  (the checker refuses such a value, and no Result may hold it); S = T gives Rational(1).

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges every step before it runs, on the scale of the cost model of
trusted.integers: product_work and a linear pass (one unit per 256 64-bit words) for a product, division_work for a
division, and for math.comb about two products of the length of its result, bounded before it is formed. The loop is
one pass over the support, where the checker makes two, so the verifier is the binding limit: balanced tables from a
total of about 60,000 and lopsided ones such as [[50000, 0], [0, 50000]] exceed the default max_work of 2,000,000 in
the producer or in the verifier, and operations_v6.run then reports resource_limit_exceeded, never a value.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/stats/test_exact_tests.py is the definition
of done.
"""

from __future__ import annotations

from math import comb, gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...matrices import Matrix
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

# The scope text of the stats area's v0.9 families (areas/stats/bivariate.py), which covers a table of counts.
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

SPECS = (
    OperationSpec("fisher_exact", "stats",
                  (Argument("table", "Matrix", None, "matrix"),),
                  "Rational", "Compute the exact two-sided p-value of Fisher's test on a 2x2 table of counts.", SCOPE,
                  version=6),
)

# The value is a v0.7 Rational: the family defines no record class, so there is nothing to render either.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.stats.exact_tests"
MAX_TOTAL = 100_000  # The card's resource limit on a + b + c + d.


_NOT_A_TABLE = "fisher_exact requires a 2 x 2 Matrix of nonnegative integer counts"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(p-value, ()) of fisher_exact for args = {"table": table}, as the module states. InvalidInput for a table that is
    no 2 x 2 Matrix of nonnegative integers and for the zero table; ResourceLimitError for a total above MAX_TOTAL and
    past the budget, whose max_matrix_entries is charged before anything but the class of the table is validated."""
    if type(operation) is not str or operation != "fisher_exact":
        raise UnsupportedOperation(f"{operation!r} is not an operation of the exact_tests family")
    if type(args) is not dict or set(args) != {"table"}:
        raise InvalidInput("fisher_exact takes exactly the argument table")
    table = args["table"]
    with computation() as budget:
        budget.tick()  # The call itself.
        if type(table) is not Matrix:  # A subclass, nested lists or any other object is no Matrix.
            raise InvalidInput(_NOT_A_TABLE)
        rows, ncols = getattr(table, "rows", None), getattr(table, "ncols", None)  # A forged Matrix may lack them.
        if type(rows) is not tuple or type(ncols) is not int:
            raise InvalidInput(_NOT_A_TABLE)
        budget.matrix(len(rows), ncols)  # First: a 65 x 65 matrix exceeds the budget before its shape is judged.
        if len(rows) != 2 or ncols != 2:
            raise InvalidInput(f"fisher_exact requires a 2 x 2 table, not {len(rows)} x {ncols}")
        a, b, c, d = _counts(budget, rows)
        total = a + b + c + d
        if total == 0:
            raise InvalidInput("fisher_exact requires a table with at least one nonzero count")
        if total > MAX_TOTAL:
            raise ResourceLimitError(f"Computation limit exceeded: fisher_exact takes a total of at most {MAX_TOTAL:,}")
        return _p_value(budget, a, b, c, d), ()


def _counts(budget: Budget, rows: tuple) -> tuple[int, int, int, int]:
    """(a, b, c, d) of the rows ((a, b), (c, d)). InvalidInput for a row that is no tuple of 2 entries and an entry
    that is no exact Rational with int parts, denominator 1 and numerator >= 0; ResourceLimitError for an integer
    longer than max_integer_bits."""
    counts = []
    for row in rows:
        budget.tick(3)
        if type(row) is not tuple or len(row) != 2:
            raise InvalidInput(_NOT_A_TABLE)
        for entry in row:
            if type(entry) is not Rational:
                raise InvalidInput(_NOT_A_TABLE)
            numerator = getattr(entry, "numerator", None)  # A forged Rational may lack a slot.
            denominator = getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int:
                raise InvalidInput(_NOT_A_TABLE)
            budget.check_int(numerator)
            budget.check_int(denominator)
            if denominator != 1:
                raise InvalidInput(f"fisher_exact requires integer counts, not {numerator}/{denominator}")
            if numerator < 0:
                raise InvalidInput(f"fisher_exact requires nonnegative counts, not {numerator}")
            counts.append(numerator)
    a, b, c, d = counts
    return a, b, c, d


def _oriented(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    """The table [[a, b], [c, d]] transposed and with swapped columns as needed, so that its first column sum is its
    smallest margin. The probability of a table with fixed margins, r1! r2! c1! c2! / (n! a! b! c! d!), is symmetric
    under transposition and under swapping the rows or the columns, which map the tables of one set of margins
    bijectively onto those of the other, so the p-value is unchanged. With c1 = min(r1, r2, c1, c2) the support is
    0 <= k <= c1, and every binomial below has at most c1 bits(n) bits: a table such as [[1, 0], [50000, 49999]] then
    costs a few units, where C(99999, 50001) and C(100000, 50001) would cost about 57,000 (the checker, whose weights
    are scaled down to the least integral start, needs about 20)."""
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    smallest = min(r1, r2, c1, c2)
    if smallest == c1:
        return a, b, c, d
    if smallest == c2:
        return b, a, d, c  # Columns swapped.
    if smallest == r1:
        return a, c, b, d  # Transposed.
    return c, a, d, b  # Transposed, then columns swapped: the first column sum is r2.


def _p_value(budget: Budget, a: int, b: int, c: int, d: int) -> Rational:
    """The two-sided p-value S / T of the module, in lowest terms."""
    budget.tick()
    a, b, c, d = _oriented(a, b, c, d)
    r1, r2, c1 = a + b, c + d, a + c
    low, high = max(0, c1 - r2), min(r1, c1)
    observed = _product(budget, _comb(budget, r1, a), _comb(budget, r2, c))  # N_a, directly.
    weight = _product(budget, _comb(budget, r1, low), _comb(budget, r2, c1 - low))  # N_lo.
    selected = 0
    for k in range(low, high + 1):
        bits = weight.bit_length()
        budget.tick(3 + _linear(2 * bits + selected.bit_length()))
        if k == a and weight != observed:
            raise ArithmeticError("the binomial recurrence disagrees with math.comb")  # Never reached.
        if weight <= observed:
            selected += weight
        if k == high:
            break
        up, down = (r1 - k) * (c1 - k), (k + 1) * (r2 - c1 + k + 1)  # Both positive inside the support.
        product = _product(budget, weight, up)
        budget.tick(1 + _division_work(product.bit_length() + 64, down.bit_length()) + _linear(product.bit_length()))
        weight, remainder = divmod(product, down)  # N_(k+1): an exact quotient of products of binomials.
        if remainder:
            raise ArithmeticError("the binomial recurrence left the integers")  # Never reached.
    whole = _comb(budget, r1 + r2, c1)  # T = C(n, c1) = sum_k N_k (Vandermonde).
    return _reduced(budget, selected, whole)


def _reduced(budget: Budget, numerator: int, denominator: int) -> Rational:
    """numerator / denominator, 0 < numerator <= denominator, in lowest terms; ResourceLimitError when either reduced
    integer is longer than max_integer_bits."""
    if numerator == denominator:
        budget.tick()
        return Rational(1)
    left, right = numerator.bit_length(), denominator.bit_length()
    words = left >> 6
    budget.tick(1 + _division_work(right - left + 64, left) + _linear(right) + (words * words >> 6))  # math.gcd.
    divisor = gcd(numerator, denominator)
    budget.tick(2 + 2 * _division_work(right + 64, divisor.bit_length()))
    numerator, denominator = numerator // divisor, denominator // divisor
    limit = budget.limits.max_integer_bits
    if numerator.bit_length() > limit or denominator.bit_length() > limit:
        raise ResourceLimitError("Computation limit exceeded: integer bits (the p-value)")
    return trusted_rational(numerator, denominator)


# ---------------------------------------------------------------------------------------------------------------
# Charged integer steps, on the scale of the cost model of trusted.integers (imported on use, as kernels are)

def _division_work(dividend_bits: int, divisor_bits: int) -> int:
    """trusted.integers.division_work: the units of a long division beyond one."""
    from ...trusted.integers import division_work

    return division_work(dividend_bits, divisor_bits)


def _linear(bits: int) -> int:
    """The units of a linear pass over integers of this many bits in all: one per 256 64-bit words."""
    return bits >> 14


def _product(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs."""
    from ...trusted.integers import product_work

    left_bits, right_bits = left.bit_length(), right.bit_length()
    budget.tick(1 + product_work(left_bits, right_bits) + _linear(left_bits + right_bits))
    return left * right


def _comb(budget: Budget, n: int, k: int) -> int:
    """math.comb(n, k) for 0 <= k <= n, charged before it runs by a bound on the length of its result,
    log2 C(n, k) <= min(n, min(k, n - k) bits(n)): CPython builds it from products of halves, about two products of
    that length in all."""
    from ...trusted.integers import product_work

    bits = min(n, min(k, n - k) * n.bit_length()) + 1
    budget.tick(1 + 2 * product_work(bits, bits) + _linear(4 * bits))
    return comb(n, k)
