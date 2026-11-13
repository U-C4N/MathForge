"""Independent checker of the family mathforge.areas.stats.exact_tests: fisher_exact.

The checker never forms a binomial coefficient, which the producer uses. It rebuilds the hypergeometric masses up to a
common positive factor from the lowest k by their adjacent ratio (docs/v09/cards/m7-extended.md) and compares the
claimed p-value with the selected sum over the total by one exact cross-multiplication.

The theorem that makes True imply the claim:

fisher_exact(table) = p/q, evidence ().
    The table is a 2 x 2 Matrix [[a, b], [c, d]] of nonnegative integers, not all zero, with total n <= 100,000. With
    r1 = a + b, r2 = c + d and c1 = a + c, the top-left count X of a table with these margins is hypergeometric:
    P(X = k) = C(r1, k) C(r2, c1 - k) / C(n, c1), positive exactly on the support lo <= k <= hi, lo = max(0, c1 - r2),
    hi = min(r1, c1), and 0 elsewhere; a lies in the support. For lo <= k < hi, with u_k = (r1 - k)(c1 - k) and
    v_k = (k + 1)(r2 - c1 + k + 1), both positive there,
        P(X = k + 1) v_k = P(X = k) u_k        (the adjacent ratio of the card),
    since C(r1, k + 1) (k + 1) = C(r1, k) (r1 - k) and C(r2, c1 - k - 1) (r2 - c1 + k + 1) = C(r2, c1 - k) (c1 - k).
    The checker forms integers W_lo, ..., W_hi with W_lo = s >= 1 and W_{k+1} v_k = W_k u_k exactly (a division whose
    remainder it requires to be 0). By induction on k, W_k = (s / P(X = lo)) P(X = k) for every k of the support: the
    integers are the masses times one positive factor. Hence, with T = sum_k W_k and S = sum of the W_k with
    W_k <= W_a (the exact form of P(X = k) <= P(X = a); a mass outside the support is 0 and adds nothing), the
    two-sided p-value is S / T. The claim p/q is a Rational of exact ints in lowest terms with q >= 1, and the checker
    requires p T == q S; as T > 0, that says p/q = S/T, and a reduced fraction with a positive denominator is unique,
    so each table has exactly one accepted value, an int (1 for Rational(1)), 2/4 for 1/2, a negative denominator and
    any other class answering False.
    The start s is the least positive integer that makes every W_k integral: a first pass from W_lo = 1 multiplies
    its running scale by v_k / gcd(W_k u_k, v_k) whenever a step leaves the integers, and also returns W_a at that
    scale. The proof rests on the second pass alone, which starts from s, checks every remainder and compares its own
    W_a with the first pass's (the comparisons for k < a used the first pass's value), answering False on any
    difference; neither happens for the true masses, so this is no condition on the claim.

Shapes. The operation must be the str "fisher_exact", args a dict of exactly "table", and the evidence the empty
tuple. The table is identified by (type(table).__module__, type(table).__qualname__) == ("mathforge.matrices",
"Matrix") and read as data: rows, a tuple of 2 tuples of 2 entries, and ncols, the exact int 2; each entry of the
exact class mathforge.model.Rational with exact int numerator >= 0 and denominator exactly 1 (a Rational made with
object.__new__ that holds 4/2 is refused, as no decoded matrix holds one). The value is of the exact class Rational,
its integers exact ints within max_integer_bits, in lowest terms, with 0 < p <= q (a p-value includes the observed
mass). A bool, a float, a str, an int or Rational subclass, a list for a tuple, another shape, the zero table, a
negative or fractional count and any evidence but () answer False, never an exception.

Resource limits. ResourceLimitError leaves a claim unverified, never verified. It is raised, before the value is
judged, for a table with more entries than max_matrix_entries (budget.matrix comes before the shape is read, as in
produce()), for a request integer over max_integer_bits (as the request's inspection raises before a checker runs) and
for a total above 100,000, the card's resource limit; and for work past max_work.

Budget. Every loop step is charged before it runs, each integer operation on the scale of the cost model of
trusted.integers: product_work and a linear pass (one unit per 256 64-bit words) for a product, division_work for a
division or remainder, one unit for a gcd of numbers below 2**34. No builtin pow is used. The W_k have at most
log2 C(n, c1) <= n bits, and each pass costs about (hi - lo) (2 + n / 1024) units; the claim enters two products only,
so a forged value of any size is refused at once.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of the class of the table.
MAX_TOTAL = 100_000  # The card's resource limit on a + b + c + d.

Pair = tuple[int, int]  # The numerator and the positive denominator of a Rational.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the two-sided Fisher p-value of args = {"table": table}, by the theorem of the
    module. False for anything but the operation's exact shapes; ResourceLimitError for a request past the limits that
    the module lists and for work past the budget."""
    if type(operation) is not str or operation != "fisher_exact":
        return False
    if type(args) is not dict or len(args) != 1 or any(type(name) is not str for name in args) or "table" not in args:
        return False
    if type(evidence) is not tuple or evidence:
        return False
    with computation() as budget:  # One budget for every step: the active one, or one of the default limits.
        budget.tick()
        counts = _counts(budget, args["table"])
        if counts is None:
            return False
        claim = _canonical_rational(budget, value)
        if claim is None:
            return False
        numerator, denominator = claim
        if numerator <= 0 or numerator > denominator:  # A p-value lies in (0, 1]: the observed mass is selected.
            return False
        selected, total = _p_value(budget, *counts)
        if total <= 0:  # Only a contradiction between the passes gives 0 (W_lo >= 1); p T = q S needs T > 0.
            return False
        return _product(budget, numerator, total) == _product(budget, denominator, selected)


# ---------------------------------------------------------------------------------------------------------------
# Shapes

def _counts(budget: Budget, table: object) -> tuple[int, int, int, int] | None:
    """(a, b, c, d) of a 2 x 2 Matrix [[a, b], [c, d]] of nonnegative integer Rationals, not all zero, read as data;
    None for anything else. budget.matrix(rows, cols) comes first, a request integer over max_integer_bits raises
    ResourceLimitError, and so does a total above MAX_TOTAL."""
    cls = type(table)
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    if type(module) is not str or type(qualname) is not str or (module, qualname) != MATRIX:
        return None
    rows, ncols = getattr(table, "rows", None), getattr(table, "ncols", None)  # A forged record may lack a field.
    if type(rows) is not tuple or type(ncols) is not int:
        return None
    budget.matrix(len(rows), ncols)  # Past max_matrix_entries: ResourceLimitError, before the shape is judged.
    if len(rows) != 2 or ncols != 2:
        return None
    counts = []
    for row in rows:
        budget.tick(3)
        if type(row) is not tuple or len(row) != 2:
            return None
        for entry in row:
            if type(entry) is not Rational:
                return None
            numerator = getattr(entry, "numerator", None)  # A forged Rational may lack a slot.
            denominator = getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int:
                return None
            budget.check_int(numerator)
            budget.check_int(denominator)
            if denominator != 1 or numerator < 0:
                return None
            counts.append(numerator)
    total = sum(counts)
    if total == 0:
        return None
    if total > MAX_TOTAL:
        raise ResourceLimitError(f"Computation limit exceeded: fisher_exact takes a total of at most {MAX_TOTAL:,}")
    a, b, c, d = counts
    return a, b, c, d


def _canonical_rational(budget: Budget, number: object) -> Pair | None:
    """(p, q) of a Rational of exact ints in lowest terms with q >= 1, both within max_integer_bits; None for
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


# ---------------------------------------------------------------------------------------------------------------
# The masses

def _p_value(budget: Budget, a: int, b: int, c: int, d: int) -> tuple[int, int]:
    """(S, T) of the module docstring: the selected and the total weight of the masses W_k, both multiples of the
    masses by one positive factor; (0, 0) when the second pass contradicts the first, which the true masses never do
    and which answers False."""
    r1, r2, c1 = a + b, c + d, a + c
    low, high = max(0, c1 - r2), min(r1, c1)
    scale, observed = _scale(budget, r1, r2, c1, low, high, a)
    weight = scale
    total = selected = 0
    observed_again = 0
    for k in range(low, high + 1):
        bits = weight.bit_length()
        budget.tick(3 + _linear(2 * bits + total.bit_length() + selected.bit_length()))
        total += weight
        if weight <= observed:
            selected += weight
        if k == a:
            observed_again = weight
        if k == high:
            break
        up, down = (r1 - k) * (c1 - k), (k + 1) * (r2 - c1 + k + 1)  # Both positive inside the support.
        product = _product(budget, weight, up)
        budget.tick(1 + division_work(product.bit_length() + 64, down.bit_length()) + _linear(product.bit_length()))
        weight, remainder = divmod(product, down)
        if remainder:  # Not reached for the true start s.
            return 0, 0
    if observed_again != observed:  # Not reached: both passes compute W_a at the scale s.
        return 0, 0
    return selected, total


def _scale(budget: Budget, r1: int, r2: int, c1: int, low: int, high: int, a: int) -> tuple[int, int]:
    """(s, W_a) of the first pass: from W_lo = 1, each step multiplies W_k by u_k / v_k, and the running scale (with
    W_a when it is already formed) by v_k / gcd(W_k u_k, v_k) whenever the quotient is no integer, so that every
    weight formed so far stays an integer at the same scale. s is then the least start of integral weights."""
    scale = weight = 1
    observed = 1 if a == low else 0
    for k in range(low, high):
        up, down = (r1 - k) * (c1 - k), (k + 1) * (r2 - c1 + k + 1)
        product = _product(budget, weight, up)
        bits = product.bit_length()
        budget.tick(2 + 2 * division_work(bits + 64, down.bit_length()) + _linear(bits))
        common = gcd(product % down, down)  # gcd(W_k u_k, v_k), from a remainder of the long product.
        factor = down // common
        if factor > 1:
            scale = _product(budget, scale, factor)
            if observed:
                observed = _product(budget, observed, factor)
        weight = product // common  # W_k u_k factor / v_k, exact.
        if k + 1 == a:
            observed = weight
    return scale, observed


# ---------------------------------------------------------------------------------------------------------------
# Charged integer steps, on the scale of the cost model of trusted.integers

def _linear(bits: int) -> int:
    """The units of a linear pass over integers of this many bits in all: one per 256 64-bit words."""
    return bits >> 14


def _product(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs."""
    left_bits, right_bits = left.bit_length(), right.bit_length()
    budget.tick(1 + product_work(left_bits, right_bits) + _linear(left_bits + right_bits))
    return left * right


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: one, a division of the longer by the shorter with a pass
    over the longer, then Lehmer's algorithm on the shorter, quadratic, one unit per 64 squared 64-bit words as
    trusted.upoly.charge_gcd charges it."""
    small, large = min(left_bits, right_bits), max(left_bits, right_bits)
    words = small >> 6
    return 1 + division_work(max(large - small, 0) + 64, small) + (large >> 12) + (words * words >> 6)
