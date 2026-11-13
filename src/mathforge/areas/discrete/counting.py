"""Counting numbers of exact nonnegative integers (mathforge.discrete): integer partitions, derangements, Bell numbers
and unsigned Stirling numbers of the first kind.

Every argument is an exact int (MCP kind and annotation "int", so operations_v6.run already rejects a bool, a float, a
str and None as invalid input), and no argument has a default. Every value is a plain int: the family defines no
record class, so TYPES is empty and there is no RENDER (the shared renderer presents ints). For each operation, its
domain, its value and the evidence that its checker, mathforge.certify.discrete.counting, verifies:

- partitions(n): p(n), the number of ways to write n as a sum of positive integers when the order of the parts does
  not count, for n >= 0; p(0) = 1. n < 0 is invalid input, n > 5000 a resource limit. Evidence (table,): the tuple
  table = (p(0), p(1), ..., p(n)) of n + 1 ints, so ((1,),) for n = 0.
- derangements(n): D(n), the number of permutations of n items without a fixed point, for n >= 0; D(0) = 1 and
  D(1) = 0. n < 0 is invalid input, n > 5900 a resource limit. Evidence ().
- bell(n): B(n), the number of partitions of a set of n items into nonempty blocks, for n >= 0; B(0) = 1. n < 0 is
  invalid input, n > 1500 a resource limit. Evidence ().
- stirling1(n, k): c(n, k), the unsigned Stirling number of the first kind: the number of permutations of n items with
  exactly k cycles, the coefficient of x**k in x(x + 1)...(x + n - 1), for n >= 0 and k >= 0; c(0, 0) = 1, c(n, 0) = 0
  for n >= 1, and c(n, k) = 0 for every k > n. n < 0 or k < 0 is invalid input, n > 1000 a resource limit. Evidence
  (row,): the tuple row = (c(n, 0), c(n, 1), ..., c(n, n)) of n + 1 ints, the full row for every k, also for k > n.

The resource limits of partitions and derangements are the card's (D(5911) is the first derangement number longer than
the default max_integer_bits). The card sets none for bell and stirling1; their checks cost about n products of up to
n * log2(n) bits and n(n + 1)/2 steps of synthetic division, and the checker verifies up to n = 1,854 and n = 1,042
under the default budget, so the limits above leave a margin and producer and checker fail at the same point. The
limits hold whatever the request's ComputationLimits are, as the card states them.

How produce() computes (docs/v09/cards/m4-linalg-geometry-stats-discrete.md). Nothing here is a proof: the checker
verifies every value with recurrences run from the claim, identities and evaluations of its own.

- Validation, before anything else: the operation is one of the four, args is a dict of exactly its argument names,
  each an exact int (type(x) is int: a bool, a float, a str, None or an int subclass is not), and none is negative:
  anything else is invalid input. Then an argument longer than max_integer_bits (k included, although every k > n
  has the value 0) and an n above the operation's limit are resource limits.
- partitions runs Euler's pentagonal recurrence, the card's producer,
      p(m) = sum over k >= 1 of (-1)**(k + 1) * (p(m - k(3k - 1)/2) + p(m - k(3k + 1)/2)),
  the terms with a negative index left out, for m = 1, ..., n in turn from p(0) = 1. It takes the generalized
  pentagonal numbers 1, 2, 5, 7, 12, 15, ... from the closed forms k(3k - 1)/2 and k(3k + 1)/2 (the checker generates
  them as sums of gaps of its own). The table of every p(m) is the evidence.
- derangements runs D(m) = (m - 1)(D(m - 1) + D(m - 2)) from D(0) = 1 and D(1) = 0 (the point 1 goes to one of the
  m - 1 others, j: the derangements in which j goes back to 1 derange the other m - 2 points, and the others become the
  derangements of the m - 1 points other than 1 once the point that goes to 1 goes to j instead), not the recurrence
  D(m) = m D(m - 1) + (-1)**m that the checker runs backwards.
- bell builds Bell's triangle (Aitken's array): row 0 is (1); each further row starts with the last entry of the row
  before and adds to its latest entry, in turn, each entry of the row before. Row r has r + 1 entries and ends with
  B(r + 1), so row n - 1 ends with B(n), and B(0) = 1. The checker uses no recurrence of the Bell numbers but the
  identity n! B(n) = sum_j C(n, j) j**n D(n - j).
- stirling1 multiplies out the rising factorial x(x + 1)...(x + n - 1) one linear factor at a time: row m + 1 is row m
  times x + m, c(m + 1, 0) = m c(m, 0), c(m + 1, i) = c(m, i - 1) + m c(m, i) and c(m + 1, m + 1) = 1, from row 0,
  (1,). Row n is the evidence, and c(n, k) the value (0 for k > n). The checker evaluates the row's polynomial at the
  points 1, 0, -1, ..., -(n - 1) instead.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call, and every operation charges at least one more on a
valid input; every further step is charged before it runs, by the lengths of the integers it will form, which the
integers before it bound. A pass over an integer, an addition or a product by a short integer, costs a unit and one
more per 8,192 bits: a term of the pentagonal recurrence and an addition of Bell's triangle are a pass each, a step of
the derangement recurrence and an entry of a Stirling row (a product by m and an addition) two. Measured with CPython
3.15 on one machine, a unit then took about 0.04 microseconds in the pentagonal recurrence, 0.15 in the derangement
recurrence (at every length up to 65,401 bits), 0.12 in Bell's triangle and 0.1 in the Stirling rows, within the
checker's scale of at most about 0.3. Budget.check_int holds against max_integer_bits, as soon as it is formed, an
integer that is at most the largest of the result: the newest partition or derangement number (both grow with m), the
last entry of each row of Bell's triangle (B(r + 1), at most B(n); no entry of a row exceeds its last), c(m, 1) =
(m - 1)! in each Stirling row (at most c(n, 1)) and the largest entry of the last row. So a request is refused with
ResourceLimitError when its result would hold an integer longer than max_integer_bits, which no Result may hold, and
also when its check would need a product that Budget.product refuses: operations_v6.run verifies in a fresh budget of
the request's own limits, not the defaults, and the checker forms n! B(n) for bell and n! for stirling1 through
Budget.product. produce() mirrors those calls (_bell_check_fits, and Budget.product((n - 1)!, n) for stirling1), so
producer and checker run out of integer bits at the same point. Under a lowered max_integer_bits that matters: with
1,000 bits B(n) fits up to n = 214 but its check only up to n = 107, so bell refuses n = 108 to 214 itself. For
partitions and derangements the checker calls no Budget.product: its integers are those of the claim, its quotients
and its sums, which are never held to max_integer_bits. Work is different: the checker spends about 1.15 times the
producer's work on partitions(5000), 5.5 times on derangements(5900) and 1.6 times on stirling1(1000, k) (0.7 times on
bell(1500)), so under a max_work between the two run() computes the value and then reports resource_limit_exceeded for
its verification, never a value, as stats.bivariate documents for its own checker; under the default max_work every
request up to the limits verifies. The charges depend on the request alone. Under the default max_work of 2,000,000
the requests at the limits take partitions(5000) about 385,000 units (0.013 s on that machine), derangements(5900)
about 50,000 (0.01 s), bell(1500) about 1,496,000 (0.2 s) and stirling1(1000, k) about 1,071,000 (0.1 s), whatever k
is.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, docs/v09/cards/m4-linalg-geometry-stats-discrete.md is the specification, and
tests/v09/discrete/test_counting.py is the definition of done.
"""

from __future__ import annotations

from collections.abc import Callable

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
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
_SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"
_N = Argument("n", "int", None, "int")

SPECS = (
    OperationSpec("partitions", "discrete", (_N,), "int",
                  "Count the integer partitions of a nonnegative integer.", _SCOPE, version=6),
    OperationSpec("derangements", "discrete", (_N,), "int",
                  "Count the permutations of n items with no fixed point.", _SCOPE, version=6),
    OperationSpec("bell", "discrete", (_N,), "int",
                  "Compute the Bell number B(n), the number of partitions of a set of n items.", _SCOPE, version=6),
    OperationSpec("stirling1", "discrete", (_N, Argument("k", "int", None, "int")), "int",
                  "Compute the unsigned Stirling number of the first kind c(n, k).", _SCOPE, version=6),
)

# No record classes: every value is an int, which serializes and renders without one.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.discrete.counting"

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
# The largest n of each operation; a larger n is a resource limit, whatever the request's limits (the docstring).
_LARGEST_N: dict[str, int] = {"partitions": 5000, "derangements": 5900, "bell": 1500, "stirling1": 1000}
_PASS_BITS = 13  # A pass over an integer costs a unit and one more per 2**13 = 8,192 bits (the Work paragraph).


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name and
    kind; the module states the algorithms and the evidence. InvalidInput for an argument that is no exact int or is
    negative, ResourceLimitError for an n above the operation's limit and past the budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the counting family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    values = [args[name] for name in names]
    if any(type(value) is not int for value in values):  # Neither a bool nor a float nor a str is an exact integer.
        raise InvalidInput(f"{operation} requires exact integers: {', '.join(names)}")
    if any(value < 0 for value in values):
        raise InvalidInput(f"{operation} requires {' and '.join(name + ' >= 0' for name in names)}")
    with computation() as budget:
        budget.tick()  # The call itself; on a valid input every operation charges at least one unit more.
        for value in values:
            budget.check_int(value)
        largest = _LARGEST_N[operation]
        if values[0] > largest:
            raise ResourceLimitError(f"Computation limit exceeded: {operation} takes n up to {largest}")
        return _PRODUCERS[operation](budget, *values)


def _pass(bits: int) -> int:
    """The units of one pass over an integer of at most the given bit length: an addition, or a product by a short
    integer."""
    return 1 + (bits >> _PASS_BITS)


def _partitions(budget: Budget, n: int) -> tuple[int, tuple]:
    """p(n) and the evidence (table,), table = (p(0), ..., p(n)), by Euler's pentagonal recurrence."""
    # The generalized pentagonal numbers up to n, increasing, with their signs: k(3k - 1)/2 and k(3k + 1)/2 have the
    # sign (-1)**(k + 1), and k(3k - 1)/2 < k(3k + 1)/2 < (k + 1)(3k + 2)/2.
    offsets: list[int] = []
    signs: list[int] = []
    k = 1
    while True:
        budget.tick()
        first = k * (3 * k - 1) // 2
        if first > n:
            break
        sign = 1 if k % 2 else -1
        offsets.append(first)
        signs.append(sign)
        if first + k <= n:  # k(3k + 1)/2 = k(3k - 1)/2 + k.
            offsets.append(first + k)
            signs.append(sign)
        k += 1
    table = [1]
    count = 0  # How many generalized pentagonal numbers are at most m: the terms of p(m).
    for m in range(1, n + 1):
        while count < len(offsets) and offsets[count] <= m:
            count += 1
        # count additions of entries no longer than p(m - 1), as p grows with m.
        budget.tick(1 + count * _pass(table[-1].bit_length() + count.bit_length()))
        total = 0
        for index in range(count):
            if signs[index] > 0:
                total += table[m - offsets[index]]
            else:
                total -= table[m - offsets[index]]
        budget.check_int(total)
        table.append(total)
    return table[n], (tuple(table),)


def _derangements(budget: Budget, n: int) -> tuple[int, tuple]:
    """D(n) and the evidence (), by D(m) = (m - 1)(D(m - 1) + D(m - 2)) from D(0) = 1 and D(1) = 0."""
    budget.tick()  # D(0) and D(1).
    if n == 0:
        return 1, ()
    previous, current = 1, 0  # D(m - 2) and D(m - 1), for m = 2 first.
    for m in range(2, n + 1):
        # An addition of D(m - 1) and D(m - 2) <= D(m - 1), then a product by m - 1: two passes.
        budget.tick(2 * _pass(current.bit_length() + m.bit_length() + 1))
        previous, current = current, (m - 1) * (current + previous)
        budget.check_int(current)
    return current, ()


def _bell(budget: Budget, n: int) -> tuple[int, tuple]:
    """B(n) and the evidence (), by Bell's triangle: row n - 1 ends with B(n)."""
    budget.tick()  # Row 0, (1,), whose last entry is B(1) = 1.
    if n == 0:
        return 1, ()
    row = [1]
    for r in range(1, n):
        # Row r from row r - 1: r additions. Its entries are at most its last, B(r + 1), which is at most (r + 1) B(r):
        # B(r), the last entry of row r - 1, plus its r entries, none above B(r).
        budget.tick(1 + r * _pass(row[-1].bit_length() + (r + 1).bit_length()))
        latest = row[-1]
        new = [latest]
        append = new.append
        for entry in row:
            latest += entry
            append(latest)
        budget.check_int(latest)  # The largest integer of the triangle so far.
        row = new
    if n >= 2:  # For n = 1 the check's one product is 1 * 1, which every max_integer_bits >= 1 admits.
        _bell_check_fits(budget, n, row[-1])
    return row[-1], ()


def _bell_check_fits(budget: Budget, n: int, value: int) -> None:
    """ResourceLimitError when the checker's products for bell(n) = value would not pass Budget.product under the
    active max_integer_bits. The verifier checks in a fresh budget of the same limits, so without this a value that
    fits would be computed and then fail its verification (operations_v6.run: "Independent verification exceeded its
    budget"); with it, producer and checker run out of integer bits at the same point."""
    # The checker forms n! B(n) with Budget.product(B(n), n!), which admits factors of up to max_integer_bits + 1
    # bits together: this mirrors that call. Its other products (the steps of n!, C(n, j) D(n - j), the powers j**k
    # for k <= n and the terms C(n, j) j**n D(n - j)) have values at most a term of its sum, a sum of nonnegative
    # terms equal to n! B(n), so each counts at most bits(term) + 1 bits. For 5 <= n <= 1500 no term has the bit
    # length of the sum (computed for every such n; the family's tests check n <= 160 and a sample), so each
    # counts at most bits(n! B(n)) <= bits(n!) + bits(B(n)) bits and passes whenever the mirrored call does. For
    # n <= 4 the checks were run at every max_integer_bits: the mirrored call is exact for n <= 3, and n = 4 needs one
    # bit more (9, for the square 16 * 16 of its term 4**4 = 256, against 8 for 15 * 24), which the factor 2 * 4! = 48
    # adds. n! is formed as the checker forms it, each step through Budget.product.
    factorial = 1
    for factor in range(2, n + 1):
        budget.product(factorial, factor)
        budget.tick(_pass(factorial.bit_length() + factor.bit_length()))
        factorial *= factor
    budget.product(value, 2 * factorial if n == 4 else factorial)


def _stirling1(budget: Budget, n: int, k: int) -> tuple[int, tuple]:
    """c(n, k) and the evidence (row,), row = (c(n, 0), ..., c(n, n)), by multiplying out x(x + 1)...(x + n - 1)."""
    budget.tick()  # Row 0, (1,).
    row = [1]
    for m in range(n):
        # Row m + 1 = row m times (x + m): m + 1 products by m and m additions. Its entries are >= 0 and sum to
        # (m + 1)!, so none is longer than (m + 1)! = (m + 1) m c(m, 1), as c(m, 1) = (m - 1)! for m >= 1.
        bits = row[1].bit_length() + m.bit_length() + (m + 1).bit_length() if m else 1
        budget.tick(1 + 2 * (m + 1) * _pass(bits))
        row = [m * row[0], *[low + m * high for low, high in zip(row, row[1:])], 1]
        budget.check_int(row[1])  # c(m + 1, 1) = m!: at most the largest entry, of this row and of every later one.
    if n:
        budget.check_int(max(row))  # The largest integer of the result.
    if n >= 2:
        # The checker forms n! = 2 * 3 * ... * n, each step through Budget.product, the last and longest from
        # (n - 1)! = c(n, 1) and n; n! may count a few bits more than the longest entry of the row. The verifier
        # checks in a fresh budget of the same limits, so this mirrored call makes production fail where
        # verification would run out of integer bits, rather than leave run() to report it after the work.
        budget.product(row[1], n)
    return (row[k] if k <= n else 0), (tuple(row),)


_PRODUCERS: dict[str, Callable[..., tuple[int, tuple]]] = {
    "partitions": _partitions,
    "derangements": _derangements,
    "bell": _bell,
    "stirling1": _stirling1,
}
