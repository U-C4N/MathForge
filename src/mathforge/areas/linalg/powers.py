"""Powers of exact rational matrices (mathforge.linalg): matrix_power, a square matrix to a nonnegative integer power.

matrix_power(M, k) -> Matrix
    Domain: M is a Matrix of exact rationals (MCP kind "matrix": operations_v6.run already rejects anything but a
    Matrix as invalid input) that is square, n x n with n >= 0, the 0 x 0 matrix included; k is an exact int >= 0
    (kind "int"). A non-square M is invalid input, and so is k < 0 ("use inverse for negative powers").
    Value: M**k as a Matrix of canonical Rationals (n rows of n entries, ncols = n), the identity of size n for
    k = 0, also for a singular M: [[1, 1], [1, 0]]**10 = [[89, 55], [55, 34]], [[2, 0], [0, 3]]**0 = I,
    [[0, 1], [0, 0]]**2 = 0 and [[1/2, 0], [0, 2]]**3 = [[1/8, 0], [0, 8]].
    Evidence: (). The checker, mathforge.certify.linalg.powers, scales the claim by s**k back to integers, bounds
    them by B = ||M_int||_inf**k and recomputes M_int**k modulo word primes whose product exceeds 2B with another
    addition chain (an N-version check: no certificate is cheaper than the power itself).
    Resource limits: current_budget().matrix(rows, cols) is charged before any validation, so that a 65 x 65 M is
    resource_limit_exceeded. For k >= 1, so is a common denominator s of M longer than max_integer_bits, and a bound
    B = ||M_int||_inf**k (the maximum absolute row sum of M_int = s M, to the k-th power) past 2**max_integer_bits,
    the card's "k * log2(||M_int||_inf) exceeds max_integer_bits" word for word: B = 2**max_integer_bits itself (a
    power of two N with k log2(N) = max_integer_bits, such as [[1, 1], [1, 0]]**65536) is computed; k = 0 needs
    neither s nor B. The checker raises ResourceLimitError on exactly these
    requests (and on request integers past max_integer_bits, which operations_v6.run refuses before produce()),
    before it judges any claimed integer. Its work is about that of the same chain of matrix products, plus
    EXTRA_PRIME_WORK = 1,200 units for each prime that P > 2B takes beyond the 64 of trusted.primes.MODULI (from
    about 3,900 bits of B on; 1,011 of them, 1.2 million units, for B close to 2**65536).

How produce() computes (docs/v09/cards/m4-linalg-geometry-stats-discrete.md). Nothing here is a proof: the checker
verifies the value with an addition chain of its own, modulo primes.

- It charges one unit for the call and max_matrix_entries for M (current_budget().matrix) before it validates
  anything but the class of M, then validates k (an exact int, within max_integer_bits, k >= 0), the shape of M and
  its entries: exact Rationals with int parts and a positive denominator, which every Matrix built by its constructor
  holds (only a Matrix forged without it can fail here). Lowest terms are not needed: s M below is an integer matrix
  for any common multiple s of the denominators.
- k = 0 gives the identity of size n, n = 0 the 0 x 0 matrix, without any arithmetic: k = 0 needs neither s nor B.
- Otherwise s, the least common multiple of the denominators of M, grown one denominator at a time and refused once it
  is longer than max_integer_bits (the checker's refusal); M_int = s M; N = ||M_int||_inf, the largest absolute row
  sum; and the card's limit on B = N**k, decided from bits(N) alone when k (bits(N) - 1) > max_integer_bits, or
  equals it for an N that is no power of two, else from N**k itself (then at most 2 max_integer_bits bits long).
  The refusals hold for k = 1 as well, which then answers M itself, each entry in lowest terms: the division of s M
  by s would take a gcd as long as s per entry.
- For k >= 2, M_int**k by repeated squaring over the integers, right-to-left binary: from the lowest bit of k up, the
  base runs through M_int**(2**i), squared once for each bit after the lowest; the lowest set bit takes it as the
  result and every later set bit multiplies it in: bits(k) - 1 squarings and popcount(k) - 1 multiplications. The
  checker's chain is left-to-right binary, a different one. Every matrix of the chain is a power M_int**j with
  1 <= j <= k, whose entries lie in [-B, B]: no integer of the chain is longer than max_integer_bits + 1 bits, and
  only an entry of +-2**max_integer_bits, which B = 2**max_integer_bits allows ([[2]]**65536), has that length: such
  a true value is refused (ResourceLimitError), as the checker answers False on it.
- Then the division by s**k: each entry w of W = M_int**k becomes (w / g) / (s**k / g) with g = gcd(w, s**k). A
  nonzero w has the denominator s**k / g >= s**k / |w| >= s**k / B, so once 2**(k (bits(s) - 1)) <= s**k reaches
  2**max_integer_bits * B, no entry but 0 fits max_integer_bits: ResourceLimitError before s**k is formed, which keeps
  s**k shorter than 4 max_integer_bits + 2 bits otherwise. A denominator still longer than max_integer_bits is refused
  entry by entry: [[1/3]]**41349 = [[1/3**41349]] has one of 65,537 bits, a true value that the checker answers False.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call, and charges every step before it runs, on the scale of the cost model of
trusted.integers; every valid input costs at least two units. A product of two n x n integer matrices with entries of
at most a and b bits costs 1 + 2 n**2 + n**3 (1 + product_work(a, b)), the checker's charge for one without its
reductions modulo P. N**k and s**k come from a square-and-multiply loop that charges each of its products. A gcd
costs one unit, a first remainder by the length of its quotient and one unit per 64 squared words of the shorter
operand; an exact division costs the length of its quotient (division_work of its words by the divisor's).

Both chains take bits(k) - 1 squarings and popcount(k) - 1 multiplications; the checker's multiplications take M_int,
the short factor, but it also pays its reductions and EXTRA_PRIME_WORK per prime past its table. Measured on 900
seeded powers (n up to 16, k up to 2**15) and on integer matrices near the default budget (n from 8 to 64),
produce() charged less than the checker on every integer matrix and for every k = 1. On small rational matrices
whose s is far longer than the denominators of M**k it charged up to about 1.8 times the checker's work, since its
gcd(w, s**k) is longer than the checker's gcd of the reduced entry. Measured with CPython on one machine, a unit takes
0.02 to 0.6 microseconds, the most for entries of one to four thousand bits, where product_work charges less than
CPython's product takes: the default max_work of 2,000,000 lasts at most about 1.2 seconds.

The family defines no record class: its only value is a Matrix (mathforge.matrices), which the shared renderer
presents, so TYPES is empty and there is no RENDER. The text of [[1/8, 0], [0, 8]] is "[[1/8, 0], [0, 8]]", its LaTeX
"\\begin{bmatrix}\\frac{1}{8} & 0 \\\\ 0 & 8\\end{bmatrix}", and the 0 x 0 matrix renders as "Matrix(0x0)".
Matrix.__pow__ (M ** k) returns produce()'s value without the independent check, unchecked arithmetic like M @ M.
mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/linalg/test_powers.py is the definition of done.
"""

from __future__ import annotations

from math import gcd
from operator import mul

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

# The scope text of every spec of wave M4 (docs/v09/cards/m4-linalg-geometry-stats-discrete.md).
SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

# The matrix is M, never A: an argument named A would be copied into Result.expression (operations_v6.source_context).
SPECS = (
    OperationSpec("matrix_power", "linalg",
                  (Argument("M", "Matrix", None, "matrix"), Argument("k", "int", None, "int")),
                  "Matrix", "Raise a square rational matrix to a nonnegative integer power.", SCOPE, version=6),
)

TYPES: dict[str, type] = {}  # The value is the v0.7 record mathforge.matrices.Matrix.
CHECKER = "mathforge.certify.linalg.powers"

Pair = tuple[int, int]  # The numerator and the positive denominator of an entry of M.
IntRows = list[list[int]]

_ZERO = trusted_rational(0, 1)
_NOT_A_MATRIX = "matrix_power requires a Matrix M of exact rationals"
_BOUND = "Computation limit exceeded: integer bits (the bound ||M_int||_inf**k)"
_COMMON = "Computation limit exceeded: integer bits (the common denominator of M)"
_DENOMINATOR = "Computation limit exceeded: integer bits (a denominator of M**k)"
_NUMERATOR = "Computation limit exceeded: integer bits (an entry of M**k)"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(M**k, ()) for args = {"M": M, "k": k}: the common denominator s of M, M_int = s M, repeated squaring over the
    integers (right-to-left binary) and the division by s**k, as the module states. InvalidInput for an M that is no
    square Matrix of exact rationals, a k that is no exact int and k < 0; ResourceLimitError past the budget, whose
    max_matrix_entries is charged before anything but the class of M is validated."""
    if type(operation) is not str or operation != "matrix_power":
        raise UnsupportedOperation(f"{operation!r} is not an operation of the powers family")
    if type(args) is not dict or set(args) != {"M", "k"}:
        raise InvalidInput("matrix_power takes exactly the arguments M and k")
    matrix, k = args["M"], args["k"]
    with computation() as budget:
        budget.tick()  # The call itself; every valid input charges at least one unit more.
        if type(matrix) is not Matrix:  # A subclass, the rows alone or any other object is no Matrix.
            raise InvalidInput(_NOT_A_MATRIX)
        rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
        if type(rows) is not tuple or type(ncols) is not int:
            raise InvalidInput(_NOT_A_MATRIX)
        budget.matrix(len(rows), ncols)  # First: a 65 x 65 M exceeds the budget before its shape is judged.
        if type(k) is not int:  # Neither a bool nor a float nor a str is an exact integer.
            raise InvalidInput("matrix_power requires an exact integer k")
        budget.check_int(k)
        n = len(rows)
        if ncols != n:
            raise InvalidInput(f"matrix_power requires a square matrix M, not {n} x {ncols}")
        if k < 0:
            raise InvalidInput("matrix_power requires k >= 0: use inverse for negative powers")
        pairs = _entries(budget, rows, n)
        if k == 0 or n == 0:  # M**0 = I, also for a singular M; the 0 x 0 matrix is its own every power.
            budget.tick(1 + n * n)
            return Matrix.identity(n), ()
        limit = budget.limits.max_integer_bits
        common = _common_denominator(budget, pairs, limit)
        int_rows, norm = _integer_rows(budget, pairs, common)
        bound_bits = _bound_bits(budget, norm, k, limit)  # Refused past the card's bound, also for k = 1.
        if k == 1:  # M itself: W / s would take a gcd as long as s for every entry, which the checker never pays.
            return _lowest_terms(budget, pairs, n), ()
        return _value(budget, _power(budget, int_rows, k), common, k, bound_bits, limit), ()


def _entries(budget: Budget, rows: tuple, n: int) -> list[list[Pair]]:
    """The entries of the square M as (numerator, denominator) pairs. InvalidInput for a row that is no tuple of n
    entries and an entry that is no exact Rational with int parts and a positive denominator, which only a Matrix
    forged without its constructor holds; ResourceLimitError for an integer longer than max_integer_bits."""
    pairs = []
    for row in rows:
        budget.tick(1 + n)
        if type(row) is not tuple or len(row) != n:
            raise InvalidInput(_NOT_A_MATRIX)
        items = []
        for entry in row:
            if type(entry) is not Rational:
                raise InvalidInput(_NOT_A_MATRIX)
            numerator = getattr(entry, "numerator", None)  # A forged Rational may lack a slot.
            denominator = getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int or denominator < 1:
                raise InvalidInput(_NOT_A_MATRIX)
            budget.check_int(numerator)
            budget.check_int(denominator)
            items.append((numerator, denominator))
        pairs.append(items)
    return pairs


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient: division_work of the quotient's words
    (one more for a short quotient) by the divisor's."""
    from ...trusted.integers import division_work

    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd (Lehmer's algorithm) of integers of the given lengths: one, a first remainder of the
    longer by the shorter one (by its quotient's length), then one unit per 64 squared 64-bit words of the shorter."""
    longer, shorter = max(left_bits, right_bits), min(left_bits, right_bits)
    words = shorter >> 6
    return 1 + _quotient_work(longer, shorter) + (words * words >> 6)


def _integer_power(budget: Budget, base: int, exponent: int) -> int:
    """base**exponent for ints base >= 2 and exponent >= 1 by right-to-left binary square-and-multiply, each product
    charged before it runs; the caller bounds the length of the result first, which bounds every square too."""
    from ...trusted.integers import product_work

    result = 1
    while True:
        if exponent & 1:
            budget.tick(1 + product_work(result.bit_length(), base.bit_length()))
            result *= base
        exponent >>= 1
        if not exponent:
            return result
        bits = base.bit_length()
        budget.tick(1 + product_work(bits, bits))
        base *= base


def _common_denominator(budget: Budget, pairs: list[list[Pair]], limit: int) -> int:
    """s, the least common multiple of the denominators of M, grown one denominator at a time. ResourceLimitError once
    it is longer than limit bits, before an lcm much longer is formed: a step adds at most the length of one
    denominator, itself within limit bits."""
    from ...trusted.integers import product_work

    common = 1
    for row in pairs:
        budget.tick(1 + len(row))
        for _, denominator in row:
            if denominator == 1:
                continue
            common_bits, denominator_bits = common.bit_length(), denominator.bit_length()
            budget.tick(1 + _quotient_work(common_bits, denominator_bits))
            if not common % denominator:
                continue
            budget.tick(_gcd_work(common_bits, denominator_bits) + 2 + _quotient_work(common_bits, denominator_bits)
                        + product_work(common_bits, denominator_bits))
            common = common // gcd(common, denominator) * denominator
            if common.bit_length() > limit:
                raise ResourceLimitError(_COMMON)
    return common


def _integer_rows(budget: Budget, pairs: list[list[Pair]], common: int) -> tuple[IntRows, int]:
    """M_int = s M as integer rows (each entry p (s / q), at most twice max_integer_bits long), and N = ||M_int||_inf,
    its largest absolute row sum."""
    from ...trusted.integers import product_work

    common_bits = common.bit_length()
    int_rows, norm = [], 0
    for row in pairs:
        items = []
        longest = 0
        for numerator, denominator in row:
            budget.tick(2 + _quotient_work(common_bits, denominator.bit_length())
                        + product_work(common_bits, numerator.bit_length()))
            entry = numerator * (common // denominator)
            longest = max(longest, entry.bit_length())
            items.append(entry)
        budget.tick(1 + len(items) * (1 + (longest >> 12)))  # The absolute row sum: linear in the lengths.
        norm = max(norm, sum(map(abs, items)))
        int_rows.append(items)
    return int_rows, norm


def _bound_bits(budget: Budget, norm: int, k: int, limit: int) -> int:
    """The length of B = N**k for k >= 1, the card's bound on every entry of M_int**k. ResourceLimitError when
    k log2(N) exceeds limit, the card's words, that is when B > 2**limit; B = 2**limit itself is allowed. Decided at
    once when 2**(k (bits(N) - 1)) <= B shows it (equality holds exactly for a power of two N), else once N**k, of
    fewer than k bits(N) <= 2 k (bits(N) - 1) <= 2 limit bits, is formed."""
    if norm <= 1:
        return norm  # 0**k = 0 has no bits and 1**k = 1 one.
    floor_bits = k * (norm.bit_length() - 1)  # 2**floor_bits <= N**k, equal exactly when N is a power of two.
    if floor_bits > limit or (floor_bits == limit and norm & (norm - 1)):
        raise ResourceLimitError(_BOUND)
    bound = _integer_power(budget, norm, k)
    if bound > 1 << limit:
        raise ResourceLimitError(_BOUND)
    return bound.bit_length()


def _longest(rows: IntRows) -> int:
    """The greatest bit length of the entries of integer rows, 0 for none."""
    return max((entry.bit_length() for row in rows for entry in row), default=0)


def _product(budget: Budget, left: IntRows, right: IntRows) -> IntRows:
    """The product of two n x n integer matrices, charged before it runs: 2 n**2 for the lengths of the entries, then
    n**3 products of at most the longest lengths on each side, with the additions."""
    from ...trusted.integers import product_work

    n = len(left)
    budget.tick(1 + 2 * n * n)
    left_bits, right_bits = _longest(left), _longest(right)
    budget.tick(n * n * n * (1 + product_work(left_bits, right_bits)))
    columns = list(zip(*right))
    return [[sum(map(mul, row, column)) for column in columns] for row in left]


def _power(budget: Budget, base: IntRows, k: int) -> IntRows:
    """M_int**k for k >= 1 by right-to-left binary repeated squaring: from the lowest bit of k up, base runs through
    M_int**(2**i), squared once for each bit after the lowest; the lowest set bit takes it as the result, and every
    later set bit multiplies it into the result."""
    digits = bin(k)[:1:-1]  # The binary digits of k, the lowest first; the last one is the leading 1.
    lowest = digits.index("1")
    for _ in range(lowest):
        base = _product(budget, base, base)
    result = base
    for digit in digits[lowest + 1:]:
        base = _product(budget, base, base)
        if digit == "1":
            result = _product(budget, result, base)
    return result


def _lowest_terms(budget: Budget, pairs: list[list[Pair]], n: int) -> Matrix:
    """The Matrix of the entries of M, each in lowest terms: M**1. A Matrix built by its constructor holds them so
    already, and only one forged without it holds an entry such as 2/4."""
    rows = []
    for row in pairs:
        budget.tick(1 + len(row))
        items = []
        for numerator, denominator in row:
            if denominator != 1:
                budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
                divisor = gcd(numerator, denominator)
                if divisor != 1:
                    divisor_bits = divisor.bit_length()
                    budget.tick(2 + _quotient_work(numerator.bit_length(), divisor_bits)
                                + _quotient_work(denominator.bit_length(), divisor_bits))
                    numerator, denominator = numerator // divisor, denominator // divisor
            items.append(trusted_rational(numerator, denominator))
        rows.append(tuple(items))
    return Matrix(tuple(rows), ncols=n)


def _value(budget: Budget, power: IntRows, common: int, k: int, bound_bits: int, limit: int) -> Matrix:
    """The Matrix W / s**k for W = M_int**k, each entry in lowest terms. ResourceLimitError for a denominator longer
    than limit bits: at once, before s**k is formed, when 2**(k (bits(s) - 1)) <= s**k reaches 2**(limit + bits(B)),
    since every nonzero entry w then has the denominator s**k / gcd(w, s**k) >= s**k / |w| > 2**limit; otherwise entry
    by entry. W = 0 needs no division and fits whatever s**k is. A numerator longer than limit bits is refused too:
    |w| <= B <= 2**limit leaves only +-2**limit, from B = 2**limit ([[2]]**65536 = [[2**65536]] and
    [[4/3]]**32768 = [[2**65536 / 3**32768]])."""
    n = len(power)
    budget.tick(1 + n * n)
    if common == 1 and bound_bits > limit and _longest(power) > limit:  # An entry +-2**limit: no value holds it.
        raise ResourceLimitError(_NUMERATOR)
    if common == 1 or not any(map(any, power)):  # Integers, or the zero matrix: nothing to divide.
        return Matrix(tuple(tuple(trusted_rational(entry, 1) for entry in row) for row in power), ncols=n)
    common_bits = common.bit_length()
    if k * (common_bits - 1) >= limit + bound_bits:
        raise ResourceLimitError(_DENOMINATOR)
    scale = _integer_power(budget, common, k)  # Shorter than k bits(s) < limit + bound_bits + k <= 4 limit + 1 bits.
    scale_bits = scale.bit_length()
    rows = []
    for row in power:
        budget.tick(1 + len(row))
        items = []
        for entry in row:
            if not entry:
                items.append(_ZERO)
                continue
            entry_bits = entry.bit_length()
            budget.tick(_gcd_work(scale_bits, entry_bits))
            divisor = gcd(entry, scale)
            divisor_bits = divisor.bit_length()
            budget.tick(2 + _quotient_work(entry_bits, divisor_bits) + _quotient_work(scale_bits, divisor_bits))
            numerator, denominator = entry // divisor, scale // divisor
            if denominator.bit_length() > limit:
                raise ResourceLimitError(_DENOMINATOR)
            if numerator.bit_length() > limit:  # Only +-2**limit, when B = 2**limit: [[4/3]]**32768.
                raise ResourceLimitError(_NUMERATOR)
            items.append(trusted_rational(numerator, denominator))  # Lowest terms, positive denominator.
        rows.append(tuple(items))
    return Matrix(tuple(rows), ncols=n)
