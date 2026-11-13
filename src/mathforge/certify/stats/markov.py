"""Independent checker of the family mathforge.areas.stats.markov: stationary_distribution.

The checker never solves the chain. It verifies the claimed distribution by exact integer identities, and its
uniqueness by a rank certificate: one nonzero minor of order n - 1, decided by elimination modulo a proved prime.

The theorem that makes True imply the claim:

stationary_distribution(M) = pi, evidence (rows, cols, prime).
    M is an n x n Matrix of canonical Rationals with n >= 1. Let s >= 1 be the least common multiple of the
    denominators of M's entries, so that S = s M is an integer matrix, and K = s (I - M) = s I - S. The checker
    verifies:
    1. M is row stochastic: every entry of S is >= 0 and every row of S sums to s (each row of M sums to 1).
    2. pi is a tuple of n canonical Rationals p_j / q_j. With t the least common multiple of the q_j and the integers
       P_j = p_j (t / q_j) = t pi_j: every P_j >= 0 (pi >= 0), sum_j P_j = t (sum pi = 1), and for every column j,
       sum_i P_i S_ij = s P_j, which is (pi M)_j = pi_j multiplied by the positive integer s t: pi M = pi.
    3. prime is proved prime by mathforge.certify.ntheory.factorization.prime_proof(prime, ()), and rows and cols are
       tuples of n - 1 exact ints, each strictly increasing within 0 .. n - 1.
    4. The (n - 1) x (n - 1) submatrix K[rows, cols] has a determinant that is nonzero modulo prime: Gaussian
       elimination over the field Z/prime finds a nonzero pivot in every column. Each step replaces a row r by
       a r - b r_k with a, the pivot, nonzero modulo the prime, which multiplies the determinant by a unit, adds a
       multiple of another row (which keeps it) or swaps two rows (which negates it): the determinant modulo the prime
       is nonzero exactly when every column has a pivot. For n = 1 the submatrix is 0 x 0, whose determinant is 1.
    Then det K[rows, cols] is a nonzero integer, so K, and with it I - M = K / s, has rank at least n - 1. The left
    kernel {x : x (I - M) = 0} therefore has dimension at most 1; pi lies in it by 2 and is not zero (its entries sum
    to 1), so the kernel is the line through pi, and every stationary distribution x, which lies in it with
    sum x = 1, equals pi. So pi is the unique stationary distribution of M, and the claim holds. (Conversely, for a
    stochastic M with rank(I - M) = n - 1 some minor of order n - 1 of K is a nonzero integer, and a prime below
    3,317,044,064,679,887,385,961,981 that does not divide it exists: every true claim has evidence that passes.)

Shapes, and what a producer must build. The operation is the exact str "stationary_distribution", args a dict of
exactly the str key "M", and M a mathforge.matrices.Matrix (identified by (type(M).__module__, type(M).__qualname__)
and read as data: rows, a tuple of n tuples of n entries, and ncols, an exact int equal to n) whose entries are of the
exact class mathforge.model.Rational in lowest terms with a positive denominator (as the Matrix constructor makes
them; a forged non-canonical entry answers False). The value is a tuple (never a list) of exactly n Rationals of that
exact class, each in lowest terms with a positive denominator and integers within max_integer_bits: an int (even 1),
a bool, a float, a str, 2/4 for 1/2 and 0/2 answer False. The evidence is a tuple of exactly three items: rows and
cols, tuples (never lists) of exact ints, strictly increasing, of length n - 1 and within 0 .. n - 1 (empty for
n = 1), and prime, an exact int that prime_proof(prime, ()) proves, which requires 2 <= prime <
3,317,044,064,679,887,385,961,981 (from there on prime_proof needs a Pocklington certificate, for which this layout
has no slot). The determinant is that of s (I - M) with s the LEAST common multiple of the denominators: another
multiple k s scales the minor by k**(n - 1), which a prime dividing k would annihilate. Anything else answers False,
never an exception.

Resource limits. ResourceLimitError leaves a claim unverified, never verified. It is raised for an M with more entries
than max_matrix_entries (budget.matrix, before the shape is read, as produce() does), for a request integer over
max_integer_bits (as the request's inspection raises before a checker runs), for a common denominator s of M longer
than max_integer_bits (it grows one denominator at a time, so none much longer is formed), and for work past max_work.
A claimed integer over max_integer_bits answers False, as no computation under the budget forms it, and so does a
claim whose common denominator t is longer than max_integer_bits: for the true pi, with v the primitive integer
vector of the kernel line (v >= 0, gcd 1) and S = sum v, pi_j = v_j / S, and the least common multiple of the reduced
denominators S / gcd(v_j, S) is S / gcd(v, S) = S, the integer that normalizes the distribution, which a computation
under the budget cannot hold past max_integer_bits either.

Budget. Every loop and every long integer operation is charged before it runs, on the scale of the cost model of
trusted.integers: product_work for a product, division_work by the length of the quotient for a long division, a gcd
by a first remainder and then quadratically in the shorter operand (as trusted.upoly.charge_gcd), one unit per entry
read and one unit per 256 words of a linear pass. No builtin pow runs here (prime_proof charges its own). The work is
about n**2 products for the stationarity identities and (n - 1)**3 / 3 short products modulo the prime for the
elimination: a 64 x 64 chain with small entries costs a few hundred thousand units of the default 2,000,000.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.certify.ntheory.factorization import prime_proof
from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work

OPERATION = "stationary_distribution"
MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of the class of M.

Pair = tuple[int, int]  # The numerator and the positive denominator of a canonical Rational.


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the unique stationary distribution of the chain args["M"], by the theorem of
    the module. False for anything but the operation's exact shapes and for a claim that fails a step;
    ResourceLimitError for a request past the limits that the module lists and for work past the budget."""
    if type(operation) is not str or operation != OPERATION:
        return False
    if type(args) is not dict or len(args) != 1 or any(type(name) is not str for name in args) or "M" not in args:
        return False
    if type(evidence) is not tuple or len(evidence) != 3:
        return False
    rows, cols, prime = evidence
    if type(rows) is not tuple or type(cols) is not tuple or type(prime) is not int:
        return False
    with computation() as budget:  # One budget for every step: the active one, or one of the default limits.
        budget.tick()
        entries = _request(budget, args["M"])
        if entries is None or not entries:  # A chain without states has no distribution.
            return False
        n = len(entries)
        claim = _distribution(budget, value, n)
        if claim is None:
            return False
        if not _indices(budget, rows, n) or not _indices(budget, cols, n):
            return False
        if not prime_proof(prime, ()):  # Charged inside; a prime over its bound or a composite answers False.
            return False
        common = _common_denominator(budget, [pair for row in entries for pair in row])
        if common is None:
            raise ResourceLimitError("Computation limit exceeded: common denominator bits")
        scaled = _scaled_rows(budget, entries, common)
        if not _stochastic(budget, scaled, common):
            return False
        if not _stationary(budget, scaled, common, claim):
            return False
        return _minor_nonzero(budget, scaled, common, rows, cols, prime)


# ---------------------------------------------------------------------------------------------------------------
# Shapes

def _named(value: object, name: tuple[str, str]) -> bool:
    """Whether the value's class has this module and name, both exact strings (a class may set __module__ to any
    object, whose comparison could answer anything)."""
    cls = type(value)
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    return type(module) is str and type(qualname) is str and module == name[0] and qualname == name[1]


def _request(budget: Budget, matrix: object) -> list[list[Pair]] | None:
    """The entries of M as (numerator, denominator) pairs when M is a square Matrix of canonical Rationals, read as
    data; None otherwise. budget.matrix(rows, cols) comes first, and a request integer over max_integer_bits raises
    ResourceLimitError, as the request's inspection does."""
    if not _named(matrix, MATRIX):
        return None
    rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
    if type(rows) is not tuple or type(ncols) is not int:
        return None
    budget.matrix(len(rows), ncols)  # Past max_matrix_entries: ResourceLimitError, before the shape is read.
    n = len(rows)
    if ncols != n:
        return None
    result = []
    for row in rows:
        budget.tick(1 + n)
        if type(row) is not tuple or len(row) != n:
            return None
        items = []
        for entry in row:
            if type(entry) is not Rational:
                return None
            numerator, denominator = getattr(entry, "numerator", None), getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int:
                return None
            budget.check_int(numerator)
            budget.check_int(denominator)
            if denominator < 1 or not _lowest_terms(budget, numerator, denominator):
                return None
            items.append((numerator, denominator))
        result.append(items)
    return result


def _distribution(budget: Budget, value: object, n: int) -> list[Pair] | None:
    """The claimed pi as (numerator, denominator) pairs when it is a tuple of n canonical Rationals with integers
    within max_integer_bits; None for anything else (a list, an int, a Rational subclass or one forged out of form)."""
    if type(value) is not tuple or len(value) != n:
        return None
    limit = budget.limits.max_integer_bits
    pairs = []
    for entry in value:
        budget.tick()
        if type(entry) is not Rational:
            return None
        numerator, denominator = getattr(entry, "numerator", None), getattr(entry, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            return None
        if numerator.bit_length() > limit or denominator.bit_length() > limit:
            return None  # No computation under the budget forms it.
        pairs.append((numerator, denominator))
    for numerator, denominator in pairs:  # The gcds once every length is known to be bounded.
        if not _lowest_terms(budget, numerator, denominator):
            return None
    return pairs


def _lowest_terms(budget: Budget, numerator: int, denominator: int) -> bool:
    """Whether numerator / denominator, with a positive denominator, is in lowest terms: 0 only as 0/1."""
    budget.tick()
    if denominator == 1:
        return True
    budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
    return gcd(numerator, denominator) == 1


def _indices(budget: Budget, indices: tuple[object, ...], n: int) -> bool:
    """Whether the indices are n - 1 exact ints, strictly increasing within 0 .. n - 1."""
    if len(indices) != n - 1:
        return False
    previous = -1
    for index in indices:
        budget.tick()
        if type(index) is not int or index <= previous or index >= n:
            return False
        previous = index
    return True


# ---------------------------------------------------------------------------------------------------------------
# The identities

def _common_denominator(budget: Budget, pairs: list[Pair]) -> int | None:
    """The least common multiple of the denominators: D % d == 0 leaves D, else D * (d / gcd(d, D % d)). None once
    it is longer than max_integer_bits: the caller decides between ResourceLimitError and False."""
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
                return None
    return common


def _scaled(budget: Budget, numerator: int, denominator: int, common: int) -> int:
    """numerator * (common / denominator), for a common multiple of the denominator, charged before it runs."""
    if denominator == common:
        budget.tick()
        return numerator
    common_bits, bits = common.bit_length(), denominator.bit_length()
    budget.tick(2 + _quotient_work(common_bits, bits) + product_work(numerator.bit_length(), common_bits - bits + 1)
                + _linear(common_bits + numerator.bit_length()))
    return numerator * (common // denominator)


def _scaled_rows(budget: Budget, entries: list[list[Pair]], common: int) -> list[list[int]]:
    """S = s M as integer rows."""
    return [[_scaled(budget, numerator, denominator, common) for numerator, denominator in row] for row in entries]


def _stochastic(budget: Budget, scaled: list[list[int]], common: int) -> bool:
    """Whether every entry of S is >= 0 and every row of S sums to s: M is row stochastic."""
    for row in scaled:
        total = 0
        for entry in row:
            budget.tick(1 + _linear(entry.bit_length() + total.bit_length()))
            if entry < 0:
                return False
            total += entry
        if total != common:
            return False
    return True


def _stationary(budget: Budget, scaled: list[list[int]], common: int, claim: list[Pair]) -> bool:
    """Whether the claimed pi is >= 0, sums to 1 and satisfies pi M = pi, through the integers P_j = t pi_j: every
    P_j >= 0, sum P_j = t and sum_i P_i S_ij = s P_j for every column j."""
    if any(numerator < 0 for numerator, _ in claim):
        return False
    scale = _common_denominator(budget, claim)  # t.
    if scale is None:  # Past max_integer_bits: no computation under the budget forms a true pi so (module docstring).
        return False
    weights = [_scaled(budget, numerator, denominator, scale) for numerator, denominator in claim]  # P_j.
    total = 0
    for weight in weights:
        budget.tick(1 + _linear(weight.bit_length() + total.bit_length()))
        total += weight
    if total != scale:
        return False
    n = len(weights)
    for j in range(n):
        column = 0
        for i in range(n):
            entry = scaled[i][j]
            if not entry:
                budget.tick()
                continue
            column += _product(budget, weights[i], entry)
            budget.tick(_linear(column.bit_length()))
        if column != _product(budget, common, weights[j]):
            return False
    return True


def _minor_nonzero(budget: Budget, scaled: list[list[int]], common: int, rows: tuple[int, ...],
                   cols: tuple[int, ...], prime: int) -> bool:
    """Whether det K[rows, cols] is nonzero modulo the proved prime, K = s I - S: Gaussian elimination over Z/prime
    with a nonzero pivot in every column (the module states why); True for the 0 x 0 submatrix."""
    prime_bits = prime.bit_length()
    matrix = []
    for r in rows:
        row = []
        for c in cols:
            entry = (common if r == c else 0) - scaled[r][c]
            budget.tick(2 + _quotient_work(entry.bit_length(), prime_bits) + _linear(entry.bit_length()))
            row.append(entry % prime)
        matrix.append(row)
    size = len(matrix)
    # One update of an entry: two products of residues and a reduction of their difference modulo the prime.
    step = 3 + 2 * product_work(prime_bits, prime_bits) + _quotient_work(2 * prime_bits + 1, prime_bits)
    for k in range(size):
        budget.tick(1 + size - k)
        pivot_index = k
        while not matrix[pivot_index][k]:
            pivot_index += 1
            if pivot_index == size:
                return False  # A column without a pivot: the determinant is 0 modulo the prime.
        if pivot_index != k:
            matrix[k], matrix[pivot_index] = matrix[pivot_index], matrix[k]
        pivot_row = matrix[k]
        pivot = pivot_row[k]
        for i in range(k + 1, size):
            row = matrix[i]
            factor = row[k]
            budget.tick()
            if not factor:
                continue
            budget.tick((size - k) * step)
            matrix[i] = row[:k] + [(pivot * a - factor * b) % prime for a, b in zip(row[k:], pivot_row[k:])]
    return True


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
