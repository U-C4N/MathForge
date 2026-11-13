"""Independent checker of the family mathforge.areas.linalg.powers: matrix_power.

The checker never reruns the producer's algorithm, repeated squaring over the integers from the low bit of k up
(right-to-left binary). It scales the claimed matrix back to integers and recomputes the power modulo word primes with
the other addition chain, left-to-right binary, as the card prescribes: an N-version check, since no certificate of a
matrix power is cheaper than the power itself. The theorem that makes True imply the claim:

matrix_power(M, k) = V, evidence ().
    M is an n x n matrix of exact rationals, n >= 0, k >= 0 is an exact int and V is an n x n Matrix of canonical
    Rationals. For k = 0, V is the identity of size n: M**0 = I for every square M, singular or not, and the 0 x 0
    matrix is its own identity.
    For k >= 1, let s >= 1 be the least common multiple of the denominators of M, M_int = s M (an integer matrix),
    N = ||M_int||_inf = max_i sum_j |M_int[i][j]| and B = N**k. The induced infinity norm is submultiplicative and
    bounds every entry of its matrix, so every entry of M_int**k = s**k M**k lies in [-B, B]. The checker verifies
    that W = s**k V is an integer matrix with every entry in [-B, B], and that W = M_int**k modulo P, the product of
    the first primes of trusted.primes.moduli() with P > 2B: modulo a product of distinct primes a congruence holds
    exactly when it holds modulo each of them (Chinese remainder theorem). Then every entry of W - M_int**k is a
    multiple of P of absolute value at most 2B < P, hence 0: W = M_int**k, and V = W / s**k = M**k.
    The power modulo P is computed by left-to-right binary square-and-multiply on balanced residues in (-P/2, P/2),
    where an integer of [-B, B] is its own residue, so W and the residues compare as integers. Every intermediate
    matrix of the chain is a power M_int**j with 1 <= j <= k, whose entries lie in [-B, B] too: the reductions never
    change a value, and the chain costs what the exact power costs, while the proof rests on the residues alone.

Shapes. The operation must be the str "matrix_power", args a dict of exactly M and k, the evidence the empty tuple.
M and the value are identified by (type(x).__module__, type(x).__qualname__) == ("mathforge.matrices", "Matrix") and
read as data: rows, a tuple of n tuples of n entries, and ncols, an exact int equal to n. Every entry is of the exact
class mathforge.model.Rational with exact int numerator and denominator and a positive denominator; the value's
entries must also be in lowest terms, with integers within max_integer_bits (M's entries need not be in lowest terms:
s M is an integer matrix for any positive common multiple s of their denominators, and every step below holds for
it). k is an exact int >= 0. A bool, a float, a str, an int or Rational
subclass, a list for a tuple, a non-square M, a negative k, a value of another shape and any evidence but () answer
False, never an exception. A value integer over max_integer_bits answers False, as no computation under the budget
forms it, and so does a non-canonical value (an int entry, 2/4 for 1/2, 0/2).

Resource limits. ResourceLimitError leaves a claim unverified, never verified. Before any claimed integer is judged,
it is raised for: an M with more entries than max_matrix_entries (budget.matrix, which produce() calls first); a
request integer (in M, or k) over max_integer_bits, as the request's inspection raises before a checker runs; and, for
k >= 1 only, a common denominator s longer than max_integer_bits (as trusted.intmatrix refuses a row scale of that
length) and the card's limit, "k * log2(||M_int||_inf) exceeds max_integer_bits", taken word for word as
B > 2**max_integer_bits: B = 2**max_integer_bits itself, a power of two N with k log2(N) = max_integer_bits, is
checked ([[1, 1], [1, 0]]**65536 holds integers of about 45,500 bits). Within these limits every entry of a true
value has integers within max_integer_bits, except denominators that only a power s**k past max_integer_bits allows
and a numerator +-2**max_integer_bits that only B = 2**max_integer_bits allows ([[2]]**65536); the producer refuses
such a value (ResourceLimitError), as it cannot be held under the budget.

Budget. Every loop and every long integer operation is charged before it runs, on the scale of the cost model of
trusted.integers: product_work for a product, division_work by the length of the quotient for a long division, a gcd
by a first remainder and then quadratically in the shorter operand (as trusted.upoly.charge_gcd), one unit per entry
read. No builtin pow and no ** runs: the integer powers N**k and s**k are square-and-multiply loops that charge each
product. A product of two n x n matrices costs n**3 * (1 + product_work(a, b)) for entries of at most a and b bits,
and n**2 reductions modulo P by the length of their quotients (nothing beyond a linear pass, as no value is ever
reduced), plus 2 n**2 for the lengths of the entries; the chain takes bits(k) - 1 squarings and popcount(k) - 1
products with M_int. The first 64 primes are the table trusted.primes.MODULI; each further prime of moduli() costs
EXTRA_PRIME_WORK, the mean work of its search below 2**61. Under the default limits a 2 x 2 claim with B close to
2**65536 ([[1, 1], [1, 0]]**65535) costs about 1.24 million of the 2 million units, nearly all of them for its 1,075
primes (0.2 seconds), and a 64 x 64 claim with entries in {-1, 0, 1} about 275,000 units per matrix product.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd
from operator import mul

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.trusted.integers import division_work, product_work
from mathforge.trusted.primes import MODULI, moduli

MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of the class of M and of the value.
# The mean work of finding one prime of trusted.primes.moduli() beyond its table: about 42 candidates (the mean gap
# between primes near 2**61), each tried by division by the thirteen small bases, about six of them by a first strong
# test (61 multiplications of one unit each) and the prime by all thirteen (793): about 1,200 units. Measured with
# CPython on one machine, the 1,011 primes after the table take 0.15 seconds, about 1,400 units each at the 0.1
# microseconds per unit of trusted.integers.
EXTRA_PRIME_WORK = 1_200

Pair = tuple[int, int]  # The numerator and the positive denominator of a Rational.


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is M**k for args = {"M": M, "k": k}, by the theorem of the module. False for
    anything but the operation's exact shapes; ResourceLimitError for a request past the limits that the module
    lists and for work past the budget."""
    if (type(operation) is not str or operation != "matrix_power" or type(args) is not dict or len(args) != 2
            or "M" not in args or "k" not in args or type(evidence) is not tuple or evidence):
        return False
    matrix, k = args["M"], args["k"]
    if type(k) is not int or k < 0:  # Neither a bool nor a float is an exact int; M**k needs k >= 0.
        return False
    with computation() as budget:  # One budget for every step: without an active budget, the call opens one.
        budget.tick()
        budget.check_int(k)
        limit = budget.limits.max_integer_bits
        rows = _request_rows(budget, matrix)
        if rows is None:
            return False
        n = len(rows)
        claimed = _claimed_rows(budget, value, n)
        if claimed is None:
            return False
        if k == 0:  # M**0 = I: entries 1/1 on the diagonal and 0/1 elsewhere, which is the canonical form.
            budget.tick(1 + n * n)
            return all(pair == ((1, 1) if i == j else (0, 1))
                       for i, row in enumerate(claimed) for j, pair in enumerate(row))
        # The request's resource limits come before any claimed integer is judged: a true claim past them is
        # unverifiable (ResourceLimitError), never refuted.
        common = _common_denominator(budget, rows, limit)
        int_rows, norm = _integer_rows(budget, rows, common)
        bound = _bound(budget, norm, k, limit)
        if not _screened(budget, claimed, bound, common, k, limit) or not _canonical(budget, claimed):
            return False
        scaled = _scaled(budget, claimed, common, k, bound)
        if scaled is None:
            return False
        modulus = _modulus(budget, 2 * bound)
        power = _power_mod(budget, int_rows, k, modulus)
        budget.tick(1 + n * n)
        return scaled == power


def _request_rows(budget: Budget, matrix: object) -> list[list[Pair]] | None:
    """The entries of M as (numerator, denominator) pairs when M is a square Matrix of exact Rationals with positive
    denominators, read as data; None otherwise. budget.matrix(rows, cols) comes first, as in produce(), and a request
    integer over max_integer_bits raises ResourceLimitError. Lowest terms are not needed: s M is an integer matrix
    for any common multiple s of the denominators, and every step below holds for it."""
    if (type(matrix).__module__, type(matrix).__qualname__) != MATRIX:
        return None
    rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)
    if type(rows) is not tuple or type(ncols) is not int:
        return None
    budget.matrix(len(rows), ncols)  # Past max_matrix_entries: ResourceLimitError, before the shape is read.
    n = len(rows)
    if ncols != n:
        return None
    pairs = []
    for row in rows:
        budget.tick(1 + n)
        if type(row) is not tuple or len(row) != n:
            return None
        items = []
        for entry in row:
            if type(entry) is not Rational:
                return None
            numerator = getattr(entry, "numerator", None)  # A forged Rational may lack a slot.
            denominator = getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int or denominator < 1:
                return None
            budget.check_int(numerator)
            budget.check_int(denominator)
            items.append((numerator, denominator))
        pairs.append(items)
    return pairs


def _claimed_rows(budget: Budget, value: object, n: int) -> list[list[Pair]] | None:
    """The entries of the claimed value as (numerator, denominator) pairs when it is an n x n Matrix of exact
    Rationals with positive denominators, read as data; None otherwise. The lengths of the integers are judged by
    _screened and lowest terms by _canonical, once the request's resource limits have been decided."""
    if (type(value).__module__, type(value).__qualname__) != MATRIX:
        return None
    rows, ncols = getattr(value, "rows", None), getattr(value, "ncols", None)
    if type(rows) is not tuple or len(rows) != n or type(ncols) is not int or ncols != n:
        return None
    pairs = []
    for row in rows:
        budget.tick(1 + n)
        if type(row) is not tuple or len(row) != n:
            return None
        items = []
        for entry in row:
            if type(entry) is not Rational:
                return None
            numerator = getattr(entry, "numerator", None)
            denominator = getattr(entry, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int or denominator < 1:
                return None
            items.append((numerator, denominator))
        pairs.append(items)
    return pairs


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd (Lehmer's algorithm) of integers of the given lengths: a first remainder of the longer
    by the shorter one (division_work), then one unit per 64 squared 64-bit words of the shorter one."""
    longer, shorter = max(left_bits, right_bits), min(left_bits, right_bits)
    words = shorter >> 6
    return 1 + division_work(longer, shorter) + (words * words >> 6)


def _quotient_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of a long division beyond one, by the length of its quotient: one word step per pair of quotient and
    divisor words (division_work charges the dividend's words for the quotient's)."""
    return division_work(max(dividend_bits - divisor_bits, 0) + 64, divisor_bits)


def _common_denominator(budget: Budget, rows: list[list[Pair]], limit: int) -> int:
    """s, the least common multiple of the denominators of M. ResourceLimitError once it is longer than limit bits:
    it grows one denominator at a time, so no lcm much longer than the bound is ever formed."""
    common = 1
    for row in rows:
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
                raise ResourceLimitError("Computation limit exceeded: integer bits (the common denominator of M)")
    return common


def _integer_rows(budget: Budget, rows: list[list[Pair]], common: int) -> tuple[list[list[int]], int]:
    """M_int = s M as integer rows, and N = ||M_int||_inf, its maximum absolute row sum."""
    common_bits = common.bit_length()
    int_rows, norm = [], 0
    for row in rows:
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


def _power(budget: Budget, base: int, exponent: int) -> int:
    """base**exponent for ints base >= 1 and exponent >= 1 by left-to-right binary square-and-multiply, each product
    charged before it runs; the caller bounds the length of the result first."""
    if base == 1:
        return 1
    result, base_bits = base, base.bit_length()
    for bit in bin(exponent)[3:]:  # The bits after the leading 1, from the top.
        bits = result.bit_length()
        budget.tick(1 + product_work(bits, bits))
        result = result * result
        if bit == "1":
            budget.tick(1 + product_work(result.bit_length(), base_bits))
            result = result * base
    return result


def _bound(budget: Budget, norm: int, k: int, limit: int) -> int:
    """B = N**k for k >= 1. ResourceLimitError when k log2(N) exceeds limit, the card's resource limit, that is when
    B > 2**limit; B = 2**limit itself is allowed. Decided from the length of N alone when 2**(k * (bits(N) - 1)) <= B
    shows it (equality holds exactly for a power of two N), so that no giant power is formed."""
    if norm <= 1:
        return norm  # 0**k = 0 and 1**k = 1 for k >= 1.
    floor_bits = k * (norm.bit_length() - 1)  # 2**floor_bits <= N**k, equal exactly when N is a power of two.
    if floor_bits > limit or (floor_bits == limit and norm & (norm - 1)):
        raise ResourceLimitError("Computation limit exceeded: integer bits (the bound ||M_int||_inf**k)")
    bound = _power(budget, norm, k)  # k <= limit here, so B has at most 2 * limit bits.
    if bound > 1 << limit:
        raise ResourceLimitError("Computation limit exceeded: integer bits (the bound ||M_int||_inf**k)")
    return bound


def _screened(budget: Budget, claimed: list[list[Pair]], bound: int, common: int, k: int, limit: int) -> bool:
    """False when bit lengths alone refute a claimed entry p/q, before any long arithmetic. An integer longer than
    limit (max_integer_bits) is no value's. For p != 0: |W| = |p| s**k / q is at least |p| when q divides s**k, so
    |p| >= 2**bits(B) is too large; q >= 2**(k bits(s)) exceeds s**k; and s**k >= 2**(k (bits(s) - 1)) >=
    2**(bits(B) + bits(q)) > B q makes |W| > B. Past this screen s**k has fewer than k + bits(B) + bits(q) bits for
    some q, which bounds the power that _scaled forms."""
    bound_bits = bound.bit_length()
    common_bits = common.bit_length()
    floor_bits, ceiling_bits = k * (common_bits - 1), k * common_bits  # 2**floor_bits <= s**k < 2**ceiling_bits.
    for row in claimed:
        budget.tick(1 + len(row))
        for numerator, denominator in row:
            numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
            if numerator_bits > limit or denominator_bits > limit:
                return False
            if numerator and (numerator_bits > bound_bits or denominator_bits > ceiling_bits
                              or floor_bits >= bound_bits + denominator_bits):
                return False
    return True


def _canonical(budget: Budget, claimed: list[list[Pair]]) -> bool:
    """Whether every claimed entry is in lowest terms: gcd(p, q) = 1, so that 0 is 0/1."""
    for row in claimed:
        budget.tick(1 + len(row))
        for numerator, denominator in row:
            if denominator != 1:
                budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
                if gcd(numerator, denominator) != 1:
                    return False
    return True


def _scaled(budget: Budget, claimed: list[list[Pair]], common: int, k: int, bound: int) -> list[list[int]] | None:
    """W = s**k V as integer rows when every entry is an integer in [-B, B]; None otherwise. s**k is formed only
    when an entry is nonzero, and the screen has bounded its length then."""
    power = 0
    scaled = []
    for row in claimed:
        budget.tick(1 + len(row))
        items = []
        for numerator, denominator in row:
            if not numerator:
                items.append(0)
                continue
            if not power:
                power = _power(budget, common, k)
            denominator_bits = denominator.bit_length()
            budget.tick(1 + _quotient_work(power.bit_length(), denominator_bits))
            quotient, remainder = divmod(power, denominator)
            if remainder:  # q does not divide s**k: W is no integer matrix.
                return None
            budget.tick(1 + product_work(quotient.bit_length(), numerator.bit_length()))
            entry = numerator * quotient
            if abs(entry) > bound:
                return None
            items.append(entry)
        scaled.append(items)
    return scaled


def _modulus(budget: Budget, bound: int) -> int:
    """P, the product of the first primes of trusted.primes.moduli() with P > bound >= 0, at least one prime. A prime
    beyond the table MODULI is charged EXTRA_PRIME_WORK before its search runs."""
    primes = moduli()
    product, count = 1, 0
    while True:
        if count >= len(MODULI):
            budget.tick(EXTRA_PRIME_WORK)
        budget.tick(1 + (product.bit_length() >> 12))  # A product with a prime below 2**61: linear in P.
        product *= next(primes)
        count += 1
        if product > bound:
            return product


def _reduction_work(bits: int, modulus_bits: int) -> int:
    """The units of a balanced residue modulo a modulus of modulus_bits bits of an integer of bits bits: the long
    division by the length of its quotient, then a linear pass for the shift into (-P/2, P/2)."""
    return 1 + _quotient_work(bits, modulus_bits) + (modulus_bits >> 12)


def _longest(rows: list[list[int]]) -> int:
    """The greatest bit length of the entries of integer rows, 0 for none."""
    return max((entry.bit_length() for row in rows for entry in row), default=0)


def _product(budget: Budget, left: list[list[int]], right: list[list[int]], modulus: int) -> list[list[int]]:
    """left * right modulo modulus in balanced residues, for square matrices of balanced residues of one size n. The
    n**3 products, the n**2 reductions and the 2 n**2 lengths are charged before they run."""
    n = len(left)
    budget.tick(1 + 2 * n * n)
    left_bits, right_bits = _longest(left), _longest(right)
    budget.tick(n * n * n * (1 + product_work(left_bits, right_bits))
                + n * n * _reduction_work(left_bits + right_bits + n.bit_length(), modulus.bit_length()))
    half = modulus >> 1
    columns = list(zip(*right))
    result = []
    for row in left:
        items = []
        for column in columns:
            residue = sum(map(mul, row, column)) % modulus
            items.append(residue - modulus if residue > half else residue)
        result.append(items)
    return result


def _power_mod(budget: Budget, int_rows: list[list[int]], k: int, modulus: int) -> list[list[int]]:
    """M_int**k modulo modulus in balanced residues, for k >= 1, by left-to-right binary square-and-multiply: a
    squaring for each bit of k after the leading one, then a product with M_int for a set bit."""
    half = modulus >> 1
    modulus_bits = modulus.bit_length()
    base = []
    for row in int_rows:
        items = []
        for entry in row:
            budget.tick(_reduction_work(entry.bit_length(), modulus_bits))
            residue = entry % modulus
            items.append(residue - modulus if residue > half else residue)
        base.append(items)
    result = base
    for bit in bin(k)[3:]:  # The bits after the leading 1, from the top.
        result = _product(budget, result, result, modulus)
        if bit == "1":
            result = _product(budget, result, base, modulus)
    return result
