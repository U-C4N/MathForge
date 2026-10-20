"""Lean functions of the discrete area: factorial, binomial, multinomial, permutations_count, fibonacci, catalan and
stirling.

Producers: math.factorial, math.comb and math.perm, fast doubling for Fibonacci numbers and v0.7's recurrence for
Stirling numbers. Checks: the criteria of v0.7's checker (combinatorics.check_operation): a balanced product of
1 ... n, the binomial recurrence reversed from the claim back to 1, the arrangements cancelled one choice at a time, a
power of the matrix [[1, 1], [1, 0]], the Catalan recurrence and the inclusion-exclusion count of onto maps. A value
comes back only when its check accepts it.

Each function falls back for an argument that is not an exact int (bools included), for an index over the default
max_integer_bits, while a computation budget is active and wherever v0.7 raises: its producers and checks keep every
product within max_integer_bits, and the functions below refuse the inputs whose largest such product (computed or
bounded here) would not fit. Where a value can be millions of bits long (C(n, k), P(n, k) and the Catalan numbers
for a large index), a lower bound on its length refuses the input before math.comb or math.perm computes it.
"""

from __future__ import annotations

from math import comb, factorial as _factorial, perm, prod

from ..limits import _ACTIVE, ComputationLimits
from . import FALLBACK

_active = _ACTIVE.get
_LIMITS = ComputationLimits()
_MAX_BITS = _LIMITS.max_integer_bits
_MAX_WORK = _LIMITS.max_work
_MAX_DEPTH = _LIMITS.max_depth
# The largest n whose n! has at most max_integer_bits bits (5910! has 65,528 bits, 5911! 65,540).
_FACTORIAL_MAX = 5910
# The largest n for which v0.7's Fibonacci doubling and its checker's matrix powers stay within max_integer_bits: both
# reach F(n + 1), which has at most 65,536 bits up to n = 94,400.
_FIBONACCI_MAX = 94_400
# Inputs that need more steps than this in the recurrences of the checks fall back: v0.7 charges one work unit per
# step (and more), so the structured path refuses them at max_work as well, or nearly.
_STEPS = _MAX_WORK // 2
# The structured path inspects the counts and the coefficients of a multinomial several times within max_nodes and
# max_evidence_items (100,000 each): longer tuples fall back.
_MAX_COUNTS = 10_000
# Up to this n, S(n, c) <= n**n and C(k, j) j**n <= 2**k n**n have fewer than 2**15 bits: no limit is in reach.
_STIRLING_SMALL = 2_000
# The largest n for which n * C(2n, n), the product that v0.7's binomial forms for catalan(n), has at most
# max_integer_bits bits (_over(C(2n, n), n) is false up to 32,764 and true from 32,765 on; C(2n, n) increases).
_CATALAN_MAX = 32_764


def _choose_too_long(n: int, k: int) -> bool:
    """Whether C(n, k), for 0 < k <= n / 2, surely has more than max_integer_bits bits, by its lower bounds
    C(n, k) >= (n / k)**k >= 2**k: log2 C(n, k) >= k (bitlen(n) - bitlen(k) - 1) and >= k. Then v0.7's product
    k * C(n, k) passes the bound as well (_over), so the answer is FALLBACK either way."""
    return k * max(1, n.bit_length() - k.bit_length() - 1) > _MAX_BITS


def _over(x: int, y: int) -> bool:
    """Whether v0.7's _mul(x, y) exceeds the default max_integer_bits: its product does (Budget.product refuses
    operands of more bits only when the product has more bits as well, and charges nothing for a zero)."""
    bits = x.bit_length() + y.bit_length()
    return bits > _MAX_BITS and (bits > _MAX_BITS + 1 or (x * y).bit_length() > _MAX_BITS)


def _index(value: object) -> bool:
    """A combinatorial index as v0.7 accepts it: an exact nonnegative int within max_integer_bits."""
    return type(value) is int and value >= 0 and value.bit_length() <= _MAX_BITS


def _balanced_product(low: int, high: int) -> int:
    """low * (low + 1) * ... * high (1 when low > high) as a product tree, v0.7's _balanced_product: leaves of 64
    consecutive factors (math.prod), then the leaves multiplied in pairs, level by level."""
    items = [prod(range(start, min(start + 64, high + 1))) for start in range(low, high + 1, 64)]
    while len(items) > 1:
        odd = items[-1] if len(items) & 1 else None
        items = [a * b for a, b in zip(items[::2], items[1::2])]
        if odd is not None:
            items.append(odd)
    return items[0] if items else 1


def _check_choose(n: int, k: int, value: int) -> bool:
    """v0.7's _check_choose: reverse the binomial recurrence from the claim back to C(n - k, 0) = 1."""
    if value < 0:
        return False
    if k > n:
        return value == 0
    k = min(k, n - k)
    base = n - k
    for index in range(k, 0, -1):
        value, remainder = divmod(value * index, base + index)
        if remainder:
            return False
    return value == 1


def factorial(n: object) -> object:
    if type(n) is not int or n < 0 or n > _FACTORIAL_MAX or _active() is not None:
        return FALLBACK
    try:
        value = _factorial(n)
        return value if value == _balanced_product(1, n) else FALLBACK
    except Exception:
        return FALLBACK


def binomial(n: object, k: object) -> object:
    if not (_index(n) and _index(k)) or _active() is not None:
        return FALLBACK
    try:
        if k > n:
            return 0
        smaller = min(k, n - k)
        if smaller > _STEPS or (smaller and _choose_too_long(n, smaller)):
            return FALLBACK
        value = comb(n, smaller)
        # v0.7 forms smaller * C(n, smaller) before its last division, and its check the same product.
        if smaller and _over(value, smaller):
            return FALLBACK
        return value if _check_choose(n, k, value) else FALLBACK
    except Exception:
        return FALLBACK


def multinomial(counts: object) -> object:
    if type(counts) not in (tuple, list) or len(counts) > _MAX_COUNTS or _active() is not None:
        return FALLBACK
    try:
        for count in counts:
            if not _index(count):
                return FALLBACK
        total, value, steps, coefficients = 0, 1, 0, []
        for count in counts:
            total += count
            smaller = min(count, total - count)
            steps += smaller
            if total.bit_length() > _MAX_BITS or steps > _STEPS or (smaller and _choose_too_long(total, smaller)):
                return FALLBACK
            coefficient = comb(total, count)
            if smaller and _over(coefficient, smaller):
                return FALLBACK
            if _over(value, coefficient):
                return FALLBACK
            value *= coefficient
            coefficients.append((total, count, coefficient))
        # v0.7's check: each coefficient by the reversed recurrence, and their product.
        product = 1
        for total, count, coefficient in coefficients:
            if not _check_choose(total, count, coefficient):
                return FALLBACK
            product *= coefficient
        return value if product == value else FALLBACK
    except Exception:
        return FALLBACK


def permutations_count(n: object, k: object) -> object:
    if not (_index(n) and _index(k)) or _active() is not None:
        return FALLBACK
    try:
        if k > n:
            return 0
        # P(n, k) >= k! (past the bound from k = 5,911 on) and P(n, k) >= (n - k + 1)**k: a value surely past
        # max_integer_bits, which the check below would refuse, is refused before math.perm computes it.
        if k > _FACTORIAL_MAX or k * ((n - k + 1).bit_length() - 1) > _MAX_BITS:
            return FALLBACK
        value = perm(n, k)
        if value.bit_length() > _MAX_BITS:  # v0.7's running product reaches the value itself.
            return FALLBACK
        # v0.7's check: cancel the choices n, n - 1, ..., n - k + 1 (here 64 consecutive ones at a time, each block's
        # product dividing exactly); the empty arrangement counts one.
        remaining, high, low = value, n, n - k
        while high > low:
            start = max(low, high - 64)
            remaining, remainder = divmod(remaining, prod(range(start + 1, high + 1)))
            if remainder:
                return FALLBACK
            high = start
        return value if remaining == 1 else FALLBACK
    except Exception:
        return FALLBACK


def _fibonacci_pair(n: int) -> tuple[int, int]:
    """(F(n), F(n + 1)) by fast doubling: F(2m) = F(m) (2 F(m + 1) - F(m)), F(2m + 1) = F(m)**2 + F(m + 1)**2."""
    a, b = 0, 1
    for bit in bin(n)[2:]:
        c = a * (2 * b - a)
        d = a * a + b * b
        a, b = (d, c + d) if bit == "1" else (c, d)
    return a, b


def _matrix_power_entry(n: int) -> int:
    """The off-diagonal entry of [[1, 1], [1, 0]]**n (v0.7's _check_fibonacci), by square-and-multiply from the left:
    a square of [[a, b], [c, d]] is [[a*a + b*c, b*(a + d)], [c*(a + d), b*c + d*d]], and a product with the matrix
    [[1, 1], [1, 0]] is [[a + b, a], [c + d, c]]."""
    a, b, c, d = 1, 0, 0, 1
    for bit in bin(n)[2:]:
        bc, trace = b * c, a + d
        a, b, c, d = a * a + bc, b * trace, c * trace, bc + d * d
        if bit == "1":
            a, b, c, d = a + b, a, c + d, c
    return b


def fibonacci(n: object) -> object:
    if type(n) is not int or n < 0 or n > _FIBONACCI_MAX or _active() is not None:
        return FALLBACK
    try:
        value = _fibonacci_pair(n)[0]
        return value if value == _matrix_power_entry(n) else FALLBACK
    except Exception:
        return FALLBACK


def catalan(n: object) -> object:
    if not _index(n) or n > _CATALAN_MAX or _active() is not None:
        return FALLBACK
    try:
        central = comb(2 * n, n)
        if n and _over(central, n):  # v0.7 forms n * C(2n, n) in its binomial, before the last division.
            return FALLBACK
        value = central // (n + 1)
        # v0.7's check: the recurrence C(i + 1) = C(i) * 2 (2i + 1) / (i + 2) from C(0) = 1.
        expected = 1
        for index in range(n):
            expected = expected * (2 * (2 * index + 1)) // (index + 2)
        return value if value == expected else FALLBACK
    except Exception:
        return FALLBACK


def _stirling_row(n: int, k: int) -> list[int]:
    """S(n, 0), ..., S(n, k) by v0.7's recurrence S(d, c) = S(d - 1, c - 1) + c S(d - 1, c), one degree at a time."""
    row = [1] + [0] * k
    for degree in range(1, n + 1):
        top = min(degree, k)
        row = [0, *(row[count - 1] + count * row[count] for count in range(1, top + 1)), *row[top + 1:]]
    return row


def _stirling_band(n: int, k: int) -> int:
    """S(n, k) by the same recurrence, kept to the band of S(d, c) that S(n, k) depends on: k - (n - d) <= c <= k."""
    row = [1] + [0] * k
    for degree in range(1, n + 1):
        low = max(1, k - (n - degree))
        for count in range(min(degree, k), low - 1, -1):
            row[count] = row[count - 1] + count * row[count]
        row[0] = 0
    return row[k]


def stirling(n: object, k: object) -> object:
    if not (_index(n) and _index(k)) or _active() is not None:
        return FALLBACK
    try:
        if k > n:
            return 0
        if k == 0:
            return int(n == 0)
        if k == 1 or k == n:
            return 1
        # S(n, k) >= k**(n - k), so a larger power refuses the input before any work; v0.7's recurrence takes about
        # n*k steps of two work units.
        if (n - k) * (k.bit_length() - 1) > _MAX_BITS or n * k > _STEPS:
            return FALLBACK
        if n <= _STIRLING_SMALL:
            # Every S(d, c) <= n**n and every term C(k, j) j**n of the check stay far below max_integer_bits here.
            value = _stirling_band(n, k)
        else:
            row = _stirling_row(n, k)
            # v0.7's recurrence forms S(d, c) for every c <= k on the way: the largest at d = n.
            if max(row).bit_length() > _MAX_BITS:
                return FALLBACK
            value = row[k]
        return value if _check_stirling(n, k, value) else FALLBACK
    except Exception:
        return FALLBACK


def _check_stirling(n: int, k: int, value: int) -> bool:
    """v0.7's _check_stirling for 1 < k < n: k! S(n, k) = sum over j of (-1)**(k - j) C(k, j) j**n (onto maps), with
    v0.7's products and partial sums kept within max_integer_bits (it refuses the input otherwise)."""
    total, coefficient = 0, 1
    for index in range(k + 1):
        power = pow(index, n)
        term = coefficient * power
        if power.bit_length() > _MAX_BITS or term.bit_length() > _MAX_BITS:
            return False
        total = total - term if (k - index) & 1 else total + term
        if total.bit_length() > _MAX_BITS:
            return False
        if index < k:
            coefficient = coefficient * (k - index) // (index + 1)
    scale = _balanced_product(1, k)
    if k.bit_length() > _MAX_DEPTH or _over(value, scale):
        return False
    return value * scale == total


LEAN = {"factorial": factorial, "binomial": binomial, "multinomial": multinomial,
        "permutations_count": permutations_count, "fibonacci": fibonacci, "catalan": catalan, "stirling": stirling}
