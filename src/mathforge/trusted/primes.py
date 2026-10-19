"""Word-sized primes for modular certificates, and a deterministic primality test.

``MODULI`` holds the 64 largest primes below 2**61, in decreasing order. Producers and independent checkers of the
linear algebra families compute determinants and ranks modulo these primes: a determinant that is not zero modulo a
prime is not zero, and agreement modulo primes whose product exceeds twice a bound fixes an integer of that bound.
``is_prime`` is Miller-Rabin with the bases 2 to 37, which is deterministic for every n below 3.3 * 10**24 (Sorenson
and Webster, 2015), so in particular for every candidate below MODULI[-1]. tests/v08/linalg/test_kernel.py checks the
table: 64 entries, strictly decreasing, below 2**61, each prime, and no prime between two entries or above the first.
"""

from __future__ import annotations

MODULI: tuple[int, ...] = (
    2305843009213693951, 2305843009213693921, 2305843009213693907,
    2305843009213693723, 2305843009213693693, 2305843009213693669,
    2305843009213693613, 2305843009213693561, 2305843009213693549,
    2305843009213693487, 2305843009213693421, 2305843009213693373,
    2305843009213693277, 2305843009213693193, 2305843009213693153,
    2305843009213693133, 2305843009213693123, 2305843009213693109,
    2305843009213693093, 2305843009213693013, 2305843009213692967,
    2305843009213692937, 2305843009213692799, 2305843009213692757,
    2305843009213692737, 2305843009213692671, 2305843009213692653,
    2305843009213692601, 2305843009213692581, 2305843009213692527,
    2305843009213692463, 2305843009213692427, 2305843009213692419,
    2305843009213692409, 2305843009213692343, 2305843009213692331,
    2305843009213692283, 2305843009213692211, 2305843009213692199,
    2305843009213692139, 2305843009213692107, 2305843009213692103,
    2305843009213692097, 2305843009213692089, 2305843009213692083,
    2305843009213692043, 2305843009213692031, 2305843009213692029,
    2305843009213692007, 2305843009213691993, 2305843009213691929,
    2305843009213691869, 2305843009213691837, 2305843009213691819,
    2305843009213691767, 2305843009213691581, 2305843009213691579,
    2305843009213691569, 2305843009213691567, 2305843009213691551,
    2305843009213691413, 2305843009213691401, 2305843009213691357,
    2305843009213691347,
)

_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_DETERMINISTIC_BELOW = 3_317_044_064_679_887_385_961_981  # Bases 2..37 decide every n below this.


def is_prime(n: int) -> bool:
    """Whether n is prime, for 0 <= n < 3.3 * 10**24 (deterministic Miller-Rabin with the bases 2 to 37)."""
    if type(n) is not int or n < 0 or n >= _DETERMINISTIC_BELOW:
        raise ValueError("is_prime takes an integer in [0, 3.3 * 10**24)")
    if n < 2:
        return False
    for p in _BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while not d & 1:
        d >>= 1
        s += 1
    for a in _BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_below(bound: int):
    """The primes below bound, in decreasing order (an iterator): MODULI's continuation when the table runs out."""
    candidate = bound - 1
    while candidate >= 2:
        if is_prime(candidate):
            yield candidate
        candidate -= 1


def moduli():
    """MODULI, then every smaller prime in decreasing order (an iterator)."""
    yield from MODULI
    yield from primes_below(MODULI[-1])
