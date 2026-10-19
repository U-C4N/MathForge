"""Integer kernel: factorization by trial division and Pollard-Brent rho, a Miller-Rabin splitting test, and v0.7's
prime certificates built with the builtin ``pow``.

This is producer code: nothing here is a proof. ``factor`` returns what ``is_probable_prime`` calls primes; every
caller certifies each prime (``number_theory._prime_certificate`` and its checker on the structured path,
``prime_certificate`` and ``mathforge.certify.ntheory.primes`` on the lean path) before a value leaves MathForge.

Work. ``factor`` and ``prime_certificate`` charge the active computation budget once per trial-division batch (the
primes tried), once per rho batch (its iterations), once per Miller-Rabin test and once per round of Pocklington
witnesses (their exponentiations), each modular multiplication weighted by the size of the modulus and charged before
it runs, so that a computation that runs away, or one too large to start, stops at ``max_work`` with
ResourceLimitError. Without an active budget they charge a private counter of the default ``max_work`` instead: the
kernel never runs unbounded.
"""

from __future__ import annotations

from math import gcd, isqrt, prod

from ..limits import ComputationLimits, ResourceLimitError, current_budget
from ..trusted.integers import powers_by_prime

__all__ = ["SMALL_PRIMES", "SPF_LIMIT", "factor", "is_probable_prime", "prime_certificate", "smallest_factor_table"]

_DEFAULT_WORK = ComputationLimits().max_work


def _primes_below(limit: int) -> tuple[int, ...]:
    sieve = bytearray([1]) * limit
    sieve[0:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytes(len(range(p * p, limit, p)))
    return tuple(index for index, flag in enumerate(sieve) if flag)


SMALL_PRIMES = _primes_below(1024)  # The trial-division primes: the 172 primes below 2**10.
_SMALL_PRODUCT = prod(SMALL_PRIMES)
# A number above 1 with no prime factor below 2**10 has its smallest prime factor at 1031 or above, so it is prime
# when it is below 1031**2.
_SMALL_BOUND = 1031 * 1031
_ODD_PRIMORIAL_29 = prod((3, 5, 7, 11, 13, 17, 19, 23, 29))

# Smallest prime factors below 2**16, as bytes: a composite m < 2**16 has its smallest prime factor below 256; a
# prime (and 0, 1) has 0. Built on first use.
SPF_LIMIT = 1 << 16
_spf: list[bytearray] = []


def smallest_factor_table() -> bytearray:
    """table[m] is the smallest prime factor of the composite m < 2**16, and 0 for a prime m (and for 0 and 1)."""
    if not _spf:
        table = bytearray(SPF_LIMIT)
        for p in reversed(_primes_below(256)):  # Larger primes first: the smallest prime factor is written last.
            table[p * p::p] = bytes((p,)) * len(range(p * p, SPF_LIMIT, p))
        if not _spf:
            _spf.append(table)
    return _spf[0]


class _Work:
    """The charges of one kernel call: the active budget's, else a private counter of the default max_work."""

    __slots__ = ("_tick", "_left")

    def __init__(self) -> None:
        budget = current_budget()
        self._tick = None if budget is None else budget.tick
        self._left = _DEFAULT_WORK

    def charge(self, units: int) -> None:
        if self._tick is not None:
            self._tick(units)
        else:
            self._left -= units
            if self._left < 0:
                raise ResourceLimitError(f"Computation limit exceeded: work ({_DEFAULT_WORK})")


def _weight(n: int) -> int:
    """The work of one modular multiplication modulo n, in units of a 64-bit one."""
    return 1 + (n.bit_length() >> 6)


def _power_work(n: int) -> int:
    """The work of one modular exponentiation modulo n by an exponent of n's size: a multiplication per bit. It is
    charged before the exponentiation, so that a number too large for the budget is refused at once."""
    return n.bit_length() * _weight(n)


# Miller-Rabin bases. With the first k primes as bases the test is deterministic below psi_k (OEIS A014233).
_MR_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_MR_BOUNDS = ((3_215_031_751, 4), (3_474_749_660_383, 6), (341_550_071_728_321, 7),
              (3_825_123_056_546_413_051, 9), (318_665_857_834_031_151_167_461, 12),
              (3_317_044_064_679_887_385_961_981, 13))
_MR_EXTRA = 8  # Bases derived from n above psi_13, where the test is no longer a proof.
_MERSENNE_61 = (1 << 61) - 1


def _bases(n: int) -> tuple[int, ...]:
    for bound, count in _MR_BOUNDS:
        if n < bound:
            return _MR_PRIMES[:count]
    seed, extra = n % _MERSENNE_61, []
    for index in range(_MR_EXTRA):
        seed = (seed * 6364136223846793005 + 1442695040888963407 + index) % _MERSENNE_61
        extra.append(2 + seed % (n - 3))
    return _MR_PRIMES + tuple(extra)


def _strong_probable_prime(n: int, d: int, s: int, base: int) -> bool:
    x = pow(base, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def is_probable_prime(n: int) -> bool:
    """Miller-Rabin with the first prime bases: deterministic below 3,317,044,064,679,887,385,961,981 (psi_13); above
    it, eight more bases derived from n. For splitting decisions only: primality proofs are prime certificates."""
    return _probable_prime(n, _Work())


def _probable_prime(n: int, work: _Work) -> bool:
    if n < SPF_LIMIT:
        return n > 1 and not smallest_factor_table()[n]
    if not n & 1 or gcd(n, _ODD_PRIMORIAL_29) != 1:
        return False
    bases = _bases(n)
    work.charge(len(bases) * _power_work(n))
    d = n - 1
    s = (d & -d).bit_length() - 1
    d >>= s
    for base in bases:
        if not _strong_probable_prime(n, d, s, base):
            return False
    return True


def _factor_small(n: int) -> list[tuple[int, int]]:
    """The factorization of 1 <= n < 2**16 from the table of smallest prime factors."""
    table, pairs = smallest_factor_table(), []
    while n > 1:
        p = table[n] or n
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        pairs.append((p, e))
    return pairs


def _rho(n: int, c: int, work: _Work) -> int:
    """Pollard's rho with Brent's cycle finding on x -> x*x + c (mod n) from 2, the differences multiplied together and
    a gcd taken once per batch: a factor of the odd composite n, which is n itself when this c fails."""
    weight, batch = _weight(n), 128
    y, r, q, g = 2, 1, 1, 1
    x = ys = y
    while g == 1:
        x = y
        for start in range(0, r, batch):  # Charged batch by batch, before the batch runs.
            steps = min(batch, r - start)
            work.charge(steps * weight)
            for _ in range(steps):
                y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            steps = min(batch, r - k)
            work.charge(2 * steps * weight)
            for _ in range(steps):
                y = (y * y + c) % n
                q = q * (x - y) % n
            g = gcd(q, n)
            k += steps
        r <<= 1
    if g == n:  # The batch overshot: step through it again, one gcd at a time.
        work.charge(batch * weight)
        while True:
            ys = (ys * ys + c) % n
            g = gcd(x - ys, n)
            if g > 1:
                return g
    return g


def _iroot(n: int, k: int) -> int:
    """The integer k-th root of n >= 1 (rounded down), by Newton's iteration from above."""
    x = 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _split(n: int, work: _Work) -> int:
    """A nontrivial factor of the composite n, which has no prime factor below 2**10."""
    root = isqrt(n)
    if root * root == n:
        return root
    c = 1
    while True:
        d = _rho(n, c, work)
        if d != n:
            return d
        c += 1
        if c == 33:  # Thirty-two polynomials cycled without a split: try the perfect powers once, then go on.
            for k in range(3, n.bit_length() // 10 + 2):
                work.charge(_power_work(n) // k)
                root = _iroot(n, k)
                if root ** k == n:
                    return root


def _divide_out(n: int, p: int) -> tuple[int, int]:
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return n, e


def factor(n: int) -> tuple[tuple[int, int], ...]:
    """The (prime, exponent) pairs of the integer n >= 1, sorted by prime; () for 1.

    Trial division by the primes below 2**10, then Pollard-Brent rho with the polynomials x*x + c for c = 1, 2, ...
    (seed 2), Miller-Rabin deciding which cofactors to split. Work is charged once per batch (see the module)."""
    if type(n) is not int or n < 1:
        raise ValueError("factor needs an integer n >= 1")
    if n < SPF_LIMIT:
        return tuple(_factor_small(n))
    return _factor(n, _Work())


def _factor(n: int, work: _Work) -> tuple[tuple[int, int], ...]:
    if n < SPF_LIMIT:
        return tuple(_factor_small(n))
    found: dict[int, int] = {}
    g = gcd(n, _SMALL_PRODUCT)
    if g > 1:
        if g < SPF_LIMIT:
            primes = [p for p, _ in _factor_small(g)]
            tried = len(primes)
        else:
            primes, tried = [], 0
            for p in SMALL_PRIMES:
                tried += 1
                if g % p == 0:
                    primes.append(p)
                    g //= p
                    if g == 1:
                        break
        work.charge(tried)
        divisions = 0
        for p in primes:
            n, found[p] = _divide_out(n, p)
            divisions += found[p]
        work.charge(divisions * _weight(n))
    pending = [n] if n > 1 else []
    while pending:
        m = pending.pop()
        if m < _SMALL_BOUND or _probable_prime(m, work):
            found[m] = found.get(m, 0) + 1
        else:
            d = _split(m, work)
            pending.append(d)
            pending.append(m // d)
    return tuple(sorted(found.items()))


# ---------------------------------------------------------------------------------------------------------------
# Prime certificates in v0.7's format (number_theory._prime_certificate), built for speed: the builtin pow, the
# kernel's factor, a strong probable-prime test to base 2 before any factoring, and prime witnesses tried in a cheap
# order. A certificate proves its claim only once a checker has accepted it (mathforge.certify.ntheory.primes).

_TRIAL_BOUND = 10_000
_PRIMORIAL_31 = prod((2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31))  # v0.7's trial divisors before a Fermat test.
_MAX_DEPTH = ComputationLimits().max_depth


def prime_certificate(n: int) -> tuple[bool, tuple]:
    """Whether the integer n is prime, with a certificate in v0.7's format: ("below", n), ("trial", n),
    ("factor", n, d), ("fermat", n, w) or ("pocklington", n, ((p, e, certificate), ...), (w_p, ...)).

    Raises ResourceLimitError past the depth or the work of the default limits (n - 1 that rho cannot factor)."""
    return _certificate(n, 0, _Work())


def _certificate(n: int, depth: int, work: _Work) -> tuple[bool, tuple]:
    if depth > _MAX_DEPTH:
        raise ResourceLimitError("Prime certificate exceeds the depth budget")
    if n < 2:
        return False, ("below", n)
    if n < _TRIAL_BOUND:
        divisor = smallest_factor_table()[n]
        return (False, ("factor", n, divisor)) if divisor else (True, ("trial", n))
    if gcd(n, _PRIMORIAL_31) != 1:
        for divisor in SMALL_PRIMES:
            if n % divisor == 0:
                return False, ("factor", n, divisor)
    m, weight = n - 1, _weight(n)
    s = (m & -m).bit_length() - 1
    work.charge(_power_work(n))
    x = pow(2, m >> s, n)
    if x != 1 and x != m:  # The strong test to base 2: its failure is a Fermat witness or a square root of 1.
        for _ in range(s - 1):
            y = x * x % n
            if y == m:
                break
            if y == 1:
                return False, ("factor", n, gcd(x - 1, n))
            x = y
        else:
            if x * x % n == 1:
                return False, ("factor", n, gcd(x - 1, n))
            return False, ("fermat", n, 2)
    factors = _factor(m, work)
    certified = []
    for prime, exponent in factors:
        proven, certificate = _certificate(prime, depth + 1, work)
        if not proven:
            raise ArithmeticError("Internal factorization produced a composite factor")
        certified.append((prime, exponent, certificate))
    # Witnesses: the primes 2, 3, 5, ... in turn, each tried at once on every prime of n - 1 still without one.
    remaining = [prime for prime, _ in factors]
    witness_of: dict[int, int] = {}
    for witness in _witness_candidates(n):
        work.charge(len(remaining).bit_length() * _power_work(n) + len(remaining) * weight)
        powers = powers_by_prime(witness, m, n, remaining)
        first = remaining[0]
        if pow(powers[first], first, n) != 1:  # witness**(n - 1), from witness**((n - 1)/first)
            return False, ("fermat", n, witness)
        still = []
        for prime in remaining:
            divisor = gcd(powers[prime] - 1, n)
            if divisor == 1:
                witness_of[prime] = witness
            elif divisor == n:
                still.append(prime)
            else:
                return False, ("factor", n, divisor)
        remaining = still
        if not remaining:
            break
    else:
        raise ArithmeticError("No Pocklington witness found")
    witnesses = tuple(witness_of[prime] for prime, _ in factors)
    return True, ("pocklington", n, tuple(certified), witnesses)


def _witness_candidates(n: int):
    """The Pocklington witnesses to try: the primes below 2**10 (a power of a witness that failed for a prime fails
    as well), then every integer from 1025 up, all below n."""
    for witness in SMALL_PRIMES:
        if witness >= n:
            return
        yield witness
    yield from range(SMALL_PRIMES[-1] + 1, n)
