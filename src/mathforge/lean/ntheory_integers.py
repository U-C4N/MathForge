"""Lean functions of the integer operations of the ntheory area: integer_gcd, integer_lcm, integer_xgcd, mod_inverse,
crt, is_prime, factor_integer, divisors and totient.

Each one is straight-line code over the C builtins (math.gcd, pow(x, -1, m)) and the integer kernel, followed by
the criterion of the v0.7 checker of its operation (number_theory.check_operation) or by the prime-certificate
checker mathforge.certify.ntheory.primes, and it returns the value only when that check accepts it. It falls back
for an argument that is not an exact int (bools included), for an integer over the default max_integer_bits, while a
computation budget is active, wherever v0.7 raises, and where v0.7's checks would exceed a default limit: their
products are bounded by max_integer_bits, which the functions below test on the same products.

The Bezout coefficients of v0.7 (number_theory._xgcd, the extended Euclidean algorithm on |a| and |b|) come from one
modular inverse: when neither of x = |a|, y = |b| divides the other, the algorithm's coefficient s of x is the
residue of (x/g)**-1 modulo y/g of least absolute value, the positive one on a tie (|s| <= y/(2g), since the last
quotient is at least 2).
"""

from __future__ import annotations

from math import gcd, prod

from ..certify.ntheory.primes import check_prime, trial_prime
from ..kernels import integers as kernel
from ..limits import _ACTIVE, ComputationLimits
from ..number_theory import CRTResult, IntegerFactorization
from . import FALLBACK

_active = _ACTIVE.get
_MAX_BITS = ComputationLimits().max_integer_bits
_new = object.__new__
_set = object.__setattr__
_SPF = kernel.smallest_factor_table()
_TRIAL_BOUND = 10_000
# The lean paths of factor_integer, divisors, totient and is_prime take numbers up to this size; larger ones go to the
# structured path, whose records hold every certificate (and whose budget stops what the kernel cannot factor).
_FACTOR_BITS = 1024
_PRIME_BITS = 4096
# The structured verifier inspects the divisors three times and the certificates twice within max_evidence_items
# (100,000), and the run inspects the divisors three times within max_nodes: these bounds keep both far below.
_CERTIFICATE_NODES = 10_000
_MAX_DIVISORS = 20_000
# A certificate of a number below 2**64 has fewer than _SMALL_NODES nodes (at most 15 primes divide n - 1, and their
# certificates share its 64 bits), so it is counted without a walk.
_SMALL_CERTIFICATES = 1 << 64
_SMALL_NODES = 1_000
_MAX_CONGRUENCES = 2_000  # The structured verifier inspects about fourteen items per congruence.


def _over(x: int, y: int) -> bool:
    """Whether v0.7's _product(x, y) exceeds the default max_integer_bits: its product does (Budget.product refuses
    operands of more bits only when the product has more bits as well, and charges nothing for a zero)."""
    bits = x.bit_length() + y.bit_length()
    return bits > _MAX_BITS and (bits > _MAX_BITS + 1 or (x * y).bit_length() > _MAX_BITS)


def _euclid(x: int, y: int) -> tuple[int, int, int]:
    """v0.7's _xgcd(x, y) for x, y >= 0: (g, s, t) with s*x + t*y = g, the extended Euclidean algorithm's."""
    if y == 0:
        return x, 1, 0
    if x == 0:
        return y, 0, 1
    g = gcd(x, y)
    if g == y:  # y divides x: one step, quotient x // y.
        return g, 0, 1
    if g == x:  # x divides y, x < y: the first step swaps them.
        return g, 1, 0
    x1, y1 = x // g, y // g
    u = pow(x1, -1, y1)
    s = u if 2 * u <= y1 else u - y1
    return g, s, (1 - s * x1) // y1


def _xgcd(a: int, b: int) -> tuple[int, int, int] | None:
    """v0.7's (g, s, t) for the integers a and b, or None when its checker's products s*a, t*b exceed the default
    integer bits, or when the Bezout criterion rejects them (g >= 0 divides a and b, s*a + t*b = g)."""
    x = -a if a < 0 else a
    y = -b if b < 0 else b
    if y == 0:
        g, s, t = x, 1, 0
    elif x == 0:
        g, s, t = y, 0, 1
    else:  # _euclid(x, y), inline.
        g = gcd(x, y)
        if g == y:
            s, t = 0, 1
        elif g == x:
            s, t = 1, 0
        else:
            x1, y1 = x // g, y // g
            u = pow(x1, -1, y1)
            s = u if 2 * u <= y1 else u - y1
            t = (1 - s * x1) // y1
    if a < 0:
        s = -s
    if b < 0:
        t = -t
    sa, tb = s * a, t * b
    if sa.bit_length() > _MAX_BITS or tb.bit_length() > _MAX_BITS:
        return None
    if g:
        if a % g or b % g or sa + tb != g:
            return None
    elif a or b or sa + tb:
        return None
    return g, s, t


def integer_gcd(a: object, b: object) -> object:
    if type(a) is not int or type(b) is not int or _active() is not None:
        return FALLBACK
    try:
        if a.bit_length() > _MAX_BITS or b.bit_length() > _MAX_BITS:
            return FALLBACK
        g = gcd(a, b)  # The producer; the Bezout triple below is its certificate.
        triple = _xgcd(a, b)
        if triple is None or triple[0] != g:
            return FALLBACK
        return g
    except Exception:
        return FALLBACK


def integer_xgcd(a: object, b: object) -> object:
    if type(a) is not int or type(b) is not int or _active() is not None:
        return FALLBACK
    try:
        if a.bit_length() > _MAX_BITS or b.bit_length() > _MAX_BITS:
            return FALLBACK
        triple = _xgcd(a, b)
        return FALLBACK if triple is None else triple
    except Exception:
        return FALLBACK


def integer_lcm(a: object, b: object) -> object:
    if type(a) is not int or type(b) is not int or _active() is not None:
        return FALLBACK
    try:
        if a.bit_length() > _MAX_BITS or b.bit_length() > _MAX_BITS:
            return FALLBACK
        if not a or not b:  # lcm(a, 0) = 0, with v0.7's triple (|a|, sign(a), 0) or (0, 1, 0).
            return 0 if _xgcd(a, b) is not None else FALLBACK
        g = gcd(a, b)
        value = abs(a * b) // g
        if value.bit_length() > _MAX_BITS:  # v0.7 multiplies a // g by b under the integer bound.
            return FALLBACK
        # v0.7's check: its Bezout triple (as _xgcd, inline) and the value |a // g * b|.
        x = -a if a < 0 else a
        y = -b if b < 0 else b
        if g == y:
            s, t = 0, 1
        elif g == x:
            s, t = 1, 0
        else:
            x1, y1 = x // g, y // g
            u = pow(x1, -1, y1)
            s = u if 2 * u <= y1 else u - y1
            t = (1 - s * x1) // y1
        if a < 0:
            s = -s
        if b < 0:
            t = -t
        sa, tb = s * a, t * b
        if sa.bit_length() > _MAX_BITS or tb.bit_length() > _MAX_BITS:
            return FALLBACK
        if a % g or b % g or sa + tb != g or value != abs(a // g * b):
            return FALLBACK
        return value
    except Exception:
        return FALLBACK


def mod_inverse(a: object, modulus: object) -> object:
    if type(a) is not int or type(modulus) is not int or _active() is not None:
        return FALLBACK
    try:
        if a.bit_length() > _MAX_BITS or modulus.bit_length() > _MAX_BITS or modulus <= 0:
            return FALLBACK
        value = pow(a, -1, modulus)  # ValueError when a has no inverse: v0.7 raises invalid_input.
        # v0.7 multiplies a by the inverse, then the modulus by (1 - a*value)/modulus, under the integer bound.
        if a.bit_length() + value.bit_length() >= _MAX_BITS:
            return FALLBACK
        if 0 <= value < modulus and (a * value - 1) % modulus == 0:
            return value
        return FALLBACK
    except Exception:
        return FALLBACK


def _crt_result(residue: int | None, modulus: int | None) -> CRTResult:
    value = _new(CRTResult)
    _set(value, "residue", residue)
    _set(value, "modulus", modulus)
    return value


def crt(congruences: object) -> object:
    if type(congruences) not in (tuple, list) or _active() is not None or len(congruences) > _MAX_CONGRUENCES:
        return FALLBACK
    try:
        pairs, bits, total_bits, modulus_bits = [], 0, 0, 0
        for pair in congruences:
            if type(pair) not in (tuple, list) or len(pair) != 2:
                return FALLBACK
            target, next_modulus = pair
            if type(target) is not int or type(next_modulus) is not int or next_modulus <= 0:
                return FALLBACK
            target_bits, next_bits = target.bit_length(), next_modulus.bit_length()
            if target_bits > _MAX_BITS or next_bits > _MAX_BITS:
                return FALLBACK
            pairs.append((target, next_modulus))
            bits = max(bits, target_bits)
            total_bits += next_bits
            modulus_bits = max(modulus_bits, next_bits)
        # v0.7's products: a running modulus (at most the product of the moduli) or a difference of residues times a
        # Bezout coefficient, a step or a reduced modulus (each at most the largest modulus). When even the largest of
        # these bounds fits, no product needs testing.
        tested = max(bits, total_bits) + modulus_bits + 1 > _MAX_BITS
        # The standard construction, step by step as v0.7's.
        residue, modulus, trace = 0, 1, []
        for target, next_modulus in pairs:
            g, s, t = _euclid(modulus, next_modulus)
            delta = target - residue
            if g == 1:
                quotient, reduced = delta, next_modulus
            elif delta % g:
                return FALLBACK  # An inconsistent system: the structured path records where it fails.
            else:
                quotient, reduced = delta // g, next_modulus // g
            step = quotient * s % reduced
            if tested and (_over(modulus, s) or _over(next_modulus, t) or _over(quotient, s)
                           or _over(modulus, step) or _over(modulus, reduced)):
                return FALLBACK
            residue += modulus * step
            modulus *= reduced
            residue %= modulus
            trace.append((g, s, t))
        # The check: v0.7's for the modulus, every step's Bezout triple (s m + t m_i = g, g dividing both, so g is their
        # gcd and each step takes the lcm); then every congruence holds for the canonical residue.
        common = 1
        for (target, next_modulus), (g, s, t) in zip(pairs, trace):
            if g < 1 or common % g or next_modulus % g or s * common + t * next_modulus != g:
                return FALLBACK
            common = common * next_modulus if g == 1 else common // g * next_modulus
        if common != modulus or not 0 <= residue < modulus:
            return FALLBACK
        for target, next_modulus in pairs:
            if (residue - target) % next_modulus:
                return FALLBACK
        return _crt_result(residue, modulus)
    except Exception:
        return FALLBACK


# ---------------------------------------------------------------------------------------------------------------
# Primes and factorizations

def is_prime(n: object) -> object:
    if type(n) is not int or _active() is not None:
        return FALLBACK
    try:
        if n < _TRIAL_BOUND:
            if n < 2:
                return FALLBACK if n.bit_length() > _MAX_BITS else False  # ("below", n)
            divisor = _SPF[n]
            if divisor:  # ("factor", n, divisor)
                return False if 1 < divisor < n and n % divisor == 0 else FALLBACK
            return True if trial_prime(n) else FALLBACK  # ("trial", n)
        if n.bit_length() > _PRIME_BITS:
            return FALLBACK
        claimed, certificate = kernel.prime_certificate(n)
        if n >= _SMALL_CERTIFICATES and _size(certificate) > _CERTIFICATE_NODES:
            return FALLBACK
        return claimed if check_prime(n, claimed, certificate) else FALLBACK
    except Exception:
        return FALLBACK


def _size(certificate: object) -> int:
    """The nodes of a certificate, as Budget.inspect counts them (a tuple and each item)."""
    count, pending = 0, [certificate]
    while pending:
        item = pending.pop()
        count += 1
        if type(item) is tuple:
            pending.extend(item)
    return count


def _factors(n: int) -> tuple[tuple[int, int], ...] | None:
    """The certified factorization of n >= 1: the kernel's pairs after v0.7's _check_factor criterion (the primes
    strictly increasing, each exponent at most n's bit length, each prime certified, the product equal to n), or
    None."""
    pairs = kernel.factor(n)
    bits, product, previous, nodes = n.bit_length(), 1, 1, 0
    for prime, exponent in pairs:
        if prime <= previous or not 1 <= exponent <= bits:
            return None
        if prime < _TRIAL_BOUND:
            if not trial_prime(prime):
                return None
        else:
            claimed, certificate = kernel.prime_certificate(prime)
            nodes += _size(certificate) if prime >= _SMALL_CERTIFICATES else _SMALL_NODES
            if not claimed or nodes > _CERTIFICATE_NODES or not check_prime(prime, True, certificate):
                return None
        product *= prime ** exponent
        if product > n:
            return None
        previous = prime
    return pairs if product == n else None


def _factorization(sign: int, factors: tuple[tuple[int, int], ...]) -> IntegerFactorization:
    value = _new(IntegerFactorization)
    _set(value, "sign", sign)
    _set(value, "factors", factors)
    return value


def factor_integer(n: object) -> object:
    if type(n) is not int or _active() is not None:
        return FALLBACK
    try:
        if n == 0 or n.bit_length() > _FACTOR_BITS:
            return FALLBACK
        factors = _factors(-n if n < 0 else n)
        if factors is None:
            return FALLBACK
        return _factorization(-1 if n < 0 else 1, factors)
    except Exception:
        return FALLBACK


def divisors(n: object) -> object:
    if type(n) is not int or _active() is not None:
        return FALLBACK
    try:
        if n == 0 or n.bit_length() > _FACTOR_BITS:
            return FALLBACK
        m = -n if n < 0 else n
        factors = _factors(m)
        if factors is None:
            return FALLBACK
        count = prod(exponent + 1 for _, exponent in factors)
        if count > _MAX_DIVISORS:
            return FALLBACK
        values = [1]
        for prime, exponent in factors:
            previous, power = values, 1
            for _ in range(exponent):
                power *= prime
                values = values + [divisor * power for divisor in previous]
        values.sort()
        # v0.7's check: as many positive divisors of n as the certified factorization counts, strictly increasing.
        if len(values) != count or values[0] != 1:
            return FALLBACK
        previous = 0
        for divisor in values:
            if divisor <= previous or m % divisor:
                return FALLBACK
            previous = divisor
        return tuple(values)
    except Exception:
        return FALLBACK


def totient(n: object) -> object:
    if type(n) is not int or _active() is not None:
        return FALLBACK
    try:
        if n <= 0 or n.bit_length() > _FACTOR_BITS:
            return FALLBACK
        factors = _factors(n)
        if factors is None:
            return FALLBACK
        value = n
        for prime, _ in factors:  # v0.7's producer: n times (1 - 1/p) for each prime p.
            value = value // prime * (prime - 1)
        expected = 1
        for prime, exponent in factors:  # v0.7's check: the product of the totients p**(e - 1) (p - 1).
            expected *= prime ** (exponent - 1) * (prime - 1)
        return value if value == expected else FALLBACK
    except Exception:
        return FALLBACK


LEAN = {"integer_gcd": integer_gcd, "integer_lcm": integer_lcm, "integer_xgcd": integer_xgcd,
        "mod_inverse": mod_inverse, "crt": crt, "is_prime": is_prime, "factor_integer": factor_integer,
        "divisors": divisors, "totient": totient}
