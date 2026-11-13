"""Producer-side evidence of the number theory families: the prime certificates and the certified factorizations
that mathforge.certify.ntheory.factorization checks.

``prime_evidence(n)`` is the certificate that prime_proof expects for a prime n: () below
trusted.primes.DETERMINISTIC_BELOW (psi_13, up to which the checker decides primality itself) and the integer
kernel's Pocklington certificate from there on. ``factorization_evidence(n)`` is ((p, e, certificate), ...) for
n >= 1: the prime factorization of n with such a certificate for each prime, the shape check_factorization accepts.

This is producer code: nothing here is a proof, and evidence proves its claim only once the checker has accepted it.
A family imports this module inside produce(), never at its top level, so that importing mathforge loads no kernel.

Work. Both functions run in the active computation budget, or in a budget of the default limits opened for the call
when none is active, and refuse an n longer than max_integer_bits with ResourceLimitError before anything else. The
integer kernel charges its work to that budget, and its ResourceLimitError propagates: from psi_13 on a prime needs
the factorization of n - 1, which the budget may not deliver. factorization_evidence removes the primes below 2**10
itself, the power of 2 by its lowest set bit and any other power p**e with about 2 log2(e) long divisions, each
charged before it runs, and hands the integer kernel only the rest: the kernel's trial division removes one factor at
a time and charges for it afterwards, at the size of what is left, so 2**65535 cost it 65,535 divisions for 65,536
units.

Before they return, both compute what the checker will charge for the evidence, with the functions that the checker
charges with (trusted.integers.prime_proof_work and factorization_work), and refuse the evidence with
ResourceLimitError when that is more than max_work, in an active budget and without one alike. The check never runs
in what the production left of the budget: operations_v6.run's verifier checks in a fresh budget of the request's
limits, the lean runner in one of the default limits, so production and check need not fit in max_work together.
prime_proof and check_factorization therefore never run out of work on evidence that they return, under the same
limits in a fresh budget (a family's checker that does more must leave room for it). Measured with the default limits
on k * 2**m + 1 primes, the check is the limit, at about 4,000 bits: 3 * 2**3912 + 1 passes with 1,945,693 units of
production and 1,832,809 of check, 5 * 2**4687 + 1 needs 2,350,863 units of check. Random primes stop far earlier:
the kernel must factor p - 1.
"""

from __future__ import annotations

from math import gcd, prod

from ..errors import InvalidInput
from ..limits import Budget, ResourceLimitError, computation
from ..trusted.integers import division_work, factorization_work, prime_proof_work, product_work
from ..trusted.primes import DETERMINISTIC_BELOW
from . import integers as kernel

__all__ = ["factorization_evidence", "prime_evidence"]

_ODD_SMALL_PRIMES = kernel.SMALL_PRIMES[1:]  # The odd primes below 2**10, which factorization_evidence removes itself.
_ODD_SMALL_PRODUCT = prod(_ODD_SMALL_PRIMES)
_UNCHECKABLE = "Computation limit exceeded: work ({}) for the independent check of the evidence"


def prime_evidence(n: int) -> tuple:
    """The certificate that certify.ntheory.factorization.prime_proof accepts for the prime n: () below psi_13, the
    Pocklington certificate ("pocklington", n, ((p, e, certificate_p), ...), (w_p, ...)) from psi_13 on.

    InvalidInput when n is not an exact int or is not prime. ResourceLimitError when n is longer than
    max_integer_bits, when the budget cannot certify n, and when prime_proof would need more than max_work for the
    certificate (see the module)."""
    if type(n) is not int:
        raise InvalidInput("An exact integer is required")
    if n < 2:
        raise InvalidInput("The integer is not prime")
    with computation() as budget:
        budget.check_int(n)
        certificate: tuple = ()
        if n < DETERMINISTIC_BELOW:
            if not kernel.is_probable_prime(n):  # Miller-Rabin is deterministic in this range.
                raise InvalidInput("The integer is not prime")
        else:
            proven, certificate = kernel.prime_certificate(n)
            if not proven:
                raise InvalidInput("The integer is not prime")
        allowance = _check_allowance(budget)
        if prime_proof_work(n, certificate, allowance) > allowance:
            raise ResourceLimitError(_UNCHECKABLE.format(budget.limits.max_work))
        return certificate


def factorization_evidence(n: int) -> tuple[tuple[int, int, tuple], ...]:
    """The certified factorization of n >= 1 that certify.ntheory.factorization.check_factorization accepts:
    ((p, e, certificate), ...) with the primes increasing and each certificate as prime_evidence gives it; () for 1.

    InvalidInput when n is not an exact int >= 1. ResourceLimitError when n is longer than max_integer_bits, when the
    budget cannot factor n or certify one of its primes, and when check_factorization would need more than max_work
    for the evidence (see the module)."""
    if type(n) is not int or n < 1:
        raise InvalidInput("An exact positive integer is required")
    with computation() as budget:
        budget.check_int(n)
        pairs, rest = _small_prime_powers(n, budget)
        if rest > 1:
            pairs.extend(kernel.factor(rest))  # Primes above 2**10, in increasing order after those below.
        evidence = []
        for prime, exponent in pairs:
            certificate: tuple = ()
            if prime >= DETERMINISTIC_BELOW:  # Below it the checker needs no certificate, and factor's test is exact.
                proven, certificate = kernel.prime_certificate(prime)
                if not proven:
                    raise ArithmeticError("Internal factorization produced a composite factor")
            evidence.append((prime, exponent, certificate))
        factors = tuple(evidence)
        allowance = _check_allowance(budget)
        if factorization_work(factors, allowance) > allowance:
            raise ResourceLimitError(_UNCHECKABLE.format(budget.limits.max_work))
        return factors


def _check_allowance(budget: Budget) -> int:
    """The work that the check of the evidence may need: all of max_work, whatever the production spent and whether
    or not the caller's budget was active. The checker runs in a fresh budget of the same limits (operations_v6.run's
    verifier, the lean runner), never in what the production left of this one."""
    return budget.limits.max_work


def _small_prime_powers(n: int, budget: Budget) -> tuple[list[tuple[int, int]], int]:
    """([(p, e), ...], rest): the exact powers p**e in n >= 1 of the primes p below 2**10, in increasing order, and
    rest = n / prod(p**e), which has no prime factor below 2**10. Every step is charged before it runs."""
    pairs: list[tuple[int, int]] = []
    if not n & 1:
        budget.tick(1 + (n.bit_length() >> 6))  # A pass over the words of n for its lowest set bit, one to shift.
        exponent = (n & -n).bit_length() - 1
        n >>= exponent
        pairs.append((2, exponent))
    if n == 1:
        return pairs, n
    budget.tick(1 + division_work(n.bit_length(), _ODD_SMALL_PRODUCT.bit_length()) + len(_ODD_SMALL_PRIMES))
    divisors = gcd(n, _ODD_SMALL_PRODUCT)  # The product of the odd primes below 2**10 that divide n.
    for p in _ODD_SMALL_PRIMES:
        if divisors == 1:
            break
        if divisors % p == 0:  # The divisions of the at most 1,419-bit gcd, one unit each in the charge above.
            divisors //= p
            n, exponent = _remove_power(n, p, budget)
            pairs.append((p, exponent))
    return pairs, n


def _remove_power(n: int, p: int, budget: Budget) -> tuple[int, int]:
    """(n / p**e, e) for the exact power p**e of p >= 2 in n >= 1, with about 2 log2(e) long divisions rather than e.

    It divides by p, p**2, p**4, ... while they divide: after k divisions p**(2**k - 1) is gone, and the power of p in
    what is left is below 2**k, since p**(2**k) does not divide it (or exceeds it). Then it divides by p**(2**(k-1)),
    ..., p**2, p, each where it divides, which removes that rest by the binary digits of its exponent. Each division
    and squaring is charged before it runs, at the size of the numbers it takes."""
    powers: list[int] = []
    exponent, power = 0, p
    while True:
        budget.tick(1 + division_work(n.bit_length(), power.bit_length()))
        quotient, remainder = divmod(n, power)
        if remainder:
            break
        n, exponent = quotient, exponent + (1 << len(powers))
        powers.append(power)
        if 2 * power.bit_length() - 2 >= n.bit_length():  # power**2 >= 2**(2 * bits - 2) > n: it cannot divide n.
            break
        budget.tick(1 + product_work(power.bit_length(), power.bit_length()))
        power *= power
    for index in range(len(powers) - 1, -1, -1):
        budget.tick(1 + division_work(n.bit_length(), powers[index].bit_length()))
        quotient, remainder = divmod(n, powers[index])
        if not remainder:
            n, exponent = quotient, exponent + (1 << index)
    return n, exponent
