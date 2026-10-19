"""Independent checker of v0.7's prime certificates, for the lean value path of is_prime, factor_integer, divisors and
totient.

It applies the criterion of ``number_theory._check_prime`` to a certificate without factoring n or n - 1, with the
builtin ``pow``:

* ("below", n): n < 2, claimed composite (not prime).
* ("factor", n, d): 1 < d < n and d divides n, claimed composite.
* ("fermat", n, w): n > 2, 1 < w < n and w**(n - 1) is not 1 modulo n, claimed composite.
* ("trial", n): 2 <= n < 10,000 and no d in 2 ... isqrt(n) divides n, claimed prime. Such a d exists exactly when a
  prime up to isqrt(n) <= 99 divides n, which a gcd with the product of the primes up to 97 decides (below 98, the
  primes up to 7 and the product 210).
* ("pocklington", n, ((p, e, certificate_p), ...), (w_p, ...)): claimed prime; the p strictly increasing, each one
  certified prime recursively, the product of the p**e equal to n - 1, and for each p a witness 1 < w_p < n with
  w_p**(n - 1) = 1 and gcd(w_p**((n - 1)/p) - 1, n) = 1. Pocklington's theorem then proves n prime. The powers
  come from mathforge.trusted.integers.powers_by_prime, one product tree per witness.

A certificate nested deeper than the default max_depth is refused, as v0.7 refuses it.
"""

from math import gcd, prod

from mathforge.limits import ComputationLimits
from mathforge.trusted.integers import powers_by_prime

MAX_DEPTH = ComputationLimits().max_depth
_P97 = prod((2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97))


def trial_prime(n):
    """The trial criterion of 2 <= n < 10,000: no d in 2 ... isqrt(n) divides n."""
    if n > 97:
        return gcd(n, _P97) == 1
    return n in (2, 3, 5, 7) or gcd(n, 210) == 1


def check_prime(n, claimed, certificate, depth=0):
    """Whether the certificate proves the claim (True: n is prime, False: n is not) for the integer n."""
    if depth > MAX_DEPTH or type(claimed) is not bool or type(certificate) is not tuple or len(certificate) < 2:
        return False
    kind, recorded = certificate[0], certificate[1]
    if type(recorded) is not int or recorded != n:
        return False
    if kind == "below":
        return len(certificate) == 2 and not claimed and n < 2
    if kind == "factor":
        if len(certificate) != 3 or type(certificate[2]) is not int:
            return False
        divisor = certificate[2]
        return not claimed and 1 < divisor < n and n % divisor == 0
    if kind == "fermat":
        if len(certificate) != 3 or type(certificate[2]) is not int:
            return False
        witness = certificate[2]
        return not claimed and n > 2 and 1 < witness < n and pow(witness, n - 1, n) != 1
    if kind == "trial":
        return len(certificate) == 2 and claimed and 2 <= n < 10_000 and trial_prime(n)
    if kind != "pocklington" or len(certificate) != 4 or not claimed or n < 2:
        return False
    factors, witnesses = certificate[2], certificate[3]
    if type(factors) is not tuple or type(witnesses) is not tuple or len(factors) != len(witnesses) or not factors:
        return False
    m = n - 1
    bits = m.bit_length()
    total, previous, groups = 1, 1, {}
    for factor, witness in zip(factors, witnesses):
        if type(factor) is not tuple or len(factor) != 3:
            return False
        prime, exponent, child = factor
        if type(prime) is not int or prime <= previous or type(exponent) is not int or not 1 <= exponent <= bits:
            return False
        if (prime.bit_length() - 1) * exponent > bits:  # prime**exponent > n - 1: refused before it is built.
            return False
        if not check_prime(prime, True, child, depth + 1):
            return False
        total *= prime ** exponent
        if total > m or type(witness) is not int or not 1 < witness < n:
            return False
        groups.setdefault(witness, []).append(prime)
        previous = prime
    if total != m:
        return False
    # The primes are distinct divisors of n - 1 now: each witness's powers w**((n - 1)/p) come from one product tree,
    # and w**(n - 1) is any of them raised to its p.
    for witness, primes in groups.items():
        powers = powers_by_prime(witness, m, n, primes)
        first = primes[0]
        if pow(powers[first], first, n) != 1:
            return False
        for prime in primes:
            if gcd(powers[prime] - 1, n) != 1:
                return False
    return True
