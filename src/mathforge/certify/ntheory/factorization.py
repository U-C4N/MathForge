"""Prime proofs and certified factorizations for the checkers of the number theory families, and a modular power that
is charged to the computation budget.

A checker never takes from its evidence that an integer is prime, or that some primes are the prime factors of an
integer: it asks this module, which proves it or answers False.

``prime_proof(n, certificate)`` is True only when n is proved prime, by the method that the size of n selects:

* 2 <= n < 10,000 (TRIAL_BELOW): the trial criterion, certify.ntheory.primes.trial_prime. n is prime exactly when no
  d in 2 ... isqrt(n) divides it, and isqrt(n) is at most 99, so a gcd with the product of the primes up to 97 (up to
  7 below 98) decides it.
* 10,000 <= n < psi_13 = 3,317,044,064,679,887,385,961,981 (trusted.primes.DETERMINISTIC_BELOW):
  trusted.primes.is_prime, the strong probable-prime test to the thirteen prime bases 2 to 41. Every prime passes it,
  and psi_13 is the smallest composite that does (Sorenson and Webster, "Strong pseudoprimes to twelve prime bases",
  Mathematics of Computation 86 (2017), 985-1003; OEIS A014233), so an integer below psi_13 that passes is prime.
* n >= psi_13: a v0.7 Pocklington certificate ("pocklington", n, ((p, e, certificate_p), ...), (w_p, ...)), which
  certify.ntheory.primes.check_prime verifies. Pocklington's theorem, with n - 1 factored completely: when n - 1 is
  the product of the p**e, every p is prime, and every p has a witness w with w**(n - 1) = 1 modulo n and
  gcd(w**((n - 1)/p) - 1, n) = 1, then n is prime. (Modulo a prime q that divides n, the order of w divides n - 1
  but not (n - 1)/p, so p**e divides that order, which divides q - 1: n - 1 divides q - 1, and q is n.)

The evidence is canonical: the certificate is () exactly where the size of n needs none, below psi_13, and a
Pocklington certificate from psi_13 on, whose tag is the exact str "pocklington" (an instance of a str subclass is no
tag). A prime that comes with any other certificate is refused. The certificates nested in a Pocklington certificate
keep v0.7's format, which check_prime verifies: ("trial", p) for a p below 10,000 or a Pocklington certificate of p,
never (); the producers build ("trial", p) below 10,000 and Pocklington certificates from there on.

``check_factorization(n, factors)`` is True only when factors, ((p, e, certificate), ...), is the prime factorization
of n >= 1: the p strictly increasing, every e an integer >= 1, every p proved prime by prime_proof, and the product
of the p**e equal to n. By the fundamental theorem of arithmetic n has no other factorization into primes, so the p
are all the primes that divide n and each e is the exact power of its p in n. 1 has the empty factorization.

``charged_pow(base, exponent, modulus)`` is the builtin pow for a checker's own exponentiations. The builtin cannot
be interrupted, so its whole work is charged before it starts.

Work. Every charge goes to the active budget, or to a budget of the default limits opened for the call when none is
active, and it is made before the operation it pays for. The amounts come from the cost model of trusted.integers
(multiplication_work, division_work, prime_proof_work, factorization_entry_work), which states the units. The
producers of kernels.certificates compute with the same functions what this module will charge for the evidence they
build, and refuse evidence that would run its check out of the budget. check_prime charges nothing itself, so
prime_proof first charges every exponentiation and gcd that check_prime can run on the certificate.
"""

from __future__ import annotations

from mathforge.errors import InvalidInput
from mathforge.limits import computation
from mathforge.trusted.integers import (TRIAL_BELOW, division_work, factorization_entry_work, multiplication_work,
                                        prime_proof_work)
from mathforge.trusted.primes import DETERMINISTIC_BELOW, is_prime

from .primes import check_prime, trial_prime

__all__ = ["TRIAL_BELOW", "charged_pow", "check_factorization", "multiplication_work", "prime_proof", "primes_of"]


def charged_pow(base, exponent, modulus) -> int:
    """pow(base, exponent, modulus), the residue in [0, modulus), for integers with exponent >= 0 and modulus >= 1,
    charged to the computation budget before the builtin runs.

    The charge is one unit, a multiplication modulo the modulus (multiplication_work) per bit of the exponent, and the
    reduction of the base, a long division (division_work). Past max_work the call raises ResourceLimitError and
    computes nothing. Without an active budget the call opens one of the default limits, so it is never unbounded.

    An argument that is not an exact int (a bool, a float, a Rational, a str), a negative exponent or a modulus below
    1 raises InvalidInput, before any charge. A checker tests its inputs before it calls, or catches InvalidInput and
    answers False; an InvalidInput that escapes a checker leaves its claim unverified, never verified.
    """
    if type(base) is not int or type(exponent) is not int or type(modulus) is not int:
        raise InvalidInput("A modular power takes exact integers")
    if exponent < 0 or modulus < 1:
        raise InvalidInput("A modular power takes an exponent >= 0 and a modulus >= 1")
    bits = modulus.bit_length()
    work = 1 + exponent.bit_length() * multiplication_work(bits) + division_work(base.bit_length(), bits)
    with computation() as budget:
        budget.tick(work)
        return pow(base, exponent, modulus)


def prime_proof(n, certificate) -> bool:
    """Whether n is proved prime with the canonical certificate of its size: () for n below psi_13 (the trial
    criterion below 10,000, then the deterministic strong tests), a Pocklington certificate from psi_13 on.

    False for an n that is not an exact int (a bool included), for a certificate that is not a tuple, for a
    certificate where none is needed and for any other shape, whatever n is: the tag of a Pocklington certificate is
    the exact str "pocklington", never an instance of a subclass of str that compares equal to it. The work that
    trusted.integers.prime_proof_work computes is charged before the proof runs, and past max_work the call raises
    ResourceLimitError. The module states the theorems."""
    if type(n) is not int or type(certificate) is not tuple or n < 2:
        return False
    if n >= DETERMINISTIC_BELOW:
        if len(certificate) != 4 or type(certificate[0]) is not str or certificate[0] != "pocklington":
            return False
    elif certificate:
        return False
    elif n < TRIAL_BELOW:
        return trial_prime(n)  # A gcd with a 121-bit constant: the one proof that needs no charge.
    with computation() as budget:
        # The walk of the certificate stops once it passes what the budget has left, so tick then refuses it.
        budget.tick(prime_proof_work(n, certificate, budget.limits.max_work - budget.work))
        return is_prime(n) if n < DETERMINISTIC_BELOW else check_prime(n, True, certificate)


def check_factorization(n, factors) -> bool:
    """Whether factors is the certified prime factorization of n: a tuple of (p, e, certificate) tuples with the p
    strictly increasing, every e an exact int >= 1, the product of the p**e equal to the exact int n >= 1 and every p
    proved prime by prime_proof(p, certificate). True means that the p are exactly the primes that divide n.

    The product is checked before any prime proof runs, and no power or partial product much longer than n is ever
    built: a power that must exceed n is refused by its bit length before it exists, so a forged exponent costs
    nothing. An n over max_integer_bits, and work past max_work, raise ResourceLimitError; on a factorization that it
    accepts, the check charges trusted.integers.factorization_work."""
    if type(n) is not int or n < 1 or type(factors) is not tuple:
        return False
    with computation() as budget:
        budget.check_int(n)
        bits = n.bit_length()
        product, previous = 1, 1
        for item in factors:
            if type(item) is not tuple or len(item) != 3:
                return False
            prime, exponent = item[0], item[1]
            if type(prime) is not int or prime <= previous or type(exponent) is not int or exponent < 1:
                return False
            prime_bits = prime.bit_length()
            # prime**exponent >= 2**((prime_bits - 1) * exponent), which is above n as soon as that reaches 2**bits.
            if (prime_bits - 1) * exponent >= bits:
                return False
            power_bits = prime_bits * exponent  # At least the length of prime**exponent, and below 2 * bits.
            budget.tick(factorization_entry_work(product.bit_length(), power_bits))
            power = prime ** exponent
            if product.bit_length() + power.bit_length() - 1 > bits:  # Then product * power >= 2**bits > n.
                return False
            product *= power
            if product > n:
                return False
            previous = prime
        if product != n:
            return False
        for prime, _, certificate in factors:
            if not prime_proof(prime, certificate):
                return False
        return True


def primes_of(factors) -> tuple:
    """The primes of a factorization that check_factorization has accepted, in increasing order."""
    return tuple([item[0] for item in factors])
