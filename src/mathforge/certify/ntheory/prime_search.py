"""Independent checker of the family mathforge.areas.ntheory.prime_search: next_prime(n) and prev_prime(n).

Both values are plain ints (type(value) is int: no bool, float, Rational or record), so there is no record class to
identify. The evidence of both operations is the pair (certificate, witnesses):

* certificate proves the value r prime through factorization.prime_proof(r, certificate): () below
  psi_13 = 3,317,044,064,679,887,385,961,981 and a v0.7 Pocklington certificate from psi_13 on.
* witnesses is a tuple of entries ("fermat", m, w) or ("factor", m, d), one for each skipped integer m that the
  checker cannot show composite on its own (criterion 4 below), in increasing order of m, and nothing else.

The skipped integers are those strictly between n and r: n < m < r for next_prime, r < m < n for prev_prime. The
checker enumerates them itself, never from the evidence, and shows each one composite by the first criterion that
applies; each criterion is a theorem:

1. m < 10,000: the trial criterion, certify.ntheory.primes.trial_prime, is exact there (a composite below 10,000 has
   a prime factor up to isqrt(9,999) = 99), so m is composite exactly when it fails. No witness is ever needed or
   accepted below 10,000, and the gcd of criterion 2 is never applied there: gcd(3, 2*3*...*97) = 3, yet 3 is prime.
2. m >= 10,000 and gcd(m, 2*3*...*97) > 1: a prime p <= 97 < m divides m.
3. m >= 10,000 and 2**(m - 1) is not 1 modulo m: by Fermat's little theorem a prime m >= 10,000, which is odd, has
   2**(m - 1) = 1 modulo m.
4. Otherwise the next witness of the evidence must name m and verify: ("factor", m, d) with 1 < d < m and d dividing
   m, or ("fermat", m, w) with 1 < w < m and w**(m - 1) not 1 modulo m (a prime m is prime to every 1 < w < m, and
   then w**(m - 1) = 1 modulo m by Fermat's little theorem). Such an m is a prime, for which no witness verifies, or a
   base-2 Fermat pseudoprime without a prime factor up to 97, such as 31621 = 103 * 307, psi_13, or the Carmichael
   number 56052361 = 211 * 421 * 631, which has a Fermat witness only among the bases that share a factor with it.

The witnesses must be exactly the entries that criterion 4 consumes, in the order of m: an extra entry (one for an m
that criteria 1 to 3 decide, or outside the skipped range), a missing one, a repeated one or one out of order is
refused. So the evidence of a claim is canonical up to the choice of each witness's kind and number.

Theorems. next_prime(n) = r: for n < 2, r must be 2, which prime_proof proves prime, and no integer below 2 is prime,
so 2 is the smallest prime above n; nothing is enumerated and the evidence is ((), ()). For n >= 2: r > n, r is
proved prime and every integer strictly between n and r is composite, so r is the smallest prime greater than n.
prev_prime(n) = r: for n >= 3: 2 <= r < n, r is proved prime and every integer strictly between r and n is composite,
so r is the largest prime smaller than n. For n <= 2 no prime lies below n (the producer rejects the input), and the
checker refuses every claim.

Work. Every charge goes to the active budget, or to a budget of the default limits opened for the call, before the
work that it pays for. n and r are first held to max_integer_bits. Then, before anything is enumerated, the whole
scan's trial tests and gcds are paid for at once: one unit, plus one per 64-bit word of the larger end of the range
(r for next_prime, n for prev_prime), for each skipped integer. A claimed value farther away than the budget allows
therefore fails at once with ResourceLimitError, as the card prescribes, before the certificate is looked at. Then
prime_proof charges the proof of r. Every exponentiation goes through factorization.charged_pow, which charges it
before the builtin runs: the base-2 test of criterion 3 for each m >= 10,000 prime to 2*3*...*97, and each Fermat
witness. A factor witness is charged factorization.multiplication_work(bits(m)) before its division. A malformed or
wrong claim answers False; the checker raises only ResourceLimitError, for an integer over max_integer_bits or for
work past max_work.
"""

from __future__ import annotations

from math import gcd, prod

from mathforge.limits import Budget, computation

from .factorization import TRIAL_BELOW, charged_pow, multiplication_work, prime_proof
from .primes import trial_prime

_OPERATIONS = ("next_prime", "prev_prime")
# The primes up to 97, the largest prime up to isqrt(9,999) = 99. Their product is the gcd screen of criterion 2.
# Which skipped integers need a witness depends on it, so the producer must use exactly the same product.
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
_PRIMORIAL = prod(_SMALL_PRIMES)


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value is next_prime(n) or prev_prime(n), as operation names it, for args = {"n": n}, proved by
    evidence = (certificate, witnesses). The module states the theorems. False for any malformed or wrong claim;
    ResourceLimitError only for an integer over max_integer_bits or for work past max_work, as for a claimed value
    farther away than the budget can scan."""
    if type(operation) is not str or operation not in _OPERATIONS or type(args) is not dict or args.keys() != {"n"}:
        return False
    n = args["n"]
    if type(n) is not int or type(value) is not int or type(evidence) is not tuple or len(evidence) != 2:
        return False
    certificate, witnesses = evidence
    if type(certificate) is not tuple or type(witnesses) is not tuple:
        return False
    with computation() as budget:
        if operation == "next_prime":
            if n < 2:
                # 2 is the smallest prime, and every integer between n and 2 lies below 2: nothing is enumerated.
                return value == 2 and not witnesses and prime_proof(2, certificate)
            if value <= n:
                return False
            low, high, top = n + 1, value, value
        else:
            if not 2 <= value < n:  # No prime lies below 2: a claim needs 2 <= value < n, hence n >= 3.
                return False
            low, high, top = value + 1, n, n
        budget.check_int(n)
        budget.check_int(value)
        # Criterion 4 can consume at most one witness for each skipped integer from 10,000 on: longer witness lists
        # are refused before anything is charged.
        if len(witnesses) > max(0, high - max(low, TRIAL_BELOW)):
            return False
        # The scan's trial tests and gcds, paid before it starts: a claimed value that is too far fails here.
        budget.tick((high - low) * (1 + (top.bit_length() >> 6)))
        return prime_proof(value, certificate) and _skipped_are_composite(budget, low, high, witnesses)


def _skipped_are_composite(budget: Budget, low: int, high: int, witnesses: tuple) -> bool:
    """Whether every integer m with 3 <= low <= m < high is shown composite by criteria 1 to 4 of the module, and the
    witnesses are exactly the entries that criterion 4 consumes, in the order of m. The caller has paid for the trial
    tests and the gcds; charged_pow charges every exponentiation."""
    used = 0
    for m in range(low, high):
        if m < TRIAL_BELOW:
            if trial_prime(m):  # A prime below 10,000 lies between n and the value.
                return False
            continue
        if gcd(m, _PRIMORIAL) > 1 or charged_pow(2, m - 1, m) != 1:
            continue
        if used == len(witnesses) or not _witness(budget, m, witnesses[used]):
            return False
        used += 1
    return used == len(witnesses)  # Any entry left over is one that criterion 4 never asked for.


def _witness(budget: Budget, m: int, entry: object) -> bool:
    """Whether entry proves the integer m >= 10,000 composite: ("factor", m, d) with 1 < d < m dividing m, or
    ("fermat", m, w) with 1 < w < m and w**(m - 1) not 1 modulo m."""
    if type(entry) is not tuple or len(entry) != 3:
        return False
    kind, number, witness = entry
    if type(kind) is not str or type(number) is not int or number != m or type(witness) is not int:
        return False
    if not 1 < witness < m:
        return False
    if kind == "factor":
        budget.tick(multiplication_work(m.bit_length()))  # The division, before it runs.
        return m % witness == 0
    if kind == "fermat":
        return charged_pow(witness, m - 1, m) != 1
    return False
