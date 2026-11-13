"""Independent checker of the family mathforge.areas.ntheory.modular: powmod, multiplicative_order, primitive_root,
legendre_symbol, jacobi_symbol and sqrt_mod.

The checker never reruns the producer's algorithms. It verifies each claim by an identity, a certificate or a second
algorithm, with modular powers of its own (certify.ntheory.factorization.charged_pow, paid for before the builtin
runs), prime proofs (prime_proof) and certified factorizations (check_factorization). Every argument must be an exact
int (type(x) is int: a bool, a float or a str is no integer), the value an int or a tuple of ints in the canonical
form below, and the evidence exactly the layout below (a list is no tuple); anything else answers False, and so does
a claim on an input outside the operation's domain, which has no value at all. Work past the budget raises
ResourceLimitError, which leaves a claim unverified, never verified. Every loop and every remainder of long integers
is charged before it runs, in the units of the cost model of trusted.integers (multiplication_work, division_work).

The theorem that makes True imply the claim, for each operation:

powmod(base, exponent, modulus) = v, evidence ().
    modulus >= 1 and 0 <= v < modulus. For exponent >= 0, v is base**exponent modulo modulus as the builtin pow computes
    it, an implementation other than the producer's square-and-multiply loop (it gives 0**0 = 1, and 0 modulo 1). For
    exponent < 0, v * base**(-exponent) = 1 (mod modulus): base**(-exponent) is then a unit with the inverse v, so base
    is a unit (a prime that divided base and modulus would divide base**(-exponent)), and v is
    (base**(-exponent))**-1 = (base**-1)**(-exponent), unique in [0, modulus). A base that is not a unit admits no v.

multiplicative_order(a, n) = d, evidence (factors,).
    factors is the certified factorization of d (check_factorization: its primes are all the primes of d), () for
    d = 1. When a**d = 1 (mod n) and a**(d/q) != 1 (mod n) for every prime q of d, d is the order of a: a is a unit
    (a * a**(d - 1) = 1), its order k divides d, and k < d would make k divide d/q for a prime q of d/k, so that
    a**(d/q) = 1. A non-unit a has no d with a**d = 1. Modulo n = 1 every power is 1 = 0, and d = 1 alone passes: a
    prime q of a d > 1 gives a**(d/q) = 1. 1 <= d <= max(1, n - 1) is tested first (an order is at most
    phi(n) <= n - 1), so that a forged giant d costs nothing.

primitive_root(p) = g, evidence (factors, refutations).
    1 <= g < p, factors is the certified factorization of p - 1, and refutations holds one prime of p - 1 for each
    h = 2, ..., g - 1, in this order. Lucas's theorem: g**(p - 1) = 1 (mod p) and g**((p - 1)/q) != 1 (mod p) for every
    prime q of p - 1 give g the order p - 1 in (Z/p)^*, a group of phi(p) <= p - 1 elements, so phi(p) = p - 1: p is
    prime and g generates the group. The refutation q of h, with h**((p - 1)/q) = 1 (mod p), bounds the order of h by
    (p - 1)/q < p - 1, so no h = 2, ..., g - 1 generates (Z/p)^*, nor does 1 when p > 2: g is the smallest generator.
    For p = 2, g = 1 with ((), ()) passes the same test, and 1 does generate the trivial group (Z/2)^*. A composite p
    fails Lucas's test whatever the evidence. A refutation may be any prime of p - 1 that refutes h.

legendre_symbol(a, p) = s, evidence (certificate,).
    p is odd and prime_proof(p, certificate) holds. Euler's criterion for an odd prime p: s is 0 when p divides a, and
    otherwise a**((p - 1)/2) is 1 or -1 modulo p and s is that sign. An even or composite p has no Legendre symbol.

jacobi_symbol(a, n) = s, evidence ().
    n is odd and >= 1. An N-version check, with the same laws of the Jacobi symbol as the producer's binary algorithm
    but another algorithm, least absolute remainders: (a/n) depends on a mod n only; (-1/n) = (-1)**((n - 1)/2);
    (2/n) = (-1)**((n*n - 1)/8); for odd coprime a, n >= 1, (a/n) = (n/a) * (-1)**((a - 1)/2 * (n - 1)/2), and both
    sides are 0 when a and n share a prime; (0/n) is 1 for n = 1 and 0 otherwise. A pass replaces a by n - a when
    2a > n, which leaves 0 < a < n/2, and continues with (n mod a, a) once a is odd, so n at least halves per pass:
    at most log2(n) + 1 passes. Two anchors back it: s = 0 exactly when gcd(a, n) > 1 (math.gcd, a third algorithm),
    and for a prime n below psi_13, which prime_proof decides without a certificate, s equals Euler's criterion.

sqrt_mod(a, p) = roots, evidence (certificate,).
    prime_proof(p, certificate) holds, so Z/p is a field, where X**2 - a has at most two roots. p = 2: x*x = x for
    every x, so a % 2 is the only root and roots = (a % 2,). An odd p that divides a: x*x = 0 forces x = 0, and
    roots = (0,). roots = (r, p - r) with 0 < r < p - r, r*r = a and (p - r)**2 = a (mod p): two distinct roots,
    hence all of them, sorted. roots = () needs p > 2 and a**((p - 1)/2) = -1 (mod p): a root x would give
    a**((p - 1)/2) = x**(p - 1) = 1 by Fermat's little theorem. No other shape passes.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer. The family defines no record class (every value is an int or a tuple of ints), so the
checker names none and tests each value's exact type instead.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

from mathforge.limits import Budget, computation
from mathforge.trusted.integers import division_work, multiplication_work
from mathforge.trusted.primes import DETERMINISTIC_BELOW

from .factorization import charged_pow, check_factorization, prime_proof, primes_of

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "powmod": ("base", "exponent", "modulus"),
    "multiplicative_order": ("a", "n"),
    "primitive_root": ("p",),
    "legendre_symbol": ("a", "p"),
    "jacobi_symbol": ("a", "n"),
    "sqrt_mod": ("a", "p"),
}
_SYMBOLS = (-1, 0, 1)  # The values of the Legendre and the Jacobi symbol.


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for arguments other than the operation's exact ints and for any malformed value
    or evidence; ResourceLimitError past the budget."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = _ARGUMENTS.get(operation)
    if names is None or len(args) != len(names) or any(type(args.get(name)) is not int for name in names):
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        return _CHECKS[operation](budget, *(args[name] for name in names), value, evidence)


def _reduced(budget: Budget, number: int, modulus: int) -> int:
    """number % modulus, in [0, modulus) for a modulus >= 1, its long division charged before it runs."""
    budget.tick(1 + division_work(number.bit_length(), modulus.bit_length()))
    return number % modulus


def _euler(budget: Budget, a: int, p: int) -> int | None:
    """Euler's criterion modulo an odd p >= 3: 0 when p divides a, else 1 or -1 as a**((p - 1)/2) is 1 or -1 modulo
    p; None when the power is neither, which no prime p allows."""
    residue = _reduced(budget, a, p)
    if residue == 0:
        return 0
    power = charged_pow(residue, (p - 1) >> 1, p)
    return 1 if power == 1 else -1 if power == p - 1 else None


def _jacobi(budget: Budget, a: int, n: int) -> int:
    """(a/n) for an odd n >= 1 and 0 <= a < n, by least absolute remainders (the module states the laws it applies)."""
    sign = 1
    while a:
        budget.tick(1 + (n.bit_length() >> 10))  # The linear operations of the pass on n: doubling, comparing, ...
        if 2 * a > n:  # (a/n) = (-1/n)((n - a)/n), and (-1/n) = -1 exactly when n = 3 (mod 4).
            a = n - a
            if n % 4 == 3:
                sign = -sign
        shift = (a & -a).bit_length() - 1  # The power of 2 in a, which lies in (0, n/2) now.
        a >>= shift
        if shift % 2 == 1 and n % 8 in (3, 5):  # (2/n) = -1 exactly when n = 3 or 5 (mod 8).
            sign = -sign
        if a % 4 == 3 and n % 4 == 3:  # Reciprocity of the odd a and n.
            sign = -sign
        # The long division n % a takes about one word step per pair of quotient and divisor words. division_work
        # charges the dividend's words for the quotient's, right for a reduction after a product; here the quotient
        # is mostly a word or two, so the pass pays for the words it has (a symbol of 65,530 bits: about a million
        # units, half the default max_work, measured at a third of a second).
        budget.tick(1 + division_work(n.bit_length() - a.bit_length() + 64, a.bit_length()))
        a, n = n % a, a  # (n/a) = ((n mod a)/a), and the new n = a < n/2.
    return sign if n == 1 else 0


def _powmod(budget: Budget, base: int, exponent: int, modulus: int, value: object, evidence: object) -> bool:
    if modulus < 1 or type(value) is not int or not 0 <= value < modulus or type(evidence) is not tuple or evidence:
        return False
    if exponent >= 0:
        return charged_pow(base, exponent, modulus) == value  # charged_pow also pays for reducing base.
    power = charged_pow(base, -exponent, modulus)
    budget.tick(multiplication_work(modulus.bit_length()))
    return value * power % modulus == 1 % modulus


def _multiplicative_order(budget: Budget, a: int, n: int, value: object, evidence: object) -> bool:
    if (n < 1 or type(value) is not int or not 1 <= value <= max(1, n - 1) or type(evidence) is not tuple
            or len(evidence) != 1 or not check_factorization(value, evidence[0])):
        return False
    residue, one = _reduced(budget, a, n), 1 % n
    if charged_pow(residue, value, n) != one:
        return False
    for prime in primes_of(evidence[0]):
        budget.tick(1 + division_work(value.bit_length(), prime.bit_length()))  # The quotient value // prime.
        if charged_pow(residue, value // prime, n) == one:
            return False
    return True


def _primitive_root(budget: Budget, p: int, value: object, evidence: object) -> bool:
    if p < 2 or type(value) is not int or not 1 <= value < p or type(evidence) is not tuple or len(evidence) != 2:
        return False
    factors, refutations = evidence
    if (type(refutations) is not tuple or len(refutations) != max(0, value - 2)
            or not check_factorization(p - 1, factors)):
        return False
    order = p - 1
    primes = primes_of(factors)
    # Lucas's test: value has the order p - 1 modulo p, so p is prime and value generates (Z/p)^*.
    if charged_pow(value, order, p) != 1:
        return False
    for prime in primes:
        budget.tick(1 + division_work(order.bit_length(), prime.bit_length()))  # The quotient order // prime.
        if charged_pow(value, order // prime, p) == 1:
            return False
    # Minimality: every h = 2, ..., value - 1 has an order below p - 1, shown by the prime of p - 1 that refutes it.
    refuting = frozenset(primes)
    for h, prime in enumerate(refutations, 2):
        budget.tick()
        if type(prime) is not int or prime not in refuting:
            return False
        budget.tick(1 + division_work(order.bit_length(), prime.bit_length()))
        if charged_pow(h, order // prime, p) != 1:
            return False
    return True


def _legendre_symbol(budget: Budget, a: int, p: int, value: object, evidence: object) -> bool:
    if (p < 3 or p % 2 == 0 or type(value) is not int or value not in _SYMBOLS or type(evidence) is not tuple
            or len(evidence) != 1 or not prime_proof(p, evidence[0])):
        return False
    return value == _euler(budget, a, p)


def _jacobi_symbol(budget: Budget, a: int, n: int, value: object, evidence: object) -> bool:
    if (n < 1 or n % 2 == 0 or type(value) is not int or value not in _SYMBOLS or type(evidence) is not tuple
            or evidence):
        return False
    residue = _reduced(budget, a, n)
    if value != _jacobi(budget, residue, n):
        return False
    words = (n.bit_length() >> 6) + 1
    # math.gcd of two integers below n, Lehmer's algorithm: quadratic, about half a division_work of n by n.
    budget.tick(1 + (words * words >> 6))
    if (value == 0) != (gcd(residue, n) > 1):
        return False
    if 3 <= n < DETERMINISTIC_BELOW and prime_proof(n, ()):  # Below psi_13, prime_proof needs no certificate.
        return value == _euler(budget, residue, n)
    return True


def _sqrt_mod(budget: Budget, a: int, p: int, value: object, evidence: object) -> bool:
    if (p < 2 or type(value) is not tuple or len(value) > 2 or any(type(root) is not int for root in value)
            or type(evidence) is not tuple or len(evidence) != 1 or not prime_proof(p, evidence[0])):
        return False
    residue = _reduced(budget, a, p)
    if p == 2 or residue == 0:  # Modulo 2, x*x = x; modulo a prime, x*x = 0 forces x = 0.
        return value == (residue,)
    if len(value) == 2:
        low, high = value
        return (0 < low < high and low + high == p and charged_pow(low, 2, p) == residue
                and charged_pow(high, 2, p) == residue)
    return not value and charged_pow(residue, (p - 1) >> 1, p) == p - 1  # Euler: a**((p - 1)/2) = -1, no root.


_CHECKS: dict[str, Callable[..., bool]] = {
    "powmod": _powmod,
    "multiplicative_order": _multiplicative_order,
    "primitive_root": _primitive_root,
    "legendre_symbol": _legendre_symbol,
    "jacobi_symbol": _jacobi_symbol,
    "sqrt_mod": _sqrt_mod,
}
