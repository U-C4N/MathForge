"""Independent checker of the family mathforge.areas.ntheory.discrete: discrete_log and two_squares.

The checker never reruns the producer's algorithms (Pohlig-Hellman with baby-step giant-step; Cornacchia with the
Brahmagupta-Fibonacci identity). It verifies each claim by modular powers of its own
(certify.ntheory.factorization.charged_pow, paid for before the builtin runs), prime proofs (prime_proof) and certified
factorizations (check_factorization). Every argument must be an exact int (type(x) is int: a bool, a float or a str is
no integer), the value a record of the family's class, identified by (type(value).__module__,
type(value).__qualname__) and read as data, in the canonical form below, and the evidence exactly the layout below (a
list is no tuple); anything else answers False, and so does a claim on an input outside the operation's domain, which
has no value at all. An argument over max_integer_bits raises ResourceLimitError, as the producer's budget refuses it,
and so does work past the budget: either leaves a claim unverified, never verified. Every loop, every long division
and every product of long integers is charged before it runs, in the units of the cost model of trusted.integers.

The theorem that makes True imply the claim, for each operation:

discrete_log(base, target, p) = DiscreteLogarithm(exponent, order), evidence (factors, certificate).
    True requires: prime_proof(p, certificate), so p is prime; base % p != 0; order an exact int >= 1 that divides
    p - 1; check_factorization(order, factors), so factors lists every prime q of order (() for order = 1);
    base**order = 1 (mod p) and base**(order/q) != 1 (mod p) for every prime q of order. Then order is the
    multiplicative order d of base modulo p: base is a unit, its order k divides d, and k < d would make k divide d/q
    for a prime q of d/k, so that base**(d/q) = 1. (order | p - 1 then holds for every true claim, by Lagrange's
    theorem, and bounds a forged order by p before anything is computed.)
    exponent an int: 0 <= exponent < order and base**exponent = target (mod p). The powers base**0, ..., base**(d - 1)
    are distinct modulo p (base**i = base**j with 0 <= i < j < d would give base**(j - i) = 1 with 0 < j - i < d), and
    base**k depends on k modulo d only, so the solutions k >= 0 are exactly exponent + t*d, t >= 0, and exponent is
    the least of them.
    exponent None: target**order != 1 (mod p). For the prime p, (Z/p)^* is cyclic of order p - 1, so it has exactly
    one subgroup of each order d | p - 1, which the d-th roots of unity form (X**d - 1 has at most d roots in the field
    Z/p, and the subgroup's d elements are roots). The powers of base form that subgroup for d = order, so target is a
    power of base exactly when target**d = 1 (mod p); target = 0 (mod p) gives 0 != 1, and 0 is no power of a unit.
    So no k solves base**k = target, and None is the claim. A composite p fails prime_proof whatever the evidence.

two_squares(n) = SumOfTwoSquares(count, a, b), evidence (factors,).
    True requires n >= 0 and: for n = 0, factors = () and count = 1; for n >= 1, check_factorization(n, factors), so
    factors lists every prime of n with its exact exponent, and count = 0 when some prime q = 3 (mod 4) has an odd
    exponent, else count = 4 * prod (e + 1) over the primes p = 1 (mod 4) of n (the prime 2 adds nothing). That is
    r2(n) by Jacobi's two-square theorem, r2(n) = 4 (d1(n) - d3(n)) with d_i(n) the number of divisors of n that are
    i modulo 4, whose multiplicative evaluation gives this product; r2(0) = 1, the pair (0, 0) alone. Then, for
    count > 0: exact ints 0 <= a <= b with a*a + b*b = n, one representation (any one passes: the value does not claim
    which); for count = 0: a and b both None. Since count = r2(n) is decided by the factorization alone, a
    representation exists exactly when count > 0, and no other count passes.

Sizes. A forged value or evidence costs at most its screening: an order longer than p, an exponent outside [0, order)
and a b longer than half of n plus a bit (b <= isqrt(n)) are refused by their bit lengths before any arithmetic, and
check_factorization and prime_proof refuse oversized or overlong factorizations and certificates on their own charges.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable

from mathforge.limits import Budget, computation
from mathforge.trusted.integers import division_work, product_work

from .factorization import charged_pow, check_factorization, prime_proof, primes_of

_FAMILY = "mathforge.areas.ntheory.discrete"  # The module of the family's record classes.
# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "discrete_log": ("base", "target", "p"),
    "two_squares": ("n",),
}
# The record class of each operation's value, by its qualified name, and the fields the checker reads from it.
_RECORDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "discrete_log": ("DiscreteLogarithm", ("exponent", "order")),
    "two_squares": ("SumOfTwoSquares", ("count", "a", "b")),
}


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for arguments other than the operation's exact ints and for any malformed value
    or evidence; ResourceLimitError for an argument over max_integer_bits and past the budget."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = _ARGUMENTS.get(operation)
    if names is None or len(args) != len(names) or any(type(args.get(name)) is not int for name in names):
        return False
    fields = _fields(operation, value)
    if fields is None or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        budget.tick()
        for name in names:
            budget.check_int(args[name])  # As the producer's budget refuses an argument over max_integer_bits.
        return _CHECKS[operation](budget, *(args[name] for name in names), *fields, evidence)


def _fields(operation: str, value: object) -> tuple | None:
    """The fields of value when its class is the operation's record class of the family; None for any other class and
    for a record forged without one of its fields."""
    cls = type(value)
    module, name = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    expected, names = _RECORDS[operation]
    if type(module) is not str or type(name) is not str or module != _FAMILY or name != expected:
        return None
    try:
        return tuple(getattr(value, field) for field in names)
    except AttributeError:  # A record made with object.__new__ may lack a field.
        return None


def _reduced(budget: Budget, number: int, modulus: int) -> int:
    """number % modulus, in [0, modulus) for a modulus >= 1, its long division charged before it runs."""
    budget.tick(1 + division_work(number.bit_length(), modulus.bit_length()))
    return number % modulus


def _discrete_log(budget: Budget, base: int, target: int, p: int, exponent: object, order: object,
                  evidence: tuple) -> bool:
    if p < 2 or len(evidence) != 2:
        return False
    # The canonical form, and order | p - 1, screened by bit lengths first: a forged giant costs nothing.
    if type(order) is not int or order < 1 or order.bit_length() > p.bit_length():
        return False
    if exponent is not None and (type(exponent) is not int or not 0 <= exponent < order):
        return False
    budget.tick(1 + division_work(p.bit_length(), order.bit_length()))
    if (p - 1) % order:
        return False
    residue = _reduced(budget, base, p)
    if residue == 0:  # base % p == 0 is outside the domain: no value at all.
        return False
    factors, certificate = evidence
    if not prime_proof(p, certificate) or not check_factorization(order, factors):
        return False
    # order is the multiplicative order of base modulo the prime p (1 % p == 1 for p >= 2).
    if charged_pow(residue, order, p) != 1:
        return False
    for prime in primes_of(factors):
        budget.tick(1 + division_work(order.bit_length(), prime.bit_length()))  # The quotient order // prime.
        if charged_pow(residue, order // prime, p) == 1:
            return False
    goal = _reduced(budget, target, p)
    if exponent is None:  # target lies outside the subgroup of the order-th roots of unity, the powers of base.
        return charged_pow(goal, order, p) != 1
    return charged_pow(residue, exponent, p) == goal


def _two_squares(budget: Budget, n: int, count: object, a: object, b: object, evidence: tuple) -> bool:
    if n < 0 or len(evidence) != 1 or type(count) is not int or count < 0:
        return False
    factors = evidence[0]
    if n == 0:
        if type(factors) is not tuple or factors:  # 0 has no factorization; its evidence is ().
            return False
        expected = 1  # x*x + y*y = 0 only for (0, 0).
    else:
        if not check_factorization(n, factors):
            return False
        expected = 4
        for prime, power, _ in factors:
            budget.tick(1 + product_work(expected.bit_length(), power.bit_length()))
            if prime % 4 == 3 and power % 2 == 1:  # Jacobi: no representation at all.
                expected = 0
                break
            if prime % 4 == 1:
                expected *= power + 1
    if count != expected:
        return False
    if count == 0:
        return a is None and b is None
    if type(a) is not int or type(b) is not int or not 0 <= a <= b:
        return False
    if b.bit_length() > (n.bit_length() >> 1) + 1:  # b <= isqrt(n) has at most half the bits of n, plus one.
        return False
    budget.tick(1 + 2 * product_work(b.bit_length(), b.bit_length()))  # The two squares and their sum.
    return a * a + b * b == n


_CHECKS: dict[str, Callable[..., bool]] = {
    "discrete_log": _discrete_log,
    "two_squares": _two_squares,
}
