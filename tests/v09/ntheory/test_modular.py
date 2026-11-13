"""The modular family of mathforge.ntheory: powmod, multiplicative_order, primitive_root, legendre_symbol,
jacobi_symbol, sqrt_mod.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register ntheory modular

Every expected value comes from the card or from a reference of this module, written here and slow on purpose (brute
force over the residues, trial division, repeated multiplication, Euler's criterion with a power of its own) or
independent of the producer (the builtin pow); never from the producer. tests/v09/ntheory/test_modular_checker.py
tests the checker on its own.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import functools
import inspect
from math import gcd, isqrt, prod
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.ntheory import modular as family
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Rational
from mathforge.trusted.integers import multiplication_work
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13, MODULI, is_prime

from .. import support

FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
PSI_12 = 318665857834031151167461  # 399165290221 * 798330580441, a strong pseudoprime to the twelve bases 2 to 37.
COMPOSITE = 2 ** 100 + 1  # Above psi_13; 2**20 + 1 divides it.
M61, M89, M107, M127 = 2 ** 61 - 1, 2 ** 89 - 1, 2 ** 107 - 1, 2 ** 127 - 1  # Mersenne primes; M89 on above psi_13.
HALF = PSI_13 + 5062  # A prime above psi_13, so that its certificate is a Pocklington certificate, ...
SAFE = 2 * HALF + 1  # ... and SAFE is prime too. Its smallest primitive root is 5: 2, 3 and 4 are squares modulo SAFE.
PROTH = 57 * 2 ** 90 + 1  # A prime above psi_13 with p - 1 = 57 * 2**90: sqrt_mod takes Cipolla's algorithm.
LARGE_TWO = 13 * 2 ** 1000 + 1  # A prime (Proth's theorem, witness 3) where Tonelli and Shanks' loop ran out of budget.
FERMAT = 65537  # p - 1 = 2**16, and 2 is a square modulo p.
NTT = (998244353, 754974721)  # Primes p with p - 1 = 2**23 * 7 * 17 and 2**24 * 3**2 * 5.
HARD_PRIME = 1633708981145020122364067  # 2 * 827493735521 * 987142809073 + 1, below psi_13: rho must split p - 1.
HARD_N = 757867734457 * 830969408791  # Two 40-bit primes, which rho must split.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact int: a float, a bool, a str, None and two containers.

# One small request per operation, the lean inputs of the integration notes (docs/v09/cards/m1-ntheory.md).
SMALL = {
    "powmod": dict(base=7, exponent=128, modulus=13),
    "multiplicative_order": dict(a=2, n=7),
    "primitive_root": dict(p=23),
    "legendre_symbol": dict(a=2, p=7),
    "jacobi_symbol": dict(a=2, n=15),
    "sqrt_mod": dict(a=2, p=7),
}

# The known values of the card: (operation, arguments, value, evidence).
KNOWN = (
    ("powmod", dict(base=7, exponent=123456, modulus=1000003), pow(7, 123456, 1000003), ()),
    ("powmod", dict(base=3, exponent=-1, modulus=7), 5, ()),
    ("powmod", dict(base=2, exponent=10, modulus=1), 0, ()),
    ("powmod", dict(base=0, exponent=0, modulus=5), 1, ()),
    ("powmod", dict(base=-2, exponent=3, modulus=5), 2, ()),
    ("multiplicative_order", dict(a=2, n=7), 3, (((3, 1, ()),),)),
    ("multiplicative_order", dict(a=3, n=7), 6, (((2, 1, ()), (3, 1, ())),)),
    ("multiplicative_order", dict(a=10, n=13), 6, (((2, 1, ()), (3, 1, ())),)),
    ("multiplicative_order", dict(a=4, n=9), 3, (((3, 1, ()),),)),
    ("multiplicative_order", dict(a=1, n=1), 1, ((),)),
    ("multiplicative_order", dict(a=5, n=1), 1, ((),)),
    ("primitive_root", dict(p=2), 1, ((), ())),
    ("primitive_root", dict(p=3), 2, (((2, 1, ()),), ())),
    ("primitive_root", dict(p=7), 3, (((2, 1, ()), (3, 1, ())), (2,))),
    ("primitive_root", dict(p=23), 5, (((2, 1, ()), (11, 1, ())), (2, 2, 2))),
    ("primitive_root", dict(p=41), 6, (((2, 3, ()), (5, 1, ())), (2, 5, 2, 2))),
    ("primitive_root", dict(p=71), 7, (((2, 1, ()), (5, 1, ()), (7, 1, ())), (2, 2, 2, 2, 2))),
    ("legendre_symbol", dict(a=2, p=7), 1, ((),)),
    ("legendre_symbol", dict(a=3, p=7), -1, ((),)),
    ("legendre_symbol", dict(a=0, p=5), 0, ((),)),
    ("legendre_symbol", dict(a=-1, p=13), 1, ((),)),
    ("legendre_symbol", dict(a=-1, p=7), -1, ((),)),
    ("jacobi_symbol", dict(a=2, n=15), 1, ()),
    ("jacobi_symbol", dict(a=1001, n=9907), -1, ()),
    ("jacobi_symbol", dict(a=19, n=45), 1, ()),
    ("jacobi_symbol", dict(a=0, n=1), 1, ()),
    ("jacobi_symbol", dict(a=3, n=9), 0, ()),
    ("sqrt_mod", dict(a=2, p=7), (3, 4), ((),)),
    ("sqrt_mod", dict(a=5, p=11), (4, 7), ((),)),
    ("sqrt_mod", dict(a=10, p=13), (6, 7), ((),)),
    ("sqrt_mod", dict(a=3, p=7), (), ((),)),
    ("sqrt_mod", dict(a=0, p=13), (0,), ((),)),
    ("sqrt_mod", dict(a=1, p=2), (1,), ((),)),
    ("sqrt_mod", dict(a=4, p=2), (0,), ((),)),
)

# Inputs outside the domains of the card, each invalid input.
INVALID = (
    ("powmod", dict(base=2, exponent=3, modulus=0)),
    ("powmod", dict(base=2, exponent=3, modulus=-1)),
    ("powmod", dict(base=2, exponent=3, modulus=-7)),
    ("powmod", dict(base=2, exponent=-1, modulus=4)),
    ("powmod", dict(base=6, exponent=-3, modulus=4)),
    ("powmod", dict(base=0, exponent=-1, modulus=5)),
    ("powmod", dict(base=10, exponent=-2, modulus=15)),
    ("multiplicative_order", dict(a=2, n=4)),
    ("multiplicative_order", dict(a=6, n=9)),
    ("multiplicative_order", dict(a=0, n=12)),
    ("multiplicative_order", dict(a=0, n=2)),
    ("multiplicative_order", dict(a=2, n=0)),
    ("multiplicative_order", dict(a=3, n=-1)),
    ("primitive_root", dict(p=9)),
    ("primitive_root", dict(p=1)),
    ("primitive_root", dict(p=0)),
    ("primitive_root", dict(p=-7)),
    ("primitive_root", dict(p=4)),
    ("primitive_root", dict(p=341)),  # A base-2 pseudoprime.
    ("primitive_root", dict(p=561)),  # A Carmichael number.
    ("primitive_root", dict(p=PSI_12)),
    ("primitive_root", dict(p=COMPOSITE)),
    ("legendre_symbol", dict(a=5, p=2)),
    ("legendre_symbol", dict(a=2, p=9)),
    ("legendre_symbol", dict(a=2, p=1)),
    ("legendre_symbol", dict(a=2, p=0)),
    ("legendre_symbol", dict(a=2, p=-7)),
    ("legendre_symbol", dict(a=3, p=15)),
    ("legendre_symbol", dict(a=3, p=2 ** 70)),
    ("legendre_symbol", dict(a=2, p=PSI_12)),
    ("legendre_symbol", dict(a=3, p=COMPOSITE)),
    ("jacobi_symbol", dict(a=2, n=8)),
    ("jacobi_symbol", dict(a=2, n=0)),
    ("jacobi_symbol", dict(a=2, n=-3)),
    ("jacobi_symbol", dict(a=1, n=2)),
    ("jacobi_symbol", dict(a=3, n=-15)),
    ("jacobi_symbol", dict(a=1, n=2 ** 70)),
    ("sqrt_mod", dict(a=1, p=15)),  # Modulo 15, 1 has the four square roots 1, 4, 11 and 14.
    ("sqrt_mod", dict(a=0, p=1)),
    ("sqrt_mod", dict(a=0, p=0)),
    ("sqrt_mod", dict(a=1, p=-7)),
    ("sqrt_mod", dict(a=0, p=4)),
    ("sqrt_mod", dict(a=4, p=PSI_12)),
    ("sqrt_mod", dict(a=4, p=COMPOSITE)),
)


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these, from the card, or from the builtin pow.

def primes_below(limit: int) -> list[int]:
    """The primes below limit, by Eratosthenes' sieve."""
    flags = bytearray([1]) * limit
    flags[:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return [n for n in range(limit) if flags[n]]


def factor(n: int, bound: int = 100_000) -> tuple[tuple[int, int], ...]:
    """The prime factorization ((p, e), ...) of n >= 1 by trial division; AssertionError when a divisor above bound
    would be needed to finish it."""
    pairs, d = [], 2
    while d * d <= n:
        if d > bound:
            raise AssertionError(f"trial division up to {bound} does not factor {n}")
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            pairs.append((d, e))
        d += 1 if d == 2 else 2
    if n > 1:  # No divisor up to its square root is left: it is prime.
        pairs.append((n, 1))
    return tuple(pairs)


def power(base: int, exponent: int, modulus: int) -> int:
    """base**exponent modulo modulus >= 1 for exponent >= 0, by right-to-left square-and-multiply written here."""
    result, base = 1 % modulus, base % modulus
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def brute_power(base: int, exponent: int, modulus: int) -> int:
    """base**exponent modulo modulus by |exponent| multiplications, a negative exponent raising the inverse that a
    search of every residue finds; ValueError when there is none. Small numbers only."""
    if exponent < 0:
        inverse = next((x for x in range(modulus) if base * x % modulus == 1 % modulus), None)
        if inverse is None:
            raise ValueError("the base is not invertible")
        base, exponent = inverse, -exponent
    value = 1 % modulus
    for _ in range(exponent):
        value = value * base % modulus
    return value


def brute_order(a: int, n: int) -> int | None:
    """The least d >= 1 with a**d = 1 modulo n, trying d = 1, 2, ..., n; None when a is not a unit modulo n."""
    one, x = 1 % n, a % n
    for d in range(1, n + 1):
        if x == one:
            return d
        x = x * a % n
    return None


def brute_root(p: int) -> int:
    """The smallest g >= 1 whose order modulo the prime p is p - 1, by brute force."""
    return next(g for g in range(1, p) if brute_order(g, p) == p - 1)


def first_refutations(p: int, g: int) -> tuple[int, ...]:
    """For each h = 2, ..., g - 1, the least prime q of p - 1 with h**((p - 1)/q) = 1 modulo p, which the card asks
    of the producer (the checker takes any such prime)."""
    primes = [q for q, _ in factor(p - 1)]
    return tuple(next(q for q in primes if power(h, (p - 1) // q, p) == 1) for h in range(2, g))


@functools.lru_cache(maxsize=None)
def square_roots(p: int) -> dict[int, tuple[int, ...]]:
    """Brute force: every residue modulo p mapped to the sorted tuple of its square roots in [0, p)."""
    roots: dict[int, list[int]] = {r: [] for r in range(p)}
    for x in range(p):
        roots[x * x % p].append(x)
    return {r: tuple(found) for r, found in roots.items()}


def legendre(a: int, p: int) -> int:
    """(a/p) for an odd prime p by brute force: 0, or whether a is a square modulo p."""
    residue = a % p
    return 0 if residue == 0 else 1 if square_roots(p)[residue] else -1


def euler(a: int, p: int) -> int:
    """(a/p) for an odd prime p by Euler's criterion, with the power written here."""
    e = power(a, (p - 1) // 2, p)
    assert e in (0, 1, p - 1)
    return 0 if e == 0 else 1 if e == 1 else -1


def jacobi_product(a: int, n: int, factors: tuple[tuple[int, int], ...] | None = None) -> int:
    """(a/n) for an odd n >= 1: the product of the Legendre symbols (a/p)**e over the factorization of n."""
    return prod(euler(a, p) ** e for p, e in (factor(n) if factors is None else factors))


def binary_jacobi(a: int, n: int) -> int:
    """(a/n) for an odd n >= 1 by the textbook binary algorithm, written here: the factors 2 of a removed one at a time,
    then a swap by reciprocity. test_binary_jacobi_worst_case_fits_the_default_budget checks it against
    jacobi_product."""
    a, sign = a % n, 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                sign = -sign
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            sign = -sign
        a %= n
    return sign if n == 1 else 0


def binary_passes(a: int, n: int) -> int:
    """The passes of the binary Jacobi algorithm on (a, n): one per swap."""
    a, passes = a % n, 0
    while a:
        a >>= (a & -a).bit_length() - 1
        a, n, passes = n % a, a, passes + 1
    return passes


def binary_worst_case(bits: int) -> tuple[int, int, int]:
    """(a, n, (a/n)), n odd of at most that many bits, with about the most passes of the binary Jacobi algorithm,
    built backwards from (2, 3): a state (b, m) follows from (a, a + b) with a = m for an even b (quotient 1) and
    a = 2m for an odd b (quotient 2 after an even a). About 0.91 passes per bit, against at most 0.79 for least
    absolute remainders, which need half as many passes here. The symbol is carried along by the laws of a pass read
    backwards: (a/n) = (2/n)**v (-1)**((m - 1)/2 * (n - 1)/2) (b/m) for a = 2**v m and n mod m = b."""
    b, m, symbol = 2, 3, -1  # (2/3) = -1, as 3 = 3 modulo 8.
    while True:
        v = b & 1
        a = m << v
        n = a + b
        if n.bit_length() > bits:
            return b, m, symbol
        if v and n % 8 in (3, 5):  # (2/n) = -1.
            symbol = -symbol
        if m % 4 == 3 and n % 4 == 3:  # Reciprocity of the odd m and n.
            symbol = -symbol
        b, m = a, n


def reference(operation: str, args: dict[str, int]) -> object:
    """The value of a request with small arguments by the brute-force references."""
    if operation == "powmod":
        return brute_power(args["base"], args["exponent"], args["modulus"])
    if operation == "multiplicative_order":
        return brute_order(args["a"], args["n"])
    if operation == "primitive_root":
        return 1 if args["p"] == 2 else brute_root(args["p"])
    if operation == "legendre_symbol":
        return legendre(args["a"], args["p"])
    if operation == "jacobi_symbol":
        return jacobi_product(args["a"], args["n"])
    return square_roots(args["p"])[args["a"] % args["p"]]


@dataclass(frozen=True)
class Residue:
    """A record that holds a right answer. No value of the modular family is a record, so the checker refuses every
    record: this one, an impostor of it and one forged with object.__new__."""
    value: object


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


class ModularTests(support.FamilyTestCase):
    FAMILY = family

    def assertAgrees(self, operation: str, args: dict[str, int], expected: object) -> tuple:
        """produce() answers expected, with its exact type, and the checker accepts that answer with its evidence,
        which is returned."""
        value, evidence = family.produce(operation, dict(args))
        self.assertEqual(value, expected)
        self.assertIs(type(value), type(expected))
        if type(value) is tuple:
            self.assertTrue(all(type(item) is int for item in value), value)
        self.assertIs(support.verdict(family, operation, args, value, evidence), True)
        return evidence

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with its four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, args, value, evidence in KNOWN:
            with self.subTest(operation=operation, **args):
                self.assertEqual(reference(operation, args), value)  # The card's value, by brute force.
                result = support.run(operation, **args)
                self.assertVerified(result, value)
                self.assertIs(type(result.value), type(value))
                self.assertEqual(result.steps[0].outputs[1], evidence)
                self.roundtrip(result)
        # Large primes: p - 1 is factored by the kernel, from M89 on p itself has a Pocklington certificate, and
        # SAFE - 1 = 2 * HALF has a prime with one.
        for p, g in ((M61, 37), (M89, 3), (M107, 3), (M127, 43), (SAFE, 5)):
            with self.subTest(operation="primitive_root", p=p):
                result = support.run("primitive_root", p=p)
                self.assertVerified(result, g)
                factors, refutations = result.steps[0].outputs[1]
                self.assertEqual(prod(q ** e for q, e, _ in factors), p - 1)
                primes = [q for q, _, _ in factors]
                self.assertEqual(refutations, tuple(next(q for q in primes if power(h, (p - 1) // q, p) == 1)
                                                    for h in range(2, g)))
                self.assertTrue(all(power(g, (p - 1) // q, p) != 1 for q in primes))
        self.assertEqual(support.run("primitive_root", p=SAFE).steps[0].outputs[1][1], (2, 2, 2))
        for a, order in ((4, HALF), (5, 2 * HALF), (SAFE - 1, 2), (SAFE + 1, 1), (2, HALF)):
            with self.subTest(operation="multiplicative_order", a=a, n=SAFE):
                # The orders modulo SAFE divide SAFE - 1 = 2 * HALF: the least of 1, 2, HALF, 2 * HALF that fits.
                self.assertEqual(next(d for d in (1, 2, HALF, 2 * HALF) if power(a, d, SAFE) == 1), order)
                result = support.run("multiplicative_order", a=a, n=SAFE)
                self.assertVerified(result, order)
                if order > 2:
                    self.assertEqual(result.steps[0].outputs[1][0][-1][:2], (HALF, 1))
                    self.assertEqual(result.steps[0].outputs[1][0][-1][2][:2], ("pocklington", HALF))
        x = power(3, 40, M89)
        for operation, args, value in (("legendre_symbol", dict(a=3, p=M89), euler(3, M89)),
                                       ("legendre_symbol", dict(a=-2, p=PROTH), euler(-2, PROTH)),
                                       ("sqrt_mod", dict(a=x * x + 5 * M89, p=M89), tuple(sorted((x, M89 - x))))):
            with self.subTest(operation=operation, **args):
                result = support.run(operation, **args)
                self.assertVerified(result, value)
                self.assertEqual(result.steps[0].outputs[1][0][:2], ("pocklington", args["p"]))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the domains of the card. produce() tests the types itself too."""
        for operation, args in SMALL.items():
            for name in args:
                for item in STAND_INS:
                    with self.subTest(operation=operation, argument=name, item=item):
                        self.assertInvalidInput(support.run(operation, **dict(args, **{name: item})))
        for operation, args in INVALID:
            with self.subTest(operation=operation, **args):
                self.assertInvalidInput(support.run(operation, **args))
        messages = (("powmod", dict(base=2, exponent=-1, modulus=4), "base is not invertible modulo modulus"),
                    ("powmod", dict(base=2, exponent=3, modulus=0), "modulus >= 1"),
                    ("multiplicative_order", dict(a=2, n=4), "gcd(a, n) = 1"),
                    ("primitive_root", dict(p=9), "a prime p"),
                    ("legendre_symbol", dict(a=5, p=2), "an odd prime p"),
                    ("jacobi_symbol", dict(a=2, n=8), "an odd n >= 1"),
                    ("sqrt_mod", dict(a=1, p=15), "a prime p"))
        for operation, args, text in messages:
            with self.subTest(operation=operation, message=text):
                self.assertIn(text, support.run(operation, **args).error.message)
        # operations_v6.run rejects a stand-in by its kind before produce() runs; produce() refuses one itself.
        for operation, args in SMALL.items():
            for name in args:
                for item in (True, False, 0.5, "1", None, Rational(3), Integer(3)):
                    with self.subTest(operation=operation, argument=name, item=item, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: item}))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(args, extra=1))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, tuple(args.items()))
        for operation in ("integer_gcd", "Powmod", "", None):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(base=7, exponent=128, modulus=13))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("jacobi_symbol", a=2, n=8)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 and max_work=1 exceed the budget. No operation of the family takes a matrix, so
        support.oversized_matrix() has nothing to exceed here. Work is charged before it runs, so a request that
        needs more than its budget fails at once, and a budget error of the integer kernel reaches the Result."""
        for operation, args in SMALL.items():
            for name in args:
                with self.subTest(operation=operation, argument=name):
                    self.assertResourceLimit(support.run(operation, **dict(args, **{name: support.HUGE})))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
        # The square-and-multiply loop is charged before it starts: 4,000 squarings and a multiplication modulo
        # 65,536 bits, some 131 million units, are refused at once.
        rng = random.Random(20261120)
        modulus = rng.getrandbits(65_536) | 1 | 1 << 65_535
        start = time.perf_counter()
        self.assertResourceLimit(support.run("powmod", base=3, exponent=2 ** 4000 + 1, modulus=modulus))
        self.assertResourceLimit(support.run("powmod", base=3, exponent=-(2 ** 4000 + 1), modulus=modulus,
                                             limits=ComputationLimits(max_work=1000)))
        n = rng.getrandbits(65_000) | 1 | 1 << 64_999
        self.assertResourceLimit(support.run("jacobi_symbol", a=rng.getrandbits(64_000), n=n,
                                             limits=ComputationLimits(max_work=5_000)))
        self.assertLess(time.perf_counter() - start, FAST)
        # p - 1 = 2 * q1 * q2 and n = q1 * q2 with 40-bit primes: rho runs out of a small budget.
        self.assertEqual(HARD_PRIME - 1, 2 * 827493735521 * 987142809073)
        # All prime (the strong tests to the thirteen bases decide below psi_13): the errors are the budget's.
        self.assertTrue(all(is_prime(q) for q in (HARD_PRIME, 827493735521, 987142809073, 757867734457,
                                                  830969408791)))
        limits = ComputationLimits(max_work=20_000)
        start = time.perf_counter()
        self.assertResourceLimit(support.run("primitive_root", p=HARD_PRIME, limits=limits))
        self.assertResourceLimit(support.run("multiplicative_order", a=2, n=HARD_N, limits=limits))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: a
        refutation by another prime than the first, and Pocklington certificates with other witnesses."""
        chosen = 0
        for p in primes_below(2000):
            g, (factors, refutations) = family.produce("primitive_root", {"p": p})
            primes = [q for q, _, _ in factors]
            largest = tuple(max(q for q in primes if power(h, (p - 1) // q, p) == 1) for h in range(2, g))
            if largest != refutations:
                chosen += 1
                with self.subTest(p=p):
                    self.assertAccepts("primitive_root", {"p": p}, g, (factors, largest))
        self.assertGreater(chosen, 10)
        # 2 modulo 31 has the order 5: both 2 and 3 refute it, and the producer names 2.
        thirty = ((2, 1, ()), (3, 1, ()), (5, 1, ()))
        self.assertEqual(family.produce("primitive_root", {"p": 31}), (3, (thirty, (2,))))
        self.assertAccepts("primitive_root", {"p": 31}, 3, (thirty, (3,)))

        def rewitnessed(certificate: tuple) -> tuple:
            """The Pocklington certificate with, for each prime q of n - 1, the least witness other than its own."""
            kind, n, factors, witnesses = certificate
            others = tuple(next(w for w in range(2, n) if w != witness and pow(w, n - 1, n) == 1
                                and gcd(pow(w, (n - 1) // q, n) - 1, n) == 1)
                           for (q, _, _), witness in zip(factors, witnesses))
            self.assertNotEqual(others, witnesses)
            return kind, n, factors, others

        x, y = power(5, 77, M89), power(7, 99, PROTH)
        for operation, args, value in (("legendre_symbol", dict(a=3, p=M89), euler(3, M89)),
                                       ("sqrt_mod", dict(a=x * x, p=M89), tuple(sorted((x, M89 - x)))),
                                       ("sqrt_mod", dict(a=y * y, p=PROTH), tuple(sorted((y, PROTH - y))))):
            with self.subTest(operation=operation, p=args["p"]):
                produced, (certificate,) = family.produce(operation, args)
                self.assertEqual(produced, value)
                self.assertAccepts(operation, args, value, (rewitnessed(certificate),))
        order, (factors,) = family.produce("multiplicative_order", dict(a=4, n=SAFE))
        self.assertEqual(order, HALF)
        self.assertAccepts("multiplicative_order", dict(a=4, n=SAFE), HALF,
                           ((factors[:-1] + ((HALF, 1, rewitnessed(factors[-1][2])),)),))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, a record of any class (an
        impostor, one forged with object.__new__), a bool, a float or a str where an int belongs, a list for a
        tuple, missing or extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the
        mathematics of a Result whose value was replaced."""
        giant = 2 ** 70_000  # Longer than max_integer_bits allows.
        claims = {
            "powmod": dict(base=7, exponent=123456, modulus=1000003),
            "multiplicative_order": dict(a=10, n=13),
            "primitive_root": dict(p=41),
            "legendre_symbol": dict(a=3, p=7),
            "jacobi_symbol": dict(a=2, n=15),
            "sqrt_mod": dict(a=10, p=13),
        }
        v = pow(7, 123456, 1000003)
        six, twelve = (((2, 1, ()), (3, 1, ())),), (((2, 2, ()), (3, 1, ())),)
        factors41 = ((2, 3, ()), (5, 1, ()))
        wrong = {
            "powmod": [((v + 1) % 1000003, ()), ((v * 7) % 1000003, ()), (v + 1000003, ()), (v - 1000003, ())],
            "multiplicative_order": [(3, (((3, 1, ()),),)), (12, twelve), (6, ((six[0][0],),)), (1, ((),)),
                                     (0, ((),)), (-6, six), (19, (((19, 1, ()),),))],
            "primitive_root": [(7, (factors41, (2, 5, 2, 2, 2))), (5, (factors41, (2, 5, 2))),
                               (6, (factors41, (2, 2, 2, 2))), (6, (factors41, (2, 5, 2, 4))),
                               (6, (factors41[:1], (2, 5, 2, 2))), (47, (factors41, (2,) * 45))],
            "legendre_symbol": [(1, ((),)), (0, ((),)), (6, ((),)), (-1, ())],
            "jacobi_symbol": [(-1, ()), (0, ()), (16, ())],
            "sqrt_mod": [((5, 8), ((),)), ((7, 6), ((),)), ((6,), ((),)), ((), ((),)), ((6, 7, 13), ((),)),
                         ((6, 20), ((),)), ((-7, 7), ((),))],
        }
        for operation, args in claims.items():
            result = support.run(operation, **args)
            value, evidence = result.value, result.steps[0].outputs[1]
            self.assertVerified(result, value)
            self.assertAccepts(operation, args, value, evidence)
            cases = [(f"wrong {item!r}", item, proof) for item, proof in wrong[operation]]
            cases += [("a record", Residue(value), evidence),
                      ("an impostor record", support.impostor(Residue(value)), evidence),
                      ("a forged record", support.forged(Residue, value=value), evidence),
                      ("evidence as a list", value, list(evidence)), ("evidence None", value, None),
                      ("an extra evidence item", value, evidence + ((),)), ("a giant value", giant, evidence),
                      ("oversized evidence", value, evidence + (0,) * 100_000)]
            if evidence:
                cases.append(("the last evidence item missing", value, evidence[:-1]))
            if type(value) is int:
                cases += [(f"value {item!r}", item, evidence)
                          for item in (float(value), str(value), Rational(value), Integer(value))]
                if value in (0, 1):
                    cases.append(("a bool", bool(value), evidence))
            else:
                cases += [("a list", list(value), evidence), ("a tuple subclass", type("Roots", (tuple,), {})(value),
                                                              evidence)]
                cases += [(f"root {item!r}", value[:1] + (item,), evidence)
                          for item in (float(value[1]), str(value[1]), Integer(value[1]))]
            for label, item, proof in cases:
                with self.subTest(operation=operation, case=label):
                    self.assertRejects(operation, args, item, proof)
            for other in claims:  # The true claim under the name of another operation.
                if other != operation:
                    with self.subTest(operation=operation, renamed=other):
                        self.assertRejects(other, args, value, evidence)
            first_wrong = wrong[operation][0][0]
            report = support.statuses(mf.verify(support.tampered(result, first_wrong)))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        x = power(5, 77, M89)
        requests = list(SMALL.items()) + [("legendre_symbol", dict(a=3, p=M89)),
                                          ("sqrt_mod", dict(a=x * x, p=M89)), ("primitive_root", dict(p=SAFE)),
                                          ("multiplicative_order", dict(a=4, n=SAFE)),
                                          ("powmod", dict(base=3, exponent=-1, modulus=7))]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, _), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("powmod", base=3, exponent=-1, modulus=7)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: an int, a negative
        int, zero and tuples of two, one and no roots."""
        x = power(5, 77, M89)
        rendered = (
            ("powmod", dict(base=7, exponent=128, modulus=13), "3", "3"),
            ("multiplicative_order", dict(a=4, n=SAFE), str(HALF), str(HALF)),
            ("primitive_root", dict(p=SAFE), "5", "5"),
            ("legendre_symbol", dict(a=3, p=7), "-1", "-1"),
            ("jacobi_symbol", dict(a=3, n=9), "0", "0"),
            ("sqrt_mod", dict(a=2, p=7), "(3, 4)", r"\left(3, 4\right)"),
            ("sqrt_mod", dict(a=0, p=13), "(0)", r"\left(0\right)"),
            ("sqrt_mod", dict(a=3, p=7), "()", r"\left(\right)"),
            ("sqrt_mod", dict(a=x * x, p=M89), f"({min(x, M89 - x)}, {max(x, M89 - x)})",
             rf"\left({min(x, M89 - x)}, {max(x, M89 - x)}\right)"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(operation=operation, **args):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("powmod", base=7, exponent=128, modulus=13).value)["data"],
                         {"type": "integer", "value": "3"})
        self.assertEqual(mf.to_data(support.run("sqrt_mod", a=2, p=7).value)["data"],
                         {"type": "tuple", "items": [{"type": "integer", "value": "3"},
                                                     {"type": "integer", "value": "4"}]})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261121)
        primes = primes_below(3000)
        cases = [("powmod", dict(base=rng.randrange(-10 ** 9, 10 ** 9), exponent=rng.randrange(-5, 10 ** 6),
                                 modulus=rng.choice(primes))) for _ in range(12)]
        cases += [("multiplicative_order", dict(a=a, n=n)) for n in rng.sample(range(2, 5000), 12)
                  for a in [rng.randrange(1, n)] if gcd(a, n) == 1]
        cases += [("primitive_root", dict(p=p)) for p in rng.sample(primes, 10) + [M61, SAFE]]
        cases += [("legendre_symbol", dict(a=rng.randrange(-10 ** 6, 10 ** 6), p=p))
                  for p in rng.sample(primes[1:], 10)]
        cases += [("jacobi_symbol", dict(a=rng.randrange(-10 ** 9, 10 ** 9), n=rng.randrange(1, 10 ** 6) | 1))
                  for _ in range(12)]
        cases += [("sqrt_mod", dict(a=rng.randrange(-10 ** 6, 10 ** 6), p=p)) for p in rng.sample(primes, 10)]
        cases += [("sqrt_mod", dict(a=rng.randrange(PROTH) ** 2, p=PROTH)), ("legendre_symbol", dict(a=7, p=M89))]
        for operation, args in cases:
            with self.subTest(operation=operation, **args):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, args in SMALL.items():
            with self.subTest(operation=operation):
                result = self.answer(support.call(operation, args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        # Integers from 2**53 on travel as decimal strings, and the evidence holds a Pocklington certificate.
        for operation, args in (("primitive_root", dict(p=SAFE)), ("legendre_symbol", dict(a=3, p=M89))):
            with self.subTest(operation=operation, large=True):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("jacobi_symbol", {"a": 2, "n": 8}), "invalid_input")
        self.assertIn("an odd n >= 1", error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 300 seeded inputs per operation (1000 n
        for jacobi_symbol, every prime below 2000 for primitive_root), and the checker accepts every answer with its
        evidence."""
        rng = random.Random(20261118)
        counts = dict.fromkeys(SMALL, 0)
        # powmod: the builtin pow, an implementation other than the producer's, and repeated multiplication.
        while counts["powmod"] < 400:
            modulus = rng.choice((1, 2, rng.randrange(1, 60), rng.randrange(1, 10 ** 6),
                                  rng.getrandbits(rng.randrange(2, 400)) | 1))
            base = rng.choice((rng.randrange(-3 * modulus - 5, 3 * modulus + 5), rng.getrandbits(900) - 2 ** 899))
            exponent = rng.choice((rng.randrange(-30, 60), rng.getrandbits(300) - 2 ** 299))
            args = dict(base=base, exponent=exponent, modulus=modulus)
            try:
                expected = pow(base, exponent, modulus)
            except ValueError:  # No inverse of base modulo modulus.
                with self.subTest(args=args), self.assertRaises(InvalidInput):
                    family.produce("powmod", args)
                continue
            with self.subTest(operation="powmod", **args):
                if modulus < 60 and abs(exponent) < 60:
                    self.assertEqual(brute_power(base, exponent, modulus), expected)
                self.assertAgrees("powmod", args, expected)
                counts["powmod"] += 1
        # multiplicative_order: brute force, with negative a and a of n or more.
        while counts["multiplicative_order"] < 300:
            n = rng.choice((rng.randrange(1, 50), rng.randrange(1, 3000)))
            a = rng.randrange(-n - 3, 2 * n + 3)
            expected = brute_order(a, n)
            if expected is None:
                with self.subTest(a=a, n=n), self.assertRaises(InvalidInput):
                    family.produce("multiplicative_order", dict(a=a, n=n))
                continue
            with self.subTest(operation="multiplicative_order", a=a, n=n):
                self.assertAgrees("multiplicative_order", dict(a=a, n=n), expected)
                counts["multiplicative_order"] += 1
        # Larger moduli, where the checker is the reference: a word-sized prime times small prime powers (a product
        # of two word-sized primes would make the kernel's rho split it, beyond the default budget).
        for _ in range(30):
            n = rng.choice(MODULI[:8]) * rng.choice((1, 2, 4, 3, 2 ** 20, 3 ** 7, 5 * 7 * 11))
            a = rng.randrange(-n, n)
            if gcd(a, n) == 1:
                with self.subTest(operation="multiplicative_order", a=a, n=n):
                    order, evidence = family.produce("multiplicative_order", dict(a=a, n=n))
                    self.assertIs(support.verdict(family, "multiplicative_order", dict(a=a, n=n), order, evidence),
                                  True)
        # primitive_root: brute force for every prime below 2000, and the card's choice of refutations.
        for p in primes_below(2000):
            with self.subTest(operation="primitive_root", p=p):
                g = 1 if p == 2 else brute_root(p)
                factors, refutations = self.assertAgrees("primitive_root", dict(p=p), g)
                self.assertEqual(factors, tuple((q, e, ()) for q, e in factor(p - 1)))
                self.assertEqual(refutations, first_refutations(p, g))
                counts["primitive_root"] += 1
        # legendre_symbol: whether a is a square modulo p, by brute force.
        odd_primes = primes_below(5000)[1:]
        while counts["legendre_symbol"] < 300:
            p = rng.choice(odd_primes)
            a = rng.choice((rng.randrange(-p, 2 * p), rng.randrange(-10 ** 30, 10 ** 30), p * rng.randrange(-4, 5)))
            with self.subTest(operation="legendre_symbol", a=a, p=p):
                self.assertEqual(self.assertAgrees("legendre_symbol", dict(a=a, p=p), legendre(a, p)), ((),))
                counts["legendre_symbol"] += 1
        for p in (M61, M89, M127, HALF, SAFE, PROTH, *NTT):  # Euler's criterion, with the power written here.
            for a in (2, 3, -1, p - 2, rng.randrange(p * p)):
                with self.subTest(operation="legendre_symbol", a=a, p=p):
                    self.assertAgrees("legendre_symbol", dict(a=a, p=p), euler(a, p))
        # jacobi_symbol: the product of Legendre symbols over the factorization of n.
        while counts["jacobi_symbol"] < 1000:
            n = rng.randrange(1, 100_000) | 1
            a = rng.choice((rng.randrange(-n, 2 * n), rng.randrange(-10 ** 30, 10 ** 30), n * rng.randrange(-3, 4),
                            rng.randrange(n) * 4))
            with self.subTest(operation="jacobi_symbol", a=a, n=n):
                self.assertAgrees("jacobi_symbol", dict(a=a, n=n), jacobi_product(a, n))
                counts["jacobi_symbol"] += 1
        for _ in range(40):  # Long n, products of word-sized primes whose factorization is known.
            chosen = rng.sample(MODULI, rng.randrange(1, 5))
            factors = tuple(sorted((q, rng.randrange(1, 3)) for q in chosen))
            n = prod(q ** e for q, e in factors)
            a = rng.choice((rng.getrandbits(n.bit_length() + 40) - 2 ** (n.bit_length() + 39), chosen[0] * 7))
            with self.subTest(operation="jacobi_symbol", a=a, n=n):
                self.assertAgrees("jacobi_symbol", dict(a=a, n=n), jacobi_product(a, n, factors))
        # sqrt_mod: every square root by brute force.
        small_primes = primes_below(3000)
        while counts["sqrt_mod"] < 300:
            p = rng.choice(small_primes)
            a = rng.choice((rng.randrange(-p, 2 * p), rng.randrange(-10 ** 30, 10 ** 30), p * rng.randrange(-4, 5),
                            rng.randrange(p) ** 2))
            with self.subTest(operation="sqrt_mod", a=a, p=p):
                self.assertEqual(self.assertAgrees("sqrt_mod", dict(a=a, p=p), square_roots(p)[a % p]), ((),))
                counts["sqrt_mod"] += 1
        # Large primes, each shape: squares built backwards, multiples of p and non-residues (Euler's criterion).
        for p in (M61, M89, M127, HALF, SAFE, PROTH, FERMAT, *NTT):
            x = rng.randrange(1, p)
            non_residue = next(z for z in range(2, 1000) if euler(z, p) == -1)
            for a, roots in ((x * x + 5 * p, tuple(sorted((x, p - x)))), (1, (1, p - 1)), (7 * p, (0,)),
                             (-p, (0,)), (non_residue, ()), (non_residue - 3 * p, ())):
                with self.subTest(operation="sqrt_mod", a=a, p=p):
                    self.assertAgrees("sqrt_mod", dict(a=a, p=p), roots)
        self.assertTrue(all(count >= 300 for count in counts.values()), counts)

    def test_the_check_has_a_budget_of_its_own(self):
        """The verifier checks in a fresh budget of the request's limits, so a certified factorization is refused only
        when check_factorization alone needs more than max_work, whatever the production spent (the rule of
        kernels.certificates). Inside an active budget that has spent all but what the production needs, produce()
        still answers, where it used to require the production and the check to fit in max_work together."""
        mersenne = 2 ** 61 - 1
        for operation, args in (("primitive_root", {"p": mersenne}), ("multiplicative_order", {"a": 3, "n": mersenne}),
                                ("multiplicative_order", {"a": 2, "n": 10 ** 12 + 39})):
            with self.subTest(operation=operation, **args):
                with computation(ComputationLimits(max_work=10 ** 9), fresh=True) as budget:
                    value, evidence = family.produce(operation, dict(args))
                production = budget.work
                limits = ComputationLimits(max_work=2_000_000)
                with computation(limits, fresh=True) as budget:
                    budget.tick(limits.max_work - production)  # Used to leave no room for the check: refused.
                    self.assertEqual(family.produce(operation, dict(args)), (value, evidence))
                    self.assertEqual(budget.work, limits.max_work)
                self.assertIs(support.verdict(family, operation, args, value, evidence), True)

    def test_full_size_inputs_fit_the_default_budget(self):
        """Requests on integers of nearly max_integer_bits complete under the default limits: a Jacobi symbol of
        65,534 bits, whose value is known by construction ((a/(3 m**2)) = (a/3) (a/m)**2 = (a/3) for gcd(a, m) = 1),
        and an inverse modulo 65,536 bits."""
        rng = random.Random(20261122)
        m = rng.getrandbits(32_766) | 1 | 1 << 32_765
        n = 3 * m * m
        self.assertLessEqual(n.bit_length(), 65_536)
        a = 3 * rng.getrandbits(65_000) + 2
        while gcd(a, m) != 1:
            a += 3
        self.assertVerified(support.run("jacobi_symbol", a=a, n=n), -1)  # (a/3) = -1 for a = 2 modulo 3.
        self.assertVerified(support.run("jacobi_symbol", a=a + 1, n=n), 0)  # 3 divides a + 1 and n.
        modulus = rng.getrandbits(65_536) | 1 | 1 << 65_535
        base = rng.getrandbits(65_536)
        while gcd(base, modulus) != 1:
            base += 1
        result = support.run("powmod", base=base, exponent=-1, modulus=modulus)
        self.assertVerified(result, result.value)
        self.assertEqual(result.value * base % modulus, 1)
        self.assertEqual(result.value, pow(base, -1, modulus))

    def test_producer_computes_no_builtin_pow(self):
        """The producer's powers come from its own square-and-multiply loop, which charges the budget: nothing in
        the family module calls the builtin pow, whose work could not be interrupted."""
        tree = ast.parse(inspect.getsource(family))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        self.assertIn("_power", called)

    def test_orders_of_small_units_modulo_prime_powers(self):
        """Regression: an order far below lambda(n) costs an exponentiation per prime of lambda(n) and short powers by
        that prime. Dividing each prime out of lambda(n) with an exponentiation by the whole cofactor for every
        division took -1 modulo 2**1000 about a thousand exponentiations of a thousand bits, past the default budget,
        for a claim that the checker verifies with some fifty units."""
        cases = ((-1, 2 ** 1000, 2), (1 + 2 ** 999, 2 ** 1000, 2), (1 + 2 ** 500, 2 ** 1000, 2 ** 500),
                 (-1, 3 ** 600, 2), (1 + 3 ** 399, 3 ** 400, 3), (-1, 2 * 5 ** 400, 2), (5, 2 ** 3000, 2 ** 2998))
        for a, n, order in cases:
            with self.subTest(a=a, n=n):
                # The criterion of the order, with the power written here: a**order = 1, a**(order/q) != 1.
                self.assertEqual(power(a, order, n), 1)
                self.assertTrue(all(power(a, order // q, n) != 1 for q, _ in factor(order)))
                with computation(fresh=True) as budget:
                    value, evidence = family.produce("multiplicative_order", {"a": a, "n": n})
                self.assertEqual((value, evidence), (order, (tuple((q, e, ()) for q, e in factor(order)),)))
                if order.bit_length() < 600:  # One exponentiation and a few squarings: measured 1,038 units for -1.
                    self.assertLess(budget.work, 50_000)
                self.assertVerified(support.run("multiplicative_order", a=a, n=n), order)

    def test_binary_jacobi_worst_case_fits_the_default_budget(self):
        """Regression: the binary algorithm's worst case, about 0.91 passes per bit, completes at 65,535 bits under the
        default limits, where the checker verifies it with about 1.5 million units. At the checker's charge for the
        linear operations of a pass (a unit per 1,024 bits) it ran out of the default budget, although its pass takes
        about a quarter of the time of the checker's: its passes are cheaper, not fewer."""
        for n in range(1, 120, 2):
            for a in range(-n, n + 1):
                self.assertEqual(binary_jacobi(a, n), jacobi_product(a, n), (a, n))
        for bits in (4, 8, 64, 512, 4096):  # The symbol carried through the construction, against the reference.
            a, n, symbol = binary_worst_case(bits)
            self.assertEqual(symbol, binary_jacobi(a, n), bits)
        self.assertGreater(binary_passes(a, n), 0.9 * 4096)  # The chain is the worst case it claims to be.
        a, n, symbol = binary_worst_case(65_535)
        self.assertEqual(n.bit_length(), 65_535)
        with computation(fresh=True) as budget:  # The default limits, as for a request.
            produced = family.produce("jacobi_symbol", {"a": a, "n": n})
        self.assertEqual(produced, (symbol, ()))
        self.assertLess(budget.work, 1_700_000)  # Measured 1,531,176 units.
        self.assertAccepts("jacobi_symbol", {"a": a, "n": n}, symbol)  # In a fresh default budget, as mf.verify.

    def test_square_roots_modulo_primes_with_a_large_power_of_two(self):
        """Regression: for p - 1 = odd * 2**m, Tonelli and Shanks' loop takes up to m(m + 3)/2 multiplications, and at
        p = 13 * 2**1000 + 1 it ran out of the default budget, where the checker verifies the roots with some 32,000
        units; Cipolla's algorithm, whose cost does not grow with m, takes over there. Both algorithms find a root on
        every prime 1 modulo 4 tried, and each one serves the primes that the module gives it."""
        p = LARGE_TWO
        self.assertEqual(power(3, (p - 1) // 2, p), p - 1)  # Proth's theorem, as 13 < 2**1000: p is prime.
        x = power(7, 10 ** 6, p)
        roots = tuple(sorted((x, p - x)))
        with computation(fresh=True) as budget:
            self.assertEqual(family.produce("sqrt_mod", {"a": x * x, "p": p})[0], roots)
        self.assertLess(budget.work, 400_000)  # Measured about 100,000 units, the certificate of p included.
        result = support.run("sqrt_mod", a=x * x, p=p)
        self.assertVerified(result, roots)
        self.assertEqual(result.steps[0].outputs[1][0][:2], ("pocklington", p))
        # The choice: Cipolla for this p; Tonelli and Shanks for 13 (m = 2) and 2**61 - 31 (p - 1 = 2**5 (2**56 - 1)).
        self.assertEqual(MODULI[1], 2 ** 61 - 31)
        with patch.object(family, "_tonelli_shanks", side_effect=AssertionError("Tonelli-Shanks ran")):
            self.assertEqual(family.produce("sqrt_mod", {"a": x * x, "p": p})[0], roots)
        with patch.object(family, "_cipolla", side_effect=AssertionError("Cipolla ran")):
            self.assertEqual(family.produce("sqrt_mod", {"a": 10, "p": 13})[0], (6, 7))
            self.assertEqual(family.produce("sqrt_mod", {"a": 4, "p": MODULI[1]})[0], (2, MODULI[1] - 2))
        # Both algorithms, called directly, on primes 1 modulo 4 with small and large m.
        rng = random.Random(20261123)
        for q in [q for q in primes_below(3000) if q % 4 == 1] + [FERMAT, *NTT, PROTH, p]:
            shift = ((q - 1) & (1 - q)).bit_length() - 1
            work = multiplication_work(q.bit_length())
            for _ in range(2):
                residue = power(rng.randrange(1, q), 2, q)
                for name in ("_tonelli_shanks", "_cipolla"):
                    if name == "_tonelli_shanks" and q == p:
                        continue  # The very loop that this test is about.
                    with self.subTest(p=q, residue=residue, algorithm=name), computation(fresh=True) as budget:
                        if name == "_tonelli_shanks":
                            root = family._tonelli_shanks(budget, residue, q, shift, work)
                        else:
                            root = family._cipolla(budget, residue, q, work)
                        self.assertTrue(0 <= root < q)
                        self.assertEqual(root * root % q, residue)
