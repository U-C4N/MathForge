"""The independent checker of mathforge.areas.ntheory.modular (mathforge.certify.ntheory.modular), tested on its own
before the family's producer exists.

Every true claim is computed here: by brute force on small numbers, or by a short reference of this module (a
square-and-multiply power, orders reduced from a known multiple, Lucas's criterion, a binary Jacobi algorithm that is
checked against products of Legendre symbols, squares built backwards), never by the producer. The evidence follows
the layouts that the family's docstring lists; mathforge.kernels.certificates builds the prime certificates and the
certified factorizations, which the checker proves itself. The checker answers False to every corrupted claim, never
with an exception, and refuses an oversized forgery within FAST seconds under the default budget.
"""

from __future__ import annotations

import ast
from collections import namedtuple
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import enum
import inspect
from math import gcd, isqrt, prod
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.ntheory import modular as family
from mathforge.certify.ntheory import factorization as helpers
from mathforge.certify.ntheory import modular as checker
from mathforge.contracts import CheckStatus
from mathforge.kernels import certificates
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.registry import Argument, OperationSpec
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact integers; certified primes, orders and residues; canonical integer solution families"
PSI_12 = 318665857834031151167461  # 399165290221 * 798330580441, a strong pseudoprime to the twelve bases 2 to 37.
M61, M89, M107, M127 = 2 ** 61 - 1, 2 ** 89 - 1, 2 ** 107 - 1, 2 ** 127 - 1  # Mersenne primes, M89 on above psi_13.
HALF = PSI_13 + 5062  # A prime above psi_13, so that its certificate is a Pocklington certificate, ...
SAFE = 2 * HALF + 1  # ... and SAFE is prime too. Its smallest primitive root is 5: 2, 3 and 4 are squares modulo SAFE.
COMPOSITE = 2 ** 100 + 1  # Above psi_13, and divisible by 2**20 + 1.
NTT = (998244353, 754974721)  # Primes p with p - 1 = 2**23 * 7 * 17 and 2**24 * 3**2 * 5.
LARGE_PRIMES = (M61, M89, M107, M127, HALF, SAFE)

factorization = certificates.factorization_evidence  # ((p, e, certificate), ...), which the checker proves.
certificate = certificates.prime_evidence  # () below psi_13, a Pocklington certificate from there on.


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these, or from brute force.

def primes_below(limit: int) -> list[int]:
    """The primes below limit, by Eratosthenes' sieve."""
    flags = bytearray([1]) * limit
    flags[:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return [n for n in range(limit) if flags[n]]


def factor(n: int, bound: int = 10_000) -> tuple[tuple[int, int], ...]:
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


def inverse(a: int, m: int) -> int | None:
    """The inverse of a modulo m >= 1, in [0, m), by the extended Euclidean algorithm; None when gcd(a, m) > 1."""
    old_r, r, old_s, s = a % m, m, 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    return old_s % m if old_r == 1 else None


def power(base: int, exponent: int, modulus: int) -> int:
    """base**exponent modulo modulus >= 1, in [0, modulus), by square-and-multiply written out here (no builtin pow);
    a negative exponent raises the inverse of base, ValueError when there is none."""
    if exponent < 0:
        base, exponent = inverse(base, modulus), -exponent
        if base is None:
            raise ValueError("the base is not invertible")
    result, base = 1 % modulus, base % modulus
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def brute_order(a: int, n: int) -> int | None:
    """The least d >= 1 with a**d = 1 modulo n, trying d = 1, 2, ..., n; None when a is not a unit modulo n."""
    one, x = 1 % n, a % n
    for d in range(1, n + 1):
        if x == one:
            return d
        x = x * a % n
    return None


def order_from(a: int, n: int, multiple: int) -> int:
    """The order of the unit a modulo n from a multiple of it: each prime of multiple, found by trial division, is
    divided out while the power stays 1."""
    order = multiple
    for q, _ in factor(multiple):
        while order % q == 0 and power(a, order // q, n) == 1 % n:
            order //= q
    assert power(a, order, n) == 1 % n
    return order


def safe_order(a: int) -> int:
    """The order of a unit modulo SAFE: the least of the divisors 1, 2, HALF and 2 * HALF of SAFE - 1 that it has."""
    return next(d for d in (1, 2, HALF, 2 * HALF) if power(a, d, SAFE) == 1)


def refuting(h: int, p: int, primes: Sequence[int]) -> list[int]:
    """The primes q of p - 1 with h**((p - 1)/q) = 1 modulo p, each of which shows that h generates no (Z/p)^*."""
    return [q for q in primes if power(h, (p - 1) // q, p) == 1]


def smallest_root(p: int, primes: Sequence[int]) -> int:
    """The smallest g >= 1 that generates (Z/p)^*, for a prime p whose p - 1 has exactly these primes: Lucas's
    criterion, no g**((p - 1)/q) is 1."""
    g = 1
    while power(g, p - 1, p) != 1 or refuting(g, p, primes):
        g += 1
    return g


def root_evidence(p: int, g: int, pick: Callable[[list[int]], int] = min) -> tuple:
    """(factors, refutations) for primitive_root(p) = g: the certified factorization of p - 1 and, for each
    h = 2, ..., g - 1, the refuting prime that pick chooses."""
    factors = factorization(p - 1)
    primes = [q for q, _, _ in factors]
    return factors, tuple(pick(refuting(h, p, primes)) for h in range(2, g))


def euler(a: int, p: int) -> int:
    """(a/p) for an odd prime p by Euler's criterion, with the power written out here."""
    e = power(a, (p - 1) // 2, p)
    assert e in (0, 1, p - 1)
    return 0 if e == 0 else 1 if e == 1 else -1


def jacobi_product(a: int, n: int) -> int:
    """(a/n) for an odd n >= 1: the product of the Legendre symbols (a/p)**e over the factorization of n."""
    return prod(euler(a, p) ** e for p, e in factor(n))


def binary_jacobi(a: int, n: int) -> int:
    """(a/n) for an odd n >= 1 by the binary algorithm of the textbooks: remove the factors 2 of a, then swap by
    reciprocity. Checked against jacobi_product in test_jacobi_symbol_accepts_true_claims."""
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


def square_roots(p: int) -> dict[int, tuple[int, ...]]:
    """Brute force: every residue modulo p mapped to the sorted tuple of its square roots in [0, p)."""
    roots: dict[int, list[int]] = {r: [] for r in range(p)}
    for x in range(p):
        roots[x * x % p].append(x)
    return {r: tuple(found) for r, found in roots.items()}


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for values and evidence that the checker must refuse.

@dataclass(frozen=True)
class Residue:
    """A record that holds a right answer. No value of the modular family is a record, so the checker refuses every
    record: this one, an impostor of it and one forged with object.__new__."""
    value: object


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


class Sign(enum.IntEnum):
    """The values of the Legendre and Jacobi symbols as an int enumeration: equal to them but no exact ints."""
    MINUS = -1
    ZERO = 0
    PLUS = 1


Pair = namedtuple("Pair", "low high")  # A tuple subclass for the two square roots of sqrt_mod.
Evidence = type("Evidence", (tuple,), {})  # A tuple subclass for an evidence tuple.


def inexact(number: int) -> tuple[object, ...]:
    """Stand-ins for an exact int that are no exact int: a bool, a float, a str, None and a Rational of it."""
    small = number.bit_length() < 1000
    return bool(number), float(number) if small else 0.5, str(number) if small else "1", None, Rational(number)


def inexact_arguments(args: dict[str, int]) -> Iterator[tuple[str, dict[str, object]]]:
    """(label, args) with each argument in turn replaced by each of its inexact stand-ins."""
    for name, number in args.items():
        for item in inexact(number):
            yield f"{name}={item!r}", dict(args, **{name: item})


def pocklington_for(n: int) -> tuple:
    """A well-formed Pocklington certificate for n = 2**k + 1: n - 1 = 2**k factored, with the witness 3. It proves
    nothing unless 3 meets Pocklington's conditions, which no composite n lets it do."""
    k = (n - 1).bit_length() - 1
    assert n - 1 == 2 ** k
    return ("pocklington", n, ((2, k, ("trial", 2)),), (3,))


def claimed(operation: str, args: dict[str, int], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence),
                          relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


class ModularCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    def assertTypesMatter(self, operation: str, args: dict[str, int], value: object, evidence: tuple) -> None:
        """The checker accepts a true claim, and refuses it once an argument or the value is no exact int (a bool, a
        float, a str, None, a Rational, an int subclass, an enumeration, a record of any class), a tuple is a list or
        a tuple subclass, or the evidence is a list, a tuple subclass, None, one item longer or one item shorter."""
        self.assertAccepts(operation, args, value, evidence)
        cases: list[tuple[str, object, object, object]] = [
            (label, changed, value, evidence) for label, changed in inexact_arguments(args)]
        stand_ins = [("a record", Residue(value)), ("an impostor record", support.impostor(Residue(value))),
                     ("a forged record", support.forged(Residue, value=value))]
        if type(value) is int:
            stand_ins += [(f"value {item!r}", item) for item in inexact(value)]
            stand_ins.append(("an int subclass", Integer(value)))
            if value in (-1, 0, 1):
                stand_ins.append(("an IntEnum", Sign(value)))
        else:
            stand_ins += [("a list", list(value)), ("a tuple subclass", Evidence(value))]
            if len(value) == 2:
                stand_ins.append(("a namedtuple", Pair(*value)))
            for index, root in enumerate(value):
                for item in (*inexact(root), Integer(root)):
                    stand_ins.append((f"root {index} = {item!r}", value[:index] + (item,) + value[index + 1:]))
        cases += [(label, args, item, evidence) for label, item in stand_ins]
        cases += [("evidence as a list", args, value, list(evidence)),
                  ("evidence as a tuple subclass", args, value, Evidence(evidence)),
                  ("evidence None", args, value, None), ("an extra evidence item", args, value, evidence + ((),))]
        if evidence:
            cases.append(("the last evidence item missing", args, value, evidence[:-1]))
        self.assertRejectsAll(operation, cases)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        def integers(*names):
            return tuple(Argument(name, "int", None, "int") for name in names)

        expected = (
            OperationSpec("powmod", "ntheory", integers("base", "exponent", "modulus"), "int",
                          "Raise an integer to an integer power modulo a positive modulus; negative powers use the "
                          "inverse.", SCOPE, version=6),
            OperationSpec("multiplicative_order", "ntheory", integers("a", "n"), "int",
                          "Compute the multiplicative order of an integer modulo a positive modulus.", SCOPE,
                          version=6),
            OperationSpec("primitive_root", "ntheory", integers("p"), "int",
                          "Find the smallest primitive root modulo a prime.", SCOPE, version=6),
            OperationSpec("legendre_symbol", "ntheory", integers("a", "p"), "int",
                          "Compute the Legendre symbol (a/p) for an odd prime p.", SCOPE, version=6),
            OperationSpec("jacobi_symbol", "ntheory", integers("a", "n"), "int",
                          "Compute the Jacobi symbol (a/n) for an odd positive n.", SCOPE, version=6),
            OperationSpec("sqrt_mod", "ntheory", integers("a", "p"), "tuple[int, ...]",
                          "List every square root of an integer modulo a prime.", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {})  # Every value is an int or a tuple of ints.
        self.assertEqual(family.CHECKER, "mathforge.certify.ntheory.modular")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_powmod_accepts_true_claims(self):
        for base, exponent, modulus, value in ((7, 123456, 1000003, None), (3, -1, 7, 5), (2, 10, 1, 0),
                                               (0, 0, 5, 1), (-2, 3, 5, 2), (0, 0, 1, 0), (5, -3, 1, 0),
                                               (0, 5, 7, 0), (10 ** 40 + 4, 1, 7, 1), (-1, -1, 10, 9),
                                               (0, -1, 1, 0)):  # gcd(0, 1) = 1: 0 is invertible modulo 1.
            with self.subTest(base=base, exponent=exponent, modulus=modulus):
                if value is None:
                    value = power(base, exponent, modulus)
                    self.assertEqual(value, pow(7, 123456, 1000003))  # The card's value.
                self.assertEqual(power(base, exponent, modulus), value)
                self.assertAccepts("powmod", dict(base=base, exponent=exponent, modulus=modulus), value)
        rng = random.Random(20261112)
        for _ in range(400):
            modulus = rng.choice((1, 2, rng.randrange(1, 100), rng.randrange(1, 10 ** 6),
                                  rng.getrandbits(rng.randrange(2, 700)) | 1))
            base = rng.choice((rng.randrange(-3 * modulus - 5, 3 * modulus + 5), rng.getrandbits(1500) - 2 ** 1499))
            exponent = rng.choice((rng.randrange(-40, 300), rng.getrandbits(600) - 2 ** 599))
            if exponent < 0 and inverse(base, modulus) is None:
                continue
            with self.subTest(base=base, exponent=exponent, modulus=modulus):
                self.assertAccepts("powmod", dict(base=base, exponent=exponent, modulus=modulus),
                                   power(base, exponent, modulus))

    def test_multiplicative_order_accepts_true_claims(self):
        for a, n, d in ((2, 7, 3), (3, 7, 6), (10, 13, 6), (4, 9, 3), (1, 1, 1), (5, 1, 1)):
            with self.subTest(a=a, n=n):
                self.assertEqual(brute_order(a, n), d)
                self.assertAccepts("multiplicative_order", dict(a=a, n=n), d, (factorization(d),))
        self.assertEqual(factorization(1), ())  # The evidence of d = 1 is ((),).
        for n in range(1, 120):
            for a in range(-2, n + 2):  # A negative a, and an a of n or more, too.
                d = brute_order(a, n)
                if d is not None:
                    self.assertAccepts("multiplicative_order", dict(a=a, n=n), d, (factorization(d),))
        # Orders reduced from a known multiple: of M61 - 1 modulo the prime M61, of the Carmichael function
        # lcm(p - 1, q - 1) modulo p * q, of 2**62 modulo 2**64 and of phi(3**40) modulo 3**40.
        product = NTT[0] * NTT[1]
        carmichael = (NTT[0] - 1) * (NTT[1] - 1) // gcd(NTT[0] - 1, NTT[1] - 1)
        cases = [(a, M61, M61 - 1) for a in (2, 3, 37, M61 - 1, 2 ** 70 + 5)]
        cases += [(a, product, carmichael) for a in (3, 5, 2 ** 40 + 1, -7)]
        cases += [(a, 2 ** 64, 2 ** 62) for a in (3, 5, -1, 2 ** 63 + 1)]
        cases += [(a, 3 ** 40, 2 * 3 ** 39) for a in (2, 4, 3 ** 40 - 1)]
        for a, n, multiple in cases:
            with self.subTest(a=a, n=n):
                d = order_from(a, n, multiple)
                self.assertAccepts("multiplicative_order", dict(a=a, n=n), d, (factorization(d),))
        # Modulo SAFE, every order but 1 and 2 has the prime HALF, whose certificate is a Pocklington certificate.
        for a, d in ((2, HALF), (4, HALF), (5, 2 * HALF), (SAFE - 1, 2), (1, 1), (SAFE + 5, 2 * HALF)):
            with self.subTest(a=a, n=SAFE):
                self.assertEqual(safe_order(a), d)
                evidence = (factorization(d),)
                if d > 2:
                    self.assertEqual(evidence[0][-1][:2], (HALF, 1))
                    self.assertEqual(evidence[0][-1][2][:2], ("pocklington", HALF))
                self.assertAccepts("multiplicative_order", dict(a=a, n=SAFE), d, evidence)

    def test_primitive_root_accepts_true_claims(self):
        for p, g in ((2, 1), (3, 2), (7, 3), (23, 5), (41, 6), (71, 7)):
            with self.subTest(p=p):
                self.assertEqual(smallest_root(p, [q for q, _ in factor(p - 1)]), g)
                self.assertAccepts("primitive_root", dict(p=p), g, root_evidence(p, g))
        self.assertEqual(root_evidence(2, 1), ((), ()))
        self.assertEqual(root_evidence(41, 6)[1], (2, 5, 2, 2))  # 3 is no square modulo 41, but 3**8 = 1.
        chosen = 0
        for p in primes_below(2000):
            g = smallest_root(p, [q for q, _ in factor(p - 1)])
            with self.subTest(p=p):
                if p < 300:  # Brute force: g has the order p - 1, and no smaller h has it.
                    self.assertEqual(brute_order(g, p), p - 1)
                    self.assertTrue(all(brute_order(h, p) < p - 1 for h in range(1, g)))
                self.assertAccepts("primitive_root", dict(p=p), g, root_evidence(p, g))
                # Any prime that refutes h will do, not only the first one, which a producer would choose.
                largest = root_evidence(p, g, pick=max)
                if largest != root_evidence(p, g):
                    chosen += 1
                    self.assertAccepts("primitive_root", dict(p=p), g, largest)
        self.assertGreater(chosen, 0)
        # Large primes: p - 1 is factored by the certificates module, and the test checks the product itself.
        for p, g in ((M61, 37), (M89, 3), (M107, 3), (M127, 43), (SAFE, 5)):
            with self.subTest(p=p):
                factors = factorization(p - 1)
                self.assertEqual(prod(q ** e for q, e, _ in factors), p - 1)
                self.assertEqual(smallest_root(p, [q for q, _, _ in factors]), g)
                self.assertAccepts("primitive_root", dict(p=p), g, root_evidence(p, g))
        # SAFE - 1 = 2 * HALF, where HALF needs a Pocklington certificate. 2, 3 and 4 are squares modulo SAFE, so the
        # prime 2 refutes each of them (h**HALF = 1); HALF refutes only 1 and -1 (h**2 = 1).
        self.assertEqual(root_evidence(SAFE, 5)[1], (2, 2, 2))

    def test_legendre_symbol_accepts_true_claims(self):
        for a, p, s in ((2, 7, 1), (3, 7, -1), (0, 5, 0), (-1, 13, 1), (-1, 7, -1)):
            with self.subTest(a=a, p=p):
                roots = square_roots(p)
                self.assertEqual(0 if a % p == 0 else 1 if roots[a % p] else -1, s)
                self.assertAccepts("legendre_symbol", dict(a=a, p=p), s, (certificate(p),))
        for p in primes_below(200)[1:]:  # The odd primes, against brute force.
            roots = square_roots(p)
            for a in range(-p, 2 * p + 1):
                s = 0 if a % p == 0 else 1 if roots[a % p] else -1
                self.assertAccepts("legendre_symbol", dict(a=a, p=p), s, ((),))
        rng = random.Random(20261113)
        for p in LARGE_PRIMES:  # From M89 on the certificate is a Pocklington certificate.
            evidence = (certificate(p),)
            self.assertEqual(evidence[0] != (), p >= PSI_13)
            for a in (0, 1, -1, 2, 3, p - 1, p, 2 * p + 3, *(rng.randrange(-p, p * p) for _ in range(8))):
                with self.subTest(a=a, p=p):
                    self.assertAccepts("legendre_symbol", dict(a=a, p=p), binary_jacobi(a, p), evidence)

    def test_jacobi_symbol_accepts_true_claims(self):
        for a, n, s in ((2, 15, 1), (1001, 9907, -1), (19, 45, 1), (0, 1, 1), (3, 9, 0)):
            with self.subTest(a=a, n=n):
                self.assertEqual(jacobi_product(a, n), s)
                self.assertAccepts("jacobi_symbol", dict(a=a, n=n), s)
        rng = random.Random(20261114)
        for _ in range(1000):  # Against the product of Legendre symbols over the factorization of n.
            n = rng.randrange(1, 100_000) | 1
            a = rng.choice((rng.randrange(-n, 2 * n), rng.randrange(-10 ** 30, 10 ** 30), n * rng.randrange(-3, 4),
                            rng.randrange(n) * 4))
            s = jacobi_product(a, n)
            with self.subTest(a=a, n=n):
                self.assertEqual(binary_jacobi(a, n), s)  # The reference for long n agrees with the product.
                self.assertAccepts("jacobi_symbol", dict(a=a, n=n), s)
        for bits in (64, 1000, 4000):
            for _ in range(6):
                n = rng.getrandbits(bits) | 1 | 1 << (bits - 1)
                a = rng.choice((rng.getrandbits(bits + 40) - 2 ** (bits + 39), n - rng.randrange(1, 9) * 2 ** 40,
                                n * rng.randrange(1, 5) + 2 ** rng.randrange(1, bits)))
                with self.subTest(a=a, n=n):
                    self.assertAccepts("jacobi_symbol", dict(a=a, n=n), binary_jacobi(a, n))
        for p in LARGE_PRIMES:  # Prime moduli, where the checker compares with Euler's criterion below psi_13 too.
            for a in (2, 3, -1, p - 2):
                with self.subTest(a=a, n=p):
                    self.assertAccepts("jacobi_symbol", dict(a=a, n=p), euler(a, p))

    def test_sqrt_mod_accepts_true_claims(self):
        for a, p, roots in ((2, 7, (3, 4)), (5, 11, (4, 7)), (10, 13, (6, 7)), (3, 7, ()), (0, 13, (0,)),
                            (1, 2, (1,)), (4, 2, (0,))):
            with self.subTest(a=a, p=p):
                self.assertEqual(square_roots(p)[a % p], roots)
                self.assertAccepts("sqrt_mod", dict(a=a, p=p), roots, (certificate(p),))
        for p in primes_below(200):  # Every residue of every prime below 200, against brute force.
            roots = square_roots(p)
            for a in range(-p, 2 * p + 1):
                self.assertAccepts("sqrt_mod", dict(a=a, p=p), roots[a % p], ((),))
        rng = random.Random(20261115)
        for p in LARGE_PRIMES:  # Claims built backwards: x*x, a multiple of p, and the first non-residue.
            evidence = (certificate(p),)
            x = rng.randrange(1, p)
            non_residue = next(a for a in range(2, 1000) if binary_jacobi(a, p) == -1)
            for a, roots in ((x * x + 5 * p, tuple(sorted((x, p - x)))), (1, (1, p - 1)), (7 * p, (0,)),
                             (-p, (0,)), (non_residue, ()), (non_residue - 3 * p, ())):
                with self.subTest(a=a, p=p):
                    self.assertAccepts("sqrt_mod", dict(a=a, p=p), roots, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_rejects_claims_on_inputs_outside_the_domain(self):
        # powmod: a modulus below 1; a negative exponent of a base that is no unit, for every candidate value.
        self.assertRejectsAll("powmod", [
            (f"modulus {m}, value {v}", dict(base=2, exponent=3, modulus=m), v, ()) for m in (0, -1, -7)
            for v in (-1, 0, 1)])
        self.assertRejectsAll("powmod", [
            (f"base {b}**{e} modulo {m}, value {v}", dict(base=b, exponent=e, modulus=m), v, ())
            for b, e, m in ((2, -1, 4), (6, -3, 4), (0, -1, 5), (10, -2, 15)) for v in range(m)])
        # multiplicative_order: a modulus below 1, or a unit that a is not.
        self.assertRejectsAll("multiplicative_order", [
            (f"a={a}, n={n}, d={d}", dict(a=a, n=n), d, (factorization(d),))
            for a, n in ((2, 4), (6, 9), (0, 12), (0, 2), (12, 18), (2, 0), (3, -1), (1, -7)) for d in range(1, 13)])
        # primitive_root: a p that is not prime fails Lucas's test whatever g and the evidence (341 is a base-2
        # pseudoprime and 561 a Carmichael number).
        cases = []
        for p in (4, 6, 8, 9, 15, 21, 25, 27, 341, 561):
            factors = factorization(p - 1)
            cases += [(f"p={p}, g={g}", dict(p=p), g, (factors, (2,) * max(0, g - 2))) for g in range(1, min(p, 40))]
        cases += [(f"p={p}", dict(p=p), 1, ((), ())) for p in (1, 0, -1, -7)]
        self.assertRejectsAll("primitive_root", cases)
        # legendre_symbol: p = 2, p < 3 and composites (PSI_12 passes the strong tests to the twelve bases 2 to 37).
        self.assertRejectsAll("legendre_symbol", [
            (f"a={a}, p={p}, s={s}", dict(a=a, p=p), s, ((),))
            for a, p in ((5, 2), (1, 2), (2, 9), (2, 1), (2, 0), (2, -7), (3, 15), (2, PSI_12), (3, COMPOSITE))
            for s in (-1, 0, 1)])
        # jacobi_symbol: an even n, or one below 1.
        self.assertRejectsAll("jacobi_symbol", [
            (f"a={a}, n={n}, s={s}", dict(a=a, n=n), s, ())
            for a, n in ((2, 8), (2, 0), (2, -3), (1, 2), (3, -15), (0, -1), (1, 2 ** 70)) for s in (-1, 0, 1)])
        # sqrt_mod: a p that is not prime. Modulo 15, 1 has the four square roots 1, 4, 11 and 14.
        self.assertRejectsAll("sqrt_mod", [
            (f"a={a}, p={p}, roots={roots}", dict(a=a, p=p), roots, ((),))
            for a, p, roots in ((1, 15, (1, 4, 11, 14)), (1, 15, (1, 14)), (1, 15, (4, 11)), (1, 15, (1,)),
                                (1, 15, ()), (0, 1, (0,)), (0, 0, (0,)), (1, -7, (1,)), (0, 4, (0,)), (1, 4, (1, 3)),
                                (4, PSI_12, (2, PSI_12 - 2)), (4, COMPOSITE, (2, COMPOSITE - 2)))])

    def test_powmod_rejects_corrupted_claims(self):
        args = dict(base=7, exponent=123456, modulus=1000003)
        value = power(7, 123456, 1000003)
        inverse_args = dict(base=3, exponent=-1, modulus=7)
        self.assertTypesMatter("powmod", args, value, ())
        self.assertTypesMatter("powmod", inverse_args, 5, ())
        self.assertTypesMatter("powmod", dict(base=0, exponent=0, modulus=5), 1, ())
        self.assertTypesMatter("powmod", dict(base=2, exponent=10, modulus=1), 0, ())
        self.assertRejectsAll("powmod", [
            ("a wrong value", args, (value + 1) % 1000003, ()),
            ("another wrong value", args, (value * 7) % 1000003, ()),
            ("congruent but above the modulus", args, value + 1000003, ()),
            ("congruent but negative", args, value - 1000003, ()),
            ("3**1 instead of its inverse", inverse_args, 3, ()),
            ("a wrong inverse", inverse_args, 4, ()),
            ("the inverse plus the modulus", inverse_args, 12, ()),
            ("the inverse minus the modulus", inverse_args, -2, ()),
            ("modulo 1, a value of 1", dict(base=2, exponent=10, modulus=1), 1, ()),
            ("0**0 as 0", dict(base=0, exponent=0, modulus=5), 0, ()),
            ("evidence where none belongs", args, value, (value,)),
            ("an empty certificate as evidence", args, value, ((),)),
            ("a str as evidence", args, value, "()"),
        ])

    def test_multiplicative_order_rejects_corrupted_claims(self):
        args = dict(a=10, n=13)
        six, twelve = factorization(6), factorization(12)
        self.assertEqual(six, ((2, 1, ()), (3, 1, ())))
        self.assertTypesMatter("multiplicative_order", args, 6, (six,))
        self.assertTypesMatter("multiplicative_order", dict(a=1, n=1), 1, ((),))
        self.assertTypesMatter("multiplicative_order", dict(a=4, n=SAFE), HALF, (factorization(HALF),))
        pocklington = certificate(HALF)
        self.assertRejectsAll("multiplicative_order", [
            ("a wrong order", args, 3, (factorization(3),)),
            ("another wrong order", args, 2, (factorization(2),)),
            ("1 instead of 6", args, 1, ((),)),
            ("a multiple of the order, 12", args, 12, (twelve,)),
            ("a multiple of the order, 18", args, 18, (factorization(18),)),
            ("an order above n - 1", args, 6 + 13, (factorization(19),)),
            ("order 0", args, 0, ((),)),
            ("a negative order", args, -6, (six,)),
            ("modulo 1, an order of 2", dict(a=1, n=1), 2, (factorization(2),)),
            ("no evidence", args, 6, ()),
            ("the factorization unwrapped", args, 6, six),
            ("an empty factorization", args, 6, ((),)),
            ("a prime missing", args, 6, (six[:1],)),
            ("the other prime missing", args, 6, (six[1:],)),
            ("a composite prime", args, 6, (((6, 1, ()),),)),
            ("the factorization of another number", args, 6, (twelve,)),
            ("an extra prime", args, 6, (six + ((5, 1, ()),),)),
            ("primes unsorted", args, 6, (six[::-1],)),
            ("a repeated prime", args, 6, (((2, 1, ()), (2, 1, ()), (3, 1, ())),)),
            ("exponent 0", args, 6, (((2, 1, ()), (3, 1, ()), (5, 0, ())),)),
            ("a needless certificate", args, 6, (((2, 1, ()), (3, 1, ("trial", 3))),)),
            ("a bool exponent", args, 6, (((2, True, ()), (3, 1, ())),)),
            ("a float prime", args, 6, (((2.0, 1, ()), (3, 1, ())),)),
            ("a str prime", args, 6, ((("2", 1, ()), (3, 1, ())),)),
            ("a Rational prime", args, 6, (((Rational(2), 1, ()), (3, 1, ())),)),
            ("an int-subclass prime", args, 6, (((Integer(2), 1, ()), (3, 1, ())),)),
            ("a list entry", args, 6, (([2, 1, ()], (3, 1, ())),)),
            ("a list of factors", args, 6, (list(six),)),
            ("a list certificate", args, 6, (((2, 1, []), (3, 1, ())),)),
            ("an entry of two fields", args, 6, (((2, 1), (3, 1, ())),)),
            # HALF is above psi_13: its prime needs its own Pocklington certificate.
            ("a missing Pocklington certificate", dict(a=4, n=SAFE), HALF, (((HALF, 1, ()),),)),
            ("the certificate of another prime", dict(a=4, n=SAFE), HALF, (((HALF, 1, certificate(M89)),),)),
            ("a certificate as a list", dict(a=4, n=SAFE), HALF, (((HALF, 1, list(pocklington)),),)),
            ("a certificate of another number", dict(a=4, n=SAFE), HALF,
             (((HALF, 1, (pocklington[0], HALF + 2, *pocklington[2:])),),)),
        ])

    def test_primitive_root_rejects_corrupted_claims(self):
        args = dict(p=41)
        factors, refutations = evidence = root_evidence(41, 6)
        self.assertTypesMatter("primitive_root", args, 6, evidence)
        self.assertTypesMatter("primitive_root", dict(p=2), 1, ((), ()))
        self.assertTypesMatter("primitive_root", dict(p=SAFE), 5, root_evidence(SAFE, 5))
        seven = root_evidence(7, 3)[0]
        safe_factors = factorization(SAFE - 1)
        self.assertEqual(safe_factors[0], (2, 1, ()))
        self.assertRejectsAll("primitive_root", [
            ("a non-generator", args, 5, (factors, refutations[:3])),
            ("2, a square", args, 2, (factors, ())),
            ("1, no generator", args, 1, (factors, ())),
            ("a larger generator, 7", args, 7, (factors, refutations + (2,))),
            ("a larger generator, 7, refuting 6 by 5", args, 7, (factors, refutations + (5,))),
            ("5 modulo 7, a generator but not the smallest", dict(p=7), 5, (seven, (2, 2, 2))),
            ("the generator plus p", args, 6 + 41, (factors, refutations + (2,) * 41)),
            ("g = p", args, 41, (factors, (2,) * 39)),
            ("modulo 2, g = 0", dict(p=2), 0, ((), ())),
            ("modulo 2, g = 2", dict(p=2), 2, ((), ())),
            ("a refutation that does not refute", args, 6, (factors, (2, 2, 2, 2))),
            ("a prime that does not divide p - 1", args, 6, (factors, (2, 3, 2, 2))),
            ("a divisor of p - 1 that is not prime", args, 6, (factors, (2, 5, 4, 2))),
            ("1, which would refute every h", args, 6, (factors, (2, 5, 1, 2))),
            ("p - 1 as a refutation", args, 6, (factors, (2, 5, 2, 40))),
            ("a refutation too many", args, 6, (factors, refutations + (2,))),
            ("a refutation too few", args, 6, (factors, refutations[:-1])),
            ("refutations reversed", args, 6, (factors, refutations[::-1])),
            ("a bool refutation", args, 6, (factors, (2, 5, 2, True))),
            ("a float refutation", args, 6, (factors, (2, 5, 2, 2.0))),
            ("a str refutation", args, 6, (factors, (2, 5, 2, "2"))),
            ("a Rational refutation", args, 6, (factors, (2, 5, 2, Rational(2)))),
            ("an int-subclass refutation", args, 6, (factors, (2, 5, 2, Integer(2)))),
            ("refutations as a list", args, 6, (factors, list(refutations))),
            ("factors as a list", args, 6, (list(factors), refutations)),
            ("no refutations", args, 6, (factors,)),
            ("an extra item", args, 6, (factors, refutations, ())),
            ("factors and refutations swapped", args, 6, (refutations, factors)),
            ("the factorization of p - 2", args, 6, (factorization(39), refutations)),
            ("a prime of p - 1 missing", args, 6, (factors[:-1], refutations)),
            ("a composite prime", args, 6, (((4, 1, ()), (10, 1, ())), refutations)),
            ("an exponent too large", args, 6, (((2, 4, ()), (5, 1, ())), refutations)),
            ("modulo 2, refutations", dict(p=2), 1, ((), (2,))),
            ("modulo 2, a factorization of 1", dict(p=2), 1, (((1, 1, ()),), ())),
            ("modulo 2, one item", dict(p=2), 1, ((),)),
            # SAFE - 1 = 2 * HALF, and HALF needs its Pocklington certificate.
            ("a missing Pocklington certificate", dict(p=SAFE), 5, ((safe_factors[0], (HALF, 1, ())), (2, 2, 2))),
            ("the certificate of another prime", dict(p=SAFE), 5,
             ((safe_factors[0], (HALF, 1, certificate(M89))), (2, 2, 2))),
            ("HALF, which refutes only 1 and -1", dict(p=SAFE), 5, (safe_factors, (HALF, 2, 2))),
            # A composite p fails Lucas's test with the true factorization of p - 1.
            ("a composite p above psi_13", dict(p=COMPOSITE), 3, (((2, 100, ()),), (2,))),
            ("PSI_12, a composite p", dict(p=PSI_12), 2, (factorization(PSI_12 - 1), ())),
        ])

    def test_legendre_symbol_rejects_corrupted_claims(self):
        args = dict(a=3, p=7)
        self.assertTypesMatter("legendre_symbol", args, -1, ((),))
        self.assertTypesMatter("legendre_symbol", dict(a=2, p=7), 1, ((),))
        self.assertTypesMatter("legendre_symbol", dict(a=0, p=5), 0, ((),))
        large = dict(a=3, p=M89)
        s = euler(3, M89)
        pocklington = certificate(M89)
        self.assertTypesMatter("legendre_symbol", large, s, (pocklington,))
        kind, recorded, factors, witnesses = pocklington
        self.assertRejectsAll("legendre_symbol", [
            ("a wrong sign", args, 1, ((),)),
            ("0 for a unit", args, 0, ((),)),
            ("a sign for a multiple of p", dict(a=14, p=7), 1, ((),)),
            ("Euler's power instead of -1", args, 6, ((),)),
            ("-1 + p", args, 6, ((),)),
            ("2", dict(a=2, p=7), 2, ((),)),
            ("no certificate", args, -1, ()),
            ("a certificate too many", args, -1, ((), ())),
            ("a list certificate", args, -1, ([],)),
            ("a None certificate", args, -1, (None,)),
            ("a str certificate", args, -1, ("()",)),
            ("a needless certificate", args, -1, (("trial", 7),)),
            ("a wrong sign modulo M89", large, -s, (pocklington,)),
            ("an empty certificate above psi_13", large, s, ((),)),
            ("the certificate of another prime", large, s, (certificate(M107),)),
            ("a certificate of another number", large, s, ((kind, M89 + 2, factors, witnesses),)),
            ("a certificate as a list", large, s, (list(pocklington),)),
            ("a truncated certificate", large, s, (pocklington[:3],)),
            ("a prime of p - 1 dropped", large, s, ((kind, recorded, factors[:-1], witnesses[:-1]),)),
            ("a composite p with a well-formed certificate", dict(a=3, p=COMPOSITE), 1,
             (pocklington_for(COMPOSITE),)),
            ("a composite p, the other sign", dict(a=3, p=COMPOSITE), -1, (pocklington_for(COMPOSITE),)),
            ("PSI_12 with no certificate", dict(a=2, p=PSI_12), 1, ((),)),
            ("PSI_12, the other sign", dict(a=2, p=PSI_12), -1, ((),)),
        ])

    def test_jacobi_symbol_rejects_corrupted_claims(self):
        self.assertTypesMatter("jacobi_symbol", dict(a=2, n=15), 1, ())
        self.assertTypesMatter("jacobi_symbol", dict(a=1001, n=9907), -1, ())
        self.assertTypesMatter("jacobi_symbol", dict(a=3, n=9), 0, ())
        self.assertTypesMatter("jacobi_symbol", dict(a=5, n=1), 1, ())
        self.assertRejectsAll("jacobi_symbol", [
            ("a wrong sign", dict(a=2, n=15), -1, ()),
            ("0 for a unit", dict(a=2, n=15), 0, ()),
            ("a sign for a common factor", dict(a=3, n=9), 1, ()),
            ("the other sign for a common factor", dict(a=3, n=9), -1, ()),
            ("-1 for a square n, 9", dict(a=2, n=9), -1, ()),
            ("(a/1) = 0", dict(a=0, n=1), 0, ()),
            ("(a/1) = -1", dict(a=5, n=1), -1, ()),
            ("the wrong sign modulo a prime", dict(a=1001, n=9907), 1, ()),
            ("-1 + n", dict(a=1001, n=9907), 9906, ()),
            ("1 + n", dict(a=2, n=15), 16, ()),
            ("2", dict(a=2, n=15), 2, ()),
            ("evidence where none belongs", dict(a=2, n=15), 1, ((),)),
            ("a str as evidence", dict(a=2, n=15), 1, ""),
        ])

    def test_sqrt_mod_rejects_corrupted_claims(self):
        args = dict(a=10, p=13)
        self.assertTypesMatter("sqrt_mod", args, (6, 7), ((),))
        self.assertTypesMatter("sqrt_mod", dict(a=1, p=2), (1,), ((),))
        self.assertTypesMatter("sqrt_mod", dict(a=0, p=13), (0,), ((),))
        self.assertTypesMatter("sqrt_mod", dict(a=3, p=7), (), ((),))
        pocklington = certificate(M89)
        x = 3 ** 40 % M89
        large_roots = tuple(sorted((x, M89 - x)))
        large = dict(a=x * x, p=M89)
        self.assertTypesMatter("sqrt_mod", large, large_roots, (pocklington,))
        self.assertRejectsAll("sqrt_mod", [
            ("wrong roots", args, (5, 8), ((),)),
            ("one root only", args, (6,), ((),)),
            ("the other root only", args, (7,), ((),)),
            ("no root for a residue", args, (), ((),)),
            ("a third root", args, (0, 6, 7), ((),)),
            ("unsorted roots", args, (7, 6), ((),)),
            ("a root above p", args, (6, 7 + 13), ((),)),
            ("a negative root", args, (-7, 7), ((),)),
            ("a root twice", args, (6, 6), ((),)),
            ("roots that do not add up to p", args, (6, 20), ((),)),
            ("0 among the roots", args, (0, 6), ((),)),
            ("roots of a non-residue", dict(a=3, p=7), (1, 6), ((),)),
            ("other roots of a non-residue", dict(a=3, p=7), (2, 5), ((),)),
            ("a root of a non-residue", dict(a=3, p=7), (0,), ((),)),
            ("modulo 2, two roots", dict(a=1, p=2), (0, 1), ((),)),
            ("modulo 2, no root", dict(a=1, p=2), (), ((),)),
            ("modulo 2, the wrong root", dict(a=1, p=2), (0,), ((),)),
            ("modulo 2, a root above 2", dict(a=1, p=2), (3,), ((),)),
            ("modulo 2, a root of 0 as 1", dict(a=4, p=2), (1,), ((),)),
            ("p dividing a, no root", dict(a=0, p=13), (), ((),)),
            ("p dividing a, 0 twice", dict(a=0, p=13), (0, 0), ((),)),
            ("p dividing a, p as the root", dict(a=26, p=13), (13,), ((),)),
            ("p dividing a, 0 and p", dict(a=0, p=13), (0, 13), ((),)),
            ("no certificate", args, (6, 7), ()),
            ("a certificate too many", args, (6, 7), ((), ())),
            ("a list certificate", args, (6, 7), ([],)),
            ("a None certificate", args, (6, 7), (None,)),
            ("a needless certificate", args, (6, 7), (("trial", 13),)),
            ("an empty certificate above psi_13", large, large_roots, ((),)),
            ("the certificate of another prime", large, large_roots, (certificate(M107),)),
            ("a certificate as a list", large, large_roots, (list(pocklington),)),
            ("a composite p with a well-formed certificate", dict(a=4, p=COMPOSITE), (2, COMPOSITE - 2),
             (pocklington_for(COMPOSITE),)),
            ("no root modulo a composite p", dict(a=3, p=COMPOSITE), (), (pocklington_for(COMPOSITE),)),
        ])

    def test_rejects_wrong_operations_and_arguments(self):
        args, value, evidence = dict(a=2, p=7), 1, ((),)
        self.assertAccepts("legendre_symbol", args, value, evidence)
        self.assertRejectsAll("legendre_symbol", [
            ("arguments as pairs", tuple(args.items()), value, evidence),
            ("arguments as a list of pairs", list(args.items()), value, evidence),
            ("arguments as a read-only mapping", MappingProxyType(args), value, evidence),
            ("None as arguments", None, value, evidence),
            ("an argument missing", dict(a=2), value, evidence),
            ("an extra argument", dict(a=2, p=7, n=15), value, evidence),
            ("an argument renamed", dict(a=2, n=7), value, evidence),
            ("an argument of another operation", dict(base=2, p=7), value, evidence),
        ])
        for operation in ("jacobi_symbol", "sqrt_mod", "primitive_root", "Legendre_symbol", "legendre_symbol ",
                          "legendre", "continued_fraction", "", None, 7, ("legendre_symbol",), b"legendre_symbol",
                          ["legendre_symbol"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)
        # A true claim of one operation, under the name of another with the same arguments.
        self.assertRejects("jacobi_symbol", dict(a=2, n=15), 1, ((),))
        self.assertRejects("legendre_symbol", dict(a=2, p=7), (3, 4), ((),))
        self.assertRejects("sqrt_mod", dict(a=2, p=7), 1, ((),))
        self.assertRejects("multiplicative_order", dict(a=2, n=15), 1, ())

    def test_oversized_forgeries_fail_fast(self):
        giant = 2 ** 70_000  # Longer than max_integer_bits allows.
        cases = [
            ("powmod", dict(base=3, exponent=5, modulus=7), giant, ()),
            ("powmod", dict(base=3, exponent=5, modulus=7), -giant, ()),
            ("powmod", dict(base=3, exponent=5, modulus=7), 5, (giant,)),
            ("powmod", dict(base=3, exponent=5, modulus=7), 5, ((0,) * 100_000)),
            ("multiplicative_order", dict(a=2, n=7), giant, (((2, 70_000, ()),),)),
            ("multiplicative_order", dict(a=2, n=7), support.HUGE, (((2, 65_536, ()),),)),
            ("multiplicative_order", dict(a=2, n=7), 3, (((3, 10 ** 100, ()),),)),
            ("multiplicative_order", dict(a=2, n=7), 3, (((3, giant, ()),),)),
            ("multiplicative_order", dict(a=2, n=7), 3, (tuple((q, 1, ()) for q in primes_below(100_000)),)),
            ("multiplicative_order", dict(a=3, n=M127), 2 ** 126, (((2, 126, ()),),)),
            ("primitive_root", dict(p=7), giant, (factorization(6), ())),
            ("primitive_root", dict(p=7), 3, (factorization(6), (2,) * 1_000_000)),
            ("primitive_root", dict(p=7), 3, (factorization(6), (giant,))),
            ("primitive_root", dict(p=M127), 10 ** 6, (factorization(M127 - 1), (2,) * (10 ** 6 - 2))),
            ("primitive_root", dict(p=M127), 43, (factorization(M127 - 1), (giant,) * 41)),
            ("legendre_symbol", dict(a=3, p=7), giant, ((),)),
            ("legendre_symbol", dict(a=3, p=7), -1, ((0,) * 100_000,)),
            ("legendre_symbol", dict(a=3, p=7), -1, ((giant,),)),
            ("jacobi_symbol", dict(a=2, n=15), giant, ()),
            ("jacobi_symbol", dict(a=2, n=15), 1, (0,) * 100_000),
            ("sqrt_mod", dict(a=2, p=7), tuple(range(100_000)), ((),)),
            ("sqrt_mod", dict(a=2, p=7), (3, giant), ((),)),
            ("sqrt_mod", dict(a=2, p=7), (giant, giant + 7), ((),)),
        ]
        for index, (operation, args, value, evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # From psi_13 on, prime_proof charges a Pocklington certificate before check_prime reads it, so a nested node
        # that claims a giant number may exhaust the budget at once instead: the claim then stays unverified, as the
        # verifier reports exhausted resources, and is never verified.
        kind, recorded, factors, witnesses = certificate(M89)
        giant_prime = 2 ** 60_000 + 1
        nested = ("pocklington", giant_prime, ((2, 60_000, ("trial", 2)),), (3,))
        forged = (kind, recorded, factors[:-1] + ((giant_prime, 1, nested),), witnesses)
        x = 3 ** 40 % M89
        for operation, args, value in (("legendre_symbol", dict(a=3, p=M89), euler(3, M89)),
                                       ("sqrt_mod", dict(a=x * x, p=M89), tuple(sorted((x, M89 - x))))):
            with self.subTest(operation=operation, certificate="a giant nested node"):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, (forged,), may_exhaust=True)
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_before_it_runs(self):
        calls = []

        def recording(*arguments):
            calls.append(arguments)
            return pow(*arguments)

        modulus = 2 ** 4096 - 1
        args = dict(base=3, exponent=2 ** 4000 + 1, modulus=modulus)
        value = power(3, 2 ** 4000 + 1, modulus)
        with patch.object(helpers, "pow", recording, create=True):
            # 4001 multiplications modulo a 4096-bit modulus cost about 516,000 units: refused before pow runs.
            with self.assertRaises(ResourceLimitError):
                support.verdict(family, "powmod", args, value, limits=ComputationLimits(max_work=100_000))
            self.assertEqual(calls, [])
            self.assertIs(support.verdict(family, "powmod", args, value), True)
            self.assertEqual(len(calls), 1)
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation("powmod", args, value, ()), True)
        self.assertGreaterEqual(budget.work, 4001 * helpers.multiplication_work(4096))
        # Every pass of the Jacobi loop is charged: a long n runs out of a small budget at once.
        rng = random.Random(20261116)
        n = rng.getrandbits(65_000) | 1 | 1 << 64_999
        a = rng.getrandbits(64_000)
        start = time.perf_counter()
        for s in (-1, 0, 1):
            with self.assertRaises(ResourceLimitError):
                support.verdict(family, "jacobi_symbol", dict(a=a, n=n), s, limits=ComputationLimits(max_work=5_000))
        self.assertLess(time.perf_counter() - start, FAST)
        # The charges depend on the claim alone: a budget one unit short of a check refuses it, its exact size passes.
        evidence = root_evidence(M127, 43)
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation("primitive_root", dict(p=M127), 43, evidence), True)
        work = budget.work
        self.assertGreater(work, 41)  # Every refutation is charged.
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "primitive_root", dict(p=M127), 43, evidence,
                            limits=ComputationLimits(max_work=work - 1))
        self.assertAccepts("primitive_root", dict(p=M127), 43, evidence, limits=ComputationLimits(max_work=work))

    def test_full_size_claims_fit_the_default_budget(self):
        """Claims on integers of nearly max_integer_bits verify under the default budget. Their values are known by
        construction: (a/(3 m**2)) = (a/3) (a/m)**2 is (a/3) when gcd(a, m) = 1, and 0 when they share a factor."""
        rng = random.Random(20261117)
        m = rng.getrandbits(32_766) | 1 | 1 << 32_765
        n = 3 * m * m
        self.assertLessEqual(n.bit_length(), 65_536)
        a = 3 * rng.getrandbits(65_000) + 2
        while gcd(a, m) != 1:
            a += 3
        self.assertAccepts("jacobi_symbol", dict(a=a, n=n), -1)  # (a/3) = -1 for a = 2 (mod 3).
        self.assertAccepts("jacobi_symbol", dict(a=a + 1, n=n), 0)  # 3 divides a + 1 and n.
        self.assertRejects("jacobi_symbol", dict(a=a, n=n), 1)
        modulus = rng.getrandbits(65_536) | 1 << 65_535
        base = rng.getrandbits(65_536)
        self.assertAccepts("powmod", dict(base=base, exponent=5, modulus=modulus), power(base, 5, modulus))

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_jacobi_anchors_are_live(self):
        """The two anchors of the Jacobi check reject a value that the least-absolute-remainder algorithm, patched to
        answer wrongly, would let through."""
        with patch.object(checker, "_jacobi", lambda budget, a, n: -1):
            self.assertRejects("jacobi_symbol", dict(a=2, n=7), -1)  # Euler's criterion: (2/7) = 1.
            self.assertRejects("jacobi_symbol", dict(a=2, n=10007), -1)  # Euler, by the strong tests: 10007 = 7 mod 8.
            self.assertRejects("jacobi_symbol", dict(a=6, n=9), -1)  # gcd(6, 9) = 3: the symbol is 0.
            self.assertAccepts("jacobi_symbol", dict(a=3, n=7), -1)  # (3/7) = -1 indeed.
        with patch.object(checker, "_jacobi", lambda budget, a, n: 0):
            self.assertRejects("jacobi_symbol", dict(a=2, n=15), 0)  # gcd(2, 15) = 1: the symbol is not 0.

    def test_every_power_goes_through_charged_pow(self):
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("charged_pow", called)
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])  # No ** either: every power is charged before it is formed.

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layouts
        survive the format, and a wrong value fails the mathematics check."""
        x = 3 ** 40 % M89
        claims = [
            ("powmod", dict(base=3, exponent=-1, modulus=7), 5, (), 4),
            ("multiplicative_order", dict(a=4, n=SAFE), HALF, (factorization(HALF),), 2 * HALF),
            ("primitive_root", dict(p=SAFE), 5, root_evidence(SAFE, 5), 7),
            ("legendre_symbol", dict(a=3, p=M89), euler(3, M89), (certificate(M89),), -euler(3, M89)),
            ("jacobi_symbol", dict(a=1001, n=9907), -1, (), 1),
            ("sqrt_mod", dict(a=x * x, p=M89), tuple(sorted((x, M89 - x))), (certificate(M89),), ()),
        ]
        for operation, args, value, evidence, wrong in claims:
            with self.subTest(operation=operation):
                result = claimed(operation, args, value, evidence)
                self.roundtrip(result)
                report = support.statuses(mf.verify(claimed(operation, args, wrong, evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
