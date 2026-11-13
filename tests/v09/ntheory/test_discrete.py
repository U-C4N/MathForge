"""The discrete family of mathforge.ntheory: discrete_log, two_squares.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register ntheory discrete

Every expected value comes from the card or from a reference of this module, written here and slow on purpose (walking
the powers of the base, counting the lattice points of a disc, trial division) or independent of the producer (the
builtin pow on known factorizations of p - 1); never from the producer. tests/v09/ntheory/test_discrete_checker.py
tests the checker on its own.
"""

from __future__ import annotations

import ast
import inspect
from math import gcd, isqrt, prod
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.ntheory import discrete as family
from mathforge.areas.ntheory.discrete import DiscreteLogarithm, SumOfTwoSquares
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Rational
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13, is_prime

from .. import support

FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
PSI_12 = 318665857834031151167461  # 399165290221 * 798330580441, a strong pseudoprime to the twelve bases 2 to 37.
COMPOSITE = 2 ** 100 + 1  # Above psi_13; 2**20 + 1 divides it.
M61, M89, M127 = 2 ** 61 - 1, 2 ** 89 - 1, 2 ** 127 - 1  # Mersenne primes; from M89 on above psi_13.
HALF = PSI_13 + 5062  # A prime above psi_13, so that its certificate is a Pocklington certificate, ...
SAFE = 2 * HALF + 1  # ... and SAFE is prime too; 2, 3 and 4 are squares modulo SAFE, 5 is not.
F4 = 65537  # The Fermat prime 2**16 + 1, where 3 generates (Z/F4)^*.
PROTH = 13 * 2 ** 1000 + 1  # A prime (Proth's theorem, witness 3): its group order 2**1000 * 13 has a long 2-part.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact int: a float, a bool, a str, None and two containers.
# The prime factorizations of p - 1 for the large primes, multiplied out and checked by trial division in the tests.
GROUPS = {
    M61: ((2, 1), (3, 2), (5, 2), (7, 1), (11, 1), (13, 1), (31, 1), (41, 1), (61, 1), (151, 1), (331, 1),
          (1321, 1)),
    M89: ((2, 1), (3, 1), (5, 1), (17, 1), (23, 1), (89, 1), (353, 1), (397, 1), (683, 1), (2113, 1),
          (2931542417, 1)),
    F4: ((2, 16),),
    PROTH: ((2, 1000), (13, 1)),
}

# One small request per operation, the lean inputs.
SMALL = {
    "discrete_log": dict(base=2, target=3, p=11),
    "two_squares": dict(n=25),
}

# The known values of the card: (operation, arguments, value). two_squares names a representation of the producer's
# choice, so its known values carry only the count, and the representation is tested to be one.
KNOWN_LOGS = (
    (dict(base=2, target=3, p=11), DiscreteLogarithm(8, 10)),
    (dict(base=3, target=13, p=17), DiscreteLogarithm(4, 16)),
    (dict(base=2, target=3, p=7), DiscreteLogarithm(None, 3)),
    (dict(base=5, target=1, p=7), DiscreteLogarithm(0, 6)),
    (dict(base=2, target=0, p=11), DiscreteLogarithm(None, 10)),
)
KNOWN_COUNTS = ((0, 1), (1, 4), (2, 4), (3, 0), (25, 12), (65, 16), (21, 0), (M89, 0))
CARD_SQUARES = {0: (0, 0), 1: (0, 1), 2: (1, 1)}  # The card names the representation where there is only one.

# Inputs outside the domains of the card, each invalid input.
INVALID = (
    ("discrete_log", dict(base=2, target=3, p=1)),
    ("discrete_log", dict(base=2, target=3, p=0)),
    ("discrete_log", dict(base=2, target=3, p=-7)),
    ("discrete_log", dict(base=2, target=3, p=4)),
    ("discrete_log", dict(base=2, target=3, p=9)),
    ("discrete_log", dict(base=2, target=3, p=341)),  # A base-2 pseudoprime.
    ("discrete_log", dict(base=2, target=3, p=561)),  # A Carmichael number.
    ("discrete_log", dict(base=2, target=3, p=PSI_12)),
    ("discrete_log", dict(base=3, target=2, p=COMPOSITE)),
    ("discrete_log", dict(base=0, target=3, p=7)),
    ("discrete_log", dict(base=14, target=3, p=7)),
    ("discrete_log", dict(base=-11, target=1, p=11)),
    ("discrete_log", dict(base=0, target=0, p=2)),
    ("discrete_log", dict(base=4, target=1, p=2)),
    ("discrete_log", dict(base=M89, target=1, p=M89)),
    ("two_squares", dict(n=-1)),
    ("two_squares", dict(n=-25)),
    ("two_squares", dict(n=-(2 ** 100))),
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


def brute_log(base: int, target: int, p: int) -> DiscreteLogarithm:
    """The least k >= 0 with base**k = target (mod p) and the order of base, by walking base**0, base**1, ... until
    the power returns to 1; the exponent is None when target is not met before that."""
    goal, x, first = target % p, 1, None
    for k in range(p):
        if first is None and x == goal:
            first = k
        x = x * base % p
        if x == 1:
            return DiscreteLogarithm(first, k + 1)
    raise AssertionError(f"{base} is no unit modulo {p}")


def order_from(base: int, p: int, group: tuple[tuple[int, int], ...]) -> int:
    """The multiplicative order of base modulo the prime p, from the known factorization of p - 1: each prime is
    divided out of p - 1 while the builtin pow of base to the quotient stays 1."""
    order = p - 1
    for q, e in group:
        for _ in range(e):
            if pow(base, order // q, p) != 1:
                break
            order //= q
    return order


def brute_r2(n: int) -> int:
    """r2(n): the pairs (x, y) of integers with x*x + y*y = n, one y-test per x in [-isqrt(n), isqrt(n)]."""
    count = 0
    for x in range(-isqrt(n), isqrt(n) + 1):
        rest = n - x * x
        y = isqrt(rest)
        if y * y == rest:
            count += 1 if y == 0 else 2  # (x, y) and (x, -y).
    return count


def jacobi_count(n: int) -> int:
    """r2(n) = 4 (d1(n) - d3(n)) for n >= 1, counting the divisors of n that are 1 and 3 modulo 4 by trial division."""
    d1 = d3 = 0
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            for divisor in {d, n // d}:
                if divisor % 4 == 1:
                    d1 += 1
                elif divisor % 4 == 3:
                    d3 += 1
    return 4 * (d1 - d3)


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


class DiscreteTests(support.FamilyTestCase):
    FAMILY = family

    def assertLog(self, args: dict[str, int], expected: DiscreteLogarithm) -> tuple:
        """produce() answers expected, and the checker accepts that answer with its evidence, which is returned."""
        value, evidence = family.produce("discrete_log", dict(args))
        self.assertEqual(value, expected)
        self.assertIs(type(value), DiscreteLogarithm)
        self.assertIs(support.verdict(family, "discrete_log", args, value, evidence), True)
        return evidence

    def assertSquares(self, n: int, count: int) -> SumOfTwoSquares:
        """produce() answers the count r2(n) and a representation 0 <= a <= b with a*a + b*b = n (None, None for a
        count of 0), and the checker accepts the answer with its evidence; the value is returned."""
        value, evidence = family.produce("two_squares", {"n": n})
        self.assertIs(type(value), SumOfTwoSquares)
        self.assertEqual(value.count, count)
        if count:
            self.assertTrue(type(value.a) is int and type(value.b) is int, value)
            self.assertTrue(0 <= value.a <= value.b, value)
            self.assertEqual(value.a * value.a + value.b * value.b, n)
        else:
            self.assertEqual((value.a, value.b), (None, None))
        self.assertIs(support.verdict(family, "two_squares", {"n": n}, value, evidence), True)
        return value

    def assertVerified(self, result: mf.Result, value: object = None) -> None:
        """A completed v6 Result (of that value, when one is given), with its four checks, which mf.verify
        accepts."""
        self.assertIsNone(result.error)
        if value is not None:
            self.assertEqual(result.value, value)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for args, value in KNOWN_LOGS:
            with self.subTest(operation="discrete_log", **args):
                self.assertEqual(brute_log(**args), value)  # The card's value, by brute force.
                result = support.run("discrete_log", **args)
                self.assertVerified(result, value)
                factors, certificate = result.steps[0].outputs[1]
                self.assertEqual(factors, tuple((q, e, ()) for q, e in factor(value.order)))
                self.assertEqual(certificate, ())
                self.roundtrip(result)
        for n, count in KNOWN_COUNTS:
            with self.subTest(operation="two_squares", n=n):
                if n < 10 ** 6:
                    self.assertEqual(brute_r2(n), count)  # The card's count, by counting lattice points.
                result = support.run("two_squares", n=n)
                self.assertVerified(result)
                self.assertEqual(result.value.count, count)
                if n in CARD_SQUARES:
                    self.assertEqual((result.value.a, result.value.b), CARD_SQUARES[n])
                if count:
                    self.assertEqual(result.value.a ** 2 + result.value.b ** 2, n)
                else:
                    self.assertEqual((result.value.a, result.value.b), (None, None))
                (factors,) = result.steps[0].outputs[1]
                if n < 2:
                    self.assertEqual(factors, ())
                elif n < 10 ** 6:
                    self.assertEqual(tuple(entry[:2] for entry in factors), factor(n))
                self.roundtrip(result)
        # M89 is prime and 3 (mod 4): one factor with a Pocklington certificate, no representation.
        (factors,) = support.run("two_squares", n=M89).steps[0].outputs[1]
        self.assertEqual(factors[0][:2], (M89, 1))
        self.assertEqual(factors[0][2][:2], ("pocklington", M89))
        # Large primes: the order from the known factorization of p - 1, the exponent of a power built here.
        for p, base, k in ((M61, 3, 10 ** 15 + 7), (M89, 3, 12345678901234567), (PROTH, 3, 10 ** 40 + 7),
                           (F4, 3, 12345)):
            with self.subTest(operation="discrete_log", p=p):
                order = order_from(base, p, GROUPS[p])
                result = support.run("discrete_log", base=base, target=pow(base, k, p), p=p)
                self.assertVerified(result, DiscreteLogarithm(k % order, order))
                if p >= PSI_13:
                    self.assertEqual(result.steps[0].outputs[1][1][:2], ("pocklington", p))
        # SAFE - 1 = 2 * HALF: 4 is a square, of the prime order HALF, far beyond the baby-step giant-step search,
        # but 5 is no square (5**HALF != 1), so no power of 4: None needs no search, and the evidence certifies HALF.
        self.assertNotEqual(pow(5, HALF, SAFE), 1)
        self.assertEqual(pow(4, HALF, SAFE), 1)
        result = support.run("discrete_log", base=4, target=5, p=SAFE)
        self.assertVerified(result, DiscreteLogarithm(None, HALF))
        factors, certificate = result.steps[0].outputs[1]
        self.assertEqual((factors[0][:2], factors[0][2][:2], certificate[:2]),
                         ((HALF, 1), ("pocklington", HALF), ("pocklington", SAFE)))

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
        messages = (("discrete_log", dict(base=2, target=3, p=9), "a prime p"),
                    ("discrete_log", dict(base=2, target=3, p=1), "a prime p"),
                    ("discrete_log", dict(base=14, target=3, p=7), "base % p != 0"),
                    ("two_squares", dict(n=-1), "n >= 0"))
        for operation, args, text in messages:
            with self.subTest(operation=operation, message=text):
                self.assertIn(text, support.run(operation, **args).error.message)
        # operations_v6.run rejects a stand-in by its kind before produce() runs; produce() refuses one itself.
        for operation, args in SMALL.items():
            for name in args:
                for item in (True, False, 0.5, "1", None, Rational(3), Integer(3), (1, 2), [1]):
                    with self.subTest(operation=operation, argument=name, item=item, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(args, **{name: item}))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(args, extra=1))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, tuple(args.items()))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, {})
        for operation in ("powmod", "Two_squares", "", None):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(n=25))
        # Regression (review of wave m7b): base % p == 0 is decided before p is proved prime, so a prime that the
        # default budget cannot certify (M4423: p - 1 is beyond the integer kernel) still gives invalid_input there.
        m4423 = 2 ** 4423 - 1
        start = time.perf_counter()
        self.assertResourceLimit(support.run("discrete_log", base=3, target=2, p=m4423))
        for base in (0, m4423, -2 * m4423):
            with self.subTest(base=base, p="M4423"):
                result = support.run("discrete_log", base=base, target=2, p=m4423)
                self.assertInvalidInput(result)
                self.assertIn("base % p != 0", result.error.message)
        self.assertLess(time.perf_counter() - start, FAST)
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value("discrete_log", base=2, target=3, p=9)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 and max_work=1 exceed the budget. No operation of the family takes a matrix, so
        support.oversized_matrix() has nothing to exceed here (as an int argument it is invalid input). A prime of
        the order too large for the baby-step giant-step search within the budget fails at once, and so does a
        factorization that the integer kernel cannot finish."""
        for operation, args in SMALL.items():
            for name in args:
                with self.subTest(operation=operation, argument=name):
                    self.assertResourceLimit(support.run(operation, **dict(args, **{name: support.HUGE})))
                    self.assertResourceLimit(support.run(operation, **dict(args, **{name: -support.HUGE})))
                    self.assertInvalidInput(support.run(operation, **dict(args, **{name: support.oversized_matrix()})))
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
        # Safe primes p = 2q + 1, where the order of 2 or 3 is q or 2q: q of 40 bits needs about 2**21 baby steps,
        # past the default max_work of 2,000,000, which the search charges before it starts.
        q40, q37 = 1099511627933, 137438954063
        self.assertTrue(all(is_prime(n) for n in (q40, 2 * q40 + 1, q37, 2 * q37 + 1)))  # Exact below psi_13.
        self.assertEqual(pow(3, q40, 2 * q40 + 1), 1)  # 3 has the prime order q40.
        start = time.perf_counter()
        self.assertResourceLimit(support.run("discrete_log", base=3, target=pow(3, 10 ** 11, 2 * q40 + 1),
                                             p=2 * q40 + 1))
        self.assertLess(time.perf_counter() - start, FAST)
        # A target outside the subgroup needs no search: None, at once.
        self.assertVerified(support.run("discrete_log", base=3, target=2 * q40, p=2 * q40 + 1),
                            DiscreteLogarithm(None, q40))
        # The same search for q of 37 bits fits the default budget (the card's limit is about 2**36).
        result = support.run("discrete_log", base=3, target=pow(3, 10 ** 11, 2 * q37 + 1), p=2 * q37 + 1)
        self.assertVerified(result)
        self.assertEqual(pow(3, result.value.exponent, 2 * q37 + 1), pow(3, 10 ** 11, 2 * q37 + 1))
        # A product of two 60-bit primes, the second squared: the kernel's rho runs out of the default budget.
        start = time.perf_counter()
        self.assertResourceLimit(support.run("two_squares", n=M61 * (2 ** 89 - 1) * M61))
        self.assertResourceLimit(support.run("two_squares", n=5 ** 20, limits=ComputationLimits(max_work=20)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with a representation that the producer did not choose and
        with Pocklington certificates of other witnesses."""
        others = 0
        for n in range(2, 400):
            value, evidence = family.produce("two_squares", {"n": n})
            if not value.count:
                continue
            for a in range(isqrt(n // 2) + 1):
                b = isqrt(n - a * a)
                if b * b == n - a * a and (a, b) != (value.a, value.b):
                    others += 1
                    with self.subTest(n=n, a=a, b=b):
                        self.assertAccepts("two_squares", {"n": n}, SumOfTwoSquares(value.count, a, b), evidence)
        self.assertGreater(others, 20)
        value, evidence = family.produce("two_squares", {"n": 25})
        other = (0, 5) if (value.a, value.b) == (3, 4) else (3, 4)
        self.assertAccepts("two_squares", {"n": 25}, SumOfTwoSquares(12, *other), evidence)

        def rewitnessed(certificate: tuple) -> tuple:
            """The Pocklington certificate with, for each prime q of n - 1, the least witness other than its own."""
            kind, n, factors, witnesses = certificate
            others = tuple(next(w for w in range(2, n) if w != witness and pow(w, n - 1, n) == 1
                                and gcd(pow(w, (n - 1) // q, n) - 1, n) == 1)
                           for (q, _, _), witness in zip(factors, witnesses))
            self.assertNotEqual(others, witnesses)
            return kind, n, factors, others

        args = dict(base=3, target=pow(3, 10 ** 20, M89), p=M89)
        value, (factors, certificate) = family.produce("discrete_log", args)
        self.assertAccepts("discrete_log", args, value, (factors, rewitnessed(certificate)))
        args = dict(base=4, target=5, p=SAFE)  # 4 is a square, of the prime order HALF, and 5 is none.
        value, (factors, certificate) = family.produce("discrete_log", args)
        self.assertEqual(value, DiscreteLogarithm(None, HALF))
        self.assertAccepts("discrete_log", args, value, (((HALF, 1, rewitnessed(factors[0][2])),), certificate))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        log_args, squares_args = SMALL["discrete_log"], SMALL["two_squares"]
        log_result, squares_result = support.run("discrete_log", **log_args), support.run("two_squares", **squares_args)
        log_value, log_evidence = log_result.value, log_result.steps[0].outputs[1]
        sq_value, sq_evidence = squares_result.value, squares_result.steps[0].outputs[1]
        forged_log, forged_sq = (lambda **fields: support.forged(DiscreteLogarithm, **fields),
                                 lambda **fields: support.forged(SumOfTwoSquares, **fields))
        log_cases = [
            ("wrong exponent", DiscreteLogarithm(7, 10), log_evidence),
            ("exponent None", DiscreteLogarithm(None, 10), log_evidence),
            ("wrong order", DiscreteLogarithm(3, 5), (((5, 1, ()),), ())),
            ("order p - 1 with its factors swapped", DiscreteLogarithm(8, 10), (((5, 1, ()), (2, 1, ())), ())),
            ("exponent + order", forged_log(exponent=18, order=10), log_evidence),
            ("negative exponent", forged_log(exponent=-2, order=10), log_evidence),
            ("order 0", forged_log(exponent=None, order=0), log_evidence),
            ("exponent a bool", forged_log(exponent=True, order=10), log_evidence),
            ("exponent a float", forged_log(exponent=8.0, order=10), log_evidence),
            ("exponent a str", forged_log(exponent="8", order=10), log_evidence),
            ("order a float", forged_log(exponent=8, order=10.0), log_evidence),
            ("a record without fields", forged_log(), log_evidence),
            ("an impostor", support.impostor(log_value), log_evidence),
            ("the other record", sq_value, log_evidence),
            ("evidence as a list", log_value, list(log_evidence)),
            ("factors as a list", log_value, (list(log_evidence[0]), log_evidence[1])),
            ("evidence missing", log_value, ()),
            ("the certificate missing", log_value, log_evidence[:1]),
            ("an extra evidence item", log_value, log_evidence + ((),)),
            ("oversized evidence", log_value, (log_evidence[0] + ((2, 1, ()),) * 100_000, ())),
        ]
        squares_cases = [
            ("wrong count", SumOfTwoSquares(8, 3, 4), sq_evidence),
            ("wrong representation", SumOfTwoSquares(12, 1, 5), sq_evidence),
            ("no representation", forged_sq(count=12, a=None, b=None), sq_evidence),
            ("count 0", SumOfTwoSquares(0, None, None), sq_evidence),
            ("a > b", forged_sq(count=12, a=4, b=3), sq_evidence),
            ("negative a", forged_sq(count=12, a=-3, b=4), sq_evidence),
            ("a a bool", forged_sq(count=12, a=False, b=5), sq_evidence),
            ("b a float", forged_sq(count=12, a=3, b=4.0), sq_evidence),
            ("count a str", forged_sq(count="12", a=3, b=4), sq_evidence),
            ("a record without fields", forged_sq(), sq_evidence),
            ("an impostor", support.impostor(sq_value), sq_evidence),
            ("the other record", log_value, sq_evidence),
            ("evidence as a list", sq_value, list(sq_evidence)),
            ("factors as a list", sq_value, (list(sq_evidence[0]),)),
            ("evidence missing", sq_value, ()),
            ("an extra evidence item", sq_value, sq_evidence + ((),)),
            ("a wrong factorization", sq_value, (((5, 1, ()),),)),
            ("oversized evidence", sq_value, (sq_evidence[0] + ((2, 1, ()),) * 100_000,)),
        ]
        for operation, args, cases in (("discrete_log", log_args, log_cases), ("two_squares", squares_args,
                                                                                squares_cases)):
            for label, item, proof in cases:
                with self.subTest(operation=operation, case=label):
                    self.assertRejects(operation, args, item, proof, may_exhaust=label == "oversized evidence")
        self.assertRejects("two_squares", log_args, log_value, log_evidence)
        self.assertRejects("discrete_log", squares_args, sq_value, sq_evidence)
        self.assertRejects("discrete_logarithm", log_args, log_value, log_evidence)
        self.assertRejects("discrete_log", dict(log_args, base=True), log_value, log_evidence)
        for result, wrong in ((log_result, DiscreteLogarithm(7, 10)), (squares_result, SumOfTwoSquares(8, 3, 4))):
            report = support.statuses(mf.verify(support.tampered(result, wrong)))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = list(SMALL.items()) + [("discrete_log", dict(base=2, target=3, p=7)),
                                          ("discrete_log", dict(base=3, target=pow(3, 10 ** 20, M89), p=M89)),
                                          ("two_squares", dict(n=M89)), ("two_squares", dict(n=0)),
                                          ("two_squares", dict(n=5 ** 30 * 3 ** 4 * 2 ** 9))]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, args), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, **args):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("two_squares", n=25)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a logarithm and
        none, a count with a representation (one of them for n = 0) and none."""
        squares25 = support.run("two_squares", n=25).value
        a, b = squares25.a, squares25.b
        rendered = (
            ("discrete_log", dict(base=2, target=3, p=11), "k = 8 (mod 10)", r"k \equiv 8 \pmod{10}"),
            ("discrete_log", dict(base=2, target=3, p=7), "no solution", r"\text{no solution}"),
            ("two_squares", dict(n=25), f"12 representations, e.g. 25 = {a}^2 + {b}^2",
             rf"12\ \text{{representations, e.g.}}\ 25 = {a}^{{2}} + {b}^{{2}}"),
            ("two_squares", dict(n=0), "1 representation, e.g. 0 = 0^2 + 0^2",
             r"1\ \text{representation, e.g.}\ 0 = 0^{2} + 0^{2}"),
            ("two_squares", dict(n=21), "no representation", r"\text{no representation}"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(operation=operation, **args):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertIs(type(decoded.value), type(result.value))
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run("discrete_log", base=2, target=3, p=11).value)["data"]["type"],
                         "v6_discrete_logarithm")
        self.assertEqual(mf.to_data(squares25)["data"]["type"], "v6_sum_of_two_squares")
        self.roundtrip(support.run("discrete_log", base=3, target=pow(3, 10 ** 20, M89), p=M89))

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261313)
        primes = primes_below(3000)
        cases = [("discrete_log", dict(base=base, target=rng.randrange(-10 ** 6, 10 ** 6), p=p))
                 for p in rng.sample(primes, 14) for base in [rng.randrange(1, p)]]
        cases += [("discrete_log", dict(base=3, target=pow(3, 10 ** 20, M89), p=M89)),
                  ("discrete_log", dict(base=2, target=3, p=7))]
        cases += [("two_squares", dict(n=rng.randrange(0, 10 ** 9))) for _ in range(14)]
        cases += [("two_squares", dict(n=n)) for n in (0, 1, 21, 5 ** 30 * 2 ** 9)]
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
        for operation, args in (("discrete_log", dict(base=3, target=pow(3, 10 ** 20, M89), p=M89)),
                                ("two_squares", dict(n=5 ** 40 * 13))):
            with self.subTest(operation=operation, large=True):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("discrete_log", {"base": 2, "target": 3, "p": 9}), "invalid_input")
        self.assertIn("a prime p", error["message"])
        self.assertToolError(support.call("two_squares", {"n": -1}), "invalid_input")

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on at least 300 seeded inputs per operation (every
        base and target modulo every prime below 30, too), and the checker accepts every answer with its evidence."""
        rng = random.Random(20261314)
        counts = dict.fromkeys(SMALL, 0)
        # discrete_log: brute force over every power of the base.
        for p in primes_below(30):
            for base in range(1, p):
                for target in range(p):
                    with self.subTest(base=base, target=target, p=p):
                        self.assertLog(dict(base=base, target=target, p=p), brute_log(base, target, p))
                        counts["discrete_log"] += 1
        primes = primes_below(5000)
        for _ in range(300):
            p = rng.choice(primes)
            base = rng.choice((rng.randrange(1, p), rng.randrange(1, p) + p * rng.randrange(-10 ** 9, 10 ** 9)))
            if base % p == 0:
                continue
            target = rng.choice((rng.randrange(p), rng.randrange(-10 ** 30, 10 ** 30), p * rng.randrange(-3, 4),
                                 pow(base, rng.randrange(p), p)))
            with self.subTest(base=base, target=target, p=p):
                self.assertLog(dict(base=base, target=target, p=p), brute_log(base, target, p))
                counts["discrete_log"] += 1
        # Large primes with known p - 1: the order by the builtin pow, targets built as powers or as non-powers.
        for p, group in GROUPS.items():
            self.assertEqual(prod(q ** e for q, e in group), p - 1)
            self.assertTrue(all(factor(q)[0] == (q, 1) for q, _ in group))  # Trial division proves each prime.
            for _ in range(8):
                base = rng.randrange(2, p - 1)
                order = order_from(base, p, group)
                k = rng.randrange(p)
                with self.subTest(p=p, base=base, k=k):
                    self.assertLog(dict(base=base, target=pow(base, k, p) + p * rng.randrange(-5, 5), p=p),
                                   DiscreteLogarithm(k % order, order))
                if order < p - 1:  # A generator g is no power of base: the powers of base have the order d.
                    g = next(g for g in range(2, 50) if order_from(g, p, group) == p - 1)
                    with self.subTest(p=p, base=base, target=g):
                        self.assertLog(dict(base=base, target=g, p=p), DiscreteLogarithm(None, order))
        # two_squares: counting lattice points, and the representation tested to be one.
        for n in range(0, 200):
            with self.subTest(n=n):
                self.assertSquares(n, brute_r2(n))
                counts["two_squares"] += 1
        for _ in range(150):
            n = rng.choice((rng.randrange(200, 10 ** 5), rng.randrange(10 ** 5, 10 ** 8)))
            with self.subTest(n=n):
                self.assertSquares(n, brute_r2(n) if n < 10 ** 6 else jacobi_count(n))
                counts["two_squares"] += 1
        # Large n from known primes: r2 by Jacobi's theorem over the factorization written here. At most one large
        # prime, at most squared: the integer kernel's rho cannot split two large primes, nor a cube of one (both
        # resource_limit_exceeded under the default budget).
        for _ in range(40):
            pairs = {q: rng.randrange(1, 12)
                     for q in rng.sample((2, 3, 5, 7, 11, 13, 17, 29, 37, 41, 1321, 2113, 65537, 99991),
                                         rng.randrange(1, 5))}
            if rng.randrange(3):
                pairs[rng.choice((M61, M89, M127, HALF, 2931542417))] = rng.randrange(1, 3)
            n = prod(q ** e for q, e in pairs.items())
            expected = 0 if any(q % 4 == 3 and e % 2 for q, e in pairs.items()) else \
                4 * prod(e + 1 for q, e in pairs.items() if q % 4 == 1)
            with self.subTest(n=n):
                self.assertSquares(n, expected)
                counts["two_squares"] += 1
        self.assertTrue(all(count >= 300 for count in counts.values()), counts)

    def test_full_size_inputs_fit_the_default_budget(self):
        """Requests on integers of nearly max_integer_bits complete under the default limits: 2**65535, whose only
        representations are (+-2**32767, +-2**32767), a power of 5 of 60,000 bits and a target of 65,536 bits."""
        result = support.run("two_squares", n=2 ** 65535)
        self.assertVerified(result, SumOfTwoSquares(4, 2 ** 32767, 2 ** 32767))
        n = 5 ** 25000 * 9
        self.assertLessEqual(n.bit_length(), 65_536)
        result = support.run("two_squares", n=n)
        self.assertVerified(result)
        self.assertEqual(result.value.count, 4 * 25001)
        self.assertEqual(result.value.a ** 2 + result.value.b ** 2, n)
        rng = random.Random(20261315)
        target = rng.getrandbits(65_535)
        self.assertVerified(support.run("discrete_log", base=3, target=target, p=F4), brute_log(3, target, F4))

    def test_long_prime_powers_of_the_order(self):
        """Regression: orders 2**1000 and 13 * 2**1000 modulo PROTH. The digit-by-digit loop of Pohlig and Hellman
        costs about f*f/2 squarings for q**f, past the default budget at f = 1000; halving f costs O(f log f)."""
        generator = next(g for g in range(2, 50) if order_from(g, PROTH, GROUPS[PROTH]) == PROTH - 1)
        self.assertEqual(order_from(3, PROTH, GROUPS[PROTH]), 2 ** 1000)
        for base in (3, generator):
            order = order_from(base, PROTH, GROUPS[PROTH])
            for k in (0, 1, 2 ** 999, order - 1, 10 ** 300):
                with self.subTest(base=base, k=k):
                    result = support.run("discrete_log", base=base, target=pow(base, k, PROTH), p=PROTH)
                    self.assertVerified(result, DiscreteLogarithm(k % order, order))
        # 9 = 3**2 has the order 2**999, and 3 is no power of it: 3**(2**999) = -1.
        self.assertEqual(pow(3, 2 ** 999, PROTH), PROTH - 1)
        self.assertVerified(support.run("discrete_log", base=9, target=3, p=PROTH),
                            DiscreteLogarithm(None, 2 ** 999))

    def test_p_minus_one_is_factored_once(self):
        """Regression (review of wave m7b): from psi_13 on, p - 1 is read from p's Pocklington certificate instead of
        being factored a second time. Here p - 1 = 2 * q1 * q2 * q3 with three primes of 34 bits, which costs the
        integer kernel about 1.15 million units of the default 2,000,000; factoring it twice made both requests
        resource_limit_exceeded, although each verified value costs at most 1.6 million units in all."""
        q1, q2, q3 = 10401486133, 16830386357, 12617408761
        p = 2 * q1 * q2 * q3 + 1
        self.assertTrue(p > PSI_13 and all(is_prime(q) for q in (q1, q2, q3)))  # is_prime is exact below psi_13.
        # Pocklington, witness 2: 2**(p - 1) = 1 and gcd(2**((p - 1)/q) - 1, p) = 1 for every q, so p is prime.
        self.assertEqual(pow(2, p - 1, p), 1)
        self.assertTrue(all(gcd(pow(2, (p - 1) // q, p) - 1, p) == 1 for q in (2, q1, q2, q3)))
        # 4 = 2**2 has the order (p - 1)/2, and 2 is no power of it (2 is a generator).
        self.assertVerified(support.run("discrete_log", base=4, target=2, p=p), DiscreteLogarithm(None, (p - 1) // 2))
        # 2**(2 q1 q2) has the prime order q3: one baby-step giant-step search of about 2**17 steps.
        base = pow(2, 2 * q1 * q2, p)
        result = support.run("discrete_log", base=base, target=pow(base, 10 ** 20, p), p=p)
        self.assertVerified(result, DiscreteLogarithm(10 ** 20 % q3, q3))
        factors, certificate = result.steps[0].outputs[1]
        self.assertEqual((factors[0][:2], certificate[:2]), ((q3, 1), ("pocklington", p)))

    def test_the_check_has_a_budget_of_its_own(self):
        """The verifier checks in a fresh budget of the request's limits, so the certified factorization of the order
        is refused only when check_factorization alone needs more than max_work, whatever the production spent."""
        args = dict(base=3, target=pow(3, 10 ** 20, M89), p=M89)
        with computation(ComputationLimits(max_work=10 ** 9), fresh=True) as budget:
            value, evidence = family.produce("discrete_log", dict(args))
        production = budget.work
        limits = ComputationLimits(max_work=2_000_000)
        with computation(limits, fresh=True) as budget:
            budget.tick(limits.max_work - production)
            self.assertEqual(family.produce("discrete_log", dict(args)), (value, evidence))
            self.assertEqual(budget.work, limits.max_work)
        self.assertIs(support.verdict(family, "discrete_log", args, value, evidence), True)

    def test_producer_computes_no_builtin_pow(self):
        """The producer's powers come from its own square-and-multiply loop, which charges the budget: nothing in
        the family module calls the builtin pow, whose work could not be interrupted."""
        tree = ast.parse(inspect.getsource(family))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        self.assertIn("_power", called)
