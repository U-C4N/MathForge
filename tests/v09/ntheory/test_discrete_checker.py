"""The independent checker of mathforge.areas.ntheory.discrete (mathforge.certify.ntheory.discrete), tested on its own
before the family's producer exists.

Every true claim is computed here: discrete logarithms and orders by walking the powers of the base, counts of sums of
two squares by enumerating the lattice points of a disc, representations by search, and claims on large primes by
construction (powers of a known exponent, squares of known integers), never by the producer. The evidence follows the
layouts of the family's docstring: (factors of the order, certificate of p) for discrete_log and (factors of n,) for
two_squares; mathforge.kernels.certificates builds the prime certificates, which the checker proves itself. The
checker answers False to every corrupted claim, never with an exception, and refuses an oversized forgery within FAST
seconds under the default budget.
"""

from __future__ import annotations

import ast
import inspect
from math import isqrt
import random
import time
from types import MappingProxyType

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.ntheory import discrete as family
from mathforge.areas.ntheory.discrete import DiscreteLogarithm, SumOfTwoSquares
from mathforge.certify.ntheory import discrete as checker
from mathforge.contracts import CheckStatus
from mathforge.errors import InvalidInput
from mathforge.kernels import certificates
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational
from mathforge.registry import Argument, OperationSpec
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = ("exact integers; certified primes, orders, residues, discrete logarithms and sums of two squares; "
         "canonical integer solution families")
M89 = 2 ** 89 - 1  # A Mersenne prime above psi_13, = 3 (mod 4).
HALF = PSI_13 + 5062  # A prime above psi_13, so that its certificate is a Pocklington certificate, ...
SAFE = 2 * HALF + 1  # ... and SAFE is prime too, = 3 (mod 4); 2, 3 and 4 are squares modulo SAFE, 5 is not.
F4 = 65537  # The Fermat prime 2**16 + 1, where 3 generates (Z/F4)^*.
COMPOSITE = 2 ** 100 + 1  # Above psi_13, and divisible by 2**20 + 1.

certificate = certificates.prime_evidence  # () below psi_13, a Pocklington certificate from there on.


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these, from brute force or from a construction.

def primes_below(limit: int) -> list[int]:
    """The primes below limit, by Eratosthenes' sieve."""
    flags = bytearray([1]) * limit
    flags[:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return [n for n in range(limit) if flags[n]]


def factor(n: int) -> tuple[tuple[int, int], ...]:
    """The prime factorization ((p, e), ...) of 1 <= n < 10**12 by trial division."""
    pairs, d = [], 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            pairs.append((d, e))
        d += 1 if d == 2 else 2
    if n > 1:
        pairs.append((n, 1))
    return tuple(pairs)


def certified(pairs: tuple[tuple[int, int], ...]) -> tuple:
    """The certified factorization ((p, e, certificate), ...) of known prime pairs ((p, e), ...)."""
    return tuple((p, e, certificate(p)) for p, e in pairs)


def factors_of(n: int) -> tuple:
    """The certified factorization of 1 <= n < 10**12, found by trial division."""
    return certified(factor(n))


def power(base: int, exponent: int, modulus: int) -> int:
    """base**exponent modulo modulus >= 1 for exponent >= 0, by square-and-multiply written out here."""
    result, base = 1 % modulus, base % modulus
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


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


def log_claim(base: int, target: int, p: int) -> tuple[dict[str, int], DiscreteLogarithm, tuple]:
    """(args, value, evidence) of the true claim on small numbers, by brute force."""
    value = brute_log(base, target, p)
    return dict(base=base, target=target, p=p), value, (factors_of(value.order), certificate(p))


def brute_r2(n: int) -> int:
    """r2(n): the pairs (x, y) of integers with x*x + y*y = n, counted over the square |x|, |y| <= isqrt(n)."""
    r = isqrt(n)
    return sum(1 for x in range(-r, r + 1) for y in range(-r, r + 1) if x * x + y * y == n)


def representations(n: int) -> list[tuple[int, int]]:
    """Every (a, b) with 0 <= a <= b and a*a + b*b = n, by search."""
    return [(a, isqrt(n - a * a)) for a in range(isqrt(n // 2) + 1) if isqrt(n - a * a) ** 2 == n - a * a]


def squares_claim(n: int) -> tuple[dict[str, int], SumOfTwoSquares, tuple]:
    """(args, value, evidence) of the true claim on a small n, by brute force; the first representation found."""
    count, found = brute_r2(n), representations(n)
    value = SumOfTwoSquares(count, *found[0]) if found else SumOfTwoSquares(0, None, None)
    return dict(n=n), value, (factors_of(n) if n else (),)


def pocklington_for(n: int) -> tuple:
    """A well-formed Pocklington certificate for n = 2**k + 1, with the witness 3: it proves nothing for a composite."""
    k = (n - 1).bit_length() - 1
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


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass for an evidence tuple.


def inexact(number: int) -> tuple[object, ...]:
    """Stand-ins for an exact int that are no exact int: a bool, a float, a str, None, a Rational and an int
    subclass of it."""
    return bool(number), float(number), str(number), None, Rational(number), Integer(number)


def logarithm(exponent: object, order: object) -> DiscreteLogarithm:
    """A DiscreteLogarithm made without its canonical-form validation."""
    return support.forged(DiscreteLogarithm, exponent=exponent, order=order)


def squares(count: object, a: object, b: object) -> SumOfTwoSquares:
    """A SumOfTwoSquares made without its canonical-form validation."""
    return support.forged(SumOfTwoSquares, count=count, a=a, b=b)


class DiscreteCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: list[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        def integers(*names):
            return tuple(Argument(name, "int", None, "int") for name in names)

        expected = (
            OperationSpec("discrete_log", "ntheory", integers("base", "target", "p"), "DiscreteLogarithm",
                          "Solve base**k = target modulo a prime p for the least k >= 0.", SCOPE, version=6),
            OperationSpec("two_squares", "ntheory", integers("n"), "SumOfTwoSquares",
                          "Count the representations of n as a sum of two squares and give one.", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {"discrete_logarithm": DiscreteLogarithm,
                                        "sum_of_two_squares": SumOfTwoSquares})
        self.assertEqual(family.CHECKER, "mathforge.certify.ntheory.discrete")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)

    def test_records_validate_their_canonical_form(self):
        for exponent, order in ((0, 1), (8, 10), (None, 3), (None, 1)):
            DiscreteLogarithm(exponent, order)
        for exponent, order in ((0, 0), (None, 0), (None, -1), (3, 3), (-1, 5), (True, 5), (1, 5.0), (1.0, 5),
                                ("1", 5), (1, True), (1, None), (2 ** 70_000, 2 ** 70_001)):
            with self.subTest(exponent=exponent, order=order), self.assertRaises((InvalidInput, ResourceLimitError)):
                DiscreteLogarithm(exponent, order)
        for count, a, b in ((1, 0, 0), (4, 0, 1), (12, 3, 4), (0, None, None)):
            SumOfTwoSquares(count, a, b)
        for count, a, b in ((-4, 0, 1), (0, 0, None), (0, None, 1), (4, None, None), (4, 1, 0), (4, -1, 0),
                            (4, 0, None), (4.0, 0, 1), (4, False, True), (True, 0, 1), (4, 0, "1"),
                            (4, 0, 2 ** 70_000)):
            with self.subTest(count=count, a=a, b=b), self.assertRaises((InvalidInput, ResourceLimitError)):
                SumOfTwoSquares(count, a, b)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_discrete_log_known_values(self):
        for base, target, p, exponent, order in ((2, 3, 11, 8, 10), (3, 13, 17, 4, 16), (2, 3, 7, None, 3),
                                                 (5, 1, 7, 0, 6), (2, 0, 11, None, 10)):
            with self.subTest(base=base, target=target, p=p):
                value = DiscreteLogarithm(exponent, order)
                self.assertEqual(brute_log(base, target, p), value)
                self.assertAccepts("discrete_log", dict(base=base, target=target, p=p), value,
                                   (factors_of(order), ()))

    def test_discrete_log_accepts_every_true_claim_on_small_primes(self):
        """Every base and target modulo every prime below 30, and the targets beyond [0, p) and below 0."""
        for p in primes_below(30):
            for base in range(1, p):
                for target in range(p):
                    args, value, evidence = log_claim(base, target, p)
                    with self.subTest(**args):
                        self.assertAccepts("discrete_log", args, value, evidence)
        for base, target, p in ((-9, -8, 11), (13, 25, 11), (-1, -1, 11), (10 ** 30 + 2, 3 - 10 ** 30, 11),
                                (12, 0, 11), (1, 5, 11), (1, 12, 11), (10, 1, 11)):
            args, value, evidence = log_claim(base, target, p)
            with self.subTest(**args):
                self.assertAccepts("discrete_log", args, value, evidence)

    def test_discrete_log_special_cases(self):
        # p = 2: every odd base has the order 1; an odd target is base**0, an even one no power.
        for base, target, exponent in ((1, 1, 0), (1, 0, None), (3, 5, 0), (-1, -4, None), (7, 2, None)):
            with self.subTest(p=2, base=base, target=target):
                self.assertAccepts("discrete_log", dict(base=base, target=target, p=2),
                                   DiscreteLogarithm(exponent, 1), ((), ()))
        # base = 1 (mod p): the order 1, its factorization empty, and only target = 1 is reached.
        self.assertAccepts("discrete_log", dict(base=12, target=23, p=11), DiscreteLogarithm(0, 1), ((), ()))
        self.assertAccepts("discrete_log", dict(base=12, target=2, p=11), DiscreteLogarithm(None, 1), ((), ()))
        # The Fermat prime 65537: 3 generates, the order 2**16 has one prime, and k is recovered from 3**k.
        evidence = (((2, 16, ()),), ())
        for k in (0, 1, 12345, 65535):
            with self.subTest(p=F4, k=k):
                self.assertAccepts("discrete_log", dict(base=3, target=power(3, k, F4), p=F4),
                                   DiscreteLogarithm(k, 65536), evidence)
        self.assertEqual(brute_log(3, power(3, 40000, F4), F4), DiscreteLogarithm(40000, 65536))
        # Above psi_13: the certificate of p is a Pocklington certificate. 4 is a square modulo SAFE, so its order is
        # the prime HALF; 5 is no square, so no power of 4. 4**k for a random k of 100 bits gives k mod HALF.
        evidence = (((HALF, 1, certificate(HALF)),), certificate(SAFE))
        k = random.Random(20261113).getrandbits(100)
        self.assertAccepts("discrete_log", dict(base=4, target=power(4, k, SAFE), p=SAFE),
                           DiscreteLogarithm(k % HALF, HALF), evidence)
        self.assertAccepts("discrete_log", dict(base=4, target=5, p=SAFE), DiscreteLogarithm(None, HALF), evidence)
        self.assertAccepts("discrete_log", dict(base=4, target=0, p=SAFE), DiscreteLogarithm(None, HALF), evidence)
        self.assertAccepts("discrete_log", dict(base=-1, target=SAFE - 1, p=SAFE), DiscreteLogarithm(1, 2),
                           (((2, 1, ()),), certificate(SAFE)))

    def test_two_squares_known_values(self):
        for n, count, a, b in ((0, 1, 0, 0), (1, 4, 0, 1), (2, 4, 1, 1), (3, 0, None, None), (25, 12, 0, 5),
                               (25, 12, 3, 4), (65, 16, 1, 8), (65, 16, 4, 7), (21, 0, None, None)):
            with self.subTest(n=n, a=a, b=b):
                self.assertEqual(brute_r2(n), count)
                self.assertAccepts("two_squares", dict(n=n), SumOfTwoSquares(count, a, b),
                                   (factors_of(n) if n else (),))
        self.assertAccepts("two_squares", dict(n=M89), SumOfTwoSquares(0, None, None),
                           (((M89, 1, certificate(M89)),),))

    def test_two_squares_accepts_every_true_claim_up_to_300(self):
        """Every n <= 300, with every representation 0 <= a <= b of it: the value names an example, any one."""
        for n in range(301):
            args, value, evidence = squares_claim(n)
            reps = representations(n) if value.count else [(None, None)]
            for a, b in reps:
                with self.subTest(n=n, a=a, b=b):
                    self.assertAccepts("two_squares", args, SumOfTwoSquares(value.count, a, b), evidence)

    def test_two_squares_special_cases(self):
        # Powers of 2, of a prime 1 (mod 4) and of a prime 3 (mod 4) together.
        n = 2 ** 4 * 3 ** 2 * 5 ** 3 * 13
        args, value, evidence = squares_claim(n)
        self.assertEqual(value.count, 4 * 4 * 2)
        self.assertAccepts("two_squares", args, value, evidence)
        # An even power of a large prime q = 3 (mod 4) above psi_13: SAFE**2 = 0**2 + SAFE**2 alone, r2 = 4.
        self.assertAccepts("two_squares", dict(n=SAFE ** 2), SumOfTwoSquares(4, 0, SAFE),
                           (((SAFE, 2, certificate(SAFE)),),))
        self.assertAccepts("two_squares", dict(n=SAFE ** 3), SumOfTwoSquares(0, None, None),
                           (((SAFE, 3, certificate(SAFE)),),))
        # Products of large primes 1 (mod 4) by construction: (x*x + y*y)(u*u + v*v) = (xu - yv)**2 + (xv + yu)**2.
        p, q = 10009, 1000033  # Both 1 (mod 4) and prime: 10009 = 3**2 + 100**2, 1000033 = 408**2 + 913**2.
        self.assertEqual(factor(p * q), ((p, 1), (q, 1)))
        self.assertEqual((representations(p), representations(q)), ([(3, 100)], [(408, 913)]))
        a, b = sorted((abs(3 * 408 - 100 * 913), 3 * 913 + 100 * 408))
        self.assertAccepts("two_squares", dict(n=p * q), SumOfTwoSquares(16, a, b), (factors_of(p * q),))
        # Nearly max_integer_bits: 2**65535 = (2**32767)**2 + (2**32767)**2, and r2 = 4.
        k = 65_535
        self.assertAccepts("two_squares", dict(n=2 ** k), SumOfTwoSquares(4, 2 ** 32767, 2 ** 32767),
                           (((2, k, ()),),))

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims, which the checker rejects

    def test_discrete_log_rejects_every_wrong_value_on_small_primes(self):
        """Modulo every prime below 14, every other exponent in [0, d), None and every other divisor d of p - 1 as
        the order are refused, with the evidence of the claimed order."""
        for p in primes_below(14):
            divisors = [d for d in range(1, p) if (p - 1) % d == 0]
            for base in range(1, p):
                for target in range(p):
                    args, true, _ = log_claim(base, target, p)
                    for order in divisors:
                        for exponent in (None, *range(order)):
                            value = DiscreteLogarithm(exponent, order)
                            if value == true:
                                continue
                            with self.subTest(**args, exponent=exponent, order=order):
                                self.assertRejects("discrete_log", args, value, (factors_of(order), ()))

    def test_discrete_log_rejects_corrupted_claims(self):
        args, value, evidence = log_claim(2, 3, 11)
        factors = evidence[0]
        self.assertAccepts("discrete_log", args, value, evidence)
        big_args = dict(base=4, target=5, p=SAFE)
        big_value, big_evidence = DiscreteLogarithm(None, HALF), (((HALF, 1, certificate(HALF)),), certificate(SAFE))
        self.assertAccepts("discrete_log", big_args, big_value, big_evidence)
        order3 = (((3, 1, ()),), ())  # Evidence for the order 3, of 2 modulo 7.
        cases = [
            ("a wrong exponent", args, DiscreteLogarithm(7, 10), evidence),
            ("None for a solvable target", args, DiscreteLogarithm(None, 10), evidence),
            ("a solution for an unsolvable target", dict(base=2, target=3, p=7), DiscreteLogarithm(1, 3), order3),
            ("a solution for target 0", dict(base=2, target=0, p=11), DiscreteLogarithm(0, 10), evidence),
            ("a non-least solution", args, logarithm(18, 10), evidence),
            ("a negative congruent solution", args, logarithm(-2, 10), evidence),
            ("a multiple of the order", args, DiscreteLogarithm(8, 20), (factors_of(20), ())),
            ("a proper divisor as the order", args, DiscreteLogarithm(3, 5), (factors_of(5), ())),
            ("order 0", args, logarithm(None, 0), ((), ())),
            ("a negative order", args, logarithm(None, -10), evidence),
            ("an exponent at the order", args, logarithm(10, 10), evidence),
            ("an impostor record", args, support.impostor(value), evidence),
            ("a forged record without exponent", args, support.forged(DiscreteLogarithm, order=10), evidence),
            ("a forged record without order", args, support.forged(DiscreteLogarithm, exponent=8), evidence),
            ("a record of the other operation", args, SumOfTwoSquares(8, 10, 10), evidence),
            ("a bare int", args, 8, evidence),
            ("a tuple", args, (8, 10), evidence),
            ("None", args, None, evidence),
            ("a bool exponent", dict(base=2, target=2, p=11), logarithm(True, 10), evidence),
            ("a bool order", dict(base=1, target=1, p=11), logarithm(0, True), ((), ())),
            # The evidence.
            ("evidence as a list", args, value, list(evidence)),
            ("evidence as a tuple subclass", args, value, Evidence(evidence)),
            ("evidence None", args, value, None),
            ("evidence ()", args, value, ()),
            ("the certificate missing", args, value, (factors,)),
            ("an extra evidence item", args, value, evidence + ((),)),
            ("evidence items swapped", args, value, ((), factors)),
            ("the factors as a list", args, value, (list(factors), ())),
            ("a factor entry as a list", args, value, (([2, 1, ()], (5, 1, ())), ())),
            ("the factors flattened", args, value, ((2, 1, (), 5, 1, ()), ())),
            ("factors of p - 1 unchanged for the order 5", dict(base=3, target=4, p=11), DiscreteLogarithm(4, 5),
             evidence),
            ("a factor missing", args, value, (((2, 1, ()),), ())),
            ("a composite listed as prime", args, value, (((10, 1, ()),), ())),
            ("factors decreasing", args, value, (((5, 1, ()), (2, 1, ())), ())),
            ("a needless certificate for 11", args, value, (factors, ("trial", 11))),
            ("a certificate as a list", big_args, big_value, (big_evidence[0], list(big_evidence[1]))),
            ("an empty certificate above psi_13", big_args, big_value, (big_evidence[0], ())),
            ("the certificate of another prime", big_args, big_value, (big_evidence[0], certificate(M89))),
            ("a factorization with an int subclass", args, value, (((Integer(2), 1, ()), (5, 1, ())), ())),
            # Internally consistent but wrong: the order 10 of 3 modulo 11 is 5, so 3**5 = 1 refutes 10.
            ("a non-minimal order with its factorization", dict(base=3, target=4, p=11), DiscreteLogarithm(4, 10),
             evidence),
            # Composite moduli with consistent claims: 2 has the order 10 modulo 341 = 11 * 31, and modulo 561.
            ("the pseudoprime 341", dict(base=2, target=1, p=341), DiscreteLogarithm(0, 10), evidence),
            ("the Carmichael number 561", dict(base=2, target=2, p=561), DiscreteLogarithm(1, 40),
             (factors_of(40), ())),
            ("a composite p with a Pocklington-shaped certificate", dict(base=3, target=1, p=COMPOSITE),
             DiscreteLogarithm(0, 1), ((), pocklington_for(COMPOSITE))),
            # Inputs outside the domain have no value.
            ("base 0 modulo p", dict(base=22, target=1, p=11), DiscreteLogarithm(0, 1), ((), ())),
            ("base 0", dict(base=0, target=0, p=11), DiscreteLogarithm(None, 1), ((), ())),
            ("p = 1", dict(base=1, target=0, p=1), DiscreteLogarithm(0, 1), ((), ())),
            ("p = 0", dict(base=1, target=0, p=0), DiscreteLogarithm(0, 1), ((), ())),
            ("p = -11", dict(base=2, target=3, p=-11), value, evidence),
            ("p = 1, base 0", dict(base=0, target=0, p=1), DiscreteLogarithm(0, 1), ((), ())),
        ]
        for name, number in args.items():
            for item in inexact(number):
                cases.append((f"argument {name}={item!r}", dict(args, **{name: item}), value, evidence))
        for item in inexact(8):
            cases.append((f"exponent {item!r}", args, logarithm(item, 10), evidence))
        for item in inexact(10):
            cases.append((f"order {item!r}", args, logarithm(8, item), evidence))
        self.assertRejectsAll("discrete_log", cases)

    def test_two_squares_rejects_corrupted_claims(self):
        args, value, evidence = dict(n=25), SumOfTwoSquares(12, 3, 4), ((((5, 2, ())),),)
        factors = evidence[0]
        self.assertAccepts("two_squares", args, value, evidence)
        none21 = SumOfTwoSquares(0, None, None)
        cases = [
            ("a wrong count", args, SumOfTwoSquares(8, 3, 4), evidence),
            ("a count off by 4", args, SumOfTwoSquares(16, 3, 4), evidence),
            ("the count of the positive quadrant", args, SumOfTwoSquares(3, 3, 4), evidence),
            ("a wrong representation", args, SumOfTwoSquares(12, 1, 4), evidence),
            ("a representation of another n", args, SumOfTwoSquares(12, 2, 4), evidence),
            ("no representation for 25", args, none21, evidence),
            ("a representation of 21", dict(n=21), SumOfTwoSquares(4, 1, 4), (factors_of(21),)),
            ("count 0 with a representation", dict(n=21), squares(0, 1, 2), (factors_of(21),)),
            ("count 0 with one None", dict(n=21), squares(0, None, 2), (factors_of(21),)),
            ("a count without representation", args, squares(12, None, None), evidence),
            ("a > b", args, squares(12, 4, 3), evidence),
            ("a negative a", args, squares(12, -3, 4), evidence),
            ("both negative", args, squares(12, -4, -3), evidence),
            ("b = 0 after a", args, squares(12, 5, 0), evidence),
            ("r2(0) = 4", dict(n=0), SumOfTwoSquares(4, 0, 0), ((),)),
            ("r2(0) = 0", dict(n=0), none21, ((),)),
            ("a representation (0, 1) for 0", dict(n=0), SumOfTwoSquares(1, 0, 1), ((),)),
            ("r2(1) = 1", dict(n=1), SumOfTwoSquares(1, 0, 1), ((),)),
            ("r2(1) = 8", dict(n=1), SumOfTwoSquares(8, 0, 1), ((),)),
            ("a negative count", args, squares(-12, 3, 4), evidence),
            ("an impostor record", args, support.impostor(value), evidence),
            ("a forged record without count", args, support.forged(SumOfTwoSquares, a=3, b=4), evidence),
            ("a forged record without b", args, support.forged(SumOfTwoSquares, count=12, a=3), evidence),
            ("a record of the other operation", args, DiscreteLogarithm(3, 12), evidence),
            ("a bare int", args, 12, evidence),
            ("a tuple", args, (12, 3, 4), evidence),
            ("a list", args, [12, 3, 4], evidence),
            ("bool representation", dict(n=2), squares(4, True, True), (((2, 1, ()),),)),
            ("bool count", dict(n=0), squares(True, 0, 0), ((),)),
            # The evidence.
            ("evidence as a list", args, value, list(evidence)),
            ("evidence as a tuple subclass", args, value, Evidence(evidence)),
            ("evidence None", args, value, None),
            ("evidence ()", args, value, ()),
            ("an extra evidence item", args, value, evidence + ((),)),
            ("the factors without their wrapper", args, value, factors),
            ("the factors as a list", args, value, (list(factors),)),
            ("a factor entry as a list", args, value, (([5, 2, ()],),)),
            ("the factors of 0 not empty", dict(n=0), SumOfTwoSquares(1, 0, 0), (((2, 1, ()),),)),
            ("the factors of 0 as a list", dict(n=0), SumOfTwoSquares(1, 0, 0), ([],)),
            ("the factors of 0 None", dict(n=0), SumOfTwoSquares(1, 0, 0), (None,)),
            ("the factors of 1 not empty", dict(n=1), SumOfTwoSquares(4, 0, 1), (((1, 1, ()),),)),
            ("an exponent too small", args, value, (((5, 1, ()),),)),
            ("the factors of another n", args, value, (factors_of(65),)),
            # Internally consistent but wrong: 9 listed as a prime 1 (mod 4) gives the true r2(9) = 4, and 45 with
            # 15 listed as prime; 21 as a prime 1 (mod 4) would give a count of 8.
            ("9 as a prime", dict(n=9), SumOfTwoSquares(4, 0, 3), (((9, 1, ()),),)),
            ("15 as a prime", dict(n=45), SumOfTwoSquares(8, 3, 6), (((3, 1, ()), (15, 1, ())),)),
            ("21 as a prime", dict(n=21), SumOfTwoSquares(8, 1, 4), (((21, 1, ()),),)),
            ("an empty certificate above psi_13", dict(n=M89), none21, (((M89, 1, ()),),)),
            ("a composite with a Pocklington-shaped certificate", dict(n=COMPOSITE), SumOfTwoSquares(4, 1, 2 ** 50),
             (((COMPOSITE, 1, pocklington_for(COMPOSITE)),),)),
            ("a factorization with an int subclass", args, value, (((Integer(5), 2, ()),),)),
            # Outside the domain.
            ("n = -1", dict(n=-1), none21, ((),)),
            ("n = -25", dict(n=-25), value, evidence),
            ("arguments as pairs", tuple(args.items()), value, evidence),
            ("arguments as a read-only mapping", MappingProxyType(args), value, evidence),
            ("an extra argument", dict(n=25, k=1), value, evidence),
            ("an argument renamed", dict(m=25), value, evidence),
            ("no arguments", {}, value, evidence),
        ]
        for item in inexact(25):
            cases.append((f"argument n={item!r}", dict(n=item), value, evidence))
        for item in inexact(12):
            cases.append((f"count {item!r}", args, squares(item, 3, 4), evidence))
        for item in inexact(3):
            cases.append((f"a {item!r}", args, squares(12, item, 4), evidence))
        for item in inexact(4):
            cases.append((f"b {item!r}", args, squares(12, 3, item), evidence))
        self.assertRejectsAll("two_squares", cases)

    def test_two_squares_rejects_every_wrong_count_up_to_100(self):
        for n in range(101):
            args, value, evidence = squares_claim(n)
            for count in range(0, 20):
                if count == value.count:
                    continue
                if count == 0:
                    wrong = SumOfTwoSquares(0, None, None)
                else:
                    wrong = SumOfTwoSquares(count, *(representations(n) or [(0, isqrt(n))])[0])
                with self.subTest(n=n, count=count):
                    self.assertRejects("two_squares", args, wrong, evidence)

    def test_rejects_wrong_operations(self):
        args, value, evidence = log_claim(2, 3, 11)
        for operation in ("two_squares", "Discrete_log", "discrete_log ", "discrete_logarithm", "discretelog",
                          "multiplicative_order", "", None, 7, ("discrete_log",), b"discrete_log", ["discrete_log"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)
        self.assertRejects("discrete_log", dict(n=25), SumOfTwoSquares(12, 3, 4), ((((5, 2, ())),),))
        self.assertRejects("two_squares", dict(n=25), value, evidence)

    def test_oversized_forgeries_fail_fast(self):
        giant = 2 ** 70_000  # Longer than max_integer_bits allows.
        args, value, evidence = log_claim(2, 3, 11)
        squares_args, squares_evidence = dict(n=25), ((((5, 2, ())),),)
        many = tuple((q, 1, ()) for q in primes_below(100_000))
        cases = [
            ("discrete_log", args, logarithm(8, giant), evidence),
            ("discrete_log", args, logarithm(giant, giant + 1), evidence),
            ("discrete_log", args, logarithm(-giant, 10), evidence),
            ("discrete_log", args, logarithm(8, support.HUGE), (((2, 65_536, ()),), ())),
            ("discrete_log", args, value, (((2, giant, ()), (5, 1, ())), ())),
            ("discrete_log", args, value, (((2, 10 ** 100, ()),), ())),
            ("discrete_log", args, value, (many, ())),
            ("discrete_log", args, value, ((0,) * 100_000, ())),
            ("discrete_log", args, value, (evidence[0], (0,) * 100_000)),
            ("discrete_log", args, value, (evidence[0], (giant,))),
            ("discrete_log", args, value, evidence + ((),) * 100_000),
            ("discrete_log", dict(base=3, target=5, p=2 ** 127 - 1), DiscreteLogarithm(1, 2 ** 126),
             (((2, 126, ()),), certificate(2 ** 127 - 1))),
            ("two_squares", squares_args, squares(giant, 3, 4), squares_evidence),
            ("two_squares", squares_args, squares(12, 3, giant), squares_evidence),
            ("two_squares", squares_args, squares(12, giant, giant), squares_evidence),
            ("two_squares", squares_args, squares(12, 0, 2 ** 40_000), squares_evidence),
            ("two_squares", squares_args, SumOfTwoSquares(12, 3, 4), (((5, giant, ()),),)),
            ("two_squares", squares_args, SumOfTwoSquares(12, 3, 4), (many,)),
            ("two_squares", squares_args, SumOfTwoSquares(12, 3, 4), ((0,) * 100_000,)),
            ("two_squares", squares_args, SumOfTwoSquares(12, 3, 4), squares_evidence + ((),) * 100_000),
            ("two_squares", dict(n=2 ** 65_535), SumOfTwoSquares(4, 2 ** 32767, 2 ** 32767),
             (((2, 65_535, ()),) * 50_000,)),
        ]
        for index, (operation, case_args, case_value, case_evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, case_args, case_value, case_evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # A certificate whose nested node claims a giant prime may exhaust the budget at once instead: the claim then
        # stays unverified, never verified (prime_proof charges a Pocklington certificate before reading it).
        kind, recorded, nested_factors, witnesses = certificate(M89)
        giant_prime = 2 ** 60_000 + 1
        nested = ("pocklington", giant_prime, ((2, 60_000, ("trial", 2)),), (3,))
        forged = (kind, recorded, nested_factors[:-1] + ((giant_prime, 1, nested),), witnesses)
        for operation, case_args, case_value, case_evidence in (
                ("two_squares", dict(n=M89), SumOfTwoSquares(0, None, None), (((M89, 1, forged),),)),
                ("discrete_log", dict(base=1, target=1, p=M89), DiscreteLogarithm(0, 1), ((), forged))):
            with self.subTest(operation=operation, certificate="a giant nested node"):
                start = time.perf_counter()
                self.assertRejects(operation, case_args, case_value, case_evidence, may_exhaust=True)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_arguments_over_max_integer_bits_exceed_the_budget(self):
        """An argument that the producer's budget refuses is refused by the checker's too: the claim stays
        unverified."""
        for operation, args, value, evidence in (
                ("two_squares", dict(n=support.HUGE), SumOfTwoSquares(4, 0, 2 ** 32768), (((2, 65_536, ()),),)),
                ("discrete_log", dict(base=support.HUGE + 1, target=1, p=11), DiscreteLogarithm(0, 1), ((), ())),
                ("discrete_log", dict(base=2, target=support.HUGE, p=11), DiscreteLogarithm(6, 10), (factors_of(10),
                                                                                                      ()))):
            with self.subTest(operation=operation):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence)
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and the checker's construction

    def test_work_is_charged(self):
        """A check costs work in proportion to its powers: a budget one unit short of a check refuses it."""
        args, value = dict(base=4, target=5, p=SAFE), DiscreteLogarithm(None, HALF)
        evidence = (((HALF, 1, certificate(HALF)),), certificate(SAFE))
        for operation, case_args, case_value, case_evidence in (
                ("discrete_log", args, value, evidence),
                ("two_squares", dict(n=2 ** 65_535), SumOfTwoSquares(4, 2 ** 32767, 2 ** 32767), (((2, 65_535, ()),),)),
                ("two_squares", dict(n=SAFE ** 2), SumOfTwoSquares(4, 0, SAFE), (((SAFE, 2, certificate(SAFE)),),))):
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, case_args, case_value, case_evidence), True)
                work = budget.work
                self.assertGreater(work, 1)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, case_args, case_value, case_evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, case_args, case_value, case_evidence,
                                   limits=ComputationLimits(max_work=work))

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
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the records and the
        evidence layouts survive the format, and a wrong value fails the mathematics check."""
        big = (((HALF, 1, certificate(HALF)),), certificate(SAFE))
        claims = [
            ("discrete_log", dict(base=2, target=3, p=11), DiscreteLogarithm(8, 10), (factors_of(10), ()),
             DiscreteLogarithm(7, 10)),
            ("discrete_log", dict(base=4, target=5, p=SAFE), DiscreteLogarithm(None, HALF), big,
             DiscreteLogarithm(1, HALF)),
            ("two_squares", dict(n=25), SumOfTwoSquares(12, 3, 4), (factors_of(25),), SumOfTwoSquares(8, 3, 4)),
            ("two_squares", dict(n=0), SumOfTwoSquares(1, 0, 0), ((),), SumOfTwoSquares(0, None, None)),
            ("two_squares", dict(n=M89), SumOfTwoSquares(0, None, None), (((M89, 1, certificate(M89)),),),
             SumOfTwoSquares(4, 0, 1)),
        ]
        for operation, args, value, evidence, wrong in claims:
            with self.subTest(operation=operation, args=args):
                self.roundtrip(claimed(operation, args, value, evidence))
                report = support.statuses(mf.verify(claimed(operation, args, wrong, evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
