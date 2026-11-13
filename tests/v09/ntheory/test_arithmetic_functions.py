"""The arithmetic_functions family of mathforge.ntheory: moebius, divisor_sigma, integer_nth_root.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register ntheory arithmetic_functions

Every expected value comes from this module, never from the producer: from the card (docs/v09/cards/m1-ntheory.md),
from a slow reference (trial division, the divisors enumerated in pairs up to the square root, bisection), or from the
factorization that a test multiplies together from primes it knows. The checker has tests of its own
(test_arithmetic_functions_checker.py); the acceptance and corruption tests here start from what the producer returns.
"""

from __future__ import annotations

import functools
from math import gcd, isqrt, prod
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.ntheory import arithmetic_functions as family
from mathforge.areas.ntheory.continued_fraction import ContinuedFraction
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.kernels.certificates import factorization_evidence, prime_evidence
from mathforge.kernels.integers import factor
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.trusted.integers import factorization_work
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13

from .. import support

SEED = 20261112
FAST = 3.0  # Seconds: an answer at the integer limit comes well within this under the default budget.
M61, M89, M107, M127 = 2 ** 61 - 1, 2 ** 89 - 1, 2 ** 107 - 1, 2 ** 127 - 1  # Primes: M61 below psi_13, the rest above.
F12 = ((2, 2, ()), (3, 1, ()))
F30 = ((2, 1, ()), (3, 1, ()), (5, 1, ()))


class Integer(int):
    """An int subclass: equal to its int, but no exact int."""


@functools.cache
def primes_between(low: int, high: int) -> tuple[int, ...]:
    """The primes p with low <= p < high, by the sieve of Eratosthenes."""
    sieve = bytearray([1]) * high
    sieve[:2] = b"\0\0"
    for p in range(2, isqrt(high - 1) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytes(len(range(p * p, high, p)))
    return tuple(p for p in range(low, high) if sieve[p])


def trial_factors(n: int) -> tuple[tuple[int, int], ...]:
    """The prime factorization ((p, e), ...) of n >= 1 by trial division."""
    found, p = [], 2
    while p * p <= n:
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            found.append((p, e))
        p += 1 if p == 2 else 2
    if n > 1:
        found.append((n, 1))
    return tuple(found)


def moebius_of(factors: tuple[tuple[int, int], ...]) -> int:
    """mu(prod p**e) by its definition: 0 when an exponent exceeds 1, else (-1)**(number of primes)."""
    return 0 if any(e > 1 for _, e in factors) else (-1) ** len(factors)


def sigma_by_divisors(n: int, k: int) -> int:
    """sigma_k(n) by its definition: d**k summed over the divisors d of n, found in pairs (d, n // d) up to isqrt(n)."""
    total = 0
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            total += d ** k
            if d * d != n:
                total += (n // d) ** k
    return total


def sigma_by_series(factors: tuple[tuple[int, int], ...], k: int) -> int:
    """sigma_k(prod p**e): the divisors of each prime power summed, prod over p of (sum of (p**i)**k, i = 0 ... e)."""
    return prod(sum((p ** i) ** k for i in range(e + 1)) for p, e in factors)


def horner_charge(factors: tuple, k: int) -> int:
    """What the checker of divisor_sigma charges after check_factorization, replayed here with the Horner sums formed:
    for each prime a unit, p**k as two products of k * bits(p) bits per bit of k, a product for each Horner step
    s -> s * p**k + 1, and the product of the sums so far with the whole sum; a product of integers of a and b bits
    costs 1 + ((a >> 6) * (b >> 6) >> 8), one unit per 256 pairs of their 64-bit words."""
    def product(left: int, right: int) -> int:
        return 1 + ((left >> 6) * (right >> 6) >> 8)

    work, total = 0, 1
    for p, e, _ in factors:
        bits = p.bit_length() * k
        work += 1 + (2 * k.bit_length() * product(bits, bits) if k else 0)
        step, part = p ** k, 1
        for _ in range(e):
            work += product(part.bit_length(), step.bit_length())
            part = part * step + 1
        work += product(total.bit_length(), part.bit_length())
        total *= part
    return work


def root_by_bisection(n: int, k: int) -> int:
    """floor(n**(1/k)) by bisection on [0, 2**(bits(n) // k + 1)), whose upper end has a k-th power above n."""
    low, high = 0, 1 << (n.bit_length() // k + 1)
    while high - low > 1:
        middle = (low + high) // 2
        if middle ** k <= n:
            low = middle
        else:
            high = middle
    return low


def root_by_counting(n: int, k: int) -> int:
    """floor(n**(1/k)) by counting up from 0."""
    r = 0
    while (r + 1) ** k <= n:
        r += 1
    return r


def pocklington(n: int, first: int) -> tuple:
    """A v0.7 Pocklington certificate of the prime n >= 10,000, built here rather than by the kernel: n - 1 factored by
    trial division, for each prime p of n - 1 the least witness w >= first with w**(n - 1) = 1 and
    gcd(w**((n - 1)/p) - 1, n) = 1 modulo n, ("trial", p) below 10,000 and the same construction above."""
    factors = trial_factors(n - 1)
    nested = tuple((p, e, ("trial", p) if p < 10_000 else pocklington(p, first)) for p, e in factors)
    witnesses = []
    for p, _ in factors:
        w = first
        while pow(w, n - 1, n) != 1 or gcd(pow(w, (n - 1) // p, n) - 1, n) != 1:
            w += 1
        witnesses.append(w)
    return ("pocklington", n, nested, tuple(witnesses))


class ArithmeticFunctionsTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, limits: ComputationLimits | None = None) -> None:
        """A completed Result with the four stored checks of a verified v6 value, which mf.verify accepts."""
        self.assertIsNone(result.error, result.error)
        self.assertEqual([check.claim for check in result.verification], list(support.CLAIMS))
        self.assertTrue(all(check.status is CheckStatus.VERIFIED for check in result.verification))
        report = mf.verify(result, limits=limits)
        self.assertTrue(report.verified, support.statuses(report))

    def assertAgrees(self, operation: str, args: dict[str, int], expected: int) -> tuple:
        """produce() answers the expected exact int with evidence that the checker accepts; returns the evidence."""
        value, evidence = family.produce(operation, dict(args))
        self.assertIs(type(value), int)
        self.assertEqual(value, expected)
        self.assertIs(support.verdict(family, operation, dict(args), value, evidence), True)
        return evidence

    # -- the structured path --------------------------------------------------------------------------------------

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts, with the card's evidence."""
        cases = (("moebius", dict(n=1), 1, ("factorization", ())),
                 ("moebius", dict(n=2), -1, ("factorization", ((2, 1, ()),))),
                 ("moebius", dict(n=4), 0, ("square", 2)),
                 ("moebius", dict(n=6), 1, ("factorization", ((2, 1, ()), (3, 1, ())))),
                 ("moebius", dict(n=12), 0, ("square", 2)),
                 ("moebius", dict(n=30), -1, ("factorization", F30)),
                 ("divisor_sigma", dict(n=12, k=1), 28, (F12,)),
                 ("divisor_sigma", dict(n=12, k=0), 6, (F12,)),
                 ("divisor_sigma", dict(n=6, k=2), 50, (((2, 1, ()), (3, 1, ())),)),
                 ("divisor_sigma", dict(n=1, k=5), 1, ((),)),
                 ("integer_nth_root", dict(n=26, k=3), 2, ()),
                 ("integer_nth_root", dict(n=27, k=3), 3, ()),
                 ("integer_nth_root", dict(n=0, k=5), 0, ()),
                 ("integer_nth_root", dict(n=1, k=100), 1, ()),
                 ("integer_nth_root", dict(n=10 ** 20, k=2), 10 ** 10, ()),
                 ("integer_nth_root", dict(n=2 ** 64 - 1, k=2), 2 ** 32 - 1, ()))
        for operation, arguments, expected, evidence in cases:
            with self.subTest(operation=operation, **arguments):
                result = support.run(operation, **arguments)
                self.assertVerified(result)
                self.assertIs(type(result.value), int)
                self.assertEqual(result.value, expected)
                self.assertEqual(result.request.version, 6)
                self.assertEqual(result.request.arguments, tuple(sorted(arguments.items())))  # Sorted by name.
                self.assertEqual(result.steps[0].rule, "v6." + operation)
                self.assertEqual(result.steps[0].outputs, (expected, evidence))
                self.roundtrip(result)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error: in every
        argument, through run() and from produce() itself, and so is every input outside the card's domains."""
        outside = (("moebius", dict(n=0)), ("moebius", dict(n=-1)), ("moebius", dict(n=-30)),
                   ("divisor_sigma", dict(n=0, k=1)), ("divisor_sigma", dict(n=-12, k=1)),
                   ("divisor_sigma", dict(n=12, k=-1)), ("divisor_sigma", dict(n=1, k=-5)),
                   ("integer_nth_root", dict(n=-1, k=2)), ("integer_nth_root", dict(n=-27, k=3)),
                   ("integer_nth_root", dict(n=8, k=0)), ("integer_nth_root", dict(n=8, k=-3)),
                   ("integer_nth_root", dict(n=-8, k=-3)))
        for operation, arguments in outside:
            with self.subTest(operation=operation, **arguments):
                self.assertInvalidInput(support.run(operation, **arguments))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(arguments))
        valid = {"moebius": dict(n=30), "divisor_sigma": dict(n=12, k=1), "integer_nth_root": dict(n=26, k=3)}
        for operation, arguments in valid.items():
            for name in arguments:
                for wrong in (0.5, -0.0, True, False, "1", None, [1], (1, 2), {"n": 1}, mf.Rational(1), Integer(1)):
                    changed = dict(arguments, **{name: wrong})
                    with self.subTest(operation=operation, argument=name, value=wrong):
                        self.assertInvalidInput(support.run(operation, **changed))
                        with self.assertRaises(InvalidInput):  # run() rejects these before produce() runs.
                            family.produce(operation, changed)

    def test_resource_limits(self):
        """An integer of 2**65536 in any argument and max_work=1 exceed the budget; the family takes no matrix."""
        huge = (("moebius", dict(n=support.HUGE)), ("divisor_sigma", dict(n=support.HUGE, k=1)),
                ("divisor_sigma", dict(n=12, k=support.HUGE)), ("integer_nth_root", dict(n=support.HUGE, k=2)),
                ("integer_nth_root", dict(n=26, k=support.HUGE)))
        for operation, arguments in huge:
            with self.subTest(operation=operation, arguments=sorted(arguments)):
                self.assertResourceLimit(support.run(operation, **arguments))
                with self.assertRaises(ResourceLimitError):
                    family.produce(operation, dict(arguments))
        negative = (("moebius", dict(n=-support.HUGE)), ("integer_nth_root", dict(n=8, k=-support.HUGE)))
        for operation, arguments in negative:
            with self.subTest(operation=operation, arguments=sorted(arguments)):  # run() inspects sizes first.
                self.assertResourceLimit(support.run(operation, **arguments))
        tiny = ComputationLimits(max_work=1)
        for operation, arguments in (("moebius", dict(n=30)), ("moebius", dict(n=2 ** 64 + 1)),
                                     ("divisor_sigma", dict(n=12, k=1)), ("divisor_sigma", dict(n=10 ** 6, k=3)),
                                     ("integer_nth_root", dict(n=26, k=3)),
                                     ("integer_nth_root", dict(n=10 ** 40, k=7))):
            with self.subTest(operation=operation, limits="max_work=1", **arguments):
                self.assertResourceLimit(support.run(operation, limits=tiny, **arguments))

    def test_divisor_sigma_cap(self):
        """k * bits(n) <= max_integer_bits is answered and verified; beyond it divisor_sigma is a resource error before
        any factoring, at the cap where the checker raises too, and wider limits answer it."""
        result = support.run("divisor_sigma", n=2 ** 15, k=4096)  # k * bits(n) = 65,536: the cap itself.
        self.assertVerified(result)
        self.assertEqual(result.value, sigma_by_series(((2, 15),), 4096))
        result = support.run("divisor_sigma", n=2 ** 65535, k=1)  # The longest n, at the cap.
        self.assertVerified(result)
        self.assertEqual(result.value, 2 ** 65536 - 1)
        self.assertVerified(support.run("divisor_sigma", n=1, k=65_536))
        for n, k, factors in ((3, 32_768, ((3, 1),)), (65_535, 4096, ((3, 1), (5, 1), (17, 1), (257, 1)))):
            with self.subTest(n=n, k=k):
                result = support.run("divisor_sigma", n=n, k=k)
                self.assertVerified(result)
                self.assertEqual(result.value, sigma_by_series(factors, k))
        # At the cap the value itself may pass max_integer_bits, here by one bit: refused like every result integer
        # that long, never answered.
        n = 2 ** 512 - 2 ** 376  # 512 bits, so k = 128 is at the cap.
        factors = ((2, 376), (3, 1), (5, 1), (17, 2), (137, 1), (953, 1), (26317, 1), (43691, 1), (131071, 1),
                   (354689, 1), (2879347902817, 1))
        self.assertEqual(prod(p ** e for p, e in factors), n)
        self.assertEqual(sigma_by_series(factors, 128).bit_length(), 65_537)
        result = support.run("divisor_sigma", n=n, k=128)
        self.assertResourceLimit(result)
        self.assertEqual(result.error.message, "Computation limit exceeded: integer bits")
        with patch("mathforge.kernels.certificates.factorization_evidence", side_effect=AssertionError("factored")):
            for n, k in ((2 ** 10, 6000), (1, 65_537), (3, 32_769), (M89 * M107, 1000), (12, 2 ** 64)):
                with self.subTest(n=n, k=k):
                    self.assertResourceLimit(support.run("divisor_sigma", n=n, k=k))
        wide = ComputationLimits(max_integer_bits=66_000)
        result = support.run("divisor_sigma", n=2 ** 10, k=6000, limits=wide)
        self.assertVerified(result, limits=wide)
        self.assertEqual(result.value, sigma_by_series(((2, 10),), 6000))
        # Under the default limits the same claim is undecided, never refuted.
        self.assertEqual(support.statuses(mf.verify(result)), {"result.resources": CheckStatus.UNKNOWN})

    def test_factorizations_beyond_the_budget_are_resource_errors(self):
        """An integer whose two prime factors of about 100 bits the kernel's rho cannot separate within the budget is
        resource_limit_exceeded, never a value; with its factorization known, the checker's tests verify it."""
        for operation, arguments in (("moebius", dict(n=M89 * M107)), ("moebius", dict(n=M89 ** 2 * M107)),
                                     ("divisor_sigma", dict(n=M89 * M107, k=1))):
            with self.subTest(operation=operation, **arguments):
                self.assertResourceLimit(support.run(operation, **arguments))

    def checker_work(self, operation: str, args: dict[str, int], value: int, evidence: tuple) -> int:
        """What the family's checker really charges to accept a claim, in a wide fresh budget; and with one unit less
        it runs out."""
        with computation(ComputationLimits(max_work=10 ** 9), fresh=True) as budget:
            self.assertIs(support.checker(family).check_operation(operation, args, value, evidence), True)
        with computation(ComputationLimits(max_work=budget.work - 1), fresh=True):
            with self.assertRaises(ResourceLimitError):
                support.checker(family).check_operation(operation, args, value, evidence)
        return budget.work

    def assertAnswersWithin(self, operation: str, args: dict[str, int], max_work: int, claim: tuple) -> None:
        """produce() answers the claim (value, evidence) under max_work, and run() returns it verified."""
        limits = ComputationLimits(max_work=max_work)
        with computation(limits, fresh=True):
            self.assertEqual(family.produce(operation, dict(args)), claim)
        result = support.run(operation, limits=limits, **args)
        self.assertVerified(result, limits=limits)
        self.assertEqual(result.steps[0].outputs, claim)

    def assertRefusesWithin(self, operation: str, args: dict[str, int], max_work: int) -> None:
        """produce() refuses under max_work because the check would not fit, and run() reports that refusal, never a
        verification that ran out."""
        limits = ComputationLimits(max_work=max_work)
        with computation(limits, fresh=True):
            with self.assertRaisesRegex(ResourceLimitError, "independent check of the evidence"):
                family.produce(operation, dict(args))
        result = support.run(operation, limits=limits, **args)
        self.assertResourceLimit(result)
        self.assertIn("independent check of the evidence", result.error.message)

    def test_moebius_refuses_evidence_its_check_cannot_afford(self):
        """kernels.certificates' rule: the check fits in max_work on its own, as the verifier checks in a fresh budget
        of the same limits, where the check is all that the family's checker charges: check_factorization's
        factorization_work and one unit per entry for the checker's own exponent screen, which kernels.certificates
        asks the family to leave room for. Production and check need not fit together: 3 * M89 costs the producer more
        than its check, and max_work = production answers. Where the check costs more, for 3 * (5 * 2**3313 + 1) with
        the kernel's factor() of the large prime replaced by its answer, moebius answers with exactly the check as
        max_work and refuses with one unit less, before the checker would run out."""
        wide = ComputationLimits(max_work=10 ** 9)
        n = 3 * M89  # Squarefree, with a prime above psi_13 that takes a Pocklington certificate.
        with computation(wide, fresh=True) as budget:
            value, evidence = family.produce("moebius", {"n": n})
            production = budget.work
        self.assertEqual((value, evidence), (1, ("factorization", ((3, 1, ()), (M89, 1, prime_evidence(M89))))))
        check = self.checker_work("moebius", {"n": n}, value, evidence)
        self.assertEqual(check, factorization_work(evidence[1], 10 ** 9) + len(evidence[1]))
        self.assertGreater(production, check)
        self.assertAnswersWithin("moebius", {"n": n}, production, (value, evidence))
        large = 5 * 2 ** 3313 + 1  # A prime whose Pocklington certificate costs less to build than to check.
        original = factor

        def answered(m: int) -> tuple:
            return ((large, 1),) if m == large else original(m)

        n = 3 * large
        with patch("mathforge.kernels.integers.factor", side_effect=answered):
            with computation(wide, fresh=True) as budget:
                value, evidence = family.produce("moebius", {"n": n})
                production = budget.work
            self.assertEqual((value, evidence), (1, ("factorization", ((3, 1, ()), (large, 1, prime_evidence(large))))))
            check = self.checker_work("moebius", {"n": n}, value, evidence)
            self.assertEqual(check, factorization_work(evidence[1], 10 ** 9) + len(evidence[1]))
            self.assertLess(production, check)
            self.assertAnswersWithin("moebius", {"n": n}, check, (value, evidence))
            self.assertRefusesWithin("moebius", {"n": n}, check - 1)
        self.assertVerified(support.run("moebius", n=3 * M89))

    def test_divisor_sigma_refuses_evidence_its_check_cannot_afford(self):
        """The same rule for divisor_sigma, whose checker charges check_factorization's factorization_work and its
        Horner evaluation with the powers p**k (horner_charge), all of which the producer reserves. Where the check
        costs more than the production, divisor_sigma answers with exactly the check as max_work and refuses with one
        unit less, and run() reports the producer's refusal, never a verification that ran out; where the production
        costs more, max_work = production suffices for both. Before the producer reserved the Horner evaluation, it
        answered sigma_32768(3), whose p**k costs the checker 131,104 units, under max_work=100_000, and verification
        ran out of work instead."""
        wide = ComputationLimits(max_work=10 ** 9)
        # p**k dominating the check; 4,096 Horner steps; four primes at the cap (these three cost the checker more
        # than the producer); Horner steps with a p**k of thousands of bits at the cap; a prime above psi_13.
        for n, k, factors in ((3, 32_768, ((3, 1),)), (2 ** 4096, 1, ((2, 4096),)),
                              (65535, 4096, ((3, 1), (5, 1), (17, 1), (257, 1))),
                              (2 ** 15 * 3 ** 7, 2427, ((2, 15), (3, 7))), (3 ** 5 * M89, 600, ((3, 5), (M89, 1)))):
            args = {"n": n, "k": k}
            with self.subTest(bits=n.bit_length(), k=k):
                with computation(wide, fresh=True) as budget:
                    value, evidence = family.produce("divisor_sigma", dict(args))
                    production = budget.work
                self.assertEqual(value, sigma_by_series(factors, k))
                check = self.checker_work("divisor_sigma", args, value, evidence)
                self.assertEqual(check, factorization_work(evidence[0], 10 ** 9) + horner_charge(evidence[0], k))
                self.assertAnswersWithin("divisor_sigma", args, max(production, check), (value, evidence))
                if production < check:
                    self.assertRefusesWithin("divisor_sigma", args, check - 1)
        self.assertRefusesWithin("divisor_sigma", {"n": 3, "k": 32_768}, 100_000)
        result = support.run("divisor_sigma", n=3, k=32_768, limits=ComputationLimits(max_work=100_000))
        self.assertEqual(result.error.message,
                         "Computation limit exceeded: work (100000) for the independent check of the evidence")

    def test_moebius_zero_needs_no_prime_certificate(self):
        """mu(n) = 0 rests on a square alone: with kernels.certificates disabled, 5 * M127**2 is still answered, with
        its least square prime M127, which is above psi_13."""
        disabled = AssertionError("moebius certified a prime for mu(n) = 0")
        with patch("mathforge.kernels.certificates.prime_evidence", side_effect=disabled), \
                patch("mathforge.kernels.certificates.factorization_evidence", side_effect=disabled):
            result = support.run("moebius", n=5 * M127 ** 2)
            self.assertVerified(result)
            self.assertEqual(result.steps[0].outputs, (0, ("square", M127)))
            failed = support.run("moebius", n=5 * M127)  # The patch is live: a squarefree n needs the certificate.
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))
        self.assertGreater(M127, PSI_13)

    def test_pseudoprimes_and_the_deterministic_bound(self):
        """Carmichael numbers and base-2 pseudoprimes above 10,000 (some with a prime factor above the trial bound
        2**10), the strong pseudoprimes psi_5 ... psi_12 to the first prime bases, and the primes next to psi_13 (the
        one below certified by (), the one above by a Pocklington certificate), alone, squared and in products: each
        value verified and equal to the one of the factorization that the test multiplies together. psi_13 itself, two
        primes of about 41 bits, is beyond what the kernel's rho factors within the default budget."""
        below, above = 3317044064679887385961813, 3317044064679887385962123  # The primes next to psi_13.
        pseudoprimes = ((7, 31, 73), (7, 11, 13, 41), (3, 5, 47, 89), (59, 1103), (5, 17, 1429),
                        (6763, 10627, 29947), (1303, 16927, 157543), (10670053, 32010157),
                        (149491, 747451, 34233211), (399165290221, 798330580441))
        for primes in pseudoprimes:
            self.assertEqual(pow(2, prod(primes) - 1, prod(primes)), 1)  # Each passes the Fermat test to base 2.
        self.assertTrue(below < PSI_13 < above)
        cases = [tuple((p, 1) for p in primes) for primes in pseudoprimes]
        cases += [((below, 1),), ((above, 1),), ((3, 1), (above, 1)), ((1031, 1), (below, 1)), ((below, 2),),
                  ((above, 2),), ((7, 1), (13, 1), (19, 1), (1031, 2))]
        for factors in cases:
            n = prod(p ** e for p, e in factors)
            with self.subTest(n=n):
                result = support.run("moebius", n=n)
                self.assertVerified(result)
                self.assertEqual(result.value, moebius_of(factors))
                for k in (0, 1, 3):
                    result = support.run("divisor_sigma", n=n, k=k)
                    self.assertVerified(result)
                    self.assertEqual(result.value, sigma_by_series(factors, k))
        self.assertResourceLimit(support.run("moebius", n=PSI_13))

    def test_integer_nth_root_at_the_integer_limit(self):
        """n of 65,536 bits, the longest that the default limits admit, with k from 2 to beyond its length: each root
        is answered and verified under the default budget, fast, and satisfies r**k <= n < (r + 1)**k."""
        for n in (2 ** 65536 - 1, 2 ** 65535 + 12345, 3 ** 41000):
            for k in (2, 3, 5, 7, 64, 1000, 2600, 5000, 30000, 65000, 65535, 65536, 10 ** 18):
                with self.subTest(bits=n.bit_length(), k=k):
                    start = time.perf_counter()
                    result = support.run("integer_nth_root", n=n, k=k)
                    self.assertVerified(result)
                    self.assertLess(time.perf_counter() - start, FAST)
                    r = result.value
                    if k >= n.bit_length():
                        self.assertEqual(r, 1)  # 2**k > n.
                    else:
                        self.assertTrue(r ** k <= n < (r + 1) ** k)
                    if k == 2:
                        self.assertEqual(r, isqrt(n))
        self.assertEqual(support.run("integer_nth_root", n=3 ** 41000, k=41000).value, 3)

    def test_unknown_operation(self):
        """produce() names no operation outside SPECS; run() reaches it only with the family's own."""
        with self.assertRaises(UnsupportedOperation):
            family.produce("continued_fraction", {"value": 1})

    # -- the checker on the producer's claims ---------------------------------------------------------------------

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: any s > 1
        with s*s | n where the producer takes the least prime, and Pocklington certificates with other witnesses."""
        for n, squares in ((900, None), (3600, None), (2 ** 10 * 3 ** 4, None), (10 ** 6, None),
                           (1031 ** 2 * 1033 ** 2, (1031, 1033, 1031 * 1033)), (3 * M61 ** 2, (M61,))):
            witnesses = squares or tuple(s for s in range(2, isqrt(n) + 1) if n % (s * s) == 0)
            with self.subTest(n=n):
                self.assertEqual(family.produce("moebius", {"n": n}), (0, ("square", witnesses[0])))
                for s in witnesses:
                    self.assertAccepts("moebius", {"n": n}, 0, ("square", s))
        other = pocklington(M89, 5)  # Witnesses from 5 on, where the kernel takes 3 first.
        self.assertNotEqual(other, prime_evidence(M89))
        self.assertEqual(family.produce("moebius", {"n": 6 * M89}),
                         (-1, ("factorization", ((2, 1, ()), (3, 1, ()), (M89, 1, prime_evidence(M89))))))
        self.assertAccepts("moebius", {"n": M89}, -1, ("factorization", ((M89, 1, other),)))
        self.assertAccepts("moebius", {"n": 6 * M89}, -1, ("factorization", ((2, 1, ()), (3, 1, ()), (M89, 1, other))))
        for k in (0, 1, 2, 5):
            with self.subTest(k=k):
                self.assertAccepts("divisor_sigma", {"n": 4 * M89, "k": k}, sigma_by_series(((2, 2), (M89, 1)), k),
                                   (((2, 2, ()), (M89, 1, other)),))
        # integer_nth_root has no evidence to choose; values that the bisection computes, at the extremes of k (every k
        # beyond bits(n) has the root of k = bits(n) + 1).
        for n, k in ((5, 10 ** 18), (0, 10 ** 18), (2 ** 200, 199), (2 ** 200, 200), (2 ** 200, 201), (10 ** 30, 3)):
            with self.subTest(n=n, k=k):
                expected = root_by_bisection(n, min(k, n.bit_length() + 1))
                self.assertAccepts("integer_nth_root", {"n": n, "k": k}, expected)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence. The family has no record class, so an
        int subclass and the records of continued_fraction stand in for impostors and forged records."""
        claims = (("moebius", {"n": 30}), ("moebius", {"n": 12}), ("moebius", {"n": 1}), ("moebius", {"n": 3 * M89}),
                  ("divisor_sigma", {"n": 12, "k": 1}), ("divisor_sigma", {"n": 1, "k": 5}),
                  ("divisor_sigma", {"n": 3 * M89, "k": 2}), ("integer_nth_root", {"n": 26, "k": 3}),
                  ("integer_nth_root", {"n": 0, "k": 5}), ("integer_nth_root", {"n": 10 ** 40, "k": 7}))
        names = [spec.name for spec in family.SPECS]
        for operation, args in claims:
            value, evidence = family.produce(operation, dict(args))
            with self.subTest(operation=operation, args=args):
                self.assertAccepts(operation, args, value, evidence)
                for wrong in {value - 1, value + 1, -value, -value - 2, 2, -2, value + 2 ** 64} - {value}:
                    self.assertRejects(operation, args, wrong, evidence)
                stand_ins = [float(value), str(value), Integer(value), (value,), [value], None,
                             ContinuedFraction((value,)), support.impostor(ContinuedFraction((value,))),
                             support.forged(ContinuedFraction, terms=(value,)), support.forged(ContinuedFraction)]
                if value in (0, 1):
                    stand_ins.append(bool(value))
                for stand_in in stand_ins:
                    self.assertRejects(operation, args, stand_in, evidence)
                broken = [list(evidence), evidence[:-1], evidence + ((),), (evidence,), None, ()]
                if evidence and type(evidence[-1]) is tuple:  # The factorization, as a list and with list entries.
                    broken.append(evidence[:-1] + (list(evidence[-1]),))
                    if evidence[-1]:
                        broken.append(evidence[:-1] + ((list(evidence[-1][0]),) + evidence[-1][1:],))
                for corrupted in broken:
                    if corrupted != evidence or type(corrupted) is not tuple:
                        self.assertRejects(operation, args, value, corrupted)
                for name in ("Moebius", "mobius", operation + " ", "", "continued_fraction", None,
                             *(other for other in names if other != operation)):
                    self.assertRejects(name, args, value, evidence)
        # Evidence and values far beyond any true claim: refused without building what they suggest.
        oversized = (("moebius", {"n": 30}, -1, ("factorization", ((2, 1, ()),) * 100_000)),
                     ("moebius", {"n": 30}, 0, ("square", support.HUGE)),
                     ("moebius", {"n": 30}, support.HUGE, ("factorization", F30)),
                     ("divisor_sigma", {"n": 12, "k": 1}, 28, (((2, 1, ()),) * 100_000,)),
                     ("divisor_sigma", {"n": 12, "k": 1}, support.HUGE, (F12,)),
                     ("divisor_sigma", {"n": 12, "k": 1}, 28, (((2, support.HUGE, ()), (3, 1, ())),)),
                     ("integer_nth_root", {"n": 26, "k": 3}, 2, (0,) * 100_000),
                     ("integer_nth_root", {"n": 26, "k": 3}, support.HUGE, ()))
        for index, (operation, args, value, evidence) in enumerate(oversized):
            with self.subTest(operation=operation, oversized=index):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, evidence)
                self.assertLess(time.perf_counter() - start, FAST)
        # mf.verify of a Result whose value or evidence was replaced: the trace holds, the mathematics fails.
        for operation, arguments, value, evidence in (("moebius", dict(n=30), 1, None),
                                                      ("moebius", dict(n=12), 0, ("square", 3)),
                                                      ("moebius", dict(n=30), -1, ("factorization", F30[:2])),
                                                      ("divisor_sigma", dict(n=12, k=1), 27, None),
                                                      ("divisor_sigma", dict(n=12, k=1), 28, (F30,)),
                                                      ("integer_nth_root", dict(n=26, k=3), 3, None)):
            with self.subTest(operation=operation, tampered=value):
                report = support.statuses(mf.verify(support.tampered(support.run(operation, **arguments), value,
                                                                     evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        results = [support.run("moebius", n=30), support.run("moebius", n=12), support.run("moebius", n=3 * M89),
                   support.run("divisor_sigma", n=360, k=2), support.run("integer_nth_root", n=10 ** 40, k=7)]
        decoded = [mf.loads(mf.dumps(result)) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for result, copy in zip(results, decoded):
                with self.subTest(operation=result.request.operation):
                    self.assertTrue(mf.verify(result).verified)
                    self.assertTrue(mf.verify(copy).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("moebius", n=30)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value, a plain int, renders as text and as LaTeX."""
        cases = (("moebius", dict(n=30), "-1"), ("moebius", dict(n=12), "0"), ("moebius", dict(n=6), "1"),
                 ("divisor_sigma", dict(n=12, k=1), "28"), ("integer_nth_root", dict(n=26, k=3), "2"),
                 ("integer_nth_root", dict(n=10 ** 9000, k=2), "1" + "0" * 4500))  # Beyond str()'s 4,300 digits.
        for operation, arguments, text in cases:
            with self.subTest(operation=operation, text=text[:8]):
                result = support.run(operation, **arguments)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                for value in (result.value, decoded.value):
                    self.assertEqual(mf.render_text(value), text)
                    self.assertEqual(mf.to_latex(value), text)
        result = support.run("divisor_sigma", n=12, k=1)
        self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: 28")
        self.assertEqual(mf.to_data(result.value)["data"], {"type": "integer", "value": "28"})
        self.assertEqual(mf.to_data(support.run("moebius", n=30).value)["data"], {"type": "integer", "value": "-1"})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path; an input
        that the structured path refuses falls back and raises its error."""
        rng = random.Random(SEED)
        cases = [("moebius", dict(n=rng.randint(1, 10 ** 12))) for _ in range(25)]
        cases += [("divisor_sigma", dict(n=rng.randint(1, 10 ** 12), k=rng.randint(0, 6))) for _ in range(25)]
        cases += [("integer_nth_root", dict(n=rng.getrandbits(rng.randint(1, 1500)), k=rng.randint(1, 40)))
                  for _ in range(25)]
        for operation, arguments in cases:
            with self.subTest(operation=operation, **arguments):
                structured = support.run(operation, **arguments)
                self.assertVerified(structured)
                found = support.lean_value(operation, **arguments)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, structured.value)
                self.assertEqual(support.value(operation, **arguments), structured.value)
        for operation, arguments, code in (("moebius", dict(n=0), "invalid_input"),
                                           ("integer_nth_root", dict(n=8, k=0), "invalid_input"),
                                           ("divisor_sigma", dict(n=2, k=70_000), "resource_limit_exceeded")):
            with self.subTest(operation=operation, **arguments):
                self.assertIs(support.lean_value(operation, **arguments), lean.FALLBACK)
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(operation, **arguments)
                self.assertEqual(raised.exception.code, code)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, arguments, values in (("moebius", {"n": 30}, dict(n=30)),
                                             ("divisor_sigma", {"n": "360", "k": 2}, dict(n=360, k=2)),
                                             ("integer_nth_root", {"n": str(10 ** 20), "k": 2}, dict(n=10 ** 20, k=2))):
            with self.subTest(operation=operation):
                result = self.answer(support.call(operation, arguments))
                expected = support.run(operation, **values)
                self.assertEqual(result, expected)
                self.assertEqual(mf.dumps(result), mf.dumps(expected))
                self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call_values("moebius", n=3 * M89))
        self.assertEqual(result.value, 1)
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(self.answer(support.call_values("integer_nth_root", n=2 ** 64 - 1, k=2)).value, 2 ** 32 - 1)
        for operation, arguments, code in (("moebius", {"n": 0}, "invalid_input"),
                                           ("moebius", {"n": True}, "invalid_input"),
                                           ("integer_nth_root", {"n": 8, "k": 0}, "invalid_input"),
                                           ("divisor_sigma", {"n": 2, "k": 70_000}, "resource_limit_exceeded")):
            with self.subTest(operation=operation, **arguments):
                self.assertToolError(support.call(operation, arguments), code)

    # -- seeded agreement with slow references ---------------------------------------------------------------------

    def test_moebius_agrees_with_a_slow_reference(self):
        """mu(n) for n = 1 ... 2000 and for 400 seeded larger n, against trial division or the factorization that the
        test multiplies together. The evidence is the least prime whose square divides n, or exactly the certified
        factorization that kernels.certificates builds, and the checker accepts it."""
        rng = random.Random(SEED + 1)
        small, pool = primes_between(2, 1031), primes_between(1031, 2 ** 20)
        cases = [(n, trial_factors(n)) for n in range(1, 2001)]
        cases += [(n, trial_factors(n)) for n in (rng.randint(2001, 10 ** 7) for _ in range(150))]
        for _ in range(250):  # Primes on both sides of the kernel's trial division (2**10), some squared or cubed.
            chosen = rng.sample(small, rng.randint(0, 3)) + rng.sample(pool, rng.randint(1, 3))
            factors = tuple(sorted((p, 1 if rng.randint(1, 5) <= 3 else rng.randint(2, 3)) for p in chosen))
            cases.append((prod(p ** e for p, e in factors), factors))
        cases.append((prod(small), tuple((p, 1) for p in small)))  # Every prime below 2**10 once.
        cases.append((M61 * 1031, ((1031, 1), (M61, 1))))
        cases.append((7 * M127, ((7, 1), (M127, 1))))
        self.assertGreaterEqual(sum(1 for n, _ in cases if n > 2000), 300)  # Seeded, beyond the exhaustive range.
        for n, factors in cases:
            expected = moebius_of(factors)
            with self.subTest(n=n):
                evidence = self.assertAgrees("moebius", {"n": n}, expected)
                if expected:
                    self.assertEqual(evidence, ("factorization", factorization_evidence(n)))
                else:
                    self.assertEqual(evidence, ("square", min(p for p, e in factors if e > 1)))

    def test_divisor_sigma_agrees_with_a_slow_reference(self):
        """sigma_k(n) for n = 1 ... 300 and k = 0 ... 3, and for 350 seeded larger pairs, against the divisors summed
        in pairs or the geometric series of a factorization that the test multiplies together. The evidence is
        exactly factorization_evidence(n), and the checker accepts it."""
        rng = random.Random(SEED + 2)
        small, pool = primes_between(2, 1031), primes_between(1031, 2 ** 20)
        cases = [(n, k, sigma_by_divisors(n, k)) for n in range(1, 301) for k in range(4)]
        for _ in range(200):
            n, k = rng.randint(301, 2 * 10 ** 5), rng.randint(0, 8)
            cases.append((n, k, sigma_by_divisors(n, k)))
        for _ in range(150):
            chosen = rng.sample(small, rng.randint(0, 3)) + rng.sample(pool, rng.randint(1, 2))
            factors = tuple(sorted((p, rng.randint(1, 4)) for p in chosen))
            k = rng.randint(0, 6)
            cases.append((prod(p ** e for p, e in factors), k, sigma_by_series(factors, k)))
        cases.append((3 * M89, 3, sigma_by_series(((3, 1), (M89, 1)), 3)))
        self.assertGreaterEqual(sum(1 for n, _, _ in cases if n > 300), 300)  # Seeded, beyond the exhaustive range.
        for n, k, expected in cases:
            with self.subTest(n=n, k=k):
                evidence = self.assertAgrees("divisor_sigma", {"n": n, "k": k}, expected)
                self.assertEqual(evidence, (factorization_evidence(n),))

    def test_integer_nth_root_agrees_with_a_slow_reference(self):
        """floor(n**(1/k)) for n = 0 ... 300 and k = 1 ... 8 by counting, and for seeded n of up to 2048 bits, perfect
        powers and their neighbours, and powers of 2 where k crosses their length, by bisection (math.isqrt as well
        for k = 2). The evidence is (), and the checker accepts every value."""
        rng = random.Random(SEED + 3)
        for n in range(301):
            for k in range(1, 9):
                with self.subTest(n=n, k=k):
                    evidence = self.assertAgrees("integer_nth_root", {"n": n, "k": k}, root_by_counting(n, k))
                    self.assertEqual(evidence, ())
        cases = [(rng.getrandbits(rng.randint(1, 2048)), rng.randint(1, 64)) for _ in range(200)]
        for _ in range(60):
            r, k = rng.randint(2, 2 ** rng.randint(1, 80)), rng.randint(2, 30)
            cases.extend((r ** k + delta, k) for delta in (-1, 0, 1))
        for m in range(8, 300, 37):
            for k in (m - 1, m, m + 1, m + 2):
                cases.extend((2 ** m + delta, k) for delta in (-1, 0, 1))
        cases.extend((2 ** 300 + 1, k) for k in (2, 3, 149, 150, 151, 299, 300, 301, 10 ** 18))
        self.assertGreaterEqual(len(cases), 300)  # Seeded, beyond the exhaustive range.
        for n, k in cases:
            expected = root_by_bisection(n, min(k, n.bit_length() + 1))  # Any k beyond bits(n) has the same root.
            with self.subTest(n=n, k=k):
                if k == 2:
                    self.assertEqual(isqrt(n), expected)
                self.assertEqual(self.assertAgrees("integer_nth_root", {"n": n, "k": k}, expected), ())
