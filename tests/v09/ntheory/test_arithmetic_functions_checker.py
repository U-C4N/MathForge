"""The independent checker of mathforge.areas.ntheory.arithmetic_functions (moebius, divisor_sigma, integer_nth_root),
tested on its own, before the family's producer exists.

Every true claim is computed here: by brute force over small integers, by bisection, Newton's iteration or math.isqrt,
or by a short reference over a factorization that the test builds from primes it knows (the Mersenne primes
2**89 - 1, 2**107 - 1 and 2**127 - 1 among them, above psi_13), never by the family's producer. The evidence has the
layout of docs/v09/cards/m1-ntheory.md; prime certificates and certified factorizations come from
mathforge.kernels.certificates, the producer-side builder of what mathforge.certify.ntheory.factorization accepts.

The checker accepts every true claim, also with evidence that a producer would not choose, and answers False, never
raising, to every corruption. The family's values are plain ints, so its impostor and forged records are those of
another family, int subclasses and tuple subclasses where an int or a tuple belongs. Oversized forged evidence and
values are refused fast under the default budget. ResourceLimitError, which is never a verdict, answers only a request
integer over max_integer_bits, divisor_sigma beyond its cap k * bits(n) <= max_integer_bits, and a budget too small
for the proof.
"""

from __future__ import annotations

import enum
from fractions import Fraction
import functools
from math import isqrt, prod
import random
import time
from types import MappingProxyType
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import registry
from mathforge.areas.ntheory import arithmetic_functions as family
from mathforge.areas.ntheory.continued_fraction import ContinuedFraction
from mathforge.kernels.certificates import factorization_evidence, prime_evidence
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13

from .. import support

FAST = 3.0  # Seconds: a forgery is refused well within this under the default budget.
M89, M107, M127 = 2 ** 89 - 1, 2 ** 107 - 1, 2 ** 127 - 1  # Primes above psi_13: their proofs take certificates.
MIDDLE = (10007, 1000003, 2 ** 61 - 1)  # Primes from 10,000 to psi_13, which the strong tests prove with ().
PSEUDOPRIME = 3215031751  # 151 * 751 * 28351, a strong pseudoprime to the bases 2, 3, 5 and 7.
F6 = ((2, 1, ()), (3, 1, ()))
F12 = ((2, 2, ()), (3, 1, ()))
F30 = ((2, 1, ()), (3, 1, ()), (5, 1, ()))


class Integer(int):
    """An int subclass: equal to its int, but not an exact int."""


class Sign(enum.IntEnum):
    """An IntEnum: its members equal -1, 0 and 1, but are no exact ints."""
    MINUS, ZERO, PLUS = -1, 0, 1


class Text(str):
    """A str subclass, for an evidence tag."""


class Row(tuple):
    """A tuple subclass, for evidence."""


@functools.cache
def certificate(p: int) -> tuple:
    """The certificate that kernels.certificates builds for the prime p: () below psi_13, Pocklington's above."""
    return prime_evidence(p)


def trial_factors(n: int) -> tuple[tuple[int, int], ...]:
    """The prime factorization ((p, e), ...) of n >= 1 by trial division: the tests' own reference."""
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


def certified(factors: tuple[tuple[int, int], ...]) -> tuple:
    """((p, e, certificate), ...) for ((p, e), ...): the certified factorization of the card's evidence."""
    return tuple((p, e, certificate(p)) for p, e in factors)


def moebius_reference(n: int) -> int:
    """mu(n) by its definition on the trial factorization."""
    factors = trial_factors(n)
    return 0 if any(e > 1 for _, e in factors) else (-1) ** len(factors)


def moebius_evidence(n: int) -> tuple:
    """The card's evidence for mu(n): ("square", p) for the least prime p with p*p | n, else ("factorization", ...)."""
    factors = trial_factors(n)
    for p, e in factors:
        if e > 1:
            return ("square", p)
    return ("factorization", certified(factors))


def sigma_brute(n: int, k: int) -> int:
    """sigma_k(n) by its definition: the sum of d**k over the positive divisors d of n."""
    return sum(d ** k for d in range(1, n + 1) if n % d == 0)


def sigma_reference(factors: tuple[tuple[int, int], ...], k: int) -> int:
    """sigma_k(prod p**e), the divisors of each prime power summed: prod over p of (sum of (p**i)**k, i = 0 ... e)."""
    return prod(sum((p ** i) ** k for i in range(e + 1)) for p, e in factors)


def root_brute(n: int, k: int) -> int:
    """floor(n**(1/k)) by counting up."""
    r = 0
    while (r + 1) ** k <= n:
        r += 1
    return r


def root_reference(n: int, k: int) -> int:
    """floor(n**(1/k)) by bisection on [0, 2**(bits(n)//k + 1)], whose upper end has a k-th power above n."""
    low, high = 0, 1 << (n.bit_length() // k + 1)
    while high - low > 1:
        middle = (low + high) // 2
        if middle ** k <= n:
            low = middle
        else:
            high = middle
    return low


def root_newton(n: int, k: int) -> int:
    """floor(n**(1/k)) for n >= 1 and k >= 2 by Newton's iteration on integers from 2**ceil(bits(n)/k), which lies
    above the root: a step from above stays at or above the root (the k-th root of n is the geometric mean of k - 1
    copies of x and n/x**(k - 1), at most their arithmetic mean) and decreases, so the first step that does not
    decrease starts from the root. For the long roots, where the bisection takes one power per bit of the root."""
    x = 1 << -(-n.bit_length() // k)
    while True:
        following = ((k - 1) * x + n // x ** (k - 1)) // k
        if following >= x:
            return x
        x = following


class ArithmeticFunctionsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsFast(self, operation: str, args: object, value: object, evidence: object = ()) -> None:
        """The checker answers False, under the default budget and within FAST seconds."""
        start = time.perf_counter()
        self.assertRejects(operation, args, value, evidence)
        self.assertLess(time.perf_counter() - start, FAST)

    def assertExhausts(self, operation: str, args: object, value: object, evidence: object = (), *,
                       limits: ComputationLimits | None = None) -> None:
        """The checker runs out of budget: neither True nor False, but ResourceLimitError."""
        with self.assertRaises(ResourceLimitError):
            support.verdict(self.FAMILY, operation, args, value, evidence, limits=limits)

    # -- true claims ------------------------------------------------------------------------------------------------

    def test_every_operation_of_the_family_reaches_this_checker(self):
        """The registry routes each operation of SPECS to the checker, which takes exactly the spec's arguments."""
        claims = {"moebius": ({"n": 6}, 1, ("factorization", F6)),
                  "divisor_sigma": ({"n": 6, "k": 1}, 12, (F6,)),
                  "integer_nth_root": ({"n": 6, "k": 2}, 2, ())}
        self.assertEqual([spec.name for spec in family.SPECS], list(claims))
        for spec in family.SPECS:
            args, value, evidence = claims[spec.name]
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.checker(spec.name).__name__, family.CHECKER)
                self.assertEqual(tuple(argument.name for argument in spec.arguments), tuple(args))
                self.assertAccepts(spec.name, args, value, evidence)

    def test_known_values_of_the_card(self):
        """Each known value of the card, recomputed here, with the evidence in the card's layout."""
        self.assertEqual(moebius_evidence(1), ("factorization", ()))
        self.assertEqual(moebius_evidence(4), ("square", 2))
        self.assertEqual(moebius_evidence(30), ("factorization", F30))
        for n, expected in ((1, 1), (2, -1), (4, 0), (6, 1), (12, 0), (30, -1)):
            with self.subTest(operation="moebius", n=n):
                self.assertEqual(moebius_reference(n), expected)
                self.assertAccepts("moebius", {"n": n}, expected, moebius_evidence(n))
        for n, k, expected in ((12, 1, 28), (12, 0, 6), (6, 2, 50), (1, 5, 1)):
            with self.subTest(operation="divisor_sigma", n=n, k=k):
                self.assertEqual(sigma_brute(n, k), expected)
                self.assertAccepts("divisor_sigma", {"n": n, "k": k}, expected, (certified(trial_factors(n)),))
        self.assertAccepts("divisor_sigma", {"n": 12, "k": 1}, 28, (F12,))
        self.assertAccepts("divisor_sigma", {"n": 1, "k": 5}, 1, ((),))
        for n, k, expected in ((26, 3, 2), (27, 3, 3), (0, 5, 0), (1, 100, 1), (10 ** 20, 2, 10 ** 10),
                               (2 ** 64 - 1, 2, 2 ** 32 - 1)):
            with self.subTest(operation="integer_nth_root", n=n, k=k):
                self.assertEqual(root_reference(n, k), expected)
                self.assertAccepts("integer_nth_root", {"n": n, "k": k}, expected, ())

    def test_moebius_by_brute_force(self):
        """mu(n) for n = 1 ... 1500 is accepted with the card's evidence, the two other values are refused."""
        for n in range(1, 1501):
            expected, evidence = moebius_reference(n), moebius_evidence(n)
            with self.subTest(n=n):
                self.assertAccepts("moebius", {"n": n}, expected, evidence)
                for other in {-1, 0, 1} - {expected}:
                    self.assertRejects("moebius", {"n": n}, other, evidence)

    def test_divisor_sigma_by_brute_force(self):
        """sigma_k(n) for n = 1 ... 240 and k = 0 ... 5 is accepted, and its neighbours are refused."""
        for n in range(1, 241):
            evidence = (certified(trial_factors(n)),)
            for k in range(6):
                expected = sigma_brute(n, k)
                with self.subTest(n=n, k=k):
                    self.assertEqual(sigma_reference(trial_factors(n), k), expected)
                    self.assertAccepts("divisor_sigma", {"n": n, "k": k}, expected, evidence)
                    self.assertRejects("divisor_sigma", {"n": n, "k": k}, expected + 1, evidence)
                    self.assertRejects("divisor_sigma", {"n": n, "k": k}, expected - 1, evidence)

    def test_integer_nth_root_by_brute_force(self):
        """floor(n**(1/k)) for n = 0 ... 400 and k = 1 ... 9 is accepted, and its neighbours are refused."""
        for n in range(401):
            for k in range(1, 10):
                expected = root_brute(n, k)
                with self.subTest(n=n, k=k):
                    self.assertEqual(root_reference(n, k), expected)
                    self.assertAccepts("integer_nth_root", {"n": n, "k": k}, expected)
                    self.assertRejects("integer_nth_root", {"n": n, "k": k}, expected + 1)
                    if expected:
                        self.assertRejects("integer_nth_root", {"n": n, "k": k}, expected - 1)

    def test_integer_nth_root_on_large_and_boundary_inputs(self):
        """Seeded integers of up to 4096 bits, the neighbours of perfect powers and of powers of 2 (where the
        checker's bit-length screens switch), each against the bisection, math.isqrt for k = 2, and the neighbours."""
        rng = random.Random(20261112)
        cases = []
        for _ in range(250):
            cases.append((rng.getrandbits(rng.randint(1, 4096)), rng.randint(1, 80)))
        for _ in range(150):
            r, k = rng.randint(2, 2 ** rng.randint(1, 64)), rng.randint(1, 40)
            cases.extend((r ** k + delta, k) for delta in (-1, 0, 1))
        for m in range(1, 140):
            for k in (1, 2, 3, 5, 7, m - 1, m, m + 1):
                if k >= 1:
                    cases.extend((2 ** m + delta, k) for delta in (-1, 0, 1))
        for n, k in cases:
            expected = root_reference(n, k)
            with self.subTest(n=n, k=k):
                if k == 2:
                    self.assertEqual(isqrt(n), expected)
                self.assertTrue(expected ** k <= n < (expected + 1) ** k)
                self.assertAccepts("integer_nth_root", {"n": n, "k": k}, expected)
                self.assertRejects("integer_nth_root", {"n": n, "k": k}, expected + 1)
                if expected:
                    self.assertRejects("integer_nth_root", {"n": n, "k": k}, expected - 1)

    def test_integer_nth_root_at_the_integer_limit(self):
        """n of 65,536 bits: roots for k = 2, 3 and 65,000 (whose powers are the largest that the checker forms) are
        accepted, and their neighbours refused, each fast. The roots come from math.isqrt, Newton's iteration and the
        bisection, each confirmed here by r**k <= n < (r + 1)**k (a bisection over a root of 32,768 or 21,846 bits
        takes seconds)."""
        n = 2 ** 65535 + 12345
        for k, expected in ((2, isqrt(n)), (3, root_newton(n, 3)), (65_000, root_reference(n, 65_000))):
            with self.subTest(k=k):
                self.assertTrue(expected ** k <= n < (expected + 1) ** k)
                start = time.perf_counter()
                self.assertAccepts("integer_nth_root", {"n": n, "k": k}, expected)
                self.assertRejects("integer_nth_root", {"n": n, "k": k}, expected + 1)
                self.assertRejects("integer_nth_root", {"n": n, "k": k}, expected - 1)
                self.assertLess(time.perf_counter() - start, FAST)
        self.assertEqual(root_reference(n, 65_000), 2)
        self.assertAccepts("integer_nth_root", {"n": 5, "k": 10 ** 18}, 1)  # (r + 1)**k by its length alone.
        self.assertAccepts("integer_nth_root", {"n": 0, "k": 10 ** 18}, 0)

    def test_large_primes_with_certificates(self):
        """Primes from 10,000 to psi_13 (certificate ()) and above it (Pocklington certificates), alone and in
        products; the values come from the factorization that the test builds."""
        for p in (*MIDDLE, M89, M107, M127):
            factors = ((p, 1, certificate(p)),)
            with self.subTest(p=p):
                self.assertEqual(certificate(p) == (), p < PSI_13)
                self.assertAccepts("moebius", {"n": p}, -1, ("factorization", factors))
                self.assertAccepts("divisor_sigma", {"n": p, "k": 0}, 2, (factors,))
                self.assertAccepts("divisor_sigma", {"n": p, "k": 1}, p + 1, (factors,))
                self.assertAccepts("divisor_sigma", {"n": p, "k": 3}, 1 + p ** 3, (factors,))
        products = (((M89, 1), (M107, 1)), ((2, 5), (3, 1), (M127, 1)), ((10007, 1), (M89, 2)),
                    ((3, 1), (1000003, 1), (M89, 1), (M107, 1), (M127, 1)), ((2, 64),))
        for factors in products:
            n = prod(p ** e for p, e in factors)
            evidence = certified(factors)
            mu = 0 if any(e > 1 for _, e in factors) else (-1) ** len(factors)
            with self.subTest(n=n):
                if mu:
                    self.assertAccepts("moebius", {"n": n}, mu, ("factorization", evidence))
                else:
                    square = next(p for p, e in factors if e > 1)
                    self.assertAccepts("moebius", {"n": n}, 0, ("square", square))
                for k in (0, 1, 2, 7):
                    self.assertAccepts("divisor_sigma", {"n": n, "k": k}, sigma_reference(factors, k), (evidence,))

    def test_evidence_from_kernels_certificates(self):
        """factorization_evidence, which the producer will call, builds what the checker accepts."""
        for n in (1, 2, 360, 2 ** 64 - 1, 10 ** 18 + 3, M89 * 6, 600851475143, 99991 ** 3 * 65537):
            factors = factorization_evidence(n)
            plain = tuple((p, e) for p, e, _ in factors)
            with self.subTest(n=n):
                self.assertEqual(prod(p ** e for p, e in plain), n)
                self.assertAccepts("divisor_sigma", {"n": n, "k": 1}, sigma_reference(plain, 1), (factors,))
                if all(e == 1 for _, e in plain):
                    self.assertAccepts("moebius", {"n": n}, (-1) ** len(plain), ("factorization", factors))

    def test_evidence_that_a_producer_would_not_choose(self):
        """mu(n) = 0 is proved by every s > 1 with s*s | n, prime or not, small or large."""
        for n in (4, 72, 3600, 2 ** 10 * 3 ** 4, 10 ** 6):
            witnesses = [s for s in range(2, isqrt(n) + 1) if n % (s * s) == 0]
            self.assertGreater(len(witnesses), 0)
            for s in witnesses:
                with self.subTest(n=n, s=s):
                    self.assertAccepts("moebius", {"n": n}, 0, ("square", s))
        self.assertAccepts("moebius", {"n": 3 * M89 ** 2}, 0, ("square", M89))
        self.assertAccepts("moebius", {"n": 2 ** 65535}, 0, ("square", 2 ** 32767))  # s*s divides n exactly.
        self.assertAccepts("moebius", {"n": 2 ** 65535}, 0, ("square", 2))

    def test_divisor_sigma_cap(self):
        """k * bits(n) <= max_integer_bits is accepted; one step beyond, a true claim exhausts the budget as the
        producer's cap does, and wider limits verify it. Wrong evidence and wrong-sized values stay False there."""
        n, k = 2 ** 15, 4096  # k * bits(n) = 65,536: exactly the cap.
        self.assertAccepts("divisor_sigma", {"n": n, "k": k}, sigma_brute(n, k), (((2, 15, ()),),))
        self.assertAccepts("divisor_sigma", {"n": 1, "k": 65_536}, 1, ((),))
        n, k = 2 ** 10, 6000  # k * bits(n) = 66,000.
        value = sigma_brute(n, k)
        self.assertEqual(value.bit_length(), 60_001)
        self.assertExhausts("divisor_sigma", {"n": n, "k": k}, value, (((2, 10, ()),),))
        self.assertExhausts("divisor_sigma", {"n": n, "k": k}, value + 1, (((2, 10, ()),),))  # Undecided, not False.
        self.assertExhausts("divisor_sigma", {"n": 1, "k": 65_537}, 1, ((),))
        self.assertAccepts("divisor_sigma", {"n": n, "k": k}, value, (((2, 10, ()),),),
                           limits=ComputationLimits(max_integer_bits=66_000))
        self.assertRejectsFast("divisor_sigma", {"n": n, "k": k}, value, (((2, 9, ()),),))
        self.assertRejectsFast("divisor_sigma", {"n": n, "k": k}, 2 ** 70_000, (((2, 10, ()),),))
        self.assertRejectsFast("divisor_sigma", {"n": n, "k": k}, 2 ** 59_999, (((2, 10, ()),),))

    def test_the_checker_never_runs_the_producer(self):
        """The checker answers while produce() raises."""
        with patch.object(family, "produce", side_effect=AssertionError("the checker ran the producer")):
            self.assertAccepts("moebius", {"n": 30}, -1, ("factorization", F30))
            self.assertAccepts("divisor_sigma", {"n": 12, "k": 1}, 28, (F12,))
            self.assertAccepts("integer_nth_root", {"n": 26, "k": 3}, 2)

    # -- corrupted claims -------------------------------------------------------------------------------------------

    def test_rejects_wrong_values(self):
        for value in (-1, 0, 2, -2):
            self.assertRejects("moebius", {"n": 6}, value, ("factorization", F6))
        for value in (-1, 1):
            self.assertRejects("moebius", {"n": 12}, value, ("square", 2))
        for value in (27, 29, 0, -28, 1, 56):
            self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, value, (F12,))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 2}, 28, (F12,))
        self.assertRejects("divisor_sigma", {"n": M89, "k": 1}, M89, (((M89, 1, certificate(M89)),),))
        for n, k, value in ((26, 3, 3), (26, 3, 1), (27, 3, 2), (27, 3, 4), (0, 5, 1), (1, 100, 0), (1, 100, 2),
                            (10 ** 20, 2, 10 ** 10 - 1), (10 ** 20, 2, 10 ** 10 + 1), (2 ** 64 - 1, 2, 2 ** 32),
                            (26, 3, -2), (26, 1, 25), (26, 1, 27)):
            with self.subTest(n=n, k=k, value=value):
                self.assertRejects("integer_nth_root", {"n": n, "k": k}, value)

    def test_rejects_values_that_are_not_exact_ints(self):
        """Values that equal the right int in Python, or name it otherwise, are no exact int: a bool, a float, a
        complex, a Fraction, a Rational, an int subclass, an IntEnum, a str, a container, None."""
        claims = (("moebius", {"n": 6}, ("factorization", F6),
                   (True, 1.0, complex(1), Fraction(1), mf.Rational(1), Integer(1), Sign.PLUS, "1", (1,), [1], None)),
                  ("moebius", {"n": 4}, ("square", 2),
                   (False, 0.0, -0.0, Fraction(0), mf.Rational(0), Integer(0), Sign.ZERO, "0", None)),
                  ("moebius", {"n": 30}, ("factorization", F30), (-1.0, Integer(-1), Sign.MINUS, mf.Rational(-1))),
                  ("divisor_sigma", {"n": 12, "k": 1}, (F12,),
                   (28.0, complex(28), Fraction(28), mf.Rational(28), Integer(28), "28", (28,), [28], None)),
                  ("divisor_sigma", {"n": 1, "k": 5}, ((),), (True, 1.0, Sign.PLUS)),
                  ("integer_nth_root", {"n": 26, "k": 3}, (), (2.0, Fraction(2), mf.Rational(2), Integer(2), "2")),
                  ("integer_nth_root", {"n": 1, "k": 100}, (), (True, 1.0, Sign.PLUS)),
                  ("integer_nth_root", {"n": 0, "k": 5}, (), (False, 0.0, Sign.ZERO, None)))
        for operation, args, evidence, values in claims:
            for value in values:
                with self.subTest(operation=operation, value=value):
                    self.assertRejects(operation, args, value, evidence)

    def test_rejects_records_where_an_int_belongs(self):
        """A record, its impostor (support.impostor) or a forged record (support.forged) is no int; the family has
        no record class of its own, so those of continued_fraction stand in."""
        for value in (ContinuedFraction((28,)), support.impostor(ContinuedFraction((28,))),
                      support.forged(ContinuedFraction, terms=(28,)), support.forged(ContinuedFraction, terms=28),
                      support.forged(ContinuedFraction)):
            with self.subTest(value=value):
                self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, value, (F12,))
                self.assertRejects("integer_nth_root", {"n": 28, "k": 1}, value)
                self.assertRejects("moebius", {"n": 6}, value, ("factorization", F6))

    def test_rejects_non_ints_in_the_request(self):
        """A bool, a float, a str, None, a Rational or an int subclass where the request holds an int: True == 1
        would otherwise make moebius(True) = 1 hold."""
        for n in (True, 1.0, "1", None, mf.Rational(1), Integer(1), Sign.PLUS):
            with self.subTest(n=n):
                self.assertRejects("moebius", {"n": n}, 1, ("factorization", ()))
                self.assertRejects("divisor_sigma", {"n": n, "k": 1}, 1, ((),))
                self.assertRejects("integer_nth_root", {"n": n, "k": 1}, 1)
        for k in (True, 1.0, "1", None, mf.Rational(1), Integer(1), Sign.PLUS):
            with self.subTest(k=k):
                self.assertRejects("divisor_sigma", {"n": 12, "k": k}, 28, (F12,))
                self.assertRejects("integer_nth_root", {"n": 26, "k": k}, 26)
        self.assertRejects("integer_nth_root", {"n": 26.0, "k": 3}, 2)
        self.assertRejects("integer_nth_root", {"n": 26, "k": 3.0}, 2)

    def test_rejects_non_ints_in_the_evidence(self):
        """A bool, a float, a Rational, a str or an int subclass in an int slot of the evidence, and a tag that is
        not exactly a str, the tag of a Pocklington certificate included. 4 % (2.0 * 2.0) == 0 and True == 1 would
        otherwise hold."""
        for s in (2.0, True, mf.Rational(2), Fraction(2), "2", Integer(2), None, (2,)):
            with self.subTest(s=s):
                self.assertRejects("moebius", {"n": 4}, 0, ("square", s))
        corrupted = (((2, 1, ()), (3, True, ())), ((2.0, 1, ()), (3, 1, ())), ((2, 1, ()), (3, 1.0, ())),
                     ((2, 1, None), (3, 1, ())), ((Integer(2), 1, ()), (3, 1, ())), ((2, Integer(1), ()), (3, 1, ())),
                     ((True, 1, ()), (3, 1, ())), ((2, 1, ()), ("3", 1, ())))
        for factors in corrupted:
            with self.subTest(factors=factors):
                self.assertRejects("moebius", {"n": 6}, 1, ("factorization", factors))
                self.assertRejects("divisor_sigma", {"n": 6, "k": 1}, 12, (factors,))
        for factors in (((2, 2.0, ()), (3, 1, ())), ((2, 2, ()), (3, True, ())), ((2, Integer(2), ()), (3, 1, ()))):
            with self.subTest(factors=factors):
                self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 28, (factors,))
        for tag in (Text("square"), b"square", None, 0):
            with self.subTest(tag=tag):
                self.assertRejects("moebius", {"n": 4}, 0, (tag, 2))
        self.assertRejects("moebius", {"n": 6}, 1, (Text("factorization"), F6))
        # certify.ntheory.factorization reads the tag of M89's certificate: it accepted this one before.
        forged = (Text("pocklington"),) + certificate(M89)[1:]
        self.assertAccepts("moebius", {"n": M89}, -1, ("factorization", ((M89, 1, certificate(M89)),)))
        self.assertRejects("moebius", {"n": M89}, -1, ("factorization", ((M89, 1, forged),)))
        self.assertRejects("divisor_sigma", {"n": M89, "k": 1}, M89 + 1, (((M89, 1, forged),),))

    def test_rejects_lists_and_tuple_subclasses_for_tuples(self):
        self.assertRejects("moebius", {"n": 4}, 0, ["square", 2])
        self.assertRejects("moebius", {"n": 4}, 0, Row(("square", 2)))
        for evidence in (["factorization", F6], ("factorization", list(F6)), ("factorization", ([2, 1, ()], F6[1])),
                         ("factorization", ((2, 1, []), F6[1])), ("factorization", Row(F6)),
                         ("factorization", (Row((2, 1, ())), F6[1]))):
            with self.subTest(evidence=evidence):
                self.assertRejects("moebius", {"n": 6}, 1, evidence)
        self.assertRejects("moebius", {"n": 1}, 1, ("factorization", []))
        for evidence in ([F12], (list(F12),), (([2, 2, ()], F12[1]),), (((2, 2, []), F12[1]),), Row((F12,)),
                         (Row(F12),)):
            with self.subTest(evidence=evidence):
                self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 28, evidence)
        self.assertRejects("divisor_sigma", {"n": 1, "k": 1}, 1, ([],))
        for evidence in ([], Row(()), None):
            with self.subTest(evidence=evidence):
                self.assertRejects("integer_nth_root", {"n": 26, "k": 3}, 2, evidence)
        for args in ([("n", 6)], (("n", 6),), MappingProxyType({"n": 6}), None):
            with self.subTest(args=args):
                self.assertRejects("moebius", args, 1, ("factorization", F6))

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        for evidence in ((), ("factorization",), ("factorization", F6, ()), (("factorization", F6),), F6,
                         ("factorisation", F6), ("square", F6), ("prime", F6), "factorization", None,
                         ("factorization", None)):
            with self.subTest(evidence=evidence):
                self.assertRejects("moebius", {"n": 6}, 1, evidence)
        for evidence in ((), ("square",), ("square", 2, 2), ("Square", 2), ("square", (2,)), (("square", 2),), "square",
                         ("factorization", 2)):
            with self.subTest(evidence=evidence):
                self.assertRejects("moebius", {"n": 4}, 0, evidence)
        for evidence in ((), (F12, ()), F12, ((),), (F12 + ((5, 1, ()),),), (F12[:1],), None, ("factorization", F12)):
            with self.subTest(evidence=evidence):
                self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 28, evidence)
        for evidence in ((0,), ((),), ("root",), (2,), None, "", ("square", 2)):
            with self.subTest(evidence=evidence):
                self.assertRejects("integer_nth_root", {"n": 26, "k": 3}, 2, evidence)

    def test_rejects_forged_certificates_and_factorizations(self):
        m89, m107 = (M89, 1, certificate(M89)), (M107, 1, certificate(M107))
        # A certificate of another number, an empty certificate where one is needed, one where none is.
        self.assertRejects("moebius", {"n": M107}, -1, ("factorization", ((M107, 1, certificate(M89)),)))
        self.assertRejects("divisor_sigma", {"n": M89 * M107, "k": 1}, (M89 + 1) * (M107 + 1),
                           (((M89, 1, certificate(M107)), (M107, 1, certificate(M89))),))
        self.assertRejects("moebius", {"n": M89}, -1, ("factorization", ((M89, 1, ()),)))
        self.assertRejects("moebius", {"n": 7}, -1, ("factorization", ((7, 1, ("trial", 7)),)))
        self.assertRejects("moebius", {"n": 1000003}, -1, ("factorization", ((1000003, 1, certificate(M89)),)))
        self.assertAccepts("moebius", {"n": M89 * M107}, 1, ("factorization", (m89, m107)))
        # A composite claimed prime, with the right value: only the evidence lies.
        self.assertEqual(moebius_reference(PSEUDOPRIME), -1)
        self.assertAccepts("moebius", {"n": PSEUDOPRIME}, -1, moebius_evidence(PSEUDOPRIME))
        self.assertRejects("moebius", {"n": PSEUDOPRIME}, -1, ("factorization", ((PSEUDOPRIME, 1, ()),)))
        self.assertRejects("moebius", {"n": 105}, -1, ("factorization", ((105, 1, ()),)))
        self.assertRejects("divisor_sigma", {"n": PSEUDOPRIME, "k": 1}, PSEUDOPRIME + 1, (((PSEUDOPRIME, 1, ()),),))
        large = M89 * M107 * M127
        relabeled = ("pocklington", large) + certificate(M89)[2:]
        self.assertRejects("moebius", {"n": large}, -1, ("factorization", ((large, 1, relabeled),)))
        self.assertRejects("moebius", {"n": large}, -1, ("factorization", ((large, 1, certificate(M89)),)))
        # Incomplete factorizations, composite cofactors, wrong order, repeated primes, zero exponents.
        self.assertRejects("moebius", {"n": 30}, 1, ("factorization", F30[:2]))
        self.assertRejects("moebius", {"n": 30}, -1, ("factorization", F30[:1] + F30[2:] + ((3, 1, ()),)))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 7, (F12[:1],))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 12, (F6,))  # sigma(6) = 12 passes the length screen.
        self.assertRejects("divisor_sigma", {"n": 360, "k": 1}, 1170, (certified(trial_factors(2 ** 3 * 3 ** 2 * 7)),))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 21, (((2, 1, ()), (6, 1, ())),))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 20, (((3, 1, ()), (4, 1, ())),))
        self.assertRejects("moebius", {"n": 12}, 1, ("factorization", ((2, 1, ()), (6, 1, ()))))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 28, (((3, 1, ()), (2, 2, ())),))
        self.assertRejects("moebius", {"n": 6}, 1, ("factorization", ((3, 1, ()), (2, 1, ()))))
        self.assertRejects("moebius", {"n": 12}, -1, ("factorization", ((2, 1, ()), (2, 1, ()), (3, 1, ()))))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 36, (((2, 1, ()), (2, 1, ()), (3, 1, ())),))
        self.assertRejects("divisor_sigma", {"n": 12, "k": 1}, 28, (F12 + ((5, 0, ()),),))
        self.assertRejects("moebius", {"n": 6}, 1, ("factorization", ((1, 1, ()),) + F6))
        # mu = 0 takes the square evidence: a factorization with an exponent 2 is not the card's form.
        self.assertRejects("moebius", {"n": 12}, 0, ("factorization", F12))
        # Square evidence: s*s must divide n, s > 1, and the value is 0.
        for s in (3, 4, 5, 1, 0, -2, -1):
            with self.subTest(s=s):
                self.assertRejects("moebius", {"n": 12}, 0, ("square", s))
        self.assertRejects("moebius", {"n": 12}, 1, ("square", 2))
        self.assertRejects("moebius", {"n": 6}, 1, ("square", 2))
        self.assertRejects("moebius", {"n": 1}, 0, ("square", 2))

    def test_rejects_pseudoprimes_claimed_prime(self):
        """Composites that Fermat or strong probable-prime tests pass, each claimed prime with the certificate () of
        its size: Carmichael numbers above 10,000, a base-2 pseudoprime with a prime factor above 2**10, the strong
        pseudoprimes psi_5 ... psi_12 to the first prime bases (psi_12 passes the twelve bases 2 ... 37 and fails 41),
        and psi_13, which passes all thirteen bases and from which on a prime needs a Pocklington certificate. Each such
        claim is refused, and the true claims on the real factorization are accepted."""
        cases = {15841: (7, 31, 73), 41041: (7, 11, 13, 41), 62745: (3, 5, 47, 89), 121465: (5, 17, 1429),
                 2152302898747: (6763, 10627, 29947), 3474749660383: (1303, 16927, 157543),
                 341550071728321: (10670053, 32010157), 3825123056546413051: (149491, 747451, 34233211),
                 318665857834031151167461: (399165290221, 798330580441), PSI_13: (1287836182261, 2575672364521)}
        for c, primes in cases.items():
            factors = certified(tuple((p, 1) for p in primes))
            with self.subTest(c=c):
                self.assertEqual(prod(primes), c)
                self.assertEqual(pow(2, c - 1, c), 1)  # A base-2 Fermat pseudoprime.
                self.assertAccepts("moebius", {"n": c}, (-1) ** len(primes), ("factorization", factors))
                self.assertAccepts("divisor_sigma", {"n": c, "k": 1}, prod(p + 1 for p in primes), (factors,))
                self.assertRejects("moebius", {"n": c}, -1, ("factorization", ((c, 1, ()),)))
                self.assertRejects("divisor_sigma", {"n": c, "k": 1}, c + 1, (((c, 1, ()),),))
                self.assertRejects("divisor_sigma", {"n": c, "k": 0}, 2, (((c, 1, ()),),))
                self.assertRejects("moebius", {"n": 2 * c}, 1, ("factorization", ((2, 1, ()), (c, 1, ()))))
        self.assertRejects("moebius", {"n": PSI_13}, -1, ("factorization", ((PSI_13, 1, certificate(M89)),)))
        self.assertRejects("moebius", {"n": PSI_13}, -1,
                           ("factorization", ((PSI_13, 1, ("pocklington", PSI_13) + certificate(M89)[2:]),)))

    def test_rejects_claims_outside_the_domain(self):
        """n < 1 for moebius and divisor_sigma, k < 0, n < 0 and k < 1 for integer_nth_root: False even where the
        arithmetic holds (0 % 4 == 0; sigma_-1(1) = 1), and never an exception (0**-1)."""
        self.assertRejects("moebius", {"n": 0}, 0, ("square", 2))
        self.assertRejects("moebius", {"n": -4}, 0, ("square", 2))
        self.assertRejects("moebius", {"n": -1}, 1, ("factorization", ()))
        self.assertRejects("moebius", {"n": -6}, 1, ("factorization", F6))
        self.assertRejects("divisor_sigma", {"n": 0, "k": 1}, 1, ((),))
        self.assertRejects("divisor_sigma", {"n": -12, "k": 1}, 28, (F12,))
        self.assertRejects("divisor_sigma", {"n": 1, "k": -1}, 1, ((),))
        self.assertRejects("divisor_sigma", {"n": 2, "k": -1}, 1, (((2, 1, ()),),))
        for n, k, value in ((8, 0, 1), (8, -1, 0), (8, -1, 1), (0, -3, 0), (-8, 3, -2), (-8, 3, 0), (8, 3, -2)):
            with self.subTest(n=n, k=k, value=value):
                self.assertRejects("integer_nth_root", {"n": n, "k": k}, value)

    def test_rejects_wrong_operation_names_and_arguments(self):
        for operation in ("Moebius", "mobius", "moebius ", "", "continued_fraction", "integer_gcd", None, 5,
                          b"moebius", ("moebius",)):
            with self.subTest(operation=operation):
                self.assertRejects(operation, {"n": 6}, 1, ("factorization", F6))
        self.assertRejects("divisor_sigma", {"n": 6}, 12, (F6,))
        self.assertRejects("divisor_sigma", {"n": 6, "k": 1}, 1, ("factorization", F6))
        self.assertRejects("integer_nth_root", {"n": 26}, 2)
        self.assertRejects("integer_nth_root", {"n": 6, "k": 2}, 1, ("factorization", F6))
        for args in ({}, {"m": 6}, {"n": 6, "k": 1}, {"n": 6, "extra": 0}):
            with self.subTest(args=args):
                self.assertRejects("moebius", args, 1, ("factorization", F6))
        for args in ({"n": 12}, {"k": 1}, {"n": 12, "k": 1, "m": 0}, {"n": 12, "m": 1}):
            with self.subTest(args=args):
                self.assertRejects("divisor_sigma", args, 28, (F12,))
                self.assertRejects("integer_nth_root", args, 12)

    def test_oversized_forgeries_are_refused_fast(self):
        """Huge values, huge s, huge exponents and primes, 100,000 factors, a bloated certificate, a huge k: each
        False within FAST seconds under the default budget, without building the powers they suggest."""
        bloated = ("pocklington", M89, ((2, 1, ("trial", 2)),) * 50_000, (3,) * 50_000)
        increasing = tuple((p, 1, ()) for p in range(2, 100_002))
        endless = ((2, 1, ()),) * 2_500_000  # More entries than max_work could pay for one at a time.
        claims = (("moebius", {"n": 30}, support.HUGE, ("factorization", F30)),
                  ("moebius", {"n": 30}, -support.HUGE, ("factorization", F30)),
                  ("moebius", {"n": 30}, 0, ("square", support.HUGE)),
                  ("moebius", {"n": 2 ** 65535}, 0, ("square", 2 ** 32768)),
                  ("moebius", {"n": 30}, -1, ("factorization", ((2, support.HUGE, ()),) + F30[1:])),
                  ("moebius", {"n": 30}, -1, ("factorization", ((support.HUGE, 1, ()),))),
                  ("moebius", {"n": 30}, -1, ("factorization", ((2, 1, ()),) * 100_000)),
                  ("moebius", {"n": 30}, 1, ("factorization", increasing)),
                  ("moebius", {"n": 2 ** 65535 - 1}, 1, ("factorization", endless)),
                  ("moebius", {"n": M89}, -1, ("factorization", ((M89, 1, bloated),))),
                  ("divisor_sigma", {"n": 12, "k": 1}, support.HUGE, (F12,)),
                  ("divisor_sigma", {"n": 12, "k": 1}, 28, (((2, support.HUGE, ()), (3, 1, ())),)),
                  ("divisor_sigma", {"n": 12, "k": 1}, 28, (((support.HUGE, 1, ()),),)),
                  ("divisor_sigma", {"n": 12, "k": 1}, 28, (increasing,)),
                  ("divisor_sigma", {"n": 2 ** 65535 - 1, "k": 0}, 2, (endless,)),
                  ("divisor_sigma", {"n": 12, "k": 10 ** 6}, support.HUGE, (F12,)),
                  ("divisor_sigma", {"n": 12, "k": 2 ** 65535}, 28, (F12,)),
                  ("divisor_sigma", {"n": M89, "k": 1}, M89 + 1, (((M89, 1, bloated),),)),
                  ("integer_nth_root", {"n": 26, "k": 3}, support.HUGE, ()),
                  ("integer_nth_root", {"n": 2 ** 65535, "k": 2}, 2 ** 32768, ()),
                  ("integer_nth_root", {"n": 10, "k": 10 ** 30}, 2, ()),
                  ("integer_nth_root", {"n": 10, "k": 2 ** 65535}, 2, ()),
                  ("integer_nth_root", {"n": 26, "k": 3}, 2, (0,) * 100_000),
                  ("integer_nth_root", {"n": 26, "k": 3}, 2, endless))
        for operation, args, value, evidence in claims:
            with self.subTest(operation=operation, args=sorted(args)):
                self.assertRejectsFast(operation, args, value, evidence)

    def test_resource_limits_are_never_verdicts(self):
        """A request integer over max_integer_bits, as run() refuses it before any checker, and a budget too small
        for a true claim's proof raise ResourceLimitError: the claim is undecided, neither verified nor refuted."""
        self.assertExhausts("moebius", {"n": support.HUGE}, 0, ("square", 2))
        self.assertExhausts("divisor_sigma", {"n": support.HUGE, "k": 0}, 2, (((2, 65536, ()),),))
        self.assertExhausts("divisor_sigma", {"n": 12, "k": support.HUGE}, 28, (F12,))
        self.assertExhausts("integer_nth_root", {"n": support.HUGE, "k": 2}, 2 ** 32768)
        self.assertExhausts("integer_nth_root", {"n": 26, "k": support.HUGE}, 1)
        self.assertExhausts("moebius", {"n": M89}, -1, ("factorization", ((M89, 1, certificate(M89)),)),
                            limits=ComputationLimits(max_work=50))
        self.assertExhausts("divisor_sigma", {"n": M127, "k": 1}, M127 + 1, (((M127, 1, certificate(M127)),),),
                            limits=ComputationLimits(max_work=50))
        n = 2 ** 65535 + 12345
        self.assertExhausts("integer_nth_root", {"n": n, "k": 65_000}, 2, limits=ComputationLimits(max_work=1000))


if __name__ == "__main__":
    unittest.main()
