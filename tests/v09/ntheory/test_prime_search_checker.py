"""The independent checker of the prime_search family (mathforge.certify.ntheory.prime_search), tested alone.

Every true claim is computed here, never by the producer: Eratosthenes' sieve below SIEVE_LIMIT and, above it, the
strong probable-prime test to the prime bases 2 to 73, whose first thirteen bases already decide every integer below
psi_13; the values from psi_13 on are certified by their Pocklington certificates as well. The witnesses are found
here too, by Fermat bases and by trial division. The prime certificates come from mathforge.kernels.certificates, the
producer-side builder of the evidence that certify.ntheory.factorization checks, and the integer kernel factors
psi_13 - 1 for a forged one. support.verdict puts each claim to the checker as the verifier does, inside a fresh
budget of the default limits unless a test passes others.

Which skipped integers need a witness is the card's rule (docs/v09/cards/m1-ntheory.md), written out here as
needs_witness: m >= 10,000, prime to 2*3*...*97 and passing Fermat's test to base 2. The card names 15841 = 7*31*73,
a Carmichael number, as one that needs a ("factor", ...) witness, but its factor 7 divides 2*3*...*97: the checker
refutes it on its own, so a witness for it is an extra entry and is refused. The base-2 pseudoprimes from 10,000 on
that need a witness have no prime factor up to 97, such as 31621 = 103*307 and the Carmichael number
56052361 = 211*421*631. No prime gap below 1.2 * 10**8 holds two of them, so every true claim here has at most one
witness.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt, prod
import time
import unittest
from unittest.mock import patch

from mathforge import registry
from mathforge.areas.ntheory import prime_search as family
from mathforge.certify.ntheory import prime_search as checker
from mathforge.certify.ntheory.primes import check_prime
from mathforge.kernels import integers as kernel
from mathforge.kernels.certificates import prime_evidence
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.model import Rational
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13

from .. import support

FAST = 2.0  # Seconds: an oversized or far-fetched claim is refused well within this under the default budget.
SIEVE_LIMIT = 100_000
SWEEP = 20_000  # The card: agreement with a sieve for every n below it.
TRIAL_BELOW = 10_000  # The card: below it the trial test decides, and no witness is ever needed.
PSI_12 = 318665857834031151167461  # 399165290221 * 798330580441, a strong pseudoprime to the twelve bases 2 to 37.
CARMICHAEL = 56052361  # 211 * 421 * 631: a Carmichael number without a prime factor up to 97.
# Base-2 pseudoprimes from 10,000 on without a prime factor up to 97, each with a proper divisor: every one needs a
# witness. 60701 has the factor 101, the least prime above 97; psi_12 and psi_13 pass the strong tests to the twelve
# bases 2 to 37, and the thirteen bases 2 to 41.
NEED_WITNESS = ((31621, 103), (60701, 101), (CARMICHAEL, 211), (2 ** 32 + 1, 641), (2 ** 64 + 1, 274177),
                (2 ** 67 - 1, 193707721), (PSI_12, 399165290221), (PSI_13, 1287836182261))
# Base-2 pseudoprimes from 10,000 on that share a factor with 2*3*...*97, which the checker refutes by the gcd:
# the Carmichael numbers 15841 = 7*31*73 (the card's example) and 10585 = 5*29*73, and 18721 = 97*193.
SHARE_A_FACTOR = ((15841, 7), (10585, 5), (18721, 97))
# Base-2 pseudoprimes below 10,000, which the trial test refutes: 341 = 11*31, 561 = 3*11*17, 645 = 3*5*43,
# 1729 = 7*13*19 and 2047 = 23*89.
BELOW_TRIAL = ((341, 11), (561, 3), (645, 3), (1729, 7), (2047, 23))


def sieve(limit: int) -> bytearray:
    """flags[n] is 1 exactly for the primes n below limit."""
    flags = bytearray([1]) * limit
    flags[0:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return flags


FLAGS = sieve(SIEVE_LIMIT)
PRIMORIAL = prod(p for p in range(98) if FLAGS[p])  # 2*3*...*97, the 25 primes up to 97.
# The first thirteen decide every n below psi_13 (OEIS A014233); psi_13 itself passes them and fails base 43.
_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73)


def is_prime(n: int) -> bool:
    """Whether n is prime: the sieve below SIEVE_LIMIT, the strong probable-prime test to the bases 2 to 73 above."""
    if n < SIEVE_LIMIT:
        return n >= 2 and FLAGS[n] == 1
    if any(n % p == 0 for p in _BASES):
        return False
    d, s = n - 1, 0
    while not d & 1:
        d >>= 1
        s += 1
    for base in _BASES:
        x = pow(base, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime_of(n: int) -> int:
    """The smallest prime greater than n."""
    m = max(n + 1, 2)
    while not is_prime(m):
        m += 1
    return m


def prev_prime_of(n: int) -> int:
    """The largest prime smaller than n >= 3."""
    m = n - 1
    while not is_prime(m):
        m -= 1
    return m


def needs_witness(m: int) -> bool:
    """The card's rule: the skipped integers that the checker cannot refute on its own."""
    return m >= TRIAL_BELOW and gcd(m, PRIMORIAL) == 1 and pow(2, m - 1, m) == 1


def witness_of(m: int) -> tuple:
    """A witness that the composite m is composite: ("fermat", m, w) for the least base w below 200 with
    w**(m - 1) not 1 modulo m, else ("factor", m, d) for its least divisor d > 1 (a Carmichael number)."""
    for base in range(2, min(m, 200)):
        if pow(base, m - 1, m) != 1:
            return ("fermat", m, base)
    for divisor in range(2, min(isqrt(m), 10 ** 6) + 1):
        if m % divisor == 0:
            return ("factor", m, divisor)
    raise AssertionError(f"no witness found for {m}")


def witnesses_for(low: int, high: int) -> tuple:
    """The card's witnesses for the integers low <= m < high: one for each composite m that needs one, in order.
    A prime has none: a claim that skips one is wrong."""
    return tuple(witness_of(m) for m in range(low, high) if needs_witness(m) and not is_prime(m))


def forged_pocklington(n: int) -> tuple:
    """The Pocklington certificate that comes closest to proving the composite n prime: n - 1 factored into primes
    with their v0.7 certificates, and for each prime the first witness below 300 that meets Pocklington's two
    conditions for it, else 2. By Pocklington's theorem some prime of n - 1 has none."""
    factors = tuple((p, e, kernel.prime_certificate(p)[1]) for p, e in kernel.factor(n - 1))
    witnesses = tuple(next((w for w in range(2, 300) if pow(w, n - 1, n) == 1
                            and gcd(pow(w, (n - 1) // p, n) - 1, n) == 1), 2) for p, _, _ in factors)
    return ("pocklington", n, factors, witnesses)


def claim(operation: str, n: int) -> tuple[int, tuple]:
    """The true claim of an operation at n, computed with the references above: (value, evidence)."""
    if operation == "next_prime":
        if n < 2:
            return 2, ((), ())
        value = next_prime_of(n)
        low, high = n + 1, value
    else:
        value = prev_prime_of(n)
        low, high = value + 1, n
    return value, (prime_evidence(value), witnesses_for(low, high))


@dataclass(frozen=True)
class Holder:
    """A record that holds the right integer, where every value of the family is a plain int."""
    value: int


class Integer(int):
    """An int subclass, which no decoded document holds: not an exact int for the checker."""


class PrimeSearchCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertFastRejects(self, operation: str, args: object, value: object, evidence: object, *,
                          may_exhaust: bool = False) -> None:
        """assertRejects, within FAST seconds."""
        start = time.perf_counter()
        self.assertRejects(operation, args, value, evidence, may_exhaust=may_exhaust)
        self.assertLess(time.perf_counter() - start, FAST)

    def test_known_values_of_the_card(self):
        for operation, n, expected in (("next_prime", -5, 2), ("next_prime", 1, 2), ("next_prime", 2, 3),
                                       ("next_prime", 13, 17), ("next_prime", 89, 97), ("prev_prime", 3, 2),
                                       ("prev_prime", 18, 17), ("prev_prime", 100, 97)):
            with self.subTest(operation=operation, n=n):
                value, evidence = claim(operation, n)
                self.assertEqual((value, evidence), (expected, ((), ())))
                self.assertAccepts(operation, {"n": n}, value, evidence)

    def test_next_prime_below_two_is_two_and_enumerates_nothing(self):
        for label, n in (("1", 1), ("0", 0), ("-1", -1), ("-5", -5), ("-10**30", -10 ** 30),
                         ("-2**65535", -(2 ** 65535))):
            with self.subTest(n=label):  # A label: unittest cannot write an integer of 65,536 bits in a report.
                self.assertAccepts("next_prime", {"n": n}, 2, ((), ()))
                for value, evidence in ((3, ((), ())), (1, ((), ())), (2, ((), (("factor", 4, 2),))),
                                        (2, (("trial", 2), ())), (2, ((),)), (2, ())):
                    self.assertRejects("next_prime", {"n": n}, value, evidence)

    def test_the_registry_serves_this_checker(self):
        """The family's specs build into the registry, and both operations reach this checker."""
        for name, summary in (("next_prime", "Find the smallest prime greater than an integer."),
                              ("prev_prime", "Find the largest prime smaller than an integer.")):
            with self.subTest(name=name):
                spec = registry.spec(name)
                self.assertEqual((spec.area, spec.arguments, spec.returns, spec.summary, spec.version),
                                 ("ntheory", (registry.Argument("n", "int", None, "int"),), "int", summary, 6))
                self.assertIs(registry.checker(name), checker)

    def test_agrees_with_the_sieve_below_20000(self):
        """Every true claim for n below SWEEP is accepted, and every claim that skips its true value, a prime, is
        refused."""
        for operation, first in (("next_prime", -3), ("prev_prime", 3)):
            for n in range(first, SWEEP):
                value, evidence = claim(operation, n)
                self.assertIs(support.verdict(family, operation, {"n": n}, value, evidence), True, (operation, n))
                if operation == "next_prime" and n >= 2:
                    later = next_prime_of(value)
                    wrong = (later, (prime_evidence(later), witnesses_for(n + 1, later)))
                elif operation == "prev_prime" and value > 2:
                    earlier = prev_prime_of(value)
                    wrong = (earlier, (prime_evidence(earlier), witnesses_for(earlier + 1, n)))
                else:
                    continue
                self.assertIs(support.verdict(family, operation, {"n": n}, *wrong), False, (operation, n))

    def test_skipping_a_small_prime_is_refused_whatever_its_gcd_with_the_primorial(self):
        """gcd(3, 2*3*...*97) = 3, yet 3 is prime: below 10,000 only the trial test decides."""
        for operation, n, value in (("next_prime", 2, 5), ("next_prime", 89, 101), ("next_prime", 1, 3),
                                    ("prev_prime", 100, 89), ("prev_prime", 5, 2), ("prev_prime", 9990, 9967),
                                    ("next_prime", 9967, 10007)):
            with self.subTest(operation=operation, n=n, value=value):
                self.assertRejects(operation, {"n": n}, value, ((), ()))

    def test_pseudoprimes_below_10000_take_the_trial_test(self):
        for m, divisor in BELOW_TRIAL:
            below, above = prev_prime_of(m), next_prime_of(m)
            with self.subTest(m=m):
                self.assertEqual((m % divisor, pow(2, m - 1, m)), (0, 1))
                self.assertEqual(witnesses_for(below + 1, above), ())
                self.assertAccepts("next_prime", {"n": below}, above, ((), ()))
                self.assertAccepts("prev_prime", {"n": above}, below, ((), ()))
                for witness in (("factor", m, divisor), ("fermat", m, divisor)):
                    self.assertRejects("next_prime", {"n": below}, above, ((), (witness,)))
                    self.assertRejects("prev_prime", {"n": above}, below, ((), (witness,)))

    def test_pseudoprimes_that_share_a_factor_with_the_primorial_need_no_witness(self):
        """15841 = 7*31*73 among them: the card's example of a ("factor", ...) witness is an extra entry."""
        for m, divisor in SHARE_A_FACTOR:
            below, above = prev_prime_of(m), next_prime_of(m)
            with self.subTest(m=m):
                self.assertEqual((m % divisor, divisor <= 97, pow(2, m - 1, m), needs_witness(m)), (0, True, 1, False))
                self.assertAccepts("next_prime", {"n": below}, above, ((), ()))
                self.assertAccepts("prev_prime", {"n": above}, below, ((), ()))
                for witness in (("factor", m, divisor), ("fermat", m, divisor)):
                    self.assertRejects("next_prime", {"n": below}, above, ((), (witness,)))
                    self.assertRejects("prev_prime", {"n": above}, below, ((), (witness,)))

    def test_pseudoprimes_prime_to_the_primorial_need_exactly_one_witness(self):
        for m, divisor in NEED_WITNESS:
            below, above = prev_prime_of(m), next_prime_of(m)
            with self.subTest(m=m):
                self.assertEqual((m % divisor, 1 < divisor < m, needs_witness(m)), (0, True, True))
                found = witness_of(m)
                self.assertEqual(witnesses_for(below + 1, above), (found,))
                certificates = {"next_prime": prime_evidence(above), "prev_prime": prime_evidence(below)}
                claims = (("next_prime", below, above), ("prev_prime", above, below))
                # The found witness, a factor, and a Fermat base that shares a factor with m (the only kind that a
                # Carmichael number has): every kind that verifies is accepted.
                for witness in (found, ("factor", m, divisor), ("factor", m, m // divisor), ("fermat", m, divisor)):
                    for operation, n, value in claims:
                        self.assertAccepts(operation, {"n": n}, value, (certificates[operation], (witness,)))
                # No witness, one that does not verify, a repeated one, or one naming another integer.
                for witnesses in ((), (("fermat", m, 2),), (("factor", m, divisor + 2),), (found, found),
                                  (("fermat", m + 2, 3),), (("factor", m - 1, 2), found),
                                  (found, ("factor", m + 1, 2))):
                    for operation, n, value in claims:
                        self.assertRejects(operation, {"n": n}, value, (certificates[operation], witnesses))

    def test_values_from_psi_13_on_need_their_pocklington_certificate(self):
        """The gap around psi_13 (whose base-2 test passes) and the one above 2**127 - 1."""
        self.assertEqual((prev_prime_of(PSI_13), next_prime_of(PSI_13)), (PSI_13 - 168, PSI_13 + 142))
        top = 2 ** 127 - 1
        self.assertEqual((is_prime(top), next_prime_of(top)), (True, 2 ** 127 + 29))
        cases = (("next_prime", PSI_13 - 168), ("prev_prime", PSI_13 + 142), ("next_prime", top),
                 ("prev_prime", 2 ** 127 + 29), ("prev_prime", 2 ** 127), ("next_prime", top - 1))
        for operation, n in cases:
            value, evidence = claim(operation, n)
            certificate, witnesses = evidence
            with self.subTest(operation=operation, n=n):
                self.assertAccepts(operation, {"n": n}, value, evidence)
                if value < PSI_13:
                    continue
                self.assertEqual(certificate[:2], ("pocklington", value))
                kind, recorded, factors, bases = certificate
                for wrong in ((), ("trial", value), prime_evidence(PSI_13 + 196),
                              (kind, recorded, factors[:-1], bases[:-1]),  # An incomplete factorization of r - 1.
                              (kind, recorded, factors, (True,) + bases[1:]),
                              (kind, recorded, list(factors), bases), list(certificate)):
                    self.assertRejects(operation, {"n": n}, value, (wrong, witnesses))

    def test_composites_claimed_prime_are_refused(self):
        """psi_12 passes the strong tests to the bases 2 to 37 and psi_13 to the bases 2 to 41; a Carmichael number
        passes Fermat's test to every base prime to it."""
        for value in (PSI_12, PSI_13, CARMICHAEL, 31621, 561, 15841):
            below = prev_prime_of(value)
            with self.subTest(value=value):
                evidence = ((), witnesses_for(below + 1, value))
                self.assertRejects("next_prime", {"n": below}, value, evidence)
                pretended = ((), witnesses_for(value + 1, next_prime_of(value) + 1))
                self.assertRejects("prev_prime", {"n": next_prime_of(value) + 1}, value, pretended)
        # psi_13 with the Pocklington certificate of the prime that follows it, and with the well-formed one that
        # comes closest to proving it prime.
        self.assertRejects("next_prime", {"n": PSI_13 - 168}, PSI_13, (prime_evidence(PSI_13 + 142), ()))
        self.assertRejects("next_prime", {"n": PSI_13 - 168}, PSI_13, (forged_pocklington(PSI_13), ()))

    def corruptions(self):
        """(label, operation, args, value, evidence) for the claim next_prime(31607) = 31627, whose gap holds 31621 =
        103 * 307, and for prev_prime(31627) = 31607: every entry is wrong."""
        args, value, certificate = {"n": 31607}, 31627, ()
        witness = ("fermat", 31621, 5)
        evidence = (certificate, (witness,))
        self.assertAccepts("next_prime", args, value, evidence)
        self.assertAccepts("prev_prime", {"n": 31627}, 31607, evidence)
        later, earlier = next_prime_of(value), prev_prime_of(31607)
        self.assertEqual((31621 % 103, is_prime(31629), later, earlier), (0, False, 31643, 31601))
        # A Pocklington certificate of 31627 (31626 = 2 * 3 * 3 * 7 * 251) that v0.7's checker accepts, but which is
        # not canonical below psi_13, where the certificate is ().
        pocklington = ("pocklington", 31627, ((2, 1, ("trial", 2)), (3, 2, ("trial", 3)), (7, 1, ("trial", 7)),
                                              (251, 1, ("trial", 251))), (3, 3, 3, 3))
        self.assertIs(check_prime(31627, True, pocklington), True)
        table = [
            ("composite value", "next_prime", args, 31629, evidence),
            ("a later prime", "next_prime", args, later, ((), (witness,))),
            ("the input itself", "next_prime", args, 31607, evidence),
            ("the prime input itself", "next_prime", {"n": 31627}, 31627, ((), ())),
            ("a smaller prime", "next_prime", args, earlier, evidence),
            ("two", "next_prime", args, 2, evidence),
            ("prev_prime above n", "prev_prime", {"n": 31627}, later, evidence),
            ("prev_prime at n", "prev_prime", {"n": 31627}, 31627, evidence),
            ("prev_prime at the prime n", "prev_prime", {"n": 31627}, 31627, ((), ())),
            ("prev_prime at a small prime n", "prev_prime", {"n": 17}, 17, ((), ())),
            ("prev_prime of 2", "prev_prime", {"n": 2}, 2, ((), ())),
            ("prev_prime of 2 below", "prev_prime", {"n": 2}, 1, ((), ())),
            ("prev_prime of 1", "prev_prime", {"n": 1}, 0, ((), ())),
            ("prev_prime of -10", "prev_prime", {"n": -10}, -11, ((), ())),
        ]
        for label, odd in (("Rational", Rational(31627)), ("float", 31627.0), ("str", "31627"), ("bool", True),
                           ("int subclass", Integer(31627)), ("impostor", support.impostor(Holder(31627))),
                           ("forged record", support.forged(Holder, value=31627)), ("tuple", (31627,)),
                           ("list", [31627]), ("None", None)):
            table.append(("value: " + label, "next_prime", args, odd, evidence))
        table.append(("value: bool two", "next_prime", {"n": 1}, True, ((), ())))
        for label, odd in (("bool", {"n": True}), ("bool below two", {"n": False}), ("float", {"n": 31607.0}),
                           ("str", {"n": "31607"}), ("Rational", {"n": Rational(31607)}), ("missing", {}),
                           ("extra", {"n": 31607, "m": 31607}), ("renamed", {"N": 31607}),
                           ("pairs", [("n", 31607)]), ("int subclass", {"n": Integer(31607)})):
            table.append(("args: " + label, "next_prime", odd, 2 if label == "bool below two" else value, evidence))
        for label, odd in (("list", [certificate, (witness,)]), ("witness list", (certificate, [witness])),
                           ("entry list", (certificate, (list(witness),))), ("certificate list", ([], (witness,))),
                           ("missing", ()), ("certificate only", (certificate,)), ("witnesses only", ((witness,),)),
                           ("None", None), ("extra item", (certificate, (witness,), ())),
                           ("no witness", (certificate, ())),
                           ("extra entry after", (certificate, (witness, ("factor", 31608, 2)))),
                           ("extra entry before", (certificate, (("factor", 31608, 2), witness))),
                           ("repeated entry", (certificate, (witness, witness))),
                           ("entry outside the gap", (certificate, (witness, ("fermat", 31645, 3))))):
            table.append(("evidence: " + label, "next_prime", args, value, odd))
        for label, entry in (("short", ("fermat", 31621)), ("long", ("fermat", 31621, 5, 0)),
                             ("unknown kind", ("lucas", 31621, 5)), ("capitalized kind", ("Fermat", 31621, 5)),
                             ("bytes kind", (b"fermat", 31621, 5)), ("float m", ("fermat", 31621.0, 5)),
                             ("Rational m", ("fermat", Rational(31621), 5)), ("bool m", ("fermat", True, 5)),
                             ("another m", ("fermat", 31623, 5)), ("float base", ("fermat", 31621, 5.0)),
                             ("bool base", ("fermat", 31621, True)), ("str base", ("fermat", 31621, "5")),
                             ("base 1", ("fermat", 31621, 1)), ("base 0", ("fermat", 31621, 0)),
                             ("negative base", ("fermat", 31621, -5)), ("base m", ("fermat", 31621, 31621)),
                             ("base above m", ("fermat", 31621, 31626)), ("base 2", ("fermat", 31621, 2)),
                             ("base -1", ("fermat", 31621, 31620)), ("not a divisor", ("factor", 31621, 2)),
                             ("divisor 1", ("factor", 31621, 1)), ("divisor m", ("factor", 31621, 31621)),
                             ("negative divisor", ("factor", 31621, -103)), ("bool divisor", ("factor", 31621, True)),
                             ("divisor above m", ("factor", 31621, 2 * 31621))):
            table.append(("witness: " + label, "next_prime", args, value, (certificate, (entry,))))
        for label, odd in (("v0.7 trial", ("trial", 31627)), ("pocklington below psi_13", pocklington),
                           ("another number's", prime_evidence(PSI_13 + 142)), ("nested", ((),)),
                           ("str", ("31627",))):
            table.append(("certificate: " + label, "next_prime", args, value, (odd, (witness,))))
        for label, name in (("the other operation", "prev_prime"), ("misspelled", "nextprime"),
                            ("capitalized", "NEXT_PRIME"), ("empty", ""), ("None", None), ("bytes", b"next_prime"),
                            ("another family's", "primitive_root")):
            table.append(("operation: " + label, name, args, value, evidence))
        table.append(("operation: next_prime for prev_prime", "next_prime", {"n": 31627}, 31607, evidence))
        return table

    def test_checker_rejects_corrupted_claims(self):
        for label, operation, args, value, evidence in self.corruptions():
            with self.subTest(label):
                self.assertRejects(operation, args, value, evidence)

    def test_oversized_forgeries_fail_fast(self):
        """Oversized evidence answers False at once; a value farther than the budget can scan, or over
        max_integer_bits, fails by resource limit, as the card says, without scanning."""
        args, value, witness = {"n": 31607}, 31627, ("fermat", 31621, 5)
        for label, evidence in (("many witnesses", ((), (witness,) * 100_000)),
                                ("huge base", ((), (("fermat", 31621, support.HUGE),))),
                                ("huge divisor", ((), (("factor", 31621, support.HUGE),))),
                                ("huge m", ((), (("fermat", support.HUGE, 5),))),
                                ("huge certificate", ((("pocklington", 31627, ((2, 1, ()),) * 50_000, (3,) * 50_000),
                                                      (witness,)))),
                                ("deep certificate", (((((((),),),),),), (witness,)))):
            with self.subTest(label):
                self.assertFastRejects("next_prime", args, value, evidence)
        # Above psi_13 a forged Pocklington certificate with 50,000 factors is walked node by node and refused.
        n, top = PSI_13 - 168, PSI_13 + 142
        crowded = ("pocklington", top, tuple((2, 1, ()) for _ in range(50_000)), (3,) * 50_000)
        self.assertFastRejects("next_prime", {"n": n}, top, (crowded, (("fermat", PSI_13, 43),)), may_exhaust=True)
        self.assertFastRejects("prev_prime", {"n": 100}, support.HUGE, ((), ()))
        self.assertFastRejects("prev_prime", {"n": 100}, -support.HUGE, ((), ()))
        for label, operation, n, far in (("10**30 + 57", "next_prime", 13, 10 ** 30 + 57),
                                         ("HUGE + 1", "next_prime", 13, support.HUGE + 1),
                                         ("of HUGE", "prev_prime", support.HUGE, 97),
                                         ("of 10**30", "prev_prime", 10 ** 30, 97),
                                         ("HUGE + 2", "next_prime", support.HUGE, support.HUGE + 2)):
            with self.subTest(label, operation=operation):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):  # The bound comes before the certificate is read.
                    support.verdict(family, operation, {"n": n}, far, (("not", "a", "certificate"), ()))
                self.assertLess(time.perf_counter() - start, FAST)

    def test_the_scan_is_charged_before_it_runs(self):
        """A true claim fails by resource limit under a budget smaller than its scan, and passes under the default.
        More witnesses than the range can need are refused before anything is charged."""
        value, evidence = claim("next_prime", 31607)
        self.assertAccepts("next_prime", {"n": 31607}, value, evidence)
        for work in (1, 18):  # The scan of the 19 integers 31608 ... 31626 costs one unit each, paid at once.
            with self.subTest(max_work=work), self.assertRaises(ResourceLimitError):
                support.verdict(family, "next_prime", {"n": 31607}, value, evidence,
                                limits=ComputationLimits(max_work=work))
        tight = ComputationLimits(max_work=1)
        with self.assertRaises(ResourceLimitError):  # The three integers 14, 15 and 16 cost three units.
            support.verdict(family, "next_prime", {"n": 13}, 17, ((), ()), limits=tight)
        self.assertRejects("next_prime", {"n": 13}, 17, ((), (("factor", 15, 3),)), limits=tight)
        self.assertRejects("next_prime", {"n": 31607}, value, ((), evidence[1] * 20), limits=tight)

    def test_every_exponentiation_is_charged_before_it_runs(self):
        """The checker never calls the builtin pow: its exponentiations are the base-2 tests of the skipped integers
        from 10,000 on that are prime to 2*3*...*97, and the Fermat witnesses, each through charged_pow."""
        def unpaid(*arguments):
            raise AssertionError("the checker called the builtin pow")

        calls, charged = [], checker.charged_pow

        def recording(base, exponent, modulus):
            calls.append((base, exponent, modulus))
            return charged(base, exponent, modulus)

        with patch.object(checker, "pow", unpaid, create=True), patch.object(checker, "charged_pow", recording):
            for operation, n in (("next_prime", 31607), ("prev_prime", 31627), ("next_prime", CARMICHAEL - 18),
                                 ("next_prime", PSI_13 - 168), ("next_prime", 9967), ("next_prime", 13)):
                value, evidence = claim(operation, n)
                low, high = (n + 1, value) if operation == "next_prime" else (value + 1, n)
                bases = {m: base for kind, m, base in evidence[1] if kind == "fermat"}
                expected = []
                for m in range(max(low, TRIAL_BELOW), high):
                    if gcd(m, PRIMORIAL) == 1:
                        expected.append((2, m - 1, m))
                        if m in bases:
                            expected.append((bases[m], m - 1, m))
                calls.clear()
                with self.subTest(operation=operation, n=n):
                    self.assertAccepts(operation, {"n": n}, value, evidence)
                    self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
