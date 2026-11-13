"""The prime_search family of mathforge.ntheory: next_prime, prev_prime.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register ntheory prime_search

Every expected value is computed here, never by the producer: Eratosthenes' sieve below SIEVE_LIMIT, trial division by
the sieve's primes below 10**12, and above that the strong probable-prime test to the prime bases 2 to 73, whose first
thirteen bases already decide every integer below psi_13. The witnesses that a claim needs follow the card's rule
(docs/v09/cards/m1-ntheory.md), written out as needs_witness: the skipped m >= 10,000 that are prime to 2*3*...*97,
pass Fermat's test to base 2 and are composite. The card's example 15841 = 7*31*73 shares the factor 7 with
2*3*...*97 and needs none (see tests/v09/ntheory/test_prime_search_checker.py, which tests the checker alone).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt, prod
import random
import time
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.ntheory import prime_search as family
from mathforge.certify.ntheory import prime_search as checker
from mathforge.contracts import CheckStatus
from mathforge.kernels import integers as kernel
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13

from .. import support

SIEVE_LIMIT = 1_000_001  # The sieve's primes are also the trial divisors of every integer below TRIAL_LIMIT.
TRIAL_LIMIT = 10 ** 12
SWEEP = 20_000  # The card: agreement with a sieve for every n below it.
PSI_12 = 318665857834031151167461  # 399165290221 * 798330580441, a strong pseudoprime to the twelve bases 2 to 37.
CARMICHAEL = 56052361  # 211 * 421 * 631: a Carmichael number without a prime factor up to 97.
MERSENNE_83 = 2 ** 83 - 1  # 167 * 57912614113275649087721: from psi_13 on, a strong pseudoprime to base 2.
CIPOLLA_71 = (4 ** 71 - 1) // 3  # (2**71 - 1)(2**71 + 1)/3: from psi_13 on, split by the strong test to base 2.
# Base-2 pseudoprimes from 10,000 on without a prime factor up to 97, with the witness that the producer gives: the
# first of the bases 3, 5, ..., 97 that its strong tests try with b**(m - 1) not 1 modulo m, else, for the Carmichael
# number, which has no such base among them, the divisor gcd(x - 1, m) from the square root x of 1 that the strong test
# to base 2 met (133141 = 211 * 631). psi_12 needs base 41, psi_13 base 43, beyond the thirteen bases 2 to 41. From
# psi_13 on 2 and 3 screen: 2**83 - 1 passes the strong test to 2 and fails Fermat's test to 3; psi_13 passes both and
# takes the bases 5 to 97 once prime_evidence has found it composite; and (4**71 - 1)/3, where base 2 meets a square
# root of 1, still gets the Fermat base 3 rather than that factor.
WITNESSED = ((31621, ("fermat", 31621, 5)), (60701, ("fermat", 60701, 3)),
             (CARMICHAEL, ("factor", CARMICHAEL, 133141)), (2 ** 32 + 1, ("fermat", 2 ** 32 + 1, 3)),
             (2 ** 64 + 1, ("fermat", 2 ** 64 + 1, 3)), (2 ** 67 - 1, ("fermat", 2 ** 67 - 1, 3)),
             (PSI_12, ("fermat", PSI_12, 41)), (PSI_13, ("fermat", PSI_13, 43)),
             (MERSENNE_83, ("fermat", MERSENNE_83, 3)), (CIPOLLA_71, ("fermat", CIPOLLA_71, 3)))
# Proth primes k * 2**e + 1 whose certificates and checks fit the default budget. Strong tests to all 25 prime bases
# up to 97 on the value, at a multiplication modulo it per bit each, made the producer refuse the two largest.
PROTH = (5 * 2 ** 1947 + 1, 3 * 2 ** 2208 + 1, 5 * 2 ** 3313 + 1)
# Queries whose first candidate from psi_13 on is a strong pseudoprime to base 2, with the witness that base 3 gives:
# prev_prime(2**139) meets 2**139 - 1 = 5625767248687 * 123876132205208335762278423601, whose m - 1 the integer kernel
# cannot factor within the budget, and next_prime(2**128) meets the Fermat number 2**128 + 1.
BASE_THREE = (("prev_prime", 2 ** 139, ("fermat", 2 ** 139 - 1, 3)),
              ("next_prime", 2 ** 128, ("fermat", 2 ** 128 + 1, 3)))
# Base-2 pseudoprimes that the checker refutes on its own: by a gcd with 2*3*...*97 from 10,000 on (the Carmichael
# numbers 15841 = 7*31*73 and 10585 = 5*29*73, and 18721 = 97*193), by the trial test below 10,000.
UNWITNESSED = (15841, 10585, 18721, 341, 561, 645, 1729, 2047)
UNCHECKABLE = "for the independent check of the evidence"  # The producer's refusal; never the verifier's.


def sieve(limit: int) -> bytearray:
    """flags[n] is 1 exactly for the primes n below limit."""
    flags = bytearray([1]) * limit
    flags[0:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return flags


FLAGS = sieve(SIEVE_LIMIT)
PRIMES = tuple(p for p in range(SIEVE_LIMIT) if FLAGS[p])
PRIMORIAL = prod(p for p in PRIMES if p <= 97)  # 2*3*...*97, the 25 primes up to 97.
_BASES = tuple(p for p in PRIMES if p <= 73)


def passes_strong_test(m: int, base: int) -> bool:
    """Whether the odd m > base passes the strong probable-prime test to base."""
    d, s = m - 1, 0
    while not d & 1:
        d >>= 1
        s += 1
    x = pow(base, d, m)
    if x == 1 or x == m - 1:
        return True
    for _ in range(s - 1):
        x = x * x % m
        if x == m - 1:
            return True
    return False


def strong_probable_prime(m: int) -> bool:
    """The strong probable-prime test to the prime bases 2 to 73, for m > 73."""
    return not any(m % p == 0 for p in _BASES) and all(passes_strong_test(m, base) for base in _BASES)


def is_prime(m: int) -> bool:
    """Whether m is prime: the sieve below SIEVE_LIMIT, trial division below TRIAL_LIMIT, the strong test above."""
    if m < SIEVE_LIMIT:
        return m >= 2 and FLAGS[m] == 1
    if m < TRIAL_LIMIT:
        root = isqrt(m)
        for p in PRIMES:
            if p > root:
                return True
            if m % p == 0:
                return False
    return strong_probable_prime(m)


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
    """The card's rule: a skipped composite m that the checker cannot refute on its own."""
    return m >= 10_000 and gcd(m, PRIMORIAL) == 1 and pow(2, m - 1, m) == 1 and not is_prime(m)


def skipped(operation: str, n: int, value: int) -> range:
    """The integers strictly between n and the value, which the checker shows composite one by one."""
    if operation == "next_prime":
        return range(n + 1, value) if n >= 2 else range(0)
    return range(value + 1, n)


@dataclass(frozen=True)
class Holder:
    """A record that holds the right integer, where every value of the family is a plain int."""
    value: int


class Integer(int):
    """An int subclass, which no decoded document holds: not an exact int for the checker."""


class PrimeSearchTests(support.FamilyTestCase):
    FAMILY = family

    def measure(self, operation: str, n: int) -> tuple[int, int]:
        """(production, check): the work of produce() on n and of the checker on its claim, each in a fresh budget."""
        with computation(ComputationLimits(max_work=10 ** 9), fresh=True) as budget:
            value, evidence = family.produce(operation, {"n": n})
        production = budget.work
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation(operation, {"n": n}, value, evidence), True)
        return production, budget.work

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, n, expected in (("next_prime", -5, 2), ("next_prime", 1, 2), ("next_prime", 2, 3),
                                       ("next_prime", 13, 17), ("next_prime", 89, 97), ("prev_prime", 3, 2),
                                       ("prev_prime", 18, 17), ("prev_prime", 100, 97)):
            with self.subTest(operation=operation, n=n):
                result = support.run(operation, n=n)
                self.assertEqual(result.value, expected)
                self.assertEqual(result.steps[0].outputs[1], ((), ()))  # No certificate below psi_13, no witness.
                self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
                self.assertTrue(mf.verify(result).verified)
                self.roundtrip(result)

    def test_pseudoprime_gaps_carry_exactly_the_witnesses_they_need(self):
        """A base-2 pseudoprime prime to 2*3*...*97 gets one witness, in either direction; a pseudoprime that the
        checker refutes itself gets none. From psi_13 on the value comes with its Pocklington certificate."""
        for m, witness in WITNESSED:
            below, above = prev_prime_of(m), next_prime_of(m)
            self.assertTrue(needs_witness(m))
            for operation, n, value in (("next_prime", below, above), ("prev_prime", above, below)):
                with self.subTest(m=m, operation=operation):
                    found, (certificate, witnesses) = family.produce(operation, {"n": n})
                    self.assertEqual((found, witnesses), (value, (witness,)))
                    self.assertEqual(certificate[:2], ("pocklington", value) if value >= PSI_13 else ())
                    self.assertAccepts(operation, {"n": n}, found, (certificate, witnesses))
        for m in UNWITNESSED:
            below, above = prev_prime_of(m), next_prime_of(m)
            self.assertFalse(needs_witness(m))
            for operation, n, value in (("next_prime", below, above), ("prev_prime", above, below)):
                with self.subTest(m=m, operation=operation):
                    self.assertEqual(family.produce(operation, {"n": n}), (value, ((), ())))
        # End to end: a Fermat witness, a factor witness, and a Pocklington certificate with a witness.
        for operation, n, value in (("next_prime", 31607, 31627), ("prev_prime", CARMICHAEL + 18, CARMICHAEL - 18),
                                    ("next_prime", PSI_13 - 168, PSI_13 + 142)):
            with self.subTest(operation=operation, n=n):
                result = support.run(operation, n=n)
                self.assertEqual(result.value, value)
                self.assertEqual(len(result.steps[0].outputs[1][1]), 1)
                self.roundtrip(result)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error."""
        for operation in ("next_prime", "prev_prime"):
            for n in (0.5, 13.0, True, False, "1", None, (13,), [13], mf.Rational(13), mf.Rational(27, 2),
                      mf.Symbol("n")):
                with self.subTest(operation=operation, n=n):
                    self.assertInvalidInput(support.run(operation, n=n))
                    with self.assertRaises(mf.InvalidInput):  # produce() itself tests the exact type.
                        family.produce(operation, {"n": n})
        # The card: no prime lies below 2, so prev_prime needs n >= 3.
        for n in (2, 1, 0, -1, -7, -10 ** 30):
            with self.subTest(operation="prev_prime", n=n):
                self.assertInvalidInput(support.run("prev_prime", n=n))
        with self.assertRaises(mf.OperationError) as raised:
            support.value("prev_prime", n=2)
        self.assertEqual(raised.exception.code, "invalid_input")
        with self.assertRaises(mf.UnsupportedOperation):
            family.produce("primitive_root", {"n": 13})

    def test_resource_limits(self):
        """An integer of 2**65536 and max_work=1 exceed the budget; the family takes no matrix argument."""
        for operation in ("next_prime", "prev_prime"):
            for label, n in (("HUGE", support.HUGE), ("-HUGE", -support.HUGE)):
                with self.subTest(operation=operation, n=label):  # A label: unittest cannot print 65,536 bits.
                    self.assertResourceLimit(support.run(operation, n=n))
        with self.assertRaises(ResourceLimitError):
            family.produce("next_prime", {"n": support.HUGE})
        tight = ComputationLimits(max_work=1)
        for operation, n in (("next_prime", 13), ("prev_prime", 100), ("next_prime", 31607), ("prev_prime", 10 ** 30)):
            with self.subTest(operation=operation, n=n, max_work=1):
                self.assertResourceLimit(support.run(operation, n=n, limits=tight))
        # A value one bit longer than max_integer_bits: 2**64 - 59 is the largest prime below 2**64.
        narrow = ComputationLimits(max_integer_bits=64)
        self.assertResourceLimit(support.run("next_prime", n=2 ** 64 - 59, limits=narrow))
        self.assertEqual(support.run("prev_prime", n=2 ** 64 - 1, limits=narrow).value, 2 ** 64 - 59)
        # The documented cliff: from psi_13 on, the certificate of the value needs the factorization of r - 1.
        self.assertResourceLimit(support.run("next_prime", n=2 ** 400))

    def test_far_claims_are_refused_by_the_producer(self):
        """The prime gaps of 3,000-bit integers cost the check more than the budget: the producer stops early, never
        the verifier ('Independent verification exceeded its budget'). Its own work, which pays for the scan and the
        strong tests that the check repeats, runs out of max_work before the check's estimate does, now that the
        check has a budget of its own."""
        for operation in ("next_prime", "prev_prime"):
            with self.subTest(operation=operation):
                start = time.perf_counter()
                result = support.run(operation, n=2 ** 3000)
                self.assertResourceLimit(result)
                self.assertNotIn("Independent verification", result.error.message)
                self.assertLess(time.perf_counter() - start, 5.0)

    def test_the_check_has_a_budget_of_its_own(self):
        """The verifier checks in a fresh budget of the request's limits, so, as in kernels.certificates, the producer
        refuses a claim only when the check alone needs more than max_work, whatever the production spent. With
        max_work the larger of the producer's work and the checker's, run() answers and the answer verifies (the
        producer used to require their sum); with max_work one unit below the checker's work, the producer refuses
        the claim itself, so the verifier never runs out of budget. Inside an active budget that has spent all but
        what the production needs, produce() still answers: the check does not run in what is left. The cases have a
        Fermat witness, a factor
        witness, Pocklington certificates, gaps across 2**63, where the checker's scan units per integer grow with the
        larger end of the range: r for next_prime, n for prev_prime, a Proth prime of 2,210 bits, and the gap of
        2**83 - 1, which passes the strong test to base 2 and fails its certificate."""
        for operation, n in (("next_prime", 13), ("prev_prime", 100), ("next_prime", 31607),
                             ("prev_prime", CARMICHAEL + 1), ("next_prime", PSI_13 - 168), ("prev_prime", 10 ** 12),
                             ("prev_prime", 10 ** 30), ("next_prime", 2 ** 63 - 25), ("prev_prime", 2 ** 63 + 29),
                             ("next_prime", PROTH[1] - 1), ("prev_prime", next_prime_of(MERSENNE_83))):
            with self.subTest(operation=operation, n=n):
                production, check = self.measure(operation, n)
                result = support.run(operation, n=n, limits=ComputationLimits(max_work=max(production, check)))
                self.assertTrue(mf.verify(result).verified, result.error)
                refused = support.run(operation, n=n, limits=ComputationLimits(max_work=check - 1))
                self.assertResourceLimit(refused)
                if production < check:  # Else the production itself runs out first.
                    self.assertIn(UNCHECKABLE, refused.error.message)
                limits = ComputationLimits(max_work=2_000_000)
                with computation(limits, fresh=True) as budget:
                    budget.tick(limits.max_work - production)  # Used to leave no room for the check: refused.
                    value, evidence = family.produce(operation, {"n": n})
                    self.assertEqual(budget.work, limits.max_work)
                self.assertIs(support.verdict(family, operation, {"n": n}, value, evidence), True)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen: the other
        witness kind, another Fermat base, a base that shares a factor with m (the only Fermat witnesses that a
        Carmichael number has), either divisor, and other Pocklington witnesses."""
        for m, divisor in ((31621, 103), (CARMICHAEL, 211), (PSI_13, 1287836182261)):
            bases = [base for base in range(3, 100) if pow(base, m - 1, m) != 1][:3]  # None for the Carmichael number.
            others = {("factor", m, divisor), ("factor", m, m // divisor), ("fermat", m, divisor),
                      *(("fermat", m, base) for base in bases)}
            for operation, n in (("next_prime", prev_prime_of(m)), ("prev_prime", next_prime_of(m))):
                value, (certificate, produced) = family.produce(operation, {"n": n})
                for witness in sorted(others - set(produced)):
                    with self.subTest(m=m, operation=operation, witness=witness):
                        self.assertAccepts(operation, {"n": n}, value, (certificate, (witness,)))
        # psi_13 + 142 with the last Pocklington witness below 300 for every prime of r - 1, where the kernel took the
        # first one.
        value, (certificate, witnesses) = family.produce("next_prime", {"n": PSI_13 - 168})
        kind, recorded, factors, chosen = certificate
        last = tuple(next(w for w in range(299, 1, -1) if pow(w, value - 1, value) == 1
                          and gcd(pow(w, (value - 1) // p, value) - 1, value) == 1) for p, _, _ in factors)
        self.assertNotEqual(last, chosen)
        self.assertAccepts("next_prime", {"n": PSI_13 - 168}, value, ((kind, recorded, factors, last), witnesses))
        # next_prime below 2 is 2 for every n, however large its absolute value.
        for label, n in (("-10**30", -10 ** 30), ("-2**65535", -(2 ** 65535))):
            with self.subTest(n=label):
                self.assertAccepts("next_prime", {"n": n}, 2, ((), ()))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, an impostor class, a record forged with object.__new__, a bool,
        a float or a str where an int belongs, a list for a tuple, missing or extra evidence, a wrong operation name
        and oversized forged evidence. A plain int value has no other canonical form to break."""
        args = {"n": 31607}
        value, evidence = family.produce("next_prime", args)
        certificate, witnesses = evidence
        self.assertEqual((value, evidence), (31627, ((), (("fermat", 31621, 5),))))
        self.assertAccepts("next_prime", args, value, evidence)
        witness = witnesses[0]
        table = [("composite value", "next_prime", args, 31629, evidence),
                 ("a later prime", "next_prime", args, 31643, evidence),
                 ("the input", "next_prime", args, 31607, evidence),
                 ("a smaller prime", "next_prime", args, 31601, evidence),
                 ("prev_prime above n", "prev_prime", {"n": 31627}, 31643, evidence),
                 ("prev_prime at n", "prev_prime", {"n": 31627}, 31627, ((), ())),
                 ("prev_prime of 2", "prev_prime", {"n": 2}, 2, ((), ())),
                 ("next_prime below 2", "next_prime", {"n": 1}, 3, ((), ()))]
        for label, odd in (("bool", True), ("float", 31627.0), ("str", "31627"), ("Rational", mf.Rational(31627)),
                           ("int subclass", Integer(31627)), ("impostor", support.impostor(Holder(31627))),
                           ("forged record", support.forged(Holder, value=31627)), ("tuple", (31627,)),
                           ("list", [31627]), ("None", None)):
            table.append(("value: " + label, "next_prime", args, odd, evidence))
        for label, odd in (("bool", {"n": True}), ("float", {"n": 31607.0}), ("str", {"n": "31607"}),
                           ("missing", {}), ("extra", {"n": 31607, "m": 1}), ("pairs", [("n", 31607)])):
            table.append(("args: " + label, "next_prime", odd, value, evidence))
        for label, odd in (("list", [certificate, witnesses]), ("witness list", (certificate, [witness])),
                           ("entry list", (certificate, (list(witness),))), ("missing", ()),
                           ("certificate only", (certificate,)), ("extra item", (certificate, witnesses, ())),
                           ("no witness", (certificate, ())), ("repeated witness", (certificate, (witness, witness))),
                           ("extra witness", (certificate, (witness, ("factor", 31622, 2)))),
                           ("Fermat liar", (certificate, (("fermat", 31621, 2),))),
                           ("not a divisor", (certificate, (("factor", 31621, 101),))),
                           ("another m", (certificate, (("fermat", 31623, 5),))),
                           ("certificate below psi_13", (("trial", 31627), witnesses)),
                           ("many witnesses", (certificate, (witness,) * 100_000)),
                           ("huge base", (certificate, (("fermat", 31621, support.HUGE),))),
                           ("huge m", (certificate, (("fermat", support.HUGE, 5),)))):
            table.append(("evidence: " + label, "next_prime", args, value, odd))
        for label, name in (("the other operation", "prev_prime"), ("misspelled", "nextprime"), ("None", None),
                            ("another family's", "primitive_root")):
            table.append(("operation: " + label, name, args, value, evidence))
        # From psi_13 on: no certificate, another prime's certificate, an incomplete factorization of r - 1.
        top, (pocklington, near) = family.produce("next_prime", {"n": PSI_13 - 168})
        kind, recorded, factors, bases = pocklington
        other = family.produce("next_prime", {"n": PSI_13 + 142})[1][0]
        for label, odd in (("none", ()), ("another prime's", other), ("incomplete", (kind, recorded, factors[:-1],
                                                                                    bases[:-1]))):
            table.append(("certificate: " + label, "next_prime", {"n": PSI_13 - 168}, top, (odd, near)))
        for label, operation, odd_args, odd_value, odd_evidence in table:
            with self.subTest(label):
                self.assertRejects(operation, odd_args, odd_value, odd_evidence)
        # mf.verify consults the checker: a tampered value, or tampered evidence, fails the mathematics check.
        result = support.run("next_prime", n=31607)
        for wrong, wrong_evidence in ((31643, None), (31627, ((), ()))):
            with self.subTest(value=wrong, evidence=wrong_evidence):
                report = support.statuses(mf.verify(support.tampered(result, wrong, wrong_evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        results = [support.run(operation, n=n) for operation, n in (("next_prime", 31607), ("prev_prime", 100),
                                                                    ("prev_prime", CARMICHAEL + 18),
                                                                    ("next_prime", PSI_13 - 168))]
        decoded = [mf.loads(mf.dumps(result)) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for result in results + decoded:
                with self.subTest(method=result.method, value=result.value):
                    self.assertTrue(mf.verify(result).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("next_prime", n=31607)
            self.assertEqual(failed.error.code, "internal_error")

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX."""
        for operation, n, text in (("next_prime", 13, "17"), ("prev_prime", 100, "97"),
                                   ("next_prime", 2 ** 127 - 1, str(2 ** 127 + 29))):
            with self.subTest(operation=operation, n=n):
                result = support.run(operation, n=n)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)  # The certificate and witnesses.
                self.assertEqual((mf.render_text(result.value), mf.to_latex(result.value)), (text, text))
                self.assertEqual(mf.render_text(decoded), "completed / value [exact; complete]: " + text)
                self.assertEqual(mf.to_data(result.value)["data"], {"type": "integer", "value": text})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(912)
        for operation, low in (("next_prime", -100), ("prev_prime", 3)):
            for n in [rng.randrange(low, 10 ** 6) for _ in range(12)] + [rng.randrange(10 ** 6, 10 ** 20)
                                                                          for _ in range(4)]:
                with self.subTest(operation=operation, n=n):
                    structured = support.run(operation, n=n).value
                    self.assertEqual(structured, next_prime_of(n) if operation == "next_prime" else prev_prime_of(n))
                    self.assertEqual(support.lean_value(operation, n=n), structured)
                    self.assertEqual(support.value(operation, n=n), structured)
        self.assertIs(support.lean_value("prev_prime", n=2), lean.FALLBACK)  # The structured path then rejects it.

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, n in (("next_prime", 31607), ("prev_prime", 100)):
            with self.subTest(operation=operation, n=n):
                result = self.answer(support.call(operation, {"n": n}))
                self.assertEqual(result, support.run(operation, n=n))
                self.assertTrue(mf.verify(result).verified)
        # A decimal string is an exact integer too, and call_values encodes the argument through its kind.
        self.assertEqual(self.answer(support.call("next_prime", {"n": "-5"})).value, 2)
        self.assertEqual(self.answer(support.call_values("prev_prime", n=PSI_13 + 142)).value, PSI_13 - 168)
        self.assertEqual(self.assertToolError(support.call("prev_prime", {"n": 2}), "invalid_input")["stage"],
                         "computation")
        for raw in (True, 0.5, None, [13]):
            with self.subTest(n=raw):
                self.assertToolError(support.call("next_prime", {"n": raw}), "invalid_input")

    def test_agrees_with_a_sieve_below_20000(self):
        """The card: every n below 20,000, in both directions. No gap below 31,621 needs a witness, and every value lies
        below psi_13, so the evidence is always ((), ()); the checker accepts every claim."""
        for operation, first in (("next_prime", -3), ("prev_prime", 3)):
            for n in range(first, SWEEP):
                value, evidence = family.produce(operation, {"n": n})
                expected = next_prime_of(n) if operation == "next_prime" else prev_prime_of(n)
                self.assertEqual((value, evidence), (expected, ((), ())), (operation, n))
                self.assertIs(support.verdict(family, operation, {"n": n}, value, evidence), True, (operation, n))

    def test_seeded_agreement_with_a_slow_reference(self):
        """At least 300 seeded inputs per operation, from below 10**5 to above psi_13: the value is the reference's,
        the witnesses name exactly the skipped integers that need one, and the checker accepts the claim. From psi_13
        on the certificate needs the factorization of r - 1, and a resource error (the documented cliff) is the only
        other outcome allowed there."""
        rng = random.Random(20261112)
        ranges = ((-50, 100_000, 130), (100_000, TRIAL_LIMIT, 110), (TRIAL_LIMIT, PSI_13, 70), (PSI_13, 2 ** 100, 40))
        for operation in ("next_prime", "prev_prime"):
            answered = 0
            for low, high, count in ranges:
                for _ in range(count):
                    n = rng.randrange(max(low, 3) if operation == "prev_prime" else low, high)
                    expected = next_prime_of(n) if operation == "next_prime" else prev_prime_of(n)
                    try:
                        value, evidence = family.produce(operation, {"n": n})
                    except ResourceLimitError:
                        self.assertGreaterEqual(expected, PSI_13, (operation, n))
                        continue
                    self.assertEqual(value, expected, (operation, n))
                    self.assertEqual(tuple(entry[1] for entry in evidence[1]),
                                     tuple(m for m in skipped(operation, n, value) if needs_witness(m)), (operation, n))
                    self.assertIs(support.verdict(family, operation, {"n": n}, value, evidence), True, (operation, n))
                    answered += 1
            self.assertGreaterEqual(answered, 300, operation)

    def test_a_fooled_screen_takes_the_kernel_witness(self):
        """From psi_13 on the strong tests prove nothing. A composite that passed them all (here psi_13, as if it were a
        strong pseudoprime to every base) fails its certificate, and the integer kernel's prime_certificate gives its
        witness. Below psi_13 the screen is exact, so a disagreement there is the producer's internal error, never
        invalid input."""
        screen, certify = family._screen, kernel.prime_certificate
        calls = []

        def fooled(budget, costs, m, bases):
            return (False, None) if m in (PSI_13, 31621) else screen(budget, costs, m, bases)

        def recording(m):
            calls.append(m)
            return certify(m)

        with patch.object(kernel, "prime_certificate", recording):
            # Unfooled, psi_13 passes the strong tests to 2 and 3, the screen from psi_13 on: prime_evidence's attempt
            # finds it composite, and base 43 gives its witness. Then the value goes to the kernel for its certificate.
            self.assertEqual(family.produce("next_prime", {"n": PSI_13 - 168})[1][1], (("fermat", PSI_13, 43),))
            self.assertEqual(calls, [PSI_13, PSI_13 + 142])
            calls.clear()
            with patch.object(family, "_screen", fooled):
                value, evidence = family.produce("next_prime", {"n": PSI_13 - 168})
                failed = support.run("next_prime", n=31607)
        self.assertEqual(calls, [PSI_13, PSI_13, PSI_13 + 142])  # prime_evidence's attempt, then the witness.
        self.assertEqual((value, evidence[1]), (PSI_13 + 142, (certify(PSI_13)[1],)))
        self.assertAccepts("next_prime", {"n": PSI_13 - 168}, value, evidence)
        self.assertEqual(failed.error.code, "internal_error")

    def test_values_from_psi_13_on_take_two_screening_tests(self):
        """From psi_13 on the strong tests to 2 and 3 screen a candidate and prime_evidence decides, so a Proth prime
        whose certificate and check fit the budget is answered from both sides, with no witness, and mf.verify accepts
        the Result. With strong tests to all 25 prime bases up to 97 on the value, the producer refused the two largest
        primes of PROTH, of 2,210 and 3,316 bits."""
        for value in PROTH:
            for operation, n in (("next_prime", value - 1), ("prev_prime", value + 1)):
                with self.subTest(operation=operation, bits=value.bit_length()):
                    result = support.run(operation, n=n)
                    self.assertEqual(result.value, value, result.error)
                    certificate, witnesses = result.steps[0].outputs[1]
                    self.assertEqual((certificate[:2], witnesses), (("pocklington", value), ()))
                    self.assertTrue(mf.verify(result).verified)

    def test_base_three_catches_the_strong_pseudoprimes_that_powers_of_two_meet(self):
        """Every composite 2**p - 1 with p prime passes the strong test to base 2, and so do the composite Fermat
        numbers. Base 3 shows them composite with one exponentiation. With base 2 alone screening from psi_13 on,
        prev_prime(2**139) went to prime_evidence(2**139 - 1), which ran out of budget factoring m - 1."""
        for operation, n, witness in BASE_THREE:
            m = witness[1]
            with self.subTest(operation=operation, bits=n.bit_length()):
                self.assertTrue(m >= PSI_13 and needs_witness(m) and passes_strong_test(m, 2))
                value = next_prime_of(n) if operation == "next_prime" else prev_prime_of(n)
                found, (certificate, witnesses) = family.produce(operation, {"n": n})
                self.assertEqual((found, witnesses), (value, (witness,)))
                self.assertAccepts(operation, {"n": n}, found, (certificate, witnesses))

    def test_witnesses_come_in_increasing_order(self):
        """No prime gap below 1.2 * 10**8 needs two witnesses, so the order is tested on a screen that also gives every
        other composite prime to 2*3*...*97 a witness, ("fermat", m, 2): in the maximal prime gap from 1,357,201 to
        1,357,333 that is thirteen witnesses, which prev_prime meets from n downwards and still lists by increasing m.
        The checker refuses such evidence (the extra entries), so this tests produce() alone."""
        screen = family._screen

        def witnessing(budget, costs, m, bases):
            composite, witness = screen(budget, costs, m, bases)
            return (True, ("fermat", m, 2)) if composite and witness is None else (composite, witness)

        expected = [m for m in range(1357202, 1357333) if gcd(m, PRIMORIAL) == 1]
        self.assertEqual(len(expected), 13)
        with patch.object(family, "_screen", witnessing):
            for operation, n, value in (("next_prime", 1357201, 1357333), ("prev_prime", 1357333, 1357201)):
                with self.subTest(operation=operation):
                    found, (certificate, witnesses) = family.produce(operation, {"n": n})
                    self.assertEqual((found, [entry[1] for entry in witnesses]), (value, expected))

    def test_screens_match_the_checker(self):
        """Which skipped integers need a witness depends on the checker's gcd screen and trial bound: the producer uses
        the same ones."""
        self.assertEqual(family._PRIMORIAL, PRIMORIAL)
        self.assertEqual(family._PRIMORIAL, checker._PRIMORIAL)
        self.assertEqual(family._BASES, tuple(p for p in PRIMES if p <= 41))
        # From psi_13 on 2 and 3 screen, and the primes from 5 to 97 only choose witnesses.
        self.assertEqual(family._LARGE_BASES, (2, 3))
        self.assertEqual(family._WITNESS_BASES, tuple(p for p in PRIMES if 5 <= p <= 97))
        self.assertEqual(checker.TRIAL_BELOW, 10_000)


if __name__ == "__main__":
    unittest.main()
