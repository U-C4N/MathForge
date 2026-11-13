"""Prime proofs and certified factorizations for the number theory checkers (mathforge.certify.ntheory.factorization),
the evidence that the producer side builds for them (mathforge.kernels.certificates), and the charged modular power.

prime_proof is tested in each of its three ranges with canonical, forged and wrongly shaped certificates;
check_factorization accepts the factorizations that factorization_evidence builds and rejects every mutation of them,
forged huge exponents at once; charged_pow agrees with pow and is paid for before the builtin runs.

Both sides use the cost model of trusted.integers: the checker charges exactly what its functions compute, and the
producers refuse evidence whose check would not fit in max_work, so that evidence produced under some limits is
accepted under the same limits in a fresh budget (seeded over limits and sizes). factorization_evidence checks the
integer cap first and removes high powers of the primes below 2**10 itself, quickly and charged.

The checker imports no producer code. These tests do, to build its inputs and to have a second opinion: the integer
kernel's factor, its Miller-Rabin test and its v0.7 prime certificates, which v0.7's certificate checker verifies.
"""

from __future__ import annotations

from math import gcd, isqrt, prod
import random
import time
import unittest
from unittest.mock import patch

from mathforge.certify.ntheory import factorization as checker
from mathforge.certify.ntheory import primes as certify_primes
from mathforge.errors import InvalidInput
from mathforge.kernels import certificates
from mathforge.kernels import integers as kernel
from mathforge.limits import ComputationLimits, ResourceLimitError, computation, current_budget
from mathforge.model import Rational
from mathforge.trusted import integers as trusted_integers
from mathforge.trusted.primes import DETERMINISTIC_BELOW as PSI_13

FAST = 0.25  # Seconds: a forgery is refused well within this, where building its powers would never end.
PSI_12 = 318665857834031151167461  # The smallest strong pseudoprime to the twelve bases 2 to 37 (OEIS A014233).
PSI_12_FACTORS = (399165290221, 798330580441)
PSI_13_FACTORS = (1287836182261, 2575672364521)  # psi_13 passes the strong tests to all thirteen bases 2 to 41.
# psi_2 ... psi_12, the smallest strong pseudoprimes to the first k prime bases from 10,000 on.
PSEUDOPRIMES = (1373653, 25326001, 3215031751, 2152302898747, 3474749660383, 341550071728321, 3825123056546413051,
                PSI_12)
# Primes from 10,000 to psi_13, which the strong tests decide: among them the neighbours of psi_12 and the last prime
# below psi_13.
MIDDLE_PRIMES = (10007, 65537, 2 ** 31 - 1, 2 ** 61 - 1, 10 ** 18 + 3, 2 ** 64 - 59, 10 ** 24 + 7,
                 318665857834031151167441, 318665857834031151167483, PSI_13 - 168)
# Primes from psi_13 on, which need a Pocklington certificate: the first three above psi_13 (the first a safe prime,
# whose certificate nests another Pocklington certificate), three Mersenne primes and the Proth prime 3 * 2**534 + 1.
LARGE_PRIMES = (PSI_13 + 142, PSI_13 + 196, PSI_13 + 210, 2 ** 89 - 1, 2 ** 107 - 1, 2 ** 127 - 1, 3 * 2 ** 534 + 1)
# Numbers whose factorizations the integer kernel finds within the default budget, from 1 to large primes.
FACTORED = (1, 2, 3, 4, 9999, 10007, 600851475143, 2 ** 64, 2 ** 64 - 1, 2 ** 128 - 1, 10 ** 30, 99991 ** 3 * 65537,
            (2 ** 89 - 1) * 6, (2 ** 89 - 1) ** 2 * 10007, (PSI_13 + 142) * 2 ** 10 * 3 ** 5, (2 ** 127 - 1) * 1000003)
# The review's k * 2**m + 1 primes of 4,690 and 5,950 bits, with the work that prime_proof needs for the kernel's
# certificates: more than the default max_work, although the kernel certifies them with 1,046,018 and 1,666,186 units.
REVIEW_PRIMES = ((5 * 2 ** 4687 + 1, 2_350_863), (5 * 2 ** 5947 + 1, 4_732_109))
# High powers of the primes below 2**10, up to the integer cap: the kernel's trial division took 0.6 s for 2**65535.
SMALL_PRIME_POWERS = (2 ** 65_535, 3 ** 41_000, 5 ** 28_000, 1021 ** 6_500, 2 ** 30_000 * 3 ** 20_000,
                      3 ** 40_000 * 1021, 2 ** 64 * 3 ** 40_000 * (2 ** 89 - 1), prod(kernel.SMALL_PRIMES) ** 7)
BIG = ComputationLimits(max_work=10 ** 12)
STANDALONE = object()  # For attempt(): no active budget, as for a call outside operations_v6.run.
REFUSED = object()  # What attempt() returns for a ResourceLimitError.


class Text(str):
    """A str subclass: equal to its str, but no exact str."""


class Liar(str):
    """A str subclass that claims to equal every object."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def sieve(limit):
    """flags[n] is 1 exactly for the primes n below limit: Eratosthenes' sieve, a second opinion on small numbers."""
    flags = bytearray([1]) * limit
    flags[0:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return flags


def replace(items, index, value):
    """The tuple items with the entry at index replaced by value."""
    return items[:index] + (value,) + items[index + 1:]


def pocklington(n):
    """The integer kernel's Pocklington certificate of the prime n >= 10,000, which v0.7's checker accepts."""
    claimed, certificate = kernel.prime_certificate(n)
    if not (claimed and certificate[0] == "pocklington" and certify_primes.check_prime(n, True, certificate)):
        raise AssertionError(f"no Pocklington certificate for {n}")
    return certificate


def spread(n, certificate):
    """The Pocklington certificate of n with a different witness for every prime of n - 1: the most exponentiations
    that a certificate with these primes makes check_prime run."""
    kind, recorded, factors, _ = certificate
    chosen = []
    for prime, _, _ in factors:
        witness = 2
        while witness in chosen or gcd(pow(witness, (n - 1) // prime, n) - 1, n) != 1:
            witness += 1
        chosen.append(witness)
    return (kind, recorded, factors, tuple(chosen))


def satisfied(n, prime, witness):
    """Whether the witness meets Pocklington's two conditions for the prime of n - 1."""
    return pow(witness, n - 1, n) == 1 and gcd(pow(witness, (n - 1) // prime, n) - 1, n) == 1


def forged(n):
    """The Pocklington certificate that comes closest to proving the composite n prime: n - 1 factored into certified
    primes, and for each prime the first witness below 300 that meets Pocklington's conditions for it, when there is
    one (2 otherwise). By Pocklington's theorem some prime of n - 1 has none."""
    factors = tuple((p, e, kernel.prime_certificate(p)[1]) for p, e in kernel.factor(n - 1))
    witnesses = []
    for prime, _, _ in factors:
        witnesses.append(next((w for w in range(2, 300) if satisfied(n, prime, w)), 2))
    return ("pocklington", n, factors, tuple(witnesses))


def exponentiations(function, *args):
    """function(*args) under a budget of 10**9 units, with every exponentiation of check_prime and powers_by_prime
    recorded: (its result, the work charged to the budget, the work of the recorded exponentiations weighed as
    charged_pow weighs one)."""
    weights = []

    def recording(*arguments):
        if len(arguments) == 3:
            weights.append(arguments[1].bit_length() * checker.multiplication_work(arguments[2].bit_length()))
        return pow(*arguments)

    with patch.object(certify_primes, "pow", recording, create=True), \
            patch.object(trusted_integers, "pow", recording, create=True), \
            computation(ComputationLimits(max_work=10 ** 9)) as budget:
        result = function(*args)
    return result, budget.work, sum(weights)


def evidence_of(pairs):
    """The certified factorization with these (prime, exponent) pairs, each prime certified by prime_evidence."""
    return tuple((p, e, certificates.prime_evidence(p)) for p, e in sorted(pairs))


def attempt(function, *args, limits=STANDALONE):
    """function(*args) in a fresh budget of the limits (None: the default ones), as verify_result runs a checker, or
    with no active budget for STANDALONE: its result, or REFUSED when it raises ResourceLimitError."""
    try:
        if limits is STANDALONE:
            return function(*args)
        with computation(limits, fresh=True):
            return function(*args)
    except ResourceLimitError:
        return REFUSED


class PrimeProofTest(unittest.TestCase):
    def test_trial_range(self):
        flags = sieve(10_000)
        for n in range(-3, 10_000):
            self.assertIs(checker.prime_proof(n, ()), n >= 2 and bool(flags[n]), n)

    def test_strong_test_range(self):
        for n in MIDDLE_PRIMES:
            with self.subTest(n=n):
                pocklington(n)  # A second opinion that runs no strong test.
                self.assertIs(checker.prime_proof(n, ()), True)
        # psi_k passes the strong tests to the first k prime bases; psi_12 passes the twelve bases 2 to 37, and before
        # base 41 was added trusted.primes.is_prime called it prime.
        for n in PSEUDOPRIMES:
            self.assertIs(checker.prime_proof(n, ()), False, n)
        self.assertEqual(prod(PSI_12_FACTORS), PSI_12)
        rng = random.Random(20261112)
        numbers = [rng.randrange(10_000, 1 << rng.randrange(15, 82)) for _ in range(300)]
        for _ in range(100):  # And primes: the first that the kernel meets from a random odd start.
            candidate = rng.getrandbits(rng.randrange(15, 81)) | 1
            while not kernel.is_probable_prime(candidate):
                candidate += 2
            numbers.append(candidate)
        for n in numbers:
            self.assertTrue(10_000 <= n < PSI_13)
            self.assertIs(checker.prime_proof(n, ()), kernel.is_probable_prime(n), n)

    def test_pocklington_range(self):
        evidence = {n: certificates.prime_evidence(n) for n in LARGE_PRIMES}
        for index, (n, certificate) in enumerate(evidence.items()):
            with self.subTest(n=n):
                self.assertEqual(certificate[:2], ("pocklington", n))
                self.assertIs(checker.prime_proof(n, certificate), True)
                self.assertIs(checker.prime_proof(n, ()), False)  # From psi_13 on a certificate is needed.
                self.assertIs(checker.prime_proof(n, list(certificate)), False)
                other = LARGE_PRIMES[(index + 1) % len(LARGE_PRIMES)]
                self.assertIs(checker.prime_proof(n, evidence[other]), False)
                self.assertIs(checker.prime_proof(other, certificate), False)
        # The certificate of the first safe prime above psi_13 nests the Pocklington certificate of (n - 1) / 2.
        self.assertEqual([child[0] for _, _, child in evidence[PSI_13 + 142][2]], ["trial", "pocklington"])

    def test_composites_from_psi_13_on_are_refused(self):
        self.assertEqual(prod(PSI_13_FACTORS), PSI_13)
        for n in (PSI_13, PSI_13_FACTORS[0] * (PSI_13 + 142), (2 ** 61 - 1) * (2 ** 31 - 1)):
            with self.subTest(n=n):
                certificate = forged(n)
                self.assertIs(checker.prime_proof(n, certificate), False)
                self.assertIs(checker.prime_proof(n, ()), False)
        # psi_13 passes the strong tests to all thirteen bases, and its forgery comes close: most primes of n - 1 have
        # a witness, only not all of them.
        _, _, factors, witnesses = forged(PSI_13)
        met = [satisfied(PSI_13, prime, witness) for (prime, _, _), witness in zip(factors, witnesses)]
        self.assertGreater(sum(met), len(met) // 2)
        self.assertFalse(all(met))

    def test_forged_certificates_are_refused(self):
        extra = (2 ** 89 - 1, 1, certificates.prime_evidence(2 ** 89 - 1))
        merged_for = []
        for n in (2 ** 127 - 1, PSI_13 + 142):
            certificate = certificates.prime_evidence(n)
            self.assertIs(checker.prime_proof(n, certificate), True)
            kind, recorded, factors, witnesses = certificate
            last = len(factors) - 1
            prime, exponent, child = factors[last]
            # A p-th power w = 3**p has w**((n - 1)/p) = 3**(n - 1) = 1: it is no witness for p.
            power = pow(3, prime, n)
            mutations = {
                "another number": (kind, n + 2, factors, witnesses),
                "witness 1": (kind, recorded, factors, replace(witnesses, last, 1)),
                "witness n": (kind, recorded, factors, replace(witnesses, last, n)),
                "witness -3": (kind, recorded, factors, replace(witnesses, last, -3)),
                "a p-th power as witness": (kind, recorded, factors, replace(witnesses, last, power)),
                "an exponent too large": (kind, recorded, replace(factors, last, (prime, exponent + 1, child)),
                                          witnesses),
                "exponent 0": (kind, recorded, replace(factors, last, (prime, 0, child)), witnesses),
                "a prime dropped": (kind, recorded, factors[:-1], witnesses[:-1]),
                "a prime added": (kind, recorded, factors + (extra,), witnesses + (3,)),
                "primes reversed": (kind, recorded, factors[::-1], witnesses[::-1]),
                "an empty child": (kind, recorded, replace(factors, last, (prime, exponent, ())), witnesses),
                "another child": (kind, recorded, replace(factors, last, (prime, exponent, factors[0][2])), witnesses),
                "lists": (kind, recorded, list(factors), list(witnesses)),
                "a witness missing": (kind, recorded, factors, witnesses[:-1]),
                "a fifth field": certificate + ((),),
                "another kind": ("Pocklington", recorded, factors, witnesses),
                "a trial certificate": ("trial", n),
                "a composite certificate": ("fermat", n, 2),
            }
            if last >= 2 and factors[last - 1][1] == exponent == 1:
                # The two largest primes of n - 1 merged into their product, with the best forged certificate for it:
                # the product of the factors stays n - 1.
                merged = factors[last - 1][0] * prime
                composite = factors[:last - 1] + ((merged, 1, forged(merged)),)
                mutations["a composite prime"] = (kind, recorded, composite, witnesses[:last])
                merged_for.append(n)
            for name, mutation in mutations.items():
                with self.subTest(n=n, mutation=name):
                    self.assertIs(checker.prime_proof(n, mutation), False)
        self.assertEqual(merged_for, [2 ** 127 - 1])

    def test_wrong_shapes_and_types(self):
        # A Pocklington certificate that check_prime accepts, for a prime that needs none: not canonical.
        needless = pocklington(10007)
        self.assertTrue(certify_primes.check_prime(10007, True, needless))
        large = certificates.prime_evidence(2 ** 89 - 1)
        for n, certificate in ((7, ("trial", 7)), (10007, needless), (10007, ("trial", 10007)), (7, None), (7, []),
                               (10007, [()]), (10007, ((),)), (2 ** 61 - 1, pocklington(2 ** 61 - 1)),
                               (2 ** 89 - 1, None), (2 ** 89 - 1, list(large)), (2 ** 89 - 1, large[:3]),
                               (2 ** 89 - 1, "pocklington"), (4, ())):
            with self.subTest(n=n, certificate=certificate):
                self.assertIs(checker.prime_proof(n, certificate), False)
        for n in (True, 7.0, "7", None, Rational(7), Rational(2 ** 89 - 1)):
            with self.subTest(n=n):
                self.assertIs(checker.prime_proof(n, ()), False)
                self.assertIs(checker.prime_proof(n, large), False)

    def test_a_str_subclass_is_no_tag(self):
        """The tag of a Pocklington certificate is the exact str "pocklington": an instance of a str subclass that
        equals it, or that claims to equal everything, is refused by prime_proof, by its type alone and before any
        charge, and so by check_factorization, though the rest of the certificate proves the prime. Both accepted the
        equal one before."""
        for n in (2 ** 89 - 1, PSI_13 + 142):
            certificate = certificates.prime_evidence(n)
            self.assertIs(checker.prime_proof(n, certificate), True)
            self.assertIs(checker.check_factorization(2 * n, ((2, 1, ()), (n, 1, certificate))), True)
            for tag in (Text("pocklington"), Liar("pocklington"), Liar("trial")):
                forged_tag = (tag,) + certificate[1:]
                with self.subTest(n=n, tag=type(tag).__name__ + repr(str(tag))):
                    self.assertEqual(forged_tag[0], "pocklington")
                    with computation(BIG) as budget:
                        self.assertIs(checker.prime_proof(n, forged_tag), False)
                    self.assertEqual(budget.work, 0)
                    self.assertIs(checker.check_factorization(2 * n, ((2, 1, ()), (n, 1, forged_tag))), False)

    def test_the_proof_is_paid_for_before_check_prime_runs(self):
        n = 3 * 2 ** 534 + 1
        certificate = certificates.prime_evidence(n)
        calls = []
        with patch.object(checker, "check_prime", lambda *arguments: calls.append(arguments) or True):
            with computation(ComputationLimits(max_work=1_000)), self.assertRaises(ResourceLimitError):
                checker.prime_proof(n, certificate)
        self.assertEqual(calls, [])
        with computation() as budget:
            self.assertIs(checker.prime_proof(n, certificate), True)
        self.assertGreater(budget.work, 1_000)
        # Below psi_13 the strong tests are paid for too: a multiplication per bit of n for each of the 13 bases.
        with computation() as budget:
            self.assertIs(checker.prime_proof(PSI_13 - 168, ()), True)
        self.assertEqual(budget.work, 13 * (PSI_13 - 168).bit_length())

    def test_the_charge_covers_what_check_prime_runs(self):
        """Every exponentiation that check_prime runs, in the nested certificates too, is paid for before it runs: the
        work charged covers the recorded exponentiations, also when every prime of n - 1 has its own witness."""
        for n in LARGE_PRIMES:
            certificate = certificates.prime_evidence(n)
            for variant in (certificate, spread(n, certificate)):
                with self.subTest(n=n, witnesses=len(set(variant[3]))):
                    verdict, charged, used = exponentiations(checker.prime_proof, n, variant)
                    self.assertIs(verdict, True)
                    self.assertGreater(used, 0)
                    self.assertLessEqual(used, charged)


class CheckFactorizationTest(unittest.TestCase):
    def test_accepts_the_produced_factorizations(self):
        rng = random.Random(20261113)
        numbers = list(FACTORED) + [rng.randrange(1, 1 << rng.randrange(1, 64)) for _ in range(300)]
        for n in numbers:
            with self.subTest(n=n):
                evidence = certificates.factorization_evidence(n)
                self.assertEqual(tuple((p, e) for p, e, _ in evidence), kernel.factor(n))
                self.assertEqual(prod(p ** e for p, e, _ in evidence), n)
                self.assertIs(checker.check_factorization(n, evidence), True)
                self.assertEqual(checker.primes_of(evidence), tuple(p for p, _, _ in evidence))
        self.assertEqual(certificates.factorization_evidence(1), ())
        self.assertEqual(checker.primes_of(certificates.factorization_evidence(360)), (2, 3, 5))

    def test_accepts_factorizations_built_by_hand(self):
        """Numbers whose factors the kernel's rho does not find within the default budget, and a power near the
        integer cap: their evidence is built from known primes with prime_evidence."""
        for pairs in (((PSI_12_FACTORS[0], 1), (PSI_12_FACTORS[1], 1)),
                      ((2, 1), (PSI_13_FACTORS[0], 1), (PSI_13_FACTORS[1], 1)),
                      ((2, 2), (3, 1), (2 ** 89 - 1, 1), (2 ** 107 - 1, 1)),
                      ((2 ** 127 - 1, 3), (PSI_13 + 142, 2), (3 * 2 ** 534 + 1, 1)),
                      ((3, 41_000),)):
            n, evidence = prod(p ** e for p, e in pairs), evidence_of(pairs)
            with self.subTest(n=n.bit_length()):
                start = time.perf_counter()
                self.assertIs(checker.check_factorization(n, evidence), True)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_rejects_mutations(self):
        n = 2 ** 3 * 3 ** 2 * 5 * 10007 * (2 ** 89 - 1)
        evidence = certificates.factorization_evidence(n)
        self.assertIs(checker.check_factorization(n, evidence), True)
        two, three, five, middle, large = evidence
        self.assertEqual([entry[:2] for entry in evidence], [(2, 3), (3, 2), (5, 1), (10007, 1), (2 ** 89 - 1, 1)])
        other = certificates.prime_evidence(2 ** 107 - 1)
        mutations = {
            "unsorted": (three, two, five, middle, large),
            "a repeated prime": ((2, 1, ()), (2, 2, ()), three, five, middle, large),
            "a composite prime": (two, (45, 1, ()), middle, large),
            "a wrong product": (two, three, five, middle, (large[0], 2, large[2])),
            "a prime missing": (two, three, five, middle),
            "an extra prime": evidence + ((2 ** 107 - 1, 1, other),),
            "exponent 0": (two, three, five, (7, 0, ()), middle, large),
            "a negative exponent": (two, three, five, (7, -1, ()), middle, large),
            "a bool exponent": (two, three, (5, True, ()), middle, large),
            "a float exponent": (two, three, (5, 1.0, ()), middle, large),
            "a str exponent": (two, three, (5, "1", ()), middle, large),
            "a Rational exponent": (two, three, (5, Rational(1), ()), middle, large),
            "a float prime": (two, three, (5.0, 1, ()), middle, large),
            "a str prime": (two, three, ("5", 1, ()), middle, large),
            "a bool prime": ((True, 1, ()), two, three, five, middle, large),
            "a list of factors": list(evidence),
            "a list entry": (two, three, [5, 1, ()], middle, large),
            "a list certificate": (two, three, five, middle, (large[0], 1, list(large[2]))),
            "an extra field": (two, three, (5, 1, (), ()), middle, large),
            "a missing field": (two, three, (5, 1), middle, large),
            "no certificate field": (two, three, five, middle, large[:2]),
            "a needless certificate": (two, three, five, (10007, 1, pocklington(10007)), large),
            "a missing certificate": (two, three, five, middle, (large[0], 1, ())),
            "another prime's certificate": (two, three, five, middle, (large[0], 1, other)),
            "a None certificate": (two, three, (5, 1, None), middle, large),
            "a str certificate": (two, three, (5, 1, "()"), middle, large),
        }
        for name, factors in mutations.items():
            with self.subTest(mutation=name):
                self.assertIs(checker.check_factorization(n, factors), False)
        for stated in (n + 1, n - 1, 2 * n, -n, 0):
            self.assertIs(checker.check_factorization(stated, evidence), False, stated)

    def test_rejects_inexact_numbers(self):
        twelve = certificates.factorization_evidence(12)
        self.assertIs(checker.check_factorization(12, twelve), True)
        self.assertIs(checker.check_factorization(1, ()), True)
        for n, factors in ((12.0, twelve), ("12", twelve), (Rational(12), twelve), (None, twelve), (True, ()),
                           (1.0, ()), (0, ()), (-1, ()), (12, None), (12, ()), (1, ((1, 1, ()),)), (1, [])):
            with self.subTest(n=n, factors=factors):
                self.assertIs(checker.check_factorization(n, factors), False)

    def test_composite_primes_are_refused(self):
        # psi_12 passes the strong tests to the twelve bases 2 to 37: before base 41 was added, trusted.primes.is_prime
        # called it prime, and these factorizations were accepted.
        for n, factors in ((PSI_12, ((PSI_12, 1, ()),)), (2 * PSI_12, ((2, 1, ()), (PSI_12, 1, ()))),
                           (PSI_12 ** 2, ((PSI_12, 2, ()),)), (4, ((4, 1, ()),)), (15, ((15, 1, ()),)),
                           (PSI_13, ((PSI_13, 1, ()),)), (PSI_13, ((PSI_13, 1, forged(PSI_13)),)),
                           (3 * PSI_13, ((3, 1, ()), (PSI_13, 1, forged(PSI_13))))):
            with self.subTest(n=n):
                self.assertIs(checker.check_factorization(n, factors), False)
        for n, pairs in ((PSI_12, tuple((p, 1) for p in PSI_12_FACTORS)),
                         (PSI_13, tuple((p, 1) for p in PSI_13_FACTORS))):
            self.assertIs(checker.check_factorization(n, evidence_of(pairs)), True)

    def test_forged_giant_exponents_fail_fast(self):
        big = 3 ** 41_000
        forgeries = (
            (2 ** 64, ((2, 10 ** 100, ()),)),
            (2 ** 64, ((2, 2 ** 65_536, ()),)),
            (big, ((3, 10 ** 30, ()),)),
            (big, ((3, 41_001, ()),)),  # One factor 3 too many: refused by the length of the product.
            (6, ((2, 1, ()), (3, 10 ** 1000, ()))),
            (2 ** 64, tuple((p, 10 ** 50, ()) for p in kernel.SMALL_PRIMES)),
            (prod(kernel.SMALL_PRIMES), tuple((p, 2, ()) for p in kernel.SMALL_PRIMES)),
            (2 ** 61 - 1, ((2 ** 61 - 1, 2 ** 64, ()),)),
            (2 ** 89 - 1, ((2 ** 89 - 1, 10 ** 6, certificates.prime_evidence(2 ** 89 - 1)),)),
        )
        for index, (n, factors) in enumerate(forgeries):
            with self.subTest(forgery=index):
                start = time.perf_counter()
                self.assertIs(checker.check_factorization(n, factors), False)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
                with computation() as budget:
                    self.assertIs(checker.check_factorization(n, factors), False)
                self.assertLess(budget.work, 2_000)

    def test_the_integer_cap(self):
        with self.assertRaises(ResourceLimitError):
            checker.check_factorization(2 ** 65_536, ((2, 65_536, ()),))
        start = time.perf_counter()
        self.assertIs(checker.check_factorization(3 ** 41_000, ((3, 41_000, ()),)), True)
        self.assertLess(time.perf_counter() - start, FAST)

    def test_the_product_is_checked_before_the_prime_proofs(self):
        n = 6 * (2 ** 89 - 1)
        evidence = certificates.factorization_evidence(n)
        proved = []

        def recording(prime, certificate):
            proved.append(prime)
            return True

        with patch.object(checker, "prime_proof", recording):
            self.assertIs(checker.check_factorization(n + 6, evidence), False)
            self.assertEqual(proved, [])
            self.assertIs(checker.check_factorization(n, evidence), True)
        self.assertEqual(proved, [2, 3, 2 ** 89 - 1])


class ChargedPowTest(unittest.TestCase):
    def test_agrees_with_pow(self):
        rng = random.Random(20261114)
        for _ in range(1_000):
            modulus = rng.randrange(1, 1 << rng.randrange(1, 600))
            exponent = rng.getrandbits(rng.randrange(0, 600))
            base = rng.randrange(-3 * modulus, 3 * modulus + 1)
            self.assertEqual(checker.charged_pow(base, exponent, modulus), pow(base, exponent, modulus))
        for base, exponent, modulus, expected in ((0, 0, 7, 1), (0, 0, 1, 0), (5, 0, 1, 0), (5, 3, 1, 0),
                                                  (-2, 3, 7, 6), (2, 10, 1000, 24), (10 ** 40 + 4, 1, 7, 1)):
            self.assertEqual(checker.charged_pow(base, exponent, modulus), expected)

    def test_it_is_paid_for_before_the_builtin_runs(self):
        calls = []

        def recording(*arguments):
            calls.append(arguments)
            return pow(*arguments)

        modulus = 2 ** 61 - 1
        with patch.object(checker, "pow", recording, create=True), \
                computation(ComputationLimits(max_work=100)) as budget:
            with self.assertRaises(ResourceLimitError):
                checker.charged_pow(3, 2 ** 200, modulus)  # 1 + 201 units.
            self.assertEqual(calls, [])
            self.assertEqual(budget.work, 0)
            self.assertEqual(checker.charged_pow(3, 2 ** 60, modulus), pow(3, 2 ** 60, modulus))
            self.assertEqual(budget.work, 1 + 61 * checker.multiplication_work(61))
            with self.assertRaises(ResourceLimitError):  # The active budget is shared: 124 units > 100.
                checker.charged_pow(3, 2 ** 60, modulus)
        self.assertEqual(len(calls), 1)

    def test_the_charge_grows_with_the_modulus(self):
        exponent = (1 << 255) + 1
        for bits in (61, 1024, 4096):
            modulus = (1 << bits) - 1
            with self.subTest(bits=bits), computation() as budget:
                self.assertEqual(checker.charged_pow(3, exponent, modulus), pow(3, exponent, modulus))
                self.assertGreaterEqual(budget.work, 256 * checker.multiplication_work(bits))
        self.assertEqual([checker.multiplication_work(bits) for bits in (64, 383, 384, 1024, 4096, 65_536)],
                         [1, 1, 2, 9, 129, 32_769])

    def test_huge_calls_are_refused_at_once(self):
        self.assertIsNone(current_budget())
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):  # Without an active budget the call opens one.
            checker.charged_pow(3, (1 << 65_535) + 3, (1 << 65_535) + 1)
        self.assertLess(time.perf_counter() - start, FAST)  # The builtin would run for minutes.
        self.assertIsNone(current_budget())

    def test_each_call_without_a_budget_opens_its_own(self):
        exponent = 1 << 1_500_000  # 1,500,001 bits: 1,500,002 units modulo a word-sized modulus.
        with patch.object(checker, "pow", lambda base, exponent, modulus: 0, create=True):
            for _ in range(3):  # 4.5 million units in all, more than one default budget holds.
                self.assertEqual(checker.charged_pow(3, exponent, 7), 0)
            with computation() as budget:
                checker.charged_pow(3, exponent, 7)
                with self.assertRaises(ResourceLimitError):
                    checker.charged_pow(3, exponent, 7)
        self.assertEqual(budget.work, 1_500_002)

    def test_rejects_inexact_arguments(self):
        inexact = (True, False, 2.0, "2", None, Rational(2), Rational(1, 2))
        calls = []
        with patch.object(checker, "pow", lambda *arguments: calls.append(arguments), create=True), \
                computation() as budget:
            for bad in inexact:
                for arguments in ((bad, 3, 7), (2, bad, 7), (2, 3, bad)):
                    with self.subTest(arguments=arguments), self.assertRaises(InvalidInput):
                        checker.charged_pow(*arguments)
            for arguments in ((2, -1, 7), (2, 3, 0), (2, 3, -7), (2, -3, -7)):
                with self.subTest(arguments=arguments), self.assertRaises(InvalidInput):
                    checker.charged_pow(*arguments)
        self.assertEqual(calls, [])
        self.assertEqual(budget.work, 0)


class EvidenceTest(unittest.TestCase):
    def test_prime_evidence(self):
        for n in (2, 3, 97, 9973) + MIDDLE_PRIMES:
            self.assertEqual(certificates.prime_evidence(n), (), n)
        for n in LARGE_PRIMES:
            with self.subTest(n=n):
                certificate = certificates.prime_evidence(n)
                self.assertEqual(certificate[:2], ("pocklington", n))
                self.assertTrue(certify_primes.check_prime(n, True, certificate))

    def test_prime_evidence_rejects_non_primes(self):
        for n in (-7, 0, 1, 4, 9999, 10001, PSI_12, PSI_13, PSI_13 + 2, (2 ** 89 - 1) ** 2,
                  (2 ** 61 - 1) * (2 ** 31 - 1), True, 7.0, "7", None, Rational(7), -(2 ** 70_000)):
            with self.subTest(n=n), self.assertRaises(InvalidInput):
                certificates.prime_evidence(n)

    def test_factorization_evidence_rejects_non_positive_numbers(self):
        for n in (0, -1, -12, True, 1.5, "12", None, Rational(12), -(2 ** 70_000)):
            with self.subTest(n=n), self.assertRaises(InvalidInput):
                certificates.factorization_evidence(n)

    def test_budget_errors_propagate(self):
        for function, n, work in ((certificates.factorization_evidence, PSI_12, 2_000),
                                  (certificates.factorization_evidence, 2 * PSI_13, 20_000),
                                  (certificates.prime_evidence, 3 * 2 ** 534 + 1, 2_000),
                                  (certificates.prime_evidence, 2 ** 61 - 1, 10)):
            with self.subTest(function=function.__name__, n=n), \
                    computation(ComputationLimits(max_work=work)), self.assertRaises(ResourceLimitError):
                function(n)


class SharedCostModelTest(unittest.TestCase):
    """The checker charges with the cost model of trusted.integers, from which the producers compute what the check of
    their evidence will cost: what it charges is exactly what those functions compute."""

    def test_the_checker_takes_its_units_from_the_trusted_layer(self):
        self.assertIs(checker.multiplication_work, trusted_integers.multiplication_work)
        self.assertEqual(checker.TRIAL_BELOW, trusted_integers.TRIAL_BELOW)

    def test_prime_proof_charges_prime_proof_work(self):
        cases = [(n, ()) for n in (-1, 2, 97, 9_973, 10_007, 2 ** 61 - 1, PSI_12, PSI_13 - 168, PSI_13)]
        cases += [(7, ("trial", 7)), (10_007, pocklington(10_007)), (True, ()), (PSI_13, forged(PSI_13))]
        for n in LARGE_PRIMES:
            certificate = certificates.prime_evidence(n)
            cases += [(n, certificate), (n, spread(n, certificate)), (n + 2, certificate), (n, certificate[:3] + ((),))]
        charged = 0
        for n, certificate in cases:
            with self.subTest(n=n, kind=certificate[:1]):
                with computation(BIG) as budget:
                    checker.prime_proof(n, certificate)
                self.assertEqual(budget.work, trusted_integers.prime_proof_work(n, certificate, BIG.max_work))
                charged += budget.work > 0
        self.assertGreater(charged, 4 * len(LARGE_PRIMES))

    def test_check_factorization_charges_factorization_work(self):
        for n in FACTORED + SMALL_PRIME_POWERS:
            evidence = certificates.factorization_evidence(n)
            with self.subTest(bits=n.bit_length()):
                with computation(BIG) as budget:
                    self.assertIs(checker.check_factorization(n, evidence), True)
                self.assertEqual(budget.work, trusted_integers.factorization_work(evidence, BIG.max_work))


class CheckAllowanceTest(unittest.TestCase):
    """The producers compute what the checker will charge for their evidence and refuse the evidence that its check
    cannot afford: when the check needs more than max_work, exactly where the checker, in a fresh budget of the same
    limits, fails. Production and check need not fit in max_work together, in an active budget or without one."""

    def test_the_review_examples_are_refused(self):
        """Under the default limits the kernel certified both primes, and prime_proof could not afford either
        certificate: the verification failed after all the production. Now the producers refuse them, in a budget and
        without one, and prime_proof and check_factorization fail on them at the very same point."""
        built = {}

        def recording(n, certificate, limit):
            built[n] = certificate
            return trusted_integers.prime_proof_work(n, certificate, limit)

        for (p, needed), limits in zip(REVIEW_PRIMES, (None, STANDALONE)):
            with self.subTest(bits=p.bit_length()):
                with patch.object(certificates, "prime_proof_work", recording, create=True):
                    self.assertIs(attempt(certificates.prime_evidence, p, limits=limits), REFUSED)
                certificate = built[p]
                self.assertEqual(certificate[:2], ("pocklington", p))
                self.assertEqual(trusted_integers.prime_proof_work(p, certificate, BIG.max_work), needed)
                for check_limits in (None, STANDALONE):
                    self.assertIs(attempt(checker.prime_proof, p, certificate, limits=check_limits), REFUSED)
                    factors = ((2, 1, ()), (p, 1, certificate))
                    self.assertIs(attempt(checker.check_factorization, 2 * p, factors, limits=check_limits), REFUSED)
                self.assertIs(attempt(certificates.factorization_evidence, 2 * p, limits=limits), REFUSED)
        # The refused certificate is sound: given the work it needs, prime_proof accepts it.
        p, needed = REVIEW_PRIMES[0]
        self.assertIs(attempt(checker.prime_proof, p, built[p], limits=ComputationLimits(max_work=needed)), True)

    def test_production_and_check_need_not_fit_together(self):
        """The check runs in a fresh budget of its own (operations_v6.run's verifier, the lean runner), so in an active
        budget, as without one, it may need all of max_work. 3 * 2**3168 + 1 costs the kernel 1,271,521 units and its
        check 964,367: each fits in the default max_work, both together do not, and the producer returns the
        certificate in a budget of the default limits and without one; prime_proof accepts it in a fresh budget of
        those limits. Before, the producer refused it in a budget, where production and check had to fit together."""
        p = 3 * 2 ** 3168 + 1
        with computation(BIG) as budget:
            certificate = certificates.prime_evidence(p)
        production, check = budget.work, trusted_integers.prime_proof_work(p, certificate, BIG.max_work)
        max_work = ComputationLimits().max_work
        self.assertLessEqual(max(production, check), max_work)
        self.assertGreater(production + check, max_work)
        self.assertEqual(certificate[:2], ("pocklington", p))
        for limits in (None, STANDALONE):
            with self.subTest(limits="default" if limits is None else "no active budget"):
                self.assertEqual(attempt(certificates.prime_evidence, p, limits=limits), certificate)
                self.assertIs(attempt(checker.prime_proof, p, certificate, limits=None), True)

    def test_the_check_alone_must_fit_in_max_work(self):
        """The producer refuses evidence whose check needs more than max_work, to the unit, however little the
        production spent. For 5 * 2**3313 + 1 the check costs more than the production: under limits between the two,
        the kernel's certificate used to come out and then fail its check. With max_work at the check the certificate
        comes out, and also at production + check - 1, where the rule of production and check together refused it."""
        p = 5 * 2 ** 3313 + 1
        with computation(BIG) as budget:
            certificate = certificates.prime_evidence(p)
        production, check = budget.work, trusted_integers.prime_proof_work(p, certificate, BIG.max_work)
        self.assertLess(production, check)
        with computation(ComputationLimits(max_work=check - 1), fresh=True):
            with self.assertRaisesRegex(ResourceLimitError, "for the independent check of the evidence"):
                certificates.prime_evidence(p)
        for max_work in (check, production + check - 1):
            with self.subTest(max_work=max_work):
                limits = ComputationLimits(max_work=max_work)
                self.assertEqual(attempt(certificates.prime_evidence, p, limits=limits), certificate)
                with computation(limits, fresh=True) as budget:
                    self.assertIs(checker.prime_proof(p, certificate), True)
                self.assertEqual(budget.work, check)

    def test_evidence_is_accepted_under_the_limits_it_was_produced_under(self):
        """Seeded: primes and integers of many sizes under limits from tight to wide (max_work from 1,000 to 3 million,
        max_integer_bits from 64 to the default), or without an active budget. Whenever a producer returns evidence,
        its checker accepts it under the same limits, in a fresh budget as verify_result runs it."""
        rng = random.Random(20261115)
        primes = list(MIDDLE_PRIMES + LARGE_PRIMES) + [5 * 2 ** 1947 + 1, 3 * 2 ** 2208 + 1, 5 * 2 ** 3313 + 1]
        for _ in range(40):  # Primes of 14 to 100 bits: the first that the kernel meets from a random odd start.
            candidate = rng.getrandbits(rng.randrange(14, 101)) | 1
            while not kernel.is_probable_prime(candidate):
                candidate += 2
            primes.append(candidate)
        numbers = list(FACTORED) + [rng.randrange(1, 1 << rng.randrange(1, 64)) for _ in range(80)]
        numbers += [prod(rng.choice(kernel.SMALL_PRIMES) ** rng.randrange(1, 1_500) for _ in range(rng.randrange(1, 5)))
                    for _ in range(30)]
        numbers += [p * rng.randrange(1, 1 << 20) for p in LARGE_PRIMES]
        cases = [(certificates.prime_evidence, checker.prime_proof, n) for n in primes]
        cases += [(certificates.factorization_evidence, checker.check_factorization, n) for n in numbers]
        produced = refused = 0
        for index, (producer, check, n) in enumerate(cases):
            if rng.randrange(5):
                scale = rng.randrange(3, 7)
                limits = ComputationLimits(max_work=rng.randrange(10 ** scale, 3 * 10 ** scale),
                                           max_integer_bits=rng.choice((64, 100, 1_000, 65_536)))
            else:
                limits = STANDALONE
            evidence = attempt(producer, n, limits=limits)
            if evidence is REFUSED:
                refused += 1
                continue
            produced += 1
            # The case, not n: unittest could not print an n of more than 4,300 digits in a failure report.
            with self.subTest(producer=producer.__name__, case=index, bits=n.bit_length(), limits=limits):
                self.assertIs(attempt(check, n, evidence, limits=limits), True)
        self.assertGreater(produced, 100)  # 164 of the 193 cases.
        self.assertGreater(refused, 20)


class SmallPrimePowersTest(unittest.TestCase):
    """factorization_evidence refuses an integer over max_integer_bits before any other work and removes the primes
    below 2**10 itself: the power of 2 by its lowest set bit, any other power p**e with about 2 log2(e) long divisions,
    each charged before it runs. The integer kernel's trial division takes one factor at a time and charges afterwards:
    2**65535 took it 65,535 divisions (0.6 s) for 65,536 units, and 2**524288, over the cap, 33 s."""

    def test_high_powers_of_small_primes_are_quick(self):
        for n in SMALL_PRIME_POWERS:
            with self.subTest(bits=n.bit_length()):
                start = time.perf_counter()
                with computation() as budget:
                    evidence = certificates.factorization_evidence(n)
                self.assertLess(time.perf_counter() - start, FAST)
                self.assertLess(budget.work, 100_000)
                self.assertEqual(prod(p ** e for p, e, _ in evidence), n)
                self.assertIs(checker.check_factorization(n, evidence), True)
        self.assertEqual(certificates.factorization_evidence(3 ** 41_000), ((3, 41_000, ()),))
        self.assertEqual(certificates.factorization_evidence(2 ** 65_535), ((2, 65_535, ()),))

    def test_too_large_or_too_costly_integers_are_refused_at_once(self):
        """Integers over the cap, and powers of primes above 2**10 whose first strong test the kernel cannot pay for."""
        for n in (2 ** 65_536, 3 ** 41_400, 2 ** 524_288, (2 ** 61 - 1) ** 1_000, 10_007 ** 4_000, 1031 ** 6_000):
            with self.subTest(bits=n.bit_length()):
                start = time.perf_counter()
                for limits in (STANDALONE, None):
                    self.assertIs(attempt(certificates.factorization_evidence, n, limits=limits), REFUSED)
                self.assertLess(time.perf_counter() - start, FAST)
        self.assertIs(attempt(certificates.prime_evidence, 2 ** 65_536 + 1), REFUSED)

    def test_the_integer_cap_comes_before_any_kernel_call(self):
        narrow = ComputationLimits(max_integer_bits=64)

        def unexpected(*arguments):
            raise AssertionError("the integer kernel ran before the size check")

        with patch.object(kernel, "factor", unexpected), patch.object(kernel, "prime_certificate", unexpected), \
                patch.object(kernel, "is_probable_prime", unexpected):
            for function, n in ((certificates.factorization_evidence, 2 ** 64),
                                (certificates.factorization_evidence, 2 ** 89 - 1),
                                (certificates.prime_evidence, 2 ** 89 - 1),
                                (certificates.prime_evidence, 2 ** 64 + 13)):
                with self.subTest(function=function.__name__, bits=n.bit_length()):
                    self.assertIs(attempt(function, n, limits=narrow), REFUSED)
        # Within the cap they produce, and the checker accepts under the same narrow limits.
        for function, check, n in ((certificates.factorization_evidence, checker.check_factorization, 2 ** 64 - 1),
                                   (certificates.prime_evidence, checker.prime_proof, 2 ** 61 - 1)):
            evidence = attempt(function, n, limits=narrow)
            self.assertIsNot(evidence, REFUSED)
            self.assertIs(attempt(check, n, evidence, limits=narrow), True)

    def test_each_division_is_paid_for_before_it_runs(self):
        records = []

        def recording(dividend, divisor):
            records.append((current_budget().work, dividend.bit_length(), divisor.bit_length()))
            return divmod(dividend, divisor)

        n = 3 ** 41_000 * 5 ** 3 * 7
        with patch.object(certificates, "divmod", recording, create=True), computation() as budget:
            evidence = certificates.factorization_evidence(n)
        self.assertEqual([(p, e) for p, e, _ in evidence], [(3, 41_000), (5, 3), (7, 1)])
        self.assertTrue(records)
        paid = 0
        for work, dividend_bits, divisor_bits in records:
            paid += 1 + trusted_integers.division_work(dividend_bits, divisor_bits)
            self.assertGreaterEqual(work, paid)
        self.assertGreaterEqual(budget.work, paid)
        # About 2 log2(e) divisions for p**e, where the kernel divided e times.
        self.assertLessEqual(len(records), sum(2 * e.bit_length() + 2 for e in (41_000, 3, 1)))


if __name__ == "__main__":
    unittest.main()
