"""Defects of the trusted layer, fixed in v0.9: trusted.primes.is_prime tested twelve bases against the bound of
thirteen, trusted.integers.powers_by_prime never returned for an empty list of primes, and the producers of prime
certificates charged by another cost model than their checker, which trusted.integers now holds for both sides."""

from math import prod
import random
import time
import unittest
from unittest.mock import patch

from mathforge.certify.ntheory.primes import MAX_DEPTH, check_prime
from mathforge.kernels import integers as kernel
from mathforge.limits import computation
from mathforge.trusted import integers as trusted_integers
from mathforge.trusted import primes, upoly
from mathforge.trusted.integers import powers_by_prime

FIRST_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
# psi_k is the smallest strong pseudoprime to the first k prime bases (OEIS A014233).
PSI = {1: 2047, 2: 1373653, 3: 25326001, 4: 3215031751, 5: 2152302898747, 6: 3474749660383, 7: 341550071728321,
       8: 341550071728321, 9: 3825123056546413051, 10: 3825123056546413051, 11: 3825123056546413051,
       12: 318665857834031151167461, 13: 3317044064679887385961981}
PSI_12_FACTORS = (399165290221, 798330580441)
PSI_13_FACTORS = (1287836182261, 2575672364521)

# (p, 2p - 1, k): both factors are prime and their product is a strong pseudoprime to exactly the first k prime
# bases. The first six products lie around psi_12, three on each side; the last three just below psi_13.
PSEUDOPRIMES = (
    (399139388101, 798278776201, 6), (399160036861, 798320073721, 6), (399162996721, 798325993441, 5),
    (399177211501, 798354423001, 5), (399184241821, 798368483641, 6), (399186489421, 798372978841, 6),
    (1287797521981, 2575595043961, 6), (1287801113821, 2575602227641, 6), (1287834090061, 2575668180121, 5),
)
# The three primes below and above psi_12, the three below psi_13, and primes across the rest of the range.
PRIMES = (
    318665857834031151167387, 318665857834031151167417, 318665857834031151167441,
    318665857834031151167483, 318665857834031151167497, 318665857834031151167501,
    3317044064679887385961783, 3317044064679887385961801, 3317044064679887385961813,
    10007, 2 ** 61 - 1, 2 ** 64 - 59, 2 ** 80 - 65, 10 ** 24 + 7, 3224192153648326577750017,
)


def strong_probable_prime(n, base):
    """Whether the odd n > 2 passes the strong probable-prime test to the base: the test's definition, written out."""
    d, s = n - 1, 0
    while not d & 1:
        d >>= 1
        s += 1
    x = pow(base, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def strong_run(n):
    """How many of the first thirteen primes, taken in order, n is a strong probable prime to."""
    run = 0
    for base in FIRST_PRIMES:
        if not strong_probable_prime(n, base):
            break
        run += 1
    return run


def proved_prime(n):
    """Whether a Pocklington certificate proves n prime: a proof that runs no strong test of trusted.primes."""
    claimed, certificate = kernel.prime_certificate(n)
    return claimed and check_prime(n, True, certificate)


class IsPrimeBasesTest(unittest.TestCase):
    def test_psi_12_is_composite(self):
        """The defect: psi_12 passes the twelve bases 2 to 37, so only base 41 can refuse it."""
        n = PSI[12]
        self.assertEqual(prod(PSI_12_FACTORS), n)
        self.assertEqual(strong_run(n), 12)
        self.assertFalse(strong_probable_prime(n, 41))
        self.assertIs(primes.is_prime(n), False)
        for factor in PSI_12_FACTORS:
            self.assertTrue(proved_prime(factor))
            self.assertIs(primes.is_prime(factor), True)

    def test_the_bound_is_psi_13(self):
        """The domain ends at psi_13, which passes all thirteen bases: the bound cannot be any higher."""
        bound = primes.DETERMINISTIC_BELOW
        self.assertEqual(bound, PSI[13])
        self.assertEqual(prod(PSI_13_FACTORS), bound)
        self.assertEqual(strong_run(bound), 13)
        for outside in (bound, bound + 2, 10 ** 30, -1, True, 7.0, "7", None):
            with self.subTest(outside=outside), self.assertRaises(ValueError):
                primes.is_prime(outside)
        self.assertIs(primes.is_prime(bound - 1), False)
        self.assertIs(primes.is_prime(bound - 168), True)  # 3317044064679887385961813, the last prime of the domain.
        self.assertEqual(bound - 168, PRIMES[8])

    def test_strong_pseudoprimes_are_refused(self):
        for bases in range(1, 13):
            n = PSI[bases]
            with self.subTest(psi=bases):
                self.assertGreaterEqual(strong_run(n), bases)
                self.assertIs(primes.is_prime(n), False)
        for p, q, run in PSEUDOPRIMES:
            n = p * q
            with self.subTest(n=n):
                self.assertEqual(q, 2 * p - 1)
                self.assertTrue(proved_prime(p) and proved_prime(q))
                self.assertEqual(strong_run(n), run)
                self.assertLess(n, primes.DETERMINISTIC_BELOW)
                self.assertIs(primes.is_prime(n), False)
        products = [p * q for p, q, _ in PSEUDOPRIMES]
        self.assertTrue(all(n < PSI[12] for n in products[:3]) and all(PSI[12] < n for n in products[3:]))
        # A Carmichael number, the square of a prime of the test's own bases, and a product of two of them.
        for n in (561, 41041, 41 * 41, 37 * 41, 41 * 43, 3 * 41, 41 ** 3):
            self.assertIs(primes.is_prime(n), False, n)

    def test_primes_around_both_bounds_are_accepted(self):
        for n in PRIMES:
            with self.subTest(n=n):
                self.assertTrue(proved_prime(n))
                self.assertIs(primes.is_prime(n), True)
        self.assertTrue(all(n < PSI[12] for n in PRIMES[:3]) and all(PSI[12] < n < PSI[13] for n in PRIMES[3:9]))
        # Nothing between those neighbours of the bounds is prime, and each base is itself a prime.
        for low, high in ((PRIMES[2], PRIMES[3]), (PRIMES[8], PSI[13])):
            self.assertFalse(any(primes.is_prime(n) for n in range(low + 1, high)))
        self.assertTrue(all(primes.is_prime(base) for base in FIRST_PRIMES))

    def test_small_numbers(self):
        limit = 20_000
        sieve = bytearray([1]) * limit
        sieve[0:2] = b"\0\0"
        for p in range(2, 142):
            if sieve[p]:
                sieve[p * p::p] = bytes(len(range(p * p, limit, p)))
        for n in range(limit):
            self.assertIs(primes.is_prime(n), bool(sieve[n]), n)

    def test_agrees_with_the_integer_kernel_below_2_61(self):
        rng = random.Random(20261111)
        numbers = [rng.getrandbits(rng.randrange(1, 62)) for _ in range(3000)]
        found = []
        for _ in range(600):  # Primes: the first one that the kernel meets from a random odd start.
            candidate = rng.getrandbits(rng.randrange(17, 61)) | 1
            while not kernel.is_probable_prime(candidate):
                candidate += 2
            found.append(candidate)
        small = [p for p in found if p < 2 ** 30]
        numbers += found + [p * q for p, q in zip(small, small[1:])]  # And semiprimes of two of them.
        self.assertGreater(len(numbers), 3700)
        accepted = 0
        for n in numbers:
            self.assertLess(n, 2 ** 61)
            expected = kernel.is_probable_prime(n)
            self.assertIs(primes.is_prime(n), expected, n)
            accepted += expected
        self.assertGreater(accepted, 600)


class PowersByPrimeTest(unittest.TestCase):
    def test_no_primes_give_an_empty_result_without_an_exponentiation(self):
        calls = []

        def counted(*arguments):
            calls.append(arguments)
            if len(calls) > 64:  # The defect: an empty group split into two empty groups, for ever.
                raise AssertionError("powers_by_prime does not terminate without primes")
            return pow(*arguments)

        with patch.object(trusted_integers, "pow", counted, create=True):
            for empty in ([], (), range(0), iter(())):
                self.assertEqual(powers_by_prime(2, 10, 7, empty), {})
            self.assertEqual(powers_by_prime(0, 0, 1, []), {})
            self.assertEqual(calls, [])
            # The counter is live: two primes take the first exponentiation and one more for each.
            self.assertEqual(powers_by_prime(2, 30, 7, [2, 3]), {2: pow(2, 15, 7), 3: pow(2, 10, 7)})
            self.assertEqual(len(calls), 3)
            # A one-shot iterator of primes, which v0.8 emptied in prod before it took the group, looping as above.
            self.assertEqual(powers_by_prime(2, 30, 7, iter([2, 3])), {2: pow(2, 15, 7), 3: pow(2, 10, 7)})
        self.assertEqual(len(calls), 6)

    def test_results_equal_direct_powers(self):
        rng = random.Random(20261112)
        pool = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 1009, 10007, 2 ** 61 - 1)
        for index in range(1500):
            chosen = rng.sample(pool, rng.randrange(1, 13))
            exponent = prod(p ** rng.randrange(1, 4) for p in chosen) * rng.randrange(1, 1000)
            modulus = rng.randrange(1, 10 ** rng.randrange(1, 40))
            base = rng.randrange(-modulus, 2 * modulus)
            expected = {p: pow(base, exponent // p, modulus) for p in chosen}
            given = chosen if index % 2 else tuple(chosen)  # The callers pass lists and tuples.
            result = powers_by_prime(base, exponent, modulus, given)
            self.assertEqual(result, expected, (base, exponent, modulus, chosen))
            self.assertEqual(sorted(result), sorted(chosen))
        self.assertEqual(powers_by_prime(3, 10, 7, [5]), {5: 2})
        self.assertEqual(powers_by_prime(3, 10, 1, [2, 5]), {2: 0, 5: 0})

    def test_the_callers_keep_their_results(self):
        """The kernel builds a Pocklington certificate with these powers and the checker verifies it with them."""
        for n in (10007, 2 ** 61 - 1, 2 ** 89 - 1):
            claimed, certificate = kernel.prime_certificate(n)
            self.assertTrue(claimed)
            self.assertEqual(certificate[0], "pocklington")
            self.assertTrue(check_prime(n, True, certificate))
            for (p, _, _), witness in zip(certificate[2], certificate[3]):
                self.assertEqual(pow(witness, n - 1, n), 1)
                self.assertNotEqual(pow(witness, (n - 1) // p, n), 1)


class CostModelTest(unittest.TestCase):
    """The cost model of the prime proofs: pure functions of sizes and evidence, which the checker charges and the
    producers compute; tests/v09/certify/test_factorization.py holds both sides to them."""

    PROTH = 3 * 2 ** 534 + 1  # 536 bits; its n - 1 is 2**534 * 3.

    def proth(self, witnesses):
        """A certificate of PROTH's shape with these witnesses for 2 and 3; the cost model never asks if it holds."""
        return ("pocklington", self.PROTH, ((2, 534, ("trial", 2)), (3, 1, ("trial", 3))), witnesses)

    def test_unit_costs(self):
        work = trusted_integers.multiplication_work
        self.assertEqual([work(bits) for bits in (0, 64, 383, 384, 1_024, 4_096, 65_536)], [1, 1, 1, 2, 9, 129, 32_769])
        for left, right in ((0, 0), (64, 65_536), (1_024, 1_024), (65_536, 65_536), (100_000, 3_000)):
            self.assertEqual(trusted_integers.product_work(left, right), upoly.size_factor(left, right) - 1)
        self.assertEqual(trusted_integers.product_work(65_536, 65_536), 4_096)
        self.assertEqual([trusted_integers.division_work(dividend, divisor) for dividend, divisor in
                          ((63, 10 ** 6), (65_536, 1), (65_536, 65_536), (4_096, 2_048))], [0, 32, 32_800, 66])

    def test_prime_proof_work_follows_the_ranges_of_prime_proof(self):
        work, limit = trusted_integers.prime_proof_work, 10 ** 9
        for n in (-7, 0, 1, 2, 9_973, True, 7.0, "10007", None):  # The trial criterion needs no budget.
            self.assertEqual(work(n, (), limit), 0, n)
        for n in (10_007, 2 ** 61 - 1, primes.DETERMINISTIC_BELOW - 1):  # Thirteen strong tests, one unit a bit.
            self.assertEqual(work(n, (), limit), 13 * n.bit_length())
            self.assertEqual(work(n, ("trial", n), limit), 0)  # prime_proof refuses it by its shape alone.
            self.assertEqual(work(n, [], limit), 0)
        sound = self.proth((5, 5))
        for certificate in ((), ("trial", self.PROTH), list(sound), ("Pocklington",) + sound[1:], sound[:3]):
            self.assertEqual(work(self.PROTH, certificate, limit), 0)
        self.assertGreater(work(self.PROTH, sound, limit), 0)

    def test_the_charge_of_a_pocklington_certificate(self):
        """For PROTH: 536 bits, two primes (one level of product tree) and g distinct witnesses give
        536 * (g + 2) + 2 * (g + 2) bits of exponents at 3 units a multiplication, two gcds of 2 units, a unit for the
        node and one for each of its two ("trial", p) children."""
        work = trusted_integers.prime_proof_work
        self.assertEqual(trusted_integers.multiplication_work(536), 3)
        self.assertEqual(work(self.PROTH, self.proth((5, 5)), 10 ** 9), 1 + 538 * 3 * 3 + 2 * 2 + 2)
        self.assertEqual(work(self.PROTH, self.proth((5, 7)), 10 ** 9), 1 + 538 * 4 * 3 + 2 * 2 + 2)
        self.assertEqual(work(self.PROTH, self.proth((5, 7)), 10 ** 9), 6_463)
        # Past the limit the walk stops: its result is only "more than the limit".
        self.assertGreater(work(self.PROTH, self.proth((5, 7)), 100), 100)

    def test_the_walk_stops_at_check_primes_depth(self):
        """A chain of Pocklington nodes of one 82-bit number, each with one factor (exponents of 2 * 82 + 2 bits, a
        gcd, the node: 168 units): check_prime examines MAX_DEPTH + 1 levels of it and refuses the next node unread,
        and so does the walk, which charges that node one unit."""
        n = primes.DETERMINISTIC_BELOW + 142

        def chain(length):
            node = ("trial", 2)
            for _ in range(length):
                node = ("pocklington", n, ((n, 1, node),), (3,))
            return node

        self.assertEqual(trusted_integers.prime_proof_work(n, chain(50), 10 ** 9), 50 * 168 + 1)
        for length in (MAX_DEPTH + 1, MAX_DEPTH + 2, 300):
            self.assertEqual(trusted_integers.prime_proof_work(n, chain(length), 10 ** 9), (MAX_DEPTH + 1) * 168 + 1)

    def test_shared_subtrees_cost_at_most_the_limit(self):
        """83 factors per node, more than the 82 bits of the number: no exponentiation, a unit per node, and about
        83**129 nodes to walk. The walk stops one unit past its limit."""
        n = primes.DETERMINISTIC_BELOW + 142
        node: tuple = ("trial", 2)
        for _ in range(MAX_DEPTH + 2):
            node = ("pocklington", n, ((n, 1, node),) * 83, (3,) * 83)
        start = time.perf_counter()
        for limit in (0, 1, 1_000, 100_000):
            self.assertEqual(trusted_integers.prime_proof_work(n, node, limit), limit + 1)
        self.assertLess(time.perf_counter() - start, 0.25)

    def test_factorization_work(self):
        claimed, certificate = kernel.prime_certificate(2 ** 89 - 1)
        self.assertTrue(claimed)
        proof = trusted_integers.prime_proof_work(2 ** 89 - 1, certificate, 10 ** 9)
        self.assertGreater(proof, 0)
        factors = ((2, 3, ()), (3, 2, ()), (10_007, 1, ()), (2 ** 89 - 1, 1, certificate))
        # Four entries of one unit (all far below 2,048 bits), no proof for 2 and 3, the strong tests of 10,007.
        self.assertEqual(trusted_integers.factorization_work(factors, 10 ** 9), 4 + 13 * 14 + proof)
        self.assertEqual(trusted_integers.factorization_work((), 10 ** 9), 0)
        # One entry of 82,000 bits: its last squaring, two 41,000-bit halves, is 640 * 640 / 256 units.
        self.assertEqual(trusted_integers.factorization_entry_work(1, 82_000), 1 + 1_600)
        self.assertEqual(trusted_integers.factorization_work(((3, 41_000, ()),), 10 ** 9), 1 + 1_600)
        self.assertGreater(trusted_integers.factorization_work(factors, 100), 100)
        # Past the limit no power is built: 2**(10**100) would never be.
        start = time.perf_counter()
        self.assertGreater(trusted_integers.factorization_work(((2, 10 ** 100, ()),), 10 ** 6), 10 ** 6)
        self.assertLess(time.perf_counter() - start, 0.25)

    def test_the_functions_are_pure(self):
        """No budget is needed, none is charged, and limits far beyond any budget are allowed."""
        claimed, certificate = kernel.prime_certificate(self.PROTH)
        self.assertTrue(claimed)
        with computation() as budget:
            values = (trusted_integers.multiplication_work(65_536), trusted_integers.division_work(65_536, 64),
                      trusted_integers.prime_proof_work(self.PROTH, certificate, 10 ** 12),
                      trusted_integers.factorization_work(((self.PROTH, 1, certificate),), 10 ** 12))
            self.assertEqual((budget.work, budget.node_count), (0, 0))
        self.assertEqual(values[3], 1 + values[2])
        self.assertEqual(trusted_integers.prime_proof_work(self.PROTH, certificate, 10 ** 12), values[2])


if __name__ == "__main__":
    unittest.main()
