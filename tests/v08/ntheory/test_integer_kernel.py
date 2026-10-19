"""The integer kernel (mathforge.kernels.integers), the shared powers (mathforge.trusted.integers) and the
prime-certificate checker (mathforge.certify.ntheory.primes)."""

import json
from math import gcd, isqrt, prod
from pathlib import Path
import random
import unittest

import mathforge as mf
from mathforge import number_theory
from mathforge.certify.ntheory.primes import check_prime, trial_prime
from mathforge.kernels import integers as kernel
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.serialization import to_data
from mathforge.trusted.integers import powers_by_prime

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "v8" / "values_v07" / "ntheory.json"


def trial_factor(n):
    """The factorization of n >= 1 by trial division: the oracle where it is cheap."""
    pairs, divisor = [], 2
    while divisor * divisor <= n:
        exponent = 0
        while n % divisor == 0:
            n //= divisor
            exponent += 1
        if exponent:
            pairs.append((divisor, exponent))
        divisor += 1 if divisor == 2 else 2
    if n > 1:
        pairs.append((n, 1))
    return tuple(pairs)


def sieve(limit):
    flags = bytearray([1]) * limit
    flags[0:2] = b"\0\0"
    for p in range(2, isqrt(limit - 1) + 1):
        if flags[p]:
            flags[p * p::p] = bytes(len(range(p * p, limit, p)))
    return flags


def v07_certificate(n):
    with computation():
        return number_theory._prime_certificate(n)


def v07_check(n, claimed, certificate):
    with computation():
        return number_theory._check_prime(n, claimed, certificate)


def random_prime(rng, low, high):
    while True:
        candidate = rng.randrange(low, high) | 1
        if kernel.is_probable_prime(candidate) and kernel.prime_certificate(candidate)[0]:
            return candidate


class IntegerKernelTest(unittest.TestCase):
    def test_tables(self):
        flags = sieve(1 << 16)
        self.assertEqual(kernel.SMALL_PRIMES, tuple(p for p in range(1024) if flags[p]))
        self.assertEqual(len(kernel.SMALL_PRIMES), 172)
        table = kernel.smallest_factor_table()
        for m in range(1 << 16):
            expected = 0 if m < 2 or flags[m] else next(p for p in range(2, m) if m % p == 0)
            self.assertEqual(table[m], expected, m)

    def test_factor_matches_v07(self):
        rng = random.Random(20261019)
        for index in range(5000):
            n = rng.randrange(1, 10 ** rng.randrange(1, 19))
            pairs = kernel.factor(n)
            with self.subTest(n=n):
                if n < 10 ** 10:
                    self.assertEqual(pairs, trial_factor(n))
                self.assertEqual(prod(p ** e for p, e in pairs), n)
                self.assertEqual([p for p, _ in pairs], sorted({p for p, _ in pairs}))
                for prime, _ in pairs:
                    claimed, certificate = kernel.prime_certificate(prime)
                    self.assertTrue(claimed and check_prime(prime, True, certificate))
            if index % 50 == 0:  # The structured path (now on the kernel) and v0.7's checker agree.
                expected = mf.operations.factor_integer(n).unwrap()
                self.assertEqual(expected.factors, pairs)
        self.assertEqual(kernel.factor(1), ())
        for bad in (0, -5, True, 2.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kernel.factor(bad)

    def test_factor_matches_the_corpus(self):
        stored = json.loads(CORPUS.read_text(encoding="utf-8"))
        inputs = {"factor_integer/micro/small_numbers#0": 360, "factor_integer/micro/small_numbers#1": 361,
                  "factor_integer/micro/small_numbers#2": 362, "factor_integer/realistic/euler3#0": 600851475143,
                  "factor_integer/realistic/semiprime_12_digits#0": 999983 * 1000003,
                  "factor_integer/edge/one": 1, "factor_integer/edge/minus_one": -1,
                  "factor_integer/edge/negative": -360, "factor_integer/edge/prime": 97,
                  "factor_integer/edge/prime_power": 1 << 20, "factor_integer/edge/square_of_prime": 9409,
                  "factor_integer/edge/two_primes": 91}
        for key, n in inputs.items():
            with self.subTest(key=key):
                self.assertIsNone(stored[key]["error"])
                value = mf.IntegerFactorization(-1 if n < 0 else 1, kernel.factor(abs(n)))
                self.assertEqual(to_data(value), stored[key]["value"])

    def test_semiprimes(self):
        rng = random.Random(7)
        for _ in range(60):
            p = random_prime(rng, 10 ** 3, 10 ** 9)
            q = random_prime(rng, max(10 ** 12 // p, 2), 10 ** 24 // p)
            n = p * q
            self.assertTrue(10 ** 12 <= n <= 10 ** 24 or p == q)
            self.assertEqual(kernel.factor(n), tuple(sorted(((p, 1), (q, 1)))) if p != q else ((p, 2),))
        for p, q in ((999983, 1000003), (1000003, 1000033), (10 ** 9 + 7, 10 ** 9 + 9)):
            self.assertEqual(kernel.factor(p * q), ((p, 1), (q, 1)))
        for p in (1031, 65537, 1000003):  # Prime powers.
            for exponent in range(2, 7):
                self.assertEqual(kernel.factor(p ** exponent), ((p, exponent),))

    def test_work_is_bounded(self):
        hard = (10 ** 12 + 39) * (10 ** 12 + 61)  # Two primes near 10**12: rho needs about a million steps.
        with computation(ComputationLimits(max_work=20_000)) as budget, self.assertRaises(ResourceLimitError):
            kernel.factor(hard)
        self.assertGreater(budget.work, 0)
        prime = next(k * hard + 1 for k in range(2, 10_000, 2) if kernel.is_probable_prime(k * hard + 1))
        with computation(ComputationLimits(max_work=20_000)), self.assertRaises(ResourceLimitError):
            kernel.prime_certificate(prime)  # prime - 1 = k * hard: rho would need about a million steps.
        with computation() as budget:  # Batched charges: far below v0.7's one tick per trial divisor.
            kernel.factor(600851475143)
        self.assertLess(budget.work, 2_000)

    def test_is_probable_prime(self):
        flags = sieve(200_000)
        for n in range(200_000):
            self.assertEqual(kernel.is_probable_prime(n), bool(flags[n]), n)
        pseudoprimes = (2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633, 65281, 74665, 80581,
                        85489, 88357, 90751, 561, 1105, 1729, 2465, 2821, 6601, 8911, 3215031751, 2152302898747,
                        3474749660383, 341550071728321, 3825123056546413051, 318665857834031151167461,
                        3317044064679887385961981)
        for n in pseudoprimes:
            self.assertFalse(kernel.is_probable_prime(n), n)
        for n in (2 ** 61 - 1, 2 ** 89 - 1, 2 ** 107 - 1, 2 ** 127 - 1, 3224192153648326577750017, 10 ** 9 + 7):
            self.assertTrue(kernel.is_probable_prime(n), n)

    def test_prime_certificates(self):
        rng = random.Random(3)
        cases = [*range(-3, 3000), *(rng.randrange(10 ** 4, 10 ** 9) for _ in range(1500)),
                 *(rng.randrange(10 ** 9, 10 ** 18) for _ in range(300)), 65537, 2 ** 61 - 1,
                 3224192153648326577750017, 561, 41041, 3215031751, 3825123056546413051]
        for n in cases:
            claimed, certificate = kernel.prime_certificate(n)
            with self.subTest(n=n):
                self.assertEqual(claimed, n >= 2 and (n < 10 ** 10 and trial_factor(n) == ((n, 1),)
                                                      or n >= 10 ** 10 and kernel.factor(n) == ((n, 1),)))
                self.assertTrue(check_prime(n, claimed, certificate))
                self.assertTrue(v07_check(n, claimed, certificate))
                if n < 10 ** 6:  # v0.7 reaches the same verdict.
                    self.assertEqual(v07_certificate(n)[0], claimed)

    def test_trial_criterion(self):
        for n in range(2, 10_000):
            expected = all(n % d for d in range(2, isqrt(n) + 1))
            self.assertEqual(trial_prime(n), expected, n)

    def test_checker_agrees_with_v07_and_refuses_forgeries(self):
        claimed, good = kernel.prime_certificate(2 ** 61 - 1)
        self.assertTrue(claimed and check_prime(2 ** 61 - 1, True, good))
        n = 2 ** 61 - 1
        kind, recorded, factors, witnesses = good
        forgeries = [
            (n, False, good), (n + 2, True, good), (n, True, ("pocklington", n, factors, witnesses[:-1])),
            (n, True, ("pocklington", n, factors[1:], witnesses[1:])),
            (n, True, ("pocklington", n, factors[::-1], witnesses[::-1])),
            (n, True, ("pocklington", n, factors, (1,) * len(witnesses))),
            (n, True, ("pocklington", n, factors, (n,) * len(witnesses))),
            (n, True, ("pocklington", n, ((2, 10 ** 9, ("trial", 2)),), (3,))),
            (n, True, ("pocklington", n, ((2 ** 89 - 1, 10 ** 6, ("trial", 2 ** 89 - 1)),), (3,))),
            (n, True, ("pocklington", n, ((factors[0][0], factors[0][1], ("trial", 4)), *factors[1:]), witnesses)),
            (n, True, ("trial", n)), (97, True, ("trial", 97, 0)), (91, True, ("trial", 91)),
            (91, False, ("factor", 91, 91)), (91, False, ("factor", 91, 1)), (91, False, ("factor", 91, 6)),
            (97, False, ("fermat", 97, 2)), (2, False, ("fermat", 2, 1)), (5, False, ("below", 5)),
            (1, True, ("below", 1)), (n, True, ["pocklington", n, factors, witnesses]), (n, 1, good),
            (n, True, ("pocklington", float(n), factors, witnesses)), (n, True, ()),
        ]
        for number, claim, certificate in forgeries:
            with self.subTest(number=number, claim=claim, certificate=str(certificate)[:60]):
                self.assertFalse(check_prime(number, claim, certificate))
        # A composite Pocklington claim: a valid-looking certificate of a Carmichael number fails.
        carmichael = 41041
        self.assertFalse(check_prime(carmichael, True, ("pocklington", carmichael,
                                                        ((2, 4, ("trial", 2)), (3, 3, ("trial", 3)),
                                                         (5, 1, ("trial", 5)), (19, 1, ("trial", 19))),
                                                        (3, 3, 3, 3))))
        # Nesting deeper than max_depth is refused, as v0.7 refuses it.
        three = ("pocklington", 3, ((2, 1, ("trial", 2)),), (2,))
        self.assertTrue(check_prime(3, True, three, depth=127))
        self.assertFalse(check_prime(3, True, three, depth=128))  # Its child would sit at depth 129.
        self.assertTrue(check_prime(2, True, ("trial", 2), depth=128))
        self.assertFalse(check_prime(2, True, ("trial", 2), depth=129))
        # Mutations of valid certificates: the checker and v0.7's agree.
        rng = random.Random(11)
        for _ in range(300):
            number = rng.randrange(10 ** 4, 10 ** 12)
            claim, certificate = kernel.prime_certificate(number)
            if certificate[0] != "pocklington":
                continue
            mutated = list(certificate)
            spot = rng.randrange(1, 4)
            if spot == 1:
                mutated[1] = number + rng.choice((-2, 2))
            elif spot == 2:
                pairs = list(mutated[2])
                index = rng.randrange(len(pairs))
                prime, exponent, child = pairs[index]
                pairs[index] = (prime, exponent + rng.choice((-1, 1)), child)
                mutated[2] = tuple(pairs)
            else:
                mutated[3] = tuple(w + rng.randrange(0, 3) for w in mutated[3])
            mutated = tuple(mutated)
            self.assertEqual(check_prime(number, claim, mutated), v07_check(number, claim, mutated), mutated)

    def test_inverse(self):
        rng = random.Random(9)
        for bits in (8, 64, 1000, 5999, 6000, 6001, 9000, 20000):
            for _ in range(30 if bits < 9000 else 4):
                m = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
                a = rng.getrandbits(bits + rng.randrange(-5, 6)) * rng.choice((1, -1))
                try:
                    expected = pow(a, -1, m)
                except ValueError:
                    with self.assertRaises(ValueError):
                        kernel.inverse(a, m)
                    continue
                self.assertEqual(kernel.inverse(a, m), expected, (bits, a, m))
        m = (1 << 7000) + 1
        for a in (0, m, 3 * ((1 << 7000) + 1), 2 * m + 1, -1, 1):
            try:
                expected = pow(a, -1, m)
            except ValueError:
                expected = ValueError
            if expected is ValueError:
                with self.assertRaises(ValueError):
                    kernel.inverse(a, m)
            else:
                self.assertEqual(kernel.inverse(a, m), expected)
        self.assertEqual(kernel.inverse(5, 1), 0)

    def test_powers_by_prime(self):
        rng = random.Random(5)
        primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)
        for _ in range(2000):
            chosen = sorted(rng.sample(primes, rng.randrange(1, 10)))
            exponent = prod(p ** rng.randrange(1, 4) for p in chosen) * rng.randrange(1, 1000)
            modulus = rng.randrange(2, 10 ** 30)
            base = rng.randrange(0, modulus)
            self.assertEqual(powers_by_prime(base, exponent, modulus, chosen),
                             {p: pow(base, exponent // p, modulus) for p in chosen})


if __name__ == "__main__":
    unittest.main()
