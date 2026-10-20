"""Inputs within the default limits whose intermediates explode (final review B, C1 a, c and g): the lean functions
fall back and the structured calls refuse with resource_limit_exceeded, each within a second, where they ran for
minutes. The inputs are the final review's repro inputs; building them is not timed."""

from __future__ import annotations

import random
import time
import unittest

import mathforge as mf
from mathforge import lean, operations, polynomial_exact
from mathforge.certify import upoly as certify
from mathforge.errors import OperationError
from mathforge.limits import ResourceLimitError, computation
from mathforge.model import Add, Pow, Rational, Sqrt
from mathforge.polynomial import Polynomial

X = mf.Symbol("x")
SECONDS = 1.0


class QuickRefusals:
    def assertRefusedQuickly(self, call, context=None):
        """call() returns FALLBACK or raises resource_limit_exceeded, within SECONDS."""
        start = time.perf_counter()
        try:
            value = call()
        except (ResourceLimitError, OperationError) as error:
            self.assertEqual(error.code, "resource_limit_exceeded", context)
        else:
            self.assertIs(value, lean.FALLBACK, context)
        self.assertLessEqual(time.perf_counter() - start, SECONDS, context)

    def assertQuick(self, call, context=None):
        """call() finishes within SECONDS, whatever it answers."""
        start = time.perf_counter()
        try:
            call()
        except (ResourceLimitError, OperationError):
            pass
        self.assertLessEqual(time.perf_counter() - start, SECONDS, context)


def polynomial_pair(degree, bits):
    rng = random.Random(5)
    return tuple(Polynomial(tuple(Rational(rng.getrandbits(bits) | 1) for _ in range(degree + 1)), X)
                 for _ in range(2))


def nested(levels, exponent):
    node = Add((Rational(1), Sqrt(Rational(2))))
    for _ in range(levels):
        node = Pow(node, exponent)
    return node


class ExplosiveIntermediatesTest(QuickRefusals, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = {degree: polynomial_pair(degree, 60_000) for degree in (128, 512)}
        rng = random.Random(5)
        cls.coefficients = tuple(rng.getrandbits(60_000) | 1 for _ in range(33))
        cls.polynomial = Polynomial(tuple(map(Rational, cls.coefficients)), X)
        big = (1 << 60_000) + 1
        cls.sparse = Polynomial((Rational(-big),) + (Rational(0),) * 999 + (Rational(1),), X)  # x**1000 - big

    def test_poly_gcd_of_long_coefficients(self):
        """Kronecker values of millions of bits (heuristic gcd); Euclid refuses its first long divisor."""
        for degree, (f, g) in self.pairs.items():
            self.assertRefusedQuickly(lambda: lean.functions.poly_gcd(f, g), degree)
            self.assertRefusedQuickly(lambda: operations.poly_gcd(f, g).unwrap(), degree)

    def test_square_free_parts_of_long_coefficients(self):
        """mathforge.RealAlgebraicRoot and real_roots take the square-free part through the same gcd."""
        self.assertRefusedQuickly(lambda: mf.RealAlgebraicRoot(self.coefficients, 0))
        self.assertRefusedQuickly(lambda: lean.functions.real_roots(self.polynomial, X))
        self.assertRefusedQuickly(lambda: operations.real_roots(self.polynomial, X).unwrap())

    def test_isolation_from_a_long_root_bound(self):
        """x**1000 - (2**60000 + 1): bisection from a root bound of 60,000 bits evaluates the Sturm chain at points
        whose powers have millions of bits."""
        self.assertRefusedQuickly(lambda: lean.functions.real_roots(self.sparse, X))
        self.assertRefusedQuickly(lambda: operations.real_roots(self.sparse, X).unwrap())
        self.assertQuick(lambda: lean.functions.square_free(self.sparse))  # The v0.7 path answers it at once.

    def test_compare_real_of_nested_powers(self):
        """((1 + sqrt 2)**4096)**4096: fixed-point enclosures of 21 million bits."""
        left = nested(2, 4096)
        self.assertRefusedQuickly(lambda: lean.functions.compare_real(left, Rational(0)))
        self.assertRefusedQuickly(lambda: operations.compare_real(left, Rational(0)).unwrap())

    def test_irreducibility_evidence_with_a_large_prime(self):
        """Forged evidence ("modular", 2**61 - 1, ...) reaches certify.upoly.check_irreducible through the check of a
        user record: trial division of the prime, and its powers, are charged as v0.7 charged them."""
        poly = mf.Polynomial((-2, 0, 1), X)
        prime = (1 << 61) - 1
        for evidence in (("modular", prime, 1, ((1, 1), (prime - 1, 1))),  # Trial division up to 2**30.5.
                         ("modular", 3, 50_000, ((1, 1), (2, 1))),  # 3**50000 has 79,249 bits.
                         ("modular", 2_147_483_647, 4_000, ((1, 1), (2, 1)))):  # Refused before the power.
            for check in (polynomial_exact._check_irreducible, polynomial_exact._check_irreducible_v07):
                with self.subTest(evidence=evidence[:3], check=check.__name__), computation():
                    self.assertRefusedQuickly(lambda: check(poly, evidence))

    def test_checker_chains_refuse_long_remainders(self):
        """certify.upoly's Euclid and subresultant Sturm chain refuse a long member before dividing by it."""
        f = self.coefficients
        rng = random.Random(6)
        g = tuple(rng.getrandbits(60_000) | 1 for _ in range(32))
        for label, call in (("euclid_gcd", lambda: certify.euclid_gcd(f, g)),
                            ("sturm_chain", lambda: certify.sturm_chain(f)),
                            ("count_roots", lambda: certify.count_roots(certify.squarefree_part(f), None, None))):
            with self.subTest(label), computation():
                self.assertRefusedQuickly(call, label)


if __name__ == "__main__":
    unittest.main()
