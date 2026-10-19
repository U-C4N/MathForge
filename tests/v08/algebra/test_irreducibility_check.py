"""The checker's irreducibility test (certify.upoly.check_irreducible) agrees with v0.7's
polynomial_exact._check_irreducible, and the lean factor equals the structured one."""

import random
import unittest

import mathforge as mf
from mathforge import lean, operations, polynomial_exact
from mathforge.certify import upoly as certify
from mathforge.kernels import upoly as kernels
from mathforge.limits import computation
from mathforge.model import Rational
from mathforge.trusted import upoly as tp

X = mf.Symbol("x", uid="00000000-0000-4000-8000-0000000000f1")


def random_product(rng):
    factors = []
    for _ in range(rng.randint(1, 4)):
        degree = rng.randint(1, 5)
        values = [rng.randint(-9, 9) for _ in range(degree)] + [rng.choice((1, 2, -3, 5))]
        factors.append(mf.Polynomial(tuple(values), X) ** rng.choice((1, 1, 2)))
    result = factors[0]
    for factor in factors[1:]:
        result = result * factor
    return result


def evidences(count=200):
    rng = random.Random(9501)
    found = []
    while len(found) < count:
        polynomial = random_product(rng)
        if polynomial.degree < 1:
            continue
        value, evidence = polynomial_exact.factor_with_evidence(polynomial)
        for entry, proof in zip(value.factors, evidence[1]):
            found.append((entry.polynomial, proof))
    return found[:count]


def tampered(proof, rng):
    """Variants of a modular evidence that must fail."""
    if proof == ("linear",):
        return [("modular", 3, 2, ((1, 1),)), ()]
    _, p, exponent, factors = proof
    modulus = p ** exponent
    variants = [("modular", p, exponent - 1, factors), ("modular", p + 1, exponent, factors),
                ("modular", p, exponent, factors[1:] or factors + factors),
                ("modular", p, 0, factors), ("linear",), ("modular", p, exponent, ())]
    first = list(factors[0])
    first[0] = (first[0] + 1) % modulus
    variants.append(("modular", p, exponent, (tuple(first),) + factors[1:]))
    return variants


class IrreducibilityCheckTest(unittest.TestCase):
    def test_agrees_with_v07(self):
        rng = random.Random(9502)
        refused = 0
        for polynomial, proof in evidences():
            integer = tp.primitive(tp.from_rationals(polynomial.coefficients)[0])
            with computation():
                expected = polynomial_exact._check_irreducible(polynomial, proof)
            self.assertIs(expected, True)
            self.assertIs(certify.check_irreducible(integer, proof), True, (polynomial, proof))
            for wrong in tampered(proof, rng):
                with computation():
                    v07 = polynomial_exact._check_irreducible(polynomial, wrong)
                self.assertEqual(certify.check_irreducible(integer, wrong), v07, (polynomial, wrong))
                refused += v07 is False
        self.assertGreater(refused, 900)  # A lower exponent can still be enough; everything else is refused.

    def test_reducible_polynomials_are_refused(self):
        # (x**2 - 2)(x**2 - 3) with the modular factors of an irreducible quartic's evidence shape: refused.
        reducible = tp.mul((-2, 0, 1), (-3, 0, 1))
        p, exponent, _, lifted = kernels.lift_data(reducible)
        self.assertFalse(certify.check_irreducible(reducible, ("modular", p, exponent, tuple(lifted))))

    def test_kernel_evidence_is_accepted(self):
        rng = random.Random(9503)
        for _ in range(60):
            polynomial = random_product(rng)
            integer = tp.primitive(tp.from_rationals(polynomial.coefficients)[0])
            for part, _ in kernels.square_free_integer(integer):
                for factor, evidence in kernels.factor_square_free(part):
                    self.assertTrue(certify.check_irreducible(factor, evidence), (factor, evidence))

    def test_lean_factor_equals_the_structured_path(self):
        lean._clear_caches()
        rng = random.Random(9504)
        for _ in range(80):
            polynomial = random_product(rng)
            if rng.random() < 0.3:
                polynomial = polynomial * Rational(rng.randint(1, 9), rng.randint(1, 9))
            value = lean.functions.factor(polynomial)
            expected = operations.factor(polynomial).unwrap()
            self.assertIsNot(value, lean.FALLBACK)
            self.assertEqual(value, expected)
            self.assertEqual(mf.to_data(value), mf.to_data(expected))


if __name__ == "__main__":
    unittest.main()
