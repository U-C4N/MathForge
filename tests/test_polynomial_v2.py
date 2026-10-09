"""P02 independent polynomial invariants and exact zero conventions."""

from dataclasses import replace
import random
import unittest

import mathforge as mf
from mathforge.errors import InvalidInput, MathForgeError, UnsupportedOperation
from mathforge.limits import ComputationLimits
from mathforge.model import Rational, symbol
from mathforge.polynomial import Polynomial, SquareFreeDecomposition, SquareFreeFactor


class PolynomialV2Tests(unittest.TestCase):
    def setUp(self):
        self.x = symbol("x")
        self.zero = Polynomial((0,), self.x)
        self.one = Polynomial((1,), self.x)

    def p(self, *coefficients):
        return Polynomial(coefficients, self.x)

    def test_arithmetic_and_exact_long_division(self):
        """POL-01–05: rational arithmetic, lower degree, zero dividend."""
        a, b = self.p(1, 2, 3), self.p(-1, 1)
        q, r = divmod(a, b)
        self.assertEqual(q, self.p(5, 3))
        self.assertEqual(r, self.p(6))
        self.assertEqual(q*b+r, a)
        self.assertEqual(divmod(b, a), (self.zero, b))
        self.assertEqual(divmod(self.zero, a), (self.zero, self.zero))
        self.assertEqual((a*b).exact_div(b), a)
        self.assertEqual(a + 2, 2 + a)
        self.assertEqual(2 - a, -(a - 2))
        self.assertEqual((a / Rational(2, 3)) * Rational(2, 3), a)

    def test_invalid_division_and_nonexact_division(self):
        """POL-06/07: failures retain mathematical meaning."""
        with self.assertRaises(InvalidInput):
            divmod(self.one, self.zero)
        with self.assertRaises(InvalidInput) as captured:
            self.p(1, 0, 1).exact_div(self.p(1, 1))
        self.assertEqual(captured.exception.code, "non_exact_division")
        with self.assertRaises(InvalidInput):
            self.one / 0

    def test_gcd_xgcd_zero_and_nonmonic(self):
        """POL-08–10: monic gcd plus exact Bézout certificates."""
        a, b = self.p(-2, 0, 2), self.p(-6, 6)
        g, s, t = a.xgcd(b)
        self.assertEqual(g, self.p(-1, 1))
        self.assertEqual(s*a+t*b, g)
        self.assertEqual(a.gcd(b), g)
        self.assertEqual(self.zero.gcd(a), a.monic())
        self.assertEqual(a.gcd(self.zero), a.monic())
        self.assertEqual(self.zero.xgcd(self.zero), (self.zero,)*3)
        self.assertEqual(self.zero.gcd(self.zero), self.zero)
        self.assertEqual(self.p(-4).gcd(a), self.one)

    def test_square_free_multiplicity_and_zero_convention(self):
        """POL-11/12: grouped multiplicities, coefficient, zero convention."""
        a, b = self.p(-1, 1), self.p(2, 1)
        result = (6*a**4*b**3).square_free()
        self.assertEqual(result, SquareFreeDecomposition(Rational(6), (
            SquareFreeFactor(b, 3), SquareFreeFactor(a, 4))))
        self.assertEqual(self.zero.square_free(), SquareFreeDecomposition(0, ()))
        self.assertEqual(self.p(-7).square_free(), SquareFreeDecomposition(-7, ()))
        self.assertEqual((a*b).square_free().factors, (SquareFreeFactor(a*b, 1),))

    def test_mixed_variable_and_invalid_exponents(self):
        """POL-14/15: symbol identity and finite dense degree scope."""
        other = Polynomial((1,), symbol("x"))
        for callback in (lambda: self.one + other, lambda: self.one.gcd(other),
                         lambda: self.one.exact_div(other)):
            with self.assertRaises(InvalidInput):
                callback()
        for exponent in (-1, True, Rational(1), 0.5):
            with self.assertRaises(InvalidInput):
                self.one ** exponent
        self.assertEqual(self.zero**0, self.one)
        with self.assertRaises(UnsupportedOperation):
            self.p(0, 1)**1025

    def test_decomposition_shape_validation(self):
        """POL-13: mutable/ambiguous factor records cannot enter certificates."""
        factor = self.p(1, 1)
        for multiplicity in (0, -1, True, 1.0):
            with self.assertRaises(InvalidInput):
                SquareFreeFactor(factor, multiplicity)
        with self.assertRaises(InvalidInput):
            SquareFreeFactor(2*factor, 1)
        with self.assertRaises(InvalidInput):
            SquareFreeDecomposition(0, (SquareFreeFactor(factor, 1),))

    def test_seeded_division_and_bezout_invariants(self):
        """POL-01/POL-09: 100 independently constructed exact identities."""
        rng = random.Random(2026100902)
        for case in range(100):
            with self.subTest(case=case):
                divisor = self.p(*(Rational(rng.randrange(-8, 9), rng.randrange(1, 6))
                                   for _ in range(rng.randrange(1, 5))), 1)
                quotient = self.p(*(rng.randrange(-8, 9) for _ in range(rng.randrange(1, 5))))
                remainder = self.p(*(rng.randrange(-8, 9) for _ in range(divisor.degree)))
                dividend = quotient*divisor+remainder
                self.assertEqual(divmod(dividend, divisor), (quotient, remainder))
                g, s, t = dividend.xgcd(divisor)
                self.assertEqual(s*dividend+t*divisor, g)
                self.assertTrue(dividend.divmod(g)[1].is_zero)
                self.assertTrue(divisor.divmod(g)[1].is_zero)

    def test_limits_are_shared_by_nested_polynomial_work(self):
        """LIM-04: work fails honestly under a tiny operation budget."""
        with self.assertRaises(MathForgeError) as captured:
            self.p(1, 2, 3, 4).xgcd(self.p(-1, 2, 1), limits=ComputationLimits(max_work=2))
        self.assertEqual(captured.exception.code, "resource_limit_exceeded")

    def test_degree_1024_boundary_does_not_break_zero_or_exact_division(self):
        """POL-15: the existing representation boundary remains usable."""
        monomial = self.p(0, 1)**1024
        self.assertEqual(monomial.degree, 1024)
        self.assertEqual((monomial*self.zero), self.zero)
        quotient = monomial.exact_div(self.p(0, 1))
        self.assertEqual(quotient.degree, 1023)
        self.assertEqual(quotient*self.p(0, 1), monomial)
        self.assertEqual(monomial.derivative().degree, 1023)

    def test_bit_growth_limit_is_checked_before_new_arithmetic(self):
        """LIM-03: exact input size is distinct from operation resource scope."""
        large = self.p(1 << 40, 1)
        with self.assertRaises(MathForgeError) as captured:
            large.gcd(self.p(1, 1), limits=ComputationLimits(max_integer_bits=16))
        self.assertEqual(captured.exception.code, "resource_limit_exceeded")

    def test_expand_operation_and_structural_expression_equality(self):
        """POL-16: expansion is explicit and does not redefine AST equality."""
        expression = (self.x+1)**3
        expected = self.x**3+3*self.x**2+3*self.x+1
        self.assertNotEqual(expression, expected)
        result = mf.operations.expand(expression, self.x)
        self.assertEqual(result.value, expected)
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(mf.expand(expression, self.x), expected)

    def test_correct_reconstruction_does_not_excuse_non_square_free_factors(self):
        """POL-13: factor square-freeness and pairwise coprimality are necessary."""
        a, b = self.p(-1, 1), self.p(2, 1)
        result = mf.operations.square_free(a**2*b**3)
        repeated_inside = SquareFreeDecomposition(1, (
            SquareFreeFactor(a**2, 1), SquareFreeFactor(b, 3)))
        self.assertFalse(mf.verify(replace(result, value=repeated_inside)).verified)
        result = mf.operations.square_free(a**3)
        shared = SquareFreeDecomposition(1, (SquareFreeFactor(a, 1), SquareFreeFactor(a, 2)))
        self.assertFalse(mf.verify(replace(result, value=shared)).verified)


if __name__ == "__main__":
    unittest.main()
