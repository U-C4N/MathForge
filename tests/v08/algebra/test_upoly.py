"""The trusted dense integer polynomials (mathforge.trusted.upoly) against mathforge.polynomial.Polynomial, and the
lean polynomial calculus and division paths against the structured path."""

import random
import unittest

import mathforge as mf
from mathforge import lean, operations
from mathforge.certify import upoly as certify
from mathforge.kernels import upoly as kernels
from mathforge.model import Rational
from mathforge.polynomial import Polynomial, trusted_polynomial
from mathforge.trusted import upoly as tp

X = mf.Symbol("x", uid="00000000-0000-4000-8000-0000000000a1")


def as_polynomial(values):
    return Polynomial(tuple(values) or (0,), X)


def integer_tuple(polynomial):
    values = []
    for coefficient in polynomial.coefficients:
        assert coefficient.denominator == 1
        values.append(coefficient.numerator)
    return tp.trim(values)


def random_integers(rng, degree, bits=8):
    values = [rng.randint(-2**bits, 2**bits) for _ in range(degree + 1)]
    if degree >= 0 and not values[-1]:
        values[-1] = 1
    return tuple(values)


def random_polynomial(rng, degree, *, rational=False, bits=8):
    values = []
    for _ in range(degree + 1):
        numerator = rng.randint(-2**bits, 2**bits)
        values.append(Rational(numerator, rng.randint(1, 9)) if rational else Rational(numerator))
    if not values[-1].numerator:
        values[-1] = Rational(1)
    return Polynomial(tuple(values), X)


def same(test, got, expected, context=None):
    test.assertIsNot(got, lean.FALLBACK, context)
    test.assertIs(type(got), type(expected), context)
    test.assertEqual(got, expected, context)
    test.assertEqual(mf.to_data(got), mf.to_data(expected), context)


class UpolyTest(unittest.TestCase):
    def test_arithmetic_matches_polynomial(self):
        rng = random.Random(9101)
        for trial in range(1000):
            p = random_integers(rng, rng.randint(-1, 12), 1 + trial % 40)
            q = random_integers(rng, rng.randint(-1, 12), 1 + trial % 30)
            P, Q = as_polynomial(p), as_polynomial(q)
            with self.subTest(trial=trial, p=p, q=q):
                self.assertEqual(as_polynomial(tp.add(p, q)), P + Q)
                self.assertEqual(as_polynomial(tp.sub(p, q)), P - Q)
                self.assertEqual(as_polynomial(tp.mul(p, q)), P * Q)
                self.assertEqual(as_polynomial(tp.scale(p, 7)), P * 7)
                self.assertEqual(as_polynomial(tp.derivative(p)), P.derivative())
                self.assertEqual(tp.neg(p), tp.scale(p, -1))
                if q:
                    remainder = tp.pseudo_remainder(p, q)
                    exponent = max(len(p) - len(q) + 1, 0)
                    expected = (P * Q.leading_coefficient ** exponent).divmod(Q)[1] if len(p) >= len(q) else P
                    self.assertEqual(as_polynomial(remainder), expected)
                    product = tp.mul(p, q)
                    self.assertEqual(tp.exact_quotient(product, q), p)
                    if len(q) > 1 and tp.divmod_exact(tp.add(product, (1,)), q) is not None:
                        quotient, rest = tp.divmod_exact(tp.add(product, (1,)), q)
                        self.assertEqual(tp.add(tp.mul(quotient, q), rest), tp.add(product, (1,)))
                if p:
                    self.assertEqual(tp.content(p) * (1 if p[-1] > 0 else -1), tp.content(p) * (1 if p[-1] > 0 else -1))
                    primitive = tp.primitive(p)
                    self.assertGreater(primitive[-1], 0)
                    self.assertEqual(tp.content(primitive), 1)
                    self.assertEqual(tp.scale(primitive, tp.content(p) * (1 if p[-1] > 0 else -1)), p)

    def test_exact_quotient_refuses_a_non_divisor(self):
        with self.assertRaises(ArithmeticError):
            tp.exact_quotient((1, 0, 1), (1, 1))
        with self.assertRaises(ArithmeticError):
            tp.exact_quotient((1, 2), (2,))  # Divisible over Q, not over Z.

    def test_sign_at_matches_rational_evaluation(self):
        rng = random.Random(9102)
        for trial in range(1000):
            p = random_integers(rng, rng.randint(-1, 15), 1 + trial % 60)
            x = Rational(rng.randint(-50, 50), rng.randint(1, 40))
            value = as_polynomial(p).evaluate(x) if p else Rational(0)
            with self.subTest(p=p, x=x):
                self.assertEqual(tp.sign_at(p, x.numerator, x.denominator),
                                 (value.numerator > 0) - (value.numerator < 0))
                self.assertEqual(tp.value_at(p, x), value)

    def test_modular_arithmetic(self):
        rng = random.Random(9103)
        prime = 1_000_003
        for trial in range(300):
            p = tp.mod(random_integers(rng, rng.randint(0, 10), 30), prime)
            q = tp.mod(random_integers(rng, rng.randint(1, 8), 30), prime)
            if not q:
                continue
            quotient, remainder = tp.divmod_mod(p, q, prime)
            self.assertEqual(tp.mod(tp.add(tp.mul(quotient, q), remainder), prime), p)
            self.assertLess(len(remainder), len(q))
            self.assertEqual(tp.mul_mod(p, q, prime), tp.mod(tp.mul(p, q), prime))
            common = tp.gcd_mod(tp.mul_mod(p, q, prime), q, prime)
            self.assertEqual(common, tp.monic_mod(q, prime))
            if len(q) > 1:
                power = tp.pow_mod(p, 5, q, prime)
                expected = (1,)
                for _ in range(5):
                    expected = tp.divmod_mod(tp.mul_mod(expected, p, prime), q, prime)[1]
                self.assertEqual(power, tp.divmod_mod(expected, q, prime)[1])

    def test_kronecker_decides_identities(self):
        rng = random.Random(9104)
        for trial in range(300):
            p, q = random_integers(rng, rng.randint(0, 20), 50), random_integers(rng, rng.randint(0, 20), 50)
            product = tp.mul(p, q)
            bits = tp.kronecker_bits(tp.max_bits(product), tp.max_bits(p) + tp.max_bits(q)
                                     + min(len(p), len(q)).bit_length())
            self.assertEqual(tp.kronecker(product, bits), tp.kronecker(p, bits) * tp.kronecker(q, bits))
            wrong = tp.add(product, (0,) * rng.randint(0, len(product)) + (rng.choice((-1, 1)),))
            self.assertNotEqual(tp.kronecker(wrong, bits), tp.kronecker(p, bits) * tp.kronecker(q, bits))

    def test_rational_conversions(self):
        rng = random.Random(9105)
        for trial in range(300):
            polynomial = random_polynomial(rng, rng.randint(0, 12), rational=True, bits=20)
            numerators, denominator = tp.from_rationals(polynomial.coefficients)
            self.assertEqual(tp.rationals(numerators, denominator), polynomial.coefficients)
            self.assertEqual(tp.monic_rationals(numerators), polynomial.monic().coefficients)
        self.assertEqual(tp.rationals(()), (Rational(0),))

    def test_expression_is_to_expression(self):
        rng = random.Random(9106)
        cases = [Polynomial((0,), X), Polynomial((5,), X), Polynomial((0, 1), X), Polynomial((0, 0, -1), X),
                 Polynomial((1, 1, 1, 1), X), Polynomial((-1, -1), X), Polynomial((3, 0, 0, 1), X)]
        for trial in range(400):
            polynomial = random_polynomial(rng, rng.randint(0, 30), rational=trial % 2 == 0, bits=3)
            cases.append(polynomial)
        for polynomial in cases:
            with self.subTest(polynomial=str(polynomial)):
                same(self, tp.expression(polynomial.coefficients, X), polynomial.to_expression())
                self.assertEqual(hash(tp.expression(polynomial.coefficients, X)), hash(polynomial.to_expression()))

    def test_trusted_polynomial_equals_the_validated_one(self):
        polynomial = Polynomial((Rational(1, 2), 0, 3), X)
        built = trusted_polynomial(polynomial.coefficients, X)
        self.assertEqual(built, polynomial)
        self.assertEqual(hash(built), hash(polynomial))
        self.assertEqual(mf.to_data(built), mf.to_data(polynomial))


class LeanCalculusTest(unittest.TestCase):
    """differentiate, antiderivative, integrate and poly_divmod of Polynomial objects: the lean value is the
    structured value."""

    def setUp(self):
        lean._clear_caches()

    def test_values_equal_the_structured_path(self):
        rng = random.Random(9107)
        for trial in range(300):
            degree = rng.randint(0, 30)
            polynomial = random_polynomial(rng, degree, rational=trial % 2 == 0, bits=3 if trial % 5 else 120)
            if trial % 17 == 0:
                polynomial = Polynomial((0,), X)
            n = rng.randint(0, degree + 2)
            with self.subTest(trial=trial):
                same(self, lean.functions.differentiate(polynomial, X, n),
                     operations.differentiate(polynomial, X, n).unwrap())
                same(self, lean.functions.antiderivative(polynomial, X),
                     operations.antiderivative(polynomial, X).unwrap())
                lower, upper = Rational(rng.randint(-9, 9), rng.randint(1, 5)), rng.randint(-4, 4)
                same(self, lean.functions.integrate(polynomial, X, lower, upper),
                     operations.integrate(polynomial, X, lower, upper).unwrap())
                divisor = random_polynomial(rng, rng.randint(0, 12), rational=trial % 3 == 0, bits=4)
                same(self, lean.functions.poly_divmod(polynomial, divisor),
                     operations.poly_divmod(polynomial, divisor).unwrap())

    def test_refused_inputs_fall_back(self):
        """Inputs the fast paths leave: the lean answer is FALLBACK exactly where the structured path fails."""
        from mathforge.lean import algebra_poly, analysis_legacy, integrals_legacy
        polynomial = Polynomial((1, 2, 3), X)
        other = mf.Symbol("x")
        positive = mf.Symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))
        for arguments in ((polynomial, other, 1), (polynomial, X, -1), (polynomial, X, True), (polynomial, positive, 1),
                          (polynomial, X, 2**70000)):
            self.assertIs(analysis_legacy._differentiate_polynomial(*arguments), lean.FALLBACK, arguments)
            self.assertIs(lean.functions.differentiate(*arguments), lean.FALLBACK, arguments)
        for arguments in ((polynomial, Polynomial((0,), X)), (polynomial, Polynomial((1, 1), other))):
            self.assertIs(algebra_poly._poly_divmod(*arguments), lean.FALLBACK)
            self.assertIs(lean.functions.poly_divmod(*arguments), lean.FALLBACK)
        high = Polynomial((0,) * 1024 + (1,), X)
        self.assertIs(integrals_legacy._antiderivative_polynomial(high, X), lean.FALLBACK)
        self.assertIs(lean.functions.antiderivative(high, X), lean.FALLBACK)
        forged = Polynomial((1, 2), X)
        object.__setattr__(forged, "coefficients", (Rational(1), 2))
        self.assertIs(algebra_poly._poly_divmod(forged, polynomial), lean.FALLBACK)

    def test_checks_reject_wrong_results(self):
        values = (Rational(1), Rational(-2), Rational(3, 4), Rational(5))
        derivative = kernels.derivative(values, 2)
        self.assertTrue(certify.check_derivative(values, 2, derivative))
        self.assertFalse(certify.check_derivative(values, 2, derivative[:-1] + (Rational(31),)))
        self.assertFalse(certify.check_derivative(values, 1, derivative))
        primitive = kernels.antiderivative(values)
        self.assertTrue(certify.check_antiderivative(values, primitive))
        self.assertFalse(certify.check_antiderivative(values, (Rational(1),) + primitive[1:]))
        quotient, remainder = kernels.divmod_rational(values, (Rational(1), Rational(2)))
        self.assertTrue(certify.check_division(values, (Rational(1), Rational(2)), quotient, remainder))
        self.assertFalse(certify.check_division(values, (Rational(1), Rational(2)), quotient,
                                                (remainder[0] + 1,)))
        self.assertFalse(certify.check_division(values, (Rational(1), Rational(2)), quotient + (Rational(1),),
                                                remainder))


if __name__ == "__main__":
    unittest.main()
