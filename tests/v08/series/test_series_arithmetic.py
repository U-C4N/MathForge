"""trusted.series (truncated power series over Q) and the series checker's node rules: properties and a SymPy
differential test of the expansions it computes."""

import importlib.util
from fractions import Fraction
import random
import unittest

import mathforge as mf
from mathforge.certify.series import taylor as checker
from mathforge.limits import computation
from mathforge.trusted import series

from tests.v08.series.test_taylor import X, _regular, function

Q = mf.Rational
SYMPY = importlib.util.find_spec("sympy") is not None


def random_series(rng, length, head=None):
    values = [Q(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(length)]
    if head is not None:
        values[0] = head
    return tuple(values)


class SeriesArithmeticTest(unittest.TestCase):
    def test_ring_properties(self):
        rng = random.Random(11)
        with computation():
            for length in (1, 2, 5, 9):
                for _ in range(20):
                    a, b, c = (random_series(rng, length) for _ in range(3))
                    self.assertEqual(series.mul(a, b), series.mul(b, a))
                    self.assertEqual(series.mul(series.mul(a, b), c), series.mul(a, series.mul(b, c)))
                    self.assertEqual(series.mul(a, series.add(b, c)), series.add(series.mul(a, b), series.mul(a, c)))
                    self.assertEqual(series.sub(series.add(a, b), b), a)
                    self.assertEqual(series.neg(a), series.scale(a, Q(-1)))
                    self.assertEqual(series.mul(a, series.constant(Q(1), length)), a)

    def test_reciprocal_power_derivative_integral(self):
        rng = random.Random(12)
        with computation():
            for length in (1, 3, 8):
                for _ in range(20):
                    a = random_series(rng, length, head=Q(rng.choice((-3, -1, 2, 5)), rng.randint(1, 4)))
                    self.assertEqual(series.mul(a, series.reciprocal(a)), series.constant(Q(1), length))
                    self.assertEqual(series.power(a, 3), series.mul(a, series.mul(a, a)))
                    self.assertEqual(series.power(a, 0), series.constant(Q(1), length))
                    self.assertEqual(series.derivative(series.integral(a)), a)
                    self.assertEqual(series.integral(series.derivative(a)), (Q(0),) + a[1:])
            with self.assertRaises(mf.InvalidInput):
                series.reciprocal((Q(0), Q(1)))
            with self.assertRaises(mf.InvalidInput):
                series.mul((Q(1),), (Q(1), Q(2)))

    def test_exact_values(self):
        """exp, sin, log(1+x), atan, sqrt(4+x) and 1/(1-x) at 0 against their textbook coefficients."""
        x = X
        expected = {
            "exp(x)": [Fraction(1, 1), 1, Fraction(1, 2), Fraction(1, 6), Fraction(1, 24), Fraction(1, 120)],
            "sin(x)": [0, 1, 0, Fraction(-1, 6), 0, Fraction(1, 120)],
            "log(1+x)": [0, 1, Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 4), Fraction(1, 5)],
            "atan(x)": [0, 1, 0, Fraction(-1, 3), 0, Fraction(1, 5)],
            "sqrt(4+x)": [2, Fraction(1, 4), Fraction(-1, 64), Fraction(1, 512), Fraction(-5, 16384),
                          Fraction(7, 131072)],
            "1/(1-x)": [1, 1, 1, 1, 1, 1],
            "tanh(x)": [0, 1, 0, Fraction(-1, 3), 0, Fraction(2, 15)],
            "asin(x)": [0, 1, 0, Fraction(1, 6), 0, Fraction(3, 40)],
        }
        for text, values in expected.items():
            with self.subTest(text=text), computation():
                got = checker._series(function(text).expression, x, Q(0), 6, {})
                self.assertEqual([Fraction(c.numerator, c.denominator) for c in got], values)

    @unittest.skipUnless(SYMPY, "SymPy is not installed")
    def test_against_sympy(self):
        """The checker's expansions of seeded compositions at 0 equal SymPy's series coefficients."""
        import sympy as sp
        rng = random.Random(20261019)
        x = sp.Symbol("x")
        compared = 0
        for _ in range(60):
            text = _regular(rng, rng.randint(1, 2))
            order = rng.randint(0, 6)
            source = function(text)
            with computation():
                try:
                    ours = checker._series(source.expression, X, Q(0), order + 1, {})
                except checker._Decline:
                    continue
            expression = sp.sympify(text, locals={"x": x})
            polynomial = sp.series(expression, x, 0, order + 1).removeO()
            theirs = [sp.Rational(polynomial.coeff(x, k)) if k else sp.Rational(polynomial.subs(x, 0))
                      for k in range(order + 1)]
            with self.subTest(text=text, order=order):
                self.assertEqual([sp.Rational(c.numerator, c.denominator) for c in ours], theirs)
            compared += 1
        self.assertGreater(compared, 40)


if __name__ == "__main__":
    unittest.main()
