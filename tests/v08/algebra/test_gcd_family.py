"""The lean gcd family (poly_gcd, poly_xgcd, square_free, rational_function) and the lean resultant and
discriminant against the structured path, and their independent checks against wrong claims."""

import random
import unittest

import mathforge as mf
from mathforge import lean, operations
from mathforge.certify import upoly as certify
from mathforge.kernels import upoly as kernels
from mathforge.model import Rational
from mathforge.polynomial import Polynomial
from mathforge.trusted import upoly as tp

X = mf.Symbol("x", uid="00000000-0000-4000-8000-0000000000c1")


def random_polynomial(rng, degree, *, rational=False, bits=4):
    values = [Rational(rng.randint(-2**bits, 2**bits), rng.randint(1, 5) if rational else 1)
              for _ in range(degree + 1)]
    if not values[-1].numerator:
        values[-1] = Rational(1)
    return Polynomial(tuple(values), X)


def outcome(call):
    try:
        return ("value", call())
    except Exception as error:
        return ("error", type(error), getattr(error, "code", None))


class GcdFamilyTest(unittest.TestCase):
    def setUp(self):
        lean._clear_caches()

    def assertLeanEquals(self, value, structured, context):
        expected = outcome(structured)
        if value is lean.FALLBACK:
            return False
        self.assertEqual(expected[0], "value", context)
        self.assertIs(type(value), type(expected[1]), context)
        self.assertEqual(value, expected[1], context)
        self.assertEqual(mf.to_data(value), mf.to_data(expected[1]), context)
        return True

    def test_values_equal_the_structured_path(self):
        rng = random.Random(9301)
        answered = 0
        for trial in range(250):
            common = random_polynomial(rng, rng.randint(0, 4), rational=trial % 3 == 0)
            f = random_polynomial(rng, rng.randint(0, 6), rational=trial % 2 == 0) * common
            g = random_polynomial(rng, rng.randint(0, 6)) * common
            if trial % 5 == 0:
                f = f * common
            a, b = random_polynomial(rng, rng.randint(0, 8)), random_polynomial(rng, rng.randint(0, 8), rational=True)
            calls = (
                ("poly_gcd", lean.functions.poly_gcd(f, g), lambda: operations.poly_gcd(f, g).unwrap()),
                ("poly_xgcd", lean.functions.poly_xgcd(f, g), lambda: operations.poly_xgcd(f, g).unwrap()),
                ("square_free", lean.functions.square_free(f), lambda: operations.square_free(f).unwrap()),
                ("rational_function", lean.functions.rational_function(f, g, X),
                 lambda: operations.rational_function(f, g, variable=X).unwrap()),
                ("resultant", lean.functions.resultant(a, b), lambda: operations.resultant(a, b).unwrap()),
                ("discriminant", lean.functions.discriminant(f), lambda: operations.discriminant(f).unwrap()),
            )
            for name, value, structured in calls:
                with self.subTest(trial=trial, operation=name):
                    answered += self.assertLeanEquals(value, structured, (name, str(f), str(g)))
        self.assertGreater(answered, 1400)

    def test_repeated_factors(self):
        rng = random.Random(9302)
        for trial in range(60):
            p = (random_polynomial(rng, 0) * random_polynomial(rng, 1) ** rng.randint(1, 3)
                 * random_polynomial(rng, 2) ** rng.randint(1, 3) * random_polynomial(rng, rng.randint(1, 3)))
            q = random_polynomial(rng, 2)
            with self.subTest(trial=trial):
                self.assertTrue(self.assertLeanEquals(lean.functions.square_free(p),
                                                      lambda: operations.square_free(p).unwrap(), str(p)))
                self.assertTrue(self.assertLeanEquals(lean.functions.rational_function(q, p, X),
                                                      lambda: operations.rational_function(q, p, variable=X).unwrap(),
                                                      str(p)))

    def test_heuristic_gcd_matches_euclid(self):
        rng = random.Random(9303)
        for trial in range(200):
            common = tp.primitive(tuple(rng.randint(-50, 50) for _ in range(rng.randint(1, 6))) + (rng.randint(1, 9),))
            f = tp.primitive(tp.mul(common, tuple(rng.randint(-50, 50) for _ in range(rng.randint(1, 8))) + (3,)))
            g = tp.primitive(tp.mul(common, tuple(rng.randint(-50, 50) for _ in range(rng.randint(1, 8))) + (5,)))
            h, u, v = kernels.gcd_cofactors(f, g)
            self.assertEqual(h, kernels.euclid_gcd(f, g))
            self.assertTrue(certify.check_gcd(f, g, h, u, v))

    def test_checks_reject_wrong_claims(self):
        f, g = (2, 3, 1), (3, 4, 1)  # (x + 1)(x + 2), (x + 1)(x + 3)
        self.assertTrue(certify.check_gcd(f, g, (1, 1), (2, 1), (3, 1)))
        self.assertFalse(certify.check_gcd(f, g, (1,), f, g))  # 1 divides both, but the cofactors share x + 1
        self.assertFalse(certify.check_gcd(f, g, (1, 1), (2, 1), (4, 1)))
        self.assertFalse(certify.coprime((1, 1), (2, 2)))
        self.assertTrue(certify.coprime((1, 1), (2, 1)))
        self.assertTrue(certify.check_resultant((2, 3, 1), (3, 1), 2))  # Res((x+1)(x+2), x+3) = (-3+1)(-3+2) = 2
        self.assertFalse(certify.check_resultant((2, 3, 1), (3, 1), 3))
        values = tuple(Rational(v) for v in (2, 3, 1))
        self.assertTrue(certify.check_square_free(values + (Rational(0),) * 0, [((Rational(2), Rational(3), Rational(1)), 1)]))
        self.assertFalse(certify.check_square_free(values, [((Rational(1), Rational(1)), 1), ((Rational(2), Rational(1)), 1)]))
        self.assertFalse(certify.check_square_free(values, [((Rational(1), Rational(1)), 2)]))
        rational = lambda *items: tuple(Rational(item) for item in items)  # noqa: E731
        self.assertTrue(certify.check_reduced_fraction(rational(2, 3, 1), rational(3, 4, 1), rational(2, 1),
                                                       rational(3, 1)))
        self.assertFalse(certify.check_reduced_fraction(rational(2, 3, 1), rational(3, 4, 1), rational(2, 3, 1),
                                                        rational(3, 4, 1)))  # not reduced
        # (x + 2) - (x + 1) = 1: the xgcd of x + 1 and x + 2 is (1, -1, 1).
        self.assertTrue(certify.check_xgcd(rational(1, 1), rational(2, 1), rational(1), rational(-1), rational(1)))
        self.assertFalse(certify.check_xgcd(rational(1, 1), rational(2, 1), rational(1), rational(-2), rational(1)))
        # Another Bezout pair, s + (x + 2) and t - (x + 1): true, but not the one of minimal degrees.
        self.assertFalse(certify.check_xgcd(rational(1, 1), rational(2, 1), rational(1), rational(1, 1),
                                            rational(0, -1)))

    def test_refused_inputs_fall_back(self):
        """The fast paths leave zero and constant operands, other variables and expressions to the generic lean
        path, whose answer is the structured one."""
        from mathforge.lean import algebra_poly
        zero, one = Polynomial((0,), X), Polynomial((1, 1), X)
        other = Polynomial((1, 1), mf.Symbol("x"))
        cases = (("poly_gcd", (zero, one)), ("poly_xgcd", (one, other)), ("square_free", (Polynomial((5,), X),)),
                 ("rational_function", (one, zero, X)), ("rational_function", (one, one, mf.Symbol("x"))),
                 ("rational_function", (X + 1, X - 1, X)), ("resultant", (zero, one)),
                 ("discriminant", (Polynomial((3,), X),)))
        for name, arguments in cases:
            with self.subTest(operation=name, arguments=arguments):
                self.assertIs(getattr(algebra_poly, "_" + name)(*arguments), lean.FALLBACK)
                value = getattr(lean.functions, name)(*arguments)
                keywords = {"variable": arguments[2]} if name == "rational_function" else {}
                positional = arguments[:2] if name == "rational_function" else arguments
                expected = outcome(lambda: getattr(operations, name)(*positional, **keywords).unwrap())
                if expected[0] == "error":
                    self.assertIs(value, lean.FALLBACK)
                else:
                    self.assertEqual(mf.to_data(value), mf.to_data(expected[1]))

if __name__ == "__main__":
    unittest.main()
