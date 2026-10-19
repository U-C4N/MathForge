"""The lean taylor path: v0.7's producer, a power-series checker, the generic checker where it declines; and the
function_calculus.produce crash outside computation()."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import function_calculus, operations
from mathforge import functions as f
from mathforge.certify.series import taylor as checker
from mathforge.lean import FALLBACK, series_taylor
from mathforge.limits import computation

Q = mf.Rational
X = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000091")
Y = mf.Symbol("y", uid="00000000-0000-4000-8000-000000000092")


def function(text):
    return mf.parse_function(text, symbols={"x": X})


def _vanishing(rng, depth):
    """Text of an expression in x that vanishes at 0."""
    choice = rng.random()
    if depth == 0 or choice < 0.3:
        power, scale = rng.randint(1, 3), rng.choice((1, 2, -1, 3, -2))
        return f"{scale}*x" if power == 1 else f"{scale}*x**{power}"
    if choice < 0.45:
        return f"({_vanishing(rng, depth - 1)}+{_vanishing(rng, depth - 1)})"
    if choice < 0.55:
        return f"({_vanishing(rng, depth - 1)})*({_regular(rng, depth - 1)})"
    name = rng.choice(("sin", "tan", "atan", "asin", "sinh", "tanh"))
    return f"{name}({_vanishing(rng, depth - 1)})"


def _regular(rng, depth):
    """Text of an expression in x that is analytic at 0, with special values at every node."""
    choice = rng.random()
    if depth == 0 or choice < 0.15:
        return str(rng.randint(1, 4))
    inner = _vanishing(rng, depth - 1)
    if choice < 0.3:
        return f"exp({inner})"
    if choice < 0.38:
        return f"cos({inner})"
    if choice < 0.43:
        return f"cosh({inner})"
    if choice < 0.53:
        return f"log(1+{inner})"
    if choice < 0.6:
        return f"sqrt({rng.choice((1, 4, 9))}+{inner})"
    if choice < 0.65:
        return f"abs({rng.choice((1, -2, 3))}+{inner})"
    if choice < 0.75:
        return f"({_regular(rng, depth - 1)}+{inner})"
    if choice < 0.88:
        return f"({_regular(rng, depth - 1)})*({_regular(rng, depth - 1)})"
    if choice < 0.94:
        return f"({_regular(rng, depth - 1)})/({rng.randint(1, 3)}+{inner})"
    return f"({rng.randint(1, 3)}+{inner})**{rng.choice((-2, -1, 2, 3))}"


def seeded(count=50, seed=20261019):
    rng = random.Random(seed)
    return [(function(_regular(rng, rng.randint(1, 3))), rng.randint(0, 7)) for _ in range(count)]


def structured(expression, point, order):
    result = operations.taylor(expression, X, point, order)
    return result.unwrap() if result.error is None else result.error


class TaylorLeanTest(unittest.TestCase):
    def test_values_equal_v07(self):
        """Seeded compositions of exp, log(1+u), sin, cos, tan, atan, asin, sinh, cosh, tanh, sqrt(c^2+u), abs(c+u)
        and integer powers at the center 0: the lean value is the structured one, and the series checker decided."""
        decided = 0
        for expression, order in seeded():
            with self.subTest(expression=str(expression), order=order):
                expected = structured(expression, 0, order)
                lean = series_taylor.function_taylor(expression, X, 0, order)
                if type(expected) is not function_calculus.TaylorPolynomial:
                    # Only a resource limit of the structured path may leave the lean path a value (rule 5).
                    if lean is not FALLBACK:
                        self.assertEqual(expected.code, "resource_limit_exceeded")
                    continue
                self.assertEqual(lean, expected)
                args = {"expression": expression, "variable": X, "point": 0, "order": order}
                decided += checker.check_operation("function_taylor", args, expected, ()) is True
        self.assertGreater(decided, 35)

    def test_v07_refusals_stay_refusals(self):
        """v0.7's checker refuses these correct traces (a tan or tanh atom it does not identify); the structured path
        fails, and the lean path falls back although the series checker would accept the producer's value."""
        for text, order in (("(1+tan(-8*x**3))**3", 3), ("sqrt(9+tan(tan(3*x**3)))", 2), ("(3+tanh(2*x**3))**2", 1)):
            with self.subTest(text=text):
                expression = function(text)
                expected = structured(expression, 0, order)
                self.assertEqual(expected.code, "execution_error")
                with computation():
                    value, _ = function_calculus.produce(
                        "function_taylor", {"expression": expression, "variable": X, "point": 0, "order": order})
                args = {"expression": expression, "variable": X, "point": 0, "order": order}
                self.assertTrue(checker.check_operation("function_taylor", args, value, ()))
                self.assertIs(series_taylor.function_taylor(expression, X, 0, order), FALLBACK)

    def test_public_function(self):
        expression = function("exp(x)*sin(x)+log(1+x)")
        value = mf.taylor(expression, X, 0, 6)
        self.assertEqual(value, structured(expression, 0, 6))
        self.assertEqual(tuple(c.value for c in value.coefficients),
                         (Q(0), Q(2), Q(1, 2), Q(2, 3), Q(-1, 4), Q(1, 6), Q(-8, 45)))

    def test_declined_inputs_take_the_v07_checker(self):
        """exp(x) at center 1, acos(x) at 0, a center sqrt(2), log(x) about 1/2: the series checker declines; the
        lean value is still the structured one (or both refuse)."""
        cases = ((function("exp(x)"), 1, 4), (function("acos(x)"), 0, 3), (function("exp(x)+x"), mf.sqrt(2), 2),
                 (function("log(x)"), Q(1, 2), 3), (function("sin(x)+pi"), 0, 3))
        for expression, point, order in cases:
            with self.subTest(expression=str(expression), point=str(point)):
                expected = structured(expression, point, order)
                args = {"expression": expression, "variable": X, "point": point, "order": order}
                if type(expected) is function_calculus.TaylorPolynomial:
                    self.assertFalse(checker.check_operation("function_taylor", args, expected, ()))
                    self.assertEqual(series_taylor.function_taylor(expression, X, point, order), expected)
                else:
                    self.assertIs(series_taylor.function_taylor(expression, X, point, order), FALLBACK)

    def test_forgeries_fail(self):
        expression = function("exp(x)*sin(x)+log(1+x)")
        value = structured(expression, 0, 4)
        args = {"expression": expression, "variable": X, "point": 0, "order": 4}
        self.assertTrue(checker.check_operation("function_taylor", args, value, ()))
        coefficients = value.coefficients
        wrong = f.as_real_expr(coefficients[2].value + 1)
        forged = {
            "coefficient": replace(value, coefficients=coefficients[:2] + (wrong,) + coefficients[3:]),
            "order": replace(value, coefficients=coefficients[:-1]),
            "point": replace(value, point=f.as_real_expr(1)),
            "polynomial": replace(value, polynomial=f.real_function(f.as_real_expr(X) * 2, (X,))),
        }
        for label, record in forged.items():
            with self.subTest(forged=label):
                self.assertFalse(checker.check_operation("function_taylor", args, record, ()))
        self.assertFalse(checker.check_operation("function_taylor", dict(args, order=5), value, ()))
        self.assertFalse(checker.check_operation("function_taylor", dict(args, expression=function("exp(x)")),
                                                 value, ()))
        self.assertFalse(checker.check_operation("function_differentiate", args, value, ()))
        # A wrong value from the producer is refused by both checkers: the lean path falls back.
        with patch.object(function_calculus, "_produce", return_value=(forged["coefficient"], ())):
            self.assertIs(series_taylor.function_taylor(expression, X, 0, 4), FALLBACK)

    def test_shared_subtrees(self):
        """sin(x)*sin(x) + sin(x): one node object may appear twice; the series is computed once and accepted."""
        s = f.sin(f.as_real_expr(X))
        expression = f.real_function(s * s + s, (X,))
        value = structured(expression, 0, 6)
        args = {"expression": expression, "variable": X, "point": 0, "order": 6}
        self.assertTrue(checker.check_operation("function_taylor", args, value, ()))
        self.assertEqual(series_taylor.function_taylor(expression, X, 0, 6), value)


class TaylorCrashTest(unittest.TestCase):
    def test_produce_without_an_active_budget(self):
        """function_calculus.produce for function_taylor, function_jacobian and function_hessian outside
        computation() equals the result inside it (v0.7 raised AttributeError on current_budget())."""
        two = mf.parse_function("x*y+exp(x)", symbols={"x": X, "y": Y})
        cases = (("function_taylor", {"expression": function("exp(x)*sin(x)"), "variable": X, "point": 0,
                                      "order": 4}),
                 ("function_jacobian", {"expressions": (two, two * two), "variables": (X, Y)}),
                 ("function_hessian", {"expression": two, "variables": (X, Y)}))
        for operation, args in cases:
            with self.subTest(operation=operation):
                outside = function_calculus.produce(operation, args)
                with computation():
                    inside = function_calculus.produce(operation, args)
                self.assertEqual(outside, inside)


if __name__ == "__main__":
    unittest.main()
