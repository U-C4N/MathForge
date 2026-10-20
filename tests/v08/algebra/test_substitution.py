"""The substitution kernel, its independent checker and the lean substitute against model.substitute_expression."""

import random
import unittest

import mathforge as mf
from mathforge import lean, operations
from mathforge.certify import substitution as certify
from mathforge.kernels import substitution as kernels
from mathforge.model import Add, Mul, Pow, Rational, Sqrt, simplify_expression, substitute_expression

X = mf.Symbol("x", uid="00000000-0000-4000-8000-0000000000b1")
Y = mf.Symbol("y", uid="00000000-0000-4000-8000-0000000000b2")
Z = mf.Symbol("z", uid="00000000-0000-4000-8000-0000000000b3")


def leaf(rng):
    draw = rng.random()
    if draw < 0.35:
        return rng.choice((X, Y, Z))
    if draw < 0.7:
        return Rational(rng.randint(-5, 5), rng.randint(1, 3))
    if draw < 0.85:
        return mf.sqrt(rng.choice((2, 3, 5, 8, 12)))
    return Sqrt(Rational(rng.choice((4, 9, 2))))  # a raw square root, possibly of a square


def tree(rng, depth):
    """Raw (unnormalized) Add, Mul and Pow nodes mixed with normalized arithmetic."""
    if depth == 0 or rng.random() < 0.2:
        return leaf(rng)
    draw, count = rng.random(), rng.randint(2, 4)
    if draw < 0.35:
        return Add(tuple(tree(rng, depth - 1) for _ in range(count)))
    if draw < 0.7:
        return Mul(tuple(tree(rng, depth - 1) for _ in range(count)))
    if draw < 0.85:
        return Pow(tree(rng, depth - 1), rng.randint(0, 4))
    left, right = tree(rng, depth - 1), tree(rng, depth - 1)
    return left * right + left - right


class SubstitutionTest(unittest.TestCase):
    def setUp(self):
        lean._clear_caches()

    def test_kernel_checker_and_lean_equal_the_model(self):
        rng = random.Random(9201)
        for trial in range(1500):
            expression = tree(rng, rng.randint(1, 4))
            bindings = {symbol: (tree(rng, rng.randint(0, 2)) if rng.random() < 0.8 else rng.randint(-4, 4))
                        for symbol in (X, Y, Z) if rng.random() < 0.6}
            expected = substitute_expression(expression, bindings)
            with self.subTest(trial=trial, expression=str(expression)):
                self.assertEqual(kernels.normalize(expression), simplify_expression(expression))
                replacements = {key.uid: kernels.normalize(Rational(value) if type(value) is int else value)
                                for key, value in bindings.items()}
                produced = kernels.substitute(expression, replacements)
                self.assertEqual(produced, expected)
                self.assertEqual(mf.to_data(produced), mf.to_data(expected))
                pairs = [(key, Rational(value) if type(value) is int else value) for key, value in bindings.items()]
                self.assertIs(certify.check_substitution(expression, pairs, expected), True)
                value = lean.functions.substitute(expression, bindings)
                self.assertIsNot(value, lean.FALLBACK)
                self.assertEqual(mf.to_data(value), mf.to_data(operations.substitute(expression, bindings).unwrap()))

    def test_checker_rejects_other_values(self):
        expression = (X + 1) * (X - 2)
        pairs = [(X, Y + 3)]
        right = substitute_expression(expression, dict(pairs))
        self.assertIs(type(right), Mul)
        self.assertTrue(certify.check_substitution(expression, pairs, right))
        # A wrong product, the same polynomial expanded, the factors out of order (equal in value, not in form).
        for wrong in ((Y + 4) * (Y + 2), Y * Y + 5 * Y + 4, Mul((right.factors[1], right.factors[0])), Rational(0)):
            self.assertFalse(certify.check_substitution(expression, pairs, wrong), wrong)

    def test_refused_inputs_fall_back(self):
        """The fast path leaves these inputs to the generic lean path, whose answer is the structured one."""
        from mathforge.lean import expr_poly
        positive = mf.Symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))
        twin = mf.Symbol("other name", uid=X.uid)
        rational_function = mf.rational_function(1, X + 1, variable=X)
        deep = X
        for _ in range(150):
            deep = Add((deep, Rational(1)))
        cases = (((X + 1), {positive: 2}), (X + 1, {"x": 2}), (X + twin, {X: 1}), (rational_function, {X: 2}),
                 (X + 1, [(X, 2)]), (X + 1, {X: 2**70000}), (mf.RealAlgebraicRoot((-2, 0, 1), 1) + X, {X: 1}),
                 (1.5, {X: 1}), (deep, {X: 1}))
        for index, (expression, bindings) in enumerate(cases):
            with self.subTest(case=index):
                self.assertIs(expr_poly._substitute(expression, bindings), lean.FALLBACK)
                value = lean.functions.substitute(expression, bindings)
                try:
                    expected = ("value", operations.substitute(expression, bindings).unwrap())
                except Exception:
                    expected = ("error",)
                if expected[0] == "error":
                    self.assertIs(value, lean.FALLBACK)
                else:
                    self.assertEqual(mf.to_data(value), mf.to_data(expected[1]))

if __name__ == "__main__":
    unittest.main()
