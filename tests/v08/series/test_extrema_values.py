"""The lean extrema_values path for rational stationary points: v0.7's values, an independent checker, declines."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import function_analysis, operations  # Imported before any patch, which it would bind otherwise.
from mathforge import lean as mathforge_lean
from mathforge.certify.analysis import extrema as checker
from mathforge.lean import FALLBACK, analysis_extrema
from mathforge.limits import computation

Q = mf.Rational
X = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000081")


def seeded_inputs(count=120, seed=20261019):
    """Polynomial expressions, Polynomial records and rational functions, many with rational stationary points."""
    rng, x = random.Random(seed), X
    out = []
    while len(out) < count:
        kind = rng.random()
        if kind < 0.3:
            coefficients = [rng.randint(-6, 6) for _ in range(rng.randint(2, 5))]
            expression = sum((c * x ** k for k, c in enumerate(coefficients) if c), Q(0))
        elif kind < 0.5:
            a, b, c = (rng.randint(-4, 4) for _ in range(3))
            expression = (x - a) ** rng.randint(1, 3) * (x - b) ** rng.randint(1, 2) * (x - c)
        elif kind < 0.6:
            expression = mf.Polynomial(tuple(Q(rng.randint(-5, 5)) for _ in range(rng.randint(2, 6))), x)
        else:
            numerator = sum((rng.randint(-4, 4) * x ** k for k in range(rng.randint(1, 3))), Q(0))
            denominator = sum((rng.randint(-4, 4) * x ** k for k in range(rng.randint(1, 3))), Q(1))
            try:
                expression = mf.rational_function(numerator, denominator, variable=x)
            except mf.MathForgeError:
                continue
        out.append(expression)
    return out


def structured(expression, variable=X):
    return operations.extrema_values(expression, variable).unwrap()


class ExtremaValuesTest(unittest.TestCase):
    def test_values_equal_v07(self):
        """The lean value equals the structured path's on seeded inputs; the rational path takes most of them."""
        taken = 0
        for expression in seeded_inputs():
            with self.subTest(expression=str(expression)):
                try:
                    expected = structured(expression)
                except mf.MathForgeError:
                    self.assertIs(analysis_extrema.extrema_values(expression, X), FALLBACK)
                    continue
                with patch("mathforge.lean.run_legacy", wraps=mathforge_lean.run_legacy) as generic:
                    lean = analysis_extrema.extrema_values(expression, X)
                if lean is not FALLBACK:
                    self.assertEqual(lean, expected)
                    taken += not generic.called
                self.assertEqual(mf.extrema_values(expression, X), expected)
        self.assertGreater(taken, 50)

    def test_textbook_cases(self):
        x = X
        cases = {
            x ** 3 - 3 * x: ((Q(-1), Q(2), "maximum"), (Q(1), Q(-2), "minimum")),
            3 * x ** 2 + 5 * x - 7: ((Q(-5, 6), Q(-109, 12), "minimum"),),
            mf.rational_function(x, x * x + 1, variable=x): ((Q(-1), Q(-1, 2), "minimum"),
                                                              (Q(1), Q(1, 2), "maximum")),
            x ** 4 - 2 * x ** 2: ((Q(-1), Q(-1), "minimum"), (Q(0), Q(0), "maximum"), (Q(1), Q(-1), "minimum")),
            x ** 3: (),  # A stationary point of even multiplicity in f' is no extremum.
            mf.rational_function((x - 1) * x * x, x - 1, variable=x): ((Q(0), Q(0), "minimum"),),  # A hole at 1.
            mf.rational_function(x ** 2 - 1, x - 1, variable=x): (),  # Its only stationary candidate is the hole.
        }
        for expression, expected in cases.items():
            with self.subTest(expression=str(expression)):
                value = analysis_extrema.extrema_values(expression, X)
                self.assertEqual(tuple((e.point, e.value, e.kind) for e in value), expected)
                self.assertEqual(value, structured(expression))

    def test_irrational_or_constant_inputs_take_the_generic_path(self):
        x = X
        for expression in (mf.rational_function(x * x + 1, x - 1, variable=x), x ** 3 - 2 * x, Q(5), x - x):
            with self.subTest(expression=str(expression)):
                with patch.object(analysis_extrema, "_check", side_effect=AssertionError("rational checker")):
                    value = analysis_extrema.extrema_values(expression, X)
                self.assertEqual(value, structured(expression))

    def record(self, expression):
        args = {"expression": expression, "variable": X}
        with computation():
            value, evidence = analysis_extrema._produce(dict(args))
        self.assertTrue(checker.check_operation(args, value, evidence))
        return args, value, evidence

    def test_forgeries_fail(self):
        x = X
        args, value, evidence = self.record(x ** 4 - 2 * x ** 2)  # P = 4x^3 - 4x = 4 (x + 1) x (x - 1).
        first = value[0]
        tag, lead, roots, quadratics = evidence
        forged_values = {
            "kind": (replace(first, kind="maximum"),) + value[1:],
            "value": (replace(first, value=Q(0)),) + value[1:],
            "point": (replace(first, point=Q(-2)),) + value[1:],
            "missing": value[1:],
            "extra": value + (value[0],),
            "order": value[::-1],
        }
        for label, forged in forged_values.items():
            with self.subTest(forged=label):
                self.assertFalse(checker.check_operation(args, forged, evidence))
        forged_evidence = {
            "lead": (tag, lead * 2, roots, quadratics),
            "root": (tag, lead, ((Q(-2), 1),) + roots[1:], quadratics),
            "multiplicity": (tag, lead, ((roots[0][0], 3),) + roots[1:], quadratics),
            "dropped root": (tag, lead, roots[1:], quadratics),
            "real quadratic": (tag, lead, roots[:1], ((Q(0), Q(-1), Q(1)),)),  # x^2 - x has real roots.
            "unsorted": (tag, lead, roots[::-1], quadratics),
            "format": ("other", lead, roots, quadratics),
        }
        for label, forged in forged_evidence.items():
            with self.subTest(evidence=label):
                self.assertFalse(checker.check_operation(args, value, forged))
        # The source decides: another function with the same claim is refused.
        self.assertFalse(checker.check_operation({"expression": x ** 4 - 3 * x ** 2, "variable": X}, value, evidence))
        self.assertFalse(checker.check_operation({"expression": x ** 4 - 2 * x ** 2, "variable": mf.Symbol("x")},
                                                 value, evidence))

    def test_quadratic_factors_and_exclusions(self):
        x = X
        # P = 2x (x^2 + 1) ... from f = (x^2 + 1)^2 / 2: a quadratic without real roots in the factorization.
        args, value, evidence = self.record(mf.rational_function((x * x + 1) ** 2, 2, variable=x))
        self.assertEqual(len(evidence[3]), 1)
        self.assertEqual(tuple((e.point, e.kind) for e in value), ((Q(0), "minimum"),))
        # A stationary point at an excluded point is no extremum, and the checker knows it.
        source = mf.rational_function(x * (x - 1) ** 2, x, variable=x)  # (x - 1)^2 with x != 0: minimum at 1.
        args, value, evidence = self.record(source)
        self.assertEqual(value, structured(source))
        hole = mf.rational_function((x - 1) ** 3, x - 1, variable=x)  # (x - 1)^2 with a hole at 1: no extremum.
        args, value, evidence = self.record(hole)
        self.assertEqual(value, ())
        self.assertEqual(value, structured(hole))

    def test_checker_runs_no_producer(self):
        args, value, evidence = self.record(X ** 3 - 3 * X)
        with patch("mathforge.rational_function.as_rational_function", side_effect=AssertionError("producer")), \
             patch("mathforge.polynomial.Polynomial.square_free", side_effect=AssertionError("producer")), \
             patch.object(function_analysis, "analyze_function", side_effect=AssertionError("producer")):
            self.assertTrue(checker.check_operation(args, value, evidence))


if __name__ == "__main__":
    unittest.main()
