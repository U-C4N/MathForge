"""Presentation/input adapters must preserve existing mathematical contracts."""

from dataclasses import fields, replace
from fractions import Fraction
import inspect
from math import isqrt
import pickle
import random
import typing
import unittest

import mathforge as m
from mathforge import operations


class FrontendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000361")
        self.y = m.Symbol("y", uid="00000000-0000-4000-8000-000000000362")
        self.symbols = {"x": self.x, "y": self.y}

    def test_exact_decimal_problem_to_verified_result_and_unchanged_codec(self):
        problem = m.parse_problem("0.1*x=0.3", symbols=self.symbols)
        result = m.solve(problem, for_=self.x)
        self.assertEqual(result.solution_set, m.FiniteSet((m.Rational(3),)))
        self.assertTrue(m.verify(result).verified)
        self.assertIs(result.for_, self.x)
        self.assertEqual(result.request.get("problem"), problem)
        encoded = m.dumps(result)
        restored = m.loads(encoded)
        self.assertEqual(restored, result)
        self.assertTrue(m.verify(restored).verified)
        self.assertIn("completed", m.render_text(restored, symbols=self.symbols))
        self.assertIn("3", m.to_latex(restored, symbols=self.symbols))
        self.assertEqual(m.dumps(restored), encoded)

    def test_parsed_rational_derivative_keeps_holes_through_every_adapter(self):
        expression = m.parse_expression("(x^2-1)/(x-1)", symbols=self.symbols)
        result = operations.differentiate(expression, self.x)
        self.assertIs(result.execution_status, m.ExecutionStatus.COMPLETED)
        self.assertTrue(m.verify(result).verified)
        self.assertEqual(result.unwrap().evaluate(2), m.Rational(1))
        self.assertFalse(result.unwrap().is_defined_at(1))
        encoded = m.dumps(result)
        restored = m.loads(encoded)
        self.assertIn("where", m.render_text(restored))
        self.assertIn("!= 0", m.render_text(restored))
        self.assertIn(r"\ne 0", m.to_latex(restored))
        self.assertTrue(m.verify(restored).verified)
        self.assertEqual(m.dumps(result), encoded)
        self.assertEqual(m.dumps(restored), encoded)

    def test_same_name_symbols_remain_distinct_across_aliases_and_codec(self):
        second_x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000363")
        aliases = {"first": self.x, "second": second_x}
        expression = m.parse_expression("first-second", symbols=aliases)
        result = operations.evaluate(expression, {self.x: 5, second_x: 2})
        self.assertEqual(result.unwrap(), m.Rational(3))
        self.assertTrue(m.verify(result).verified)
        restored = m.loads(m.dumps(expression))
        rendered = m.render_text(restored, symbols=aliases)
        reparsed = m.parse_expression(rendered, symbols=aliases)
        self.assertEqual(m.evaluate(reparsed, {self.x: 5, second_x: 2}), m.Rational(3))
        with self.assertRaises(m.InvalidInput):
            m.render_text(restored)

    def test_unsupported_calculus_remains_unsupported_not_no_solution(self):
        expression = m.parse_expression("x+sqrt(2)", symbols=self.symbols)
        result = operations.differentiate(expression, self.x)
        self.assertIs(result.execution_status, m.ExecutionStatus.UNSUPPORTED)
        self.assertIsNone(result.value)
        self.assertIs(result.outcome, m.Outcome.NOT_APPLICABLE)
        text = m.render_text(m.loads(m.dumps(result)))
        self.assertIn("unsupported", text)
        self.assertNotIn("no_solution", text)
        self.assertNotIn("verified", text)

    def test_rootof_rendering_does_not_turn_its_dummy_variable_into_a_symbol(self):
        # A free symbol may be named t, just like the presentation's local root
        # polynomial variable.  Only the free Symbol participates in binding.
        t = m.Symbol("t", uid="00000000-0000-4000-8000-000000000364")
        root = m.real_roots(self.x**3 - 2, self.x)[0].root
        expression = t + root
        encoded = m.dumps(expression)
        text = m.render_text(expression, symbols={"outside": t})
        self.assertIn("outside", text)
        self.assertIn("RootOf", text)
        self.assertIn("real_index=0", text)
        self.assertIn(r"\operatorname{RootOf}", m.to_latex(expression, symbols={"outside": t}))
        self.assertEqual(m.evaluate(expression, {t: 0}), root)
        self.assertEqual(m.dumps(expression), encoded)
        self.assertEqual(m.loads(encoded), expression)

    def test_raw_ast_text_preserves_math_with_independent_fraction_oracle(self):
        randomizer = random.Random(36103)
        bindings = {self.x: m.Rational(-2, 3), self.y: m.Rational(5, 2)}

        def generate(depth):
            if depth == 0 or randomizer.randrange(4) == 0:
                kind = randomizer.randrange(4)
                if kind == 0:
                    return self.x
                if kind == 1:
                    return self.y
                if kind == 2:
                    return m.Rational(randomizer.randrange(-5, 6), randomizer.randrange(1, 5))
                return m.Sqrt(m.Rational(4, 9))
            kind = randomizer.randrange(3)
            if kind == 0:
                return m.Add((generate(depth - 1), generate(depth - 1)))
            if kind == 1:
                return m.Mul((generate(depth - 1), generate(depth - 1)))
            return m.Pow(generate(depth - 1), randomizer.randrange(4))

        def exact(value):
            if type(value) is m.Rational:
                return Fraction(value.numerator, value.denominator)
            if type(value) is m.Symbol:
                return exact(bindings[value])
            if type(value) is m.Sqrt:
                return Fraction(isqrt(value.radicand.numerator), isqrt(value.radicand.denominator))
            if type(value) is m.Add:
                return sum((exact(term) for term in value.terms), Fraction(0))
            if type(value) is m.Mul:
                product = Fraction(1)
                for factor in value.factors:
                    product *= exact(factor)
                return product
            return exact(value.base) ** value.exponent

        for _ in range(160):
            expression = generate(4)
            encoded = m.dumps(expression)
            text = m.render_text(expression, symbols=self.symbols)
            with self.subTest(text=text):
                reparsed = m.parse_expression(text, symbols=self.symbols)
                expected = exact(expression)
                self.assertEqual(m.evaluate(reparsed, bindings),
                                 m.Rational(expected.numerator, expected.denominator))
                self.assertTrue(m.to_latex(expression, symbols=self.symbols))
                self.assertEqual(m.dumps(expression), encoded)

    def test_generic_result_preserves_runtime_type_dataclass_and_codec(self):
        arguments = dict(execution_status=m.ExecutionStatus.COMPLETED, outcome=m.Outcome.VALUE,
                         value=m.Rational(2, 3), exactness=m.Exactness.EXACT,
                         completeness=m.Completeness.COMPLETE)
        typed = m.Result[m.Rational](**arguments)
        untyped = m.Result(**arguments)
        self.assertIs(type(typed), m.Result)
        self.assertEqual(typed, untyped)
        self.assertEqual(hash(typed), hash(untyped))
        self.assertEqual(tuple(field.name for field in fields(typed)),
                         tuple(field.name for field in fields(untyped)))
        self.assertEqual(typed.unwrap(), m.Rational(2, 3))
        self.assertEqual(m.dumps(typed), m.dumps(untyped))
        self.assertEqual(m.loads(m.dumps(typed)), untyped)
        self.assertEqual(pickle.loads(pickle.dumps(typed)), untyped)
        self.assertEqual(replace(typed, value=m.Rational(5)).unwrap(), m.Rational(5))

    def test_generic_result_failure_still_retains_the_original_error_contract(self):
        failure = m.Result[m.Rational](execution_status=m.ExecutionStatus.INVALID_INPUT,
                                       outcome=m.Outcome.NOT_APPLICABLE,
                                       error=m.ErrorInfo("example_error", "example message"))
        with self.assertRaises(m.OperationError) as caught:
            failure.unwrap()
        self.assertIs(caught.exception.result, failure)
        self.assertEqual(caught.exception.code, "example_error")
        self.assertEqual(m.loads(m.dumps(failure)), failure)

    def test_public_type_hints_are_resolvable_at_runtime(self):
        for module, names in ((m, m.__all__), (operations, vars(operations))):
            for name in names:
                if name.startswith("_"):
                    continue
                value = getattr(module, name)
                if inspect.isfunction(value) and value.__module__ in {
                    "mathforge", "mathforge.operations", "mathforge.parsing", "mathforge.rendering"
                }:
                    with self.subTest(module=module.__name__, name=name):
                        self.assertIn("return", typing.get_type_hints(value))


if __name__ == "__main__":
    unittest.main()
