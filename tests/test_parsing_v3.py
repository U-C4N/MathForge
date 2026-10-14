"""Bounded exact text input, explicit identities, and persistent domain holes."""

from dataclasses import FrozenInstanceError, fields
from fractions import Fraction
import random
import sys
from types import MappingProxyType
import unittest
from unittest import mock

import mathforge as m
from mathforge import parsing
from mathforge.errors import InvalidInput, MathForgeError, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.model import free_symbols
from mathforge.parsing import ParseError, ParseLimits, parse_expression, parse_problem


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000301")
        self.y = m.Symbol("y", uid="00000000-0000-4000-8000-000000000302")
        self.symbols = {"x": self.x, "y": self.y}

    def parse(self, text, **kwargs):
        return parse_expression(text, symbols=self.symbols, **kwargs)

    def test_exact_integer_fraction_and_finite_decimal_literals(self):
        cases = {
            "0": m.Rational(0), "00012": m.Rational(12), "-12": m.Rational(-12),
            "1/3": m.Rational(1, 3), "-2 / 6": m.Rational(-1, 3),
            "0.1": m.Rational(1, 10), "0.1+0.2": m.Rational(3, 10),
            ".125": m.Rational(1, 8), "12.": m.Rational(12),
            "001.2500": m.Rational(5, 4), "-.5": m.Rational(-1, 2),
            "0.000": m.Rational(0), "2.5/0.5": m.Rational(5),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_expression(text, symbols={}), expected)

    def test_operator_precedence_and_associativity(self):
        cases = {
            "1 + 2 * 3": m.Rational(7), "(1 + 2)*3": m.Rational(9),
            "8/4/2": m.Rational(1), "8/(4/2)": m.Rational(4),
            "5-2-1": m.Rational(2), "5-(2-1)": m.Rational(4),
            "2**3**2": m.Rational(512), "2^3^2": m.Rational(512),
            "-2**2": m.Rational(-4), "(-2)**2": m.Rational(4),
            "-x**2": -(self.x**2), "(-x)**2": self.x**2,
            "2*-3": m.Rational(-6), "--x": self.x, "+-+x": -self.x,
            "x**(1+1)": self.x**2, "x**2.0": self.x**2,
            "x**(y-y+2)": self.x**2, "x^0": m.Rational(1),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.parse(text), expected)

    def test_constant_square_roots_reuse_the_existing_exact_model(self):
        for text, expected in (("sqrt(4)", m.Rational(2)),
                               ("sqrt(1/4)", m.Rational(1, 2)),
                               ("sqrt(.25)", m.Rational(1, 2)),
                               ("sqrt(2)", m.sqrt(2)),
                               ("x+sqrt(2)", self.x + m.sqrt(2))):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text), expected)
        for text in ("sqrt(-1)", "sqrt(x)", "sqrt(sqrt(2))", "sqrt(x/x)"):
            with self.subTest(text=text):
                with self.assertRaises(UnsupportedOperation) as caught:
                    self.parse(text)
                self.assertEqual(caught.exception.code, "unsupported_operation")
                self.assertEqual(caught.exception.start, 0)

    def test_explicit_symbols_are_required_and_never_created_from_names(self):
        with self.assertRaises(TypeError):
            parse_expression("x")
        with self.assertRaises(ParseError) as caught:
            parse_expression("x", symbols={})
        self.assertEqual(caught.exception.code, "unknown_symbol")
        self.assertEqual((caught.exception.start, caught.exception.end), (0, 1))
        self.assertIs(self.parse("x"), self.x)
        self.assertEqual(self.symbols, {"x": self.x, "y": self.y})

    def test_aliases_and_distinct_identities_with_the_same_display_name(self):
        other = m.Symbol("x", uid="00000000-0000-4000-8000-000000000303")
        aliases = MappingProxyType({"left": self.x, "right": other})
        expression = parse_expression("left-right", symbols=aliases)
        self.assertEqual(free_symbols(expression), frozenset((self.x, other)))
        self.assertNotEqual(expression, m.Rational(0))
        for _ in range(2):
            self.assertIs(parse_expression("left", symbols=aliases), self.x)
        self.assertIs(parse_problem("left=right", symbols=aliases).lhs, self.x)

    def test_symbol_table_is_snapshotted_before_construction(self):
        symbols = {"x": self.x}
        original = parsing.sqrt

        def mutate(value):
            symbols["x"] = self.y
            return original(value)

        with mock.patch.object(parsing, "sqrt", side_effect=mutate):
            value = parse_expression("sqrt(4)+x", symbols=symbols)
        self.assertEqual(value, 2 + self.x)
        self.assertIs(symbols["x"], self.y)

    def test_invalid_symbol_tables_have_a_stable_error_code(self):
        conflicting = m.Symbol("renamed", uid=self.x.uid)
        cases = (None, [], {"sqrt": self.x}, {"2x": self.x}, {"_x": self.x},
                 {"x-y": self.x}, {"α": self.x}, {1: self.x}, {"x": 1},
                 {"x": self.x, "alias": self.x}, {"x": self.x, "other": conflicting})
        for table in cases:
            with self.subTest(table=table):
                with self.assertRaises(ParseError) as caught:
                    parse_expression("1", symbols=table)
                self.assertEqual(caught.exception.code, "invalid_symbol_table")

    def test_whitespace_and_unicode_character_based_error_coordinates(self):
        self.assertEqual(self.parse("\t x\n +\r\n2 \u00a0"), self.x + 2)
        source = "\u00a0x+\n  unknown"
        with self.assertRaises(ParseError) as caught:
            self.parse(source)
        error = caught.exception
        self.assertEqual((error.start, error.end), (6, len(source)))
        self.assertEqual((error.line, error.column), (2, 3))
        self.assertIsInstance(error, InvalidInput)

    def test_error_spans_for_syntax_end_of_input_and_core_failures(self):
        cases = (("x + )", 4, 5), ("x +", 3, 3), ("(x", 2, 2),
                 ("x.foo", 1, 2), ("1,2", 1, 2))
        for source, start, end in cases:
            with self.subTest(source=source):
                with self.assertRaises(ParseError) as caught:
                    self.parse(source)
                self.assertEqual((caught.exception.start, caught.exception.end), (start, end))
                self.assertEqual(caught.exception.code, "parse_syntax")
        with self.assertRaises(UnsupportedOperation) as caught:
            self.parse("x ** -1")
        self.assertEqual((caught.exception.start, caught.exception.end), (2, 4))
        with self.assertRaises(InvalidInput) as caught:
            self.parse("1 / 0")
        self.assertEqual(caught.exception.code, "division_by_zero")
        self.assertEqual((caught.exception.start, caught.exception.end), (2, 3))

    def test_malformed_and_out_of_scope_syntax_is_not_python(self):
        malformed = ("", " ", "()", "x y", "2x", "x(1)", "sin(x)",
                     "sqrt", "sqrt()", "sqrt(1,2)", "1e3", "1E-3", ".", "1..2",
                     "x//2", "x%2", "x[0]", "x.__class__", "x:=1", "x!=1",
                     "[1,2]", "{'x':1}", "'x'", '"x"', "__import__('os')",
                     "lambda: 1", "x;1", "x\x00", "١", "𝟙", "NaN", "Infinity")
        with mock.patch("builtins.eval", side_effect=AssertionError("eval must not be called")), \
                mock.patch("builtins.exec", side_effect=AssertionError("exec must not be called")):
            for source in malformed:
                with self.subTest(source=source):
                    with self.assertRaises(ParseError):
                        self.parse(source)

    def test_fractional_negative_and_variable_exponents_stay_unsupported(self):
        for source in ("x**-1", "x^-2", "x**(1/2)", "x**.5", "x**y", "x**(y/y)"):
            with self.subTest(source=source):
                with self.assertRaises(UnsupportedOperation):
                    self.parse(source)

    def test_equations_and_inequalities_use_existing_problem_types(self):
        for operator in ("=", "=="):
            problem = parse_problem(f"x**2 {operator} 2", symbols=self.symbols)
            self.assertEqual(problem, m.Eq(self.x**2, 2))
            result = m.solve(problem, for_=self.x)
            self.assertEqual(result.solution_set, m.solve(m.Eq(self.x**2, 2), for_=self.x).solution_set)
            self.assertTrue(m.verify(result).verified)
        for operator, relation in (("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge")):
            problem = parse_problem(f"x {operator} 2", symbols=self.symbols)
            self.assertEqual(problem, m.Inequality(self.x, relation, 2))
        for source in ("x", "x=", "=x", "x<y<2", "x=x=1", "(x=1)", "x=(y=1)"):
            with self.subTest(source=source):
                with self.assertRaises(ParseError):
                    parse_problem(source, symbols=self.symbols)
        with self.assertRaises(ParseError):
            self.parse("x=1")

    def test_rational_functions_keep_all_domain_holes(self):
        cases = {"x/x": 1, "0*(1/x)": 0, "(1/x)**0": 1, "1/(1/x)": 2,
                 "(x**2-1)/(x-1)": 3, "1/x-1/x": 0, "+(1/x)": m.Rational(1, 2)}
        for source, at_two in cases.items():
            with self.subTest(source=source):
                value = self.parse(source)
                self.assertIs(type(value), m.RationalFunction)
                self.assertEqual(value.evaluate(2), m.Rational(at_two) if type(at_two) is int else at_two)
                excluded_point = 1 if source == "(x**2-1)/(x-1)" else 0
                self.assertFalse(value.is_defined_at(excluded_point))
                self.assertEqual(m.loads(m.dumps(value)), value)
        value = self.parse("1/(x-1)+1/(x+1)")
        self.assertFalse(value.is_defined_at(1))
        self.assertFalse(value.is_defined_at(-1))
        self.assertEqual(value.evaluate(2), m.Rational(4, 3))

    def test_cancelled_hole_is_not_a_solution_after_parsing(self):
        problem = parse_problem("(x**2-1)/(x-1)=2", symbols=self.symbols)
        result = m.solve(problem, for_=self.x)
        self.assertIs(result.outcome, m.Outcome.NO_SOLUTION)
        self.assertTrue(m.verify(result).verified)
        problem = parse_problem("x/x=1", symbols=self.symbols)
        result = m.solve(problem, for_=self.x)
        self.assertTrue(m.verify(result).verified)
        self.assertFalse(result.solution_set.contains(0))
        self.assertTrue(result.solution_set.contains(2))

    def test_zero_denominators_fail_instead_of_becoming_empty_solutions(self):
        for source in ("1/0", "0/0", "1/(x-x)", "1/(x/x-1)", "1/(0*(1/x))"):
            with self.subTest(source=source):
                with self.assertRaises(InvalidInput) as caught:
                    self.parse(source)
                self.assertEqual(caught.exception.code, "division_by_zero")

    def test_rational_constant_division_does_not_force_a_univariate_model(self):
        self.assertEqual(self.parse("(x+y)/2"), (self.x + self.y)/2)
        for source in ("1/(x+y)", "x/y", "1/x+1/y", "y+1/x", "1/sqrt(2)", "sqrt(2)/x"):
            with self.subTest(source=source):
                with self.assertRaises(UnsupportedOperation):
                    self.parse(source)

    def test_existing_symbol_domain_metadata_is_preserved_not_extended(self):
        integer = m.symbol("n", domain=m.Integers)
        positive = m.symbol("p", assumptions=(m.Assumption("positive", m.Truth.TRUE),))
        for variable in (integer, positive):
            self.assertIs(parse_expression("a", symbols={"a": variable}), variable)
            with self.assertRaises(UnsupportedOperation):
                parse_expression("1/a", symbols={"a": variable})
        # Parsing is not a new solver and does not claim to add radical coefficients.
        value = self.parse("x+sqrt(2)")
        result = m.operations.differentiate(value, self.x)
        self.assertIs(result.execution_status, m.ExecutionStatus.UNSUPPORTED)

    def test_parsed_outputs_remain_v1_v2_serializable(self):
        for source in ("x**2-2", "0.1+x", "1/x", "sqrt(2)+x"):
            with self.subTest(source=source):
                value = self.parse(source)
                self.assertEqual(m.loads(m.dumps(value)), value)
        problem = parse_problem("x**2<=2", symbols=self.symbols)
        self.assertEqual(m.loads(m.dumps(problem)), problem)


class ParseLimitTests(unittest.TestCase):
    def setUp(self):
        self.x = m.symbol("x")

    def parse(self, text, **overrides):
        return parse_expression(text, symbols={"x": self.x}, parse_limits=ParseLimits(**overrides))

    def test_limits_are_frozen_positive_exact_integers(self):
        self.assertEqual(ParseLimits().max_chars, 16384)
        for field in fields(ParseLimits):
            for invalid in (0, -1, True, 1.5, "2", None):
                with self.subTest(field=field.name, invalid=invalid):
                    with self.assertRaises(InvalidInput):
                        ParseLimits(**{field.name: invalid})
        with self.assertRaises(FrozenInstanceError):
            ParseLimits().max_chars = 1

    def test_character_limit_boundary(self):
        self.assertEqual(self.parse("1 ", max_chars=2), m.Rational(1))
        with self.assertRaises(ResourceLimitError) as caught:
            self.parse("1  ", max_chars=2)
        self.assertEqual((caught.exception.start, caught.exception.end), (2, 3))

    def test_token_limit_boundary_excludes_end_marker(self):
        self.assertEqual(self.parse("1+2", max_tokens=3), m.Rational(3))
        with self.assertRaises(ResourceLimitError) as caught:
            self.parse("1+2", max_tokens=2)
        self.assertEqual((caught.exception.start, caught.exception.end), (2, 3))
        self.assertEqual(self.parse("1", max_tokens=1), m.Rational(1))

    def test_node_limit_boundary_counts_operators_and_comparisons(self):
        self.assertEqual(self.parse("1+2", max_nodes=3), m.Rational(3))
        self.assertEqual(self.parse("(1)", max_nodes=1), m.Rational(1))
        for source, maximum in (("1+2", 2), ("-1", 1), ("sqrt(1)", 1)):
            with self.subTest(source=source):
                with self.assertRaises(ResourceLimitError):
                    self.parse(source, max_nodes=maximum)
        self.assertIsInstance(parse_problem("1=1", symbols={}, parse_limits=ParseLimits(max_nodes=3)), m.Eq)
        with self.assertRaises(ResourceLimitError):
            parse_problem("1=1", symbols={}, parse_limits=ParseLimits(max_nodes=2))

    def test_depth_boundary_and_deep_input_do_not_leak_recursion_errors(self):
        self.assertEqual(self.parse("1", max_depth=1), m.Rational(1))
        self.assertEqual(self.parse("((1))", max_depth=3), m.Rational(1))
        with self.assertRaises(ResourceLimitError):
            self.parse("((1))", max_depth=2)
        for source in ("("*100 + "1" + ")"*100, "-"*100 + "1", "1^"*100 + "1"):
            with self.subTest(prefix=source[:5]):
                with self.assertRaises(ResourceLimitError):
                    self.parse(source)
        with self.assertRaises(ResourceLimitError):
            self.parse("("*1500 + "1" + ")"*1500, max_depth=2000)

    def test_literal_digit_boundary_includes_decimal_fraction_and_leading_zeros(self):
        for source, expected in (("123", m.Rational(123)), ("1.23", m.Rational(123, 100)),
                                 (".123", m.Rational(123, 1000)), ("000", m.Rational(0))):
            with self.subTest(source=source):
                self.assertEqual(self.parse(source, max_integer_digits=3), expected)
        for source in ("1234", "1.234", ".1234", "0000"):
            with self.subTest(source=source):
                with self.assertRaises(ResourceLimitError) as caught:
                    self.parse(source, max_integer_digits=3)
                self.assertEqual((caught.exception.start, caught.exception.end), (0, len(source)))

    def test_large_integer_input_does_not_modify_python_global_conversion_limit(self):
        source = "9" * 4500
        before = sys.get_int_max_str_digits()
        value = parse_expression(source, symbols={},
                                 parse_limits=ParseLimits(max_integer_digits=5000))
        self.assertEqual(value.numerator, 10**4500 - 1)
        self.assertEqual(sys.get_int_max_str_digits(), before)

    def test_exponent_limit_boundary_is_checked_after_exact_evaluation(self):
        self.assertEqual(self.parse("x**4", max_exponent=4), self.x**4)
        self.assertEqual(self.parse("x**(2+2)", max_exponent=4), self.x**4)
        for source in ("x**5", "x**(2+3)", "2**2**3"):
            with self.subTest(source=source):
                with self.assertRaises(ResourceLimitError):
                    self.parse(source, max_exponent=4)

    def test_symbol_limit_boundary_includes_unused_entries(self):
        y = m.symbol("y")
        self.assertEqual(parse_expression("1", symbols={"x": self.x},
                                         parse_limits=ParseLimits(max_symbols=1)), m.Rational(1))
        with self.assertRaises(ResourceLimitError):
            parse_expression("1", symbols={"x": self.x, "y": y},
                             parse_limits=ParseLimits(max_symbols=1))

    def test_computation_budget_also_bounds_number_and_ast_construction(self):
        self.assertEqual(parse_expression("65535", symbols={}, limits=ComputationLimits(max_integer_bits=16)),
                         m.Rational(65535))
        for source, limits in (("65536", ComputationLimits(max_integer_bits=16)),
                               ("2**32", ComputationLimits(max_integer_bits=16)),
                               (".0000001", ComputationLimits(max_integer_bits=16)),
                               ("1", ComputationLimits(max_work=1)),
                               ("1", ComputationLimits(max_nodes=1))):
            with self.subTest(source=source, limits=limits):
                with self.assertRaises(ResourceLimitError) as caught:
                    parse_expression(source, symbols={}, limits=limits)
                self.assertEqual(caught.exception.code, "resource_limit_exceeded")
                self.assertTrue(hasattr(caught.exception, "start"))

    def test_wrong_input_and_limit_types_are_typed_errors(self):
        for invalid in (None, 1, True, b"1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ParseError):
                    parse_expression(invalid, symbols={})
        with self.assertRaises(ParseError):
            parse_expression("1", symbols={}, parse_limits={})
        with self.assertRaises(InvalidInput):
            parse_expression("1", symbols={}, limits={})

    def test_process_control_exceptions_are_not_translated(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception.__name__):
                with mock.patch.object(parsing, "sqrt", side_effect=exception):
                    with self.assertRaises(exception):
                        parse_expression("sqrt(4)", symbols={})


class ParserGeneratedTests(unittest.TestCase):
    def test_seeded_numeric_trees_agree_with_fraction_arithmetic(self):
        randomizer = random.Random(30103)

        def generate(depth):
            if depth == 0 or randomizer.randrange(4) == 0:
                numerator, denominator = randomizer.randrange(-9, 10), randomizer.randrange(1, 10)
                return f"({numerator}/{denominator})", Fraction(numerator, denominator)
            left_text, left = generate(depth - 1)
            right_text, right = generate(depth - 1)
            operator = randomizer.choice(("+", "-", "*", "/"))
            if operator == "/" and right == 0:
                operator = "+"
            if operator == "+":
                value = left + right
            elif operator == "-":
                value = left - right
            elif operator == "*":
                value = left * right
            else:
                value = left / right
            return f"({left_text}{operator}{right_text})", value

        for _ in range(120):
            source, expected = generate(4)
            with self.subTest(source=source):
                actual = parse_expression(source, symbols={})
                self.assertEqual(actual, m.Rational(expected.numerator, expected.denominator))

    def test_seeded_arbitrary_text_never_raises_untyped_internal_failures(self):
        randomizer = random.Random(30991)
        x = m.symbol("x")
        alphabet = "0123xy sqrt.+-*/^()<>!=_,'[]\n\t\u03b1\x00"
        limits = ParseLimits(max_chars=256, max_tokens=128, max_nodes=128,
                             max_depth=12, max_integer_digits=12, max_exponent=16)
        computation_limits = ComputationLimits(max_work=20_000, max_integer_bits=1024,
                                               max_nodes=2048, max_depth=32)
        for _ in range(800):
            source = "".join(randomizer.choice(alphabet) for _ in range(randomizer.randrange(65)))
            for function in (parse_expression, parse_problem):
                try:
                    function(source, symbols={"x": x}, parse_limits=limits, limits=computation_limits)
                except MathForgeError as error:
                    self.assertTrue(hasattr(error, "start"), (source, type(error)))
                    self.assertGreaterEqual(error.start, 0)
                    self.assertGreaterEqual(error.end, error.start)
                    self.assertLessEqual(error.end, len(source))


if __name__ == "__main__":
    unittest.main()
