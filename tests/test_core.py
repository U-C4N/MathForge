"""Exact invariants for the number system, safe AST, and polynomial algorithms."""

from dataclasses import FrozenInstanceError
import random
import unittest

from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.model import (
    Add, Assumption, Integers, Mul, Pow, Rational, Rationals, Reals, Sqrt,
    Truth, add, evaluate_expression, free_symbols, mul, simplify_expression,
    sqrt, substitute_expression, symbol,
)
from mathforge.polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial


class RationalTests(unittest.TestCase):
    def test_normalization_and_exact_arithmetic(self):
        self.assertEqual(Rational(6, -8), Rational(-3, 4))
        self.assertEqual(Rational(-6, -8), Rational(3, 4))
        self.assertEqual(Rational(0, -912), Rational(0, 1))
        self.assertEqual(Rational(1, 10) + Rational(2, 10), Rational(3, 10))
        self.assertEqual(Rational(5, 6) / Rational(10, 9), Rational(3, 4))
        self.assertEqual(3 - Rational(2, 3), Rational(7, 3))
        self.assertEqual(3 / Rational(2, 3), Rational(9, 2))
        self.assertEqual(Rational(-2, 3) ** 3, Rational(-8, 27))
        self.assertLess(Rational(-3, 2), -1)
        self.assertGreater(1, Rational(2, 3))

    def test_arithmetic_invariants(self):
        rng = random.Random(782)
        for _ in range(100):
            a = Rational(rng.randrange(-50, 51), rng.randrange(1, 30))
            b = Rational(rng.randrange(-50, 51), rng.randrange(1, 30))
            c = Rational(rng.randrange(-50, 51), rng.randrange(1, 30))
            self.assertEqual(a * (b + c), a * b + a * c)
            self.assertEqual((a + b) - b, a)
            if b.numerator:
                self.assertEqual((a / b) * b, a)

    def test_zero_denominator_and_inexact_inputs(self):
        for callback in (lambda: Rational(1, 0), lambda: Rational(1) / 0):
            with self.assertRaises(ZeroDivisionError):
                callback()
        for value in (0.1, True, False, "1", 1 + 0j):
            with self.assertRaises(TypeError):
                Rational(value)
            with self.assertRaises(TypeError):
                Rational(1, value)
            with self.assertRaises(TypeError):
                Rational(1) + value

    def test_structural_equality_is_not_python_coercion(self):
        self.assertEqual(Rational(2, 4), Rational(1, 2))
        self.assertNotEqual(Rational(1), 1)
        self.assertEqual(hash(Rational(2, 4)), hash(Rational(1, 2)))


class ExpressionTests(unittest.TestCase):
    def setUp(self):
        self.x = symbol("x")
        self.y = symbol("y")

    def test_identity_immutability_and_domain(self):
        self.assertNotEqual(self.x, symbol("x"))
        self.assertEqual(self.x.domain, Reals)
        original = [self.x, self.y]
        node = Add(original)
        original.append(Rational(8))
        self.assertEqual(node.terms, (self.x, self.y))
        with self.assertRaises(FrozenInstanceError):
            self.x.name = "z"
        with self.assertRaises(FrozenInstanceError):
            node.terms = ()
        with self.assertRaises(TypeError):
            bool(self.x)

    def test_metadata_subclasses_cannot_add_mutable_state(self):
        class MutableAssumption(Assumption):
            pass

        class MutableString(str):
            pass

        assumption = MutableAssumption("positive")
        assumption.items = []
        with self.assertRaises(InvalidInput):
            symbol("x", assumptions=(assumption,))
        with self.assertRaises(InvalidInput):
            symbol(MutableString("x"))
        with self.assertRaises(InvalidInput):
            Assumption(MutableString("positive"))
        with self.assertRaises(InvalidInput):
            type(self.x)("x", uid=MutableString(self.x.uid))

        class MutableSymbol(type(self.x)):
            pass

        mutable_symbol = MutableSymbol("s")
        mutable_symbol.items = []
        with self.assertRaises(InvalidInput):
            Polynomial((1, 2), mutable_symbol)
        with self.assertRaises(InvalidInput):
            substitute_expression(self.x, {mutable_symbol: 1})

    def test_conservative_rules(self):
        x = self.x
        self.assertEqual(x + x + 2 * x, 4 * x)
        self.assertEqual(x * x * x**2, x**4)
        self.assertEqual((x + 2) - (x + 2), Rational(0))
        self.assertEqual(x * 0, Rational(0))
        self.assertEqual(x / 2, Rational(1, 2) * x)
        self.assertEqual(x**0, Rational(1))
        self.assertEqual(Rational(0)**0, Rational(1))
        self.assertEqual(simplify_expression(Add((x, Rational(0)))), x)
        self.assertNotEqual((x + 1)**2, x**2 + 2*x + 1)
        for callback in (lambda: x/x, lambda: 1/x, lambda: x**-1,
                         lambda: x**Rational(1, 2), lambda: sqrt(x**2)):
            with self.assertRaises(UnsupportedOperation):
                callback()

    def test_rational_and_symbolic_square_roots(self):
        self.assertEqual(sqrt(Rational(9, 16)), Rational(3, 4))
        self.assertEqual(sqrt(0), Rational(0))
        root = sqrt(2)
        self.assertIsInstance(root, Sqrt)
        self.assertEqual(root * root, Rational(2))
        self.assertEqual(root**4, Rational(4))
        self.assertEqual(root**3, 2 * root)
        self.assertEqual((-root)**2, Rational(2))
        self.assertEqual((-Rational(3, 2) * root)**4, Rational(81, 4))
        self.assertEqual(root + root, 2 * root)
        self.assertEqual(simplify_expression(Sqrt(Rational(4))), Rational(2))
        with self.assertRaises(UnsupportedOperation):
            sqrt(-1)

    def test_substitution_simultaneous_and_evaluation_total(self):
        x, y = self.x, self.y
        self.assertEqual(substitute_expression(x + 2*y, {x: y, y: x}), y + 2*x)
        self.assertEqual(substitute_expression(x**2 - 2, {x: sqrt(2)}), Rational(0))
        self.assertEqual(evaluate_expression(x**2 - 2, {x: -sqrt(2)}), Rational(0))
        self.assertEqual(evaluate_expression(x**2 + y, {x: Rational(3, 2), y: 1}), Rational(13, 4))
        self.assertEqual(free_symbols(x + y), frozenset((x, y)))
        self.assertEqual(substitute_expression(x + y, {x: 2}), 2 + y)
        with self.assertRaises(InvalidInput):
            evaluate_expression(x + y, {x: 2})
        with self.assertRaises(InvalidInput):
            substitute_expression(x, {"x": 2})
        self.assertEqual(substitute_expression(x, {symbol("x"): 2}), x)

    def test_assumptions_and_domain_are_not_discarded(self):
        nonzero = symbol("n", assumptions=(Assumption("nonzero", Truth.TRUE),))
        with self.assertRaises(InvalidInput):
            evaluate_expression(nonzero, {nonzero: 0})
        self.assertEqual(evaluate_expression(nonzero, {nonzero: 2}), Rational(2))
        with self.assertRaises(UnsupportedOperation):
            substitute_expression(nonzero, {nonzero: self.x})
        unknown = symbol("u", assumptions=(Assumption("positive"),))
        self.assertEqual(evaluate_expression(unknown, {unknown: -2}), Rational(-2))
        for domain, value in ((Integers, Rational(1, 2)), (Rationals, sqrt(2))):
            restricted = symbol("r", domain=domain)
            with self.assertRaises(InvalidInput):
                evaluate_expression(restricted, {restricted: value})
        with self.assertRaises(InvalidInput):
            symbol("impossible", assumptions=(Assumption("positive", Truth.TRUE), Assumption("nonzero", Truth.FALSE)))

    def test_normalization_idempotent(self):
        expressions = [self.x + self.y + 2, (2*self.x)*(3*self.x), sqrt(2)**3,
                       Add((self.x, Add((self.y, Rational(0))))), (self.x+1)**5]
        for expression in expressions:
            normalized = simplify_expression(expression)
            self.assertEqual(simplify_expression(normalized), normalized)

    def test_raw_ast_normalization_and_integer_evaluation_invariants(self):
        rng = random.Random(9374)

        def expression(depth, radicals=False):
            if depth == 0:
                leaves = [self.x, self.y, Rational(-2), Rational(0), Rational(3)]
                if radicals:
                    leaves += [Sqrt(Rational(2)), Sqrt(Rational(8)), Sqrt(Rational(9))]
                return rng.choice(leaves)
            choice = rng.randrange(3)
            if choice == 0:
                return Add((expression(depth-1, radicals), expression(depth-1, radicals)))
            if choice == 1:
                return Mul((expression(depth-1, radicals), expression(depth-1, radicals)))
            return Pow(expression(depth-1, radicals), rng.randrange(4))

        def integer_oracle(node):
            if isinstance(node, Rational):
                return node.numerator
            if node == self.x:
                return -3
            if node == self.y:
                return 2
            if isinstance(node, Add):
                return sum(integer_oracle(child) for child in node.terms)
            if isinstance(node, Mul):
                result = 1
                for child in node.factors:
                    result *= integer_oracle(child)
                return result
            return integer_oracle(node.base) ** node.exponent

        for _ in range(150):
            raw = expression(3)
            expected = Rational(integer_oracle(raw))
            self.assertEqual(evaluate_expression(raw, {self.x: -3, self.y: 2}), expected)
            with_radicals = expression(3, radicals=True)
            normalized = simplify_expression(with_radicals)
            self.assertEqual(simplify_expression(normalized), normalized)


class PolynomialTests(unittest.TestCase):
    def setUp(self):
        self.x = symbol("x")

    def test_expansion_derivative_and_exact_evaluation(self):
        polynomial = Polynomial.from_expression((self.x + 1)**3, self.x)
        self.assertEqual(polynomial.coefficients, tuple(Rational(n) for n in (1, 3, 3, 1)))
        self.assertEqual(polynomial.derivative().coefficients, tuple(Rational(n) for n in (3, 6, 3)))
        self.assertEqual(polynomial.evaluate(Rational(1, 2)), Rational(27, 8))
        self.assertEqual(polynomial.derivative().evaluate(Rational(1, 2)), Rational(27, 4))
        self.assertEqual(Polynomial.from_expression(polynomial.to_expression(), self.x), polynomial)

    def test_zero_normalization_and_degree(self):
        polynomial = Polynomial((0, 2, 0, 0), self.x)
        self.assertEqual(polynomial.degree, 1)
        self.assertEqual(polynomial.coefficients, (Rational(0), Rational(2)))
        zero = Polynomial((), self.x)
        self.assertTrue(zero.is_zero)
        self.assertEqual(zero.degree, -1)
        self.assertEqual(zero.derivative(), zero)
        self.assertEqual(Polynomial((1,), self.x).derivative(), zero)

    def test_derivative_monomial_rule_independent_expectation(self):
        for degree in range(1, 12):
            coefficient = Rational(3, 7)
            polynomial = Polynomial.from_expression(coefficient * self.x**degree, self.x)
            at = Rational(2, 3)
            expected = Rational(3 * degree * 2**(degree-1), 7 * 3**(degree-1))
            self.assertEqual(polynomial.derivative().evaluate(at), expected)

    def test_polynomial_identities_on_rational_points(self):
        rng = random.Random(9123)
        for _ in range(30):
            a, b, c = (Rational(rng.randrange(-6, 7), rng.randrange(1, 7)) for _ in range(3))
            at = Rational(rng.randrange(-5, 6), rng.randrange(1, 6))
            expression = (a*self.x + b)*(self.x + c)
            polynomial = Polynomial.from_expression(expression, self.x)
            self.assertEqual(polynomial.coefficients, Polynomial((b*c, a*c + b, a), self.x).coefficients)
            self.assertEqual(polynomial.evaluate(at), (a*at + b)*(at + c))
            self.assertEqual(polynomial.derivative().evaluate(at), 2*a*at + a*c + b)

    def test_scope_rejections(self):
        for expression in (self.x + symbol("y"), sqrt(2)*self.x,
                           self.x**(MAX_POLYNOMIAL_DEGREE + 1)):
            with self.assertRaises(UnsupportedOperation):
                Polynomial.from_expression(expression, self.x)
        with self.assertRaises(UnsupportedOperation):
            Polynomial((sqrt(2),), self.x)
        with self.assertRaises(TypeError):
            Polynomial((0.1, 1), self.x)


if __name__ == "__main__":
    unittest.main()
