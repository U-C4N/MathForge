"""Exact arithmetic, selector forgery, independent checking and bounded work."""

from dataclasses import replace
from unittest import TestCase, main, mock

import mathforge as mf
from mathforge import algebraic_arithmetic as aa
from mathforge.algebraic import compare_real
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ResourceLimitError


class AlgebraicArithmeticTests(TestCase):
    def setUp(self):
        self.x = mf.symbol("x")
        self.cube = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)

    def run_operation(self, operation, **args):
        value, evidence = aa.produce(operation, args)
        self.assertTrue(aa.check_operation(operation, args, value, evidence))
        return value, evidence

    def normal(self, expression):
        return self.run_operation("algebraic", expression=expression)[0]

    def test_rational_and_radical_normalization(self):
        self.assertEqual(self.normal(3), mf.Rational(3))
        self.assertEqual(self.normal(mf.sqrt(2)), mf.RealAlgebraicRoot((-2, 0, 1), 1))
        total = self.normal(mf.sqrt(2) + mf.sqrt(3))
        self.assertEqual(total, mf.RealAlgebraicRoot((1, 0, -10, 0, 1), 3))
        self.assertEqual(self.normal((mf.sqrt(2)+mf.sqrt(3))**2),
                         self.normal(5 + 2*mf.sqrt(6)))

    def test_roots_from_nonminimal_polynomials_are_reduced_without_mutation(self):
        polynomial = (self.x**3-2)*(self.x**2-3)
        source = mf.RealAlgebraicRoot(tuple(c.numerator for c in mf.Polynomial.from_expression(polynomial, self.x).coefficients), 1)
        before, hashed = mf.dumps(source), hash(source)
        self.assertEqual(self.normal(source), self.cube)
        self.assertEqual(mf.dumps(source), before)
        self.assertEqual(hash(source), hashed)
        self.assertNotEqual(source, self.cube)
        self.assertEqual(compare_real(source, self.cube), 0)
        self.assertEqual(self.normal(mf.RealAlgebraicRoot((-2, 2), 0)), mf.Rational(1))

    def test_signed_powers_and_reciprocals(self):
        for n, expected in ((0, mf.Rational(1)), (1, self.cube), (3, mf.Rational(2)),
                            (6, mf.Rational(4)), (-3, mf.Rational(1, 2))):
            with self.subTest(n=n):
                self.assertEqual(self.run_operation("algebraic_pow", base=self.cube, n=n)[0], expected)
        reciprocal = self.run_operation("algebraic_div", left=1, right=mf.sqrt(2))[0]
        self.assertEqual(reciprocal, mf.RealAlgebraicRoot((-1, 0, 2), 1))
        self.assertEqual(self.run_operation("algebraic_div", left=self.cube, right=self.cube)[0], mf.Rational(1))
        self.assertEqual(self.run_operation("algebraic_pow", base=0, n=0)[0], mf.Rational(1))

    def test_negative_and_cross_field_products_choose_correct_conjugate(self):
        for expression, expected in ((-mf.sqrt(2), mf.RealAlgebraicRoot((-2, 0, 1), 0)),
                                     (mf.sqrt(2)*mf.sqrt(3), mf.RealAlgebraicRoot((-6, 0, 1), 1)),
                                     (-mf.sqrt(2)*mf.sqrt(3), mf.RealAlgebraicRoot((-6, 0, 1), 0))):
            self.assertEqual(self.normal(expression), expected)
        value = self.run_operation("algebraic_div", left=-mf.sqrt(2), right=mf.sqrt(3))[0]
        self.assertEqual(value, mf.RealAlgebraicRoot((-2, 0, 3), 0))

    def test_minimal_polynomial_is_monic_over_q_and_binds_variable(self):
        value, evidence = self.run_operation("minimal_polynomial", expression=mf.sqrt(mf.Rational(1, 2)), variable=self.x)
        self.assertEqual(value, mf.Polynomial((mf.Rational(-1, 2), 0, 1), self.x))
        other = mf.symbol("x")
        self.assertFalse(aa.check_operation("minimal_polynomial", dict(expression=mf.sqrt(mf.Rational(1, 2)), variable=other), value, evidence))
        self.assertEqual(self.run_operation("minimal_polynomial", expression=mf.Rational(2, 3), variable=self.x)[0],
                         mf.Polynomial((mf.Rational(-2, 3), 1), self.x))

    def test_compare_closed_expressions_and_equality_across_representations(self):
        for left, right, expected in ((mf.sqrt(2)+mf.sqrt(3), 3, 1),
                                       ((mf.sqrt(2)+mf.sqrt(3))**2, 5+2*mf.sqrt(6), 0),
                                       (-mf.sqrt(2)-mf.sqrt(3), 0, -1)):
            value, evidence = self.run_operation("compare_real", left=left, right=right)
            self.assertEqual(value, expected)
            self.assertFalse(aa.check_operation("compare_real", dict(left=left, right=right), True, evidence))

    def test_explicit_evaluation_and_simultaneous_bindings(self):
        value, _ = self.run_operation("evaluate_algebraic", expression=self.x**3, bindings=((self.x, self.cube),))
        self.assertEqual(value, mf.Rational(2))
        self.assertNotEqual(mf.evaluate(self.x**3, {self.x: self.cube}), mf.Rational(2))
        y = mf.symbol("y")
        self.assertEqual(self.run_operation("evaluate_algebraic", expression=self.x+y,
                         bindings=((self.x, mf.sqrt(2)), (y, mf.sqrt(3))))[0], self.normal(mf.sqrt(2)+mf.sqrt(3)))
        with self.assertRaises(UnsupportedOperation):
            aa.produce("evaluate_algebraic", dict(expression=self.x+y, bindings=((self.x, y), (y, 1))))

    def test_rational_function_evaluation_preserves_holes(self):
        f = mf.rational_function(self.x**3-2, self.x-1, variable=self.x)
        self.assertEqual(self.run_operation("evaluate_algebraic", expression=f, bindings=((self.x, self.cube),))[0], mf.Rational(0))
        hole = mf.rational_function(self.x**2-2, self.x**2-2, variable=self.x)
        with self.assertRaises(InvalidInput) as raised:
            aa.produce("evaluate_algebraic", dict(expression=hole, bindings=((self.x, mf.sqrt(2)),)))
        self.assertEqual(raised.exception.code, "undefined_at_point")

    def test_invalid_scope_and_division_by_mathematically_zero(self):
        for operation, args in (
            ("algebraic", dict(expression=self.x)),
            ("algebraic", dict(expression=1.5)),
            ("algebraic_pow", dict(base=self.cube, n=True)),
            ("algebraic_pow", dict(base=0, n=-1)),
            ("algebraic_div", dict(left=1, right=self.cube**3-2)),
            ("evaluate_algebraic", dict(expression=self.x, bindings=((self.x, 1), (self.x, 2)))),
        ):
            with self.subTest(operation=operation, args=args):
                with self.assertRaises((InvalidInput, UnsupportedOperation, TypeError)):
                    aa.produce(operation, args)

    def test_checker_rejects_value_source_and_root_selector_mutations(self):
        args = dict(expression=mf.sqrt(2)+mf.sqrt(3))
        value, evidence = self.run_operation("algebraic", **args)
        wrong = mf.RealAlgebraicRoot(value.integer_coefficients, 0)
        self.assertFalse(aa.check_operation("algebraic", args, wrong, evidence))
        self.assertFalse(aa.check_operation("algebraic", dict(expression=mf.sqrt(2)-mf.sqrt(3)), value, evidence))
        nodes = list(evidence[1])
        last = list(nodes[-1])
        certificate = list(last[3])
        certificate[0] = certificate[0] + 1
        last[3] = tuple(certificate)
        nodes[-1] = tuple(last)
        forged = (evidence[0], tuple(nodes), evidence[2])
        self.assertFalse(aa.check_operation("algebraic", args, value, forged))
        nodes = list(evidence[1])
        last = list(nodes[-1])
        certificate = list(last[3])
        certificate[3] = ((mf.Rational(-2), mf.Rational(-1)), *certificate[3][1:])
        last[3] = tuple(certificate)
        nodes[-1] = tuple(last)
        self.assertFalse(aa.check_operation("algebraic", args, value, (evidence[0], tuple(nodes), evidence[2])))

    def test_malformed_extra_and_cyclic_trace_nodes_are_rejected(self):
        args = dict(expression=mf.sqrt(2)+mf.sqrt(3))
        value, evidence = self.run_operation("algebraic", **args)
        nodes = evidence[1]
        for malformed in (None, (), ("bad", nodes, evidence[2]),
                          (evidence[0], nodes + (("rational", (mf.Rational(10),), mf.Rational(10), ()),), evidence[2]),
                          (evidence[0], nodes, (True,))):
            self.assertFalse(aa.check_operation("algebraic", args, value, malformed))
        last = list(nodes[-1])
        last[1] = (len(nodes)-1, len(nodes)-1)
        self.assertFalse(aa.check_operation("algebraic", args, value, (evidence[0], (*nodes[:-1], tuple(last)), evidence[2])))

    def test_checker_never_calls_arithmetic_or_factorization_producers(self):
        args = dict(expression=(mf.sqrt(2)+mf.sqrt(3))**3)
        value, evidence = self.run_operation("algebraic", **args)
        blocked = AssertionError("producer called during checking")
        with (mock.patch.object(aa, "produce", side_effect=blocked),
              mock.patch.object(aa._Engine, "read", side_effect=blocked),
              mock.patch.object(aa, "_annihilator", side_effect=blocked),
              mock.patch.object(aa, "_interpolate", side_effect=blocked),
              mock.patch("mathforge.polynomial_exact.factor_with_evidence", side_effect=blocked),
              mock.patch("mathforge.polynomial_exact.resultant", side_effect=blocked)):
            self.assertTrue(aa.check_operation("algebraic", args, value, evidence))

    def test_checker_uses_independent_source_substitution_and_semantics(self):
        cases = (("algebraic", dict(expression=(mf.sqrt(2)+mf.sqrt(3))**3)),
                 ("evaluate_algebraic", dict(expression=self.x**3, bindings=((self.x, self.cube),))),
                 ("algebraic_div", dict(left=mf.sqrt(2), right=mf.sqrt(3))),
                 ("algebraic_pow", dict(base=self.cube, n=-2)))
        results = [(operation, args, *self.run_operation(operation, **args)) for operation, args in cases]
        producer_helpers = ("_recipe", "_operation_recipe", "_evaluation_recipe", "_polynomial_recipe",
                            "_sample_polynomials", "_enclosure", "_rational_result", "_defining",
                            "_modular_power", "_combine", "_bindings", "substitute_expression")
        patches = [mock.patch.object(aa, name, side_effect=AssertionError("producer semantics called"))
                   for name in producer_helpers]
        for patch in patches:
            patch.start()
        try:
            for operation, args, value, evidence in results:
                self.assertTrue(aa.check_operation(operation, args, value, evidence))
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_injected_producer_source_and_direct_arithmetic_faults_are_detected(self):
        args = dict(expression=mf.sqrt(2)+mf.sqrt(3))
        with mock.patch.object(aa, "_recipe", return_value=("rational", mf.Rational(0))):
            value, evidence = aa.produce("algebraic", args)
            self.assertFalse(aa.check_operation("algebraic", args, value, evidence))
        eval_args = dict(expression=self.x**2, bindings=((self.x, mf.Rational(3)),))
        with mock.patch.object(aa, "substitute_expression", return_value=mf.Rational(0)):
            value, evidence = aa.produce("evaluate_algebraic", eval_args)
            self.assertFalse(aa.check_operation("evaluate_algebraic", eval_args, value, evidence))
        power_args = dict(base=3, n=2)
        with mock.patch.object(aa, "_rational_result", return_value=mf.Rational(1)):
            value, evidence = aa.produce("algebraic_pow", power_args)
            self.assertFalse(aa.check_operation("algebraic_pow", power_args, value, evidence))

    def test_injected_resultant_specialization_and_conjugate_faults_are_detected(self):
        args = dict(expression=mf.sqrt(2)+mf.sqrt(3))

        def false_specialization(kind, values, point, exponent=None):
            return aa._poly((0, 1)), aa._poly((point-3,))

        with mock.patch.object(aa, "_sample_polynomials", side_effect=false_specialization):
            value, evidence = aa.produce("algebraic", args)
            self.assertEqual(value, mf.Rational(3))
            self.assertFalse(aa.check_operation("algebraic", args, value, evidence))
        enclosure = aa._enclosure

        def wrong_conjugate(kind, intervals, exponent=None):
            return (mf.Rational(-4), mf.Rational(-3)) if kind == "add" else enclosure(kind, intervals, exponent)

        with mock.patch.object(aa, "_enclosure", side_effect=wrong_conjugate):
            value, evidence = aa.produce("algebraic", args)
            self.assertEqual(value.real_index, 0)
            self.assertFalse(aa.check_operation("algebraic", args, value, evidence))

    def test_budgets_apply_to_producer_and_checker(self):
        args = dict(expression=mf.sqrt(2)+mf.sqrt(3))
        value, evidence = self.run_operation("algebraic", **args)
        for limits in (mf.ComputationLimits(max_work=1), mf.ComputationLimits(max_evidence_items=1),
                       mf.ComputationLimits(max_refinements=1, max_work=1)):
            with self.assertRaises(ResourceLimitError):
                aa.produce("algebraic", args, limits=limits)
            with self.assertRaises(ResourceLimitError):
                aa.check_operation("algebraic", args, value, evidence, limits=limits)

    def test_coefficient_growth_is_rejected_before_resultant_sampling(self):
        args = dict(expression=mf.sqrt(2)+mf.sqrt(3))
        with mock.patch.object(aa, "_sample_polynomials", side_effect=AssertionError("sample allocation began")):
            with self.assertRaises(ResourceLimitError):
                aa.produce("algebraic", args, limits=mf.ComputationLimits(max_integer_bits=12))
        value, evidence = aa.produce("algebraic", args, limits=mf.ComputationLimits(max_integer_bits=128))
        self.assertTrue(aa.check_operation("algebraic", args, value, evidence))
        with mock.patch.object(aa, "_modular_power", side_effect=AssertionError("large modular power began")):
            with self.assertRaises(ResourceLimitError):
                aa.produce("algebraic_pow", dict(base=self.cube, n=10**20),
                           limits=mf.ComputationLimits(max_integer_bits=128))

    def test_forged_nested_rational_and_root_records_are_rejected(self):
        rational = object.__new__(mf.Rational)
        object.__setattr__(rational, "numerator", 2)
        object.__setattr__(rational, "denominator", 2)
        root = object.__new__(mf.RealAlgebraicRoot)
        object.__setattr__(root, "integer_coefficients", (-4, 0, 2))
        object.__setattr__(root, "real_index", 1)
        for source in (rational, root, mf.Add((rational, mf.sqrt(2)))):
            with self.assertRaises(InvalidInput):
                aa.produce("algebraic", dict(expression=source))
        malformed_evidence = ("algebraic_trace_v1", (("rational", (rational,), rational, ()),), (0,))
        self.assertFalse(aa.check_operation("algebraic", dict(expression=rational), rational, malformed_evidence))
        args = dict(expression=mf.sqrt(2))
        value, evidence = self.run_operation("algebraic", **args)
        nodes = list(evidence[1])
        first = list(nodes[0])
        first[2] = root
        nodes[0] = tuple(first)
        self.assertFalse(aa.check_operation("algebraic", args, root, (evidence[0], tuple(nodes), evidence[2])))


if __name__ == "__main__":
    main()
