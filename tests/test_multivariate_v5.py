"""Sparse rational multivariate calculus and independent coefficient checks."""

from dataclasses import FrozenInstanceError
from unittest import TestCase, main, mock

import mathforge as mf
from mathforge import multivariate as mv
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Mul, Pow, Rational


class MultivariateV5Tests(TestCase):
    def setUp(self):
        self.x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000051")
        self.y = mf.Symbol("y", uid="00000000-0000-4000-8000-000000000052")
        self.z = mf.Symbol("z", uid="00000000-0000-4000-8000-000000000053")
        self.axes = (self.x, self.y)

    def test_expansion_collects_cancels_and_preserves_exact_fractions(self):
        x, y = self.axes
        expression = (x + y)**3 - x**3 - y**3 + Rational(1, 3) * x * y
        expected = 3*x**2*y + 3*x*y**2 + Rational(1, 3)*x*y
        self.assertEqual(mv.expand_multivariate(expression, self.axes), expected)
        self.assertEqual(mv.expand_multivariate((x+y)*(x-y) - x*x + y*y, self.axes), 0*x)
        self.assertEqual(mv.expand_multivariate(7, self.axes), Rational(7))
        self.assertEqual(mv.expand_multivariate(expression, tuple(reversed(self.axes))), expected)

    def test_partial_derivatives_infer_axes_and_handle_order_zero_and_absence(self):
        x, y = self.axes
        expression = Rational(1, 6)*x**3*y**2 + 5*x*y
        self.assertEqual(mv.partial_diff(expression, x), Rational(1, 2)*x**2*y**2 + 5*y)
        self.assertEqual(mv.partial_diff(expression, x, 3), y**2)
        self.assertEqual(mv.partial_diff(expression, x, 4), Rational(0))
        self.assertEqual(mv.partial_diff(expression, self.z), Rational(0))
        self.assertEqual(mv.partial_diff((x+y)**2, x, 0), x**2 + 2*x*y + y**2)
        self.assertEqual(mv.partial_diff(5, x), Rational(0))

    def test_gradient_jacobian_and_hessian_obey_requested_axis_order(self):
        x, y = self.axes
        expression = x**3*y + Rational(1, 2)*y**2 + 5*x
        self.assertEqual(mv.gradient(expression, (y, x)), (x**3 + y, 3*x**2*y + 5))
        self.assertEqual(mv.jacobian((expression, x*y, 7), (y, x)),
                         ((x**3 + y, 3*x**2*y + 5), (x, y), (Rational(0), Rational(0))))
        result = mv.hessian(expression, (y, x))
        self.assertEqual(result, ((Rational(1), 3*x**2), (3*x**2, 6*x*y)))
        self.assertIs(result[0][1], result[1][0])

    def test_shared_expansion_is_used_once_per_vector_operation(self):
        expression = (self.x + self.y + self.z)**5
        axes = (*self.axes, self.z)
        for function, first in ((mv.gradient, expression), (mv.hessian, expression),
                                (mv.jacobian, (expression, expression))):
            with self.subTest(function=function.__name__):
                with mock.patch.object(mv, "_convert_many", wraps=mv._convert_many) as convert:
                    function(first, axes)
                self.assertEqual(convert.call_count, 1)

    def test_sparse_large_degrees_and_orders_do_not_allocate_dense_arrays(self):
        source = self.x**1_000_000 + self.y
        with computation(ComputationLimits(max_work=10_000, max_nodes=5_000)):
            self.assertEqual(mv.partial_diff(source, self.x), 1_000_000*self.x**999_999)
            self.assertEqual(mv.partial_diff(source, self.x, 10**100), Rational(0))

    def test_result_nodes_and_nested_tuples_are_immutable(self):
        result = mv.hessian(self.x**3*self.y, self.axes)
        with self.assertRaises(TypeError):
            result[0][0] = Rational(0)
        with self.assertRaises(FrozenInstanceError):
            result[0][0].factors = ()

    def test_axis_contract_and_identity_are_strict(self):
        for variables in ((), [self.x], self.x, (self.x, self.x), ("x",)):
            with self.subTest(variables=variables), self.assertRaises(InvalidInput):
                mv.gradient(self.x, variables)
        same_name = mf.symbol("x")
        with self.assertRaises(UnsupportedOperation):
            mv.expand_multivariate(same_name, (self.x,))
        self.assertEqual(mv.gradient(self.x*same_name, (self.x, same_name)), (same_name, self.x))
        renamed = mf.Symbol("renamed", uid=self.x.uid)
        for variables in ((self.x, renamed), (self.x,)):
            with self.assertRaises(InvalidInput):
                mv.expand_multivariate(renamed, variables)
        with self.assertRaises(InvalidInput):
            mv.partial_diff(Add((self.x, renamed)), self.y)

    def test_unsupported_symbols_are_rejected_even_when_derivative_is_zero(self):
        constrained = mf.symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))
        integer = mf.symbol("n", domain=mf.Integers)
        for variable in (constrained, integer):
            with self.assertRaises(UnsupportedOperation):
                mv.gradient(0, (variable,))
            with self.assertRaises(UnsupportedOperation):
                mv.partial_diff(variable, self.x, 0)
            with self.assertRaises(UnsupportedOperation):
                mv.partial_diff(Mul((Rational(0), variable)), self.x)

    def test_every_raw_symbol_is_validated_before_cancellation(self):
        foreign = mf.symbol("foreign")
        for expression in (Mul((Rational(0), foreign)), Pow(foreign, 0)):
            with self.assertRaises(UnsupportedOperation):
                mv.expand_multivariate(expression, self.axes)

    def test_nonrational_and_transport_inputs_are_rejected(self):
        for expression in (mf.sqrt(2), mf.Polynomial((1, 1), self.x),
                           mf.rational_function(1, self.x, variable=self.x)):
            with self.subTest(expression=expression), self.assertRaises(UnsupportedOperation):
                mv.partial_diff(expression, self.x)
        for expression in (True, 1.0, "x", None):
            with self.assertRaises(InvalidInput):
                mv.expand_multivariate(expression, self.axes)
        for order in (-1, True, 1.0, "1"):
            with self.assertRaises(InvalidInput):
                mv.partial_diff(self.x, self.x, order)
        for expressions in ((), [self.x], self.x):
            with self.assertRaises(InvalidInput):
                mv.jacobian(expressions, self.axes)

    def test_negative_powers_cannot_enter_through_a_forged_node(self):
        expression = Pow(self.x, 2)
        object.__setattr__(expression, "exponent", -1)
        with self.assertRaises(UnsupportedOperation):
            mv.expand_multivariate(expression, self.axes)

    def test_resource_limits_bound_work_nodes_bits_depth_and_dimensions(self):
        expression = (self.x + self.y)**12
        for limits in (ComputationLimits(max_work=10), ComputationLimits(max_nodes=20),
                       ComputationLimits(max_integer_bits=4), ComputationLimits(max_depth=2)):
            with self.subTest(limits=limits), self.assertRaises(ResourceLimitError):
                mv.expand_multivariate(expression, self.axes, limits=limits)
        with self.assertRaises(ResourceLimitError):
            mv.hessian(self.x*self.y, self.axes, limits=ComputationLimits(max_matrix_entries=3))
        with self.assertRaises(ResourceLimitError):
            mv.jacobian((self.x, self.y), self.axes, limits=ComputationLimits(max_matrix_entries=3))
        with self.assertRaises(ResourceLimitError):
            mv.expand_multivariate(Rational(2**100), self.axes,
                                   limits=ComputationLimits(max_integer_bits=64))
        with self.assertRaises(ResourceLimitError):
            mv.partial_diff(self.x, self.x, 2**100, limits=ComputationLimits(max_integer_bits=64))

    def cases(self):
        expression = Rational(1, 6)*self.x**3*self.y**2 + 5*self.y
        return (
            ("expand_multivariate", {"expression": expression, "variables": self.axes},
             mv.expand_multivariate(expression, self.axes)),
            ("partial_diff", {"expression": expression, "variable": self.x, "n": 2},
             mv.partial_diff(expression, self.x, 2)),
            ("gradient", {"expression": expression, "variables": self.axes},
             mv.gradient(expression, self.axes)),
            ("jacobian", {"expressions": (expression, self.x*self.y), "variables": self.axes},
             mv.jacobian((expression, self.x*self.y), self.axes)),
            ("hessian", {"expression": expression, "variables": self.axes},
             mv.hessian(expression, self.axes)),
        )

    def test_checker_is_independent_of_producer_calculus_and_assembly(self):
        cases = self.cases()
        forbidden = AssertionError("Checker called producer logic")
        with (mock.patch.object(mv, "expand_multivariate", side_effect=forbidden),
              mock.patch.object(mv, "partial_diff", side_effect=forbidden),
              mock.patch.object(mv, "gradient", side_effect=forbidden),
              mock.patch.object(mv, "jacobian", side_effect=forbidden),
              mock.patch.object(mv, "hessian", side_effect=forbidden),
              mock.patch.object(mv, "_differentiate", side_effect=forbidden),
              mock.patch.object(mv, "_to_expression", side_effect=forbidden)):
            for operation, args, value in cases:
                with self.subTest(operation=operation):
                    self.assertTrue(mv.check_operation(operation, args, value, ()))

    def test_checker_rejects_wrong_values_shapes_and_evidence(self):
        for operation, args, value in self.cases():
            with self.subTest(operation=operation):
                self.assertFalse(mv.check_operation(operation, args, Rational(999), ()))
                self.assertFalse(mv.check_operation(operation, args, value, []))
                self.assertFalse(mv.check_operation(operation, args, value, ("invented",)))
                self.assertFalse(mv.check_operation(operation, {**args, "extra": 0}, value, ()))
                self.assertFalse(mv.check_operation(operation, {}, value, ()))
                if type(value) is tuple:
                    self.assertFalse(mv.check_operation(operation, args, list(value), ()))
                    self.assertFalse(mv.check_operation(operation, args, value[:-1], ()))
                    self.assertFalse(mv.check_operation(operation, args, tuple(reversed(value)), ()))
                if "variables" in args:
                    reordered = {**args, "variables": tuple(reversed(self.axes))}
                    if operation != "expand_multivariate":
                        self.assertFalse(mv.check_operation(operation, reordered, value, ()))
        self.assertFalse(mv.check_operation("unknown", {}, Rational(0), ()))

    def test_checker_validates_derivative_order_and_same_uuid_metadata(self):
        args = {"expression": self.x**3*self.y, "variable": self.x, "n": 1}
        value = mv.partial_diff(args["expression"], self.x)
        for n in (0, 2, -1, True, 1.0):
            self.assertFalse(mv.check_operation("partial_diff", {**args, "n": n}, value, ()))
        renamed = mf.Symbol("renamed", uid=self.x.uid)
        self.assertFalse(mv.check_operation("partial_diff", {**args, "variable": renamed}, value, ()))
        self.assertFalse(mv.check_operation("partial_diff", args, 3*renamed**2*self.y, ()))

    def test_checker_rejects_unexpanded_uncollected_and_noncanonical_claims(self):
        args = {"expression": (self.x+self.y)**2, "variables": self.axes}
        self.assertFalse(mv.check_operation("expand_multivariate", args, args["expression"], ()))
        value = mv.expand_multivariate(args["expression"], self.axes)
        self.assertTrue(mv.check_operation("expand_multivariate", args, value, ()))
        self.assertFalse(mv.check_operation("expand_multivariate", args, Add(tuple(reversed(value.terms))), ()))
        twice = {"expression": 2*self.x, "variables": self.axes}
        self.assertFalse(mv.check_operation("expand_multivariate", twice, Add((self.x, self.x)), ()))
        self.assertFalse(mv.check_operation("expand_multivariate", twice, Mul((self.x, Rational(2))), ()))

    def test_checker_resource_exhaustion_propagates(self):
        operation, args, value = self.cases()[0]
        with computation(ComputationLimits(max_work=1)):
            with self.assertRaises(ResourceLimitError):
                mv.check_operation(operation, args, value, ())

    def test_public_operations_round_trip_and_verify(self):
        for operation, args, expected in self.cases():
            result = getattr(mf.operations, operation)(**args)
            with self.subTest(operation=operation):
                self.assertIs(result.execution_status, mf.ExecutionStatus.COMPLETED, result.error)
                self.assertEqual(result.value, expected)
                self.assertTrue(mf.verify(result).verified)
                restored = mf.loads(mf.dumps(result))
                self.assertTrue(mf.verify(restored).verified)
                self.assertEqual(getattr(mf, operation)(**args), expected)


if __name__ == "__main__":
    main()
