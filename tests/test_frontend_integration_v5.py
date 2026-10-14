"""Multivariate source claims survive presentation, storage and hostile edits."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge.verification import verify_step


class MultivariateFrontendTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = mf.symbol("x"), mf.symbol("y")
        self.f = self.x**2*self.y + self.y**3

    def results(self):
        x, y, f = self.x, self.y, self.f
        return (
            mf.operations.expand_multivariate((x+y)**3, (x, y)),
            mf.operations.partial_diff(f, x),
            mf.operations.gradient(f, (x, y)),
            mf.operations.jacobian((f, x*y), (x, y)),
            mf.operations.hessian(f, (x, y)),
        )

    def test_public_values_and_explicit_axis_order(self):
        x, y, f = self.x, self.y, self.f
        self.assertEqual(mf.partial_diff(f, x), 2*x*y)
        self.assertEqual(mf.gradient(f, (x, y)), (2*x*y, x**2+3*y**2))
        self.assertEqual(mf.gradient(f, (y, x)), (x**2+3*y**2, 2*x*y))
        self.assertEqual(mf.jacobian((f, x*y), (x, y)),
                         ((2*x*y, x**2+3*y**2), (y, x)))
        self.assertEqual(mf.hessian(f, (x, y)), ((2*y, 2*x), (2*x, 6*y)))
        # Rational-only Matrix remains a distinct type with its existing contract.
        with self.assertRaises(mf.InvalidInput):
            mf.Matrix(mf.hessian(f, (x, y)))

    def test_parser_presentation_transport_and_workspace(self):
        parsed = mf.parse_expression("x^2*y + y^3", symbols={"x": self.x, "y": self.y})
        self.assertEqual(parsed, self.f)
        workspace = mf.Workspace()
        for result in self.results():
            with self.subTest(operation=result.method):
                self.assertIs(result.execution_status, mf.ExecutionStatus.COMPLETED, result.error)
                self.assertTrue(mf.verify(result).verified)
                self.assertEqual(mf.to_data(result)["version"], 2)
                encoded = mf.dumps(result)
                restored = mf.loads(encoded)
                self.assertEqual(mf.dumps(restored), encoded)
                self.assertTrue(mf.verify(restored).verified)
                self.assertTrue(mf.render_text(restored))
                self.assertTrue(mf.to_latex(restored))
                reference = workspace.put(result)
                self.assertEqual(workspace.get(reference), result)
        restored_workspace = mf.loads(mf.dumps(workspace))
        self.assertEqual(mf.dumps(restored_workspace), mf.dumps(workspace))

    def test_verifier_does_not_call_producers(self):
        from contextlib import ExitStack
        results = self.results()
        with ExitStack() as stack:
            for name in ("expand_multivariate", "partial_diff", "gradient", "jacobian", "hessian"):
                stack.enter_context(patch("mathforge.multivariate." + name,
                                          side_effect=AssertionError("producer called")))
            for result in results:
                self.assertTrue(mf.verify(result).verified, result.error)
                self.assertIs(verify_step(result.steps[0]).status, mf.CheckStatus.VERIFIED)

    def test_forged_value_source_axis_method_and_trace_fail(self):
        x, y = self.x, self.y
        original = mf.operations.gradient(self.f, (x, y))
        fake_value = (mf.Rational(0), mf.Rational(0))
        forged_step = replace(original.steps[0], outputs=(fake_value, ()))
        self.assertFalse(mf.verify(replace(original, value=fake_value, steps=(forged_step,))).verified)
        for forged in (replace(original, expression=self.f+1),
                       replace(original, method="hessian"),
                       replace(original, steps=()),
                       replace(original, exactness=mf.Exactness.APPROXIMATE),
                       replace(original, for_=x)):
            self.assertFalse(mf.verify(forged).verified)
        request = mf.OperationRequest("gradient", (("expression", self.f), ("variables", (y, x))))
        step = replace(original.steps[0], inputs=(request,))
        self.assertFalse(mf.verify(replace(original, request=request, steps=(step,))).verified)
        wrong_order = mf.operations.partial_diff(self.f, y)
        request = mf.OperationRequest("partial_diff", (("expression", self.f), ("variable", y), ("n", 2)))
        self.assertFalse(mf.verify(replace(wrong_order, request=request,
            steps=(replace(wrong_order.steps[0], inputs=(request,)),))).verified)

    def test_failures_are_explicit_and_serializable(self):
        x, y = self.x, self.y
        failures = (
            mf.operations.gradient(self.f, (x,)),
            mf.operations.gradient(self.f, (x, x)),
            mf.operations.gradient(self.f, ()),
            mf.operations.jacobian((), (x, y)),
            mf.operations.partial_diff(self.f, x, True),
            mf.operations.partial_diff(self.f, x, -1),
            mf.operations.gradient(mf.sqrt(2)*x, (x,)),
            mf.operations.hessian((x+y)**20, (x, y), limits=mf.ComputationLimits(max_work=10)),
        )
        for result in failures:
            with self.subTest(error=result.error):
                self.assertIsNot(result.execution_status, mf.ExecutionStatus.COMPLETED)
                self.assertIsNone(result.value)
                self.assertIsNotNone(result.error)
                self.assertEqual(mf.loads(mf.dumps(result)), result)
        self.assertEqual(failures[-1].error.code, "resource_limit_exceeded")

    def test_iterators_are_consumed_with_a_budget(self):
        result = mf.operations.gradient(self.f, iter((self.x, self.y)))
        self.assertEqual(result.unwrap(), mf.gradient(self.f, (self.x, self.y)))
        result = mf.operations.jacobian(iter((self.f, 1)), iter((self.x, self.y)))
        self.assertEqual(result.unwrap()[1], (mf.Rational(0), mf.Rational(0)))
        from itertools import repeat
        limited = mf.operations.gradient(self.f, repeat(self.x),
                                         limits=mf.ComputationLimits(max_nodes=100))
        self.assertEqual(limited.error.code, "resource_limit_exceeded")


if __name__ == "__main__":
    unittest.main()
