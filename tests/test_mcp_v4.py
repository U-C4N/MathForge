"""v0.4 limit and function-analysis calls through the JSON/text boundary."""

from dataclasses import replace
import unittest

import mathforge as mf
from mathforge.analysis_types import FunctionAnalysis, LimitValue
from mathforge.mcp.boundary import call_tool
from mathforge.mcp.schemas import input_schemas


UID = "72f494cc-f9cd-4f81-abfa-748d75509f6e"


class AnalysisBoundaryTests(unittest.TestCase):
    def invoke(self, name, **arguments):
        return call_tool("mathforge." + name, arguments)

    def result(self, response):
        self.assertFalse(response.is_error, response.payload)
        self.assertEqual(response.payload["api_version"], 1)
        self.assertIsNone(response.payload["error"])
        self.assertEqual(response.payload["result"]["version"], 3)
        self.assertTrue(response.payload["text"])
        self.assertTrue(response.payload["latex"])
        value = mf.from_data(response.payload["result"])
        self.assertTrue(mf.verify(value).verified)
        return value

    def test_limit_hole_sdk_parity_and_symbol_identity(self):
        response = self.invoke("limit", expression="(x^2-1)/(x-1)", point="1", symbol_ids={"x": UID})
        result = self.result(response)
        self.assertEqual(result.value, LimitValue("finite", mf.Rational(2)))
        x = mf.Symbol("x", uid=UID)
        function = mf.rational_function(x**2-1, x-1, variable=x)
        expected = mf.operations.limit(function, x, 1)
        self.assertEqual(mf.dumps(result), mf.dumps(expected))
        self.assertFalse(result.expression.is_defined_at(mf.Rational(1)))
        self.assertEqual(response.payload["symbols"], [{"name": "x", "uid": UID}])

    def test_exact_fraction_point_and_custom_variable(self):
        result = self.result(self.invoke("limit", expression="t^2", variable="t", point="-1/2"))
        self.assertEqual(result.value, LimitValue("finite", mf.Rational(1, 4)))
        self.assertEqual(result.for_.name, "t")

    def test_dne_is_a_completed_exact_value_and_directions_are_respected(self):
        result = self.result(self.invoke("limit", expression="1/x", point="0"))
        self.assertEqual(result.execution_status, mf.ExecutionStatus.COMPLETED)
        self.assertEqual(result.outcome, mf.Outcome.VALUE)
        self.assertEqual(result.exactness, mf.Exactness.EXACT)
        self.assertEqual(result.completeness, mf.Completeness.COMPLETE)
        self.assertEqual(result.value.kind, "does_not_exist")
        for side, kind in (("left", "negative_infinity"), ("right", "positive_infinity")):
            result = self.result(self.invoke("limit", expression="1/x", point="0", side=side))
            self.assertEqual(result.value.kind, kind)
        for point, kind in (("-inf", "negative_infinity"), ("+inf", "positive_infinity")):
            result = self.result(self.invoke("limit", expression="x", point=point))
            self.assertEqual(result.value.kind, kind)

    def test_function_analysis_preserves_domain_and_maximal_regions(self):
        result = self.result(self.invoke("analyze_function", expression="(x^2-1)/(x-1)"))
        self.assertIsInstance(result.value, FunctionAnalysis)
        self.assertFalse(result.value.domain.contains(1))
        self.assertEqual(len(result.value.excluded_points), 1)
        self.assertEqual(result.value.excluded_points[0].kind, "hole")
        self.assertEqual(len(result.value.monotonic_regions), 2)
        self.assertTrue(all(region.behavior == "increasing" for region in result.value.monotonic_regions))
        self.assertEqual(result.value.strict_extrema, ())

    def test_boundary_requires_exact_scalars_and_rejects_extra_arguments(self):
        for point in (0, True, False, 0.1, None, "0.1.2", "sqrt(-2)", "inf", "1/0", [], {}):
            with self.subTest(point=point):
                response = self.invoke("limit", expression="x", point=point)
                self.assertTrue(response.is_error)
                self.assertEqual(response.payload["error"]["field"], "point")
        for side in (True, None, "up", "", 1):
            response = self.invoke("limit", expression="x", point="0", side=side)
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["field"], "side")
        for operation, arguments in (("limit", {"expression": "x"}),
                                     ("limit", {"expression": "x", "point": "0", "unexpected": True}),
                                     ("analyze_function", {"expression": "x", "point": "0"})):
            self.assertTrue(self.invoke(operation, **arguments).is_error)

    def test_infinite_point_with_explicit_side_is_an_invalid_operation(self):
        for point in ("-inf", "+inf"):
            for side in ("left", "right"):
                response = self.invoke("limit", expression="x", point=point, side=side)
                self.assertTrue(response.is_error)
                result = mf.from_data(response.payload["result"])
                self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)
                self.assertEqual(response.payload["error"]["code"], "invalid_input")
                self.assertIsNone(result.value)

    def test_small_budgets_and_unsupported_inputs_never_become_math_conclusions(self):
        for operation, arguments in (("limit", {"expression": "1/(x-1)^3", "point": "1"}),
                                     ("analyze_function", {"expression": "x^5-x-1"})):
            response = self.invoke(operation, **arguments, limits={"max_work": 1})
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["code"], "resource_limit_exceeded")
            if response.payload["result"] is not None:
                self.assertIsNone(mf.from_data(response.payload["result"]).value)
        for operation, arguments in (("limit", {"expression": "sqrt(2)*x", "point": "0"}),
                                     ("analyze_function", {"expression": "sqrt(2)*x"})):
            response = self.invoke(operation, **arguments)
            self.assertTrue(response.is_error)
            result = mf.from_data(response.payload["result"])
            self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)

    def test_verify_accepts_v3_envelopes_and_rechecks_tampered_limits(self):
        for operation, arguments in (("limit", {"expression": "1/x", "point": "0"}),
                                     ("analyze_function", {"expression": "x^2"})):
            response = self.invoke(operation, **arguments)
            result = self.result(response)
            report = self.invoke("verify", result=response.payload["result"])
            self.assertFalse(report.is_error, report.payload)
            self.assertEqual(report.payload["api_version"], 1)
            self.assertTrue(mf.from_data(report.payload["result"]).verified)
            if operation == "limit":
                forged = replace(result, value=LimitValue("finite", mf.Rational(42)))
                rejected = self.invoke("verify", result=mf.to_data(forged))
                self.assertFalse(rejected.is_error)
                self.assertFalse(mf.from_data(rejected.payload["result"]).verified)

    def test_new_schemas_bound_point_and_side_without_changing_envelope_version(self):
        schemas = input_schemas()
        limit = schemas["mathforge.limit"]
        self.assertEqual(set(limit["required"]), {"expression", "point"})
        self.assertFalse(limit["additionalProperties"])
        self.assertEqual(limit["properties"]["side"]["enum"], ["both", "left", "right"])
        self.assertEqual(limit["properties"]["side"]["default"], "both")
        choices = limit["properties"]["point"]["anyOf"]
        self.assertEqual([choice["type"] for choice in choices], ["string", "string", "object"])
        self.assertEqual(choices[1]["enum"], ["-inf", "+inf"])
        self.assertEqual(schemas["mathforge.analyze_function"]["required"], ["expression"])


if __name__ == "__main__":
    unittest.main()
