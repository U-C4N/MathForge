"""Exact multivariate operations through the bounded JSON/text boundary."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge.mcp.boundary import MAX_EQUATIONS, MAX_VARIABLES, call_tool
from mathforge.mcp.schemas import input_schemas


UIDS = {"x": "00000000-0000-4000-8000-000000000501",
        "y": "00000000-0000-4000-8000-000000000502"}
TOOLS = ("expand_multivariate", "partial_diff", "gradient", "jacobian", "hessian")


class MultivariateBoundaryTests(unittest.TestCase):
    def invoke(self, name, **arguments):
        return call_tool("mathforge." + name, arguments)

    def result(self, response):
        self.assertFalse(response.is_error, response.payload)
        self.assertIsNone(response.payload["error"])
        self.assertEqual(response.payload["api_version"], 1)
        self.assertEqual(response.payload["result"]["version"], 2)
        self.assertTrue(response.payload["text"])
        self.assertTrue(response.payload["latex"])
        result = mf.from_data(response.payload["result"])
        self.assertIs(result.execution_status, mf.ExecutionStatus.COMPLETED)
        self.assertTrue(mf.verify(result).verified)
        return result

    def test_every_tool_matches_sdk_and_reuses_declared_symbol_ids(self):
        x, y = (mf.Symbol(name, uid=UIDS[name]) for name in ("x", "y"))
        symbols = {"x": x, "y": y}
        expression_text = "(x+y)^3 + x*y/2"
        expression = mf.parse_expression(expression_text, symbols=symbols)
        cases = (
            ("expand_multivariate", {"expression": expression_text},
             lambda: mf.operations.expand_multivariate(expression, (y, x))),
            ("partial_diff", {"expression": expression_text, "variable": "y", "order": 2},
             lambda: mf.operations.partial_diff(expression, y, n=2)),
            ("gradient", {"expression": expression_text},
             lambda: mf.operations.gradient(expression, (y, x))),
            ("jacobian", {"expressions": [expression_text, "x*y"]},
             lambda: mf.operations.jacobian((expression, x*y), (y, x))),
            ("hessian", {"expression": expression_text},
             lambda: mf.operations.hessian(expression, (y, x))),
        )
        for operation, arguments, expected in cases:
            with self.subTest(operation=operation):
                response = self.invoke(operation, **arguments, variables=["y", "x"], symbol_ids=UIDS)
                result = self.result(response)
                self.assertEqual(mf.dumps(result), mf.dumps(expected()))
                self.assertEqual(response.payload["symbols"],
                                 [{"name": "y", "uid": UIDS["y"]}, {"name": "x", "uid": UIDS["x"]}])
                checked = self.invoke("verify", result=response.payload["result"])
                self.assertFalse(checked.is_error, checked.payload)
                self.assertTrue(mf.from_data(checked.payload["result"]).verified)

    def test_vector_and_matrix_coordinates_follow_caller_order(self):
        x, y = (mf.Symbol(name, uid=UIDS[name]) for name in ("x", "y"))
        arguments = {"variables": ["y", "x"], "symbol_ids": UIDS}
        gradient = self.result(self.invoke("gradient", expression="x^2*y + 3*y^2", **arguments))
        self.assertIs(type(gradient.value), tuple)
        self.assertEqual(tuple(mf.evaluate(item, {x: 2, y: 3}) for item in gradient.value),
                         (mf.Rational(22), mf.Rational(12)))
        jacobian = self.result(self.invoke("jacobian", expressions=["x*y", "x^2 + 3*y"], **arguments))
        hessian = self.result(self.invoke("hessian", expression="x^2*y + 3*y^2", **arguments))
        for value, expected in ((jacobian.value, ((2, 3), (3, 4))),
                                (hessian.value, ((6, 4), (4, 6)))):
            self.assertIs(type(value), tuple)
            self.assertTrue(all(type(row) is tuple for row in value))
            self.assertEqual(tuple(tuple(mf.evaluate(item, {x: 2, y: 3}) for item in row)
                                   for row in value),
                             tuple(tuple(mf.Rational(item) for item in row) for row in expected))

    def test_partial_derivative_default_and_zero_order(self):
        for order, expected in ((None, 6), (0, 18), (3, 0)):
            arguments = {} if order is None else {"order": order}
            response = self.invoke("partial_diff", expression="x^2*y/2", variables=["x", "y"],
                                   variable="x", symbol_ids=UIDS, **arguments)
            result = self.result(response)
            bindings = {mf.Symbol(name, uid=uid): value for (name, uid), value in zip(UIDS.items(), (6, 1))}
            self.assertEqual(mf.evaluate(result.value, bindings), mf.Rational(expected))

    def test_declared_names_and_partial_variable_are_required(self):
        for operation in TOOLS:
            base = {"expressions": ["x"]} if operation == "jacobian" else {"expression": "x"}
            if operation == "partial_diff":
                base["variable"] = "x"
            for variables in (None, [], "x", ["x", "x"], [True], ["sqrt"],
                              ["x" + str(i) for i in range(MAX_VARIABLES + 1)]):
                with self.subTest(operation=operation, variables=variables):
                    response = self.invoke(operation, **base, variables=variables)
                    self.assertTrue(response.is_error)
                    self.assertEqual(response.payload["error"]["field"], "variables")
            self.assertTrue(self.invoke(operation, **base).is_error)
            self.assertTrue(self.invoke(operation, **base, variables=["x"], unknown=True).is_error)
        for variable in (None, True, {}, "z"):
            response = self.invoke("partial_diff", expression="x*y", variables=["x", "y"], variable=variable)
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["field"], "variable")
        for order in (-1, 1025, True, 1.5, "2", None):
            response = self.invoke("partial_diff", expression="x", variables=["x"], variable="x", order=order)
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["field"], "order")
        self.assertTrue(self.invoke("partial_diff", expression="x", variables=["x"]).is_error)

    def test_jacobian_rejects_invalid_arrays_and_preserves_parse_location(self):
        for expressions in (None, [], "x", ["x"] * (MAX_EQUATIONS + 1)):
            response = self.invoke("jacobian", expressions=expressions, variables=["x"])
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["field"], "expressions")
        for expression in (True, {}, "y", "x +"):
            response = self.invoke("jacobian", expressions=["x", expression], variables=["x"])
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["field"], "expressions[1]")
            self.assertEqual(response.payload["error"]["stage"], "parsing")
            if type(expression) is str:
                self.assertIsInstance(response.payload["error"]["start"], int)

    def test_symbol_identity_validation_is_shared_by_multivariate_tools(self):
        for ids in ({"z": UIDS["x"]}, {"x": UIDS["x"], "y": UIDS["x"]}, {"y": "bad UUID"}):
            response = self.invoke("gradient", expression="x*y", variables=["x", "y"], symbol_ids=ids)
            self.assertTrue(response.is_error)
            self.assertTrue(response.payload["error"]["field"].startswith("symbol_ids"))
        response = self.invoke("gradient", expression="x*z", variables=["x", "y"])
        self.assertTrue(response.is_error)
        self.assertEqual(response.payload["error"]["stage"], "parsing")
        self.assertEqual(response.payload["error"]["field"], "expression")

    def test_resource_and_output_bounds_do_not_return_truncated_values(self):
        for operation, arguments, limits in (
            ("expand_multivariate", {"expression": "(x+y)^8"}, {"max_work": 1}),
            ("jacobian", {"expressions": ["x", "y"]}, {"max_matrix_entries": 3}),
            ("hessian", {"expression": "x*y"}, {"max_matrix_entries": 3}),
        ):
            response = self.invoke(operation, **arguments, variables=["x", "y"], limits=limits)
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["code"], "resource_limit_exceeded")
            if response.payload["result"] is not None:
                self.assertIsNone(mf.from_data(response.payload["result"]).value)
        with patch("mathforge.mcp.boundary.MAX_RESPONSE_BYTES", 100):
            response = self.invoke("gradient", expression="x*y", variables=["x", "y"])
        self.assertTrue(response.is_error)
        self.assertEqual(response.payload["error"]["code"], "response_too_large")
        self.assertIsNone(response.payload["result"])

    def test_verify_rejects_tampered_multivariate_values(self):
        response = self.invoke("gradient", expression="x*y", variables=["x", "y"])
        result = self.result(response)
        forged = replace(result, value=(mf.Rational(999), mf.Rational(999)))
        checked = self.invoke("verify", result=mf.to_data(forged))
        self.assertFalse(checked.is_error)
        self.assertFalse(mf.from_data(checked.payload["result"]).verified)

    def test_capabilities_and_schemas_publish_bounded_polynomial_scope(self):
        capabilities = mf.capabilities()
        self.assertEqual(len(capabilities["tools"]), 30)
        operations = {item["name"]: item for item in capabilities["operations"]}
        schemas = input_schemas()
        for name in TOOLS:
            self.assertEqual(operations[name]["mcp_tool"], "mathforge." + name)
            self.assertIn("polynomial", operations[name]["scope"])
            schema = schemas["mathforge." + name]
            self.assertFalse(schema["additionalProperties"])
            variables = schema["properties"]["variables"]
            self.assertTrue(variables["uniqueItems"])
            self.assertEqual(variables["minItems"], 1)
            self.assertEqual(variables["maxItems"], MAX_VARIABLES)
        self.assertEqual(set(schemas["mathforge.partial_diff"]["required"]),
                         {"expression", "variables", "variable"})
        expressions = schemas["mathforge.jacobian"]["properties"]["expressions"]
        self.assertEqual(expressions["minItems"], 1)
        self.assertEqual(expressions["maxItems"], MAX_EQUATIONS)


if __name__ == "__main__":
    unittest.main()
