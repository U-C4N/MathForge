"""Domain-aware parsing and fixed-registry canonical computation over MCP."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge.capabilities import area_tools
from mathforge.dispatch_v5 import operation_arguments
from mathforge.mcp.boundary import _FIELDS, call_tool
from mathforge.mcp.schemas import input_schemas
from mathforge.operations_v5 import run


X_ID = "00000000-0000-4000-8000-000000000701"
Y_ID = "00000000-0000-4000-8000-000000000702"


class V7BoundaryTests(unittest.TestCase):
    def compute(self, operation, **arguments):
        return call_tool("mathforge.compute", {"operation": operation,
            "arguments": {key: m.to_data(value) for key, value in arguments.items()}})

    def result(self, response):
        self.assertFalse(response.is_error, response.payload)
        self.assertIsNone(response.payload["error"])
        self.assertTrue(response.payload["text"])
        self.assertTrue(response.payload["latex"])
        return m.from_data(response.payload["result"])

    def test_v07_curated_tools_and_registry_enum(self):
        schemas, capabilities = input_schemas(), m.capabilities()
        # The 30 v0.7 tools lead, unchanged; the area tools with operations and the catalog follow.
        areas = area_tools()
        self.assertEqual(list(schemas), ["mathforge."+name for name in _FIELDS] + areas + ["mathforge.catalog"])
        self.assertEqual(list(schemas), [tool["name"] for tool in capabilities["tools"]])
        published = schemas["mathforge.compute"]["properties"]["operation"]["enum"]
        self.assertEqual(published, sorted(operation_arguments()))
        ops = {item["name"]: item for item in capabilities["operations"]}
        for op, keys in operation_arguments().items():
            self.assertEqual(ops[op]["mcp_tool"], "mathforge.compute")
            for key in keys:
                self.assertIn(key, ops[op]["scope"])
        for name, (required, allowed) in _FIELDS.items():
            self.assertEqual(set(schemas["mathforge."+name]["required"]), required)
            self.assertEqual(set(schemas["mathforge."+name]["properties"]), allowed)

    def test_parser_preserves_axes_domains_and_exact_literals(self):
        response = call_tool("mathforge.parse_function", {"text": "log(x)+0.1*y", "variables": ["y", "x"],
            "symbol_ids": {"x": X_ID, "y": Y_ID}})
        f = self.result(response)
        x, y = m.Symbol("x", uid=X_ID), m.Symbol("y", uid=Y_ID)
        self.assertEqual(f.variables, (y, x))
        self.assertEqual(response.payload["symbols"], [{"name": "y", "uid": Y_ID}, {"name": "x", "uid": X_ID}])
        self.assertEqual(f.contains({x: -1, y: 0}), m.Truth.FALSE)
        self.assertEqual(f.contains({x: 1, y: 0}), m.Truth.TRUE)
        self.assertEqual(response.payload["result"]["version"], 5)
        self.assertEqual(f, m.parse_function("log(x)+0.1*y", symbols={"x": x, "y": y}, variables=(y, x)))

    def test_compute_sdk_parity_and_canonical_chaining(self):
        cases = (("integer_gcd", {"a": 84, "b": 30}),
                 ("binomial", {"n": 10, "k": 3}),
                 ("bernoulli", {"p": m.Rational(1, 3)}),
                 ("variance", {"data": (1, 2, 4), "ddof": 1}),
                 ("lu", {"A": m.Matrix(((0, 1), (2, 3)))}),
                 ("least_squares", {"A": m.Matrix(((1, 1), (2, 2))), "b": (1, 3)}))
        for op, args in cases:
            with self.subTest(op=op):
                response = self.compute(op, **args)
                result = self.result(response)
                self.assertEqual(m.dumps(result), m.dumps(run(op, args)))
                self.assertEqual(response.payload["result"]["version"], 5)
                self.assertTrue(m.verify(result).verified)
                checked = self.result(call_tool("mathforge.verify", {"result": response.payload["result"]}))
                self.assertTrue(checked.verified)
        distribution = self.result(self.compute("bernoulli", p=m.Rational(1, 3))).value
        expected = self.result(self.compute("expectation", distribution=distribution))
        self.assertEqual(expected.value, m.Rational(1, 3))

    def test_function_derivative_and_deferred_integral(self):
        x = m.Symbol("x", uid=X_ID)
        parsed = call_tool("mathforge.parse_function", {"text": "sin(x)/x", "variables": ["x"],
            "symbol_ids": {"x": X_ID}})
        f = self.result(parsed)
        derivative = self.result(self.compute("function_differentiate", expression=f, variable=x, n=1))
        self.assertTrue(m.verify(derivative).verified)
        outside = m.parse_function("exp(x*x)", symbols={"x": x}, variables=(x,))
        deferred = self.result(self.compute("function_antiderivative", expression=outside, variable=x))
        self.assertIs(deferred.execution_status, m.ExecutionStatus.COMPLETED)
        self.assertIs(deferred.outcome, m.Outcome.NO_CONCLUSION)
        self.assertIsNone(deferred.value)
        self.assertIs(deferred.exactness, m.Exactness.NOT_APPLICABLE)
        self.assertIsInstance(deferred.deferred, m.UnevaluatedIntegral)
        self.assertFalse(m.verify(deferred).verified)

    def test_only_canonical_data_operands_and_declared_operations(self):
        valid = {"operation": "integer_gcd", "arguments": {"a": m.to_data(6), "b": m.to_data(4)}}
        mutations = []
        for operation in ("os.system", "__import__", "solve", "capabilities", True):
            request = deepcopy(valid)
            request["operation"] = operation
            mutations.append(request)
        for operand in (6, "6", True, 1.0, None, [], {}, m.to_data(run("integer_gcd", {"a": 6, "b": 4})),
                        m.to_data((run("integer_gcd", {"a": 6, "b": 4}),))):
            request = deepcopy(valid)
            request["arguments"]["a"] = operand
            mutations.append(request)
        request = deepcopy(valid)
        request["arguments"]["a"]["version"] = 5
        mutations.append(request)
        request = deepcopy(valid)
        request["arguments"]["extra"] = m.to_data(0)
        mutations.append(request)
        for request in mutations:
            with self.subTest(request=request):
                response = call_tool("mathforge.compute", request)
                self.assertTrue(response.is_error)
                self.assertIsNone(response.payload["result"])

    def test_cross_operand_identity_conflicts_and_parser_injection(self):
        x, conflict = m.Symbol("x", uid=X_ID), m.Symbol("other", uid=X_ID)
        f = m.parse_function("x*x", symbols={"x": x})
        response = self.compute("function_differentiate", expression=f, variable=conflict, n=1)
        self.assertTrue(response.is_error)
        self.assertIsNone(response.payload["result"])
        for text in ("__import__('os')", "x.__class__", "[x for x in ()]", "lambda x: x", "sin(x, x)", "z+1"):
            response = call_tool("mathforge.parse_function", {"text": text, "variables": ["x"]})
            self.assertTrue(response.is_error)
            self.assertIsNone(response.payload["result"])
        for variables in ("x", ["x", "x"], ["sin"], ["x"]*65):
            self.assertTrue(call_tool("mathforge.parse_function", {"text": "1", "variables": variables}).is_error)

    def test_limits_large_response_and_forged_v5_result(self):
        response = call_tool("mathforge.compute", {"operation": "factorial", "arguments": {"n": m.to_data(100)},
            "limits": {"max_work": 1}})
        self.assertTrue(response.is_error)
        self.assertEqual(response.payload["error"]["code"], "resource_limit_exceeded")
        with patch("mathforge.mcp.boundary.MAX_RESPONSE_BYTES", 100):
            response = self.compute("integer_gcd", a=84, b=30)
        self.assertTrue(response.is_error)
        self.assertIsNone(response.payload["result"])
        self.assertEqual(response.payload["error"]["code"], "response_too_large")
        original = self.result(self.compute("integer_gcd", a=84, b=30))
        forged = replace(original, value=999)
        report = self.result(call_tool("mathforge.verify", {"result": m.to_data(forged)}))
        self.assertFalse(report.verified)


if __name__ == "__main__":
    unittest.main()
