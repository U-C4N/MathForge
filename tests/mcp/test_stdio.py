"""Exercise the actual official SDK against a spawned MathForge stdio server."""

import asyncio
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import mathforge as mf
from mathforge import registry
from mathforge.mcp.boundary import _FIELDS
from mathforge.mcp.schemas import capabilities_schema, catalog_schema, input_schemas, output_schema


try:
    MCP_AVAILABLE = metadata.version("mcp").split(".")[0] == "2"
except metadata.PackageNotFoundError:
    MCP_AVAILABLE = False


@unittest.skipUnless(MCP_AVAILABLE, "Optional MCP SDK v2 is not installed")
class StdioTests(unittest.TestCase):
    def test_real_stdio_round_trip(self):
        asyncio.run(self._round_trip())

    async def _round_trip(self):
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters

        env = dict(os.environ, PYTHONPATH=str(Path(mf.__file__).resolve().parent.parent))
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "mathforge.mcp"], env=env)
        async with Client(parameters, read_timeout_seconds=30) as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            # The 30 v0.7 tools lead, unchanged; the area tools with operations and the catalog follow.
            areas = sorted(tool for tool in registry.MCP_TOOLS if registry.tool_operations(tool))
            self.assertEqual(list(tools), ["mathforge." + name for name in _FIELDS] + areas + ["mathforge.catalog"])
            self.assertTrue({"mathforge.limit", "mathforge.analyze_function"} <= set(tools))
            self.assertTrue({"mathforge.parse_function", "mathforge.compute"} <= set(tools))
            self.assertTrue(all(tool.output_schema is not None for tool in listed.tools))
            schemas = input_schemas()
            for name, tool in tools.items():
                self.assertEqual(tool.input_schema, schemas[name])
                self.assertEqual(tool.output_schema, {"mathforge.capabilities": capabilities_schema(),
                                                      "mathforge.catalog": catalog_schema()}.get(name, output_schema()))
            capabilities = await client.call_tool("mathforge.capabilities", {})
            self.assertFalse(capabilities.is_error)
            self.assertEqual(capabilities.structured_content["transports"], ["stdio"])
            invalid_capabilities = await client.call_tool("mathforge.capabilities", {"unexpected": True})
            self.assertTrue(invalid_capabilities.is_error)
            uid = "72f494cc-f9cd-4f81-abfa-748d75509f6e"
            solved = await client.call_tool("mathforge.solve", {"problem": "x^2-2=0", "symbol_ids": {"x": uid}})
            self.assertFalse(solved.is_error)
            payload = solved.structured_content
            self.assertEqual(json.loads(solved.content[0].text), payload)
            x = mf.Symbol("x", uid=uid)
            self.assertEqual(payload["result"], mf.to_data(mf.solve(mf.Eq(x**2-2, 0), for_=x)))
            checked = await client.call_tool("mathforge.verify", {"result": payload["result"]})
            self.assertFalse(checked.is_error)
            self.assertTrue(mf.from_data(checked.structured_content["result"]).verified)
            cases = {
                "solve_system": {"equations": ["x+y=3", "x-y=1"], "variables": ["x", "y"]},
                "real_roots": {"expression": "x^3-x"},
                "differentiate": {"expression": "x^3", "order": 2},
                "expand_multivariate": {"expression": "(x+y)^3", "variables": ["x", "y"]},
                "partial_diff": {"expression": "x^3*y", "variables": ["x", "y"], "variable": "x", "order": 2},
                "gradient": {"expression": "x^2*y", "variables": ["y", "x"]},
                "jacobian": {"expressions": ["x*y", "x^2+y^2"], "variables": ["y", "x"]},
                "hessian": {"expression": "x^2*y", "variables": ["x", "y"]},
                "integrate": {"expression": "x^2", "lower": "0", "upper": "1"},
                "limit": {"expression": "(x^2-1)/(x-1)", "point": "1"},
                "analyze_function": {"expression": "x^2"},
                "det": {"rows": [["1", "2"], ["3", "4"]]},
                "rref": {"rows": [], "ncols": 3},
                "factor": {"expression": "x^4-1"},
                "resultant": {"f": "x^2-2", "g": "x-1"},
                "discriminant": {"expression": "x^2-2"},
                "algebraic": {"expression": "sqrt(2)+sqrt(3)"},
                "algebraic_div": {"left": "1", "right": "sqrt(2)"},
                "algebraic_pow": {"base": "sqrt(2)", "n": -2},
                "minimal_polynomial": {"expression": "sqrt(2)"},
                "evaluate_algebraic": {"expression": "x^2", "variables": ["x"], "bindings": {"x": "sqrt(2)"}},
                "apart": {"expression": "(2*x+3)/(x+1)^2"},
                "extrema_values": {"expression": "x^2"},
                "charpoly": {"rows": [["1", "2"], ["3", "4"]]},
                "real_eigenvalues": {"rows": [["0", "2"], ["1", "0"]]},
            }
            for tool, arguments in cases.items():
                response = await client.call_tool("mathforge." + tool, arguments)
                self.assertFalse(response.is_error, response)
                self.assertTrue(mf.verify(mf.from_data(response.structured_content["result"])).verified)
                if tool in ("limit", "analyze_function"):
                    self.assertEqual(response.structured_content["result"]["version"], 3)
                    self.assertTrue(response.structured_content["text"])
                    self.assertTrue(response.structured_content["latex"])
                    checked_new = await client.call_tool("mathforge.verify", {"result": response.structured_content["result"]})
                    self.assertFalse(checked_new.is_error)
                    self.assertTrue(mf.from_data(checked_new.structured_content["result"]).verified)
            parsed = await client.call_tool("mathforge.parse_function", {
                "text": "log(x)+sin(x)", "variables": ["x"], "symbol_ids": {"x": uid}})
            self.assertFalse(parsed.is_error, parsed)
            function = mf.from_data(parsed.structured_content["result"])
            self.assertEqual(function.variables, (x,))
            self.assertIs(function.contains({x: -1}), mf.Truth.FALSE)
            self.assertEqual(parsed.structured_content["result"]["version"], 5)
            computed = await client.call_tool("mathforge.compute", {
                "operation": "function_differentiate", "arguments": {
                    "expression": mf.to_data(function), "variable": mf.to_data(x), "n": mf.to_data(1)}})
            self.assertFalse(computed.is_error, computed)
            self.assertEqual(json.loads(computed.content[0].text), computed.structured_content)
            computed_result = mf.from_data(computed.structured_content["result"])
            self.assertTrue(mf.verify(computed_result).verified)
            checked_v5 = await client.call_tool("mathforge.verify", {"result": computed.structured_content["result"]})
            self.assertFalse(checked_v5.is_error)
            self.assertTrue(mf.from_data(checked_v5.structured_content["result"]).verified)
            probability = await client.call_tool("mathforge.compute", {
                "operation": "bernoulli", "arguments": {"p": mf.to_data(mf.Rational(1, 3))}})
            self.assertFalse(probability.is_error, probability)
            distribution = mf.from_data(probability.structured_content["result"]).value
            expectation = await client.call_tool("mathforge.compute", {
                "operation": "expectation", "arguments": {"distribution": mf.to_data(distribution)}})
            self.assertFalse(expectation.is_error, expectation)
            self.assertEqual(mf.from_data(expectation.structured_content["result"]).value, mf.Rational(1, 3))
            unresolved = await client.call_tool("mathforge.compute", {
                "operation": "function_antiderivative", "arguments": {
                    "expression": mf.to_data(mf.parse_function("exp(x*x)", symbols={"x": x})),
                    "variable": mf.to_data(x)}})
            self.assertFalse(unresolved.is_error, unresolved)
            deferred = mf.from_data(unresolved.structured_content["result"])
            self.assertIs(deferred.outcome, mf.Outcome.NO_CONCLUSION)
            self.assertIsNone(deferred.value)
            self.assertIsInstance(deferred.deferred, mf.UnevaluatedIntegral)
            checked_deferred = await client.call_tool("mathforge.verify", {"result": unresolved.structured_content["result"]})
            self.assertFalse(checked_deferred.is_error)
            deferred_report = mf.from_data(checked_deferred.structured_content["result"])
            self.assertFalse(deferred_report.verified)
            self.assertFalse(any(check.status is mf.CheckStatus.FAILED for check in deferred_report.checks))
            bad_operation = await client.call_tool("mathforge.compute", {"operation": "__import__", "arguments": {}})
            self.assertTrue(bad_operation.is_error)
            # An area tool runs the v6 pilot from friendly JSON; the client checks each payload's output schema.
            fraction = await client.call_tool("mathforge.ntheory", {
                "operation": "continued_fraction", "arguments": {"value": "415/93"}})
            self.assertFalse(fraction.is_error, fraction)
            self.assertEqual(json.loads(fraction.content[0].text), fraction.structured_content)
            self.assertEqual(fraction.structured_content["result"]["version"], 6)
            self.assertIn("[4; 2, 6, 7]", fraction.structured_content["text"])
            self.assertEqual(mf.from_data(fraction.structured_content["result"]).value.terms, (4, 2, 6, 7))
            checked_v6 = await client.call_tool("mathforge.verify", {"result": fraction.structured_content["result"]})
            self.assertFalse(checked_v6.is_error)
            self.assertTrue(mf.from_data(checked_v6.structured_content["result"]).verified)
            typed = await client.call_tool("mathforge.analysis", {
                "operation": "function_differentiate", "symbol_ids": {"x": uid},
                "arguments": {"expression": {"text": "log(x)+sin(x)", "variables": ["x"]}, "variable": "x"}})
            self.assertFalse(typed.is_error, typed)
            self.assertEqual(typed.structured_content["result"], computed.structured_content["result"])
            inexact = await client.call_tool("mathforge.ntheory", {
                "operation": "integer_gcd", "arguments": {"a": 0.5, "b": 2}})
            self.assertTrue(inexact.is_error)
            self.assertEqual(inexact.structured_content["error"]["field"], "arguments.a")
            catalog = await client.call_tool("mathforge.catalog", {"query": "continued"})
            self.assertFalse(catalog.is_error, catalog)
            entry, = (entry for entry in catalog.structured_content["operations"] if entry["name"] == "continued_fraction")
            self.assertEqual(entry["tool"], "mathforge.ntheory")
            absent_limit = await client.call_tool("mathforge.limit", {"expression": "1/x", "point": "0"})
            self.assertFalse(absent_limit.is_error)
            absent_result = mf.from_data(absent_limit.structured_content["result"])
            self.assertEqual(absent_result.outcome, mf.Outcome.VALUE)
            self.assertEqual(absent_result.value.kind, "does_not_exist")
            directed = await client.call_tool("mathforge.limit", {"expression": "1/x", "point": "0", "side": "left"})
            self.assertFalse(directed.is_error)
            self.assertEqual(mf.from_data(directed.structured_content["result"]).value.kind, "negative_infinity")
            infinite = await client.call_tool("mathforge.limit", {"expression": "x", "point": "+inf"})
            self.assertFalse(infinite.is_error)
            self.assertEqual(mf.from_data(infinite.structured_content["result"]).value.kind, "positive_infinity")
            invalid_direction = await client.call_tool("mathforge.limit", {"expression": "x", "point": "+inf", "side": "left"})
            self.assertTrue(invalid_direction.is_error)
            self.assertEqual(invalid_direction.structured_content["error"]["code"], "invalid_input")
            invalid_point = await client.call_tool("mathforge.limit", {"expression": "x", "point": 0.1})
            self.assertTrue(invalid_point.is_error)
            empty = await client.call_tool("mathforge.solve", {"problem": "(x^2-1)/(x-1)=2"})
            self.assertFalse(empty.is_error)
            self.assertEqual(mf.from_data(empty.structured_content["result"]).outcome, mf.Outcome.NO_SOLUTION)
            bad = await client.call_tool("mathforge.solve", {"problem": "x + = 0"})
            self.assertTrue(bad.is_error)
            self.assertEqual(bad.structured_content["error"]["stage"], "parsing")
            invalid = await client.call_tool("mathforge.det", {"rows": [[0.1]]})
            self.assertTrue(invalid.is_error)
            # Decode a returned root and pass its canonical scalar envelope to
            # subsequent calls; displayed RootOf text is never executable input.
            roots = await client.call_tool("mathforge.real_roots", {"expression": "x^3-2"})
            root = mf.from_data(roots.structured_content["result"]).value[0].root
            root_envelope = mf.to_data(root)
            cubed = await client.call_tool("mathforge.algebraic_pow", {"base": root_envelope, "n": 3})
            self.assertFalse(cubed.is_error, cubed)
            self.assertEqual(mf.from_data(cubed.structured_content["result"]).value, mf.Rational(2))
            exact_limit = await client.call_tool("mathforge.limit", {"expression": "x^3", "point": root_envelope})
            self.assertFalse(exact_limit.is_error, exact_limit)
            self.assertEqual(mf.from_data(exact_limit.structured_content["result"]).value.value, mf.Rational(2))
            exact_integral = await client.call_tool("mathforge.integrate", {"expression": "x", "lower": "0", "upper": "sqrt(2)"})
            self.assertFalse(exact_integral.is_error, exact_integral)
            self.assertEqual(mf.from_data(exact_integral.structured_content["result"]).value, mf.Rational(1))
            invalid_operand = await client.call_tool("mathforge.algebraic", {"expression": roots.structured_content["result"]})
            self.assertTrue(invalid_operand.is_error)
            exhausted = await client.call_tool("mathforge.factor", {"expression": "x^4-1", "limits": {"max_work": 1}})
            self.assertTrue(exhausted.is_error)
            self.assertEqual(exhausted.structured_content["error"]["code"], "resource_limit_exceeded")
            # A rejected call must not poison the long-lived subprocess.
            again = await client.call_tool("mathforge.det", {"rows": []})
            self.assertFalse(again.is_error)

    def test_no_http_or_other_command_line_modes(self):
        env = dict(os.environ, PYTHONPATH=str(Path(mf.__file__).resolve().parent.parent))
        process = subprocess.run([sys.executable, "-m", "mathforge.mcp", "--http"],
                                 env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, "")
        self.assertIn("stdio only", process.stderr)


if __name__ == "__main__":
    unittest.main()
