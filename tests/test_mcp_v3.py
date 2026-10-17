"""Dependency-free tests of the MCP boundary, without importing the MCP SDK."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import registry
from mathforge.capabilities import capabilities
from mathforge.mcp.boundary import call_tool
from mathforge.mcp.schemas import input_schemas


UID = "72f494cc-f9cd-4f81-abfa-748d75509f6e"


class BoundaryTests(unittest.TestCase):
    def invoke(self, name, **arguments):
        return call_tool("mathforge." + name, arguments)

    def result(self, response):
        self.assertFalse(response.is_error, response.payload)
        return mf.from_data(response.payload["result"])

    def test_curated_capabilities_are_fresh_and_match_schemas(self):
        expected = {"capabilities", "solve", "solve_system", "real_roots", "differentiate", "integrate",
                    "limit", "analyze_function", "det", "rref", "verify",
                    "expand_multivariate", "partial_diff", "gradient", "jacobian", "hessian",
                    "factor", "resultant", "discriminant", "algebraic", "algebraic_div", "algebraic_pow",
                    "minimal_polynomial", "evaluate_algebraic", "apart", "extrema_values", "charpoly", "real_eigenvalues",
                    "parse_function", "compute"}
        payload = self.invoke("capabilities").payload
        names = [item["name"] for item in payload["tools"]]
        # The 30 v0.7 tools lead, unchanged; the area tools with operations and the catalog follow.
        areas = sorted(tool for tool in registry.MCP_TOOLS if registry.tool_operations(tool))
        self.assertEqual({name.split(".")[1] for name in names[:30]}, expected)
        self.assertEqual(names[30:], areas + ["mathforge.catalog"])
        self.assertEqual(list(input_schemas()), names)
        payload["tools"].clear()
        self.assertEqual(len(capabilities()["tools"]), 30 + len(areas) + 1)
        self.assertEqual(capabilities()["transports"], ["stdio"])
        operations = {item["name"]: item for item in capabilities()["operations"]}
        self.assertIsNone(operations["poly_gcd"]["mcp_tool"])
        self.assertEqual(operations["solve"]["mcp_tool"], "mathforge.solve")

    def test_core_import_never_imports_optional_sdk(self):
        code = "import mathforge, mathforge.mcp, mathforge.mcp.boundary; import sys; assert 'mcp' not in sys.modules"
        env = dict(os.environ, PYTHONPATH=str(Path(mf.__file__).resolve().parent.parent))
        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_solve_exact_sdk_parity_and_uuid_reuse(self):
        response = self.invoke("solve", problem="x^2-2=0", symbol_ids={"x": UID})
        result = self.result(response)
        x = mf.Symbol("x", uid=UID)
        self.assertEqual(mf.dumps(result), mf.dumps(mf.solve(mf.Eq(x**2-2, 0), for_=x)))
        self.assertEqual(response.payload["symbols"], [{"name": "x", "uid": UID}])
        self.assertIn("sqrt", response.payload["text"])
        self.assertIn("sqrt", response.payload["latex"])
        self.assertTrue(mf.verify(result).verified)
        repeated = self.invoke("solve", problem="x^2-2=0", symbol_ids={"x": UID})
        self.assertEqual(response.payload, repeated.payload)

    def test_requests_do_not_intern_names_globally(self):
        first = self.invoke("real_roots", expression="x^2-1")
        second = self.invoke("real_roots", expression="x^2-1")
        self.assertNotEqual(first.payload["symbols"], second.payload["symbols"])

    def test_rational_holes_and_empty_set_are_success(self):
        result = self.result(self.invoke("solve", problem="(x^2-1)/(x-1)=2"))
        self.assertEqual(result.outcome, mf.Outcome.NO_SOLUTION)
        inequality = self.result(self.invoke("solve", problem="(x^2-1)/(x-1)>=0"))
        self.assertFalse(inequality.value.contains(1))

    def test_all_calculation_tools(self):
        roots = self.result(self.invoke("real_roots", expression="(x-1)^2*(x+2)"))
        self.assertEqual([root.multiplicity for root in roots.value], [1, 2])
        derivative = self.result(self.invoke("differentiate", expression="x^3", order=2))
        self.assertEqual(str(derivative.value), "6*x")
        integral = self.result(self.invoke("integrate", expression="x^2", lower="0", upper="1"))
        self.assertEqual(integral.value, mf.Rational(1, 3))
        system = self.result(self.invoke("solve_system", equations=["x+2*y=3", "2*x+4*y=6"], variables=["x", "y"]))
        self.assertEqual(len(system.value.basis), 1)
        determinant = self.result(self.invoke("det", rows=[["1", "2"], ["3", "4"]]))
        self.assertEqual(determinant.value, mf.Rational(-2))
        reduced = self.result(self.invoke("rref", rows=[["1/2", "1"], ["1", "2"]]))
        self.assertEqual(reduced.value.pivot_columns, (0,))
        empty = self.result(self.invoke("rref", rows=[], ncols=3))
        self.assertEqual(empty.value.matrix.ncols, 3)
        self.assertEqual(self.result(self.invoke("det", rows=[])).value, mf.Rational(1))

    def test_bad_inputs_never_become_no_solution(self):
        cases = [
            ("solve", {"problem": "x=0", "unexpected": 1}),
            ("solve", {"problem": "y=0"}),
            ("solve", {"problem": {"data": "not text"}}),
            ("solve", {"problem": "x=0", "variable": True}),
            ("solve", {"problem": "x=0", "variable": "_x"}),
            ("solve", {"problem": "x=0", "variable": "sqrt"}),
            ("solve", {"problem": "x=0", "symbol_ids": {"x": "bad UUID"}}),
            ("solve", {"problem": "x=0", "symbol_ids": {"y": UID}}),
            ("solve_system", {"equations": ["x>0"], "variables": ["x"]}),
            ("solve_system", {"equations": ["x=0"], "variables": ["x", "x"]}),
            ("solve_system", {"equations": ["x+y=0"], "variables": ["x", "y"], "symbol_ids": {"x": UID, "y": UID}}),
            ("differentiate", {"expression": "x", "order": True}),
            ("differentiate", {"expression": "x", "order": -1}),
            ("integrate", {"expression": "x", "lower": "0.1.2", "upper": "1"}),
            ("integrate", {"expression": "x", "lower": "1/0", "upper": "1"}),
            ("det", {"rows": [[0.1]]}),
            ("det", {"rows": [[True]]}),
            ("det", {"rows": [["1"], ["2", "3"]]}),
            ("rref", {"rows": [], "ncols": True}),
            ("solve", {"problem": "x=0", "limits": {"max_work": 2_000_001}}),
            ("solve", {"problem": "x=0", "limits": {"max_work": True}}),
            ("solve", {"problem": "x=0", "limits": {"unknown": 1}}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                response = self.invoke(name, **arguments)
                self.assertTrue(response.is_error)
                self.assertIsNotNone(response.payload["error"])
                if response.payload["result"] is not None:
                    self.assertIsNone(mf.from_data(response.payload["result"]).solution_set)

    def test_parse_source_locations(self):
        response = self.invoke("solve", problem="x + = 2")
        self.assertTrue(response.is_error)
        self.assertEqual(response.payload["error"]["stage"], "parsing")
        self.assertEqual(response.payload["error"]["field"], "problem")
        self.assertIsInstance(response.payload["error"]["start"], int)

    def test_unsupported_and_exhausted_results_remain_explicit(self):
        response = self.invoke("solve", problem="x^3-sqrt(2)=0")
        self.assertTrue(response.is_error)
        result = mf.from_data(response.payload["result"])
        self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)
        exhausted = self.invoke("real_roots", expression="x^5-x-1", limits={"max_work": 1})
        self.assertTrue(exhausted.is_error)
        self.assertEqual(exhausted.payload["error"]["code"], "resource_limit_exceeded")

    def test_verify_canonical_envelope_and_failed_claim(self):
        good = self.invoke("solve", problem="x^2-2=0", symbol_ids={"x": UID})
        report = self.result(self.invoke("verify", result=good.payload["result"]))
        self.assertIsInstance(report, mf.VerificationReport)
        self.assertTrue(report.verified)
        result = mf.from_data(good.payload["result"])
        forged = replace(result, value=mf.FiniteSet((mf.Rational(99),)))
        bad = self.invoke("verify", result=mf.to_data(forged))
        self.assertFalse(bad.is_error)
        self.assertFalse(mf.from_data(bad.payload["result"]).verified)
        for data in (mf.to_data(mf.Rational(1)), mf.to_data(mf.Workspace()), {"format": "unknown"}):
            self.assertTrue(self.invoke("verify", result=data).is_error)

    def test_invalid_transport_and_response_bounds(self):
        self.assertTrue(call_tool("os.system", {}).is_error)
        self.assertTrue(call_tool("mathforge.solve", None).is_error)
        with patch("mathforge.mcp.boundary.MAX_REQUEST_BYTES", 8):
            response = self.invoke("solve", problem="x=1")
            self.assertEqual(response.payload["error"]["code"], "resource_limit_exceeded")
        with patch("mathforge.mcp.boundary.MAX_RESPONSE_BYTES", 100):
            response = self.invoke("solve", problem="x=1")
            self.assertEqual(response.payload["error"]["code"], "response_too_large")
            self.assertIsNone(response.payload["result"])

    def test_render_failure_preserves_mathematics(self):
        with patch("mathforge.rendering.to_latex", side_effect=InvalidRender("no room")):
            response = self.invoke("solve", problem="x=1")
        self.assertTrue(response.is_error)
        self.assertEqual(response.payload["error"]["stage"], "presentation")
        self.assertTrue(mf.verify(mf.from_data(response.payload["result"])).verified)


class InvalidRender(mf.InvalidInput):
    pass


if __name__ == "__main__":
    unittest.main()
