"""Bounded exact-algebra tools, canonical scalar chaining and legacy parity."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge.capabilities import area_tools
from mathforge.exact_dispatch import NEW_OPERATIONS
from mathforge.mcp.boundary import MAX_SCALAR_CHARACTERS, MAX_VARIABLES, _FIELDS, call_tool
from mathforge.mcp.schemas import input_schemas


UID = "00000000-0000-4000-8000-000000000620"


class ExactAlgebraBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.Symbol("x", uid=UID)
        self.common = {"variable": "x", "symbol_ids": {"x": UID}}

    def invoke(self, operation, **arguments):
        return call_tool("mathforge." + operation, arguments)

    def result(self, response):
        self.assertFalse(response.is_error, response.payload)
        self.assertIsNone(response.payload["error"])
        self.assertEqual(response.payload["api_version"], 1)
        self.assertTrue(response.payload["text"])
        self.assertTrue(response.payload["latex"])
        result = mf.from_data(response.payload["result"])
        self.assertIs(result.execution_status, mf.ExecutionStatus.COMPLETED)
        self.assertTrue(mf.verify(result).verified)
        return result

    def test_all_twelve_new_operations_match_sdk_and_verify(self):
        x = self.x
        P = lambda coefficients: mf.Polynomial(coefficients, x)
        source = mf.rational_function(2*x+3, (x+1)**2, variable=x)
        cases = (
            ("factor", {"expression": "x^4-1", **self.common}, lambda: mf.operations.factor(P((-1, 0, 0, 0, 1)))),
            ("resultant", {"f": "x^2-2", "g": "x-1", **self.common},
             lambda: mf.operations.resultant(P((-2, 0, 1)), P((-1, 1)))),
            ("discriminant", {"expression": "x^2-2", **self.common}, lambda: mf.operations.discriminant(P((-2, 0, 1)))),
            ("algebraic", {"expression": "sqrt(2)+sqrt(3)"}, lambda: mf.operations.algebraic(mf.sqrt(2)+mf.sqrt(3))),
            ("algebraic_div", {"left": "1", "right": "sqrt(2)"}, lambda: mf.operations.algebraic_div(1, mf.sqrt(2))),
            ("algebraic_pow", {"base": "sqrt(2)", "n": -2}, lambda: mf.operations.algebraic_pow(mf.sqrt(2), -2)),
            ("minimal_polynomial", {"expression": "sqrt(2)", **self.common}, lambda: mf.operations.minimal_polynomial(mf.sqrt(2), x)),
            ("evaluate_algebraic", {"expression": "x^2", "variables": ["x"], "bindings": {"x": "sqrt(2)"},
                                    "symbol_ids": {"x": UID}}, lambda: mf.operations.evaluate_algebraic(x**2, {x: mf.sqrt(2)})),
            ("apart", {"expression": "(2*x+3)/(x+1)^2", **self.common}, lambda: mf.operations.apart(source)),
            ("extrema_values", {"expression": "x^2", **self.common}, lambda: mf.operations.extrema_values(x**2, x)),
            ("charpoly", {"rows": [["1", "2"], ["3", "4"]], **self.common},
             lambda: mf.operations.charpoly(mf.Matrix(((1, 2), (3, 4))), x)),
            ("real_eigenvalues", {"rows": [["0", "2"], ["1", "0"]]},
             lambda: mf.operations.real_eigenvalues(mf.Matrix(((0, 2), (1, 0))))),
        )
        self.assertEqual({case[0] for case in cases}, set(NEW_OPERATIONS))
        for operation, arguments, expected in cases:
            with self.subTest(operation=operation):
                response = self.invoke(operation, **arguments)
                result = self.result(response)
                self.assertEqual(mf.dumps(result), mf.dumps(expected()))
                self.assertEqual(response.payload["result"]["version"], 4)
                checked = self.invoke("verify", result=response.payload["result"])
                self.assertFalse(checked.is_error, checked.payload)
                self.assertTrue(mf.from_data(checked.payload["result"]).verified)

    def test_real_root_output_can_be_reused_as_canonical_scalar(self):
        roots = self.result(self.invoke("real_roots", expression="x^3-2", **self.common))
        root = roots.value[0].root
        envelope = mf.to_data(root)
        self.assertEqual(envelope["version"], 2)
        for operation, arguments, expected in (
            ("algebraic_div", {"left": envelope, "right": envelope}, mf.Rational(1)),
            ("algebraic_pow", {"base": envelope, "n": 3}, mf.Rational(2)),
            ("evaluate_algebraic", {"expression": "x^3", "variables": ["x"], "bindings": {"x": envelope}}, mf.Rational(2)),
            ("limit", {"expression": "x^3", "point": envelope}, None),
        ):
            with self.subTest(operation=operation):
                result = self.result(self.invoke(operation, **arguments))
                self.assertEqual(result.value.value if operation == "limit" else result.value,
                                 mf.Rational(2) if expected is None else expected)
        minimum = self.result(self.invoke("minimal_polynomial", expression=envelope, **self.common))
        self.assertEqual(minimum.value, mf.Polynomial((-2, 0, 0, 1), self.x))

    def test_algebraic_and_decimal_bounds_preserve_exactness_and_legacy_envelopes(self):
        for point in ("sqrt(2)", mf.to_data(mf.sqrt(2))):
            result = self.result(self.invoke("limit", expression="x^2", point=point, **self.common))
            self.assertEqual(result.value.value, mf.Rational(2))
        integral = self.result(self.invoke("integrate", expression="x", lower="0", upper="sqrt(2)", **self.common))
        self.assertEqual(integral.value, mf.Rational(1))
        integral = self.result(self.invoke("integrate", expression="x", lower="0.1", upper="1", **self.common))
        self.assertEqual(integral.value, mf.Rational(99, 200))
        self.assertEqual(mf.to_data(integral)["version"], 2)
        for point in ("0.1", mf.to_data(mf.Rational(1, 10))):
            response = self.invoke("limit", expression="x", point=point, **self.common)
            result = self.result(response)
            self.assertEqual(result.value.value, mf.Rational(1, 10))
            self.assertEqual(response.payload["result"], mf.to_data(mf.operations.limit(self.x, self.x, mf.Rational(1, 10))))
            self.assertEqual(response.payload["result"]["version"], 3)

    def test_scalar_envelopes_cannot_smuggle_results_workspaces_or_open_expressions(self):
        result = mf.operations.real_roots(self.x**2-2, self.x)
        workspace = mf.Workspace()
        workspace.put(result)
        invalid = [mf.to_data(result), mf.to_data(workspace), mf.to_data(self.x), mf.to_data(self.x+1),
                   mf.to_data(mf.Matrix(((1,),))), mf.to_data((mf.Rational(1),)),
                   mf.to_data(mf.Polynomial((1, 1), self.x)), {}, [], 1, True, 1.2, None,
                   "RootOf(t^3-2; real_index=0)", "x", "__import__('os')", "sin(1)", "sqrt(-1)", "1/0"]
        noncanonical = mf.to_data(mf.Rational(1))
        noncanonical["data"].update(numerator="2", denominator="2")
        invalid.append(noncanonical)
        wrong_version = mf.to_data(mf.Rational(1))
        wrong_version["version"] = 4
        invalid.append(wrong_version)
        for operand in invalid:
            with self.subTest(operand=operand):
                response = self.invoke("algebraic", expression=operand)
                self.assertTrue(response.is_error)
                self.assertEqual(response.payload["error"]["field"], "expression")
                self.assertIsNone(response.payload["result"])

    def test_complete_bindings_and_bounded_signed_exponents_are_required(self):
        for bindings in ({}, {"y": "1"}, {"x": "1", "y": "2"}, [], {"x": self.common}):
            response = self.invoke("evaluate_algebraic", expression="x", variables=["x"], bindings=bindings)
            self.assertTrue(response.is_error)
            self.assertTrue(response.payload["error"]["field"].startswith("bindings"))
        for variables in ([], "x", ["x", "x"], ["v"+str(i) for i in range(MAX_VARIABLES+1)]):
            response = self.invoke("evaluate_algebraic", expression="x", variables=variables, bindings={"x": "1"})
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["field"], "variables")
        for n in (True, "2", 1.5, None, 1025, -1025):
            response = self.invoke("algebraic_pow", base="2", n=n)
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["field"], "n")
        for operation, arguments in (("algebraic_div", {"left": "1", "right": "0"}),
                                      ("algebraic_pow", {"base": "0", "n": -1})):
            response = self.invoke(operation, **arguments)
            self.assertTrue(response.is_error)
            self.assertIsNone(mf.from_data(response.payload["result"]).value)

    def test_domain_holes_and_nonreal_spectra_are_preserved(self):
        response = self.invoke("apart", expression="(x^2-1)/(x-1)", **self.common)
        result = self.result(response)
        self.assertEqual(result.value.to_rational_function(), mf.rational_function(self.x**2-1, self.x-1, variable=self.x))
        self.assertTrue(result.value.excluded)
        for rows in ([], [["0", "-1"], ["1", "0"]]):
            result = self.result(self.invoke("real_eigenvalues", rows=rows))
            self.assertEqual(result.value, ())
            self.assertIs(result.outcome, mf.Outcome.VALUE)

    def test_limits_invalid_domains_and_oversized_outputs_do_not_claim_values(self):
        cases = (("factor", {"expression": "x^4-1"}), ("algebraic", {"expression": "sqrt(2)+sqrt(3)"}),
                 ("apart", {"expression": "1/(x-1)^2"}), ("charpoly", {"rows": [["1", "2"], ["3", "4"]]}))
        for operation, arguments in cases:
            response = self.invoke(operation, **arguments, limits={"max_work": 1})
            self.assertTrue(response.is_error)
            self.assertEqual(response.payload["error"]["code"], "resource_limit_exceeded")
            if response.payload["result"] is not None:
                self.assertIsNone(mf.from_data(response.payload["result"]).value)
        for operation, arguments in (("factor", {"expression": "sqrt(2)*x"}),
                                      ("resultant", {"f": "x", "g": "y"}),
                                      ("charpoly", {"rows": [["1", "2"]]}),
                                      ("real_eigenvalues", {"rows": [["sqrt(2)"]]})):
            self.assertTrue(self.invoke(operation, **arguments).is_error)
        response = self.invoke("algebraic", expression="1"*(MAX_SCALAR_CHARACTERS+1))
        self.assertEqual(response.payload["error"]["code"], "resource_limit_exceeded")
        with patch("mathforge.mcp.boundary.MAX_RESPONSE_BYTES", 100):
            response = self.invoke("factor", expression="x^4-1")
        self.assertTrue(response.is_error)
        self.assertEqual(response.payload["error"]["code"], "response_too_large")
        self.assertIsNone(response.payload["result"])

    def test_forged_v4_values_are_not_certified_by_boundary_verify(self):
        result = self.result(self.invoke("charpoly", rows=[["1", "2"], ["3", "4"]], **self.common))
        forged = replace(result, value=result.value+1)
        response = self.invoke("verify", result=mf.to_data(forged))
        self.assertFalse(response.is_error)
        self.assertFalse(mf.from_data(response.payload["result"]).verified)

    def test_schemas_and_capabilities_match_all_explicit_boundary_contracts(self):
        schemas = input_schemas()
        capabilities = mf.capabilities()
        # The 30 v0.7 tools lead, unchanged; the area tools with operations and the catalog follow.
        areas = area_tools()
        self.assertEqual(list(schemas), ["mathforge."+name for name in _FIELDS] + areas + ["mathforge.catalog"])
        self.assertEqual(list(schemas), [tool["name"] for tool in capabilities["tools"]])
        for name, (required, allowed) in _FIELDS.items():
            schema = schemas["mathforge."+name]
            self.assertEqual(set(schema["required"]), required)
            self.assertEqual(set(schema["properties"]), allowed)
            self.assertFalse(schema["additionalProperties"])
        scalar = schemas["mathforge.algebraic"]["properties"]["expression"]["anyOf"]
        self.assertEqual(scalar[0]["maxLength"], MAX_SCALAR_CHARACTERS)
        self.assertEqual(scalar[1]["properties"]["kind"], {"const": "object"})
        self.assertNotIn("result", scalar[1]["properties"]["data"]["properties"]["type"]["enum"])
        power = schemas["mathforge.algebraic_pow"]["properties"]["n"]
        self.assertEqual((power["minimum"], power["maximum"]), (-1024, 1024))
        for operation in NEW_OPERATIONS:
            self.assertTrue(self.invoke(operation, arbitrary_dispatch="os.system").is_error)


if __name__ == "__main__":
    unittest.main()
