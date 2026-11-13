"""The reason and next_actions of every MCP error object (v0.9, MCP hardening).

Each error object of the boundary keeps its six keys and their values, and adds "reason", a stable snake_case code,
and "next_actions", short imperative suggestions; both derive from the error's code, stage and field, never from its
message. contracts.ErrorInfo and serialized Results do not change.
"""

import dataclasses
import json
import unittest
from unittest.mock import patch

try:
    from jsonschema import Draft202012Validator
except ImportError:  # Development-only (requirements-dev.txt); the isolated distribution harness has none.
    Draft202012Validator = None

import mathforge as mf
from mathforge.contracts import ErrorInfo
from mathforge.mcp import handles
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.mcp.boundary import NEXT_ACTIONS, _AT_LIMIT, _CLOSED_TEXT, _advice, _error, call_tool
from mathforge.mcp.schemas import _error_schema

KEYS = ["stage", "code", "message", "field", "start", "end", "reason", "next_actions"]
CATALOG_ARGUMENTS = 'call mathforge.catalog with query "{0}" to see the arguments of {0} and their kinds'
CORRECT_FIELD = 'correct the input that "field" names and call again'
SCHEMA = ["check the arguments against this tool's input schema",
          "call mathforge.catalog with the operation's name as query to see its arguments and their kinds"]
LIMITS = ["retry with larger limits (each field at most its default maximum) or a smaller input"]
AT_LIMIT = ["retry with a smaller or simpler input; no limits field can raise this limit further"]
DOMAIN = 'the inputs are outside what {0} accepts, as "message" says; change them and call again'
SYMBOL_IDS = ['pass symbol_ids as an object from variable names to UUID strings, copied from the "symbols" of an '
              'earlier response']
X_ID = "00000000-0000-4000-8000-000000000911"


def area(tool, operation, *, limits=None, symbol_ids=None, **arguments):
    request = {"operation": operation, "arguments": arguments}
    if limits is not None:
        request["limits"] = limits
    if symbol_ids is not None:
        request["symbol_ids"] = symbol_ids
    return call_tool("mathforge." + tool, request)


class ReasonTests(unittest.TestCase):
    def setUp(self):
        handles.clear()
        self.addCleanup(handles.clear)

    def error(self, response):
        self.assertTrue(response.is_error)
        error = response.payload["error"]
        self.assertEqual(list(error), KEYS)
        if Draft202012Validator is not None:
            self.assertEqual(list(Draft202012Validator(_error_schema()).iter_errors(error)), [])
        self.assertRegex(error["reason"], r"\A[a-z][a-z0-9_]*\Z")
        self.assertTrue(all(type(action) is str and action for action in error["next_actions"]))
        return error

    def check(self, response, stage, code, reason, next_actions, field=None):
        error = self.error(response)
        self.assertEqual((error["stage"], error["code"], error["reason"], error["next_actions"]),
                         (stage, code, reason, next_actions))
        if field is not None:
            self.assertEqual(error["field"], field)
        return error

    def test_input_errors(self):
        self.check(call_tool("mathforge.nope", {}), "input", "unknown_tool", "unknown_tool",
                   ["call mathforge.catalog to find the tool that runs an operation"])
        unknown_operation = ["call mathforge.catalog with a query to find the operation and the tool that runs it"]
        self.check(area("ntheory", "nope"), "input", "invalid_input", "unknown_operation", unknown_operation,
                   "operation")
        self.check(area("ntheory", "binomial", n=4, k=2), "input", "invalid_input", "unknown_operation",
                   unknown_operation, "operation")  # An operation of another tool
        self.check(call_tool("mathforge.compute", {"operation": "nope", "arguments": {}}), "decoding",
                   "invalid_input", "unknown_operation", unknown_operation, "operation")
        self.check(call_tool("mathforge.solve", {}), "input", "invalid_input", "invalid_input", SCHEMA)
        self.check(call_tool("mathforge.ntheory", {"operation": "integer_gcd"}), "input", "invalid_input",
                   "invalid_input", SCHEMA)
        self.check(area("ntheory", "integer_gcd", a=1.5, b=2), "decoding", "invalid_input", "invalid_input",
                   [CATALOG_ARGUMENTS.format("integer_gcd"), CORRECT_FIELD], "arguments.a")
        self.check(area("ntheory", "integer_gcd", a=1), "decoding", "invalid_input", "invalid_input",
                   [CATALOG_ARGUMENTS.format("integer_gcd"), CORRECT_FIELD], "arguments.b")
        self.check(call_tool("mathforge.compute", {"operation": "integer_gcd", "arguments": {"a": 1, "b": 2}}),
                   "decoding", "invalid_input", "invalid_input",
                   [CATALOG_ARGUMENTS.format("integer_gcd"), CORRECT_FIELD], "arguments.a")
        self.check(call_tool("mathforge.det", {"rows": [["1"], "2"]}), "input", "invalid_input", "invalid_input",
                   SCHEMA, "rows")

    def test_limits(self):
        self.check(call_tool("mathforge.solve", {"problem": "x=1", "limits": {"max_work": 0}}), "input",
                   "limit_out_of_range", "limit_out_of_range",
                   ['give each limits field a positive integer no greater than its default; mathforge.capabilities '
                    'lists the defaults under "limits"'], "limits.max_work")
        self.check(area("ntheory", "integer_gcd", a=1, b=2, limits={"max_depth": 10 ** 9}), "input",
                   "limit_out_of_range", "limit_out_of_range",
                   ['give each limits field a positive integer no greater than its default; mathforge.capabilities '
                    'lists the defaults under "limits"'], "limits.max_depth")
        # Larger limits can help only when the request lowered them below the defaults, which are the maxima.
        self.check(call_tool("mathforge.real_roots", {"expression": "x^5-x-1", "limits": {"max_work": 1}}),
                   "parsing", "resource_limit_exceeded", "resource_limit_exceeded", LIMITS)
        self.check(area("ntheory", "integer_gcd", a="9" * 100, b=2, limits={"max_integer_bits": 64}), "decoding",
                   "resource_limit_exceeded", "resource_limit_exceeded", LIMITS, "arguments.a")
        self.assertEqual(AT_LIMIT, list(_AT_LIMIT))
        for limits, actions in ((None, AT_LIMIT), (ComputationLimits(), AT_LIMIT),
                                (ComputationLimits(max_work=5), LIMITS)):
            with self.subTest(limits=limits):
                self.assertEqual(_error(ResourceLimitError("out"), "computation", limits=limits)["next_actions"],
                                 actions)
        # A fixed cap, which no limits field sets, never advises larger limits, even below the defaults.
        lowered = {"max_work": 1000}
        with patch("mathforge.mcp.boundary.MAX_REQUEST_BYTES", 8):
            self.check(call_tool("mathforge.solve", {"problem": "x=1"}), "input", "resource_limit_exceeded",
                       "resource_limit_exceeded", AT_LIMIT)
        self.check(call_tool("mathforge.algebraic", {"expression": "9" * 1100, "limits": lowered}), "parsing",
                   "resource_limit_exceeded", "resource_limit_exceeded", AT_LIMIT, "expression")
        self.check(area("ntheory", "integer_gcd", a="9" * 2000, b=2, limits=lowered), "decoding",
                   "resource_limit_exceeded", "resource_limit_exceeded", AT_LIMIT, "arguments.a")
        self.check(area("numeric", "digits", value="1" * 1100, places=3, limits=lowered), "decoding",
                   "resource_limit_exceeded", "resource_limit_exceeded", AT_LIMIT, "arguments.value")
        self.check(call_tool("mathforge.solve", {"problem": "(" * 300 + "x" + ")" * 300 + "=1", "limits": lowered}),
                   "parsing", "resource_limit_exceeded", "resource_limit_exceeded", AT_LIMIT, "problem")  # ParseLimits
        self.check(call_tool("mathforge.gradient", {"expression": "1", "variables": [f"v{i}" for i in range(65)],
                                                    "limits": lowered}),
                   "input", "resource_limit_exceeded", "resource_limit_exceeded", AT_LIMIT, "variables")
        gcd = call_tool("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": 12, "b": 18}})
        request = {"operation": "integer_gcd", "arguments": {"a": gcd.payload["handle"], "b": 4}, "limits": lowered}
        with patch("mathforge.mcp.boundary.MAX_REQUEST_BYTES", len(json.dumps(request)) + 1):
            self.check(call_tool("mathforge.ntheory", request), "decoding", "resource_limit_exceeded",
                       "resource_limit_exceeded", AT_LIMIT, "arguments.a")

    def test_encoding_and_presentation(self):
        with patch("mathforge.mcp.boundary.MAX_RESPONSE_BYTES", 100):
            self.check(call_tool("mathforge.solve", {"problem": "x=1"}), "encoding", "response_too_large",
                       "response_too_large", ["retry with a smaller input, or split the computation into smaller "
                                              "calls; the response byte limit is fixed"])
        with patch("mathforge.rendering.to_latex", side_effect=mf.InvalidInput("no room")):
            response = call_tool("mathforge.solve", {"problem": "x=1"})
        error = self.check(response, "presentation", "invalid_input", "presentation_failed",
                           ['use the canonical result in "result"; only its text and LaTeX presentation failed'])
        self.assertEqual(error["message"], "no room")
        self.assertIsNotNone(response.payload["result"])

    def test_handles(self):
        self.check(area("ntheory", "integer_gcd", a="mf:sha256:" + "a" * 64, b=2), "input", "unknown_handle",
                   "unknown_handle", ["repeat the call that returned this handle and pass the handle of its new "
                                      "response instead", "or send this argument in full, in its friendly form when "
                                      "its kind has one (mathforge.catalog names the kind)"], "arguments.a")
        self.check(call_tool("mathforge.compute", {"operation": "integer_gcd", "arguments": {
            "a": "mf:sha256:" + "a" * 64, "b": mf.to_data(2)}}), "input", "unknown_handle", "unknown_handle",
            ["repeat the call that returned this handle and pass the handle of its new response instead",
             "or send this operand in full, as a canonical object envelope in its minimum format version"],
            "arguments.a")

    def test_parsing(self):
        error = self.check(call_tool("mathforge.solve", {"problem": "x^2="}), "parsing", "parse_syntax",
                           "parse_syntax", ['correct the mathematical text of the field that "field" names, at the '
                                            'characters from "start" to "end", and call again'], "problem")
        self.assertEqual((error["start"], error["end"]), (4, 4))
        self.check(call_tool("mathforge.factor", {"expression": "x^2+y"}), "parsing", "unknown_symbol",
                   "unknown_symbol", ["declare every name that the text uses, in variables or as the variable "
                                      "argument, and call again"], "expression")
        self.check(area("algebra", "complex_algebraic", real="1/0", imag=0), "decoding", "division_by_zero",
                   "division_by_zero", [], "arguments.real")
        # A field of closed text, a number, has no name to declare.
        closed = ['write the number as closed mathematical text that uses no names, such as "sqrt(2)/3", or send '
                  'its canonical envelope, and call again']
        self.assertEqual(closed, list(_CLOSED_TEXT))
        self.check(call_tool("mathforge.algebraic", {"expression": "x+1"}), "parsing", "unknown_symbol",
                   "unknown_symbol", closed, "expression")
        self.check(call_tool("mathforge.integrate", {"expression": "x", "lower": "0", "upper": "a"}), "computation",
                   "unknown_symbol", "unknown_symbol", closed, "upper")  # integrate reads its bounds at that stage
        self.check(area("numeric", "digits", value="x+1", places=3), "decoding", "unknown_symbol", "unknown_symbol", closed,
                   "arguments.value")

    def test_symbols(self):
        self.check(call_tool("mathforge.solve", {"problem": "x=1", "symbol_ids": {"y": X_ID}}), "input",
                   "invalid_input", "invalid_symbol_ids", SYMBOL_IDS, "symbol_ids")
        self.check(area("ntheory", "integer_gcd", a=1, b=2, symbol_ids={"x": "not-a-uuid"}), "input",
                   "invalid_input", "invalid_symbol_ids", SYMBOL_IDS, "symbol_ids.x")
        # One identity under two names: the area tool and compute detect it alike.
        x, other = mf.Symbol("x", uid=X_ID), mf.Symbol("other", uid=X_ID)
        f = mf.parse_function("x*x", symbols={"x": x})
        conflict = ['give each symbol identity one name: pass symbol_ids built from the "symbols" of the earlier '
                    'response and use those names']
        compute = self.check(call_tool("mathforge.compute", {"operation": "function_differentiate", "arguments": {
            "expression": mf.to_data(f), "variable": mf.to_data(other), "n": mf.to_data(1)}}),
            "decoding", "invalid_serialization", "symbol_conflict", conflict, "arguments")
        typed = self.check(area("analysis", "function_differentiate", expression=mf.to_data(f),
                                variable=mf.to_data(other)), "decoding", "invalid_serialization", "symbol_conflict",
                           conflict, "arguments")
        self.assertEqual(compute, typed)
        friendly = area("analysis", "function_differentiate", expression={"text": "x*x", "variables": ["x"]},
                        variable=mf.to_data(other), symbol_ids={"x": X_ID})
        self.assertEqual(self.error(friendly), typed)

    def test_documents(self):
        # verify takes no handle, so its advice never suggests one, also when it was sent a handle.
        verify = ['send the "result" of an earlier response exactly as it was returned: mathforge.verify takes the '
                  'whole Result document, not a handle']
        self.check(call_tool("mathforge.verify", {"result": {"format": "unknown"}}), "decoding",
                   "invalid_serialization", "invalid_document", verify, "result")
        det = call_tool("mathforge.det", {"rows": [["1", "2"], ["3", "4"]]}).payload
        self.check(call_tool("mathforge.verify", {"result": det["handle"]}), "decoding", "invalid_serialization",
                   "invalid_document", verify, "result")
        self.check(call_tool("mathforge.verify", {"result": dict(det["result"], version=99)}), "decoding",
                   "invalid_serialization", "invalid_document", verify, "result")
        self.check(call_tool("mathforge.verify", {"result": mf.to_data(mf.Rational(1))}), "decoding",
                   "invalid_input", "invalid_input", SCHEMA, "result")
        # The tools that take handles may suggest one; a curated tool's scalar envelope may not.
        self.check(area("ntheory", "integer_gcd", a={"format": "mathforge", "version": 1, "kind": "object",
                                                     "data": {"type": "integer", "value": "x"}}, b=2),
                   "decoding", "invalid_serialization", "invalid_document",
                   list(NEXT_ACTIONS["invalid_document"]), "arguments.a")
        self.check(call_tool("mathforge.algebraic", {"expression": {"format": "mathforge", "version": 1,
                                                                    "kind": "object", "data": {"type": "rational",
                                                                                               "value": "x"}}}),
                   "parsing", "invalid_serialization", "invalid_document",
                   ["send the canonical envelope exactly as an earlier response returned it"], "expression")

    def test_computation_errors(self):
        unsupported = call_tool("mathforge.solve", {"problem": "x^3-sqrt(2)=0"})
        self.assertIs(mf.from_data(unsupported.payload["result"]).execution_status, mf.ExecutionStatus.UNSUPPORTED)
        error = self.check(unsupported, "computation", "unsupported_operation", "unsupported",
                           ["call mathforge.catalog with a query to find an operation that covers this input"])
        self.assertEqual(error["code"], mf.from_data(unsupported.payload["result"]).error.code)
        self.check(call_tool("mathforge.algebraic_div", {"left": "1", "right": "0"}), "computation",
                   "division_by_zero", "division_by_zero", [])
        # An operation that refused its inputs names no field, so the advice points at the message, not a field.
        domain = [CATALOG_ARGUMENTS.format("integer_gcd"), DOMAIN.format("integer_gcd")]
        self.check(area("ntheory", "integer_gcd", a=mf.to_data(mf.Rational(1, 2)), b=2), "computation",
                   "invalid_input", "invalid_input", domain, None)
        self.check(call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                   "arguments": {"a": mf.to_data(mf.Rational(1, 2)),
                                                                 "b": mf.to_data(2)}}),
                   "computation", "invalid_input", "invalid_input", domain)
        error = self.check(area("ntheory", "mod_inverse", a=2, modulus=4), "computation", "invalid_input",
                           "invalid_input", [CATALOG_ARGUMENTS.format("mod_inverse"), DOMAIN.format("mod_inverse")])
        self.assertIsNone(error["field"])

    def test_catalog_errors(self):
        catalog = ["call mathforge.catalog with only the optional fields query, area and limit, as its input "
                   "schema describes"]
        for arguments, field in (({"bad": 1}, None), ({"limit": 0}, "limit"), ({"area": "nowhere"}, "area"),
                                 ({"query": "q" * 200}, "query")):
            with self.subTest(arguments=str(arguments)[:40]):
                response = call_tool("mathforge.catalog", arguments)
                self.assertEqual(response.payload["operations"], [])
                self.check(response, "input", "invalid_input", "invalid_input", catalog, field)

    def test_deferred_results_are_no_errors(self):
        reply = area("analysis", "function_integrate", expression={"text": "exp(x^2)", "variables": ["x"]},
                     variable="x", lower=0, upper=1)
        self.assertFalse(reply.is_error)
        self.assertIsNone(reply.payload["error"])
        self.assertIs(mf.from_data(reply.payload["result"]).outcome, mf.Outcome.NO_CONCLUSION)

    def test_errors_are_deterministic(self):
        for name, arguments in (("mathforge.solve", {"problem": "x^2=", "symbol_ids": {"x": X_ID}}),
                                ("mathforge.nope", {}),
                                ("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": 1.5, "b": 2}})):
            with self.subTest(tool=name):
                self.assertEqual(call_tool(name, arguments).payload, call_tool(name, arguments).payload)

    def test_the_existing_keys_keep_their_values(self):
        """The six keys of v0.8 are exactly as before; only reason and next_actions are new."""
        error = self.error(call_tool("mathforge.solve", {"problem": "x^2="}))
        self.assertEqual({key: error[key] for key in KEYS[:6]},
                         {"stage": "parsing", "code": "parse_syntax",
                          "message": "Expected a number, symbol, sqrt call, or parenthesized expression",
                          "field": "problem", "start": 4, "end": 4})
        error = self.error(area("discrete", "binomial", n=10.0, k=3))
        self.assertEqual({key: error[key] for key in KEYS[:6]},
                         {"stage": "decoding", "code": "invalid_input",
                          "message": "An exact integer is required: a JSON integer or a string of decimal digits; "
                                     "JSON decimal numbers and booleans are not exact",
                          "field": "arguments.n", "start": None, "end": None})

    def test_error_info_and_results_do_not_change(self):
        self.assertEqual([field.name for field in dataclasses.fields(ErrorInfo)], ["code", "message"])
        failed = call_tool("mathforge.algebraic_div", {"left": "1", "right": "0"})
        self.assertEqual(set(failed.payload["result"]["data"]["error"]), {"type", "code", "message"})
        self.assertNotIn("reason", json.dumps(failed.payload["result"]))

    def test_the_reason_table(self):
        """Each code maps to its reason; an unlisted code is its own reason, with no suggestion."""
        for code, reason in (("unknown_tool", "unknown_tool"), ("resource_limit_exceeded", "resource_limit_exceeded"),
                             ("limit_out_of_range", "limit_out_of_range"), ("response_too_large", "response_too_large"),
                             ("unknown_handle", "unknown_handle"), ("parse_syntax", "parse_syntax"),
                             ("unknown_symbol", "unknown_symbol"), ("invalid_symbol_table", "invalid_symbol_table"),
                             ("invalid_serialization", "invalid_document"), ("unsupported_operation", "unsupported"),
                             ("division_by_zero", "division_by_zero"), ("singular_matrix", "singular_matrix"),
                             ("execution_error", "execution_error"), ("something_new", "something_new")):
            with self.subTest(code=code):
                found, actions = _advice(code, "computation", None)
                self.assertEqual(found, reason)
                self.assertEqual(actions, list(NEXT_ACTIONS.get(reason, ())))
        self.assertEqual(_advice("invalid_input", "computation", None, status=mf.ExecutionStatus.UNSUPPORTED)[0],
                         "unsupported")
        self.assertEqual(_advice("anything", "presentation", None)[0], "presentation_failed")
        self.assertEqual(_advice("invalid_input", "input", "symbol_ids.x")[0], "invalid_symbol_ids")
        self.assertEqual(_advice("invalid_input", "input", None, reason="unknown_operation")[0], "unknown_operation")
        self.assertEqual(_advice("invalid_input", "computation", None, mismatch=True)[0], "symbol_mismatch")
        self.assertEqual(_advice("division_by_zero", "computation", None, mismatch=True)[0], "division_by_zero")
        self.assertEqual(_advice("resource_limit_exceeded", "parsing", None, raisable=False)[1], list(_AT_LIMIT))
        self.assertEqual(_advice("unknown_symbol", "parsing", "x", closed=True)[1], list(_CLOSED_TEXT))
        for actions in NEXT_ACTIONS.values():
            self.assertTrue(actions and all(type(action) is str and action for action in actions))


if __name__ == "__main__":
    unittest.main()
