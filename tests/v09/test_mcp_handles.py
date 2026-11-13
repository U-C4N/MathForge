"""Result handles of the MCP boundary (v0.9, MCP hardening).

Every response of a computation tool names its result with "mf:sha256:<hex>", the SHA-256 digest of the result's
canonical JSON, and keeps that JSON in a bounded per-process store; an argument that takes a canonical envelope also
takes a handle, which stands for the stored document (a Result for its value) and is decoded as that document sent in
full.
"""

import asyncio
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

try:
    from jsonschema import Draft202012Validator
except ImportError:  # Development-only (requirements-dev.txt); the isolated distribution harness has none.
    Draft202012Validator = None

import mathforge as mf
from mathforge import registry
from mathforge.algebraic import as_endpoint
from mathforge.capabilities import area_tools
from mathforge.limits import ComputationLimits, computation
from mathforge.mcp import handles
from mathforge.mcp.boundary import _FIELDS, _UNKNOWN_OPERAND, NEXT_ACTIONS, InputError, call_tool, resolve_handle
from mathforge.mcp.handles import HandleStore
from mathforge.mcp.kinds import KINDS, Context
from mathforge.mcp.schemas import catalog_schema, input_schemas, output_schema
from mathforge.workspace import Workspace, _content_ref

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v8"
UNKNOWN = "mf:sha256:" + "0" * 64  # Never the digest of a stored document in these tests.
X_ID = "00000000-0000-4000-8000-000000000901"
try:
    MCP_AVAILABLE = metadata.version("mcp").split(".")[0] == "2"
except metadata.PackageNotFoundError:
    MCP_AVAILABLE = False


def area(tool, operation, limits=None, symbol_ids=None, **arguments):
    request = {"operation": operation, "arguments": arguments}
    if limits is not None:
        request["limits"] = limits
    if symbol_ids is not None:
        request["symbol_ids"] = symbol_ids
    return call_tool("mathforge." + tool, request)


def value_envelope(response):
    """The canonical envelope of a response's Result value, as a client would send it in full."""
    return mf.to_data(mf.from_data(response.payload["result"]).value)


class StoreIsolation(unittest.TestCase):
    def setUp(self):
        handles.clear()
        self.addCleanup(handles.clear)

    def ok(self, response):
        self.assertFalse(response.is_error, response.payload["error"])
        return response.payload

    def error(self, response):
        self.assertTrue(response.is_error)
        self.assertIsNone(response.payload["result"])
        self.assertIsNone(response.payload["handle"])
        return response.payload["error"]


class HandleNamingTests(StoreIsolation):
    def test_the_handle_is_the_digest_of_the_canonical_result(self):
        payload = self.ok(area("ntheory", "continued_fraction", value="415/93"))
        canonical = json.dumps(payload["result"], sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(",", ":"))
        self.assertEqual(payload["handle"], "mf:sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        self.assertRegex(payload["handle"], r"\Amf:sha256:[0-9a-f]{64}\Z")

    def test_the_handle_and_the_workspace_reference_name_the_same_document(self):
        for response in (area("ntheory", "continued_fraction", value="415/93"),
                         call_tool("mathforge.solve", {"problem": "x^2-2=0", "symbol_ids": {"x": X_ID}}),
                         call_tool("mathforge.parse_function", {"text": "log(x)", "variables": ["x"]})):
            payload = self.ok(response)
            value = mf.from_data(payload["result"])
            self.assertEqual(payload["handle"], "mf:" + _content_ref(value).identifier)
            self.assertEqual(payload["handle"], "mf:" + Workspace().put(value).identifier)
            self.assertEqual(payload["handle"], "mf:sha256:" + hashlib.sha256(mf.dumps(value).encode()).hexdigest())

    def test_responses_stay_deterministic(self):
        for name, arguments in (("mathforge.solve", {"problem": "x^2-2=0", "symbol_ids": {"x": X_ID}}),
                                ("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": 12, "b": 18}}),
                                ("mathforge.compute", {"operation": "integer_gcd",
                                                       "arguments": {"a": mf.to_data(12), "b": mf.to_data(18)}})):
            with self.subTest(tool=name):
                first, second = call_tool(name, arguments), call_tool(name, arguments)
                self.assertEqual(first.payload, second.payload)
                self.assertIsNotNone(first.payload["handle"])

    def test_every_computation_payload_has_a_handle_key(self):
        cases = [("mathforge.solve", {"problem": "x=1"}), ("mathforge.solve", {"problem": "x="}),
                 ("mathforge.det", {"rows": [["1", "2"], ["3", "4"]]}), ("mathforge.nope", {}),
                 ("mathforge.verify", {"result": self.ok(call_tool("mathforge.solve", {"problem": "x=1"}))["result"]}),
                 ("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": 4, "b": 6}})]
        for name, arguments in cases:
            with self.subTest(tool=name):
                payload = call_tool(name, arguments).payload
                self.assertEqual(list(payload), ["api_version", "result", "handle", "text", "latex", "symbols",
                                                 "error"])
                self.assertEqual(payload["handle"] is None, payload["result"] is None)

    def test_catalog_and_capabilities_have_no_handle(self):
        self.assertNotIn("handle", call_tool("mathforge.capabilities", {}).payload)
        self.assertEqual(set(call_tool("mathforge.catalog", {"query": "gcd"}).payload),
                         {"api_version", "operations", "total", "error"})
        self.assertEqual(len(handles.STORE), 0)

    def test_returned_results_are_stored_and_failures_are_not(self):
        payload = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        self.assertIn(payload["handle"], handles.STORE)
        self.assertEqual(handles.STORE.get(payload["handle"]), handles.canonical_text(payload["result"]))
        self.error(call_tool("mathforge.solve", {"problem": "x="}))
        self.assertEqual(len(handles.STORE), 1)

    def test_a_response_too_large_names_and_stores_nothing(self):
        with patch("mathforge.mcp.boundary.MAX_RESPONSE_BYTES", 100):
            error = self.error(call_tool("mathforge.solve", {"problem": "x=1"}))
        self.assertEqual(error["code"], "response_too_large")
        self.assertEqual(len(handles.STORE), 0)

    def test_a_presentation_failure_keeps_and_names_the_result(self):
        with patch("mathforge.rendering.to_latex", side_effect=mf.InvalidInput("no room")):
            response = call_tool("mathforge.solve", {"problem": "x=1"})
        self.assertTrue(response.is_error)
        self.assertEqual(response.payload["handle"], handles.handle_of(response.payload["result"]))
        self.assertIn(response.payload["handle"], handles.STORE)


class RoundTripTests(StoreIsolation):
    def test_a_matrix_power_result_feeds_eigenvectors(self):
        power = self.ok(area("linalg", "matrix_power", M=[[2, 1], [0, 3]], k=2))
        by_handle = area("linalg", "eigenvectors", M=power["handle"])
        in_full = area("linalg", "eigenvectors", M=value_envelope(call_tool(
            "mathforge.linalg", {"operation": "matrix_power", "arguments": {"M": [[2, 1], [0, 3]], "k": 2}})))
        self.assertEqual(self.ok(by_handle), self.ok(in_full))  # The same request, so the same response.
        result = mf.from_data(by_handle.payload["result"])
        self.assertTrue(mf.verify(result).verified)
        self.assertEqual(result.request.arguments[0][1], mf.from_data(power["result"]).value)

    def test_an_algebra_result_is_passed_on(self):
        total = self.ok(area("algebra", "complex_add", left=[1, "sqrt(2)"], right=[2, 0]))
        squared = mf.from_data(self.ok(area("algebra", "complex_pow", base=total["handle"], n=2))["result"]).value
        self.assertEqual(squared, mf.from_data(total["result"]).value ** 2)

    def test_a_handle_may_stand_for_an_element_or_an_interval_end(self):
        root = self.ok(call_tool("mathforge.algebraic", {"expression": "sqrt(2)"}))
        alpha = mf.from_data(root["result"]).value
        element = self.ok(area("algebra", "complex_add", left=[root["handle"], 0], right=[1, 0]))
        self.assertEqual(mf.from_data(element["result"]).value, mf.ComplexAlgebraic(alpha + 1, 0))
        interval = {"lower": 0, "upper": root["handle"], "left_closed": True, "right_closed": False}
        union = self.ok(area("algebra", "set_union", a=[interval], b=[]))
        self.assertEqual(mf.from_data(union["result"]).value,
                         mf.IntervalSet((mf.Interval(0, as_endpoint(alpha), True, False),)))
        bound = self.ok(area("analysis", "function_evaluate", expression={"text": "x^2", "variables": ["x"]},
                             bindings={"x": root["handle"]}))
        self.assertTrue(mf.verify(mf.from_data(bound["result"])).verified)

    def test_a_document_that_is_not_a_result_stands_for_itself(self):
        parsed = self.ok(call_tool("mathforge.parse_function", {"text": "log(x)+sin(x)", "variables": ["x"],
                                                                "symbol_ids": {"x": X_ID}}))
        by_handle = area("analysis", "function_differentiate", expression=parsed["handle"], variable="x",
                         symbol_ids={"x": X_ID})
        in_full = area("analysis", "function_differentiate", expression=parsed["result"], variable="x",
                       symbol_ids={"x": X_ID})
        self.assertEqual(self.ok(by_handle), self.ok(in_full))

    def test_compute_accepts_handles_in_its_arguments(self):
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        by_handle = call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                    "arguments": {"a": gcd["handle"], "b": mf.to_data(4)}})
        in_full = call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                  "arguments": {"a": mf.to_data(6), "b": mf.to_data(4)}})
        self.assertEqual(self.ok(by_handle), self.ok(in_full))
        self.assertEqual(mf.from_data(by_handle.payload["result"]).value, 2)


class ReferenceErrorTests(StoreIsolation):
    def test_an_unknown_handle_is_an_explicit_input_error(self):
        cases = (("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": UNKNOWN, "b": 4}}, "arguments.a"),
                 ("mathforge.algebra", {"operation": "complex_add",
                                        "arguments": {"left": [1, UNKNOWN], "right": [1, 0]}}, "arguments.left[1]"),
                 ("mathforge.algebra", {"operation": "set_union", "arguments": {"a": [
                     {"lower": UNKNOWN, "upper": None, "left_closed": False, "right_closed": False}], "b": []}},
                  "arguments.a[0].lower"),
                 ("mathforge.compute", {"operation": "integer_gcd",
                                        "arguments": {"a": mf.to_data(4), "b": UNKNOWN}}, "arguments.b"))
        for tool, arguments, field in cases:
            with self.subTest(field=field, tool=tool):
                error = self.error(call_tool(tool, arguments))
                self.assertEqual((error["stage"], error["code"], error["field"], error["reason"]),
                                 ("input", "unknown_handle", field, "unknown_handle"))
                self.assertEqual(error["next_actions"], list(_UNKNOWN_OPERAND if tool == "mathforge.compute"
                                                             else NEXT_ACTIONS["unknown_handle"]))

    def test_an_evicted_handle_is_unknown(self):
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        handles.clear()
        error = self.error(area("ntheory", "integer_gcd", a=gcd["handle"], b=4))
        self.assertEqual(error["code"], "unknown_handle")

    def test_the_advice_of_an_unknown_handle_recovers_the_call(self):
        """Both next actions of unknown_handle work for the handle of a Result, whose document itself is no operand."""
        request = {"operation": "matrix_power", "arguments": {"M": [[2, 1], [0, 3]], "k": 2}}
        power = self.ok(call_tool("mathforge.linalg", request))
        handles.clear()
        error = self.error(area("linalg", "eigenvectors", M=power["handle"]))
        self.assertEqual(error["next_actions"], list(NEXT_ACTIONS["unknown_handle"]))
        # The Result document that the handle named is no operand, so the advice never says to send it.
        self.assertEqual(self.error(area("linalg", "eigenvectors", M=power["result"]))["code"], "invalid_input")
        again = self.ok(call_tool("mathforge.linalg", request))  # Repeat the call ...
        self.assertEqual(again["handle"], power["handle"])
        by_handle = self.ok(area("linalg", "eigenvectors", M=again["handle"]))  # ... and pass its new handle,
        friendly = self.ok(area("linalg", "eigenvectors", M=[[4, 5], [0, 9]]))  # or send the friendly form.
        self.assertEqual(by_handle["result"], friendly["result"])
        # compute has no friendly form: its advice names the operand's canonical envelope.
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        handles.clear()
        error = self.error(call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                           "arguments": {"a": gcd["handle"], "b": mf.to_data(4)}}))
        self.assertEqual(error["next_actions"], list(_UNKNOWN_OPERAND))
        full = call_tool("mathforge.compute", {"operation": "integer_gcd",
                                               "arguments": {"a": mf.to_data(6), "b": mf.to_data(4)}})
        self.assertEqual(mf.from_data(self.ok(full)["result"]).value, 2)

    def test_a_symbolic_handle_without_symbol_ids_names_the_mismatch(self):
        """Text of a later call without symbol_ids makes a new symbol; the error says to pass symbol_ids."""
        derivative = self.ok(area("analysis", "function_differentiate",
                                  expression={"text": "x^3", "variables": ["x"]}, variable="x"))
        ids = {entry["name"]: entry["uid"] for entry in derivative["symbols"]}
        for operation, arguments in (("function_integrate", {"variable": "x", "lower": 0, "upper": 1}),
                                     ("function_antiderivative", {"variable": "x"}),
                                     ("function_taylor", {"variable": "x", "point": 0, "order": 2})):
            with self.subTest(operation=operation):
                reply = area("analysis", operation, expression=derivative["handle"], **arguments)
                self.assertTrue(reply.is_error)  # A computation error keeps the failed Result.
                error = reply.payload["error"]
                self.assertEqual((error["stage"], error["code"], error["reason"], error["next_actions"]),
                                 ("computation", "invalid_input", "symbol_mismatch",
                                  list(NEXT_ACTIONS["symbol_mismatch"])))
                self.ok(area("analysis", operation, expression=derivative["handle"], symbol_ids=ids, **arguments))
        # Without a shared name, an invalid input keeps its ordinary reason.
        reply = area("analysis", "function_integrate", expression=derivative["handle"], variable="y", lower=0,
                     upper=1, symbol_ids=ids)
        self.assertTrue(reply.is_error)
        self.assertEqual(reply.payload["error"]["reason"], "invalid_input")

    def test_every_kind_reads_a_handle_as_a_reference_never_as_friendly_input(self):
        """No friendly string form holds a colon, so each kind reports the unknown handle, also those that take
        names, text or enumerated strings; the same string without its last digit is friendly input again."""
        for name, kind in KINDS.items():
            with self.subTest(kind=name):
                context = Context(ComputationLimits())
                with self.assertRaises(mf.MathForgeError) as caught:
                    kind.decode(UNKNOWN, context)
                self.assertEqual((caught.exception.code, caught.exception.field), ("unknown_handle", "arguments"))
                with self.assertRaises(mf.MathForgeError) as caught:
                    kind.decode(UNKNOWN[:-1], Context(ComputationLimits()))
                self.assertNotEqual(caught.exception.code, "unknown_handle")
        from mathforge.functions import parse_function
        from mathforge.parsing import parse_expression, parse_problem
        for parse in (lambda text: parse_expression(text, symbols={}), lambda text: parse_problem(text, symbols={}),
                      lambda text: parse_function(text, symbols={}, variables=())):
            for text in (UNKNOWN, "a:b", "1:2"):
                with self.subTest(text=text), self.assertRaises(mf.InvalidInput):
                    parse(text)

    def test_a_malformed_reference_is_an_ordinary_invalid_string(self):
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        digest = gcd["handle"][len("mf:sha256:"):]
        expected = self.error(area("ntheory", "integer_gcd", a="twelve", b=4))
        for malformed in ("mf:sha256:" + digest.upper(), "mf:sha256:" + digest[:-1], "mf:sha256:" + digest + "0",
                          "sha256:" + digest, "mf:sha512:" + digest, "MF:sha256:" + digest,
                          gcd["handle"] + "\n", " " + gcd["handle"]):
            with self.subTest(reference=malformed):
                self.assertEqual(self.error(area("ntheory", "integer_gcd", a=malformed, b=4)), expected)
                scalar = self.error(area("algebra", "complex_algebraic", real=malformed, imag=0))
                self.assertEqual((scalar["stage"], scalar["code"], scalar["field"]),
                                 ("decoding", "parse_syntax", "arguments.real"))
                compute = self.error(call_tool("mathforge.compute", {
                    "operation": "integer_gcd", "arguments": {"a": malformed, "b": mf.to_data(4)}}))
                self.assertEqual((compute["code"], compute["field"]), ("invalid_input", "arguments.a"))

    def test_a_result_without_a_value_stands_for_no_argument(self):
        failed = call_tool("mathforge.algebraic_div", {"left": "1", "right": "0"})
        self.assertTrue(failed.is_error)
        self.assertIsNone(mf.from_data(failed.payload["result"]).value)
        self.assertIn(failed.payload["handle"], handles.STORE)
        deferred = self.ok(area("analysis", "function_integrate", expression={"text": "exp(x^2)", "variables": ["x"]},
                                variable="x", lower=0, upper=1))
        self.assertIsNone(mf.from_data(deferred["result"]).value)
        for reference in (failed.payload["handle"], deferred["handle"]):
            for tool, arguments in (("mathforge.ntheory", {"operation": "integer_gcd",
                                                           "arguments": {"a": reference, "b": 4}}),
                                    ("mathforge.compute", {"operation": "integer_gcd",
                                                           "arguments": {"a": reference, "b": mf.to_data(4)}})):
                with self.subTest(tool=tool, reference=reference):
                    error = self.error(call_tool(tool, arguments))
                    self.assertEqual((error["code"], error["field"], error["reason"]),
                                     ("invalid_input", "arguments.a", "invalid_input"))
                    self.assertIn("without a value", error["message"])

    def test_the_store_never_returns_a_document_that_the_handle_does_not_name(self):
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        other = self.ok(area("ntheory", "integer_gcd", a=10, b=15))
        self.assertNotEqual(mf.from_data(gcd["result"]).value, mf.from_data(other["result"]).value)
        for corrupt in (handles.STORE.get(other["handle"]), handles.STORE.get(gcd["handle"]).replace("6", "7"), "{}"):
            with self.subTest(corrupt=corrupt[:40]):
                self.ok(area("ntheory", "integer_gcd", a=12, b=18))  # Stores the true text again.
                with handles.STORE._lock:
                    handles.STORE._entries[gcd["handle"]] = corrupt
                    handles.STORE._bytes += len(corrupt) - len(handles.canonical_text(gcd["result"]))
                error = self.error(area("ntheory", "integer_gcd", a=gcd["handle"], b=4))
                self.assertEqual(error["code"], "unknown_handle")
                self.assertNotIn(gcd["handle"], handles.STORE)  # The mismatching entry is dropped.
                self.assertEqual(handles.STORE.nbytes, len(handles.STORE.get(other["handle"])))

    def test_handles_get_the_checks_of_the_kind(self):
        """A handle is decoded exactly as its document sent in full: every refusal of the envelope stays."""
        power = self.ok(area("linalg", "matrix_power", M=[[2, 1], [0, 3]], k=2))
        matrix = value_envelope(call_tool("mathforge.linalg", {"operation": "matrix_power",
                                                               "arguments": {"M": [[2, 1], [0, 3]], "k": 2}}))
        by_handle = call_tool("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": power["handle"], "b": 4}})
        in_full = call_tool("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": matrix, "b": 4}})
        self.assertTrue(by_handle.is_error)
        self.assertEqual(by_handle.payload["error"], in_full.payload["error"])
        self.assertEqual(by_handle.payload["error"]["code"], "invalid_input")
        report = self.ok(call_tool("mathforge.verify", {"result": power["result"]}))  # A VerificationReport document
        refused = self.error(area("ntheory", "integer_gcd", a=report["handle"], b=4))
        full = self.error(area("ntheory", "integer_gcd", a=report["result"], b=4))
        self.assertEqual(refused, full)
        self.assertEqual((refused["stage"], refused["field"]), ("decoding", "arguments.a"))
        record = self.ok(area("ntheory", "continued_fraction", value="7/3"))  # Its value is a v6 record.
        compute = call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                  "arguments": {"a": record["handle"], "b": mf.to_data(4)}})
        compute_full = call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                       "arguments": {"a": value_envelope(call_tool(
                                                           "mathforge.ntheory", {"operation": "continued_fraction",
                                                                                 "arguments": {"value": "7/3"}})),
                                                                     "b": mf.to_data(4)}})
        self.assertEqual(self.error(compute), self.error(compute_full))

    def test_resolving_a_handle_charges_the_active_request_budget(self):
        """resolve_handle rebuilds a Result's value in the request's own budget, never in a fresh one."""
        rows = [[(i * 7 + j) % 5 for j in range(6)] for i in range(6)]
        power = self.ok(area("linalg", "matrix_power", M=rows, k=3))
        limits = ComputationLimits()
        with computation(limits, fresh=True) as budget:
            before = (budget.work, budget.node_count)
            envelope = resolve_handle(power["handle"], "arguments.M", limits)
            self.assertGreater((budget.work, budget.node_count), before)
        self.assertEqual(envelope, value_envelope(call_tool("mathforge.linalg", {"operation": "matrix_power",
                                                                                 "arguments": {"M": rows, "k": 3}})))
        # A budget that the rebuild alone exhausts fails in resolve_handle, before any kind sees the envelope.
        tight = ComputationLimits(max_nodes=20)
        with computation(tight, fresh=True), self.assertRaises(InputError) as raised:
            resolve_handle(power["handle"], "arguments.M", tight)
        self.assertEqual((raised.exception.code, raised.exception.field), ("resource_limit_exceeded", "arguments.M"))

    def test_resolving_a_handle_applies_the_request_decode_limits(self):
        """The rebuild uses the decoder limits of an envelope sent in the request: its depth follows limits."""
        power = self.ok(area("linalg", "matrix_power", M=[[2, 1], [0, 3]], k=2))
        shallow = ComputationLimits(max_depth=2)  # Below the depth of a matrix document, far below DecodeLimits()'s.
        with computation(shallow, fresh=True), self.assertRaises(InputError) as raised:
            resolve_handle(power["handle"], "arguments.M", shallow)
        self.assertEqual((raised.exception.code, raised.exception.field), ("resource_limit_exceeded", "arguments.M"))
        with computation(ComputationLimits(), fresh=True):
            resolve_handle(power["handle"], "arguments.M", ComputationLimits())

    def test_a_handle_is_charged_to_the_request_budget(self):
        rows = [[(i * 7 + j) % 5 for j in range(8)] for i in range(8)]
        power = self.ok(area("linalg", "matrix_power", M=rows, k=3))
        matrix = value_envelope(call_tool("mathforge.linalg", {"operation": "matrix_power",
                                                               "arguments": {"M": rows, "k": 3}}))
        tight = {"max_nodes": 50}
        full = area("linalg", "eigenvectors", M=matrix, limits=tight)
        by_handle = area("linalg", "eigenvectors", M=power["handle"], limits=tight)
        self.assertEqual(self.error(full)["code"], "resource_limit_exceeded")
        self.assertEqual(self.error(by_handle)["code"], "resource_limit_exceeded")
        self.assertEqual(self.error(by_handle)["field"], "arguments.M")
        compute = call_tool("mathforge.compute", {"operation": "trace", "arguments": {"A": power["handle"]},
                                                  "limits": tight})
        self.assertEqual(self.error(compute)["code"], "resource_limit_exceeded")
        # The request byte limit counts each document a handle stands for, as if it had been sent in full.
        size = len(json.dumps({"operation": "eigenvectors", "arguments": {"M": matrix}}).encode())
        with patch("mathforge.mcp.boundary.MAX_REQUEST_BYTES", size - 1):
            self.assertEqual(self.error(area("linalg", "eigenvectors", M=matrix))["code"], "resource_limit_exceeded")
            error = self.error(area("linalg", "eigenvectors", M=power["handle"]))
            self.assertEqual((error["code"], error["field"]), ("resource_limit_exceeded", "arguments.M"))
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        request = {"operation": "integer_gcd", "arguments": {"a": gcd["handle"], "b": gcd["handle"]}}
        grows = len(json.dumps(mf.to_data(6))) - len(json.dumps(gcd["handle"]))  # Each handle's document adds this.
        self.assertGreater(grows, 0)
        limit = len(json.dumps(request).encode()) + grows  # Room for one document in place of its handle, not two.
        with patch("mathforge.mcp.boundary.MAX_REQUEST_BYTES", limit):
            for tool, arguments in (("mathforge.ntheory", request["arguments"]),
                                    ("mathforge.compute", request["arguments"])):
                with self.subTest(tool=tool):
                    twice = self.error(call_tool(tool, {"operation": "integer_gcd", "arguments": arguments}))
                    self.assertEqual((twice["code"], twice["field"]), ("resource_limit_exceeded", "arguments.b"))
            once = call_tool("mathforge.ntheory", {"operation": "integer_gcd",
                                                   "arguments": {"a": gcd["handle"], "b": 6}})
            self.assertEqual(mf.from_data(self.ok(once)["result"]).value, 6)


class StoreTests(unittest.TestCase):
    def test_default_bounds(self):
        self.assertEqual((handles.MAX_DOCUMENTS, handles.MAX_BYTES), (256, 64 * 1024 * 1024))
        self.assertEqual((handles.STORE.max_documents, handles.STORE.max_bytes), (256, 64 * 1024 * 1024))
        self.assertEqual(handles.PATTERN, "^mf:sha256:[0-9a-f]{64}$")

    @staticmethod
    def put(store, text):
        handle = handles.handle_of_text(text)
        return handle, store.put(handle, text)

    def test_least_recently_used_documents_go_first_by_count(self):
        store = HandleStore(max_documents=3, max_bytes=10 ** 6)
        names = [self.put(store, json.dumps([i]))[0] for i in range(3)]
        self.assertEqual(store.get(names[0]), "[0]")  # Now the most recently used.
        fourth, kept = self.put(store, "[3]")
        self.assertTrue(kept)
        self.assertEqual(len(store), 3)
        self.assertIsNone(store.get(names[1]))
        self.assertEqual([store.get(name) for name in (names[0], names[2], fourth)], ["[0]", "[2]", "[3]"])

    def test_least_recently_used_documents_go_first_by_bytes(self):
        store = HandleStore(max_documents=100, max_bytes=100)
        texts = ["[" + str(i) * 38 + "]" for i in range(1, 4)]  # 40 characters each
        names = [self.put(store, text)[0] for text in texts]
        self.assertEqual(len(store), 2)
        self.assertEqual(store.nbytes, 80)
        self.assertIsNone(store.get(names[0]))
        self.assertEqual(store.get(names[2]), texts[2])
        big = "[" + "9" * 99 + "]"  # 101 characters: never stored, and nothing is evicted for it.
        handle, kept = self.put(store, big)
        self.assertFalse(kept)
        self.assertIsNone(store.get(handle))
        self.assertEqual((len(store), store.nbytes), (2, 80))
        exact = "[" + "8" * 98 + "]"  # Exactly the cap: it evicts every other document.
        handle, kept = self.put(store, exact)
        self.assertTrue(kept)
        self.assertEqual((len(store), store.nbytes, store.get(handle)), (1, 100, exact))

    def test_storing_again_refreshes_without_growing(self):
        store = HandleStore(max_documents=2, max_bytes=1000)
        first, _ = self.put(store, "[1]")
        self.put(store, "[2]")
        self.put(store, "[1]")
        self.put(store, "[3]")
        self.assertEqual((len(store), store.nbytes, store.get(first)), (2, 6, "[1]"))

    def test_eviction_through_the_boundary(self):
        small = HandleStore(max_documents=2)
        with patch.object(handles, "STORE", small):
            first = call_tool("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": 12, "b": 18}})
            for b in (19, 20):
                call_tool("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": 12, "b": b}})
            error = call_tool("mathforge.ntheory", {"operation": "integer_gcd",
                                                    "arguments": {"a": first.payload["handle"], "b": 4}})
            self.assertEqual(error.payload["error"]["code"], "unknown_handle")
            self.assertEqual(len(small), 2)

    def test_concurrent_use_keeps_the_store_consistent(self):
        store = HandleStore(max_documents=16, max_bytes=2000)
        texts = [json.dumps(list(range(i % 40))) for i in range(200)]
        failures = []

        def work(offset):
            try:
                for i in range(400):
                    text = texts[(offset * 7 + i) % len(texts)]
                    handle = handles.handle_of_text(text)
                    store.put(handle, text)
                    found = store.get(handle)
                    if found is not None and found != text:
                        failures.append((handle, found))
            except Exception as exc:  # A failure in a thread must fail the test.
                failures.append(exc)

        threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])
        self.assertLessEqual(len(store), 16)
        with store._lock:
            entries = dict(store._entries)
        self.assertEqual(store.nbytes, sum(map(len, entries.values())))
        self.assertLessEqual(store.nbytes, 2000)
        for handle, text in entries.items():
            self.assertEqual(handles.handle_of_text(text), handle)

    def test_is_handle_matches_the_whole_string(self):
        self.assertTrue(handles.is_handle(UNKNOWN))
        for value in (UNKNOWN + "\n", UNKNOWN.upper(), UNKNOWN[:-1], "x" + UNKNOWN, None, 1, [UNKNOWN],
                      UNKNOWN.replace("0", "g")):
            self.assertFalse(handles.is_handle(value), value)


class SchemaTests(StoreIsolation):
    def test_output_schema_requires_the_handle(self):
        schema = output_schema()
        self.assertIn("handle", schema["required"])
        self.assertEqual(schema["properties"]["handle"], {"type": ["string", "null"],
                                                          "pattern": "^mf:sha256:[0-9a-f]{64}$"})
        self.assertFalse(schema["additionalProperties"])

    def test_real_payloads_satisfy_the_output_schema(self):
        if Draft202012Validator is None:
            self.skipTest("Development-only jsonschema is unavailable")
        validator = Draft202012Validator(output_schema())
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        with patch("mathforge.rendering.to_latex", side_effect=mf.InvalidInput("no room")):
            presentation = call_tool("mathforge.solve", {"problem": "x=1"})
        with patch("mathforge.mcp.boundary.MAX_RESPONSE_BYTES", 100):
            too_large = call_tool("mathforge.solve", {"problem": "x=1"})
        responses = [area("ntheory", "integer_gcd", a=12, b=18), area("ntheory", "integer_gcd", a=1.5, b=18),
                     area("ntheory", "integer_gcd", a=UNKNOWN, b=18), area("ntheory", "integer_gcd", a=gcd["handle"], b=4),
                     area("ntheory", "nope"), call_tool("mathforge.nope", {}),
                     call_tool("mathforge.solve", {"problem": "x^3-sqrt(2)=0"}),
                     call_tool("mathforge.algebraic_div", {"left": "1", "right": "0"}),
                     area("analysis", "function_integrate", expression={"text": "exp(x^2)", "variables": ["x"]},
                          variable="x", lower=0, upper=1),
                     call_tool("mathforge.verify", {"result": gcd["result"]}),
                     call_tool("mathforge.parse_function", {"text": "log(x)", "variables": ["x"]}),
                     call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                     "arguments": {"a": gcd["handle"], "b": mf.to_data(4)}}),
                     call_tool("mathforge.solve", {"problem": "x=1", "limits": {"max_work": 0}}),
                     presentation, too_large]
        for response in responses:
            with self.subTest(payload=json.dumps(response.payload)[:100]):
                self.assertEqual(sorted(validator.iter_errors(response.payload), key=str), [])
        catalog = Draft202012Validator(catalog_schema())
        for arguments in ({"query": "gcd"}, {"limit": 0}):
            self.assertEqual(list(catalog.iter_errors(call_tool("mathforge.catalog", arguments).payload)), [])

    def test_area_schemas_advertise_handles(self):
        if Draft202012Validator is None:
            self.skipTest("Development-only jsonschema is unavailable")
        schemas = input_schemas()
        for tool in area_tools():
            with self.subTest(tool=tool):
                schema = schemas[tool]
                self.assertEqual(schema["$defs"]["handle"]["pattern"], handles.PATTERN)
                for name in registry.KIND_NAMES:
                    if name in schema["$defs"]:
                        self.assertIn({"$ref": "#/$defs/handle"}, schema["$defs"][name]["anyOf"])
                self.assertLessEqual(len(json.dumps(schema, separators=(",", ":")).encode()), 16 * 1024)
        self.assertLessEqual(sum(len(json.dumps(schema, separators=(",", ":")).encode())
                                 for schema in schemas.values()), 128 * 1024)
        validator = Draft202012Validator(schemas["mathforge.linalg"])
        self.assertTrue(validator.is_valid({"operation": "eigenvectors", "arguments": {"M": UNKNOWN}}))
        self.assertFalse(validator.is_valid({"operation": "eigenvectors", "arguments": {"M": UNKNOWN.upper()}}))

    def test_the_v07_input_schemas_are_unchanged(self):
        frozen = json.loads((FIXTURES / "mcp_v7_input_schemas.json").read_text(encoding="utf-8"))
        schemas = input_schemas()
        self.assertEqual(set(frozen), {"mathforge." + name for name in _FIELDS})
        for name, schema in frozen.items():
            with self.subTest(tool=name):
                self.assertEqual(json.dumps(schemas[name], sort_keys=True), json.dumps(schema, sort_keys=True))
        self.assertEqual(schemas["mathforge.verify"]["properties"]["result"], {"type": "object"})

    def test_verify_takes_no_handle(self):
        gcd = self.ok(area("ntheory", "integer_gcd", a=12, b=18))
        response = call_tool("mathforge.verify", {"result": gcd["handle"]})
        self.assertTrue(response.is_error)
        self.assertNotEqual(response.payload["error"]["code"], "unknown_handle")



@unittest.skipUnless(MCP_AVAILABLE, "Optional MCP SDK v2 is not installed")
class StdioHandleTests(unittest.TestCase):
    def test_a_handle_lives_as_long_as_the_server_process(self):
        asyncio.run(self._round_trip())

    async def _round_trip(self):
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters

        env = dict(os.environ, PYTHONPATH=str(Path(mf.__file__).resolve().parent.parent))
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "mathforge.mcp"], env=env)
        for attempt in range(2):  # A fresh server process knows none of the previous one's handles.
            async with Client(parameters, read_timeout_seconds=30) as client:
                await client.list_tools()  # The client validates structured content against the output schemas.
                if attempt:
                    stale = await client.call_tool("mathforge.linalg", {"operation": "eigenvectors",
                                                                        "arguments": {"M": handle}})
                    self.assertTrue(stale.is_error)
                    self.assertEqual(stale.structured_content["error"]["reason"], "unknown_handle")
                    continue
                power = await client.call_tool("mathforge.linalg", {"operation": "matrix_power",
                                                                    "arguments": {"M": [[2, 1], [0, 3]], "k": 2}})
                handle = power.structured_content["handle"]
                self.assertEqual(handle, handles.handle_of(power.structured_content["result"]))
                vectors = await client.call_tool("mathforge.linalg", {"operation": "eigenvectors",
                                                                      "arguments": {"M": handle}})
                self.assertFalse(vectors.is_error, vectors)
                self.assertTrue(mf.verify(mf.from_data(vectors.structured_content["result"])).verified)
                self.assertEqual(json.loads(vectors.content[0].text), vectors.structured_content)

if __name__ == "__main__":
    unittest.main()
