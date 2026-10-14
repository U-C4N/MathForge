"""Adversarial provenance, wire compatibility and request-local resource checks."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge.limits import computation, current_budget
from mathforge.mcp.boundary import call_tool
from mathforge.verification import verify_step


class IntegrationAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000661")
        self.result = m.operations.charpoly(m.Matrix(((1, 2), (3, 4))), self.x)

    def test_forged_routing_fields_are_rejected_without_attribute_errors(self):
        for request in (object(), "charpoly", (), []):
            forged = copy(self.result)
            object.__setattr__(forged, "request", request)
            self.assertFalse(m.verify(forged).verified)
        for operation in (None, [], 4):
            request = copy(self.result.request)
            object.__setattr__(request, "operation", operation)
            forged = copy(self.result)
            object.__setattr__(forged, "request", request)
            self.assertFalse(m.verify(forged).verified)
        for steps in (None, [self.result.steps[0]], (object(),)):
            forged = copy(self.result)
            object.__setattr__(forged, "steps", steps)
            self.assertFalse(m.verify(forged).verified)
        for rule in (None, [], 4):
            step = copy(self.result.steps[0])
            object.__setattr__(step, "rule", rule)
            forged = copy(self.result)
            object.__setattr__(forged, "steps", (step,))
            self.assertFalse(m.verify(forged).verified)
            self.assertIs(verify_step(step).status, m.CheckStatus.FAILED)
        for step in (object(), None, "v4.charpoly"):
            self.assertIs(verify_step(step).status, m.CheckStatus.FAILED)

    def test_every_contract_and_source_alias_is_checked_despite_stored_audit(self):
        result = self.result
        changes = ({"outcome": m.Outcome.PARTIAL}, {"exactness": m.Exactness.APPROXIMATE},
                   {"completeness": m.Completeness.PARTIAL}, {"domain": None},
                   {"for_": None}, {"problem": m.Eq(self.x, 0)},
                   {"expression": m.Matrix(((0, 0), (0, 0)))}, {"method": "det"},
                   {"precision": 10}, {"tolerance": m.Rational(0)},
                   {"error_bound": m.Rational(0)},
                   {"error": m.ErrorInfo("forged", "Forged success with an error")},
                   {"steps": ()}, {"steps": result.steps * 2})
        self.assertTrue(result.verification)
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(m.verify(replace(result, **change)).verified)
        step = replace(result.steps[0], outputs=(result.value + 1, result.steps[0].outputs[1]))
        self.assertTrue(step.audit)
        self.assertFalse(m.verify(replace(result, value=result.value+1, steps=(step,))).verified)

    def test_request_allowlist_and_exact_argument_names_prevent_dispatch_smuggling(self):
        result = self.result
        for arguments in ((), result.request.arguments + (("extra", 1),),
                          tuple(pair for pair in result.request.arguments if pair[0] != "A")):
            request = replace(result.request, arguments=arguments)
            forged = replace(result, request=request, steps=(replace(result.steps[0], inputs=(request,)),))
            self.assertFalse(m.verify(forged).verified)
        request = copy(result.request)
        object.__setattr__(request, "arguments", result.request.arguments + result.request.arguments[:1])
        self.assertFalse(m.verify(replace(result, request=request,
            steps=(replace(result.steps[0], inputs=(request,)),))).verified)
        request = m.OperationRequest("os.system", (("command", "arbitrary-code"),))
        step = m.Step("v4.os.system", (request,), (m.Rational(1), ()), relation=m.StepRelation.EVALUATION)
        with patch("os.system", side_effect=AssertionError("Persisted requests must not execute code")):
            restored = m.loads(m.dumps(request))
            self.assertEqual(restored, request)
            self.assertFalse(verify_step(step).status is m.CheckStatus.VERIFIED)
            forged = replace(result, request=request, method="os.system", steps=(step,))
            self.assertFalse(m.verify(forged).verified)

    def test_same_uuid_with_conflicting_metadata_cannot_rebind_a_certificate(self):
        alias = m.Symbol("renamed", uid=self.x.uid)
        request = replace(self.result.request, arguments=(("A", self.result.expression), ("variable", alias)))
        forged = replace(self.result, for_=alias, request=request,
                         steps=(replace(self.result.steps[0], inputs=(request,)),))
        self.assertFalse(m.verify(forged).verified)
        workspace = m.Workspace()
        workspace.put(self.result)
        with self.assertRaises(m.SerializationError):
            workspace.put(alias)
        document = m.to_data((self.x, m.Symbol("other")))
        symbols = document["data"]["items"]
        symbols[1]["uid"] = symbols[0]["uid"]
        with self.assertRaises(m.SerializationError):
            m.from_data(document)

    def test_legacy_generic_and_failed_integrate_requests_keep_bytes_and_references(self):
        source = Path(__file__).parent / "fixtures/v5/serialization_v3.py"
        spec = importlib.util.spec_from_file_location("mathforge._audit_frozen_v3", source)
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        requests = (m.OperationRequest("integrate", ()),
                    m.OperationRequest("integrate", (("arbitrary", "old generic request"),)),
                    m.OperationRequest("integrate", (("expression", self.x), ("variable", self.x),
                                                    ("lower", m.Rational(0)), ("upper", m.sqrt(2)))))
        failed = m.Result(m.ExecutionStatus.UNSUPPORTED, m.Outcome.NOT_APPLICABLE,
                          request=requests[-1], expression=self.x, for_=self.x,
                          domain=m.Reals, method="integrate",
                          error=m.ErrorInfo("unsupported_operation", "Rational bounds required"))
        for value in (*requests, failed):
            with self.subTest(value=value):
                encoded = frozen.dumps(value)
                self.assertEqual(m.dumps(value), encoded)
                self.assertEqual(m.dumps(m.loads(encoded)), encoded)
                reference = m.Workspace().put(value)
                self.assertEqual(reference.identifier, "sha256:" + hashlib.sha256(encoded.encode()).hexdigest())
                self.assertEqual(m.to_data(value)["version"], 2)

    def test_mixed_workspace_hashes_and_versions_survive_until_content_is_changed(self):
        values = (m.Rational(7), m.RealAlgebraicRoot((-2, 0, 0, 1), 0),
                  m.operations.limit(self.x, self.x, 0), self.result)
        workspace = m.Workspace()
        refs = tuple(workspace.put(value) for value in values)
        workspace.record("charpoly", {"matrix": refs[0]}, {}, refs[-1])
        document = m.to_data(workspace)
        self.assertEqual(document["version"], 4)
        restored = m.from_data(document)
        self.assertEqual(tuple(restored.put(value) for value in values), refs)
        self.assertEqual(tuple(m.to_data(restored.get(ref))["version"] for ref in refs), (1, 2, 3, 4))
        forged = deepcopy(document)
        entry = next(entry for entry in forged["data"]["objects"] if entry["object"]["type"] == "rational")
        entry["object"]["numerator"] = "8"
        with self.assertRaises(m.SerializationError):
            m.from_data(forged)
        for version in (1, 2, 3):
            forged = deepcopy(document)
            forged["version"] = version
            with self.assertRaises(m.SerializationError):
                m.from_data(forged)
        limit_request = m.OperationRequest("limit", (("expression", self.x), ("variable", self.x),
                                                    ("point", m.sqrt(2)), ("side", "both")))
        document = m.to_data(limit_request)
        self.assertEqual(document["version"], 4)
        document["version"] = 3
        with self.assertRaises(m.SerializationError):
            m.from_data(document)

    def test_decoder_limits_and_budget_context_restore_after_failure(self):
        from mathforge import serialization
        document = m.to_data(m.Rational(123))
        outer_limits = m.DecodeLimits(max_nodes=10)
        token = serialization._decode_limits.set(outer_limits)
        try:
            with computation(m.ComputationLimits(max_work=1), fresh=True) as budget:
                for decode_limits in (m.DecodeLimits(max_depth=1), m.DecodeLimits(max_integer_digits=1)):
                    with self.assertRaises(m.SerializationError):
                        m.from_data(document, decode_limits=decode_limits)
                    self.assertIs(serialization._decode_limits.get(), outer_limits)
                    self.assertIs(current_budget(), budget)
                self.assertEqual(m.from_data(document).numerator, 123)
                self.assertEqual(budget.work, 0)
        finally:
            serialization._decode_limits.reset(token)
        self.assertIsNone(current_budget())

    def test_each_mcp_request_enforces_its_own_limits_inside_an_ambient_budget(self):
        with computation(m.ComputationLimits(max_work=1), fresh=True) as outer:
            outer.tick()
            for operation, arguments in (("factor", {"expression": "x^2-1"}),
                                         ("charpoly", {"rows": [["1", "2"], ["3", "4"]]})):
                failed = call_tool("mathforge."+operation, arguments | {"limits": {"max_work": 1}})
                self.assertTrue(failed.is_error)
                self.assertEqual(failed.payload["error"]["code"], "resource_limit_exceeded")
                passed = call_tool("mathforge."+operation, arguments)
                self.assertFalse(passed.is_error, passed.payload)
                self.assertIs(current_budget(), outer)
                self.assertEqual((outer.work, outer.node_count, outer.evidence_count), (1, 0, 0))
        self.assertIsNone(current_budget())

    def test_concurrent_mcp_requests_do_not_charge_an_inherited_shared_context(self):
        def invoke(tight):
            inherited = current_budget()
            arguments = {"expression": "x^4-1"}
            if tight:
                arguments["limits"] = {"max_work": 1}
            response = call_tool("mathforge.factor", arguments)
            return response, current_budget() is inherited

        with computation(m.ComputationLimits(max_work=1), fresh=True) as outer:
            outer.tick()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(copy_context().run, invoke, tight) for tight in (True, False, True, False)]
                for tight, future in zip((True, False, True, False), futures):
                    response, restored = future.result()
                    self.assertEqual(response.is_error, tight, response.payload)
                    self.assertTrue(restored)
            self.assertEqual((outer.work, outer.node_count, outer.evidence_count), (1, 0, 0))
        self.assertIsNone(current_budget())


if __name__ == "__main__":
    unittest.main()
