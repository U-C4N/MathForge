"""JSON v6 core: the generic Unevaluated record and the v6 run/verify path, driven by a demo family."""

from dataclasses import replace
import types
import unittest

import mathforge as m
from mathforge import operations_v6, registry
from mathforge.contracts import (Check, CheckStatus, Completeness, ErrorInfo, ExecutionStatus, Outcome,
                                 VerificationLevel)
from mathforge.deferred import Reason, Unevaluated
from mathforge.errors import OperationError, SerializationError

from . import demo_family
from .demo_family import DemoPair

UNKNOWN_OPERATION = Check("result.operation", CheckStatus.UNKNOWN, VerificationLevel.NONE, "verification_scope",
                          details="unknown v6 operation")


def statuses(report):
    """Check status by claim."""
    return {check.claim: check.status for check in report.checks}


class V6CoreTests(unittest.TestCase):
    def test_v6_result_roundtrips_and_verifies(self):
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_double", {"n": 21})
            self.assertEqual(r.value, 42)
            self.assertEqual([check.claim for check in r.verification],
                             ["result.source", "result.contract", "result.trace", "result.mathematics"])
            self.assertEqual(m.to_data(r)["version"], 6)
            self.assertEqual(m.loads(m.dumps(r)), r)
            self.assertTrue(m.verify(m.loads(m.dumps(r))).verified)

    def test_v6_record_tag(self):
        with registry._testing_families(demo_family):
            document = m.to_data(DemoPair(1, 2))
            self.assertEqual(document["data"]["type"], "v6_demo_pair")
            data = {"type": "v6_demo_pair", "a": {"type": "integer", "value": "1"},
                    "b": {"type": "integer", "value": "2"}}
            self.assertEqual(document, {"format": "mathforge", "version": 6, "kind": "object", "data": data})
            self.assertEqual(m.from_data(document), DemoPair(1, 2))

    def test_v6_tag_rejected_below_version_6(self):
        with registry._testing_families(demo_family):
            document = m.to_data(DemoPair(1, 2))
            document["version"] = 5
            with self.assertRaisesRegex(SerializationError, "^This mathematical type requires format v6$"):
                m.from_data(document)
            step = m.to_data(m.Step("v6.demo_double", (), ()))
            self.assertEqual(step["version"], 6)
            step["version"] = 5
            with self.assertRaisesRegex(SerializationError, "^This evidence rule requires format v6$"):
                m.from_data(step)
            request = m.to_data(m.OperationRequest("demo_double", (("n", 21),)))
            self.assertEqual(request["version"], 6)
            request["version"] = 5
            with self.assertRaisesRegex(SerializationError, "^This operation request requires format v6$"):
                m.from_data(request)
            with self.assertRaises(SerializationError):  # A v6 request must name exactly its spec's arguments.
                m.to_data(m.OperationRequest("demo_double", (("m", 21),)))
            unknown = {"format": "mathforge", "version": 6, "kind": "object", "data": {"type": "v6_nope"}}
            with self.assertRaisesRegex(SerializationError, "^Unknown v6 mathematical type$"):
                m.from_data(unknown)

    def test_unknown_v6_operation_is_unknown_not_failed(self):
        """Review Focus 4: the result of a v6 operation this build does not know is UNKNOWN, never FAILED."""
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_double", {"n": 21})
            self.assertTrue(m.verify(r).verified)
            text = m.dumps(r)
        for label, result in (("in memory", r), ("decoded", m.loads(text))):
            with self.subTest(result=label):
                report = m.verify(result)
                self.assertFalse(report.verified)
                self.assertIn(UNKNOWN_OPERATION, report.checks)
                self.assertNotIn(CheckStatus.FAILED, [check.status for check in report.checks])

    def test_deferral_is_no_conclusion(self):
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_defer", {"n": 1})
            self.assertIs(r.execution_status, ExecutionStatus.COMPLETED)
            self.assertIs(r.outcome, Outcome.NO_CONCLUSION)
            self.assertIs(r.completeness, Completeness.UNKNOWN)
            self.assertIsNone(r.value)
            self.assertEqual(r.steps, ())
            self.assertEqual(r.deferred.request, r.request)
            self.assertEqual(r.deferred, Unevaluated("sum", r.request, Reason("no_applicable_rule")))
            report = statuses(m.verify(r))
            self.assertIs(report["result.unevaluated"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.UNKNOWN)
            with self.assertRaises(OperationError):
                r.unwrap()
            deferred = m.to_data(r)["data"]["deferred"]
            self.assertEqual((deferred["type"], deferred["reason"]["type"], deferred["candidate"]),
                             ("v6_unevaluated", "v6_reason", {"type": "none"}))
            self.assertEqual(m.loads(m.dumps(r)), r)
            # A top-level Unevaluated must carry its own result's request.
            with self.assertRaises(ValueError):
                replace(r, request=m.OperationRequest("demo_defer", (("n", 2),)))

    def test_producer_crash_is_internal_error(self):
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_crash", {"n": 1})
            self.assertIs(r.execution_status, ExecutionStatus.ERROR)
            self.assertEqual(r.error.code, "internal_error")
            self.assertEqual(r.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_value_shape_failures_are_internal_errors(self):
        """A return that cannot become a Result is the producer's bug, not the caller's invalid input."""
        with registry._testing_families(demo_family):
            for operation in ("demo_float", "demo_misdefer"):  # A float value; a deferral of another request.
                with self.subTest(operation=operation):
                    r = operations_v6.run(operation, {"n": 1})
                    self.assertIs(r.execution_status, ExecutionStatus.ERROR)
                    self.assertEqual(r.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_scope_and_budget_errors_of_the_producer_keep_their_codes(self):
        with registry._testing_families(demo_family):
            for operation, status, error in (
                    ("demo_unsupported", ExecutionStatus.UNSUPPORTED,
                     ErrorInfo("unsupported_operation", "demo_unsupported applies to no request")),
                    ("demo_exhausted", ExecutionStatus.ERROR,
                     ErrorInfo("resource_limit_exceeded", "demo_exhausted always exceeds the budget"))):
                with self.subTest(operation=operation):
                    r = operations_v6.run(operation, {"n": 1})
                    self.assertIs(r.execution_status, status)
                    self.assertEqual(r.error, error)  # The producer's own message: raised inside produce.
            # A value over the budget, found once produce has returned, is a budget error too.
            r = operations_v6.run("demo_double", {"n": 2**63}, limits=m.ComputationLimits(max_integer_bits=64))
            self.assertIs(r.execution_status, ExecutionStatus.ERROR)
            self.assertEqual((r.error.code, r.request.operation), ("resource_limit_exceeded", "demo_double"))

    def test_context_rules_match_the_source_check(self):
        x = m.Symbol("x")
        A = m.Matrix(((1, 2), (3, 4), (5, 6)))
        with registry._testing_families(demo_family):
            echo = operations_v6.run("demo_echo", {"expression": x**2 + 1, "variable": x})
            self.assertEqual((echo.value, echo.expression, echo.for_), (x**2 + 1, x**2 + 1, x))
            self.assertIs(echo.domain, m.Reals)
            rows = operations_v6.run("demo_rows", {"A": A})
            self.assertEqual((rows.value, rows.expression, rows.for_, rows.domain), (3, A, None, None))
            for r in (echo, rows):
                with self.subTest(operation=r.method):
                    self.assertTrue(m.verify(r).verified)
                    self.assertEqual(m.loads(m.dumps(r)), r)

    def test_failures_keep_the_variable_context_that_verification_expects(self):
        """A failure keeps the variable's own domain and assumptions (operations._failure); verify agrees."""
        t = m.Symbol("t", domain=m.Integers, assumptions=(m.Assumption("positive", m.Truth.TRUE),))
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_echo", {"expression": t + 1, "variable": t})
            self.assertIs(r.execution_status, ExecutionStatus.INVALID_INPUT)
            self.assertEqual((r.expression, r.for_, r.domain, r.assumptions), (t + 1, t, m.Integers, t.assumptions))
            report = m.verify(r)
            self.assertIs(statuses(report)["result.source"], CheckStatus.VERIFIED)
            self.assertNotIn(CheckStatus.FAILED, [check.status for check in report.checks])

    def test_invalid_arguments_are_invalid_input(self):
        with registry._testing_families(demo_family):
            # Argument names are checked before the producer runs; the producer rejects a bool itself.
            for arguments in ({"m": 1}, {}, {"n": 1, "m": 1}, {"n": True}):
                with self.subTest(arguments=arguments):
                    r = operations_v6.run("demo_double", arguments)
                    self.assertIs(r.execution_status, ExecutionStatus.INVALID_INPUT)
                    self.assertEqual(r.error.code, "invalid_input")

    def test_operations_outside_the_v6_registry_are_unsupported(self):
        with registry._testing_families(demo_family):
            for operation, arguments in (("demo_nope", {"n": 1}), ("integer_gcd", {"a": 84, "b": 30})):
                with self.subTest(operation=operation):
                    r = operations_v6.run(operation, arguments)
                    self.assertIs(r.execution_status, ExecutionStatus.UNSUPPORTED)

    def test_v5_results_keep_version_5(self):
        with registry._testing_families(demo_family):
            self.assertEqual(m.to_data(m.operations.integer_gcd(84, 30))["version"], 5)

    def test_testing_families_restore_the_registry(self):
        before = registry.V6_FAMILIES
        self.assertNotIn("demo_double", registry.specs())  # The tables exist before the block...
        with self.assertRaises(KeyError), registry._testing_families(demo_family):
            self.assertEqual(registry._tables.cache_info().currsize, 0)  # ...and entering clears them.
            self.assertIn("demo_double", registry.specs())
            self.assertTrue(registry.record_module(DemoPair.__module__))
            self.assertIs(registry.record_types(6)["demo_pair"], DemoPair)
            raise KeyError("leave the block early")
        self.assertEqual(registry.V6_FAMILIES, before)
        self.assertNotIn("demo_double", registry.specs())
        self.assertFalse(registry.record_module(DemoPair.__module__))
        self.assertNotIn("demo_pair", registry.record_types(6))
        self.assertEqual({tag: registry.record_types(6)[tag] for tag in ("reason", "unevaluated")},
                         {"reason": Reason, "unevaluated": Unevaluated})

    def test_testing_families_must_be_imported_modules(self):
        """The build imports families by name, so a module outside sys.modules is refused up front."""
        before = registry.V6_FAMILIES
        with self.assertRaisesRegex(ValueError, "^A testing family must be an imported module: v08_stray$"):
            with registry._testing_families(types.ModuleType("v08_stray")):
                self.fail("the block must not run")
        self.assertEqual(registry.V6_FAMILIES, before)


if __name__ == "__main__":
    unittest.main()
