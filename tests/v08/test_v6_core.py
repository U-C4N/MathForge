"""JSON v6 core: the generic Unevaluated record and the v6 run/verify path, driven by a demo family."""

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
import importlib
import pkgutil
import types
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge import operations_v6, registry
from mathforge.contracts import (Check, CheckStatus, Completeness, ErrorInfo, ExecutionStatus, Outcome,
                                 VerificationLevel)
from mathforge.deferred import Reason, Unevaluated
from mathforge.errors import OperationError, SerializationError
from mathforge.limits import Budget

from . import demo_checker, demo_family
from .demo_family import DemoPair

UNKNOWN_OPERATION = Check("result.operation", CheckStatus.UNKNOWN, VerificationLevel.NONE, "verification_scope",
                          details="unknown v6 operation")


def statuses(report):
    """Check status by claim."""
    return {check.claim: check.status for check in report.checks}


def package_dataclasses():
    """Every dataclass that a module of the mathforge package defines; the server's __main__ module is left out,
    because importing it runs the server."""
    found = []
    for info in pkgutil.walk_packages(m.__path__, "mathforge."):
        if info.name.endswith(".__main__"):
            continue
        module = importlib.import_module(info.name)
        found.extend(cls for cls in vars(module).values()
                     if isinstance(cls, type) and is_dataclass(cls) and cls.__module__ == module.__name__)
    return found


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

    def test_checker_bugs_are_unknown_and_only_false_is_failed(self):
        """A checker that raises or answers no bool says nothing about the result; False means a wrong value."""
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_double", {"n": 21})
            for behaviour, status, details in (
                    ({"side_effect": RuntimeError("bug")}, CheckStatus.UNKNOWN, "The checker raised RuntimeError"),
                    ({"side_effect": KeyError("n")}, CheckStatus.UNKNOWN, "The checker raised KeyError"),
                    ({"return_value": None}, CheckStatus.UNKNOWN, "The checker returned no verdict: NoneType"),
                    ({"return_value": 1}, CheckStatus.UNKNOWN, "The checker returned no verdict: int"),
                    ({"return_value": False}, CheckStatus.FAILED, ""),
                    ({"return_value": True}, CheckStatus.VERIFIED, "")):
                with self.subTest(behaviour=behaviour), patch.object(demo_checker, "check_operation", **behaviour):
                    report = m.verify(r)
                    mathematics = next(check for check in report.checks if check.claim == "result.mathematics")
                    self.assertEqual((mathematics.status, mathematics.details), (status, details))
                    others = [check.status for check in report.checks if check is not mathematics]
                    self.assertEqual(others, [CheckStatus.VERIFIED] * 3)  # Source, contract and trace.
                    self.assertIs(report.verified, status is CheckStatus.VERIFIED)
            # A checker out of budget still makes the whole report a resources UNKNOWN, as before.
            with patch.object(demo_checker, "check_operation", side_effect=m.ResourceLimitError("out of budget")):
                self.assertEqual(statuses(m.verify(r)), {"result.resources": CheckStatus.UNKNOWN})

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

    def test_v6_requests_carry_their_format_version(self):
        """A request of format 6 names it, so a build that does not know the operation still keeps the format."""
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_double", {"n": 21})
            self.assertEqual(r.request.version, 6)  # Every request of run() carries it (plan ruling 3).
            document = m.to_data(r)
            self.assertEqual(document["data"]["request"]["version"], 6)
            self.assertEqual(document["data"]["steps"][0]["inputs"][0]["version"], 6)
            restored = m.from_data(document)
            self.assertEqual(restored.request.version, 6)  # Kept on decode...
            self.assertEqual(m.to_data(restored), document)  # ...so the document re-encodes to the same bytes.
        later = m.from_data(document)  # A build without the operation takes the format from the request itself.
        self.assertEqual((later.request.version, m.to_data(later)), (6, document))
        # Requests of v1-v5 operations never carry the field: their bytes are unchanged.
        gcd = m.to_data(m.operations.integer_gcd(84, 30))
        self.assertEqual(gcd["version"], 5)
        self.assertNotIn("version", gcd["data"]["request"])
        self.assertIsNone(m.from_data(gcd).request.version)
        older = m.to_data(m.OperationRequest("operation_of_a_later_build", (("value", 1),)))
        self.assertEqual((older["version"], set(older["data"])), (2, {"type", "operation", "arguments"}))

    def test_a_later_build_workspace_keeps_its_v6_requests(self):
        """A workspace holding a v6 failure result and a bare v6 request loads in a build without their operation.

        Each request carries its format version, so both objects re-encode to the bytes their references hash.
        """
        with registry._testing_families(demo_family):
            failure = operations_v6.run("demo_double", {"n": True})  # The int kind check rejects a bool.
            self.assertIs(failure.execution_status, ExecutionStatus.INVALID_INPUT)
            self.assertEqual(failure.steps, ())
            request = m.OperationRequest("demo_double", (("n", 21),))
            workspace = m.Workspace()
            references = (workspace.put(failure), workspace.put(request))
            text = m.dumps(workspace)
        loaded = m.loads(text)
        self.assertEqual(m.dumps(loaded), text)
        stored_failure, stored_request = (loaded.get(reference) for reference in references)
        self.assertEqual((stored_failure, stored_request), (failure, request))
        self.assertEqual((stored_failure.request.version, stored_request.version), (6, 6))
        self.assertEqual(m.to_data(stored_failure)["version"], 6)
        self.assertEqual(m.verify(stored_failure).checks, (UNKNOWN_OPERATION,))

    def test_request_versions_that_disagree_are_rejected(self):
        with registry._testing_families(demo_family):
            known = m.to_data(m.OperationRequest("demo_double", (("n", 21),)))
        later = m.to_data(m.OperationRequest("operation_of_a_later_build", (("value", 1),), version=6))
        tampered = []
        for version in (5, 7, "6", 6.0, True, None):
            document = deepcopy(later)
            document["data"]["version"] = version
            message = "requires format v7" if version == 7 else "version is an integer of at least 6"
            tampered.append((f"later build, {version!r}", document, message))
        unmarked = deepcopy(known)
        del unmarked["data"]["version"]
        tampered.append(("registered v6 operation without its version", unmarked, "^Noncanonical operation_request"))
        x = m.Symbol("x")
        for operation, arguments, version in (("integer_gcd", (("a", 84), ("b", 30)), 5),
                                              ("solve", (("problem", 1),), 2),
                                              ("factor", (("f", m.Polynomial((-2, 0, 1), x)),), 4)):
            document = m.to_data(m.OperationRequest(operation, arguments))
            self.assertEqual(document["version"], version)
            document["version"], document["data"]["version"] = 6, 6
            tampered.append((f"{operation} marked as v6", document, f"^This operation request belongs to format "
                                                                    f"v{version}, not v6$"))
        for label, document, message in tampered:
            with self.subTest(document=label), registry._testing_families(demo_family):
                with self.assertRaisesRegex(SerializationError, message):
                    m.from_data(document)
        with self.assertRaisesRegex(SerializationError, "^This operation request belongs to format v5, not v6$"):
            m.to_data(m.OperationRequest("integer_gcd", (("a", 84), ("b", 30)), version=6))
        for version, error in ((5, ValueError), (True, TypeError), ("6", TypeError), (6.0, TypeError)):
            with self.subTest(version=version), self.assertRaises(error):
                m.OperationRequest("operation_of_a_later_build", (), version=version)

    def test_request_version_is_limited_to_supported_formats(self):
        """A request names format 6, the one request format this build writes, or none. No route reaches to_data
        with a version 7: no request can carry it, and no document that names it loads."""
        for version in (7, 8, 10**6, 2**64):
            message = rf"^Unsupported operation request format version {version}: "
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, message):
                    m.OperationRequest("x", (), version=version)
                with self.assertRaisesRegex(ValueError, message):  # Nor can a request be moved on to it.
                    replace(m.OperationRequest("x", (), version=6), version=version)
        for request, written in ((m.OperationRequest("x", ()), None), (m.OperationRequest("x", (), version=6), 6)):
            with self.subTest(written=written):
                document = m.to_data(request)
                self.assertEqual(document["data"].get("version"), written)
                self.assertEqual(document["version"], 2 if written is None else 6)
        document = m.to_data(m.OperationRequest("x", (), version=6))
        for header in (6, 7):  # A hand-written 7 does not load in a document of format 6, and there is no format 7.
            tampered = deepcopy(document)
            tampered["version"], tampered["data"]["version"] = header, 7
            with self.subTest(header=header), self.assertRaises(SerializationError):
                m.from_data(tampered)

    def test_the_version_marker_leaves_equality_hash_repr_and_budgets_alone(self):
        plain = m.OperationRequest("demo_double", (("n", 1),))
        marked = m.OperationRequest("demo_double", (("n", 1),), version=6)
        self.assertEqual((plain, hash(plain), repr(plain)), (marked, hash(marked), repr(marked)))
        budget = Budget()
        budget.inspect(marked)  # The request, operation, arguments, one pair, its name and value: not the version.
        self.assertEqual(budget.node_count, 6)
        with registry._testing_families(demo_family):
            r = operations_v6.run("demo_defer", {"n": 1})
            restored = m.loads(m.dumps(r))
        self.assertEqual((r.request.version, restored.request.version, restored.deferred.request.version),
                         (6, 6, 6))
        self.assertIsNone(r.deferred.request.version)  # The family's own request names no version; it is equal.
        self.assertEqual(restored, r)
        self.assertEqual(restored.deferred.request, restored.request)
        self.assertEqual(restored.deferred.request, r.request)

    def test_budget_skips_only_marked_fields(self):
        """A field outside comparison is still content, and still bounded, unless it is marked
        metadata={"budget": "skip"}; in the whole package that is the request's version and nothing else."""
        marked = {(cls.__qualname__, f.name) for cls in package_dataclasses() for f in fields(cls)
                  if f.metadata.get("budget") == "skip"}
        self.assertEqual(marked, {("OperationRequest", "version")})

        @dataclass(frozen=True)
        class Probe:
            compared: int
            unmarked: int = field(default=0, compare=False)
            marked: int = field(default=0, compare=False, metadata={"budget": "skip"})

        budget = Budget()
        budget.inspect(Probe(1, 2, 3))
        self.assertEqual(budget.node_count, 3)  # The probe, compared and unmarked; the marked field is not walked.
        small = m.ComputationLimits(max_integer_bits=64)
        with self.assertRaises(m.ResourceLimitError):  # An unmarked field outside comparison is bounded...
            Budget(small).inspect(Probe(1, 2**100))
        Budget(small).inspect(Probe(1, 0, 2**100))  # ...a marked one is not.

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
            # Argument names and the int kind are checked before the producer runs: a bool is no exact integer.
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
        before = registry.v6_families()
        self.assertNotIn("demo_double", registry.specs())  # The tables exist before the block...
        with self.assertRaises(KeyError), registry._testing_families(demo_family):
            self.assertEqual(registry._tables.cache_info().currsize, 0)  # ...and entering clears them.
            self.assertIn("demo_double", registry.specs())
            self.assertTrue(registry.record_module(DemoPair.__module__))
            self.assertIs(registry.record_types(6)["demo_pair"], DemoPair)
            raise KeyError("leave the block early")
        self.assertEqual(registry.v6_families(), before)
        self.assertNotIn("demo_double", registry.specs())
        self.assertFalse(registry.record_module(DemoPair.__module__))
        self.assertNotIn("demo_pair", registry.record_types(6))
        self.assertEqual({tag: registry.record_types(6)[tag] for tag in ("reason", "unevaluated")},
                         {"reason": Reason, "unevaluated": Unevaluated})

    def test_testing_families_must_be_imported_modules(self):
        """The build imports families by name, so a module outside sys.modules is refused up front."""
        before = registry.v6_families()
        with self.assertRaisesRegex(ValueError, "^A testing family must be an imported module: v08_stray$"):
            with registry._testing_families(types.ModuleType("v08_stray")):
                self.fail("the block must not run")
        self.assertEqual(registry.v6_families(), before)


if __name__ == "__main__":
    unittest.main()
