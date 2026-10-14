"""Public v0.4 provenance, dispatch, verification status and failure contracts."""

from dataclasses import replace
from unittest import TestCase, main, mock

import mathforge as mf
# Import every patch target before mocking any dependency aliases.
import mathforge.function_analysis
import mathforge.rational_limits
import mathforge.real_sets
from mathforge.limits import ResourceLimitError
from mathforge.verification import verify_step


_UNCHANGED = object()


class AnalysisVerificationV4Tests(TestCase):
    def setUp(self):
        self.x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000044")
        self.function = mf.rational_function((self.x - 1) * (self.x**2 + 2),
                                             self.x * (self.x - 1), variable=self.x)

    def results(self):
        return (mf.operations.analyze_function(self.function, self.x),
                mf.operations.limit(self.function, self.x, 0))

    def assertVerified(self, result):
        self.assertIs(result.execution_status, mf.ExecutionStatus.COMPLETED, result.error)
        report = mf.verify(result)
        self.assertTrue(report.verified, report)
        self.assertIs(result.exactness, mf.Exactness.EXACT)
        self.assertIs(result.completeness, mf.Completeness.COMPLETE)
        self.assertTrue(all(check.level is mf.VerificationLevel.ALGORITHMIC_CHECK for check in report.checks))
        self.assertIs(verify_step(result.steps[0]).status, mf.CheckStatus.VERIFIED)
        return result

    def forge(self, result, *, value=_UNCHANGED, evidence=_UNCHANGED, request=_UNCHANGED):
        value = result.value if value is _UNCHANGED else value
        request = result.request if request is _UNCHANGED else request
        evidence = result.steps[0].outputs[1] if evidence is _UNCHANGED else evidence
        step = replace(result.steps[0], inputs=(request,), outputs=(value, evidence))
        return replace(result, value=value, request=request, steps=(step,))

    def assertFailed(self, result):
        report = mf.verify(result)
        self.assertFalse(report.verified)
        self.assertTrue(any(check.status is mf.CheckStatus.FAILED for check in report.checks), report)

    def test_both_operations_have_checked_v3_results_and_public_value_wrappers(self):
        analysis, limit = self.results()
        for result in (analysis, limit):
            self.assertVerified(result)
            self.assertEqual(result.steps[0].rule, "v3." + result.request.operation)
            self.assertEqual(mf.to_data(result)["version"], 3)
            self.assertVerified(mf.loads(mf.dumps(result)))
        self.assertEqual(mf.analyze_function(self.function, self.x), analysis.value)
        self.assertEqual(mf.limit(self.function, self.x, 0), limit.value)

    def test_nonexistent_two_sided_limit_is_an_exact_complete_value(self):
        result = self.assertVerified(mf.operations.limit(self.function, self.x, 0))
        self.assertEqual(result.value, mf.LimitValue("does_not_exist"))
        self.assertIs(result.outcome, mf.Outcome.VALUE)
        self.assertIsNone(result.solution_set)
        self.assertEqual(mf.limit(self.function, self.x, 0, side="left").kind, "negative_infinity")
        self.assertEqual(mf.limit(self.function, self.x, 0, side="right").kind, "positive_infinity")

    def test_source_aliases_method_and_symbol_metadata_are_checked(self):
        renamed_same_id = mf.Symbol("different_name", uid=self.x.uid)
        for result in self.results():
            mutations = (
                replace(result, expression=self.function + 1),
                replace(result, for_=mf.symbol("x")),
                replace(result, for_=renamed_same_id),
                replace(result, domain=mf.Integers),
                replace(result, problem=mf.Eq(self.x, 0)),
                replace(result, assumptions=(mf.Assumption("positive", mf.Truth.TRUE),)),
                replace(result, method="different_operation"),
            )
            for changed in mutations:
                with self.subTest(operation=result.method, change=changed):
                    self.assertFailed(changed)

    def test_complete_contract_cannot_be_relabelled_as_partial_or_approximate(self):
        for result in self.results():
            for changes in ({"outcome": mf.Outcome.PARTIAL},
                            {"exactness": mf.Exactness.APPROXIMATE},
                            {"completeness": mf.Completeness.UNKNOWN},
                            {"completeness": mf.Completeness.PARTIAL},
                            {"precision": 12}, {"tolerance": mf.Rational(0)},
                            {"error_bound": mf.Rational(0)},
                            {"error": mf.ErrorInfo("invented", "not a completed result")}):
                with self.subTest(operation=result.method, changes=changes):
                    self.assertFailed(replace(result, **changes))

    def test_step_source_linkage_shape_and_relation_are_checked(self):
        for result in self.results():
            step = result.steps[0]
            wrong_steps = (
                (), (step, step),
                (replace(step, inputs=()),),
                (replace(step, outputs=(result.value,)),),
                (replace(step, rule="v2." + result.method),),
                (replace(step, relation=mf.StepRelation.APPROXIMATION),),
                (replace(step, preconditions=(mf.Condition(mf.Rational(0), "eq", mf.Rational(0)),)),),
            )
            for changed in wrong_steps:
                with self.subTest(operation=result.method, steps=changed):
                    self.assertFailed(replace(result, steps=changed))
            request = replace(result.request, arguments=tuple(reversed(result.request.arguments)))
            self.assertEqual(request, result.request)
            self.assertVerified(self.forge(result, request=request))

    def test_malformed_request_contracts_fail_without_exceptions(self):
        for result in self.results():
            pairs = result.request.arguments
            malformed = (
                mf.OperationRequest(result.method, pairs[:-1]),
                mf.OperationRequest(result.method, (*pairs, ("extra", mf.Rational(0)))),
                mf.OperationRequest(result.method, tuple((key, mf.Rational(0) if key == "variable" else value)
                                                         for key, value in pairs)),
            )
            for request in malformed:
                self.assertFailed(self.forge(result, request=request))
            self.assertFailed(self.forge(result, value=mf.Rational(0)))
            self.assertFailed(self.forge(result, evidence=()))

    def test_limit_point_side_and_expression_remain_bound_when_answer_is_equal(self):
        zero = mf.rational_function(0, self.x, variable=self.x)
        result = self.assertVerified(mf.operations.limit(zero, self.x, 0))
        for key, replacement in (("point", mf.Rational(1)), ("side", "left"),
                                 ("expression", mf.rational_function(0, 1, variable=self.x))):
            arguments = dict(result.request.arguments)
            arguments[key] = replacement
            request = replace(result.request, arguments=tuple(arguments.items()))
            changed = self.forge(result, request=request)
            if key == "expression":
                changed = replace(changed, expression=replacement)
            self.assertFailed(changed)

    def test_stored_labels_cannot_certify_forged_claims_even_after_serialization(self):
        analysis, limit = self.results()
        forged = (
            self.forge(analysis, value=replace(analysis.value, stationary_set=mf.IntervalSet(()))),
            self.forge(limit, value=mf.LimitValue("finite", mf.Rational(0))),
        )
        for changed in forged:
            restored = mf.loads(mf.dumps(changed))
            self.assertTrue(all(check.status is mf.CheckStatus.VERIFIED for check in restored.verification))
            self.assertTrue(all(check.status is mf.CheckStatus.VERIFIED for check in restored.steps[0].audit))
            self.assertFailed(restored)

    def test_unknown_operation_or_missing_source_is_not_silently_verified(self):
        for result in self.results():
            unknown = replace(result.request, operation="future_operation")
            for changed in (self.forge(result, request=unknown), replace(result, request=None)):
                report = mf.verify(changed)
                self.assertFalse(report.verified)
                self.assertTrue(any(check.status is mf.CheckStatus.UNKNOWN for check in report.checks))

    def test_public_verification_never_dispatches_producers(self):
        analysis, limit = self.results()
        forbidden = AssertionError("Verification dispatched a producer")
        with (mock.patch("mathforge.operations.analyze_function", side_effect=forbidden),
              mock.patch("mathforge.operations.limit", side_effect=forbidden),
              mock.patch("mathforge.function_analysis.analyze_function", side_effect=forbidden),
              mock.patch("mathforge.function_analysis._assemble_analysis", side_effect=forbidden),
              mock.patch("mathforge.function_analysis._producer_regions", side_effect=forbidden),
              mock.patch("mathforge.function_analysis.real_roots", side_effect=forbidden),
              mock.patch("mathforge.root_isolation.real_roots", side_effect=forbidden),
              mock.patch("mathforge.real_sets.solve_real_set", side_effect=forbidden),
              mock.patch("mathforge.rational_limits.limit_value", side_effect=forbidden),
              mock.patch.object(mf.RationalFunction, "derivative", side_effect=forbidden)):
            self.assertVerified(analysis)
            self.assertVerified(limit)
        # A subsequent normal solve proves patched aliases did not leak.
        self.assertTrue(mf.verify(mf.solve(mf.Inequality(self.x**2, "gt", 1), for_=self.x)).verified)

    def test_verifier_resource_exhaustion_is_unknown_not_mathematically_false(self):
        for result in self.results():
            for limits in (mf.ComputationLimits(max_work=1), mf.ComputationLimits(max_evidence_items=1)):
                report = mf.verify(result, limits=limits)
                self.assertFalse(report.verified)
                self.assertTrue(any(check.status is mf.CheckStatus.UNKNOWN for check in report.checks))
                self.assertFalse(any(check.status is mf.CheckStatus.FAILED for check in report.checks))
        with mock.patch("mathforge.function_analysis.check_analysis", side_effect=ResourceLimitError("test budget")):
            result = mf.operations.analyze_function(self.x, self.x)
        self.assertIs(result.execution_status, mf.ExecutionStatus.ERROR)
        self.assertEqual(result.error.code, "resource_limit_exceeded")
        self.assertIsNone(result.value)
        self.assertIs(result.completeness, mf.Completeness.NOT_APPLICABLE)

    def test_failed_self_check_never_returns_a_completed_mathematical_value(self):
        with mock.patch("mathforge.function_analysis.check_analysis", return_value=False):
            result = mf.operations.analyze_function(self.x, self.x)
        self.assertIs(result.execution_status, mf.ExecutionStatus.ERROR)
        self.assertEqual(result.error.code, "execution_error")
        self.assertIs(result.outcome, mf.Outcome.NOT_APPLICABLE)
        self.assertIsNone(result.value)
        report = mf.verify(result)
        self.assertFalse(report.verified)
        self.assertTrue(any(check.status is mf.CheckStatus.UNKNOWN for check in report.checks))

    def test_unsupported_and_invalid_calls_remain_serializable_without_claims(self):
        n = mf.symbol("n", domain=mf.Integers)
        positive = mf.symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))
        calls = (
            (lambda: mf.operations.analyze_function(0, n), mf.ExecutionStatus.UNSUPPORTED),
            (lambda: mf.operations.analyze_function(0, positive), mf.ExecutionStatus.UNSUPPORTED),
            (lambda: mf.operations.analyze_function(self.x + mf.symbol("y"), self.x), mf.ExecutionStatus.UNSUPPORTED),
            (lambda: mf.operations.analyze_function(self.x + mf.sqrt(2), self.x), mf.ExecutionStatus.UNSUPPORTED),
            (lambda: mf.operations.limit(0, self.x, mf.symbol("unbound")), mf.ExecutionStatus.UNSUPPORTED),
            (lambda: mf.operations.limit(0, self.x, "+inf", side="left"), mf.ExecutionStatus.INVALID_INPUT),
            (lambda: mf.operations.limit(0, self.x, True), mf.ExecutionStatus.INVALID_INPUT),
            (lambda: mf.operations.analyze_function(0, self.x, limits=mf.ComputationLimits(max_work=1)),
             mf.ExecutionStatus.ERROR),
        )
        for call, status in calls:
            result = call()
            with self.subTest(result=result):
                self.assertIs(result.execution_status, status)
                self.assertIs(result.outcome, mf.Outcome.NOT_APPLICABLE)
                self.assertIs(result.exactness, mf.Exactness.NOT_APPLICABLE)
                self.assertIs(result.completeness, mf.Completeness.NOT_APPLICABLE)
                self.assertIsNone(result.value)
                self.assertIsNone(result.solution_set)
                restored = mf.loads(mf.dumps(result))
                report = mf.verify(restored)
                self.assertFalse(report.verified)
                self.assertTrue(any(check.status is mf.CheckStatus.UNKNOWN for check in report.checks))


if __name__ == "__main__":
    main()
