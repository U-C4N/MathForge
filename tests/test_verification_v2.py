"""Independent source, evidence and result checks for every v2 operation family."""

from dataclasses import replace
import unittest
from unittest import mock

import mathforge as m
from mathforge import contracts as c
from mathforge import operations as op
from mathforge.matrices import Matrix, LinearSystem
from mathforge.polynomial import Polynomial
from mathforge.rational_function import rational_function


class V2VerificationTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000020")
        self.y = m.Symbol("y", uid="00000000-0000-4000-8000-000000000021")
        self.f = Polynomial((-1, 0, 1), self.x)
        self.g = Polynomial((-1, 1), self.x)

    def assertVerified(self, result):
        self.assertEqual(result.execution_status, c.ExecutionStatus.COMPLETED, result)
        report = m.verify(result)
        self.assertTrue(report.verified, report)
        restored = m.loads(m.dumps(result))
        self.assertTrue(m.verify(restored).verified, m.verify(restored))

    def forge(self, result, *, value=None, evidence=None, request=None):
        value = result.value if value is None else value
        request = result.request if request is None else request
        old_step = result.steps[0]
        evidence = old_step.outputs[1] if evidence is None else evidence
        step = replace(old_step, inputs=(request,), outputs=(value, evidence))
        return replace(result, value=value, request=request, steps=(step,))

    def test_polynomial_and_root_families(self):
        """VER-08/SER-19: all polynomial result families independently roundtrip."""
        results = (
            op.expand((self.x + 1)**3, self.x),
            op.poly_divmod(self.f, self.g), op.poly_gcd(self.f, self.g),
            op.poly_xgcd(self.f, self.g), op.square_free(self.f * self.g),
            op.real_roots(self.x**3 - 2, self.x), op.compare_real(m.sqrt(2), 1),
        )
        for result in results:
            with self.subTest(operation=result.request.operation if result.request else result.method):
                self.assertVerified(result)

    def test_calculus_and_rewrites(self):
        """CAL/VER: coefficients and source bindings are checked after transport."""
        expr = 3 * self.x**2 + 2 * self.x + 1
        rf = rational_function(self.x**2 - 1, self.x - 1, variable=self.x)
        results = (
            op.differentiate(expr, self.x), op.differentiate(expr, self.x, 2),
            op.differentiate(expr, self.x, 0), op.differentiate(rf, self.x),
            op.antiderivative(expr, self.x), op.integrate(expr, self.x, 0, 1),
            op.integrate(self.x**1024, self.x, 0, 1),
            op.evaluate(expr, {self.x: 2}), op.substitute(expr, {self.x: self.y}),
            op.simplify(expr), op.evaluate(rf, {self.x: 2}),
            op.substitute(rf, {self.x: self.y}), op.simplify(rf),
        )
        for result in results:
            with self.subTest(operation=result.request.operation if result.request else result.method):
                self.assertVerified(result)

    def test_matrix_families_and_solutions(self):
        """LIN/VER: row replay and independent identities cover matrix results."""
        A = Matrix(((1, 2), (3, 4)))
        system = LinearSystem(Matrix(((1, 1),)), (1,), (self.x, self.y))
        results = (op.rref(A), op.rank(A), op.det(A), op.inverse(A), op.nullspace(A),
                   op.solve_system(system),
                   op.solve_system((m.Eq(self.x + self.y, 1),), for_=(self.x, self.y)),
                   op.solve(m.Eq(self.x**3, 2), for_=self.x),
                   op.solve(c.Inequality(self.x**2, "ge", m.Rational(1)), for_=self.x))
        for result in results:
            with self.subTest(operation=result.request.operation if result.request else result.method):
                self.assertVerified(result)

    def test_source_alias_and_request_order_tampering(self):
        """VER-03/04: a valid answer cannot authenticate a different source."""
        result = op.differentiate(self.x**3, self.x)
        self.assertFalse(m.verify(replace(result, expression=self.x**4)).verified)
        arguments = dict(result.request.arguments)
        arguments["n"] = 2
        request = c.OperationRequest(result.request.operation, tuple(arguments.items()))
        self.assertFalse(m.verify(self.forge(result, request=request)).verified)
        other = op.differentiate(self.x**4, self.x)
        self.assertFalse(m.verify(replace(result, steps=other.steps)).verified)

    def test_wrong_values_and_evidence_do_not_verify(self):
        """VER-02/CAL-16/SER-18: stored success labels cannot authenticate a forged answer."""
        derivative = op.differentiate(self.x**3, self.x)
        forged = self.forge(derivative, value=m.Rational(0))
        self.assertFalse(m.verify(forged).verified)
        restored = m.loads(m.dumps(forged))
        self.assertTrue(any(check.status is c.CheckStatus.VERIFIED for check in restored.verification))
        self.assertFalse(m.verify(restored).verified)
        integral = op.integrate(self.x**2 + 1, self.x, 0, 1)
        self.assertFalse(m.verify(self.forge(integral, evidence=integral.steps[0].outputs[1][:-1])).verified)
        gcd = op.poly_gcd(self.f, self.g)
        zero = Polynomial((0,), self.x)
        self.assertFalse(m.verify(self.forge(gcd, evidence=(zero, zero))).verified)
        det = op.det(Matrix(((1, 2), (3, 4))))
        self.assertFalse(m.verify(self.forge(det, value=m.Rational(0))).verified)

    def test_dropped_rf_hole_is_rejected(self):
        """RF-22/CAL-11: matching values do not justify losing the domain."""
        source = rational_function(self.x**2 - 1, self.x - 1, variable=self.x)
        result = op.differentiate(source, self.x)
        unpunctured = rational_function(1, 1, variable=self.x)
        self.assertFalse(m.verify(self.forge(result, value=unpunctured)).verified)

    def test_root_certificates_bind_every_interval_and_chain_sign(self):
        """ROOT-21/VER-09: bind each root's proof and reject a consistently omitted root."""
        result = op.real_roots(self.x**4 - 5 * self.x**2 + 6, self.x)
        self.assertVerified(result)
        chain, intervals = result.steps[0].outputs[1]
        self.assertGreaterEqual(len(intervals), 2)
        swapped = (intervals[1], intervals[0], *intervals[2:])
        corruptions = (
            (chain, swapped),
            (chain, intervals[:-1]),
            ((*chain[:-1], -chain[-1]), intervals),
        )
        for evidence in corruptions:
            with self.subTest(evidence=evidence):
                self.assertFalse(m.verify(self.forge(result, evidence=evidence)).verified)
        records = result.value
        swapped_records = (replace(records[0], root=records[1].root),
                           replace(records[1], root=records[0].root), *records[2:])
        self.assertFalse(m.verify(self.forge(result, value=swapped_records)).verified)
        missing = self.forge(result, value=records[:-1], evidence=(chain, intervals[:-1]))
        self.assertFalse(m.verify(missing).verified)
        solved = op.solve(m.Eq(self.x**3, 2), for_=self.x)
        report = m.verify(replace(solved, steps=()))
        self.assertTrue(any(check.status is c.CheckStatus.FAILED for check in report.checks))

    def test_matrix_and_system_context_cannot_be_transplanted(self):
        """VER-03/04: source aliases bind matrix and normalized system proofs."""
        A, B = Matrix(((1, 2), (3, 4))), Matrix(((1, 0), (0, 1)))
        result = op.rref(A)
        self.assertFalse(m.verify(replace(result, expression=B)).verified)
        unrelated = op.rref(B)
        self.assertFalse(m.verify(replace(result, steps=unrelated.steps)).verified)
        system = LinearSystem(Matrix(((1, 1),)), (1,), (self.x, self.y))
        wrong = LinearSystem(Matrix(((1, 1),)), (2,), (self.x, self.y))
        result = op.solve_system(system)
        self.assertFalse(m.verify(replace(result, problem=wrong)).verified)

    def test_unknown_operation_and_rule_are_unknown(self):
        """VER-01/06/07: empty or unknown evidence never receives VERIFIED."""
        self.assertFalse(c.VerificationReport(()).verified)
        result = op.differentiate(self.x**2, self.x)
        unknown = c.OperationRequest("future_operation", ())
        report = m.verify(replace(result, request=unknown))
        self.assertFalse(report.verified)
        self.assertTrue(any(check.status is c.CheckStatus.UNKNOWN for check in report.checks))
        step = replace(result.steps[0], rule="future.rule")
        report = m.verify(replace(result, steps=(step,)))
        self.assertFalse(report.verified)
        self.assertTrue(any(check.status is c.CheckStatus.UNKNOWN for check in report.checks))

    def test_known_request_missing_extra_and_duplicate_arguments(self):
        """VER-05: argument schemas are checked even if all stored labels say verified."""
        result = op.differentiate(self.x**2, self.x)
        arguments = dict(result.request.arguments)
        missing = dict(arguments)
        del missing["n"]
        extra = dict(arguments, unrelated=1)
        for args in (missing, extra):
            request = c.OperationRequest("differentiate", tuple(args.items()))
            report = m.verify(self.forge(result, request=request))
            self.assertFalse(report.verified)
            self.assertTrue(any(check.status is c.CheckStatus.FAILED for check in report.checks))
        with self.assertRaises((TypeError, ValueError)):
            c.OperationRequest("differentiate", (("n", 1), ("n", 2)))

    def test_legacy_binding_record_remains_unknown_without_authoritative_source(self):
        """VER-14: legacy evaluation traces do not invent a missing source request."""
        step = c.Step("expression.evaluate", (self.x, ((self.x, 2),)), (m.Rational(2),),
                      relation=c.StepRelation.EVALUATION)
        result = c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.VALUE, m.Rational(2),
                          c.Exactness.EXACT, c.Completeness.COMPLETE,
                          method="expression.evaluate", steps=(step,), expression=self.x)
        restored = m.loads(m.dumps(result))
        self.assertIsNone(restored.request)
        report = m.verify(restored)
        self.assertFalse(report.verified)
        self.assertTrue(any(check.status is c.CheckStatus.UNKNOWN for check in report.checks))

    def test_producer_operations_are_not_called(self):
        """VER-08: producer dispatch can be disabled after the claims are created."""
        results = (op.differentiate(self.x**3, self.x),
                   op.integrate(self.x**2, self.x, 0, 1),
                   op.det(Matrix(((1, 2), (3, 4)))),
                   op.solve(m.Eq(self.x**3, 2), for_=self.x))
        patches = [mock.patch.object(op, name, side_effect=AssertionError("producer called"))
                   for name in ("differentiate", "integrate", "det", "solve")]
        for patch in patches:
            patch.start()
        try:
            for result in results:
                self.assertTrue(m.verify(result).verified, m.verify(result))
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_axes_tampering_is_rejected(self):
        """VER-02: exactness/completeness/context are part of the claim."""
        result = op.differentiate(self.x**2, self.x)
        for changes in ({"exactness": c.Exactness.APPROXIMATE},
                        {"completeness": c.Completeness.UNKNOWN},
                        {"error": c.ErrorInfo("fake", "fake")},
                        {"domain": None}, {"precision": 10}):
            with self.subTest(changes=changes):
                self.assertFalse(m.verify(replace(result, **changes)).verified)

    def test_correct_manual_claim_does_not_prove_an_operation_ran(self):
        """VER-16: mathematics and a manually recorded history have distinct meaning."""
        expression, value = self.x**2, 2 * self.x
        request = c.OperationRequest("differentiate", (("expression", expression),
                                                      ("variable", self.x), ("n", 1)))
        step = c.Step("v2.differentiate", (request,), (value, ()),
                      relation=c.StepRelation.EVALUATION)
        claim = c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.VALUE, value,
                         c.Exactness.EXACT, c.Completeness.COMPLETE,
                         method="manually_constructed", domain=m.Reals, steps=(step,),
                         expression=expression, for_=self.x, request=request)
        with mock.patch.object(op, "differentiate", side_effect=AssertionError("producer must not run")):
            self.assertTrue(m.verify(claim).verified)
            ws = m.Workspace()
            source_ref, result_ref = ws.put(expression), ws.put(claim)
            record = ws.record("operation_that_was_not_executed", {"source": source_ref}, {}, result_ref)
        self.assertEqual(record.operation, "operation_that_was_not_executed")
        self.assertTrue(m.verify(ws.get(result_ref)).verified)


if __name__ == "__main__":
    unittest.main()
