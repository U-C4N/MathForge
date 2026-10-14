"""Source-linked domains, symbol identities and canonical v5 transport."""
from copy import deepcopy
from dataclasses import replace
import unittest

import mathforge as m
from mathforge import functions as f
from mathforge.errors import SerializationError
from mathforge.limits import computation, ComputationLimits, ResourceLimitError
from mathforge.mcp.boundary import call_tool


class DomainTransportTests(unittest.TestCase):
    def test_legacy_equations_do_not_accept_partial_function_nodes(self):
        x = m.symbol("x")
        for value in (f.log(x), f.real_function(f.log(x), (x,))):
            with self.assertRaises(TypeError):
                m.Eq(value, 0)
            with self.assertRaises(TypeError):
                m.Inequality(value, "gt", 0)

    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000078")

    def function(self, text):
        return m.parse_function(text, symbols={"x": self.x})

    def assertRejectedClaim(self, result):
        decoded = m.loads(m.dumps(result))
        self.assertEqual(decoded, result)
        report = m.verify(decoded)
        self.assertTrue(any(check.status is m.CheckStatus.FAILED for check in report.checks), report)
        response = call_tool("mathforge.verify", {"result": m.to_data(decoded)})
        self.assertFalse(response.is_error, response.payload)
        remote_report = m.from_data(response.payload["result"])
        self.assertTrue(any(check.status is m.CheckStatus.FAILED for check in remote_report.checks), remote_report)

    def test_canonical_records_roundtrip_with_domains_and_exact_symbols(self):
        x = self.x
        source = self.function("0*log(x)")
        primitive = m.operations.function_antiderivative(
            self.function("(x-1)/((x-1)*(x*x*x+x+1))"), x).value
        values = (source.expression, source.domain, source, m.bernoulli(m.Rational(1, 3)),
                  m.complex_algebraic(1, 2), m.lu(m.Matrix(((0, 1), (2, 3)))),
                  m.least_squares(m.Matrix(((1, 1),)), (2,)), primitive,
                  m.operations.function_integrate(self.function("exp(x)"), x, 0, 1))
        for value in values:
            with self.subTest(type=type(value).__name__):
                document = m.to_data(value)
                decoded = m.from_data(document)
                self.assertEqual(decoded, value)
                self.assertEqual(m.to_data(decoded), document)
                self.assertEqual(m.dumps(decoded), m.dumps(value))
                self.assertEqual(document["version"], 5)
        restored = m.loads(m.dumps(source))
        self.assertIs(restored.contains({x: -1}), m.Truth.FALSE)
        self.assertIs(restored.contains({x: 1}), m.Truth.TRUE)
        self.assertEqual(restored.variables[0].uid, x.uid)
        self.assertIs(primitive.domain.contains({x: 1}), m.Truth.FALSE)

    def test_decoder_rejects_dropped_intrinsic_domain_and_noncanonical_records(self):
        document = m.to_data(self.function("log(x)"))
        forged = deepcopy(document)
        forged["data"]["domain"] = m.to_data(f.TRUE)["data"]
        variants = [forged]
        forged = deepcopy(document)
        forged["data"]["unexpected"] = {"type": "none"}
        variants.append(forged)
        forged = deepcopy(document)
        forged["data"]["type"] = "v5_executable"
        variants.append(forged)
        forged = deepcopy(document)
        forged["version"] = 4
        variants.append(forged)
        distribution = m.to_data(m.bernoulli(m.Rational(1, 2)))
        distribution["data"]["outcomes"]["items"].reverse()
        variants.append(distribution)
        for forged in variants:
            with self.subTest(document=forged):
                with self.assertRaises(SerializationError):
                    m.from_data(forged)

    def test_decoded_result_source_and_same_name_different_identity_are_checked(self):
        result = m.operations.function_differentiate(self.function("x*x"), self.x)
        self.assertTrue(m.verify(m.loads(m.dumps(result))).verified)
        self.assertRejectedClaim(replace(result, expression=self.function("x*x*x")))
        other_x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000077")
        self.assertRejectedClaim(replace(result, for_=other_x))
        request = m.OperationRequest(result.request.operation,
                    tuple((name, self.function("x*x*x") if name == "expression" else value)
                          for name, value in result.request.arguments))
        self.assertRejectedClaim(replace(result, request=request))

    def test_conflicting_metadata_for_one_uuid_is_rejected_on_both_boundaries(self):
        result = m.operations.function_differentiate(self.function("x*x"), self.x)
        conflict = m.Symbol("different_name", uid=self.x.uid)
        with self.assertRaises(SerializationError):
            m.dumps(replace(result, for_=conflict))
        document = m.to_data(result)
        document["data"]["for_"]["name"] = "different_name"
        with self.assertRaises(SerializationError):
            m.from_data(document)
        response = call_tool("mathforge.verify", {"result": document})
        self.assertTrue(response.is_error)
        self.assertIsNone(response.payload["result"])

    def test_derivative_domain_cannot_be_removed_even_when_value_is_zero(self):
        result = m.operations.function_differentiate(self.function("0*log(x)"), self.x)
        unrestricted = f.real_function(0, (self.x,))
        step = result.steps[0]
        trace, = step.outputs[1]
        bad_trace = replace(trace, derivatives=(unrestricted,))
        forged = replace(result, value=unrestricted,
                         steps=(replace(step, outputs=(unrestricted, (bad_trace,))),))
        self.assertRejectedClaim(forged)

    def test_formal_rational_fallback_must_preserve_source_cancellation_hole(self):
        source = self.function("(x-1)/(x-1)") * self.function("1/(x*x*x+x+1)")
        result = m.operations.function_antiderivative(source, self.x)
        self.assertTrue(m.verify(result).verified)
        primitive = result.value
        natural_domain = f.real_function(primitive.source).domain
        forged_primitive = replace(primitive, domain=natural_domain)
        self.assertIs(forged_primitive.domain.contains({self.x: 1}), m.Truth.TRUE)
        step = replace(result.steps[0], outputs=(forged_primitive, result.steps[0].outputs[1]))
        self.assertRejectedClaim(replace(result, value=forged_primitive, steps=(step,)))

    def test_deferred_integral_content_is_linked_to_source_kind_and_bounds(self):
        source = self.function("exp(x*x)")
        result = m.operations.function_antiderivative(source, self.x)
        report = m.verify(result)
        self.assertFalse(report.verified)
        self.assertFalse(any(check.status is m.CheckStatus.FAILED for check in report.checks))
        replacements = (
            m.UnevaluatedIntegral(self.function("sin(x)"), self.x),
            m.UnevaluatedLimit(source, self.x, f.as_real_expr(7), "left"),
            m.UnevaluatedIntegral(source, self.x, m.Rational(5), m.Rational(6)),
        )
        for deferred in replacements:
            with self.subTest(deferred=deferred):
                self.assertRejectedClaim(replace(result, deferred=deferred))
        definite = m.operations.function_integrate(source, self.x, 0, m.sqrt(2))
        self.assertIsInstance(definite.deferred, m.UnevaluatedIntegral)
        self.assertFalse(any(check.status is m.CheckStatus.FAILED
                             for check in m.verify(m.loads(m.dumps(definite))).checks))
        self.assertRejectedClaim(replace(definite, deferred=replace(definite.deferred, lower=m.Rational(1))))
        self.assertRejectedClaim(replace(definite, deferred=replace(definite.deferred, lower=None, upper=None)))

    def test_deferred_limit_content_is_linked_to_point_and_side(self):
        result = m.operations.function_limit(self.function("sin(1/x)"), self.x, 0, side="right")
        self.assertIsInstance(result.deferred, m.UnevaluatedLimit)
        self.assertFalse(any(check.status is m.CheckStatus.FAILED
                             for check in m.verify(m.loads(m.dumps(result))).checks))
        self.assertRejectedClaim(replace(result, deferred=replace(result.deferred, point=f.as_real_expr(2))))
        self.assertRejectedClaim(replace(result, deferred=replace(result.deferred, side="left")))
        self.assertRejectedClaim(replace(result, deferred=m.UnevaluatedIntegral(result.deferred.expression, self.x)))

    def test_unrelated_operation_and_approximation_metadata_cannot_claim_valid_deferral(self):
        deferred_result = m.operations.function_antiderivative(self.function("exp(x*x)"), self.x)
        self.assertRejectedClaim(replace(deferred_result, precision=20))
        self.assertRejectedClaim(replace(deferred_result, tolerance=m.Rational(1, 100)))
        self.assertRejectedClaim(replace(deferred_result, error_bound=m.Rational(1, 100)))
        integer_result = m.operations.integer_gcd(6, 4)
        forged = replace(integer_result, value=None, steps=(), verification=(),
                         outcome=m.Outcome.NO_CONCLUSION, completeness=m.Completeness.UNKNOWN,
                         exactness=m.Exactness.NOT_APPLICABLE, deferred=deferred_result.deferred)
        self.assertRejectedClaim(forged)

    def test_log_root_sum_record_validation_respects_active_node_budget(self):
        from mathforge.rational_integration import LogRootSum
        zero, one = m.Rational(0), m.Rational(1)
        q = (zero, one)
        h = ((zero,),)*100+((one,),)
        with self.assertRaises(ResourceLimitError):
            with computation(ComputationLimits(max_work=2, max_nodes=1)):
                LogRootSum(q, h)


if __name__ == "__main__":
    unittest.main()
