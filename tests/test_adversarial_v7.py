"""Source linkage, preserved domains and bounded hostile mathematical claims."""
from copy import copy, deepcopy
from dataclasses import replace
import unittest

import mathforge as m
from mathforge import functions as f


class AdversarialV7Tests(unittest.TestCase):
    def test_changed_sources_and_values_are_not_authenticated_by_stored_audit(self):
        result = m.operations.integer_gcd(30, 21)
        request = m.OperationRequest("integer_gcd", (("a", 31), ("b", 21)))
        forged = replace(result, request=request,
                         steps=(replace(result.steps[0], inputs=(request,)),))
        self.assertFalse(m.verify(forged).verified)
        for changed in (replace(result, value=7), replace(result, method="integer_lcm"),
                        replace(result, steps=()), replace(result, exactness=m.Exactness.APPROXIMATE)):
            self.assertFalse(m.verify(changed).verified)

    def test_filled_source_domain_is_rejected_even_when_derivative_is_zero(self):
        x = m.symbol("x")
        source = m.real_function(0*f.log(x), (x,))
        result = m.operations.function_differentiate(source, x)
        self.assertTrue(m.verify(result).verified)
        self.assertIs(result.value.contains({x: -1}), m.Truth.FALSE)
        filled = m.RealFunction(result.value.expression, (x,), f.TRUE)
        trace = result.steps[0].outputs[1][0]
        evidence = (replace(trace, derivatives=(filled,)),)
        forged = replace(result, value=filled,
                         steps=(replace(result.steps[0], outputs=(filled, evidence)),))
        self.assertFalse(m.verify(forged).verified)

    def test_wrong_prime_proof_and_distribution_claims_are_rejected(self):
        result = m.operations.is_prime(65537)
        value, evidence = result.steps[0].outputs
        certificate = evidence[0]
        bad_evidence = (certificate[:-1]+((1,),),)
        forged = replace(result, steps=(replace(result.steps[0], outputs=(value, bad_evidence)),))
        self.assertFalse(m.verify(forged).verified)
        distribution = m.operations.binomial_distribution(4, m.Rational(1, 2))
        other = m.bernoulli(m.Rational(1, 2))
        forged = replace(distribution, value=other,
                         steps=(replace(distribution.steps[0], outputs=(other, distribution.steps[0].outputs[1])),))
        self.assertFalse(m.verify(forged).verified)

    def test_unknown_request_names_and_argument_smuggling_are_rejected(self):
        document = m.to_data(m.operations.factorial(6))
        for bad_name in ("__import__", "pathlib.Path.unlink", "unknown_operation"):
            bad = deepcopy(document)
            bad["data"]["request"]["operation"] = bad_name
            # A decodable old generic request still cannot verify as the v5 claim.
            try:
                result = m.from_data(bad)
            except m.SerializationError:
                continue
            self.assertFalse(m.verify(result).verified)
        request = m.OperationRequest("factorial", (("n", 6), ("extra", 1)))
        with self.assertRaises(m.SerializationError):
            m.dumps(request)

    def test_verifier_and_decoder_resource_failures_are_not_mathematical_answers(self):
        result = m.operations.factor_integer(360)
        report = m.verify(result, limits=m.ComputationLimits(max_work=1))
        self.assertFalse(report.verified)
        failed = m.operations.factorial(100, limits=m.ComputationLimits(max_integer_bits=16))
        self.assertIs(failed.execution_status, m.ExecutionStatus.ERROR)
        self.assertIsNone(failed.value)
        self.assertIsNot(failed.outcome, m.Outcome.NO_SOLUTION)
        self.assertEqual(m.loads(m.dumps(failed)), failed)
        with self.assertRaises(m.SerializationError):
            m.loads(m.dumps(result), decode_limits=m.DecodeLimits(max_nodes=3))

    def test_deferred_claim_cannot_borrow_a_verified_audit_or_gain_an_answer(self):
        x = m.symbol("x")
        source = m.real_function(f.sin(f.as_real_expr(x)**2), (x,))
        deferred = m.operations.function_antiderivative(source, x)
        audit = m.operations.integer_gcd(30, 21).verification
        for changes in ({"verification": audit},
                        {"value": m.Rational(0), "outcome": m.Outcome.VALUE},
                        {"deferred": None},
                        {"exactness": m.Exactness.EXACT, "completeness": m.Completeness.COMPLETE}):
            # Bypass constructor guards to exercise the independent verifier's
            # trust boundary against malformed in-memory objects as well.
            forged = copy(deferred)
            for name, value in changes.items():
                object.__setattr__(forged, name, value)
            self.assertFalse(m.verify(forged).verified)

    def test_same_uuid_cannot_change_source_identity(self):
        x = m.symbol("x")
        result = m.operations.function_differentiate(m.real_function(f.sin(x), (x,)), x)
        other = m.Symbol("different_name", uid=x.uid)
        request = replace(result.request, arguments=tuple((k, other if k == "variable" else v)
                                                          for k, v in result.request.arguments))
        forged = replace(result, request=request,
                         steps=(replace(result.steps[0], inputs=(request,)),))
        self.assertFalse(m.verify(forged).verified)


if __name__ == "__main__":
    unittest.main()
