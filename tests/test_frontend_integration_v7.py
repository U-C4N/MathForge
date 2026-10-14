"""Every v0.7 operation crosses the SDK, verifier, strict codec and renderers."""
from copy import deepcopy
from importlib import resources
import json
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge import functions as f
from mathforge.dispatch_v5 import operation_arguments, modules


def samples():
    x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000701")
    y = m.Symbol("y", uid="00000000-0000-4000-8000-000000000702")
    z = m.ComplexAlgebraic(3, 4)
    distribution = m.FiniteDistribution((0, 1, 2), (m.Rational(1, 4), m.Rational(1, 2), m.Rational(1, 4)))
    A = m.Matrix(((1, 2), (2, 4), (1, 1)))
    square = m.Matrix(((0, 2), (1, 3)))
    single = m.real_function(f.sin(x), (x,))
    multi = m.real_function(f.sin(x)*f.cos(y), (x, y))
    rf = m.rational_function(1, x, variable=x)
    cases = {
        "integer_gcd": (30, 21), "integer_lcm": (30, 21), "integer_xgcd": (30, 21),
        "mod_inverse": (-3, 11), "crt": (((2, 6), (8, 9)),), "is_prime": (65537,),
        "factor_integer": (-360,), "divisors": (24,), "totient": (36,),
        "factorial": (12,), "binomial": (20, 8), "multinomial": ((2, 3, 1),),
        "permutations_count": (10, 4), "fibonacci": (100,), "catalan": (12,), "stirling": (8, 3),
        "complex_algebraic": (m.sqrt(2), 1), "complex_add": (z, 2), "complex_sub": (z, 1),
        "complex_mul": (z, z), "complex_div": (z, 2), "complex_conjugate": (z,),
        "complex_real": (z,), "complex_imag": (z,), "complex_norm_squared": (z,),
        "complex_pow": (z, -2), "complex_sqrt": (z,), "quadratic_roots": (1, 0, 2),
        "bernoulli": (m.Rational(1, 3),), "binomial_distribution": (5, m.Rational(1, 3)),
        "discrete_uniform": ((1, 2, 3),), "hypergeometric": (10, 4, 3),
        "probability": (distribution, (1, 2)), "condition": (distribution, (1, 2)),
        "cdf": (distribution, 1), "moment": (distribution, 3), "expectation": (distribution,),
        "distribution_variance": (distribution,), "mean": ((1, 2, 4),), "median": ((1, 2, 4, 8),),
        "variance": ((1, 2, 4), 1), "stddev": ((1, 2, 4), 0),
        "covariance": ((1, 2, 4), (2, 3, 7), 1), "trace": (square,),
        "rowspace": (A,), "columnspace": (A,), "lu": (square,), "least_squares": (A, (1, 3, 2)),
        "function_differentiate": (single, x), "function_partial_diff": (multi, x),
        "function_gradient": (multi, (x, y)), "function_jacobian": ((multi, multi*2), (x, y)),
        "function_hessian": (multi, (x, y)), "function_taylor": (m.real_function(f.exp(x), (x,)), x, 0, 4),
        "function_evaluate": (single, {x: 0}), "function_antiderivative": (single, x),
        "function_integrate": (single, x, 0, 1),
        "function_limit": (m.real_function(f.sin(x)/f.as_real_expr(x), (x,)), x, 0),
        "rational_antiderivative": (rf, x), "rational_integrate": (rf, x, 1, 2),
    }
    return cases


def results():
    return {name: getattr(m.operations, "taylor" if name == "function_taylor" else name)(*args)
            for name, args in samples().items()}


class FrontendV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = results()

    def test_entire_registry_has_roundtrippable_rendered_verified_results(self):
        self.assertEqual(set(self.results), set(operation_arguments()))
        for name, result in self.results.items():
            with self.subTest(operation=name):
                self.assertIs(result.execution_status, m.ExecutionStatus.COMPLETED, result.error)
                self.assertIs(result.outcome, m.Outcome.VALUE)
                self.assertTrue(m.verify(result).verified)
                self.assertEqual(m.to_data(result)["version"], 5)
                restored = m.loads(m.dumps(result))
                self.assertEqual(restored, result)
                self.assertTrue(m.verify(restored).verified)
                self.assertTrue(m.render_text(restored))
                self.assertTrue(m.to_latex(restored))

    def test_verifier_is_independent_of_every_registered_producer(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch("mathforge.dispatch_v5.produce", side_effect=AssertionError("dispatch producer")))
            stack.enter_context(patch("mathforge.operations_v5.run", side_effect=AssertionError("SDK producer")))
            for module in modules():
                if hasattr(module, "produce"):
                    stack.enter_context(patch.object(module, "produce", side_effect=AssertionError(module.__name__)))
            for name, result in self.results.items():
                with self.subTest(operation=name):
                    self.assertTrue(m.verify(result).verified)

    def test_packaged_v5_schema_accepts_results_and_rejects_corrupt_shapes(self):
        try:
            from jsonschema import Draft202012Validator, ValidationError
        except ImportError:
            self.skipTest("Development-only jsonschema is unavailable")
        schema = json.loads(resources.files("mathforge").joinpath("schemas/mathforge-v5.schema.json").read_text())
        validator = Draft202012Validator(schema)
        for result in self.results.values():
            validator.validate(m.to_data(result))
        corrupt = m.to_data(self.results["factor_integer"])
        corrupt["data"]["unexpected"] = True
        with self.assertRaises(ValidationError):
            validator.validate(corrupt)

    def test_v5_documents_reject_downgrade_and_extra_fields(self):
        for name in ("integer_gcd", "complex_sqrt", "function_differentiate", "rational_antiderivative"):
            document = m.to_data(self.results[name])
            for version in (1, 2, 3, 4):
                bad = deepcopy(document)
                bad["version"] = version
                with self.subTest(name=name, version=version), self.assertRaises(m.SerializationError):
                    m.from_data(bad)
            bad = deepcopy(document)
            bad["data"]["arbitrary"] = True
            with self.assertRaises(m.SerializationError):
                m.from_data(bad)

    def test_mixed_workspace_retains_legacy_references(self):
        old = m.Rational(2, 3)
        workspace = m.Workspace()
        original = workspace.put(old)
        for result in self.results.values():
            workspace.put(result)
        document = m.to_data(workspace)
        self.assertEqual(document["version"], 5)
        restored = m.from_data(document)
        self.assertEqual(restored.put(old), original)
        self.assertEqual(m.to_data(old)["version"], 1)

    def test_deferred_requests_never_masquerade_as_verified_values(self):
        x = m.symbol("x")
        expression = m.real_function(f.sin(f.as_real_expr(x)**2), (x,))
        result = m.operations.function_antiderivative(expression, x)
        self.assertIs(result.execution_status, m.ExecutionStatus.COMPLETED)
        self.assertIs(result.outcome, m.Outcome.NO_CONCLUSION)
        self.assertIsNotNone(result.deferred)
        self.assertIsNone(result.value)
        self.assertFalse(m.verify(result).verified)
        with self.assertRaises(m.OperationError):
            result.unwrap()
        restored = m.loads(m.dumps(result))
        self.assertEqual(restored, result)
        self.assertFalse(m.verify(restored).verified)
        self.assertTrue(m.render_text(restored))
        self.assertTrue(m.to_latex(restored))


if __name__ == "__main__":
    unittest.main()
