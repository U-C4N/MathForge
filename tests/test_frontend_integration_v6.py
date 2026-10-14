"""The complete v0.6 public surface, provenance, strict wire v4 and presentation."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mathforge as m


class ExactAlgebraIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000061")
        self.alpha = m.RealAlgebraicRoot((-2, 0, 0, 1), 0)

    def results(self):
        x, a = self.x, self.alpha
        p = m.Polynomial.from_expression(x**4 + 4, x)
        A = m.Matrix(((0, 2), (1, 0)))
        f = m.rational_function(x+1, x*x-1, variable=x)
        return (
            m.operations.factor(p), m.operations.resultant(p, m.Polynomial((-1, 1), x)),
            m.operations.discriminant(p), m.operations.algebraic(m.sqrt(2)+m.sqrt(3)),
            m.operations.algebraic_div(a, a), m.operations.algebraic_pow(a, -3),
            m.operations.minimal_polynomial(a, x), m.operations.evaluate_algebraic(x**3, {x: a}),
            m.operations.apart(f), m.operations.extrema_values(x**3-6*x, x),
            m.operations.charpoly(A, x), m.operations.real_eigenvalues(A),
            m.operations.limit(x, x, a), m.operations.integrate(x, x, 0, m.sqrt(2)),
            m.operations.compare_real(m.sqrt(2)+m.sqrt(3), 3),
        )

    def test_all_operations_have_checked_roundtrippable_rendered_results(self):
        for result in self.results():
            with self.subTest(operation=result.method):
                self.assertIs(result.execution_status, m.ExecutionStatus.COMPLETED, result.error)
                self.assertTrue(m.verify(result).verified)
                document = m.to_data(result)
                self.assertEqual(document["version"], 4)
                restored = m.loads(m.dumps(result))
                self.assertEqual(restored, result)
                self.assertTrue(m.verify(restored).verified)
                self.assertTrue(m.render_text(restored))
                self.assertTrue(m.to_latex(restored))
                forged = replace(result, value=m.Rational(999))
                self.assertFalse(m.verify(forged).verified)
                self.assertFalse(m.verify(replace(result, method="other")).verified)

    def test_source_linkage_and_producer_independence(self):
        result = m.operations.algebraic_pow(self.alpha, 3)
        request = replace(result.request, arguments=(("base", self.alpha), ("n", 2)))
        forged = replace(result, request=request,
                         steps=(replace(result.steps[0], inputs=(request,)),))
        self.assertFalse(m.verify(forged).verified)
        with patch("mathforge.exact_dispatch.produce", side_effect=AssertionError("producer")), \
             patch("mathforge.operations._run", side_effect=AssertionError("SDK producer")):
            self.assertTrue(m.verify(result).verified)

    def test_old_and_new_mixed_values_never_downgrade_version(self):
        old = m.operations.limit(self.x, self.x, 0)
        new = m.operations.factor(m.Polynomial((1, 0, 0, 0, 1), self.x))
        for values in ((old, new), (new, old)):
            self.assertEqual(m.to_data(values)["version"], 4)
            self.assertEqual(m.loads(m.dumps(values)), values)
            workspace = m.Workspace()
            refs = [workspace.put(v) for v in values]
            document = m.to_data(workspace)
            self.assertEqual(document["version"], 4)
            restored = m.from_data(document)
            self.assertEqual([restored.put(v) for v in values], refs)
        self.assertEqual(m.to_data(old)["version"], 3)
        self.assertEqual(m.to_data(self.alpha)["version"], 2)
        self.assertEqual(m.to_data(m.Rational(1))["version"], 1)

    def test_v4_rejects_downgrades_extra_fields_and_noncanonical_records(self):
        result = m.operations.factor(m.Polynomial((4, 0, 0, 0, 1), self.x))
        document = m.to_data(result)
        for version in (1, 2, 3):
            forged = deepcopy(document)
            forged["version"] = version
            with self.assertRaises(m.SerializationError):
                m.from_data(forged)
        forged = m.to_data(result.value)
        forged["data"]["extra"] = True
        with self.assertRaises(m.SerializationError):
            m.from_data(forged)
        finite = m.to_data(m.LimitValue("finite", self.alpha))
        self.assertEqual(finite["version"], 4)
        finite["version"] = 3
        with self.assertRaises(m.SerializationError):
            m.from_data(finite)
        source = Path(__file__).parent / "fixtures/v5/serialization_v3.py"
        spec = importlib.util.spec_from_file_location("mathforge._frozen_v3", source)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        with self.assertRaises(m.SerializationError):
            reader.loads(m.dumps(result))

    def test_failures_are_serializable_and_have_no_mathematical_claim(self):
        x = self.x
        results = (m.operations.factor(m.Polynomial((0,), x)), m.operations.factor(1),
                   m.operations.algebraic(x), m.operations.algebraic_div(1, 0),
                   m.operations.algebraic_pow(0, -1), m.operations.charpoly(m.Matrix(((1, 2),)), x),
                   m.operations.evaluate_algebraic(x, {}),
                   m.operations.factor(m.Polynomial((4, 0, 0, 0, 1), x), limits=m.ComputationLimits(max_work=1)))
        for result in results:
            with self.subTest(error=result.error):
                self.assertIsNot(result.execution_status, m.ExecutionStatus.COMPLETED)
                self.assertIsNone(result.value)
                self.assertIsNot(result.outcome, m.Outcome.NO_SOLUTION)
                self.assertEqual(m.loads(m.dumps(result)), result)
                self.assertFalse(m.verify(result).verified)

    def test_new_documents_match_packaged_v4_schema(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("Development-only jsonschema is unavailable")
        schema = json.loads((Path(m.__file__).parent / "schemas/mathforge-v4.schema.json").read_text())
        validator = Draft202012Validator(schema)
        for result in self.results():
            validator.validate(m.to_data(result))


if __name__ == "__main__":
    unittest.main()
