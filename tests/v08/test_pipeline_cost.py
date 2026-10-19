"""The structured pipeline's fixed cost: records built once from pieces inspected once, with v0.7's bytes.

tests/fixtures/v8/structured_v07.json holds, for every v0.7 operation that v0.8 does not supersede, the dumps of one
structured call made on the v0.7 tree by tools/snapshot_structured.py; the current code must give the same text.
"""

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import operations, operations_v5, registry
from mathforge.limits import Budget
from mathforge.registry_v07 import LEGACY_CATALOG, V07_SPECS

REPOSITORY = Path(__file__).resolve().parents[2]
GOLDEN = REPOSITORY / "tests" / "fixtures" / "v8" / "structured_v07.json"
TOOL = REPOSITORY / "tools" / "snapshot_structured.py"
# The modules of the structured pipeline: run(), the records' builders and the verifier.
PIPELINE = frozenset({"mathforge.operations", "mathforge.operations_v5", "mathforge.dispatch_v5", "mathforge.contracts",
                      "mathforge.verification", "mathforge.verification_v5"})


def load_tool():
    spec = importlib.util.spec_from_file_location("snapshot_structured", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pipeline_inspections(call):
    """Run call() with Budget.inspect recording each inspection that a pipeline module makes, as (budget, value,
    evidence); returns call()'s return value and the records."""
    recorded = []
    inspect = Budget.inspect

    def recording(self, value, *, evidence=False):
        if sys._getframe(1).f_globals.get("__name__") in PIPELINE:
            recorded.append((self, value, evidence))
        return inspect(self, value, evidence=evidence)

    with patch.object(Budget, "inspect", recording):
        return call(), recorded


class PipelineCostTest(unittest.TestCase):
    def test_structured_bytes_unchanged(self):
        stored = json.loads(GOLDEN.read_text(encoding="utf-8"))
        tool = load_tool()
        expected = ({spec.name for spec in V07_SPECS} | {entry.name for entry in LEGACY_CATALOG}) - tool.EXCLUDED
        self.assertEqual(set(stored), expected)
        arguments = tool.arguments()
        self.assertEqual(set(arguments), expected)
        for operation in sorted(stored):
            with self.subTest(operation=operation):
                self.assertEqual(tool.call(operation, arguments[operation]), stored[operation])

    def test_to_data_of_old_values_builds_no_registry(self):
        x = mf.Symbol("x", uid="00000000-0000-0000-0000-000000000001")
        values = (mf.Rational(1, 2), x + 1, mf.Polynomial((1, 2), x), mf.Matrix(((1, 2), (3, 4))), mf.Eq(x, 1),
                  mf.FiniteSet((1, 2)), mf.RealAlgebraicRoot((-2, 0, 1), 1))
        for value in values:
            with self.subTest(value=type(value).__name__):
                registry._clear_caches()
                self.addCleanup(registry._clear_caches)
                mf.to_data(value)
                self.assertEqual(registry._tables.cache_info().currsize, 0)
        with self.subTest("rejected value"):
            registry._clear_caches()
            with self.assertRaises(mf.SerializationError):
                mf.to_data(object())
            self.assertEqual(registry._tables.cache_info().currsize, 0)

    def test_each_piece_inspected_once(self):
        """run() inspects the arguments, the value and the evidence once each in its budget; the verifier inspects
        the Result once in a fresh budget; building the records inspects nothing."""
        result, recorded = pipeline_inspections(lambda: operations_v5.run("integer_gcd", {"a": 84, "b": 30}))
        self.assertEqual(result.unwrap(), 6)
        producer = recorded[0][0]
        self.assertEqual([(value, evidence) for budget, value, evidence in recorded if budget is producer],
                         [({"a": 84, "b": 30}, False), (6, False), (result.steps[0].outputs[1], True)])
        verifier = [(budget, value, evidence) for budget, value, evidence in recorded if budget is not producer]
        self.assertEqual(len(verifier), 1)
        self.assertIs(verifier[0][1], result)
        self.assertTrue(verifier[0][2])

    def test_legacy_run_inspects_each_piece_once(self):
        """_run inspects the arguments once and the value and the evidence once each; the records add nothing to
        what the verifier inspects."""
        matrix = mf.Matrix(((1, 2), (3, 4)))
        result, recorded = pipeline_inspections(lambda: operations.det(matrix))
        self.assertEqual(result.unwrap(), mf.Rational(-2))
        own = [(value, evidence) for _, value, evidence in recorded]
        self.assertEqual(own, [({"A": matrix}, False), (mf.Rational(-2), True), ((), True)])
        self.assertEqual(result.steps[0].audit, (result.verification[-1],))

    def test_trusted_records_equal_public_ones(self):
        """The records that the pipelines build equal those that the public constructors build from their fields."""
        from dataclasses import fields, replace
        x = mf.Symbol("x", uid="00000000-0000-0000-0000-000000000001")
        for result in (operations_v5.run("integer_gcd", {"a": 84, "b": 30}),
                       operations.det(mf.Matrix(((1, 2), (3, 4)))),
                       operations.solve(mf.Eq(x ** 2, 4), for_=x), operations.solve(mf.Eq(x ** 3 - x, 0), for_=x)):
            rebuilt = replace(result)
            self.assertEqual(rebuilt, result)
            self.assertEqual(hash(rebuilt), hash(result))
            step = result.steps[0]
            self.assertEqual(replace(step), step)
            self.assertEqual(replace(result.request), result.request)
            for check in result.verification:
                self.assertEqual(replace(check), check)
            self.assertEqual({field.name for field in fields(result)}, set(vars(result)))
            self.assertEqual(mf.loads(mf.dumps(result)), result)


if __name__ == "__main__":
    unittest.main()
