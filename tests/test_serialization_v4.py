"""Strict v3 analysis transport with stable historical bytes and references."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
from importlib import resources
from pathlib import Path
import unittest

import mathforge as mf
from mathforge.analysis_types import ExcludedPoint, FunctionAnalysis, LimitValue, LocalExtremum, MonotonicRegion


class AnalysisSerializationTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000041")
        self.zero = mf.Rational(0)
        self.all_reals = mf.IntervalSet((mf.Interval(None, None),))
        self.empty = mf.IntervalSet(())
        self.at_zero = mf.IntervalSet((mf.Interval(self.zero, self.zero, True, True),))
        self.function = mf.rational_function(self.x**2, 1, variable=self.x)
        self.analysis = FunctionAnalysis(
            self.function, self.all_reals, (), self.at_zero,
            mf.IntervalSet((mf.Interval(None, self.zero), mf.Interval(self.zero, None))),
            self.empty, mf.rational_function(2*self.x, 1, variable=self.x), self.at_zero,
            (MonotonicRegion(mf.Interval(None, self.zero, False, True), "decreasing"),
             MonotonicRegion(mf.Interval(self.zero, None, True, False), "increasing")),
            (LocalExtremum(self.zero, "minimum"),),
        )

    def request(self, point=None):
        return mf.OperationRequest("limit", (
            ("expression", self.function), ("variable", self.x),
            ("point", self.zero if point is None else point), ("side", "both"),
        ))

    def assert_roundtrip(self, value):
        payload = mf.dumps(value)
        self.assertEqual(mf.to_data(value)["version"], 3)
        restored = mf.loads(payload)
        self.assertEqual(restored, value)
        self.assertEqual(mf.dumps(restored), payload)
        return mf.to_data(value)

    def test_all_records_and_requests(self):
        root = mf.RealAlgebraicRoot((-2, 0, 1), 1)
        values = (
            LimitValue("finite", mf.Rational(2, 3)), LimitValue("positive_infinity"),
            LimitValue("negative_infinity"), LimitValue("does_not_exist"),
            ExcludedPoint(root, "hole"), ExcludedPoint(self.zero, "pole"),
            LocalExtremum(root, "maximum"), self.analysis.strict_extrema[0],
            *self.analysis.monotonic_regions,
            MonotonicRegion(mf.Interval(None, None), "constant"), self.analysis,
            self.request(), self.request("+inf"), self.request("-inf"),
            mf.OperationRequest("analyze_function", (("expression", self.function), ("variable", self.x))),
        )
        for value in values:
            with self.subTest(type=type(value).__name__, value=value):
                self.assert_roundtrip(value)

    def test_public_result_envelopes(self):
        for result in (mf.operations.limit(self.function, self.x, 0),
                       mf.operations.limit(self.function, self.x, "+inf"),
                       mf.operations.analyze_function(self.function, self.x)):
            with self.subTest(operation=result.request.operation):
                self.assertEqual(result.execution_status, mf.ExecutionStatus.COMPLETED)
                self.assert_roundtrip(result)
                self.assertTrue(mf.verify(mf.loads(mf.dumps(result))).verified)

    def test_nested_promotion_never_downgrades(self):
        limit = LimitValue("finite", self.zero)
        old_result = mf.Result(mf.ExecutionStatus.COMPLETED, mf.Outcome.VALUE, mf.Rational(2))
        old_request = mf.OperationRequest("fixture", (("x", self.x),))
        old_poly_result = replace(old_result, expression=mf.Polynomial((1, 2), self.x))
        new_result_with_old_request = replace(old_result, value=limit, request=old_request)
        for graph in ((limit, mf.Matrix(((1,),))), (limit, old_poly_result),
                      (limit, old_result, old_request), (new_result_with_old_request,),
                      (mf.Eq(self.function, 0), limit),
                      (old_result, limit, old_poly_result)):
            with self.subTest(graph=graph):
                self.assert_roundtrip(graph)
        restored = mf.loads(mf.dumps((limit, old_result)))[1]
        self.assertEqual(mf.dumps(restored), mf.dumps(old_result))
        self.assertEqual(mf.to_data(restored)["version"], 1)

    def test_mixed_workspace_preserves_historical_refs_and_history(self):
        fixture = Path(__file__).parent / "fixtures/v1/legacy_workspace.json"
        workspace = mf.loads(fixture.read_text(encoding="utf-8"))
        old_matrix = mf.Matrix(((1, 2), (3, 4)))
        matrix_ref = workspace.put(old_matrix)
        originals = {ref: mf.dumps(value) for ref, value in workspace.objects.items()}
        analysis_ref = workspace.put(self.analysis)
        limit_ref = workspace.put(LimitValue("finite", self.zero))
        workspace.record("fixture", {"source": matrix_ref}, {"version": 3}, analysis_ref)
        payload = mf.dumps(workspace)
        self.assertEqual(mf.to_data(workspace)["version"], 3)
        restored = mf.loads(payload)
        self.assertEqual(mf.dumps(restored), payload)
        self.assertEqual(restored.get(analysis_ref), self.analysis)
        self.assertEqual(restored.get(limit_ref), LimitValue("finite", self.zero))
        for ref, original in originals.items():
            self.assertEqual(mf.dumps(restored.get(ref)), original)
            self.assertEqual(restored.put(restored.get(ref)), ref)

    def test_new_tags_and_requests_require_v3(self):
        for value in (self.analysis, LimitValue("positive_infinity"), self.request()):
            data = mf.to_data(value)
            for version in (1, 2, 999, True):
                with self.subTest(value=value, version=version):
                    bad = deepcopy(data)
                    bad["version"] = version
                    with self.assertRaises(mf.SerializationError):
                        mf.from_data(bad)

    def test_historical_v2_reader_rejects_new_documents(self):
        source = Path(__file__).parent / "fixtures/v2/serialization_v2.py"
        spec = importlib.util.spec_from_file_location("mathforge._analysis_old_reader", source)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        for value in (self.analysis, self.request(), LimitValue("does_not_exist")):
            with self.subTest(value=value), self.assertRaises(mf.SerializationError):
                reader.loads(mf.dumps(value))

    def test_legacy_wire_and_schema_are_frozen(self):
        fixture_dir = Path(__file__).parent / "fixtures"
        for path in (fixture_dir / "v1/legacy_workspace.json", fixture_dir / "v2/objects.json"):
            payload = path.read_text(encoding="utf-8")
            # The v1 workspace fixture is pretty-printed; compare canonical bytes.
            restored = mf.loads(payload)
            expected_version = 1 if path.parent.name == "v1" else 2
            self.assertEqual(mf.to_data(restored)["version"], expected_version)
            if expected_version == 2:
                self.assertEqual(mf.dumps(restored), payload)
        schema = resources.files("mathforge").joinpath("schemas/mathforge-v2.schema.json")
        # Git checkouts may use CRLF; the historical schema content is frozen.
        self.assertEqual(hashlib.sha256(schema.read_text(encoding="utf-8").encode()).hexdigest(),
                         "d88e582879bcccd37c5d0b11ae0a94933700ec53a569168e491c9af439200873")

    def test_strict_shapes_and_canonical_values(self):
        cases = []
        for value, updates in (
            (LimitValue("finite", self.zero), {"value": None}),
            (LimitValue("positive_infinity"), {"value": mf.to_data(self.zero)["data"]}),
            (LimitValue("finite", self.zero), {"kind": "unknown"}),
            (ExcludedPoint(self.zero, "pole"), {"kind": "infinite"}),
            (LocalExtremum(self.zero, "maximum"), {"point": {"type": "integer", "value": "0"}}),
            (self.analysis.monotonic_regions[0], {"behavior": "flat"}),
            (self.analysis, {"extra": True}),
            (self.analysis, {"zero_set": {"type": "text", "value": "zero"}}),
            (self.analysis, {"excluded_points": [{"type": "untrusted.custom_type"}]}),
        ):
            bad = mf.to_data(value)
            bad["data"].update(updates)
            cases.append(bad)
        bad = mf.to_data(LimitValue("finite", mf.Rational(1, 2)))
        bad["data"]["value"].update(numerator="2", denominator="4")
        cases.append(bad)
        bad = mf.to_data(self.analysis)
        del bad["data"]["derivative"]
        cases.append(bad)
        for bad in cases:
            with self.subTest(data=bad), self.assertRaises(mf.SerializationError):
                mf.from_data(bad)

    def test_new_requests_reject_invalid_signature_order_and_types(self):
        data = mf.to_data(self.request())
        mutations = []
        bad = deepcopy(data)
        bad["data"]["arguments"].reverse()
        mutations.append(bad)
        bad = deepcopy(data)
        bad["data"]["arguments"].pop()
        mutations.append(bad)
        for name, value in (("point", {"type": "text", "value": "inf"}),
                            ("point", {"type": "integer", "value": "1"}),
                            ("side", {"type": "text", "value": "up"}),
                            ("variable", mf.to_data(self.zero)["data"])):
            bad = deepcopy(data)
            next(item for item in bad["data"]["arguments"] if item["name"] == name)["value"] = value
            mutations.append(bad)
        for bad in mutations:
            with self.subTest(data=bad), self.assertRaises(mf.SerializationError):
                mf.from_data(bad)

    def test_decode_and_computation_budgets(self):
        payload = mf.dumps(self.analysis)
        for limits in (mf.DecodeLimits(max_bytes=10), mf.DecodeLimits(max_depth=2),
                       mf.DecodeLimits(max_nodes=3)):
            with self.subTest(limits=limits), self.assertRaises(mf.SerializationError) as caught:
                mf.loads(payload, decode_limits=limits)
            self.assertEqual(caught.exception.code, "resource_limit_exceeded")
        with self.assertRaises(mf.SerializationError) as caught:
            mf.loads(mf.dumps(LimitValue("finite", mf.Rational(12345))),
                     decode_limits=mf.DecodeLimits(max_integer_digits=3))
        self.assertEqual(caught.exception.code, "resource_limit_exceeded")
        with self.assertRaises(mf.SerializationError) as caught:
            mf.loads(payload, computation_limits=mf.ComputationLimits(max_nodes=1))
        self.assertEqual(caught.exception.code, "resource_limit_exceeded")


if __name__ == "__main__":
    unittest.main()
