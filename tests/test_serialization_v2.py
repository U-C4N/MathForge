"""Strict v2 codecs, bounded decoding and mixed-workspace compatibility."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mathforge as m
from mathforge import contracts as c
from mathforge.algebraic import RealAlgebraicRoot, RootRecord
from mathforge.intervals import Interval, IntervalSet
from mathforge.limits import ComputationLimits, DecodeLimits
from mathforge.matrices import AffineSolutionSet, LinearSystem, Matrix, RREFResult
from mathforge.polynomial import Polynomial, SquareFreeDecomposition, SquareFreeFactor
from mathforge.rational_function import RationalFunction, rational_function


class V2SerializationTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000010")
        self.y = m.Symbol("y", uid="00000000-0000-4000-8000-000000000011")
        self.p = Polynomial((0, 1), self.x)

    def roundtrip(self, value):
        data = m.to_data(value)
        self.assertEqual(data["version"], 2)
        payload = m.dumps(value)
        restored = m.loads(payload)
        self.assertEqual(restored, value)
        self.assertEqual(m.dumps(restored), payload)
        return restored

    def test_all_new_value_types(self):
        """SER-03/SER-11/SER-19: every new type has a strict stable codec."""
        root = RealAlgebraicRoot((-2, 0, 0, 1), 0)
        rf = rational_function(Polynomial((-1, 0, 1), self.x), Polynomial((-1, 1), self.x), variable=self.x)
        matrix = Matrix(((1, 2), (0, 1)))
        values = (
            SquareFreeFactor(self.p, 2),
            SquareFreeDecomposition(m.Rational(3), (SquareFreeFactor(self.p, 2),)),
            root, RootRecord(root, 1), rf, matrix,
            Matrix((), ncols=3), Matrix(((), ())), Matrix(()),
            RREFResult(Matrix(((1, 0), (0, 1))), (0, 1)),
            LinearSystem(matrix, (3, 1), (self.x, self.y)),
            AffineSolutionSet((self.x, self.y), (1, 0), ((-1, 1),)),
            c.Inequality(rf, "ge", m.Rational(0)),
            Interval(None, m.Rational(0), False, False),
            Interval(m.Rational(0), root, True, False),
            IntervalSet((Interval(None, m.Rational(0), False, False),)),
            c.OperationRequest("fixture", (("expression", self.x), ("order", 2))),
        )
        for value in values:
            with self.subTest(type=type(value).__name__):
                self.roundtrip(value)

    def test_result_extended_context_and_request(self):
        """SER-03: legacy-compatible fields alone do not hide a new request."""
        request = c.OperationRequest("differentiate", (("expression", self.x), ("variable", self.x)))
        result = c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.VALUE, m.Rational(1),
                          c.Exactness.EXACT, c.Completeness.COMPLETE,
                          expression=self.x, for_=self.x, request=request)
        self.roundtrip(result)
        # Polynomial is a v1 type but was not a v1 Result.expression type.
        extended = replace(result, expression=self.p, request=None)
        self.roundtrip(extended)

    def test_old_result_in_v2_graph_retains_standalone_v1(self):
        """SER-02/SER-05: enclosing v2 never upgrades a legacy object's identity."""
        old = c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.VALUE, m.Rational(2))
        payload = m.dumps(old)
        graph = (old, Matrix(((1,),)))
        data = m.to_data(graph)
        self.assertEqual(data["version"], 2)
        self.assertIsNone(data["data"]["items"][0]["request"])
        restored = m.from_data(data)
        self.assertEqual(m.dumps(restored[0]), payload)

    def test_mixed_workspace_preserves_legacy_references(self):
        """SER-04/SER-05/SER-14: each ref hashes its own minimum-version envelope."""
        fixture = Path(__file__).parent / "fixtures" / "v1" / "legacy_workspace.json"
        ws = m.loads(fixture.read_text(encoding="utf-8"))
        originals = {ref: m.dumps(value) for ref, value in ws.objects.items()}
        new_ref = ws.put(Matrix(((1, 2),)))
        restored = m.loads(m.dumps(ws))
        self.assertEqual(m.to_data(ws)["version"], 2)
        self.assertEqual(restored.get(new_ref), Matrix(((1, 2),)))
        for ref, payload in originals.items():
            self.assertEqual(m.dumps(restored.get(ref)), payload)
            self.assertEqual(restored.put(restored.get(ref)), ref)

    def test_refined_isolation_does_not_change_root_reference(self):
        """SER-13: the content-addressed root excludes its narrowing certificate."""
        from mathforge.algebraic import interval_for
        from mathforge.root_isolation import refine_interval
        root = RealAlgebraicRoot((-2, 0, 0, 1), 0)
        ws = m.Workspace()
        ref, encoded = ws.put(root), m.dumps(root)
        original = interval_for(root)
        narrowed = refine_interval(Polynomial(root.integer_coefficients, self.x), original)
        self.assertNotEqual(narrowed, original)
        self.assertEqual(m.dumps(root), encoded)
        self.assertEqual(ws.put(root), ref)

    def test_workspace_corruption_does_not_return_a_partial_snapshot(self):
        """SER-15: a valid prefix does not turn a corrupt workspace into success."""
        ws = m.Workspace()
        ws.put(m.Rational(1))
        ws.put(Matrix(((1,),)))
        original = m.dumps(ws)
        data = m.to_data(ws)
        data["data"]["objects"][-1]["object"] = {"type": "unrecognized_corrupt_object"}
        with mock.patch.object(m.Workspace, "_restore", wraps=m.Workspace._restore) as restore:
            with self.assertRaises(m.SerializationError):
                m.from_data(data)
            restore.assert_not_called()
        self.assertEqual(m.dumps(ws), original)

    def test_rf_normal_form_roundtrip_does_not_add_guards(self):
        """SER-21/RF-21: preserve guards; reject a reduced denominator's uncovered root."""
        rf = rational_function(Polynomial((-1, 0, 1), self.x), Polynomial((2, -3, 1), self.x), variable=self.x)
        self.assertEqual(rf.denominator, Polynomial((-2, 1), self.x))
        self.assertEqual(len(rf.excluded), 1)
        self.roundtrip(rf)
        ws = m.Workspace()
        ref = ws.put(rf)
        self.assertEqual(m.loads(m.dumps(ws)).put(rf), ref)
        data = m.to_data(rf)
        data["data"]["excluded"] = []
        with self.assertRaises(m.SerializationError):
            m.from_data(data)

    def test_v1_rejects_v2_and_unknown_versions(self):
        """SER-06/SER-07: version is enforced before accepting new tags."""
        data = m.to_data(Matrix(((1,),)))
        for version in (1, 999, True):
            with self.subTest(version=version):
                modified = deepcopy(data)
                modified["version"] = version
                with self.assertRaises(m.SerializationError):
                    m.from_data(modified)
        old = c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.VALUE, m.Rational(1))
        data = m.to_data(old)
        data["data"]["request"] = None
        with self.assertRaises(m.SerializationError):
            m.from_data(data)

    def test_root_noncanonical_coefficients_and_invalid_index(self):
        """SER-09/ROOT-18: shape validation is not a substitute for index validity."""
        data = m.to_data(RealAlgebraicRoot((-2, 0, 0, 1), 0))
        mutations = (
            {"integer_coefficients": ["-4", "0", "0", "2"]},
            {"real_index": "1"}, {"real_index": "-1"}, {"real_index": True},
            {"real_index": "00"}, {"unexpected": "field"},
        )
        for mutation in mutations:
            modified = deepcopy(data)
            modified["data"].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(m.SerializationError):
                m.from_data(modified)

    def test_request_and_interval_noncanonical_order(self):
        """SER-07/SER-10: decoder rejects inputs that constructors would normalize."""
        request = c.OperationRequest("fixture", (("a", 1), ("b", 2)))
        data = m.to_data(request)
        data["data"]["arguments"].reverse()
        with self.assertRaises(m.SerializationError):
            m.from_data(data)
        intervals = IntervalSet((Interval(m.Rational(0), m.Rational(1)),
                                 Interval(m.Rational(3), m.Rational(4))))
        data = m.to_data(intervals)
        data["data"]["intervals"].reverse()
        with self.assertRaises(m.SerializationError):
            m.from_data(data)
        data = m.to_data(Interval(m.Rational(0), m.Rational(1)))
        data["data"]["left_closed"] = 1
        with self.assertRaises(m.SerializationError):
            m.from_data(data)

    def test_uuid_conflicts_inside_new_request(self):
        """SER-12: the symbol registry spans new request and problem graphs."""
        request = c.OperationRequest("fixture", (("first", self.x), ("second", self.x)))
        data = m.to_data(request)
        data["data"]["arguments"][1]["value"]["name"] = "different"
        with self.assertRaises(m.SerializationError):
            m.from_data(data)

    def test_decode_limits_and_cycle_preflight(self):
        """SER-17/LIM-07: reject bounded failures before expensive construction."""
        payload = m.dumps(m.Rational(123456789))
        for limits in (DecodeLimits(max_bytes=10), DecodeLimits(max_depth=1),
                       DecodeLimits(max_integer_digits=5)):
            with self.subTest(limits=limits), self.assertRaises(m.SerializationError) as caught:
                m.loads(payload, decode_limits=limits)
            self.assertEqual(caught.exception.code, "resource_limit_exceeded")
        with self.assertRaises(m.SerializationError):
            m.loads(m.dumps((1, 2, 3)), decode_limits=DecodeLimits(max_nodes=2))
        data = {"format": "mathforge", "version": 1, "kind": "object", "data": {}}
        data["data"]["cycle"] = data
        with self.assertRaises(m.SerializationError):
            m.from_data(data)
        # Brackets and escaped quotes inside strings are not nesting.
        value = '[{\\\"' * 200
        self.assertEqual(m.loads(m.dumps(value), decode_limits=DecodeLimits(max_depth=8)), value)

    def test_computation_budget_applies_to_root_semantics(self):
        """LIM-08: well-shaped roots still require bounded semantic validation."""
        data = m.to_data(RealAlgebraicRoot((-2, 0, 0, 1), 0))
        with self.assertRaises(m.SerializationError) as caught:
            m.from_data(data, computation_limits=ComputationLimits(max_work=1))
        self.assertEqual(caught.exception.code, "resource_limit_exceeded")

    def test_workspace_limit_propagation(self):
        """SER-22: put and load use explicitly supplied finite decode limits."""
        ws = m.Workspace()
        with self.assertRaises(m.SerializationError):
            ws.put("x" * 1000, decode_limits=DecodeLimits(max_bytes=100))
        self.assertEqual(len(ws.objects), 0)
        ref = ws.put("x" * 1000, decode_limits=DecodeLimits(max_bytes=10000))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workspace.json"
            ws.save(path)
            with self.assertRaises(m.SerializationError):
                m.Workspace.load(path, decode_limits=DecodeLimits(max_bytes=100))
            restored = m.Workspace.load(path, decode_limits=DecodeLimits(max_bytes=10000))
            self.assertEqual(restored.get(ref), "x" * 1000)

    def test_atomic_save_failure_keeps_existing_file(self):
        """SER-16: fsync/replace failure leaves old bytes and no temporary file."""
        ws = m.Workspace()
        ws.put(m.Rational(1))
        for target in ("mathforge.workspace.os.fsync", "mathforge.workspace.os.replace"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "workspace.json"
                path.write_text("previous document", encoding="utf-8")
                with mock.patch(target, side_effect=OSError("injected failure")):
                    with self.assertRaises(OSError):
                        ws.save(path)
                self.assertEqual(path.read_text(encoding="utf-8"), "previous document")
                self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_atomic_save_write_and_flush_failure_clean_up(self):
        """SER-16: failures before fsync also preserve the previous snapshot."""
        from mathforge import workspace
        real_temporary_file = workspace.tempfile.NamedTemporaryFile
        ws = m.Workspace()
        ws.put(m.Rational(1))
        for method in ("write", "flush"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "workspace.json"
                path.write_text("previous document", encoding="utf-8")

                def failing_stream(*args, **kwargs):
                    stream = real_temporary_file(*args, **kwargs)
                    setattr(stream, method, mock.Mock(side_effect=OSError("injected failure")))
                    return stream

                with mock.patch("mathforge.workspace.tempfile.NamedTemporaryFile", side_effect=failing_stream):
                    with self.assertRaises(OSError):
                        ws.save(path)
                self.assertEqual(path.read_text(encoding="utf-8"), "previous document")
                self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
