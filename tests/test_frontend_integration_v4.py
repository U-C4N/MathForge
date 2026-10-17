"""v0.4 results remain exact and source-bound across every public adapter."""

from dataclasses import FrozenInstanceError, replace
import unittest

import mathforge as mf
from mathforge.capabilities import area_tools
from mathforge.mcp.boundary import _FIELDS


class AnalysisFrontendTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000404")
        self.f = mf.parse_expression("((x-1)*(x^2+2))/(x*(x-1))", symbols={"x": self.x})

    def test_demo_roundtrip_presentation_preserves_exact_source(self):
        result = mf.operations.analyze_function(self.f, self.x)
        self.assertIs(result.execution_status, mf.ExecutionStatus.COMPLETED, result.error)
        payload = mf.dumps(result)
        restored = mf.loads(payload)
        self.assertTrue(mf.verify(restored).verified)
        self.assertEqual(mf.to_data(restored)["version"], 3)
        value = restored.unwrap()
        self.assertFalse(value.function.is_defined_at(0))
        self.assertFalse(value.function.is_defined_at(1))
        self.assertFalse(value.derivative.is_defined_at(1))
        text = mf.render_text(restored)
        latex = mf.to_latex(restored)
        for fragment in ("domain:", "hole", "pole", "strictly increasing", "strictly decreasing",
                         "stationary points", "strict local minimum", "strict local maximum"):
            self.assertIn(fragment, text)
        self.assertIn(r"\begin{aligned}", latex)
        self.assertIn(r"\operatorname{RootOf}", latex)
        self.assertNotIn("verified", text)
        self.assertEqual(mf.dumps(restored), payload)
        self.assertEqual(mf.limit(self.f, self.x, 1), mf.LimitValue("finite", mf.Rational(3)))

    def test_all_limit_value_displays(self):
        for value, text, latex in (
            (mf.LimitValue("finite", mf.Rational(-1, 2)), "-1/2", r"-\frac{1}{2}"),
            (mf.LimitValue("positive_infinity"), "+inf", r"+\infty"),
            (mf.LimitValue("negative_infinity"), "-inf", r"-\infty"),
            (mf.LimitValue("does_not_exist"), "limit does not exist", r"\text{limit does not exist}"),
        ):
            with self.subTest(kind=value.kind):
                self.assertEqual(mf.render_text(value), text)
                self.assertEqual(mf.to_latex(value), latex)

    def test_analysis_aliases_escape_and_render_limits(self):
        result = mf.analyze_function(self.x**2, self.x)
        self.assertIn(r"\text{curve\_a}", mf.to_latex(result, symbols={"curve_a": self.x}))
        for renderer in (mf.render_text, mf.to_latex):
            for bounds in (mf.RenderLimits(max_nodes=3), mf.RenderLimits(max_depth=1),
                           mf.RenderLimits(max_chars=32)):
                with self.subTest(renderer=renderer, limits=bounds), self.assertRaises(mf.ResourceLimitError):
                    renderer(result, limits=bounds)

    def test_invalid_and_unsupported_public_results_are_serializable(self):
        calls = (
            lambda: mf.operations.limit(self.x, self.x, "inf"),
            lambda: mf.operations.limit(self.x, self.x, 0.5),
            lambda: mf.operations.limit(self.x, self.x, mf.symbol("unbound")),
            lambda: mf.operations.limit(self.x, self.x, 0, side="up"),
            lambda: mf.operations.limit(self.x, self.x, "+inf", side="right"),
            lambda: mf.operations.limit(self.x, "x", 0),
            lambda: mf.operations.analyze_function(mf.Matrix(((1,),)), self.x),
            lambda: mf.operations.analyze_function(self.x, "x"),
            lambda: mf.operations.analyze_function(mf.sqrt(2) + self.x, self.x),
        )
        for call in calls:
            result = call()
            with self.subTest(error=result.error):
                self.assertIn(result.execution_status, (mf.ExecutionStatus.INVALID_INPUT, mf.ExecutionStatus.UNSUPPORTED))
                self.assertIsNone(result.value)
                self.assertIs(result.outcome, mf.Outcome.NOT_APPLICABLE)
                self.assertEqual(mf.loads(mf.dumps(result)), result)
                self.assertFalse(mf.verify(result).verified)
                with self.assertRaises(mf.OperationError):
                    result.unwrap()

    def test_immutable_records_reject_inconsistent_shapes(self):
        value = mf.LimitValue("finite", mf.Rational(2))
        with self.assertRaises(FrozenInstanceError):
            value.kind = "does_not_exist"
        for kind, number in (("finite", None), ("finite", 2), ("positive_infinity", mf.Rational(2)),
                             ("unknown", None)):
            with self.subTest(kind=kind, number=number), self.assertRaises(mf.InvalidInput):
                mf.LimitValue(kind, number)
        with self.assertRaises(mf.InvalidInput):
            mf.LocalExtremum(mf.sqrt(2), "minimum")
        with self.assertRaises(mf.InvalidInput):
            mf.MonotonicRegion(mf.Interval(1, 0), "increasing")
        analysis = mf.analyze_function(self.x, self.x)
        with self.assertRaises(mf.InvalidInput):
            replace(analysis, zero_set=mf.EmptySet())

    def test_capabilities_match_new_public_contract(self):
        capabilities = mf.capabilities()
        tools = {item["name"]: item for item in capabilities["tools"]}
        # The 30 v0.7 tools lead, unchanged; the area tools with operations and the catalog follow.
        names = ["mathforge." + name for name in _FIELDS]
        names += area_tools() + ["mathforge.catalog"]
        self.assertEqual(list(tools), names)
        for name in ("limit", "analyze_function"):
            self.assertEqual(tools["mathforge." + name]["sdk"], "operations." + name)
        self.assertIn("v3", capabilities["output"]["mathematics"])


if __name__ == "__main__":
    unittest.main()
