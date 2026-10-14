"""Source-linked function analysis, maximality and independent evidence checks."""

from dataclasses import replace
from unittest import TestCase, main, mock

import mathforge as mf
# Load every patched module before replacing imported aliases. Otherwise the
# first import of real_sets can permanently retain a mocked real_roots alias.
import mathforge.real_sets
from mathforge.analysis_types import ExcludedPoint, LocalExtremum, MonotonicRegion
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.function_analysis import analyze_function, check_analysis
from mathforge.limits import ResourceLimitError
from mathforge.rational_function import RationalFunction


class FunctionAnalysisV4Tests(TestCase):
    def setUp(self):
        self.x = mf.symbol("x")

    def rf(self, numerator, denominator):
        return mf.rational_function(numerator, denominator, variable=self.x)

    def analyze(self, expression):
        value, evidence = analyze_function(expression, self.x)
        self.assertTrue(check_analysis(expression, self.x, value, evidence))
        return value, evidence

    def assertIntervals(self, value, expected):
        self.assertEqual(len(value.intervals), len(expected))
        for interval, (lower, upper, lc, rc) in zip(value, expected):
            for actual, wanted in ((interval.lower, lower), (interval.upper, upper)):
                if wanted is None:
                    self.assertIsNone(actual)
                else:
                    self.assertEqual(mf.compare_real(actual, wanted), 0)
            self.assertEqual((interval.left_closed, interval.right_closed), (lc, rc))

    def assertRegions(self, value, expected):
        self.assertEqual(len(value.monotonic_regions), len(expected))
        for region, (lower, upper, lc, rc, behavior) in zip(value.monotonic_regions, expected):
            self.assertEqual(region.behavior, behavior)
            self.assertIntervals(mf.IntervalSet((region.interval,)), ((lower, upper, lc, rc),))

    def test_demo_hole_pole_signs_algebraic_stationary_points_and_extrema(self):
        x = self.x
        f = self.rf((x - 1) * (x*x + 2), x * (x - 1))
        value, _ = self.analyze(f)
        self.assertEqual([(p.point, p.kind) for p in value.excluded_points],
                         [(mf.Rational(0), "pole"), (mf.Rational(1), "hole")])
        self.assertIntervals(value.domain, ((None, 0, False, False), (0, 1, False, False),
                                             (1, None, False, False)))
        self.assertIntervals(value.zero_set, ())
        self.assertIntervals(value.positive_set, ((0, 1, False, False), (1, None, False, False)))
        self.assertIntervals(value.negative_set, ((None, 0, False, False),))
        root = mf.sqrt(2)
        self.assertIntervals(value.stationary_set, ((-root, -root, True, True), (root, root, True, True)))
        self.assertRegions(value, ((None, -root, False, True, "increasing"),
                                  (-root, 0, True, False, "decreasing"),
                                  (0, 1, False, False, "decreasing"),
                                  (1, root, False, True, "decreasing"),
                                  (root, None, True, False, "increasing")))
        self.assertEqual([e.kind for e in value.strict_extrema], ["maximum", "minimum"])
        self.assertEqual(mf.compare_real(value.strict_extrema[0].point, -root), 0)
        self.assertEqual(mf.compare_real(value.strict_extrema[1].point, root), 0)
        self.assertFalse(value.derivative.is_defined_at(1))

    def test_stationary_inflection_does_not_split_strict_increase(self):
        value, _ = self.analyze(self.x**3)
        self.assertRegions(value, ((None, None, False, False, "increasing"),))
        self.assertIntervals(value.stationary_set, ((0, 0, True, True),))
        self.assertEqual(value.strict_extrema, ())

    def test_defined_extremum_is_in_both_maximal_monotonic_regions(self):
        value, _ = self.analyze(self.x**2)
        self.assertRegions(value, ((None, 0, False, True, "decreasing"),
                                  (0, None, True, False, "increasing")))
        self.assertEqual(value.strict_extrema, (LocalExtremum(mf.Rational(0), "minimum"),))
        self.assertIntervals(value.zero_set, ((0, 0, True, True),))
        self.assertIntervals(value.positive_set, ((None, 0, False, False), (0, None, False, False)))

    def test_constant_zero_positive_and_negative_functions_keep_punctured_domains(self):
        for constant in (0, 1, -3):
            with self.subTest(constant=constant):
                f = self.rf(constant * self.x, self.x)
                value, _ = self.analyze(f)
                self.assertEqual(value.stationary_set, value.domain)
                self.assertRegions(value, ((None, 0, False, False, "constant"),
                                          (0, None, False, False, "constant")))
                self.assertEqual(value.strict_extrema, ())
                self.assertEqual(value.excluded_points, (ExcludedPoint(mf.Rational(0), "hole"),))
                for actual, enabled in ((value.zero_set, constant == 0),
                                        (value.positive_set, constant > 0),
                                        (value.negative_set, constant < 0)):
                    self.assertEqual(actual, value.domain if enabled else mf.IntervalSet(()))
        value, _ = self.analyze(0)
        self.assertIntervals(value.domain, ((None, None, False, False),))
        self.assertEqual(value.zero_set, value.stationary_set)

    def test_cancelled_stationary_point_is_not_an_extremum(self):
        value, _ = self.analyze(self.rf(self.x**3, self.x))
        self.assertIntervals(value.stationary_set, ())
        self.assertIntervals(value.zero_set, ())
        self.assertEqual(value.strict_extrema, ())
        self.assertRegions(value, ((None, 0, False, False, "decreasing"),
                                  (0, None, False, False, "increasing")))

    def test_same_direction_never_merges_across_a_hole_or_pole(self):
        value, _ = self.analyze(self.rf(self.x**2, self.x))
        self.assertRegions(value, ((None, 0, False, False, "increasing"),
                                  (0, None, False, False, "increasing")))
        value, _ = self.analyze(self.rf(1, self.x))
        self.assertRegions(value, ((None, 0, False, False, "decreasing"),
                                  (0, None, False, False, "decreasing")))
        self.assertEqual(value.strict_extrema, ())

    def test_nonderivative_zeros_do_not_split_monotonic_regions(self):
        value, _ = self.analyze(self.x - 2)
        self.assertRegions(value, ((None, None, False, False, "increasing"),))
        self.assertIntervals(value.zero_set, ((2, 2, True, True),))

    def test_nonreal_and_overlapping_guards_do_not_invent_excluded_points(self):
        x = self.x
        f = self.rf(x * (x*x + 1), x * (x*x + 1)) * self.rf(x*x, x*x)
        value, _ = self.analyze(f)
        self.assertEqual(value.excluded_points, (ExcludedPoint(mf.Rational(0), "hole"),))
        self.assertEqual(len(value.monotonic_regions), 2)

    def test_three_stationary_points_include_algebraic_minima(self):
        value, _ = self.analyze((self.x**2 - 2)**2)
        self.assertEqual([e.kind for e in value.strict_extrema], ["minimum", "maximum", "minimum"])
        self.assertEqual(mf.compare_real(value.strict_extrema[0].point, -mf.sqrt(2)), 0)
        self.assertEqual(value.strict_extrema[1].point, mf.Rational(0))
        self.assertEqual(mf.compare_real(value.strict_extrema[2].point, mf.sqrt(2)), 0)

    def test_repeated_denominator_factors_cancel_in_the_derivative_identity(self):
        f = self.rf(self.x, (self.x - 1)**3)
        value, _ = self.analyze(f)
        expected = self.rf(-2*self.x - 1, (self.x - 1)**4)
        self.assertEqual(value.derivative.numerator, expected.numerator)
        self.assertEqual(value.derivative.denominator, expected.denominator)
        self.assertIntervals(value.stationary_set, ((mf.Rational(-1, 2), mf.Rational(-1, 2), True, True),))
        self.assertEqual(value.strict_extrema, (LocalExtremum(mf.Rational(-1, 2), "maximum"),))

    def test_constant_with_only_nonreal_exclusions_has_the_full_real_domain(self):
        for constant in (0, 7, -2):
            f = self.rf(constant * (self.x**2 + 1), self.x**2 + 1)
            value, _ = self.analyze(f)
            self.assertEqual(value.excluded_points, ())
            self.assertIntervals(value.domain, ((None, None, False, False),))
            self.assertEqual(value.stationary_set, value.domain)
            self.assertRegions(value, ((None, None, False, False, "constant"),))

    def test_algebraic_holes_break_regions_and_are_not_extrema(self):
        x = self.x
        value, _ = self.analyze(self.rf((x*x - 2) * (x*x + 1), x*x - 2))
        root = mf.sqrt(2)
        self.assertEqual([p.kind for p in value.excluded_points], ["hole", "hole"])
        self.assertEqual(mf.compare_real(value.excluded_points[0].point, -root), 0)
        self.assertEqual(mf.compare_real(value.excluded_points[1].point, root), 0)
        self.assertEqual(value.positive_set, value.domain)
        self.assertRegions(value, ((None, -root, False, False, "decreasing"),
                                  (-root, 0, False, True, "decreasing"),
                                  (0, root, True, False, "increasing"),
                                  (root, None, False, False, "increasing")))
        self.assertEqual(value.strict_extrema, (LocalExtremum(mf.Rational(0), "minimum"),))

    def test_multiple_same_sign_stationary_points_merge_into_one_region(self):
        f = self.x**5 / 5 - 2*self.x**3 / 3 + self.x
        value, _ = self.analyze(f)
        self.assertRegions(value, ((None, None, False, False, "increasing"),))
        self.assertIntervals(value.stationary_set, ((-1, -1, True, True), (1, 1, True, True)))
        self.assertEqual(value.strict_extrema, ())

    def test_every_analysis_claim_is_checked(self):
        x = self.x
        f = self.rf((x - 1) * (x*x + 2), x * (x - 1))
        value, evidence = self.analyze(f)
        empty, all_reals = mf.IntervalSet(()), mf.IntervalSet((mf.Interval(None, None),))
        mutations = {
            "function": replace(value, function=f + 1),
            "domain": replace(value, domain=all_reals),
            "missing_exclusion": replace(value, excluded_points=value.excluded_points[:-1]),
            "exclusion_kind": replace(value, excluded_points=(replace(value.excluded_points[0], kind="hole"),
                                                               value.excluded_points[1])),
            "zero_set": replace(value, zero_set=all_reals),
            "positive_set": replace(value, positive_set=empty),
            "negative_set": replace(value, negative_set=empty),
            "derivative_value": replace(value, derivative=value.derivative + 1),
            "derivative_domain": replace(value, derivative=self.rf(value.derivative.numerator,
                                                                    value.derivative.denominator)),
            "stationary_set": replace(value, stationary_set=empty),
            "region_direction": replace(value, monotonic_regions=(replace(value.monotonic_regions[0],
                                                    behavior="decreasing"), *value.monotonic_regions[1:])),
            "missing_region": replace(value, monotonic_regions=value.monotonic_regions[:-1]),
            "missing_extremum": replace(value, strict_extrema=value.strict_extrema[:-1]),
            "extremum_kind": replace(value, strict_extrema=(replace(value.strict_extrema[0], kind="minimum"),
                                                              value.strict_extrema[1])),
        }
        for name, mutation in mutations.items():
            with self.subTest(claim=name):
                self.assertFalse(check_analysis(f, x, mutation, evidence))

    def test_checker_rejects_nonmaximal_regions_and_wrong_endpoint_closure(self):
        f = self.x**3
        value, evidence = self.analyze(f)
        split = (MonotonicRegion(mf.Interval(None, 0, right_closed=True), "increasing"),
                 MonotonicRegion(mf.Interval(0, None, left_closed=True), "increasing"))
        self.assertFalse(check_analysis(f, self.x, replace(value, monotonic_regions=split), evidence))
        f = self.x**2
        value, evidence = self.analyze(f)
        regions = (replace(value.monotonic_regions[0], interval=mf.Interval(None, 0)),
                   value.monotonic_regions[1])
        self.assertFalse(check_analysis(f, self.x, replace(value, monotonic_regions=regions), evidence))
        f = self.rf(self.x**2, self.x)
        value, evidence = self.analyze(f)
        merged = (MonotonicRegion(mf.Interval(None, None), "increasing"),)
        self.assertFalse(check_analysis(f, self.x, replace(value, monotonic_regions=merged), evidence))

    def test_constant_or_inflection_or_hole_cannot_be_a_strict_extremum(self):
        for f in (self.x**3, 1, self.rf(self.x**3, self.x)):
            value, evidence = self.analyze(f)
            forged = replace(value, strict_extrema=(LocalExtremum(mf.Rational(0), "minimum"),))
            self.assertFalse(check_analysis(f, self.x, forged, evidence))

    def test_evidence_binds_complete_roots_source_cells_and_signs(self):
        f = self.x**2
        value, evidence = self.analyze(f)
        groups, samples = evidence
        source, roots, certificate = groups[0]
        altered = (
            (groups[:-1], samples),
            (((source, (), (certificate[0], ())), *groups[1:]), samples),
            (((source + 1, roots, certificate), *groups[1:]), samples),
            (groups, samples[:-1]),
            (groups, ((mf.Rational(0), samples[0][1], samples[0][2]), *samples[1:])),
            (groups, ((samples[0][0], -samples[0][1], samples[0][2]), *samples[1:])),
            (groups, ((samples[0][0], samples[0][1], -samples[0][2]), *samples[1:])),
            (groups, ((samples[0][0], True, samples[0][2]), *samples[1:])),
            (groups, tuple(reversed(samples))),
        )
        for mutation in altered:
            with self.subTest(evidence=mutation):
                self.assertFalse(check_analysis(f, self.x, value, mutation))
        for malformed in (None, (), ((),), (None, samples), (groups, [*samples]), (((),), samples)):
            self.assertFalse(check_analysis(f, self.x, value, malformed))

    def test_checker_does_not_call_any_producer_or_result_assembler(self):
        f = self.rf((self.x - 1) * (self.x**2 + 2), self.x * (self.x - 1))
        value, evidence = self.analyze(f)
        blocked = AssertionError("A producer was called during verification")
        with (mock.patch("mathforge.function_analysis.analyze_function", side_effect=blocked),
              mock.patch("mathforge.function_analysis._assemble_analysis", side_effect=blocked),
              mock.patch("mathforge.function_analysis._producer_regions", side_effect=blocked),
              mock.patch("mathforge.function_analysis.real_roots", side_effect=blocked),
              mock.patch("mathforge.root_isolation.real_roots", side_effect=blocked),
              mock.patch("mathforge.real_sets.solve_real_set", side_effect=blocked),
              mock.patch.object(RationalFunction, "derivative", side_effect=blocked)):
            self.assertTrue(check_analysis(f, self.x, value, evidence))
            self.assertFalse(check_analysis(f, self.x, replace(value, stationary_set=mf.IntervalSet(())), evidence))

    def test_source_identity_and_supported_variable_scope(self):
        value, evidence = self.analyze(self.x**2)
        self.assertFalse(check_analysis(self.x**3, self.x, value, evidence))
        other = mf.symbol("x")
        self.assertFalse(check_analysis(other**2, other, value, evidence))
        with self.assertRaises(InvalidInput):
            analyze_function(self.rf(1, self.x), other)
        for variable in (mf.symbol("n", domain=mf.Integers),
                         mf.symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))):
            with self.assertRaises(UnsupportedOperation):
                analyze_function(variable, variable)

    def test_producer_and_checker_have_finite_work_and_evidence_budgets(self):
        f = (self.x**2 - 2)**2
        value, evidence = self.analyze(f)
        for limits in (mf.ComputationLimits(max_work=1), mf.ComputationLimits(max_evidence_items=1)):
            with self.subTest(limits=limits):
                with self.assertRaises(ResourceLimitError):
                    analyze_function(f, self.x, limits=limits)
                with self.assertRaises(ResourceLimitError):
                    check_analysis(f, self.x, value, evidence, limits=limits)


if __name__ == "__main__":
    main()
