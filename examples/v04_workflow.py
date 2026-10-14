"""Exact limits, signs and variation with one hole and one pole (v0.4)."""

import mathforge as mf


def main():
    x = mf.symbol("x", domain=mf.Reals)
    f = mf.parse_expression("((x-1)*(x^2+2))/(x*(x-1))", symbols={"x": x})
    result = mf.operations.analyze_function(f, x)
    analysis = result.unwrap()
    assert mf.verify(result).verified
    assert analysis.domain == mf.IntervalSet((
        mf.Interval(None, 0), mf.Interval(0, 1), mf.Interval(1, None)))
    assert analysis.excluded_points == (
        mf.ExcludedPoint(mf.Rational(0), "pole"), mf.ExcludedPoint(mf.Rational(1), "hole"))
    assert analysis.zero_set == mf.IntervalSet(())
    assert analysis.positive_set == mf.IntervalSet((mf.Interval(0, 1), mf.Interval(1, None)))
    assert analysis.negative_set == mf.IntervalSet((mf.Interval(None, 0),))
    assert analysis.derivative == mf.diff(f, x)
    negative_root = mf.RealAlgebraicRoot((-2, 0, 1), 0)
    positive_root = mf.RealAlgebraicRoot((-2, 0, 1), 1)
    assert analysis.stationary_set == mf.IntervalSet((
        mf.Interval(negative_root, negative_root, True, True),
        mf.Interval(positive_root, positive_root, True, True)))
    assert analysis.monotonic_regions == (
        mf.MonotonicRegion(mf.Interval(None, negative_root, False, True), "increasing"),
        mf.MonotonicRegion(mf.Interval(negative_root, 0, True, False), "decreasing"),
        mf.MonotonicRegion(mf.Interval(0, 1), "decreasing"),
        mf.MonotonicRegion(mf.Interval(1, positive_root, False, True), "decreasing"),
        mf.MonotonicRegion(mf.Interval(positive_root, None, True, False), "increasing"))
    assert analysis.strict_extrema == (
        mf.LocalExtremum(negative_root, "maximum"), mf.LocalExtremum(positive_root, "minimum"))

    assert mf.limit(f, x, 1) == mf.LimitValue("finite", mf.Rational(3))
    assert mf.limit(f, x, 0, side="left").kind == "negative_infinity"
    assert mf.limit(f, x, 0, side="right").kind == "positive_infinity"
    no_limit = mf.operations.limit(f, x, 0)
    assert no_limit.execution_status is mf.ExecutionStatus.COMPLETED
    assert no_limit.unwrap() == mf.LimitValue("does_not_exist")
    assert mf.verify(no_limit).verified
    assert mf.limit(f, x, "+inf").kind == "positive_infinity"
    assert mf.limit(f, x, "-inf").kind == "negative_infinity"

    # A stationary point need not turn; a hole always separates connected regions.
    cubic = mf.analyze_function(x**3, x)
    assert cubic.monotonic_regions == (
        mf.MonotonicRegion(mf.Interval(None, None), "increasing"),)
    assert cubic.stationary_set.contains(0) and cubic.strict_extrema == ()
    constant = mf.analyze_function(mf.rational_function(x-1, x-1, variable=x), x)
    assert constant.stationary_set == constant.domain
    assert constant.monotonic_regions == (
        mf.MonotonicRegion(mf.Interval(None, 1), "constant"),
        mf.MonotonicRegion(mf.Interval(1, None), "constant"))
    assert constant.strict_extrema == ()
    punctured = mf.analyze_function(mf.rational_function(x**3, x, variable=x), x)
    assert punctured.stationary_set == mf.IntervalSet(())

    # A mixed v3 workspace retains the original v2 object's content reference.
    workspace = mf.Workspace()
    legacy = mf.operations.det(mf.Matrix(((1, 2), (3, 4))))
    legacy_ref, legacy_json = workspace.put(legacy), mf.dumps(legacy)
    analysis_ref = workspace.put(result)
    assert mf.to_data(legacy)["version"] == 2
    assert mf.to_data(workspace)["version"] == 3
    restored = mf.loads(mf.dumps(workspace))
    assert mf.dumps(restored.get(legacy_ref)) == legacy_json
    assert restored.put(restored.get(legacy_ref)) == legacy_ref
    assert mf.verify(restored.get(analysis_ref)).verified
    print(mf.render_text(analysis))
    print("Hole limit: 3; pole limits: -infinity / +infinity; analysis rechecked.")


if __name__ == "__main__":
    main()
