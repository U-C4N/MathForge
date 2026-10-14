# Compute limits and analyze real functions

These operations cover univariate polynomials and rational functions with
rational coefficients and an unconstrained real variable. Finite limit points
may be rational or real algebraic. The [exact algebra guide](exact-algebra.md)
also covers algebraic integral bounds and the separate `extrema_values` API.
Complex analysis and transcendental functions are outside this release.
Verification checks algorithms and certificates; it is not a formal
proof system.

## Keep a hole separate from a pole

```python
import mathforge as mf

x = mf.symbol("x", domain=mf.Reals)
f = mf.parse_expression("((x-1)*(x^2+2))/(x*(x-1))", symbols={"x": x})
result = mf.operations.analyze_function(f, x)
analysis = result.unwrap()
assert mf.verify(result).verified
assert analysis.domain == mf.IntervalSet((
    mf.Interval(None, 0), mf.Interval(0, 1), mf.Interval(1, None)))
assert analysis.excluded_points == (
    mf.ExcludedPoint(mf.Rational(0), "pole"),
    mf.ExcludedPoint(mf.Rational(1), "hole"))
assert analysis.zero_set == mf.IntervalSet(())
assert analysis.positive_set == mf.IntervalSet((mf.Interval(0, 1), mf.Interval(1, None)))
assert analysis.negative_set == mf.IntervalSet((mf.Interval(None, 0),))
print(mf.render_text(analysis))
```

The reduced formula is `(x^2+2)/x`, but the original denominator still excludes
`x=1`. Analysis classifies it as a **hole** because the reduced denominator is
nonzero there. At `x=0` the reduced denominator vanishes: it is a **pole**.
Both `function` and `derivative` retain the original domain exclusions.

## Distinguish limit values from execution errors

```python
hole_limit = mf.operations.limit(f, x, 1)
assert hole_limit.unwrap() == mf.LimitValue("finite", mf.Rational(3))
assert mf.verify(hole_limit).verified
assert not analysis.domain.contains(1)

assert mf.limit(f, x, 0, side="left").kind == "negative_infinity"
assert mf.limit(f, x, 0, side="right").kind == "positive_infinity"
two_sided = mf.operations.limit(f, x, 0)
assert two_sided.execution_status is mf.ExecutionStatus.COMPLETED
assert two_sided.outcome is mf.Outcome.VALUE
assert two_sided.unwrap() == mf.LimitValue("does_not_exist")
assert mf.verify(two_sided).verified
assert mf.limit(f, x, "-inf").kind == "negative_infinity"
assert mf.limit(f, x, "+inf").kind == "positive_infinity"
```

`LimitValue.kind` is `finite`, `positive_infinity`, `negative_infinity` or
`does_not_exist`. Only `finite` carries a `Rational` or `RealAlgebraicRoot` in `value`; the others carry
`None`. For example, a two-sided odd-order pole has opposite infinite one-sided
limits and produces `does_not_exist`. A same-sign infinite limit is represented
by its signed infinity kind.

Use `side="both"` at `"-inf"` and `"+inf"`. At a finite rational or real algebraic
point, `"left"`, `"right"` and `"both"` are supported. Python floats are not
accepted as limit points. Unsupported mathematics, invalid arguments and resource
exhaustion retain their separate structured error statuses.

## Read maximal monotonic regions

Increasing/decreasing means **strictly** increasing/decreasing. Regions are
maximal connected intervals within the original domain. A defined critical
endpoint is included; a hole or pole is always open and never joins regions.

```python
square = mf.analyze_function(x**2, x)
assert square.monotonic_regions == (
    mf.MonotonicRegion(mf.Interval(None, 0, False, True), "decreasing"),
    mf.MonotonicRegion(mf.Interval(0, None, True, False), "increasing"))
assert square.strict_extrema == (mf.LocalExtremum(mf.Rational(0), "minimum"),)

cubic = mf.analyze_function(x**3, x)
assert cubic.stationary_set == mf.IntervalSet((mf.Interval(0, 0, True, True),))
assert cubic.monotonic_regions == (
    mf.MonotonicRegion(mf.Interval(None, None), "increasing"),)
assert cubic.strict_extrema == ()
```

The stationary point of `x^3` does not split its increasing region because the
derivative keeps the same sign on either side. `stationary_set` contains points
in the domain where the derivative is zero. It uses `IntervalSet`, so a constant
function can be stationary throughout an interval, not just at isolated points.

```python
punctured_constant = mf.rational_function(x - 1, x - 1, variable=x)
constant = mf.analyze_function(punctured_constant, x)
assert constant.stationary_set == constant.domain
assert constant.monotonic_regions == (
    mf.MonotonicRegion(mf.Interval(None, 1), "constant"),
    mf.MonotonicRegion(mf.Interval(1, None), "constant"))
assert constant.strict_extrema == ()

punctured_square = mf.rational_function(x**3, x, variable=x)
assert mf.analyze_function(punctured_square, x).stationary_set == mf.IntervalSet(())
```

Strict local extrema carry their **x-coordinate only**, as a `Rational` or
`RealAlgebraicRoot`; `analyze_function` does not return y-values or global extrema.
Use `extrema_values` for exact `(point, value, kind)` records. Excluded
stationary candidates are removed. Constant regions have no strict extrema.
For the main example, the strict local maximum is at `-sqrt(2)` and the strict
local minimum at `sqrt(2)`; these coordinates are represented exactly.

## Save and recheck an analysis

```python
payload = mf.dumps(result)
assert mf.to_data(result)["version"] == 3
restored = mf.loads(payload)
assert restored == result
assert mf.verify(restored).verified
```

The `FunctionAnalysis` records and rational-point requests illustrated above use
mathematical JSON v3. Algebraic endpoint computations and `extrema_values` use v4.
Stored verification labels
do not establish correctness; `verify` checks the original source, result and
certificate again. Old v1/v2 values keep their original bytes and references,
including inside a mixed workspace. See the [record fields](api.md) and
[complete executable workflow](../examples/v04_workflow.py).
