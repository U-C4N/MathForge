# Show exact results

`render_text` and `to_latex` are presentation helpers. They do not change objects,
run a solver, certify stored verification labels, or replace `dumps`/`loads`.
Existing `str()` output is unchanged. LaTeX is returned without math delimiters;
no TeX installation is needed.

```python
import mathforge as mf

x = mf.symbol("x")
assert mf.render_text(mf.Rational(1, 3)) == "1/3"
assert mf.to_latex(mf.Rational(1, 3)) == r"\frac{1}{3}"
rf = mf.rational_function(x, x, variable=x)
assert "x != 0" in mf.render_text(rf)
assert r"\ne 0" in mf.to_latex(rf)
```

Supported values include expressions, polynomials, rational functions, root
records, matrices, RREF records, linear systems, affine solutions, finite/interval
sets, square-free decompositions, limits, function-analysis records, results and
verification reports. The v0.7 renderer also covers real elementary functions
and domains, complex algebraic values, finite distributions, integer
factorizations, CRT records, LU/least-squares results, Taylor polynomials,
formal rational primitives and unevaluated requests. Root indices
and multiplicities, interval closure, empty matrix dimensions and all rational
function exclusions remain visible. There is no approximate numeric fallback.

Function analysis shows holes/poles, sign and stationary sets, monotonic regions,
and strict local extrema x-coordinates. Limit output distinguishes finite values,
signed infinity and nonexistence; none is presented as a solver failure.

`RealFunction` displays its retained domain restrictions. `LogRootSum` and
`RationalPrimitive` displays describe formal logarithmic primitives with locally
chosen branches; their text does not supply a pointwise complex-log evaluator.
Use canonical v5 JSON for these values when transporting them between clients.
The renderer still supports earlier mathematical formats.

## Resolve ambiguous symbol names

```python
other_x = mf.symbol("x")
aliases = {"left": x, "right": other_x}
assert mf.render_text(mf.Add((x, other_x)), symbols=aliases) == "left + right"
```

Without disambiguating aliases, distinct identities with the same visible name
raise `InvalidInput`. Symbol names are escaped for LaTeX, never inserted as raw
commands. An affine solution's generated parameters are independent real scalars,
not additional stored MathForge symbols.

## Display result status without inventing proof

```python
result = mf.solve(mf.Eq(x**2, 2), for_=x)
print(mf.render_text(result))
report = mf.verify(result)
assert report.verified
print(mf.render_text(report))
```

A rendered `Result` reports execution, outcome, exactness and completeness; it
does not newly assert verification. A rendered `VerificationReport` displays its
checks. Use a freshly computed report for imported/untrusted results. No new
teaching steps are fabricated from a single certificate step.

A deferred result displays its `UnevaluatedIntegral` or `UnevaluatedLimit` request
alongside `NO_CONCLUSION`, with no exact-value claim. A deferred limit is distinct
from a verified `does_not_exist` value. Always inspect `outcome`, `value` and
`deferred` in the structured result when deciding whether a computation answered
the question.

Default `RenderLimits`: 65,536 output characters, 10,000 visited nodes and depth
128. Exhaustion raises `ResourceLimitError`; output is never silently truncated.
Rendering is not promised to be parseable back into an identical object.
