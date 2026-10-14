# Choose a Python operation

Convenience functions return mathematical values and raise `OperationError` on
failure; `solve` and `solve_system` return structured results directly. Use
`mathforge.operations` when every operation should return a `Result[T]`.

```python
import mathforge as mf

A = mf.Matrix(((1, 2), (3, 4)))
result = mf.operations.det(A)
assert result.unwrap() == mf.Rational(-2)
assert mf.verify(result).verified
```

## Operations and boundaries

| Family | Public operations | Mathematical scope |
| --- | --- | --- |
| Expression | `simplify`, `evaluate`, `substitute` | Documented local exact rules and explicit bindings |
| Polynomial | `expand`, `poly_divmod`, `poly_gcd`, `poly_xgcd`, `square_free`, `factor`, `resultant`, `discriminant` | Univariate rational-coefficient polynomials; complete factorization over Q |
| Algebraic numbers | `algebraic`, `algebraic_div`, `algebraic_pow`, `minimal_polynomial`, `evaluate_algebraic` | Explicit exact real algebraic normalization and evaluation |
| Rational functions | `apart` | Partial fractions over Q, preserving original domain exclusions |
| Multivariate | `expand_multivariate`, `partial_diff`, `gradient`, `jacobian`, `hessian` | Rational-coefficient polynomial expressions over ordered real axes; expression/tuple output |
| Roots and sets | `real_roots`, `compare_real`, `solve`, `rational_function` | Real roots; polynomial/rational equations and inequalities; supported exact comparisons |
| Calculus | `diff`, `antiderivative`, `integrate`, `rational_antiderivative`, `rational_integrate` | Legacy polynomial calculus, full rational primitives and certified proper finite integrals |
| Real elementary analysis | `real_function`, `parse_function`, `function_differentiate`, `function_partial_diff`, `function_gradient`, `function_jacobian`, `function_hessian`, `taylor`, `function_evaluate`, `function_antiderivative`, `function_integrate`, `function_limit` | Separate domain-aware AST; exact derivatives, finite Taylor polynomials, bounded integration/limit rules |
| Complex algebra | `complex_algebraic`, `complex_add`, `complex_sub`, `complex_mul`, `complex_div`, `complex_pow`, `complex_conjugate`, `complex_real`, `complex_imag`, `complex_norm_squared`, `complex_sqrt`, `quadratic_roots` | Exact real algebraic components; principal square root; all rational quadratic roots |
| Integer arithmetic | `integer_gcd`, `integer_lcm`, `integer_xgcd`, `mod_inverse`, `crt`, `is_prime`, `factor_integer`, `divisors`, `totient` | Deterministic bounded algorithms and independently checked prime certificates |
| Combinatorics | `factorial`, `binomial`, `multinomial`, `permutations_count`, `fibonacci`, `catalan`, `stirling` | Exact integer counts; second-kind Stirling numbers |
| Probability | `bernoulli`, `binomial_distribution`, `discrete_uniform`, `hypergeometric`, `probability`, `condition`, `cdf`, `moment`, `expectation`, `distribution_variance` | Finite rational outcomes and normalized rational probabilities |
| Statistics | `mean`, `median`, `variance`, `stddev`, `covariance` | Rational data; explicit `ddof`; exact real square roots |
| Function analysis | `limit`, `analyze_function`, `extrema_values` | Real univariate rational functions; real algebraic finite limit points and signed infinity |
| Linear algebra | `rref`, `rank`, `det`, `inverse`, `nullspace`, `solve_system`, `charpoly`, `real_eigenvalues`, `trace`, `rowspace`, `columnspace`, `lu`, `least_squares` | Rational matrices; exact real spectrum; `P*A=L*U`; all least-squares minimizers |
| Verification | `verify` | Independent algorithmic checks; not a formal proof system |
| Transport/storage | `to_data`, `from_data`, `dumps`, `loads`, `Workspace` | Strict v1–v5 mathematical JSON, content references, explicit history |
| Input/output | `parse_expression`, `parse_problem`, `render_text`, `to_latex`, `capabilities` | Bounded front-end helpers; no new solving algorithms |

The structured counterpart of `diff` is `operations.differentiate`. Division/GCD
operations accept `Polynomial` operands, not arbitrary strings. Parsing is an
explicit first step, and renderers accept mathematical values, not raw LaTeX.

See the [multivariate guide](multivariate.md) for derivative orders, matrix shapes,
variable identity and the difference between `partial_diff` and `diff`.
See [exact algebra](exact-algebra.md) for the v0.6 operations, immutable records,
factor conventions and reusable root values. Every operation accepts `limits`.
See the [v0.7 guide](v07-mathematics.md) for elementary functions, discrete
mathematics and new records. `operations` counterparts always return `Result`;
the existing calculus entry points also dispatch `RealFunction`/`RealExpr` inputs.

## Compute a limit or analyze a function

`limit(expression, variable, point, *, side="both", limits=None)` returns a
`LimitValue`; `operations.limit` returns `Result[LimitValue]`. Expressions may be
integers, rational-coefficient polynomial expressions, `Polynomial` values or
`RationalFunction` values. Finite points are Python integers, `Rational` values,
`RealAlgebraicRoot` values or supported closed real algebraic expressions;
use the exact strings `"-inf"` and `"+inf"` for infinity. `side` is `"left"`,
`"right"` or `"both"`; infinity requires `"both"`.

`analyze_function(expression, variable, *, limits=None)` returns a
`FunctionAnalysis`; `operations.analyze_function` returns
`Result[FunctionAnalysis]`. Both operations require an unconstrained real
variable and use the original rational-function domain, including canceled holes.

| Record | Fields | Meaning |
| --- | --- | --- |
| `LimitValue` | `kind`, `value` | `finite` carries a `Rational` or `RealAlgebraicRoot`; other kinds carry `None` |
| `ExtremumValue` | `point`, `value`, `kind` | Exact x and y coordinates of a strict local `minimum` or `maximum` |
| `ExcludedPoint` | `point`, `kind` | Exact rational/algebraic x-coordinate classified as `hole` or `pole` |
| `MonotonicRegion` | `interval`, `behavior` | Maximal connected `Interval`, with behavior `increasing`, `decreasing` or `constant` |
| `LocalExtremum` | `point`, `kind` | Strict local `minimum` or `maximum`; x-coordinate only |
| `FunctionAnalysis` | `function`, `derivative` | Domain-preserving `RationalFunction` values |
| `FunctionAnalysis` | `domain`, `zero_set`, `positive_set`, `negative_set`, `stationary_set` | Exact `IntervalSet` values, including singleton intervals or an empty set |
| `FunctionAnalysis` | `excluded_points`, `monotonic_regions`, `strict_extrema` | Tuples of the corresponding frozen records |

```python
x = mf.symbol("x")
pole = mf.rational_function(1, x, variable=x)
no_limit = mf.operations.limit(pole, x, 0)
assert no_limit.execution_status is mf.ExecutionStatus.COMPLETED
assert no_limit.outcome is mf.Outcome.VALUE
assert no_limit.unwrap() == mf.LimitValue("does_not_exist")
assert mf.verify(no_limit).verified
assert mf.limit(pole, x, 0, side="right").kind == "positive_infinity"
```

See [limits and function analysis](function-analysis.md) for endpoint semantics,
constant functions and an example with both a hole and a pole. These APIs do not
return plots or decimal approximations. Use the separate `extrema_values` API for
exact y-coordinates; `FunctionAnalysis` retains its original record structure.

## Typed values and errors

`Result[T].value` is `T | None`; `unwrap()` returns `T` or raises `OperationError`
carrying the original result. Checking a status enum does not automatically narrow
`value` for type checkers. Heterogeneous requests and certificate records retain
their existing general-purpose field types.

```python
x = mf.symbol("x")
bad = mf.operations.evaluate(x)
assert bad.execution_status is mf.ExecutionStatus.INVALID_INPUT
assert bad.value is None
assert bad.error is not None
try:
    bad.unwrap()
except mf.OperationError as error:
    assert error.result is bad
```

`unsupported` indicates scope limits, `invalid_input` indicates invalid arguments,
and `error` includes `resource_limit_exceeded`. None of these is `no_solution`.
Use the result's exactness, completeness and fresh verification report together.
An unanswered elementary integral or limit returns a completed `NO_CONCLUSION`
with `deferred`, `value=None`, `completeness=UNKNOWN` and no solved claim.
`unwrap()` raises and `verify()` does not certify it as a value. Resource errors
and invalid input are never converted to this form.
Represented expressions need not be supported by every algorithm: general
complex polynomial roots, symbolic matrix entries and unrestricted elementary
integration are not provided. Real algebraic normalization is explicit: legacy
`evaluate`, `substitute` and expression constructors retain their existing rules.
