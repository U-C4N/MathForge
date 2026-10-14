# Differentiate a multivariate polynomial

MathForge v0.5 expands and differentiates polynomials with exact rational
coefficients in several real variables. A sparse internal representation stores
only nonzero monomials. Public values remain ordinary expressions and immutable
tuples, so the existing evaluator, text/LaTeX renderers and JSON codec apply.

## Compute a gradient and Hessian

```python
import mathforge as mf

x, y = mf.symbol("x"), mf.symbol("y")
f = mf.parse_expression("x^2*y + y^3", symbols={"x": x, "y": y})
assert mf.partial_diff(f, x) == 2*x*y
assert mf.partial_diff(f, y, n=2) == 6*y
assert mf.gradient(f, (x, y)) == (2*x*y, x**2+3*y**2)
assert mf.hessian(f, (x, y)) == ((2*y, 2*x), (2*x, 6*y))
```

`partial_diff` holds every other symbol fixed. Its nonnegative integer order
defaults to one; order zero produces the expanded polynomial. Unlike `diff`, it
accepts multiple variables and is restricted to polynomial expressions.

For `gradient`, `hessian` and `expand_multivariate`, provide a nonempty ordered
iterable of distinct unconstrained real symbols containing all free symbols in
the expression. Extra axes are allowed and yield zero derivatives. Symbol UUIDs,
not display names, determine identity. Duplicate identities and conflicting
metadata are rejected. Reordering axes reorders the answer.

## Compute a Jacobian or evaluate an expanded expression

```python
assert mf.jacobian((f, x*y), (x, y)) == ((2*x*y, x**2+3*y**2), (y, x))
assert mf.gradient(f, (y, x)) == (x**2+3*y**2, 2*x*y)
expanded = mf.expand_multivariate((x+y)**3, (x, y))
assert mf.evaluate(expanded, {x: 2, y: 3}) == mf.Rational(125)
```

Jacobian rows follow the nonempty input function sequence; columns follow the
variable sequence. The Hessian entry `[i][j]` differentiates with respect to axes
`i` and `j`. Both matrices are tuples of expression tuples. `Matrix` continues to
accept rational numbers only; these APIs do not add symbolic matrix inversion.

| Function | Convenience return | Structured counterpart |
| --- | --- | --- |
| `expand_multivariate(expression, variables)` | `Expression` | `operations.expand_multivariate` |
| `partial_diff(expression, variable, n=1)` | `Expression` | `operations.partial_diff` |
| `gradient(expression, variables)` | `tuple[Expression, ...]` | `operations.gradient` |
| `jacobian(expressions, variables)` | `tuple[tuple[Expression, ...], ...]` | `operations.jacobian` |
| `hessian(expression, variables)` | `tuple[tuple[Expression, ...], ...]` | `operations.hessian` |

Every entry accepts keyword-only `limits=ComputationLimits(...)`. Structured
counterparts return `Result[T]`; convenience functions unwrap or raise
`OperationError`. Supported inputs are expression objects or Python integers,
not raw strings or `Polynomial`/`RationalFunction` containers. Parse text first.

## Check and store a derivative claim

```python
result = mf.operations.hessian(f, (x, y))
assert result.execution_status is mf.ExecutionStatus.COMPLETED
assert mf.verify(result).verified
loaded = mf.loads(mf.dumps(result))
assert mf.verify(loaded).verified
assert mf.to_data(loaded)["version"] == 2
print(mf.render_text(loaded.unwrap()))
print(mf.to_latex(loaded.unwrap()))
```

The checker compares exact polynomial coefficients and derivative identities
without calling the producing operation. Source expressions, variable order,
derivative order and value shape are part of the claim. This is algorithmic
verification, not formal proof. New requests use the existing generic v2
operation envelope; older implementations may decode them but cannot verify
the new operation names. Historical object bytes and hashes remain unchanged.

## Bound expansion and derivative costs

Gradient and Hessian production expand the input once. Hessians reuse symmetric
mixed partials; output still contains every entry. Expansion uses sparse
convolution and exponentiation by squaring. A dense expansion can still contain
many monomials, so finite work, node, integer-size and matrix-entry budgets apply.

```python
limited = mf.operations.hessian((x+y)**20, (x, y),
    limits=mf.ComputationLimits(max_work=10))
assert limited.error.code == "resource_limit_exceeded"
assert limited.value is None
```

Nonrational coefficients, transcendental functions, symbolic denominators,
general multivariate equation solving and optimization are outside this release.
See the [executable workflow](../examples/v05_workflow.py) and
[MCP guide](mcp.md) for the same operations through an agent.
