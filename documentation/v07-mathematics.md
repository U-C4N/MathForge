# Using the v0.7 mathematics modules

MathForge v0.7 adds exact integer arithmetic, counting, finite probability, descriptive statistics, complex algebraic scalars, linear algebra extensions and domain-aware elementary calculus. The core still requires Python 3.11+ and has no runtime dependencies. The optional MCP server requires `mathforge[mcp]`.

**A completed request can remain mathematically unanswered.** Inspect the structured result's `outcome` and `deferred` fields before treating an integral or limit as solved. Logarithmic root sums are formal antiderivatives; they are not arbitrary pointwise complex-log evaluators. No approximate floating-point result is substituted for an exact answer.

The previous expression model and successful v1–v4 encodings remain available. Elementary real functions use the separate `RealFunction` model, while complex arithmetic uses `ComplexAlgebraic`.

## Start with a complete workflow

Run the tested [v0.7 workflow](../examples/v07_workflow.py) after installing the package:

```console
python -m pip install -e .
python examples/v07_workflow.py
```

Use convenience functions for values and `mf.operations` for source-bearing results:

```python
import mathforge as mf

assert mf.integer_gcd(84, 30) == 6
result = mf.operations.integer_gcd(84, 30)
assert result.value == 6
assert mf.verify(result).verified
assert mf.loads(mf.dumps(result)) == result
```

The structured form is useful when preserving evidence, inspecting a resource failure or passing a result between processes. The value-returning wrapper calls `unwrap()` and raises if there is no mathematical value.

## Work with integers and counts

| Family | Public operations | Input rules |
| --- | --- | --- |
| Integer arithmetic | `integer_gcd`, `integer_lcm`, `integer_xgcd` | Exact Python integers; Bézout coefficients accompany xgcd |
| Congruences | `mod_inverse`, `crt` | Positive moduli; CRT accepts `(residue, modulus)` pairs, including noncoprime moduli |
| Primes and divisors | `is_prime`, `factor_integer`, `divisors`, `totient` | Deterministic bounded algorithms and checked prime evidence; no probable-prime success label |
| Counting | `factorial`, `binomial`, `multinomial`, `permutations_count` | Nonnegative integer indices; multinomial takes category counts |
| Sequences | `fibonacci`, `catalan`, `stirling` | Nonnegative indices; Stirling numbers are of the second kind |

`CRTResult(residue=None, modulus=None)` represents an inconsistent congruence system. Resource exhaustion is a separate execution failure. Large exact integers are representable, but that does not promise that every factorization or prime certificate fits the configured work budget.

## Compute finite probabilities and statistics

`FiniteDistribution(outcomes, probabilities)` stores rational outcomes and weights. Construction aggregates duplicate outcomes, removes zero weights and sorts the remaining support. Every weight must lie in `[0, 1]`, and the sum must be exactly one. Inputs such as `0.1` Python floats are rejected; use `Rational(1, 10)`.

```python
import mathforge as mf

Q = mf.Rational
trials = mf.binomial_distribution(4, Q(1, 2))
assert mf.probability(trials, (3, 4)) == Q(5, 16)
assert mf.expectation(trials) == Q(2)
assert mf.distribution_variance(trials) == Q(1)

selected = mf.condition(trials, (3, 4))
assert mf.probability(selected, (3, 4)) == Q(1)
assert mf.variance((1, 2, 4), ddof=1) == Q(7, 3)
```

Factories are `bernoulli(p)`, `binomial_distribution(n, p)`, `discrete_uniform(outcomes)` and `hypergeometric(population, successes, draws)`. Uniform outcomes must be distinct. Events are finite collections of outcomes, never executable predicates; repeated event values do not duplicate probability. An outcome outside the support contributes zero. Conditioning on a probability-zero event is invalid.

`cdf(distribution, x)` includes outcomes equal to `x`. `moment(distribution, order)` returns a raw moment for a nonnegative integer order. `expectation` and `distribution_variance` return exact rationals.

For nonempty rational samples, use `mean`, `median`, `variance`, `stddev` and paired `covariance`. The default `ddof=0` uses population normalization; `ddof=1` uses sample normalization. Require `0 <= ddof < sample_size`. Standard deviation may be an exact radical. There are no continuous distributions, hypothesis tests, random sampling or implicit independence assumptions in this release.

## Preserve all least-squares solutions

`rowspace(A)` returns an RREF row basis. `columnspace(A)` returns the corresponding original pivot columns. `trace(A)` requires a square matrix, as does `lu(A)`. The PLU convention is **`P @ A == L @ U`** with unit lower-triangular `L`; singular and empty square matrices are supported.

`least_squares(A, b)` returns every minimizer as

`particular + t_1*basis[0] + ... + t_k*basis[k-1]`.

Its `residual` is `b - A @ particular`. A rank-deficient system therefore retains its free directions rather than returning an arbitrary unique answer. The checker verifies residual orthogonality, nullspace membership, basis independence and completeness.

```python
import mathforge as mf

A = mf.Matrix(((1, 1), (2, 2)))
fit = mf.least_squares(A, (1, 3))
assert fit.particular == (mf.Rational(7, 5), mf.Rational(0))
assert fit.basis == ((mf.Rational(-1), mf.Rational(1)),)
assert fit.residual == (mf.Rational(-2, 5), mf.Rational(1, 5))
```

Matrix entries remain rational. Algebraic-coefficient matrices, SVD, general Jordan forms and complex eigenspaces are outside this extension.

## Use exact complex scalars

`complex_algebraic(real, imag=0)` normalizes two real algebraic components. Use `complex_add`, `complex_sub`, `complex_mul`, `complex_div`, `complex_pow`, `complex_conjugate`, `complex_real`, `complex_imag`, `complex_norm_squared` and `complex_sqrt`. Division checks a nonzero denominator, and signed integer powers remain bounded.

`quadratic_roots(a, b, c)` returns exact complex root records with multiplicities. This API does not provide arbitrary-degree complex root isolation, complex transcendental calculus or an ordering on complex numbers. The historical real `sqrt` semantics do not change.

## Keep elementary function domains visible

```python
import mathforge as mf

x = mf.Symbol("x")
f = mf.parse_function("log(x) + sin(x)", symbols={"x": x})
assert f.contains({x: -1}) is mf.Truth.FALSE
assert f.contains({x: 1}) is mf.Truth.TRUE
derivative = mf.operations.function_differentiate(f, x)
assert mf.verify(derivative).verified
```

`parse_function` requires an explicit symbol mapping. Its optional `variables` argument defines coordinate order; otherwise mapping order is used. The parser supports exact decimal literals, integer powers, arithmetic, `pi`, `E`, and `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`. Attribute access, imports, comprehensions and arbitrary calls are not part of the grammar.

`RealFunction` retains denominator exclusions and the real domains of logarithms, radicals and inverse trigonometric functions before simplification. An expression such as `0*log(x)` does not become defined at negative `x` merely because its value simplifies to zero. Domain membership returns `TRUE`, `FALSE` or `UNKNOWN`; an unknown sign is not assumed positive. Derivatives may carry sufficient smoothness restrictions rather than a claimed maximal domain.

| Task | Public operation | Supported behavior |
| --- | --- | --- |
| Differentiate | `function_differentiate`, `function_partial_diff` | Repeated exact chain, product and quotient rules with preserved domains |
| Multivariable calculus | `function_gradient`, `function_jacobian`, `function_hessian` | Explicit ordered axes determine coordinates |
| Evaluate | `function_evaluate` | Complete exact bindings and certified domain membership |
| Taylor polynomial | `taylor` | Univariate, finite real algebraic center, certified regularity through the requested order |
| Elementary primitive | `function_antiderivative` | Documented elementary rule catalogue; unmatched requests are deferred |
| Definite integral | `function_integrate` | Supported real primitives and certified finite intervals; otherwise deferred |
| Limit | `function_limit` | Directional finite-point rules and supported exact asymptotic families; otherwise deferred |

Taylor order `n` includes coefficients through degree `n`. `TaylorPolynomial` does not claim an error bound, radius of convergence or a convergent infinite series.

The elementary integration catalogue includes linearity, polynomials, affine elementary arguments, polynomial multiples of affine `exp/sin/cos/log`, and exact `u' F(u)` matches for rational `u`. It is not a full Risch decision procedure. Local limit analysis uses certified leading coefficients through order 12; infinity rules cover rational functions, selected compositions and finite polynomial/exponential/logarithmic scales. Bounded-factor squeeze and affine trigonometric oscillation rules distinguish `sin(x)/x -> 0` from the nonexistent limit of `sin(x)` at infinity. Reciprocal substitution handles supported directional endpoint limits such as `x*log(x) -> 0` at `0+`. Unrestricted asymptotics and multivariable limits are not promised.

## Integrate rational functions without expanding every root

`rational_antiderivative(expression, variable)` accepts a univariate rational function over `Q`. It returns a `RationalPrimitive` containing its source, rational part, logarithmic terms and retained domain. Hermite reduction removes repeated denominator factors; LRT records encode the remaining logarithmic contribution. The independent checker differentiates the representation using exact quotient-algebra traces.

| Record | Fields and meaning |
| --- | --- |
| `RationalPrimitive` | `source`, `rational_part`, `log_terms`, `elementary_terms`, `domain`; the domain retains original exclusions, including canceled factors and restrictions inherited from a `RealFunction` |
| `LogRootSum` | `residue_polynomial`, `logarithm_coefficients`; a formal sum over algebraic residues, with locally chosen logarithm branches |

A root-sum term represents a sum of algebraic residues times logarithms **modulo locally constant branch choices**. It certifies a formal primitive. It does not define the principal complex logarithm at arbitrary real points or assert a single global branch across poles.

`primitive.as_function()` converts residual linear/quadratic real factors to ordinary real log/atan expressions. If a higher-degree residual prevents this conversion, the formal root sum remains exact and `.as_function()` raises `UnsupportedOperation`. `rational_integrate` evaluates supported proper finite integrals; a pole, an unproved domain condition or an unavailable pointwise primitive leaves an `UnevaluatedIntegral`. Do not infer an endpoint value directly from a formal root-sum display.

## Handle unanswered requests explicitly

```python
import mathforge as mf

x = mf.Symbol("x")
f = mf.parse_function("exp(x*x)", symbols={"x": x})
result = mf.operations.function_antiderivative(f, x)
assert result.execution_status is mf.ExecutionStatus.COMPLETED
assert result.outcome is mf.Outcome.NO_CONCLUSION
assert result.value is None
assert isinstance(result.deferred, mf.UnevaluatedIntegral)
```

A deferred result has `completeness=UNKNOWN`, `exactness=NOT_APPLICABLE`, and no successful mathematical verification claim. It does not assert divergence, nonexistence of a limit or nonexistence of an elementary primitive. Invalid inputs, unsupported operation contracts and exhausted resource budgets remain distinct execution outcomes.

## Chain canonical data through MCP

The 28 existing tools remain available. Two additions make 30 curated tools:

- `mathforge.parse_function`: `text`, ordered `variables`, optional `symbol_ids` and `limits`; returns a canonical `RealFunction` object.
- `mathforge.compute`: a published `operation` name, an `arguments` mapping, and optional `limits`; returns the same structured result as the Python SDK.

Every compute argument is an individual canonical object envelope, including integers, strings, tuples and `None`. Pass all required arguments, including default values such as `ddof=0`, `n=1` or `side="both"`. `capabilities()` lists argument names in each operation's scope, and the compute schema enumerates every allowed operation. The registry name for `taylor` is `function_taylor`.

```python
import mathforge as mf
from mathforge.mcp.boundary import call_tool

response = call_tool("mathforge.compute", {
    "operation": "integer_gcd",
    "arguments": {"a": mf.to_data(84), "b": mf.to_data(30)},
})
assert not response.is_error
result = mf.from_data(response.payload["result"])
assert result.value == 6
assert mf.verify(result).verified
```

For a second computation, encode `result.value` as the next operand. Result, workspace and execution-metadata envelopes are rejected as operands, including inside tuples. All operands share one symbol-identity decoding context, so a UUID cannot silently acquire conflicting metadata across arguments. `mathforge.verify` accepts canonical v1–v5 Result envelopes and rechecks claims without trusting stored labels. Deferred requests are transported intact.

The adapter is stateless and stdio-only. It exposes no filesystem operation, network transport, arbitrary Python evaluator or unrestricted command dispatcher. Per-call limits can only be reduced through MCP. Oversized responses fail without truncating a certificate.

## Validate and extend the coverage

The executable workflow and focused tests provide the evidence for these guarantees:

| Evidence | Verified behavior | Implementation path |
| --- | --- | --- |
| [Probability tests](../tests/test_probability_v7.py) and [statistics tests](../tests/test_statistics_v7.py) | Exact definitions, invalid inputs, modified values and producer-independent checks | [probability.py](../src/mathforge/probability.py), [statistics.py](../src/mathforge/statistics.py) |
| [Linear algebra tests](../tests/test_linear_extensions_v7.py) | Singular PLU, empty shapes, complete rank-deficient least squares | [linear_extensions.py](../src/mathforge/linear_extensions.py) |
| [MCP tests](../tests/test_mcp_v7.py) | Canonical chaining, deferred outcomes, malformed operands, budgets and v5 verification | [MCP boundary](../src/mathforge/mcp/boundary.py) |

Nonlinear polynomial systems, general complex root isolation, continuous probability, numerical solvers, ODE/PDE, special functions, geometric objects and graph algorithms remain future topic families. Consult [capabilities](../src/mathforge/capabilities.py) for the executable coverage inventory rather than treating representability as a promise of a solver.
