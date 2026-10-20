<p align="center">
  <img src="assets/mark.svg" width="112" height="112" alt="">
</p>

<h1 align="center">MathForge</h1>

<p align="center">
  Exact mathematics. Built from first principles.<br>
  <sub>By Umutcan EDIZASLAN</sub>
</p>

<p align="center">
  <sub>Python 3.11+ &nbsp;·&nbsp; Dependency-free core &nbsp;·&nbsp; v0.8.0 alpha</sub>
</p>

<p align="center">
  <a href="https://github.com/U-C4N/MathForge/releases">Releases</a> &nbsp; / &nbsp;
  <a href="CHANGELOG.md">Changes</a> &nbsp; / &nbsp;
  <a href="examples/v07_workflow.py">Example</a> &nbsp; / &nbsp;
  <a href="LICENSE">MIT license</a>
</p>

<br>

## New in v0.8

- **Areas and one registry:** every operation belongs to one of 18 mathematical areas, such as `mf.ntheory` or `mf.linalg`. Each area module loads on first use, and the Python API is generated from a single operation registry.
- **Mathematical JSON v6:** operations added in v0.8 use v6, with a generic `Unevaluated` record for a request that reaches no verified answer. v1–v5 documents keep their bytes and get the same verdicts.
- **MCP area tools:** typed tools run a named operation of an area, and `mathforge.catalog` finds operations. The thirty v0.7 tools are unchanged.
- **Continued fractions:** `mf.ntheory.continued_fraction` returns the finite simple continued fraction of an exact rational, confirmed by an independent checker.
- **Cheaper exact computation:** `Rational` fast paths, exact integer kernels with independent checkers, a lean value path that calls a producer and its checker directly, and a structured pipeline with a lower fixed cost. Values match v0.7.

```python
import mathforge as mf

cf = mf.ntheory.continued_fraction(mf.Rational(415, 93))
assert cf.terms == (4, 2, 6, 7)
result = mf.operations.continued_fraction(mf.Rational(415, 93))
assert mf.to_data(result)["version"] == 6
assert mf.verify(mf.loads(mf.dumps(result))).verified
assert mf.ntheory.integer_gcd(12, 18) == 6
```

See the [area guide](documentation/areas.md) for the modules and names, the
[MCP guide](documentation/mcp.md) for the area tools and
[performance notes](documentation/performance.md) for the cheaper computation
path. Every v0.7 operation keeps its name, signature and result. Some inputs just
past a resource limit may now succeed, because exact computation got cheaper: a
value function may answer where `mathforge.operations` reports the limit, never
with a different value. Some verifier checks that ran out of budget in v0.7 now
complete.

Known limitations: `integer_gcd` and `integer_lcm` on very large integers, from
about 2,000 bits (RSA size) upward, remain comparatively slow, because every
result carries an independently checked Bézout certificate. `solve_system` still
uses the v0.7 structured pipeline and will be accelerated in a later version.

## Added in v0.7

- **Real elementary functions:** explicit domain-preserving functions, exact constants, repeated and multivariate derivatives, Taylor polynomials and certified limit rules.
- **Exact integration:** Hermite reduction and Lazard–Rioboo–Trager primitives for rational functions, a bounded elementary catalogue and proper finite integrals with whole-interval domain checks.
- **Complex algebra:** exact algebraic real/imaginary components, arithmetic, principal square roots and all rational quadratic roots.
- **Discrete mathematics:** certified integer arithmetic, generalized CRT, factorization, primality certificates, combinatorial counts and sequences.
- **Probability and linear algebra:** finite rational distributions, exact descriptive statistics, LU, subspaces and the complete least-squares minimizer family.
- **Connected evidence:** independent checks, mathematical JSON v5 and thirty local MCP tools. Unanswered integrals and limits carry explicit deferred requests.

```python
import mathforge as mf
from mathforge import functions as fn

x = mf.symbol("x")
f = mf.real_function(fn.sin(x)/fn.as_real_expr(x), (x,))
result = mf.operations.function_limit(f, x, 0)
assert result.unwrap().value == fn.as_real_expr(1)
assert mf.verify(mf.loads(mf.dumps(result))).verified
assert mf.expectation(mf.discrete_uniform((1, 2, 3, 4, 5, 6))) == mf.Rational(7, 2)
assert mf.complex_pow(mf.ComplexAlgebraic(0, 1), 2) == mf.ComplexAlgebraic(-1, 0)
```

See the [v0.7 mathematics guide](documentation/v07-mathematics.md) for domains,
supported families and API examples. A formal `RationalPrimitive` certifies an
antiderivative identity; general `LogRootSum` point evaluation is outside scope.
Improper integrals, general complex root isolation, symbolic matrices and
approximate numerical computation remain future work.

## Added in v0.6

- **Exact polynomial algebra:** complete irreducible factorization over the rationals, resultants, discriminants and partial fractions that preserve excluded points.
- **Reusable algebraic numbers:** explicit normalization, division, integer powers, minimal polynomials and evaluation using real algebraic roots.
- **Connected calculus:** rational-function limits at algebraic points, polynomial integrals with algebraic endpoints and exact extrema coordinates.
- **Spectral algebra:** rational characteristic polynomials and every real eigenvalue with algebraic multiplicity.
- **Checked end to end:** independent certificate checks, mathematical JSON v4, historical byte/hash compatibility and twenty-eight optional local MCP tools.

```python
import mathforge as mf

x = mf.symbol("x")
factors = mf.factor(mf.Polynomial.from_expression(x**4 + 4, x))
assert len(factors.factors) == 2
alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
assert mf.algebraic(alpha**3) == mf.Rational(2)
assert mf.evaluate_algebraic(x**3, {x: alpha}) == mf.Rational(2)
assert mf.integrate(x, x, 0, mf.sqrt(2)) == mf.Rational(1)
assert mf.verify(mf.operations.charpoly(mf.Matrix(((0, 2), (1, 0))), x)).verified
```

Read the [exact algebra guide](documentation/exact-algebra.md) for polynomial
domains, certificates and explicit real algebraic arithmetic.

## Added in v0.5

- **Multivariate calculus:** sparse rational-polynomial expansion, partial derivatives, gradients, Jacobians and Hessians with explicit variable order.
- **Less repeated arithmetic:** Sturm isolation carries endpoint signs between subdivisions; RREF skips known zero prefixes while retaining replayable row evidence.
- **Checked results:** independent coefficient checks, exact text/LaTeX and sixteen optional local MCP tools.
- **Compatible values:** new derivatives use existing expressions, immutable tuples and mathematical JSON v2 requests; no new wire types.

See the [multivariate guide](documentation/multivariate.md) and
[measured performance](documentation/performance.md). Local three-run medians
showed 2.37x faster degree-20 roots and 1.50x faster 32-by-32 RREF; timings depend
on the machine and input, and are not performance guarantees.

## Added in v0.4

- **Rational limits:** exact finite, one-sided and infinite limits; a nonexistent limit is a completed mathematical result.
- **Function analysis:** domain, holes and poles, signs, stationary points, maximal monotonic intervals and strict local extrema locations.
- **Checked results:** independent algorithmic checks and exact text/LaTeX.
- **Additive transport:** new analysis records use mathematical JSON v3; existing v1/v2 objects keep their bytes and content references.

MathForge implements its own exact arithmetic with finite computation budgets.
The core includes real polynomial roots, rational equations and inequalities,
exact linear algebra, multivariate polynomial calculus, and rational-function limits and
analysis. Verification is algorithmic, not
formal proof. General complex polynomial roots, unrestricted transcendental
calculus, approximate output and remote HTTP serving are outside this alpha.

## Quick start

From a local checkout:

```sh
python -m pip install -e .
```

```python
import mathforge as mf

x = mf.symbol("x", domain=mf.Reals)
problem = mf.parse_problem("x^2 - 2 = 0", symbols={"x": x})
result = mf.solve(problem, for_=x)

print(mf.render_text(result.solution_set))
print(mf.to_latex(result.solution_set))
assert mf.verify(result).verified
assert mf.parse_expression("0.1", symbols={}) == mf.Rational(1, 10)
```

## Exact mathematics, including excluded points

```python
roots = mf.real_roots(x**3 - x, x)  # -1, 0, 1 with multiplicities
rf = mf.rational_function(x**2 - 1, x - 1, variable=x)
assert mf.solve(mf.Eq(rf, 2), for_=x).outcome is mf.Outcome.NO_SOLUTION
assert mf.integrate(x**2, x, 0, 1) == mf.Rational(1, 3)
assert mf.det(mf.Matrix(((1, 2), (3, 4)))) == mf.Rational(-2)
```

[Full workflow](examples/v02_workflow.py) · [Linear systems](examples/linear_system.py) ·
[Rational inequalities](examples/rational_inequality.py)

## Differentiate in several variables

```python
y = mf.symbol("y")
f = x**2*y + y**3
assert mf.gradient(f, (x, y)) == (2*x*y, x**2 + 3*y**2)
assert mf.hessian(f, (x, y)) == ((2*y, 2*x), (2*x, 6*y))
assert mf.verify(mf.operations.jacobian((f, x*y), (x, y))).verified
```

Jacobian and Hessian values are immutable tuples of expression rows. The
existing `Matrix` type remains rational-valued.

## Analyze a function with a hole and a pole

```python
f = mf.parse_expression("((x-1)*(x^2+2))/(x*(x-1))", symbols={"x": x})
analysis_result = mf.operations.analyze_function(f, x)
analysis = analysis_result.unwrap()
assert not analysis.domain.contains(0) and not analysis.domain.contains(1)
assert [(p.point, p.kind) for p in analysis.excluded_points] == [
    (mf.Rational(0), "pole"), (mf.Rational(1), "hole")]
assert mf.limit(f, x, 1) == mf.LimitValue("finite", mf.Rational(3))
assert mf.limit(f, x, 0).kind == "does_not_exist"
assert mf.verify(analysis_result).verified
print(mf.render_text(analysis))
```

The hole at `x=1` remains excluded even though its limit is `3`. The two-sided
limit at `x=0` does not exist because its one-sided limits have opposite signs.
See the [function analysis guide](documentation/function-analysis.md) and the
[complete executable example](examples/v04_workflow.py).

## Connect an agent locally

Install the optional adapter into the environment used by your MCP client:

```sh
python -m pip install -e ".[mcp]"
mathforge-mcp
```

The process speaks MCP on stdin/stdout, not an interactive terminal prompt. The
base install has no runtime dependencies; the `mcp` extra installs the official
MCP SDK and its dependencies. There is no API key or HTTP server.

See [client configuration and tool inputs](documentation/mcp.md),
[the user guide](documentation/index.md), and [API reference](documentation/api.md).

## Development

```sh
python -m unittest discover -s tests -v
python tools/check_docs.py
python -m pip install -r requirements-dev.txt
python -m build
python tools/check_distribution.py
```

Core tests use Python's standard library. Protocol tests use the optional MCP
SDK. Build, schema, typing and the seeded SymPy comparison suite are development
dependencies. Distribution
checks validate both base and MCP-extra installations. Public `documentation/`
is tracked and included in the sdist; `docs/` remains local working material and
is excluded from Git and release packages.
