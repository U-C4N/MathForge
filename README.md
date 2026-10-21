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
  <a href="documentation/index.md">Guide</a> &nbsp; / &nbsp;
  <a href="CHANGELOG.md">Changes</a> &nbsp; / &nbsp;
  <a href="examples/v07_workflow.py">Example</a> &nbsp; / &nbsp;
  <a href="LICENSE">MIT license</a>
</p>

<br>

MathForge is a pure-Python library for exact mathematics. It computes with
integers, rationals and exact algebraic numbers, never with floating point, and
an independent checker confirms each answer before you see it. The evidence
travels with the result, so anyone can check it again later.

- **Exact.** Rationals, real algebraic numbers, exact complex numbers and exact
  interval sets. Even `0.1` parses as `1/10`.
- **Independently checked.** Each operation pairs a producer with an independent
  checker, and a value is returned only after the checker accepts it.
- **Re-verifiable.** Structured results carry their request and evidence as
  mathematical JSON. `mf.verify` checks a stored document again, without trusting
  the code that produced it.
- **Bounded.** Every computation runs under a budget. Running out is an explicit
  error, never a wrong answer or a false "no solution".
- **Dependency-free.** The core is plain Python 3.11–3.14. An optional MCP server
  lets AI agents use it.

## New in v0.8.0: the speed release

v0.8 makes exact computation many times cheaper without skipping a single
check. Here are the same twelve calls, timed with v0.7.0 and v0.8.0:

| Call | Input | v0.7.0 | v0.8.0 | Faster |
| --- | --- | ---: | ---: | ---: |
| `mf.rref` | 16 × 16 integer matrix | 117 ms | 306 µs | **383×** |
| `mf.det` | 8 × 8 integer matrix | 5.4 ms | 114 µs | **48×** |
| `mf.real_roots` | x⁵ − 4x + 2 | 63 ms | 1.3 ms | **50×** |
| `mf.factor` | x¹² − 1 | 21 ms | 1.1 ms | **20×** |
| `mf.evaluate_algebraic` | x³ + 2x at x = ∛2 | 47 ms | 5.1 ms | **9.3×** |
| `mf.factor_integer` | 2³² + 1 | 2.1 ms | 55 µs | **37×** |
| `mf.integer_gcd` | 1071 and 462 | 238 µs | 2.8 µs | **85×** |
| `mf.binomial` | C(100, 50) | 317 µs | 6.6 µs | **48×** |
| `mf.complex_sqrt` | √(3 + 4i) | 2.7 ms | 32 µs | **84×** |
| `mf.expectation` | Binomial(20, 1/3) | 2.1 ms | 222 µs | **9.3×** |
| `mf.median` | 101 integers | 15 ms | 35 µs | **427×** |
| `mf.taylor` | sin(x)·eˣ to order 6 | 152 ms | 23 ms | **6.6×** |

Each figure is the median of three alternating rounds of nine calls on one
Windows machine with Python 3.14.6. MathForge's result caches were cleared
before every call, and both versions check every answer they return. Timings
depend on the machine and the input; they are not guarantees.

What changed:

- **Exact kernels with their own checkers.** Integer linear algebra, dense integer
  polynomials, an integer kernel for factorization and primality, and complex and
  quadratic-surd arithmetic. Separate checkers confirm their answers without
  rerunning the producer: modular certificates for determinants and echelon
  forms, prime certificates, truncated power series for Taylor coefficients and
  a counting certificate for medians.
- **A lean value path.** A value function such as `mf.det` calls the kernel and
  its checker directly and builds no provenance record. Any call it does not
  handle takes the full structured path, so values and errors stay the same.
- **Cheaper everything else.** `Rational` fast paths, an inline computation
  budget, and a structured pipeline, `mathforge.operations`, that inspects its
  inputs once. Records and JSON bytes are unchanged.
- **Area modules.** Operations are grouped into modules such as `mf.ntheory`,
  `mf.linalg` and `mf.stats`, which load on first use. The Python API and the MCP
  tools are generated from one operation registry.
- **Mathematical JSON v6.** New operations use v6, with a generic `Unevaluated`
  record for a question that has no verified answer.
- **MCP area tools.** Agents find any operation with `mathforge.catalog` and run
  it through an area tool with typed arguments. The thirty v0.7 tools are
  unchanged.
- **Continued fractions.** `mf.ntheory.continued_fraction` is the first v6
  operation.

```python
import mathforge as mf

cf = mf.ntheory.continued_fraction(mf.Rational(415, 93))
assert cf.terms == (4, 2, 6, 7)
result = mf.operations.continued_fraction(mf.Rational(415, 93))
assert mf.to_data(result)["version"] == 6
assert mf.verify(mf.loads(mf.dumps(result))).verified
assert mf.ntheory.integer_gcd(12, 18) == 6
```

Every v0.7 operation keeps its name, signature and result, and v1–v5 documents
keep their bytes and verdicts. Some inputs just past a resource limit may now
succeed, because exact computation got cheaper: a value function may answer
where `mathforge.operations` reports the limit, never with a different value.
Some verifier checks that ran out of budget in v0.7 now complete.

Known limitations: `integer_gcd` and `integer_lcm` on very large integers, from
about 2,000 bits (RSA size) upward, remain comparatively slow, because every
result carries an independently checked Bézout certificate. `solve_system` still
uses the v0.7 structured pipeline and will be accelerated in a later version, as
will `solve` on the smallest equations, where that pipeline's fixed cost dominates.

See the [area guide](documentation/areas.md), the [MCP guide](documentation/mcp.md)
and the [performance notes](documentation/performance.md).

## Quick start

Install the wheel from the [v0.8.0 release](https://github.com/U-C4N/MathForge/releases/tag/v0.8.0):

```sh
python -m pip install https://github.com/U-C4N/MathForge/releases/download/v0.8.0/mathforge-0.8.0-py3-none-any.whl
```

From a local checkout, use `python -m pip install -e .` instead. Then:

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

## A short tour

Every example in this README runs as written: `tools/check_docs.py` executes
them with their assertions.

### Exact answers, excluded points included

```python
roots = mf.real_roots(x**3 - x, x)  # -1, 0, 1 with multiplicities
rf = mf.rational_function(x**2 - 1, x - 1, variable=x)
assert mf.solve(mf.Eq(rf, 2), for_=x).outcome is mf.Outcome.NO_SOLUTION
assert mf.integrate(x**2, x, 0, 1) == mf.Rational(1, 3)
assert mf.det(mf.Matrix(((1, 2), (3, 4)))) == mf.Rational(-2)
```

[Full workflow](examples/v02_workflow.py) · [Linear systems](examples/linear_system.py) ·
[Rational inequalities](examples/rational_inequality.py)

### Differentiate in several variables

```python
y = mf.symbol("y")
f = x**2*y + y**3
assert mf.gradient(f, (x, y)) == (2*x*y, x**2 + 3*y**2)
assert mf.hessian(f, (x, y)) == ((2*y, 2*x), (2*x, 6*y))
assert mf.verify(mf.operations.jacobian((f, x*y), (x, y))).verified
```

Jacobian and Hessian values are immutable tuples of expression rows. The
existing `Matrix` type remains rational-valued. See the
[multivariate guide](documentation/multivariate.md).

### Analyze a function with a hole and a pole

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

### Factor polynomials and compute with algebraic numbers

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

### Elementary functions, complex numbers and probability

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
supported families and API examples.

## Connect an AI agent

MathForge includes an optional [MCP](https://modelcontextprotocol.io) server.
Install it into the environment your MCP client uses:

```sh
python -m pip install -e ".[mcp]"
mathforge-mcp
```

The process speaks MCP on stdin/stdout, not an interactive terminal prompt. It
offers 37 tools: the thirty curated tools of v0.7, area tools that run an
operation of their areas by name, and `mathforge.catalog`, which finds operations
and the arguments they take. The base install has no runtime dependencies; the
`mcp` extra installs the official MCP SDK and its dependencies. There is no API
key or HTTP server.

See [client configuration and tool inputs](documentation/mcp.md),
[the user guide](documentation/index.md), and [API reference](documentation/api.md).

## How answers are checked

Each operation has a producer, which computes, and a checker, which confirms.
A checker confirms the answer another way, usually by examining evidence
instead of repeating the computation: a Bézout identity for a gcd, a modular
certificate for a determinant, a prime certificate for a factorization,
truncated power series for a Taylor polynomial. The checkers in
`mathforge.certify` share no code with the producers beyond a small trusted
layer of arithmetic. CI runs `tools/check_layering.py`, which fails if a checker
could reach producer code, even through chained imports. Verification is
algorithmic, not formal proof.

## Scope

MathForge implements its own exact arithmetic with finite computation budgets.
It covers real polynomial roots, rational equations and inequalities, exact
linear algebra, polynomial factorization and real algebraic numbers,
multivariate derivatives, limits and analysis of rational functions, real
elementary functions, exact integration of rational functions, complex algebra,
number theory, combinatorics, probability and statistics. A formal
`RationalPrimitive` certifies an antiderivative identity; general `LogRootSum`
point evaluation is outside scope. Improper integrals, general complex root
isolation, symbolic matrices, approximate numerical output and remote HTTP
serving are outside this alpha.

## Release history

| Version | Date | Highlights |
| --- | --- | --- |
| 0.8.0 | 2026-10-21 | The speed release: exact kernels with independent checkers, a lean value path, area modules, mathematical JSON v6, MCP area tools and continued fractions |
| 0.7.0 | 2026-10-14 | Real elementary functions, exact integration, complex algebra, number theory, combinatorics, probability, statistics and JSON v5 |
| 0.6.0 | 2026-10-14 | Complete polynomial factorization, resultants, real algebraic arithmetic, characteristic polynomials, real eigenvalues and JSON v4 |
| 0.5.0 | 2026-10-14 | Multivariate derivatives, gradients, Jacobians and Hessians; faster root isolation and RREF |
| 0.4.0 | 2026-10-11 | Exact limits of rational functions and complete function analysis, JSON v3 |
| 0.3.0 | 2026-10-10 | Math-text parsing, exact text and LaTeX rendering, the optional MCP server |
| 0.2.0 | 2026-10-09 | Polynomials, real roots, rational equations and inequalities, matrices and JSON v2 |
| 0.1.0 | 2026-10-08 | First public alpha: the exact rational core |

The [changelog](CHANGELOG.md) has the details of each release.

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
dependencies. Distribution checks validate both base and MCP-extra
installations. Public `documentation/` is tracked and included in the sdist;
`docs/` remains local working material and is excluded from Git and release
packages.

## License

MathForge is released under the [MIT license](LICENSE).
