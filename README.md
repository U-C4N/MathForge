<p align="center">
  <img src="assets/mark.svg" width="112" height="112" alt="">
</p>

<h1 align="center">MathForge</h1>

<p align="center">
  Exact mathematics. Built from first principles.<br>
  <sub>By Umutcan EDIZASLAN</sub>
</p>

<p align="center">
  <sub>Python 3.11+ &nbsp;·&nbsp; Dependency-free core &nbsp;·&nbsp; v0.9.0 alpha</sub>
</p>

<p align="center">
  <a href="https://github.com/U-C4N/MathForge/releases">Releases</a> &nbsp; / &nbsp;
  <a href="documentation/index.md">Guide</a> &nbsp; / &nbsp;
  <a href="CHANGELOG.md">Changes</a> &nbsp; / &nbsp;
  <a href="examples/v09_workflow.py">Example</a> &nbsp; / &nbsp;
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

## New in v0.9.0: more mathematics

v0.9 adds 65 operations, from 103 to 168, and opens five new areas: equations,
sets, decimal digits, vectors and plane geometry. Every new operation is checked
by its own independent checker, and its result re-verifies from its JSON.

| Area | New operations |
| --- | --- |
| Number theory | `powmod`, `multiplicative_order`, `primitive_root`, `legendre_symbol`, `jacobi_symbol`, `sqrt_mod`, `next_prime`, `prev_prime`, `linear_diophantine`, `quadratic_continued_fraction`, `convergents`, `pell_fundamental`, `moebius`, `divisor_sigma`, `integer_nth_root`, `discrete_log`, `two_squares` |
| Algebra | `count_real_roots`, `complex_roots`, `interpolate`, `denest_sqrt`, `sturm_sequence`, `cyclotomic_polynomial` |
| Equations and sets | `solve_abs`, `solve_sqrt`, `set_union`, `set_intersection`, `set_difference`, `set_complement`, `set_symmetric_difference` |
| Decimal digits and approximation | `digits`, `rational_approximation` |
| Sums and series | `sum_polynomial`, `geometric_sum`, `geometric_series`, `polynomial_geometric_series`, `series_coefficients`, `residue` |
| Integrals | `trig_power_antiderivative`, `trig_power_integral` |
| Linear algebra | `matrix_power` (also `Matrix ** k`), `eigenvectors`, `definiteness`, `gram_schmidt`, `hermite_normal_form`, `smith_normal_form` |
| Vectors and geometry | `dot`, `cross`, `projection`, `triple_product`, `distance`, `polygon_area`, `line_intersection`, `orientation`, `convex_hull` |
| Statistics and counting | `correlation`, `linear_regression`, `quantile`, `mode`, `stationary_distribution`, `fisher_exact`, `partitions`, `derangements`, `bell`, `stirling1` |

```python
import mathforge as mf

Q = mf.Rational
x = mf.symbol("x")
F = lambda text: mf.parse_function(text, symbols={"x": x})

assert mf.ntheory.pell_fundamental(61) == (1766319049, 226153980)
assert mf.render_text(mf.equations.solve_abs(F("abs(x - 1) + abs(x + 2)"), F("3"), x)) == "[-2, 1]"
assert mf.render_text(mf.numeric.digits(mf.sqrt(2), 20)) == "1.41421356237309504880..."
assert mf.render_text(mf.integrals.trig_power_integral(F("sin(x)^2"), x, 0, 1)) == "pi/2"
assert mf.linalg.matrix_power(mf.Matrix(((1, 1), (1, 0))), 10) == mf.Matrix(((89, 55), (55, 34)))
assert mf.geometry.polygon_area(mf.Matrix(((0, 0), (4, 0), (0, 3)))) == Q(6)
assert mf.numeric.rational_approximation(mf.sqrt(2), 100) == Q(140, 99)
assert mf.linalg.smith_normal_form(mf.Matrix(((2, 4), (6, 8)))) == (2, 4)
result = mf.operations.eigenvectors(mf.Matrix(((2, 1), (1, 2))))
assert mf.verify(mf.loads(mf.dumps(result))).verified
```

A question outside an operation's rules, such as the complex roots of x³ − 2, is
deferred with a reason rather than guessed. Of the 167 natural requests in the
probe scoreboard (`tools/check_probes.py`), 133 now get a verified answer, up
from 88 in v0.8, and none a wrong one. The MCP server gains three area tools,
`mathforge.equations`, `mathforge.geometry` and `mathforge.numeric`, for 40 in
all.

Every v0.8 operation keeps its name, signature and result, and v1–v6 documents
keep their bytes and verdicts. `Matrix * Matrix` still raises `InvalidInput`; its
message now points to `@`.

Known limitations: each operation documents the input sizes that fit the default
budget, and larger inputs end as `resource_limit_exceeded`, never as a wrong
value. `integer_gcd` and `integer_lcm` on integers of thousands of bits, and
`solve_system`, keep the costs that v0.8 documented.

See the [v0.9 mathematics guide](documentation/v09-mathematics.md), the [area guide](documentation/areas.md), the
[complete executable example](examples/v09_workflow.py) and the
[changelog](CHANGELOG.md). v0.8.0, the speed release, made exact computation many
times cheaper; its figures are in the [performance notes](documentation/performance.md).

## Quick start

Install the wheel from the [v0.9.0 release](https://github.com/U-C4N/MathForge/releases/tag/v0.9.0):

```sh
python -m pip install https://github.com/U-C4N/MathForge/releases/download/v0.9.0/mathforge-0.9.0-py3-none-any.whl
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
offers 40 tools: the thirty curated tools of v0.7, area tools that run an
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
elementary functions, exact integration of rational functions and of powers of
sine and cosine, power series, sums, residues, complex algebra, number theory,
equations with absolute values and square roots, interval sets, exact decimal
digits and best rational approximations, integer lattices (Hermite and Smith
forms), vectors, plane geometry and convex hulls, combinatorics, probability and
statistics, including Markov chains and Fisher's exact test. A formal
`RationalPrimitive` certifies an antiderivative identity; general `LogRootSum`
point evaluation is outside scope. Improper integrals, complex roots beyond
irreducible factors of degree two, symbolic matrices, approximate numerical
output and remote HTTP serving are outside this alpha.

## Release history

| Version | Date | Highlights |
| --- | --- | --- |
| 0.9.0 | 2026-11-14 | More mathematics: 65 new operations in number theory, algebra, equations, sets, decimal digits, series, integrals, linear algebra, vectors, geometry, statistics and counting; three new MCP area tools |
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
