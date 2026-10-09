<p align="center">
  <img src="assets/mark.svg" width="112" height="112" alt="">
</p>

<h1 align="center">MathForge</h1>

<p align="center">
  Exact mathematics. Built from first principles.<br>
  <sub>By Umutcan EDIZASLAN</sub>
</p>

<p align="center">
  <sub>Python 3.11+ &nbsp;·&nbsp; Zero runtime dependencies &nbsp;·&nbsp; v0.2.0 alpha</sub>
</p>

<p align="center">
  <a href="https://github.com/U-C4N/MathForge/releases">Releases</a> &nbsp; / &nbsp;
  <a href="CHANGELOG.md">Changes</a> &nbsp; / &nbsp;
  <a href="examples/v02_workflow.py">Example</a> &nbsp; / &nbsp;
  <a href="LICENSE">MIT license</a>
</p>

<br>

## New in v0.2

- **Exact algebra:** polynomial division, gcd, square-free decomposition, and real roots with multiplicities.
- **More solvers:** rational equations and inequalities that preserve excluded points, plus matrices and complete linear-system solutions.
- **Calculus and verification:** repeated derivatives, polynomial integrals, independently checked results, and v2 JSON that preserves existing v1 records.

MathForge implements its own exact arithmetic with finite computation budgets.
Verification is algorithmic, not formal proof. Complex roots, general transcendental
calculus, and MCP are outside this alpha.

## Quick start

From a local checkout:

```sh
python -m pip install -e .
```

```python
import mathforge as mf

x = mf.symbol("x", domain=mf.Reals)
expression = x**2 - 2
result = mf.solve(mf.Eq(expression, 0), for_=x)

print(result.solution_set)         # {-sqrt(2), sqrt(2)}
print(mf.diff(expression, x))      # 2*x
assert mf.verify(result).verified
```

## A taste of v0.2

```python
roots = mf.real_roots(x**3 - x, x)  # -1, 0, 1 with multiplicities
rf = mf.rational_function(x**2 - 1, x - 1, variable=x)
assert mf.solve(mf.Eq(rf, 2), for_=x).outcome is mf.Outcome.NO_SOLUTION
assert mf.integrate(x**2, x, 0, 1) == mf.Rational(1, 3)
assert mf.det(mf.Matrix(((1, 2), (3, 4)))) == mf.Rational(-2)
```

[Full workflow](examples/v02_workflow.py) · [Linear systems](examples/linear_system.py) ·
[Rational inequalities](examples/rational_inequality.py)

## Development

```sh
python -m unittest discover -s tests -v
python tools/check_docs.py
python -m pip install -r requirements-dev.txt
python -m build
python tools/check_distribution.py
```

Tests use Python's standard library. Build, schema, and typing tools are development
dependencies only. `docs/` is local working material and is excluded from Git and
release packages.
