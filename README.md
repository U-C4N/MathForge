<p align="center">
  <img src="docs/assets/mark.svg" width="112" height="112" alt="">
</p>

<h1 align="center">MathForge</h1>

<p align="center">
  Exact mathematics. Built from first principles.<br>
  <sub>By Umutcan EDIZASLAN</sub>
</p>

<p align="center">
  <sub>Python 3.11+ &nbsp;·&nbsp; Zero runtime dependencies &nbsp;·&nbsp; v0.1 alpha</sub>
</p>

<p align="center">
  <a href="docs/guide.md">Guide</a> &nbsp; / &nbsp;
  <a href="examples/end_to_end.py">Example</a> &nbsp; / &nbsp;
  <a href="LICENSE">MIT license</a>
</p>

<br>

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

## A small, exact core

- **Build** with rational numbers and immutable expressions.
- **Solve** real linear and quadratic equations; differentiate polynomials.
- **Keep** auditable results in workspaces with versioned JSON.

MathForge implements its own mathematics. Root membership and completeness are
checked separately; verification is algorithmic, not formal proof.

This alpha supports rational-coefficient univariate polynomials. Symbolic
denominators, general symbolic roots, MCP, and visualization are outside this
release. [Explore the scope →](docs/guide.md#supported-scope)

<br>

---

[Run the full example](examples/end_to_end.py) &nbsp;·&nbsp;
[Development](docs/guide.md#development-and-testing) &nbsp;·&nbsp;
[Architecture](docs/guide.md#architecture-and-next-steps)
