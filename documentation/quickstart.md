# Solve your first problem

Python 3.11 or newer is required. From a local checkout:

```sh
python -m pip install -e .
```

## Solve and check an equation

```python
import mathforge as mf

x = mf.symbol("x")
symbols = {"x": x}
problem = mf.parse_problem("x^3 - 2 = 0", symbols=symbols)
result = mf.solve(problem, for_=x)
assert result.execution_status is mf.ExecutionStatus.COMPLETED
assert mf.verify(result).verified
print(mf.render_text(result.unwrap()))
print(mf.to_latex(result.unwrap()))
```

An algebraic `RootOf` identifies an exact real root, not an approximation. Its
index is zero-based in increasing real-root order. Use `compare_real` for
mathematical ordering; ordinary equality is structural.

## Preserve excluded points

```python
problem = mf.parse_problem("(x^2 - 1)/(x - 1) = 2", symbols=symbols)
result = mf.solve(problem, for_=x)
assert result.outcome is mf.Outcome.NO_SOLUTION
assert mf.verify(result).verified
```

The apparent candidate `x=1` is excluded by the original denominator. Parsing,
solving and rendering retain that restriction even after cancellation.

## Store a result without losing exactness

```python
encoded = mf.dumps(result)
restored = mf.loads(encoded)
assert restored == result
assert mf.verify(restored).verified

workspace = mf.Workspace()
reference = workspace.put(result)
assert workspace.get(reference) == result
```

Loading checks the data format, not the truth of mathematical claims. Recheck
imported results with `verify`. Workspace content references identify structural
content; they are not mathematical equivalence identifiers.
