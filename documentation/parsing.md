# Read mathematical text

Use `parse_expression` for a value and `parse_problem` for one equation or
inequality. Both require an explicit `symbols` mapping. They return existing
mathematical objects, not a `Result` and not a newly verified claim.

```python
import mathforge as mf

x = mf.symbol("x")
bindings = {"x": x}
expression = mf.parse_expression("(x + 1)^2", symbols=bindings)
assert expression == (x + 1)**2
assert mf.parse_expression("0.1 + 1/5", symbols={}) == mf.Rational(3, 10)
problem = mf.parse_problem("x^2 >= 2", symbols=bindings)
assert isinstance(problem, mf.Inequality)
assert mf.verify(mf.solve(problem, for_=x)).verified
```

## Supported notation

| Input | Meaning |
| --- | --- |
| Integers, finite decimals, `/` | Exact rational arithmetic; no float conversion |
| `+ - * /`, parentheses, unary signs | Conventional arithmetic precedence |
| `^` or `**` | Right-associative, nonnegative integer exponent, at most 1024 |
| `sqrt(...)` | Argument must reduce to a nonnegative rational constant |
| `= == < <= > >=` | Exactly one top-level relation in `parse_problem` |
| ASCII identifiers | Explicit aliases from `symbols`; `sqrt` is reserved |

`-x^2` means `-(x^2)` and `x^2^3` means `x^8`. Multiplication must be
explicit: use `2*x`, not `2x`. Scientific notation, negative/fractional powers,
chained comparisons, `!=`, symbolic square roots and arbitrary function calls
are not supported. Python statements, attributes and indexing are not a grammar.

## Reuse symbol identity

Creating two symbols with the same name does not make them the same variable.
Keep and reuse the mapping when parsing separate expressions or equations.
The parser never modifies it and never creates missing symbols implicitly.

```python
left = mf.parse_expression("x + 1", symbols=bindings)
right = mf.parse_expression("x - 1", symbols=bindings)
assert mf.substitute(left + right, {x: 3}) == mf.Rational(6)
```

Symbolic denominators are supported only within the existing single-variable,
unconstrained-real rational-function model. A rational function remains a
rational function when its value becomes constant, preserving exclusions.

## Handle input failures and resource limits

```python
try:
    mf.parse_expression("missing + 1", symbols=bindings)
except mf.ParseError as error:
    assert error.code == "unknown_symbol"
    assert (error.start, error.end) == (0, 7)

small = mf.ParseLimits(max_chars=3)
assert mf.parse_expression("1/2", symbols={}, parse_limits=small) == mf.Rational(1, 2)
```

`ParseError` is an `InvalidInput` with stable code and zero-based, half-open
Unicode-character `start`/`end` positions. Mathematical scope errors remain
`UnsupportedOperation`; exhausted resources remain `ResourceLimitError`.
Neither means that an equation has no solutions.

Default `ParseLimits`: 16,384 characters; 4,096 tokens and syntax nodes;
depth 64; 1,024 digits per numeric literal; 64 symbols; exponent 1,024.
`limits=` separately supplies existing `ComputationLimits` for construction.
Both limits are finite and accept explicit positive integer overrides in Python.
