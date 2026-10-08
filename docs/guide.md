# MathForge guide

[← Back to MathForge](../README.md)

MathForge is an independent, exact mathematics core for Python applications and AI agents. The first release covers rational arithmetic, symbolic expressions, univariate polynomials, and linear and quadratic equations over the real numbers.

**v0.1 is an alpha release.** It is not a general-purpose computer algebra system, an approximate computation engine, or a formal proof checker. It has no runtime dependencies and does not delegate calculations to other mathematics engines. MCP and visualization integrations are not implemented yet.

## Installation and your first calculation

Python 3.11 or newer is required. From the repository root:

```sh
python -m pip install -e .
```

```python
import mathforge as mf

x = mf.symbol("x", domain=mf.Reals)
expression = x**2 - 2
problem = mf.Eq(expression, 0)

result = mf.solve(problem, for_=x)
assert result.execution_status is mf.ExecutionStatus.COMPLETED
assert result.exactness is mf.Exactness.EXACT
assert result.completeness is mf.Completeness.COMPLETE
assert set(result.solution_set.values) == {-mf.sqrt(2), mf.sqrt(2)}
assert mf.verify(result).verified

derivative = mf.diff(expression, x)
assert mf.evaluate(derivative, {x: mf.Rational(3, 2)}) == mf.Rational(3)
print(result.solution_set)  # {-sqrt(2), sqrt(2)}
print(result.method)
```

After installation, run the complete calculation and persistence workflow:

```sh
python examples/end_to_end.py
python examples/end_to_end.py --output workspace.json
```

The first command uses a temporary file. The second saves a Workspace to the specified file, replacing any existing file at that path.

## Exact numbers and symbols

`Rational` is MathForge's own numeric type, not a wrapper around `Fraction`. Python integers are automatically converted to rational constants in expressions. The numerator and denominator are normalized by their greatest common divisor, the denominator is positive, and zero is stored as `0/1`.

```python
import mathforge as mf

assert mf.Rational(2, -4) == mf.Rational(-1, 2)
assert mf.Rational(1, 10) + mf.Rational(2, 10) == mf.Rational(3, 10)
assert mf.sqrt(mf.Rational(9, 16)) == mf.Rational(3, 4)
assert mf.sqrt(2)**2 == mf.Rational(2)
```

`Rational(0.1)`, adding `0.1` to an expression, and passing `bool` as an exact numeric input are rejected. An approximate number type has not been implemented. `Rational(1)` and Python's `1` are structurally different objects; use `Rational(1)` when comparing exact results. Ordering comparisons between rationals and Python integers are supported.

Each `symbol()` call creates a new UUID identity. The name is only a display label. `x = symbol("x")` and `y = symbol("x")` create distinct symbols; substitution keys are symbol objects, not names. Saving and loading preserves the UUID.

Symbols can have the domain `Reals`, `Rationals`, or `Integers`. Records such as `Assumption("positive", Truth.TRUE)` represent the properties `positive`, `nonnegative`, and `nonzero`. `Truth.UNKNOWN` imposes no constraint and is not treated as true. Contradictory assumptions are rejected; substitution preserves domains and known assumptions. The solver only supports real symbols with an empty assumptions list.

## Transforming expressions

The `Expression` types `Rational`, `Symbol`, `Add`, `Mul`, `Pow`, and `Sqrt` are immutable. Natural Python operators construct the appropriate nodes. `==` tests structural equality; use `Eq` to construct an equation.

```python
import mathforge as mf

x, y = mf.symbol("x"), mf.symbol("y")
expression = x + 2*y
partial = mf.substitute(expression, {x: 3})
assert partial == 3 + 2*y
assert mf.evaluate(partial, {y: 4}) == mf.Rational(11)
assert mf.simplify(x + x - 2*x) == mf.Rational(0)

# Structural equality is not general mathematical equivalence.
assert (x + 1)**2 != x**2 + 2*x + 1
```

Substitution is simultaneous: a replacement expression is not substituted again using the same bindings. `substitute` allows partial bindings; `evaluate` produces an invalid-input result if any free symbols remain. Numeric radical expressions may remain as an abstract syntax tree (AST) without being approximated; `evaluate` is not a general algebraic number normalizer.

Simplification covers exact constant arithmetic, identity elements, flattening nested sums and products, collecting structurally identical terms, rational scalar distribution, and safe power and square-root rules. It does not perform general factorization or equivalence searches. Structural ordering is not numerical ordering.

### Supported scope

| Operation | v0.1 behavior |
| --- | --- |
| Addition, subtraction, multiplication | Exact symbolic operations |
| Division | Only by a nonzero rational constant |
| Powers | Only nonnegative Python integer exponents |
| Zeroth power | Polynomial convention: `x**0 = 1`, including `0**0 = 1` |
| Square roots | Nonnegative exact rational radicands |
| `x/x`, `x**-1`, `sqrt(x**2)` | Explicit `UnsupportedOperation` |
| Polynomials | One symbol, rational coefficients, dense representation; degree at most 1024 |
| Differentiation | Rational-coefficient polynomials in a real variable with no additional assumptions |
| Equation solving | Equations over the reals with effective degree at most 2 |
| Complex roots, integration, limits, matrices/tensors, other areas | Not implemented yet |

Restricting radicands to rational constants prevents the incorrect simplification of `sqrt(x**2)` to `x`. Symbolic denominators cannot be constructed, so denominator cancellation cannot lose excluded points. Unsupported calculations are not replaced with approximate answers.

## Working with polynomials

Coefficients are given in ascending degree order, starting with the constant term. Zero coefficients at the high-degree end are removed; the zero polynomial has degree `-1` by convention.

```python
import mathforge as mf

x = mf.symbol("x")
p = mf.Polynomial((1, -3, 2, 0), variable=x)  # 1 - 3x + 2x²
assert p.degree == 2
assert p.evaluate(3) == mf.Rational(10)
assert p.derivative().coefficients == (mf.Rational(-3), mf.Rational(4))
assert mf.Polynomial.from_expression((x-1)*(2*x-1), x) == p
```

Coefficient addition and multiplication, Horner evaluation, and coefficient-based differentiation are implemented locally. The dense expansion limit guards against accidental excessive memory use from very large powers; general expression trees are not subject to this degree limit. Resources are finite, and there is no general-purpose computation quota system.

## Result and error contract

`solve` returns a structured `Result`. `diff`, `evaluate`, `substitute`, and `simplify` return the successful value or raise an `OperationError` carrying the full result in `.result`. Constructors and direct expression operators may raise `TypeError`, `InvalidInput`, `ZeroDivisionError`, or `UnsupportedOperation`.

Use `mf.operations.solve`, `differentiate`, `evaluate`, `substitute`, and `simplify` to obtain structured results from every operation. SDK convenience functions unwrap the values from these same operations; there is no separate computation path.

| Field | Meaning |
| --- | --- |
| `execution_status` | `completed`, `unsupported`, `invalid_input`, `error` |
| `outcome` | `value`, `solutions`, `candidates`, `no_solution`, `no_conclusion`, `partial`, `not_applicable` |
| `exactness` | `exact`, `approximate`, `mixed`, `not_applicable` |
| `completeness` | `complete`, `partial`, `unknown`, `not_applicable` |
| `value` / `solution_set` | Operation value / represented solution set, if available |
| `problem`, `expression`, `for_` | Source problem or expression and target symbol |
| `method`, `domain`, `assumptions` | Method identifier and structured context |
| `verification`, `steps` | Claim checks and actual operation steps |
| `precision`, `tolerance`, `error_bound` | Numerical metadata; `None` in the initial algorithms |
| `error` | Stable `code` and descriptive `message` |

An exact representation does not imply completeness. The contract can represent candidates, partial results, and inconclusive outcomes; the current bounded solver does not implement additional algorithms that produce those states. A method's failure to find a result is not encoded as `no_solution`.

```python
import mathforge as mf

x = mf.symbol("x")
empty = mf.solve(mf.Eq(x**2 + 1, 0), for_=x)
assert empty.outcome is mf.Outcome.NO_SOLUTION
assert isinstance(empty.solution_set, mf.EmptySet)

unsupported = mf.solve(mf.Eq(x**3 - 2, 0), for_=x)
assert unsupported.execution_status is mf.ExecutionStatus.UNSUPPORTED
assert unsupported.solution_set is None

all_reals = mf.solve(mf.Eq(0*x, 0), for_=x)
assert isinstance(all_reals.solution_set, mf.UniversalSet)
```

Quadratic roots are computed in center-and-radius form: `-b/(2a) ± sqrt(D/(4a²))`. This also handles a negative leading coefficient and gives `±sqrt(2)` directly for `x²−2`. The minus branch is stored first, followed by the plus branch. `FiniteSet` does not provide general set algebra; it removes structurally identical duplicates. The solver classifies repeated roots separately.

## Verification and solution steps

The solver records polynomial normalization, discriminant calculation, and degree-based solution steps as they occur. Each `Step` contains its rule, inputs and outputs, preconditions, transformation relation, and audit. The `explanation` field is instructional text, not proof.

`mf.verify(result)` rechecks the source equation and steps without trusting stored `verified` labels. Rational-pair arithmetic substitutes roots of the form `u + v√d` into the source expression exactly. Membership means that a candidate satisfies the equation. Completeness is checked separately using the effective degree, discriminant sign, and number of distinct roots. Incorrect roots, missing roots, and altered discriminants fail verification.

Verification levels are `exact_substitution` and `algorithmic_check`; no formal proof is claimed. The verifier independently checks only supported equation results and solution rules. Other operation steps are recorded but lie outside the independent verification scope; `verify` returns an unknown result for them. An empty list of checks does not count as verified.

## Saving objects and workspaces

Workspace is optional; calculations require no storage backend, server, or file.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import mathforge as mf

x = mf.symbol("x")
problem = mf.Eq(x**2 - 2, 0)
result = mf.solve(problem, for_=x)
workspace = mf.Workspace()
problem_ref = workspace.put(problem)
symbol_ref = workspace.put(x)
result_ref = workspace.put(result)
workspace.record("solve", {"problem": problem_ref, "for_": symbol_ref}, {}, result_ref)

with TemporaryDirectory() as directory:
    path = Path(directory) / "workspace.json"
    workspace.save(path)
    restored = mf.Workspace.load(path)
    assert restored.get(result_ref) == result
    assert mf.verify(restored.get(result_ref)).verified

assert mf.loads(mf.dumps(result)) == result
assert mf.from_data(mf.to_data(problem)) == problem
```

`put` stores an immutable object by its content address: identical content produces the same reference, while changed content produces a new one. A reference is the SHA-256 digest of the versioned canonical JSON. Structural content identity is not proof of mathematical equivalence. `get(ref)` retrieves the object. `record` appends only the explicitly supplied operation record; it does not replay the operation or independently prove that the calculation took place.

The JSON envelope contains `format="mathforge"`, `version=1`, `kind`, and `data`. Mathematical integers and rational numerators and denominators are encoded as decimal **strings**, independent of JavaScript number precision. UUIDs, domains, assumptions, radical ASTs, result axes, and steps are preserved.

Loading uses a fixed type allowlist and never executes `eval`, `pickle`, or dynamic imports. Unknown versions, types, or fields; duplicate JSON keys; `NaN`/`Infinity`; invalid or unnormalized rationals; contradictory result states; incorrect content hashes; missing references; and conflicting symbol identities are rejected. Workspace loading is all-or-nothing: an invalid document is never partially loaded. Validating the data format does not verify its mathematical claims; call `verify` again for that.

## Architecture and next steps

| Layer | Responsibility |
| --- | --- |
| `model`, `polynomial` | Exact types, expression rules, and polynomial structure |
| `contracts` | Problems, results, solution sets, preconditions, and audit records |
| `algorithms`, `verification` | Solving algorithms and separate rechecking |
| `operations`, package API | Shared operation results and Python ergonomics |
| `serialization`, `workspace` | Versioned transport, content references, and persistence |

The solver does not call Workspace or the serialization layer. The verifier does not call the solving algorithm; it shares the fundamental mathematical types and polynomial representation. This separation provides a small, testable verification boundary, not a fully independent formal proof kernel.

**Representing**, **solving**, and **verifying** a problem family are separate capabilities. For example, a cubic expression can be constructed and differentiated, but the v0.1 solver cannot solve cubic equations. Unimplemented areas have no placeholder modules.

The next MCP phase will add a thin adapter around the validated SDK capabilities, using the same `operations` functions and versioned data contract. It will include input validation at the boundary, SDK/MCP result parity tests, and a separate justification for any protocol dependency. No server is implemented in this release. Any future Vesora integration will first verify its current API, capabilities, and license; approximate plotting samples will remain separate from exact objects. A Rust/C++ layer will only be considered in response to measured needs.

## Development and testing

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/end_to_end.py
python -m pip wheel --no-deps . --wheel-dir dist
```

`setuptools>=77` is required only for PEP 517 package builds and MIT license metadata. Tests use the standard library's `unittest`; no additional test dependencies are required. The runtime dependency list is empty.

Tests exercise rational ring identities, polynomials derived from known factors, exact root substitution, incorrect or incomplete solution records, and corrupted serialization scenarios. Exact tests use no floating-point tolerances. Tests and examples should be checked against both the source tree and a wheel installed in an isolated environment.

## License

[MIT](../LICENSE). Copyright 2026 MathForge contributors.
