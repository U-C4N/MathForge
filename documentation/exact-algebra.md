# Compute with exact algebraic values

MathForge v0.6 adds complete rational polynomial factorization, real algebraic
arithmetic, partial fractions and rational-matrix spectral operations. The core
has no runtime dependency on another mathematics library.

All algorithms have finite computation budgets. A resource error means the
operation could not be completed within its budget; it never means that a
polynomial is irreducible or an equation has no solution. Verification is
independent algorithmic checking, not a formal proof kernel.

## Factor a rational polynomial

`factor` accepts a `Polynomial` over the rationals. It returns a
`PolynomialFactorization(coefficient, factors)` whose factors are monic,
irreducible over the rationals, deterministically ordered and paired with positive
multiplicities. This is stronger than square-free decomposition.

```python
import mathforge as mf

x = mf.symbol("x")
p = mf.Polynomial.from_expression(x**4 + 4, x)
result = mf.operations.factor(p)
decomposition = result.unwrap()
assert decomposition.coefficient == mf.Rational(1)
assert len(decomposition.factors) == 2
rebuilt = mf.Polynomial((decomposition.coefficient,), x)
for item in decomposition.factors:
    rebuilt *= item.polynomial**item.multiplicity
assert rebuilt == p and mf.verify(result).verified
```

Nonzero constants have an empty factor tuple; the zero polynomial is invalid.
Every nonconstant factor carries an independently checked irreducibility
certificate. Multiplying factors back together alone does not establish this claim.

`resultant(f, g)` requires the same explicit variable identity. A zero operand
gives zero, and two nonzero constants give one. `discriminant(f)` is zero for
constant polynomials, one for degree one, and detects repeated roots in higher
degrees. The producer uses a fraction-free subresultant remainder sequence;
checking uses the
Sylvester determinant definition.

## Reuse an irrational root

Ordinary Python arithmetic builds the existing immutable expression tree.
Call `algebraic` to normalize a closed real algebraic expression. Its result is a
`Rational` or a `RealAlgebraicRoot` with a primitive irreducible integer defining
polynomial. Existing root descriptors need not already use minimal polynomials.

```python
import mathforge as mf

x = mf.symbol("x")
alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
assert mf.algebraic(alpha**3) == mf.Rational(2)
assert mf.algebraic_div(alpha, alpha) == mf.Rational(1)
assert mf.algebraic_pow(alpha, -3) == mf.Rational(1, 2)
assert mf.minimal_polynomial(alpha, x) == mf.Polynomial((-2, 0, 0, 1), x)
assert mf.compare_real((mf.sqrt(2)+mf.sqrt(3))**2, 5+2*mf.sqrt(6)) == 0
assert mf.evaluate_algebraic(x**3, {x: alpha}) == mf.Rational(2)
```

`algebraic_div` proves that its denominator is nonzero. `algebraic_pow` accepts
signed Python integer exponents; zero to a negative power is invalid. As in the
existing expression model, exponent zero gives one. Free variables, floats,
complex values and transcendental functions are unsupported.

`==` and hashing still express structural identity. Use `compare_real(a, b) == 0`
for mathematical equality. Existing `evaluate` and `substitute` retain their
historical expression behavior; `evaluate_algebraic` provides explicit closed
numeric evaluation with simultaneous bindings.

## Preserve the original function domain

`apart` decomposes a `RationalFunction` over irreducible rational polynomial
factors. The result contains a polynomial part, `PartialFractionTerm` records
with `numerator`, `factor`, `power`, and the original `excluded` polynomials.
Its `to_rational_function()` method restores both value and domain.

```python
import mathforge as mf

x = mf.symbol("x")
f = mf.rational_function(x+1, x*x-1, variable=x)
parts = mf.apart(f)
assert parts.to_rational_function() == f
assert not parts.to_rational_function().is_defined_at(mf.Rational(-1))
assert mf.limit(f, x, -1) == mf.LimitValue("finite", mf.Rational(-1, 2))
```

Canceling a factor never restores a point excluded by the source function.
Evaluation at a hole is invalid, although a finite limit there can exist.
Partial fractions do not add logarithmic antiderivatives or general rational
integration.

## Use algebraic calculus endpoints

Finite limit points and polynomial integral endpoints may be closed real
algebraic expressions. Rational input paths retain their previous records and
wire versions. New algebraic computations use v4 evidence.

```python
import mathforge as mf

x = mf.symbol("x")
assert mf.integrate(x, x, 0, mf.sqrt(2)) == mf.Rational(1)
assert mf.integrate(x, x, mf.sqrt(2), 0) == mf.Rational(-1)
limit = mf.limit(x*x, x, mf.sqrt(2))
assert limit == mf.LimitValue("finite", mf.Rational(2))
extrema = mf.extrema_values(x**3-6*x, x)
assert len(extrema) == 2
assert [entry.kind for entry in extrema] == ["maximum", "minimum"]
for entry in extrema:
    assert mf.compare_real(entry.value, mf.evaluate_algebraic(x**3-6*x, {x: entry.point})) == 0
```

`extrema_values` returns immutable `ExtremumValue(point, value, kind)` records.
The existing `FunctionAnalysis.strict_extrema` remains x-only. Finite
`LimitValue.value` may now be rational or a real root descriptor. Infinite
integration bounds and algebraic coefficients in source polynomials remain
outside scope.

## Find a matrix's real spectrum

```python
import mathforge as mf

t = mf.symbol("t")
A = mf.Matrix(((0, 2), (1, 0)))
assert mf.charpoly(A, t) == mf.Polynomial((-2, 0, 1), t)
roots = mf.real_eigenvalues(A)
assert len(roots) == 2 and all(root.multiplicity == 1 for root in roots)
rotation = mf.Matrix(((0, -1), (1, 0)))
assert mf.real_eigenvalues(rotation) == ()
assert mf.charpoly(mf.Matrix(()), t) == mf.Polynomial((1,), t)
```

`charpoly` uses `det(tI-A)`. Its checker verifies the determinant polynomial at
enough exact points; Cayley–Hamilton alone is insufficient. `real_eigenvalues`
returns every distinct real eigenvalue with its algebraic multiplicity, in
increasing order. An empty tuple means there are no **real** eigenvalues.
Matrices remain rational-valued; complex spectra and general eigenvectors are
not provided.

## Transport and verification

All operations have value-returning `mathforge` wrappers and structured
`mathforge.operations` counterparts accepting `limits=ComputationLimits(...)`.
Successful structured results are checked before being returned. Re-run
`verify(result)` after loading a result; stored verification labels are not proof.

New records and semantic evidence use mathematical JSON v4. Existing objects
retain their v1/v2/v3 bytes and structural content references. Mixed workspaces
select the greatest required version independently of insertion order.

The optional MCP adapter exposes the same operations with exact text/LaTeX and
canonical JSON. Scalar operands accept closed mathematical text or a canonical
scalar object envelope, allowing root outputs to become subsequent inputs without
parsing their display text. See [MCP inputs](mcp.md) and [compatibility](compatibility.md).
