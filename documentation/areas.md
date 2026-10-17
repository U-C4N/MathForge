# Find operations by area

Each operation of v0.7 and later belongs to one mathematical area, and each area has its own module:
`mathforge.expr`, `mathforge.algebra`, `mathforge.sets`, `mathforge.logic`, `mathforge.equations`,
`mathforge.analysis`, `mathforge.series`, `mathforge.integrals`, `mathforge.ode`, `mathforge.special`,
`mathforge.linalg`, `mathforge.stats`, `mathforge.ntheory`, `mathforge.discrete`, `mathforge.geometry`,
`mathforge.vector`, `mathforge.numeric` and `mathforge.crypto`.

An area module's `__all__` lists the value functions of its operations and the record classes that their
results use, such as `mf.ntheory.CRTResult`. A module whose area has no operation yet has an empty `__all__`.
`import mathforge` loads none of these modules; the first access to an attribute such as `mf.ntheory` imports
that one module.

## Values and structured results

An area function returns the mathematical value and raises `OperationError` when there is none. The function
of the same name in `mathforge.operations` returns the whole `Result`: the request, the evidence and the
independent checks, which `mf.verify` repeats and `mf.dumps` serializes.

```python
import mathforge as mf
cf = mf.ntheory.continued_fraction(mf.Rational(415, 93))
assert cf.terms == (4, 2, 6, 7)
result = mf.operations.continued_fraction(mf.Rational(415, 93))
assert mf.verify(result).verified and mf.to_data(result)["version"] == 6
```

`continued_fraction` takes an exact rational, a `Rational` or an `int`, and returns its finite simple continued
fraction as a `ContinuedFraction` record: `[a0; a1, ..., an]`, where every term after the first is positive
and the last of several terms is at least 2. Each rational has exactly one such form. A float or a boolean is
invalid input, never an approximation.

```python
assert mf.render_text(cf) == "[4; 2, 6, 7]"
assert mf.ntheory.continued_fraction(-5).terms == (-5,)
assert mf.operations.continued_fraction(0.5).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

The results of operations added in v0.8 use JSON format version 6, which MathForge 0.7 does not read. Results
of the earlier operations keep their format version.

## Names

The operations published in v0.7 keep their names in the `mathforge` namespace too, so `mf.integer_gcd` and
`mf.ntheory.integer_gcd` are the same function. Operations added later are reached only through their area
module and `mathforge.operations`. Operations older than v0.7, such as `mf.solve` and `mf.factor`, stay where
they were and are not part of any area module.

```python
assert mf.ntheory.integer_gcd is mf.integer_gcd
assert not hasattr(mf, "continued_fraction")
assert mf.operations.continued_fraction(7).value == mf.ntheory.ContinuedFraction((7,))
```
