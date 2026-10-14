"""A checked exact-algebra workflow, including transport and domain preservation."""
import mathforge as mf


def main():
    x = mf.symbol("x")
    p = mf.Polynomial.from_expression(x**4 + 4, x)
    factors = mf.operations.factor(p)
    assert mf.verify(factors).verified
    assert len(factors.unwrap().factors) == 2
    assert mf.discriminant(mf.Polynomial((-2, 0, 0, 1), x)) == mf.Rational(-108)

    alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
    assert mf.algebraic(alpha**3) == mf.Rational(2)
    assert mf.algebraic_pow(alpha, -3) == mf.Rational(1, 2)
    assert mf.evaluate_algebraic(x**3, {x: alpha}) == mf.Rational(2)
    assert mf.compare_real((mf.sqrt(2)+mf.sqrt(3))**2, 5+2*mf.sqrt(6)) == 0
    assert mf.integrate(x, x, 0, mf.sqrt(2)) == mf.Rational(1)

    f = mf.rational_function(x+1, x*x-1, variable=x)
    decomposition = mf.apart(f)
    assert decomposition.to_rational_function() == f
    assert not decomposition.to_rational_function().is_defined_at(mf.Rational(-1))
    assert mf.limit(f, x, -1).value == mf.Rational(-1, 2)

    A = mf.Matrix(((0, 2), (1, 0)))
    assert mf.charpoly(A, x) == mf.Polynomial((-2, 0, 1), x)
    assert len(mf.real_eigenvalues(A)) == 2
    assert len(mf.extrema_values(x**3-6*x, x)) == 2

    workspace = mf.Workspace()
    reference = workspace.put(factors)
    workspace.put(mf.operations.limit(x, x, 0))
    restored = mf.loads(mf.dumps(workspace))
    assert restored.put(factors) == reference
    assert mf.to_data(workspace)["version"] == 4
    print(mf.render_text(factors.unwrap()))
    print("All exact-algebra workflow checks passed.")


if __name__ == "__main__":
    main()
