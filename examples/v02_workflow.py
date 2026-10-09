"""Exact v0.2 capabilities; run after installing MathForge."""
from pathlib import Path
from tempfile import TemporaryDirectory
import mathforge as mf


def main():
    x, y = mf.symbol("x"), mf.symbol("y")
    polynomial = mf.Polynomial.from_expression((x - 1)**3 * (x + 2)**2, x)
    roots = mf.operations.real_roots(polynomial, x)
    assert [record.multiplicity for record in roots.value] == [2, 3]
    cubic = mf.solve(mf.Eq(x**3 - 2, 0), for_=x)
    assert isinstance(cubic.value.values[0], mf.RealAlgebraicRoot)
    assert mf.compare_real(cubic.value.values[0], 1) == 1

    function = mf.rational_function(x**2 - 1, x - 1, variable=x)
    excluded = mf.solve(mf.Eq(function, 2), for_=x)
    assert isinstance(excluded.value, mf.EmptySet)
    inequality = mf.solve(mf.Inequality(function, "ge", 0), for_=x)
    assert inequality.value.contains(-1) and not inequality.value.contains(1)
    assert mf.diff(function, x).excluded == function.excluded

    A = mf.Matrix(((1, 2), (2, 4)))
    system = mf.solve_system(mf.LinearSystem(A, (3, 6), (x, y)))
    assert mf.rank(A) == 1 and len(system.value.basis) == 1
    integral = mf.operations.integrate(x**1024, x, 0, 1)
    assert integral.value == mf.Rational(1, 1025)

    workspace = mf.Workspace()
    legacy = workspace.put(x)
    results = (roots, cubic, excluded, inequality, system, integral)
    references = tuple(workspace.put(result) for result in results)
    assert all(mf.verify(result).verified for result in results)
    with TemporaryDirectory(prefix="mathforge-v02-") as directory:
        path = Path(directory) / "workspace.json"
        workspace.save(path)
        restored = mf.Workspace.load(path)
        assert restored.get(legacy) == x
        assert tuple(restored.get(ref) for ref in references) == results
        assert all(mf.verify(restored.get(ref)).verified for ref in references)
    print("v0.2 exact roots, domain holes, inequalities, systems, calculus and mixed workspace: passed")


if __name__ == "__main__":
    main()
