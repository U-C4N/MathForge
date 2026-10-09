"""Complete affine solutions and independent row-evidence verification."""
import mathforge as mf


def main():
    x, y, z = mf.symbol("x"), mf.symbol("y"), mf.symbol("z")
    result = mf.solve_system((mf.Eq(x + 2*y - z, 3), mf.Eq(2*x + 4*y - 2*z, 6)),
                             for_=(x, y, z))
    assert result.execution_status is mf.ExecutionStatus.COMPLETED
    family = result.solution_set
    assert isinstance(family, mf.AffineSolutionSet)
    assert family.particular == (mf.Rational(3), mf.Rational(0), mf.Rational(0))
    assert len(family.basis) == 2
    assert mf.verify(result).verified
    restored = mf.loads(mf.dumps(result))
    assert restored == result and mf.verify(restored).verified
    print("Affine system: one particular solution, two independent free directions, verified")


if __name__ == "__main__":
    main()
