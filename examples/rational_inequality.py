"""A canceled hole stays excluded; integrate a separately chosen polynomial."""
import mathforge as mf


def main():
    x = mf.symbol("x")
    quotient = mf.rational_function(x**2 - 1, x - 1, variable=x)
    result = mf.solve(mf.Inequality(quotient, "ge", 0), for_=x)
    assert result.solution_set == mf.IntervalSet((
        mf.Interval(-1, 1, True, False), mf.Interval(1, None)))
    assert mf.verify(result).verified
    undefined = mf.operations.evaluate(quotient, {x: 1})
    assert undefined.error.code == "undefined_at_point"
    # The polynomial is a separate total function; RF integration is unsupported.
    integral = mf.operations.integrate(x + 1, x, -1, 1)
    assert integral.value == mf.Rational(2) and mf.verify(integral).verified
    assert mf.operations.integrate(quotient, x, -1, 1).execution_status is mf.ExecutionStatus.UNSUPPORTED
    print("Rational inequality: [-1, 1) union (1, infinity); polynomial integral: 2; verified")


if __name__ == "__main__":
    main()
