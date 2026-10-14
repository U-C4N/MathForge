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
    # The polynomial is a separate total function. Its rational counterpart
    # retains a hole at the endpoint and is not a proper closed-interval integral.
    integral = mf.operations.integrate(x + 1, x, -1, 1)
    assert integral.value == mf.Rational(2) and mf.verify(integral).verified
    unanswered = mf.operations.integrate(quotient, x, -1, 1)
    assert unanswered.execution_status is mf.ExecutionStatus.COMPLETED
    assert unanswered.outcome is mf.Outcome.NO_CONCLUSION
    assert unanswered.value is None and unanswered.deferred is not None
    assert not mf.verify(unanswered).verified
    print("Rational inequality: [-1, 1) union (1, infinity); polynomial integral: 2; verified")


if __name__ == "__main__":
    main()
