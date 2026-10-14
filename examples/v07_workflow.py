"""Run after installing MathForge, or with PYTHONPATH=src from the repository."""

import mathforge as mf
from mathforge.mcp.boundary import call_tool


def main():
    Q = mf.Rational
    x = mf.Symbol("x")

    assert mf.integer_gcd(84, 30) == 6
    g, s, t = mf.integer_xgcd(84, 30)
    assert g == 84*s + 30*t == 6
    residues = mf.crt(((2, 3), (3, 5)))
    assert (residues.residue, residues.modulus) == (8, 15)
    assert mf.binomial(10, 3) == 120

    trials = mf.binomial_distribution(4, Q(1, 2))
    assert mf.probability(trials, (3, 4)) == Q(5, 16)
    assert mf.expectation(trials) == Q(2)
    assert mf.distribution_variance(trials) == Q(1)
    assert mf.median((9, 1, 3, 2)) == Q(5, 2)
    assert mf.variance((1, 2, 4), ddof=1) == Q(7, 3)

    A = mf.Matrix(((1, 1), (2, 2)))
    fit = mf.least_squares(A, (1, 3))
    assert fit.particular == (Q(7, 5), Q(0))
    assert fit.basis == ((Q(-1), Q(1)),)
    assert fit.residual == (Q(-2, 5), Q(1, 5))
    factors = mf.lu(A)
    assert factors.P @ A == factors.L @ factors.U

    imaginary_unit = mf.complex_algebraic(0, 1)
    squared = mf.complex_mul(imaginary_unit, imaginary_unit)
    assert squared.real == Q(-1) and squared.imag == Q(0)

    function = mf.parse_function("exp(x)+sin(x)", symbols={"x": x})
    derivative = mf.operations.function_differentiate(function, x)
    assert mf.verify(derivative).verified
    series = mf.taylor(mf.parse_function("sin(x)", symbols={"x": x}), x, 0, 5)
    assert len(series.coefficients) == 6
    assert series.coefficients[1] == mf.functions.as_real_expr(1)

    source = mf.rational_function(1, x**3+x+1, variable=x)
    primitive = mf.operations.rational_antiderivative(source, x)
    assert mf.verify(primitive).verified
    assert primitive.value.log_terms

    question = mf.parse_function("exp(x*x)", symbols={"x": x})
    deferred = mf.operations.function_antiderivative(question, x)
    assert deferred.outcome is mf.Outcome.NO_CONCLUSION
    assert deferred.value is None
    assert isinstance(deferred.deferred, mf.UnevaluatedIntegral)

    response = call_tool("mathforge.compute", {
        "operation": "integer_gcd",
        "arguments": {"a": mf.to_data(84), "b": mf.to_data(30)},
    })
    assert not response.is_error, response.payload
    result = mf.from_data(response.payload["result"])
    assert result.value == 6 and mf.verify(result).verified
    assert mf.loads(mf.dumps(result)) == result

    print("Integer, finite probability, statistics and linear algebra checks passed.")
    print("Complex algebraic and domain-aware calculus checks passed.")
    print("Cubic rational integral retained an exact logarithmic root-sum primitive.")
    print("Unanswered elementary integral retained an explicit deferred request.")
    print("Canonical MCP computation and independent verification passed.")


if __name__ == "__main__":
    main()
