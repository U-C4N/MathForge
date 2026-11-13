"""Run after installing MathForge, or with PYTHONPATH=src from the repository."""

import mathforge as mf
from mathforge.mcp.boundary import call_tool


def main():
    Q = mf.Rational
    x = mf.symbol("x")
    P = lambda expression: mf.Polynomial.from_expression(expression, x)
    F = lambda text: mf.parse_function(text, symbols={"x": x})

    # Number theory: modular arithmetic, primes, Pell's equation.
    assert mf.ntheory.powmod(7, 123456, 1000003) == pow(7, 123456, 1000003)
    assert mf.ntheory.primitive_root(23) == 5
    assert mf.ntheory.sqrt_mod(10, 13) == (6, 7)
    assert mf.ntheory.next_prime(3828000) == 3828007  # Past the Carmichael number 3828001.
    assert mf.ntheory.pell_fundamental(61) == (1766319049, 226153980)
    assert mf.ntheory.discrete_log(2, 3, 11).exponent == 8  # 2**8 = 256 = 3 (mod 11)
    assert mf.render_text(mf.ntheory.linear_diophantine(6, 9, 15)) == "(1 + 3*t, 1 - 2*t) for every integer t"

    # Algebra: real and complex roots, interpolation, nested radicals.
    assert mf.algebra.count_real_roots(P(x**5 - 3*x + 1), 0, 2) == 2
    assert len(mf.algebra.complex_roots(P(x**3 - 1))) == 3
    assert mf.render_text(mf.algebra.interpolate((0, 1, 2), (1, 3, 7), x)) == "x^2 + x + 1"
    assert mf.render_text(mf.algebra.denest_sqrt(3, 2, 2)) == "sqrt(2) + 1"

    # Equations with absolute values and square roots, and interval set algebra.
    assert mf.render_text(mf.equations.solve_abs(F("abs(x - 1) + abs(x + 2)"), F("3"), x)) == "[-2, 1]"
    assert mf.render_text(mf.equations.solve_sqrt(F("sqrt(x - 1)"), F("x - 3"), x, "<")) == "(5, inf)"
    a = mf.IntervalSet((mf.Interval(Q(0), Q(2), True, True),))   # [0, 2]
    b = mf.IntervalSet((mf.Interval(Q(1), Q(3), True, False),))  # [1, 3)
    assert mf.render_text(mf.sets.set_union(a, b)) == "[0, 3)"
    assert (a - b) == mf.sets.set_difference(a, b)

    # Decimal digits of exact numbers, without floating point.
    assert mf.render_text(mf.numeric.digits(mf.sqrt(2), 30)) == "1.414213562373095048801688724209..."
    assert mf.numeric.rational_approximation(mf.sqrt(2), 100) == Q(140, 99)

    # Sums, power series and trigonometric integrals.
    assert mf.render_text(mf.series.sum_polynomial(P(x**2))) == "1/3*x^3 + 1/2*x^2 + 1/6*x"
    assert mf.series.polynomial_geometric_series(P(x), Q(1, 2)) == Q(2)
    assert mf.render_text(mf.series.series_coefficients(F("exp(x)"), x, 0, 4)) == \
        "1 + x + 1/2*x^2 + 1/6*x^3 + 1/24*x^4 + O(x^5)"
    assert mf.render_text(mf.integrals.trig_power_integral(F("sin(x)^2"), x, 0, 1)) == "pi/2"

    # Linear algebra, vectors and plane geometry.
    M = mf.Matrix(((1, 1), (1, 0)))
    assert mf.linalg.matrix_power(M, 10) == M ** 10 == mf.Matrix(((89, 55), (55, 34)))
    spaces = mf.linalg.eigenvectors(mf.Matrix(((2, 1), (1, 2))))
    assert [space.value for space in spaces.eigenspaces] == [Q(1), Q(3)]
    assert mf.vector.cross((1, 2, 3), (4, 5, 6)) == (Q(-3), Q(6), Q(-3))
    assert mf.geometry.polygon_area(mf.Matrix(((0, 0), (4, 0), (0, 3)))) == Q(6)
    assert mf.geometry.line_intersection((0, 0), (1, 1), (0, 1), (1, 0)).point == (Q(1, 2), Q(1, 2))
    assert mf.linalg.smith_normal_form(mf.Matrix(((2, 4), (6, 8)))) == (2, 4)

    # Statistics and counting.
    assert mf.render_text(mf.stats.linear_regression((0, 1, 2), (1, 2, 4))) == "y = 3/2*x + 5/6"
    assert mf.discrete.partitions(100) == 190569292
    assert mf.discrete.derangements(10) == 1334961

    # Every operation has a structured twin whose result re-verifies from its JSON.
    result = mf.operations.eigenvectors(mf.Matrix(((2, 1), (1, 2))))
    assert mf.to_data(result)["version"] == 6
    assert mf.verify(mf.loads(mf.dumps(result))).verified

    # A question outside an operation's rules is deferred, with a reason, never guessed.
    cubic = mf.operations.complex_roots(P(x**3 - 2))
    assert cubic.outcome is mf.Outcome.NO_CONCLUSION and cubic.value is None

    # The MCP area tools run the same operations from JSON arguments.
    response = call_tool("mathforge.geometry", {
        "operation": "distance",
        "arguments": {"p": [0, 0], "q": [3, 4]},
    })
    assert not response.is_error, response.payload
    distance = mf.from_data(response.payload["result"])
    assert distance.value == Q(5) and mf.verify(distance).verified

    print("MathForge v0.9 workflow verified")


if __name__ == "__main__":
    main()
