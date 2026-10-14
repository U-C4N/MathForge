"""Exact text -> existing verified operation -> presentation and persistence."""
import mathforge as mf


def main() -> None:
    x = mf.symbol("x")
    symbols = {"x": x}
    problem = mf.parse_problem("x^3 - 2 = 0", symbols=symbols)
    result = mf.solve(problem, for_=x)
    assert mf.verify(result).verified
    print(mf.render_text(result.unwrap()))
    print(mf.to_latex(result.unwrap()))
    assert mf.verify(mf.loads(mf.dumps(result))).verified
    assert mf.parse_expression("0.1", symbols={}) == mf.Rational(1, 10)
    hole = mf.parse_problem("(x^2 - 1)/(x - 1) = 2", symbols=symbols)
    assert mf.solve(hole, for_=x).outcome is mf.Outcome.NO_SOLUTION


if __name__ == "__main__":
    main()
