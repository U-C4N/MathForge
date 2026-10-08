"""Run after installing MathForge; optionally retain the exported workspace."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import mathforge as mf


def run(path: Path) -> None:
    x = mf.symbol("x", domain=mf.Reals)
    expression = x**2 - 2
    problem = mf.Eq(expression, 0)
    assert mf.substitute(expression, {x: 2}) == mf.Rational(2)

    derivative = mf.diff(expression, x)
    value = mf.evaluate(derivative, {x: mf.Rational(3, 2)})
    assert value == mf.Rational(3)

    result = mf.solve(problem, for_=x)
    assert result.execution_status is mf.ExecutionStatus.COMPLETED
    assert result.completeness is mf.Completeness.COMPLETE
    assert result.exactness is mf.Exactness.EXACT
    assert set(result.solution_set.values) == {-mf.sqrt(2), mf.sqrt(2)}
    assert mf.verify(result).verified

    workspace = mf.Workspace()
    expression_ref = workspace.put(expression)
    problem_ref = workspace.put(problem)
    variable_ref = workspace.put(x)
    result_ref = workspace.put(result)
    workspace.record("solve", {"problem": problem_ref, "for_": variable_ref}, {}, result_ref)
    workspace.save(path)

    restored = mf.Workspace.load(path)
    loaded_result = restored.get(result_ref)
    assert restored.get(expression_ref) == expression
    assert loaded_result == result
    assert mf.verify(loaded_result).verified
    assert restored.history == workspace.history

    print(f"Problem: {problem}")
    print(f"Derivative: {derivative}")
    print(f"Derivative at 3/2: {value}")
    print(f"Exact solutions: {result.solution_set}")
    print(f"Method: {result.method}")
    print(f"Steps: {len(result.steps)}; independent verification: passed")
    print(f"Workspace round trip: passed ({path})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Keep the workspace JSON at this path.")
    args = parser.parse_args()
    if args.output:
        run(args.output)
    else:
        with TemporaryDirectory(prefix="mathforge-example-") as directory:
            run(Path(directory) / "workspace.json")


if __name__ == "__main__":
    main()
