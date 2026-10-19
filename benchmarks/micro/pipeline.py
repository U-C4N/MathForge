"""Structured pipeline micro benchmark: the fixed cost of a structured call around its producer and checker.

Run: ``python benchmarks/micro/pipeline.py [--cpu N] [--rounds 9]``.

It times the structured functions of ``mathforge.operations`` (each returns a verified Result) on
``integer_gcd(84, 30)``, ``binomial(200, 80)`` and the legacy ``det`` of a 2x2 matrix.

Targets (v0.8 Phase 1, Task 3):
- structured ``integer_gcd(84, 30)`` takes at most 50 microseconds;
- structured ``binomial(200, 80)`` takes at most 80 microseconds.
"""

from __future__ import annotations

from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

import mathforge  # noqa: E402
from mathforge import operations  # noqa: E402
from micro import Case, Target, arguments, measure, pin, report  # noqa: E402

GCD, BINOMIAL, DET = "structured integer_gcd(84, 30)", "structured binomial(200, 80)", "structured det(2x2)"


def cases() -> list[Case]:
    matrix = mathforge.Matrix(((1, 2), (3, 4)))
    namespace = {"operations": operations, "matrix": matrix}
    # Each call must succeed: a failed Result would time the error path.
    for result, expected in ((operations.integer_gcd(84, 30), 6), (operations.binomial(200, 80), None),
                             (operations.det(matrix), mathforge.Rational(-2))):
        value = result.unwrap()
        if expected is not None and value != expected:
            raise SystemExit(f"unexpected value {value!r}, not {expected!r}")
    return [Case(GCD, "operations.integer_gcd(84, 30)", namespace),
            Case(BINOMIAL, "operations.binomial(200, 80)", namespace),
            Case(DET, "operations.det(matrix)", namespace)]


def main() -> None:
    options = arguments(__doc__.split("\n\n")[0])
    if options.cpu is not None:
        pin(options.cpu)
    medians = measure(cases(), options.rounds)
    targets = [Target(GCD, medians[GCD] / 1000, 50.0, " us"),
               Target(BINOMIAL, medians[BINOMIAL] / 1000, 80.0, " us")]
    cpu = "unpinned" if options.cpu is None else f"CPU {options.cpu}"
    title = (f"MathForge {mathforge.__version__} structured pipeline micro benchmark: Python "
             f"{platform.python_version()}, {cpu}, {options.rounds} rounds")
    if not report(title, medians, targets):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
