"""Core micro benchmark: Rational arithmetic, the trusted constructor, budget charges and computation().

Run: ``python benchmarks/micro/core.py [--cpu N] [--rounds 9]``.

Targets (v0.8 Phase 1, Task 1):
- Rational + and x are at most 2.0x Fraction + without a budget, and at most 2.5x with one;
- entering and leaving ``computation()`` takes at most 1 microsecond.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

import mathforge  # noqa: E402
from mathforge.limits import Budget, ComputationLimits, computation  # noqa: E402
from mathforge.model import Rational, trusted_rational  # noqa: E402
from micro import Case, Target, arguments, measure, pin, report  # noqa: E402

# Large enough that no measurement exhausts a counter.
UNBOUNDED = ComputationLimits(max_work=10**15, max_nodes=10**15)
OPERATIONS = (("+", "a + b"), ("-", "a - b"), ("x", "a * b"), ("/", "a / b"))


def cases() -> list[Case]:
    a, b = Rational(1, 2), Rational(1, 3)
    budget = Budget(UNBOUNDED)

    def within_budget():
        return computation(UNBOUNDED, fresh=True)

    result = [Case("Fraction +", "a + b", {"a": Fraction(1, 2), "b": Fraction(1, 3)})]
    for symbol, statement in OPERATIONS:
        result.append(Case(f"Rational {symbol}", statement, {"a": a, "b": b}))
    for symbol, statement in OPERATIONS:
        result.append(Case(f"Rational {symbol} (budget)", statement, {"a": a, "b": b}, within_budget))
    result += [
        Case("Rational(3, 4)", "Rational(3, 4)", {"Rational": Rational}),
        Case("trusted_rational(3, 4)", "trusted_rational(3, 4)", {"trusted_rational": trusted_rational}),
        Case("Budget.tick()", "budget.tick()", {"budget": budget}),
        Case("Budget.inspect(Rational(1, 2))", "budget.inspect(a)", {"budget": budget, "a": a}),
        Case("with computation():", "with computation():\n    pass", {"computation": computation}),
    ]
    return result


def main() -> None:
    options = arguments(__doc__.split("\n\n")[0])
    if options.cpu is not None:
        pin(options.cpu)
    medians = measure(cases(), options.rounds)
    fraction = medians["Fraction +"]
    targets = [
        Target("Rational + / Fraction +", medians["Rational +"] / fraction, 2.0, "x"),
        Target("Rational x / Fraction +", medians["Rational x"] / fraction, 2.0, "x"),
        Target("Rational + (budget) / Fraction +", medians["Rational + (budget)"] / fraction, 2.5, "x"),
        Target("Rational x (budget) / Fraction +", medians["Rational x (budget)"] / fraction, 2.5, "x"),
        Target("with computation():", medians["with computation():"] / 1000, 1.0, " us"),
    ]
    cpu = "unpinned" if options.cpu is None else f"CPU {options.cpu}"
    title = (f"MathForge {mathforge.__version__} core micro benchmark: Python {platform.python_version()}, {cpu}, "
             f"{options.rounds} rounds")
    if not report(title, medians, targets):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
