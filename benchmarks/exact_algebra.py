"""v0.6 exact-algebra resource measurements, including independent verification.

Every fresh Budget is recorded, including nested verification and serialization
budgets. Deterministic work/nodes/evidence/JSON bytes are release gates. Elapsed
time and tracemalloc peak allocation are observations, never thresholds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
from time import perf_counter
import tracemalloc
from unittest.mock import patch

import mathforge as mf
from mathforge import limits as limit_module


@dataclass(frozen=True)
class Case:
    name: str
    operation: object
    execution_status: str = "completed"
    error_code: str | None = None


def cases():
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000610")
    y = mf.Symbol("y", uid="00000000-0000-4000-8000-000000000611")
    poly = lambda *c: mf.Polynomial(c, x)
    sqrt2, sqrt3 = mf.sqrt(2), mf.sqrt(3)
    sophie, cyclotomic = poly(4, 0, 0, 0, 1), poly(1, 0, 0, 0, 1)
    nonmonic = (poly(1, 2)**2 * poly(2, 0, 1)) / mf.Rational(7)
    quintic, cubic = poly(-1, -1, 0, 0, 0, 1), poly(2, 1, 0, 1)
    partial = mf.rational_function(x+1, (x-1)**2*(x*x+1), variable=x)
    evaluate = mf.rational_function(x+1, x*x+1, variable=x)
    hole = mf.rational_function(x*x-2, x*x-2, variable=x)
    legacy = mf.rational_function(x*x-1, x-2, variable=x)
    matrix = mf.Matrix(((2, 1, 0, 0), (1, 2, 1, 0), (0, 1, 2, 1), (0, 0, 1, 2)))
    spectrum = mf.Matrix(((0, 2, 0), (1, 0, 0), (0, 0, 1)))
    tiny = mf.ComputationLimits(max_work=10)
    return (
        Case("factor_sophie_germain", lambda: mf.operations.factor(sophie)),
        Case("factor_irreducible_cyclotomic", lambda: mf.operations.factor(cyclotomic)),
        Case("factor_nonmonic_repeated", lambda: mf.operations.factor(nonmonic)),
        Case("resultant_degree5_degree3", lambda: mf.operations.resultant(quintic, cubic)),
        Case("discriminant_degree5", lambda: mf.operations.discriminant(quintic)),
        Case("algebraic_independent_quadratics", lambda: mf.operations.algebraic(sqrt2+sqrt3)),
        Case("algebraic_inverse", lambda: mf.operations.algebraic_div(1, sqrt2+sqrt3)),
        Case("algebraic_negative_power", lambda: mf.operations.algebraic_pow(1+sqrt2, -2)),
        Case("minimal_polynomial_quartic", lambda: mf.operations.minimal_polynomial(sqrt2+sqrt3, x)),
        Case("evaluate_algebraic_rational", lambda: mf.operations.evaluate_algebraic(evaluate, {x: sqrt2})),
        Case("apart_repeated_and_irreducible", lambda: mf.operations.apart(partial)),
        Case("charpoly_tridiagonal4", lambda: mf.operations.charpoly(matrix, x)),
        Case("real_eigenvalues_mixed3", lambda: mf.operations.real_eigenvalues(spectrum)),
        Case("limit_algebraic_cancelled_hole", lambda: mf.operations.limit(hole, x, sqrt2)),
        Case("integrate_algebraic_endpoint", lambda: mf.operations.integrate(x*x, x, 0, sqrt2)),
        Case("extrema_algebraic_values", lambda: mf.operations.extrema_values(x**3-2*x, x)),
        Case("legacy_v04_function_analysis", lambda: mf.operations.analyze_function(legacy, x)),
        Case("legacy_v05_hessian", lambda: mf.operations.hessian((x*x+y*y)**3, (x, y))),
        Case("intentional_factor_work_limit", lambda: mf.operations.factor(cyclotomic, limits=tiny),
             "error", "resource_limit_exceeded"),
        Case("intentional_algebraic_work_limit", lambda: mf.operations.algebraic(sqrt2+sqrt3, limits=tiny),
             "error", "resource_limit_exceeded"),
        Case("factor_zero_invalid", lambda: mf.operations.factor(poly(0)), "invalid_input", "invalid_input"),
    )


def measure(case):
    case.operation()  # Warm imports outside recorded budgets and memory samples.
    budgets = []
    original = limit_module.Budget

    class RecordedBudget(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            budgets.append(self)

    tracemalloc.start()
    start = perf_counter()
    try:
        with patch.object(limit_module, "Budget", RecordedBudget):
            result = case.operation()
            serialized = mf.dumps(result).encode("utf-8")
        elapsed = perf_counter()-start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    error = result.error.code if result.error is not None else None
    verified = bool(result.verification) and all(check.status is mf.CheckStatus.VERIFIED for check in result.verification)
    assert result.execution_status.value == case.execution_status, (case.name, result.error)
    assert error == case.error_code, (case.name, error)
    if case.execution_status == "completed":
        assert result.value is not None and verified, (case.name, "missing value or verification")
        assert result.exactness is mf.Exactness.EXACT and result.completeness is mf.Completeness.COMPLETE, case.name
    else:
        assert result.value is None and not verified, case.name
    return {
        "case": case.name,
        "execution_status": result.execution_status.value,
        "outcome": result.outcome.value,
        "error_code": error,
        "verified": verified,
        "work": sum(budget.work for budget in budgets),
        "nodes": sum(budget.node_count for budget in budgets),
        "evidence_items": sum(budget.evidence_count for budget in budgets),
        "output_json_bytes": len(serialized),
        "seconds": elapsed,
        "peak_traced_bytes": peak,
    }


def run():
    return {
        "mathforge_version": mf.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timing_is_non_gating": True,
        "memory_is_non_gating": True,
        "warmups_per_case": 1,
        "counter_scope": "All producer, independent verifier and serialization budgets; excludes fixture construction",
        "measurements": [measure(case) for case in cases()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(run(), indent=2)
    print(output)
    if args.output:
        args.output.write_text(output+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
