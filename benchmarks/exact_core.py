"""REL-14: Reproducible, non-gating exact-core measurements.

Run with an installed MathForge: ``python benchmarks/exact_core.py``.  The JSON
report includes deterministic work counters and actual execution status. Times
are machine-specific observations; no timing threshold or superiority is claimed.
Integer size is the maximum present in stored inputs/results, not an estimate of
unobserved transient coefficients during the algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import mathforge as mf
from mathforge.limits import ComputationLimits, computation, current_budget


def stored_integer_bits(value) -> int:
    pending = [value]
    largest = 0
    while pending:
        item = pending.pop()
        if type(item) is int:
            largest = max(largest, abs(item).bit_length())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
    return largest


def benchmark(name, inputs, operation, *, limits=None, expected_error=None):
    with computation(limits, fresh=True):
        budget = current_budget()
        start = perf_counter()
        result = operation()
        elapsed = perf_counter() - start
        counters = {"work": budget.work, "nodes": budget.node_count,
                    "refinements": budget.refinements, "evidence_items": budget.evidence_count}
    error_code = result.error.code if result.error is not None else None
    if expected_error is not None and error_code != expected_error:
        raise AssertionError((name, "expected", expected_error, "actual", error_code))
    return {
        "case": name,
        "seconds": elapsed,
        "execution_status": result.execution_status.value,
        "outcome": result.outcome.value,
        "error_code": error_code,
        "error_message": result.error.message if result.error is not None else None,
        **counters,
        "stored_input_or_result_integer_bits": stored_integer_bits((inputs, result)),
        "certificate_json_bytes": len(mf.dumps(result.steps).encode("utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    parser.add_argument("--quick", action="store_true", help="Use only small smoke cases")
    arguments = parser.parse_args()
    # A fixed UUID makes the recorded mathematical inputs reproducible.
    x = mf.Symbol("x", uid="01234567-89ab-4cde-8fab-0123456789ab")
    measurements = []
    for size in ((4, 8) if arguments.quick else (4, 8, 16, 32)):
        A = mf.Matrix(tuple(tuple(size + 1 if i == j else (1 if abs(i - j) == 1 else 0)
                                  for j in range(size)) for i in range(size)))
        measurements.append(benchmark(f"rref_tridiagonal_{size}", A,
                                      lambda A=A: mf.operations.rref(A)))
        measurements.append(benchmark(f"det_tridiagonal_{size}", A,
                                      lambda A=A: mf.operations.det(A)))
    root_cases = [
        ("roots_quadratic", x**2 - 2),
        ("roots_quintic_irrational", x**5 - x - 1),
    ]
    if not arguments.quick:
        for degree in (10, 20):
            expression = mf.Rational(1)
            for root in range(degree):
                expression = expression * (x - root)
            root_cases.append((f"roots_known_degree_{degree}", expression))
        root_cases.extend((
            ("roots_repeated_degree_20", (x - 1)**10 * (x + 2)**10),
            ("roots_clustered_rational", (1000*x - 1)*(1000*x - 2)*(1000*x - 3)),
        ))
    for name, expression in root_cases:
        measurements.append(benchmark(name, expression,
                                      lambda expression=expression: mf.operations.real_roots(expression, x)))
    denominator = (x - 1) * (x - 2) * (x - 3)
    rf = mf.rational_function(denominator * (x + 1), denominator, variable=x)
    inequality = mf.Inequality(rf, "ge", 0)
    measurements.append(benchmark("rational_inequality_three_holes", inequality,
                                  lambda: mf.operations.solve(inequality, for_=x)))
    small = ComputationLimits(max_work=10)
    quintic = x**5 - x - 1
    measurements.append(benchmark("intentional_root_work_limit", quintic,
                                  lambda: mf.operations.real_roots(quintic, x, limits=small),
                                  limits=small, expected_error="resource_limit_exceeded"))
    report = {
        "mathforge_version": mf.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timing_is_non_gating": True,
        "integer_metric": "Maximum integer bit length in stored inputs/results, including trace values",
        "measurements": measurements,
    }
    text = json.dumps(report, indent=2, ensure_ascii=True)
    print(text)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
