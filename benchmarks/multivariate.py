"""Reproduce checked v0.5 polynomial timings; elapsed time is never a CI gate."""

import argparse
import json
from math import ceil
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import mathforge as mf
from mathforge.limits import computation


def measure(name, operation, repeat):
    operation().unwrap()  # Warm imports and code paths outside timed samples.
    samples = []
    counters = []
    for _ in range(repeat):
        with computation(fresh=True) as budget:
            start = perf_counter()
            result = operation()
            elapsed = perf_counter() - start
            result.unwrap()
            counters.append((budget.work, budget.node_count, budget.evidence_count))
        samples.append(elapsed)
    if len(set(counters)) != 1:
        raise AssertionError(f"Nondeterministic work counters: {name}")
    return {
        "case": name, "seconds": samples, "median_seconds": median(samples),
        "p95_seconds": sorted(samples)[ceil(0.95 * len(samples)) - 1],
        "work": counters[0][0], "nodes": counters[0][1], "evidence_items": counters[0][2],
        "execution_status": result.execution_status.value,
        "verification": mf.verify(result).verified,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    x, y, z = (mf.Symbol(name, uid=f"00000000-0000-4000-8000-00000000050{index}")
               for index, name in enumerate(("x", "y", "z")))
    axes = (x, y, z)
    dense = (x+y+z)**12
    radial = (x**2+y**2+z**2)**4
    functions = ((x+y)**4, (y+z)**4, (z+x)**4, x*y*z)
    sparse = x**1_000_000*y**3 + z
    cases = (
        ("expand_three_variables_degree_12", lambda: mf.operations.expand_multivariate(dense, axes)),
        ("gradient_radial_degree_8", lambda: mf.operations.gradient(radial, axes)),
        ("hessian_radial_degree_8", lambda: mf.operations.hessian(radial, axes)),
        ("jacobian_four_functions_three_variables", lambda: mf.operations.jacobian(functions, axes)),
        ("partial_sparse_degree_1000003", lambda: mf.operations.partial_diff(sparse, x, n=2)),
    )
    report = {
        "mathforge_version": mf.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "repeat": args.repeat, "warmups_per_case": 1,
        "timing_is_non_gating": True,
        "timed_scope": "Structured SDK operation including independent verification; excludes parsing and serialization",
        "measurements": [measure(name, operation, args.repeat) for name, operation in cases],
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
