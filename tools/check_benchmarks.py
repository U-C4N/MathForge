"""Gate frozen historical costs and explicit v0.7 work/evidence/transport budgets."""

import json
from pathlib import Path
import subprocess
import sys


BASELINE_PATH = Path(__file__).resolve().parents[1] / "benchmarks/baseline-python314.json"
BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
if BASELINE["mathforge_version"] != "0.2.0":
    raise AssertionError("The benchmark baseline must remain the frozen v0.2 release")
THRESHOLDS = {item["case"]: (item["work"] * 110 // 100, item["evidence_items"] * 110 // 100)
              for item in BASELINE["measurements"]}
if len(THRESHOLDS) != len(BASELINE["measurements"]):
    raise AssertionError("Duplicate baseline benchmark cases")

V06_PATH = Path(__file__).resolve().parents[1] / "benchmarks/v06-exact-algebra.json"
V06_BASELINE = json.loads(V06_PATH.read_text(encoding="utf-8"))
V06_COUNTERS = ("work", "nodes", "evidence_items", "output_json_bytes")
V06_THRESHOLDS = {item["case"]: item["ceilings"] for item in V06_BASELINE["measurements"]}
if len(V06_THRESHOLDS) != len(V06_BASELINE["measurements"]):
    raise AssertionError("Duplicate v0.6 benchmark cases")
if V06_BASELINE.get("baseline_release") != "0.6.0":
    raise AssertionError("The exact-algebra baseline must identify the v0.6 release")
for ceilings in V06_THRESHOLDS.values():
    if set(ceilings) != set(V06_COUNTERS) or any(type(v) is not int or v < 0 for v in ceilings.values()):
        raise AssertionError("Invalid exact-algebra resource ceilings")


def check_report(report):
    measurements = report["measurements"]
    if len(measurements) != len(THRESHOLDS) or {item["case"] for item in measurements} != set(THRESHOLDS):
        raise AssertionError("Benchmark cases changed; explicitly review the deterministic baseline")
    for measurement in measurements:
        name = measurement["case"]
        expected_error = name == "intentional_root_work_limit"
        if expected_error:
            assert measurement["execution_status"] == "error", name
            assert measurement["error_code"] == "resource_limit_exceeded", name
        else:
            assert measurement["execution_status"] == "completed", (name, measurement)
            assert measurement["error_code"] is None, (name, measurement)
        for counter, ceiling in zip(("work", "evidence_items"), THRESHOLDS[name], strict=True):
            value = measurement[counter]
            assert type(value) is int and 0 <= value <= ceiling, (name, counter, value, ceiling)


def check_v06_report(report):
    measurements = report["measurements"]
    assert len(measurements) == len(V06_THRESHOLDS), "Changed v0.6 benchmark case count"
    assert {item["case"] for item in measurements} == set(V06_THRESHOLDS), "Changed v0.6 benchmark cases"
    expected = {item["case"]: item for item in V06_BASELINE["measurements"]}
    for item in measurements:
        name = item["case"]
        for field in ("execution_status", "outcome", "error_code"):
            assert field in item and item[field] == expected[name][field], (name, field, item)
        assert item.get("verified") is expected[name]["verified"], (name, "verification")
        for counter in V06_COUNTERS:
            assert counter in item, (name, "missing counter", counter)
            value, ceiling = item[counter], V06_THRESHOLDS[name][counter]
            minimum = 1 if counter == "output_json_bytes" else 0
            assert type(value) is int and minimum <= value <= ceiling, (name, counter, value, ceiling)


# Initial v0.7 measurements, 2026-10-14: roughly 25% headroom, rounded upward.
# Counts include every independent verifier and serialization budget; elapsed
# time and process memory are not gates. Historical baselines stay frozen.
V07_COUNTERS = ("work", "nodes", "evidence_items", "output_json_bytes")
V07_THRESHOLDS = {
    "certified_prime": (600, 25, 270, 2_300),
    "integer_factorization": (100, 80, 500, 4_100),
    "multinomial": (75, 45, 215, 2_200),
    "complex_algebraic_sqrt": (480_000, 35_000, 22_000, 46_000),
    "finite_distribution": (3_200, 320, 700, 6_100),
    "least_squares": (3_250, 425, 960, 6_350),
    "elementary_gradient": (770, 325, 2_100, 19_500),
    "rational_quintic_primitive": (1_650_000, 69_000, 2_350, 20_200),
    "explicitly_deferred_integral": (1_200, 190, 150, 4_150),
    "intentional_v07_work_limit": (2, 15, 35, 800),
}


def v07_cases():
    import mathforge as mf
    from mathforge import functions as f
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000710")
    y = mf.Symbol("y", uid="00000000-0000-4000-8000-000000000711")
    source = mf.real_function(f.sin(x)*f.exp(y), (x, y))
    deferred = mf.real_function(f.sin(f.as_real_expr(x)**2), (x,))
    rational = mf.rational_function(1, x**5-x+1, variable=x)
    A = mf.Matrix(((1, 2), (2, 4), (1, 1)))
    z = mf.ComplexAlgebraic(1, 1)
    return (
        ("certified_prime", lambda: mf.operations.is_prime(65537)),
        ("integer_factorization", lambda: mf.operations.factor_integer(-360360)),
        ("multinomial", lambda: mf.operations.multinomial((12, 8, 5))),
        ("complex_algebraic_sqrt", lambda: mf.operations.complex_sqrt(z)),
        ("finite_distribution", lambda: mf.operations.binomial_distribution(12, mf.Rational(1, 3))),
        ("least_squares", lambda: mf.operations.least_squares(A, (1, 3, 2))),
        ("elementary_gradient", lambda: mf.operations.function_gradient(source, (x, y))),
        ("rational_quintic_primitive", lambda: mf.operations.rational_antiderivative(rational, x)),
        ("explicitly_deferred_integral", lambda: mf.operations.function_antiderivative(deferred, x)),
        ("intentional_v07_work_limit", lambda: mf.operations.factorial(100, limits=mf.ComputationLimits(max_work=1))),
    )


def measure_v07(name, operation):
    import mathforge as mf
    from mathforge import limits as limit_module
    from unittest.mock import patch
    operation()  # Import warming and fixture construction are outside the counters.
    budgets = []
    original = limit_module.Budget

    class RecordedBudget(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            budgets.append(self)

    with patch.object(limit_module, "Budget", RecordedBudget):
        result = operation()
        serialized = mf.dumps(result).encode("utf-8")
        verified = mf.verify(result).verified
    return {"case": name, "execution_status": result.execution_status.value,
        "outcome": result.outcome.value, "error_code": None if result.error is None else result.error.code,
        "verified": verified, "deferred": result.deferred is not None,
        "work": sum(b.work for b in budgets), "nodes": sum(b.node_count for b in budgets),
        "evidence_items": sum(b.evidence_count for b in budgets), "output_json_bytes": len(serialized)}


def run_v07():
    return {"measurements": [measure_v07(name, operation) for name, operation in v07_cases()]}


def check_v07_report(report):
    measurements = report["measurements"]
    assert len(measurements) == len(V07_THRESHOLDS)
    assert {item["case"] for item in measurements} == set(V07_THRESHOLDS)
    for item in measurements:
        name = item["case"]
        failed = name == "intentional_v07_work_limit"
        deferred = name == "explicitly_deferred_integral"
        assert item["execution_status"] == ("error" if failed else "completed"), item
        assert item["outcome"] == ("not_applicable" if failed else "no_conclusion" if deferred else "value"), item
        assert item["error_code"] == ("resource_limit_exceeded" if failed else None), item
        assert item["verified"] is (not failed and not deferred), item
        assert item["deferred"] is deferred, item
        for counter, ceiling in zip(V07_COUNTERS, V07_THRESHOLDS[name]):
            value = item[counter]
            assert type(value) is int and (1 if counter == "output_json_bytes" else 0) <= value <= ceiling, (name, counter, value, ceiling)


def main():
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output([sys.executable, str(root / "benchmarks/exact_core.py")],
                                     cwd=root, text=True)
    report = json.loads(output)
    check_report(report)
    print(f"{len(THRESHOLDS)} exact benchmark cases passed status and <=10% work/evidence growth gates (timing is not gated).")
    output = subprocess.check_output([sys.executable, str(root / "benchmarks/exact_algebra.py")],
                                     cwd=root, text=True)
    check_v06_report(json.loads(output))
    print(f"{len(V06_THRESHOLDS)} v0.6 cases passed work/node/evidence/output-byte ceilings and exact status checks "
          "(timing and observed peak memory are not gated).")
    check_v07_report(run_v07())
    print(f"{len(V07_THRESHOLDS)} v0.7 cases passed deterministic producer/verifier/transport budgets and deferred-result checks.")


if __name__ == "__main__":
    main()
