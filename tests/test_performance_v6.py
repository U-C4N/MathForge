"""Release gates reject resource growth, missing verification and silent failure."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

import mathforge as mf
from mathforge.limits import computation


def load_file(name, relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PerformanceV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load_file("mathforge_v06_gate_test", "tools/check_benchmarks.py")
        cls.benchmark = load_file("mathforge_v06_benchmark_test", "benchmarks/exact_algebra.py")

    def test_new_cases_fit_reviewed_ceilings_and_keep_old_baseline(self):
        self.assertEqual(len(self.gate.THRESHOLDS), 16)
        self.assertEqual(self.gate.BASELINE["mathforge_version"], "0.2.0")
        actual = self.benchmark.run()
        self.gate.check_v06_report(actual)
        self.assertEqual(len(actual["measurements"]), 21)
        self.assertTrue(actual["timing_is_non_gating"])
        self.assertTrue(actual["memory_is_non_gating"])
        names = {item["case"] for item in actual["measurements"]}
        self.assertIn("legacy_v04_function_analysis", names)
        self.assertIn("legacy_v05_hessian", names)
        for item in actual["measurements"]:
            self.assertGreaterEqual(item["seconds"], 0)
            self.assertGreater(item["peak_traced_bytes"], 0)

    def test_every_deterministic_counter_is_a_real_gate(self):
        report = deepcopy(self.gate.V06_BASELINE)
        self.gate.check_v06_report(report)
        for field in self.gate.V06_COUNTERS:
            for replacement in ("above", "negative", "bool", "missing"):
                bad = deepcopy(report)
                first = bad["measurements"][0]
                if replacement == "above":
                    first[field] = self.gate.V06_THRESHOLDS[first["case"]][field] + 1
                elif replacement == "negative":
                    first[field] = -1
                elif replacement == "bool":
                    first[field] = True
                else:
                    del first[field]
                with self.subTest(field=field, replacement=replacement), self.assertRaises(AssertionError):
                    self.gate.check_v06_report(bad)

    def test_status_error_verification_and_case_coverage_fail_closed(self):
        report = deepcopy(self.gate.V06_BASELINE)
        for change in ("status", "outcome", "error", "verification", "missing", "duplicate", "error_swallowed"):
            bad = deepcopy(report)
            first = bad["measurements"][0]
            if change == "status":
                first["execution_status"] = "error"
            elif change == "outcome":
                first["outcome"] = "partial"
            elif change == "error":
                first["error_code"] = "resource_limit_exceeded"
            elif change == "verification":
                first["verified"] = False
            elif change == "missing":
                bad["measurements"].pop()
            elif change == "duplicate":
                bad["measurements"][-1] = first
            else:
                limited = next(item for item in bad["measurements"] if item["case"] == "intentional_factor_work_limit")
                limited["execution_status"] = "completed"
                limited["error_code"] = None
            with self.subTest(change=change), self.assertRaises(AssertionError):
                self.gate.check_v06_report(bad)

    def test_wall_time_and_memory_are_observations_only(self):
        report = deepcopy(self.gate.V06_BASELINE)
        for item in report["measurements"]:
            item["seconds"] = 10**20
            item["peak_traced_bytes"] = 10**30
        self.gate.check_v06_report(report)

    def test_verifier_work_and_multiple_budgets_count_with_deterministic_runs(self):
        case = self.benchmark.cases()[0]
        first, second = self.benchmark.measure(case), self.benchmark.measure(case)
        for field in self.gate.V06_COUNTERS:
            self.assertEqual(first[field], second[field], field)
        from mathforge.polynomial_exact import factor_with_evidence
        x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000610")
        with computation(fresh=True) as budget:
            factor_with_evidence(mf.Polynomial((4, 0, 0, 0, 1), x))
            producer_work = budget.work
        self.assertGreater(first["work"], producer_work)

        def additional_budget():
            with computation(fresh=True) as budget:
                budget.tick(17)
            return case.operation()
        extended = self.benchmark.measure(self.benchmark.Case("extra_budget", additional_budget))
        self.assertEqual(extended["work"], first["work"]+17)


if __name__ == "__main__":
    unittest.main()
