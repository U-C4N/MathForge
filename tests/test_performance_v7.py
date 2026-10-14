"""Resource gates must count independent checking and fail on corrupt reports."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


def _gate():
    path = Path(__file__).resolve().parents[1] / "tools/check_benchmarks.py"
    spec = importlib.util.spec_from_file_location("mathforge_v07_budget_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PerformanceV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = _gate()
        cls.report = cls.gate.run_v07()

    def test_every_family_fits_reviewed_resource_ceilings(self):
        self.gate.check_v07_report(self.report)
        self.assertEqual(self.gate.BASELINE["mathforge_version"], "0.2.0")
        self.assertEqual(self.gate.V06_BASELINE["baseline_release"], "0.6.0")

    def test_measurements_are_deterministic_across_fresh_budgets(self):
        name, operation = self.gate.v07_cases()[0]
        first = self.gate.measure_v07(name, operation)
        second = self.gate.measure_v07(name, operation)
        self.assertEqual(first, second)

    def test_each_counter_and_status_is_a_real_gate(self):
        self.gate.check_v07_report(self.report)
        for counter in self.gate.V07_COUNTERS:
            for corruption in ("above", "negative", "bool", "missing"):
                bad = deepcopy(self.report)
                item = bad["measurements"][0]
                if corruption == "above":
                    index = self.gate.V07_COUNTERS.index(counter)
                    item[counter] = self.gate.V07_THRESHOLDS[item["case"]][index]+1
                elif corruption == "negative":
                    item[counter] = -1
                elif corruption == "bool":
                    item[counter] = True
                else:
                    del item[counter]
                with self.subTest(counter=counter, corruption=corruption), self.assertRaises((AssertionError, KeyError)):
                    self.gate.check_v07_report(bad)
        for name in ("certified_prime", "explicitly_deferred_integral", "intentional_v07_work_limit"):
            bad = deepcopy(self.report)
            item = next(v for v in bad["measurements"] if v["case"] == name)
            item["verified"] = not item["verified"]
            with self.assertRaises(AssertionError):
                self.gate.check_v07_report(bad)
        bad = deepcopy(self.report)
        bad["measurements"].pop()
        with self.assertRaises(AssertionError):
            self.gate.check_v07_report(bad)


if __name__ == "__main__":
    unittest.main()
