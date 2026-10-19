"""tools/check_coverage.py: the prechecks, the frozen-inventory guard, the freeze manifest, the report and the gate.

Every test builds a synthetic inventory, examples file and freeze manifest in a temporary directory, and passes
synthetic Facts and stand-in adapters; none reads the local docs/ inventory or writes tools/coverage_gate/freeze.json.
None needs SymPy.
"""

import contextlib
import dataclasses
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))  # The coverage gate is tooling beside the scripts, not a package.

import check_coverage  # noqa: E402
from coverage_gate import criteria  # noqa: E402
from coverage_gate import inventory as model  # noqa: E402
from coverage_gate.examples import Example, load_examples, write_examples  # noqa: E402

import mathforge as mf  # noqa: E402

VERIFIED = mf.operations.continued_fraction(mf.Rational(415, 93))  # A real Result: it round-trips and verifies.
OPERATIONS = frozenset({"op_yes", "op_slow", "op_shallow"})
# op_slow misses only the speed limit; op_shallow's row misses only depth (its adapter never matches).
FACTS = criteria.Facts(operations=OPERATIONS, checkers=OPERATIONS, mcp=OPERATIONS, documented=OPERATIONS,
                       differential=OPERATIONS, speed={"op_yes": 1.0, "op_slow": 4.0, "op_shallow": 2.0})
ADAPTERS = {"91.01": lambda example: (criteria.Outcome.MATCH, VERIFIED),
            "91.02": lambda example: (criteria.Outcome.MATCH, VERIFIED),
            "91.03": lambda example: (criteria.Outcome.MISMATCH, VERIFIED)}


def make_row(row_id, area, v07_status, tier, plan_group, wave, family, operations, missing=""):
    return model.Row(id=row_id, area=area, subtopic=f"Subtopic {row_id}", sympy_apis="f", sympy_api_paths=("sympy.f",),
                     v07_status=v07_status, tier=tier, note="", plan_group=plan_group, wave=wave, family=family,
                     missing=missing, sympy_probe="f(1)", mathforge_operations=tuple(operations), status="no")


# The four rows: yes, partial (all but speed), partial (all but depth), no (no operations).
ROWS = (make_row("91.01", "Synthetic algebra", "yes", None, "v07", None, None, ("op_yes",)),
        make_row("91.02", "Synthetic algebra", "partial", None, "partial", "A", "A1", ("op_slow",), missing="speed"),
        make_row("91.03", "Synthetic algebra", "no", "T1", "t1", "B", "B2", ("op_shallow",)),
        make_row("92.01", "Synthetic geometry", "no", "T3", "out_of_scope", None, None, ()))


def make_examples(row_id, count=3):
    return [Example(id=f"{row_id}/{n:02d}", row=row_id, source=f"docstring:sympy.f:{n}", setup="", code=f"f({n})",
                    output=str(n), srepr=f"Integer({n})", kind="exact") for n in range(1, count + 1)]


class CheckCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.inventory = self.directory / "subtopics.json"
        self.examples = self.directory / "reference_examples.json"  # The default: beside the inventory.
        self.manifest = self.directory / "freeze.json"
        self.reports = self.directory / "coverage"  # The default report directory.
        model.write_inventory(self.inventory, model.Inventory({"inventory": "synthetic", "frozen": False}, ROWS))
        write_examples(self.examples, {row.id: make_examples(row.id) for row in ROWS})
        patcher = mock.patch.dict(criteria.ADAPTERS, ADAPTERS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_tool(self, *arguments, facts=FACTS, today="2026-10-18"):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = check_coverage.main([str(self.inventory), *arguments], manifest=self.manifest, facts=facts,
                                       today=today)
        return code, stdout.getvalue(), stderr.getvalue()

    def digest(self):
        return model.frozen_digest(model.load_inventory(self.inventory), load_examples(self.examples))

    def replace_row(self, index, **changes):
        inventory = model.load_inventory(self.inventory)
        rows = list(inventory.rows)
        rows[index] = dataclasses.replace(rows[index], **changes)
        model.write_inventory(self.inventory, dataclasses.replace(inventory, rows=tuple(rows)))

    def test_report_totals(self):
        code, out, err = self.run_tool()
        self.assertEqual(code, 0, err)
        report = json.loads((self.reports / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["totals"], {"yes": 1, "partial": 2, "no": 1, "yes_without_speed": 2})
        self.assertEqual((report["denominator"], report["examples"], report["ratio"]), (4, 12, 0.25))
        self.assertEqual(report["thresholds"], {"0.50": 2, "0.55": 3})  # ceil(0.5 x 4), ceil(0.55 x 4)
        self.assertEqual((report["digest"], report["frozen"]), (self.digest(), None))
        self.assertEqual([(row["id"], row["status"]) for row in report["rows"]],
                         [("91.01", "yes"), ("91.02", "partial"), ("91.03", "partial"), ("92.01", "no")])
        unmet = {row["id"]: [name for name, met in row["criteria"].items() if not met] for row in report["rows"]}
        self.assertEqual(unmet["91.02"], ["speed"])
        self.assertEqual(unmet["91.03"], ["depth"])
        self.assertEqual(report["rows"][2]["details"], ["depth: 0 of 3 examples match; 3 needed", "91.03/01: mismatch",
                                                        "91.03/02: mismatch", "91.03/03: mismatch"])
        markdown = (self.reports / "report.md").read_text(encoding="utf-8")
        for line in ("| yes | 1 | 25.0% |", "| partial | 2 | 50.0% |", "| no | 1 | 25.0% |",
                     "| yes without speed | 2 | 50.0% |", "Denominator: 4 rows (12 reference examples, 4 short sets).",
                     "50% needs 2 yes rows and 55% needs 3; 1 row is yes: below the 50% gate.",
                     "| Synthetic algebra | 3 | 1 | 2 | 0 | 2 |", "| Synthetic geometry | 1 | 0 | 0 | 1 | 0 |",
                     "| 91.02 | Subtopic 91.02 | partial | speed |", "| 92.01 | Subtopic 92.01 | no | capabilities, checker, "
                     "json, mcp, docs, differential, speed, depth |", "| speed | 2 |", "| t1 | 1 | 0 | 1 | 0 |"):
            self.assertIn(line, markdown)
        self.assertIn("1 of 4 rows yes (25.0%)", out)
        # The same inputs give the same bytes.
        first = [(self.reports / name).read_bytes() for name in ("report.json", "report.md")]
        self.assertEqual(self.run_tool()[0], 0)
        self.assertEqual([(self.reports / name).read_bytes() for name in ("report.json", "report.md")], first)

    def test_default_and_explicit_paths(self):
        with mock.patch.object(criteria, "gather_facts", return_value=FACTS) as gather:
            self.assertEqual(self.run_tool(facts=None)[0], 0)
        gather.assert_called_once_with(check_coverage.REPOSITORY, speed_file=self.directory / "speed" / "latest.json")
        examples, reports, speed = self.directory / "sets.json", self.directory / "out", self.directory / "fast.json"
        self.examples.rename(examples)
        with mock.patch.object(criteria, "gather_facts", return_value=FACTS) as gather:
            code, out, err = self.run_tool("--examples", str(examples), "--report", str(reports), "--speed", str(speed),
                                           facts=None)
        self.assertEqual(code, 0, err)
        gather.assert_called_once_with(check_coverage.REPOSITORY, speed_file=speed)
        self.assertTrue((reports / "report.json").is_file() and (reports / "report.md").is_file())

    def test_gate_exit_codes(self):
        code, out, err = self.run_tool("--gate")
        self.assertEqual(code, 2)  # The gate counts the frozen inventory only.
        self.assertIn("--freeze", err)
        self.assertEqual(self.run_tool("--freeze")[0], 0)
        code, out, err = self.run_tool("--gate")
        self.assertEqual(code, 1)  # 1 of 4 rows: 25% < 50%.
        self.assertIn("below the 50% gate", out + err)
        self.assertTrue((self.reports / "report.md").is_file())  # A failed gate still writes its report.
        fast = dataclasses.replace(FACTS, speed={**FACTS.speed, "op_slow": 3.0})  # 91.02 turns yes: 2 of 4 = 50%.
        code, out, err = self.run_tool("--gate", facts=fast)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.run_tool(facts=dataclasses.replace(FACTS, speed={}))[0], 0)  # No --gate: no gate.

    def test_digest_mismatch_stops(self):
        self.assertEqual(self.run_tool("--freeze")[0], 0)
        frozen = json.loads(self.manifest.read_text(encoding="utf-8"))["digest"]
        original_inventory, original_examples = self.inventory.read_bytes(), self.examples.read_bytes()
        # A frozen field changed by hand or by a merge: the inventory ...
        self.replace_row(2, sympy_probe="f(2)")
        code, out, err = self.run_tool()
        self.assertEqual(code, 2)
        self.assertIn(f"Frozen inventory changed: {frozen} != {self.digest()}", err)
        self.assertFalse(self.reports.exists())  # Nothing is evaluated or written.
        self.inventory.write_bytes(original_inventory)
        # ... or a recorded example.
        examples = load_examples(self.examples)
        examples["91.03"][0] = dataclasses.replace(examples["91.03"][0], output="9", srepr="Integer(9)")
        write_examples(self.examples, examples)
        code, out, err = self.run_tool("--write-status")
        self.assertEqual(code, 2)
        self.assertIn(f"Frozen inventory changed: {frozen} != {self.digest()}", err)
        self.assertEqual(self.inventory.read_bytes(), original_inventory)  # --write-status wrote nothing.
        self.examples.write_bytes(original_examples)
        # The fields that stay editable after the freeze do not count.
        self.replace_row(0, status="partial", note="curated", mathforge_operations=("op_yes", "op_slow"))
        code, out, err = self.run_tool()
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads((self.reports / "report.json").read_text(encoding="utf-8"))["frozen"], "2026-10-18")

    def test_refreeze_requires_reason(self):
        code, out, err = self.run_tool("--freeze")
        self.assertEqual(code, 0, err)
        first = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(first, {"digest": self.digest(), "rows": 4, "examples": 12, "frozen": "2026-10-18",
                                 "history": []})
        code, out, err = self.run_tool("--freeze")
        self.assertEqual(code, 2)
        self.assertIn("--reason", err)
        code, out, err = self.run_tool("--freeze", "--reason", "nothing changed")
        self.assertEqual(code, 2)  # An unchanged digest has nothing to re-freeze.
        self.assertEqual(json.loads(self.manifest.read_text(encoding="utf-8")), first)
        # A recorded output is corrected: a re-freeze needs the reason, which joins the history with the old record.
        examples = load_examples(self.examples)
        examples["91.03"][0] = dataclasses.replace(examples["91.03"][0], output="7", srepr="Integer(7)")
        write_examples(self.examples, examples)
        self.assertEqual(self.run_tool("--freeze")[0], 2)
        self.assertEqual(self.run_tool()[0], 2)  # Until then the guard stops every run.
        code, out, err = self.run_tool("--freeze", "--reason", "91.03/01 was recorded wrongly", today="2026-10-20")
        self.assertEqual((code, err), (0, ""))  # The same counts: nothing to warn about.
        second = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(second, {"digest": self.digest(), "rows": 4, "examples": 12, "frozen": "2026-10-20",
                                  "history": [{"digest": first["digest"], "rows": 4, "examples": 12,
                                               "date": "2026-10-18", "reason": "91.03/01 was recorded wrongly"}]})
        self.assertEqual(list(second["history"][0]), ["digest", "rows", "examples", "date", "reason"])
        self.assertEqual(self.run_tool()[0], 0)
        self.assertTrue(self.manifest.read_bytes().endswith(b"}\n") and b"\r" not in self.manifest.read_bytes())
        # --reason belongs to a re-freeze only.
        for arguments in (("--reason", "why"), ("--freeze", "--gate"), ("--freeze", "--write-status")):
            with self.subTest(arguments), self.assertRaises(SystemExit) as raised:
                self.run_tool(*arguments)
            self.assertEqual(raised.exception.code, 2)
        self.manifest.unlink()
        code, out, err = self.run_tool("--freeze", "--reason", "first")
        self.assertEqual(code, 2)
        self.assertFalse(self.manifest.exists())

    def test_refreeze_records_the_replaced_counts_and_warns_when_they_shrink(self):
        self.assertEqual(self.run_tool("--freeze")[0], 0)
        first = json.loads(self.manifest.read_text(encoding="utf-8"))
        # One example fewer: the re-freeze is written, with a warning that names what shrank.
        examples = load_examples(self.examples)
        examples["91.03"] = examples["91.03"][:2]
        write_examples(self.examples, examples)
        code, out, err = self.run_tool("--freeze", "--reason", "91.03/03 repeated 91.03/02", today="2026-10-20")
        self.assertEqual(code, 0, err)
        self.assertEqual(err.strip(), "warning: the new freeze has fewer examples (12 -> 11) than the one it "
                                      "replaces; a re-freeze corrects a recorded error and never drops a hard example "
                                      "or shrinks a set")
        second = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual((second["rows"], second["examples"]), (4, 11))
        self.assertEqual(second["history"], [{"digest": first["digest"], "rows": 4, "examples": 12,
                                              "date": "2026-10-18", "reason": "91.03/03 repeated 91.03/02"}])
        # A row fewer (92.01, with its three examples): both counts are named.
        inventory = model.load_inventory(self.inventory)
        model.write_inventory(self.inventory, dataclasses.replace(inventory, rows=inventory.rows[:3]))
        write_examples(self.examples, {key: value for key, value in examples.items() if key != "92.01"})
        code, out, err = self.run_tool("--freeze", "--reason", "92.01 is not in SymPy 1.14", today="2026-10-21")
        self.assertEqual(code, 0, err)
        self.assertIn("warning: the new freeze has fewer rows (4 -> 3) and fewer examples (11 -> 8) than the one it "
                      "replaces", err)
        third = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(third["history"][1], {"digest": second["digest"], "rows": 4, "examples": 11,
                                               "date": "2026-10-20", "reason": "92.01 is not in SymPy 1.14"})
        # More examples than before: no warning.
        write_examples(self.examples, {**load_examples(self.examples), "91.03": make_examples("91.03", 5)})
        code, out, err = self.run_tool("--freeze", "--reason", "91.03 gains two examples", today="2026-10-22")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(self.manifest.read_text(encoding="utf-8"))["history"][2]["examples"], 8)

    def test_malformed_manifest_stops(self):
        self.assertEqual(self.run_tool("--freeze")[0], 0)
        record = json.loads(self.manifest.read_text(encoding="utf-8"))
        entry = {"digest": record["digest"], "rows": 4, "examples": 12, "date": "2026-10-17", "reason": "corrected"}
        for broken in ({**record, "digest": "abc"}, {key: value for key, value in record.items() if key != "history"},
                       {**record, "rows": "4"}, {**record, "history": [{"digest": record["digest"]}]}, [],
                       {**record, "history": [{key: entry[key] for key in ("digest", "date", "reason")}]},
                       {**record, "history": [{**entry, "rows": "4"}]},
                       {**record, "history": [{**entry, "examples": -1}]},
                       {**record, "history": [{**entry, "examples": True}]},
                       {**record, "history": [{**entry, "reason": None}]}):
            with self.subTest(broken=broken):
                self.manifest.write_text(json.dumps(broken), encoding="utf-8")
                code, out, err = self.run_tool()
                self.assertEqual(code, 2)
                self.assertIn(str(self.manifest), err)
                self.assertFalse(self.reports.exists())

    def test_invalid_inventory_stops_before_report(self):
        self.replace_row(2, sympy_probe=None)
        self.replace_row(3, plan_group="t1")
        problems = model.validate(model.load_inventory(self.inventory), complete=True)
        self.assertGreaterEqual(len(problems), 2)
        for arguments in ((), ("--freeze",), ("--write-status",)):
            with self.subTest(arguments):
                before = self.inventory.read_bytes()
                code, out, err = self.run_tool(*arguments)
                self.assertEqual(code, 2)
                for problem in problems:
                    self.assertIn(problem, err)
                self.assertFalse(self.reports.exists())
                self.assertFalse(self.manifest.exists())
                self.assertEqual(self.inventory.read_bytes(), before)

    def test_example_counts_are_checked(self):
        base = load_examples(self.examples)
        cases = {"no examples": ({key: value for key, value in base.items() if key != "91.03"},
                                 "91.03: 0 reference examples; every row needs 1 to 20"),
                 "an empty set": ({**base, "91.03": []}, "91.03: 0 reference examples; every row needs 1 to 20"),
                 "21 examples": ({**base, "91.03": make_examples("91.03", 21)},
                                 "91.03: 21 reference examples; every row needs 1 to 20"),
                 "a stranger": ({**base, "93.01": make_examples("93.01", 2)},
                                "93.01: the examples file lists a row that is not in the inventory")}
        for name, (examples, message) in cases.items():
            with self.subTest(name):
                write_examples(self.examples, examples)
                code, out, err = self.run_tool()
                self.assertEqual(code, 2)
                self.assertIn(message, err)
                self.assertFalse(self.reports.exists())
        write_examples(self.examples, {**base, "91.03": make_examples("91.03", 20), "92.01": make_examples("92.01", 1)})
        code, out, err = self.run_tool()
        self.assertEqual(code, 0, err)  # 1 and 20 are both allowed.
        self.examples.unlink()
        code, out, err = self.run_tool()
        self.assertEqual(code, 2)
        self.assertIn(str(self.examples), err)

    def test_write_status_updates_only_status(self):
        self.assertEqual(self.run_tool("--freeze")[0], 0)
        frozen = json.loads(self.manifest.read_text(encoding="utf-8"))["digest"]
        before = model.load_inventory(self.inventory)
        code, out, err = self.run_tool("--write-status")
        self.assertEqual(code, 0, err)
        after = model.load_inventory(self.inventory)
        self.assertEqual([row.status for row in after.rows], ["yes", "partial", "partial", "no"])
        self.assertEqual([dataclasses.replace(row, status="no") for row in after.rows], list(before.rows))
        self.assertEqual(after.meta, before.meta)
        self.assertEqual(self.digest(), frozen)
        self.assertEqual(self.run_tool("--gate")[0], 1)  # The guard still passes; only the gate fails (25%).
        with mock.patch.object(check_coverage, "write_inventory", wraps=model.write_inventory) as writer:
            self.assertEqual(self.run_tool("--write-status")[0], 0)
        writer.assert_not_called()  # Unchanged statuses leave the file alone.


if __name__ == "__main__":
    unittest.main()
