"""The coverage-gate inventory model: reading and writing, the validation rules, the group counts and the frozen digest.

Every test builds a synthetic inventory (in a temporary directory when a file is needed). None reads the local docs/
inventory, which neither CI nor the distribution harness has.
"""

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))  # The coverage gate is tooling beside the scripts, not a package.

from coverage_gate import inventory as model  # noqa: E402

# The canonical file: meta keys in their loaded order with removed last, then the rows with their keys in the Row
# field order; one space per indent level, UTF-8 kept as it is, "\n" line ends and a closing newline.
CANONICAL = """{
 "inventory": "Synthetic inventory for the model tests",
 "created": "2026-10-18",
 "source": "tests/v08/test_coverage_inventory.py",
 "frozen": false,
 "notes": [
  "Fourier–Motzkin has no SymPy 1.14 implementation: ß, ü, 例"
 ],
 "owner": "curator",
 "removed": [
  {
   "id": "11.06",
   "subtopic": "Multivariate linear inequalities",
   "reason": "SymPy 1.14 cannot reduce them"
  }
 ],
 "rows": [
  {
   "id": "01.01",
   "area": "Core & numbers",
   "subtopic": "Exact integers & rationals",
   "sympy_apis": "Integer, Rational, S.Half",
   "sympy_api_paths": [
    "sympy.Integer",
    "sympy.Rational",
    "S.Half"
   ],
   "v07_status": "yes",
   "tier": null,
   "note": "Rational model; integer-bit budgets",
   "plan_group": "v07",
   "wave": null,
   "family": null,
   "missing": "",
   "sympy_probe": "Integer(2) + 2",
   "mathforge_operations": [
    "exact_add"
   ],
   "status": "yes"
  },
  {
   "id": "01.02",
   "area": "Core & numbers",
   "subtopic": "Named constants (π, e)",
   "sympy_apis": "pi, E, EulerGamma, Catalan",
   "sympy_api_paths": [
    "sympy.pi",
    "sympy.E",
    "sympy.EulerGamma",
    "sympy.Catalan"
   ],
   "v07_status": "partial",
   "tier": null,
   "note": "only pi and E, in the function layer",
   "plan_group": "partial",
   "wave": "B",
   "family": "A1",
   "missing": "EulerGamma and Catalan; evalf of every constant",
   "sympy_probe": "pi.evalf(5)",
   "mathforge_operations": [
    "pi_value"
   ],
   "status": "partial"
  },
  {
   "id": "02.01",
   "area": "Number theory",
   "subtopic": "Prime factorization",
   "sympy_apis": "factorint (multiplicities)",
   "sympy_api_paths": [
    "sympy.factorint"
   ],
   "v07_status": "no",
   "tier": "T1",
   "note": "T1",
   "plan_group": "t1",
   "wave": "A",
   "family": "F-INT",
   "missing": "",
   "sympy_probe": "factorint(360)",
   "mathforge_operations": [],
   "status": "no"
  }
 ]
}
"""

# An inventory as the first curation pass wrote it (PowerShell on Windows: BOM and CRLF): none of the later fields,
# the empty legacy examples list, no meta.removed, and a top-level key this model does not know.
LEGACY = {
    "inventory": "SymPy 1.14.0 mathematical subtopics vs MathForge",
    "created": "2026-10-17",
    "source": "docs/v08/design/01-sympy-inventory.md",
    "frozen": False,
    "notes": ["Row validation happens in Phase 0 before freezing."],
    "reviewer": "umut",
    "rows": [
        {"id": "01.01", "area": "Core & numbers", "subtopic": "Exact integers & rationals (bignum)",
         "sympy_apis": "Integer, Rational, S.Half", "v07_status": "yes", "tier": None,
         "note": "Rational model; integer-bit budgets", "family": None, "plan_group": None, "sympy_probe": None,
         "reference_examples": [], "mathforge_operations": [], "status": "yes"},
        {"id": "01.02", "area": "Core & numbers", "subtopic": "Arbitrary-precision floats, evalf/N",
         "sympy_apis": "Float, evalf, N, nfloat", "v07_status": "no", "tier": "T1",
         "note": "T1; excluded by design", "family": None, "plan_group": None, "sympy_probe": None,
         "reference_examples": [], "mathforge_operations": [], "status": "no"},
    ],
}

BASE_ROW = model.Row(
    id="01.01", area="Core & numbers", subtopic="Exact integers", sympy_apis="Integer, Rational",
    sympy_api_paths=("sympy.Integer", "sympy.Rational"), v07_status="yes", tier=None, note="v0.7 has it",
    plan_group="v07", wave=None, family=None, missing="", sympy_probe="Integer(2) + 2",
    mathforge_operations=("exact_add",), status="yes")

# For every plan group, the fields of a row that satisfies every rule; the id and area come from the group's position.
GROUP_FIELDS = {
    "v07": dict(v07_status="yes", tier=None, plan_group="v07", wave=None, family=None, missing=""),
    "partial": dict(v07_status="partial", tier=None, plan_group="partial", wave="A", family="A1",
                    missing="Only the real case"),
    "t1": dict(v07_status="no", tier="T1", plan_group="t1", wave="B", family="B13", missing=""),
    "t2_selected": dict(v07_status="no", tier="T2", plan_group="t2_selected", wave="C", family="C2",
                        missing="Laurent series"),
    "t2_flexible": dict(v07_status="no", tier="T2", plan_group="t2_flexible", wave=None, family=None, missing=""),
    "buffer": dict(v07_status="no", tier="T3", plan_group="buffer", wave="E", family="E4", missing=""),
    "t3_unplanned": dict(v07_status="no", tier="T3", plan_group="t3_unplanned", wave=None, family=None, missing=""),
    "out_of_scope": dict(v07_status="no", tier="T2", plan_group="out_of_scope", wave=None, family=None, missing=""),
}


def make_row(**changes):
    return dataclasses.replace(BASE_ROW, **changes)


def group_row(group, **changes):
    index = model.PLAN_GROUPS.index(group) + 1
    return make_row(**{"id": f"{index:02d}.01", "area": f"Area {index}", **GROUP_FIELDS[group], **changes})


def inventory_of(*rows, **meta):
    return model.Inventory(meta={"inventory": "synthetic", "removed": [], **meta}, rows=tuple(rows))


def problems(*rows, complete=False):
    return model.validate(inventory_of(*rows), complete=complete)


@dataclasses.dataclass(frozen=True)
class SampleExample:
    """The fields of coverage_gate.examples.Example, which is a separate module and not needed here."""
    id: str
    row: str
    source: str
    setup: str
    code: str
    output: str
    srepr: str
    kind: str


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)


class FileTests(TemporaryDirectoryTestCase):
    def test_round_trip_keeps_bytes(self):
        source = self.directory / "inventory.json"
        source.write_bytes(CANONICAL.encode("utf-8"))
        loaded = model.load_inventory(source)
        self.assertEqual([row.id for row in loaded.rows], ["01.01", "01.02", "02.01"])
        self.assertEqual(loaded.rows[0].sympy_api_paths, ("sympy.Integer", "sympy.Rational", "S.Half"))
        self.assertEqual(loaded.rows[2].mathforge_operations, ())
        self.assertEqual(loaded.meta["removed"][0]["id"], "11.06")
        target = self.directory / "written.json"
        model.write_inventory(target, loaded)
        self.assertEqual(target.read_bytes(), CANONICAL.encode("utf-8"))
        self.assertEqual(model.load_inventory(target), loaded)
        # Writing over an existing inventory replaces it and leaves no temporary file behind.
        model.write_inventory(source, loaded)
        self.assertEqual(source.read_bytes(), CANONICAL.encode("utf-8"))
        self.assertEqual(sorted(os.listdir(self.directory)), ["inventory.json", "written.json"])

    def test_legacy_inventory_is_read_with_empty_new_fields_and_written_canonically(self):
        text = json.dumps(LEGACY, ensure_ascii=False, indent=1).replace("\n", "\r\n") + "\r\n"
        legacy = self.directory / "legacy.json"
        legacy.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
        loaded = model.load_inventory(legacy)
        for row in loaded.rows:
            self.assertEqual((row.sympy_api_paths, row.wave, row.missing), ((), None, ""))
            self.assertEqual((row.family, row.plan_group, row.sympy_probe), (None, None, None))
        # Unknown top-level keys survive in place; the absent removed list reads as empty and sits last.
        self.assertEqual(list(loaded.meta),
                         ["inventory", "created", "source", "frozen", "notes", "reviewer", "removed"])
        self.assertEqual(loaded.meta["removed"], [])
        self.assertEqual(loaded.rows[1].tier, "T1")
        written = self.directory / "written.json"
        model.write_inventory(written, loaded)
        data = written.read_bytes()
        self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", data)
        self.assertTrue(data.endswith(b"}\n"))
        document = json.loads(data.decode("utf-8"))
        self.assertEqual(list(document), ["inventory", "created", "source", "frozen", "notes", "reviewer", "removed",
                                          "rows"])
        row_keys = [field.name for field in dataclasses.fields(model.Row)]  # Fixed order, no reference_examples.
        for row in document["rows"]:
            self.assertEqual(list(row), row_keys)
        # The next write is byte-identical: the canonical form is a fixed point.
        again = self.directory / "again.json"
        model.write_inventory(again, model.load_inventory(written))
        self.assertEqual(again.read_bytes(), data)

    def test_a_row_may_leave_out_what_curation_fills_in_later(self):
        minimal = {"id": "01.01", "area": "Core & numbers", "subtopic": "Exact integers", "sympy_apis": "Integer",
                   "v07_status": "yes", "tier": None, "note": "", "status": "yes"}
        path = self.directory / "minimal.json"
        path.write_bytes(json.dumps({"rows": [minimal]}).encode("utf-8"))
        row = model.load_inventory(path).rows[0]
        self.assertEqual((row.sympy_api_paths, row.mathforge_operations, row.missing), ((), (), ""))
        self.assertEqual((row.wave, row.family, row.plan_group, row.sympy_probe), (None, None, None, None))
        self.assertEqual((row.id, row.sympy_apis, row.status), ("01.01", "Integer", "yes"))

    def test_write_gives_a_hand_built_inventory_its_removed_list(self):
        meta = {"inventory": "hand built"}
        path = self.directory / "inventory.json"
        model.write_inventory(path, model.Inventory(meta=meta, rows=(BASE_ROW,)))
        document = json.loads(path.read_bytes().decode("utf-8"))
        self.assertEqual(list(document), ["inventory", "removed", "rows"])
        self.assertEqual(document["removed"], [])
        self.assertEqual(meta, {"inventory": "hand built"})  # The caller's mapping is not touched.
        self.assertEqual(model.load_inventory(path).rows, (BASE_ROW,))

    def test_malformed_inventories_are_refused(self):
        good = json.loads(CANONICAL)

        def with_row(**changes):
            document = copy.deepcopy(good)
            document["rows"][0].update(changes)
            return document

        def without_row_field(name):
            document = copy.deepcopy(good)
            del document["rows"][0][name]
            return document

        def row_name(field):
            return "row #0" if field == "id" else "row 01.01"  # A row without a usable id is named by its position.

        cases = {
            "not an object": ([], "an object with a 'rows' list"),
            "no rows": ({"inventory": "x"}, "an object with a 'rows' list"),
            "rows is not a list": ({"rows": {}}, "an object with a 'rows' list"),
            "a row is not an object": ({"rows": ["01.01"]}, r"row #0: expected an object"),
            "removed is not a list": ({**good, "removed": {}}, "'removed' must be a list"),
            "unknown row field": (with_row(famliy="A1"), "row 01.01: unknown field famliy"),
            "path that is not text": (with_row(sympy_api_paths=["sympy.Integer", 3]),
                                      "row 01.01: sympy_api_paths must be a list of strings"),
            "examples in the legacy field": (with_row(reference_examples=[{"id": "01.01/01"}]),
                                             "row 01.01: reference_examples is not empty"),
            "text that is not JSON": (b'{"rows": [', "Expecting value"),
            "bytes that are not UTF-8": ('{"rows": []} ı'.encode("cp1254"), "codec can't decode"),
        }
        for name in ("id", "area", "subtopic", "sympy_apis", "v07_status", "tier", "note", "status"):
            cases[f"no {name}"] = (without_row_field(name), f"{row_name(name)}: missing field {name}")
        for name in (field.name for field in dataclasses.fields(model.Row)):
            cases[f"a number as {name}"] = (with_row(**{name: 12345}), f"{row_name(name)}: {name} must be")
        path = self.directory / "bad.json"
        for name, (document, pattern) in cases.items():
            with self.subTest(name):
                path.write_bytes(document if isinstance(document, bytes)
                                 else json.dumps(document, ensure_ascii=False).encode("utf-8"))
                with self.assertRaisesRegex(ValueError, pattern) as refusal:
                    model.load_inventory(path)
                self.assertIn("bad.json: ", str(refusal.exception))  # The message names the file.

    def test_failed_write_leaves_the_inventory_as_it_was(self):
        path = self.directory / "inventory.json"
        path.write_bytes(CANONICAL.encode("utf-8"))
        loaded = model.load_inventory(path)
        shorter = dataclasses.replace(loaded, rows=loaded.rows[:1])
        with mock.patch.object(os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                model.write_inventory(path, shorter)
        self.assertEqual(path.read_bytes(), CANONICAL.encode("utf-8"))
        self.assertEqual(os.listdir(self.directory), ["inventory.json"])  # The temporary file is gone too.
        unserializable = dataclasses.replace(loaded, meta={**loaded.meta, "when": object()})
        with self.assertRaises(TypeError):
            model.write_inventory(path, unserializable)
        self.assertEqual(path.read_bytes(), CANONICAL.encode("utf-8"))
        self.assertEqual(os.listdir(self.directory), ["inventory.json"])


class ValidationTests(unittest.TestCase):
    def test_shape_rules(self):
        self.assertEqual(problems(*(group_row(group) for group in model.PLAN_GROUPS)), [])
        self.assertEqual(problems(make_row(), make_row(subtopic="Another subtopic"), make_row(subtopic="A third")),
                         ["01.01: duplicate id (3 rows)"])
        # fullmatch and ASCII digits: a trailing newline and Arabic-Indic digits are not an id.
        for bad in ("1.01", "01.1", "01.001", "ab.cd", "01-01", "01.01 ", " 01.01", "01.01\n", "٠١.٠١", ""):
            with self.subTest(id=bad):
                self.assertEqual(problems(make_row(id=bad)), [f"{bad}: id must look like NN.NN"])
        self.assertEqual(
            problems(make_row(id="01.01", area="Core & numbers"), make_row(id="01.02", area="Core and numbers")),
            ["01.02: area 'Core and numbers' differs from 'Core & numbers' of 01.01, which has the same id prefix"])
        self.assertEqual(
            problems(make_row(id="01.01", area="Core"), make_row(id="02.01", area="Core")),
            ["02.01: area 'Core' is also the area of 01.01, which has another id prefix"])
        self.assertEqual(problems(make_row(v07_status="maybe")),
                         ["01.01: v07_status must be yes, partial or no, not 'maybe'"])
        self.assertEqual(problems(make_row(v07_status="no", tier=None)),
                         ["01.01: v07_status 'no' needs a tier of T1, T2 or T3, not None"])
        self.assertEqual(problems(make_row(v07_status="no", tier="T4")),
                         ["01.01: v07_status 'no' needs a tier of T1, T2 or T3, not 'T4'"])
        self.assertEqual(problems(make_row(v07_status="yes", tier="T1")),
                         ["01.01: tier 'T1' is only for v07_status 'no', not 'yes'"])
        self.assertEqual(problems(make_row(v07_status="partial", tier="T2")),
                         ["01.01: tier 'T2' is only for v07_status 'no', not 'partial'"])

    def test_shape_rules_are_all_that_an_incomplete_inventory_is_held_to(self):
        unfinished = make_row(plan_group="bogus", wave="Z", family="??", missing="", sympy_probe=None,
                              sympy_api_paths=())
        self.assertEqual(problems(unfinished), [])
        self.assertEqual(problems(unfinished, complete=True), [
            "01.01: plan_group 'bogus' does not fit v07_status 'yes'; expected v07",
            "01.01: a row with wave 'Z' needs a family like A3, B13 or F-INT, not '??'",
            "01.01: sympy_api_paths is empty",
            "01.01: sympy_probe is empty"])

    def test_messages_start_with_the_row_id_and_follow_the_row_order(self):
        """A curator greps for the id; the shape rules of a row are reported before its complete rules."""
        rows = (make_row(id="01.01", v07_status="no", tier=None, plan_group=None, sympy_probe=None),
                make_row(id="3.1", area="Another area"),
                make_row(id="05.05", area="A third area", v07_status="maybe", wave="A", family="x"))
        self.assertEqual(problems(*rows, complete=True), [
            "01.01: v07_status 'no' needs a tier of T1, T2 or T3, not None",
            "01.01: sympy_probe is empty",
            "3.1: id must look like NN.NN",
            "05.05: v07_status must be yes, partial or no, not 'maybe'",
            "05.05: wave 'A' does not fit plan_group 'v07'; expected no wave",
            "05.05: a row with wave 'A' needs a family like A3, B13 or F-INT, not 'x'"])

    def test_complete_rules(self):
        everything_well = [group_row(group) for group in model.PLAN_GROUPS] + [
            group_row("out_of_scope", tier="T3", id="08.02")]
        self.assertEqual(problems(*everything_well, complete=True), [])
        for row in everything_well:
            with self.subTest(group=row.plan_group, tier=row.tier):
                self.assertEqual(problems(row, complete=True), [])
        for good in ("A3", "B13", "C1", "E8", "F-INT", "F-REG", "F-DP"):
            with self.subTest(family=good):
                self.assertEqual(problems(group_row("t1", family=good), complete=True), [])

        bad_family = "{id}: a row with wave 'B' needs a family like A3, B13 or F-INT, not %r"
        cases = [
            ("t2_selected without missing", "t2_selected", dict(missing=""),
             ["{id}: plan_group 't2_selected' needs a non-empty missing"]),
            ("partial with a blank missing", "partial", dict(missing="  "),
             ["{id}: plan_group 'partial' needs a non-empty missing"]),
            ("t1 without a wave", "t1", dict(wave=None, family=None),
             ["{id}: wave None does not fit plan_group 't1'; expected A, B, C or D"]),
            ("t1 in the buffer wave", "t1", dict(wave="E"),
             ["{id}: wave 'E' does not fit plan_group 't1'; expected A, B, C or D"]),
            ("buffer in wave A", "buffer", dict(wave="A"),
             ["{id}: wave 'A' does not fit plan_group 'buffer'; expected E"]),
            ("buffer without a wave", "buffer", dict(wave=None, family=None),
             ["{id}: wave None does not fit plan_group 'buffer'; expected E"]),
            ("v07 with a wave", "v07", dict(wave="A", family="A1"),
             ["{id}: wave 'A' does not fit plan_group 'v07'; expected no wave"]),
            ("a wave without a family", "t1", dict(family=None), [bad_family % None]),
            ("a wave with a lower-case family", "t1", dict(family="b13"), [bad_family % "b13"]),
            ("a wave with a family outside A-E", "t1", dict(family="G1"), [bad_family % "G1"]),
            ("a wave with a family of three digits", "t1", dict(family="A123"), [bad_family % "A123"]),
            ("a wave with a digit in an F family", "t1", dict(family="F-A1"), [bad_family % "F-A1"]),
            ("a wave with a trailing space", "t1", dict(family="A3 "), [bad_family % "A3 "]),
            ("no plan group", "t1", dict(plan_group=None, wave=None, family=None),
             ["{id}: plan_group None does not fit v07_status 'no', tier 'T1'; expected t1"]),
            ("an unknown plan group", "t1", dict(plan_group="t4", wave=None, family=None),
             ["{id}: plan_group 't4' does not fit v07_status 'no', tier 'T1'; expected t1"]),
            ("a yes row in t1", "v07", dict(plan_group="t1", wave="A", family="A1"),
             ["{id}: plan_group 't1' does not fit v07_status 'yes'; expected v07"]),
            ("a partial row in v07", "partial", dict(plan_group="v07", wave=None, family=None, missing=""),
             ["{id}: plan_group 'v07' does not fit v07_status 'partial'; expected partial"]),
            ("a T1 row out of scope", "t1", dict(plan_group="out_of_scope", wave=None, family=None),
             ["{id}: plan_group 'out_of_scope' does not fit v07_status 'no', tier 'T1'; expected t1"]),
            ("a T2 row in the buffer", "t2_flexible", dict(plan_group="buffer", wave="E", family="E1"),
             ["{id}: plan_group 'buffer' does not fit v07_status 'no', tier 'T2'; "
              "expected t2_selected, t2_flexible or out_of_scope"]),
            ("a T3 row in t2_flexible", "t3_unplanned", dict(plan_group="t2_flexible"),
             ["{id}: plan_group 't2_flexible' does not fit v07_status 'no', tier 'T3'; "
              "expected buffer, t3_unplanned or out_of_scope"]),
            ("no api paths", "v07", dict(sympy_api_paths=()), ["{id}: sympy_api_paths is empty"]),
            ("no probe", "v07", dict(sympy_probe=None), ["{id}: sympy_probe is empty"]),
            ("a blank probe", "v07", dict(sympy_probe=" "), ["{id}: sympy_probe is empty"]),
            ("several at once, in rule order", "t2_selected",
             dict(wave="Z", family=None, missing="", sympy_api_paths=(), sympy_probe=""),
             ["{id}: wave 'Z' does not fit plan_group 't2_selected'; expected A, B, C or D",
              "{id}: a row with wave 'Z' needs a family like A3, B13 or F-INT, not None",
              "{id}: plan_group 't2_selected' needs a non-empty missing",
              "{id}: sympy_api_paths is empty",
              "{id}: sympy_probe is empty"]),
        ]
        for name, group, changes, expected in cases:
            row = group_row(group, **changes)
            with self.subTest(name):
                self.assertEqual(problems(row, complete=True), [message.format(id=row.id) for message in expected])
                self.assertEqual(problems(row), [])  # None of these is a shape problem.


class DigestTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rows = (make_row(id="01.01", area="Core & numbers", subtopic="Exact integers"),
                make_row(id="02.01", area="Number theory", subtopic="Fourier–Motzkin", plan_group="t1",
                         v07_status="no", tier="T1", wave="A", family="F-INT", sympy_probe="factorint(360)"))
        examples = {
            "01.01": [SampleExample("01.01/01", "01.01", "docstring:sympy.Integer", "", "Integer(2) + 2", "4",
                                    "Integer(4)", "exact"),
                      SampleExample("01.01/02", "01.01", "docstring:sympy.Rational", "x = Rational(1, 3)",
                                    "x + x", "2/3", "Rational(2, 3)", "exact")],
            "02.01": [SampleExample("02.01/01", "02.01", "tests:test_ntheory::test_factorint", "", "factorint(12)",
                                    "{2: 2, 3: 1}", "Dict(...)", "exact")],
        }
        return inventory_of(*rows, frozen=False), examples

    def test_digest_ignores_mutable_fields(self):
        inventory, examples = self.fixture()
        digest = model.frozen_digest(inventory, examples)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

        def digest_with(**changes):
            rows = (dataclasses.replace(inventory.rows[0], **changes),) + inventory.rows[1:]
            return model.frozen_digest(dataclasses.replace(inventory, rows=rows), examples)

        for name, value in (("status", "partial"), ("mathforge_operations", ("exact_add", "exact_mul")),
                            ("note", "a different note")):
            with self.subTest(mutable=name):
                self.assertEqual(digest_with(**{name: value}), digest)
        # Neither does anything in meta: the freeze date is written there after the digest is taken.
        for name, value in (("frozen", "2026-10-30"), ("notes", ["a note"]), ("removed", [{"id": "11.06"}])):
            with self.subTest(meta=name):
                meta = {**inventory.meta, name: value}
                self.assertEqual(model.frozen_digest(dataclasses.replace(inventory, meta=meta), examples), digest)
        self.assertNotEqual(digest_with(sympy_probe="Integer(3) + 2"), digest)
        self.assertNotEqual(digest_with(subtopic="Exact integers and rationals"), digest)
        changed = copy.deepcopy(examples)
        changed["01.01"][1] = dataclasses.replace(changed["01.01"][1], output="2/3 ")
        self.assertNotEqual(model.frozen_digest(inventory, changed), digest)

    def test_digest_covers_every_frozen_field_and_every_example_field(self):
        inventory, examples = self.fixture()
        digest = model.frozen_digest(inventory, examples)

        def altered(value):
            if value is None:
                return "x"
            return (value + ("sympy.extra",)) if isinstance(value, tuple) else (value + "x")

        row = inventory.rows[0]
        for name in model.FROZEN_FIELDS:
            with self.subTest(frozen=name):
                rows = (dataclasses.replace(row, **{name: altered(getattr(row, name))}),) + inventory.rows[1:]
                self.assertNotEqual(model.frozen_digest(dataclasses.replace(inventory, rows=rows), examples), digest)
        for field in dataclasses.fields(SampleExample):
            with self.subTest(example_field=field.name):
                changed = copy.deepcopy(examples)
                example = changed["02.01"][0]
                changed["02.01"][0] = dataclasses.replace(example, **{field.name: getattr(example, field.name) + "x"})
                self.assertNotEqual(model.frozen_digest(inventory, changed), digest)
        for name, changed in (("an example added", {**examples, "02.01": examples["02.01"] + examples["01.01"][:1]}),
                              ("an example removed", {**examples, "01.01": examples["01.01"][:1]}),
                              ("a row's examples dropped", {"01.01": examples["01.01"]})):
            with self.subTest(name):
                self.assertNotEqual(model.frozen_digest(inventory, changed), digest)
        with self.subTest("a row removed"):
            fewer = dataclasses.replace(inventory, rows=inventory.rows[:1])
            self.assertNotEqual(model.frozen_digest(fewer, examples), digest)

    def test_every_row_field_is_frozen_or_deliberately_mutable(self):
        """A field added to Row has to be classified: frozen into the digest, or listed here as free to change."""
        fields = {field.name for field in dataclasses.fields(model.Row)}
        self.assertLessEqual(set(model.FROZEN_FIELDS), fields)
        self.assertEqual(fields - set(model.FROZEN_FIELDS), {"note", "mathforge_operations", "status"})

    def test_digest_recipe_is_pinned(self):
        """The freeze manifest will hold a digest, so the recipe may never drift: this is its definition. The rows and
        the examples go in out of order; the canonical text has both in id order, every key sorted, no spaces and the
        non-ASCII character as it is."""
        inventory, examples = self.fixture()
        out_of_order = dataclasses.replace(inventory, rows=inventory.rows[::-1])
        canonical = (
            '{"examples":{"01.01":['
            '{"code":"Integer(2) + 2","id":"01.01/01","kind":"exact","output":"4","row":"01.01","setup":"",'
            '"source":"docstring:sympy.Integer","srepr":"Integer(4)"},'
            '{"code":"x + x","id":"01.01/02","kind":"exact","output":"2/3","row":"01.01","setup":"x = Rational(1, 3)",'
            '"source":"docstring:sympy.Rational","srepr":"Rational(2, 3)"}],'
            '"02.01":[{"code":"factorint(12)","id":"02.01/01","kind":"exact","output":"{2: 2, 3: 1}","row":"02.01",'
            '"setup":"","source":"tests:test_ntheory::test_factorint","srepr":"Dict(...)"}]},'
            '"rows":['
            '{"area":"Core & numbers","family":null,"id":"01.01","missing":"","plan_group":"v07",'
            '"subtopic":"Exact integers","sympy_api_paths":["sympy.Integer","sympy.Rational"],'
            '"sympy_apis":"Integer, Rational","sympy_probe":"Integer(2) + 2","tier":null,"v07_status":"yes",'
            '"wave":null},'
            '{"area":"Number theory","family":"F-INT","id":"02.01","missing":"","plan_group":"t1",'
            '"subtopic":"Fourier–Motzkin","sympy_api_paths":["sympy.Integer","sympy.Rational"],'
            '"sympy_apis":"Integer, Rational","sympy_probe":"factorint(360)","tier":"T1","v07_status":"no",'
            '"wave":"A"}]}')
        scrambled = {"02.01": examples["02.01"], "01.01": examples["01.01"][::-1]}
        self.assertEqual(model.frozen_digest(out_of_order, scrambled),
                         hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    def test_digest_does_not_depend_on_order_or_on_how_examples_are_represented(self):
        inventory, examples = self.fixture()
        digest = model.frozen_digest(inventory, examples)
        reordered = dataclasses.replace(inventory, rows=inventory.rows[::-1])
        backwards = {row: list(reversed(group)) for row, group in reversed(list(examples.items()))}
        self.assertEqual(model.frozen_digest(reordered, backwards), digest)
        as_mappings = {row: [dataclasses.asdict(example) for example in group] for row, group in examples.items()}
        self.assertEqual(model.frozen_digest(inventory, as_mappings), digest)
        # A row recorded with no examples is the same fact as a row that is not mentioned.
        self.assertEqual(model.frozen_digest(inventory, {**examples, "03.01": []}), digest)
        with self.assertRaises(TypeError):
            model.frozen_digest(inventory, {"01.01": ["not an example"]})


class GroupCountTests(unittest.TestCase):
    def test_group_counts(self):
        rows = (group_row("v07"), group_row("t1"), group_row("t1", id="03.02"), group_row("buffer"),
                group_row("t3_unplanned", plan_group=None), group_row("t2_flexible", plan_group="t2flexible"))
        counts = model.group_counts(inventory_of(*rows))
        self.assertEqual(counts, {"v07": 1, "partial": 0, "t1": 2, "t2_selected": 0, "t2_flexible": 0, "buffer": 1,
                                  "t3_unplanned": 0, "out_of_scope": 0, "unassigned": 1, "t2flexible": 1})
        # Fixed order, every group present, and a mistyped group stays visible: the counts add up to the rows.
        self.assertEqual(list(counts), [*model.PLAN_GROUPS, "unassigned", "t2flexible"])
        self.assertEqual(sum(counts.values()), len(rows))
        self.assertEqual(model.group_counts(inventory_of()), dict.fromkeys([*model.PLAN_GROUPS, "unassigned"], 0))


if __name__ == "__main__":
    unittest.main()
