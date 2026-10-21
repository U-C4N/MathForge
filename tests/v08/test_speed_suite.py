"""The speed suite (benchmarks/speed_suite.py), its gate (tools/check_speed.py) and the exemption allowlist of the
coverage gate's speed file (tools/coverage_gate/criteria.py).

SpeedSuiteTest holds the five checks that define the task; the other classes test the parts. The suite's end-to-end
tests start its worker processes on tests/v08/speed_demo_fixtures.py, and a worker asserts that mathforge comes from
the repository's src, so those tests skip where there is none (the distribution harness installs the package and has
no src). The rest runs in this process: the measuring functions on synthetic fixtures, check_speed on crafted details
files, the criteria loader on crafted speed files. Nothing needs SymPy; it is used when installed.
"""

import argparse
import contextlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY, REPOSITORY / "tools"):  # benchmarks.* from the root; the tools and the gate beside them.
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import check_speed  # noqa: E402
from benchmarks import speed_suite  # noqa: E402
from benchmarks.speed.fixture import EdgeCase, ExemptFixture, Fixture  # noqa: E402
from coverage_gate import criteria  # noqa: E402

import mathforge as mf  # noqa: E402

SUITE = REPOSITORY / "benchmarks" / "speed_suite.py"
TOOL = REPOSITORY / "tools" / "check_speed.py"
CAPABILITIES = REPOSITORY / "tests" / "fixtures" / "v8" / "capabilities_v07.json"
PACKAGE = "benchmarks.speed.fixtures"
DEMO_MODULE = "tests.v08.speed_demo_fixtures"
DEMO = ["--modules", DEMO_MODULE, "--rounds", "1", "--repeat", "2", "--warmups", "1", "--sample-ms", "0.2",
        "--allow-quick"]  # One round of two samples is below the protocol of a gate run: the flag says so.
HAS_SRC = (REPOSITORY / "src" / "mathforge" / "__init__.py").is_file()
NEEDS_SRC = unittest.skipUnless(HAS_SRC, "needs the repository's src, which the distribution harness does not have")
HASH = "a" * 64  # A source hash for the crafted details files; check_speed is told it is the working tree's.
INSTRUMENT = "d" * 64  # The same for the instrument hash (the suite and the fixtures).
ROUNDS = 3  # The rounds of crafted gate-protocol details.


def capability_names():
    document = json.loads(CAPABILITIES.read_text(encoding="utf-8"))
    return [entry["name"] for entry in document["operations"]]


def run_suite(arguments, environment=None):
    return subprocess.run([sys.executable, str(SUITE), *map(str, arguments)], cwd=REPOSITORY, env=environment,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)


def instant(i):
    return lambda: None


def gcd_fixture(label="x", size="micro", variants=20, agree=lambda mathforge_value, sympy_value: True, sympy=instant,
                operation="integer_gcd"):
    return Fixture(operation, size, label, lambda i: (mf.integer_gcd, (84 + 6 * i, 30 + 6 * i), {}), sympy, agree,
                   "synthetic", variants=variants)


def problems_in(module, operations, ids, edges, *, real=True):
    """Everything that is wrong with a fixture module, for the real modules and for synthetic ones alike."""
    name, problems = module.__name__, []
    fixtures, edge_cases = getattr(module, "FIXTURES", None), getattr(module, "EDGE_CASES", None)
    for what, value in (("FIXTURES", fixtures), ("EDGE_CASES", edge_cases)):
        if not isinstance(value, (tuple, list)):
            problems.append(f"{name} defines no {what} tuple")
    for entry in fixtures if isinstance(fixtures, (tuple, list)) else ():
        if not isinstance(entry, (Fixture, ExemptFixture)):
            problems.append(f"{name}.FIXTURES holds a {type(entry).__name__}, not a Fixture or an ExemptFixture")
            continue
        identifier = f"{entry.operation}/{entry.size}/{entry.label}"
        if identifier in ids:
            problems.append(f"{identifier}: defined in {ids[identifier]} and in {name}")
        ids[identifier] = name
        if entry.operation not in operations:
            problems.append(f"{identifier}: {entry.operation} is not an operation of capabilities_v07.json")
        if entry.size not in speed_suite.SIZES:
            problems.append(f"{identifier}: the size is {entry.size!r}, not micro or realistic")
        if not (isinstance(entry.label, str) and entry.label):
            problems.append(f"{identifier}: the label must be a non-empty string")
        if isinstance(entry, Fixture):
            if entry.operation == "verify":
                problems.append(f"{identifier}: verify has no SymPy counterpart; it takes ExemptFixtures")
            if real and entry.size == "micro" and entry.variants < 1000:
                problems.append(f"{identifier}: a micro fixture needs at least 1,000 variants, not {entry.variants}")
            if not (isinstance(entry.reference, str) and entry.reference.strip()):
                problems.append(f"{identifier}: reference must name the SymPy call chosen")
            fields = ("mathforge", "sympy", "agree")
        else:
            if entry.operation not in criteria.SPEED_EXEMPTIONS:
                problems.append(f"{identifier}: {entry.operation} is not in criteria.SPEED_EXEMPTIONS")
            if not (isinstance(entry.reason, str) and entry.reason.strip()):
                problems.append(f"{identifier}: an ExemptFixture must say why")
            fields = ("timed", "budget")
        problems.extend(f"{identifier}: {field} is not callable" for field in fields
                        if not callable(getattr(entry, field)))
    for case in edge_cases if isinstance(edge_cases, (tuple, list)) else ():
        if not isinstance(case, EdgeCase):
            problems.append(f"{name}.EDGE_CASES holds a {type(case).__name__}, not an EdgeCase")
            continue
        identifier = f"{case.operation}/edge/{case.label}"
        if identifier in edges:
            problems.append(f"{identifier}: defined in {edges[identifier]} and in {name}")
        edges[identifier] = name
        if case.operation not in operations:
            problems.append(f"{identifier}: {case.operation} is not an operation of capabilities_v07.json")
        if not callable(case.mathforge):
            problems.append(f"{identifier}: mathforge is not callable")
    return problems


def fixture_entry(operation, size, ratio=1.5):
    return {"operation": operation, "size": size, "label": "x", "ratio": ratio, "round_ratios": [ratio] * ROUNDS,
            "mathforge": {"median_us": ratio * 2.0}, "sympy_reset": {"median_us": 2.0}}


def exempt_entry(operation="verify", size="micro", met=True):
    return {"operation": operation, "size": size, "label": "x", "reason": "no SymPy counterpart", "budget_met": met,
            "timed": {"median_us": 1.0}, "budget": {"median_us": 2.0}}


def gate_meta():
    """The meta of details that meet the measurement protocol: 3 rounds of 9 repeats, a pinned cpu, SymPy 1.14.0."""
    return {"source_sha256": HASH, "instrument_sha256": INSTRUMENT, "partial": False, "sympy": check_speed.SYMPY_VERSION,
            "rounds": ROUNDS, "repeat": 9, "cpu": 0, "sample_ms": 2.0, "areas": None, "ops": None}


def passing_details():
    """Details that pass: both sizes for every one of the 102 operations, verify through ExemptFixtures."""
    fixtures = {f"{name}/{size}/x": fixture_entry(name, size) for name in capability_names() if name != "verify"
                for size in speed_suite.SIZES}
    exempt = {f"verify/{size}/x": exempt_entry("verify", size) for size in speed_suite.SIZES}
    return {"meta": gate_meta(), "fixtures": fixtures, "exempt": exempt}


def set_ratios(details, identifier, ratio, rounds=None):
    """The ratio of a fixture, and its rounds (all equal to it unless given)."""
    entry = details["fixtures"][identifier]
    entry["ratio"], entry["round_ratios"] = ratio, list(rounds) if rounds is not None else [ratio] * ROUNDS


def set_rounds(details, rounds):
    """The number of rounds of crafted details: the meta, and one round ratio per round for every fixture."""
    details["meta"]["rounds"] = rounds
    for entry in details["fixtures"].values():
        entry["round_ratios"] = (entry["round_ratios"] * rounds)[:rounds]


def with_ratio(identifier, ratio):
    details = passing_details()
    set_ratios(details, identifier, ratio)
    return details


def real_details():
    """Details that pass for this checkout: an entry for every fixture and exemption that its modules define, with its
    real source and instrument hashes."""
    fixture_ids, exempt_ids = speed_suite.current_ids()
    details = {"meta": gate_meta(),
               "fixtures": {identifier: fixture_entry(*identifier.split("/", 2)[:2]) for identifier in fixture_ids},
               "exempt": {identifier: exempt_entry(*identifier.split("/", 2)[:2]) for identifier in exempt_ids}}
    details["meta"].update(source_sha256=speed_suite.source_hash(REPOSITORY),
                           instrument_sha256=speed_suite.instrument_hash(REPOSITORY))
    return details


def run_check(details, *extra, working_hash=HASH, instrument_hash=INSTRUMENT, current_ids=None):
    """check_speed.main on a details document: (exit status, stdout, stderr). The current ids are the document's own
    unless given, so that a crafted document is by default the one that the fixture modules would produce."""
    if current_ids is None:
        current_ids = (set(details["fixtures"]), set(details["exempt"]))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "details.json"
        path.write_text(json.dumps(details), encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = check_speed.main(["--details", str(path), *extra], working_hash=working_hash,
                                      instrument_hash=instrument_hash, current_ids=current_ids)
    return status, stdout.getvalue(), stderr.getvalue()


class SpeedSuiteTest(unittest.TestCase):
    """The five checks that define the task."""

    def test_fixture_ids_unique_and_known(self):
        # Every area module that exists: the ids are unique across them all, every operation is one of the 102, every
        # size is micro or realistic. (Without area modules there is nothing to check.)
        operations, ids, edges, problems = set(capability_names()), {}, {}, []
        for area in speed_suite.discover(PACKAGE, None):
            problems += problems_in(importlib.import_module(f"{PACKAGE}.{area}"), operations, ids, edges)
        self.assertEqual(problems, [])

    @NEEDS_SRC
    def test_smoke_run_writes_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latest.json"
            completed = run_suite(["--output", output, *DEMO])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("worst ratio", completed.stdout)
            self.assertIn("not a gate run (--allow-quick)", completed.stdout)  # One round of two samples.
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                             ["latest-details.json", "latest.json"])  # The default details path, and no leftovers.
            details = json.loads((Path(directory) / "latest-details.json").read_text(encoding="utf-8"))
            speed = criteria._speed_entries(output)  # The coverage gate loads the speed file as it is.

        meta = details["meta"]
        self.assertEqual(meta["source_sha256"], speed_suite.source_hash(REPOSITORY))
        self.assertEqual(meta["instrument_sha256"], speed_suite.instrument_hash(REPOSITORY))
        self.assertEqual((meta["rounds"], meta["repeat"], meta["warmups"], meta["sample_ms"]), (1, 2, 1, 0.2))
        self.assertEqual((meta["modules"], meta["areas"], meta["ops"], meta["partial"], meta["cpu"]),
                         ([DEMO_MODULE], None, None, False, None))
        self.assertEqual(meta["sympy_ground_types"], "python" if meta["sympy"] else None)
        self.assertEqual(meta["mathforge_version"], mf.__version__)
        for key in ("python", "platform", "date", "sympy", "mpmath", "mpmath_backend"):  # The versions are recorded.
            self.assertIn(key, meta)
        self.assertEqual(sorted(details["fixtures"]), ["factorial/realistic/demo", "integer_gcd/micro/demo"])
        for identifier, entry in details["fixtures"].items():
            with self.subTest(identifier):
                self.assertTrue(1 <= entry["k"] <= entry["variants"])
                self.assertEqual(entry["k_by_round"], [entry["k"]])
                self.assertEqual(entry["operation"], identifier.split("/")[0])
                for mode in ("mathforge", "sympy_reset", "sympy_warm"):
                    statistics = entry[mode]
                    self.assertEqual(len(statistics["samples_us"]), 2)  # rounds x repeat
                    self.assertEqual(len(statistics["round_medians_us"]), 1)
                    self.assertGreater(statistics["min_us"], 0)
                    self.assertLessEqual(statistics["min_us"], statistics["median_us"])
                    self.assertLessEqual(statistics["median_us"], statistics["p95_us"])
                self.assertAlmostEqual(entry["ratio"], entry["mathforge"]["median_us"]
                                       / entry["sympy_reset"]["median_us"], delta=1e-4 * entry["ratio"])
                self.assertEqual(entry["round_ratios"], [entry["ratio"]])
                self.assertGreater(entry["first_call"]["median_us"], 0)
                self.assertGreater(entry["structured"]["median_us"], 0)
                self.assertIsInstance(entry["structured"]["overhead_us"], float)
                self.assertEqual(len(entry["structured"]["samples_us"]), 2)
                self.assertTrue(entry["reference"].startswith("demo"))
        self.assertEqual(list(details["exempt"]), ["verify/micro/demo"])
        exempt = details["exempt"]["verify/micro/demo"]
        self.assertIs(exempt["budget_met"], True)
        self.assertLess(exempt["timed"]["median_us"], exempt["budget"]["median_us"])
        self.assertEqual(exempt["reason"], "demo: no SymPy counterpart")

        self.assertEqual(set(speed), {"integer_gcd", "factorial", "verify"})
        self.assertEqual(speed["integer_gcd"], details["fixtures"]["integer_gcd/micro/demo"]["ratio"])
        self.assertEqual(speed["factorial"], details["fixtures"]["factorial/realistic/demo"]["ratio"])
        self.assertEqual(speed["verify"], criteria.SpeedExemption("demo: no SymPy counterpart", True))

    @NEEDS_SRC
    def test_partial_run_writes_no_criteria_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output, details_path = Path(directory) / "smoke.json", Path(directory) / "elsewhere" / "partial.json"
            completed = run_suite(["--output", output, "--details", details_path, "--ops", "integer_gcd", *DEMO])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("partial run", completed.stdout)
            self.assertFalse(output.exists())
            self.assertFalse((Path(directory) / "smoke-details.json").exists())
            details = json.loads(details_path.read_text(encoding="utf-8"))
        self.assertEqual((details["meta"]["partial"], details["meta"]["ops"]), (True, ["integer_gcd"]))
        self.assertEqual(list(details["fixtures"]), ["integer_gcd/micro/demo"])
        self.assertEqual(details["exempt"], {})

    def test_check_speed_failures(self):
        def without(*identifiers):
            def change(details):
                for identifier in identifiers:
                    del details["fixtures"][identifier]
            return change

        def set_ratio(identifier, ratio):
            def change(details):
                set_ratios(details, identifier, ratio)
            return change

        def add_exempt(details):
            details["exempt"]["det/micro/x"] = exempt_entry("det", "micro")

        def unmet(details):
            details["exempt"]["verify/realistic/x"]["budget_met"] = False

        def stale(details):
            details["meta"]["source_sha256"] = "b" * 64

        status, stdout, stderr = run_check(passing_details())  # They pass, so each failure below is one change.
        self.assertEqual((status, stderr), (0, ""), stdout)
        cases = {
            "an operation without fixtures": (without("det/micro/x", "det/realistic/x"), ["det"]),
            "an operation without a micro fixture": (without("rank/micro/x"), ["no micro fixture", "rank"]),
            "an operation without a realistic fixture": (without("rref/realistic/x"),
                                                         ["no realistic fixture", "rref"]),
            "verify without its exempt fixtures": (lambda details: details["exempt"].clear(), ["verify"]),
            "a stale source hash": (stale, ["stale", "bbbbbbbbbbbb", "aaaaaaaaaaaa"]),
            "a ratio of 3.2": (set_ratio("integer_gcd/micro/x", 3.2), ["ratio above", "integer_gcd/micro/x", "3.20"]),
            "an exemption outside the allowlist": (add_exempt, ["det/micro/x", "SPEED_EXEMPTIONS"]),
            "an unmet budget": (unmet, ["budget not met", "verify/realistic/x"]),
        }
        for name, (change, expected) in cases.items():
            with self.subTest(name):
                details = passing_details()
                change(details)
                status, stdout, stderr = run_check(details)
                self.assertEqual(status, 1, stdout + stderr)
                for text in expected:
                    self.assertIn(text, stderr)

    def test_exemption_allowlist(self):
        self.assertEqual(dict(criteria.SPEED_EXEMPTIONS), {
            "verify": "re-verification of a record; no SymPy counterpart",
            "is_prime": "proven primality above 3.3e24; no SymPy counterpart (spec H2)"})
        exemption = {"exempt": "no SymPy counterpart", "budget_met": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speed.json"

            def load(operations):
                path.write_text(json.dumps({"operations": operations}), encoding="utf-8")
                return criteria._speed_entries(path)

            for name in criteria.SPEED_EXEMPTIONS:
                with self.subTest(allowed=name):
                    self.assertEqual(load({name: exemption}),
                                     {name: criteria.SpeedExemption("no SymPy counterpart", True)})
            for name in ("det", "integer_gcd", "op_alpha"):
                with self.subTest(refused=name):
                    with self.assertRaises(ValueError) as raised:
                        load({name: exemption})
                    self.assertIn(name, str(raised.exception))
                    self.assertIn(str(path), str(raised.exception))
                    self.assertIn("SPEED_EXEMPTIONS", str(raised.exception))
                    self.assertEqual(load({name: 2.5}), {name: 2.5})  # A ratio is open to every operation.
            with self.assertRaises(ValueError) as raised:  # One refused operation refuses the whole file.
                load({"verify": exemption, "det": exemption, "rank": 1.5})
            self.assertIn("det", str(raised.exception))


class FixtureModuleTest(unittest.TestCase):
    def test_the_checks_catch_what_they_are_for(self):
        module = types.ModuleType("synthetic")
        module.FIXTURES = (
            gcd_fixture("a"), gcd_fixture("a"),                        # Twice the same id.
            gcd_fixture("b", operation="no_such_operation"),           # Not an operation of the 102.
            gcd_fixture("c", size="huge"),                             # Not a size.
            gcd_fixture("d", operation="verify"),                      # verify takes ExemptFixtures.
            gcd_fixture("e", variants=10),                             # A micro fixture with too few variants.
            ExemptFixture("det", "micro", "f", lambda: None, lambda: None, "a reason"),  # Outside the allowlist.
            ExemptFixture("verify", "micro", "g", lambda: None, lambda: None, " "),      # No reason.
            "not a fixture",
        )
        module.EDGE_CASES = (EdgeCase("integer_gcd", "h", lambda: None), EdgeCase("integer_gcd", "h", lambda: None))
        problems = "\n".join(problems_in(module, set(capability_names()), {}, {}))
        for expected in ("integer_gcd/micro/a: defined in", "no_such_operation/micro/b: no_such_operation is not an op",
                         "integer_gcd/huge/c: the size is 'huge'", "verify has no SymPy counterpart",
                         "integer_gcd/micro/e: a micro fixture needs at least 1,000 variants",
                         "det/micro/f: det is not in criteria.SPEED_EXEMPTIONS", "verify/micro/g: an ExemptFixture",
                         "holds a str", "integer_gcd/edge/h: defined in"):
            self.assertIn(expected, problems)
        self.assertEqual(problems_in(types.ModuleType("empty"), set(), {}, {}),
                         ["empty defines no FIXTURES tuple", "empty defines no EDGE_CASES tuple"])
        demo = importlib.import_module(DEMO_MODULE)
        self.assertEqual(problems_in(demo, set(capability_names()), {}, {}, real=False), [])

    def test_no_fixture_module_keeps_a_rule_that_prefers_a_call_to_a_faster_one(self):
        """A reference is the fastest same-object call. The tie rules of the first fixtures (a documented function wins
        within 30% or 15%, the shortest expression wins even when a faster spelling exists) made it a call that favours
        MathForge, and they are gone from every module, its comments and its reference texts."""
        banned = ("within the margin", "wins any tie", "wins a tie within", "is not preferred",
                  "but does not replace it", "shortest expression even", "the shortest expression")
        sources = sorted((REPOSITORY / "benchmarks" / "speed" / "fixtures").glob("*.py"))
        self.assertGreaterEqual(len(sources), 10)
        found = []
        for path in sources:
            text = path.read_text(encoding="utf-8")
            found += [f"{path.name}: {phrase!r}" for phrase in banned if phrase in text]
            found += [f"{path.name}: {match!r}" for match in re.findall(r"within \d+%", text)]  # "within 15%" and so on
        self.assertEqual(found, [])

    def test_modules_are_discovered_by_name_and_underscored_ones_are_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "speed_demo_discovery"
            package.mkdir()
            for name in ("__init__", "alpha", "beta", "_helpers"):
                (package / f"{name}.py").write_text("FIXTURES = ()\nEDGE_CASES = ()\n")
            sys.path.insert(0, directory)
            self.addCleanup(sys.path.remove, directory)
            self.addCleanup(importlib.invalidate_caches)
            importlib.invalidate_caches()
            self.assertEqual(speed_suite.discover("speed_demo_discovery", None), ["alpha", "beta"])
            self.assertEqual(speed_suite.discover("speed_demo_discovery", ["beta"]), ["beta"])
            with self.assertRaises(speed_suite.SuiteError) as raised:
                speed_suite.discover("speed_demo_discovery", ["alpha", "gamma"])
            self.assertIn("no fixture module gamma", str(raised.exception))
            self.assertIn("alpha, beta", str(raised.exception))
            with self.assertRaises(speed_suite.SuiteError):
                speed_suite.discover("speed_demo_no_such_package", None)

    def test_loading_refuses_what_it_cannot_measure(self):
        def load(*fixtures, ops=None):
            module = types.ModuleType("speed_demo_loading")
            module.FIXTURES = fixtures
            with mock.patch.dict(sys.modules, {"speed_demo_loading": module}):
                return speed_suite.load_entries(["speed_demo_loading"], ops)

        entries = load(gcd_fixture("a"), gcd_fixture("a", size="realistic"),
                       ExemptFixture("verify", "micro", "a", lambda: None, lambda: None, "why"))
        self.assertEqual([speed_suite.entry_id(entry) for entry in entries],
                         ["integer_gcd/micro/a", "integer_gcd/realistic/a", "verify/micro/a"])
        self.assertEqual([speed_suite.entry_id(entry) for entry in load(*entries, ops=["verify"])], ["verify/micro/a"])
        for what, fixtures in (("twice", [gcd_fixture("a"), gcd_fixture("a")]),
                               ("a size", [gcd_fixture("a", size="huge")]),
                               ("variants", [gcd_fixture("a", variants=0)]),
                               ("a reason", [ExemptFixture("verify", "micro", "a", None, None, " ")]),
                               ("a thing", ["not a fixture"])):
            with self.subTest(what), self.assertRaises(speed_suite.SuiteError):
                load(*fixtures)
        with self.assertRaises(speed_suite.SuiteError) as raised:
            load(gcd_fixture("a"), ops=["det"])
        self.assertIn("no fixture for operation det", str(raised.exception))
        module = types.ModuleType("speed_demo_empty")  # No FIXTURES at all.
        with mock.patch.dict(sys.modules, {"speed_demo_empty": module}), self.assertRaises(speed_suite.SuiteError):
            speed_suite.load_entries(["speed_demo_empty"], None)


class SourceHashTest(unittest.TestCase):
    def test_hash_covers_the_sources_in_path_order_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src" / "mathforge"
            (package / "sub").mkdir(parents=True)
            (package / "b.py").write_bytes(b"b = 1\n")
            (package / "sub" / "a.py").write_bytes(b"a = 1\n")
            first = speed_suite.source_hash(root)
            self.assertEqual(len(first), 64)
            self.assertEqual(first, speed_suite.source_hash(root))
            (package / "notes.txt").write_text("not a source")
            (package / "__pycache__").mkdir()
            (package / "__pycache__" / "b.cpython-314.pyc").write_bytes(b"\0")
            self.assertEqual(speed_suite.source_hash(root), first)
            (package / "sub" / "a.py").write_bytes(b"a = 2\n")
            self.assertNotEqual(speed_suite.source_hash(root), first)
            (package / "sub" / "a.py").write_bytes(b"a = 1\n")
            (package / "c.py").write_bytes(b"")
            self.assertNotEqual(speed_suite.source_hash(root), first)
            with self.assertRaises(speed_suite.SuiteError):
                speed_suite.source_hash(root / "elsewhere")

    @NEEDS_SRC
    def test_the_gate_uses_the_suites_hash(self):
        self.assertEqual(speed_suite.source_hash(REPOSITORY), speed_suite.source_hash())

    def test_the_instrument_hash_covers_the_suite_and_every_file_of_the_speed_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmarks" / "speed" / "fixtures").mkdir(parents=True)
            suite, fixture = root / "benchmarks" / "speed_suite.py", root / "benchmarks" / "speed" / "fixtures" / "a.py"
            data = root / "benchmarks" / "speed" / "data.json"
            suite.write_bytes(b"suite = 1\n")
            fixture.write_bytes(b"FIXTURES = ()\n")
            first = speed_suite.instrument_hash(root)
            self.assertEqual(len(first), 64)
            self.assertEqual(first, speed_suite.instrument_hash(root))
            (root / "benchmarks" / "speed" / "fixtures" / "__pycache__").mkdir()  # Bytecode is not the instrument.
            (root / "benchmarks" / "speed" / "fixtures" / "__pycache__" / "a.cpython-314.pyc").write_bytes(b"\0")
            (root / "benchmarks" / "other.py").write_bytes(b"not part of it")
            self.assertEqual(speed_suite.instrument_hash(root), first)
            for what, change in (("the suite", lambda: suite.write_bytes(b"suite = 2\n")),
                                 ("a fixture module", lambda: fixture.write_bytes(b"FIXTURES = (1,)\n")),
                                 ("a data file", lambda: data.write_bytes(b"{}")),
                                 ("a new module", lambda: (root / "benchmarks" / "speed" / "b.py").write_bytes(b""))):
                before = speed_suite.instrument_hash(root)
                change()
                self.assertNotEqual(speed_suite.instrument_hash(root), before, what)
            with self.assertRaises(speed_suite.SuiteError):
                speed_suite.instrument_hash(root / "elsewhere")

    @NEEDS_SRC
    def test_the_real_instrument_hash_is_stable_and_not_the_source_hash(self):
        self.assertEqual(speed_suite.instrument_hash(REPOSITORY), speed_suite.instrument_hash())
        self.assertNotEqual(speed_suite.instrument_hash(REPOSITORY), speed_suite.source_hash(REPOSITORY))

    def test_a_change_during_a_round_is_refused_whichever_hash_it_moves(self):
        with mock.patch.object(speed_suite, "source_hash", return_value="2" * 64), \
                mock.patch.object(speed_suite, "instrument_hash", return_value="4" * 64):
            speed_suite.require_unchanged("2" * 64, "4" * 64, "round 1")  # Both as they were: nothing to say.
            with self.assertRaises(speed_suite.SuiteError) as raised:
                speed_suite.require_unchanged("1" * 64, "4" * 64, "round 2 of 3")
            self.assertIn("round 2 of 3", str(raised.exception))
            self.assertIn("src/mathforge 111111111111 became 222222222222", str(raised.exception))
            self.assertNotIn("benchmarks", str(raised.exception))
            with self.assertRaises(speed_suite.SuiteError) as raised:
                speed_suite.require_unchanged("2" * 64, "3" * 64, "round 1")
            self.assertIn("benchmarks/speed_suite.py and benchmarks/speed 333333333333 became 444444444444",
                          str(raised.exception))
            with self.assertRaises(speed_suite.SuiteError) as raised:
                speed_suite.require_unchanged("1" * 64, "3" * 64, "round 1")
            self.assertIn("src/mathforge", str(raised.exception))
            self.assertIn("benchmarks/speed", str(raised.exception))


class FilesTest(unittest.TestCase):
    def test_arrays_of_numbers_are_one_line_and_nothing_else_changes(self):
        document = {"samples_us": [1, 2.5, -3e-07, 1.5e+20], "names": ["a", "b"], "nested": [[1, 2], [3]],
                    "empty": [], "mixed": [1, "x"], "objects": [{"a": [4, 5]}, {"b": 6}], "flag": [True, False],
                    "inner": {"values": [0.1, 0.2, 0.3]}}
        text = speed_suite.json_text(document)
        self.assertEqual(json.loads(text), document)
        self.assertTrue(text.endswith("}\n"))
        self.assertIn('"samples_us": [1, 2.5, -3e-07, 1.5e+20]', text)
        self.assertIn('"values": [0.1, 0.2, 0.3]', text)
        self.assertIn('"empty": []', text)
        self.assertIn('"names": [\n', text)  # Strings are left as indented.
        self.assertLess(len(text.splitlines()), len(json.dumps(document, indent=1).splitlines()))

    def test_a_file_is_written_whole_with_lf_line_ends_and_unwritable_paths_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new" / "latest.json"
            speed_suite.require_writable(path)  # Directories that do not exist yet are fine.
            speed_suite.write_json(path, {"operations": {"det": 1.5}})
            self.assertEqual(path.read_bytes(), b'{\n "operations": {\n  "det": 1.5\n }\n}\n')
            self.assertEqual([entry.name for entry in path.parent.iterdir()], ["latest.json"])  # No temporary file.
            speed_suite.write_json(path, {"operations": {}})  # Replaced in one step.
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"operations": {}})
            speed_suite.require_writable(path)
            with self.assertRaises(speed_suite.SuiteError):
                speed_suite.require_writable(path.parent)  # A directory is not a file to write.
            with self.assertRaises(speed_suite.SuiteError):
                speed_suite.require_writable(path / "inside.json")  # Below a file.


class MeasurementTest(unittest.TestCase):
    """The measuring functions on synthetic fixtures, in this process."""

    @classmethod
    def setUpClass(cls):
        cls.reset = speed_suite.Reset()
        cls.config = speed_suite.Config(repeat=3, warmups=1, sample_ms=0.2)

    def rng(self):
        return random.Random(7)

    def test_the_timed_loop_runs_every_call_in_order_with_the_collector_off_and_restores_it(self):
        import gc
        order, states = [], []
        calls = [lambda n=n: (order.append(n), states.append(gc.isenabled())) for n in range(5)]
        for before in (True, False):
            with self.subTest(collector_enabled_before=before):
                (gc.enable if before else gc.disable)()
                self.addCleanup(gc.enable)
                order.clear(), states.clear()
                self.assertGreater(speed_suite.run_calls(calls), 0)
                self.assertEqual((order, set(states), gc.isenabled()), ([0, 1, 2, 3, 4], {False}, before))

                def fails():
                    raise ZeroDivisionError

                with self.assertRaises(ZeroDivisionError):
                    speed_suite.run_calls([fails])
                self.assertEqual(gc.isenabled(), before)  # A call that raises does not leave it off.

    def test_lists_are_split_deduplicated_and_never_empty(self):
        self.assertIsNone(speed_suite.parse_list(None))
        self.assertEqual(speed_suite.parse_list("det, rank,det"), ["det", "rank"])
        for text in ("", "a,,b", " ,a"):
            with self.subTest(text), self.assertRaises(speed_suite.SuiteError):
                speed_suite.parse_list(text)

    def test_rounds_that_measured_different_fixtures_are_refused(self):
        def record(*fixtures):
            return {"fixtures": {name: {} for name in fixtures}, "exempt": {}}

        with self.assertRaises(speed_suite.SuiteError) as raised:
            speed_suite.build_details([record("a/micro/x"), record("a/micro/x", "b/micro/x")], {})
        self.assertIn("did not measure the same fixtures", str(raised.exception))

    def test_the_table_has_a_line_per_fixture_and_exemption_and_names_the_worst(self):
        def mode(median):
            return {"median_us": median}

        def entry(ratio, structured):
            return {"k": 10, "mathforge": mode(ratio), "sympy_reset": mode(1.0), "sympy_warm": mode(0.5),
                    "ratio": ratio, "first_call": mode(100.0), "structured": structured}

        details = {"fixtures": {"a/micro/x": entry(1.5, {"overhead_us": 12.5}),
                                "b/micro/x": entry(4.25, {"same_function": True, "overhead_us": 0.0}),
                                "c/micro/x": entry(2.0, {"error": "boom", "overhead_us": None})},
                   "exempt": {"verify/micro/x": {"timed": mode(5.0), "budget": mode(9.0), "budget_met": True}}}
        lines = speed_suite.render_table(details).splitlines()
        self.assertEqual(len(lines), 1 + 3 + 1 + 1)  # Header, fixtures, the exemption, the worst.
        self.assertIn("exempt: timed 5 us, budget 9 us, budget met", lines[4])
        self.assertEqual(lines[-1], "worst ratio 4.25: b/micro/x")
        self.assertIn("12.5", lines[1])
        self.assertTrue(lines[3].split()[-2] == "-")  # The structured error shows no overhead.

    def test_a_round_of_one_fixture(self):
        fixture = gcd_fixture(variants=40)
        record = speed_suite.measure_fixture(fixture, self.config, self.rng(), self.reset)
        self.assertEqual((record["operation"], record["size"], record["label"], record["variants"]),
                         ("integer_gcd", "micro", "x", 40))
        self.assertTrue(1 <= record["k"] <= 40)
        self.assertGreater(record["first_call_us"], 0)
        self.assertEqual(set(record["samples_us"]), {"mathforge", "sympy_reset", "sympy_warm"})
        for mode, samples in record["samples_us"].items():
            with self.subTest(mode):
                self.assertEqual(len(samples), 3)
                self.assertTrue(all(isinstance(value, float) and value >= 0 for value in samples))
        self.assertGreater(min(record["samples_us"]["mathforge"]), 0)  # A verified gcd takes real time.
        structured = record["structured"]
        self.assertEqual(structured["status"], "ok")  # mathforge.operations.integer_gcd on variant 0.
        self.assertEqual((len(structured["structured_us"]), len(structured["value_us"])), (3, 3))

    def test_k_is_enough_for_the_sample_and_never_more_than_the_variants(self):
        def calls_last_100_us_each(calls):
            return len(calls) * 100_000  # Nanoseconds: no timer, no noise.

        config = speed_suite.Config(repeat=1, warmups=1, sample_ms=3.05)  # A sample must last 3.05 ms.
        with mock.patch.object(speed_suite, "run_calls", calls_last_100_us_each):
            # One call is 0.1 ms: the first estimate is 34 calls (30.5, and a tenth more), whose 3.4 ms are enough.
            self.assertEqual(speed_suite.calibrate(gcd_fixture(variants=500), config, self.reset), 34)
            self.assertEqual(speed_suite.calibrate(gcd_fixture(variants=10), config, self.reset), 10)  # Not above that.
            self.assertEqual(speed_suite.calibrate(gcd_fixture(variants=1), config, self.reset), 1)
            slow = speed_suite.Config(repeat=1, warmups=1, sample_ms=0.05)  # One call is already enough.
            self.assertEqual(speed_suite.calibrate(gcd_fixture(variants=500), slow, self.reset), 1)

    def test_a_real_calibration_is_one_call_for_a_slow_reference_and_every_variant_for_a_quick_one(self):
        def slow(i):
            return lambda: time.sleep(0.003)  # Longer than the 0.2 ms sample on its own.

        self.assertEqual(speed_suite.calibrate(gcd_fixture(variants=50, sympy=slow), self.config, self.reset), 1)
        self.assertEqual(speed_suite.calibrate(gcd_fixture(variants=7), self.config, self.reset), 7)

    def test_agree_false_aborts_and_names_the_fixture(self):
        fixture = gcd_fixture("disagreeing", agree=lambda mathforge_value, sympy_value: False)
        with self.assertRaises(speed_suite.SuiteError) as raised:
            speed_suite.measure_fixture(fixture, self.config, self.rng(), self.reset)
        message = str(raised.exception)
        self.assertIn("integer_gcd/micro/disagreeing", message)
        self.assertIn("disagree on variant 0", message)
        self.assertIn("MathForge gave 6", message)

    def test_agree_is_checked_on_the_first_five_variants_in_every_warmup(self):
        seen = []

        def agree(mathforge_value, sympy_value):
            seen.append(mathforge_value)
            return True

        speed_suite.measure_fixture(gcd_fixture(variants=20, agree=agree), speed_suite.Config(1, 1, 0.2), self.rng(),
                                    self.reset)
        self.assertEqual(seen, [math.gcd(84 + 6 * i, 30 + 6 * i) for i in range(5)])
        seen.clear()
        speed_suite.measure_fixture(gcd_fixture(variants=3, agree=agree), speed_suite.Config(1, 2, 0.2), self.rng(),
                                    self.reset)
        self.assertEqual(seen, [math.gcd(84 + 6 * i, 30 + 6 * i) for i in range(3)] * 2)

    def test_an_agree_or_a_call_that_raises_names_the_fixture(self):
        def broken(mathforge_value, sympy_value):
            raise ValueError("cannot compare")
        with self.assertRaises(speed_suite.SuiteError) as raised:
            speed_suite.measure_fixture(gcd_fixture("a", agree=broken), self.config, self.rng(), self.reset)
        self.assertIn("integer_gcd/micro/a: agree on variant 0 raised ValueError: cannot compare",
                      str(raised.exception))

        def sympy_fails(i):
            def call():
                raise ZeroDivisionError("boom")
            return call
        with self.assertRaises(speed_suite.SuiteError) as raised:
            speed_suite.measure_fixture(gcd_fixture("b", sympy=sympy_fails), self.config, self.rng(), self.reset)
        self.assertIn("integer_gcd/micro/b: the SymPy call of variant 0 raised ZeroDivisionError",
                      str(raised.exception))

    def test_structured_is_reported_not_required(self):
        # The value function is the structured function itself: nothing to compare, and no overhead.
        same = Fixture("integer_gcd", "micro", "same", lambda i: (mf.operations.integer_gcd, (84, 30), {}), instant,
                       lambda a, b: True, "synthetic", variants=5)
        record = speed_suite.measure_fixture(same, self.config, self.rng(), self.reset)
        self.assertEqual(record["structured"], {"status": "same"})
        # An operation that mathforge.operations has no function for: the error is reported, the run goes on.
        unknown = gcd_fixture("unknown", operation="no_such_operation")
        record = speed_suite.measure_fixture(unknown, self.config, self.rng(), self.reset)
        self.assertEqual(record["structured"]["status"], "error")
        self.assertIn("no_such_operation", record["structured"]["message"])
        self.assertEqual(len(record["samples_us"]["mathforge"]), 3)

    def test_a_structured_call_that_raises_is_reported_and_the_rest_is_still_timed(self):
        # The same operation with arguments that only the structured function refuses.
        refused = Fixture("integer_gcd", "micro", "refused", lambda i: (lambda a, b, extra=None: math.gcd(a, b),
                                                                         (84, 30), {"extra": 1}), instant,
                          lambda a, b: True, "synthetic", variants=5)
        record = speed_suite.measure_fixture(refused, self.config, self.rng(), self.reset)
        self.assertEqual(record["structured"]["status"], "error")
        self.assertIn("raised TypeError", record["structured"]["message"])
        self.assertEqual(len(record["samples_us"]["sympy_warm"]), 3)
        summary = speed_suite.summarize_fixture("integer_gcd/micro/refused", [record])
        self.assertEqual(summary["structured"]["overhead_us"], None)
        self.assertIn("raised TypeError", summary["structured"]["error"])

    def test_the_summary_of_rounds_pools_the_samples_and_takes_the_median(self):
        def record(mathforge, sympy, k, first):
            return {"operation": "det", "size": "micro", "label": "x", "variants": 1000, "reference": "ref",
                    "v07": "", "k": k, "first_call_us": first,
                    "samples_us": {"mathforge": mathforge, "sympy_reset": sympy, "sympy_warm": [1.0, 1.0, 1.0]},
                    "structured": {"status": "ok", "structured_us": [12.0, 14.0, 13.0], "value_us": [11.0, 11.0, 12.0]}}

        rounds = [record([9.0, 10.0, 11.0], [4.0, 5.0, 6.0], 10, 100.0),
                  record([29.0, 30.0, 31.0], [9.0, 10.0, 11.0], 12, 300.0)]
        summary = speed_suite.summarize_fixture("det/micro/x", rounds)
        self.assertEqual(summary["mathforge"]["median_us"], 20.0)  # Of all six samples: 9 10 11 29 30 31.
        self.assertEqual(summary["mathforge"]["round_medians_us"], [10.0, 30.0])
        self.assertEqual((summary["mathforge"]["min_us"], summary["mathforge"]["p95_us"]), (9.0, 31.0))
        self.assertEqual(summary["mathforge"]["samples_us"], [9.0, 10.0, 11.0, 29.0, 30.0, 31.0])
        self.assertEqual(summary["sympy_reset"]["median_us"], 7.5)
        self.assertEqual(summary["ratio"], 2.66667)  # 20 / 7.5, the pooled medians: not 2.5, the mean of 2 and 3.
        self.assertEqual(summary["round_ratios"], [2.0, 3.0])
        self.assertEqual((summary["k"], summary["k_by_round"]), (10, [10, 12]))
        self.assertEqual(summary["first_call"], {"median_us": 200.0, "rounds_us": [100.0, 300.0]})
        self.assertEqual(summary["structured"]["median_us"], 13.0)
        self.assertEqual(summary["structured"]["value_median_us"], 11.0)
        # The paired differences are 1, 3, 1 in each round: their median is 1, not 13 - 11, the difference of medians.
        self.assertEqual(summary["structured"]["overhead_us"], 1.0)
        self.assertEqual(summary["structured"]["round_overhead_us"], [1.0, 1.0])
        broken = [record([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1, 1.0)]
        with self.assertRaises(speed_suite.SuiteError):
            speed_suite.summarize_fixture("det/micro/x", broken)  # A ratio of nothing says nothing.

    def test_the_reset_clears_mathforge_caches_and_sympy_when_it_is_loaded(self):
        from mathforge.trusted.caches import BoundedCache
        cache = BoundedCache("speed-suite-test", 4)
        cache.get("key", lambda: "value")
        self.assertEqual(len(cache), 1)
        speed_suite.Reset()()
        self.assertEqual(len(cache), 0)
        if "sympy" in sys.modules:  # The suite's own reset of SymPy runs as it does in a worker.
            import sympy
            sympy.factorint(360)
            speed_suite.Reset()()

    def test_exempt_fixtures_are_held_to_their_budget(self):
        def quick():
            return sum, ((),), {}

        def slow():
            return time.sleep, (0.002,), {}

        def summary(timed, budget):
            fixture = ExemptFixture("verify", "micro", "x", timed, budget, "no SymPy counterpart")
            records = [speed_suite.measure_exempt(fixture, self.config, self.rng(), self.reset) for _ in range(2)]
            for record in records:
                self.assertEqual({role: len(values) for role, values in record["samples_us"].items()},
                                 {"timed": 3, "budget": 3})
            return speed_suite.summarize_exempt(records)

        kept, missed = summary(quick, slow), summary(slow, quick)
        self.assertIs(kept["budget_met"], True)
        self.assertIs(missed["budget_met"], False)
        self.assertEqual((kept["operation"], kept["reason"]), ("verify", "no SymPy counterpart"))
        self.assertLess(kept["ratio"], 1.0)
        self.assertGreater(missed["ratio"], 1.0)

    def test_a_timed_call_that_does_not_verify_stops_the_run(self):
        report = types.SimpleNamespace(verified=False)
        fixture = ExemptFixture("verify", "micro", "x", lambda: (lambda: report, (), {}), lambda: (sum, ((),), {}), "r")
        with self.assertRaises(speed_suite.SuiteError) as raised:
            speed_suite.measure_exempt(fixture, self.config, self.rng(), self.reset)
        self.assertIn("verify/micro/x", str(raised.exception))
        self.assertIn("did not verify", str(raised.exception))

    def test_the_criteria_file_takes_the_worst_ratio_or_the_failed_exemption(self):
        def exempt(operation, met, reason="no counterpart"):
            return {"operation": operation, "reason": reason, "budget_met": met}

        details = {"fixtures": {"det/micro/a": fixture_entry("det", "micro", 1.5),
                                "det/realistic/a": fixture_entry("det", "realistic", 2.5),
                                "is_prime/micro/a": fixture_entry("is_prime", "micro", 1.2),
                                "rank/micro/a": fixture_entry("rank", "micro", 0.5)},
                   "exempt": {"verify/micro/a": exempt("verify", True, "re-verification"),
                              "verify/realistic/a": exempt("verify", True, "re-verification"),
                              "is_prime/realistic/big": exempt("is_prime", False, "proven primality")}}
        self.assertEqual(speed_suite.build_criteria(details), {"operations": {
            "det": 2.5, "rank": 0.5, "is_prime": {"exempt": "proven primality", "budget_met": False},
            "verify": {"exempt": "re-verification", "budget_met": True}}})
        details["exempt"]["is_prime/realistic/big"]["budget_met"] = True  # Its ratios are what counts again.
        self.assertEqual(speed_suite.build_criteria(details)["operations"]["is_prime"], 1.2)
        details["exempt"]["verify/micro/b"] = exempt("verify", False, "another reason")
        self.assertEqual(speed_suite.build_criteria(details)["operations"]["verify"],
                         {"exempt": "re-verification; another reason", "budget_met": False})

    def test_the_protocol_shortfalls_of_a_run(self):
        def run(**options):
            return speed_suite.protocol_shortfalls(argparse.Namespace(**{"rounds": 3, "repeat": 9, "cpu": 0, **options}))

        self.assertEqual((speed_suite.MIN_ROUNDS, speed_suite.MIN_REPEAT), (3, 9))
        self.assertEqual(run(), [])
        self.assertEqual(run(rounds=5, repeat=30, cpu=7), [])
        self.assertEqual(run(rounds=2), ["2 rounds, not 3"])
        self.assertEqual(run(repeat=8), ["8 repeats, not 9"])
        self.assertEqual(run(cpu=None), ["no --cpu, so the workers are not pinned"])
        self.assertEqual(len(run(rounds=1, repeat=1, cpu=None)), 3)

    def test_current_ids_tell_fixtures_from_exempt_fixtures(self):
        fixtures, exempt = speed_suite.current_ids()
        self.assertTrue(fixtures and exempt)
        self.assertEqual(fixtures & exempt, set())
        self.assertIn("integer_gcd/micro/pairs", fixtures)
        self.assertTrue(all(identifier.split("/")[0] in criteria.SPEED_EXEMPTIONS for identifier in exempt))
        discrete, _ = speed_suite.current_ids(["discrete"])
        self.assertTrue(discrete and discrete < fixtures)
        self.assertTrue(all(identifier.split("/")[0] in ("factorial", "binomial", "multinomial", "permutations_count",
                                                         "fibonacci", "catalan", "stirling") for identifier in discrete))
        with self.assertRaises(speed_suite.SuiteError):
            speed_suite.current_ids(["no_such_area"])

    @NEEDS_SRC
    def test_the_worker_hashes_the_source_and_the_instrument_again_at_the_end_of_its_round(self):
        def worker(result, source_hashes, instrument_hash="3" * 64):
            arguments = argparse.Namespace(cpu=None, areas=None, modules=DEMO_MODULE, ops="integer_gcd", repeat=2,
                                           warmups=1, sample_ms=0.2, round_index=0, worker_result=result)
            with mock.patch.object(speed_suite, "source_hash", side_effect=source_hashes), \
                    mock.patch.object(speed_suite, "instrument_hash", return_value=instrument_hash), \
                    mock.patch.object(speed_suite, "reference_versions", return_value={"sympy": None}), \
                    contextlib.redirect_stdout(io.StringIO()):
                return speed_suite.run_worker(arguments)

        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "round.json"
            self.assertEqual(worker(result, ["1" * 64, "1" * 64]), 0)  # Unchanged at the start and at the end.
            record = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual((record["source_sha256"], record["instrument_sha256"]), ("1" * 64, "3" * 64))
            result.unlink()
            with self.assertRaises(speed_suite.SuiteError) as raised:  # Edited while the round was measuring.
                worker(result, ["1" * 64, "2" * 64])
            self.assertIn("round 1", str(raised.exception))
            self.assertIn("src/mathforge 111111111111 became 222222222222", str(raised.exception))
            self.assertFalse(result.exists())  # Nothing of that round is kept.

    def test_options_are_checked(self):
        for arguments in (["--output", "x.json", "--rounds", "0"], ["--output", "x.json", "--repeat", "0"],
                          ["--output", "x.json", "--warmups", "0"], ["--output", "x.json", "--sample-ms", "0"],
                          ["--output", "x.json", "--sample-ms", "nan"], ["--output", "x.json", "--cpu", "-1"],
                          [], ["--output", "x.json", "--modules", "m", "--areas", "a"]):
            with self.subTest(arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    speed_suite.main(arguments)
                self.assertEqual(raised.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()) as stderr:  # Refused in this process, before any worker starts.
            self.assertEqual(speed_suite.main(["--output", "x.json", "--areas", "no_such_area"]), 2)
        self.assertIn("no fixture module no_such_area", stderr.getvalue())
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()) as stderr, \
                contextlib.redirect_stdout(io.StringIO()):  # The not-a-gate-run notice of --allow-quick.
            path = Path(directory) / "latest.json"
            self.assertEqual(speed_suite.main(["--output", str(path), "--details", str(path), *DEMO]), 2)
            self.assertIn("name the same file", stderr.getvalue())
            path.write_text("a file, not a directory")
            self.assertEqual(speed_suite.main(["--output", str(path / "inside.json"), *DEMO]), 2)  # Before any worker.
            self.assertIn(f"cannot write {path / 'inside-details.json'}", stderr.getvalue())  # The details come first.


@NEEDS_SRC
class SuiteRunTest(unittest.TestCase):
    """The suite as a command: more of what worker processes on the demo fixtures do."""

    def test_unknown_areas_and_operations_are_refused_and_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latest.json"
            completed = run_suite(["--output", output, "--areas", "no_such_area"])
            self.assertEqual(completed.returncode, 2)
            self.assertIn("no fixture module no_such_area", completed.stderr)  # Said by the parent, before any worker.
            self.assertNotIn("round 1", completed.stderr)
            completed = run_suite(["--output", output, "--ops", "no_such_op", *DEMO])
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("no fixture for operation no_such_op", completed.stderr)  # The worker says what is wrong.
            self.assertIn("round 1 of 1 failed", completed.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_the_worker_is_pinned_when_asked(self):
        if hasattr(os, "sched_getaffinity"):
            cpu = min(os.sched_getaffinity(0))
        elif sys.platform == "win32":
            cpu = 0
        else:
            self.skipTest(f"no process affinity on {sys.platform}")
        with tempfile.TemporaryDirectory() as directory:
            completed = run_suite(["--output", Path(directory) / "latest.json", "--cpu", cpu, *DEMO])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            details = json.loads((Path(directory) / "latest-details.json").read_text(encoding="utf-8"))
        self.assertEqual(details["meta"]["cpu"], cpu)

    def test_a_cpu_that_cannot_be_pinned_stops_the_run(self):
        if not (hasattr(os, "sched_setaffinity") or sys.platform == "win32"):
            self.skipTest(f"no process affinity on {sys.platform}, so --cpu is refused for every value")
        with tempfile.TemporaryDirectory() as directory:
            completed = run_suite(["--output", Path(directory) / "latest.json", "--cpu", 4096, *DEMO])
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("4096", completed.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_a_run_below_the_protocol_needs_allow_quick_and_says_that_it_is_not_a_gate_run(self):
        small = ["--modules", DEMO_MODULE, "--rounds", "1", "--repeat", "2", "--warmups", "1", "--sample-ms", "0.2"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latest.json"
            completed = run_suite(["--output", output, *small])
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            for text in ("not a gate run", "1 rounds, not 3", "2 repeats, not 9", "no --cpu", "--allow-quick"):
                self.assertIn(text, completed.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])  # Refused before a worker started.
            completed = run_suite(["--output", output, *small, "--allow-quick"])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            lines = completed.stdout.splitlines()
            self.assertTrue(lines[0].startswith("not a gate run (--allow-quick): 1 rounds, not 3"), lines[0])
            self.assertEqual(lines[-1], lines[0])  # Said again after the table, where the numbers are read.

    def test_a_partial_run_that_meets_the_protocol_needs_no_flag(self):
        cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
        if not (hasattr(os, "sched_setaffinity") or sys.platform == "win32"):
            self.skipTest(f"no process affinity on {sys.platform}, so a pinned run is impossible")
        with tempfile.TemporaryDirectory() as directory:
            completed = run_suite(["--output", Path(directory) / "latest.json", "--ops", "integer_gcd", "--cpu", cpu,
                                   "--modules", DEMO_MODULE, "--warmups", "1", "--sample-ms", "0.2"])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertNotIn("not a gate run", completed.stdout)
            details = json.loads((Path(directory) / "latest-details.json").read_text(encoding="utf-8"))
        self.assertEqual((details["meta"]["rounds"], details["meta"]["repeat"], details["meta"]["partial"]),
                         (speed_suite.MIN_ROUNDS, speed_suite.MIN_REPEAT, True))  # The defaults are the protocol.

    def test_a_disagreeing_fixture_aborts_the_run_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "speed_demo_disagreeing"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "fixtures.py").write_text(textwrap.dedent('''
                import mathforge as mf
                from benchmarks.speed.fixture import Fixture

                FIXTURES = (Fixture("integer_gcd", "micro", "wrong", lambda i: (mf.integer_gcd, (84 + i, 30), {}),
                                    lambda i: (lambda: 0), lambda mathforge_value, sympy_value: mathforge_value == 0,
                                    "demo", variants=10),)
                EDGE_CASES = ()
            '''), encoding="utf-8")
            output = Path(directory) / "out" / "latest.json"
            environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
                filter(None, [directory, os.environ.get("PYTHONPATH", "")])))
            completed = run_suite(["--output", output, "--modules", "speed_demo_disagreeing.fixtures", "--rounds", "2",
                                   "--repeat", "1", "--warmups", "1", "--allow-quick"], environment)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("integer_gcd/micro/wrong", completed.stderr)
            self.assertIn("disagree", completed.stderr)
            self.assertIn("round 1 of 2 failed", completed.stderr)
            self.assertFalse(output.parent.exists())

    def test_the_worker_refuses_to_measure_another_source_tree(self):
        # A copy of the suite whose src has no importable package: mathforge then comes from elsewhere (here
        # PYTHONPATH, as an editable or an installed one would), and the worker refuses to measure that.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmarks" / "speed").mkdir(parents=True)
            (root / "benchmarks" / "speed" / "placeholder.py").write_text("")  # Hashed with the suite.
            (root / "src" / "mathforge").mkdir(parents=True)
            (root / "src" / "mathforge" / "placeholder.py").write_text("")  # Hashed, but not a package.
            (root / "benchmarks" / "speed_suite.py").write_bytes(SUITE.read_bytes())
            environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
                [str(REPOSITORY / "src"), str(REPOSITORY)] + ([os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ
                                                              else [])))
            completed = subprocess.run([sys.executable, str(root / "benchmarks" / "speed_suite.py"), "--output",
                                        str(root / "latest.json"), "--modules", DEMO_MODULE, "--rounds", "1",
                                        "--allow-quick"],
                                       cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", timeout=300)
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("mathforge was imported from", completed.stderr)
            self.assertIn("not from", completed.stderr)
            self.assertFalse((root / "latest.json").exists())


class CheckSpeedTest(unittest.TestCase):
    """More of tools/check_speed.py on crafted details files."""

    def test_the_limit_is_inclusive_and_the_target_lists_operations(self):
        status, stdout, stderr = run_check(with_ratio("det/micro/x", 3.0))
        self.assertEqual((status, stderr), (0, ""))
        status, _, stderr = run_check(with_ratio("det/micro/x", 3.0001))
        self.assertEqual(status, 1)
        self.assertIn("det/micro/x", stderr)
        # Above the target 2.0 (and within the limit): listed, not failed.
        status, stdout, stderr = run_check(with_ratio("det/realistic/x", 2.5))
        self.assertEqual((status, stderr), (0, ""))
        self.assertIn("above the target 2.0", stdout)
        self.assertIn("det", stdout.split("above the target 2.0", 1)[1])
        status, stdout, _ = run_check(with_ratio("det/realistic/x", 2.5), "--target", "2.6")
        self.assertEqual(status, 0)
        self.assertIn("above the target 2.6: 0", stdout)
        status, _, stderr = run_check(with_ratio("det/realistic/x", 2.5), "--limit", "2.4", "--target", "2.0")
        self.assertEqual(status, 1)
        self.assertIn("ratio above the limit 2.4", stderr)

    def test_the_worst_ratios_are_printed_worst_first(self):
        details = passing_details()
        for index, name in enumerate(capability_names()[:20]):
            set_ratios(details, f"{name}/micro/x", 2.0 + index / 100)  # 20 above the others' 1.5.
        status, stdout, _ = run_check(details)
        self.assertEqual(status, 0)
        self.assertIn("15 worst ratios", stdout)
        section = stdout.split("15 worst ratios:\n", 1)[1].split("operations above the target", 1)[0]
        ratios = [float(line.split()[0]) for line in section.splitlines()]
        self.assertEqual(ratios, [2.19 - step / 100 for step in range(15)])  # Worst first, and no more than 15.
        self.assertIn("nullspace/micro/x", section.splitlines()[0])

    def test_a_partial_run_is_named_when_the_gate_fails(self):
        details = passing_details()
        details["meta"].update(partial=True, areas=["ntheory"])
        del details["fixtures"]["det/micro/x"]
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("partial run", stderr)
        self.assertIn("--areas ntheory", stderr)

    def test_operations_beyond_the_102_are_measured_and_gated_too(self):
        details = passing_details()  # A later phase adds operations: they have fixtures, and the same limit.
        details["fixtures"]["new_operation/micro/x"] = fixture_entry("new_operation", "micro", 1.2)
        self.assertEqual(run_check(details)[0], 0)
        set_ratios(details, "new_operation/micro/x", 3.5)
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("new_operation/micro/x", stderr)

    def test_an_exempt_operation_may_be_covered_by_fixtures_or_by_exempt_fixtures(self):
        details = passing_details()  # is_prime with a fixture and an exemption (a later phase): both count.
        details["exempt"]["is_prime/realistic/big"] = exempt_entry("is_prime", "realistic")
        self.assertEqual(run_check(details)[0], 0)
        details["exempt"]["is_prime/realistic/big"]["budget_met"] = False
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("is_prime/realistic/big", stderr)
        # An exemption covers the sizes of an operation that may be exempt, and no other.
        details = passing_details()
        del details["fixtures"]["det/realistic/x"]
        details["exempt"]["det/realistic/x"] = exempt_entry("det", "realistic")
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("no realistic fixture for 1 operation: det", stderr)
        self.assertIn("exemption outside criteria.SPEED_EXEMPTIONS", stderr)

    def test_several_causes_are_all_named(self):
        details = passing_details()
        details["meta"]["source_sha256"] = "c" * 64
        set_ratios(details, "rank/micro/x", 9.0)
        details["exempt"]["verify/micro/x"]["budget_met"] = False
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        for text in ("stale details", "rank/micro/x = 9.00", "verify/micro/x", "3 causes"):
            self.assertIn(text, stderr)

    def test_unreadable_details_are_an_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.json"
            for text in (None, "{", "[]", json.dumps({"meta": {}, "fixtures": {}, "exempt": {}}),
                         json.dumps({"meta": {"source_sha256": "a"}, "fixtures": []}),
                         json.dumps({"meta": {"source_sha256": "a"}, "fixtures": {"a/micro/x": {
                             "operation": "a", "size": "micro", "ratio": "fast"}}, "exempt": {}}),
                         json.dumps({"meta": {"source_sha256": "a"}, "fixtures": {}, "exempt": {"a/micro/x": {
                             "operation": "a", "size": "micro", "budget_met": 1}}}),
                         # The rounds are what the gate holds to the limit: none, none positive, or not one per round.
                         json.dumps({"meta": {"source_sha256": "a"}, "fixtures": {"a/micro/x": {
                             "operation": "a", "size": "micro", "ratio": 1.5}}, "exempt": {}}),
                         json.dumps({"meta": {"source_sha256": "a"}, "fixtures": {"a/micro/x": {
                             "operation": "a", "size": "micro", "ratio": 1.5, "round_ratios": [1.5, 0]}},
                             "exempt": {}}),
                         json.dumps({"meta": {"source_sha256": "a"}, "fixtures": {"a/micro/x": {
                             "operation": "a", "size": "micro", "ratio": 1.5, "round_ratios": []}}, "exempt": {}}),
                         json.dumps({"meta": {"source_sha256": "a", "rounds": 3}, "fixtures": {"a/micro/x": {
                             "operation": "a", "size": "micro", "ratio": 1.5, "round_ratios": [1.5]}},
                             "exempt": {}})):
                with self.subTest(text=text):
                    if text is not None:
                        path.write_text(text, encoding="utf-8")
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        status = check_speed.main(["--details", str(path)], working_hash=HASH)
                    self.assertEqual(status, 2)
                    self.assertIn(str(path), stderr.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            for arguments in (["--limit", "0"], ["--target", "-1"], ["--limit", "2", "--target", "3"],
                              ["--limit", "inf"]):
                with self.subTest(arguments), self.assertRaises(SystemExit) as raised:
                    check_speed.main(arguments, working_hash=HASH)
                self.assertEqual(raised.exception.code, 2)

    @NEEDS_SRC
    def test_the_command_line_with_the_real_hashes_and_the_real_fixtures(self):
        details = real_details()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.json"

            def run():
                path.write_text(json.dumps(details), encoding="utf-8")
                return subprocess.run([sys.executable, str(TOOL), "--details", str(path)], cwd=REPOSITORY,
                                      capture_output=True, text=True, timeout=300)

            completed = run()
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("the speed gate passes", completed.stdout)
            self.assertIn("waived fixture within the limit", completed.stdout)  # The waived fixtures at 1.5.
            for what, change, expected in (
                    ("other sources", lambda: details["meta"].update(source_sha256="0" * 64), "stale details"),
                    ("another instrument", lambda: details["meta"].update(instrument_sha256="0" * 64),
                     "stale details: measured with instrument"),
                    ("a lost fixture", lambda: details["fixtures"].pop(next(iter(details["fixtures"]))),
                     "the details lack 1 fixture"),
                    ("an unknown fixture", lambda: details["fixtures"].update(
                        {"integer_gcd/micro/not_a_fixture": fixture_entry("integer_gcd", "micro")}),
                     "that no fixture module defines now: integer_gcd/micro/not_a_fixture")):
                details = real_details()
                change()
                with self.subTest(what):
                    completed = run()
                    self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
                    self.assertIn(expected, completed.stderr)


# The fixtures whose reference was re-pointed to the fastest same-object call (the final review of the speed gate found
# the references that favoured MathForge): each still has to give the very object that MathForge gives.
REPOINTED = (
    "cdf/realistic/outcomes_120", "complex_add/micro/gaussian_integers", "complex_mul/realistic/gaussian_fractions",
    "complex_norm_squared/micro/gaussian_integer", "complex_norm_squared/realistic/gaussian_fraction",
    "complex_pow/realistic/unit_circle_surds", "condition/micro/five_outcomes", "expand/micro/two_binomials",
    "expand_multivariate/micro/two_bilinear_factors", "expand_multivariate/realistic/scaled_twelfth_power",
    "expand_multivariate/realistic/trinomial_eighth_power", "expectation/micro/five_outcomes",
    "expectation/realistic/binomial41", "expectation/realistic/outcomes_120",
    "function_antiderivative/realistic/polynomial_times_exp", "function_antiderivative/realistic/x_exp",
    "function_antiderivative/realistic/x_log_x", "function_differentiate/micro/sin_kx",
    "function_differentiate/realistic/scaled_six_terms_first_derivative", "function_hessian/micro/two_variables",
    "function_integrate/micro/cos_kx_unit_interval", "function_integrate/realistic/x_log_x_one_to_two",
    "function_jacobian/realistic/scaled_four_by_four", "function_limit/micro/sinc_scaled",
    "function_limit/realistic/sinc", "function_limit/realistic/x_log_x_from_right",
    "function_partial_diff/micro/polynomial_and_sine", "function_taylor/realistic/scaled_exp_order20",
    "function_taylor/realistic/sinh_exp_order6", "integer_xgcd/realistic/bits4096", "inverse/realistic/scaled_16x16",
    "limit/realistic/removable_singularity", "moment/micro/five_outcomes", "moment/realistic/binomial41",
    "moment/realistic/outcomes_120", "probability/micro/five_outcomes", "probability/realistic/binomial41",
    "probability/realistic/outcomes_120", "rational_antiderivative/realistic/linear_over_irreducible_quadratic",
    "rational_antiderivative/realistic/scaled_four_factors", "rational_integrate/realistic/scaled_double_pole",
    "rational_integrate/realistic/two_simple_poles_unit_interval", "real_roots/realistic/scaled_degree_ten",
    "rowspace/micro/3x4", "rowspace/realistic/8x12_rank6", "rowspace/realistic/scaled_14x20_rank9",
    "stddev/micro/five_values", "stddev/realistic/values_200", "stddev/realistic/values_500",
    "stirling/realistic/s300_150",
)


@NEEDS_SRC
@unittest.skipUnless(importlib.util.find_spec("sympy") is not None, "needs SymPy, the reference")
class RepointedReferencesTest(unittest.TestCase):
    def test_each_repointed_reference_still_gives_the_object_that_mathforge_gives(self):
        entries = {speed_suite.entry_id(entry): entry
                   for entry in speed_suite.load_entries(speed_suite.module_names(None, None), None)}
        reset = speed_suite.Reset()
        self.assertEqual(sorted(set(REPOINTED) - set(entries)), [])
        for identifier in REPOINTED:
            with self.subTest(identifier):
                speed_suite.check_agreement(entries[identifier], 0, reset)  # SuiteError when they disagree

    def test_the_rowspace_reference_is_the_reduced_basis_and_not_an_echelon_one(self):
        import sympy
        from benchmarks.speed.fixtures import linalg
        rows = ((1, 2, 3), (2, 4, 6), (1, 0, 1))
        mine = mf.rowspace(mf.Matrix(rows))
        self.assertEqual(mine, ((mf.Rational(1), mf.Rational(0), mf.Rational(1)),
                                (mf.Rational(0), mf.Rational(1), mf.Rational(1))))
        for reference in (linalg._rowspace_matrix, linalg._rowspace_domain):  # The two calls that are references.
            with self.subTest(reference.__name__):
                self.assertTrue(linalg._same_rowspace(mine, reference(rows)()))
        echelon = sympy.Matrix(rows).rowspace()  # Matrix.rowspace(): [1, 2, 3] and [0, -2, -2], not reduced.
        self.assertFalse(linalg._same_rowspace(mine, echelon))
        self.assertTrue(linalg._same_rowspace((), []))  # The zero matrix has an empty basis.

    def test_the_stddev_reference_is_the_unreduced_root_and_the_comparison_accepts_both_forms(self):
        import sympy
        from benchmarks.speed.fixtures import stats
        variance = sympy.Rational(8, 3)
        for root in (sympy.sqrt(variance, evaluate=False), sympy.sqrt(variance)):  # Unreduced, and simplified.
            self.assertTrue(stats._same_sqrt(mf.Sqrt(mf.Rational(8, 3)), root), root)
            self.assertFalse(stats._same_sqrt(mf.Sqrt(mf.Rational(7, 3)), root), root)
        square = sympy.Rational(9, 4)  # A perfect square: MathForge returns the Rational root.
        for root in (sympy.sqrt(square, evaluate=False), sympy.sqrt(square)):
            self.assertTrue(stats._same_sqrt(mf.Rational(3, 2), root), root)
            self.assertFalse(stats._same_sqrt(mf.Rational(-3, 2), root), root)
            self.assertFalse(stats._same_sqrt(mf.Rational(1, 2), root), root)
        self.assertFalse(stats._same_sqrt(mf.Rational(3, 2), sympy.sqrt(sympy.Rational(2), evaluate=False)))


class GateChecksTest(unittest.TestCase):
    """What keeps tools/check_speed.py from passing stale, unpinned or too small details: the instrument hash, the
    measured ids, SymPy's version, the measurement protocol, the worst round and the waivers."""

    def fails_with(self, change, *expected, extra=(), **kwargs):
        details = passing_details()
        change(details)
        status, stdout, stderr = run_check(details, *extra, **kwargs)
        self.assertEqual(status, 1, stdout + stderr)
        for text in expected:
            self.assertIn(text, stderr)
        return stdout, stderr

    def test_details_that_meet_the_protocol_and_match_the_tree_pass(self):
        status, stdout, stderr = run_check(passing_details())
        self.assertEqual((status, stderr), (0, ""), stdout)
        self.assertIn("the speed gate passes", stdout)
        self.assertNotIn("not a gate run", stdout)

    def test_a_changed_instrument_or_a_missing_instrument_hash_makes_the_details_stale(self):
        status, _, stderr = run_check(passing_details(), instrument_hash="e" * 64)
        self.assertEqual(status, 1)
        for text in ("stale details", "instrument dddddddddddd", "eeeeeeeeeeee", "benchmarks/speed"):
            self.assertIn(text, stderr)
        self.fails_with(lambda details: details["meta"].pop("instrument_sha256"), "stale details", "no instrument_sha256")

    def test_the_measured_ids_must_be_the_ids_that_the_modules_define_now(self):
        details = passing_details()
        ids = (set(details["fixtures"]), set(details["exempt"]))
        # A fixture added after the run, one removed, an exemption added, one removed.
        for what, current, expected in (
                ("added", (ids[0] | {"det/micro/new"}, ids[1]), "the details lack 1 fixture that the fixture modules "
                                                                 "define now: det/micro/new"),
                ("removed", (ids[0] - {"det/micro/x"}, ids[1]), "the details measured 1 fixture that no fixture "
                                                                 "module defines now: det/micro/x"),
                ("exempt added", (ids[0], ids[1] | {"verify/micro/new"}), "the details lack 1 exempt fixture"),
                ("exempt removed", (ids[0], ids[1] - {"verify/micro/x"}), "measured 1 exempt fixture that no fixture")):
            with self.subTest(what):
                status, _, stderr = run_check(passing_details(), current_ids=current)
                self.assertEqual(status, 1)
                self.assertIn(expected, stderr)
                self.assertIn("the fixtures changed after the run", stderr)
        # Many ids are named briefly.
        status, _, stderr = run_check(passing_details(), current_ids=({f"new/micro/{n}" for n in range(9)}, ids[1]))
        self.assertIn("lack 9 fixtures", stderr)
        self.assertIn("new/micro/0, new/micro/1, new/micro/2, new/micro/3, new/micro/4, ...", stderr)
        # A partial run measured part of the fixtures: the ones it did not measure are not "lacking", an unknown id is.
        details = passing_details()
        details["meta"].update(partial=True, ops=["det"])
        details["fixtures"] = {key: value for key, value in details["fixtures"].items() if key.startswith("det/")}
        status, _, stderr = run_check(details, "--allow-quick", current_ids=ids)
        self.assertNotIn("the details lack", stderr)
        details["fixtures"]["det/micro/unknown"] = fixture_entry("det", "micro")
        status, _, stderr = run_check(details, "--allow-quick", current_ids=ids)
        self.assertIn("that no fixture module defines now: det/micro/unknown", stderr)

    def test_sympy_must_be_1_14_0(self):
        self.assertEqual(check_speed.SYMPY_VERSION, "1.14.0")
        for version in ("1.13.3", "1.14.1", None):
            with self.subTest(version):
                _, stderr = self.fails_with(lambda details: details["meta"].update(sympy=version), "SymPy 1.14.0")
                self.assertIn("no SymPy" if version is None else f"SymPy {version}", stderr)
        status, _, _ = run_check(passing_details(), "--allow-quick")
        self.assertEqual(status, 0)

    def test_the_measurement_protocol_is_required(self):
        for what, change, expected in (
                ("rounds", lambda details: set_rounds(details, 2), "too few rounds: 2"),
                ("repeat", lambda details: details["meta"].update(repeat=8), "too few repeats: 8"),
                ("cpu", lambda details: details["meta"].update(cpu=None), "not pinned to a cpu"),
                ("sample_ms missing", lambda details: details["meta"].pop("sample_ms"), "no sample_ms was recorded"),
                ("sample_ms zero", lambda details: details["meta"].update(sample_ms=0), "no sample_ms was recorded"),
                ("partial", lambda details: details["meta"].update(partial=True, ops=["det"]), "partial run"),
                ("partial missing", lambda details: details["meta"].pop("partial"), "not marked as a complete run")):
            with self.subTest(what):
                self.fails_with(change, expected)
        # Three rounds, nine repeats and a cpu number are enough, and cpu 0 is a cpu.
        details = passing_details()
        details["meta"].update(rounds=3, repeat=9, cpu=0)
        self.assertEqual(run_check(details)[0], 0)
        details = passing_details()
        details["meta"].update(repeat=21, cpu=3, sample_ms=0.5)
        set_rounds(details, 5)
        self.assertEqual(run_check(details)[0], 0)

    def test_allow_quick_waives_the_protocol_and_says_that_the_result_is_not_a_gate_run(self):
        details = passing_details()
        details["meta"].update(repeat=3, cpu=None)
        set_rounds(details, 1)
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)  # A small run does not pass the gate...
        for text in ("too few rounds: 1", "too few repeats: 3", "not pinned to a cpu"):
            self.assertIn(text, stderr)
        status, stdout, stderr = run_check(details, "--allow-quick")  # ...unless it is asked for explicitly.
        self.assertEqual((status, stderr), (0, ""), stdout)
        self.assertIn("not a gate run (--allow-quick)", stdout.splitlines()[0])
        self.assertIn("check_speed: not a gate run", stdout.splitlines()[-1])
        self.assertNotIn("the speed gate passes", stdout)  # It is never reported as the gate passing.
        set_ratios(details, "det/micro/x", 3.4, [3.4])  # The limit and the other checks stay on.
        status, stdout, stderr = run_check(details, "--allow-quick")
        self.assertEqual(status, 1)
        self.assertIn("det/micro/x = 3.40", stderr)
        self.assertIn("not a gate run (--allow-quick)", stdout)
        details = passing_details()  # The staleness checks stay on too.
        status, _, stderr = run_check(details, "--allow-quick", working_hash="f" * 64)
        self.assertEqual(status, 1)
        self.assertIn("stale details", stderr)

    def test_a_fixture_is_held_to_the_limit_in_every_round_while_the_median_is_what_is_reported(self):
        details = passing_details()
        set_ratios(details, "det/micro/x", 2.1, [1.5, 3.4, 2.1])  # The median 2.1 is fine; one round is not.
        status, stdout, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("ratio above the limit 3.0 in a round: det/micro/x = 3.40", stderr)
        self.assertIn("median 2.10", stderr)
        self.assertIn("rounds 1.50, 3.40, 2.10", stderr)
        self.assertIn("2.10  det/micro/x", stdout)  # The report keeps the median.
        self.assertIn("rounds 1.50, 3.40, 2.10", stdout)
        set_ratios(details, "det/micro/x", 2.1, [1.5, 3.0, 2.1])  # The limit is inclusive in a round too.
        self.assertEqual(run_check(details)[0], 0)
        set_ratios(details, "det/micro/x", 3.2, [1.5, 2.0, 2.1])  # A pooled median above the limit still fails.
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("ratio above the limit 3.0: det/micro/x = 3.20", stderr)

    def test_waived_fixtures_are_printed_with_their_reason_and_do_not_fail_the_gate(self):
        bezout = ("the independently checked Bezout certificate dominates at this size (v0.8.0 known limitation)")
        structured = "solve_system still runs the v0.7 structured pipeline; accelerated in v0.9"
        details = passing_details()
        for identifier, ratio in (("integer_gcd/realistic/bits30000", 17.7), ("integer_gcd/realistic/rsa_moduli", 14.3),
                                  ("integer_lcm/realistic/bits20000", 16.4), ("integer_lcm/realistic/rsa_moduli", 10.5),
                                  ("solve_system/micro/two_by_two", 14.9), ("solve_system/realistic/dense_6_by_6", 9.7),
                                  ("solve_system/realistic/scaled_dense_12_by_12", 8.5),
                                  ("solve_system/realistic/underdetermined_4_by_6", 12.1)):
            details["fixtures"][identifier] = fixture_entry(identifier.split("/")[0], identifier.split("/")[1], ratio)
            details["fixtures"][identifier]["round_ratios"] = [ratio, ratio * 1.1, ratio * 0.9]
        status, stdout, stderr = run_check(details)
        self.assertEqual((status, stderr), (0, ""), stdout)
        self.assertIn("the speed gate passes", stdout)
        waived = [line for line in stdout.splitlines() if line.startswith("waived: ")]
        self.assertEqual(len(waived), 8)
        self.assertEqual(sum(line.startswith(f"waived: {bezout} [") for line in waived), 4)
        self.assertEqual(sum(line.startswith(f"waived: {structured} [") for line in waived), 4)
        self.assertTrue(any("[integer_gcd/realistic/bits30000: ratio 17.70, worst 19.47" in line for line in waived))
        # A waiver is for its fixture only: an unwaived one next to it still fails.
        set_ratios(details, "integer_gcd/micro/x", 17.7)
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("integer_gcd/micro/x = 17.70", stderr)
        self.assertNotIn("bits30000", stderr)
        # A waived fixture that comes back within the limit is reported as such.
        details = passing_details()
        details["fixtures"]["integer_lcm/realistic/rsa_moduli"] = fixture_entry("integer_lcm", "realistic", 2.4)
        status, stdout, stderr = run_check(details)
        self.assertEqual((status, stderr), (0, ""), stdout)
        self.assertNotIn("waived: ", stdout)
        self.assertIn("waived fixture within the limit: integer_lcm/realistic/rsa_moduli: ratio 2.40", stdout)
        self.assertIn("its waiver (the independently checked Bezout certificate", stdout)
        self.assertIn("is no longer needed", stdout)

    def test_a_waiver_does_not_excuse_other_failures_of_the_same_fixture(self):
        details = passing_details()
        details["fixtures"]["solve_system/micro/two_by_two"] = fixture_entry("solve_system", "micro", 14.9)
        details["meta"]["sympy"] = "1.13.3"  # The waiver is for the ratio only.
        status, _, stderr = run_check(details)
        self.assertEqual(status, 1)
        self.assertIn("SymPy 1.14.0", stderr)
        self.assertNotIn("two_by_two", stderr)

    def test_the_waivers_name_real_fixtures_and_cover_every_solve_system_fixture(self):
        fixture_ids, _ = speed_suite.current_ids()
        self.assertEqual(set(check_speed.WAIVED) - fixture_ids, set())  # No waiver for a fixture that is gone.
        self.assertEqual({identifier for identifier in fixture_ids if identifier.startswith("solve_system/")},
                         {identifier for identifier in check_speed.WAIVED if identifier.startswith("solve_system/")})
        self.assertEqual({identifier for identifier in check_speed.WAIVED if not identifier.startswith("solve_system/")},
                         {"integer_gcd/realistic/bits30000", "integer_gcd/realistic/rsa_moduli",
                          "integer_lcm/realistic/bits20000", "integer_lcm/realistic/rsa_moduli",
                          "solve/micro/quadratic_with_integer_roots"})  # The user's v0.8.0 decisions, nothing more.
        self.assertEqual({reason for identifier, reason in check_speed.WAIVED.items()
                          if identifier.startswith("solve_system/")},
                         {"solve_system still runs the v0.7 structured pipeline; accelerated in v0.9"})
        self.assertEqual({reason for identifier, reason in check_speed.WAIVED.items()
                          if not identifier.startswith("solve_system/")},
                         {"the independently checked Bezout certificate dominates at this size "
                          "(v0.8.0 known limitation)"})


if __name__ == "__main__":
    unittest.main()
