"""The v0.7 value corpus holds: every stored entry, rebuilt from its fixture module and run on the current code,
gives the same to_data value or raises the same exception class with the same code.

The corpus is tests/fixtures/v8/values_v07/<area>.json, written from a v0.7 tree by tools/snapshot_v07_values.py. Its
entries are rebuilt from benchmarks/speed/fixtures/<area>.py. With no corpus files there is nothing to check. The
rest of this file tests the snapshot tool and the check itself on throwaway fixture packages, so that both are known
to work before the first corpus file exists.
"""

import ast
import contextlib
import dataclasses
import importlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import mathforge
from mathforge.serialization import to_data

REPOSITORY = Path(__file__).resolve().parents[2]
CORPUS = REPOSITORY / "tests" / "fixtures" / "v8" / "values_v07"
TOOL = REPOSITORY / "tools" / "snapshot_v07_values.py"
SPEED = REPOSITORY / "benchmarks" / "speed"
PACKAGE = "benchmarks.speed.fixtures"
REFERENCE_PACKAGES = ("sympy", "mpmath")  # Fixture modules import these only inside their sympy builders.
PACKAGE_COUNTER = itertools.count()


def load_tool():
    spec = importlib.util.spec_from_file_location("snapshot_v07_values", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(outcome):
    """A short account of a stored or fresh outcome for a failure message."""
    if outcome["error"] is not None:
        return "raises " + "/".join(str(part) for part in outcome["error"])
    text = json.dumps(outcome["value"], sort_keys=True)
    return "returns " + (text if len(text) <= 160 else text[:157] + "...")


def well_formed(entry):
    """A stored entry has exactly the keys value and error, and exactly one of them is not null."""
    if type(entry) is not dict or set(entry) != {"value", "error"}:
        return False
    error = entry["error"]
    if error is None:
        return type(entry["value"]) is dict
    return (entry["value"] is None and type(error) is list and len(error) == 2 and type(error[0]) is str
            and (error[1] is None or type(error[1]) is str))


def problems_in(tool, stored, package, area):
    """Every way in which the stored entries of one area differ from what the current code gives for them."""
    try:
        entries = tool.module_entries(tool.import_area(package, area))
    except tool.SnapshotError as error:
        return [str(error)]
    problems = []
    for entry_id, expected in sorted(stored.items()):
        if not well_formed(expected):
            problems.append(f"{entry_id}: the stored entry is not {{value, error}} with exactly one of them set")
        elif entry_id not in entries:
            problems.append(f"{entry_id}: stored, but {package}.{area} defines no such fixture variant or edge case")
        else:
            try:
                current = tool.run_entry(entry_id, entries[entry_id])
            except tool.SnapshotError as error:
                problems.append(str(error))
                continue
            if current != expected:
                problems.append(f"{entry_id}: v0.7 {describe(expected)}, now {describe(current)}")
    return problems


def import_time_imports(tree):
    """The modules imported by statements that run when a module loads, that is, outside function bodies."""
    names, pending = [], list(ast.iter_child_nodes(tree))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.append(node.module or "")
        pending.extend(ast.iter_child_nodes(node))
    return names


class StoredValuesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()
        cls.tool.repository_importable()

    def test_every_stored_entry_still_holds(self):
        for path in sorted(CORPUS.glob("*.json")) if CORPUS.is_dir() else ():
            with self.subTest(area=path.stem):
                stored = json.loads(path.read_text(encoding="utf-8"))
                self.assertIs(type(stored), dict, "a corpus file is one object, entry id -> outcome")
                problems = problems_in(self.tool, stored, PACKAGE, path.stem)
                self.assertFalse(problems, f"{len(problems)} stored entries differ from the current code:\n"
                                 + "\n".join(problems[:20]))

    def test_fixture_modules_import_the_references_only_inside_builders(self):
        paths = [SPEED / "fixture.py", *sorted((SPEED / "fixtures").glob("*.py"))]
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            with self.subTest(path=path.relative_to(REPOSITORY).as_posix()):
                imported = import_time_imports(ast.parse(path.read_bytes(), filename=str(path)))
                banned = sorted({name for name in imported if name.partition(".")[0] in REFERENCE_PACKAGES})
                self.assertEqual(banned, [], "import SymPy and mpmath inside the sympy builders, not at module level")

    def test_import_time_imports_skip_function_bodies_only(self):
        source = ("import json\n"
                  "from os import path\n"
                  "try:\n    import sympy\nexcept ImportError:\n    sympy = None\n"
                  "class Holder:\n    import mpmath\n"
                  "def builder():\n    import sympy.stats\n"
                  "async def later():\n    import mpmath\n"
                  "from . import sibling\n")
        self.assertEqual(sorted(import_time_imports(ast.parse(source))), ["json", "mpmath", "os", "sympy"])


class FixtureInterfaceTest(unittest.TestCase):
    """The interface that the fixture, snapshot and suite agents share is fixed: field names, order and defaults."""

    @classmethod
    def setUpClass(cls):
        load_tool().repository_importable()
        from benchmarks.speed import fixture
        cls.fixture = fixture

    def test_the_fields_are_the_pinned_ones_in_order(self):
        def names(cls):
            return [field.name for field in dataclasses.fields(cls)]
        self.assertEqual(names(self.fixture.Fixture), ["operation", "size", "label", "mathforge", "sympy", "agree",
                                                       "reference", "variants", "v07"])
        self.assertEqual(names(self.fixture.ExemptFixture), ["operation", "size", "label", "timed", "budget", "reason"])
        self.assertEqual(names(self.fixture.EdgeCase), ["operation", "label", "mathforge"])

    def test_defaults_ids_and_immutability(self):
        fixture = self.fixture.Fixture("integer_gcd", "micro", "small", lambda i: (len, ((),), {}),
                                       lambda i: (lambda: None), lambda a, b: True, "igcd")
        self.assertEqual((fixture.variants, fixture.v07), (1, ""))
        self.assertEqual(fixture.id, "integer_gcd/micro/small")
        for instance in (fixture, self.fixture.ExemptFixture("verify", "micro", "x", None, None, "no reference"),
                         self.fixture.EdgeCase("integer_gcd", "x", None)):
            with self.subTest(type(instance).__name__), self.assertRaises(dataclasses.FrozenInstanceError):
                instance.operation = "other"

    def test_symbols_have_fixed_uids(self):
        x = self.fixture.symbol("x", 1)
        self.assertIs(type(x), mathforge.Symbol)
        self.assertEqual((x.name, x.uid), ("x", "00000000-0000-0000-0000-000000000001"))
        self.assertEqual(x, self.fixture.symbol("x", 1))
        self.assertNotEqual(x, self.fixture.symbol("x", 2))
        self.assertEqual(self.fixture.symbol("y", 255).uid, "00000000-0000-0000-0000-0000000000ff")
        self.assertEqual(to_data(x), to_data(self.fixture.symbol("x", 1)))  # Documents agree too.


DEMO_AREA = '''
    import mathforge as mf
    from benchmarks.speed.fixture import EdgeCase, ExemptFixture, Fixture, symbol


    class DemoError(Exception):
        code = "demo_code"


    def never(i):
        raise AssertionError("the SymPy side is not timed here")


    def agree(mathforge_value, sympy_value):
        return True


    def fail_with_code():
        raise DemoError("boom")


    def fail_without_code():
        raise ZeroDivisionError("division by zero")


    def a_list():
        return [1, 2, 3]


    def a_float():
        return 1.5


    def a_mixed_tuple():
        return (1, 2.5)


    def nothing():
        return None


    def a_symbol():
        return symbol("x", 7)


    FIXTURES = (
        Fixture("integer_gcd", "micro", "small", lambda i: (mf.integer_gcd, (84 + 6 * i, 30 + 6 * i), {}), never,
                agree, "demo", variants=1000),
        Fixture("factorial", "realistic", "one", lambda i: (mf.factorial, (20,), {}), never, agree, "demo"),
        ExemptFixture("verify", "micro", "demo", lambda: (nothing, (), {}), lambda: (nothing, (), {}), "demo"),
    )
    EDGE_CASES = (
        EdgeCase("integer_gcd", "error_with_code", lambda: (fail_with_code, (), {})),
        EdgeCase("integer_gcd", "error_without_code", lambda: (fail_without_code, (), {})),
        EdgeCase("integer_gcd", "list", lambda: (a_list, (), {})),
        EdgeCase("integer_gcd", "float", lambda: (a_float, (), {})),
        EdgeCase("integer_gcd", "tuple_with_float", lambda: (a_mixed_tuple, (), {})),
        EdgeCase("integer_gcd", "none", lambda: (nothing, (), {})),
        EdgeCase("integer_gcd", "symbol", lambda: (a_symbol, (), {})),
    )
'''

DEMO_OTHER_AREA = '''
    import mathforge as mf
    from benchmarks.speed.fixture import EdgeCase, Fixture


    def never(i):
        raise AssertionError("the SymPy side is not timed here")


    FIXTURES = (
        Fixture("factorial", "micro", "small", lambda i: (mf.factorial, (5 + i,), {}), never, lambda a, b: True,
                "demo", variants=5),
    )
    EDGE_CASES = (
        EdgeCase("factorial", "third", lambda: (mf.Rational, (1, 3), {})),
    )
'''


def value(data):
    return {"value": to_data(data), "error": None}


def error(name, code):
    return {"value": None, "error": [name, code]}


DEMO_AREA_DOCUMENT = {
    "integer_gcd/micro/small#0": value(6),
    "integer_gcd/micro/small#1": value(18),
    "integer_gcd/micro/small#2": value(6),
    "factorial/realistic/one#0": value(math.factorial(20)),
    "integer_gcd/edge/error_with_code": error("DemoError", "demo_code"),
    "integer_gcd/edge/error_without_code": error("ZeroDivisionError", None),
    "integer_gcd/edge/list": {"value": {"repr": "[1, 2, 3]"}, "error": None},
    "integer_gcd/edge/float": {"value": {"repr": "1.5"}, "error": None},
    "integer_gcd/edge/tuple_with_float": {"value": {"repr": "(1, 2.5)"}, "error": None},
    "integer_gcd/edge/none": value(None),
    "integer_gcd/edge/symbol": value(mathforge.Symbol("x", uid="00000000-0000-0000-0000-000000000007")),
}

DEMO_OTHER_DOCUMENT = {  # Five variants are declared; entries cover variants 0, 1 and 2.
    "factorial/micro/small#0": value(math.factorial(5)),
    "factorial/micro/small#1": value(math.factorial(6)),
    "factorial/micro/small#2": value(math.factorial(7)),
    "factorial/edge/third": value(mathforge.Rational(1, 3)),
}


class DemoPackages(unittest.TestCase):
    """Base class for tests that snapshot throwaway fixture packages."""

    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()
        cls.tool.repository_importable()

    def make_package(self, **modules):
        """Write a throwaway fixture package {area: source} into a temporary directory and make it importable.

        Returns the package name and the directory that holds it.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        name = f"demo_speed_fixtures_{next(PACKAGE_COUNTER)}"
        package = Path(directory.name) / name
        package.mkdir()
        (package / "__init__.py").write_text('"""Demo fixtures."""\n', encoding="utf-8")
        for area, source in modules.items():
            (package / f"{area}.py").write_text(textwrap.dedent(source), encoding="utf-8")
        sys.path.insert(0, directory.name)
        importlib.invalidate_caches()

        def forget():
            sys.path.remove(directory.name)
            for loaded in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
                del sys.modules[loaded]
        self.addCleanup(forget)  # Runs before the directory is removed.
        return name, Path(directory.name)


class SnapshotToolTest(DemoPackages):
    def snapshot(self, *arguments):
        """Run main in this process; return (stdout, stderr). A refusal raises SystemExit."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.tool.main(list(arguments))
        return out.getvalue(), err.getvalue()

    def test_a_fixture_contributes_variants_zero_to_two_and_edge_cases_one_entry_each(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        entries = self.tool.module_entries(self.tool.import_area(package, "mini"))
        self.assertEqual(set(entries), set(DEMO_AREA_DOCUMENT))  # No entry for the ExemptFixture.
        self.assertEqual(len(entries), 11)

    def test_the_stored_forms_of_values_errors_and_plain_builtins(self):
        package, _ = self.make_package(mini=DEMO_AREA, other=DEMO_OTHER_AREA)
        self.assertEqual(self.tool.snapshot_area(package, "mini"), DEMO_AREA_DOCUMENT)
        self.assertEqual(self.tool.snapshot_area(package, "other"), DEMO_OTHER_DOCUMENT)

    def test_the_command_line_writes_one_pinned_file_per_area(self):
        package, directory = self.make_package(mini=DEMO_AREA, other=DEMO_OTHER_AREA)
        out_dir = directory / "out"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(directory), str(Path(mathforge.__file__).resolve().parents[1])]
                                            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with tempfile.TemporaryDirectory() as elsewhere:  # The tool must not depend on the working directory.
            done = subprocess.run([sys.executable, str(TOOL), "--package", package, "--areas", "mini,other",
                                   "--out-dir", str(out_dir)], capture_output=True, text=True, env=env, cwd=elsewhere,
                                  timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(f"snapshotting mathforge {mathforge.__version__} from {Path(mathforge.__file__).resolve().parent}",
                      done.stderr)  # The tree that was snapshotted is stated.
        self.assertEqual(sorted(path.name for path in out_dir.iterdir()), ["mini.json", "other.json"])
        self.assertEqual([line.rpartition(" (")[2] for line in done.stdout.splitlines()],
                         ["11 entries)", "4 entries)"])
        mini = (out_dir / "mini.json").read_bytes()
        self.assertEqual(json.loads(mini), DEMO_AREA_DOCUMENT)
        self.assertNotIn(b"\r", mini)  # LF line ends, whatever the platform.
        self.assertEqual(mini.decode("ascii"), json.dumps(DEMO_AREA_DOCUMENT, sort_keys=True, indent=1) + "\n")
        self.assertEqual(json.loads((out_dir / "other.json").read_bytes()), DEMO_OTHER_DOCUMENT)

    def test_a_rerun_changes_nothing_and_a_different_snapshot_needs_force(self):
        package, directory = self.make_package(mini=DEMO_AREA)
        out_dir = directory / "out"
        path = out_dir / "mini.json"
        arguments = ["--package", package, "--areas", "mini", "--out-dir", str(out_dir)]
        self.snapshot(*arguments)
        first, stamp = path.read_bytes(), path.stat().st_mtime_ns
        self.snapshot(*arguments)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (first, stamp))  # Not even rewritten.
        # Change a fixture: a different length, so that no stale bytecode of the module can be reused.
        source = directory / package / "mini.py"
        source.write_text(source.read_text(encoding="utf-8").replace("(20,), {})", "(2 * 10 + 1,), {})"),
                          encoding="utf-8")
        sys.modules.pop(f"{package}.mini", None)
        importlib.invalidate_caches()
        with self.assertRaises(SystemExit) as refused:
            self.snapshot(*arguments)
        self.assertIn("Refusing to replace", str(refused.exception))
        self.assertIn("mini.json", str(refused.exception))
        self.assertEqual(path.read_bytes(), first)
        self.snapshot(*arguments, "--force")
        self.assertEqual(json.loads(path.read_bytes())["factorial/realistic/one#0"], value(math.factorial(21)))

    def test_nothing_is_written_when_any_file_is_refused(self):
        package, directory = self.make_package(mini=DEMO_AREA, other=DEMO_OTHER_AREA)
        out_dir = directory / "out"
        out_dir.mkdir()
        (out_dir / "other.json").write_text("{}\n", encoding="utf-8")  # A different, existing reference.
        with self.assertRaises(SystemExit) as refused:
            self.snapshot("--package", package, "--areas", "mini,other", "--out-dir", str(out_dir))
        self.assertIn("other.json", str(refused.exception))
        self.assertNotIn("mini.json", str(refused.exception))
        self.assertEqual(sorted(path.name for path in out_dir.iterdir()), ["other.json"])
        self.assertEqual((out_dir / "other.json").read_text(encoding="utf-8"), "{}\n")

    def test_the_default_package_is_the_repository_fixtures(self):
        arguments = self.tool.build_parser().parse_args(["--areas", "ntheory", "--out-dir", "somewhere"])
        self.assertEqual(arguments.package, "benchmarks.speed.fixtures")
        self.assertEqual(arguments.areas, "ntheory")
        self.assertFalse(arguments.force)

    def test_areas_are_validated(self):
        for text in ("", "a,,b", "a,a", "a b", "../x", "a/b"):
            with self.subTest(text), self.assertRaises(self.tool.SnapshotError):
                self.tool.parse_areas(text)
        self.assertEqual(self.tool.parse_areas("ntheory, discrete"), ["ntheory", "discrete"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as missing:
            self.tool.main(["--out-dir", "somewhere"])
        self.assertEqual(missing.exception.code, 2)

    def test_a_value_to_data_rejects_that_is_not_a_plain_builtin_is_refused(self):
        package, _ = self.make_package(bad="""
            from benchmarks.speed.fixture import EdgeCase
            FIXTURES = ()
            EDGE_CASES = (EdgeCase("integer_gcd", "an_object", lambda: (object, (), {})),)
        """)
        with self.assertRaisesRegex(self.tool.SnapshotError, r"integer_gcd/edge/an_object: to_data rejects a object"):
            self.tool.snapshot_area(package, "bad")

    def test_inputs_that_carry_random_state_are_refused(self):
        package, _ = self.make_package(random="""
            import mathforge as mf
            from benchmarks.speed.fixture import EdgeCase
            FIXTURES = ()
            EDGE_CASES = (EdgeCase("integer_gcd", "fresh_symbol", lambda: (mf.Symbol, ("x",), {})),)
        """)
        with self.assertRaisesRegex(self.tool.SnapshotError, r"integer_gcd/edge/fresh_symbol: two runs gave different"):
            self.tool.snapshot_area(package, "random")

    def test_a_fixture_that_cannot_build_its_call_is_reported_with_its_entry(self):
        package, _ = self.make_package(broken="""
            from benchmarks.speed.fixture import Fixture
            def broken(i):
                raise KeyError(i)
            FIXTURES = (Fixture("integer_gcd", "micro", "x", broken, broken, broken, "demo", variants=4),)
            EDGE_CASES = ()
        """)
        with self.assertRaisesRegex(self.tool.SnapshotError, r"integer_gcd/micro/x#0: the fixture could not build"):
            self.tool.snapshot_area(package, "broken")

    def test_malformed_modules_are_reported(self):
        package, _ = self.make_package(
            nothing="X = 1\n",
            only_fixtures="FIXTURES = ()\n",
            duplicate="""
                from benchmarks.speed.fixture import EdgeCase
                FIXTURES = ()
                EDGE_CASES = (EdgeCase("a", "same", lambda: (len, ((),), {})),
                              EdgeCase("a", "same", lambda: (len, ((),), {})))
            """,
            alien="FIXTURES = (1,)\nEDGE_CASES = ()\n",
            alien_edge="FIXTURES = ()\nEDGE_CASES = ('x',)\n",
            no_variants="""
                from benchmarks.speed.fixture import Fixture
                FIXTURES = (Fixture("a", "micro", "x", None, None, None, "demo", variants=0),)
                EDGE_CASES = ()
            """)
        expected = {
            "missing": "no fixture module",
            "nothing": "defines no FIXTURES",
            "only_fixtures": "defines no EDGE_CASES",
            "duplicate": "the entry id a/edge/same occurs twice",
            "alien": "holds a int",
            "alien_edge": "holds a str, not an EdgeCase",
            "no_variants": "variants must be a positive integer",
        }
        for area, message in expected.items():
            with self.subTest(area), self.assertRaisesRegex(self.tool.SnapshotError, message):
                self.tool.snapshot_area(package, area)

    def test_an_import_error_inside_a_fixture_module_is_not_reported_as_a_missing_module(self):
        package, _ = self.make_package(
            needs_more="import module_that_does_not_exist_anywhere\nFIXTURES = ()\nEDGE_CASES = ()\n")
        with self.assertRaises(ModuleNotFoundError) as raised:
            self.tool.snapshot_area(package, "needs_more")
        self.assertEqual(raised.exception.name, "module_that_does_not_exist_anywhere")


class CorpusCheckTest(DemoPackages):
    """problems_in, the check that test_every_stored_entry_still_holds applies to the real corpus."""

    def test_an_unchanged_corpus_has_no_problems(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        self.assertEqual(problems_in(self.tool, DEMO_AREA_DOCUMENT, package, "mini"), [])

    def test_a_changed_value_is_reported_with_both_sides(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        stored = {**DEMO_AREA_DOCUMENT, "integer_gcd/micro/small#1": value(19)}
        (problem,) = problems_in(self.tool, stored, package, "mini")
        self.assertTrue(problem.startswith("integer_gcd/micro/small#1: v0.7 returns "), problem)
        self.assertIn("now returns ", problem)

    def test_a_value_that_became_an_error_and_an_error_that_became_a_value_are_reported(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        stored = {**DEMO_AREA_DOCUMENT,
                  "integer_gcd/micro/small#0": error("InvalidInput", "invalid_input"),
                  "integer_gcd/edge/error_with_code": value(6)}
        problems = problems_in(self.tool, stored, package, "mini")
        self.assertEqual(sorted(problem.partition(":")[0] for problem in problems),
                         ["integer_gcd/edge/error_with_code", "integer_gcd/micro/small#0"])
        self.assertTrue(any("v0.7 raises InvalidInput/invalid_input, now returns " in problem for problem in problems))
        self.assertTrue(any("v0.7 returns " in problem and "now raises DemoError/demo_code" in problem
                            for problem in problems))

    def test_a_changed_error_class_or_code_is_reported(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        stored = {**DEMO_AREA_DOCUMENT,
                  "integer_gcd/edge/error_with_code": error("DemoError", "other_code"),
                  "integer_gcd/edge/error_without_code": error("ArithmeticError", None)}
        self.assertEqual(len(problems_in(self.tool, stored, package, "mini")), 2)

    def test_a_stored_entry_without_a_fixture_is_reported(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        stored = {**DEMO_AREA_DOCUMENT, "integer_gcd/micro/small#3": value(6)}  # Only variants 0-2 have entries.
        (problem,) = problems_in(self.tool, stored, package, "mini")
        self.assertIn("integer_gcd/micro/small#3: stored, but", problem)
        self.assertIn("defines no such fixture variant or edge case", problem)

    def test_a_malformed_stored_entry_is_reported(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        for label, entry in (("both", {"value": value(6)["value"], "error": ["X", None]}),
                             ("neither", {"value": None, "error": None}),
                             ("extra key", {"value": value(6)["value"], "error": None, "note": 1}),
                             ("not a dict", [1]),
                             ("error of one part", {"value": None, "error": ["X"]}),
                             ("non-text code", {"value": None, "error": ["X", 3]})):
            with self.subTest(label):
                (problem,) = problems_in(self.tool, {"integer_gcd/micro/small#0": entry}, package, "mini")
                self.assertIn("the stored entry is not", problem)

    def test_a_missing_fixture_module_is_a_problem_not_a_pass(self):
        package, _ = self.make_package(mini=DEMO_AREA)
        (problem,) = problems_in(self.tool, DEMO_AREA_DOCUMENT, package, "vanished")
        self.assertIn("no fixture module", problem)

    def test_a_fixture_whose_value_changed_is_reported(self):
        changed = DEMO_AREA.replace("84 + 6 * i, 30 + 6 * i", "85 + 6 * i, 30 + 6 * i")
        package, _ = self.make_package(mini=changed)
        problems = problems_in(self.tool, DEMO_AREA_DOCUMENT, package, "mini")
        self.assertEqual(sorted(problem.partition(":")[0] for problem in problems),
                         ["integer_gcd/micro/small#0", "integer_gcd/micro/small#1", "integer_gcd/micro/small#2"])


if __name__ == "__main__":
    unittest.main()
