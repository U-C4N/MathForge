"""The v0.7 value corpus holds: every stored entry, rebuilt from its fixture module and run on the current code,
gives the same to_data value or raises the same exception class with the same code.

The corpus is tests/fixtures/v8/values_v07/<area>.json, written from a v0.7 tree by tools/snapshot_v07_values.py. Its
entries are rebuilt from benchmarks/speed/fixtures/<area>.py. The corpus must be there: the ten area files exist and their
entry ids cover the 101 operations of capabilities_v07.json (every one but verify), so that the check cannot pass
because it found nothing to check. The rest of this file tests the snapshot tool and the check itself on throwaway
fixture packages.
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
import types
import unittest

import mathforge
from mathforge.serialization import to_data

REPOSITORY = Path(__file__).resolve().parents[2]
CORPUS = REPOSITORY / "tests" / "fixtures" / "v8" / "values_v07"
CAPABILITIES = REPOSITORY / "tests" / "fixtures" / "v8" / "capabilities_v07.json"
TOOL = REPOSITORY / "tools" / "snapshot_v07_values.py"
SPEED = REPOSITORY / "benchmarks" / "speed"
PACKAGE = "benchmarks.speed.fixtures"
AREAS = ("algebra", "analysis", "discrete", "equations", "expr", "integrals", "linalg", "ntheory", "series", "stats")
REFERENCE_PACKAGES = ("sympy", "mpmath")  # Fixture modules import these only inside their sympy builders.
PACKAGE_COUNTER = itertools.count()

# A fresh interpreter that blocks SymPy and mpmath (an import of either raises ImportError), then imports the fixture
# interface and every named area module of a package: what a module-level call into a builder cannot survive.
IMPORT_WITHOUT_REFERENCES = """
    import importlib, sys
    package_root, repository, package, areas = sys.argv[1:5]
    sys.path[:0] = [path for path in (repository, package_root, *sys.argv[5:]) if path]
    for name in ("sympy", "mpmath"):
        sys.modules[name] = None
    try:
        import sympy
    except ImportError:
        pass
    else:
        raise SystemExit("the block did not hold: sympy imported")
    import benchmarks.speed.fixture
    for area in areas.split(","):
        importlib.import_module(package + "." + area)
    print("imported", len(areas.split(",")), "modules")
"""


def import_without_references(package, areas, *extra_paths):
    """Import the area modules of a fixture package in a fresh interpreter in which SymPy and mpmath cannot be imported."""
    package_root = Path(mathforge.__file__).resolve().parents[1]
    return subprocess.run([sys.executable, "-I", "-c", textwrap.dedent(IMPORT_WITHOUT_REFERENCES), str(package_root),
                           str(REPOSITORY), package, ",".join(areas), *map(str, extra_paths)],
                          capture_output=True, text=True, timeout=300)


def capability_names():
    return [entry["name"] for entry in json.loads(CAPABILITIES.read_text(encoding="utf-8"))["operations"]]


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

    def test_the_corpus_is_there_the_ten_area_files_covering_the_101_value_operations(self):
        """Without this, a missing corpus is a pass with nothing checked."""
        self.assertTrue(CORPUS.is_dir(), f"{CORPUS} is missing: the v0.7 value corpus is part of the tests")
        self.assertEqual(sorted(path.stem for path in CORPUS.glob("*.json")), sorted(AREAS))
        modules = sorted(path.stem for path in (SPEED / "fixtures").glob("*.py") if path.stem != "__init__"
                         and not path.stem.startswith("_"))
        self.assertEqual(modules, sorted(AREAS), "every fixture module has a corpus file and the other way round")
        names = capability_names()
        self.assertEqual(len(names), 102)
        expected = set(names) - {"verify"}  # verify has no value to store: its fixtures are exempt.
        self.assertEqual(len(expected), 101)
        covered = set()
        for area in AREAS:
            stored = json.loads((CORPUS / f"{area}.json").read_text(encoding="utf-8"))
            self.assertTrue(stored, f"{area}.json holds no entries")
            covered |= {entry_id.partition("/")[0] for entry_id in stored}
        self.assertEqual(sorted(expected - covered), [], "operations without a stored entry")
        self.assertEqual(sorted(covered - expected), [], "entries of something that is not one of the 101 operations")

    def test_every_stored_entry_still_holds(self):
        for area in AREAS:  # Not whatever files exist: the corpus test above says that all ten do.
            with self.subTest(area=area):
                path = CORPUS / f"{area}.json"
                self.assertTrue(path.is_file(), f"{path} is missing")
                stored = json.loads(path.read_text(encoding="utf-8"))
                self.assertIs(type(stored), dict, "a corpus file is one object, entry id -> outcome")
                self.assertTrue(stored, f"{area}.json holds no entries")
                problems = problems_in(self.tool, stored, PACKAGE, area)
                self.assertFalse(problems, f"{len(problems)} stored entries differ from the current code:\n"
                                 + "\n".join(problems[:20]))

    def test_every_fixture_module_imports_without_sympy_or_mpmath(self):
        """In a fresh interpreter that blocks both, so that a call into a builder at module level cannot pass for an
        import inside a function (the analysis module built a SymPy reference at import once)."""
        done = import_without_references(PACKAGE, AREAS)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("imported 10 modules", done.stdout)

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
        """Run main in this process; return (stdout, stderr). A refusal raises SystemExit. The tree check is skipped:
        the mathforge of this process is the code under test, not a v0.7 tree."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.tool.main([*arguments, "--allow-current-tree"])
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
                                   "--out-dir", str(out_dir), "--allow-current-tree"], capture_output=True, text=True,
                                  env=env, cwd=elsewhere, timeout=300)
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
        self.assertFalse(arguments.allow_current_tree)  # The check that mathforge is a v0.7.x tree is on by default.

    def test_a_tree_that_is_not_a_v07_tree_outside_this_repository_is_refused(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            def tree(version, directory=None):
                where = Path(directory or elsewhere) / "mathforge" / "__init__.py"
                return types.SimpleNamespace(__file__=str(where), __version__=version)

            for version in ("0.7.0", "0.7.9", "0.7.12", "0.7.0rc1"):
                with self.subTest(accepted=version):
                    self.tool.check_tree(tree(version))
            for version in ("0.8.0", "0.7", "0.70.1", "0.6.9", "1.7.0", "", None, 7):
                with self.subTest(refused=version), self.assertRaises(self.tool.SnapshotError) as refused:
                    self.tool.check_tree(tree(version))
                self.assertIn("not a 0.7.x release", str(refused.exception))
            # Under this repository's src: refused whatever the version says, so that the code the corpus judges
            # cannot pass for the tree it is a snapshot of.
            with self.assertRaises(self.tool.SnapshotError) as refused:
                self.tool.check_tree(tree("0.7.0", REPOSITORY / "src"))
            self.assertIn("under this repository's src", str(refused.exception))

    def test_the_command_line_refuses_the_mathforge_of_this_repository(self):
        package, directory = self.make_package(mini=DEMO_AREA)
        out_dir = directory / "out"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(directory), str(Path(mathforge.__file__).resolve().parents[1])]
                                            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        done = subprocess.run([sys.executable, str(TOOL), "--package", package, "--areas", "mini", "--out-dir",
                               str(out_dir)], capture_output=True, text=True, env=env, cwd=directory, timeout=300)
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("snapshot_v07_values:", done.stderr)
        self.assertIn("v0.7", done.stderr)  # This tree's src, or in an installed harness a 0.8 version.
        self.assertFalse(out_dir.exists())  # Nothing was written.

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


class ImportWithoutReferencesTest(DemoPackages):
    """The subprocess check that fixture modules load without SymPy and mpmath, on throwaway packages: it passes a
    module that imports them inside a builder, and fails one that imports them at module level or calls into a builder
    that does."""

    def test_an_import_inside_a_builder_is_fine(self):
        package, directory = self.make_package(lazy="""
            def builder(i):
                import sympy
                return lambda: sympy.Integer(i)
            FIXTURES = ()
            EDGE_CASES = ()
        """)
        done = import_without_references(package, ["lazy"], directory)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("imported 1 modules", done.stdout)

    def test_an_import_at_module_level_is_refused(self):
        package, directory = self.make_package(top="""
            import sympy
            FIXTURES = ()
            EDGE_CASES = ()
        """)
        done = import_without_references(package, ["top"], directory)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("sympy", done.stderr)

    def test_a_call_into_a_builder_at_module_level_is_refused(self):
        package, directory = self.make_package(called="""
            def builder(i):
                import sympy
                return lambda: sympy.Integer(i)
            REFERENCE = builder(0)  # What analysis.py once did: a reference built when the module loads.
            FIXTURES = ()
            EDGE_CASES = ()
        """)
        done = import_without_references(package, ["called"], directory)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("sympy", done.stderr)

    def test_the_block_holds_for_mpmath_too(self):
        package, directory = self.make_package(top="""
            import mpmath
            FIXTURES = ()
            EDGE_CASES = ()
        """)
        done = import_without_references(package, ["top"], directory)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("mpmath", done.stderr)


if __name__ == "__main__":
    unittest.main()
