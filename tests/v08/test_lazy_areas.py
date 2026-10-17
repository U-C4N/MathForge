"""The area modules load lazily: attribute access imports one of them, ``import mathforge`` imports none."""

import ast
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import unittest

import mathforge
from mathforge import registry

# The tested package's root: src/ in a checkout, site-packages in the distribution harness (no src/).
SRC = Path(mathforge.__file__).resolve().parents[1]
IMPORT_CEILING = 0.30  # Seconds: the median `import mathforge` over fresh interpreters.
IMPORT_RUNS = 5
THREADS = 8

# For each area in turn, in a fresh interpreter: the attribute is the module, cached on the package, and
# loading it imports no other area.
FIRST_ACCESS = """
    import json, types
    import mathforge
    areas = json.loads(sys.argv[2])
    loaded = lambda: {area for area in areas if "mathforge." + area in sys.modules}
    problems = []
    for area in areas:
        before = loaded()
        module = getattr(mathforge, area)
        if not isinstance(module, types.ModuleType) or module.__name__ != "mathforge." + area:
            problems.append(f"{area}: resolved to {module!r}")
        if sys.modules.get("mathforge." + area) is not module:
            problems.append(f"{area}: is not the sys.modules entry")
        if vars(mathforge).get(area) is not module:
            problems.append(f"{area}: is not cached on the package")
        if loaded() - before != {area}:
            problems.append(f"{area}: loaded {sorted(loaded() - before)}")
    print(json.dumps(problems))
    """

# Threads released together by a barrier read the first area module and the registry tables of a cold
# interpreter. Each import takes a little longer, to widen the race window.
CONCURRENT = """
    import importlib, json, threading, time
    import mathforge
    from mathforge import registry
    count = int(sys.argv[2])
    cold = registry._tables.cache_info().currsize == 0 and "mathforge.ntheory" not in sys.modules
    real_import = importlib.import_module
    def slow_import(name, package=None):
        time.sleep(0.005)
        return real_import(name, package)
    importlib.import_module = slow_import
    barrier, seen, errors = threading.Barrier(count), [], []
    def read(index):
        try:
            barrier.wait(30)
            if index % 2:  # Both orders, so that each reader meets the other's half-finished work.
                table = registry.specs()
                module = mathforge.ntheory
            else:
                module = mathforge.ntheory
                table = registry.specs()
            seen.append((id(module), id(table)))
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
    threads = [threading.Thread(target=read, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    importlib.import_module = real_import
    print(json.dumps({"cold": cold, "errors": errors, "stuck": sum(thread.is_alive() for thread in threads),
                      "modules": [module for module, table in seen], "tables": [table for module, table in seen],
                      "module": id(sys.modules.get("mathforge.ntheory")), "table": id(registry.specs())}))
    """


def probe(code, *arguments):
    """Run code in a fresh isolated interpreter that imports the package under test; return its JSON output.

    -I makes the interpreter ignore PYTHONPATH, so the package root goes onto sys.path from the inside.
    """
    program = "import sys\nsys.path.insert(0, sys.argv[1])\n" + textwrap.dedent(code)
    output = subprocess.run([sys.executable, "-I", "-c", program, str(SRC), *arguments],
                            capture_output=True, text=True, timeout=120)
    if output.returncode:
        raise AssertionError(output.stderr)
    return json.loads(output.stdout)


class LazyAreaTests(unittest.TestCase):
    def test_import_does_not_load_area_modules(self):
        loaded = probe("""
            import json
            import mathforge
            areas = json.loads(sys.argv[2])
            print(json.dumps([area for area in areas if "mathforge." + area in sys.modules]))
            """, json.dumps(registry.AREAS))
        self.assertEqual(loaded, [])

    def test_area_attribute_loads_module(self):
        self.assertIs(mathforge.ntheory, importlib.import_module("mathforge.ntheory"))
        with self.assertRaises(AttributeError) as raised:
            mathforge.nope
        self.assertIn("'nope'", str(raised.exception))
        self.assertFalse(hasattr(mathforge, "nope"))
        # Whatever the earlier tests imported, the hook itself resolves every area to its module.
        for area in registry.AREAS:
            with self.subTest(area=area):
                self.assertIs(mathforge.__getattr__(area), importlib.import_module("mathforge." + area))

    def test_first_access_loads_exactly_that_area(self):
        self.assertEqual(probe(FIRST_ACCESS, json.dumps(registry.AREAS)), [])

    def test_names_that_are_not_areas_raise_attribute_error(self):
        # The names include real submodules: the hook must not import whatever it is asked for.
        for name in ("nope", "statistics", "registry", "NTHEORY", "ntheory.py", ""):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError) as raised:
                    mathforge.__getattr__(name)
                self.assertEqual(str(raised.exception), f"module 'mathforge' has no attribute {name!r}")

    def test_dir_lists_areas(self):
        listing = dir(mathforge)
        self.assertLessEqual(set(registry.AREAS), set(listing))
        self.assertEqual(listing, sorted(set(listing)))
        self.assertLessEqual({"solve", "Rational", "operations", "__version__"}, set(listing))
        self.assertIsInstance(mathforge.__dir__(), list)
        # Listing the areas must not import them: completion would load all 18 modules.
        cold = probe("""
            import json
            import mathforge
            areas = json.loads(sys.argv[2])
            listing = dir(mathforge)
            print(json.dumps({"missing": [area for area in areas if area not in listing],
                              "loaded": [area for area in areas if "mathforge." + area in sys.modules]}))
            """, json.dumps(registry.AREAS))
        self.assertEqual(cold, {"missing": [], "loaded": []})

    def test_type_checking_block_names_every_area(self):
        # Type checkers resolve mathforge.<area> from this hand-written block, so it must not drift.
        tree = ast.parse(Path(mathforge.__file__).read_text(encoding="utf-8"))
        named = [alias.asname or alias.name
                 for block in tree.body
                 if isinstance(block, ast.If) and isinstance(block.test, ast.Name) and block.test.id == "TYPE_CHECKING"
                 for node in block.body
                 if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
                 for alias in node.names]
        self.assertEqual(sorted(named), sorted(registry.AREAS))

    def test_concurrent_first_access(self):
        report = probe(CONCURRENT, str(THREADS))
        self.assertTrue(report["cold"], "the registry or an area was loaded before the threads started")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stuck"], 0)
        self.assertEqual(report["modules"], [report["module"]] * THREADS)
        self.assertEqual(report["tables"], [report["table"]] * THREADS)

    # No docstring: unittest would print it on its own line, between the test's name and the measurement.
    def test_import_time_ceiling(self):
        timings = [probe("""
            import json, time
            start = time.perf_counter()
            import mathforge
            print(json.dumps(time.perf_counter() - start))
            """) for _ in range(IMPORT_RUNS)]
        median = statistics.median(timings)
        print(f"import mathforge: median {median:.3f} s over {IMPORT_RUNS} fresh interpreters "
              f"(ceiling {IMPORT_CEILING:.2f} s)", file=sys.stderr)
        self.assertLessEqual(median, IMPORT_CEILING, timings)


if __name__ == "__main__":
    unittest.main()
