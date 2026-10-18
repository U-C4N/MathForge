"""The coverage-gate reference examples: SymPy API resolution, the sandboxed example runner, the harvest and its files.

SymPy 1.14 is a development dependency (installed in .venv-mcp together with its test modules): every test that needs it
skips when it is absent, as in CI's core job, while the file, curation and runner-protocol tests, which need no SymPy,
always run. The tests build synthetic rows, guide pages, test modules, inventories and example files in temporary
directories; none reads the local docs/ inventory, the curation file or the downloaded SymPy guide sources.
"""

import contextlib
import dataclasses
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
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
sys.path.insert(0, str(REPOSITORY / "tools"))  # The coverage gate is tooling beside the scripts, not a package.

from coverage_gate import examples as harvesting  # noqa: E402
from coverage_gate import inventory as model  # noqa: E402

SYMPY = importlib.util.find_spec("sympy") is not None
NO_SYMPY = "SymPy 1.14, a development dependency, is not installed"
CONTINUED_FRACTIONS = ("sympy.continued_fraction", "sympy.continued_fraction_convergents",
                       "sympy.continued_fraction_periodic", "sympy.continued_fraction_reduce")
PERFECT_NUMBERS = ("sympy.ntheory.is_perfect", "sympy.ntheory.is_abundant", "sympy.ntheory.is_deficient",
                   "sympy.ntheory.is_amicable", "sympy.ntheory.abundance")


def make_row(row_id, *paths, subtopic="Synthetic subtopic", area="Number theory"):
    return model.Row(id=row_id, area=area, subtopic=subtopic, sympy_apis=", ".join(paths), sympy_api_paths=paths,
                     v07_status="no", tier="T1", note="", plan_group=None, wave=None, family=None, missing="",
                     sympy_probe=None, mathforge_operations=(), status="no")


def example(api, n, *, row="27.14", source=None, code=None, output=None):
    """A synthetic candidate of the docstring of sympy.<api>, at line n."""
    return harvesting.Example("", row, source or f"docstring:sympy.{api}:{n}", "", code or f"{api}({n})",
                              str(n) if output is None else output, f"Integer({n})", "exact")


def lines_of(text, statement):
    """The 1-based lines of the doctest prompts that hold statement."""
    return [number for number, line in enumerate(text.splitlines(), 1) if line == f">>> {statement}"]


def line_of(text, statement):
    return lines_of(text, statement)[0]


def line_containing(text, fragment):
    return next(number for number, line in enumerate(text.splitlines(), 1) if fragment in line)


def docstring_line(path, statement):
    """The line of statement in the docstring of the API path, as harvest numbers it."""
    return line_of(inspect.getdoc(harvesting.resolve_api(path)), statement)


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)


class RunnerTestCase(TemporaryDirectoryTestCase):
    """Shares one worker per hash seed across a class: starting SymPy costs a second or two."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not SYMPY:
            raise unittest.SkipTest(NO_SYMPY)
        cls.first = harvesting.ExampleRunner(hash_seed=0)
        cls.addClassCleanup(cls.first.close)
        cls.second = harvesting.ExampleRunner(hash_seed=1)
        cls.addClassCleanup(cls.second.close)

    def pid(self, runner):
        """The process id of the runner's current worker (a new worker gives a new id)."""
        return runner.run("import os", "os.getpid()")[0]


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        if not SYMPY:
            self.skipTest(NO_SYMPY)

    def test_resolve_api(self):
        import sympy
        self.assertIs(harvesting.resolve_api("sympy.factorint"), sympy.factorint)
        self.assertIs(harvesting.resolve_api("Matrix.rref"), sympy.Matrix.rref)
        self.assertIs(harvesting.resolve_api("Poly.primitive"), sympy.Poly.primitive)
        self.assertIs(harvesting.resolve_api("S.Half"), sympy.S.Half)
        # Review Focus 3: the resolver never guesses; the message names the path and the reason.
        for path, reason in (("rref", "not qualified"), ("sympy.no_such_name", "no attribute 'no_such_name'")):
            with self.subTest(path), self.assertRaises(LookupError) as raised:
                harvesting.resolve_api(path)
            self.assertTrue(str(raised.exception).startswith(f"{path}: "), raised.exception)
            self.assertIn(reason, str(raised.exception))
        with self.assertRaises(LookupError) as raised:
            harvesting.resolve_api("rref")
        self.assertIn("Matrix.rref", str(raised.exception))  # A method name gets the qualified forms that exist.

    def test_resolve_api_prefers_public_objects_and_refuses_modules(self):
        import sympy
        # Each package re-exports the public function under its own subpackage's or module's name.
        for path, value in (("sympy.simplify", sympy.simplify), ("sympy.series", sympy.series),
                            ("sympy.ntheory.continued_fraction", sympy.continued_fraction)):
            with self.subTest(path):
                self.assertIs(harvesting.resolve_api(path), value)
        module = importlib.import_module("sympy.ntheory.continued_fraction")
        self.assertIs(harvesting.resolve_api("sympy.ntheory.continued_fraction.continued_fraction"),
                      module.continued_fraction)
        cases = (("sympy.simplify.simplify", "names the module sympy.simplify.simplify"),
                 ("sympy.ntheory", "names the module sympy.ntheory"),
                 ("ntheory.factorint", "is a module"),
                 ("Matrix.no_such_method", "no attribute 'no_such_method'"),
                 ("sympy.factorint.", "not a dotted name"),
                 ("", "not a dotted name"))
        for path, reason in cases:
            with self.subTest(path), self.assertRaises(LookupError) as raised:
                harvesting.resolve_api(path)
            self.assertIn(reason, str(raised.exception))
        row = make_row("27.25", "sympy.continued_fraction", "rref", "sympy.simplify.simplify")
        unresolved = harvesting.unresolved_apis(row)
        self.assertEqual([path for path, _ in unresolved], ["rref", "sympy.simplify.simplify"])
        self.assertTrue(unresolved[1][1].startswith("names the module"), unresolved[1][1])
        self.assertEqual(harvesting.unresolved_apis(make_row("27.25")), [("", "the row has no sympy_api_paths")])

    def test_resolve_api_refuses_ambiguous_paths(self):
        import sympy
        # Two different objects behind one path: sympy._probe is an object with an attribute f, while the import
        # system has a module sympy._probe with another f.
        module = types.ModuleType("sympy._mathforge_probe")
        module.f = lambda: "imported"
        with mock.patch.dict(sys.modules, {"sympy._mathforge_probe": module}), \
                mock.patch.object(sympy, "_mathforge_probe", types.SimpleNamespace(f=lambda: "attribute"),
                                  create=True), self.assertRaises(LookupError) as raised:
            harvesting.resolve_api("sympy._mathforge_probe.f")
        self.assertIn("ambiguous", str(raised.exception))

    def test_only_sympy_1_14_is_accepted(self):
        import sympy
        with mock.patch.object(sympy, "__version__", "1.13.3"):
            for call in (harvesting.require_sympy, lambda: harvesting.resolve_api("sympy.factorint")):
                with self.assertRaises(RuntimeError) as raised:
                    call()
                self.assertIn("SymPy 1.13.3 is installed; the coverage gate records SymPy 1.14", str(raised.exception))
        self.assertTrue(harvesting.require_sympy().startswith("1.14."))


class RunnerTests(RunnerTestCase):
    def test_runner_records_exact_values(self):
        self.assertEqual(self.first.run("x = symbols('x')", "factor(x**2 - 1)"),
                         ("(x - 1)*(x + 1)", "Mul(Add(Symbol('x'), Integer(-1)), Add(Symbol('x'), Integer(1)))"))
        self.assertEqual(self.first.run("from sympy.abc import x", "Poly(x**2 + 1).degree()"), ("2", "2"))

    def test_runner_timeout_restarts_worker(self):
        # Review Focus 2: a hung example costs its timeout, the worker is replaced and the next example runs.
        with harvesting.ExampleRunner(hash_seed=0, timeout=1) as runner:
            started = time.monotonic()
            self.assertEqual(runner.run("import time", "time.sleep(5)"), "timeout")
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertEqual(runner.run("", "Integer(2) + 2"), ("4", "Integer(4)"))

    def test_runner_classifies_failures_and_survives_crashes(self):
        cases = (
            ("", "1/0", "raises"),
            ("", "undefined_name", "raises"),
            ("", "pprint(Integer(2))", "printing"),               # It printed; its value is None.
            ("", "print('shown') or Integer(3)", "printing"),     # A value, but the example also prints.
            ("", "input()", "raises"),                            # Examples cannot read the protocol channel.
            ("import surely_missing_module_for_mathforge", "1", "optional-dependency"),
            ("", "lambda: 0", "unstable"),                        # Its str() carries a memory address.
            ("import os", "os._exit(3)", "raises"),               # The worker dies; the runner starts a new one.
        )
        for setup, code, reason in cases:
            with self.subTest(code):
                self.assertEqual(self.first.run(setup, code), reason)
        # Bytes written to file descriptor 1 cannot corrupt the protocol either.
        self.assertEqual(self.first.run("import os", "os.write(1, b'{}\\n')"), ("3", "3"))
        self.assertEqual(self.first.run("", "Integer(2) + 2"), ("4", "Integer(4)"))
        # An example that leaves the worker unable to reset its state does not take the next example down with it.
        self.assertEqual(self.first.run("import mpmath", "setattr(mpmath, 'mp', None) or 1"), ("1", "1"))
        self.assertEqual(self.first.run("", "Integer(3) + 2"), ("5", "Integer(5)"))
        with self.assertRaises(TypeError):
            self.first.run("", None)

    def test_runner_records_none_and_derives_kind_from_the_value(self):
        # None is an answer (three-valued logic), not printing.
        none = self.first.run("x = Symbol('x')", "ask(Q.positive(x))")
        self.assertEqual((none, none.kind), (("None", "None"), "exact"))
        for code, kind in (("nan", "exact"), ("oo", "exact"), ("zoo", "exact"), ("Rational(1, 2)", "exact"),
                           ("Float('0.5')", "numeric"), ("0.5", "numeric"), ("[Integer(1), 0.5]", "numeric"),
                           ("{Integer(1): 2.5}", "numeric"), ("2 + 3j", "numeric"), ("sqrt(2).n(3)", "numeric"),
                           ("Matrix([[Float('0.5'), 1]])", "numeric"), ("Poly(Symbol('x') + 1.0)", "numeric"),
                           ("__import__('mpmath').mpf(1)", "numeric"), ("[nan, oo]", "exact")):
            with self.subTest(code):
                result = self.first.run("", code)
                self.assertNotIsInstance(result, str, result)
                self.assertEqual(result.kind, kind)

    def test_runner_reports_values_that_depend_on_the_random_generators(self):
        hidden = "getattr(__import__('ran' + 'dom'), 'rand' + 'int')(1, 10**9)"  # No random name to see.
        for setup, code in (("import random", "random.randint(1, 10**9)"), ("", hidden),
                            ("from sympy.polys.galoistools import gf_random", "gf_random(4, 7, ZZ)")):
            with self.subTest(code):
                self.assertEqual(self.first.run(setup, code), "random")
                drawn = self.first.run(setup, code, random_allowed=True)  # A row about randomness keeps it.
                self.assertNotIsInstance(drawn, str)
                self.assertEqual(self.first.run(setup, code, random_allowed=True), drawn)
                self.assertEqual(self.second.run(setup, code, random_allowed=True), drawn)
        # Drawing from a generator is fine when the value does not depend on it.
        self.assertEqual(self.first.run("import random", "(random.random(), 5)[1]"), ("5", "5"))

    def test_runner_replaces_a_worker_whose_namespace_an_example_changed(self):
        cases = (("import sympy", "setattr(sympy, 'factorint', lambda n: 'patched') or 1", "factorint(12)",
                  ("{2: 2, 3: 1}", "{2: 2, 3: 1}")),
                 ("import builtins", "setattr(builtins, 'abs', lambda n: 0) or 1", "abs(-3)", ("3", "3")),
                 ("from sympy import Predicate", "setattr(Q, 'P', Predicate('P')) or 1", "hasattr(Q, 'P')",
                  ("False", "False")))
        for setup, change, check, expected in cases:
            with self.subTest(change):
                before = self.pid(self.first)
                self.assertEqual(self.first.run(setup, change), ("1", "1"))
                self.assertEqual(self.first.run("", check), expected)
                self.assertNotEqual(self.pid(self.first), before)
        # Importing a submodule, or predicates memoizing themselves on Q at the first ask(), changes nothing the next
        # example sees: a fresh worker (no earlier ask, sympy.categories not imported yet) keeps going.
        with harvesting.ExampleRunner(hash_seed=0) as fresh:
            before = self.pid(fresh)
            self.assertEqual(fresh.run("import sympy.categories", "Integer(1)"), ("1", "Integer(1)"))
            self.assertEqual(fresh.run("x = Symbol('x')", "ask(Q.hermitian(x))"), ("None", "None"))
            self.assertEqual(self.pid(fresh), before)

    def test_runner_results_do_not_depend_on_history_or_process(self):
        setup, code = "x = Symbol('x')", "solveset(sin(x), x)"
        output, representation = self.first.run(setup, code)
        self.assertIn("ImageSet(Lambda", representation)
        self.assertIn("dummy_index=0", representation)  # Dummy indices are renumbered: the base is random per process.
        self.assertNotRegex(representation, r"dummy_index=\d{2,}")
        with harvesting.ExampleRunner(hash_seed=0) as other:
            self.assertEqual(other.run(setup, code), (output, representation))
        # Global state that one example changes is reset before the next one.
        self.first.run("import mpmath\nmpmath.mp.dps = 50", "Integer(1)")
        self.assertEqual(self.first.run("import mpmath", "mpmath.mpf(1)/3")[0], "0.333333333333333")
        self.assertEqual(self.first.run("import random", "random.random()", random_allowed=True),
                         self.first.run("import random", "random.random()", random_allowed=True))
        # The hash seed reaches the worker.
        self.assertEqual(self.first.run("", "hash('mathforge')"), self.first.run("", "hash('mathforge')"))
        self.assertNotEqual(self.first.run("", "hash('mathforge')"), self.second.run("", "hash('mathforge')"))

    def test_runner_environment_holds_only_the_hash_seed(self):
        variables = {"SYMPY_USE_CACHE": "no", "SYMPY_GROUND_TYPES": "python", "MPMATH_NOGMPY": "1",
                     "PYTHONOPTIMIZE": "2"}
        with mock.patch.dict(os.environ, variables), harvesting.ExampleRunner(hash_seed=7) as runner:
            seen = runner.run("import os", "sorted(name for name in os.environ "
                                           "if name.upper().startswith(('PYTHON', 'SYMPY', 'MPMATH')))")
        self.assertEqual(seen[0], "['PYTHONHASHSEED']")


class ProtocolTests(unittest.TestCase):
    """The parent side of the runner protocol, against a fake worker that needs no SymPy."""

    FAKE_WORKER = textwrap.dedent("""\
        import json, os, sys
        print(json.dumps({"ready": True, "sympy": "1.14.0"}), flush=True)
        for line in sys.stdin:
            code = json.loads(line)["code"]
            if code == "garbage":
                print("not json", flush=True)
            elif code == "keys":
                print(json.dumps({"output": "1", "srepr": "1"}), flush=True)
            elif code == "kind":
                print(json.dumps({"output": "1", "srepr": "1", "kind": "approximate"}), flush=True)
            elif code == "reason":
                print(json.dumps({"error": "made-up"}), flush=True)
            elif code == "die":
                sys.exit(3)
            elif code == "changed":
                print(json.dumps({"output": "1", "srepr": "1", "kind": "exact", "restart": True}), flush=True)
            elif code == "pid":
                print(json.dumps({"output": str(os.getpid()), "srepr": "pid", "kind": "exact"}), flush=True)
            else:
                print(json.dumps({"output": code, "srepr": code, "kind": "exact"}), flush=True)
        """)

    def test_runner_never_trusts_a_worker_that_breaks_the_protocol(self):
        with mock.patch.object(harvesting, "_WORKER", self.FAKE_WORKER), \
                harvesting.ExampleRunner(hash_seed=0) as runner:
            self.assertEqual(runner.sympy_version, "1.14.0")
            for code in ("garbage", "keys", "kind", "reason", "die"):
                with self.subTest(code):
                    before = runner.run("", "pid")
                    self.assertEqual(runner.run("", "pid"), before)  # A good answer keeps the worker.
                    self.assertEqual(runner.run("", code), "raises")
                    self.assertNotEqual(runner.run("", "pid"), before)  # A bad one replaces it.
            # A worker that reports a changed namespace answers, and is then replaced.
            before = runner.run("", "pid")
            changed = runner.run("", "changed")
            self.assertEqual((changed, changed.kind), (("1", "1"), "exact"))
            self.assertNotEqual(runner.run("", "pid"), before)
        with self.assertRaises(RuntimeError):
            runner.run("", "pid")  # Closed.

    def test_runner_reports_a_worker_that_does_not_start(self):
        for script, detail in (("import sys; sys.exit(1)", "did not start (no answer)"),
                               ('import json; print(json.dumps({"ready": False, "error": "ImportError: sympy"}))',
                                "did not start (ImportError: sympy)"),
                               ('import json; print(json.dumps({"ready": True, "sympy": "1.13.3"}))',
                                "runs SymPy 1.13.3; the coverage gate records SymPy 1.14")):
            with self.subTest(detail), mock.patch.object(harvesting, "_WORKER", script):
                with self.assertRaises(RuntimeError) as raised:
                    harvesting.ExampleRunner(hash_seed=0)
                self.assertIn(detail, str(raised.exception))
        for arguments in ({"hash_seed": -1}, {"hash_seed": True}, {"hash_seed": 0, "timeout": 0}):
            with self.subTest(arguments), self.assertRaises(ValueError):
                harvesting.ExampleRunner(**arguments)


class HarvestTests(RunnerTestCase):
    def guide(self, name, text):
        root = self.directory / "doc-src"
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return root, path.read_text(encoding="utf-8")

    def guide_items(self, items):
        return [item for item in items if item.source.startswith("guide:")]

    def test_unstable_outputs_are_excluded(self):
        # Review Focus 1: an output that depends on the hash seed is excluded before selection.
        row = make_row("27.14", "sympy.factorint")
        self.assertEqual(harvesting.examine(row, "guide:synthetic.rst:1", "", "hash('mathforge')", self.first,
                                            self.second),
                         harvesting.Excluded("27.14", "guide:synthetic.rst:1", "hash('mathforge')", "unstable"))
        stable = harvesting.examine(row, "guide:synthetic.rst:2", "", "factorint(12)", self.first, self.second)
        self.assertEqual((stable.output, stable.srepr, stable.kind), ("{2: 2, 3: 1}", "{2: 2, 3: 1}", "exact"))
        # The same holds inside the harvest: the unstable guide example never reaches select().
        doc_src, text = self.guide("synthetic.rst", """\
            >>> from sympy import factorint
            >>> factorint(12)
            {2: 2, 3: 1}
            >>> (factorint(12), hash('mathforge'))
            ({2: 2, 3: 1}, 5286171517341885545)
            """)
        candidates, excluded = harvesting.harvest(row, self.first, doc_src=doc_src, check_runner=self.second)
        unstable = "(factorint(12), hash('mathforge'))"
        source = f"guide:synthetic.rst:{line_of(text, unstable)}"
        self.assertIn(harvesting.Excluded("27.14", source, unstable, "unstable"), excluded)
        self.assertNotIn(source, [candidate.source for candidate in candidates])
        self.assertIn(f"guide:synthetic.rst:{line_of(text, 'factorint(12)')}",
                      [candidate.source for candidate in candidates])

    def test_selection_rules(self):
        # The harvest excludes each unusable candidate with its reason and keeps one of each code and record.
        row = make_row("27.14", "sympy.factorint")
        doc_src, text = self.guide("guides/synthetic.rst", """\
            Synthetic guide page
            ====================

            >>> from sympy import factorint, pprint, plot, Symbol
            >>> from sympy.abc import x
            >>> factorint(360)
            {2: 3, 3: 2, 5: 1}
            >>> pprint(factorint(360))
            {2: 3, 3: 2, 5: 1}
            >>> plot(x*factorint(360)[2], show=False)
            Plot object containing:
            >>> import numpy
            >>> numpy.array(sorted(factorint(360)))
            array([2, 3, 5])
            >>> factorint(Symbol('y'))
            Traceback (most recent call last):
            ...
            ValueError: y is not an integer
            >>> factorint(-1.5)
            {-1: 1}
            >>> factorint(360)
            {2: 3, 3: 2, 5: 1}
            >>> from random import randint
            >>> factorint(randint(2, 9))
            {2: 1}
            >>> for p in factorint(360):
            ...     print(p)
            2
            3
            5
            """)
        candidates, excluded = harvesting.harvest(row, self.first, doc_src=doc_src, check_runner=self.second)
        guide = {(item.code, item.reason) for item in self.guide_items(excluded)}
        self.assertEqual(guide, {("pprint(factorint(360))", "printing"),
                                 ("plot(x*factorint(360)[2], show=False)", "plotting"),
                                 ("numpy.array(sorted(factorint(360)))", "optional-dependency"),
                                 ("factorint(Symbol('y'))", "raises"),
                                 ("factorint(-1.5)", "raises"),
                                 ("factorint(360)", "duplicate"),
                                 ("factorint(randint(2, 9))", "random"),
                                 ("for p in factorint(360):\n    print(p)", "printing")})
        first, second = lines_of(text, "factorint(360)")
        self.assertIn(harvesting.Excluded("27.14", f"guide:guides/synthetic.rst:{second}", "factorint(360)",
                                          "duplicate"), excluded)
        self.assertEqual([(item.source, item.setup, item.code, item.output) for item in self.guide_items(candidates)],
                         [(f"guide:guides/synthetic.rst:{first}", "from sympy import factorint, pprint, plot, Symbol",
                           "factorint(360)", "{2: 3, 3: 2, 5: 1}")])  # The setup keeps only what the code needs.
        self.assertTrue(all(item.source.startswith("docstring:sympy.factorint:") for item in candidates[:-1]))

        # select(): at most 20, taken in turn from the row's APIs; fewer than 10 are kept as they are (a short row).
        apis = ("sympy.factorint", "sympy.divisors", "sympy.totient")
        a = [example("factorint", n) for n in range(1, 16)]
        b = [example("divisors", n) for n in range(1, 7)]
        c = [example("totient", n) for n in range(1, 5)]
        self.assertEqual(len(a + b + c), 25)
        self.assertEqual(harvesting.select(a + b + c, apis=apis), a[:10] + b + c)
        self.assertEqual(harvesting.select(a + b + c), a[:10] + b + c)  # Docstring sources name their API.
        self.assertEqual(harvesting.select(a + b + c, low=3, high=5, apis=apis), a[:2] + b[:2] + c[:1])
        seven = a[:4] + b[:3]
        self.assertEqual(harvesting.select(seven, apis=apis), seven)
        self.assertLess(len(harvesting.select(seven, apis=apis)), 10)  # short
        # One of each code and record (code formatting aside), the first in candidate order.
        twin = dataclasses.replace(a[0], source="docstring:sympy.divisors:9", code="factorint( 1 )")
        self.assertEqual(harvesting.select([a[0], twin, b[1]], apis=apis), [a[0], b[1]])
        other = dataclasses.replace(twin, output="2", srepr="Integer(2)")  # Same code, another record.
        self.assertEqual(harvesting.select([a[0], other, b[1]], apis=apis), [a[0], other, b[1]])
        # Test-module asserts only top a set up to 10, again in turn across the APIs.
        tests = [example("factorint", n, source=f"tests:sympy.ntheory.tests.test_factor_::test_{n}:{n}")
                 for n in range(20, 25)]
        self.assertEqual(harvesting.select(a[:8] + tests, apis=apis), a[:8] + tests[:2])
        self.assertEqual(harvesting.select(a[:12] + tests, apis=apis), a[:12])

    def test_duplicates_are_judged_by_their_records(self):
        # The same code with another setup is another example; with the same record it is a duplicate.
        row = make_row("27.14", "sympy.factorint")
        doc_src, text = self.guide("guides/repeats.rst", """\
            >>> from sympy import factorint
            >>> n = 12
            >>> factorint(n)
            {2: 2, 3: 1}
            >>> n = 18
            >>> factorint(n)
            {2: 1, 3: 2}
            >>> n = 12
            >>> factorint(n)
            {2: 2, 3: 1}
            """)
        candidates, excluded = harvesting.harvest(row, self.first, doc_src=doc_src, check_runner=self.second)
        twelve, eighteen, again = (f"guide:guides/repeats.rst:{line}" for line in lines_of(text, "factorint(n)"))
        self.assertEqual([(item.source, item.setup, item.output) for item in self.guide_items(candidates)],
                         [(twelve, "from sympy import factorint\nn = 12", "{2: 2, 3: 1}"),
                          (eighteen, "from sympy import factorint\nn = 18", "{2: 1, 3: 2}")])
        self.assertEqual(self.guide_items(excluded), [harvesting.Excluded("27.14", again, "factorint(n)", "duplicate")])
        # SymPy's own expand docstring shows log(x**2*y).expand(log=True) twice, with and without positive symbols.
        candidates, _ = harvesting.harvest(make_row("16.01", "sympy.expand"), self.first, doc_src=None,
                                           check_runner=self.second)
        pair = [(item.source, item.output) for item in candidates if item.code == "log(x**2*y).expand(log=True)"]
        first, second = lines_of(inspect.getdoc(harvesting.resolve_api("sympy.expand")),
                                 "log(x**2*y).expand(log=True)")
        self.assertEqual(pair, [(f"docstring:sympy.expand:{first}", "log(x**2*y)"),
                                (f"docstring:sympy.expand:{second}", "2*log(x) + log(y)")])

    def test_unstable_examples_block_nothing(self):
        # Two candidates share code and seed-0 record; the first only by chance of the hash seed. Once it is
        # excluded as unstable, the second, stable one takes its place instead of staying a duplicate.
        seed_zero_hash = self.first.run("", "hash('mathforge')")[0]
        row = make_row("27.14", "sympy.factorint")
        doc_src, text = self.guide("guides/chance.rst", f"""\
            >>> from sympy import factorint
            >>> n = 12 if hash('mathforge') == {seed_zero_hash} else 18
            >>> factorint(n)
            {{2: 2, 3: 1}}
            >>> n = 12
            >>> factorint(n)
            {{2: 2, 3: 1}}
            """)
        candidates, excluded = harvesting.harvest(row, self.first, doc_src=doc_src, check_runner=self.second)
        chance, steady = (f"guide:guides/chance.rst:{line}" for line in lines_of(text, "factorint(n)"))
        self.assertEqual([item.source for item in self.guide_items(candidates)], [steady])
        self.assertEqual(self.guide_items(excluded), [harvesting.Excluded("27.14", chance, "factorint(n)", "unstable")])
        self.assertIn(steady, [item.source for item in harvesting.select(candidates, apis=row.sympy_api_paths)])

    def test_static_rules_name_only_plotting_and_catch_random_generators(self):
        row = make_row("31.08", "sympy.Segment")
        kept = (("from sympy import Point, Segment", "Segment(Point(0, 0), Point(1, 1)).plot_interval()", "[t, 0, 1]"),
                ("plot_data = [1, 2, 3]", "len(plot_data)", "3"),
                ("from sympy.stats import Normal, random_symbols\nX = Normal('X', 0, 1)", "random_symbols(X + 1)",
                 "[X]"))
        for setup, code, output in kept:
            with self.subTest(code):
                result = harvesting.examine(row, "guide:g.rst:1", setup, code, self.first, self.second)
                self.assertIsInstance(result, harvesting.Example, result)
                self.assertEqual(result.output, output)
        excluded = (("from sympy.abc import x", "plot(x**2, show=False)", "plotting"),
                    ("from sympy.plotting import plot3d\nfrom sympy.abc import x, y", "plot3d(x*y, show=False)",
                     "plotting"),
                    ("from sympy.polys.galoistools import gf_random", "gf_random(4, 7, ZZ)", "random"),
                    ("from sympy.polys.densebasic import dup_random", "dup_random(3, -5, 5, ZZ)", "random"),
                    ("", "getattr(__import__('ran' + 'dom'), 'rand' + 'int')(1, 10**9)", "random"))  # By running.
        for setup, code, reason in excluded:
            with self.subTest(code):
                self.assertEqual(harvesting.examine(row, "guide:g.rst:1", setup, code, self.first, self.second),
                                 harvesting.Excluded("31.08", "guide:g.rst:1", code, reason))
        # Where randomness is the row's subject, a random example is kept with the draw of the fixed seed.
        drawn = harvesting.examine(make_row("27.17", "sympy.randprime"), "guide:g.rst:1", "", "randprime(10, 100)",
                                   self.first, self.second)
        self.assertIsInstance(drawn, harvesting.Example, drawn)

    def test_decorated_registrations_stay_in_the_setup(self):
        # SymPy's own examples register handlers with decorators that the code reaches without naming them.
        for path, code, output, decorator in (
                ("sympy.Predicate", "ask(Q.sexyprime(5, 11))", "True", "@Q.sexyprime.register(Integer, Integer)"),
                ("sympy.core.relational.is_ge", "is_ge(b, a)", "True", "@dispatch(MyExpr, MyExpr)"),
                ("sympy.tensor.functions.shape", "shape(NewExpr(M))", "(2, 1)", "@shape.register(NewExpr)")):
            with self.subTest(path):
                candidates, _ = harvesting.harvest(make_row("02.08", path), self.first, doc_src=None,
                                                   check_runner=self.second)
                found = [item for item in candidates if item.code == code]
                self.assertEqual([(item.source, item.output) for item in found],
                                 [(f"docstring:{path}:{docstring_line(path, code)}", output)])
                self.assertIn(f"{decorator}\ndef ", found[0].setup)

    def test_guide_pages_in_markdown_and_with_malformed_doctests(self):
        row = make_row("27.14", "sympy.factorint")
        doc_src, markdown = self.guide("guides/solving.md", """\
            # Factoring

            ```py
            >>> from sympy import factorint
            >>> n = factorint(48)
            ```

            ```py
            >>> factorint(n[2] * 7)
            {2: 2, 7: 1}
            ```
            """)
        _, broken = self.guide("modules/broken.rst", """\
            A malformed example spoils doctest's parse of the whole page:

              >>> y = (1,
             ... 2)

            >>> factorint(6)
            {2: 1, 3: 1}
            """)
        self.guide("modules/ignored.py", ">>> factorint(10)\n{2: 1, 5: 1}\n")  # Not a guide page.
        candidates, excluded = harvesting.harvest(row, self.first, doc_src=doc_src, check_runner=self.second)
        # The closing fence after `n = ...` is not output, so the assignment is setup, not a printing example.
        self.assertEqual([(item.source, item.setup, item.code, item.output) for item in self.guide_items(candidates)],
                         [(f"guide:guides/solving.md:{line_of(markdown, 'factorint(n[2] * 7)')}",
                           "from sympy import factorint\nn = factorint(48)", "factorint(n[2] * 7)", "{2: 2, 7: 1}"),
                          (f"guide:modules/broken.rst:{line_of(broken, 'factorint(6)')}", "", "factorint(6)",
                           "{2: 1, 3: 1}")])
        self.assertEqual(self.guide_items(excluded), [])

    def test_harvest_docstring_examples(self):
        row = make_row("27.25", *CONTINUED_FRACTIONS, subtopic="Continued fractions")
        candidates, excluded = harvesting.harvest(row, self.first, doc_src=None, check_runner=self.second)
        first = candidates[0]
        self.assertTrue(first.source.startswith("docstring:"))
        code = "continued_fraction((1 + 2*sqrt(3))/5)"
        self.assertEqual((first.source, first.setup, first.code, first.output, first.kind),
                         (f"docstring:sympy.continued_fraction:{docstring_line('sympy.continued_fraction', code)}",
                          "from sympy.ntheory.continued_fraction import continued_fraction\nfrom sympy import sqrt",
                          code, "[0, 1, [8, 3, 34, 3]]", "exact"))
        self.assertEqual(first.srepr, "[Integer(0), Integer(1), [Integer(8), Integer(3), Integer(34), Integer(3)]]")
        self.assertEqual({re.sub(r":\d+$", "", item.source) for item in candidates},
                         {f"docstring:{path}" for path in row.sympy_api_paths})
        self.assertEqual(len({item.source for item in candidates}), len(candidates))  # Line-qualified, so unique.
        numeric = [item for item in candidates if item.kind == "numeric"]
        self.assertEqual([item.code for item in numeric],
                         ["continued_fraction_reduce([2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]).n(10)"])
        self.assertEqual(numeric[0].output, "2.718281835")
        # The loops that print the convergents are not single expressions.
        self.assertEqual([item.reason for item in excluded], ["printing", "printing"])
        self.assertTrue(all(item.source.startswith("docstring:sympy.continued_fraction_convergents:")
                            for item in excluded))
        selected = harvesting.select(candidates, apis=row.sympy_api_paths)
        self.assertEqual(len(selected), len(candidates))
        self.assertGreaterEqual(len(selected), 10)
        # Every record is reproducible from its setup and code alone, in another process.
        with harvesting.ExampleRunner(hash_seed=0) as other:
            for item in selected:
                self.assertEqual(other.run(item.setup, item.code), (item.output, item.srepr), item.code)

    def test_harvest_tops_up_short_sets_from_test_modules(self):
        package = self.directory / "sympy"
        tests = package / "ntheory" / "tests"
        tests.mkdir(parents=True)
        module = textwrap.dedent("""\
            from contextlib import nullcontext
            from sympy import Rational, sqrt
            from sympy.ntheory.continued_fraction import continued_fraction as cf


            with nullcontext():  # Like SymPy's deprecated-mixin classes: a definition inside a with block.
                def helper(n):
                    return Rational(n, 7)


            def test_values():
                assert cf(Rational(415, 93)) == [4, 2, 6, 7]
                t = sqrt(2)
                assert cf(t) == [1, [2]]
                assert cf(Rational(3, 7)) == [0, 2, 3]
                assert cf(helper(10)) == [1, 2, 3]
                assert cf(Rational(5, 1)) == [6]
                for n in range(3):
                    assert cf(n) == [n]
                assert cf(Rational(1, 3)) == [0, 3]


            def test_more():
                assert cf(Rational(7, 2)) == [3, 2]
                assert cf(Rational(4, 9)) == [0, 2, 4]
                assert cf(Rational(9, 4)) == [2, 4]
                assert cf(Rational(13, 5)) == [2, 1, 1, 2]
                assert cf(Rational(15, 4)) == [3, 1, 3]
            """)
        (tests / "test_synthetic_cf.py").write_text(module, encoding="utf-8")
        row = make_row("27.25", "sympy.continued_fraction", subtopic="Continued fractions")
        with mock.patch.object(harvesting, "_sympy_package_directory", return_value=package):
            candidates, excluded = harvesting.harvest(row, self.first, doc_src=None, check_runner=self.second)
        selected = harvesting.select(candidates, apis=row.sympy_api_paths)
        self.assertEqual(selected, candidates)
        self.assertEqual(len(selected), 10)  # One docstring example, then asserts up to ten.
        self.assertTrue(selected[0].source.startswith("docstring:sympy.continued_fraction:"))

        def source(function, code):
            line = line_containing(module, f"assert {code} ==")
            return f"tests:sympy.ntheory.tests.test_synthetic_cf::{function}:{line}"

        expected = [("test_values", "cf(Rational(415, 93))", "[4, 2, 6, 7]"),
                    ("test_values", "cf(t)", "[1, [2]]"),
                    ("test_values", "cf(Rational(3, 7))", "[0, 2, 3]"),
                    ("test_values", "cf(helper(10))", "[1, 2, 3]"),
                    ("test_values", "cf(Rational(1, 3))", "[0, 3]"),   # cf(5) == [6] is false: not a candidate.
                    ("test_more", "cf(Rational(7, 2))", "[3, 2]"),
                    ("test_more", "cf(Rational(4, 9))", "[0, 2, 4]"),
                    ("test_more", "cf(Rational(9, 4))", "[2, 4]"),
                    ("test_more", "cf(Rational(13, 5))", "[2, 1, 1, 2]")]
        self.assertEqual([(item.source, item.code, item.output) for item in selected[1:]],
                         [(source(function, code), code, output) for function, code, output in expected])
        imports = "from sympy.ntheory.continued_fraction import continued_fraction as cf"
        self.assertEqual(selected[2].setup, f"from sympy import Rational, sqrt\n{imports}\nt = sqrt(2)")
        # The loop before cf(Rational(1, 3)) only calls cf: it changes nothing the code needs and stays out.
        self.assertEqual(selected[5].setup, f"from sympy import Rational, sqrt\n{imports}")
        # The helper is defined inside a with block: the slicer still sees the binding and keeps the whole block.
        self.assertEqual(selected[4].setup, "from contextlib import nullcontext\nfrom sympy import Rational, sqrt\n"
                                            f"{imports}\nwith nullcontext():\n\n    def helper(n):\n"
                                            "        return Rational(n, 7)")
        self.assertEqual(excluded, [])


class CommandLineTests(RunnerTestCase):
    @staticmethod
    def harvest(arguments):
        """Run the command line in this process; its progress lines and messages are captured, not printed."""
        import harvest_examples
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            status = harvest_examples.main(arguments)
        return status, errors.getvalue()

    def test_cli_merges_rows_applies_curation_and_numbers_examples(self):
        rows = (make_row("01.01", "sympy.Integer", area="Core & numbers", subtopic="Exact integers"),
                make_row("27.14", *PERFECT_NUMBERS, subtopic="Perfect numbers"),
                make_row("27.25", *CONTINUED_FRACTIONS, subtopic="Continued fractions"))
        inventory = self.directory / "inventory.json"
        model.write_inventory(inventory, model.Inventory({"inventory": "synthetic"}, rows))
        kept = harvesting.Example("01.01/01", "01.01", "docstring:sympy.Integer:3", "", "Integer(2) + 2", "4",
                                  "Integer(4)", "exact")
        stale = dataclasses.replace(kept, id="27.25/01", row="27.25", source="docstring:sympy.continued_fraction:3",
                                    code="continued_fraction(Rational(1, 2))")
        out = self.directory / "examples.json"
        harvesting.write_examples(out, {"01.01": [kept], "27.25": [stale]})
        code = "continued_fraction_reduce([-2, 1, 9, 7, 1, 2])"
        dropped = (f"docstring:sympy.continued_fraction_reduce:"
                   f"{docstring_line('sympy.continued_fraction_reduce', code)}", code)
        curation = self.directory / "curation.json"
        curation.write_text(json.dumps({"27.25": {
            "drop": [{"source": dropped[0], "code": dropped[1], "reason": "duplicate"}],
            "add": [{"source": "guide:custom.rst:3", "setup": "", "code": "continued_fraction_reduce([1, 2])"}]}}),
            encoding="utf-8")
        report = self.directory / "reports" / "harvest.md"
        arguments = [str(inventory), "--out", str(out), "--rows", "27.25", "--report", str(report),
                     "--curation", str(curation)]

        self.assertEqual(self.harvest(arguments), (0, ""))
        loaded = harvesting.load_examples(out)
        self.assertEqual(loaded["01.01"], [kept])  # Rows outside the run stay as they were.
        harvested = loaded["27.25"]
        self.assertEqual([item.id for item in harvested], [f"27.25/{n:02d}" for n in range(1, len(harvested) + 1)])
        self.assertNotIn(dropped, [(item.source, item.code) for item in harvested])
        self.assertNotIn(stale.code, [item.code for item in harvested])
        self.assertEqual((harvested[-1].source, harvested[-1].code, harvested[-1].output, harvested[-1].srepr),
                         ("guide:custom.rst:3", "continued_fraction_reduce([1, 2])", "3/2", "Rational(3, 2)"))
        text = report.read_text(encoding="utf-8")
        for expected in ("27.25", "Continued fractions", dropped[0], "guide:custom.rst:3",
                         "not used (--doc-src not given)"):
            self.assertIn(expected, text)
        self.assertNotIn("01.01", text)

        # Two rows on two threads in another process with another hash seed: the same bytes. Nothing recorded depends
        # on caches of this process, on thread scheduling or on the harvester's own hash seed.
        both = ["--rows", "27.14,27.25", "--curation", str(curation)]
        here, there = self.directory / "here.json", self.directory / "there.json"
        self.assertEqual(self.harvest([str(inventory), "--out", str(here), "--report", str(self.directory / "here.md"),
                                       *both]), (0, ""))
        process = subprocess.run([sys.executable, str(REPOSITORY / "tools" / "harvest_examples.py"), str(inventory),
                                  "--out", str(there), "--report", str(self.directory / "there.md"), "--workers", "2",
                                  *both], env={**os.environ, "PYTHONHASHSEED": "4321"}, capture_output=True,
                                 text=True, timeout=900)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(here.read_bytes(), there.read_bytes())
        self.assertEqual(set(harvesting.load_examples(here)), {"27.14", "27.25"})

        # Curation mistakes stop the run before anything is written.
        written = out.read_bytes()
        for name, entry in (("unknown drop target", {"drop": [{"source": dropped[0], "code": "no_such_call()",
                                                               "reason": "duplicate"}]}),
                            ("unusable addition", {"add": [{"source": "guide:custom.rst:9", "setup": "",
                                                            "code": "pprint(continued_fraction([1]))"}]})):
            with self.subTest(name):
                curation.write_text(json.dumps({"27.25": entry}), encoding="utf-8")
                status, errors = self.harvest(arguments)
                self.assertEqual(status, 2)
                self.assertIn("nothing was written", errors)
                self.assertEqual(out.read_bytes(), written)
        status, errors = self.harvest([str(inventory), "--out", str(out), "--rows", "99.99", "--report", str(report),
                                       "--curation", str(curation)])
        self.assertEqual(status, 2)
        self.assertIn("99.99", errors)
        import sympy
        with mock.patch.object(sympy, "__version__", "1.13.3"):  # Another SymPy would record other outputs.
            status, errors = self.harvest(arguments)
        self.assertEqual(status, 2)
        self.assertIn("SymPy 1.13.3 is installed; the coverage gate records SymPy 1.14", errors)
        self.assertEqual(out.read_bytes(), written)


class FileTests(TemporaryDirectoryTestCase):
    def test_examples_round_trip(self):
        examples = {
            "27.25": [harvesting.Example("27.25/01", "27.25", "docstring:sympy.continued_fraction:7",
                                         "from sympy.ntheory.continued_fraction import continued_fraction\n"
                                         "from sympy import sqrt", "continued_fraction((1 + 2*sqrt(3))/5)",
                                         "[0, 1, [8, 3, 34, 3]]", "[Integer(0), Integer(1), [Integer(8)]]", "exact"),
                      harvesting.Example("27.25/02", "27.25", "guide:modules/ntheory.rst:12", "", "sqrt(2).n(3)",
                                         "1.41", "Float('1.41', precision=13)", "numeric")],
            "01.01": [harvesting.Example("01.01/01", "01.01",
                                         "tests:sympy.core.tests.test_numbers::test_Rational:40", "",
                                         "Rational(1, 2)", "1/2 — √", "Rational(1, 2)", "exact")],
        }
        path = self.directory / "examples.json"
        harvesting.write_examples(path, examples)
        written = path.read_bytes()
        loaded = harvesting.load_examples(path)
        self.assertEqual(loaded, examples)
        self.assertEqual(list(loaded), ["01.01", "27.25"])
        harvesting.write_examples(path, loaded)
        self.assertEqual(path.read_bytes(), written)  # Byte-stable.
        text = written.decode("utf-8")
        self.assertTrue(text.startswith('{\n "rows": {\n  "01.01": [\n   {\n    "id": "01.01/01",\n'), text[:80])
        self.assertTrue(text.endswith("}\n") and "\r" not in text and "1/2 — √" in text)
        fields = [field.name for field in dataclasses.fields(harvesting.Example)]
        self.assertEqual(list(json.loads(text)["rows"]["27.25"][0]), fields)

    def test_example_files_are_validated(self):
        good = {"id": "27.25/01", "row": "27.25", "source": "docstring:sympy.continued_fraction:7", "setup": "",
                "code": "continued_fraction(2)", "output": "[2]", "srepr": "[Integer(2)]", "kind": "exact"}
        cases = (
            ("not an object", []),
            ("no rows", {"examples": {}}),
            ("bad row id", {"rows": {"27.250": [good]}}),
            ("missing field", {"rows": {"27.25": [{k: v for k, v in good.items() if k != "srepr"}]}}),
            ("extra field", {"rows": {"27.25": [{**good, "short": "yes"}]}}),
            ("not a string", {"rows": {"27.25": [{**good, "output": 2}]}}),
            ("row mismatch", {"rows": {"27.25": [{**good, "row": "27.24"}]}}),
            ("bad id", {"rows": {"27.25": [{**good, "id": "27.24/01"}]}}),
            ("duplicate id", {"rows": {"27.25": [good, {**good, "code": "continued_fraction(3)"}]}}),
            ("bad kind", {"rows": {"27.25": [{**good, "kind": "approximate"}]}}),
        )
        path = self.directory / "examples.json"
        for name, document in cases:
            with self.subTest(name):
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ValueError) as raised:
                    harvesting.load_examples(path)
                self.assertIn(str(path), str(raised.exception))
        path.write_text(json.dumps({"rows": {"27.25": [good], "27.26": []}}), encoding="utf-8")
        self.assertEqual(harvesting.load_examples(path)["27.26"], [])
        with self.assertRaises(ValueError):
            harvesting.write_examples(path, {"27.25": [harvesting.Example(**{**good, "row": "27.24"})]})

    def test_curation_files_are_validated(self):
        drop = {"source": "docstring:sympy.factorint:5", "code": "factorint(12)", "reason": "not-the-subject"}
        add = {"source": "guide:modules/ntheory.rst:40", "setup": "from sympy.abc import x", "code": "factorint(12)"}
        cases = (
            ("not an object", []),
            ("bad row id", {"27.1": {"drop": [drop]}}),
            ("unknown key", {"27.14": {"replace": []}}),
            ("difficulty is never a reason", {"27.14": {"drop": [{**drop, "reason": "too-hard"}]}}),
            ("missing drop field", {"27.14": {"drop": [{"source": drop["source"], "code": drop["code"]}]}}),
            ("unknown source kind", {"27.14": {"add": [{**add, "source": "web:sympy.org"}]}}),
            ("code is not one expression", {"27.14": {"add": [{**add, "code": "y = factorint(12)"}]}}),
            ("setup does not parse", {"27.14": {"add": [{**add, "setup": "from sympy import ("}]}}),
        )
        path = self.directory / "curation.json"
        for name, document in cases:
            with self.subTest(name):
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(harvesting.CurationError) as raised:
                    harvesting.load_curation(path)
                self.assertIn(str(path), str(raised.exception))
        path.write_text(json.dumps({"27.14": {"drop": [drop], "add": [add]}}), encoding="utf-8")
        self.assertEqual(harvesting.load_curation(path), {"27.14": {"drop": [drop], "add": [add]}})
        self.assertEqual(set(harvesting.DROP_REASONS),
                         {"not-the-subject", "duplicate"} | set(harvesting.EXCLUSION_REASONS))

    def test_curation_drops_one_of_two_same_code_examples_by_its_line(self):
        code = "log(x**2*y).expand(log=True)"
        plain = harvesting.Example("", "16.01", "docstring:sympy.expand:95", "from sympy.abc import x, y", code,
                                   "log(x**2*y)", "log(Mul(Pow(Symbol('x'), Integer(2)), Symbol('y')))", "exact")
        positive = dataclasses.replace(plain, source="docstring:sympy.expand:100",
                                       setup="x, y = symbols('x,y', positive=True)", output="2*log(x) + log(y)",
                                       srepr="Add(Mul(Integer(2), log(Symbol('x', positive=True))), "
                                             "log(Symbol('y', positive=True)))")
        row = make_row("16.01", "sympy.expand")
        entry = {"drop": [{"source": positive.source, "code": code, "reason": "not-the-subject"}], "add": []}
        examples, notes = harvesting.curate(row, [plain, positive], entry, None, None)  # No additions: no runners.
        self.assertEqual(examples, [dataclasses.replace(plain, id="16.01/01")])
        self.assertEqual(notes, [("drop", positive.source, code, "not-the-subject")])
        # A drop that cannot name exactly one selected example is an error, never a guess.
        for selected, found in (([plain], "no selected example"),
                                ([positive, dataclasses.replace(positive, srepr="other")], "2 selected examples")):
            with self.subTest(found), self.assertRaises(harvesting.CurationError) as raised:
                harvesting.curate(row, selected, entry, None, None)
            self.assertIn(f"matches {found}", str(raised.exception))

    def test_module_imports_without_sympy(self):
        """The core CI job has no SymPy: importing the module must not import it."""
        source = (REPOSITORY / "tools" / "coverage_gate" / "examples.py").read_text(encoding="utf-8")
        top_level = re.findall(r"^(?:import|from)\s+(\S+)", source, re.MULTILINE)
        self.assertNotIn("sympy", [name.split(".")[0] for name in top_level])


if __name__ == "__main__":
    unittest.main()
