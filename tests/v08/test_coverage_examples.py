"""The coverage-gate reference examples: SymPy API resolution, the sandboxed example runner, the harvest and its files.

SymPy 1.14 is a development dependency (installed in .venv-mcp together with its test modules): every test that needs it
skips when it is absent, as in CI's core job, while the file, curation and runner-protocol tests, which need no SymPy,
always run. The tests build synthetic rows, guide pages, test modules, inventories and example files in temporary
directories; none reads the local docs/ inventory, the curation file or the downloaded SymPy guide sources.

Harvests verify their selection with a long-lived seed-1 runner where the selection logic is under test (one worker
for all, which is fast); the verification in a fresh worker per example, which the command line always uses, is tested
on its own and through the command line.
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


def own_docstring(path):
    """The docstring that harvest reads for the API path: the object's own (for a bound method its function's),
    cleaned like inspect.getdoc cleans it, but never inherited."""
    value = harvesting.resolve_api(path)
    return inspect.cleandoc(value.__func__.__doc__ if isinstance(value, types.MethodType) else value.__doc__)


def docstring_line(path, statement):
    """The line of statement in the docstring of the API path, as harvest numbers it."""
    return line_of(own_docstring(path), statement)


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

    def test_resolve_api_compares_bound_methods_by_function_and_receiver(self):
        # A classmethod gives a new bound method on every access, so the two routes never find the identical object.
        import sympy.combinatorics  # noqa: F401  The attribute route works only once the package is imported.
        from sympy.combinatorics import Permutation
        for name in ("unrank_lex", "unrank_trotterjohnson", "unrank_nonlex"):
            for path in (f"sympy.combinatorics.Permutation.{name}",
                         f"sympy.combinatorics.permutations.Permutation.{name}"):
                with self.subTest(path):
                    method = harvesting.resolve_api(path)
                    self.assertIs(method.__func__, getattr(Permutation, name).__func__)
                    self.assertIs(method.__self__, Permutation)
        # Bound methods of an instance compare the same way: the object and the function, not the bound method.
        self.assertEqual(harvesting.resolve_api("sympy.S.Half.as_numer_denom")(), (1, 2))

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
        # The second run is a request of its own: two runs of 1.2 s fit a 2 s timeout each.
        with harvesting.ExampleRunner(hash_seed=0, timeout=2) as runner:
            self.assertEqual(runner.run("import random, time", "(time.sleep(1.2), random.random())[1]"), "random")

    def test_dummy_numbering_does_not_depend_on_history(self):
        # Unnamed Dummies carry the process-wide Dummy counter in their names; it is put back before each example.
        examples = ("Dummy()", "Dummy() + Dummy('x')", "[Dummy() for _ in range(3)]")
        with harvesting.ExampleRunner(hash_seed=0) as runner:
            first = runner.run("", "Dummy()")  # The first example of a new worker.
        with harvesting.FreshRunner(hash_seed=0) as fresh:
            alone = [fresh.run("", code) for code in examples]  # Each in a worker of its own.
        self.assertEqual(alone[0], first)
        self.assertIn("_Dummy_", first[0])
        for _ in range(2):  # Other examples that create Dummies ran in this worker before.
            self.first.run("", "[Dummy() for _ in range(5)]")
            self.assertEqual([self.first.run("", code) for code in examples], alone)
        # The index base moves on, so a Dummy that outlived its example never equals a new one of the same name.
        self.assertEqual(self.first.run("import sympy.abc", "setattr(sympy.abc, '_mathforge_kept', Dummy('z')) or 1"),
                         ("1", "1"))
        self.assertEqual(self.first.run("import sympy.abc", "Dummy('z') == sympy.abc._mathforge_kept"),
                         ("False", "False"))

    def test_module_stores_do_not_leak(self):
        # Stores that clear_cache() leaves alone are put back before each example: CRootOf's refined isolating
        # intervals and the prime sieve, compared with workers that have run nothing.
        root = "from sympy import CRootOf\nfrom sympy.abc import x\nr = CRootOf(x**5 - x + 1, 0)"
        probes = ((root, "r._get_interval()"), ("from sympy import sieve", "len(sieve._list)"))
        with harvesting.FreshRunner(hash_seed=0) as fresh:
            alone = [fresh.run(*probe) for probe in probes]
        self.first.run(root, "r.eval_rational(n=2)")  # Refines the root's cached interval.
        self.first.run("from sympy import sieve", "sieve.extend(1000) or 1")
        self.assertEqual([self.first.run(*probe) for probe in probes], alone)

    def test_global_assumptions_do_not_leak(self):
        add = ("from sympy.assumptions.assume import global_assumptions\nx = Symbol('x')",
               "global_assumptions.add(Q.real(x)) or 1")
        ask = ("x = Symbol('x')", "ask(Q.real(x))")
        scoped = ("x = Symbol('x')\nwith assuming(Q.real(x)):\n    inside = ask(Q.real(x))", "(inside, ask(Q.real(x)))")
        before = [self.first.run(*ask), self.first.run(*scoped)]
        self.assertEqual(before, [("None", "None"), ("(True, None)", "(True, None)")])
        self.assertEqual(self.first.run(*add), ("1", "1"))
        self.assertEqual([self.first.run(*ask), self.first.run(*scoped)], before)  # The order does not matter.

    def test_fresh_runner_gives_each_example_a_worker_of_its_own(self):
        with harvesting.FreshRunner(hash_seed=1, spares=1) as fresh:
            pids = [fresh.run("import os", "os.getpid()")[0] for _ in range(3)]
            self.assertEqual(len(set(pids)), 3)
            self.assertNotIn(self.pid(self.second), pids)
            self.assertEqual(fresh.run("x = symbols('x')", "factor(x**2 - 1)"),
                             self.second.run("x = symbols('x')", "factor(x**2 - 1)"))
            self.assertEqual(fresh.run("", "hash('mathforge')"), self.second.run("", "hash('mathforge')"))
        with self.assertRaises(RuntimeError):
            fresh.run("", "1")  # Closed.
        for arguments in ({"hash_seed": 1, "spares": 0}, {"hash_seed": -1}, {"hash_seed": 1, "timeout": 0}):
            with self.subTest(arguments), self.assertRaises(ValueError):
                harvesting.FreshRunner(**arguments)

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
        import json, os, sys, time
        print(json.dumps({"ready": True, "sympy": "1.14.0"}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            code = request["code"]
            if code in ("draw", "slowdraw"):  # Its value is the random generators' seed.
                if code == "slowdraw":
                    time.sleep(1.2)
                print(json.dumps({"output": str(request["seed"]), "srepr": "0", "kind": "exact", "drew": True}),
                      flush=True)
            elif code == "steadydraw":  # Draws, but its value does not depend on the draw.
                print(json.dumps({"output": "5", "srepr": "5", "kind": "exact", "drew": True}), flush=True)
            elif code == "garbage":
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

    def test_runner_runs_a_drawing_example_again_as_a_request_of_its_own(self):
        with mock.patch.object(harvesting, "_WORKER", self.FAKE_WORKER), \
                harvesting.ExampleRunner(hash_seed=0, timeout=2) as runner:
            self.assertEqual(runner.run("", "draw"), "random")
            self.assertEqual(runner.run("", "draw", random_allowed=True), ("0", "0"))
            self.assertEqual(runner.run("", "steadydraw"), ("5", "5"))
            # Two runs of 1.2 s each: each request has its own 2 s timeout, so this is "random", not "timeout".
            started = time.monotonic()
            self.assertEqual(runner.run("", "slowdraw"), "random")
            self.assertGreater(time.monotonic() - started, 2.0)

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
            >>> print(latex(factorint(360)))
            a LaTeX string
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
        self.assertEqual(guide, {("latex(factorint(360))", "printing"),  # print() shows it; LaTeX is the subject.
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
        first, second = lines_of(own_docstring("sympy.expand"), "log(x**2*y).expand(log=True)")
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

    def test_selected_examples_are_verified_in_fresh_workers(self):
        # A class attribute set by one example leaks into later examples of the same worker: neither the resets nor
        # the namespace check see it. Only a worker without that history can tell.
        row = make_row("01.01", "sympy.Integer")
        leak = ("from sympy import Integer", "setattr(Integer, '_mathforge_leak', 1) or 1")
        probe = "hasattr(Integer(5), '_mathforge_leak')"
        with harvesting.ExampleRunner(hash_seed=0) as polluted, harvesting.ExampleRunner(hash_seed=1) as shared, \
                harvesting.FreshRunner(hash_seed=1) as fresh:
            for runner in (polluted, shared):
                self.assertEqual(runner.run(*leak), ("1", "1"))
            # A verifier with the same kind of history agrees with the leaked record...
            leaked = harvesting.examine(row, "guide:g.rst:1", "", probe, polluted, shared)
            self.assertEqual((leaked.output, leaked.srepr), ("True", "True"))
            # ...a worker of its own does not, so the record is excluded.
            self.assertEqual(harvesting.examine(row, "guide:g.rst:2", "", f"{probe} ", polluted, fresh),
                             harvesting.Excluded("01.01", "guide:g.rst:2", f"{probe} ", "unstable"))
            # harvest verifies in fresh workers unless told otherwise.
            doc_src, text = self.guide("leak.rst", f">>> {probe}\nTrue\n")
            candidates, excluded = harvesting.harvest(row, polluted, doc_src=doc_src)
        self.assertIn(harvesting.Excluded("01.01", f"guide:leak.rst:{line_of(text, probe)}", probe, "unstable"),
                      excluded)
        self.assertNotIn(probe, [item.code for item in candidates])
        self.assertTrue(candidates)  # Integer's own docstring examples pass the same verification.

    def test_three_hash_seeds_catch_orders_that_two_miss(self):
        # The order of these models depends on the hash seed, yet seeds 0 and 1 happen to agree on it (seed 2 not).
        row = make_row("04.07", "sympy.satisfiable")
        setup, code = "from sympy.abc import a, b", "list(satisfiable(a | b, all_models=True))"
        self.assertEqual(self.first.run(setup, code), self.second.run(setup, code))
        self.assertIsInstance(harvesting.examine(row, "guide:g.rst:1", setup, code, self.first, self.second),
                              harvesting.Example)  # What one verifying seed misses...
        with harvesting.ExampleRunner(hash_seed=2) as third:
            self.assertEqual(harvesting.examine(row, "guide:g.rst:1", setup, code, self.first, (self.second, third)),
                             harvesting.Excluded("04.07", "guide:g.rst:1", code, "unstable"))  # ...two catch.
            for verifiers in ((self.second, self.second), (self.first,), ()):  # Seeds of their own, not the first's.
                with self.subTest(len(verifiers)), self.assertRaises(ValueError):
                    harvesting.examine(row, "guide:g.rst:1", setup, code, self.first, verifiers)
            examples, _ = harvesting.curate(row, [], {"add": [{"source": "guide:g.rst:2", "setup": setup,
                                                               "code": code}]}, self.first, self.second)
            self.assertEqual(len(examples), 1)
            with self.assertRaises(harvesting.CurationError):  # Additions are verified the same way.
                harvesting.curate(row, [], {"add": [{"source": "guide:g.rst:2", "setup": setup, "code": code}]},
                                  self.first, (self.second, third))

    def test_display_calls_record_the_value_they_show(self):
        # SymPy's ODE hint docstrings show their results through pprint(): the shown expression is the example.
        path = "sympy.solvers.ode.single.Separable"
        candidates, excluded = harvesting.harvest(make_row("13.01", path), self.first, doc_src=None,
                                                  check_runner=self.second)
        code = "dsolve(genform, f(x), hint='separable_Integral')"
        line = line_containing(own_docstring(path), f">>> pprint({code})")
        self.assertEqual([(item.source, item.output) for item in candidates if item.code == code],
                         [(f"docstring:{path}:{line}",
                           "Eq(Integral(b(_y)/d(_y), (_y, f(x))), C1 + Integral(c(x)/a(x), x))")])
        self.assertEqual(len(candidates), 3)  # Every pprint() example of the docstring; none was kept before.
        self.assertEqual([item for item in excluded if item.reason == "printing"], [])
        # Keywords that only change the display are unwrapped too; a call that shows two values, or a printer that is
        # the subject (latex() inside print()), stays printing.
        doc_src, text = self.guide("guides/display.rst", """\
            >>> from sympy import factorint, pprint
            >>> pprint(factorint(12), use_unicode=False)
            {2: 2, 3: 1}
            >>> print(factorint(18), factorint(20))
            {2: 1, 3: 2} {2: 2, 5: 1}
            >>> print(latex(factorint(28)))
            a LaTeX string
            """)
        candidates, excluded = harvesting.harvest(make_row("27.14", "sympy.factorint"), self.first, doc_src=doc_src,
                                                  check_runner=self.second)
        self.assertEqual([(item.code, item.setup, item.output) for item in self.guide_items(candidates)],
                         [("factorint(12)", "from sympy import factorint, pprint", "{2: 2, 3: 1}")])
        self.assertEqual({(item.code, item.reason) for item in self.guide_items(excluded)},
                         {("print(factorint(18), factorint(20))", "printing"), ("latex(factorint(28))", "printing")})

    def test_methods_are_attributed_only_through_attributes(self):
        doc_src, _ = self.guide("guides/exponentials.rst", """\
            >>> from sympy import Matrix, exp
            >>> from sympy.abc import x
            >>> exp(2*x).diff(x)
            2*exp(2*x)
            >>> M = Matrix([[0, 1], [0, 0]])
            >>> M.exp()
            Matrix([
            [1, 1],
            [0, 1]])
            """)
        # Matrix.exp is reached only as an attribute: the bare call exp(x) is the scalar function.
        method, _ = harvesting.harvest(make_row("24.26", "Matrix.exp"), self.first, doc_src=doc_src,
                                       check_runner=self.second)
        self.assertEqual([item.code for item in self.guide_items(method)], ["M.exp()"])
        # sympy.exp, a module-level function, is reached by the bare call (and, as before, by the attribute).
        function, _ = harvesting.harvest(make_row("20.01", "sympy.exp"), self.first, doc_src=doc_src,
                                         check_runner=self.second)
        self.assertEqual([item.code for item in self.guide_items(function)], ["exp(2*x).diff(x)", "M.exp()"])

    def test_curation_can_add_the_rows_probe(self):
        # A row without harvested examples gets its own sympy_probe, verified like any addition.
        row = dataclasses.replace(make_row("27.25", "sympy.continued_fraction"),
                                  sympy_probe="continued_fraction(Rational(415, 93))")
        add = {"source": "probe:27.25", "setup": "", "code": row.sympy_probe}
        with harvesting.ExampleRunner(hash_seed=2) as third:
            examples, notes = harvesting.curate(row, [], {"add": [add]}, self.first, (self.second, third))
            self.assertEqual([(item.id, item.source, item.setup, item.code, item.output) for item in examples],
                             [("27.25/01", "probe:27.25", "", row.sympy_probe, "[4, 2, 6, 7]")])
            self.assertEqual(notes, [("add", "probe:27.25", row.sympy_probe, "[4, 2, 6, 7]", "")])
            for wrong in ({**add, "source": "probe:27.24"}, {**add, "setup": "x = 1"},
                          {**add, "code": "continued_fraction(2)"}):
                with self.subTest(wrong), self.assertRaises(harvesting.CurationError):
                    harvesting.curate(row, [], {"add": [wrong]}, self.first, (self.second, third))
        path = self.directory / "examples.json"
        harvesting.write_examples(path, {"27.25": examples})
        self.assertEqual(harvesting.load_examples(path), {"27.25": examples})

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

            A statement that does not parse (old Python 2 code) cannot be run; it names no API either:

            >>> print factorint(10)
            {2: 1, 5: 1}
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

    def test_only_an_apis_own_docstring_is_harvested(self):
        # ArrayDerivative has no docstring of its own: inspect.getdoc hands it Derivative's, whose examples were then
        # recorded under ArrayDerivative's name. Derivative, a class with doctests of its own, keeps them.
        inherits, owns = "sympy.tensor.array.array_derivatives.ArrayDerivative", "sympy.Derivative"
        self.assertIsNone(harvesting.resolve_api(inherits).__doc__)
        self.assertEqual(inspect.getdoc(harvesting.resolve_api(inherits)), own_docstring(owns))  # What it inherits.
        no_tests = self.directory / "sympy"  # No test module to top a short set up from.
        no_tests.mkdir()
        with mock.patch.object(harvesting, "_sympy_package_directory", return_value=no_tests):
            # Nor do Sum.doit (it overrides Basic.doit without a docstring), SparseMatrix and Array take any.
            for path in (inherits, "Sum.doit", "sympy.SparseMatrix", "sympy.Array"):
                with self.subTest(path):
                    self.assertEqual(harvesting.harvest(make_row("26.03", path), self.first, doc_src=None,
                                                        check_runner=self.second), ([], []))
            # Listed first, ArrayDerivative used to take Derivative's text, and Derivative's own copy was then skipped
            # as already seen; now every example of that text is Derivative's.
            candidates, excluded = harvesting.harvest(make_row("26.03", inherits, owns), self.first, doc_src=None,
                                                      check_runner=self.second)
        self.assertGreaterEqual(len(candidates), 10)
        self.assertEqual({re.sub(r":\d+$", "", item.source) for item in candidates + excluded}, {f"docstring:{owns}"})
        code = "Derivative(x**2, x, evaluate=True)"
        self.assertEqual([(item.source, item.output) for item in candidates if item.code == code],
                         [(f"docstring:{owns}:{docstring_line(owns, code)}", "2*x")])

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


    def test_cli_never_fails_silently(self):
        import harvest_examples
        rows = (make_row("27.14", *PERFECT_NUMBERS), make_row("27.25", *CONTINUED_FRACTIONS))
        inventory = self.directory / "inventory.json"
        model.write_inventory(inventory, model.Inventory({"inventory": "synthetic"}, rows))
        curation = self.directory / "curation.json"
        curation.write_text("{}", encoding="utf-8")
        out, report = self.directory / "examples.json", self.directory / "report.md"
        arguments = [str(inventory), "--out", str(out), "--report", str(report), "--curation", str(curation),
                     "--workers", "2"]
        # A worker thread that fails: status 1, its traceback and a message on stderr, and no file written.
        with mock.patch.object(harvest_examples, "_harvest_row", side_effect=RuntimeError("injected worker failure")):
            status, errors = self.harvest(arguments)
        self.assertEqual(status, 1)
        for expected in ("Traceback", "RuntimeError: injected worker failure", "a harvest thread failed",
                         "nothing was written"):
            self.assertIn(expected, errors)
        self.assertFalse(out.exists() or report.exists())
        # A failure in the main thread after the harvest: the same, and still nothing is written.
        with mock.patch.object(harvest_examples, "_harvest_row",
                               side_effect=lambda row, *rest: harvest_examples.RowResult(row, (), 0, (), (), ())), \
                mock.patch.object(harvest_examples, "render_report", side_effect=ValueError("injected report failure")):
            status, errors = self.harvest(arguments)
        self.assertEqual(status, 1)
        self.assertIn("ValueError: injected report failure", errors)
        self.assertFalse(out.exists() or report.exists())


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
            ("unknown drop field", {"27.14": {"drop": [{**drop, "note": "see OEIS"}]}}),
            ("evidence is for drops", {"27.14": {"add": [{**add, "evidence": "see OEIS"}]}}),
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
        probe = {"source": "probe:27.14", "setup": "", "code": "is_perfect(28)"}  # The row's own sympy_probe
        path.write_text(json.dumps({"27.14": {"add": [probe]}}), encoding="utf-8")
        self.assertEqual(harvesting.load_curation(path), {"27.14": {"drop": [], "add": [probe]}})
        self.assertEqual(set(harvesting.DROP_REASONS),
                         {"not-the-subject", "sympy-incorrect", "duplicate"} | set(harvesting.EXCLUSION_REASONS))

    def test_curation_drops_sympy_incorrect_examples_only_with_evidence(self):
        self.assertEqual(harvesting.EVIDENCE_REASON, "sympy-incorrect")
        self.assertNotIn("sympy-incorrect", harvesting.EXCLUSION_REASONS)  # Only a curator can tell, never the harvest.
        evidence = "motzkin(10) is 2188 (OEIS A001006); SymPy 1.14 returns 835"
        drop = {"source": "docstring:sympy.motzkin:26", "code": "motzkin(10)", "reason": "sympy-incorrect",
                "evidence": evidence}
        path = self.directory / "curation.json"

        def load(entry):
            path.write_text(json.dumps({"22.07": entry}), encoding="utf-8")
            return harvesting.load_curation(path)

        # With evidence the drop is accepted, and the evidence is kept as written.
        self.assertEqual(load({"drop": [drop]}), {"22.07": {"drop": [drop], "add": []}})
        # Without it (missing, empty, blank, not a string) the file is refused. The message names the file, the row and
        # the drop: here the second of two, so it has to say which.
        later = {"source": "docstring:sympy.catalan:40", "code": "catalan(10)", "reason": "sympy-incorrect"}
        for name, bad in (("missing", later), ("empty", {**later, "evidence": ""}),
                          ("blank", {**later, "evidence": " \t\n"}), ("not a string", {**later, "evidence": None})):
            with self.subTest(name), self.assertRaises(harvesting.CurationError) as raised:
                load({"drop": [drop, bad]})
            message = str(raised.exception)
            for named in (str(path), "22.07", later["source"], repr(later["code"]), "'evidence'"):
                self.assertIn(named, message)
            self.assertNotIn(drop["source"], message)
        # Only that reason takes evidence: every other keeps its shape, and the field is refused on it like any field
        # an entry is not defined to hold.
        for reason in (reason for reason in harvesting.DROP_REASONS if reason != "sympy-incorrect"):
            plain = {"source": drop["source"], "code": drop["code"], "reason": reason}
            with self.subTest(reason):
                self.assertEqual(load({"drop": [plain]}), {"22.07": {"drop": [plain], "add": []}})
                with self.assertRaises(harvesting.CurationError) as raised:
                    load({"drop": [{**plain, "evidence": evidence}]})
                for named in (str(path), "22.07", plain["source"], repr(plain["code"]), "'evidence'"):
                    self.assertIn(named, str(raised.exception))

    def test_report_lists_each_drop_with_its_reason_and_the_evidence_of_a_sympy_incorrect_one(self):
        import harvest_examples
        row = make_row("22.07", "sympy.catalan", "sympy.motzkin", subtopic="Catalan & Motzkin",
                       area="Combinatorial numbers")
        kept = harvesting.Example("", "22.07", "docstring:sympy.catalan:12", "", "catalan(5)", "42", "Integer(42)",
                                  "exact")
        wrong = dataclasses.replace(kept, source="docstring:sympy.motzkin:26", code="motzkin(10)", output="835",
                                    srepr="Integer(835)")
        other = dataclasses.replace(kept, source="docstring:sympy.catalan:30", code="catalan(10)", output="16796",
                                    srepr="Integer(16796)")
        awkward = dataclasses.replace(wrong, source="docstring:sympy.motzkin:28", code="motzkin(11)", output="2188",
                                      srepr="Integer(2188)")
        evidence = "motzkin(10) is 2188 (OEIS A001006); SymPy 1.14 returns 835"
        long_evidence = ('motzkin(11) is 5798 (OEIS A001006, "a(11)").\nSymPy 1.14 returns 2188, which is `a(10)` | '
                         'one off — not a rounding matter.')
        entry = {"drop": [{"source": wrong.source, "code": wrong.code, "reason": "sympy-incorrect",
                           "evidence": evidence},
                          {"source": other.source, "code": other.code, "reason": "not-the-subject"},
                          {"source": awkward.source, "code": awkward.code, "reason": "sympy-incorrect",
                           "evidence": long_evidence}]}
        examples, notes = harvesting.curate(row, [kept, wrong, other, awkward], entry, None, None)
        self.assertEqual([item.code for item in examples], ["catalan(5)"])
        self.assertEqual(notes, [("drop", wrong.source, wrong.code, "sympy-incorrect", evidence),
                                 ("drop", other.source, other.code, "not-the-subject", ""),
                                 ("drop", awkward.source, awkward.code, "sympy-incorrect", long_evidence)])

        text = harvest_examples.render_report(
            inventory=Path("inventory.json"), out=Path("examples.json"), doc_src=None,
            curation="`curation.json` (1 rows)", sympy="1.14.0",
            results=[harvest_examples.RowResult(row, (), 4, tuple(examples), (), tuple(notes))])
        lines = text.splitlines()
        self.assertIn(f'- dropped (sympy-incorrect): `{wrong.source}` `"motzkin(10)"`; '
                      f'evidence: `{json.dumps(evidence)}`', lines)
        self.assertIn(f'- dropped (not-the-subject): `{other.source}` `"catalan(10)"`', lines)  # No evidence to show.
        # The evidence is shown whole, on one line, as a JSON string that can be pasted back into a curation file.
        shown = [line for line in lines if line.startswith(f"- dropped (sympy-incorrect): `{awkward.source}`")]
        self.assertEqual(len(shown), 1)
        self.assertEqual(json.loads(shown[0].partition("; evidence: ")[2].strip("` ")), long_evidence)

    def test_report_header_says_why_exclusion_counts_can_vary(self):
        import harvest_examples
        text = harvest_examples.render_report(inventory=Path("inventory.json"), out=Path("examples.json"), doc_src=None,
                                              curation="none", sympy="1.14.0", results=[])
        header = text.partition("## Summary")[0].splitlines()
        self.assertIn("- Exclusion counts can vary by a few between runs: whether an example near the 60 s timeout "
                      "finishes depends on the machine's load. Each selected record, by contrast, was verified in "
                      "fresh processes.", header)

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
        self.assertEqual(notes, [("drop", positive.source, code, "not-the-subject", "")])
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
