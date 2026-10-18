"""The coverage-gate reference examples: SymPy API resolution, the sandboxed example runner, the harvest and its files.

SymPy 1.14 is a development dependency (installed in .venv-mcp together with its test modules): every test that needs it
skips when it is absent, as in CI's core job, while the file, kind and runner-protocol tests, which need no SymPy, always
run. The tests build synthetic rows, guide pages, test modules, inventories and example files in temporary directories;
none reads the local docs/ inventory, the curation file or the downloaded SymPy guide sources.
"""

import contextlib
import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import time
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


def make_row(row_id, *paths, subtopic="Synthetic subtopic", area="Number theory"):
    return model.Row(id=row_id, area=area, subtopic=subtopic, sympy_apis=", ".join(paths), sympy_api_paths=paths,
                     v07_status="no", tier="T1", note="", plan_group=None, wave=None, family=None, missing="",
                     sympy_probe=None, mathforge_operations=(), status="no")


def example(api, n, *, row="27.14", source=None, code=None):
    """A synthetic candidate of the docstring of sympy.<api>."""
    return harvesting.Example("", row, source or f"docstring:sympy.{api}", "", code or f"{api}({n})", str(n),
                              f"Integer({n})", "exact")


def line_of(text, statement):
    """The 1-based line of the doctest prompt that holds statement."""
    return text.splitlines().index(f">>> {statement}") + 1


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

    def test_resolve_api_refuses_ambiguous_and_misplaced_paths(self):
        cases = (
            # The package re-exports the function under its module's name: attribute and import disagree.
            ("sympy.ntheory.continued_fraction", "ambiguous"),
            ("ntheory.factorint", "is a module"),
            ("Matrix.no_such_method", "no attribute 'no_such_method'"),
            ("sympy.factorint.", "not a dotted name"),
            ("", "not a dotted name"),
        )
        for path, reason in cases:
            with self.subTest(path), self.assertRaises(LookupError) as raised:
                harvesting.resolve_api(path)
            self.assertIn(reason, str(raised.exception))
        # `import sympy.ntheory.continued_fraction as m` would bind the function too: ask the import system.
        module = importlib.import_module("sympy.ntheory.continued_fraction")
        self.assertIs(harvesting.resolve_api("sympy.ntheory.continued_fraction.continued_fraction"),
                      module.continued_fraction)
        row = make_row("27.25", "sympy.continued_fraction", "rref", "sympy.ntheory.continued_fraction")
        unresolved = harvesting.unresolved_apis(row)
        self.assertEqual([path for path, _ in unresolved], ["rref", "sympy.ntheory.continued_fraction"])
        self.assertIn("ambiguous", unresolved[1][1])
        self.assertEqual(harvesting.unresolved_apis(make_row("27.25")), [("", "the row has no sympy_api_paths")])


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
            ("", "pprint(Integer(2))", "printing"),               # The value is None: the output was printed.
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
        self.assertEqual(self.first.run("import random", "random.random()"),
                         self.first.run("import random", "random.random()"))
        # The hash seed reaches the worker.
        self.assertEqual(self.first.run("", "hash('mathforge')"), self.first.run("", "hash('mathforge')"))
        self.assertNotEqual(self.first.run("", "hash('mathforge')"), self.second.run("", "hash('mathforge')"))


class ProtocolTests(unittest.TestCase):
    """The parent side of the runner protocol, against a fake worker that needs no SymPy."""

    FAKE_WORKER = textwrap.dedent("""\
        import json, os, sys
        print(json.dumps({"ready": True, "sympy": "fake"}), flush=True)
        for line in sys.stdin:
            code = json.loads(line)["code"]
            if code == "garbage":
                print("not json", flush=True)
            elif code == "keys":
                print(json.dumps({"output": "1"}), flush=True)
            elif code == "reason":
                print(json.dumps({"error": "made-up"}), flush=True)
            elif code == "die":
                sys.exit(3)
            elif code == "pid":
                print(json.dumps({"output": str(os.getpid()), "srepr": "pid"}), flush=True)
            else:
                print(json.dumps({"output": code, "srepr": code}), flush=True)
        """)

    def test_runner_never_trusts_a_worker_that_breaks_the_protocol(self):
        with mock.patch.object(harvesting, "_WORKER", self.FAKE_WORKER), \
                harvesting.ExampleRunner(hash_seed=0) as runner:
            self.assertEqual(runner.sympy_version, "fake")
            for code in ("garbage", "keys", "reason", "die"):
                with self.subTest(code):
                    before = runner.run("", "pid")
                    self.assertEqual(runner.run("", "pid"), before)  # A good answer keeps the worker.
                    self.assertEqual(runner.run("", code), "raises")
                    self.assertNotEqual(runner.run("", "pid"), before)  # A bad one replaces it.
        with self.assertRaises(RuntimeError):
            runner.run("", "pid")  # Closed.

    def test_runner_reports_a_worker_that_does_not_start(self):
        for script, detail in (("import sys; sys.exit(1)", "no answer"),
                               ('import json; print(json.dumps({"ready": False, "error": "ImportError: sympy"}))',
                                "ImportError: sympy")):
            with self.subTest(detail), mock.patch.object(harvesting, "_WORKER", script):
                with self.assertRaises(RuntimeError) as raised:
                    harvesting.ExampleRunner(hash_seed=0)
                self.assertIn("did not start", str(raised.exception))
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
        # The harvest excludes each unusable candidate with its reason and keeps one of each code.
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
        guide = {(item.code, item.reason) for item in excluded if item.source.startswith("guide:")}
        self.assertEqual(guide, {("pprint(factorint(360))", "printing"),
                                 ("plot(x*factorint(360)[2], show=False)", "plotting"),
                                 ("numpy.array(sorted(factorint(360)))", "optional-dependency"),
                                 ("factorint(Symbol('y'))", "raises"),
                                 ("factorint(-1.5)", "raises"),
                                 ("factorint(360)", "duplicate"),
                                 ("factorint(randint(2, 9))", "random"),
                                 ("for p in factorint(360):\n    print(p)", "printing")})
        kept = [candidate for candidate in candidates if candidate.source.startswith("guide:")]
        self.assertEqual([(item.source, item.setup, item.code, item.output) for item in kept],
                         [(f"guide:guides/synthetic.rst:{line_of(text, 'factorint(360)')}",
                           "from sympy import factorint, pprint, plot, Symbol", "factorint(360)",
                           "{2: 3, 3: 2, 5: 1}")])  # The setup keeps only what the code needs.
        self.assertTrue(all(item.source.startswith("docstring:sympy.factorint") for item in candidates[:-1]))
        self.assertEqual(len({item.code for item in candidates}), len(candidates))

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
        # One of each code (formatting aside), the first in candidate order.
        twin = dataclasses.replace(b[0], code="factorint( 1 )")
        self.assertEqual(harvesting.select([a[0], twin, b[1]], apis=apis), [a[0], b[1]])
        # Test-module asserts only top a set up to 10, again in turn across the APIs.
        tests = [example("factorint", n, source=f"tests:sympy.ntheory.tests.test_factor_::test_{n}")
                 for n in range(20, 25)]
        self.assertEqual(harvesting.select(a[:8] + tests, apis=apis), a[:8] + tests[:2])
        self.assertEqual(harvesting.select(a[:12] + tests, apis=apis), a[:12])

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
        self.assertEqual([(item.source, item.setup, item.code, item.output) for item in candidates
                          if item.source.startswith("guide:")],
                         [(f"guide:guides/solving.md:{line_of(markdown, 'factorint(n[2] * 7)')}",
                           "from sympy import factorint\nn = factorint(48)", "factorint(n[2] * 7)", "{2: 2, 7: 1}"),
                          (f"guide:modules/broken.rst:{line_of(broken, 'factorint(6)')}", "", "factorint(6)",
                           "{2: 1, 3: 1}")])
        self.assertEqual([item for item in excluded if item.source.startswith("guide:")], [])

    def test_harvest_docstring_examples(self):
        row = make_row("27.25", *CONTINUED_FRACTIONS, subtopic="Continued fractions")
        candidates, excluded = harvesting.harvest(row, self.first, doc_src=None, check_runner=self.second)
        first = candidates[0]
        self.assertTrue(first.source.startswith("docstring:"))
        self.assertEqual((first.source, first.setup, first.code, first.output, first.kind),
                         ("docstring:sympy.continued_fraction",
                          "from sympy.ntheory.continued_fraction import continued_fraction\nfrom sympy import sqrt",
                          "continued_fraction((1 + 2*sqrt(3))/5)", "[0, 1, [8, 3, 34, 3]]", "exact"))
        self.assertEqual(first.srepr, "[Integer(0), Integer(1), [Integer(8), Integer(3), Integer(34), Integer(3)]]")
        self.assertTrue(all(item.source.startswith("docstring:") for item in candidates))
        self.assertEqual(sorted({item.source for item in candidates}), sorted(f"docstring:{p}" for p in row.sympy_api_paths))
        numeric = [item for item in candidates if item.kind == "numeric"]
        self.assertEqual([item.code for item in numeric],
                         ["continued_fraction_reduce([2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]).n(10)"])
        self.assertEqual(numeric[0].output, "2.718281835")
        # The loops that print the convergents are not single expressions.
        self.assertEqual([item.reason for item in excluded], ["printing", "printing"])
        self.assertTrue(all(item.source == "docstring:sympy.continued_fraction_convergents" for item in excluded))
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
        (tests / "test_synthetic_cf.py").write_text(textwrap.dedent("""\
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
            """), encoding="utf-8")
        row = make_row("27.25", "sympy.continued_fraction", subtopic="Continued fractions")
        with mock.patch.object(harvesting, "_sympy_package_directory", return_value=package):
            candidates, excluded = harvesting.harvest(row, self.first, doc_src=None, check_runner=self.second)
        selected = harvesting.select(candidates, apis=row.sympy_api_paths)
        self.assertEqual(selected, candidates)
        self.assertEqual(len(selected), 10)  # One docstring example, then asserts up to ten.
        self.assertEqual(selected[0].source, "docstring:sympy.continued_fraction")
        values = "tests:sympy.ntheory.tests.test_synthetic_cf::test_values"
        more = "tests:sympy.ntheory.tests.test_synthetic_cf::test_more"
        self.assertEqual([(item.source, item.code, item.output) for item in selected[1:]],
                         [(values, "cf(Rational(415, 93))", "[4, 2, 6, 7]"),
                          (values, "cf(t)", "[1, [2]]"),
                          (values, "cf(Rational(3, 7))", "[0, 2, 3]"),
                          (values, "cf(helper(10))", "[1, 2, 3]"),
                          (values, "cf(Rational(1, 3))", "[0, 3]"),   # cf(5) == [6] is false: not a candidate.
                          (more, "cf(Rational(7, 2))", "[3, 2]"),
                          (more, "cf(Rational(4, 9))", "[0, 2, 4]"),
                          (more, "cf(Rational(9, 4))", "[2, 4]"),
                          (more, "cf(Rational(13, 5))", "[2, 1, 1, 2]")])
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
                make_row("27.25", *CONTINUED_FRACTIONS, subtopic="Continued fractions"))
        inventory = self.directory / "inventory.json"
        model.write_inventory(inventory, model.Inventory({"inventory": "synthetic"}, rows))
        kept = harvesting.Example("01.01/01", "01.01", "docstring:sympy.Integer", "", "Integer(2) + 2", "4",
                                  "Integer(4)", "exact")
        stale = dataclasses.replace(kept, id="27.25/01", row="27.25", source="docstring:sympy.continued_fraction",
                                    code="continued_fraction(Rational(1, 2))")
        out = self.directory / "examples.json"
        harvesting.write_examples(out, {"01.01": [kept], "27.25": [stale]})
        dropped = ("docstring:sympy.continued_fraction_reduce", "continued_fraction_reduce([-2, 1, 9, 7, 1, 2])")
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
        for expected in ("27.25", "Continued fractions", dropped[1], "guide:custom.rst:3",
                         "not used (--doc-src not given)"):
            self.assertIn(expected, text)
        self.assertNotIn("01.01", text)

        # A second run writes the same bytes: nothing depends on timing, threads or the hash seed.
        written, reported = out.read_bytes(), report.read_bytes()
        self.assertEqual(self.harvest(arguments + ["--workers", "2"]), (0, ""))
        self.assertEqual((out.read_bytes(), report.read_bytes()), (written, reported))

        # Curation mistakes stop the run before anything is written.
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


class FileTests(TemporaryDirectoryTestCase):
    def test_examples_round_trip(self):
        examples = {
            "27.25": [harvesting.Example("27.25/01", "27.25", "docstring:sympy.continued_fraction",
                                         "from sympy.ntheory.continued_fraction import continued_fraction\n"
                                         "from sympy import sqrt", "continued_fraction((1 + 2*sqrt(3))/5)",
                                         "[0, 1, [8, 3, 34, 3]]", "[Integer(0), Integer(1), [Integer(8)]]", "exact"),
                      harvesting.Example("27.25/02", "27.25", "guide:modules/ntheory.rst:12", "", "sqrt(2).n(3)",
                                         "1.41", "Float('1.41', precision=13)", "numeric")],
            "01.01": [harvesting.Example("01.01/01", "01.01", "tests:sympy.core.tests.test_numbers::test_Rational",
                                         "", "Rational(1, 2)", "1/2 — √", "Rational(1, 2)", "exact")],
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
        good = {"id": "27.25/01", "row": "27.25", "source": "docstring:sympy.continued_fraction", "setup": "",
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
        drop = {"source": "docstring:sympy.factorint", "code": "factorint(12)", "reason": "not-the-subject"}
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


class KindTests(unittest.TestCase):
    def test_kind_follows_floating_point_content(self):
        for representation, kind in (("Integer(4)", "exact"),
                                     ("Add(Symbol('x'), Rational(1, 2))", "exact"),
                                     ("Symbol('x', real=True)", "exact"),
                                     ("{Integer(2): Integer(2)}", "exact"),
                                     ("Float('2.718281835201', precision=37)", "numeric"),
                                     ("[0.5, 1]", "numeric"),
                                     ("(1+2j)", "numeric"),
                                     ("mpf('0.5')", "numeric"),
                                     ("Poly(Add(Symbol('x'), Float('1.0', precision=53)), Symbol('x'), domain='RR')",
                                      "numeric"),
                                     ("<not Python>", "exact"),
                                     ("Float('1.5', precision=53", "numeric")):
            with self.subTest(representation):
                self.assertEqual(harvesting.kind_of(representation), kind)

    def test_module_imports_without_sympy(self):
        """The core CI job has no SymPy: importing the module must not import it."""
        source = (REPOSITORY / "tools" / "coverage_gate" / "examples.py").read_text(encoding="utf-8")
        top_level = re.findall(r"^(?:import|from)\s+(\S+)", source, re.MULTILINE)
        self.assertNotIn("sympy", [name.split(".")[0] for name in top_level])


if __name__ == "__main__":
    unittest.main()
