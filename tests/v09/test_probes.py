"""tools/check_probes.py and benchmarks/probes: the probe scoreboard inside the unittest suite.

One fresh run of every probe is compared in-process with the committed baseline, benchmarks/probes-v08.json, as
`python tools/check_probes.py` compares it. The other tests drive the tool's modes with stand-in outcomes and
temporary baselines, and the classification of outcomes with synthetic attempts.
"""

import ast
import contextlib
from fractions import Fraction
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))  # check_probes is a script beside the other tools, not a package.

import check_probes  # noqa: E402  (it puts the repository root on sys.path, for benchmarks.probes)
from benchmarks import probes  # noqa: E402

import mathforge as mf  # noqa: E402
from mathforge.contracts import ErrorInfo, ExecutionStatus, Outcome as ResultOutcome  # noqa: E402

RETIRED = "algebra_retired_probe"  # An id that no probe has.


def run_tool(argv, outcomes):
    """check_probes.main on argv with stand-in outcomes: its exit status and everything it printed."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = check_probes.main(argv, outcomes=outcomes)
    return code, output.getvalue()


def attempt(run, expect=None):
    """The outcome of a synthetic probe."""
    return probes.run_probe(probes.Probe("test_probe", "test", "A synthetic request.", run, expect), mf)


def fail(error):
    raise error


def failed_result(mf):
    return mf.Result(ExecutionStatus.ERROR, ResultOutcome.NOT_APPLICABLE,
                     error=ErrorInfo("internal_error", "A synthetic failure."))


class ScoreboardTests(unittest.TestCase):
    """The committed baseline matches what this MathForge answers."""

    @classmethod
    def setUpClass(cls):
        cls.outcomes = probes.run_all(mf)

    def test_the_scoreboard_matches_the_committed_baseline(self):
        code, output = run_tool(["--baseline", str(check_probes.BASELINE)], self.outcomes)
        self.assertEqual(code, 0, output)
        self.assertIn("Differences from benchmarks/probes-v08.json: none", output)

    def test_no_answer_is_wrong(self):
        wrong = {key: outcome.detail for key, outcome in self.outcomes.items() if outcome.status == "wrong"}
        self.assertEqual(wrong, {})

    def test_every_status_is_one_of_the_seven(self):
        self.assertLessEqual({outcome.status for outcome in self.outcomes.values()}, set(probes.STATUSES))


class DefinitionTests(unittest.TestCase):
    def test_ids_are_unique_snake_case_and_start_with_their_group(self):
        found = probes.all_probes()
        ids = [probe.id for probe in found]
        self.assertEqual(len(ids), len(set(ids)))
        for probe in found:
            self.assertRegex(probe.id, r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
            self.assertTrue(probe.id.startswith(probe.group + "_"), probe.id)

    def test_every_group_has_probes(self):
        self.assertEqual({probe.group for probe in probes.all_probes()}, set(probes.GROUPS))

    def test_at_least_fifty_probes_check_their_answer(self):
        self.assertGreaterEqual(sum(probe.expect is not None for probe in probes.all_probes()), 50)

    def test_the_committed_baseline_is_canonical(self):
        text = check_probes.BASELINE.read_text(encoding="utf-8").replace("\r\n", "\n")  # A CRLF checkout too.
        baseline = check_probes.load(check_probes.BASELINE)
        self.assertEqual(text, check_probes.dumps(baseline))
        self.assertEqual(set(baseline["probes"]), {probe.id for probe in probes.all_probes()})
        self.assertNotIn("wrong", set(baseline["probes"].values()))

    def test_the_package_never_imports_mathforge(self):
        for path in sorted((REPOSITORY / "benchmarks" / "probes").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertNotIn("mathforge", [name.split(".")[0] for name in names], path.name)

    def test_a_malformed_probe_is_rejected(self):
        module = importlib.import_module("benchmarks.probes.logic")
        saved = module.PROBES
        self.addCleanup(setattr, module, "PROBES", saved)
        for probe in (probes.Probe("logic Bad", "logic", "A request.", lambda mf: 1),
                      probes.Probe("algebra_misplaced", "logic", "A request.", lambda mf: 1),
                      probes.Probe("logic_no_request", "logic", " ", lambda mf: 1),
                      probes.Probe("logic_no_run", "logic", "A request.", None),
                      ("logic_not_a_probe",),
                      saved[0]):
            module.PROBES = saved + (probe,)
            with self.assertRaises(ValueError):
                probes.all_probes()


class ClassificationTests(unittest.TestCase):
    """run_probe's status of each kind of outcome."""

    def assertStatus(self, status, run, expect=None):
        outcome = attempt(run, expect)
        self.assertEqual(outcome.status, status, outcome.detail)

    def test_values(self):
        self.assertStatus("ok", lambda mf: 3, lambda v: v == 3)
        self.assertStatus("ok", lambda mf: mf.Rational(1, 3))
        self.assertStatus("ok", lambda mf: mf.operations.integer_gcd(12, 18), lambda v: v == 6)  # A Result's value.
        self.assertStatus("wrong", lambda mf: 3, lambda v: v == 4)
        self.assertStatus("wrong", lambda mf: 3, lambda v: v.missing)  # A check that raises.
        self.assertStatus("wrong", lambda mf: 3, lambda v: 1)  # A check must return True itself.
        self.assertIn("the check returned False", attempt(lambda mf: 3, lambda v: False).detail)

    def test_deferred(self):
        def antiderivative(mf):
            x, = probes.symbols(mf, "x")
            return mf.operations.function_antiderivative(mf.parse_function("sin(x)^2", symbols={"x": x}), x)

        self.assertStatus("deferred", antiderivative)
        self.assertStatus("deferred", lambda mf: antiderivative(mf).unwrap())  # The OperationError of a value API.
        self.assertStatus("deferred", lambda mf: antiderivative(mf).deferred)  # An unevaluated record as the value.

    def test_unsupported(self):
        def parametric(mf):
            x, a = probes.symbols(mf, "x a")
            return mf.solve(mf.parse_problem("x^2 - a = 0", symbols={"x": x, "a": a}), for_=x)

        self.assertStatus("unsupported", lambda mf: mf.ntheory.no_such_operation(1))
        self.assertStatus("unsupported", lambda mf: mf.no_such_area)
        self.assertStatus("unsupported", lambda mf: probes.planned(mf, "no_such_operation", "ntheory")(1))
        self.assertStatus("unsupported", lambda mf: mf.Rational(1, 3).no_such_method())
        self.assertStatus("unsupported", lambda mf: fail(mf.UnsupportedOperation("not yet")))
        self.assertStatus("unsupported", parametric)
        self.assertStatus("unsupported", lambda mf: parametric(mf).unwrap())

    def test_invalid_input(self):
        self.assertStatus("invalid_input", lambda mf: mf.parse_expression("1/0", symbols={}))
        self.assertStatus("invalid_input", lambda mf: mf.parse_expression("exp(1)", symbols={}))  # A ParseError.
        self.assertStatus("invalid_input", lambda mf: mf.mod_inverse(6, 9))
        self.assertStatus("invalid_input", lambda mf: mf.operations.mod_inverse(6, 9))

    def test_resource_limit(self):
        self.assertStatus("resource_limit", lambda mf: mf.factorial(10000))
        self.assertStatus("resource_limit", lambda mf: mf.operations.factorial(10000))
        self.assertStatus("resource_limit", lambda mf: fail(mf.ResourceLimitError("Computation limit exceeded")))

    def test_error(self):
        self.assertStatus("error", lambda mf: mf.Matrix(((1, 0), (0, 1))) / 2)  # Matrix ** 2 works since v0.9.
        self.assertStatus("error", lambda mf: 1 // 0)
        self.assertStatus("error", lambda mf: None.missing)  # Missing on no MathForge value.
        self.assertStatus("error", lambda mf: mf.ntheory.continued_fraction.no_such_attribute)  # A plain function.
        self.assertStatus("error", failed_result)
        self.assertStatus("error", lambda mf: failed_result(mf).unwrap())

    def test_a_failure_inside_mathforge_is_no_missing_api(self):
        # Code compiled under a file name of the mathforge package stands for mathforge's own code: an attribute that
        # it misses, even on one of its modules, is an internal error and not an operation that MathForge lacks.
        def broken(mf):
            code = compile("import mathforge.ntheory\nmathforge.ntheory.no_such_name\n", mf.__file__, "exec")
            exec(code, {})

        self.assertStatus("error", broken)

    def test_planned_looks_in_the_areas_then_in_mathforge_operations(self):
        self.assertIs(probes.planned(mf, "continued_fraction", "algebra", "ntheory"), mf.ntheory.continued_fraction)
        self.assertIs(probes.planned(mf, "continued_fraction", "algebra"), mf.operations.continued_fraction)
        with self.assertRaises(AttributeError) as caught:
            probes.planned(mf, "no_such_operation", "ntheory")
        self.assertIs(caught.exception.obj, mf.operations)

    def test_exact_takes_only_exact_rationals(self):
        self.assertEqual(probes.exact(mf.Rational(-3, 6)), Fraction(-1, 2))
        for value in (True, 0.5, "1", None):
            with self.assertRaises(TypeError):
                probes.exact(value)

    def test_normal_forms_ignore_order_but_not_value(self):
        x, = probes.symbols(mf, "x")
        X, SIN = probes.atom("x"), probes.apply("sin", probes.atom("x"))
        self.assertEqual(probes.function_poly(mf.parse_function("sin(x)*x + 1/2", symbols={"x": x})),
                         Fraction(1, 2) + X * SIN)
        self.assertNotEqual(probes.function_poly(mf.parse_function("sin(x)*x", symbols={"x": x})), X * SIN + 1)
        square = mf.parse_expression("(x + 1)^2", symbols={"x": x})
        self.assertEqual(probes.expression_poly(square), X**2 + 2 * X + 1)
        self.assertFalse(probes.is_monomial_sum(square))
        self.assertTrue(probes.is_monomial_sum(mf.expand(square, x)))


class ToolTests(unittest.TestCase):
    """The tool's modes on temporary baselines and stand-in outcomes."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "probes.json"
        self.ids = [probe.id for probe in probes.all_probes()]
        self.first = self.ids[0]
        self.statuses = {key: "unsupported" for key in self.ids}

    def write(self, statuses, version=mf.__version__):
        self.path.write_text(check_probes.dumps(check_probes.document(version, statuses)), encoding="utf-8")

    def fresh(self, first=None):
        """Stand-in outcomes: every probe unsupported, the first one in the given status."""
        statuses = dict(self.statuses) if first is None else {**self.statuses, self.first: first}
        return {key: probes.Outcome(status, "a stand-in outcome") for key, status in statuses.items()}

    def test_a_matching_scoreboard_passes(self):
        self.write(self.statuses)
        code, output = run_tool(["--baseline", str(self.path)], self.fresh())
        self.assertEqual(code, 0, output)
        self.assertRegex(output, r"\ntotal +0 +0 +0 +%d " % len(self.ids))

    def test_an_improvement_is_a_difference(self):
        self.write(self.statuses)
        code, output = run_tool(["--baseline", str(self.path)], self.fresh("ok"))
        self.assertEqual(code, 1)
        self.assertIn(f"{self.first}: unsupported -> ok", output)

    def test_a_wrong_answer_fails(self):
        self.write(self.statuses)
        code, output = run_tool(["--baseline", str(self.path)], self.fresh("wrong"))
        self.assertEqual(code, 1)
        self.assertIn("Wrong answers: 1", output)

    def test_added_and_removed_probes_are_differences(self):
        statuses = {key: status for key, status in self.statuses.items() if key != self.first}
        self.write({**statuses, RETIRED: "unsupported"})
        code, output = run_tool(["--baseline", str(self.path)], self.fresh())
        self.assertEqual(code, 1)
        self.assertIn(f"{self.first}: absent -> unsupported", output)
        self.assertIn(f"{RETIRED}: unsupported -> absent", output)

    def test_another_version_is_only_noted(self):
        self.write(self.statuses, version="0.0.0")
        code, output = run_tool(["--baseline", str(self.path)], self.fresh())
        self.assertEqual(code, 0, output)
        self.assertIn("measured with mathforge 0.0.0", output)

    def test_update_refuses_a_regression_unless_it_is_accepted(self):
        self.write({**self.statuses, self.first: "ok"})
        before = self.path.read_bytes()
        code, output = run_tool(["--update", "--baseline", str(self.path)], self.fresh())
        self.assertEqual(code, 1)
        self.assertIn(f"no longer ok: {self.first}", output)
        self.assertEqual(self.path.read_bytes(), before)
        code, output = run_tool(["--update", "--baseline", str(self.path), "--accept-regression", self.first],
                                self.fresh())
        self.assertEqual(code, 0, output)
        written = self.path.read_bytes()
        self.assertNotIn(b"\r", written)
        self.assertTrue(written.endswith(b"}\n"))
        baseline = check_probes.load(self.path)
        self.assertEqual(baseline["probes"][self.first], "unsupported")
        self.assertEqual(baseline["totals"]["all"], len(self.ids))
        self.assertEqual(baseline["mathforge"], mf.__version__)

    def test_update_refuses_to_drop_a_probe_that_was_ok(self):
        self.write({**self.statuses, RETIRED: "ok"})
        code, output = run_tool(["--update", "--baseline", str(self.path)], self.fresh())
        self.assertEqual(code, 1)
        self.assertIn(f"no longer ok: {RETIRED}", output)
        code, _ = run_tool(["--update", "--baseline", str(self.path), "--accept-regression", RETIRED], self.fresh())
        self.assertEqual(code, 0)
        self.assertNotIn(RETIRED, check_probes.load(self.path)["probes"])

    def test_update_never_writes_a_wrong_answer(self):
        self.write(self.statuses)
        before = self.path.read_bytes()
        code, output = run_tool(["--update", "--baseline", str(self.path)], self.fresh("wrong"))
        self.assertEqual(code, 1)
        self.assertIn("a wrong answer never enters the baseline", output)
        self.assertEqual(self.path.read_bytes(), before)

    def test_update_records_an_improvement_and_creates_a_missing_baseline(self):
        code, output = run_tool(["--update", "--baseline", str(self.path)], self.fresh("ok"))
        self.assertEqual(code, 0)
        self.assertIn("Updated", output)
        self.assertEqual(check_probes.load(self.path)["probes"], {**self.statuses, self.first: "ok"})
        code, output = run_tool(["--update", "--baseline", str(self.path)], self.fresh("ok"))
        self.assertEqual(code, 0)
        self.assertIn("is up to date", output)

    def test_accept_regression_must_name_a_regression_and_needs_update(self):
        self.write(self.statuses)
        code, output = run_tool(["--update", "--baseline", str(self.path), "--accept-regression", self.first],
                                self.fresh())
        self.assertEqual(code, 2)
        self.assertIn("names no regression", output)
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
            check_probes.main(["--accept-regression", self.first], outcomes=self.fresh())
        self.assertEqual(caught.exception.code, 2)

    def test_list_prints_id_status_and_request(self):
        code, output = run_tool(["--list"], self.fresh("ok"))
        self.assertEqual(code, 0)
        first = probes.all_probes()[0]
        line = next(line for line in output.splitlines() if line.startswith(first.id + " "))
        self.assertEqual(line.split(), [first.id, "ok", *first.request.split()])
        self.assertRegex(output, r"\ntotal +1 ")

    def test_an_unreadable_or_inconsistent_baseline_is_a_usage_error(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(run_tool(["--baseline", str(self.path)], self.fresh())[0], 2)
        document = check_probes.document(mf.__version__, self.statuses)
        document["totals"] = {**document["totals"], "ok": 1}
        self.path.write_text(check_probes.dumps(document), encoding="utf-8")
        self.assertEqual(run_tool(["--baseline", str(self.path)], self.fresh())[0], 2)
        self.assertEqual(run_tool(["--baseline", str(self.path.with_name("absent.json"))], self.fresh())[0], 2)


if __name__ == "__main__":
    unittest.main()
