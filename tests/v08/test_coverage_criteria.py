"""The coverage gate's eight criteria, the adapter protocol and the pilot adapter (row 27.25, continued fractions).

The criteria tests build synthetic rows, examples, Facts and stand-in adapters; the facts tests read this checkout's
documentation/ and tools/check_sympy.py, or a synthetic documentation directory. Only the scan of the frozen records
reads the local docs/v08/reference_examples.json, and it skips where that file is absent, as in CI and the distribution
harness. The tests of same_value and of the pilot adapter need SymPy 1.14 and skip without it, as in CI's core job;
everything else runs without SymPy.
"""

import ast
import collections
import dataclasses
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock
import warnings

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))  # The coverage gate is tooling beside the scripts, not a package.

import check_sympy  # noqa: E402
from coverage_gate import criteria  # noqa: E402
from coverage_gate import inventory as model  # noqa: E402
from coverage_gate.adapters import ntheory as ntheory_adapters  # noqa: E402
from coverage_gate.examples import Example, load_examples  # noqa: E402

import mathforge as mf  # noqa: E402
from mathforge import registry  # noqa: E402
from mathforge.areas.ntheory.continued_fraction import ContinuedFraction  # noqa: E402
from mathforge.capabilities import area_tool  # noqa: E402
from mathforge.registry_v07 import LEGACY_CATALOG  # noqa: E402

SYMPY = importlib.util.find_spec("sympy") is not None
NO_SYMPY = "SymPy 1.14, a development dependency, is not installed"
FROZEN_EXAMPLES = REPOSITORY / "docs" / "v08" / "reference_examples.json"  # Local data: CI does not have it.

Outcome = criteria.Outcome
ROW = "90.01"  # A synthetic row id that no adapter module registers.
OPERATIONS = ("op_alpha", "op_beta")  # Synthetic names, unknown to the registry, so each is its own public name.
VERIFIED = mf.operations.continued_fraction(mf.Rational(415, 93))  # A real Result: it round-trips and verifies.


def tampered():
    """VERIFIED with a forged value: it still round-trips, but verify() fails its mathematics check."""
    wrong = ContinuedFraction((4, 2, 6, 6))
    return dataclasses.replace(VERIFIED, value=wrong, verification=(),
                               steps=(dataclasses.replace(VERIFIED.steps[0], outputs=(wrong, ())),))


def make_row(operations=OPERATIONS, row_id=ROW):
    return model.Row(id=row_id, area="Synthetic area", subtopic="Synthetic subtopic", sympy_apis="f",
                     sympy_api_paths=("sympy.f",), v07_status="no", tier="T1", note="", plan_group="t1", wave="A",
                     family="A1", missing="", sympy_probe="f(1)", mathforge_operations=tuple(operations), status="no")


def make_examples(count, row_id=ROW):
    return [Example(id=f"{row_id}/{n:02d}", row=row_id, source=f"docstring:sympy.f:{n}", setup="", code=f"f({n})",
                    output=str(n), srepr=f"Integer({n})", kind="exact") for n in range(1, count + 1)]


def make_facts(**changes):
    """Facts under which every criterion but depth and json holds for OPERATIONS."""
    names = frozenset(OPERATIONS)
    facts = criteria.Facts(operations=names, checkers=names, mcp=names, documented=names, differential=names,
                           speed={name: 1.5 for name in OPERATIONS})
    return dataclasses.replace(facts, **changes)


def adapter_giving(replies=None, default=(Outcome.MATCH, VERIFIED)):
    """A stand-in adapter: the reply listed for an example id, else default."""
    replies = replies or {}
    return lambda example: replies.get(example.id, default)


MATCHING = adapter_giving()


def evaluate(*, row=None, examples=None, facts=None, adapter=MATCHING):
    """evaluate_row with ADAPTERS holding adapter for the row (no adapter at all when adapter is None)."""
    row = row or make_row()
    examples = make_examples(10) if examples is None else examples
    with mock.patch.dict(criteria.ADAPTERS, {} if adapter is None else {row.id: adapter}):
        return criteria.evaluate_row(row, examples, facts or make_facts())


def frozen_srepr(value):
    """srepr(value) as the frozen examples file holds it: the harvester's worker renumbers the Dummy indices from 0 in
    order of appearance, so the Dummies of the value rebuilt from it are never an adapter's."""
    import sympy
    numbers = {}
    return re.sub(r"dummy_index=(\d+)",
                  lambda match: f"dummy_index={numbers.setdefault(match.group(1), len(numbers))}", sympy.srepr(value))


def frozen_example(value):
    """A synthetic example that records value as the harvester records it: its frozen srepr and its str."""
    return Example(id=f"{ROW}/01", row=ROW, source="guide:synthetic.rst:1", setup="", code="value",
                   output=str(value), srepr=frozen_srepr(value), kind="exact")


class Printed:
    """A stand-in candidate that prints as the given text and is nothing else."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class CriteriaTests(unittest.TestCase):
    def test_all_eight_make_yes(self):
        self.assertEqual(criteria.CRITERIA,
                         ("capabilities", "checker", "json", "mcp", "docs", "differential", "speed", "depth"))
        self.assertEqual((criteria.DEPTH_THRESHOLD, criteria.SPEED_LIMIT), (0.8, 3.0))
        verdict = evaluate()
        self.assertEqual(verdict.status, "yes")
        self.assertEqual(list(verdict.criteria.items()), [(name, True) for name in criteria.CRITERIA])
        self.assertEqual((verdict.row, verdict.matched, verdict.examples, verdict.details), (ROW, 10, 10, ()))

    def test_each_criterion_can_fail_alone(self):
        facts = make_facts()
        only_alpha = frozenset({"op_alpha"})
        broken = {
            "capabilities": dict(facts=dataclasses.replace(facts, operations=only_alpha)),
            "checker": dict(facts=dataclasses.replace(facts, checkers=only_alpha)),
            # One MATCH carries a Result that fails verification: it does not count, but 9 of 10 still meet depth.
            "json": dict(adapter=adapter_giving({f"{ROW}/04": (Outcome.MATCH, tampered())})),
            "mcp": dict(facts=dataclasses.replace(facts, mcp=only_alpha)),
            "docs": dict(facts=dataclasses.replace(facts, documented=only_alpha)),
            "differential": dict(facts=dataclasses.replace(facts, differential=only_alpha)),
            "speed": dict(facts=dataclasses.replace(facts, speed={"op_alpha": 1.5, "op_beta": 3.5})),
            "depth": dict(adapter=adapter_giving({f"{ROW}/{n:02d}": (Outcome.MISMATCH, VERIFIED) for n in (2, 5, 9)})),
        }
        self.assertEqual(tuple(broken), criteria.CRITERIA)
        explanations = {"capabilities": "unknown operation: op_beta", "checker": "checker: ", "json": f"json: {ROW}/04: ",
                        "mcp": "mcp: ", "docs": "docs: ", "differential": "differential: ", "speed": "speed: ",
                        "depth": "depth: 7 of 10 "}
        for criterion, change in broken.items():
            with self.subTest(criterion):
                verdict = evaluate(**change)
                self.assertEqual(dict(verdict.criteria), {name: name != criterion for name in criteria.CRITERIA})
                # The plan's status rule: criterion 1 decides between partial and no, so a row whose operations
                # capabilities() does not list is "no" even when the other seven criteria hold.
                self.assertEqual(verdict.status, "no" if criterion == "capabilities" else "partial")
                self.assertTrue(any(line.startswith(explanations[criterion]) for line in verdict.details),
                                verdict.details)
                if criterion not in ("capabilities", "json", "depth"):
                    self.assertTrue(any("op_beta" in line for line in verdict.details), verdict.details)
        self.assertEqual(evaluate(**broken["json"]).matched, 9)
        self.assertIn("speed: ratio above 3.0: op_beta (3.5)", evaluate(**broken["speed"]).details)

    def test_no_operations_is_no(self):
        verdict = evaluate(row=make_row(()))
        self.assertEqual(verdict.status, "no")
        # Without operations, none of the per-operation criteria holds: there is nothing for them to hold for.
        for name in ("capabilities", "checker", "mcp", "docs", "differential", "speed"):
            self.assertFalse(verdict.criteria[name], name)
        self.assertIn("capabilities: the row lists no MathForge operations", verdict.details)

    def test_depth_threshold(self):
        def depth(matches, count):
            examples = make_examples(count)
            replies = {example.id: (Outcome.MISMATCH, VERIFIED) for example in examples[matches:]}
            verdict = evaluate(examples=examples, adapter=adapter_giving(replies))
            return verdict.criteria["depth"], verdict.matched, verdict.examples

        self.assertEqual(depth(8, 10), (True, 8, 10))
        self.assertEqual(depth(7, 10), (False, 7, 10))
        self.assertEqual(depth(6, 7), (True, 6, 7))  # A short set of 7: ceil(0.8 x 7) = 6.
        self.assertEqual(depth(5, 7), (False, 5, 7))
        self.assertEqual(depth(16, 20), (True, 16, 20))
        self.assertEqual(depth(15, 20), (False, 15, 20))
        without_adapter = evaluate(adapter=None)
        self.assertFalse(without_adapter.criteria["depth"])
        self.assertIn(f"depth: no adapter for row {ROW}", without_adapter.details)
        self.assertFalse(without_adapter.criteria["json"])  # Nothing was computed, so nothing round-tripped.
        without_examples = evaluate(examples=[])
        self.assertFalse(without_examples.criteria["depth"])  # Never vacuously: 0 of 0 is not 80%.
        self.assertIn("depth: the row has no reference examples", without_examples.details)

    def test_only_verified_matches_count(self):
        def failing(example):
            raise ValueError("boom")

        replies = {f"{ROW}/01": (Outcome.UNEVALUATED, None), f"{ROW}/02": (Outcome.UNVERIFIED, None),
                   f"{ROW}/03": (Outcome.ERROR, None), f"{ROW}/05": (Outcome.MATCH, None),
                   f"{ROW}/06": ("match", VERIFIED), f"{ROW}/07": (Outcome.MATCH, "a Result")}
        adapter = adapter_giving(replies)
        verdict = evaluate(adapter=lambda example: failing(example) if example.id == f"{ROW}/04" else adapter(example))
        self.assertEqual((verdict.matched, verdict.examples), (3, 10))  # 08, 09 and 10.
        self.assertFalse(verdict.criteria["depth"])
        self.assertTrue(verdict.criteria["json"])  # The Results returned (08-10) round-trip and verify.
        for line in (f"{ROW}/01: unevaluated", f"{ROW}/02: unverified", f"{ROW}/03: error",
                     f"{ROW}/04: error: ValueError: boom", f"{ROW}/05: match, not counted: no Result",
                     "depth: 3 of 10 examples match; 8 needed"):
            self.assertIn(line, verdict.details)
        for number in ("06", "07"):  # A reply outside the protocol is an error, never a match.
            self.assertTrue(any(line.startswith(f"{ROW}/{number}: error: the adapter returned ")
                                for line in verdict.details), verdict.details)
        with self.assertRaises(ValueError):  # Examples of another row are a caller's mistake.
            evaluate(examples=make_examples(3, row_id="90.02"))

    def test_unknown_operation_is_reported(self):
        verdict = evaluate(row=make_row(("op_alpha", "op_missing")))
        self.assertEqual(verdict.details[0], "unknown operation: op_missing")
        self.assertFalse(verdict.criteria["capabilities"])
        self.assertEqual(verdict.status, "no")
        # The same with the real facts: a misspelled operation name is reported by name, not silently unmet.
        facts = criteria.gather_facts(REPOSITORY, speed_file=None)
        verdict = criteria.evaluate_row(make_row(("continued_fraction", "continued_fractoin")), [], facts)
        self.assertIn("unknown operation: continued_fractoin", verdict.details)
        self.assertNotIn("unknown operation: continued_fraction", verdict.details)
        self.assertFalse(verdict.criteria["capabilities"])

    def test_missing_speed_file_fails_speed_only(self):
        verdict = evaluate(facts=make_facts(speed={}))
        self.assertEqual(dict(verdict.criteria), {name: name != "speed" for name in criteria.CRITERIA})
        self.assertEqual(verdict.status, "partial")
        self.assertIn("speed: no speed ratio: op_alpha, op_beta", verdict.details)
        with tempfile.TemporaryDirectory() as directory:
            speed_file = Path(directory) / "latest.json"
            self.assertEqual(dict(criteria.gather_facts(REPOSITORY, speed_file=None).speed), {})
            self.assertEqual(dict(criteria.gather_facts(REPOSITORY, speed_file=speed_file).speed), {})  # Absent.
            speed_file.write_text(json.dumps({"suite": "phase 1", "operations": {"continued_fraction": 2.5,
                                                                                 "integer_gcd": 3}}), encoding="utf-8")
            self.assertEqual(dict(criteria.gather_facts(REPOSITORY, speed_file=speed_file).speed),
                             {"continued_fraction": 2.5, "integer_gcd": 3.0})
            for document in ([], {"ratios": {}}, {"operations": []}, {"operations": {"integer_gcd": "fast"}},
                             {"operations": {"integer_gcd": True}}, {"operations": {"integer_gcd": 0}},
                             {"operations": {"integer_gcd": float("inf")}}):
                with self.subTest(document=document):
                    speed_file.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError) as raised:  # A malformed file never reads as absent.
                        criteria.gather_facts(REPOSITORY, speed_file=speed_file)
                    self.assertIn(str(speed_file), str(raised.exception))

    def test_exempt_operations_meet_speed_by_their_budget(self):
        # Spec H2: an operation that SymPy has no counterpart to time against (verify, proof records, the proven
        # is_prime) is exempt from the ratio gate and has an absolute time budget instead.
        kept = criteria.SpeedExemption("no SymPy counterpart: verify", True)
        missed = criteria.SpeedExemption("no SymPy counterpart: proof record", False)
        verdict = evaluate(facts=make_facts(speed={"op_alpha": 3.0, "op_beta": kept}))  # 3.0 is still within the limit.
        self.assertEqual((verdict.status, verdict.criteria["speed"], verdict.details), ("yes", True, ()))
        verdict = evaluate(facts=make_facts(speed={"op_alpha": 1.5, "op_beta": missed}))
        self.assertEqual(dict(verdict.criteria), {name: name != "speed" for name in criteria.CRITERIA})
        self.assertEqual(verdict.details,
                         ("speed: absolute time budget not met: op_beta (exempt: no SymPy counterpart: proof record)",))
        # In the speed file an operation has either form; the exemption object is checked strictly.
        with tempfile.TemporaryDirectory() as directory:
            speed_file = Path(directory) / "latest.json"
            speed_file.write_text(json.dumps({"operations": {
                "continued_fraction": 2.5, "verify": {"exempt": "no SymPy counterpart", "budget_met": True},
                "integer_is_prime": {"budget_met": False, "exempt": "proven primality (spec H2)"}}}), encoding="utf-8")
            self.assertEqual(dict(criteria.gather_facts(REPOSITORY, speed_file=speed_file).speed),
                             {"continued_fraction": 2.5,
                              "verify": criteria.SpeedExemption("no SymPy counterpart", True),
                              "integer_is_prime": criteria.SpeedExemption("proven primality (spec H2)", False)})
            for entry in ({"exempt": "no SymPy counterpart"}, {"budget_met": True}, {},
                          {"exempt": "no SymPy counterpart", "budget_met": True, "ratio": 1.0},
                          {"exempt": "", "budget_met": True}, {"exempt": " \t", "budget_met": True},
                          {"exempt": None, "budget_met": True}, {"exempt": ["verify"], "budget_met": True},
                          {"exempt": "no SymPy counterpart", "budget_met": 1},
                          {"exempt": "no SymPy counterpart", "budget_met": "true"},
                          {"exempt": "no SymPy counterpart", "budget_met": None},
                          ["no SymPy counterpart", True], "exempt", None):
                with self.subTest(entry=entry):
                    speed_file.write_text(json.dumps({"operations": {"verify": entry}}), encoding="utf-8")
                    with self.assertRaises(ValueError) as raised:
                        criteria.gather_facts(REPOSITORY, speed_file=speed_file)
                    for named in (str(speed_file), "verify"):
                        self.assertIn(named, str(raised.exception))

    def test_an_example_over_the_time_budget_is_an_error_and_the_gate_goes_on(self):
        release = threading.Event()
        self.addCleanup(release.set)  # The stuck call cannot be stopped; this lets its thread end with the test.

        def sleepy(example):
            if example.id == f"{ROW}/03":
                release.wait(30)  # Sleeps far past the budget.
            return Outcome.MATCH, VERIFIED

        self.assertEqual(criteria.EXAMPLE_BUDGET, 60.0)
        started = time.monotonic()
        with mock.patch.object(criteria, "EXAMPLE_BUDGET", 0.2):
            verdict = evaluate(adapter=sleepy)
        self.assertLess(time.monotonic() - started, 10)  # The gate did not wait for the stuck call.
        self.assertEqual((verdict.matched, verdict.examples), (9, 10))
        self.assertIn(f"{ROW}/03: error: time budget exceeded (0.2 s)", verdict.details)
        self.assertTrue(verdict.criteria["depth"])  # The examples after it ran: 9 of 10 still meet depth.
        # Released, the adapter answers at once: every call ends within the budget and keeps its outcome. What an
        # adapter raises in its thread is still reported as that example's failure.
        release.set()
        with mock.patch.object(criteria, "EXAMPLE_BUDGET", 5.0):
            self.assertEqual(evaluate(adapter=sleepy).matched, 10)
            failing = evaluate(adapter=lambda example: 1 / 0)
        self.assertIn(f"{ROW}/01: error: ZeroDivisionError: division by zero", failing.details)


class FactsTests(unittest.TestCase):
    def test_covered_operations_are_registered(self):
        registered = set(registry.specs()) | {entry.name for entry in LEGACY_CATALOG}
        self.assertIsInstance(check_sympy.COVERED_OPERATIONS, frozenset)
        self.assertEqual(sorted(check_sympy.COVERED_OPERATIONS - registered), [])
        self.assertIn("continued_fraction", check_sympy.COVERED_OPERATIONS)

    def test_covered_operations_follow_the_compared_calls(self):
        """COVERED_OPERATIONS names exactly the operations that the seeded groups of check_sympy.py call and compare.

        The calls that only build inputs (values and the rational_function constructor) compare nothing; mf.diff is
        the flat name of the differentiate operation; an area module's function is its registry operation."""
        tree = ast.parse((REPOSITORY / "tools" / "check_sympy.py").read_text(encoding="utf-8"))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id == "mf":
                    called.add(node.func.attr)
                elif (isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name)
                      and owner.value.id == "mf" and owner.attr in registry.AREAS):
                    called.add(next(spec.name for spec in registry.area_operations(owner.attr)
                                    if spec.public_name == node.func.attr))
        inputs = {"Symbol", "Rational", "Polynomial", "Matrix", "sqrt", "rational_function"}
        self.assertEqual({{"diff": "differentiate"}.get(name, name) for name in called - inputs},
                         set(check_sympy.COVERED_OPERATIONS))

    def test_gather_facts(self):
        facts = criteria.gather_facts(REPOSITORY, speed_file=None)
        entries = mf.capabilities()["operations"]
        self.assertEqual(facts.operations, {entry["name"] for entry in entries})
        self.assertEqual(len(facts.operations), len(entries))
        legacy = {entry.name for entry in LEGACY_CATALOG}
        self.assertEqual(facts.checkers, set(registry.specs()) | legacy)  # Legacy operations count (ruling 1).
        self.assertEqual(facts.mcp, {spec.name for spec in registry.specs().values()
                                     if spec.version == 5 or area_tool(spec) is not None}
                         | {entry.name for entry in LEGACY_CATALOG if entry.tool})
        self.assertTrue({"continued_fraction", "integer_gcd", "det"} <= facts.mcp)
        self.assertNotIn("rank", facts.mcp)  # A legacy operation without an MCP tool of its own.
        self.assertEqual(facts.differential, check_sympy.COVERED_OPERATIONS)
        self.assertIn("continued_fraction", facts.documented)  # documentation/areas.md: mf.ntheory.continued_fraction
        public = {criteria.public_name(name) for name in facts.operations}
        self.assertLessEqual(facts.documented, public)
        self.assertEqual(dict(facts.speed), {})

    def test_documented_names_come_from_python_fences(self):
        page = textwrap.dedent('''\
            # A page

            ```python
            import mathforge as mf
            from mathforge import ntheory
            mf.operations.solve(problem)
            result = mf.ntheory.continued_fraction(mf.Rational(1, 2))
            ntheory.continued_fraction(3)
            series = mf.taylor(function, x, 0, 3)  # function_taylor's public name
            # mf.det(matrix) is only a comment
            text = "mf.rank(matrix)"
            inverse(matrix)
            mf.verify(result)
            ```

            ```text
            mf.nullspace(matrix)
            ```
            ''')
        with tempfile.TemporaryDirectory() as directory:
            documentation = Path(directory) / "documentation"
            documentation.mkdir()
            (documentation / "page.md").write_text(page, encoding="utf-8")
            facts = criteria.gather_facts(Path(directory), speed_file=None)
        self.assertEqual(facts.documented, {"solve", "continued_fraction", "taylor", "verify"})
        self.assertEqual(criteria.public_name("function_taylor"), "taylor")
        self.assertEqual(criteria.public_name("rank"), "rank")

    def test_imports_without_sympy(self):
        code = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; import coverage_gate.criteria; "
                "import coverage_gate.adapters.ntheory; "
                "assert not [name for name in sys.modules if name.split('.')[0] == 'sympy'], 'sympy imported'")
        completed = subprocess.run([sys.executable, "-c", code, str(REPOSITORY / "src"), str(REPOSITORY / "tools")],
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class AdapterPackageTests(unittest.TestCase):
    def test_every_adapter_module_is_imported(self):
        """Importing criteria imports every module of coverage_gate/adapters/: an area module that the package's
        __init__ leaves out would never register its adapters, and its rows would silently have none."""
        code = ("import json, pkgutil, sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; import coverage_gate.criteria; "
                "import coverage_gate.adapters as package; "
                "print(json.dumps([sorted(module.name for module in pkgutil.iter_modules(package.__path__)), "
                "sorted(name.split('.')[2] for name in sys.modules "
                "if name.startswith('coverage_gate.adapters.') and name.count('.') == 2)]))")
        completed = subprocess.run([sys.executable, "-c", code, str(REPOSITORY / "src"), str(REPOSITORY / "tools")],
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        on_disk, imported = json.loads(completed.stdout)
        self.assertIn("ntheory", on_disk)
        self.assertEqual(imported, on_disk)


@unittest.skipUnless(SYMPY, NO_SYMPY)
class SameValueTests(unittest.TestCase):
    def test_same_value(self):
        import sympy
        x = sympy.Symbol("x")
        srepr = sympy.srepr
        self.assertTrue(criteria.same_value("Rational(1, 2)", Fraction(1, 2)))
        self.assertTrue(criteria.same_value("Integer(4)", 4))
        self.assertTrue(criteria.same_value("[Integer(4), Integer(2)]", (4, 2)))
        self.assertTrue(criteria.same_value("{2: 3, 5: 1}", {5: 1, 2: 3}))
        self.assertTrue(criteria.same_value(srepr((x + 1)**2), x**2 + 2*x + 1))  # Not structural: simplify(a - b).
        self.assertTrue(criteria.same_value(srepr(sympy.Integer(1)), sympy.sin(x)**2 + sympy.cos(x)**2))
        self.assertTrue(criteria.same_value("[Integer(1), [Integer(2)]]", [1, [2]]))
        self.assertFalse(criteria.same_value("Rational(1, 2)", Fraction(1, 3)))
        self.assertFalse(criteria.same_value(srepr(x + 1), x + 2))
        self.assertFalse(criteria.same_value("[Integer(4), Integer(2)]", (4,)))
        self.assertFalse(criteria.same_value("[Integer(4), Integer(2)]", 4))
        self.assertFalse(criteria.same_value("{2: 3}", {2: 4}))
        self.assertFalse(criteria.same_value("Rational(1, 2)", mf.Rational(1, 2)))  # Adapters convert MathForge values.
        self.assertFalse(criteria.same_value("Rational(1, 2)", "1/2"))
        with self.assertRaises(ValueError):
            criteria.same_value("NoSuchClass(1)", 1)

    def test_dummies_match_up_to_a_consistent_renaming(self):
        import sympy
        from sympy import Dummy, Function, ImageSet, Integral, Lambda, RootSum, S, Symbol, Tuple, exp, log, pi, sin
        x, f = Symbol("x"), Function("f")

        def image_set(n, shift=pi / 2):
            return ImageSet(Lambda(n, 2 * n * pi + shift), S.Integers)

        def root_sum(z, i):
            return RootSum(4 * z**2 + 1, Lambda(i, i * log(2 * i + exp(x))))

        def integral(t):
            return Integral(f(t) / t, (t, 1, x))

        # SymPy's records, and their adapter-side twins with Dummies of their own, also under other names. simplify
        # may not run: the renaming alone has to make them equal.
        records = {"ImageSet": sympy.solveset(sin(x) - 1, x, S.Reals),
                   "RootSum": sympy.integrate(exp(x) / (1 + exp(2 * x)), x),
                   "Integral": integral(Dummy("t"))}
        twins = {"ImageSet": (image_set(Dummy("n", integer=True)), image_set(Dummy("k", integer=True))),
                 "RootSum": (root_sum(Dummy("z"), Dummy("i")), root_sum(Dummy("w"), Dummy("a"))),
                 "Integral": (integral(Dummy("t")), integral(Dummy("u")))}
        self.assertEqual(frozen_srepr(records["ImageSet"]), frozen_srepr(twins["ImageSet"][0]))  # The same record.
        with mock.patch.object(sympy, "simplify", side_effect=AssertionError("simplify ran")):
            for name, record in records.items():
                for twin in twins[name]:
                    with self.subTest(name, twin=twin):
                        self.assertIn("dummy_index=0", frozen_srepr(record))
                        self.assertTrue(criteria.same_value(frozen_srepr(record), twin))
        # Genuinely different values still differ: another set, a Dummy of other assumptions (a renaming keeps them),
        # a Symbol for the Dummy, another integrand (simplify finds no zero), a renaming that is not consistent
        # ((a, b) -> a - b is not (c, d) -> d - c), two Dummies for one.
        a, b, c, d = Dummy("a"), Dummy("b"), Dummy("c"), Dummy("d")
        for record, value in ((records["ImageSet"], image_set(Dummy("n", integer=True), shift=pi)),
                              (records["ImageSet"], image_set(Dummy("n"))),
                              (records["ImageSet"], image_set(Symbol("n", integer=True))),
                              (records["Integral"], Integral(f(c) * c, (c, 1, x))),
                              (Lambda((a, b), a - b), Lambda((c, d), d - c)),
                              (Tuple(a, b), Tuple(c, c))):
            with self.subTest(record=record, value=value):
                self.assertFalse(criteria.same_value(frozen_srepr(record), value))
        self.assertTrue(criteria.same_value(frozen_srepr(Lambda((a, b), a - b)), Lambda((c, d), c - d)))

    def test_records_of_classes_outside_the_top_level_rebuild(self):
        import sympy
        from sympy import MatrixSymbol, Piecewise, Symbol
        x = Symbol("x")
        # A Piecewise prints its pieces as ExprCondPair, a MatrixSymbol its name as Str: neither is a top-level name.
        piecewise, matrix = frozen_example(Piecewise((x, x < 1), (1, True))), frozen_example(MatrixSymbol("A", 2, 2))
        for record, name in ((piecewise, "ExprCondPair"), (matrix, "Str")):
            self.assertIn(f"{name}(", record.srepr)
            self.assertNotIn(name, vars(sympy))
        self.assertTrue(criteria.same_value(piecewise, Piecewise((x, x < 1), (1, True))))
        self.assertFalse(criteria.same_value(piecewise, Piecewise((x, x < 2), (1, True))))
        self.assertFalse(criteria.same_value(piecewise, Printed(piecewise.output)))  # Rebuilt: compared as a value.
        self.assertTrue(criteria.same_value(matrix, MatrixSymbol("A", 2, 2)))
        self.assertFalse(criteria.same_value(matrix, MatrixSymbol("B", 2, 2)))
        self.assertTrue(criteria.same_value(matrix.srepr, MatrixSymbol("A", 2, 2)))  # The srepr alone rebuilds too.

    def test_records_that_do_not_rebuild_compare_their_printed_output_strictly(self):
        from sympy import QQ, Integer, Mul, Rational, Symbol
        # A class prints as <'sympy.core.numbers.Integer'>, which is not Python.
        type_record = frozen_example(type(Integer(2)))
        self.assertEqual((type_record.srepr, type_record.output),
                         ("<'sympy.core.numbers.Integer'>", "<class 'sympy.core.numbers.Integer'>"))
        self.assertTrue(criteria.same_value(type_record, Integer))
        self.assertFalse(criteria.same_value(type_record, Rational))
        # A QQ element prints as 5/3, which Python evaluates to a float: the srepr evaluates to something else.
        element = frozen_example(QQ(5, 3))
        self.assertEqual((element.srepr, element.output), ("5/3", "5/3"))
        for candidate, verdict in ((Rational(5, 3), True), (Fraction(5, 3), True), (Rational(10, 6), True),
                                   (5 / 3, False), (Rational(5, 4), False)):
            with self.subTest(candidate=candidate):
                self.assertIs(criteria.same_value(element, candidate), verdict)
        # Strict: an unevaluated product rebuilds as the evaluated sum, so the record is compared by its output; a
        # candidate of the same value that prints otherwise does not match.
        x = Symbol("x")
        unevaluated = frozen_example(Mul(Rational(1, 2), x + 2, evaluate=False))
        self.assertEqual(unevaluated.output, "(x + 2)/2")
        self.assertFalse(criteria.same_value(unevaluated, x / 2 + 1))
        self.assertTrue(criteria.same_value(unevaluated, Mul(Rational(1, 2), x + 2, evaluate=False)))
        # An srepr alone has no output to fall back on.
        with self.assertRaises(ValueError) as raised:
            criteria.same_value(element.srepr, Rational(5, 3))
        self.assertIn("srepr prints other text", str(raised.exception))

    def test_the_rebuild_namespace_keeps_the_shallowest_object_and_pins_its_clashes(self):
        import sympy
        from sympy.diffgeom import Point as DiffgeomPoint
        from sympy.physics.vector import Vector as PhysicsVector
        names, clashes = criteria._rebuild_names()
        self.assertIs(criteria._rebuild_names(), criteria._rebuild_names())  # Built once.
        self.assertEqual(dict(clashes), dict(criteria.REBUILD_CLASHES))
        # The package's own names first, then the shallowest module: geometry's Point, not diffgeom's; sympy.vector's
        # Vector, not physics.vector's.
        self.assertEqual((clashes["Point"], clashes["Vector"]), ("sympy", "sympy.vector"))
        self.assertIs(names["Point"], sympy.Point)
        self.assertIsNot(names["Point"], DiffgeomPoint)
        self.assertIs(names["Vector"], sympy.vector.Vector)
        self.assertIsNot(names["Vector"], PhysicsVector)
        for name, module in (("Str", "sympy.core.symbol"), ("ExprCondPair", "sympy.functions.elementary.piecewise"),
                             ("VectorAdd", "sympy.vector.vector"),
                             ("DomainMatrix", "sympy.polys.matrices.domainmatrix")):
            with self.subTest(name):
                self.assertNotIn(name, vars(sympy))
                self.assertEqual((names[name].__name__, names[name].__module__), (name, module))
        self.assertNotIn("x", names)  # The symbols of sympy.abc are objects, not classes or functions.

    def test_every_renaming_is_tried_when_an_order_follows_the_dummies(self):
        from sympy import Dummy, Tuple, sin
        # The Add sorts sin(_a) + sin(_a) by the Dummies' indices: the record's first Dummy is created first, the
        # twin's last, so pairing the Dummies in order of first appearance swaps them.
        first, second = Dummy("a"), Dummy("a")
        record = frozen_srepr(Tuple(sin(first) + sin(second), first))
        other = Dummy("a")
        twin_of_first = Dummy("a")
        twin = Tuple(sin(twin_of_first) + sin(other), twin_of_first)
        with mock.patch.object(criteria, "_RENAMINGS", 0):  # Only the renaming in order of first appearance
            self.assertFalse(criteria.same_value(record, twin))
        self.assertTrue(criteria.same_value(record, twin))


@unittest.skipUnless(SYMPY, NO_SYMPY)
class FrozenRecordTests(unittest.TestCase):
    def test_every_frozen_record_is_rebuilt_or_compared_by_its_output(self):
        """Every record of the local examples file either rebuilds SymPy's value from its srepr or falls back to its
        printed output, and either way compares equal to what it records; the counts are printed."""
        if not FROZEN_EXAMPLES.is_file():
            self.skipTest(f"{FROZEN_EXAMPLES} is local data that is never committed")
        import sympy
        examples = [example for items in load_examples(FROZEN_EXAMPLES).values() for example in items]
        reasons = collections.Counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Some records rebuild deprecated forms, such as non-Expr matrix entries.
            for example in examples:
                value, problem = criteria._reference(sympy, example.srepr, example.output)
                reasons[problem or "rebuilt"] += 1
                with self.subTest(example.id, problem=problem):
                    self.assertTrue(criteria.same_value(example, Printed(example.output) if problem else value))
        rebuilt = reasons.pop("rebuilt", 0)
        self.assertEqual(rebuilt + sum(reasons.values()), len(examples))
        print(f"\nfrozen records: {rebuilt} of {len(examples)} rebuilt from their srepr; compared by their output: "
              + ", ".join(f"{count} {reason}" for reason, count in sorted(reasons.items())), flush=True)


@unittest.skipUnless(SYMPY, NO_SYMPY)
class PilotAdapterTests(unittest.TestCase):
    PERIODIC = "from sympy.ntheory.continued_fraction import continued_fraction_periodic as cf_p"

    def record(self, code, setup="", number=1):
        """The example as the harvester records it: the value of code after `from sympy import *` and setup."""
        import sympy
        namespace = {}
        exec("from sympy import *", namespace)
        exec(setup, namespace)
        value = eval(code, namespace)
        return Example(id=f"27.25/{number:02d}", row="27.25", source=f"docstring:sympy.continued_fraction:{number}",
                       setup=setup, code=code, output=str(value), srepr=sympy.srepr(value), kind="exact")

    def test_continued_fraction_adapter(self):
        self.assertIs(criteria.ADAPTERS["27.25"], ntheory_adapters.continued_fraction)
        matching = (("continued_fraction(Rational(415, 93))", "", (4, 2, 6, 7)),
                    ("continued_fraction(3)", "", (3,)),
                    ("continued_fraction(Rational(-7, 3))", "", (-3, 1, 2)),
                    ("continued_fraction_periodic(4, 3, 0)", "", (1, 3)),
                    ("continued_fraction_periodic(4, 3, 49)", "", (3, 1, 2)),  # (4 + 7)/3: a square d is rational.
                    ("cf_p(3245, 10000)", self.PERIODIC, (0, 3, 12, 4, 13)))  # An alias from the setup.
        for code, setup, terms in matching:
            with self.subTest(code):
                outcome, result = criteria.ADAPTERS["27.25"](self.record(code, setup))
                self.assertIs(outcome, Outcome.MATCH)
                self.assertEqual(result.value.terms, terms)
                self.assertTrue(mf.verify(result).verified)
        # Quadratic irrationals (periodic expansions), an expression that is not an exact rational as written, and the
        # APIs that the operation does not compute are never matched.
        unevaluated = ("continued_fraction(sqrt(2))", "continued_fraction((1 + 2*sqrt(3))/5)",
                       "continued_fraction_periodic(3, 2, 7)", "continued_fraction((1 + sqrt(2))*(1 - sqrt(2)))",
                       "list(continued_fraction_convergents([0, 2, 1, 2]))", "continued_fraction_reduce([1, 2, 3, 4, 5])")
        for code in unevaluated:
            with self.subTest(code):
                outcome, result = criteria.ADAPTERS["27.25"](self.record(code))
                self.assertIs(outcome, Outcome.UNEVALUATED)
                self.assertIsNone(result)
        # A recorded answer that differs is a mismatch: the comparison is real.
        wrong = dataclasses.replace(self.record("continued_fraction(Rational(415, 93))"),
                                    srepr="[Integer(4), Integer(2), Integer(6), Integer(6)]", output="[4, 2, 6, 6]")
        outcome, result = criteria.ADAPTERS["27.25"](wrong)
        self.assertIs(outcome, Outcome.MISMATCH)
        self.assertEqual(result.value.terms, (4, 2, 6, 7))

    def test_pilot_row_end_to_end(self):
        codes = ("continued_fraction(Rational(415, 93))", "continued_fraction_periodic(4, 3, 0)",
                 "continued_fraction_periodic(4, 3, 49)", "continued_fraction(sqrt(2))",
                 "continued_fraction_periodic(3, 2, 7)", "continued_fraction_reduce([1, 2, 3, 4, 5])")
        examples = [self.record(code, number=number) for number, code in enumerate(codes, 1)]
        row = make_row(("continued_fraction",), row_id="27.25")
        verdict = criteria.evaluate_row(row, examples, criteria.gather_facts(REPOSITORY, speed_file=None))
        self.assertEqual((verdict.matched, verdict.examples), (3, 6))
        self.assertEqual(dict(verdict.criteria), {"capabilities": True, "checker": True, "json": True, "mcp": True,
                                                  "docs": True, "differential": True, "speed": False, "depth": False})
        self.assertEqual(verdict.status, "partial")
        self.assertIn("depth: 3 of 6 examples match; 5 needed", verdict.details)


if __name__ == "__main__":
    unittest.main()
