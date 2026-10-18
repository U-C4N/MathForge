"""The coverage gate's eight criteria, the adapter protocol and the pilot adapter (row 27.25, continued fractions).

The criteria tests build synthetic rows, examples, Facts and stand-in adapters; the facts tests read this checkout's
documentation/ and tools/check_sympy.py, or a synthetic documentation directory. None reads the local docs/ inventory or
examples, which neither CI nor the distribution harness has. The tests of same_value and of the pilot adapter need SymPy
1.14 and skip without it, as in CI's core job; everything else runs without SymPy.
"""

import ast
import dataclasses
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))  # The coverage gate is tooling beside the scripts, not a package.

import check_sympy  # noqa: E402
from coverage_gate import criteria  # noqa: E402
from coverage_gate import inventory as model  # noqa: E402
from coverage_gate.adapters import ntheory as ntheory_adapters  # noqa: E402
from coverage_gate.examples import Example  # noqa: E402

import mathforge as mf  # noqa: E402
from mathforge import registry  # noqa: E402
from mathforge.areas.ntheory.continued_fraction import ContinuedFraction  # noqa: E402
from mathforge.capabilities import area_tool  # noqa: E402
from mathforge.registry_v07 import LEGACY_CATALOG  # noqa: E402

SYMPY = importlib.util.find_spec("sympy") is not None
NO_SYMPY = "SymPy 1.14, a development dependency, is not installed"

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
                                    srepr="[Integer(4), Integer(2), Integer(6), Integer(6)]")
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
