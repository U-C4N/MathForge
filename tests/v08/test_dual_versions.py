"""A v6 family that supersedes a v0.7 operation: new results are v6, declined inputs and older records stay v0.7.

Also the per-area family lists, the lazily imported checkers, the supersession rules of the registry build, the
generated wrappers of superseded operations, and the argument-kind checks that operations_v6.run makes.
"""

import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

import mathforge as mf
from mathforge import dispatch_v5, operations_v6, registry
from mathforge.contracts import CheckStatus, ExecutionStatus
from mathforge.mcp.boundary import call_tool
from mathforge.registry import Argument, OperationSpec

from . import demo_dual_checker, demo_kind_family
from . import demo_dual_family as demo

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SRC = Path(mf.__file__).resolve().parents[1]  # src/ in a checkout, site-packages in the distribution harness.
TOOL = Path(__file__).resolve().parents[2] / "tools" / "generate_api.py"
FAMILY = "mathforge._dual_versions_test_family"
PILOT = "mathforge.areas.ntheory.continued_fraction"
# The stored documents that hold v5 integer_gcd and v2 rref results.
DOCUMENTS = ("v2/objects.json", "v2/workspace.json", "v7/objects.json", "v7/workspace.json")


def results(value):
    """The Results inside a decoded document: the value itself, the items of a tuple, a workspace's objects."""
    if type(value) is mf.Result:
        return [value]
    if type(value) is tuple:
        return [result for item in value for result in results(item)]
    if type(value) is mf.Workspace:
        return [result for item in value.objects.values() for result in results(item)]
    return []


def outcome(call):
    """("value", v) or ("error", class, code, message), for comparing value functions."""
    try:
        return ("value", call())
    except Exception as error:
        return ("error", type(error), getattr(error, "code", None), str(error))


def probe(code):
    """Run code in a fresh isolated interpreter that imports the package under test; return its JSON output."""
    program = "import json, sys\nsys.path.insert(0, sys.argv[1])\n" + code
    output = subprocess.run([sys.executable, "-I", "-c", program, str(SRC)], capture_output=True, text=True,
                            timeout=120)
    if output.returncode:
        raise AssertionError(output.stderr)
    return json.loads(output.stdout)


def load_tool():
    spec = importlib.util.spec_from_file_location("generate_api_dual", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gcd_spec(**changes):
    """The demo's superseding integer_gcd spec, with some fields changed."""
    fields = dict(name="integer_gcd", area="ntheory", arguments=demo.V07_GCD.arguments, returns="int",
                  summary="Test.", scope="tests only", version=6)
    return OperationSpec(**(fields | changes))


RANK = OperationSpec("rank", "linalg", (Argument("A", "Matrix", None, "matrix"),), "int",
                     "Compute the exact rank of a rational matrix.", "tests only", version=6)


class DualVersionTest(unittest.TestCase):
    def family(self, specs, supersedes):
        """A throwaway testing family module, its own sys.modules entry for the rest of this test."""
        module = types.ModuleType(FAMILY)
        module.SPECS, module.TYPES, module.CHECKER, module.SUPERSEDES = specs, {}, demo_dual_checker, supersedes
        module.produce = demo.produce
        sys.modules[FAMILY] = module
        self.addCleanup(sys.modules.pop, FAMILY, None)
        return module

    def test_superseding_result_is_v6(self):
        with registry._testing_families(demo):
            result = mf.operations.integer_gcd(12, 18)
            self.assertEqual(result.steps[0].rule, "v6.integer_gcd")
            self.assertEqual(result.request.version, 6)
            self.assertEqual(result.value, 6)
            self.assertTrue(mf.verify(result).verified)
            text = mf.dumps(result)
            self.assertEqual(json.loads(text)["version"], 6)
            self.assertEqual(mf.dumps(mf.loads(text)), text)
            self.assertTrue(mf.verify(mf.loads(text)).verified)
            # The value functions keep their names and values, on the lean path and with explicit limits.
            self.assertEqual((mf.integer_gcd(12, 18), mf.ntheory.integer_gcd(-12, 18)), (6, 6))
            self.assertEqual(mf.integer_gcd(12, 18, limits=mf.ComputationLimits()), 6)
            self.assertTrue(registry.is_v6_operation("integer_gcd"))

    def test_declined_input_keeps_the_v07_record(self):
        inputs = ((0, 5), (True, 4), (1.5, 4), (4, "6"))
        expected = {pair: mf.dumps(mf.operations.integer_gcd(*pair)) for pair in inputs}
        values = {pair: outcome(lambda: mf.integer_gcd(*pair)) for pair in inputs}
        with registry._testing_families(demo):
            result = mf.operations.integer_gcd(0, 5)
            self.assertEqual(result.steps[0].rule, "v5.integer_gcd")
            self.assertEqual(mf.to_data(result)["version"], 5)
            self.assertTrue(mf.verify(result).verified)
            for pair in inputs:  # v0.7's records and errors stay: the family declines what v0.7 rejects.
                with self.subTest(arguments=pair):
                    self.assertEqual(mf.dumps(mf.operations.integer_gcd(*pair)), expected[pair])
                    self.assertEqual(outcome(lambda: mf.integer_gcd(*pair)), values[pair])

    def test_old_documents_unchanged(self):
        found = {"integer_gcd": 0, "rref": 0}
        with registry._testing_families(demo):
            for name in DOCUMENTS:
                with self.subTest(document=name):
                    text = (FIXTURES / name).read_text(encoding="utf-8")
                    value = mf.loads(text)
                    self.assertEqual(mf.dumps(value), text)
                    for result in results(value):
                        operation = result.request.operation if result.request is not None else None
                        if operation in found:
                            found[operation] += 1
                            self.assertEqual(mf.to_data(result)["version"], {"integer_gcd": 5, "rref": 2}[operation])
                            self.assertTrue(mf.verify(result).verified, operation)
            # The v5 view keeps the v0.7 family, with which dispatch_v5 decodes and verifies those records.
            self.assertIs(dispatch_v5.module_for("integer_gcd"), registry.family("integer_gcd", 5))
            self.assertEqual(registry.family("integer_gcd").__name__, "mathforge.number_theory")
            self.assertIs(registry.family("integer_gcd", 6), demo)
            self.assertIs(registry.checker("integer_gcd"), demo_dual_checker)
            self.assertIn("integer_gcd", registry.operation_arguments(5))
            self.assertEqual(registry.spec("integer_gcd").version, 6)
        self.assertTrue(all(found.values()), found)

    def test_mcp_paths_return_v6(self):
        def compute(a, b):
            return call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                   "arguments": {"a": mf.to_data(a), "b": mf.to_data(b)}})

        def area(a, b):
            return call_tool("mathforge.ntheory", {"operation": "integer_gcd", "arguments": {"a": a, "b": b}})

        with registry._testing_families(demo):
            for path in (compute, area):
                with self.subTest(path=path.__name__):
                    response = path(12, 18)
                    self.assertFalse(response.is_error, response.payload)
                    self.assertEqual(response.payload["result"]["version"], 6)
                    result = mf.from_data(response.payload["result"])
                    self.assertEqual((result.steps[0].rule, result.value), ("v6.integer_gcd", 6))
                    declined = path(0, 5)  # A declined input answers with the v0.7 envelope.
                    self.assertEqual(declined.payload["result"]["version"], 5, declined.payload)

    def test_run_dual_with_an_older_thunk(self):
        sentinel = mf.operations.integer_gcd(7, 7)
        with registry._testing_families(demo):
            self.assertIs(operations_v6.run_dual("integer_gcd", {"a": 0, "b": 5}, older=lambda: sentinel), sentinel)
            taken = operations_v6.run_dual("integer_gcd", {"a": 4, "b": 6}, older=lambda: self.fail("older ran"))
            self.assertEqual(taken.steps[0].rule, "v6.integer_gcd")
            # Without declinable, a decline is the producer's internal error, in a v6 failure.
            failure = operations_v6.run("integer_gcd", {"a": 0, "b": 5})
            self.assertEqual((failure.error.code, failure.request.version), ("internal_error", 6))
            self.assertNotIn(CheckStatus.FAILED, [check.status for check in mf.verify(failure).checks])
            self.assertEqual(mf.to_data(failure)["version"], 6)
            with self.assertRaises(operations_v6.Declined):
                operations_v6.run("integer_gcd", {"a": 0, "b": 5}, declinable=True)
            with self.assertRaises(operations_v6.Declined):  # Kinds that v0.7 reports itself are declined too.
                operations_v6.run("integer_gcd", {"a": True, "b": 5}, declinable=True)
        self.assertFalse(issubclass(operations_v6.Declined, mf.MathForgeError))
        # Once no family supersedes the operation, run_dual answers with older().
        self.assertEqual(operations_v6.run_dual("integer_gcd", {"a": 4, "b": 6}).steps[0].rule, "v5.integer_gcd")
        with self.assertRaises(TypeError):  # Only a v0.7 v5 operation has a default older.
            operations_v6.run_dual("continued_fraction", {"value": 1})

    def test_build_rejects_bad_supersession(self):
        a, b = demo.V07_GCD.arguments
        cases = [
            ("(a) not a v0.7 operation", (gcd_spec(name="demo_gcd"),), frozenset({"demo_gcd"}),
             f"^v6 family {FAMILY} supersedes demo_gcd, which is not a v0.7 operation$"),
            ("(a) a v6 operation", (gcd_spec(name="continued_fraction"),), frozenset({"continued_fraction"}),
             f"^v6 family {FAMILY} supersedes continued_fraction, which is not a v0.7 operation$"),
            ("(b) a v0.7 v5 operation", (gcd_spec(),), frozenset(), "^Duplicate operation: integer_gcd$"),
            ("(b) a legacy operation", (RANK,), frozenset(),
             "^Operation name collides with a legacy catalogue entry: rank$"),
            ("(c) argument order", (gcd_spec(arguments=(b, a)),), frozenset({"integer_gcd"}),
             f"^v6 family {FAMILY} supersedes integer_gcd with other arguments than v0.7's: "),
            ("(c) argument default", (gcd_spec(arguments=(a, Argument("b", "int", "0", "int"))),),
             frozenset({"integer_gcd"}), "with other arguments than v0.7's"),
            ("(c) argument names", (gcd_spec(arguments=(a, Argument("c", "int", None, "int"))),),
             frozenset({"integer_gcd"}), "with other arguments than v0.7's"),
            ("(c) legacy argument names", (OperationSpec("rank", "linalg", (Argument("M", "", None, "matrix"),),
                                                          "int", "Test.", "tests only", version=6),),
             frozenset({"rank"}), f"^v6 family {FAMILY} supersedes rank with other arguments than v0.7's: "),
            ("(d) public name", (gcd_spec(sdk="gcd"),), frozenset({"integer_gcd"}),
             f"^v6 family {FAMILY} supersedes integer_gcd under another public name than v0.7's: gcd instead of "
             "integer_gcd$"),
            ("(d) legacy public name", (OperationSpec("rank", "linalg", RANK.arguments, "int", "Test.", "tests only",
                                                      version=6, sdk="matrix_rank"),), frozenset({"rank"}),
             "^v6 family .* supersedes rank under another public name than v0.7's: matrix_rank instead of rank$"),
            ("not a frozenset", (gcd_spec(),), {"integer_gcd"},
             f"^SUPERSEDES of v6 family {FAMILY} must be a frozenset of operation names$"),
            ("not among SPECS", (RANK,), frozenset({"rank", "integer_gcd"}),
             f"^v6 family {FAMILY} supersedes integer_gcd, which none of its SPECS defines$"),
            ("no structured function", (OperationSpec("verify", "expr", (), "int", "Test.", "tests only", version=6),),
             frozenset({"verify"}), "^v6 family .* supersedes verify, which has no structured function in "),
        ]
        for label, specs, supersedes, message in cases:
            with self.subTest(case=label):
                module = self.family(specs, supersedes)
                with registry._testing_families(module), self.assertRaisesRegex(RuntimeError, message):
                    registry.specs()
        # The same families with the declarations right build: a legacy operation is superseded as well.
        module = self.family((gcd_spec(), RANK), frozenset({"integer_gcd", "rank"}))
        with registry._testing_families(module):
            self.assertEqual({registry.spec(name).version for name in ("integer_gcd", "rank")}, {6})
            self.assertTrue(registry.is_v6_operation("rank") and registry.is_v6_operation("integer_gcd"))
            self.assertEqual(list(registry.specs())[0], "integer_gcd")  # It keeps its place in the v0.7 order.

    def test_capabilities_list_each_name_once(self):
        frozen = json.loads((FIXTURES / "v8" / "capabilities_v07.json").read_text(encoding="utf-8"))
        with registry._testing_families(demo):
            operations = mf.capabilities()["operations"]
            names = [entry["name"] for entry in operations]
            # The 102 frozen v0.7 entries, then the v6 operations in the order of their families (area packages in
            # name order: continued_fraction and the seventeen v0.9 number theory operations among the 66 of v0.8 and
            # v0.9), and the demo's demo_variable: the superseding integer_gcd keeps its one frozen v0.7 entry.
            self.assertEqual(len(operations), len(frozen["operations"]) + 67)
            self.assertEqual(names[len(frozen["operations"]):],
                             ["count_real_roots", "complex_roots", "interpolate", "denest_sqrt", "sturm_sequence",
                              "cyclotomic_polynomial", "partitions", "derangements", "bell", "stirling1", "solve_abs",
                              "solve_sqrt", "distance", "polygon_area", "line_intersection", "orientation",
                              "convex_hull", "trig_power_antiderivative", "trig_power_integral", "matrix_power",
                              "eigenvectors", "definiteness", "gram_schmidt", "hermite_normal_form",
                              "smith_normal_form", "continued_fraction", "powmod", "multiplicative_order",
                              "primitive_root", "legendre_symbol", "jacobi_symbol", "sqrt_mod", "next_prime",
                              "prev_prime", "linear_diophantine", "quadratic_continued_fraction", "convergents",
                              "pell_fundamental", "moebius", "divisor_sigma", "integer_nth_root", "discrete_log",
                              "two_squares", "digits", "rational_approximation", "sum_polynomial", "geometric_sum",
                              "geometric_series", "polynomial_geometric_series", "series_coefficients", "residue",
                              "set_union", "set_intersection", "set_difference", "set_complement",
                              "set_symmetric_difference", "correlation", "linear_regression", "quantile", "mode",
                              "stationary_distribution", "fisher_exact", "dot", "cross", "projection",
                              "triple_product", "demo_variable"])
            self.assertEqual(names.count("integer_gcd"), 1)
            self.assertEqual(len(names), len(set(names)))
            catalog = [entry["name"] for entry in call_tool("mathforge.catalog", {"limit": 200}).payload["operations"]]
            self.assertEqual(catalog.count("integer_gcd"), 1)
            from .test_mcp_areas import AreaToolTests  # The frozen v0.7 surface, with the demo active.
            outcome_ = unittest.TestResult()
            AreaToolTests("test_capabilities_v07_entries_unchanged").run(outcome_)
            self.assertTrue(outcome_.wasSuccessful(), outcome_.failures + outcome_.errors)

    def test_failure_after_validation_records_the_domain(self):
        x = mf.Symbol("x")
        with registry._testing_families(demo):
            failure = operations_v6.run("demo_variable", {"expression": x + 1, "variable": x})
            self.assertIs(failure.execution_status, ExecutionStatus.INVALID_INPUT)
            self.assertEqual((failure.for_, failure.domain, failure.request.version), (x, mf.Reals, 6))
            self.assertIs(failure.domain, mf.Reals)
            checks = {check.claim: check.status for check in mf.verify(failure).checks}
            self.assertIs(checks["result.source"], CheckStatus.VERIFIED)
            self.assertNotIn(CheckStatus.FAILED, checks.values())
            early = operations_v6.run("demo_variable", {"expression": x + 1})  # Before validation: neither.
            self.assertIs(early.execution_status, ExecutionStatus.INVALID_INPUT)
            self.assertEqual((early.for_, early.domain, early.request), (None, None, None))

    def test_v6_requests_of_a_superseded_name_keep_their_format(self):
        with registry._testing_families(demo):
            arguments = (("a", 12), ("b", 18))
            self.assertEqual(mf.to_data(mf.OperationRequest("integer_gcd", arguments, version=6))["version"], 6)
            older = mf.to_data(mf.OperationRequest("integer_gcd", arguments))
            self.assertEqual(older["version"], 5)
            self.assertNotIn("version", older["data"])
        with self.assertRaisesRegex(mf.SerializationError, "^This operation request belongs to format v5, not v6$"):
            mf.to_data(mf.OperationRequest("integer_gcd", (("a", 12), ("b", 18)), version=6))


class GeneratedWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def rendered(self, *families):
        with registry._testing_families(*families):
            files = self.tool.render_files(SRC)
        return {path.relative_to(SRC / "mathforge").as_posix(): text for path, text in files.items()}

    def test_superseded_v07_operation_calls_run_dual(self):
        files = self.rendered(demo)
        structured = files["_generated/structured.py"]
        self.assertIn("from ..operations_v6 import run_dual\n", structured)
        self.assertIn("def integer_gcd(a: int, b: int, *, limits: ComputationLimits | None = None) -> Result[int]:\n"
                      '    return run_dual("integer_gcd", dict(a=a, b=b), limits=limits)\n', structured)
        self.assertIn('    return run("integer_lcm", dict(a=a, b=b), limits=limits)\n', structured)
        self.assertIn("from ..operations_v6 import run_v07 as run\n", structured)
        self.assertIn("def integer_gcd(a: int, b: int, *, limits: ComputationLimits | None = None) -> int:\n",
                      files["_generated/values.py"])
        self.assertIn("    integer_gcd,\n", files["ntheory.py"])

    def test_superseded_legacy_operation_keeps_its_public_functions(self):
        module = types.ModuleType(FAMILY)
        module.SPECS, module.TYPES, module.CHECKER, module.SUPERSEDES = (RANK,), {}, demo_dual_checker, frozenset(
            {"rank"})
        module.produce = demo.produce
        sys.modules[FAMILY] = module
        self.addCleanup(sys.modules.pop, FAMILY, None)
        files = self.rendered(module)
        for name in ("_generated/structured.py", "_generated/values.py"):
            self.assertNotIn("def rank(", files[name])
            self.assertNotIn('"rank"', files[name])
        self.assertIn("from . import rank\n", files["linalg.py"])
        self.assertIn('    "rank",\n', files["linalg.py"])


class DualSchemaTest(unittest.TestCase):
    def test_v6_schema_takes_both_request_formats_of_a_superseded_name(self):
        """A v6 document may hold an older record of a superseded operation beside a new one."""
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("Development-only jsonschema is unavailable")
        path = Path(__file__).resolve().parents[2] / "tools" / "generate_schema.py"
        spec = importlib.util.spec_from_file_location("generate_schema_dual", path)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        legacy = types.ModuleType(FAMILY)
        legacy.SPECS, legacy.TYPES, legacy.CHECKER, legacy.SUPERSEDES = (RANK,), {}, demo_dual_checker, frozenset(
            {"rank"})
        legacy.produce = demo.produce
        sys.modules[FAMILY] = legacy
        self.addCleanup(sys.modules.pop, FAMILY, None)
        matrix = mf.Matrix(((1, 2), (3, 4)))
        older = (mf.operations.integer_gcd(0, 5), mf.operations.integer_gcd(12, 18), mf.operations.rank(matrix))
        with registry._testing_families(demo, legacy):
            validator = Draft202012Validator(tool.build_schema(6))
            workspace = mf.Workspace()
            for value in (*older, mf.operations.integer_gcd(12, 18), mf.OperationRequest("rank", (("A", matrix),),
                                                                                         version=6)):
                workspace.put(value)
            document = mf.to_data(workspace)
            self.assertEqual(document["version"], 6)
            validator.validate(document)
            wrong = mf.to_data(mf.OperationRequest("integer_gcd", (("a", 1), ("b", 2)), version=6))
            wrong["data"]["arguments"][0]["name"] = "c"  # A v6 request keeps its spec's argument names.
            self.assertFalse(validator.is_valid(wrong))


class FamilyListTest(unittest.TestCase):
    def test_families_come_from_area_packages(self):
        self.assertEqual(registry.v6_families(), ("mathforge.areas.algebra.roots",
                                                  "mathforge.areas.algebra.polynomial_tools",
                                                  "mathforge.areas.algebra.radicals",
                                                  "mathforge.areas.algebra.sequences",
                                                  "mathforge.areas.discrete.counting",
                                                  "mathforge.areas.equations.piecewise",
                                                  "mathforge.areas.geometry.plane",
                                                  "mathforge.areas.geometry.hull",
                                                  "mathforge.areas.integrals.trig_powers",
                                                  "mathforge.areas.linalg.powers",
                                                  "mathforge.areas.linalg.eigen",
                                                  "mathforge.areas.linalg.forms",
                                                  "mathforge.areas.linalg.lattice",
                                                  PILOT,
                                                  "mathforge.areas.ntheory.modular",
                                                  "mathforge.areas.ntheory.prime_search",
                                                  "mathforge.areas.ntheory.diophantine",
                                                  "mathforge.areas.ntheory.quadratic_irrational",
                                                  "mathforge.areas.ntheory.arithmetic_functions",
                                                  "mathforge.areas.ntheory.discrete",
                                                  "mathforge.areas.numeric.digits",
                                                  "mathforge.areas.numeric.approximation",
                                                  "mathforge.areas.series.sums",
                                                  "mathforge.areas.series.power_series",
                                                  "mathforge.areas.series.laurent",
                                                  "mathforge.areas.sets.interval_algebra",
                                                  "mathforge.areas.sets.symmetric",
                                                  "mathforge.areas.stats.bivariate",
                                                  "mathforge.areas.stats.order",
                                                  "mathforge.areas.stats.markov",
                                                  "mathforge.areas.stats.exact_tests",
                                                  "mathforge.areas.vector.products"))
        self.assertEqual(importlib.import_module("mathforge.areas.ntheory").FAMILIES,
                         (PILOT, "mathforge.areas.ntheory.modular", "mathforge.areas.ntheory.prime_search",
                          "mathforge.areas.ntheory.diophantine", "mathforge.areas.ntheory.quadratic_irrational",
                          "mathforge.areas.ntheory.arithmetic_functions", "mathforge.areas.ntheory.discrete"))
        self.assertIs(registry.v6_families(), registry.v6_families())  # Cached.
        with registry._testing_families(demo):
            self.assertEqual(registry.v6_families(), ("mathforge.areas.algebra.roots",
                                                      "mathforge.areas.algebra.polynomial_tools",
                                                      "mathforge.areas.algebra.radicals",
                                                      "mathforge.areas.algebra.sequences",
                                                      "mathforge.areas.discrete.counting",
                                                      "mathforge.areas.equations.piecewise",
                                                      "mathforge.areas.geometry.plane",
                                                      "mathforge.areas.geometry.hull",
                                                      "mathforge.areas.integrals.trig_powers",
                                                      "mathforge.areas.linalg.powers",
                                                      "mathforge.areas.linalg.eigen",
                                                      "mathforge.areas.linalg.forms",
                                                      "mathforge.areas.linalg.lattice",
                                                      PILOT,
                                                      "mathforge.areas.ntheory.modular",
                                                      "mathforge.areas.ntheory.prime_search",
                                                      "mathforge.areas.ntheory.diophantine",
                                                      "mathforge.areas.ntheory.quadratic_irrational",
                                                      "mathforge.areas.ntheory.arithmetic_functions",
                                                      "mathforge.areas.ntheory.discrete",
                                                      "mathforge.areas.numeric.digits",
                                                      "mathforge.areas.numeric.approximation",
                                                      "mathforge.areas.series.sums",
                                                      "mathforge.areas.series.power_series",
                                                      "mathforge.areas.series.laurent",
                                                      "mathforge.areas.sets.interval_algebra",
                                                      "mathforge.areas.sets.symmetric",
                                                      "mathforge.areas.stats.bivariate",
                                                      "mathforge.areas.stats.order",
                                                      "mathforge.areas.stats.markov",
                                                      "mathforge.areas.stats.exact_tests",
                                                      "mathforge.areas.vector.products",
                                                      demo.__name__))
            self.assertTrue(registry.record_module(demo.__name__))
        self.assertEqual(registry.v6_families(), ("mathforge.areas.algebra.roots",
                                                  "mathforge.areas.algebra.polynomial_tools",
                                                  "mathforge.areas.algebra.radicals",
                                                  "mathforge.areas.algebra.sequences",
                                                  "mathforge.areas.discrete.counting",
                                                  "mathforge.areas.equations.piecewise",
                                                  "mathforge.areas.geometry.plane",
                                                  "mathforge.areas.geometry.hull",
                                                  "mathforge.areas.integrals.trig_powers",
                                                  "mathforge.areas.linalg.powers",
                                                  "mathforge.areas.linalg.eigen",
                                                  "mathforge.areas.linalg.forms",
                                                  "mathforge.areas.linalg.lattice",
                                                  PILOT,
                                                  "mathforge.areas.ntheory.modular",
                                                  "mathforge.areas.ntheory.prime_search",
                                                  "mathforge.areas.ntheory.diophantine",
                                                  "mathforge.areas.ntheory.quadratic_irrational",
                                                  "mathforge.areas.ntheory.arithmetic_functions",
                                                  "mathforge.areas.ntheory.discrete",
                                                  "mathforge.areas.numeric.digits",
                                                  "mathforge.areas.numeric.approximation",
                                                  "mathforge.areas.series.sums",
                                                  "mathforge.areas.series.power_series",
                                                  "mathforge.areas.series.laurent",
                                                  "mathforge.areas.sets.interval_algebra",
                                                  "mathforge.areas.sets.symmetric",
                                                  "mathforge.areas.stats.bivariate",
                                                  "mathforge.areas.stats.order",
                                                  "mathforge.areas.stats.markov",
                                                  "mathforge.areas.stats.exact_tests",
                                                  "mathforge.areas.vector.products"))

    def test_spec_lookup_imports_no_checker_kernel_or_lean(self):
        report = probe(
            "import mathforge\n"
            "from mathforge import registry\n"
            "before = set(sys.modules)\n"
            "registry.specs()\n"
            "loaded = sorted(set(sys.modules) - before)\n"
            "inside = lambda name: name.startswith(('mathforge.certify', 'mathforge.kernels', 'mathforge.lean.'))\n"
            "barred = sorted(name for name in loaded if inside(name) or name == 'mathforge.lean')\n"
            "anywhere = sorted(name for name in sys.modules if inside(name))\n"
            "registry.checker('continued_fraction')\n"
            "print(json.dumps({'barred': barred, 'anywhere': anywhere, 'family': '" + PILOT + "' in loaded,\n"
            "                  'checker': 'mathforge.certify.ntheory.continued_fraction' in sys.modules}))\n")
        self.assertEqual(report, {"barred": [], "anywhere": [], "family": True, "checker": True})


class ArgumentKindTest(unittest.TestCase):
    def test_kind_precheck_rejects_before_produce(self):
        x = mf.Symbol("x")
        valid = {"n": 1, "q": mf.Rational(1, 2), "counts": [1, 2], "data": (1, mf.Rational(1, 3)),
                 "A": mf.Matrix(((1,),)), "variable": x}
        demo_kind_family.CALLS.clear()
        self.addCleanup(demo_kind_family.CALLS.clear)
        with registry._testing_families(demo_kind_family):
            for name, value in (("n", 1.5), ("q", True), ("A", [[1]]), ("n", True), ("n", "1"), ("q", 0.5),
                                ("counts", [1, True]), ("counts", 3), ("data", [1, 0.5]), ("variable", "x")):
                with self.subTest(argument=name, value=value):
                    result = operations_v6.run("demo_kinds", dict(valid, **{name: value}))
                    self.assertIs(result.execution_status, ExecutionStatus.INVALID_INPUT)
                    self.assertEqual(result.error.code, "invalid_input")
                    if "0.5" in repr(value) or "1.5" in repr(value):  # No request holds a float, as on the v5 path.
                        self.assertIsNone(result.request)
                        continue
                    self.assertEqual(result.error.message.split(" must be ")[0], f"Argument {name} of demo_kinds")
                    self.assertEqual(result.request.version, 6)  # As the v5 path does, the result keeps its request.
            self.assertEqual(demo_kind_family.CALLS, [])
            result = operations_v6.run("demo_kinds", valid)
            self.assertEqual((result.value, len(demo_kind_family.CALLS)), (1, 1))
            self.assertTrue(mf.verify(result).verified)


if __name__ == "__main__":
    unittest.main()
