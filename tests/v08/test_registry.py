"""The declarative registry describes the v0.7 operations exactly and is built only once."""

import ast
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge import contracts, dispatch_v5, registry, registry_v07
from mathforge.errors import UnsupportedOperation

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v8"
FAMILY = "mathforge._registry_test_family"
CHECKER = "mathforge._registry_test_checker"


def published(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def demo_spec(name="demo_triple", area="discrete", **options):
    """A throwaway v6 spec; options override sdk/mcp."""
    return registry.OperationSpec(name, area, (registry.Argument("n", "int", None, "int"),), "int",
                                  "Triple an exact integer.", "test only", version=6, **options)


def run_probe(code):
    """Run code in a fresh isolated interpreter that imports this checkout's mathforge; return stdout."""
    root = str(Path(m.__file__).resolve().parents[1])
    output = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); " + code, root],
                            capture_output=True, text=True)
    if output.returncode:
        raise AssertionError(output.stderr)
    return output.stdout.strip()


def family_tables(attribute):
    """Merge one table of the v5 family modules, read directly (dispatch_v5 delegates to the registry)."""
    merged = {}
    for name in registry.V5_FAMILIES:
        for key, value in getattr(importlib.import_module(name), attribute, {}).items():
            merged[key] = value
    return merged


def signature(spec, returns):
    """Render a spec as str(inspect.signature(...)) renders the v0.7 wrappers (PEP 563 string annotations)."""
    parameters = []
    for argument in spec.arguments:
        text = argument.name + (f": {argument.annotation!r}" if argument.annotation else "")
        if argument.default is not None:
            text += (" = " if argument.annotation else "=") + argument.default
        parameters.append(text)
    parameters += ["*", "limits: 'ComputationLimits | None' = None"]
    return f"({', '.join(parameters)}) -> {returns!r}"


def expected_kind(operation, argument):
    """The plan's argument-kind table, rule by rule."""
    if argument == "expression":
        return "rational_function" if operation.startswith("rational_") else "function"
    if operation == "quadratic_roots":
        return "scalar"
    if (operation, argument) == ("least_squares", "b"):
        return "rational_list"
    if (operation, argument) == ("cdf", "x"):
        return "rational"
    if argument == "point":
        return "point" if operation == "function_limit" else "scalar"
    return {
        "a": "int", "b": "int", "modulus": "int", "n": "int", "k": "int", "population": "int",
        "successes": "int", "draws": "int", "order": "int", "ddof": "int",
        "congruences": "int_pairs", "counts": "int_list",
        "real": "scalar", "imag": "scalar", "lower": "scalar", "upper": "scalar",
        "left": "complex", "right": "complex", "value": "complex", "base": "complex", "p": "rational",
        "outcomes": "rational_list", "event": "rational_list", "data": "rational_list",
        "x": "rational_list", "y": "rational_list", "distribution": "distribution", "A": "matrix",
        "expressions": "function_list", "variable": "symbol", "variables": "symbols",
        "bindings": "bindings", "side": "side",
    }[argument]


@dataclass(frozen=True)
class _Box:
    __module__ = FAMILY  # v6 records are defined in their family module.
    n: int


@dataclass(frozen=True)
class _Stray:  # A record class outside every family module.
    n: int


class RegistryTests(unittest.TestCase):
    def install_family(self, *specs, records=None):
        """Register a throwaway v6 family (string CHECKER) for the rest of this test; the cache starts cold."""
        family, checker = types.ModuleType(FAMILY), types.ModuleType(CHECKER)
        family.SPECS, family.TYPES, family.CHECKER = specs, dict(records or {}), CHECKER
        for module in (family, checker):
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__, None)
        patcher = patch.object(registry, "V6_FAMILIES", (FAMILY,))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(registry._clear_caches)
        registry._clear_caches()
        return family, checker

    def test_v07_specs_match_published_arguments(self):
        expected = {name: frozenset(keys) for name, keys in family_tables("ARGUMENTS").items()}
        specs = {name: frozenset(a.name for a in s.arguments)
                 for name, s in registry.specs().items() if s.version == 5}
        self.assertEqual(specs, expected)
        self.assertEqual(len(registry_v07.V07_SPECS), 60)
        # Same dispatch order: capabilities lists the v0.7 operations family by family.
        self.assertEqual(list(registry.operation_arguments(5).items()), list(expected.items()))

    def test_every_spec_has_known_area_and_kind(self):
        self.assertEqual(registry.AREAS, ("expr", "algebra", "sets", "logic", "equations", "analysis", "series",
                                          "integrals", "ode", "special", "linalg", "stats", "ntheory", "discrete",
                                          "geometry", "vector", "numeric", "crypto"))
        self.assertEqual(len(registry.KIND_NAMES), 18)
        self.assertEqual(len(registry.MCP_TOOLS), 12)
        covered = [area for areas in registry.MCP_TOOLS.values() for area in areas]
        self.assertEqual(sorted(covered), sorted(registry.AREAS))
        for spec in registry.specs().values():
            with self.subTest(operation=spec.name):
                self.assertIn(spec.area, registry.AREAS)
                for argument in spec.arguments:
                    self.assertIn(argument.kind, registry.KIND_NAMES)

    def test_area_assignment(self):
        def names(area):
            return {spec.name for spec in registry.area_operations(area)}

        self.assertEqual(names("ntheory"), {"integer_gcd", "integer_lcm", "integer_xgcd", "mod_inverse", "crt",
                                            "is_prime", "factor_integer", "divisors", "totient"})
        self.assertEqual(names("series"), {"function_taylor", "function_limit"})
        self.assertEqual(names("integrals"), {"function_antiderivative", "rational_antiderivative",
                                              "rational_integrate", "function_integrate"})
        sizes = {area: len(registry.area_operations(area)) for area in registry.AREAS}
        self.assertEqual({area: size for area, size in sizes.items() if size},
                         {"algebra": 12, "analysis": 6, "series": 2, "integrals": 4, "linalg": 5, "stats": 15,
                          "ntheory": 9, "discrete": 7})
        self.assertEqual([spec.public_name for spec in registry.area_operations("series")], ["function_limit", "taylor"])
        analysis = registry.tool_operations("mathforge.analysis")
        self.assertEqual(len(analysis), 12)
        self.assertEqual({spec.area for spec in analysis}, {"analysis", "series", "integrals"})
        self.assertEqual(registry.tool_operations("mathforge.logic"), ())

    def test_registry_is_built_once(self):
        guard = AssertionError("the registry imported a module after warm-up")
        self.addCleanup(registry._clear_caches)
        registry._clear_caches()
        with patch("importlib.import_module", side_effect=guard), self.assertRaises(AssertionError):
            registry.specs()  # a cold build does go through importlib.import_module
        registry.specs(), registry.family("crt"), registry.record_types(5)
        with patch("importlib.import_module", side_effect=guard):
            self.assertIs(registry.specs()["crt"], registry.spec("crt"))
            self.assertEqual(registry.family("crt").__name__, "mathforge.number_theory")
            self.assertIs(registry.record_types(5)["crt_result"], m.CRTResult)
            self.assertIs(dispatch_v5.module_for("crt"), registry.family("crt"))
            self.assertEqual(tuple(module.__name__ for module in dispatch_v5.modules()), registry.V5_FAMILIES)
            self.assertIn("crt", dispatch_v5.operation_arguments())
            self.assertIn("crt_result", dispatch_v5.record_types())

    def test_v5_record_types_unchanged(self):
        expected = family_tables("TYPES")
        self.assertEqual(list(registry.record_types(5).items()), list(expected.items()))
        self.assertIs(registry.record_types(5)["crt_result"], m.CRTResult)
        self.assertIs(registry.record_types(5)["FiniteDistribution"], m.FiniteDistribution)
        self.assertEqual(registry.all_record_types(), frozenset(expected.values()))
        self.assertEqual(registry.V5_FAMILIES, tuple("mathforge." + name for name in (
            "number_theory", "combinatorics", "complex_numbers", "probability", "statistics",
            "linear_extensions", "functions", "function_calculus", "rational_integration", "function_integration")))

    def test_v07_specs_reproduce_published_signatures(self):
        signatures = published("api_v7_signatures.json")
        self.assertEqual(sorted(spec.public_name for spec in registry_v07.V07_SPECS), sorted(signatures["api_v7"]))
        self.assertEqual(registry.spec("function_taylor").public_name, "taylor")
        for spec in registry_v07.V07_SPECS:
            with self.subTest(operation=spec.name):
                self.assertEqual(signature(spec, f"Result[{spec.returns}]"), signatures["api_v7"][spec.public_name])
                self.assertEqual(signature(spec, spec.returns), signatures["values_v7"][spec.public_name])

    def test_v07_specs_reproduce_capability_entries(self):
        entries = [item for item in published("capabilities_v07.json")["operations"]
                   if item["mcp_tool"] == "mathforge.compute"]
        self.assertEqual([spec.name for spec in registry_v07.V07_SPECS], [item["name"] for item in entries])
        for spec, item in zip(registry_v07.V07_SPECS, entries):
            with self.subTest(operation=spec.name):
                self.assertEqual(spec.public_name, item["sdk"])
                arguments = ", ".join(sorted(argument.name for argument in spec.arguments))
                self.assertEqual(spec.scope + "; arguments: " + arguments, item["scope"])

    def test_argument_kinds_follow_the_table(self):
        for spec in registry_v07.V07_SPECS:
            for argument in spec.arguments:
                with self.subTest(operation=spec.name, argument=argument.name):
                    self.assertEqual(argument.kind, expected_kind(spec.name, argument.name))

    def test_legacy_catalog(self):
        legacy = [item for item in published("capabilities_v07.json")["operations"]
                  if item["mcp_tool"] != "mathforge.compute"]
        catalog = registry_v07.LEGACY_CATALOG
        self.assertEqual(len(catalog), 42)
        self.assertEqual([entry.name for entry in catalog], [item["name"] for item in legacy])
        self.assertFalse({entry.name for entry in catalog} & set(registry.specs()))
        for entry, item in zip(catalog, legacy):
            with self.subTest(operation=entry.name):
                self.assertEqual(entry.tool, item["mcp_tool"] or "")
                self.assertIn(entry.area, registry.AREAS)
        areas = {entry.name: entry.area for entry in catalog}  # Controller rulings, Task 2 fix round 1.
        self.assertEqual((areas["verify"], areas["expand"], areas["expand_multivariate"]), ("expr", "algebra", "algebra"))

    def test_summaries_are_single_sentences(self):
        summaries = [spec.summary for spec in registry_v07.V07_SPECS] + [e.summary for e in registry_v07.LEGACY_CATALOG]
        for summary in summaries:
            with self.subTest(summary=summary):
                self.assertTrue(summary[0].isupper() and summary.endswith("."))
                self.assertNotIn(". ", summary)

    def test_lookups_reject_unknown_names(self):
        for lookup in (registry.spec, registry.family, registry.checker):
            with self.subTest(lookup=lookup.__name__), self.assertRaisesRegex(UnsupportedOperation, "^Unknown operation$"):
                lookup("taylor")
        with self.assertRaisesRegex(UnsupportedOperation, "^Unknown v5 operation$"):
            dispatch_v5.module_for("taylor")
        for call in (lambda: registry.area_operations("calculus"), lambda: registry.tool_operations("mathforge.compute"),
                     lambda: registry.record_types(4), lambda: registry.operation_arguments(7)):
            with self.assertRaises(ValueError):
                call()
        for name in registry.operation_arguments(5):
            self.assertIs(registry.checker(name), registry.family(name))

    def test_v6_families_join_the_registry_but_not_the_v5_views(self):
        family, checker = self.install_family(demo_spec(mcp=False), records={"demo_box": _Box})
        self.assertIs(registry.family("demo_triple"), family)
        self.assertIs(registry.checker("demo_triple"), checker)
        self.assertEqual(dict(registry.operation_arguments(6)), {"demo_triple": frozenset({"n"})})
        self.assertIn("demo_triple", registry.operation_arguments())
        self.assertNotIn("demo_triple", registry.operation_arguments(5))
        self.assertEqual(dict(registry.record_types(6)), {"demo_box": _Box})
        self.assertIn(_Box, registry.all_record_types())
        self.assertIn("demo_triple", [spec.name for spec in registry.area_operations("discrete")])
        self.assertNotIn("demo_triple", [spec.name for spec in registry.tool_operations("mathforge.discrete")])
        # The contracts guard admits a v6 record (its family is a record module) and nothing else from outside.
        self.assertTrue(contracts._extended_value(_Box(1)))
        self.assertFalse(contracts._extended_value(_Stray(1)))
        # mathforge.compute stays v0.7-only: the dispatch_v5 views never include v6 families.
        self.assertEqual(dict(dispatch_v5.operation_arguments()), dict(registry.operation_arguments(5)))
        self.assertNotIn("demo_box", dispatch_v5.record_types())
        with self.assertRaisesRegex(UnsupportedOperation, "^Unknown v5 operation$"):
            dispatch_v5.module_for("demo_triple")

    def test_build_rejects_duplicate_public_names(self):
        self.install_family(demo_spec("demo_taylor", "series", sdk="taylor"))
        with self.assertRaisesRegex(RuntimeError, "^Duplicate public name: taylor$"):
            registry.specs()

    def test_build_rejects_operation_names_of_the_legacy_catalog(self):
        self.install_family(demo_spec("factor", "algebra"))
        with self.assertRaisesRegex(RuntimeError, "^Operation name collides with a legacy catalogue entry: factor$"):
            registry.specs()

    def test_build_rejects_public_names_of_the_legacy_catalog(self):
        """Its generated wrapper would shadow mathforge.operations.limit."""
        self.install_family(demo_spec("demo_limit", "series", sdk="limit"))
        with self.assertRaisesRegex(RuntimeError, "^Public name collides with a legacy catalogue entry: limit$"):
            registry.specs()

    def test_build_rejects_records_outside_the_record_modules(self):
        """contracts trusts a static module guard, so every record class must live in a guarded module."""
        self.install_family(demo_spec(), records={"stray": _Stray})
        with self.assertRaisesRegex(RuntimeError, "^Record type stray is defined outside the record modules: "):
            registry.specs()

    def test_concurrent_first_build_shares_one_table(self):
        """Review Focus 5: threads racing on a cold registry all receive the same tables."""
        import_module = importlib.import_module

        def slow_import(name, package=None):
            time.sleep(0.005)  # Widen the race window of the import phase, which runs before the lock.
            return import_module(name, package)

        self.addCleanup(registry._clear_caches)
        registry._clear_caches()
        barrier, seen, errors = threading.Barrier(8), [], []

        def read():
            try:
                barrier.wait()
                seen.append(registry.specs())
            except BaseException as exc:
                errors.append(exc)

        with patch("importlib.import_module", slow_import):
            threads = [threading.Thread(target=read) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(seen), 8)
        self.assertEqual(len({id(table) for table in seen}), 1)

    def test_registry_import_stays_light(self):
        """mathforge/__init__.py will import the registry, so it must not import families or tables at import time."""
        tree = ast.parse(Path(registry.__file__).read_text(encoding="utf-8"))
        internal = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                internal.update(alias.name for alias in node.names if alias.name.split(".")[0] == "mathforge")
            elif isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] == "mathforge"):
                internal.add("." * node.level + (node.module or ""))
        self.assertLessEqual(internal, {".errors"})
        self.assertEqual(run_probe("import mathforge.registry as r; print(r._tables.cache_info().currsize)"), "0")

    def test_v1_to_v4_results_leave_the_registry_unbuilt(self):
        """contracts consults the registry only for values defined in a record module (the v0.7 guard)."""
        probe = (
            "import mathforge as m, mathforge.registry as r; x = m.Symbol('x'); "
            "results = (m.operations.solve(m.Eq(x**2, 2), for_=x), m.operations.det(m.Matrix(((1, 2), (3, 4)))), "
            "m.operations.limit(m.rational_function(1, x, variable=x), x, '+inf'), "
            "m.operations.factor(m.Polynomial((-2, 0, 1), x)), m.operations.minimal_polynomial(m.sqrt(2), x)); "
            "assert all(res.execution_status is m.ExecutionStatus.COMPLETED for res in results), results; "
            "print(r._tables.cache_info().currsize)")
        self.assertEqual(run_probe(probe), "0")


if __name__ == "__main__":
    unittest.main()
