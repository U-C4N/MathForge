"""The declarative registry describes the v0.7 operations exactly and is built only once."""

import ast
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import mathforge as m
from mathforge import contracts, deferred, dispatch_v5, registry, registry_v07
from mathforge.errors import UnsupportedOperation
from mathforge.number_theory import IntegerFactorization

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v8"
FAMILY = "mathforge._registry_test_family"
CHECKER = "mathforge.certify._registry_test_checker"  # A production family's checker lives below mathforge.certify.
OUTSIDE = "mathforge._registry_test_outside_checker"  # A checker path outside mathforge.certify.


def published(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def demo_spec(name="demo_triple", area="discrete", arguments=(registry.Argument("n", "int", None, "int"),),
              **options):
    """A throwaway v6 spec; options override sdk/mcp."""
    return registry.OperationSpec(name, area, arguments, "int", "Triple an exact integer.", "test only", version=6,
                                  **options)


def record(name):
    """A record class of the throwaway family, with the given class name."""
    return dataclass(frozen=True)(type(name, (), {"__module__": FAMILY, "__annotations__": {"n": int}}))


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
    def modules(self, *names):
        """Fresh empty modules, each its own sys.modules entry for the rest of this test."""
        created = tuple(types.ModuleType(name) for name in names)
        for module in created:
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__, None)
        return created

    def install_family(self, *specs, records=None, name=FAMILY):
        """Register a throwaway production v6 family for the rest of this test; the cache starts cold.

        It replaces the production families, as the one entry of v6_families(), and names its checker by a path
        below mathforge.certify, as a production family must.
        """
        family, checker = self.modules(name, CHECKER)
        family.SPECS, family.TYPES, family.CHECKER = specs, dict(records or {}), CHECKER
        family.produce = lambda operation, args: (3 * args["n"], ())
        patcher = patch.object(registry, "v6_families", lambda: (name,))
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
                                            "is_prime", "factor_integer", "divisors", "totient",
                                            "continued_fraction"})  # The v0.8 pilot family is v6.
        self.assertEqual(names("series"), {"function_taylor", "function_limit"})
        self.assertEqual(names("integrals"), {"function_antiderivative", "rational_antiderivative",
                                              "rational_integrate", "function_integrate"})
        sizes = {area: len(registry.area_operations(area)) for area in registry.AREAS}
        self.assertEqual({area: size for area, size in sizes.items() if size},
                         {"algebra": 12, "analysis": 6, "series": 2, "integrals": 4, "linalg": 5, "stats": 15,
                          "ntheory": 10, "discrete": 7})
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
        self.assertEqual(registry.all_record_types(),
                         frozenset(expected.values()) | frozenset(registry.record_types(6).values()))
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
        # The generic deferral records lead the v6 types, ahead of the families' records.
        self.assertEqual(list(registry.record_types(6).items()),
                         [("reason", deferred.Reason), ("unevaluated", deferred.Unevaluated), ("demo_box", _Box)])
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

    def test_v6_operation_lookup_rules_out_older_names_without_a_build(self):
        """verify() asks it for every request; the legacy operations verify their own results."""
        self.addCleanup(registry._clear_caches)
        registry._clear_caches()
        for name in ("solve", "factor", "integer_gcd", "function_limit"):
            self.assertFalse(registry.is_v6_operation(name))
        self.assertEqual(registry._tables.cache_info().currsize, 0)
        self.install_family(demo_spec(mcp=False))
        self.assertTrue(registry.is_v6_operation("demo_triple"))
        self.assertFalse(registry.is_v6_operation("demo_nope"))

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

    def test_build_rejects_family_records_named_like_the_deferral_records(self):
        """A family TYPES tag must not shadow Reason or Unevaluated in record_types(6)."""
        for tag in deferred.TYPES:
            with self.subTest(tag=tag):
                self.install_family(demo_spec(), records={tag: _Box})
                with self.assertRaisesRegex(RuntimeError, f"^Duplicate v6 type: {tag}$"):
                    registry.specs()

    def test_build_rejects_records_outside_the_record_modules(self):
        """contracts trusts a static module guard, so every record class must live in a guarded module."""
        self.install_family(demo_spec(), records={"stray": _Stray})
        with self.assertRaisesRegex(RuntimeError, "^Record type stray is defined outside the record modules: "):
            registry.specs()

    def test_v6_types_live_in_their_family_module(self):
        """A v6 family registers only the record classes that it defines: a checker names a record class by its
        module and name, so a class from another record module could not be told from the family's own."""
        pilot = importlib.import_module("mathforge.areas.ntheory.continued_fraction")
        # A v5 record, a deferral record and the record of another v6 family: each lives in a record module.
        for cls in (IntegerFactorization, deferred.Reason, pilot.ContinuedFraction):
            with self.subTest(record=f"{cls.__module__}.{cls.__name__}"):
                family, checker = self.modules(FAMILY, CHECKER)
                family.SPECS, family.TYPES, family.CHECKER = (demo_spec(),), {"probe_box": cls}, checker
                family.produce = lambda operation, args: (3 * args["n"], ())
                message = (f"^v6 record class {cls.__name__} of type probe_box must be defined in its family module "
                           f"{FAMILY}, not in {re.escape(cls.__module__)}$")
                with registry._testing_families(family), self.assertRaisesRegex(RuntimeError, message):
                    registry.specs()

    def test_render_functions_receive_the_shared_renderer(self):
        """RENDER maps a record class to render(value, renderer, child); child keeps aliases and LaTeX fractions."""
        def render(value, renderer, child):
            return renderer.label("box ") + child(value.n)

        family, _ = self.install_family(demo_spec(), records={"demo_box": _Box})
        family.RENDER = {_Box: render}
        self.assertEqual(dict(registry.record_renderers()),
                         {deferred.Reason: None, deferred.Unevaluated: None, _Box: render})
        x = m.Symbol("x")
        self.assertEqual(m.render_text(_Box(x), symbols={"y": x}), "box y")
        self.assertEqual(m.to_latex(_Box(m.Rational(1, 2))), r"\text{box }\frac{1}{2}")

    def test_build_rejects_render_that_is_not_a_mapping(self):
        family, _ = self.install_family(demo_spec(), records={"demo_box": _Box})
        family.RENDER = [(_Box, lambda value, renderer, child: "box")]
        with self.assertRaisesRegex(RuntimeError, f"^RENDER must be a mapping: {FAMILY}$"):
            registry.specs()

    def test_build_rejects_render_keys_outside_the_family_types(self):
        """A key must be a record class of the family's own TYPES, not a tag and not another class."""
        message = f"^RENDER keys must be record classes of the family's own TYPES: {FAMILY}$"
        for key in (_Stray, "demo_box", deferred.Reason):
            with self.subTest(key=key):
                family, _ = self.install_family(demo_spec(), records={"demo_box": _Box})
                family.RENDER = {key: lambda value, renderer, child: "box"}
                with self.assertRaisesRegex(RuntimeError, message):
                    registry.specs()

    def test_build_rejects_render_values_that_are_not_callable(self):
        family, _ = self.install_family(demo_spec(), records={"demo_box": _Box})
        family.RENDER = {_Box: "box {n}"}
        with self.assertRaisesRegex(RuntimeError, f"^RENDER values must be callable: {FAMILY}$"):
            registry.specs()

    def test_production_checkers_live_below_certify(self):
        """tools/check_layering.py guards mathforge.certify only, so a production checker must be a path there."""
        (outside,) = self.modules(OUTSIDE)
        family, checker = self.install_family(demo_spec())
        self.assertIs(registry.checker("demo_triple"), checker)
        message = f"^v6 family {FAMILY} must name its CHECKER by an import path below mathforge.certify, other "
        for value in (OUTSIDE, outside, checker, FAMILY, "mathforge.certifying.checker", None):
            with self.subTest(checker=value):
                family.CHECKER = value
                registry._clear_caches()
                with self.assertRaisesRegex(RuntimeError, message):
                    registry.specs()

    def test_production_checkers_are_not_their_own_family(self):
        name = "mathforge.certify._registry_test_family"  # A family module inside the checker package.
        family, _ = self.install_family(demo_spec(), name=name)
        family.CHECKER = name
        with self.assertRaisesRegex(RuntimeError, f"^v6 family {name} must name its CHECKER .* own module: "):
            registry.specs()

    def test_testing_families_may_give_any_checker(self):
        """Only _testing_families may name a checker by a module object or by a path outside mathforge.certify."""
        family, outside = self.modules(FAMILY, OUTSIDE)
        family.SPECS, family.TYPES, family.produce = (demo_spec(),), {}, lambda operation, args: (3 * args["n"], ())
        for value, expected in ((outside, outside), (OUTSIDE, outside)):
            with self.subTest(checker=value), registry._testing_families(family):
                family.CHECKER = value
                self.assertIs(registry.checker("demo_triple"), expected)
        self.assertNotIn("demo_triple", registry.specs())

    def test_build_rejects_v6_families_missing_part_of_the_contract(self):
        self.assertEqual(registry.V6_CONTRACT, ("SPECS", "TYPES", "CHECKER", "produce"))
        for name in registry.V6_CONTRACT:
            with self.subTest(missing=name):
                family, _ = self.install_family(demo_spec())
                delattr(family, name)
                with self.assertRaisesRegex(RuntimeError, f"^v6 family {FAMILY} does not define {name}$"):
                    registry.specs()

    def test_build_rejects_names_of_mcp_schema_definitions(self):
        """An area tool's $defs hold each operation under its name, beside the kinds and the envelope."""
        self.assertEqual(registry.DEFINITION_NAMES, (*registry.KIND_NAMES, "envelope"))
        for name in ("rational", "envelope"):
            with self.subTest(operation=name):
                self.install_family(demo_spec(name))
                with self.assertRaisesRegex(RuntimeError, f"^Operation name collides with an MCP schema definition: "
                                                          f"{name}$"):
                    registry.specs()
            with self.subTest(public=name):
                self.install_family(demo_spec(sdk=name))
                with self.assertRaisesRegex(RuntimeError, f"^Public name collides with an MCP schema definition: "
                                                          f"{name}$"):
                    registry.specs()

    def test_build_rejects_names_the_generated_wrappers_use(self):
        self.assertEqual(registry.WRAPPER_NAMES,
                         ("limits", "dict", "run", "run_v6", "structured", "_LEAN", "FALLBACK", "_value", "run_dual"))
        for name in registry.WRAPPER_NAMES:
            with self.subTest(argument=name):
                self.install_family(demo_spec(arguments=(registry.Argument(name, "int", None, "int"),)))
                with self.assertRaisesRegex(RuntimeError, "^Argument name of demo_triple collides with a name the "
                                                          f"generated wrappers use: {name}$"):
                    registry.specs()
            with self.subTest(public=name):
                self.install_family(demo_spec(sdk=name))
                with self.assertRaisesRegex(RuntimeError, "^Public name collides with a name the generated wrappers "
                                                          f"use: {name}$"):
                    registry.specs()

    def test_build_rejects_names_that_are_not_identifiers(self):
        for name in ("2n", "class", "n-1", ""):
            with self.subTest(argument=name):
                self.install_family(demo_spec(arguments=(registry.Argument(name, "int", None, "int"),)))
                with self.assertRaisesRegex(RuntimeError, "^Argument name of demo_triple is not a Python identifier: "
                                                          f"{re.escape(repr(name))}$"):
                    registry.specs()
        for label, spec in (("Operation name", demo_spec("demo-triple")), ("Public name", demo_spec(sdk="lambda"))):
            with self.subTest(label=label):
                self.install_family(spec)
                with self.assertRaisesRegex(RuntimeError, f"^{label} is not a Python identifier: "):
                    registry.specs()

    def test_build_accepts_only_literal_defaults(self):
        """A wrapper copies the default's text into its signature; the MCP tools evaluate it with literal_eval."""
        for default in ("x", "Rational(1, 2)", "1e999", "[1, 2", "", "...", "1 if True else 2"):
            with self.subTest(default=default):
                self.install_family(demo_spec(arguments=(registry.Argument("n", "int", default, "int"),)))
                with self.assertRaisesRegex(RuntimeError, "^Default of argument n of demo_triple is not a Python "
                                                          f"literal: {re.escape(repr(default))}$"):
                    registry.specs()
        for default in ("0", "-1", "'both'", "(1, 2)", "None", "True"):
            with self.subTest(default=default):
                self.install_family(demo_spec(arguments=(registry.Argument("n", "int", default, "int"),)))
                self.assertEqual(registry.spec("demo_triple").arguments[0].default, default)

    def test_build_rejects_record_classes_that_share_a_name(self):
        """The generator and the area modules find record classes by name, across v5 and v6."""
        for records, first in (({"demo_crt": record("CRTResult")}, "mathforge.number_theory"),
                               ({"demo_a": record("DemoBox"), "demo_b": record("DemoBox")}, FAMILY)):
            name = next(iter(records.values())).__name__
            with self.subTest(name=name):
                self.install_family(demo_spec(), records=records)
                with self.assertRaisesRegex(RuntimeError, f"^Two record classes are named {name}: in {first} and "
                                                          f"{FAMILY}$"):
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
                barrier.wait(30)  # A thread that never arrives breaks the barrier instead of hanging the test.
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
