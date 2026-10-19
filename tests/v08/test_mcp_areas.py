"""MCP area tools and mathforge.catalog: typed, registry-derived tools beside the frozen thirty v0.7 tools."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import mathforge as m
from mathforge import operations_v6, registry
from mathforge.capabilities import area_tools
from mathforge.contracts import CheckStatus
from mathforge.mcp.boundary import call_tool
from mathforge.mcp.schemas import input_schemas
from mathforge.operations_v5 import _freeze
from mathforge.registry_v07 import LEGACY_CATALOG

from . import demo_family

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v8"
X_ID = "00000000-0000-4000-8000-000000000701"  # The UUIDs of the symbols in the v0.7 frontend samples.
Y_ID = "00000000-0000-4000-8000-000000000702"
AREA_SCHEMA_BUDGET = 16 * 1024  # Bytes of compact JSON: one area tool's input schema...
SCHEMA_BUDGET = 128 * 1024  # ...and the input schemas of all tools together.
SCHEMA_MAPS = ("properties", "$defs")  # Keyword values that map names to schemas
SUBSCHEMAS = ("items", "additionalProperties", "propertyNames", "not", "if", "then", "else", "contains")
SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")


def published(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def frontend_samples():
    """test_frontend_integration_v7.samples(): positional SDK arguments of the 60 v0.7 operations, loaded by path."""
    path = Path(__file__).resolve().parents[1] / "test_frontend_integration_v7.py"
    spec = importlib.util.spec_from_file_location("frontend_integration_v7_samples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.samples()


def tool_of(operation):
    return next(tool for tool in area_tools() if operation in {s.name for s in registry.tool_operations(tool)})


def schema_keywords(schema):
    """(keyword, value) for each keyword of a schema and of its subschemas; enum, const and default hold data."""
    found, pending = [], [schema]
    while pending:
        node = pending.pop()
        if type(node) is not dict:  # For example "additionalProperties": false
            continue
        for key, value in node.items():
            found.append((key, value))
            if key in SCHEMA_MAPS:
                pending.extend(value.values())
            elif key in SUBSCHEMAS:
                pending.append(value)
            elif key in SCHEMA_LISTS:
                pending.extend(value)
    return found


def symbol_ids(values):
    """The name and UUID of every symbol in some values, read from their canonical JSON."""
    ids, pending = {}, [m.to_data(_freeze(tuple(values)))["data"]]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            if node.get("type") == "symbol":
                ids[node["name"]] = node["uid"]
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)
    return ids


class AreaToolTests(unittest.TestCase):
    def call(self, tool, operation, arguments, **fields):
        return call_tool(tool, dict(fields, operation=operation, arguments=arguments))

    def result(self, response):
        self.assertFalse(response.is_error, response.payload)
        self.assertIsNone(response.payload["error"])
        return m.from_data(response.payload["result"])

    def error(self, response):
        """The error of a failed response; a failed computation also keeps its canonical Result."""
        self.assertTrue(response.is_error, response.payload)
        if response.payload["error"]["stage"] != "computation":
            self.assertIsNone(response.payload["result"])
        return response.payload["error"]

    def test_v07_tool_schemas_unchanged(self):
        schemas, frozen = input_schemas(), published("mcp_v7_input_schemas.json")
        names = [tool["name"] for tool in published("capabilities_v07.json")["tools"]]
        self.assertEqual(sorted(names), sorted(frozen))
        self.assertEqual(list(schemas)[:30], names)
        self.assertEqual(json.loads(json.dumps({name: schemas[name] for name in names})), frozen)

    def test_capabilities_v07_entries_unchanged(self):
        current, frozen = m.capabilities(), published("capabilities_v07.json")
        self.assertEqual(current["tools"][:30], frozen["tools"])
        self.assertEqual(current["operations"][:len(frozen["operations"])], frozen["operations"])

        def descriptive(document):
            return {key: value for key, value in document.items() if key not in ("tools", "operations")}

        self.assertEqual(descriptive(current), descriptive(frozen))
        # v6 operations follow the frozen entries, each naming the area tool that runs it; one that supersedes a
        # v0.7 operation keeps that operation's frozen entry.
        later = current["operations"][len(frozen["operations"]):]
        self.assertEqual([entry["name"] for entry in later],
                         [spec.name for spec in registry.specs().values()
                          if spec.version == 6 and not registry.is_older_operation(spec.name)])
        self.assertIn({"name": "continued_fraction", "sdk": "continued_fraction", "mcp_tool": "mathforge.ntheory",
                       "scope": "exact rationals; canonical finite simple continued fraction; arguments: value"}, later)

    def test_tool_list_is_registry_derived(self):
        names = [tool["name"] for tool in m.capabilities()["tools"]]
        v07 = [tool["name"] for tool in published("capabilities_v07.json")["tools"]]
        self.assertEqual(names, v07 + area_tools() + ["mathforge.catalog"])
        self.assertEqual(list(input_schemas()), names)
        self.assertIn("mathforge.ntheory", names)
        self.assertNotIn("mathforge.logic", names)  # No operation serves the logic area yet.
        self.assertEqual(area_tools(), sorted(area_tools()))  # The tools serving at least one operation, by name.
        for tool in registry.MCP_TOOLS:
            self.assertEqual(tool in area_tools(), bool(registry.tool_operations(tool)), tool)

    # No docstring: unittest would print it on its own line, between the test's name and the measurement.
    def test_input_schemas_stay_within_their_size_budget(self):
        # Every MCP client holds the tool schemas in its context. Sizes are those of compact JSON.
        sizes = {name: len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
                 for name, schema in input_schemas().items()}
        total = sum(sizes.values())
        print("MCP input schemas: " + ", ".join(f"{tool} {sizes[tool] / 1024:.1f} KB" for tool in area_tools())
              + f"; all {len(sizes)} tools {total / 1024:.1f} KB (budgets: {AREA_SCHEMA_BUDGET // 1024} KB per area "
              f"tool, {SCHEMA_BUDGET // 1024} KB in all)", file=sys.stderr)
        for tool in area_tools():
            with self.subTest(tool=tool):
                self.assertLessEqual(sizes[tool], AREA_SCHEMA_BUDGET)
        self.assertLessEqual(total, SCHEMA_BUDGET)

    def test_area_tools_match_compute(self):
        from mathforge.mcp.kinds import KINDS
        samples = frontend_samples()
        self.assertEqual(set(samples), set(registry.operation_arguments(5)))  # Every v0.7 operation is compared.
        for operation, positional in samples.items():
            spec = registry.spec(operation)
            given = dict(zip((argument.name for argument in spec.arguments), positional))
            complete = {argument.name: given[argument.name] if argument.name in given
                        else ast.literal_eval(argument.default) for argument in spec.arguments}
            kinds = {argument.name: argument.kind for argument in spec.arguments}
            with self.subTest(operation=operation):
                area = self.call(tool_of(operation), operation,
                                 {name: KINDS[kinds[name]].encode(value) for name, value in given.items()},
                                 symbol_ids=symbol_ids(given.values()))
                compute = call_tool("mathforge.compute", {"operation": operation, "arguments": {
                    name: m.to_data(_freeze(value)) for name, value in complete.items()}})
                self.assertEqual(m.dumps(self.result(area)), m.dumps(self.result(compute)))

    def test_area_tool_rejects_foreign_operation(self):
        self.assertEqual(self.error(self.call("mathforge.ntheory", "binomial", {"n": 5, "k": 2}))["field"], "operation")
        for operation in ("__import__", "solve", "compute", "demo_double", None, ["integer_gcd"]):
            with self.subTest(operation=operation):
                error = self.error(self.call("mathforge.ntheory", operation, {"a": 6, "b": 4}))
                self.assertEqual(error["field"], "operation")

    def test_catalog_finds_pilot(self):
        response = call_tool("mathforge.catalog", {"query": "continued"})
        self.assertFalse(response.is_error, response.payload)
        self.assertEqual((response.payload["api_version"], response.payload["error"]), (1, None))
        entry = next(entry for entry in response.payload["operations"] if entry["name"] == "continued_fraction")
        self.assertEqual(entry, {
            "name": "continued_fraction", "area": "ntheory", "tool": "mathforge.ntheory",
            "summary": "Return the finite simple continued fraction of an exact rational.",
            "arguments": [{"name": "value", "kind": "rational", "required": True, "default": None}],
            "returns": "ContinuedFraction"})

    def test_inexact_json_numbers_rejected(self):
        """Review Focus 1: JSON decimals and booleans are never exact numbers."""
        for value in (0.5, True, 2.0, False):
            with self.subTest(value=value):
                error = self.error(self.call("mathforge.ntheory", "integer_gcd", {"a": value, "b": 2}))
                self.assertEqual((error["code"], error["field"]), ("invalid_input", "arguments.a"))
                error = self.error(self.call("mathforge.stats", "bernoulli", {"p": value}))
                self.assertEqual((error["code"], error["field"]), ("invalid_input", "arguments.p"))
        error = self.error(self.call("mathforge.discrete", "multinomial", {"counts": [2, 1.5]}))
        self.assertEqual(error["field"], "arguments.counts[1]")

    def test_symbol_ids_reused(self):
        """Review Focus 2: requests with the same symbol_ids return the same symbol identities."""
        arguments = {"expression": {"text": "sin(x)*x", "variables": ["x"]}, "variable": "x"}
        first, second = (self.call("mathforge.analysis", "function_differentiate", arguments, symbol_ids={"x": X_ID})
                         for _ in range(2))
        self.assertEqual(first.payload["symbols"], [{"name": "x", "uid": X_ID}])
        self.assertEqual(second.payload["symbols"], first.payload["symbols"])
        self.assertEqual(m.dumps(self.result(first)), m.dumps(self.result(second)))
        fresh = [self.call("mathforge.analysis", "function_differentiate", arguments).payload["symbols"]
                 for _ in range(2)]
        self.assertNotEqual(fresh[0], fresh[1])  # Without symbol_ids every request mints its own identity.

    def test_oversized_integer_is_resource_error(self):
        """Review Focus 3: input beyond the budget is a resource_limit_exceeded response, never an exception.

        Over MCP an integer that large arrives as a digit string: the SDK's JSON parser rejects JSON numbers
        beyond Python's integer conversion limit (4300 digits) before the server sees them.
        """
        for value in ("1" + "0" * 21100, "-" + "9" * 1100):
            with self.subTest(digits=len(value)):
                error = self.error(self.call("mathforge.ntheory", "integer_gcd", {"a": value, "b": 3}))
                self.assertEqual((error["code"], error["field"]), ("resource_limit_exceeded", "arguments.a"))
        error = self.error(self.call("mathforge.ntheory", "continued_fraction", {"value": "1/" + "7" * 1100}))
        self.assertEqual((error["code"], error["field"]), ("resource_limit_exceeded", "arguments.value"))
        # A JSON integer within the transport limit still meets the request's integer budget while decoding.
        error = self.error(self.call("mathforge.ntheory", "integer_gcd", {"a": 2 ** 200, "b": 3},
                                     limits={"max_integer_bits": 100}))
        self.assertEqual((error["code"], error["field"]), ("resource_limit_exceeded", "arguments.a"))
        self.assertEqual(self.error(self.call("mathforge.discrete", "multinomial", {"counts": [2 ** 200]},
                                              limits={"max_integer_bits": 100}))["field"], "arguments.counts[0]")

    def test_verify_tool_accepts_unknown_v6_operation(self):
        """Review Focus 4: a v6 result of an operation this build does not know verifies as UNKNOWN."""
        with registry._testing_families(demo_family):
            document = m.to_data(operations_v6.run("demo_double", {"n": 21}))
        response = call_tool("mathforge.verify", {"result": document})
        self.assertFalse(response.is_error, response.payload)
        report = m.from_data(response.payload["result"])
        self.assertFalse(report.verified)
        statuses = {check.claim: check.status for check in report.checks}
        self.assertIs(statuses["result.operation"], CheckStatus.UNKNOWN)
        self.assertNotIn(CheckStatus.FAILED, statuses.values())

    def test_v6_pilot_runs_through_its_area_tool(self):
        response = self.call("mathforge.ntheory", "continued_fraction", {"value": "415/93"})
        result = self.result(response)
        self.assertEqual(response.payload["result"]["version"], 6)
        self.assertEqual(result.value.terms, (4, 2, 6, 7))
        self.assertIn("[4; 2, 6, 7]", response.payload["text"])
        self.assertEqual(m.dumps(result), m.dumps(m.operations.continued_fraction(m.Rational(415, 93))))
        self.assertTrue(self.result(call_tool("mathforge.verify", {"result": response.payload["result"]})).verified)
        # "p/q" is a Rational even when q is 1; a JSON integer or a digit string is an int.
        for value, expected in (("3/1", m.Rational(3)), (3, 3), ("-3", -3)):
            with self.subTest(value=value):
                request = self.result(self.call("mathforge.ntheory", "continued_fraction", {"value": value})).request
                self.assertEqual(request.arguments, (("value", expected),))
                self.assertIs(type(request.arguments[0][1]), type(expected))

    def test_arguments_follow_the_spec(self):
        """Required arguments must be present, unknown ones are rejected, and only spec defaults are filled in."""
        gcd = ("mathforge.ntheory", "integer_gcd")
        self.assertEqual(self.error(self.call(*gcd, {"a": 6}))["field"], "arguments.b")
        self.assertEqual(self.error(self.call(*gcd, {"a": 6, "b": 4, "c": 1}))["field"], "arguments.c")
        for arguments in (None, [6, 4], "a=6"):
            self.assertEqual(self.error(self.call(*gcd, arguments))["field"], "arguments")
        for request in ({"operation": "integer_gcd"}, {"operation": "integer_gcd", "arguments": {}, "extra": 1}):
            self.assertTrue(call_tool("mathforge.ntheory", request).is_error)
        function = {"text": "x^3", "variables": ["x"]}
        implicit = self.call("mathforge.analysis", "function_differentiate", {"expression": function, "variable": "x"},
                             symbol_ids={"x": X_ID})
        explicit = self.call("mathforge.analysis", "function_differentiate",
                             {"expression": function, "variable": "x", "n": 1}, symbol_ids={"x": X_ID})
        self.assertEqual(m.dumps(self.result(implicit)), m.dumps(self.result(explicit)))
        limit = self.result(self.call("mathforge.analysis", "function_limit", {
            "expression": {"text": "1/x", "variables": ["x"]}, "variable": "x", "point": "+inf"}))
        self.assertIn(("side", "both"), limit.request.arguments)

    def test_envelopes_get_compute_checks(self):
        """Every kind also takes a canonical envelope (v1-v6), with compute's shape, tag and canonical checks."""
        gcd = ("mathforge.ntheory", "integer_gcd")
        expected = m.dumps(m.operations.integer_gcd(84, 30))
        self.assertEqual(m.dumps(self.result(self.call(*gcd, {"a": m.to_data(84), "b": 30}))), expected)
        padded = m.to_data(84)
        padded["version"] = 5  # Not the canonical minimum format of an integer.
        result_operand = m.to_data(m.operations.integer_gcd(6, 4))
        for operand in (padded, result_operand, {"format": "mathforge"},
                        dict(m.to_data(84), version=7), dict(m.to_data(84), extra=True)):
            with self.subTest(operand=operand):
                error = self.error(self.call(*gcd, {"a": operand, "b": 30}))
                self.assertEqual((error["stage"], error["field"]), ("decoding", "arguments.a"))
        # A v6 record passes decoding (later families take records as arguments); the producer rejects this one.
        record = m.to_data(m.operations.continued_fraction(m.Rational(7, 3)).value)
        self.assertEqual(record["version"], 6)
        error = self.error(self.call("mathforge.ntheory", "continued_fraction", {"value": record}))
        self.assertEqual((error["stage"], error["code"]), ("computation", "invalid_input"))
        nested = self.call("mathforge.analysis", "function_evaluate", {
            "expression": {"text": "x^2", "variables": ["x"]}, "bindings": {"x": m.to_data(m.sqrt(2))}})
        self.assertEqual(self.result(nested).value, m.functions.as_real_expr(m.Rational(2)))

    def test_one_identity_has_one_symbol_across_arguments(self):
        """Envelopes decode one by one, so decoding ends by checking every symbol together, as compute does."""
        x, other = m.Symbol("x", uid=X_ID), m.Symbol("other", uid=X_ID)
        f = m.parse_function("x*x", symbols={"x": x})
        compute = call_tool("mathforge.compute", {"operation": "function_differentiate", "arguments": {
            "expression": m.to_data(f), "variable": m.to_data(other), "n": m.to_data(1)}})
        expected = self.error(compute)
        self.assertEqual((expected["stage"], expected["field"]), ("decoding", "arguments"))
        for symbol_ids, expression in ((None, m.to_data(f)), ({"x": X_ID}, {"text": "x*x", "variables": ["x"]})):
            with self.subTest(friendly=symbol_ids is not None):
                response = self.call("mathforge.analysis", "function_differentiate",
                                     {"expression": expression, "variable": m.to_data(other)}, symbol_ids=symbol_ids)
                self.assertEqual(self.error(response), expected)

    def test_kinds_cover_the_registry(self):
        from mathforge.mcp.kinds import KINDS, definitions
        self.assertEqual(tuple(KINDS), registry.KIND_NAMES)
        envelope = {"$ref": "#/$defs/envelope"}
        self.assertEqual(KINDS["object"].schema, {"anyOf": [envelope]})
        table = definitions(registry.KIND_NAMES)
        self.assertEqual(list(table), [*registry.KIND_NAMES, "envelope"])
        self.assertEqual(table["envelope"]["properties"]["format"], {"const": "mathforge"})
        self.assertEqual(definitions(["int_pairs"]), {"int_pairs": KINDS["int_pairs"].schema,
                                                      "int": KINDS["int"].schema, "envelope": table["envelope"]})
        for name, kind in KINDS.items():
            with self.subTest(kind=name):
                self.assertEqual(kind.name, name)
                self.assertEqual(kind.schema["anyOf"][-1], envelope)
                self.assertEqual(len(kind.schema["anyOf"]), 1 if name == "object" else 2)
                self.assertTrue(set(definitions([name])) <= set(table))

    def test_encode_rules(self):
        from mathforge.limits import ComputationLimits
        from mathforge.mcp.kinds import KINDS, Context
        x = m.Symbol("x", uid=X_ID)
        cases = [
            ("int", 5, 5), ("int", 2 ** 53 - 1, 2 ** 53 - 1), ("int", 2 ** 53, "9007199254740992"),
            ("int", -2 ** 53, "-9007199254740992"), ("rational", m.Rational(3), "3/1"),
            ("rational", m.Rational(-1, 3), "-1/3"), ("rational", 3, 3), ("scalar", m.sqrt(2), "sqrt(2)"),
            ("scalar", m.Rational(1, 2), "1/2"), ("point", "-inf", "-inf"), ("side", "left", "left"),
            ("complex", m.ComplexAlgebraic(3, 4), ["3/1", "4/1"]), ("int_pairs", ((2, 6), (8, 9)), [[2, 6], [8, 9]]),
            ("matrix", m.Matrix(((1, 2),)), [["1/1", "2/1"]]), ("symbol", x, "x"), ("symbols", (x,), ["x"]),
            ("bindings", {x: 0}, {"x": 0}), ("object", 5, m.to_data(5)),
            ("function", m.real_function(m.functions.sin(x), (x,)), {"text": "sin(x)", "variables": ["x"]}),
            ("rational_function", m.rational_function(1, x, variable=x), {"text": "(1)/(x)", "variable": "x"}),
            ("distribution", m.FiniteDistribution((0, 1), (m.Rational(1, 2), m.Rational(1, 2))),
             {"outcomes": ["0/1", "1/1"], "probabilities": ["1/2", "1/2"]}),
        ]
        # Values without a friendly form that decodes to the identical object fall back to their envelope.
        integer = m.Symbol("x", domain=m.Integers, uid=X_ID)  # A name decodes to a real symbol.
        root = m.operations.real_roots(x ** 3 - 2, x).value[0].root  # The real cube root of 2.
        hole = m.rational_function(m.Polynomial((-1, 0, 1), x), m.Polynomial((-1, 1), x), variable=x)
        empty = m.Matrix((), ncols=2)
        cases += [("complex", 2, m.to_data(2)), ("symbol", integer, m.to_data(integer)),
                  ("scalar", root, m.to_data(root)), ("rational_function", hole, m.to_data(hole)),
                  ("int", 10 ** 2000, m.to_data(10 ** 2000)), ("matrix", empty, m.to_data(empty))]
        for kind, value, expected in cases:
            with self.subTest(kind=kind, value=value):
                encoded = KINDS[kind].encode(value)
                self.assertEqual(encoded, expected)
                context = Context(ComputationLimits(), symbol_ids((value,)))
                decoded = KINDS[kind].decode(encoded, context)
                self.assertEqual(m.to_data(_freeze(decoded)), m.to_data(_freeze(value)))

    def test_malformed_arguments_are_located_errors(self):
        """No input escapes call_tool as an exception; a decoding error names the argument it concerns."""
        from mathforge.mcp.kinds import KINDS
        malformed = (None, True, 1.5, "", "\x00", "x" * 2000, [], [None], [[1.5]], {}, {"text": None},
                     {"format": "mathforge"}, {"format": "mathforge", "version": 6, "kind": "object", "data": []},
                     {"outcomes": [1], "probabilities": "1"}, {"text": "x", "variables": "x"}, {"x": [1]})
        for tool in area_tools():
            for spec in registry.tool_operations(tool):
                valid = {argument.name: KINDS[argument.kind].encode(ast.literal_eval(argument.default or "0"))
                         for argument in spec.arguments}
                for argument in spec.arguments:
                    for value in malformed:
                        with self.subTest(operation=spec.name, argument=argument.name, value=value):
                            response = self.call(tool, spec.name, dict(valid, **{argument.name: value}))
                            error = response.payload["error"]
                            if error is not None and error["stage"] == "decoding":
                                self.assertTrue(error["field"].startswith("arguments."), error)
        # Well-formed envelopes of the wrong type inside compound forms reach constructors while decoding.
        function = {"text": "x", "variables": ["x"]}
        for envelope in (m.to_data(m.Symbol("q")), m.to_data(("a", 1)), m.to_data("text"), m.to_data(None)):
            for tool, operation, arguments in (
                    ("mathforge.linalg", "trace", {"A": [[envelope, 1], [2, 3]]}),
                    ("mathforge.stats", "expectation", {"distribution": {"outcomes": [envelope, 1],
                                                                         "probabilities": ["1/2", "1/2"]}}),
                    ("mathforge.algebra", "complex_add", {"left": [envelope, 1], "right": [1, 0]}),
                    ("mathforge.analysis", "function_evaluate", {"expression": function, "bindings": {"x": envelope}}),
                    ("mathforge.ntheory", "crt", {"congruences": [[envelope, 3]]}),
                    ("mathforge.analysis", "function_jacobian", {"expressions": [envelope], "variables": ["x"]})):
                with self.subTest(operation=operation, envelope=envelope["data"]["type"]):
                    self.assertTrue(self.call(tool, operation, arguments).is_error)

    def test_area_schemas_define_each_operation_and_kind_once(self):
        from mathforge.mcp.kinds import arguments_schema, definitions
        schemas = input_schemas()
        for tool in area_tools():
            operations = registry.tool_operations(tool)
            with self.subTest(tool=tool):
                schema = schemas[tool]
                self.assertEqual(schema["required"], ["operation", "arguments"])
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(set(schema["properties"]), {"operation", "arguments", "symbol_ids", "limits"})
                self.assertEqual(schema["properties"]["operation"]["enum"], [spec.name for spec in operations])
                self.assertEqual(schema["properties"]["arguments"]["anyOf"],
                                 [{"$ref": "#/$defs/" + spec.name} for spec in operations])
                kinds = definitions(argument.kind for spec in operations for argument in spec.arguments)
                entries = {spec.name: {key: value for key, value in arguments_schema(spec).items() if key != "$defs"}
                           for spec in operations}
                self.assertEqual(list(schema["$defs"]), [*entries, *kinds])
                self.assertEqual(schema["$defs"], entries | kinds)
                for spec in operations:
                    self.assertTrue(schema["$defs"][spec.name]["description"].startswith(spec.name + ": "))
        variance = arguments_schema(registry.spec("variance"))
        self.assertEqual(variance["required"], ["data"])
        self.assertFalse(variance["additionalProperties"])
        self.assertEqual(variance["properties"], {"data": {"$ref": "#/$defs/rational_list"},
                                                  "ddof": {"$ref": "#/$defs/int", "default": 0}})
        self.assertEqual(variance["$defs"], definitions(["rational_list", "int"]))
        catalog = schemas["mathforge.catalog"]
        self.assertEqual((catalog["required"], set(catalog["properties"])), ([], {"query", "area", "limit"}))
        self.assertEqual(catalog["properties"]["area"]["enum"], list(registry.AREAS))

    def test_input_schemas_are_portable(self):
        """Every tool: an object root without a combinator, no if/then/else, single type names, resolvable $refs.

        The new tools keep to the keywords of the frozen v0.7 schemas plus $defs and $ref, without "not"; the v0.7
        name schema keeps its frozen {"not": {"const": "sqrt"}}.
        """
        frozen = published("mcp_v7_input_schemas.json")
        allowed = {key for schema in frozen.values() for key, _ in schema_keywords(schema)} - {"not"}
        allowed |= {"$defs", "$ref"}
        schemas = input_schemas()
        self.assertGreaterEqual(len(schemas), 37)
        for name, schema in schemas.items():
            with self.subTest(tool=name):
                self.assertEqual(schema["type"], "object")
                self.assertFalse({"anyOf", "oneOf", "allOf", "not", "if", "then", "else", "$ref"} & set(schema))
                used = schema_keywords(schema)
                self.assertFalse({"if", "then", "else"} & {key for key, _ in used})
                for key, value in used:
                    if key == "type":
                        self.assertIs(type(value), str)
                    elif key == "$ref":
                        self.assertTrue(value.startswith("#/$defs/"), value)
                        self.assertIn(value.removeprefix("#/$defs/"), schema["$defs"])
                    elif key == "not":
                        self.assertIn(name, frozen)
                        self.assertEqual(value, {"const": "sqrt"})
                if name not in frozen:
                    self.assertLessEqual({key for key, _ in used}, allowed)
                if "$defs" in schema:  # Each definition is used.
                    references = {value for key, value in used if key == "$ref"}
                    self.assertEqual(references, {"#/$defs/" + entry for entry in schema["$defs"]})

    def test_schemas_accept_what_the_tools_accept(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("Development-only jsonschema is unavailable")
        from mathforge.mcp.kinds import KINDS, arguments_schema
        from mathforge.mcp.schemas import catalog_schema, output_schema
        schemas = input_schemas()
        for schema in (*schemas.values(), output_schema(), catalog_schema()):
            Draft202012Validator.check_schema(schema)
        ntheory = Draft202012Validator(schemas["mathforge.ntheory"])
        self.assertTrue(ntheory.is_valid({"operation": "integer_gcd", "arguments": {"a": 6, "b": "4"}}))
        self.assertTrue(ntheory.is_valid({"operation": "continued_fraction", "arguments": {"value": "7/3"}}))
        for invalid in ({"operation": "integer_gcd", "arguments": {"a": 0.5, "b": 4}},
                        {"operation": "integer_gcd", "arguments": {"a": 6}},
                        {"operation": "integer_gcd", "arguments": {"a": 6, "b": 4, "c": 1}},
                        {"operation": "binomial", "arguments": {"n": 5, "k": 2}}):
            self.assertFalse(ntheory.is_valid(invalid), invalid)
        # The schema lets "arguments" match another operation of the tool; the server checks the operation named.
        crossed = {"operation": "integer_gcd", "arguments": {"n": 5}}
        self.assertTrue(ntheory.is_valid(crossed))
        self.assertEqual(self.error(call_tool("mathforge.ntheory", crossed))["field"], "arguments.n")
        for operation, positional in frontend_samples().items():
            spec = registry.spec(operation)
            arguments = {argument.name: KINDS[argument.kind].encode(value)
                         for argument, value in zip(spec.arguments, positional)}
            with self.subTest(operation=operation):
                Draft202012Validator(schemas[tool_of(operation)]).validate(
                    {"operation": operation, "arguments": arguments, "symbol_ids": {"x": X_ID, "y": Y_ID}})
                Draft202012Validator(arguments_schema(spec)).validate(arguments)  # Complete on its own
        payload = self.call("mathforge.ntheory", "continued_fraction", {"value": "415/93"}).payload
        Draft202012Validator(output_schema()).validate(payload)
        catalog = Draft202012Validator(catalog_schema())
        catalog.validate(call_tool("mathforge.catalog", {"limit": 200}).payload)
        catalog.validate(call_tool("mathforge.catalog", {"limit": 0}).payload)

    def test_catalog_filters_sorts_and_limits(self):
        listing = call_tool("mathforge.catalog", {"limit": 200}).payload
        everything = listing["operations"]
        # "total" counts the whole catalog beyond the limit, so it stays right when the catalog outgrows the listing.
        self.assertEqual(listing["total"], len(registry.specs()) + len(LEGACY_CATALOG))
        self.assertEqual(len(everything), min(listing["total"], 200))
        self.assertEqual(everything, sorted(everything, key=lambda entry: (entry["area"], entry["name"])))
        entries = {entry["name"]: entry for entry in everything}
        self.assertEqual(entries["solve"], {"name": "solve", "area": "equations", "tool": "mathforge.solve",
                                            "summary": "Solve a real univariate polynomial/rational equation or inequality.",
                                            "arguments": [], "returns": ""})
        self.assertEqual(entries["expand"]["tool"], "")
        for spec in registry.specs().values():
            self.assertEqual(entries[spec.name]["tool"], tool_of(spec.name))
        defaults = {(name, argument["name"]): argument["default"] for name, entry in entries.items()
                    for argument in entry["arguments"] if not argument["required"]}
        self.assertEqual(defaults, {("complex_algebraic", "imag"): 0, ("variance", "ddof"): 0,
                                    ("stddev", "ddof"): 0, ("covariance", "ddof"): 0,
                                    ("function_differentiate", "n"): 1, ("function_partial_diff", "n"): 1,
                                    ("function_taylor", "point"): 0, ("function_taylor", "order"): 5,
                                    ("function_limit", "side"): "both"})
        default = call_tool("mathforge.catalog", {}).payload
        self.assertEqual(len(default["operations"]), 50)
        self.assertEqual(default["operations"], everything[:50])
        # "total" counts every match, beyond the limit too: there is no paging, so a client narrows or raises limit.
        self.assertEqual(default["total"], len(registry.specs()) + len(LEGACY_CATALOG))
        for request in ({"area": "stats", "limit": 1}, {"query": "continued", "limit": 1}):
            with self.subTest(request=request):
                payload = call_tool("mathforge.catalog", request).payload
                matches = call_tool("mathforge.catalog", dict(request, limit=200)).payload["operations"]
                self.assertEqual((payload["total"], payload["operations"]), (len(matches), matches[:1]))
        stats = call_tool("mathforge.catalog", {"area": "stats", "limit": 200}).payload["operations"]
        self.assertEqual({entry["area"] for entry in stats}, {"stats"})
        self.assertEqual(len(stats), len(registry.area_operations("stats")))
        self.assertIn("continued_fraction", [e["name"] for e in call_tool("mathforge.catalog", {
            "query": "CONTINUED Fraction"}).payload["operations"]])
        by_area = call_tool("mathforge.catalog", {"query": "NTHEORY", "limit": 200}).payload["operations"]
        self.assertIn("integer_gcd", [entry["name"] for entry in by_area])
        # Every error keeps the catalog payload shape, also for limits, which the catalog does not take.
        for request, field in (({"limit": 0}, "limit"), ({"limit": 201}, "limit"), ({"limit": True}, "limit"),
                               ({"limit": None}, "limit"), ({"area": "calculus"}, "area"), ({"area": None}, "area"),
                               ({"query": "x" * 129}, "query"), ({"query": 5}, "query"), ({"query": None}, "query"),
                               ({"unexpected": 1}, None), ({"limits": "bad"}, None), ({"limits": {"max_work": 1}}, None)):
            with self.subTest(request=request):
                response = call_tool("mathforge.catalog", request)
                self.assertTrue(response.is_error)
                self.assertEqual(set(response.payload), {"api_version", "operations", "total", "error"})
                self.assertEqual((response.payload["api_version"], response.payload["operations"],
                                  response.payload["total"]), (1, [], 0))
                self.assertEqual(response.payload["error"]["field"], field)
        self.assertTrue(call_tool("mathforge.catalog", None).is_error)

    def test_failures_keep_the_symbol_table(self):
        response = self.call("mathforge.analysis", "function_differentiate",
                             {"expression": {"text": "sin(x", "variables": ["x"]}, "variable": "x"},
                             symbol_ids={"x": X_ID})
        error = self.error(response)
        self.assertEqual((error["stage"], error["field"]), ("decoding", "arguments.expression.text"))
        self.assertEqual(response.payload["symbols"], [{"name": "x", "uid": X_ID}])
        for ids in ({"x": "not-a-uuid"}, {"x": X_ID, "y": X_ID}, {"1x": X_ID}, ["x"]):
            with self.subTest(symbol_ids=ids):
                error = self.error(self.call("mathforge.analysis", "function_differentiate",
                                             {"expression": {"text": "x", "variables": ["x"]}, "variable": "x"},
                                             symbol_ids=ids))
                self.assertTrue(error["field"].startswith("symbol_ids"), error)

    def test_request_limits_apply_to_area_tools(self):
        error = self.error(self.call("mathforge.discrete", "factorial", {"n": 100}, limits={"max_work": 1}))
        self.assertEqual(error["code"], "resource_limit_exceeded")
        self.assertEqual(self.error(self.call("mathforge.discrete", "factorial", {"n": 5}, limits={"max_work": 0})
                                    )["field"], "limits.max_work")
        padded = {"n": 5, "padding": "x" * 1_100_000}  # Beyond the request byte limit.
        self.assertEqual(self.error(self.call("mathforge.discrete", "factorial", padded))["code"],
                         "resource_limit_exceeded")
        # A friendly list costs one node per item while it decodes, as an envelope's walk does.
        error = self.error(self.call("mathforge.stats", "mean", {"data": [1] * 150_000}))
        self.assertEqual((error["stage"], error["code"], error["field"]),
                         ("decoding", "resource_limit_exceeded", "arguments.data"))
        error = self.error(self.call("mathforge.stats", "mean", {"data": [1, 2]}, limits={"max_nodes": 1}))
        self.assertEqual((error["stage"], error["code"], error["field"]),
                         ("decoding", "resource_limit_exceeded", "arguments.data"))


if __name__ == "__main__":
    unittest.main()
