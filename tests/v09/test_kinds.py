"""The argument kinds added in v0.9: polynomial, interval_set and relation.

mcp.kinds gives each kind a friendly JSON form beside the canonical envelope of its value, and operations_v6.run
admits one exact Python type for it. No production operation takes these kinds yet, so the throwaway family
demo_kinds_v9 carries them through operations_v6.run and through the MCP area tools.
"""

import ast
import importlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import limits, operations_v6, registry
from mathforge.contracts import ExecutionStatus
from mathforge.lean import FALLBACK, run_value
from mathforge.limits import Budget, ComputationLimits, computation
from mathforge.mcp.boundary import call_tool, operand_tags
from mathforge.mcp.kinds import KINDS, MAX_INTERVALS, RESULT_COPIES, Context, arguments_schema, definitions
from mathforge.mcp.schemas import input_schemas

from . import demo_kinds_v9 as demo

SRC = Path(mf.__file__).resolve().parents[1]  # src/ in a checkout, site-packages in the distribution harness.
TOOL = Path(__file__).resolve().parents[2] / "tools" / "generate_api.py"
V08_KINDS = ("int", "int_list", "int_pairs", "rational", "rational_list", "scalar", "complex", "matrix",
             "distribution", "function", "function_list", "rational_function", "symbol", "symbols", "bindings",
             "point", "side", "object")
NEW_KINDS = ("polynomial", "interval_set", "relation")
RELATIONS = ("=", "<", "<=", ">", ">=")
# The JSON Schema keywords of the frozen v0.7 input schemas, without "not", plus $defs and $ref: the portable subset
# that tests/v08/test_mcp_areas.py holds every area tool to.
KEYWORDS = frozenset(("$defs", "$ref", "additionalProperties", "anyOf", "const", "default", "description", "enum",
                      "format", "items", "maxItems", "maxLength", "maxProperties", "maximum", "minItems", "minimum",
                      "pattern", "properties", "propertyNames", "required", "type", "uniqueItems"))
ENVELOPE = {"$ref": "#/$defs/envelope"}
HANDLE = {"$ref": "#/$defs/handle"}  # Since v0.9: a handle may stand for an envelope.
X_ID = "00000000-0000-4000-8000-000000000901"
X = mf.Symbol("x", uid=X_ID)
SQRT2 = mf.RealAlgebraicRoot((-2, 0, 1), 1)  # The end that the text "sqrt(2)" decodes to.
SQRT10 = mf.RealAlgebraicRoot((-10, 0, 1), 1)
CUBE_ROOT = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)  # The real cube root of 2: no closed text denotes it.
# A root of degree 19 that costs some hundreds of work units to rebuild from its envelope.
COSTLY = mf.RealAlgebraicRoot((1, -3, 0, 2, -1, 0, 0, 5, -2, 1, 0, 0, 3, 0, 0, 0, -1, 2, 0, 1), 0)
NOT_OPERANDS = "Results, workspaces and execution metadata are not operands"  # v0.8's error for a foreign tag.
# Each demo operation: its MCP tool, its name and the name of its one argument.
ECHO_POLYNOMIAL = ("mathforge.algebra", "demo_echo_polynomial", "polynomial")
ECHO_INTERVALS = ("mathforge.algebra", "demo_echo_interval_set", "intervals")
ECHO_RELATION = ("mathforge.equations", "demo_echo_relation", "relation")
POLYNOMIAL = mf.Polynomial((-1, 0, 1), X)
INTERVALS = mf.IntervalSet((mf.Interval(None, SQRT2, False, True), mf.Interval(3, SQRT10), mf.Interval(5, None, True)))
VALUES = ((ECHO_POLYNOMIAL, POLYNOMIAL), (ECHO_INTERVALS, INTERVALS), (ECHO_RELATION, "<="))
# The default limits that the copies in a Result share, divided by RESULT_COPIES = 10: the reload check's budget.
RELOAD_LIMITS = ComputationLimits(max_work=200_000, max_nodes=10_000, max_refinements=10_000, max_evidence_items=10_000)
WIDE = ComputationLimits(max_work=10 ** 9, max_nodes=10 ** 7, max_refinements=10 ** 7)


def decode(kind, raw, symbol_ids=None):
    """Decode raw as the argument "value" of a request, in a position of its own, as decode_arguments does."""
    context = Context(ComputationLimits(), symbol_ids)
    with context.at(".value"):
        return KINDS[kind].decode(raw, context)


def interval(lower, upper, left_closed=False, right_closed=False):
    """The friendly JSON form of one interval."""
    return {"lower": lower, "upper": upper, "left_closed": left_closed, "right_closed": right_closed}


def keywords(schema):
    """(keyword, value) for each keyword of a schema and of its subschemas; a property name is no keyword."""
    found, pending = [], [schema]
    while pending:
        node = pending.pop()
        for key, value in node.items():
            found.append((key, value))
            if key in ("properties", "$defs"):
                pending.extend(value.values())
            elif key in ("items", "additionalProperties", "propertyNames") and type(value) is dict:
                pending.append(value)
            elif key == "anyOf":
                pending.extend(value)
    return found


def load_tool():
    spec = importlib.util.spec_from_file_location("generate_api_kinds", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KindTableTests(unittest.TestCase):
    def test_new_kinds_follow_the_v08_kinds(self):
        """The order is published, so the kinds of v0.8 keep their places and the new ones follow "object"."""
        self.assertEqual(registry.KIND_NAMES, (*V08_KINDS, *NEW_KINDS))
        self.assertEqual(tuple(KINDS), registry.KIND_NAMES)
        self.assertEqual(list(definitions(V08_KINDS)), [*V08_KINDS, "envelope", "handle"])
        self.assertEqual(list(definitions(registry.KIND_NAMES)), [*V08_KINDS, *NEW_KINDS, "envelope", "handle"])
        self.assertEqual(list(definitions(["interval_set"])), ["scalar", "interval_set", "envelope", "handle"])
        self.assertEqual(list(definitions(["relation", "polynomial"])), ["polynomial", "relation", "envelope",
                                                                         "handle"])

    def test_schemas_are_a_friendly_form_or_an_envelope(self):
        name = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]*$", "maxLength": 128}
        end = {"anyOf": [{"$ref": "#/$defs/scalar"}, {"type": "null"}]}
        friendly = {
            "polynomial": {"type": "object", "additionalProperties": False, "required": ["text", "variable"],
                           "properties": {"text": {"type": "string", "maxLength": 16384}, "variable": name}},
            "interval_set": {"type": "array", "description": "A union of real intervals; null is an infinite end.",
                             "maxItems": 32,
                             "items": {"type": "object", "additionalProperties": False,
                                       "required": ["lower", "upper", "left_closed", "right_closed"],
                                       "properties": {"lower": end, "upper": end, "left_closed": {"type": "boolean"},
                                                      "right_closed": {"type": "boolean"}}}},
            "relation": {"type": "string", "enum": list(RELATIONS)},
        }
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(KINDS[kind].name, kind)
                self.assertEqual(KINDS[kind].schema, {"anyOf": [friendly[kind], ENVELOPE, HANDLE]})
                used = keywords(KINDS[kind].schema)
                self.assertLessEqual({key for key, _ in used}, KEYWORDS)
                self.assertEqual({type(value) for key, value in used if key == "type"}, {str})

    def test_a_foreign_value_encodes_as_its_envelope(self):
        """A value that is not of the kind has no friendly form: it encodes as its canonical envelope."""
        foreign = {"polynomial": (5, "x", X, mf.rational_function(X, 1, variable=X)),
                   "interval_set": (5, (), mf.Interval(0, 1), mf.EmptySet()),
                   "relation": (5, None, "!=", "==")}
        for kind, values in foreign.items():
            for value in values:
                with self.subTest(kind=kind, value=value):
                    self.assertEqual(KINDS[kind].encode(value), mf.to_data(value))


class PolynomialKindTests(unittest.TestCase):
    def test_friendly_form_decodes_to_a_polynomial(self):
        half = mf.Rational(1, 2)
        cases = (("x^2 - 1", (-1, 0, 1)), ("(x + 1)^3/2 - 0.5*x", (half, 1, mf.Rational(3, 2), half)),
                 ("x/2 + x/(1 + 1)", (0, 1)), ("sqrt(4)*x", (0, 2)), ("7", (7,)), ("0", (0,)))
        for text, coefficients in cases:
            with self.subTest(text=text):
                value = decode("polynomial", {"text": text, "variable": "x"}, {"x": X_ID})
                self.assertIs(type(value), mf.Polynomial)
                self.assertEqual(value, mf.Polynomial(coefficients, X))
        fresh = decode("polynomial", {"text": "t + 1", "variable": "t"})  # Without symbol_ids: a new real symbol.
        self.assertEqual((fresh.variable.name, fresh.variable.domain, fresh.variable.assumptions), ("t", mf.Reals, ()))
        self.assertEqual(fresh, mf.Polynomial((1, 1), fresh.variable))

    def test_envelope_decodes_to_the_same_polynomial(self):
        integer = mf.Polynomial((1, 1), mf.Symbol("n", domain=mf.Integers))
        for value in (POLYNOMIAL, integer):
            with self.subTest(value=value):
                decoded = decode("polynomial", mf.to_data(value))
                self.assertIs(type(decoded), mf.Polynomial)
                self.assertEqual(decoded, value)

    def test_encode_round_trips(self):
        integer = mf.Polynomial((1, 1), mf.Symbol("n", domain=mf.Integers))  # A name decodes to a real symbol.
        cases = ((POLYNOMIAL, {"text": "x^2 - 1", "variable": "x"}),
                 (mf.Polynomial((mf.Rational(-1, 2), 3), X), {"text": "3*x - 1/2", "variable": "x"}),
                 (mf.Polynomial((0,), X), {"text": "0", "variable": "x"}),
                 (integer, mf.to_data(integer)))
        for value, expected in cases:
            with self.subTest(value=value):
                encoded = KINDS["polynomial"].encode(value)
                self.assertEqual(encoded, expected)
                self.assertEqual(decode("polynomial", encoded, {"x": X_ID}), value)


class IntervalSetKindTests(unittest.TestCase):
    def test_friendly_form_decodes_to_the_canonical_set(self):
        """The intervals may come in any order, overlap or be empty: the value is their canonical union."""
        decoded = decode("interval_set", [interval(3, None, True), interval("-1/2", "sqrt(2)", False, True),
                                          interval(None, 0, False, True), interval(5, 5)])
        self.assertIs(type(decoded), mf.IntervalSet)
        self.assertEqual(decoded.intervals, (mf.Interval(None, SQRT2, False, True), mf.Interval(3, None, True)))
        self.assertEqual(decode("interval_set", []), mf.IntervalSet(()))
        self.assertEqual(decode("interval_set", [interval(None, None)]), mf.IntervalSet((mf.Interval(None, None),)))
        self.assertEqual(decode("interval_set", [interval(2, 2, True, True)]),
                         mf.IntervalSet((mf.Interval(2, 2, True, True),)))

    def test_ends_are_rationals_or_real_algebraic_roots(self):
        """Exactly what Interval admits: closed text is converted as the solver converts its endpoints."""
        cases = ((2, mf.Rational(2)), ("-3/4", mf.Rational(-3, 4)), ("0.25", mf.Rational(1, 4)),
                 ("sqrt(2)*sqrt(8)", mf.Rational(4)), ("sqrt(2)", SQRT2), ("sqrt(8) - sqrt(2)", SQRT2),
                 ("-sqrt(2)", mf.RealAlgebraicRoot((-2, 0, 1), 0)),
                 ("(1 + sqrt(5))/2", mf.RealAlgebraicRoot((-1, -1, 1), 1)),
                 # An end may itself be an envelope, as the elements of the other compound forms may.
                 (mf.to_data(7), mf.Rational(7)), (mf.to_data(mf.Rational(1, 3)), mf.Rational(1, 3)),
                 (mf.to_data(CUBE_ROOT), CUBE_ROOT),
                 (mf.to_data(mf.sqrt(2) + 1), mf.RealAlgebraicRoot((-1, -2, 1), 1)))
        for raw, expected in cases:
            with self.subTest(end=raw):
                (decoded,) = decode("interval_set", [interval(raw, None, True)])
                self.assertEqual(decoded.lower, expected)
                self.assertIs(type(decoded.lower), type(expected))

    def test_envelope_decodes_to_the_same_set(self):
        decoded = decode("interval_set", mf.to_data(INTERVALS))
        self.assertIs(type(decoded), mf.IntervalSet)
        self.assertEqual(decoded, INTERVALS)
        reordered = mf.to_data(INTERVALS)
        reordered["data"]["intervals"].reverse()  # An envelope must be canonical, unlike the friendly form.
        with self.assertRaises(mf.MathForgeError) as caught:
            decode("interval_set", reordered)
        self.assertEqual(caught.exception.field, "arguments.value")

    def test_encode_round_trips(self):
        whole = mf.IntervalSet((mf.Interval(None, None),))
        large = mf.IntervalSet((mf.Interval(2 ** 53, 10 ** 30),))
        huge = mf.IntervalSet((mf.Interval(10 ** 1100, None),))  # Its end is longer than closed text may be.
        cases = ((mf.IntervalSet(()), []), (whole, [interval(None, None)]),
                 (mf.IntervalSet((mf.Interval(None, 1, False, True), mf.Interval(mf.Rational(3, 2), 2))),
                  [interval(None, 1, False, True), interval("3/2", 2)]),
                 (large, [interval("9007199254740992", "1000000000000000000000000000000")]),
                 # A real algebraic root has no friendly form: the end is its envelope.
                 (mf.IntervalSet((mf.Interval(0, CUBE_ROOT, True),)), [interval(0, mf.to_data(CUBE_ROOT), True)]),
                 (mf.IntervalSet((mf.Interval(SQRT2, 2, True, True),)), [interval(mf.to_data(SQRT2), 2, True, True)]),
                 (huge, mf.to_data(huge)))
        for value, expected in cases:
            with self.subTest(value=mf.render_text(value)[:60]):
                encoded = KINDS["interval_set"].encode(value)
                self.assertEqual(encoded, expected)
                self.assertEqual(decode("interval_set", encoded), value)


class RelationKindTests(unittest.TestCase):
    def test_every_relation_decodes_and_encodes_as_itself(self):
        for relation in RELATIONS:
            with self.subTest(relation=relation):
                self.assertEqual(decode("relation", relation), relation)
                self.assertEqual(decode("relation", mf.to_data(relation)), relation)
                self.assertEqual(KINDS["relation"].encode(relation), relation)

    def test_other_text_is_no_relation(self):
        for raw in ("==", "!=", "", "le", " <", "=<", None, True, 1, ["="]):
            with self.subTest(raw=raw):
                with self.assertRaises(mf.MathForgeError) as caught:
                    decode("relation", raw)
                self.assertEqual((caught.exception.code, caught.exception.field), ("invalid_input", "arguments.value"))


def v08_positions(payload):
    """(tool, operation, arguments, field) for positions of v0.8 kinds that decode an envelope, each holding payload.

    The kinds are int, rational, scalar, symbol, point, side, function and rational_function as whole arguments, a
    bindings value, and an element of rational_list, int_list, function_list, matrix, complex, int_pairs and
    distribution.
    """
    function = {"text": "x^2", "variables": ["x"]}
    return (("mathforge.ntheory", "integer_gcd", {"a": payload, "b": 4}, "a"),
            ("mathforge.ntheory", "continued_fraction", {"value": payload}, "value"),
            ("mathforge.algebra", "complex_algebraic", {"real": payload}, "real"),
            ("mathforge.analysis", "function_integrate",
             {"expression": function, "variable": "x", "lower": 0, "upper": payload}, "upper"),
            ("mathforge.analysis", "function_differentiate", {"expression": function, "variable": payload},
             "variable"),
            ("mathforge.analysis", "function_limit", {"expression": function, "variable": "x", "point": payload},
             "point"),
            ("mathforge.analysis", "function_limit",
             {"expression": function, "variable": "x", "point": 0, "side": payload}, "side"),
            ("mathforge.analysis", "function_evaluate", {"expression": payload, "bindings": {}}, "expression"),
            ("mathforge.analysis", "rational_antiderivative", {"expression": payload, "variable": "x"}, "expression"),
            ("mathforge.analysis", "function_evaluate", {"expression": function, "bindings": {"x": payload}},
             "bindings.x"),
            ("mathforge.stats", "mean", {"data": [1, payload]}, "data[1]"),
            ("mathforge.discrete", "multinomial", {"counts": [payload]}, "counts[0]"),
            ("mathforge.analysis", "function_jacobian", {"expressions": [payload], "variables": ["x"]},
             "expressions[0]"),
            ("mathforge.linalg", "trace", {"A": [[1, 2], [3, payload]]}, "A[1][1]"),
            ("mathforge.algebra", "complex_add", {"left": [0, payload], "right": [1, 0]}, "left[1]"),
            ("mathforge.ntheory", "crt", {"congruences": [[2, payload]]}, "congruences[0][1]"),
            ("mathforge.stats", "expectation", {"distribution": {"outcomes": [payload], "probabilities": [1]}},
             "distribution.outcomes[0]"))


class OperandTagTests(unittest.TestCase):
    def test_interval_sets_are_operands_of_interval_set_arguments_only(self):
        """Only the whole-argument envelope of an interval_set argument may hold an interval set or an interval.

        mathforge.compute keeps the operand tags that its frozen v0.7 schema enumerates, and every other kind, an end
        of an interval included, keeps v0.8's located decoding error.
        """
        self.assertFalse({"interval", "interval_set"} & set(operand_tags()))
        response = call_tool("mathforge.compute", {"operation": "integer_gcd",
                                                   "arguments": {"a": mf.to_data(INTERVALS), "b": mf.to_data(4)}})
        self.assertTrue(response.is_error)
        self.assertEqual((response.payload["error"]["stage"], response.payload["error"]["field"]),
                         ("decoding", "arguments.a"))
        self.assertEqual(decode("interval_set", mf.to_data(INTERVALS)), INTERVALS)
        self.assertEqual(decode("interval_set", mf.to_data(mf.Interval(0, 1))), mf.Interval(0, 1))  # run() rejects it.
        for kind, raw, position in (
                ("object", mf.to_data((INTERVALS, "<=")), ""), ("object", mf.to_data(mf.Interval(0, 1)), ""),
                ("polynomial", mf.to_data(INTERVALS), ""), ("relation", mf.to_data(mf.IntervalSet(())), ""),
                ("interval_set", [interval(0, mf.to_data(INTERVALS))], "[0].upper"),
                ("interval_set", [interval(mf.to_data(mf.Interval(0, 1)), None)], "[0].lower")):
            with self.subTest(kind=kind, raw=str(raw)[:60]):
                with self.assertRaises(mf.MathForgeError) as caught:
                    decode(kind, raw)
                self.assertEqual((caught.exception.code, caught.exception.field, str(caught.exception)),
                                 ("invalid_input", "arguments.value" + position, NOT_OPERANDS))
        for kind in NEW_KINDS:  # A Result stays no operand of any kind.
            with self.subTest(kind=kind):
                with self.assertRaises(mf.MathForgeError) as caught:
                    decode(kind, mf.to_data(mf.operations.integer_gcd(6, 4)))
                self.assertEqual(caught.exception.field, "arguments.value")

    def test_v08_kinds_keep_v08_errors_for_interval_envelopes(self):
        """The area tools answer an interval set or an interval in a v0.8 kind exactly as v0.8.0 does.

        Each response is v0.8.0's, checked against a copy of its sources: a decoding error that names the position.
        """
        payloads = {"interval set": INTERVALS, "empty set": mf.IntervalSet(()), "interval": mf.Interval(0, 1),
                    "set in a tuple": (INTERVALS, 1)}
        for label, value in payloads.items():
            for tool, operation, arguments, field in v08_positions(mf.to_data(value)):
                with self.subTest(payload=label, operation=operation, field=field):
                    response = call_tool(tool, {"operation": operation, "arguments": arguments})
                    self.assertTrue(response.is_error)
                    self.assertIsNone(response.payload["result"])
                    self.assertEqual(response.payload["error"],
                                     {"stage": "decoding", "code": "invalid_input", "message": NOT_OPERANDS,
                                      "field": "arguments." + field, "start": None, "end": None,
                                      # The keys that v0.9 adds to every error object:
                                      "reason": "invalid_input",
                                      "next_actions": [f'call mathforge.catalog with query "{operation}" to see the '
                                                       f"arguments of {operation} and their kinds",
                                                       'correct the input that "field" names and call again']})


class DecodingBudgetTests(unittest.TestCase):
    def test_envelopes_are_rebuilt_within_the_request_budget(self):
        """The kinds of v0.9 rebuild the value of an envelope, also of an interval end, in the request's budget.

        from_data, which the kinds of v0.8 call, opens a fresh budget for every envelope. The one other budget is the
        reload check's, which rebuilds a decoded interval_set argument once more within the default limits divided by
        RESULT_COPIES.
        """
        opened = []

        class RecordedBudget(Budget):  # The pattern of tests/v08/test_budget_v8.py.
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                opened.append(self)

        cases = (("polynomial", mf.to_data(POLYNOMIAL)), ("interval_set", mf.to_data(INTERVALS)),
                 ("interval_set", [interval(mf.to_data(COSTLY), None, True)]),
                 ("interval_set", [interval(0, mf.to_data(SQRT2))]), ("relation", mf.to_data("<")))
        for kind, raw in cases:
            with self.subTest(kind=kind, raw=str(raw)[:60]):
                opened.clear()
                with patch.object(limits, "Budget", RecordedBudget):
                    with computation(ComputationLimits(), fresh=True) as request:
                        decode(kind, raw, {"x": X_ID})
                self.assertIs(opened[0], request)
                if kind == "interval_set":
                    self.assertEqual(len(opened), 2)
                    self.assertEqual(opened[1].limits, RELOAD_LIMITS)
                    self.assertGreater(opened[1].work + opened[1].node_count, 0)
                else:
                    self.assertEqual(len(opened), 1)


class DemoFamilyTests(unittest.TestCase):
    """The three kinds end to end, through operations_v6.run and through the MCP area tools."""

    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(registry._testing_families(demo))

    def setUp(self):
        demo.CALLS.clear()
        self.addCleanup(demo.CALLS.clear)

    def call(self, target, value, **fields):
        tool, operation, name = target
        return call_tool(tool, dict(fields, operation=operation, arguments={name: value}))

    def result(self, response):
        self.assertFalse(response.is_error, response.payload)
        self.assertIsNone(response.payload["error"])
        return mf.from_data(response.payload["result"])

    def error(self, response):
        self.assertTrue(response.is_error, response.payload)
        return response.payload["error"]

    def test_run_echoes_a_value_of_each_kind(self):
        for (_, operation, name), value in VALUES:
            with self.subTest(operation=operation):
                result = operations_v6.run(operation, {name: value})
                self.assertIs(result.execution_status, ExecutionStatus.COMPLETED)
                self.assertEqual(result.value, value)
                self.assertIs(type(result.value), type(value))
                self.assertEqual((result.steps[0].rule, result.request.version), ("v6." + operation, 6))
                self.assertEqual(result.request.arguments, ((name, value),))
                self.assertTrue(mf.verify(result).verified)
                text = mf.dumps(result)
                self.assertEqual(mf.to_data(result)["version"], 6)
                self.assertEqual(mf.dumps(mf.loads(text)), text)
                self.assertTrue(mf.verify(mf.loads(text)).verified)
        self.assertEqual(demo.CALLS, [operation for (_, operation, _), _ in VALUES])
        for relation in RELATIONS:  # Exactly the relations that the MCP schema enumerates.
            self.assertEqual(operations_v6.run("demo_echo_relation", {"relation": relation}).value, relation)

    def test_run_rejects_other_python_types_before_produce(self):
        texts = {"polynomial": "a Polynomial", "intervals": "an IntervalSet",
                 "relation": "a relation: '=', '<', '<=', '>' or '>='"}
        near = {  # Besides a bool, a float, a str, a list and other records: the near misses of each kind.
            "polynomial": (X ** 2 - 1, mf.rational_function(X ** 2 - 1, 1, variable=X), (mf.Rational(-1), 0, 1)),
            "intervals": (mf.Interval(0, 1), (mf.Interval(0, 1),), mf.EmptySet(), mf.FiniteSet((1, 2))),
            "relation": ("==", "le", "", ("<", "=")),
        }
        for (_, operation, name), _ in VALUES:
            for value in (True, 1.5, "x", [1], mf.Matrix(((1,),)), mf.Rational(1, 2), None, 0, *near[name]):
                with self.subTest(operation=operation, value=value):
                    result = operations_v6.run(operation, {name: value})
                    self.assertIs(result.execution_status, ExecutionStatus.INVALID_INPUT)
                    self.assertEqual(result.error.code, "invalid_input")
                    if type(value) is float:  # No request holds a float, so it fails before the kinds are checked.
                        self.assertIsNone(result.request)
                        continue
                    self.assertEqual(result.error.message, f"Argument {name} of {operation} must be {texts[name]}")
                    self.assertEqual(result.request.version, 6)
        self.assertEqual(demo.CALLS, [])

    def test_lean_value_path_makes_the_same_checks(self):
        """The generic lean runner, which serves the value functions, leaves what run() rejects to run()."""
        for (_, operation, name), value in VALUES:
            with self.subTest(operation=operation):
                self.assertEqual(run_value(operation, {name: value}), value)
                for wrong in (5, True, "x", None, [1]):
                    self.assertIs(run_value(operation, {name: wrong}), FALLBACK)
        self.assertEqual(demo.CALLS, [operation for (_, operation, _), _ in VALUES])

    def test_area_tools_decode_the_friendly_forms(self):
        friendly = {"polynomial": {"text": "x^2 - 1", "variable": "x"}, "relation": "<=",
                    "intervals": [interval(5, None, True), interval(3, "sqrt(10)"),
                                  interval(None, "sqrt(2)", False, True)]}
        for target, value in VALUES:
            _, operation, name = target
            with self.subTest(operation=operation):
                response = self.call(target, friendly[name], symbol_ids={"x": X_ID})
                result = self.result(response)
                self.assertEqual(result.value, value)
                self.assertEqual(response.payload["result"]["version"], 6)
                self.assertEqual(response.payload["symbols"],
                                 [{"name": "x", "uid": X_ID}] if name == "polynomial" else [])
                self.assertEqual(mf.dumps(result), mf.dumps(operations_v6.run(operation, {name: value})))
                verified = call_tool("mathforge.verify", {"result": response.payload["result"]})
                self.assertTrue(self.result(verified).verified)
        self.assertIn("x^2 - 1", self.call(ECHO_POLYNOMIAL, friendly["polynomial"]).payload["text"])
        self.assertIn("(-inf, RootOf(t^2 - 2; real_index=1)] union (3, ",
                      self.call(ECHO_INTERVALS, friendly["intervals"]).payload["text"])

    def test_area_tools_take_the_envelope_of_a_value(self):
        for target, value in VALUES:
            _, operation, name = target
            with self.subTest(operation=operation):
                for encoded in (mf.to_data(value), KINDS[registry.spec(operation).arguments[0].kind].encode(value)):
                    result = self.result(self.call(target, encoded, symbol_ids={"x": X_ID}))
                    self.assertEqual(mf.dumps(result), mf.dumps(operations_v6.run(operation, {name: value})))
        demo.CALLS.clear()
        # The envelope of another operand passes decoding, as for every kind; run() then rejects its type.
        for target, value in ((ECHO_POLYNOMIAL, 5), (ECHO_INTERVALS, mf.Interval(0, 1)), (ECHO_INTERVALS, POLYNOMIAL),
                              (ECHO_RELATION, "!="), (ECHO_RELATION, 1)):
            with self.subTest(operation=target[1], value=value):
                error = self.error(self.call(target, mf.to_data(value)))
                self.assertEqual((error["stage"], error["code"]), ("computation", "invalid_input"))
                self.assertTrue(error["message"].startswith(f"Argument {target[2]} of {target[1]} must be "), error)
        # An interval set or an interval is an operand of an interval_set argument only.
        for target, value in ((ECHO_POLYNOMIAL, INTERVALS), (ECHO_POLYNOMIAL, mf.Interval(0, 1)),
                              (ECHO_RELATION, INTERVALS)):
            with self.subTest(operation=target[1], value=value):
                error = self.error(self.call(target, mf.to_data(value)))
                self.assertEqual((error["stage"], error["code"], error["field"], error["message"]),
                                 ("decoding", "invalid_input", "arguments." + target[2], NOT_OPERANDS))
        self.assertEqual(demo.CALLS, [])

    def test_costly_end_envelopes_stop_at_the_request_budget(self):
        """Ends that each fit the request's limits alone stop together where the request's work runs out.

        Each end is rebuilt within the request budget, not in a fresh budget of its own, so many costly ends stop
        after about one budget of work instead of spending a budget each.
        """
        with computation(fresh=True) as alone:
            mf.RealAlgebraicRoot(COSTLY.integer_coefficients, COSTLY.real_index)
        work = {"max_work": alone.work * 5 // 2}  # Room for two such ends, not for three.
        end = interval(mf.to_data(COSTLY), None, True)
        self.assertEqual(self.result(self.call(ECHO_INTERVALS, [end], limits=work)).value,
                         mf.IntervalSet((mf.Interval(COSTLY, None, True),)))
        error = self.error(self.call(ECHO_INTERVALS, [end] * 8, limits=work))
        self.assertEqual((error["stage"], error["code"], error["field"]),
                         ("decoding", "resource_limit_exceeded", "arguments.intervals[2].lower"))
        self.assertEqual(demo.CALLS, ["demo_echo_interval_set"])

    def test_a_friendly_interval_set_lists_at_most_32_intervals(self):
        """At the cap, a Result that holds the set still loads and verifies under the default limits."""
        self.assertEqual(MAX_INTERVALS, 32)
        # Ends in distinct quadratic fields, which cost more to rebuild and compare than rational ends.
        intervals = [interval(f"sqrt({4 * i * i + 4 * i + 2})", f"sqrt({4 * i * i + 8 * i + 5})") for i in range(33)]
        response = self.call(ECHO_INTERVALS, intervals[:32])
        self.assertEqual(len(self.result(response).value), 32)
        self.assertTrue(self.result(call_tool("mathforge.verify", {"result": response.payload["result"]})).verified)
        self.assertTrue(mf.verify(mf.from_data(response.payload["result"])).verified)
        error = self.error(self.call(ECHO_INTERVALS, intervals))
        self.assertEqual((error["stage"], error["code"], error["field"]),
                         ("decoding", "invalid_input", "arguments.intervals"))
        self.assertEqual(error["message"], "An interval set has at most 32 intervals")
        self.assertEqual(demo.CALLS, ["demo_echo_interval_set"])
        # The cap is the friendly form's: a larger set encodes as its envelope, which decodes.
        larger = mf.IntervalSet(tuple(mf.Interval(2 * i, 2 * i + 1) for i in range(33)))
        self.assertEqual(KINDS["interval_set"].encode(larger), mf.to_data(larger))
        self.assertEqual(decode("interval_set", mf.to_data(larger)), larger)

    def test_a_set_that_its_result_could_not_reload_is_refused(self):
        """Reloading a Result rebuilds every copy of a set that it holds within the default limits, so an interval_set
        argument, in either form, must rebuild from its envelope within the default limits divided by RESULT_COPIES:
        8 copies for a set operation on two sets, and a margin. Before, one open interval (sqrt(10^400),
        sqrt(10^400 + 1)) and 32 intervals with ends sqrt(10^24 + 2k), sqrt(10^24 + 2k + 1), which each decode within
        the request's limits, were echoed, and mathforge.verify could not reload the Results, as it still cannot reload
        those of operations_v6.run, which has no such check: now, in either form, they are resource_limit_exceeded at
        decoding, located at the argument. 32 intervals with small ends still complete and verify."""
        self.assertEqual(RESULT_COPIES, 10)
        close = [interval("sqrt(1" + "0" * 400 + ")", "sqrt(1" + "0" * 399 + "1)")]
        costly = [interval(f"sqrt(10^24 + {2 * k})", f"sqrt(10^24 + {2 * k + 1})") for k in range(MAX_INTERVALS)]

        def root(j):  # sqrt(10^24 + j) as an end: 10^12 for j = 0, else the positive root of x^2 - 10^24 - j.
            return mf.Rational(10 ** 12) if j == 0 else mf.RealAlgebraicRoot((-10 ** 24 - j, 0, 1), 1)

        with computation(WIDE, fresh=True):  # sqrt(10^400) is 10^200; sqrt(10^400 + 1) is a root of x^2 - 10^400 - 1.
            upper = mf.RealAlgebraicRoot((-10 ** 400 - 1, 0, 1), 1)
            one = mf.IntervalSet((mf.Interval(mf.Rational(10 ** 200), upper),))
            many = mf.IntervalSet(tuple(mf.Interval(root(2 * k), root(2 * k + 1)) for k in range(MAX_INTERVALS)))
        for label, value in (("one interval", one), ("32 intervals", many)):
            with self.subTest(label):
                mf.from_data(mf.to_data(value), computation_limits=ComputationLimits())  # The set alone fits them.
                echoed = operations_v6.run("demo_echo_interval_set", {"intervals": value})  # run() has no such check.
                self.assertIs(echoed.execution_status, ExecutionStatus.COMPLETED)
                reloaded = self.error(call_tool("mathforge.verify", {"result": mf.to_data(echoed)}))
                self.assertEqual((reloaded["stage"], reloaded["code"]), ("decoding", "resource_limit_exceeded"))
        demo.CALLS.clear()
        for label, raw in (("one interval", close), ("its envelope", mf.to_data(one)), ("32 intervals", costly),
                           ("their envelope", mf.to_data(many))):
            with self.subTest(label):
                error = self.error(self.call(ECHO_INTERVALS, raw))
                self.assertEqual((error["stage"], error["code"], error["field"]),
                                 ("decoding", "resource_limit_exceeded", "arguments.intervals"))
                self.assertTrue(error["message"].startswith(
                    "The interval set costs more than 1/10 of the default limits to rebuild, so a result that holds it "
                    "might not reload within them (Computation limit exceeded: "), error["message"])
        self.assertEqual(demo.CALLS, [])
        small = [interval(2 * k, f"{4 * k + 1}/2", k % 2 == 0, k % 3 == 0) for k in range(MAX_INTERVALS)]
        response = self.call(ECHO_INTERVALS, small)
        self.assertEqual(len(self.result(response).value), MAX_INTERVALS)
        self.assertTrue(self.result(call_tool("mathforge.verify", {"result": response.payload["result"]})).verified)
        self.assertTrue(mf.verify(mf.loads(mf.dumps(mf.from_data(response.payload["result"])))).verified)
        self.assertEqual(demo.CALLS, ["demo_echo_interval_set"])

    def test_default_applies_and_catalog_names_the_kinds(self):
        tool, operation, name = ECHO_RELATION
        response = call_tool(tool, {"operation": operation, "arguments": {}})
        self.assertEqual(self.result(response).value, "=")
        self.assertEqual(arguments_schema(registry.spec(operation))["properties"],
                         {name: {"$ref": "#/$defs/relation", "default": "="}})
        found = call_tool("mathforge.catalog", {"query": "demo_echo"}).payload["operations"]
        self.assertEqual({entry["name"]: entry["arguments"] for entry in found}, {
            "demo_echo_polynomial": [{"name": "polynomial", "kind": "polynomial", "required": True, "default": None}],
            "demo_echo_interval_set": [{"name": "intervals", "kind": "interval_set", "required": True,
                                        "default": None}],
            "demo_echo_relation": [{"name": "relation", "kind": "relation", "required": False, "default": "="}]})
        self.assertEqual({entry["name"]: entry["tool"] for entry in found}, {
            operation: tool for (tool, operation, _), _ in VALUES})

    def test_malformed_values_are_located_errors(self):
        """A malformed value is a decoding error that names its position: no exception, and no call of produce()."""
        result = mf.to_data(mf.operations.integer_gcd(6, 4))  # A Result is no operand.
        code = "invalid_input"
        malformed = {
            ECHO_POLYNOMIAL: (
                (None, "", code), (True, "", code), (1.5, "", code), ("x^2 - 1", "", code), ([], "", code),
                ({}, "", code), ({"text": "x"}, "", code), ({"text": "x", "variable": "x", "degree": 1}, "", code),
                ({"text": 5, "variable": "x"}, ".text", code), ({"text": "x", "variable": 5}, ".variable", code),
                ({"text": "x", "variable": "sqrt"}, ".variable", code),
                ({"text": "x +", "variable": "x"}, ".text", "parse_syntax"),
                ({"text": "x*y", "variable": "x"}, ".text", "unknown_symbol"),
                # No polynomial in x: a division by x, also one that cancels, and an irrational coefficient.
                ({"text": "1/x", "variable": "x"}, ".text", code),
                ({"text": "(x^2 - 1)/(x - 1)", "variable": "x"}, ".text", code),
                ({"text": "sqrt(2)*x", "variable": "x"}, ".text", "unsupported_operation"),
                ({"text": "x^1024*x", "variable": "x"}, ".text", "unsupported_operation"),
                ({"text": "x^2000", "variable": "x"}, ".text", "resource_limit_exceeded"),
                ({"text": "x" * 20000, "variable": "x"}, ".text", "resource_limit_exceeded"),
                ({"format": "mathforge"}, "", code), (result, "", code)),
            ECHO_INTERVALS: (
                (None, "", code), ({}, "", code), ("all", "", code), (5, "", code), ([None], "[0]", code),
                ([[0, 1]], "[0]", code), ([{"lower": 0, "upper": 1}], "[0]", code),
                ([dict(interval(0, 1), open=True)], "[0]", code),
                ([interval(0.5, 1)], "[0].lower", code), ([interval(True, 1)], "[0].lower", code),
                ([interval(0, [1])], "[0].upper", code), ([interval(0, {"type": "rational"})], "[0].upper", code),
                ([interval(0, "x")], "[0].upper", "unknown_symbol"),
                # The infinities of the point kind: an infinite end is null.
                ([interval(0, "inf")], "[0].upper", code), ([interval("-inf", 0)], "[0].lower", code),
                ([interval(0, " +inf")], "[0].upper", code), ([interval(0, "oo")], "[0].upper", "unknown_symbol"),
                ([interval("1/0", 1)], "[0].lower", "division_by_zero"),
                # Ends that Interval does not admit: outside every quadratic field, a symbol, no number at all.
                ([interval(0, "sqrt(2) + sqrt(3)")], "[0].upper", code),
                ([interval(0, mf.to_data(X))], "[0].upper", code),
                ([interval(0, mf.to_data((1, 2)))], "[0].upper", code),
                ([interval(0, mf.to_data("1"))], "[0].upper", code), ([interval(0, result)], "[0].upper", code),
                ([interval(0, 1, 1, True)], "[0].left_closed", code),
                ([interval(0, 1, True, None)], "[0].right_closed", code),
                ([interval(None, 1, True)], "[0].left_closed", code),  # An infinite end must be open.
                ([interval(0, None, False, True)], "[0].right_closed", code),
                ([interval(0, 1), interval(2, 1)], "[1]", code),  # Reversed ends.
                ([interval(0, 1), interval(3, "9" * 2000)], "[1].upper", "resource_limit_exceeded"),
                ({"format": "mathforge"}, "", code), (result, "", code)),
            ECHO_RELATION: tuple((raw, "", code) for raw in ("==", "!=", "", "le", " <", None, True, 1, ["="],
                                                             {"format": "mathforge"}, result)),
        }
        for target, cases in malformed.items():
            for raw, position, expected in cases:
                with self.subTest(operation=target[1], raw=str(raw)[:60]):
                    response = self.call(target, raw)
                    error = self.error(response)
                    self.assertEqual((error["stage"], error["field"], error["code"]),
                                     ("decoding", "arguments." + target[2] + position, expected), error)
                    self.assertIsNone(response.payload["result"])
        for target, raw, message in (
                (ECHO_POLYNOMIAL, {"text": "1/x", "variable": "x"}, "A polynomial divides by rational constants only"),
                (ECHO_INTERVALS, [interval(None, 1, True)], "An infinite end must be open"),
                (ECHO_INTERVALS, [interval(0, 1, 1)], "left_closed must be true or false"),
                (ECHO_INTERVALS, [interval("-inf", 0)], "An infinite end is null, not text"),
                # sqrt(6) lies in one quadratic field, but as_endpoint reads its square roots, as the solver does.
                (ECHO_INTERVALS, [interval(0, "sqrt(2)*sqrt(3)")],
                 "An interval end must be a rational number or a number of one quadratic field, written with square "
                 "roots that are rational multiples of each other, such as '(1+sqrt(5))/2'; send another real "
                 "algebraic number as a RealAlgebraicRoot envelope"),
                (ECHO_RELATION, "==", "A relation is '=', '<', '<=', '>' or '>='")):
            with self.subTest(message=message):
                self.assertEqual(self.error(self.call(target, raw))["message"], message)
        # A friendly array costs one node per interval while it decodes, as an envelope's walk does.
        error = self.error(self.call(ECHO_INTERVALS, [interval(0, 1)] * 3, limits={"max_nodes": 2}))
        self.assertEqual((error["stage"], error["field"], error["code"]),
                         ("decoding", "arguments.intervals", "resource_limit_exceeded"))
        self.assertEqual(demo.CALLS, [])

    def test_schemas_accept_the_forms_that_the_tools_accept(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("Development-only jsonschema is unavailable")
        schemas = input_schemas()
        for (tool, operation, name), value in VALUES:
            spec = registry.spec(operation)
            arguments = Draft202012Validator(arguments_schema(spec))  # Complete on its own.
            with self.subTest(operation=operation):
                Draft202012Validator.check_schema(schemas[tool])
                for encoded in (KINDS[spec.arguments[0].kind].encode(value), mf.to_data(value)):
                    arguments.validate({name: encoded})
                    Draft202012Validator(schemas[tool]).validate({"operation": operation,
                                                                  "arguments": {name: encoded}})
                for invalid in (1.5, None, [1.5], {"text": "x"}, "all"):
                    self.assertFalse(arguments.is_valid({name: invalid}), invalid)
        self.assertTrue(Draft202012Validator(arguments_schema(registry.spec("demo_echo_relation"))).is_valid({}))

    def test_area_tool_schemas_define_the_kinds_in_registry_order(self):
        schemas = input_schemas()
        algebra, equations = (schemas[tool]["$defs"] for tool in ("mathforge.algebra", "mathforge.equations"))
        self.assertTrue({"demo_echo_polynomial", "demo_echo_interval_set", "polynomial", "interval_set", "scalar",
                         "envelope"} <= set(algebra))
        self.assertTrue({"demo_echo_relation", "relation", "envelope"} <= set(equations))
        for table in (algebra, equations):
            shared = [name for name in table if name in registry.DEFINITION_NAMES]
            self.assertEqual(shared, [name for name in registry.DEFINITION_NAMES if name in table])
            self.assertEqual(list(table)[-len(shared):], shared)  # The operations come first.
            self.assertEqual(shared[-2:], ["envelope", "handle"])
            for name in shared[:-2]:
                self.assertEqual(table[name], KINDS[name].schema)

    def test_v08_area_tool_rules_hold_with_the_new_kinds(self):
        """The rules of tests/v08/test_mcp_areas.py for every area tool hold with the demo operations among them.

        They are portable schemas, each definition once, and located errors for malformed values of every argument.
        """
        # This package is tests.v09 for ``python -m unittest tests.v09...`` and v09 for discovery from tests/.
        rules = importlib.import_module(__package__.replace("v09", "v08") + ".test_mcp_areas").AreaToolTests
        for name in ("test_input_schemas_are_portable", "test_area_schemas_define_each_operation_and_kind_once",
                     "test_kinds_cover_the_registry", "test_malformed_arguments_are_located_errors",
                     "test_schemas_accept_what_the_tools_accept"):
            with self.subTest(test=name):
                outcome = unittest.TestResult()
                rules(name).run(outcome)
                self.assertEqual(outcome.testsRun, 1)
                self.assertTrue(outcome.wasSuccessful(), outcome.failures + outcome.errors)


class GeneratorTests(unittest.TestCase):
    def test_v6_specs_may_name_the_core_classes(self):
        """tools/generate_api.py resolves Polynomial, IntervalSet, Interval and FiniteSet in annotations."""
        tool = load_tool()
        sources = {"Polynomial": "mathforge.polynomial", "IntervalSet": "mathforge.intervals",
                   "Interval": "mathforge.intervals", "FiniteSet": "mathforge.contracts"}
        # Appended: the import lines of the v0.7 names, which follow this order, stay as generated. The names that
        # the M2-M4 family annotations need follow them, then the one that the M7 families add.
        later = {"Literal": "typing", "EmptySet": "mathforge.contracts", "UniversalSet": "mathforge.contracts",
                 "RationalFunction": "mathforge.rational_function"}
        self.assertEqual(list(tool.NAME_SOURCES.items())[-len(sources) - len(later):],
                         [*sources.items(), *later.items()])
        for name, module in sources.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(importlib.import_module(module), name).__module__, module)
        with registry._testing_families(demo):
            files = {path.relative_to(SRC / "mathforge").as_posix(): text
                     for path, text in tool.render_files(SRC).items()}
        structured = files["_generated/structured.py"]
        self.assertIn("def demo_echo_polynomial(polynomial: Polynomial, *, limits: ComputationLimits | None = None) "
                      "-> Result[Polynomial]:\n"
                      '    return run_v6("demo_echo_polynomial", dict(polynomial=polynomial), limits=limits)\n',
                      structured)
        self.assertIn("def demo_echo_interval_set(intervals: IntervalSet, *, limits: ComputationLimits | None = None) "
                      "-> IntervalSet:\n", files["_generated/values.py"])
        self.assertIn("def demo_echo_relation(relation: str = '=', *, limits: ComputationLimits | None = None) "
                      "-> Result[str]:\n", structured)
        for text in (structured, files["_generated/values.py"]):
            runtime, checking = set(), set()
            for node in ast.parse(text).body:
                if isinstance(node, ast.ImportFrom):
                    runtime.update(alias.name for alias in node.names)
                elif isinstance(node, ast.If):  # The one top-level if: TYPE_CHECKING.
                    checking.update(alias.name for child in node.body for alias in child.names)
            # Names that only v6 wrappers use are imported for type checking alone, as family records are.
            self.assertTrue({"Polynomial", "IntervalSet"} <= checking, checking)
            self.assertFalse({"Polynomial", "IntervalSet"} & runtime)


if __name__ == "__main__":
    unittest.main()
