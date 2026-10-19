"""The lean value path: a value function runs a producer and an independent checker directly, and leaves every
input it does not handle to the structured path, so that values and errors stay those of the structured path."""

from dataclasses import replace
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import types
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import lean, operations, registry
from mathforge.errors import OperationError
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Add

from .lean_inputs import (LEGACY_NAMES, SMALL_INPUTS, argument_names, operations as all_operations,
                          public_function, structured_function, symbol)

# The tested package's root: src/ in a checkout, site-packages in the distribution harness (no src/).
SRC = Path(mf.__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[2]
CORPUS = REPOSITORY / "tests" / "fixtures" / "v8" / "values_v07"
TYPE_EDGES = (True, 1.5, "1", None)
OVER_LIMIT = 2**65536  # 65,537 bits: one more than the default max_integer_bits.
THREADS = 8
# Operations whose lean path may answer FALLBACK even for their small input: parse_function has no producer table
# entry and keeps its own implementation.
NO_LEAN_PRODUCER = frozenset({"parse_function"})
# Inputs that the public functions route away from the legacy table's producer: function inputs go to the registry
# operations, algebraic bounds and points to the v4 exact path, and an unsupported v2 comparison to the v4 one.
_x, _y = symbol("x", 1), symbol("y", 2)
_root = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
_sine = mf.real_function(mf.functions.sin(_x), (_x,))
_wave = mf.real_function(mf.functions.sin(_x) * mf.functions.cos(_y), (_x, _y))
ROUTED_INPUTS = {
    "differentiate": dict(expression=_sine, variable=_x, n=2),
    "partial_diff": dict(expression=_wave, variable=_y, n=1),
    "gradient": dict(expression=_wave, variables=[_x, _y]),
    "jacobian": dict(expressions=[_wave, _sine], variables=(_x, _y)),
    "hessian": dict(expression=_wave, variables=(_x, _y)),
    "evaluate": dict(expression=mf.functions.as_real_expr(mf.Rational(3)) + 1, bindings=None),
    "antiderivative": dict(expression=mf.operations.rational_function(1, _x, variable=_x).unwrap(), variable=_x),
    "integrate": dict(expression=_x, variable=_x, lower=0, upper=mf.sqrt(2)),
    "limit": dict(expression=_x, variable=_x, point=_root, side="both"),
    "compare_real": dict(left=mf.sqrt(2) + mf.sqrt(3), right=3),
}
ROUTED_FUNCTION_INPUTS = {
    "antiderivative": dict(expression=_sine, variable=_x),
    "integrate": dict(expression=_sine, variable=_x, lower=0, upper=1),
    "limit": dict(expression=mf.real_function(mf.functions.sin(_x) / mf.functions.as_real_expr(_x), (_x,)),
                  variable=_x, point=0, side="both"),
    "evaluate": dict(expression=_sine, bindings={_x: 0}),
}


def outcome(call):
    """("value", value) or ("error", exception class, code); a failed Result counts as its unwrap's error."""
    try:
        value = call()
        if isinstance(value, mf.Result):
            value = value.unwrap()
        return ("value", value)
    except Exception as error:
        return ("error", type(error), getattr(error, "code", None))


def data(value):
    try:
        return mf.to_data(value)
    except Exception as error:
        return ("not serializable", type(error).__name__)


def first_refused(call, hint):
    """The smallest size at which call(size), a structured call, fails, when every smaller size succeeds.

    hint - 1 and hint are tried first; otherwise the size is found by doubling from the hint (sizes grow when the
    arithmetic gets cheaper) or by bisection below it."""
    def refused(size):
        return outcome(lambda: call(size))[0] == "error"

    if refused(hint):
        if not refused(hint - 1):
            return hint
        low, high = 1, hint - 1
        if refused(low):
            raise AssertionError("the smallest size fails too")
    else:
        low, high = hint, 2 * hint
        while not refused(high):
            if high > 16 * hint:
                raise AssertionError(f"no failing size up to {high}")
            low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if refused(middle):
            high = middle
        else:
            low = middle
    return high


def positional(operation, arguments):
    return [arguments[name] for name in argument_names(operation)]


def value_operations():
    """The operations that have a public value function, verify excepted."""
    return [name for name in all_operations() if public_function(name) is not None]


def corpus_cases():
    """(operation, arguments) of each v0.7 value corpus entry whose area file exists: fixture variants 0-2 and the
    edge cases, rebuilt from benchmarks/speed/fixtures/<area>.py."""
    cases = []
    if not CORPUS.is_dir():
        return cases
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    for path in sorted(CORPUS.glob("*.json")):
        module = importlib.import_module("benchmarks.speed.fixtures." + path.stem)
        calls = []
        for fixture in getattr(module, "FIXTURES", ()):
            if fixture.operation != "verify" and callable(getattr(fixture, "mathforge", None)):
                calls.extend((fixture.operation, lambda fixture=fixture, i=i: fixture.mathforge(i))
                             for i in range(min(3, fixture.variants)))
        calls.extend((edge.operation, edge.mathforge) for edge in getattr(module, "EDGE_CASES", ())
                     if edge.operation != "verify")
        for operation, build in calls:
            function, args, kwargs = build()
            try:
                bound = inspect.signature(function).bind(*args, **kwargs)
            except TypeError:
                continue  # An edge case that calls with a wrong arity never reaches an operation.
            bound.apply_defaults()
            arguments = {name: value for name, value in bound.arguments.items() if name != "limits"}
            if set(arguments) == set(argument_names(operation)):
                cases.append((operation, arguments))
    return cases


def probe(code, *arguments):
    """Run code in a fresh isolated interpreter that imports the package under test; return its JSON output."""
    program = "import sys\nsys.path.insert(0, sys.argv[1])\n" + textwrap.dedent(code)
    output = subprocess.run([sys.executable, "-I", "-c", program, str(SRC), *arguments],
                            capture_output=True, text=True, timeout=300)
    if output.returncode:
        raise AssertionError(output.stderr)
    return json.loads(output.stdout)


# Threads released together by a barrier make the first calls of a cold interpreter: registry tables, lean modules
# and lean functions resolve under contention. Each thread calls every operation, in its own rotation.
CONCURRENT = """
    import json, threading
    sys.path.insert(0, sys.argv[2])
    import mathforge
    from mathforge import lean, registry
    from tests.v08.lean_inputs import SMALL_INPUTS, public_function
    names, count = json.loads(sys.argv[3]), int(sys.argv[4])
    cold = (registry._tables.cache_info().currsize == 0 and not vars(lean.functions)
            and not any(name.startswith("mathforge.lean.") for name in sys.modules))
    def call(name):
        try:
            return ("value", public_function(name)(**SMALL_INPUTS[name]))
        except Exception as error:
            return ("error", type(error).__name__, getattr(error, "code", None))
    barrier, results, errors = threading.Barrier(count), {}, []
    def run(index):
        try:
            barrier.wait(60)
            order = names[index % len(names):] + names[:index % len(names)]
            results[index] = {name: call(name) for name in order}
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(240)
    sequential = {name: call(name) for name in names}
    def same(a, b):
        if a[0] != b[0]:
            return False
        if a[0] == "error":
            return a == b
        if type(a[1]) is not type(b[1]) or a[1] != b[1]:
            return False
        try:
            return mathforge.to_data(a[1]) == mathforge.to_data(b[1])
        except Exception:
            return True  # Plain builtins: equal values of one type.
    mismatches = sorted({name for result in results.values() for name in names
                         if name not in result or not same(result[name], sequential[name])})
    print(json.dumps({"cold": cold, "errors": errors, "stuck": sum(thread.is_alive() for thread in threads),
                      "finished": len(results), "mismatches": mismatches,
                      "failed": sorted(name for name in names if sequential[name][0] != "value")}))
    """


class LeanPathTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        lean._clear_caches()
        self.addCleanup(lean._clear_caches)

    def assertSameValue(self, got, expected, context=None):
        self.assertIs(type(got), type(expected), context)
        self.assertEqual(got, expected, context)
        self.assertEqual(data(got), data(expected), context)

    def assertSameOutcome(self, got, expected, context=None):
        self.assertEqual(got[0], expected[0], (context, got, expected))
        if got[0] == "value":
            self.assertSameValue(got[1], expected[1], context)
        else:
            self.assertEqual(got, expected, context)

    def assertLeanAgrees(self, value, expected, context):
        """A lean answer is FALLBACK, or the structured value; it is FALLBACK whenever the structured path fails."""
        if expected[0] == "error":
            self.assertIs(value, lean.FALLBACK, (context, expected))
        elif value is not lean.FALLBACK:
            self.assertSameValue(value, expected[1], context)

    def test_import_loads_no_lean_module(self):
        loaded = probe("""
            import json
            import mathforge
            print(json.dumps(sorted(name for name in sys.modules if name.startswith("mathforge.lean."))))
            """)
        self.assertEqual(loaded, [])

    def test_legacy_lean_paths_leave_the_registry_unbuilt(self):
        """As in v0.7, the legacy operations never build the registry tables, on the lean path either."""
        report = probe("""
            import json
            import mathforge as m
            from mathforge import lean, registry
            x = m.Symbol("x")
            values = [m.det(m.Matrix(((1, 2), (3, 4)))), m.rational_function(1, x, variable=x), m.expand((x + 1)**2, x),
                      m.diff(x**3, x), m.compare_real(m.sqrt(2), 1), m.factor(m.Polynomial((-2, 0, 1), x)),
                      m.limit(x, x, 2)]
            print(json.dumps({"tables": registry._tables.cache_info().currsize,
                              "lean": sorted(vars(lean.functions))}))
            """)
        self.assertEqual(report["tables"], 0)
        self.assertEqual(report["lean"], ["compare_real", "det", "differentiate", "expand", "factor", "limit",
                                          "rational_function"])

    def test_small_inputs_cover_every_operation(self):
        self.assertEqual(set(SMALL_INPUTS), set(all_operations()))
        for operation, arguments in SMALL_INPUTS.items():
            with self.subTest(operation=operation):
                self.assertEqual(set(arguments), set(argument_names(operation)))
                self.assertEqual(outcome(lambda: structured_function(operation)(**arguments))[0], "value")

    def test_fallback_and_namespace(self):
        self.assertEqual(repr(lean.FALLBACK), "FALLBACK")
        self.assertIs(lean.functions.integer_gcd, lean.functions.integer_gcd)  # Resolved once, then cached.
        self.assertIn("integer_gcd", vars(lean.functions))
        for name in ("no_such_operation", "__wrapped__", "verify_and_more"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                getattr(lean.functions, name)
        self.assertIs(lean.run_value("no_such_operation", {}), lean.FALLBACK)
        self.assertIs(lean.run_value("integer_gcd", {"a": 1}), lean.FALLBACK)  # A missing argument.
        self.assertIs(lean.run_value("integer_gcd", None), lean.FALLBACK)
        self.assertIs(lean.functions.integer_gcd(84), lean.FALLBACK)  # A wrong arity.

    def test_legacy_table(self):
        from mathforge import exact_dispatch
        self.assertEqual(set(operations.LEGACY), set(LEGACY_NAMES) - {"verify", "parse_function"})
        self.assertEqual(set(operations.LEGACY_EXACT), set(exact_dispatch.ARGUMENTS))
        for name, entry in (*operations.LEGACY.items(), *operations.LEGACY_EXACT.items()):
            with self.subTest(operation=name):
                self.assertIsInstance(entry, operations.LegacyOperation)
                self.assertTrue(callable(entry.produce) and callable(entry.check))
        for name in exact_dispatch.NEW_OPERATIONS:
            self.assertIs(operations.LEGACY[name], operations.LEGACY_EXACT[name])

    def test_every_value_function_routes_through_lean(self):
        resolve = lean._resolve
        recorded = []

        def recording(name):
            recorded.append(name)
            return resolve(name)

        with patch.object(lean, "_resolve", side_effect=recording):
            for operation in value_operations():
                with self.subTest(operation=operation):
                    lean._clear_caches()
                    recorded.clear()
                    arguments = SMALL_INPUTS[operation]
                    got = outcome(lambda: public_function(operation)(**arguments))
                    self.assertIn(operation, recorded)
                    self.assertSameOutcome(got, outcome(lambda: structured_function(operation)(**arguments)),
                                           operation)

    def test_generic_runner_equals_structured(self):
        for operation in all_operations():
            arguments = SMALL_INPUTS[operation]
            expected = outcome(lambda: structured_function(operation)(**arguments))
            with self.subTest(operation=operation):
                self.assertEqual(expected[0], "value")
                value = lean.run_value(operation, dict(arguments))
                self.assertLeanAgrees(value, expected, operation)
                if operation not in NO_LEAN_PRODUCER:  # The runner answers, rather than always falling back.
                    self.assertIsNot(value, lean.FALLBACK, operation)
                function_value = getattr(lean.functions, operation)(*positional(operation, arguments))
                self.assertLeanAgrees(function_value, expected, operation)
                if operation not in NO_LEAN_PRODUCER:
                    self.assertIsNot(function_value, lean.FALLBACK, operation)
        for operation, arguments in corpus_cases():
            expected = outcome(lambda: structured_function(operation)(**arguments))
            with self.subTest(operation=operation, corpus=True):
                self.assertLeanAgrees(lean.run_value(operation, dict(arguments)), expected, operation)
                self.assertLeanAgrees(getattr(lean.functions, operation)(*positional(operation, arguments)),
                                      expected, operation)

    def test_dispatchers_follow_the_public_routes(self):
        for operation, arguments in (*ROUTED_INPUTS.items(), *ROUTED_FUNCTION_INPUTS.items()):
            expected = outcome(lambda: structured_function(operation)(**arguments))
            with self.subTest(operation=operation, arguments=arguments):
                self.assertEqual(expected[0], "value")
                value = getattr(lean.functions, operation)(*positional(operation, arguments))
                self.assertIsNot(value, lean.FALLBACK)
                self.assertSameValue(value, expected[1], operation)
                self.assertSameOutcome(outcome(lambda: public_function(operation)(**arguments)), expected, operation)
                # The generic runner takes no route: it answers only for the inputs that reach the legacy producer.
                self.assertIs(lean.run_value(operation, dict(arguments)), lean.FALLBACK)

    def test_type_edges_fall_back(self):
        for operation in all_operations():
            names = argument_names(operation)
            for index, name in enumerate(names):
                for edge in TYPE_EDGES:
                    arguments = {**SMALL_INPUTS[operation], name: edge}
                    expected = outcome(lambda: structured_function(operation)(**arguments))
                    with self.subTest(operation=operation, argument=name, edge=edge):
                        value = getattr(lean.functions, operation)(*positional(operation, arguments))
                        self.assertLeanAgrees(value, expected, (operation, name, edge))
                        self.assertLeanAgrees(lean.run_value(operation, arguments), expected, (operation, name, edge))
                        public = public_function(operation)
                        if public is not None:
                            self.assertSameOutcome(outcome(lambda: public(**arguments)), expected,
                                                   (operation, name, edge))

    def test_containers_the_record_refuses_fall_back(self):
        """Producers iterate any container, but a record holds only tuples: ranges and sets fail as in v0.7."""
        cases = 0
        for operation in all_operations():
            for name, value in SMALL_INPUTS[operation].items():
                if type(value) is not tuple:
                    continue
                for container in (range(len(value)), set(value), frozenset(value), list(value)):
                    arguments = {**SMALL_INPUTS[operation], name: container}
                    expected = outcome(lambda: structured_function(operation)(**arguments))
                    cases += expected[0] == "error"
                    with self.subTest(operation=operation, argument=name, container=type(container).__name__):
                        self.assertLeanAgrees(lean.run_value(operation, arguments), expected, operation)
                        self.assertLeanAgrees(getattr(lean.functions, operation)(*positional(operation, arguments)),
                                              expected, operation)
                        public = public_function(operation)
                        if public is not None:
                            self.assertSameOutcome(outcome(lambda: public(**arguments)), expected, operation)
        self.assertGreater(cases, 30)

    def test_values_need_an_accepting_check(self):
        """Verification is never lazy: a value comes back only when the independent check answers True."""
        from mathforge import number_theory
        from mathforge.certify.ntheory import continued_fraction as checker
        det = operations.LEGACY["det"]
        broken = {"wrong value": operations.LegacyOperation(lambda a: (mf.Rational(999), ()), det.check),
                  "truthy check": operations.LegacyOperation(det.produce, lambda *a: 1),
                  "rejecting check": operations.LegacyOperation(det.produce, lambda *a: False)}
        for label, entry in broken.items():
            with self.subTest(label), patch.dict(operations.LEGACY, {"det": entry}):
                self.assertIs(lean.run_value("det", dict(SMALL_INPUTS["det"])), lean.FALLBACK)
                self.assertIs(lean.functions.det(SMALL_INPUTS["det"]["A"]), lean.FALLBACK)
        produce = number_theory.produce

        def off_by_one(operation, args):
            value, evidence = produce(operation, args)
            return value + 1, evidence

        for label, patcher in (("wrong value", patch.object(number_theory, "produce", off_by_one)),
                               ("truthy check", patch.object(number_theory, "check_operation", lambda *a: 1))):
            with self.subTest(label), patcher:
                self.assertIs(lean.run_value("integer_gcd", {"a": 84, "b": 30}), lean.FALLBACK)
                self.assertIs(lean.functions.integer_gcd(84, 30), lean.FALLBACK)
        value = mf.Rational(415, 93)
        with self.subTest("rejecting v6 check"), patch.object(checker, "check_operation", lambda *a: False):
            self.assertIs(lean.run_value("continued_fraction", {"value": value}), lean.FALLBACK)
            self.assertIs(lean.functions.continued_fraction(value), lean.FALLBACK)
        self.assertEqual(lean.run_value("integer_gcd", {"a": 84, "b": 30}), 6)
        self.assertIsNot(lean.run_value("continued_fraction", {"value": value}), lean.FALLBACK)

    def test_over_limit_inputs_fall_back(self):
        with computation(ComputationLimits(max_matrix_entries=65 * 65)):
            large = mf.Matrix(tuple(tuple((i + 2 * j) % 7 for j in range(65)) for i in range(65)))
        cases = []
        for operation in all_operations():
            for name, value in SMALL_INPUTS[operation].items():
                if type(value) is int:
                    cases.append((operation, {**SMALL_INPUTS[operation], name: OVER_LIMIT}))
                elif type(value) is mf.Matrix:
                    replaced = {**SMALL_INPUTS[operation], name: large}
                    if operation == "least_squares":
                        replaced["b"] = tuple(range(65))
                    cases.append((operation, replaced))
        self.assertGreater(len(cases), 60)
        for operation, arguments in cases:
            expected = outcome(lambda: structured_function(operation)(**arguments))
            with self.subTest(operation=operation, arguments=sorted(arguments)):
                self.assertEqual(expected, ("error", OperationError, "resource_limit_exceeded"))
                self.assertIs(getattr(lean.functions, operation)(*positional(operation, arguments)), lean.FALLBACK)
                self.assertIs(lean.run_value(operation, arguments), lean.FALLBACK)
                public = public_function(operation)
                if public is not None:
                    self.assertSameOutcome(outcome(lambda: public(**arguments)), expected, operation)

    def test_record_limits_fall_back(self):
        """Inputs whose own inspection fits the limits while the record built around them does not."""
        def nested(depth, leaf):
            expression = leaf
            for _ in range(depth):
                expression = Add((expression, mf.Rational(1)))
            return expression

        calls = {"simplify": lambda e: dict(expression=e), "evaluate": lambda e: dict(expression=e, bindings={_x: 1}),
                 "expand": lambda e: dict(expression=e, variable=_x),
                 "complex_algebraic": lambda e: dict(real=e, imag=1), "algebraic": lambda e: dict(expression=e),
                 "compare_real": lambda e: dict(left=e, right=1)}
        failures = 0
        for operation, build in calls.items():
            leaf = _x if operation in ("simplify", "evaluate", "expand") else mf.sqrt(2)
            for depth in range(57, 63):  # The record's depth limit falls at 60 or 61 levels of nesting.
                arguments = build(nested(depth, leaf))
                expected = outcome(lambda: structured_function(operation)(**arguments))
                failures += expected[0] == "error"
                with self.subTest(operation=operation, depth=depth):
                    self.assertLeanAgrees(getattr(lean.functions, operation)(*positional(operation, arguments)),
                                          expected, (operation, depth))
                    self.assertSameOutcome(outcome(lambda: public_function(operation)(**arguments)), expected,
                                           (operation, depth))
        self.assertGreater(failures, 0)
        # Where a budget runs out, the lean path answers exactly as long as the structured path does: just below the
        # first refused size it gives the structured value, at that size FALLBACK. The sizes are located on the
        # structured path, because they move when the arithmetic gets cheaper; the hints are the last ones measured.
        # The hand-written median answers only for samples within stats_descriptive.SAMPLE_CAP: below the first
        # refused size it gives the structured value or FALLBACK, never another value (the refused size lies far above
        # the cap, now that the structured median check is one counting pass).
        sum_of = lambda terms: Add(tuple(mf.Rational(i, 7) * _x for i in range(1, terms)))  # noqa: E731
        sample = lambda count: tuple(mf.Rational(i, 7) for i in range(1, count + 1))  # noqa: E731
        cases = (("simplify", sum_of, lambda e: dict(expression=e), 2500, True),  # The verifier reaches max_evidence_items.
                 ("median", sample, lambda d: dict(data=d), 16659, False))  # So does the verifier, for a sample.
        for operation, build, arguments_of, hint, exact in cases:
            refused = first_refused(lambda size: structured_function(operation)(**arguments_of(build(size))), hint)
            for size, fits in ((refused - 1, True), (refused, False)):
                arguments = arguments_of(build(size))
                expected = outcome(lambda: structured_function(operation)(**arguments))
                with self.subTest(operation=operation, size=size):
                    self.assertEqual(expected[0], "value" if fits else "error")
                    value = getattr(lean.functions, operation)(*positional(operation, arguments))
                    if fits and (exact or value is not lean.FALLBACK):
                        self.assertSameValue(value, expected[1])
                    elif not fits:
                        self.assertIs(value, lean.FALLBACK)

    def test_record_checks_fall_back(self):
        """Inputs that the producer accepts and the verifier's record checks reject: one Symbol uid under two names,
        and variables with a domain or assumptions."""
        uid = "00000000-0000-4000-8000-0000000000aa"
        a, b = mf.Symbol("a", uid=uid), mf.Symbol("b", uid=uid)
        positive = mf.Symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))
        integer = mf.Symbol("k", mf.Integers)
        cases = [("simplify", dict(expression=Add((a, b)))), ("substitute", dict(expression=a + 1, bindings={b: 2})),
                 ("evaluate", dict(expression=Add((a, b)), bindings={})),
                 ("expand", dict(expression=a + b, variable=a)),
                 ("function_evaluate", dict(expression=mf.functions.as_real_expr(a) + mf.functions.as_real_expr(b),
                                            bindings={a: 1}))]
        for variable in (positive, integer):
            for operation in ("expand", "differentiate", "partial_diff", "antiderivative", "integrate", "real_roots",
                              "rational_function", "limit", "analyze_function", "extrema_values",
                              "minimal_polynomial", "charpoly", "function_differentiate", "function_antiderivative"):
                cases.append((operation, {**SMALL_INPUTS[operation], "variable": variable}))
            for operation in ("gradient", "hessian"):
                cases.append((operation, {**SMALL_INPUTS[operation], "variables": (variable,)}))
        for operation, arguments in cases:
            expected = outcome(lambda: structured_function(operation)(**arguments))
            with self.subTest(operation=operation, arguments=arguments):
                self.assertEqual(expected[0], "error")
                self.assertIs(lean.run_value(operation, arguments), lean.FALLBACK)
                self.assertIs(getattr(lean.functions, operation)(*positional(operation, arguments)), lean.FALLBACK)
                self.assertSameOutcome(outcome(lambda: public_function(operation)(**arguments)), expected, operation)

    def test_record_normalization_matches_contracts(self):
        """The runner normalizes a request, a value and evidence as contracts._immutable_value does for records."""
        from mathforge.contracts import _immutable_value
        lean._prepare()
        samples = [value for arguments in SMALL_INPUTS.values() for value in arguments.values()]
        for operation in all_operations():
            if operation != "parse_function":
                result = structured_function(operation)(**SMALL_INPUTS[operation])
                samples.extend((result.value, result.steps[0].outputs[1]))
        samples += [[1, [2, (3, mf.Rational(1, 2))]], (), [], (None, True, "s"), {1: 2}, 1.5, range(3), {1, 2},
                    [_x, (mf.sqrt(2), [1.5])]]
        for sample in samples:
            expected = outcome(lambda: _immutable_value(sample))
            got = outcome(lambda: lean._immutable(sample))
            with self.subTest(sample=sample):
                self.assertEqual(got[0], expected[0])
                if got[0] == "value":
                    self.assertIs(type(got[1]), type(expected[1]))
                    self.assertEqual(got[1], expected[1])
                else:
                    self.assertEqual(got, expected)

    def test_record_sizes_match_the_structured_records(self):
        """The runner charges what the structured path's records inspect: their sizes are those of real records."""
        limits = ComputationLimits(max_nodes=10**9, max_evidence_items=10**9)

        def count(value):
            with computation(limits, fresh=True) as budget:
                budget.inspect(value, evidence=True)
                return budget.evidence_count

        lean._prepare()
        for operation in all_operations():
            if operation == "parse_function":
                continue
            result = structured_function(operation)(**SMALL_INPUTS[operation])
            with self.subTest(operation=operation):
                self.assertEqual(len(result.steps), 1)
                step = replace(result.steps[0], audit=())
                arguments = dict(result.request.arguments)
                request, value, evidence = count(result.request), count(result.value), count(step.outputs[1])
                self.assertEqual(lean._request_size(count(arguments), len(arguments)), request)
                self.assertEqual(lean._step_inputs_size(request, value, evidence),
                                 count(((result.request,), step.outputs, ())))
                self.assertEqual(lean._step_size(request, value, evidence), count(step))
                sizes = {key: count({key: item}) - 2 for key, item in arguments.items()}
                context = (count(result.problem) + count(result.expression) + count(result.for_)
                           + sum(count(assumption) for assumption in result.assumptions))
                if operation in LEGACY_NAMES:
                    problem = count(result.problem) if operation == "solve_system" else sizes.get("problem", 1)
                    self.assertEqual(lean._legacy_context_size(operation, sizes,
                                                               operations._context(operation, arguments), problem),
                                     context)
                else:
                    self.assertEqual(lean._registry_context_size(sizes, arguments.get("variable")), context)
                self.assertEqual(lean._result_size(request, value, evidence, context),
                                 count(replace(result, verification=(), steps=(step,))))

    def test_explicit_limits_skip_lean(self):
        limits = ComputationLimits(max_work=1)
        with patch.object(lean, "_resolve", side_effect=AssertionError("a lean function was resolved")) as resolve:
            for operation in value_operations():
                arguments = SMALL_INPUTS[operation]
                with self.subTest(operation=operation):
                    expected = outcome(lambda: structured_function(operation)(**arguments, limits=limits))
                    got = outcome(lambda: public_function(operation)(**arguments, limits=limits))
                    self.assertSameOutcome(got, expected, operation)
        resolve.assert_not_called()

    def test_active_budget_skips_lean(self):
        """Inside an enclosing computation, the structured path charges that budget, as in v0.7."""
        for operation in ("integer_gcd", "det", "differentiate", "continued_fraction", "mean"):
            arguments = SMALL_INPUTS[operation]
            # A first call imports modules whose top-level constants would charge an enclosing budget.
            outcome(lambda: structured_function(operation)(**arguments))
            outcome(lambda: public_function(operation)(**arguments))
            counters = []
            for call in (lambda: structured_function(operation)(**arguments),
                         lambda: public_function(operation)(**arguments)):
                with computation(ComputationLimits(max_work=1_000_000)) as budget:
                    got = outcome(call)
                counters.append((got, (budget.work, budget.node_count, budget.evidence_count)))
            with self.subTest(operation=operation):
                self.assertSameOutcome(counters[1][0], counters[0][0], operation)
                self.assertEqual(counters[1][1], counters[0][1])
            with computation():
                self.assertIs(lean.run_value(operation, dict(arguments)), lean.FALLBACK)
        # The enclosing budget's limits apply: one unit of work does not compute a determinant.
        with computation(ComputationLimits(max_work=1)):
            expected = outcome(lambda: mf.operations.det(SMALL_INPUTS["det"]["A"]))
        with computation(ComputationLimits(max_work=1)):
            self.assertSameOutcome(outcome(lambda: mf.det(SMALL_INPUTS["det"]["A"])), expected)
        self.assertEqual(expected, ("error", mf.OperationError, "resource_limit_exceeded"))

    def test_one_shot_iterables_are_not_consumed(self):
        cases = [("gradient", lambda: dict(expression=_x**2 * _y, variables=iter((_x, _y)))),
                 ("gradient", lambda: dict(expression=_wave, variables=iter((_x, _y)))),
                 ("jacobian", lambda: dict(expressions=iter((_x * _y,)), variables=iter((_x, _y)))),
                 ("expand_multivariate", lambda: dict(expression=(_x + _y)**2, variables=iter((_x, _y)))),
                 ("solve_system", lambda: dict(system_or_equations=iter((mf.Eq(_x + _y, 3), mf.Eq(_x - _y, 1))),
                                               for_=iter((_x, _y)))),
                 ("mean", lambda: dict(data=iter((1, 2, 4)))),
                 ("function_gradient", lambda: dict(expression=_wave, variables=iter((_x, _y))))]
        for operation, build in cases:
            with self.subTest(operation=operation):
                expected = outcome(lambda: structured_function(operation)(**build()))
                public = public_function(operation)
                if public is not None:
                    self.assertSameOutcome(outcome(lambda: public(**build())), expected, operation)
                arguments = build()
                self.assertIs(getattr(lean.functions, operation)(*positional(operation, arguments)), lean.FALLBACK)
                self.assertSameOutcome(outcome(lambda: structured_function(operation)(**arguments)), expected,
                                       operation)  # The lean attempt left the iterators untouched.

    def test_duplicate_lean_definitions_fail(self):
        first, second = (types.ModuleType("mathforge.lean.discrete_test_" + label) for label in ("one", "two"))
        for module in (first, second):
            module.LEAN = {"factorial": lambda n: lean.FALLBACK}
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__, None)
        with patch.object(lean, "_area_modules", return_value=(first.__name__, second.__name__)):
            lean._clear_caches()
            with self.assertRaises(RuntimeError) as raised:
                lean.functions.factorial
            self.assertIn(first.__name__, str(raised.exception))
            self.assertIn(second.__name__, str(raised.exception))
            self.assertIn("factorial", str(raised.exception))
            self.assertNotIn("factorial", vars(lean.functions))
        lean._clear_caches()
        self.assertEqual(mf.factorial(12), 479001600)

    def test_lean_modules_define_operations_of_their_area(self):
        from mathforge.registry_v07 import LEGACY_CATALOG
        areas = {entry.name: entry.area for entry in LEGACY_CATALOG}
        areas.update({name: spec.area for name, spec in registry.specs().items()})
        modules = [name for area in registry.AREAS for name in lean._area_modules(area)]
        self.assertTrue(modules)
        for name in modules:
            module = importlib.import_module(name)
            area = name.rsplit(".", 1)[1].split("_", 1)[0]
            with self.subTest(module=name):
                self.assertIn(area, registry.AREAS)
                self.assertTrue(module.LEAN)
                for operation, function in module.LEAN.items():
                    self.assertEqual(areas[operation], area, operation)
                    self.assertTrue(callable(function))

    def test_concurrent_first_resolution(self):
        names = ["integer_gcd", "binomial", "mean", "complex_add", "continued_fraction", "trace",
                 "function_differentiate", "rational_antiderivative", "det", "inverse", "expand", "poly_gcd",
                 "differentiate", "integrate", "limit", "evaluate", "compare_real", "factor", "simplify",
                 "real_roots"]
        self.assertEqual(len(names), 20)
        report = probe(CONCURRENT, str(REPOSITORY), json.dumps(names), str(THREADS))
        self.assertTrue(report["cold"], "the registry or a lean module was loaded before the threads started")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stuck"], 0)
        self.assertEqual(report["finished"], THREADS)
        self.assertEqual(report["failed"], [])
        self.assertEqual(report["mismatches"], [])


if __name__ == "__main__":
    unittest.main()
