"""Persistence tests use structural identities and intentionally corrupted data."""

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest

from mathforge import contracts as c
from mathforge import model as m
from mathforge.errors import InvalidInput, SerializationError
from mathforge.polynomial import Polynomial
from mathforge.serialization import dumps, from_data, loads, to_data
from mathforge.workspace import ObjectRef, Workspace


class CodecTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", assumptions=(m.Assumption("nonzero"),))

    def roundtrip(self, value):
        self.assertEqual(loads(dumps(value)), value)
        self.assertEqual(from_data(to_data(value)), value)
        self.assertEqual(dumps(loads(dumps(value))), dumps(value))

    def test_exact_numbers_and_large_integers(self):
        # Beyond both IEEE 754 precision and Python's default decimal limit.
        big = 10 ** 5000 + 37
        for number in (0, 1, -1, 2**100 + 1, big, -big):
            with self.subTest(sign=number < 0, bits=number.bit_length()):
                self.roundtrip(number)
                self.roundtrip(m.Rational(number, 11))
                self.assertIsInstance(to_data(number)["data"]["value"], str)
        self.assertEqual(to_data(m.Rational(1, 10))["data"]["denominator"], "10")

    def test_every_ast_node_domain_and_identity(self):
        values = (
            self.x, m.Rational(1, 3), m.Add((self.x, m.Rational(2))),
            m.Mul((m.Rational(-2), self.x)), m.Pow(self.x, 3), m.Sqrt(m.Rational(2)),
            Polynomial((m.Rational(2), m.Rational(-3), m.Rational(1)), self.x),
            c.Eq(self.x**2, 2), m.Reals, m.Rationals, m.Integers,
            m.Assumption("positive", m.Truth.FALSE), m.Truth.UNKNOWN,
        )
        for value in values:
            with self.subTest(type=type(value).__name__):
                self.roundtrip(value)
        restored = loads(dumps(c.Eq(self.x + 1, self.x)))
        self.assertEqual(restored.rhs.uid, self.x.uid)
        self.assertNotEqual(restored.rhs, m.Symbol("x", assumptions=self.x.assumptions))
        self.assertEqual(restored.rhs.assumptions[0].truth, m.Truth.UNKNOWN)

    def test_all_result_axes_and_evidence_are_preserved(self):
        check = c.Check("root_membership", c.CheckStatus.VERIFIED,
                        c.VerificationLevel.EXACT_SUBSTITUTION, "substitution", m.Rational(1), "1 - 1 = 0")
        condition = c.Condition(m.Rational(1), "ne", m.Rational(0), m.Truth.TRUE)
        step = c.Step("linear", (c.Eq(self.x - 1, 0),), (c.FiniteSet((m.Rational(1),)),),
                      (condition,), c.StepRelation.EQUIVALENCE, (check,), "Divide by 1")
        result = c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.SOLUTIONS,
                          c.FiniteSet((m.Rational(1),)), c.Exactness.EXACT, c.Completeness.COMPLETE,
                          (check,), "linear", m.Reals, self.x.assumptions, (step,),
                          c.Eq(self.x - 1, 0), self.x - 1, self.x)
        self.roundtrip(result)
        for execution in c.ExecutionStatus:
            if execution is not c.ExecutionStatus.COMPLETED:
                with self.subTest(execution=execution):
                    self.roundtrip(c.Result(execution, c.Outcome.NOT_APPLICABLE,
                                            problem=result.problem, for_=self.x,
                                            error=c.ErrorInfo(execution.value, "Operation unavailable")))
        outcomes = (
            (c.Outcome.VALUE, m.Rational(1), c.Exactness.EXACT, c.Completeness.COMPLETE),
            (c.Outcome.SOLUTIONS, c.UniversalSet(), c.Exactness.EXACT, c.Completeness.COMPLETE),
            (c.Outcome.CANDIDATES, result.value, c.Exactness.EXACT, c.Completeness.UNKNOWN),
            (c.Outcome.NO_SOLUTION, c.EmptySet(), c.Exactness.EXACT, c.Completeness.COMPLETE),
            (c.Outcome.NO_CONCLUSION, None, c.Exactness.NOT_APPLICABLE, c.Completeness.UNKNOWN),
            (c.Outcome.PARTIAL, result.value, c.Exactness.EXACT, c.Completeness.PARTIAL),
            (c.Outcome.NOT_APPLICABLE, None, c.Exactness.NOT_APPLICABLE, c.Completeness.NOT_APPLICABLE),
        )
        for outcome, value, exactness, completeness in outcomes:
            with self.subTest(outcome=outcome):
                self.roundtrip(replace(result, outcome=outcome, value=value,
                                       exactness=exactness, completeness=completeness))
        for exactness in (c.Exactness.EXACT, c.Exactness.APPROXIMATE, c.Exactness.MIXED):
            for completeness in (c.Completeness.COMPLETE, c.Completeness.PARTIAL, c.Completeness.UNKNOWN):
                self.roundtrip(replace(result, exactness=exactness, completeness=completeness))
        for status in c.CheckStatus:
            for level in c.VerificationLevel:
                self.roundtrip(replace(check, status=status, level=level))
        for relation in c.StepRelation:
            self.roundtrip(replace(step, relation=relation))
        self.roundtrip(c.VerificationReport((check,)))
        self.roundtrip(replace(result, precision=60, tolerance=m.Rational(1, 10**60),
                               error_bound=m.Rational(1, 10**70), error=c.ErrorInfo("example", "details")))

    def test_contradictory_result_wire_claims_rejected(self):
        base = to_data(c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.SOLUTIONS,
                               c.FiniteSet((m.Rational(1),)), c.Exactness.EXACT,
                               c.Completeness.COMPLETE))
        mutations = (
            {"execution_status": "unsupported", "outcome": "no_solution",
             "value": to_data(c.EmptySet())["data"]},
            {"outcome": "no_solution"},
            {"outcome": "solutions", "value": to_data(c.EmptySet())["data"]},
            {"outcome": "solutions", "value": to_data(c.FiniteSet(()))["data"]},
            {"outcome": "solutions", "value": to_data(m.Rational(1))["data"]},
            {"outcome": "candidates", "value": to_data(c.FiniteSet(()))["data"]},
            {"outcome": "no_conclusion"},
        )
        for changes in mutations:
            data = copy.deepcopy(base)
            data["data"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(SerializationError):
                from_data(data)
        unavailable = to_data(c.Result(c.ExecutionStatus.UNSUPPORTED, c.Outcome.NOT_APPLICABLE))
        for changes in ({"value": to_data(c.EmptySet())["data"]},
                        {"exactness": "exact"}, {"completeness": "complete"}):
            data = copy.deepcopy(unavailable)
            data["data"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(SerializationError):
                from_data(data)

    def test_mutable_expression_subclasses_are_not_transport_types(self):
        class MutableExpression(m.Expression):
            def __init__(self):
                self.contents = []

        value = MutableExpression()
        with self.assertRaises(SerializationError):
            dumps(value)
        with self.assertRaises(SerializationError):
            Workspace().put(value)

    def test_solution_set_variants(self):
        for value in (c.FiniteSet((-m.sqrt(2), m.sqrt(2))), c.EmptySet(), c.UniversalSet()):
            self.roundtrip(value)

    def test_metadata_primitives_are_not_coerced_to_numbers(self):
        for value in (None, True, False, "explanation", (True, 3, "x", None)):
            self.roundtrip(value)
        with self.assertRaises(SerializationError):
            from_data({"format": "mathforge", "version": 1, "kind": "object",
                       "data": {"type": "integer", "value": True}})

    def test_malformed_envelopes_and_tags_rejected(self):
        baseline = to_data(self.x)
        mutations = (
            lambda data: data.update(version=999),
            lambda data: data.update(version=True),
            lambda data: data.update(format="pickle"),
            lambda data: data.update(kind="workspace"),
            lambda data: data.update(extra=True),
            lambda data: data["data"].update(type="os.system"),
            lambda data: data["data"].update(extra=True),
            lambda data: data["data"].update(uid="not-a-uuid"),
            lambda data: data["data"].update(uid=self.x.uid.upper()),
            lambda data: data["data"].update(domain="complex"),
        )
        for mutate in mutations:
            data = copy.deepcopy(baseline)
            mutate(data)
            with self.assertRaises(SerializationError):
                from_data(data)

    def test_invalid_exact_wire_numbers_rejected(self):
        for number in ("01", "-0", "+1", "1.0", "1e3", "", 1, 0.1, True):
            with self.subTest(number=number), self.assertRaises(SerializationError):
                from_data({"format": "mathforge", "version": 1, "kind": "object",
                           "data": {"type": "rational", "numerator": number, "denominator": "1"}})
        for n, d in (("2", "4"), ("0", "2"), ("1", "-1"), ("1", "0")):
            with self.assertRaises(SerializationError):
                from_data({"format": "mathforge", "version": 1, "kind": "object",
                           "data": {"type": "rational", "numerator": n, "denominator": d}})

    def test_duplicates_nonfinite_and_executable_text_rejected(self):
        text = dumps(m.Rational(2))
        with self.assertRaises(SerializationError):
            loads(text.replace('"version":1', '"version":1,"version":1'))
        with self.assertRaises(SerializationError):
            loads(text.replace('"numerator":"2"', '"numerator":"2","numerator":"3"'))
        for value in ("NaN", "Infinity", "-Infinity", "__import__('os').system('bad')"):
            with self.assertRaises(SerializationError):
                loads(value)

    def test_invalid_ast_and_collections_rejected(self):
        for value, field, replacement in (
            (m.Pow(self.x, 2), "exponent", "-1"),
            (m.Sqrt(m.Rational(2)), "radicand", to_data(m.Rational(-1))["data"]),
            (m.Add((self.x, m.Rational(1))), "terms", []),
            (self.x, "assumptions", [to_data(m.Assumption("nonzero"))["data"]] * 2),
        ):
            data = to_data(value)
            data["data"][field] = replacement
            with self.assertRaises(SerializationError):
                from_data(data)
        for value in (0.1, {}, [], object()):
            with self.assertRaises(SerializationError):
                dumps(value)

    def test_conflicting_symbol_metadata_rejected_in_both_directions(self):
        other = replace(self.x, name="other")
        with self.assertRaises(SerializationError):
            dumps(c.Eq(self.x, other))
        data = to_data(c.Eq(self.x, self.x))
        data["data"]["rhs"]["name"] = "other"
        with self.assertRaises(SerializationError):
            from_data(data)

    def test_reloaded_problem_retains_exact_evaluation(self):
        polynomial = Polynomial((m.Rational(1, 3), m.Rational(-7, 2), m.Rational(4)), self.x)
        restored = loads(dumps(polynomial))
        for point in (-11, -1, 0, 3, m.Rational(2, 7)):
            self.assertEqual(restored.evaluate(point), polynomial.evaluate(point))


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x")
        self.workspace = Workspace()
        self.expression = self.x**2 - 2
        self.expression_ref = self.workspace.put(self.expression)
        self.problem = c.Eq(self.expression, 0)
        self.problem_ref = self.workspace.put(self.problem)
        self.result = c.Result(c.ExecutionStatus.COMPLETED, c.Outcome.SOLUTIONS,
                               c.FiniteSet((-m.sqrt(2), m.sqrt(2))),
                               c.Exactness.EXACT, c.Completeness.COMPLETE,
                               method="quadratic", domain=m.Reals, problem=self.problem, for_=self.x)
        self.result_ref = self.workspace.put(self.result)

    def test_content_identity_and_immutable_snapshots(self):
        self.assertEqual(self.workspace.put(self.expression), self.expression_ref)
        self.assertNotEqual(self.workspace.put(self.expression + 1), self.expression_ref)
        self.assertNotEqual(self.workspace.put(m.Symbol("x")**2 - 2), self.expression_ref)
        self.assertNotEqual(self.workspace.put((self.x + 1)**2),
                            self.workspace.put(self.x**2 + 2*self.x + 1))
        self.assertEqual(self.workspace.get(self.expression_ref), self.expression)
        with self.assertRaises(FrozenInstanceError):
            self.workspace.get(self.result_ref).method = "changed"
        with self.assertRaises(TypeError):
            self.workspace.objects[self.expression_ref] = m.Rational(0)
        with self.assertRaises(SerializationError):
            self.workspace.put([self.expression])
        with self.assertRaises(InvalidInput):
            self.workspace.put(Workspace())
        with self.assertRaises(SerializationError):
            self.workspace.put((Workspace(),))

    def test_operation_history_deep_freeze_and_roundtrip(self):
        options = {"nested": {"domains": ["reals", "rationals"], "digits": 10**100}, "trace": True}
        record = self.workspace.record("solve", {"problem": self.problem_ref}, options, self.result_ref)
        options["nested"]["domains"].append("integers")
        self.assertEqual(record.options["nested"]["domains"], ("reals", "rationals"))
        with self.assertRaises(TypeError):
            record.options["nested"]["digits"] = 0
        restored = loads(dumps(self.workspace))
        self.assertEqual(restored.history, self.workspace.history)
        self.assertEqual(dict(restored.objects), dict(self.workspace.objects))
        self.assertEqual(dumps(restored), dumps(self.workspace))

    def test_save_load_files_and_atomic_replacement(self):
        self.workspace.record("solve", {"problem": self.problem_ref}, {}, self.result_ref)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.json"
            self.workspace.save(path)
            restored = Workspace.load(path)
            self.assertEqual(restored.get(self.result_ref), self.result)
            self.assertEqual(restored.history, self.workspace.history)
            self.workspace.put(m.Rational(1, 3))
            self.workspace.save(path)
            self.assertEqual(dumps(Workspace.load(path)), dumps(self.workspace))
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["workspace.json"])
            path.write_text(dumps(self.x), encoding="utf-8")
            with self.assertRaises(SerializationError):
                Workspace.load(path)

    def test_actual_solver_evidence_rechecks_after_loading(self):
        from mathforge.operations import solve
        from mathforge.verification import verify

        result = solve(self.problem, for_=self.x)
        self.assertEqual(result.exactness, c.Exactness.EXACT)
        self.assertTrue(verify(result).verified)
        reference = self.workspace.put(result)
        restored = loads(dumps(self.workspace))
        self.assertTrue(verify(restored.get(reference)).verified)
        self.assertEqual(restored.get(reference).solution_set, result.solution_set)

    def test_tampered_content_or_refs_are_rejected(self):
        self.workspace.record("solve", {"problem": self.problem_ref}, {}, self.result_ref)
        original = to_data(self.workspace)
        data = copy.deepcopy(original)
        data["data"]["objects"][0]["ref"] = "sha256:" + "0" * 64
        with self.assertRaises(SerializationError):
            from_data(data)
        data = copy.deepcopy(original)
        data["data"]["objects"].append(copy.deepcopy(data["data"]["objects"][0]))
        with self.assertRaises(SerializationError):
            from_data(data)
        data = copy.deepcopy(original)
        data["data"]["history"][0]["result_ref"]["identifier"] = "sha256:" + "0" * 64
        with self.assertRaises(SerializationError):
            from_data(data)
        data = copy.deepcopy(original)
        data["data"]["history"][0]["inputs"].append(data["data"]["history"][0]["inputs"][0])
        with self.assertRaises(SerializationError):
            from_data(data)

    def test_conflicting_symbol_ids_across_objects_rejected(self):
        before = dumps(self.workspace)
        with self.assertRaises(SerializationError):
            self.workspace.put(replace(self.x, domain=m.Integers))
        self.assertEqual(dumps(self.workspace), before)

    def test_missing_references_and_invalid_options_do_not_append(self):
        missing = ObjectRef("sha256:" + "0" * 64)
        with self.assertRaises(SerializationError):
            self.workspace.record("solve", {"problem": missing}, {}, self.result_ref)
        for bad in (float("nan"), float("inf"), self.x, object()):
            with self.assertRaises(InvalidInput):
                self.workspace.record("solve", {}, {"bad": bad}, self.result_ref)
        self.assertEqual(self.workspace.history, ())
        with self.assertRaises(KeyError):
            self.workspace.get(missing)

    def test_mutable_reference_and_text_subclasses_do_not_enter_history(self):
        class CustomRef(ObjectRef):
            pass

        class CustomText(str):
            pass

        reference = CustomRef(self.result_ref.identifier)
        with self.assertRaises(InvalidInput):
            ObjectRef(CustomText(self.result_ref.identifier))
        with self.assertRaises(InvalidInput):
            self.workspace.get(reference)
        calls = (
            ("solve", {"problem": reference}, {}, self.result_ref),
            ("solve", {"problem": self.problem_ref}, {}, reference),
            (CustomText("solve"), {}, {}, self.result_ref),
            ("solve", {CustomText("problem"): self.problem_ref}, {}, self.result_ref),
            ("solve", {}, {CustomText("option"): True}, self.result_ref),
            ("solve", {}, {"nested": {CustomText("option"): True}}, self.result_ref),
        )
        for args in calls:
            with self.subTest(args=args), self.assertRaises(InvalidInput):
                self.workspace.record(*args)
        self.assertEqual(self.workspace.history, ())


if __name__ == "__main__":
    unittest.main()
