"""Public workflow and contract tests, independent of implementation layout."""

from dataclasses import FrozenInstanceError, replace
import unittest

import mathforge as mf


class PublicWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.x = mf.symbol("x", domain=mf.Reals)

    def test_sdk_and_structured_operations_agree(self):
        expression = self.x**3 - 3 * self.x + 7
        structured = mf.operations.differentiate(expression, self.x)
        self.assertEqual(structured.execution_status, mf.ExecutionStatus.COMPLETED)
        self.assertEqual(mf.diff(expression, self.x), structured.value)
        self.assertEqual(mf.evaluate(structured.value, {self.x: 3}), mf.Rational(24))

    def test_convenience_error_preserves_common_result(self):
        with self.assertRaises(mf.OperationError) as raised:
            mf.evaluate(self.x + 2)
        result = raised.exception.result
        self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.solution_set)

    def test_substitute_and_evaluate_have_distinct_binding_requirements(self):
        y = mf.symbol("y")
        expression = self.x + 2 * y
        partial = mf.substitute(expression, {self.x: 3})
        self.assertEqual(partial, 3 + 2 * y)
        with self.assertRaises(mf.OperationError):
            mf.evaluate(expression, {self.x: 3})
        self.assertEqual(mf.evaluate(partial, {y: 4}), mf.Rational(11))

    def test_structural_equality_is_not_equivalence(self):
        expanded = self.x**2 + 2 * self.x + 1
        factored = (self.x + 1)**2
        self.assertNotEqual(expanded, factored)
        self.assertEqual(mf.Polynomial.from_expression(expanded, self.x),
                         mf.Polynomial.from_expression(factored, self.x))

    def test_full_exact_workflow_and_reverification(self):
        expression = self.x**2 - 2
        result = mf.solve(mf.Eq(expression, 0), for_=self.x)
        self.assertEqual(set(result.solution_set.values), {-mf.sqrt(2), mf.sqrt(2)})
        self.assertTrue(mf.verify(result).verified)
        restored = mf.loads(mf.dumps(result))
        self.assertEqual(restored, result)
        self.assertTrue(mf.verify(restored).verified)
        self.assertEqual(restored.for_.uid, self.x.uid)
        self.assertEqual(mf.evaluate(mf.diff(expression, self.x), {self.x: 5}), mf.Rational(10))

    def test_restrictions_are_not_silently_discarded(self):
        positive = mf.symbol("p", assumptions=(mf.Assumption("positive", mf.Truth.TRUE),))
        result = mf.solve(mf.Eq(positive**2 - 1, 0), for_=positive)
        self.assertEqual(result.execution_status, mf.ExecutionStatus.UNSUPPORTED)
        self.assertIsNone(result.solution_set)
        self.assertEqual(result.assumptions, positive.assumptions)

    def test_exact_and_approximate_inputs_are_not_conflated(self):
        self.assertEqual(mf.evaluate(self.x, {self.x: mf.Rational(1, 10)}), mf.Rational(1, 10))
        result = mf.operations.evaluate(self.x, {self.x: 0.1})
        self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)
        with self.assertRaises(TypeError):
            mf.Rational(0.1)

    def test_unsupported_python_subclasses_return_invalid_input(self):
        class ForeignEquation(mf.Eq):
            pass

        class ForeignSymbol(mf.Symbol):
            pass

        class ForeignExpression(mf.Expression):
            pass

        calls = (
            lambda: mf.solve(ForeignEquation(self.x, 0), for_=self.x),
            lambda: mf.solve(mf.Eq(self.x, 0), for_=ForeignSymbol("z")),
            lambda: mf.operations.differentiate(self.x, ForeignSymbol("z")),
            lambda: mf.operations.simplify(ForeignExpression()),
        )
        for call in calls:
            result = call()
            self.assertEqual(result.execution_status, mf.ExecutionStatus.INVALID_INPUT)
            self.assertEqual(mf.loads(mf.dumps(result)), result)


class ContractTests(unittest.TestCase):
    def test_unknown_mutable_subclasses_are_not_mathematical_values(self):
        class MutableExpression(mf.Expression):
            def __init__(self):
                self.items = []

        value = MutableExpression()
        for kwargs in ({"value": value}, {"expression": value}):
            with self.assertRaises(TypeError):
                mf.Result(mf.ExecutionStatus.COMPLETED, mf.Outcome.VALUE, **kwargs)
        with self.assertRaises(TypeError):
            mf.Step("example", (value,), ())

    def test_uncompleted_results_cannot_claim_empty_solutions(self):
        with self.assertRaises(ValueError):
            mf.Result(mf.ExecutionStatus.UNSUPPORTED, mf.Outcome.NO_SOLUTION,
                      value=mf.EmptySet())
        plain_set = mf.Result(mf.ExecutionStatus.COMPLETED, mf.Outcome.VALUE,
                              value=mf.EmptySet())
        self.assertIsNone(plain_set.solution_set)

    def test_nested_collections_are_frozen(self):
        check = mf.Check("membership", mf.CheckStatus.VERIFIED,
                         mf.VerificationLevel.EXACT_SUBSTITUTION, "substitution")
        step = mf.Step("example", [mf.Rational(1)], [mf.Rational(1)], audit=[check])
        result = mf.Result(mf.ExecutionStatus.COMPLETED, mf.Outcome.VALUE,
                           value=mf.Rational(1), steps=[step], verification=[check])
        self.assertIsInstance(result.steps, tuple)
        self.assertIsInstance(result.steps[0].inputs, tuple)
        self.assertIsInstance(result.verification, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.value = mf.Rational(2)
        with self.assertRaises(TypeError):
            mf.Result(mf.ExecutionStatus.COMPLETED, mf.Outcome.VALUE, value={"mutable": []})

    def test_no_checks_is_not_verified(self):
        self.assertFalse(mf.VerificationReport(()).verified)
        unchecked = mf.Check("claim", mf.CheckStatus.UNCHECKED,
                             mf.VerificationLevel.NONE, "none")
        self.assertFalse(mf.VerificationReport((unchecked,)).verified)

    def test_solution_set_does_not_exist_on_unsupported_result(self):
        x = mf.symbol("x")
        unsupported = mf.solve(mf.Eq(x**3 - 2, 0), for_=x)
        empty = mf.solve(mf.Eq(x**2 + 1, 0), for_=x)
        self.assertIsNone(unsupported.solution_set)
        self.assertIsInstance(empty.solution_set, mf.EmptySet)
        self.assertNotEqual(unsupported.outcome, empty.outcome)

    def test_claim_labels_do_not_establish_verification(self):
        x = mf.symbol("x")
        correct = mf.solve(mf.Eq(x**2 - 2, 0), for_=x)
        forged = replace(correct, value=mf.FiniteSet((mf.Rational(100),)))
        self.assertEqual(forged.verification, correct.verification)
        self.assertFalse(mf.verify(forged).verified)


if __name__ == "__main__":
    unittest.main()
