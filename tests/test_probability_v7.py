import unittest
from importlib import import_module
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from mathforge import Rational as Q
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
p = import_module("mathforge.probability")


class FiniteProbabilityTests(unittest.TestCase):
    def checked(self, op, **args):
        value, evidence = p.produce(op, args)
        self.assertTrue(p.check_operation(op, args, value, evidence))
        self.assertFalse(p.check_operation(op, args, value, ("forged",)))
        return value

    def test_canonical_distribution(self):
        d = p.FiniteDistribution((2, 1, 2, 99), (Q(1, 4), Q(1, 2), Q(1, 4), 0))
        self.assertEqual(d.outcomes, (Q(1), Q(2)))
        self.assertEqual(d.probabilities, (Q(1, 2), Q(1, 2)))
        with self.assertRaises(FrozenInstanceError):
            d.outcomes = ()
        for outcomes, weights in (((), ()), ((1,), (Q(1, 2),)), ((1, 2), (-1, 2)), ((True,), (1,))):
            with self.assertRaises((InvalidInput, TypeError)):
                p.FiniteDistribution(outcomes, weights)

    def test_named_distributions(self):
        self.assertEqual(self.checked("bernoulli", p=Q(1, 3)).probabilities, (Q(2, 3), Q(1, 3)))
        d = self.checked("binomial_distribution", n=3, p=Q(1, 2))
        self.assertEqual(d.probabilities, (Q(1, 8), Q(3, 8), Q(3, 8), Q(1, 8)))
        self.assertEqual(self.checked("binomial_distribution", n=0, p=1).outcomes, (Q(0),))
        self.assertEqual(self.checked("binomial_distribution", n=5, p=1).outcomes, (Q(5),))
        self.assertEqual(self.checked("discrete_uniform", outcomes=(3, 1, 2)).probabilities, (Q(1, 3),)*3)
        d = self.checked("hypergeometric", population=10, successes=4, draws=3)
        self.assertEqual(d.probabilities, (Q(1, 6), Q(1, 2), Q(3, 10), Q(1, 30)))
        self.assertEqual(self.checked("hypergeometric", population=0, successes=0, draws=0).outcomes, (Q(0),))

    def test_events_and_moments(self):
        d = p.FiniteDistribution((-1, 0, 2), (Q(1, 4), Q(1, 4), Q(1, 2)))
        self.assertEqual(self.checked("probability", distribution=d, event=(-1, -1, 99)), Q(1, 4))
        self.assertEqual(self.checked("probability", distribution=d, event=()), Q(0))
        self.assertEqual(self.checked("cdf", distribution=d, x=0), Q(1, 2))
        self.assertEqual(self.checked("moment", distribution=d, order=0), Q(1))
        self.assertEqual(self.checked("expectation", distribution=d), Q(3, 4))
        self.assertEqual(self.checked("distribution_variance", distribution=d), Q(27, 16))
        conditioned = self.checked("condition", distribution=d, event=(0, 2))
        self.assertEqual(conditioned.probabilities, (Q(1, 3), Q(2, 3)))
        with self.assertRaises(InvalidInput):
            p.condition(d, (99,))

    def test_independent_checks_and_corruption(self):
        args = {"n": 6, "p": Q(2, 5)}
        value, evidence = p.produce("binomial_distribution", args)
        with patch.object(p, "binomial_distribution", side_effect=AssertionError("producer used")):
            self.assertTrue(p.check_operation("binomial_distribution", args, value, evidence))
            self.assertFalse(p.check_operation("binomial_distribution", args, p.FiniteDistribution((0,), (1,)), evidence))
        source = p.FiniteDistribution((0, 1), (Q(1, 2), Q(1, 2)))
        for op in ("expectation", "distribution_variance"):
            value, evidence = p.produce(op, {"distribution": source})
            with patch.object(p, op, side_effect=AssertionError("producer used")):
                self.assertTrue(p.check_operation(op, {"distribution": source}, value, evidence))
                self.assertFalse(p.check_operation(op, {"distribution": source}, value+1, evidence))
        forged = object.__new__(p.FiniteDistribution)
        object.__setattr__(forged, "outcomes", (Q(0), Q(0)))
        object.__setattr__(forged, "probabilities", (Q(1, 2), Q(1, 2)))
        self.assertFalse(p.check_operation("bernoulli", {"p": 0}, forged, ()))

    def test_invalid_inputs_and_budgets(self):
        for call in (lambda: p.bernoulli(Q(3, 2)), lambda: p.binomial_distribution(True, Q(1, 2)),
                     lambda: p.discrete_uniform((1, 1)), lambda: p.hypergeometric(3, 4, 1)):
            with self.assertRaises(InvalidInput):
                call()
        with computation(ComputationLimits(max_nodes=10)):
            with self.assertRaises(ResourceLimitError):
                p.binomial_distribution(100, Q(1, 2))
        with computation(ComputationLimits(max_work=20)):
            with self.assertRaises(ResourceLimitError):
                p.hypergeometric(10000, 5000, 5000)


if __name__ == "__main__":
    unittest.main()
