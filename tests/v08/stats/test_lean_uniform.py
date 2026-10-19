"""The lean discrete_uniform: the structured distribution wherever it answers, FALLBACK elsewhere, and no value
without an accepting check."""

import importlib
import random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.lean import FALLBACK, stats_descriptive as sd, stats_distributions as sdist
from mathforge.limits import computation

from .inputs import R, corpus_calls, distinct_outcomes, exact, forged, outcome, structured

probability = importlib.import_module("mathforge.probability")  # The attribute mathforge.probability is a function.
FiniteDistribution = probability.FiniteDistribution
CAP = sd.SAMPLE_CAP


def lean_uniform(*arguments):
    return lean.functions.discrete_uniform(*arguments)


class LeanUniformCase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        lean._clear_caches()
        self.addCleanup(lean._clear_caches)

    def assertSameValue(self, got, expected, context=None):
        self.assertIs(type(got), type(expected), context)
        self.assertEqual(got, expected, context)
        self.assertEqual(mf.to_data(got), mf.to_data(expected), context)
        self.assertEqual((hash(got), repr(got)), (hash(expected), repr(expected)), context)

    def assertAnswers(self, outcomes):
        expected = structured("discrete_uniform", outcomes)
        self.assertEqual(expected[0], "value", outcomes)
        got = lean_uniform(outcomes)
        self.assertIsNot(got, FALLBACK, outcomes)
        self.assertSameValue(got, expected[1], outcomes)
        return got

    def assertAgrees(self, outcomes):
        expected = structured("discrete_uniform", outcomes)
        got = lean_uniform(outcomes)
        if expected[0] == "error":
            self.assertIs(got, FALLBACK, (outcomes, expected))
        elif got is not FALLBACK:
            self.assertSameValue(got, expected[1], outcomes)
        return got


class ResolutionAndCorpusTest(LeanUniformCase):
    def test_the_hand_written_function_is_resolved(self):
        self.assertIs(lean.functions.discrete_uniform, sdist.discrete_uniform)
        self.assertEqual(set(sdist.LEAN), {"discrete_uniform"})

    def test_fixture_variants_are_answered(self):
        calls = [call for call in corpus_calls({"discrete_uniform"}) if "/edge/" not in call[0]]
        self.assertGreaterEqual(len(calls), 3)
        for label, operation, arguments in calls:
            with self.subTest(label):
                self.assertAnswers(*arguments)

    def test_edge_cases_agree(self):
        calls = [call for call in corpus_calls({"discrete_uniform"}) if "/edge/" in call[0]]
        self.assertGreaterEqual(len(calls), 15)
        answered = 0
        for label, operation, arguments in calls:
            with self.subTest(label):
                answered += self.assertAgrees(*arguments) is not FALLBACK
        self.assertGreaterEqual(answered, 4)  # single, unsorted, mixed_int_rational and list_input.

    def test_the_public_function_gives_the_corpus_outcomes(self):
        for label, operation, arguments in corpus_calls({"discrete_uniform"}):
            with self.subTest(label):
                expected = structured("discrete_uniform", *arguments)
                got = outcome(lambda: mf.discrete_uniform(*arguments))
                self.assertEqual(got[0], expected[0])
                if expected[0] == "value":
                    self.assertSameValue(got[1], expected[1])
                else:
                    self.assertEqual(got, expected)

    def test_seeded_random_outcomes_are_answered_with_the_structured_value(self):
        rng = random.Random(20261019)
        for index in range(150):
            outcomes = distinct_outcomes(rng, rng.choice((1, 2, 3, 4, 5, 8, 13, 30, 60)))
            with self.subTest(index=index, size=len(outcomes)):
                self.assertAnswers(outcomes)
                self.assertAnswers(list(outcomes))
        wide = tuple(R(rng.randint(-2 ** 150, 2 ** 150), rng.randint(1, 2 ** 80)) for _ in range(40))
        self.assertAnswers(wide)

    def test_the_distribution_is_the_constructors(self):
        value = lean_uniform((R(5, 2), 1, -3, R(1, 3)))
        built = FiniteDistribution((R(5, 2), 1, -3, R(1, 3)), (R(1, 4),) * 4)
        self.assertSameValue(value, built)
        self.assertEqual(value.outcomes, (R(-3), R(1, 3), R(1), R(5, 2)))
        self.assertEqual(value.probabilities, (R(1, 4),) * 4)
        self.assertEqual(value, FiniteDistribution(value.outcomes, value.probabilities))
        self.assertEqual(probability._trusted_distribution(value.outcomes, value.probabilities), value)
        with self.assertRaises(Exception):
            value.outcomes = ()  # Frozen.


class EdgesTest(LeanUniformCase):
    def test_types_the_structured_path_refuses_fall_back(self):
        bad = [True, 1.5, "1", None, b"12", {1: 2}, 5, R(1), (1.5, 2), (True, 2), ("1", 2), (None,), ((1, 2),),
               ([1],), (1, {2}), [1.5], (R(1, 2), 1.5), (object(),), (), []]
        for outcomes in bad:
            with self.subTest(outcomes=repr(outcomes)):
                self.assertAgrees(outcomes)
                self.assertIs(lean_uniform(outcomes), FALLBACK)

    def test_duplicates_are_refused(self):
        for outcomes in ((1, 1, 2), (1, 2, 1), (R(1, 2), R(2, 4)), (1, R(1)), (R(3, 2), 2, R(6, 4)), (0, R(0), 5)):
            with self.subTest(outcomes=outcomes):
                self.assertEqual(structured("discrete_uniform", outcomes)[0], "error")
                self.assertIs(lean_uniform(outcomes), FALLBACK)

    def test_containers_other_than_tuples_and_lists_fall_back(self):
        for outcomes in (iter((1, 2, 3)), (value for value in (1, 2, 3)), {1, 2, 3}, frozenset((1, 2, 3)), range(1, 4),
                         {1: 2}.keys()):
            with self.subTest(outcomes=type(outcomes).__name__):
                self.assertIs(lean_uniform(outcomes), FALLBACK)
        data = iter((1, 2, 3))
        self.assertIs(lean_uniform(data), FALLBACK)
        self.assertEqual(list(data), [1, 2, 3])  # Not consumed.

    def test_forged_rationals_fall_back(self):
        class Fake(mf.Rational):
            pass

        for fake in (Fake(1, 2), forged(2, 4), forged(1, -2), forged(0, 5), forged(1, 0), forged(1.5, 1),
                     forged(True, 1), forged(3, 3)):
            with self.subTest(fake=repr(fake)):
                self.assertAgrees((1, fake, 3))
                self.assertIs(lean_uniform((1, fake, 3)), FALLBACK)

    def test_the_size_and_the_bits(self):
        outcomes = tuple(R(index, 7) for index in range(1, CAP + 1))
        self.assertEqual(len(set(outcomes)), CAP)
        self.assertAnswers(outcomes)
        over = tuple(R(index, 7) for index in range(1, CAP + 2))
        self.assertIs(lean_uniform(over), FALLBACK)
        expected = structured("discrete_uniform", over)
        self.assertEqual(expected[0], "value")  # Within the structured limits.
        self.assertSameValue(mf.discrete_uniform(over), expected[1])
        for bits, answers in ((sd.ELEMENT_BITS, True), (sd.ELEMENT_BITS + 1, False)):
            data = (R((1 << bits) - 1, 2), 3, R(1, 5))
            if answers:
                self.assertAnswers(data)
            else:
                self.assertIs(lean_uniform(data), FALLBACK)
                self.assertSameValue(mf.discrete_uniform(data), structured("discrete_uniform", data)[1])
        self.assertAgrees(((1 << 65537), 1))
        self.assertIs(lean_uniform(((1 << 65537), 1)), FALLBACK)

    def test_an_active_budget_or_explicit_limits_leave_the_call_to_the_structured_path(self):
        with computation():
            self.assertIs(lean_uniform((1, 2, 3)), FALLBACK)
        limits = mf.ComputationLimits(max_work=1)
        got = outcome(lambda: mf.discrete_uniform((1, 2, 3), limits=limits))
        self.assertEqual((got[0], got[2]), ("error", "resource_limit_exceeded"))

    def test_a_single_outcome(self):
        value = self.assertAnswers((R(7, 3),))
        self.assertEqual(value.probabilities, (R(1),))


class TamperedValueTest(LeanUniformCase):
    """A distribution that is not the uniform one on the outcomes is never returned."""

    OUTCOMES = (R(5, 2), 1, -3, R(1, 3), 8)

    def wrong_values(self):
        support = (R(-3), R(1, 3), R(1), R(5, 2), R(8))
        uniform = (R(1, 5),) * 5
        build = probability._trusted_distribution
        return {
            "skewed weights": FiniteDistribution(support, (R(1, 2), R(1, 8), R(1, 8), R(1, 8), R(1, 8))),
            "other outcome": FiniteDistribution((R(-3), R(1, 3), R(1), R(5, 2), R(9)), uniform),
            "one outcome fewer": FiniteDistribution(support[:4], (R(1, 4),) * 4),
            "one outcome more": FiniteDistribution(support + (R(11),), (R(1, 6),) * 6),
            "unsorted": build(support[::-1], uniform),
            "duplicate outcome": build((R(-3), R(-3), R(1), R(5, 2), R(8)), uniform),
            "weights not summing to one": build(support, (R(1, 5),) * 4 + (R(1, 4),)),
            "weights as lists": build(support, list(uniform)),
            "outcomes as a list": build(list(support), uniform),
            "weights of the wrong size": build(support, (R(1, 5),) * 4),
            "not a distribution": support,
            "none": None,
        }

    def test_each_wrong_distribution_is_refused(self):
        true = lean_uniform(self.OUTCOMES)
        self.assertEqual(true, FiniteDistribution(self.OUTCOMES, (R(1, 5),) * 5))
        for label, wrong in self.wrong_values().items():
            with self.subTest(wrong=label), patch.object(sdist, "_trusted_distribution", lambda a, b, _w=wrong: _w):
                self.assertIs(lean_uniform(self.OUTCOMES), FALLBACK)
                self.assertSameValue(mf.discrete_uniform(self.OUTCOMES), true)  # Through the structured path.

    def test_the_check_agrees_with_v07s(self):
        nums = [exact(item).numerator for item in self.OUTCOMES]
        dens = [exact(item).denominator for item in self.OUTCOMES]
        args = {"outcomes": self.OUTCOMES}
        true = structured("discrete_uniform", self.OUTCOMES)[1]
        self.assertTrue(sdist._check_uniform(true, nums, dens, 5))
        self.assertTrue(probability.check_operation("discrete_uniform", args, true, ()))
        for label, wrong in self.wrong_values().items():
            with self.subTest(wrong=label):
                self.assertFalse(sdist._check_uniform(wrong, nums, dens, 5))
                self.assertFalse(probability.check_operation("discrete_uniform", args, wrong, ()))

    def test_the_check_rejects_a_sample_that_is_not_the_outcomes(self):
        # The same value, checked against other outcomes: refused, as v0.7's check refuses it.
        value = structured("discrete_uniform", (1, 2, 3))[1]
        for other in ((1, 2, 4), (1, 2, 2), (1, 2), (3, 2, 1, 0), (R(1), R(2), R(7, 2))):
            nums = [exact(item).numerator for item in other]
            dens = [exact(item).denominator for item in other]
            with self.subTest(other=other):
                self.assertFalse(sdist._check_uniform(value, nums, dens, len(other)))
                self.assertFalse(probability.check_operation("discrete_uniform", {"outcomes": other}, value, ()))
        self.assertTrue(sdist._check_uniform(value, [3, 1, 2], [1, 1, 1], 3))

    def test_a_check_that_does_not_answer_true_gives_fallback(self):
        for label, answer in (("rejecting", lambda *a: False), ("truthy", lambda *a: 1), ("none", lambda *a: None)):
            with self.subTest(check=label), patch.object(sdist, "_check_uniform", answer):
                self.assertIs(lean_uniform(self.OUTCOMES), FALLBACK)
        self.assertIsNot(lean_uniform(self.OUTCOMES), FALLBACK)

    def test_the_check_does_not_use_the_producers_scaling(self):
        true = structured("discrete_uniform", self.OUTCOMES)[1]
        nums = [exact(item).numerator for item in self.OUTCOMES]
        dens = [exact(item).denominator for item in self.OUTCOMES]
        with patch.object(sd, "_scale", side_effect=AssertionError("the check scaled the sample")):
            self.assertTrue(sdist._check_uniform(true, nums, dens, 5))


if __name__ == "__main__":
    unittest.main()
