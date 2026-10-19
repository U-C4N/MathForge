"""The median's counting certificate: one pass, and the verdict of v0.7's quadratic check for every sample and value."""

import random
import unittest

import mathforge as mf
from mathforge import statistics
from mathforge.limits import ComputationLimits, computation
from mathforge.model import trusted_rational

from .inputs import R, exact, outcome, sample, structured


def v07_verdict(data, value):
    """v0.7's check of the median, as it was written: both order statistics located by counting every candidate
    against every observation, then value * 2 == their sum."""
    observations = [exact(item) for item in data]
    selected = []
    for rank in ((len(observations) - 1) // 2, len(observations) // 2):
        found = None
        for candidate in observations:
            lower = equal = 0
            for observation in observations:
                lower += observation < candidate
                equal += observation == candidate
            if lower <= rank < lower + equal:
                found = candidate
                break
        if found is None:
            return False
        selected.append(found)
    return value * R(2) == selected[0] + selected[1]


def parts(data):
    items = [exact(item) for item in data]
    return [item.numerator for item in items], [item.denominator for item in items]


def true_median(data):
    ordered = sorted(exact(item) for item in data)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / R(2)


def candidates(rng, data):
    """The true median, its neighbours, every entry and every average of two entries, and a few random values."""
    items = [exact(item) for item in data]
    true = true_median(data)
    values = {true, true + R(1, 12), true - R(1, 12), true + 1, true - 1, R(0)}
    values.update(items)
    values.update((a + b) / R(2) for a in items for b in items)
    values.update(R(rng.randint(-30, 30), rng.randint(1, 8)) for _ in range(4))
    return values


class CertificateVerdictTest(unittest.TestCase):
    def test_the_verdict_is_v07s_on_random_samples_and_values(self):
        rng = random.Random(20261019)
        compared = accepted = 0
        for index in range(400):
            size = rng.randint(1, 9)
            data = sample(rng, size, ("ints", "fractions", "mixed", "ties", "ties", "decimals")[index % 6])
            nums, dens = parts(data)
            for value in candidates(rng, data):
                expected = v07_verdict(data, value)
                got = statistics.median_certificate(nums, dens, value.numerator, value.denominator)
                self.assertIs(got, expected, (data, value))
                compared += 1
                accepted += expected
        self.assertGreater(compared, 8000)
        self.assertEqual(accepted, 400)  # Exactly the median is accepted, once per sample.

    def test_the_verdict_on_the_tie_and_even_cases(self):
        cases = (((1, 2), R(3, 2), True), ((1, 2), R(1), False), ((1, 2), R(2), False), ((1, 1, 2), R(1), True),
                 ((1, 2, 2), R(2), True), ((1, 1, 3, 3), R(2), True), ((1, 1, 3, 3), R(1), False),
                 ((1, 1, 3, 3), R(3), False), ((1, 3, 3, 3), R(3), True), ((1, 1, 1, 3), R(1), True),
                 ((1, 3), R(2), True), ((5,), R(5), True), ((5,), R(4), False), ((1, 2, 3), R(2), True),
                 ((1, 2, 3), R(3, 2), False), ((1, 2, 3, 4), R(5, 2), True), ((1, 2, 3, 4), R(2), False),
                 ((1, 2, 3, 4), R(3), False), ((R(1, 3), R(1, 2)), R(5, 12), True))
        for data, value, expected in cases:
            nums, dens = parts(data)
            with self.subTest(data=data, value=str(value)):
                self.assertIs(statistics.median_certificate(nums, dens, value.numerator, value.denominator), expected)
                self.assertIs(v07_verdict(data, value), expected)

    def test_every_entry_of_the_sample_is_a_wrong_median_unless_it_is_the_median(self):
        rng = random.Random(4)
        for index in range(200):
            data = sample(rng, rng.randint(2, 11), "ties" if index % 2 else "fractions")
            nums, dens = parts(data)
            true = true_median(data)
            for entry in {exact(item) for item in data}:
                verdict = statistics.median_certificate(nums, dens, entry.numerator, entry.denominator)
                self.assertEqual(verdict, entry == true, (data, entry))


class StructuredCheckTest(unittest.TestCase):
    def check(self, data, value, evidence=()):
        return statistics.check_operation("median", {"data": data}, value, evidence)

    def test_the_structured_check_gives_v07s_verdicts(self):
        rng = random.Random(11)
        for index in range(150):
            data = sample(rng, rng.randint(1, 8), ("ints", "fractions", "mixed", "ties")[index % 4])
            for value in candidates(rng, data):
                with self.subTest(data=data, value=str(value)):
                    self.assertIs(self.check(data, value), v07_verdict(data, value))

    def test_the_other_refusals_are_unchanged(self):
        data = (1, 3, 8)
        self.assertTrue(self.check(data, R(3)))
        for value in (3, 3.0, True, None, "3", (3,), mf.Sqrt(R(9))):  # Not a Rational.
            self.assertFalse(self.check(data, value), repr(value))
        self.assertFalse(self.check(data, R(3), ("forged",)))
        self.assertFalse(self.check(data, R(3), [()]))
        self.assertFalse(self.check(data, R(3), None))
        self.assertFalse(self.check(data, trusted_rational(6, 2)))  # Noncanonical.
        self.assertFalse(self.check(data, trusted_rational(3, 0)))
        self.assertFalse(self.check(data, trusted_rational(-3, -1)))
        for bad in ((), (1.5, 2), (True,), "12", None, 5, ((1,),)):
            self.assertFalse(self.check(bad, R(1)), repr(bad))
        self.assertFalse(statistics.check_operation("median", {"data": data, "ddof": 0}, R(3), ()))
        self.assertFalse(statistics.check_operation("median", {"x": data}, R(3), ()))
        self.assertFalse(statistics.check_operation("median", [data], R(3), ()))

    def test_the_check_never_uses_the_producer(self):
        from unittest.mock import patch
        data = (9, 1, 3, 2)
        value, evidence = statistics.produce("median", {"data": data})
        with patch.object(statistics, "median", side_effect=AssertionError("producer used")):
            self.assertTrue(statistics.check_operation("median", {"data": data}, value, evidence))
            self.assertFalse(statistics.check_operation("median", {"data": data}, R(3), evidence))

    def test_the_check_is_linear_and_a_large_sample_is_accepted(self):
        size = 3001
        data = tuple(R((index * 7919) % 6007, 11) for index in range(size))
        expected = true_median(data)
        with computation() as budget:
            self.assertTrue(self.check(data, expected))
            self.assertFalse(self.check(data, expected + R(1, 11)))
        self.assertLess(budget.work, 40 * size)  # v0.7's check took two passes over the sample for each of 2n + ...
        # The structured path no longer reaches max_work (2,000,000) at about 1,000 entries.
        found = structured("median", data)
        self.assertEqual(found[0], "value")
        self.assertEqual(found[1], expected)
        self.assertEqual(outcome(lambda: mf.median(data))[1], expected)
        # With the work limit that the quadratic check passed at 1,000 entries and failed at 1,001: the structured
        # path now answers with a smaller one as well.
        limits = ComputationLimits(max_work=200_000)
        self.assertEqual(outcome(lambda: mf.operations.median(data, limits=limits))[0], "value")


if __name__ == "__main__":
    unittest.main()
