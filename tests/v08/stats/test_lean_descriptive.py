"""The lean mean, median, variance, stddev and covariance: equal to the structured path wherever they answer, FALLBACK
wherever the structured path fails or the input lies outside what they handle, and never a value without an accepting
check."""

import random
import unittest
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.lean import FALLBACK, stats_descriptive as sd
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Sqrt, trusted_rational

from .inputs import DESCRIPTIVE, R, corpus_calls, forged, outcome, random_cases, sample, structured

PUBLIC = {name: getattr(mf, name) for name in DESCRIPTIVE}


def lean_of(operation):
    return getattr(lean.functions, operation)


class LeanStatisticsCase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        lean._clear_caches()
        self.addCleanup(lean._clear_caches)

    def assertSameValue(self, got, expected, context=None):
        self.assertIs(type(got), type(expected), context)
        self.assertEqual(got, expected, context)
        self.assertEqual(mf.to_data(got), mf.to_data(expected), context)

    def assertAnswers(self, operation, *arguments):
        """The lean function answers and gives the structured value."""
        expected = structured(operation, *arguments)
        self.assertEqual(expected[0], "value", (operation, arguments, expected))
        got = lean_of(operation)(*arguments)
        self.assertIsNot(got, FALLBACK, (operation, arguments))
        self.assertSameValue(got, expected[1], (operation, arguments))
        return got

    def assertFallsBack(self, operation, *arguments):
        self.assertIs(lean_of(operation)(*arguments), FALLBACK, (operation, arguments))

    def assertAgrees(self, operation, *arguments):
        """FALLBACK whenever the structured path fails; the structured value whenever the lean path answers."""
        expected = structured(operation, *arguments)
        got = lean_of(operation)(*arguments)
        if expected[0] == "error":
            self.assertIs(got, FALLBACK, (operation, arguments, expected))
        elif got is not FALLBACK:
            self.assertSameValue(got, expected[1], (operation, arguments))
        return got


class ResolutionTest(LeanStatisticsCase):
    def test_the_hand_written_functions_are_the_ones_the_value_functions_call(self):
        for operation in DESCRIPTIVE:
            self.assertIs(lean_of(operation), sd.LEAN[operation], operation)
        from mathforge.lean import stats_distributions
        self.assertIs(lean_of("discrete_uniform"), stats_distributions.LEAN["discrete_uniform"])
        # The other operations of the area keep the generic runner.
        for operation in ("bernoulli", "cdf", "moment", "expectation", "probability"):
            self.assertEqual(lean_of(operation).__module__, "mathforge.lean", operation)

    def test_the_public_functions_use_them(self):
        calls = []
        for operation in DESCRIPTIVE:
            function = sd.LEAN[operation]

            def recording(*arguments, _function=function, _operation=operation):
                calls.append(_operation)
                return _function(*arguments)
            with patch.dict(sd.LEAN, {operation: recording}):
                lean._clear_caches()
                arguments = {"covariance": ((1, 2, 4), (2, 3, 7))}.get(operation, ((1, 2, 4),))
                value = PUBLIC[operation](*arguments)
                self.assertIn(operation, calls)
                self.assertSameValue(value, structured(operation, *arguments)[1])


class CorpusInputsTest(LeanStatisticsCase):
    def test_fixture_variants_are_answered_with_the_structured_value(self):
        calls = [call for call in corpus_calls(set(DESCRIPTIVE)) if "/edge/" not in call[0]]
        self.assertGreaterEqual(len(calls), 20)
        for label, operation, arguments in calls:
            with self.subTest(label):
                self.assertAnswers(operation, *arguments)

    def test_edge_cases_agree(self):
        calls = [call for call in corpus_calls(set(DESCRIPTIVE)) if "/edge/" in call[0]]
        self.assertGreaterEqual(len(calls), 90)
        answered = succeeded = 0
        for label, operation, arguments in calls:
            with self.subTest(label):
                succeeded += structured(operation, *arguments)[0] == "value"
                answered += self.assertAgrees(operation, *arguments) is not FALLBACK
        self.assertGreater(succeeded, 25)
        self.assertEqual(answered, succeeded)  # Every edge case that succeeds is small: the lean path takes it.

    def test_the_public_functions_give_the_corpus_outcomes(self):
        """Value or error, whichever path answers: the public function equals the structured function."""
        for label, operation, arguments in corpus_calls(set(DESCRIPTIVE)):
            with self.subTest(label):
                expected = structured(operation, *arguments)
                got = outcome(lambda: PUBLIC[operation](*arguments))
                self.assertEqual(got[0], expected[0], label)
                if expected[0] == "value":
                    self.assertSameValue(got[1], expected[1], label)
                else:
                    self.assertEqual(got, expected, label)


class RandomSamplesTest(LeanStatisticsCase):
    def test_seeded_random_samples_are_answered_with_the_structured_value(self):
        cases = 0
        for seed in (20261019, 7, 1234567):
            for size, flavor, data, other, ddof in random_cases(seed, 90):
                with self.subTest(seed=seed, size=size, flavor=flavor):
                    for operation in ("mean", "median"):
                        self.assertAnswers(operation, data)
                    for operation in ("variance", "stddev"):
                        self.assertAnswers(operation, data, ddof)
                        self.assertAnswers(operation, list(data))  # A list is a sample as well.
                    self.assertAnswers("covariance", data, other, ddof)
                    cases += 1
        self.assertEqual(cases, 270)

    def test_lists_and_tuples_give_the_same_value(self):
        data = (R(1, 3), 4, R(-5, 7), 4, 0)
        for operation in ("mean", "median"):
            self.assertSameValue(lean_of(operation)(list(data)), lean_of(operation)(data))
        self.assertSameValue(lean_of("covariance")(list(data), list(reversed(data)), 1),
                             lean_of("covariance")(data, tuple(reversed(data)), 1))

    def test_stddev_gives_a_rational_for_perfect_squares_and_a_root_otherwise(self):
        perfect = self.assertAnswers("stddev", (R(1, 2), R(3, 2)), 0)
        self.assertEqual(perfect, R(1, 2))
        self.assertIs(type(perfect), mf.Rational)
        irrational = self.assertAnswers("stddev", (1, 2, 4), 0)
        self.assertIs(type(irrational), Sqrt)
        self.assertEqual(irrational, Sqrt(R(14, 9)))
        self.assertEqual(self.assertAnswers("stddev", (3, 3, 3), 0), R(0))

    def test_ties_and_the_two_middle_values_of_the_median(self):
        for data, expected in (((5, 1, 4, 2, 3), R(3)), ((4, 1, 3, 2), R(5, 2)), ((2, 2, 2, 9, 9), R(2)),
                               ((3, 3, 3, 3), R(3)), ((1, 1, 2, 2), R(3, 2)), ((R(1, 2), R(1, 3), R(1, 5), R(1, 7)),
                                                                            R(4, 15)), ((7,), R(7))):
            self.assertEqual(self.assertAnswers("median", data), expected, data)


class EdgesTest(LeanStatisticsCase):
    def operations_with(self, data, *rest):
        """Every descriptive operation with a first sample `data` (and a valid second one for covariance)."""
        return [("mean", (data,)), ("median", (data,)), ("variance", (data, *rest)), ("stddev", (data, *rest)),
                ("covariance", (data, data, *rest))]

    def test_types_the_structured_path_refuses_fall_back(self):
        bad = [True, 1.5, "1", None, b"12", {1: 2}, 5, R(1), (1.5, 2), (True, 2), ("1", 2), (None,), ((1, 2),), ([1],),
               (1, {2}), [1.5], (R(1, 2), 1.5), (object(),)]
        for data in bad:
            for operation, arguments in self.operations_with(data, 0)[:4]:
                with self.subTest(operation=operation, data=repr(data)):
                    self.assertAgrees(operation, *arguments)
                    self.assertFallsBack(operation, *arguments)
        for data in bad:
            with self.subTest(operation="covariance", data=repr(data)):
                self.assertAgrees("covariance", data, (1, 2, 3), 0)
                self.assertFallsBack("covariance", data, (1, 2, 3), 0)
                self.assertFallsBack("covariance", (1, 2, 3), data, 0)

    def test_containers_other_than_tuples_and_lists_fall_back(self):
        containers = (iter((1, 2, 4)), (value for value in (1, 2, 4)), {1, 2, 4}, frozenset((1, 2, 4)), range(1, 5),
                      {1: 2}.keys(), {1: 2}.values())
        for data in containers:
            for operation, arguments in self.operations_with(data, 0):
                if operation == "covariance":
                    continue
                with self.subTest(operation=operation, data=type(data).__name__):
                    self.assertFallsBack(operation, *arguments)

    def test_a_one_shot_iterator_is_not_consumed(self):
        for operation in ("mean", "median", "variance", "stddev"):
            data = iter((1, 2, 4))
            self.assertFallsBack(operation, data)
            self.assertEqual(list(data), [1, 2, 4], operation)
        first, second = iter((1, 2, 4)), iter((2, 3, 7))
        self.assertFallsBack("covariance", first, second)
        self.assertEqual((list(first), list(second)), ([1, 2, 4], [2, 3, 7]))

    def test_subclasses_and_forged_rationals_fall_back(self):
        class Fake(mf.Rational):
            pass

        fakes = [Fake(1, 2), forged(2, 4), forged(1, -2), forged(0, 5), forged(1, 0), forged(1.5, 1), forged(True, 1),
                 forged(1, 2.0), forged(3, 3)]
        for fake in fakes:
            for operation, arguments in self.operations_with((1, fake, 3), 0):
                with self.subTest(operation=operation, fake=repr(fake)):
                    self.assertAgrees(operation, *arguments)
                    self.assertFallsBack(operation, *arguments)

    def test_empty_samples_and_bad_ddof_fall_back(self):
        for operation, arguments in self.operations_with((), 0):
            with self.subTest(operation=operation):
                self.assertFallsBack(operation, *arguments)
        self.assertFallsBack("covariance", (), (1,), 0)
        self.assertFallsBack("covariance", (1,), (), 0)
        self.assertFallsBack("covariance", (1, 2, 4), (2, 3), 0)  # A length mismatch.
        for ddof in (-1, 3, 4, 2 ** 65536, True, False, 1.5, "1", None, R(1), (1,)):
            for operation, arguments in (("variance", ((1, 2, 4), ddof)), ("stddev", ((1, 2, 4), ddof)),
                                         ("covariance", ((1, 2, 4), (2, 3, 7), ddof))):
                with self.subTest(operation=operation, ddof="huge" if type(ddof) is int and ddof > 10 ** 6 else repr(ddof)):
                    self.assertAgrees(operation, *arguments)
                    self.assertFallsBack(operation, *arguments)
        for ddof in (0, 1, 2):
            self.assertAnswers("variance", (1, 2, 4), ddof)
            self.assertAnswers("stddev", (1, 2, 4), ddof)
            self.assertAnswers("covariance", (1, 2, 4), (2, 3, 7), ddof)
        self.assertAnswers("variance", (5,), 0)
        self.assertFallsBack("variance", (5,), 1)

    def test_defaults_are_zero(self):
        self.assertSameValue(sd.variance((1, 2, 4)), structured("variance", (1, 2, 4))[1])
        self.assertSameValue(sd.covariance((1, 2, 4), (2, 3, 7)), structured("covariance", (1, 2, 4), (2, 3, 7))[1])

    def test_an_active_budget_or_explicit_limits_leave_the_call_to_the_structured_path(self):
        for operation, arguments in self.operations_with((1, 2, 4, 8), 1):
            with self.subTest(operation=operation):
                with computation():
                    self.assertFallsBack(operation, *arguments)
                expected = structured(operation, *arguments)
                limits = ComputationLimits(max_work=1)
                got = outcome(lambda: PUBLIC[operation](*arguments, limits=limits))
                self.assertEqual(got[0], "error")
                self.assertEqual(got[2], "resource_limit_exceeded")
                self.assertEqual(outcome(lambda: PUBLIC[operation](*arguments))[0], expected[0])

    def test_values_are_exact_rationals(self):
        for operation, arguments in self.operations_with((R(1, 3), 2, R(5, 7)), 0):
            value = lean_of(operation)(*arguments)
            self.assertIs(type(value), mf.Rational if operation != "stddev" else Sqrt, operation)


class BoundsTest(LeanStatisticsCase):
    """The lean regime: sizes and bits at which the lean path stops answering, and the structured path still does."""

    def test_the_cap_on_the_size_of_a_sample(self):
        cap = sd.SAMPLE_CAP
        big = tuple(R(index, 7) for index in range(1, cap + 1))
        other = tuple(R(index * index % 101, 11) for index in range(1, cap + 1))
        for operation, arguments in (("mean", (big,)), ("median", (big,)), ("variance", (big, 1)),
                                     ("stddev", (big, 1)), ("covariance", (big, other, 1))):
            with self.subTest(operation=operation, size=cap):
                self.assertAnswers(operation, *arguments)
        over = big + (R(1, 3),)
        over_other = other + (R(2, 3),)
        for operation, arguments in (("mean", (over,)), ("median", (over,)), ("variance", (over, 1)),
                                     ("stddev", (over, 1)), ("covariance", (over, over_other, 1))):
            with self.subTest(operation=operation, size=cap + 1):
                self.assertFallsBack(operation, *arguments)
                expected = structured(operation, *arguments)
                self.assertEqual(expected[0], "value")  # The structured path is still within its limits.
                got = PUBLIC[operation](*arguments)
                self.assertSameValue(got, expected[1])

    def test_the_bits_of_an_entry(self):
        limit = sd.ELEMENT_BITS
        for bits, answers in ((limit, True), (limit + 1, False)):
            entry = R((1 << bits) - 1, 2)  # An odd numerator of `bits` bits over 2: already in lowest terms.
            self.assertEqual(entry.numerator.bit_length(), bits)
            data = (entry, 3, R(1, 5))
            for operation, arguments in (("mean", (data,)), ("median", (data,)), ("variance", (data, 1)),
                                         ("stddev", (data, 0)), ("covariance", (data, data[::-1], 1))):
                with self.subTest(operation=operation, bits=bits):
                    if answers:
                        self.assertAnswers(operation, *arguments)
                    else:
                        self.assertFallsBack(operation, *arguments)
                        expected = structured(operation, *arguments)
                        self.assertEqual(expected[0], "value")
                        self.assertSameValue(PUBLIC[operation](*arguments), expected[1])
        huge = (1 << (mf.ComputationLimits().max_integer_bits + 1), 1)
        for operation, arguments in (("mean", (huge,)), ("median", (huge,)), ("variance", (huge, 0)),
                                     ("covariance", (huge, (1, 2), 0))):
            with self.subTest(operation=operation, bits="over the default limit"):
                self.assertAgrees(operation, *arguments)
                self.assertFallsBack(operation, *arguments)

    def test_the_extremes_of_the_regime_are_within_the_structured_limits(self):
        """Entries of ELEMENT_BITS bits over one denominator of COMMON_BITS bits: the structured path answers."""
        denominator = 3 ** 5150  # 8,165 bits: inside both caps.
        self.assertLessEqual(denominator.bit_length(), sd.COMMON_BITS)
        rng = random.Random(5)
        entries = tuple(R(rng.getrandbits(8100) | 1, denominator) for _ in range(6))
        other = tuple(R(rng.getrandbits(8100) | 1, denominator) for _ in range(6))
        for operation, arguments in (("mean", (entries,)), ("median", (entries,)), ("variance", (entries, 1)),
                                     ("stddev", (entries, 1)), ("covariance", (entries, other, 1))):
            with self.subTest(operation=operation):
                self.assertAnswers(operation, *arguments)
        # Six distinct denominators of 1,361 bits (a common denominator of about 8,160 bits), entries of 8,100 bits:
        # the structured intermediates are larger here (about 45% of max_integer_bits), and it still answers.
        denominators = [(1 << 1360) + 2 * k + 1 for k in range(1, 7)]
        entries = tuple(R(rng.getrandbits(8100) | 1, d) for d in denominators)
        other = tuple(R(rng.getrandbits(8100) | 1, d) for d in reversed(denominators))
        self.assertLessEqual(sd._scale(*sd._prepare(entries)[1:])[0].bit_length(), sd.COMMON_BITS)
        for operation, arguments in (("mean", (entries,)), ("median", (entries,)), ("variance", (entries, 1)),
                                     ("stddev", (entries, 1)), ("covariance", (entries, other, 1))):
            with self.subTest(operation=operation, denominators="distinct"):
                self.assertAnswers(operation, *arguments)

    def test_the_common_denominator(self):
        # Distinct large denominators whose lcm has too many bits: the lean path steps aside, the structured path
        # (or its resource limit) decides.
        primes = [(1 << 3000) + offset for offset in (7, 19, 39)]
        data = tuple(R(1, prime) for prime in primes)
        self.assertGreater(sum(prime.bit_length() for prime in primes), sd.COMMON_BITS)
        for operation, arguments in (("mean", (data,)), ("median", (data,)), ("variance", (data, 0))):
            with self.subTest(operation=operation, lcm="too large"):
                self.assertAgrees(operation, *arguments)
                self.assertFallsBack(operation, *arguments)
        # Many entries whose denominators sum to many bits but whose lcm is small: answered.
        wide = tuple(R(index, (1 << 100) + 1) for index in range(1, 150))
        self.assertGreater(sum(entry.denominator.bit_length() for entry in wide), sd.COMMON_BITS)
        for operation, arguments in (("mean", (wide,)), ("median", (wide,)), ("variance", (wide, 1)),
                                     ("covariance", (wide, wide[::-1], 1))):
            with self.subTest(operation=operation, lcm="small"):
                self.assertAnswers(operation, *arguments)

    def test_scale_and_prepare(self):
        self.assertEqual(sd._scale([3, 5], [2, 3]), (6, [9, 10]))
        self.assertEqual(sd._scale([3, 5], [1, 1]), (1, [3, 5]))
        self.assertIsNone(sd._scale([1, 1, 1], [(1 << 4000) + 7, (1 << 4000) + 9, (1 << 4000) + 15]))
        self.assertEqual(sd._scale([1, 2, 3], [(1 << 4000) + 7] * 3)[0], (1 << 4000) + 7)
        items, nums, dens = sd._prepare([R(1, 2), 3, R(-5, 6)])
        self.assertEqual((items, nums, dens), ((R(1, 2), 3, R(-5, 6)), [1, 3, -5], [2, 1, 6]))
        for bad in ([], (), [True], [1.5], "12", None, 7):
            self.assertIsNone(sd._prepare(bad))


class TamperedProducerTest(LeanStatisticsCase):
    """Verification is never lazy: a value comes back only when its check answers True."""

    DATA = (R(1, 3), 4, R(-5, 7), 9, 2)

    def calls(self):
        return (("mean", (self.DATA,)), ("median", (self.DATA,)), ("variance", (self.DATA, 1)),
                ("stddev", (self.DATA, 0)), ("covariance", (self.DATA, self.DATA[::-1], 1)))

    def test_a_wrong_value_is_never_returned(self):
        for operation, arguments in self.calls():
            true = lean_of(operation)(*arguments)
            self.assertIsNot(true, FALLBACK)
            for wrong in (R(99, 7), true + 1 if type(true) is R else R(2), R(0)):
                with self.subTest(operation=operation, wrong=str(wrong)), \
                        patch.object(sd, "_fraction", lambda numerator, denominator, _wrong=wrong: _wrong):
                    self.assertIs(lean_of(operation)(*arguments), FALLBACK)
                    # The public function still gives the right value, through the structured path.
                    self.assertSameValue(PUBLIC[operation](*arguments), true)

    def test_a_wrong_square_root_is_never_returned(self):
        arguments = ((1, 2, 4), 0)  # sqrt(14/9): not a perfect square.
        true = lean_of("stddev")(*arguments)
        self.assertEqual(true, Sqrt(R(14, 9)))
        # A wrong variance under the root (the producer's variance step is broken): refused.
        for wrong in (R(15, 9), R(14, 8), R(0), R(1, 4)):
            with self.subTest(variance=str(wrong)), patch.object(sd, "_variance_value", lambda loaded, ddof, _w=wrong: _w):
                self.assertIs(lean_of("stddev")(*arguments), FALLBACK)
        # A perfect square kept under the root, or handed back as its negative root: refused by the check.
        perfect = (R(1, 2), R(3, 2))
        self.assertEqual(lean_of("stddev")(perfect, 0), R(1, 2))
        self.assertTrue(sd._check_stddev(perfect, 2, 0, R(1, 2)))
        self.assertFalse(sd._check_stddev(perfect, 2, 0, Sqrt(R(1, 4))))
        self.assertFalse(sd._check_stddev(perfect, 2, 0, R(-1, 2)))
        with patch.object(sd, "trusted_rational", lambda n, d: trusted_rational(-n, d)):
            self.assertIs(lean_of("stddev")(perfect, 0), FALLBACK)

    def test_a_check_that_does_not_answer_true_gives_fallback(self):
        names = {"mean": "_check_mean", "variance": "_check_variance", "stddev": "_check_stddev",
                 "covariance": "_check_covariance", "median": "median_certificate"}
        for operation, arguments in self.calls():
            for label, answer in (("rejecting", lambda *a: False), ("truthy", lambda *a: 1), ("none", lambda *a: None)):
                with self.subTest(operation=operation, check=label), patch.object(sd, names[operation], answer):
                    self.assertIs(lean_of(operation)(*arguments), FALLBACK)
        for operation, arguments in self.calls():
            self.assertIsNot(lean_of(operation)(*arguments), FALLBACK)

    def test_the_checks_do_not_use_the_producers_arithmetic(self):
        """The checks run on the sample's own Rationals: with the scaling and the reduction broken they still reject."""
        with patch.object(sd, "_scale", side_effect=AssertionError("the check scaled the sample")):
            items = self.DATA
            self.assertTrue(sd._check_mean(items, 5, structured("mean", items)[1]))
            self.assertTrue(sd._check_variance(items, 5, 1, structured("variance", items, 1)[1]))
            self.assertTrue(sd._check_covariance(items, items[::-1], 5, 1,
                                                 structured("covariance", items, items[::-1], 1)[1]))
            self.assertTrue(sd._check_stddev(items, 5, 0, structured("stddev", items, 0)[1]))


class ChecksAgreeWithV07Test(LeanStatisticsCase):
    """The lean checks accept and reject the values that v0.7's checker (statistics.check_operation) does."""

    def candidates(self, true, other):
        values = [true, other, R(0), R(1), true + R(1, 997), true - R(1, 997), true + 1, -true if type(true) is R else R(-1)]
        if type(true) is R:
            values += [true * 2, true * true, R(true.denominator, true.numerator) if true.numerator else R(5)]
        return values

    def test_mean_variance_covariance_checks(self):
        from mathforge import statistics
        rng = random.Random(99)
        for index in range(60):
            size = rng.randint(1, 12)
            flavor = ("ints", "fractions", "mixed", "ties")[index % 4]
            data, other = sample(rng, size, flavor), sample(rng, size, flavor)
            ddof = rng.randrange(size)
            true = {"mean": structured("mean", data)[1], "variance": structured("variance", data, ddof)[1],
                    "covariance": structured("covariance", data, other, ddof)[1]}
            decoys = {"mean": structured("mean", other)[1], "variance": structured("variance", other, ddof)[1],
                      "covariance": structured("covariance", other, data[::-1], ddof)[1]}
            checks = {
                "mean": (lambda v: sd._check_mean(data, size, v), {"data": data}),
                "variance": (lambda v: sd._check_variance(data, size, ddof, v), {"data": data, "ddof": ddof}),
                "covariance": (lambda v: sd._check_covariance(data, other, size, ddof, v),
                               {"x": data, "y": other, "ddof": ddof}),
            }
            for operation, (lean_check, args) in checks.items():
                for value in self.candidates(true[operation], decoys[operation]):
                    with self.subTest(operation=operation, index=index, value=str(value)):
                        self.assertEqual(lean_check(value), statistics.check_operation(operation, args, value, ()))
                        self.assertEqual(lean_check(value), value == true[operation])

    def test_stddev_check(self):
        from mathforge import statistics
        rng = random.Random(3)
        for index in range(40):
            size = rng.randint(1, 10)
            data = sample(rng, size, ("ints", "fractions", "ties", "mixed")[index % 4])
            ddof = rng.randrange(size)
            args = {"data": data, "ddof": ddof}
            true = structured("stddev", data, ddof)[1]
            variance = structured("variance", data, ddof)[1]
            candidates = [true, Sqrt(variance + 1), R(0), R(1), variance, -variance, Sqrt(R(1, 2)), 1.5, None]
            if type(true) is R:
                candidates += [-true, true + 1]
            for value in candidates:
                with self.subTest(index=index, value=repr(value)):
                    self.assertEqual(sd._check_stddev(data, size, ddof, value),
                                     statistics.check_operation("stddev", args, value, ()))
            self.assertTrue(sd._check_stddev(data, size, ddof, true))
            # The one difference: v0.7's check also accepts a Sqrt whose radicand is the square of a rational, the
            # lean check pins the form that model.sqrt gives.
            unreduced = Sqrt(variance)
            self.assertTrue(statistics.check_operation("stddev", args, unreduced, ()))
            self.assertEqual(sd._check_stddev(data, size, ddof, unreduced), type(true) is Sqrt)


if __name__ == "__main__":
    unittest.main()
