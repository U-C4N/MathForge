"""BoundedCache: least-recently-used eviction, hits charged their nominal cost, thread safety, the process-wide reset."""

import gc
import random
import threading
import time
import unittest
import weakref

from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation, current_budget
from mathforge.trusted.caches import BoundedCache, clear_all

WORKERS = 8


def counters(budget):
    """The four counters that a cache hit replays: work, node_count, refinements, evidence_count."""
    return (budget.work, budget.node_count, budget.refinements, budget.evidence_count)


def charging(work=0, nodes=0, refinements=0, evidence=0, value="value"):
    """A compute function that charges the active budget (if any), returns value and counts its runs in .runs.

    ``Budget.refine(n)`` also ticks work by n, so a miss moves the counters to
    (work + refinements, nodes, refinements, evidence).
    """
    def compute():
        compute.runs += 1
        budget = current_budget()
        if budget is not None:
            budget.tick(work)
            budget.nodes(nodes)
            budget.refine(refinements)
            budget.evidence(evidence)
        return value
    compute.runs = 0
    return compute


class EvictionTest(unittest.TestCase):
    def test_least_recently_used_entry_is_evicted_first(self):
        cache = BoundedCache("test.lru", 3)
        computed = []

        def get(key):
            return cache.get(key, lambda: computed.append(key) or key * 10)

        for key in (1, 2, 3):
            self.assertEqual(get(key), key * 10)
        get(1)  # A hit makes 1 the most recent: oldest to newest is now 2, 3, 1.
        get(4)  # Evicts 2.
        self.assertEqual(computed, [1, 2, 3, 4])
        self.assertEqual((get(1), get(3), get(4)), (10, 30, 40))  # Hits; oldest to newest is now 1, 3, 4.
        self.assertEqual(computed, [1, 2, 3, 4])
        get(2)  # A miss; it evicts 1, the least recent.
        self.assertEqual(computed, [1, 2, 3, 4, 2])
        get(3)
        get(4)
        get(2)  # Hits.
        self.assertEqual(computed, [1, 2, 3, 4, 2])
        get(1)  # Evicted above, so computed again.
        self.assertEqual(computed, [1, 2, 3, 4, 2, 1])

    def test_a_cache_never_holds_more_than_maxsize_entries(self):
        cache = BoundedCache("test.size", 5)
        for key in range(100):
            cache.get(key, lambda: key)
            self.assertLessEqual(len(cache), 5)
        self.assertEqual(len(cache), 5)

    def test_a_cache_of_one_entry_keeps_the_latest(self):
        cache = BoundedCache("test.single", 1)
        first, second = charging(value="a"), charging(value="b")
        cache.get("a", first)
        cache.get("b", second)
        cache.get("b", second)
        cache.get("a", first)
        self.assertEqual((first.runs, second.runs), (2, 1))


class BasicsTest(unittest.TestCase):
    def test_none_and_other_falsy_values_are_cached(self):
        cache = BoundedCache("test.falsy", 8)
        for value in (None, False, 0, ()):
            compute = charging(value=value)
            self.assertIs(cache.get(("k", repr(value)), compute), value)
            self.assertIs(cache.get(("k", repr(value)), compute), value)
            self.assertEqual(compute.runs, 1)

    def test_arguments_are_validated(self):
        for name, maxsize in (("", 1), (None, 1), (3, 1), ("test.zero", 0), ("test.negative", -1),
                              ("test.bool", True), ("test.float", 1.5), ("test.text", "2")):
            with self.subTest(name=name, maxsize=maxsize), self.assertRaises(InvalidInput):
                BoundedCache(name, maxsize)


class ResetTest(unittest.TestCase):
    def test_clear_all_clears_every_cache(self):
        first, second = BoundedCache("test.first", 4), BoundedCache("test.second", 4)
        a, b = charging(value=1), charging(value=2)
        for _ in range(2):
            first.get("k", a)
            second.get("k", b)
        self.assertEqual((a.runs, b.runs, len(first), len(second)), (1, 1, 1, 1))
        clear_all()
        self.assertEqual((len(first), len(second)), (0, 0))
        self.assertEqual((first.get("k", a), second.get("k", b)), (1, 2))
        self.assertEqual((a.runs, b.runs), (2, 2))

    def test_clear_clears_only_its_own_cache(self):
        first, second = BoundedCache("test.cleared", 4), BoundedCache("test.kept", 4)
        a, b = charging(value=1), charging(value=2)
        first.get("k", a)
        second.get("k", b)
        first.clear()
        first.get("k", a)
        second.get("k", b)
        self.assertEqual((a.runs, b.runs), (2, 1))

    def test_clear_all_reaches_caches_made_in_other_threads(self):
        made = []
        worker = threading.Thread(target=lambda: made.append(BoundedCache("test.thread", 2)))
        worker.start()
        worker.join()
        (cache,) = made
        compute = charging()
        cache.get("k", compute)
        clear_all()
        cache.get("k", compute)
        self.assertEqual(compute.runs, 2)

    def test_the_registry_does_not_keep_a_dropped_cache_alive(self):
        cache = BoundedCache("test.dropped", 2)
        probe = weakref.ref(cache)
        del cache
        gc.collect()
        self.assertIsNone(probe())
        clear_all()  # Must cope with the dropped entry.


class NominalChargeTest(unittest.TestCase):
    CHARGE = dict(work=5, nodes=3, refinements=7, evidence=2)
    EXPECTED = (12, 3, 7, 2)  # refine(7) ticks work by 7 as well: 5 + 7

    def test_the_work_after_a_hit_equals_the_work_after_a_miss(self):
        cache = BoundedCache("test.nominal", 4)
        compute = charging(**self.CHARGE)
        with computation() as budget:
            self.assertEqual(counters(budget), (0, 0, 0, 0))
            cache.get("k", compute)  # Miss.
            after_miss = counters(budget)
            cache.get("k", compute)  # Hit.
            after_hit = counters(budget)
        self.assertEqual(compute.runs, 1)
        self.assertEqual(after_miss, self.EXPECTED)
        self.assertEqual(after_hit, tuple(2 * count for count in self.EXPECTED))

    def test_a_cold_and_a_warm_call_leave_equal_counters_in_separate_budgets(self):
        cache = BoundedCache("test.separate", 4)
        compute = charging(**self.CHARGE)
        seen = []
        for _ in range(3):
            with computation() as budget:
                cache.get("k", compute)
                seen.append(counters(budget))
        self.assertEqual(compute.runs, 1)
        self.assertEqual(seen, [self.EXPECTED] * 3)

    def test_a_hit_adds_to_what_the_budget_already_holds(self):
        cache = BoundedCache("test.accumulate", 4)
        compute = charging(**self.CHARGE)
        with computation():
            cache.get("k", compute)
        with computation() as budget:
            budget.tick(100)
            budget.nodes(10)
            budget.refine(1)
            budget.evidence(1)
            cache.get("k", compute)
            self.assertEqual(counters(budget), (100 + 1 + 12, 10 + 3, 1 + 7, 1 + 2))

    def test_a_miss_records_only_its_own_charges(self):
        cache = BoundedCache("test.delta", 4)
        compute = charging(**self.CHARGE)
        with computation() as budget:
            budget.tick(1000)
            budget.refine(40)
            cache.get("k", compute)
            self.assertEqual(counters(budget), (1000 + 40 + 12, 3, 40 + 7, 2))
        with computation() as budget:
            cache.get("k", compute)
            self.assertEqual(counters(budget), self.EXPECTED)

    def test_charges_of_every_counter_are_replayed_alone(self):
        for label, charge, expected in (
                ("work", dict(work=9), (9, 0, 0, 0)),
                ("nodes", dict(nodes=9), (0, 9, 0, 0)),
                ("refinements", dict(refinements=9), (9, 0, 9, 0)),
                ("evidence", dict(evidence=9), (0, 0, 0, 9)),
                ("nothing", dict(), (0, 0, 0, 0))):
            with self.subTest(label):
                cache = BoundedCache("test." + label, 4)
                compute = charging(**charge)
                for _ in range(2):
                    with computation() as budget:
                        cache.get("k", compute)
                        self.assertEqual(counters(budget), expected)
                self.assertEqual(compute.runs, 1)

    def test_a_hit_charges_through_the_limit_checks(self):
        # Each case: the limit that is exactly enough for one charge, and a call that uses one unit of that counter.
        cases = (("work", "max_work", 12, lambda budget: budget.tick(1)),
                 ("nodes", "max_nodes", 3, lambda budget: budget.nodes(1)),
                 ("refinements", "max_refinements", 7, lambda budget: budget.refine(1)),
                 ("evidence", "max_evidence_items", 2, lambda budget: budget.evidence(1)))
        for label, field, recorded, use_one in cases:
            with self.subTest(label):
                limits = ComputationLimits(**{field: recorded})
                cache = BoundedCache("test.limit." + label, 4)
                compute = charging(**self.CHARGE)
                with computation(limits) as budget:  # The miss fits exactly.
                    cache.get("k", compute)
                    self.assertEqual(counters(budget), self.EXPECTED)
                with computation(limits) as budget:  # So does a hit.
                    cache.get("k", compute)
                    self.assertEqual(counters(budget), self.EXPECTED)
                with computation(limits) as budget:  # With one unit already used, the replay is one over the limit.
                    use_one(budget)
                    with self.assertRaisesRegex(ResourceLimitError, "Computation limit exceeded"):
                        cache.get("k", compute)
                self.assertEqual(compute.runs, 1, "the later calls must have been hits")

    def test_outcomes_under_other_counter_limits_are_those_of_a_cold_call(self):
        for label, field, recorded in (("work", "max_work", 12), ("nodes", "max_nodes", 3),
                                       ("refinements", "max_refinements", 7), ("evidence", "max_evidence_items", 2)):
            with self.subTest(label):
                cache = BoundedCache("test.other." + label, 4)
                compute = charging(**self.CHARGE)
                with computation():
                    cache.get("k", compute)
                with computation(ComputationLimits(**{field: recorded})) as budget:  # Exactly enough.
                    self.assertEqual(cache.get("k", compute), "value")
                    self.assertEqual(counters(budget), self.EXPECTED)
                with computation(ComputationLimits(**{field: recorded - 1})):  # One short.
                    with self.assertRaisesRegex(ResourceLimitError, "Computation limit exceeded"):
                        cache.get("k", compute)

    def test_a_hit_cannot_hide_a_limit_failure_across_calls_in_one_budget(self):
        cache = BoundedCache("test.cumulative", 4)
        compute = charging(work=60)
        with computation(ComputationLimits(max_work=100)):
            cache.get("k", compute)  # Miss: 60 of 100.
            with self.assertRaisesRegex(ResourceLimitError, r"work \(100\)"):
                cache.get("k", compute)  # Hit: 120 would exceed 100, as a second cold computation would.
        self.assertEqual(compute.runs, 1)

    def test_nothing_is_charged_without_a_budget(self):
        cache = BoundedCache("test.nobudget", 4)
        compute = charging(**self.CHARGE)
        self.assertIsNone(current_budget())
        self.assertEqual((cache.get("k", compute), cache.get("k", compute)), ("value", "value"))
        self.assertEqual(compute.runs, 1)
        self.assertIsNone(current_budget())

    def test_a_value_made_without_a_budget_is_not_free_under_one(self):
        cache = BoundedCache("test.unmetered", 4)
        compute = charging(**self.CHARGE)
        cache.get("k", compute)  # No budget is active: nothing could be measured.
        for _ in range(2):
            with computation() as budget:
                cache.get("k", compute)
                self.assertEqual(counters(budget), self.EXPECTED)
        self.assertEqual(compute.runs, 2)  # One call without a budget, one miss with; then a hit.
        cache.get("k", compute)  # Back without a budget: its own entry, still a hit.
        self.assertEqual(compute.runs, 2)

    def test_a_value_is_not_served_under_limits_other_than_the_ones_it_was_made_under(self):
        cache = BoundedCache("test.scope", 8)

        def compute():
            compute.runs += 1
            current_budget().check_int(2**100)  # Refused under max_integer_bits=64.
            return "big"
        compute.runs = 0

        with computation():
            self.assertEqual(cache.get("k", compute), "big")
        with computation(ComputationLimits(max_integer_bits=64)):
            with self.assertRaisesRegex(ResourceLimitError, "integer bits"):
                cache.get("k", compute)  # A cold computation under these limits fails, so this must too.
        self.assertEqual(compute.runs, 2)
        with computation():
            self.assertEqual(cache.get("k", compute), "big")  # The default-limits entry survived.
        self.assertEqual(compute.runs, 2)

    def test_a_cap_read_from_a_counter_limit_is_enforced_on_every_call(self):
        cache = BoundedCache("test.cap", 8)

        def compute():  # As functions.py does: a request larger than max_work is refused whatever has been used.
            compute.runs += 1
            if 1000 > current_budget().limits.max_work:
                raise ResourceLimitError("Power exceeds computation budget")
            return "allowed"
        compute.runs = 0

        with computation():
            self.assertEqual(cache.get("k", compute), "allowed")
        with computation(ComputationLimits(max_work=500)):
            with self.assertRaisesRegex(ResourceLimitError, "Power exceeds computation budget"):
                cache.get("k", compute)
        self.assertEqual(compute.runs, 2)

    def test_equal_limits_share_entries_between_budgets(self):
        cache = BoundedCache("test.shared", 8)
        compute = charging(work=3)
        for _ in range(3):
            with computation(ComputationLimits(max_work=1234, max_depth=7)) as budget:  # A new, equal object each time.
                cache.get("k", compute)
                self.assertEqual(budget.work, 3)
        self.assertEqual(compute.runs, 1)

    def test_a_failed_computation_stores_nothing(self):
        cache = BoundedCache("test.failure", 4)
        compute = charging(work=100)
        with computation(ComputationLimits(max_work=50)):
            with self.assertRaises(ResourceLimitError):
                cache.get("k", compute)
        self.assertEqual(len(cache), 0)
        with computation() as budget:
            self.assertEqual(cache.get("k", compute), "value")
            self.assertEqual(budget.work, 100)
        self.assertEqual(compute.runs, 2)

    def test_an_exception_from_compute_passes_through_unchanged(self):
        cache = BoundedCache("test.passthrough", 4)

        def compute():
            raise KeyError("boom")

        with computation(), self.assertRaisesRegex(KeyError, "boom"):
            cache.get("k", compute)
        self.assertEqual(len(cache), 0)

    def test_nested_caches_charge_their_whole_cost_on_every_hit(self):
        inner, outer = BoundedCache("test.inner", 4), BoundedCache("test.outer", 4)
        inner_compute = charging(work=4, nodes=1)

        def outer_compute():
            outer_compute.runs += 1
            current_budget().tick(2)
            return inner.get("i", inner_compute) + "!"
        outer_compute.runs = 0

        def run():
            with computation() as budget:
                value = outer.get("o", outer_compute)
                return value, counters(budget)

        cold = run()  # Both miss.
        outer_hit = run()  # The outer cache answers; the inner one is not consulted.
        outer.clear()
        inner_hit = run()  # The outer cache misses and its computation hits the inner one.
        self.assertEqual(cold, ("value!", (6, 1, 0, 0)))
        self.assertEqual(outer_hit, cold)
        self.assertEqual(inner_hit, cold)
        self.assertEqual((inner_compute.runs, outer_compute.runs), (1, 2))

    def test_compute_may_use_the_cache_it_runs_in(self):
        cache = BoundedCache("test.reentrant", 8)

        def square(n):
            return cache.get(("square", n), lambda: n * n)

        def sum_of_squares(n):
            return cache.get(("sum", n), lambda: sum(square(i) for i in range(n + 1)))

        results = []
        worker = threading.Thread(target=lambda: results.append(sum_of_squares(5)), daemon=True)
        worker.start()
        worker.join(10)
        self.assertFalse(worker.is_alive(), "get() must not hold its lock while it computes")
        self.assertEqual(results, [55])

    def test_unhashable_keys_are_rejected_before_anything_is_charged(self):
        cache = BoundedCache("test.unhashable", 4)
        compute = charging(**self.CHARGE)
        with computation() as budget:
            with self.assertRaises(TypeError):
                cache.get(["not", "hashable"], compute)
            self.assertEqual(counters(budget), (0, 0, 0, 0))
        self.assertEqual(compute.runs, 0)


class ThreadSafetyTest(unittest.TestCase):
    KEYS = 40

    def hammer(self, cache, with_budget):
        """Run WORKERS threads that fetch every key in their own order; return each thread's outcome."""
        barrier = threading.Barrier(WORKERS)
        outcomes = [None] * WORKERS
        failures = []

        def compute_for(key):
            def compute():
                time.sleep(0.0002)  # Long enough for several threads to miss on one key together.
                budget = current_budget()
                if budget is not None:
                    budget.tick(key + 1)
                    budget.nodes(2)
                return (key, key * key)  # A new tuple on every call, so identity shows who won the race.
            return compute

        def work(index):
            try:
                order = list(range(self.KEYS))
                random.Random(index).shuffle(order)
                barrier.wait(timeout=30)
                if with_budget:
                    with computation() as budget:
                        values = {key: cache.get(key, compute_for(key)) for key in order}
                        outcomes[index] = (values, counters(budget))
                else:
                    outcomes[index] = ({key: cache.get(key, compute_for(key)) for key in order}, None)
            except BaseException as error:  # Reported below, with the thread's index.
                failures.append((index, repr(error)))

        threads = [threading.Thread(target=work, args=(index,)) for index in range(WORKERS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertEqual(failures, [])
        self.assertFalse(any(thread.is_alive() for thread in threads))
        return outcomes

    def test_eight_threads_sharing_one_cache_get_identical_values(self):
        cache = BoundedCache("test.threads", 64)
        outcomes = self.hammer(cache, with_budget=False)
        expected = {key: (key, key * key) for key in range(self.KEYS)}
        for values, _ in outcomes:
            self.assertEqual(values, expected)
        for key in range(self.KEYS):
            self.assertEqual(len({id(values[key]) for values, _ in outcomes}), 1, f"key {key}: more than one object")
        self.assertEqual(len(cache), self.KEYS)

    def test_eight_threads_with_budgets_are_charged_identically(self):
        cache = BoundedCache("test.threads.budget", 64)
        outcomes = self.hammer(cache, with_budget=True)
        expected_work = sum(key + 1 for key in range(self.KEYS))
        expected = (expected_work, 2 * self.KEYS, 0, 0)
        self.assertEqual({counts for _, counts in outcomes}, {expected})
        # A later, fully warm pass is charged the same.
        with computation() as budget:
            for key in range(self.KEYS):
                cache.get(key, lambda: self.fail("a warm cache must not compute"))
            self.assertEqual(counters(budget), expected)

    def test_threads_hammering_a_small_cache_stay_consistent(self):
        cache = BoundedCache("test.threads.small", 7)  # Far fewer entries than keys: constant eviction.
        outcomes = self.hammer(cache, with_budget=True)
        expected = {key: (key, key * key) for key in range(self.KEYS)}
        for values, counts in outcomes:
            self.assertEqual(values, expected)
            self.assertEqual(counts, (sum(key + 1 for key in range(self.KEYS)), 2 * self.KEYS, 0, 0))
        self.assertLessEqual(len(cache), 7)


if __name__ == "__main__":
    unittest.main()
