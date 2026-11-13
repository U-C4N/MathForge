"""The exact_tests family of mathforge.stats: fisher_exact.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register stats exact_tests

Run it by name with python -m unittest tests.v09.stats.test_exact_tests. The reference below sums the masses by brute force with
math.comb at every point of the support, as the producer does not (it steps a recurrence and takes C(n, c1) for the
total), and the checker forms no binomial coefficient at all.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.stats import exact_tests as family
from mathforge.contracts import ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational

from .. import support

Table = tuple[tuple[int, int], tuple[int, int]]

KNOWN: tuple[tuple[Table, Rational], ...] = (  # docs/v09/cards/m7-extended.md
    (((1, 9), (11, 3)), Rational(41, 14858)),
    (((3, 1), (1, 3)), Rational(17, 35)),
    (((0, 5), (5, 0)), Rational(1, 126)),
    (((2, 2), (2, 2)), Rational(1)),
    (((8, 2), (1, 5)), Rational(5, 143)),
)


def reference(table: Table) -> Rational:
    """The two-sided Fisher p-value by brute force: every mass C(r1, k) C(r2, c1 - k) / C(n, c1) of the support by
    math.comb, and the sum of those not above the observed one, as a Fraction."""
    (a, b), (c, d) = table
    r1, r2, c1 = a + b, c + d, a + c
    weights = {k: comb(r1, k) * comb(r2, c1 - k) for k in range(max(0, c1 - r2), min(r1, c1) + 1)}
    observed = weights[a]
    value = Fraction(sum(weight for weight in weights.values() if weight <= observed), sum(weights.values()))
    return Rational(value.numerator, value.denominator)


def matrix(table: Table) -> mf.Matrix:
    return mf.Matrix(table)


def seeded_table(rng: random.Random) -> Table:
    """A 2 x 2 table of counts, not all zero: mostly small counts, some with zeros, some up to a few hundred."""
    while True:
        shape = rng.randrange(4)
        if shape == 0:
            counts = [rng.randrange(0, 6) for _ in range(4)]
        elif shape == 1:
            counts = [rng.randrange(0, 30) for _ in range(4)]
        elif shape == 2:
            counts = [rng.randrange(0, 300) for _ in range(4)]
        else:
            counts = [rng.choice((0, rng.randrange(1, 60))) for _ in range(4)]
        if any(counts):
            a, b, c, d = counts
            return (a, b), (c, d)


def wide_matrix(table: tuple) -> mf.Matrix:
    """A Matrix with integers past the default max_integer_bits, built inside a budget wide enough for it."""
    with computation(ComputationLimits(max_integer_bits=1 << 20), fresh=True):
        return mf.Matrix(table)


class ExactTestsTests(support.FamilyTestCase):
    FAMILY = family

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for table, expected in KNOWN:
            with self.subTest(table=table):
                result = support.run("fisher_exact", table=matrix(table))
                self.assertIsNone(result.error)
                self.assertEqual(result.value, expected)
                self.assertIs(type(result.value), Rational)
                self.assertEqual(result.value, reference(table))
                self.assertEqual(result.steps[0].outputs[1], ())
                report = mf.verify(result)
                self.assertTrue(report.verified, support.statuses(report))
                self.assertTrue(set(support.CLAIMS) <= set(support.statuses(report)), support.statuses(report))
                self.roundtrip(result)
        # Degenerate margins: one point of support, p = 1; the observed count at both ends of the support.
        for table in (((100000, 0), (0, 0)), ((0, 0), (0, 7)), ((5, 0), (3, 0)), ((0, 4), (9, 0)), ((7, 0), (0, 9)),
                      ((300, 200), (150, 350))):
            with self.subTest(table=table):
                result = support.run("fisher_exact", table=matrix(table))
                self.assertEqual(result.value, reference(table))
                self.assertTrue(mf.verify(result).verified)
        self.assertEqual(support.run("fisher_exact", table=matrix(((100000, 0), (0, 0)))).value, Rational(1))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error."""
        r = Rational
        cases = [0.5, True, "1", None, [[1, 2], [3, 4]], ((1, 2), (3, 4)), 3, r(3), mf.Symbol("x")]
        cases += [matrix(rows) for rows in (((1, 2, 3), (4, 5, 6)), ((1, 2), (3, 4), (5, 6)), ((1,),), ((1, 2),))]
        cases += [mf.Matrix((), ncols=0), mf.Matrix((), ncols=2), mf.Matrix([[1, 2]] * 3).T]
        cases += [matrix(((0, 0), (0, 0))), matrix(((-1, 2), (3, 4))), matrix(((1, 2), (3, -4))),
                  mf.Matrix([[r(1, 2), 2], [3, 4]]), mf.Matrix([[1, 2], [3, r(7, 3)]])]
        one = r(1)
        for bad in (0.5, True, "1", None, 1, [one, one], r(1, 2)):  # In every entry of a Matrix forged past __init__.
            for position in range(4):
                entries = [one] * 4
                entries[position] = bad
                rows = ((entries[0], entries[1]), (entries[2], entries[3]))
                cases.append(support.forged(mf.Matrix, rows=rows, ncols=2))
        forged_rows = ((one, one), (one, one))
        for ncols in (0.5, True, "2", None, 3):
            cases.append(support.forged(mf.Matrix, rows=forged_rows, ncols=ncols))
        for rows in ([(one, one), (one, one)], ((one, one), [one, one]), ((one, one),), ((one, one, one), (one, one)),
                     None, "rows"):
            cases.append(support.forged(mf.Matrix, rows=rows, ncols=2))
        cases.append(support.forged(mf.Matrix, rows=((support.forged(Rational, numerator=4, denominator=2), one),
                                                     (one, one)), ncols=2))
        for index, table in enumerate(cases):
            with self.subTest(case=index):
                self.assertInvalidInput(support.run("fisher_exact", table=table))
        # A Rational without its denominator slot stops the request's inspection before produce() runs; produce()
        # itself refuses it as well.
        slotless = support.forged(mf.Matrix, rows=((support.forged(Rational, numerator=4), one), (one, one)), ncols=2)
        with computation(fresh=True), self.assertRaises(InvalidInput):
            family.produce("fisher_exact", {"table": slotless})
        for bad in (0.5, True, "1", None):  # The constructor refuses them before any request is made.
            with self.subTest(entry=bad), self.assertRaises(InvalidInput):
                mf.Matrix([[bad, 1], [1, 1]])
        # Wrong argument sets and operation names reach produce() only directly.
        with computation(fresh=True):
            for args in ({}, {"table": matrix(((1, 2), (3, 4))), "extra": 1}, {"M": matrix(((1, 2), (3, 4)))},
                         [("table", matrix(((1, 2), (3, 4))))]):
                with self.subTest(args=args), self.assertRaises(InvalidInput):
                    family.produce("fisher_exact", args)
            for name in ("fisher", "Fisher_exact", None, b"fisher_exact"):
                with self.subTest(operation=name), self.assertRaises(UnsupportedOperation):
                    family.produce(name, {"table": matrix(((1, 2), (3, 4)))})
        # Over MCP: a ragged table fails at decoding, a fractional count in the computation.
        self.assertToolError(support.call("fisher_exact", {"table": [[1, 9], [11]]}), "invalid_input")
        error = self.assertToolError(support.call("fisher_exact", {"table": [["1/2", 9], [11, 3]]}), "invalid_input")
        self.assertIn("integer counts", error["message"])
        error = self.assertToolError(support.call("fisher_exact", {"table": [[0, 0], [0, 0]]}), "invalid_input")
        self.assertIn("nonzero", error["message"])

    def test_resource_limits(self):
        """An integer of 2**65536, a 65 x 65 matrix where one is taken, and max_work=1 exceed the budget."""
        self.assertResourceLimit(support.run("fisher_exact", table=wide_matrix(((support.HUGE, 1), (1, 1)))))
        self.assertResourceLimit(support.run("fisher_exact", table=wide_matrix(((1, 1), (1, support.HUGE)))))
        self.assertResourceLimit(support.run("fisher_exact", table=support.oversized_matrix()))
        self.assertResourceLimit(support.run("fisher_exact", table=matrix(((1, 9), (11, 3))),
                                             limits=ComputationLimits(max_work=1)))
        # The card's limit: a total above 100,000, also when the table is trivial.
        for table in (((100001, 0), (0, 0)), ((25001, 25000), (25000, 25000)), ((1, 2), (3, 99995))):
            with self.subTest(table=table):
                self.assertResourceLimit(support.run("fisher_exact", table=matrix(table)))
        with computation(fresh=True), self.assertRaisesRegex(ResourceLimitError, "at most 100,000"):
            family.produce("fisher_exact", {"table": matrix(((100001, 0), (0, 0)))})
        # The 65 x 65 matrix is refused before its shape (not 2 x 2) is judged; a 64 x 64 one is only invalid.
        with computation(fresh=True), self.assertRaises(ResourceLimitError):
            family.produce("fisher_exact", {"table": support.oversized_matrix()})
        with computation(fresh=True), self.assertRaises(InvalidInput):
            family.produce("fisher_exact", {"table": mf.Matrix([[1] * 64] * 64)})
        # A p-value longer than max_integer_bits is never returned: the checker would refuse it.
        with computation(ComputationLimits(max_integer_bits=64), fresh=True), \
                self.assertRaisesRegex(ResourceLimitError, "the p-value"):
            family.produce("fisher_exact", {"table": matrix(((40, 60), (60, 40)))})
        self.assertResourceLimit(support.run("fisher_exact", table=matrix(((40, 60), (60, 40))),
                                             limits=ComputationLimits(max_integer_bits=64)))
        # Heavy tables within the total stop at max_work, in the producer or in the verifier, never with a value.
        for table in (((50000, 0), (0, 50000)), ((15000, 15000), (15000, 15000))):
            with self.subTest(table=table):
                start = time.perf_counter()
                self.assertResourceLimit(support.run("fisher_exact", table=matrix(table)))
                self.assertLess(time.perf_counter() - start, 5)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen."""
        # The evidence is () and the value unique, so a true claim is the reference's value; built here by hand.
        rng = random.Random(20261113)
        tables = [table for table, _ in KNOWN] + [seeded_table(rng) for _ in range(20)]
        for table in tables:
            with self.subTest(table=table):
                self.assertAccepts("fisher_exact", {"table": matrix(table)}, reference(table))
        self.assertAccepts("fisher_exact", {"table": mf.Matrix([[Rational(6, 2), 1], [1, 3]])}, Rational(17, 35))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        table = ((1, 9), (11, 3))
        args = {"table": matrix(table)}
        true = Rational(41, 14858)
        values = (Rational(42, 14858), Rational(1) - true, Rational(0), Rational(-41, 14858), Rational(2), 1, 0.5,
                  True, "41/14858", None, Fraction(41, 14858), (41, 14858),
                  support.forged(Rational, numerator=82, denominator=29716),
                  support.forged(Rational, numerator=-41, denominator=-14858),
                  support.forged(Rational, numerator=41), support.forged(Rational, numerator=41.0, denominator=14858))
        for value in values:
            with self.subTest(value=value):
                self.assertRejects("fisher_exact", args, value)
        self.assertRejects("fisher_exact", {"table": matrix(((2, 2), (2, 2)))}, 1)  # An int 1 is no Rational(1).
        for evidence in ((0,), (true,), [], None, ((),), ("",)):
            with self.subTest(evidence=evidence):
                self.assertRejects("fisher_exact", args, true, evidence)
        self.assertRejects("fisher_exact", args, true, tuple(range(1_000_000)), may_exhaust=True)
        self.assertRejects("Fisher_exact", args, true)
        self.assertRejects("fisher_exact", {"M": matrix(table)}, true)
        self.assertRejects("fisher_exact", dict(args, extra=1), true)
        one = Rational(1)
        for forged_table in (support.forged(mf.Matrix, rows=[(one, Rational(9)), (Rational(11), Rational(3))],
                                            ncols=2),
                             support.forged(mf.Matrix, rows=((one, Rational(9)), (Rational(11), Rational(3))),
                                            ncols=True),
                             support.forged(mf.Matrix, rows=((True, Rational(9)), (Rational(11), Rational(3))),
                                            ncols=2),
                             support.impostor(matrix(table))):
            with self.subTest(table=forged_table):
                self.assertRejects("fisher_exact", {"table": forged_table}, true)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        tables = [table for table, _ in KNOWN] + [((100000, 0), (0, 0)), ((300, 200), (150, 350))]
        results = [support.run("fisher_exact", table=matrix(table)) for table in tables]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for table, result, text in zip(tables, results, texts):
                with self.subTest(table=table):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("fisher_exact", table=matrix(((1, 9), (11, 3))))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))
        # A tampered Result is rejected: verification judges the stored value.
        tampered = support.tampered(results[0], Rational(42, 14858))
        self.assertFalse(mf.verify(tampered).verified)

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX."""
        result = support.run("fisher_exact", table=matrix(((1, 9), (11, 3))))
        decoded = self.roundtrip(result)
        self.assertEqual(decoded.value, Rational(41, 14858))
        self.assertEqual(mf.render_text(result.value), "41/14858")
        self.assertEqual(mf.to_latex(result.value), r"\frac{41}{14858}")
        one = support.run("fisher_exact", table=matrix(((2, 2), (2, 2))))
        self.assertIs(type(self.roundtrip(one).value), Rational)
        self.assertEqual(mf.render_text(one.value), "1")
        self.assertEqual(mf.to_latex(one.value), "1")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261114)
        tables = [table for table, _ in KNOWN] + [seeded_table(rng) for _ in range(40)]
        for table in tables:
            with self.subTest(table=table):
                expected = support.run("fisher_exact", table=matrix(table)).value
                self.assertIsNotNone(expected)
                found = support.lean_value("fisher_exact", table=matrix(table))
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertIs(type(found), Rational)
                self.assertEqual(support.value("fisher_exact", table=matrix(table)), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        result = self.answer(support.call("fisher_exact", {"table": [[1, 9], [11, 3]]}))
        self.assertEqual(result, support.run("fisher_exact", table=matrix(((1, 9), (11, 3)))))
        self.assertEqual(result.value, Rational(41, 14858))
        self.assertTrue(mf.verify(result).verified)
        for table, expected in KNOWN:
            with self.subTest(table=table):
                result = self.answer(support.call_values("fisher_exact", table=matrix(table)))
                self.assertEqual(result, support.run("fisher_exact", table=matrix(table)))
                self.assertEqual(result.value, expected)
                self.assertTrue(mf.verify(result).verified)

    def test_seeded_agreement_with_the_reference(self):
        """produce() agrees with the brute-force reference on at least 200 seeded tables, each value an exact Rational
        in (0, 1] with the evidence (), and the checker accepts every answer."""
        rng = random.Random(20261115)
        tables = [seeded_table(rng) for _ in range(220)]
        tables += [((a, b), (c, d)) for a in range(3) for b in range(3) for c in range(3) for d in range(3)
                   if a + b + c + d]  # Every small table, ties and degenerate margins among them.
        checked = 0
        for table in tables:
            with self.subTest(table=table):
                args = {"table": matrix(table)}
                with computation(fresh=True):
                    value, evidence = family.produce("fisher_exact", args)
                self.assertIs(type(value), Rational)
                self.assertEqual(evidence, ())
                self.assertEqual(value, reference(table))
                self.assertTrue(0 < value <= 1)
                self.assertIs(support.verdict(family, "fisher_exact", args, value, evidence), True)
                checked += 1
        self.assertGreaterEqual(checked, 200)

    def test_a_small_margin_costs_little_whatever_the_total(self):
        """The producer orients the table so that its smallest margin is c1: a table with a small margin and a large
        total costs a few units, as it does the checker. Before, C(99999, 50001) and C(100000, 50001) cost about
        57,000 units, so a request with max_work=1000 failed although the checker needs about 20."""
        tables = (((1, 0), (50000, 49999)), ((50000, 50000), (0, 0)), ((0, 50000), (0, 50000)),
                  ((49999, 50000), (1, 0)), ((0, 49999), (2, 49999)), ((3, 0), (0, 99997)))
        limits = ComputationLimits(max_work=1000)
        for table in tables:
            with self.subTest(table=table):
                with computation(limits, fresh=True):
                    value, evidence = family.produce("fisher_exact", {"table": matrix(table)})
                self.assertEqual(value, reference(table))
                result = support.run("fisher_exact", table=matrix(table), limits=limits)
                self.assertIsNone(result.error)
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)

    def test_symmetries_of_the_table(self):
        """The p-value is the same for the transposed table and for the tables with swapped rows or columns, which
        reverse the support: the producer steps the recurrence from the other end."""
        rng = random.Random(20261116)
        for _ in range(60):
            (a, b), (c, d) = seeded_table(rng)
            value = support.run("fisher_exact", table=matrix(((a, b), (c, d)))).value
            for other in (((a, c), (b, d)), ((c, d), (a, b)), ((b, a), (d, c)), ((d, c), (b, a))):
                with self.subTest(table=((a, b), (c, d)), other=other):
                    self.assertEqual(support.run("fisher_exact", table=matrix(other)).value, value)
