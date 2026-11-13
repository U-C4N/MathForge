"""The checker of the exact_tests family alone: mathforge.certify.stats.exact_tests (fisher_exact).

Every accepted claim is computed here, by a brute-force reference with binomial coefficients (the checker forms none),
never by the producer. Every corrupted claim must answer False, never raise; a ResourceLimitError is expected only
where docs/v09/cards/m7-extended.md and the common rules put a resource limit (a total above 100,000, a request
integer over max_integer_bits, a matrix over max_matrix_entries, work past max_work).

Run it by name with python -m unittest tests.v09.stats.test_exact_tests_checker.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
import random
import time

import mathforge as mf
from mathforge.areas.stats import exact_tests as family
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.model import Rational

from .. import support

KNOWN = (  # docs/v09/cards/m7-extended.md
    (((1, 9), (11, 3)), Rational(41, 14858)),
    (((3, 1), (1, 3)), Rational(17, 35)),
    (((0, 5), (5, 0)), Rational(1, 126)),
    (((2, 2), (2, 2)), Rational(1)),
    (((8, 2), (1, 5)), Rational(5, 143)),
)


def masses(table: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, dict[int, int]]:
    """(C(n, c1), {k: C(r1, k) C(r2, c1 - k)}) over the support of the top-left count: the hypergeometric masses
    times C(n, c1), by binomial coefficients."""
    (a, b), (c, d) = table
    r1, r2, c1 = a + b, c + d, a + c
    weights = {k: comb(r1, k) * comb(r2, c1 - k) for k in range(max(0, c1 - r2), min(r1, c1) + 1)}
    return comb(r1 + r2, c1), weights


def reference(table: tuple[tuple[int, int], tuple[int, int]]) -> Rational:
    """The two-sided Fisher p-value by brute force: the masses not above the observed one, summed."""
    total, weights = masses(table)
    observed = weights[table[0][0]]
    value = Fraction(sum(weight for weight in weights.values() if weight <= observed), total)
    return Rational(value.numerator, value.denominator)


def rational(value: Fraction) -> Rational:
    return Rational(value.numerator, value.denominator)


def args(table: object) -> dict[str, object]:
    """The request's arguments for a table given as rows of ints (or any object, passed as it is)."""
    if type(table) is tuple:
        return {"table": mf.Matrix(table)}
    return {"table": table}


def forged_matrix(rows: object, ncols: object = 2) -> object:
    return support.forged(mf.Matrix, rows=rows, ncols=ncols)


def entries(*values: int) -> tuple[tuple[Rational, Rational], tuple[Rational, Rational]]:
    a, b, c, d = (Rational(value) for value in values)
    return (a, b), (c, d)


class ExactTestsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def accepts(self, table: tuple[tuple[int, int], tuple[int, int]], value: object = None) -> None:
        self.assertAccepts("fisher_exact", args(table), reference(table) if value is None else value)

    def rejects(self, table: object, value: object, evidence: object = (), **options: object) -> None:
        self.assertRejects("fisher_exact", args(table), value, evidence, **options)

    # --- true claims ---------------------------------------------------------------------------------------------

    def test_accepts_the_known_values(self):
        for table, value in KNOWN:
            with self.subTest(table=table):
                self.assertEqual(reference(table), value)
                self.accepts(table, value)

    def test_accepts_the_brute_force_reference_on_seeded_tables(self):
        generator = random.Random(20261113)
        for bound, count in ((6, 60), (30, 60), (400, 15)):
            for _ in range(count):
                table = tuple(tuple(generator.randint(0, bound) for _ in range(2)) for _ in range(2))
                if not any(map(any, table)):
                    continue
                with self.subTest(table=table):
                    self.accepts(table)

    def test_accepts_degenerate_margins(self):
        """A zero row or column leaves one table with these margins: the support is one point, the p-value 1."""
        for table in (((5, 0), (0, 0)), ((0, 0), (0, 7)), ((0, 3), (0, 4)), ((0, 0), (2, 5)), ((1, 0), (0, 0)),
                      ((4, 0), (3, 0)), ((0, 0), (0, 1)), ((100000, 0), (0, 0))):
            with self.subTest(table=table):
                self.assertEqual(reference(table), Rational(1))
                self.accepts(table, Rational(1))

    def test_accepts_observed_counts_at_both_ends_of_the_support(self):
        for table in (((0, 5), (5, 0)), ((5, 0), (0, 5)), ((0, 4), (6, 1)), ((3, 0), (2, 9)), ((0, 9), (1, 0)),
                      ((7, 0), (1, 2))):
            with self.subTest(table=table):
                self.accepts(table)

    def test_ties_with_the_observed_mass_are_included(self):
        """Symmetric margins give P(X = k) = P(X = c1 - k): the mirrored mass counts, decided exactly."""
        total, weights = masses(((1, 3), (3, 1)))
        self.assertEqual(weights[1], weights[3])
        for table in (((1, 3), (3, 1)), ((3, 1), (1, 3))):
            with self.subTest(table=table):
                self.accepts(table, Rational(17, 35))
                strict = Fraction(sum(w for w in weights.values() if w < weights[table[0][0]]), total)
                self.rejects(table, rational(strict))  # 1/35: the tie left out.
        self.accepts(((2, 5), (5, 2)))
        self.accepts(((5, 5), (5, 5)), Rational(1))  # The observed count is the mode.

    def test_accepts_a_large_table_and_a_matrix_of_rationals(self):
        self.accepts(((300, 200), (150, 350)))
        self.accepts(((5000, 5000), (5000, 5000)), Rational(1))  # A mode: about 330,000 units, well in the budget.
        rows = ((Rational(1), Rational(9)), (Rational(11), Rational(3)))
        self.assertAccepts("fisher_exact", {"table": mf.Matrix(rows)}, Rational(41, 14858))

    def test_accepts_under_an_outer_budget(self):
        self.assertAccepts("fisher_exact", args(((1, 9), (11, 3))), Rational(41, 14858),
                           limits=ComputationLimits(max_work=10_000))

    # --- corrupted claims ----------------------------------------------------------------------------------------

    def test_rejects_wrong_values(self):
        table = ((1, 9), (11, 3))
        total, weights = masses(table)
        observed = weights[1]
        one_sided = Fraction(sum(weights[k] for k in weights if k <= 1), total)
        mid_p = Fraction(sum(w for w in weights.values() if w < observed), total) + Fraction(observed, 2 * total)
        for value in (Rational(41, 14857), Rational(41, 14859), Rational(14817, 14858), Rational(observed, total),
                      rational(one_sided), rational(mid_p), Rational(0), Rational(-41, 14858), Rational(2),
                      Rational(1), Rational(1, 10 ** 30)):
            with self.subTest(value=value):
                self.rejects(table, value)
        for table, value in KNOWN:  # Each known value belongs to its own table only.
            for other, _ in KNOWN:
                if reference(other) != value:
                    with self.subTest(table=other, value=value):
                        self.rejects(other, value)

    def test_rejects_non_canonical_values(self):
        table = ((1, 9), (11, 3))
        for value in (support.forged(Rational, numerator=82, denominator=29716),
                      support.forged(Rational, numerator=-41, denominator=-14858),
                      support.forged(Rational, numerator=41, denominator=0),
                      support.forged(Rational, numerator=41),
                      support.forged(Rational, denominator=14858),
                      Fraction(41, 14858), (41, 14858), mf.Matrix(((41, 14858),))):
            with self.subTest(value=value):
                self.rejects(table, value)
        self.rejects(((2, 2), (2, 2)), 1)  # An int: the value is Rational(1).
        self.rejects(((2, 2), (2, 2)), support.forged(Rational, numerator=2, denominator=2))

        class Subclass(Rational):
            __slots__ = ()

        self.rejects(table, Subclass(41, 14858))

    def test_rejects_impostor_classes(self):
        table = ((1, 9), (11, 3))
        self.rejects(table, support.impostor(Rational(41, 14858)))
        self.rejects(support.impostor(mf.Matrix(table)), Rational(41, 14858))

    def test_rejects_forged_tables(self):
        value = Rational(41, 14858)
        cases = {
            "int entries": ((1, 9), (11, 3)),
            "an entry 4/2": ((Rational(1), Rational(9)), (Rational(11), support.forged(Rational, numerator=6,
                                                                                         denominator=2))),
            "a fractional entry": ((Rational(1), Rational(9)), (Rational(11), Rational(7, 2))),
            "a negative entry": entries(1, 9, 11, -3),
            "a negative denominator": ((Rational(1), Rational(9)),
                                       (Rational(11), support.forged(Rational, numerator=-3, denominator=-1))),
            "a zero denominator": ((Rational(1), Rational(9)),
                                   (Rational(11), support.forged(Rational, numerator=3, denominator=0))),
            "an entry without its denominator": ((Rational(1), Rational(9)),
                                                 (Rational(11), support.forged(Rational, numerator=3))),
            "a short row": ((Rational(1), Rational(9)), (Rational(11),)),
            "a long row": ((Rational(1), Rational(9), Rational(0)), (Rational(11), Rational(3), Rational(0))),
            "one row": ((Rational(1), Rational(9)),),
            "three rows": entries(1, 9, 11, 3) + ((Rational(0), Rational(0)),),
            "a row that is no tuple": ((Rational(1), Rational(9)), 14),
        }
        for label, rows in cases.items():
            with self.subTest(label):
                self.rejects(forged_matrix(rows), value)
        self.rejects(forged_matrix(entries(1, 9, 11, 3), 3), value)
        self.rejects(forged_matrix(entries(1, 9, 11, 3), -2), value)
        self.rejects(support.forged(mf.Matrix, rows=entries(1, 9, 11, 3)), value)  # No ncols.
        self.rejects(support.forged(mf.Matrix, ncols=2), value)  # No rows.
        self.assertAccepts("fisher_exact", {"table": forged_matrix(entries(1, 9, 11, 3))}, value)  # The control.

    def test_rejects_invalid_tables(self):
        for table in (((0, 0), (0, 0)), ((1, 2),), ((1,), (2,)), ((1, 2, 3), (4, 5, 6)),
                      ((1, 2), (3, 4), (5, 6)), ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            with self.subTest(table=table):
                self.rejects(table, Rational(1))
        self.rejects(mf.Matrix((), ncols=0), Rational(1))
        self.rejects(((1, 9), (11, Rational(-3))), Rational(41, 14858))
        self.rejects(((1, 9), (11, Rational(1, 3))), Rational(1))

    def test_rejects_bool_float_and_str(self):
        value = Rational(41, 14858)
        for bad in (True, 3.0, "3", None):
            with self.subTest(entry=bad):
                self.rejects(forged_matrix(((Rational(1), Rational(9)), (Rational(11), bad))), value)
                self.rejects(forged_matrix(((Rational(1), Rational(9)),
                                            (Rational(11), support.forged(Rational, numerator=bad, denominator=1)))),
                             value)
                self.rejects(forged_matrix(((Rational(1), Rational(9)),
                                            (Rational(11), support.forged(Rational, numerator=3, denominator=bad)))),
                             value)
        self.rejects(forged_matrix(entries(1, 9, 11, 3), True), value)
        self.rejects(forged_matrix(entries(1, 9, 11, 3), 2.0), value)
        for bad in (True, 1.0, "1", 0.5, None):
            with self.subTest(value=bad):
                self.rejects(((2, 2), (2, 2)), bad)
        for numerator, denominator in ((True, 1), (1.0, 1), ("1", 1), (1, True), (41, 14858.0), (41, "14858")):
            with self.subTest(numerator=numerator, denominator=denominator):
                self.rejects(((2, 2), (2, 2)), support.forged(Rational, numerator=numerator, denominator=denominator))
        self.rejects(((1, 9), (11, 3)), support.forged(Rational, numerator=41, denominator=14858.0))

    def test_rejects_lists_for_tuples(self):
        value = Rational(41, 14858)
        self.rejects(forged_matrix([(Rational(1), Rational(9)), (Rational(11), Rational(3))]), value)
        self.rejects(forged_matrix(((Rational(1), Rational(9)), [Rational(11), Rational(3)])), value)
        self.rejects([[1, 9], [11, 3]], value)  # Rows of ints, not a Matrix.
        self.rejects(((1, 9), (11, 3)), value, [])

    def test_rejects_any_evidence(self):
        """The evidence is (): missing, extra, wrong-shape and internally consistent forged evidence answer False."""
        table = ((1, 9), (11, 3))
        total, weights = masses(table)
        consistent = (total, tuple(sorted(weights.items())))  # The true masses: still not the card's layout.
        for evidence in (None, [], (0,), ((),), ("fisher_exact",), (Rational(41, 14858),), consistent,
                         (Rational(41, 14858), Rational(14817, 14858))):
            with self.subTest(evidence=evidence):
                self.rejects(table, Rational(41, 14858), evidence)

    def test_rejects_a_wrong_operation_or_wrong_arguments(self):
        table, value = ((1, 9), (11, 3)), Rational(41, 14858)
        for operation in ("fisher", "Fisher_exact", "fisher_exact ", "correlation", None, b"fisher_exact", 1):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args(table), value)
        matrix = mf.Matrix(table)
        for arguments in ({}, {"M": matrix}, {"table": matrix, "alternative": "two-sided"}, {1: matrix},
                          [("table", matrix)], None, ({"table": matrix},)):
            with self.subTest(arguments=arguments):
                self.assertRejects("fisher_exact", arguments, value)

    def test_oversized_claims_fail_fast(self):
        table = ((1, 9), (11, 3))
        for value in (Rational(1, support.HUGE + 1), Rational(support.HUGE - 1, support.HUGE),
                      Rational(support.HUGE, support.HUGE + 1)):
            start = time.perf_counter()
            self.rejects(table, value)
            self.assertLess(time.perf_counter() - start, 2.0)
        start = time.perf_counter()
        self.rejects(table, Rational(41, 14858), (0,) * 1_000_000)
        self.rejects(table, Rational(41, 14858), ((Rational(support.HUGE),) * 1000,))
        self.assertLess(time.perf_counter() - start, 2.0)

    def test_resource_limits(self):
        """The card's resource limits raise ResourceLimitError: an unverified claim, never a verified one."""
        value = Rational(1)
        for table in (mf.Matrix(((100_001, 0), (0, 0))), mf.Matrix(((50_000, 50_000), (1, 0))),
                      forged_matrix(((Rational(support.HUGE), Rational(0)), (Rational(0), Rational(1)))),
                      support.oversized_matrix()):
            with self.subTest(table=table.shape):
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, "fisher_exact", {"table": table}, value)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "fisher_exact", args(((1, 9), (11, 3))), Rational(41, 14858),
                            limits=ComputationLimits(max_work=1))

    def test_heavy_tables_stop_within_the_budget(self):
        """Tables of total 100,000 whose masses are too long for the default budget stop in a few seconds; a true
        claim is then unverified (ResourceLimitError) and a false one is never accepted."""
        for table, value in ((((25_000, 25_000), (25_000, 25_000)), Rational(1)),
                             (((50_000, 0), (0, 50_000)), Rational(1, 2)),
                             (((1, 49_999), (49_999, 1)), Rational(1, 3))):
            with self.subTest(table=table):
                start = time.perf_counter()
                try:
                    answer = support.verdict(family, "fisher_exact", args(table), value)
                except ResourceLimitError:
                    answer = None
                self.assertLess(time.perf_counter() - start, 5.0)
                if value != Rational(1):
                    self.assertIsNot(answer, True)
