"""The markov family of mathforge.stats: stationary_distribution.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register stats markov

Every expected value comes from the card or from the reference of this module, written here and slow on purpose, never
from the producer: the Markov chain tree theorem in its adjugate form. For L = I - M of rank n - 1, L 1 = 0 and
pi L = 0 give adj(L) = c 1 pi with c != 0, so pi_j is the j-th principal minor of order n - 1 of L divided by the sum of
all of them, each determinant by fractions.Fraction elimination; the sum is 0 exactly when the rank is below n - 1 (no
unique distribution). The producer eliminates the transpose of the row-scaled diag(t) (I - M) instead, and the checker
eliminates a minor modulo a prime. The evidence is compared with a determinant of K = s (I - M) computed here and the
least prime that does not divide it. The family has no deferred case: every valid input has a value, and every other
input is invalid input. tests/v09/stats/test_markov_checker.py tests the checker on its own.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from math import lcm
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.stats import markov as family
from mathforge.contracts import ErrorInfo, Outcome
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational

from .. import support

OPERATION = "stationary_distribution"
FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
STAND_INS = (0.5, True, "1", None)

Rows = Sequence[Sequence[Fraction | int]]


# ---------------------------------------------------------------------------------------------------------------
# The reference and the request builders.

def frac(x: object) -> Fraction:
    """A Fraction of an int, a Fraction or a mathforge Rational (which fractions.Fraction does not take)."""
    if type(x) is Rational:
        return Fraction(x.numerator, x.denominator)
    return Fraction(x)  # type: ignore[arg-type]


def matrix(rows: Rows) -> mf.Matrix:
    """The Matrix of exact rows (Fractions or ints)."""
    return mf.Matrix(tuple(tuple(Rational(frac(x).numerator, frac(x).denominator) for x in row)
                           for row in rows))


def value(pi: Sequence[Fraction | int]) -> tuple[Rational, ...]:
    """A distribution as the value holds it: a tuple of Rationals, never ints."""
    return tuple(Rational(frac(x).numerator, frac(x).denominator) for x in pi)


def determinant(square: list[list[Fraction]]) -> Fraction:
    """The determinant of a rational matrix by Gaussian elimination with fractions; 1 for the 0 x 0 matrix."""
    work = [list(row) for row in square]
    size = len(work)
    result = Fraction(1)
    for k in range(size):
        pivot = next((r for r in range(k, size) if work[r][k]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != k:
            work[k], work[pivot] = work[pivot], work[k]
            result = -result
        result *= work[k][k]
        for r in range(k + 1, size):
            factor = work[r][k] / work[k][k]
            if factor:
                work[r] = [a - factor * b for a, b in zip(work[r], work[k])]
    return result


def stochastic(rows: Rows) -> bool:
    """Whether the rows form a square row-stochastic matrix with at least one state."""
    n = len(rows)
    return n >= 1 and all(len(row) == n and all(frac(x) >= 0 for x in row) and sum(map(frac, row)) == 1
                          for row in rows)


def reference(rows: Rows) -> tuple[Fraction, ...] | None:
    """The unique stationary distribution by the adjugate form of the Markov chain tree theorem, or None for a matrix
    that is not stochastic or a chain without a unique stationary distribution."""
    if not stochastic(rows):
        return None
    n = len(rows)
    laplacian = [[Fraction(i == j) - frac(rows[i][j]) for j in range(n)] for i in range(n)]
    minors = [determinant([[laplacian[i][j] for j in range(n) if j != k] for i in range(n) if i != k])
              for k in range(n)]
    total = sum(minors, Fraction(0))
    if not total:
        return None
    pi = tuple(m / total for m in minors)
    # The definition, independently of the theorem: pi >= 0, pi M = pi and sum pi = 1.
    assert all(p >= 0 for p in pi)
    assert all(sum((pi[i] * frac(rows[i][j]) for i in range(n)), Fraction(0)) == pi[j] for j in range(n))
    assert sum(pi) == 1
    return pi


def scaled_minor(rows: Rows, chosen_rows: Sequence[int], chosen_cols: Sequence[int]) -> int:
    """det K[chosen_rows, chosen_cols] for K = s (I - M), s the least common multiple of all denominators of M."""
    s = lcm(*(frac(x).denominator for row in rows for x in row))
    k = [[s * (Fraction(i == j) - frac(x)) for j, x in enumerate(row)] for i, row in enumerate(rows)]
    d = determinant([[k[r][c] for c in chosen_cols] for r in chosen_rows])
    assert d.denominator == 1
    return int(d)


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def least_prime_not_dividing(d: int) -> int:
    p = 2
    while d % p == 0 or not is_prime(p):
        p += 1
    return p


def random_chain(rng: random.Random, n: int) -> list[list[Fraction]]:
    """A random stochastic matrix of one of four shapes: irreducible dense, irreducible sparse, one closed class with
    transient states, or (for n >= 2, about one draw in eight) two closed classes, which has no unique distribution."""
    shape = rng.randrange(8)
    top = rng.choice((1, 3, 9, 100, 10 ** 6))
    weights = [[rng.randint(0, top) if (shape != 1 or rng.random() < 0.4) else 0 for _ in range(n)]
               for _ in range(n)]
    if shape == 7 and n >= 2:  # Two closed classes: {0, ..., k - 1} and {k, ..., n - 1} never leave themselves.
        k = rng.randint(1, n - 1)
        for i in range(n):
            for j in range(n):
                if (i < k) != (j < k):
                    weights[i][j] = 0
            weights[i][(i + 1) % k if i < k else k + (i - k + 1) % (n - k)] += 1
    elif shape in (5, 6):  # One closed class {0, ..., k - 1}; the other states may move anywhere and reach it.
        k = rng.randint(1, n)
        for i in range(n):
            if i < k:
                for j in range(k, n):
                    weights[i][j] = 0
                weights[i][(i + 1) % k] += 1
            else:
                weights[i][rng.randrange(k)] += 1
    else:  # Irreducible through the cycle 0 -> 1 -> ... -> n - 1 -> 0.
        for i in range(n):
            weights[i][(i + 1) % n] += 1
    return [[Fraction(w, sum(row)) for w in row] for row in weights]


KNOWN: list[tuple[Rows, tuple[Fraction | int, ...]]] = [
    ([[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 4), Fraction(3, 4)]], (Fraction(1, 3), Fraction(2, 3))),
    ([[0, 1], [1, 0]], (Fraction(1, 2), Fraction(1, 2))),
    ([[1]], (1,)),
]
MORE: list[tuple[Rows, tuple[Fraction | int, ...]]] = [
    ([[1, 0], [Fraction(1, 2), Fraction(1, 2)]], (1, 0)),  # State 0 absorbing, state 1 transient.
    ([[0, 1, 0], [0, 0, 1], [1, 0, 0]], (Fraction(1, 3),) * 3),  # Periodic, integer, s = 1.
    ([[Fraction(9, 10), Fraction(1, 10)], [Fraction(1, 2), Fraction(1, 2)]], (Fraction(5, 6), Fraction(1, 6))),
    ([[0, 1, 0, 0], [Fraction(1, 2), 0, Fraction(1, 2), 0], [0, Fraction(1, 2), 0, Fraction(1, 2)], [0, 0, 1, 0]],
     (Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6))),  # A random walk on a path.
    ([[Fraction(1, 2), Fraction(1, 2), 0], [Fraction(1, 2), Fraction(1, 2), 0],
      [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)]], (Fraction(1, 2), Fraction(1, 2), 0)),
]
INVALID: list[tuple[str, mf.Matrix, str]] = [
    ("identity, not unique", matrix([[1, 0], [0, 1]]), "no unique stationary distribution"),
    ("two closed classes", matrix([[Fraction(1, 2), Fraction(1, 2), 0, 0], [Fraction(1, 2), Fraction(1, 2), 0, 0],
                                   [0, 0, Fraction(1, 3), Fraction(2, 3)], [0, 0, 1, 0]]),
     "no unique stationary distribution"),
    ("two absorbing states", matrix([[1, 0, 0], [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)], [0, 0, 1]]),
     "no unique stationary distribution"),
    ("a negative entry", matrix([[2, -1], [0, 1]]), "negative entry"),
    ("a row that sums to 1/2", matrix([[Fraction(1, 2), 0], [0, 1]]), "sums to 1"),
    ("a row that sums to 2", matrix([[1, 1], [0, 1]]), "sums to 1"),
    ("[[2]]", matrix([[2]]), "sums to 1"),
    ("[[0]]", matrix([[0]]), "sums to 1"),
    ("the zero matrix", matrix([[0, 0], [0, 0]]), "sums to 1"),
    ("not square", mf.Matrix(((1, 0),)), "square"),
    ("not square, 3 x 2", mf.Matrix(((1, 0), (0, 1), (1, 0))), "square"),
    ("0 x 0", mf.Matrix(()), "at least one state"),
]


class MarkovTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, pi: Sequence[Fraction | int]) -> None:
        """A completed v6 Result of that distribution, a tuple of Rationals, with its four checks, which mf.verify
        accepts."""
        expected = value(pi)
        self.assertEqual(result.value, expected, result.error)
        self.assertIs(type(result.value), tuple)
        self.assertTrue(all(type(item) is Rational for item in result.value))
        self.assertIs(result.outcome, Outcome.VALUE)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertEvidence(self, rows: Rows, evidence: object) -> None:
        """The producer's evidence: n - 1 strictly increasing row and column indices of a nonzero minor of
        K = s (I - M), and the least prime that does not divide it (both computed here)."""
        n = len(rows)
        self.assertIs(type(evidence), tuple)
        chosen_rows, chosen_cols, prime = evidence  # type: ignore[misc]
        for indices in (chosen_rows, chosen_cols):
            self.assertIs(type(indices), tuple)
            self.assertEqual(len(indices), n - 1)
            self.assertTrue(all(type(i) is int for i in indices))
            self.assertEqual(list(indices), sorted(set(indices)))
            self.assertTrue(all(0 <= i < n for i in indices))
        self.assertIs(type(prime), int)
        d = scaled_minor(rows, chosen_rows, chosen_cols)
        self.assertNotEqual(d, 0)
        self.assertEqual(prime, least_prime_not_dividing(d))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for rows, pi in KNOWN + MORE:
            with self.subTest(rows=rows):
                self.assertEqual(reference(rows), tuple(map(frac, pi)))  # The card's value, by the reference.
                result = support.run(OPERATION, M=matrix(rows))
                self.assertVerified(result, pi)
                self.assertEvidence(rows, result.steps[0].outputs[1])
                self.roundtrip(result)
        # The evidence of the card's chains: s = 4, K = [[2, -2], [-1, 1]], minor (0, 0) = 2, so the prime is 3; the
        # swap chain has K = [[1, -1], [-1, 1]]; [[1]] has empty indices and the prime 2.
        self.assertEqual(support.run(OPERATION, M=matrix(KNOWN[0][0])).steps[0].outputs[1], ((0,), (0,), 3))
        self.assertEqual(support.run(OPERATION, M=matrix(KNOWN[1][0])).steps[0].outputs[1], ((0,), (0,), 2))
        self.assertEqual(support.run(OPERATION, M=matrix([[1]])).steps[0].outputs[1], ((), (), 2))
        # The transient chain: K = [[0, 0], [-1, 1]]; only minors from row 1 are nonzero. Integral entries of pi
        # are Rational(1) and Rational(0), never ints.
        result = support.run(OPERATION, M=matrix(MORE[0][0]))
        self.assertEqual(result.steps[0].outputs[1], ((1,), (0,), 2))
        self.assertEqual([type(item) for item in result.value], [Rational, Rational])
        # Rows whose denominators differ (t = 2, 3, 5; s = 30): the evidence still refers to K = s (I - M).
        rows = [[Fraction(1, 2), Fraction(1, 2), 0], [Fraction(1, 3), 0, Fraction(2, 3)],
                [Fraction(1, 5), Fraction(3, 5), Fraction(1, 5)]]
        result = support.run(OPERATION, M=matrix(rows))
        expected = reference(rows)
        assert expected is not None
        self.assertVerified(result, expected)
        self.assertEvidence(rows, result.steps[0].outputs[1])

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error; so is
        every input outside the domain of the card (not square, 0 x 0, a negative entry, a row that does not sum to 1,
        a chain without a unique stationary distribution). produce() tests the types itself too."""
        rows = KNOWN[0][0]
        containers = ([[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 4), Fraction(3, 4)]], ((1, 0), (0, 1)),
                      [[1]], 3, Rational(1), Fraction(1, 2), {"M": 1}, (value(KNOWN[0][1]),))
        for item in (*STAND_INS, *containers):
            with self.subTest(item=item):
                self.assertInvalidInput(support.run(OPERATION, M=item))
        for label, m, text in INVALID:
            with self.subTest(case=label):
                self.assertIsNone(reference(m.rows))
                result = support.run(OPERATION, M=m)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
        # produce() refuses on its own what operations_v6.run would let through or refuse earlier: other objects
        # than a Matrix, a Matrix subclass, and Matrices forged without the constructor (entries out of lowest terms,
        # an int or a float entry, rows of the wrong length or of the wrong container, a missing field).
        m = matrix(rows)

        class SubMatrix(mf.Matrix):
            pass

        forged_entries = [support.forged(Rational, numerator=2, denominator=4),
                          support.forged(Rational, numerator=-1, denominator=-2),
                          support.forged(Rational, numerator=1, denominator=0),
                          support.forged(Rational, numerator=True, denominator=2),
                          support.forged(Rational, numerator=1, denominator=2.0),
                          support.forged(Rational, numerator=1), 1, 0.5, Fraction(1, 2), True]
        objects: list[tuple[str, object]] = [(repr(item), item) for item in (*STAND_INS, *containers)]
        objects += [("subclass", SubMatrix(m.rows)), ("no ncols", support.forged(mf.Matrix, rows=m.rows)),
                    ("rows as a list", support.forged(mf.Matrix, rows=list(m.rows), ncols=2)),
                    ("a row as a list", support.forged(mf.Matrix, rows=(m.rows[0], list(m.rows[1])), ncols=2)),
                    ("ncols wrong", support.forged(mf.Matrix, rows=m.rows, ncols=3)),
                    ("ragged", support.forged(mf.Matrix, rows=(m.rows[0], m.rows[1][:1]), ncols=2)),
                    ("ncols a bool", support.forged(mf.Matrix, rows=m.rows, ncols=True))]
        for index, entry in enumerate(forged_entries):
            objects.append((f"entry {index}", support.forged(mf.Matrix, rows=(m.rows[0], (entry, m.rows[1][1])),
                                                             ncols=2)))
        for label, item in objects:
            with self.subTest(path="produce", item=label), self.assertRaises(InvalidInput):
                family.produce(OPERATION, {"M": item})
        for label, m_bad, _ in INVALID:
            with self.subTest(path="produce", case=label), self.assertRaises(InvalidInput):
                family.produce(OPERATION, {"M": m_bad})
        for shape in ({"M": m, "extra": 1}, {}, {"A": m}, (("M", m),), None, [m]):
            with self.subTest(path="produce", arguments=shape), self.assertRaises(InvalidInput):
                family.produce(OPERATION, shape)  # type: ignore[arg-type]
        for operation in ("stationary_distributions", "Stationary_distribution", "", None, 7):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, {"M": m})  # type: ignore[arg-type]
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OPERATION, M=matrix([[1, 0], [0, 1]]))
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in M, a 65 x 65 matrix and max_work=1 exceed the budget; so do a common denominator
        s past max_integer_bits, a value integer past it, and an elimination of long entries, charged before it
        runs and refused at once."""
        wide = ComputationLimits(max_integer_bits=70_000)
        with computation(wide, fresh=True):
            huge_rows = [matrix([[Fraction(1, support.HUGE), 1 - Fraction(1, support.HUGE)], [1, 0]]),
                         matrix([[Fraction(support.HUGE - 1, support.HUGE), Fraction(1, support.HUGE)], [1, 0]])]
        for m in huge_rows:
            with self.subTest(case="HUGE"):
                self.assertResourceLimit(support.run(OPERATION, M=m))
                with computation(fresh=True), self.assertRaises(ResourceLimitError):
                    family.produce(OPERATION, {"M": m})
        oversized = support.oversized_matrix()
        self.assertResourceLimit(support.run(OPERATION, M=oversized))
        with computation(fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OPERATION, {"M": oversized})
        # 65 x 65 counts its entries before the shape: a 65 x 66 Matrix is refused for the budget, not as non-square.
        with computation(ComputationLimits(max_matrix_entries=65 * 66), fresh=True):
            rectangle = mf.Matrix(tuple(tuple(int(j == i) for j in range(66)) for i in range(65)))
        self.assertResourceLimit(support.run(OPERATION, M=rectangle))
        for rows, _ in KNOWN + MORE:
            with self.subTest(rows=rows, max_work=1):
                self.assertResourceLimit(support.run(OPERATION, M=matrix(rows), limits=ComputationLimits(max_work=1)))
        # s = 2003 * 2011 has 22 bits, past a max_integer_bits of 20, although every entry has 11.
        coprime = matrix([[Fraction(1, 2003), Fraction(2002, 2003)], [Fraction(1, 2011), Fraction(2010, 2011)]])
        self.assertResourceLimit(support.run(OPERATION, M=coprime, limits=ComputationLimits(max_integer_bits=20)))
        self.assertVerified(support.run(OPERATION, M=coprime), reference(coprime.rows))  # type: ignore[arg-type]
        # A chain of halves (state i moves to i + 1 or back to 0): pi_i = 2**(n - 1 - i) / (2**n - 1), whose
        # denominator has 30 bits for n = 30, past a max_integer_bits of 20 although every entry has 2.
        n = 30
        halves = [[Fraction(1, 2) if j in (0, i + 1) else 0 for j in range(n)] for i in range(n - 1)]
        halves.append([1] + [0] * (n - 1))
        expected = reference(halves)
        assert expected is not None and max(p.denominator for p in expected) == 2 ** n - 1
        self.assertResourceLimit(support.run(OPERATION, M=matrix(halves),
                                             limits=ComputationLimits(max_integer_bits=20)))
        self.assertVerified(support.run(OPERATION, M=matrix(halves)), expected)
        # 40 states with weights of 2,000 bits summing to 2**2000 in each row (s = 2**2000): the elimination's long
        # products are charged before it runs, far past the default max_work, so the request fails at once.
        rng = random.Random(20261113)
        size, total = 40, 2 ** 2000
        long_rows = []
        for _ in range(size):
            weights = [rng.randrange(total // size) for _ in range(size - 1)]
            weights.append(total - sum(weights))
            long_rows.append([Fraction(w, total) for w in weights])
        start = time.perf_counter()
        self.assertResourceLimit(support.run(OPERATION, M=matrix(long_rows)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every nonzero minor of K = s (I - M) with any proved prime that does not divide it, not
        only the producer's least prime at its echelon's minor, and a canonical value forged with object.__new__."""
        for rows, pi in KNOWN + MORE:
            n = len(rows)
            args = {"M": matrix(rows)}
            produced, evidence = family.produce(OPERATION, dict(args))
            self.assertEqual(produced, value(pi))
            claims = 0
            for chosen_rows in combinations(range(n), n - 1):
                for chosen_cols in combinations(range(n), n - 1):
                    d = scaled_minor(rows, chosen_rows, chosen_cols)
                    for prime in (least_prime_not_dividing(d) if d else 2, 10_007, 1_000_000_007, 2 ** 61 - 1):
                        if d % prime:
                            with self.subTest(rows=rows, minor=(chosen_rows, chosen_cols), prime=prime):
                                self.assertAccepts(OPERATION, args, value(pi), (chosen_rows, chosen_cols, prime))
                                claims += 1
            self.assertGreaterEqual(claims, 1)
            forged = tuple(support.forged(Rational, numerator=item.numerator, denominator=item.denominator)
                           for item in produced)
            with self.subTest(rows=rows, case="forged canonical value"):
                self.assertAccepts(OPERATION, args, forged, evidence)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        rows, pi = KNOWN[0]
        args = {"M": matrix(rows)}
        result = support.run(OPERATION, **args)
        good, evidence = result.value, result.steps[0].outputs[1]
        self.assertAccepts(OPERATION, args, good, evidence)
        third, two_thirds = good
        cases: list[tuple[str, object, object, object, object]] = [
            ("swapped", OPERATION, args, (two_thirds, third), evidence),
            ("uniform", OPERATION, args, value((Fraction(1, 2), Fraction(1, 2))), evidence),
            ("sum 2", OPERATION, args, value((Fraction(2, 3), Fraction(4, 3))), evidence),
            ("negative", OPERATION, args, value((Fraction(-1, 3), Fraction(4, 3))), evidence),
            ("too short", OPERATION, args, (third,), evidence),
            ("too long", OPERATION, args, (third, two_thirds, Rational(0)), evidence),
            ("2/6", OPERATION, args, (support.forged(Rational, numerator=2, denominator=6), two_thirds), evidence),
            ("impostor", OPERATION, args, (support.impostor(third), two_thirds), evidence),
            ("a float", OPERATION, args, (1 / 3, two_thirds), evidence),
            ("a str", OPERATION, args, ("1/3", two_thirds), evidence),
            ("a list", OPERATION, args, list(good), evidence),
            ("evidence as a list", OPERATION, args, good, list(evidence)),
            ("evidence None", OPERATION, args, good, None),
            ("evidence ()", OPERATION, args, good, ()),
            ("an extra evidence item", OPERATION, args, good, evidence + (0,)),
            ("a missing evidence item", OPERATION, args, good, evidence[:2]),
            ("prime a bool", OPERATION, args, good, ((0,), (0,), True)),
            ("index a bool", OPERATION, args, good, ((False,), (0,), 3)),
            ("index a str", OPERATION, args, good, (("0",), (0,), 3)),
            ("prime divides the minor", OPERATION, args, good, ((0,), (0,), 2)),
            ("composite modulus", OPERATION, args, good, ((0,), (0,), 9)),
            ("wrong operation", "stationary", args, good, evidence),
            ("operation None", None, args, good, evidence),
            ("M as rows", OPERATION, {"M": args["M"].rows}, good, evidence),
        ]
        for label, operation, claim_args, claim, proof in cases:
            with self.subTest(case=label):
                self.assertRejects(operation, claim_args, claim, proof)  # type: ignore[arg-type]
        # Oversized forgeries: refused, or out of budget, never accepted.
        for label, claim, proof in [("giant prime", good, ((0,), (0,), support.HUGE + 1)),
                                    ("giant index", good, ((support.HUGE,), (0,), 3)),
                                    ("long indices", good, (tuple(range(10 ** 5)), (0,), 3)),
                                    ("long value", tuple(third for _ in range(10 ** 5)), evidence),
                                    ("giant value", (Rational(1, support.HUGE), two_thirds), evidence)]:
            with self.subTest(case=label):
                self.assertRejects(OPERATION, args, claim, proof, may_exhaust=True)
        # mf.verify of a Result whose value was replaced fails.
        for wrong in [(two_thirds, third), value((1, 0))]:
            with self.subTest(tampered=wrong):
                self.assertFalse(mf.verify(support.tampered(result, wrong)).verified)
        self.assertFalse(mf.verify(support.tampered(result, good, ((0,), (0,), 2))).verified)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        rng = random.Random(20261114)
        chains = [rows for rows, _ in KNOWN + MORE]
        while len(chains) < 20:
            candidate = random_chain(rng, rng.randint(2, 6))
            if reference(candidate) is not None:
                chains.append(candidate)
        results = [support.run(OPERATION, M=matrix(rows)) for rows in chains]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for result, text in zip(results, texts):
                with self.subTest(value=result.value):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(OPERATION, M=matrix(KNOWN[0][0]))
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX."""
        rendered = [
            (KNOWN[0][0], "(1/3, 2/3)", r"\left(\frac{1}{3}, \frac{2}{3}\right)"),
            (KNOWN[1][0], "(1/2, 1/2)", r"\left(\frac{1}{2}, \frac{1}{2}\right)"),
            (KNOWN[2][0], "(1)", r"\left(1\right)"),
            (MORE[0][0], "(1, 0)", r"\left(1, 0\right)"),
        ]
        for rows, text, latex in rendered:
            with self.subTest(rows=rows):
                result = support.run(OPERATION, M=matrix(rows))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run(OPERATION, M=matrix(KNOWN[0][0])).value)["data"],
                         {"type": "tuple", "items": [{"type": "rational", "numerator": "1", "denominator": "3"},
                                                     {"type": "rational", "numerator": "2", "denominator": "3"}]})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261115)
        chains = [rows for rows, _ in KNOWN + MORE]
        while len(chains) < 30:
            candidate = random_chain(rng, rng.randint(1, 6))
            if reference(candidate) is not None:
                chains.append(candidate)
        for rows in chains:
            m = matrix(rows)
            with self.subTest(rows=rows):
                expected = support.run(OPERATION, M=m).value
                self.assertIsNotNone(expected)
                found = support.lean_value(OPERATION, M=m)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertIs(type(found), tuple)
                self.assertEqual(support.value(OPERATION, M=m), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        result = self.answer(support.call(OPERATION, {"M": [["1/2", "1/2"], ["1/4", "3/4"]]}))
        self.assertEqual(result, support.run(OPERATION, M=matrix(KNOWN[0][0])))
        self.assertEqual(result.value, value(KNOWN[0][1]))
        self.assertTrue(mf.verify(result).verified)
        # Rationals travel as "p/q" strings, integers from 2**53 on as decimal strings.
        tiny = Fraction(1, 2 ** 70)
        rows = [[1 - tiny, tiny, 0], [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)], [0, 1, 0]]
        expected = reference(rows)
        assert expected is not None
        for label, m in (("chain", matrix(rows)), ("one state", matrix([[1]]))):
            with self.subTest(case=label):
                result = self.answer(support.call_values(OPERATION, M=m))
                self.assertEqual(result, support.run(OPERATION, M=m))
                self.assertTrue(mf.verify(result).verified)
        self.assertEqual(self.answer(support.call_values(OPERATION, M=matrix(rows))).value, value(expected))
        error = self.assertToolError(support.call(OPERATION, {"M": [[1, 0], [0, 1]]}), "invalid_input")
        self.assertIn("no unique stationary distribution", error["message"])
        error = self.assertToolError(support.call(OPERATION, {"M": [["1/2", "1/4"], [0, 1]]}), "invalid_input")
        self.assertIn("sums to 1", error["message"])

    def test_seeded_agreement_with_reference(self):
        """produce() agrees with the tree-theorem reference on at least 200 seeded chains, each answer a tuple of
        Rationals that the checker accepts, with the evidence of the reference's minor and least prime; the chains
        without a unique distribution are refused as invalid input."""
        rng = random.Random(20261116)
        accepted = refused = 0
        transient = 0
        for _ in range(300):  # A fixed number of chains, so that a failing producer cannot keep the loop going.
            rows = random_chain(rng, rng.randint(1, 7))
            args = {"M": matrix(rows)}
            expected = reference(rows)
            with self.subTest(rows=rows):
                if expected is None:
                    with self.assertRaises(InvalidInput):
                        family.produce(OPERATION, dict(args))
                    refused += 1
                    continue
                produced, evidence = family.produce(OPERATION, dict(args))
                self.assertEqual(produced, value(expected))
                self.assertTrue(all(type(item) is Rational for item in produced))
                self.assertEvidence(rows, evidence)
                self.assertIs(support.verdict(family, OPERATION, args, produced, evidence), True)
                accepted += 1
                transient += any(p == 0 for p in expected)
        self.assertGreaterEqual(accepted, 200)
        self.assertGreater(refused, 0)
        self.assertGreater(transient, 0)

    def test_full_size_chains_fit_the_default_budget(self):
        """A dense 64-state chain with rows of different denominators and a 64-state cycle complete and verify under
        the default limits within a second each."""
        rng = random.Random(64)
        n = 64
        dense = []
        for i in range(n):
            weights = [rng.randint(0, 9) for _ in range(n)]
            weights[(i + 1) % n] += 1
            dense.append([Fraction(w, sum(weights)) for w in weights])
        cycle = [[int(j == (i + 1) % n) for j in range(n)] for i in range(n)]
        for label, rows in (("dense", dense), ("cycle", cycle)):
            with self.subTest(case=label):
                start = time.perf_counter()
                result = support.run(OPERATION, M=matrix(rows))
                self.assertLess(time.perf_counter() - start, FAST)
                self.assertIsNotNone(result.value, result.error)
                self.assertTrue(mf.verify(result).verified)
                self.assertEqual(sum(map(frac, result.value)), 1)
        self.assertEqual(support.run(OPERATION, M=matrix(cycle)).value, value((Fraction(1, n),) * n))

    def test_forward_elimination_regressions(self):
        """Review fixes of the elimination. (1) A dense 64-state chain with 36-bit weights: the reduced echelon form
        that produce() first used was charged about 2,230,000 units and refused under the default max_work; forward
        elimination with back substitution costs about 1,070,000 and completes. (2) A doubly stochastic 30-state chain
        (the mean of 7 permutation matrices) under max_integer_bits=64: its value is uniform, but the last pivot of
        the elimination, a minor of order 29 of 7 (I - M), has 77 bits, which the reduced echelon form refused as an
        integer past the limit; the pivots are now bounded by their charged work only."""
        rng = random.Random(5)
        n = 64
        dense = []
        for _ in range(n):
            weights = [rng.getrandbits(36) for _ in range(n)]
            dense.append([Fraction(w, sum(weights)) for w in weights])
        start = time.perf_counter()
        result = support.run(OPERATION, M=matrix(dense))
        self.assertLess(time.perf_counter() - start, 3 * FAST)
        self.assertIsNotNone(result.value, result.error)
        self.assertTrue(mf.verify(result).verified)  # The checker's rank certificate makes the value unique.
        pi = [frac(p) for p in result.value]  # The definition, here: pi >= 0, sum pi = 1 and pi M = pi.
        self.assertTrue(all(p >= 0 for p in pi))
        self.assertEqual(sum(pi), 1)
        for j in range(n):
            self.assertEqual(sum(pi[i] * dense[i][j] for i in range(n)), pi[j])
        rng = random.Random(1)
        n, k = 30, 7
        perms = [rng.sample(range(n), n) for _ in range(k)]
        mixture = [[Fraction(sum(1 for p in perms if p[i] == j), k) for j in range(n)] for i in range(n)]
        self.assertTrue(stochastic(mixture) and all(sum(row[j] for row in mixture) == 1 for j in range(n)))
        result = support.run(OPERATION, M=matrix(mixture), limits=ComputationLimits(max_integer_bits=64))
        self.assertEqual(result.value, value((Fraction(1, n),) * n), result.error)
        self.assertEvidence(mixture, result.steps[0].outputs[1])

    def test_family_module(self):
        """The spec of the card, no records, the checker path, and no import of a checker or of a generated module."""
        spec = family.SPECS[0]
        self.assertEqual((spec.name, spec.area, spec.version, spec.returns), (OPERATION, "stats", 6,
                                                                             "tuple[Rational, ...]"))
        self.assertEqual(spec.summary, "Find the unique stationary distribution of a finite Markov chain with a "
                                       "rational transition matrix.")
        self.assertEqual([(a.name, a.annotation, a.default, a.kind) for a in spec.arguments],
                         [("M", "Matrix", None, "matrix")])
        self.assertEqual(family.TYPES, {})
        self.assertEqual(family.CHECKER, "mathforge.certify.stats.markov")
        with open(family.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        package = ["mathforge", "areas", "stats"]  # A relative import of level k starts k - 1 packages up.
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                imported.add(".".join(base + ([node.module] if node.module else [])))
        self.assertIn("mathforge.trusted", imported)  # from ...trusted import upoly, inside the helpers.
        for name in imported:
            for forbidden in ("mathforge.certify", "mathforge.lean", "mathforge.mcp", "mathforge._generated",
                              "mathforge.stats"):
                self.assertFalse(name == forbidden or name.startswith(forbidden + "."), name)
