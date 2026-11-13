"""The independent checker of mathforge.areas.stats.markov (mathforge.certify.stats.markov), tested on its own before
the family's producer exists.

Every true claim is computed here, never by the producer: the stationary distribution by Gauss-Jordan elimination
with fractions.Fraction on the system pi (M - I) = 0, sum pi = 1 (checked against pi M = pi afterwards), and the
evidence (rows, cols, prime) by a search over the minors of order n - 1 of K = s (I - M), each determinant by the
Leibniz formula for small orders and by Fraction elimination for larger ones, and the least prime that does not
divide it by trial division. The checker answers False to every corrupted claim, never with an exception, and refuses
an oversized forgery within FAST seconds under the default budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, permutations
from math import lcm, prod
import random
import time
from typing import Any

import mathforge as mf
from mathforge.areas.stats import markov as family
from mathforge.certify.stats import markov as checker
from mathforge.limits import ComputationLimits
from mathforge.model import Rational

from .. import support

OPERATION = "stationary_distribution"
FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
PSI_13 = 3_317_044_064_679_887_385_961_981  # prime_proof needs a Pocklington certificate from here on.

Rows = Sequence[Sequence[Fraction | int]]


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def matrix(rows: Rows) -> mf.Matrix:
    """The Matrix of exact rows (Fractions or ints)."""
    return mf.Matrix(tuple(tuple(Rational(Fraction(x).numerator, Fraction(x).denominator) for x in row)
                           for row in rows))


def value(pi: Sequence[Fraction | int]) -> tuple[Rational, ...]:
    """A distribution as the value holds it: a tuple of Rationals, never ints."""
    return tuple(Rational(Fraction(x).numerator, Fraction(x).denominator) for x in pi)


def is_prime(n: int) -> bool:
    """Trial division, for the small primes of the tests."""
    return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def solve(rows: Rows) -> tuple[Fraction, ...] | None:
    """The unique pi with pi (M - I) = 0 and sum pi = 1 by Gauss-Jordan elimination over Q, or None when the system
    has no unique solution. The n + 1 equations in the n unknowns pi_i: column j of M - I, then the row of ones."""
    n = len(rows)
    system = [[Fraction(rows[i][j]) - (i == j) for i in range(n)] + [Fraction(0)] for j in range(n)]
    system.append([Fraction(1)] * n + [Fraction(1)])
    rank = 0
    for column in range(n):
        pivot = next((r for r in range(rank, n + 1) if system[r][column]), None)
        if pivot is None:
            return None
        system[rank], system[pivot] = system[pivot], system[rank]
        lead = system[rank][column]
        system[rank] = [x / lead for x in system[rank]]
        for r in range(n + 1):
            if r != rank and system[r][column]:
                factor = system[r][column]
                system[r] = [a - factor * b for a, b in zip(system[r], system[rank])]
        rank += 1
    if any(system[r][n] for r in range(rank, n + 1)):
        return None
    pi = tuple(system[i][n] for i in range(n))
    # The definition, independently of the elimination: pi M = pi and sum pi = 1.
    assert all(sum((pi[i] * Fraction(rows[i][j]) for i in range(n)), Fraction(0)) == pi[j] for j in range(n))
    assert sum(pi, Fraction(0)) == 1
    return pi


def kernel_matrix(rows: Rows) -> list[list[int]]:
    """K = s (I - M), s the least common multiple of the denominators of M."""
    s = lcm(*(Fraction(x).denominator for row in rows for x in row))
    return [[int(s * ((i == j) - Fraction(x))) for j, x in enumerate(row)] for i, row in enumerate(rows)]


def determinant(square: list[list[int]]) -> int:
    """The determinant of an integer matrix: the Leibniz formula up to order 6, Fraction elimination above."""
    size = len(square)
    if size <= 6:
        total = 0
        for perm in permutations(range(size)):
            inversions = sum(1 for a in range(size) for b in range(a + 1, size) if perm[a] > perm[b])
            total += (-1) ** inversions * prod(square[i][perm[i]] for i in range(size))
        return total
    work = [[Fraction(x) for x in row] for row in square]
    result = Fraction(1)
    for k in range(size):
        pivot = next((r for r in range(k, size) if work[r][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            work[k], work[pivot] = work[pivot], work[k]
            result = -result
        result *= work[k][k]
        for r in range(k + 1, size):
            factor = work[r][k] / work[k][k]
            work[r] = [a - factor * b for a, b in zip(work[r], work[k])]
    assert result.denominator == 1
    return int(result)


def minor(rows: Rows, chosen_rows: tuple[int, ...], chosen_cols: tuple[int, ...]) -> int:
    """det K[chosen_rows, chosen_cols]."""
    k = kernel_matrix(rows)
    return determinant([[k[r][c] for c in chosen_cols] for r in chosen_rows])


def least_prime_not_dividing(d: int) -> int:
    """The least prime that does not divide the nonzero integer d."""
    p = 2
    while d % p == 0 or not is_prime(p):
        p += 1
    return p


def all_evidence(rows: Rows) -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
    """Every (rows, cols, prime) with a nonzero minor, each with its least admissible prime."""
    n = len(rows)
    found = []
    for chosen_rows in combinations(range(n), n - 1):
        for chosen_cols in combinations(range(n), n - 1):
            d = minor(rows, chosen_rows, chosen_cols)
            if d:
                found.append((chosen_rows, chosen_cols, least_prime_not_dividing(d)))
    return found


def evidence(rows: Rows) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """One evidence: the principal minors (state k left out) first, which the matrix-tree theorem makes nonzero for an
    irreducible chain, then every other minor."""
    n = len(rows)
    for k in range(n):
        kept = tuple(i for i in range(n) if i != k)
        d = minor(rows, kept, kept)
        if d:
            return kept, kept, least_prime_not_dividing(d)
    return all_evidence(rows)[0]


def random_chain(rng: random.Random, n: int, top: int, sparse: bool = False) -> list[list[Fraction]]:
    """A random row-stochastic matrix, each row integer weights in 0 .. top divided by their sum, irreducible through
    the cycle 0 -> 1 -> ... -> n-1 -> 0 that every row keeps."""
    rows = []
    for i in range(n):
        weights = [0 if sparse and rng.random() < 0.6 else rng.randint(0, top) for _ in range(n)]
        weights[(i + 1) % n] += 1
        total = sum(weights)
        rows.append([Fraction(w, total) for w in weights])
    return rows


KNOWN: list[tuple[Rows, tuple[Fraction | int, ...]]] = [
    ([[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 4), Fraction(3, 4)]], (Fraction(1, 3), Fraction(2, 3))),
    ([[0, 1], [1, 0]], (Fraction(1, 2), Fraction(1, 2))),
    ([[1]], (1,)),
]


class MarkovCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def accepts(self, rows: Rows, pi: Sequence[Fraction | int], proof: object, **kwargs: Any) -> None:
        self.assertAccepts(OPERATION, {"M": matrix(rows)}, value(pi), proof, **kwargs)

    def rejects(self, args: object, claim: object, proof: object, operation: object = OPERATION,
                **kwargs: Any) -> None:
        self.assertRejects(operation, args, claim, proof, **kwargs)  # type: ignore[arg-type]

    def rejects_fast(self, args: object, claim: object, proof: object, operation: object = OPERATION) -> None:
        start = time.perf_counter()
        self.rejects(args, claim, proof, operation)
        self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # True claims

    def test_known_values(self):
        """The card's known values, with the evidence computed here; [[1]] with empty indices."""
        for rows, pi in KNOWN:
            with self.subTest(rows=rows):
                self.assertEqual(solve(rows), tuple(Fraction(x) for x in pi))
                self.accepts(rows, pi, evidence(rows))
        self.accepts([[1]], (1,), ((), (), 2))
        # The first known value: s = 4, K = [[2, -2], [-1, 1]], every 1 x 1 minor nonzero.
        self.assertEqual(kernel_matrix(KNOWN[0][0]), [[2, -2], [-1, 1]])
        self.accepts(KNOWN[0][0], KNOWN[0][1], ((0,), (0,), 3))
        self.accepts(KNOWN[0][0], KNOWN[0][1], ((1,), (0,), 2))  # det -1: every prime passes.

    def test_every_evidence_and_prime_is_accepted(self):
        """Every nonzero minor with its least admissible prime and with other primes up to the bound of prime_proof,
        not only the evidence a producer would choose."""
        chains = [KNOWN[0][0], KNOWN[1][0],
                  [[Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)], [Fraction(1, 2), 0, Fraction(1, 2)], [0, 1, 0]]]
        for rows in chains:
            pi = solve(rows)
            assert pi is not None
            found = all_evidence(rows)
            self.assertTrue(found)
            for chosen_rows, chosen_cols, prime in found:
                d = minor(rows, chosen_rows, chosen_cols)
                for p in (prime, 10_007, 2 ** 61 - 1, 1_000_000_007):
                    if d % p:
                        with self.subTest(rows=rows, minor=(chosen_rows, chosen_cols), prime=p):
                            self.accepts(rows, pi, (chosen_rows, chosen_cols, p))
        for p in (2, 3, 97, 9_973, 10_007, 2 ** 61 - 1, 2 ** 64 - 59):
            with self.subTest(n=1, prime=p):
                self.accepts([[1]], (1,), ((), (), p))

    def test_special_cases(self):
        """Transient states (zeros in pi), a periodic chain, an integer matrix, one closed class among others."""
        cases: list[tuple[Rows, tuple[Fraction | int, ...]]] = [
            ([[1, 0], [Fraction(1, 2), Fraction(1, 2)]], (1, 0)),  # State 0 absorbing, state 1 transient.
            ([[0, 1, 0], [0, 0, 1], [1, 0, 0]], (Fraction(1, 3),) * 3),  # Periodic, integer, s = 1.
            ([[Fraction(1, 2), Fraction(1, 2), 0], [Fraction(1, 2), Fraction(1, 2), 0],
              [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)]], (Fraction(1, 2), Fraction(1, 2), 0)),
            ([[Fraction(9, 10), Fraction(1, 10)], [Fraction(1, 2), Fraction(1, 2)]], (Fraction(5, 6), Fraction(1, 6))),
            ([[0, 1, 0, 0], [Fraction(1, 2), 0, Fraction(1, 2), 0], [0, Fraction(1, 2), 0, Fraction(1, 2)],
              [0, 0, 1, 0]], (Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6))),  # Random walk.
        ]
        for rows, pi in cases:
            with self.subTest(rows=rows):
                self.assertEqual(solve(rows), tuple(Fraction(x) for x in pi))
                self.accepts(rows, pi, evidence(rows))
        # The transient example: s = 2, K = [[0, 0], [-1, 1]]; only the minors from row 1 are nonzero.
        absorbing = cases[0][0]
        self.assertEqual(kernel_matrix(absorbing), [[0, 0], [-1, 1]])
        self.accepts(absorbing, (1, 0), ((1,), (0,), 2))
        self.accepts(absorbing, (1, 0), ((1,), (1,), 2))

    def test_seeded_random_chains(self):
        """Random irreducible chains of 2 to 7 states, dense and sparse, with the reference's pi and evidence."""
        rng = random.Random(20261113)
        for trial in range(30):
            n = rng.randint(2, 7)
            rows = random_chain(rng, n, rng.randint(1, 12), sparse=bool(trial % 2))
            pi = solve(rows)
            assert pi is not None
            with self.subTest(trial=trial, n=n):
                self.accepts(rows, pi, evidence(rows))

    def test_largest_chains_fit_the_default_budget(self):
        """A 64-state cycle (integer, s = 1) and a dense 20-state rational chain verify under the default limits."""
        n = 64
        cycle = [[int(j == (i + 1) % n) for j in range(n)] for i in range(n)]
        kept = tuple(range(n - 1))
        # K = I - P with the last state left out is upper bidiagonal with ones on its diagonal: det 1.
        start = time.perf_counter()
        self.accepts(cycle, (Fraction(1, n),) * n, (kept, kept, 2))
        self.assertLess(time.perf_counter() - start, FAST)
        rng = random.Random(64)
        size = 20
        dense = [[Fraction(rng.randint(1, 9), 1) for _ in range(size)] for _ in range(size)]
        dense = [[x / sum(row) for x in row] for row in dense]
        pi = solve(dense)
        assert pi is not None
        kept = tuple(range(size - 1))
        d = determinant([[kernel_matrix(dense)[r][c] for c in kept] for r in kept])
        start = time.perf_counter()
        self.accepts(dense, pi, (kept, kept, least_prime_not_dividing(d)))
        self.assertLess(time.perf_counter() - start, FAST)
        # The same claim within a tight work budget is never verified.
        self.rejects({"M": matrix(dense)}, value(pi), (kept, kept, least_prime_not_dividing(d)),
                     limits=ComputationLimits(max_work=50), may_exhaust=True)

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims

    def test_wrong_values(self):
        """A wrong distribution, a stationary vector that does not sum to 1, a wrong length."""
        rows, pi = KNOWN[0]
        args, proof = {"M": matrix(rows)}, evidence(rows)
        for claim in [value((Fraction(2, 3), Fraction(1, 3))), value((Fraction(1, 2), Fraction(1, 2))),
                      value((Fraction(2, 3), Fraction(4, 3))), value((0, 0)), value((1, 0)),
                      value((Fraction(1, 3), Fraction(2, 3), 0)), value((Fraction(1, 3),)), (),
                      value((Fraction(-1, 3), Fraction(4, 3)))]:
            with self.subTest(claim=claim):
                self.rejects(args, claim, proof)
        self.accepts(rows, pi, proof)

    def test_chains_without_a_unique_distribution(self):
        """Two closed classes: every stationary claim is refused, as no minor of order n - 1 is nonzero; a negative
        vector in the kernel is refused as well."""
        identity = [[1, 0], [0, 1]]
        self.assertEqual(all_evidence(identity), [])
        blocks = [[Fraction(1, 2), Fraction(1, 2), 0, 0], [Fraction(1, 2), Fraction(1, 2), 0, 0],
                  [0, 0, Fraction(1, 3), Fraction(2, 3)], [0, 0, Fraction(1, 3), Fraction(2, 3)]]
        self.assertEqual(all_evidence(blocks), [])
        self.assertIsNone(solve(identity))
        self.assertIsNone(solve(blocks))
        for rows, claims in [(identity, [(Fraction(1, 2), Fraction(1, 2)), (1, 0), (0, 1), (2, -1)]),
                             (blocks, [(Fraction(1, 4), Fraction(1, 4), Fraction(1, 6), Fraction(1, 3)),
                                       (Fraction(1, 2), Fraction(1, 2), 0, 0)])]:
            n = len(rows)
            for claim in claims:
                for chosen_rows in combinations(range(n), n - 1):
                    for chosen_cols in combinations(range(n), n - 1):
                        for p in (2, 3, 5, 7):
                            with self.subTest(rows=rows, claim=claim, minor=(chosen_rows, chosen_cols), prime=p):
                                self.rejects({"M": matrix(rows)}, value(claim), (chosen_rows, chosen_cols, p))

    def test_matrices_that_are_not_stochastic(self):
        """A negative entry or a row that does not sum to 1, with a claim that passes every other step."""
        # pi = (0, 1): pi M = pi, sum 1; K = [[-1, 1], [0, 0]], minor (0, 0) = -1. Only the entry -1 is wrong.
        self.rejects({"M": matrix([[2, -1], [0, 1]])}, value((0, 1)), ((0,), (0,), 2))
        # pi = (0, 1): pi M = pi, sum 1; s = 2, K = [[1, 0], [0, 0]], minor (0, 0) = 1. Row 0 sums to 1/2.
        self.rejects({"M": matrix([[Fraction(1, 2), 0], [0, 1]])}, value((0, 1)), ((0,), (0,), 2))
        self.rejects({"M": matrix([[2]])}, value((1,)), ((), (), 2))
        self.rejects({"M": matrix([[0]])}, value((1,)), ((), (), 2))
        self.rejects({"M": mf.Matrix(())}, (), ((), (), 2))  # 0 x 0: a chain without states.
        self.rejects({"M": mf.Matrix(((1, 0),))}, value((1,)), ((), (), 2))  # Not square.
        self.rejects({"M": mf.Matrix(((1, 0),))}, value((1, 0)), ((0,), (0,), 2))

    def test_non_canonical_values(self):
        """An int for a Rational, a Rational out of lowest terms or with a negative denominator, forged with
        object.__new__, and an impostor class of the same name and fields."""
        rows, pi = KNOWN[0]
        args, proof = {"M": matrix(rows)}, evidence(rows)
        third, two_thirds = value(pi)
        forged_cases = [
            (support.forged(Rational, numerator=2, denominator=6), two_thirds),
            (support.forged(Rational, numerator=-1, denominator=-3), two_thirds),
            (support.forged(Rational, numerator=1), two_thirds),  # A slot left unset.
            (support.forged(Rational, numerator=1, denominator=3.0), two_thirds),
            (support.forged(Rational, numerator=True, denominator=3), two_thirds),
            (support.impostor(third), two_thirds),
            (third, support.impostor(two_thirds)),
        ]
        for claim in forged_cases:
            with self.subTest(claim=claim):
                self.rejects(args, claim, proof)
        # Integral entries as ints, 0 as 0/2.
        absorbing = [[1, 0], [Fraction(1, 2), Fraction(1, 2)]]
        args, proof = {"M": matrix(absorbing)}, ((1,), (0,), 2)
        for claim in [(1, 0), (Rational(1), 0), (1, Rational(0)), (Rational(1), support.forged(Rational, numerator=0,
                                                                                               denominator=2))]:
            with self.subTest(claim=claim):
                self.rejects(args, claim, proof)
        self.rejects({"M": matrix([[1]])}, (1,), ((), (), 2))

    def test_bool_float_str_and_lists(self):
        """A bool, a float or a str in a Rational or int slot; a list for a tuple, in the value and the evidence."""
        rows, pi = KNOWN[0]
        args, proof = {"M": matrix(rows)}, ((0,), (0,), 3)
        good = value(pi)
        absorbing = {"M": matrix([[1, 0], [Fraction(1, 2), Fraction(1, 2)]])}
        for claim in [(True, Rational(0)), (1.0, 0.0), ("1", "0"), (Rational(1), False), (Rational(1), None)]:
            with self.subTest(claim=claim):
                self.rejects(absorbing, claim, ((1,), (0,), 2))
        self.rejects(args, list(good), proof)
        self.rejects(args, good, [(0,), (0,), 3])
        self.rejects(args, good, ([0], (0,), 3))
        self.rejects(args, good, ((0,), [0], 3))
        for bad in [((False,), (0,), 3), ((0,), (0.0,), 3), (("0",), (0,), 3), ((0,), (0,), 3.0), ((0,), (0,), True),
                    ((0,), (0,), "3"), ((0,), (0,), Rational(3)), ((0,), (0,), None)]:
            with self.subTest(evidence=bad):
                self.rejects(args, good, bad)
        self.rejects({"M": matrix([[1]])}, (Rational(1),), ((), (), True))
        self.accepts(rows, pi, proof)

    def test_evidence_of_the_wrong_shape(self):
        """Missing, extra or misshapen evidence: indices of the wrong count, out of range, repeated or out of order."""
        rows = [[Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)], [Fraction(1, 2), 0, Fraction(1, 2)], [0, 1, 0]]
        pi = solve(rows)
        assert pi is not None
        args, good = {"M": matrix(rows)}, value(pi)
        chosen_rows, chosen_cols, prime = evidence(rows)
        self.accepts(rows, pi, (chosen_rows, chosen_cols, prime))
        for bad in [(), (chosen_rows,), (chosen_rows, chosen_cols), (chosen_rows, chosen_cols, prime, ()),
                    (chosen_rows, chosen_cols, prime, prime), ((0,), (0, 1), prime), ((0, 1, 2), (0, 1), prime),
                    ((0, 3), (0, 1), prime), ((-1, 0), (0, 1), prime), ((0, 0), (0, 1), prime),
                    ((1, 0), (0, 1), prime), ((0, 1), (1, 0), prime), ((), (), prime), (((0, 1),), chosen_cols, prime),
                    ((chosen_rows, chosen_cols), prime), None, prime]:
            with self.subTest(evidence=bad):
                self.rejects(args, good, bad)
        self.rejects({"M": matrix([[1]])}, (Rational(1),), ((0,), (), 2))
        self.rejects({"M": matrix([[1]])}, (Rational(1),), ((), ()))

    def test_forged_consistent_evidence(self):
        """Evidence that is well formed but wrong: a singular minor, a prime that divides the minor, a composite
        modulus that leaves it nonzero, a prime past the bound of prime_proof."""
        absorbing = [[1, 0], [Fraction(1, 2), Fraction(1, 2)]]  # K = [[0, 0], [-1, 1]].
        args, good = {"M": matrix(absorbing)}, value((1, 0))
        self.rejects(args, good, ((0,), (0,), 3))  # A zero minor.
        self.rejects(args, good, ((0,), (1,), 3))
        rows, pi = KNOWN[0]  # K = [[2, -2], [-1, 1]].
        args, good = {"M": matrix(rows)}, value(pi)
        self.rejects(args, good, ((0,), (0,), 2))  # 2 divides the minor 2.
        self.rejects(args, good, ((0,), (1,), 2))  # and the minor -2.
        for modulus in (1, 0, -3, 9, 15, 2 ** 61 + 1, 3 * 1_000_000_007):  # det 2 is nonzero modulo each composite.
            with self.subTest(modulus=modulus):
                self.rejects(args, good, ((0,), (0,), modulus))
        mersenne_89 = 2 ** 89 - 1  # Prime, but past PSI_13, where prime_proof needs a certificate.
        self.assertGreater(mersenne_89, PSI_13)
        self.rejects(args, good, ((0,), (0,), mersenne_89))
        # A 3-state chain: s = 6 and a minor 2 * 3 * ... divisible by small primes.
        chain = [[Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)], [0, Fraction(1, 2), Fraction(1, 2)],
                 [Fraction(1, 3), 0, Fraction(2, 3)]]
        pi_chain = solve(chain)
        assert pi_chain is not None
        args = {"M": matrix(chain)}
        for chosen_rows in combinations(range(3), 2):
            for chosen_cols in combinations(range(3), 2):
                d = minor(chain, chosen_rows, chosen_cols)
                for p in (2, 3, 5, 7, 11, 13):
                    with self.subTest(minor=(chosen_rows, chosen_cols), prime=p):
                        if d % p:
                            self.accepts(chain, pi_chain, (chosen_rows, chosen_cols, p))
                        else:
                            self.rejects(args, value(pi_chain), (chosen_rows, chosen_cols, p))

    def test_wrong_operation_and_arguments(self):
        """A wrong operation name, a wrong argument set, an M of another class, an impostor or forged Matrix."""
        rows, pi = KNOWN[0]
        m = matrix(rows)
        good, proof = value(pi), evidence(rows)
        for operation in ["stationary_distributions", "correlation", "", 5, None, b"stationary_distribution"]:
            with self.subTest(operation=operation):
                self.rejects({"M": m}, good, proof, operation=operation)
        for args in [{}, {"A": m}, {"M": m, "k": 1}, {1: m}, [("M", m)], None, {"M": rows},
                     {"M": tuple(tuple(row) for row in m.rows)}, {"M": support.impostor(m)},
                     {"M": support.forged(mf.Matrix, rows=m.rows)},
                     {"M": support.forged(mf.Matrix, rows=list(m.rows), ncols=2)},
                     {"M": support.forged(mf.Matrix, rows=m.rows, ncols=3)},
                     {"M": support.forged(mf.Matrix, rows=(m.rows[0], list(m.rows[1])), ncols=2)},
                     {"M": support.forged(mf.Matrix, rows=(m.rows[0], (Rational(1, 4), 0.75)), ncols=2)},
                     {"M": support.forged(mf.Matrix, rows=(m.rows[0], (Rational(1, 4), 1)), ncols=2)},
                     {"M": support.forged(mf.Matrix, rows=(m.rows[0], (support.forged(Rational, numerator=2,
                                                                                      denominator=8),
                                                                       m.rows[1][1])), ncols=2)}]:
            with self.subTest(args=args):
                self.rejects(args, good, proof)
        self.accepts(rows, pi, proof)

    def test_oversized_forgeries_fail_fast(self):
        """Giant integers in the value or the evidence, a giant value tuple or index tuple, a common denominator of
        the claim past max_integer_bits: refused (False) at once under the default budget."""
        rows, pi = KNOWN[0]
        args, good, proof = {"M": matrix(rows)}, value(pi), ((0,), (0,), 3)
        huge = support.HUGE
        self.rejects_fast(args, (Rational(huge, huge + 1), Rational(1, huge + 1)), proof)
        self.rejects_fast(args, (Rational(1, 3), Rational(huge * 2, huge * 3 + 1)), proof)
        self.rejects_fast(args, good, ((0,), (0,), huge + 1))
        self.rejects_fast(args, good, ((0,), (0,), 2 ** 4000 + 1))
        self.rejects_fast(args, good, ((huge,), (0,), 3))
        self.rejects_fast(args, good, (tuple(range(10 ** 5)), (0,), 3))
        self.rejects_fast(args, tuple(Rational(1, 3) for _ in range(10 ** 5)), proof)
        # Two denominators of about 40,000 bits each, coprime: their lcm passes max_integer_bits (65,536).
        q1, q2 = 2 ** 40_000 + 1, 2 ** 40_000 + 3
        self.rejects_fast(args, (Rational(1, q1), Rational(q1 - 1, q1)), proof)  # Wrong but small lcm.
        self.rejects_fast(args, (Rational(1, q1), Rational(1, q2)), proof)
        # The request side: an M past max_matrix_entries or with an integer past max_integer_bits is unverifiable.
        self.rejects({"M": support.oversized_matrix()}, good, proof, may_exhaust=True)

    def test_checker_never_raises_on_garbage(self):
        """Odd objects anywhere answer False."""
        rows, pi = KNOWN[0]
        args, good, proof = {"M": matrix(rows)}, value(pi), evidence(rows)
        garbage: list[object] = [None, 0, 1.5, "x", object(), [], {}, (None,), ((),), frozenset()]
        for item in garbage:
            with self.subTest(item=item):
                self.rejects(args, item, proof)
                self.rejects(args, good, item)
                self.rejects({"M": item}, good, proof)
                self.rejects(item, good, proof)

    def test_checker_module_states_its_theorem(self):
        """The docstring states the theorem and the evidence layout that the producer relies on."""
        doc = checker.__doc__ or ""
        for phrase in ("rank at least n - 1", "least common multiple", "strictly increasing", "prime_proof"):
            self.assertIn(phrase, doc)
        spec = family.SPECS[0]
        self.assertEqual((spec.name, spec.version, spec.returns), (OPERATION, 6, "tuple[Rational, ...]"))
        self.assertEqual([(a.name, a.annotation, a.default, a.kind) for a in spec.arguments],
                         [("M", "Matrix", None, "matrix")])
