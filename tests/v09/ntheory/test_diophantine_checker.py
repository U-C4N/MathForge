"""The checker of linear_diophantine alone (mathforge.certify.ntheory.diophantine), written before its producer.

Every true claim here is computed by the test: by a search over small integers that uses neither a gcd nor Euclid's
algorithm, or by the short extended Euclid below for long integers, never by the family's producer. Each claim comes
with the evidence of the card's layout: () for a solvable equation, (g,) for an unsolvable one. Every corrupted claim
is answered with False, never with an exception, and oversized forged values and evidence are refused at once under
the default budget. Under that budget the checker raises ResourceLimitError only for arguments over
max_integer_bits, which the producer's budget refuses too.
"""

import random
import time

from mathforge.areas.ntheory import diophantine as family
from mathforge.areas.ntheory.diophantine import DiophantineSolution
from mathforge.errors import InvalidInput
from mathforge.kernels.certificates import factorization_evidence, prime_evidence
from mathforge.limits import ComputationLimits, ResourceLimitError, computation

from .. import support

OPERATION = "linear_diophantine"
NONE = DiophantineSolution(None, None, None, None)  # The value of an equation without integer solution.
FAST = 2.0  # Seconds within which the checker refuses an oversized forged claim under the default budget.
SOLVED = DiophantineSolution(1, 1, 3, -2)  # 6x + 9y = 15: (1 + 3t, 1 - 2t).


def arguments(a, b, c):
    """The arguments of a request, as the verifier hands them to the checker."""
    return {"a": a, "b": b, "c": c}


def forged(x, y, dx, dy):
    """A DiophantineSolution made without its canonical-form validation, as no decoded document can be."""
    return support.forged(DiophantineSolution, x=x, y=y, dx=dx, dy=dy)


class Int(int):
    """An int subclass, which is no exact int to the checker: its arithmetic could be anything."""


def searched(a, b, c):
    """The canonical claim (value, evidence) on small integers, found by search with neither a gcd nor Euclid.

    For b != 0 the step is the least dx >= 1 with b | a*dx (it is |b|/gcd(a, b)), dy = -a*dx/b, and x is the one
    residue in [0, dx) that solves the equation, if any. For b == 0 the step is (0, 1), y = 0 and x = c/a. The
    evidence of an unsolvable equation is the greatest common divisor of a and b, found by search."""
    if b == 0:
        if c % a == 0:
            return DiophantineSolution(c // a, 0, 0, 1), ()
    else:
        dx = next(d for d in range(1, abs(b) + 1) if a * d % b == 0)
        x = next((x for x in range(dx) if (c - a * x) % b == 0), None)
        if x is not None:
            return DiophantineSolution(x, (c - a * x) // b, dx, -a * dx // b), ()
    return NONE, (max(d for d in range(1, max(abs(a), abs(b)) + 1) if a % d == 0 and b % d == 0),)


def solutions_in_box(a, b, c, radius):
    """Every integer solution (x, y) of a*x + b*y = c with |x|, |y| <= radius, by enumeration."""
    span = range(-radius, radius + 1)
    return {(x, y) for x in span for y in span if a * x + b * y == c}


def family_in_box(value, radius):
    """The pairs (x + t*dx, y + t*dy) of a claimed solution family with |x|, |y| <= radius. One of dx, dy is nonzero,
    so |t| <= radius + max(|x|, |y|) reaches every pair of the box."""
    if value.x is None:
        return set()
    reach = radius + max(abs(value.x), abs(value.y))
    pairs = {(value.x + t * value.dx, value.y + t * value.dy) for t in range(-reach, reach + 1)}
    return {(x, y) for x, y in pairs if abs(x) <= radius and abs(y) <= radius}


def euclid(a, b):
    """(g, u, v) with g = gcd(a, b) >= 0 and u*a + v*b = g: the extended Euclidean algorithm."""
    r0, r1, u0, u1, v0, v1 = a, b, 1, 0, 0, 1
    while r1:
        q = r0 // r1
        r0, r1, u0, u1, v0, v1 = r1, r0 - q * r1, u1, u0 - q * u1, v1, v0 - q * v1
    return (r0, u0, v0) if r0 >= 0 else (-r0, -u0, -v0)


def reference(a, b, c):
    """The canonical claim (value, evidence) for (a, b) != (0, 0), from the extended Euclidean algorithm: the
    solution u*c/g, v*c/g moved along the step (|b|/g, -sign(b)*a/g) until 0 <= x < dx; (g,) when g does not
    divide c."""
    g, u, v = euclid(a, b)
    if c % g:
        return NONE, (g,)
    if b == 0:
        return DiophantineSolution(c // a, 0, 0, 1), ()
    x, y = u * (c // g), v * (c // g)
    dx, dy = abs(b) // g, -a // g if b > 0 else a // g
    shift = x // dx
    return DiophantineSolution(x - shift * dx, y - shift * dy, dx, dy), ()


class DiophantineCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRefusedFast(self, args, value, evidence=()):
        """The checker answers False at once: an oversized forged claim costs no arithmetic on its integers."""
        start = time.perf_counter()
        self.assertRejects(OPERATION, args, value, evidence)
        self.assertLess(time.perf_counter() - start, FAST)

    # -- True claims --------------------------------------------------------------------------------------------

    def test_accepts_the_known_values_of_the_card(self):
        for (a, b, c), value, evidence in (((6, 9, 15), SOLVED, ()), ((6, 9, 10), NONE, (3,)),
                                           ((0, 5, 10), DiophantineSolution(0, 2, 1, 0), ()),
                                           ((4, 0, 8), DiophantineSolution(2, 0, 0, 1), ()),
                                           ((3, -5, 1), DiophantineSolution(2, 1, 5, 3), ())):
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(searched(a, b, c), (value, evidence))
                self.assertEqual(reference(a, b, c), (value, evidence))
                self.assertAccepts(OPERATION, arguments(a, b, c), value, evidence)
        # (0, 0, 0) is invalid input: nothing is accepted for it (test_rejects_the_excluded_equation_...).
        self.assertRejects(OPERATION, arguments(0, 0, 0), DiophantineSolution(0, 0, 1, 0))

    def test_accepts_every_small_equation_found_by_search(self):
        """Every equation with |a|, |b|, |c| <= 5: the searched claim generates exactly the solutions that an
        enumeration finds in a box that holds its particular solution, the reference agrees, and the checker
        accepts it. The special cases of the card are among them: b == 0 (step (0, 1), y = 0), a == 0 (step
        (1, 0), x = 0), c == 0, a negative b (dx = |b|/g > 0) and the unsolvable equations."""
        radius = 12  # |x| <= 5 and |y| <= |a| + |c| <= 10 for the canonical particular solution.
        span = range(-5, 6)
        for a in span:
            for b in span:
                if a == 0 and b == 0:
                    continue
                for c in span:
                    value, evidence = searched(a, b, c)
                    with self.subTest(a=a, b=b, c=c):
                        self.assertEqual(family_in_box(value, radius), solutions_in_box(a, b, c, radius))
                        self.assertEqual(reference(a, b, c), (value, evidence))
                        self.assertAccepts(OPERATION, arguments(a, b, c), value, evidence)

    def test_accepts_exactly_the_canonical_claim_among_all_small_claims(self):
        """For every equation with |a|, |b|, |c| <= 2, of all 625 records with components in [-2, 2] (which holds
        every canonical value of these equations) the checker accepts exactly the searched one, and of the
        divisors -1 ... 6 exactly those g >= 1 that divide a and b but not c."""
        span = range(-2, 3)
        candidates = [forged(x, y, dx, dy) for x in span for y in span for dx in span for dy in span]
        for a in span:
            for b in span:
                if a == 0 and b == 0:
                    continue
                for c in span:
                    value, evidence = searched(a, b, c)
                    args = arguments(a, b, c)
                    with self.subTest(a=a, b=b, c=c):
                        accepted = [(r.x, r.y, r.dx, r.dy) for r in candidates
                                    if support.verdict(family, OPERATION, args, r) is True]
                        expected = [] if value.x is None else [(value.x, value.y, value.dx, value.dy)]
                        self.assertEqual(accepted, expected)
                        divisors = [g for g in range(-1, 7) if support.verdict(family, OPERATION, args, NONE, (g,))]
                        self.assertEqual(divisors, [g for g in range(1, 7) if a % g == 0 and b % g == 0 and c % g])

    def test_accepts_long_equations_solved_by_the_reference(self):
        """Seeded equations up to about 4,000 bits, with common factors so that some are unsolvable, and with a = 0,
        b = 0 or c = 0 among them; the reference agrees with the search on the small ones."""
        rng = random.Random(20261112)
        for case in range(160):
            bits = rng.choice((4, 16, 64, 256, 1024, 2048))
            factor = rng.randrange(1, 1 << rng.randrange(1, bits + 1))
            a, b = factor * rng.randrange(-(1 << bits), 1 << bits), factor * rng.randrange(-(1 << bits), 1 << bits)
            if case % 9 == 0:
                a = 0
            elif case % 9 == 1:
                b = 0
            if a == 0 and b == 0:
                b = factor
            c = rng.choice((0, factor * rng.randrange(-(1 << bits), 1 << bits), rng.randrange(-(1 << bits), 1 << bits)))
            value, evidence = reference(a, b, c)
            with self.subTest(case=case, bits=bits):
                if max(abs(a), abs(b), abs(c)) <= 200:
                    self.assertEqual(searched(a, b, c), (value, evidence))
                self.assertAccepts(OPERATION, arguments(a, b, c), value, evidence)

    def test_accepts_the_longest_admitted_arguments(self):
        """Arguments of 65,536 bits, the default max_integer_bits, within the default budget."""
        n = 65_535
        dx, dy = (1 << n) + 1, (1 << n) - 1  # Odd, two apart: coprime.
        a, b = -dy, dx  # So (b/g, -a/g) = (dx, dy) with g = 1.
        x, y = dx - 1, dy - 1  # The largest admissible x; then c = dy - dx = -2.
        c = a * x + b * y
        value = DiophantineSolution(x, y, dx, dy)
        self.assertEqual((max(a.bit_length(), b.bit_length()), c), (65_536, -2))
        self.assertEqual(reference(a, b, c), (value, ()))  # Three steps of Euclid on these two.
        self.assertAccepts(OPERATION, arguments(a, b, c), value)
        # b == 0 and a == 0, with a 32,768-bit quotient k in c of 65,536 bits.
        a, k = (1 << 32_767) + 3, (1 << 32_768) - 5
        self.assertEqual((a * k).bit_length(), 65_536)
        self.assertAccepts(OPERATION, arguments(a, 0, a * k), DiophantineSolution(k, 0, 0, 1))
        self.assertAccepts(OPERATION, arguments(0, -a, a * k), DiophantineSolution(0, -k, 1, 0))
        # Unsolvable, with a 30,000-bit common divisor g of a and b that leaves c the remainder 1: true by
        # construction (Euclid on these would take seconds here).
        rng = random.Random(65536)
        g = rng.getrandbits(30_000) | 1 << 29_999
        a, b = g * rng.getrandbits(35_000), -g * rng.getrandbits(35_000)
        c = g * rng.getrandbits(35_000) + 1
        self.assertLessEqual(max(a.bit_length(), b.bit_length(), c.bit_length()), 65_536)
        self.assertAccepts(OPERATION, arguments(a, b, c), NONE, (g,))

    def test_accepts_any_common_divisor_that_misses_c(self):
        """The producer hands over gcd(a, b); any other g >= 1 that divides a and b but not c proves the claim as
        well, prime or composite."""
        for (a, b, c), divisors in (((12, 18, 9), (2, 6)), ((0, 6, 3), (2, 6)), ((8, 0, 6), (4, 8)),
                                    ((-12, 18, -9), (2, 6)), ((30, 42, 7), (2, 3, 6))):
            for g in divisors:
                with self.subTest(a=a, b=b, c=c, g=g):
                    self.assertAccepts(OPERATION, arguments(a, b, c), NONE, (g,))

    # -- Corrupted claims ---------------------------------------------------------------------------------------

    def test_rejects_wrong_values(self):
        args = arguments(6, 9, 15)
        for value in (DiophantineSolution(2, 1, 3, -2), DiophantineSolution(1, 2, 3, -2),
                      DiophantineSolution(1, 1, 3, 2), DiophantineSolution(0, 1, 1, 0),
                      DiophantineSolution(0, 0, 0, 1), forged(1, 1, 6, -4)):  # Half the solutions, gcd 2.
            with self.subTest(value=value):
                self.assertRejects(OPERATION, args, value)
        # Unsolvable for a solvable equation, whatever g, and solvable for an unsolvable one, whatever the value.
        for g in (1, 2, 3, 5, 15):
            self.assertRejects(OPERATION, args, NONE, (g,))
        for value in (SOLVED, DiophantineSolution(0, 0, 3, -2), DiophantineSolution(2, -2, 3, -2)):
            self.assertRejects(OPERATION, arguments(6, 9, 10), value)
        for (a, b, c), value in (((4, 0, 8), DiophantineSolution(3, 0, 0, 1)),
                                 ((4, 0, 8), DiophantineSolution(-2, 0, 0, 1)),
                                 ((0, 5, 10), DiophantineSolution(0, 3, 1, 0)),
                                 ((0, 5, 10), DiophantineSolution(0, 2, 1, 1)),
                                 ((3, -5, 1), DiophantineSolution(2, 1, 5, -3)),
                                 ((3, -5, 1), DiophantineSolution(1, 1, 5, 3))):
            with self.subTest(a=a, b=b, c=c, value=value):
                self.assertRejects(OPERATION, arguments(a, b, c), value)

    def test_rejects_correct_but_non_canonical_values(self):
        """Each record describes the true solution set, so only the canonical-form rule that it breaks can reject
        it; the record class refuses to build it, and object.__new__ bypasses that."""
        for (a, b, c), fields in (((6, 9, 15), (4, -1, 3, -2)),  # x >= dx
                                  ((6, 9, 15), (-2, 3, 3, -2)),  # x < 0
                                  ((6, 9, 15), (1, 1, -3, 2)),  # dx < 0
                                  ((6, 9, 15), (-2, 3, -3, 2)),
                                  ((4, 0, 8), (2, 0, 0, -1)),  # b == 0 needs dy == 1
                                  ((4, 0, 8), (2, 7, 0, 1)),  # and y == 0
                                  ((0, 5, 10), (0, 2, -1, 0)),
                                  ((0, 5, 10), (1, 2, 1, 0)),
                                  ((3, -5, 1), (7, 4, 5, 3)),
                                  ((3, -5, 1), (-3, -2, 5, 3))):
            with self.subTest(a=a, b=b, c=c, fields=fields):
                self.assertEqual(family_in_box(forged(*fields), 30), solutions_in_box(a, b, c, 30))
                with self.assertRaises(InvalidInput):
                    DiophantineSolution(*fields)
                self.assertRejects(OPERATION, arguments(a, b, c), forged(*fields))
        # None in some fields only.
        for fields in ((None, None, None, 0), (1, None, 3, -2), (None, 1, 3, -2), (1, 1, 3, None)):
            with self.subTest(fields=fields):
                with self.assertRaises(InvalidInput):
                    DiophantineSolution(*fields)
                self.assertRejects(OPERATION, arguments(6, 9, 15), forged(*fields))
                self.assertRejects(OPERATION, arguments(6, 9, 10), forged(*fields), (3,))

    def test_rejects_impostors_and_forged_records(self):
        class Subclass(DiophantineSolution):
            """A subclass in another module, with the canonical fields."""

        cases = ((arguments(6, 9, 15), support.impostor(SOLVED), ()),
                 (arguments(6, 9, 10), support.impostor(NONE), (3,)),
                 (arguments(6, 9, 15), Subclass(1, 1, 3, -2), ()),
                 (arguments(6, 9, 15), (1, 1, 3, -2), ()),
                 (arguments(6, 9, 15), {"x": 1, "y": 1, "dx": 3, "dy": -2}, ()),
                 (arguments(6, 9, 15), None, ()),
                 (arguments(6, 9, 10), None, (3,)),
                 (arguments(6, 9, 10), (), (3,)),
                 # Records made with object.__new__ that lack fields.
                 (arguments(6, 9, 15), support.forged(DiophantineSolution, x=1, y=1, dx=3), ()),
                 (arguments(6, 9, 10), support.forged(DiophantineSolution), (3,)))
        for index, (args, value, evidence) in enumerate(cases):
            with self.subTest(case=index, kind=type(value).__qualname__):  # A record without fields has no repr.
                self.assertRejects(OPERATION, args, value, evidence)
        self.assertEqual(type(support.impostor(SOLVED)).__qualname__, "DiophantineSolution")

    def test_rejects_inexact_numbers(self):
        """A bool, a float, a str, None or an int subclass where an exact int belongs: in the value, in the
        arguments and in the evidence. Each sits where the right int would make the claim true."""
        for (a, b, c), fields in (((6, 9, 15), (True, 1, 3, -2)), ((6, 9, 15), (1, True, 3, -2)),
                                  ((0, 5, 10), (False, 2, 1, 0)), ((0, 5, 10), (0, 2, True, 0)),
                                  ((0, 5, 10), (0, 2, 1, False)), ((4, 0, 8), (2, 0, 0, True)),
                                  ((6, 9, 15), (1.0, 1, 3, -2)), ((6, 9, 15), (1, 1, 3, -2.0)),
                                  ((6, 9, 15), ("1", 1, 3, -2)), ((6, 9, 15), (Int(1), 1, 3, -2)),
                                  ((6, 9, 15), (1, 1, Int(3), -2)), ((6, 9, 15), ([1], 1, 3, -2))):
            with self.subTest(a=a, b=b, c=c, fields=fields):
                self.assertRejects(OPERATION, arguments(a, b, c), forged(*fields))
        self.assertAccepts(OPERATION, arguments(1, 0, 1), DiophantineSolution(1, 0, 0, 1))
        self.assertAccepts(OPERATION, arguments(4, 0, 8), DiophantineSolution(2, 0, 0, 1))
        for args, value in ((arguments(True, 0, 1), DiophantineSolution(1, 0, 0, 1)),
                            (arguments(4, False, 8), DiophantineSolution(2, 0, 0, 1)),
                            (arguments(6.0, 9, 15), SOLVED), (arguments(6, 9.0, 15), SOLVED),
                            (arguments(6, 9, "15"), SOLVED), (arguments(6, 9, None), SOLVED),
                            (arguments(Int(6), 9, 15), SOLVED), (arguments(6, 9, Int(15)), SOLVED)):
            with self.subTest(args=args):
                self.assertRejects(OPERATION, args, value)
        for evidence in ((3.0,), ("3",), (True,), (Int(3),), (None,)):
            with self.subTest(evidence=evidence):
                self.assertRejects(OPERATION, arguments(6, 9, 10), NONE, evidence)

    def test_rejects_lists_for_tuples(self):
        self.assertRejects(OPERATION, arguments(6, 9, 10), NONE, [3])
        self.assertRejects(OPERATION, arguments(6, 9, 15), SOLVED, [])
        self.assertRejects(OPERATION, arguments(6, 9, 15), [1, 1, 3, -2])
        self.assertRejects(OPERATION, arguments(6, 9, 10), NONE, ([3],))
        self.assertRejects(OPERATION, arguments(6, 9, 15), forged([1], [1], [3], [-2]))

    def test_rejects_missing_extra_and_misshapen_evidence(self):
        for evidence in ((0,), (1,), (3,), ((),), ("",), (None,), ((3,),)):
            with self.subTest(evidence=evidence):
                self.assertRejects(OPERATION, arguments(6, 9, 15), SOLVED, evidence)
        for evidence in ((), (3, 3), (3, 1), ((3,),), (3, ()), "3", 3, None, {3: 3}):
            with self.subTest(evidence=evidence):
                self.assertRejects(OPERATION, arguments(6, 9, 10), NONE, evidence)
        self.assertRejects(OPERATION, arguments(6, 9, 15), SOLVED, None)

    def test_rejects_forged_divisor_certificates(self):
        """The evidence of an unsolvable equation is one integer g; prime certificates and factorizations from
        mathforge.kernels.certificates in its place, a g of another equation and a g that divides c are refused."""
        args = arguments(6, 9, 10)
        for evidence in ((factorization_evidence(3),), (prime_evidence(3),), (3, prime_evidence(3)),
                         (factorization_evidence(6)[:1],), factorization_evidence(3), (factorization_evidence(9),)):
            with self.subTest(evidence=evidence):
                self.assertRejects(OPERATION, args, NONE, evidence)
        # 6 = gcd(12, 18) and 5 | 10 are divisors of other numbers; 0 and the negative divisors are no g >= 1.
        for g in (6, 5, 2, 9, 0, -3, -1):
            with self.subTest(g=g):
                self.assertRejects(OPERATION, args, NONE, (g,))
        # 12x + 18y = 9: g must divide 12 and 18 but not 9.
        for g in (1, 3, 4, 9, 12, 18, 36, 216, 0, -6):
            with self.subTest(g=g):
                self.assertRejects(OPERATION, arguments(12, 18, 9), NONE, (g,))

    def test_rejects_a_wrong_operation_or_malformed_arguments(self):
        for operation in ("linear_diophantine ", "LINEAR_DIOPHANTINE", "integer_gcd", "continued_fraction", "",
                          None):
            with self.subTest(operation=operation):
                self.assertRejects(operation, arguments(6, 9, 15), SOLVED)
        for args in ({"a": 6, "b": 9}, {"a": 6, "b": 9, "c": 15, "d": 0}, {"a": 6, "b": 9, "z": 15},
                     [("a", 6), ("b", 9), ("c", 15)], (6, 9, 15), None):
            with self.subTest(args=args):
                self.assertRejects(OPERATION, args, SOLVED)

    def test_rejects_the_excluded_equation_with_a_and_b_zero(self):
        """a = b = 0 is invalid input. (0, 0, 1, 0) satisfies the four identities of a solvable claim for c = 0,
        and every g >= 2 the divisibility certificate for c = 1, yet neither claim may verify."""
        self.assertRejects(OPERATION, arguments(0, 0, 0), DiophantineSolution(0, 0, 1, 0))
        self.assertRejects(OPERATION, arguments(0, 0, 0), DiophantineSolution(0, 0, 0, 1))
        self.assertRejects(OPERATION, arguments(0, 0, 0), NONE, (1,))
        self.assertRejects(OPERATION, arguments(0, 0, 1), NONE, (2,))
        self.assertRejects(OPERATION, arguments(0, 0, 5), NONE, (3,))

    def test_refuses_oversized_forged_values_and_evidence_fast(self):
        """A component or a g longer than the arguments allow is refused by its bit length, before any product,
        gcd or charge for them. (1, 1, 3*s, -2*s) with s = 2**2,000,000 satisfies conditions (1), (2) and (4) for
        6x + 9y = 15; only the gcd s refutes it, and the charge for that gcd alone exceeds the default budget, so
        without the screening the claim would end as exhausted resources instead of False."""
        long, huge = 1 << 2_000_000, support.HUGE
        for args, value, evidence in ((arguments(6, 9, 15), forged(1, 1, 3 * long, -2 * long), ()),
                                      (arguments(6, 9, 15), forged(huge, 1, 3, -2), ()),
                                      (arguments(6, 9, 15), forged(1, -huge, 3, -2), ()),
                                      (arguments(6, 9, 15), forged(0, 0, long + 1, long), ()),
                                      (arguments(6, 9, 15), forged(long, long, long, long), ()),
                                      (arguments(0, 5, 10), forged(0, long, 1, 0), ()),
                                      (arguments(4, 0, 8), forged(long, 0, 0, 1), ()),
                                      (arguments(6, 9, 10), NONE, (huge,)),
                                      (arguments(6, 9, 10), NONE, (long + 3,)),
                                      (arguments(6, 9, 10), NONE, (3,) * 1_000_000),
                                      (arguments(6, 9, 15), SOLVED, (0,) * 1_000_000)):
            with self.subTest(value=type(value).__name__, evidence=len(evidence)):
                self.assertRefusedFast(args, value, evidence)
        # At the longest admitted arguments, a component one bit beyond the bound.
        n = 65_535
        a, b, c = -((1 << n) - 1), (1 << n) + 1, -2
        self.assertRefusedFast(arguments(a, b, c), forged(1 << 65_537, 0, (1 << n) + 1, (1 << n) - 1))
        self.assertRefusedFast(arguments(a, b, c), forged(0, 0, (1 << 65_537) + 1, 1 << 65_537))
        # Refused after the screening alone, which a budget of ten work units pays for.
        self.assertRejects(OPERATION, arguments(6, 9, 15), forged(1, 1, 3 * long, -2 * long),
                           limits=ComputationLimits(max_work=10))

    def test_arguments_over_the_integer_limit_exceed_the_budget(self):
        """An argument over max_integer_bits never reaches a producer under the same limits: the checker raises
        ResourceLimitError (the verifier reports the resources, never a verified claim) at once."""
        for args, value in ((arguments(support.HUGE, 9, 15), SOLVED), (arguments(6, -support.HUGE, 15), SOLVED),
                            (arguments(6, 9, support.HUGE), NONE)):
            start = time.perf_counter()
            with self.subTest(args=sorted(name for name, item in args.items() if item.bit_length() > 65_536)):
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, OPERATION, args, value, (3,) if value is NONE else ())
            self.assertLess(time.perf_counter() - start, FAST)
        # A wider budget admits them; the record itself is built inside one, since it checks its integers too.
        wide = ComputationLimits(max_integer_bits=70_000)
        with computation(wide, fresh=True):
            value = DiophantineSolution(0, 0, 1, -support.HUGE)
        self.assertAccepts(OPERATION, arguments(support.HUGE, 1, 0), value, limits=wide)

    def test_charges_the_budget(self):
        """The products and the gcd of a claim on 65,536-bit integers cost about 33,000 work units, which the
        default budget pays and a budget of 1,000 units does not."""
        n = 65_535
        dx, dy = (1 << n) + 1, (1 << n) - 1
        args, value = arguments(-dy, dx, -2), DiophantineSolution(dx - 1, dy - 1, dx, dy)
        self.assertAccepts(OPERATION, args, value)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, args, value, limits=ComputationLimits(max_work=1_000))
        self.assertAccepts(OPERATION, arguments(6, 9, 15), SOLVED, limits=ComputationLimits(max_work=20))
