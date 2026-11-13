"""The diophantine family of mathforge.ntheory: linear_diophantine, from its producer to the MCP area tool.

These tests are the definition of done of the family; tests/v09/ntheory/test_diophantine_checker.py tests its checker
alone. tests/v09/support.py holds the shared helpers, and tests/v08/test_continued_fraction.py tests the pilot family.
A FamilyTestCase keeps FAMILY registered while its tests run, before and after

    python tools/new_family.py register ntheory diophantine

No expected value comes from the producer. Each one is a known value of the card (docs/v09/cards/m1-ntheory.md) or
worked by hand, the answer of enumerated(), which reads the canonical form off the solutions that it finds by trial,
with neither a gcd nor Euclid's algorithm, or the answer of reference(), the extended Euclidean algorithm with one
division per step, which shares no code with the producer's library calls (math.gcd and kernels.integers.inverse).
"""

from collections import Counter
from collections.abc import Iterator
from math import gcd
import random
from unittest.mock import patch

import mathforge as mf
from mathforge import lean, operations_v6
from mathforge.areas.ntheory import diophantine as family
from mathforge.areas.ntheory.diophantine import DiophantineSolution
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation

from .. import support

OPERATION = "linear_diophantine"
NONE = DiophantineSolution(None, None, None, None)  # The value of an equation without integer solution.
SOLVED = DiophantineSolution(1, 1, 3, -2)  # 6x + 9y = 15: (1 + 3t, 1 - 2t).
# ((a, b, c), value, evidence): the known values of the card, then values worked by hand that reach every sign of a
# and b, c = 0, a = 0 and b = 0 with either sign of the other, and unsolvable equations with a = 0 and with b = 0.
CARD = (((6, 9, 15), SOLVED, ()),
        ((6, 9, 10), NONE, (3,)),
        ((0, 5, 10), DiophantineSolution(0, 2, 1, 0), ()),
        ((4, 0, 8), DiophantineSolution(2, 0, 0, 1), ()),
        ((3, -5, 1), DiophantineSolution(2, 1, 5, 3), ()))
BY_HAND = (((-6, 9, 15), DiophantineSolution(2, 3, 3, 2), ()),  # -12 + 27 = 15, the step (9/3, 6/3).
           ((-6, -9, -15), SOLVED, ()),  # 6x + 9y = 15 negated: the same solutions, the same canonical form.
           ((5, 3, -1), DiophantineSolution(1, -2, 3, -5), ()),  # 5 - 6 = -1.
           ((2, 10, 0), DiophantineSolution(0, 0, 5, -1), ()),
           ((1, -1, 0), DiophantineSolution(0, 0, 1, 1), ()),
           ((0, -5, 10), DiophantineSolution(0, -2, 1, 0), ()),
           ((-4, 0, 8), DiophantineSolution(-2, 0, 0, 1), ()),
           ((0, 5, 3), NONE, (5,)),
           ((4, 0, 6), NONE, (4,)),
           ((12, -18, 9), NONE, (6,)))
# ((a, b, c), text, LaTeX) for every shape of the value: both coordinates moving, with signs; one coordinate fixed
# (a = 0, b = 0); a zero particular solution (zero terms are left out); no solution.
RENDERINGS = (((6, 9, 15), "(1 + 3*t, 1 - 2*t) for every integer t", r"(1 + 3t,\ 1 - 2t),\ t \in \mathbb{Z}"),
              ((3, -5, 1), "(2 + 5*t, 1 + 3*t) for every integer t", r"(2 + 5t,\ 1 + 3t),\ t \in \mathbb{Z}"),
              ((5, 3, -1), "(1 + 3*t, -2 - 5*t) for every integer t", r"(1 + 3t,\ -2 - 5t),\ t \in \mathbb{Z}"),
              ((2, 10, 0), "(5*t, -t) for every integer t", r"(5t,\ -t),\ t \in \mathbb{Z}"),
              ((1, -1, 0), "(t, t) for every integer t", r"(t,\ t),\ t \in \mathbb{Z}"),
              ((0, 5, 10), "(t, 2) for every integer t", r"(t,\ 2),\ t \in \mathbb{Z}"),
              ((0, -5, 10), "(t, -2) for every integer t", r"(t,\ -2),\ t \in \mathbb{Z}"),
              ((4, 0, 8), "(2, t) for every integer t", r"(2,\ t),\ t \in \mathbb{Z}"),
              ((-4, 0, 8), "(-2, t) for every integer t", r"(-2,\ t),\ t \in \mathbb{Z}"),
              ((6, 9, 10), "no integer solution", r"\varnothing"))
INEXACT = (0.5, True, False, "1", None, 6.0, mf.Rational(1, 2), mf.Rational(6), mf.Symbol("x"), (6,), [6], {"a": 6})


def arguments(a, b, c):
    """The arguments of a request, as run() and the verifier hand them on."""
    return {"a": a, "b": b, "c": c}


def brief(item) -> str:
    """A short text of a claim, a record or an integer for a failure message, an integer of more than 64 bits by its
    length: assertEqual would diff the full texts of long integers, character by character, for minutes."""
    if type(item) is int:
        return str(item) if item.bit_length() <= 64 else f"<int of {item.bit_length()} bits>"
    if type(item) is tuple:
        return "(" + ", ".join(brief(part) for part in item) + ("," if len(item) == 1 else "") + ")"
    if type(item) is DiophantineSolution:
        return "DiophantineSolution" + brief((item.x, item.y, item.dx, item.dy))
    return repr(item)


def forged(x, y, dx, dy):
    """A DiophantineSolution made without its canonical-form validation, as no decoded document can be."""
    return support.forged(DiophantineSolution, x=x, y=y, dx=dx, dy=dy)


def produced(a, b, c, limits=None):
    """(value, evidence) from the producer itself, in a fresh budget."""
    with computation(limits, fresh=True):
        return family.produce(OPERATION, arguments(a, b, c))


def enumerated(a, b, c):
    """The canonical claim (value, evidence) on small integers, read off the solutions found by trial, with neither a
    gcd nor Euclid's algorithm. For b != 0 the solutions are the x with b | c - a*x, each with y = (c - a*x)/b: the
    least x >= 0 among them is the particular solution, and the next one is one step further. Both lie below 2*|b|,
    since the canonical x is below dx <= |b|. For b == 0, x = c/a when a divides c, with the step (0, 1) and y = 0.
    The evidence of an unsolvable equation is the largest common divisor of a and b, found by trial."""
    if b == 0:
        if c % a == 0:
            return DiophantineSolution(c // a, 0, 0, 1), ()
    else:
        found = [x for x in range(2 * abs(b)) if (c - a * x) % b == 0]
        if found:
            x, after = found[0], found[1]
            y = (c - a * x) // b
            return DiophantineSolution(x, y, after - x, (c - a * after) // b - y), ()
    return NONE, (max(d for d in range(1, max(abs(a), abs(b)) + 1) if a % d == 0 and b % d == 0),)


def euclid(a, b):
    """(g, u, v) with g = gcd(a, b) >= 0 and u*a + v*b = g: the extended Euclidean algorithm, a division per step."""
    r0, r1, u0, u1, v0, v1 = a, b, 1, 0, 0, 1
    while r1:
        q, r = divmod(r0, r1)
        r0, r1, u0, u1, v0, v1 = r1, r, u1, u0 - q * u1, v1, v0 - q * v1
    return (r0, u0, v0) if r0 >= 0 else (-r0, -u0, -v0)


def reference(a, b, c):
    """The canonical claim (value, evidence) for (a, b) != (0, 0) by the extended Euclidean algorithm: the solution
    (u, v)*c/g of u*a + v*b = g, moved along the step (|b|/g, -sign(b)*a/g) into 0 <= x < dx; (g,) when g does not
    divide c."""
    g, u, v = euclid(a, b)
    if c % g:
        return NONE, (g,)
    if b == 0:
        return DiophantineSolution(c // a, 0, 0, 1), ()
    dx, dy = abs(b) // g, (-a if b > 0 else a) // g
    x, y = u * (c // g), v * (c // g)
    shift = x // dx
    return DiophantineSolution(x - shift * dx, y - shift * dy, dx, dy), ()


def seeded_equations(seed, count, sizes) -> Iterator[tuple[int, int, int]]:
    """count seeded equations (a, b, c), of a bit size drawn from sizes each: a and b share a factor that a random c
    usually misses, so that a third or more are unsolvable. One in ten has a = 0 and one in ten b = 0, and half of
    those take a multiple of the other coefficient for c, so that every shape of the value occurs; a = b = 0 never."""
    rng = random.Random(seed)
    for case in range(count):
        bits = rng.choice(sizes)
        factor = rng.randrange(1, 1 << rng.randrange(1, bits + 1))
        a, b = factor * rng.randrange(-(1 << bits), 1 << bits), factor * rng.randrange(-(1 << bits), 1 << bits)
        c = rng.choice((0, factor * rng.randrange(-(1 << bits), 1 << bits), rng.randrange(-(1 << bits), 1 << bits)))
        if case % 10 == 0:
            a = 0
        elif case % 10 == 1:
            b = 0
        if a == 0 and b == 0:
            b = factor
        if case % 20 in (0, 1):  # a = 0 or b = 0: a multiple of the nonzero coefficient, solvable.
            c = (a or b) * rng.randrange(-(1 << bits), 1 << bits)
        yield a, b, c


def shape(a, b, c, value) -> str:
    """The case of the card that an equation and its value fall in."""
    if value.x is None:
        return "unsolvable"
    return "a = 0" if a == 0 else "b = 0" if b == 0 else "c = 0" if c == 0 else "general"


def corruptions(value, evidence) -> Iterator[tuple[object, object]]:
    """Claims (value, evidence) that differ from a true one in one respect, each wrong or not canonical. For a solution
    family: a field moved by one, the step negated or doubled, the particular solution moved by a step (still a
    solution, not the canonical one), evidence added, or no solution claimed. For no solution: another g near
    gcd(a, b), missing or extra evidence, or a made-up solution family."""
    if value.x is None:
        (g,) = evidence
        # g >= 2, as 1 divides c. Every common divisor of a and b divides g, which g - 1 does only when it is 1.
        for other in (g - 1, g + 1, 2 * g, -g, 0):
            yield NONE, (other,)
        yield NONE, ()
        yield NONE, (g, g)
        yield forged(0, 0, 1, 0), ()
        yield forged(0, 0, 0, 1), ()
        return
    x, y, dx, dy = value.x, value.y, value.dx, value.dy
    for fields in ((x + 1, y, dx, dy), (x - 1, y, dx, dy), (x, y + 1, dx, dy), (x, y - 1, dx, dy),
                   (x, y, dx + 1, dy), (x, y, dx - 1, dy), (x, y, dx, dy + 1), (x, y, dx, dy - 1),
                   (x, y, -dx, -dy), (x, y, 2 * dx, 2 * dy), (x + dx, y + dy, dx, dy), (x - dx, y - dy, dx, dy)):
        yield forged(*fields), evidence
    yield value, (1,)
    for g in (1, 2, 3):
        yield NONE, (g,)


class DiophantineTests(support.FamilyTestCase):
    FAMILY = family

    def assertClaim(self, found, expected) -> None:
        """found == expected for claims (value, evidence) whose integers may be long, with a short message."""
        if found != expected:
            self.fail(f"{brief(found)} != {brief(expected)}")

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for (a, b, c), value, evidence in CARD + BY_HAND:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(enumerated(a, b, c), (value, evidence))  # The worked values, against the search.
                result = support.run(OPERATION, a=a, b=b, c=c)
                self.assertEqual(result.value, value)
                self.assertEqual(result.request.version, 6)
                self.assertEqual(result.steps[0].rule, "v6." + OPERATION)
                self.assertEqual(result.steps[0].outputs, (value, evidence))
                self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
                self.assertTrue(mf.verify(result).verified)
                self.roundtrip(result)
                self.assertEqual(support.value(OPERATION, a=a, b=b, c=c), value)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error."""
        # The card's invalid input: a = b = 0, whatever c (all of Z^2 solves it for c = 0, nothing otherwise).
        for c in (0, 1, -1, 15, 1 << 65_535):
            with self.subTest(c_bits=c.bit_length()):
                result = support.run(OPERATION, a=0, b=0, c=c)
                self.assertInvalidInput(result)
                self.assertRegex(result.error.message, "^linear_diophantine requires a or b nonzero")
                with self.assertRaisesRegex(InvalidInput, "^linear_diophantine requires a or b nonzero"):
                    produced(0, 0, c)
        # An inexact number, a str, None or a container in each argument in turn, the others valid.
        for name in ("a", "b", "c"):
            for item in INEXACT:
                args = arguments(6, 9, 15)
                args[name] = item
                with self.subTest(name=name, item=item):
                    self.assertInvalidInput(support.run(OPERATION, **args))
        # The producer checks the types itself, first (False is no 0 for it), as direct calls and the lean path reach
        # it without run()'s checks; it answers its own operation only.
        message = "^linear_diophantine requires exact integers a, b and c$"
        for args in (arguments(0.5, 9, 15), arguments(6, True, 15), arguments(6, 9, "15"), arguments(6, 9, None),
                     arguments(mf.Rational(6), 9, 15), arguments(0, False, 0)):
            with self.subTest(args=args), self.assertRaisesRegex(InvalidInput, message), computation(fresh=True):
                family.produce(OPERATION, args)
        with self.assertRaises(UnsupportedOperation), computation(fresh=True):
            family.produce("integer_gcd", arguments(6, 9, 15))
        # A wrong argument set, or arguments that are no mapping at all, as operations_v6.run receives them.
        for args in ({"a": 6, "b": 9}, {"a": 6, "b": 9, "c": 15, "d": 0}, {"a": 6, "b": 9, "z": 15},
                     [("a", 6), ("b", 9), ("c", 15)], (6, 9, 15), None):
            with self.subTest(args=args):
                self.assertInvalidInput(operations_v6.run(OPERATION, args))
        # The MCP area tool reports each at the stage that finds it.
        error = self.assertToolError(support.call(OPERATION, {"a": 0, "b": 0, "c": 0}), "invalid_input")
        self.assertEqual(error["stage"], "computation")
        for raw in (0.5, 9.0, True, None, [9], "9.0"):
            with self.subTest(raw=raw):
                error = self.assertToolError(support.call(OPERATION, {"a": 6, "b": raw, "c": 15}), "invalid_input")
                self.assertEqual((error["stage"], error["field"]), ("decoding", "arguments.b"))

    def test_resource_limits(self):
        """An integer of 2**65536 and max_work=1 exceed the budget (this family takes no matrix)."""
        huge = support.HUGE
        for args in (arguments(huge, 9, 15), arguments(6, -huge, 15), arguments(6, 9, huge), arguments(huge, 0, huge),
                     arguments(0, huge, 0)):
            with self.subTest(huge=[name for name, item in args.items() if item.bit_length() > 65_536]):
                self.assertResourceLimit(support.run(OPERATION, **args))
                with self.assertRaises(ResourceLimitError):
                    produced(**args)
        # The producer charges at least two work units for any input, so max_work=1 stops it before any check.
        for (a, b, c), _, _ in CARD:
            with self.subTest(a=a, b=b, c=c):
                self.assertResourceLimit(support.run(OPERATION, a=a, b=b, c=c, limits=ComputationLimits(max_work=1)))
                with self.assertRaises(ResourceLimitError):
                    produced(a, b, c, limits=ComputationLimits(max_work=1))

    def test_longest_admitted_arguments(self):
        """Arguments of 65,536 bits, the default max_integer_bits: the value equals the reference's, and run()
        verifies it within the default budget. The producer charges the extended Euclid of its inverse before it runs
        it, 131,072 of its 188,422 work units on the long case below, where the checker needs 32,778: a budget of
        150,000 units pays the checker, and the producer without either half of that charge, but not the producer."""
        n = 65_535
        dx, dy = (1 << n) + 1, (1 << n) - 1  # Odd and two apart: coprime.
        rng = random.Random(65_536)
        a, b = rng.getrandbits(65_536) | 1 << 65_535, -(rng.getrandbits(65_536) | 1 << 65_535)
        c = gcd(a, b) * rng.getrandbits(65_000)  # Solvable, with a long extended Euclid.
        k, short_a, short_b = rng.getrandbits(32_000), a >> 33_536, b >> 33_536  # Of 32,000 bits each.
        long = ((a, b, c), *reference(a, b, c))
        for index, ((a, b, c), value, evidence) in enumerate((
                ((-dy, dx, dy - dx), DiophantineSolution(dx - 1, dy - 1, dx, dy), ()),  # The largest x, dx - 1.
                long,
                ((short_a, 0, short_a * k), DiophantineSolution(k, 0, 0, 1), ()),
                ((0, short_b, short_b * k), DiophantineSolution(0, k, 1, 0), ()))):
            with self.subTest(case=index):
                self.assertLessEqual(max(a.bit_length(), b.bit_length(), c.bit_length()), 65_536)
                self.assertNotEqual(value, NONE)
                result = support.run(OPERATION, a=a, b=b, c=c)
                self.assertClaim(result.steps[0].outputs, (value, evidence))
                self.assertTrue(mf.verify(result).verified)
                if index == 1:
                    self.roundtrip(result)
        # Unsolvable, with a common divisor of 30,000 bits that leaves c the remainder 1.
        g = rng.getrandbits(30_000) | 1 << 29_999
        a, b, c = g * rng.getrandbits(35_000), -g * rng.getrandbits(35_000), g * rng.getrandbits(35_000) + 1
        value, evidence = reference(a, b, c)
        self.assertEqual((value, evidence[0] % g), (NONE, 0))
        self.assertClaim(support.run(OPERATION, a=a, b=b, c=c).steps[0].outputs, (value, evidence))
        # The budget: the checker accepts the long solvable claim within 150,000 work units; the producer does not.
        (a, b, c), value, _ = long
        small = ComputationLimits(max_work=150_000)
        self.assertAccepts(OPERATION, arguments(a, b, c), value, limits=small)
        with self.assertRaises(ResourceLimitError):
            produced(a, b, c, limits=small)
        self.assertResourceLimit(support.run(OPERATION, a=a, b=b, c=c, limits=small))

    def test_short_coefficients_with_a_long_c_cost_little(self):
        """Regression: the producer charged its gcd of a and b by the length of c, which that gcd never reads, about
        16,000 work units where the check costs at most 10, so run() failed under max_work=100 on every shape."""
        small = ComputationLimits(max_work=100)
        c = 15 << 65_000  # 2**65000 = 1 modulo 3, so x = (c/3) / (6/3) = 5 * 2 = 1 modulo 3 (2 inverts 2).
        for (a, b, c), value, evidence in (
                ((6, 9, c), DiophantineSolution(1, (c - 6) // 9, 3, -2), ()),
                ((6, 9, 10 << 65_000), NONE, (3,)),  # 10 * 2**65000 = 1 modulo 3.
                ((0, 7, 7 << 65_000), DiophantineSolution(0, 1 << 65_000, 1, 0), ()),
                ((-5, 0, 5 << 65_000), DiophantineSolution(-(1 << 65_000), 0, 0, 1), ())):
            with self.subTest(a=a, b=b, c_bits=c.bit_length()):
                self.assertClaim(reference(a, b, c), (value, evidence))
                self.assertClaim(produced(a, b, c, limits=small), (value, evidence))
                result = support.run(OPERATION, a=a, b=b, c=c, limits=small)
                self.assertClaim(result.steps[0].outputs, (value, evidence))
                self.assertTrue(mf.verify(result).verified)

    def test_no_inverse_when_b_divides_c(self):
        """Regression: the producer charged and ran the extended Euclid of its inverse also when b divides c, where
        x = 0 needs none: 188,422 work units for a homogeneous equation of 65,536 bits, whose check costs 32,778. It
        now pays 45,062, and 14,270 for c = b*m with 32,768-bit coefficients (81,400 before, 8,202 for the check), so
        run() verifies each under max_work=60,000. The values are built backwards: for coprime p > 0 and q, a = -q
        and b = p (or both negated) have the canonical step (p, q), and b | c gives x = 0 and y = c/b."""
        rng = random.Random(4_096)
        cases = []
        for bits in (65_536, 32_768):
            p = rng.getrandbits(bits) | 1 << (bits - 1)
            q = -(rng.getrandbits(bits) | 1 << (bits - 1))
            while gcd(p, q) != 1:
                q += 1
            if bits == 65_536:
                cases += [((-q, p, 0), DiophantineSolution(0, 0, p, q)),
                          ((q, -p, 0), DiophantineSolution(0, 0, p, q)),
                          ((-q, p, -p), DiophantineSolution(0, -1, p, q))]
            else:
                m = -rng.getrandbits(32_000)
                cases += [((-q, p, p * m), DiophantineSolution(0, m, p, q)),
                          ((q, -p, p * m), DiophantineSolution(0, -m, p, q))]
        limits = ComputationLimits(max_work=60_000)
        for index, ((a, b, c), value) in enumerate(cases):
            with self.subTest(case=index):
                self.assertLessEqual(max(a.bit_length(), b.bit_length(), c.bit_length()), 65_536)
                self.assertClaim(produced(a, b, c, limits=limits), (value, ()))
                result = support.run(OPERATION, a=a, b=b, c=c, limits=limits)
                self.assertClaim(result.steps[0].outputs, (value, ()))
                self.assertTrue(mf.verify(result).verified)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen."""
        # Any common divisor of a and b that misses c proves an equation unsolvable; the producer hands over the
        # greatest, gcd(a, b).
        for (a, b, c), divisors in (((12, 18, 9), (2, 6)), ((30, 42, 7), (2, 3, 6)), ((0, 12, 3), (2, 4, 6, 12)),
                                    ((8, 0, 6), (4, 8)), ((-12, -18, 9), (2, 6))):
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(produced(a, b, c), (NONE, (max(divisors),)))
                for g in divisors:
                    self.assertAccepts(OPERATION, arguments(a, b, c), NONE, (g,))
        # Solution families built backwards from a canonical form, never by solving: a primitive step (p, q) with
        # p > 0, a common factor k and a sign s give a = -s*k*q and b = s*k*p, whose gcd is k and whose canonical step
        # is (p, q); then any 0 <= x < p and any y give c = a*x + b*y. For b = 0: any x and c = a*x.
        rng = random.Random(1112)
        for case in range(200):
            bits = rng.choice((2, 8, 32, 128, 512, 2048))
            if case % 8 == 0:
                a, b = rng.choice((1, -1)) * rng.randrange(1, 1 << bits), 0
                value = DiophantineSolution(rng.randrange(-(1 << bits), 1 << bits), 0, 0, 1)
            else:
                p, q = rng.randrange(1, 1 << bits), rng.randrange(-(1 << bits), 1 << bits)
                while gcd(p, q) != 1:
                    q += 1
                k, s = rng.randrange(1, 1 << rng.randrange(1, bits + 1)), rng.choice((1, -1))
                a, b = -s * k * q, s * k * p
                value = DiophantineSolution(rng.randrange(p), rng.randrange(-(1 << bits), 1 << bits), p, q)
            c = a * value.x + b * value.y
            with self.subTest(case=case, bits=bits):
                self.assertAccepts(OPERATION, arguments(a, b, c), value)
                self.assertClaim(produced(a, b, c), (value, ()))  # The canonical form is unique.

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        # Every produced claim, corrupted in one respect at a time (corruptions()).
        equations = [args for args, _, _ in CARD + BY_HAND] + list(seeded_equations(3, 60, (4, 64, 512)))
        for index, (a, b, c) in enumerate(equations):
            with self.subTest(case=index):
                value, evidence = produced(a, b, c)
                self.assertIs(support.verdict(family, OPERATION, arguments(a, b, c), value, evidence), True)
                for corrupted, other in corruptions(value, evidence):
                    self.assertRejects(OPERATION, arguments(a, b, c), corrupted, other)
        # The other corruptions, each next to the true claim it spoils.
        solvable, unsolvable = arguments(6, 9, 15), arguments(6, 9, 10)
        for args, value, evidence in (
                (solvable, support.impostor(SOLVED), ()), (unsolvable, support.impostor(NONE), (3,)),
                (solvable, support.forged(DiophantineSolution, x=1, y=1, dx=3), ()),  # A field left unset.
                (unsolvable, support.forged(DiophantineSolution), (3,)),
                (solvable, (1, 1, 3, -2), ()), (unsolvable, None, (3,)),
                (solvable, forged(True, 1, 3, -2), ()), (solvable, forged(1, 1, 3, -2.0), ()),
                (solvable, forged("1", 1, 3, -2), ()), (solvable, forged(1, None, 3, -2), ()),
                (unsolvable, NONE, (True,)), (unsolvable, NONE, (3.0,)), (unsolvable, NONE, ("3",)),
                (unsolvable, NONE, [3]), (solvable, SOLVED, []), (solvable, forged([1], 1, [3], -2), ()),
                (unsolvable, NONE, ((3,),)), (solvable, SOLVED, None), (unsolvable, NONE, 3),
                (arguments(True, 0, 1), DiophantineSolution(1, 0, 0, 1), ()),  # A bool argument.
                (arguments(0, 0, 0), DiophantineSolution(0, 0, 1, 0), ())):  # a = b = 0: invalid input.
            with self.subTest(value=type(value).__qualname__, evidence=repr(evidence)[:40]):
                self.assertRejects(OPERATION, args, value, evidence)
        for operation in ("integer_gcd", "continued_fraction", "LINEAR_DIOPHANTINE", "", None):
            with self.subTest(operation=operation):
                self.assertRejects(operation, solvable, SOLVED)
        # Oversized forged values and evidence are refused, or exhaust the budget; never accepted.
        for args, value, evidence in ((unsolvable, NONE, (support.HUGE,)), (unsolvable, NONE, (3,) * 1_000_000),
                                      (solvable, forged(support.HUGE, 1, 3, -2), ()),
                                      (solvable, forged(1, 1, 3 << 2_000_000, -2 << 2_000_000), ())):
            with self.subTest(value=type(value).__qualname__, evidence=len(evidence)):
                self.assertRejects(OPERATION, args, value, evidence, may_exhaust=True)
        # mf.verify of a tampered Result: the trace holds, the mathematics fails.
        for (a, b, c), value, evidence in (((6, 9, 15), DiophantineSolution(2, 1, 3, -2), None),
                                           ((6, 9, 15), DiophantineSolution(1, 1, 3, 2), None),
                                           ((6, 9, 10), NONE, (5,)), ((6, 9, 10), NONE, (1,)),
                                           ((6, 9, 15), NONE, (3,))):
            with self.subTest(a=a, b=b, c=c, value=value, evidence=evidence):
                tampered = support.tampered(support.run(OPERATION, a=a, b=b, c=c), value, evidence)
                report = support.statuses(mf.verify(tampered))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        results = [support.run(OPERATION, a=a, b=b, c=c) for (a, b, c), _, _ in CARD + BY_HAND]
        decoded = [mf.loads(mf.dumps(result)) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for index, (result, copy) in enumerate(zip(results, decoded)):
                with self.subTest(case=index):
                    self.assertTrue(mf.verify(result).verified)
                    self.assertTrue(mf.verify(copy).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(OPERATION, a=6, b=9, c=15)
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX."""
        for (a, b, c), text, latex in RENDERINGS:
            with self.subTest(a=a, b=b, c=c):
                decoded = self.roundtrip(support.run(OPERATION, a=a, b=b, c=c))
                self.assertEqual(mf.render_text(decoded.value), text)
                self.assertEqual(mf.to_latex(decoded.value), latex)
        self.assertEqual(mf.render_text(support.run(OPERATION, a=6, b=9, c=15)),
                         "completed / value [exact; complete]: (1 + 3*t, 1 - 2*t) for every integer t")
        self.assertEqual(mf.render_text(support.run(OPERATION, a=6, b=9, c=10)),
                         "completed / value [exact; complete]: no integer solution")
        self.assertEqual(mf.to_data(SOLVED)["data"],
                         {"type": "v6_diophantine_solution", "x": {"type": "integer", "value": "1"},
                          "y": {"type": "integer", "value": "1"}, "dx": {"type": "integer", "value": "3"},
                          "dy": {"type": "integer", "value": "-2"}})
        self.assertEqual(mf.to_data(NONE)["data"],
                         {"type": "v6_diophantine_solution", "x": {"type": "none"}, "y": {"type": "none"},
                          "dx": {"type": "none"}, "dy": {"type": "none"}})
        # A document of a record that is not canonical, or that mixes None and integers, does not decode.
        for field, item in (("x", {"type": "integer", "value": "4"}), ("dx", {"type": "integer", "value": "-3"}),
                            ("y", {"type": "none"})):
            document = mf.to_data(SOLVED)
            document["data"][field] = item
            with self.subTest(field=field), self.assertRaisesRegex(mf.SerializationError,
                                                                   "^Invalid mathematical document$"):
                mf.from_data(document)
        # The shared integer helper renders integers that str() refuses (over 4,300 digits).
        long = self.roundtrip(support.run(OPERATION, a=0, b=1, c=10 ** 5000))
        self.assertEqual(mf.render_text(long.value), "(t, 1" + "0" * 5000 + ") for every integer t")
        self.assertEqual(mf.to_latex(long.value), r"(t,\ 1" + "0" * 5000 + r"),\ t \in \mathbb{Z}")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        cases = [args for args, _, _ in CARD + BY_HAND] + list(seeded_equations(7, 40, (4, 64, 1024)))
        for index, (a, b, c) in enumerate(cases):
            with self.subTest(case=index):
                structured = support.run(OPERATION, a=a, b=b, c=c)
                self.assertTrue(mf.verify(structured).verified)
                self.assertEqual(support.lean_value(OPERATION, a=a, b=b, c=c), structured.value)
                self.assertEqual(support.value(OPERATION, a=a, b=b, c=c), structured.value)
        # Invalid input: the lean path leaves the call to the structured path, whose error the value function raises.
        self.assertIs(support.lean_value(OPERATION, a=0, b=0, c=0), lean.FALLBACK)
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OPERATION, a=0, b=0, c=0)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for (a, b, c), value, _ in CARD:
            with self.subTest(a=a, b=b, c=c):
                result = self.answer(support.call(OPERATION, {"a": a, "b": b, "c": c}))
                self.assertEqual(result, support.run(OPERATION, a=a, b=b, c=c))
                self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)
        # Integers from 2**53 on travel as strings of decimal digits, as call_values encodes them.
        a, b, c = 6 * 3 ** 100, 9 * 3 ** 100, 15 * 3 ** 100
        result = self.answer(support.call_values(OPERATION, a=a, b=b, c=c))
        self.assertEqual(result, support.run(OPERATION, a=a, b=b, c=c))
        self.assertEqual(result.value, SOLVED)
        self.assertTrue(mf.verify(result).verified)

    def test_seeded_agreement_with_slow_references(self):
        """Every equation with |a|, |b|, |c| <= 6 (2,184 of them) against enumerated(); 400 seeded equations of 4 to
        4,096 bits against reference(), and against enumerated() where they are small; 30 seeded equations whose step
        dx has about 5,990 to 16,384 bits, on both sides of the 6,000 bits of dx from which kernels.integers.inverse
        takes Lehmer's algorithm instead of the builtin pow, against reference(). Each time the producer succeeds, the
        checker accepts its claim, and the claim equals the reference's, evidence included."""
        span = range(-6, 7)
        small = [(a, b, c) for a in span for b in span for c in span if a or b]
        for a, b, c in small:
            with self.subTest(a=a, b=b, c=c):
                value, evidence = produced(a, b, c)
                self.assertEqual((value, evidence), enumerated(a, b, c))
                self.assertEqual((value, evidence), reference(a, b, c))
                self.assertIs(support.verdict(family, OPERATION, arguments(a, b, c), value, evidence), True)
        self.assertEqual(len(small), 2_184)
        shapes = Counter()
        for index, (a, b, c) in enumerate(seeded_equations(20_261_113, 400, (4, 8, 16, 64, 256, 1024, 4096))):
            with self.subTest(case=index, bits=max(a.bit_length(), b.bit_length(), c.bit_length())):
                value, evidence = produced(a, b, c)
                shapes[shape(a, b, c, value)] += 1
                self.assertClaim((value, evidence), reference(a, b, c))
                if max(abs(a), abs(b), abs(c)) <= 1_000:
                    self.assertEqual((value, evidence), enumerated(a, b, c))
                self.assertIs(support.verdict(family, OPERATION, arguments(a, b, c), value, evidence), True)
        for name in ("unsolvable", "a = 0", "b = 0", "c = 0", "general"):  # Every case of the card is reached.
            self.assertGreaterEqual(shapes[name], 20, shapes)
        # Long steps: b has exactly the drawn length and gcd(a, b) is mostly small, so dx = |b|/gcd(a, b) is about as
        # long; c is a multiple of gcd(a, b), so the producer reaches the inverse every time.
        rng = random.Random(6_000)
        lehmer = Counter()
        for case in range(30):
            bits = (5_990, 6_000, 6_010, 8_192, 12_000, 16_384)[case % 6]
            a = rng.choice((1, -1)) * rng.getrandbits(bits)
            b = rng.choice((1, -1)) * (rng.getrandbits(bits) | 1 << (bits - 1))
            c = rng.choice((1, -1)) * gcd(a, b) * rng.getrandbits(bits - 64)
            with self.subTest(case=case, bits=bits):
                value, evidence = produced(a, b, c)
                lehmer[value.dx.bit_length() >= 6_000] += 1
                self.assertClaim((value, evidence), reference(a, b, c))
                self.assertIs(support.verdict(family, OPERATION, arguments(a, b, c), value, evidence), True)
        self.assertGreaterEqual(min(lehmer[True], lehmer[False]), 5, lehmer)  # Both sides of the switch.
