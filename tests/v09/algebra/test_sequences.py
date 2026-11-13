"""The sequences family of mathforge.algebra: sturm_sequence, cyclotomic_polynomial.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register algebra sequences

Every expected value comes from the card or from a reference of this module, never from the producer: Sturm sequences
by the Euclidean algorithm over fractions.Fraction, cyclotomic polynomials by Moebius products of binomials x^e - 1 in
Z[x] (multiplication and exact division by x^e - 1, each linear), cross-checked against long division of x^n - 1 by
the cyclotomic polynomials of the proper divisors for small n and against closed forms (primes, prime powers). Neither
operation has a deferred input (no Unevaluated result): every valid input has its value, which only the budget, or the
card's limits on n, keep from it. tests/v09/algebra/test_sequences_checker.py tests the checker on its own.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.algebra import sequences as family
from mathforge.contracts import CheckStatus, ErrorInfo, ExecutionStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Integers, Rational, Reals, Symbol, Truth, sqrt
from mathforge.polynomial import Polynomial

from .. import support

F = Fraction
Q = Rational
X = Symbol("x")
T = Symbol("t", Integers, (Assumption("positive", Truth.TRUE),))
STURM, CYCLOTOMIC = "sturm_sequence", "cyclotomic_polynomial"
FAST = 2.0  # Seconds: a request beyond its budget fails well within this.
STAND_INS = (0.5, True, "1", None)  # No exact number and no record: a float, a bool, a str and None.
VERIFIED, FAILED = CheckStatus.VERIFIED, CheckStatus.FAILED

Number = int | Fraction | Rational


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value of the tests comes from these or from the card.

def frac(number: Number) -> Fraction:
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def trimmed(values: Sequence[Fraction]) -> list[Fraction]:
    """The coefficients without trailing zeros; [] for zero."""
    result = list(values)
    while result and result[-1] == 0:
        result.pop()
    return result


def long_division(dividend: Sequence[Fraction], divisor: Sequence[Fraction]) -> tuple[list[Fraction], list[Fraction]]:
    """(quotient, remainder) over Q, lowest degree first, the divisor nonzero and no longer than the dividend."""
    rest = list(dividend)
    quotient = [F(0)] * (len(rest) - len(divisor) + 1)
    for index in range(len(quotient) - 1, -1, -1):
        factor = rest[index + len(divisor) - 1] / divisor[-1]
        quotient[index] = factor
        for offset, coefficient in enumerate(divisor):
            rest[index + offset] -= factor * coefficient
    return quotient, trimmed(rest[:len(divisor) - 1])


def sturm_reference(coefficients: Sequence[Number]) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """The card's Sturm sequence of a nonzero polynomial, with no scaling, and the quotients of its divisions."""
    p = trimmed([frac(c) for c in coefficients])
    assert p
    derivative = trimmed([k * c for k, c in enumerate(p)][1:])
    if not derivative:
        return [p], []
    members, quotients = [p, derivative], []
    while True:
        quotient, remainder = long_division(members[-2], members[-1])
        quotients.append(quotient)
        if not remainder:
            return members, quotients
        members.append([-c for c in remainder])


def primes_of(n: int) -> list[int]:
    found, rest, d = [], n, 2
    while d * d <= rest:
        if not rest % d:
            found.append(d)
            while not rest % d:
                rest //= d
        d += 1
    return found + ([rest] if rest > 1 else [])


def totient(n: int) -> int:
    result = n
    for p in primes_of(n):
        result = result // p * (p - 1)
    return result


def moebius(n: int) -> int:
    primes = primes_of(n)
    for p in primes:
        if not n % (p * p):
            return 0
    return -1 if len(primes) % 2 else 1


def times_binomial(values: list[int], e: int) -> list[int]:
    """values * (x^e - 1)."""
    result = [-v for v in values] + [0] * e
    for k, v in enumerate(values):
        result[k + e] += v
    return result


def over_binomial(values: list[int], e: int) -> list[int]:
    """values / (x^e - 1), asserted exact: Q_k = Q_(k-e) - A_k."""
    quotient = [0] * (len(values) - e)
    for k in range(len(quotient)):
        quotient[k] = (quotient[k - e] if k >= e else 0) - values[k]
    assert times_binomial(quotient, e) == values
    return quotient


def cyclotomic_moebius(n: int) -> tuple[int, ...]:
    """Phi_n = prod_{d | n} (x^d - 1)^mu(n/d), every product first, then every exact division."""
    divisors = [d for d in range(1, n + 1) if not n % d]
    values = [1]
    for d in divisors:
        if moebius(n // d) == 1:
            values = times_binomial(values, d)
    for d in divisors:
        if moebius(n // d) == -1:
            values = over_binomial(values, d)
    return tuple(values)


def cyclotomic_division(n: int) -> tuple[int, ...]:
    """Phi_n by long division of x^n - 1 by Phi_d for every proper divisor d (slow; small n only)."""
    rest = [F(-1)] + [F(0)] * (n - 1) + [F(1)]
    for d in range(1, n):
        if not n % d:
            rest, remainder = long_division(rest, [F(c) for c in cyclotomic_division(d)])
            assert not remainder
    assert all(c.denominator == 1 for c in rest)
    return tuple(int(c) for c in rest)


def rationals(values: Sequence[Number]) -> tuple[Rational, ...]:
    return tuple(Q(frac(v).numerator, frac(v).denominator) for v in values)


def poly(values: Sequence[Number], variable: Symbol = X) -> Polynomial:
    return Polynomial(rationals(values) or (Q(0),), variable)


def sturm_value(coefficients: Sequence[Number], variable: Symbol = X) -> tuple[tuple[Polynomial, ...], tuple]:
    """The reference's value and evidence as the family's layout holds them."""
    members, quotients = sturm_reference(coefficients)
    return tuple(poly(m, variable) for m in members), tuple(rationals(q) for q in quotients)


def cyclotomic_value(n: int, variable: Symbol = X) -> tuple[Polynomial, tuple]:
    divisors = [d for d in range(1, n) if not n % d]
    return poly(cyclotomic_moebius(n), variable), tuple((d, cyclotomic_moebius(d)) for d in divisors)


def random_rational(rng: random.Random, bits: int) -> Fraction:
    return F(rng.randrange(-(2 ** bits), 2 ** bits + 1), rng.randrange(1, 2 ** bits + 1))


def random_polynomial(rng: random.Random, degree: int, bits: int, rational: bool) -> list[Fraction]:
    values = [random_rational(rng, bits) if rational else F(rng.randrange(-(2 ** bits), 2 ** bits + 1))
              for _ in range(degree + 1)]
    while values[-1] == 0:
        values[-1] = F(rng.randrange(1, 10))
    return values


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins

class Integer(int):
    """An int subclass, equal to the right number but no exact int."""


class Subpolynomial(Polynomial):
    """A Polynomial subclass: the right fields, another class."""


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def forged_poly(coefficients: object, variable: object = X) -> Polynomial:
    return support.forged(Polynomial, coefficients=coefficients, variable=variable)


def replaced(items: tuple, index: int, item: object) -> tuple:
    return items[:index] + (item,) + items[index + 1:]


class SequencesTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object, evidence: object) -> None:
        """A completed v6 Result of that value and evidence, with its four checks, which mf.verify accepts."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertEqual(result.value, value)
        self.assertEqual(result.steps[0].outputs[1], evidence)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertSturm(self, coefficients: Sequence[Number], variable: Symbol = X) -> None:
        """produce() answers the reference's Sturm sequence and quotients, every member in the request's variable
        itself and every quotient coefficient a canonical Rational, and the checker accepts the claim."""
        args = {"polynomial": poly(coefficients, variable)}
        value, evidence = family.produce(STURM, dict(args))
        expected_value, expected_evidence = sturm_value(coefficients, variable)
        self.assertEqual((value, evidence), (expected_value, expected_evidence))
        self.assertIs(type(value), tuple)
        self.assertTrue(all(type(m) is Polynomial and m.variable is variable for m in value))
        self.assertTrue(all(type(q) is tuple and all(type(c) is Rational for c in q) for q in evidence))
        self.assertIs(support.verdict(family, STURM, args, value, evidence), True)

    def assertCyclotomic(self, n: int, variable: Symbol = X) -> None:
        args = {"n": n, "variable": variable}
        value, evidence = family.produce(CYCLOTOMIC, dict(args))
        self.assertEqual((value, evidence), cyclotomic_value(n, variable))
        self.assertIs(value.variable, variable)
        self.assertTrue(all(type(d) is int and type(c) is tuple and all(type(k) is int for k in c)
                            for d, c in evidence))
        self.assertIs(support.verdict(family, CYCLOTOMIC, args, value, evidence), True)

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_references_agree_with_closed_forms(self):
        """The references agree with each other and with closed forms, so the expected values below are right."""
        for n in range(1, 60):
            self.assertEqual(cyclotomic_moebius(n), cyclotomic_division(n), n)
            self.assertEqual(len(cyclotomic_moebius(n)) - 1, totient(n))
        for p in (2, 3, 5, 7, 31, 101):
            self.assertEqual(cyclotomic_moebius(p), (1,) * p)
            for k in (2, 3):
                if p ** k <= 2000:
                    spread = [0] * ((p - 1) * p ** (k - 1) + 1)
                    spread[::p ** (k - 1)] = [1] * p
                    self.assertEqual(cyclotomic_moebius(p ** k), tuple(spread))
        # A Sturm sequence counts real roots: x^3 - x has 3, x^2 + 1 none (sign changes at -oo minus at +oo).
        for coefficients, roots in (((0, -1, 0, 1), 3), ((1, 0, 1), 0), ((-2, 0, 1), 2), ((6, -5, 1), 2)):
            members, _ = sturm_reference(coefficients)

            def changes(signs: list[int]) -> int:
                signs = [s for s in signs if s]
                return sum(1 for a, b in zip(signs, signs[1:]) if a != b)
            minus = changes([(1 if m[-1] > 0 else -1) * (-1) ** (len(m) - 1) for m in members])
            plus = changes([1 if m[-1] > 0 else -1 for m in members])
            self.assertEqual(minus - plus, roots, coefficients)

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts: the card's, with the evidence
        layouts of the checker, a rational polynomial, and a variable with a domain and an assumption."""
        sturm = (
            ((-2, 0, 1), ((-2, 0, 1), (0, 2), (2,)), ((0, F(1, 2)), (0, 1))),
            ((0, -1, 0, 1), ((0, -1, 0, 1), (-1, 0, 3), (0, F(2, 3)), (1,)),
             ((0, F(1, 3)), (0, F(9, 2)), (0, F(2, 3)))),
            ((5,), ((5,),), ()),
            ((-3, 1), ((-3, 1), (1,)), ((-3, 1),)),
        )
        for coefficients, members, quotients in sturm:
            with self.subTest(polynomial=coefficients):
                value = tuple(poly(m) for m in members)
                evidence = tuple(rationals(q) for q in quotients)
                self.assertEqual(sturm_value(coefficients), (value, evidence))
                result = support.run(STURM, polynomial=poly(coefficients))
                self.assertVerified(result, value, evidence)
                self.assertIs(type(result.value), tuple)
                self.roundtrip(result)
        cyclotomic = ((1, (-1, 1)), (2, (1, 1)), (6, (1, -1, 1)), (12, (1, 0, -1, 0, 1)))
        for n, coefficients in cyclotomic:
            with self.subTest(n=n):
                self.assertEqual(cyclotomic_moebius(n), coefficients)
                result = support.run(CYCLOTOMIC, n=n, variable=X)
                self.assertVerified(result, poly(coefficients), cyclotomic_value(n)[1])
                self.assertEqual(result.for_, X)
                self.roundtrip(result)
        self.assertEqual(support.run(CYCLOTOMIC, n=1, variable=X).steps[0].outputs[1], ())
        self.assertEqual(support.run(CYCLOTOMIC, n=6, variable=X).steps[0].outputs[1],
                         ((1, (-1, 1)), (2, (1, 1)), (3, (1, 1, 1))))
        result = support.run(CYCLOTOMIC, n=105, variable=X)
        self.assertVerified(result, *cyclotomic_value(105))
        coefficients = result.value.coefficients
        self.assertEqual((len(coefficients), coefficients[7], coefficients[41]), (49, Q(-2), Q(-2)))
        self.assertEqual([k for k, c in enumerate(coefficients) if c == Q(-2)], [7, 41])
        self.roundtrip(result)
        # A rational polynomial and a variable with a domain and an assumption, which every member keeps.
        for coefficients, variable in (((F(1, 2), F(-3, 4), 1), X), ((0, -1, 0, 1), T), ((F(-7, 3), 0, 0, 0, 2), T)):
            with self.subTest(polynomial=[str(c) for c in coefficients], variable=variable.name):
                result = support.run(STURM, polynomial=poly(coefficients, variable))
                self.assertVerified(result, *sturm_value(coefficients, variable))
                self.assertTrue(all(member.variable == variable for member in result.value))
                self.roundtrip(result)
        result = support.run(CYCLOTOMIC, n=12, variable=T)
        self.assertVerified(result, *cyclotomic_value(12, T))
        self.assertEqual(result.value.variable, T)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so are the card's cases (the zero polynomial, n < 1), Polynomials and Symbols made with
        object.__new__ outside their canonical form, which run() admits by their class, and a matrix, which no
        argument takes. produce() tests the types itself too."""
        matrix = support.oversized_matrix()
        cubic = poly((0, -1, 0, 1))
        for item in (*STAND_INS, 5, Q(5), (Q(1), Q(1)), [Q(1)], X, X ** 2 - 2, matrix, "x^2 - 2",
                     {"text": "x", "variable": "x"}, Subpolynomial((Q(1), Q(1)), X), support.impostor(cubic),
                     poly((0,)), forged_poly((Q(0),)), forged_poly(()), forged_poly((Q(1), Q(0))),
                     forged_poly((1, 2)), forged_poly([Q(1), Q(1)]), forged_poly((Q(1), 0.5)),
                     forged_poly((Q(1), True)), forged_poly((forged_rational(2, 4), Q(1))),
                     forged_poly((forged_rational(1, -2), Q(1))), forged_poly((forged_rational(1, 0), Q(1))),
                     forged_poly((forged_rational(True, 1), Q(1))), forged_poly((Q(1),) * 1026),
                     forged_poly((Q(1), Q(1)), "x"), forged_poly((Q(1), Q(1)), support.impostor(X)),
                     forged_poly((Q(1), Q(1)), support.forged(Symbol, name="x", domain=Reals, assumptions=(), uid=1))):
            with self.subTest(argument="polynomial", item=type(item).__name__):
                self.assertInvalidInput(support.run(STURM, polynomial=item))
                with self.assertRaises(InvalidInput):
                    family.produce(STURM, {"polynomial": item})
        zero = support.run(STURM, polynomial=poly((0,)))
        self.assertIn("nonzero polynomial", zero.error.message)
        for item in (*STAND_INS, 0, -1, -7, -(2 ** 80), Integer(6), Q(6), F(6), (6,), [6], matrix, "6", X, 6.0):
            with self.subTest(argument="n", item=repr(item)[:40]):
                self.assertInvalidInput(support.run(CYCLOTOMIC, n=item, variable=X))
                with self.assertRaises(InvalidInput):
                    family.produce(CYCLOTOMIC, {"n": item, "variable": X})
        self.assertIn("n >= 1", support.run(CYCLOTOMIC, n=0, variable=X).error.message)
        for item in (*STAND_INS, "x", 1, Q(1), cubic, matrix, support.impostor(X), sqrt(2),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=(), uid=1)):
            with self.subTest(argument="variable", item=type(item).__name__):
                self.assertInvalidInput(support.run(CYCLOTOMIC, n=6, variable=item))
        # produce() refuses forged Symbols that run()'s inspection of the arguments, or the failure Result it builds
        # around the variable, meets first; the checker reads the same fields from the instance itself.
        for item in (*STAND_INS, "x", support.impostor(X), support.forged(Symbol, name="x"),
                     support.forged(Symbol, name="x", domain="reals", assumptions=(), uid=X.uid),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=[], uid=X.uid),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=("positive",), uid=X.uid),
                     support.forged(Symbol, name="x", domain=Reals, assumptions=(Assumption("positive"),) * 4,
                                    uid=X.uid)):
            with self.subTest(argument="variable", item=type(item).__name__, path="produce"):
                with self.assertRaises(InvalidInput):
                    family.produce(CYCLOTOMIC, {"n": 6, "variable": item})
                with self.assertRaises(InvalidInput):
                    family.produce(STURM, {"polynomial": forged_poly((Q(1), Q(1)), item)})
        # A Polynomial without fields: run()'s inspection of the arguments fails on it before produce() runs.
        for item in (support.forged(Polynomial), support.forged(Polynomial, variable=X),
                     support.forged(Polynomial, coefficients=(Q(1),))):
            with self.subTest(polynomial="no fields", path="produce"), self.assertRaises(InvalidInput):
                family.produce(STURM, {"polynomial": item})
        # Every field of its own and of its exact type: a Symbol made with object.__new__ is then a Symbol.
        plain = support.forged(Symbol, name="x", domain=Reals, assumptions=(), uid=X.uid)
        self.assertEqual(family.produce(CYCLOTOMIC, {"n": 2, "variable": plain})[0], poly((1, 1), X))
        for operation, args in ((STURM, {"polynomial": cubic}), (CYCLOTOMIC, {"n": 6, "variable": X})):
            for wrong in (dict(args, extra=1), {}, tuple(args.items()), None, list(args.items()),
                          {"polynomial": cubic, "n": 6, "variable": X}):
                with self.subTest(operation=operation, arguments=repr(wrong)[:40]), self.assertRaises(InvalidInput):
                    family.produce(operation, wrong)
        for operation in ("Sturm_sequence", "sturm_sequence ", "sturm", "cyclotomic", "interpolate", "", None, 1):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, {"polynomial": cubic})
        with self.assertRaises(mf.OperationError) as raised:
            support.value(CYCLOTOMIC, n=0, variable=X)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in n or in a coefficient, n > 100,000 and phi(n) > 1,024 (the card's limits), and
        max_work=1 exceed the budget. No argument takes a matrix: support.oversized_matrix() is invalid input
        (test_invalid_inputs). A member longer than max_integer_bits, and a dense polynomial whose unscaled Sturm
        sequence outgrows the default budget, fail at once."""
        for n in (support.HUGE, -support.HUGE, 100_001, 10 ** 6, 2 ** 64, 1031, 4096, 2 * 1031, 9240):
            with self.subTest(n=n if n.bit_length() < 100 else "huge"):
                self.assertResourceLimit(support.run(CYCLOTOMIC, n=n, variable=X))
                with self.assertRaises(ResourceLimitError):
                    family.produce(CYCLOTOMIC, {"n": n, "variable": X})
        self.assertGreater(totient(9240), 1024)
        self.assertEqual(totient(4620), 960)
        self.assertEqual(totient(2048), 1024)  # The largest degree a Polynomial holds: a value.
        self.assertVerified(support.run(CYCLOTOMIC, n=2048, variable=X), *cyclotomic_value(2048))
        wide = ComputationLimits(max_integer_bits=70_000)
        with computation(wide, fresh=True):
            huge = [poly((support.HUGE, 1)), poly((1, F(1, support.HUGE), 1)), poly((-support.HUGE, 0, 3))]
        for item in huge:
            with self.subTest(polynomial=len(item.coefficients)):
                self.assertResourceLimit(support.run(STURM, polynomial=item))
                with self.assertRaises(ResourceLimitError):
                    family.produce(STURM, {"polynomial": item})
        for operation, args in ((STURM, {"polynomial": poly((-2, 0, 1))}), (STURM, {"polynomial": poly((5,))}),
                                (CYCLOTOMIC, {"n": 1, "variable": X}), (CYCLOTOMIC, {"n": 105, "variable": X})):
            with self.subTest(operation=operation, max_work=1, args=repr(args)[:40]):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **args))
                with computation(ComputationLimits(max_work=1), fresh=True), self.assertRaises(ResourceLimitError):
                    family.produce(operation, dict(args))
        # Members whose integers outgrow max_integer_bits: no Rational of the request's limits holds them.
        rng = random.Random(20261301)
        dense = poly(random_polynomial(rng, 12, 3, False))
        small = ComputationLimits(max_integer_bits=64)
        self.assertResourceLimit(support.run(STURM, polynomial=dense, limits=small))
        with computation(small, fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(STURM, {"polynomial": dense})
        # A dense polynomial of degree 70: the producer stops at its budget well within FAST seconds.
        start = time.perf_counter()
        self.assertResourceLimit(support.run(STURM, polynomial=poly(random_polynomial(rng, 70, 3, False))))
        self.assertResourceLimit(support.run(STURM, polynomial=poly(random_polynomial(rng, 40, 3, False)),
                                             limits=ComputationLimits(max_work=50_000)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """Value and evidence are unique for both operations, so a true claim that the producer did not make is one
        built by hand from the references, in other variables and on other polynomials; the checker accepts each, and
        the producer, asked, answers the same."""
        rng = random.Random(20261302)
        for case in range(30):
            degree = rng.randrange(0, 9)
            coefficients = random_polynomial(rng, degree, rng.choice((2, 6)), bool(case % 2))
            variable = (X, T, Symbol("y"))[case % 3]
            args = {"polynomial": poly(coefficients, variable)}
            with self.subTest(case=case):
                self.assertAccepts(STURM, args, *sturm_value(coefficients, variable))
                self.assertSturm(coefficients, variable)
        for n in (1, 2, 3, 4, 9, 30, 64, 105, 210, 385, 1001, 2310):
            for variable in (T, Symbol("z")):
                with self.subTest(n=n, variable=variable.name):
                    self.assertAccepts(CYCLOTOMIC, {"n": n, "variable": variable}, *cyclotomic_value(n, variable))
                    self.assertCyclotomic(n, variable)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, an int where a Rational belongs, a list for a tuple, missing or extra evidence, a
        wrong operation name and oversized forged evidence, all on Results of the producer; mf.verify fails the
        mathematics of a Result whose value or evidence was replaced, while its trace stays verified. v0.7's
        root_isolation.sturm_sequence, a primitive chain, is not the card's sequence."""
        from mathforge import root_isolation

        coefficients = (0, -1, 0, 1)
        args = {"polynomial": poly(coefficients)}
        result = support.run(STURM, **args)
        value, evidence = result.value, result.steps[0].outputs[1]
        self.assertVerified(result, *sturm_value(coefficients))
        older = root_isolation.sturm_sequence(args["polynomial"])
        self.assertNotEqual(older, value)
        quotient = evidence[0]
        values = [
            ("v0.7's primitive chain", older),
            ("a member scaled", replaced(value, 2, poly((0, 2)))),
            ("the last member dropped", value[:-1]), ("an extra member", value + (poly((1,)),)),
            ("p1 negated", replaced(value, 1, poly((1, 0, -3)))),
            ("another Symbol named x", tuple(poly(m) for m in ((0, -1, 0, 1),)) + tuple(
                Polynomial(m.coefficients, Symbol("x")) for m in value[1:])),
            ("a list", list(value)), ("None", None), ("the polynomial", value[0]),
            ("an impostor member", replaced(value, 3, support.impostor(value[3]))),
            ("a member subclass", replaced(value, 3, Subpolynomial(value[3].coefficients, X))),
            ("int coefficients", replaced(value, 3, forged_poly((1,)))),
            ("a trailing zero", replaced(value, 3, forged_poly((Q(1), Q(0))))),
            ("a coefficient in higher terms", replaced(value, 2, forged_poly((Q(0), forged_rational(4, 6))))),
        ]
        for label, item in values:
            with self.subTest(value=label):
                self.assertRejects(STURM, args, item, evidence)
        evidences = [
            ("None", None), ("a list", list(evidence)), ("missing", evidence[:-1]), ("extra", evidence + (quotient,)),
            ("a quotient wrong", replaced(evidence, 1, (Q(0), Q(9)))), ("ints", replaced(evidence, 0, (0, Q(1, 3)))),
            ("a list quotient", replaced(evidence, 0, list(quotient))),
            ("oversized", (quotient,) * 100_000),
            ("a giant", replaced(evidence, 0, (Q(0), forged_rational(2 ** 70_000 + 1, 3 * 2 ** 70_000)))),
        ]
        for label, item in evidences:
            with self.subTest(evidence=label):
                self.assertRejects(STURM, args, value, item, may_exhaust=label == "oversized")
        for operation in ("Sturm_sequence", "sturm", CYCLOTOMIC, "", None):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)
        report = support.statuses(mf.verify(support.tampered(result, older)))
        self.assertIs(report["result.trace"], VERIFIED)
        self.assertIs(report["result.mathematics"], FAILED)
        report = support.statuses(mf.verify(support.tampered(result, value, replaced(evidence, 1, (Q(0), Q(9))))))
        self.assertIs(report["result.mathematics"], FAILED)
        # cyclotomic_polynomial on Phi_105, whose coefficient -2 turned into -1.
        args = {"n": 105, "variable": X}
        result = support.run(CYCLOTOMIC, **args)
        value, evidence = result.value, result.steps[0].outputs[1]
        wrong = list(value.coefficients)
        wrong[7] = Q(-1)
        values = [
            ("a -2 as -1", Polynomial(tuple(wrong), X)), ("Phi_35", poly(cyclotomic_moebius(35))),
            ("x^105 - 1 over Phi_1", poly(over_binomial([-1] + [0] * 104 + [1], 1))),
            ("another Symbol named x", Polynomial(value.coefficients, Symbol("x"))),
            ("int coefficients", forged_poly(tuple(int(frac(c)) for c in value.coefficients))),
            ("an impostor", support.impostor(value)), ("the coefficients", value.coefficients), ("None", None),
        ]
        for label, item in values:
            with self.subTest(value=label):
                self.assertRejects(CYCLOTOMIC, args, item, evidence)
        evidences = [
            ("None", None), ("a list", list(evidence)), ("missing", evidence[:-1]), ("extra", evidence + ((105, ()),)),
            ("reversed", evidence[::-1]), ("Rationals", replaced(evidence, 0, (1, rationals((-1, 1))))),
            ("a wrong Phi_3", replaced(evidence, 1, (3, (1, 2, 1)))), ("oversized", ((1, (-1, 1)),) * 100_000),
        ]
        for label, item in evidences:
            with self.subTest(evidence=label):
                self.assertRejects(CYCLOTOMIC, args, value, item)
        self.assertRejects(CYCLOTOMIC, {"n": 104, "variable": X}, value, evidence)
        report = support.statuses(mf.verify(support.tampered(result, Polynomial(tuple(wrong), X))))
        self.assertIs(report["result.trace"], VERIFIED)
        self.assertIs(report["result.mathematics"], FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [(STURM, {"polynomial": poly(c)}) for c in ((-2, 0, 1), (0, -1, 0, 1), (5,), (-3, 1),
                                                               (F(1, 2), F(-3, 4), 1), (1, 0, 0, 0, 0, -3, 2))]
        requests.append((STURM, {"polynomial": poly((0, -1, 0, 1), T)}))
        requests += [(CYCLOTOMIC, {"n": n, "variable": v}) for n, v in ((1, X), (6, X), (105, T), (4620, X))]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, args), result, text in zip(requests, results, texts):
                with self.subTest(operation=operation, args=repr(args)[:50]):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            failed = support.run(CYCLOTOMIC, n=6, variable=X)  # The patch is live.
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a tuple of
        Polynomials (the members separated by ", "), a single member, rational members, and a Polynomial."""
        rendered = (
            (STURM, {"polynomial": poly((-2, 0, 1))}, "(x^2 - 2, 2*x, 2)", r"\left(x^{2} - 2, 2\, x, 2\right)"),
            (STURM, {"polynomial": poly((0, -1, 0, 1))}, "(x^3 - x, 3*x^2 - 1, 2/3*x, 1)",
             r"\left(x^{3} - x, 3\, x^{2} - 1, \frac{2}{3}\, x, 1\right)"),
            (STURM, {"polynomial": poly((5,))}, "(5)", r"\left(5\right)"),
            (STURM, {"polynomial": poly((-3, 1), Symbol("t"))}, "(t - 3, 1)", r"\left(t - 3, 1\right)"),
            (STURM, {"polynomial": poly((F(1, 2), F(-3, 4), 1))}, "(x^2 - 3/4*x + 1/2, 2*x - 3/4, -23/64)",
             r"\left(x^{2} - \frac{3}{4}\, x + \frac{1}{2}, 2\, x - \frac{3}{4}, -\frac{23}{64}\right)"),
            (CYCLOTOMIC, {"n": 1, "variable": X}, "x - 1", "x - 1"),
            (CYCLOTOMIC, {"n": 2, "variable": X}, "x + 1", "x + 1"),
            (CYCLOTOMIC, {"n": 6, "variable": X}, "x^2 - x + 1", "x^{2} - x + 1"),
            (CYCLOTOMIC, {"n": 12, "variable": Symbol("t")}, "t^4 - t^2 + 1", "t^{4} - t^{2} + 1"),
        )
        for operation, args, text, latex in rendered:
            with self.subTest(text=text):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual((mf.render_text(result.value), mf.to_latex(result.value)), (text, latex))
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        phi = support.run(CYCLOTOMIC, n=105, variable=X).value
        self.assertIn("- 2*x^41", mf.render_text(phi))
        self.assertIn(r"- 2\, x^{7}", mf.to_latex(phi))
        data = mf.to_data(support.run(STURM, polynomial=poly((-3, 1))).value)["data"]
        self.assertEqual((data["type"], [item["type"] for item in data["items"]]), ("tuple", ["polynomial"] * 2))
        self.assertEqual(data["items"][0]["coefficients"], [{"type": "rational", "numerator": "-3", "denominator": "1"},
                                                            {"type": "rational", "numerator": "1", "denominator": "1"}])
        self.assertEqual({item["variable"]["uid"] for item in data["items"]}, {X.uid})

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path; on an
        input without a value the lean path falls back and the value function raises the structured path's error."""
        rng = random.Random(20261303)
        cases = [(STURM, {"polynomial": poly(random_polynomial(rng, rng.randrange(0, 8), 4, bool(k % 2)),
                                             (X, T)[k % 2])}) for k in range(12)]
        cases += [(CYCLOTOMIC, {"n": rng.randrange(1, 400), "variable": (X, T)[k % 2]}) for k in range(12)]
        cases.append((CYCLOTOMIC, {"n": 4620, "variable": X}))
        for operation, args in cases:
            with self.subTest(operation=operation, args=repr(args)[:60]):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, **args), expected)
        for operation, args, code in ((STURM, {"polynomial": poly((0,))}, "invalid_input"),
                                      (CYCLOTOMIC, {"n": 0, "variable": X}, "invalid_input"),
                                      (CYCLOTOMIC, {"n": 1031, "variable": X}, "resource_limit_exceeded")):
            with self.subTest(code=code, operation=operation):
                self.assertIs(support.lean_value(operation, **args), lean.FALLBACK)
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(operation, **args)
                self.assertEqual(raised.exception.code, code)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: the polynomial travels
        as {"text", "variable"}, n as a JSON integer and the variable as its name."""
        for operation, args in ((STURM, {"polynomial": poly((0, -1, 0, 1))}),
                                (STURM, {"polynomial": poly((F(1, 2), F(-3, 4), 1), Symbol("t"))}),
                                (CYCLOTOMIC, {"n": 105, "variable": X}),
                                (CYCLOTOMIC, {"n": 1, "variable": Symbol("y")})):
            with self.subTest(operation=operation, args=repr(args)[:50]):
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call(STURM, {"polynomial": {"text": "x^2 - 2", "variable": "x"}}))
        self.assertEqual([m.coefficients for m in result.value], [rationals(c) for c in ((-2, 0, 1), (0, 2), (2,))])
        self.assertTrue(mf.verify(result).verified)
        result = self.answer(support.call(CYCLOTOMIC, {"n": 12, "variable": "t"}))
        self.assertEqual((result.value.coefficients, result.value.variable.name), (rationals((1, 0, -1, 0, 1)), "t"))
        self.assertTrue(mf.verify(result).verified)
        errors = (
            (STURM, {"polynomial": {"text": "0", "variable": "x"}}, "invalid_input", "nonzero polynomial"),
            (CYCLOTOMIC, {"n": 0, "variable": "x"}, "invalid_input", "n >= 1"),
            (CYCLOTOMIC, {"n": 1031, "variable": "x"}, "resource_limit_exceeded", "phi(n) <= 1024"),
            (CYCLOTOMIC, {"n": 100_001, "variable": "x"}, "resource_limit_exceeded", "n <= 100000"),
        )
        for operation, request, code, text in errors:
            with self.subTest(error=str(request)[:60]):
                self.assertIn(text, self.assertToolError(support.call(operation, request), code)["message"])
        self.assertToolError(support.call(CYCLOTOMIC, {"n": 6, "variable": "x"}, limits={"max_work": 1}),
                             "resource_limit_exceeded")

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references on at least 200 seeded inputs per operation, and the checker accepts
        every answer: random integer and rational polynomials of degree 0 to 14, sparse polynomials of high degree and
        polynomials with repeated roots (whose sequence ends at their gcd with the derivative); and n over every shape
        of factorization with phi(n) <= 1,024 (primes, prime powers, square-free and mixed n)."""
        rng = random.Random(20261304)
        count = 0
        while count < 220:
            kind = count % 4
            if kind == 0:
                coefficients = random_polynomial(rng, rng.randrange(0, 15), rng.choice((1, 3, 8)), False)
            elif kind == 1:
                coefficients = random_polynomial(rng, rng.randrange(0, 9), rng.choice((2, 6)), True)
            elif kind == 2:  # Sparse, of high degree: up to three terms up to x^300 (four terms often outgrow the
                degree = rng.randrange(1, 300)  # budget, for the checker too: their sequences fill in).
                coefficients = [F(0)] * (degree + 1)
                for _ in range(rng.randrange(1, 3)):
                    coefficients[rng.randrange(0, degree)] = random_rational(rng, 4)
                coefficients[-1] = F(rng.randrange(1, 9), rng.randrange(1, 4))
            else:  # Repeated roots: a product of (x - r)^k.
                coefficients = [F(rng.randrange(1, 5))]
                for _ in range(rng.randrange(1, 4)):
                    root = F(rng.randrange(-6, 7), rng.randrange(1, 4))
                    for _ in range(rng.randrange(1, 4)):
                        coefficients = [(coefficients[k - 1] if k else 0) - root * (coefficients[k] if k < len(
                            coefficients) else 0) for k in range(len(coefficients) + 1)]
            with self.subTest(case=count, kind=kind):
                self.assertSturm(coefficients, (X, T)[count % 2])
            count += 1
        candidates = [n for n in range(1, 4621) if totient(n) <= 1024]
        chosen = sorted(set(rng.sample(candidates, 190)) | {1, 2, 4, 1024, 2048, 3 ** 6, 2310, 4290, 4620, 1021,
                                                             2 * 1021})
        self.assertGreaterEqual(len(chosen), 200)
        for n in chosen:
            with self.subTest(n=n):
                self.assertCyclotomic(n)
        self.assertGreater(sum(1 for n in chosen if moebius(n) == 0), 40)  # Not square-free.
        self.assertGreater(sum(1 for n in chosen if len(primes_of(n)) >= 3), 40)

    # -----------------------------------------------------------------------------------------------------------
    # The producer's paths and work

    def test_work_is_charged_before_it_runs(self):
        """The producer charges the budget before each step: its work on an input is the same on every run, a budget
        one unit short of it refuses the input, and longer numbers cost more. The measured work of the module
        docstring holds: a random integer polynomial of degree 40, and the costliest n with phi(n) <= 1,024. Nothing
        in the family module calls the builtin pow or uses **, which could not be interrupted."""
        rng = random.Random(20261305)
        for operation, args in ((STURM, {"polynomial": poly(random_polynomial(rng, 20, 3, False))}),
                                (CYCLOTOMIC, {"n": 2310, "variable": X})):
            works = []
            for _ in range(2):
                with computation(fresh=True) as budget:
                    family.produce(operation, dict(args))
                works.append(budget.work)
            with self.subTest(operation=operation):
                self.assertEqual(works[0], works[1])
                with computation(ComputationLimits(max_work=works[0]), fresh=True):
                    family.produce(operation, dict(args))
                with computation(ComputationLimits(max_work=works[0] - 1), fresh=True):
                    with self.assertRaises(ResourceLimitError):
                        family.produce(operation, dict(args))
        base = random_polynomial(rng, 10, 3, False)
        costs = []
        for scale in (1, 2 ** 3000 + 1):
            with computation(fresh=True) as budget:
                family.produce(STURM, {"polynomial": poly([c * scale for c in base])})
            costs.append(budget.work)
        self.assertGreater(costs[1], 2 * costs[0])
        with computation(fresh=True) as budget:
            family.produce(STURM, {"polynomial": poly(random_polynomial(rng, 40, 3, False))})
        self.assertLess(budget.work, 400_000)  # Measured about 200,000 units.
        with computation(fresh=True) as budget:
            family.produce(CYCLOTOMIC, {"n": 4290, "variable": X})
        self.assertLess(budget.work, 250_000)  # Measured 218,207 units.
        tree = ast.parse(inspect.getsource(family))
        called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func,
                                                                                                         ast.Name)}
        self.assertNotIn("pow", called)
        self.assertEqual([node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                          and isinstance(node.op, ast.Pow)], [])

    def test_sparse_divisions_cost_their_nonzero_terms(self):
        """A degree drop by a sparse divisor is cheap: x^1000 - x has the sequence (x^1000 - x, 1000 x^999 - 1,
        999/1000 x, 1), whose third division, by 999/1000 x, a pseudo-division would make a thousand passes over a
        thousand coefficients."""
        coefficients = [0, -1] + [0] * 998 + [1]
        start = time.perf_counter()
        with computation(fresh=True) as budget:
            value, evidence = family.produce(STURM, {"polynomial": poly(coefficients)})
        self.assertLess(budget.work, 20_000)
        self.assertEqual(len(value), 4)
        self.assertEqual(value[2], poly((0, F(999, 1000))))
        self.assertEqual(value[3], poly((1,)))
        self.assertEqual((value, evidence), sturm_value(coefficients))
        self.assertVerified(support.run(STURM, polynomial=poly(coefficients)), value, evidence)
        self.assertLess(time.perf_counter() - start, FAST)

    def test_cyclotomic_paths(self):
        """Every way the producer builds Phi_d gives the reference: d = 1, a prime p (Phi_1(x^p) / Phi_1), a
        square-free d = m p from Phi_m, and a d with a square factor from its radical by substituting x^(d/r)."""
        for n in (1, 2, 97, 1021, 6, 30, 210, 2310, 4, 8, 1024, 2048, 9, 729, 12, 360, 1800, 4620):
            with self.subTest(n=n):
                self.assertCyclotomic(n)

    def test_every_order_within_the_degree_bound(self):
        """Review regression (completeness at the card's limits): 4,620 is the largest n <= 100,000 with
        phi(n) <= 1,024, and for each of the 2,003 orders with phi(n) <= 1,024 the producer answers a claim that the
        checker accepts under the default budget; orders above 4,620 are refused with ResourceLimitError."""
        top = family.MAX_ORDER
        phi = list(range(top + 1))  # A totient sieve.
        for p in range(2, top + 1):
            if phi[p] == p:
                for k in range(p, top + 1, p):
                    phi[k] -= phi[k] // p
        orders = [n for n in range(1, top + 1) if phi[n] <= 1024]
        self.assertEqual((len(orders), orders[-1]), (2003, 4620))
        checker = support.checker(family)
        for n in orders:
            args = {"n": n, "variable": X}
            value, evidence = family.produce(CYCLOTOMIC, dict(args))
            if support.verdict(family, CYCLOTOMIC, args, value, evidence) is not True:
                self.fail(f"the checker refuses the producer's claim on n = {n}")
            self.assertEqual(len(value.coefficients), phi[n] + 1)
        self.assertEqual(checker.MAX_ORDER, top)
        for n in random.Random(20261306).sample(range(4621, top + 1), 60):
            with self.subTest(n=n), self.assertRaises(ResourceLimitError):
                family.produce(CYCLOTOMIC, {"n": n, "variable": X})
