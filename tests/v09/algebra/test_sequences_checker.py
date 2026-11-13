"""The independent checker of mathforge.areas.algebra.sequences (mathforge.certify.algebra.sequences), tested on its own
before the family's producer exists.

Every true claim is computed here, never by the producer: Sturm sequences by the Euclidean algorithm over
fractions.Fraction written in this module, cyclotomic polynomials by exact long division of x^n - 1 by the cyclotomic
polynomials of the proper divisors of n (a different route from the checker's Moebius products), checked against
closed forms (primes, prime powers) and the card's known values. The evidence follows the card's layouts. The checker
answers False to every corrupted claim, never with an exception, and refuses an oversized forgery within FAST seconds
under the default budget; a ResourceLimitError is accepted only for n > 100,000 and for a request integer over
max_integer_bits, where the card and the verifier's inspection put the resource limit.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
from math import gcd
import random
import time

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.algebra import sequences as family
from mathforge.certify.algebra import sequences as checker
from mathforge.contracts import CheckStatus
from mathforge.errors import UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Integers, Rational, Rationals, Reals, Symbol, Truth
from mathforge.polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets, "
         "decimal digits, Sturm sequences and cyclotomic polynomials")
X = Symbol("x")
Y = Symbol("y", Integers, (Assumption("positive", Truth.TRUE),))
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
LARGE = 2 ** 60_000 + 1  # Within max_integer_bits.

Number = int | Fraction


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these.

def trimmed(values: Sequence[Fraction]) -> list[Fraction]:
    """The coefficients without trailing zeros; [] for zero."""
    result = list(values)
    while result and result[-1] == 0:
        result.pop()
    return result


def long_division(dividend: Sequence[Fraction], divisor: Sequence[Fraction]) -> tuple[list[Fraction], list[Fraction]]:
    """(quotient, remainder) of polynomials over Q, coefficients lowest degree first, divisor nonzero."""
    rest = [Fraction(value) for value in dividend]
    if len(rest) < len(divisor):
        return [], trimmed(rest)
    quotient = [Fraction(0)] * (len(rest) - len(divisor) + 1)
    for index in range(len(quotient) - 1, -1, -1):
        factor = rest[index + len(divisor) - 1] / divisor[-1]
        quotient[index] = factor
        for offset, coefficient in enumerate(divisor):
            rest[index + offset] -= factor * coefficient
    return quotient, trimmed(rest[:len(divisor) - 1])


def sturm_reference(coefficients: Sequence[Number]) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """The card's canonical Sturm sequence of a nonzero polynomial and the quotients of its divisions."""
    p = trimmed([Fraction(value) for value in coefficients])
    assert p, "the zero polynomial has no Sturm sequence"
    members = [p]
    derivative = trimmed([index * value for index, value in enumerate(p)][1:])
    if not derivative:
        return members, []
    members.append(derivative)
    quotients = []
    while True:
        quotient, remainder = long_division(members[-2], members[-1])
        quotients.append(quotient)
        if not remainder:
            return members, quotients
        members.append([-value for value in remainder])


_CYCLOTOMIC: dict[int, tuple[int, ...]] = {}


def monic_division(dividend: Sequence[int], divisor: Sequence[int]) -> list[int]:
    """The quotient of integer polynomials by a monic divisor that divides exactly (asserted)."""
    assert divisor[-1] == 1
    rest = list(dividend)
    quotient = [0] * (len(rest) - len(divisor) + 1)
    terms = [(offset, coefficient) for offset, coefficient in enumerate(divisor[:-1]) if coefficient]
    top = len(divisor) - 1
    for index in range(len(quotient) - 1, -1, -1):
        factor = rest[index + top]
        quotient[index] = factor
        if factor:
            for offset, coefficient in terms:
                rest[index + offset] -= factor * coefficient
    assert not any(rest[:top])
    return quotient


def cyclotomic(n: int) -> tuple[int, ...]:
    """Phi_n, integer coefficients lowest degree first: x^n - 1 divided exactly by Phi_d for every proper divisor d
    (each monic, so the division stays in the integers)."""
    if n not in _CYCLOTOMIC:
        rest = [-1] + [0] * (n - 1) + [1]
        for d in range(1, n):
            if not n % d:
                rest = monic_division(rest, cyclotomic(d))
        _CYCLOTOMIC[n] = tuple(rest)
    return _CYCLOTOMIC[n]


def totient(n: int) -> int:
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


def rational(value: Number) -> Rational:
    value = Fraction(value)
    return Rational(value.numerator, value.denominator)


def poly(values: Sequence[Number], variable: Symbol = X) -> Polynomial:
    return Polynomial(tuple(rational(value) for value in values) or (Rational(0),), variable)


def forged_rational(numerator: object, denominator: object = 1) -> Rational:
    """A Rational made with object.__new__, so that its slots may hold anything (no __init__ reduces them)."""
    return support.forged(Rational, numerator=numerator, denominator=denominator)


def forged_poly(coefficients: object, variable: object = X) -> Polynomial:
    """A Polynomial made with object.__new__: its fields may break the canonical form."""
    return support.forged(Polynomial, coefficients=coefficients, variable=variable)


def sturm_claim(coefficients: Sequence[Number], variable: Symbol = X) -> tuple[dict[str, object], tuple, tuple]:
    """(args, value, evidence) of the true claim on a polynomial, in the card's layouts."""
    members, quotients = sturm_reference(coefficients)
    args = {"polynomial": poly(coefficients, variable)}
    value = tuple(poly(member, variable) for member in members)
    evidence = tuple(tuple(rational(item) for item in quotient) for quotient in quotients)
    return args, value, evidence


def cyclotomic_claim(n: int, variable: Symbol = X) -> tuple[dict[str, object], Polynomial, tuple]:
    """(args, value, evidence) of the true claim on n, in the card's layouts."""
    evidence = tuple((d, cyclotomic(d)) for d in range(1, n) if not n % d)
    return {"n": n, "variable": variable}, poly(cyclotomic(n), variable), evidence


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    # run() records a Symbol variable argument as for_, with the domain Reals.
    variable = {"for_": args["variable"], "domain": Reals} if "variable" in args else {}
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,), **variable)


def replaced(items: tuple, index: int, item: object) -> tuple:
    return items[:index] + (item,) + items[index + 1:]


class Integer(int):
    """An int subclass, equal to the right answer but no exact int."""


Arguments = type("Arguments", (dict,), {})  # A dict subclass.
Evidence = type("Evidence", (tuple,), {})  # A tuple subclass.

CUBIC = sturm_claim([0, -1, 0, 1])  # x^3 - x: (x^3 - x, 3x^2 - 1, 2/3 x, 1).
SIX = cyclotomic_claim(6)


class SequencesCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        expected = (
            OperationSpec("sturm_sequence", "algebra", (Argument("polynomial", "Polynomial", None, "polynomial"),),
                          "tuple[Polynomial, ...]", "Compute the Sturm sequence of a nonzero rational polynomial.",
                          SCOPE, version=6),
            OperationSpec("cyclotomic_polynomial", "algebra",
                          (Argument("n", "int", None, "int"), Argument("variable", "Symbol", None, "symbol")),
                          "Polynomial", "Compute the n-th cyclotomic polynomial.", SCOPE, version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {})  # Polynomial records and tuples of them: no record class of its own.
        self.assertEqual(family.CHECKER, "mathforge.certify.algebra.sequences")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)
        self.assertEqual(checker.MAX_DEGREE, MAX_POLYNOMIAL_DEGREE)  # The checker cannot import the constant.
        self.assertEqual(checker.MAX_ORDER, family.MAX_ORDER)
        with self.assertRaises(UnsupportedOperation):  # An operation of another family.
            family.produce("roots", {"polynomial": poly([1, 1])})

    def test_references_agree_with_closed_forms(self):
        """The test's own references: Sturm sequences by hand, cyclotomic polynomials of primes and prime powers."""
        self.assertEqual(sturm_reference([-2, 0, 1]), ([[-2, 0, 1], [0, 2], [2]], [[0, Fraction(1, 2)], [0, 1]]))
        for p in (2, 3, 5, 7, 13):
            self.assertEqual(cyclotomic(p), (1,) * p)
            self.assertEqual(cyclotomic(p * p), tuple(1 if k % p == 0 else 0 for k in range(p * (p - 1) + 1)))
        for n in range(1, 60):
            self.assertEqual(len(cyclotomic(n)) - 1, totient(n))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_sturm_sequence_accepts_the_known_values(self):
        third = Fraction(2, 3)
        known = [
            ([-2, 0, 1], [[-2, 0, 1], [0, 2], [2]]),
            ([0, -1, 0, 1], [[0, -1, 0, 1], [-1, 0, 3], [0, third], [1]]),
            ([5], [[5]]),
            ([-3, 1], [[-3, 1], [1]]),
        ]
        for coefficients, members in known:
            with self.subTest(polynomial=coefficients):
                args, value, evidence = sturm_claim(coefficients)
                self.assertEqual(value, tuple(poly(member) for member in members))
                self.assertAccepts("sturm_sequence", args, value, evidence)
        self.assertEqual(CUBIC[2], ((Rational(0), Rational(1, 3)), (Rational(0), Rational(9, 2)),
                                    (Rational(0), Rational(2, 3))))
        self.assertEqual(sturm_claim([5])[2], ())
        self.assertEqual(sturm_claim([-3, 1])[2], ((Rational(-3), Rational(1)),))

    def test_sturm_sequence_accepts_reference_claims(self):
        h = Fraction(1, 2)
        cases = [
            [0, 0, 1],  # x^2: (x^2, 2x), the remainder is zero at once.
            [-1, 3, -3, 1],  # (x - 1)^3: p' divides p.
            [Fraction(-7, 3)],  # A rational constant.
            [Fraction(3, 4), Fraction(-5, 2)],  # A linear polynomial with a negative leading coefficient.
            [4, -4, -3, 2, 1],  # (x - 1)^2 (x + 2)^2 ... with a repeated factor: the sequence ends in their gcd.
            [h, 0, -3, 0, 1, Fraction(-1, 7)],
            [1, 0, 0, 0, 0, 0, 0, 1],  # x^7 + 1
            [-1, 0, 0, 0, 1],  # x^4 - 1
        ]
        rng = random.Random(20261113)
        for degree in (3, 5, 8, 12, 16):
            cases.append([Fraction(rng.randrange(-30, 31), rng.randrange(1, 6)) for _ in range(degree)]
                         + [Fraction(rng.choice((-3, -1, 1, 2)), rng.randrange(1, 4))])
        for coefficients in cases:
            for variable in (X, Y):
                with self.subTest(polynomial=coefficients, variable=variable.name):
                    self.assertAccepts("sturm_sequence", *sturm_claim(coefficients, variable))

    def test_cyclotomic_polynomial_accepts_the_known_values(self):
        known = {1: (-1, 1), 2: (1, 1), 6: (1, -1, 1), 12: (1, 0, -1, 0, 1)}
        for n, coefficients in known.items():
            with self.subTest(n=n):
                args, value, evidence = cyclotomic_claim(n)
                self.assertEqual(value, poly(coefficients))
                self.assertAccepts("cyclotomic_polynomial", args, value, evidence)
        phi105 = cyclotomic(105)
        self.assertEqual(len(phi105) - 1, 48)
        self.assertEqual([k for k, value in enumerate(phi105) if value == -2], [7, 41])
        self.assertAccepts("cyclotomic_polynomial", *cyclotomic_claim(105))
        self.assertEqual(cyclotomic_claim(1)[2], ())
        self.assertEqual(SIX[2], ((1, (-1, 1)), (2, (1, 1)), (3, (1, 1, 1))))

    def test_cyclotomic_polynomial_accepts_reference_claims(self):
        for n in (3, 4, 8, 9, 16, 30, 36, 64, 97, 210, 256, 385, 1023, 1024):
            for variable in (X, Y):
                with self.subTest(n=n, variable=variable.name):
                    self.assertAccepts("cyclotomic_polynomial", *cyclotomic_claim(n, variable))

    def test_larger_claims_fit_the_default_budget(self):
        """The largest orders that a record holds: 4,620 (phi = 960, 48 divisors, the largest n with phi(n) <= 1,024)
        and 4,080 (phi = 1,024); and a Sturm sequence of degree 30 with rational coefficients."""
        start = time.perf_counter()
        for n in (4620, 4080):
            with self.subTest(n=n):
                args, value, evidence = cyclotomic_claim(n)
                self.assertEqual(len(value.coefficients) - 1, totient(n))
                self.assertAccepts("cyclotomic_polynomial", args, value, evidence)
                self.assertRejects("cyclotomic_polynomial", args, value, evidence[:-1] + evidence[-1:] * 2)
        rng = random.Random(7)
        args, value, evidence = sturm_claim([Fraction(rng.randrange(-9, 10), rng.randrange(1, 4)) for _ in range(30)]
                                            + [1])
        self.assertEqual(len(value), 31)
        self.assertAccepts("sturm_sequence", args, value, evidence)
        self.assertRejects("sturm_sequence", args, value[:-1], evidence[:-1])
        self.assertLess(time.perf_counter() - start, 5 * FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Claims on inputs outside the domain, and malformed requests

    def test_rejects_wrong_operations_and_arguments(self):
        args, value, evidence = CUBIC
        cases = [
            ("cyclotomic_polynomial", args, value, evidence),
            ("Sturm_sequence", args, value, evidence),
            ("sturm", args, value, evidence),
            ("", args, value, evidence),
            (None, args, value, evidence),
            ("sturm_sequence", Arguments(args), value, evidence),
            ("sturm_sequence", list(args.items()), value, evidence),
            ("sturm_sequence", {}, value, evidence),
            ("sturm_sequence", dict(args, variable=X), value, evidence),
            ("sturm_sequence", {"p": args["polynomial"]}, value, evidence),
            ("sturm_sequence", {1: args["polynomial"]}, value, evidence),
            ("sturm_polynomial", SIX[0], SIX[1], SIX[2]),
            ("cyclotomic_polynomial", dict(SIX[0], extra=1), SIX[1], SIX[2]),
            ("cyclotomic_polynomial", {"n": 6}, SIX[1], SIX[2]),
            ("cyclotomic_polynomial", Arguments(SIX[0]), SIX[1], SIX[2]),
        ]
        for index, (operation, case_args, case_value, case_evidence) in enumerate(cases):
            with self.subTest(case=index):
                self.assertRejects(operation, case_args, case_value, case_evidence)

    def test_sturm_sequence_rejects_inputs_outside_the_domain(self):
        _, value, evidence = CUBIC
        zero = poly([0])
        self.assertRejectsAll("sturm_sequence", [
            ("zero polynomial", {"polynomial": zero}, (zero,), ()),
            ("int", {"polynomial": 5}, (poly([5]),), ()),
            ("Rational", {"polynomial": Rational(5)}, (poly([5]),), ()),
            ("expression", {"polynomial": X}, value, evidence),
            ("unreduced coefficient", {"polynomial": forged_poly((Rational(0), Rational(-1), Rational(0),
                                                                 forged_rational(2, 2)))}, value, evidence),
            ("trailing zero", {"polynomial": forged_poly((Rational(0), Rational(-1), Rational(0), Rational(1),
                                                         Rational(0)))}, value, evidence),
            ("int coefficient", {"polynomial": forged_poly((0, -1, 0, 1))}, value, evidence),
            ("list of coefficients", {"polynomial": forged_poly([Rational(0), Rational(-1), Rational(0),
                                                                Rational(1)])}, value, evidence),
            ("str variable", {"polynomial": forged_poly(CUBIC[0]["polynomial"].coefficients, "x")}, value, evidence),
            ("impostor polynomial", {"polynomial": support.impostor(CUBIC[0]["polynomial"])}, value, evidence),
        ])

    def test_cyclotomic_polynomial_rejects_inputs_outside_the_domain(self):
        _, value, evidence = SIX
        self.assertRejectsAll("cyclotomic_polynomial", [
            ("n = 0", {"n": 0, "variable": X}, poly([1]), ()),
            ("n = -6", {"n": -6, "variable": X}, value, evidence),
            ("bool", {"n": True, "variable": X}, poly([-1, 1]), ()),
            ("float", {"n": 6.0, "variable": X}, value, evidence),
            ("str", {"n": "6", "variable": X}, value, evidence),
            ("Rational", {"n": Rational(6), "variable": X}, value, evidence),
            ("int subclass", {"n": Integer(6), "variable": X}, value, evidence),
            ("None", {"n": None, "variable": X}, value, evidence),
            ("str variable", {"n": 6, "variable": "x"}, value, evidence),
            ("forged variable", {"n": 6, "variable": support.forged(Symbol, name="x", uid=X.uid)}, value, evidence),
        ])

    def test_cyclotomic_polynomial_limits(self):
        """phi(n) > 1,024 for n <= 100,000: no record holds Phi_n, so every claim is False. n > 100,000 is the card's
        resource limit, and a request integer over max_integer_bits that of the request's inspection."""
        for n in (1031, 3465, 99991):  # A prime; 3^2*5*7*11 (phi 1,440); a prime below 100,000.
            with self.subTest(n=n):
                start = time.perf_counter()
                big = forged_poly(tuple(Rational(1) for _ in range(min(n, 2000))))
                self.assertRejects("cyclotomic_polynomial", {"n": n, "variable": X}, big, ())
                self.assertRejects("cyclotomic_polynomial", {"n": n, "variable": X}, poly([1] * 1025), ())
                self.assertLess(time.perf_counter() - start, FAST)
        for n in (100_001, 2 ** 61 - 1, 10 ** 1000, support.HUGE):
            with self.subTest(n=n.bit_length()):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, "cyclotomic_polynomial", {"n": n, "variable": X}, poly([1, 1]), ())
                self.assertLess(time.perf_counter() - start, FAST)
        self.assertAccepts("cyclotomic_polynomial", *cyclotomic_claim(97))

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims

    def test_sturm_sequence_rejects_corrupted_values(self):
        args, value, evidence = CUBIC
        p0, p1, p2, p3 = value
        x2 = sturm_claim([-2, 0, 1])
        other = Symbol("x")  # The same name, another identity.
        cases = [
            ("wrong member", args, (p0, p1, poly([0, 1]), p3), evidence),
            ("negated remainder", args, (p0, p1, poly([0, Fraction(-2, 3)]), poly([-1])), evidence),
            ("missing last member", args, value[:-1], evidence[:-1]),
            ("extra member", args, value + (poly([1]),), evidence + ((Rational(1),),)),
            ("swapped members", args, (p0, p2, p1, p3), evidence),
            ("another polynomial's sequence", args, x2[1], x2[2]),
            ("p0 is not p", args, (poly([0, -1, 0, 2]),) + value[1:], evidence),
            ("p1 is not p'", args, (p0, poly([-2, 0, 6])) + value[2:],
             ((Rational(0), Rational(1, 6)), (Rational(0), Rational(9)), (Rational(0), Rational(4, 3)))),
            # A positively scaled sequence is a Sturm sequence too, but not the card's canonical one.
            ("primitive members", args, (p0, p1, poly([0, 2]), p3), evidence),
            ("primitive members, own quotients", args, (p0, p1, poly([0, 2]), p3),
             ((Rational(0), Rational(1, 3)), (Rational(0), Rational(3, 2)), (Rational(0), Rational(2)))),
            ("empty", args, (), ()),
            ("constant, missing p'", sturm_claim([-3, 1])[0], (poly([-3, 1]),), ()),
            ("member in y", args, (p0, p1, poly([0, Fraction(2, 3)], Y), p3), evidence),
            ("member in another x", args, (p0, p1, poly([0, Fraction(2, 3)], other), p3), evidence),
            ("zero member", args, value + (poly([0]),), evidence + ((Rational(0),),)),
            ("list", args, list(value), evidence),
            ("tuple subclass", args, Evidence(value), evidence),
            ("impostor member", args, (p0, p1, support.impostor(p2), p3), evidence),
            ("impostor first member", args, (support.impostor(p0),) + value[1:], evidence),
            ("int member", args, (p0, p1, p2, 1), evidence),
            ("Rational member", args, (p0, p1, p2, Rational(1)), evidence),
            ("tuple member", args, (p0, p1, p2, (Rational(1),)), evidence),
            ("None", args, None, evidence),
            # Records forged with object.__new__, breaking the canonical form.
            ("unreduced coefficient", args, (p0, p1, forged_poly((Rational(0), forged_rational(4, 6))), p3), evidence),
            ("negative denominator", args, (p0, p1, forged_poly((Rational(0), forged_rational(-2, -3))), p3),
             evidence),
            ("trailing zero", args, (p0, p1, p2, forged_poly((Rational(1), Rational(0)))), evidence),
            ("forged zero", args, value + (forged_poly((Rational(0),)),), evidence + ((Rational(0),),)),
            ("empty coefficients", args, (p0, p1, p2, forged_poly(())), evidence),
            ("int coefficient", args, (p0, p1, p2, forged_poly((1,))), evidence),
            ("bool coefficient", args, (p0, p1, p2, forged_poly((True,))), evidence),
            ("float coefficient", args, (p0, p1, p2, forged_poly((1.0,))), evidence),
            ("str coefficient", args, (p0, p1, p2, forged_poly(("1",))), evidence),
            ("bool numerator", args, (p0, p1, p2, forged_poly((forged_rational(True),))), evidence),
            ("float denominator", args, (p0, p1, forged_poly((Rational(0), forged_rational(2, 3.0))), p3),
             evidence),
            ("list coefficients", args, (p0, p1, p2, forged_poly([Rational(1)])), evidence),
            ("str variable", args, (p0, p1, p2, forged_poly((Rational(1),), "x")), evidence),
            ("missing variable", args, (p0, p1, p2, support.forged(Polynomial, coefficients=(Rational(1),))),
             evidence),
            ("missing coefficients", args, (p0, p1, p2, support.forged(Polynomial, variable=X)), evidence),
        ]
        self.assertRejectsAll("sturm_sequence", cases)

    def test_sturm_sequence_rejects_corrupted_evidence(self):
        args, value, evidence = CUBIC
        q1, q2, q3 = evidence
        half = Rational(1, 2)
        cases = [
            ("missing", args, value, ()),
            ("one fewer", args, value, evidence[:-1]),
            ("one more", args, value, evidence + ((Rational(1),),)),
            ("wrong quotient", args, value, (q1, (Rational(0), Rational(9, 4)), q3)),
            ("wrong constant term", args, value, (q1, q2, (Rational(1), Rational(2, 3)))),
            ("swapped", args, value, (q2, q1, q3)),
            ("too long", args, value, (q1 + (Rational(0),), q2, q3)),
            ("too short", args, value, ((Rational(1, 3),), q2, q3)),
            ("leading zero", args, value, ((Rational(0), Rational(1, 3), Rational(0)), q2, q3)),
            ("int coefficients", args, value, ((0, half), q2, q3)),
            ("int in place of Rational(1)", sturm_claim([-3, 1])[0], sturm_claim([-3, 1])[1], ((-3, 1),)),
            ("bool", args, value, (q1, q2, (Rational(0), True))),
            ("float", args, value, (q1, q2, (Rational(0), 2 / 3))),
            ("str", args, value, (q1, q2, (Rational(0), "2/3"))),
            ("unreduced", args, value, (q1, q2, (Rational(0), forged_rational(4, 6)))),
            ("zero denominator", args, value, (q1, q2, (Rational(0), forged_rational(2, 0)))),
            ("list evidence", args, value, list(evidence)),
            ("tuple subclass", args, value, Evidence(evidence)),
            ("list quotient", args, value, (list(q1), q2, q3)),
            ("Polynomial quotient", args, value, (poly([0, Fraction(1, 3)]), q2, q3)),
            ("flat", args, value, q1 + q2 + q3),
            ("nested", args, value, (evidence,)),
            ("None", args, value, None),
            ("constant with a quotient", sturm_claim([5])[0], sturm_claim([5])[1], ((Rational(1),),)),
        ]
        # Forged evidence that is internally consistent but wrong: p2 replaced by p2 + p1 with q1 + 1 satisfies the
        # identity p0 + p2 == q1 * p1, but deg p2 is no longer below deg p1, so -p2 is no remainder.
        p0, p1, p2, p3 = value
        shifted = poly([-1, Fraction(2, 3), 3])
        cases.append(("consistent, degree not lower", args, (p0, p1, shifted, p3),
                      ((Rational(1), Rational(1, 3)), q2, q3)))
        # Every identity holds, and only the degrees are wrong: for x^2, (x^2, 2x, 2x) with the quotients x/2 + 1 and 1
        # (x^2 + 2x == (x/2 + 1) * 2x and 2x == 1 * 2x), but 2x is not of lower degree than 2x.
        cases.append(("consistent, equal degree", {"polynomial": poly([0, 0, 1])}, (poly([0, 0, 1]), poly([0, 2]),
                      poly([0, 2])), ((Rational(1), Rational(1, 2)), (Rational(1),))))
        # Every identity and degree holds, but p1 is p'/2: for x^2, (x^2, x) with the quotient x.
        cases.append(("consistent, p1 = p'/2", {"polynomial": poly([0, 0, 1])}, (poly([0, 0, 1]), poly([0, 1])),
                      ((Rational(0), Rational(1)),)))
        # The same for p0: (2x^2, 4x) with the quotient x/2 is the sequence of 2x^2, not of x^2.
        cases.append(("consistent, p0 = 2p", {"polynomial": poly([0, 0, 1])}, (poly([0, 0, 2]), poly([0, 4])),
                      ((Rational(0), Rational(1, 2)),)))
        # A sequence that skips the last division: (x^3 - x, 3x^2 - 1) with a "zero remainder" quotient fails.
        cases.append(("early end", args, value[:2], (q1,)))
        self.assertRejectsAll("sturm_sequence", cases)

    def test_cyclotomic_polynomial_rejects_corrupted_claims(self):
        args, value, evidence = SIX
        e1, e2, e3 = evidence
        twelve = cyclotomic_claim(12)
        flipped = ((1, (-1, 1)), (2, (-1, -1)), (3, (-1, -1, -1)))  # The identity at d = 6 holds; at d = 2 it fails.
        cases = [
            ("wrong value", args, poly([1, 1, 1]), evidence),
            ("negated value", args, poly([-1, 1, -1]), evidence),
            ("Phi_3", args, poly(cyclotomic(3)), evidence),
            ("x^6 - 1", args, poly([-1, 0, 0, 0, 0, 0, 1]), evidence),
            ("another n's claim", args, twelve[1], twelve[2]),
            ("other variable", args, poly(cyclotomic(6), Y), evidence),
            ("same name, other identity", args, poly(cyclotomic(6), Symbol("x")), evidence),
            ("impostor value", args, support.impostor(value), evidence),
            ("tuple value", args, (value,), evidence),
            ("coefficient tuple value", args, cyclotomic(6), evidence),
            ("None value", args, None, evidence),
            # Records forged with object.__new__, breaking the canonical form of the value.
            ("int coefficients", args, forged_poly((1, -1, 1)), evidence),
            ("bool coefficient", args, forged_poly((True, Rational(-1), Rational(1))), evidence),
            ("float coefficient", args, forged_poly((Rational(1), -1.0, Rational(1))), evidence),
            ("denominator 2", args, forged_poly((forged_rational(2, 2), Rational(-1), Rational(1))), evidence),
            ("bool numerator", args, forged_poly((forged_rational(True), Rational(-1), Rational(1))), evidence),
            ("trailing zero", args, forged_poly(value.coefficients + (Rational(0),)), evidence),
            ("list coefficients", args, forged_poly(list(value.coefficients)), evidence),
            ("missing variable", args, support.forged(Polynomial, coefficients=value.coefficients), evidence),
            # Evidence.
            ("missing", args, value, ()),
            ("missing a divisor", args, value, (e1, e3)),
            ("n included", args, value, evidence + ((6, cyclotomic(6)),)),
            ("non-divisor", args, value, (e1, e2, (4, cyclotomic(4)))),
            ("wrong order", args, value, (e2, e1, e3)),
            ("duplicate", args, value, (e1, e1, e2, e3)),
            ("wrong polynomial", args, value, (e1, (2, (1, 2)), e3)),
            ("wrong polynomial of the right degree", args, value, (e1, e2, (3, (1, -1, 1)))),
            ("consistent at n only", args, value, flipped),
            ("Rational coefficients", args, value, (e1, e2, (3, (Rational(1),) * 3))),
            ("Rational divisor", args, value, ((Rational(1), (-1, 1)), e2, e3)),
            ("bool divisor", args, value, ((True, (-1, 1)), e2, e3)),
            ("bool coefficient", args, value, ((1, (-1, True)), e2, e3)),
            ("float coefficient", args, value, ((1, (-1, 1.0)), e2, e3)),
            ("str coefficient", args, value, ((1, (-1, "1")), e2, e3)),
            ("int subclass", args, value, ((1, (-1, Integer(1))), e2, e3)),
            ("trailing zero", args, value, ((1, (-1, 1, 0)), e2, e3)),
            ("list evidence", args, value, list(evidence)),
            ("tuple subclass", args, value, Evidence(evidence)),
            ("list entry", args, value, ([1, (-1, 1)], e2, e3)),
            ("list coefficients", args, value, ((1, [-1, 1]), e2, e3)),
            ("Polynomial entry", args, value, ((1, poly([-1, 1])), e2, e3)),
            ("triple entry", args, value, ((1, (-1, 1), ()), e2, e3)),
            ("flat", args, value, (1, (-1, 1), 2, (1, 1), 3, (1, 1, 1))),
            ("None", args, value, None),
            ("n = 1 with evidence", {"n": 1, "variable": X}, poly([-1, 1]), ((1, (-1, 1)),)),
        ]
        self.assertRejectsAll("cyclotomic_polynomial", cases)

    def test_oversized_forgeries_fail_fast(self):
        args, value, evidence = CUBIC
        q1, q2, q3 = evidence
        p0, p1, p2, p3 = value
        six_args, six_value, six_evidence = SIX
        e1, e2, e3 = six_evidence
        large_member = poly([0, Fraction(LARGE, LARGE - 2)])  # Within max_integer_bits.
        # A member with long, distinct denominators: its common denominator would outgrow the budget, and the modular
        # filter refuses the claim before anything forms it.
        long_args, long_value, long_evidence = sturm_claim([(-1) ** k * (k + 1) for k in range(25)] + [1])
        spread = poly([Fraction(1, LARGE + 2 * k) for k in range(len(long_value[2].coefficients))])
        cases = [
            ("sturm_sequence", args, value * 50_000, evidence),
            ("sturm_sequence", args, value, evidence * 50_000),
            ("sturm_sequence", args, value + (p3,) * 100_000, evidence + (q3,) * 100_000),
            ("sturm_sequence", args, (p0, p1, p2, forged_poly((forged_rational(GIANT),))), evidence),
            ("sturm_sequence", args, (p0, p1, p2, forged_poly((forged_rational(1, GIANT),))), evidence),
            ("sturm_sequence", args, value, (q1, q2, (Rational(0), forged_rational(GIANT)))),
            ("sturm_sequence", args, value, (q1, q2, (Rational(0), Rational(LARGE)))),
            ("sturm_sequence", args, value, (q1, q2, (Rational(0), Rational(1, LARGE)))),
            ("sturm_sequence", args, (p0, p1, large_member, p3), evidence),
            ("sturm_sequence", args, value, ((Rational(LARGE), Rational(1, 3)), q2, q3)),
            ("sturm_sequence", args, (p0, p1, p2, forged_poly((Rational(1),) * 1_000_000)), evidence),
            ("sturm_sequence", {"polynomial": forged_poly((Rational(1),) * 1_000_000)}, value, evidence),
            ("sturm_sequence", long_args, replaced(long_value, 2, spread), long_evidence),
            ("sturm_sequence", {"polynomial": forged_poly((forged_rational(GIANT), Rational(1)))}, value, evidence),
            ("cyclotomic_polynomial", six_args, six_value, six_evidence * 50_000),
            ("cyclotomic_polynomial", six_args, six_value, (e1, e2, (3, (1,) * 1_000_000))),
            ("cyclotomic_polynomial", six_args, six_value, (e1, e2, (3, (1, GIANT, 1)))),
            ("cyclotomic_polynomial", six_args, six_value, (e1, e2, (3, (1, LARGE, 1)))),
            ("cyclotomic_polynomial", six_args, six_value, (e1, e2, (GIANT, (1, 1, 1)))),
            ("cyclotomic_polynomial", six_args, forged_poly((Rational(1),) * 1_000_000), six_evidence),
            ("cyclotomic_polynomial", six_args, forged_poly((Rational(1), forged_rational(GIANT), Rational(1))),
             six_evidence),
            ("cyclotomic_polynomial", six_args, poly([1, LARGE, 1]), six_evidence),
            ("cyclotomic_polynomial", {"n": 4620, "variable": X}, poly([1] * 961),
             tuple((d, (1,) * (totient(d) + 1)) for d in range(1, 4620) if not 4620 % d)),
        ]
        for index, (operation, case_args, case_value, case_evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, case_args, case_value, case_evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_deterministically(self):
        """The charges depend on the claim alone: a budget one unit short of a check refuses it, its exact size
        passes, and a tiny budget runs out at once."""
        claims = [("sturm_sequence",) + sturm_claim([Fraction(1, 3), -2, 0, Fraction(5, 7), 1]),
                  ("cyclotomic_polynomial",) + cyclotomic_claim(360)]
        for operation, args, claim_value, claim_evidence in claims:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, claim_value, claim_evidence), True)
                work = budget.work
                self.assertGreater(work, 20)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, claim_value, claim_evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, claim_value, claim_evidence,
                                   limits=ComputationLimits(max_work=work))
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, claim_value, claim_evidence,
                                    limits=ComputationLimits(max_work=5))

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_every_power_is_charged(self):
        """No builtin pow and no ** in the checker: every product it forms is charged before it is formed."""
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])

    def test_checker_imports_no_producer_code(self):
        tree = ast.parse(inspect.getsource(checker))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
        allowed = ("mathforge.certify", "mathforge.trusted", "mathforge.errors", "mathforge.limits", "mathforge.model")
        for module in imported:
            with self.subTest(module=module):
                self.assertTrue(module in ("__future__", "math") or module.startswith(allowed), module)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertFalse(names & {"importlib", "__import__", "modules", "sys", "builtins"}, names)

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layouts
        survive the format, and a wrong value fails the mathematics check."""
        args, value, evidence = CUBIC
        claims = [
            ("sturm_sequence", args, value, evidence, value[:3] + (poly([2]),)),
            ("sturm_sequence",) + sturm_claim([5]) + ((poly([1]),),),
            ("cyclotomic_polynomial",) + SIX + (poly([1, 1, 1]),),
            ("cyclotomic_polynomial",) + cyclotomic_claim(105, Y) + (poly(cyclotomic(105)),),
        ]
        for operation, claim_args, claim_value, claim_evidence, wrong in claims:
            with self.subTest(operation=operation, args=claim_args):
                self.roundtrip(claimed(operation, claim_args, claim_value, claim_evidence))
                report = support.statuses(mf.verify(claimed(operation, claim_args, wrong, claim_evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_an_equal_symbol_is_the_same_variable(self):
        """Review regression (completeness): the variable is compared by its fields (name, domain, assumptions, uid),
        as Symbol equality compares it, not by identity, so a value whose members, or whose cyclotomic polynomial,
        carry another Symbol object equal to the request's variable, as a JSON round trip produces, is accepted; a
        Symbol that differs in any one field is another variable and is refused."""
        twins = {X: Symbol("x", Reals, (), X.uid), Y: Symbol("y", Integers, (Assumption("positive", Truth.TRUE),),
                                                             Y.uid)}
        for variable, twin in twins.items():
            self.assertEqual(twin, variable)
            self.assertIsNot(twin, variable)
            others = (Symbol(variable.name, variable.domain, variable.assumptions),  # Another uid.
                      Symbol(variable.name, Rationals, (), variable.uid),  # Another domain.
                      Symbol(variable.name, variable.domain, (Assumption("nonzero", Truth.TRUE),), variable.uid),
                      Symbol("z", variable.domain, variable.assumptions, variable.uid))  # Another name.
            args, value, evidence = sturm_claim([0, -1, 0, 1], variable)
            with self.subTest(operation="sturm_sequence", variable=variable.name):
                self.assertAccepts("sturm_sequence", args, tuple(Polynomial(m.coefficients, twin) for m in value),
                                   evidence)
                self.assertAccepts("sturm_sequence", {"polynomial": Polynomial(args["polynomial"].coefficients, twin)},
                                   value, evidence)
                for other in others:
                    self.assertRejects("sturm_sequence", args,
                                       value[:2] + (Polynomial(value[2].coefficients, other),) + value[3:], evidence)
            args, value, evidence = cyclotomic_claim(12, variable)
            with self.subTest(operation="cyclotomic_polynomial", variable=variable.name):
                self.assertAccepts("cyclotomic_polynomial", args, Polynomial(value.coefficients, twin), evidence)
                self.assertAccepts("cyclotomic_polynomial", {"n": 12, "variable": twin}, value, evidence)
                for other in others:
                    self.assertRejects("cyclotomic_polynomial", args, Polynomial(value.coefficients, other), evidence)
