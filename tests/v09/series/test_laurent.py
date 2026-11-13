"""The laurent family of mathforge.series: residue.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register series laurent

Every expected value comes from the card or from the reference of this module, written here, slow on purpose and
independent of the producer: over Fractions, the Taylor coefficients of a polynomial at a by the derivative formula
T_j = sum_i binomial(i, j) A_i a^(i - j); the pole order k as the number of vanishing Taylor coefficients of D; the
cofactor R = sum_{j>=k} T_j(D) (x - a)^(j - k), expanded back to powers of x by the binomial theorem; the principal
part from the power series quotient of the Taylor series of N by that of R. At a simple pole the residue is also
N(a)/D'(a). Never from the producer, which divides once by (q x - p)^k and runs Horner's rule in integers. The
family defers nothing (the card names no Unevaluated case), so no test expects a NO_CONCLUSION result.
tests/v09/series/test_laurent_checker.py tests the checker on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import comb, lcm
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.series import laurent as family
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Rational, Symbol
from mathforge.polynomial import Polynomial
from mathforge.rational_function import RationalFunction

from .. import support

X, Y, T = Symbol("x"), Symbol("y"), Symbol("t")
OTHER_X = Symbol("x")  # The same name, another identity.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No exact number: a float, a bool, a str, None and two containers.
WIDE = ComputationLimits(max_integer_bits=1 << 20)  # Wide enough to build arguments beyond the default limits.

Number = Fraction | int


class Ratio(Rational):
    """A Rational subclass: no exact Rational."""
    __slots__ = ()


class Integer(int):
    """An int subclass, equal to an exact int but none."""


# ---------------------------------------------------------------------------------------------------------------
# The reference: every expected value of the tests comes from it, or from the card.

def fraction(number: object) -> Fraction:
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)  # type: ignore[arg-type]


def rational(number: object) -> Rational:
    value = fraction(number)
    return Rational(value.numerator, value.denominator)


def exact(number: Number) -> int | Rational:
    """An argument of the kind "rational": an int stays an int, a Fraction becomes a Rational."""
    return number if type(number) is int else rational(number)


def coefficients(polynomial: Polynomial) -> list[Fraction]:
    return [fraction(c) for c in polynomial.coefficients]


def evaluate(values: Sequence[Fraction], point: Fraction) -> Fraction:
    return sum((c * point ** i for i, c in enumerate(values)), Fraction(0))


def taylor(values: Sequence[Fraction], point: Fraction) -> list[Fraction]:
    """Every Taylor coefficient of the polynomial at point, by the derivative formula A^(j)(a)/j!."""
    return [sum((comb(i, j) * values[i] * point ** (i - j) for i in range(j, len(values))), Fraction(0))
            for j in range(len(values))]


def reference(function: RationalFunction, point: Number) -> tuple[Rational, tuple]:
    """(value, evidence (k, R, c)) of residue, from the record's N and D (module docstring)."""
    a = Fraction(point)
    n, d = coefficients(function.numerator), coefficients(function.denominator)
    t_d = taylor(d, a)
    k = next(j for j, c in enumerate(t_d) if c)  # D != 0.
    shifted = t_d[k:]  # R(x) = sum_j shifted[j] (x - a)^j.
    cofactor = [Fraction(0)] * len(shifted)
    for j, c in enumerate(shifted):
        for i in range(j + 1):
            cofactor[i] += c * comb(j, i) * (-a) ** (j - i)
    t_n = taylor(n, a) + [Fraction(0)] * k
    principal: list[Fraction] = []
    for m in range(k):  # N/R = sum h_m (x - a)^m, so N/D = sum h_m (x - a)^(m - k).
        r = sum((shifted[m - i] * principal[i] for i in range(m) if m - i < len(shifted)), Fraction(0))
        principal.append((t_n[m] - r) / shifted[0])
    residue = principal[-1] if k else Fraction(0)
    if k == 1:  # A simple pole: N(a)/D'(a), without the series.
        assert residue == evaluate(n, a) / evaluate([i * c for i, c in enumerate(d)][1:], a)
    return rational(residue), (k, tuple(map(rational, cofactor)), tuple(map(rational, principal)))


def polynomial(values: Sequence[object], variable: Symbol = X) -> Polynomial:
    return Polynomial(tuple(rational(c) for c in values), variable)


def fn(numerator: object, denominator: object, variable: Symbol = X) -> RationalFunction:
    return mf.rational_function(numerator, denominator, variable=variable)


def product(*factors: Sequence[Fraction]) -> list[Fraction]:
    result = [Fraction(1)]
    for factor in factors:
        out = [Fraction(0)] * (len(result) + len(factor) - 1)
        for i, x in enumerate(result):
            for j, y in enumerate(factor):
                out[i + j] += x * y
        result = out
    return result


def random_fraction(rng: random.Random, size: int = 9, denominators: int = 6) -> Fraction:
    return Fraction(rng.randrange(-size, size + 1), rng.randrange(1, denominators + 1))


def random_polynomial(rng: random.Random, degree: int, size: int = 9, denominators: int = 6) -> list[Fraction]:
    values = [random_fraction(rng, size, denominators) for _ in range(degree + 1)]
    while not values[-1]:
        values[-1] = random_fraction(rng, size, denominators)
    return values


def random_case(rng: random.Random) -> tuple[RationalFunction, Number]:
    """A function whose denominator has a root of order 0 to 5 at the point, often among other roots, and a point."""
    a = rng.choice((Fraction(rng.randrange(-5, 6)), random_fraction(rng, 9, 9), random_fraction(rng, 10 ** 6, 10 ** 4)))
    order = rng.randrange(6)
    other = random_polynomial(rng, rng.randrange(4))
    if not rng.randrange(4):  # Another root nearby.
        other = product(other, [-(a + 1), Fraction(1)])
    d = product(*([[-a, Fraction(1)]] * order), other)
    n = random_polynomial(rng, rng.randrange(6), rng.choice((9, 10 ** 9)), rng.choice((1, 6, 10 ** 5)))
    if not rng.randrange(8):
        n = [Fraction(0)]
    elif not rng.randrange(6):  # A factor that cancels with the denominator.
        n = product(n, [-a, Fraction(1)])
    variable = rng.choice((X, T))
    return fn(polynomial(n, variable), polynomial(d, variable), variable), exact(a)


def args(function: object, point: object, variable: object = X) -> dict[str, object]:
    return {"function": function, "variable": variable, "point": point}


# The card's known values: (numerator, denominator, point, residue).
KNOWN = (
    (1, X ** 2 - 1, 1, Fraction(1, 2)),
    (1, X ** 3, 0, Fraction(0)),
    (X + 1, X ** 2, 0, Fraction(1)),
    (1, X - 2, 3, Fraction(0)),
    (X ** 2, (X - 1) ** 3, 1, Fraction(1)),
)


class LaurentTests(support.FamilyTestCase):
    FAMILY = family

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.small = fn(1, X ** 2 - 1)

    def assertVerified(self, result: mf.Result, value: object, evidence: object = None) -> None:
        """A completed v6 Result of that value, an exact Rational, with four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), Rational)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        if evidence is not None:
            self.assertEqual(result.steps[0].outputs[1], evidence)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertCanonical(self, value: object, evidence: tuple) -> None:
        """The value and every number of the evidence are exact Rationals, k an exact int, both containers tuples."""
        self.assertIs(type(value), Rational)
        self.assertIs(type(evidence), tuple)
        k, cofactor, principal = evidence
        self.assertIs(type(k), int)
        self.assertIs(type(cofactor), tuple)
        self.assertIs(type(principal), tuple)
        self.assertEqual(len(principal), k)
        self.assertTrue(all(type(c) is Rational for c in cofactor + principal), evidence)

    def assertAgrees(self, function: RationalFunction, point: object) -> tuple[Rational, tuple]:
        """produce() answers the reference's value and evidence, canonical, and the checker accepts them."""
        variable = function.variable
        arguments = args(function, point, variable)
        produced = family.produce("residue", dict(arguments))
        expected = reference(function, fraction(point))
        self.assertEqual(produced, expected)
        self.assertCanonical(*produced)
        self.assertIs(support.verdict(family, "residue", arguments, *produced), True)
        return produced

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts, with the point as an int and as
        a Rational, and the evidence of the card's layout."""
        r = Rational
        layouts = ((1, (r(1), r(1)), (r(1, 2),)), (3, (r(1),), (r(1), r(0), r(0))), (2, (r(1),), (r(1), r(1))),
                   (0, (r(-2), r(1)), ()), (3, (r(1),), (r(1), r(2), r(1))))
        for (numerator, denominator, point, residue), layout in zip(KNOWN, layouts):
            function = fn(numerator, denominator)
            self.assertEqual(reference(function, point), (rational(residue), layout))
            for given in (point, Rational(point)):
                with self.subTest(function=str(function), point=repr(given)):
                    result = support.run("residue", function=function, variable=X, point=given)
                    self.assertVerified(result, rational(residue), layout)
                    self.assertEqual(result.steps[0].rule, "v6.residue")
                    self.roundtrip(result)
        # Special cases: the zero function, a constant, a polynomial, a removable excluded point (no pole), a pole
        # beside an excluded point, a negative residue, a rational point and rational coefficients, another variable.
        extra = (
            (fn(0, 1), 7, Fraction(0)),
            (fn(Rational(5, 2), 1), Rational(1, 9), Fraction(0)),
            (fn(X ** 4 - 3, 1), 2, Fraction(0)),
            (fn(X ** 2 - 1, X - 1), 1, Fraction(0)),
            (fn(X - 1, (X - 1) ** 2 * X), 1, Fraction(1)),
            (fn(1, X ** 2 - 1), -1, Fraction(-1, 2)),
            (fn(X / 2 + Rational(1, 3), (3 * X - 1) ** 2 * (X + 5)), Rational(1, 3), None),
            (fn(Y ** 3 - 2 * Y + 7, (Y - 2) ** 4 * (Y + 1) ** 2, Y), 2, None),
        )
        for function, point, residue in extra:
            with self.subTest(function=str(function), point=str(point)):
                value, evidence = reference(function, fraction(point))
                if residue is not None:
                    self.assertEqual(value, rational(residue))
                result = support.run("residue", function=function, variable=function.variable, point=point)
                self.assertVerified(result, value, evidence)
                self.roundtrip(result)
        self.assertTrue(fn(X ** 2 - 1, X - 1).excluded)  # x + 1 with x = 1 excluded: no pole there.

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so are a Polynomial or an Expression for the function, a function in another variable (also a
        Symbol of the same name), records forged with object.__new__ outside the canonical form, and an oversized
        matrix (no argument takes one). produce() tests the types itself too."""
        good = args(self.small, 1)
        others = (Fraction(1, 2), Ratio(1, 2), Integer(1), support.oversized_matrix(), polynomial((1, 2)), X ** 2 - 1,
                  support.impostor(self.small))
        for name in good:
            for item in STAND_INS + others + ((self.small,) if name != "function" else (X,)):
                with self.subTest(argument=name, item=type(item).__name__):
                    self.assertInvalidInput(support.run("residue", **dict(good, **{name: item})))
                    with self.assertRaises(InvalidInput):
                        family.produce("residue", dict(good, **{name: item}))
        for variable in (Y, OTHER_X, T):
            with self.subTest(variable=variable.name):
                result = support.run("residue", **dict(good, variable=variable))
                self.assertInvalidInput(result)
                self.assertIn("rational function in the variable", result.error.message)
        numerator, denominator = self.small.numerator, self.small.denominator

        def forged(**fields: object) -> RationalFunction:
            return support.forged(RationalFunction, **{"numerator": numerator, "denominator": denominator,
                                                       "excluded": (), **fields})

        def forged_polynomial(values: object, variable: object = X) -> Polynomial:
            return support.forged(Polynomial, coefficients=values, variable=variable)

        corrupt = {
            "non-monic denominator": forged(numerator=polynomial((2,)), denominator=polynomial((-2, 0, 2))),
            "zero denominator": forged(denominator=polynomial((0,))),
            "trailing zero": forged(denominator=forged_polynomial(denominator.coefficients + (Rational(0),))),
            "list coefficients": forged(denominator=forged_polynomial(list(denominator.coefficients))),
            "int coefficient": forged(numerator=forged_polynomial((1,))),
            "subclass coefficient": forged(numerator=forged_polynomial((Ratio(1),))),
            "non-reduced coefficient": forged(numerator=forged_polynomial(
                (support.forged(Rational, numerator=2, denominator=2),))),
            "negative denominator": forged(numerator=forged_polynomial(
                (support.forged(Rational, numerator=1, denominator=-2),))),
            "no coefficient": forged(numerator=forged_polynomial(())),
            "1,026 coefficients": forged(numerator=forged_polynomial((Rational(1),) * 1026)),
            "numerator in another variable": forged(numerator=polynomial((1,), Y)),
            "str variable": forged(numerator=forged_polynomial((Rational(1),), "x")),
            "Symbol with an int name": forged(numerator=forged_polynomial((Rational(1),), support.forged(
                Symbol, name=7, domain=X.domain, assumptions=(), uid=X.uid))),
            "impostor numerator": forged(numerator=support.impostor(numerator)),
        }
        for label, item in corrupt.items():
            with self.subTest(forged=label):
                self.assertInvalidInput(support.run("residue", **dict(good, function=item)))
                with self.assertRaises(InvalidInput):
                    family.produce("residue", dict(good, function=item))
        # Records whose missing or float fields the request's own inspection may meet first: produce() refuses them.
        missing = (support.forged(RationalFunction, denominator=denominator, excluded=()),
                   forged(numerator=support.forged(Polynomial, coefficients=(Rational(1),))),
                   forged(numerator=forged_polynomial((1.0,))),
                   support.forged(Rational, numerator=1), support.forged(Rational, numerator=1.0, denominator=2),
                   support.forged(Rational, numerator=2, denominator=4),
                   support.forged(Rational, numerator=1, denominator=0),
                   support.forged(Symbol, name="x"), support.forged(Symbol, name="x", domain="reals", assumptions=(),
                                                                    uid=X.uid))
        for name in good:
            for index, item in enumerate(missing):
                with self.subTest(argument=name, item=index, path="produce"):
                    with self.assertRaises(InvalidInput):
                        family.produce("residue", dict(good, **{name: item}))
        for point in (support.forged(Rational, numerator=2, denominator=4),
                      support.forged(Rational, numerator=-1, denominator=-2)):
            with self.subTest(point=(point.numerator, point.denominator)):
                result = support.run("residue", **dict(good, point=point))
                self.assertInvalidInput(result)
                self.assertIn("lowest terms with a positive denominator", result.error.message)
        for wrong in (dict(good, extra=1), {}, tuple(good.items()), None, {**good, 1: 2},
                      {"function": self.small, "variable": X}):
            with self.subTest(arguments=repr(wrong)[:40]), self.assertRaises(InvalidInput):
                family.produce("residue", wrong)
        for operation in ("residues", "Residue", "residue ", "continued_fraction", "", None, 1):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, dict(good))
        with self.assertRaises(mf.OperationError) as raised:  # The value function raises the structured error.
            support.value("residue", function=self.small, variable=X, point=0.5)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 in the point (an int or inside a Rational) or in a coefficient, a common denominator
        of the function that long, a residue or principal part with an integer that long, and max_work=1 exceed the
        budget. No argument takes a matrix: support.oversized_matrix() is invalid input (test_invalid_inputs)."""
        huge = support.HUGE
        good = args(self.small, 1)
        for point in (huge, -huge, Rational(huge + 1, 3), Rational(1, huge)):
            with self.subTest(point=point.bit_length() if type(point) is int else "Rational"):
                self.assertResourceLimit(support.run("residue", **dict(good, point=point)))
                with self.assertRaises(ResourceLimitError):
                    family.produce("residue", dict(good, point=point))
        q1, q2 = 2 ** 40_000 + 1, 2 ** 40_000 - 1  # Coprime, each within the limit, their product beyond it.
        with computation(WIDE, fresh=True):
            functions = {
                "huge coefficient": fn(polynomial((huge,)), polynomial((-1, 1))),
                "huge denominator": fn(polynomial((Fraction(1, huge + 1),)), polynomial((-1, 1))),
                "long common denominator": fn(polynomial((Fraction(1, q1), Fraction(1, q2))), polynomial((0, 1))),
                # 1/(x^2 (x - 2^40000)) at 0: c_{-1} = -1/2^80000.
                "long residue": fn(polynomial((1,)), polynomial((0, 0, -2 ** 40_000, 1))),
            }
        for label, function in functions.items():
            with self.subTest(function=label):
                self.assertResourceLimit(support.run("residue", **dict(good, function=function, point=0)))
                with self.assertRaises(ResourceLimitError):
                    family.produce("residue", dict(good, function=function, point=0))
        # The long residue is true, and the checker can only refuse to read it: it answers False, never True.
        long_args = args(functions["long residue"], 0)
        self.assertRejects("residue", long_args, Rational(-1, 2 ** 80_000),
                           (2, (Rational(-2 ** 40_000), Rational(1)), (Rational(-1, 2 ** 40_000),
                                                                       Rational(-1, 2 ** 80_000))))
        with self.subTest(max_work=1):
            self.assertResourceLimit(support.run("residue", limits=ComputationLimits(max_work=1), **good))
            with computation(ComputationLimits(max_work=1), fresh=True), self.assertRaises(ResourceLimitError):
                family.produce("residue", dict(good))
        # Every pass is charged before it runs: a full-degree request on a small budget fails at once.
        deep = RationalFunction(polynomial((1,)), polynomial([comb(1024, j) * (-1) ** (1024 - j) for j in range(1025)]),
                                (polynomial((-1, 1)),))  # 1/(x - 1)^1024.
        start = time.perf_counter()
        self.assertResourceLimit(support.run("residue", limits=ComputationLimits(max_work=20_000), **args(deep, 1)))
        self.assertLess(time.perf_counter() - start, 1.0)
        with computation(ComputationLimits(max_work=10 ** 9), fresh=True) as budget:
            family.produce("residue", dict(good))
        self.assertGreaterEqual(budget.work, 2)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The claim is unique (the checker's docstring), so the true claims come from the reference: for functions in
        other variables, points given as a Rational of denominator 1, and a record forged outside lowest terms, which
        both sides read as the quotient N/D."""
        rng = random.Random(20261301)
        for _ in range(30):
            function, point = random_case(rng)
            value, evidence = reference(function, fraction(point))
            for given in (point, rational(point)):
                with self.subTest(function=str(function), point=repr(given)):
                    self.assertAccepts("residue", args(function, given, function.variable), value, evidence)
        forged = support.forged(RationalFunction, numerator=polynomial((0, 1)), denominator=polynomial((0, 0, 1)),
                                excluded=())  # x/x^2, judged as 1/x.
        claim = (Rational(1), (2, (Rational(1),), (Rational(0), Rational(1))))
        self.assertEqual(reference(forged, 0), claim)
        self.assertAccepts("residue", args(forged, 0), *claim)
        self.assertEqual(family.produce("residue", args(forged, 0)), claim)

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics of a
        Result whose value was replaced."""
        function = fn(X ** 2 + 3, (X - 1) ** 2 * (X + 2))
        result = support.run("residue", function=function, variable=X, point=1)
        value, evidence = result.value, result.steps[0].outputs[1]
        self.assertVerified(result, *reference(function, 1))
        k, cofactor, principal = evidence
        arguments = args(function, 1)
        long = Rational(2 ** 60_000 + 1, 3)
        cases = {
            "plus one": (rational(fraction(value) + 1), evidence),
            "the opposite sign": (-value, evidence),
            "c_{-2} as the value": (principal[0], evidence),
            "an int value": (int(value) if value.denominator == 1 else value.numerator, evidence),
            "a bool value": (True, evidence),
            "a float value": (float(fraction(value)), evidence),
            "a str value": (str(value), evidence),
            "a subclass value": (Ratio(value.numerator, value.denominator), evidence),
            "an impostor value": (support.impostor(value), evidence),
            "a forged value": (support.forged(Rational, numerator=2 * value.numerator,
                                              denominator=2 * value.denominator), evidence),
            "no evidence": (value, ()),
            "extra evidence": (value, evidence + ((),)),
            "a list": (value, list(evidence)),
            "R as a list": (value, (k, list(cofactor), principal)),
            "c as a list": (value, (k, cofactor, list(principal))),
            "k as a bool": (value, (True, cofactor, principal)),
            "k as a float": (value, (float(k), cofactor, principal)),
            "k as a str": (value, (str(k), cofactor, principal)),
            "k minus one": (value, (k - 1, cofactor, principal[1:])),
            "c of ints": (value, (k, cofactor, tuple(int(c.numerator) for c in principal))),
            "a wrong c_{-2}": (value, (k, cofactor, (principal[0] + 1,) + principal[1:])),
            "a wrong cofactor": (value, (k, (cofactor[0] + 1,) + cofactor[1:], principal)),
            "an oversized principal part": (value, (k, cofactor, (Rational(0),) * 100_000)),
            "a long entry": (long, (k, cofactor, principal[:-1] + (long,))),
        }
        for label, (claimed, proof) in cases.items():
            with self.subTest(label):
                start = time.perf_counter()
                self.assertRejects("residue", arguments, claimed, proof)
                self.assertLess(time.perf_counter() - start, 3.0)
        for operation in ("residues", "Residue", "", None, "sum_polynomial"):
            with self.subTest(operation=operation):
                self.assertRejects(operation, arguments, value, evidence)
        for label, item in (("impostor function", support.impostor(function)),
                            ("a Polynomial", function.denominator),
                            ("non-monic", support.forged(RationalFunction, numerator=polynomial((2,)),
                                                         denominator=polynomial((-2, 0, 2)), excluded=()))):
            with self.subTest(argument=label):
                self.assertRejects("residue", dict(arguments, function=item), value, evidence)
        report = support.statuses(mf.verify(support.tampered(result, rational(fraction(value) + 1))))
        self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
        self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [args(fn(numerator, denominator), point) for numerator, denominator, point, _ in KNOWN]
        requests += [args(fn(X / 2 + Rational(1, 3), (3 * X - 1) ** 2 * (X + 5)), Rational(1, 3)),
                     args(fn(0, 1), Rational(-4, 7))]
        results = [support.run("residue", **request) for request in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for result, text in zip(results, texts):
                with self.subTest(function=str(result.request.arguments[0][1])):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run("residue", **requests[0])
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value, a Rational, renders as text and as LaTeX:
        fractional, integral, negative and zero."""
        rendered = (
            (fn(1, X ** 2 - 1), 1, "1/2", r"\frac{1}{2}"),
            (fn(X ** 2, (X - 1) ** 3), 1, "1", "1"),
            (fn(1, X ** 2 - 1), -1, "-1/2", r"-\frac{1}{2}"),
            (fn(1, X - 2), 3, "0", "0"),
            (fn(3 * X + 5, X * (X - 4)), 0, "-5/4", r"-\frac{5}{4}"),
        )
        for function, point, text, latex in rendered:
            with self.subTest(function=str(function), point=point):
                result = support.run("residue", function=function, variable=X, point=point)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual((mf.render_text(result.value), mf.to_latex(result.value)), (text, latex))
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        data = mf.to_data(support.run("residue", function=fn(1, X ** 2 - 1), variable=X, point=1).value)["data"]
        self.assertEqual(data, {"type": "rational", "numerator": "1", "denominator": "2"})
        evidence = mf.to_data(support.run("residue", function=fn(1, X ** 3), variable=X, point=0)
                              .steps[0].outputs[1])["data"]
        self.assertEqual(evidence["type"], "tuple")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path. On an
        input without a value the lean path falls back, and the value function raises the structured path's error."""
        rng = random.Random(20261302)
        for _ in range(25):
            function, point = random_case(rng)
            arguments = args(function, point, function.variable)
            with self.subTest(function=str(function), point=str(point)):
                expected = support.run("residue", **arguments).value
                self.assertIsNotNone(expected)
                found = support.lean_value("residue", **arguments)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value("residue", **arguments), expected)
        bad = args(self.small, 1, Y)
        self.assertIs(support.lean_value("residue", **bad), lean.FALLBACK)
        with self.assertRaises(mf.OperationError) as raised:
            support.value("residue", **bad)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path: a rational function
        travels as its text and variable, the variable as its name, the point as a JSON integer or a string "p/q"."""
        requests = (args(fn(1, X ** 2 - 1), 1), args(fn(X ** 2, (X - 1) ** 3), Rational(1)),
                    args(fn(X / 2 + Rational(1, 3), (3 * X - 1) ** 2 * (X + 5)), Rational(1, 3)))
        for request in requests:
            with self.subTest(function=str(request["function"]), point=str(request["point"])):
                result = self.answer(support.call_values("residue", **request))
                self.assertEqual(result, support.run("residue", **request))
                self.assertTrue(mf.verify(result).verified)
        raw = (
            ({"function": {"text": "1/(x^2 - 1)", "variable": "x"}, "variable": "x", "point": 1}, Rational(1, 2)),
            ({"function": {"text": "(x + 1)/x^2", "variable": "x"}, "variable": "x", "point": 0}, Rational(1)),
            ({"function": {"text": "1/(3*t - 1)^2/(t + 5)", "variable": "t"}, "variable": "t", "point": "1/3"},
             None),
            ({"function": {"text": "t^2 + 1", "variable": "t"}, "variable": "t", "point": "-2/5"}, Rational(0)),
        )
        for request, value in raw:
            with self.subTest(request=request):
                result = self.answer(support.call("residue", request))
                decoded = dict(result.request.arguments)
                self.assertEqual(result.value, reference(decoded["function"], fraction(decoded["point"]))[0])
                if value is not None:
                    self.assertEqual(result.value, value)
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("residue", {"function": {"text": "1/(x - 1)", "variable": "x"},
                                                              "variable": "y", "point": 1}), "invalid_input")
        self.assertIn("variable", error["message"])

    def test_seeded_agreement_with_the_reference(self):
        """produce() agrees with the reference of this module on at least 200 seeded inputs, and the checker accepts
        every answer with its evidence."""
        rng = random.Random(20261303)
        orders = dict.fromkeys(range(6), 0)
        count = 0
        while count < 220:
            function, point = random_case(rng)
            with self.subTest(function=str(function), point=str(point)):
                _, evidence = self.assertAgrees(function, point)
                orders[evidence[0]] += 1
                count += 1
        self.assertTrue(all(orders[k] >= 10 for k in range(5)), orders)  # Every pole order up to 4 is exercised.

    def test_full_size_inputs_fit_the_default_budget(self):
        """Full-degree denominators complete under the default limits, verified, each in a few seconds at most: poles
        of order 1024 at 0 and at 1, two poles of order 512, and the checker report's worst cases."""
        # The binomial theorem writes the large denominators, and the records are built directly with their
        # exclusions, both far faster than expanding powers of expressions.
        one = polynomial((1,))
        with computation(ComputationLimits(max_work=10 ** 9), fresh=True):  # Building the records takes work too.
            cases = (
                (fn(1, X ** 1024), 0, Rational(0)),
                (RationalFunction(one, polynomial([comb(1024, j) * (-1) ** (1024 - j) for j in range(1025)]),
                                  (polynomial((-1, 1)),)), 1, Rational(0)),
                # (1 + x^3)/(x - 1) = -(1 + x^3)(1 + x + x^2 + ...): the coefficient of x^999 is -2.
                (fn(X ** 3 + 1, X ** 1000 * (X - 1)), 0, Rational(-2)),
                (RationalFunction(one, polynomial([0 if j % 2 else comb(512, j // 2) * (-1) ** (512 - j // 2)
                                                   for j in range(1025)]), (polynomial((-1, 0, 1)),)), 1, None),
                (fn(X ** 20 - X + 1, (3 * X + 1) ** 30 * (X - 2) ** 5), Rational(-1, 3), None),
            )
        values = []
        for function, point, value in cases:
            with self.subTest(degree=len(function.denominator.coefficients) - 1, point=str(point)):
                start = time.perf_counter()
                result = support.run("residue", function=function, variable=X, point=point)
                values.append(result.value)
                self.assertLess(time.perf_counter() - start, 5.0)
                self.assertTrue(mf.verify(result).verified, result.error)
                if value is not None:
                    self.assertEqual(result.value, value)
        # 1/((x - 1)^512 (x + 1)^512) at 1: 1/(x + 1)^512 = 2^-512 sum_m binomial(-512, m) (y/2)^m with y = x - 1,
        # so the residue is binomial(-512, 511) 2^(-1023) = -binomial(1022, 511)/2^1023.
        self.assertEqual(values[3], Rational(-comb(1022, 511), 2 ** 1023))

    def test_poles_of_full_order_away_from_small_points(self):
        """Regression (m7b review): dividing the denominator by q x - p again and again cost k passes over all of it,
        so 1/(x - a)^1024 at a = 100, 7/5 or 1000/999 exceeded the producer's default budget, although the checker
        verifies each claim within it. Now k is guessed modulo the prime 2^61 - 1 and confirmed by one exact division.
        A cofactor whose value at the point is divisible by that prime makes the guess too large: the exact division
        refutes it and a binary search finds k."""
        one = polynomial((1,))
        for point in (100, Rational(7, 5), Rational(1000, 999)):
            a = fraction(point)
            with computation(ComputationLimits(max_work=10 ** 9), fresh=True):  # Building the record takes work.
                function = RationalFunction(one, polynomial([comb(1024, j) * (-a) ** (1024 - j) for j in range(1025)]),
                                            (polynomial((-a, 1)),))
            with self.subTest(point=str(point)):
                start = time.perf_counter()
                result = support.run("residue", function=function, variable=X, point=point)
                self.assertVerified(result, Rational(0), (1024, (Rational(1),), (Rational(1),) + (Rational(0),) * 1023))
                self.assertLess(time.perf_counter() - start, 5.0)
        prime = (1 << 61) - 1
        for point in (1, Rational(1, 2)):
            a = fraction(point)
            denominator = product([-a, Fraction(1)], [-a, Fraction(1)], [-(a + prime), Fraction(1)])
            function = fn(1, polynomial(denominator))
            with self.subTest(point=str(point), path="fallback"):
                # The modular guess is 3, one more than the pole order: R(a) = -prime.
                common = lcm(*(c.denominator for c in denominator))  # D' = F D for D = (x - a)^2 (x - a - prime).
                integers = [int(c * common) for c in denominator]
                with computation(fresh=True) as budget:
                    self.assertEqual(family._modular_order(budget, integers, a.numerator, a.denominator), 3)
                _, evidence = self.assertAgrees(function, point)
                self.assertEqual(evidence[0], 2)
                self.assertVerified(support.run("residue", function=function, variable=X, point=point),
                                    *reference(function, a))

    def test_high_pole_order_with_a_small_cofactor(self):
        """Regression (m7b review): the power series quotient once rescaled every earlier term whenever their common
        denominator grew, about k^2/2 more products: on N/((x - 2/3)^1020 R) with N of degree 10 and R of degree 4 it
        took 2,160,000 units, past the default budget, while the checker verifies the claim with 1,135,000. Now it
        rescales only the terms the recursion still reads (the last deg R) and stays within the checker's work."""
        rng = random.Random(20261305)
        other = [Fraction(rng.randrange(-9, 10)) for _ in range(4)] + [Fraction(1)]
        numerator = [Fraction(rng.randrange(-9, 10)) for _ in range(10)] + [Fraction(1)]
        a = Fraction(2, 3)
        denominator = product([comb(1020, j) * (-a) ** (1020 - j) for j in range(1021)], other)
        # Built directly: the record's own checks of a degree-1024 denominator are slow.
        function = support.forged(RationalFunction, numerator=polynomial(numerator),
                                  denominator=polynomial(denominator), excluded=())
        arguments = args(function, rational(a))
        with computation(fresh=True) as budget:  # The default limits.
            value, evidence = family.produce("residue", dict(arguments))
        self.assertLess(budget.work, 1_600_000)
        self.assertEqual(evidence[0], 1020)
        self.assertCanonical(value, evidence)
        self.assertIs(support.verdict(family, "residue", arguments, value, evidence), True)
