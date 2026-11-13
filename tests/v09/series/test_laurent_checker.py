"""The independent checker of mathforge.areas.series.laurent (mathforge.certify.series.laurent), tested on its own
before the family's producer exists.

Every true claim is computed here, never by the producer: the pole order and the cofactor by repeated synthetic division
of the denominator by x - a over Fractions, and the principal part as the first k coefficients of the power series
quotient N(y + a)/R(y + a), each Taylor expansion by repeated synthetic division. Independently of that reference, the
residue of each known value is also checked by partial fractions of a simple-pole function and by the card's table.
The evidence follows the layout (k, R, c) of the family's docstring. The checker answers False to every corrupted
claim, never with an exception, and refuses an oversized forgery within FAST seconds under the default budget.
"""

from __future__ import annotations

from fractions import Fraction
import time

import mathforge as mf
from mathforge.areas.series import laurent as family
from mathforge.certify.series import laurent as checker
from mathforge.limits import ComputationLimits, ResourceLimitError
from mathforge.model import Rational, Symbol
from mathforge.polynomial import Polynomial
from mathforge.rational_function import RationalFunction

from .. import support

FAST = 3.0  # Seconds: an oversized forgery is refused well within this under the default budget.
X, Y = Symbol("x"), Symbol("y")
OTHER_X = Symbol("x")  # The same name, another identity.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
LONG = 2 ** 60_000 + 1  # Within max_integer_bits, and far longer than the true values of the claims it corrupts.


class Ratio(Rational):
    """A Rational subclass: no exact Rational."""
    __slots__ = ()


# ---------------------------------------------------------------------------------------------------------------
# The reference: every true claim of the tests comes from it.

def fractions(polynomial: Polynomial) -> list[Fraction]:
    return [Fraction(c.numerator, c.denominator) for c in polynomial.coefficients]


def rational(number: Fraction | int) -> Rational:
    number = Fraction(number)
    return Rational(number.numerator, number.denominator)


def divide(coefficients: list[Fraction], point: Fraction) -> tuple[list[Fraction], Fraction]:
    """(quotient, remainder) of a polynomial (lowest degree first) by x - point."""
    quotient, accumulator = [Fraction(0)] * (len(coefficients) - 1), Fraction(0)
    for i in range(len(coefficients) - 1, -1, -1):
        accumulator = accumulator * point + coefficients[i]
        if i:
            quotient[i - 1] = accumulator
    return quotient, accumulator


def taylor(coefficients: list[Fraction], point: Fraction, count: int) -> list[Fraction]:
    """The first count coefficients of P(y + point)."""
    out, current = [], list(coefficients)
    for _ in range(count):
        if not current:
            out.append(Fraction(0))
            continue
        current, remainder = divide(current, point)
        out.append(remainder)
    return out


def reference(function: RationalFunction, point: Fraction | int) -> tuple[Rational, tuple]:
    """(value, evidence) of residue, by synthetic division and a power series quotient."""
    point = Fraction(point)
    numerator, cofactor, order = fractions(function.numerator), fractions(function.denominator), 0
    while len(cofactor) > 1:
        quotient, remainder = divide(cofactor, point)
        if remainder:
            break
        cofactor, order = quotient, order + 1
    n_series, r_series = taylor(numerator, point, order), taylor(cofactor, point, order)
    series: list[Fraction] = []
    for m in range(order):  # N/R = sum h_m y^m, so N/D = sum h_m y^(m - k).
        series.append((n_series[m] - sum(r_series[m - i] * series[i] for i in range(m))) / r_series[0])
    residue = series[-1] if order else Fraction(0)
    return rational(residue), (order, tuple(rational(c) for c in cofactor), tuple(rational(c) for c in series))


def args(function: object, point: object, variable: object = X) -> dict[str, object]:
    return {"function": function, "variable": variable, "point": point}


def fn(numerator: object, denominator: object, variable: Symbol = X) -> RationalFunction:
    return mf.rational_function(numerator, denominator, variable=variable)


def polynomial(coefficients: tuple, variable: Symbol = X) -> Polynomial:
    return Polynomial(tuple(rational(c) for c in coefficients), variable)


# The card's known values: (numerator, denominator, point, residue).
KNOWN = (
    (1, X ** 2 - 1, 1, Fraction(1, 2)),
    (1, X ** 3, 0, Fraction(0)),
    (X + 1, X ** 2, 0, Fraction(1)),
    (1, X - 2, 3, Fraction(0)),
    (X ** 2, (X - 1) ** 3, 1, Fraction(1)),
)


class LaurentCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def accepts(self, function: RationalFunction, point: object) -> tuple[Rational, tuple]:
        value, evidence = reference(function, point if type(point) is int else
                                    Fraction(point.numerator, point.denominator))
        self.assertAccepts("residue", args(function, point), value, evidence)
        return value, evidence

    def rejects(self, arguments: object, value: object, evidence: object, operation: object = "residue") -> None:
        self.assertRejects(operation, arguments, value, evidence)

    def fast_rejects(self, arguments: object, value: object, evidence: object) -> None:
        start = time.perf_counter()
        self.assertRejects("residue", arguments, value, evidence)
        self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # True claims

    def test_known_values(self):
        """The card's table: each residue, with the evidence of the reference, and the reference agrees."""
        for numerator, denominator, point, residue in KNOWN:
            with self.subTest(numerator=str(numerator), denominator=str(denominator), point=point):
                function = fn(numerator, denominator)
                value, evidence = self.accepts(function, point)
                self.assertEqual(value, rational(residue))
                self.assertIs(type(value), Rational)
                self.assertAccepts("residue", args(function, Rational(point)), value, evidence)

    def test_evidence_layout_of_the_known_values(self):
        """The evidence (k, R, c) spelled out for the table, so that the layout itself is pinned."""
        r = Rational
        self.assertEqual(reference(fn(1, X ** 2 - 1), 1)[1], (1, (r(1), r(1)), (r(1, 2),)))
        self.assertEqual(reference(fn(1, X ** 3), 0)[1], (3, (r(1),), (r(1), r(0), r(0))))
        self.assertEqual(reference(fn(X + 1, X ** 2), 0)[1], (2, (r(1),), (r(1), r(1))))
        self.assertEqual(reference(fn(1, X - 2), 3)[1], (0, (r(-2), r(1)), ()))
        self.assertEqual(reference(fn(X ** 2, (X - 1) ** 3), 1)[1], (3, (r(1),), (r(1), r(2), r(1))))

    def test_simple_poles_agree_with_partial_fractions(self):
        """At a simple pole the residue is N(a)/D'(a), computed here without the reference."""
        function = fn(X ** 2 + 3, (X - 1) * (X + 2) * (2 * X - 1))
        for point in (1, -2, Fraction(1, 2)):
            with self.subTest(point=point):
                n = fractions(function.numerator)
                d = fractions(function.denominator)
                derivative = [i * c for i, c in enumerate(d)][1:]
                at = lambda cs, t: sum(c * t ** i for i, c in enumerate(cs))  # noqa: E731
                expected = at(n, Fraction(point)) / at(derivative, Fraction(point))
                exact_point = point if type(point) is int else rational(point)
                value, _ = self.accepts(function, exact_point)
                self.assertEqual(value, rational(expected))

    def test_rational_points_and_rational_coefficients(self):
        """Non-integral points, rational coefficients, poles of several orders and points that are no pole."""
        function = fn(X / 2 + Rational(1, 3), (3 * X - 1) ** 2 * (X + 5) * (X ** 2 + 1))
        for point in (Rational(1, 3), -5, 0, Rational(-7, 4)):
            with self.subTest(point=str(point)):
                self.accepts(function, point)
        high = fn(X ** 3 - 2 * X + 7, (X - 2) ** 4 * (X + 1) ** 2)
        for point in (2, -1, 5):
            with self.subTest(point=point):
                self.accepts(high, point)

    def test_special_cases(self):
        """The zero function, a constant, a removable point (an exclusion that is no pole), a polynomial."""
        zero = fn(0, 1)
        self.assertEqual(self.accepts(zero, 7), (Rational(0), (0, (Rational(1),), ())))
        self.accepts(fn(Rational(5, 2), 1), Rational(1, 9))
        removable = fn(X, X)  # 1 with the point 0 excluded: no pole, residue 0.
        self.assertTrue(removable.excluded)
        self.assertEqual(self.accepts(removable, 0), (Rational(0), (0, (Rational(1),), ())))
        removable_pole = fn(X - 1, (X - 1) ** 2 * X)  # 1/(x(x - 1)) with x = 1 excluded: still a simple pole.
        self.assertEqual(self.accepts(removable_pole, 1)[0], Rational(1))
        self.accepts(fn(X ** 4 - 3, 1), 2)

    def test_non_reduced_record_is_judged_as_the_quotient(self):
        """The theorem holds for any N and D != 0, so a forged x/x^2 is judged as the quotient 1/x."""
        forged = support.forged(RationalFunction, numerator=polynomial((0, 1)), denominator=polynomial((0, 0, 1)),
                                excluded=())
        self.assertAccepts("residue", args(forged, 0), Rational(1), (2, (Rational(1),), (Rational(0), Rational(1))))
        self.assertRejects("residue", args(forged, 0), Rational(0), (2, (Rational(1),), (Rational(0), Rational(0))))

    def test_large_true_claims_within_the_default_budget(self):
        """Full-degree records verify under the default limits."""
        cases = (
            (fn(1, X ** 1024), 0),
            (fn(X ** 3 + 1, X ** 1000 * (X - 1)), 0),
            (fn(X + 2, (X - 1) ** 60), 1),
            (fn(X ** 20 - X + 1, (3 * X + 1) ** 30 * (X - 2) ** 5), Rational(-1, 3)),
            (fn(1, X ** 1024 + 1), 1),
        )
        for function, point in cases:
            with self.subTest(degree=len(function.denominator.coefficients) - 1, point=str(point)):
                start = time.perf_counter()
                self.accepts(function, point)
                self.assertLess(time.perf_counter() - start, 3 * FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims

    def test_wrong_values_and_wrong_operations(self):
        function = fn(1, X ** 2 - 1)
        value, evidence = reference(function, 1)
        self.rejects(args(function, 1), Rational(1, 3), evidence)
        self.rejects(args(function, 1), Rational(-1, 2), evidence)
        self.rejects(args(function, -1), value, evidence)  # The residue at -1 is -1/2, with other evidence.
        self.rejects(args(function, 2), value, evidence)
        for operation in ("residues", "Residue", "", None, 5, b"residue"):
            with self.subTest(operation=operation):
                self.rejects(args(function, 1), value, evidence, operation)

    def test_non_canonical_values(self):
        """Mathematically right values in a non-canonical form."""
        half_args, half = args(fn(1, X ** 2 - 1), 1), reference(fn(1, X ** 2 - 1), 1)[1]
        one_args, one = args(fn(X ** 2, (X - 1) ** 3), 1), reference(fn(X ** 2, (X - 1) ** 3), 1)[1]
        zero_args, zero = args(fn(1, X - 2), 3), reference(fn(1, X - 2), 3)[1]
        self.rejects(one_args, 1, one)  # An int, never a Rational.
        self.rejects(zero_args, 0, zero)
        self.rejects(one_args, True, one)
        self.rejects(zero_args, False, zero)
        self.rejects(half_args, 0.5, half)
        self.rejects(half_args, "1/2", half)
        self.rejects(half_args, Fraction(1, 2), half)
        self.rejects(half_args, support.forged(Rational, numerator=2, denominator=4), half)
        self.rejects(half_args, support.forged(Rational, numerator=-1, denominator=-2), half)
        self.rejects(half_args, support.forged(Rational, numerator=1), half)
        self.rejects(half_args, support.forged(Rational, numerator=True, denominator=2), half)
        self.rejects(half_args, support.impostor(Rational(1, 2)), half)
        self.rejects(half_args, Ratio(1, 2), half)
        self.rejects(half_args, None, half)
        self.rejects(half_args, (Rational(1, 2),), half)

    def test_wrong_argument_shapes(self):
        function = fn(1, X ** 2 - 1)
        value, evidence = reference(function, 1)
        good = args(function, 1)
        for name in ("function", "variable", "point"):
            with self.subTest(missing=name):
                self.rejects({key: item for key, item in good.items() if key != name}, value, evidence)
        self.rejects(dict(good, extra=1), value, evidence)
        self.rejects(list(good.items()), value, evidence)
        self.rejects(None, value, evidence)
        self.rejects({**{1: function}, "variable": X, "point": 1}, value, evidence)
        for point in (True, 1.0, "1", None, Fraction(1), Ratio(1), support.forged(Rational, numerator=2,
                                                                                    denominator=2)):
            with self.subTest(point=point):
                self.rejects(args(function, point), value, evidence)
        for variable in (Y, OTHER_X, "x", None):
            with self.subTest(variable=variable):
                self.rejects(args(function, 1, variable), value, evidence)
        self.rejects(args(fn(1, Y ** 2 - 1, Y), 1, X), value, evidence)

    def test_impostor_and_forged_arguments(self):
        function = fn(1, X ** 2 - 1)
        value, evidence = reference(function, 1)
        self.rejects(args(support.impostor(function), 1), value, evidence)
        self.rejects(args(function.denominator, 1), value, evidence)  # A Polynomial is no RationalFunction.
        self.rejects(args(X ** 2 - 1, 1), value, evidence)
        numerator, denominator = function.numerator, function.denominator

        def forged(**fields: object) -> RationalFunction:
            return support.forged(RationalFunction, **{"numerator": numerator, "denominator": denominator,
                                                       "excluded": (), **fields})

        self.assertAccepts("residue", args(forged(), 1), value, evidence)  # The forging itself keeps the claim.
        corrupt = {
            "impostor numerator": forged(numerator=support.impostor(numerator)),
            "impostor denominator": forged(denominator=support.impostor(denominator)),
            "non-monic denominator": forged(numerator=polynomial((2,)), denominator=polynomial((-2, 0, 2))),
            "zero denominator": forged(denominator=polynomial((0,))),
            "trailing zero": forged(denominator=support.forged(Polynomial, coefficients=denominator.coefficients
                                                               + (Rational(0),), variable=X)),
            "list coefficients": forged(denominator=support.forged(Polynomial, coefficients=list(
                denominator.coefficients), variable=X)),
            "int coefficient": forged(numerator=support.forged(Polynomial, coefficients=(1,), variable=X)),
            "float coefficient": forged(numerator=support.forged(Polynomial, coefficients=(1.0,), variable=X)),
            "non-reduced coefficient": forged(numerator=support.forged(
                Polynomial, coefficients=(support.forged(Rational, numerator=2, denominator=2),), variable=X)),
            "empty coefficients": forged(numerator=support.forged(Polynomial, coefficients=(), variable=X)),
            "other variable": forged(numerator=polynomial((1,), Y)),
            "missing variable": forged(numerator=support.forged(Polynomial, coefficients=(Rational(1),))),
            "missing numerator": support.forged(RationalFunction, denominator=denominator, excluded=()),
        }
        for label, item in corrupt.items():
            with self.subTest(label):
                self.rejects(args(item, 1), value, evidence)
        # A non-monic denominator with evidence that fits it: true of 2/(2x^2 - 2), but no record holds it.
        non_monic = forged(numerator=polynomial((2,)), denominator=polynomial((-2, 0, 2)))
        self.rejects(args(non_monic, 1), value, (1, (Rational(2), Rational(2)), (Rational(1, 2),)))

    def test_wrong_evidence_shapes(self):
        function = fn(1, X ** 2 - 1)
        value, (k, cofactor, principal) = reference(function, 1)
        arguments = args(function, 1)
        shapes = {
            "empty": (), "none": None, "list": [k, cofactor, principal], "two": (k, cofactor),
            "four": (k, cofactor, principal, ()), "nested": ((k, cofactor, principal),),
            "k bool": (True, cofactor, principal), "k float": (1.0, cofactor, principal),
            "k str": ("1", cofactor, principal), "k Rational": (Rational(1), cofactor, principal),
            "k negative": (-1, cofactor, principal), "k too big": (3, cofactor, principal),
            "R list": (k, list(cofactor), principal), "c list": (k, cofactor, list(principal)),
            "R short": (k, cofactor[:1], principal), "R long": (k, cofactor + (Rational(0),), principal),
            "R int": (k, (1, 1), principal), "R float": (k, (1.0, Rational(1)), principal),
            "R non-reduced": (k, (support.forged(Rational, numerator=3, denominator=3), Rational(1)), principal),
            "R subclass": (k, (Ratio(1), Rational(1)), principal),
            "c empty": (k, cofactor, ()), "c long": (k, cofactor, principal + (Rational(0),)),
            "c int": (k, cofactor, (0,)), "c str": (k, cofactor, ("1/2",)),
            "c float": (k, cofactor, (0.5,)), "c Fraction": (k, cofactor, (Fraction(1, 2),)),
            "c non-reduced": (k, cofactor, (support.forged(Rational, numerator=2, denominator=4),)),
            "c impostor": (k, cofactor, (support.impostor(Rational(1, 2)),)),
        }
        for label, evidence in shapes.items():
            with self.subTest(label):
                self.rejects(arguments, value, evidence)

    def test_consistent_but_wrong_evidence(self):
        """Forged evidence that agrees with its own value but breaks a condition of the theorem."""
        r = Rational
        # 1/(x^2 - 1) at 1 claimed as no pole: D = (x - 1)^0 D, but D(1) = 0.
        self.rejects(args(fn(1, X ** 2 - 1), 1), r(0), (0, (r(-1), r(0), r(1)), ()))
        # 1/x^3 at 0 claimed as a double pole with R = x: D = x^2 * x, but R(0) = 0.
        self.rejects(args(fn(1, X ** 3), 0), r(0), (2, (r(0), r(1)), (r(1), r(0))))
        # (x + 1)/x^2: the right k and R, a principal part whose last entry is the claimed value.
        self.rejects(args(fn(X + 1, X ** 2), 0), r(2), (2, (r(1),), (r(1), r(2))))
        self.rejects(args(fn(X + 1, X ** 2), 0), r(1), (2, (r(2),), (r(2), r(1))))  # R != D/x^2.
        self.rejects(args(fn(X + 1, X ** 2), 0), r(1), (2, (r(1),), (r(3), r(1))))  # A wrong c_{-2}.
        # 1/(x - 2) at 3 claimed as a pole: x - 2 != (x - 3) * 1.
        self.rejects(args(fn(1, X - 2), 3), r(1), (1, (r(1),), (r(1),)))
        # x^2/(x - 1)^3 at 1 with a wrong cofactor that still has R(1) != 0.
        self.rejects(args(fn(X ** 2, (X - 1) ** 3), 1), r(1), (2, (r(-1), r(1)), (r(1), r(1))))
        # At a simple pole, the value of another pole.
        function = fn(1, (X - 1) * (X + 1))
        value, evidence = reference(function, -1)
        self.rejects(args(function, 1), value, evidence)
        # The right value with the evidence of the point -1.
        self.rejects(args(function, 1), r(1, 2), evidence)

    def test_oversized_forgeries_fail_fast(self):
        """Huge integers in the value or the evidence answer False within FAST seconds under the default budget."""
        function = fn(1, X ** 2 - 1)
        value, (k, cofactor, principal) = reference(function, 1)
        arguments = args(function, 1)
        long, giant = Rational(LONG, 3), Rational(1, GIANT + 1)
        self.fast_rejects(arguments, long, (k, cofactor, (long,)))
        self.fast_rejects(arguments, giant, (k, cofactor, (giant,)))
        self.fast_rejects(arguments, value, (k, (long, Rational(1)), principal))
        self.fast_rejects(arguments, value, (k, (Rational(LONG, 3 ** 30_000), Rational(1)), principal))
        self.fast_rejects(arguments, value, (2 ** 100_000, cofactor, principal))
        self.fast_rejects(arguments, Rational(GIANT), (k, cofactor, principal))
        # A pole of order 1024 with a principal part of 1024 long entries.
        deep = fn(1, X ** 1024)
        true_value, (k, cofactor, principal) = reference(deep, 0)
        huge = Rational(LONG, 3 ** 37_000)
        self.fast_rejects(args(deep, 0), huge, (k, cofactor, principal[:-1] + (huge,)))
        self.fast_rejects(args(deep, 0), true_value, (k, cofactor, (huge,) * 1024))
        self.fast_rejects(args(deep, 0), true_value, (k, cofactor, (Rational(LONG),) * 1024))
        # A full-degree cofactor of long entries.
        wide = fn(1, X ** 1024 + 1)
        true_value, (k, cofactor, principal) = reference(wide, 0)
        self.fast_rejects(args(wide, 0), true_value, (k, (huge,) * 1024 + (Rational(1),), principal))
        self.fast_rejects(args(wide, 0), true_value, (k, (Rational(LONG),) * 1024 + (Rational(1),), principal))

    def test_arguments_beyond_the_budget(self):
        """An argument whose integers exceed max_integer_bits is beyond the budget: ResourceLimitError, never True."""
        function = fn(1, X ** 2 - 1)
        value, evidence = reference(function, 1)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "residue", args(function, Rational(GIANT, 3)), value, evidence)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "residue", args(function, 1), value, evidence,
                            limits=ComputationLimits(max_work=50))

    def test_checker_reads_nothing_from_the_producer(self):
        """The checker module imports no producer code and names record classes only by strings."""
        self.assertEqual(checker.RATIONAL_FUNCTION, (RationalFunction.__module__, RationalFunction.__qualname__))
        self.assertEqual(checker.POLYNOMIAL, (Polynomial.__module__, Polynomial.__qualname__))
        self.assertEqual(support.checker(family), checker)
