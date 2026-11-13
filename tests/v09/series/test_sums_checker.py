"""The independent checker of mathforge.areas.series.sums (mathforge.certify.series.sums), tested on its own before the
family's producer exists.

Every true claim is computed here, never by the producer: the closed form of a polynomial sum by solving
F(x) - F(x - 1) = p(x) from the top coefficient down, checked against partial sums added up term by term, and for sums
of powers by Faulhaber's formula with Bernoulli numbers; finite geometric sums term by term; infinite geometric series
as a/(1 - r), checked against long partial sums; and for the series of p(k) r^k the polynomial Q of the evidence by
solving r Q(x + 1) - Q(x) = p(x) from the top, its value -Q(0) checked against Newton's series
sum_j Delta^j p(0) r^j/(1 - r)^(j + 1). Large claims, up to the full degree of a Polynomial record, are built backwards
from a chosen F or Q with the binomial theorem. The evidence follows the layouts of the family's docstring. The
checker answers False to every corrupted claim, never with an exception, and refuses an oversized forgery within FAST
seconds under the default budget.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
from math import comb
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, registry
from mathforge.areas.series import sums as family
from mathforge.certify.series import sums as checker
from mathforge.contracts import CheckStatus
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Integers, Rational, Symbol, Truth
from mathforge.polynomial import Polynomial
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact rationals and real functions; certified closed forms of sums, series and trigonometric integrals"
X, Y = Symbol("x"), Symbol("y")
N = Symbol("n", Integers, (Assumption("positive", Truth.TRUE),))  # A variable with a domain and an assumption.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
LONG = 2 ** 60_000 + 1  # Within max_integer_bits, and far longer than the true values of the claims it corrupts.

Number = Fraction | int
Claim = tuple[str, dict[str, object], object, tuple]  # (operation, args, value, evidence)


class Ratio(Rational):
    """A Rational subclass: it compares unequal to the Rational of the same value, and is no exact Rational."""
    __slots__ = ()


class Integer(int):
    """An int subclass, equal to an exact int but none."""


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass for an evidence tuple or a coefficient tuple.


# ---------------------------------------------------------------------------------------------------------------
# References: every true claim of the tests comes from these, or from brute force.

def fraction(number: Number | Rational) -> Fraction:
    """A Fraction of an int, a Fraction or a mathforge Rational (which Fraction itself does not take)."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def rational(number: Number | Rational) -> Rational:
    number = fraction(number)
    return Rational(number.numerator, number.denominator)


def exact(number: Number) -> int | Rational:
    """An argument of the kind "rational": an int stays an int, a Fraction becomes a Rational."""
    return number if type(number) is int else rational(number)


def polynomial(coefficients: Sequence[Number], variable: Symbol = X) -> Polynomial:
    return Polynomial(tuple(rational(c) for c in coefficients), variable)


def trimmed(coefficients: Sequence[Number]) -> list[Fraction]:
    """The coefficients without trailing zeros, [0] for the zero polynomial."""
    values = [Fraction(c) for c in coefficients]
    while len(values) > 1 and not values[-1]:
        values.pop()
    return values or [Fraction(0)]


def evaluate(coefficients: Sequence[Number | Rational], point: Number) -> Fraction:
    total = Fraction(0)
    for c in reversed(coefficients):
        total = total * point + fraction(c)
    return total


def times(left: Sequence[Number], right: Sequence[Number]) -> list[Fraction]:
    product = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    return product


def shifted(coefficients: Sequence[Number], shift: int) -> list[Number]:
    """The coefficients of P(x + shift), by the binomial theorem with one row of Pascal's triangle per degree (ints stay
    ints): the checker's Taylor shift is repeated synthetic division instead."""
    out: list[Number] = [0] * len(coefficients)
    row = [1]
    for i, c in enumerate(coefficients):
        if i:
            row = [1, *(row[k] + row[k + 1] for k in range(len(row) - 1)), 1]
        if c:
            power = 1
            for j in range(i, -1, -1):  # c binomial(i, j) shift^(i - j) x^j.
                out[j] += c * row[j] * power
                power *= shift
    return out


def through(points: Sequence[tuple[Number, Number]]) -> list[Fraction]:
    """The polynomial of degree below len(points) through the points, by Lagrange's formula."""
    result = [Fraction(0)] * len(points)
    for i, (xi, yi) in enumerate(points):
        basis, scale = [Fraction(1)], Fraction(1)
        for j, (xj, _) in enumerate(points):
            if j != i:
                basis = times(basis, [-xj, 1])
                scale *= xi - xj
        for k, c in enumerate(basis):
            result[k] += yi * c / scale
    return result


def sum_reference(p: Sequence[Number]) -> list[Fraction]:
    """The closed form F of p(1) + ... + p(n): the coefficient of x^j in F(x) - F(x - 1) is
    sum_{i>j} F_i binomial(i, j) (-1)^(i-j+1), which gives F_(j+1) from the higher ones; F_0 = 0, and [0] for p = 0."""
    p = trimmed(p)
    if p == [0]:
        return [Fraction(0)]
    d = len(p) - 1
    f = [Fraction(0)] * (d + 2)
    for j in range(d, -1, -1):
        rest = sum((f[i] * comb(i, j) * (-1) ** (i - j + 1) for i in range(j + 2, d + 2)), Fraction(0))
        f[j + 1] = (p[j] - rest) / (j + 1)
    return f


def faulhaber(d: int) -> list[Fraction]:
    """The closed form of 1^d + ... + n^d by Faulhaber's formula, (1/(d + 1)) sum_j binomial(d + 1, j) B_j n^(d+1-j),
    with the Bernoulli numbers of B_1 = +1/2 from sum_{k<=m} binomial(m + 1, k) B_k = 0 (where B_1 = -1/2)."""
    b = [Fraction(1)] + [Fraction(0)] * d
    for m in range(1, d + 1):
        b[m] = -sum((comb(m + 1, k) * b[k] for k in range(m)), Fraction(0)) / (m + 1)
    if d >= 1:
        b[1] = Fraction(1, 2)
    f = [Fraction(0)] * (d + 2)
    for j in range(d + 1):
        f[d + 1 - j] += comb(d + 1, j) * b[j] / (d + 1)
    return f


def partial_sums(p: Sequence[Number], count: int) -> list[Fraction]:
    """[p(1) + ... + p(n) for n = 0, ..., count], added up term by term."""
    sums, total = [Fraction(0)], Fraction(0)
    for k in range(1, count + 1):
        total += evaluate(p, k)
        sums.append(total)
    return sums


def geometric_terms(first: Number, ratio: Number, count: int) -> Fraction:
    """first + first*ratio + ... + first*ratio^(count - 1), term by term."""
    total, term = Fraction(0), Fraction(first)
    for _ in range(count):
        total += term
        term *= ratio
    return total


def q_reference(p: Sequence[Number], r: Number) -> list[Fraction]:
    """Q with r Q(x + 1) - Q(x) = p(x) and deg Q = deg p, for r != 1: the coefficient of x^i is
    (r - 1) Q_i + r sum_{j>i} binomial(j, i) Q_j = p_i, solved from the top."""
    p, r = trimmed(p), Fraction(r)
    d = len(p) - 1
    q = [Fraction(0)] * (d + 1)
    for i in range(d, -1, -1):
        rest = sum((comb(j, i) * q[j] for j in range(i + 1, d + 1)), Fraction(0))
        q[i] = (p[i] - r * rest) / (r - 1)
    return q


def series_reference(p: Sequence[Number], r: Number) -> Fraction:
    """sum_{k>=0} p(k) r^k for |r| < 1 by Newton's series: p(k) = sum_j Delta^j p(0) binomial(k, j), and
    sum_k binomial(k, j) r^k = r^j/(1 - r)^(j + 1)."""
    r = Fraction(r)
    values = [evaluate(p, k) for k in range(len(p))]
    total = Fraction(0)
    for j in range(len(p)):
        total += values[0] * r ** j / (1 - r) ** (j + 1)
        values = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return total


def sum_claim(p: Sequence[Number], variable: Symbol = X) -> Claim:
    return "sum_polynomial", {"polynomial": polynomial(p, variable)}, polynomial(sum_reference(p), variable), ()


def sum_args(first: Number, ratio: Number, count: object) -> dict[str, object]:
    return {"first": exact(first), "ratio": exact(ratio), "count": count}


def series_args(first: Number, ratio: Number) -> dict[str, object]:
    return {"first": exact(first), "ratio": exact(ratio)}


def evidence_of(q: Sequence[Number]) -> tuple:
    """The evidence (Q,) of polynomial_geometric_series: Q's coefficients as a tuple of Rationals."""
    return (tuple(rational(c) for c in q),)


def pgs_claim(p: Sequence[Number], r: Number) -> Claim:
    q = q_reference(p, r)
    return ("polynomial_geometric_series", {"polynomial": polynomial(p), "ratio": exact(r)}, rational(-q[0]),
            evidence_of(q))


def backward_sum_claim(f: Sequence[int]) -> Claim:
    """A true sum_polynomial claim built from its value: F with F(0) = 0, and p(x) = F(x) - F(x - 1)."""
    assert f[0] == 0 and f[-1] != 0
    p = trimmed([a - b for a, b in zip(f, shifted(f, -1))])
    return "sum_polynomial", {"polynomial": polynomial(p)}, polynomial(f), ()


def backward_series_claim(q: Sequence[int], r: Fraction) -> Claim:
    """A true polynomial_geometric_series claim built from its evidence: p(x) = r Q(x + 1) - Q(x), |r| < 1."""
    assert abs(r) < 1 and q[-1] != 0
    p = trimmed([r * a - b for a, b in zip(shifted(q, 1), q)])
    return ("polynomial_geometric_series", {"polynomial": polynomial(p), "ratio": rational(r)}, rational(-q[0]),
            evidence_of(q))


def random_fraction(rng: random.Random, size: int = 40) -> Fraction:
    return Fraction(rng.randrange(-size, size + 1), rng.randrange(1, size))


def random_polynomial(rng: random.Random, degree: int) -> list[Fraction]:
    p = [random_fraction(rng) for _ in range(degree + 1)]
    while not p[-1]:
        p[-1] = random_fraction(rng)
    return p


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins for arguments, values and evidence that the checker must refuse.

def rational_stand_ins(number: Number) -> list[tuple[str, object]]:
    """Stand-ins for an exact rational that are no canonical Rational: a bool, a float, a str, None, a Fraction, a
    Rational subclass, a list and a pair of its integers, an impostor class, and Rationals forged without __init__."""
    number = Fraction(number)
    n, d = number.numerator, number.denominator
    return [("a bool", bool(number)), ("a float", float(number)), ("a str", str(number)), ("None", None),
            ("a Fraction", number), ("a Rational subclass", Ratio(n, d)), ("a list", [n, d]), ("a pair", (n, d)),
            ("an impostor of Rational", support.impostor(Rational(n, d))),
            ("a Rational forged without fields", support.forged(Rational)),
            ("a Rational forged without its denominator", support.forged(Rational, numerator=n)),
            ("a Rational not in lowest terms", support.forged(Rational, numerator=3 * n, denominator=3 * d)),
            ("a Rational with a negative denominator", support.forged(Rational, numerator=-n, denominator=-d)),
            ("a Rational with a float numerator", support.forged(Rational, numerator=float(n), denominator=d)),
            ("a Rational with a bool denominator", support.forged(Rational, numerator=n, denominator=True))]


def integer_stand_ins(number: int) -> list[tuple[str, object]]:
    """Stand-ins for an exact int that are none: a bool, a float, a str, None, a Rational, an int subclass and a
    Fraction."""
    return [("a bool", bool(number)), ("a float", float(number)), ("a str", str(number)), ("None", None),
            ("a Rational", Rational(number)), ("an int subclass", Integer(number)), ("a Fraction", Fraction(number))]


def polynomial_stand_ins(record: Polynomial) -> list[tuple[str, object]]:
    """Stand-ins for a canonical Polynomial: records of another class, records forged without __init__ that break the
    canonical form (most of them equal to the record as polynomials), and objects of other types."""
    coefficients, variable = record.coefficients, record.variable
    head, rest = coefficients[0], coefficients[1:]
    n, d = head.numerator, head.denominator

    def forged(values: object = coefficients, symbol: object = variable) -> Polynomial:
        return support.forged(Polynomial, coefficients=values, variable=symbol)

    items = [("an impostor of Polynomial", support.impostor(record)),
             ("a Polynomial forged without a variable", support.forged(Polynomial, coefficients=coefficients)),
             ("a Polynomial forged without coefficients", support.forged(Polynomial, variable=variable)),
             ("a trailing zero", forged(coefficients + (Rational(0),))),
             ("coefficients as a list", forged(list(coefficients))),
             ("coefficients as a tuple subclass", forged(Evidence(coefficients))),
             ("no coefficient", forged(())),
             ("a Fraction coefficient", forged((Fraction(n, d),) + rest)),
             ("a float coefficient", forged((n / d,) + rest)),
             ("a bool coefficient", forged((bool(n),) + rest)),
             ("a str coefficient", forged((str(head),) + rest)),
             ("None as a coefficient", forged((None,) + rest)),
             ("a Rational subclass coefficient", forged((Ratio(n, d),) + rest)),
             ("a coefficient not in lowest terms", forged((support.forged(Rational, numerator=2 * n,
                                                                          denominator=2 * d),) + rest)),
             ("a coefficient with a negative denominator", forged((support.forged(Rational, numerator=-n,
                                                                                  denominator=-d),) + rest)),
             ("a coefficient forged without its denominator", forged((support.forged(Rational, numerator=n),) + rest)),
             ("a str variable", forged(symbol=variable.name)),
             ("a Symbol forged without a uid", forged(symbol=support.forged(Symbol, name=variable.name))),
             ("an impostor of the Symbol", forged(symbol=support.impostor(variable))),
             ("the expression", record.to_expression()),
             ("the coefficient tuple", coefficients),
             ("a Rational", head),
             ("None", None)]
    if d == 1:
        items.append(("an int coefficient", forged((n,) + rest)))
    return items


def evidence_stand_ins(evidence: tuple) -> list[tuple[str, object]]:
    """Evidence of the wrong shape: a list, a tuple subclass, None, a str, one item more and one item fewer."""
    items = [("evidence as a list", list(evidence)), ("evidence as a tuple subclass", Evidence(evidence)),
             ("None as evidence", None), ("a str as evidence", "()"), ("an extra evidence item", evidence + ((),))]
    if evidence:
        items.append(("the last evidence item missing", evidence[:-1]))
    return items


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence),
                          relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,))


class SumsCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: str, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    def assertTypesMatter(self, operation: str, args: dict[str, object], value: object, evidence: tuple) -> None:
        """The checker accepts a true claim, and refuses it once an argument, the value or the evidence is replaced by
        a stand-in that is not of its exact type and canonical form."""
        self.assertAccepts(operation, args, value, evidence)
        cases: list[tuple[str, object, object, object]] = []
        for name, item in args.items():
            if type(item) is Polynomial:
                stand_ins = polynomial_stand_ins(item)
            elif name == "count":
                stand_ins = integer_stand_ins(item)
            else:
                stand_ins = rational_stand_ins(Fraction(item.numerator, item.denominator) if type(item) is Rational
                                               else item)
            cases += [(f"{name}: {label}", dict(args, **{name: stand_in}), value, evidence)
                      for label, stand_in in stand_ins]
        if type(value) is Polynomial:
            stand_ins = polynomial_stand_ins(value)
        else:
            stand_ins = rational_stand_ins(Fraction(value.numerator, value.denominator))
            if value.denominator == 1:
                stand_ins.append(("an int", value.numerator))
        cases += [(f"value: {label}", args, stand_in, evidence) for label, stand_in in stand_ins]
        cases += [(label, args, value, stand_in) for label, stand_in in evidence_stand_ins(evidence)]
        self.assertRejectsAll(operation, cases)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        polynomial_argument = Argument("polynomial", "Polynomial", None, "polynomial")
        first = Argument("first", "Rational | int", None, "rational")
        ratio = Argument("ratio", "Rational | int", None, "rational")
        expected = (
            OperationSpec("sum_polynomial", "series", (polynomial_argument,), "Polynomial",
                          "Find the closed form F(n) of the sum of p(k) for k = 1 .. n, a polynomial in the same "
                          "variable.", SCOPE, version=6),
            OperationSpec("geometric_sum", "series", (first, ratio, Argument("count", "int", None, "int")), "Rational",
                          "Add the first count terms of a geometric progression with exact rational first term and "
                          "ratio.", SCOPE, version=6),
            OperationSpec("geometric_series", "series", (first, ratio), "Rational",
                          "Sum an infinite geometric series with |ratio| < 1.", SCOPE, version=6),
            OperationSpec("polynomial_geometric_series", "series", (polynomial_argument, ratio), "Rational",
                          "Sum p(k) * ratio^k over k = 0, 1, 2, ... for a polynomial p and |ratio| < 1.", SCOPE,
                          version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {})  # Every value is a Polynomial or a Rational.
        self.assertEqual(family.CHECKER, "mathforge.certify.series.sums")
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_sum_polynomial_accepts_true_claims(self):
        half, third, quarter, sixth = Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 6)
        known = [((0, 1), (0, half, half)), ((0, 0, 1), (0, sixth, half, third)),
                 ((0, 0, 0, 1), (0, 0, quarter, half, quarter)), ((1,), (0, 1)), ((0,), (0,)), ((-1, 2), (0, 0, 1)),
                 ((-1, 1), (0, -half, half)), ((Fraction(-3, 7),), (0, Fraction(-3, 7)))]
        for p, f in known:  # The card's known values, then p = x - 1 and a constant.
            with self.subTest(p=p):
                self.assertEqual(sum_reference(p), [Fraction(c) for c in f])
                self.assertEqual([evaluate(f, n) for n in range(9)], partial_sums(p, 8))
                self.assertAccepts("sum_polynomial", {"polynomial": polynomial(p)}, polynomial(f))
        rng = random.Random(20261112)
        for degree in [*range(13)] * 3:
            p = random_polynomial(rng, degree)
            operation, args, value, evidence = sum_claim(p)
            with self.subTest(p=p):
                self.assertEqual([evaluate(value.coefficients, n) for n in range(degree + 5)],
                                 partial_sums(p, degree + 4))
                self.assertAccepts(operation, args, value, evidence)
        # Any variable: one with a domain and an assumption, another name, and another Symbol named x.
        for variable in (N, Y, Symbol("x")):
            with self.subTest(variable=variable):
                self.assertAccepts(*sum_claim((3, 0, Fraction(-5, 2)), variable))
        # Long coefficients, and a degree of 60 with rational coefficients.
        self.assertAccepts(*sum_claim((Fraction(2 ** 900 + 1, 3 ** 500), 0, -(2 ** 1000))))
        self.assertAccepts(*sum_claim(random_polynomial(rng, 60)))
        # Sums of powers by Faulhaber's formula, a second reference: x^300 has Bernoulli numbers of about 1,400 bits.
        for d in range(13):
            with self.subTest(power=d):
                self.assertEqual(faulhaber(d), sum_reference([0] * d + [1]))
        f = faulhaber(300)
        self.assertEqual([evaluate(f, n) for n in range(6)], partial_sums([0] * 300 + [1], 5))
        self.assertAccepts("sum_polynomial", {"polynomial": polynomial([0] * 300 + [1])}, polynomial(f))

    def test_geometric_sum_accepts_true_claims(self):
        for first, ratio, count, value in ((1, 2, 10, 1023), (1, Fraction(1, 2), 3, Fraction(7, 4)), (5, 1, 4, 20),
                                           (3, 7, 0, 0), (2, -1, 5, 2)):  # The card's known values.
            with self.subTest(first=first, ratio=ratio, count=count):
                self.assertEqual(geometric_terms(first, ratio, count), value)
                self.assertAccepts("geometric_sum", sum_args(first, ratio, count), rational(value))
        # Every special ratio and count: no division by 1 - r happens, so r = 1 is no case of its own.
        for first in (Fraction(-7, 3), 4, 0):
            for ratio in (1, -1, 0, Fraction(1, 2), Fraction(-5, 2), 3, Fraction(2, 3)):
                for count in (0, 1, 2, 3, 4, 7, 16, 33):
                    with self.subTest(first=first, ratio=ratio, count=count):
                        self.assertAccepts("geometric_sum", sum_args(first, ratio, count),
                                           rational(geometric_terms(first, ratio, count)))
        # Both forms of an exact rational argument: an int and a Rational with denominator 1.
        self.assertAccepts("geometric_sum", {"first": Rational(1), "ratio": Rational(2), "count": 10}, Rational(1023))
        rng = random.Random(20261113)
        for _ in range(200):
            first, ratio, count = random_fraction(rng), random_fraction(rng, 9), rng.randrange(60)
            with self.subTest(first=first, ratio=ratio, count=count):
                self.assertAccepts("geometric_sum", sum_args(first, ratio, count),
                                   rational(geometric_terms(first, ratio, count)))
        # Long sums: 3,000 terms of ratio -3/2, and huge counts where the terms do not grow (or are all 0).
        self.assertAccepts("geometric_sum", sum_args(Fraction(5, 7), Fraction(-3, 2), 3000),
                           rational(geometric_terms(Fraction(5, 7), Fraction(-3, 2), 3000)))
        big = 10 ** 40
        for first, ratio, count, value in ((Fraction(2, 3), 1, big, Fraction(2, 3) * big),
                                           (Fraction(2, 3), -1, big + 1, Fraction(2, 3)), (Fraction(2, 3), -1, big, 0),
                                           (Fraction(2, 3), 0, big, Fraction(2, 3)), (0, 2, big, 0),
                                           (1, 1, 2 ** 65535, 2 ** 65535)):  # A count of 65,536 bits.
            with self.subTest(ratio=ratio, count=count):
                self.assertAccepts("geometric_sum", sum_args(first, ratio, count), rational(value))

    def test_geometric_series_accepts_true_claims(self):
        for first, ratio, value in ((1, Fraction(1, 2), 2), (3, Fraction(-1, 3), Fraction(9, 4)), (0, 5, 0)):
            with self.subTest(first=first, ratio=ratio):  # The card's known values.
                self.assertAccepts("geometric_series", series_args(first, ratio), rational(value))
        for ratio in (5, -1, 1, Fraction(7, 3), -9, 0, Fraction(1, 2)):  # first = 0: every term is 0.
            with self.subTest(ratio=ratio):
                self.assertAccepts("geometric_series", series_args(0, ratio), Rational(0))
        self.assertAccepts("geometric_series", {"first": Rational(3), "ratio": Rational(0)}, Rational(3))
        self.assertAccepts("geometric_series", {"first": Rational(0), "ratio": Rational(-4)}, Rational(0))
        rng = random.Random(20261114)
        for _ in range(200):
            first, ratio = random_fraction(rng), random_fraction(rng, 30)
            if abs(ratio) >= 1:
                ratio = 1 / ratio if ratio.numerator else ratio
            if abs(ratio) >= 1:
                continue
            value = first / (1 - ratio)
            # The partial sums approach the value: |S_n - a/(1 - r)| = |a| |r|^n/|1 - r|.
            self.assertEqual(abs(geometric_terms(first, ratio, 40) - value),
                             abs(first) * abs(ratio) ** 40 / (1 - ratio))
            with self.subTest(first=first, ratio=ratio):
                self.assertAccepts("geometric_series", series_args(first, ratio), rational(value))

    def test_polynomial_geometric_series_accepts_true_claims(self):
        for p, r, value in (((0, 1), Fraction(1, 2), 2), ((0, 0, 1), Fraction(1, 2), 6), ((1,), Fraction(1, 2), 2),
                            ((0, 1), Fraction(-1, 2), Fraction(-2, 9))):  # The card's known values.
            with self.subTest(p=p, r=r):
                self.assertEqual(series_reference(p, r), value)
                self.assertEqual(-q_reference(p, r)[0], value)
                self.assertAccepts(*pgs_claim(p, r))
        self.assertEqual(q_reference((0, 1), Fraction(1, 2)), [-2, -2])  # (x, 1/2): Q = -2x - 2, so Q(0) = -2.
        self.assertEqual(pgs_claim((0, 1), Fraction(1, 2))[3], ((Rational(-2), Rational(-2)),))
        # p = 0: the value 0 with Q = (0,) for every ratio, divergent ones and 1 included.
        for r in (1, -1, 5, Fraction(1, 2), 0, Fraction(-7, 3)):
            with self.subTest(r=r):
                self.assertAccepts("polynomial_geometric_series", {"polynomial": polynomial((0,)), "ratio": exact(r)},
                                   Rational(0), ((Rational(0),),))
        # ratio 0: the sum is p(0); a constant p is the geometric series c/(1 - r).
        self.assertAccepts(*pgs_claim((Fraction(7, 2), -1, 4), 0))
        self.assertEqual(pgs_claim((Fraction(7, 2), -1, 4), 0)[2], Rational(7, 2))
        self.assertEqual(pgs_claim((5,), Fraction(-2, 3))[2], Rational(3))
        self.assertAccepts(*pgs_claim((5,), Fraction(-2, 3)))
        self.assertAccepts("polynomial_geometric_series", {"polynomial": polynomial((0, 1)), "ratio": Rational(1, 2)},
                           Rational(2), ((Rational(-2), Rational(-2)),))
        rng = random.Random(20261115)
        for degree in [*range(9)] * 3:
            p = random_polynomial(rng, degree)
            r = Fraction(rng.randrange(-11, 12), 12)
            with self.subTest(p=p, r=r):
                self.assertEqual(series_reference(p, r), -q_reference(p, r)[0])
                self.assertAccepts(*pgs_claim(p, r))
        self.assertAccepts(*pgs_claim(random_polynomial(rng, 40), Fraction(-99, 100)))

    def test_full_size_claims_fit_the_default_budget(self):
        """Claims of the largest degree a Polynomial record holds, built backwards, verify under the default budget:
        F of degree 1024 (p of degree 1023), Q and p of degree 1024; and a claim with long numerators over short
        denominators, whose gcds cost a long division each, not a quadratic gcd of the long operand."""
        rng = random.Random(20261116)
        f = [0] + [rng.randrange(-99, 100) for _ in range(1023)] + [rng.randrange(1, 100)]
        q = [rng.randrange(-99, 100) for _ in range(1024)] + [rng.randrange(1, 100)]
        # F = G/6 with G of degree 200 and coefficients of 60,000 bits: 202 coefficients with denominators 1 to 6.
        g = [0] + [rng.getrandbits(60_000) - (1 << 59_999) for _ in range(200)] + [(1 << 60_000) - 1]
        sixth = [Fraction(c, 6) for c in g]
        long_p = trimmed([a - b for a, b in zip(sixth, shifted(sixth, -1))])
        long_claim = ("sum_polynomial", {"polynomial": polynomial(long_p)}, polynomial(sixth), ())
        claims = [backward_sum_claim(f), backward_series_claim(q, Fraction(-2, 5)), long_claim]
        self.assertEqual(len(claims[0][2].coefficients), 1025)
        self.assertEqual(len(claims[1][3][0]), 1025)
        for operation, args, value, evidence in claims:
            with self.subTest(operation=operation, length=len(args["polynomial"].coefficients)):
                start = time.perf_counter()
                self.assertAccepts(operation, args, value, evidence)
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_sum_polynomial_rejects_corrupted_claims(self):
        p = [Fraction(1, 2), -1, 3]  # 3x^2 - x + 1/2
        f = sum_reference(p)
        operation, args, value, evidence = sum_claim(p)
        self.assertTypesMatter(operation, args, value, evidence)
        self.assertTypesMatter(*sum_claim((0,)))
        self.assertTypesMatter(*sum_claim((Fraction(-3, 7),), N))
        # F + 5 x(x - 1)(x - 2)(x - 3) has the values of F at 0, ..., 3, so it passes the card's point test: the degree
        # test refuses it, and the coefficient identity would.
        bump = [Fraction(1)]
        for k in range(4):
            bump = times(bump, [-k, 1])
        crafted = [a + 5 * b for a, b in zip([*f, 0], bump)]
        self.assertEqual([evaluate(crafted, k) for k in range(4)], [evaluate(f, k) for k in range(4)])
        self.assertRejectsAll(operation, [
            ("a wrong constant term", args, polynomial([1, *f[1:]]), ()),
            ("a wrong leading coefficient", args, polynomial([*f[:-1], 2 * f[-1]]), ()),
            ("a wrong middle coefficient", args, polynomial([f[0], f[1] + 1, *f[2:]]), ()),
            ("the sum from k = 0", args, polynomial([f[0] + p[0], *f[1:]]), ()),
            ("the sum up to n - 1", args, polynomial(trimmed([a - b for a, b in zip(f, [*p, 0])])), ()),
            ("the sum up to n + 1", args, polynomial(trimmed(shifted(f, 1))), ()),
            ("twice the sum", args, polynomial([2 * c for c in f]), ()),
            ("the sum of another polynomial", args, polynomial(sum_reference([Fraction(3, 2), -1, 3])), ()),
            ("degree d + 2 with the values of F at 0 .. d + 1", args, polynomial(crafted), ()),
            ("one degree too low", args, polynomial(f[:-1]), ()),
            ("the zero polynomial", args, polynomial((0,)), ()),
            ("p itself", args, polynomial(p), ()),
            ("the sum in another variable", args, polynomial(f, Y), ()),
            ("the sum in another Symbol named x", args, polynomial(f, Symbol("x")), ()),
            ("the value as evidence", args, value, (value,)),
            ("evidence where none belongs", args, value, ((),)),
        ])
        # The card's example, F = 0 for p = x - 1 (which the card's point test at k = 2 refuses as well), and, for
        # p = 0, polynomials with F(0) = 0: p = 0 has no points to test, and the checker requires F = 0.
        self.assertAccepts(*sum_claim((-1, 1)))
        self.assertRejectsAll(operation, [
            ("F = 0 for p = x - 1", {"polynomial": polynomial((-1, 1))}, polynomial((0,)), ()),
            ("F = x for p = 0", {"polynomial": polynomial((0,))}, polynomial((0, 1)), ()),
            ("F = x^2 - x for p = 0", {"polynomial": polynomial((0,))}, polynomial((0, -1, 1)), ()),
            ("F = 1 for p = 0", {"polynomial": polynomial((0,))}, polynomial((1,)), ()),
            ("F = x for p = 1, in another variable", {"polynomial": polynomial((1,))}, polynomial((0, 1), Y), ()),
        ])

    def test_geometric_sum_rejects_corrupted_claims(self):
        first, ratio, count = Fraction(3, 2), Fraction(-2, 3), 7
        args = sum_args(first, ratio, count)
        value = geometric_terms(first, ratio, count)
        self.assertTypesMatter("geometric_sum", args, rational(value), ())
        self.assertTypesMatter("geometric_sum", sum_args(1, 2, 10), Rational(1023), ())
        self.assertTypesMatter("geometric_sum", sum_args(5, 1, 4), Rational(20), ())
        self.assertTypesMatter("geometric_sum", sum_args(3, 7, 0), Rational(0), ())
        closed = first * (1 - ratio ** count) / (1 - ratio)
        self.assertEqual(closed, value)
        self.assertRejectsAll("geometric_sum", [
            ("one term more", args, rational(geometric_terms(first, ratio, count + 1)), ()),
            ("one term fewer", args, rational(geometric_terms(first, ratio, count - 1)), ()),
            ("the last term", args, rational(first * ratio ** (count - 1)), ()),
            ("the opposite sign", args, rational(-value), ()),
            ("plus one", args, rational(value + 1), ()),
            ("a sign error in the closed form", args, rational(first * (1 + ratio ** count) / (1 - ratio)), ()),
            ("the infinite series", args, rational(first / (1 - ratio)), ()),
            ("1024 for 1023", sum_args(1, 2, 10), Rational(1024), ()),
            ("2047 for 1023", sum_args(1, 2, 10), Rational(2047), ()),
            ("7/8 for 7/4", sum_args(1, Fraction(1, 2), 3), Rational(7, 8), ()),
            ("ratio 1: one term more", sum_args(5, 1, 4), Rational(25), ()),
            ("ratio 1: the first term", sum_args(5, 1, 4), Rational(5), ()),
            ("ratio 1: the count", sum_args(5, 1, 4), Rational(4), ()),
            ("ratio -1: the even sum", sum_args(2, -1, 5), Rational(0), ()),
            ("ratio -1: the opposite sign", sum_args(2, -1, 5), Rational(-2), ()),
            ("ratio 0: zero", sum_args(3, 0, 5), Rational(0), ()),
            ("count 0: the first term", sum_args(3, 7, 0), Rational(3), ()),
            ("count 0: one", sum_args(3, 7, 0), Rational(1), ()),
            ("first 0: not zero", sum_args(0, 2, 10), Rational(1023), ()),
            ("evidence where none belongs", args, rational(value), ((),)),
            ("the value as evidence", args, rational(value), (rational(value),)),
        ])
        # A negative count has no sum, whatever the value.
        self.assertRejectsAll("geometric_sum", [
            (f"count {count}, value {value}", sum_args(1, ratio, count), rational(value), ())
            for count in (-1, -3) for ratio in (2, 1, 0, Fraction(1, 2))
            for value in (0, 1, -1, count, Fraction(1, 2))])

    def test_geometric_series_rejects_corrupted_claims(self):
        args = series_args(3, Fraction(-1, 3))
        self.assertTypesMatter("geometric_series", args, Rational(9, 4), ())
        self.assertTypesMatter("geometric_series", series_args(1, Fraction(1, 2)), Rational(2), ())
        self.assertTypesMatter("geometric_series", series_args(0, 5), Rational(0), ())
        self.assertRejectsAll("geometric_series", [
            ("a/(1 + r)", args, Rational(9, 2), ()),
            ("a(1 - r)", args, Rational(4), ()),
            ("plus one", args, Rational(13, 4), ()),
            ("the opposite sign", args, Rational(-9, 4), ()),
            ("the reciprocal", args, Rational(4, 9), ()),
            ("the first term", args, Rational(3), ()),
            ("first 0: not zero", series_args(0, Fraction(1, 2)), Rational(1), ()),
            ("first 0, divergent ratio: not zero", series_args(0, 5), Rational(-1, 4), ()),
            ("evidence where none belongs", args, Rational(9, 4), ((),)),
        ])
        # A divergent series has no value, not even its formal a/(1 - r) or its Abel sum.
        self.assertRejectsAll("geometric_series", [
            (f"first {a}, ratio {r}, value {v}", series_args(a, r), rational(v), ())
            for a, r, values in ((1, 1, (0, Fraction(1, 2), 1, -1)), (1, -1, (Fraction(1, 2), 0, 1)),
                                 (1, 2, (-1, 0)), (1, Fraction(-3, 2), (Fraction(2, 5),)), (-4, 5, (1,)),
                                 (Fraction(1, 3), Fraction(7, 3), (Fraction(-1, 4),)))
            for v in values])

    def test_polynomial_geometric_series_rejects_corrupted_claims(self):
        p, r = [Fraction(1, 2), -3, 1], Fraction(-2, 5)  # x^2 - 3x + 1/2
        operation, args, value, evidence = pgs_claim(p, r)
        q = q_reference(p, r)
        self.assertEqual(series_reference(p, r), -q[0])
        self.assertTypesMatter(operation, args, value, evidence)
        self.assertTypesMatter(*pgs_claim((5,), Fraction(1, 3)))
        self.assertTypesMatter("polynomial_geometric_series", {"polynomial": polynomial((0,)), "ratio": 3},
                               Rational(0), ((Rational(0),),))
        # E of degree d + 1 through (k, 7 r^(-k)), k = 0, ..., d + 1, has r E(k + 1) = E(k) there: Q + E passes the
        # card's point test at 0, ..., d, with the value s - 7; the degree test refuses it, and the identity would.
        e = through([(k, 7 / r ** k) for k in range(4)])
        crafted = [a + b for a, b in zip([*q, 0], e)]
        self.assertEqual(len(trimmed(crafted)), 4)
        self.assertTrue(all(r * evaluate(crafted, k + 1) - evaluate(crafted, k) == evaluate(p, k) for k in range(3)))
        other_ratio, other_polynomial = q_reference(p, Fraction(-1, 5)), q_reference([Fraction(3, 2), -3, 1], r)
        q_cells = evidence[0]

        def entries(*values: object) -> tuple:
            return (values,)

        self.assertRejectsAll(operation, [
            ("plus one", args, rational(-q[0] + 1), evidence),
            ("Q(0) instead of -Q(0)", args, rational(q[0]), evidence),
            ("the value of another ratio", args, rational(series_reference(p, Fraction(-1, 5))), evidence),
            ("Q and the value of another ratio", args, rational(-other_ratio[0]), evidence_of(other_ratio)),
            ("Q and the value of another polynomial", args, rational(-other_polynomial[0]),
             evidence_of(other_polynomial)),
            ("Q + 1 and its value", args, rational(-q[0] - 1), evidence_of([q[0] + 1, *q[1:]])),
            ("Q with another leading coefficient", args, value, evidence_of([*q[:-1], 2 * q[-1]])),
            ("Q + E of degree d + 1, which passes the card's point test", args, rational(-crafted[0]),
             evidence_of(crafted)),
            ("Q + E with the true value", args, value, evidence_of(crafted)),
            ("Q one degree too low", args, rational(-q[0]), evidence_of(q[:-1])),
            ("Q with a trailing zero", args, value, entries(*q_cells, Rational(0))),
            ("Q with an int for its last entry", args, value, entries(*q_cells[:-1], 1)),
            ("Q with a Fraction entry", args, value, entries(q[0], *q_cells[1:])),
            ("Q with a float entry", args, value, entries(float(q[0]), *q_cells[1:])),
            ("Q with a str entry", args, value, entries(str(q[0]), *q_cells[1:])),
            ("Q with a bool entry", args, value, entries(*q_cells[:-1], True)),
            ("Q with a Rational subclass entry", args, value,
             entries(Ratio(q[0].numerator, q[0].denominator), *q_cells[1:])),
            ("Q with an entry not in lowest terms", args, value,
             entries(support.forged(Rational, numerator=2 * q[0].numerator, denominator=2 * q[0].denominator),
                     *q_cells[1:])),
            ("Q with a negative denominator", args, value,
             entries(support.forged(Rational, numerator=-q[0].numerator, denominator=-q[0].denominator),
                     *q_cells[1:])),
            ("Q with an impostor entry", args, value, entries(support.impostor(q_cells[0]), *q_cells[1:])),
            ("Q as a list", args, value, (list(q_cells),)),
            ("Q as a tuple subclass", args, value, (Evidence(q_cells),)),
            ("Q unwrapped", args, value, q_cells),
            ("Q as a Polynomial", args, value, (polynomial(q),)),
            ("Q twice", args, value, (q_cells, q_cells)),
            ("no Q", args, value, ()),
            ("an empty Q", args, value, ((),)),
            ("None as Q", args, value, (None,)),
            ("the value as evidence", args, value, (value,)),
        ])
        # (x, 1/2): Q = -2x - 2 has integral entries, which must still be Rationals; p = 0 needs Rational(0) too.
        integral = {"polynomial": polynomial((0, 1)), "ratio": Rational(1, 2)}
        zero_args = {"polynomial": polynomial((0,)), "ratio": Rational(1, 2)}
        self.assertAccepts(operation, integral, Rational(2), ((Rational(-2), Rational(-2)),))
        self.assertRejectsAll(operation, [
            ("Q of ints", integral, Rational(2), ((-2, -2),)),
            ("Q with one int", integral, Rational(2), ((Rational(-2), -2),)),
            ("Q of bools and ints", integral, Rational(2), ((-2, True),)),
            ("p = 0, Q = (0,) of an int", zero_args, Rational(0), ((0,),)),
            ("p = 0, the value as an int", zero_args, 0, ((Rational(0),),)),
            ("p = 0, Q = (False,)", zero_args, Rational(0), ((False,),)),
        ])
        # Divergent series: the identity holds for the true Q of the ratio, and the formal value is still refused.
        divergent = []
        for terms, ratio in (((1,), 2), ((0, 1), -1), ((1,), -1), ((1, 1), Fraction(3, 2)), ((2,), -3)):
            formal = q_reference(terms, ratio)
            divergent.append((f"p = {terms}, ratio {ratio}", {"polynomial": polynomial(terms), "ratio": exact(ratio)},
                              rational(-formal[0]), evidence_of(formal)))
        self.assertEqual(divergent[1][2], Rational(-1, 4))  # The Abel sum of 0 - 1 + 2 - 3 + ...
        # Ratio 1, where no Q exists for p != 0; and p = 0, where Q must be 0: Q = 1 satisfies 1*Q(x + 1) - Q(x) = 0,
        # but r^n Q(n) = 1 does not tend to 0, so its value -1 is no sum.
        zero = {"polynomial": polynomial((0,)), "ratio": 1}
        divergent += [
            ("p = 1, ratio 1, Q = 0", {"polynomial": polynomial((1,)), "ratio": 1}, Rational(0), ((Rational(0),),)),
            ("p = 1, ratio 1, Q = 1", {"polynomial": polynomial((1,)), "ratio": 1}, Rational(-1), ((Rational(1),),)),
            ("p = 0, ratio 1, Q = 1", zero, Rational(-1), ((Rational(1),),)),
            ("p = 0, ratio 1, Q = 1 with the value 0", zero, Rational(0), ((Rational(1),),)),
            ("p = 0, value 1", zero, Rational(1), ((Rational(0),),)),
            ("p = 0, an empty Q", zero, Rational(0), ((),)),
            ("p = 0, Q = (0, 0)", zero, Rational(0), ((Rational(0), Rational(0)),)),
            ("p = 0, Q of degree 1", zero, Rational(0), ((Rational(0), Rational(1)),)),
            ("p = 0, no evidence", zero, Rational(0), ()),
        ]
        self.assertRejectsAll(operation, divergent)

    def test_rejects_wrong_operations_and_arguments(self):
        args, value = series_args(1, Fraction(1, 2)), Rational(2)
        self.assertAccepts("geometric_series", args, value)
        self.assertRejectsAll("geometric_series", [
            ("arguments as pairs", tuple(args.items()), value, ()),
            ("arguments as a list of pairs", list(args.items()), value, ()),
            ("arguments as a read-only mapping", MappingProxyType(args), value, ()),
            ("None as arguments", None, value, ()),
            ("an argument missing", {"first": 1}, value, ()),
            ("an extra argument", dict(args, count=3), value, ()),
            ("an argument renamed", {"first": 1, "r": Rational(1, 2)}, value, ()),
            ("an argument of another operation", {"polynomial": polynomial((1,)), "ratio": Rational(1, 2)}, value, ()),
        ])
        for operation in ("geometric_sum", "sum_polynomial", "polynomial_geometric_series", "Geometric_series",
                          "geometric_series ", "geometric", "series", "continued_fraction", "", None, 7,
                          ("geometric_series",), b"geometric_series", ["geometric_series"]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value)
        # True claims of one operation offered as claims of another.
        constant = {"polynomial": polynomial((1,)), "ratio": Rational(1, 2)}
        self.assertAccepts("polynomial_geometric_series", constant, Rational(2), ((Rational(-2),),))
        self.assertRejects("polynomial_geometric_series", constant, Rational(2))  # The series has no evidence.
        self.assertRejects("geometric_sum", sum_args(1, Fraction(1, 2), 3), Rational(2))  # The infinite sum.
        self.assertRejects("sum_polynomial", {"polynomial": polynomial((1,))}, Rational(1))  # A Rational for F.
        self.assertRejects("geometric_series", args, polynomial((2,)))  # A Polynomial for a Rational.

    def test_oversized_forgeries_fail_fast(self):
        p = [Fraction(1, 2), -1, 3]
        f = sum_reference(p)
        small = {"polynomial": polynomial(p)}
        q_operation, q_args, q_value, q_evidence = pgs_claim(p, Fraction(-2, 5))
        q = q_evidence[0]
        rng = random.Random(20261117)
        big_f = [0] + [rng.randrange(-99, 100) for _ in range(300)] + [1]
        _, big_args, big_value, _ = backward_sum_claim(big_f)
        big_q = [rng.randrange(-99, 100) for _ in range(300)] + [1]
        _, big_q_args, big_q_value, big_q_evidence = backward_series_claim(big_q, Fraction(-2, 5))
        coprime = [Fraction(1, LONG + 2 * i) for i in range(3)]  # Odd, pairwise coprime denominators.
        top = polynomial([1] * 1025)  # Degree 1024: its sum would have degree 1025, beyond every Polynomial.
        cases = [
            ("sum_polynomial", small, support.forged(Polynomial, coefficients=(Rational(0),) * 100_000 + (Rational(1),),
                                                     variable=X), ()),
            ("sum_polynomial", small, polynomial([*f[:-1], GIANT]), ()),
            ("sum_polynomial", small, polynomial([0, Fraction(1, GIANT), *f[2:]]), ()),
            ("sum_polynomial", small, polynomial([0, LONG, *f[2:]]), ()),
            ("sum_polynomial", small, polynomial([0, *coprime[:2], f[3]]), ()),
            ("sum_polynomial", small, polynomial([*f, LONG]), ()),
            ("sum_polynomial", small, polynomial(f), (0,) * 100_000),
            ("sum_polynomial", small, polynomial(f), (GIANT,)),
            ("sum_polynomial", {"polynomial": top},
             support.forged(Polynomial, coefficients=(Rational(0),) + (Rational(1),) * 1025, variable=X), ()),
            ("sum_polynomial", big_args, polynomial([0, LONG, *big_value.coefficients[2:]]), ()),
            ("sum_polynomial", big_args, polynomial([0, *big_value.coefficients[1:-1], Fraction(1, LONG)]), ()),
            ("geometric_sum", sum_args(1, 2, 10), Rational(GIANT), ()),
            ("geometric_sum", sum_args(1, 2, 10), Rational(1, GIANT), ()),
            ("geometric_sum", sum_args(1, 2, 10), Rational(LONG), ()),
            ("geometric_sum", sum_args(1, 2, 10), Rational(LONG, LONG + 2), ()),
            ("geometric_sum", sum_args(1, 2, 10), Rational(1023), (GIANT,)),
            ("geometric_sum", sum_args(1, 2, 10), Rational(1023), (0,) * 100_000),
            ("geometric_series", series_args(1, Fraction(1, 2)), Rational(GIANT), ()),
            ("geometric_series", series_args(1, Fraction(1, 2)), Rational(LONG, LONG + 2), ()),
            ("geometric_series", series_args(1, Fraction(1, 2)), Rational(2), (0,) * 100_000),
            (q_operation, q_args, Rational(GIANT), q_evidence),
            (q_operation, q_args, q_value, ((Rational(0),) * 100_000,)),
            (q_operation, q_args, q_value, ((Rational(GIANT),) + q[1:],)),
            (q_operation, q_args, Rational(-LONG), ((Rational(LONG),) + q[1:],)),
            (q_operation, q_args, q_value, (q[:1] + tuple(map(rational, coprime[:2])),)),
            (q_operation, q_args, q_value, (q,) * 100_000),
            (q_operation, big_q_args, big_q_value, ((big_q_evidence[0][0], Rational(LONG)) + big_q_evidence[0][2:],)),
            (q_operation, big_q_args, big_q_value, (big_q_evidence[0][:-1] + (Rational(1, LONG),),)),
        ]
        for index, (operation, args, value, evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                self.assertRejects(operation, args, value, evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)

    def test_resource_limits(self):
        """Where a claim is beyond the budget rather than wrong, the checker raises ResourceLimitError, fast: the
        card's resource limit of geometric_sum, and arguments longer than max_integer_bits, which the producer's
        budget refuses too."""
        cases = [
            ("geometric_sum", sum_args(1, 2, 10 ** 6), Rational(1), ()),
            ("geometric_sum", sum_args(1, Fraction(-3, 2), 2 ** 40), Rational(1), ()),
            ("geometric_sum", sum_args(1, 2 ** 1000 + 1, 10 ** 4), Rational(1), ()),
            ("geometric_sum", sum_args(GIANT, 2, 3), Rational(7), ()),
            ("geometric_sum", sum_args(1, 2, GIANT), Rational(1), ()),
            ("geometric_series", series_args(1, Fraction(1, GIANT)), Rational(1), ()),
            ("sum_polynomial", {"polynomial": polynomial((GIANT,))}, polynomial((0, GIANT)), ()),
            ("polynomial_geometric_series", {"polynomial": polynomial((1,)), "ratio": rational(Fraction(1, GIANT))},
             Rational(1), ((Rational(-1),),)),
        ]
        # A polynomial argument whose coefficients have a common denominator longer than max_integer_bits.
        wide = [Fraction(1, 2 ** 40_000 + 1), Fraction(1, 2 ** 40_000 + 3)]
        cases.append(("sum_polynomial", {"polynomial": polynomial(wide)}, polynomial((0, 1, 1)), ()))
        cases.append(("polynomial_geometric_series",
                      {"polynomial": polynomial(wide), "ratio": rational(Fraction(1, 2))}, Rational(1),
                      ((Rational(-1), Rational(1)),)))
        for operation, args, value, evidence in cases:
            with self.subTest(operation=operation, args=sorted(args)):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence)
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_before_it_runs(self):
        rng = random.Random(20261118)
        f = [0] + [rng.randrange(-99, 100) for _ in range(200)] + [3]
        claim = backward_sum_claim(f)
        with self.assertRaises(ResourceLimitError):  # About 7,100 units, 5,076 of them for the shift, paid before it.
            support.verdict(family, *claim, limits=ComputationLimits(max_work=5_000))
        self.assertAccepts(*claim)
        # The charges depend on the claim alone: a budget one unit short of a check refuses it, its exact size passes.
        claims = [claim, backward_series_claim([rng.randrange(-9, 10) for _ in range(80)] + [1], Fraction(3, 7)),
                  ("geometric_sum", sum_args(Fraction(5, 7), Fraction(-3, 2), 3000),
                   rational(geometric_terms(Fraction(5, 7), Fraction(-3, 2), 3000)), ()),
                  ("geometric_sum", sum_args(1, 1, 2 ** 2000), Rational(2 ** 2000), ()),
                  ("geometric_series", series_args(Fraction(2 ** 900, 3), Fraction(-5, 7 ** 300)),
                   rational(Fraction(2 ** 900, 3) / (1 + Fraction(5, 7 ** 300))), ())]
        for operation, args, value, evidence in claims:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, evidence), True)
                work = budget.work
                self.assertGreater(work, 1)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, value, evidence, limits=ComputationLimits(max_work=work))

    def test_the_bounds_refuse_before_the_shift(self):
        """A forged claim of full length whose integers exceed the denominator bound or the size bound of the module's
        theorems is refused before the quadratic shift runs: fast, under the default budget. A true claim of the same
        length runs one shift, and so does a wrong claim within the bounds."""
        rng = random.Random(20261119)
        f = [0] + [rng.randrange(-99, 100) for _ in range(1023)] + [1]
        operation, args, value, evidence = backward_sum_claim(f)
        q = [rng.randrange(-99, 100) for _ in range(1024)] + [1]
        q_operation, q_args, q_value, q_evidence = backward_series_claim(q, Fraction(-2, 5))
        longs = [LONG + 2 * i for i in range(1025)]  # 60,001 bits: within max_integer_bits, beyond both size bounds.
        short = 2 ** 5000 + 1  # Odd: consecutive odd denominators are coprime, and two of them pass the bound.
        forgeries = [
            ("F with long numerators", operation, args, polynomial([0, *longs[1:]]), ()),
            ("F with long denominators", operation, args, polynomial([0, *(Fraction(1, n) for n in longs[1:])]), ()),
            ("F with coprime denominators", operation, args,
             polynomial([0, *(Fraction(1, short + 2 * i) for i in range(1024))]), ()),
            ("Q with long numerators", q_operation, q_args, Rational(-longs[0]), (tuple(map(Rational, longs)),)),
            ("Q with long denominators", q_operation, q_args, Rational(-1, longs[0]),
             (tuple(Rational(1, n) for n in longs),)),
        ]
        with patch.object(checker, "_shift", wraps=checker._shift) as shift:
            for label, *claim in forgeries:
                with self.subTest(case=label):
                    start = time.perf_counter()
                    self.assertRejects(*claim)
                    self.assertLess(time.perf_counter() - start, FAST)
            self.assertEqual(shift.call_count, 0)
            self.assertAccepts(operation, args, value, evidence)
            self.assertAccepts(q_operation, q_args, q_value, q_evidence)
            self.assertEqual(shift.call_count, 2)
            self.assertRejects(operation, args, polynomial([0, f[1] + 1, *f[2:]]), evidence)  # Within the bounds.
            self.assertEqual(shift.call_count, 3)

    def test_the_shift_is_the_taylor_shift(self):
        """checker._shift gives the coefficients of P(x + 1) and of P(x - 1), as the binomial theorem gives them, and
        keeps them below 2^(bits + length), the bound that its charge assumes; it leaves its argument unchanged."""
        rng = random.Random(20261120)
        for length in (1, 2, 3, 7, 40, 200):
            for bits in (1, 64, 3000):
                integers = [rng.getrandbits(bits) - (1 << (bits - 1)) for _ in range(length)]
                copy, top = list(integers), max(c.bit_length() for c in integers)
                for step in (1, -1):
                    with self.subTest(length=length, bits=bits, step=step), computation(fresh=True) as budget:
                        result = checker._shift(budget, integers, top, step)
                        self.assertEqual(result, shifted(integers, step))
                        self.assertTrue(all(c.bit_length() <= top + length for c in result))
                        self.assertEqual(integers, copy)
                        self.assertGreater(budget.work, 0)

    def test_foreign_code_never_runs(self):
        """Objects whose comparison, hashing or attributes would run foreign code answer False, never an exception: a
        Symbol forged with such a field, a class that names its module with such an object, a class of the record's
        module and name whose fields are properties that raise, and argument keys that are no str."""

        class Hostile:
            def __eq__(self, other: object) -> bool:
                raise AssertionError("compared")

            def __hash__(self) -> int:
                return hash("polynomial")

        class Computed:
            @property
            def coefficients(self) -> tuple:
                raise AssertionError("computed")

            @property
            def variable(self) -> Symbol:
                raise AssertionError("computed")

        Computed.__module__, Computed.__qualname__ = checker.POLYNOMIAL
        named = type("Polynomial", (), {"__module__": Hostile()})()
        p, f = polynomial((1, 2)), polynomial(sum_reference((1, 2)))
        named.coefficients, named.variable = f.coefficients, f.variable

        def symbol(**fields: object) -> Symbol:
            defaults = {"name": "x", "domain": X.domain, "assumptions": (), "uid": X.uid}
            return support.forged(Symbol, **{**defaults, **fields})

        def record(variable: object, coefficients: tuple = f.coefficients) -> Polynomial:
            return support.forged(Polynomial, coefficients=coefficients, variable=variable)

        assumption = support.forged(Assumption, name="positive", truth=Hostile())
        symbols = [("a hostile domain", symbol(domain=Hostile())), ("a hostile name", symbol(name=Hostile())),
                   ("a hostile uid", symbol(uid=Hostile())), ("hostile assumptions", symbol(assumptions=Hostile())),
                   ("a hostile assumption", symbol(assumptions=(Hostile(),))),
                   ("an assumption with a hostile truth", symbol(assumptions=(assumption,))),
                   ("four assumptions", symbol(assumptions=N.assumptions * 4))]
        cases = [(f"value: {label}", "sum_polynomial", {"polynomial": p}, record(item), ()) for label, item in symbols]
        cases += [(f"argument: {label}", "sum_polynomial", {"polynomial": record(item, p.coefficients)}, f, ())
                  for label, item in symbols]
        cases += [
            ("a module named by a hostile object", "sum_polynomial", {"polynomial": p}, named, ()),
            ("a value whose fields raise", "sum_polynomial", {"polynomial": p}, Computed(), ()),
            ("an argument whose fields raise", "sum_polynomial", {"polynomial": Computed()}, f, ()),
            ("an argument key that is no str", "sum_polynomial", {Hostile(): p}, f, ()),
            ("a str key and a hostile one", "geometric_series", {"first": 1, Hostile(): Rational(1, 2)}, Rational(2),
             ()),
            ("a polynomial argument of the series whose fields raise", "polynomial_geometric_series",
             {"polynomial": Computed(), "ratio": Rational(1, 2)}, Rational(-2), ((Rational(2), Rational(-2)),)),
        ]
        self.assertAccepts("sum_polynomial", {"polynomial": p}, f)
        for label, *claim in cases:
            with self.subTest(case=label):
                self.assertRejects(*claim)

    def test_the_checker_forms_no_power_and_imports_no_producer(self):
        tree = ast.parse(inspect.getsource(checker))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))
                  and isinstance(node.op, ast.Pow)]
        self.assertEqual(powers, [])  # Every power is a chain of charged products.
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertLessEqual(imported, {"__future__", "collections.abc", "math", "mathforge.limits",
                                        "mathforge.model", "mathforge.trusted.integers"})

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layouts
        survive the format, and a wrong value fails the mathematics check."""
        claims = [
            (*sum_claim((0, 0, 1), N), polynomial((0, 1), N)),
            ("geometric_sum", sum_args(1, 2, 10), Rational(1023), (), Rational(1024)),
            ("geometric_series", series_args(3, Fraction(-1, 3)), Rational(9, 4), (), Rational(9, 2)),
            (*pgs_claim((0, 1), Fraction(-1, 2)), Rational(2, 9)),
            ("polynomial_geometric_series", {"polynomial": polynomial((0,)), "ratio": 7}, Rational(0),
             ((Rational(0),),), Rational(1)),
        ]
        for operation, args, value, evidence, wrong in claims:
            with self.subTest(operation=operation):
                self.roundtrip(claimed(operation, args, value, evidence))
                report = support.statuses(mf.verify(claimed(operation, args, wrong, evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
