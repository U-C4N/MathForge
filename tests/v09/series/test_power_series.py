"""The power_series family of mathforge.series: series_coefficients.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register series power_series

Every expected value comes from the card, from closed forms of the textbooks, or from reference() of this module,
never from the producer: Taylor coefficients by Horner's rule over fractions.Fraction, each function's Maclaurin series
(closed forms, and for tan and tanh the Bernoulli numbers of the Akiyama-Tanigawa algorithm) composed with the series
of its argument about the argument's value at the center. That is the method of neither the producer (Newton
iteration, monomial tables, Brent and Harvey's tangent numbers, Kronecker products) nor the checker (the first-order
ODE recurrences of mathforge.certify.series.taylor), and test_reference_agrees_with_identities checks it against
identities that it does not build in. tests/v09/series/test_power_series_checker.py tests the checker on its own.
"""

from __future__ import annotations

import ast
from fractions import Fraction
import inspect
from math import comb, factorial, isqrt
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import functions as f
from mathforge import lean
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.series import power_series as family
from mathforge.contracts import CheckStatus, ExecutionStatus, Outcome
from mathforge.deferred import Unevaluated
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.kernels import series as kernel
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Rational, Reals, Symbol, Truth, trusted_rational
from mathforge.polynomial import Polynomial
from mathforge.rational_function import RationalFunction

from .. import support

OP = "series_coefficients"
FAST = 2.0  # Seconds: a request that its budget refuses fails well within this.
X, Y = Symbol("x"), Symbol("y")
F0, F1 = Fraction(0), Fraction(1)
T = ("x",)  # The tree of the variable; reference() describes the trees.
VANISHING = ("exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "atan", "asin")  # Expanded about an argument of 0.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # A float, a bool, a str, None and two containers.
CENTERS = (0, 0, 1, -1, 2, Fraction(1, 2), Fraction(-2, 3), Fraction(7, 5))


class Integer(int):
    """An int subclass, equal to an exact int but none."""


class Ratio(Rational):
    """A Rational subclass, equal in value to a Rational but no exact Rational."""
    __slots__ = ()


# ---------------------------------------------------------------------------------------------------------------
# The reference: Horner's rule over Fractions with Maclaurin series of closed forms.

def fraction(number: object) -> Fraction:
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def rational(number: object) -> Rational:
    number = fraction(number)
    return Rational(number.numerator, number.denominator)


def exact(number: object) -> int | Rational:
    """An argument of the kind "rational": an int stays an int, anything else becomes a Rational."""
    return number if type(number) is int else rational(number)


def constant(value: object, n: int) -> tuple[Fraction, ...]:
    return (Fraction(value),) + (F0,) * (n - 1)


def times(a: tuple, b: tuple) -> tuple[Fraction, ...]:
    """The product of two truncated series of one length, truncated to it, by the schoolbook loop."""
    out = [F0] * len(a)
    for i, u in enumerate(a):
        for j in range(len(a) - i):
            out[i + j] += u * b[j]
    return tuple(out)


def horner(table: list[Fraction], v: tuple) -> tuple[Fraction, ...]:
    """sum table[k] v^k for a truncated series v without constant term, by Horner's rule: v^k starts at t^k, so the
    first len(v) terms of the table are all that reach the series."""
    assert v[0] == 0
    out = constant(0, len(v))
    for coefficient in reversed(table[:len(v)]):
        out = times(out, v)
        out = (out[0] + coefficient,) + out[1:]
    return out


def bernoulli(n: int) -> list[Fraction]:
    """B_0, ..., B_n by the Akiyama-Tanigawa algorithm (B_1 = +1/2 there; only the even ones are used)."""
    row, out = [F0] * (n + 1), []
    for m in range(n + 1):
        row[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            row[j - 1] = j * (row[j - 1] - row[j])
        out.append(row[0])
    return out


def binomial(r: object, k: int) -> Fraction:
    """r (r - 1) ... (r - k + 1)/k! for any rational r."""
    out = F1
    for i in range(k):
        out = out * (Fraction(r) - i) / (i + 1)
    return out


def maclaurin(name: str, n: int) -> list[Fraction]:
    """The first n Maclaurin coefficients at 0 of exp, sin, cos, sinh, cosh, tan, tanh, atan and asin, and those of
    log(1 + t). tan t = sum (-1)^(m-1) 4^m (4^m - 1) B_2m t^(2m-1)/(2m)!, tanh without the sign."""
    numbers = bernoulli(n + 1) if name in ("tan", "tanh") else []
    table = []
    for k in range(n):
        odd = k % 2 == 1
        if name == "exp":
            c = Fraction(1, factorial(k))
        elif name in ("sin", "sinh"):
            c = Fraction((-1) ** (k // 2) if name == "sin" else 1, factorial(k)) if odd else F0
        elif name in ("cos", "cosh"):
            c = F0 if odd else Fraction((-1) ** (k // 2) if name == "cos" else 1, factorial(k))
        elif name in ("tan", "tanh"):
            m = (k + 1) // 2
            sign = (-1) ** (m - 1) if name == "tan" else 1
            c = sign * 4 ** m * (4 ** m - 1) * numbers[2 * m] / factorial(2 * m) if odd else F0
        elif name == "atan":
            c = Fraction((-1) ** (k // 2), k) if odd else F0
        elif name == "asin":
            c = Fraction(comb(k - 1, (k - 1) // 2), 4 ** ((k - 1) // 2) * k) if odd else F0
        elif name == "log":
            c = Fraction((-1) ** (k + 1), k) if k else F0
        else:
            raise ValueError(name)
        table.append(c)
    return table


def square_root(value: Fraction) -> Fraction:
    p, q = isqrt(value.numerator), isqrt(value.denominator)
    assert value > 0 and p * p == value.numerator and q * q == value.denominator, value
    return Fraction(p, q)


def power(u: tuple, m: int) -> tuple[Fraction, ...]:
    """u^m for an int m: the binomial series about u[0] when u[0] != 0 (a negative m needs it), repeated products
    when u[0] = 0."""
    head, v = u[0], (F0,) + tuple(u[1:])
    if head:
        return horner([binomial(m, k) * head ** (m - k) for k in range(len(u))], v)
    assert m >= 0
    out = constant(1, len(u))
    for _ in range(min(m, len(u))):
        out = times(out, u)
    return out


def reference(tree: tuple, center: object, n: int) -> tuple[Fraction, ...]:
    """The first n Taylor coefficients at the center of the function of a tree: ("x",), ("c", rational),
    (op, tree, tree) for add, mul and div, ("pow", tree, int), or (name, tree) for exp, sin, cos, tan, sinh, cosh,
    tanh, atan, asin, log, sqrt and abs; every inner value must be one about which its function is expanded."""
    op = tree[0]
    if op == "x":
        return (fraction(center), F1, *([F0] * (n - 2)))[:n]
    if op == "c":
        return constant(tree[1], n)
    if op in ("add", "mul", "div"):
        a, b = reference(tree[1], center, n), reference(tree[2], center, n)
        if op == "add":
            return tuple(p + q for p, q in zip(a, b))
        return times(a, b if op == "mul" else power(b, -1))
    if op == "pow":
        return power(reference(tree[1], center, n), tree[2])
    u = reference(tree[1], center, n)
    head, v = u[0], (F0,) + u[1:]
    if op == "abs":
        assert head != 0
        return u if head > 0 else tuple(-c for c in u)
    if op == "log":
        assert head == 1
        return horner(maclaurin("log", n), v)
    if op == "sqrt":
        q = square_root(head)
        return horner([binomial(Fraction(1, 2), k) * q ** (1 - 2 * k) for k in range(n)], v)
    assert head == 0, (op, head)
    return horner(maclaurin(op, n), u)


def expression(tree: tuple) -> f.RealExpr:
    """The RealExpr of a tree, built with the operators of mathforge.functions."""
    op = tree[0]
    if op == "x":
        return f.as_real_expr(X)
    if op == "c":
        return f.as_real_expr(rational(tree[1]))
    if op in ("add", "mul", "div"):
        a, b = expression(tree[1]), expression(tree[2])
        return a + b if op == "add" else a * b if op == "mul" else a / b
    if op == "pow":
        return expression(tree[1]) ** tree[2]
    return getattr(f, op)(expression(tree[1]))


def real(tree: tuple) -> f.RealFunction:
    """The RealFunction of a tree in X: real_function simplifies the expression and keeps its domain."""
    return f.real_function(expression(tree), (X,))


def parsed(text: str) -> f.RealFunction:
    return mf.parse_function(text, symbols={"x": X})


def shifted(tree: tuple, center: object, target: object) -> tuple:
    """The tree plus a constant, so that its value at the center is target."""
    offset = Fraction(target) - reference(tree, center, 1)[0]
    return tree if offset == 0 else ("add", tree, ("c", offset))


def random_tree(rng: random.Random, center: object, depth: int) -> tuple:
    """A seeded tree of at most depth levels of functions, every node in the card's scope at the center."""
    if depth == 0 or rng.random() < 0.2:
        return T if rng.random() < 0.65 else ("c", Fraction(rng.randint(-5, 5), rng.randint(1, 4)))
    child = random_tree(rng, center, depth - 1)
    kind = rng.choice(("add", "mul", "div", "pow", "abs", "log", "sqrt") + VANISHING)
    nonzero = Fraction(rng.choice((-3, -1, 1, 2, 5)), rng.choice((1, 2, 3)))
    if kind in ("add", "mul"):
        return (kind, child, random_tree(rng, center, depth - 1))
    if kind == "div":
        return ("div", random_tree(rng, center, depth - 1), shifted(child, center, nonzero))
    if kind == "pow":
        k = rng.choice((-3, -1, 0, 2, 3, 5))
        return ("pow", shifted(child, center, nonzero) if k < 0 else child, k)
    if kind == "abs":
        return ("abs", shifted(child, center, nonzero))
    if kind == "log":
        return ("log", shifted(child, center, 1))
    if kind == "sqrt":
        return ("sqrt", shifted(child, center, Fraction(rng.choice((1, 2, 3, 5)), rng.choice((1, 2))) ** 2))
    return (kind, shifted(child, center, 0))


def sin_exp(n: int) -> list[Fraction]:
    """The card's sin(x) exp(x) = Im(exp((1 + i) x)) at 0: Im((1 + i)^k)/k! = 2^(k/2) sin(k pi/4)/k!."""
    out, re, im = [], 1, 0
    for k in range(n):
        out.append(Fraction(im, factorial(k)))
        re, im = re - im, re + im  # Times 1 + i.
    return out


def args_of(function: object, center: object, order: int, variable: Symbol = X) -> dict[str, object]:
    return dict(function=function, variable=variable, center=exact(center), order=order)


def series(center: object, coefficients, variable: Symbol = X) -> family.PowerSeries:
    """The value of series_coefficients: the center always a Rational, every coefficient a Rational."""
    return family.PowerSeries(variable, rational(center), tuple(rational(c) for c in coefficients))


def request_of(args: dict[str, object]) -> mf.OperationRequest:
    """The request that operations_v6.run builds for the arguments, which an Unevaluated record carries."""
    return mf.OperationRequest(OP, tuple(args.items()), 6)


def removable() -> f.RealFunction:
    """(x^2 - 1)/(x - 1) as a RationalFunction made a RealFunction: x + 1 with the domain nonzero(x - 1)."""
    one = Polynomial((Rational(1),), X)
    quotient = (RationalFunction(Polynomial((Rational(-1), Rational(0), Rational(1)), X), one)
                / RationalFunction(Polynomial((Rational(-1), Rational(1)), X), one))
    return f.real_function(quotient)


def forged_function(expression_: object, domain: object = f.TRUE, variables: object = (X,)) -> f.RealFunction:
    """A RealFunction made with object.__new__: nothing walks or validates its tree or its domain."""
    return support.forged(f.RealFunction, expression=expression_, variables=variables, domain=domain)


def node(**fields: object) -> f.RealExpr:
    return support.forged(f.RealExpr, **fields)


# The card's known values: (function, center, order, coefficients).
KNOWN = (
    (parsed("exp(x)"), 0, 4, (1, 1, Fraction(1, 2), Fraction(1, 6), Fraction(1, 24))),
    (parsed("sin(x)*exp(x)"), 0, 12, tuple(sin_exp(13))),  # The v0.8 Taylor cliff.
    (parsed("log(x)"), 1, 3, (0, 1, Fraction(-1, 2), Fraction(1, 3))),
    *((parsed("1/(1 - x)"), 0, order, (1,) * (order + 1)) for order in (0, 1, 2, 5, 17, 40, 300)),
)

# Functions outside the card's scope at the center, each Unevaluated: (label, function, center).
UNEVALUATED = (
    ("exp(x) at 1, the card's", parsed("exp(x)"), 1),
    ("exp(1 + x) at 0", parsed("exp(1 + x)"), 0),
    ("cos(x) at 1", parsed("cos(x)"), 1),
    ("tan(x) at 1", parsed("tan(x)"), 1),
    ("atan(x) at 1", parsed("atan(x)"), 1),
    ("asin(x) at 1, the end of its domain", parsed("asin(x)"), 1),
    ("log(x) at 2", parsed("log(x)"), 2),
    ("sqrt(x) at 2", parsed("sqrt(x)"), 2),
    ("sqrt(x) at 0", parsed("sqrt(x)"), 0),
    ("sqrt(x**2) at 0, which is abs(x)", parsed("sqrt(x**2)"), 0),
    ("abs(x) at 0", parsed("abs(x)"), 0),
    ("1/x at 0", parsed("1/x"), 0),
    ("x**-1 at 0", parsed("x**-1"), 0),
    ("sin(x)/x at 0, the series of its limit", parsed("sin(x)/x"), 0),
    ("0/x at 0, which simplifies to 0 where x != 0", parsed("0/x"), 0),
    ("acos(x) at 0", parsed("acos(x)"), 0),
    ("acos(x) at 1", parsed("acos(x)"), 1),
    ("x + pi", parsed("x + pi"), 0),
    ("E*x", parsed("E*x"), 0),
    ("sqrt(2)*x", f.real_function(f.sqrt(2) * f.as_real_expr(X), (X,)), 0),
    ("a RealAlgebraicRoot scalar, even one equal to 2",
     f.real_function(f.as_real_expr(RealAlgebraicRoot((-2, 1), 0)) * f.as_real_expr(X), (X,)), 0),
    ("0*exp(x) at 1, in a tree that keeps the product",
     f.RealFunction(f.RealExpr("mul", (f.ZERO, f.exp(f.as_real_expr(X)))), (X,)), 1),
    ("2x where x > 0, at 0", f.real_function(2 * f.as_real_expr(X), (X,), domain=f.condition("positive",
                                                                                            f.as_real_expr(X))), 0),
    ("x where x >= 0, at 0: a domain on one side",
     f.real_function(f.as_real_expr(X), (X,), domain=f.DomainPredicate("nonnegative", (f.as_real_expr(X),))), 0),
    ("x + 1 where x != 1, at 1", removable(), 1),
    ("(x**2 - 1)/(x - 1) at 1", parsed("(x**2 - 1)/(x - 1)"), 1),
    ("the domain false", f.RealFunction(f.as_real_expr(X), (X,), f.FALSE), 0),
)


class PowerSeriesTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: family.PowerSeries) -> None:
        """A completed v6 Result of that record, with the evidence () and its four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), family.PowerSeries)
        self.assertEqual(result.steps[0].outputs[1], ())
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def assertUnevaluated(self, result: mf.Result, args: dict[str, object]) -> None:
        """A completed Result without a value: NO_CONCLUSION with the card's Unevaluated record, whose request is the
        Result's own, which mf.verify confirms as unevaluated (the mathematics stays unknown)."""
        self.assertIs(result.execution_status, ExecutionStatus.COMPLETED, result.error)
        self.assertIs(result.outcome, Outcome.NO_CONCLUSION)
        self.assertIsNone(result.value)
        self.assertEqual(result.steps, ())
        deferred = result.deferred
        self.assertIs(type(deferred), Unevaluated)
        self.assertEqual((deferred.kind, deferred.reason.code), ("series", "no_applicable_rule"))
        self.assertEqual(deferred.request, request_of(args))
        self.assertEqual(result.request, request_of(args))
        report = support.statuses(mf.verify(result))
        self.assertIs(report["result.source"], CheckStatus.VERIFIED)
        self.assertIs(report["result.unevaluated"], CheckStatus.VERIFIED)
        self.assertIs(report["result.mathematics"], CheckStatus.UNKNOWN)

    # -----------------------------------------------------------------------------------------------------------
    # The definition of done

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for function, center, order, coefficients in KNOWN:
            args = args_of(function, center, order)
            with self.subTest(function=str(function), center=center, order=order):
                self.assertEqual(len(coefficients), order + 1)
                if order <= 40:  # The card's values, by the reference too.
                    tree = {"exp(x)": ("exp", T), "(sin(x) * exp(x))": ("mul", ("sin", T), ("exp", T)),
                            "log(x)": ("log", T), "(1 / (1 + (-1 * x)))": ("div", ("c", 1), ("add", ("c", 1),
                                                                                             ("mul", ("c", -1), T)))}
                    self.assertEqual(reference(tree[str(function)], center, order + 1),
                                     tuple(Fraction(c) for c in coefficients))
                result = support.run(OP, **args)
                self.assertVerified(result, series(center, coefficients))
                self.roundtrip(result)
                # The center as a Rational argument: the same value, a Rational center in the record either way.
                self.assertEqual(support.run(OP, **dict(args, center=rational(center))).value, result.value)
        self.assertEqual(sin_exp(5), [0, 1, 1, Fraction(1, 3), 0])
        # The card's Unevaluated value: exp(x) at 1.
        args = args_of(parsed("exp(x)"), 1, 4)
        self.assertUnevaluated(support.run(OP, **args), args)

    def test_unevaluated_requests(self):
        """Functions out of the card's scope at the center, and functions whose domain holds no neighbourhood of it,
        are NO_CONCLUSION results without a value, whose Unevaluated record carries the request; produce() returns
        that record with the evidence (), the value function raises, and the lean path falls back."""
        for label, function, center in UNEVALUATED:
            args = args_of(function, center, 3)
            with self.subTest(case=label):
                result = support.run(OP, **args)
                self.assertUnevaluated(result, args)
                self.roundtrip_unevaluated(result)
                with computation(fresh=True):
                    value, evidence = family.produce(OP, args)
                self.assertEqual((value, evidence), (result.deferred, ()))
                self.assertIs(support.lean_value(OP, **args), lean.FALLBACK)
                with self.assertRaises(mf.OperationError):
                    support.value(OP, **args)
        details = {label: support.run(OP, **args_of(function, center, 2)).deferred.reason.detail
                   for label, function, center in UNEVALUATED}
        self.assertIn("vanishes at the center", details["exp(x) at 1, the card's"])
        self.assertIn("equal to 1", details["log(x) at 2"])
        self.assertIn("neighbourhood", details["2x where x > 0, at 0"])
        self.assertIn("false", details["the domain false"])

    def roundtrip_unevaluated(self, result: mf.Result) -> None:
        text = mf.dumps(result)
        decoded = mf.loads(text)
        self.assertEqual(decoded, result)
        self.assertEqual(mf.dumps(decoded), text)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error."""
        exp = parsed("exp(x)")
        base = args_of(exp, 0, 3)
        assumed = Symbol("x", assumptions=(Assumption("positive", Truth.TRUE),), uid=X.uid)
        invalid = [
            ("order -1", dict(base, order=-1), "order must be at least 0"),
            ("order -10**6", dict(base, order=-10 ** 6), "order must be at least 0"),
            ("another variable", dict(base, variable=Y), "only variable"),
            ("a variable of the same name", dict(base, variable=Symbol("x")), "only variable"),
            ("the variable with an assumption", dict(base, variable=assumed), "unconstrained real Symbol"),
            ("a function of two variables",
             dict(base, function=mf.parse_function("exp(x)*y", symbols={"x": X, "y": Y})), "only variable"),
            ("a function of x and y, y unused",
             dict(base, function=mf.parse_function("exp(x)", symbols={"x": X, "y": Y}, variables=(X, Y))),
             "only variable"),
            ("a constant without variables", dict(base, function=f.real_function(Rational(5))), "only variable"),
            ("a center not in lowest terms", dict(base, center=trusted_rational(2, 4)), "lowest terms"),
            ("a center with a negative denominator", dict(base, center=trusted_rational(1, -2)),
             "positive denominator"),
            ("a center with a zero denominator", dict(base, center=trusted_rational(1, 0)), "positive denominator"),
            ("the RealExpr instead of the function", dict(base, function=exp.expression), "RealFunction"),
            ("a Polynomial function", dict(base, function=Polynomial((Rational(1), Rational(1)), X)), "RealFunction"),
            ("a str function", dict(base, function="exp(x)"), "RealFunction"),
        ]
        for label, args, text in invalid:
            with self.subTest(case=label):
                result = support.run(OP, **args)
                self.assertInvalidInput(result)
                self.assertIn(text, result.error.message)
                with computation(fresh=True), self.assertRaises(InvalidInput):
                    family.produce(OP, args)
        # operations_v6.run rejects a stand-in by its kind (symbol, rational, int) before produce() runs, and
        # produce() rejects one in function; called directly, produce() refuses each itself.
        for name in base:
            for item in STAND_INS:
                with self.subTest(argument=name, item=item):
                    self.assertInvalidInput(support.run(OP, **dict(base, **{name: item})))
                    with computation(fresh=True), self.assertRaises(InvalidInput):
                        family.produce(OP, dict(base, **{name: item}))
        for name, item in (("order", Integer(3)), ("order", Rational(3)), ("center", Integer(1)),
                           ("center", Ratio(1, 2)), ("center", Fraction(1, 2)), ("variable", f.as_real_expr(X))):
            with self.subTest(argument=name, item=type(item).__name__), self.assertRaises(InvalidInput):
                with computation(fresh=True):
                    family.produce(OP, dict(base, **{name: item}))
        with self.assertRaises(InvalidInput):
            family.produce(OP, dict(base, point=1))
        with self.assertRaises(InvalidInput):
            family.produce(OP, {name: value for name, value in base.items() if name != "order"})
        with self.assertRaises(InvalidInput):
            family.produce(OP, tuple(base.items()))
        for operation in ("function_taylor", "taylor", "Series_coefficients", "", None, 7):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, base)
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(OP, **dict(base, order=-1))
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_functions_that_no_real_function_can_be(self):
        """A RealFunction forged with object.__new__ may hold a tree or a domain of classes and shapes that RealExpr and
        DomainPredicate refuse, another symbol, a non-canonical Rational or a node below itself: invalid input. run()
        passes the function kind to produce(); a cycle never gets that far (run() inspects the arguments first, and
        runs out of depth), so produce() is called directly for those."""
        x = f.as_real_expr(X)
        cyclic = node(op="add", args=(), value=None)
        object.__setattr__(cyclic, "args", (cyclic, x))
        first, second = node(op="sin", args=(), value=None), node(op="sin", args=(), value=None)
        object.__setattr__(first, "args", (second,))
        object.__setattr__(second, "args", (first,))
        atom = f.DomainPredicate("positive", (x + 1,))
        cyclic_domain = support.forged(f.DomainPredicate, kind="and", operands=())
        object.__setattr__(cyclic_domain, "operands", (atom, cyclic_domain))
        liar = support.forged(Symbol, name=type("Liar", (str,), {"__eq__": lambda self, other: True,
                                                                "__hash__": str.__hash__})("x"),
                              domain=Reals, assumptions=(), uid=X.uid)
        trees = [
            ("an op that is no str", node(op=1, args=(), value=None)),
            ("args as a list", node(op="sin", args=[x], value=None)),
            ("an unknown op", node(op="erf", args=(x,), value=None)),
            ("a float scalar", node(op="scalar", args=(), value=0.5)),
            ("an int scalar", node(op="scalar", args=(), value=1)),
            ("a bool scalar", node(op="scalar", args=(), value=True)),
            ("a scalar not in lowest terms", node(op="scalar", args=(), value=trusted_rational(2, 4))),
            ("a scalar with a negative denominator", node(op="scalar", args=(), value=trusted_rational(1, -2))),
            ("a scalar with an operand", node(op="scalar", args=(x,), value=Rational(1))),
            ("another symbol", f.as_real_expr(Y)),
            ("a symbol of the same name", f.as_real_expr(Symbol("x"))),
            ("a str symbol", node(op="symbol", args=(), value="x")),
            ("a lying symbol", f.exp(node(op="symbol", args=(), value=liar))),
            ("add of three", node(op="add", args=(x, x, x), value=None)),
            ("add with a value", node(op="add", args=(x, x), value=Rational(1))),
            ("sin of two", node(op="sin", args=(x, x), value=None)),
            ("pi with an operand", node(op="pi", args=(x,), value=None)),
            ("pow with a float exponent", node(op="pow", args=(x,), value=2.0)),
            ("pow with a bool exponent", node(op="pow", args=(x,), value=True)),
            ("a str instead of a node", "x"), ("None", None), ("a Rational", Rational(0)), ("the Symbol", X),
            ("an impostor node", support.impostor(f.sin(x))),
            ("sin of a forged node", f.sin(node(op="erf", args=(x,), value=None))),
        ]
        domains = [
            ("domain None", None), ("a str domain", "true"), ("the domain False", False),
            ("an unknown kind", support.forged(f.DomainPredicate, kind="real", operands=())),
            ("true with an operand", support.forged(f.DomainPredicate, kind="true", operands=(x,))),
            ("and of one", support.forged(f.DomainPredicate, kind="and", operands=(atom,))),
            ("operands as a list", support.forged(f.DomainPredicate, kind="nonzero", operands=[x + 1])),
            ("an atom of no expression", support.forged(f.DomainPredicate, kind="positive", operands=("x",))),
            ("an atom of another symbol", f.DomainPredicate("nonzero", (f.as_real_expr(Y),))),
            ("an impostor predicate", support.impostor(f.TRUE)),
        ]
        functions = [(label, forged_function(item)) for label, item in trees]
        functions += [(label, forged_function(x, domain)) for label, domain in domains]
        functions += [("variables as a list", forged_function(x, f.TRUE, [X]))]
        for label, function in functions:
            args = args_of(function, 0, 2)
            with self.subTest(case=label):
                with computation(fresh=True), self.assertRaises(InvalidInput):
                    family.produce(OP, args)
                self.assertInvalidInput(support.run(OP, **args))
        # produce() alone: a cycle never reaches it from run(), and neither does a record that lacks a field, since
        # run()'s argument inspection (mathforge.limits.inspect, shared code) reads every dataclass field first and
        # its AttributeError becomes an execution_error before any producer runs.
        for label, function in (("a node below itself", forged_function(cyclic)),
                                ("two nodes below each other", forged_function(first)),
                                ("a predicate inside itself", forged_function(x, cyclic_domain)),
                                ("a node with no field", forged_function(node())),
                                ("a predicate with no field", forged_function(x, support.forged(f.DomainPredicate))),
                                ("a function without an expression", support.forged(f.RealFunction, variables=(X,))),
                                ("a function without variables", support.forged(f.RealFunction, expression=x))):
            with self.subTest(case=label), computation(fresh=True), self.assertRaises(InvalidInput):
                family.produce(OP, args_of(function, 0, 2))
        # Well-formed records out of scope stay Unevaluated, and a domain left unset reads the class default, TRUE.
        for item in (f.acos(x), f.pi, f.E):
            with self.subTest(out_of_scope=str(item)), computation(fresh=True):
                self.assertIs(type(family.produce(OP, args_of(forged_function(item), 0, 2))[0]), Unevaluated)
        with computation(fresh=True):
            self.assertEqual(family.produce(OP, args_of(support.forged(f.RealFunction, expression=x, variables=(X,)),
                                                        0, 2))[0], series(0, (0, 1, 0)))

    def test_resource_limits(self):
        """An integer of 2**65536 in any argument and max_work=1 exceed the budget; no argument of the family takes a
        matrix, so support.oversized_matrix() has nothing to exceed here. An order above 4096 is the card's resource
        limit, an expression deeper than MAX_DEPTH the checker's, and a product of long coefficients past the budget
        is refused before it runs."""
        base = args_of(parsed("exp(x)"), 0, 3)
        for name in base:
            with self.subTest(argument=name):
                self.assertResourceLimit(support.run(OP, **dict(base, **{name: support.HUGE})))
        huge_scalar = f.RealFunction(f.as_real_expr(X) + f.as_real_expr(Rational(support.HUGE)), (X,))
        for label, args in (("a center 1/2**65536", dict(base, center=Rational(1, support.HUGE))),
                            ("a scalar of 2**65536", dict(base, function=huge_scalar)),
                            ("order 4097", dict(base, order=4097)), ("order 10**9", dict(base, order=10 ** 9))):
            with self.subTest(case=label):
                self.assertResourceLimit(support.run(OP, **args))
        with computation(fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OP, dict(base, order=4097))
        for function, center, order, _ in KNOWN[:3]:
            with self.subTest(max_work=1, function=str(function)):
                self.assertResourceLimit(support.run(OP, limits=ComputationLimits(max_work=1),
                                                     **args_of(function, center, order)))
        # A chain of 201 nodes is deeper than the checker evaluates (truncated_series recurses per level): run()
        # refuses it by max_depth already, and with a raised max_depth the producer refuses it itself.
        chain = f.as_real_expr(X)
        for _ in range(family.MAX_DEPTH):
            chain = f.sin(chain)
        deep = args_of(f.RealFunction(chain, (X,)), 0, 3)
        self.assertResourceLimit(support.run(OP, **deep))
        self.assertResourceLimit(support.run(OP, limits=ComputationLimits(max_depth=1000), **deep))
        with computation(fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OP, deep)
        # Products of series whose coefficients grow like factorials, past the budget, fail at once.
        for text, order in (("sin(x)*exp(x)", 1000), ("exp(x)**2", 900), ("exp(sin(x))", 1500)):
            with self.subTest(function=text, order=order):
                start = time.perf_counter()
                self.assertResourceLimit(support.run(OP, **args_of(parsed(text), 0, order)))
                self.assertLess(time.perf_counter() - start, FAST)

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim: here claims built by hand from closed forms of the textbooks, which
        the producer's values equal, on functions given as trees that the producer may expand otherwise; the
        evidence is () for every claim of the family."""
        n = 10
        claims = [
            (parsed("exp(2*x)"), 0, [Fraction(2 ** k, factorial(k)) for k in range(n)]),
            (parsed("exp(x)*exp(x)"), 0, [Fraction(2 ** k, factorial(k)) for k in range(n)]),
            (parsed("cos(x)"), 0, [Fraction((-1) ** (k // 2), factorial(k)) if k % 2 == 0 else F0 for k in range(n)]),
            (parsed("cosh(x)"), 0, [Fraction(1, factorial(k)) if k % 2 == 0 else F0 for k in range(n)]),
            (parsed("sinh(x)"), 0, [Fraction(1, factorial(k)) if k % 2 else F0 for k in range(n)]),
            (parsed("1/(1 - x)**2"), 0, [Fraction(k + 1) for k in range(n)]),
            (parsed("1/(1 + x**2)"), 0, [Fraction((-1) ** (k // 2)) if k % 2 == 0 else F0 for k in range(n)]),
            (parsed("log(x)"), 1, [Fraction((-1) ** (k + 1), k) if k else F0 for k in range(n)]),
            (parsed("log(1 + x) - log(1 - x)"), 0, [Fraction(2, k) if k % 2 else F0 for k in range(n)]),
            (parsed("atan(x)"), 0, [Fraction((-1) ** (k // 2), k) if k % 2 else F0 for k in range(n)]),
            (parsed("asin(x)"), 0, [Fraction(comb(k - 1, (k - 1) // 2), 4 ** ((k - 1) // 2) * k) if k % 2 else F0
                                    for k in range(n)]),
            (parsed("tan(x)"), 0, [F0, F1, F0, Fraction(1, 3), F0, Fraction(2, 15), F0, Fraction(17, 315), F0,
                                   Fraction(62, 2835)]),
            (parsed("sin(x)/cos(x)"), 0, [F0, F1, F0, Fraction(1, 3), F0, Fraction(2, 15), F0, Fraction(17, 315), F0,
                                          Fraction(62, 2835)]),
            (parsed("x**5"), 2, [Fraction(comb(5, k) * 2 ** (5 - k)) for k in range(n)]),
            (parsed("sqrt(x)"), Fraction(9, 4), [binomial(Fraction(1, 2), k) * Fraction(3, 2) ** (1 - 2 * k)
                                                for k in range(n)]),
            (parsed("sqrt(1 + 2*x)"), 0, [binomial(Fraction(1, 2), k) * 2 ** k for k in range(n)]),
        ]
        for function, center, coefficients in claims:
            args = args_of(function, center, n - 1)
            value = series(center, coefficients)
            with self.subTest(function=str(function), center=center):
                self.assertAccepts(OP, args, value, ())
                self.assertAccepts(OP, dict(args, center=rational(center)), value, ())
                with computation(fresh=True):
                    self.assertEqual(family.produce(OP, args), (value, ()))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, a record of another class (an
        impostor, one forged with object.__new__), a bool, a float or a str where a number belongs, a list for a
        tuple, any evidence but (), a wrong operation name and oversized forged evidence; mf.verify fails the
        mathematics of a Result whose value was replaced."""
        for function, center, order in ((parsed("log(x)"), 1, 4), (parsed("sin(x)*exp(x)"), 0, 12),
                                        (parsed("sqrt(x)"), Fraction(9, 4), 3)):
            args = args_of(function, center, order)
            result = support.run(OP, **args)
            value = result.value
            self.assertVerified(result, value)
            self.assertAccepts(OP, args, value, ())
            c, a, record, forged = value.coefficients, value.center, family.PowerSeries, support.forged

            def with_coefficient(k: int, item: object) -> tuple:
                return c[:k] + (item,) + c[k + 1:]

            wrong = [record(X, a, with_coefficient(k, c[k] + 1)) for k in range(order + 1)]
            cases = [(f"coefficient {k} plus 1", item, ()) for k, item in enumerate(wrong)]
            cases += [(label, item, ()) for label, item in (
                ("the coefficients reversed", record(X, a, c[::-1])),
                ("the signs flipped", record(X, a, tuple(-v for v in c))),
                ("another center", record(X, a + 1, c)),
                ("another variable", record(Y, a, c)),
                ("a variable of the same name", record(Symbol("x"), a, c)),
                ("one coefficient fewer", record(X, a, c[:-1])),
                ("one coefficient more", record(X, a, c + (Rational(0),))),
                ("a coefficient not in lowest terms", forged(record, variable=X, center=a, coefficients=with_coefficient(
                    1, trusted_rational(2 * c[1].numerator, 2 * c[1].denominator)))),
                ("an int coefficient", forged(record, variable=X, center=a, coefficients=with_coefficient(0, 7))),
                ("a float coefficient", forged(record, variable=X, center=a, coefficients=with_coefficient(1, 0.5))),
                ("a str coefficient", forged(record, variable=X, center=a, coefficients=with_coefficient(1, "1"))),
                ("a bool coefficient", forged(record, variable=X, center=a, coefficients=with_coefficient(1, True))),
                ("None as a coefficient", forged(record, variable=X, center=a, coefficients=with_coefficient(1, None))),
                ("an int center", forged(record, variable=X, center=fraction(a).numerator, coefficients=c)
                 if a.denominator == 1 else forged(record, variable=X, center=float(fraction(a)), coefficients=c)),
                ("coefficients as a list", forged(record, variable=X, center=a, coefficients=list(c))),
                ("coefficients as a tuple subclass", forged(record, variable=X, center=a,
                                                            coefficients=type("Coefficients", (tuple,), {})(c))),
                ("a record without coefficients", forged(record, variable=X, center=a)),
                ("a record with no field", forged(record)),
                ("an impostor record", support.impostor(value)),
                ("the bare coefficients", c),
                ("None as the value", None),
                ("a giant coefficient", forged(record, variable=X, center=a, coefficients=with_coefficient(
                    1, Rational(2 ** 70_000)))),
            )]
            cases += [(label, value, evidence) for label, evidence in (
                ("evidence as a list", []), ("evidence None", None), ("a zero as evidence", (0,)),
                ("the coefficients as evidence", (c,)), ("oversized evidence", (0,) * 100_000))]
            for label, item, evidence in cases:
                with self.subTest(function=str(function), case=label):
                    self.assertRejects(OP, args, item, evidence)
            for operation in ("function_taylor", "series_coefficient", "SERIES_COEFFICIENTS", "", None):
                with self.subTest(function=str(function), operation=operation):
                    self.assertRejects(operation, args, value)
            report = support.statuses(mf.verify(support.tampered(result, wrong[-1])))
            self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
            report = support.statuses(mf.verify(support.tampered(result, value, (0,))))
            self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text; an
        Unevaluated Result keeps its report as well."""
        requests = [args_of(function, center, order) for function, center, order, _ in KNOWN]
        requests += [args_of(parsed("tan(x + x**2)"), 0, 30), args_of(parsed("sqrt(x)"), 4, 12),
                     args_of(removable(), Fraction(-1, 2), 5), args_of(parsed("asin(x/2)*cosh(x)"), 0, 20)]
        results = [support.run(OP, **args) for args in requests]
        texts = [mf.dumps(result) for result in results]
        unevaluated_args = args_of(parsed("exp(x)"), 1, 4)
        unevaluated = support.run(OP, **unevaluated_args)
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for index, (result, text) in enumerate(zip(results, texts)):
                with self.subTest(request=index):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            self.assertUnevaluated(unevaluated, unevaluated_args)
            self.assertUnevaluated(mf.loads(mf.dumps(unevaluated)), unevaluated_args)

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: the center 0, a
        positive and a negative center, a fractional one, zero terms left out, coefficients of 1 and -1, and the
        remainder alone for a zero series."""
        rendered = (
            (parsed("exp(x)"), 0, 4, "1 + x + 1/2*x^2 + 1/6*x^3 + 1/24*x^4 + O(x^5)",
             r"1 + x + \frac{1}{2}\, x^{2} + \frac{1}{6}\, x^{3} + \frac{1}{24}\, x^{4} + \mathcal{O}\left(x^{5}\right)"),
            (parsed("log(x)"), 1, 3, "(x - 1) - 1/2*(x - 1)^2 + 1/3*(x - 1)^3 + O((x - 1)^4)",
             r"\left(x - 1\right) - \frac{1}{2}\, \left(x - 1\right)^{2} + \frac{1}{3}\, \left(x - 1\right)^{3}"
             r" + \mathcal{O}\left(\left(x - 1\right)^{4}\right)"),
            (parsed("1/(1 + 2*x)"), Fraction(1, 2), 2, "1/2 - 1/2*(x - 1/2) + 1/2*(x - 1/2)^2 + O((x - 1/2)^3)",
             r"\frac{1}{2} - \frac{1}{2}\, \left(x - \frac{1}{2}\right) + \frac{1}{2}\, \left(x - \frac{1}{2}\right)"
             r"^{2} + \mathcal{O}\left(\left(x - \frac{1}{2}\right)^{3}\right)"),
            (parsed("x"), Fraction(-1, 2), 1, "-1/2 + (x + 1/2) + O((x + 1/2)^2)",
             r"-\frac{1}{2} + \left(x + \frac{1}{2}\right) + \mathcal{O}\left(\left(x + \frac{1}{2}\right)^{2}\right)"),
            (parsed("x**2"), 3, 3, "9 + 6*(x - 3) + (x - 3)^2 + O((x - 3)^4)",
             r"9 + 6\, \left(x - 3\right) + \left(x - 3\right)^{2} + \mathcal{O}\left(\left(x - 3\right)^{4}\right)"),
            (parsed("cos(x)"), 0, 5, "1 - 1/2*x^2 + 1/24*x^4 + O(x^6)",
             r"1 - \frac{1}{2}\, x^{2} + \frac{1}{24}\, x^{4} + \mathcal{O}\left(x^{6}\right)"),
            (parsed("-sin(x)"), 0, 2, "-x + O(x^3)", r"-x + \mathcal{O}\left(x^{3}\right)"),
            (parsed("sin(x)"), 0, 0, "O(x)", r"\mathcal{O}\left(x\right)"),
            (f.real_function(Rational(0), (X,)), 2, 2, "O((x - 2)^3)",
             r"\mathcal{O}\left(\left(x - 2\right)^{3}\right)"),
        )
        for function, center, order, text, latex in rendered:
            with self.subTest(function=str(function), center=center, order=order):
                result = support.run(OP, **args_of(function, center, order))
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text)
        self.assertEqual(mf.to_data(support.run(OP, **args_of(parsed("exp(x)"), 0, 2)).value)["data"]["type"],
                         "v6_power_series")
        unevaluated = support.run(OP, **args_of(parsed("exp(x)"), 1, 2))
        self.assertEqual(mf.render_text(unevaluated),
                         "completed / no_conclusion [not_applicable; unknown]: Unevaluated(series, series_coefficients, "
                         "Reason(no_applicable_rule, exp is expanded only about an argument that vanishes at the "
                         "center))")

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261112)
        for index in range(24):
            center = rng.choice(CENTERS)
            tree = random_tree(rng, center, rng.randint(1, 3))
            args = args_of(real(tree), center, rng.randint(0, 10))
            with self.subTest(case=index, tree=tree, center=center):
                expected = support.run(OP, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(OP, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(OP, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        request = {"function": {"text": "sin(x)*exp(x)", "variables": ["x"]}, "variable": "x", "center": 0,
                   "order": 12}
        result = self.answer(support.call(OP, request, symbol_ids={"x": X.uid}))
        function = mf.parse_function("sin(x)*exp(x)", symbols={"x": X}, variables=(X,))
        self.assertEqual(result, support.run(OP, **args_of(function, 0, 12)))
        self.assertEqual(result.value, series(0, sin_exp(13)))
        self.assertTrue(mf.verify(result).verified)
        for function, center, order in ((parsed("log(x)"), 1, 3), (parsed("sqrt(x)"), Fraction(9, 4), 4),
                                        (removable(), Fraction(-7, 3), 2)):
            with self.subTest(function=str(function), center=center):
                result = self.answer(support.call_values(OP, **args_of(function, center, order)))
                self.assertEqual(result, support.run(OP, **args_of(function, center, order)))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call(OP, dict(request, order=-1)), "invalid_input")
        self.assertIn("order must be at least 0", error["message"])
        deferred = self.answer(support.call(OP, dict(request, center=1)))
        self.assertIs(deferred.outcome, Outcome.NO_CONCLUSION)
        self.assertIsNone(deferred.value)

    # -----------------------------------------------------------------------------------------------------------
    # Agreement with the reference

    def test_reference_agrees_with_identities(self):
        """The reference builds none of these in: sin^2 + cos^2 = 1, cosh^2 - sinh^2 = 1, tan = sin/cos and tanh =
        sinh/cosh (so the Bernoulli numbers are right), exp(log(1 + t)) = 1 + t, tan(atan t) = t, sin(asin t) = t,
        sqrt(4 + t)^2 = 4 + t, u u^-1 = 1, and the first terms that the textbooks list."""
        n = 14
        t = (F0, F1) + (F0,) * (n - 2)
        one = constant(1, n)
        s = {name: horner(maclaurin(name, n), t) for name in VANISHING}

        def plus(a, b):
            return tuple(p + q for p, q in zip(a, b))

        self.assertEqual(plus(times(s["sin"], s["sin"]), times(s["cos"], s["cos"])), one)
        self.assertEqual(plus(times(s["cosh"], s["cosh"]), tuple(-v for v in times(s["sinh"], s["sinh"]))), one)
        self.assertEqual(s["tan"], times(s["sin"], power(s["cos"], -1)))
        self.assertEqual(s["tanh"], times(s["sinh"], power(s["cosh"], -1)))
        self.assertEqual(horner(maclaurin("exp", n), horner(maclaurin("log", n), t)), (F1, F1) + (F0,) * (n - 2))
        self.assertEqual(horner(maclaurin("tan", n), s["atan"]), t)
        self.assertEqual(horner(maclaurin("sin", n), s["asin"]), t)
        self.assertEqual(power(reference(("sqrt", ("add", ("c", 4), T)), 0, n), 2), plus(constant(4, n), t))
        u = (Fraction(-2, 3), F1, Fraction(5, 7), F0, Fraction(-1, 2)) + (F1,) * (n - 5)
        self.assertEqual(times(u, power(u, -1)), one)
        self.assertEqual(s["tan"][:10], (0, 1, 0, Fraction(1, 3), 0, Fraction(2, 15), 0, Fraction(17, 315), 0,
                                         Fraction(62, 2835)))
        self.assertEqual(s["asin"][:8], (0, 1, 0, Fraction(1, 6), 0, Fraction(3, 40), 0, Fraction(5, 112)))
        self.assertEqual(bernoulli(8)[::2], [1, Fraction(1, 6), Fraction(-1, 30), Fraction(1, 42), Fraction(-1, 30)])
        self.assertEqual(reference(("sqrt", T), 4, 6), (2, Fraction(1, 4), Fraction(-1, 64), Fraction(1, 512),
                                                        Fraction(-5, 16384), Fraction(7, 131072)))
        with self.assertRaises(AssertionError):
            reference(("exp", T), 1, 3)  # exp(1) is no rational: the reference refuses what the card leaves out.

    def test_seeded_agreement_with_the_reference(self):
        """At least 200 seeded inputs, trees of up to three levels of functions at eight centers and orders up to 12:
        the producer succeeds, the checker accepts its claim, and the claim equals the reference's. Both of the
        producer's methods run (monomial tables and Newton iteration), and each seeded input moved out of scope
        (an argument shifted off its expansion point) is Unevaluated."""
        rng = random.Random(20261113)
        newton, tables = patch.object(kernel, "_newton_steps", wraps=kernel._newton_steps), \
            patch.object(kernel, "compose", wraps=kernel.compose)
        with newton as newton_steps, tables as compose:
            for index in range(220):
                center = rng.choice(CENTERS)
                tree = random_tree(rng, center, rng.randint(1, 3))
                order = rng.randint(0, 12)
                args = args_of(real(tree), center, order)
                with self.subTest(case=index, tree=tree, center=center, order=order):
                    with computation(fresh=True):
                        value, evidence = family.produce(OP, args)
                    self.assertEqual((value, evidence), (series(center, reference(tree, center, order + 1)), ()))
                    self.assertIs(support.verdict(family, OP, args, value, evidence), True)
        self.assertGreater(newton_steps.call_count, 50)  # 96 Newton iterations with this seed.
        self.assertGreater(compose.call_count, 50)  # 126 monomial tables with this seed.
        moved = 0
        for index in range(60):
            center = rng.choice(CENTERS)
            tree = random_tree(rng, center, rng.randint(0, 2))
            name, target = rng.choice([(name, 1) for name in VANISHING] + [("log", 2), ("log", Fraction(1, 2)),
                                                                           ("sqrt", 2), ("sqrt", Fraction(3, 4))])
            args = args_of(real((name, shifted(tree, center, target))), center, rng.randint(0, 6))
            with self.subTest(moved=index, name=name, tree=tree, center=center):
                result = support.run(OP, **args)
                self.assertUnevaluated(result, args)
                moved += 1
        self.assertEqual(moved, 60)

    def test_large_orders_fit_the_default_budget(self):
        """exp(x) and cos(x) at order 4096, the card's largest, are monomial tables and verify under the default
        budget; so do tan(x) and tanh(x) at order 200 (Brent and Harvey's tangent numbers), the geometric series at
        order 1000 and a product of two compositions at order 80."""
        n = 4097
        cases = [
            (parsed("exp(x)"), 0, n - 1, [Fraction(1, factorial(k)) for k in range(n)]),
            (parsed("cos(x)"), 0, n - 1, [Fraction((-1) ** (k // 2), factorial(k)) if k % 2 == 0 else F0
                                          for k in range(n)]),
            (parsed("exp(x - 1)"), 1, 1500, [Fraction(1, factorial(k)) for k in range(1501)]),
            (parsed("1/(1 - x)"), 0, 1000, [F1] * 1001),
            # The argument is x itself, so the Maclaurin tables are the coefficients (Horner's rule at this length
            # would cost seconds).
            (parsed("tan(x)"), 0, 200, maclaurin("tan", 201)),
            (parsed("tanh(x)"), 0, 200, maclaurin("tanh", 201)),
        ]
        for function, center, order, coefficients in cases:
            with self.subTest(function=str(function), order=order):
                start = time.perf_counter()
                result = support.run(OP, **args_of(function, center, order))
                self.assertEqual(result.value, series(center, coefficients), result.error)
                self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
                self.assertLess(time.perf_counter() - start, 3.0)
        result = support.run(OP, **args_of(parsed("sin(x + x**2)*exp(x)"), 0, 80))
        self.assertEqual(result.value, series(0, reference(("mul", ("sin", ("add", T, ("mul", T, T))), ("exp", T)),
                                                           0, 81)))
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)

    def test_values_the_check_cannot_afford_are_refused(self):
        """Regression: the check charges each series operation by the length of its coefficients, and the producer
        refuses at once a value whose check would exceed the budget, by a lower bound of the check's work. sqrt(1 + x)
        at order 2000 and tanh(x) at order 1024 used to be produced in 0.01 to 0.06 s and then checked for 15 to 20 s
        before resource_limit_exceeded; a power (x + x^2)^(2^65535) at order 4000 is the zero series for the kernel,
        and costs the checker 65,536 truncated products. The bound never exceeds the check's work: wherever the check
        fits a budget that the producer's own work fits as well, the producer answers, and the checker accepts the
        value."""
        huge_power = f.RealFunction(f.RealExpr("pow", (f.as_real_expr(X) + f.as_real_expr(X) ** 2,), 2 ** 65535),
                                    (X,))
        for label, function, order in (("sqrt(1 + x)", parsed("sqrt(1 + x)"), 2000),
                                       ("tanh(x)", parsed("tanh(x)"), 1024), ("(x + x^2)^(2^65535)", huge_power, 4000)):
            with self.subTest(function=label, order=order):
                start = time.perf_counter()
                self.assertResourceLimit(support.run(OP, **args_of(function, 0, order)))
                with computation(fresh=True), self.assertRaises(ResourceLimitError):
                    family.produce(OP, args_of(function, 0, order))
                self.assertLess(time.perf_counter() - start, FAST)
        big = ComputationLimits(max_work=10 ** 9)
        for text, order in (("sqrt(1 + x)", 600), ("tan(x)", 300), ("exp(x)**2", 400), ("1/(1 - x/3)", 800),
                            ("atan(x)", 400), ("(1 + x)**(-3)", 300), ("asin(x/3)", 200)):
            with self.subTest(function=text, order=order):
                args = args_of(parsed(text), 0, order)
                with computation(big, fresh=True) as produced:
                    value, evidence = family.produce(OP, args)
                with computation(big, fresh=True) as checked:
                    self.assertIs(support.checker(family).check_operation(OP, args, value, evidence), True)
                limits = ComputationLimits(max_work=max(produced.work, checked.work))
                with computation(limits, fresh=True):
                    self.assertEqual(family.produce(OP, args), (value, evidence))
                self.assertIs(support.verdict(family, OP, args, value, evidence, limits=limits), True)

    def test_shared_subtrees_and_domains(self):
        """A node shared by several parents is expanded once: (1 + x)^(2^60) as 60 nested squares of one node is
        expanded at once by produce(), and the checker accepts the claim. run() refuses that tree by its node count
        (the argument inspection of mathforge.limits counts a shared node once per parent, 2^61 of them), so produce()
        and the checker are called directly. A domain that holds near the center lets the value through."""
        x = f.as_real_expr(X)
        shared = x + 1
        for _ in range(60):
            shared = f.RealExpr("mul", (shared, shared))
        start = time.perf_counter()
        for center, coefficients in ((0, [binomial(2 ** 60, k) for k in range(6)]), (-1, [0] * 6)):
            args = args_of(forged_function(shared), center, 5)
            with computation(fresh=True):
                value, evidence = family.produce(OP, args)
            self.assertEqual((value, evidence), (series(center, coefficients), ()))
            self.assertIs(support.verdict(family, OP, args, value, evidence), True)
            self.assertResourceLimit(support.run(OP, **args))
        # At 1 the constant term is 2^(2^60), far longer than max_integer_bits: the value exceeds the budget.
        with computation(fresh=True), self.assertRaises(ResourceLimitError):
            family.produce(OP, args_of(forged_function(shared), 1, 5))
        self.assertLess(time.perf_counter() - start, FAST)
        for function, center, coefficients in (
                (f.real_function(2 * x, (X,), domain=f.condition("positive", x)), 1, (2, 2, 0)),
                (f.real_function(x, (X,), domain=f.DomainPredicate("nonnegative", (x,))), 3, (3, 1, 0)),
                (removable(), 0, (1, 1, 0)),
                (removable(), 2, (3, 1, 0)),
                (parsed("(x**2 - 1)/(x - 1)"), 0, (1, 1, 0)),
                (forged_function(x, f.DomainPredicate("and", (f.DomainPredicate("and", (
                    f.DomainPredicate("positive", (x + 1,)), f.DomainPredicate("nonzero", (x + 2,)))), f.TRUE))),
                 0, (0, 1, 0))):
            with self.subTest(function=str(function), center=center):
                self.assertVerified(support.run(OP, **args_of(function, center, 2)), series(center, coefficients))

    # -----------------------------------------------------------------------------------------------------------
    # The kernel

    def test_kernel_operations(self):
        """The kernel's series operations against the reference on seeded series: products by the schoolbook loop and
        by Kronecker substitution (negative and long coefficients), the Newton iterations and the monomial tables,
        integer powers, tangent numbers and the charged rational arithmetic."""
        rng = random.Random(7)

        def random_series(n: int, head: object, terms: int = 0) -> tuple[Rational, ...]:
            out = [rational(head)] + [Rational(0)] * (n - 1)
            for k in rng.sample(range(1, n), min(terms or n - 1, n - 1)):
                out[k] = Rational(rng.randint(-10 ** rng.randint(1, 30), 10 ** 30), rng.randint(1, 10 ** 8))
            return tuple(out)

        def fractions(s: tuple) -> tuple[Fraction, ...]:
            return tuple(fraction(c) for c in s)

        with computation(ComputationLimits(max_work=10 ** 9), fresh=True):
            for n in (1, 2, 3, 9, 24):
                for terms in (1, 3, 0):
                    a, b = random_series(n, rng.randint(-3, 3), terms), random_series(n, rng.randint(-3, 3))
                    with self.subTest(n=n, terms=terms):
                        self.assertEqual(fractions(kernel.mul(a, b)), times(fractions(a), fractions(b)))
                        self.assertEqual(fractions(kernel.mul(b, a)), times(fractions(a), fractions(b)))
                        u = random_series(n, Fraction(rng.choice((-3, 1, 2)), rng.choice((1, 5))), terms)
                        self.assertEqual(fractions(kernel.reciprocal(u)), power(fractions(u), -1))
                        for k in (-3, 2, 5):
                            self.assertEqual(fractions(kernel.power(u, k)), power(fractions(u), k))
                        v = random_series(n, 0, terms)
                        fv = fractions(v)
                        self.assertEqual(fractions(kernel.exp(v)), horner(maclaurin("exp", n), fv))
                        self.assertEqual(fractions(kernel.atan(v)), horner(maclaurin("atan", n), fv))
                        self.assertEqual(fractions(kernel.asin(v)), horner(maclaurin("asin", n), fv))
                        self.assertEqual(fractions(kernel.tangent(v, 1)), horner(maclaurin("tan", n), fv))
                        self.assertEqual(fractions(kernel.tangent(v, -1)), horner(maclaurin("tanh", n), fv))
                        for hyperbolic, names in ((False, ("sin", "cos")), (True, ("sinh", "cosh"))):
                            pair = kernel.sin_cos(v, hyperbolic)
                            self.assertEqual(tuple(fractions(item) for item in pair),
                                             tuple(horner(maclaurin(name, n), fv) for name in names))
                        self.assertEqual(fractions(kernel.log(kernel.shift_constant(v, kernel.ONE))),
                                         horner(maclaurin("log", n), fv))
                        root = Fraction(rng.choice((1, 2, 3)), rng.choice((1, 2, 7)))
                        w = kernel.shift_constant(v, rational(root * root))
                        self.assertEqual(fractions(kernel.sqrt(w, rational(root))),
                                         horner([binomial(Fraction(1, 2), k) * root ** (1 - 2 * k) for k in range(n)],
                                                fv))
            self.assertEqual(kernel.tangent_numbers(8), [1, 2, 16, 272, 7936, 353792, 22368256, 1903757312])
            for _ in range(200):
                x = Fraction(rng.randint(-10 ** 40, 10 ** 40), rng.randint(1, 10 ** rng.randint(1, 40)))
                y = Fraction(rng.randint(-10 ** 40, 10 ** 40), rng.randint(1, 10 ** rng.randint(1, 40))) or F1
                qx, qy = rational(x), rational(y)
                self.assertEqual(fraction(kernel.plus(qx, qy)), x + y)
                self.assertEqual(fraction(kernel.minus(qx, qy)), x - y)
                self.assertEqual(fraction(kernel.times(qx, qy)), x * y)
                self.assertEqual(fraction(kernel.quotient(qx, qy)), x / y)
                for value in (kernel.plus(qx, qy), kernel.times(qx, qy), kernel.quotient(qx, qy)):
                    self.assertEqual((value.numerator, value.denominator),
                                     (fraction(value).numerator, fraction(value).denominator))  # Lowest terms.
            self.assertEqual(kernel.rational_power(Rational(-2, 3), -3), Rational(-27, 8))
            for values in ([0], [5, -5, 0, 1], [-(2 ** 200), 2 ** 200 - 1, -1, 0], [-1] * 7):
                width = (max(abs(v) for v in values).bit_length() + 2 + 7) // 8
                self.assertEqual(kernel._unpack(kernel._pack(values, width), width, len(values)), values)
        # The work estimates grow with the operands, and charge 65,536-bit products near their measured 0.84 ms.
        self.assertLess(kernel.product_work(64, 64), kernel.product_work(4096, 4096))
        self.assertLess(kernel.product_work(4096, 2 ** 20), kernel.product_work(2 ** 20, 2 ** 20))
        self.assertTrue(3_500 < kernel.product_work(65_536, 65_536) < 5_000)
        self.assertGreater(kernel.gcd_work(65_536, 65_536), kernel.gcd_work(65_536, 64))

    def test_kernel_is_independent_of_the_checker(self):
        """The producer and its kernel import nothing of mathforge.certify and not the shared series arithmetic of
        mathforge.trusted.series that the checker's evaluator uses; the kernel imports only the limits and the
        model. Every kernel function charges its budget: under max_work=1000 a long product fails at once."""
        imported = set()
        for item in ast.walk(ast.parse(inspect.getsource(kernel))):
            if isinstance(item, ast.Import):
                imported.update(alias.name for alias in item.names)
            elif isinstance(item, ast.ImportFrom):
                imported.add("." * item.level + (item.module or ""))
        self.assertEqual(imported, {"__future__", "math", "..limits", "..model"})
        family_imports = set()  # Every module the family imports, at the top and inside functions.
        for item in ast.walk(ast.parse(inspect.getsource(family))):
            if isinstance(item, ast.Import):
                family_imports.update(alias.name for alias in item.names)
            elif isinstance(item, ast.ImportFrom):
                family_imports.add("." * item.level + (item.module or ""))
        self.assertIn("...kernels", family_imports)
        for name in family_imports:
            with self.subTest(module=name):
                self.assertNotIn("certify", name)
                self.assertNotIn("trusted", name)
        long = tuple(Rational(1, factorial(k)) for k in range(400))
        with computation(ComputationLimits(max_work=1000), fresh=True), self.assertRaises(ResourceLimitError):
            kernel.mul(long, long)
