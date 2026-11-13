"""The independent checker of mathforge.areas.integrals.trig_powers (mathforge.certify.integrals.trig_powers), tested on
its own before the family's producer exists.

Every true claim is computed here, by a route that shares nothing with the producer's reduction formulas or with the
checker's formal derivative: f is expanded in z = exp(i a x), where S = (z - 1/z)/(2i) and C = (z + 1/z)/2, into a
Laurent polynomial over the Gaussian rationals and integrated term by term (z**m becomes z**m/(i m a), the constant
term lambda*x). The antiderivative goes back to powers of S and C through the Chebyshev polynomials, cos(m t) = T_m(C)
and sin(m t) = S U_(m-1)(C), without its constant term; the definite integrals come straight from the Laurent
expansion and the 24th roots of unity, computed here as powers of exp(i pi/12) in Q(sqrt 2, sqrt 3). The reference
reproduces the card's known values, and its two halves agree with each other. The values and the evidence follow the
layouts of the family's docstring; the canonical PiLinear is mathforge.model's normal form, pinned by hand on the known
values. The checker answers False to every corrupted claim, never with an exception, and refuses an oversized forgery
within FAST seconds under the default budget.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
from math import isqrt
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, functions, operations_v6, registry
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.integrals import trig_powers as family
from mathforge.certify.integrals import trig_powers as checker
from mathforge.contracts import CheckStatus
from mathforge.errors import InvalidInput
from mathforge.functions import DomainPredicate, RealExpr, RealFunction, parse_function, real_function
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Add, Assumption, Integers, Mul, Pow, Rational, Reals, Sqrt, Symbol, Truth, add, mul, sqrt
from mathforge.polynomial import Polynomial
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
SCOPE = "exact rationals and real functions; certified closed forms of sums, series and trigonometric integrals"
ANTIDERIVATIVE, INTEGRAL = "trig_power_antiderivative", "trig_power_integral"
X, Y = Symbol("x"), Symbol("y")
N = Symbol("n", Integers, (Assumption("positive", Truth.TRUE),))  # A variable with a domain and an assumption.
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
LONG = 2 ** 60_000 + 1  # Within max_integer_bits, and far longer than the coefficients of the requests below.

Number = Fraction | int
Poly = dict[tuple[int, int], Number]  # {(j, k): c}: the request's polynomial, sum of c S**j C**k.
NormalForm = dict[tuple[int, int, int], Fraction]  # {(i, j, k): c}: sum of c x**i S**j C**k with j <= 1.
Element = tuple[Fraction, Fraction, Fraction, Fraction]  # c0 + c1 sqrt 2 + c2 sqrt 3 + c3 sqrt 6.
Gauss = tuple[Fraction, Fraction]  # A Gaussian rational re + i im.
Claim = tuple[str, dict[str, object], object, tuple]  # (operation, args, value, evidence)


class Ratio(Rational):
    """A Rational subclass: no exact Rational."""
    __slots__ = ()


class Integer(int):
    """An int subclass, equal to an exact int but none."""


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass for an evidence tuple.


# ---------------------------------------------------------------------------------------------------------------
# The reference: every true claim of the tests comes from it.

ZERO_G: Gauss = (Fraction(0), Fraction(0))


def g_add(p: Gauss, q: Gauss) -> Gauss:
    return p[0] + q[0], p[1] + q[1]


def g_sub(p: Gauss, q: Gauss) -> Gauss:
    return p[0] - q[0], p[1] - q[1]


def g_mul(p: Gauss, q: Gauss) -> Gauss:
    return p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]


def l_add(p: dict[int, Gauss], q: dict[int, Gauss]) -> dict[int, Gauss]:
    out = dict(p)
    for m, c in q.items():
        out[m] = g_add(out.get(m, ZERO_G), c)
    return {m: c for m, c in out.items() if c != ZERO_G}


def l_mul(p: dict[int, Gauss], q: dict[int, Gauss]) -> dict[int, Gauss]:
    out: dict[int, Gauss] = {}
    for m, a in p.items():
        for n, b in q.items():
            out[m + n] = g_add(out.get(m + n, ZERO_G), g_mul(a, b))
    return {m: c for m, c in out.items() if c != ZERO_G}


Z_SINE = {1: (Fraction(0), Fraction(-1, 2)), -1: (Fraction(0), Fraction(1, 2))}  # sin = (z - 1/z)/(2i)
Z_COSINE = {1: (Fraction(1, 2), Fraction(0)), -1: (Fraction(1, 2), Fraction(0))}  # cos = (z + 1/z)/2


def laurent(poly: Poly) -> dict[int, Gauss]:
    """sum c S**j C**k as a Laurent polynomial in z = exp(i a x)."""
    total: dict[int, Gauss] = {}
    for (j, k), c in poly.items():
        term = {0: (Fraction(c), Fraction(0))}
        for _ in range(j):
            term = l_mul(term, Z_SINE)
        for _ in range(k):
            term = l_mul(term, Z_COSINE)
        total = l_add(total, term)
    return total


def chebyshev(n: int) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """T_0, ..., T_n and U_0, ..., U_n as coefficient lists in C, lowest degree first: cos(m t) = T_m(cos t) and
    sin(m t) = sin(t) U_(m-1)(cos t), both by the recurrence P_(m+1) = 2 C P_m - P_(m-1)."""
    t = [[Fraction(1)], [Fraction(0), Fraction(1)]]
    u = [[Fraction(1)], [Fraction(0), Fraction(2)]]
    while len(t) <= n:
        for sequence in (t, u):
            following = [Fraction(0)] + [2 * c for c in sequence[-1]]
            for i, c in enumerate(sequence[-2]):
                following[i] -= c
            sequence.append(following)
    return t, u


def antiderivative(poly: Poly, a: Number) -> NormalForm:
    """The canonical antiderivative of poly(sin(a x), cos(a x)) as a normal form: c_0 integrates to c_0 x and c_m z**m
    (m != 0) to c_m z**m/(i m a); the pair c_m z**m + c_-m z**-m becomes a combination of cos(m a x) = T_m(C) and
    sin(m a x) = S U_(m-1)(C), and the constant term of the sum is dropped."""
    a = Fraction(a)
    expansion = laurent(poly)
    out: dict[tuple[int, int, int], Fraction] = {}
    constant = expansion.get(0, ZERO_G)
    assert constant[1] == 0
    if constant[0]:
        out[(1, 0, 0)] = constant[0]
    top = max((abs(m) for m in expansion), default=0)
    t, u = chebyshev(top)
    for m in range(1, top + 1):
        plus, minus = expansion.get(m, ZERO_G), expansion.get(-m, ZERO_G)
        # (plus z**m - minus z**-m)/(i m a) = (plus - minus)/i cos(m a x)/(m a) + (plus + minus) sin(m a x)/(m a).
        cosine_part = g_mul((Fraction(0), Fraction(-1)), g_sub(plus, minus))
        sine_part = g_add(plus, minus)
        assert cosine_part[1] == 0 and sine_part[1] == 0  # A real function has real parts.
        for k, c in enumerate(t[m]):
            if c:
                out[(0, 0, k)] = out.get((0, 0, k), Fraction(0)) + cosine_part[0] * c / (m * a)
        for k, c in enumerate(u[m - 1]):
            if c:
                out[(0, 1, k)] = out.get((0, 1, k), Fraction(0)) + sine_part[0] * c / (m * a)
    out.pop((0, 0, 0), None)
    return {key: c for key, c in out.items() if c}


E0: Element = (Fraction(0),) * 4
E1: Element = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))


def e_add(p: Element, q: Element) -> Element:
    return tuple(x + y for x, y in zip(p, q))


def e_sub(p: Element, q: Element) -> Element:
    return tuple(x - y for x, y in zip(p, q))


def e_scale(p: Element, c: Number) -> Element:
    return tuple(x * c for x in p)


def e_mul(p: Element, q: Element) -> Element:
    """Products in Q(sqrt 2, sqrt 3): sqrt 2 sqrt 3 = sqrt 6, sqrt 2 sqrt 6 = 2 sqrt 3, sqrt 3 sqrt 6 = 3 sqrt 2."""
    a0, a1, a2, a3 = p
    b0, b1, b2, b3 = q
    return (a0 * b0 + 2 * a1 * b1 + 3 * a2 * b2 + 6 * a3 * b3,
            a0 * b1 + a1 * b0 + 3 * a2 * b3 + 3 * a3 * b2,
            a0 * b2 + a2 * b0 + 2 * a1 * b3 + 2 * a3 * b1,
            a0 * b3 + a3 * b0 + a1 * b2 + a2 * b1)


def e_power(p: Element, k: int) -> Element:
    result = E1
    for _ in range(k):
        result = e_mul(result, p)
    return result


OMEGA = ((Fraction(0), Fraction(1, 4), Fraction(0), Fraction(1, 4)),   # cos(pi/12) = (sqrt 6 + sqrt 2)/4
         (Fraction(0), Fraction(-1, 4), Fraction(0), Fraction(1, 4)))  # sin(pi/12) = (sqrt 6 - sqrt 2)/4


def c_mul(p: tuple[Element, Element], q: tuple[Element, Element]) -> tuple[Element, Element]:
    """(cos, sin) pairs multiply as complex numbers: the angles add."""
    return e_sub(e_mul(p[0], q[0]), e_mul(p[1], q[1])), e_add(e_mul(p[0], q[1]), e_mul(p[1], q[0]))


def powers_of_omega() -> tuple[tuple[Element, Element], ...]:
    roots = [(E1, E0)]
    while len(roots) < 24:
        roots.append(c_mul(roots[-1], OMEGA))
    return tuple(roots)


ROOTS = powers_of_omega()


def zeta(n: int) -> tuple[Element, Element]:
    """exp(i n pi/12) = (cos(n pi/12), sin(n pi/12)), a power of OMEGA."""
    return ROOTS[n % 24]


def fraction(number: Number | Rational) -> Fraction:
    """A number of the tests, an int, a Fraction or a mathforge Rational, as a Fraction."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def angle(a: Number, t: Number | Rational) -> int:
    """The integer k = 12 a t of the angle a t pi = k pi/12."""
    k = 12 * fraction(a) * fraction(t)
    assert k.denominator == 1
    return k.numerator


def integral(poly: Poly, a: Number, lower: Number | Rational, upper: Number | Rational) -> tuple[Element, Fraction]:
    """(algebraic part, pi coefficient) of the integral of poly(sin(a x), cos(a x)) from lower pi to upper pi, from
    the Laurent expansion alone: z**m integrates to z**m/(i m a), and z = exp(i k pi/12) at a bound, k = 12 a t."""
    a, lower, upper = fraction(a), fraction(lower), fraction(upper)
    expansion = laurent(poly)
    pi_coefficient = expansion.get(0, ZERO_G)[0] * (upper - lower)
    real, imaginary = E0, E0
    for m, c in expansion.items():
        if not m:
            continue
        g = (c[1] / (m * a), -c[0] / (m * a))  # c/(i m a)
        high_cos, high_sin = zeta(m * angle(a, upper))
        low_cos, low_sin = zeta(m * angle(a, lower))
        dr, di = e_sub(high_cos, low_cos), e_sub(high_sin, low_sin)
        real = e_add(real, e_sub(e_scale(dr, g[0]), e_scale(di, g[1])))
        imaginary = e_add(imaginary, e_add(e_scale(di, g[0]), e_scale(dr, g[1])))
    assert imaginary == E0, imaginary  # The integral of a real function is real.
    return real, pi_coefficient


def evaluate(nf: NormalForm, a: Number, t: Number) -> tuple[Element, Fraction]:
    """(algebraic part, pi coefficient) of a normal form at x = t pi, 12 a t an integer."""
    cosine, sine = zeta(angle(a, t)) if nf.keys() - {(1, 0, 0)} else (E0, E0)
    algebraic = E0
    for (i, j, k), c in nf.items():
        if not i:
            algebraic = e_add(algebraic, e_scale(e_mul(sine if j else E1, e_power(cosine, k)), c))
    return algebraic, nf.get((1, 0, 0), Fraction(0)) * Fraction(t)


ENCLOSURES = {r: (Fraction(isqrt(r * 10 ** 24), 10 ** 12), Fraction(isqrt(r * 10 ** 24) + 1, 10 ** 12))
              for r in (2, 3, 6)}


def sign(element: Element) -> int:
    """The sign of c0 + c1 sqrt 2 + c2 sqrt 3 + c3 sqrt 6, from rational enclosures of the square roots."""
    low = high = element[0]
    for c, r in zip(element[1:], (2, 3, 6)):
        lo, hi = ENCLOSURES[r]
        low, high = (low + c * lo, high + c * hi) if c >= 0 else (low + c * hi, high + c * lo)
    if low > 0:
        return 1
    if high < 0:
        return -1
    assert element == E0, element
    return 0


# ---------------------------------------------------------------------------------------------------------------
# Trees of mathforge.functions, built here.

def q(number: Number | Rational) -> Rational:
    if type(number) is Rational:
        return number
    number = Fraction(number)
    return Rational(number.numerator, number.denominator)


def scalar(number: Number | Rational) -> RealExpr:
    return RealExpr("scalar", (), q(number))


def leaf(variable: Symbol = X) -> RealExpr:
    return RealExpr("symbol", (), variable)


def argument(a: Number, variable: Symbol = X) -> RealExpr:
    """u = a*x: the variable itself for a = 1, else a scalar times it."""
    return leaf(variable) if Fraction(a) == 1 else RealExpr("mul", (scalar(a), leaf(variable)))


def sin_node(u: RealExpr) -> RealExpr:
    return RealExpr("sin", (u,))


def cos_node(u: RealExpr) -> RealExpr:
    return RealExpr("cos", (u,))


def power(base: RealExpr, k: int) -> RealExpr:
    return base if k == 1 else RealExpr("pow", (base,), k)


def product(nodes: Sequence[RealExpr]) -> RealExpr:
    result = nodes[0]
    for item in nodes[1:]:
        result = RealExpr("mul", (result, item))
    return result


def balanced(terms: Sequence[RealExpr]) -> RealExpr:
    """The sum of the terms as a balanced tree, shallow enough for any number of terms; 0 for none."""
    terms = list(terms)
    while len(terms) > 1:
        terms = [RealExpr("add", (terms[i], terms[i + 1])) if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0] if terms else scalar(0)


def poly_tree(poly: Poly, u: RealExpr) -> RealExpr:
    """sum c sin(u)**j cos(u)**k, a term per entry."""
    terms = []
    for (j, k), c in sorted(poly.items()):
        factors = ([power(sin_node(u), j)] if j else []) + ([power(cos_node(u), k)] if k else [])
        if not factors:
            terms.append(scalar(c))
        else:
            terms.append(product(factors) if c == 1 else RealExpr("mul", (scalar(c), product(factors))))
    return balanced(terms)


def nf_tree(nf: NormalForm, u: RealExpr, style: str = "mul", variable: Symbol = X) -> RealExpr:
    """A normal form as a sum of monomials: c times x, sin(u) and a power of cos(u). Styles: "mul" (scalar times the
    monomial), "div" (the numerator times the monomial, divided by the denominator) and "neg" (a negative c as -1 times
    |c| times the monomial, as RealExpr's unary minus writes it)."""
    terms = []
    for (i, j, k), c in sorted(nf.items()):
        factors = [leaf(variable)] * i + [sin_node(u)] * j + ([power(cos_node(u), k)] if k else [])
        monomial = product(factors)
        if style == "div":
            top = RealExpr("mul", (scalar(c.numerator), monomial))
            terms.append(top if c.denominator == 1 else RealExpr("div", (top, scalar(c.denominator))))
        elif style == "neg" and c < 0:
            terms.append(RealExpr("mul", (scalar(-1), RealExpr("mul", (scalar(-c), monomial)))))
        else:
            terms.append(RealExpr("mul", (scalar(c), monomial)))
    return balanced(terms)


def horner_tree(nf: NormalForm, u: RealExpr, variable: Symbol = X) -> RealExpr:
    """lambda*x + N0(C) + S*N1(C), each N by Horner's rule in C (deep: for small degrees only)."""
    parts = []
    if (1, 0, 0) in nf:
        parts.append(RealExpr("mul", (scalar(nf[(1, 0, 0)]), leaf(variable))))
    for j in (0, 1):
        top = max((k for (i, jj, k) in nf if not i and jj == j), default=None)
        if top is None:
            continue
        accumulator = scalar(nf.get((0, j, top), 0))
        for k in range(top - 1, -1, -1):
            accumulator = RealExpr("add", (RealExpr("mul", (accumulator, cos_node(u))), scalar(nf.get((0, j, k), 0))))
        parts.append(RealExpr("mul", (sin_node(u), accumulator)) if j else accumulator)
    return balanced(parts)


def function(expression: RealExpr, variable: Symbol = X) -> RealFunction:
    return RealFunction(expression, (variable,))


def text(source: str) -> RealFunction:
    return parse_function(source, symbols={"x": X})


def node(op: object, args: object = (), value: object = None) -> RealExpr:
    """A RealExpr made with object.__new__: its fields may break every rule of the class."""
    return support.forged(RealExpr, op=op, args=args, value=value)


def corrupted(tree: RealExpr, path: Sequence[int], replacement: object) -> RealExpr:
    """The tree with the node at the path (operand indices from the root) replaced, the nodes above it forged."""
    if not path:
        return replacement
    args = list(tree.args)
    args[path[0]] = corrupted(args[path[0]], path[1:], replacement)
    return node(tree.op, tuple(args), tree.value)


def forged_function(expression: object, variables: object = (X,), domain: object = functions.TRUE) -> RealFunction:
    return support.forged(RealFunction, expression=expression, variables=variables, domain=domain)


def sin2_tree() -> RealExpr:
    """x/2 - sin(x)*cos(x)/2 as add(div(x, 2), mul(-1/2, mul(sin(x), cos(x)))), whose paths corrupted() takes:
    (0, 0) is x, (0, 1) the divisor 2, (1, 0) the coefficient -1/2, (1, 1) the product, (1, 1, 0) the sine and
    (1, 1, 0, 0) its argument, (1, 1, 1) the cosine."""
    return RealExpr("add", (RealExpr("div", (leaf(), scalar(2))),
                            RealExpr("mul", (scalar(Fraction(-1, 2)),
                                             RealExpr("mul", (sin_node(leaf()), cos_node(leaf())))))))


# ---------------------------------------------------------------------------------------------------------------
# Claims, built from the reference.

SIN2: Poly = {(2, 0): 1}


def bound(number: Number | Rational) -> int | Rational:
    """lower_pi or upper_pi: an int when the number is one, else a Rational."""
    number = fraction(number)
    return number.numerator if number.denominator == 1 else Rational(number.numerator, number.denominator)


def pi_linear(element: Element, pi_coefficient: Number) -> family.PiLinear:
    """The canonical PiLinear: mathforge.model's normal form of the algebraic part, a Rational pi coefficient."""
    c0, c1, c2, c3 = (q(c) for c in element)
    return family.PiLinear(add(c0, mul(c1, sqrt(2)), mul(c2, sqrt(3)), mul(c3, sqrt(6))), q(pi_coefficient))


def antiderivative_claim(poly: Poly, a: Number, *, style: str = "mul", variable: Symbol = X) -> Claim:
    u = argument(a, variable)
    args = {"function": function(poly_tree(poly, u), variable), "variable": variable}
    return ANTIDERIVATIVE, args, function(nf_tree(antiderivative(poly, a), u, style, variable), variable), ()


def integral_claim(poly: Poly, a: Number, lower: Number, upper: Number, *, style: str = "mul",
                   variable: Symbol = X) -> Claim:
    _, args, antiderivative_value, _ = antiderivative_claim(poly, a, style=style, variable=variable)
    element, pi_coefficient = integral(poly, a, lower, upper)
    return (INTEGRAL, dict(args, lower_pi=bound(lower), upper_pi=bound(upper)), pi_linear(element, pi_coefficient),
            (antiderivative_value,))


def claimed(operation: str, args: dict[str, object], value: object, evidence: tuple) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(operation, tuple(args.items()), 6)
    step = contracts.Step("v6." + operation, (request,), (value, evidence),
                          relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=operation, request=request,
                            steps=(step,), **operations_v6.source_context(args))


FREQUENCIES = (1, -1, 2, 3, Fraction(1, 2), Fraction(-3, 4), Fraction(5, 3))


def random_poly(rng: random.Random, degree: int, terms: int = 5, size: int = 30) -> Poly:
    poly: dict[tuple[int, int], Fraction] = {}
    for _ in range(terms):
        j = rng.randrange(degree + 1)
        k = rng.randrange(degree + 1 - j)
        poly[(j, k)] = poly.get((j, k), Fraction(0)) + Fraction(rng.randrange(-size, size + 1), rng.randrange(1, size))
    return {key: c for key, c in poly.items() if c}


def random_bound(rng: random.Random, a: Number) -> Fraction:
    """A multiple of pi/(12 a), as the bounds of a request with a sine or a cosine must be."""
    return Fraction(rng.randrange(-40, 41)) / (12 * Fraction(a))


def rational_stand_ins(number: Number) -> list[tuple[str, object]]:
    """Stand-ins for an exact rational that are no canonical Rational: a bool, a float, a str, None, a Fraction, a
    Rational subclass, an int subclass, a list and a pair of its integers, an impostor class, and Rationals forged
    without __init__."""
    number = Fraction(number)
    n, d = number.numerator, number.denominator
    return [("a bool", bool(number)), ("a float", float(number)), ("a str", str(number)), ("None", None),
            ("a Fraction", number), ("a Rational subclass", Ratio(n, d)), ("an int subclass", Integer(n)),
            ("a list", [n, d]), ("a pair", (n, d)), ("an impostor of Rational", support.impostor(Rational(n, d))),
            ("a Rational forged without fields", support.forged(Rational)),
            ("a Rational forged without its denominator", support.forged(Rational, numerator=n)),
            ("a Rational not in lowest terms", support.forged(Rational, numerator=3 * n, denominator=3 * d)),
            ("a Rational with a negative denominator", support.forged(Rational, numerator=-n, denominator=-d)),
            ("a Rational with a float numerator", support.forged(Rational, numerator=float(n), denominator=d)),
            ("a Rational with a bool denominator", support.forged(Rational, numerator=n, denominator=True))]


class TrigPowersCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def assertRejectsAll(self, operation: object, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(operation=operation, case=label):
                self.assertRejects(operation, args, value, evidence)

    def assertAcceptsAll(self, claims: Sequence[tuple[str, Claim]]) -> None:
        for label, claim in claims:
            with self.subTest(claim=label):
                self.assertAccepts(*claim)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_specs_of_the_card(self):
        function_argument = Argument("function", "RealFunction", None, "function")
        variable_argument = Argument("variable", "Symbol", None, "symbol")
        lower_argument = Argument("lower_pi", "Rational | int", None, "rational")
        upper_argument = Argument("upper_pi", "Rational | int", None, "rational")
        expected = (
            OperationSpec(ANTIDERIVATIVE, "integrals", (function_argument, variable_argument), "RealFunction",
                          "Integrate a polynomial in sin(a*x) and cos(a*x) in closed form, with powers of the same "
                          "sine and cosine.", SCOPE, version=6),
            OperationSpec(INTEGRAL, "integrals",
                          (function_argument, variable_argument, lower_argument, upper_argument), "PiLinear",
                          "Integrate a polynomial in sin(a*x) and cos(a*x) between rational multiples of pi.", SCOPE,
                          version=6),
        )
        self.assertEqual(family.SPECS, expected)
        self.assertEqual(family.TYPES, {"pi_linear": family.PiLinear})
        self.assertEqual(family.CHECKER, "mathforge.certify.integrals.trig_powers")
        self.assertEqual(set(family.RENDER), {family.PiLinear})
        self.assertEqual(checker.RECORD, (family.__name__, "PiLinear"))
        self.assertEqual(checker.OPERATIONS, {spec.name: tuple(argument.name for argument in spec.arguments)
                                              for spec in expected})
        for spec in expected:
            with self.subTest(operation=spec.name):
                self.assertEqual(registry.spec(spec.name), spec)
                self.assertIs(registry.checker(spec.name), checker)
        self.assertIs(support.checker(family), checker)
        with computation(fresh=True):  # The producer refuses only operations of other families.
            with self.assertRaises(mf.UnsupportedOperation):
                family.produce("trig_power_derivative", {"function": text("sin(x)^2"), "variable": X})

    def test_pi_linear_canonical_form_and_rendering(self):
        root3 = Sqrt(Rational(3))
        four = Add((Rational(1, 2), Sqrt(Rational(6)), Mul((Rational(-1, 4), Sqrt(Rational(2)))),
                    Mul((Rational(3, 8), root3))))
        canonical = [  # (algebraic, pi coefficient, its coefficients, text, LaTeX), the forms written out by hand.
            (Rational(0), Rational(1, 2), E0, "pi/2", r"\frac{\pi}{2}"),
            (Rational(4, 3), Rational(0), (Fraction(4, 3), 0, 0, 0), "4/3", r"\frac{4}{3}"),
            (Mul((Rational(-1, 8), root3)), Rational(1, 6), (0, 0, Fraction(-1, 8), 0), "-sqrt(3)/8 + pi/6",
             r"-\frac{\sqrt{3}}{8} + \frac{\pi}{6}"),
            (Mul((Rational(3, 8), root3)), Rational(1, 6), (0, 0, Fraction(3, 8), 0), "3*sqrt(3)/8 + pi/6",
             r"\frac{3 \sqrt{3}}{8} + \frac{\pi}{6}"),
            (Sqrt(Rational(2)), Rational(0), (0, 1, 0, 0), "sqrt(2)", r"\sqrt{2}"),
            (Rational(0), Rational(0), E0, "0", "0"),
            (Rational(0), Rational(-2), E0, "-2*pi", r"-2 \pi"),
            (four, Rational(1, 6), (Fraction(1, 2), Fraction(-1, 4), Fraction(3, 8), 1),
             "1/2 - sqrt(2)/4 + 3*sqrt(3)/8 + sqrt(6) + pi/6",
             r"\frac{1}{2} - \frac{\sqrt{2}}{4} + \frac{3 \sqrt{3}}{8} + \sqrt{6} + \frac{\pi}{6}"),
        ]
        for algebraic, pi_coefficient, coefficients, rendered, latex in canonical:
            with self.subTest(value=rendered):
                value = family.PiLinear(algebraic, pi_coefficient)
                self.assertEqual(value, pi_linear(coefficients, pi_coefficient))
                self.assertEqual(mf.render_text(value), rendered)
                self.assertEqual(mf.to_latex(value), latex)
                self.assertEqual(mf.loads(mf.dumps(value)), value)
        unsorted = Add((Rational(1, 2), Mul((Rational(-1, 4), Sqrt(Rational(2)))), Sqrt(Rational(6)),
                        Mul((Rational(3, 8), root3))))
        for label, algebraic, pi_coefficient in [
                ("an int pi coefficient", Rational(0), 1), ("a float pi coefficient", Rational(0), 0.5),
                ("a bool pi coefficient", Rational(0), True),
                ("a Fraction pi coefficient", Rational(0), Fraction(1, 2)),
                ("a Rational subclass", Rational(0), Ratio(1, 2)),
                ("a pi coefficient not in lowest terms", Rational(0),
                 support.forged(Rational, numerator=2, denominator=4)),
                ("an int algebraic part", 0, Rational(1)), ("None", None, Rational(1)),
                ("an unsorted Add", unsorted, Rational(1)), ("an Add with a zero term", Add((Rational(0), root3)),
                                                              Rational(1)),
                ("1*sqrt(3)", Mul((Rational(1), root3)), Rational(1)),
                ("0*sqrt(3)", Mul((Rational(0), root3)), Rational(1)),
                ("the factors reversed", Mul((root3, Rational(-1, 8))), Rational(1)),
                ("sqrt(12)/2", Mul((Rational(1, 2), Sqrt(Rational(12)))), Rational(1)),
                ("sqrt(2)*sqrt(3)", Mul((Sqrt(Rational(2)), root3)), Rational(1)),
                ("sqrt(5)", Sqrt(Rational(5)), Rational(1)), ("sqrt(1/2)", Sqrt(Rational(1, 2)), Rational(1)),
                ("a power", Pow(root3, 3), Rational(1)), ("a Symbol", X, Rational(1)),
                ("two terms of sqrt(3)", Add((Mul((Rational(-1, 8), root3)), root3)), Rational(1)),
                ("a nested Add", Add((Rational(1), Add((Sqrt(Rational(2)), root3)))), Rational(1)),
                ("a RealExpr", scalar(1), Rational(1))]:
            with self.subTest(case=label), self.assertRaises(InvalidInput):
                family.PiLinear(algebraic, pi_coefficient)

    # -----------------------------------------------------------------------------------------------------------
    # The reference and the table

    def test_reference_reproduces_the_known_values(self):
        half = Fraction(1, 2)
        for poly, a, expected in [(SIN2, 1, {(1, 0, 0): half, (0, 1, 1): -half}),
                                  ({(0, 2): 1}, 1, {(1, 0, 0): half, (0, 1, 1): half}),
                                  ({(3, 0): 1}, 1, {(0, 0, 1): Fraction(-1), (0, 0, 3): Fraction(1, 3)}),
                                  ({(1, 2): 1}, 2, {(0, 0, 3): Fraction(-1, 6)}),
                                  ({(1, 1): 1}, 1, {(0, 0, 2): -half}),
                                  ({(0, 0): 3}, 5, {(1, 0, 0): Fraction(3)}), ({}, 1, {})]:
            with self.subTest(poly=poly, a=a):
                self.assertEqual(antiderivative(poly, a), expected)
        for poly, lower, upper, expected in [(SIN2, 0, 1, (E0, half)), ({(0, 2): 1}, 0, half, (E0, Fraction(1, 4))),
                                             ({(3, 0): 1}, 0, 1, ((Fraction(4, 3), 0, 0, 0), 0)),
                                             (SIN2, 0, Fraction(1, 3), ((0, 0, Fraction(-1, 8), 0), Fraction(1, 6)))]:
            with self.subTest(poly=poly, upper=upper):
                self.assertEqual(integral(poly, 1, lower, upper), expected)
        # The roots: exp(i pi/12)**n, with cos**2 + sin**2 = 1, the right values at pi/6, pi/4, pi/3 and pi/2, and
        # positive sines in (0, pi): the six 24th roots z with z**6 = i leave only exp(i pi/12).
        third = (Fraction(0), Fraction(0), half, Fraction(0))
        self.assertEqual(zeta(2), (third, (half, 0, 0, 0)))
        self.assertEqual(zeta(3), ((0, half, 0, 0), (0, half, 0, 0)))
        self.assertEqual(zeta(4), ((half, 0, 0, 0), third))
        self.assertEqual(zeta(6), (E0, E1))
        self.assertEqual(zeta(12), (e_scale(E1, -1), E0))
        self.assertEqual(c_mul(zeta(23), OMEGA), zeta(0))
        for n in range(24):
            cosine, sine = zeta(n)
            self.assertEqual(e_add(e_mul(cosine, cosine), e_mul(sine, sine)), E1)
            self.assertEqual(sign(sine), 1 if 0 < n < 12 else -1 if n > 12 else 0)
        # The two halves of the reference agree: the antiderivative at the bounds gives the integral.
        rng = random.Random(20261112)
        for _ in range(60):
            a = rng.choice(FREQUENCIES)
            poly = random_poly(rng, rng.randrange(7))
            lower, upper = random_bound(rng, a), random_bound(rng, a)
            nf = antiderivative(poly, a)
            (high, high_pi), (low, low_pi) = evaluate(nf, a, upper), evaluate(nf, a, lower)
            self.assertEqual(integral(poly, a, lower, upper), (e_sub(high, low), high_pi - low_pi))

    def test_table_of_sines_and_cosines(self):
        for k in range(-60, 61):
            with self.subTest(k=k):
                cosine, sine = zeta(k)
                for found, expected in ((checker.sine(k), sine), (checker.cosine(k), cosine)):
                    self.assertTrue(all(type(item) is Rational for item in found))
                    self.assertEqual(tuple(Fraction(item.numerator, item.denominator) for item in found), expected)
        self.assertEqual(checker.FIELD.base, (2, 3))

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_antiderivative_accepts_the_known_values(self):
        for source, value, poly, a in [("sin(x)^2", "x/2 - sin(x)*cos(x)/2", SIN2, 1),
                                       ("cos(x)^2", "x/2 + sin(x)*cos(x)/2", {(0, 2): 1}, 1),
                                       ("sin(x)^3", "-cos(x) + cos(x)^3/3", {(3, 0): 1}, 1),
                                       ("sin(2*x)*cos(2*x)^2", "-cos(2*x)^3/6", {(1, 2): 1}, 2),
                                       ("sin(x)*cos(x)", "-cos(x)^2/2", {(1, 1): 1}, 1)]:
            with self.subTest(function=source):
                args = {"function": text(source), "variable": X}
                self.assertAccepts(ANTIDERIVATIVE, args, text(value))
                self.assertAccepts(*antiderivative_claim(poly, a))  # The reference's tree of the same values.
                self.assertAccepts(ANTIDERIVATIVE, args, antiderivative_claim(poly, a)[2])

    def test_antiderivative_accepts_true_claims(self):
        cancelling = function(RealExpr("add", (sin_node(leaf()), RealExpr("mul", (scalar(-1), sin_node(leaf()))))))
        claims = [
            ("a constant", antiderivative_claim({(0, 0): 3}, 1)),
            ("a negative constant", antiderivative_claim({(0, 0): Fraction(-2, 7)}, 1)),
            ("zero", antiderivative_claim({}, 1)),
            ("sines that cancel", (ANTIDERIVATIVE, {"function": cancelling, "variable": X}, function(scalar(0)), ())),
            ("sin(x)**0", (ANTIDERIVATIVE, {"function": function(power(sin_node(leaf()), 0)), "variable": X},
                           function(leaf()), ())),
            ("sin(x)**9 cos(x)**6", antiderivative_claim({(9, 6): 1}, 1)),
            ("cos(x)**31", antiderivative_claim({(0, 31): 1}, Fraction(2, 3))),
            ("sin(x)**31", antiderivative_claim({(31, 0): Fraction(-5, 2)}, -1)),
            ("sin(x) + cos(x) + 1", antiderivative_claim({(1, 0): 1, (0, 1): 1, (0, 0): 1}, 7)),
            ("in another variable", antiderivative_claim({(2, 3): 4, (1, 0): -1}, Fraction(3, 2), variable=Y)),
            ("a long frequency", antiderivative_claim({(1, 1): 1, (2, 0): 3}, Fraction(10 ** 30 + 7, 3 ** 40))),
        ]
        rng = random.Random(20261113)
        styles = ("mul", "div", "neg")
        for index in range(45):
            a = rng.choice(FREQUENCIES)
            poly = random_poly(rng, rng.randrange(1, 9), terms=rng.randrange(1, 7))
            claims.append((f"random {index}", antiderivative_claim(poly, a, style=styles[index % 3])))
        self.assertAcceptsAll(claims)

    def test_antiderivative_accepts_any_tree_of_the_canonical_function(self):
        """Any tree of the admitted nodes that denotes the canonical antiderivative passes, whatever its shape."""
        x = leaf()
        s, c = sin_node(x), cos_node(x)
        args = {"function": text("sin(x)^2"), "variable": X}
        canonical = sin2_tree()
        identity = RealExpr("add", (RealExpr("add", (power(s, 2), power(c, 2))), scalar(-1)))  # S**2 + C**2 - 1
        nf = antiderivative(SIN2, 1)
        trees = [
            ("written out", canonical),
            ("parsed", text("x/2 - sin(x)*cos(x)/2").expression),
            ("by RealExpr arithmetic", x / 2 - functions.sin(x) * functions.cos(x) / 2),
            ("simplified by real_function", real_function(x / 2 - functions.sin(x) * functions.cos(x) / 2,
                                                          (X,)).expression),
            ("in Horner's form", horner_tree(nf, x)),
            ("the terms reversed", RealExpr("add", (canonical.args[1], canonical.args[0]))),
            ("with 7 (S**2 + C**2 - 1)", RealExpr("add", (canonical, RealExpr("mul", (scalar(7), identity))))),
            ("x (S**2 + C**2)/2 - S C/2", RealExpr("add", (
                RealExpr("div", (RealExpr("mul", (x, RealExpr("add", (power(s, 2), power(c, 2))))), scalar(2))),
                canonical.args[1]))),
            ("x/2 - ((S + C)**2 - 1)/4", RealExpr("add", (canonical.args[0], RealExpr("div", (
                RealExpr("add", (power(RealExpr("add", (s, c)), 2), scalar(-1))), scalar(-4)))))),
            ("a shared sine and a shared product", RealExpr("add", (canonical, RealExpr("add", (
                RealExpr("mul", (s, c)), RealExpr("mul", (scalar(-1), RealExpr("mul", (s, c))))))))),
            ("x as x**1 and the product as a power 1", RealExpr("add", (
                RealExpr("div", (RealExpr("pow", (x,), 1), scalar(2))),
                RealExpr("mul", (scalar(Fraction(-1, 2)), RealExpr("pow", (RealExpr("mul", (s, c)),), 1)))))),
            ("a division by -2", RealExpr("add", (RealExpr("div", (x, scalar(2))),
                                                  RealExpr("div", (RealExpr("mul", (s, c)), scalar(-2)))))),
            ("a multiple of sin(x)**0", RealExpr("add", (canonical, RealExpr("mul", (scalar(0),
                                                                                   RealExpr("pow", (s,), 0)))))),
        ]
        for label, tree in trees:
            with self.subTest(tree=label):
                self.assertAccepts(ANTIDERIVATIVE, args, function(tree))
        # An unreduced sum for sin(x)**2 + cos(x)**2: the antiderivatives of both terms, whose products cancel.
        both = {"function": text("sin(x)^2 + cos(x)^2"), "variable": X}
        unreduced = RealExpr("add", (canonical, function(nf_tree(antiderivative({(0, 2): 1}, 1), x)).expression))
        self.assertAccepts(ANTIDERIVATIVE, both, function(unreduced))
        self.assertAccepts(ANTIDERIVATIVE, both, function(x))
        # Arguments: the same object in f and F, other trees of the same a, a constant that cancels.
        u = argument(2)
        poly = {(1, 2): 1}
        f = function(poly_tree(poly, u))
        nf = antiderivative(poly, 2)
        for label, v in [("f's own argument", u), ("x + x", RealExpr("add", (leaf(), leaf()))),
                         ("x/(1/2)", RealExpr("div", (leaf(), scalar(Fraction(1, 2))))),
                         ("x*2", RealExpr("mul", (leaf(), scalar(2)))),
                         ("(2x + 1) - 1", RealExpr("add", (RealExpr("add", (u, scalar(1))), scalar(-1))))]:
            with self.subTest(argument=label):
                self.assertAccepts(ANTIDERIVATIVE, {"function": f, "variable": X}, function(nf_tree(nf, v)))
                self.assertAccepts(ANTIDERIVATIVE, {"function": function(poly_tree(poly, v)), "variable": X},
                                   function(nf_tree(nf, u)))

    def test_integral_accepts_the_known_values(self):
        for source, value, lower, upper, algebraic, pi_coefficient, rendered in [
                ("sin(x)^2", "x/2 - sin(x)*cos(x)/2", 0, 1, Rational(0), Rational(1, 2), "pi/2"),
                ("cos(x)^2", "x/2 + sin(x)*cos(x)/2", 0, Rational(1, 2), Rational(0), Rational(1, 4), "pi/4"),
                ("sin(x)^3", "-cos(x) + cos(x)^3/3", 0, 1, Rational(4, 3), Rational(0), "4/3"),
                ("sin(x)^2", "x/2 - sin(x)*cos(x)/2", 0, Rational(1, 3), Mul((Rational(-1, 8), Sqrt(Rational(3)))),
                 Rational(1, 6), "-sqrt(3)/8 + pi/6")]:
            with self.subTest(function=source, upper=upper):
                expected = family.PiLinear(algebraic, pi_coefficient)
                self.assertEqual(mf.render_text(expected), rendered)
                poly = {(2, 0): 1} if source == "sin(x)^2" else {(0, 2): 1} if source == "cos(x)^2" else {(3, 0): 1}
                self.assertEqual(pi_linear(*integral(poly, 1, lower, upper)), expected)
                args = {"function": text(source), "variable": X, "lower_pi": lower, "upper_pi": upper}
                self.assertAccepts(INTEGRAL, args, expected, (text(value),))
                self.assertAccepts(*integral_claim(poly, 1, lower, upper))

    def test_integral_accepts_true_claims(self):
        mixed = {(3, 2): 1, (0, 1): 2, (1, 1): Fraction(-1, 3), (0, 0): Fraction(1, 5)}
        claims = []
        for k in range(24):  # Every angle class modulo 2 pi, from 0 and between two of them.
            claims.append((f"[0, {k}/12]", integral_claim(mixed, 1, 0, Fraction(k, 12))))
            claims.append((f"[{k}/12, {k + 5}/12]", integral_claim(mixed, 1, Fraction(k, 12), Fraction(k + 5, 12))))
        claims += [
            ("the bounds reversed", integral_claim(mixed, 1, Fraction(7, 12), Fraction(-1, 4))),
            ("equal bounds", integral_claim(mixed, 1, Fraction(5, 12), Fraction(5, 12))),
            ("int bounds", integral_claim(SIN2, 1, -3, 2)),
            ("long bounds", integral_claim(mixed, 1, Fraction(-(10 ** 30) - 1, 12), Fraction(10 ** 40 + 5, 12))),
            ("a = 1/2, multiples of pi/6", integral_claim({(1, 3): 1, (2, 0): 2}, Fraction(1, 2), Fraction(-1, 6),
                                                          Fraction(5, 6))),
            ("a = -3/4, multiples of pi/9", integral_claim({(2, 2): 1}, Fraction(-3, 4), Fraction(1, 9),
                                                           Fraction(4, 9))),
            ("a = 5/3, multiples of pi/20", integral_claim({(0, 4): 1}, Fraction(5, 3), Fraction(-1, 20),
                                                           Fraction(3, 20))),
            ("a constant on any rational bounds", integral_claim({(0, 0): 3}, 1, Fraction(1, 7), Fraction(2, 9))),
            ("zero", integral_claim({}, 1, Fraction(-5, 11), 3)),
            ("in another variable", integral_claim(mixed, 2, 0, Fraction(1, 8), variable=Y)),
            ("divisions", integral_claim(mixed, 1, Fraction(1, 12), Fraction(11, 12), style="div")),
            ("negations", integral_claim(mixed, 1, Fraction(1, 12), Fraction(11, 12), style="neg")),
        ]
        # Sines that cancel still fix a, and the bounds are multiples of pi/12 then; Rational bounds that are ints.
        cancelling = function(RealExpr("add", (scalar(1), RealExpr("add", (sin_node(leaf()), RealExpr(
            "mul", (scalar(-1), sin_node(leaf()))))))))
        claims.append(("sines that cancel", (INTEGRAL, {"function": cancelling, "variable": X, "lower_pi": 0,
                                                        "upper_pi": Rational(1, 12)},
                                             pi_linear(E0, Fraction(1, 12)), (function(leaf()),))))
        operation, args, value, evidence = integral_claim(SIN2, 1, -3, 2)
        claims.append(("Rational bounds that are ints", (operation, dict(args, lower_pi=Rational(-3),
                                                                         upper_pi=Rational(2)), value, evidence)))
        rng = random.Random(20261114)
        surds = 0
        for index in range(50):
            a = rng.choice(FREQUENCIES)
            poly = random_poly(rng, rng.randrange(1, 8), terms=rng.randrange(1, 7))
            claim = integral_claim(poly, a, random_bound(rng, a), random_bound(rng, a),
                                   style=("mul", "div", "neg")[index % 3])
            algebraic = claim[2].algebraic
            surds += type(algebraic) is Add and len(algebraic.terms) == 4
            claims.append((f"random {index}", claim))
        self.assertGreater(surds, 0)  # Some values have all of 1, sqrt 2, sqrt 3 and sqrt 6.
        self.assertAcceptsAll(claims)

    def test_integral_bounds_are_free_when_f_reduces_to_a_constant(self):
        """When f reduces to a constant c, F is c*x and the integral is c (upper - lower) pi on any rational bounds,
        with no table to evaluate: the checker accepts that true claim also where f has sines that cancel and the
        bounds are no multiples of pi/(12 a). The producer refuses those requests, as the card says; the checker's
        domain contains the producer's. Every other value and every other antiderivative fails there."""
        x = leaf()
        s, c = sin_node(x), cos_node(x)
        requests = [
            ("3 + S - S", function(RealExpr("add", (scalar(3), RealExpr("add", (
                s, RealExpr("mul", (scalar(-1), s))))))), Fraction(3)),
            ("sin(x)**0", function(power(s, 0)), Fraction(1)),
            ("S**2 + C**2", function(RealExpr("add", (power(s, 2), power(c, 2)))), Fraction(1)),
            ("(S**2 + C**2)/(-2) with a = 2/3", function(RealExpr("div", (RealExpr("add", (
                power(sin_node(argument(Fraction(2, 3))), 2), power(cos_node(argument(Fraction(2, 3))), 2))),
                scalar(-2)))), Fraction(-1, 2)),
            ("0*cos(2x)", function(RealExpr("mul", (scalar(0), cos_node(argument(2))))), Fraction(0)),
        ]
        for label, f, constant in requests:
            antiderivative_value = function(RealExpr("mul", (scalar(constant), x)) if constant else scalar(0))
            self.assertAccepts(ANTIDERIVATIVE, {"function": f, "variable": X}, antiderivative_value)
            for lower, upper in [(0, Fraction(1, 7)), (Fraction(-2, 5), Fraction(1, 3)), (Fraction(1, 12), 2)]:
                with self.subTest(function=label, lower=lower, upper=upper):
                    args = {"function": f, "variable": X, "lower_pi": bound(lower), "upper_pi": bound(upper)}
                    pi_part = constant * (Fraction(upper) - Fraction(lower))
                    self.assertAccepts(INTEGRAL, args, pi_linear(E0, pi_part), (antiderivative_value,))
                    self.assertRejectsAll(INTEGRAL, [
                        ("another pi coefficient", args, pi_linear(E0, pi_part + 1), (antiderivative_value,)),
                        ("an algebraic part", args, pi_linear((Fraction(1, 2), 0, 0, 0), pi_part),
                         (antiderivative_value,)),
                        ("a surd", args, pi_linear((0, 0, 1, 0), pi_part), (antiderivative_value,)),
                        ("F with a constant term", args, pi_linear(E0, pi_part),
                         (function(RealExpr("add", (antiderivative_value.expression, scalar(1)))),)),
                        ("F with a sine", args, pi_linear(E0, pi_part),
                         (function(RealExpr("add", (antiderivative_value.expression, s))),)),
                        ("F of another constant", args, pi_linear(E0, pi_part),
                         (function(RealExpr("mul", (scalar(constant + 1), x))),)),
                    ])

    def test_arguments_are_read_for_what_they_denote(self):
        """The argument of a sine or a cosine is any tree of the admitted nodes but sin and cos that denotes a*x,
        whatever the degrees of its subtrees: x*x - x*x + x is x. A request so written is in the input class, and its
        canonical antiderivative passes with a small u, with f's own argument tree, or with another such tree; so
        does its integral. An argument that denotes anything else (x**2, zero, a constant, x + 1) fails, in the
        request and in the claim."""
        x = leaf()
        square = RealExpr("mul", (x, x))
        cube = RealExpr("pow", (x,), 3)
        contrived = [  # (a, a tree of a*x with subtrees of a higher degree)
            (1, RealExpr("add", (RealExpr("add", (square, RealExpr("mul", (scalar(-1), square)))), x))),
            (2, RealExpr("add", (RealExpr("mul", (RealExpr("add", (x, RealExpr("mul", (scalar(-1), x)))), x)),
                                 argument(2)))),
            (Fraction(-3, 4), RealExpr("div", (RealExpr("add", (cube, RealExpr("add", (
                RealExpr("mul", (scalar(-3), x)), RealExpr("mul", (scalar(-1), cube)))))), scalar(4)))),
            (Fraction(5, 3), RealExpr("mul", (RealExpr("pow", (RealExpr("add", (x, scalar(1))),), 0),
                                              argument(Fraction(5, 3))))),
        ]
        poly = {(1, 2): 1, (2, 0): Fraction(-2, 3), (0, 3): 5}
        for a, u in contrived:
            with self.subTest(a=a):
                nf = antiderivative(poly, a)
                small = argument(a)
                args = {"function": function(poly_tree(poly, u)), "variable": X}
                self.assertAccepts(ANTIDERIVATIVE, args, function(nf_tree(nf, small)))
                self.assertAccepts(ANTIDERIVATIVE, args, function(nf_tree(nf, u)))  # f's own argument tree
                self.assertAccepts(ANTIDERIVATIVE, {"function": function(poly_tree(poly, small)), "variable": X},
                                   function(nf_tree(nf, u)))  # Such a tree in F alone.
                lower, upper = Fraction(-1) / (12 * Fraction(a)), Fraction(7) / (12 * Fraction(a))
                element, pi_coefficient = integral(poly, a, lower, upper)
                integral_args = dict(args, lower_pi=bound(lower), upper_pi=bound(upper))
                evidence = (function(nf_tree(nf, u)),)
                self.assertAccepts(INTEGRAL, integral_args, pi_linear(element, pi_coefficient), evidence)
                self.assertRejects(INTEGRAL, integral_args, pi_linear(element, pi_coefficient + 1), evidence)
        sin2 = {"function": text("sin(x)^2"), "variable": X}
        sin2_value = antiderivative(SIN2, 1)
        others = [("x*x", square),
                  ("x*x - x*x", RealExpr("add", (square, RealExpr("mul", (scalar(-1), square))))),
                  ("x*x - x*x + 1", RealExpr("add", (RealExpr("add", (square, RealExpr("mul", (scalar(-1), square)))),
                                                     scalar(1)))),
                  ("(x + 1)**1", RealExpr("pow", (RealExpr("add", (x, scalar(1))),), 1)),
                  ("(x + 1)*(x - 1) - x*x", RealExpr("add", (RealExpr("mul", (RealExpr("add", (x, scalar(1))), RealExpr(
                      "add", (x, scalar(-1))))), RealExpr("mul", (scalar(-1), square))))),
                  ("x**0", RealExpr("pow", (x,), 0))]
        cases = []
        for label, u in others:
            value = function(nf_tree(sin2_value, u))
            cases.append((f"a request with sin({label})", {"function": function(power(sin_node(u), 2)), "variable": X},
                          value, ()))
            cases.append((f"a claim with sin({label})", sin2, value, ()))
        self.assertRejectsAll(ANTIDERIVATIVE, cases)

    def test_symbols_are_compared_field_by_field(self):
        """A Symbol is the variable when its name and uid are the variable's exact strs, its domain Reals and its
        assumptions none: a Symbol forged with fields that claim to equal everything, or that raise when compared,
        is no variable (in the request, in a function's variables or in a symbol node), and nothing escapes. A
        Symbol built anew with the variable's name and uid is the variable."""

        class Anything:
            """Equal to everything."""
            __hash__ = object.__hash__

            def __eq__(self, other: object) -> bool:
                return True

            def __ne__(self, other: object) -> bool:
                return False

        class Explosive:
            """Raises when compared."""
            __hash__ = object.__hash__

            def __eq__(self, other: object) -> bool:
                raise RuntimeError("compared")

            __ne__ = __eq__

        args = {"function": text("sin(x)^2"), "variable": X}
        _, integral_args, integral_value, _ = integral_claim(SIN2, 1, 0, Fraction(1, 3))
        canonical = sin2_tree()
        self.assertAccepts(ANTIDERIVATIVE, args, function(canonical))
        twin = Symbol("x", Reals, (), X.uid)  # Equal to X, another object.
        self.assertIsNot(twin, X)
        self.assertAccepts(ANTIDERIVATIVE, {"function": text("sin(x)^2"), "variable": twin}, function(canonical))
        self.assertAccepts(ANTIDERIVATIVE, args, forged_function(corrupted(canonical, (0, 0), leaf(twin)), (twin,)))
        impostors = []
        for kind in (Anything, Explosive):
            impostors += [(f"{kind.__name__} name and uid", support.forged(Symbol, name=kind(), domain=Reals,
                                                                        assumptions=(), uid=kind())),
                          (f"{kind.__name__} name", support.forged(Symbol, name=kind(), domain=Reals, assumptions=(),
                                                                   uid=X.uid)),
                          (f"{kind.__name__} assumptions", support.forged(Symbol, name="x", domain=Reals,
                                                                          assumptions=kind(), uid=X.uid))]
        impostors.append(("a Symbol forged without a name", support.forged(Symbol, domain=Reals, assumptions=(),
                                                                            uid=X.uid)))
        cases = []
        for label, impostor in impostors:
            cases += [
                (f"{label} in a symbol node", args, forged_function(corrupted(canonical, (0, 0), node("symbol", (),
                                                                                                    impostor))), ()),
                (f"{label} as the function's variable", args, forged_function(canonical, (impostor,)), ()),
                (f"{label} as the request's variable", {"function": text("sin(x)^2"), "variable": impostor},
                 function(canonical), ()),
            ]
        self.assertRejectsAll(ANTIDERIVATIVE, cases)
        self.assertRejectsAll(INTEGRAL, [(label, dict(integral_args, variable=case_args["variable"]), integral_value,
                                          (case_value,)) for label, case_args, case_value, _ in cases])

    # -----------------------------------------------------------------------------------------------------------
    # False claims, which the checker refuses

    def test_rejects_claims_on_inputs_outside_the_domain(self):
        """The card's input class: one argument a*x, rational coefficients, the variable inside the arguments only,
        the domain TRUE, the variable alone. Every claim on another request is False, also a mathematically true
        one; so is every integral whose bounds are no multiples of pi/(12 a) when f reduces to more than a
        constant."""
        x = leaf()
        s, c = sin_node(x), cos_node(x)
        sin2 = power(s, 2)
        cases = [
            ("x sin(x)", RealExpr("mul", (x, s)), text("sin(x) - x*cos(x)")),
            ("sin(x + 1)", sin_node(RealExpr("add", (x, scalar(1)))), text("-cos(x + 1)")),
            ("sin(x**2)", sin_node(RealExpr("pow", (x,), 2)), text("-cos(x)")),
            ("sin(2), a constant argument", sin_node(scalar(2)), function(RealExpr("mul", (sin_node(scalar(2)), x)))),
            ("sin(0*x)", sin_node(RealExpr("mul", (scalar(0), x))), function(scalar(0))),
            ("sin(x) cos(2x)", RealExpr("mul", (s, cos_node(argument(2)))), text("-cos(3*x)/6 + cos(x)/2")),
            ("sin(x) sin(-x)", RealExpr("mul", (s, sin_node(argument(-1)))), text("-x/2 + sin(x)*cos(x)/2")),
            ("tan(x) cos(x)", RealExpr("mul", (RealExpr("tan", (x,)), c)), text("-cos(x)")),
            ("exp(x)", RealExpr("exp", (x,)), text("exp(x)")),
            ("sqrt(2) sin(x)", RealExpr("mul", (functions.sqrt(scalar(2)), s)), function(RealExpr(
                "mul", (functions.sqrt(scalar(2)), RealExpr("mul", (scalar(-1), c)))))),
            ("pi sin(x)", RealExpr("mul", (functions.pi, s)), function(RealExpr("mul", (functions.pi, RealExpr(
                "mul", (scalar(-1), c)))))),
            ("E cos(x)", RealExpr("mul", (functions.E, c)), function(RealExpr("mul", (functions.E, s)))),
            ("abs(cos(x))", RealExpr("abs", (c,)), text("sin(x)")),
            ("an algebraic coefficient", RealExpr("mul", (RealExpr("scalar", (), RealAlgebraicRoot((-2, 0, 1), 1)), c)),
             function(RealExpr("mul", (RealExpr("scalar", (), RealAlgebraicRoot((-2, 0, 1), 1)), s)))),
            ("a negative power", RealExpr("mul", (power(s, 3), RealExpr("pow", (s,), -1))),
             text("x/2 - sin(x)*cos(x)/2")),
            ("a division by the cosine", RealExpr("div", (RealExpr("mul", (s, c)), c)), text("-cos(x)")),
            ("x**0 outside the arguments", RealExpr("mul", (RealExpr("pow", (x,), 0), sin2)),
             text("x/2 - sin(x)*cos(x)/2")),
        ]
        rejected = []
        for label, expression, value in cases:
            try:
                f = function(expression)
            except Exception:  # Some trees need a forged function: their domain would not be TRUE.
                f = forged_function(expression)
            rejected.append((label, {"function": f, "variable": X}, value, ()))
        sin2_function = function(sin2)
        value = text("x/2 - sin(x)*cos(x)/2")
        rejected += [
            ("two variables", {"function": RealFunction(sin2, (X, Y)), "variable": X}, value, ()),
            ("another variable", {"function": function(power(sin_node(leaf(Y)), 2), Y), "variable": X}, value, ()),
            ("a domain other than TRUE", {"function": RealFunction(sin2, (X,), DomainPredicate("nonzero", (x,))),
                                          "variable": X}, value, ()),
            ("the domain FALSE", {"function": forged_function(sin2, domain=functions.FALSE), "variable": X}, value, ()),
            ("a variable over the integers", {"function": sin2_function, "variable": N}, value, ()),
            ("a variable with an assumption", {"function": sin2_function, "variable": Symbol(
                "x", Reals, (Assumption("positive", Truth.TRUE),))}, value, ()),
            ("a str variable", {"function": sin2_function, "variable": "x"}, value, ()),
            ("the variable as a RealExpr", {"function": sin2_function, "variable": leaf()}, value, ()),
            ("a RealExpr for the function", {"function": sin2, "variable": X}, value, ()),
            ("a Polynomial for the function", {"function": Polynomial((Rational(1),), X), "variable": X},
             function(leaf()), ()),
            ("a str function", {"function": "sin(x)^2", "variable": X}, value, ()),
        ]
        self.assertRejectsAll(ANTIDERIVATIVE, rejected)
        # The integral: bounds that are no multiples of pi/(12 a) where f reduces to more than a constant, whose
        # antiderivative the table must then evaluate there (test_integral_bounds_are_free_when_f_reduces_to_a_constant
        # has the other requests).
        antiderivative_value = text("x/2 - sin(x)*cos(x)/2")
        args = {"function": sin2_function, "variable": X, "lower_pi": 0, "upper_pi": Rational(1, 7)}
        half = argument(Fraction(1, 2))
        self.assertRejectsAll(INTEGRAL, [
            ("sin(x)**2 on [0, pi/7], its pi part", args, pi_linear(E0, Fraction(1, 14)), (antiderivative_value,)),
            ("sin(x)**2 on [pi/7, pi/12]", dict(args, lower_pi=Rational(1, 7), upper_pi=Rational(1, 12)),
             pi_linear(E0, Fraction(0)), (antiderivative_value,)),
            ("sin(x)**2 on [0, pi/7] with sines that cancel added", dict(args, function=function(RealExpr("add", (
                sin2, RealExpr("add", (c, RealExpr("mul", (scalar(-1), c)))))))), pi_linear(E0, Fraction(1, 14)),
             (antiderivative_value,)),
            ("a = 1/2 at pi/12", {"function": function(power(sin_node(half), 2)), "variable": X, "lower_pi": 0,
                                  "upper_pi": Rational(1, 12)}, pi_linear(E0, Fraction(1, 24)),
             (function(nf_tree(antiderivative(SIN2, Fraction(1, 2)), half)),)),
            ("x sin(x) on [0, pi]", dict(args, function=function(RealExpr("mul", (x, s))), upper_pi=1),
             pi_linear(E0, 1), (text("sin(x) - x*cos(x)"),)),
        ])

    def test_antiderivative_rejects_corrupted_claims(self):
        args = {"function": text("sin(x)^2"), "variable": X}
        x = leaf()
        s, c = sin_node(x), cos_node(x)
        canonical = sin2_tree()
        value = function(canonical)
        self.assertAccepts(ANTIDERIVATIVE, args, value)
        sin_cos = {"function": text("sin(x)*cos(x)"), "variable": X}
        self.assertAccepts(ANTIDERIVATIVE, sin_cos, text("-cos(x)^2/2"))
        self.assertRejectsAll(ANTIDERIVATIVE, [
            # Wrong values.
            ("a wrong coefficient", args, text("x/2 - sin(x)*cos(x)/3"), ()),
            ("the antiderivative of cos(x)**2", args, text("x/2 + sin(x)*cos(x)/2"), ()),
            ("its negative", args, text("-x/2 + sin(x)*cos(x)/2"), ()),
            ("f itself", args, text("sin(x)^2"), ()),
            ("the derivative of f", args, text("2*sin(x)*cos(x)"), ()),
            ("x/2 alone", args, text("x/2"), ()),
            ("zero", args, function(scalar(0)), ()),
            ("x sin(x)", args, function(RealExpr("mul", (x, s))), ()),
            ("the antiderivative of sin(x)**2 for sin(2x)**2", {"function": text("sin(2*x)^2"), "variable": X},
             value, ()),
            ("-cos(x)**2/2 for sin(x)**2", args, text("-cos(x)^2/2"), ()),
            # Antiderivatives that are not the canonical value.
            ("a constant term", args, function(RealExpr("add", (canonical, scalar(5)))), ()),
            ("a constant term as (S**2 + C**2)/2", args, function(RealExpr("add", (canonical, RealExpr(
                "div", (RealExpr("add", (power(s, 2), power(c, 2))), scalar(2)))))), ()),
            ("sin(x)**2/2 for sin(x)*cos(x)", sin_cos, text("sin(x)^2/2"), ()),
            ("a multiple angle", args, text("x/2 - sin(2*x)/4"), ()),
            ("a multiple angle for sin(x)*cos(x)", sin_cos, text("-cos(2*x)/4"), ()),
            ("x**2 that cancels", args, function(RealExpr("add", (canonical, RealExpr("add", (
                RealExpr("pow", (x,), 2), RealExpr("mul", (scalar(-1), RealExpr("pow", (x,), 2)))))))), ()),
            ("a domain other than TRUE", args, RealFunction(canonical, (X,), DomainPredicate("nonzero", (c,))), ()),
            ("two variables", args, RealFunction(canonical, (X, Y)), ()),
            ("another variable", args, function(nf_tree(antiderivative(SIN2, 1), leaf(Y), variable=Y), Y), ()),
            ("the bare expression", args, canonical, ()),
            # Evidence.
            ("evidence where none belongs", args, value, (value,)),
            ("an empty item as evidence", args, value, ((),)),
            ("evidence as a list", args, value, []),
            ("evidence as a tuple subclass", args, value, Evidence(())),
            ("None as evidence", args, value, None),
            ("a str as evidence", args, value, "()"),
        ])

    def test_integral_rejects_corrupted_claims(self):
        operation, args, value, evidence = integral_claim(SIN2, 1, 0, Fraction(1, 3))
        antiderivative_value = evidence[0]
        root3 = Sqrt(Rational(3))
        self.assertEqual(value, family.PiLinear(Mul((Rational(-1, 8), root3)), Rational(1, 6)))
        self.assertAccepts(operation, args, value, evidence)
        whole = integral_claim(SIN2, 1, 0, 2)  # pi, with the algebraic part 0.
        self.assertEqual(whole[2], family.PiLinear(Rational(0), Rational(1)))

        def forged(algebraic: object = value.algebraic, pi_coefficient: object = value.pi_coefficient) -> object:
            return support.forged(family.PiLinear, algebraic=algebraic, pi_coefficient=pi_coefficient)

        x = leaf()
        s, c = sin_node(x), cos_node(x)
        canonical = antiderivative_value.expression
        cos2_value = function(nf_tree(antiderivative({(0, 2): 1}, 1), x))
        self.assertRejectsAll(INTEGRAL, [
            # Wrong values.
            ("a wrong pi coefficient", args, pi_linear((0, 0, Fraction(-1, 8), 0), Fraction(1, 3)), evidence),
            ("the opposite pi coefficient", args, pi_linear((0, 0, Fraction(-1, 8), 0), Fraction(-1, 6)), evidence),
            ("no pi part", args, pi_linear((0, 0, Fraction(-1, 8), 0), 0), evidence),
            ("the opposite algebraic part", args, pi_linear((0, 0, Fraction(1, 8), 0), Fraction(1, 6)), evidence),
            ("no algebraic part", args, pi_linear(E0, Fraction(1, 6)), evidence),
            ("sqrt(2) for sqrt(3)", args, pi_linear((0, Fraction(-1, 8), 0, 0), Fraction(1, 6)), evidence),
            ("the bounds swapped", args, pi_linear((0, 0, Fraction(1, 8), 0), Fraction(-1, 6)), evidence),
            ("the integral of cos(x)**2", args, pi_linear((0, 0, Fraction(1, 8), 0), Fraction(1, 6)), evidence),
            # The right number in another form.
            ("an int pi coefficient", whole[1], forged(Rational(0), 1), whole[3]),
            ("a bool pi coefficient", whole[1], forged(Rational(0), True), whole[3]),
            ("a float pi coefficient", whole[1], forged(Rational(0), 1.0), whole[3]),
            ("a str pi coefficient", whole[1], forged(Rational(0), "1"), whole[3]),
            ("a Fraction pi coefficient", args, forged(pi_coefficient=Fraction(1, 6)), evidence),
            ("a pi coefficient not in lowest terms", args,
             forged(pi_coefficient=support.forged(Rational, numerator=2, denominator=12)), evidence),
            ("a pi coefficient with a negative denominator", args,
             forged(pi_coefficient=support.forged(Rational, numerator=-1, denominator=-6)), evidence),
            ("an int algebraic part", whole[1], forged(0, Rational(1)), whole[3]),
            ("a float algebraic part", whole[1], forged(0.0, Rational(1)), whole[3]),
            ("a Fraction algebraic part", whole[1], forged(Fraction(0), Rational(1)), whole[3]),
            ("the factors reversed", args, forged(Mul((root3, Rational(-1, 8)))), evidence),
            ("sqrt(12)/(-16)", args, forged(Mul((Rational(-1, 16), Sqrt(Rational(12))))), evidence),
            ("three factors", args, forged(Mul((Rational(-1, 8), root3, Rational(1)))), evidence),
            ("an Add with a zero term", args, forged(Add((Rational(0), Mul((Rational(-1, 8), root3))))), evidence),
            ("-sqrt(3)/8 as a RealExpr", args, forged(RealExpr("mul", (scalar(Fraction(-1, 8)), functions.sqrt(
                scalar(3))))), evidence),
            ("a Rational subclass coefficient", args, forged(support.forged(Mul, factors=(Ratio(-1, 8), root3))),
             evidence),
            ("a coefficient not in lowest terms", args,
             forged(Mul((support.forged(Rational, numerator=-2, denominator=16), root3))), evidence),
            ("factors as a list", args, forged(support.forged(Mul, factors=[Rational(-1, 8), root3])), evidence),
            ("a radicand not in lowest terms", args,
             forged(Mul((Rational(-1, 8), support.forged(Sqrt, radicand=support.forged(Rational, numerator=6,
                                                                                       denominator=2))))), evidence),
            ("an impostor PiLinear", args, support.impostor(value), evidence),
            ("a PiLinear forged without fields", args, support.forged(family.PiLinear), evidence),
            ("a PiLinear forged without its pi coefficient", args,
             support.forged(family.PiLinear, algebraic=value.algebraic), evidence),
            ("a pair of its fields", args, (value.algebraic, value.pi_coefficient), evidence),
            ("a Rational", whole[1], Rational(1), whole[3]),
            ("the antiderivative as the value", args, antiderivative_value, evidence),
            # Evidence of the wrong shape.
            ("no evidence", args, value, ()),
            ("the antiderivative twice", args, value, (antiderivative_value, antiderivative_value)),
            ("an extra item", args, value, (antiderivative_value, ())),
            ("evidence as a list", args, value, [antiderivative_value]),
            ("evidence as a tuple subclass", args, value, Evidence((antiderivative_value,))),
            ("None as evidence", args, value, None),
            ("None as the antiderivative", args, value, (None,)),
            ("the antiderivative nested", args, value, ((antiderivative_value,),)),
            ("the bare expression", args, value, (canonical,)),
            ("a str", args, value, ("x/2 - sin(x)*cos(x)/2",)),
            # Evidence that is consistent with the value but wrong.
            ("the antiderivative of cos(x)**2 with its value", args,
             pi_linear(*integral({(0, 2): 1}, 1, 0, Fraction(1, 3))), (cos2_value,)),
            ("a constant term", args, value, (function(RealExpr("add", (canonical, scalar(7)))),)),
            ("a constant term hidden in S**2 + C**2", args, value, (function(RealExpr("add", (canonical, RealExpr(
                "add", (power(s, 2), power(c, 2)))))),)),
            ("a multiple angle", args, value, (text("x/2 - sin(2*x)/4"),)),
            ("the antiderivative for a = 2", {**args, "function": text("sin(2*x)^2")},
             pi_linear(*integral(SIN2, 2, 0, Fraction(1, 3))), (antiderivative_value,)),
            ("x**2 that cancels", args, value, (function(RealExpr("add", (canonical, RealExpr("add", (
                RealExpr("pow", (x,), 2), RealExpr("mul", (scalar(-1), RealExpr("pow", (x,), 2)))))))),)),
        ])

    def test_rejects_forged_trees_and_functions(self):
        """Nodes and functions forged without __init__, impostor classes, and nodes outside the admitted ones, in the
        antiderivative and in the evidence of the integral; most of them leave the function's value unchanged."""
        args = {"function": text("sin(x)^2"), "variable": X}
        _, integral_args, integral_value, _ = integral_claim(SIN2, 1, 0, Fraction(1, 3))
        canonical = sin2_tree()
        self.assertAccepts(ANTIDERIVATIVE, args, function(canonical))
        self.assertAccepts(INTEGRAL, integral_args, integral_value, (function(canonical),))
        x = leaf()
        s, c = sin_node(x), cos_node(x)
        product_node = RealExpr("mul", (s, c))
        two = scalar(2)
        replacements = [
            ((0, 1), "the divisor as an int", node("scalar", (), 2)),
            ((0, 1), "the divisor as a float", node("scalar", (), 2.0)),
            ((0, 1), "the divisor as a bool", node("scalar", (), True)),
            ((0, 1), "the divisor as a str", node("scalar", (), "2")),
            ((0, 1), "the divisor as a Fraction", node("scalar", (), Fraction(2))),
            ((0, 1), "the divisor as a Rational subclass", node("scalar", (), Ratio(2))),
            ((0, 1), "the divisor not in lowest terms", node("scalar", (), support.forged(Rational, numerator=4,
                                                                                            denominator=2))),
            ((0, 1), "the divisor with a negative denominator",
             node("scalar", (), support.forged(Rational, numerator=-2, denominator=-1))),
            ((0, 1), "the divisor forged without a denominator", node("scalar", (), support.forged(Rational,
                                                                                                   numerator=2))),
            ((0, 1), "a division by zero", node("scalar", (), Rational(0))),
            ((0, 1), "a division by the variable", leaf()),
            ((0, 1), "a division by 1 + 1", RealExpr("add", (scalar(1), scalar(1)))),
            ((0, 1), "a division by an algebraic 2", RealExpr("scalar", (), RealAlgebraicRoot((-4, 0, 1), 1))),
            ((0, 1), "a divisor with an operand", node("scalar", (two,), Rational(2))),
            ((0, 1), "an impostor divisor", support.impostor(two)),
            ((1, 0), "a coefficient as a Fraction", node("scalar", (), Fraction(-1, 2))),
            ((1, 0), "a coefficient as a float", node("scalar", (), -0.5)),
            ((1, 0), "a coefficient not in lowest terms", node("scalar", (), support.forged(Rational, numerator=-2,
                                                                                              denominator=4))),
            ((1, 0), "a coefficient with a negative denominator",
             node("scalar", (), support.forged(Rational, numerator=1, denominator=-2))),
            ((1, 0), "a coefficient that is None", node("scalar", (), None)),
            ((1, 0), "an algebraic -1/2", RealExpr("scalar", (), RealAlgebraicRoot((-1, 0, 4), 0))),
            ((1, 0), "an impostor coefficient", support.impostor(scalar(Fraction(-1, 2)))),
            ((0, 0), "the symbol of another variable", leaf(Y)),
            ((0, 0), "a symbol node holding a str", node("symbol", (), "x")),
            ((0, 0), "a symbol node with an operand", node("symbol", (x,), X)),
            ((0, 0), "a Symbol forged without a uid", node("symbol", (), support.forged(Symbol, name="x"))),
            ((0, 0), "an impostor symbol node", support.impostor(x)),
            ((1, 1, 0), "tan for sin", RealExpr("tan", (x,))),
            ((1, 1, 0), "sinh for sin", RealExpr("sinh", (x,))),
            ((1, 1, 0), "a sine of two operands", node("sin", (x, x))),
            ((1, 1, 0), "a sine with a value", node("sin", (x,), 1)),
            ((1, 1, 0), "a sine's operands as a list", node("sin", [x])),
            ((1, 1, 0), "an impostor sine", support.impostor(s)),
            ((1, 1, 0, 0), "a sine of x + 1", RealExpr("add", (x, scalar(1)))),
            ((1, 1, 0, 0), "a sine of x**2", RealExpr("pow", (x,), 2)),
            ((1, 1, 0, 0), "a sine of 2", scalar(2)),
            ((1, 1, 0, 0), "a sine of 0*x", RealExpr("mul", (scalar(0), x))),
            ((1, 1, 0, 0), "a sine of 2x", argument(2)),
            ((1, 1, 0, 0), "a sine of -x", argument(-1)),
            ((1, 1, 0, 0), "a sine of sin(x)", s),
            ((1, 1, 0, 0), "a sine of pi*x", RealExpr("mul", (functions.pi, x))),
            ((1, 1, 0, 0), "an impostor argument", support.impostor(x)),
            ((1, 1, 1), "a cosine of 3x", cos_node(argument(3))),
            ((1, 1), "a product of three operands", node("mul", (s, c, scalar(1)))),
            ((1, 1), "a product of one operand", node("mul", (product_node,))),
            ((1, 1), "a product's operands as a list", node("mul", [s, c])),
            ((1, 1), "a product with a value", node("mul", (s, c), 1)),
            ((1, 1), "a quotient by the cosine", node("div", (RealExpr("mul", (s, power(c, 2))), c))),
            ((1, 1), "a power -1 of its inverse", RealExpr("pow", (RealExpr("pow", (product_node,), -1),), -1)),
            ((1, 1), "a power with a bool exponent", node("pow", (product_node,), True)),
            ((1, 1), "a power with a float exponent", node("pow", (product_node,), 1.0)),
            ((1, 1), "an exponential of its logarithm", RealExpr("exp", (RealExpr("log", (product_node,)),))),
            ((1, 1), "a square root of its square", RealExpr("sqrt", (power(product_node, 2),))),
            ((1, 1), "an absolute value", RealExpr("abs", (product_node,))),
            ((1, 1), "pi", functions.pi),
            ((1, 1), "E", functions.E),
            ((1, 1), "an unknown operation", node("foo", (product_node,))),
            ((1, 1), "an op that is no str", node(b"mul", (s, c))),
            ((), "a sum with a value", node("add", canonical.args, 0)),
            ((), "an impostor root", support.impostor(canonical)),
        ]
        loop = RealExpr("add", (x, scalar(0)))
        object.__setattr__(loop, "args", (x, loop))  # A node that is its own descendant.
        trees = [(label, corrupted(canonical, path, replacement)) for path, label, replacement in replacements]
        trees.append(("a node that is its own descendant", RealExpr("add", (canonical, loop))))
        functions_forged = [
            ("variables as a list", forged_function(canonical, [X])),
            ("no variables", forged_function(canonical, ())),
            ("a str variable", forged_function(canonical, ("x",))),
            ("an impostor Symbol", forged_function(canonical, (support.impostor(X),))),
            ("the domain FALSE", forged_function(canonical, domain=functions.FALSE)),
            ("a str domain", forged_function(canonical, domain="true")),
            ("domain operands as a list", forged_function(canonical, domain=support.forged(
                DomainPredicate, kind="true", operands=[]))),
            ("a domain kind that is no str", forged_function(canonical, domain=support.forged(
                DomainPredicate, kind=True, operands=()))),
            ("an impostor domain", forged_function(canonical, domain=support.impostor(functions.TRUE))),
            ("no expression", support.forged(RealFunction, variables=(X,), domain=functions.TRUE)),
            ("a model expression", forged_function(add(Rational(1, 2), X))),
            ("an impostor RealFunction", support.impostor(function(canonical))),
            ("a tuple of its fields", (canonical, (X,), functions.TRUE)),
            ("None", None),
        ]
        cases = [(label, forged_function(tree)) for label, tree in trees] + functions_forged
        self.assertRejectsAll(ANTIDERIVATIVE, [(label, args, item, ()) for label, item in cases])
        self.assertRejectsAll(INTEGRAL, [(label, integral_args, integral_value, (item,)) for label, item in cases])
        # The same forgeries in the request: no request outside the input class has a value.
        true_value = function(canonical)
        self.assertRejectsAll(ANTIDERIVATIVE, [(label, {"function": item, "variable": X}, true_value, ())
                                               for label, item in cases])
        # A function forged without its domain reads the dataclass default TRUE, the class attribute, as every other
        # reader of it does (its JSON text names the domain TRUE): it is the canonical function, and passes.
        unset = support.forged(RealFunction, expression=canonical, variables=(X,))
        self.assertNotIn("domain", vars(unset))
        self.assertIs(unset.domain, functions.TRUE)
        self.assertAccepts(ANTIDERIVATIVE, args, unset)
        self.assertAccepts(INTEGRAL, integral_args, integral_value, (unset,))

    def test_rejects_inexact_arguments(self):
        operation, args, value, evidence = integral_claim(SIN2, 1, 0, Fraction(1, 3))
        cases = [(f"lower_pi {label}", dict(args, lower_pi=item), value, evidence)
                 for label, item in rational_stand_ins(0)]
        cases += [(f"upper_pi {label}", dict(args, upper_pi=item), value, evidence)
                  for label, item in rational_stand_ins(Fraction(1, 3))]
        for label, item in [("a str", "x"), ("None", None), ("an impostor Symbol", support.impostor(X)),
                            ("a Symbol forged without fields", support.forged(Symbol)),
                            ("a Symbol forged with a str domain",
                             support.forged(Symbol, name="x", domain="REALS", assumptions=(), uid=X.uid)),
                            ("the symbol node", leaf()), ("another Symbol", Y)]:
            cases.append((f"variable {label}", dict(args, variable=item), value, evidence))
        f = args["function"]
        for label, item in [("the expression", f.expression), ("a str", "sin(x)^2"), ("None", None),
                            ("an impostor", support.impostor(f)),
                            ("forged without fields", support.forged(RealFunction)),
                            ("a Rational", Rational(1)), ("a Polynomial", Polynomial((Rational(1),), X))]:
            cases.append((f"function {label}", dict(args, function=item), value, evidence))
        self.assertRejectsAll(operation, cases)
        antiderivative_args = {"function": f, "variable": X}
        self.assertRejectsAll(ANTIDERIVATIVE, [(label, {**antiderivative_args, **{key: changed[key] for key in (
            "function", "variable")}}, evidence[0], ()) for label, changed, _, _ in cases
            if label.startswith(("variable", "function"))])

    def test_rejects_wrong_operations_and_arguments(self):
        _, args, value, evidence = integral_claim(SIN2, 1, 0, Fraction(1, 3))
        antiderivative_args = {"function": args["function"], "variable": X}
        self.assertAccepts(INTEGRAL, args, value, evidence)
        self.assertAccepts(ANTIDERIVATIVE, antiderivative_args, evidence[0])
        self.assertRejectsAll(INTEGRAL, [
            ("arguments as pairs", tuple(args.items()), value, evidence),
            ("arguments as a list of pairs", list(args.items()), value, evidence),
            ("arguments as a read-only mapping", MappingProxyType(args), value, evidence),
            ("None as arguments", None, value, evidence),
            ("an argument missing", {key: args[key] for key in ("function", "variable", "lower_pi")}, value, evidence),
            ("an extra argument", dict(args, order=3), value, evidence),
            ("an argument renamed", {**antiderivative_args, "lower": 0, "upper_pi": args["upper_pi"]}, value, evidence),
            ("the antiderivative's arguments", antiderivative_args, value, evidence),
        ])
        for operation in ("trig_power", "Trig_power_integral", "trig_power_integral ", "integral",
                          "continued_fraction", "", None, 7, (INTEGRAL,), b"trig_power_integral", [INTEGRAL]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value, evidence)
        # True claims of one operation under the name of the other, or with the other's value.
        self.assertRejects(ANTIDERIVATIVE, args, value, evidence)
        self.assertRejects(ANTIDERIVATIVE, args, evidence[0], ())
        self.assertRejects(INTEGRAL, antiderivative_args, evidence[0], ())
        self.assertRejects(ANTIDERIVATIVE, antiderivative_args, value, ())
        self.assertRejects(INTEGRAL, args, evidence[0], evidence)

    def test_oversized_forgeries_fail_fast(self):
        args = {"function": text("sin(x)^2"), "variable": X}
        _, integral_args, value, evidence = integral_claim(SIN2, 1, 0, Fraction(1, 3))
        canonical = sin2_tree()
        x = leaf()
        s, c = sin_node(x), cos_node(x)
        zero = scalar(0)
        chain = zero
        for _ in range(120_000):  # A left-deep chain of distinct nodes, longer than max_nodes.
            chain = node("add", (chain, zero))
        wide = balanced([node("add", (zero, zero)) for _ in range(50_000)])  # A balanced tree of distinct nodes.
        operand = RealExpr("add", (scalar(1), RealExpr("add", (s, c))))  # 1 + S + C, shared by every product.
        products = balanced([RealExpr("mul", (operand, operand)) for _ in range(5_000)])
        doubling, squaring = x, x
        for _ in range(300):  # 300 distinct nodes, each used twice: a tree of 2**300 leaves.
            doubling, squaring = RealExpr("add", (doubling, doubling)), RealExpr("mul", (squaring, squaring))
        long_argument = node("add", (chain, x))
        sines = balanced([sin_node(x) for _ in range(30_000)])
        loop = RealExpr("add", (x, zero))
        object.__setattr__(loop, "args", (x, loop))
        unchanged = RealExpr("mul", (scalar(LONG), RealExpr("add", (RealExpr("add", (power(s, 2), power(c, 2))),
                                                                     scalar(-1)))))  # LONG (S**2 + C**2 - 1) = 0
        trees = [
            ("a chain of 120,000 nodes", chain),
            ("50,000 distinct nodes", RealExpr("add", (canonical, wide))),
            ("a giant scalar", RealExpr("add", (canonical, scalar(GIANT)))),
            ("a scalar of 1/GIANT", RealExpr("add", (canonical, scalar(Fraction(1, GIANT))))),
            ("a long scalar", RealExpr("add", (canonical, scalar(LONG)))),
            ("a long multiple of zero", RealExpr("add", (canonical, unchanged))),
            ("3**40000", RealExpr("add", (canonical, RealExpr("pow", (scalar(3),), 40_000)))),
            ("3**(10**9)", RealExpr("add", (canonical, RealExpr("pow", (scalar(3),), 10 ** 9)))),
            ("sin(x)**(10**9)", RealExpr("add", (canonical, RealExpr("pow", (s,), 10 ** 9)))),
            ("x**(10**9)", RealExpr("add", (canonical, RealExpr("pow", (x,), 10 ** 9)))),
            ("1**(10**18)", RealExpr("add", (canonical, RealExpr("pow", (scalar(1),), 10 ** 18)))),
            ("5,000 products", RealExpr("add", (canonical, products))),
            ("a sum of 2**300 x", doubling),
            ("a product of 2**300 x", squaring),
            ("a sine of a chain", sin_node(long_argument)),
            ("a sine of x**(10**9)", RealExpr("add", (canonical, sin_node(RealExpr("pow", (x,), 10 ** 9))))),
            ("a sine of (x + 1)**100000 - x**100000 + ...", RealExpr("add", (canonical, sin_node(RealExpr("add", (
                RealExpr("pow", (RealExpr("add", (x, scalar(1))),), 100_000),
                RealExpr("mul", (scalar(-1), RealExpr("pow", (x,), 100_000))))))))),
            ("30,000 distinct sines", RealExpr("add", (canonical, sines))),
            ("a node that is its own descendant", RealExpr("add", (canonical, loop))),
        ]
        cases = [(ANTIDERIVATIVE, args, forged_function(tree), ()) for _, tree in trees]
        cases += [(INTEGRAL, integral_args, value, (forged_function(tree),)) for _, tree in trees]
        cases += [
            (ANTIDERIVATIVE, args, function(canonical), (0,) * 100_000),
            (ANTIDERIVATIVE, args, function(canonical), (function(canonical),) * 100_000),
            (INTEGRAL, integral_args, value, evidence * 100_000),
            (INTEGRAL, integral_args, value, evidence + (GIANT,)),
            (INTEGRAL, integral_args, support.forged(family.PiLinear, algebraic=value.algebraic,
                                                     pi_coefficient=Rational(GIANT)), evidence),
            (INTEGRAL, integral_args, support.forged(family.PiLinear, algebraic=value.algebraic,
                                                     pi_coefficient=Rational(1, GIANT)), evidence),
            (INTEGRAL, integral_args, support.forged(family.PiLinear, algebraic=value.algebraic,
                                                     pi_coefficient=Rational(LONG, 3)), evidence),
            (INTEGRAL, integral_args, support.forged(family.PiLinear, algebraic=Mul((Rational(GIANT), Sqrt(
                Rational(3)))), pi_coefficient=value.pi_coefficient), evidence),
            (INTEGRAL, integral_args, support.forged(family.PiLinear, algebraic=support.forged(
                Add, terms=(Rational(1),) * 100_000), pi_coefficient=value.pi_coefficient), evidence),
            (INTEGRAL, integral_args, support.forged(family.PiLinear, algebraic=Rational(LONG),
                                                     pi_coefficient=value.pi_coefficient), evidence),
        ]
        labels = [label for label, _ in trees] * 2 + ["long evidence"] * 3 + ["a giant item"] + ["PiLinear"] * 6
        for (operation, case_args, case_value, case_evidence), label in zip(cases, labels):
            with self.subTest(operation=operation, case=label):
                start = time.perf_counter()
                self.assertRejects(operation, case_args, case_value, case_evidence)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)

    def test_resource_limits(self):
        """Where the request itself is beyond the limits, the checker raises ResourceLimitError, fast, as the producer's
        budget refuses it too: integers over max_integer_bits in the function or the bounds, a degree bound over
        max_work, a function too large to expand, more than max_nodes distinct nodes."""
        x = leaf()
        s = sin_node(x)
        value = text("x/2 - sin(x)*cos(x)/2")
        _, integral_args, integral_value, evidence = integral_claim(SIN2, 1, 0, Fraction(1, 3))
        zero = scalar(0)
        chain = s
        for _ in range(110_000):
            chain = node("add", (chain, zero))
        functions_beyond = [
            ("a giant coefficient", function(RealExpr("mul", (scalar(GIANT), power(s, 2))))),
            ("a coefficient of 1/GIANT", function(RealExpr("mul", (scalar(Fraction(1, GIANT)), power(s, 2))))),
            ("a giant scalar in the argument", function(sin_node(RealExpr("mul", (scalar(GIANT), x))))),
            ("sin(x)**(10**9)", function(RealExpr("pow", (s,), 10 ** 9))),
            ("sin(x)**(2**16)", function(RealExpr("pow", (s,), 2 ** 16))),
            ("(3 + 2 sin(x))**40000", function(RealExpr("pow", (RealExpr("add", (scalar(3), RealExpr(
                "mul", (scalar(2), s)))),), 40_000))),
            ("110,000 distinct nodes", forged_function(chain)),
            ("a sine of x**(10**9)", function(power(sin_node(RealExpr("pow", (x,), 10 ** 9)), 2))),
            ("a sine of x**(3*10**6) - x**(3*10**6) + x", function(power(sin_node(RealExpr("add", (RealExpr("add", (
                RealExpr("pow", (x,), 3 * 10 ** 6), RealExpr("mul", (scalar(-1), RealExpr("pow", (x,), 3 * 10 ** 6))))),
                x))), 2))),
        ]
        cases = [(ANTIDERIVATIVE, {"function": f, "variable": X}, value, ()) for _, f in functions_beyond]
        cases += [(INTEGRAL, dict(integral_args, lower_pi=GIANT), integral_value, evidence),
                  (INTEGRAL, dict(integral_args, upper_pi=Rational(1, GIANT)), integral_value, evidence),
                  (INTEGRAL, dict(integral_args, lower_pi=Rational(GIANT, 3)), integral_value, evidence),
                  (INTEGRAL, dict(integral_args, function=functions_beyond[0][1]), integral_value, evidence)]
        for index, (operation, case_args, case_value, case_evidence) in enumerate(cases):
            with self.subTest(case=index, operation=operation):
                start = time.perf_counter()
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, case_args, case_value, case_evidence)
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # Work and budget

    def test_work_is_charged_before_it_runs(self):
        """The charges depend on the claim alone: a budget one unit short of a check refuses it, its exact size
        passes, for the reading of both trees and for the evaluation in Q(sqrt 2, sqrt 3)."""
        claims = [antiderivative_claim({(0, 12): 1, (3, 1): Fraction(2, 3)}, Fraction(3, 2)),
                  antiderivative_claim({(1, 1): Fraction(LONG, 3)}, 1),
                  integral_claim({(5, 4): 1, (0, 3): -2}, 1, Fraction(1, 12), Fraction(7, 4)),
                  integral_claim({(0, 40): 1}, 1, 0, Fraction(1, 12))]
        for operation, args, value, evidence in claims:
            with self.subTest(operation=operation):
                with computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, value, evidence), True)
                work = budget.work
                self.assertGreater(work, 10)
                with self.assertRaises(ResourceLimitError):
                    support.verdict(family, operation, args, value, evidence,
                                    limits=ComputationLimits(max_work=work - 1))
                self.assertAccepts(operation, args, value, evidence, limits=ComputationLimits(max_work=work))

    def test_the_allowance_bounds_the_work_on_any_claim(self):
        """Reading a claim costs at most the allowance that its request fixes: an expensive forgery within max_nodes
        and max_integer_bits is False, after no more work than the request's and the allowance; without the
        allowance the same forgery would exhaust the budget. The canonical antiderivative, written as a sum of
        monomials or in Horner's form, costs a small fraction of the allowance."""
        rng = random.Random(20261115)
        for poly, a in [(SIN2, 1), ({(0, 20): 1}, 2), ({(7, 4): 1, (0, 0): 3}, Fraction(-3, 4)),
                        (random_poly(rng, 10, terms=8), Fraction(5, 3)), ({(2, 1): Fraction(3 ** 2000, 7 ** 900)}, 1)]:
            operation, args, value, _ = antiderivative_claim(poly, a)
            with computation(fresh=True) as budget:
                source = checker._source(budget, args)
            request_work = budget.work
            u = argument(a)
            nf = antiderivative(poly, a)
            for label, tree in [("monomials", value.expression), ("divisions", nf_tree(nf, u, "div")),
                                ("Horner", horner_tree(nf, u))]:
                with self.subTest(poly=poly, tree=label), computation(fresh=True) as budget:
                    self.assertIs(checker.check_operation(operation, args, function(tree), ()), True)
                    self.assertLess(budget.work - request_work, source.allowance // 10)
        # An expensive forgery: 10,000 distinct products of a shared polynomial of degree 10 for sin(x)**20, each
        # within the degree bound 21, which would cost about 3,000,000 units.
        args = {"function": function(power(sin_node(leaf()), 20)), "variable": X}
        operand = balanced([scalar(1), sin_node(leaf())] + [power(cos_node(leaf()), k) for k in range(1, 11)])
        forgery = forged_function(balanced([RealExpr("mul", (operand, operand)) for _ in range(10_000)]))
        with computation(fresh=True) as budget:
            source = checker._source(budget, args)
        request_work = budget.work
        start = time.perf_counter()
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation(ANTIDERIVATIVE, args, forgery, ()), False)
        self.assertLessEqual(budget.work, request_work + source.allowance + 64)
        self.assertLess(time.perf_counter() - start, FAST)
        unlimited = checker._source

        def without_allowance(budget, source_args):
            return unlimited(budget, source_args)._replace(allowance=10 ** 15)

        with patch.object(checker, "_source", without_allowance), self.assertRaises(ResourceLimitError):
            support.verdict(family, ANTIDERIVATIVE, args, forgery)

    def test_full_size_claims_fit_the_default_budget(self):
        """Large true claims verify under the default budget: degree 200, 303 and big coefficients and frequencies."""
        claims = [
            ("cos(x)**200", antiderivative_claim({(0, 200): 1}, 1)),
            ("sin(x)**301 cos(x)**2", antiderivative_claim({(301, 2): 1}, 1)),
            ("cos(x)**200 on [pi/12, 7 pi/12]", integral_claim({(0, 200): 1}, 1, Fraction(1, 12), Fraction(7, 12))),
            ("a coefficient of 20,000 bits", integral_claim({(2, 1): Fraction(2 ** 20_000 + 1, 3), (0, 3): -1}, 3,
                                                            Fraction(-1, 36), Fraction(5, 36))),
            ("a frequency of 6,000 bits", antiderivative_claim({(1, 3): 5, (2, 0): 1},
                                                              Fraction(2 ** 3000 + 1, 3 ** 1500))),
        ]
        for label, claim in claims:
            with self.subTest(claim=label):
                start = time.perf_counter()
                self.assertAccepts(*claim)
                self.assertLess(time.perf_counter() - start, 3 * FAST)

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_the_checker_forms_no_power_and_imports_no_producer(self):
        tree = ast.parse(inspect.getsource(checker))
        called = {node_.func.id for node_ in ast.walk(tree)
                  if isinstance(node_, ast.Call) and isinstance(node_.func, ast.Name)}
        self.assertNotIn("pow", called)
        powers = [node_ for node_ in ast.walk(tree) if isinstance(node_, (ast.BinOp, ast.AugAssign))
                  and isinstance(node_.op, ast.Pow)]
        self.assertEqual(powers, [])  # Every power of a polynomial is a chain of charged products.
        imported = set()
        for node_ in ast.walk(tree):
            if isinstance(node_, ast.Import):
                imported.update(alias.name for alias in node_.names)
            elif isinstance(node_, ast.ImportFrom):
                imported.add(node_.module)
        self.assertLessEqual(imported, {"__future__", "collections.abc", "math", "typing", "mathforge.errors",
                                        "mathforge.limits", "mathforge.model", "mathforge.trusted.surd"})

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip: the evidence layouts
        survive the format, and a wrong value fails the mathematics check."""
        sin2 = {"function": text("sin(x)^2"), "variable": X}
        claims = [
            ((ANTIDERIVATIVE, sin2, text("x/2 - sin(x)*cos(x)/2"), ()), text("x/2 + sin(x)*cos(x)/2")),
            (antiderivative_claim({(1, 2): 1}, 2), text("-cos(2*x)^3/3")),
            (antiderivative_claim({(0, 0): Fraction(-2, 7)}, 1), text("2*x/7")),
            (integral_claim(SIN2, 1, 0, Fraction(1, 3)), pi_linear((0, 0, Fraction(1, 8), 0), Fraction(1, 6))),
            (integral_claim({(3, 2): 1, (0, 1): 2, (1, 1): Fraction(-1, 3)}, 1, Fraction(1, 12), Fraction(3, 4)),
             pi_linear(E0, 0)),
            (integral_claim({(0, 0): 3}, 1, Fraction(1, 7), Fraction(2, 9)), pi_linear(E0, Fraction(1, 7))),
        ]
        for (operation, args, value, evidence), wrong in claims:
            with self.subTest(operation=operation, value=mf.render_text(value)):
                self.roundtrip(claimed(operation, args, value, evidence))
                report = support.statuses(mf.verify(claimed(operation, args, wrong, evidence)))
                self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
