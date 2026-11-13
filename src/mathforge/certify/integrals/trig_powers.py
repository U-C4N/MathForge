"""Independent checker of the family mathforge.areas.integrals.trig_powers: trig_power_antiderivative and
trig_power_integral.

It never integrates. It reads the request's function f and the claimed antiderivative F as polynomials, differentiates
F formally and compares; for the definite integral it evaluates F at the bounds with its own table of the sines and
cosines of the multiples of pi/12 in the field Q(sqrt 2, sqrt 3) (mathforge.trusted.surd). The trees of
mathforge.functions (RealFunction, RealExpr, DomainPredicate) and the family's PiLinear record are read as data: their
classes are identified by (type(value).__module__, type(value).__qualname__), and no producer module is imported.

Reading a tree. With S = sin(a*x) and C = cos(a*x), a tree of the admitted nodes denotes a polynomial in (x, S, C) over
Q. The admitted nodes are add and mul of two operands; div whose divisor is a scalar node holding a nonzero Rational;
pow with an exact int exponent >= 0; scalar nodes holding an exact Rational (two exact ints in lowest terms, the
denominator positive); the symbol node of the variable; and sin(u) and cos(u), whose argument u, read the same way
without sin and cos, is the polynomial a*x for a nonzero rational a, the same a for every sine and cosine of f and of
F (the trees of the arguments may differ, whatever their intermediate degrees: sin(2*x), cos(x + x) and
sin(x*x - x*x + 2*x) share a = 2, while sin(x) and sin(-x) do not). Every node of a tree is read, also below a power
0 or a factor 0: sin(2*x)**0 has the argument 2*x, and x**0 holds the variable. Any other node (tan, exp, sqrt, abs,
pi, E, a RealAlgebraicRoot scalar, a negative power, a division by anything but a scalar node holding a nonzero
Rational, a sine of x + 1, of x**2 or of a constant) answers False. While reading, S**2 becomes 1 - C**2, so that the
result is the reduced normal form N0(x, C) + S*N1(x, C): the coefficients of the monomials x**i S**j C**k with
j in {0, 1}.

Lemma (normal form). Let phi map a polynomial Q(x, S, C) to the function t -> Q(t, sin(a t), cos(a t)) on R. phi is a
ring homomorphism, and phi(D Q) = phi(Q)' for the derivation D of Q[x, S, C] with D x = 1, D S = a C, D C = -a S (both
sides are derivations that agree on x, S and C). Its kernel holds S**2 + C**2 - 1, so a polynomial and its reduced
normal form have the same image. The images of the reduced monomials, t**i cos(a t)**k and t**i sin(a t) cos(a t)**k,
are linearly independent over Q: cos(a t)**k is a combination of the cos(m a t), m <= k, with the coefficient 2**(1-k)
at m = k >= 1, sin(a t) cos(a t)**k one of the sin(m a t), 1 <= m <= k + 1, with the coefficient 2**-k at m = k + 1,
and the functions t**i, t**i cos(m a t) and t**i sin(m a t) (m >= 1) are linearly independent for a != 0. So phi(Q) = 0
exactly when the reduced normal form of Q is zero: the zero tests below are sound and complete.

trig_power_antiderivative(function, variable) = F, evidence ().
    The request: variable is a real Symbol without assumptions (an exact str name and uid, the domain Reals), and
    function is a RealFunction f with the variables (variable,) and the domain TRUE whose expression reads as a
    polynomial P in S and C alone (no symbol node of the variable outside the arguments of its sines and cosines), with
    one a or none (a tree without sin and cos is a rational constant). A Symbol is the variable when its name and uid
    are the variable's, compared as exact strs, with the domain Reals and no assumptions: no __eq__ of the claim runs.
    Any other request has no value, and every claim on it is False. True requires F to be a RealFunction with the
    variables (variable,) and the domain TRUE whose expression reads as a polynomial Q_F with the same a, such that
    (1) the reduced normal form of Q_F has no constant term (no monomial x**0 S**0 C**0) and (2) the normal form of
    D Q_F - P is zero. By the lemma, F' = phi(Q_F)' = phi(D Q_F) = phi(P) = f on all of R. Since f does not involve the
    variable, F is lambda*x + T(S, C): the highest power x**m of the normal form of F with m >= 2 would leave
    m x**(m-1) times a periodic function in F'. The value is unique: two such F differ by a function with derivative
    zero, a constant c, whose normal form is c itself, and (1) makes c = 0. "No constant term" is read on the reduced
    normal form, the only reading that makes the value unique: sin(x)**2/2 and -cos(x)**2/2 are both antiderivatives
    of sin(x)*cos(x) without a constant term as polynomials in S and C, but only -cos(x)**2/2 has a reduced normal
    form without one (sin(x)**2/2 reduces to 1/2 - C**2/2). Any tree of the admitted nodes that denotes the canonical
    F passes, whatever its shape, within the bounds below; a constant term, a multiple angle (sin(2*x) when f is a
    polynomial in sin(x) and cos(x)), another domain, other variables and every node outside the admitted ones fail.

trig_power_integral(function, variable, lower_pi, upper_pi) = PiLinear(algebraic, pi_coefficient), evidence (F,).
    The request as above, with exact rationals lower_pi and upper_pi (each an int or a Rational, in either order). True
    requires F to pass the antiderivative check against f, with its normal form lambda*x + T(S, C), and the value to be
    what F gives at the bounds: pi_coefficient == lambda*(upper_pi - lower_pi), and algebraic the canonical expression
    of T(upper_pi*pi) - T(lower_pi*pi). When T = 0, that is 0 for any bounds. Otherwise f has a sine or a cosine, its a
    is F's, 12*a*lower_pi and 12*a*upper_pi must be integers (else the claim is False: the table cannot evaluate T),
    and T at t*pi is N0(C) + S*N1(C) for S = sine(k), C = cosine(k) and the integer k = 12*a*t. By the fundamental
    theorem of calculus (F' = f is continuous on R), the integral of f from lower_pi*pi to upper_pi*pi is
    F(upper_pi*pi) - F(lower_pi*pi) = lambda*(upper_pi - lower_pi)*pi + T(upper_pi*pi) - T(lower_pi*pi), and
    a*t*pi = k*pi/12, where sine(k) and cosine(k) are sin and cos exactly. The value is the only PiLinear of that
    number: 1, sqrt 2, sqrt 3 and sqrt 6 are linearly independent over Q, so the algebraic part has one set of
    coefficients and one canonical expression, and pi is transcendental, so no element of Q(sqrt 2, sqrt 3) is a nonzero
    rational multiple of pi, and the split into algebraic + pi_coefficient*pi is unique.
    T = 0 exactly when P is a constant c: then phi(T)' = c - lambda, so the bounded function phi(T) is (c - lambda)*t
    plus a constant, which forces lambda = c and phi(T) constant, and the lemma makes T, whose normal form has no
    constant term, zero; T = 0 gives P = lambda. So the bounds of a request whose P is a constant are free: a rational
    constant f, and also an f whose sines cancel, such as sin(x)**0 or sin(x) - sin(x) + 1. The producer follows the
    card and refuses every request whose tree has a sine or a cosine and whose bounds are no multiples of pi/(12*a),
    even when the sines cancel; the checker's domain contains the producer's, and its verdict on the extra requests is
    still the truth.

The table. sine(k) = sin(k*pi/12) for every integer k, from the first quadrant: sin(0) = 0, sin(pi/12) =
(sqrt 6 - sqrt 2)/4 and sin(5*pi/12) = cos(pi/12) = (sqrt 6 + sqrt 2)/4 (the difference formulas at pi/3 - pi/4),
sin(pi/6) = 1/2, sin(pi/4) = sqrt(2)/2, sin(pi/3) = sqrt(3)/2 and sin(pi/2) = 1; then sin(pi - t) = sin(t) and
sin(t + pi) = -sin(t), and cosine(k) = sine(k + 6). The tests check that the 24 values satisfy s**2 + c**2 = 1 and the
addition formulas, which make k -> c_k + i s_k a homomorphism from Z/24 to the unit circle, with s_6 = 1, so that
c_1 + i s_1 is one of the six 24th roots of unity z with z**6 = i, and s_k > 0 for 0 < k < 12, which leaves only
exp(i pi/12).

The canonical PiLinear. pi_coefficient is an exact Rational in lowest terms (an int is no Rational here); algebraic is
c0 + c1 sqrt(2) + c2 sqrt(3) + c3 sqrt(6) exactly as mathforge.model builds it with add(c0, mul(c1, sqrt(2)),
mul(c2, sqrt(3)), mul(c3, sqrt(6))): a Rational when c1 = c2 = c3 = 0 (Rational(0) for zero), else a Sqrt, a Mul of a
Rational and a Sqrt, or an Add of such terms in the order of mathforge.model. The checker builds that expression from
its own coefficients and compares it with the claim node by node, with exact types, so a right number in another shape
(sqrt(2)*sqrt(3) for sqrt(6), sqrt(12)/2 for sqrt(3), an unsorted Add, an int for a Rational, 1*sqrt(2)) is False.

Sizes and budget. Every tree is walked without recursion, each distinct node once (a node shared by several parents
is read once), for a work unit each; a node that is its own descendant (only object.__setattr__ can make one) is
False. Each coefficient operation is charged before it runs: one unit per operation of a polynomial sum, product,
derivative or comparison, in bulk before its loop, and for long integers e(bits) more, one unit per sixteen squared
64-bit words of the longest operand (its products and its gcd); each step of the evaluation in the field the same way.
No builtin pow and no ** runs: powers are repeated products.

The claim's outer shape (the evidence tuple and its length; for the integral also the value's record class and its pi
coefficient) is tested first, for a few work units; then the request is read, and its own size is never refuted: a
function with more than max_nodes distinct nodes, whose degree bound exceeds max_work (also inside the argument of a
sine or a cosine), or whose integers outgrow max_integer_bits while it is read, and bounds longer than
max_integer_bits, raise ResourceLimitError, as the producer's budget refuses them. The rest of the claim is then
screened against the request before and while it is read, and a claim beyond these bounds is False at once:
- at most max_nodes distinct nodes in F, every integer of a scalar and every coefficient met while reading F within
  max_integer_bits, the evidence of the exact length;
- the degree of every subtree of F outside the arguments of its sines and cosines, bounded from below upwards (add: the
  larger, mul: the sum, div: the dividend's, pow k: k times its operand's), at most 1 in the variable and at most
  D + 1 in x, S and C together, where D is that bound for f's tree, at least the total degree d of the reduced normal
  form P of f (the canonical F has degree at most max(d, 1)); inside an argument, at most max_work, as in f;
- the allowance: reading F (its walk, its arithmetic, its arguments and its derivative) may cost at most
  A = 2*W + 4096 + (2*d + 2)*(512 + 32*e(B + 64)) work units, where W is the work that reading f took and
  B = n + s + b + (d // 2 + 1)*bits(d + 1) + bits(d + 2) + 1, with n the longest numerator of P's coefficients, s the
  sum of the bit lengths of their denominators and b the bit lengths of a's numerator and denominator together.
B bounds every numerator and denominator of the canonical F's coefficients: with L the lcm of P's denominators (at
most s bits), F has at most 2*d + 1 monomials; lambda is sum over even j of p_j (j-1)!!/j!!, the coefficient of
S*C**k is (1/a) sum over j of p_j times a quotient of a product of at most d // 2 integers up to d by a product of at
most d // 2 + 1 of them, the same denominator product for every j, and the coefficient of C**(j+1) is
-q_j/((j + 1) a): over the common denominator (a's numerator) * L * (that product), each is a sum of at most d + 1
terms of at most n + s + b + (d // 2)*bits(d + 1) bits. The canonical F written as a sum of monomials (each a scalar
times x, sin(u) and a power of cos(u), u a small tree of a*x) costs about 50 units and a handful of coefficient
operations per monomial, a small fraction of A; Horner's form in C stays within it as well. A claim within these
bounds that the budget cannot pay for raises ResourceLimitError, and so does a budget too small for a true claim: such
a claim is beyond the budget, unverified, never refuted.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
from typing import NamedTuple

from mathforge.errors import InvalidInput
from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Add, Mul, Rational, Reals, Sqrt, Symbol, add, mul, sqrt, trusted_rational
from mathforge.trusted.surd import Field

# The argument names of each operation, in the order of its spec.
OPERATIONS: dict[str, tuple[str, ...]] = {
    "trig_power_antiderivative": ("function", "variable"),
    "trig_power_integral": ("function", "variable", "lower_pi", "upper_pi"),
}
RECORD = ("mathforge.areas.integrals.trig_powers", "PiLinear")  # Module and name of trig_power_integral's value.
_REAL_EXPR = ("mathforge.functions", "RealExpr")
_REAL_FUNCTION = ("mathforge.functions", "RealFunction")
_DOMAIN = ("mathforge.functions", "DomainPredicate")
_MISSING = object()  # A field that a record made with object.__new__ lacks.

# A polynomial is a dict {(i, j, k): (numerator, denominator)} of the monomials x**i S**j C**k, j in {0, 1}, with
# nonzero coefficients in lowest terms and positive denominators.
_ONE = (1, 1)
_CONSTANT, _LINEAR, _SINE, _COSINE = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)

# Q(sqrt 2, sqrt 3): an element is (c0, c1, c2, c3), the coefficients of 1, sqrt 2, sqrt 3 and sqrt 6.
FIELD = Field((2, 3))


def _element(*pairs: tuple[int, int]) -> tuple[Rational, ...]:
    return tuple(trusted_rational(numerator, denominator) for numerator, denominator in pairs)


# sin(k*pi/12) for k = 0, ..., 6 (the module docstring derives them).
_QUADRANT = (
    _element((0, 1), (0, 1), (0, 1), (0, 1)),   # 0
    _element((0, 1), (-1, 4), (0, 1), (1, 4)),  # (sqrt 6 - sqrt 2)/4
    _element((1, 2), (0, 1), (0, 1), (0, 1)),   # 1/2
    _element((0, 1), (1, 2), (0, 1), (0, 1)),   # sqrt(2)/2
    _element((0, 1), (0, 1), (1, 2), (0, 1)),   # sqrt(3)/2
    _element((0, 1), (1, 4), (0, 1), (1, 4)),   # (sqrt 6 + sqrt 2)/4
    _element((1, 1), (0, 1), (0, 1), (0, 1)),   # 1
)


def sine(k: int) -> tuple[Rational, ...]:
    """sin(k*pi/12) as an element of FIELD, for any int k: sin(pi - t) = sin(t) and sin(t + pi) = -sin(t)."""
    k %= 24
    value = _QUADRANT[min(k % 12, 12 - k % 12)]
    return tuple(-c for c in value) if k >= 12 else value


def cosine(k: int) -> tuple[Rational, ...]:
    """cos(k*pi/12) = sin((k + 6)*pi/12) as an element of FIELD."""
    return sine(k + 6)


class _Reject(Exception):
    """The claim is malformed, oversized or about a request outside the operation's domain: the answer is False."""


def _named(value: object) -> tuple[str, str]:
    kind = type(value)
    return kind.__module__, kind.__qualname__


def _is_variable(candidate: object, variable: Symbol) -> bool:
    """Whether candidate is the variable: a Symbol with the variable's name and uid as exact strs, the domain Reals and
    no assumptions. The fields are compared one by one, so that no __eq__ of a forged field runs; for Symbols made by
    their constructor this is Symbol equality, since the variable has the domain Reals and no assumptions."""
    if type(candidate) is not Symbol:
        return False
    name, uid = getattr(candidate, "name", _MISSING), getattr(candidate, "uid", _MISSING)
    assumptions = getattr(candidate, "assumptions", _MISSING)
    return (type(name) is str and type(uid) is str and name == variable.name and uid == variable.uid
            and getattr(candidate, "domain", _MISSING) is Reals and type(assumptions) is tuple and not assumptions)


def _extra(bits: int) -> int:
    """e(bits): what one exact rational operation on integers of the given bit length costs beyond one work unit, its
    products and its gcd, quadratic in the 64-bit words (nothing below 256 bits, 65,536 units at 65,536 bits)."""
    words = bits >> 6
    return words * words >> 4


def _rational(tick: Callable[[int], None], max_bits: int, value: object,
              oversized: type[Exception]) -> tuple[int, int]:
    """(numerator, denominator) of an exact Rational in canonical form: two exact ints in lowest terms with a positive
    denominator (a Rational made with object.__new__ may break that form or lack a field). _Reject for anything else;
    oversized (_Reject for the claim, ResourceLimitError for the request) for an integer over max_bits, before its gcd
    is charged."""
    if type(value) is not Rational:
        raise _Reject("not an exact Rational")
    numerator, denominator = getattr(value, "numerator", _MISSING), getattr(value, "denominator", _MISSING)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise _Reject("a malformed Rational")
    bits = max(numerator.bit_length(), denominator.bit_length())
    if bits > max_bits:
        raise oversized("Computation limit exceeded: integer bits")
    tick(1 + _extra(bits))  # The gcd.
    if gcd(numerator, denominator) != 1:
        raise _Reject("a Rational not in lowest terms")
    return numerator, denominator


class _Reader:
    """Reads trees of RealExpr nodes as reduced polynomials (the module docstring lists the admitted nodes) for one
    variable, with the frequency a that every sine and cosine shares. A reader of the claim answers an oversized tree
    with _Reject (False), and so does every charge past its allowance; a reader of the request's function answers an
    oversized tree with ResourceLimitError and has no allowance but the budget."""

    __slots__ = ("budget", "variable", "oversized", "max_bits", "max_nodes", "nodes", "frequency", "arguments",
                 "allowance")

    def __init__(self, budget: Budget, variable: Symbol, claim: bool, frequency: tuple[int, int] | None = None,
                 allowance: int | None = None):
        self.budget = budget
        self.variable = variable
        self.oversized: type[Exception] = _Reject if claim else ResourceLimitError
        self.max_bits = budget.limits.max_integer_bits
        self.max_nodes = budget.limits.max_nodes
        self.nodes = 0  # The distinct nodes read so far, the arguments of sines and cosines included.
        self.frequency = frequency  # (numerator, denominator) of a, once a sine or a cosine fixes it.
        self.arguments: dict[int, tuple[object, tuple[int, int]]] = {}  # id(argument) -> (argument, its a).
        self.allowance = allowance  # The work units the claim may still cost; None for the request.

    def tick(self, n: int = 1) -> None:
        """Charge n work units: first against the allowance, so that a claim past it is False before the budget pays
        for it, then against the budget."""
        if self.allowance is not None:
            self.allowance -= n
            if self.allowance < 0:
                raise _Reject("a claim whose reading costs more than its request allows")
        self.budget.tick(n)

    # --- Coefficients ------------------------------------------------------------------------------------------------

    def _charge(self, *integers: int) -> None:
        extra = _extra(max(integer.bit_length() for integer in integers))
        if extra:
            self.tick(extra)

    def _fits(self, numerator: int, denominator: int, capped: bool) -> None:
        if capped and (numerator.bit_length() > self.max_bits or denominator.bit_length() > self.max_bits):
            raise self.oversized("Computation limit exceeded: integer bits")

    def q_add(self, p: tuple[int, int], q: tuple[int, int], capped: bool = True) -> tuple[int, int] | None:
        """p + q in lowest terms, None for zero."""
        (n1, d1), (n2, d2) = p, q
        self._charge(n1, d1, n2, d2)
        if d1 == d2:
            numerator, denominator = n1 + n2, d1
        else:
            numerator, denominator = n1 * d2 + n2 * d1, d1 * d2
        if not numerator:
            return None
        divisor = gcd(numerator, denominator)
        if divisor != 1:
            numerator, denominator = numerator // divisor, denominator // divisor
        self._fits(numerator, denominator, capped)
        return numerator, denominator

    def q_mul(self, p: tuple[int, int], q: tuple[int, int], capped: bool = True) -> tuple[int, int]:
        """p * q in lowest terms, for nonzero p and q."""
        (n1, d1), (n2, d2) = p, q
        self._charge(n1, d1, n2, d2)
        divisor = gcd(n1, d2)
        if divisor != 1:
            n1, d2 = n1 // divisor, d2 // divisor
        divisor = gcd(n2, d1)
        if divisor != 1:
            n2, d1 = n2 // divisor, d1 // divisor
        numerator, denominator = n1 * n2, d1 * d2
        self._fits(numerator, denominator, capped)
        return numerator, denominator

    def _accumulate(self, target: dict, key: tuple[int, int, int], coefficient: tuple[int, int],
                    capped: bool = True) -> None:
        old = target.get(key)
        if old is None:
            target[key] = coefficient
            return
        total = self.q_add(old, coefficient, capped)
        if total is None:
            del target[key]
        else:
            target[key] = total

    # --- Polynomials -------------------------------------------------------------------------------------------------

    def p_add(self, p: dict, q: dict, own_p: bool, own_q: bool) -> dict:
        """p + q; an operand that the caller owns may become the sum."""
        if p is q:
            own_p = own_q = False
        if len(p) < len(q):
            p, q, own_p, own_q = q, p, own_q, own_p
        self.tick(1 + len(q) + (0 if own_p else len(p) >> 4))  # The copy of p, unless it is reused.
        target = p if own_p else dict(p)
        for key, coefficient in q.items():
            self._accumulate(target, key, coefficient)
        return target

    def p_mul(self, p: dict, q: dict) -> dict:
        """p * q, with S**2 replaced by 1 - C**2."""
        if not p or not q:
            return {}
        self.tick(2 * len(p) * len(q))  # Before the loops: a product too large stops before it starts.
        target: dict = {}
        for (i1, j1, k1), c1 in p.items():
            for (i2, j2, k2), c2 in q.items():
                coefficient = self.q_mul(c1, c2)
                if j1 and j2:
                    self._accumulate(target, (i1 + i2, 0, k1 + k2), coefficient)
                    self._accumulate(target, (i1 + i2, 0, k1 + k2 + 2), (-coefficient[0], coefficient[1]))
                else:
                    self._accumulate(target, (i1 + i2, j1 + j2, k1 + k2), coefficient)
        return target

    def p_scale(self, p: dict, factor: tuple[int, int], own: bool) -> dict:
        """factor * p for a nonzero factor."""
        self.tick(1 + len(p))
        target = p if own else {}
        for key, coefficient in p.items():
            target[key] = self.q_mul(coefficient, factor)  # Never zero: no key is removed while p is iterated.
        return target

    def p_power(self, p: dict, exponent: int, own: bool) -> dict:
        """p**exponent by repeated squaring, every product charged before it runs."""
        if exponent == 0:
            return {_CONSTANT: _ONE}
        if not p:
            return {}
        if exponent == 1:
            return p if own else dict(p)
        if len(p) == 1 and _CONSTANT in p:
            numerator, denominator = p[_CONSTANT]
            if denominator == 1 and numerator in (1, -1):
                return {_CONSTANT: (-1 if numerator < 0 and exponent & 1 else 1, 1)}
            if exponent >= self.max_bits:  # |c| != 1, so c**exponent has an integer of more than exponent bits.
                raise self.oversized("Computation limit exceeded: integer bits")
        result, base = None, p
        while True:
            self.tick()
            if exponent & 1:
                result = base if result is None else self.p_mul(result, base)
            exponent >>= 1
            if not exponent:
                return result if result is not p or own else dict(result)
            base = self.p_mul(base, base)

    def derivative(self, p: dict) -> dict:
        """D p with D x = 1, D S = a C and D C = -a S, reduced; its coefficients are not capped (they exceed those of p
        by the bits of a and of an exponent at most)."""
        a = self.frequency
        self.tick(1 + 4 * len(p))
        target: dict = {}
        for (i, j, k), coefficient in p.items():
            if i:
                self._accumulate(target, (i - 1, j, k), self.q_mul(coefficient, (i, 1), False), False)
            if not j and not k:
                continue
            if a is None:  # A sine or a cosine fixes a whenever a monomial has S or C.
                raise _Reject("a trigonometric monomial without a frequency")
            scaled = self.q_mul(coefficient, a, False)
            if j:  # (S C**k)' holds a C**(k + 1).
                self._accumulate(target, (i, 0, k + 1), scaled, False)
            if k:  # (C**k)' = -k a S C**(k - 1), and S S = 1 - C**2.
                numerator, denominator = self.q_mul(scaled, (k, 1), False)
                if j:
                    self._accumulate(target, (i, 0, k - 1), (-numerator, denominator), False)
                    self._accumulate(target, (i, 0, k + 1), (numerator, denominator), False)
                else:
                    self._accumulate(target, (i, 1, k - 1), (-numerator, denominator), False)
        return target

    # --- Trees -------------------------------------------------------------------------------------------------------

    def _divisor(self, node: object) -> tuple[int, int]:
        """The inverse of a divisor, which must be a scalar node holding a nonzero Rational."""
        self.tick()
        op, args = getattr(node, "op", _MISSING), getattr(node, "args", _MISSING)
        if _named(node) != _REAL_EXPR or type(op) is not str or op != "scalar" or type(args) is not tuple or args:
            raise _Reject("a division by anything but a rational scalar")
        numerator, denominator = _rational(self.tick, self.max_bits, getattr(node, "value", _MISSING), self.oversized)
        if not numerator:
            raise _Reject("a division by zero")
        return (denominator, numerator) if numerator > 0 else (-denominator, -numerator)

    def _plan(self, node: object, trig: bool) -> tuple[str, tuple, object]:
        """(op, operands to read first, payload) of an admitted node; _Reject for every other node."""
        if _named(node) != _REAL_EXPR:
            raise _Reject("not a RealExpr")
        op, args, value = (getattr(node, name, _MISSING) for name in ("op", "args", "value"))
        if type(op) is not str or type(args) is not tuple:
            raise _Reject("a malformed RealExpr")
        if op == "scalar" and not args:
            return op, (), _rational(self.tick, self.max_bits, value, self.oversized)
        if op == "symbol" and not args and _is_variable(value, self.variable):
            return op, (), None
        if op in ("add", "mul") and len(args) == 2 and value is None:
            return op, args, None
        if op == "div" and len(args) == 2 and value is None:
            return op, args[:1], self._divisor(args[1])
        if op == "pow" and len(args) == 1 and type(value) is int and value >= 0:
            return op, args, value
        if trig and op in ("sin", "cos") and len(args) == 1 and value is None:
            return op, (), args[0]
        raise _Reject("a node outside the admitted ones")

    def _walk(self, root: object, trig: bool) -> tuple[list[int], dict[int, tuple], dict[int, int]]:
        """The distinct nodes below root in post-order (operands first), their plans and how many parents use each,
        without recursion. Every node read counts one work unit and against max_nodes."""
        order: list[int] = []
        plans: dict[int, tuple] = {}
        uses: dict[int, int] = {id(root): 1}
        state: dict[int, bool] = {}  # False while a node's operands are read, True once it is done.
        stack = [(root, False)]
        while stack:
            node, leaving = stack.pop()
            key = id(node)
            if leaving:
                state[key] = True
                order.append(key)
                continue
            done = state.get(key)
            if done:
                continue
            if done is False:  # Reached again while its own operands are read: it is its own descendant.
                raise _Reject("a node that is its own descendant")
            self.nodes += 1
            if self.nodes > self.max_nodes:
                raise self.oversized(f"Computation limit exceeded: node_count ({self.max_nodes})")
            self.tick()
            plan = self._plan(node, trig)
            plans[key] = plan
            state[key] = False
            stack.append((node, True))
            for operand in plan[1]:
                uses[id(operand)] = uses.get(id(operand), 0) + 1
                if not state.get(id(operand)):
                    stack.append((operand, False))
        return order, plans, uses

    def read(self, root: object, x_cap: int, degree_cap: int, degree_error: type[Exception], trig: bool = True,
             x_error: type[Exception] = _Reject) -> tuple[dict, int]:
        """(reduced polynomial, degree bound) of a tree. The bound of every subtree in the variable must stay within
        x_cap (else x_error) and its bound in x, S and C together within degree_cap (else degree_error), both tested
        before the subtree's arithmetic. A polynomial is freed, or reused, once its last parent has taken it."""
        order, plans, uses = self._walk(root, trig)
        results: dict[int, tuple[int, int, dict]] = {}

        def take(operand: object) -> tuple[int, int, dict, bool]:
            key = id(operand)
            uses[key] -= 1
            x_degree, degree, polynomial = results[key]
            if uses[key]:
                return x_degree, degree, polynomial, False
            del results[key]
            return x_degree, degree, polynomial, True

        for key in order:
            op, operands, payload = plans.pop(key)
            taken = [take(operand) for operand in operands]
            if op == "scalar":
                x_degree, degree = 0, 0
            elif op == "symbol":
                x_degree, degree = 1, 1
            elif op in ("sin", "cos"):
                self._frequency(payload)
                x_degree, degree = 0, 1
            elif op == "add":
                x_degree, degree = max(taken[0][0], taken[1][0]), max(taken[0][1], taken[1][1])
            elif op == "mul":
                x_degree, degree = taken[0][0] + taken[1][0], taken[0][1] + taken[1][1]
            elif op == "div":
                x_degree, degree = taken[0][0], taken[0][1]
            else:  # pow
                x_degree, degree = taken[0][0] * payload, taken[0][1] * payload
            if x_degree > x_cap:
                raise x_error("a power of the variable outside the admitted degree")
            if degree > degree_cap:
                raise degree_error(f"Computation limit exceeded: a polynomial degree above {degree_cap}")
            if op == "scalar":
                polynomial = {_CONSTANT: payload} if payload[0] else {}
            elif op == "symbol":
                polynomial = {_LINEAR: _ONE}
            elif op in ("sin", "cos"):
                polynomial = {_SINE if op == "sin" else _COSINE: _ONE}
            elif op == "add":
                polynomial = self.p_add(taken[0][2], taken[1][2], taken[0][3], taken[1][3])
            elif op == "mul":
                polynomial = self.p_mul(taken[0][2], taken[1][2])
            elif op == "div":
                polynomial = self.p_scale(taken[0][2], payload, taken[0][3])
            else:
                polynomial = self.p_power(taken[0][2], payload, taken[0][3])
            results[key] = (x_degree, degree, polynomial)
        _, degree, polynomial = results[id(root)]
        return polynomial, degree

    def _frequency(self, argument: object) -> None:
        """Read the argument of a sine or a cosine, which must be a*x for a nonzero rational a, and hold a to the
        frequency of the others. Its subtrees may have any degree up to max_work, as the request's whole tree may: an
        argument is admitted for what it denotes (x*x - x*x + x is x), so that a producer that reads every node of an
        argument exactly, as a linear form or as a polynomial, admits no argument that this reading refuses. Past
        max_work the argument is oversized: ResourceLimitError in the request, False in the claim (whose allowance
        bounds the work of every argument below that)."""
        found = self.arguments.get(id(argument))
        if found is None:
            cap = self.budget.limits.max_work
            polynomial, _ = self.read(argument, cap, cap, self.oversized, trig=False, x_error=self.oversized)
            if len(polynomial) != 1 or _LINEAR not in polynomial:
                raise _Reject("a sine or a cosine of something other than a*x with a rational a != 0")
            found = self.arguments[id(argument)] = (argument, polynomial[_LINEAR])
        if self.frequency is None:
            self.frequency = found[1]
        elif self.frequency != found[1]:
            raise _Reject("sines and cosines of different arguments")


# --- The checks ----------------------------------------------------------------------------------------------------

class _Source(NamedTuple):
    """The request's function read: its variable, its reduced normal form P, its a (None for a constant f), the total
    degree d of P, the degree bound D of its tree and the allowance A of the claim (the module docstring defines
    them)."""
    variable: Symbol
    polynomial: dict
    frequency: tuple[int, int] | None
    degree: int
    bound: int
    allowance: int


def _variable(args: dict) -> Symbol:
    """The request's variable: a Symbol with an exact str name and uid, the domain Reals and no assumptions, the form
    that _is_variable compares against; _Reject for anything else."""
    variable = args["variable"]
    name, uid = getattr(variable, "name", _MISSING), getattr(variable, "uid", _MISSING)
    assumptions = getattr(variable, "assumptions", _MISSING)
    if (type(variable) is not Symbol or type(name) is not str or type(uid) is not str
            or getattr(variable, "domain", _MISSING) is not Reals or type(assumptions) is not tuple or assumptions):
        raise _Reject("the variable is no real Symbol without assumptions")
    return variable


def _expression(function: object, variable: Symbol) -> object:
    """The expression of a RealFunction of the single variable with the domain TRUE; _Reject for anything else."""
    if _named(function) != _REAL_FUNCTION:
        raise _Reject("not a RealFunction")
    expression, variables, domain = (getattr(function, name, _MISSING)
                                     for name in ("expression", "variables", "domain"))
    if (type(variables) is not tuple or len(variables) != 1 or not _is_variable(variables[0], variable)
            or _named(domain) != _DOMAIN):
        raise _Reject("not a function of the variable alone")
    kind, operands = getattr(domain, "kind", _MISSING), getattr(domain, "operands", _MISSING)
    if type(kind) is not str or kind != "true" or type(operands) is not tuple or operands:
        raise _Reject("a function whose domain is not TRUE")
    return expression


def _coefficient_bits(polynomial: dict, frequency: tuple[int, int] | None, degree: int) -> int:
    """B, a bound on every numerator and denominator of the canonical antiderivative's coefficients (the module
    docstring derives it)."""
    longest = max((numerator.bit_length() for numerator, _ in polynomial.values()), default=0)
    denominators = sum(denominator.bit_length() for _, denominator in polynomial.values())
    frequency_bits = 0 if frequency is None else frequency[0].bit_length() + frequency[1].bit_length()
    return (longest + denominators + frequency_bits + (degree // 2 + 1) * (degree + 1).bit_length()
            + (degree + 2).bit_length() + 1)


def _source(budget: Budget, args: dict) -> _Source:
    """The request's function, a polynomial in S and C alone, read; the claim's allowance follows from it."""
    variable = _variable(args)
    start = budget.work
    reader = _Reader(budget, variable, claim=False)
    polynomial, bound = reader.read(_expression(args["function"], variable), 0, budget.limits.max_work,
                                    ResourceLimitError)
    budget.tick(1 + len(polynomial))  # The degree and the bit lengths below.
    degree = max((j + k for _, j, k in polynomial), default=0)
    bits = _coefficient_bits(polynomial, reader.frequency, degree)
    allowance = 2 * (budget.work - start) + 4096 + (2 * degree + 2) * (512 + 32 * _extra(bits + 64))
    return _Source(variable, polynomial, reader.frequency, degree, bound, allowance)


def _antiderivative(budget: Budget, source: _Source, claim: object) -> tuple[dict, _Reader]:
    """The reduced normal form of the claimed antiderivative, and its reader, once (1) and (2) of the module
    docstring hold; _Reject otherwise. The reader keeps no allowance afterwards: what follows depends on the request
    alone."""
    reader = _Reader(budget, source.variable, claim=True, frequency=source.frequency, allowance=source.allowance)
    polynomial, _ = reader.read(_expression(claim, source.variable), 1, source.bound + 1, _Reject)
    if _CONSTANT in polynomial:
        raise _Reject("a constant term")
    derivative = reader.derivative(polynomial)
    reader.tick(1 + len(source.polynomial))  # The comparison.
    if derivative != source.polynomial:
        raise _Reject("F' != f")
    reader.allowance = None
    return polynomial, reader


def _check_antiderivative(budget: Budget, args: dict, value: object, evidence: object) -> bool:
    if type(evidence) is not tuple or evidence:
        return False
    _antiderivative(budget, _source(budget, args), value)
    return True


def _bound(budget: Budget, value: object) -> tuple[int, int]:
    """(numerator, denominator) of lower_pi or upper_pi, an exact int or Rational; ResourceLimitError beyond
    max_integer_bits, as run() and the verifier refuse such an argument before any checker runs."""
    if type(value) is int:
        budget.check_int(value)
        return value, 1
    return _rational(budget.tick, budget.limits.max_integer_bits, value, ResourceLimitError)


def _angle(budget: Budget, frequency: tuple[int, int], bound: tuple[int, int]) -> int:
    """The integer k = 12*a*t, where a*t*pi = k*pi/12; _Reject when 12*a*t is no integer."""
    (a_numerator, a_denominator), (t_numerator, t_denominator) = frequency, bound
    budget.tick(1 + _extra(max(a_numerator.bit_length(), t_numerator.bit_length(), a_denominator.bit_length(),
                               t_denominator.bit_length())))
    numerator, denominator = 12 * a_numerator * t_numerator, a_denominator * t_denominator
    if numerator % denominator:
        raise _Reject("a bound that is no multiple of pi/(12*a)")
    return numerator // denominator


def _bits(*elements: tuple[Rational, ...]) -> int:
    """The longest integer of some elements of FIELD."""
    return max(max(item.numerator.bit_length(), item.denominator.bit_length()) for element in elements
               for item in element)


def _evaluate(budget: Budget, trig: dict, k: int) -> tuple[Rational, ...]:
    """N0(C) + S*N1(C) in FIELD at S = sine(k), C = cosine(k), by Horner's rule in C. Each step is charged before it
    runs: at most 8 products (a table entry has at most two nonzero coordinates) and 8 sums of coordinates."""
    s, c = sine(k), cosine(k)
    budget.tick(1 + len(trig))
    top = max(power for _, _, power in trig)
    halves = []
    for j in (0, 1):
        accumulator = FIELD.zero
        for power in range(top, -1, -1):
            coefficient = trig.get((0, j, power))
            term = None if coefficient is None else FIELD.constant(trusted_rational(*coefficient))
            budget.tick(1 + 16 * _extra(_bits(accumulator) if term is None else _bits(accumulator, term)))
            accumulator = FIELD.mul(accumulator, c)
            if term is not None:
                accumulator = FIELD.add(accumulator, term)
        halves.append(accumulator)
    budget.tick(1 + 16 * _extra(_bits(*halves)))
    return FIELD.add(halves[0], FIELD.mul(s, halves[1]))


def _identical(expected: object, claimed: object) -> bool:
    """Whether the claimed expression is the expected canonical one, node by node with exact types (no __eq__ of the
    claim runs). The expected tree has at most three levels, so the walk is short whatever the claim holds."""
    kind = type(expected)
    if type(claimed) is not kind:
        return False
    if kind is Rational:
        numerator, denominator = getattr(claimed, "numerator", _MISSING), getattr(claimed, "denominator", _MISSING)
        return (type(numerator) is int and type(denominator) is int and numerator == expected.numerator
                and denominator == expected.denominator)
    if kind is Sqrt:
        return _identical(expected.radicand, getattr(claimed, "radicand", _MISSING))
    if kind in (Mul, Add):
        name = "factors" if kind is Mul else "terms"
        parts, claimed_parts = getattr(expected, name), getattr(claimed, name, _MISSING)
        return (type(claimed_parts) is tuple and len(claimed_parts) == len(parts)
                and all(_identical(part, claimed_part) for part, claimed_part in zip(parts, claimed_parts)))
    return False


def _check_integral(budget: Budget, args: dict, value: object, evidence: object) -> bool:
    if type(evidence) is not tuple or len(evidence) != 1 or _named(value) != RECORD:
        return False
    claimed_pi = _rational(budget.tick, budget.limits.max_integer_bits, getattr(value, "pi_coefficient", _MISSING),
                           _Reject)
    lower, upper = _bound(budget, args["lower_pi"]), _bound(budget, args["upper_pi"])
    source = _source(budget, args)
    polynomial, reader = _antiderivative(budget, source, evidence[0])
    budget.tick(1 + len(polynomial))
    if any(key[0] for key in polynomial if key != _LINEAR):  # (2) leaves only lambda*x, which a claim cannot dodge.
        return False
    # The pi coefficient: lambda*(upper - lower).
    slope = polynomial.get(_LINEAR)
    width = reader.q_add(upper, (-lower[0], lower[1]), False)
    expected_pi = (0, 1) if slope is None or width is None else reader.q_mul(slope, width, False)
    if claimed_pi != expected_pi:
        return False
    # The algebraic part: T(upper*pi) - T(lower*pi) in Q(sqrt 2, sqrt 3), 0 for T = 0 on any bounds (P is then a
    # constant, whose sines, if f has any, cancel: the module docstring explains why such bounds are free).
    trig = {key: coefficient for key, coefficient in polynomial.items() if key != _LINEAR}
    if not trig:
        difference = FIELD.zero
    elif source.frequency is None:  # Only a sine or a cosine of f fixes a, and (2) gives F no trigonometric part else.
        return False
    else:
        k_lower, k_upper = _angle(budget, source.frequency, lower), _angle(budget, source.frequency, upper)
        high, low = _evaluate(budget, trig, k_upper), _evaluate(budget, trig, k_lower)
        budget.tick(1 + 4 * _extra(_bits(high, low)))
        difference = FIELD.sub(high, low)
    c0, c1, c2, c3 = difference
    expected = add(c0, mul(c1, sqrt(2)), mul(c2, sqrt(3)), mul(c3, sqrt(6)))
    return _identical(expected, getattr(value, "algebraic", _MISSING))


_CHECKS: dict[str, Callable[[Budget, dict, object, object], bool]] = {
    "trig_power_antiderivative": _check_antiderivative,
    "trig_power_integral": _check_integral,
}


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for other argument names and for any malformed or oversized claim;
    ResourceLimitError past the budget or for a request over the limits."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = OPERATIONS.get(operation)
    if names is None or len(args) != len(names) or any(name not in args for name in names):
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        try:
            return _CHECKS[operation](budget, args, value, evidence)
        except _Reject:
            return False
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError, InvalidInput):
            return False  # A claim forged without __init__ or with odd fields; fail closed.
