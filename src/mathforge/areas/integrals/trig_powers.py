r"""Integrals of polynomials in sin(a*x) and cos(a*x) (mathforge.integrals): antiderivatives in closed form, in powers
of the same sine and cosine, and definite integrals between rational multiples of pi.

The input class of both operations (docs/v09/cards/m3-analysis.md): function is a RealFunction f = P(sin(a*x),
cos(a*x)) with the variables (variable,) and the domain TRUE, where P is a polynomial with rational coefficients and a
is ONE nonzero rational for the whole input. Its tree may use add, mul, div whose divisor is a scalar node holding a
nonzero Rational, pow with an int exponent >= 0, scalar nodes holding a Rational, sin and cos, and the symbol node of
the variable inside their arguments only (not even x**0 outside them). The argument of every sine and cosine is a tree
of the same nodes but sin and cos that, read as a polynomial in the variable, is a*x without a constant term, with the
same a everywhere (the trees may differ: x + x is 2*x, and (x + 1) - 1 and x*x - x*x + x are x, while sin(x) and
sin(-x) have different arguments). Every node counts, also below a power 0 or a factor 0: sin(2*x)**0 + sin(x) has two
different arguments, and x**0*sin(x) has the variable outside them. A tree without sin and cos, a rational constant,
is in the class too. variable is a real Symbol without assumptions. Every other input is invalid input
("trig_power_antiderivative accepts polynomials in sin(a*x) and cos(a*x)"); several different arguments are out of
scope, because the identities between them break the normal-form test.

- trig_power_antiderivative(function, variable) -> RealFunction F = lambda*x + T(sin(a*x), cos(a*x)) with F' = f,
  lambda rational and T a polynomial with rational coefficients in the sine and cosine of the same a*x (no multiple
  angles, no negative powers, no division but by nonzero rational scalars), with the variables (variable,), the domain
  TRUE and no constant term: once S**2 is replaced by 1 - C**2 (S = sin(a*x), C = cos(a*x)), the reduced normal form
  N0(x, C) + S*N1(x, C) of F has no monomial x**0 S**0 C**0. That makes F unique: sin(x)*cos(x) gives -cos(x)**2/2,
  never sin(x)**2/2 = 1/2 - cos(x)**2/2, and sin(x)**3 gives -cos(x) + cos(x)**3/3. Evidence ().
- trig_power_integral(function, variable, lower_pi, upper_pi) -> PiLinear: the integral of f from lower_pi*pi to
  upper_pi*pi, for exact rationals lower_pi and upper_pi in either order. When f has a sine or a cosine,
  12*a*lower_pi and 12*a*upper_pi must be integers (the angles are then multiples of pi/12, where sine and cosine lie
  in Q(sqrt 2, sqrt 3)), else invalid input, even when the sines of f cancel: the card's restriction. (The checker,
  whose domain contains the producer's, also verifies a claim on other bounds when P reduces to a constant, whose
  integral needs no table.) Evidence (F,): the antiderivative above, in its canonical form.

The value of trig_power_integral is the PiLinear record algebraic + pi_coefficient*pi, in the canonical form that each
such number has exactly once: pi_coefficient a Rational (never an int), and algebraic c0 + c1*sqrt(2) + c2*sqrt(3) +
c3*sqrt(6) for Rationals c0, ..., c3 exactly as mathforge.model builds it with add(c0, mul(c1, sqrt(2)),
mul(c2, sqrt(3)), mul(c3, sqrt(6))) (_algebraic below): a Rational when c1 = c2 = c3 = 0, else a Sqrt, a Mul of a
Rational and a Sqrt, or an Add of such terms in model's order. It renders as pi/2 and 3*sqrt(3)/8 + pi/6, in LaTeX
\frac{\pi}{2} and \frac{3 \sqrt{3}}{8} + \frac{\pi}{6}.

The checker, mathforge.certify.integrals.trig_powers, states the theorems: it reads f and F as polynomials in
(x, S, C), differentiates F formally, reduces F' - f to N0(x, C) + S*N1(x, C) and requires zero, and evaluates F at the
bounds with its own table of sin and cos at the multiples of pi/12. Beyond the mathematics it asks of the producer's F:
a RealFunction with the variables (variable,) and the domain TRUE, any tree of the nodes above (the variable also
outside the arguments) whose sines and cosines have f's a, with at most max_nodes distinct nodes, in which every
subtree outside the arguments of the sines and cosines has a degree of at most 1 in the variable and at most D + 1 in
x, S and C together (D the degree bound of f's tree, from below upwards: add the larger, mul the sum, div the
dividend's, pow k times its operand's, sin and cos 1), no integer over max_integer_bits, and a reading that costs the
checker at most the allowance that f fixes (the checker's docstring gives it). F built from the reduced normal form of
f as a balanced sum of monomials, each a scalar times x, sin(u) and a power of cos(u), with one small u shared by every
sine and cosine (the symbol node of the variable for a = 1, else mul(scalar(a), variable)), meets all of these with a
wide margin; the balanced sum also keeps a Result within max_depth when the budget inspects it.

How produce() computes (the card's algorithm; nothing here is a proof, the checker verifies every value with a formal
derivative and a table of its own). It validates the arguments, then reads f in four passes over the distinct nodes of
its tree, without recursion: (1) it checks the shape of every node of the main tree, which must not hold the variable;
(2) it checks every distinct argument of a sine or a cosine, then reads it as a polynomial in the variable, which must
be a*x for one nonzero rational a; (3) it bounds the degree of every subtree from below upwards (add: the larger, mul:
the sum, div: the dividend's, pow k: k times its operand's, sin and cos: 1; inside an argument the same in the
variable), and a bound above max_work makes the request too big for the budget (ResourceLimitError), as the checker
reads it; (4) it multiplies f out into its reduced normal form P = N0(C) + S*N1(C), replacing S**2 by 1 - C**2 in every
product, with Rational coefficients. Every node counts in passes (1) to (3), also below a power 0 or a factor 0, so
an input outside the class is invalid input before any arithmetic runs. It then integrates term by term: S*C**j gives
-C**(j+1)/((j+1)*a), and C**j the reduction int C**j = C**(j-1)*S/(j*a) + (j-1)/j int C**(j-2), run from the highest
power down, which carries (j-1)/j times the coefficient of C**j to C**(j-2) (int C = S/a), until the constant term gives
lambda*x. F is written as a balanced sum of the monomials lambda*x, then c*cos(u)**k for k = 1, 2, ..., then
c*sin(u)*cos(u)**k for k = 0, 1, ..., each a scalar(c) times the monomial (the scalar left out for c = 1, a power 1 as
the cosine itself), with one u, one sin(u) and one cos(u) node that every monomial shares: u is the symbol node of the
variable for a = 1 and mul(scalar(a), variable) otherwise; F = 0 is scalar(0). So F depends only on P, a and the
variable, and every request gets the same F for the same function. trig_power_integral then evaluates T = F - lambda*x
at both angles k*pi/12, k = 12*a*t for t = lower_pi and upper_pi, by Horner's rule in C, in Q(sqrt 2, sqrt 3) with
elements c0 + c1*sqrt(2) + c2*sqrt(3) + c3*sqrt(6) and an exact table of cos(k*pi/12) (sin(k*pi/12) =
cos((6 - k)*pi/12)); the pi coefficient is lambda*(upper_pi - lower_pi).

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and one per node read (the arguments of the sines and
cosines included, against max_nodes as well), and every coefficient operation is an operation of Rational, which
charges its own unit and keeps its integers within max_integer_bits. Before a polynomial product, a sum, a scaling or a
step of the integration or of Horner's rule, it charges what long integers add, on the checker's scale (one unit per
sixteen squared 64-bit words of the longest operand, for the products and the gcd of a rational operation), and it
refuses a product that the budget cannot pay for before the product starts. A power of a constant other than 0, 1 and
-1 with an exponent of max_integer_bits or more has an integer that is too long. F's reading costs the checker a small
fraction of its allowance, and the balanced sum keeps a Result within max_depth whatever the number of monomials.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/integrals/test_trig_powers.py is the definition of done.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Add, Mul, Rational, Reals, Sqrt, Symbol, add, mul, sqrt, trusted_rational
from ...registry import Argument, OperationSpec

# The producer rules:
# - Imports at the top come only from modules that `import mathforge` already loads; kernels are imported inside
#   produce(). Never import mathforge.certify, mathforge.lean, mathforge.mcp, mathforge._generated or a generated
#   area module (mathforge.ntheory, ...).
# - produce() validates its inputs first, with tests like `type(x) is int` (a bool, a float or a str is never an
#   exact number), and raises InvalidInput. It builds records only from validated data. For a Matrix argument it
#   calls current_budget().matrix(rows, cols) before it validates the shape.
# - The value is never None. The evidence is a tuple of ints, Rationals, tuples, str and records: no list, dict or
#   float. Under the verifier, 23 + 2 * request + 2 * value + evidence nodes stay within max_evidence_items.
# - Every operation takes at least one argument. Only an Expression or a Matrix that Result.expression should hold
#   is named `expression` or `A`.
# - Operation names are new, and nothing is superseded (no SUPERSEDES): tools/new_family.py checked these names.
# - No speed-suite fixture and no hand-written lean module: the generic lean runner serves v6 operations.

# The scope text of every operation of wave M3 (docs/v09/cards/m3-analysis.md).
_SCOPE = "exact rationals and real functions; certified closed forms of sums, series and trigonometric integrals"

SPECS = (
    OperationSpec("trig_power_antiderivative", "integrals",
                  (Argument("function", "RealFunction", None, "function"),
                   Argument("variable", "Symbol", None, "symbol")),
                  "RealFunction",
                  "Integrate a polynomial in sin(a*x) and cos(a*x) in closed form, with powers of the same sine and "
                  "cosine.", _SCOPE, version=6),
    OperationSpec("trig_power_integral", "integrals",
                  (Argument("function", "RealFunction", None, "function"),
                   Argument("variable", "Symbol", None, "symbol"),
                   Argument("lower_pi", "Rational | int", None, "rational"),
                   Argument("upper_pi", "Rational | int", None, "rational")),
                  "PiLinear",
                  "Integrate a polynomial in sin(a*x) and cos(a*x) between rational multiples of pi.", _SCOPE,
                  version=6),
)

_RADICANDS = (2, 3, 6)  # The square roots of the algebraic part, after the rational term.
_MISSING = object()  # A field that an object made with object.__new__ lacks.


def _algebraic(c0: Rational, c1: Rational, c2: Rational, c3: Rational) -> object:
    """c0 + c1*sqrt(2) + c2*sqrt(3) + c3*sqrt(6) in the normal form of mathforge.model: the canonical algebraic part."""
    return add(c0, mul(c1, sqrt(2)), mul(c2, sqrt(3)), mul(c3, sqrt(6)))


def _integers(budget, value: object, name: str) -> tuple[int, int]:
    """(numerator, denominator) of an exact Rational field in lowest terms with a positive denominator; a Rational made
    with object.__new__ may lack a field or break that form. ResourceLimitError for an integer over max_integer_bits."""
    if type(value) is not Rational:
        raise InvalidInput(f"The {name} of a PiLinear is a Rational: an int, a float or a bool is no Rational")
    numerator, denominator = getattr(value, "numerator", _MISSING), getattr(value, "denominator", _MISSING)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"The {name} of a PiLinear is a Rational of two exact integers")
    budget.check_int(numerator)
    budget.check_int(denominator)
    words = max(numerator.bit_length(), denominator.bit_length()) >> 6
    budget.tick(1 + (words * words >> 6))  # math.gcd, Lehmer's algorithm: quadratic in the length.
    if gcd(numerator, denominator) != 1:
        raise InvalidInput(f"The {name} of a PiLinear is a Rational in lowest terms")
    return numerator, denominator


def _coefficients(budget, algebraic: object) -> tuple[Rational, Rational, Rational, Rational]:
    """The coefficients (c0, c1, c2, c3) of 1, sqrt(2), sqrt(3) and sqrt(6) in a Rational, a Sqrt, a Mul of a Rational
    and a Sqrt, or an Add of two to four such terms with distinct radicals; InvalidInput for anything else. The normal
    form itself (no zero or unit coefficient in a Mul, model's order) is left to the comparison with _algebraic."""
    shape = InvalidInput("The algebraic part of a PiLinear is a Rational or a closed Expression in sqrt(2), sqrt(3) "
                         "and sqrt(6)")
    terms = getattr(algebraic, "terms", _MISSING) if type(algebraic) is Add else (algebraic,)
    if type(terms) is not tuple or not 1 <= len(terms) <= 4:
        raise shape
    found: dict[int, Rational] = {}
    for term in terms:
        budget.tick()
        if type(term) is Rational:
            radicand, coefficient = 1, term
        else:
            coefficient, root = trusted_rational(1, 1), term
            if type(term) is Mul:
                factors = getattr(term, "factors", _MISSING)
                if type(factors) is not tuple or len(factors) != 2:
                    raise shape
                coefficient, root = factors
            if type(root) is not Sqrt:
                raise shape
            numerator, denominator = _integers(budget, getattr(root, "radicand", _MISSING), "radicand")
            if denominator != 1 or numerator not in _RADICANDS:
                raise shape
            radicand = numerator
        _integers(budget, coefficient, "coefficient")
        if radicand in found:
            raise shape
        found[radicand] = coefficient
    zero = trusted_rational(0, 1)
    return found.get(1, zero), found.get(2, zero), found.get(3, zero), found.get(6, zero)


def _identical(expected: object, claimed: object) -> bool:
    """Whether claimed is the expression expected, node by node with exact types."""
    kind = type(expected)
    if type(claimed) is not kind:
        return False
    if kind is Rational:
        return (type(claimed.numerator) is int and type(claimed.denominator) is int
                and (claimed.numerator, claimed.denominator) == (expected.numerator, expected.denominator))
    if kind is Sqrt:
        return _identical(expected.radicand, claimed.radicand)
    name = "factors" if kind is Mul else "terms"
    parts, claimed_parts = getattr(expected, name), getattr(claimed, name)
    return (type(claimed_parts) is tuple and len(claimed_parts) == len(parts)
            and all(_identical(part, claimed_part) for part, claimed_part in zip(parts, claimed_parts)))


@dataclass(frozen=True)
class PiLinear:
    """algebraic + pi_coefficient * pi, with algebraic a closed expression in Q(sqrt 2, sqrt 3).

    The canonical form, which each such number has exactly once (1, sqrt 2, sqrt 3 and sqrt 6 are linearly independent
    over Q, and pi is transcendental): pi_coefficient is a Rational in lowest terms (an int, a float or a bool is no
    Rational here), and algebraic is a Rational or a legacy closed Expression with square roots of 2, 3 and 6, exactly
    as _algebraic builds it from its coefficients: mathforge.model's normal form of c0 + c1*sqrt(2) + c2*sqrt(3) +
    c3*sqrt(6), which is Rational(0) for zero."""
    algebraic: object  # A Rational or a legacy closed Expression with square roots of 2, 3, 6.
    pi_coefficient: Rational

    def __post_init__(self):
        with computation() as budget:
            budget.tick()
            _integers(budget, self.pi_coefficient, "pi coefficient")
            if not _identical(_algebraic(*_coefficients(budget, self.algebraic)), self.algebraic):
                raise InvalidInput("The algebraic part of a PiLinear is c0 + c1*sqrt(2) + c2*sqrt(3) + c3*sqrt(6) in "
                                   "the normal form of mathforge.model")


TYPES = {"pi_linear": PiLinear}
CHECKER = "mathforge.certify.integrals.trig_powers"

# --- The producer --------------------------------------------------------------------------------------------------

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
_CLASS = "{} accepts polynomials in sin(a*x) and cos(a*x)"  # The card's text for a function outside the class.
_ZERO, _ONE = trusted_rational(0, 1), trusted_rational(1, 1)

# A polynomial in one unknown is a dict {k: c} of its nonzero Rational coefficients c of the k-th powers: of the
# variable inside an argument, of C = cos(a*x) in the main tree. A polynomial in S = sin(a*x) and C is held in its
# reduced normal form N0(C) + S*N1(C), as the pair (N0, N1). No dict changes once it is built, so results share them.
Poly = dict[int, Rational]
Trig = tuple[Poly, Poly]
# An element c0 + c1*sqrt(2) + c2*sqrt(3) + c3*sqrt(6) of Q(sqrt 2, sqrt 3) is the tuple (c0, c1, c2, c3) of Rationals.
Element = tuple[Rational, Rational, Rational, Rational]
_NOTHING: Element = (_ZERO, _ZERO, _ZERO, _ZERO)
# The products of the basis 1, sqrt 2, sqrt 3, sqrt 6: (index of the product, its rational factor); sqrt(2)*sqrt(6)
# = 2*sqrt(3) and sqrt(3)*sqrt(6) = 3*sqrt(2).
_BASIS = (((0, 1), (1, 1), (2, 1), (3, 1)),
          ((1, 1), (0, 2), (3, 1), (2, 2)),
          ((2, 1), (3, 1), (0, 3), (1, 3)),
          ((3, 1), (2, 2), (1, 3), (0, 6)))
# cos(m*pi/12) for m = 0, ..., 6 as (numerator, denominator) pairs of c0, ..., c3: 1, (sqrt 2 + sqrt 6)/4 =
# cos(pi/4 - pi/6), sqrt(3)/2, sqrt(2)/2, 1/2, (sqrt 6 - sqrt 2)/4 = cos(pi/4 + pi/6) and 0.
_QUADRANT = (((1, 1), (0, 1), (0, 1), (0, 1)),
             ((0, 1), (1, 4), (0, 1), (1, 4)),
             ((0, 1), (0, 1), (1, 2), (0, 1)),
             ((0, 1), (1, 2), (0, 1), (0, 1)),
             ((1, 2), (0, 1), (0, 1), (0, 1)),
             ((0, 1), (-1, 4), (0, 1), (1, 4)),
             ((0, 1), (0, 1), (0, 1), (0, 1)))


def _cosines() -> tuple[Element, ...]:
    """cos(k*pi/12) for k = 0, ..., 23, from the first quadrant: cos(2*pi - t) = cos(t) and cos(pi - t) = -cos(t)."""
    table = []
    for k in range(24):
        m, sign = min(k, 24 - k), 1
        if m > 6:
            m, sign = 12 - m, -1
        table.append(tuple(trusted_rational(sign * numerator, denominator) for numerator, denominator in _QUADRANT[m]))
    return tuple(table)


_COSINES = _cosines()  # sin(k*pi/12) = cos((6 - k)*pi/12) = _COSINES[(6 - k) % 24].


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of trig_power_antiderivative or trig_power_integral in the layouts of the module docstring,
    for arguments that operations_v6.run has checked by name (a direct call is checked here as well). InvalidInput for
    a function outside the input class, a variable that is no real Symbol without assumptions, a bound that is no exact
    rational, and bounds that are no multiples of pi/(12*a) where f has a sine or a cosine; ResourceLimitError past the
    budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the trig_powers family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    from ...functions import DomainPredicate, RealExpr, RealFunction

    outside = _CLASS.format(operation)
    variable, integral = args["variable"], operation == "trig_power_integral"
    if not _real_symbol(variable):
        raise InvalidInput(f"{outside}: variable must be a real Symbol without assumptions")
    if integral:
        for name in ("lower_pi", "upper_pi"):
            if type(args[name]) is not int and type(args[name]) is not Rational:  # Neither a bool nor a float is exact.
                raise InvalidInput(f"{name} must be an exact rational: an int or a Rational")
    expression = _expression(args["function"], variable, outside, RealFunction, DomainPredicate)
    with computation() as budget:
        budget.tick()  # The call; every node of the function charges one more.
        bounds = [_bound(budget, args[name], name) for name in ("lower_pi", "upper_pi")] if integral else []
        polynomial, frequency = _read(budget, expression, variable, outside, RealExpr)
        angles = None
        if integral and frequency is not None:  # The card's restriction, also where the sines and cosines cancel.
            angles = [_angle(budget, frequency, bound) for bound in bounds]
        slope, cosines, sines = _integrate(budget, polynomial, frequency)
        antiderivative = RealFunction(_tree(variable, frequency, slope, cosines, sines, RealExpr), (variable,))
        if not integral:
            return antiderivative, ()
        lower, upper = bounds
        difference = _NOTHING
        if angles is not None and (cosines or sines):
            high = _value_at(budget, cosines, sines, angles[1])
            low = _value_at(budget, cosines, sines, angles[0])
            difference = _field_sum(budget, high, low, -1)
        return PiLinear(_algebraic(*difference), slope * (upper - lower)), (antiderivative,)


def _real_symbol(value: object) -> bool:
    """Whether value is a Symbol with an exact str name and uid, the domain Reals and no assumptions, read field by
    field: no comparison of a forged field runs."""
    assumptions = getattr(value, "assumptions", _MISSING)
    return (type(value) is Symbol and type(getattr(value, "name", _MISSING)) is str
            and type(getattr(value, "uid", _MISSING)) is str and getattr(value, "domain", _MISSING) is Reals
            and type(assumptions) is tuple and not assumptions)


def _is_variable(candidate: object, variable: Symbol) -> bool:
    """Whether candidate is the variable: a real Symbol without assumptions with the variable's name and uid."""
    return _real_symbol(candidate) and candidate.name == variable.name and candidate.uid == variable.uid


def _expression(function: object, variable: Symbol, outside: str, real_function: type,
                domain_predicate: type) -> object:
    """The expression of a RealFunction of the variable alone with the domain TRUE; InvalidInput for anything else (a
    function made with object.__new__ may lack a field)."""
    if type(function) is not real_function:
        raise InvalidInput(f"{outside}: function must be a RealFunction")
    variables, domain = getattr(function, "variables", _MISSING), getattr(function, "domain", _MISSING)
    if type(variables) is not tuple or len(variables) != 1 or not _is_variable(variables[0], variable):
        raise InvalidInput(f"{outside}: the variables of the function must be (variable,)")
    kind, operands = getattr(domain, "kind", _MISSING), getattr(domain, "operands", _MISSING)
    if (type(domain) is not domain_predicate or type(kind) is not str or kind != "true" or type(operands) is not tuple
            or operands):
        raise InvalidInput(f"{outside}: the domain of the function must be TRUE")
    return getattr(function, "expression", _MISSING)


def _bits(value: Rational) -> int:
    """The length of the longer integer of a Rational."""
    return max(value.numerator.bit_length(), value.denominator.bit_length())


def _long(bits: int) -> int:
    """What one rational operation on integers of the given bit length costs beyond its unit, on the checker's scale:
    one unit per sixteen squared 64-bit words (its products and its gcd are quadratic), nothing below 256 bits."""
    words = bits >> 6
    return words * words >> 4


def _afford(budget: Budget, units: int, extra: int = 0) -> None:
    """Charge extra, what long integers add to some rational operations, and refuse at once operations whose units the
    budget cannot pay: each operation of Rational charges its own unit as it runs."""
    if extra:
        budget.tick(extra)
    if budget.work + units > budget.limits.max_work:
        raise ResourceLimitError(f"Computation limit exceeded: work ({budget.limits.max_work})")


def _canonical(budget: Budget, value: object, what: str) -> None:
    """Refuse, with InvalidInput, a Rational that is not two exact ints in lowest terms with a positive denominator (a
    Rational made with object.__new__ may break that form); ResourceLimitError for an integer over max_integer_bits."""
    numerator, denominator = getattr(value, "numerator", _MISSING), getattr(value, "denominator", _MISSING)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"{what} must be a Rational of two exact integers with a positive denominator")
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator != 1:
        budget.tick(1 + _long(max(numerator.bit_length(), denominator.bit_length())))  # The gcd.
        if gcd(numerator, denominator) != 1:
            raise InvalidInput(f"{what} must be a Rational in lowest terms")


def _bound(budget: Budget, value: int | Rational, name: str) -> Rational:
    """lower_pi or upper_pi, an exact int or Rational, as a canonical Rational."""
    if type(value) is int:
        budget.check_int(value)
        return Rational(value)
    _canonical(budget, value, name)
    return value


def _angle(budget: Budget, frequency: Rational, bound: Rational) -> int:
    """The integer k = 12*a*t of the angle a*t*pi = k*pi/12 at a bound t; InvalidInput when 12*a*t is no integer."""
    budget.tick(1 + _long(2 * max(_bits(frequency), _bits(bound))))
    numerator, denominator = 12 * frequency.numerator * bound.numerator, frequency.denominator * bound.denominator
    if numerator % denominator:
        raise InvalidInput("trig_power_integral needs bounds whose angles a*lower_pi*pi and a*upper_pi*pi are "
                           "multiples of pi/12, for the a*x in the sines and cosines of the function: 12*a*lower_pi "
                           "and 12*a*upper_pi must be integers")
    return numerator // denominator


# --- Reading the function ------------------------------------------------------------------------------------------

def _read(budget: Budget, root: object, variable: Symbol, outside: str,
          real_expr: type) -> tuple[Trig, Rational | None]:
    """(P, a): the reduced normal form P = (N0, N1) of the function's tree and the a of its sines and cosines, None
    when it has none, by the four passes of the module docstring. InvalidInput for a tree outside the class,
    ResourceLimitError for one beyond the budget."""
    counted = [0]  # The distinct nodes read, those of the arguments included.
    cap = budget.limits.max_work
    order = _walk(budget, root, lambda node: _admit(budget, node, variable, outside, real_expr, False), counted)
    arguments = {id(payload): payload for _, op, _, payload in order if op in ("sin", "cos")}
    walks = [_walk(budget, argument, lambda node: _admit(budget, node, variable, outside, real_expr, True), counted)
             for argument in arguments.values()]
    frequency = None
    for walk in walks:
        a = _linear(budget, walk, cap, outside)
        if frequency is None:
            frequency = a
        elif (a.numerator, a.denominator) != (frequency.numerator, frequency.denominator):
            raise InvalidInput(f"{outside} with one a for the whole function: it has sines and cosines of different "
                               "arguments")
    degrees: dict[int, int] = {}
    for node, op, operands, payload in order:
        if op == "scalar":
            degree = 0
        elif op in ("sin", "cos"):
            degree = 1
        elif op == "add":
            degree = max(degrees[id(operands[0])], degrees[id(operands[1])])
        elif op == "mul":
            degree = degrees[id(operands[0])] + degrees[id(operands[1])]
        elif op == "div":
            degree = degrees[id(operands[0])]
        else:  # pow
            degree = degrees[id(operands[0])] * payload
        if degree > cap:
            raise ResourceLimitError(f"Computation limit exceeded: a polynomial degree above {cap}")
        degrees[id(node)] = degree
    return _expand(budget, order), frequency


def _walk(budget: Budget, root: object, admit, counted: list[int]) -> list[tuple[object, str, tuple, object]]:
    """The distinct nodes of a tree, operands before the nodes that use them, each as (node, op, operands, payload)
    where admit(node) validates the node and gives the rest. Without recursion; one work unit per node, and counted,
    the distinct nodes read for the request so far, stays within max_nodes. InvalidInput for a node below itself,
    which only object.__setattr__ can make."""
    order: list[tuple[object, str, tuple, object]] = []
    state: dict[int, bool] = {}  # False while the operands of a node are read, True once the node is in order.
    limit = budget.limits.max_nodes
    stack: list[tuple[object, tuple | None]] = [(root, None)]
    while stack:
        node, plan = stack.pop()
        key = id(node)
        if plan is not None:
            state[key] = True
            order.append((node, *plan))
            continue
        done = state.get(key)
        if done:
            continue
        if done is False:  # Reached again while its own operands are read.
            raise InvalidInput("A function is a finite tree: a node of it lies below itself")
        counted[0] += 1
        if counted[0] > limit:
            raise ResourceLimitError(f"Computation limit exceeded: node_count ({limit})")
        budget.tick()
        plan = admit(node)
        state[key] = False
        stack.append((node, plan))
        stack.extend((operand, None) for operand in plan[1] if not state.get(id(operand)))
    return order


def _admit(budget: Budget, node: object, variable: Symbol, outside: str, real_expr: type,
           argument: bool) -> tuple[str, tuple, object]:
    """(op, the operands to read first, payload) of a node of the input class, in the main tree of the function
    (argument False) or in the argument of a sine or a cosine (argument True); InvalidInput for any other node. The
    payload is the Rational of a scalar, 1/c for a division by c, the exponent of a power and the argument of a sine or
    a cosine, whose own nodes are read apart."""
    if type(node) is not real_expr:
        raise InvalidInput(f"{outside}: every node of the function is a RealExpr")
    op, operands, value = (getattr(node, name, _MISSING) for name in ("op", "args", "value"))
    if type(op) is not str or type(operands) is not tuple or any(type(item) is not real_expr for item in operands):
        raise InvalidInput(f"{outside}: a node of the function is malformed")
    if op == "scalar" and not operands:
        if type(value) is not Rational:
            raise InvalidInput(f"{outside}: its constants are rationals, no other real algebraic numbers")
        _canonical(budget, value, "A constant of the function")
        return op, (), value
    if op in ("add", "mul") and len(operands) == 2 and value is None:
        return op, operands, None
    if op == "div" and len(operands) == 2 and value is None:
        return op, operands[:1], _inverse(budget, operands[1], outside, real_expr)
    if op == "pow" and len(operands) == 1 and type(value) is int:
        if value < 0:
            raise InvalidInput(f"{outside}: its powers have exponents >= 0")
        budget.check_int(value)
        return op, operands, value
    if op == "symbol" and not operands:
        if not argument:
            raise InvalidInput(f"{outside}: the variable appears only inside the arguments of the sines and cosines")
        if not _is_variable(value, variable):
            raise InvalidInput(f"{outside}: the variable is its only symbol")
        return op, (), None
    if op in ("sin", "cos") and len(operands) == 1 and value is None:
        if argument:
            raise InvalidInput(f"{outside}: the argument of a sine or a cosine holds no sine or cosine")
        return op, (), operands[0]
    raise InvalidInput(f"{outside}: {op[:20]!r} with {len(operands)} operands is not admitted (add, mul, div by a "
                       "nonzero rational, pow with an exponent >= 0, rational constants, sin and cos)")


def _inverse(budget: Budget, divisor: object, outside: str, real_expr: type) -> Rational:
    """1/c for a divisor that is a scalar node holding a nonzero Rational c; InvalidInput for any other divisor."""
    budget.tick()
    op, operands, value = (getattr(divisor, name, _MISSING) for name in ("op", "args", "value"))
    if (type(divisor) is not real_expr or type(op) is not str or op != "scalar" or type(operands) is not tuple
            or operands or type(value) is not Rational):
        raise InvalidInput(f"{outside}: it divides only by nonzero rational constants")
    _canonical(budget, value, "A divisor of the function")
    if not value.numerator:
        raise InvalidInput(f"{outside}: it divides only by nonzero rational constants, never by 0")
    return Rational(value.denominator, value.numerator)


def _uses(order: list[tuple[object, str, tuple, object]]) -> dict[int, int]:
    """How many nodes of an order take each node as an operand."""
    uses: dict[int, int] = {}
    for _, _, operands, _ in order:
        for operand in operands:
            uses[id(operand)] = uses.get(id(operand), 0) + 1
    return uses


def _linear(budget: Budget, walk: list[tuple[object, str, tuple, object]], cap: int, outside: str) -> Rational:
    """The a of the argument of a sine or a cosine, read as a polynomial in the variable that must be a*x for a
    nonzero rational a; InvalidInput for any other argument, ResourceLimitError for a subtree whose degree bound in the
    variable exceeds cap. A node's polynomial is freed once its last parent has used it."""
    uses, results = _uses(walk), {}

    def take(operand: object) -> tuple[int, Poly]:
        key = id(operand)
        uses[key] -= 1
        return results[key] if uses[key] else results.pop(key)

    for node, op, operands, payload in walk:
        parts = [take(operand) for operand in operands]
        if op == "scalar":
            degree = 0
        elif op == "symbol":
            degree = 1
        elif op == "add":
            degree = max(parts[0][0], parts[1][0])
        elif op == "mul":
            degree = parts[0][0] + parts[1][0]
        elif op == "div":
            degree = parts[0][0]
        else:  # pow
            degree = parts[0][0] * payload
        if degree > cap:
            raise ResourceLimitError(f"Computation limit exceeded: a polynomial degree above {cap}")
        if op == "scalar":
            polynomial = {0: payload} if payload.numerator else {}
        elif op == "symbol":
            polynomial = {1: _ONE}
        elif op == "add":
            polynomial = _sum(budget, parts[0][1], parts[1][1])
        elif op == "mul":
            polynomial = _product(budget, parts[0][1], parts[1][1])
        elif op == "div":
            polynomial = _scaled(budget, parts[0][1], payload)
        elif payload == 0:
            polynomial = {0: _ONE}
        elif payload == 1:
            polynomial = parts[0][1]
        elif len(parts[0][1]) <= 1 and 1 not in parts[0][1] and not any(parts[0][1].keys() - {0}):
            constant = _constant_power(budget, parts[0][1].get(0, _ZERO), payload)
            polynomial = {0: constant} if constant.numerator else {}
        else:
            polynomial = _power(budget, parts[0][1], payload, _product)
        results[id(node)] = (degree, polynomial)
    polynomial = results[id(walk[-1][0])][1]
    if len(polynomial) != 1 or 1 not in polynomial:
        raise InvalidInput(f"{outside}: the argument of every sine and cosine is a*x for a rational a != 0")
    return polynomial[1]


def _expand(budget: Budget, order: list[tuple[object, str, tuple, object]]) -> Trig:
    """The reduced normal form of the main tree, from its nodes in order, S**2 replaced by 1 - C**2 in every product.
    A node's polynomial is freed once its last parent has used it."""
    uses, results = _uses(order), {}

    def take(operand: object) -> Trig:
        key = id(operand)
        uses[key] -= 1
        return results[key] if uses[key] else results.pop(key)

    for node, op, operands, payload in order:
        parts = [take(operand) for operand in operands]
        if op == "scalar":
            polynomial = ({0: payload} if payload.numerator else {}, {})
        elif op == "sin":
            polynomial = ({}, {0: _ONE})
        elif op == "cos":
            polynomial = ({1: _ONE}, {})
        elif op == "add":
            polynomial = (_sum(budget, parts[0][0], parts[1][0]), _sum(budget, parts[0][1], parts[1][1]))
        elif op == "mul":
            polynomial = _times(budget, parts[0], parts[1])
        elif op == "div":
            polynomial = (_scaled(budget, parts[0][0], payload), _scaled(budget, parts[0][1], payload))
        else:  # pow
            even, odd = base = parts[0]
            if payload == 0:
                polynomial = ({0: _ONE}, {})
            elif payload == 1:
                polynomial = base
            elif not odd and not any(even.keys() - {0}):  # A constant: its power needs no product of polynomials.
                constant = _constant_power(budget, even.get(0, _ZERO), payload)
                polynomial = ({0: constant} if constant.numerator else {}, {})
            else:
                polynomial = _power(budget, base, payload, _times)
        results[id(node)] = polynomial
    return results[id(order[-1][0])]


# --- Polynomial arithmetic -----------------------------------------------------------------------------------------

def _widest(*polynomials: Poly) -> int:
    """The longest integer of the coefficients of some polynomials, 0 for none."""
    return max((_bits(coefficient) for polynomial in polynomials for coefficient in polynomial.values()), default=0)


def _sum(budget: Budget, p: Poly, q: Poly) -> Poly:
    """p + q; each addition is charged before it runs."""
    if not p or not q:
        return p or q
    if len(p) < len(q):
        p, q = q, p
    budget.tick(1 + (len(p) >> 4))  # The copy of p.
    total = dict(p)
    for k, c in q.items():
        old = total.get(k)
        if old is None:
            total[k] = c
            continue
        bits = max(_bits(old), _bits(c))
        if bits > 255:
            budget.tick(_long(bits))
        new = old + c
        if new.numerator:
            total[k] = new
        else:
            del total[k]
    return total


def _product(budget: Budget, p: Poly, q: Poly) -> Poly:
    """p * q; a product that the budget cannot pay for is refused before its loops start."""
    if not p or not q:
        return {}
    # A product and a sum per pair of terms c, d: on the checker's scale they cost at most about
    # 3*(_long(bits(c)) + _long(bits(d))) beyond their units (the product's gcds on the longer operand, the sum on
    # integers of bits(c) + bits(d) bits, and (u + v)**2 <= 2*u**2 + 2*v**2), so the pairs cost at most
    # 3*(len(q)*sum over p + len(p)*sum over q). Charging every pair at the widest coefficient instead would refuse
    # powers whose coefficients vary widely (the binomials of sin(x)**1500) that the checker verifies in the budget.
    count = len(p) * len(q)
    extra = 3 * (len(q) * sum(_long(_bits(c)) for c in p.values()) + len(p) * sum(_long(_bits(d)) for d in q.values()))
    _afford(budget, 2 * count, extra)
    total: Poly = {}
    for i, c in p.items():
        for j, d in q.items():
            k, term = i + j, c * d
            old = total.get(k)
            if old is None:
                total[k] = term
                continue
            new = old + term
            if new.numerator:
                total[k] = new
            else:
                del total[k]
    return total


def _scaled(budget: Budget, p: Poly, factor: Rational) -> Poly:
    """factor * p for a nonzero factor."""
    if not p:
        return p
    _afford(budget, len(p), len(p) * _long(_widest(p) + _bits(factor)))
    return {k: c * factor for k, c in p.items()}


def _times(budget: Budget, p: Trig, q: Trig) -> Trig:
    """p * q in reduced normal form: (p0 + S*p1)*(q0 + S*q1) = p0*q0 + (1 - C**2)*p1*q1 + S*(p0*q1 + p1*q0)."""
    (p0, p1), (q0, q1) = p, q
    budget.tick()
    square = _product(budget, p1, q1)
    even = _sum(budget, _product(budget, p0, q0), square)
    if square:
        budget.tick(1 + len(square))
        even = _sum(budget, even, {k + 2: -c for k, c in square.items()})
    return even, _sum(budget, _product(budget, p0, q1), _product(budget, p1, q0))


def _power(budget: Budget, base, exponent: int, times):
    """base**exponent for an exponent >= 2 by repeated squaring, with times(budget, p, q) for the products."""
    result = None
    while True:
        budget.tick()
        if exponent & 1:
            result = base if result is None else times(budget, result, base)
        exponent >>= 1
        if not exponent:
            return result
        base = times(budget, base, base)


def _constant_power(budget: Budget, c: Rational, exponent: int) -> Rational:
    """c**exponent for an exponent >= 2 by repeated squaring. 0, 1 and -1 cost nothing more, whatever the exponent; any
    other c**exponent has an integer of more than exponent bits, so an exponent of max_integer_bits or more is refused
    at once."""
    if not c.numerator:
        return _ZERO
    if c.denominator == 1 and c.numerator in (1, -1):
        return c if exponent & 1 else _ONE
    if exponent >= budget.limits.max_integer_bits:
        raise ResourceLimitError("Computation limit exceeded: integer bits")
    result = None
    while True:
        budget.tick(1 + 2 * _long(2 * _bits(c)))
        if exponent & 1:
            result = c if result is None else result * c
        exponent >>= 1
        if not exponent:
            return result
        c = c * c


# --- The antiderivative and its values -----------------------------------------------------------------------------

def _integrate(budget: Budget, polynomial: Trig, frequency: Rational | None) -> tuple[Rational, Poly, Poly]:
    """(lambda, G0, G1) of the antiderivative F = lambda*x + G0(C) + S*G1(C) of P = N0(C) + S*N1(C) without a constant
    term: S*C**j gives -C**(j+1)/((j+1)*a); C**j gives C**(j-1)*S/(j*a) and carries (j-1)/j times its coefficient to
    C**(j-2), from the highest power down, so that the coefficient left at C**0 is lambda. frequency is None only for a
    constant P (only a sine or a cosine fixes a)."""
    even, odd = polynomial
    budget.tick(1 + len(even) + len(odd))
    cosines: Poly = {}
    for j, c in odd.items():
        # Three operations on integers of at most bits(c) + bits(a) + bits(j + 1) bits, those of the quotient.
        _afford(budget, 3, 3 * _long(_bits(c) + _bits(frequency) + (j + 1).bit_length()))
        cosines[j + 1] = -c / (frequency * (j + 1))
    sines: Poly = {}
    carried: Poly = {}
    for j in range(max(even, default=0), 0, -1):
        budget.tick()
        coefficient, carry = even.get(j), carried.pop(j, None)
        if carry is not None:
            coefficient = carry if coefficient is None else coefficient + carry
        if coefficient is None or not coefficient.numerator:
            continue
        _afford(budget, 4, 4 * _long(_bits(coefficient) + _bits(frequency) + j.bit_length()))  # As above.
        sines[j - 1] = coefficient / (frequency * j)
        if j > 1:
            carried[j - 2] = coefficient * trusted_rational(j - 1, j)  # j - 1 and j are coprime.
    slope, carry = even.get(0, _ZERO), carried.pop(0, None)
    if carry is not None:
        slope = slope + carry
    return slope, cosines, sines


def _tree(variable: Symbol, frequency: Rational | None, slope: Rational, cosines: Poly, sines: Poly,
          real_expr: type) -> object:
    """F as the balanced sum of its monomials (the module docstring gives their order and shape), with one u, one
    sin(u), one cos(u) and one power of cos(u) for each exponent shared by the monomials that hold it."""
    x = real_expr("symbol", (), variable)
    terms = [_monomial(slope, x, real_expr)] if slope.numerator else []
    if cosines or sines:
        if (frequency.numerator, frequency.denominator) == (1, 1):
            u = x
        else:
            u = real_expr("mul", (real_expr("scalar", (), frequency), x))
        sine, cosine = real_expr("sin", (u,)), real_expr("cos", (u,))
        powers = {1: cosine}

        def power(k: int) -> object:
            found = powers.get(k)
            if found is None:
                found = powers[k] = real_expr("pow", (cosine,), k)
            return found

        terms += [_monomial(cosines[k], power(k), real_expr) for k in sorted(cosines)]
        terms += [_monomial(sines[k], sine if k == 0 else real_expr("mul", (sine, power(k))), real_expr)
                  for k in sorted(sines)]
    while len(terms) > 1:
        terms = [real_expr("add", (terms[i], terms[i + 1])) if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0] if terms else real_expr("scalar", (), _ZERO)


def _monomial(coefficient: Rational, base: object, real_expr: type) -> object:
    """coefficient*base: the base itself for the coefficient 1, else mul(scalar(coefficient), base)."""
    if coefficient.numerator == 1 and coefficient.denominator == 1:
        return base
    return real_expr("mul", (real_expr("scalar", (), coefficient), base))


def _value_at(budget: Budget, cosines: Poly, sines: Poly, k: int) -> Element:
    """T(k*pi/12) = G0(C) + S*G1(C) at S = sin(k*pi/12), C = cos(k*pi/12), in Q(sqrt 2, sqrt 3)."""
    cosine, sine = _COSINES[k % 24], _COSINES[(6 - k) % 24]
    even = _horner(budget, cosines, cosine)
    return _field_sum(budget, even, _field_product(budget, sine, _horner(budget, sines, cosine)), 1)


def _horner(budget: Budget, coefficients: Poly, point: Element) -> Element:
    """The sum of the c*point**k by Horner's rule, from the highest power down."""
    value = _NOTHING
    for k in range(max(coefficients, default=-1), -1, -1):
        budget.tick()
        value = _field_product(budget, value, point)
        coefficient = coefficients.get(k)
        if coefficient is not None:
            budget.tick(_long(max(_bits(value[0]), _bits(coefficient))))
            value = (value[0] + coefficient, value[1], value[2], value[3])
    return value


def _field_product(budget: Budget, x: Element, y: Element) -> Element:
    """x*y in Q(sqrt 2, sqrt 3), from the products of the nonzero coordinates only (a value of the table has at most
    two). One of x and y is always a value of the table, whose integers have at most 3 bits."""
    pairs = [(i, a, j, b) for i, a in enumerate(x) if a.numerator for j, b in enumerate(y) if b.numerator]
    if not pairs:
        return _NOTHING
    widest = max(max(_bits(a), _bits(b)) for _, a, _, b in pairs)
    # Three operations per pair (a*b, the basis factor, the sum) on integers of at most widest + 6 bits, since one
    # factor is a value of the table: charging 2*widest, as for two long factors, refused integrals that the checker
    # verifies in the budget.
    _afford(budget, 3 * len(pairs), 3 * len(pairs) * _long(widest + 6))
    total: list[Rational | None] = [None, None, None, None]
    for i, a, j, b in pairs:
        index, factor = _BASIS[i][j]
        term = a * b if factor == 1 else a * b * factor
        total[index] = term if total[index] is None else total[index] + term
    return tuple(_ZERO if item is None else item for item in total)


def _field_sum(budget: Budget, x: Element, y: Element, sign: int) -> Element:
    """x + y for sign 1, x - y for sign -1."""
    _afford(budget, 4, 4 * _long(max(max(_bits(a), _bits(b)) for a, b in zip(x, y))))
    return tuple(a + b if sign == 1 else a - b for a, b in zip(x, y))


def _term(renderer, coefficient: Rational, atom: str | None) -> str:
    r"""One term, with a leading "-" when it is negative: the coefficient alone (atom None), or the coefficient times
    the atom: 3*sqrt(3)/8, -sqrt(2), pi/2, 2*pi; in LaTeX \frac{3 \sqrt{3}}{8}, -\sqrt{2}, \frac{\pi}{2}, 2 \pi."""
    numerator, denominator = coefficient.numerator, coefficient.denominator
    sign, top = "-" if numerator < 0 else "", renderer.integer(abs(numerator))
    if atom is not None:
        top = atom if top == "1" else top + (" " if renderer.latex else "*") + atom
    if denominator == 1:
        return sign + top
    bottom = renderer.integer(denominator)
    return sign + (r"\frac{" + top + "}{" + bottom + "}" if renderer.latex else top + "/" + bottom)


def _render(number, renderer, child):
    r"""The rational term, then the multiples of sqrt(2), sqrt(3) and sqrt(6), then the multiple of pi, zero terms left
    out: pi/2, 3*sqrt(3)/8 + pi/6, 4/3, -sqrt(3)/8 + pi/6, 0; in LaTeX \frac{\pi}{2} and
    \frac{3 \sqrt{3}}{8} + \frac{\pi}{6}."""
    latex, algebraic = renderer.latex, number.algebraic
    parts: dict[int, Rational] = {}
    for term in algebraic.terms if type(algebraic) is Add else (algebraic,):
        if type(term) is Rational:
            parts[1] = term
        elif type(term) is Sqrt:
            parts[term.radicand.numerator] = trusted_rational(1, 1)
        else:
            parts[term.factors[1].radicand.numerator] = term.factors[0]
    terms = []
    for radicand in (1, *_RADICANDS):
        coefficient = parts.get(radicand)
        if coefficient is not None and coefficient.numerator:
            atom = None if radicand == 1 else (r"\sqrt{%d}" if latex else "sqrt(%d)") % radicand
            terms.append(_term(renderer, coefficient, atom))
    if number.pi_coefficient.numerator:
        terms.append(_term(renderer, number.pi_coefficient, r"\pi" if latex else "pi"))
    if not terms:
        return "0"
    return renderer.join((term if index == 0 else " - " + term[1:] if term.startswith("-") else " + " + term
                          for index, term in enumerate(terms)), "")


RENDER = {PiLinear: _render}
