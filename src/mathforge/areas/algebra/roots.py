"""Roots of exact rational polynomials (mathforge.algebra): distinct real roots counted in a closed interval, and every
complex root with its multiplicity when the irreducible factors over Q have degree at most two.

count_real_roots(polynomial, lower='-inf', upper='+inf') takes a nonzero Polynomial and the ends of the closed
interval [lower, upper]: each end an exact int or Rational, or '-inf' for lower and '+inf' for upper (an infinite end
is open by nature), with lower <= upper when both are finite. Anything else is invalid input: the zero polynomial, an
end of another type (a bool, a float, a str other than the infinity of its side, an irrational Expression), '+inf' as
lower or '-inf' as upper, and lower > upper. The value is the int number of distinct real roots in [lower, upper]; a
nonzero constant has none. Evidence: ().

complex_roots(polynomial) takes a nonzero Polynomial; the zero polynomial is invalid input. When every irreducible
factor of the polynomial over Q has degree one or two, the value is the tuple of v0.7's
mathforge.complex_numbers.ComplexRoot records (fields root, a ComplexAlgebraic, and multiplicity, a positive int), one
per distinct complex root, sorted by real part, then by imaginary part, increasing; a nonzero constant gives (). Each
part of a root is in the canonical form that v0.7's quadratic_roots builds: a Rational when it is rational, else the
RealAlgebraicRoot of its minimal polynomial (primitive, with a positive leading coefficient) and its index among that
polynomial's real roots. For the monic factor x**2 + b*x + c with D = b**2 - 4*c, the roots are (-b -+ sqrt(D))/2,
each with the real part RealAlgebraicRoot(F, 0), RealAlgebraicRoot(F, 1) (F the primitive integer form of the factor)
and imaginary part 0 when D > 0, and the real part -b/2 with the imaginary parts -+ sqrt(-D)/2 when D < 0 (Rationals
when -D is a rational square, else RealAlgebraicRoot(G, 0) and RealAlgebraicRoot(G, 1), G the primitive integer form
of 4*y**2 + D). The multiplicity of every root of a factor is the factor's exponent. When an irreducible factor has
degree three or more, the value is Unevaluated(kind="solve", reason=Reason("no_applicable_rule", "an irreducible
factor of degree three or more")), which is no claim. Evidence: (lc, factors), where lc is the leading coefficient of
the polynomial, a Rational, and factors is ((coefficients, exponent), ...): one entry per distinct monic irreducible
factor, its coefficients a tuple of Rationals lowest degree first (two or three of them, the last Rational(1)) and its
exponent a positive int, the entries in strictly increasing order of (degree, ((numerator, denominator) of each
coefficient, lowest degree first)), the order of polynomial_exact.factor_with_evidence; (lc, ()) for a constant.

The family defines no record class: count_real_roots answers an int and complex_roots v0.7's records, which serialize
and render without one, so TYPES is empty and there is no RENDER. Its checker, mathforge.certify.algebra.roots, states
the theorems and the exact layouts that it accepts. mathforge.areas.ntheory.continued_fraction spells out the contract
of a v6 family, documentation/families.md is the checklist, and tests/v09/algebra/test_roots.py is the definition of
done (docs/v09/cards/m2-algebra-equations-sets-numeric.md is the specification).

How produce() computes (the card's producers). Nothing here is a proof: the checker verifies every value with
algorithms of its own (its own square-free part and Sturm chain, the product identity of the factors, the closed forms
of the roots substituted in Q(sqrt(d)), exact comparisons of surds).

- The inputs. produce() validates every argument itself with exact type tests, also the polynomial, whose kind
  operations_v6.run checks: run() admits every object of the Polynomial and Rational classes, also one made with
  object.__new__ that breaks the canonical form, and checks nothing for the point kind of lower and upper. The
  polynomial must be a Polynomial with a Symbol variable and a nonempty tuple of at most 1,025 Rationals in lowest
  terms with positive denominators, without a trailing zero, and not the zero polynomial; an end must be an int, such a
  Rational, or the infinity of its side. Anything else is invalid input. Once the inputs are validated, a library
  function that raises InvalidInput has a bug, which produce() reports as an ArithmeticError (the producer's internal
  error), never as rejected input.
- count_real_roots is root_isolation.count_real_roots: v0.7's signed primitive Sturm chain of the square-free part
  (mathforge.kernels.upoly) counts the roots in the open interval (lower, upper), and each finite end that is a root is
  added; the point interval [a, a] counts a when it is a root. A nonzero constant has the chain ((1,),) and no root.
- complex_roots factors the polynomial over Q with polynomial_exact.factor_with_evidence (v0.7's square-free
  decomposition, Berlekamp's algorithm, Hensel lifting and Zassenhaus recombination on integer tuples): its leading
  coefficient and its distinct monic irreducible factors with their exponents, already in the order of the evidence. A
  factor of degree three or more gives the Unevaluated record of the second paragraph, with the evidence (); its
  request is the one that operations_v6.run builds for the same arguments, which the verifier compares. Otherwise the
  linear factor x + a gives the record of the rational root -a with the imaginary part Rational(0), and the quadratic
  factor x**2 + b*x + c the two records of complex_numbers._quadratic(1, b, c), v0.7's quadratic formula through the
  algebraic normalizer, which are what mf.quadratic_roots(1, b, c) builds; each record takes the factor's exponent as
  its multiplicity. The records are sorted by algebraic.compare_real on the real parts, then on the imaginary parts,
  inside one root_work context, so that each isolating interval is computed once for the whole sort. The roots of
  distinct factors differ, so no two records compare equal. The irreducibility evidence of factor_with_evidence
  ("qq-factorization-v1") stays out of the evidence.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and one for each coefficient of the polynomial,
refuses an integer longer than max_integer_bits with ResourceLimitError, and charges each gcd of the validation one unit
and trusted.upoly.charge_gcd of its length before it runs; every valid input costs at least two units. The library
functions charge their own work to the same budget: the Sturm chain and its evaluations, the factorization (the
Berlekamp matrix of each factor a_i of the square-free decomposition p = c * prod a_i**i counts deg(a_i)**2 entries
against max_matrix_entries, 4,096 by default, so an a_i of degree 65 or more is beyond the default limits, and the
Zassenhaus recombination of many modular factors can exhaust max_work before that; produce() charges beforehand the
Hensel liftings that the kernels leave uncharged, estimated by Mignotte's bound for the square-free parts of x * q
where p = x**k * q, so that a power of x such as x**1024 is charged no lifting), the algebraic normalizer of each
quadratic factor (about 4,000 units for a small factor with irrational roots, about 120 for one with rational parts)
and the refinements of the sort. Measured on one machine: count_real_roots of a dense polynomial of degree 160 with
one-digit coefficients takes about 560,000 units (0.2 s; the checker about 165,000), and of degree 256 about 4.5
million, beyond the default max_work of 2,000,000; complex_roots of a polynomial of degree about 80 with 24 quadratic
and 13 linear factors takes 0.9 to 1 million units (0.3 s; the checker under 10,000), most of them in the
factorization.
"""

from __future__ import annotations

from functools import cmp_to_key
from math import gcd

from ...contracts import OperationRequest
from ...deferred import Reason, Unevaluated
from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Rational, Symbol
from ...polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
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

# The scope text of every operation of wave M2 (docs/v09/cards/m2-algebra-equations-sets-numeric.md).
_SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
          "and decimal digits")
# The annotation of the point kind: an exact rational end, or the infinity of its side.
_POINT = "Rational | int | Literal['-inf', '+inf']"

SPECS = (
    OperationSpec("count_real_roots", "algebra",
                  (Argument("polynomial", "Polynomial", None, "polynomial"),
                   Argument("lower", _POINT, "'-inf'", "point"), Argument("upper", _POINT, "'+inf'", "point")),
                  "int", "Count the distinct real roots of a rational polynomial in a closed interval with rational or "
                  "infinite ends.", _SCOPE, version=6),
    OperationSpec("complex_roots", "algebra",
                  (Argument("polynomial", "Polynomial", None, "polynomial"),),
                  "tuple[ComplexRoot, ...]", "List every complex root with multiplicity of a rational polynomial whose "
                  "irreducible factors have degree at most two.", _SCOPE, version=6),
)

# No record classes: count_real_roots answers an int, complex_roots a tuple of v0.7's ComplexRoot records, which the
# registry does not let a v6 family claim as its own (every TYPES class must be defined in the family module).
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.algebra.roots"

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
_INFINITIES = {"lower": "-inf", "upper": "+inf"}  # The infinity that each end admits.
_NOT_A_POLYNOMIAL = "{} requires a Polynomial"
_NOT_CANONICAL = ("{} requires a canonical Polynomial: a Symbol variable and a nonempty tuple of at most 1,025 "
                  "Rationals in lowest terms with positive denominators, without a trailing zero")
_ZERO_POLYNOMIAL = "{} requires a nonzero polynomial: every number is a root of the zero polynomial"
_NOT_AN_END = "count_real_roots requires {} to be an int, a Rational or '{}'"
_NOT_CANONICAL_END = ("count_real_roots requires a Rational {} of two exact integers in lowest terms with a positive "
                      "denominator")
_REVERSED = "count_real_roots requires lower <= upper"
_DEFERRAL = "an irreducible factor of degree three or more"  # The card's reason for an Unevaluated complex_roots.


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name; the
    module states the algorithms and the evidence. InvalidInput for an argument of the wrong type or outside the
    operation's domain, ResourceLimitError past the budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the roots family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    with computation() as budget:
        budget.tick()  # The call itself; the validation charges at least one unit more.
        polynomial = _polynomial(budget, args["polynomial"], operation)
        lower = upper = None
        if operation == "count_real_roots":
            lower = _end(budget, args["lower"], "lower")
            upper = _end(budget, args["upper"], "upper")
            if lower is not None and upper is not None and lower > upper:
                raise InvalidInput(_REVERSED)
        try:
            if operation == "count_real_roots":
                return _count_real_roots(polynomial, lower, upper), ()
            return _complex_roots(budget, polynomial, args)
        except InvalidInput as error:  # Every input is validated: a refusal now is a bug, never rejected input.
            raise ArithmeticError(f"{operation}: a library function refused validated input") from error


def _integers(budget: Budget, value: object, message: str) -> tuple[int, int]:
    """(numerator, denominator) of a Rational, which must hold two exact ints in lowest terms with a positive
    denominator: one made with object.__new__ may lack a field or break that form, InvalidInput(message) then.
    ResourceLimitError for an integer longer than max_integer_bits, before the gcd that its length bounds."""
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(message)
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator != 1:
        from ...trusted.upoly import charge_gcd  # Importing mathforge loads no trusted module.
        budget.tick()
        charge_gcd(max(numerator.bit_length(), denominator.bit_length()))
        if gcd(numerator, denominator) != 1:
            raise InvalidInput(message)
    return numerator, denominator


def _polynomial(budget: Budget, polynomial: object, operation: str) -> Polynomial:
    """The polynomial argument once it is a canonical nonzero Polynomial (module docstring); InvalidInput otherwise.
    One unit for each coefficient."""
    if type(polynomial) is not Polynomial:
        raise InvalidInput(_NOT_A_POLYNOMIAL.format(operation))
    coefficients, variable = getattr(polynomial, "coefficients", None), getattr(polynomial, "variable", None)
    if (type(variable) is not Symbol or type(coefficients) is not tuple or not coefficients
            or len(coefficients) > MAX_POLYNOMIAL_DEGREE + 1):
        raise InvalidInput(_NOT_CANONICAL.format(operation))
    message = _NOT_CANONICAL.format(operation)
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not Rational:  # Neither an int nor a float nor a Rational subclass.
            raise InvalidInput(message)
        _integers(budget, coefficient, message)
    if not coefficients[-1].numerator:
        raise InvalidInput((_ZERO_POLYNOMIAL if len(coefficients) == 1 else _NOT_CANONICAL).format(operation))
    _common_denominator(budget, coefficients)
    return polynomial


def _common_denominator(budget: Budget, coefficients: tuple[Rational, ...]) -> None:
    """Refuse with ResourceLimitError, as the checker does, validated coefficients whose least common denominator is
    longer than max_integer_bits (module docstring, Work). One unit and the gcd's charge for each denominator that
    changes it."""
    from ...trusted.upoly import charge_gcd

    limit = budget.limits.max_integer_bits
    common = 1
    for coefficient in coefficients:
        denominator = coefficient.denominator
        if denominator != 1 and common % denominator:
            budget.tick()
            charge_gcd(max(common.bit_length(), denominator.bit_length()))
            common = common // gcd(common, denominator) * denominator
            if common.bit_length() > limit:
                raise ResourceLimitError("Computation limit exceeded: integer bits")


def _end(budget: Budget, end: object, name: str) -> Rational | None:
    """An end of the interval of count_real_roots: the Rational of an int or of a canonical Rational, None for the
    infinity of its side; InvalidInput for anything else, the other infinity included."""
    infinity = _INFINITIES[name]
    if type(end) is str:
        if end != infinity:
            raise InvalidInput(_NOT_AN_END.format(name, infinity))
        return None
    if type(end) is int:  # A bool is no exact integer.
        budget.check_int(end)
        return Rational(end)
    if type(end) is not Rational:  # A float, an irrational Expression, a Rational subclass, None, a container.
        raise InvalidInput(_NOT_AN_END.format(name, infinity))
    _integers(budget, end, _NOT_CANONICAL_END.format(name))
    return end


def _count_real_roots(polynomial: Polynomial, lower: Rational | None, upper: Rational | None) -> int:
    """The distinct real roots of the polynomial in [lower, upper], None for an infinite end: the open interval and
    each finite end that is a root."""
    from ... import root_isolation

    return root_isolation.count_real_roots(polynomial, lower, upper, include_lower=lower is not None,
                                           include_upper=upper is not None)


def _charge_lifting(budget: Budget, coefficients: tuple[Rational, ...]) -> None:
    """Charge, before factor_with_evidence runs, the work of its Hensel liftings that its kernels leave uncharged
    (module docstring, Work). The liftings run on the square-free parts of degree two or more, never on the whole
    polynomial: for p = x**k * q with q(0) != 0, each such part divides x * q, so the estimate is that of q with one
    degree more when k > 0. A power of x is thus charged nothing (x**1024 costs the factorization about 1.1 million
    units, the estimate of the whole polynomial billions). A part of degree one or less is factored without a
    lifting."""
    shift = 0
    while not coefficients[shift].numerator:  # A nonzero polynomial: the loop stops at its leading coefficient.
        shift += 1
    coefficients = coefficients[shift:]
    degree = len(coefficients) - 1 + (1 if shift else 0)  # The largest degree of a square-free part of x * q.
    if degree < 2:
        return
    from ...kernels.upoly import rational_primitive
    from ...trusted.upoly import max_bits

    # Q, the primitive integer form of q: P = x**k * Q has the same nonzero coefficients, and a divisor of P of
    # degree m has coefficients below 2**m * ||P||_2 = 2**m * ||Q||_2 (Mignotte), with m <= degree for every part.
    integers = rational_primitive(coefficients)
    lead = integers[-1].bit_length()
    # Mignotte's bound: a divisor of P has coefficients below 2**degree * ||P||_2, so no lifting has a modulus longer
    # than `bits` bits or more than `bits` steps.
    bits = lead + 2 * degree + max_bits(integers) + ((degree + 1).bit_length() + 1) // 2 + 3
    inverse = (lead >> 6) ** 2  # One pow(lc, -1, modulus) per step: Euclid on integers of up to `lead` bits.
    products = 2 * (degree + 1) ** 2 * (((bits >> 6) ** 2) >> 8) // 3  # charge_product's extra, the moduli growing.
    budget.tick(bits * (2 * inverse + products))


def _complex_roots(budget: Budget, polynomial: Polynomial, args: dict[str, object]) -> tuple[object, tuple]:
    """(roots, (lc, factors)) of a nonzero polynomial whose irreducible factors over Q have degree at most two, and
    (Unevaluated, ()) for one with a factor of degree three or more (module docstring)."""
    from ...algebraic import compare_real
    from ...complex_numbers import ComplexRoot, _quadratic, _record
    from ...polynomial_exact import factor_with_evidence
    from ...root_isolation import root_work

    _charge_lifting(budget, polynomial.coefficients)
    factorization, _ = factor_with_evidence(polynomial)  # v0.7's irreducibility evidence is no part of ours.
    factors = factorization.factors
    if any(len(factor.polynomial.coefficients) > 3 for factor in factors):
        return _unevaluated(args), ()
    zero, one = Rational(0), Rational(1)
    roots = []
    for factor in factors:
        coefficients, multiplicity = factor.polynomial.coefficients, factor.multiplicity
        if len(coefficients) == 2:  # x + a: the canonical Rational -a, set as _quadratic's records set their parts.
            roots.append(ComplexRoot(_record(-coefficients[0], zero), multiplicity))
        else:  # x**2 + b*x + c, irreducible: two simple roots of a nonzero discriminant.
            pair, _ = _quadratic(one, coefficients[1], coefficients[0])
            roots.extend(ComplexRoot(root.root, multiplicity) for root in pair)

    def order(left: ComplexRoot, right: ComplexRoot) -> int:
        real = compare_real(left.root.real, right.root.real)
        return real if real else compare_real(left.root.imag, right.root.imag)

    with root_work():  # One cache of isolating intervals for every comparison of the sort.
        roots.sort(key=cmp_to_key(order))
    entries = tuple((factor.polynomial.coefficients, factor.multiplicity) for factor in factors)
    return tuple(roots), (factorization.coefficient, entries)


def _unevaluated(args: dict[str, object]) -> Unevaluated:
    """The card's record of a polynomial with an irreducible factor of degree three or more, with the request that
    operations_v6.run builds for the same arguments (OperationRequest sorts them, and its version takes no part in
    equality)."""
    from ...operations_v6 import REQUEST_VERSION

    return Unevaluated("solve", OperationRequest("complex_roots", tuple(args.items()), REQUEST_VERSION),
                       Reason("no_applicable_rule", _DEFERRAL))
