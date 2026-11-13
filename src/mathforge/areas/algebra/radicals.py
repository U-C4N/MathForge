"""Denesting of square roots of real quadratic irrationals (mathforge.algebra): sqrt(a + b*sqrt(d)) written as
sqrt(x) + sqrt(y) or sqrt(x) - sqrt(y) with rational x and y.

denest_sqrt(a, b, d) takes exact rationals a and b (Rationals or ints, MCP kind "rational") and an exact integer d
(kind "int"). Its domain (docs/v09/cards/m2-algebra-equations-sets-numeric.md): b != 0, d >= 2 not a perfect square,
and a + b*sqrt(d) >= 0, decided exactly (it is never 0 there, as sqrt(d) is irrational); anything else is invalid
input. A Rational argument is one that the Rational constructor made: two exact ints in lowest terms with a positive
denominator. run() admits every object of the class, also one made with object.__new__ that breaks this form, so
produce() refuses such a Rational as invalid input (the checker would answer False to any claim on it).

The value exists exactly when a > 0 and a^2 - b^2*d = c^2 for a rational c > 0. It is then the DenestedRoot record
(x, y, sign) with x = (a + c)/2, y = (a - c)/2 and sign = sign(b), meaning sqrt(x) + sign*sqrt(y), in the canonical
form that each such number has exactly once: x and y are Rationals (an integral one included, never an int), x > y > 0,
and sign is the int -1 or 1. Otherwise the result is Unevaluated(kind="simplify", reason=Reason("no_applicable_rule",
"a^2 - b^2*d is not a rational square: no denesting of the form sqrt(x) +- sqrt(y)")), which makes no claim.
Evidence: (), always.

The checker, mathforge.certify.algebra.radicals, states the theorem: x + y == a, 4*x*y == b^2*d, sign == sign(b) and
x > y > 0 make (sqrt(x) + sign*sqrt(y))^2 = a + b*sqrt(d) with a positive left side. It also refuses every claim on an
input outside the domain, and refuses at once a claimed x or y whose integers are longer than the true value's can be
(bitlen(A) + bitlen(Da) + bitlen(Db) + 2 bits for a = A/Da and b = B/Db in lowest terms). An argument longer than
max_integer_bits raises ResourceLimitError there, as run() and the verifier refuse it before any checker runs.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/algebra/test_radicals.py is the definition of done.

How produce() computes (the card's closed form; nothing here is a proof, the checker verifies the identities of the
value). With a = A/Da and b = B/Db in lowest terms (an int n is n/1), one integer decides everything:
M = (A*Db)^2 - (B*Da)^2*d = (a^2 - b^2*d)*(Da*Db)^2.

- The domain. a + b*sqrt(d) is never 0, as sqrt(d) is irrational and b != 0. It is positive when neither a nor b is
  negative, negative when neither is positive, and otherwise has the sign of the term of the larger square, a^2 or
  b^2*d: for a > 0 > b it is the sign of M, for b > 0 > a that of -M. A negative a + b*sqrt(d) is invalid input.
- The value. (Da*Db)^2 is a square, so a^2 - b^2*d is the square of a rational c > 0 exactly when M is the square of an
  integer m > 0, and then c = m/(Da*Db); math.isqrt decides it. M = 0 cannot occur: it would make d the square of the
  rational A*Db/(B*Da), and so a perfect square. For a > 0 and M = m^2:
  x = (A*Db + m)/(2*Da*Db), y = (A*Db - m)/(2*Da*Db), sign = sign(B),
  where y > 0 because (A*Db)^2 - m^2 = (B*Da)^2*d > 0, and x > y because m > 0. Each numerator is reduced with the
  denominator by their gcd before its Rational is built, so that only a value whose own integers exceed
  max_integer_bits is refused (ResourceLimitError, as the Rational constructor checks the integers it is given): the
  unreduced A*Db + m is up to twice as long as max_integer_bits allows when d holds a square factor that the
  denominator of b cancels, as for a = A, b = 2/D and d = 2*D^2, where b*sqrt(d) = 2*sqrt(2) and x, y are about as long
  as A.
- No value. a <= 0 (x + y = a > 0 fails; inside the domain M < 0 then), M < 0, or M no perfect square give the
  Unevaluated record of the second paragraph, whose request is the one that operations_v6.run builds for the same
  arguments (the verifier compares them).

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and refuses an argument longer than max_integer_bits
with ResourceLimitError. Every further step is charged before it runs, on the scale of the checker's charges: each
product, exact division and square root one unit and what trusted.upoly.charge_product adds for the lengths of its
operands (a square root of n bits as the product of n bits with n bits), and each gcd one unit and charge_gcd of its
length; the Rational constructor and the record charge their own checks. Where a value exists M has at most 262,144
bits, four times max_integer_bits, and its square root is charged 65,537 units. Under the default limits the charges
of one call add up to at most 421,917 units, every step taken at its longest operands (integers of 65,536 bits,
products of 131,072 and M of 262,144): a fifth of the default max_work of 2,000,000, so that only lower limits run a
request out of work. Measured on one machine, the costliest arguments tried, near max_integer_bits, took about 336,000
units and 0.05 s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import gcd, isqrt

from ...contracts import OperationRequest
from ...deferred import Reason, Unevaluated
from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, computation
from ...model import Add, Rational, mul, sqrt
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

SPECS = (
    OperationSpec("denest_sqrt", "algebra",
                  (Argument("a", "Rational | int", None, "rational"), Argument("b", "Rational | int", None, "rational"),
                   Argument("d", "int", None, "int")),
                  "DenestedRoot",
                  "Denest sqrt(a + b*sqrt(d)) as sqrt(x) + sqrt(y) or sqrt(x) - sqrt(y) with rational x and y.",
                  _SCOPE, version=6),
)


def _integers(budget: Budget, value: object, name: str) -> tuple[int, int]:
    """(numerator, denominator) of a Rational field, which must be exact ints in lowest terms with a positive
    denominator; a Rational made with object.__new__ may lack a field or break that form. ResourceLimitError for an
    integer longer than max_integer_bits."""
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"The {name} of a denested root is a Rational of two exact integers")
    budget.check_int(numerator)
    budget.check_int(denominator)
    words = max(numerator.bit_length(), denominator.bit_length()) >> 6
    budget.tick(1 + (words * words >> 6))  # math.gcd, Lehmer's algorithm: quadratic in the length.
    if gcd(numerator, denominator) != 1:
        raise InvalidInput(f"The {name} of a denested root is a Rational in lowest terms")
    return numerator, denominator


@dataclass(frozen=True)
class DenestedRoot:
    """sqrt(x) + sign*sqrt(y), the denested form of sqrt(a + b*sqrt(d)).

    The canonical form, which each such number has exactly once: x and y are Rationals in lowest terms (an int is no
    Rational here, nor is a bool or a float), x > y > 0, and sign is the exact int -1 or 1. At most one of x and y is
    the square of a rational: both would make b^2*d = 4*x*y a rational square."""
    x: Rational
    y: Rational
    sign: int

    def __post_init__(self) -> None:
        with computation() as budget:
            budget.tick()
            if type(self.x) is not Rational or type(self.y) is not Rational:
                raise InvalidInput("A denested root has Rationals x and y: an int, a float or a bool is no Rational")
            x_numerator, x_denominator = _integers(budget, self.x, "x")
            y_numerator, y_denominator = _integers(budget, self.y, "y")
            if type(self.sign) is not int or self.sign not in (-1, 1):  # A bool is no exact integer.
                raise InvalidInput("The sign of a denested root is the int -1 or 1")
            # x > y by cross-multiplication: two products, charged by their 64-bit words like trusted.upoly's
            # charge_product (which this module does not import: importing mathforge loads no trusted module).
            words = max(x_numerator.bit_length(), x_denominator.bit_length(), y_numerator.bit_length(),
                        y_denominator.bit_length()) >> 6
            budget.tick(2 + 2 * (words * words >> 8))
            if y_numerator <= 0 or x_numerator * y_denominator <= y_numerator * x_denominator:
                raise InvalidInput("A denested root has x > y > 0")


TYPES = {"denested_root": DenestedRoot}
CHECKER = "mathforge.certify.algebra.radicals"

_OPERATION = "denest_sqrt"
_ARGUMENTS = frozenset(("a", "b", "d"))
_NO_DENESTING = "a^2 - b^2*d is not a rational square: no denesting of the form sqrt(x) +- sqrt(y)"  # The card's.
_NEGATIVE = "denest_sqrt requires a + b*sqrt(d) >= 0: the square root of a negative number is not real"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of denest_sqrt, by the closed form of the module docstring: the DenestedRoot, or Unevaluated
    when no denesting exists, each with the evidence (). InvalidInput for an a or b that is no exact rational, a d that
    is no exact int, and an input outside the domain; ResourceLimitError past the budget or max_integer_bits."""
    if type(operation) is not str or operation != _OPERATION:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the radicals family")
    if type(args) is not dict or args.keys() != _ARGUMENTS:
        raise InvalidInput("denest_sqrt takes exactly the arguments a, b and d")
    with computation() as budget:
        budget.tick()  # The call itself.
        big_a, a_denominator = _fraction(budget, args["a"], "a")
        big_b, b_denominator = _fraction(budget, args["b"], "b")
        d = args["d"]
        if type(d) is not int:  # Neither a bool nor a float nor a Rational is an exact integer.
            raise InvalidInput("denest_sqrt requires an exact integer d")
        budget.check_int(d)
        if big_b == 0:
            raise InvalidInput("denest_sqrt requires b != 0: sqrt(a + 0*sqrt(d)) has no inner square root")
        if d < 2:
            raise InvalidInput("denest_sqrt requires d >= 2: sqrt(d) is rational or not real for d < 2")
        _charge(budget, d.bit_length(), d.bit_length(), 2)  # The square root of d and its square.
        root = isqrt(d)
        if root * root == d:
            raise InvalidInput("denest_sqrt requires d that is no perfect square")
        if big_a <= 0 and big_b < 0:  # Neither term is positive: refused before any product (the test below
            raise InvalidInput(_NEGATIVE)  # would refuse it too).
        centre, excess = _square_difference(budget, big_a, a_denominator, big_b, b_denominator, d)
        if (big_b < 0 and excess < 0) or (big_a < 0 and excess > 0):
            raise InvalidInput(_NEGATIVE)  # Terms of opposite signs, and the negative one has the larger square.
        value = None
        if big_a > 0 and excess > 0:  # Otherwise x + y = a > 0 or a^2 - b^2*d = c^2 fails: no denesting.
            value = _denested(budget, centre, a_denominator, b_denominator, 1 if big_b > 0 else -1, excess)
        return (_unevaluated(args) if value is None else value), ()


def _charge(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, count products, exact divisions or square roots of integers of the given bit lengths:
    one unit each and what their lengths cost beyond it (trusted.upoly.charge_product)."""
    from ...trusted.upoly import charge_product  # Importing mathforge loads no trusted module.
    budget.tick(count)
    charge_product(left_bits, right_bits, count)


def _charged_gcd(budget: Budget, left: int, right: int) -> int:
    """gcd(left, right), charged before it runs: one unit and trusted.upoly.charge_gcd of the longer length."""
    from ...trusted.upoly import charge_gcd
    budget.tick()
    charge_gcd(max(left.bit_length(), right.bit_length()))
    return gcd(left, right)


def _fraction(budget: Budget, value: object, name: str) -> tuple[int, int]:
    """(numerator, denominator) of an exact rational argument: an int over 1, or a Rational of two exact ints in lowest
    terms with a positive denominator, which only a Rational made with object.__new__ can fail to be (run() admits
    every object of the class). InvalidInput for anything else, ResourceLimitError for an integer longer than
    max_integer_bits."""
    if type(value) is int:  # A bool is no exact integer.
        budget.check_int(value)
        return value, 1
    if type(value) is not Rational:
        raise InvalidInput(f"denest_sqrt requires an exact rational {name}: an int or a Rational")
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"denest_sqrt requires a Rational {name} of two exact integers with a positive denominator")
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator != 1 and _charged_gcd(budget, numerator, denominator) != 1:
        raise InvalidInput(f"denest_sqrt requires a Rational {name} in lowest terms")
    return numerator, denominator


def _square_difference(budget: Budget, big_a: int, a_denominator: int, big_b: int, b_denominator: int,
                       d: int) -> tuple[int, int]:
    """(A*Db, M) with M = (A*Db)^2 - (B*Da)^2*d = (a^2 - b^2*d)*(Da*Db)^2, each product charged before it runs."""
    _charge(budget, big_a.bit_length(), b_denominator.bit_length())
    centre = big_a * b_denominator
    _charge(budget, big_b.bit_length(), a_denominator.bit_length())
    scaled = big_b * a_denominator
    _charge(budget, centre.bit_length(), centre.bit_length())
    _charge(budget, scaled.bit_length(), scaled.bit_length())
    _charge(budget, 2 * scaled.bit_length(), d.bit_length())
    budget.tick()  # The subtraction, linear in the length.
    return centre, centre * centre - scaled * scaled * d


def _denested(budget: Budget, centre: int, a_denominator: int, b_denominator: int, sign: int,
              excess: int) -> DenestedRoot | None:
    """For a > 0, centre = A*Db and M = excess > 0, the DenestedRoot when M is a perfect square m^2:
    x = (A*Db + m)/(2*Da*Db) and y = (A*Db - m)/(2*Da*Db), each reduced before its Rational is built, and the sign of
    b; None when M is no perfect square."""
    bits = excess.bit_length()
    _charge(budget, bits, bits)  # The square root, charged as the product of M with itself.
    m = isqrt(excess)
    _charge(budget, m.bit_length(), m.bit_length())
    if m * m != excess:
        return None
    _charge(budget, a_denominator.bit_length() + 1, b_denominator.bit_length())
    denominator = 2 * a_denominator * b_denominator
    budget.tick(2)  # The two additions below, linear in the length.
    parts = []
    for numerator in (centre + m, centre - m):
        divisor = _charged_gcd(budget, numerator, denominator)
        _charge(budget, max(numerator.bit_length(), denominator.bit_length()), divisor.bit_length(), 2)
        parts.append(Rational(numerator // divisor, denominator // divisor))
    return DenestedRoot(parts[0], parts[1], sign)


def _unevaluated(args: dict[str, object]) -> Unevaluated:
    """The record of a request inside the domain that has no denesting, with the request that operations_v6.run
    builds for the same arguments (OperationRequest sorts them, and its version takes no part in equality)."""
    from ...operations_v6 import REQUEST_VERSION
    return Unevaluated("simplify", OperationRequest(_OPERATION, tuple(args.items()), REQUEST_VERSION),
                       Reason("no_applicable_rule", _NO_DENESTING))


def _render(root: DenestedRoot, renderer: object, child: Callable[..., str]) -> str:
    r"""The legacy expression sqrt(x) + sign*sqrt(y), built with mathforge.model so that a rational square root
    shows as a rational, in the order x, then y (Add keeps it; add() would sort the terms): sqrt(2) + 1,
    sqrt(3) + sqrt(2), 2 - sqrt(3), sqrt(3/2) + sqrt(1/2); in LaTeX \sqrt{2} + 1 and 2 - \sqrt{3}."""
    return child(Add((sqrt(root.x), mul(root.sign, sqrt(root.y)))))


RENDER = {DenestedRoot: _render}
