"""Independent checker of the family mathforge.areas.algebra.radicals: denest_sqrt.

It never computes a denesting itself (no square root of a^2 - b^2*d is taken): it verifies the identities that the
claimed record must satisfy, with exact integer arithmetic.

denest_sqrt(a, b, d): value DenestedRoot(x, y, sign), evidence ().
The claim, for exact rationals a and b (an int, or a Rational in lowest terms with a positive denominator) and an
exact int d, with b != 0, d >= 2 no perfect square and a + b*sqrt(d) >= 0: sqrt(a + b*sqrt(d)) = sqrt(x) + sign*sqrt(y),
the value in its canonical form: x and y Rationals in lowest terms (an int is no Rational here), x > y > 0, sign the
exact int -1 or 1. True requires
  (0) the arguments: exactly a, b and d, with b != 0 and d >= 2 no perfect square (isqrt(d)^2 != d); a record of the
      class (module, qualname) == RECORD, with x and y exact Rationals of exact ints, positive denominators and
      lowest terms, and sign an exact int in {-1, 1}; the evidence the empty tuple;
  (1) x + y == a;
  (2) 4*x*y == b^2*d;
  (3) sign == sign(b);
  (4) x > y > 0.
Theorem. By (4), sqrt(x) > sqrt(y) > 0, so s = sqrt(x) + sign*sqrt(y) > 0 for either sign. By (1), (2) and (3),
s^2 = x + y + 2*sign*sqrt(x*y) = a + 2*sign*sqrt(b^2*d/4) = a + sign*|b|*sqrt(d) = a + b*sqrt(d). So a + b*sqrt(d) = s^2
is positive and s is its positive square root: the value equals sqrt(a + b*sqrt(d)). It is the canonical value: (1) and
(2) make x and y the two roots of t^2 - a*t + b^2*d/4, which x > y orders, and (3) fixes the sign, so no other
canonical record passes. Conversely the true value passes: sqrt(x) + sign*sqrt(y) = sqrt(a + b*sqrt(d)) squared gives
x + y - a = b*sqrt(d) - 2*sign*sqrt(x*y); squaring again, (x + y - a)^2 + b^2*d - 4*x*y = 2*(x + y - a)*b*sqrt(d), and
as sqrt(d) is irrational and b != 0, x + y = a, then 4*x*y = b^2*d, and sign = sign(b) since sign*sqrt(x*y) has the
sign of b; x > y > 0 and lowest terms are its canonical form.
The identities (1) to (4) alone also hold for some claims on inputs outside the domain, where the operation has no
value at all: sqrt(5 + 2*sqrt(4)) = sqrt(4) + sqrt(1) satisfies them for d = 4. So (0) refuses b = 0, d < 2 and a
perfect square d, as the producer refuses those inputs. a + b*sqrt(d) < 0 needs no test of its own: (1) to (4) make it
the square s^2 > 0.

Sizes. With a = A/Da and b = B/Db in lowest terms, a value exists only when m^2 = A^2*Db^2 - B^2*Da^2*d for an integer
m > 0, and then x, y = (A*Db +- m)/(2*Da*Db): their reduced denominators divide 2*Da*Db, and their numerators are below
2*A*Db, since 0 < y < x < a. A claimed x or y with an integer longer than bitlen(A) + bitlen(Da) + bitlen(Db) + 2 bits
is therefore no true value, and is refused before any arithmetic: an oversized forged value answers False at once, and
every product that follows has at most a few times the size of the arguments. Each product, square root and gcd is
charged to the budget before it runs (trusted.upoly.charge_product and charge_gcd), and so is every step: there is no
other loop. No builtin pow runs, and no ** either: squares are products. An argument longer than max_integer_bits
raises ResourceLimitError, as run() and the verifier refuse it before any checker runs: such a claim is beyond the
budget, not wrong. Anything else malformed (a record forged without __init__ or with a missing field, a bool, a float,
a str or None where a number belongs, a list for a tuple, any evidence but ()) answers False.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) with RECORD instead. tools/check_layering.py follows every import, and
tests/v08/test_layering.py rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd, isqrt

from mathforge.limits import Budget, computation
from mathforge.model import Rational
from mathforge.trusted.upoly import charge_gcd, charge_product

OPERATION = "denest_sqrt"
RECORD = ("mathforge.areas.algebra.radicals", "DenestedRoot")  # Module and name of the value's class.
ARGUMENTS = frozenset(("a", "b", "d"))
_MISSING = object()  # A field that a record made with object.__new__ lacks.


def _charge(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, count products or square roots of integers of the given bit lengths: one unit each
    and what their lengths cost beyond it (trusted.upoly.charge_product)."""
    budget.tick(count)
    charge_product(left_bits, right_bits, count)


def _times(budget: Budget, *factors: int) -> int:
    """The product of the factors, each multiplication charged before it runs."""
    product = factors[0]
    for factor in factors[1:]:
        _charge(budget, product.bit_length(), factor.bit_length())
        product *= factor
    return product


def _lowest(budget: Budget, numerator: int, denominator: int) -> bool:
    """Whether numerator/denominator, with denominator >= 1, is in lowest terms; the gcd is charged before it runs."""
    if denominator == 1:
        return True
    budget.tick()
    charge_gcd(max(numerator.bit_length(), denominator.bit_length()))
    return gcd(numerator, denominator) == 1


def _rational(value: object) -> tuple[int, int] | None:
    """(numerator, denominator) of a Rational, each field read once, when both are exact ints and the denominator is
    positive; None for any other class (an int and a class of another module named Rational included) and for a
    Rational made with object.__new__ that lacks a field or holds anything else."""
    if type(value) is not Rational:
        return None
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:  # A bool is no exact int.
        return None
    return numerator, denominator


def _argument(budget: Budget, value: object) -> tuple[int, int] | None:
    """(numerator, denominator) of an exact rational argument, an int (over 1) or a Rational in lowest terms; None
    for anything else. ResourceLimitError for an integer longer than max_integer_bits."""
    fraction = (value, 1) if type(value) is int else _rational(value)
    if fraction is None:
        return None
    budget.check_int(fraction[0])
    budget.check_int(fraction[1])
    return fraction if _lowest(budget, *fraction) else None


def _record_fields(value: object) -> tuple[object, ...] | None:
    """(x, y, sign) of a value of the record class, each read once; None for any other class or for a record that
    lacks a field. The class is identified by its module and name, never imported."""
    cls = type(value)
    module, name = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    if type(module) is not str or type(name) is not str or (module, name) != RECORD:
        return None
    x, y, sign = getattr(value, "x", _MISSING), getattr(value, "y", _MISSING), getattr(value, "sign", _MISSING)
    if x is _MISSING or y is _MISSING or sign is _MISSING:  # Identity tests: a forged field may compare oddly.
        return None
    return x, y, sign


def _check_denest(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    """Conditions (0) to (4) of the module docstring, the size bound before any arithmetic on the value."""
    if args.keys() != ARGUMENTS or evidence:
        return False
    a, b, d = _argument(budget, args["a"]), _argument(budget, args["b"]), args["d"]
    if a is None or b is None or type(d) is not int:
        return False
    (big_a, a_denominator), (big_b, b_denominator) = a, b
    if big_b == 0 or d < 2:
        return False  # Invalid input: no claim on it is true.
    budget.check_int(d)
    _charge(budget, d.bit_length(), d.bit_length())  # isqrt(d), charged like a product of d with itself.
    root = isqrt(d)
    if _times(budget, root, root) == d:
        return False  # A perfect square d is invalid input.
    if big_a <= 0:
        return False  # x + y == a > 0 for x > y > 0: no value exists.
    fields = _record_fields(value)
    if fields is None:
        return False
    x, y, sign = fields
    x_fraction, y_fraction = _rational(x), _rational(y)
    if x_fraction is None or y_fraction is None or type(sign) is not int or sign not in (-1, 1):
        return False
    (x_numerator, x_denominator), (y_numerator, y_denominator) = x_fraction, y_fraction
    room = big_a.bit_length() + a_denominator.bit_length() + b_denominator.bit_length() + 2
    if max(x_numerator.bit_length(), x_denominator.bit_length(), y_numerator.bit_length(),
           y_denominator.bit_length()) > room:
        return False  # Longer than any integer of the true value (module docstring): no arithmetic on it.
    if not _lowest(budget, x_numerator, x_denominator) or not _lowest(budget, y_numerator, y_denominator):
        return False  # Not the canonical form.
    if sign != (1 if big_b > 0 else -1):
        return False  # Condition 3.
    x_scaled, y_scaled = _times(budget, x_numerator, y_denominator), _times(budget, y_numerator, x_denominator)
    if y_numerator <= 0 or x_scaled <= y_scaled:
        return False  # Condition 4: y > 0 and x > y, the denominators being positive.
    # Condition 1, cross-multiplied: (X*Dy + Y*Dx)*Da == A*Dx*Dy.
    if _times(budget, x_scaled + y_scaled, a_denominator) != _times(budget, big_a, x_denominator, y_denominator):
        return False
    # Condition 2, cross-multiplied: 4*X*Y*Db^2 == B^2*d*Dx*Dy.
    return (_times(budget, 4, x_numerator, y_numerator, b_denominator, b_denominator)
            == _times(budget, big_b, big_b, d, x_denominator, y_denominator))


def check_operation(operation: object, args: object, value: object, evidence: object) -> bool:
    """Whether value, with evidence, is the verified answer of denest_sqrt on args, by the theorem of the module.
    False for another operation, for arguments other than exactly a, b and d, and for any malformed value or
    evidence; ResourceLimitError for an argument longer than max_integer_bits or past the budget."""
    if type(operation) is not str or operation != OPERATION or type(args) is not dict or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        return _check_denest(budget, args, value, evidence)
