"""Independent checker of linear_diophantine (family mathforge.areas.ntheory.diophantine).

The claim, for exact integers a, b, c with (a, b) != (0, 0): the value DiophantineSolution(x, y, dx, dy) says that
the integer solutions of a*X + b*Y = c are exactly the pairs (x + t*dx, y + t*dy), t in Z, written in the canonical
form; the value with four None fields says that there is none. The checker never solves the equation: it verifies
four conditions on a solvable claim and a divisibility certificate for an unsolvable one.

Solvable claim, evidence (). True requires
  (1) a*x + b*y == c,
  (2) a*dx + b*dy == 0,
  (3) gcd(dx, dy) == 1,
  (4) the canonical form: dx > 0 and 0 <= x < dx, or dx == 0, dy == 1 and y == 0.
Theorem: with g = gcd(a, b) > 0, the integer solutions of a*X + b*Y = 0 are exactly the multiples t*(b/g, -a/g),
t in Z: a/g and b/g are coprime, so (a/g)*X = -(b/g)*Y forces b/g to divide X (for b = 0: X = 0, Y free). By (2),
(dx, dy) = s*(b/g, -a/g) for an integer s, and gcd(dx, dy) = |s| because gcd(b/g, a/g) = 1, so (3) gives s = +-1:
the multiples of (dx, dy) are all the solutions of the homogeneous equation. With the solution (x, y) of (1), the
solutions of a*X + b*Y = c are then exactly (x, y) + t*(dx, dy). (4) selects one of the representations of that set:
for b != 0, dx = +-b/g is nonzero, dx > 0 fixes the sign of the step and 0 <= x < dx the particular solution; for
b == 0, dx = 0 and dy = 1 fix the step and y == 0 the particular solution. So the true value passes, and no other.

Unsolvable claim, four None fields, evidence (g,). True requires g >= 1, a % g == 0, b % g == 0 and c % g != 0.
Theorem: g then divides a*X + b*Y for all integers X, Y, but not c, so the equation has no integer solution. Every
unsolvable equation has such a g: by Bezout, a*X + b*Y takes exactly the multiples of gcd(a, b), so the equation is
unsolvable exactly when gcd(a, b) does not divide c, and g = gcd(a, b) serves. Any other g that passes is accepted.

The equation with a = b = 0 is refused whatever the claim: the producer rejects it as invalid input, and the theorem
above needs (a, b) != (0, 0). (For 0*X + 0*Y = 0 every pair is a solution, yet (1) to (4) hold for (0, 0, 1, 0).)

Sizes and work. The components of the true value are bounded by the arguments: 0 <= dx <= |b|, |dy| <= max(|a|, 1),
|x| <= max(|b|, |c|), |y| <= |a| + |c| (for b != 0, y = (c - a*x)/b with |x| < |b|), and a g that divides a and b,
one of them nonzero, is at most max(|a|, |b|). So every component, and g, has at most max(bits(a), bits(b),
bits(c)) + 1 bits, and anything longer is refused by its bit length before any arithmetic: an oversized forged value
or g costs nothing, and no true claim is refused. Products, remainders and the gcd are charged to the budget before
they run (trusted.upoly.charge_product and charge_gcd). An argument over max_integer_bits raises ResourceLimitError,
as the producer's budget refuses it: such a claim is beyond the budget, not wrong. No prime, factorization or modular
power occurs, so the helpers of certify.ntheory.factorization are not needed.

The record class is identified by its module and name: a checker never imports producer code. Anything malformed,
a record forged without __init__ or with a missing field included, answers False.
"""

from math import gcd

from mathforge.limits import Budget, computation
from mathforge.trusted.upoly import charge_gcd, charge_product

OPERATION = "linear_diophantine"
ARGUMENTS = frozenset(("a", "b", "c"))
RECORD = ("mathforge.areas.ntheory.diophantine", "DiophantineSolution")  # Module and name of the value's class.


def _fields(value) -> tuple | None:
    """(x, y, dx, dy) of a value of the record class; None for any other class or for a missing field."""
    cls = type(value)
    module, name = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    if type(module) is not str or type(name) is not str or (module, name) != RECORD:
        return None
    try:
        return value.x, value.y, value.dx, value.dy
    except AttributeError:  # A record made with object.__new__ may lack a field.
        return None


def _solvable(budget: Budget, a: int, b: int, c: int, fields: tuple, limit: int) -> bool:
    """Conditions (1) to (4) of the module docstring, for components of at most limit bits."""
    budget.tick(len(fields))  # One unit per component screened.
    if any(type(field) is not int or field.bit_length() > limit for field in fields):
        return False  # A bool, a float, a str or None is no exact integer; a longer component is no true one.
    x, y, dx, dy = fields
    if not ((dx > 0 and 0 <= x < dx) or (dx == 0 and dy == 1 and y == 0)):
        return False
    budget.tick(4)  # The four products of (1) and (2), and beyond one unit each what their lengths cost.
    charge_product(a.bit_length(), max(x.bit_length(), dx.bit_length()), 2)
    charge_product(b.bit_length(), max(y.bit_length(), dy.bit_length()), 2)
    if a * x + b * y != c or a * dx + b * dy != 0:
        return False
    budget.tick()
    charge_gcd(max(dx.bit_length(), dy.bit_length()))
    return gcd(dx, dy) == 1


def _unsolvable(budget: Budget, a: int, b: int, c: int, evidence: tuple, limit: int) -> bool:
    """g >= 1 divides a and b but not c, for evidence (g,) with g of at most limit bits."""
    if len(evidence) != 1 or type(evidence[0]) is not int:
        return False
    g = evidence[0]
    if g < 1 or g.bit_length() > limit:
        return False
    budget.tick(3)  # Three remainders, long divisions by g.
    charge_product(limit, g.bit_length(), 3)
    return a % g == 0 and b % g == 0 and c % g != 0


def check_operation(operation, args, value, evidence):
    if operation != OPERATION or type(args) is not dict or args.keys() != ARGUMENTS or type(evidence) is not tuple:
        return False
    a, b, c = args["a"], args["b"], args["c"]
    if type(a) is not int or type(b) is not int or type(c) is not int or (a == 0 and b == 0):
        return False  # A bool is no exact integer; a = b = 0 is invalid input (module docstring).
    fields = _fields(value)
    if fields is None:
        return False
    with computation() as budget:
        budget.tick()
        budget.check_int(a)  # ResourceLimitError over max_integer_bits, as the producer's budget refuses it.
        budget.check_int(b)
        budget.check_int(c)
        # The longest component of the true value, and the longest g, of any claim on these arguments.
        limit = max(a.bit_length(), b.bit_length(), c.bit_length()) + 1
        if all(field is None for field in fields):
            return _unsolvable(budget, a, b, c, evidence, limit)
        return not evidence and _solvable(budget, a, b, c, fields, limit)
