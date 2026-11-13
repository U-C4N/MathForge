"""Linear Diophantine equations in two unknowns: every integer solution of a*x + b*y = c (mathforge.ntheory).

linear_diophantine(a, b, c) takes exact integers with (a, b) != (0, 0) and returns a DiophantineSolution: the whole
solution set {(x + t*dx, y + t*dy) : t in Z} in its canonical form, or the record with four None fields when the
equation has no integer solution. With g = gcd(a, b), the steps that generate the solutions of a*x + b*y = 0 are
exactly +-(b/g, -a/g), so the canonical form is unique: for b != 0 the step has dx = |b|/g > 0 and the particular
solution 0 <= x < dx; for b == 0 the step is (0, 1) and y = 0 (then x = c/a). a = b = 0 is invalid input: the
solution set would be all of Z^2 (c = 0) or empty, and no record of this form describes it.

Evidence: () for a solvable equation, whose value the checker verifies directly; (g,) for an unsolvable one, where
g >= 1 divides a and b but not c. The producer hands over g = gcd(a, b); the checker accepts any such g. The checker,
mathforge.certify.ntheory.diophantine, states the theorems.

The producer (docs/v09/cards/m1-ntheory.md) is the extended Euclidean algorithm, in two library calls. First
g = gcd(a, b) (math.gcd): the equation is unsolvable exactly when g does not divide c (Bezout), and then the value
has four None fields and the evidence is (g,). For b == 0, g = |a| divides c, and the value is (c/a, 0, 0, 1). For
b != 0 the step is (dx, dy) = (|b|/g, -sign(b)*a/g), and x is the Bezout coefficient u of a/g modulo dx, which
kernels.integers.inverse computes (the builtin pow below 6,000 bits, Lehmer's extended Euclid above), times c/g, both
reduced modulo dx. Since (a/g)*u = 1 modulo dx, a*x = c modulo g*dx = |b|, so y = (c - a*x)/b is an exact quotient,
and 0 <= x < dx makes the particular solution canonical. Every component is then at most max(|a|, |b|, |c|) in
absolute value (|y| <= (|c| + |a|*(|b| - 1))/|b|), so the record never exceeds the bits of the arguments. No inverse
is needed when b divides c (dx divides c/g), every homogeneous equation (c = 0) among them: then x = 0 and
y = c/b. That covers a == 0: g = |b|, so dx = 1 and the value is (0, c/b, 1, 0).

Work: the producer charges the request budget before each step with the functions that the checker charges with
(trusted.upoly.charge_gcd and charge_product), and at least two units for any input, so that max_work=1 refuses it.
The gcd of a and b is charged by their lengths alone, and the inverse only when it runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import computation
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

SPECS = (
    OperationSpec("linear_diophantine", "ntheory",
                  (Argument("a", "int", None, "int"), Argument("b", "int", None, "int"),
                   Argument("c", "int", None, "int")),
                  "DiophantineSolution",
                  "Solve a*x + b*y = c in integers: a particular solution and the step of all solutions.",
                  "exact integers; certified primes, orders and residues; canonical integer solution families",
                  version=6),
)


@dataclass(frozen=True)
class DiophantineSolution:
    """All integer solutions (x + t*dx, y + t*dy), t in Z, of a*x + b*y = c; four None fields: none.

    The canonical form, which each solution set has exactly once: four exact ints with gcd(dx, dy) = 1 and either
    dx > 0 and 0 <= x < dx, or dx = 0, dy = 1 and y = 0. The record of no solution has four None fields."""
    x: int | None
    y: int | None
    dx: int | None
    dy: int | None

    def __post_init__(self):
        fields = (self.x, self.y, self.dx, self.dy)
        if all(field is None for field in fields):
            return
        with computation() as budget:
            budget.tick()
            for field in fields:
                if type(field) is not int:  # Neither bool nor float is an exact integer; None only all four.
                    raise InvalidInput("A Diophantine solution has four exact integers, or four None fields")
                budget.check_int(field)
            x, y, dx, dy = fields
            if dx > 0:
                if not 0 <= x < dx:
                    raise InvalidInput("The particular solution of a Diophantine solution has 0 <= x < dx")
            elif dx != 0 or dy != 1 or y != 0:
                raise InvalidInput("A Diophantine solution has dx > 0, or dx = 0 with dy = 1 and y = 0")
            words = max(dx.bit_length(), dy.bit_length()) >> 6
            budget.tick(words * words >> 6)  # Before the gcd, quadratic in the length (Lehmer).
            if gcd(dx, dy) != 1:
                raise InvalidInput("The step of a Diophantine solution is primitive: gcd(dx, dy) = 1")


TYPES = {"diophantine_solution": DiophantineSolution}
CHECKER = "mathforge.certify.ntheory.diophantine"


def produce(operation: str, args: dict[str, object]) -> tuple[DiophantineSolution, tuple[int, ...]]:
    """(value, evidence) of linear_diophantine(a, b, c), by the extended Euclidean algorithm of the module docstring:
    the canonical solution set and (), or the record of four None fields and (gcd(a, b),)."""
    if operation != "linear_diophantine":
        raise UnsupportedOperation(f"{operation} is not an operation of the diophantine family")
    a, b, c = args["a"], args["b"], args["c"]
    if type(a) is not int or type(b) is not int or type(c) is not int:  # Neither bool nor float is an exact int.
        raise InvalidInput("linear_diophantine requires exact integers a, b and c")
    if a == 0 and b == 0:
        raise InvalidInput("linear_diophantine requires a or b nonzero: with a = b = 0 every pair of integers "
                           "solves the equation, or none does")
    from ...kernels.integers import inverse  # Kernels load on first use: importing mathforge loads none.
    from ...trusted.upoly import charge_gcd, charge_product
    with computation() as budget:
        budget.tick()
        for value in (a, b, c):
            budget.check_int(value)  # run() refuses longer arguments already; a direct call is refused here.
        bits = max(a.bit_length(), b.bit_length(), c.bit_length())
        budget.tick()
        # By the lengths of a and b alone: gcd(a, b) never reads c. Charged by the length of c as well, a long c with
        # short a and b would cost the producer about 16,000 units where the checker needs 10.
        charge_gcd(max(a.bit_length(), b.bit_length()))
        g = gcd(a, b)  # At least 1, as a or b is nonzero.
        budget.tick()
        charge_product(bits, g.bit_length(), 4)  # c % g, then c // a or the exact quotients of a, b and c by g.
        if c % g:
            return DiophantineSolution(None, None, None, None), (g,)
        if b == 0:  # Then g = |a|, which divides c.
            return DiophantineSolution(c // a, 0, 0, 1), ()
        a_g, b_g, c_g = a // g, b // g, c // g
        dx, dy = abs(b_g), (-a_g if b > 0 else a_g)  # dx > 0 and a*dx + b*dy = 0.
        x = 0  # The solution modulo dx = 1, which covers a == 0 (then g = |b|), and whenever dx divides c/g.
        if dx > 1:
            steps = dx.bit_length()
            budget.tick()
            charge_product(bits, steps)  # c/g modulo dx.
            residue = c_g % dx
            # residue == 0 exactly when b divides c, every homogeneous equation (c = 0) among them: then x = 0 with
            # no inverse, whose charge (131,073 units at 65,536 bits) would otherwise dwarf the checker's.
            if residue:
                # The extended Euclid of the inverse: about one step per bit, and its quadratic part, four gcds'
                # worth (measured with CPython: 27 ms at 65,536 bits, 1.5 ms at 6,000, where it is the builtin
                # pow); then a/g modulo dx, the product and its remainder. With the tick and the product above, the
                # same charges as one step per bit plus one, four gcds and four products of these lengths.
                budget.tick(steps)
                charge_gcd(steps, 4)
                charge_product(bits, steps, 3)
                x = residue * inverse(a_g % dx, dx) % dx
        budget.tick()
        charge_product(bits, dx.bit_length(), 2)  # a*x, then the exact quotient by b.
        return DiophantineSolution(x, (c - a * x) // b, dx, dy), ()


def _coordinate(base, step, renderer, child):
    """base + step*t without its zero terms: 1 + 3*t, 1 - 2*t, 2, t, -t, 5*t, 0; LaTeX writes 3t for 3*t."""
    if step == 0:
        return child(base)
    term = "t" if step in (1, -1) else renderer.join((child(abs(step)), "" if renderer.latex else "*", "t"))
    if base == 0:
        return term if step > 0 else renderer.join(("-", term))
    return renderer.join((child(base), " + " if step > 0 else " - ", term))


def _render(solution, renderer, child):
    r"""(1 + 3*t, 1 - 2*t) for every integer t, or: no integer solution. In LaTeX (1 + 3t,\ 1 - 2t),\ t \in \mathbb{Z},
    or \varnothing."""
    if solution.x is None:
        return r"\varnothing" if renderer.latex else renderer.label("no integer solution")
    x = _coordinate(solution.x, solution.dx, renderer, child)
    y = _coordinate(solution.y, solution.dy, renderer, child)
    if renderer.latex:
        return renderer.join(("(", x, r",\ ", y, r"),\ t \in \mathbb{Z}"))
    return renderer.join(("(", x, ", ", y, ") for every integer t"))


RENDER = {DiophantineSolution: _render}
