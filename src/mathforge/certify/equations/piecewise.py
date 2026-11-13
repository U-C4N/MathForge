"""Independent checker of the family mathforge.areas.equations.piecewise: solve_abs and solve_sqrt.

It never solves anything. It reads the problem, forms the problem polynomials itself, verifies with the cell engine
mathforge.certify.cells (whose module docstring proves its items 1 to 4) the cut of the real line that the evidence
describes, decides the relation on every member of that cut, and compares the claimed set with those truth values.
Both operations take (lhs, rhs, variable, relation) and claim the canonical solution set of lhs relation rhs over the
reals: {x real : x in the domain of F and F(x) relation 0}, with F = lhs - rhs.

Arguments (both). args holds exactly lhs, rhs, variable and relation. variable is a Symbol of domain Reals (the enum
member itself) without assumptions, and relation one of cells.RELATIONS. lhs and rhs are records of the class
RealFunction of mathforge.functions, read by module and name, whose variables are () or (variable,) and whose domains
are DomainPredicate records. Their expressions are RealExpr trees, read node by node (a node shared by several parents
once, a cycle refused) in this grammar: scalar (a canonical Rational: a constant), symbol (a Symbol equal to
variable), add, mul, pow (an exact int exponent >= 0), div (by a node free of abs and sqrt whose polynomial is a
nonzero constant), abs and sqrt (one operand). A node free of abs and sqrt is a polynomial expression. Every other node
(pi, E, any other function, an irrational scalar, another symbol, a negative power, a division by a non-constant or by
zero) puts the arguments outside both classes, and the answer is False. The checker forms exact polynomials over Q from
these nodes with its own arithmetic (trusted.upoly on integer numerators over one positive denominator), and reads
signs only from integer forms that are positive multiples of what they stand for, never from the sign-normalized
results of cells.master_polynomial or Cells.master.

Value (both). EmptySet or UniversalSet of mathforge.contracts with the domain Reals, or an IntervalSet of
mathforge.intervals other than the empty set and the whole line (those two are EmptySet and UniversalSet), in
canonical form: sorted, disjoint, unmergeable and nonempty intervals with bool closure flags, infinite ends open, a
single point a closed degenerate interval (Cells.matches enforces this structure). A rational end is a canonical
Rational. An irrational end is a RealAlgebraicRoot(P, i) of mathforge.algebraic whose polynomial P, primitive with a
positive leading coefficient as the class stores it, has degree 2 or more, divides one problem polynomial (below) over
Z and has no rational root; i numbers the end among the real roots of P. The minimal polynomial of an irrational end
has these properties, and it is the name the producer writes: every end of the true set is a root of a problem
polynomial (theorems below), which the minimal polynomial therefore divides, and an irreducible polynomial of degree
two or more has no rational root. No irreducibility test is available here, so a reducible name with these properties
is accepted as well, such as v0.7's solve path writes for the roots of a square-free factor of degree three or more
without its rational roots. Any other name answers False.

Evidence (both): (cuts, samples), two strictly increasing tuples of canonical Rationals.
- cuts: rational cut points, each a root of some problem polynomial. Every problem polynomial loses the cut points that
  are its roots (cells.split_rational_roots), and the master polynomial is H = cells.master_polynomial(the rests). The
  producer lists every rational root of every problem polynomial; a rational root left out stays a root of H, which is
  accepted as well.
- samples: rho_0 < ... < rho_r, where the points alpha_1 < ... < alpha_r are the real roots of H and the cut points,
  each value once: one sample below alpha_1, one strictly between alpha_k and alpha_(k+1) for each k, one above
  alpha_r, so exactly one sample more than points and none of them a point; one sample when there is no point.
When cells.Cells(H, samples, cuts) accepts them (item 1), the points are exactly the real roots of all the problem
polynomials, and rho_k lies in the open cell C_k between alpha_k and alpha_(k+1) (C_0 unbounded below and C_r above;
C_0 = R when there is no point). Every finite end of a claimed IntervalSet must then be a point (item 4).

Sizes. A degree budget D, computed from the arguments alone, bounds the total degree of the problem polynomials and so
the number of points. Before any work on what they bound, the claim has at most D + 1 intervals and end polynomials of
at most D + 1 coefficients, the evidence at most D cut points and D + 1 samples, and every integer of the value and of
the evidence lies within max_integer_bits. A factor P of Q in Z[x] has max |p_i| <= 2^deg(P) * ||Q||_2 (Landau and
Mignotte), so an end polynomial longer in bits than max_bits(Q) + len(Q) + bits(len(Q)) for every problem polynomial Q
is refused before any division. An oversized forged claim or evidence answers False at once.

solve_abs(lhs, rhs, variable, relation)
Class: F is a polynomial expression in the variable and in atoms abs(P_i), the P_i the distinct polynomials under an
abs, each a polynomial expression (no abs inside an abs; no atom at all is allowed); no sqrt; both domains TRUE (the
domain of every such function, so that the domain of F is R). With m the sum of the degrees of the atoms and d the
degree bound of F with every atom replaced by +-P_i, D = m + (m + 1) * d. True requires
  (1) patterns: K_k, the signs of the atoms at rho_k, with no nonconstant atom vanishing at a sample;
  (2) pieces: for each pattern K among K_0, ..., K_r, G_K = F with every abs(P_i) replaced by K(i) * P_i;
  (3) the cut, the problem polynomials being the nonconstant atoms and the pieces that are not constant;
  (4) cells: s_k = sign F(rho_k), from the original trees evaluated exactly with rationals (abs of a rational is
      exact), equal to the sign of G_(K_k) at rho_k; C_k is true iff holds(relation, s_k);
  (5) points: t_k, the sign of G_(K_(k-1)) at alpha_k (Cells.root_signs), equal to the sign of G_(K_k) there;
      alpha_k is true iff holds(relation, t_k);
  (6) the claim: EmptySet iff no member is true, UniversalSet iff every member is true, and an IntervalSet iff
      Cells.matches accepts it with these truth values and the names of its irrational ends pass (Names).
Theorem. Every root of every nonconstant atom is a point, so on the open cell C_k no atom vanishes and each keeps the
sign it has at rho_k: |P_i| = K_k(i) * P_i on C_k, and F = G_(K_k) there. G_(K_k) is constant or a problem polynomial,
whose roots are points, so F keeps one sign on C_k, the sign s_k it has at rho_k, and the relation holds on all of C_k
or nowhere on it, as (4) says. F and every G_K are continuous, and alpha_k lies in the closures of C_(k-1) and of C_k,
so F(alpha_k) = G_(K_(k-1))(alpha_k) = G_(K_k)(alpha_k), whose sign Cells.root_signs gives exactly (cells, item 2):
(5) is the truth at alpha_k. So (4) and (5) are the truth values of the solution set T on every member of the cut, and
(6) proves the claim to be T in canonical form (cells, items 3 and 4; for EmptySet and UniversalSet, T is empty or the
whole line). Every pattern that occurs on R is among the K_k, since an open interval free of the atoms' roots contains
an open cell, hence a sample. The patterns come from the samples of this one cut rather than from a separate cut of the
atoms alone (the card's H0 cells): the argument needs no other, and the pieces are the same, since every H0 cell holds
a cell of the cut and every cell lies in one H0 cell. The equalities of signs in (4) and (5) follow from the theorem;
they guard the checker's own construction of the pieces.

solve_sqrt(lhs, rhs, variable, relation)
Class: F = A * sqrt(P) + B with polynomials A, B and P. Every node is a node of a polynomial expression or a sqrt; at
least one sqrt occurs; every radicand is a polynomial expression of the same polynomial P (one distinct radicand, which
may occur several times); no mul has two factors that both contain a sqrt, and no pow with an exponent >= 2 has a base
that contains one (linear occurrences only); no abs. Each domain predicate is TRUE, nonnegative(Q) for a polynomial
expression Q of the polynomial P, a conjunction of such predicates, or FALSE when P is a negative constant. Each of
them describes a set containing {P >= 0}, while a RealFunction's domain lies inside the domain {P >= 0} of its square
roots, so the domain of the problem is {P >= 0}. A and B are formed node by node: a sqrt is (1, 0), a polynomial node
(0, its polynomial), and add, mul (one factor free of sqrt), div and pow follow; N = A^2 * P - B^2. D is the sum of the
degrees of those of P, A, B and N that are not identically zero. True requires
  (1) the cut, the problem polynomials being those of P, A, B and N that are not constant;
  (2) truth by Cells.truth with the decision of catalogue 2.3 on the signs (sP, sA, sB, sN) at each point and each
      sample: outside the domain when sP < 0 (false for every relation); otherwise t = sA if sP > 0 else 0, u = sB, and
      sign F = u if t == 0, t if u == 0 or u == t, t * sN if u == -t; the member is true iff holds(relation, sign F);
  (3) at every sample rho_k, the same answer from the original trees evaluated exactly in Q(sqrt(p)), p = P(rho_k):
      outside the domain when the radicand is negative there, otherwise the sign of a + b * sqrt(p);
  (4) the claim as in (6) of solve_abs.
Theorem. Each of P, A, B and N is constant or has all its real roots among the points, so it keeps one sign on every
open cell, its sign at the cell's sample, and Cells.root_signs gives its sign at each alpha_k (cells, item 2). The
decision is the sign of F at any x of the domain: with P(x) > 0, sqrt(P(x)) > 0 and A*sqrt(P) has the sign t = sA; with
P(x) = 0 it vanishes and t = 0; if t == 0 then F = B; if u == 0 then F = A*sqrt(P); if u == t both terms have the sign
t; if u == -t, |A*sqrt(P)| > |B| exactly when A^2 P - B^2 > 0, so F has the sign t * sN. A function of the four signs is
constant on every open cell, so (2) gives the truth of the relation on each cell and at each point, and (4) proves the
claim to be the solution set in canonical form, as for solve_abs. N identically zero (sqrt(x^2) - x = 0 holds on
[0, inf)) and isolated points of the domain (sqrt(-x^2) = 0 holds at 0 alone) need nothing special. (3) follows from
the theorem; it guards the checker's own construction of A, B and P.

Names (both). Before the cut is built, the polynomial P of every RealAlgebraicRoot end of the claim must divide one
problem polynomial Q exactly, over Z and so over Q (P is primitive: Gauss). Every real root of P is then a root of Q,
hence a point, which Cells.locate finds (item 4) and Cells.root_signs reads (item 2). Once the claim matches, P must
vanish at no cut point, and at every other point alpha_k where P vanishes, with c > 0 its leading coefficient and
L = c + max |p_i| over its other coefficients, a binary search must find no integer m with P(m / c) = 0 among those
with c * rho_(k-1) < m < c * rho_k and |m| < L. Theorem: alpha_k is the one root of P in (rho_(k-1), rho_k), and the
sign of P changes between the two samples (tested), so P has the sign it has at rho_(k-1) below alpha_k and the
opposite one above it; comparing the sign of P(m / c) with it, the search keeps in its range every m with m / c =
alpha_k. A rational root a / b of P in lowest terms has b | c (rational root theorem) and |a / b| < L / c (Cauchy's
bound), so a rational alpha_k would be such an m / c. Hence P has no rational root: its real roots are points, none of
them at a cut point and none rational elsewhere.

Sizes and failures. Every loop charges the budget, and every product, gcd, division and evaluation is charged before
it runs; no builtin pow runs (powers are repeated products, each charged). An integer of the arguments longer than
max_integer_bits raises ResourceLimitError, as run() and the verifier refuse such arguments before any checker runs,
and so does a problem whose polynomials outgrow the budget. Anything malformed (a record forged without __init__ or
with a missing field, a bool, a float, a str or None where a number belongs, a list for a tuple, an impostor class of
the right name from another module, evidence of another shape) answers False, never an exception.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.certify import cells
from mathforge.limits import Budget, ResourceLimitError, computation, current_budget
from mathforge.model import Rational, Reals, Symbol
from mathforge.trusted import upoly as tp

OPERATIONS = ("solve_abs", "solve_sqrt")
ARGUMENTS = frozenset(("lhs", "rhs", "variable", "relation"))

# The record classes read here, by module and qualified name: a checker never imports them.
FUNCTION = ("mathforge.functions", "RealFunction")
EXPRESSION = ("mathforge.functions", "RealExpr")
PREDICATE = ("mathforge.functions", "DomainPredicate")
EMPTY_SET = ("mathforge.contracts", "EmptySet")
UNIVERSAL_SET = ("mathforge.contracts", "UniversalSet")
INTERVAL_SET = ("mathforge.intervals", "IntervalSet")
INTERVAL = ("mathforge.intervals", "Interval")
ALGEBRAIC_ROOT = ("mathforge.algebraic", "RealAlgebraicRoot")

_MISSING = object()  # A field that a record made with object.__new__ lacks.

# A polynomial over Q is (numerators, denominator): integer coefficients lowest degree first without a trailing zero,
# over a positive denominator that shares no factor with all of them, so that equal polynomials are equal pairs.
_ZERO = ((), 1)
_ONE = ((1,), 1)
_X = ((0, 1), 1)
# A rational in the evaluations is (numerator, denominator) in lowest terms with a positive denominator.
_Q0 = (0, 1)
_Q1 = (1, 1)

# A node of a compiled expression: (op, operand indices, payload, contains an abs, contains a sqrt). Operands come
# before the node, so one pass in index order meets every operand before its parent.
_ABS, _SQRT = 3, 4


class _Refused(Exception):
    """The arguments, the value or the evidence fail a condition of the module docstring: the answer is False."""


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified solution set of the operation on args, by the theorems of the
    module docstring. False for an unknown operation, for arguments outside the operation's class and for any
    malformed value or evidence; ResourceLimitError past the budget."""
    if type(operation) is not str or operation not in OPERATIONS or type(args) is not dict or set(args) != ARGUMENTS:
        return False
    with computation() as budget:  # One budget for every helper: without an active one, each would open its own.
        try:
            if operation == "solve_abs":
                return _solve_abs(budget, args, value, evidence)
            return _solve_sqrt(budget, args, value, evidence)
        except (_Refused, cells.Rejected):
            return False
        except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
            return False  # A forged field that breaks an operation fails closed (ResourceLimitError is none of them).


# ---------------------------------------------------------------------------------------------------------------
# The two operations

def _solve_abs(budget: Budget, args: dict, value: object, evidence: object) -> bool:
    variable, relation, lhs, rhs = _arguments(budget, args)
    for function in (lhs, rhs):
        budget.tick()
        if _conjuncts(budget, _field(function, "domain")) != [("true", None)]:
            raise _Refused("The domain of an absolute value problem is the whole line")
    nodes: list = []
    index: dict = {}
    left = _compile(budget, _field(lhs, "expression"), variable, nodes, index)
    right = _compile(budget, _field(rhs, "expression"), variable, nodes, index)
    budget.tick(len(nodes))
    if any(node[_SQRT] for node in nodes):
        raise _Refused("No square root in solve_abs")
    pure = _pure(budget, nodes)
    atoms, atom_of = _atoms(budget, nodes, pure)
    degree = _abs_degrees(budget, nodes, pure, atoms, atom_of)
    budget.tick(len(atoms))
    atom_degrees = sum([max(len(atom[0]) - 1, 0) for atom in atoms])
    bound = atom_degrees + (atom_degrees + 1) * max(degree[left], degree[right], 0)
    kind, claim, ends = _claim(budget, value, bound)
    cuts, samples = _evidence(budget, evidence, bound)
    budget.tick(len(atoms))
    integer_atoms = [_integer(atom) for atom in atoms]
    patterns = []
    for sample in samples:
        budget.tick(1 + len(integer_atoms))
        pattern = []
        for atom in integer_atoms:
            sign = _sign_at(atom, sample.numerator, sample.denominator)
            if not sign and len(atom) > 1:
                raise _Refused("A sample is a root of an atom")
            pattern.append(sign)
        patterns.append(tuple(pattern))
    with_abs = [i for i, node in enumerate(nodes) if node[_ABS]]
    pieces: dict = {}
    for pattern in patterns:
        budget.tick()
        if pattern not in pieces:
            pieces[pattern] = _integer(_piece(budget, nodes, pure, with_abs, atoms, atom_of, pattern, left, right))
    problem = [atom for atom in integer_atoms if len(atom) > 1]
    problem.extend(piece for piece in pieces.values() if len(piece) > 1)
    _names(budget, ends, problem)
    grid = _grid(budget, problem, cuts, samples)
    r = grid.root_count
    budget.tick(len(patterns))
    at = [pieces[pattern] for pattern in patterns]  # The piece that equals F on each open cell.
    cell_truth = []
    for k, sample in enumerate(samples):
        budget.tick()
        sign = _abs_sign_at(budget, nodes, left, right, sample)
        if sign != grid.sample_signs(at[k])[k]:
            raise _Refused("F and its piece differ at a sample")
        cell_truth.append(cells.holds(relation, sign))
    root_truth = []
    for k in range(1, r + 1):
        budget.tick()
        sign = grid.root_signs(at[k - 1])[k - 1]
        if sign != grid.root_signs(at[k])[k - 1]:
            raise _Refused("The pieces of two neighbouring cells differ at their common point")
        root_truth.append(cells.holds(relation, sign))
    return _matches(grid, kind, claim, tuple(root_truth), tuple(cell_truth)) and _irrational(budget, grid, ends)


def _solve_sqrt(budget: Budget, args: dict, value: object, evidence: object) -> bool:
    variable, relation, lhs, rhs = _arguments(budget, args)
    nodes: list = []
    index: dict = {}
    left = _compile(budget, _field(lhs, "expression"), variable, nodes, index)
    right = _compile(budget, _field(rhs, "expression"), variable, nodes, index)
    count = len(nodes)  # The nodes of F; the operands of the domain predicates follow.
    budget.tick(count)
    if any(node[_ABS] for node in nodes):
        raise _Refused("No absolute value in solve_sqrt")
    conjuncts = _conjuncts(budget, _field(lhs, "domain")) + _conjuncts(budget, _field(rhs, "domain"))
    operands = [_compile(budget, operand, variable, nodes, index) if predicate == "nonnegative" else None
                for predicate, operand in conjuncts]
    pure = _pure(budget, nodes)
    radicand, a, b = _linear_form(budget, nodes, count, pure, left, right)
    for (predicate, _), operand in zip(conjuncts, operands):
        budget.tick()
        if predicate == "false":  # Only a negative constant radicand has an empty domain.
            if len(radicand[0]) != 1 or radicand[0][0] >= 0:
                raise _Refused("A FALSE domain needs a negative constant radicand")
        elif predicate == "nonnegative":
            if nodes[operand][_ABS] or nodes[operand][_SQRT] or pure[operand] != radicand:
                raise _Refused("A domain predicate other than nonnegative(radicand)")
    n = _poly_sub(_poly_mul(_poly_mul(a, a), radicand), _poly_mul(b, b))
    budget.tick(4)
    polynomials = [_integer(polynomial) for polynomial in (radicand, a, b, n)]
    bound = sum([len(polynomial) - 1 for polynomial in polynomials if polynomial])
    kind, claim, ends = _claim(budget, value, bound)
    cuts, samples = _evidence(budget, evidence, bound)
    problem = [polynomial for polynomial in polynomials if len(polynomial) > 1]
    _names(budget, ends, problem)
    grid = _grid(budget, problem, cuts, samples)

    def decide(signs: tuple[int, ...]) -> bool:
        sign = _sqrt_sign(*signs)
        return sign is not None and cells.holds(relation, sign)

    root_truth, cell_truth = grid.truth(decide, *polynomials)
    columns = [grid.sample_signs(polynomial) for polynomial in polynomials]
    for k, sample in enumerate(samples):
        budget.tick()
        if _sqrt_sign(*[column[k] for column in columns]) != _surd_sign_at(budget, nodes, count, left, right,
                                                                          sample):
            raise _Refused("F and its decision differ at a sample")
    return _matches(grid, kind, claim, root_truth, cell_truth) and _irrational(budget, grid, ends)


def _sqrt_sign(sp: int, sa: int, sb: int, sn: int) -> int | None:
    """The sign of A*sqrt(P) + B from the signs of P, A, B and N = A^2 P - B^2 at one point (catalogue 2.3); None
    outside the domain, where P < 0."""
    if sp < 0:
        return None
    t = sa if sp > 0 else 0
    if t == 0:
        return sb
    if sb == 0 or sb == t:
        return t
    return t * sn


def _matches(grid: cells.Cells, kind: str, claim: object, root_truth: tuple, cell_truth: tuple) -> bool:
    """Whether the claim is the set of the truth values in canonical form: EmptySet for none, UniversalSet for all,
    otherwise the IntervalSet that Cells.matches accepts."""
    if kind == "empty":
        return not any(root_truth) and not any(cell_truth)
    if kind == "universal":
        return all(root_truth) and all(cell_truth)
    return grid.matches(claim, root_truth, cell_truth)


# ---------------------------------------------------------------------------------------------------------------
# Reading the data

def _class(value: object) -> tuple[object, object]:
    kind = type(value)
    return getattr(kind, "__module__", None), getattr(kind, "__qualname__", None)


def _field(record: object, name: str) -> object:
    """A field of a record read as data, _MISSING when the record itself does not hold it. A record with a __dict__
    is read there and only there: getattr would answer a class-level default for a field that a record forged with
    object.__new__ left unset (EmptySet declares domain = Reals, RealExpr args = () and value = None). A record
    without a __dict__ keeps its fields in slots (Rational), which hold no default."""
    try:
        fields = vars(record)
    except TypeError:  # No __dict__: slots.
        return getattr(record, name, _MISSING)
    if type(fields) is not dict:
        return _MISSING
    return fields.get(name, _MISSING)


def _symbol(value: object) -> tuple[str, str] | None:
    """(name, uid) of an unconstrained real Symbol: domain Reals, the enum member itself (Domain is a str enum, and
    "reals" == Reals), and no assumptions; None for anything else."""
    if type(value) is not Symbol:
        return None
    name, domain, assumptions, uid = (_field(value, field) for field in ("name", "domain", "assumptions", "uid"))
    if type(name) is not str or domain is not Reals or type(assumptions) is not tuple or assumptions:
        return None
    if type(uid) is not str:
        return None
    return name, uid


def _arguments(budget: Budget, args: dict) -> tuple[tuple[str, str], str, object, object]:
    """(the variable's name and uid, relation, lhs, rhs), each tested as the module docstring says."""
    variable = _symbol(args["variable"])
    relation = args["relation"]
    if variable is None or type(relation) is not str or relation not in cells.RELATIONS:
        raise _Refused("An unconstrained real variable and a relation of cells.RELATIONS")
    functions = (args["lhs"], args["rhs"])
    for function in functions:
        budget.tick()
        if _class(function) != FUNCTION:
            raise _Refused("lhs and rhs are RealFunctions")
        axes = _field(function, "variables")
        if type(axes) is not tuple or len(axes) > 1 or (axes and _symbol(axes[0]) != variable):
            raise _Refused("A function of the variable alone")
    return variable, relation, functions[0], functions[1]


def _fraction(budget: Budget, value: object) -> tuple[int, int]:
    """(numerator, denominator) of a canonical Rational of the value or the evidence: exact ints in lowest terms with
    a positive denominator, both within max_integer_bits (tested before the gcd, which they bound)."""
    if type(value) is not Rational:
        raise _Refused("An exact Rational is required")
    numerator, denominator = _field(value, "numerator"), _field(value, "denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise _Refused("A Rational holds exact ints and a positive denominator")
    bits = max(numerator.bit_length(), denominator.bit_length())
    if bits > budget.limits.max_integer_bits:
        raise _Refused("An integer longer than max_integer_bits")
    budget.tick()
    if denominator != 1:
        tp.charge_gcd(bits)
        if gcd(numerator, denominator) != 1:
            raise _Refused("A Rational in lowest terms")
    return numerator, denominator


def _constant(budget: Budget, value: object) -> tuple[int, int]:
    """(numerator, denominator) of the canonical Rational of a scalar node; an integer of the arguments longer than
    max_integer_bits raises ResourceLimitError, as everywhere else in the library."""
    if type(value) is Rational:
        for name in ("numerator", "denominator"):
            item = _field(value, name)
            if type(item) is int:
                budget.check_int(item)
    return _fraction(budget, value)


def _conjuncts(budget: Budget, predicate: object) -> list[tuple[str, object]]:
    """The conjuncts of a DomainPredicate as (kind, operand): ("true", None), ("false", None) or ("nonnegative",
    RealExpr record); a conjunction gives its parts, which conjunction() never nests. Any other predicate (positive,
    nonzero, a nested and) lies outside both classes."""
    if _class(predicate) != PREDICATE:
        raise _Refused("A DomainPredicate is required")
    kind, operands = _field(predicate, "kind"), _field(predicate, "operands")
    if type(kind) is not str or type(operands) is not tuple:
        raise _Refused("A predicate holds a kind and a tuple of operands")
    if kind in ("true", "false"):
        if operands:
            raise _Refused("TRUE and FALSE have no operands")
        return [(kind, None)]
    if kind == "nonnegative":
        if len(operands) != 1:
            raise _Refused("nonnegative has one operand")
        return [(kind, operands[0])]
    if kind != "and" or len(operands) < 2:
        raise _Refused("A domain predicate outside both classes")
    parts = []
    for part in operands:
        budget.tick()
        if _class(part) != PREDICATE:
            raise _Refused("A conjunction of DomainPredicates")
        inner, items = _field(part, "kind"), _field(part, "operands")
        if type(inner) is not str or inner != "nonnegative" or type(items) is not tuple or len(items) != 1:
            raise _Refused("A conjunction of nonnegative predicates")
        parts.append((inner, items[0]))
    return parts


def _evidence(budget: Budget, evidence: object, bound: int) -> tuple[tuple[Rational, ...], tuple[Rational, ...]]:
    """(cuts, samples) of evidence that is a pair of strictly increasing tuples of canonical Rationals: at most bound
    cut points and from 1 to bound + 1 samples, the lengths tested before any item is read."""
    if type(evidence) is not tuple or len(evidence) != 2:
        raise _Refused("The evidence is (cuts, samples)")
    cuts, samples = evidence
    if type(cuts) is not tuple or type(samples) is not tuple:
        raise _Refused("Cut points and samples are tuples of Rationals")
    if len(cuts) > bound or not 1 <= len(samples) <= bound + 1:
        raise _Refused("More cut points or samples than the problem's degrees allow")
    return _points(budget, cuts), _points(budget, samples)


def _points(budget: Budget, values: tuple) -> tuple:
    """values, when it is a strictly increasing tuple of canonical Rationals."""
    budget.tick(len(values))
    fractions = [_fraction(budget, item) for item in values]
    for (a, b), (c, d) in zip(fractions, fractions[1:]):
        tp.charge_product(max(a.bit_length(), b.bit_length()), max(c.bit_length(), d.bit_length()), 2)
        if a * d >= c * b:
            raise _Refused("Cut points and samples are strictly increasing")
    return values


def _claim(budget: Budget, value: object, bound: int) -> tuple[str, object, tuple[tuple[int, ...], ...]]:
    """(kind, claim, ends): ("empty", None, ()), ("universal", None, ()) or ("set", the IntervalSet record, the
    distinct polynomials of its RealAlgebraicRoot ends) for a value of a canonical shape. Its interval count and the
    lengths of its end polynomials are bounded by the degree budget before anything is built from them; Cells.matches
    reads the rest (the flags, the Rational ends, the indexes of the roots)."""
    named = _class(value)
    if named in (EMPTY_SET, UNIVERSAL_SET):
        if _field(value, "domain") is not Reals:
            raise _Refused("A solution set over the reals")
        return ("empty" if named == EMPTY_SET else "universal"), None, ()
    if named != INTERVAL_SET:
        raise _Refused("EmptySet, UniversalSet or IntervalSet")
    records = _field(value, "intervals")
    if type(records) is not tuple or not records:
        raise _Refused("The empty set is EmptySet(Reals)")
    if len(records) > bound + 1:
        raise _Refused("More intervals than the problem's points allow")
    ends: list = []
    seen: set = set()
    for record in records:
        budget.tick()
        if _class(record) != INTERVAL:
            raise _Refused("An IntervalSet holds Interval records")
        fields = [_field(record, name) for name in ("lower", "upper", "left_closed", "right_closed")]
        if any(item is _MISSING for item in fields):
            raise _Refused("An Interval record holds its four fields")
        for end in fields[:2]:
            if end is None or type(end) is Rational:
                continue
            if _class(end) != ALGEBRAIC_ROOT:
                raise _Refused("An end is a Rational, a RealAlgebraicRoot or None")
            polynomial = _end_polynomial(budget, end, bound)
            if polynomial not in seen:
                seen.add(polynomial)
                ends.append(polynomial)
    if len(records) == 1 and fields[0] is None and fields[1] is None:
        raise _Refused("The whole line is UniversalSet(Reals)")
    return "set", value, tuple(ends)


def _end_polynomial(budget: Budget, end: object, bound: int) -> tuple[int, ...]:
    """The polynomial of a RealAlgebraicRoot end: a tuple of 3 to bound + 1 exact ints within max_integer_bits,
    primitive with a positive leading coefficient. A rational end is a Rational, never a root of degree one."""
    coefficients = _field(end, "integer_coefficients")
    if type(coefficients) is not tuple or not 3 <= len(coefficients) <= bound + 1:
        raise _Refused("A RealAlgebraicRoot end has degree from 2 to the problem's degree budget")
    budget.tick(len(coefficients))
    for coefficient in coefficients:
        if type(coefficient) is not int:
            raise _Refused("A RealAlgebraicRoot holds exact ints")
    bits = tp.max_bits(coefficients)
    if bits > budget.limits.max_integer_bits:
        raise _Refused("An integer longer than max_integer_bits")
    tp.charge_gcd(bits, len(coefficients))
    if coefficients[-1] <= 0 or gcd(*coefficients) != 1:
        raise _Refused("A RealAlgebraicRoot holds a primitive polynomial with a positive leading coefficient")
    return coefficients


# ---------------------------------------------------------------------------------------------------------------
# Expressions

def _compile(budget: Budget, root: object, variable: tuple[str, str], nodes: list, index: dict) -> int:
    """The index in nodes of a RealExpr tree, appending its nodes that index does not hold yet, operands first.
    index maps id(record) to a node index: the records stay alive in the arguments while the check runs. A record that
    is its own descendant is refused."""
    if id(root) in index:
        return index[id(root)]
    visiting: set = set()
    pending: dict = {}
    stack = [(root, False)]
    while stack:
        budget.tick()
        record, ready = stack.pop()
        key = id(record)
        if ready:
            op, operands, payload = pending.pop(key)
            children = tuple([index[id(operand)] for operand in operands])
            has_abs = op == "abs" or any(nodes[child][_ABS] for child in children)
            has_sqrt = op == "sqrt" or any(nodes[child][_SQRT] for child in children)
            visiting.discard(key)
            index[key] = len(nodes)
            nodes.append((op, children, payload, has_abs, has_sqrt))
            continue
        if key in index:
            continue
        op, operands, payload = _node(budget, record, variable)
        visiting.add(key)
        pending[key] = (op, operands, payload)
        stack.append((record, True))
        for operand in operands:
            if id(operand) in visiting:
                raise _Refused("An expression that contains itself")
            if id(operand) not in index:
                stack.append((operand, False))
    return index[id(root)]


def _node(budget: Budget, record: object, variable: tuple[str, str]) -> tuple[str, tuple, object]:
    """(op, operand records, payload) of a RealExpr record of the classes' grammar (module docstring)."""
    if _class(record) != EXPRESSION:
        raise _Refused("A RealExpr record is required")
    op, operands, payload = _field(record, "op"), _field(record, "args"), _field(record, "value")
    if type(op) is not str or type(operands) is not tuple:
        raise _Refused("A RealExpr holds an op and a tuple of operands")
    if op == "scalar":
        if operands:
            raise _Refused("A constant has no operand")
        return op, (), _constant(budget, payload)  # A RealAlgebraicRoot constant is refused here.
    if op == "symbol":
        if operands or _symbol(payload) != variable:
            raise _Refused("The one variable is the only symbol")
        return op, (), None
    if op in ("add", "mul", "div"):
        if len(operands) != 2 or payload is not None:
            raise _Refused("A binary node has two operands")
    elif op == "pow":
        if len(operands) != 1 or type(payload) is not int or payload < 0:
            raise _Refused("A power has one operand and an exact int exponent >= 0")
        budget.check_int(payload)
    elif op in ("abs", "sqrt"):
        if len(operands) != 1 or payload is not None:
            raise _Refused("abs and sqrt have one operand")
    else:
        raise _Refused("pi, E and every other function lie outside both classes")
    return op, operands, payload


def _pure(budget: Budget, nodes: list) -> list:
    """The polynomial of every node free of abs and sqrt (None for the others), and the test that every division is
    one by a nonzero constant polynomial free of abs and sqrt."""
    values: list = [None] * len(nodes)
    for i, (op, operands, payload, has_abs, has_sqrt) in enumerate(nodes):
        budget.tick()
        if op == "div":
            divisor = nodes[operands[1]]
            if divisor[_ABS] or divisor[_SQRT] or len(values[operands[1]][0]) != 1:
                raise _Refused("A division by a non-constant or by zero")
        if has_abs or has_sqrt:
            continue
        if op == "scalar":
            values[i] = ((payload[0],), payload[1]) if payload[0] else _ZERO
        elif op == "symbol":
            values[i] = _X
        else:
            values[i] = _operation(budget, op, payload, [values[operand] for operand in operands])
    return values


def _operation(budget: Budget, op: str, payload: object, operands: list) -> tuple:
    """add, mul, div (by a nonzero constant) or pow of polynomials over Q."""
    if op == "add":
        return _poly_add(operands[0], operands[1])
    if op == "mul":
        return _poly_mul(operands[0], operands[1])
    if op == "div":
        return _poly_divide(operands[0], operands[1])
    if op == "pow":
        return _poly_power(budget, operands[0], payload)
    raise _Refused("No polynomial operation")


def _atoms(budget: Budget, nodes: list, pure: list) -> tuple[list, dict]:
    """(atoms, atom_of): the distinct polynomials under an abs in their order of appearance, and the atom number of
    each abs node; an abs inside an abs is refused."""
    atoms: list = []
    atom_of: dict = {}
    numbers: dict = {}
    for i, (op, operands, _, _, _) in enumerate(nodes):
        budget.tick()
        if op == "abs":
            operand = operands[0]
            if nodes[operand][_ABS]:
                raise _Refused("An abs inside an abs")
            atom = pure[operand]
            if atom not in numbers:
                numbers[atom] = len(atoms)
                atoms.append(atom)
            atom_of[i] = numbers[atom]
    return atoms, atom_of


def _abs_degrees(budget: Budget, nodes: list, pure: list, atoms: list, atom_of: dict) -> list:
    """A bound for the degree of every node with each atom replaced by +-P_i: exact for nodes free of abs."""
    degrees = [0] * len(nodes)
    for i, (op, operands, payload, has_abs, _) in enumerate(nodes):
        budget.tick()
        if not has_abs:
            degrees[i] = max(len(pure[i][0]) - 1, 0)
        elif op == "abs":
            degrees[i] = max(len(atoms[atom_of[i]][0]) - 1, 0)
        elif op == "add":
            degrees[i] = max(degrees[operands[0]], degrees[operands[1]])
        elif op == "mul":
            degrees[i] = degrees[operands[0]] + degrees[operands[1]]
        elif op == "pow":
            degrees[i] = payload * degrees[operands[0]]
        else:  # div, by a constant
            degrees[i] = degrees[operands[0]]
    return degrees


def _piece(budget: Budget, nodes: list, pure: list, with_abs: list, atoms: list, atom_of: dict, pattern: tuple,
           left: int, right: int) -> tuple:
    """G_K for the pattern K: F with every abs(P_i) replaced by K(i) * P_i, over Q."""
    values: dict = {}

    def value(i: int) -> tuple:
        return values[i] if nodes[i][_ABS] else pure[i]

    for i in with_abs:
        budget.tick()
        op, operands, payload = nodes[i][:3]
        if op == "abs":
            atom, sign = atoms[atom_of[i]], pattern[atom_of[i]]
            values[i] = atom if sign > 0 else _poly_neg(atom) if sign < 0 else _ZERO
        else:
            values[i] = _operation(budget, op, payload, [value(operand) for operand in operands])
    return _poly_sub(value(left), value(right))


def _linear_form(budget: Budget, nodes: list, count: int, pure: list, left: int, right: int) -> tuple:
    """(P, A, B) with F = A*sqrt(P) + B, by the rules of the solve_sqrt class: one radicand P, linear occurrences."""
    radicand = None
    pairs: dict = {}

    def pair(i: int) -> tuple:
        return pairs[i] if nodes[i][_SQRT] else (_ZERO, pure[i])

    for i in range(count):
        budget.tick()
        op, operands, payload, _, has_sqrt = nodes[i]
        if not has_sqrt:
            continue
        if op == "sqrt":
            operand = operands[0]
            if nodes[operand][_SQRT]:
                raise _Refused("A sqrt inside a sqrt")
            if radicand is None:
                radicand = pure[operand]
            elif pure[operand] != radicand:
                raise _Refused("A second radicand")
            pairs[i] = (_ONE, _ZERO)
        elif op == "add":
            (a1, b1), (a2, b2) = pair(operands[0]), pair(operands[1])
            pairs[i] = (_poly_add(a1, a2), _poly_add(b1, b2))
        elif op == "mul":
            first, second = operands
            if nodes[first][_SQRT] and nodes[second][_SQRT]:
                raise _Refused("A product of two square roots is not linear")
            if not nodes[first][_SQRT]:
                first, second = second, first
            (a, b), (_, c) = pair(first), pair(second)  # c is the polynomial of the factor free of sqrt.
            pairs[i] = (_poly_mul(a, c), _poly_mul(b, c))
        elif op == "div":
            (a, b), divisor = pair(operands[0]), pure[operands[1]]
            pairs[i] = (_poly_divide(a, divisor), _poly_divide(b, divisor))
        elif op == "pow":
            if payload >= 2:
                raise _Refused("A power of a square root is not linear")
            pairs[i] = pair(operands[0]) if payload == 1 else (_ZERO, _ONE)
        else:
            raise _Refused("No such node in solve_sqrt")
    if radicand is None:
        raise _Refused("solve_sqrt needs a square root")
    (a1, b1), (a2, b2) = pair(left), pair(right)
    return radicand, _poly_sub(a1, a2), _poly_sub(b1, b2)


# ---------------------------------------------------------------------------------------------------------------
# Polynomials over Q: (numerators, positive denominator), charged through trusted.upoly

def _bounded(coefficients: tuple[int, ...]) -> tuple[int, ...]:
    """The coefficients, unless one outgrows max_integer_bits: then the budget is exhausted, as in cells."""
    budget = current_budget()
    if coefficients and budget is not None and tp.max_bits(coefficients) > budget.limits.max_integer_bits:
        raise ResourceLimitError("Computation limit exceeded: integer bits")
    return coefficients


def _reduced(numerators: tuple[int, ...], denominator: int) -> tuple:
    if not numerators:
        return _ZERO
    if denominator == 1:
        return _bounded(numerators), 1
    tp.charge_gcd(max(tp.max_bits(numerators), denominator.bit_length()), len(numerators))
    divisor = gcd(denominator, *numerators)
    if divisor != 1:
        numerators = tuple([value // divisor for value in numerators])
        denominator //= divisor
    tp.check_bits(denominator)
    return _bounded(numerators), denominator


def _scaled(numerators: tuple[int, ...], factor: int) -> tuple[int, ...]:
    tp.charge_product(tp.max_bits(numerators), factor.bit_length(), len(numerators))
    return tp.scale(numerators, factor)


def _poly_add(p: tuple, q: tuple) -> tuple:
    (a, da), (b, db) = p, q
    if not a:
        return q
    if not b:
        return p
    if da == db:
        return _reduced(tp.add(a, b), da)
    tp.charge_product(da.bit_length(), db.bit_length())
    return _reduced(tp.add(_scaled(a, db), _scaled(b, da)), da * db)


def _poly_neg(p: tuple) -> tuple:
    tp.charge(len(p[0]))
    return tp.neg(p[0]), p[1]


def _poly_sub(p: tuple, q: tuple) -> tuple:
    return _poly_add(p, _poly_neg(q))


def _poly_mul(p: tuple, q: tuple) -> tuple:
    (a, da), (b, db) = p, q
    if not a or not b:
        return _ZERO
    tp.charge_product(da.bit_length(), db.bit_length())
    return _reduced(tp.mul(a, b), da * db)


def _poly_divide(p: tuple, constant: tuple) -> tuple:
    """p divided by a nonzero constant polynomial (k,)/d, that is multiplied by d/k."""
    (a, da), ((k,), d) = p, constant
    if not a:
        return _ZERO
    tp.charge_product(da.bit_length(), abs(k).bit_length())
    return _reduced(_scaled(a, d if k > 0 else -d), da * abs(k))


def _poly_power(budget: Budget, p: tuple, exponent: int) -> tuple:
    """p ** exponent by repeated squaring, each product charged (no builtin pow)."""
    result, base = _ONE, p
    while exponent:
        budget.tick()
        if exponent & 1:
            result = _poly_mul(result, base)
        exponent >>= 1
        if exponent:
            base = _poly_mul(base, base)
    return result


def _integer(p: tuple) -> tuple[int, ...]:
    """The integer polynomial with the sign of p at every point: its numerators, a positive multiple of p (the
    denominator is positive), divided by their positive content; () for zero. Never sign-normalized."""
    numerators = p[0]
    if not numerators:
        return ()
    tp.charge(len(numerators))
    tp.charge_gcd(tp.max_bits(numerators), len(numerators))
    content = gcd(*numerators)
    return numerators if content == 1 else tuple([value // content for value in numerators])


def _sign_at(polynomial: tuple[int, ...], numerator: int, denominator: int) -> int:
    """The sign of an integer polynomial at numerator / denominator (denominator > 0), by exact evaluation, charged
    by the sizes of the coefficients and of the point before it runs (as cells charges its own evaluations)."""
    if not polynomial:
        return 0
    length, bits = len(polynomial), tp.max_bits(polynomial)
    point_bits = max(numerator.bit_length(), denominator.bit_length())
    tp.charge(length * (bits >> 14))
    tp.charge_product(bits, point_bits, length)
    if denominator != 1:
        tp.charge_product(bits, length * point_bits, length)
    return tp.sign_at(polynomial, numerator, denominator)


# ---------------------------------------------------------------------------------------------------------------
# The cut and the names of irrational ends

def _grid(budget: Budget, problem: list, cuts: tuple, samples: tuple) -> cells.Cells:
    """Cells(H, samples, cuts) for the problem polynomials (sign-preserving integer forms, nonconstant): every cut
    point a root of some problem polynomial, split off from each problem polynomial that it is a root of, and H the
    master polynomial of the rests; Rejected or _Refused otherwise."""
    used = [False] * len(cuts)
    rests = []
    for polynomial in problem:
        budget.tick(1 + len(cuts))
        roots = []
        for j, cut in enumerate(cuts):
            if not _sign_at(polynomial, cut.numerator, cut.denominator):
                roots.append(cut)
                used[j] = True
        rests.append(cells.split_rational_roots(polynomial, tuple(roots))[1])
    if not all(used):
        raise _Refused("A cut point that is no root of a problem polynomial")
    return cells.Cells(cells.master_polynomial(*rests), samples, cuts)


def _names(budget: Budget, ends: tuple, problem: list) -> None:
    """Refuses unless the polynomial of every RealAlgebraicRoot end divides one problem polynomial (Names)."""
    for polynomial in ends:
        budget.tick(1 + len(problem))
        if not any(_divides(polynomial, candidate) for candidate in problem):
            raise _Refused("The polynomial of a RealAlgebraicRoot end divides no problem polynomial")


def _divides(divisor: tuple[int, ...], polynomial: tuple[int, ...]) -> bool:
    """Whether the primitive divisor (positive leading coefficient) divides the integer polynomial exactly over Z.

    A divisor longer in bits than the Landau-Mignotte bound of the polynomial (module docstring, Sizes) cannot divide
    it and is refused before the division, which is charged before it runs: each of its steps can lengthen the
    remainder by the divisor's bits."""
    if len(divisor) > len(polynomial):
        return False
    divisor_bits, bits = tp.max_bits(divisor), tp.max_bits(polynomial)
    if divisor_bits > bits + len(polynomial) + len(polynomial).bit_length():
        return False
    tp.charge(1)
    if polynomial[-1] % divisor[-1]:
        return False
    tp.charge_product(bits + len(polynomial) * (divisor_bits + 1), divisor_bits, len(polynomial) * len(divisor))
    division = tp.divmod_exact(polynomial, divisor)
    return division is not None and not division[1]


def _irrational(budget: Budget, grid: cells.Cells, ends: tuple) -> bool:
    """True when no end polynomial has a rational root (Names); _Refused otherwise. Every end polynomial divides a
    problem polynomial (_names), and the claim has matched, so that Cells.root_signs reads it at every point."""
    for polynomial in ends:
        budget.tick(1 + len(grid.cuts) + len(polynomial))
        for cut in grid.cuts:
            if not _sign_at(polynomial, cut.numerator, cut.denominator):
                raise _Refused("A RealAlgebraicRoot end whose polynomial has a rational root at a cut point")
        at_roots = grid.root_signs(polynomial)
        at_samples = grid.sample_signs(polynomial)
        lead = polynomial[-1]
        reach = lead + max([abs(coefficient) for coefficient in polynomial[:-1]])  # L of the module docstring.
        budget.tick(grid.root_count)
        for k in range(1, grid.root_count + 1):
            if at_roots[k - 1]:
                continue
            below = at_samples[k - 1]
            if not below or at_samples[k] != -below:
                raise _Refused("An end polynomial keeps its sign across one of its roots")
            _no_rational_root(budget, polynomial, grid.samples[k - 1], grid.samples[k], below, reach)
    return True


def _no_rational_root(budget: Budget, polynomial: tuple[int, ...], lower: Rational, upper: Rational, below: int,
                      reach: int) -> None:
    """Refuses when the one root of the polynomial between the samples lower and upper, where its sign changes from
    below to -below, is m / c for an integer m (c the leading coefficient) with c * lower < m < c * upper and
    |m| < reach, which a binary search on the signs of the polynomial at those m / c decides."""
    lead = polynomial[-1]
    tp.charge_product(lead.bit_length(), max(_bits(lower), _bits(upper)), 2)
    low = max(lead * lower.numerator // lower.denominator + 1, 1 - reach)  # The least m above c * lower.
    high = min(-(-lead * upper.numerator // upper.denominator) - 1, reach - 1)  # The greatest m below c * upper.
    while low <= high:
        budget.tick()
        middle = (low + high) // 2
        sign = _sign_at(polynomial, middle, lead)
        if not sign:
            raise _Refused("A RealAlgebraicRoot end whose polynomial has a rational root")
        if sign == below:
            low = middle + 1
        else:
            high = middle - 1


def _bits(value: Rational) -> int:
    return max(value.numerator.bit_length(), value.denominator.bit_length())


# ---------------------------------------------------------------------------------------------------------------
# Exact evaluation of the original trees at a sample

def _q_reduced(numerator: int, denominator: int) -> tuple[int, int]:
    if denominator == 1:
        return numerator, 1
    tp.charge_gcd(max(numerator.bit_length(), denominator.bit_length()))
    divisor = gcd(numerator, denominator)
    return (numerator, denominator) if divisor == 1 else (numerator // divisor, denominator // divisor)


def _q_times(x: int, y: int) -> int:
    tp.charge_product(x.bit_length(), y.bit_length())
    return x * y


def _q_add(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
    (a, b), (c, d) = x, y
    tp.charge(1)
    if b == d:
        return _q_reduced(a + c, b)
    return _q_reduced(_q_times(a, d) + _q_times(c, b), _q_times(b, d))


def _q_neg(x: tuple[int, int]) -> tuple[int, int]:
    return -x[0], x[1]


def _q_mul(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
    tp.charge(1)
    return _q_reduced(_q_times(x[0], y[0]), _q_times(x[1], y[1]))


def _q_div(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
    """x / y for y nonzero."""
    (a, b), (c, d) = x, y
    tp.charge(1)
    if c < 0:
        a, c = -a, -c
    return _q_reduced(_q_times(a, d), _q_times(b, c))


def _q_sign(x: tuple[int, int]) -> int:
    return (x[0] > 0) - (x[0] < 0)


def _q_power(budget: Budget, x: tuple[int, int], exponent: int) -> tuple[int, int]:
    """x ** exponent by repeated squaring, each product charged (no builtin pow)."""
    result, base = _Q1, x
    while exponent:
        budget.tick()
        if exponent & 1:
            result = _q_mul(result, base)
        exponent >>= 1
        if exponent:
            base = _q_mul(base, base)
    return result


def _abs_sign_at(budget: Budget, nodes: list, left: int, right: int, point: Rational) -> int:
    """The sign of F at a rational point, from the original trees with exact rationals: every node of F, an abs of a
    rational being exact."""
    x = (point.numerator, point.denominator)
    values: list = [None] * len(nodes)
    for i, (op, operands, payload, _, _) in enumerate(nodes):
        budget.tick()
        if op == "scalar":
            values[i] = payload
        elif op == "symbol":
            values[i] = x
        elif op == "add":
            values[i] = _q_add(values[operands[0]], values[operands[1]])
        elif op == "mul":
            values[i] = _q_mul(values[operands[0]], values[operands[1]])
        elif op == "div":
            values[i] = _q_div(values[operands[0]], values[operands[1]])
        elif op == "pow":
            values[i] = _q_power(budget, values[operands[0]], payload)
        elif op == "abs":
            value = values[operands[0]]
            values[i] = (abs(value[0]), value[1])
        else:
            raise _Refused("No such node in solve_abs")
    return _q_sign(_q_add(values[left], _q_neg(values[right])))


def _surd_mul(x: tuple, y: tuple, p: tuple[int, int] | None) -> tuple:
    """(a1 + b1 sqrt p)(a2 + b2 sqrt p) in Q(sqrt p)."""
    (a1, b1), (a2, b2) = x, y
    first = _q_mul(a1, a2)
    if b1[0] and b2[0]:
        first = _q_add(first, _q_mul(_q_mul(b1, b2), p))
    return first, _q_add(_q_mul(a1, b2), _q_mul(a2, b1))


def _surd_sign_at(budget: Budget, nodes: list, count: int, left: int, right: int, point: Rational) -> int | None:
    """The sign of F at a rational point, from the original trees evaluated exactly in Q(sqrt(p)) with p = P(point):
    a value is a + b*sqrt(p) with rationals a and b. None outside the domain, where the radicand is negative."""
    x = (point.numerator, point.denominator)
    p = None
    values: list = [None] * count
    for i in range(count):
        budget.tick()
        op, operands, payload = nodes[i][:3]
        if op == "scalar":
            values[i] = (payload, _Q0)
        elif op == "symbol":
            values[i] = (x, _Q0)
        elif op == "sqrt":
            radicand, irrational = values[operands[0]]
            if irrational[0]:
                raise _Refused("A radicand is a polynomial")
            if radicand[0] < 0:
                return None
            if p is None:
                p = radicand
            elif radicand != p:
                raise _Refused("One radicand has one value")
            values[i] = (_Q0, _Q1 if radicand[0] else _Q0)
        elif op == "add":
            (a1, b1), (a2, b2) = values[operands[0]], values[operands[1]]
            values[i] = (_q_add(a1, a2), _q_add(b1, b2))
        elif op == "mul":
            values[i] = _surd_mul(values[operands[0]], values[operands[1]], p)
        elif op == "div":
            (a, b), (c, irrational) = values[operands[0]], values[operands[1]]
            if irrational[0] or not c[0]:
                raise _Refused("A division by a nonzero rational")
            values[i] = (_q_div(a, c), _q_div(b, c))
        elif op == "pow":
            result, base, exponent = (_Q1, _Q0), values[operands[0]], payload
            while exponent:
                budget.tick()
                if exponent & 1:
                    result = _surd_mul(result, base, p)
                exponent >>= 1
                if exponent:
                    base = _surd_mul(base, base, p)
            values[i] = result
        else:
            raise _Refused("No such node in solve_sqrt")
    (a1, b1), (a2, b2) = values[left], values[right]
    a, b = _q_add(a1, _q_neg(a2)), _q_add(b1, _q_neg(b2))
    sa = _q_sign(a)
    sb = _q_sign(b) if p is not None and p[0] else 0
    if sb == 0:
        return sa
    if sa == 0 or sa == sb:
        return sb
    return sa * _q_sign(_q_add(_q_mul(a, a), _q_neg(_q_mul(_q_mul(b, b), p))))  # a^2 - b^2 p decides
