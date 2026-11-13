"""Equations and inequalities with absolute values of polynomials, or with one square root of a polynomial, solved over
the reals (mathforge.equations): the canonical solution set, certified by one cut of the real line.

Both operations take (lhs, rhs, variable, relation): lhs and rhs are RealFunctions (MCP kind "function") of the one
variable, whose variables are () or (variable,); variable is an unconstrained real Symbol (domain Reals, no
assumptions); relation is "=", "<", "<=", ">" or ">=" (kind "relation", default "="). The problem is lhs relation rhs,
that is F relation 0 with F = lhs - rhs, over the reals. The expressions are read node by node as their RealExpr trees
store them. real_function and the MCP decoder simplify a tree first and rewrite sqrt(x^2) as abs(x): sqrt(x^2) - x
reaches solve_sqrt only as a RealFunction built directly from its RealExpr, and through the MCP it is a solve_abs
problem.

- A polynomial expression is made of rational constants (scalar nodes holding a Rational), the variable (symbol nodes
  whose Symbol equals variable), add, mul, pow with an exact int exponent >= 0, and div by a polynomial expression
  whose polynomial is a nonzero constant.
- solve_abs: F is a polynomial expression in the variable and in atoms abs(P), each P a polynomial expression (no
  abs inside an abs). No atom at all is allowed: F is then a polynomial. Both domain predicates are TRUE.
- solve_sqrt: F = A*sqrt(P) + B with polynomials A, B and P. The nodes are those of a polynomial expression and
  sqrt(Q), each Q a polynomial expression; at least one sqrt occurs; every radicand Q is the same polynomial P
  (exactly one distinct radicand, which may occur several times); and every occurrence is linear: no mul whose two
  factors both contain a sqrt, no pow with an exponent >= 2 whose base contains one. The domain is P >= 0: each domain
  predicate is TRUE, nonnegative(Q) with Q a polynomial expression of the polynomial P, a conjunction ("and") of such
  predicates, or FALSE when P is a negative constant.

Anything else is invalid input whose message names the construct: an argument that is no RealFunction, a function of
another variable, an irrational constant (a RealAlgebraicRoot scalar), pi, E, any other function (sqrt in solve_abs,
abs in solve_sqrt, sin, ...), another symbol, a negative power, a division by a non-constant or by zero, an abs inside
an abs, a sqrt inside a sqrt, no sqrt or a second radicand in solve_sqrt, a nonlinear square root, any other domain
predicate.

Value: the solution set {x real : x in the domain and F(x) relation 0} in the canonical form of v0.7's solve real-set
path, IntervalSet(...).to_solution_set(): EmptySet(Reals) when it is empty, UniversalSet(Reals) when it is the whole
line, and otherwise the canonical IntervalSet: sorted, disjoint, unmergeable and nonempty intervals, infinite ends
open, a single point as a closed degenerate interval (a set of points is such an IntervalSet, never a FiniteSet). A
rational endpoint is a Rational. An irrational endpoint is RealAlgebraicRoot(Q, i) with Q of degree >= 2; the producer
writes Q as the minimal polynomial of the endpoint over the rationals (primitive, positive leading coefficient, as
the class stores it) and i the index of the endpoint among the real roots of Q. The checker accepts as Q any
polynomial of degree >= 2 that divides one problem polynomial (below) and has no rational root, which the minimal
polynomial does, and refuses every other name: a factor outside the problem, a rational root, a degree of one.

Evidence: (cuts, samples), two strictly increasing tuples of Rationals.
- The problem polynomials. solve_abs: every nonconstant atom P, and every piece G_K that is not constant, where K runs
  over the sign patterns of the atoms on the open intervals between their real roots and G_K is F with each abs(P)
  replaced by sign_K(P)*P (G_K is identically zero on [-2, 1] for |x - 1| + |x + 2| = 3). solve_sqrt: those of P, A, B
  and N = A^2*P - B^2 that are not constant.
- The points: every distinct real root of every problem polynomial, also a root of a piece outside the interval of its
  pattern. The polynomials of the claim's RealAlgebraicRoot endpoints add no point: each divides a problem polynomial.
- cuts lists rational points: the producer lists every rational root of every problem polynomial. Each must be a root
  of a problem polynomial. (A rational root left out stays a root of the checker's master polynomial; the checker
  accepts that as well.)
- samples holds one rational in each open interval between two neighbouring points, one below every point and one
  above every point: exactly one sample more than points, none of them a point (one sample for the whole line when
  there is no point).

The family defines no record class: its values are v0.7's EmptySet, UniversalSet and IntervalSet, which serialize
and render without one, so TYPES is empty and there is no RENDER. Its checker, mathforge.certify.equations.piecewise,
states the theorems and the exact layouts that it accepts. mathforge.areas.ntheory.continued_fraction spells out the
contract of a v6 family, documentation/families.md is the checklist, tests/v09/equations/test_piecewise.py is the
definition of done, and docs/v09/cards/m2-algebra-equations-sets-numeric.md is the specification.

How produce() computes. Nothing here is a proof: the checker verifies every claim on its own cut (signs at the samples
and Sturm counts of mathforge.certify.cells), and never factors anything.

- It validates the arguments, then reads the two trees once each, operands first and every record once (a record that
  contains itself is invalid input): every node free of abs and sqrt gets its polynomial over Q, and each construct
  outside the operation's class is refused there, before any root is sought.
- solve_abs collects the atoms. The real roots of the nonconstant atoms cut the line into the card's H0 cells, and the
  signs of the atoms at the sample of a cell are its pattern K; the piece G_K is F with every abs(P) replaced by
  K(P) * P, formed over the same nodes, once per pattern.
- solve_sqrt reads F = A * sqrt(P) + B over the nodes of F (a sqrt is (1, 0), a node free of sqrt (0, its polynomial);
  add, mul by a factor free of sqrt, div by a constant and the powers 0 and 1 follow), checks every domain predicate
  against P, and forms N = A^2 P - B^2.
- The points. Each problem polynomial is factored over Z: its square-free decomposition, then Zassenhaus' algorithm on
  each square-free factor (mathforge.kernels.upoly). Distinct irreducible factors share no root, so the points are the
  roots of the distinct factors, each found once. A linear factor gives a rational point; a factor M of degree two or
  more has irrational roots only, isolated by kernels.upoly.isolate and named RealAlgebraicRoot(M, i), M being the
  minimal polynomial of each and i its place in the isolation, which sorts the roots of M. The isolating intervals are
  bisected on the sign of their factor (a simple root inside, so the sign at the lower end tells the half that holds
  it) until no two of them meet and none holds a rational point. The samples are the rationals of least denominator
  strictly between neighbouring points, the greatest integer below the first point and the least above the last.
- Truth. On an open cell, the sign of F at its sample: solve_abs evaluates the piece of the sample's pattern, which is
  F on the whole cell; solve_sqrt decides from the signs of P, A, B and N there by the rule of the card. At a point,
  the sign of a polynomial Q (the piece of the cell above the point, or each of P, A, B and N): at a rational point by
  evaluation, at an irrational one 0 when the point's minimal polynomial is a factor of Q, and otherwise the sign of Q
  at the sample above, since every real root of Q is a point and no point lies between the two. F is continuous, so at
  a point of solve_abs it equals the piece of either neighbouring cell.
- The value joins the maximal runs of true members, in the order cell 0, point 1, cell 1, ..., point r, cell r, into
  IntervalSet(...).to_solution_set(); the evidence lists the rational points (cuts) and the samples.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and one per node, per pass over the points and per
bisection (a refinement as well); the polynomial arithmetic of trusted.upoly and the kernels' square-free
decomposition, factorization (whose Berlekamp matrix counts against max_matrix_entries, so a square-free factor of
degree above 64 exceeds the default limits) and isolation charge their own work. An integer of the arguments longer than
max_integer_bits, or a polynomial whose coefficients outgrow it, raises ResourceLimitError.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Rational, Reals, Symbol
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
# The arguments of both operations, in this order: two functions of the variable, the variable and the relation.
_ARGUMENTS = (Argument("lhs", "RealFunction", None, "function"), Argument("rhs", "RealFunction", None, "function"),
              Argument("variable", "Symbol", None, "symbol"), Argument("relation", "str", "'='", "relation"))
# The value is v0.7's EquationSolutionSet without FiniteSet. tools/generate_api.py must resolve EmptySet and
# UniversalSet (module mathforge.contracts) through NAME_SOURCES before the family is registered.
_RETURNS = "EmptySet | UniversalSet | IntervalSet"

SPECS = (
    OperationSpec("solve_abs", "equations", _ARGUMENTS, _RETURNS,
                  "Solve an equation or inequality built from polynomials and absolute values of polynomials over the "
                  "reals.", _SCOPE, version=6),
    OperationSpec("solve_sqrt", "equations", _ARGUMENTS, _RETURNS,
                  "Solve an equation or inequality linear in one square root of a polynomial over the reals.", _SCOPE,
                  version=6),
)

# No record classes: the values are v0.7's EmptySet, UniversalSet and IntervalSet records, which the registry does not
# let a v6 family claim as its own (every TYPES class must be defined in the family module).
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.equations.piecewise"

_OPERATIONS = ("solve_abs", "solve_sqrt")
_NAMES = frozenset(("lhs", "rhs", "variable", "relation"))
# Whether a number of the sign -1, 0 or 1 satisfies "number relation 0", by relation.
_HOLDS = {"=": (False, True, False), "<": (True, False, False), "<=": (True, True, False),
          ">": (False, False, True), ">=": (False, True, True)}
_MISSING = object()  # A field that a record made without its constructor lacks.
_OUTSIDE = "{} lies outside the class of {}"

# A polynomial over Q is (numerators, denominator): ints lowest degree first without a trailing zero, over a positive
# denominator that shares no factor with all of them, so that equal polynomials are equal pairs (the radicands of
# solve_sqrt are compared so). Zero is ((), 1).
_ZERO: tuple[tuple[int, ...], int] = ((), 1)
_ONE: tuple[tuple[int, ...], int] = ((1,), 1)
_X: tuple[tuple[int, ...], int] = ((0, 1), 1)


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of solve_abs or solve_sqrt, for arguments that operations_v6.run has checked by name; the
    module states the class, the algorithm and the layout of the evidence. InvalidInput for arguments outside the
    operation's class, ResourceLimitError past the budget."""
    if type(operation) is not str or operation not in _OPERATIONS:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the piecewise family")
    if type(args) is not dict or set(args) != _NAMES:
        raise InvalidInput(f"{operation} takes exactly the arguments lhs, rhs, variable and relation")
    with computation() as budget:
        budget.tick()  # The call itself; every valid input charges more (its nodes at least).
        variable, relation, functions = _arguments(operation, args)
        tree = _Tree(budget, operation, variable)
        left = tree.compile(_field(functions[0], "expression"))
        right = tree.compile(_field(functions[1], "expression"))
        if operation == "solve_abs":
            return _solve_abs(budget, tree, functions, left, right, relation)
        return _solve_sqrt(budget, tree, functions, left, right, relation)


# ---------------------------------------------------------------------------------------------------------------
# The two operations

def _solve_abs(budget: Budget, tree: _Tree, functions: list, left: int, right: int,
               relation: str) -> tuple[object, tuple]:
    """The solution set of F relation 0 for F a polynomial expression in the variable and atoms abs(P)."""
    for function in functions:
        conjuncts = _conjuncts(budget, _field(function, "domain"))
        if conjuncts != [("true", None)]:
            raise InvalidInput("solve_abs takes functions defined on the whole real line: the domain restriction "
                               f"{conjuncts[0][0]} lies outside its class")
    atoms: list = []  # The distinct polynomials under an abs, in their order of appearance.
    numbers: dict = {}
    atom_of: dict[int, int] = {}  # The atom number of every abs node.
    for i, op in enumerate(tree.ops):
        budget.tick()
        if op == "abs":
            atom = tree.polynomials[tree.operands[i][0]]
            if atom not in numbers:
                numbers[atom] = len(atoms)
                atoms.append(atom)
            atom_of[i] = numbers[atom]
    integer_atoms = [_integer(atom) for atom in atoms]
    line = _Line(budget)
    for atom in integer_atoms:
        if len(atom) > 1:
            line.add(atom)
    _, cell_samples = line.build()  # The H0 cells: one sample in each.
    with_abs = [i for i in range(len(tree.ops)) if tree.has_abs[i]]
    pieces: dict[tuple[int, ...], tuple[int, ...]] = {}  # Pattern -> the integer form of G_K.
    for sample in cell_samples:
        pattern = _pattern(budget, integer_atoms, sample)
        if pattern not in pieces:
            pieces[pattern] = _integer(_piece(budget, tree, with_abs, atoms, atom_of, pattern, left, right))
    for piece in pieces.values():
        if len(piece) > 1:
            line.add(piece)
    roots, samples = line.build()
    cell_truth, above, signs = [], [], []
    for sample in samples:
        piece = pieces.get(_pattern(budget, integer_atoms, sample))
        if piece is None:  # Every cell of the cut lies in a cell of the atoms: only a bug gets here.
            raise ArithmeticError("A cell of the cut has a sign pattern of no cell of the atoms")
        sign = _sign_at(piece, sample)
        above.append(piece)
        signs.append(sign)
        cell_truth.append(_HOLDS[relation][sign + 1])
    point_truth = []
    for k, root in enumerate(roots):
        budget.tick()
        sign = line.sign(above[k + 1], root, signs[k + 1])  # F equals the piece of the cell above on its closure.
        point_truth.append(_HOLDS[relation][sign + 1])
    return _outcome(budget, roots, samples, cell_truth, point_truth)


def _solve_sqrt(budget: Budget, tree: _Tree, functions: list, left: int, right: int,
                relation: str) -> tuple[object, tuple]:
    """The solution set of F relation 0 for F = A*sqrt(P) + B over the domain P >= 0."""
    count = len(tree.ops)  # The nodes of F; those of the domain predicates follow.
    conjuncts: list = []
    for function in functions:
        conjuncts.extend(_conjuncts(budget, _field(function, "domain")))
    operands = [tree.compile(operand) if kind == "nonnegative" else None for kind, operand in conjuncts]
    radicand, a, b = _linear_form(budget, tree, count, left, right)
    for (kind, _), operand in zip(conjuncts, operands):
        budget.tick()
        if kind == "false":
            if len(radicand[0]) != 1 or radicand[0][0] >= 0:
                raise InvalidInput("solve_sqrt takes an empty domain (FALSE) only with a negative constant radicand")
        elif kind == "nonnegative":
            if tree.has_abs[operand] or tree.has_sqrt[operand] or tree.polynomials[operand] != radicand:
                raise InvalidInput("solve_sqrt takes the domain radicand >= 0 only: a domain predicate on another "
                                   "expression lies outside its class")
    n = _sub(_mul(_mul(a, a), radicand), _mul(b, b))
    polynomials = [_integer(polynomial) for polynomial in (radicand, a, b, n)]
    line = _Line(budget)
    for polynomial in polynomials:
        if len(polynomial) > 1:
            line.add(polynomial)
    roots, samples = line.build()
    cell_truth, columns = [], []
    for sample in samples:
        budget.tick()
        signs = tuple([_sign_at(polynomial, sample) for polynomial in polynomials])
        columns.append(signs)
        cell_truth.append(_truth(relation, _decide(*signs)))
    point_truth = []
    for k, root in enumerate(roots):
        budget.tick()
        signs = tuple([line.sign(polynomial, root, columns[k + 1][j]) for j, polynomial in enumerate(polynomials)])
        point_truth.append(_truth(relation, _decide(*signs)))
    return _outcome(budget, roots, samples, cell_truth, point_truth)


def _decide(sp: int, sa: int, sb: int, sn: int) -> int | None:
    """The sign of A*sqrt(P) + B at a point from the signs of P, A, B and N = A^2 P - B^2 there (the card's rule);
    None outside the domain, where P < 0."""
    if sp < 0:
        return None
    t = sa if sp > 0 else 0  # The sign of A*sqrt(P).
    if t == 0:
        return sb
    if sb == 0 or sb == t:
        return t
    return t * sn  # Opposite signs: |A sqrt(P)| > |B| exactly when A^2 P > B^2.


def _truth(relation: str, sign: int | None) -> bool:
    return sign is not None and _HOLDS[relation][sign + 1]


def _pattern(budget: Budget, atoms: list, sample: Fraction) -> tuple[int, ...]:
    """The signs of the atoms (integer forms with their signs) at a sample."""
    budget.tick(len(atoms))
    return tuple([_sign_at(atom, sample) for atom in atoms])


def _piece(budget: Budget, tree: _Tree, with_abs: list, atoms: list, atom_of: dict, pattern: tuple, left: int,
           right: int) -> tuple:
    """G_K over Q: F with every abs(P_i) replaced by K(i) * P_i, formed over the nodes of F that contain an abs."""
    values: dict[int, tuple] = {}

    def value(i: int) -> tuple:
        return values[i] if tree.has_abs[i] else tree.polynomials[i]

    for i in with_abs:
        budget.tick()
        op = tree.ops[i]
        if op == "abs":
            number = atom_of[i]
            sign, atom = pattern[number], atoms[number]
            values[i] = atom if sign > 0 else _neg(atom) if sign < 0 else _ZERO
        else:
            values[i] = _operation(budget, op, tree.payloads[i], [value(operand) for operand in tree.operands[i]])
    return _sub(value(left), value(right))


def _linear_form(budget: Budget, tree: _Tree, count: int, left: int, right: int) -> tuple[tuple, tuple, tuple]:
    """(P, A, B) with F = A*sqrt(P) + B, read over the first count nodes (those of F): one radicand, which may occur
    several times, and only linear occurrences of its square root."""
    radicand = None
    pairs: dict[int, tuple] = {}

    def pair(i: int) -> tuple:
        return pairs[i] if tree.has_sqrt[i] else (_ZERO, tree.polynomials[i])

    for i in range(count):
        if not tree.has_sqrt[i]:
            continue
        budget.tick()
        op, operands, payload = tree.ops[i], tree.operands[i], tree.payloads[i]
        if op == "sqrt":
            value = tree.polynomials[operands[0]]  # Free of sqrt (no nesting) and of abs (refused in solve_sqrt).
            if radicand is None:
                radicand = value
            elif value != radicand:
                raise InvalidInput("solve_sqrt takes one radicand: a second radicand lies outside its class")
            pairs[i] = (_ONE, _ZERO)
        elif op == "add":
            (a1, b1), (a2, b2) = pair(operands[0]), pair(operands[1])
            pairs[i] = (_add(a1, a2), _add(b1, b2))
        elif op == "mul":
            first, second = operands
            if tree.has_sqrt[first] and tree.has_sqrt[second]:
                raise InvalidInput("solve_sqrt is linear in its square root: a product of two square roots lies "
                                   "outside its class")
            if not tree.has_sqrt[first]:
                first, second = second, first
            (a, b), factor = pair(first), tree.polynomials[second]
            pairs[i] = (_mul(a, factor), _mul(b, factor))
        elif op == "div":  # By a nonzero constant free of sqrt, as the tree verified.
            (a, b), divisor = pair(operands[0]), tree.polynomials[operands[1]]
            pairs[i] = (_divide(a, divisor), _divide(b, divisor))
        elif op == "pow":
            if payload >= 2:
                raise InvalidInput("solve_sqrt is linear in its square root: a power of a square root lies outside "
                                   "its class")
            pairs[i] = pair(operands[0]) if payload == 1 else (_ZERO, _ONE)
        else:  # The tree refuses abs in solve_sqrt, and no other node contains a sqrt.
            raise ArithmeticError(f"An unexpected {op} node holds a square root")
    if radicand is None:
        raise InvalidInput("solve_sqrt needs a square root of a polynomial in lhs or rhs")
    (a1, b1), (a2, b2) = pair(left), pair(right)
    return radicand, _sub(a1, a2), _sub(b1, b2)


def _outcome(budget: Budget, roots: list, samples: list, cell_truth: list, point_truth: list) -> tuple[object, tuple]:
    """(the canonical solution set of the truth values, (cuts, samples)). The members are cell 0, point 1, cell 1, ...,
    point r, cell r; each maximal run of true members is one interval."""
    from ...algebraic import RealAlgebraicRoot
    from ...intervals import Interval, IntervalSet
    from ...root_isolation import root_work

    ends: dict[int, object] = {}

    def end(k: int) -> object:
        """The endpoint record of point k (from 0): a Rational, or the RealAlgebraicRoot of its minimal polynomial."""
        if k not in ends:
            root = roots[k]
            if root.lower == root.upper:
                ends[k] = Rational(root.lower.numerator, root.lower.denominator)
            else:
                ends[k] = RealAlgebraicRoot(root.factor, root.index)
        return ends[k]

    members = [cell_truth[0]]
    for k in range(len(roots)):
        members += [point_truth[k], cell_truth[k + 1]]
    last = len(members) - 1
    budget.tick(len(members))
    intervals = []
    i = 0
    while i <= last:
        if not members[i]:
            i += 1
            continue
        j = i
        while j < last and members[j + 1]:
            j += 1
        # Member 2k - 1 is point k (roots[k - 1]) and member 2k cell k, so roots[i // 2] is point (i + 1) // 2.
        if i % 2:
            lower, left_closed = end(i // 2), True
        else:
            lower, left_closed = (None if i == 0 else end(i // 2 - 1)), False
        if j % 2:
            upper, right_closed = end(j // 2), True
        else:
            upper, right_closed = (None if j == last else end(j // 2)), False
        intervals.append((lower, upper, left_closed, right_closed))
        i = j + 1
    with root_work():  # One cache of root isolations for every comparison of the constructors below.
        value = IntervalSet(tuple([Interval(*interval) for interval in intervals])).to_solution_set()
    cuts = tuple([end(k) for k, root in enumerate(roots) if root.lower == root.upper])
    budget.tick(len(samples))
    return value, (cuts, tuple([Rational(sample.numerator, sample.denominator) for sample in samples]))


# ---------------------------------------------------------------------------------------------------------------
# Reading the arguments and the trees

def _field(record: object, name: str) -> object:
    """A field of a record, read from the record itself: a record made without its constructor that lacks the field
    is invalid input (getattr would answer a class-level default, such as RealFunction's domain TRUE)."""
    try:
        fields = vars(record)
    except TypeError:  # No __dict__: slots, which hold no default (Rational).
        value = getattr(record, name, _MISSING)
    else:
        value = fields.get(name, _MISSING) if type(fields) is dict else _MISSING
    if value is _MISSING:
        raise InvalidInput(f"A {type(record).__name__} record lacks its field {name}")
    return value


def _symbol_key(value: object) -> tuple[str, str] | None:
    """(name, uid) of an unconstrained real Symbol (domain Reals, no assumptions); None for anything else."""
    if type(value) is not Symbol:
        return None
    try:
        name, domain, assumptions, uid = (_field(value, field) for field in ("name", "domain", "assumptions", "uid"))
    except InvalidInput:
        return None
    if type(name) is not str or domain is not Reals or type(assumptions) is not tuple or assumptions:
        return None
    return (name, uid) if type(uid) is str else None


def _arguments(operation: str, args: dict[str, object]) -> tuple[tuple[str, str], str, list]:
    """(the variable's name and uid, relation, [lhs, rhs]), each validated: an unconstrained real Symbol, one of the
    five relations, and two RealFunctions whose variables are () or (variable,)."""
    from ...functions import RealFunction

    variable = args["variable"]
    key = _symbol_key(variable)
    if key is None:
        raise InvalidInput(f"{operation} requires variable to be an unconstrained real Symbol")
    relation = args["relation"]
    if type(relation) is not str or relation not in _HOLDS:
        raise InvalidInput(f"{operation} requires relation to be '=', '<', '<=', '>' or '>='")
    functions = []
    for name in ("lhs", "rhs"):
        function = args[name]
        if type(function) is not RealFunction:
            raise InvalidInput(f"{operation} requires {name} to be a RealFunction of the variable")
        axes = _field(function, "variables")
        if type(axes) is not tuple or len(axes) > 1 or (axes and _symbol_key(axes[0]) != key):
            raise InvalidInput(f"{operation} requires {name} to be a function of the variable {key[0]} alone")
        functions.append(function)
    return key, relation, functions


def _conjuncts(budget: Budget, predicate: object) -> list[tuple[str, object]]:
    """The conjuncts of a domain predicate as (kind, operand): ("true", None), ("false", None) or ("nonnegative", a
    RealExpr record); a conjunction gives its parts, which must all be nonnegative predicates. Any other predicate
    (positive, nonzero, a nested and) lies outside both classes."""
    from ...functions import DomainPredicate

    if type(predicate) is not DomainPredicate:
        raise InvalidInput("The domain of a RealFunction is a DomainPredicate")
    kind, operands = _field(predicate, "kind"), _field(predicate, "operands")
    if type(kind) is not str or type(operands) is not tuple:
        raise InvalidInput("A DomainPredicate holds a kind and a tuple of operands")
    if kind in ("true", "false") and not operands:
        return [(kind, None)]
    if kind == "nonnegative" and len(operands) == 1:
        return [(kind, operands[0])]
    if kind == "and" and len(operands) >= 2:
        parts = []
        for part in operands:
            budget.tick()
            if type(part) is not DomainPredicate:
                raise InvalidInput("A conjunction holds DomainPredicates")
            inner, items = _field(part, "kind"), _field(part, "operands")
            # An exact str: the DomainPredicate constructor admits a str subclass equal to "nonnegative", which the
            # checker refuses, so accepting it here would yield a claim that fails verification.
            if type(inner) is not str or inner != "nonnegative" or type(items) is not tuple or len(items) != 1:
                raise InvalidInput(_OUTSIDE.format(f"The domain restriction {inner}", "both piecewise operations"))
            parts.append((inner, items[0]))
        return parts
    raise InvalidInput(_OUTSIDE.format(f"The domain restriction {kind}", "both piecewise operations"))


class _Tree:
    """The RealExpr trees of one problem as numbered nodes, operands before their parents, each record once (by
    identity): its operation, operand numbers and payload ((numerator, denominator) of a constant, the exponent of a
    power), whether it contains an abs or a sqrt, and, when it contains neither, its polynomial over Q. Reading a node
    validates it against the grammar of the operation's class (module docstring)."""

    def __init__(self, budget: Budget, operation: str, variable: tuple[str, str]) -> None:
        self.budget = budget
        self.operation = operation
        self.variable = variable
        self.ops: list[str] = []
        self.operands: list[tuple[int, ...]] = []
        self.payloads: list[object] = []
        self.has_abs: list[bool] = []
        self.has_sqrt: list[bool] = []
        self.polynomials: list = []
        self._index: dict[int, int] = {}  # id(record) -> node number; the records live in the arguments meanwhile.

    def compile(self, root: object) -> int:
        """The number of a tree's root node, its new nodes appended operands first (an explicit stack, no recursion).
        A record met again while its own operands are pending contains itself: invalid input."""
        index = self._index
        if id(root) in index:
            return index[id(root)]
        pending: dict[int, tuple[str, tuple, object]] = {}
        stack: list[tuple[object, bool]] = [(root, False)]
        while stack:
            self.budget.tick()
            record, ready = stack.pop()
            key = id(record)
            if ready:
                op, operands, payload = pending.pop(key)
                index[key] = self._append(op, tuple([index[id(operand)] for operand in operands]), payload)
                continue
            if key in index:
                continue
            if key in pending:
                raise InvalidInput("A RealExpr tree that contains itself")
            pending[key] = self._read(record)
            stack.append((record, True))
            stack.extend([(operand, False) for operand in pending[key][1]])
        return index[id(root)]

    def _read(self, record: object) -> tuple[str, tuple, object]:
        """(op, operand records, payload) of one RealExpr record, validated against the class's grammar."""
        from ...algebraic import RealAlgebraicRoot
        from ...functions import RealExpr

        operation = self.operation
        if type(record) is not RealExpr:
            raise InvalidInput(f"{operation} requires lhs and rhs to hold RealExpr trees")
        op, operands, payload = (_field(record, name) for name in ("op", "args", "value"))
        if type(op) is not str or type(operands) is not tuple:
            raise InvalidInput("A RealExpr holds an operation name and a tuple of operands")
        if op == "scalar":
            if operands:
                raise InvalidInput("A RealExpr constant has no operand")
            if type(payload) is RealAlgebraicRoot:
                raise InvalidInput(_OUTSIDE.format(f"The irrational constant {payload}", operation))
            if type(payload) is not Rational:
                raise InvalidInput("A RealExpr constant is an exact Rational")
            numerator, denominator = _field(payload, "numerator"), _field(payload, "denominator")
            if type(numerator) is not int or type(denominator) is not int:
                raise InvalidInput("A Rational holds exact ints")
            self.budget.check_int(numerator)
            self.budget.check_int(denominator)
            if denominator < 1 or gcd(numerator, denominator) != 1:
                raise InvalidInput("A Rational holds its lowest terms with a positive denominator")
            return op, (), (numerator, denominator)
        if op == "symbol":
            if operands or _symbol_key(payload) != self.variable:
                name = getattr(payload, "name", payload)
                raise InvalidInput(_OUTSIDE.format(f"The symbol {name}, which is not the variable "
                                                   f"{self.variable[0]},", operation))
            return op, (), None
        if op in ("add", "mul", "div"):
            if len(operands) != 2 or payload is not None:
                raise InvalidInput(f"A RealExpr {op} has two operands")
        elif op == "pow":
            if len(operands) != 1 or type(payload) is not int:
                raise InvalidInput("A RealExpr power has one operand and an exact int exponent")
            self.budget.check_int(payload)
            if payload < 0:
                raise InvalidInput(_OUTSIDE.format(f"The negative power {payload}", operation))
        elif op in ("abs", "sqrt"):
            if len(operands) != 1 or payload is not None:
                raise InvalidInput(f"A RealExpr {op} has one operand")
            if op == "sqrt" and operation == "solve_abs":
                raise InvalidInput(_OUTSIDE.format("sqrt", operation) + "; solve_sqrt solves one square root")
            if op == "abs" and operation == "solve_sqrt":
                raise InvalidInput(_OUTSIDE.format("abs", operation) + "; solve_abs solves absolute values")
        elif op in ("pi", "E"):
            raise InvalidInput(_OUTSIDE.format(f"The constant {op}", operation))
        else:
            raise InvalidInput(_OUTSIDE.format(f"The function {op}", operation))
        return op, operands, payload

    def _append(self, op: str, children: tuple[int, ...], payload: object) -> int:
        """Append a validated node: its flags, the checks that need its operands, and its polynomial over Q when it
        contains no abs and no sqrt."""
        has_abs = op == "abs" or any([self.has_abs[child] for child in children])
        has_sqrt = op == "sqrt" or any([self.has_sqrt[child] for child in children])
        if op == "abs" and self.has_abs[children[0]]:
            raise InvalidInput(_OUTSIDE.format("An abs inside an abs", self.operation))
        if op == "sqrt" and self.has_sqrt[children[0]]:
            raise InvalidInput(_OUTSIDE.format("A sqrt inside a sqrt", self.operation))
        if op == "div":
            divisor = children[1]
            if self.has_abs[divisor] or self.has_sqrt[divisor] or len(self.polynomials[divisor][0]) > 1:
                raise InvalidInput(_OUTSIDE.format("A division by a non-constant", self.operation))
            if not self.polynomials[divisor][0]:
                raise InvalidInput(_OUTSIDE.format("A division by zero", self.operation))
        polynomial = None
        if not has_abs and not has_sqrt:
            if op == "scalar":
                polynomial = ((payload[0],), payload[1]) if payload[0] else _ZERO  # type: ignore[index]
            elif op == "symbol":
                polynomial = _X
            else:
                polynomial = _operation(self.budget, op, payload, [self.polynomials[child] for child in children])
        self.ops.append(op)
        self.operands.append(children)
        self.payloads.append(payload)
        self.has_abs.append(has_abs)
        self.has_sqrt.append(has_sqrt)
        self.polynomials.append(polynomial)
        return len(self.ops) - 1


# ---------------------------------------------------------------------------------------------------------------
# Polynomials over Q: (numerators, positive denominator), charged through trusted.upoly

def _canonical(numerators: tuple[int, ...], denominator: int) -> tuple[tuple[int, ...], int]:
    """The canonical pair of numerators / denominator (denominator > 0, numerators without a trailing zero);
    ResourceLimitError when an integer of it outgrows max_integer_bits."""
    from ...trusted import upoly as tp

    if not numerators:
        return _ZERO
    if denominator != 1:
        tp.charge_gcd(max(tp.max_bits(numerators), denominator.bit_length()), len(numerators))
        divisor = gcd(denominator, *numerators)
        if divisor != 1:
            numerators = tuple([value // divisor for value in numerators])
            denominator //= divisor
    tp.check_bits(denominator)
    if numerators:
        tp.check_bits(max(max(numerators), -min(numerators)))
    return numerators, denominator


def _operation(budget: Budget, op: str, payload: object, values: list) -> tuple:
    """add, mul, div (by a nonzero constant) or pow of polynomials over Q."""
    if op == "add":
        return _add(values[0], values[1])
    if op == "mul":
        return _mul(values[0], values[1])
    if op == "div":
        return _divide(values[0], values[1])
    return _power(budget, values[0], payload)  # type: ignore[arg-type]


def _add(p: tuple, q: tuple) -> tuple:
    from ...trusted import upoly as tp

    (a, da), (b, db) = p, q
    if not a:
        return q
    if not b:
        return p
    if da == db:
        return _canonical(tp.add(a, b), da)
    tp.charge_product(tp.max_bits(a) + tp.max_bits(b), da.bit_length() + db.bit_length(), max(len(a), len(b)) + 1)
    return _canonical(tp.add(tp.scale(a, db), tp.scale(b, da)), da * db)


def _neg(p: tuple) -> tuple:
    return tuple([-value for value in p[0]]), p[1]


def _sub(p: tuple, q: tuple) -> tuple:
    return _add(p, _neg(q))


def _mul(p: tuple, q: tuple) -> tuple:
    from ...trusted import upoly as tp

    (a, da), (b, db) = p, q
    if not a or not b:
        return _ZERO
    tp.charge_product(da.bit_length(), db.bit_length())
    return _canonical(tp.mul(a, b), da * db)


def _divide(p: tuple, constant: tuple) -> tuple:
    """p divided by a nonzero constant polynomial ((k,), d): p times d / k."""
    from ...trusted import upoly as tp

    (a, da), ((k,), d) = p, constant
    if not a:
        return _ZERO
    tp.charge_product(tp.max_bits(a), d.bit_length(), len(a))
    tp.charge_product(da.bit_length(), k.bit_length())
    return _canonical(tp.scale(a, d if k > 0 else -d), da * abs(k))


def _power(budget: Budget, p: tuple, exponent: int) -> tuple:
    """p ** exponent by repeated squaring (p ** 0 = 1, also for p = 0, as the trees' evaluation has it)."""
    result, base = _ONE, p
    while exponent:
        budget.tick()
        if exponent & 1:
            result = _mul(result, base)
        exponent >>= 1
        if exponent:
            base = _mul(base, base)
    return result


def _integer(p: tuple) -> tuple[int, ...]:
    """The integer polynomial with the sign of p at every point: its numerators (over a positive denominator) divided
    by their positive content; () for zero."""
    from ...trusted import upoly as tp

    numerators = p[0]
    if not numerators:
        return ()
    tp.charge_gcd(tp.max_bits(numerators), len(numerators))
    content = gcd(*numerators)
    return numerators if content == 1 else tuple([value // content for value in numerators])


def _sign_at(polynomial: tuple[int, ...], point: Fraction) -> int:
    """The sign of an integer polynomial at a rational point, by exact evaluation (charged by trusted.upoly)."""
    from ...trusted import upoly as tp

    return tp.sign_at(polynomial, point.numerator, point.denominator)


# ---------------------------------------------------------------------------------------------------------------
# The points: the real roots of the problem polynomials, separated and sampled

class _Root:
    """A real root of an irreducible factor (primitive, positive leading coefficient): a rational one with lower ==
    upper, its value; an irrational one inside the open interval (lower, upper), where its factor has no other root,
    with index its place among the real roots of the factor and below the sign of the factor at lower."""

    __slots__ = ("lower", "upper", "factor", "index", "below")

    def __init__(self, lower: Fraction, upper: Fraction, factor: tuple[int, ...], index: int, below: int) -> None:
        self.lower, self.upper, self.factor, self.index, self.below = lower, upper, factor, index, below


class _Line:
    """The real roots of some nonconstant integer polynomials, each value once, with the distinct irreducible factors
    of each polynomial; build() orders and separates them and samples the open cells between them."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.factors_of: dict[tuple[int, ...], frozenset] = {}  # Polynomial -> its distinct irreducible factors.
        self.known: set[tuple[int, ...]] = set()  # Every irreducible factor met, whose roots are in self.roots.
        self.roots: list[_Root] = []

    def add(self, polynomial: tuple[int, ...]) -> None:
        """Factor a nonconstant integer polynomial over Z and add the roots of its new irreducible factors."""
        if polynomial in self.factors_of:
            return
        from ...kernels import upoly as kernels
        from ...trusted import upoly as tp

        found = set()
        for part, _ in kernels.square_free_integer(tp.primitive(polynomial)):
            for factor, _ in kernels.factor_square_free(part):
                found.add(factor)
                if factor not in self.known:
                    self.known.add(factor)
                    self.roots.extend(self._roots(factor))
        self.factors_of[polynomial] = frozenset(found)

    def _roots(self, factor: tuple[int, ...]) -> list[_Root]:
        """The real roots of an irreducible factor: the rational root of a linear one, else isolating intervals."""
        from ...kernels import upoly as kernels
        from ...trusted import upoly as tp

        if len(factor) == 2:
            root = Fraction(-factor[0], factor[1])
            return [_Root(root, root, factor, 0, 0)]
        roots = []
        for index, ((lower_n, lower_d), (upper_n, upper_d)) in enumerate(kernels.isolate(factor)):
            if (lower_n, lower_d) == (upper_n, upper_d):  # An irreducible factor of degree two or more.
                raise ArithmeticError("An irreducible factor of degree two or more has a rational root")
            below = tp.sign_at(factor, lower_n, lower_d)
            roots.append(_Root(Fraction(lower_n, lower_d), Fraction(upper_n, upper_d), factor, index, below))
        return roots

    def _refine(self, root: _Root) -> None:
        """Halve the isolating interval of an irrational root: its factor changes sign across the one simple root
        inside, so the sign at the midpoint tells the half that holds it."""
        from ...trusted import upoly as tp

        self.budget.refine()
        middle = (root.lower + root.upper) / 2
        sign = tp.sign_at(root.factor, middle.numerator, middle.denominator)
        if not sign:
            raise ArithmeticError("An irreducible factor of degree two or more vanishes at a rational point")
        if sign == root.below:
            root.lower = middle
        else:
            root.upper = middle

    def build(self) -> tuple[list[_Root], list[Fraction]]:
        """(roots, samples): the roots in increasing order, their intervals refined until no two meet and none holds a
        rational root, and r + 1 samples for r roots (module docstring)."""
        roots = sorted(self.roots, key=_lower)
        while True:
            self.budget.tick(1 + len(roots))
            meeting = False
            for first, second in zip(roots, roots[1:]):
                if second.lower <= first.upper:  # Closed intervals sorted by their lower ends: neighbours suffice.
                    if first.lower == first.upper and second.lower == second.upper:
                        raise ArithmeticError("Two distinct linear factors share a root")
                    meeting = True
                    for root in (first, second):
                        if root.lower != root.upper:
                            self._refine(root)
            if not meeting:
                break
            roots.sort(key=_lower)
        if not roots:
            return roots, [Fraction(0)]
        lowest = roots[0].lower
        samples = [Fraction(-((-lowest.numerator) // lowest.denominator) - 1)]  # The greatest integer below.
        for first, second in zip(roots, roots[1:]):
            self.budget.tick()
            samples.append(_simplest(self.budget, first.upper, second.lower))
        highest = roots[-1].upper
        samples.append(Fraction(highest.numerator // highest.denominator + 1))  # The least integer above.
        return roots, samples

    def sign(self, polynomial: tuple[int, ...], root: _Root, above: int) -> int:
        """The sign at a point of a polynomial that is constant or one of the polynomials added here, given its sign
        at the sample above the point: by evaluation at a rational point; at an irrational one 0 when the point's
        factor divides it, otherwise its sign above (every real root of it is a point, and none lies in between)."""
        from ...trusted import upoly as tp

        if len(polynomial) <= 1:
            return (polynomial[0] > 0) - (polynomial[0] < 0) if polynomial else 0
        if root.lower == root.upper:
            return tp.sign_at(polynomial, root.lower.numerator, root.lower.denominator)
        return 0 if root.factor in self.factors_of[polynomial] else above


def _lower(root: _Root) -> Fraction:
    return root.lower


def _simplest(budget: Budget, low: Fraction, high: Fraction) -> Fraction:
    """The rational of least denominator strictly between low < high, the one nearest to 0 among those: 0 when it lies
    between them, else by the continued fractions of the two ends (each step charged)."""
    if low < 0 < high:
        return Fraction(0)
    sign = 1
    if high <= 0:
        low, high, sign = -high, -low, -1
    prefix = []  # 0 <= low < high from here on.
    while True:
        budget.tick()
        whole = low.numerator // low.denominator
        if whole + 1 < high:  # An integer strictly inside; whole + 1 is the least, as low >= whole.
            value = Fraction(whole + 1)
            break
        if low == whole:  # (whole, high) with high <= whole + 1: whole + 1/m for the least m with 1/m < high - whole.
            gap = high - whole
            value = whole + Fraction(1, gap.denominator // gap.numerator + 1)
            break
        prefix.append(whole)  # whole < low < high <= whole + 1: x = whole + 1/y with y between the reciprocals.
        low, high = 1 / (high - whole), 1 / (low - whole)
    for whole in reversed(prefix):
        value = whole + 1 / value
    return sign * value
