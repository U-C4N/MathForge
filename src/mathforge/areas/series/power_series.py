"""Exact power series coefficients of real functions at rational centers (mathforge.series): series_coefficients.

series_coefficients(function, variable, center, order) takes a RealFunction (MCP kind "function"), its variable, a
Symbol (kind "symbol"), an exact rational center a (kind "rational": an int, or a Rational in lowest terms with a
positive denominator) and an exact int order n >= 0 (kind "int"); no argument has a default. Its domain
(docs/v09/cards/m3-analysis.md): variable is the function's only variable and n >= 0, else invalid input; n above
MAX_ORDER = 4096 is a resource limit. The value is the PowerSeries record (variable, a, (c_0, ..., c_n)), c_k the k-th
Taylor coefficient of the function at a, so that the function equals c_0 + c_1 (x - a) + ... + c_n (x - a)^n +
O((x - a)^(n+1)) near a. Its canonical form: the variable a Symbol, the center a Rational (never an int, even for an
integral center), the coefficients a nonempty tuple of exactly n + 1 Rationals (zeros included), every Rational in
lowest terms with a positive denominator. Evidence: (), always.

The value exists, and the checker mathforge.certify.series.power_series accepts it, exactly when (its docstring states
the theorem):
- every node of the function's expression (a RealExpr tree, whose shared subtrees count once) has a rational series at
  a: a scalar holding a Rational (a RealAlgebraicRoot scalar is out of scope even where it is rational); the variable;
  add and mul; div by a node whose value at a is not 0; pow with an int exponent, a negative one only of a node whose
  value at a is not 0; exp, sin, cos, tan, sinh, cosh, tanh, atan and asin of a node whose value at a is 0; log of a
  node whose value at a is 1; sqrt of a node whose value at a is q^2 for a rational q > 0; abs of a node whose value at
  a is not 0. acos, pi, E and any other symbol are out of scope, and one node out of scope puts the whole function out
  of scope, also where it would not matter (0*exp(x) at 1, in a tree that keeps the product: real_function and
  parse_function simplify it to 0);
- the expression is at most MAX_DEPTH = 200 nodes deep (the checker refuses a deeper one with ResourceLimitError);
- the function's domain holds on a neighbourhood of a, which the checker decides strictly at a: every atom
  positive(D) or nonnegative(D) of the domain has D(a) > 0 and every atom nonzero(D) has D(a) != 0, with D in scope by
  the same rules; a domain "false" fails. The intrinsic domain of an expression in scope always passes, so this
  matters only for a domain that the caller narrowed: real_function(..., domain=...), or the excluded points of a
  RationalFunction, such as (x^2 - 1)/(x - 1), the function x + 1 with the domain nonzero(x - 1), at the center 1.
Anything else answers Unevaluated(kind="series", reason=Reason("no_applicable_rule", ...)), which makes no claim: exp(x)
at 1, sin(x)/x at 0, log(x) at 2. Its request is the one that operations_v6.run builds for the same arguments.

Invalid input, besides the kinds that operations_v6.run checks first: a function that is no RealFunction, a variable
that is no unconstrained real Symbol (a str name and uid, the domain Reals, no assumptions) or not the function's only
variable (functions without variables or of two included), a center Rational that is not in lowest terms with a
positive denominator, an order below 0, and a function that no RealFunction can be: a tree or a domain of other
classes or shapes than RealExpr and DomainPredicate admit (records forged with object.__new__), a scalar Rational not
in lowest terms, a symbol other than the variable, a node or a predicate below itself. An order above MAX_ORDER and an
expression deeper than MAX_DEPTH are resource limits, as in the checker.

How produce() computes (nothing here is a proof; the checker recomputes every coefficient by its own method). It
validates the arguments, then the expression and the domain, each distinct node once, and evaluates every node of the
expression and of the domain's atoms at the center: exact rational arithmetic that applies the scope rules above, so
that the first node out of scope, a failing atom or the domain "false" gives the Unevaluated record before any series
is expanded. Then it expands the expression node by node with the truncated power series of mathforge.kernels.series,
over Q: composition with Maclaurin series where a function's argument is its value at the center plus one monomial
(exp(x), sin(2x), 1/(1 - x), sqrt(4 + x^2), ...), and Newton iteration otherwise (the reciprocal, exp through log, the
inverse square root, tan and tanh through atan and atanh, sin, cos, sinh and cosh by the half-angle formulas, log, atan
and asin as integrals); never the first-order ODE recurrences of the checker's evaluator. The kernel's docstring states
the methods and their work: the default budget expands exp(x) or cos(x) to order 4096 at once, and compositions until
the products of their series, with coefficients that grow like factorials, reach it (a product of two series of 500
coefficients of about 4,000 bits is near the limit). The check charges every operation on two coefficients by their
length, so its work can exceed the producer's many times over (a square root or a tangent costs it a product of every
pair of coefficients): after each node, produce() adds a lower bound of what the check will charge for it
(_check_work) and raises ResourceLimitError as soon as the sum exceeds max_work, since the value could not be verified.
The bound stays below the check's work (it leaves out the additions of long products, and a few rules), so a value
between the two is still produced and then refused by the verifier, within its own budget: sqrt(1 + x) at order 500
(refused by the verifier in about 1.3 seconds; at order 700 and above the bound itself exceeds the default budget).

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/series/test_power_series.py is the definition of done.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt

from ...contracts import OperationRequest
from ...deferred import Reason, Unevaluated
from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Rational, Reals, Symbol, trusted_rational
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

MAX_ORDER = 4096  # An order above it is a resource limit, in the producer and in the checker.
MAX_DEPTH = 200  # Nodes along the deepest path of an expression that the checker evaluates (it recurses per level).
_LOOP_ITERATIONS_PER_UNIT = 16  # The checker's charge for the uncharged iterations of an ODE loop (see _check_work).

# The scope text of every operation of wave M3 (docs/v09/cards/m3-analysis.md).
_SCOPE = "exact rationals and real functions; certified closed forms of sums, series and trigonometric integrals"

SPECS = (
    OperationSpec("series_coefficients", "series",
                  (Argument("function", "RealFunction", None, "function"),
                   Argument("variable", "Symbol", None, "symbol"),
                   Argument("center", "Rational | int", None, "rational"),
                   Argument("order", "int", None, "int")),
                  "PowerSeries",
                  "Compute the exact power series coefficients of a real function at a rational center up to an order.",
                  _SCOPE, version=6),
)


def _exact_rational(budget: Budget, value: object, what: str) -> None:
    """Refuse, with InvalidInput, a field that is not a Rational of two exact ints with a positive denominator (a
    Rational made with object.__new__ may lack a field or break that form); ResourceLimitError for an integer longer
    than max_integer_bits. Lowest terms are the Rational constructor's guarantee: no gcd is recomputed here for each
    of up to 4097 coefficients, and the checker, which compares them with Rationals of its own, refuses any other."""
    if type(value) is not Rational:  # An int is no Rational here, nor is a bool or a float.
        raise InvalidInput(f"The {what} of a power series is an exact Rational")
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"The {what} of a power series is a Rational of two exact integers")
    budget.check_int(numerator)
    budget.check_int(denominator)


@dataclass(frozen=True)
class PowerSeries:
    """c_0 + c_1 (x - a) + ... + c_n (x - a)^n + O((x - a)^(n+1)) at the rational center a.

    The canonical form: variable is a Symbol, center a Rational and coefficients a nonempty tuple of Rationals, every
    Rational in lowest terms with a positive denominator (an int is no Rational here, nor is a bool or a float). The
    tuple holds every coefficient up to the order, zeros included: its length is the order plus one."""
    variable: Symbol
    center: Rational
    coefficients: tuple[Rational, ...]

    def __post_init__(self):
        with computation() as budget:
            if type(self.variable) is not Symbol:
                raise InvalidInput("The variable of a power series is a Symbol")
            _exact_rational(budget, self.center, "center")
            if type(self.coefficients) is not tuple or not self.coefficients:
                raise InvalidInput("A power series has a nonempty tuple of coefficients")
            for coefficient in self.coefficients:
                budget.tick()
                _exact_rational(budget, coefficient, "coefficient")


TYPES = {"power_series": PowerSeries}
CHECKER = "mathforge.certify.series.power_series"

_OPERATION = "series_coefficients"
_ARGUMENTS = frozenset(("function", "variable", "center", "order"))
_MISSING = object()  # A field that a record made with object.__new__ lacks.
_ZERO, _ONE = trusted_rational(0, 1), trusted_rational(1, 1)
_BINARY = frozenset(("add", "mul", "div"))
_UNARY = frozenset(("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "sqrt", "abs"))
_VANISHING = frozenset(("exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "atan", "asin"))  # Expanded about 0.
_ATOMS = frozenset(("positive", "nonnegative", "nonzero"))
_MALFORMED = "function must be a RealFunction: a tree of RealExpr nodes with a DomainPredicate domain"
_ONLY_VARIABLE = "variable must be the only variable of function"


class _OutsideScope(Exception):
    """A node or a domain atom without a rational power series at the center: the result is Unevaluated."""


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of series_coefficients, as the module states them: a PowerSeries and (), or the Unevaluated
    record and () for a function out of scope at the center. InvalidInput for invalid arguments, ResourceLimitError
    for an order above MAX_ORDER, an expression deeper than MAX_DEPTH and work past the budget."""
    if type(operation) is not str or operation != _OPERATION:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the power_series family")
    if type(args) is not dict or set(args) != _ARGUMENTS:
        raise InvalidInput("series_coefficients takes exactly the arguments function, variable, center and order")
    from ...functions import RealFunction

    function, variable, center, order = args["function"], args["variable"], args["center"], args["order"]
    if type(function) is not RealFunction:
        raise InvalidInput("function must be a RealFunction")
    if not _real_symbol(variable):
        raise InvalidInput("variable must be an unconstrained real Symbol")
    if type(center) is not int and type(center) is not Rational:  # Neither a bool nor a float is exact.
        raise InvalidInput("center must be an exact rational: an int or a Rational")
    if type(order) is not int:
        raise InvalidInput("order must be an exact int")
    variables = getattr(function, "variables", _MISSING)
    if (type(variables) is not tuple or len(variables) != 1 or not _real_symbol(variables[0])
            or variables[0] != variable):
        raise InvalidInput(_ONLY_VARIABLE)
    if order < 0:
        raise InvalidInput("order must be at least 0")
    with computation() as budget:
        budget.tick()  # The call; every valid input charges more for its nodes.
        from ...kernels import series as kernel

        a = _center(budget, kernel, center)
        if order > MAX_ORDER:
            raise ResourceLimitError(f"Computation limit exceeded: series_coefficients takes orders up to {MAX_ORDER}")
        expression = getattr(function, "expression", _MISSING)
        seen: dict[int, tuple[object, int]] = {}  # id(node) -> (node, its height), for the nodes validated so far.
        nodes = _nodes(budget, kernel, expression, variable, seen)
        atoms = _domain_atoms(budget, getattr(function, "domain", _MISSING))
        atom_nodes = [node for _, atom in atoms or () for node in _nodes(budget, kernel, atom, variable, seen)]
        try:
            if atoms is None:
                raise _OutsideScope("the domain of the function is false: it is defined nowhere")
            values = _values(budget, kernel, nodes + atom_nodes, a)
            for kind, atom in atoms:
                budget.tick()
                sign = values[id(atom)].numerator
                if (sign == 0) if kind == "nonzero" else (sign <= 0):
                    raise _OutsideScope("the domain of the function does not contain a neighbourhood of the center")
        except _OutsideScope as reason:
            return _unevaluated(args, str(reason)), ()
        if order == 0:
            coefficients = (values[id(expression)],)
        else:
            coefficients = _expansion(budget, kernel, nodes, values, a, order + 1)
        return PowerSeries(variable, a, coefficients), ()


def _real_symbol(value: object) -> bool:
    """Whether value is a Symbol of the kind a RealFunction takes, with fields of the exact types that Symbol makes (a
    Symbol forged with object.__new__ could hold a str subclass that compares equal to everything)."""
    assumptions = getattr(value, "assumptions", _MISSING)
    return (type(value) is Symbol and type(getattr(value, "name", _MISSING)) is str
            and type(getattr(value, "uid", _MISSING)) is str and getattr(value, "domain", _MISSING) is Reals
            and type(assumptions) is tuple and not assumptions)


def _canonical(budget: Budget, kernel, value: object, what: str) -> None:
    """Refuse, with InvalidInput, a Rational that is not two exact ints in lowest terms with a positive denominator (a
    Rational made with object.__new__ may break that form); ResourceLimitError for an integer beyond the budget."""
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"{what} must be a Rational of two exact integers with a positive denominator")
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator != 1:
        budget.tick(kernel.gcd_work(numerator.bit_length(), denominator.bit_length()))
        if gcd(numerator, denominator) != 1:
            raise InvalidInput(f"{what} must be a Rational in lowest terms")


def _center(budget: Budget, kernel, center: int | Rational) -> Rational:
    """The center as a Rational, validated."""
    if type(center) is int:
        budget.check_int(center)
        return Rational(center)
    _canonical(budget, kernel, center, "center")
    return center


def _nodes(budget: Budget, kernel, root: object, variable: Symbol, seen: dict[int, tuple[object, int]]) -> list:
    """The nodes below root that seen does not hold yet, each once, children before their parents; seen gains them
    with their heights. InvalidInput for a malformed tree or a node below itself, ResourceLimitError for a path of
    more than MAX_DEPTH nodes. One unit per node."""
    from ...algebraic import RealAlgebraicRoot
    from ...functions import RealExpr

    order = []
    path: set[int] = set()  # The nodes being expanded: the ancestors of the node at the top of the stack.
    stack: list[tuple[object, tuple | None]] = [(root, None)]
    while stack:
        node, children = stack.pop()
        key = id(node)
        if children is not None:  # Every child is done.
            path.discard(key)
            height = 1 + max((seen[id(child)][1] for child in children), default=0)
            if height > MAX_DEPTH:
                raise ResourceLimitError(f"Computation limit exceeded: an expression deeper than {MAX_DEPTH} nodes")
            seen[key] = (node, height)
            order.append(node)
            continue
        if key in seen:
            continue
        if key in path:
            raise InvalidInput(_MALFORMED + ", which is finite: a node lies below itself")
        budget.tick()
        found = _operands(budget, kernel, node, variable, RealExpr, RealAlgebraicRoot)
        path.add(key)
        if len(path) > MAX_DEPTH:
            raise ResourceLimitError(f"Computation limit exceeded: an expression deeper than {MAX_DEPTH} nodes")
        stack.append((node, found))
        stack.extend((child, None) for child in found if id(child) not in seen)
    return order


def _operands(budget: Budget, kernel, node: object, variable: Symbol, real_expr: type, algebraic: type) -> tuple:
    """The operands of a node of a shape that RealExpr admits, with a canonical Rational scalar and no symbol but the
    variable; InvalidInput for anything else."""
    if type(node) is not real_expr:
        raise InvalidInput(_MALFORMED)
    op, operands = getattr(node, "op", _MISSING), getattr(node, "args", _MISSING)
    value = getattr(node, "value", _MISSING)
    if type(op) is not str or type(operands) is not tuple or any(type(item) is not real_expr for item in operands):
        raise InvalidInput(_MALFORMED)
    if op == "scalar":
        if operands or (type(value) is not Rational and type(value) is not algebraic):
            raise InvalidInput(_MALFORMED)
        if type(value) is Rational:
            _canonical(budget, kernel, value, "a scalar of function")
    elif op == "symbol":
        if operands or not _real_symbol(value) or value != variable:
            raise InvalidInput(_ONLY_VARIABLE)
    elif op in ("pi", "E"):
        if operands or value is not None:
            raise InvalidInput(_MALFORMED)
    elif op == "pow":
        if len(operands) != 1 or type(value) is not int:
            raise InvalidInput(_MALFORMED)
        budget.check_int(value)
    elif op in _BINARY:
        if len(operands) != 2 or value is not None:
            raise InvalidInput(_MALFORMED)
    elif op in _UNARY:
        if len(operands) != 1 or value is not None:
            raise InvalidInput(_MALFORMED)
    else:
        raise InvalidInput(_MALFORMED)
    return operands


def _domain_atoms(budget: Budget, domain: object) -> list[tuple[str, object]] | None:
    """The atoms (kind, expression) of a DomainPredicate, each distinct predicate once; None when it holds "false"
    (the function is defined nowhere). InvalidInput for a malformed predicate or one inside itself."""
    from ...functions import DomainPredicate, RealExpr

    atoms: list[tuple[str, object]] = []
    false = False
    done: set[int] = set()
    path: set[int] = set()  # The "and" predicates being expanded.
    stack: list[tuple[object, bool]] = [(domain, False)]
    while stack:
        predicate, leaving = stack.pop()
        key = id(predicate)
        if leaving:
            path.discard(key)
            done.add(key)
            continue
        if key in done:
            continue
        if key in path:
            raise InvalidInput(_MALFORMED + ": a domain predicate lies inside itself")
        budget.tick()
        if type(predicate) is not DomainPredicate:
            raise InvalidInput(_MALFORMED)
        kind, operands = getattr(predicate, "kind", _MISSING), getattr(predicate, "operands", _MISSING)
        if type(kind) is not str or type(operands) is not tuple:
            raise InvalidInput(_MALFORMED)
        if kind == "and":
            if len(operands) < 2 or any(type(item) is not DomainPredicate for item in operands):
                raise InvalidInput(_MALFORMED)
            path.add(key)
            stack.append((predicate, True))
            stack.extend((item, False) for item in operands)
            continue
        if kind in _ATOMS:
            if len(operands) != 1 or type(operands[0]) is not RealExpr:
                raise InvalidInput(_MALFORMED)
            atoms.append((kind, operands[0]))
        elif kind in ("true", "false") and not operands:
            false = false or kind == "false"
        else:
            raise InvalidInput(_MALFORMED)
        done.add(key)
    return None if false else atoms


def _rational_root(budget: Budget, kernel, value: Rational) -> Rational | None:
    """The rational q > 0 with q^2 = value, None when there is none."""
    numerator, denominator = value.numerator, value.denominator
    if numerator <= 0:
        return None
    budget.tick(kernel.product_work(numerator.bit_length(), numerator.bit_length())
                + kernel.product_work(denominator.bit_length(), denominator.bit_length()))
    p, q = isqrt(numerator), isqrt(denominator)
    return trusted_rational(p, q) if p * p == numerator and q * q == denominator else None


def _values(budget: Budget, kernel, nodes: list, a: Rational) -> dict[int, Rational]:
    """The value at the center of every node, children before parents, by the rules of the scope (the module
    docstring), with the kernel's rational arithmetic, which charges every integer operation by its length;
    _OutsideScope at the first node out of scope."""
    values: dict[int, Rational] = {}
    for node in nodes:
        budget.tick()
        op, operands = node.op, node.args
        u = values[id(operands[0])] if operands else None
        if op == "scalar":
            if type(node.value) is not Rational:
                raise _OutsideScope("a real algebraic scalar: only rational constants are expanded")
            value = node.value
        elif op == "symbol":
            value = a
        elif op in ("pi", "E"):
            raise _OutsideScope(f"the constant {op} has no rational power series")
        elif op == "add":
            value = kernel.plus(u, values[id(operands[1])])
        elif op == "mul":
            value = kernel.times(u, values[id(operands[1])])
        elif op == "div":
            divisor = values[id(operands[1])]
            if not divisor.numerator:
                raise _OutsideScope("a quotient whose denominator vanishes at the center")
            value = kernel.quotient(u, divisor)
        elif op == "pow":
            if node.value < 0 and not u.numerator:
                raise _OutsideScope("a negative power of an expression that vanishes at the center")
            value = kernel.rational_power(u, node.value)
        elif op in _VANISHING:
            if u.numerator:
                raise _OutsideScope(f"{op} is expanded only about an argument that vanishes at the center")
            value = _ONE if op in ("exp", "cos", "cosh") else _ZERO
        elif op == "log":
            if u != _ONE:
                raise _OutsideScope("log is expanded only about an argument equal to 1 at the center")
            value = _ZERO
        elif op == "sqrt":
            value = _rational_root(budget, kernel, u)
            if value is None:
                raise _OutsideScope("sqrt is expanded only about the square of a positive rational")
        elif op == "abs":
            if not u.numerator:
                raise _OutsideScope("abs is not analytic where its argument vanishes")
            value = u if u.numerator > 0 else kernel.negative(u)
        else:  # acos: irrational at every interior point, not analytic at 1 and -1.
            raise _OutsideScope(f"{op} has no rational power series")
        values[id(node)] = value
    return values


def _expansion(budget: Budget, kernel, nodes: list, values: dict[int, Rational], a: Rational,
               length: int) -> tuple[Rational, ...]:
    """The first length Taylor coefficients at a of the last node (the root), node by node with the kernel, for
    nodes whose values at the center _values found in scope. Each node charges length units first.

    The check recomputes every node's series by its own rules, and its work grows with the length of the
    coefficients where this kernel's does not (a square root or a tangent costs it products of every pair of
    coefficients). After each node, a lower bound of what the check will charge for the nodes so far (_check_work)
    is compared with max_work, the check's own budget: past it the value could not be verified, so produce() raises
    ResourceLimitError at once rather than return it."""
    series: dict[int, tuple] = {}
    pairs: dict[tuple[int, bool], tuple] = {}  # (id(argument), hyperbolic) -> (sin, cos) or (sinh, cosh).
    check, limit = 0, budget.limits.max_work
    for node in nodes:
        budget.tick(length)
        check += length  # truncated_series charges each distinct node length units.
        op, operands = node.op, node.args
        reciprocal = None
        if op == "scalar":
            result = kernel.constant(node.value, length)
        elif op == "symbol":
            result = kernel.identity(a, length)
        else:
            u = series[id(operands[0])]
            if op == "add":
                result = kernel.add(u, series[id(operands[1])])
            elif op == "mul":
                result = kernel.mul(u, series[id(operands[1])])
            elif op == "div":
                v = series[id(operands[1])]
                if kernel.monomial_of(v) == (1, _ZERO):  # A constant divisor.
                    result = kernel.scale(u, kernel.inverse(v[0]))
                else:
                    reciprocal = kernel.reciprocal(v)
                    result = kernel.mul(u, reciprocal)
            elif op == "pow":
                result = kernel.power(u, node.value)
            elif op == "exp":
                result = kernel.exp(u)
            elif op == "log":
                result = kernel.log(u)
            elif op == "sqrt":
                result = kernel.sqrt(u, values[id(node)])
            elif op == "abs":
                result = u if u[0].numerator > 0 else kernel.neg(u)
            elif op == "atan":
                result = kernel.atan(u)
            elif op == "asin":
                result = kernel.asin(u)
            elif op in ("tan", "tanh"):
                result = kernel.tangent(u, 1 if op == "tan" else -1)
            else:  # sin, cos, sinh, cosh
                hyperbolic = op in ("sinh", "cosh")
                key = (id(operands[0]), hyperbolic)
                if key not in pairs:
                    pairs[key] = kernel.sin_cos(u, hyperbolic)
                result = pairs[key][0 if op in ("sin", "sinh") else 1]
            check += _check_work(kernel, node, series, result, reciprocal, pairs, length)
            if check > limit:
                raise ResourceLimitError("Computation limit exceeded: work (the check of these coefficients would "
                                         f"charge more than {limit} units)")
        series[id(node)] = result
    return series[id(nodes[-1])]


def _check_work(kernel, node, series: dict[int, tuple], result: tuple, reciprocal: tuple | None,
                pairs: dict[tuple[int, bool], tuple], length: int) -> int:
    """A lower bound of the units that the checker charges for this node at the full length L, beyond the L units of
    the node itself: what its evaluator (mathforge.certify.series.taylor over mathforge.trusted.series) and the checker
    charge before they run, and the products of pairs of coefficients that the evaluator runs, by
    kernel.convolution_work, from the series of the operands and of the node, which are the same Taylor coefficients on
    both sides.

    Charged up front, whatever the coefficients: a truncated product or reciprocal of length L charges L(L + 1)/2
    units; the checker charges (L(L - 1)/2) // LOOP_ITERATIONS_PER_UNIT for each exp, sin, cos, sinh, cosh, tan and
    tanh node, whose ODE loops step over zero coefficients without charging. The evaluator's rules: a product is one
    truncated product of the operands U V; a quotient U / V the reciprocal R of V, from the pairs (V[j], R[k - j]),
    j >= 1, then the product U R, also for a constant V; a power U^k with k >= 0 squares and multiplies,
    bit_length(k) - 1 + popcount(k) truncated products, the first squaring U U when k >= 2; a power with k < 0 does
    the same with R = 1/U after one reciprocal (U[0] != 0 there); exp, sin, cos, sinh and cosh run F' = U' G, a product
    of j U[j] (j >= 1) with every coefficient of G = F, C or S; tan and tanh square F for every j with U[j] != 0, so at
    least once from the first such j on; sqrt sums F[j] F[k - j] for 1 <= j < k; log, atan and asin integrate U' / W
    with one reciprocal of length L and one product of length L - 1 (L > 1), W = U for log, 1 + U^2 for atan and
    sqrt(1 - U^2) for asin, the last two after the product U U. Everything else counts 0 here: the bound only has to
    stay below the check's work."""
    op, operands = node.op, node.args
    u = series[id(operands[0])]
    full = length * (length + 1) // 2  # A truncated product or reciprocal of length L.
    if op == "mul":
        return full + kernel.convolution_work(u, series[id(operands[1])], length)
    if op == "div":
        v = series[id(operands[1])]
        if reciprocal is None:  # A constant divisor: R = (1/V[0], 0, ...), with the words of V[0].
            return 2 * full + kernel.convolution_work(u, v[:1], length)
        return (2 * full + kernel.convolution_work(v[1:], reciprocal, length - 1)
                + kernel.convolution_work(u, reciprocal, length))
    if op == "pow":
        exponent = node.value
        if exponent == 0:
            return 0
        magnitude = -exponent if exponent < 0 else exponent
        work = (magnitude.bit_length() - 1 + magnitude.bit_count()) * full
        if exponent > 0:
            return work + (kernel.convolution_work(u, u, length) if exponent >= 2 else 0)
        work += full  # The reciprocal R of U.
        return work + (kernel.convolution_work(u[1:], result, length - 1) if exponent == -1 else 0)
    looping = length * (length - 1) // 2 // _LOOP_ITERATIONS_PER_UNIT
    if op == "exp":
        return looping + kernel.convolution_work(u[1:], result, length - 1, 1)
    if op in ("sin", "cos", "sinh", "cosh"):
        sine, cosine = pairs[(id(operands[0]), op in ("sinh", "cosh"))]
        return (looping + kernel.convolution_work(u[1:], sine, length - 1, 1)
                + kernel.convolution_work(u[1:], cosine, length - 1, 1))
    if op in ("tan", "tanh"):
        first = kernel.valuation(u[1:]) + 1
        return looping + (kernel.convolution_work(result, result, length - first) if first < length else 0)
    if op == "sqrt":
        return kernel.convolution_work(result[1:], result[1:], length - 2) if length > 2 else 0
    if op in ("log", "atan", "asin"):
        work = (full + (length - 1) * length // 2) if length > 1 else 0  # The integral of U' / W.
        if op != "log":
            work += full + kernel.convolution_work(u, u, length)  # The product U U.
        return work
    return 0


def _unevaluated(args: dict[str, object], detail: str) -> Unevaluated:
    """The record of a request out of scope, with the request that operations_v6.run builds for the same arguments
    (OperationRequest sorts them, and its version takes no part in equality)."""
    from ...operations_v6 import REQUEST_VERSION
    return Unevaluated("series", OperationRequest(_OPERATION, tuple(args.items()), REQUEST_VERSION),
                       Reason("no_applicable_rule", detail))


def _render(series: PowerSeries, renderer, child) -> str:
    r"""1 + x + 1/2*x^2 + O(x^3) at the center 0; at another center the powers of (x - a), as
    (x - 1) - 1/2*(x - 1)^2 + 1/3*(x - 1)^3 + O((x - 1)^4) or 1 - (x + 1/2) + O((x + 1/2)^2); zero terms are left out,
    so that a zero series is its remainder alone, O(x^3). LaTeX: 1 + x + \frac{1}{2}\, x^{2} +
    \mathcal{O}\left(x^{3}\right)."""
    latex = renderer.latex
    name, center = child(series.variable), series.center
    if center.numerator == 0:
        base = bracketed = name
    else:  # x - a for a > 0 and x + |a| for a < 0, bracketed wherever a power or a product follows.
        base = renderer.join((name, " - " if center.numerator > 0 else " + ", child(abs(center))))
        bracketed = renderer.parens(base)

    def power(exponent: int) -> str:
        if exponent == 1:
            return bracketed
        digits = renderer.integer(exponent)
        return renderer.join((bracketed, "^{" + digits + "}" if latex else "^" + digits))

    terms = []
    for exponent, coefficient in enumerate(series.coefficients):
        numerator, denominator = coefficient.numerator, coefficient.denominator
        if numerator == 0:
            continue
        if exponent == 0:
            terms.append(child(coefficient))
        elif denominator == 1 and numerator in (1, -1):
            terms.append(power(exponent) if numerator == 1 else renderer.join(("-", power(exponent))))
        else:
            terms.append(renderer.join((child(coefficient), r"\, " if latex else "*", power(exponent))))
    remainder = base if len(series.coefficients) == 1 else power(len(series.coefficients))
    terms.append(renderer.join((r"\mathcal{O}" if latex else "O", renderer.parens(remainder))))
    return renderer.join(term if index == 0 else (" - " + term[1:] if term.startswith("-") else " + " + term)
                         for index, term in enumerate(terms))


RENDER = {PowerSeries: _render}
