"""Independent checker of the family mathforge.areas.series.power_series: series_coefficients.

An N-version check. The producer computes the coefficients with its own kernel (mathforge/kernels/series.py:
composition with Maclaurin series and Newton iteration over truncated power series); this checker never runs it. It
recomputes them node by node with the evaluator of the v0.8 Taylor checker, truncated_series of
mathforge.certify.series.taylor (truncated series arithmetic for add, mul, div and integer powers; the first-order ODE
recurrences that define exp, sin and cos, sinh and cosh, tan, tanh, log, sqrt, atan and asin), and compares.

series_coefficients(function, variable, center, order) = PowerSeries(variable, center, coefficients), evidence ().
The claim, for a RealFunction f of the one variable x, a rational a and an int 0 <= n <= MAX_ORDER: near a,
f(x) = c_0 + c_1 (x - a) + ... + c_n (x - a)^n + O((x - a)^(n+1)) with coefficients = (c_0, ..., c_n), the value in
its canonical form. True requires
  (0) the arguments: exactly function, variable, center and order; variable an unconstrained real Symbol (the domain
      Reals, no assumptions, a str name and a str uid, so that symbols compare honestly); function a RealFunction (by
      module and name) whose variables are (variable,); center an int, or a Rational of two exact ints in lowest terms
      with a positive denominator; order an exact int >= 0. A record of the class (module, qualname) == RECORD whose
      variable is the argument, whose center is a Rational in lowest terms equal to the argument (an int is no Rational
      here), and whose coefficients are a tuple of exactly n + 1 Rationals (of two exact ints, a positive
      denominator); the evidence the empty tuple;
  (1) the expression E of f is a tree of RealExpr nodes (by module and name; shared subtrees count once, a cycle
      fails) of the shapes that RealExpr admits, at most MAX_DEPTH nodes deep, whose scalars are Rationals in lowest
      terms, whose symbols are the variable (with the fields of (0)) and whose operations are add, mul, div, pow (an
      int exponent), exp, log, sqrt, abs, sin, cos, tan, sinh, cosh, tanh, atan and asin; the atoms of f's domain are
      such trees as well;
  (2) truncated_series accepts every node of E at a, and returns exactly the claimed n + 1 coefficients, compared as
      (numerator, denominator) pairs: the evaluator's Rationals are in lowest terms, so only such a coefficient passes;
  (3) every atom of f's domain holds at a with a strict sign: positive(D) and nonnegative(D) need D(a) > 0, nonzero(D)
      needs D(a) != 0, where D(a) is what truncated_series returns at length 1 (it must accept every node of D);
      "true" passes, "and" needs each of its operands, "false" fails, and so does a predicate inside itself.
Theorem. truncated_series accepts a node only where its rule applies (the table of mathforge.certify.series.taylor): a
Rational scalar, the variable, add, mul; div when the divisor's value at a is not 0; pow k >= 0, and k < 0 when the
base's value at a is not 0; exp, sin, cos, sinh, cosh, tan, tanh, atan and asin when the argument's value at a is 0;
log when it is 1; sqrt when it is q^2 for a rational q > 0; abs when it is not 0. By induction over the nodes, an
accepted node is a function analytic on a neighbourhood of a: a composition of analytic functions whose inner value at
a lies where the outer one is analytic (0 for the functions analytic at 0, 1 for log, q^2 > 0 for sqrt, a value other
than 0 for a reciprocal and for abs). Its tuple is exactly its first n + 1 Taylor coefficients at a: products and
reciprocals of truncated series truncate the exact ones, a power is repeated products, and each ODE F' = U' G(F), with
F(a) given, has one power series solution, which its recurrence builds coefficient by coefficient. So by (2), E defines
a function g analytic near a with g^(k)(a)/k! = c_k for every k <= n. By (3), every atom's expression D is analytic,
hence continuous, near a, so its strict sign at a persists on a neighbourhood: the domain of f contains a neighbourhood
of a, where f = g. Hence the claim. Taylor coefficients are unique, so no other value with the canonical form of (0)
passes, and every true claim in scope does: the evaluator computes the true coefficients.
The domain test (3) is this checker's addition to the card (docs/v09/cards/m3-analysis.md), in the safe direction: a
RealFunction keeps a domain that its caller may have narrowed (real_function(x, domain=...), a RationalFunction's
excluded points), and such a function has no power series at a center outside the interior of its domain. The
intrinsic domain of an expression accepted by (2) always passes (3); a nonnegative(D) atom with D(a) = 0 fails although
it may hold near a (x^2 >= 0), and the producer leaves such a function unevaluated as well.

Sizes and work. Arguments beyond the budget raise ResourceLimitError, as run() and the verifier refuse them before any
checker runs, and are beyond the budget rather than wrong: an order above MAX_ORDER (the card's resource limit), an
expression deeper than MAX_DEPTH (truncated_series recurses once per level), an integer of a scalar, an exponent or the
center longer than max_integer_bits. A claimed value that cannot be the true one answers False before any series is
computed: a wrong variable, center or number of coefficients, a coefficient that is no Rational of two exact ints with
a positive denominator or whose integers exceed max_integer_bits (the evaluator's own Rationals never do). Then the
checker validates the trees (one unit per node; the gcd of a scalar's lowest-terms test is charged before it runs),
evaluates E and every domain atom at length 1, which rejects a node out of scope and a wrong constant term cheaply, then
the first PREFIX coefficients, which a wrong value usually fails as well at a cost that does not grow with the order,
and only then the whole series. truncated_series charges each node and each series operation: its products and
reciprocals L(L + 1)/2 units up front, every Rational result one unit, and, before it runs, every addition, product
and quotient of two coefficients by the length of its operands (trusted.series.operation_work of their 64-bit words:
one unit per 64 pairs of words and one per 4 words, since the gcds and products of long Rationals grow quadratically
with their length). The loops of its ODE rules (_flow, _sine_pair) step over the zero coefficients of their argument
without charging: before a series of length L is computed, the checker pays (L(L - 1)/2)/16 units for each distinct
exp, sin, cos, sinh, cosh, tan and tanh node, one unit per 16 iterations (measured with CPython: about 14 ns an
iteration). So the budget bounds the wall time whatever the length of the coefficients, at about 0.1 to 0.7
microseconds a unit (measured with CPython 3.15: exp(x) and cos(x) at order 4096 verify in 0.35 to 0.4 seconds for 1.5
million units, tan(x) at order 400 in 0.6 seconds for 1.8 million, sqrt(1 + x) at order 446 in 1.4 seconds for 2
million; sqrt(1 + x) at order 1000 needs 32 million units, so the default budget refuses it, as tan(x) at order 500,
within about 1.4 seconds; a sum of 1000 nodes 1/(x - c) with c of 4000 bits at order 15, coefficients of 64,000 bits,
is refused in 0.4 seconds). Most functions without such long coefficients exceed the budget before order 2000, where the
products and reciprocals of truncated_series alone cost L(L + 1)/2 units each. Anything else malformed (a record
forged without __init__ or with a missing field, a bool, a float, a str or None where a number belongs, a list for a
tuple, any evidence but ()) answers False; a node out of scope answers False as well, since the producer answers
Unevaluated there and makes no claim.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) with RECORD instead, and reads RealFunction, RealExpr and DomainPredicate as data, by their
module and name. tools/check_layering.py follows every import, and tests/v08/test_layering.py rejects importlib,
__import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational, Reals, Symbol

from .taylor import SeriesDeclined, truncated_series

OPERATION = "series_coefficients"
RECORD = ("mathforge.areas.series.power_series", "PowerSeries")  # Module and name of the value's class.
ARGUMENTS = frozenset(("function", "variable", "center", "order"))
MAX_ORDER = 4096  # The card: an order above it is a resource limit.
MAX_DEPTH = 200  # Nodes along the deepest path: truncated_series recurses per level, two Python frames at most each.
LOOP_ITERATIONS_PER_UNIT = 16  # An uncharged iteration of an ODE loop takes about 14 ns; a unit about 0.2 us.
PREFIX = 32  # The coefficients compared before the whole series is computed.

_REAL_EXPR = ("mathforge.functions", "RealExpr")
_REAL_FUNCTION = ("mathforge.functions", "RealFunction")
_DOMAIN = ("mathforge.functions", "DomainPredicate")
_BINARY = frozenset(("add", "mul", "div"))
_UNARY = frozenset(("exp", "log", "sqrt", "abs", "sin", "cos", "tan", "sinh", "cosh", "tanh", "atan", "asin"))
_LOOPING = frozenset(("exp", "sin", "cos", "sinh", "cosh", "tan", "tanh"))  # Their rules run _flow or _sine_pair.
_ATOMS = frozenset(("positive", "nonnegative", "nonzero"))
_MISSING = object()  # A field that a record made with object.__new__ lacks.
_DECLINED = (SeriesDeclined, AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError)


def _named(value: object) -> tuple[object, object]:
    """(module, qualname) of the value's class, which identifies a record without importing its module."""
    cls = type(value)
    return getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)


def _field(value: object, name: str) -> object:
    return getattr(value, name, _MISSING)


def _real_symbol(value: object) -> bool:
    """Whether value is a Symbol of the kind a RealFunction takes, with fields of the exact types that Symbol makes (a
    Symbol forged with object.__new__ could hold a str subclass whose __eq__ lies): a str name and uid, the domain
    Reals, no assumptions. Symbols compare by these fields."""
    assumptions = _field(value, "assumptions")
    return (type(value) is Symbol and type(_field(value, "name")) is str and type(_field(value, "uid")) is str
            and _field(value, "domain") is Reals and type(assumptions) is tuple and not assumptions)


def _fraction(value: object) -> tuple[int, int] | None:
    """(numerator, denominator) of a Rational whose fields are exact ints with a positive denominator; None for any
    other class (an int, a bool, a float included) and for a Rational made with object.__new__ that breaks that."""
    if type(value) is not Rational:
        return None
    numerator, denominator = _field(value, "numerator"), _field(value, "denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:  # A bool is no exact int.
        return None
    return numerator, denominator


def _lowest(budget: Budget, numerator: int, denominator: int) -> bool:
    """Whether numerator/denominator, denominator >= 1, is in lowest terms. math.gcd is charged before it runs: its
    first remainder is linear in the longer integer, Lehmer's steps after it quadratic in the shorter one (measured with
    CPython: 2.5 ms for two integers of 65,536 bits, 11 us for one of 65,536 bits and 1; a unit is about 0.2 us)."""
    if denominator == 1:
        return True
    longer, shorter = sorted((numerator.bit_length() >> 6, denominator.bit_length() >> 6), reverse=True)
    budget.tick(1 + (longer + shorter) // 4 + (longer + 1) * (shorter + 1) // 32)  # In 64-bit words.
    return gcd(numerator, denominator) == 1


def _argument_rational(budget: Budget, value: object) -> bool:
    """Whether an exact rational of the arguments (an int, or a Rational in lowest terms) is well formed;
    ResourceLimitError for an integer longer than max_integer_bits, which run() refuses before any producer."""
    if type(value) is int:
        budget.check_int(value)
        return True
    fraction = _fraction(value)
    if fraction is None:
        return False
    budget.check_int(fraction[0])
    budget.check_int(fraction[1])
    return _lowest(budget, *fraction)


def _claimed_rational(budget: Budget, value: object) -> tuple[int, int] | None:
    """(numerator, denominator) of a Rational of the claimed value, None for anything else and for an integer longer
    than max_integer_bits: no Rational of the evaluator is that long, so such a value is no true one. Lowest terms are
    not tested here: the pair is compared with the evaluator's Rational or the center's pair, both in lowest terms, so
    a pair that is not fails that comparison, and no gcd of a long coefficient is paid for."""
    fraction = _fraction(value)
    if fraction is None:
        return None
    limit = budget.limits.max_integer_bits
    if fraction[0].bit_length() > limit or fraction[1].bit_length() > limit:
        return None
    return fraction


def _children(budget: Budget, node: object, variable: Symbol) -> tuple | None:
    """The operands of a RealExpr node of a shape in scope (condition 1), None for anything else: another class, a
    shape that RealExpr refuses, acos, pi, E, another symbol, a scalar that is no Rational in lowest terms."""
    if _named(node) != _REAL_EXPR:
        return None
    op, args, value = _field(node, "op"), _field(node, "args"), _field(node, "value")
    if type(op) is not str or type(args) is not tuple:
        return None
    if op == "scalar":  # A RealAlgebraicRoot scalar is out of scope, even a rational one: truncated_series declines it.
        return args if not args and type(value) is Rational and _argument_rational(budget, value) else None
    if op == "symbol":
        return args if not args and _real_symbol(value) and value == variable else None
    if op == "pow":
        if len(args) != 1 or type(value) is not int:
            return None
        budget.check_int(value)  # truncated_series raises to it bit by bit.
        return args
    if (op in _BINARY and len(args) == 2) or (op in _UNARY and len(args) == 1):
        return args if value is None else None
    return None


def _tree(budget: Budget, root: object, variable: Symbol, heights: dict[int, tuple[object, int]]) -> int | None:
    """Validate the tree below root (condition 1), each distinct node once, one unit each: the number of its nodes not
    yet in heights whose rule runs an ODE loop (_LOOPING), or None when a node is out of scope or the tree has a
    cycle. heights maps id(node) to (node, the number of nodes on its deepest path) for the nodes validated so far;
    ResourceLimitError for a path longer than MAX_DEPTH."""
    looping = 0
    path: set[int] = set()  # The nodes being expanded: the ancestors of the node at the top of the stack.
    stack: list[tuple[object, tuple | None]] = [(root, None)]
    while stack:
        node, children = stack.pop()
        key = id(node)
        if children is not None:  # Every child is done: the node's height follows from theirs.
            path.discard(key)
            height = 1 + max((heights[id(child)][1] for child in children), default=0)
            if height > MAX_DEPTH:
                raise ResourceLimitError(f"Computation limit exceeded: an expression deeper than {MAX_DEPTH} nodes")
            heights[key] = (node, height)
            continue
        if key in heights:
            continue
        if key in path:
            return None  # A node below itself: no finite tree.
        budget.tick()
        found = _children(budget, node, variable)
        if found is None:
            return None
        looping += node.op in _LOOPING
        path.add(key)
        if len(path) > MAX_DEPTH:
            raise ResourceLimitError(f"Computation limit exceeded: an expression deeper than {MAX_DEPTH} nodes")
        stack.append((node, found))
        stack.extend((child, None) for child in found if id(child) not in heights)
    return looping


def _domain_atoms(budget: Budget, domain: object) -> list[tuple[str, object]] | None:
    """The atoms (kind, expression) of a DomainPredicate, each distinct predicate once and one unit each; None for a
    malformed predicate, for a predicate inside itself and for "false", where f is defined nowhere. The predicates
    stay alive in the function's domain while the check runs, so their ids identify them."""
    atoms: list[tuple[str, object]] = []
    done: set[int] = set()
    path: set[int] = set()  # The "and" predicates being expanded: the ancestors of the one at the top of the stack.
    stack: list[tuple[object, bool]] = [(domain, False)]
    while stack:
        predicate, leaving = stack.pop()
        key = id(predicate)
        if leaving:  # Every operand of this "and" is done.
            path.discard(key)
            done.add(key)
            continue
        if key in done:
            continue
        if key in path:
            return None
        budget.tick()
        if _named(predicate) != _DOMAIN:
            return None
        kind, operands = _field(predicate, "kind"), _field(predicate, "operands")
        if type(kind) is not str or type(operands) is not tuple:
            return None
        if kind == "and" and len(operands) >= 2:
            path.add(key)
            stack.append((predicate, True))
            stack.extend((operand, False) for operand in operands)
            continue
        if kind in _ATOMS and len(operands) == 1:
            atoms.append((kind, operands[0]))
        elif kind != "true" or operands:
            return None
        done.add(key)
    return atoms


def _check(budget: Budget, args: dict, value: object, evidence: tuple) -> bool:
    """Conditions (0) to (3) of the module docstring, the cheap ones first."""
    if args.keys() != ARGUMENTS or evidence:
        return False
    function, variable, center, order = args["function"], args["variable"], args["center"], args["order"]
    if type(order) is not int or order < 0 or not _real_symbol(variable) or _named(function) != _REAL_FUNCTION:
        return False
    variables = _field(function, "variables")
    if (type(variables) is not tuple or len(variables) != 1 or not _real_symbol(variables[0])
            or variables[0] != variable or not _argument_rational(budget, center)):
        return False
    a = Rational(center) if type(center) is int else center
    if _named(value) != RECORD:
        return False
    claimed_variable, claimed_center = _field(value, "variable"), _field(value, "center")
    coefficients = _field(value, "coefficients")
    if not _real_symbol(claimed_variable) or claimed_variable != variable:
        return False
    if _claimed_rational(budget, claimed_center) != (a.numerator, a.denominator):
        return False
    if type(coefficients) is not tuple or len(coefficients) != order + 1:
        return False
    if order > MAX_ORDER:
        raise ResourceLimitError(f"Computation limit exceeded: series_coefficients takes orders up to {MAX_ORDER}")
    for coefficient in coefficients:
        budget.tick()
        if _claimed_rational(budget, coefficient) is None:
            return False
    expression, domain = _field(function, "expression"), _field(function, "domain")
    heights: dict[int, tuple[object, int]] = {}
    looping = _tree(budget, expression, variable, heights)
    atoms = _domain_atoms(budget, domain)
    if looping is None or atoms is None:
        return False
    for _, atom in atoms:
        if _tree(budget, atom, variable, heights) is None:
            return False
    # Length 1 first: every node out of scope and a wrong constant term show at the cost of a unit or two per node.
    constants: dict = {}
    if truncated_series(expression, variable, a, 1, constants)[0] != coefficients[0]:
        return False
    for kind, atom in atoms:
        budget.tick()
        sign = truncated_series(atom, variable, a, 1, constants)[0].numerator
        if (sign == 0) if kind == "nonzero" else (sign <= 0):
            return False
    length = order + 1
    # Then a prefix of PREFIX coefficients, which a wrong value usually fails as well, at a cost independent of the
    # order; a truncated series is the prefix of every longer one. Last, the whole series.
    for size in sorted({min(length, PREFIX), length} - {1}):
        budget.tick(looping * (size * (size - 1) // 2 // LOOP_ITERATIONS_PER_UNIT))
        expected = truncated_series(expression, variable, a, size, {})
        budget.tick(size)  # The comparison, term by term.
        if expected != coefficients[:size]:
            return False
    return True


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of series_coefficients on args, by the theorem of the
    module. False for another operation, for arguments other than exactly function, variable, center and order, for
    any malformed value or evidence and for a function out of scope; ResourceLimitError for an argument beyond the
    budget (order, depth, integer lengths) and past the budget."""
    if type(operation) is not str or operation != OPERATION or type(args) is not dict or type(evidence) is not tuple:
        return False
    with computation() as budget:  # One budget for every helper: truncated_series needs an active one.
        try:
            return _check(budget, args, value, evidence)
        except _DECLINED:  # A node that the evaluator declines, or a forged field that breaks an operation.
            return False
