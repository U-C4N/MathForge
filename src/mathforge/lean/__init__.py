"""The lean value path: value functions that call a producer and an independent checker directly.

A value function (``mathforge.integer_gcd``, ``mathforge.ntheory.integer_gcd``, ``mathforge.det``, ...) called
without ``limits`` first calls its lean function, ``functions.<operation>``. The lean function returns the verified
value, or ``FALLBACK`` for any call it does not handle; the value function then runs the structured path
(``mathforge.operations``) and unwraps its Result, so its values and errors stay exactly the structured path's.
With explicit ``limits``, a value function takes the structured path at once.

The contract of a lean function:

- It takes the positional parameters of the public value function, ``limits`` excepted, and it never raises.
- It returns a value only after an independent check accepted it (a certify checker, or the criterion of the v0.7
  checker of the operation), and only a value equal to what ``structured(...).unwrap()`` returns for that input.
- Anything it does not handle gives ``FALLBACK``: argument types and sizes, deferrals, a failed or exhausted check,
  and a call made while a computation budget is active (the structured path then charges that budget, as in v0.7).
- It leaves its arguments as it found them: a one-shot iterator falls back before anything iterates it.

**Lean modules.** ``mathforge/lean/<area>_<topic>.py`` defines ``LEAN: dict[str, Callable[..., object]]``, lean
functions by operation name. ``functions.<operation>`` imports only the modules whose name starts with the
operation's area (the registry spec's, else the legacy catalogue's) followed by ``_``, and is then cached. Two modules
defining one name raise RuntimeError, naming both. An operation that no module defines resolves to the generic
runner below, with its positional arguments mapped to its argument names. The ``<area>_legacy`` modules hold the
dispatchers of the legacy operations whose public functions route on the input type before ``_run``; a dispatcher
takes the same routes, through other operations' lean functions or ``run_legacy``, and never calls ``run_value`` for
its own operation.

**The generic runner,** ``run_value(operation, arguments)``, returns the verified value or ``FALLBACK`` and never
raises an Exception. For a registry operation it repeats ``operations_v5.run`` or ``operations_v6.run`` without the
records: it freezes and inspects the arguments, checks their names, normalizes them as the request would, runs the
family's ``produce`` in a fresh budget, falls back on a deferral or a decline, inspects the value and the evidence,
and runs the checker (the family's ``check_operation`` for v5, ``registry.checker(...)`` for v6) in another fresh
budget, as the verifier does. For a legacy operation it repeats ``operations._run`` with the operation's
``operations.LEGACY`` entry: ``_run``'s argument normalization, the producer, the record's source checks and the
entry's check, all in one budget, as there; it falls back for the inputs that the public function routes elsewhere.
As in v0.7, a legacy operation never builds the registry tables: unless ``registry.is_v6_operation`` says that a v6
family took it over, its lean function resolves from the legacy catalogue alone.

Building no records, the runner still charges each budget at least what the structured path charges for them: the
request's and the step's inspections, and the verifier's inspection of the whole Result. Its own inspections run with
``max_depth`` lowered by the depth at which the Result nests an argument, so an argument too deep for the record
falls back here too. A value returned here is therefore one that the structured path returns as well; at the edge of
a limit, the lean path falls back sooner, never later.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import functools
import importlib
from types import MappingProxyType
from typing import Any

from .. import registry
from ..errors import UnsupportedOperation
from ..limits import ComputationLimits, computation, current_budget

__all__ = ["FALLBACK", "functions", "run_legacy", "run_value"]


class _Fallback:
    """What a lean function returns for a call that it leaves to the structured path."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "FALLBACK"


FALLBACK = _Fallback()

# The structured path's verifier inspects the whole Result, where an argument lies six levels below the level at
# which the request inspects it (Result.steps[0].inputs[0].arguments[i][1]) and a value or the evidence four levels
# below the Result. The runner's budgets inspect with max_depth lowered by six, so that an argument or a value whose
# depth the record would exceed is refused here as well.
_RECORD_DEPTH = 6
_PRODUCER_LIMITS = ComputationLimits(max_depth=ComputationLimits().max_depth - _RECORD_DEPTH)


class _Functions:
    """``functions.<operation>``: the operation's lean function, resolved on first access, then cached here."""

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("__"):  # Protocol probes (copy, pickle, mock) are not operations.
            raise AttributeError(name)
        return vars(self).setdefault(name, _resolve(name))

    def __dir__(self) -> list[str]:
        return sorted(vars(self))


functions = _Functions()


def _area_modules(area: str) -> tuple[str, ...]:
    """The lean modules of an area: this package's modules named ``<area>_<topic>``, sorted."""
    import pkgutil
    prefix = area + "_"
    return tuple(sorted(f"{__name__}.{module.name}" for module in pkgutil.iter_modules(__path__)
                        if module.name.startswith(prefix)))


_AREA_ENTRIES: dict[str, Mapping[str, Callable[..., Any]]] = {}


def _area_entries(area: str) -> Mapping[str, Callable[..., Any]]:
    """The LEAN entries of an area's lean modules, merged; RuntimeError when two modules define one name."""
    entries = _AREA_ENTRIES.get(area)
    if entries is None:
        merged: dict[str, Callable[..., Any]] = {}
        owners: dict[str, str] = {}
        for name in _area_modules(area):  # The import system locks each module; no lock is held across imports.
            table = getattr(importlib.import_module(name), "LEAN", None)
            if not isinstance(table, Mapping):
                raise RuntimeError(f"Lean module {name} must define LEAN, a mapping of operation names to functions")
            for operation, function in table.items():
                if operation in owners:
                    raise RuntimeError(f"Lean function {operation!r} is defined by both {owners[operation]} and {name}")
                if not callable(function):
                    raise RuntimeError(f"Lean module {name} maps {operation!r} to a value that is not callable")
                owners[operation] = name
                merged[operation] = function
        entries = _AREA_ENTRIES.setdefault(area, MappingProxyType(merged))
    return entries


@functools.cache
def _legacy_areas() -> Mapping[str, str]:
    """The area of each legacy catalogue operation: static data."""
    from ..registry_v07 import LEGACY_CATALOG
    return MappingProxyType({entry.name: entry.area for entry in LEGACY_CATALOG})


def _spec(operation: str) -> Any:
    """The registry spec of an operation, or None. A legacy operation that no v6 family took over has none, which is
    decided without building the registry: as in v0.7, the legacy operations never build it."""
    if operation in _legacy_areas() and not registry.is_v6_operation(operation):
        return None
    return registry.specs().get(operation)


def _area(operation: str) -> str:
    spec = _spec(operation)
    if spec is not None:
        return str(spec.area)
    area = _legacy_areas().get(operation)
    if area is None:
        raise AttributeError(f"mathforge.lean.functions has no operation {operation!r}")
    return area


def _resolve(operation: str) -> Callable[..., Any]:
    """The lean function of an operation: its lean module entry, else the generic runner."""
    function = _area_entries(_area(operation)).get(operation)
    return function if function is not None else _generic(operation)


def _argument_names(operation: str) -> tuple[str, ...]:
    """The operation's argument names in the order of its public function's parameters; () when it has no runner."""
    spec = _spec(operation)
    if spec is not None:
        return tuple(argument.name for argument in spec.arguments)
    from .. import operations
    if operation not in operations.LEGACY:
        return ()
    import inspect
    parameters = inspect.signature(getattr(operations, operation)).parameters.values()
    if any(parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY) for parameter in parameters):
        return ()  # Not the operation's own function (a test double, say): leave every call to the structured path.
    return tuple(parameter.name for parameter in parameters if parameter.name != "limits")


def _generic(operation: str) -> Callable[..., Any]:
    """The generic lean function of an operation: run_value with its positional arguments mapped to their names."""
    names = _argument_names(operation)
    count = len(names)

    def lean(*values: object) -> object:
        if count == 0 or len(values) != count:
            return FALLBACK
        return run_value(operation, dict(zip(names, values)))

    lean.__name__ = lean.__qualname__ = operation
    lean.__doc__ = f"The generic lean function of {operation}: run_value with the arguments by name."
    return lean


def _clear_caches() -> None:
    """Forget resolved lean functions, merged lean modules and runner plans (for tests)."""
    vars(functions).clear()
    _AREA_ENTRIES.clear()
    _PLANS.clear()


# ---------------------------------------------------------------------------------------------------------------
# The generic runner

# operation -> (the registry spec it was planned for, None for a legacy operation; the plan). A registry rebuilt with
# other families gives new spec objects, and so new plans.
_PLANS: dict[str, tuple[Any, Callable[[Mapping[str, object]], object]]] = {}


def run_value(operation: str, arguments: Mapping[str, object]) -> object:
    """The verified value of an operation for arguments by name, or FALLBACK; never raises an Exception."""
    try:
        spec = _spec(operation)
        planned = _PLANS.get(operation)
        if planned is None or planned[0] is not spec:
            planned = _PLANS[operation] = (spec, _plan(operation, spec))
        return planned[1](arguments)
    except Exception:
        return FALLBACK


def run_legacy(operation: str, arguments: Mapping[str, object], *, exact: bool = False,
               unsupported: object = FALLBACK) -> object:
    """The lean path of a legacy operation's ``_run`` with its ``operations.LEGACY`` entry (with ``exact``: its
    ``_run_exact`` entry, ``operations.LEGACY_EXACT``), for the dispatchers of the lean modules. It takes no route:
    the caller has routed the input. Returns ``unsupported`` when the producer raises UnsupportedOperation, which the
    structured path reports as an unsupported result; FALLBACK for everything else that gives no value."""
    try:
        from .. import operations
        _prepare()
        entry = (operations.LEGACY_EXACT if exact else operations.LEGACY)[operation]
        return _legacy(operation, arguments, entry, unsupported)
    except Exception:
        return FALLBACK


def _no_plan(arguments: Mapping[str, object]) -> object:
    return FALLBACK


def _plan(operation: str, spec: Any) -> Callable[[Mapping[str, object]], object]:
    _prepare()
    if spec is not None:
        return _registry_plan(operation, spec)
    from .. import operations
    if operation in operations.LEGACY:
        return _legacy_plan(operation)
    return _no_plan


def _registry_plan(operation: str, spec: Any) -> Callable[[Mapping[str, object]], object]:
    """The lean path of operations_v5.run or operations_v6.run for one registry operation."""
    from ..model import Symbol
    from ..operations_v5 import _freeze
    family = registry.family(operation, spec.version)  # A superseded v0.7 operation's spec is its v6 family's.
    checker = registry.checker(operation, spec.version)  # v5: the family itself; v6: its CHECKER module.
    expected = registry.operation_arguments(spec.version)[operation]
    if spec.version == 5:
        from ..functions import UnevaluatedIntegral, UnevaluatedLimit
        deferrals: tuple[type, ...] = (UnevaluatedIntegral, UnevaluatedLimit)
        partial = getattr(family, "is_partial", None)
    else:
        from ..deferred import Unevaluated
        deferrals, partial = (Unevaluated,), None
    # The v5 verifier wants a recorded variable to be a Symbol (run records only a Symbol); v6 records what it gets.
    symbol_variable = spec.version == 5 and "variable" in expected
    # The arguments that run copies into the Result: the expression (else A), and a Symbol variable.
    context_keys = frozenset(expected & {"expression", "A", "variable"})

    def run(arguments: Mapping[str, object]) -> object:
        if current_budget() is not None:
            return FALLBACK
        args = {key: _freeze(value) for key, value in arguments.items()}
        if args.keys() != expected:
            return FALLBACK
        variable = args.get("variable")
        if symbol_variable and variable is not None and type(variable) is not Symbol:
            return FALLBACK
        expression = args["expression"] if "expression" in args else args.get("A")
        if expression is not None and not _record_expression(expression):
            return FALLBACK
        with computation(_PRODUCER_LIMITS, fresh=True) as budget:
            size, sizes = _inspect_arguments(budget, args, context_keys)
            request = {key: _immutable(args[key]) for key in sorted(args)}
            budget.nodes(size + len(args))  # The request inspects its (name, value) pairs.
            value, evidence = family.produce(operation, args)
            if type(value) in deferrals:
                return FALLBACK
            before = budget.node_count
            budget.inspect(value)
            value_size = budget.node_count - before
            before = budget.evidence_count
            budget.inspect(evidence, evidence=True)
            evidence_size = budget.evidence_count - before
            if partial is not None:
                partial(operation, args, value)
            request_size = _request_size(size, len(args))
            budget.evidence(_step_inputs_size(request_size, value_size, evidence_size))
            value, evidence = _immutable(value), _immutable(evidence)
        if value is None or type(evidence) is not tuple:
            return FALLBACK
        context_size = _registry_context_size(sizes, variable)
        with computation(None, fresh=True) as budget:
            budget.evidence(_result_size(request_size, value_size, evidence_size, context_size))  # The verifier's.
            if not _symbols_consistent((tuple(request.items()), value, evidence)):
                return FALLBACK
            if partial is not None:
                partial(operation, request, value)
            if checker.check_operation(operation, request, value, evidence) is not True:
                return FALLBACK
        return value

    return run


def _legacy_plan(operation: str) -> Callable[[Mapping[str, object]], object]:
    """The generic runner of a legacy operation: the lean path of its LEGACY entry, for the inputs that the public
    function sends to that entry (the dispatchers of the lean modules take the other routes)."""
    from .. import operations
    plain = _plain_routes().get(operation)

    def run(arguments: Mapping[str, object]) -> object:
        if plain is not None and not plain(arguments):
            return FALLBACK
        return _legacy(operation, arguments, operations.LEGACY[operation], FALLBACK)

    return run


def _plain_routes() -> dict[str, Callable[[Mapping[str, Any]], bool]]:
    """For each legacy operation whose public function routes on the input before ``_run``: whether the arguments
    take its LEGACY entry. The conditions are those of the public functions in mathforge.operations."""
    from ..model import Expression, Rational
    from ..operations import _function_input
    from ..rational_function import RationalFunction

    def scalar(a: Mapping[str, Any]) -> bool:
        return not _function_input(a["expression"])

    def polynomial(a: Mapping[str, Any]) -> bool:
        return scalar(a) and type(a["expression"]) is not RationalFunction

    def algebraic(value: Any) -> bool:
        return isinstance(value, Expression) and type(value) is not Rational

    return {
        "partial_diff": scalar, "gradient": scalar, "hessian": scalar, "differentiate": scalar, "evaluate": scalar,
        "jacobian": lambda a: not (type(a["expressions"]) in (tuple, list)
                                   and any(_function_input(item) for item in a["expressions"])),
        "antiderivative": polynomial,
        "integrate": lambda a: polynomial(a) and not (algebraic(a["lower"]) or algebraic(a["upper"])),
        "limit": lambda a: scalar(a) and not algebraic(a["point"]),
    }


def _legacy(operation: str, arguments: Mapping[str, object], entry: Any, unsupported: object) -> object:
    """The lean path of ``_run(operation, arguments, entry.produce)``: one budget for the producer and the check."""
    if current_budget() is not None:
        return FALLBACK
    if any(hasattr(type(value), "__next__") for value in arguments.values()):
        return FALLBACK  # _run would iterate a one-shot iterator; the structured path still needs it.
    from .. import operations
    with computation(_PRODUCER_LIMITS, fresh=True) as budget:
        try:
            args = operations._arguments(operation, arguments)
            context = operations._context(operation, args)
            size, sizes = _inspect_arguments(budget, args, _LEGACY_CONTEXT_KEYS)
            request = {key: _immutable(args[key]) for key in sorted(args)}
            budget.nodes(2 * size + len(args))  # _run inspects the raw arguments (at most these) and the request.
            produced = entry.produce(args)
        except UnsupportedOperation:
            return unsupported
        if type(produced) is not tuple:
            return FALLBACK  # A whole Result: solve's equations of degree two at most.
        value, evidence = produced
        # The source context that the Result restates (operations._context): an expression or A, a variable with its
        # assumptions, and a problem. A linear system's problem is built after the producer, and inspected here.
        problem_size = sizes.get("problem", 1)
        if operation == "solve_system":
            context["problem"] = problem = operations._system(args)
            before = budget.evidence_count
            budget.inspect(problem, evidence=True)
            problem_size = 0
        before = budget.evidence_count
        budget.inspect(value, evidence=True)
        value_size = budget.evidence_count - before
        before = budget.evidence_count
        budget.inspect(evidence, evidence=True)
        evidence_size = budget.evidence_count - before
        value, evidence = _immutable(value), _immutable(evidence)
        if value is None or not _record_context(operation, context, value):
            return FALLBACK
        context_size = _legacy_context_size(operation, sizes, context, problem_size)
        # What _run's records inspect beyond the value, the evidence and a linear system inspected above: the step
        # that _run builds, the verifier's whole Result and its step, and the step again when _run attaches its audit.
        request_size = _request_size(size, len(args))
        step = _step_inputs_size(request_size, value_size, evidence_size)
        budget.evidence(step - value_size - evidence_size
                        + _result_size(request_size, value_size, evidence_size, context_size)
                        + _step_size(request_size, value_size, evidence_size) + step)
        if not _legacy_source(operation, request, context):
            return FALLBACK
        if not _symbols_consistent((tuple(request.items()), value, evidence, context.get("problem"))):
            return FALLBACK
        if entry.check(request, value, evidence) is not True:
            return FALLBACK
    return value


_LEGACY_CONTEXT_KEYS = frozenset({"expression", "A", "variable", "for_", "problem"})


# Node counts of the records that the structured path builds and inspects (Budget.inspect counts one node per value,
# tuple, enum member and dataclass, plus the contents of tuples and of dataclass fields). tests/v08/test_lean_paths.py
# checks them against real records.

def _request_size(arguments_size: int, count: int) -> int:
    """An OperationRequest: itself, its operation name and its tuple of (name, value) pairs, given the node count of
    the arguments' dict and the number of arguments."""
    return arguments_size + count + 2


def _step_inputs_size(request_size: int, value_size: int, evidence_size: int) -> int:
    """What building a Step inspects: (inputs, outputs, preconditions) with the request as the one input, the value
    and the evidence as the outputs, and no preconditions."""
    return request_size + value_size + evidence_size + 4


def _step_size(request_size: int, value_size: int, evidence_size: int) -> int:
    """A whole Step with no audit, as the legacy verifiers inspect it."""
    return request_size + value_size + evidence_size + 8


def _context_size(expression: int, variable: int, problem: int, assumptions: int) -> int:
    """A Result's source context: its expression, for_ and problem fields (1 each when None) and the items of its
    assumptions tuple."""
    return expression + variable + problem + assumptions


def _result_size(request_size: int, value_size: int, evidence_size: int, context_size: int) -> int:
    """A whole Result with one step and no verification, as a verifier inspects it: 23 fixed nodes, the request and
    the value twice (as fields and in the step), the evidence once and the source context."""
    return 23 + 2 * request_size + 2 * value_size + evidence_size + context_size


def _assumptions_size(symbol_size: int) -> int:
    """The items of a Symbol's assumptions, given the Symbol's node count: the Symbol, its name, domain, assumptions
    tuple and uid count 5."""
    return symbol_size - 5


def _registry_context_size(sizes: Mapping[str, int], variable: object) -> int:
    """The source context of operations_v5.run's and operations_v6.run's Results: the expression (else A) and a
    Symbol variable, no problem and no assumptions; sizes holds the node counts of those arguments."""
    expression = sizes["expression"] if "expression" in sizes else sizes.get("A", 1)
    return _context_size(expression, sizes["variable"] if type(variable) is _types["Symbol"] else 1, 1, 0)


def _legacy_context_size(operation: str, sizes: Mapping[str, int], context: Mapping[str, Any],
                         problem_size: int) -> int:
    """The source context of operations._run's Results (operations._context): A or the expression, the variable (for
    solve: for_) with its assumptions, and the problem, whose node count is given."""
    expression = sizes["A"] if "A" in sizes else sizes.get("expression", 1)
    key = "for_" if operation == "solve" else "variable" if "for_" in context else None
    if key is None:
        return _context_size(expression, 1, problem_size, 0)
    return _context_size(expression, sizes[key], problem_size, _assumptions_size(sizes[key]))


def _inspect_arguments(budget: Any, args: Mapping[str, object],
                       keys: frozenset[str]) -> tuple[int, dict[str, int]]:
    """Inspect the arguments as the structured path inspects their dict, and size the values of the given keys.

    Returns the dict's node count and, for each given key present, its value's node count. Each such value is
    inspected under a one-entry dict, at the depth that the whole dict gives it, and costs one node more."""
    sizes: dict[str, int] = {}
    rest: dict[str, object] = {}
    for key, value in args.items():
        if key in keys:
            before = budget.node_count
            budget.inspect({key: value})
            sizes[key] = budget.node_count - before - 2
        else:
            rest[key] = value
    before = budget.node_count
    budget.inspect(rest)
    return budget.node_count - before + sum(size + 1 for size in sizes.values()), sizes


def _legacy_source(operation: str, request: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    """The record checks of the legacy verifiers besides symbol consistency: a recorded variable is an
    unconstrained real Symbol, and a linear system restates its source."""
    variable = request.get("for_" if operation == "solve" else "variable", _ABSENT)
    if variable is not _ABSENT and not (type(variable) is _types["Symbol"] and variable.domain is _types["Reals"]
                                        and not variable.assumptions):
        return False
    if operation == "solve_system":
        from ..verification_v2 import _system
        return bool(_system(request) == context["problem"])
    return True


_ABSENT = object()

# Classes the runner tests by exact type, imported on first use: importing this package stays light.
_types: dict[str, Any] = {}
_SAME: frozenset[type] = frozenset()
_ATOMS: frozenset[type] = frozenset()
_RECORD_EXPRESSIONS: frozenset[type] = frozenset()


def _prepare() -> None:
    global _SAME, _ATOMS, _RECORD_EXPRESSIONS
    if _types:
        return
    from .. import contracts, matrices
    from ..algebraic import RealAlgebraicRoot
    from ..functions import RealExpr, RealFunction
    from ..intervals import IntervalSet
    from ..model import Add, Mul, Pow, Rational, Reals, Sqrt, Symbol
    from ..polynomial import Polynomial
    from ..rational_function import RationalFunction
    from ..rational_integration import RationalPrimitive
    # The classes whose values contracts._immutable_value returns unchanged (tests/v08/test_lean_paths.py checks it).
    _SAME = frozenset({int, str, bool, type(None), Rational, Symbol, Add, Mul, Pow, Sqrt, RealAlgebraicRoot,
                       Polynomial, matrices.Matrix, RationalFunction})
    # Values in which verification_v2._symbol_consistency finds no Symbol.
    _ATOMS = frozenset({int, str, bool, type(None), Rational})
    # The expression classes that a Result accepts without contracts.as_scalar.
    _RECORD_EXPRESSIONS = frozenset({Polynomial, matrices.Matrix, RealExpr, RealFunction, RationalPrimitive})
    _types.update(
        immutable=contracts._immutable_value, as_scalar=contracts.as_scalar, Symbol=Symbol, Reals=Reals,
        problems=(contracts.Eq, contracts.Inequality, matrices.LinearSystem), EmptySet=contracts.EmptySet,
        sets=(contracts.FiniteSet, IntervalSet), whole=(contracts.UniversalSet, matrices.AffineSolutionSet))


def _immutable(value: Any) -> Any:
    """contracts._immutable_value, which a record applies to its request, its value and its evidence."""
    kind = type(value)
    if kind in _SAME:
        return value
    if kind is tuple or kind is list:  # No record class is a plain tuple or list: the items, normalized, as a tuple.
        return tuple([_immutable(item) for item in value])
    return _types["immutable"](value)


def _record_expression(expression: Any) -> bool:
    """Whether a Result accepts the expression of its source context."""
    if type(expression) in _RECORD_EXPRESSIONS:
        return True
    try:
        _types["as_scalar"](expression)
    except Exception:
        return False
    return True


def _record_context(operation: str, context: Mapping[str, Any], value: Any) -> bool:
    """Whether _run's Result accepts this source context and value, as Result.__post_init__ checks them."""
    problem = context.get("problem")
    if problem is not None and type(problem) not in _types["problems"]:
        return False
    expression = context.get("expression")
    if expression is not None and not _record_expression(expression):
        return False
    variable = context.get("for_")
    if variable is not None and type(variable) is not _types["Symbol"]:
        return False
    if operation in ("solve", "solve_system"):  # The outcome of a solution set constrains the set.
        kind = type(value)
        return (kind is _types["EmptySet"] or kind in _types["whole"]
                or (kind in _types["sets"] and len(value) > 0))
    return True


_FIELDS: dict[type, tuple[str, ...]] = {}


def _symbols_consistent(values: tuple[Any, ...]) -> bool:
    """verification_v2._symbol_consistency, which every verifier applies to the whole record: one Symbol uid has one
    name, domain and assumptions. It walks tuples and dataclass fields, as that function does."""
    pending, seen = list(values), set()
    symbols: dict[str, tuple[Any, ...]] = {}
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind in _ATOMS:
            continue
        if kind is _types["Symbol"]:
            metadata = (item.name, item.domain, item.assumptions)
            if symbols.setdefault(item.uid, metadata) != metadata:
                return False
        elif kind is tuple:
            if id(item) not in seen:
                seen.add(id(item))
                pending.extend(item)
        else:
            names = _FIELDS.get(kind)
            if names is None:
                names = _FIELDS.setdefault(kind, _field_names(kind))
            if names and id(item) not in seen:
                seen.add(id(item))
                pending.extend(getattr(item, name) for name in names)
    return True


def _field_names(kind: type) -> tuple[str, ...]:
    from dataclasses import fields, is_dataclass
    return tuple(field.name for field in fields(kind)) if is_dataclass(kind) else ()
