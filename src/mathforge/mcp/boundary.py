"""SDK-independent, bounded adapters: the thirty curated v0.7 tools, the area tools and mathforge.catalog.

The area tools and the catalog derive from the operation registry. An area tool runs one operation of its
areas, named in the request, with arguments typed by the argument kinds of mcp.kinds.

Only this module and mcp.kinds translate text and ordinary JSON into core objects. They do not implement
mathematical algorithms or turn persisted operation names into executable code: an operation name selects
a registry spec, never a Python attribute. Responses preserve the core codec and its integer strings.
"""

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, TypedDict

from .. import operations
from ..capabilities import CATALOG_TOOL, area_tool, area_tools, capabilities
from ..contracts import Eq, ExecutionStatus, Result, VerificationReport
from ..errors import InvalidInput, MathForgeError
from ..limits import ComputationLimits, DecodeLimits, computation, current_budget
from ..matrices import Matrix
from ..model import Expression, Rational, Symbol, free_symbols
from ..serialization import from_data, to_data
from ..verification import verify


MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4_194_304
MAX_EQUATIONS = 64
MAX_VARIABLES = 64
MAX_SCALAR_CHARACTERS = 1024
MAX_TEXT_CHARACTERS = 16384  # Function text
MAX_QUERY_CHARACTERS = 128
CATALOG_LIMIT, MAX_CATALOG_LIMIT = 50, 200


class BoundaryError(TypedDict):
    stage: str
    code: str
    message: str
    field: str | None
    start: int | None
    end: int | None


@dataclass(frozen=True)
class ToolResponse:
    payload: dict[str, Any]
    is_error: bool = False


class InputError(InvalidInput):
    def __init__(self, message: str, *, code: str = "invalid_input", field: str | None = None):
        super().__init__(message, code=code)
        self.field = field


def _error(exc: Exception, stage: str, field: str | None = None) -> BoundaryError:
    return {"stage": stage, "code": getattr(exc, "code", "invalid_input"),
            "message": str(exc), "field": getattr(exc, "field", None) or field,
            "start": getattr(exc, "start", None), "end": getattr(exc, "end", None)}


def _response(value=None, *, symbols=None, error=None, is_error=False) -> ToolResponse:
    symbols = symbols or {}
    payload = {"api_version": 1, "result": to_data(value) if value is not None else None,
               "text": "", "latex": None,
               "symbols": [{"name": name, "uid": symbol.uid}
                           for name, symbol in symbols.items()], "error": error}
    if value is not None:
        try:
            from ..rendering import render_text, to_latex
            payload["text"] = render_text(value, symbols=symbols)
            payload["latex"] = to_latex(value, symbols=symbols)
        except (MathForgeError, TypeError, ValueError, RecursionError) as exc:
            payload["text"] = "The canonical mathematical result is retained; presentation failed."
            payload["error"] = _error(exc, "presentation")
            is_error = True
    elif error is not None:
        payload["text"] = error["message"]
    if len(json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        error = _error(InputError("The full result exceeds the response byte limit; no truncated certificate was returned.",
                                 code="response_too_large"), "encoding")
        payload.update(result=None, text=error["message"], latex=None, error=error)
        is_error = True
    return ToolResponse(payload, is_error)


def _limits(value: Any) -> ComputationLimits:
    defaults = asdict(ComputationLimits())
    if value is None:
        return ComputationLimits()
    if type(value) is not dict or set(value) - set(defaults):
        raise InputError("limits must contain only named ComputationLimits fields", field="limits")
    for key, number in value.items():
        if type(number) is not int or not 0 < number <= defaults[key]:
            raise InputError(f"{key} must be a positive integer no greater than {defaults[key]}",
                             code="limit_out_of_range", field="limits." + key)
    return ComputationLimits(**(defaults | value))


def _text(value: Any, field: str) -> str:
    if type(value) is not str:
        raise InputError("A mathematical text string is required", field=field)
    return value


def _rational(value: Any, field: str) -> Rational:
    value = _text(value, field)
    if len(value) > MAX_SCALAR_CHARACTERS:
        raise InputError("Rational input is too long", code="resource_limit_exceeded", field=field)
    if re.fullmatch(r"[+-]?[0-9]+(?:/[+-]?[0-9]+)?", value) is None:
        raise InputError("Use an integer or rational-fraction string, for example '-3/4'", field=field)
    parts = value.split("/")
    try:
        return Rational(int(parts[0]), int(parts[1]) if len(parts) == 2 else 1)
    except (ValueError, ZeroDivisionError, MathForgeError) as exc:
        code = "resource_limit_exceeded" if getattr(exc, "code", None) == "resource_limit_exceeded" else "invalid_input"
        raise InputError(str(exc), code=code, field=field) from exc


def _scalar(value: Any, field: str, limits: ComputationLimits) -> Expression:
    """Read a closed math expression or a canonical scalar envelope, never code."""
    from ..parsing import ParseLimits, parse_expression

    if type(value) is str:
        if len(value) > MAX_SCALAR_CHARACTERS:
            raise InputError("Scalar text exceeds the character limit", code="resource_limit_exceeded", field=field)
        scalar = parse_expression(value, symbols={}, limits=limits,
                                  parse_limits=ParseLimits(max_chars=MAX_SCALAR_CHARACTERS))
    elif type(value) is dict:
        # Restrict top-level types before reconstructing potentially expensive
        # compound records. Nested node validation remains the core decoder's job.
        if (value.get("kind") != "object" or type(value.get("data")) is not dict
                or value["data"].get("type") not in {"rational", "sqrt", "real_algebraic_root", "add", "mul", "pow"}):
            raise InputError("A canonical scalar object envelope is required; Result and workspace envelopes are not operands", field=field)
        scalar = from_data(value, decode_limits=DecodeLimits(max_bytes=MAX_REQUEST_BYTES,
                           max_depth=min(128, limits.max_depth), max_nodes=limits.max_nodes),
                           computation_limits=limits)
        if to_data(scalar) != value:
            raise InputError("The scalar envelope must use its canonical minimum format", field=field)
    else:
        raise InputError("Use closed mathematical text or a canonical scalar object envelope", field=field)
    if not isinstance(scalar, Expression) or free_symbols(scalar):
        raise InputError("Scalar operands must be closed real algebraic expressions", field=field)
    return scalar


def _limit_point(value: Any, limits: ComputationLimits) -> Expression | str:
    if type(value) is str and value in ("-inf", "+inf"):
        return value
    return _scalar(value, "point", limits)


def _symbols(names: list[str], ids: Any) -> dict[str, Symbol]:
    if len(names) > MAX_VARIABLES:
        raise InputError("Too many variables", code="resource_limit_exceeded", field="variables")
    if any(type(name) is not str or len(name) > 128 or name == "sqrt" or
           re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None for name in names):
        raise InputError("Variable names must be text identifiers", field="variables")
    if len(set(names)) != len(names):
        raise InputError("Variable names must be unique", field="variables")
    ids = {} if ids is None else ids
    if type(ids) is not dict or set(ids) - set(names):
        raise InputError("symbol_ids must map only declared variable names to UUID strings", field="symbol_ids")
    symbols = {}
    for name in names:
        try:
            symbols[name] = Symbol(name, uid=ids[name]) if name in ids else Symbol(name)
        except MathForgeError as exc:
            raise InputError(str(exc), field="symbol_ids." + name) from exc
    if len({symbol.uid for symbol in symbols.values()}) != len(symbols):
        raise InputError("Distinct names must not share a symbol UUID", field="symbol_ids")
    return symbols


def operand_tags():
    """Data-only operand tags accepted by the fixed v0.7 computation tool."""
    from ..dispatch_v5 import record_types
    return {"integer", "rational", "sqrt", "add", "mul", "pow", "real_algebraic_root",
            "symbol", "tuple", "text", "none", "matrix", "polynomial", "rational_function"} | {
                "v5_"+tag for tag in record_types()}


def _envelope(envelope, field, limits, *, tags, versions=(1, 2, 3, 4, 5)):
    """Check the shape of a canonical object envelope, then walk its data within the request budget.

    Only the given operand tags may appear: results, workspaces and execution metadata are not operands.
    """
    if (type(envelope) is not dict or set(envelope) != {"format", "version", "kind", "data"}
            or envelope.get("format") != "mathforge" or envelope.get("kind") != "object"
            or type(envelope.get("version")) is not int or envelope["version"] not in versions
            or type(envelope.get("data")) is not dict):
        raise InputError("Each argument must be a canonical mathematical object envelope", field=field)
    pending = [(envelope["data"], 0)]
    while pending:
        node, depth = pending.pop()
        current_budget().nodes()
        if depth > limits.max_depth:
            raise InputError("Operand nesting exceeds the depth limit", code="resource_limit_exceeded", field=field)
        if type(node) is dict:
            if "type" in node and (type(node["type"]) is not str or node["type"] not in tags):
                raise InputError("Results, workspaces and execution metadata are not operands", field=field)
            current_budget().nodes(len(node))
            pending.extend((item, depth+1) for item in node.values())
        elif type(node) is list:
            current_budget().nodes(len(node))
            pending.extend((item, depth+1) for item in node)


def _decode_envelopes(envelopes, fields, limits):
    """Decode checked envelopes; each must be in its canonical minimum wire format."""
    # A single decoder shares symbol metadata and a budget across every operand.
    document = {"format": "mathforge", "version": max((item["version"] for item in envelopes), default=1),
                "kind": "object", "data": {"type": "tuple", "items": [item["data"] for item in envelopes]}}
    values = from_data(document, decode_limits=DecodeLimits(max_bytes=MAX_REQUEST_BYTES,
                       max_depth=min(128, limits.max_depth), max_nodes=limits.max_nodes),
                       computation_limits=limits)
    for field, envelope, value in zip(fields, envelopes, values):
        current_budget().nodes()
        if to_data(value) != envelope:
            raise InputError("Operands must use their canonical minimum wire format", field=field)
    return values


def _compute_arguments(operation, arguments, limits):
    from ..dispatch_v5 import operation_arguments
    contracts = operation_arguments()
    if type(operation) is not str or operation not in contracts:
        raise InputError("operation must be a published v0.7 operation", field="operation")
    if type(arguments) is not dict or set(arguments) != contracts[operation]:
        raise InputError("arguments must contain exactly the operation's published arguments", field="arguments")
    allowed_tags = operand_tags()
    fields = ["arguments." + name for name in arguments]
    for field, envelope in zip(fields, arguments.values()):
        _envelope(envelope, field, limits, tags=allowed_tags)
    return dict(zip(arguments, _decode_envelopes(list(arguments.values()), fields, limits)))


_FIELDS = {
    "capabilities": (set(), set()),
    "solve": ({"problem"}, {"problem", "variable", "symbol_ids", "limits"}),
    "solve_system": ({"equations", "variables"}, {"equations", "variables", "symbol_ids", "limits"}),
    "real_roots": ({"expression"}, {"expression", "variable", "symbol_ids", "limits"}),
    "differentiate": ({"expression"}, {"expression", "variable", "order", "symbol_ids", "limits"}),
    "expand_multivariate": ({"expression", "variables"}, {"expression", "variables", "symbol_ids", "limits"}),
    "partial_diff": ({"expression", "variables", "variable"},
                     {"expression", "variables", "variable", "order", "symbol_ids", "limits"}),
    "gradient": ({"expression", "variables"}, {"expression", "variables", "symbol_ids", "limits"}),
    "jacobian": ({"expressions", "variables"}, {"expressions", "variables", "symbol_ids", "limits"}),
    "hessian": ({"expression", "variables"}, {"expression", "variables", "symbol_ids", "limits"}),
    "integrate": ({"expression", "lower", "upper"}, {"expression", "variable", "lower", "upper", "symbol_ids", "limits"}),
    "limit": ({"expression", "point"}, {"expression", "variable", "point", "side", "symbol_ids", "limits"}),
    "analyze_function": ({"expression"}, {"expression", "variable", "symbol_ids", "limits"}),
    "det": ({"rows"}, {"rows", "limits"}),
    "rref": ({"rows"}, {"rows", "ncols", "limits"}),
    "verify": ({"result"}, {"result", "limits"}),
    "factor": ({"expression"}, {"expression", "variable", "symbol_ids", "limits"}),
    "resultant": ({"f", "g"}, {"f", "g", "variable", "symbol_ids", "limits"}),
    "discriminant": ({"expression"}, {"expression", "variable", "symbol_ids", "limits"}),
    "algebraic": ({"expression"}, {"expression", "limits"}),
    "algebraic_div": ({"left", "right"}, {"left", "right", "limits"}),
    "algebraic_pow": ({"base", "n"}, {"base", "n", "limits"}),
    "minimal_polynomial": ({"expression"}, {"expression", "variable", "symbol_ids", "limits"}),
    "evaluate_algebraic": ({"expression", "variables", "bindings"},
                           {"expression", "variables", "bindings", "symbol_ids", "limits"}),
    "apart": ({"expression"}, {"expression", "variable", "symbol_ids", "limits"}),
    "extrema_values": ({"expression"}, {"expression", "variable", "symbol_ids", "limits"}),
    "charpoly": ({"rows"}, {"rows", "variable", "symbol_ids", "limits"}),
    "real_eigenvalues": ({"rows"}, {"rows", "limits"}),
    "parse_function": ({"text", "variables"}, {"text", "variables", "symbol_ids", "limits"}),
    "compute": ({"operation", "arguments"}, {"operation", "arguments", "limits"}),
}
_AREA_FIELDS = ({"operation", "arguments"}, {"operation", "arguments", "symbol_ids", "limits"})
# The failures that a call reports as an error response.
_FAILURES = (MathForgeError, TypeError, ValueError, ZeroDivisionError, RecursionError, OverflowError)


def call_tool(name: str, arguments: dict[str, Any]) -> ToolResponse:
    """Give every request its own budget, including inherited worker contexts."""
    if name == CATALOG_TOOL:  # It takes no limits; every catalog error keeps the catalog payload shape.
        return _catalog(arguments)
    try:
        limits = _limits(arguments.get("limits")) if type(arguments) is dict else None
        with computation(limits, fresh=True):
            return _call_tool(name, arguments)
    except _FAILURES as exc:
        return _response(error=_error(exc, "input"), is_error=True)


def _call_tool(name: str, arguments: dict[str, Any]) -> ToolResponse:
    """Execute one curated boundary call, without importing the optional SDK."""
    stage, field, symbols = "input", None, {}
    try:
        # The v0.7 table first: a v0.7 tool call never needs the registry.
        curated = type(name) is str and name.startswith("mathforge.") and name[10:] in _FIELDS
        area = not curated and type(name) is str and name in area_tools()
        if not curated and not area:
            raise InputError("Unknown MathForge tool", code="unknown_tool")
        operation = name[10:]
        required, allowed = _AREA_FIELDS if area else _FIELDS[operation]
        if type(arguments) is not dict or not required <= set(arguments) or set(arguments) - allowed:
            raise InputError(f"Expected required fields {sorted(required)} and optional fields {sorted(allowed-required)}")
        if len(json.dumps(arguments, ensure_ascii=True, allow_nan=False).encode("utf-8")) > MAX_REQUEST_BYTES:
            raise InputError("Request exceeds the byte limit", code="resource_limit_exceeded")
        if operation == "capabilities":
            return ToolResponse(dict(capabilities()))
        limits = _limits(arguments.get("limits"))
        if area:
            from .. import operations_v5, operations_v6, registry
            from .kinds import Context, decode_arguments
            specs = {spec.name: spec for spec in registry.tool_operations(name)}
            if type(arguments["operation"]) is not str or arguments["operation"] not in specs:
                raise InputError("operation must name an operation of this tool", field="operation")
            spec = specs[arguments["operation"]]
            context = Context(limits, arguments.get("symbol_ids"))
            stage, symbols = "decoding", context.symbols  # The table fills while the arguments decode.
            decoded = decode_arguments(spec, arguments["arguments"], context)
            stage = "computation"
            run = operations_v6.run if spec.version == 6 else operations_v5.run
            return _operation_response(run(spec.name, decoded, limits=limits), symbols)
        if operation == "parse_function":
            from ..functions import parse_function
            names = arguments["variables"]
            if type(names) is not list:
                raise InputError("variables must be an ordered array of declared names", field="variables")
            symbols = _symbols(names, arguments.get("symbol_ids"))
            text = _text(arguments["text"], "text")
            if len(text) > MAX_TEXT_CHARACTERS:
                raise InputError("Function text exceeds the character limit", code="resource_limit_exceeded", field="text")
            stage, field = "parsing", "text"
            value = parse_function(text, symbols=symbols, variables=tuple(symbols.values()), limits=limits)
            return _response(value, symbols=symbols)
        if operation == "compute":
            from ..operations_v5 import run
            stage, field = "decoding", "arguments"
            decoded = _compute_arguments(arguments["operation"], arguments["arguments"], limits)
            stage, field = "computation", None
            result = run(arguments["operation"], decoded, limits=limits)
            return _operation_response(result, symbols)
        if operation == "verify":
            stage, field = "decoding", "result"
            value = from_data(arguments["result"], decode_limits=DecodeLimits(max_bytes=MAX_REQUEST_BYTES),
                              computation_limits=limits)
            if type(value) is not Result:
                raise InputError("verify requires a canonical Result envelope", field="result")
            stage = "verification"
            return _response(verify(value, limits=limits))
        if operation in ("algebraic", "algebraic_div", "algebraic_pow", "minimal_polynomial"):
            stage = "parsing"
            if operation == "algebraic_div":
                field = "left"
                left = _scalar(arguments[field], field, limits)
                field = "right"
                right = _scalar(arguments[field], field, limits)
                stage, field = "computation", None
                result = operations.algebraic_div(left, right, limits=limits)
            elif operation == "algebraic_pow":
                field = "base"
                base = _scalar(arguments[field], field, limits)
                field = "n"
                exponent = arguments[field]
                if type(exponent) is not int or not -1024 <= exponent <= 1024:
                    raise InputError("n must be an integer from -1024 through 1024", field=field)
                stage, field = "computation", None
                result = operations.algebraic_pow(base, exponent, limits=limits)
            else:
                field = "expression"
                scalar = _scalar(arguments[field], field, limits)
                if operation == "minimal_polynomial":
                    symbols = _symbols([arguments.get("variable", "x")], arguments.get("symbol_ids"))
                    variable = next(iter(symbols.values()))
                    stage, field = "computation", None
                    result = operations.minimal_polynomial(scalar, variable, limits=limits)
                else:
                    stage, field = "computation", None
                    result = operations.algebraic(scalar, limits=limits)
        elif operation in ("det", "rref", "charpoly", "real_eigenvalues"):
            rows = arguments["rows"]
            if type(rows) is not list or any(type(row) is not list for row in rows):
                raise InputError("rows must be an array of arrays of rational strings", field="rows")
            if len(rows) > 4096 or sum(map(len, rows)) > limits.max_matrix_entries:
                raise InputError("Matrix input exceeds the entry limit", code="resource_limit_exceeded", field="rows")
            ncols = arguments.get("ncols")
            if ncols is not None and (type(ncols) is not int or not 0 <= ncols <= limits.max_matrix_entries):
                raise InputError("ncols must be a bounded nonnegative integer", field="ncols")
            matrix = Matrix(tuple(tuple(_rational(item, f"rows[{i}][{j}]") for j, item in enumerate(row))
                                  for i, row in enumerate(rows)), ncols=ncols)
            stage = "computation"
            if operation == "charpoly":
                symbols = _symbols([arguments.get("variable", "x")], arguments.get("symbol_ids"))
                result = operations.charpoly(matrix, next(iter(symbols.values())), limits=limits)
            elif operation == "real_eigenvalues":
                result = operations.real_eigenvalues(matrix, limits=limits)
            elif operation == "det":
                result = operations.det(matrix, limits=limits)
            else:
                result = operations.rref(matrix, limits=limits)
        else:
            from ..parsing import parse_expression, parse_problem
            multivariate = operation in ("expand_multivariate", "partial_diff", "gradient", "jacobian", "hessian")
            if operation == "solve_system":
                names, equations = arguments["variables"], arguments["equations"]
                if type(names) is not list or type(equations) is not list or len(equations) > MAX_EQUATIONS:
                    raise InputError("equations and variables must be bounded arrays", field="equations")
            elif multivariate or operation == "evaluate_algebraic":
                names = arguments["variables"]
                if type(names) is not list or not names:
                    raise InputError("variables must be a nonempty array of distinct names", field="variables")
                if operation == "jacobian":
                    expressions = arguments["expressions"]
                    if type(expressions) is not list or not 1 <= len(expressions) <= MAX_EQUATIONS:
                        raise InputError("expressions must be a nonempty bounded array", field="expressions")
            else:
                names = [arguments.get("variable", "x")]
            symbols = _symbols(names, arguments.get("symbol_ids"))
            if operation == "partial_diff":
                variable_name = arguments["variable"]
                if type(variable_name) is not str or variable_name not in symbols:
                    raise InputError("variable must name one of the declared variables", field="variable")
                order = arguments.get("order", 1)
                if type(order) is not int or not 0 <= order <= 1024:
                    raise InputError("order must be an integer from 0 through 1024", field="order")
            stage = "parsing"
            if operation == "solve_system":
                problems = []
                for i, equation in enumerate(equations):
                    field = f"equations[{i}]"
                    problem = parse_problem(_text(equation, field), symbols=symbols, limits=limits)
                    if type(problem) is not Eq:
                        raise InputError("Linear systems require equations, not inequalities", field=field)
                    problems.append(problem)
                stage, field = "computation", None
                result = operations.solve_system(tuple(problems), for_=tuple(symbols.values()), limits=limits)
            elif operation == "evaluate_algebraic":
                field = "bindings"
                bindings = arguments[field]
                if type(bindings) is not dict or set(bindings) != set(names):
                    raise InputError("bindings must supply exactly the declared variable names", field=field)
                replacements = {}
                for name in names:
                    field = "bindings." + name
                    replacements[symbols[name]] = _scalar(bindings[name], field, limits)
                field = "expression"
                expression = parse_expression(_text(arguments[field], field), symbols=symbols, limits=limits)
                stage, field = "computation", None
                result = operations.evaluate_algebraic(expression, replacements, limits=limits)
            elif multivariate:
                variables = tuple(symbols.values())
                if operation == "jacobian":
                    parsed = []
                    for i, expression in enumerate(expressions):
                        field = f"expressions[{i}]"
                        parsed.append(parse_expression(_text(expression, field), symbols=symbols, limits=limits))
                    stage, field = "computation", None
                    result = operations.jacobian(tuple(parsed), variables, limits=limits)
                else:
                    field = "expression"
                    expression = parse_expression(_text(arguments[field], field), symbols=symbols, limits=limits)
                    stage, field = "computation", None
                    if operation == "expand_multivariate":
                        result = operations.expand_multivariate(expression, variables, limits=limits)
                    elif operation == "partial_diff":
                        result = operations.partial_diff(expression, symbols[variable_name], n=order, limits=limits)
                    elif operation == "gradient":
                        result = operations.gradient(expression, variables, limits=limits)
                    else:
                        result = operations.hessian(expression, variables, limits=limits)
            else:
                variable = symbols[names[0]]
                if operation == "resultant":
                    from ..polynomial import Polynomial
                    field = "f"
                    f = parse_expression(_text(arguments[field], field), symbols=symbols, limits=limits)
                    f = Polynomial.from_expression(f, variable, limits=limits)
                    field = "g"
                    g = parse_expression(_text(arguments[field], field), symbols=symbols, limits=limits)
                    g = Polynomial.from_expression(g, variable, limits=limits)
                    stage, field = "computation", None
                    result = operations.resultant(f, g, limits=limits)
                    return _operation_response(result, symbols)
                field = "problem" if operation == "solve" else "expression"
                parse = parse_problem if operation == "solve" else parse_expression
                expression = parse(_text(arguments[field], field), symbols=symbols, limits=limits)
                stage, field = "computation", None
                if operation == "solve":
                    result = operations.solve(expression, for_=variable, limits=limits)
                elif operation == "real_roots":
                    result = operations.real_roots(expression, variable, limits=limits)
                elif operation == "differentiate":
                    order = arguments.get("order", 1)
                    if type(order) is not int or not 0 <= order <= 1024:
                        raise InputError("order must be an integer from 0 through 1024", field="order")
                    result = operations.differentiate(expression, variable, n=order, limits=limits)
                elif operation == "limit":
                    field = "point"
                    point = _limit_point(arguments[field], limits)
                    side = arguments.get("side", "both")
                    if type(side) is not str or side not in ("both", "left", "right"):
                        raise InputError("side must be 'both', 'left' or 'right'", field="side")
                    field = None
                    result = operations.limit(expression, variable, point, side=side, limits=limits)
                elif operation == "analyze_function":
                    result = operations.analyze_function(expression, variable, limits=limits)
                elif operation in ("factor", "discriminant"):
                    from ..polynomial import Polynomial
                    polynomial = Polynomial.from_expression(expression, variable, limits=limits)
                    result = (operations.factor(polynomial, limits=limits) if operation == "factor"
                              else operations.discriminant(polynomial, limits=limits))
                elif operation == "apart":
                    from ..rational_function import as_rational_function
                    with computation(limits):
                        expression = as_rational_function(expression, variable)
                    result = operations.apart(expression, limits=limits)
                elif operation == "extrema_values":
                    result = operations.extrema_values(expression, variable, limits=limits)
                else:
                    field = "lower"
                    lower = _scalar(arguments[field], field, limits)
                    field = "upper"
                    upper = _scalar(arguments[field], field, limits)
                    field = None
                    result = operations.integrate(expression, variable, lower, upper, limits=limits)
        return _operation_response(result, symbols)
    except _FAILURES as exc:
        return _response(symbols=symbols, error=_error(exc, stage, field), is_error=True)


def _operation_response(result, symbols):
    failed = result.execution_status is not ExecutionStatus.COMPLETED
    error = None
    if failed and result.error is not None:
        error = _error(MathForgeError(result.error.message, code=result.error.code), "computation")
    return _response(result, symbols=symbols, error=error, is_error=failed)


def _catalog(arguments: Any) -> ToolResponse:
    """Search the registry operations and the pre-registry ones; errors keep the catalog payload shape.

    A registry operation names the area tool that serves it, even when mathforge.compute also runs it. "total"
    counts every match, also those beyond the limit; there is no paging.
    """
    from .. import registry
    from ..registry_v07 import LEGACY_CATALOG
    from .kinds import encoded_default
    try:
        if type(arguments) is not dict or set(arguments) - {"query", "area", "limit"}:
            raise InputError("Expected optional fields ['area', 'limit', 'query']")
        query, area = arguments.get("query", ""), arguments.get("area")
        limit = arguments.get("limit", CATALOG_LIMIT)
        if type(query) is not str or len(query) > MAX_QUERY_CHARACTERS:
            raise InputError(f"query must be text of at most {MAX_QUERY_CHARACTERS} characters", field="query")
        if "area" in arguments and (type(area) is not str or area not in registry.AREAS):  # null is not an area
            raise InputError("area must name one of the registry areas", field="area")
        if type(limit) is not int or not 1 <= limit <= MAX_CATALOG_LIMIT:
            raise InputError(f"limit must be an integer from 1 through {MAX_CATALOG_LIMIT}", field="limit")
        entries = [{"name": spec.name, "area": spec.area, "tool": area_tool(spec) or "", "summary": spec.summary,
                    "arguments": [{"name": argument.name, "kind": argument.kind, "required": argument.default is None,
                                   "default": encoded_default(argument)} for argument in spec.arguments],
                    "returns": spec.returns} for spec in registry.specs().values()]
        entries += [{"name": entry.name, "area": entry.area, "tool": entry.tool, "summary": entry.summary,
                     "arguments": [], "returns": ""} for entry in LEGACY_CATALOG]
        needle = query.casefold()
        found = sorted((entry for entry in entries if area in (None, entry["area"])
                        and any(needle in entry[key].casefold() for key in ("name", "summary", "area"))),
                       key=lambda entry: (entry["area"], entry["name"]))
        return ToolResponse({"api_version": 1, "operations": found[:limit], "total": len(found), "error": None})
    except _FAILURES as exc:
        return ToolResponse({"api_version": 1, "operations": [], "total": 0, "error": _error(exc, "input")}, True)
