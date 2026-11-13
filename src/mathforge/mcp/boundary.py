"""SDK-independent, bounded adapters: the thirty curated v0.7 tools, the area tools and mathforge.catalog.

The area tools and the catalog derive from the operation registry. An area tool runs one operation of its
areas, named in the request, with arguments typed by the argument kinds of mcp.kinds.

Only this module and mcp.kinds translate text and ordinary JSON into core objects. They do not implement
mathematical algorithms or turn persisted operation names into executable code: an operation name selects
a registry spec, never a Python attribute; the spec's public structured function in mathforge.operations
runs it, so that an operation which a v6 family supersedes answers as mathforge.operations does. Responses
preserve the core codec and its integer strings.

Every response of a computation tool names its result with a handle (mcp.handles) and keeps the result in this
process's bounded store; an argument that accepts a canonical envelope also accepts a handle, which resolves to the
stored document and is then decoded exactly as that document sent in full. Every error object carries a stable
"reason" and the "next_actions" that an agent may try, derived from the error's code, stage and field.
"""

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, TypedDict

from .. import operations
from ..capabilities import CATALOG_TOOL, area_tool, area_tools, capabilities
from ..contracts import Eq, ExecutionStatus, Result, VerificationReport
from ..errors import InvalidInput, MathForgeError, SerializationError
from ..limits import ComputationLimits, DecodeLimits, computation, current_budget
from ..matrices import Matrix
from ..model import Expression, Rational, Symbol, free_symbols
from ..serialization import from_data, to_data
from ..verification import verify
from . import handles


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
    reason: str
    next_actions: list[str]


@dataclass(frozen=True)
class ToolResponse:
    payload: dict[str, Any]
    is_error: bool = False


class InputError(InvalidInput):
    """A boundary input failure. reason, when given, refines the reason that the error object reports (its code
    stays as it is): for example unknown_operation for an invalid_input that names no operation of the tool. stage,
    when given, is the stage that the error object reports, whichever stage the call had reached. fixed=True marks a
    resource_limit_exceeded of a cap that no limits field raises, such as the request byte limit."""

    def __init__(self, message: str, *, code: str = "invalid_input", field: str | None = None,
                 reason: str | None = None, stage: str | None = None, fixed: bool = False):
        super().__init__(message, code=code)
        self.field = field
        if reason is not None:
            self.reason = reason
        self.stage = stage
        if fixed:
            self.fixed_limit = True


_SYMBOL_IDS = ("pass symbol_ids as an object from variable names to UUID strings, copied from the \"symbols\" of "
               "an earlier response",)
# What an agent may try next, for each reason; a reason that is absent here has no suggestion.
NEXT_ACTIONS: dict[str, tuple[str, ...]] = {
    "unknown_tool": ("call mathforge.catalog to find the tool that runs an operation",),
    "unknown_operation": ("call mathforge.catalog with a query to find the operation and the tool that runs it",),
    "unsupported": ("call mathforge.catalog with a query to find an operation that covers this input",),
    "resource_limit_exceeded": ("retry with larger limits (each field at most its default maximum) or a smaller "
                                "input",),
    "limit_out_of_range": ("give each limits field a positive integer no greater than its default; "
                           "mathforge.capabilities lists the defaults under \"limits\"",),
    "response_too_large": ("retry with a smaller input, or split the computation into smaller calls; the response "
                           "byte limit is fixed",),
    "presentation_failed": ("use the canonical result in \"result\"; only its text and LaTeX presentation failed",),
    "unknown_handle": ("repeat the call that returned this handle and pass the handle of its new response instead",
                       "or send this argument in full, in its friendly form when its kind has one (mathforge.catalog "
                       "names the kind)"),
    "invalid_symbol_ids": _SYMBOL_IDS,
    "invalid_symbol_table": _SYMBOL_IDS,
    "symbol_conflict": ("give each symbol identity one name: pass symbol_ids built from the \"symbols\" of the "
                        "earlier response and use those names",),
    "symbol_mismatch": ("pass symbol_ids copied from the \"symbols\" of the earlier response, so that each name of "
                        "this request is the symbol of that name in the documents and handles it passes on; pass the "
                        "same symbol_ids to every call whose results are combined",),
    "invalid_document": ("send the canonical envelope exactly as an earlier response returned it, or that "
                         "response's handle",),
    "parse_syntax": ("correct the mathematical text of the field that \"field\" names, at the characters from "
                     "\"start\" to \"end\", and call again",),
    "unknown_symbol": ("declare every name that the text uses, in variables or as the variable argument, and call "
                       "again",),
}
_REASONS = {"unsupported_operation": "unsupported", "invalid_serialization": "invalid_document"}
# The variants of some reasons' next actions, chosen by the call's context.
_AT_LIMIT = ("retry with a smaller or simpler input; no limits field can raise this limit further",)
_CLOSED_TEXT = ("write the number as closed mathematical text that uses no names, such as \"sqrt(2)/3\", or send "
                "its canonical envelope, and call again",)
_UNKNOWN_OPERAND = ("repeat the call that returned this handle and pass the handle of its new response instead",
                    "or send this operand in full, as a canonical object envelope in its minimum format version")
_VERIFY_DOCUMENT = ("send the \"result\" of an earlier response exactly as it was returned: mathforge.verify takes "
                    "the whole Result document, not a handle",)
_PLAIN_DOCUMENT = ("send the canonical envelope exactly as an earlier response returned it",)
COMPUTE_TOOL = "mathforge.compute"
VERIFY_TOOL = "mathforge.verify"


def _takes_handles(tool: str | None) -> bool:
    """Whether a tool's arguments take handles: the area tools and mathforge.compute; None, an unknown tool, counts."""
    return tool is None or tool == COMPUTE_TOOL or tool in area_tools()


def _advice(code: str, stage: str, field: str | None, *, reason: str | None = None, status: Any = None,
            operation: str | None = None, tool: str | None = None, raisable: bool = True, closed: bool = False,
            mismatch: bool = False) -> tuple[str, list[str]]:
    """The reason and next actions of an error, from its code, stage, field and the call's context, never its text.

    operation is a registry operation that the call named (never unchecked request text), tool the tool called.
    raisable=False says that no limits field can raise the limit that ran out: it is a fixed cap, or the request's
    limits are already the defaults, which are also the maxima. closed=True says that the field takes closed text, a
    number, so that a name cannot be declared for it. mismatch=True says that the decoded arguments hold two symbol
    identities under one name, as a handle or an envelope of an earlier call without symbol_ids does beside text.
    """
    if stage == "presentation":
        reason = "presentation_failed"
    elif reason is None:
        if status is ExecutionStatus.UNSUPPORTED:
            reason = "unsupported"
        elif code == "invalid_input" and (field or "").startswith("symbol_ids"):
            reason = "invalid_symbol_ids"
        else:
            reason = _REASONS.get(code, code)
        if mismatch and reason == "invalid_input":
            reason = "symbol_mismatch"
    if reason == "invalid_input":
        if tool == CATALOG_TOOL:
            actions: tuple[str, ...] = ("call mathforge.catalog with only the optional fields query, area and limit, "
                                        "as its input schema describes",)
        elif operation is not None and field is None:  # The inputs reached the operation, which refused them.
            actions = (f"call mathforge.catalog with query \"{operation}\" to see the arguments of {operation} and "
                       "their kinds", f"the inputs are outside what {operation} accepts, as \"message\" says; change "
                       "them and call again")
        elif operation is not None:
            actions = (f"call mathforge.catalog with query \"{operation}\" to see the arguments of {operation} and "
                       "their kinds", "correct the input that \"field\" names and call again")
        else:
            actions = ("check the arguments against this tool's input schema",
                       "call mathforge.catalog with the operation's name as query to see its arguments and their kinds")
    elif reason == "resource_limit_exceeded" and not raisable:
        actions = _AT_LIMIT
    elif reason == "unknown_symbol" and closed:
        actions = _CLOSED_TEXT
    elif reason == "unknown_handle" and tool == COMPUTE_TOOL:
        actions = _UNKNOWN_OPERAND
    elif reason == "invalid_document" and not _takes_handles(tool):
        actions = _VERIFY_DOCUMENT if tool == VERIFY_TOOL else _PLAIN_DOCUMENT
    else:
        actions = NEXT_ACTIONS.get(reason, ())
    return reason, list(actions)


def _error(exc: Exception, stage: str, field: str | None = None, *, status: Any = None,
           operation: str | None = None, tool: str | None = None, limits: ComputationLimits | None = None,
           mismatch: bool = False) -> BoundaryError:
    """An error object; limits are the request's, None before they were read."""
    code = getattr(exc, "code", "invalid_input")
    stage = (exc.stage if type(exc) is InputError else None) or stage
    field = getattr(exc, "field", None) or field
    # A larger limit can help only below the defaults, which are the maxima, and only for a limit that a field sets.
    raisable = (limits is not None and limits != ComputationLimits() and not getattr(exc, "fixed_limit", False)
                and not getattr(exc, "parse_limit", False))
    reason, actions = _advice(code, stage, field, reason=getattr(exc, "reason", None), status=status,
                              operation=operation, tool=tool, raisable=raisable,
                              closed=getattr(exc, "closed_text", False), mismatch=mismatch)
    return {"stage": stage, "code": code, "message": str(exc), "field": field,
            "start": getattr(exc, "start", None), "end": getattr(exc, "end", None),
            "reason": reason, "next_actions": actions}


def _response(value=None, *, symbols=None, error=None, is_error=False) -> ToolResponse:
    """The response wrapper; a returned result is named by its handle and kept in the process store."""
    symbols = symbols or {}
    result = to_data(value) if value is not None else None
    text = handle = None
    if result is not None:
        text = handles.canonical_text(result)
        handle = handles.handle_of_text(text)
    payload = {"api_version": 1, "result": result, "handle": handle,
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
        payload.update(result=None, handle=None, text=error["message"], latex=None, error=error)
        is_error = True
    elif handle is not None:
        handles.STORE.put(handle, text)
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
        raise InputError("Rational input is too long", code="resource_limit_exceeded", field=field, fixed=True)
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
            raise InputError("Scalar text exceeds the character limit", code="resource_limit_exceeded", field=field,
                             fixed=True)
        try:
            scalar = parse_expression(value, symbols={}, limits=limits,
                                      parse_limits=ParseLimits(max_chars=MAX_SCALAR_CHARACTERS))
        except MathForgeError as exc:
            if getattr(exc, "code", None) == "unknown_symbol":
                exc.closed_text = True  # Read by _error: the field takes a number, with no name to declare.
            raise
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
        raise InputError("Too many variables", code="resource_limit_exceeded", field="variables", fixed=True)
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


def _request_decode_limits(limits: ComputationLimits) -> DecodeLimits:
    """The decoder limits of an envelope sent in a request."""
    return DecodeLimits(max_bytes=MAX_REQUEST_BYTES, max_depth=min(128, limits.max_depth), max_nodes=limits.max_nodes)


class Resolver:
    """Resolves the handles of one request.

    request_bytes starts at the size of the request as sent; each resolved handle adds the size of its document less
    that of the handle, so that the request, with every handle replaced by its document, stays within
    MAX_REQUEST_BYTES as a request that sent those documents in full must.
    """

    def __init__(self, request_bytes: int = 0):
        self.request_bytes = request_bytes

    def __call__(self, reference: str, field: str, limits: ComputationLimits) -> dict[str, Any]:
        return resolve_handle(reference, field, limits, account=lambda envelope: self._account(envelope, reference,
                                                                                                field))

    def _account(self, envelope: Any, reference: str, field: str) -> None:
        self.request_bytes += (len(json.dumps(envelope, ensure_ascii=True, allow_nan=False).encode("utf-8"))
                               - len(json.dumps(reference)))
        if self.request_bytes > MAX_REQUEST_BYTES:
            raise InputError("The request, with its handles replaced by their documents, exceeds the byte limit",
                             code="resource_limit_exceeded", field=field, fixed=True)


def resolve_handle(reference: str, field: str, limits: ComputationLimits, *, account=None) -> dict[str, Any]:
    """The canonical object envelope that a handle stands for, to be checked and decoded as one sent in full.

    The handle names a document of this process's store, or the call fails with unknown_handle. A Result stands for
    its value: the value's data is rebuilt within the request budget, under the Result's format version and the
    decoder limits of an envelope sent in a request, and its canonical envelope is returned; a Result without a value
    is an input error. Any other document stands for itself. account, if given, sees the document or the value's
    envelope before it is decoded. The caller then applies the checks of an envelope sent in full; this function
    adds no exemption from them.
    """
    text = handles.STORE.get(reference)
    if text is None:
        raise InputError("This server process holds no document with this handle: it may have been evicted from "
                         "the bounded store, or it was made by another process", code="unknown_handle", field=field,
                         stage="input")
    document = json.loads(text)  # The canonical text of a returned result, whose digest the store has just checked.
    if (type(document) is not dict or document.get("kind") != "object"
            or type(document.get("data")) is not dict or type(document.get("version")) is not int):
        raise InputError("The handle names no mathematical object", field=field)
    data = document["data"]
    if data.get("type") != "result":
        if account is not None:
            account(document)
        return document
    if data.get("value") is None:
        raise InputError("The handle names a Result without a value, so it stands for no argument", field=field)
    candidate = {"format": document["format"], "version": document["version"], "kind": "object",
                 "data": data["value"]}
    if account is not None:
        account(candidate)
    return to_data(_decode_in_budget(candidate, field, limits))


def _decode_in_budget(envelope: dict[str, Any], field: str, limits: ComputationLimits) -> Any:
    """from_data's steps (guard, decoder, error mapping) with the reconstruction charged to the active request budget,
    as kinds._decode_charged rebuilds the envelopes of the v0.9 kinds."""
    from ..serialization import _Decoder, _decode_limits, _guard_data
    decode_limits = _request_decode_limits(limits)
    token = _decode_limits.set(decode_limits)
    try:
        _guard_data(envelope, decode_limits)
        with computation(limits):  # The active request budget, never a fresh one.
            return _Decoder(version=envelope["version"]).decode(envelope["data"])
    except SerializationError as exc:
        raise InputError(str(exc), code=exc.code, field=field) from exc
    except (MathForgeError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        if getattr(exc, "code", None) == "resource_limit_exceeded":
            raise InputError(str(exc), code="resource_limit_exceeded", field=field) from exc
        raise InputError("Invalid mathematical document", code="invalid_serialization", field=field) from exc
    finally:
        _decode_limits.reset(token)


def _compute_arguments(operation, arguments, limits, request_bytes=0):
    from ..dispatch_v5 import operation_arguments
    contracts = operation_arguments()
    if type(operation) is not str or operation not in contracts:
        raise InputError("operation must be a published v0.7 operation", field="operation", reason="unknown_operation")
    if type(arguments) is not dict or set(arguments) != contracts[operation]:
        raise InputError("arguments must contain exactly the operation's published arguments", field="arguments")
    resolve = Resolver(request_bytes)
    arguments = {name: resolve(value, "arguments." + name, limits) if handles.is_handle(value) else value
                 for name, value in arguments.items()}
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
    limits = None
    try:
        limits = _limits(arguments.get("limits")) if type(arguments) is dict else None
        with computation(limits, fresh=True):
            return _call_tool(name, arguments)
    except _FAILURES as exc:
        return _response(error=_error(exc, "input", tool=name if type(name) is str else None, limits=limits),
                         is_error=True)


def _call_tool(name: str, arguments: dict[str, Any]) -> ToolResponse:
    """Execute one curated boundary call, without importing the optional SDK."""
    stage, field, symbols = "input", None, {}
    known = None  # The registry operation that the call names, once checked; it refines the errors' next actions.
    limits = None  # The request's limits, once read.
    mismatch = False  # Whether the decoded arguments of an area tool hold one name under two symbol identities.
    tool = name if type(name) is str else None
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
        request_bytes = len(json.dumps(arguments, ensure_ascii=True, allow_nan=False).encode("utf-8"))
        if request_bytes > MAX_REQUEST_BYTES:
            raise InputError("Request exceeds the byte limit", code="resource_limit_exceeded", fixed=True)
        if operation == "capabilities":
            return ToolResponse(dict(capabilities()))
        limits = _limits(arguments.get("limits"))
        if area:
            from .. import registry
            from .kinds import Context, decode_arguments
            specs = {spec.name: spec for spec in registry.tool_operations(name)}
            if type(arguments["operation"]) is not str or arguments["operation"] not in specs:
                raise InputError("operation must name an operation of this tool", field="operation",
                                 reason="unknown_operation")
            spec = specs[arguments["operation"]]
            known = spec.name
            context = Context(limits, arguments.get("symbol_ids"), request_bytes=request_bytes)
            stage, symbols = "decoding", context.symbols  # The table fills while the arguments decode.
            decoded = decode_arguments(spec, arguments["arguments"], context)
            stage, mismatch = "computation", bool(context.mismatched)
            structured = getattr(operations, spec.public_name, None)  # A registry spec's name, never the request's.
            if structured is None:  # A family registered without regenerated wrappers, as a test may register one.
                from .. import operations_v6
                return _operation_response(operations_v6.run(spec.name, decoded, limits=limits), symbols, known,
                                           limits=limits, tool=tool, mismatch=mismatch)
            return _operation_response(structured(**decoded, limits=limits), symbols, known, limits=limits,
                                       tool=tool, mismatch=mismatch)
        if operation == "parse_function":
            from ..functions import parse_function
            names = arguments["variables"]
            if type(names) is not list:
                raise InputError("variables must be an ordered array of declared names", field="variables")
            symbols = _symbols(names, arguments.get("symbol_ids"))
            text = _text(arguments["text"], "text")
            if len(text) > MAX_TEXT_CHARACTERS:
                raise InputError("Function text exceeds the character limit", code="resource_limit_exceeded", field="text",
                                 fixed=True)
            stage, field = "parsing", "text"
            value = parse_function(text, symbols=symbols, variables=tuple(symbols.values()), limits=limits)
            return _response(value, symbols=symbols)
        if operation == "compute":
            from .. import registry
            from ..dispatch_v5 import operation_arguments
            if type(arguments["operation"]) is str and arguments["operation"] in operation_arguments():
                known = arguments["operation"]
            stage, field = "decoding", "arguments"
            decoded = _compute_arguments(arguments["operation"], arguments["arguments"], limits, request_bytes)
            stage, field = "computation", None
            # The operation's public structured function: a v0.7 operation that a v6 family supersedes answers
            # through it; any other one through operations_v5.run, as in v0.7.
            structured = getattr(operations, registry.spec(arguments["operation"]).public_name)
            result = structured(**decoded, limits=limits)
            return _operation_response(result, symbols, known, limits=limits, tool=tool)
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
                    return _operation_response(result, symbols, limits=limits, tool=tool)
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
        return _operation_response(result, symbols, limits=limits, tool=tool)
    except _FAILURES as exc:
        return _response(symbols=symbols, error=_error(exc, stage, field, operation=known, tool=tool, limits=limits,
                                                       mismatch=mismatch and stage == "computation"),
                         is_error=True)


def _operation_response(result, symbols, operation=None, *, limits=None, tool=None, mismatch=False):
    failed = result.execution_status is not ExecutionStatus.COMPLETED
    error = None
    if failed and result.error is not None:
        error = _error(MathForgeError(result.error.message, code=result.error.code), "computation",
                       status=result.execution_status, operation=operation, tool=tool, limits=limits,
                       mismatch=mismatch)
    return _response(result, symbols=symbols, error=error, is_error=failed)


def _catalog(arguments: Any) -> ToolResponse:
    """Search the registry operations and the pre-registry ones; errors keep the catalog payload shape.

    A registry operation names the area tool that serves it, even when mathforge.compute also runs it; a legacy
    operation that a v6 family supersedes is listed once, as the registry operation it now is. "total" counts
    every match, also those beyond the limit; there is no paging.
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
        specs = registry.specs()
        entries = [{"name": spec.name, "area": spec.area, "tool": area_tool(spec) or "", "summary": spec.summary,
                    "arguments": [{"name": argument.name, "kind": argument.kind, "required": argument.default is None,
                                   "default": encoded_default(argument)} for argument in spec.arguments],
                    "returns": spec.returns} for spec in specs.values()]
        entries += [{"name": entry.name, "area": entry.area, "tool": entry.tool, "summary": entry.summary,
                     "arguments": [], "returns": ""} for entry in LEGACY_CATALOG if entry.name not in specs]
        needle = query.casefold()
        found = sorted((entry for entry in entries if area in (None, entry["area"])
                        and any(needle in entry[key].casefold() for key in ("name", "summary", "area"))),
                       key=lambda entry: (entry["area"], entry["name"]))
        return ToolResponse({"api_version": 1, "operations": found[:limit], "total": len(found), "error": None})
    except _FAILURES as exc:
        return ToolResponse({"api_version": 1, "operations": [], "total": 0,
                             "error": _error(exc, "input", tool=CATALOG_TOOL)}, True)
