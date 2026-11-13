"""Argument kinds of the MCP area tools: the JSON form of every registry argument kind.

Each registry Argument names a kind (registry.KIND_NAMES). A Kind holds the JSON Schema of such an argument,
a decoder from JSON to the Python value the operation receives, and an encoder from a Python value to JSON.
Every kind takes a friendly JSON form or a canonical object envelope, exactly what to_data returns (format
versions 1-6), as mathforge.compute does; the "object" kind takes only the envelope. Compound forms are made
of their element kinds, so an element may itself be an envelope, for example a real algebraic root binding.
Envelopes pass compute's checks: their shape, a walk within the request budget that admits operand tags only
(compute's and those of the v6 family records; an interval_set argument's own envelope also those of an interval
set), and the canonical minimum format. The kinds of v0.9 charge the reconstruction of their envelopes, the ends
of intervals included, to the request budget; the kinds of v0.8 keep v0.8's fresh budget for each envelope. An
interval_set argument, in either form, is then rebuilt once more from its canonical envelope, in a fresh budget of the
default limits divided by RESULT_COPIES, the copies of it that a Result may hold, so that such a Result reloads.

Wherever a kind takes an envelope, it also takes a handle, the string "mf:sha256:<64 hex digits>" that a response
returned (mcp.handles). The handle resolves through boundary.resolve_handle to the canonical envelope of the stored
document, or of a stored Result's value, which then goes through exactly the checks of that envelope sent in full; an
unknown handle is an unknown_handle error at the argument. No friendly string form contains a colon, so a handle never
reads as friendly input, and a string that only resembles a handle is friendly input, decoded as before.

Decoding runs no code. Integer and fraction strings are bounded digit strings; scalar, function, polynomial
and rational-function text goes through the bounded MathForge parsers; names resolve through one request-local
symbol table that symbol_ids seeds. JSON decimal numbers and true/false are never exact numbers.

Schemas are definitions: a kind's schema refers to the kinds it is made of, and to the envelope, as
"#/$defs/<name>", and definitions() collects the $defs table they need. arguments_schema(spec) carries its
own table, and an area tool's schema collects the tables of its operations once.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
import re
from types import MappingProxyType
from typing import Any

from .. import registry
from ..algebraic import RealAlgebraicRoot, as_endpoint
from ..complex_numbers import ComplexAlgebraic
from ..errors import MathForgeError, SerializationError, UnsupportedOperation
from ..functions import RealFunction, parse_function
from ..intervals import Interval, IntervalSet
from ..limits import ComputationLimits, DecodeLimits, computation
from ..matrices import Matrix
from ..model import Expression, Rational, Symbol
from ..operations_v5 import _freeze
from ..operations_v6 import _RELATIONS
from ..polynomial import Polynomial
from ..probability import FiniteDistribution
from ..rational_function import RationalFunction, as_rational_function
from ..serialization import from_data, to_data
from .boundary import (MAX_REQUEST_BYTES, MAX_SCALAR_CHARACTERS, MAX_TEXT_CHARACTERS, MAX_VARIABLES, InputError,
                       Resolver, _decode_envelopes, _envelope, _rational, _scalar, _symbols, operand_tags)
from .handles import PATTERN as HANDLE_PATTERN, is_handle
from .schemas import NAME, TEXT

ENVELOPE_VERSIONS = (1, 2, 3, 4, 5, 6)
_INTEGER = re.compile(r"[+-]?[0-9]+")
_SIDES = ("both", "left", "right")
_INTERVAL_FIELDS = ("lower", "upper", "left_closed", "right_closed")
# The tags of an IntervalSet and its Intervals: operands only in the whole-argument envelope of an interval_set
# argument. Every other kind, an end of an interval included, keeps v0.8's located decoding error for them, and
# compute's operand tags stay the ones that its frozen v0.7 schema enumerates.
_INTERVAL_TAGS = frozenset({"interval", "interval_set"})
# The most intervals that a friendly interval_set may list; an envelope may hold more. A Result repeats its request in
# its step, and its value too, and reloading it rebuilds every copy with full checks: under the default limits,
# mathforge.verify reloaded an echo's Result, which holds the set four times, with at most 666 intervals of small
# rational ends, 97 of ends such as 2*k + sqrt(2) and 49 of sqrt(10^12 + k) (measured 2026-11-12). The cap bounds the
# request; RESULT_COPIES bounds the cost of rebuilding the set, also of fewer intervals with costlier ends. 32
# intervals with ends of the first two kinds pass both.
MAX_INTERVALS = 32
# Reloading a Result (mathforge.verify, mf.loads, mf.from_data) rebuilds every copy of a value that it holds, in one
# budget of the default limits. A set operation on two sets holds each argument in its request and in its step's copy
# of the request, and its value in the step's outputs and as the value, where the value may be as large as both
# arguments together: 2 + 2 + 2 * 2 = 8 copies of the larger argument. Rebuilding a request also inspects its
# arguments, which adds up to about 28% of their nodes for rational ends, and a value can cost a little more than its
# parts: with 9 copies, unions of two disjoint sets of 337 intervals with rational ends that each passed did not
# reload, with 10 every such union tried did (measured 2026-11-12). So an interval_set argument, in either form, must
# rebuild from its canonical envelope within the default limits divided by RESULT_COPIES: those that the copies share
# (work, nodes, refinements and evidence items, and the decoder's bytes and tagged nodes). The limits on one integer,
# on depth and on a matrix stay whole. This bounds the copies of the arguments, not a value that compares ends of
# different arguments, which neither argument's rebuilding compared: A = {(sqrt(10^400 + 1), 2 * 10^200)} and
# B = {(sqrt(10^400 + 2), 3 * 10^200)} each pass, while their difference A \ B = {(sqrt(10^400 + 1), sqrt(10^400 + 2)]}
# costs 1,661,069 units of work to rebuild (measured 2026-11-12). An operation whose value pairs ends of different
# arguments must bound the rebuilding of its value itself.
RESULT_COPIES = 10


def _shares(limits: Any, *names: str) -> Any:
    """Limits with each named limit divided by RESULT_COPIES."""
    return replace(limits, **{name: getattr(limits, name) // RESULT_COPIES for name in names})


_RELOAD_LIMITS = _shares(ComputationLimits(), "max_work", "max_nodes", "max_refinements", "max_evidence_items")
_RELOAD_DECODE_LIMITS = _shares(DecodeLimits(), "max_bytes", "max_nodes")
_ENVELOPE = {"type": "object", "additionalProperties": False, "required": ["format", "version", "kind", "data"],
             "description": "A canonical MathForge object envelope, exactly as to_data returns it.",
             "properties": {"format": {"const": "mathforge"}, "version": {"enum": list(ENVELOPE_VERSIONS)},
                            "kind": {"const": "object"}, "data": {"type": "object", "required": ["type"]}}}
_HANDLE = {"type": "string", "pattern": HANDLE_PATTERN,
           "description": "The handle of an earlier result of this server process, in place of its envelope."}
_DEFINITION = "#/$defs/"
_SHARED = ("envelope", "handle")  # The $defs entries that are not kinds.


def _ref(name: str) -> dict[str, str]:
    """A reference to the $defs entry of a kind, or of the envelope."""
    return {"$ref": _DEFINITION + name}


class Context:
    """The decoding state of one request: its limits, its symbol table and the position being decoded."""

    def __init__(self, limits: ComputationLimits, symbol_ids: Any = None, request_bytes: int = 0):
        self.limits = limits
        self.field = "arguments"
        self._resolver = Resolver(request_bytes)  # request_bytes: the size of the request as sent
        self.symbols: dict[str, Symbol] = {}  # The names the request used, in first-use order.
        # The names that two symbol identities share in the decoded arguments, set by decode_arguments: a handle or an
        # envelope of an earlier call made without symbol_ids, beside text of this one, gives a name two identities.
        self.mismatched: tuple[str, ...] = ()
        ids = {} if symbol_ids is None else symbol_ids
        if type(ids) is not dict:
            raise InputError("symbol_ids must map variable names to UUID strings", field="symbol_ids")
        try:
            self._declared = _symbols(list(ids), ids)
        except InputError as exc:
            if (exc.field or "").startswith("symbol_ids"):
                raise
            raise InputError(str(exc), code=exc.code, field="symbol_ids") from exc
        self._tags: frozenset[str] | None = None

    def error(self, message: str, *, code: str = "invalid_input", fixed: bool = False) -> InputError:
        """An input error at the current position; fixed=True marks a cap that no limits field raises."""
        return InputError(message, code=code, field=self.field, fixed=fixed)

    @contextmanager
    def at(self, part: str) -> Iterator[None]:
        """Decode a nested position; an error raised inside without a field gets this position."""
        parent, self.field = self.field, self.field + part
        try:
            yield
        except Exception as exc:
            if getattr(exc, "field", None) is None:
                setattr(exc, "field", self.field)  # Read by boundary._error, as parsing attaches spans.
            raise
        finally:
            self.field = parent

    def symbol(self, name: Any) -> Symbol:
        """The Symbol of a name: the one symbol_ids declares, else a fresh one; the request reuses it."""
        if type(name) is str and name in self.symbols:
            return self.symbols[name]
        if len(self.symbols) >= MAX_VARIABLES:
            raise self.error("Too many variables", code="resource_limit_exceeded", fixed=True)
        if type(name) is str and name in self._declared:
            symbol = self._declared[name]
        else:
            try:
                symbol = _symbols([name], None)[name]
            except InputError as exc:
                raise self.error(str(exc), code=exc.code, fixed=getattr(exc, "fixed_limit", False)) from exc
        self.symbols[name] = symbol
        return symbol

    def resolve(self, reference: str) -> dict[str, Any]:
        """The canonical envelope that a handle stands for at the current position, not yet checked or decoded."""
        return self._resolver(reference, self.field, self.limits)

    def envelope(self, raw: Any, *, extra_tags: frozenset[str] = frozenset(), charged: bool = False) -> Any:
        """Decode a canonical object envelope (versions 1-6) with mathforge.compute's checks.

        extra_tags widens the operand tags for this envelope alone; only the interval_set kind passes any, the
        tags of an interval set, for its whole argument. charged=True, which the kinds of v0.9 pass, charges the
        reconstruction of the value to the active request budget.
        """
        if self._tags is None:
            # compute's operand tags and the family records of v6; the deferral records are not operands.
            self._tags = frozenset(operand_tags()) | {"v6_" + tag for tag, cls in registry.record_types(6).items()
                                                      if cls.__module__ != registry.DEFERRED_MODULE}
        tags = (self._tags | extra_tags) if extra_tags else self._tags
        _envelope(raw, self.field, self.limits, tags=tags, versions=ENVELOPE_VERSIONS)
        if charged:
            return _decode_charged(raw, self.field, self.limits)
        # v0.8 behaviour, kept for the kinds of v0.8 until the MCP hardening milestone: from_data rebuilds the value
        # in a fresh budget of its own, so the request budget pays only for the walk above.
        return _decode_envelopes([raw], [self.field], self.limits)[0]


def _decode_charged(envelope: dict[str, Any], field: str, limits: ComputationLimits) -> Any:
    """Decode a checked envelope as boundary._decode_envelopes does, but within the active request budget.

    from_data always rebuilds a value in a fresh budget, so every nested envelope, each end of an interval among
    them, would get a whole budget of its own. These are from_data's steps (the same one-item tuple document, guard,
    decoder and error mapping) with the reconstruction charged to the request instead.
    """
    from ..serialization import _Decoder, _decode_limits, _guard_data
    document = {"format": "mathforge", "version": envelope["version"], "kind": "object",
                "data": {"type": "tuple", "items": [envelope["data"]]}}
    decode_limits = DecodeLimits(max_bytes=MAX_REQUEST_BYTES, max_depth=min(128, limits.max_depth),
                                 max_nodes=limits.max_nodes)
    token = _decode_limits.set(decode_limits)
    try:
        _guard_data(document, decode_limits)
        with computation(limits) as budget:  # The active request budget, never a fresh one.
            (value,) = _Decoder(version=envelope["version"]).decode(document["data"])
    except SerializationError:
        raise
    except (MathForgeError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        if getattr(exc, "code", None) == "resource_limit_exceeded":
            raise SerializationError(str(exc), code="resource_limit_exceeded") from exc
        raise SerializationError("Invalid mathematical document") from exc
    finally:
        _decode_limits.reset(token)
    budget.nodes()  # As _decode_envelopes charges each operand's canonical check.
    if to_data(value) != envelope:
        raise InputError("Operands must use their canonical minimum wire format", field=field)
    return value


@dataclass(frozen=True)
class Kind:
    name: str
    # The kind's $defs entry, anyOf[friendly form, canonical object envelope]; references point to #/$defs/<name>.
    schema: dict[str, Any]
    decode: Callable[[Any, Context], Any]
    encode: Callable[[Any], Any]           # The friendly form; the envelope when that would not round-trip


def _kind(name: str, friendly: dict[str, Any] | None, decode_friendly: Callable[[Any, Context], Any] | None,
          encode_friendly: Callable[[Any], Any] | None, *, charged: bool = False,
          extra_tags: frozenset[str] = frozenset(), check: Callable[[Any, Context], None] | None = None) -> Kind:
    """A kind; charged and extra_tags apply to its whole-argument envelope, as Context.envelope takes them. check, if
    given, runs on the value that the kind's decoder returns, from either form; the encoder's round trip skips it."""
    def decode(raw: Any, context: Context) -> Any:
        with computation(context.limits):
            if is_handle(raw):  # The envelope it stands for, checked below as one sent in full.
                raw = context.resolve(raw)
            if type(raw) is dict and raw.get("format") == "mathforge":
                return context.envelope(raw, extra_tags=extra_tags, charged=charged)
            if decode_friendly is None:
                raise context.error("A canonical MathForge object envelope is required")
            return decode_friendly(raw, context)

    def encode(value: Any) -> Any:
        return _encode(value, decode, encode_friendly)

    envelope, handle = _ref("envelope"), _ref("handle")
    schema = {"anyOf": [envelope, handle] if friendly is None else [friendly, envelope, handle]}
    if check is None:
        return Kind(name, schema, decode, encode)
    after = check

    def checked(raw: Any, context: Context) -> Any:
        value = decode(raw, context)
        after(value, context)
        return value

    return Kind(name, schema, checked, encode)


def _encode(value: Any, decode: Callable[[Any, Context], Any], friendly: Callable[[Any], Any] | None) -> Any:
    """The friendly form of a value if it decodes to an identical object, else the canonical envelope.

    "Identical" compares canonical JSON, after the freezing that the runners apply to their arguments.
    """
    envelope = to_data(_freeze(value))
    if friendly is not None:
        try:
            with computation(ComputationLimits(), fresh=True):
                candidate = friendly(value)
                if candidate is not None:
                    decoded = decode(candidate, Context(ComputationLimits(), _symbol_ids(envelope)))
                    if to_data(_freeze(decoded)) == envelope:
                        return candidate
        except (MathForgeError, TypeError, ValueError, ArithmeticError, RecursionError):
            pass
    return envelope


def _symbol_ids(envelope: dict[str, Any]) -> dict[str, Any]:
    """The UUID of every symbol name in an envelope; a name that two symbols share has no friendly form."""
    ids: dict[str, Any] = {}
    pending = [envelope["data"]]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            if node.get("type") == "symbol" and ids.setdefault(node.get("name"), node.get("uid")) != node.get("uid"):
                raise ValueError("Two symbols share a name")
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)
    return ids


def _items(raw: Any, context: Context, decode: Callable[[Any, Context], Any], message: str) -> tuple[Any, ...]:
    if type(raw) is not list:
        raise context.error(message)
    with computation() as budget:  # As compute's walk charges containers, so a huge list fails at its field.
        budget.nodes(len(raw))
    values = []
    for index, item in enumerate(raw):
        with context.at(f"[{index}]"):
            values.append(decode(item, context))
    return tuple(values)


def _encode_items(encode: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: [encode(item) for item in value] if type(value) in (tuple, list) else None


def _exact(value: int) -> int:
    with computation() as budget:  # Callers decode inside the request budget.
        budget.check_int(value)
    return value


def _decode_int(raw: Any, context: Context) -> int:
    if type(raw) is int:
        return _exact(raw)
    if type(raw) is str and (len(raw) > MAX_SCALAR_CHARACTERS or _INTEGER.fullmatch(raw)):
        return _rational(raw, context.field).numerator  # Its length cap precedes any int() conversion.
    raise context.error("An exact integer is required: a JSON integer or a string of decimal digits; "
                        "JSON decimal numbers and booleans are not exact")


def _encode_int(value: Any) -> Any:
    if type(value) is not int:
        return None
    return value if abs(value) < 2 ** 53 else str(value)


def _decode_rational(raw: Any, context: Context) -> int | Rational:
    if type(raw) is int:
        return _exact(raw)
    if type(raw) is str:
        value = _rational(raw, context.field)
        return value if "/" in raw else value.numerator
    raise context.error("An exact rational is required: a JSON integer or a string such as '-3/4' or '12'; "
                        "JSON decimal numbers and booleans are not exact")


def _encode_rational(value: Any) -> Any:
    if type(value) is Rational:  # "p/q" even when q is 1: a digit string decodes to an int.
        return f"{value.numerator}/{value.denominator}"
    return _encode_int(value)


def _decode_scalar(raw: Any, context: Context) -> int | Expression:
    if type(raw) is int:
        return _exact(raw)
    if type(raw) is str:
        return _scalar(raw, context.field, context.limits)
    raise context.error("A scalar is a JSON integer or closed exact mathematical text, such as 'sqrt(2)/3'")


def _encode_scalar(value: Any) -> Any:
    if type(value) is not Rational and isinstance(value, Expression):
        from ..rendering import render_text
        return render_text(value)
    return _encode_rational(value)


def _decode_point(raw: Any, context: Context) -> int | Expression | str:
    if type(raw) is str and raw in ("-inf", "+inf"):
        return raw
    if type(raw) not in (int, str):
        raise context.error("A point is a JSON integer, closed exact mathematical text, '-inf' or '+inf'")
    return _decode_scalar(raw, context)


def _decode_side(raw: Any, context: Context) -> str:
    if type(raw) is str and raw in _SIDES:
        return raw
    raise context.error("side must be 'both', 'left' or 'right'")


def _decode_pair(raw: Any, context: Context) -> tuple[int, ...]:
    message = "Each pair is an array of two integers"
    if type(raw) is not list or len(raw) != 2:
        raise context.error(message)
    return _items(raw, context, INT.decode, message)


def _decode_complex(raw: Any, context: Context) -> ComplexAlgebraic:
    message = "A complex number is [real part, imaginary part], two scalars"
    if type(raw) is not list or len(raw) != 2:
        raise context.error(message)
    return ComplexAlgebraic(*_items(raw, context, SCALAR.decode, message))


def _decode_matrix(raw: Any, context: Context) -> Matrix:
    message = "A matrix is an array of rows, each an array of rationals"
    if type(raw) is not list or any(type(row) is not list for row in raw):
        raise context.error(message)
    if len(raw) > 4096 or sum(map(len, raw)) > context.limits.max_matrix_entries:
        raise context.error("Matrix input exceeds the entry limit", code="resource_limit_exceeded")
    return Matrix(_items(raw, context, lambda row, context: _items(row, context, RATIONAL.decode, message), message))


def _decode_distribution(raw: Any, context: Context) -> FiniteDistribution:
    if type(raw) is not dict or set(raw) != {"outcomes", "probabilities"}:
        raise context.error('A distribution is {"outcomes": [...], "probabilities": [...]}')
    parts = []
    for key in ("outcomes", "probabilities"):
        with context.at("." + key):
            parts.append(RATIONAL_LIST.decode(raw[key], context))
    return FiniteDistribution(*parts)


def _text(raw: Any, context: Context) -> str:
    if type(raw) is not str:
        raise context.error("Mathematical text is required")
    if len(raw) > MAX_TEXT_CHARACTERS:
        raise context.error("Mathematical text exceeds the character limit", code="resource_limit_exceeded",
                            fixed=True)
    return raw


def _decode_function(raw: Any, context: Context) -> RealFunction:
    if type(raw) is not dict or set(raw) != {"text", "variables"}:
        raise context.error('A function is {"text": ..., "variables": [...]}, with its variables in order')
    with context.at(".variables"):
        names = _items(raw["variables"], context, lambda name, context: context.symbol(name).name,
                       "variables must be an array of names")
        if len(set(names)) != len(names):
            raise context.error("variables must be distinct")
    symbols = {name: context.symbol(name) for name in names}
    with context.at(".text"):
        return parse_function(_text(raw["text"], context), symbols=symbols, variables=tuple(symbols.values()),
                              limits=context.limits)


def _encode_function(value: Any) -> Any:
    if type(value) is not RealFunction:
        return None
    from ..rendering import render_text
    return {"text": render_text(value.expression), "variables": [symbol.name for symbol in value.variables]}


def _decode_rational_function(raw: Any, context: Context) -> RationalFunction:
    if type(raw) is not dict or set(raw) != {"text", "variable"}:
        raise context.error('A rational function is {"text": ..., "variable": name}')
    with context.at(".variable"):
        variable = context.symbol(raw["variable"])
    with context.at(".text"):
        from ..parsing import parse_expression
        expression = parse_expression(_text(raw["text"], context), symbols={variable.name: variable},
                                      limits=context.limits)
        return as_rational_function(expression, variable)


def _encode_rational_function(value: Any) -> Any:
    if type(value) is not RationalFunction:
        return None
    from ..rendering import render_text
    return {"text": f"({render_text(value.numerator)})/({render_text(value.denominator)})",
            "variable": value.variable.name}


def _decode_polynomial(raw: Any, context: Context) -> Polynomial:
    if type(raw) is not dict or set(raw) != {"text", "variable"}:
        raise context.error('A polynomial is {"text": ..., "variable": name}')
    with context.at(".variable"):
        variable = context.symbol(raw["variable"])
    with context.at(".text"):
        from ..parsing import parse_expression
        expression = parse_expression(_text(raw["text"], context), symbols={variable.name: variable},
                                      limits=context.limits)
        if type(expression) is RationalFunction:  # The text divides by an expression in the variable.
            raise context.error("A polynomial divides by rational constants only")
        return Polynomial.from_expression(expression, variable)


def _encode_polynomial(value: Any) -> Any:
    if type(value) is not Polynomial:
        return None
    from ..rendering import render_text
    return {"text": render_text(value), "variable": value.variable.name}


def _decode_end(raw: Any, context: Context) -> Rational | RealAlgebraicRoot | None:
    """An end of an interval, as Interval admits it: None for an infinite end, else a Rational or a root."""
    if raw is None:
        return None
    if type(raw) not in (int, str, dict):
        raise context.error("An interval end is a JSON integer, closed exact mathematical text, or null for an "
                            "infinite end; JSON decimal numbers and booleans are not exact")
    if type(raw) is str and raw.strip() in ("inf", "+inf", "-inf"):  # How the point kind spells an infinity.
        raise context.error("An infinite end is null, not text")
    if is_handle(raw):  # The envelope it stands for, checked below as one sent in full.
        raw = context.resolve(raw)
    if type(raw) is dict and raw.get("format") == "mathforge":  # The scalar kind's test for an envelope.
        # A Rational or a RealAlgebraicRoot needs no interval tags; the request budget pays for the rebuilding.
        value = context.envelope(raw, charged=True)
    else:
        value = _decode_scalar(raw, context)
    if type(value) is int or isinstance(value, Expression):  # Else the envelope of another object.
        try:
            return as_endpoint(value, limits=context.limits)
        except UnsupportedOperation:  # A symbol, or square roots that are not rational multiples of each other.
            pass
    raise context.error("An interval end must be a rational number or a number of one quadratic field, written with "
                        "square roots that are rational multiples of each other, such as '(1+sqrt(5))/2'; send "
                        "another real algebraic number as a RealAlgebraicRoot envelope")


def _decode_interval(raw: Any, context: Context) -> Interval:
    if type(raw) is not dict or set(raw) != set(_INTERVAL_FIELDS):
        raise context.error('An interval is {"lower": ..., "upper": ..., "left_closed": ..., "right_closed": ...}; '
                            "null is an infinite end")
    ends, flags = [], []
    for end, flag in (("lower", "left_closed"), ("upper", "right_closed")):
        with context.at("." + end):
            ends.append(_decode_end(raw[end], context))
        with context.at("." + flag):
            if type(raw[flag]) is not bool:
                raise context.error(flag + " must be true or false")
            if raw[flag] and ends[-1] is None:
                raise context.error("An infinite end must be open")
            flags.append(raw[flag])
    return Interval(*ends, *flags)  # It rejects reversed ends.


def _decode_interval_set(raw: Any, context: Context) -> IntervalSet:
    """The canonical union of the intervals given: sorted, merged, and without the empty ones."""
    if type(raw) is list and len(raw) > MAX_INTERVALS:  # Before any end is decoded.
        raise context.error(f"An interval set has at most {MAX_INTERVALS} intervals")
    return IntervalSet(_items(raw, context, _decode_interval, "An interval set is an array of intervals"))


def _reloadable(value: Any, context: Context) -> None:
    """Refuse a decoded interval_set argument that a Result holding it RESULT_COPIES times could not reload.

    from_data rebuilds the canonical envelope of the value, as reloading a Result rebuilds each copy, in a fresh
    budget of the default limits divided by RESULT_COPIES; the request budget, which the decoding has charged, pays
    nothing for it. Running out there is a resource_limit_exceeded located at the argument."""
    try:
        from_data(to_data(value), decode_limits=_RELOAD_DECODE_LIMITS, computation_limits=_RELOAD_LIMITS)
    except SerializationError as exc:
        if exc.code != "resource_limit_exceeded":
            raise
        raise context.error(f"The interval set costs more than 1/{RESULT_COPIES} of the default limits to rebuild, so "
                            f"a result that holds it might not reload within them ({exc})",
                            code="resource_limit_exceeded", fixed=True) from exc


def _encode_end(value: Any) -> Any:
    if value is None:
        return None
    if type(value) is Rational and value.denominator == 1:  # An end decodes an integer to this same Rational.
        return _encode_int(value.numerator)
    return SCALAR.encode(value)


def _encode_interval_set(value: Any) -> Any:
    if type(value) is not IntervalSet:
        return None
    return [{"lower": _encode_end(item.lower), "upper": _encode_end(item.upper), "left_closed": item.left_closed,
             "right_closed": item.right_closed} for item in value.intervals]


def _decode_relation(raw: Any, context: Context) -> str:
    if type(raw) is str and raw in _RELATIONS:
        return raw
    raise context.error("A relation is '=', '<', '<=', '>' or '>='")


def _decode_bindings(raw: Any, context: Context) -> dict[Symbol, Any]:
    if type(raw) is not dict:
        raise context.error("bindings map variable names to scalars")
    bindings = {}
    for name, item in raw.items():
        with context.at("." + str(name)):
            bindings[context.symbol(name)] = SCALAR.decode(item, context)
    return bindings


def _encode_bindings(value: Any) -> Any:
    if not isinstance(value, Mapping) or any(type(key) is not Symbol for key in value):
        return None
    return {key.name: SCALAR.encode(item) for key, item in value.items()}


def _number(pattern: str | None = None, description: str | None = None) -> dict[str, Any]:
    """A JSON integer or a bounded string; anyOf rather than a list of types keeps the schema portable."""
    text: dict[str, Any] = {"type": "string", "maxLength": MAX_SCALAR_CHARACTERS}
    if pattern is not None:
        text["pattern"] = pattern
    schema: dict[str, Any] = {"anyOf": [{"type": "integer"}, text]}
    if description is not None:
        schema["description"] = description
    return schema


INT = _kind("int", _number("^[+-]?[0-9]+$"), _decode_int, _encode_int)
RATIONAL = _kind("rational", _number("^[+-]?[0-9]+(?:/[+-]?[0-9]+)?$"), _decode_rational, _encode_rational)
SCALAR = _kind("scalar", _number(description="A JSON integer or closed exact mathematical text."),
               _decode_scalar, _encode_scalar)
RATIONAL_LIST = _kind("rational_list", {"type": "array", "items": _ref("rational")},
                      lambda raw, context: _items(raw, context, RATIONAL.decode, "An array of rationals is required"),
                      _encode_items(RATIONAL.encode))
SYMBOL = _kind("symbol", NAME, lambda raw, context: context.symbol(raw),
               lambda value: value.name if type(value) is Symbol else None)
FUNCTION = _kind("function", {"type": "object", "additionalProperties": False, "required": ["text", "variables"],
                              "properties": {"text": TEXT, "variables": {"type": "array", "items": NAME,
                                                                         "maxItems": MAX_VARIABLES,
                                                                         "uniqueItems": True}}},
                 _decode_function, _encode_function)
_END = {"anyOf": [_ref("scalar"), {"type": "null"}]}  # An end of an interval; null is an infinite one.

KINDS: Mapping[str, Kind] = MappingProxyType({kind.name: kind for kind in (
    INT,
    _kind("int_list", {"type": "array", "items": _ref("int")},
          lambda raw, context: _items(raw, context, INT.decode, "An array of integers is required"),
          _encode_items(INT.encode)),
    _kind("int_pairs", {"type": "array", "items": {"type": "array", "items": _ref("int"), "minItems": 2,
                                                   "maxItems": 2}},
          lambda raw, context: _items(raw, context, _decode_pair, "An array of integer pairs is required"),
          _encode_items(lambda pair: [INT.encode(item) for item in pair]
                        if type(pair) in (tuple, list) and len(pair) == 2 else None)),
    RATIONAL,
    RATIONAL_LIST,
    SCALAR,
    _kind("complex", {"type": "array", "items": _ref("scalar"), "minItems": 2, "maxItems": 2,
                      "description": "[real part, imaginary part]"},
          _decode_complex,
          lambda value: [SCALAR.encode(value.real), SCALAR.encode(value.imag)]
          if type(value) is ComplexAlgebraic else None),
    _kind("matrix", {"type": "array", "maxItems": 4096, "items": {"type": "array", "maxItems": 4096,
                                                                  "items": _ref("rational")}},
          _decode_matrix,
          lambda value: [[RATIONAL.encode(item) for item in row] for row in value.rows]
          if type(value) is Matrix else None),
    _kind("distribution", {"type": "object", "additionalProperties": False,
                           "required": ["outcomes", "probabilities"],
                           "properties": {"outcomes": _ref("rational_list"), "probabilities": _ref("rational_list")}},
          _decode_distribution,
          lambda value: {"outcomes": [RATIONAL.encode(item) for item in value.outcomes],
                         "probabilities": [RATIONAL.encode(item) for item in value.probabilities]}
          if type(value) is FiniteDistribution else None),
    FUNCTION,
    _kind("function_list", {"type": "array", "items": _ref("function")},
          lambda raw, context: _items(raw, context, FUNCTION.decode, "An array of functions is required"),
          _encode_items(FUNCTION.encode)),
    _kind("rational_function", {"type": "object", "additionalProperties": False, "required": ["text", "variable"],
                                "properties": {"text": TEXT, "variable": NAME}},
          _decode_rational_function, _encode_rational_function),
    SYMBOL,
    _kind("symbols", {"type": "array", "items": NAME, "maxItems": MAX_VARIABLES},
          lambda raw, context: _items(raw, context, lambda name, context: context.symbol(name),
                                      "symbols are an array of names"),
          _encode_items(lambda symbol: symbol.name if type(symbol) is Symbol else None)),
    _kind("bindings", {"type": "object", "propertyNames": NAME, "additionalProperties": _ref("scalar"),
                       "maxProperties": MAX_VARIABLES}, _decode_bindings, _encode_bindings),
    _kind("point", _number(description="A JSON integer, closed exact mathematical text, '-inf' or '+inf'."),
          _decode_point, lambda value: value if type(value) is str else _encode_scalar(value)),
    _kind("side", {"type": "string", "enum": list(_SIDES)}, _decode_side,
          lambda value: value if type(value) is str else None),
    _kind("object", None, None, None),
    # The kinds of v0.9 charge the reconstruction of their envelopes to the request budget.
    _kind("polynomial", {"type": "object", "additionalProperties": False, "required": ["text", "variable"],
                         "properties": {"text": TEXT, "variable": NAME}},
          _decode_polynomial, _encode_polynomial, charged=True),
    _kind("interval_set", {"type": "array", "description": "A union of real intervals; null is an infinite end.",
                           "maxItems": MAX_INTERVALS,
                           "items": {"type": "object", "additionalProperties": False,
                                     "required": list(_INTERVAL_FIELDS),
                                     "properties": {"lower": _END, "upper": _END,
                                                    "left_closed": {"type": "boolean"},
                                                    "right_closed": {"type": "boolean"}}}},
          _decode_interval_set, _encode_interval_set, charged=True, extra_tags=_INTERVAL_TAGS, check=_reloadable),
    _kind("relation", {"type": "string", "enum": list(_RELATIONS)}, _decode_relation,
          lambda value: value if type(value) is str else None, charged=True),
)})


def definitions(kinds: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The $defs entries that the schemas of some kinds need, in KIND_NAMES order, then the envelope's and the handle's.

    They cover the kinds given and every kind that their schemas refer to.
    """
    needed: set[str] = set()
    pending = list(kinds)
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            pending.extend(_references(KINDS[name].schema))
    return {name: KINDS[name].schema for name in registry.KIND_NAMES if name in needed} | {"envelope": _ENVELOPE,
                                                                                          "handle": _HANDLE}


def _references(schema: Any) -> set[str]:
    """The kinds that a schema refers to."""
    found, pending = set(), [schema]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            reference = node.get("$ref")
            if type(reference) is str and reference.removeprefix(_DEFINITION) not in _SHARED:
                found.add(reference.removeprefix(_DEFINITION))
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)
    return found


def encoded_default(argument: registry.Argument) -> Any:
    """An argument's default value as JSON, through its kind; None when the argument is required."""
    if argument.default is None:
        return None
    return KINDS[argument.kind].encode(ast.literal_eval(argument.default))


def arguments_schema(spec: registry.OperationSpec) -> dict[str, Any]:
    """The JSON Schema of an operation's arguments object, complete with its own $defs.

    An argument with a default may be omitted. An area tool's schema keeps this schema, without its $defs, as
    the $defs entry of the operation, and gathers the $defs of all its operations at its root, where every
    reference resolves to the same definition.
    """
    properties = {}
    for argument in spec.arguments:
        reference = _ref(argument.kind)
        properties[argument.name] = (reference if argument.default is None
                                     else dict(reference, default=encoded_default(argument)))
    return {"type": "object", "description": f"{spec.name}: {spec.summary}", "properties": properties,
            "required": [argument.name for argument in spec.arguments if argument.default is None],
            "additionalProperties": False, "$defs": definitions(argument.kind for argument in spec.arguments)}


def decode_arguments(spec: registry.OperationSpec, raw: Mapping[str, Any], context: Context) -> dict[str, Any]:
    """Decode an operation's arguments in spec order; omitted arguments take their spec defaults."""
    context.field = "arguments"
    if type(raw) is not dict:
        raise context.error("arguments must be an object of the operation's named arguments")
    known = {argument.name for argument in spec.arguments}
    for name in raw:
        if name not in known:
            raise InputError("Unknown argument of " + spec.name, field="arguments." + str(name))
    for argument in spec.arguments:
        if argument.default is None and argument.name not in raw:
            raise InputError("Missing required argument of " + spec.name, field="arguments." + argument.name)
    values = {}
    with computation(context.limits):
        for argument in spec.arguments:
            if argument.name in raw:
                with context.at("." + argument.name):
                    values[argument.name] = KINDS[argument.kind].decode(raw[argument.name], context)
            elif argument.default is not None:  # Always, once the required arguments are present.
                values[argument.name] = ast.literal_eval(argument.default)
    # Each envelope has its own decoder, so one encoder checks every symbol together, as compute's single
    # decoder does: a symbol identity has one name, domain and set of assumptions in a request.
    try:
        document = to_data(_freeze(tuple(values.values())))
    except SerializationError as exc:
        raise InputError(str(exc), code=exc.code, field="arguments", reason=getattr(exc, "reason", None)) from exc
    context.mismatched = _shared_names(document)
    return values


def _shared_names(document: dict[str, Any]) -> tuple[str, ...]:
    """The symbol names that more than one symbol identity carries in a canonical document, sorted.

    The call still runs, as before; an error of its computation then has the reason symbol_mismatch, which tells the
    agent to pass the symbol_ids of the earlier response.
    """
    identities: dict[Any, set[Any]] = {}
    pending = [document["data"]]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            if node.get("type") == "symbol":
                identities.setdefault(node.get("name"), set()).add(node.get("uid"))
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)
    return tuple(sorted(str(name) for name, uids in identities.items() if len(uids) > 1))
