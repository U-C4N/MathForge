"""Bounded mathematical text input using the existing exact MathForge model.

Names are resolved only through an explicit caller-provided symbol table.  This
module is a front end, not an additional solver or a Python evaluator.  Decimal
literals are exact rationals; symbolic division uses RationalFunction so that
cancelled denominator zeros remain excluded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import re
from types import MappingProxyType
from typing import Callable, cast

from .contracts import Eq, Inequality
from .errors import InvalidInput, MathForgeError, UnsupportedOperation
from .limits import Budget, ComputationLimits, ResourceLimitError, computation
from .model import Expression, Rational, Symbol, free_symbols, sqrt
from .rational_function import RationalFunction, as_rational_function


__all__ = ["ParseError", "ParseLimits", "parse_expression", "parse_problem"]

Scalar = Expression | RationalFunction
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_RELATIONS = {"=": "eq", "==": "eq", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
_INFIX = {"+": (10, 11), "-": (10, 11), "*": (20, 21), "/": (20, 21),
          "**": (30, 30), "^": (30, 30)}


def _locate(error: MathForgeError, text: str, start: int, end: int) -> None:
    """Attach source coordinates without changing a core error's type or code."""
    if hasattr(error, "start"):
        return
    setattr(error, "start", start)
    setattr(error, "end", end)
    setattr(error, "line", text.count("\n", 0, start) + 1)
    setattr(error, "column", start - text.rfind("\n", 0, start))


class ParseError(InvalidInput):
    """Syntax/name error with a zero-based, half-open character span.

    ``line`` and ``column`` are one-based.  Offsets count Python string
    characters, not UTF-8 bytes.  The offending input is not retained.
    """

    code = "parse_syntax"
    start: int
    end: int
    line: int
    column: int

    def __init__(self, message: str, *, text: str = "", start: int = 0,
                 end: int | None = None, code: str | None = None) -> None:
        super().__init__(message, code=code)
        _locate(self, text, start, start if end is None else end)


@dataclass(frozen=True)
class ParseLimits:
    """Syntax limits, additional to an operation-local ComputationLimits.

    Tokens exclude the end marker.  Nodes count literals, symbols, unary/binary
    operations, square-root calls, and the top-level comparison; parentheses do
    not add nodes.  Depth counts active expression frames, including the root.
    Integer digits count all digits in a literal, on both sides of a decimal
    point.  Exponent limits apply to the evaluated integer exponent.
    """

    max_chars: int = 16_384
    max_tokens: int = 4_096
    max_nodes: int = 4_096
    max_depth: int = 64
    max_integer_digits: int = 1_024
    max_symbols: int = 64
    max_exponent: int = 1_024

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise InvalidInput(f"{field.name} must be a positive integer")


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int


def _syntax(text: str, token: _Token, message: str, *, code: str | None = None) -> ParseError:
    return ParseError(message, text=text, start=token.start, end=token.end, code=code)


def _resource(text: str, start: int, end: int, name: str) -> ResourceLimitError:
    error = ResourceLimitError(f"Parse limit exceeded: {name}")
    _locate(error, text, start, end)
    return error


def _symbol_table(values: Mapping[str, Symbol], text: str, limits: ParseLimits,
                  budget: Budget) -> Mapping[str, Symbol]:
    if not isinstance(values, Mapping):
        raise ParseError("symbols must be an explicit name-to-Symbol mapping", text=text,
                         code="invalid_symbol_table")
    copied: dict[str, Symbol] = {}
    identities: set[str] = set()
    for name, value in values.items():
        budget.tick()
        if len(copied) >= limits.max_symbols:
            raise _resource(text, 0, 0, "symbols")
        if type(name) is not str or _IDENTIFIER.fullmatch(name) is None or name == "sqrt":
            raise ParseError("Symbol aliases must be ASCII identifiers other than sqrt", text=text,
                             code="invalid_symbol_table")
        if type(value) is not Symbol:
            raise ParseError("Every symbol alias must refer to a Symbol", text=text,
                             code="invalid_symbol_table")
        if value.uid in identities:
            raise ParseError("A Symbol identity may have only one alias in a symbol table", text=text,
                             code="invalid_symbol_table")
        identities.add(value.uid)
        copied[name] = value
    return MappingProxyType(copied)


def _tokenize(text: str, limits: ParseLimits, budget: Budget) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        char = text[position]
        if char.isspace():
            position += 1
            continue
        start = position
        if "0" <= char <= "9" or (char == "." and position + 1 < len(text)
                                     and "0" <= text[position + 1] <= "9"):
            while position < len(text) and "0" <= text[position] <= "9":
                position += 1
            if position < len(text) and text[position] == ".":
                position += 1
                while position < len(text) and "0" <= text[position] <= "9":
                    position += 1
            raw = text[start:position]
            if len(raw) - raw.count(".") > limits.max_integer_digits:
                raise _resource(text, start, position, "integer digits")
            kind = "number"
        elif "A" <= char <= "Z" or "a" <= char <= "z":
            position += 1
            while position < len(text):
                current = text[position]
                if not ("A" <= current <= "Z" or "a" <= current <= "z"
                        or "0" <= current <= "9" or current == "_"):
                    break
                position += 1
            kind = "name"
        else:
            pair = text[position:position + 2]
            if pair in ("**", "==", "<=", ">="):
                position += 2
            elif char in "+-*/^()=<>":
                position += 1
            else:
                raise ParseError("Unexpected character in mathematical input", text=text,
                                 start=start, end=start + 1)
            kind = text[start:position]
        if len(tokens) >= limits.max_tokens:
            raise _resource(text, start, position, "tokens")
        try:
            budget.tick()
        except MathForgeError as error:
            _locate(error, text, start, position)
            raise
        tokens.append(_Token(kind, text[start:position], start, position))
    tokens.append(_Token("end", "", len(text), len(text)))
    return tuple(tokens)


class _Parser:
    def __init__(self, text: str, symbols: Mapping[str, Symbol], limits: ParseLimits,
                 budget: Budget) -> None:
        self.text = text
        self.symbols = symbols
        self.limits = limits
        self.budget = budget
        self.tokens = _tokenize(text, limits, budget)
        self.position = 0
        self.nodes = 0
        self.depth = 0

    @property
    def token(self) -> _Token:
        return self.tokens[self.position]

    def _take(self) -> _Token:
        token = self.token
        if token.kind != "end":
            self.position += 1
        return token

    def _node(self, token: _Token) -> None:
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise _resource(self.text, token.start, token.end, "nodes")
        try:
            self.budget.tick()
        except MathForgeError as error:
            _locate(error, self.text, token.start, token.end)
            raise

    def _expect(self, kind: str) -> _Token:
        if self.token.kind != kind:
            raise _syntax(self.text, self.token, f"Expected {kind!r}")
        return self._take()

    def _perform(self, token: _Token, operation: Callable[[], Scalar]) -> Scalar:
        try:
            return operation()
        except MathForgeError as error:
            _locate(error, self.text, token.start, token.end)
            raise
        except ZeroDivisionError as cause:
            failure = InvalidInput("Division by zero", code="division_by_zero")
            _locate(failure, self.text, token.start, token.end)
            raise failure from cause

    def _number(self, token: _Token) -> Rational:
        whole, dot, fraction = token.text.partition(".")
        digits = whole + fraction
        integer = 0
        # Chunked conversion respects an explicitly raised digit limit without
        # modifying Python's process-global integer-string conversion limit.
        for offset in range(0, len(digits), 9):
            chunk = digits[offset:offset + 9]
            scale = 10 ** len(chunk)
            self.budget.product(integer, scale)
            integer = integer * scale + int(chunk)
            self.budget.check_int(integer)
        if not dot or not fraction:
            return Rational(integer)
        denominator = Rational(10) ** len(fraction)
        return Rational(integer, denominator.numerator)

    @staticmethod
    def _variables(value: Scalar) -> frozenset[Symbol]:
        if type(value) is RationalFunction:
            return frozenset((value.variable,))
        return free_symbols(value)

    def _binary(self, token: _Token, left: Scalar, right: Scalar) -> Scalar:
        kind = token.kind
        if kind in ("**", "^"):
            if type(right) is not Rational or right.denominator != 1 or right.numerator < 0:
                raise UnsupportedOperation("An exponent must be a nonnegative exact integer")
            if right.numerator > self.limits.max_exponent:
                raise _resource(self.text, token.start, token.end, "exponent")
            return left ** right.numerator
        if kind == "/" and type(right) is Rational and right.numerator == 0:
            raise InvalidInput("Division by zero", code="division_by_zero")
        rational_operands = type(left) is RationalFunction or type(right) is RationalFunction
        if rational_operands or (kind == "/" and type(right) is not Rational):
            variables = self._variables(left) | self._variables(right)
            if len(variables) != 1:
                raise UnsupportedOperation("Symbolic division requires exactly one real variable")
            variable = next(iter(variables))
            left = as_rational_function(left, variable)
            right = as_rational_function(right, variable)
        if kind == "+":
            return left + right
        if kind == "-":
            return left - right
        if kind == "*":
            return left * right
        return left / right

    def expression(self, minimum: int = 0) -> Scalar:
        self.depth += 1
        try:
            if self.depth > self.limits.max_depth:
                token = self.token
                raise _resource(self.text, token.start, token.end, "depth")
            token = self._take()
            if token.kind == "number":
                self._node(token)
                left = self._perform(token, lambda: self._number(token))
            elif token.kind == "name":
                self._node(token)
                if token.text == "sqrt":
                    self._expect("(")
                    argument = self.expression()
                    self._expect(")")
                    if type(argument) is RationalFunction:
                        error = UnsupportedOperation("Square roots require a constant rational radicand")
                        _locate(error, self.text, token.start, token.end)
                        raise error
                    left = self._perform(token, lambda: sqrt(cast(Expression, argument)))
                elif token.text in self.symbols:
                    left = self.symbols[token.text]
                else:
                    raise _syntax(self.text, token, f"Unknown symbol {token.text!r}", code="unknown_symbol")
            elif token.kind in ("+", "-"):
                self._node(token)
                argument = self.expression(25)
                left = argument if token.kind == "+" else self._perform(token, lambda: -argument)
            elif token.kind == "(":
                left = self.expression()
                self._expect(")")
            else:
                raise _syntax(self.text, token, "Expected a number, symbol, sqrt call, or parenthesized expression")
            while self.token.kind in _INFIX:
                operator = self.token
                binding, right_binding = _INFIX[operator.kind]
                if binding < minimum:
                    break
                self._take()
                self._node(operator)
                right = self.expression(right_binding)
                left = self._perform(operator, lambda: self._binary(operator, left, right))
            return left
        finally:
            self.depth -= 1

    def finish(self, problem: bool) -> Scalar | Eq | Inequality:
        left = self.expression()
        if problem:
            token = self.token
            if token.kind not in _RELATIONS:
                raise _syntax(self.text, token, "Expected one equation or inequality relation")
            self._take()
            self._node(token)
            right = self.expression()
            self._expect("end")
            relation = _RELATIONS[token.kind]
            value = Eq(left, right) if relation == "eq" else Inequality(left, relation, right)
            self.budget.inspect(value)
            return value
        self._expect("end")
        self.budget.inspect(left)
        return left


def _parse(text: str, symbols: Mapping[str, Symbol], parse_limits: ParseLimits | None,
           limits: ComputationLimits | None, *, problem: bool) -> Scalar | Eq | Inequality:
    if type(text) is not str:
        raise ParseError("Mathematical input must be text")
    if parse_limits is not None and type(parse_limits) is not ParseLimits:
        raise ParseError("parse_limits must be ParseLimits or None")
    syntax_limits = ParseLimits() if parse_limits is None else parse_limits
    if len(text) > syntax_limits.max_chars:
        raise _resource(text, syntax_limits.max_chars, len(text), "characters")
    parser = None
    try:
        with computation(limits) as budget:
            table = _symbol_table(symbols, text, syntax_limits, budget)
            parser = _Parser(text, table, syntax_limits, budget)
            return parser.finish(problem)
    except MathForgeError as error:
        token = parser.token if parser is not None else _Token("end", "", 0, 0)
        _locate(error, text, token.start, token.end)
        raise
    except RecursionError as cause:
        # An explicitly raised syntax depth may reach the host's stack limit
        # before its requested limit.  This remains a resource failure.
        token = parser.token if parser is not None else _Token("end", "", 0, 0)
        raise _resource(text, token.start, token.end, "host recursion depth") from cause


def parse_expression(text: str, *, symbols: Mapping[str, Symbol],
                     parse_limits: ParseLimits | None = None,
                     limits: ComputationLimits | None = None) -> Scalar:
    """Parse exact scalar math, resolving names only through ``symbols``.

    Supports integers, finite decimals, explicit arithmetic, nonnegative integer
    powers (``**`` or ``^``), parentheses and ``sqrt`` of rational constants.
    Scientific notation, implicit multiplication and arbitrary function calls
    are not accepted.  ``symbols={}`` is explicit for a constant expression.
    """
    value = _parse(text, symbols, parse_limits, limits, problem=False)
    assert isinstance(value, (Expression, RationalFunction))
    return value


def parse_problem(text: str, *, symbols: Mapping[str, Symbol],
                  parse_limits: ParseLimits | None = None,
                  limits: ComputationLimits | None = None) -> Eq | Inequality:
    """Parse exactly one ``=``/``==`` equation or ``< <= > >=`` inequality."""
    value = _parse(text, symbols, parse_limits, limits, problem=True)
    assert isinstance(value, (Eq, Inequality))
    return value
