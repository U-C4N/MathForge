"""Bounded exact presentation, deliberately separate from computation and codecs.

No output is a serialization format or a new verification claim. Existing
``str`` methods and content-addressed mathematical objects remain untouched.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from itertools import chain
import json
import re
from typing import Any

from .algebraic import RealAlgebraicRoot, RootRecord
from .analysis_types import (ExcludedPoint, FunctionAnalysis, LimitValue,
                             LocalExtremum, MonotonicRegion, ExtremumValue)
from .contracts import (EmptySet, Eq, FiniteSet, Inequality, Result, UniversalSet,
                        VerificationReport)
from .errors import InvalidInput, UnsupportedOperation
from .intervals import Interval, IntervalSet
from .limits import ResourceLimitError
from .matrices import AffineSolutionSet, LinearSystem, Matrix, RREFResult
from .model import Add, Domain, Mul, Pow, Rational, Sqrt, Symbol
from .polynomial import Polynomial, SquareFreeDecomposition, SquareFreeFactor
from .polynomial_exact import PolynomialFactor, PolynomialFactorization
from .exact_extensions import PartialFractionTerm, PartialFractionDecomposition
from .rational_function import RationalFunction


@dataclass(frozen=True)
class RenderLimits:
    max_chars: int = 65_536
    max_nodes: int = 10_000
    max_depth: int = 128

    def __post_init__(self) -> None:
        for field in fields(self):
            if type(getattr(self, field.name)) is not int or getattr(self, field.name) <= 0:
                raise InvalidInput(f"{field.name} must be a positive integer")


_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
            "_": r"\_", "^": r"\textasciicircum{}", "%": r"\%",
            "&": r"\&", "#": r"\#", "$": r"\$", "~": r"\textasciitilde{}"}


def _children(value: Any) -> Iterable[Any]:
    """Only visible, allowlisted fields are inspected; never walk certificates."""
    t = type(value)
    if t in (int, bool, str, Rational, Symbol, type(None), Domain):
        return ()
    if t is Add:
        return value.terms
    if t is Mul:
        return value.factors
    if t is Pow:
        return value.base, value.exponent
    if t is Sqrt:
        return (value.radicand,)
    if t is Polynomial:
        return chain((value.variable,), value.coefficients)
    if t is RationalFunction:
        return chain((value.numerator, value.denominator), value.excluded)
    if t is LimitValue:
        return (value.value,) if value.value is not None else ()
    if t in (ExcludedPoint, LocalExtremum):
        return value.point, value.kind
    if t is ExtremumValue:
        return value.point, value.value, value.kind
    if t is PolynomialFactor:
        return value.polynomial, value.multiplicity
    if t is PolynomialFactorization:
        return value.coefficient, value.factors
    if t is PartialFractionTerm:
        return value.numerator, value.factor, value.power
    if t is PartialFractionDecomposition:
        return value.polynomial, value.terms, value.excluded
    if t is MonotonicRegion:
        return value.interval, value.behavior
    if t is FunctionAnalysis:
        return (value.function, value.domain, value.excluded_points, value.zero_set,
                value.positive_set, value.negative_set, value.derivative,
                value.stationary_set, value.monotonic_regions, value.strict_extrema)
    if t is RealAlgebraicRoot:
        return chain(value.integer_coefficients, (value.real_index,))
    if t is RootRecord:
        return value.root, value.multiplicity
    if t in (Eq, Inequality):
        return value.lhs, value.rhs
    if t is Matrix:
        return value.rows
    if t is RREFResult:
        return value.matrix, value.pivot_columns
    if t is LinearSystem:
        return value.coefficients, value.rhs, value.variables
    if t is AffineSolutionSet:
        return value.variables, value.particular, value.basis
    if t is Interval:
        return value.lower, value.upper
    if t is IntervalSet:
        return value.intervals
    if t is FiniteSet:
        return value.values
    if t in (EmptySet, UniversalSet):
        return ()
    if t is SquareFreeFactor:
        return value.polynomial, value.multiplicity
    if t is SquareFreeDecomposition:
        return value.coefficient, value.factors
    if t is Result:
        return (value.value, value.deferred)
    if t is VerificationReport:
        return (check.claim for check in value.checks)
    if t in (tuple, list):
        return value
    from . import rendering_v6
    if rendering_v6.supported(value):
        return rendering_v6.children(value)
    from .rendering_v5 import supported, children
    if supported(value):
        return children(value)
    raise UnsupportedOperation(f"Presentation is unsupported for {t.__name__}")


class _Renderer:
    def __init__(self, value: Any, symbols: Mapping[str, Symbol] | None,
                 limits: RenderLimits | None, latex: bool):
        if limits is not None and type(limits) is not RenderLimits:
            raise InvalidInput("limits must be RenderLimits or None")
        self.limits = limits or RenderLimits()
        self.latex = latex
        self.nodes = 0
        aliases: dict[str, str] = {}
        identities: dict[str, Symbol] = {}
        if symbols is not None:
            if not isinstance(symbols, Mapping):
                raise InvalidInput("symbols must map aliases to Symbols")
            for count, (alias, symbol) in enumerate(symbols.items(), 1):
                if count > self.limits.max_nodes:
                    self.exhausted("symbol count")
                if type(alias) is not str or not alias.strip() or type(symbol) is not Symbol:
                    raise InvalidInput("symbols must map nonempty aliases to Symbols")
                self.bound(alias)
                if symbol.uid in aliases:
                    raise InvalidInput("A Symbol identity cannot have multiple aliases")
                aliases[symbol.uid], identities[symbol.uid] = alias, symbol
        stack = [(value, 0)]
        found: dict[str, Symbol] = {}
        scheduled = 1
        while stack:
            node, depth = stack.pop()
            if depth > self.limits.max_depth:
                self.exhausted("nodes/depth")
            if type(node) is Symbol:
                previous = identities.get(node.uid, found.get(node.uid))
                if previous is not None and previous != node:
                    raise InvalidInput("Conflicting Symbol metadata for the same identity")
                found[node.uid] = node
            # Charge before enqueueing, not on popping: wide nested inputs must
            # not allocate an unbounded pending stack before reaching the cap.
            for child in _children(node):
                scheduled += 1
                if scheduled > self.limits.max_nodes:
                    self.exhausted("nodes")
                stack.append((child, depth + 1))
        names: dict[str, str] = {}
        self.names: dict[str, str] = {}
        for uid, symbol in found.items():
            name = aliases.get(uid, symbol.name)
            self.bound(name)
            if name in names and names[name] != uid:
                raise InvalidInput("Ambiguous Symbol names; supply distinct aliases in symbols")
            names[name] = uid
            self.names[uid] = name

    def exhausted(self, what: str) -> None:
        raise ResourceLimitError(f"Render limit exceeded: {what}")

    def bound(self, value: str) -> str:
        if len(value) > self.limits.max_chars:
            self.exhausted("characters")
        return value

    def join(self, parts: Iterable[str], separator: str = "") -> str:
        output: list[str] = []
        length = 0
        for part in parts:
            length += len(part) + (len(separator) if output else 0)
            if length > self.limits.max_chars:
                self.exhausted("characters")
            output.append(part)
        return separator.join(output)

    def integer(self, value: int) -> str:
        # Avoid Python's process-global int-to-string limit without changing it.
        if value.bit_length() > self.limits.max_chars * 4:
            self.exhausted("integer characters")
        number, chunks = abs(value), []
        while number >= 1_000_000_000:
            number, tail = divmod(number, 1_000_000_000)
            chunks.append(f"{tail:09d}")
        return self.bound(("-" if value < 0 else "") + str(number) + "".join(reversed(chunks)))

    def label(self, text: str) -> str:
        self.bound(text)
        if not self.latex:
            return self.bound(text)
        escaped = self.join(_ESCAPES.get(char, char if char.isprintable() else " ") for char in text)
        return self.bound(r"\text{" + escaped + "}")

    def name(self, text: str) -> str:
        if self.latex:
            display = text if text.isprintable() else json.dumps(text, ensure_ascii=True)
            return text if re.fullmatch(r"[A-Za-z]", text) else self.label(display)
        return text if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text) else self.bound(json.dumps(text, ensure_ascii=True))

    def parens(self, text: str) -> str:
        return self.bound((r"\left(" + text + r"\right)") if self.latex else "(" + text + ")")

    def sequence(self, values: Iterable[Any], depth: int, *, braces: bool = False) -> str:
        contents = self.join((self.render(v, depth=depth + 1) for v in values), ", ")
        if braces:
            return self.bound((r"\left\{" + contents + r"\right\}") if self.latex else "{" + contents + "}")
        return self.parens(contents)

    def sum_text(self, parts: Iterable[str]) -> str:
        def signed_parts() -> Iterable[str]:
            for index, part in enumerate(parts):
                yield part if index == 0 else (" - " + part[1:] if part.startswith("-") else " + " + part)
        return self.join(signed_parts()) or "0"

    def polynomial(self, coefficients: Iterable[Rational | int], variable: str, depth: int) -> str:
        def terms() -> Iterable[str]:
            for power, coefficient in reversed(tuple(enumerate(coefficients))):
                n, d = ((coefficient.numerator, coefficient.denominator) if type(coefficient) is Rational else (coefficient, 1))
                if n == 0:
                    continue
                c = self.render(coefficient, depth=depth + 1)
                if power:
                    term = variable if power == 1 else self.bound(variable + ("^{" + self.integer(power) + "}" if self.latex else "^" + self.integer(power)))
                    c = ("-" if n == -1 and d == 1 else "" if n == 1 and d == 1 else c + (r"\, " if self.latex else "*")) + term
                yield c
        return self.sum_text(terms())

    def render(self, value: Any, precedence: int = 0, depth: int = 0) -> str:
        self.nodes += 1
        if self.nodes > self.limits.max_nodes or depth > self.limits.max_depth:
            self.exhausted("nodes/depth")
        t = type(value)
        child = lambda v, p=0: self.render(v, p, depth + 1)
        level = 100
        if t is int:
            text = self.integer(value)
            level = 25 if value < 0 else 100
        elif t is bool:
            text = self.label("true" if value else "false")
        elif t is Rational:
            n, d = self.integer(value.numerator), self.integer(value.denominator)
            if value.denominator == 1:
                text = n
            elif self.latex:
                text = ("-" if n.startswith("-") else "") + r"\frac{" + n.lstrip("-") + "}{" + d + "}"
            else:
                text = n + "/" + d
            level = 25 if value.numerator < 0 else (20 if value.denominator != 1 and not self.latex else 100)
        elif t is Symbol:
            text = self.name(self.names[value.uid])
        elif t is Add:
            level, text = 10, self.sum_text(child(term, 10) for term in value.terms)
        elif t is Mul:
            factors = value.factors
            negative = type(factors[0]) is Rational and factors[0] == Rational(-1)
            if negative:
                factors = factors[1:]
            level = 20
            # A persisted raw Mul need not be normalized. Explicit multiplication
            # and parentheses keep numeric/negative factors unambiguous as well.
            text = ("-" if negative else "") + self.join((child(factor, 26) for factor in factors), r" \cdot " if self.latex else "*")
        elif t is Pow:
            level = 30
            text = child(value.base, 31) + ("^{" + self.integer(value.exponent) + "}" if self.latex else "^" + self.integer(value.exponent))
        elif t is Sqrt:
            text = (r"\sqrt{" + child(value.radicand) + "}") if self.latex else "sqrt(" + child(value.radicand) + ")"
        elif t is Polynomial:
            level, text = 10, self.polynomial(value.coefficients, child(value.variable), depth)
        elif t is RealAlgebraicRoot:
            polynomial = self.polynomial(value.integer_coefficients, "t", depth)
            index = self.integer(value.real_index)
            text = (r"\operatorname{RootOf}_{\mathbb{R}}\left(" + polynomial + "; " + index + r"\right)") if self.latex else f"RootOf({polynomial}; real_index={index})"
        elif t is RootRecord:
            text = child(value.root) + (r"\; " if self.latex else "; ") + self.label("multiplicity=" + self.integer(value.multiplicity))
        elif t is RationalFunction:
            n, d = child(value.numerator), child(value.denominator)
            text = (r"\frac{" + n + "}{" + d + "}") if self.latex else self.parens(n) + "/" + self.parens(d)
            if value.excluded:
                conditions = self.join((child(p) + (r" \ne 0" if self.latex else " != 0") for p in value.excluded), ", ")
                text += (r"\quad\text{where } " if self.latex else " where ") + conditions
            level = 0
        elif t in (Eq, Inequality):
            relation = "=" if t is Eq else {"lt": "<", "le": r"\le" if self.latex else "<=", "gt": ">", "ge": r"\ge" if self.latex else ">="}[value.relation]
            text, level = child(value.lhs, 1) + " " + relation + " " + child(value.rhs, 1), 0
        elif t is LimitValue:
            if value.kind == "finite":
                text = child(value.value)
            elif value.kind == "does_not_exist":
                text = self.label("limit does not exist")
            else:
                text = ("+" if value.kind == "positive_infinity" else "-") + (r"\infty" if self.latex else "inf")
        elif t in (ExcludedPoint, LocalExtremum):
            text = child(value.point) + (r"\; " if self.latex else ": ") + self.label(
                "strict local " + value.kind if t is LocalExtremum else value.kind)
        elif t is ExtremumValue:
            text = self.parens(child(value.point) + ", " + child(value.value)) + (r"\; " if self.latex else ": ") + self.label("strict local " + value.kind)
        elif t is PartialFractionTerm:
            denominator = self.parens(child(value.factor)) + ("^{" + self.integer(value.power) + "}" if self.latex else "^" + self.integer(value.power))
            text = (r"\frac{" + child(value.numerator) + "}{" + denominator + "}") if self.latex else self.parens(child(value.numerator)) + "/" + self.parens(denominator)
        elif t is PartialFractionDecomposition:
            text = self.join((child(item) for item in (value.polynomial, *value.terms)), " + ")
            if value.excluded:
                text += (r"\quad\text{where } " if self.latex else " where ") + self.join((child(p) + (r" \ne 0" if self.latex else " != 0") for p in value.excluded), ", ")
        elif t is MonotonicRegion:
            text = child(value.interval) + (r"\; " if self.latex else ": ") + self.label(
                value.behavior if value.behavior == "constant" else "strictly " + value.behavior)
        elif t is FunctionAnalysis:
            rows = (("function", value.function), ("domain", value.domain),
                    ("excluded points", value.excluded_points), ("zeros", value.zero_set),
                    ("positive", value.positive_set), ("negative", value.negative_set),
                    ("derivative", value.derivative), ("stationary points", value.stationary_set),
                    ("monotonic regions", value.monotonic_regions),
                    ("strict extrema (x)", value.strict_extrema))
            text = self.join((self.label(label) + (" & = " if self.latex else ": ") + child(item)
                              for label, item in rows), r" \\ " if self.latex else "\n")
            if self.latex:
                text = r"\begin{aligned}" + text + r"\end{aligned}"
        elif t is Matrix:
            if value.nrows == 0 or value.ncols == 0:
                text = self.label(f"Matrix({value.nrows}x{value.ncols})")
            elif self.latex:
                text = r"\begin{bmatrix}" + self.join((self.join((child(entry) for entry in row), " & ") for row in value.rows), r" \\ ") + r"\end{bmatrix}"
            else:
                text = "[" + self.join(("[" + self.join((child(entry) for entry in row), ", ") + "]" for row in value.rows), ", ") + "]"
        elif t is RREFResult:
            text = child(value.matrix) + (r"\quad " if self.latex else "; ") + self.label("pivot columns (0-based): ") + child(value.pivot_columns)
        elif t is LinearSystem:
            text = child(value.coefficients) + (r"\, " if self.latex else " * ") + child(value.variables) + " = " + child(value.rhs)
        elif t is AffineSolutionSet:
            def affine_terms() -> Iterable[str]:
                yield child(value.particular)
                for index, vector in enumerate(value.basis):
                    parameter = (r"\lambda_{" + str(index) + "}") if self.latex else f"parameter[{index}]"
                    yield parameter + (r"\, " if self.latex else "*") + child(vector)
            text = child(value.variables) + " = " + self.join(affine_terms(), " + ")
            if value.basis:
                text += (r"\quad " if self.latex else "; ") + self.label("parameters are independent real scalars")
        elif t is Interval:
            low = child(value.lower) if value.lower is not None else (r"-\infty" if self.latex else "-inf")
            high = child(value.upper) if value.upper is not None else (r"\infty" if self.latex else "inf")
            text = ("[" if value.left_closed else "(") + low + ", " + high + ("]" if value.right_closed else ")")
        elif t is IntervalSet:
            text = self.join((child(item) for item in value.intervals), r" \cup " if self.latex else " union ") or (r"\varnothing" if self.latex else "{}")
        elif t is FiniteSet:
            text = self.sequence(value.values, depth, braces=True)
        elif t is EmptySet:
            text = r"\varnothing" if self.latex else "{}"
        elif t in (UniversalSet, Domain):
            domain = value.domain if t is UniversalSet else value
            text = {"reals": r"\mathbb{R}", "rationals": r"\mathbb{Q}", "integers": r"\mathbb{Z}"}[domain.value] if self.latex else "All(" + domain.value + ")"
        elif t in (SquareFreeFactor, PolynomialFactor):
            text = self.parens(child(value.polynomial)) + ("^{" + self.integer(value.multiplicity) + "}" if self.latex else "^" + self.integer(value.multiplicity))
        elif t in (SquareFreeDecomposition, PolynomialFactorization):
            text = self.join((child(item) for item in (value.coefficient, *value.factors)), r"\, " if self.latex else "*")
        elif t is Result:
            status = f"{value.execution_status.value} / {value.outcome.value} [{value.exactness.value}; {value.completeness.value}]"
            text = self.label(status) + (r"\quad " if self.latex else ": ")
            text += child(value.value) if value.value is not None else child(value.deferred) if value.deferred is not None else self.label(value.error.code + ": " + value.error.message if value.error else "No mathematical value")
        elif t is VerificationReport:
            text = self.label("Check report: ") + self.join((self.label(check.claim + "=" + check.status.value) for check in value.checks), r";\; " if self.latex else "; ")
        elif t in (tuple, list):
            text = self.sequence(value, depth)
        elif t is str:
            text = self.label(value)
        elif value is None:
            text = self.label("None")
        else:
            from . import rendering_v6
            if rendering_v6.supported(value):
                text = rendering_v6.render(value, self, child)
            else:
                from .rendering_v5 import supported, render
                if not supported(value):
                    raise UnsupportedOperation(f"Presentation is unsupported for {t.__name__}")
                text = render(value, self, child)
        self.bound(text)
        return self.parens(text) if level < precedence else text


def render_text(value: Any, *, symbols: Mapping[str, Symbol] | None = None,
                limits: RenderLimits | None = None) -> str:
    """Render exact values without changing them or certifying stored claims."""
    return _render(value, symbols, limits, False)


def to_latex(value: Any, *, symbols: Mapping[str, Symbol] | None = None,
             limits: RenderLimits | None = None) -> str:
    """Return escaped LaTeX math source (no delimiters, compiler or approximation)."""
    return _render(value, symbols, limits, True)


def _render(value: Any, symbols: Mapping[str, Symbol] | None,
            limits: RenderLimits | None, latex: bool) -> str:
    try:
        return _Renderer(value, symbols, limits, latex).render(value)
    except RecursionError as exc:
        raise ResourceLimitError("Render limit exceeded: host recursion depth") from exc
