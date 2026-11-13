"""A scoreboard of natural mathematics requests, so that what MathForge can answer is measured release by release.

Each group module, ``benchmarks/probes/<group>.py`` for a group of ``GROUPS``, defines ``PROBES``: a tuple of
:class:`Probe` records. A probe has a stable snake_case id that starts with its group's name, the request in plain
words, ``run``, a callable that takes the ``mathforge`` module and returns a value or a ``Result``, and, wherever the
right answer is known, ``expect``, a callable that takes the value and returns True for the right answer. A request
that v0.8 cannot answer but a planned v0.9 operation will (docs/v09/catalogue.md) calls that planned API, so that its
status turns from unsupported to ok by itself when the operation lands; every other request keeps the natural v0.8
attempt. Inputs are fixed: no randomness at run time, and symbols carry fixed uids (:func:`symbols`).

:func:`run_probe` gives a probe one status, decided only from the outcome of its attempt and never from timing:

``ok``              a value that ``expect``, if the probe has one, accepts
``wrong``           a value that ``expect`` rejects, or on which it raises
``deferred``        a Result with the outcome NO_CONCLUSION, or an unevaluated record as the value
``unsupported``     UnsupportedOperation, an attribute or operation that MathForge lacks, the execution status
                    UNSUPPORTED, or an OperationError that carries such a Result
``invalid_input``   InvalidInput (a ParseError too), the execution status INVALID_INPUT
``resource_limit``  ResourceLimitError, an error Result with the code resource_limit_exceeded
``error``           any other exception (the TypeError of an operator that a class lacks, say) or error Result

tools/check_probes.py compares the statuses with the committed baseline, benchmarks/probes-v08.json, and
tests/v09/test_probes.py runs the same comparison in the unittest suite. The package never imports mathforge: the
caller passes the module it runs against, a checkout's src/ or an installed distribution, and the helpers below
recognize MathForge's values by the module and name of their class. benchmarks/ is a namespace package: import this
one from the repository root, as ``benchmarks.probes``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
import importlib
import os
import re
import traceback
from types import ModuleType
from typing import Any
import uuid

GROUPS = ("algebra", "equations", "calculus", "linalg", "ntheory", "discrete", "stats", "sets", "logic", "geometry",
          "numeric", "scaling")
STATUSES = ("ok", "wrong", "deferred", "unsupported", "invalid_input", "resource_limit", "error")

_ID = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)+")
# How a module without the attribute words its AttributeError, also when its __getattr__ raises it (the lazy area
# modules of mathforge): on Python 3.11 such an error need not carry the module as its obj.
_MISSING_MODULE_ATTRIBUTE = re.compile(r"module 'mathforge(\.\w+)*' has no attribute '\w+'")
_UNEVALUATED = frozenset({"Unevaluated", "UnevaluatedIntegral", "UnevaluatedLimit"})
_UID_NAMESPACE = uuid.UUID("6f1c5e1a-3d2b-4c8e-9a7f-0b5d4e3c2a19")
_DETAIL = 100  # Characters of a value's description in a detail.


@dataclass(frozen=True)
class Probe:
    """A natural request: its id, group and wording, its attempt and, where the answer is known, its check."""

    id: str
    group: str
    request: str
    run: Callable[[ModuleType], Any]
    expect: Callable[[Any], bool] | None = None


@dataclass(frozen=True)
class Outcome:
    """A probe's status and a one-line description of what its attempt returned or raised."""

    status: str
    detail: str


def all_probes() -> tuple[Probe, ...]:
    """Every probe, group by group in the order of GROUPS; ValueError for a malformed or duplicated probe."""
    found: list[Probe] = []
    seen: set[str] = set()
    for group in GROUPS:
        module = importlib.import_module(f"{__name__}.{group}")
        for probe in module.PROBES:
            if type(probe) is not Probe:
                raise ValueError(f"{module.__name__}.PROBES holds a {type(probe).__name__}, not a Probe")
            if probe.group != group or not _ID.fullmatch(probe.id) or not probe.id.startswith(group + "_"):
                raise ValueError(f"probe {probe.id!r} of {module.__name__}: the id must be snake_case and start "
                                 f"with {group + '_'!r}, and the group must be {group!r}")
            if probe.id in seen:
                raise ValueError(f"probe id {probe.id!r} is defined twice")
            if type(probe.request) is not str or not probe.request.strip():
                raise ValueError(f"probe {probe.id!r} has no request")
            if not callable(probe.run) or (probe.expect is not None and not callable(probe.expect)):
                raise ValueError(f"probe {probe.id!r} needs a callable run and a callable or absent expect")
            seen.add(probe.id)
            found.append(probe)
    return tuple(found)


def run_all(mf: ModuleType, probes: Iterable[Probe] | None = None) -> dict[str, Outcome]:
    """The outcome of every probe (of all_probes() by default) against the mathforge module mf, in order."""
    return {probe.id: run_probe(probe, mf) for probe in (all_probes() if probes is None else probes)}


def run_probe(probe: Probe, mf: ModuleType) -> Outcome:
    """Attempt the probe's request and classify the outcome; only KeyboardInterrupt and the like escape."""
    try:
        value = probe.run(mf)
    except Exception as exc:  # What the attempt raised is its outcome.
        return Outcome(_exception_status(exc, mf), _describe_exception(exc))
    if isinstance(value, mf.Result):
        status = _result_status(value)
        if status is not None:
            return Outcome(status, _describe_result(value))
        value = value.value
    if _unevaluated(value):
        return Outcome("deferred", f"an unevaluated record ({type(value).__name__})")
    if probe.expect is None:
        return Outcome("ok", describe(value, mf))
    try:
        right = probe.expect(value)
    except Exception as exc:  # An answer that the check cannot even read is no right answer.
        return Outcome("wrong", f"the check raised {_describe_exception(exc)} on {describe(value, mf)}")
    if right is True:
        return Outcome("ok", describe(value, mf))
    return Outcome("wrong", f"the check returned {right!r} on {describe(value, mf)}")


def _result_status(result: Any) -> str | None:
    """The status of a Result, or None when it carries a value."""
    status = result.execution_status.name
    if status == "COMPLETED":
        return "deferred" if result.outcome.name == "NO_CONCLUSION" or result.deferred is not None else None
    if status == "UNSUPPORTED":
        return "unsupported"
    if status == "INVALID_INPUT":
        return "invalid_input"
    code = result.error.code if result.error is not None else ""
    if code == "resource_limit_exceeded":
        return "resource_limit"
    if code == "unsupported_operation":
        return "unsupported"
    return "error"


def _exception_status(exc: Exception, mf: ModuleType) -> str:
    if isinstance(exc, mf.OperationError):  # A value function's failure: its Result tells what happened.
        result = getattr(exc, "result", None)
        status = _result_status(result) if isinstance(result, mf.Result) else None
        return "error" if status is None else status
    if isinstance(exc, mf.ResourceLimitError):
        return "resource_limit"
    if isinstance(exc, mf.InvalidInput):
        return "invalid_input"
    if isinstance(exc, mf.UnsupportedOperation):
        return "unsupported"
    if isinstance(exc, AttributeError) and _missing_api(exc, mf):
        return "unsupported"
    return "error"


def _missing_api(exc: AttributeError, mf: ModuleType) -> bool:
    """Whether the attempt asked MathForge for an attribute that it lacks, rather than code inside it failing.

    The attribute must be missing on a mathforge module or a MathForge value, and the error must not come from code
    in the mathforge package, apart from the __getattr__ that loads its area modules.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return False
    innermost = frames[-1]
    module_wording = _MISSING_MODULE_ATTRIBUTE.fullmatch(str(exc)) is not None
    package = os.path.normcase(os.path.dirname(os.path.abspath(mf.__file__))) + os.sep
    inside = os.path.normcase(os.path.abspath(innermost.filename)).startswith(package)  # No file system access.
    if inside and not (innermost.name == "__getattr__" and module_wording):
        return False
    owner = getattr(exc, "obj", None)
    if isinstance(owner, ModuleType):
        return owner.__name__ == "mathforge" or owner.__name__.startswith("mathforge.")
    if owner is not None:
        return type(owner).__module__.split(".")[0] == "mathforge"
    return module_wording


def _unevaluated(value: Any) -> bool:
    return type(value).__name__ in _UNEVALUATED and type(value).__module__.split(".")[0] == "mathforge"


def _describe_exception(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return _shorten(f"{type(exc).__name__}: {text}" if text else type(exc).__name__)


def _describe_result(result: Any) -> str:
    text = f"Result {result.execution_status.name}/{result.outcome.name}"
    if result.error is not None:
        text += f" {result.error.code}: {' '.join(result.error.message.split())}"
    elif result.deferred is not None:
        text += f" deferred={type(result.deferred).__name__}"
    return _shorten(text)


def describe(value: Any, mf: ModuleType) -> str:
    """A short deterministic one-line description of a value: its text rendering or repr, a big integer's size."""
    if type(value) is int and value.bit_length() > 200:  # str() of a huge int is slow, or refused above 4,300 digits.
        return f"an integer of {value.bit_length()} bits"
    if type(value) in (tuple, list) and len(value) > 8:
        return f"a {type(value).__name__} of {len(value)} items"
    try:
        text = mf.render_text(value)
    except Exception:  # Not every value has a text rendering.
        try:
            text = repr(value)
        except Exception:  # A value whose repr fails is still described.
            text = f"a {type(value).__name__}"
    return _shorten(" ".join(text.split()))


def _shorten(text: str) -> str:
    return text if len(text) <= _DETAIL else text[:_DETAIL - 3] + "..."


# Helpers for writing probes.


def symbols(mf: ModuleType, names: str) -> tuple[Any, ...]:
    """Real MathForge symbols for the space-separated names, with uids fixed by name: every run asks the same."""
    return tuple(mf.Symbol(name, uid=str(uuid.uuid5(_UID_NAMESPACE, name))) for name in names.split())


def planned(mf: ModuleType, name: str, *areas: str) -> Callable[..., Any]:
    """The function of a planned operation whose area the catalogue leaves open.

    It is looked up in the given area modules, then in mathforge, then as the structured function of
    mathforge.operations (whose Result run_probe reads like a value). An operation found nowhere raises
    AttributeError on mathforge.operations, which run_probe counts as unsupported.
    """
    for owner in (*(getattr(mf, area) for area in areas), mf, mf.operations):
        function = getattr(owner, name, None)
        if callable(function):
            return function
    raise AttributeError(f"MathForge has no operation {name!r}", name=name, obj=mf.operations)


def _is(value: Any, name: str) -> bool:
    """Whether value is an instance of the MathForge class of that name."""
    return type(value).__name__ == name and type(value).__module__.split(".")[0] == "mathforge"


def exact(value: Any) -> Fraction:
    """The Fraction of an exact rational answer: an int, a Fraction, a MathForge Rational or a scalar RealExpr.

    Anything else, a bool or a float included, raises TypeError. MathForge's Rational never equals an int or a
    Fraction, so expects compare through this.
    """
    if type(value) is int or type(value) is Fraction:
        return Fraction(value)
    if _is(value, "Rational"):
        return Fraction(value.numerator, value.denominator)
    if _is(value, "RealExpr") and value.op == "scalar":
        return exact(value.value)
    raise TypeError(f"not an exact rational number: {type(value).__name__}")


def exact_tuple(values: Iterable[Any]) -> tuple[Fraction, ...]:
    return tuple(exact(value) for value in values)


def coefficients(polynomial: Any) -> tuple[Fraction, ...]:
    """A MathForge Polynomial's coefficients, constant term first, without trailing zeros (the zero polynomial: ())."""
    terms = list(exact_tuple(polynomial.coefficients))
    while terms and terms[-1] == 0:
        terms.pop()
    return tuple(terms)


def record(value: Any) -> tuple[Any, ...]:
    """The field values of a record (a dataclass instance) in order, or the items of a tuple."""
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(getattr(value, field.name) for field in fields(value))
    if type(value) is tuple:
        return value
    raise TypeError(f"not a record: {type(value).__name__}")


def point_set(value: Any) -> frozenset[Fraction]:
    """The rational points of a finite real set: a FiniteSet, an EmptySet, or an IntervalSet of single points."""
    if _is(value, "EmptySet"):
        return frozenset()
    if _is(value, "FiniteSet"):
        return frozenset(exact(item) for item in value.values)
    if _is(value, "IntervalSet"):
        points = intervals(value)
        if all(lower == upper and left and right for lower, upper, left, right in points):
            return frozenset(lower for lower, _, _, _ in points)
        raise ValueError("an interval set that is not finite")
    raise TypeError(f"not a real set: {type(value).__name__}")


def intervals(value: Any) -> tuple[tuple[Fraction | None, Fraction | None, bool, bool], ...]:
    """An IntervalSet as (lower, upper, left_closed, right_closed) tuples; None is an infinite end."""
    if not _is(value, "IntervalSet"):
        raise TypeError(f"not an interval set: {type(value).__name__}")
    return tuple((None if item.lower is None else exact(item.lower), None if item.upper is None else exact(item.upper),
                  item.left_closed, item.right_closed) for item in value.intervals)


def algebraic_roots(value: Any) -> frozenset[tuple[tuple[int, ...], int]]:
    """A FiniteSet of real algebraic roots as (integer coefficients, real index) pairs."""
    if not _is(value, "FiniteSet") or not all(_is(item, "RealAlgebraicRoot") for item in value.values):
        raise TypeError("not a finite set of real algebraic roots")
    return frozenset((tuple(item.integer_coefficients), item.real_index) for item in value.values)


def signed_sqrt(value: Any) -> tuple[int, Fraction]:
    """(sign, q) of a value written as sqrt(q) or -sqrt(q) with a rational q."""
    if _is(value, "Sqrt"):
        return 1, exact(value.radicand)
    if _is(value, "Mul") and len(value.factors) == 2 and _is(value.factors[1], "Sqrt") and \
            exact(value.factors[0]) == -1:
        return -1, exact(value.factors[1].radicand)
    raise TypeError("not a signed square root of a rational")


class Poly:
    """A polynomial with Fraction coefficients in atoms: the normal form in which expects compare symbolic answers.

    An atom is a symbol's name, a named constant ("pi", "E") or an elementary function applied to a normal form, so
    that cos(x)*x^2 + 2*x*sin(x) and its reorderings normalize alike. It knows no identity between atoms (none of
    sin^2 + cos^2 = 1), so an answer in another closed form of the same function does not compare equal.
    """

    __slots__ = ("terms",)

    def __init__(self, terms: Mapping[tuple[Any, ...], Any] | None = None):
        self.terms = {monomial: Fraction(c) for monomial, c in (terms or {}).items() if c != 0}

    @staticmethod
    def lift(value: Poly | int | Fraction) -> Poly:
        return value if isinstance(value, Poly) else Poly({(): Fraction(value)})

    def key(self) -> tuple[Any, ...]:
        return tuple(sorted(self.terms.items(), key=repr))

    def __add__(self, other: Poly | int | Fraction) -> Poly:
        terms = dict(self.terms)
        for monomial, c in Poly.lift(other).terms.items():
            terms[monomial] = terms.get(monomial, 0) + c
        return Poly(terms)

    __radd__ = __add__

    def __neg__(self) -> Poly:
        return Poly({monomial: -c for monomial, c in self.terms.items()})

    def __sub__(self, other: Poly | int | Fraction) -> Poly:
        return self + -Poly.lift(other)

    def __rsub__(self, other: Poly | int | Fraction) -> Poly:
        return Poly.lift(other) + -self

    def __mul__(self, other: Poly | int | Fraction) -> Poly:
        terms: dict[tuple[Any, ...], Fraction] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in Poly.lift(other).terms.items():
                exponents = dict(m1)
                for atom_key, exponent in m2:
                    exponents[atom_key] = exponents.get(atom_key, 0) + exponent
                monomial = tuple(sorted(exponents.items(), key=repr))
                terms[monomial] = terms.get(monomial, 0) + c1 * c2
        return Poly(terms)

    __rmul__ = __mul__

    def __truediv__(self, other: int | Fraction) -> Poly:
        return self * (1 / Fraction(other))

    def __pow__(self, exponent: int) -> Poly:
        if type(exponent) is not int or exponent < 0:
            raise ValueError("a normal form takes only nonnegative integer powers")
        result, base = Poly.lift(1), self
        while exponent:
            if exponent & 1:
                result = result * base
            base, exponent = base * base, exponent >> 1
        return result

    def constant(self) -> Fraction | None:
        """The value of a constant normal form, else None."""
        if not self.terms:
            return Fraction(0)
        return self.terms[()] if tuple(self.terms) == ((),) else None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Poly, int, Fraction)) or type(other) is bool:
            return NotImplemented
        return self.terms == Poly.lift(other).terms

    def __hash__(self) -> int:
        return hash(frozenset(self.terms.items()))

    def __repr__(self) -> str:
        return f"Poly({self.key()!r})"


def atom(name: str) -> Poly:
    """The normal form of a symbol or a named constant ("pi", "E")."""
    return Poly({((name, 1),): 1})


def apply(function: str, argument: Poly | int | Fraction) -> Poly:
    """The normal form of an elementary function (sin, cos, exp, log, sqrt, ...) of a normal form."""
    return Poly({(((function, Poly.lift(argument).key()), 1),): 1})


def expression_poly(expression: Any) -> Poly:
    """The normal form of a MathForge Expression built from rationals, symbols, sums, products, powers and roots."""
    if type(expression) is int:
        return Poly.lift(expression)
    if _is(expression, "Rational"):
        return Poly.lift(exact(expression))
    if _is(expression, "Symbol"):
        return atom(expression.name)
    if _is(expression, "Add"):
        return sum((expression_poly(term) for term in expression.terms), Poly())
    if _is(expression, "Mul"):
        product = Poly.lift(1)
        for factor in expression.factors:
            product = product * expression_poly(factor)
        return product
    if _is(expression, "Pow"):
        return expression_poly(expression.base) ** expression.exponent
    if _is(expression, "Sqrt"):
        return apply("sqrt", expression_poly(expression.radicand))
    raise TypeError(f"no normal form for {type(expression).__name__}")


def is_monomial_sum(expression: Any) -> bool:
    """Whether an Expression is expanded: a sum of products of a rational and powers of symbols."""
    def monomial(item: Any) -> bool:
        if _is(item, "Rational") or _is(item, "Symbol"):
            return True
        if _is(item, "Pow"):
            return _is(item.base, "Symbol")
        return _is(item, "Mul") and all(monomial(factor) and not _is(factor, "Mul") for factor in item.factors)

    return all(monomial(term) for term in expression.terms) if _is(expression, "Add") else monomial(expression)


def function_poly(expression: Any) -> Poly:
    """The normal form of a RealFunction or a RealExpr; a quotient only by a nonzero constant."""
    if _is(expression, "RealFunction"):
        expression = expression.expression
    if not _is(expression, "RealExpr"):
        raise TypeError(f"not a real expression: {type(expression).__name__}")
    op, args = expression.op, expression.args
    if op == "scalar":
        return Poly.lift(exact(expression.value))
    if op == "symbol":
        return atom(expression.value.name)
    if op in ("pi", "E"):
        return atom(op)
    if op == "add":
        return function_poly(args[0]) + function_poly(args[1])
    if op == "mul":
        return function_poly(args[0]) * function_poly(args[1])
    if op == "div":
        denominator = function_poly(args[1]).constant()
        if not denominator:
            raise ValueError("a quotient by a non-constant or zero")
        return function_poly(args[0]) / denominator
    if op == "pow":
        return function_poly(args[0]) ** expression.value
    if len(args) == 1:
        return apply(op, function_poly(args[0]))
    raise ValueError(f"no normal form for the operation {op!r}")
