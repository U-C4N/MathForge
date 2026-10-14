"""Exact real elementary expressions with explicit, preserved domains.

This module deliberately does not extend the historical total-expression AST.
Raw ``RealExpr`` arithmetic retains its operands, even in ``0*f`` and ``f**0``.
``real_function`` collects their domains *before* simplifying their values.
Domain membership is three-valued; an unknown sign is never an assumption.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from collections.abc import Mapping, Iterable

from .errors import InvalidInput, UnsupportedOperation
from .limits import current_budget, computation, ResourceLimitError
from .model import Rational, Symbol, Reals, Truth, Expression, Add, Mul, Pow, Sqrt
from .algebraic import RealAlgebraicRoot, compare_real


_UNARY = frozenset(("sin", "cos", "tan", "asin", "acos", "atan", "sinh",
                    "cosh", "tanh", "exp", "log", "sqrt", "abs"))


def _tick(n=1):
    budget = current_budget()
    if budget is not None:
        budget.tick(n)


@dataclass(frozen=True)
class RealExpr:
    op: str
    args: tuple[RealExpr, ...] = ()
    value: object = None

    def __post_init__(self):
        _tick()
        object.__setattr__(self, "args", tuple(self.args))
        if any(type(arg) is not RealExpr for arg in self.args):
            raise InvalidInput("Real expression arguments must be RealExpr values")
        if self.op == "scalar":
            if self.args or type(self.value) not in (Rational, RealAlgebraicRoot):
                raise InvalidInput("A real scalar requires an exact real algebraic value")
        elif self.op == "symbol":
            if self.args or type(self.value) is not Symbol or self.value.domain is not Reals or self.value.assumptions:
                raise InvalidInput("Function symbols must be unconstrained real symbols")
        elif self.op in ("pi", "E"):
            if self.args or self.value is not None:
                raise InvalidInput("A named constant has no operands")
        elif self.op == "pow":
            if len(self.args) != 1 or type(self.value) is not int:
                raise InvalidInput("Function powers require an exact signed integer exponent")
            budget = current_budget()
            if budget is not None:
                budget.check_int(self.value)
        elif self.op in ("add", "mul", "div"):
            if len(self.args) != 2 or self.value is not None:
                raise InvalidInput("A binary operation requires two operands")
        elif self.op in _UNARY:
            if len(self.args) != 1 or self.value is not None:
                raise InvalidInput("An elementary function requires one operand")
        else:
            raise InvalidInput("Unknown real expression operation")

    def __add__(self, other): return RealExpr("add", (self, as_real_expr(other)))
    def __radd__(self, other): return as_real_expr(other) + self
    def __mul__(self, other): return RealExpr("mul", (self, as_real_expr(other)))
    def __rmul__(self, other): return as_real_expr(other) * self
    def __neg__(self): return as_real_expr(-1) * self
    def __sub__(self, other): return self + (-as_real_expr(other))
    def __rsub__(self, other): return as_real_expr(other) + (-self)
    def __truediv__(self, other): return RealExpr("div", (self, as_real_expr(other)))
    def __rtruediv__(self, other): return as_real_expr(other) / self
    def __pow__(self, n): return RealExpr("pow", (self,), n)
    def __abs__(self): return RealExpr("abs", (self,))
    def __bool__(self): raise TypeError("A symbolic real expression has no Python truth value")

    @property
    def free_symbols(self):
        if self.op == "symbol": return frozenset((self.value,))
        return frozenset().union(*(arg.free_symbols for arg in self.args))

    def __str__(self):
        if self.op in ("scalar", "symbol"): return str(self.value)
        if self.op in ("pi", "E"): return self.op
        if self.op == "pow": return f"({self.args[0]})**{self.value}"
        if self.op in ("add", "mul", "div"):
            return f"({self.args[0]} {'+' if self.op == 'add' else '*' if self.op == 'mul' else '/'} {self.args[1]})"
        return f"{self.op}({self.args[0]})"


def as_real_expr(value) -> RealExpr:
    if type(value) is RealExpr: return value
    if type(value) is RealFunction:
        raise InvalidInput("Use RealFunction arithmetic to preserve an existing function domain")
    if type(value) is int: value = Rational(value)
    if type(value) in (Rational, RealAlgebraicRoot): return RealExpr("scalar", (), value)
    if type(value) is Symbol: return RealExpr("symbol", (), value)
    if type(value) is Add:
        result = as_real_expr(value.terms[0])
        for term in value.terms[1:]: result = result + as_real_expr(term)
        return result
    if type(value) is Mul:
        result = as_real_expr(value.factors[0])
        for factor in value.factors[1:]: result = result * as_real_expr(factor)
        return result
    if type(value) is Pow: return as_real_expr(value.base) ** value.exponent
    if type(value) is Sqrt: return sqrt(value.radicand)
    from .polynomial import Polynomial
    if type(value) is Polynomial: return as_real_expr(value.to_expression())
    raise InvalidInput("Expected an exact real expression; floats and booleans are not exact inputs")


@dataclass(frozen=True)
class DomainPredicate:
    kind: str
    operands: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "operands", tuple(self.operands))
        if self.kind in ("true", "false"):
            valid = not self.operands
        elif self.kind == "and":
            valid = len(self.operands) >= 2 and all(type(p) is DomainPredicate for p in self.operands)
        elif self.kind in ("positive", "nonnegative", "nonzero"):
            valid = len(self.operands) == 1 and type(self.operands[0]) is RealExpr
        else: valid = False
        if not valid: raise InvalidInput("Invalid domain predicate")

    def __and__(self, other): return conjunction(self, other)

    def contains(self, bindings=None) -> Truth:
        mapping = _bindings(bindings)
        if self.kind == "false": return Truth.FALSE
        binding_guards=tuple(intrinsic_domain(value).contains() for value in mapping.values())
        if Truth.FALSE in binding_guards:return Truth.FALSE
        if Truth.UNKNOWN in binding_guards:return Truth.UNKNOWN
        if self.kind == "true": return Truth.TRUE
        if self.kind == "and":
            answers = tuple(p.contains(mapping) for p in self.operands)
            if Truth.FALSE in answers: return Truth.FALSE
            return Truth.TRUE if all(a is Truth.TRUE for a in answers) else Truth.UNKNOWN
        expr = substitute_expr(self.operands[0], mapping)
        guard=intrinsic_domain(expr)
        if guard.kind!="true":
            guard_truth=guard.contains()
            if guard_truth is not Truth.TRUE:return guard_truth
        sign = sign_of(expr)
        if sign is None: return Truth.UNKNOWN
        accepted = sign > 0 if self.kind == "positive" else sign >= 0 if self.kind == "nonnegative" else sign != 0
        return Truth.TRUE if accepted else Truth.FALSE


TRUE = DomainPredicate("true")
FALSE = DomainPredicate("false")


def conjunction(*predicates):
    flat = []
    for predicate in predicates:
        if type(predicate) is not DomainPredicate: raise InvalidInput("A domain predicate is required")
        if predicate.kind == "false": return FALSE
        if predicate.kind == "true": continue
        flat.extend(predicate.operands if predicate.kind == "and" else (predicate,))
    unique = tuple(sorted(set(flat), key=repr))
    return TRUE if not unique else unique[0] if len(unique) == 1 else DomainPredicate("and", unique)


def condition(kind, expression):
    raw=as_real_expr(expression)
    guards=intrinsic_domain(raw)
    expr = simplify_expr(raw)
    if kind=="nonnegative" and _known_nonnegative(expr):return guards
    if kind=="positive" and expr.op=="add":
        a,b=expr.args
        if (sign_of(a)==1 and _known_nonnegative(b)) or (sign_of(b)==1 and _known_nonnegative(a)):return guards
    if kind=="nonzero" and expr.op in ("abs","sqrt","pow"):
        if expr.op!="pow" or expr.value!=0:
            return conjunction(guards,condition("nonzero",expr.args[0]))
    if kind=="positive" and expr.op=="abs":return conjunction(guards,condition("nonzero",expr.args[0]))
    if kind=="positive" and expr.op=="sqrt":return conjunction(guards,condition("positive",expr.args[0]))
    result = DomainPredicate(kind, (expr,))
    # Some symbolic signs (exp/cosh and their products) are also certified.
    truth = result.contains()
    if truth is Truth.TRUE: return guards
    if truth is Truth.FALSE: return FALSE
    return conjunction(guards,result)


def _known_nonnegative(expr):
    if expr.op=="scalar":return compare_real(expr.value,Rational(0))>=0
    if expr.op in ("abs","sqrt","exp","cosh","pi","E"):return True
    if expr.op=="pow" and expr.value%2==0:return True
    if expr.op in ("add","mul"):return all(_known_nonnegative(a) for a in expr.args)
    return False


def intrinsic_domain(expression):
    expr = as_real_expr(expression)
    predicates = [intrinsic_domain(arg) for arg in expr.args]
    if expr.op == "div": predicates.append(condition("nonzero", expr.args[1]))
    if expr.op == "pow" and expr.value < 0: predicates.append(condition("nonzero", expr.args[0]))
    if expr.op == "log": predicates.append(condition("positive", expr.args[0]))
    if expr.op == "sqrt": predicates.append(condition("nonnegative", expr.args[0]))
    if expr.op == "tan": predicates.append(condition("nonzero", cos(expr.args[0])))
    if expr.op in ("asin", "acos"): predicates.append(condition("nonnegative", 1-expr.args[0]**2))
    return conjunction(*predicates)


def _axes(variables, free=frozenset()):
    if type(variables) is Symbol: variables = (variables,)
    if isinstance(variables, (str, bytes, dict)): raise InvalidInput("An ordered symbol sequence is required")
    values = tuple(variables)
    if any(type(v) is not Symbol or v.domain is not Reals or v.assumptions for v in values):
        raise InvalidInput("Axes must be unconstrained real symbols")
    if len({v.uid for v in values}) != len(values): raise InvalidInput("Function axes must have distinct identities")
    if not free <= set(values): raise InvalidInput("Function axes must include every free symbol")
    return values


@dataclass(frozen=True)
class RealFunction:
    expression: RealExpr
    variables: tuple[Symbol, ...]
    domain: DomainPredicate = TRUE

    def __post_init__(self):
        if type(self.expression) is not RealExpr or type(self.domain) is not DomainPredicate:
            raise InvalidInput("A function requires a real expression and domain predicate")
        axes = _axes(self.variables, self.expression.free_symbols | predicate_symbols(self.domain))
        object.__setattr__(self, "variables", axes)
        # Direct construction must also retain all restrictions of its expression.
        object.__setattr__(self, "domain", conjunction(self.domain, intrinsic_domain(self.expression)))

    def contains(self, bindings=None): return self.domain.contains(bindings)
    def evaluate(self, bindings=None, *, limits=None):
        with computation(limits) as budget:
            budget.inspect(self)
            mapping = _bindings(bindings)
            if not set(self.variables) <= set(mapping): raise InvalidInput("Evaluation requires every function axis")
            if any(value.free_symbols for value in mapping.values()):raise InvalidInput("Evaluation bindings must be closed exact expressions")
            truth = self.domain.contains(mapping)
            if truth is Truth.FALSE: raise InvalidInput("Function is undefined at the supplied point")
            if truth is Truth.UNKNOWN: raise UnsupportedOperation("Function domain membership is unknown")
            value=simplify_expr(substitute_expr(self.expression, mapping))
            if value.free_symbols:raise InvalidInput("Evaluation requires complete bindings")
            return value

    def _binary(self, other, op):
        if type(other) is RealFunction:
            if self.variables != other.variables: raise InvalidInput("Function axis order must match")
            expr, domain = other.expression, other.domain
        else: expr, domain = as_real_expr(other), TRUE
        return real_function(RealExpr(op, (self.expression, expr)), self.variables,
                             domain=conjunction(self.domain, domain))
    def __add__(self, other): return self._binary(other, "add")
    def __radd__(self, other): return self + other
    def __mul__(self, other): return self._binary(other, "mul")
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other if type(other) is RealFunction else -as_real_expr(other))
    def __rsub__(self, other): return -self + other
    def __truediv__(self, other): return self._binary(other, "div")
    def __rtruediv__(self, other):
        return real_function(as_real_expr(other)/self.expression, self.variables, domain=self.domain)
    def __pow__(self, n): return real_function(self.expression**n, self.variables, domain=self.domain)
    def __str__(self): return str(self.expression)


def predicate_symbols(predicate):
    if predicate.kind in ("true", "false"): return frozenset()
    if predicate.kind == "and": return frozenset().union(*(predicate_symbols(p) for p in predicate.operands))
    return predicate.operands[0].free_symbols


def real_function(expression, variables=None, *, domain=None, limits=None) -> RealFunction:
    with computation(limits) as budget:
        budget.inspect(expression)
        if type(expression) is RealFunction:
            axes = expression.variables if variables is None else _axes(variables)
            if axes != expression.variables: raise InvalidInput("Explicit function axes must match")
            return RealFunction(expression.expression, axes, conjunction(expression.domain, domain or TRUE))
        from .rational_function import RationalFunction
        if type(expression) is RationalFunction:
            source = expression
            expr = as_real_expr(source.numerator)/as_real_expr(source.denominator)
            restrictions = conjunction(*(condition("nonzero", p.to_expression()) for p in source.excluded))
            variables = (source.variable,) if variables is None else variables
        else:
            expr, restrictions = as_real_expr(expression), TRUE
        axes = _axes(variables if variables is not None else tuple(sorted(expr.free_symbols, key=lambda s: s.uid)), expr.free_symbols)
        restrictions = conjunction(restrictions, intrinsic_domain(expr), domain or TRUE)
        return RealFunction(simplify_expr(expr), axes, restrictions)


def _bindings(bindings):
    if bindings is None: return {}
    if not isinstance(bindings, Mapping):
        try: bindings = dict(bindings)
        except (ValueError, TypeError) as exc: raise InvalidInput("Bindings must be symbol/value pairs") from exc
    result = {}
    for key, value in bindings.items():
        if type(key) is not Symbol: raise InvalidInput("Binding keys must be Symbol identities")
        result[key] = as_real_expr(value)
    return result


def substitute_expr(expression, bindings):
    _tick()
    expr = as_real_expr(expression)
    if expr.op == "symbol": return as_real_expr(bindings.get(expr.value, expr))
    return RealExpr(expr.op, tuple(substitute_expr(a, bindings) for a in expr.args), expr.value)


def substitute_predicate(predicate, bindings):
    if predicate.kind in ("true", "false"): return predicate
    if predicate.kind == "and": return conjunction(*(substitute_predicate(p, bindings) for p in predicate.operands))
    return condition(predicate.kind, substitute_expr(predicate.operands[0], bindings))


def _unary(name, value):
    if type(value) is RealFunction:
        return real_function(RealExpr(name, (value.expression,)), value.variables, domain=value.domain)
    return RealExpr(name, (as_real_expr(value),))


def sin(value): return _unary("sin", value)
def cos(value): return _unary("cos", value)
def tan(value): return _unary("tan", value)
def asin(value): return _unary("asin", value)
def acos(value): return _unary("acos", value)
def atan(value): return _unary("atan", value)
def sinh(value): return _unary("sinh", value)
def cosh(value): return _unary("cosh", value)
def tanh(value): return _unary("tanh", value)
def exp(value): return _unary("exp", value)
def log(value): return _unary("log", value)
def sqrt(value): return _unary("sqrt", value)
def abs(value): return _unary("abs", value)
pi = RealExpr("pi")
E = RealExpr("E")
ZERO = RealExpr("scalar", (), Rational(0))
ONE = RealExpr("scalar", (), Rational(1))


def _rational(expr): return expr.value if expr.op == "scalar" and type(expr.value) is Rational else None


def _pi_multiple(expr):
    if expr.op == "pi": return Rational(1)
    if expr.op == "mul":
        a, b = expr.args
        if _rational(a) is not None:
            q = _pi_multiple(b)
            if q is not None: return a.value*q
        if _rational(b) is not None:
            q = _pi_multiple(a)
            if q is not None: return b.value*q
    if expr.op == "div" and _rational(expr.args[1]) not in (None, Rational(0)):
        q = _pi_multiple(expr.args[0])
        if q is not None: return q/expr.args[1].value
    if expr == ZERO: return Rational(0)
    return None


def _special_trig(op, argument):
    q = _pi_multiple(argument)
    if q is None: return None
    q = Rational(q.numerator % (2*q.denominator), q.denominator)
    if op == "cos": q = q+Rational(1, 2)
    if op in ("sin", "cos"):
        q = Rational(q.numerator % (2*q.denominator), q.denominator)
        sign = 1
        if q > 1: q, sign = q-1, -1
        if q > Rational(1, 2): q = 1-q
        if q == Rational(0): return ZERO
        if q == Rational(1, 6): return as_real_expr(Rational(sign, 2))
        if q == Rational(1, 4): return as_real_expr(sign)*sqrt(2)/2
        if q == Rational(1, 3): return as_real_expr(sign)*sqrt(3)/2
        if q == Rational(1, 2): return as_real_expr(sign)
    if op == "tan":
        s, c = _special_trig("sin", argument), _special_trig("cos", argument)
        if s is not None and c is not None and c != ZERO: return s/c
    return None


def simplify_expr(expression):
    """Value-only local normalization. Callers must retain the source domain."""
    _tick()
    expr = as_real_expr(expression)
    args = tuple(simplify_expr(a) for a in expr.args)
    if not args: return expr
    if expr.op in ("add", "mul", "div"):
        a, b = args
        ar, br = _rational(a), _rational(b)
        if ar is not None and br is not None:
            if expr.op == "add": return as_real_expr(ar+br)
            if expr.op == "mul": return as_real_expr(ar*br)
            if br != Rational(0): return as_real_expr(ar/br)
        if expr.op == "add":
            if a == ZERO: return b
            if b == ZERO: return a
            if a == b: return RealExpr("mul", (as_real_expr(2), a))
        if expr.op == "mul":
            if a == ZERO or b == ZERO: return ZERO
            if a == ONE: return b
            if b == ONE: return a
            if a == b: return simplify_expr(a**2)
        if expr.op == "div":
            if b == ONE: return a
            if a == ZERO and b != ZERO: return ZERO
            if a == b and a != ZERO: return ONE
    elif expr.op == "pow":
        a, n = args[0], expr.value
        if n == 0: return ONE
        if n == 1: return a
        ar = _rational(a)
        if ar is not None and (ar != Rational(0) or n >= 0):
            budget = current_budget()
            if budget is not None:
                if builtins.abs(n) > budget.limits.max_work: raise ResourceLimitError("Power exceeds computation budget")
                if max(ar.numerator.bit_length(), ar.denominator.bit_length())*builtins.abs(n) > budget.limits.max_integer_bits:
                    raise ResourceLimitError("Power exceeds integer budget")
                budget.tick(builtins.abs(n))
            if n < 0: ar, n = Rational(ar.denominator, ar.numerator), -n
            return as_real_expr(Rational(ar.numerator**n, ar.denominator**n))
        if n == 2 and a.op in ("sqrt", "abs"):
            return a.args[0] if a.op == "sqrt" else simplify_expr(a.args[0]**2)
    elif expr.op in _UNARY:
        a = args[0]
        if expr.op in ("sin", "cos", "tan"):
            special = _special_trig(expr.op, a)
            if special is not None: return simplify_expr(special)
        if a == ZERO and expr.op in ("sinh", "tanh", "asin", "atan", "abs", "sqrt"): return ZERO
        if a == ZERO and expr.op in ("exp", "cosh"): return ONE
        if expr.op == "exp" and a == ONE: return E
        if expr.op == "exp" and a.op == "log": return a.args[0]
        if expr.op == "log" and a.op == "exp": return a.args[0]
        if expr.op == "log" and a == ONE: return ZERO
        if expr.op == "log" and a == E: return ONE
        if expr.op == "acos" and a == ONE: return ZERO
        if expr.op == "acos" and a == ZERO: return simplify_expr(pi/2)
        if expr.op == "asin" and a == ONE: return simplify_expr(pi/2)
        if expr.op == "atan" and a == ONE: return simplify_expr(pi/4)
        if expr.op == "abs" and _rational(a) is not None: return as_real_expr(builtins.abs(a.value))
        if expr.op == "sqrt" and _rational(a) is not None and a.value >= 0:
            from .model import sqrt as old_sqrt
            old = old_sqrt(a.value)
            if type(old) is Rational: return as_real_expr(old)
        if expr.op == "sqrt" and a.op == "pow" and a.value == 2: return simplify_expr(abs(a.args[0]))
        if expr.op == "abs" and a.op == "abs": return a
    return RealExpr(expr.op, args, expr.value)


def _legacy_closed(expr):
    """Convert the algebraic subset only; never approximate transcendental atoms."""
    if expr.op == "scalar": return expr.value
    if expr.op == "sqrt":
        from .model import sqrt as old_sqrt
        a = _legacy_closed(expr.args[0])
        if type(a) is Rational: return old_sqrt(a)
        return None
    if expr.op not in ("add", "mul", "div", "pow", "abs"): return None
    args = tuple(_legacy_closed(a) for a in expr.args)
    if any(a is None for a in args): return None
    # Existing public wrappers are intentionally not used from predicate checks.
    if expr.op == "add": return args[0]+args[1]
    if expr.op == "mul": return args[0]*args[1]
    if expr.op == "pow" and expr.value >= 0: return args[0]**expr.value
    if expr.op == "div" and type(args[1]) is Rational and args[1] != Rational(0): return args[0]/args[1]
    if expr.op == "abs":
        sign = compare_real(args[0], Rational(0))
        return args[0] if sign >= 0 else -args[0]
    return None


def sign_of(expression):
    expr = simplify_expr(expression)
    if expr.op == "scalar": return compare_real(expr.value, Rational(0))
    if expr.op in ("pi", "E", "exp", "cosh"): return 1
    if expr.op in ("abs", "sqrt"):
        sign = sign_of(expr.args[0])
        if sign is not None and (expr.op == "abs" or sign >= 0): return 0 if sign == 0 else 1
    if expr.op in ("mul", "div"):
        a, b = (sign_of(x) for x in expr.args)
        if a is not None and b is not None and (expr.op != "div" or b != 0): return a*b
    if expr.op == "add":
        a, b = (sign_of(x) for x in expr.args)
        if a == 0: return b
        if b == 0: return a
        if a is not None and a == b: return a
    if expr.op == "pow":
        a = sign_of(expr.args[0])
        if a is not None and (a != 0 or expr.value >= 0): return 0 if a==0 else 1 if expr.value%2==0 else a
    if expr.op in ("sinh", "tanh", "atan", "asin"):
        return sign_of(expr.args[0])
    if expr.op == "log" and _rational(expr.args[0]) is not None:
        argument=expr.args[0].value
        if argument>0:return (argument>1)-(argument<1)
    if not expr.free_symbols:
        try:
            legacy = _legacy_closed(expr)
            if legacy is not None: return compare_real(legacy, Rational(0))
        except (UnsupportedOperation, InvalidInput, ArithmeticError): pass
    return None


def parse_function(text, *, symbols, variables=None, limits=None):
    """Parse a bounded arithmetic grammar without executing Python code."""
    if type(text) is not str or not text.strip(): raise InvalidInput("Function text must be nonempty")
    if len(text) > 100_000: raise ResourceLimitError("Function text exceeds parser budget")
    if not isinstance(symbols, Mapping) or any(type(k) is not str or type(v) is not Symbol or k != v.name for k, v in symbols.items()):
        raise InvalidInput("Explicit names must map to matching Symbol objects")
    if set(symbols) & (_UNARY | {"pi", "E"}): raise InvalidInput("Function names and constants are reserved")
    source = text.replace("^", "**")
    try: tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc: raise InvalidInput("Invalid function syntax") from exc
    with computation(limits) as budget:
        def visit(node, depth=0):
            budget.tick(); budget.nodes()
            if depth > budget.limits.max_depth: raise ResourceLimitError("Function parse depth exceeded")
            if type(node) is ast.Constant and type(node.value) in (int, float):
                token = ast.get_source_segment(source, node)
                if token is None or len(token) > 20_000: raise ResourceLimitError("Numeric literal exceeds budget")
                try:
                    value = Decimal(token.replace("_", ""))
                    if not value.is_finite() or builtins.abs(value.adjusted()) > 20_000: raise InvalidInput("Numeric literal is too large")
                    a, b = value.as_integer_ratio()
                    return as_real_expr(Rational(a, b))
                except (InvalidOperation, ValueError) as exc: raise InvalidInput("Invalid exact numeric literal") from exc
            if type(node) is ast.Name:
                if node.id == "pi": return pi
                if node.id == "E": return E
                if node.id not in symbols: raise InvalidInput("Undeclared function symbol")
                return as_real_expr(symbols[node.id])
            if type(node) is ast.UnaryOp and type(node.op) in (ast.UAdd, ast.USub):
                arg = visit(node.operand, depth+1)
                return arg if type(node.op) is ast.UAdd else -arg
            if type(node) is ast.BinOp and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
                a, b = visit(node.left, depth+1), visit(node.right, depth+1)
                if type(node.op) is ast.Add: return a+b
                if type(node.op) is ast.Sub: return a-b
                if type(node.op) is ast.Mult: return a*b
                if type(node.op) is ast.Div: return a/b
                if b.free_symbols or intrinsic_domain(b).contains() is not Truth.TRUE:
                    raise UnsupportedOperation("An integer exponent must be a defined closed exact expression")
                exponent = _rational(simplify_expr(b))
                if exponent is None or exponent.denominator != 1: raise UnsupportedOperation("Only signed integer function powers are supported")
                return a**exponent.numerator
            if type(node) is ast.Call and type(node.func) is ast.Name and node.func.id in _UNARY and len(node.args) == 1 and not node.keywords:
                return _unary(node.func.id, visit(node.args[0], depth+1))
            raise InvalidInput("Unsupported function syntax")
        expr = visit(tree.body)
        axes = tuple(symbols.values()) if variables is None else variables
        return real_function(expr, axes)


@dataclass(frozen=True)
class UnevaluatedIntegral:
    expression: RealFunction
    variable: Symbol
    lower: object = None
    upper: object = None

    def __post_init__(self):
        if type(self.expression) is not RealFunction or type(self.variable) is not Symbol or self.variable not in self.expression.variables:
            raise InvalidInput("An unevaluated integral requires a function and its axis")
        if (self.lower is None) != (self.upper is None): raise InvalidInput("Both integral bounds must be provided")


@dataclass(frozen=True)
class UnevaluatedLimit:
    expression: RealFunction
    variable: Symbol
    point: object
    side: str = "both"

    def __post_init__(self):
        if type(self.expression) is not RealFunction or type(self.variable) is not Symbol or self.variable not in self.expression.variables:
            raise InvalidInput("An unevaluated limit requires a function and its axis")
        if self.side not in ("left", "right", "both"): raise InvalidInput("Invalid limit side")


TYPES = {"real_expr": RealExpr, "domain_predicate": DomainPredicate, "real_function": RealFunction,
         "unevaluated_integral": UnevaluatedIntegral, "unevaluated_limit": UnevaluatedLimit}


def differentiate(expression, variable, n=1, *, limits=None):
    from .values_v7 import function_differentiate as impl
    return impl(expression, variable, n, limits=limits)


def partial_diff(expression, variable, n=1, *, limits=None): return differentiate(expression, variable, n, limits=limits)


def gradient(expression, variables, *, limits=None):
    from .values_v7 import function_gradient as impl
    return impl(expression, variables, limits=limits)


def jacobian(expressions, variables, *, limits=None):
    from .values_v7 import function_jacobian as impl
    return impl(expressions, variables, limits=limits)


def hessian(expression, variables, *, limits=None):
    from .values_v7 import function_hessian as impl
    return impl(expression, variables, limits=limits)


def taylor(expression, variable, point=0, order=5, *, limits=None):
    from .values_v7 import taylor as impl
    return impl(expression, variable, point, order, limits=limits)
