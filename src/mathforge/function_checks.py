"""Independent coefficient arithmetic for elementary-function certificates.

The checker propagates truncated jets, rather than invoking differentiation or
integration producers. Expression identities are checked by cross-multiplying
sparse polynomials in opaque elementary atoms; no numerical samples are used.
"""
from .errors import UnsupportedOperation
from .model import Rational, Truth
from .functions import (RealExpr, ZERO, ONE, as_real_expr, simplify_expr, _tick,
                        sin, cos, sinh, cosh, exp, log, sqrt, abs, asin, acos, atan,intrinsic_domain)


def _sum(items):
    result = ZERO
    for item in items: result = simplify_expr(result+item)
    return result


def _add(a, b): return tuple(simplify_expr(x+y) for x, y in zip(a, b))
def _scale(a, q): return tuple(simplify_expr(q*x) for x in a)


def _mul(a, b):
    return tuple(_sum(a[k]*b[i-k] for k in range(i+1)) for i in range(len(a)))


def _inv(a):
    out = [simplify_expr(ONE/a[0])]
    for n in range(1, len(a)):
        out.append(simplify_expr(-_sum(a[k]*out[n-k] for k in range(1, n+1))/a[0]))
    return tuple(out)


def _power(a, n):
    one = (ONE,)+(ZERO,)*(len(a)-1)
    if n < 0: return _power(_inv(a), -n)
    out = one
    while n:
        _tick()
        if n & 1: out = _mul(out, a)
        n >>= 1
        if n: a = _mul(a, a)
    return out


def _sincos(a, hyperbolic=False):
    s = [simplify_expr(sinh(a[0]) if hyperbolic else sin(a[0]))]
    c = [simplify_expr(cosh(a[0]) if hyperbolic else cos(a[0]))]
    for n in range(1, len(a)):
        s.append(simplify_expr(_sum(k*a[k]*c[n-k] for k in range(1, n+1))/n))
        c.append(simplify_expr(_sum(k*a[k]*s[n-k] for k in range(1, n+1))*(1 if hyperbolic else -1)/n))
    return tuple(s), tuple(c)


def _root(a):
    out = [simplify_expr(sqrt(a[0]))]
    for n in range(1, len(a)):
        out.append(simplify_expr((a[n]-_sum(out[k]*out[n-k] for k in range(1,n)))/(2*out[0])))
    return tuple(out)


def jet(expression, variable, point, order):
    """Formal truncated Taylor coefficients, valid where smoothness is checked."""
    _tick()
    expr = simplify_expr(expression)
    point = as_real_expr(point)
    if variable not in expr.free_symbols: return (expr,)+(ZERO,)*order
    if expr.op == "symbol": return (point, ONE)+(ZERO,)*(order-1) if order else (point,)
    parts = tuple(jet(a, variable, point, order) for a in expr.args)
    op = expr.op
    if op == "add": return _add(*parts)
    if op == "mul": return _mul(*parts)
    if op == "div": return _mul(parts[0], _inv(parts[1]))
    if op == "pow": return _power(parts[0], expr.value)
    a = parts[0]
    if op == "exp":
        out = [simplify_expr(exp(a[0]))]
        for n in range(1, order+1): out.append(simplify_expr(_sum(k*a[k]*out[n-k] for k in range(1,n+1))/n))
        return tuple(out)
    if op in ("sin", "cos", "tan", "sinh", "cosh", "tanh"):
        s, c = _sincos(a, op in ("sinh", "cosh", "tanh"))
        return s if op in ("sin", "sinh") else c if op in ("cos", "cosh") else _mul(s, _inv(c))
    if op == "sqrt": return _root(a)
    if op == "abs":
        # abs(u)^2=u^2; the nonzero-domain witness selects the real branch.
        out = list(_root(_mul(a,a)))
        out[0] = simplify_expr(abs(a[0]))
        # _root already uses sqrt(a0*a0), which normalizes to abs(a0).
        return tuple(out)
    if op in ("log", "asin", "acos", "atan"):
        derivative = tuple(simplify_expr((k+1)*a[k+1]) for k in range(order))+(ZERO,)
        one = (ONE,)+(ZERO,)*order
        if op == "log": denominator, first = a, log(a[0])
        elif op == "atan": denominator, first = _add(one, _mul(a,a)), atan(a[0])
        else:
            denominator = _root(_add(one, _scale(_mul(a,a), -1)))
            first = asin(a[0]) if op == "asin" else acos(a[0])
            if op == "acos": derivative = _scale(derivative,-1)
        quotient = _mul(derivative, _inv(denominator))
        return (simplify_expr(first),)+tuple(simplify_expr(quotient[n-1]/n) for n in range(1,order+1))
    raise UnsupportedOperation("No independent jet rule for this function")


_QZERO, _QONE = Rational(0), Rational(1)


def _padd(a,b,sign=1):
    out = dict(a)
    for monomial, coefficient in b.items():
        _tick()
        value = out.get(monomial,_QZERO)+sign*coefficient
        if value == _QZERO: out.pop(monomial,None)
        else: out[monomial] = value
    return out


def _pmul(a,b):
    out = {}
    for am, av in a.items():
        for bm,bv in b.items():
            _tick()
            powers = dict(am)
            for atom, power in bm: powers[atom] = powers.get(atom,0)+power
            monomial = tuple(sorted(powers.items(),key=lambda p:repr(p[0])))
            value = out.get(monomial,_QZERO)+av*bv
            if value == _QZERO: out.pop(monomial,None)
            else: out[monomial] = value
    return out


def _ppow(a,n):
    out = {():_QONE}
    while n:
        _tick()
        if n&1: out=_pmul(out,a)
        n>>=1
        if n:a=_pmul(a,a)
    return out


def _fraction(expr,aliases=None):
    _tick()
    expr=simplify_expr(expr)
    one={():_QONE}
    if expr.op in ("tan","tanh"):
        return _fraction((sin(expr.args[0])/cos(expr.args[0])) if expr.op=="tan" else (sinh(expr.args[0])/cosh(expr.args[0])),aliases)
    if expr.op=="scalar" and type(expr.value) is Rational:
        return ({} if expr.value==_QZERO else {():expr.value}),one
    if expr.op in ("add","mul","div"):
        a,b=_fraction(expr.args[0],aliases);c,d=_fraction(expr.args[1],aliases)
        if expr.op=="add":return _padd(_pmul(a,d),_pmul(c,b)),_pmul(b,d)
        if expr.op=="mul":return _pmul(a,c),_pmul(b,d)
        return _pmul(a,d),_pmul(b,c)
    if expr.op=="pow":
        a,b=_fraction(expr.args[0],aliases);n=expr.value
        if n<0:a,b,n=b,a,-n
        return _ppow(a,n),_ppow(b,n)
    return {(((aliases or {}).get(expr,expr),1),):_QONE},one


def _reduce_squares(poly,aliases=None):
    """Reduce real sqrt/abs square relations with polynomial radicands."""
    pending=list(poly.items());out={}
    while pending:
        _tick()
        monomial,coefficient=pending.pop()
        selected=None
        for atom,power in monomial:
            if atom.op in ("sin","sinh") and power>=2:
                other=RealExpr("cos" if atom.op=="sin" else "cosh",atom.args)
                other=(aliases or {}).get(other,other)
                replacement=_padd({():_QONE},{((other,2),):_QONE},-1) if atom.op=="sin" else _padd({((other,2),):_QONE},{():_QONE},-1)
                selected=(atom,power,replacement);break
            if atom.op in ("sqrt","abs") and power>=2:
                a,b=_fraction(atom.args[0] if atom.op=="sqrt" else atom.args[0]**2,aliases)
                if b=={():_QONE}: selected=(atom,power,a);break
        if selected is None:
            out=_padd(out,{monomial:coefficient});continue
        atom,power,replacement=selected
        residual=tuple((a,p-2 if a==atom else p) for a,p in monomial if a!=atom or p>2)
        pending.extend(_pmul({residual:coefficient},replacement).items())
    return out


def _aliases(left,right):
    atoms=[];pending=[as_real_expr(left),as_real_expr(right)];seen=set()
    while pending:
        atom=pending.pop()
        if atom in seen:continue
        seen.add(atom);pending.extend(atom.args)
        if atom.op not in ("add","mul","div","pow") and atom.args:atoms.append(atom)
    aliases={};representatives=[]
    for atom in sorted(atoms,key=repr):
        for candidate in representatives:
            _tick()
            if atom.op==candidate.op and len(atom.args)==len(candidate.args) and all(equivalent(a,b) for a,b in zip(atom.args,candidate.args)):
                aliases[atom]=candidate;break
        else:representatives.append(atom)
    return aliases


def equivalent(left,right):
    # Normalization may erase undefined operands (0*log(-1), for example).
    # Closed values must denote certified real numbers before comparison.
    for expression in (left,right):
        expression=as_real_expr(expression)
        domain=intrinsic_domain(expression)
        if domain.kind=="false" or not expression.free_symbols and domain.contains() is not Truth.TRUE:return False
    if simplify_expr(left)==simplify_expr(right):return True
    aliases=_aliases(left,right)
    a,b=_fraction(left,aliases);c,d=_fraction(right,aliases)
    if not b or not d:return False
    return not _reduce_squares(_padd(_pmul(a,d),_pmul(c,b),-1),aliases)
