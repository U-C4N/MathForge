"""Hermite reduction and Lazard--Rioboo--Trager logarithmic primitives.

LogRootSum(q,h) means sum_{q(t)=0} t log(h(t,x)), modulo locally
constant choices of logarithm. It is a formal primitive, never a pointwise
principal-complex-log evaluator. A separate real elementary conversion covers
linear/quadratic denominator factors and proper finite integrals.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidInput, UnsupportedOperation
from .limits import computation, current_budget
from .model import Rational, Symbol
from .polynomial import Polynomial
from .rational_function import RationalFunction, as_rational_function, validate_variable
from .functions import DomainPredicate


def _tick(n=1):
    current_budget().tick(n)


def _zero(value):
    if isinstance(value, _K): return value.p.is_zero
    if isinstance(value, _F): return value.n.is_zero
    return value == Rational(0) if type(value) is Rational else value == 0


def _trim(values):
    values = list(values)
    while values and _zero(values[-1]): values.pop()
    return tuple(values)


def _add(a, b):
    if not a: return b
    if not b: return a
    zero = a[0]*0
    return _trim((a[i] if i < len(a) else zero) + (b[i] if i < len(b) else zero)
                 for i in range(max(len(a), len(b))))


def _neg(a): return tuple(-x for x in a)
def _sub(a, b): return _add(a, _neg(b))


def _mul(a, b):
    if not a or not b: return ()
    current_budget().nodes(len(a)+len(b)-1)
    values = [a[0]*0 for _ in range(len(a)+len(b)-1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            _tick(); values[i+j] = values[i+j] + x*y
    return _trim(values)


def _divmod(a, b):
    if not b: raise ZeroDivisionError("Zero polynomial divisor")
    zero = b[0]*0
    q = [zero for _ in range(max(0, len(a)-len(b)+1))]
    r = a
    while r and len(r) >= len(b):
        _tick(); k = len(r)-len(b); factor = r[-1]/b[-1]
        q[k] = q[k]+factor
        r = _sub(r, (zero,)*k + tuple(factor*v for v in b))
    return _trim(q), r


def _xgcd(a, b, one):
    s, u, t, v = (one,), (), (), (one,)
    while b:
        _tick(); q, r = _divmod(a,b)
        a,b,s,u,t,v = b,r,u,_sub(s,_mul(q,u)),v,_sub(t,_mul(q,v))
    if not a: return (), (), ()
    scale = a[-1]
    return (tuple(x/scale for x in a), tuple(x/scale for x in s), tuple(x/scale for x in t))


class _K:
    """A coefficient in Q[t]/q for the producer's polynomial Euclidean step."""
    def __init__(self, p, q):
        self.q = q
        self.p = (p if type(p) is Polynomial else Polynomial((p,),q.variable)) % q
    def _coerce(self, value): return value if isinstance(value,_K) else _K(value,self.q)
    def __add__(self, other): return _K(self.p+self._coerce(other).p,self.q)
    __radd__ = __add__
    def __neg__(self): return _K(-self.p,self.q)
    def __sub__(self, other): return self+-self._coerce(other)
    def __mul__(self, other): return _K(self.p*self._coerce(other).p,self.q)
    __rmul__ = __mul__
    def __truediv__(self, other):
        other = self._coerce(other)
        g,s,_ = other.p.xgcd(self.q)
        if g.degree != 0 or g.is_zero: raise InvalidInput("Noninvertible residue coefficient")
        return _K(self.p*s,self.q)


class _F:
    """Q(x) coefficients used only by the independent trace checker.

    These are formal field elements: no evaluation domain is silently inferred
    from intermediate denominators. The public source keeps its own exclusions.
    """
    def __init__(self, n, d=None):
        self.n=n
        self.d=Polynomial((1,),n.variable) if d is None else d
        if self.d.is_zero: raise ZeroDivisionError("Zero formal denominator")
        g=self.n.gcd(self.d)
        self.n,self.d=self.n.exact_div(g),self.d.exact_div(g)
        scale=self.d.leading_coefficient
        self.n,self.d=self.n/scale,self.d/scale
    def _coerce(self, x):
        return x if isinstance(x,_F) else _F(Polynomial((x,),self.n.variable))
    def __add__(self, x):
        x=self._coerce(x); return _F(self.n*x.d+x.n*self.d,self.d*x.d)
    __radd__=__add__
    def __neg__(self): return _F(-self.n,self.d)
    def __sub__(self,x): return self+-self._coerce(x)
    def __mul__(self,x):
        x=self._coerce(x); return _F(self.n*x.n,self.d*x.d)
    __rmul__=__mul__
    def __truediv__(self,x):
        x=self._coerce(x); return _F(self.n*x.d,self.d*x.n)


@dataclass(frozen=True)
class LogRootSum:
    residue_polynomial: tuple[Rational, ...]
    logarithm_coefficients: tuple[tuple[Rational, ...], ...]

    def __post_init__(self):
        q,h=self.residue_polynomial,self.logarithm_coefficients
        with computation() as budget:
            budget.inspect((q,h))
        if (type(q) is not tuple or len(q)<2 or any(type(x) is not Rational for x in q)
                or q[-1] != Rational(1) or type(h) is not tuple or len(h)<2
                or any(type(row) is not tuple or not row or len(row)>=len(q)
                       or any(type(c) is not Rational for c in row)
                       or (len(row)>1 and row[-1]==Rational(0)) for row in h)):
            raise InvalidInput("Invalid logarithmic root-sum data")
        if h[-1] != (Rational(1),):
            raise InvalidInput("A logarithm polynomial must be monic in x")


@dataclass(frozen=True)
class RationalPrimitive:
    source: RationalFunction
    rational_part: RationalFunction
    log_terms: tuple[LogRootSum, ...]
    elementary_terms: tuple[tuple[Polynomial, Polynomial], ...]
    domain: DomainPredicate | None = None

    def __post_init__(self):
        with computation() as budget:
            budget.inspect((self.source,self.rational_part,self.log_terms,self.elementary_terms,self.domain))
        if type(self.source) is not RationalFunction or type(self.rational_part) is not RationalFunction:
            raise InvalidInput("Rational primitive source and rational part are required")
        if self.source.variable != self.rational_part.variable:
            raise InvalidInput("Primitive variable mismatch")
        if type(self.log_terms) is not tuple or any(type(t) is not LogRootSum for t in self.log_terms):
            raise InvalidInput("Logarithmic terms must be an immutable tuple")
        if type(self.elementary_terms) is not tuple or any(type(t) is not tuple or len(t)!=2
                or any(type(p) is not Polynomial or p.variable != self.source.variable for p in t)
                or t[1].degree<1 or t[0].degree>=t[1].degree for t in self.elementary_terms):
            raise InvalidInput("Invalid square-free residual terms")
        from .functions import real_function, conjunction
        source_domain = real_function(self.source).domain
        if self.domain is not None and type(self.domain) is not DomainPredicate:
            raise InvalidInput("A rational primitive requires a mathematical domain")
        object.__setattr__(self, "domain", source_domain if self.domain is None
                           else conjunction(source_domain, self.domain))

    @property
    def variable(self): return self.source.variable

    def as_function(self):
        """Return a real primitive when residual factors have degree at most two."""
        from . import functions as f
        with computation():
            if any(d.degree>2 for _,d in self.elementary_terms):
                raise UnsupportedOperation("This formal root sum has no pointwise evaluator")
            result=f.real_function(self.rational_part,(self.variable,))
            x=f.as_real_expr(self.variable)
            for b,d in self.elementary_terms:
                dx=f.as_real_expr(d)
                if d.degree==1:
                    term=f.as_real_expr(b.coefficients[0])*f.log(f.abs(dx))
                else:
                    p,q=d.coefficients[1],d.coefficients[0]
                    a=b.coefficients[1]/2 if b.degree==1 else Rational(0)
                    c=b.coefficients[0]-a*p
                    delta=4*q-p*p
                    term=f.as_real_expr(a)*f.log(f.abs(dx))
                    if c != Rational(0):
                        if delta>0:
                            s=f.sqrt(delta)
                            term=term+f.as_real_expr(c*2)/s*f.atan((2*x+p)/s)
                        elif delta<0:
                            s=f.sqrt(-delta)
                            term=term+f.as_real_expr(c)/s*f.log(f.abs((2*x+p-s)/(2*x+p+s)))
                        else:
                            raise InvalidInput("Residual denominator is not square-free")
                result=result+f.real_function(term,(self.variable,))
            return f.RealFunction(result.expression,(self.variable,),
                                  f.conjunction(result.domain,self.domain))


TYPES={"log_root_sum":LogRootSum,"rational_primitive":RationalPrimitive}
ARGUMENTS={"rational_antiderivative":{"expression","variable"},
           "rational_integrate":{"expression","variable","lower","upper"}}


def _source(expression,variable):
    validate_variable(variable)
    result=as_rational_function(expression,variable)
    if result.variable!=variable: raise InvalidInput("Variable mismatch")
    return result


def _lrt(b,d):
    """Interpolate the residue resultant, then compute subresultant gcds.

    Euclidean gcd over each irreducible residue field gives the monic h of
    Lazard--Rioboo--Trager without requiring explicit complex root isolation.
    """
    from .polynomial_exact import resultant, factor_with_evidence
    x=d.variable; one=Polynomial((1,),x); t=Polynomial((0,1),x)
    samples=[resultant(d,b-d.derivative()*k) for k in range(d.degree+1)]
    q=Polynomial((0,),x)
    for k,y in enumerate(samples):
        basis=one; divisor=1
        for j in range(d.degree+1):
            if j!=k:
                basis=basis*(t-j); divisor*=k-j; _tick()
        q=q+basis*(y/divisor)
    factors,_=factor_with_evidence(q)
    terms=[]
    for factor in factors.factors:
        modulus=factor.polynomial
        d_k=tuple(_K(c,modulus) for c in d.coefficients)
        residue=_K(t,modulus)
        bd=tuple(_K(b.coefficients[i] if i<len(b.coefficients) else 0,modulus)
                 - residue*(d.derivative().coefficients[i] if i<len(d.derivative().coefficients) else 0)
                 for i in range(max(len(b.coefficients),len(d.derivative().coefficients))))
        g,_,_=_xgcd(_trim(d_k),_trim(bd),_K(1,modulus))
        if len(g)<2: raise ArithmeticError("Residue resultant did not select a logarithmic factor")
        terms.append(LogRootSum(modulus.coefficients,tuple(v.p.coefficients for v in g)))
    return tuple(terms)


def _primitive(source,domain=None):
    from .exact_extensions import _apart
    parts,_=_apart(source)
    x=source.variable
    p=Polynomial((0,)+tuple(c/(i+1) for i,c in enumerate(parts.polynomial.coefficients)),x)
    rational=as_rational_function(p,x)
    residuals={}
    for entry in parts.terms:
        b,d,k=entry.numerator,entry.factor,entry.power
        while k>1:
            _tick()
            _,inverse,_=d.derivative().xgcd(d)
            a=(-b*inverse/(k-1))%d
            rational=rational+RationalFunction._normalize(a,d**(k-1),source.excluded)
            b=(b-a.derivative()*d+(k-1)*a*d.derivative()).exact_div(d)
            k-=1
        residuals[d]=residuals.get(d,Polynomial((0,),x))+b
    terms=tuple((b,d) for d,b in residuals.items() if not b.is_zero)
    logs=tuple(t for b,d in terms for t in _lrt(b,d))
    rational=RationalFunction._normalize(rational.numerator,rational.denominator,source.excluded)
    return RationalPrimitive(source,rational,logs,terms,domain)


def _trace_derivative(term,variable):
    """Independent Q(x)[t]/q trace of t*h_x/h; no LRT or Hermite calls."""
    q=Polynomial(term.residue_polynomial,variable)
    if q.gcd(q.derivative()).degree!=0:
        raise InvalidInput("Residue polynomial must be square-free")
    coeff=lambda c:_F(Polynomial((c,),variable))
    mod=tuple(coeff(c) for c in q.coefficients)
    width=max(len(row) for row in term.logarithm_coefficients)
    h=tuple(_F(Polynomial(tuple(row[j] if j<len(row) else Rational(0)
                     for row in term.logarithm_coefficients),variable)) for j in range(width))
    hx=tuple(_F(c.n.derivative()) for c in h)
    g,inv,_=_xgcd(_trim(h),mod,coeff(1))
    if len(g)!=1 or g[0].n!=g[0].d:
        raise InvalidInput("Logarithmic polynomial is not invertible in Q(x)[t]/q")
    z=_divmod((coeff(0),)+_mul(hx,inv),mod)[1]
    trace=coeff(0)
    for i in range(q.degree):
        column=_divmod((coeff(0),)*i+z,mod)[1]
        if i<len(column): trace=trace+column[i]
    return trace


def _check_primitive(source,value,domain=None):
    if type(value) is not RationalPrimitive or value.source!=source:
        return False
    from .functions import real_function, conjunction
    expected_domain=real_function(source).domain
    if domain is not None: expected_domain=conjunction(expected_domain,domain)
    if value.domain!=expected_domain: return False
    if value.rational_part.excluded!=source.excluded: return False
    derivative=value.rational_part.derivative()
    actual=_F(derivative.numerator,derivative.denominator)
    trace=_F(Polynomial((0,),source.variable))
    for term in value.log_terms:
        # Each logarithm factor must divide the source denominator in Q[t]/q.
        # A cancelling trace alone cannot justify adding logarithmic holes.
        q=Polynomial(term.residue_polynomial,source.variable)
        h=tuple(_K(Polynomial(row,source.variable),q) for row in term.logarithm_coefficients)
        denominator=tuple(_K(c,q) for c in source.denominator.coefficients)
        if _divmod(denominator,h)[1]: return False
        trace=trace+_trace_derivative(term,source.variable)
    total=actual+trace
    if total.n*source.denominator != source.numerator*total.d: return False
    residual=_F(Polynomial((0,),source.variable))
    for b,d in value.elementary_terms:
        if d.leading_coefficient!=Rational(1) or d.gcd(d.derivative()).degree!=0: return False
        if not (source.denominator % d).is_zero: return False
        residual=residual+_F(b,d)
    return residual.n*trace.d==trace.n*residual.d


def produce(operation,args):
    from . import functions as f
    with computation():
        source=_source(args["expression"],args["variable"])
        primitive=_primitive(source)
        if operation=="rational_antiderivative": return primitive,()
        if operation!="rational_integrate": raise UnsupportedOperation("Unknown rational integral operation")
        from .function_integration import finish_integral
        return finish_integral(f.real_function(source),primitive,args)


def check_operation(operation,args,value,evidence):
    from . import functions as f
    with computation():
        source=_source(args["expression"],args["variable"])
        if operation=="rational_antiderivative":
            return evidence==() and _check_primitive(source,value)
        if operation!="rational_integrate": return False
        from .function_integration import check_integral
        return check_integral(f.real_function(source),args,value,evidence,rational=source)
