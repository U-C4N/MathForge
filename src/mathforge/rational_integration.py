"""Hermite reduction and Lazard--Rioboo--Trager logarithmic primitives.

LogRootSum(q,h) means sum_{q(t)=0} t log(h(t,x)), modulo locally
constant choices of logarithm. It is a formal primitive, never a pointwise
principal-complex-log evaluator. A separate real elementary conversion covers
linear/quadratic denominator factors and proper finite integrals.

The field arithmetic of the LRT step and of the trace checker (Q[t]/q and Q(x))
runs on plain coefficient tuples, constant first, with no trailing zeros and ()
for zero: no Polynomial record per operation, and no gcd per Q(x) operation.
A Q(x) element keeps its numerator and denominator unreduced and is reduced
only when its denominator grows past a few terms; comparisons cross-multiply.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidInput, UnsupportedOperation
from .limits import computation, current_budget
from .model import Rational, Symbol
from .polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from .rational_function import RationalFunction, as_rational_function, validate_variable
from .functions import DomainPredicate


def _tick(n=1):
    current_budget().tick(n)


def _zero(value):
    if isinstance(value, _K): return not value.p
    if isinstance(value, _F): return not value.n
    return value == Rational(0) if type(value) is Rational else value == 0


# --- Dense coefficient tuples over Q: constant first, trimmed, () for zero -------------------------------------

_Q0 = Rational(0)
_Q1 = Rational(1)
_UNIT = (_Q1,)


def _tuple(polynomial):
    """A Polynomial's coefficients as a trimmed tuple: () for the zero polynomial."""
    coefficients = polynomial.coefficients
    return () if len(coefficients) == 1 and not coefficients[0].numerator else coefficients


def _record(coefficients):
    """A trimmed tuple as Polynomial.coefficients spell it: (Rational(0),) for zero."""
    return coefficients if coefficients else (_Q0,)


def _scalar(value):
    """An int or a Rational as a trimmed tuple."""
    if type(value) is not Rational:
        value = Rational(value)
    return (value,) if value.numerator else ()


def _norm(values):
    end = len(values)
    while end and not values[end - 1].numerator:
        end -= 1
    return tuple(values[:end])


def _padd(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return a
    out = list(a)
    for i, y in enumerate(b):
        out[i] = out[i] + y
    return _norm(out) if len(a) == len(b) else tuple(out)


def _psub(a, b):
    if not b:
        return a
    size = max(len(a), len(b))
    out = [(a[i] if i < len(a) else _Q0) - (b[i] if i < len(b) else _Q0) for i in range(size)]
    return _norm(out) if len(a) == len(b) else tuple(out)


def _pneg(a):
    return tuple([-y for y in a])


def _pmul(a, b):
    if not a or not b:
        return ()
    _tick(len(a) * len(b))
    out = [_Q0] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        if u.numerator:
            for j, v in enumerate(b):
                if v.numerator:
                    out[i + j] = out[i + j] + u * v
    return tuple(out)  # Over a field the leading product is nonzero.


def _pdivmod(a, b):
    """Quotient and remainder by a nonzero trimmed b."""
    shift = len(a) - len(b)
    if shift < 0:
        return (), a
    _tick(shift + 1)
    lead = b[-1]
    top = len(b) - 1
    remainder = list(a)
    quotient = [_Q0] * (shift + 1)
    for k in range(shift, -1, -1):
        c = remainder[k + top]
        if c.numerator:
            if lead != _Q1:
                c = c / lead
            quotient[k] = c
            for j in range(top):
                if b[j].numerator:
                    remainder[k + j] = remainder[k + j] - c * b[j]
    return _norm(quotient), _norm(remainder[:top])


def _pmod(a, b):
    return a if len(a) < len(b) else _pdivmod(a, b)[1]


def _pmonic(a):
    lead = a[-1]
    return a if lead == _Q1 else tuple([y / lead for y in a])


def _pgcd(a, b):
    """The monic gcd; () when both are zero."""
    while b:
        _tick()
        a, b = b, _pmod(a, b)
    return _pmonic(a) if a else ()


def _pinverse(a, modulus):
    """The inverse of a modulo a monic modulus of degree >= 1, or None when gcd(a, modulus) is not 1."""
    old_r, r, old_s, s = a, modulus, _UNIT, ()
    while r:
        _tick()
        quotient, remainder = _pdivmod(old_r, r)
        old_r, r = r, remainder
        old_s, s = s, _psub(old_s, _pmul(quotient, s))
    if len(old_r) != 1:
        return None
    lead = old_r[0]
    return old_s if lead == _Q1 else tuple([y / lead for y in old_s])


def _pderivative(a):
    return tuple([a[i] * i for i in range(1, len(a))])


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
    """A coefficient in Q[t]/q for the producer's polynomial Euclidean step: a coefficient tuple reduced modulo
    the monic coefficient tuple q."""
    __slots__ = ("p", "q")

    def __init__(self, p, q):
        self.q = q
        self.p = p if len(p) < len(q) else _pdivmod(p, q)[1]

    def _of(self, value): return value.p if type(value) is _K else _scalar(value)
    def __add__(self, other): return _K(_padd(self.p, self._of(other)), self.q)
    __radd__ = __add__
    def __neg__(self): return _K(_pneg(self.p), self.q)
    def __sub__(self, other): return _K(_psub(self.p, self._of(other)), self.q)
    def __mul__(self, other): return _K(_pmul(self.p, self._of(other)), self.q)
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._of(other)
        if other == _UNIT:
            return self
        inverse = _pinverse(other, self.q)
        if inverse is None: raise InvalidInput("Noninvertible residue coefficient")
        return _K(_pmul(self.p, inverse), self.q)


# A Q(x) fraction is reduced by its gcd only once its denominator has more coefficients than this: constant and
# linear denominators stay unreduced (no gcd), while the remainder sequences over Q(x) of high-degree residue
# polynomials, where unreduced fractions swell, keep their fractions reduced.
_REDUCE_ABOVE = 2


class _F:
    """Q(x) coefficients used only by the independent trace checker: a numerator and a nonzero denominator,
    coefficient tuples, kept unreduced (see _REDUCE_ABOVE). Zero tests read the numerator, and equalities
    cross-multiply, so no comparison depends on the reduction.

    These are formal field elements: no evaluation domain is silently inferred
    from intermediate denominators. The public source keeps its own exclusions.
    """
    __slots__ = ("n", "d")

    def __init__(self, n, d=_UNIT):
        if not d: raise ZeroDivisionError("Zero formal denominator")
        if not n:
            d = _UNIT
        elif len(d) > _REDUCE_ABOVE:
            common = _pgcd(n, d)
            if len(common) > 1:
                n, d = _pdivmod(n, common)[0], _pdivmod(d, common)[0]
            lead = d[-1]
            if lead != _Q1:
                n, d = tuple([y / lead for y in n]), tuple([y / lead for y in d])
        self.n = n
        self.d = d

    @staticmethod
    def _of(x): return x if type(x) is _F else _F(_scalar(x))

    def __add__(self, x):
        x = _F._of(x)
        if not x.n: return self
        if not self.n: return x
        if self.d == x.d: return _F(_padd(self.n, x.n), self.d)
        return _F(_padd(_pmul(self.n, x.d), _pmul(x.n, self.d)), _pmul(self.d, x.d))
    __radd__ = __add__
    def __neg__(self): return _F(_pneg(self.n), self.d)
    def __sub__(self, x): return self + -_F._of(x)

    def __mul__(self, x):
        x = _F._of(x)
        if not self.n or not x.n: return _F(())
        return _F(_pmul(self.n, x.n), _pmul(self.d, x.d))
    __rmul__ = __mul__

    def __truediv__(self, x):
        x = _F._of(x)
        return _F(_pmul(self.n, x.d), _pmul(self.d, x.n))


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
    x=d.variable; derivative=d.derivative()
    samples=[resultant(d,b-derivative*k) for k in range(d.degree+1)]
    q=()
    for k,y in enumerate(samples):
        basis=_UNIT; divisor=1
        for j in range(d.degree+1):
            if j!=k:
                basis=_pmul(basis,(Rational(-j),_Q1)); divisor*=k-j; _tick()
        weight=y/divisor
        if weight.numerator:
            q=_padd(q,tuple([c*weight for c in basis]))
    factors,_=factor_with_evidence(Polynomial(_record(q),x))
    b_coefficients,d_coefficients,slope=_tuple(b),d.coefficients,_tuple(derivative)
    terms=[]
    for factor in factors.factors:
        modulus=factor.polynomial.coefficients
        d_k=tuple(_K(_scalar(c),modulus) for c in d_coefficients)
        residue=_K((_Q0,_Q1),modulus)
        bd=tuple(_K(_scalar(b_coefficients[i]) if i<len(b_coefficients) else (),modulus)
                 - residue*(slope[i] if i<len(slope) else 0)
                 for i in range(max(len(b_coefficients),len(slope))))
        g,_,_=_xgcd(_trim(d_k),_trim(bd),_K(_UNIT,modulus))
        if len(g)<2: raise ArithmeticError("Residue resultant did not select a logarithmic factor")
        terms.append(LogRootSum(modulus,tuple(_record(v.p) for v in g)))
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


def _dense(values):
    """A coefficient sequence as Polynomial(values, x) would hold it, as a trimmed tuple: exact rationals only and
    the dense degree bound, as that constructor demands."""
    if any(type(value) is not Rational for value in values):
        if any(type(value) not in (Rational, int) for value in values):
            raise UnsupportedOperation("Polynomial coefficients must be rational")
        values = tuple(Rational(value) for value in values)
    values = _norm(values)
    if len(values) - 1 > MAX_POLYNOMIAL_DEGREE:
        raise UnsupportedOperation(f"Dense polynomial degree exceeds {MAX_POLYNOMIAL_DEGREE}")
    return values


def _trace_derivative(term,variable):
    """Independent Q(x)[t]/q trace of t*h_x/h; no LRT or Hermite calls."""
    q=_dense(term.residue_polynomial)
    if _pgcd(q,_pderivative(q))!=_UNIT:
        raise InvalidInput("Residue polynomial must be square-free")
    zero=_F(())
    mod=tuple(_F(_scalar(c)) for c in q)
    rows=term.logarithm_coefficients
    width=max(len(row) for row in rows)
    columns=[_dense(tuple(row[j] if j<len(row) else _Q0 for row in rows)) for j in range(width)]
    h=tuple(_F(c) for c in columns)
    hx=tuple(_F(_pderivative(c)) for c in columns)
    g,inv,_=_xgcd(_trim(h),mod,_F(_UNIT))
    if len(g)!=1 or g[0].n!=g[0].d:
        raise InvalidInput("Logarithmic polynomial is not invertible in Q(x)[t]/q")
    column=_divmod((zero,)+_mul(hx,inv),mod)[1]
    trace=zero
    for i in range(len(q)-1):
        # column = t^i * z mod q, from the previous one with a single reduction step.
        if i<len(column): trace=trace+column[i]
        column=_divmod((zero,)+column,mod)[1]
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
    actual=_F(_tuple(derivative.numerator),_tuple(derivative.denominator))
    trace=_F(())
    numerator,denominator=_tuple(source.numerator),_tuple(source.denominator)
    for term in value.log_terms:
        # Each logarithm factor must divide the source denominator in Q[t]/q.
        # A cancelling trace alone cannot justify adding logarithmic holes.
        q=_dense(term.residue_polynomial)
        h=tuple(_K(_dense(row),q) for row in term.logarithm_coefficients)
        denominator_k=tuple(_K(_scalar(c),q) for c in denominator)
        if _divmod(denominator_k,h)[1]: return False
        trace=trace+_trace_derivative(term,source.variable)
    total=actual+trace
    if _pmul(total.n,denominator)!=_pmul(numerator,total.d): return False
    residual=_F(())
    for b,d in value.elementary_terms:
        d_coefficients=_tuple(d)
        if d.leading_coefficient!=Rational(1) or _pgcd(d_coefficients,_pderivative(d_coefficients))!=_UNIT: return False
        if _pmod(denominator,d_coefficients): return False
        residual=residual+_F(_tuple(b),d_coefficients)
    return _pmul(residual.n,trace.d)==_pmul(trace.n,residual.d)


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
