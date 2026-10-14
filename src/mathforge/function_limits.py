"""Source-linked elementary limit certificates, with independent replay.

Local producers use successive derivatives; checkers propagate Taylor jets.
Infinity certificates use exact exponential-polynomial scale identities, not
unproved heuristics or approximate comparisons. Bounded-factor squeeze and
affine trigonometric oscillation have explicit theorem certificates.
"""
from __future__ import annotations
from dataclasses import dataclass
from .errors import InvalidInput, UnsupportedOperation
from .model import Rational, Symbol, Truth
from .limits import computation, current_budget
from .functions import (RealExpr,RealFunction,DomainPredicate,UnevaluatedLimit,ZERO,ONE,TRUE,
                        as_real_expr,real_function,simplify_expr,substitute_expr,intrinsic_domain,
                        conjunction,sign_of,exp,log,sqrt,pi,substitute_predicate,_tick)
from . import function_calculus as c
from .function_checks import equivalent,jet


@dataclass(frozen=True)
class LeadingCoefficient:
    expression: RealExpr
    order: int | None
    coefficient: RealExpr

    def __post_init__(self):
        if type(self.expression) is not RealExpr or type(self.coefficient) is not RealExpr:
            raise InvalidInput("A leading coefficient certificate requires exact expressions")
        if self.order is not None and (type(self.order) is not int or self.order<0):
            raise InvalidInput("A leading order must be nonnegative")


@dataclass(frozen=True)
class ExponentialScale:
    expression: RealExpr
    terms: tuple
    children: tuple[ExponentialScale,...] = ()

    def __post_init__(self):
        object.__setattr__(self,"terms",tuple(self.terms));object.__setattr__(self,"children",tuple(self.children))
        if type(self.expression) is not RealExpr or any(type(v) is not ExponentialScale for v in self.children):
            raise InvalidInput("Invalid exact scale certificate")
        keys=[]
        for term in self.terms:
            if type(term) is not tuple or len(term)!=4:raise InvalidInput("Invalid exact scale term")
            p,k,l,a=term
            if type(p) is not tuple or any(type(q) is not Rational for q in p) or type(k) is not int or type(l) is not int or type(a) is not RealExpr or a.free_symbols:
                raise InvalidInput("Invalid exact scale coefficient")
            if p and p[-1]==Rational(0):raise InvalidInput("Scale polynomial must be trimmed")
            keys.append((p,k,l))
        if len(set(keys))!=len(keys):raise InvalidInput("Duplicate exact scale terms")


@dataclass(frozen=True)
class FunctionLimitCertificate:
    rule: str
    source: RealFunction
    variable: Symbol
    point: object
    side: str
    value: c.FunctionLimitValue
    data: tuple = ()
    children: tuple[FunctionLimitCertificate,...] = ()

    def __post_init__(self):
        object.__setattr__(self,"data",tuple(self.data));object.__setattr__(self,"children",tuple(self.children))
        if self.rule not in ("continuity","quotient","log_zero","both","scale","elementary","rational","scale_quotient",
                             "oscillation","bounded_product","reciprocal"):
            raise InvalidInput("Unknown elementary limit certificate rule")
        if type(self.source) is not RealFunction or type(self.variable) is not Symbol or self.source.variables!=(self.variable,):
            raise InvalidInput("Limit certificates require a single explicit axis")
        if self.side not in ("both","left","right") or type(self.value) is not c.FunctionLimitValue:
            raise InvalidInput("Invalid limit certificate result")
        if any(type(child) is not FunctionLimitCertificate for child in self.children):
            raise InvalidInput("Invalid child limit certificate")


def _leading(expr,variable,point):
    if equivalent(expr,ZERO):return LeadingCoefficient(expr,None,ZERO)
    regular=conjunction(intrinsic_domain(expr),c.smooth_domain(expr,variable))
    if regular.contains({variable:point}) is not Truth.TRUE:
        raise UnsupportedOperation("Local analytic neighborhood is not certified")
    derivative=expr;factorial=1
    for n in range(13):
        _tick()
        coefficient=simplify_expr(substitute_expr(derivative,{variable:point})/factorial)
        if not equivalent(coefficient,ZERO):
            if sign_of(coefficient) is None:raise UnsupportedOperation("Local leading coefficient sign is unknown")
            return LeadingCoefficient(expr,n,coefficient)
        derivative=c._differentiate_expr(derivative,variable)
        factorial*=n+1
    raise UnsupportedOperation("Local cancellation exceeds supported expansion order")


def _check_leading(record,expr,variable,point):
    if type(record) is not LeadingCoefficient or record.expression!=expr:return False
    if record.order is None:return record.coefficient==ZERO and equivalent(expr,ZERO)
    if record.order>12:return False
    regular=conjunction(intrinsic_domain(expr),c.smooth_domain(expr,variable))
    if regular.contains({variable:point}) is not Truth.TRUE:return False
    coefficients=jet(expr,variable,point,record.order)
    return (all(equivalent(q,ZERO) for q in coefficients[:-1]) and
            equivalent(coefficients[-1],record.coefficient) and sign_of(record.coefficient) in (-1,1))


def _domain_atoms(predicate):
    if predicate.kind=="true":return ()
    if predicate.kind=="and":return tuple(p for child in predicate.operands for p in _domain_atoms(child))
    return (predicate,)


def _accept(predicate,sign):
    return sign!=0 if predicate.kind=="nonzero" else sign>=0 if predicate.kind=="nonnegative" else sign>0


def _near_evidence(predicate,variable,point,side):
    result=[]
    for atom in _domain_atoms(predicate):
        if atom.kind=="false":raise UnsupportedOperation("The source domain is empty")
        record=_leading(atom.operands[0],variable,point)
        sign=0 if record.order is None else sign_of(record.coefficient)*(-1 if side=="left" and record.order%2 else 1)
        if not _accept(atom,sign):raise UnsupportedOperation("The requested approach has no source domain")
        result.append(record)
    return tuple(result)


def _check_near(predicate,variable,point,side,records):
    atoms=_domain_atoms(predicate)
    if type(records) is not tuple or len(records)!=len(atoms):return False
    for atom,record in zip(atoms,records):
        if atom.kind=="false" or not _check_leading(record,atom.operands[0],variable,point):return False
        sign=0 if record.order is None else sign_of(record.coefficient)*(-1 if side=="left" and record.order%2 else 1)
        if not _accept(atom,sign):return False
    return True


def _scale_terms(mapping):
    return tuple((p,k,l,coefficient) for (p,k,l),coefficient in sorted(mapping.items(),key=lambda item:repr(item[0])))


def _scale_certificate(expr,variable):
    expr=simplify_expr(expr)
    mapping=c._scale(expr,variable)
    if variable not in expr.free_symbols or expr.op in ("symbol","exp","log"):children=()
    else:children=tuple(_scale_certificate(arg,variable) for arg in expr.args)
    return ExponentialScale(expr,_scale_terms(mapping),children)


def _trim(values):
    values=list(values)
    while values and values[-1]==Rational(0):values.pop()
    return tuple(values)


def _poly_add(a,b,sign=1):
    return _trim(tuple((a[i] if i<len(a) else Rational(0))+sign*(b[i] if i<len(b) else Rational(0)) for i in range(max(len(a),len(b)))))


def _poly_mul(a,b):
    if not a or not b:return ()
    values=[Rational(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):_tick();values[i+j]=values[i+j]+x*y
    return _trim(values)


def _rational_coefficients(expr,variable):
    """Checker-side polynomial expansion independent of Polynomial.from_expression."""
    _tick()
    if expr.op=="scalar" and type(expr.value) is Rational:return _trim((expr.value,))
    if expr.op=="symbol" and expr.value==variable:return (Rational(0),Rational(1))
    if expr.op in ("add","mul"):
        a,b=(_rational_coefficients(x,variable) for x in expr.args)
        return _poly_add(a,b) if expr.op=="add" else _poly_mul(a,b)
    if expr.op=="div":
        a,b=(_rational_coefficients(x,variable) for x in expr.args)
        if len(b)!=1 or b[0]==Rational(0):raise UnsupportedOperation("Not a polynomial coefficient division")
        return tuple(q/b[0] for q in a)
    if expr.op=="pow" and expr.value>=0:
        a=_rational_coefficients(expr.args[0],variable);n=expr.value;result=(Rational(1),)
        while n:
            _tick()
            if n&1:result=_poly_mul(result,a)
            n>>=1
            if n:a=_poly_mul(a,a)
        return result
    raise UnsupportedOperation("Not an exact rational polynomial")


def _check_scale(record,expr,variable):
    if type(record) is not ExponentialScale or record.expression!=simplify_expr(expr):return False
    expr=record.expression
    declared={(p,k,l):a for p,k,l,a in record.terms}
    if any(variable in a.free_symbols for a in declared.values()):return False
    if variable not in expr.free_symbols:
        expected={((),0,0):expr}
        if record.children:return False
    elif expr.op=="symbol":
        if record.children or expr.value!=variable:return False
        expected={((),1,0):ONE}
    elif expr.op=="exp":
        if record.children:return False
        p=_rational_coefficients(expr.args[0],variable)
        expected={(p[1:],0,0):simplify_expr(exp(as_real_expr(p[0] if p else Rational(0))))}
    elif expr.op=="log":
        if record.children:return False
        numerator,denominator=c._fraction_expression(expr.args[0])
        terms=[]
        for part in (numerator,denominator):
            p=_rational_coefficients(part,variable)
            nonzero=[(i,q) for i,q in enumerate(p) if q!=Rational(0)]
            if len(nonzero)!=1:return False
            terms.append(nonzero[0])
        degree=terms[0][0]-terms[1][0];coefficient=terms[0][1]/terms[1][1]
        if coefficient<=0:return False
        constant=simplify_expr(log(as_real_expr(coefficient)))
        expected={((),0,1):as_real_expr(degree)}
        if constant!=ZERO:expected[((),0,0)]=constant
    else:
        if len(record.children)!=len(expr.args):return False
        if not all(_check_scale(child,arg,variable) for child,arg in zip(record.children,expr.args)):return False
        maps=tuple({(p,k,l):a for p,k,l,a in child.terms} for child in record.children)
        expected={}
        if expr.op=="add":
            for mapping in maps:
                for key,coefficient in mapping.items():expected[key]=simplify_expr(expected.get(key,ZERO)+coefficient)
        elif expr.op in ("mul","div"):
            if expr.op=="div" and len(maps[1])!=1:return False
            for (p,k,l),a in maps[0].items():
                for (q,j,m),b in maps[1].items():
                    key=(_poly_add(p,q,1 if expr.op=="mul" else -1),k+j if expr.op=="mul" else k-j,l+m if expr.op=="mul" else l-m)
                    coefficient=a*b if expr.op=="mul" else a/b
                    expected[key]=simplify_expr(expected.get(key,ZERO)+coefficient)
        elif expr.op=="pow" and len(maps[0])==1:
            ((p,k,l),a),=maps[0].items()
            expected={(_trim(tuple(q*expr.value for q in p)),k*expr.value,l*expr.value):simplify_expr(a**expr.value)}
        else:return False
    return declared.keys()==expected.keys() and all(equivalent(declared[key],expected[key]) for key in expected)


def _growth_key_compare(a,b):
    p,k,l=a;q,j,m=b
    for i in range(max(len(p),len(q))-1,-1,-1):
        difference=(p[i] if i<len(p) else Rational(0))-(q[i] if i<len(q) else Rational(0))
        if difference!=Rational(0):return 1 if difference>0 else -1
    return (k>j)-(k<j) if k!=j else (l>m)-(l<m)


def _dominant(record):
    terms={(p,k,l):a for p,k,l,a in record.terms if not equivalent(a,ZERO)}
    if not terms:return None,ZERO
    key=next(iter(terms))
    for candidate in terms:
        if _growth_key_compare(candidate,key)>0:key=candidate
    coefficient=terms[key]
    if sign_of(coefficient) is None:raise UnsupportedOperation("Asymptotic sign is unknown")
    return key,coefficient


def _scale_value(record):
    key,coefficient=_dominant(record)
    if key is None:return c.FunctionLimitValue("finite",ZERO)
    growth=_growth_key_compare(key,((),0,0))
    if growth<0:return c.FunctionLimitValue("finite",ZERO)
    if growth==0:return c.FunctionLimitValue("finite",coefficient)
    return c.FunctionLimitValue("positive_infinity" if sign_of(coefficient)>0 else "negative_infinity")


def _infinity_expression(expr,variable,point):
    return simplify_expr(substitute_expr(expr,{variable:-as_real_expr(variable)})) if point=="-inf" else expr


def _eventual_evidence(predicate,variable,point):
    records=[]
    for atom in _domain_atoms(predicate):
        if atom.kind=="false":raise UnsupportedOperation("The source domain is empty")
        expr=atom.operands[0]
        # A limiting zero alone cannot establish an eventual sign. Exact scale
        # identities can, even when their magnitude tends to zero.
        try:
            record=_scale_certificate(_infinity_expression(expr,variable,point),variable)
            _,coefficient=_dominant(record);sign=sign_of(coefficient)
        except UnsupportedOperation:
            child=_produce(real_function(expr,(variable,)),variable,point,"both")
            if child.value.kind=="finite":sign=sign_of(child.value.value)
            else:sign=1 if child.value.kind=="positive_infinity" else -1 if child.value.kind=="negative_infinity" else None
            if sign in (None,0):raise UnsupportedOperation("Eventual domain sign is unknown")
            record=child
        if sign is None or not _accept(atom,sign):raise UnsupportedOperation("Source domain excludes the infinite approach")
        records.append(record)
    return tuple(records)


def _check_eventual(predicate,variable,point,records):
    atoms=_domain_atoms(predicate)
    if type(records) is not tuple or len(records)!=len(atoms):return False
    for atom,record in zip(atoms,records):
        if atom.kind=="false":return False
        expr=atom.operands[0]
        if type(record) is ExponentialScale:
            if not _check_scale(record,_infinity_expression(expr,variable,point),variable):return False
            _,coefficient=_dominant(record);sign=sign_of(coefficient)
        elif type(record) is FunctionLimitCertificate:
            if record.source!=real_function(expr,(variable,)) or record.variable!=variable or record.point!=point or record.side!="both" or not _verify(record):return False
            sign=sign_of(record.value.value) if record.value.kind=="finite" else 1 if record.value.kind=="positive_infinity" else -1 if record.value.kind=="negative_infinity" else None
            if sign in (None,0):return False
        else:return False
        if sign is None or not _accept(atom,sign):return False
    return True


def _elementary_value(op,inner):
    if inner.kind=="finite":
        function=real_function(RealExpr(op,(inner.value,)),())
        if function.domain.contains() is Truth.TRUE:return c.FunctionLimitValue("finite",function.expression)
        return None
    positive=inner.kind=="positive_infinity";negative=inner.kind=="negative_infinity"
    if op=="exp" and (positive or negative):return c.FunctionLimitValue("positive_infinity") if positive else c.FunctionLimitValue("finite",ZERO)
    if op=="log" and positive:return c.FunctionLimitValue("positive_infinity")
    if op=="sqrt" and positive:return c.FunctionLimitValue("positive_infinity")
    if op=="abs" and (positive or negative):return c.FunctionLimitValue("positive_infinity")
    if op=="sinh" and (positive or negative):return inner
    if op=="cosh" and (positive or negative):return c.FunctionLimitValue("positive_infinity")
    if op in ("tanh","atan") and (positive or negative):
        sign=1 if positive else -1
        return c.FunctionLimitValue("finite",as_real_expr(sign) if op=="tanh" else simplify_expr(sign*pi/2))
    return None


def _same_value(a,b):
    return a.kind==b.kind and (a.value is None and b.value is None or type(a.value) is RealExpr and type(b.value) is RealExpr and equivalent(a.value,b.value))


def _absolute_bound(expr):
    """A global rational bound on the expression wherever it is defined."""
    _tick()
    if expr.op in ("sin","cos","tanh"):return Rational(1)
    if expr.op=="scalar" and type(expr.value) is Rational:
        return expr.value if expr.value>=0 else -expr.value
    if expr.op in ("add","mul"):
        a,b=(_absolute_bound(arg) for arg in expr.args)
        return a+b if expr.op=="add" else a*b
    if expr.op=="pow" and expr.value>=0:
        return _absolute_bound(expr.args[0])**expr.value
    raise UnsupportedOperation("No global bound theorem matches")


def _bounded_parts(expr):
    """Structural factorizations only; no speculative algebraic cancellation."""
    candidates=[]
    if expr.op=="mul":
        candidates.extend(((expr.args[0],expr.args[1]),(expr.args[1],expr.args[0])))
    elif expr.op=="div":candidates.append((expr.args[0],ONE/expr.args[1]))
    for bounded,coefficient in candidates:
        try:bound=_absolute_bound(bounded)
        except UnsupportedOperation:continue
        yield bounded,coefficient,bound


def _reciprocal_source(source,variable,point,side):
    replacement=point+(1 if side=="right" else -1)/as_real_expr(variable)
    return real_function(substitute_expr(source.expression,{variable:replacement}),(variable,),
                         domain=substitute_predicate(source.domain,{variable:replacement}))


def _produce(source,variable,point,side):
    _tick()
    infinite=type(point) is str and point in ("-inf","+inf")
    if not infinite and side=="both":
        left=_produce(source,variable,point,"left");right=_produce(source,variable,point,"right")
        value=left.value if _same_value(left.value,right.value) else c.FunctionLimitValue("does_not_exist")
        return FunctionLimitCertificate("both",source,variable,point,side,value,(),(left,right))
    if infinite:
        domain=_eventual_evidence(source.domain,variable,point)
        expr=_infinity_expression(source.expression,variable,point)
        # Legacy rational certificates remain independently replayable.
        try:
            from .rational_function import rational_function
            from .rational_limits import limit_value
            numerator,denominator=c._fraction_expression(source.expression)
            rf=rational_function(c._polynomial(numerator,variable),c._polynomial(denominator,variable),variable=variable)
            old_value,old_evidence=limit_value(rf,variable,point,"both")
            value=c.FunctionLimitValue(old_value.kind,as_real_expr(old_value.value) if old_value.kind=="finite" else None)
            return FunctionLimitCertificate("rational",source,variable,point,side,value,(domain,rf,old_evidence))
        except (UnsupportedOperation,InvalidInput,TypeError):pass
        try:
            scale=_scale_certificate(expr,variable)
            value=_scale_value(scale)
            return FunctionLimitCertificate("scale",source,variable,point,side,value,(domain,scale))
        except UnsupportedOperation:pass
        try:
            numerator,denominator=c._fraction_expression(expr)
            top,bottom=_scale_certificate(numerator,variable),_scale_certificate(denominator,variable)
            key_a,coefficient_a=_dominant(top);key_b,coefficient_b=_dominant(bottom)
            if key_b is None:raise InvalidInput("Identically zero asymptotic denominator")
            if key_a is None:value=c.FunctionLimitValue("finite",ZERO)
            else:
                key=(_poly_add(key_a[0],key_b[0],-1),key_a[1]-key_b[1],key_a[2]-key_b[2])
                value=_scale_value(ExponentialScale(expr,((key[0],key[1],key[2],simplify_expr(coefficient_a/coefficient_b)),)))
            return FunctionLimitCertificate("scale_quotient",source,variable,point,side,value,(domain,top,bottom))
        except UnsupportedOperation:pass
        if source.expression.op in ("sin","cos"):
            try:
                polynomial=c._polynomial(source.expression.args[0],variable)
                if polynomial.degree==1:
                    value=c.FunctionLimitValue("does_not_exist")
                    return FunctionLimitCertificate("oscillation",source,variable,point,side,value,
                                                    (domain,polynomial.coefficients))
            except UnsupportedOperation:pass
        for bounded,coefficient,bound in _bounded_parts(source.expression):
            try:child=_produce(real_function(coefficient,(variable,)),variable,point,"both")
            except UnsupportedOperation:continue
            if child.value.kind=="finite" and equivalent(child.value.value,ZERO):
                return FunctionLimitCertificate("bounded_product",source,variable,point,side,
                                                c.FunctionLimitValue("finite",ZERO),(domain,bounded,bound),(child,))
        if source.expression.op in ("sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","exp","log","sqrt","abs"):
            child=_produce(real_function(source.expression.args[0],(variable,)),variable,point,"both")
            value=_elementary_value(source.expression.op,child.value)
            if value is not None:return FunctionLimitCertificate("elementary",source,variable,point,side,value,(domain,),(child,))
        raise UnsupportedOperation("No certified infinite-limit rule matches")
    domain=_near_evidence(source.domain,variable,point,side)
    if source.domain.contains({variable:point}) is Truth.TRUE:
        value=simplify_expr(substitute_expr(source.expression,{variable:point}))
        return FunctionLimitCertificate("continuity",source,variable,point,side,c.FunctionLimitValue("finite",value),(domain,))
    if source.expression.op=="log":
        child=_produce(real_function(source.expression.args[0],(variable,)),variable,point,side)
        if child.value.kind=="finite" and equivalent(child.value.value,ZERO):
            return FunctionLimitCertificate("log_zero",source,variable,point,side,c.FunctionLimitValue("negative_infinity"),(domain,),(child,))
    numerator,denominator=c._fraction_expression(source.expression)
    try:top,bottom=_leading(numerator,variable,point),_leading(denominator,variable,point)
    except UnsupportedOperation:
        child=_produce(_reciprocal_source(source,variable,point,side),variable,"+inf","both")
        return FunctionLimitCertificate("reciprocal",source,variable,point,side,child.value,(domain,),(child,))
    if bottom.order is None:raise InvalidInput("Identically zero limit denominator")
    if top.order is None:value=c.FunctionLimitValue("finite",ZERO)
    else:
        degree=top.order-bottom.order;coefficient=simplify_expr(top.coefficient/bottom.coefficient)
        if degree>0:value=c.FunctionLimitValue("finite",ZERO)
        elif degree==0:value=c.FunctionLimitValue("finite",coefficient)
        else:
            sign=sign_of(coefficient)*(-1 if side=="left" and degree%2 else 1)
            value=c.FunctionLimitValue("positive_infinity" if sign>0 else "negative_infinity")
    return FunctionLimitCertificate("quotient",source,variable,point,side,value,(domain,top,bottom))


def _verify(record):
    _tick()
    if type(record) is not FunctionLimitCertificate:return False
    source,variable,point,side,value=record.source,record.variable,record.point,record.side,record.value
    infinite=type(point) is str and point in ("-inf","+inf")
    if infinite and side!="both":return False
    if not infinite and (type(point) is not RealExpr or point.free_symbols):return False
    if record.rule=="both":
        if infinite or side!="both" or record.data or len(record.children)!=2:return False
        left,right=record.children
        if any(child.source!=source or child.variable!=variable or child.point!=point or child.side!=direction or not _verify(child) for child,direction in ((left,"left"),(right,"right"))):return False
        expected=left.value if _same_value(left.value,right.value) else c.FunctionLimitValue("does_not_exist")
        return _same_value(value,expected)
    if not record.data:return False
    if infinite:
        if not _check_eventual(source.domain,variable,point,record.data[0]):return False
    elif side=="both" or not _check_near(source.domain,variable,point,side,record.data[0]):return False
    if record.rule=="continuity":
        return (not infinite and len(record.data)==1 and not record.children and
                source.domain.contains({variable:point}) is Truth.TRUE and value.kind=="finite" and
                equivalent(value.value,substitute_expr(source.expression,{variable:point})))
    if record.rule=="reciprocal":
        if infinite or len(record.data)!=1 or len(record.children)!=1:return False
        child=record.children[0]
        return (child.source==_reciprocal_source(source,variable,point,side) and child.variable==variable and
                child.point=="+inf" and child.side=="both" and _verify(child) and _same_value(value,child.value))
    if record.rule=="oscillation":
        if not infinite or len(record.data)!=2 or record.children or source.expression.op not in ("sin","cos"):return False
        coefficients=_rational_coefficients(source.expression.args[0],variable)
        # A nonconstant affine phase reaches alternating extrema on two
        # sequences tending to the requested infinity. Its full tail is in
        # the source domain, as established by the common domain certificate.
        return (len(coefficients)==2 and coefficients[1]!=Rational(0) and record.data[1]==coefficients and
                value.kind=="does_not_exist")
    if record.rule=="bounded_product":
        if not infinite or len(record.data)!=3 or len(record.children)!=1:return False
        child=record.children[0];bounded,bound=record.data[1:]
        if child.variable!=variable or child.point!=point or child.side!="both" or not _verify(child):return False
        if child.value.kind!="finite" or not equivalent(child.value.value,ZERO):return False
        # Replay the global bound and the exact structural factorization.
        matches=any(bounded==part and bound==maximum and child.source==real_function(coefficient,(variable,))
                    for part,coefficient,maximum in _bounded_parts(source.expression))
        return matches and value.kind=="finite" and equivalent(value.value,ZERO)
    if record.rule=="quotient":
        if infinite or len(record.data)!=3 or record.children:return False
        numerator,denominator=c._fraction_expression(source.expression);top,bottom=record.data[1:]
        if not _check_leading(top,numerator,variable,point) or not _check_leading(bottom,denominator,variable,point) or bottom.order is None:return False
        if top.order is None:return value.kind=="finite" and equivalent(value.value,ZERO)
        degree=top.order-bottom.order;coefficient=simplify_expr(top.coefficient/bottom.coefficient)
        if degree>=0:return value.kind=="finite" and equivalent(value.value,ZERO if degree>0 else coefficient)
        sign=sign_of(coefficient)
        if side=="left" and degree%2:sign=-sign
        return value.kind==("positive_infinity" if sign>0 else "negative_infinity")
    if record.rule in ("log_zero","elementary"):
        if len(record.data)!=1 or len(record.children)!=1:return False
        child=record.children[0]
        if source.expression.op not in ("sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","exp","log","sqrt","abs"):return False
        if child.source!=real_function(source.expression.args[0],(variable,)) or child.variable!=variable or child.point!=point or child.side!=side or not _verify(child):return False
        if record.rule=="log_zero":return not infinite and source.expression.op=="log" and child.value.kind=="finite" and equivalent(child.value.value,ZERO) and value.kind=="negative_infinity"
        if not infinite:return False
        # Replay the limit theorem; no producer or limit-search entrypoint runs.
        expected=_elementary_value(source.expression.op,child.value)
        return expected is not None and _same_value(value,expected)
    if record.rule=="scale":
        if not infinite or len(record.data)!=2 or record.children:return False
        scale=record.data[1]
        if not _check_scale(scale,_infinity_expression(source.expression,variable,point),variable):return False
        terms={(p,k,l):a for p,k,l,a in scale.terms if not equivalent(a,ZERO)}
        if not terms:return value.kind=="finite" and equivalent(value.value,ZERO)
        # Independently prove one term dominates every distinct exact scale.
        for key,coefficient in terms.items():
            if any(_growth_key_compare(key,other)<0 for other in terms):continue
            sign=sign_of(coefficient)
            if sign is None:return False
            growth=_growth_key_compare(key,((),0,0))
            if growth<=0:return value.kind=="finite" and equivalent(value.value,ZERO if growth<0 else coefficient)
            return value.kind==("positive_infinity" if sign>0 else "negative_infinity")
        return False
    if record.rule=="rational":
        if not infinite or len(record.data)!=3 or record.children:return False
        from .rational_function import rational_function
        from .rational_limits import check_limit
        from .analysis_types import LimitValue
        numerator,denominator=c._fraction_expression(source.expression)
        rf=rational_function(c._polynomial(numerator,variable),c._polynomial(denominator,variable),variable=variable)
        if record.data[1]!=rf:return False
        if value.kind=="finite" and value.value.op!="scalar":return False
        old_value=LimitValue(value.kind,value.value.value if value.kind=="finite" else None)
        return check_limit(rf,variable,point,"both",old_value,record.data[2])
    if record.rule=="scale_quotient":
        if not infinite or len(record.data)!=3 or record.children:return False
        numerator,denominator=c._fraction_expression(_infinity_expression(source.expression,variable,point))
        top,bottom=record.data[1:]
        if not _check_scale(top,numerator,variable) or not _check_scale(bottom,denominator,variable):return False
        key_a,coefficient_a=_dominant(top);key_b,coefficient_b=_dominant(bottom)
        if key_b is None:return False
        if key_a is None:return value.kind=="finite" and equivalent(value.value,ZERO)
        growth=_growth_key_compare(key_a,key_b);coefficient=simplify_expr(coefficient_a/coefficient_b)
        if growth<=0:return value.kind=="finite" and equivalent(value.value,ZERO if growth<0 else coefficient)
        sign=sign_of(coefficient)
        return sign in (-1,1) and value.kind==("positive_infinity" if sign>0 else "negative_infinity")
    return False


def produce(expression,variable,point,side="both"):
    source=c._source(expression,variable)
    if source.variables!=(variable,):raise UnsupportedOperation("Function limits are univariate")
    if side not in ("left","right","both"):raise InvalidInput("Invalid limit side")
    if type(point) is str:
        if point not in ("-inf","+inf"):raise InvalidInput("Invalid infinite limit point")
        if side!="both":raise InvalidInput("Infinite limits use side='both'")
        at=point
    else:
        raw_point=as_real_expr(point)
        if raw_point.free_symbols or intrinsic_domain(raw_point).contains() is not Truth.TRUE:
            raise InvalidInput("Limit point must be a defined closed real value")
        at=simplify_expr(raw_point)
    try:
        certificate=_produce(source,variable,at,side)
        return certificate.value,(certificate,)
    except UnsupportedOperation:return UnevaluatedLimit(source,variable,at,side),()


def check(args,value,evidence):
    if type(value) is not c.FunctionLimitValue or type(evidence) is not tuple or len(evidence)!=1:return False
    certificate=evidence[0]
    if type(certificate) is not FunctionLimitCertificate:return False
    if type(args["point"]) is str:point=args["point"]
    else:
        raw_point=as_real_expr(args["point"])
        if raw_point.free_symbols or intrinsic_domain(raw_point).contains() is not Truth.TRUE:return False
        point=simplify_expr(raw_point)
    return (certificate.source==c._source(args["expression"],args["variable"]) and certificate.variable==args["variable"] and
            certificate.point==point and certificate.side==args["side"] and certificate.value==value and _verify(certificate))


TYPES={"leading_coefficient":LeadingCoefficient,"exponential_scale":ExponentialScale,
       "function_limit_certificate":FunctionLimitCertificate}
