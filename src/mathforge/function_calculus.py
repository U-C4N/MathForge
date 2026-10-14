"""Certified elementary calculus over the separate domain-aware function model.

Differentiation produces symbolic chain-rule expressions. Its checker uses
independent truncated coefficient propagation, including for Taylor records.
The integration catalogue is intentionally explicit: a failed match is a
deferred question, never a claim that no elementary antiderivative exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from .errors import InvalidInput, UnsupportedOperation
from .limits import computation, current_budget, ResourceLimitError
from .model import Rational, Symbol, Truth
from .functions import (
    RealExpr, RealFunction, DomainPredicate, UnevaluatedIntegral, UnevaluatedLimit,
    ZERO, ONE, TRUE, pi, E, as_real_expr, real_function, simplify_expr,
    intrinsic_domain, conjunction, condition, substitute_expr, substitute_predicate,
    _axes, _rational, _tick, sign_of, sin, cos, tan, asin, acos, atan, sinh,
    cosh, tanh, exp, log, sqrt, abs,
)


def _order(n):
    if type(n) is not int or n < 0: raise InvalidInput("Derivative/Taylor order must be a nonnegative integer")
    budget=current_budget()
    if budget is not None:
        budget.tick(n)
        budget.check_int(n)
    return n


def _source(expression, variable=None, variables=None):
    if type(expression) is RealFunction:
        source=expression
        if variables is not None and source.variables != tuple(variables):
            # Reordering axes is explicit and changes the output coordinate order.
            axes=_axes(variables,source.expression.free_symbols)
            source=RealFunction(source.expression,axes,source.domain)
    else:
        expr=as_real_expr(expression)
        axes=variables
        if axes is None:
            axes=tuple(sorted(expr.free_symbols | ({variable} if variable is not None else set()),key=lambda v:v.uid))
        source=real_function(expr,axes)
    if variable is not None and (type(variable) is not Symbol or variable not in source.variables):
        raise InvalidInput("Derivative variable must be an explicit function axis")
    return source


def smooth_domain(expr,variable):
    """A sufficient smoothness region, not a general maximal-domain solver."""
    if variable not in expr.free_symbols:return TRUE
    required=[smooth_domain(arg,variable) for arg in expr.args]
    if expr.op in ("sqrt","abs"):
        required.append(condition("positive" if expr.op=="sqrt" else "nonzero",expr.args[0]))
    if expr.op in ("asin","acos"):
        required.append(condition("positive",1-expr.args[0]**2))
    return conjunction(*required)


def interior_domain(predicate,variable):
    """A sufficient open neighborhood for ordinary two-sided derivatives."""
    if predicate.kind in ("true","false"):return predicate
    if predicate.kind=="and":return conjunction(*(interior_domain(p,variable) for p in predicate.operands))
    if predicate.kind=="nonnegative" and variable in predicate.operands[0].free_symbols:
        return condition("positive",predicate.operands[0])
    return predicate


def _differentiate_expr(expr,variable):
    _tick()
    if variable not in expr.free_symbols:return ZERO
    if expr.op=="symbol":return ONE
    args=expr.args
    derivatives=tuple(_differentiate_expr(a,variable) for a in args)
    if expr.op=="add":out=derivatives[0]+derivatives[1]
    elif expr.op=="mul":out=derivatives[0]*args[1]+args[0]*derivatives[1]
    elif expr.op=="div":out=(derivatives[0]*args[1]-args[0]*derivatives[1])/(args[1]**2)
    elif expr.op=="pow":out=expr.value*args[0]**(expr.value-1)*derivatives[0]
    else:
        u,du=args[0],derivatives[0]
        if expr.op=="sin":out=cos(u)*du
        elif expr.op=="cos":out=-sin(u)*du
        elif expr.op=="tan":out=du/cos(u)**2
        elif expr.op=="asin":out=du/sqrt(1-u**2)
        elif expr.op=="acos":out=-du/sqrt(1-u**2)
        elif expr.op=="atan":out=du/(1+u**2)
        elif expr.op=="sinh":out=cosh(u)*du
        elif expr.op=="cosh":out=sinh(u)*du
        elif expr.op=="tanh":out=du/cosh(u)**2
        elif expr.op=="exp":out=exp(u)*du
        elif expr.op=="log":out=du/u
        elif expr.op=="sqrt":out=du/(2*sqrt(u))
        elif expr.op=="abs":out=u*du/abs(u)
        else:raise UnsupportedOperation("Unknown elementary derivative")
    return simplify_expr(out)


@dataclass(frozen=True)
class FunctionDerivativeTrace:
    source: RealFunction
    variable: Symbol
    derivatives: tuple[RealFunction,...]

    def __post_init__(self):
        object.__setattr__(self,"derivatives",tuple(self.derivatives))
        if type(self.source) is not RealFunction or type(self.variable) is not Symbol or self.variable not in self.source.variables:
            raise InvalidInput("Invalid derivative trace source")
        if any(type(d) is not RealFunction or d.variables!=self.source.variables for d in self.derivatives):
            raise InvalidInput("Invalid derivative trace values")


def _trace(source,variable,n):
    values=[];previous=source
    for _ in range(_order(n)):
        raw=_differentiate_expr(previous.expression,variable)
        domain=conjunction(previous.domain,interior_domain(previous.domain,variable),smooth_domain(previous.expression,variable))
        previous=real_function(raw,source.variables,domain=domain)
        values.append(previous)
    return FunctionDerivativeTrace(source,variable,tuple(values))


def _last(trace):return trace.derivatives[-1] if trace.derivatives else trace.source


def differentiate(expression,variable,n=1,*,limits=None):
    with computation(limits) as budget:
        source=_source(expression,variable)
        budget.inspect(source)
        return _last(_trace(source,variable,n))


def gradient(expression,variables,*,limits=None):
    with computation(limits) as budget:
        axes=_axes(variables)
        if not axes:raise InvalidInput("Gradient axes cannot be empty")
        source=_source(expression,variables=axes);budget.inspect(source)
        return tuple(_last(_trace(source,v,1)) for v in axes)


def jacobian(expressions,variables,*,limits=None):
    with computation(limits) as budget:
        axes=_axes(variables);values=tuple(expressions)
        if not axes or not values:raise InvalidInput("Jacobian requires functions and axes")
        budget.matrix(len(values),len(axes))
        return tuple(gradient(value,axes) for value in values)


def hessian(expression,variables,*,limits=None):
    with computation(limits) as budget:
        axes=_axes(variables)
        if not axes:raise InvalidInput("Hessian axes cannot be empty")
        budget.matrix(len(axes),len(axes))
        first=gradient(expression,axes)
        return tuple(tuple(differentiate(d,v) for v in axes) for d in first)


@dataclass(frozen=True)
class TaylorPolynomial:
    variable: Symbol
    point: RealExpr
    coefficients: tuple[RealExpr,...]
    polynomial: RealFunction

    def __post_init__(self):
        object.__setattr__(self,"coefficients",tuple(self.coefficients))
        if type(self.variable) is not Symbol or type(self.point) is not RealExpr or not _finite_algebraic(self.point):
            raise InvalidInput("Taylor center must be a closed exact real expression")
        if not self.coefficients or any(type(c) is not RealExpr or c.free_symbols for c in self.coefficients):
            raise InvalidInput("Taylor coefficients must be nonempty closed exact expressions")
        if type(self.polynomial) is not RealFunction or self.polynomial.variables!=(self.variable,):
            raise InvalidInput("Taylor polynomial must have its single declared axis")

    @property
    def order(self):return len(self.coefficients)-1


def _finite_algebraic(expr):
    if expr.op=="scalar":return True
    if expr.op not in ("add","mul","div","pow","sqrt","abs"):return False
    return all(_finite_algebraic(a) for a in expr.args) and intrinsic_domain(expr).contains() is Truth.TRUE


def _taylor(source,variable,point,order):
    if source.variables!=(variable,):raise UnsupportedOperation("Taylor expansion is univariate with no free parameters")
    raw_point=as_real_expr(point)
    if raw_point.free_symbols or intrinsic_domain(raw_point).contains() is not Truth.TRUE:
        raise InvalidInput("Taylor center must be a defined closed real value")
    at=simplify_expr(raw_point);n=_order(order)
    if at.free_symbols:raise InvalidInput("Taylor center must be closed")
    if not _finite_algebraic(at):raise UnsupportedOperation("Taylor centers must be finite real algebraic values")
    trace=_trace(source,variable,n)
    derivatives=(source,)+trace.derivatives
    coefficients=[];denominator=1
    for k,item in enumerate(derivatives):
        if k:
            current_budget().product(denominator,k)
            denominator*=k
        regular=conjunction(item.domain,smooth_domain(item.expression,variable))
        if regular.contains({variable:at}) is not Truth.TRUE:
            raise UnsupportedOperation("Taylor expansion requires certified regularity at its center")
        coefficients.append(simplify_expr(substitute_expr(item.expression,{variable:at})/denominator))
    polynomial=ZERO;x=as_real_expr(variable)
    for k,c in enumerate(coefficients):polynomial=polynomial+c*(x-at)**k
    value=TaylorPolynomial(variable,at,tuple(coefficients),real_function(polynomial,(variable,)))
    return value,trace


def taylor(expression,variable,point=0,order=5,*,limits=None):
    with computation(limits) as budget:
        source=_source(expression,variable);budget.inspect(source)
        return _taylor(source,variable,point,order)[0]


def _linear(expr,variable):
    """Return exact rational slope/intercept for a polynomial linear argument."""
    try:
        polynomial=_polynomial(expr,variable)
        if polynomial.degree>1:return None
        return polynomial.coefficients[1] if polynomial.degree==1 else Rational(0),polynomial.coefficients[0] if polynomial.coefficients else Rational(0)
    except (UnsupportedOperation,InvalidInput,TypeError):return None


def _legacy(expr):
    if expr.op in ("scalar","symbol"):return expr.value
    if expr.op=="add":return _legacy(expr.args[0])+_legacy(expr.args[1])
    if expr.op=="mul":return _legacy(expr.args[0])*_legacy(expr.args[1])
    if expr.op=="div":return _legacy(expr.args[0])/_legacy(expr.args[1])
    if expr.op=="pow" and expr.value>=0:return _legacy(expr.args[0])**expr.value
    raise UnsupportedOperation("Not a rational polynomial")


def _polynomial(expr,variable):
    from .polynomial import Polynomial
    return Polynomial.from_expression(_legacy(expr),variable)


def _polynomial_expr(poly):return as_real_expr(poly.to_expression())


def _elementary_primitive(op,u):
    catalogue={"sin":lambda:-cos(u),"cos":lambda:sin(u),"exp":lambda:exp(u),
               "sinh":lambda:cosh(u),"cosh":lambda:sinh(u),
               "tan":lambda:-log(abs(cos(u))),"tanh":lambda:log(cosh(u)),
               "log":lambda:u*log(u)-u,"sqrt":lambda:as_real_expr(Rational(2,3))*u*sqrt(u),
               "abs":lambda:u*abs(u)/2,
               "asin":lambda:u*asin(u)+sqrt(1-u**2),
               "acos":lambda:u*acos(u)-sqrt(1-u**2),
               "atan":lambda:u*atan(u)-log(1+u**2)/2}
    return catalogue[op]() if op in catalogue else None


def _rational_argument(expr,variable):
    try:
        a,b=_fraction_expression(expr)
        _polynomial(a,variable);_polynomial(b,variable)
        return True
    except (UnsupportedOperation,InvalidInput,TypeError):return False


def _substitution_primitive(expr,variable):
    """Match complete u' F(u) identities, with a rational inner function u."""
    from .function_checks import equivalent
    pending=[expr];visited=set()
    while pending:
        _tick();node=pending.pop()
        if node in visited:continue
        visited.add(node);pending.extend(node.args)
        if node.op in ("sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","exp","log","sqrt","abs"):
            u=node.args[0]
            if variable not in u.free_symbols or not _rational_argument(u,variable):continue
            du=_differentiate_expr(u,variable)
            if equivalent(expr,du*node):return _elementary_primitive(node.op,u)
        if node.op=="pow":
            u=node.args[0];n=node.value
            if variable not in u.free_symbols or not _rational_argument(u,variable):continue
            du=_differentiate_expr(u,variable)
            if equivalent(expr,du*node):return log(abs(u)) if n==-1 else u**(n+1)/(n+1)
            if equivalent(expr,du/(1+u**2)):return atan(u)
        if node.op=="div":
            u=node.args[1]
            if variable not in u.free_symbols or not _rational_argument(u,variable):continue
            if equivalent(expr,_differentiate_expr(u,variable)/u):return log(abs(u))
    return None


def _primitive(expr,variable):
    """Elementary catalogue; return None for an unanswered integration question."""
    _tick()
    expr=simplify_expr(expr);x=as_real_expr(variable)
    if variable not in expr.free_symbols:return expr*x
    try:
        p=_polynomial(expr,variable)
        out=ZERO
        for degree,coefficient in enumerate(p.coefficients):out=out+as_real_expr(coefficient)*x**(degree+1)/(degree+1)
        return simplify_expr(out)
    except (UnsupportedOperation,InvalidInput,TypeError):pass
    if expr.op=="add":
        a,b=(_primitive(arg,variable) for arg in expr.args)
        return None if a is None or b is None else simplify_expr(a+b)
    if expr.op=="mul":
        for constant,body in (expr.args,expr.args[::-1]):
            if variable not in constant.free_symbols:
                answer=_primitive(body,variable)
                if answer is not None:return simplify_expr(constant*answer)
        for body,elementary in (expr.args,expr.args[::-1]):
            if elementary.op!="log":continue
            linear=_linear(elementary.args[0],variable)
            if linear is None or linear[0]==Rational(0):continue
            try:p=_polynomial(body,variable)
            except (UnsupportedOperation,InvalidInput,TypeError):continue
            from .polynomial import Polynomial
            a,b=linear;u=elementary.args[0]
            primitive_poly=Polynomial((Rational(0),)+tuple(q/Rational(i+1) for i,q in enumerate(p.coefficients)),variable)
            quotient,remainder=(primitive_poly*a).divmod(Polynomial((b,a),variable))
            residual=_primitive(_polynomial_expr(quotient),variable)
            residual=residual+_polynomial_expr(remainder)*log(abs(u))/as_real_expr(a)
            return simplify_expr(_polynomial_expr(primitive_poly)*log(u)-residual)
        # Polynomial times exp/sin/cos of an affine rational argument.
        for body,elementary in (expr.args,expr.args[::-1]):
            if elementary.op not in ("exp","sin","cos"):continue
            linear=_linear(elementary.args[0],variable)
            if linear is None or linear[0]==Rational(0):continue
            try:p=_polynomial(body,variable)
            except (UnsupportedOperation,InvalidInput,TypeError):continue
            a,_=linear;u=elementary.args[0];out=ZERO;current=p
            for k in range(p.degree+1):
                coefficient=_polynomial_expr(current)/as_real_expr(a)**(k+1)
                if elementary.op=="exp":term=((-1)**k)*coefficient*exp(u)
                else:
                    # Repeated primitives: sin -> -cos -> -sin -> cos -> sin.
                    cycle=((-cos(u),-sin(u),cos(u),sin(u)) if elementary.op=="sin" else
                           (sin(u),-cos(u),-sin(u),cos(u)))
                    term=((-1)**k)*coefficient*cycle[k%4]
                out=out+term;current=current.derivative()
            return simplify_expr(out)
    if expr.op in ("sin","cos","exp","sinh","cosh","tan","tanh","log","sqrt","abs","asin","acos","atan"):
        u=expr.args[0];linear=_linear(u,variable)
        if linear is not None and linear[0]!=Rational(0):
            a,_=linear
            return simplify_expr(_elementary_primitive(expr.op,u)/as_real_expr(a))
    if expr.op=="pow":
        u=expr.args[0];linear=_linear(u,variable)
        if linear is not None and linear[0]!=Rational(0):
            a,_=linear;n=expr.value
            return simplify_expr(log(abs(u))/as_real_expr(a) if n==-1 else u**(n+1)/(as_real_expr(a)*(n+1)))
    if expr.op=="div":
        numerator,denominator=expr.args
        if variable not in numerator.free_symbols:
            linear=_linear(denominator,variable)
            if linear is not None and linear[0]!=Rational(0):
                return simplify_expr(numerator*log(abs(denominator))/as_real_expr(linear[0]))
            from .function_checks import equivalent
            if equivalent(denominator,1+x**2):return simplify_expr(numerator*atan(x))
            if denominator.op=="sqrt" and equivalent(denominator.args[0],1-x**2):
                return simplify_expr(numerator*asin(x))
        # Exact derivative-of-argument patterns, independent check below is final.
        for candidate in (denominator,):
            d=_differentiate_expr(candidate,variable)
            from .function_checks import equivalent
            if equivalent(numerator,d):return log(abs(candidate))
    return _substitution_primitive(expr,variable)


def antiderivative(expression,variable,*,limits=None):
    with computation(limits) as budget:
        source=_source(expression,variable);budget.inspect(source)
        primitive=_primitive(source.expression,variable)
        if primitive is None:
            from .function_integration import rational_fallback
            rational=rational_fallback(source)
            return UnevaluatedIntegral(source,variable) if rational is None else rational
        return real_function(primitive,source.variables,domain=conjunction(source.domain,interior_domain(source.domain,variable),smooth_domain(primitive,variable)))


@dataclass(frozen=True)
class FunctionLimitValue:
    kind: str
    value: RealExpr | None = None

    def __post_init__(self):
        if self.kind not in ("finite","positive_infinity","negative_infinity","does_not_exist"):
            raise InvalidInput("Invalid elementary limit kind")
        if (self.kind=="finite") != (type(self.value) is RealExpr):
            raise InvalidInput("Only a finite function limit has an exact expression value")
        if self.value is not None and self.value.free_symbols:raise InvalidInput("A finite limit value must be closed")


def _fraction_expression(expr):
    """Clear rational operations while treating elementary calls as atoms."""
    _tick()
    if expr.op in ("add","mul","div"):
        a,b=_fraction_expression(expr.args[0]);c,d=_fraction_expression(expr.args[1])
        if expr.op=="add":return simplify_expr(a*d+c*b),simplify_expr(b*d)
        if expr.op=="mul":return simplify_expr(a*c),simplify_expr(b*d)
        return simplify_expr(a*d),simplify_expr(b*c)
    if expr.op=="pow":
        a,b=_fraction_expression(expr.args[0]);n=expr.value
        if n<0:a,b,n=b,a,-n
        return simplify_expr(a**n),simplify_expr(b**n)
    return expr,ONE


def _scale(expr,variable):
    """Exact sums of c*x^k*log(x)^l*exp(P(x)), for x -> +infinity."""
    from .polynomial import Polynomial
    expr=simplify_expr(expr)
    if variable not in expr.free_symbols:return {((),0,0):expr}
    if expr.op=="symbol":return {((),1,0):ONE}
    if expr.op=="add":
        out=dict(_scale(expr.args[0],variable))
        for key,c in _scale(expr.args[1],variable).items():out[key]=simplify_expr(out.get(key,ZERO)+c)
        return out
    if expr.op in ("mul","div"):
        left,right=_scale(expr.args[0],variable),_scale(expr.args[1],variable)
        if expr.op=="div" and len(right)!=1:raise UnsupportedOperation("Only monomial asymptotic divisors are supported")
        out={}
        for (p,k,l),a in left.items():
            for (q,j,m),b in right.items():
                first=Polynomial((0,)+p,variable);second=Polynomial((0,)+q,variable)
                poly=first+second if expr.op=="mul" else first-second
                key=(poly.coefficients[1:],k+j if expr.op=="mul" else k-j,l+m if expr.op=="mul" else l-m)
                c=a*b if expr.op=="mul" else a/b
                out[key]=simplify_expr(out.get(key,ZERO)+c)
        return out
    if expr.op=="pow":
        base=_scale(expr.args[0],variable)
        if len(base)!=1:raise UnsupportedOperation("Only monomial asymptotic powers are supported")
        ((p,k,l),c),=base.items();n=expr.value
        return {(tuple(a*n for a in p),k*n,l*n):simplify_expr(c**n)}
    if expr.op=="exp":
        p=_polynomial(expr.args[0],variable)
        constant=p.coefficients[0] if p.coefficients else Rational(0)
        return {(p.coefficients[1:],0,0):simplify_expr(exp(as_real_expr(constant)))}
    if expr.op=="log":
        numerator,denominator=_fraction_expression(expr.args[0])
        terms=[]
        for part in (numerator,denominator):
            p=_polynomial(part,variable)
            nonzero=[(i,q) for i,q in enumerate(p.coefficients) if q!=Rational(0)]
            if len(nonzero)!=1:raise UnsupportedOperation("Logarithmic scales require a rational monomial argument")
            terms.append(nonzero[0])
        degree=terms[0][0]-terms[1][0];coefficient=terms[0][1]/terms[1][1]
        if coefficient<=0:raise UnsupportedOperation("Logarithmic scales require a positive monomial argument")
        constant=simplify_expr(log(as_real_expr(coefficient)))
        result={((),0,1):as_real_expr(degree)}
        if constant!=ZERO:result[((),0,0)]=constant
        return result
    raise UnsupportedOperation("Expression is outside exact exponential-polynomial scales")


def limit(expression,variable,point,*,side="both",limits=None):
    with computation(limits) as budget:
        budget.inspect(expression)
        from .function_limits import produce as limit_produce
        return limit_produce(expression,variable,point,side)[0]


ARGUMENTS={
    "function_differentiate":{"expression","variable","n"},
    "function_partial_diff":{"expression","variable","n"},
    "function_gradient":{"expression","variables"},
    "function_jacobian":{"expressions","variables"},
    "function_hessian":{"expression","variables"},
    "function_taylor":{"expression","variable","point","order"},
    "function_evaluate":{"expression","bindings"},
    "function_antiderivative":{"expression","variable"},
    "function_limit":{"expression","variable","point","side"},
}

TYPES={"function_derivative_trace":FunctionDerivativeTrace,"taylor_polynomial":TaylorPolynomial,
       "function_limit_value":FunctionLimitValue}


def produce(operation,args):
    if operation not in ARGUMENTS:raise UnsupportedOperation("Unknown function operation")
    if operation in ("function_differentiate","function_partial_diff"):
        source=_source(args["expression"],args["variable"])
        trace=_trace(source,args["variable"],args["n"])
        return _last(trace),(trace,)
    if operation=="function_gradient":
        axes=_axes(args["variables"])
        if not axes:raise InvalidInput("Gradient axes cannot be empty")
        source=_source(args["expression"],variables=axes)
        traces=tuple(_trace(source,v,1) for v in axes)
        return tuple(_last(t) for t in traces),traces
    if operation=="function_jacobian":
        axes=_axes(args["variables"]);sources=tuple(_source(e,variables=axes) for e in args["expressions"])
        if not axes or not sources:raise InvalidInput("Jacobian requires functions and axes")
        current_budget().matrix(len(sources),len(axes))
        traces=tuple(_trace(s,v,1) for s in sources for v in axes)
        return tuple(tuple(_last(traces[i*len(axes)+j]) for j in range(len(axes))) for i in range(len(sources))),traces
    if operation=="function_hessian":
        axes=_axes(args["variables"])
        if not axes:raise InvalidInput("Hessian axes cannot be empty")
        current_budget().matrix(len(axes),len(axes));source=_source(args["expression"],variables=axes)
        first=tuple(_trace(source,v,1) for v in axes)
        second=tuple(_trace(_last(t),v,1) for t in first for v in axes)
        return tuple(tuple(_last(second[i*len(axes)+j]) for j in range(len(axes))) for i in range(len(axes))),first+second
    if operation=="function_taylor":
        source=_source(args["expression"],args["variable"])
        value,trace=_taylor(source,args["variable"],args["point"],args["order"])
        return value,(trace,)
    if operation=="function_evaluate":
        source=_source(args["expression"])
        return source.evaluate(args["bindings"]),()
    if operation=="function_antiderivative":
        source=_source(args["expression"],args["variable"])
        value=antiderivative(source,args["variable"])
        return value,()
    if operation=="function_limit":
        from .function_limits import produce as limit_produce
        return limit_produce(args["expression"],args["variable"],args["point"],args["side"])
    raise UnsupportedOperation("Unknown function operation")


def _check_trace(trace,source,variable,n):
    from .function_checks import jet,equivalent
    if type(trace) is not FunctionDerivativeTrace or trace.source!=source or trace.variable!=variable or len(trace.derivatives)!=n:return False
    previous=source
    for claimed in trace.derivatives:
        if claimed.variables!=source.variables:return False
        expected=jet(previous.expression,variable,as_real_expr(variable),1)[1]
        if not equivalent(expected,claimed.expression):return False
        required=conjunction(previous.domain,interior_domain(previous.domain,variable),smooth_domain(previous.expression,variable),intrinsic_domain(claimed.expression))
        if claimed.domain!=required:return False
        previous=claimed
    return True


def check_operation(operation,args,value,evidence):
    """Check source linkage, shape, domains and independent coefficient identities."""
    from .function_checks import jet,equivalent
    if type(evidence) is not tuple:return False
    if operation in ("function_differentiate","function_partial_diff"):
        source=_source(args["expression"],args["variable"]);n=_order(args["n"])
        return len(evidence)==1 and _check_trace(evidence[0],source,args["variable"],n) and value==_last(evidence[0])
    if operation in ("function_gradient","function_jacobian","function_hessian"):
        axes=_axes(args["variables"])
        if not axes:return False
        if operation=="function_jacobian":sources=tuple(_source(e,variables=axes) for e in args["expressions"])
        else:sources=(_source(args["expression"],variables=axes),)
        expected_count=len(axes)*len(sources)
        if operation=="function_hessian":expected_count=len(axes)+len(axes)**2
        if len(evidence)!=expected_count:return False
        first_count=len(axes)*len(sources)
        for i,source in enumerate(sources):
            for j,variable in enumerate(axes):
                if not _check_trace(evidence[i*len(axes)+j],source,variable,1):return False
        if operation=="function_hessian":
            for i in range(len(axes)):
                for j,variable in enumerate(axes):
                    if not _check_trace(evidence[len(axes)+i*len(axes)+j],_last(evidence[i]),variable,1):return False
            expected=tuple(tuple(_last(evidence[len(axes)+i*len(axes)+j]) for j in range(len(axes))) for i in range(len(axes)))
        elif operation=="function_gradient":expected=tuple(_last(t) for t in evidence)
        else:expected=tuple(tuple(_last(evidence[i*len(axes)+j]) for j in range(len(axes))) for i in range(len(sources)))
        return type(value) is tuple and value==expected
    if operation=="function_taylor":
        source=_source(args["expression"],args["variable"]);variable=args["variable"];n=_order(args["order"])
        raw_point=as_real_expr(args["point"])
        if raw_point.free_symbols or intrinsic_domain(raw_point).contains() is not Truth.TRUE:return False
        point=simplify_expr(raw_point)
        if type(value) is not TaylorPolynomial or not _finite_algebraic(point) or source.variables!=(variable,) or value.variable!=variable or value.point!=point or value.order!=n:return False
        if len(evidence)!=1 or not _check_trace(evidence[0],source,variable,n):return False
        for item in (source,)+evidence[0].derivatives:
            if conjunction(item.domain,smooth_domain(item.expression,variable)).contains({variable:point}) is not Truth.TRUE:return False
        coefficients=jet(source.expression,variable,point,n)
        if any(not equivalent(a,b) for a,b in zip(coefficients,value.coefficients)):return False
        rebuilt=ZERO;x=as_real_expr(variable)
        for k,c in enumerate(value.coefficients):rebuilt=rebuilt+c*(x-point)**k
        return value.polynomial.domain==intrinsic_domain(simplify_expr(rebuilt)) and equivalent(rebuilt,value.polynomial.expression)
    if operation=="function_evaluate":
        from .functions import _bindings
        source=_source(args["expression"]);bindings=_bindings(args["bindings"])
        if evidence or any(expression.free_symbols for expression in bindings.values()) or not set(source.variables)<=set(bindings) or source.domain.contains(bindings) is not Truth.TRUE:return False
        return type(value) is RealExpr and not value.free_symbols and equivalent(substitute_expr(source.expression,bindings),value)
    if operation=="function_antiderivative":
        source=_source(args["expression"],args["variable"])
        if evidence:return False
        if type(value) is not RealFunction:
            from .function_integration import check_rational_fallback
            return check_rational_fallback(source,value)
        if value.variables!=source.variables:return False
        if value.domain!=conjunction(source.domain,interior_domain(source.domain,args["variable"]),intrinsic_domain(value.expression),smooth_domain(value.expression,args["variable"])):return False
        # Independent formal coefficient propagation verifies the derivative.
        return equivalent(jet(value.expression,args["variable"],as_real_expr(args["variable"]),1)[1],source.expression)
    if operation=="function_limit":
        # Limit records carry explicit exact local/asymptotic evidence in the
        # next layer; unsupported certificates must never be accepted here.
        return _check_limit(args,value,evidence)
    return False


def is_partial(operation,args,value):
    if operation in ("function_differentiate","function_partial_diff"):
        source=_source(args["expression"],args["variable"])
        return args.get("n",1)>0 and type(value) is RealFunction and value.domain!=source.domain
    if operation in ("function_gradient","function_hessian"):
        source=_source(args["expression"],variables=args["variables"])
        components=value if operation=="function_gradient" else tuple(item for row in value for item in row)
        return any(item.domain!=source.domain for item in components)
    if operation=="function_jacobian":
        return any(item.domain!=_source(expression,variables=args["variables"]).domain
                   for expression,row in zip(args["expressions"],value) for item in row)
    if operation=="function_antiderivative" and type(value) is RealFunction:
        return value.domain!=_source(args["expression"],args["variable"]).domain
    return False


def _check_limit(args,value,evidence):
    from .function_limits import check
    return check(args,value,evidence)


from .function_limits import TYPES as _LIMIT_TYPES
TYPES.update(_LIMIT_TYPES)
