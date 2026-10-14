"""Presentation for allowlisted v0.7 values, using the shared renderer budget."""
from dataclasses import fields


def supported(value):
    from .dispatch_v5 import record_types
    return type(value) in record_types().values()


def children(value):
    return tuple(getattr(value,f.name) for f in fields(value))


def render(value, renderer, child):
    from .functions import RealExpr,RealFunction,DomainPredicate,UnevaluatedIntegral,UnevaluatedLimit
    from .complex_numbers import ComplexAlgebraic,ComplexRoot
    from .number_theory import IntegerFactorization,CRTResult
    from .rational_integration import RationalPrimitive,LogRootSum
    from .probability import FiniteDistribution
    from .linear_extensions import LUDecomposition,LeastSquaresResult
    from .function_calculus import TaylorPolynomial,FunctionLimitValue
    latex=renderer.latex
    label=renderer.label
    parens=renderer.parens
    if type(value) is RealExpr:
        op=value.op
        if op in ("scalar","symbol"):return child(value.value)
        if op=="pi":return r"\pi" if latex else "pi"
        if op=="E":return "e" if latex else "E"
        a=child(value.args[0])
        if op=="pow":return parens(a)+(r"^{"+renderer.integer(value.value)+"}" if latex else "^"+renderer.integer(value.value))
        if op in ("add","mul","div"):
            b=child(value.args[1])
            if op=="div":return r"\frac{"+a+"}{"+b+"}" if latex else parens(a)+"/"+parens(b)
            return parens(a+(" + " if op=="add" else r" \cdot " if latex else "*")+b)
        if op=="sqrt":return r"\sqrt{"+a+"}" if latex else "sqrt("+a+")"
        if op=="abs":return r"\left|"+a+r"\right|" if latex else "abs("+a+")"
        return (r"\operatorname{"+op+r"}\left("+a+r"\right)") if latex else op+"("+a+")"
    if type(value) is DomainPredicate:
        if value.kind in ("true","false"):return label(value.kind)
        if value.kind=="and":return renderer.join((child(x) for x in value.operands),r" \land " if latex else " and ")
        relation={"positive":">","nonnegative":r"\ge" if latex else ">=","nonzero":r"\ne" if latex else "!="}[value.kind]
        return child(value.operands[0])+" "+relation+" 0"
    if type(value) is RealFunction:
        text=child(value.expression)
        if value.domain.kind!="true":text+=(r"\quad\text{where } " if latex else " where ")+child(value.domain)
        return text
    if type(value) is ComplexAlgebraic:
        return parens(child(value.real)+" + "+child(value.imag)+(r"\,i" if latex else "*i"))
    if type(value) is ComplexRoot:
        return child(value.root)+"; "+label("multiplicity=")+renderer.integer(value.multiplicity)
    if type(value) is IntegerFactorization:
        return child(value.sign)+(r"\cdot " if latex else "*")+renderer.join(
            (child(p)+(r"^{"+child(k)+"}" if latex else "^"+child(k)) for p,k in value.factors),r"\cdot " if latex else "*") if value.factors else child(value.sign)
    if type(value) is CRTResult:
        return label("inconsistent congruences") if value.residue is None else child(value.residue)+(r"\pmod{"+child(value.modulus)+"}" if latex else " mod "+child(value.modulus))
    if type(value) is FiniteDistribution:
        return label("PMF")+parens(renderer.join((child(x)+": "+child(p) for x,p in zip(value.outcomes,value.probabilities)),", "))
    if type(value) is LUDecomposition:
        return renderer.join((label(key)+" = "+child(getattr(value,key)) for key in ("P","L","U")),r";\quad " if latex else "; ")
    if type(value) is LeastSquaresResult:
        return renderer.join((label(key)+" = "+child(getattr(value,key)) for key in ("particular","basis","residual")),r";\quad " if latex else "; ")
    if type(value) is TaylorPolynomial:
        return child(value.polynomial)+"; "+label("Taylor order=")+child(value.order)+", "+label("center=")+child(value.point)
    if type(value) is FunctionLimitValue:
        if value.kind=="finite":return child(value.value)
        return label(value.kind.replace("_"," "))
    if type(value) is LogRootSum:
        return label("LogRootSum")+parens(label("q=")+child(value.residue_polynomial)+", "+label("h=")+child(value.logarithm_coefficients))
    if type(value) is RationalPrimitive:
        text=renderer.join((child(term) for term in (value.rational_part,)+value.log_terms), " + ")
        text+="; "+label("formal primitive, locally constant branch choices")
        if value.domain.kind!="true":
            text+=(r"\quad\text{where } " if latex else " where ")+child(value.domain)
        return text
    if type(value) is UnevaluatedIntegral:
        if latex:
            bounds="" if value.lower is None else "_{"+child(value.lower)+"}^{"+child(value.upper)+"}"
            return label("unevaluated")+r"\quad\int"+bounds+" "+child(value.expression)+r"\,d"+child(value.variable)
        return ("UnevaluatedIntegral("+child(value.expression)+", "+child(value.variable)+
            ("" if value.lower is None else ", "+child(value.lower)+", "+child(value.upper))+")")
    if type(value) is UnevaluatedLimit:
        return label("UnevaluatedLimit")+parens(child(value.expression)+", "+child(value.variable)+" -> "+child(value.point)+", "+label(value.side))
    return label(type(value).__name__)+parens(renderer.join(
        (label(f.name)+"="+child(getattr(value,f.name)) for f in fields(value)),", "))
