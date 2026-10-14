"""Proper finite integration with independently checked interval sign charts.

Every source and primitive restriction must hold on the entire closed interval.
Unsupported domain proofs remain unevaluated, including improper integrals.
"""
from .errors import InvalidInput, UnsupportedOperation
from .limits import computation, current_budget
from .model import Rational, Truth
from .functions import (RealFunction, UnevaluatedIntegral, real_function,
                        conjunction, condition, as_real_expr, simplify_expr,
                        substitute_expr)

ARGUMENTS = {"function_integrate": {"expression", "variable", "lower", "upper"}}
TYPES = {}


def rational_source(source):
    """Convert Q(x) values; original restrictions are retained separately."""
    from .function_calculus import _fraction_expression, _polynomial
    from .rational_function import rational_function
    if len(source.variables) != 1:
        raise UnsupportedOperation("Rational integration is univariate")
    x, = source.variables
    numerator, denominator = _fraction_expression(source.expression)
    return rational_function(_polynomial(numerator, x), _polynomial(denominator, x), variable=x)


def rational_fallback(source):
    from .rational_integration import _primitive
    try:
        rational = rational_source(source)
    except (InvalidInput, UnsupportedOperation):
        return None
    return _primitive(rational, source.domain)


def check_rational_fallback(source, value):
    from .rational_integration import _check_primitive
    return _check_primitive(rational_source(source), value, source.domain)


def _reduce_predicate(predicate):
    if predicate.kind in ("true", "false", "and"): return predicate
    expr = predicate.operands[0]
    if expr.op == "abs" or (expr.op == "pow" and expr.value > 0 and expr.value % 2 == 0):
        if predicate.kind == "nonnegative":
            from .functions import TRUE
            return TRUE
        return condition("nonzero", expr.args[0])
    return predicate


def _problem(predicate, variable):
    from .contracts import Inequality
    rational = rational_source(real_function(predicate.operands[0], (variable,)))
    # u != 0 iff u^2 > 0, including rational-function exclusions.
    if predicate.kind == "nonzero": rational = rational * rational
    return Inequality(rational, "ge" if predicate.kind == "nonnegative" else "gt", Rational(0))


def _covers(solution, lower, upper):
    from .intervals import IntervalSet
    from .algebraic import compare_real
    if compare_real(lower, upper) > 0: lower, upper = upper, lower
    return any(part.contains(lower) and part.contains(upper)
               for part in IntervalSet.from_solution_set(solution).intervals)


def interval_proof(predicate, variable, lower, upper):
    """Build an explicit whole-interval domain certificate."""
    current_budget().tick()
    reduced = _reduce_predicate(predicate)
    if reduced != predicate: return ("reduced", interval_proof(reduced, variable, lower, upper))
    if predicate.kind == "true": return ("true",)
    if predicate.kind == "false": raise UnsupportedOperation("The integration interval is outside the domain")
    if predicate.kind == "and":
        return ("and", tuple(interval_proof(p, variable, lower, upper) for p in predicate.operands))
    if predicate.contains() is Truth.TRUE: return ("global",)
    from .real_sets import solve_real_set
    problem = _problem(predicate, variable)
    value, proof = solve_real_set(problem, variable)
    if not _covers(value, lower, upper):
        raise UnsupportedOperation("The entire closed interval must belong to the function domain")
    return ("sign_chart", value, proof)


def check_interval(predicate, variable, lower, upper, proof):
    """Check root certificates and signs without invoking sign-chart solvers."""
    current_budget().tick()
    if type(proof) is not tuple or not proof: return False
    reduced = _reduce_predicate(predicate)
    if reduced != predicate:
        return len(proof) == 2 and proof[0] == "reduced" and check_interval(reduced, variable, lower, upper, proof[1])
    if predicate.kind == "true": return proof == ("true",)
    if predicate.kind == "false": return False
    if predicate.kind == "and":
        return (len(proof) == 2 and proof[0] == "and" and type(proof[1]) is tuple
                and len(proof[1]) == len(predicate.operands)
                and all(check_interval(p, variable, lower, upper, c) for p, c in zip(predicate.operands, proof[1])))
    if proof == ("global",): return predicate.contains() is Truth.TRUE
    if len(proof) != 3 or proof[0] != "sign_chart": return False
    from .real_sets import check_solution
    return (check_solution(_problem(predicate, variable), variable, proof[1], proof[2])
            and _covers(proof[1], lower, upper))


def _bounds(args):
    from .algebraic_arithmetic import produce as normalize
    from .model import as_expression
    return tuple(normalize("algebraic", {"expression": as_expression(args[key])}) for key in ("lower", "upper"))


def _check_bounds(args, lower, upper, proofs):
    from .algebraic_arithmetic import check_operation
    from .model import as_expression
    return type(proofs) is tuple and len(proofs) == 2 and all(
        check_operation("algebraic", {"expression": as_expression(args[key])}, value, proof)
        for key, value, proof in zip(("lower", "upper"), (lower, upper), proofs))


def _region(source, primitive, variable):
    from .function_calculus import smooth_domain
    return conjunction(source.domain, primitive.domain, smooth_domain(primitive.expression, variable))


def _difference(primitive, variable, lower, upper):
    return simplify_expr(substitute_expr(primitive.expression, {variable: upper})
                         - substitute_expr(primitive.expression, {variable: lower}))


def finish_integral(source, primitive, args):
    """Producer shared by the elementary and rational entry points."""
    variable = args["variable"]
    (lower, lo_proof), (upper, hi_proof) = _bounds(args)
    from .rational_integration import RationalPrimitive
    formal = primitive if type(primitive) is RationalPrimitive else None
    from .model import as_expression
    unanswered = UnevaluatedIntegral(source, variable,
                                     as_expression(args["lower"]), as_expression(args["upper"]))
    try:
        real = primitive.as_function() if formal is not None else primitive
        if type(real) is not RealFunction:
            return unanswered, ()
        proof = interval_proof(_region(source, real, variable), variable, lower, upper)
    except UnsupportedOperation:
        return unanswered, ()
    return _difference(real, variable, lower, upper), (real, formal, lower, upper, (lo_proof, hi_proof), proof)


def check_integral(source, args, value, evidence, *, rational=None):
    if type(evidence) is not tuple or len(evidence) != 6: return False
    primitive, formal, lower, upper, bounds_proof, domain_proof = evidence
    if type(primitive) is not RealFunction or primitive.variables != source.variables: return False
    if not _check_bounds(args, lower, upper, bounds_proof): return False
    from .function_checks import jet, equivalent
    variable = args["variable"]
    if formal is not None:
        from .rational_integration import _check_primitive
        if not (_check_primitive(rational, formal) if rational is not None else check_rational_fallback(source, formal)): return False
    # Verify a supplied expression, never rerun an antiderivative algorithm.
    if not equivalent(jet(primitive.expression, variable, as_real_expr(variable), 1)[1], source.expression): return False
    if not check_interval(_region(source, primitive, variable), variable, lower, upper, domain_proof): return False
    return value == _difference(primitive, variable, lower, upper)


def produce(operation, args):
    from .function_calculus import _source, antiderivative
    with computation():
        if operation != "function_integrate": raise UnsupportedOperation("Unknown function integral operation")
        source = _source(args["expression"], args["variable"])
        if source.variables != (args["variable"],):
            raise UnsupportedOperation("Definite integration is univariate with no free parameters")
        primitive = antiderivative(source, args["variable"])
        return finish_integral(source, primitive, args)


def check_operation(operation, args, value, evidence):
    from .function_calculus import _source
    with computation():
        return operation == "function_integrate" and check_integral(
            _source(args["expression"], args["variable"]), args, value, evidence)
