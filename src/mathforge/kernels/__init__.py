"""Producer algorithms: factorization, gcd algorithms, Groebner bases, root-isolation strategies, the
integration rule engine, ODE solvers and limit algorithms. Producers may build on mathforge.trusted, but
the independent checkers in mathforge.certify never import this package, directly or through any other
module, so that a checker cannot reuse the algorithm whose result it is verifying;
tools/check_layering.py enforces this transitively.
"""
