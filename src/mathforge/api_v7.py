"""Public v0.7 operations; all computations share source-bearing Result records.

tools/generate_api.py generates the wrappers from mathforge.registry into mathforge._generated.structured.
This module re-exports exactly the published v0.7 names. Operations added later appear in the area modules
and in mathforge.operations, which import them from mathforge._generated, never here.
"""
from __future__ import annotations
from ._generated.structured import (
    integer_gcd, integer_lcm, integer_xgcd, mod_inverse, crt, is_prime, factor_integer, divisors, totient,
    factorial, fibonacci, catalan, binomial, permutations_count, stirling, multinomial,
    complex_algebraic, complex_add, complex_sub, complex_mul, complex_div, complex_conjugate, complex_real,
    complex_imag, complex_norm_squared, complex_sqrt, complex_pow, quadratic_roots,
    bernoulli, binomial_distribution, discrete_uniform, hypergeometric, probability, condition, cdf, moment,
    expectation, distribution_variance, mean, median, variance, stddev, covariance,
    trace, rowspace, columnspace, lu, least_squares, rational_antiderivative, rational_integrate,
    function_differentiate, function_partial_diff, function_gradient, function_jacobian, function_hessian,
    taylor, function_evaluate, function_antiderivative, function_integrate, function_limit,
)

__all__ = ["integer_gcd","integer_lcm","integer_xgcd","mod_inverse","crt","is_prime","factor_integer","divisors","totient","factorial","fibonacci","catalan","binomial","permutations_count","stirling","multinomial","complex_algebraic","complex_add","complex_sub","complex_mul","complex_div","complex_conjugate","complex_real","complex_imag","complex_norm_squared","complex_sqrt","complex_pow","quadratic_roots","bernoulli","binomial_distribution","discrete_uniform","hypergeometric","probability","condition","cdf","moment","expectation","distribution_variance","mean","median","variance","stddev","covariance","trace","rowspace","columnspace","lu","least_squares","rational_antiderivative","rational_integrate","function_differentiate","function_partial_diff","function_gradient","function_jacobian","function_hessian","taylor","function_evaluate","function_antiderivative","function_integrate","function_limit"]
