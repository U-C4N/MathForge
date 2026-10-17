"""Small trusted exact arithmetic that producers and independent checkers share: integer and rational
arithmetic, dense and sparse polynomial arithmetic (add, multiply, divide, evaluate), matrix
multiplication, number-field arithmetic, truncated-series arithmetic, the Expr CF1 constructors and the
identity and zero-test layers. A mistake here would corrupt both sides of a verification silently, so
keep this layer small and cover it densely with property tests and SymPy differential tests. It must
never import producer code (mathforge.kernels, mathforge.areas): tools/check_layering.py follows the
imports of every checker through this package and fails when they lead to a producer.
"""
