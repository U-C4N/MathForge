"""Independent checkers, one module per family (certify/<area>/<family>.py). A checker verifies a claim
with a certificate or with a different algorithm (a gcd is checked through its Bezout product, never
recomputed), so it may use only the trusted layer and its own code: modules here may import only
mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model.
tools/check_layering.py follows imports transitively through each of those modules and fails on any
path from a checker to producer code (mathforge.kernels, mathforge.areas and the rest).
"""
