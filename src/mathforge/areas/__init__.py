"""Producer families grouped by mathematical area, one module per family (areas/<area>/<family>.py).
They are producers like mathforge.kernels: they may import mathforge.trusted and mathforge.kernels, but
the independent checkers in mathforge.certify never import them, directly or through any other module,
so a checker cannot reuse the code whose result it is verifying; tools/check_layering.py enforces this
transitively.
"""
