"""Independent checks of the lean complex scalar operations (mathforge.lean.algebra_complex).

``gaussian`` checks answers on Gaussian rationals, elements of Q(i), by recomputing each answer another way in
integers and comparing by cross-multiplication. ``surd`` checks answers whose parts are quadratic surds, elements of
a field Q(sqrt b_1, ..., sqrt b_t) computed with mathforge.trusted.surd: every part named by a descriptor is checked
against its element, products, quotients and powers are recomputed by other formulas, and each of them is also
checked by fixed-point interval arithmetic on the operands' enclosures, which never multiplies two field elements.
Neither module imports the producer or the record classes: a RealAlgebraicRoot is recognized by its module and
qualified name, and a ComplexAlgebraic is read through its two parts.
"""
