"""Geometry and vector probes: distances, areas, intersections and products of vectors."""

from . import Probe, exact, exact_tuple, planned

GROUP = "geometry"
# The catalogue (section 4) names these operations but not their areas.
_GEOMETRY = ("geometry", "vector")
_VECTOR = ("vector", "linalg", "geometry")


def _vector_area(mf):
    """The operations of the vector area; while it has none the request is unsupported."""
    names = tuple(mf.vector.__all__)
    if not names:
        raise mf.UnsupportedOperation("the vector area has no operation yet")
    return names


PROBES = (
    Probe("geometry_distance", GROUP, "What is the distance between (0, 0) and (3, 4)?",
          lambda mf: planned(mf, "distance", *_GEOMETRY)((0, 0), (3, 4)), lambda v: exact(v) == 5),
    Probe("geometry_line_intersection", GROUP,
          "Where does the line through (0, 0) and (1, 1) meet the line through (0, 1) and (1, 0)?",
          lambda mf: planned(mf, "line_intersection", *_GEOMETRY)((0, 0), (1, 1), (0, 1), (1, 0))),
    Probe("geometry_triangle_area", GROUP, "What is the area of the triangle (0, 0), (4, 0), (0, 3)?",
          lambda mf: planned(mf, "polygon_area", *_GEOMETRY)(mf.Matrix(((0, 0), (4, 0), (0, 3)))),
          lambda v: exact(v) == 6),
    Probe("geometry_dot_product", GROUP, "What is the dot product of (1, 2, 3) and (4, 5, 6)?",
          lambda mf: planned(mf, "dot", *_VECTOR)((1, 2, 3), (4, 5, 6)), lambda v: exact(v) == 32),
    Probe("geometry_cross_product", GROUP, "What is the cross product of (1, 2, 3) and (4, 5, 6)?",
          lambda mf: planned(mf, "cross", *_VECTOR)((1, 2, 3), (4, 5, 6)),
          lambda v: exact_tuple(v) == (-3, 6, -3)),
    Probe("geometry_vector_area", GROUP, "Which operations does the vector area offer?", _vector_area),
)
