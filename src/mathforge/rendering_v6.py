"""Presentation for v6 records, using the shared renderer budget.

The generic deferral records render here. A family record renders through the RENDER functions of the family
module that defines its class, whose text the renderer bounds like any other, or else as its class name and
fields.
"""
from dataclasses import fields
import sys

from . import registry
from .deferred import Reason, Unevaluated


def supported(value):
    cls = type(value)
    return registry.record_module(cls.__module__) and cls in registry.record_types(6).values()


def children(value):
    """What the renderer walks below a v6 record, counting nodes and collecting Symbols: the reason of an
    Unevaluated, nothing below a Reason, and the fields of any other record."""
    if type(value) is Unevaluated:
        return (value.reason,)
    if type(value) is Reason:
        return ()
    return tuple(getattr(value, field.name) for field in fields(value))


def _family_functions(cls):
    """The (text, LaTeX) pair declared for cls in the RENDER mapping of its family module, or None."""
    return getattr(sys.modules.get(cls.__module__), "RENDER", {}).get(cls)


def render(value, renderer, child):
    label, parens = renderer.label, renderer.parens
    if type(value) is Reason:
        return label("Reason") + parens(renderer.join((label(text) for text in (value.code, value.detail) if text),
                                                      ", "))
    if type(value) is Unevaluated:
        return label("Unevaluated") + parens(renderer.join(
            (label(value.kind), label(value.request.operation), child(value.reason)), ", "))
    functions = _family_functions(type(value))
    if functions is not None:
        text, latex = functions
        return (latex if renderer.latex else text)(value)
    return label(type(value).__name__) + parens(renderer.join(
        (label(field.name) + "=" + child(getattr(value, field.name)) for field in fields(value)), ", "))
