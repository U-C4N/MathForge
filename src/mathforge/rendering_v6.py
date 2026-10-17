"""Presentation for v6 records through the shared renderer.

Every v6 record renders with one function render(value, renderer, child) -> str, the signature of
rendering_v5.render. renderer.latex selects LaTeX; renderer.label, parens, join and integer apply the shared
budget; child(nested) renders a nested value with the symbol aliases and the budget of the whole rendering.
The deferral records use the functions below. A family record uses the function that its family declares in
RENDER (registry.record_renderers()); any other v6 record is shown as its class name and fields.
"""
from dataclasses import fields

from . import registry
from .deferred import Reason, Unevaluated


def supported(value):
    cls = type(value)  # The static module test comes first, so that other values never build the registry.
    return registry.record_module(cls.__module__) and cls in registry.record_renderers()


def children(value):
    """What the renderer walks below a v6 record, counting nodes and collecting Symbols: the reason of an
    Unevaluated, nothing below a Reason, and the fields of any other record."""
    if type(value) is Unevaluated:
        return (value.reason,)
    if type(value) is Reason:
        return ()
    return tuple(getattr(value, field.name) for field in fields(value))


def _reason(value, renderer, child):
    return renderer.label("Reason") + renderer.parens(renderer.join(
        (renderer.label(text) for text in (value.code, value.detail) if text), ", "))


def _unevaluated(value, renderer, child):
    return renderer.label("Unevaluated") + renderer.parens(renderer.join(
        (renderer.label(value.kind), renderer.label(value.request.operation), child(value.reason)), ", "))


def _fields(value, renderer, child):
    return renderer.label(type(value).__name__) + renderer.parens(renderer.join(
        (renderer.label(field.name) + "=" + child(getattr(value, field.name)) for field in fields(value)), ", "))


_DEFERRALS = {Reason: _reason, Unevaluated: _unevaluated}  # Shaped like a family's RENDER.


def render(value, renderer, child):
    function = _DEFERRALS.get(type(value)) or registry.record_renderers()[type(value)] or _fields
    return function(value, renderer, child)
