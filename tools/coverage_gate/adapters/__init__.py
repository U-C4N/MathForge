"""Row adapters for the depth criterion: one function per inventory row, which poses that row's SymPy reference examples
to its MathForge operations (coverage_gate.criteria describes the protocol).

Each area module registers its adapters in criteria.ADAPTERS when it is imported; criteria imports this package last,
so importing criteria makes every adapter available. A new area module joins the imports at the end of this file. The
adapters import SymPy, a development dependency, only when they run.

An example is SymPy's own documentation or test code, frozen with the inventory. example_call executes its setup in this
process, as the harvester's worker did when it recorded the example (but without the worker's isolation), to evaluate
the call that the example makes.
"""

import ast
from typing import Any

from ..examples import Example

__all__ = ["example_call"]


def example_call(example: Example) -> tuple[Any, tuple[Any, ...], dict[str, Any]] | None:
    """The function, positional arguments and keyword arguments of an example whose code is one call f(a, ..., k=v),
    each evaluated, in Python's order, in a fresh namespace that holds `from sympy import *` and the example's setup.
    None for code of any other shape: an operator expression, a name, a starred or ** argument. f may be an alias that
    the setup imports, or a method of another call's value."""
    try:
        call = ast.parse(example.code, mode="eval").body
    except SyntaxError:
        return None
    if (not isinstance(call, ast.Call) or any(isinstance(argument, ast.Starred) for argument in call.args)
            or any(keyword.arg is None for keyword in call.keywords)):
        return None
    namespace: dict[str, Any] = {}
    exec("from sympy import *", namespace)
    exec(example.setup, namespace)

    def value(node: ast.expr) -> Any:
        return eval(compile(ast.Expression(node), "<example>", "eval"), namespace)

    function = value(call.func)
    arguments = tuple(value(argument) for argument in call.args)
    return function, arguments, {str(keyword.arg): value(keyword.value) for keyword in call.keywords}


from . import ntheory  # noqa: E402,F401  The area modules use example_call.
