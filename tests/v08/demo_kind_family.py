"""A throwaway v6 family whose produce() counts its calls, for the argument-kind checks of operations_v6.run.

``registry._testing_families(demo_kind_family)`` registers it; production never does. demo_kinds takes one
argument of each kind that run() checks before produce(), and answers the number of calls so far.
"""

import sys

from mathforge.registry import Argument, OperationSpec

ARGUMENTS = (Argument("n", "int", None, "int"), Argument("q", "Rational | int", None, "rational"),
             Argument("counts", "", None, "int_list"), Argument("data", "", None, "rational_list"),
             Argument("A", "Matrix", None, "matrix"), Argument("variable", "Symbol", None, "symbol"))
SPECS = (OperationSpec("demo_kinds", "discrete", ARGUMENTS, "int", "Count the calls that reach the producer.",
                       "tests only", version=6, mcp=False),)
TYPES: dict[str, type] = {}
CHECKER = sys.modules[__name__]  # Only a testing family may check itself.
CALLS: list[dict] = []


def produce(operation, args):
    CALLS.append(dict(args))
    return len(CALLS), ()


def check_operation(operation, args, value, evidence):
    return operation == "demo_kinds" and evidence == () and value == len(CALLS)
