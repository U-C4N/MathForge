# Adding an operation family

Operations added since v0.8 come in families: a producer module that computes the values and an independent
checker that verifies them. Everything else follows from the family through the registry: the generated
functions of `mathforge.operations` and of the area module, the JSON v6 records, the MCP area tools and the
catalog. The pilot family, `mathforge.areas.ntheory.continued_fraction`, with its checker
`mathforge.certify.ntheory.continued_fraction`, is the template; its docstring spells out the same contract.

```python
import mathforge as mf
from mathforge import registry
from mathforge.areas import ntheory
from mathforge.areas.ntheory import continued_fraction as family

assert family.__name__ in ntheory.FAMILIES and family.__name__ in registry.v6_families()
assert all(hasattr(family, name) for name in registry.V6_CONTRACT)
assert family.CHECKER.startswith(registry.CHECKER_PACKAGE + ".")
assert [(spec.name, spec.version) for spec in family.SPECS] == [("continued_fraction", 6)]

value, evidence = family.produce("continued_fraction", {"value": mf.Rational(415, 93)})
assert value == family.ContinuedFraction((4, 2, 6, 7)) and evidence == ()
checker = registry.checker("continued_fraction")
assert checker.check_operation("continued_fraction", {"value": mf.Rational(415, 93)}, value, evidence) is True
assert checker.check_operation("continued_fraction", {"value": mf.Rational(415, 92)}, value, evidence) is False

result = mf.operations.continued_fraction(mf.Rational(415, 93))
assert result.steps[0].rule == "v6.continued_fraction" and result.request.version == 6
```

## Write the family module

A family lives in `src/mathforge/areas/<area>/<family>.py`, and its full module name joins `FAMILIES`, the tuple
that its area package declares in `src/mathforge/areas/<area>/__init__.py`; `registry.v6_families()` reads the
`FAMILIES` of every area package. The module defines:

- `SPECS`: its operations, each a `registry.OperationSpec` with `version=6`: the name, one of the 18 areas, the
  arguments (name, annotation text, default text, MCP argument kind), the return type and the catalog texts.
- `TYPES`: the record classes of its values by tag, defined in this module. A record serializes as `"v6_"` plus
  its tag.
- `CHECKER`: the import path of its checker, a string below `mathforge.certify`.
- `produce(operation, args)`: it returns `(value, evidence)`, where `value` may be an `Unevaluated` record of
  `mathforge.deferred` when the family reaches no verified answer. `operations_v6.run` has already checked the
  argument names and calls it inside the request's computation budget.
- `RENDER`, optionally: a mapping from a record class of `TYPES` to `render(value, renderer, child) -> str`, the
  signature of the shared renderer.
- `SUPERSEDES`, optionally: the v0.7 operations whose names its `SPECS` take over (below).

Keep the module light. The registry imports every family module to read its specs, and a lookup such as
`registry.specs()` must stay cheap, so the module declares its specs and records and imports its kernels inside
`produce`. The checker loads on its first `registry.checker(...)` call, never with the specs.

Before `produce` runs, `operations_v6.run` rejects an argument whose exact type its kind does not admit: `int` takes
an `int`; `rational` an `int` or a `Rational`; `int_list` and `rational_list` a tuple or list of those; `matrix` a
`Matrix`; `symbol` a `Symbol`. A `bool` or a `float` is never an exact number. The family checks the other kinds,
and everything else about its inputs: `produce` validates them first and raises `InvalidInput` for those it
rejects, testing `type(value) is int`, never `isinstance`. Once the inputs are validated, a family builds records
only from validated data: an `InvalidInput` raised later still reads as rejected input. Any other exception, apart
from `UnsupportedOperation` and `ResourceLimitError`, is reported as the producer's `internal_error`. `run` copies
an `expression` (or `A`) argument into the result, where a raw `int` serializes but does not load again: the first
family that takes such an argument coerces an `int` to a `Rational` before it reaches the `Result`, in
`operations_v6.source_context`, and round-trips its results in a test.

Every request that `operations_v6.run` builds carries its format version, `version=6`, and its document says so.
A result records the `expression` (or `A`) argument and a `Symbol` variable, as `for_` with the domain `Reals`. A
failure records them too once the argument names are validated, the variable with its own domain and assumptions,
which a real variable without assumptions shares with a success; a failure before that records neither.

## Supersede a v0.7 operation

A family may take over a v0.7 operation, of the 60 v0.7 operations or of the older legacy catalog, when the
operation's v0.7 evidence cannot be checked fast enough. It names the operation in `SUPERSEDES`, a `frozenset`,
and defines a v6 spec of the same name with the v0.7 argument names, order and defaults and the v0.7 public name.
New results of the name are then v6 records, with a `v6.<operation>` step and a request of `version=6`, while the
older records of the name keep their format: they still decode, encode to the same bytes and verify with their
own verifiers, because `dispatch_v5` and the legacy catalog keep the old name.

A superseding family need not take every input. Its `produce` raises `operations_v6.Declined`, which is no
`MathForgeError`, for an input that it leaves to the v0.7 implementation, and `operations_v6.run_dual` then
answers with the v0.7 record. A family declines every input that it does not handle, also every input that v0.7
rejects, so that v0.7's errors stay exactly as they were; `run_dual` declines for it the inputs that fail the
argument checks before `produce`. Decline before any expensive work: a budget that the caller shares also pays
for the attempt.

```
from mathforge.operations_v6 import Declined

SUPERSEDES = frozenset({"rowspace"})

def produce(operation, args):
    if type(args["A"]).__name__ != "Matrix" or not args["A"].rows:
        raise Declined("the v0.7 implementation answers this input")
    ...
```

The generated structured wrapper of a superseded v0.7 operation calls `run_dual`, and its value function keeps
its name and values. A superseded legacy operation gets no generated wrapper, and no code routes it to `run_dual`
yet: its public functions in `mathforge.operations` and `mathforge` stay, and a family that supersedes one must
change them by hand to call `run_dual` with `older`, a function that returns the v0.7 Result. Its area module
re-exports the `mathforge` value function. `mathforge.capabilities()` and the catalog list the name once, and the
MCP tools run it through its public structured function.

## Keep records plain

A record is a frozen dataclass whose `__post_init__` checks its canonical form. It holds data and has no
computing methods: a checker necessarily reads the records it verifies, and a method would be producer code
running inside the checker. No two record classes share a name, and when a record can be the `expression` or
`A` argument of an operation, its serialized tag (`"v6_"` plus its tag) joins `expression_tags` in
`tools/generate_schema.py`.

## Write the checker

The checker lives in `src/mathforge/certify/<area>/<family>.py` and defines
`check_operation(operation, args, value, evidence)`. It verifies the claim with a certificate or a different
algorithm, never by running the producer again, and returns `True` for a verified value and `False` for a wrong
one. A checker that raises gives an `UNKNOWN` check, not a verified one.

A checker imports only `mathforge.certify`, `mathforge.trusted`, `mathforge.errors`, `mathforge.limits` and
`mathforge.model`, and never the producer, not even to name a record class: it compares the module and name of
the value's class instead. `python tools/check_layering.py` follows every chain of imports from
`mathforge.certify` but cannot follow an import made at run time, so tests reject the ways around it: a checker
never names `importlib`, `__import__` or `sys.modules`, and never imports `sys`, `builtins` or `importlib` in any
form (`import sys as s`, `from sys import *`, `import importlib.util`).

## Never change a released argument set

Once a release ships a v6 operation, its argument names stay as they are: every stored request names exactly the
arguments of its operation, and the decoder, the JSON schema and the MCP tools check them. Extend an operation by
adding a new operation name, never by changing the arguments of a released one.

## Names the registry rejects

The registry raises `RuntimeError` when it builds its tables, at the first lookup, for:

- a family that lacks `SPECS`, `TYPES`, `CHECKER` or `produce`, or whose `CHECKER` is not an import path below
  `mathforge.certify` or names the family itself;
- an operation or public name equal to an argument kind or `envelope`, which name definitions of the MCP area-tool
  schemas, or already taken, by another operation or by an older one of the legacy catalog, unless the family
  supersedes that older operation;
- a `SUPERSEDES` that is not a `frozenset` of names, that names an operation which is not a v0.7 one or which the
  family's `SPECS` do not define, and a superseding spec whose argument names, order or defaults, or public name,
  differ from v0.7's;
- an argument or public name that the generated functions use: `limits`, `dict`, `run`, `run_v6`, `structured`,
  `_LEAN`, `FALLBACK`, `_value` and `run_dual` (`registry.WRAPPER_NAMES`);
- a name that is not a Python identifier, and a default whose text is not a Python literal;
- two record classes with the same name, and a class in `TYPES` that the family module does not define itself,
  even when another record module defines it: every `TYPES` class must have the family's own module name as its
  `__module__`.

## Regenerate and check

```sh
python tools/generate_api.py
python tools/generate_schema.py
python tools/check_layering.py
```

The first command writes the generated functions and the area modules, the second the JSON schemas. Then add a
seeded case to the [development differential suite](performance.md) and a typing assertion to
`tests/typing/public_api.py`, alongside the family's own tests. When the family gives an area tool its first
operation, that tool joins the MCP tool list, and the list that the [MCP guide](mcp.md) states and checks
changes too.

`tools/generate_api.py` imports `mathforge` to read the registry, so a broken or conflicted generated file, as
two families merged in parallel can leave behind, stops it; it then prints how to recover. Restore the
generated files from `HEAD`, which also works for a path that the merge left unmerged, and regenerate:

```sh
git checkout HEAD -- src/mathforge/_generated src/mathforge/<area>.py
python tools/generate_api.py
```
