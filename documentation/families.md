# Adding an operation family

Operations added since v0.8 come in families: a producer module that computes the values and an independent
checker that verifies them. Everything else follows from the family through the registry: the generated
functions of `mathforge.operations` and of the area module, the JSON v6 records, the MCP area tools and the
catalog. The pilot family, `mathforge.areas.ntheory.continued_fraction`, with its checker
`mathforge.certify.ntheory.continued_fraction`, is the template; its docstring spells out the same contract.

```python
import mathforge as mf
from mathforge import registry
from mathforge.areas.ntheory import continued_fraction as family

assert family.__name__ in registry.V6_FAMILIES
assert all(hasattr(family, name) for name in registry.V6_CONTRACT)
assert family.CHECKER.startswith(registry.CHECKER_PACKAGE + ".")
assert [(spec.name, spec.version) for spec in family.SPECS] == [("continued_fraction", 6)]

value, evidence = family.produce("continued_fraction", {"value": mf.Rational(415, 93)})
assert value == family.ContinuedFraction((4, 2, 6, 7)) and evidence == ()
checker = registry.checker("continued_fraction")
assert checker.check_operation("continued_fraction", {"value": mf.Rational(415, 93)}, value, evidence) is True
assert checker.check_operation("continued_fraction", {"value": mf.Rational(415, 92)}, value, evidence) is False
```

## Write the family module

A family lives in `src/mathforge/areas/<area>/<family>.py`, and its name joins `V6_FAMILIES` in
`src/mathforge/registry.py`. The module defines:

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

`produce` validates its inputs first and raises `InvalidInput` for those it rejects. A Python `float` or `bool`
is never an exact number, so test `type(value) is int`, never `isinstance`. Once the inputs are validated, a
family builds records only from validated data: an `InvalidInput` raised later still reads as rejected input.
Any other exception, apart from `UnsupportedOperation` and `ResourceLimitError`, is reported as the producer's
`internal_error`. `run` copies an `expression` (or `A`) argument into the result, where a raw `int` serializes
but does not load again: the first family that takes such an argument coerces an `int` to a `Rational` before it
reaches the `Result`, in `operations_v6.source_context`, and round-trips its results in a test.

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
the value's class instead. It never uses `importlib`, `__import__` or `sys.modules`.
`python tools/check_layering.py` follows every chain of imports from `mathforge.certify`, and a test rejects the
dynamic forms.

## Never change a released argument set

Once a release ships a v6 operation, its argument names stay as they are: every stored request names exactly the
arguments of its operation, and the decoder, the JSON schema and the MCP tools check them. Extend an operation by
adding a new operation name, never by changing the arguments of a released one.

## Names the registry rejects

The registry raises `RuntimeError` when it builds its tables, at the first lookup, for:

- a family that lacks `SPECS`, `TYPES`, `CHECKER` or `produce`, or whose `CHECKER` is not an import path below
  `mathforge.certify` or names the family itself;
- an operation or public name equal to an argument kind or `envelope`, which name definitions of the MCP area-tool
  schemas, or already taken, by another operation or by an older one of the legacy catalog;
- an argument or public name that the generated functions use: `limits`, `dict`, `run`, `run_v6` and
  `structured` (`registry.WRAPPER_NAMES`);
- a name that is not a Python identifier, and a default whose text is not a Python literal;
- two record classes with the same name, and a record class defined outside its family module.

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
generated files and regenerate:

```sh
git checkout -- src/mathforge/_generated src/mathforge/<area>.py
python tools/generate_api.py
```
