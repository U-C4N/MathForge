"""Declarative operation registry: the single source of truth for operations.

Each operation is described once by an :class:`OperationSpec`: its area, its
arguments (source-text annotation and default, MCP argument kind), its return
type and its catalogue texts. Wrappers, capabilities and MCP schemas derive
from these specs.

Importing this module stays light, because the package imports it during its
own initialization: the v0.7 table and the producer ("family") modules load
on the first lookup. The tables are then built once, under a lock, and
memoized. Family modules must not call into the registry at import time.

The v6 families are the modules that the area packages below mathforge.areas
list in FAMILIES (v6_families()). A family module stays light: it declares its
specs and records, and imports its kernels inside produce(). Its checker loads
on the first checker() call. A family may supersede v0.7 operations
(SUPERSEDES): its v6 spec then takes the operation's name for new results,
while the v0.7 view that dispatch_v5 serves keeps the v0.7 family, so that
older records of the name still decode and verify.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator, Mapping
import contextlib
from dataclasses import dataclass
import functools
import importlib
import keyword
import sys
import threading
from types import MappingProxyType, ModuleType

from .errors import UnsupportedOperation


@dataclass(frozen=True)
class Argument:
    name: str
    annotation: str = ""        # Source text; "" = unannotated, as in v0.7
    default: str | None = None  # Python literal source text; None = required
    kind: str = "object"        # MCP argument kind, from KIND_NAMES


@dataclass(frozen=True)
class OperationSpec:
    name: str
    area: str
    arguments: tuple[Argument, ...]
    returns: str                # Source text of the return type
    summary: str                # The one sentence shown in the catalogue
    scope: str                  # capabilities scope text, without the argument list
    sdk: str = ""               # Public function name; "" = name
    version: int = 5            # JSON format version of the results: 5 or 6
    mcp: bool = True

    @property
    def public_name(self) -> str:
        return self.sdk or self.name


AREAS = ("expr", "algebra", "sets", "logic", "equations", "analysis", "series", "integrals", "ode",
         "special", "linalg", "stats", "ntheory", "discrete", "geometry", "vector", "numeric", "crypto")

# MCP area tools and the areas each one serves.
MCP_TOOLS: dict[str, tuple[str, ...]] = {
    "mathforge.algebra": ("expr", "algebra", "sets"),
    "mathforge.logic": ("logic",),
    "mathforge.equations": ("equations",),
    "mathforge.analysis": ("analysis", "series", "integrals"),
    "mathforge.ode": ("ode",),
    "mathforge.special": ("special",),
    "mathforge.linalg": ("linalg",),
    "mathforge.stats": ("stats",),
    "mathforge.ntheory": ("ntheory", "crypto"),
    "mathforge.discrete": ("discrete",),
    "mathforge.geometry": ("geometry", "vector"),
    "mathforge.numeric": ("numeric",),
}

KIND_NAMES = ("int", "int_list", "int_pairs", "rational", "rational_list", "scalar", "complex", "matrix",
              "distribution", "function", "function_list", "rational_function", "symbol", "symbols",
              "bindings", "point", "side", "object")

# Names that the build keeps operations and arguments from taking. The $defs of an MCP area-tool schema hold
# each operation's arguments under the operation's name, beside the argument kinds and the canonical envelope.
DEFINITION_NAMES = (*KIND_NAMES, "envelope")
# The names that a wrapper written by tools/generate_api.py uses besides its own arguments: its keyword-only
# limits, the dict builtin, the run functions and the structured module. An argument or a public name equal
# to one of them would shadow it; the generator checks that its wrappers use no other name.
WRAPPER_NAMES = ("limits", "dict", "run", "run_v6", "structured")

# A value wrapper also uses the lean functions, their FALLBACK answer and its local _value (mathforge.lean).
WRAPPER_NAMES += ("_LEAN", "FALLBACK", "_value")  # type: ignore[assignment]  # The tuple grows by three names.

# The structured wrapper of a v0.7 operation that a v6 family supersedes calls operations_v6.run_dual.
WRAPPER_NAMES += ("run_dual",)  # type: ignore[assignment]  # The tuple grows by one name.

# v0.7 producer modules. Their ARGUMENTS must match the v0.7 specs; their TYPES are the v5 records.
V5_FAMILIES = tuple("mathforge." + name for name in (
    "number_theory", "combinatorics", "complex_numbers", "probability", "statistics",
    "linear_extensions", "functions", "function_calculus", "rational_integration", "function_integration"))

# Modules defining the v5 record classes: the families, plus function_limits, whose records
# function_calculus.TYPES re-exports.
V5_RECORD_MODULES = V5_FAMILIES + ("mathforge.function_limits",)

# The v6 family modules are listed by the area packages below AREAS_PACKAGE, each in its FAMILIES (v6_families()).
# A family defines SPECS, TYPES (record classes that the module itself defines), CHECKER (the import path of its
# checker below CHECKER_PACKAGE) and produce(), and optionally RENDER: {record class of its TYPES: render(value,
# renderer, child) -> str} and SUPERSEDES: the v0.7 operations whose names its SPECS take over. The build checks
# all of them.
AREAS_PACKAGE = "mathforge.areas"
V6_CONTRACT = ("SPECS", "TYPES", "CHECKER", "produce")
# tools/check_layering.py keeps every module of this package away from producer code.
CHECKER_PACKAGE = "mathforge.certify"
# The families that _testing_families added, after the declared ones: only they may give CHECKER as a module or
# another path.
_TESTING_FAMILIES: tuple[str, ...] = ()

# The generic deferral records (Reason, Unevaluated): their TYPES lead record_types(6).
DEFERRED_MODULE = "mathforge.deferred"


def record_module(module: str) -> bool:
    """Whether a module may define registry records; an answer that needs no build.

    It reads the FAMILIES of the area packages (v6_families()), never a family module. contracts relies
    on it so that v1-v4 values never build the registry; the build checks that every record class lives
    in such a module, and that a v6 family defines its own.
    """
    return module in V5_RECORD_MODULES or module == DEFERRED_MODULE or module in v6_families()


@functools.cache
def v6_families() -> tuple[str, ...]:
    """The v6 family modules: the FAMILIES of every area package below mathforge.areas, in package order, then
    the families that _testing_families registers. Cached; it imports the small area packages, never a family.

    An area package declares FAMILIES, a tuple of the full module names of its own families
    (areas/<area>/__init__.py); a package without one has none yet.
    """
    import pkgutil
    package = importlib.import_module(AREAS_PACKAGE)
    names: list[str] = []
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if not info.ispkg:
            continue
        declared: object = getattr(importlib.import_module(info.name), "FAMILIES", ())
        if type(declared) is not tuple or any(type(name) is not str or not name.startswith(info.name + ".")
                                              for name in declared):
            raise RuntimeError(f"FAMILIES of {info.name} must be a tuple of the full module names of its families: "
                               f"{declared!r}")
        names.extend(declared)
    return (*names, *_TESTING_FAMILIES)


def _supersedes(family: ModuleType) -> frozenset[str]:
    """A family's SUPERSEDES declaration, frozenset() when it has none."""
    declared: object = getattr(family, "SUPERSEDES", frozenset())
    if type(declared) is not frozenset or any(type(name) is not str for name in declared):
        raise RuntimeError(f"SUPERSEDES of v6 family {family.__name__} must be a frozenset of operation names")
    return declared


@functools.cache
def _superseded() -> frozenset[str]:
    """The older operation names that the v6 families supersede, read from the family modules' SUPERSEDES without
    building the tables; the build checks each declaration."""
    return frozenset(name for family in v6_families() for name in _supersedes(importlib.import_module(family)))


@dataclass(frozen=True)
class _Tables:
    specs: Mapping[str, OperationSpec]
    families: Mapping[int, Mapping[str, ModuleType]]  # By result format version: operation name -> family.
    checkers: Mapping[int, Mapping[str, object]]  # By version: a checker module, or the import path of a v6 one.
    loaded: dict[tuple[int, str], ModuleType]  # The checkers imported so far, by (version, operation name).
    modules: Mapping[int, tuple[ModuleType, ...]]
    arguments: Mapping[int | None, Mapping[str, frozenset[str]]]
    types: Mapping[int, Mapping[str, type]]
    all_types: frozenset[type]
    renderers: Mapping[type, Callable[..., str] | None]
    areas: Mapping[str, tuple[OperationSpec, ...]]
    tools: Mapping[str, tuple[OperationSpec, ...]]


_LOCK = threading.Lock()
_BUILT: list[_Tables] = []  # The one assembled table set, appended under _LOCK.


@functools.cache
def _tables() -> _Tables:
    # Imports run outside the lock: the import system locks each module itself,
    # and holding _LOCK across imports could deadlock against it.
    from .registry_v07 import LEGACY_CATALOG, V07_SPECS
    v5 = tuple(importlib.import_module(name) for name in V5_FAMILIES)
    v6 = tuple(importlib.import_module(name) for name in v6_families())
    checkers = tuple(_checker_source(module) for module in v6)  # Checked here, imported on first use.
    deferred = importlib.import_module(DEFERRED_MODULE)
    legacy = frozenset(entry.name for entry in LEGACY_CATALOG)
    signatures = _legacy_signatures(v6, legacy)
    with _LOCK:
        # functools.cache may run this body twice when threads race on the first
        # call; every caller still receives the tables assembled first.
        if not _BUILT:
            _BUILT.append(_assemble(V07_SPECS, legacy, v5, v6, checkers, deferred, signatures))
        return _BUILT[0]


def _clear_caches() -> None:
    """Forget the built tables and the family lists, so that the next lookup rebuilds them (for tests)."""
    with _LOCK:
        _BUILT.clear()
        _tables.cache_clear()
        for cached in _FAMILY_CACHES:  # The cached functions themselves, also while a test patches their names.
            cached.cache_clear()


_FAMILY_CACHES = (v6_families, _superseded)


@contextlib.contextmanager
def _testing_families(*modules: ModuleType) -> Iterator[None]:
    """Register v6 family modules for the duration of a with block (tests only).

    The modules join v6_families(), after the declared families: the one list that record_module(),
    is_v6_operation() and the table build read, so they always agree. The build imports families by name,
    so each module must be its own sys.modules entry. Unlike a production family, a testing family may give
    CHECKER as a module object or as any import path. The caches are cleared on entry and on exit; the
    family list is restored even when the block raises.
    """
    global _TESTING_FAMILIES
    stray = [module.__name__ for module in modules if sys.modules.get(module.__name__) is not module]
    if stray:
        raise ValueError("A testing family must be an imported module: " + ", ".join(stray))
    previous = _TESTING_FAMILIES
    _TESTING_FAMILIES = previous + tuple(module.__name__ for module in modules)
    try:
        _clear_caches()
        yield
    finally:
        _TESTING_FAMILIES = previous
        _clear_caches()


def _checker_source(family: ModuleType) -> object:
    """The CHECKER of a v6 family, once the family is known to define the whole contract; checker() imports it.

    A production family names its checker by an import path below CHECKER_PACKAGE, never its own module, so
    that tools/check_layering.py covers the checker and the checker is not the code it verifies.
    """
    missing = [name for name in V6_CONTRACT if not hasattr(family, name)]
    if missing:
        raise RuntimeError(f"v6 family {family.__name__} does not define {', '.join(missing)}")
    checker = family.CHECKER
    if family.__name__ not in _TESTING_FAMILIES and (
            type(checker) is not str or not checker.startswith(CHECKER_PACKAGE + ".") or checker == family.__name__):
        raise RuntimeError(f"v6 family {family.__name__} must name its CHECKER by an import path below "
                           f"{CHECKER_PACKAGE}, other than its own module: {checker!r}")
    return checker


_REQUIRED = object()  # The default of a required argument, in the argument lists that supersession compares.


def _legacy_signatures(v6: tuple[ModuleType, ...],
                       legacy: frozenset[str]) -> dict[str, tuple[tuple[str, object], ...] | None]:
    """For each legacy operation that a family supersedes: the (name, default) pairs of its v0.7 public function's
    parameters, limits excepted, from mathforge.operations; None when that module has no such function.

    Read before the build takes its lock, because it may import mathforge.operations.
    """
    wanted = sorted({name for family in v6 for name in _supersedes(family) if name in legacy})
    if not wanted:
        return {}
    import inspect
    from . import operations
    signatures: dict[str, tuple[tuple[str, object], ...] | None] = {}
    for name in wanted:
        function = getattr(operations, name, None)
        signatures[name] = None if not callable(function) else tuple(
            (parameter.name, _REQUIRED if parameter.default is parameter.empty else parameter.default)
            for parameter in inspect.signature(function).parameters.values() if parameter.name != "limits")
    return signatures


def _record_types(modules: tuple[ModuleType, ...], version: int) -> dict[str, type]:
    merged: dict[str, type] = {}
    for module in modules:
        for tag, cls in getattr(module, "TYPES", {}).items():
            if tag in merged:
                raise RuntimeError(f"Duplicate v{version} type: {tag}")
            if not record_module(cls.__module__):
                raise RuntimeError(f"Record type {tag} is defined outside the record modules: {cls.__module__}")
            # A v6 module registers only the classes it defines, so a checker can name a record class by its module
            # and name. The frozen v0.7 families differ: function_calculus.TYPES re-exports function_limits records.
            if version == 6 and cls.__module__ != module.__name__:
                raise RuntimeError(f"v6 record class {cls.__name__} of type {tag} must be defined in its family "
                                   f"module {module.__name__}, not in {cls.__module__}")
            merged[tag] = cls
    return merged


def _renderers(families: tuple[ModuleType, ...],
               v6_types: Mapping[str, type]) -> dict[type, Callable[..., str] | None]:
    """Every v6 record class, mapped to the render function that its family declares in RENDER, else None.

    RENDER is optional. A family that defines it maps classes of its own TYPES to callables.
    """
    renderers: dict[type, Callable[..., str] | None] = dict.fromkeys(v6_types.values())
    for module in families:
        if not hasattr(module, "RENDER"):
            continue
        if not isinstance(module.RENDER, Mapping):
            raise RuntimeError("RENDER must be a mapping: " + module.__name__)
        own = frozenset(getattr(module, "TYPES", {}).values())
        for cls, render in module.RENDER.items():
            if cls not in own:
                raise RuntimeError(
                    "RENDER keys must be record classes of the family's own TYPES: " + module.__name__)
            if not callable(render):
                raise RuntimeError("RENDER values must be callable: " + module.__name__)
            renderers[cls] = render
    return renderers


def _identifier(name: object) -> bool:
    return type(name) is str and name.isidentifier() and not keyword.iskeyword(name)


def _literal(text: object) -> bool:
    """Whether a default's source text is a Python literal whose value repr() writes back as an equal literal.

    The generated wrappers copy the text into their signatures and the MCP tools evaluate it with
    ast.literal_eval, so anything else would break ``import mathforge`` or the server's start.
    """
    if type(text) is not str:
        return False
    try:
        value = ast.literal_eval(text)
        return bool(ast.literal_eval(repr(value)) == value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def _check_names(spec: OperationSpec) -> None:
    """Reject the names and defaults that the generated wrappers or the MCP area-tool schemas cannot carry."""
    for label, name in (("Operation name", spec.name), ("Public name", spec.public_name)):
        if not _identifier(name):
            raise RuntimeError(f"{label} is not a Python identifier: {name!r}")
        if name in DEFINITION_NAMES:
            raise RuntimeError(f"{label} collides with an MCP schema definition: {name}")
    if spec.public_name in WRAPPER_NAMES:  # A wrapper named dict, say, would shadow the builtin in every wrapper.
        raise RuntimeError("Public name collides with a name the generated wrappers use: " + spec.public_name)
    for argument in spec.arguments:
        if not _identifier(argument.name):
            raise RuntimeError(f"Argument name of {spec.name} is not a Python identifier: {argument.name!r}")
        if argument.name in WRAPPER_NAMES:
            raise RuntimeError(f"Argument name of {spec.name} collides with a name the generated wrappers use: "
                               f"{argument.name}")
        if argument.default is not None and not _literal(argument.default):
            raise RuntimeError(f"Default of argument {argument.name} of {spec.name} is not a Python literal: "
                               f"{argument.default!r}")


def _literal_default(argument: Argument) -> object:
    return _REQUIRED if argument.default is None else ast.literal_eval(argument.default)


def _same_values(left: tuple[tuple[str, object], ...], right: tuple[tuple[str, object], ...]) -> bool:
    """Equal (name, default) lists; a default equals another only of the same type (True is not 1)."""
    return len(left) == len(right) and all(
        name == other and type(value) is type(default) and value == default
        for (name, value), (other, default) in zip(left, right))


def _check_supersession(family: ModuleType, spec: OperationSpec, older: OperationSpec | None,
                        signature: tuple[tuple[str, object], ...] | None) -> None:
    """A superseding spec keeps the v0.7 operation's argument names, order and defaults, and its public name.

    older is the v0.7 spec of a v0.7 v5 operation, None for a legacy one, whose arguments are the parameters
    (signature) of its public function in mathforge.operations.
    """
    owner = family.__name__
    if older is not None:
        expected: tuple[tuple[str, object], ...] = tuple((a.name, a.default) for a in older.arguments)
        public = older.public_name
        same = tuple((a.name, a.default) for a in spec.arguments) == expected
    else:
        if signature is None:
            raise RuntimeError(f"v6 family {owner} supersedes {spec.name}, which has no structured function in "
                               "mathforge.operations")
        expected, public = signature, spec.name
        same = _same_values(tuple((a.name, _literal_default(a)) for a in spec.arguments), signature)
    if not same:
        found = tuple((a.name, a.default) for a in spec.arguments)
        raise RuntimeError(f"v6 family {owner} supersedes {spec.name} with other arguments than v0.7's: "
                           f"{found!r} instead of {expected!r}")
    if spec.public_name != public:
        raise RuntimeError(f"v6 family {owner} supersedes {spec.name} under another public name than v0.7's: "
                           f"{spec.public_name} instead of {public}")


def _assemble(v07_specs: tuple[OperationSpec, ...], legacy: frozenset[str], v5: tuple[ModuleType, ...],
              v6: tuple[ModuleType, ...], v6_checkers: tuple[object, ...], deferred: ModuleType,
              signatures: Mapping[str, tuple[tuple[str, object], ...] | None]) -> _Tables:
    specs: dict[str, OperationSpec] = {}
    public: set[str] = set()
    families: dict[int, dict[str, ModuleType]] = {5: {}, 6: {}}
    checkers: dict[int, dict[str, object]] = {5: {}, 6: {}}
    arguments: dict[int | None, dict[str, frozenset[str]]] = {None: {}, 5: {}, 6: {}}

    def add(spec: OperationSpec, version: int, family: ModuleType, checker: object, superseding: bool = False) -> None:
        """Register spec; a superseding v6 spec takes the place of the v0.7 operation of its name in specs."""
        if spec.name in families[version] or (spec.name in specs and not superseding):
            raise RuntimeError("Duplicate operation: " + spec.name)
        if not superseding:
            if spec.public_name in public:
                raise RuntimeError("Duplicate public name: " + spec.public_name)
            if spec.name in legacy:
                raise RuntimeError("Operation name collides with a legacy catalogue entry: " + spec.name)
            if spec.public_name in legacy:  # Its generated wrapper would shadow mathforge.operations.<name>.
                raise RuntimeError("Public name collides with a legacy catalogue entry: " + spec.public_name)
        if (spec.version != version or spec.area not in AREAS
                or any(argument.kind not in KIND_NAMES for argument in spec.arguments)):
            raise RuntimeError("Malformed operation spec: " + spec.name)
        _check_names(spec)
        specs[spec.name] = spec  # A superseded v0.7 operation keeps its place in the published order.
        families[version][spec.name], checkers[version][spec.name] = family, checker
        arguments[None][spec.name] = arguments[version][spec.name] = frozenset(
            argument.name for argument in spec.arguments)
        public.add(spec.public_name)

    producers: dict[str, tuple[ModuleType, frozenset[str]]] = {}
    for module in v5:
        for name, keys in getattr(module, "ARGUMENTS", {}).items():
            if name in producers:
                raise RuntimeError("Duplicate v5 operation: " + name)
            producers[name] = module, frozenset(keys)
    for spec in v07_specs:
        producer = producers.pop(spec.name, None)
        if producer is None or producer[1] != frozenset(argument.name for argument in spec.arguments):
            raise RuntimeError("The v0.7 spec does not match its family's ARGUMENTS: " + spec.name)
        add(spec, 5, producer[0], producer[0])
    if producers:
        raise RuntimeError("v0.7 operations without a spec: " + ", ".join(producers))
    v07 = {spec.name: spec for spec in v07_specs}
    for module, checker in zip(v6, v6_checkers):
        supersedes = _supersedes(module)
        defined = {spec.name for spec in module.SPECS}
        for name in sorted(supersedes):
            if name not in v07 and name not in legacy:
                raise RuntimeError(f"v6 family {module.__name__} supersedes {name}, which is not a v0.7 operation")
            if name not in defined:
                raise RuntimeError(f"v6 family {module.__name__} supersedes {name}, which none of its SPECS defines")
        for spec in module.SPECS:
            superseding = spec.name in supersedes
            if superseding:
                _check_supersession(module, spec, v07.get(spec.name), signatures.get(spec.name))
            add(spec, 6, module, checker, superseding)

    # A family tag equal to a deferral tag ("reason", "unevaluated") is a duplicate v6 type.
    types = {5: _record_types(v5, 5), 6: _record_types((deferred, *v6), 6)}
    named: dict[str, type] = {}  # The generator and the area modules find record classes by their names.
    for cls in (cls for table in types.values() for cls in table.values()):
        if named.setdefault(cls.__name__, cls) is not cls:
            raise RuntimeError(f"Two record classes are named {cls.__name__}: in {named[cls.__name__].__module__} "
                               f"and {cls.__module__}")
    renderers = _renderers(v6, types[6])
    areas = {area: tuple(sorted((spec for spec in specs.values() if spec.area == area),
                                key=lambda spec: spec.public_name)) for area in AREAS}
    tools = {tool: tuple(sorted((spec for spec in specs.values() if spec.mcp and spec.area in served),
                                key=lambda spec: spec.name)) for tool, served in MCP_TOOLS.items()}
    return _Tables(
        specs=MappingProxyType(specs),
        families=MappingProxyType({key: MappingProxyType(value) for key, value in families.items()}),
        checkers=MappingProxyType({key: MappingProxyType(value) for key, value in checkers.items()}), loaded={},
        modules=MappingProxyType({5: v5, 6: v6}),
        arguments=MappingProxyType({key: MappingProxyType(value) for key, value in arguments.items()}),
        types=MappingProxyType({key: MappingProxyType(value) for key, value in types.items()}),
        all_types=frozenset(cls for table in types.values() for cls in table.values()),
        renderers=MappingProxyType(renderers), areas=MappingProxyType(areas), tools=MappingProxyType(tools))


def _lookup(table, key, kind):
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"Unknown registry {kind}: {key!r}") from None


def specs() -> Mapping[str, OperationSpec]:
    """Every registered operation by name, v0.7 operations first in their published order."""
    return _tables().specs


def spec(name: str) -> OperationSpec:
    try:
        return _tables().specs[name]
    except KeyError:
        raise UnsupportedOperation("Unknown operation") from None


def family(name: str, version: int | None = None) -> ModuleType:
    """The producer module of an operation, of a result format version (5 or 6), or by default of its v0.7 view.

    By default a v0.7 operation gives its v0.7 family, also once a v6 family supersedes it: dispatch_v5 runs,
    decodes and verifies the operation's v5 records with that module. version=6 gives the v6 family of a v6
    operation, superseding or not.
    """
    tables = _tables()
    if version is None:
        version = 5 if name in tables.families[5] else 6
    try:
        return _lookup(tables.families, version, "version")[name]
    except KeyError:
        raise UnsupportedOperation("Unknown operation") from None


def checker(name: str, version: int | None = None) -> ModuleType:
    """The checker module of an operation: the family itself for v5, the family's CHECKER for v6.

    By default the checker of the operation's spec in specs(), a superseding family's for a superseded v0.7
    operation; version=5 or 6 picks one. A v6 checker is imported on the first call, then cached.
    """
    tables = _tables()
    if version is None:
        spec = tables.specs.get(name)
        version = 5 if spec is None else spec.version
    key = (version, name)
    module = tables.loaded.get(key)
    if module is None:
        try:
            source = _lookup(tables.checkers, version, "version")[name]
        except KeyError:
            raise UnsupportedOperation("Unknown operation") from None
        module = tables.loaded.setdefault(key, importlib.import_module(source) if isinstance(source, str) else source)
    return module


def operation_arguments(version: int | None = None) -> Mapping[str, frozenset[str]]:
    """Argument names by operation, for one result format version or (None) for all."""
    return _lookup(_tables().arguments, version, "version")


@functools.cache
def _older_names() -> frozenset[str]:
    """The operation names of the legacy catalogue and the v0.7 table: static data, no family module loads."""
    from .registry_v07 import LEGACY_CATALOG, V07_SPECS
    return frozenset(entry.name for entry in LEGACY_CATALOG) | frozenset(spec.name for spec in V07_SPECS)


def is_older_operation(name: str) -> bool:
    """Whether a name belongs to an operation of format v1-v5: a legacy catalogue entry or a v0.7 operation.

    Static data: the answer builds no tables, and the build rejects v6 specs with such names.
    """
    return name in _older_names()


def is_v6_operation(name: str) -> bool:
    """Whether new results of an operation have format version 6: a v6 family defines or supersedes it.

    An older name, of the legacy catalogue or of a v0.7 operation, is v6 only when a family's SUPERSEDES names
    it. That is read from the family modules without building the tables, so that checking a v1-v5 request or
    running a legacy operation builds none.
    """
    if is_older_operation(name):
        return name in _superseded()
    return name in operation_arguments(6)


def record_types(version: int) -> Mapping[str, type]:
    """Record classes by tag; the serialized prefix ("v5_", "v6_") is not part of the tag.

    Version 6 holds the generic deferral records of mathforge.deferred, then the families' records.
    """
    return _lookup(_tables().types, version, "version")


def all_record_types() -> frozenset[type]:
    return _tables().all_types


def record_renderers() -> Mapping[type, Callable[..., str] | None]:
    """Every v6 record class, mapped to the render(value, renderer, child) function of its family's RENDER.

    The deferral records, which rendering_v6 presents itself, and records without a RENDER entry map to None.
    """
    return _tables().renderers


def area_operations(area: str) -> tuple[OperationSpec, ...]:
    """The operations of one area, sorted by public name."""
    return _lookup(_tables().areas, area, "area")


def tool_operations(tool: str) -> tuple[OperationSpec, ...]:
    """The MCP-exposed operations of an area tool's areas, sorted by name."""
    return _lookup(_tables().tools, tool, "tool")


def _family_modules(version: int) -> tuple[ModuleType, ...]:
    """The family modules of one version, in declaration order (dispatch_v5.modules)."""
    return _lookup(_tables().modules, version, "version")
