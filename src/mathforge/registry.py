"""Declarative operation registry: the single source of truth for operations.

Each operation is described once by an :class:`OperationSpec`: its area, its
arguments (source-text annotation and default, MCP argument kind), its return
type and its catalogue texts. Wrappers, capabilities and MCP schemas derive
from these specs.

Importing this module stays light, because the package imports it during its
own initialization: the v0.7 table and the producer ("family") modules load
on the first lookup. The tables are then built once, under a lock, and
memoized. Family modules must not call into the registry at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import functools
import importlib
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

# v0.7 producer modules. Their ARGUMENTS must match the v0.7 specs; their TYPES are the v5 records.
V5_FAMILIES = tuple("mathforge." + name for name in (
    "number_theory", "combinatorics", "complex_numbers", "probability", "statistics",
    "linear_extensions", "functions", "function_calculus", "rational_integration", "function_integration"))

# Modules defining the v5 record classes: the families, plus function_limits, whose records
# function_calculus.TYPES re-exports.
V5_RECORD_MODULES = V5_FAMILIES + ("mathforge.function_limits",)

# v6 family modules; each defines SPECS, TYPES, CHECKER (a module or its import path) and produce().
V6_FAMILIES: tuple[str, ...] = ()


def _record_module(module: str) -> bool:
    """Whether a module may define registry records; a static answer that needs no build.

    contracts relies on it so that v1-v4 values never build the registry; the build checks
    that every record class lives in such a module.
    """
    return module in V5_RECORD_MODULES or module in V6_FAMILIES


@dataclass(frozen=True)
class _Tables:
    specs: Mapping[str, OperationSpec]
    families: Mapping[str, ModuleType]
    checkers: Mapping[str, ModuleType]
    modules: Mapping[int, tuple[ModuleType, ...]]
    arguments: Mapping[int | None, Mapping[str, frozenset[str]]]
    types: Mapping[int, Mapping[str, type]]
    all_types: frozenset[type]
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
    v6 = tuple(importlib.import_module(name) for name in V6_FAMILIES)
    checkers = tuple(importlib.import_module(module.CHECKER) if isinstance(module.CHECKER, str)
                     else module.CHECKER for module in v6)
    legacy = frozenset(entry.name for entry in LEGACY_CATALOG)
    with _LOCK:
        # functools.cache may run this body twice when threads race on the first
        # call; every caller still receives the tables assembled first.
        if not _BUILT:
            _BUILT.append(_assemble(V07_SPECS, legacy, v5, v6, checkers))
        return _BUILT[0]


def _clear_caches() -> None:
    """Forget the built tables, so that the next lookup rebuilds them (for tests)."""
    with _LOCK:
        _BUILT.clear()
        _tables.cache_clear()


def _record_types(modules: tuple[ModuleType, ...], version: int) -> dict[str, type]:
    merged: dict[str, type] = {}
    for module in modules:
        for tag, cls in getattr(module, "TYPES", {}).items():
            if tag in merged:
                raise RuntimeError(f"Duplicate v{version} type: {tag}")
            if not _record_module(cls.__module__):
                raise RuntimeError(f"Record type {tag} is defined outside the record modules: {cls.__module__}")
            merged[tag] = cls
    return merged


def _assemble(v07_specs: tuple[OperationSpec, ...], legacy: frozenset[str], v5: tuple[ModuleType, ...],
              v6: tuple[ModuleType, ...], v6_checkers: tuple[ModuleType, ...]) -> _Tables:
    specs: dict[str, OperationSpec] = {}
    public: set[str] = set()
    families: dict[str, ModuleType] = {}
    checkers: dict[str, ModuleType] = {}

    def add(spec: OperationSpec, version: int, family: ModuleType, checker: ModuleType) -> None:
        if spec.name in specs:
            raise RuntimeError("Duplicate operation: " + spec.name)
        if spec.public_name in public:
            raise RuntimeError("Duplicate public name: " + spec.public_name)
        if spec.name in legacy:
            raise RuntimeError("Operation name collides with a legacy catalogue entry: " + spec.name)
        if (spec.version != version or spec.area not in AREAS
                or any(argument.kind not in KIND_NAMES for argument in spec.arguments)):
            raise RuntimeError("Malformed operation spec: " + spec.name)
        specs[spec.name], families[spec.name], checkers[spec.name] = spec, family, checker
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
    for module, checker in zip(v6, v6_checkers):
        for spec in module.SPECS:
            add(spec, 6, module, checker)

    arguments: dict[int | None, dict[str, frozenset[str]]] = {None: {}, 5: {}, 6: {}}
    for spec in specs.values():
        arguments[None][spec.name] = arguments[spec.version][spec.name] = frozenset(
            argument.name for argument in spec.arguments)
    types = {5: _record_types(v5, 5), 6: _record_types(v6, 6)}
    areas = {area: tuple(sorted((spec for spec in specs.values() if spec.area == area),
                                key=lambda spec: spec.public_name)) for area in AREAS}
    tools = {tool: tuple(sorted((spec for spec in specs.values() if spec.mcp and spec.area in served),
                                key=lambda spec: spec.name)) for tool, served in MCP_TOOLS.items()}
    return _Tables(
        specs=MappingProxyType(specs), families=MappingProxyType(families),
        checkers=MappingProxyType(checkers), modules=MappingProxyType({5: v5, 6: v6}),
        arguments=MappingProxyType({key: MappingProxyType(value) for key, value in arguments.items()}),
        types=MappingProxyType({key: MappingProxyType(value) for key, value in types.items()}),
        all_types=frozenset(cls for table in types.values() for cls in table.values()),
        areas=MappingProxyType(areas), tools=MappingProxyType(tools))


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


def family(name: str) -> ModuleType:
    """The producer module of an operation."""
    try:
        return _tables().families[name]
    except KeyError:
        raise UnsupportedOperation("Unknown operation") from None


def checker(name: str) -> ModuleType:
    """The checker module of an operation: the family itself for v5, the family's CHECKER for v6."""
    try:
        return _tables().checkers[name]
    except KeyError:
        raise UnsupportedOperation("Unknown operation") from None


def operation_arguments(version: int | None = None) -> Mapping[str, frozenset[str]]:
    """Argument names by operation, for one result format version or (None) for all."""
    return _lookup(_tables().arguments, version, "version")


def record_types(version: int) -> Mapping[str, type]:
    """Record classes by tag; the serialized prefix ("v5_", "v6_") is not part of the tag."""
    return _lookup(_tables().types, version, "version")


def all_record_types() -> frozenset[type]:
    return _tables().all_types


def area_operations(area: str) -> tuple[OperationSpec, ...]:
    """The operations of one area, sorted by public name."""
    return _lookup(_tables().areas, area, "area")


def tool_operations(tool: str) -> tuple[OperationSpec, ...]:
    """The MCP-exposed operations of an area tool's areas, sorted by name."""
    return _lookup(_tables().tools, tool, "tool")


def _family_modules(version: int) -> tuple[ModuleType, ...]:
    """The family modules of one version, in declaration order (dispatch_v5.modules)."""
    return _lookup(_tables().modules, version, "version")
