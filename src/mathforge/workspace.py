"""Optional in-memory storage of immutable mathematical records.

References identify structural content, never mathematical equivalence.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from .errors import InvalidInput, SerializationError


@dataclass(frozen=True, slots=True)
class ObjectRef:
    identifier: str

    def __post_init__(self) -> None:
        if type(self.identifier) is not str or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.identifier
        ):
            raise InvalidInput("An object reference must contain a lowercase SHA-256 digest")

    def __str__(self) -> str:
        return self.identifier


@dataclass(frozen=True, slots=True)
class FrozenOptions(Mapping[str, Any]):
    """Deeply immutable JSON options; JSON arrays are represented as tuples."""

    _entries: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        entries = tuple(self._entries)
        if any(not isinstance(item, tuple) or len(item) != 2 for item in entries):
            raise InvalidInput("Options must contain key/value pairs")
        if any(type(key) is not str for key, _ in entries):
            raise InvalidInput("Option keys must be strings")
        if len({key for key, _ in entries}) != len(entries):
            raise InvalidInput("Duplicate option key")
        object.__setattr__(self, "_entries", tuple(sorted(
            (key, _freeze_json(value)) for key, value in entries
        )))

    def __getitem__(self, key: str) -> Any:
        for name, value in self._entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def _freeze_json(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidInput("Options cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return FrozenOptions(tuple(value.items()))
    if type(value) in (list, tuple):
        return tuple(_freeze_json(item) for item in value)
    raise InvalidInput("Options must be JSON-compatible values")


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation: str
    inputs: tuple[tuple[str, ObjectRef], ...]
    options: FrozenOptions
    result_ref: ObjectRef

    def __post_init__(self) -> None:
        if type(self.operation) is not str or not self.operation.strip():
            raise InvalidInput("An operation name is required")
        pairs = tuple(self.inputs.items()) if isinstance(self.inputs, Mapping) else tuple(self.inputs)
        if any(not isinstance(item, tuple) or len(item) != 2 for item in pairs):
            raise InvalidInput("Inputs must contain named references")
        if any(type(name) is not str or not name or type(ref) is not ObjectRef
               for name, ref in pairs):
            raise InvalidInput("Inputs must contain named references")
        if len({name for name, _ in pairs}) != len(pairs):
            raise InvalidInput("Duplicate operation input name")
        if not isinstance(self.options, Mapping):
            raise InvalidInput("Operation options must be a mapping")
        if type(self.result_ref) is not ObjectRef:
            raise InvalidInput("Operation result must be an object reference")
        object.__setattr__(self, "inputs", tuple(sorted(pairs)))
        object.__setattr__(self, "options", _freeze_json(self.options))


def _content_ref(value: Any) -> ObjectRef:
    from .serialization import dumps

    return ObjectRef("sha256:" + hashlib.sha256(dumps(value).encode("utf-8")).hexdigest())


class Workspace:
    """Content-addressed immutable objects with explicit operation history."""

    __slots__ = ("_objects", "_history")

    def __init__(self) -> None:
        self._objects: dict[str, Any] = {}
        self._history: list[OperationRecord] = []

    @property
    def objects(self) -> Mapping[ObjectRef, Any]:
        return MappingProxyType({ObjectRef(key): value for key, value in self._objects.items()})

    @property
    def history(self) -> tuple[OperationRecord, ...]:
        return tuple(self._history)

    def put(self, value: Any, *, decode_limits=None, computation_limits=None) -> ObjectRef:
        from .serialization import _check_symbol_consistency, dumps, loads

        if isinstance(value, Workspace):
            raise InvalidInput("A mutable workspace cannot be stored as an object")
        # Reconstruct through the allowlisted codec to reject mutable/foreign payloads.
        snapshot = loads(dumps(value), decode_limits=decode_limits,
                         computation_limits=computation_limits)
        _check_symbol_consistency((*self._objects.values(), snapshot))
        reference = _content_ref(snapshot)
        self._validate_references(snapshot)
        self._objects.setdefault(reference.identifier, snapshot)
        return reference

    def get(self, reference: ObjectRef) -> Any:
        if type(reference) is not ObjectRef:
            raise InvalidInput("get() requires an ObjectRef")
        return self._objects[reference.identifier]

    def record(
        self,
        operation: str,
        inputs: Mapping[str, ObjectRef],
        options: Mapping[str, Any],
        result_ref: ObjectRef,
    ) -> OperationRecord:
        if not isinstance(inputs, Mapping) or not isinstance(options, Mapping):
            raise InvalidInput("Inputs and options must be mappings")
        record = OperationRecord(operation, tuple(inputs.items()), _freeze_json(options), result_ref)
        self._validate_references(record)
        self._history.append(record)
        return record

    def _validate_references(self, value: Any) -> None:
        pending = [value]
        visited = set()
        while pending:
            item = pending.pop()
            if type(item) is ObjectRef:
                if item.identifier not in self._objects:
                    raise SerializationError(f"Unknown object reference: {item.identifier}")
            elif type(item) is tuple:
                if id(item) not in visited:
                    visited.add(id(item))
                    pending.extend(item)
            elif is_dataclass(item) and not isinstance(item, type):
                if id(item) not in visited:
                    visited.add(id(item))
                    pending.extend(getattr(item, field.name) for field in fields(item))

    def save(self, path: str | os.PathLike[str]) -> None:
        from .serialization import dumps

        destination = Path(path)
        payload = dumps(self, indent=2)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, decode_limits=None,
             computation_limits=None) -> Workspace:
        from .serialization import _limits, _resource_limit, loads

        limits = _limits(decode_limits)
        with Path(path).open("rb") as stream:
            payload = stream.read(limits.max_bytes + 1)
        if len(payload) > limits.max_bytes:
            _resource_limit("Workspace file exceeds the configured byte limit")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SerializationError("Workspace file must contain UTF-8 JSON") from exc
        value = loads(text, decode_limits=limits, computation_limits=computation_limits)
        if not isinstance(value, Workspace):
            raise SerializationError("The document does not contain a Workspace")
        return value

    @classmethod
    def _restore(
        cls, objects: dict[str, Any], history: tuple[OperationRecord, ...]
    ) -> Workspace:
        from .serialization import _check_symbol_consistency

        candidate = cls()
        candidate._objects = dict(objects)
        _check_symbol_consistency(candidate._objects.values())
        for identifier, value in candidate._objects.items():
            if isinstance(value, Workspace):
                raise SerializationError("Nested workspaces are not immutable objects")
            if _content_ref(value).identifier != identifier:
                raise SerializationError(f"Object content hash mismatch: {identifier}")
            candidate._validate_references(value)
        for record in history:
            candidate._validate_references(record)
        candidate._history = list(history)
        return candidate
