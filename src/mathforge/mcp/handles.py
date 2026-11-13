"""Result handles: content names for the results of MCP responses, and a bounded store of this process.

A handle is "mf:sha256:" followed by the 64 lowercase hexadecimal digits of the SHA-256 digest of a result's
canonical JSON: the compact, key-sorted, ASCII text that mathforge.serialization.dumps writes, so that the handle
mf:sha256:<h> of a response and the workspace ObjectRef sha256:<h> of the same value always name the same document.

The store keeps the canonical text of recent results under their handles, at most MAX_DOCUMENTS documents and
MAX_BYTES characters of text in all, and evicts the least recently used. It lives in this process only: a handle
from another process, or one that has been evicted, is unknown. get() recomputes the digest of the text it holds
before returning it, so the store never returns a document other than the one that a handle names.

This module stores and names text only; mcp.boundary decodes a resolved document as it decodes one sent in full.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import re
import threading
from typing import Any

PREFIX = "mf:sha256:"
PATTERN = "^mf:sha256:[0-9a-f]{64}$"  # The JSON Schema form; is_handle() matches the whole string.
MAX_DOCUMENTS = 256
MAX_BYTES = 64 * 1024 * 1024
_HANDLE = re.compile(r"mf:sha256:[0-9a-f]{64}")


def is_handle(value: Any) -> bool:
    """Whether a value is a reference: exactly the handle pattern, nothing before or after it."""
    return type(value) is str and _HANDLE.fullmatch(value) is not None


def canonical_text(document: Any) -> str:
    """The canonical JSON of a document: serialization.dumps's rule, the input of workspace content references."""
    return json.dumps(document, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def handle_of_text(text: str) -> str:
    return PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def handle_of(document: Any) -> str:
    """The handle of a JSON document, such as a response's "result"."""
    return handle_of_text(canonical_text(document))


class HandleStore:
    """A thread-safe, bounded, least-recently-used map from handles to canonical JSON text."""

    def __init__(self, max_documents: int = MAX_DOCUMENTS, max_bytes: int = MAX_BYTES):
        self.max_documents = max_documents
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, str] = OrderedDict()
        self._bytes = 0

    def put(self, handle: str, text: str) -> bool:
        """Keep a document; False, and nothing kept, when the text alone exceeds the byte cap.

        The text is ASCII (canonical_text escapes the rest), so its length is its size in bytes.
        """
        size = len(text)
        if size > self.max_bytes:
            return False
        with self._lock:
            if handle in self._entries:
                self._entries.move_to_end(handle)
                return True
            self._entries[handle] = text
            self._bytes += size
            while len(self._entries) > self.max_documents or self._bytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= len(evicted)
        return True

    def get(self, handle: str) -> str | None:
        """The text that a handle names, or None when this store holds no such document."""
        with self._lock:
            text = self._entries.get(handle)
            if text is None:
                return None
            self._entries.move_to_end(handle)
        if handle_of_text(text) != handle:  # Hashed outside the lock; a mismatching entry is dropped, never returned.
            with self._lock:
                if self._entries.get(handle) is text:
                    del self._entries[handle]
                    self._bytes -= len(text)
            return None
        return text

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, handle: object) -> bool:
        with self._lock:
            return handle in self._entries

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes


STORE = HandleStore()


def remember(document: Any) -> str:
    """Keep a document in the process store and return its handle."""
    text = canonical_text(document)
    handle = handle_of_text(text)
    STORE.put(handle, text)
    return handle


def clear() -> None:
    """Forget every stored document, as tests do between cases."""
    STORE.clear()
