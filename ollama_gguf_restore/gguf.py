"""Lightweight GGUF reader / validator.

The parser is intentionally small and stdlib-only. It understands the GGUF
header (magic, version, tensor + kv counts), the v2/v3 kv pairs and the tensor
info section. It is used to verify that a blob really is a valid GGUF file and
to summarise it. It does *not* decode tensor data.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

GGUF_MAGIC = b"GGUF"

# GGUF value types
TYP_F32 = 0
TYP_F64 = 1
TYP_I8 = 2
TYP_I16 = 3
TYP_I32 = 4
TYP_I64 = 5
TYP_BOOL = 6
TYP_STR = 7
TYP_ARRAY = 8

_PRIMITIVES = {
    TYP_F32: "<f",
    TYP_F64: "<d",
    TYP_I8: "<b",
    TYP_I16: "<h",
    TYP_I32: "<i",
    TYP_I64: "<q",
    TYP_BOOL: "<?",
}


@dataclass
class GgufKV:
    key: str
    vtype: int


@dataclass
class GgufTensorInfo:
    name: str
    n_dims: int
    dims: List[int]
    ggml_type: int
    offset: int


@dataclass
class GgufInfo:
    """Summary of a GGUF file header (metadata + tensor info, not data)."""

    magic: str = ""
    version: int = 0
    n_tensors: int = 0
    n_kv: int = 0
    metadata_offset: Optional[int] = None  # v3+
    kvs: List[GgufKV] = field(default_factory=list)
    tensors: List[GgufTensorInfo] = field(default_factory=list)
    data_section_offset: Optional[int] = None
    parse_complete: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.parse_complete and not self.error

    def summary(self) -> str:
        lines = [
            f"magic       = {self.magic!r}",
            f"version     = {self.version}",
            f"n_tensors   = {self.n_tensors}",
            f"n_kv        = {self.n_kv}",
        ]
        if self.metadata_offset is not None:
            lines.append(f"meta_off    = {self.metadata_offset}")
        if self.data_section_offset is not None:
            lines.append(f"data_off    = {self.data_section_offset}")
        lines.append(f"complete    = {self.parse_complete}")
        if self.error:
            lines.append(f"error       = {self.error}")
        return "\n".join(lines)


class _Reader:
    """Cursor-based reader over an in-memory byte buffer.

    Alignment in GGUF is always relative to the start of the kv section (for
    kv values) or the start of the tensor-info section, which follows
    immediately after the last kv value.
    """

    __slots__ = ("data", "off")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.off = 0

    def _bounds(self, size: int) -> None:
        if self.off + size > len(self.data):
            raise ValueError(f"read past end of buffer (need {size} at {self.off})")

    def u8(self) -> int:
        self._bounds(1)
        v = self.data[self.off]
        self.off += 1
        return v

    def i32(self) -> int:
        self._bounds(4)
        v = struct.unpack_from("<i", self.data, self.off)[0]
        self.off += 4
        return v

    def u32(self) -> int:
        self._bounds(4)
        v = struct.unpack_from("<I", self.data, self.off)[0]
        self.off += 4
        return v

    def i64(self) -> int:
        self._bounds(8)
        v = struct.unpack_from("<q", self.data, self.off)[0]
        self.off += 8
        return v

    def u64(self) -> int:
        self._bounds(8)
        v = struct.unpack_from("<Q", self.data, self.off)[0]
        self.off += 8
        return v

    def string_terminated(self) -> str:
        start = self.off
        end = self.data.find(b"\0", start)
        if end == -1:
            raise ValueError(f"unterminated string at offset {start}")
        s = self.data[start:end].decode("utf-8", errors="replace")
        self.off = end + 1
        return s

    def _align_to_8(self, base: int) -> None:
        """Pad so the next byte is 8-aligned relative to ``base``."""
        delta = self.off - base
        rem = delta % 8
        if rem:
            pad = 8 - rem
            self._bounds(pad)
            self.off += pad

    def skip_value(self, vtype: int, base: int) -> None:
        """Skip the bytes of a single GGUF value of ``vtype``.

        ``base`` is the alignment origin (start of the kv section or the start
        of the tensor-info section).
        """
        if vtype in _PRIMITIVES:
            fmt = _PRIMITIVES[vtype]
            size = struct.calcsize(fmt)
            self._bounds(size)
            self.off += size
            return
        if vtype == TYP_STR:
            length = self.u64()
            self._bounds(length)
            self.off += length
            self._align_to_8(base)
            return
        if vtype == TYP_ARRAY:
            _count = self.i32()
            _elem = self.i32()
            return  # array skipped lazily: caller should recurse; we just consume header
        raise ValueError(f"unsupported GGUF value type {vtype}")

    def skip_value_recursive(self, vtype: int, base: int) -> None:
        """Fully consume a value including array element bytes."""
        if vtype in _PRIMITIVES:
            fmt = _PRIMITIVES[vtype]
            size = struct.calcsize(fmt)
            self._bounds(size)
            self.off += size
            return
        if vtype == TYP_STR:
            length = self.u64()
            self._bounds(length)
            self.off += length
            self._align_to_8(base)
            return
        if vtype == TYP_ARRAY:
            count = self.i32()
            elem = self.i32()
            for _ in range(count):
                self.skip_value_recursive(elem, base)
            return
        raise ValueError(f"unsupported GGUF value type {vtype}")


def parse_gguf_header(
    data: bytes,
    max_tensors: int = 1_000_000,
    max_kvs: int = 1_000_000,
) -> GgufInfo:
    """Parse a GGUF file header from an in-memory buffer.

    Never raises; returns a :class:`GgufInfo` whose ``error`` field is set when
    the buffer is not a structurally valid GGUF.
    """
    info = GgufInfo()
    try:
        if len(data) < 8:
            info.error = f"buffer too small ({len(data)} bytes)"
            return info
        if data[:4] != GGUF_MAGIC:
            info.error = f"bad magic {data[:4]!r}"
            return info
        info.magic = "GGUF"

        r = _Reader(data)
        r.u32()  # magic
        version = r.u32()
        info.version = version
        if version not in (2, 3):
            info.parse_complete = False
            info.error = f"unsupported GGUF version {version}"
            return info
        n_tensors = r.u64()
        n_kv = r.u64()
        info.n_tensors = n_tensors
        info.n_kv = n_kv
        if n_tensors > max_tensors or n_kv > max_kvs:
            info.parse_complete = False
            info.error = "tensor/kv count implausible"
            return info

        # v3 inserts a metadata_offset u64 right after the counts; jumping to
        # it is harmless (in practice it points straight at the kv section).
        kv_start = r.off
        if version >= 3:
            metadata_offset = r.u64()
            info.metadata_offset = metadata_offset
            if metadata_offset < kv_start:
                info.parse_complete = False
                info.error = f"metadata_offset {metadata_offset} before kv start"
                return info
            kv_start = metadata_offset
            r.off = metadata_offset
        for _ in range(n_kv):
            key = r.string_terminated()
            vtype = r.i32()
            info.kvs.append(GgufKV(key, vtype))
            r.skip_value_recursive(vtype, kv_start)
        kv_end = r.off

        # ---- tensor info section ----
        # Per the GGUF spec, the tensor info section starts right after the
        # last kv value. Its alignment base is kv_start as well.
        for _ in range(n_tensors):
            name = r.string_terminated()
            n_dims = r.u32()
            if n_dims > 16:
                raise ValueError(f"n_dims too large ({n_dims})")
            dims = [r.u64() for _ in range(n_dims)]
            ggml_type = r.u32()
            offset = r.u64()
            info.tensors.append(GgufTensorInfo(name, n_dims, dims, ggml_type, offset))
        data_start = r.off
        info.data_section_offset = data_start

        # ---- data section bounds sanity ----
        if data_start > len(data):
            info.parse_complete = False
            info.error = f"data section start {data_start} exceeds buffer {len(data)}"
            return info
        info.parse_complete = True
        return info
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        info.parse_complete = False
        info.error = f"{type(exc).__name__}: {exc}"
        return info


def is_gguf(path_or_bytes) -> bool:
    """Cheap magic check on a file path or a bytes object."""
    try:
        if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
            return bytes(path_or_bytes[:4]) == GGUF_MAGIC
        with open(path_or_bytes, "rb") as fh:
            magic = fh.read(4)
        return magic == GGUF_MAGIC
    except (OSError, ValueError):
        return False


def sniff_header(path: str, max_bytes: int = 4 * 1024 * 1024) -> GgufInfo:
    """Read up to ``max_bytes`` from ``path`` and parse the header.

    For very large GGUF files we cannot read the whole file just to inspect
    metadata; we cap the bytes we pull. Tensor offsets that live beyond the
    cap will make the read "incomplete", which is fine for a first-pass check
    and is reflected in ``parse_complete``.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(max_bytes)
    except OSError as exc:
        return GgufInfo(error=str(exc))
    info = parse_gguf_header(head)
    return info
