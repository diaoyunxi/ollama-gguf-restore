"""Lightweight GGUF reader / validator.

The parser is stdlib-only.  It verifies structural validity and summarises
header fields.  It does **not** decode tensor data.

Two GGUF families exist:

* **standard** (llama.cpp / gguf.c):  null-terminated keys, string values
  use ``u64 length + bytes`` with 8-byte alignment, value type table:
  f32=0, f64=1, i8=2, i16=3, i32=4, i64=5, bool=6, string=7, array=8.

* **ollama** (Ollama's Go writer, ``fs/gguf/gguf.go``):  every string
  (key or value) is encoded as ``u64 length + exactly length bytes`` with
  **no** null terminator and **no** alignment padding.  The value type table
  is renumbered:  uint8=0, int8=1, uint16=2, int16=3, uint32=4, int32=5,
  float32=6, bool=7, string=8, array=9, uint64=10, int64=11, float64=12.

  The KV section starts at offset 24 (right after ``n_kv``) in both
  families; the ollama v3 header has no extra field.

Because this tool is meant to verify Ollama blobs, the parser auto-detects
the family by trying each hypothesis and picking the one that parses the
full metadata cleanly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

GGUF_MAGIC = b"GGUF"
GGUF_MAGIC_BE = b"FUGG"


# ---- value-type size tables ---------------------------------------------

# Ollama Go writer (fs/gguf/gguf.go)
# uint8,i8,uint16,int16,uint32,int32,float32,bool,string,array,uint64,int64,float64
_OLLAMA_PRIM = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_OLLAMA_STR, _OLLAMA_ARR = 8, 9
_OLLAMA_ARR_COUNT_SIZE = 8   # u64 count in Ollama

# standard gguf.c
# f32,f64,i8,i16,i32,i64,bool,string,array
_STANDARD_PRIM = {0: 4, 1: 8, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1}
_STANDARD_STR, _STANDARD_ARR = 7, 8
_STANDARD_ARR_COUNT_SIZE = 4  # i32 count in standard


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
    kv_start: int = 0
    encoding: str = ""  # "ollama" | "standard"
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
            f"kv_start    = {self.kv_start}",
            f"encoding    = {self.encoding or '?'}",
        ]
        if self.data_section_offset is not None:
            lines.append(f"data_off    = {self.data_section_offset}")
        lines.append(f"complete    = {self.parse_complete}")
        if self.error:
            lines.append(f"error       = {self.error}")
        return "\n".join(lines)


class _Cursor:
    __slots__ = ("data", "off")

    def __init__(self, data: bytes, off: int = 0) -> None:
        self.data = data
        self.off = off

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.data, self.off)[0]
        self.off += 4
        return v

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.data, self.off)[0]
        self.off += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.data, self.off)[0]
        self.off += 8
        return v

    def nul_terminated(self) -> bytes:
        end = self.data.find(b"\0", self.off)
        seg = self.data[self.off:end]
        self.off = end + 1
        return seg


def _skip_value(cur: _Cursor, vtype: int, enc: str, base: int = 0) -> None:
    """Advance past one value of ``vtype`` under encoding family ``enc``.

    ``base`` is the alignment origin (start of the metadata section).  The
    standard family pads every string value to 8 bytes relative to ``base``;
    the Ollama family packs values tightly with no padding.

    May raise ``ValueError`` on an implausible type / count.  Reading past
    the end of a truncated buffer also raises ``ValueError``; the caller
    treats that as "incomplete" rather than "invalid".
    """
    if enc == "ollama":
        prim, str_id, arr_id, cnt_sz = _OLLAMA_PRIM, _OLLAMA_STR, _OLLAMA_ARR, _OLLAMA_ARR_COUNT_SIZE
        align = False
    else:
        prim, str_id, arr_id, cnt_sz = _STANDARD_PRIM, _STANDARD_STR, _STANDARD_ARR, _STANDARD_ARR_COUNT_SIZE
        align = True

    if vtype in prim:
        cur.off += prim[vtype]
        return
    if vtype == str_id:
        ln = cur.u64()
        if ln > (16 << 20):
            raise ValueError(f"string length {ln} implausible")
        cur.off += ln
        if align:
            delta = cur.off - base
            cur.off += (-delta) % 8
        return
    if vtype == arr_id:
        elem = cur.u32()
        cnt = cur.u64() if cnt_sz == 8 else cur.i32()
        if cnt < 0 or cnt > (64 << 20):
            raise ValueError(f"array count {cnt} implausible")
        for _ in range(cnt):
            _skip_value(cur, elem, enc, base)
        return
    raise ValueError(f"unknown value type {vtype} (family={enc})")


def _read_key(cur: _Cursor, enc: str) -> bytes:
    """Read one key under the given encoding family."""
    if enc == "ollama":
        # u64 length prefix, then exactly `length` bytes (no null)
        ln = cur.u64()
        if ln > (16 << 20):
            raise ValueError(f"key length {ln} implausible")
        key = cur.data[cur.off:cur.off + ln]
        cur.off += ln
        return key
    else:
        # null-terminated
        return cur.nul_terminated()


def _try_parse(
    data: bytes, n_kv: int, n_tensors: int, kv_start: int, enc: str,
) -> Optional[Tuple[int, List[GgufKV], List[GgufTensorInfo], int]]:
    """Attempt one hypothesis.  Returns ``(score, kvs, tensors, data_start)``
    on a clean parse, else ``None``."""
    try:
        cur = _Cursor(data, kv_start)
        kvs: List[GgufKV] = []
        for _ in range(n_kv):
            key_bytes = _read_key(cur, enc)
            vtype = cur.u32()
            _skip_value(cur, vtype, enc, base=kv_start)
            kvs.append(GgufKV(key_bytes.decode("latin1"), vtype))

        tensors: List[GgufTensorInfo] = []
        for _ in range(n_tensors):
            name = _read_key(cur, enc)
            n_dims = cur.u32()
            if n_dims > 16:
                return None
            dims = [cur.u64() for _ in range(n_dims)]
            ggml_type = cur.u32()
            offset = cur.u64()
            tensors.append(GgufTensorInfo(name.decode("latin1"), n_dims, dims, ggml_type, offset))
        data_start = cur.off
    except (ValueError, struct.error, IndexError):
        return None

    # A clean parse must have all keys look like dotted identifiers.
    score = 0
    for kv in kvs:
        if kv.key and all(0x20 <= ord(c) <= 0x7E for c in kv.key):
            score += 1
    return score, kvs, tensors, data_start


def parse_gguf_header(
    data: bytes,
    max_kv: int = 1_000_000,
    max_tensors: int = 1_000_000,
) -> GgufInfo:
    """Parse a GGUF header, auto-detecting the writing family.

    Never raises; returns a :class:`GgufInfo`.  ``data`` may be a truncated
    head (from ``sniff_header``); in that case a truncated parse is reported
    incomplete but not erroneous.
    """
    info = GgufInfo()
    try:
        if len(data) < 8:
            info.error = f"buffer too small ({len(data)} bytes)"
            return info
        if data[:4] == GGUF_MAGIC:
            info.magic = "GGUF"
        elif data[:4] == GGUF_MAGIC_BE:
            info.magic = "FUGG (big-endian)"
        else:
            info.error = f"bad magic {data[:4]!r}"
            return info

        version = struct.unpack_from("<I", data, 4)[0]
        info.version = version
        if version not in (2, 3):
            info.parse_complete = False
            info.error = f"unsupported GGUF version {version}"
            return info

        n_tensors = struct.unpack_from("<Q", data, 8)[0]
        n_kv = struct.unpack_from("<Q", data, 16)[0]
        info.n_tensors = n_tensors
        info.n_kv = n_kv
        if n_tensors > max_tensors or n_kv > max_kv:
            info.parse_complete = False
            info.error = "tensor/kv count implausible"
            return info

        # KV section starts right after the header (offset 24).
        kv_start = 24

        # Try both encoding families; pick the one with the most sensible keys.
        candidates = [
            (kv_start, "ollama"),
            (kv_start, "standard"),
        ]
        best = None  # (score, family, kvs, tensors, data_start)
        for start, enc in candidates:
            if start + 1 > len(data):
                continue
            res = _try_parse(data, n_kv, n_tensors, start, enc)
            if res is None:
                continue
            score, kvs, tensors, data_start = res
            if best is None or score > best[0]:
                best = (score, enc, kvs, tensors, data_start)

        if best is None:
            # Could not walk metadata (likely a truncated sniff).  The header
            # prefix (magic + version + counts) is still trustworthy.
            info.kv_start = kv_start
            if n_kv == 0 and n_tensors == 0 and len(data) >= kv_start:
                info.parse_complete = True
                info.encoding = "ollama"
                return info
            info.parse_complete = False
            info.error = "no encoding hypothesis parsed the metadata"
            return info

        score, enc, kvs, tensors, data_start = best
        info.kv_start = kv_start
        info.encoding = enc
        info.kvs = kvs
        info.tensors = tensors
        info.data_section_offset = data_start

        all_sensible = score == n_kv
        info.parse_complete = all_sensible
        if not all_sensible:
            info.error = f"parsed {score}/{n_kv} sensible kv keys (family={enc})"
        return info
    except Exception as exc:  # noqa: BLE001
        info.parse_complete = False
        info.error = f"{type(exc).__name__}: {exc}"
        return info


def is_gguf(path_or_bytes) -> bool:
    """Cheap magic check on a file path or a bytes object."""
    try:
        if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
            return bytes(path_or_bytes[:4]) in (GGUF_MAGIC, GGUF_MAGIC_BE)
        with open(path_or_bytes, "rb") as fh:
            magic = fh.read(4)
        return magic in (GGUF_MAGIC, GGUF_MAGIC_BE)
    except (OSError, ValueError):
        return False


def sniff_header(path: str, max_bytes: int = 32 * 1024 * 1024) -> GgufInfo:
    """Read up to ``max_bytes`` from ``path`` and parse the header.

    Only a few KB is needed; we cap the read so we never pull a multi-GB
    file just to inspect metadata.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(max_bytes)
    except OSError as exc:
        return GgufInfo(error=str(exc))
    return parse_gguf_header(head)
