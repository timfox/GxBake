"""KeyValues3: the binary encodings inside Source 2 resources (DATA, CTRL, ... blocks).

Covers every binary layout ValveResourceFormat reads: the legacy VKV3 header
(uncompressed, block compressed or LZ4) and KV3 versions 1 to 5 (uncompressed,
LZ4 or Zstandard, with binary blobs). Ported from ValveResourceFormat's
BinaryKV3 (MIT, see THIRD_PARTY_NOTICES.md).

Values come back as plain Python: dict, list, str, int, float, bool, None and
bytes. Typed arrays of numbers come back as numpy arrays (uint8 ones as bytes),
so a vertex buffer stays one buffer instead of a list of a million ints.
"""

from __future__ import annotations

import struct

import numpy as np

from . import valvecomp

MAGIC_VKV3 = 0x03564B56
_KV3_SIG = 0x4B563300

# KV3 node types
_NULL, _BOOL, _INT64, _UINT64, _DOUBLE, _STRING, _BLOB, _ARRAY, _OBJECT, _ARRAY_TYPED = range(1, 11)
_INT32, _UINT32, _TRUE, _FALSE, _I64_0, _I64_1, _D_0, _D_1, _FLOAT, _INT16, _UINT16, _INT8, _UINT8 = range(11, 24)
_ARRAY_BYTELEN, _ARRAY_AUX, _U32_0, _U32_1 = 24, 25, 26, 27

_CONST = {_NULL: None, _TRUE: True, _FALSE: False, _I64_0: 0, _I64_1: 1, _D_0: 0.0, _D_1: 1.0, _U32_0: 0, _U32_1: 1}
# type -> (lane width, numpy dtype)
_LANE = {
	_BOOL: (1, "u1"), _INT8: (1, "i1"), _UINT8: (1, "u1"),
	_INT16: (2, "<i2"), _UINT16: (2, "<u2"),
	_INT32: (4, "<i4"), _UINT32: (4, "<u4"), _FLOAT: (4, "<f4"),
	_INT64: (8, "<i8"), _UINT64: (8, "<u8"), _DOUBLE: (8, "<f8"),
}


def is_kv3(data: bytes, pos: int = 0) -> bool:
	if len(data) - pos < 4:
		return False
	m = struct.unpack_from("<I", data, pos)[0]
	return m == MAGIC_VKV3 or (m & 0xFFFFFF00) == _KV3_SIG and 1 <= (m & 0xFF) <= 5


class _Lane:
	"""One byte width of a KV3 buffer, read front to back."""

	__slots__ = ("buf", "pos", "end")

	def __init__(self, buf: bytes = b"", pos: int = 0, end: int = 0):
		self.buf, self.pos, self.end = buf, pos, end

	def take(self, width: int, count: int = 1) -> bytes:
		p = self.pos
		self.pos += width * count
		if self.pos > self.end:
			raise ValueError("KV3: read past the end of a buffer lane")
		return self.buf[p:self.pos]


class _Buffers:
	__slots__ = ("l1", "l2", "l4", "l8")

	def __init__(self):
		self.l1 = self.l2 = self.l4 = self.l8 = _Lane()

	def lane(self, width: int) -> _Lane:
		return {1: self.l1, 2: self.l2, 4: self.l4, 8: self.l8}[width]


class _Ctx:
	def __init__(self, version: int):
		self.version = version
		self.types = b""
		self.tpos = 0
		self.strings: list[str] = []
		self.buf = _Buffers()
		self.aux = _Buffers()
		self.obj_lengths: _Lane | None = None
		self.blobs: list[bytes] = []
		self.next_blob = 0


def _align(n: int, a: int) -> int:
	return (n + a - 1) // a * a


def _read_strings(buf: bytes, pos: int, count: int) -> tuple[list[str], int]:
	out = []
	for _ in range(count):
		end = buf.index(b"\0", pos)
		out.append(buf[pos:end].decode("utf-8", "replace"))
		pos = end + 1
	return out, pos


def _read_type(ctx: _Ctx) -> int:
	t = ctx.types[ctx.tpos]
	ctx.tpos += 1
	if ctx.version >= 3:
		if t & 0x80:
			ctx.tpos += 1	# flag byte (resource, resource name, panorama, ...)
		if t & 0x40:
			ctx.tpos += 1
		return t & 0x3F
	if t & 0x80:
		ctx.tpos += 1
	return t & 0x7F


def _scalar(t: int, lane: _Buffers):
	w, dt = _LANE[t]
	v = np.frombuffer(lane.lane(w).take(w), dt)[0]
	if t == _BOOL:
		return bool(v)
	if t in (_FLOAT, _DOUBLE):
		return float(v)
	return int(v)


def _packed(t: int, lane: _Buffers, count: int):
	w, dt = _LANE[t]
	raw = lane.lane(w).take(w, count)
	if t == _UINT8:
		return bytes(raw)
	arr = np.frombuffer(raw, dt).copy()
	if t == _BOOL:
		return [bool(x) for x in arr]
	return arr


def _value(ctx: _Ctx, t: int, lane: _Buffers):
	if t in _CONST:
		return _CONST[t]
	if t in _LANE:
		return _scalar(t, lane)
	buf = ctx.buf
	if t == _STRING:
		i = struct.unpack("<i", buf.l4.take(4))[0]
		return ctx.strings[i] if 0 <= i < len(ctx.strings) else ""
	if t == _BLOB:
		if ctx.version < 2:
			n = struct.unpack("<i", buf.l4.take(4))[0]
			return bytes(buf.l1.take(1, n)) if n > 0 else b""
		b = ctx.blobs[ctx.next_blob]
		ctx.next_blob += 1
		return b
	if t == _ARRAY:
		n = struct.unpack("<i", buf.l4.take(4))[0]
		return [_value(ctx, _read_type(ctx), ctx.buf) for _ in range(n)]
	if t in (_ARRAY_TYPED, _ARRAY_BYTELEN, _ARRAY_AUX):
		n = struct.unpack("<i", buf.l4.take(4))[0] if t == _ARRAY_TYPED else buf.l1.take(1)[0]
		sub = _read_type(ctx)
		el = ctx.aux if t == _ARRAY_AUX else buf
		if sub in _LANE:
			return _packed(sub, el, n)
		return [_value(ctx, sub, el) for _ in range(n)]
	if t == _OBJECT:
		if ctx.version >= 5:
			n = struct.unpack("<i", ctx.obj_lengths.take(4))[0]
		else:
			n = struct.unpack("<i", buf.l4.take(4))[0]
		out = {}
		for _ in range(n):
			st = _read_type(ctx)
			i = struct.unpack("<i", ctx.buf.l4.take(4))[0]
			name = ctx.strings[i] if 0 <= i < len(ctx.strings) else ""
			out[name] = _value(ctx, st, ctx.buf)
		return out
	raise ValueError(f"KV3: unknown node type {t}")


def _decompress(method: int, data: bytes, pos: int, csize: int, usize: int) -> bytes:
	chunk = data[pos:pos + csize]
	if method == 0:
		return bytes(data[pos:pos + usize])
	if method == 1:
		return valvecomp.lz4_block(chunk, usize)
	if method == 2:
		return valvecomp.zstd(chunk, usize)[:usize] if usize else b""
	raise ValueError(f"KV3: unknown compression {method}")


def read(data: bytes, pos: int = 0, end: int | None = None):
	"""Parse the KV3 document at data[pos:end] and return its root value."""
	end = len(data) if end is None else end
	magic = struct.unpack_from("<I", data, pos)[0]
	if magic == MAGIC_VKV3:
		return _read_legacy(data, pos + 4, end)
	if (magic & 0xFFFFFF00) != _KV3_SIG:
		if data[pos:pos + 4] == b"<!--":
			from .kv3text import parse

			return parse(data[pos:end].split(b"\0", 1)[0].decode("utf-8", "replace"))
		raise ValueError("not a KV3 block")
	version = magic & 0xFF
	p = pos + 4 + 16	# magic, format GUID
	method = struct.unpack_from("<I", data, p)[0]
	p += 4
	frame = 0
	c2 = c_types = blocks = blob_bytes = sizes_bytes = 0
	u1 = cs1 = u2 = cs2 = 0
	c1b2 = c2b2 = c4b2 = c8b2 = objs2 = 0
	if version == 1:
		c1, c4, c8, u_total = struct.unpack_from("<4i", data, p)
		p += 16
		c_total = end - p
	else:
		_dict, frame = struct.unpack_from("<HH", data, p)
		p += 4
		c1, c4, c8, c_types = struct.unpack_from("<4i", data, p)
		p += 16 + 4	# + object and array counts (uint16 each)
		u_total, c_total, blocks, blob_bytes = struct.unpack_from("<4i", data, p)
		p += 16
	if version >= 4:
		c2, sizes_bytes = struct.unpack_from("<2i", data, p)
		p += 8
	if version >= 5:
		u1, cs1, u2, cs2, c1b2, c2b2, c4b2, c8b2 = struct.unpack_from("<8i", data, p)
		p += 32
		_nodes, objs2, _arrays, _elements = struct.unpack_from("<4i", data, p)
		p += 16
	else:
		u1, cs1 = u_total, c_total
	zstd_joined = version < 5 and method == 2
	b1 = _decompress(method, data, p, cs1, u1 + (blob_bytes if zstd_joined else 0))
	p += cs1 if method else u1
	ctx = _Ctx(version)
	buf1 = _Buffers()
	off = 0

	def lane(width: int, count: int) -> _Lane:
		nonlocal off
		if count <= 0:
			return _Lane(b1, off, off)
		off = _align(off, width)
		ln = _Lane(b1, off, off + width * count)
		off += width * count
		return ln

	buf1.l1 = lane(1, c1)
	buf1.l2 = lane(2, c2)
	buf1.l4 = lane(4, c4)
	buf1.l8 = lane(8, c8)
	if c8 <= 0 and version < 5:
		off = _align(off, 8)
	nstrings = struct.unpack("<i", buf1.l4.take(4))[0]
	ctx.aux = buf1
	blob_sizes_at = None
	if version >= 5:
		ctx.strings, buf1.l1.pos = _read_strings(b1, buf1.l1.pos, nstrings)
	else:
		ctx.buf = buf1
		start = off
		ctx.strings, off = _read_strings(b1, off, nstrings)
		tlen = (u_total - off - 4) if version == 1 else (c_types - off + start)
		ctx.types = b1[off:off + tlen]
		off += tlen
		if blocks == 0:
			if struct.unpack_from("<I", b1, off)[0] != 0xFFEEDD00:
				raise ValueError("KV3: bad trailer")
		else:
			blob_sizes_at = (b1, off)
	if version >= 5:
		b2 = _decompress(method, data, p, cs2, u2)
		p += cs2 if method else u2
		buf2 = _Buffers()
		ctx.buf = buf2
		ctx.obj_lengths = _Lane(b2, 0, objs2 * 4)
		off2 = objs2 * 4
		for width, count in ((1, c1b2), (2, c2b2), (4, c4b2), (8, c8b2)):
			if count > 0:
				off2 = _align(off2, width)
				setattr(buf2, f"l{width}", _Lane(b2, off2, off2 + width * count))
				off2 += width * count
		ctx.types = b2[off2:off2 + c_types]
		off2 += c_types
		if blocks == 0:
			if struct.unpack_from("<I", b2, off2)[0] != 0xFFEEDD00:
				raise ValueError("KV3: bad trailer")
		else:
			blob_sizes_at = (b2, off2)
	if blocks:
		sb, so = blob_sizes_at
		lengths = struct.unpack_from(f"<{blocks}i", sb, so)
		so += blocks * 4
		if struct.unpack_from("<I", sb, so)[0] != 0xFFEEDD00:
			raise ValueError("KV3: bad blob trailer")
		so += 4
		if method == 0:
			for n in lengths:
				ctx.blobs.append(bytes(data[p:p + n]))
				p += n
		elif method == 1:
			history = bytearray()
			for n in lengths:
				blob = bytearray()
				while len(blob) < n:
					clen = struct.unpack_from("<H", sb, so)[0]
					so += 2
					want = min(frame or 16384, n - len(blob))
					part = valvecomp.lz4_block(data[p:p + clen], want, history)
					p += clen
					blob += part
					history += part
					if len(history) > 1 << 17:
						del history[:-65536]
				ctx.blobs.append(bytes(blob))
		else:
			if version >= 5:
				raw = valvecomp.zstd(data[p:p + (c_total - cs1 - cs2)], blob_bytes)
				p += c_total - cs1 - cs2
			else:
				raw = b1[u1:u1 + blob_bytes]
			q = 0
			for n in lengths:
				ctx.blobs.append(bytes(raw[q:q + n]))
				q += n
	root_t = _read_type(ctx)
	return _value(ctx, root_t, ctx.buf)


def _read_legacy(data: bytes, p: int, end: int):
	encoding = bytes(data[p:p + 16])
	p += 32	# encoding GUID, format GUID
	if encoding == _ENC_BC:
		buf, _n = valvecomp.block_decompress(data, p)
	elif encoding == _ENC_LZ4:
		size = struct.unpack_from("<i", data, p)[0]
		buf = valvecomp.lz4_block(data[p + 4:end], size)
	else:
		buf = bytes(data[p:end])
	return _LegacyReader(buf).root()


_ENC_BC = bytes([0x46, 0x1A, 0x79, 0x95, 0xBC, 0x95, 0x6C, 0x4F, 0xA7, 0x0B, 0x05, 0xBC, 0xA1, 0xB7, 0xDF, 0xD2])
_ENC_LZ4 = bytes([0x8A, 0x34, 0x47, 0x68, 0xA1, 0x63, 0x5C, 0x4F, 0xA1, 0x97, 0x53, 0x80, 0x6F, 0xD9, 0xB1, 0x19])


class _LegacyReader:
	def __init__(self, buf: bytes):
		self.b = buf
		self.p = 0

	def u(self, fmt: str):
		v = struct.unpack_from(fmt, self.b, self.p)
		self.p += struct.calcsize(fmt)
		return v[0]

	def root(self):
		n = self.u("<I")
		self.strings, self.p = _read_strings(self.b, self.p, n)
		v = self.value(self.typ())
		return v

	def typ(self) -> int:
		t = self.b[self.p]
		self.p += 1
		if t & 0x80:
			self.p += 1
		return t & 0x7F

	def string(self) -> str:
		i = self.u("<i")
		return self.strings[i] if 0 <= i < len(self.strings) else ""

	def value(self, t: int):
		if t in _CONST:
			return _CONST[t]
		if t == _BOOL:
			return self.u("<B") != 0
		if t == _INT64:
			return self.u("<q")
		if t == _UINT64:
			return self.u("<Q")
		if t == _INT32:
			return self.u("<i")
		if t == _UINT32:
			return self.u("<I")
		if t == _DOUBLE:
			return self.u("<d")
		if t == _STRING:
			return self.string()
		if t == _BLOB:
			n = self.u("<i")
			b = self.b[self.p:self.p + n]
			self.p += n
			return bytes(b)
		if t == _ARRAY:
			return [self.value(self.typ()) for _ in range(self.u("<i"))]
		if t == _ARRAY_TYPED:
			n = self.u("<i")
			sub = self.typ()
			return [self.value(sub) for _ in range(n)]
		if t == _OBJECT:
			out = {}
			for _ in range(self.u("<i")):
				name = self.string()
				out[name] = self.value(self.typ())
			return out
		raise ValueError(f"KV3: unknown legacy node type {t}")


# -- small helpers for consumers ----------------------------------------------


def floats(v, n: int | None = None) -> "np.ndarray":
	"""A KV3 vector/array (list or numpy) as float64."""
	a = np.asarray(v, np.float64).ravel()
	return a if n is None else a[:n]


def ints(v) -> "np.ndarray":
	if v is None:
		return np.zeros(0, np.int64)
	if isinstance(v, (bytes, bytearray)):
		return np.frombuffer(v, np.uint8).astype(np.int64)
	return np.asarray(v, np.int64).ravel()


def blob(v) -> bytes:
	"""Binary data stored either as a blob or as a typed uint8 array."""
	if v is None:
		return b""
	if isinstance(v, (bytes, bytearray)):
		return bytes(v)
	return np.asarray(v, np.uint8).tobytes()
