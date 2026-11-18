"""Decompressors used by Source 2 resources: LZ4 blocks, Valve block compression, Zstandard, meshoptimizer.

LZ4 and Valve's block compression are small enough to decode in pure Python
(the `lz4` package is used when installed, it is much faster). Zstandard needs
a native decoder: Python 3.14's compression.zstd, or the `zstandard` package
(pip install gxbake[valve]). The meshoptimizer vertex and index codecs are
ported from ValveResourceFormat (MIT, see THIRD_PARTY_NOTICES.md), which ports
them from meshoptimizer (MIT, Arseny Kapoulkine).
"""

from __future__ import annotations

import struct

import numpy as np


# -- LZ4 ----------------------------------------------------------------------


def lz4_block(src: bytes, size: int, history: bytes | bytearray = b"") -> bytes:
	"""Decode one raw LZ4 block to exactly `size` bytes.

	`history` is earlier output that back-references may reach into (chained
	frames); it is not part of the result.
	"""
	try:
		import lz4.block as _lz4

		if history:
			return _lz4.decompress(bytes(src), uncompressed_size=size, dict=bytes(history[-65536:]))
		return _lz4.decompress(bytes(src), uncompressed_size=size)
	except ImportError:
		pass
	base = len(history)
	out = bytearray(history)
	pos = 0
	n = len(src)
	end = base + size
	while pos < n:
		token = src[pos]
		pos += 1
		lit = token >> 4
		if lit == 15:
			while True:
				b = src[pos]
				pos += 1
				lit += b
				if b != 255:
					break
		out += src[pos:pos + lit]
		pos += lit
		if pos >= n or len(out) >= end:
			break
		off = src[pos] | (src[pos + 1] << 8)
		pos += 2
		ml = token & 15
		if ml == 15:
			while True:
				b = src[pos]
				pos += 1
				ml += b
				if b != 255:
					break
		ml += 4
		if off == 0 or off > len(out):
			raise ValueError("LZ4: bad match offset")
		start = len(out) - off
		if off >= ml:
			out += out[start:start + ml]
		else:
			for i in range(ml):
				out.append(out[start + i])
	if len(out) != end:
		raise ValueError(f"LZ4: expected {size} bytes, got {len(out) - base}")
	return bytes(out[base:])


# -- Valve block compression ----------------------------------------------------


def block_decompress(data: bytes, pos: int = 0) -> tuple[bytes, int]:
	"""CBlockCompress::FastDecompress: (output, bytes consumed incl. the size word)."""
	size = struct.unpack_from("<I", data, pos)[0]
	pos += 4
	if size & 0x80000000:
		size &= 0x7FFFFFFF
		return bytes(data[pos:pos + size]), 4 + size
	start = pos - 4
	out = bytearray()
	mask = 0
	left = 0
	while len(out) < size:
		if left == 0:
			mask = data[pos] | (data[pos + 1] << 8)
			pos += 2
			left = 16
		if mask & 1:
			v = data[pos] | (data[pos + 1] << 8)
			pos += 2
			off = (v >> 4) + 1
			ln = (v & 15) + 3
			s = len(out) - off
			if s < 0:
				raise ValueError("block compression: bad back reference")
			if off >= ln:
				out += out[s:s + ln]
			else:
				for i in range(ln):
					out.append(out[s + i])
		else:
			out.append(data[pos])
			pos += 1
		mask >>= 1
		left -= 1
	return bytes(out[:size]), pos - start


# -- Zstandard -----------------------------------------------------------------


def zstd_available() -> bool:
	try:
		from compression import zstd	# noqa: F401  (Python 3.14+)
		return True
	except ImportError:
		pass
	try:
		import zstandard	# noqa: F401
		return True
	except ImportError:
		return False


def zstd(src: bytes, size: int | None = None) -> bytes:
	"""Decode Zstandard frames (one or more, back to back)."""
	try:
		from compression import zstd as _z	# Python 3.14+

		return _z.decompress(bytes(src))
	except ImportError:
		pass
	try:
		import zstandard
	except ImportError as exc:
		raise RuntimeError("this file is Zstandard compressed: pip install zstandard (or gxbake[valve])") from exc
	d = zstandard.ZstdDecompressor()
	out = bytearray()
	buf = bytes(src)
	while buf:
		obj = d.decompressobj()
		out += obj.decompress(buf)
		buf = obj.unused_data
	if size is not None and len(out) < size:
		raise ValueError(f"zstd: expected {size} bytes, got {len(out)}")
	return bytes(out)


# -- meshoptimizer vertex codec ------------------------------------------------

_BYTE_GROUP = 16
_BITS_V0 = (0, 2, 4, 8)
_BITS_V1 = (0, 1, 2, 4, 8)


def _block_size(vertex_size: int) -> int:
	r = (8192 // vertex_size) & ~(_BYTE_GROUP - 1)
	return min(r, 256)


def _decode_group(data: bytes, pos: int, bits: int) -> tuple[list[int], int]:
	if bits == 0:
		return [0] * 16, pos
	if bits == 8:
		return list(data[pos:pos + 16]), pos + 16
	per = 8 // bits
	head = 16 // per	# bytes holding the packed values
	extra = pos + head
	out = []
	full = (1 << bits) - 1
	for i in range(head):
		b = data[pos + i]
		for k in range(per):
			if bits == 1:
				v = (b >> k) & 1	# 1-bit groups are stored least significant bit first
			else:
				v = (b >> (8 - bits * (k + 1))) & full
			if v == full:
				v = data[extra]
				extra += 1
			out.append(v)
	return out, extra


def _decode_bytes(data: bytes, pos: int, count: int, bits_table) -> tuple[list[int], int]:
	groups = count // _BYTE_GROUP
	header = (groups + 3) // 4
	hpos = pos
	pos += header
	out: list[int] = []
	for g in range(groups):
		k = (data[hpos + g // 4] >> ((g % 4) * 2)) & 3
		vals, pos = _decode_group(data, pos, bits_table[k])
		out += vals
	return out, pos


def meshopt_vertex(count: int, size: int, data: bytes) -> bytes:
	"""meshopt_decodeVertexBuffer (versions 0 and 1)."""
	if size <= 0 or size > 256 or size % 4:
		raise ValueError("meshopt: vertex size must be a multiple of 4 up to 256")
	if not data or data[0] & 0xF0 != 0xA0:
		raise ValueError("meshopt: not a vertex buffer")
	version = data[0] & 0x0F
	if version > 1:
		raise ValueError(f"meshopt: vertex codec version {version}")
	tail = size + (0 if version == 0 else size // 4)
	tail_pad = max(tail, 32 if version == 0 else 24)
	if len(data) - 1 < tail_pad:
		raise ValueError("meshopt: buffer too small")
	last = bytearray(data[len(data) - tail:len(data) - tail + size])
	channels = data[len(data) - tail + size:len(data)] if version else b""
	out = np.zeros((count, size), np.uint8)
	pos = 1
	bsize = _block_size(size)
	v0 = 0
	while v0 < count:
		n = min(bsize, count - v0)
		aligned = (n + 15) & ~15
		control_size = 0 if version == 0 else size // 4
		control = data[pos:pos + control_size]
		pos += control_size
		block = np.zeros((n, size), np.uint8)
		for k in range(0, size, 4):
			ctrl_byte = 0 if version == 0 else control[k // 4]
			lanes = []
			for j in range(4):
				ctrl = (ctrl_byte >> (j * 2)) & 3
				if ctrl == 3:
					lanes.append(list(data[pos:pos + n]))
					pos += n
				elif ctrl == 2:
					lanes.append([0] * n)
				else:
					table = _BITS_V0 if version == 0 else _BITS_V1[ctrl:]
					vals, pos = _decode_bytes(data, pos, aligned, table)
					lanes.append(vals[:n])
			ch = 0 if version == 0 else channels[k // 4]
			kind = ch & 3
			if kind == 0:	# byte deltas, zigzag
				for j in range(4):
					p = last[k + j]
					col = block[:, k + j]
					for i, v in enumerate(lanes[j]):
						p = (((-(v & 1)) ^ (v >> 1)) + p) & 0xFF
						col[i] = p
			elif kind == 1:	# 16-bit deltas, zigzag
				for j in (0, 2):
					p = last[k + j] | (last[k + j + 1] << 8)
					lo, hi = lanes[j], lanes[j + 1]
					for i in range(n):
						v = lo[i] | (hi[i] << 8)
						p = (((-(v & 1)) ^ (v >> 1)) + p) & 0xFFFF
						block[i, k + j] = p & 0xFF
						block[i, k + j + 1] = p >> 8
			elif kind == 2:	# 32-bit xor deltas with a rotate
				rot = (32 - (ch >> 4)) & 31
				p = struct.unpack_from("<I", last, k)[0]
				for i in range(n):
					v = lanes[0][i] | (lanes[1][i] << 8) | (lanes[2][i] << 16) | (lanes[3][i] << 24)
					v = ((v << rot) | (v >> (32 - rot))) & 0xFFFFFFFF if rot else v
					p = v ^ p
					block[i, k:k + 4] = (p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF, p >> 24)
			else:
				raise ValueError("meshopt: bad channel mode")
		out[v0:v0 + n] = block
		last[:] = block[n - 1].tobytes()
		v0 += n
	if len(data) - pos != tail_pad:
		raise ValueError("meshopt: vertex data size mismatch")
	return out.tobytes()


# -- meshoptimizer index codec -------------------------------------------------


def _vbyte(data: bytes, pos: int) -> tuple[int, int]:
	lead = data[pos]
	pos += 1
	if lead < 128:
		return lead, pos
	result = lead & 127
	shift = 7
	for _ in range(4):
		g = data[pos]
		pos += 1
		result |= (g & 127) << shift
		shift += 7
		if g < 128:
			break
	return result, pos


def meshopt_index(count: int, size: int, data: bytes) -> bytes:
	"""meshopt_decodeIndexBuffer (versions 0 and 1)."""
	if count % 3:
		raise ValueError("meshopt: index count must be a multiple of 3")
	if size not in (2, 4):
		raise ValueError("meshopt: index size must be 2 or 4")
	if not data or data[0] & 0xF0 != 0xE0:
		raise ValueError("meshopt: not an index buffer")
	version = data[0] & 0x0F
	if version > 1:
		raise ValueError(f"meshopt: index codec version {version}")
	ntri = count // 3
	code = data[1:1 + ntri]
	body = data[1 + ntri:]
	safe_end = len(body) - 16
	aux_table = body[safe_end:]
	efifo = [(0, 0)] * 16
	vfifo = [0] * 16
	eo = vo = 0
	nxt = last = 0
	fecmax = 13 if version >= 1 else 15
	pos = 0
	out = np.zeros(count, np.uint32)
	o = 0
	M = 0xFFFFFFFF

	def decode_index(last: int, pos: int) -> tuple[int, int]:
		v, pos = _vbyte(body, pos)
		d = ((v >> 1) ^ -(v & 1)) & M
		return (last + d) & M, pos

	for codetri in code:
		if codetri < 0xF0:
			fe = codetri >> 4
			a, b = efifo[(eo - 1 - fe) & 15]
			fec = codetri & 15
			if fec < fecmax:
				c = nxt if fec == 0 else vfifo[(vo - 1 - fec) & 15]
				vfifo[vo] = c
				if fec == 0:
					nxt += 1
					vo = (vo + 1) & 15
			else:
				if fec != 15:
					c = (last + fec * 2 - 27) & M
				else:
					c, pos = decode_index(last, pos)
				last = c
				vfifo[vo] = c
				vo = (vo + 1) & 15
			efifo[eo] = (c, b)
			eo = (eo + 1) & 15
			efifo[eo] = (a, c)
			eo = (eo + 1) & 15
			out[o:o + 3] = (a, b, c)
		elif codetri < 0xFE:
			aux = aux_table[codetri & 15]
			feb, fec = aux >> 4, aux & 15
			a = nxt
			nxt += 1
			b = nxt if feb == 0 else vfifo[(vo - feb) & 15]
			if feb == 0:
				nxt += 1
			c = nxt if fec == 0 else vfifo[(vo - fec) & 15]
			if fec == 0:
				nxt += 1
			out[o:o + 3] = (a, b, c)
			vfifo[vo] = a
			vo = (vo + 1) & 15
			vfifo[vo] = b
			vo = (vo + (1 if feb == 0 else 0)) & 15
			vfifo[vo] = c
			vo = (vo + (1 if fec == 0 else 0)) & 15
			efifo[eo] = (b, a)
			eo = (eo + 1) & 15
			efifo[eo] = (c, b)
			eo = (eo + 1) & 15
			efifo[eo] = (a, c)
			eo = (eo + 1) & 15
		else:
			aux = body[pos]
			pos += 1
			fea = 0 if codetri == 0xFE else 15
			feb, fec = aux >> 4, aux & 15
			if aux == 0:
				nxt = 0
			if fea == 0:
				a = nxt
				nxt += 1
			else:
				a = 0
			if feb == 0:
				b = nxt
				nxt += 1
			else:
				b = vfifo[(vo - feb) & 15]
			if fec == 0:
				c = nxt
				nxt += 1
			else:
				c = vfifo[(vo - fec) & 15]
			if fea == 15:
				a, pos = decode_index(last, pos)
				last = a
			if feb == 15:
				b, pos = decode_index(last, pos)
				last = b
			if fec == 15:
				c, pos = decode_index(last, pos)
				last = c
			out[o:o + 3] = (a, b, c)
			vfifo[vo] = a
			vo = (vo + 1) & 15
			vfifo[vo] = b
			vo = (vo + (1 if feb in (0, 15) else 0)) & 15
			vfifo[vo] = c
			vo = (vo + (1 if fec in (0, 15) else 0)) & 15
			efifo[eo] = (b, a)
			eo = (eo + 1) & 15
			efifo[eo] = (c, b)
			eo = (eo + 1) & 15
			efifo[eo] = (a, c)
			eo = (eo + 1) & 15
		o += 3
	if pos != safe_end:
		raise ValueError("meshopt: index data malformed")
	return out.astype("<u2" if size == 2 else "<u4").tobytes()
