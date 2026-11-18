"""Synthetic Final Fantasy XI files for the tests (no game data in the repo)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def chunk(name: str, ctype: int, body: bytes, last: bool = False) -> bytes:
	body = body + b"\0" * (-len(body) % 16)
	units = (16 + len(body)) // 16
	return name.encode("latin-1")[:4].ljust(4, b"\0") + struct.pack("<I", ctype | (units << 7)) + b"\0" * 8 + body


def dat(*chunks: bytes) -> bytes:
	return b"".join(chunks)


# -- textures -------------------------------------------------------------------


def rgb565(r: int, g: int, b: int) -> int:
	return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _tex_header(flag: int, name: str, w: int, h: int, bits: int) -> bytes:
	return (bytes([flag]) + name.encode("latin-1").ljust(16)[:16]
		+ struct.pack("<IiiHHIIIIII", 40, w, h, 1, bits, 0, 0, 0, 0, 0, 0))	# 57 bytes


def dxt_texture(name: str, w: int, h: int, color=(255, 0, 0), alpha_nibble: int | None = None) -> bytes:
	"""Every block one solid colour; DXT3 when alpha_nibble is given."""
	c0 = rgb565(*color)
	block = struct.pack("<HHI", c0, 0, 0)	# c0 > c1: index 0 = c0
	fourcc = b"1TXD"
	if alpha_nibble is not None:
		block = bytes([alpha_nibble | (alpha_nibble << 4)]) * 8 + block
		fourcc = b"3TXD"
	n = max(1, w // 4) * max(1, h // 4)
	data = block * n
	return _tex_header(0xA1, name, w, h, 32) + fourcc + struct.pack("<II", len(data), n) + data


def palette_texture(name: str, w: int, h: int, colors: list[tuple[int, int, int, int]], pixels: "np.ndarray",
		at: int = 61) -> bytes:
	"""0xB1: colours as (r, g, b, a) with FFXI alpha 0..0x80; palette at byte `at` (61 or 64)."""
	pal = bytearray(1024)
	for i, (r, g, b, a) in enumerate(colors):
		pal[i * 4:i * 4 + 4] = bytes([b, g, r, a])
	for i in range(len(colors), 256):
		pal[i * 4 + 3] = 0x80
	head = _tex_header(0xB1, name, w, h, 8)
	head += b"\0" * (at - len(head))
	if at == 61:
		head = head[:57] + struct.pack("<I", 10)
	return head + bytes(pal) + np.asarray(pixels, np.uint8).tobytes()


# -- skeleton -------------------------------------------------------------------


def skeleton(bones: list[tuple[int, tuple, tuple]]) -> bytes:
	"""bones: (parent or 255, quaternion xyzw, translation)."""
	out = struct.pack("<Hh", 0, len(bones))
	for parent, q, t in bones:
		out += struct.pack("<BB7f", parent, 0, *q, *t)
	return out


# -- meshes ---------------------------------------------------------------------


def tri_list(faces: list[tuple[int, int, int]], uvs: list[tuple] | None = None) -> bytes:
	out = struct.pack("<HH", 0x0054, len(faces))
	for i, f in enumerate(faces):
		uv = uvs[i] if uvs else (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
		out += struct.pack("<3h6f", *f, *uv)
	return out


def tri_strip(indices: list[int]) -> bytes:
	out = struct.pack("<HH", 0x5453, len(indices) - 2)
	out += struct.pack("<3h6f", indices[0], indices[1], indices[2], 0, 0, 1, 0, 0, 1)
	for i in indices[3:]:
		out += struct.pack("<h2f", i, 0.5, 0.5)
	return out


def set_texture(name: str) -> bytes:
	return struct.pack("<H", 0x8000) + name.encode("latin-1").ljust(16)[:16]


def mesh(one: list[tuple], assign: list[tuple[int, int]], polys: bytes, two: list[tuple] = (),
		table: list[int] = (), flip: bool = False, cloth: bool = False, indirect: bool = False) -> bytes:
	"""one: (x, y, z, nx, ny, nz) (or x, y, z for cloth); two: (x1, y1, z1, x2, y2, z2, w1, w2, nx, ny, nz);
	assign: raw (low, high) words per vertex, see bone()."""
	verts = b""
	for v in one:
		verts += struct.pack("<3f" if cloth else "<6f", *v)
	for x1, y1, z1, x2, y2, z2, w1, w2, *n in two:
		verts += struct.pack("<8f", x1, x2, y1, y2, z1, z2, w1, w2)
		if not cloth:
			nx, ny, nz = n
			verts += struct.pack("<6f", nx, nx, ny, ny, nz, nz)
	tbl = struct.pack(f"<{len(table)}H", *table)
	weight = struct.pack("<HH", len(one), len(two))
	bones = b"".join(struct.pack("<HH", lo, hi) for lo, hi in assign)
	polys = polys + struct.pack("<H", 0xFFFF)
	sections = [tbl, weight, bones, verts, polys]
	offs, pos = [], 64
	for sec in sections:
		offs.append(pos // 2)
		pos += len(sec) + (len(sec) & 1)
	mtype = (1 if cloth else 0) | (0x80 if indirect else 0)
	head = struct.pack("<BBHHIHIHIHIHIH", 1, 0, mtype, 1 if flip else 0,
		offs[4], len(polys) // 2, offs[0], len(table), offs[1], 2, offs[2], len(bones) // 2, offs[3], len(verts) // 2)
	head = head.ljust(64, b"\0")
	return head + b"".join(sec + b"\0" * (len(sec) & 1) for sec in sections)


def bone(left: int, right: int | None = None, mirror: int = 0) -> int:
	"""One bone slot: left bone, mirror bone (for the flipped half) and the mirror axis (1 x, 2 y, 3 z)."""
	return (left & 0x7F) | (((left if right is None else right) & 0x7F) << 7) | ((mirror & 3) << 14)


def box(cx: float, cy: float, cz: float, s: float = 0.1) -> tuple[list[tuple], list[tuple[int, int, int]]]:
	"""An axis-aligned cube with outward normals, wound like D3D (clockwise from outside)."""
	corners = [(cx + dx * s, cy + dy * s, cz + dz * s) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
	verts = [(*c, *(np.sign(np.subtract(c, (cx, cy, cz))) / np.sqrt(3))) for c in corners]
	quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]	# outward CCW
	faces = []
	for a, b, c, d in quads:
		faces += [(a, c, b), (a, d, c)]	# reversed: clockwise
	return verts, faces


# -- motions --------------------------------------------------------------------


def motion(frames: int, speed: float, elements: list[dict]) -> bytes:
	"""elements: {"bone": i, "rot": (frames, 4) array or None, "trans": (frames, 3) or None, "skip": bool}."""
	n = len(elements)
	pool: list[float] = []
	base = 21 * n	# the element table itself is the start of the float pool
	descs = b""
	for e in elements:
		iq, q = [0, 0, 0, 0], [0.0, 0.0, 0.0, 1.0]
		it, t = [0, 0, 0], [0.0, 0.0, 0.0]
		if e.get("rot") is not None:
			r = np.asarray(e["rot"], np.float32)
			for c in range(4):
				iq[c] = base + len(pool)
				pool += list(r[:, c])
		if e.get("trans") is not None:
			tr = np.asarray(e["trans"], np.float32)
			for c in range(3):
				it[c] = base + len(pool)
				pool += list(tr[:, c])
		if e.get("skip"):
			iq[0] = -1
		descs += struct.pack("<i4i4f3i3f3i3f", e["bone"], *iq, *q, *it, *t, 0, 0, 0, 1.0, 1.0, 1.0)
	return struct.pack("<BBHHf", 1, 0, n, frames, speed) + descs + struct.pack(f"<{len(pool)}f", *pool)


# -- sounds ---------------------------------------------------------------------


def spw(blocks: list[bytes], rate: int = 22050, block_size: int = 16, channels: int = 1, loop: int = -1) -> bytes:
	"""A SeWave file around raw ADPCM blocks ((1 + block_size/2) bytes per channel per block)."""
	body = b"".join(blocks)
	head = b"SeWave\0\0" + struct.pack("<iIiiiiii", 0x30 + len(body), 0, 7, len(blocks) // channels, loop,
		rate, 0, 0) + bytes([0, 0, channels, block_size]) + struct.pack("<i", 0)
	assert len(head) == 0x30
	return head + body


def bgw(blocks: list[bytes], rate: int = 44100, block_size: int = 16, channels: int = 2) -> bytes:
	body = b"".join(blocks)
	head = b"BGMStream\0\0\0" + struct.pack("<Iiiiiiii", 0, 0x30 + len(body), 3, len(blocks), -1, rate, 0, 0) \
		+ bytes([0, 0, channels, block_size])
	assert len(head) == 0x30
	return head + body


def write(path: Path, data: bytes) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(data)
	return path
