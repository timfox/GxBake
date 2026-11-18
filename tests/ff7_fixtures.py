"""Synthetic Final Fantasy VII PC files for the tests (no game data in the repo)."""

from __future__ import annotations

import struct

import numpy as np


def p_model(verts, faces, normals=None, colors=None, groups=None, texcoords=None) -> bytes:
	"""groups: (poly_start, poly_count, vert_start, tex_start, textured, texture); default one untextured group.
	normals are per vertex; polygons index them with the vertex index."""
	verts = np.asarray(verts, np.float32)
	nv, npoly = len(verts), len(faces)
	normals = np.zeros((0, 3), np.float32) if normals is None else np.asarray(normals, np.float32)
	texcoords = np.zeros((0, 2), np.float32) if texcoords is None else np.asarray(texcoords, np.float32)
	colors = colors if colors is not None else [(128, 128, 128, 255)] * nv
	groups = groups or [(0, npoly, 0, 0, 0, 0)]
	head = struct.pack("<16i", 1, 1, 1, nv, len(normals), 0, len(texcoords), nv, 0, npoly, 0, 0, 0, len(groups), 0, 0)
	head += b"\0" * 64
	out = head + verts.tobytes() + normals.tobytes() + texcoords.tobytes()
	out += b"".join(bytes([b, g, r, a]) for r, g, b, a in colors)
	out += b"".join(bytes([100, 100, 100, 255]) for _ in faces)
	for f in faces:
		n = f if len(normals) else (0, 0, 0)
		out += struct.pack("<12H", 0, *f, *n, 0, 0, 0, 0, 0)
	for ps, pc, vs, ts, textured, tex in groups:
		out += struct.pack("<14I", 1, ps, pc, vs, nv, 0, 0, 0, 0, 0, 0, ts, textured, tex)
	return out


def tex(width: int, height: int, palette: list[tuple[int, int, int]], pixels, color_key: bool = True) -> bytes:
	h = bytearray(0xEC)

	def put(off, v):
		struct.pack_into("<I", h, off, v)

	put(0x00, 1)
	put(0x08, 1 if color_key else 0)
	put(0x30, 1)
	put(0x34, len(palette))
	put(0x38, 8)
	put(0x3C, width)
	put(0x40, height)
	put(0x4C, 1)
	put(0x58, len(palette))
	put(0x5C, len(palette))
	put(0x64, 8)
	put(0x68, 1)
	pal = b"".join(bytes([b, g, r, 255]) for r, g, b in palette)
	return bytes(h) + pal + np.asarray(pixels, np.uint8).tobytes()


def hrc(name: str, bones: list[tuple[str, str, float, list[str]]]) -> str:
	lines = [":HEADER_BLOCK 2", f":SKELETON {name}", f":BONES {len(bones)}", ""]
	for bname, parent, length, res in bones:
		lines += [bname, parent, f"{length}", " ".join([str(len(res))] + res), ""]
	return "\n".join(lines)


def rsd(ply: str, textures: list[str] = ()) -> str:
	lines = ["@RSD940102", f"PLY={ply}.PLY", f"MAT={ply}.MAT", f"GRP={ply}.GRP", f"NTEX={len(textures)}"]
	lines += [f"TEX[{i}]={t}.TIM" for i, t in enumerate(textures)]
	return "\n".join(lines) + "\n"


def anim(frames: list[tuple[tuple, tuple, list[tuple]]]) -> bytes:
	"""frames: (root rotation degrees, root translation, [bone rotation degrees...])."""
	bones = len(frames[0][2]) if frames else 0
	out = struct.pack("<iii", 1, len(frames), bones) + bytes([1, 0, 2, 0]) + b"\0" * 20
	for rr, rt, rots in frames:
		out += struct.pack("<6f", *rr, *rt)
		for r in rots:
			out += struct.pack("<3f", *r)
	return out


def lgp(files: dict[str, bytes]) -> bytes:
	names = list(files)
	head = b"\0\0SQUARESOFT" + struct.pack("<I", len(names))
	toc_size = 27 * len(names)
	data_at = 16 + toc_size + 3600 + 2
	toc, body = b"", b""
	for n in names:
		toc += struct.pack("<20sIBH", n.encode(), data_at + len(body), 14, 0)
		body += struct.pack("<20sI", n.encode(), len(files[n])) + files[n]
	return head + toc + b"\0" * 3602 + body + b"FINAL FANTASY 7"


def cube(s: float = 1.0, center=(0.0, 0.0, 0.0), mixed: bool = True):
	"""8 corners, per-vertex outward normals, 12 triangles; half of them wound the other way when mixed."""
	c = np.array(center, float)
	verts = [c + s * np.array([dx, dy, dz]) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
	normals = [np.sign(v - c) / np.sqrt(3) for v in verts]
	quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
	faces = []
	for i, (a, b, cc, d) in enumerate(quads):
		t1, t2 = (a, b, cc), (a, cc, d)
		if mixed and i % 2:
			t1, t2 = (a, cc, b), (a, d, cc)
		faces += [t1, t2]
	return verts, normals, faces


def battle_skeleton(bones: list[tuple[int, float, bool]], textures: int = 0, body_anims: int = 0, weapon_anims: int = 0) -> bytes:
	"""bones: (parent, length, has part)."""
	h = bytearray(52)
	struct.pack_into("<i", h, 12, len(bones))
	struct.pack_into("<ii", h, 24, textures, body_anims)
	struct.pack_into("<i", h, 40, weapon_anims)
	return bytes(h) + b"".join(struct.pack("<ifi", p, ln, int(part)) for p, ln, part in bones)


class _BitWriter:
	def __init__(self):
		self.bits: list[int] = []

	def put(self, value: int, n: int) -> None:
		value &= (1 << n) - 1
		self.bits += [(value >> (n - 1 - i)) & 1 for i in range(n)]

	def data(self) -> bytes:
		b = self.bits + [0] * (-len(self.bits) % 8)
		return bytes(int("".join(map(str, b[i:i + 8])), 2) for i in range(0, len(b), 8))


def _put_delta(w: _BitWriter, delta: int, key: int) -> None:
	v = delta >> key
	assert v << key == delta, "deltas must be multiples of 1 << key"
	if v == 0:
		w.put(0, 1)
		return
	w.put(1, 1)
	if v == -1:
		w.put(0, 3)
		return
	for n in range(1, 7):
		half = 1 << (n - 1)
		if half <= v <= 2 * half - 1:
			w.put(n, 3)
			w.put(v - half, n)
			return
		if -2 * half <= v <= -half - 1:
			w.put(n, 3)
			w.put(v + half, n)
			return
	w.put(7, 3)
	w.put(v, 12 - key)


def da_animation(frames: list[tuple[tuple[int, int, int], list[tuple[int, int, int]]]], key: int = 0,
		missing_counter: bool = False) -> bytes:
	"""frames: (translation, raw rotations per slot, 4096 = a full turn). One pack block."""
	w = _BitWriter()
	t0, r0 = frames[0]
	for c in t0:
		w.put(c, 16)
	for r in r0:
		for c in r:
			assert (c >> key) << key == c
			w.put(c >> key, 12 - key)
	prev_t, prev_r = t0, r0
	for t, r in frames[1:]:
		for c in range(3):
			d = t[c] - prev_t[c]
			if -64 <= d < 64:
				w.put(0, 1)
				w.put(d, 7)
			else:
				w.put(1, 1)
				w.put(d, 16)
		for s in range(len(r)):
			for c in range(3):
				_put_delta(w, r[s][c] - prev_r[s][c], key)
		prev_t, prev_r = t, r
	stream = w.data()
	slots = len(frames[0][1])
	if missing_counter:	# such blocks still count a 5-byte mini header
		block = struct.pack("<HB", len(stream), key) + stream + b"\0\0"
	else:
		block = struct.pack("<hHB", len(frames), len(stream), key) + stream
	return struct.pack("<III", slots, len(frames), len(block)) + block


def da_pack(blocks: list[bytes]) -> bytes:
	return struct.pack("<I", len(blocks)) + b"".join(blocks)


def da_empty() -> bytes:
	return struct.pack("<III", 0, 0, 4) + b"\0" * 4
