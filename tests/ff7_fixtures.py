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
