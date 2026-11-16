"""Builders for tiny synthetic GoldSrc files, so tests need no game data."""

from __future__ import annotations

import struct

import numpy as np


def palette(seed: int = 0) -> bytes:
	rng = np.random.default_rng(seed)
	pal = rng.integers(0, 256, (256, 3), dtype=np.uint8)
	pal[255] = (0, 0, 255)
	return pal.tobytes()


def miptex(name: str, w: int, h: int, seed: int = 0, embed: bool = True) -> bytes:
	rng = np.random.default_rng(seed)
	head = struct.pack("<16sII", name.encode()[:15], w, h)
	if not embed:
		return head + struct.pack("<4I", 0, 0, 0, 0)
	pix = rng.integers(0, 255, w * h, dtype=np.uint8)
	if name.startswith("{"):
		pix[: w * h // 4] = 255
	mips = [pix.tobytes()] + [bytes((w >> i) * (h >> i)) for i in (1, 2, 3)]
	offs, cur = [], 40
	for m in mips:
		offs.append(cur)
		cur += len(m)
	return head + struct.pack("<4I", *offs) + b"".join(mips) + struct.pack("<H", 256) + palette(seed) + b"\0\0"


def wad3(textures: dict[str, bytes]) -> bytes:
	body = bytearray(b"WAD3" + struct.pack("<ii", 0, 0))
	entries = []
	for name, data in textures.items():
		entries.append((len(body), len(data), name))
		body += data
	table = len(body)
	for pos, size, name in entries:
		body += struct.pack("<iiibbh16s", pos, size, size, 0x43, 0, 0, name.encode()[:15])
	struct.pack_into("<ii", body, 4, len(entries), table)
	return bytes(body)


def bsp30(entities: str, textures: list[bytes]) -> bytes:
	lumps = [b""] * 15
	lumps[0] = entities.encode() + b"\0"
	tex = bytearray(struct.pack("<i", len(textures)))
	offs, cur = [], 4 + 4 * len(textures)
	for t in textures:
		offs.append(cur)
		cur += len(t)
	tex += struct.pack(f"<{len(offs)}i", *offs) + b"".join(textures)
	lumps[2] = bytes(tex)
	out = bytearray(struct.pack("<i", 30) + bytes(15 * 8))
	for i, data in enumerate(lumps):
		struct.pack_into("<ii", out, 4 + i * 8, len(out), len(data))
		out += data
		out += bytes(-len(out) & 3)
	return bytes(out)


def studio(textures: list[tuple[str, int, int, int]], mesh_bytes: int = 400) -> bytes:
	"""IDST v10 with `mesh_bytes` of fake geometry then embedded skins."""
	hdr = bytearray(244)
	hdr[0:4] = b"IDST"
	struct.pack_into("<i", hdr, 4, 10)
	hdr[8:72] = b"models/test.mdl".ljust(64, b"\0")
	body = bytearray(hdr) + bytes(range(256)) * (mesh_bytes // 256) + bytes(mesh_bytes % 256)
	texindex = len(body)
	body += bytes(80 * len(textures))
	skinindex = len(body)
	body += struct.pack(f"<{len(textures)}h", *range(len(textures)))
	for i, (name, flags, w, h) in enumerate(textures):
		index = len(body)
		body += np.random.default_rng(i).integers(0, 256, w * h, dtype=np.uint8).tobytes() + palette(i)
		off = texindex + i * 80
		body[off:off + 64] = name.encode().ljust(64, b"\0")
		struct.pack_into("<iiii", body, off + 64, flags, w, h, index)
	struct.pack_into("<i", body, 72, len(body))
	struct.pack_into("<i", body, 180, len(textures))
	struct.pack_into("<i", body, 184, texindex)
	struct.pack_into("<i", body, 188, texindex + 80 * len(textures))
	struct.pack_into("<i", body, 192, len(textures))
	struct.pack_into("<i", body, 196, 1)
	struct.pack_into("<i", body, 200, skinindex)
	return bytes(body)


def sprite(w: int = 32, h: int = 32, texformat: int = 3) -> bytes:
	pix = np.full((h, w), 255, np.uint8)
	pix[h // 2, :] = 7	# a one-texel horizontal stroke
	pix[:, w // 2] = 9
	out = b"IDSP" + struct.pack("<iii", 2, 2, texformat) + struct.pack("<f", 1.0)
	out += struct.pack("<ii", w, h) + struct.pack("<iii", 1, 0, 0) + struct.pack("<h", 256) + palette(3)
	out += struct.pack("<i", 0) + struct.pack("<iiii", -w // 2, h // 2, w, h) + pix.tobytes()
	return out


# Unit cube, outward counter-clockwise faces, Y-up; the -Z face is the "back".
_CUBE_FACES = {
	"px": [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)],
	"nx": [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
	"py": [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)],
	"ny": [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
	"pz": [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
	"nz": [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)],
}


def obj_cube(directory, name: str = "crate", texture: bool = True) -> "Path":
	"""crate.obj + .mtl: textured top/bottom/sides material and a plain red one."""
	from pathlib import Path

	from PIL import Image

	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)
	lines = [f"mtllib {name}.mtl"]
	vt = [(0, 0), (1, 0), (1, 1), (0, 1)]
	lines += [f"vt {u} {v}" for u, v in vt]
	n = 0
	for i, (face, quad) in enumerate(_CUBE_FACES.items()):
		lines += [f"v {x} {y} {z}" for x, y, z in quad]
		lines.append(f"usemtl {'red' if face == 'pz' else 'wood'}")
		lines.append("f " + " ".join(f"{n + k + 1}/{k + 1}" for k in range(4)))
		n += 4
	(directory / f"{name}.obj").write_text("\n".join(lines) + "\n")
	mtl = "newmtl red\nKd 1 0 0\n\nnewmtl wood\nKd 1 1 1\n"
	if texture:
		mtl += "map_Kd C:\\Users\\artist\\Desktop\\wood.png\n"	# absolute path from another machine
		y, x = np.mgrid[0:48, 0:40]
		rgb = np.stack(((x * 6) % 256, (y * 5) % 256, np.full_like(x, 60)), -1).astype(np.uint8)
		Image.fromarray(rgb).save(directory / "wood.png")
	(directory / f"{name}.mtl").write_text(mtl)
	return directory / f"{name}.obj"


def glb_quad(path, rgba: "np.ndarray | None" = None) -> None:
	"""Minimal glTF 2.0 binary: one textured quad with the PNG embedded in the BIN chunk."""
	import io
	import json

	from PIL import Image

	if rgba is None:
		rgba = np.zeros((16, 16, 4), np.uint8)
		rgba[...] = (0, 200, 0, 255)
		rgba[:8, :8, 3] = 0	# a cut-out corner → masked skin
	png = io.BytesIO()
	Image.fromarray(rgba).save(png, "PNG")
	png = png.getvalue()
	pos = np.array([[-1, 0, 0], [1, 0, 0], [1, 2, 0], [-1, 2, 0]], np.float32)
	uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
	idx = np.array([0, 1, 2, 0, 2, 3], np.uint16)
	blobs = [pos.tobytes(), uv.tobytes(), idx.tobytes() + b"\0\0", png]
	views, off = [], 0
	for b in blobs:
		views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b)})
		off += len(b) + (-len(b) % 4)
	binchunk = b"".join(b + bytes(-len(b) % 4) for b in blobs)
	views[2]["byteLength"] = 12
	doc = {
		"asset": {"version": "2.0"},
		"scenes": [{"nodes": [0]}], "scene": 0,
		"nodes": [{"mesh": 0}],
		"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
		"materials": [{"name": "leaf", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}, "alphaMode": "MASK"}],
		"textures": [{"source": 0}],
		"images": [{"bufferView": 3, "mimeType": "image/png"}],
		"buffers": [{"byteLength": len(binchunk)}],
		"bufferViews": views,
		"accessors": [
			{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, 0, 0], "max": [1, 2, 0]},
			{"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
			{"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"},
		],
	}
	js = json.dumps(doc).encode()
	js += b" " * (-len(js) % 4)
	body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(binchunk), b"BIN\0") + binchunk
	with open(path, "wb") as f:
		f.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
