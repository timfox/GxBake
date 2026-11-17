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


def glb_arm(path, angle_deg: float = 90.0, name: str = "bend") -> None:
	"""Skinned two-bone glTF: a 1x4x1 box along +Y; joint 'upper' at y=0, 'lower' at y=2.

	Vertices with y < 2 are weighted to upper, the rest to lower. One animation
	rotates 'lower' about +Z from 0 to angle_deg over one second (2 keys).
	"""
	import json
	import math

	pos, idx, joints, weights = [], [], [], []
	ys = (0.0, 1.0, 2.0, 3.0, 4.0)
	ring = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
	for y in ys:
		for x, z in ring:
			pos.append((x, y, z))
			joints.append((0 if y < 2 else 1, 0, 0, 0))
			weights.append((1.0, 0.0, 0.0, 0.0))
	for r in range(len(ys) - 1):
		for k in range(4):
			a, b = r * 4 + k, r * 4 + (k + 1) % 4
			c, d = a + 4, b + 4
			idx += [a, b, d, a, d, c]	# outward when viewed from outside (CCW)
	idx += [0, 2, 1, 0, 3, 2]	# bottom cap
	top = (len(ys) - 1) * 4
	idx += [top, top + 1, top + 2, top, top + 2, top + 3]
	pos = np.array(pos, np.float32)
	idx = np.array(idx, np.uint16)
	joints = np.array(joints, np.uint16)
	weights = np.array(weights, np.float32)
	# inverse bind matrices (column-major): translate by -joint position
	ibm = np.zeros((2, 16), np.float32)
	for i, jy in enumerate((0.0, 2.0)):
		m = np.eye(4, dtype=np.float32)
		m[1, 3] = -jy
		ibm[i] = m.T.reshape(-1)
	times = np.array([0.0, 1.0], np.float32)
	h = math.radians(angle_deg) / 2
	rots = np.array([[0, 0, 0, 1], [0, 0, math.sin(h), math.cos(h)]], np.float32)
	blobs = [pos.tobytes(), idx.tobytes(), joints.tobytes(), weights.tobytes(), ibm.tobytes(), times.tobytes(), rots.tobytes()]
	views, off, binchunk = [], 0, b""
	for b in blobs:
		pad = b + bytes(-len(b) % 4)
		views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b)})
		binchunk += pad
		off += len(pad)
	doc = {
		"asset": {"version": "2.0"},
		"scenes": [{"nodes": [0, 1]}], "scene": 0,
		"nodes": [
			{"name": "arm", "mesh": 0, "skin": 0},
			{"name": "upper", "children": [2]},
			{"name": "lower", "translation": [0, 2, 0]},
		],
		"skins": [{"joints": [1, 2], "inverseBindMatrices": 4, "skeleton": 1}],
		"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1, "material": 0}]}],
		"materials": [{"name": "skin", "pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.4, 0.9, 1]}}],
		"animations": [{"name": name, "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}],
			"samplers": [{"input": 5, "output": 6, "interpolation": "LINEAR"}]}],
		"buffers": [{"byteLength": len(binchunk)}],
		"bufferViews": views,
		"accessors": [
			{"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
				"min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
			{"bufferView": 1, "componentType": 5123, "count": len(idx), "type": "SCALAR"},
			{"bufferView": 2, "componentType": 5123, "count": len(pos), "type": "VEC4"},
			{"bufferView": 3, "componentType": 5126, "count": len(pos), "type": "VEC4"},
			{"bufferView": 4, "componentType": 5126, "count": 2, "type": "MAT4"},
			{"bufferView": 5, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
			{"bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC4"},
		],
	}
	js = json.dumps(doc).encode()
	js += b" " * (-len(js) % 4)
	body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(binchunk), b"BIN\0") + binchunk
	with open(path, "wb") as f:
		f.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


# -- ROSE Online (ZMS / ZMD / ZMO / ZSC / CHR) ------------------------------------


def _cstr(s: str) -> bytes:
	return s.encode() + b"\0"


def rose_zms(positions, tris, uvs=None, normals=None, joints=None, weights=None, bone_map=None) -> bytes:
	"""ZMS0008 mesh. positions in metres; joints index into bone_map (ZMD bone ids)."""
	pos = np.asarray(positions, np.float32)
	n = len(pos)
	fmt = 1 << 1
	if normals is not None:
		fmt |= 1 << 2
	if joints is not None:
		fmt |= (1 << 4) | (1 << 5)
	if uvs is not None:
		fmt |= 1 << 7
	bone_map = list(bone_map or [])
	out = bytearray(_cstr("ZMS0008") + struct.pack("<I", fmt))
	out += struct.pack("<6f", *pos.min(0), *pos.max(0))
	out += struct.pack("<H", len(bone_map)) + struct.pack(f"<{len(bone_map)}H", *bone_map)
	out += struct.pack("<H", n) + pos.tobytes()
	if normals is not None:
		out += np.asarray(normals, np.float32).tobytes()
	if joints is not None:
		for w, j in zip(np.asarray(weights, np.float32), np.asarray(joints, np.int64)):
			out += struct.pack("<4f4H", *w, *j)
	if uvs is not None:
		out += np.asarray(uvs, np.float32).tobytes()
	t = np.asarray(tris, np.uint16)
	out += struct.pack("<H", len(t)) + t.tobytes()
	out += struct.pack("<H", 1) + struct.pack("<H", len(t))	# material face counts
	out += struct.pack("<H", 0) + struct.pack("<H", 0)	# no strips, pool type
	return bytes(out)


def rose_zmd(bones, dummies=()) -> bytes:
	"""bones: (name, parent, pos_cm, quat_wxyz); bone 0 is its own parent."""
	out = bytearray(b"ZMD0003" + struct.pack("<I", len(bones)))
	for i, (name, parent, pos, q) in enumerate(bones):
		out += struct.pack("<I", i if parent < 0 else parent) + _cstr(name) + struct.pack("<3f4f", *pos, *q)
	out += struct.pack("<I", len(dummies))
	for name, parent, pos, q in dummies:
		out += _cstr(name) + struct.pack("<I", parent) + struct.pack("<3f4f", *pos, *q)
	return bytes(out)


def rose_zmo(fps: int, frames: int, channels) -> bytes:
	"""channels: (kind, bone, values[frames]) with kind 2 = position (cm), 4 = rotation (wxyz)."""
	out = bytearray(_cstr("ZMO0002") + struct.pack("<III", fps, frames, len(channels)))
	for kind, bone, _v in channels:
		out += struct.pack("<II", kind, bone)
	for f in range(frames):
		for _k, _b, v in channels:
			out += np.asarray(v[f], np.float32).tobytes()
	return bytes(out)


def rose_zsc(meshes, materials, objects) -> bytes:
	"""materials: (path, two_sided, alpha_test); objects: lists of part dicts
	(mesh, material, pos_cm?, quat_wxyz?, bone?, parent? (1-based))."""
	out = bytearray(struct.pack("<H", len(meshes)))
	for m in meshes:
		out += _cstr(m)
	out += struct.pack("<H", len(materials))
	for path, two_sided, atest in materials:
		out += _cstr(path) + struct.pack("<9H", 0, int(atest), int(two_sided), int(atest), 128, 1, 1, 0, 0)
		out += struct.pack("<f", 1.0) + struct.pack("<H3f", 0, 0, 0, 0)
	out += struct.pack("<H", 0)	# effects
	out += struct.pack("<H", len(objects))
	for parts in objects:
		out += struct.pack("<I2i", 100, 0, 0) + struct.pack("<H", len(parts))
		for p in parts:
			out += struct.pack("<HH", p["mesh"], p["material"])
			if "pos" in p:
				out += struct.pack("<BB3f", 1, 12, *p["pos"])
			if "quat" in p:
				out += struct.pack("<BB4f", 2, 16, *p["quat"])
			if "bone" in p:
				out += struct.pack("<BBH", 5, 2, p["bone"])
			if "dummy" in p:
				out += struct.pack("<BBH", 6, 2, p["dummy"])
			if "parent" in p:
				out += struct.pack("<BBH", 7, 2, p["parent"])
			out += b"\0"
		if parts:
			out += struct.pack("<H", 0) + struct.pack("<6f", 0, 0, 0, 0, 0, 0)
	return bytes(out)


def rose_chr(skeletons, motions, characters) -> bytes:
	"""characters: None (disabled) or (skeleton, name, objects, [(slot, motion index)])."""
	out = bytearray()
	for lst in (skeletons, motions, []):
		out += struct.pack("<H", len(lst)) + b"".join(_cstr(s) for s in lst)
	out += struct.pack("<H", len(characters))
	for c in characters:
		if c is None:
			out += b"\0"
			continue
		skel, name, objs, mots = c
		out += b"\1" + struct.pack("<H", skel) + _cstr(name)
		out += struct.pack("<H", len(objs)) + struct.pack(f"<{len(objs)}H", *objs)
		out += struct.pack("<H", len(mots)) + b"".join(struct.pack("<HH", *m) for m in mots)
		out += struct.pack("<H", 0)
	return bytes(out)


def box(center, half) -> tuple[np.ndarray, np.ndarray]:
	"""Axis-aligned box: (8 corners, 12 counter-clockwise outward triangles)."""
	c, h = np.asarray(center, float), np.asarray(half, float)
	corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h + c
	tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
		[2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
	return corners, tris
