"""Tiny synthetic Source, Quake and Quake II files, so the native parsers are tested without game data."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np


# -- Source -------------------------------------------------------------------------


def vtf(rgba: "np.ndarray", fmt: str = "bgra", version: tuple[int, int] = (7, 2)) -> bytes:
	"""A VTF with a full mip chain. fmt: 'bgra' (BGRA8888) or 'dxt1' (blocks of each 4x4's first texel)."""
	h, w = rgba.shape[:2]
	code = {"bgra": 12, "dxt1": 13}[fmt]
	levels = []
	img = rgba
	while True:
		levels.append(img)
		if img.shape[0] == 1 and img.shape[1] == 1:
			break
		img = img[::2, ::2] if min(img.shape[:2]) > 1 else img[: max(1, img.shape[0] // 2), : max(1, img.shape[1] // 2)]

	def enc(a: "np.ndarray") -> bytes:
		if fmt == "bgra":
			return a[..., [2, 1, 0, 3]].tobytes()
		lh, lw = a.shape[:2]
		out = bytearray()
		for by in range(0, max(lh, 1), 4):
			for bx in range(0, max(lw, 1), 4):
				r, g, b = (int(x) for x in a[by, bx, :3])
				c = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
				out += struct.pack("<HHI", c, c, 0)	# c0 == c1: every texel is colour 0
		return bytes(out)

	body = b"".join(enc(lv) for lv in reversed(levels))	# smallest mip first
	if version >= (7, 3):
		header_size = 80 + 8
	else:
		header_size = 80
	head = bytearray(header_size)
	head[0:4] = b"VTF\0"
	struct.pack_into("<III", head, 4, version[0], version[1], header_size)
	struct.pack_into("<HHIHH", head, 16, w, h, 0, 1, 0)
	struct.pack_into("<i", head, 52, code)
	head[56] = len(levels)
	struct.pack_into("<i", head, 57, -1)	# no low-res thumbnail
	struct.pack_into("<H", head, 63, 1)
	if version >= (7, 3):
		struct.pack_into("<I", head, 68, 1)
		struct.pack_into("<3sBI", head, 80, b"\x30\0\0", 0, header_size)
	return bytes(head) + body


def _strings(table: dict[str, int], base: bytearray, s: str) -> int:
	if s not in table:
		table[s] = len(base)
		base += s.encode() + b"\0"
	return table[s]


def source_model(directory: Path, name: str = "arm") -> Path:
	"""models/<name>.mdl (+ .vvd, .dx90.vtx) and materials/models/test/crate.{vmt,vtf} under `directory`.

	Two bones: root at the origin and 'elbow' 10 units up. A flat strip of six
	vertices: z=0 on the root, z=15 shared 50/50, z=20 on the elbow. One
	sequence 'bend' turns the elbow 90° about X over two frames.
	"""
	models = directory / "models"
	mats = directory / "materials/models/test"
	models.mkdir(parents=True, exist_ok=True)
	mats.mkdir(parents=True, exist_ok=True)
	tex = np.zeros((16, 16, 4), np.uint8)
	tex[..., 0] = 200
	tex[..., 3] = 255
	(mats / "crate.vtf").write_bytes(vtf(tex))
	(mats / "crate.vmt").write_text('"VertexLitGeneric"\n{\n\t"$basetexture" "models/test/crate"\n}\n')

	# geometry: (pos, bone ids, weights)
	verts = [
		((-1, 0, 0), (0, 0), (1.0, 0.0)), ((1, 0, 0), (0, 0), (1.0, 0.0)),
		((-1, 0, 15), (0, 1), (0.5, 0.5)), ((1, 0, 15), (0, 1), (0.5, 0.5)),
		((-1, 0, 20), (1, 1), (1.0, 0.0)), ((1, 0, 20), (1, 1), (1.0, 0.0)),
	]
	# Faces toward -Y; Source winds fronts clockwise as seen from the front.
	tris = [(0, 2, 1), (1, 2, 3), (2, 4, 3), (3, 4, 5)]

	vvd = bytearray(64)
	vvd[0:4] = b"IDSV"
	struct.pack_into("<iiii", vvd, 4, 4, 0, 1, len(verts))
	struct.pack_into("<iii", vvd, 48, 0, 64, 64)
	for i, (p, bones, w) in enumerate(verts):
		vvd += struct.pack("<3f3bB3f3f2f", w[0], w[1], 0.0, bones[0], bones[1], 0, 2 if w[1] else 1,
			*p, 0.0, -1.0, 0.0, (p[0] + 1) / 2, 1 - p[2] / 20)
	(models / f"{name}.vvd").write_bytes(bytes(vvd))

	# VTX: one bodypart/model/LOD/mesh/strip group/strip (triangle list)
	vtx = bytearray(36)
	struct.pack_into("<iiHHiiiiii", vtx, 0, 7, 24, 53, 9, 3, 0, 1, 0, 1, 36)
	bp = len(vtx)
	vtx += struct.pack("<ii", 1, 8)
	mo = len(vtx)
	vtx += struct.pack("<ii", 1, 8)
	lod = len(vtx)
	vtx += struct.pack("<iif", 1, 12, 0.0)
	mesh = len(vtx)
	vtx += struct.pack("<iiB", 1, 9, 0)
	sg = len(vtx)
	vtx += bytes(25)
	vo = len(vtx)
	for i in range(len(verts)):
		vtx += struct.pack("<3BBH3b", 0, 1, 2, 1, i, 0, 0, 0)
	io = len(vtx)
	idx = [i for t in tris for i in t]
	vtx += struct.pack(f"<{len(idx)}H", *idx)
	so = len(vtx)
	vtx += struct.pack("<iiiihBii", len(idx), 0, len(verts), 0, 1, 1, 0, 0)
	struct.pack_into("<6iB", vtx, sg, len(verts), vo - sg, len(idx), io - sg, 1, so - sg, 0)
	assert bp and mo and lod and mesh
	(models / f"{name}.dx90.vtx").write_bytes(bytes(vtx))

	# MDL v48
	d = bytearray(408)
	strings: dict[str, int] = {}
	names = bytearray()
	d[0:4] = b"IDST"
	struct.pack_into("<i", d, 4, 48)
	d[12:76] = f"test/{name}.mdl".encode().ljust(64, b"\0")

	def place(blob: bytes) -> int:
		off = len(d)
		d.extend(blob)
		return off

	bone_off = len(d)
	bone_pos = [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]
	world_t = [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]
	d.extend(bytes(216 * 2))
	bone_names = ["root", "elbow"]
	name_fix: list[tuple[int, str, int]] = []	# (field offset, string, struct base)
	for i in range(2):
		o = bone_off + i * 216
		name_fix.append((o, bone_names[i], o))
		struct.pack_into("<i", d, o + 4, -1 if i == 0 else 0)
		struct.pack_into("<3f4f3f3f3f", d, o + 32, *bone_pos[i], 0, 0, 0, 1, 0, 0, 0,
			1 / 512, 1 / 512, 1 / 512, 0.001, 0.001, 0.001)
		struct.pack_into("<12f", d, o + 96, 1, 0, 0, -world_t[i][0], 0, 1, 0, -world_t[i][1], 0, 0, 1, -world_t[i][2])
	tex_off = place(bytes(64))
	name_fix.append((tex_off, "crate", tex_off))
	cd_off = place(bytes(4))
	skin_off = place(struct.pack("<h", 0) + b"\0\0")
	body_off = place(bytes(16))
	name_fix.append((body_off, "body", body_off))
	model_off = place(bytes(148))
	mesh_off = place(bytes(116))
	struct.pack_into("<iii", d, body_off + 4, 1, 1, model_off - body_off)
	d[model_off:model_off + 64] = b"strip".ljust(64, b"\0")
	struct.pack_into("<iiii", d, model_off + 72, 1, mesh_off - model_off, len(verts), 0)
	struct.pack_into("<iiii", d, mesh_off, 0, model_off - mesh_off, len(verts), 0)

	# animation: elbow ANIMROT about X: frame 0 → 0, frame 1 → 1571 × 0.001 rad
	anim_off = place(bytes(100))
	name_fix.append((anim_off + 4, "@bend", anim_off))
	struct.pack_into("<fii", d, anim_off + 8, 10.0, 0, 2)
	chain = struct.pack("<BBh", 1, 0x08, 0) + struct.pack("<3h", 6, 0, 0) + bytes([2, 2]) + struct.pack("<2h", 0, 1571)
	chain_off = place(chain + b"\0\0")
	struct.pack_into("<ii", d, anim_off + 52, 0, chain_off - anim_off)
	seq_off = place(bytes(212))
	name_fix.append((seq_off + 4, "bend", seq_off))
	idx_off = place(struct.pack("<h", 0) + b"\0\0")
	struct.pack_into("<i", d, seq_off + 12, 1)	# looping
	struct.pack_into("<ii", d, seq_off + 56, 1, idx_off - seq_off)

	strings_base = len(d)
	for field, text, base in name_fix:
		struct.pack_into("<i", d, field, strings_base + _strings(strings, names, text) - base)
	cd_str = strings_base + _strings(strings, names, "models\\test\\")
	d.extend(names)
	struct.pack_into("<i", d, cd_off, cd_str)
	struct.pack_into("<21i", d, 156, 2, bone_off, 0, 0, 0, 0, 1, anim_off, 1, seq_off, 0, 0, 1, tex_off, 1, cd_off,
		1, 1, skin_off, 1, body_off)
	struct.pack_into("<i", d, 76, len(d))
	path = models / f"{name}.mdl"
	path.write_bytes(bytes(d))
	return path


# -- Quake ---------------------------------------------------------------------------


def quake_palette() -> bytes:
	pal = np.zeros((256, 3), np.uint8)
	pal[:, 0] = np.arange(256)
	pal[:, 1] = 255 - np.arange(256)
	pal[:, 2] = 77
	return pal.tobytes()


def quake_mdl(frames: int = 3) -> bytes:
	"""A tetrahedron alias model: one 8x8 skin (index 200), clockwise fronts, `frames` frames."""
	sw = sh = 8
	pts = [(0, 0, 0), (16, 0, 0), (0, 16, 0), (0, 0, 16)]
	# clockwise as seen from outside (Quake's front faces)
	tris = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
	head = b"IDPO" + struct.pack("<i", 6) + struct.pack("<3f", 1, 1, 1) + struct.pack("<3f", 0, 0, 0)
	head += struct.pack("<f", 16) + struct.pack("<3f", 0, 0, 0)
	head += struct.pack("<8i", 1, sw, sh, len(pts), len(tris), frames, 0, 0) + struct.pack("<f", 1.0)
	out = bytearray(head)
	out += struct.pack("<i", 0) + bytes([200]) * (sw * sh)
	for i in range(len(pts)):
		out += struct.pack("<3i", 0, i * 2, i * 2)
	for a, b, c in tris:
		out += struct.pack("<4i", 1, a, c, b)	# reversed: (a, c, b) is clockwise from outside
	for f in range(frames):
		out += struct.pack("<i", 0) + bytes(4) + bytes(4) + f"frame{f}".encode().ljust(16, b"\0")
		for x, y, z in pts:
			out += bytes([x + f, y, z, 0])
	return bytes(out)


def quake_miptex(name: str, w: int, h: int, index: int = 10) -> bytes:
	pix = bytes([index]) * (w * h)
	mips = [pix] + [bytes([index]) * ((w >> i) * (h >> i)) for i in (1, 2, 3)]
	offs, cur = [], 40
	for m in mips:
		offs.append(cur)
		cur += len(m)
	return struct.pack("<16sII", name.encode()[:15], w, h) + struct.pack("<4I", *offs) + b"".join(mips)


def bsp29(textures: list[bytes]) -> bytes:
	from .fixtures import bsp30

	data = bytearray(bsp30('{\n"classname" "worldspawn"\n}\n', textures))
	struct.pack_into("<i", data, 0, 29)
	return bytes(data)


def wad2(textures: dict[str, bytes]) -> bytes:
	body = bytearray(b"WAD2" + struct.pack("<ii", 0, 0))
	entries = []
	for name, data in textures.items():
		entries.append((len(body), len(data), name))
		body += data
	table = len(body)
	for pos, size, name in entries:
		body += struct.pack("<iiibbh16s", pos, size, size, 0x44, 0, 0, name.encode()[:15])
	struct.pack_into("<ii", body, 4, len(entries), table)
	return bytes(body)


def pak(files: dict[str, bytes]) -> bytes:
	body = bytearray(b"PACK" + bytes(8))
	entries = []
	for name, data in files.items():
		entries.append((name, len(body), len(data)))
		body += data
	dirofs = len(body)
	for name, pos, size in entries:
		body += struct.pack("<56sii", name.encode(), pos, size)
	struct.pack_into("<ii", body, 4, dirofs, len(entries) * 64)
	return bytes(body)


def vpk(files: dict[str, bytes]) -> bytes:
	"""A v1 VPK directory with every file's data inline after the tree (archive 0x7fff)."""
	by_ext: dict[str, dict[str, list[tuple[str, bytes]]]] = {}
	for full, data in files.items():
		folder, _, fname = full.rpartition("/")
		stem, _, ext = fname.rpartition(".")
		by_ext.setdefault(ext, {}).setdefault(folder or " ", []).append((stem, data))
	tree = bytearray()
	blob = bytearray()
	for ext, folders in by_ext.items():
		tree += ext.encode() + b"\0"
		for folder, items in folders.items():
			tree += folder.encode() + b"\0"
			for stem, data in items:
				tree += stem.encode() + b"\0"
				tree += struct.pack("<IHHIIH", 0, 0, 0x7FFF, len(blob), len(data), 0xFFFF)
				blob += data
			tree += b"\0"
		tree += b"\0"
	tree += b"\0"
	return struct.pack("<III", 0x55AA1234, 1, len(tree)) + bytes(tree) + bytes(blob)


def quake_sprite(w: int = 16, h: int = 16) -> bytes:
	pix = np.full((h, w), 255, np.uint8)
	pix[h // 2, :] = 40
	head = b"IDSP" + struct.pack("<iifiiifi", 1, 2, math.hypot(w / 2, h / 2), w, h, 1, 0.0, 0)
	return head + struct.pack("<iiiii", 0, -w // 2, h // 2, w, h) + pix.tobytes()


def pcx8(idx: "np.ndarray", palette: bytes) -> bytes:
	from io import BytesIO

	from PIL import Image

	img = Image.fromarray(idx.astype(np.uint8), "P")
	img.putpalette(palette)
	buf = BytesIO()
	img.save(buf, "PCX")
	return buf.getvalue()


def sp2(frames: list[tuple[int, int, str]]) -> bytes:
	out = b"IDS2" + struct.pack("<ii", 2, len(frames))
	for w, h, name in frames:
		out += struct.pack("<iiii64s", w, h, w // 2, h // 2, name.encode())
	return out


def wal(name: str, w: int, h: int, index: int = 30) -> bytes:
	head = struct.pack("<32sII", name.encode(), w, h)
	offs = [100, 100 + w * h, 100 + w * h * 5 // 4, 100 + w * h * 21 // 16]
	head += struct.pack("<4I", *offs) + bytes(32) + struct.pack("<3i", 0, 0, 0)
	pix = np.full((h, w), index, np.uint8)
	pix[0, 0] = 255
	return head + pix.tobytes() + bytes(w * h // 4 + w * h // 16 + w * h // 64)


def wav_pcm(path: Path, rate: int = 22050, seconds: float = 0.5, width: int = 1) -> None:
	import wave

	n = int(rate * seconds)
	t = np.arange(n) / rate
	x = np.sin(2 * math.pi * 440 * t)
	with wave.open(str(path), "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(width)
		w.setframerate(rate)
		if width == 1:
			w.writeframes(((x * 100) + 128).astype(np.uint8).tobytes())
		else:
			w.writeframes((x * 20000).astype("<i2").tobytes())
