"""Final Fantasy XI (PC) data → GoldSrc studio MDL, WAV and GCPCM.

Where things live in an FFXI install (the PlayOnline "FINAL FANTASY XI" folder):

	VTABLE.DAT / FTABLE.DAT, ROM/<dir>/<file>.DAT
		The base game. Expansions add ROM2..ROM9 with their own VTABLE2.DAT /
		FTABLE2.DAT inside the ROMn folder. A file id is looked up in every
		VTABLEn: a byte equal to n means "in ROMn", and the uint16 at
		FTABLEn[id] packs the path as dir = v >> 7, file = v & 0x7F.
	sound/win/se/**.spw, sound/win/music/data/**.bgw (and sound2..)
		Sound effects ("SeWave") and music ("BGMStream"): 4-bit ADPCM.

A model DAT is a chain of chunks. Each chunk has a 16-byte header: a 4-character
name, a uint32 whose low 7 bits are the type and whose next 19 bits are the
chunk length in 16-byte units (header included), and 8 unused bytes. Types read
here:

	0x20	texture: DXT1/DXT3 (flag 0xA1/0x81) or 256-colour palette (0xB1/0x91)
	0x29	skeleton: parent, quaternion and translation per bone
	0x2A	mesh: single- and two-bone vertices (bone-local), triangle lists and
		strips with per-corner UVs, texture changes, and a "flip" flag that
		draws a mirrored copy bound to the mirror bones (left/right halves)
	0x2B	motion: per-bone rotation/translation/scale keys at 30·speed fps

NPC and monster DATs carry their own skeleton and motions. Player equipment and
faces use the race skeleton from another DAT; give it with --ffxi-skeleton or in
a .ffximdl manifest:

	{"root": "C:/.../FINAL FANTASY XI",	# optional: lets parts be "id:<file id>"
	 "skeleton": "ROM/27/82.dat",
	 "parts": ["ROM/28/53.dat", "id:12345"],
	 "motions": ["ROM/32/13.dat"]}

Format references (read for layout, no code copied): POLUtils (Apache-2.0) for
the file tables, image headers and the ADPCM codec; Vanalytics' ffxi-dat reader
(MIT) for the chunk chain, skeleton, mesh and motion layouts.

Axes: FFXI is y-down, and characters face +X. GoldSrc is Z-up, X forward, so
(x, y, z) → (x, z, -y) · ffxi_scale, then a turn of ffxi_yaw degrees about Z.
The facing and the unit size are taken from viewers, not checked on real data.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as _m
from . import skeleton as _sk

MANIFEST_EXTENSION = ".ffximdl"
AUDIO_EXTENSIONS = {".spw", ".bgw"}
DEFAULT_SCALE = 39.37	# GoldSrc units per FFXI unit (assumed metres)

T_TEXTURE, T_SKELETON, T_MESH, T_MOTION = 0x20, 0x29, 0x2A, 0x2B
CHUNK_NAMES = {
	0x00: "end", 0x01: "directory", 0x1C: "zone (MZB)", 0x20: "texture", 0x29: "skeleton", 0x2A: "mesh",
	0x2B: "motion", 0x2E: "zone mesh (MMB)",
}
_HEAD = 16
_MAX_CHUNKS = 4096


# -- chunk chain ------------------------------------------------------------------


@dataclass
class Chunk:
	name: str
	type: int
	offset: int	# of the 16-byte header
	size: int	# header included

	@property
	def start(self) -> int:
		return self.offset + _HEAD

	@property
	def end(self) -> int:
		return self.offset + self.size

	@property
	def kind(self) -> str:
		return CHUNK_NAMES.get(self.type, f"type {self.type:#04x}")


def _chunk_header(head: bytes, offset: int) -> tuple[str, int, int]:
	raw, packed = struct.unpack_from("<4sI", head, 0)
	name = raw.split(b"\0")[0].decode("latin-1")
	return name, packed & 0x7F, ((packed >> 7) & 0x7FFFF) * 16


def _plausible_name(raw: bytes) -> bool:
	return all(32 <= c < 127 or c == 0 for c in raw)


def read_chunks(data: bytes, name: str = "dat") -> list[Chunk]:
	"""The chunk chain of a DAT. Raises ValueError when the bytes are not one."""
	out: list[Chunk] = []
	pos = 0
	while pos + _HEAD <= len(data):
		cname, ctype, size = _chunk_header(data[pos:pos + 8], pos)
		if not _plausible_name(data[pos:pos + 4]):
			raise ValueError(f"{name}: no chunk at byte {pos}")
		if size == 0:	# the last chunk may run to the end of the file
			size = len(data) - pos
		if size < _HEAD or pos + size > len(data):
			raise ValueError(f"{name}: chunk {cname!r} at byte {pos} runs past the end")
		out.append(Chunk(cname, ctype, pos, size))
		pos += size
		if len(out) > _MAX_CHUNKS:
			raise ValueError(f"{name}: too many chunks")
	if not out or len(data) - pos >= _HEAD:
		raise ValueError(f"{name}: not an FFXI chunk file")
	return out


_sniff_cache: dict[tuple[str, int], dict[int, int] | None] = {}


def chunk_census(path: Path) -> dict[int, int] | None:
	"""{chunk type: count} of a DAT, walking only the headers; None if it is not a chunk file."""
	path = Path(path)
	try:
		st = path.stat()
	except OSError:
		return None
	key = (str(path), st.st_mtime_ns)
	if key in _sniff_cache:
		return _sniff_cache[key]
	found: dict[int, int] | None = {}
	try:
		with path.open("rb") as f:
			pos, n = 0, 0
			while pos + _HEAD <= st.st_size:
				f.seek(pos)
				head = f.read(8)
				if not _plausible_name(head[:4]):
					found = None
					break
				_name, ctype, size = _chunk_header(head, pos)
				if size == 0:
					size = st.st_size - pos
				if size < _HEAD or pos + size > st.st_size:
					found = None
					break
				found[ctype] = found.get(ctype, 0) + 1
				pos += size
				n += 1
				if n > _MAX_CHUNKS:
					found = None
					break
			if found is not None and (not found or st.st_size - pos >= _HEAD):
				found = None
	except OSError:
		found = None
	if len(_sniff_cache) > 50000:
		_sniff_cache.clear()
	_sniff_cache[key] = found
	return found


def is_model_dat(path: Path) -> bool:
	"""A .dat holding at least one FFXI mesh chunk."""
	path = Path(path)
	if path.suffix.lower() != ".dat" or path.name.lower().startswith("main_file_cache"):
		return False
	census = chunk_census(path)
	return bool(census) and census.get(T_MESH, 0) > 0


def is_ffxi(path: Path) -> bool:
	"""Anything this module converts: a model DAT, a .ffximdl manifest, or a .spw/.bgw sound."""
	ext = Path(path).suffix.lower()
	return ext == MANIFEST_EXTENSION or ext in AUDIO_EXTENSIONS or is_model_dat(path)


def dat_stem(path: Path) -> str:
	"""Output name for a DAT: ROM2/45/12.DAT → ffxi_rom2_45_12, so same-named DATs don't collide."""
	p = Path(path)
	parts = [x.lower() for x in p.parts[-3:]]
	if len(parts) == 3 and re.fullmatch(r"rom\d*", parts[0]) and parts[1].isdigit():
		return f"ffxi_{parts[0]}_{parts[1]}_{Path(parts[2]).stem}"
	return p.stem.lower()


# -- file tables ------------------------------------------------------------------


def _find_ci(folder: Path, name: str) -> Path | None:
	p = folder / name
	if p.exists():
		return p
	try:
		for c in folder.iterdir():
			if c.name.lower() == name.lower():
				return c
	except OSError:
		pass
	return None


class Rom:
	"""An FFXI install's VTABLE/FTABLE pairs: file id → ROMn/dir/file.DAT."""

	def __init__(self, root: Path):
		self.root = Path(root)
		self.tables: list[tuple[int, bytes, bytes]] = []	# (rom number, vtable, ftable)
		for n in range(1, 20):
			folder = self.root if n == 1 else _find_ci(self.root, f"ROM{n}")
			if folder is None:
				continue
			suffix = "" if n == 1 else str(n)
			vt, ft = _find_ci(folder, f"VTABLE{suffix}.DAT"), _find_ci(folder, f"FTABLE{suffix}.DAT")
			if vt is not None and ft is not None:
				self.tables.append((n, vt.read_bytes(), ft.read_bytes()))
		if not self.tables:
			raise ValueError(f"{self.root}: no VTABLE.DAT / FTABLE.DAT (is this the FINAL FANTASY XI folder?)")

	@staticmethod
	def is_root(path: Path) -> bool:
		p = Path(path)
		return p.is_dir() and _find_ci(p, "VTABLE.DAT") is not None and _find_ci(p, "FTABLE.DAT") is not None

	def locate(self, file_id: int) -> tuple[int, int, int] | None:
		"""(rom, dir, file) for a file id, or None."""
		for n, vt, ft in self.tables:
			if 0 <= file_id < len(vt) and vt[file_id] == n and 2 * file_id + 2 <= len(ft):
				v = struct.unpack_from("<H", ft, 2 * file_id)[0]
				return n, v >> 7, v & 0x7F
		return None

	def path(self, file_id: int) -> Path | None:
		loc = self.locate(file_id)
		if loc is None:
			return None
		n, d, f = loc
		rom = _find_ci(self.root, "ROM" if n == 1 else f"ROM{n}")
		sub = _find_ci(rom, str(d)) if rom is not None else None
		found = _find_ci(sub, f"{f}.DAT") if sub is not None else None
		return found or self.root / ("ROM" if n == 1 else f"ROM{n}") / str(d) / f"{f}.DAT"

	def count(self) -> int:
		return sum(sum(1 for b in vt if b == n) for n, vt, _ in self.tables)


# -- textures ---------------------------------------------------------------------


@dataclass
class Texture:
	name: str
	rgba: "np.ndarray"
	format: str


def _rgb565(c: "np.ndarray") -> "np.ndarray":
	r = (c >> 11) & 31
	g = (c >> 5) & 63
	b = c & 31
	return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.int32)


def decode_dxt(data: bytes, width: int, height: int, fourcc: str) -> "np.ndarray":
	"""DXT1/DXT3 → (h, w, 4) uint8. DXT3 alpha follows FFXI's half range: nibble 8+ is opaque."""
	bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
	step = 16 if fourcc == "DXT3" else 8
	need = bw * bh * step
	if len(data) < need:
		raise ValueError(f"{fourcc} data is {len(data)} bytes, {need} needed")
	blocks = np.frombuffer(data[:need], np.uint8).reshape(-1, step)
	col = blocks[:, -8:]
	c0 = col[:, 0].astype(np.int32) | (col[:, 1].astype(np.int32) << 8)
	c1 = col[:, 2].astype(np.int32) | (col[:, 3].astype(np.int32) << 8)
	idx = np.frombuffer(col[:, 4:8].tobytes(), "<u4").astype(np.int64)
	p0, p1 = _rgb565(c0), _rgb565(c1)
	four = (c0 > c1) | (fourcc != "DXT1")
	p2 = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
	p3 = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
	pal = np.stack([p0, p1, p2, p3], 1)	# (n, 4, 3)
	sel = (idx[:, None] >> (2 * np.arange(16))) & 3	# (n, 16)
	rgb = np.take_along_axis(pal, sel[:, :, None].repeat(3, 2), 1)
	if fourcc == "DXT3":
		nib = blocks[:, :8]
		a = np.stack([nib & 15, nib >> 4], -1).reshape(-1, 16).astype(np.int32)
		alpha = np.minimum(255, a * 32)
	else:	# punch-through black is drawn opaque: FFXI blends by material, not by DXT1 alpha
		alpha = np.full(sel.shape, 255, np.int32)
	px = np.concatenate([rgb, alpha[:, :, None]], -1).astype(np.uint8)
	img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
	return np.ascontiguousarray(img[:height, :width])


def _palette_at(data: bytes, at: int) -> "np.ndarray | None":
	if at + 1024 > len(data):
		return None
	return np.frombuffer(data[at:at + 1024], np.uint8).reshape(256, 4)


def _semi_alpha(a: "np.ndarray") -> "np.ndarray":
	"""FFXI palettes store alpha 0..0x80 (0x80 opaque)."""
	a = a.astype(np.int32)
	return np.where(a >= 0x80, 255, a * 2).astype(np.uint8)


def read_texture(data: bytes, name: str = "texture") -> Texture:
	"""A texture chunk's body (after the 16-byte header)."""
	if len(data) < 57:
		raise ValueError(f"{name}: texture header truncated")
	flag = data[0]
	tex_name = data[1:17].decode("latin-1").replace("\0", " ").strip()
	bi_size, width, height = struct.unpack_from("<Iii", data, 17)
	bits = struct.unpack_from("<H", data, 31)[0]
	if not (0 < width <= 4096 and 0 < height <= 4096):
		raise ValueError(f"{name}: bad texture size {width}x{height}")
	if flag in (0xA1, 0x81):
		fourcc = data[57:61][::-1].decode("latin-1")
		size = struct.unpack_from("<I", data, 61)[0]
		body = data[69:69 + size] if size else data[69:]
		if fourcc not in ("DXT1", "DXT3"):
			raise ValueError(f"{name}: texture format {fourcc!r} is not supported")
		return Texture(tex_name, decode_dxt(body, width, height, fourcc), fourcc)
	if flag in (0xB1, 0x91):
		if bits != 8:
			raise ValueError(f"{name}: {bits}-bit bitmaps are not supported")
		# POLUtils puts the palette right after the 57-byte header (after a 4-byte word
		# for 0xB1); Vanalytics reads 0xB1 at 64. Take the start whose alpha bytes look
		# like FFXI's 0..0x80 alpha.
		starts = [57] if flag == 0x91 else [61, 64]
		best, pal = None, None
		for at in starts:
			p = _palette_at(data, at)
			if p is None or at + 1024 + width * height > len(data):
				continue
			score = int((p[:, 3] <= 0x80).sum()) + int((p[:, 3] == 0x80).sum())
			if best is None or score > best[0]:
				best, pal = (score, at), p
		if pal is None:
			raise ValueError(f"{name}: palette texture truncated")
		at = best[1] + 1024
		idx = np.frombuffer(data[at:at + width * height], np.uint8).reshape(height, width)
		rgba = np.stack([pal[:, 2], pal[:, 1], pal[:, 0], _semi_alpha(pal[:, 3])], -1)[idx]
		return Texture(tex_name, np.ascontiguousarray(rgba), "8-bit palette")
	raise ValueError(f"{name}: texture flag {flag:#04x} is not supported")


# -- skeleton ---------------------------------------------------------------------


@dataclass
class Bone:
	parent: int	# -1 for the root
	rotation: "np.ndarray"	# (x, y, z, w)
	position: "np.ndarray"


def read_skeleton(data: bytes, name: str = "skeleton") -> list[Bone]:
	if len(data) < 4:
		raise ValueError(f"{name}: truncated")
	count = struct.unpack_from("<h", data, 2)[0]
	if not 0 < count <= 255 or 4 + 30 * count > len(data):
		raise ValueError(f"{name}: bad bone count {count}")
	bones = []
	for i in range(count):
		parent, _term, qx, qy, qz, qw, tx, ty, tz = struct.unpack_from("<BB7f", data, 4 + 30 * i)
		if i == 0 or parent == 255 or parent >= i:
			parent = -1 if i == 0 or parent == 255 else 0
		bones.append(Bone(parent, np.array([qx, qy, qz, qw]), np.array([tx, ty, tz])))
	return bones


def _local(q: "np.ndarray", t: "np.ndarray") -> "np.ndarray":
	m = np.eye(4)
	m[:3, :3] = _sk.quat_matrix(np.asarray(q, np.float64))
	m[:3, 3] = t
	return m


def world_matrices(bones: list[Bone], rot: "list | None" = None, pos: "list | None" = None) -> list["np.ndarray"]:
	"""FFXI-space 4x4 per bone (child = parent @ local). rot/pos override the bind pose."""
	out: list[np.ndarray] = []
	for i, b in enumerate(bones):
		m = _local(b.rotation if rot is None else rot[i], b.position if pos is None else pos[i])
		out.append(m if b.parent < 0 else out[b.parent] @ m)
	return out


# -- mesh -------------------------------------------------------------------------


@dataclass
class Mesh:
	name: str
	cloth: bool
	indirect: bool
	flip: bool
	bone_table: list[int]
	assign: "np.ndarray"	# (n, 6): low-left, low-right, low-mirror, high-left, high-right, high-mirror
	pos1: "np.ndarray"	# (n, 3) bone-local position for the first bone (weight-scaled for two-bone verts)
	pos2: "np.ndarray"	# (n, 3) second bone (zero for one-bone verts)
	weights: "np.ndarray"	# (n, 2)
	normals: "np.ndarray | None"	# (n, 3), bone-local, first bone
	two_bone: "np.ndarray"	# (n,) bool
	faces: "np.ndarray"	# (m, 3) vertex indices
	uvs: "np.ndarray"	# (m, 3, 2)
	textures: list[str]	# per face


def _words(data: bytes, off_words: int, count: int) -> list[int]:
	at = off_words * 2
	if at + 2 * count > len(data):
		raise ValueError("table runs past the mesh chunk")
	return list(struct.unpack_from(f"<{count}H", data, at))


def read_mesh(data: bytes, name: str = "mesh") -> Mesh:
	"""A mesh chunk's body: vertices, bone assignments and polygon commands."""
	if len(data) < 0x2A:
		raise ValueError(f"{name}: header truncated")
	(_ver, _nazo, mtype, flip, off_poly, poly_n, off_tbl, tbl_n, off_w, _w_n, off_bone, bone_n,
		off_vert, _vert_n) = struct.unpack_from("<BBHHIHIHIHIHIH", data, 0)
	cloth = (mtype & 0x7F) == 1
	indirect = bool(mtype & 0x80)
	if off_w * 2 + 4 > len(data):
		raise ValueError(f"{name}: weight table runs past the chunk")
	nb1, nb2 = struct.unpack_from("<HH", data, off_w * 2)
	n = nb1 + nb2
	table = _words(data, off_tbl, tbl_n) if tbl_n else []
	raw = _words(data, off_bone, (bone_n // 2) * 2) if bone_n else []
	assign = np.zeros((n, 6), np.int32)
	for i in range(min(n, len(raw) // 2)):
		lo, hi = raw[2 * i], raw[2 * i + 1]
		assign[i] = (lo & 0x7F, (lo >> 7) & 0x7F, (lo >> 14) & 3, hi & 0x7F, (hi >> 7) & 0x7F, (hi >> 14) & 3)
	at = off_vert * 2
	s1, s2 = (12, 32) if cloth else (24, 56)
	if at + nb1 * s1 + nb2 * s2 > len(data):
		raise ValueError(f"{name}: vertices run past the chunk")
	pos1 = np.zeros((n, 3))
	pos2 = np.zeros((n, 3))
	weights = np.zeros((n, 2))
	normals = None if cloth else np.zeros((n, 3))
	if nb1:
		v = np.frombuffer(data, "<f4", nb1 * s1 // 4, at).reshape(nb1, s1 // 4).astype(np.float64)
		pos1[:nb1] = v[:, :3]
		weights[:nb1, 0] = 1.0
		if normals is not None:
			normals[:nb1] = v[:, 3:6]
	if nb2:
		v = np.frombuffer(data, "<f4", nb2 * s2 // 4, at + nb1 * s1).reshape(nb2, s2 // 4).astype(np.float64)
		pos1[nb1:] = v[:, [0, 2, 4]]
		pos2[nb1:] = v[:, [1, 3, 5]]
		weights[nb1:] = v[:, 6:8]
		if normals is not None:
			normals[nb1:] = v[:, [8, 10, 12]]
	two = np.zeros(n, bool)
	two[nb1:] = True
	faces, uvs, texs = _polygons(data, off_poly, poly_n, name)
	return Mesh(name, cloth, indirect, bool(flip), table, assign, pos1, pos2, weights, normals, two,
		np.array(faces, np.int64).reshape(-1, 3), np.array(uvs, np.float64).reshape(-1, 3, 2), texs)


def _polygons(data: bytes, off: int, count: int, name: str) -> tuple[list, list, list[str]]:
	faces: list = []
	uvs: list = []
	texs: list[str] = []
	tex = ""
	pos, end = off * 2, min(len(data), (off + count) * 2)
	while pos + 2 <= end:
		cmd = struct.unpack_from("<H", data, pos)[0]
		if cmd == 0xFFFF:
			break
		if cmd & 0x80F0 == 0x8010:	# material state
			pos += 46
		elif cmd & 0x80F0 == 0x8000:	# texture change
			tex = data[pos + 2:pos + 18].decode("latin-1").replace("\0", " ").strip()
			pos += 18
		elif cmd == 0x0054:	# 'T': triangle list, 30 bytes per triangle
			k = struct.unpack_from("<H", data, pos + 2)[0]
			pos += 4
			for _ in range(k):
				i1, i2, i3, *uv = struct.unpack_from("<3h6f", data, pos)
				faces.append((i1, i2, i3))
				uvs.append(uv)
				texs.append(tex)
				pos += 30
		elif cmd == 0x5453:	# 'ST': strip, first triangle then 10 bytes per extra corner
			k = struct.unpack_from("<H", data, pos + 2)[0]
			pos += 4
			i1, i2, i3, u1, v1, u2, v2, u3, v3 = struct.unpack_from("<3h6f", data, pos)
			pos += 30
			strip = [(i1, u1, v1), (i2, u2, v2), (i3, u3, v3)]
			for _ in range(max(0, k - 1)):
				strip.append(struct.unpack_from("<h2f", data, pos))
				pos += 10
			for j in range(len(strip) - 2):
				a, b, c = strip[j], strip[j + 1], strip[j + 2]
				if a[0] == b[0] or b[0] == c[0] or a[0] == c[0]:
					continue
				if j % 2:
					b, c = c, b
				faces.append((a[0], b[0], c[0]))
				uvs.append((a[1], a[2], b[1], b[2], c[1], c[2]))
				texs.append(tex)
		elif cmd == 0x4353:	# 'SC'
			k = struct.unpack_from("<H", data, pos + 2)[0]
			pos += k * 20 + 12
		elif cmd == 0x0043:	# 'C'
			k = struct.unpack_from("<H", data, pos + 2)[0]
			pos += k * 10 + 4
		else:
			break	# unknown command: keep what was read
	return faces, uvs, texs


# -- motions ----------------------------------------------------------------------


@dataclass
class Track:
	bone: int
	rotation: "np.ndarray"	# (frames, 4) or (1, 4)
	translation: "np.ndarray"	# (frames, 3) or (1, 3)
	scale: "np.ndarray"	# (frames, 3) or (1, 3)


@dataclass
class Motion:
	name: str
	frames: int
	speed: float
	tracks: list[Track]

	@property
	def fps(self) -> float:
		return 30.0 * (self.speed if self.speed > 0 else 1.0)


def read_motion(data: bytes, name: str = "motion") -> Motion:
	if len(data) < 10:
		raise ValueError(f"{name}: truncated")
	_ver, _nazo, count, frames, speed = struct.unpack_from("<BBHHf", data, 0)
	if not 0 < count <= 512 or 10 + 84 * count > len(data):
		raise ValueError(f"{name}: bad element count {count}")
	pool = np.frombuffer(data, "<f4", (len(data) - 10) // 4, 10)

	def keys(idx: list[int], default: tuple) -> np.ndarray:
		if not any(idx):
			return np.array([default], np.float64)
		out = np.empty((max(1, frames), len(idx)))
		for c, (i, d) in enumerate(zip(idx, default)):
			if i > 0:
				if i + frames > len(pool):
					raise ValueError(f"{name}: key index {i} runs past the chunk")
				out[:, c] = pool[i:i + max(1, frames)]
			else:
				out[:, c] = d
		return out

	tracks = []
	for e in range(count):
		v = struct.unpack_from("<i4i4f3i3f3i3f", data, 10 + 84 * e)
		bone, iq, q, it, t, isc, sc = v[0], v[1:5], v[5:9], v[9:12], v[12:15], v[15:18], v[18:21]
		if iq[0] & 0x80000000 or iq[0] < 0:	# element not animated
			continue
		tracks.append(Track(bone, keys(list(iq), q), keys(list(it), t), keys(list(isc), sc)))
	return Motion(name, int(frames), float(speed), tracks)


def group_motions(motions: list[Motion]) -> list[tuple[str, list[Motion]]]:
	"""Motion chunks of one animation share a name prefix, frame count and speed and drive
	different bones (upper and lower body come as separate chunks). Group them."""
	groups: list[tuple[str, list[Motion]]] = []
	for mo in motions:
		key = mo.name[:3].lower()
		if groups:
			gname, cur = groups[-1]
			bones = {t.bone for m in cur for t in m.tracks}
			same = (cur[0].name[:3].lower() == key and cur[0].frames == mo.frames
				and abs(cur[0].speed - mo.speed) < 1e-4 and not bones & {t.bone for t in mo.tracks})
			if same:
				cur.append(mo)
				continue
		groups.append((key or "anim", [mo]))
	return groups


# -- DAT contents -----------------------------------------------------------------


@dataclass
class Dat:
	path: Path
	chunks: list[Chunk]
	textures: dict[str, Texture] = field(default_factory=dict)
	skeleton: list[Bone] | None = None
	meshes: list[Mesh] = field(default_factory=list)
	motions: list[Motion] = field(default_factory=list)
	problems: list[str] = field(default_factory=list)


def read_dat(path: Path, data: bytes | None = None) -> Dat:
	path = Path(path)
	data = path.read_bytes() if data is None else data
	dat = Dat(path, read_chunks(data, path.name))
	for c in dat.chunks:
		body = data[c.start:c.end]
		label = f"{path.name} {c.kind} {c.name!r}"
		try:
			if c.type == T_TEXTURE:
				tex = read_texture(body, label)
				dat.textures.setdefault(tex.name, tex)
			elif c.type == T_SKELETON and dat.skeleton is None:
				dat.skeleton = read_skeleton(body, label)
			elif c.type == T_MESH:
				dat.meshes.append(read_mesh(body, c.name or f"mesh{len(dat.meshes)}"))
			elif c.type == T_MOTION:
				dat.motions.append(read_motion(body, c.name or f"motion{len(dat.motions)}"))
		except (ValueError, struct.error) as exc:
			dat.problems.append(str(exc))
	return dat


def describe(path: Path) -> dict:
	"""What `gxbake ffxi` prints for a DAT."""
	dat = read_dat(path)
	counts: dict[str, int] = {}
	for c in dat.chunks:
		counts[c.kind] = counts.get(c.kind, 0) + 1
	return {
		"path": str(path), "chunks": counts,
		"textures": [{"name": t.name, "format": t.format, "size": f"{t.rgba.shape[1]}x{t.rgba.shape[0]}"}
			for t in dat.textures.values()],
		"bones": len(dat.skeleton) if dat.skeleton else 0,
		"meshes": [{"name": m.name, "vertices": len(m.pos1), "triangles": len(m.faces), "mirrored": m.flip}
			for m in dat.meshes],
		"motions": [{"name": g, "chunks": len(ms), "frames": ms[0].frames, "fps": round(ms[0].fps, 2)}
			for g, ms in group_motions(dat.motions)],
		"problems": dat.problems,
	}


# -- sounds -----------------------------------------------------------------------

_F0 = (0, 0xF0, 0x1CC, 0x188, 0x1E8)
_F1 = (0, 0, -0xD0, -0xDC, -0xF0)


@dataclass
class SoundInfo:
	kind: str	# "effect" or "music"
	format: int	# 0 ADPCM, 1 PCM, 3 ATRAC3
	id: int
	blocks: int
	loop_start: int	# in sample blocks (ADPCM) or samples; -1 = no loop
	rate: int
	channels: int
	block_size: int


def read_sound_header(data: bytes) -> SoundInfo:
	if data[:8] == b"SeWave\0\0":
		_size, fmt = struct.unpack_from("<iI", data, 8)
		kind = "effect"
		at = 16
	elif data[:12] == b"BGMStream\0\0\0":
		fmt, _size = struct.unpack_from("<Ii", data, 12)
		kind = "music"
		at = 20
	else:
		raise ValueError("not an FFXI .spw/.bgw sound (no SeWave / BGMStream marker)")
	sid, blocks, loop, hi, lo, _unk = struct.unpack_from("<6i", data, at)
	ch, bsize = data[at + 26], data[at + 27]
	return SoundInfo(kind, fmt, sid, blocks, loop, hi + lo, ch, bsize)


def decode_sound(data: bytes) -> tuple[SoundInfo, "np.ndarray"]:
	"""(header, int16 samples of shape (frames, channels))."""
	info = read_sound_header(data)
	body = data[0x30:]
	if info.channels < 1 or info.channels > 6 or info.rate <= 0:
		raise ValueError(f"bad sound header: {info.channels} channels at {info.rate} Hz")
	if info.format == 1:
		n = len(body) // (2 * info.channels) * info.channels
		return info, np.frombuffer(body[:n * 2], "<i2").reshape(-1, info.channels).copy()
	if info.format != 0:
		raise ValueError(f"sound format {info.format} (ATRAC3?) is not supported")
	bs = info.block_size
	if bs < 2 or bs % 2:
		raise ValueError(f"bad ADPCM block size {bs}")
	per = 1 + bs // 2
	frame = per * info.channels
	nblocks = min(info.blocks if info.blocks > 0 else 1 << 30, len(body) // frame)
	raw = np.frombuffer(body[:nblocks * frame], np.uint8).reshape(nblocks, info.channels, per)
	out = np.zeros((nblocks * bs, info.channels), np.int16)
	for c in range(info.channels):
		s0 = s1 = 0
		hdr = raw[:, c, 0]
		nib = raw[:, c, 1:]
		vals = np.stack([nib & 15, nib >> 4], -1).reshape(nblocks, bs).astype(np.int32)
		vals = np.where(vals >= 8, vals - 16, vals)
		col = out[:, c]
		for b in range(nblocks):
			scale = 0x0C - (int(hdr[b]) & 15)
			fi = int(hdr[b]) >> 4
			if fi >= 5:	# the reference decoder leaves such blocks silent
				continue
			f0, f1 = _F0[fi], _F1[fi]
			base = b * bs
			row = vals[b]
			for k in range(bs):
				v = (int(row[k]) << scale) if scale >= 0 else (int(row[k]) >> -scale)
				p = s0 * f0 + s1 * f1
				v += int(p / 256) if p >= 0 else -int(-p / 256)	# C# integer division truncates
				s1 = s0
				s0 = 32767 if v > 32767 else -32768 if v < -32768 else v
				col[base + k] = s0
	return info, out


def sound_to_wav(data: bytes) -> tuple[bytes, SoundInfo]:
	"""A .spw/.bgw as a 16-bit PCM WAV (with a cue point at the loop start)."""
	from . import sfx

	info, pcm = decode_sound(data)
	loop = None
	if info.loop_start >= 0:
		loop = info.loop_start * (info.block_size if info.format == 0 else 1)
		loop = loop if loop < len(pcm) else None
	snd = sfx.Sound(info.rate, pcm.astype(np.float32) / 32768, 2, loop_start=loop)
	return sfx.write_wav(snd), info


# -- import -----------------------------------------------------------------------


@dataclass
class _Parts:
	skeleton: list[Bone] | None
	meshes: list[tuple[Mesh, dict[str, Texture]]]
	motions: list[Motion]


def _resolve(ref: str, base: Path, rom: Rom | None) -> Path:
	ref = str(ref)
	if ref.lower().startswith("id:"):
		if rom is None:
			raise ValueError(f"{ref}: file ids need \"root\" (the FINAL FANTASY XI folder) in the manifest")
		p = rom.path(int(ref[3:]))
		if p is None:
			raise ValueError(f"{ref}: not in the file tables")
		return p
	p = Path(ref)
	if not p.is_absolute():
		for b in ([rom.root] if rom else []) + [base]:
			q = b / p
			if q.exists():
				return q
		return base / p
	return p


def _gather(path: Path, opts: "_m.ModelOptions", warnings: list[str]) -> tuple[str, _Parts]:
	path = Path(path)
	skel_ref = getattr(opts, "ffxi_skeleton", "") or ""
	if path.suffix.lower() == MANIFEST_EXTENSION:
		spec = json.loads(path.read_text(encoding="utf-8"))
		rom = Rom(_resolve(spec["root"], path.parent, None)) if spec.get("root") else None
		parts = [_resolve(r, path.parent, rom) for r in spec.get("parts", [])]
		motions = [_resolve(r, path.parent, rom) for r in spec.get("motions", [])]
		if spec.get("skeleton"):
			skel_ref = str(_resolve(spec["skeleton"], path.parent, rom))
		name = spec.get("name") or path.stem
	else:
		parts, motions, name = [path], [], dat_stem(path)
	out = _Parts(None, [], [])
	for p in parts:
		dat = read_dat(p)
		warnings += [f"{p.name}: {x}" for x in dat.problems]
		if dat.skeleton and out.skeleton is None:
			out.skeleton = dat.skeleton
		out.meshes += [(m, dat.textures) for m in dat.meshes]
		out.motions += dat.motions
	if skel_ref and out.skeleton is None:
		sdat = read_dat(Path(skel_ref))
		if sdat.skeleton is None:
			raise ValueError(f"{Path(skel_ref).name}: no skeleton chunk")
		out.skeleton = sdat.skeleton
		out.motions += sdat.motions
	for p in motions:
		mdat = read_dat(p)
		warnings += [f"{p.name}: {x}" for x in mdat.problems]
		out.motions += mdat.motions
	if not out.meshes:
		raise ValueError(f"{path.name}: no meshes")
	return name, out


def import_ffxi(path: Path, opts: "_m.ModelOptions | None" = None) -> "_m.ImportedModel":
	"""A model DAT (NPC, monster, equipment, weapon) or a .ffximdl manifest → ImportedModel."""
	opts = opts or _m.ModelOptions()
	warnings: list[str] = []
	name, parts = _gather(path, opts, warnings)
	return _assemble(name, parts, opts, warnings)


def _to_gold(opts: "_m.ModelOptions") -> tuple["np.ndarray", float]:
	base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])	# y-down → Z-up
	a = math.radians(getattr(opts, "ffxi_yaw", 0.0))
	c, s = math.cos(a), math.sin(a)
	yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
	return yaw @ base, float(getattr(opts, "ffxi_scale", DEFAULT_SCALE))


def _gold(m: "np.ndarray", rot: "np.ndarray", k: float) -> "np.ndarray":
	out = np.eye(4)
	out[:3, :3] = rot @ m[:3, :3] @ rot.T
	out[:3, 3] = k * (rot @ m[:3, 3])
	return out


_MIRROR = {1: np.diag([-1.0, 1.0, 1.0]), 2: np.diag([1.0, -1.0, 1.0]), 3: np.diag([1.0, 1.0, -1.0])}


def _skin(mesh: Mesh, bones: list[Bone] | None, worlds: list["np.ndarray"], mirror: bool):
	"""Bind-pose FFXI-space positions, normals and the heaviest bone of every vertex."""
	n = len(mesh.pos1)
	nb = len(worlds)

	def bone_of(slot: "np.ndarray") -> "np.ndarray":
		idx = slot.copy()
		if mesh.indirect:
			tbl = np.array(mesh.bone_table + [0], np.int64)
			idx = tbl[np.minimum(idx, len(tbl) - 1)]
		return np.where(idx < nb, idx, 0)

	lo = bone_of(mesh.assign[:, 1 if mirror else 0])
	hi = bone_of(mesh.assign[:, 4 if mirror else 3])
	if bones is None:
		pos = mesh.pos1.copy()
		nrm = mesh.normals.copy() if mesh.normals is not None else None
		if mirror:
			flg = mesh.assign[:, 2]
			for f, mm in _MIRROR.items():
				sel = flg == f
				pos[sel] = pos[sel] @ mm
				if nrm is not None:
					nrm[sel] = nrm[sel] @ mm
		return pos, nrm, np.zeros(n, np.int64)
	w = np.array(worlds)
	p1, p2 = mesh.pos1.copy(), mesh.pos2.copy()
	nrm = mesh.normals.copy() if mesh.normals is not None else None
	if mirror:
		for col, arr in ((2, p1), (5, p2)):
			flg = mesh.assign[:, col]
			for f, mm in _MIRROR.items():
				sel = flg == f
				arr[sel] = arr[sel] @ mm
				if nrm is not None and col == 2:
					nrm[sel] = nrm[sel] @ mm
	r1, t1 = w[lo, :3, :3], w[lo, :3, 3]
	r2, t2 = w[hi, :3, :3], w[hi, :3, 3]
	one = np.einsum("nij,nj->ni", r1, p1) + t1
	# two-bone vertices: each position is pre-scaled by its weight, the weight is the w term
	two = (np.einsum("nij,nj->ni", r1, p1) + mesh.weights[:, :1] * t1
		+ np.einsum("nij,nj->ni", r2, p2) + mesh.weights[:, 1:] * t2)
	pos = np.where(mesh.two_bone[:, None], two, one)
	if nrm is not None:
		nrm = np.einsum("nij,nj->ni", r1, nrm)
	heavy = np.where(mesh.two_bone & (mesh.weights[:, 1] > mesh.weights[:, 0]), hi, lo)
	return pos, nrm, heavy


def _assemble(name: str, parts: _Parts, opts: "_m.ModelOptions", warnings: list[str]) -> "_m.ImportedModel":
	rot, k = _to_gold(opts)
	model = _m.ImportedModel(name, [], [], warnings)
	bones = parts.skeleton
	if bones is not None and len(bones) > _sk.MAXSTUDIOBONES:
		warnings.append(f"{len(bones)} bones exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static")
		bones = None
	worlds = world_matrices(bones) if bones else []
	gold_world = [_gold(w, rot, k) for w in worlds] if bones else [np.eye(4)]
	if bones:
		model.bones = []
		for i, b in enumerate(bones):
			g = gold_world[i]
			local = g if b.parent < 0 else _sk.rigid_inverse(gold_world[b.parent]) @ g
			model.bones.append(_m.Bone(f"bone{i}", b.parent, _sk.pose_from_matrix(local)))
	else:
		skinned = any(m.assign[:, [0, 3]].any() for m, _ in parts.meshes)
		if skinned:
			warnings.append("meshes are bound to bones but no skeleton was found: give the race skeleton DAT "
				"(--ffxi-skeleton or a .ffximdl manifest), or the parts will sit at the origin")

	mat_index: dict[str, int] = {}
	blended = False
	for mesh, textures in parts.meshes:
		if len(mesh.faces) == 0:
			continue
		nv = len(mesh.pos1)
		good = (mesh.faces >= 0).all(1) & (mesh.faces < nv).all(1)
		if not good.all():
			warnings.append(f"mesh {mesh.name}: {int((~good).sum())} triangles with bad vertex indices dropped")
		if bones and (mesh.two_bone & (mesh.weights.min(1) > 0.001)).any():
			blended = True
		for mirror in ((False, True) if mesh.flip else (False,)):
			pos, nrm, heavy = _skin(mesh, bones, worlds, mirror)
			world = k * (pos @ rot.T)
			faces = mesh.faces[good]
			uvs = mesh.uvs[good]
			texs = [t for t, ok in zip(mesh.textures, good) if ok]
			if mirror:	# a mirror image turns the winding over
				faces = faces[:, [0, 2, 1]]
				uvs = uvs[:, [0, 2, 1]]
			# wind every mesh counter-clockwise from outside, judged by its stored normals
			fn = np.cross(world[faces[:, 1]] - world[faces[:, 0]], world[faces[:, 2]] - world[faces[:, 0]])
			if nrm is not None:
				vn = nrm @ rot.T
				score = np.einsum("ni,ni->n", fn, vn[faces].sum(1))
				if (score < 0).sum() > (score > 0).sum():
					faces = faces[:, [0, 2, 1]]
					uvs = uvs[:, [0, 2, 1]]
					fn = -fn
			else:
				vn = np.zeros_like(world)
				for c in range(3):
					np.add.at(vn, faces[:, c], fn)
			for tex in dict.fromkeys(texs):
				sel = np.array([t == tex for t in texs])
				f = faces[sel]
				corner = f.reshape(-1)
				p = world[corner]
				nn = vn[corner]
				lens = np.linalg.norm(nn, axis=1, keepdims=True)
				nn = np.where(lens > 1e-9, nn / np.maximum(lens, 1e-9), [0.0, 0.0, 1.0])
				vb = heavy[corner].astype(np.int32)
				if bones:
					bw = np.array(gold_world)[vb]
					local = np.einsum("nji,nj->ni", bw[:, :3, :3], p - bw[:, :3, 3])
					ln = np.einsum("nji,nj->ni", bw[:, :3, :3], nn)
				else:
					local, ln = p, nn
				if tex not in mat_index:
					mat_index[tex] = len(model.materials)
					model.materials.append(_material(tex, textures, opts, warnings))
				tris = np.arange(len(p), dtype=np.uint32).reshape(-1, 3)
				model.meshes.append(_m.Submesh(mat_index[tex], local.astype(np.float32), ln.astype(np.float32),
					uvs[sel].reshape(-1, 2).astype(np.float32), tris, vb, p.astype(np.float32)))
	if not model.meshes:
		raise ValueError(f"{name}: no triangles")
	if blended:
		warnings.append("two-bone vertices are snapped to their heavier bone (studio skinning is rigid)")

	if bones and not opts.static:
		used: set[str] = set()
		for gname, group in group_motions(parts.motions):
			frames = _bake(group, bones, rot, k, opts.anim_fps, warnings)
			if frames is not None:
				model.sequences.append(_m.Sequence(_m._unique(_m._seq_name(gname), used), float(opts.anim_fps), frames))
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _material(tex: str, textures: dict[str, Texture], opts: "_m.ModelOptions", warnings: list[str]) -> "_m.Material":
	t = textures.get(tex)
	if t is None and tex:
		low = {k.lower(): v for k, v in textures.items()}
		t = low.get(tex.lower())
	if t is None:
		if tex:
			warnings.append(f"texture {tex!r} is not in the DAT; using grey")
		rgba = np.full((8, 8, 4), (200, 200, 200, 255), np.uint8)
	else:
		rgba = t.rgba
	rgba = _m._fit_skin(rgba, opts.skin_max_edge).copy()
	masked = bool((rgba[..., 3] < opts.alpha_threshold).any())
	rgba[..., 3] = np.where(rgba[..., 3] < opts.alpha_threshold, 0, 255) if masked else 255
	label = re.sub(r"[^A-Za-z0-9_.-]+", "_", tex).strip("_") or "untextured"
	return _m.Material(label[:63], rgba, masked, tex)


def _qmul(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
	ax, ay, az, aw = a
	bx, by, bz, bw = b
	return np.array([
		aw * bx + ax * bw + ay * bz - az * by,
		aw * by - ax * bz + ay * bw + az * bx,
		aw * bz + ax * by - ay * bx + az * bw,
		aw * bw - ax * bx - ay * by - az * bz,
	])


def _sample(arr: "np.ndarray", t: float, quat: bool) -> "np.ndarray":
	if len(arr) == 1:
		return arr[0]
	i0 = int(math.floor(t))
	i0 = min(max(i0, 0), len(arr) - 1)
	i1 = min(i0 + 1, len(arr) - 1)
	f = t - i0
	if quat:
		a = arr[i0] / (np.linalg.norm(arr[i0]) or 1.0)
		b = arr[i1] / (np.linalg.norm(arr[i1]) or 1.0)
		return _sk._slerp(a, b, f)
	return arr[i0] + (arr[i1] - arr[i0]) * f


def _bake(group: list[Motion], bones: list[Bone], rot: "np.ndarray", k: float, fps: int,
		warnings: list[str]) -> "np.ndarray | None":
	"""Motion chunks → (frames, bones, 6) studio poses. A key's rotation is applied before the
	bone's bind rotation and its translation is added to the bind translation."""
	tracks = [t for m in group for t in m.tracks if 0 <= t.bone < len(bones)]
	if not tracks:
		warnings.append(f"motion {group[0].name}: drives no bone of this skeleton; skipped")
		return None
	nsrc = max(1, group[0].frames)
	src_fps = group[0].fps
	length = (nsrc - 1) / src_fps
	count = max(1, int(round(length * fps)) + 1)
	scaled = False
	frames = np.zeros((count, len(bones), 6))
	for f in range(count):
		t = min(nsrc - 1, f / fps * src_fps)
		qs = [b.rotation for b in bones]
		ts = [b.position for b in bones]
		for tr in tracks:
			b = bones[tr.bone]
			q = _sample(tr.rotation, t, True)
			qs[tr.bone] = _qmul(q / (np.linalg.norm(q) or 1.0), b.rotation)
			ts[tr.bone] = b.position + _sample(tr.translation, t, False)
			if not scaled and np.abs(_sample(tr.scale, t, False) - 1.0).max() > 0.01:
				scaled = True
		worlds = [_gold(w, rot, k) for w in world_matrices(bones, qs, ts)]
		for i, b in enumerate(bones):
			rel = worlds[i] if b.parent < 0 else _sk.rigid_inverse(worlds[b.parent]) @ worlds[i]
			frames[f, i] = _sk.pose_from_matrix(rel)
	if scaled:
		warnings.append(f"motion {group[0].name}: scale keys ignored (studio bones are rigid)")
	return frames


def extract_textures(path: Path, out_dir: Path) -> list[Path]:
	"""Every texture of a DAT as PNG (for the image pipeline, or to look at)."""
	from PIL import Image

	dat = read_dat(path)
	out_dir.mkdir(parents=True, exist_ok=True)
	written = []
	used: set[str] = set()
	for t in dat.textures.values():
		base = re.sub(r"[^A-Za-z0-9_.-]+", "_", t.name).strip("_") or "texture"
		stem, n = base, 1
		while stem.lower() in used:
			n += 1
			stem = f"{base}_{n}"
		used.add(stem.lower())
		p = out_dir / f"{stem}.png"
		Image.fromarray(t.rgba, "RGBA").save(p)
		written.append(p)
	return written
