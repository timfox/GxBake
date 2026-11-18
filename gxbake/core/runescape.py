"""RuneScape caches → GoldSrc studio MDL.

Two cache layouts are read in place, without extracting anything:

	JS5 (Old School RuneScape, and RS2 from 2006 on)
		main_file_cache.dat2 + main_file_cache.idx0..idxN, plus idx255 (the
		reference tables). Archives are containers (none / bzip2 / gzip), and an
		archive may hold several files.
	Legacy (2004-era RS2, "#317"-style clients)
		main_file_cache.dat + main_file_cache.idx0..idx4. Store 1 holds the
		models, each one gzipped.

What is converted (OSRS index numbers):

	7	models: all four model encodings (the original format, and the ones the
		file marks with 0xFFFF / 0xFFFE / 0xFFFD in its last two bytes)
	2/9	NPC definitions: models, recolours, retextures, width/height scale,
		stand and walk sequences
	2/12	sequences, 0 frames, 1 framemaps: classic vertex-group animation
	9, 8	texture definitions → sprites, for textured faces

RuneScape models carry a 16-bit HSL colour per face instead of textures. Those
colours are baked into a small palette skin (one 4x4 CMPR block per colour) and
each face samples the middle of its block. Textured faces get the texture's
sprite as a skin and their UVs from the model's texture triangles.

Animations move groups of vertices ("labels"), not bones. Each frame is a list of
origin / translate / rotate / scale operations on sets of labels, applied to the
whole group, so every label group moves rigidly (scale aside). GxBake gives every
label group used by the model its own studio bone and bakes each frame's group
transforms into bone poses.

Axes: RuneScape model space is x east, y down, z north, and a model with
orientation 0 faces south (-z). GoldSrc is X forward, Y left, Z up, so
(x, y, z) → (-z, x, -y) · rs_scale, then a turn of rs_yaw degrees about Z.
128 model units are one tile.
"""

from __future__ import annotations

import bz2
import gzip
import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as _m
from . import skeleton as _sk

CACHE_NAMES = {"main_file_cache.dat2", "main_file_cache.dat"}
ENTRY_KINDS = ("npc", "model")

# OSRS index and config-archive numbers.
IDX_FRAMES, IDX_FRAMEMAPS, IDX_CONFIGS, IDX_MODELS, IDX_SPRITES, IDX_TEXTURES = 0, 1, 2, 7, 8, 9
CFG_NPC, CFG_SEQ = 9, 12
LEGACY_MODELS = 1	# legacy store holding gzipped models

SECTOR = 520
TICK = 0.02	# sequence frame lengths are in 20 ms client cycles
DEFAULT_SCALE = 0.35	# GoldSrc units per model unit: a ~200 unit tall player becomes ~70 units
BRIGHTNESS = 0.7	# the client's default brightness for HSL → RGB


def is_cache(path: Path) -> bool:
	return Path(path).name.lower() in CACHE_NAMES


# -- bytes ----------------------------------------------------------------------


class _Buf:
	"""Big-endian reader with the client's "smart" encodings."""

	def __init__(self, data: bytes, pos: int = 0, name: str = ""):
		self.data, self.pos, self.name = data, pos, name

	def need(self, n: int) -> None:
		if self.pos < 0 or self.pos + n > len(self.data):
			raise ValueError(f"{self.name}: truncated at byte {self.pos}")

	def u8(self) -> int:
		self.need(1)
		self.pos += 1
		return self.data[self.pos - 1]

	def i8(self) -> int:
		v = self.u8()
		return v - 256 if v > 127 else v

	def u16(self) -> int:
		self.need(2)
		self.pos += 2
		return (self.data[self.pos - 2] << 8) | self.data[self.pos - 1]

	def i16(self) -> int:
		v = self.u16()
		return v - 65536 if v > 32767 else v

	def u24(self) -> int:
		return (self.u8() << 16) | self.u16()

	def i32(self) -> int:
		self.need(4)
		self.pos += 4
		return struct.unpack_from(">i", self.data, self.pos - 4)[0]

	def u32(self) -> int:
		return self.i32() & 0xFFFFFFFF

	def peek(self) -> int:
		self.need(1)
		return self.data[self.pos]

	def smart(self) -> int:
		"""Signed short smart: one byte -64..63, else two bytes -16384..16383."""
		return self.u8() - 64 if self.peek() < 128 else self.u16() - 0xC000

	def big_smart(self) -> int:
		return self.u16() if self.peek() < 128 else self.i32() & 0x7FFFFFFF

	def string(self) -> str:
		end = self.data.find(b"\0", self.pos)
		if end < 0:
			raise ValueError(f"{self.name}: unterminated string at byte {self.pos}")
		raw = self.data[self.pos:end]
		self.pos = end + 1
		return raw.decode("cp1252", errors="replace")

	def skip(self, n: int) -> None:
		self.need(n)
		self.pos += n


# -- cache store ----------------------------------------------------------------


@dataclass
class ArchiveRef:
	id: int
	name_hash: int = 0
	crc: int = 0
	revision: int = 0
	files: list[int] = field(default_factory=list)


@dataclass
class RefTable:
	protocol: int
	revision: int
	archives: dict[int, ArchiveRef]


def read_reference_table(data: bytes) -> RefTable:
	"""A JS5 index's reference table (archive 'idx' in store 255)."""
	b = _Buf(data, name="reference table")
	protocol = b.u8()
	if not 5 <= protocol <= 7:
		raise ValueError(f"reference table protocol {protocol} is not supported (5-7)")
	revision = b.i32() if protocol >= 6 else 0
	flags = b.u8()
	named, sized = bool(flags & 1), bool(flags & 4)
	if flags & ~5:
		raise ValueError(f"reference table flags {flags:#x} are not supported (RS3 cache?)")
	count_fn = b.big_smart if protocol >= 7 else b.u16
	n = count_fn()
	ids, last = [], 0
	for _ in range(n):
		last += count_fn()
		ids.append(last)
	arcs = [ArchiveRef(i) for i in ids]
	if named:
		for a in arcs:
			a.name_hash = b.i32()
	for a in arcs:
		a.crc = b.i32()
	if sized:
		b.skip(8 * n)
	for a in arcs:
		a.revision = b.i32()
	counts = [count_fn() for _ in range(n)]
	for a, c in zip(arcs, counts):
		last = 0
		for _ in range(c):
			last += count_fn()
			a.files.append(last)
	# file name hashes follow when named; nothing here needs them
	return RefTable(protocol, revision, {a.id: a for a in arcs})


def decompress_container(data: bytes, name: str = "container") -> bytes:
	"""A JS5 container: compression byte, length, payload (+ an optional revision)."""
	b = _Buf(data, name=name)
	kind = b.u8()
	length = b.i32()
	if length < 0:
		raise ValueError(f"{name}: bad container length")
	if kind == 0:
		b.need(length)
		return data[5:5 + length]
	b.need(4 + length)
	size = b.i32()
	payload = data[9:9 + length]
	if kind == 1:	# bzip2 with the "BZh1" header stripped
		out = bz2.decompress(b"BZh1" + payload)
	elif kind == 2:
		out = gzip.decompress(payload)
	else:
		raise ValueError(f"{name}: compression {kind} is not supported (LZMA is RS3 only)")
	if len(out) != size:
		raise ValueError(f"{name}: decompressed {len(out)} bytes, expected {size}")
	return out


def split_archive(data: bytes, count: int) -> list[bytes]:
	"""Split a multi-file archive into its files (chunk table at the end)."""
	if count <= 1:
		return [data]
	chunks = data[-1]
	table = len(data) - 1 - chunks * count * 4
	if table < 0:
		raise ValueError("archive chunk table is truncated")
	b = _Buf(data, table, "archive")
	sizes = np.zeros((chunks, count), np.int64)
	for c in range(chunks):
		size = 0
		for f in range(count):
			size += b.i32()
			sizes[c, f] = size
	parts: list[list[bytes]] = [[] for _ in range(count)]
	pos = 0
	for c in range(chunks):
		for f in range(count):
			n = int(sizes[c, f])
			parts[f].append(data[pos:pos + n])
			pos += n
	return [b"".join(p) for p in parts]


class Cache:
	"""A RuneScape cache folder: main_file_cache.dat2 (JS5) or .dat (legacy)."""

	def __init__(self, path: Path):
		path = Path(path)
		folder = path if path.is_dir() else path.parent
		dat2, dat = _find(folder, "main_file_cache.dat2"), _find(folder, "main_file_cache.dat")
		if path.is_file() and is_cache(path):
			main = path
		else:
			main = dat2 or dat
		if main is None:
			raise ValueError(f"{path}: no main_file_cache.dat2 or main_file_cache.dat")
		self.path = main
		self.folder = main.parent
		self.legacy = main.name.lower() == "main_file_cache.dat"
		self._data = main.read_bytes()
		self._idx: dict[int, bytes | None] = {}
		self._refs: dict[int, RefTable | None] = {}
		self._archives: dict[tuple[int, int], dict[int, bytes]] = {}

	@property
	def kind(self) -> str:
		return "legacy (2004-era RS2)" if self.legacy else "JS5 (OSRS / RS2)"

	def _index(self, index: int) -> bytes | None:
		if index not in self._idx:
			p = _find(self.folder, f"main_file_cache.idx{index}")
			self._idx[index] = p.read_bytes() if p else None
		return self._idx[index]

	def indices(self) -> list[int]:
		return [i for i in range(255) if self._index(i)]

	def count(self, index: int) -> int:
		idx = self._index(index)
		return len(idx) // 6 if idx else 0

	def raw(self, index: int, archive: int) -> bytes | None:
		"""An archive's stored bytes, following its sector chain."""
		idx = self._index(index)
		if idx is None or archive < 0 or (archive + 1) * 6 > len(idx):
			return None
		e = idx[archive * 6:archive * 6 + 6]
		size = (e[0] << 16) | (e[1] << 8) | e[2]
		sector = (e[3] << 16) | (e[4] << 8) | e[5]
		if size <= 0 or sector <= 0:
			return None
		big = archive > 0xFFFF
		head = 10 if big else 8
		out = bytearray()
		part = 0
		data = self._data
		while len(out) < size:
			if sector <= 0 or (sector + 1) * SECTOR > len(data) + SECTOR:
				return None
			off = sector * SECTOR
			h = data[off:off + head]
			if len(h) < head:
				return None
			if big:
				a, p, nxt, ix = struct.unpack(">IH3sB", h)
			else:
				a, p, nxt, ix = struct.unpack(">HH3sB", h)
			# JS5 stores the index id; the legacy cache stores it plus one
			if a != archive or p != part or ix not in (index, index + 1):
				return None
			take = min(size - len(out), SECTOR - head)
			out += data[off + head:off + head + take]
			sector = int.from_bytes(nxt, "big")
			part += 1
		return bytes(out)

	def reference(self, index: int) -> RefTable | None:
		if self.legacy:
			return None
		if index not in self._refs:
			raw = self.raw(255, index)
			try:
				self._refs[index] = read_reference_table(decompress_container(raw, f"index {index} table")) if raw else None
			except ValueError:
				self._refs[index] = None
		return self._refs[index]

	def archive_ids(self, index: int) -> list[int]:
		ref = self.reference(index)
		if ref is not None:
			return sorted(ref.archives)
		return [a for a in range(self.count(index)) if self._entry_size(index, a)]

	def _entry_size(self, index: int, archive: int) -> int:
		idx = self._index(index)
		e = idx[archive * 6:archive * 6 + 3] if idx else b""
		return int.from_bytes(e, "big") if len(e) == 3 else 0

	def files(self, index: int, archive: int) -> dict[int, bytes]:
		"""The files of one JS5 archive, by file id ({} when missing)."""
		key = (index, archive)
		if key in self._archives:
			return self._archives[key]
		raw = self.raw(index, archive)
		out: dict[int, bytes] = {}
		if raw is not None:
			data = decompress_container(raw, f"index {index} archive {archive}")
			ref = self.reference(index)
			ids = ref.archives[archive].files if ref and archive in ref.archives else [0]
			out = dict(zip(ids, split_archive(data, len(ids))))
		if len(self._archives) > 64:
			self._archives.clear()
		self._archives[key] = out
		return out

	def file(self, index: int, archive: int, file: int = 0) -> bytes | None:
		files = self.files(index, archive)
		if file in files:
			return files[file]
		return next(iter(files.values())) if len(files) == 1 and file == 0 else None

	def model_data(self, model: int) -> bytes | None:
		if self.legacy:
			raw = self.raw(LEGACY_MODELS, model)
			if raw is None:
				return None
			try:
				return gzip.decompress(raw)
			except (OSError, EOFError, zlib.error):
				return raw
		files = self.files(IDX_MODELS, model)
		return next(iter(files.values()), None)

	def model_ids(self) -> list[int]:
		return self.archive_ids(LEGACY_MODELS if self.legacy else IDX_MODELS)


def _find(folder: Path, name: str) -> Path | None:
	p = folder / name
	if p.is_file():
		return p
	try:
		for c in folder.iterdir():
			if c.name.lower() == name and c.is_file():
				return c
	except OSError:
		pass
	return None


# -- models ---------------------------------------------------------------------


@dataclass
class RsModel:
	vertices: "np.ndarray"	# (n, 3) int32, model units, y down
	faces: "np.ndarray"	# (m, 3) int32
	colors: "np.ndarray"	# (m,) uint16 HSL (or texture id in old textured faces before decoding)
	render_types: "np.ndarray"	# (m,) int8: 0 smooth, 1 flat, 2 hidden, 3 flat black
	alphas: "np.ndarray"	# (m,) uint8, 0 opaque
	textures: "np.ndarray"	# (m,) int32, -1 untextured
	tex_coords: "np.ndarray"	# (m,) int32, texture triangle per face, -1 = the face itself
	tex_tris: "np.ndarray"	# (k, 3) int32 (P, M, N vertices)
	groups: "np.ndarray | None"	# (n,) label per vertex, for animation
	encoding: str = "old"
	rigged: bool = False	# carries skeletal ("animaya") weights, which GxBake ignores

	@property
	def num_vertices(self) -> int:
		return len(self.vertices)

	@property
	def num_faces(self) -> int:
		return len(self.faces)


def read_model(data: bytes, name: str = "model") -> RsModel:
	if len(data) < 18:
		raise ValueError(f"{name}: too short for a model")
	tail = data[-2:]
	if tail == b"\xff\xfd":
		return _read_new(data, name, rigged=True)
	if tail == b"\xff\xfe":
		return _read_old(data, name, rigged=True)
	if tail == b"\xff\xff":
		return _read_new(data, name, rigged=False)
	return _read_old(data, name, rigged=False)


def _vertices(data: bytes, flags_at: int, xyz_at: tuple[int, int, int], n: int, name: str) -> "np.ndarray":
	f = _Buf(data, flags_at, name)
	bx, by, bz = (_Buf(data, o, name) for o in xyz_at)
	out = np.zeros((n, 3), np.int32)
	x = y = z = 0
	for v in range(n):
		fl = f.u8()
		x += bx.smart() if fl & 1 else 0
		y += by.smart() if fl & 2 else 0
		z += bz.smart() if fl & 4 else 0
		out[v] = (x, y, z)
	return out


def _faces(data: bytes, types_at: int, idx_at: int, m: int, name: str) -> "np.ndarray":
	t = _Buf(data, types_at, name)
	d = _Buf(data, idx_at, name)
	out = np.zeros((m, 3), np.int32)
	a = b = c = last = 0
	for i in range(m):
		kind = t.u8()
		if kind == 1:
			a = d.smart() + last
			b = d.smart() + a
			c = d.smart() + b
			last = c
		elif kind == 2:
			b = c
			c = d.smart() + last
			last = c
		elif kind == 3:
			a = c
			c = d.smart() + last
			last = c
		elif kind == 4:
			a, b = b, a
			c = d.smart() + last
			last = c
		out[i] = (a, b, c)
	return out


def _groups(data: bytes, at: int, n: int, packed: bool, rigged: bool, name: str) -> "np.ndarray | None":
	b = _Buf(data, at, name)
	groups = np.array([b.u8() for _ in range(n)], np.int32) if packed else None
	if rigged:	# skeletal weights follow: per vertex a count, then (bone, weight) pairs
		for _ in range(n):
			b.skip(2 * b.u8())
	return groups


def _read_old(data: bytes, name: str, rigged: bool) -> RsModel:
	"""The original encoding, and its rigged variant (0xFFFE)."""
	h = _Buf(data, len(data) - (23 if rigged else 18), name)
	nv, nf, nt = h.u16(), h.u16(), h.u8()
	textured, prio, has_alpha, face_groups, vgroups = h.u8(), h.u8(), h.u8(), h.u8(), h.u8()
	rig = h.u8() if rigged else 0
	lx, ly, lz, lf = h.u16(), h.u16(), h.u16(), h.u16()
	lg = h.u16() if rigged else (nv if vgroups == 1 else 0)
	o = nv
	o_types = o; o += nf
	o_prio = o; o += nf if prio == 255 else 0
	o_fgroups = o; o += nf if face_groups == 1 else 0
	o_tex = o; o += nf if textured == 1 else 0
	o_groups = o; o += lg
	o_alpha = o; o += nf if has_alpha == 1 else 0
	o_idx = o; o += lf
	o_col = o; o += nf * 2
	o_tt = o; o += nt * 6
	o_x = o; o += lx
	o_y = o; o += ly
	o_z = o; o += lz
	if o > len(data):
		raise ValueError(f"{name}: sections run past the end ({o} > {len(data)} bytes)")
	verts = _vertices(data, 0, (o_x, o_y, o_z), nv, name)
	groups = _groups(data, o_groups, nv, vgroups == 1, rigged and rig == 1, name)
	col = _Buf(data, o_col, name)
	colors = np.array([col.u16() for _ in range(nf)], np.int32)
	render = np.zeros(nf, np.int8)
	textures = np.full(nf, -1, np.int32)
	coords = np.full(nf, -1, np.int32)
	if textured == 1:
		tb = _Buf(data, o_tex, name)
		for i in range(nf):
			fl = tb.u8()
			render[i] = fl & 1
			if fl & 2:
				coords[i] = fl >> 2
				textures[i] = colors[i]
				colors[i] = 127
	alphas = np.frombuffer(data, np.uint8, nf, o_alpha).copy() if has_alpha == 1 else np.zeros(nf, np.uint8)
	faces = _faces(data, o_types, o_idx, nf, name)
	tt = _Buf(data, o_tt, name)
	tex_tris = np.array([(tt.u16(), tt.u16(), tt.u16()) for _ in range(nt)], np.int32).reshape(-1, 3)
	return RsModel(verts, faces, colors.astype(np.uint16), render, alphas, textures, coords, tex_tris, groups,
		"rigged (0xFFFE)" if rigged else "old", rigged and rig == 1)


def _read_new(data: bytes, name: str, rigged: bool) -> RsModel:
	"""The 0xFFFF encoding (texture types, 16-bit texture ids) and its rigged variant (0xFFFD)."""
	h = _Buf(data, len(data) - (26 if rigged else 23), name)
	nv, nf, nt = h.u16(), h.u16(), h.u8()
	flags, prio, has_alpha, face_groups, textured, vgroups = h.u8(), h.u8(), h.u8(), h.u8(), h.u8(), h.u8()
	rig = h.u8() if rigged else 0
	lx, ly, lz, lf, lc = h.u16(), h.u16(), h.u16(), h.u16(), h.u16()
	lg = h.u16() if rigged else (nv if vgroups == 1 else 0)
	tex_types = list(data[:nt])
	simple = sum(1 for t in tex_types if t == 0)
	o = nt
	o_vflags = o; o += nv
	o_render = o; o += nf if flags & 1 else 0
	o_types = o; o += nf
	o_prio = o; o += nf if prio == 255 else 0
	o_fgroups = o; o += nf if face_groups == 1 else 0
	o_groups = o; o += lg
	o_alpha = o; o += nf if has_alpha == 1 else 0
	o_idx = o; o += lf
	o_ftex = o; o += nf * 2 if textured == 1 else 0
	o_coords = o; o += lc
	o_col = o; o += nf * 2
	o_x = o; o += lx
	o_y = o; o += ly
	o_z = o; o += lz
	o_tt = o; o += simple * 6
	if o > len(data):
		raise ValueError(f"{name}: sections run past the end ({o} > {len(data)} bytes)")
	verts = _vertices(data, o_vflags, (o_x, o_y, o_z), nv, name)
	groups = _groups(data, o_groups, nv, vgroups == 1, rigged and rig == 1, name)
	col = _Buf(data, o_col, name)
	colors = np.array([col.u16() for _ in range(nf)], np.int32)
	render = np.frombuffer(data, np.int8, nf, o_render).copy() if flags & 1 else np.zeros(nf, np.int8)
	alphas = np.frombuffer(data, np.uint8, nf, o_alpha).copy() if has_alpha == 1 else np.zeros(nf, np.uint8)
	textures = np.full(nf, -1, np.int32)
	coords = np.full(nf, -1, np.int32)
	if textured == 1:
		ft, fc = _Buf(data, o_ftex, name), _Buf(data, o_coords, name)
		for i in range(nf):
			textures[i] = ft.u16() - 1
			if nt > 0 and textures[i] != -1:
				coords[i] = fc.u8() - 1
	faces = _faces(data, o_types, o_idx, nf, name)
	tex_tris = np.zeros((nt, 3), np.int32)
	tt = _Buf(data, o_tt, name)
	for i, t in enumerate(tex_types):
		if t == 0:
			tex_tris[i] = (tt.u16(), tt.u16(), tt.u16())
		else:	# cylinder/cube/sphere mapping (RS2 HD era): no plain triangle to map from
			coords[coords == i] = -2
	return RsModel(verts, faces, colors.astype(np.uint16), render.astype(np.int8), alphas, textures, coords,
		tex_tris, groups, "rigged (0xFFFD)" if rigged else "new (0xFFFF)", rigged and rig == 1)


def merge_models(parts: list[RsModel]) -> RsModel:
	"""Join several models (an NPC's body parts) into one, like the client does."""
	if len(parts) == 1:
		return parts[0]
	voff = np.cumsum([0] + [p.num_vertices for p in parts])
	toff = np.cumsum([0] + [len(p.tex_tris) for p in parts])
	coords = []
	for i, p in enumerate(parts):
		c = p.tex_coords.copy()
		c[c >= 0] += toff[i]
		coords.append(c)
	groups = None
	if any(p.groups is not None for p in parts):
		groups = np.concatenate([p.groups if p.groups is not None else np.zeros(p.num_vertices, np.int32) for p in parts])
	return RsModel(
		np.concatenate([p.vertices for p in parts]),
		np.concatenate([p.faces + voff[i] for i, p in enumerate(parts)]),
		np.concatenate([p.colors for p in parts]),
		np.concatenate([p.render_types for p in parts]),
		np.concatenate([p.alphas for p in parts]),
		np.concatenate([p.textures for p in parts]),
		np.concatenate(coords),
		np.concatenate([p.tex_tris + voff[i] for i, p in enumerate(parts)]).reshape(-1, 3).astype(np.int32),
		groups, "merged", any(p.rigged for p in parts))


def face_uvs(model: RsModel) -> "np.ndarray":
	"""(m, 3, 2) texture coordinates per face corner, from the model's texture triangles."""
	m = model.num_faces
	out = np.zeros((m, 3, 2), np.float64)
	out[:, 1] = (1.0, 0.0)
	out[:, 2] = (0.0, 1.0)
	v = model.vertices.astype(np.float64)
	idx = np.nonzero((model.textures >= 0) & (model.tex_coords >= 0) & (model.tex_coords < len(model.tex_tris)))[0]
	if len(idx) == 0:
		return out
	tri = model.tex_tris[model.tex_coords[idx]]
	if (tri >= len(v)).any() or (model.faces[idx] >= len(v)).any():
		return out
	p = v[tri[:, 0]]
	e1 = v[tri[:, 1]] - p
	e2 = v[tri[:, 2]] - p
	n = np.cross(e1, e2)
	pu = np.cross(e2, n)	# perpendicular to e2, in the plane: measures along e1
	pv = np.cross(e1, n)
	du = np.einsum("ij,ij->i", pu, e1)
	dv = np.einsum("ij,ij->i", pv, e2)
	ok = (np.abs(du) > 1e-9) & (np.abs(dv) > 1e-9)
	for corner in range(3):
		d = v[model.faces[idx, corner]] - p
		with np.errstate(divide="ignore", invalid="ignore"):
			u = np.einsum("ij,ij->i", pu, d) / du
			w = np.einsum("ij,ij->i", pv, d) / dv
		out[idx[ok], corner, 0] = u[ok]
		out[idx[ok], corner, 1] = w[ok]
	return out


def hsl_to_rgb(hsl: "np.ndarray", brightness: float = BRIGHTNESS) -> "np.ndarray":
	"""The client's 16-bit HSL (6 hue, 3 saturation, 7 lightness bits) → (n, 3) uint8 RGB."""
	hsl = np.asarray(hsl, np.int64)
	h = (hsl >> 10 & 63) / 64.0 + 0.5 / 64.0
	s = (hsl >> 7 & 7) / 8.0 + 0.5 / 8.0
	l = (hsl & 127) / 128.0
	c = (1.0 - np.abs(2.0 * l - 1.0)) * s
	x = c * (1.0 - np.abs((h * 6.0) % 2.0 - 1.0))
	m = l - c / 2.0
	sector = np.minimum((h * 6.0).astype(int), 5)
	zero = np.zeros_like(c)
	table = [(c, x, zero), (x, c, zero), (zero, c, x), (zero, x, c), (x, zero, c), (c, zero, x)]
	rgb = np.zeros((len(hsl), 3))
	for k, (r, g, b) in enumerate(table):
		sel = sector == k
		rgb[sel] = np.stack([r[sel], g[sel], b[sel]], 1)
	rgb = np.clip(rgb + m[:, None], 0.0, 1.0)
	rgb = np.power(rgb, brightness)	# the client's brightness curve
	return np.clip(rgb * 256.0, 0, 255).astype(np.uint8)


# -- sprites and textures -------------------------------------------------------


def read_sprite(data: bytes, name: str = "sprite") -> list["np.ndarray"]:
	"""A sprite group (index 8): its frames as RGBA, each placed on the group's canvas."""
	b = _Buf(data, len(data) - 2, name)
	count = b.u16()
	b.pos = len(data) - 7 - count * 8
	width, height, npal = b.u16(), b.u16(), b.u8() + 1
	ox = [b.u16() for _ in range(count)]
	oy = [b.u16() for _ in range(count)]
	ws = [b.u16() for _ in range(count)]
	hs = [b.u16() for _ in range(count)]
	b.pos = len(data) - 7 - count * 8 - (npal - 1) * 3
	pal = np.zeros((npal, 4), np.uint8)
	for i in range(1, npal):
		c = b.u24() or 1
		pal[i] = (c >> 16 & 255, c >> 8 & 255, c & 255, 255)
	b.pos = 0
	frames = []
	for i in range(count):
		w, h = ws[i], hs[i]
		flags = b.u8()
		b.need(w * h)
		idx = np.frombuffer(data, np.uint8, w * h, b.pos).copy()
		b.pos += w * h
		idx = idx.reshape(w, h).T if flags & 1 else idx.reshape(h, w)
		rgba = pal[idx]
		if flags & 2:
			b.need(w * h)
			a = np.frombuffer(data, np.uint8, w * h, b.pos).copy()
			b.pos += w * h
			a = a.reshape(w, h).T if flags & 1 else a.reshape(h, w)
			rgba[..., 3] = np.where(idx != 0, 255, a)
		canvas = np.zeros((max(height, oy[i] + h), max(width, ox[i] + w), 4), np.uint8)
		canvas[oy[i]:oy[i] + h, ox[i]:ox[i] + w] = rgba
		frames.append(canvas)
	return frames


def texture_sprite(data: bytes) -> int:
	"""The sprite a texture definition (index 9) draws from."""
	b = _Buf(data, name="texture")
	if len(data) == 7:	# 2023+ layout: sprite, average colour, flag, animation direction and speed
		return b.u16()
	b.skip(3)	# average colour, opaque flag
	count = b.u8()
	if count < 1:
		raise ValueError("texture has no sprite")
	return b.u16()


# -- definitions ------------------------------------------------------------------


@dataclass
class NpcDef:
	id: int
	name: str = ""
	models: list[int] = field(default_factory=list)
	size: int = 1
	stand: int = -1
	walk: int = -1
	recolor: dict[int, int] = field(default_factory=dict)
	retexture: dict[int, int] = field(default_factory=dict)
	width_scale: int = 128
	height_scale: int = 128
	run: int = -1
	partial: bool = False	# stopped at an opcode this reader does not know


def read_npc(data: bytes, npc: int = 0) -> NpcDef:
	"""An NPC definition (configs archive 9). Stops at the first opcode it can't size;
	everything GxBake uses comes before those."""
	d = NpcDef(npc)
	b = _Buf(data, name=f"npc {npc}")
	while b.pos < len(data):
		op = b.u8()
		if op == 0:
			return d
		if op in (1, 61):
			d.models = [b.u16() if op == 1 else b.i32() for _ in range(b.u8())]
		elif op == 2:
			d.name = b.string()
		elif op == 12:
			d.size = b.u8()
		elif op == 13:
			d.stand = b.u16()
		elif op == 14:
			d.walk = b.u16()
		elif op in (15, 16, 18, 42, 95, 103, 124, 126, 146) or 74 <= op <= 79:
			b.skip(2)
		elif op == 17:
			d.walk = b.u16()
			b.skip(6)
		elif 30 <= op < 35:
			b.string()
		elif op in (40, 41):
			pairs = {}
			for _ in range(b.u8()):
				k = b.u16()
				pairs[k] = b.u16()
			(d.recolor if op == 40 else d.retexture).update(pairs)
		elif op == 60:
			b.skip(2 * b.u8())
		elif op == 62:
			b.skip(4 * b.u8())
		elif op in (93, 99, 107, 109, 111, 122, 123, 129, 130, 145):
			pass
		elif op == 97:
			d.width_scale = b.u16()
		elif op == 98:
			d.height_scale = b.u16()
		elif op in (100, 101):
			b.skip(1)
		elif op == 114:
			d.run = b.u16()
		elif op in (115, 117):
			v = b.u16()
			d.run = v if op == 115 else d.run
			b.skip(6)
		elif op == 116:
			b.skip(2)
		else:	# 102 (head icons) and later opcodes changed size between revisions
			d.partial = True
			return d
	return d


@dataclass
class SeqDef:
	id: int
	frame_lengths: list[int]
	frame_ids: list[int]	# (frame archive << 16) | frame file
	loop: bool = True
	skeletal: bool = False


def read_sequence(data: bytes, seq: int = 0) -> SeqDef:
	"""A sequence (configs archive 12): only the frame list matters here."""
	b = _Buf(data, name=f"sequence {seq}")
	d = SeqDef(seq, [], [])
	while b.pos < len(data):
		op = b.u8()
		if op == 0:
			break
		if op == 1:
			n = b.u16()
			d.frame_lengths = [b.u16() for _ in range(n)]
			d.frame_ids = [b.u16() for _ in range(n)]
			for i in range(n):
				d.frame_ids[i] |= b.u16() << 16
		elif op in (2, 6, 7):
			b.skip(2)
		elif op == 3:
			b.skip(b.u8())
		elif op == 4:
			pass
		elif op in (5, 9, 10, 11):
			b.skip(1)
		elif op == 8:
			b.skip(1)
		elif op == 12:
			b.skip(4 * b.u8())
		else:	# 13+ are sounds and skeletal ("animaya") data, revision dependent
			if not d.frame_ids:
				d.skeletal = True
			break
	return d


@dataclass
class Framemap:
	types: list[int]
	labels: list[list[int]]


def read_framemap(data: bytes) -> Framemap:
	b = _Buf(data, name="framemap")
	n = b.u8()
	types = [b.u8() for _ in range(n)]
	counts = [b.u8() for _ in range(n)]
	return Framemap(types, [[b.u8() for _ in range(c)] for c in counts])


@dataclass
class Frame:
	framemap: int
	ops: list[tuple[int, int, int, int]]	# (framemap slot, dx, dy, dz)


def read_frame(data: bytes, fmap: Framemap, framemap_id: int | None = None) -> Frame:
	"""One animation frame (index 0). Slot types come from its framemap."""
	head = _Buf(data, name="frame")
	fm_id = head.u16()
	n = head.u8()
	vals = _Buf(data, 3 + n, "frame")
	ops = []
	last = -1
	for i in range(n):
		fl = head.u8()
		if fl == 0 or i >= len(fmap.types):
			continue
		if fmap.types[i] != 0:	# a transform needs its origin: the nearest origin slot before it
			for j in range(i - 1, last, -1):
				if fmap.types[j] == 0:
					ops.append((j, 0, 0, 0))
					break
		dflt = 128 if fmap.types[i] == 3 else 0
		dx = vals.smart() if fl & 1 else dflt
		dy = vals.smart() if fl & 2 else dflt
		dz = vals.smart() if fl & 4 else dflt
		ops.append((i, dx, dy, dz))
		last = i
	return Frame(fm_id, ops)


def frame_framemap_id(data: bytes) -> int:
	return _Buf(data, name="frame").u16()


# -- listing ------------------------------------------------------------------------


@dataclass
class RsEntry:
	key: str	# "npc:123" or "model:456"
	label: str


def parse_entry(key: str) -> tuple[str, int]:
	kind, _, num = str(key).partition(":")
	if kind not in ENTRY_KINDS or not num.lstrip("-").isdigit():
		raise ValueError(f"{key!r}: expected npc:<id> or model:<id>")
	return kind, int(num)


def entries(path: Path, wanted: list[str] | None = None) -> list[RsEntry]:
	"""What a cache offers: the given npc:/model: keys, or by default every named NPC
	(JS5) or every model (legacy, which has no readable NPC list here)."""
	cache = Cache(path)
	if wanted:
		out = []
		for key in wanted:
			kind, num = parse_entry(key)
			label = ""
			if kind == "npc" and not cache.legacy:
				raw = cache.file(IDX_CONFIGS, CFG_NPC, num)
				label = read_npc(raw, num).name if raw else ""
			out.append(RsEntry(f"{kind}:{num}", label))
		return out
	if cache.legacy:
		return [RsEntry(f"model:{m}", "") for m in cache.model_ids()]
	out = []
	for nid, raw in sorted(cache.files(IDX_CONFIGS, CFG_NPC).items()):
		try:
			d = read_npc(raw, nid)
		except ValueError:
			continue
		if d.models and d.name and d.name.lower() != "null":
			out.append(RsEntry(f"npc:{nid}", d.name))
	return out


def npcs(cache: Cache, find: str = "") -> list[NpcDef]:
	"""Every NPC definition whose name contains `find` (case-insensitive)."""
	if cache.legacy:
		return []
	out = []
	for nid, raw in sorted(cache.files(IDX_CONFIGS, CFG_NPC).items()):
		try:
			d = read_npc(raw, nid)
		except ValueError:
			continue
		if find.lower() in d.name.lower():
			out.append(d)
	return out


def describe(cache: Cache) -> dict:
	"""Summary of a cache for `gxbake rs`."""
	d: dict = {"path": str(cache.path), "layout": cache.kind, "indices": {}}
	for i in cache.indices():
		ref = cache.reference(i)
		d["indices"][i] = len(ref.archives) if ref is not None else len(cache.archive_ids(i))
	d["models"] = len(cache.model_ids())
	return d


def entry_stem(entry: RsEntry) -> str:
	kind, num = parse_entry(entry.key)
	label = "".join(ch if ch.isalnum() else "_" for ch in entry.label.lower()).strip("_")[:24]
	label = "_".join(p for p in label.split("_") if p)
	return f"rs_{kind}{num}" + (f"_{label}" if label and label.isascii() else "")


# -- import ---------------------------------------------------------------------------


@dataclass
class _Anim:
	name: str
	seq: SeqDef


def import_runescape(path: Path, opts: "_m.ModelOptions | None" = None, entry: str | None = None) -> "_m.ImportedModel":
	"""One NPC or model from a cache, as an ImportedModel ready for build_studio_mdl."""
	opts = opts or _m.ModelOptions()
	cache = Cache(path)
	warnings: list[str] = []
	if entry is None:
		found = entries(path)
		if not found:
			raise ValueError(f"{path.name}: no NPCs or models found")
		entry = found[0].key
	kind, num = parse_entry(entry)
	anims: list[_Anim] = []
	wscale = hscale = 128
	recolor: dict[int, int] = {}
	retexture: dict[int, int] = {}
	if kind == "npc":
		if cache.legacy:
			raise ValueError("NPC definitions are not read from legacy caches; use model:<id>")
		raw = cache.file(IDX_CONFIGS, CFG_NPC, num)
		if raw is None:
			raise ValueError(f"NPC {num} is not in this cache")
		npc = read_npc(raw, num)
		if not npc.models:
			raise ValueError(f"NPC {num} has no models")
		name = npc.name or f"npc{num}"
		ids = npc.models
		recolor, retexture = npc.recolor, npc.retexture
		wscale, hscale = npc.width_scale, npc.height_scale
		seen = set()
		for label, sid in (("idle", npc.stand), ("walk", npc.walk), ("run", npc.run)):
			if sid >= 0 and sid != 65535 and sid not in seen:
				seen.add(sid)
				sraw = cache.file(IDX_CONFIGS, CFG_SEQ, sid)
				if sraw is None:
					warnings.append(f"{label} sequence {sid} not found")
					continue
				anims.append(_Anim(label, read_sequence(sraw, sid)))
	else:
		name = f"model{num}"
		ids = [num]
	parts = []
	for mid in ids:
		data = cache.model_data(mid)
		if data is None:
			warnings.append(f"model {mid} not found")
			continue
		parts.append(read_model(data, f"model {mid}"))
	if not parts:
		raise ValueError(f"{name}: none of its models were found")
	model = merge_models(parts)
	if model.rigged:
		warnings.append("skeletal (rigged) weights are ignored; only label-group animation is baked")
	if recolor:
		for src, dst in recolor.items():
			model.colors[(model.colors == src) & (model.textures < 0)] = dst
	if retexture:
		for src, dst in retexture.items():
			model.textures[model.textures == src] = dst
	if wscale != 128 or hscale != 128:
		model.vertices = (model.vertices * np.array([wscale, hscale, wscale]) // 128).astype(np.int32)
	if opts.static:
		anims = []
	return _assemble(name, model, anims, cache, opts, warnings)


def _to_gold(opts: "_m.ModelOptions") -> "np.ndarray":
	"""3x3: RuneScape model axes → GoldSrc, scaled and turned."""
	base = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
	a = math.radians(opts.rs_yaw)
	c, s = math.cos(a), math.sin(a)
	yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
	return opts.rs_scale * (yaw @ base)


def _palette(colors: "np.ndarray") -> tuple["np.ndarray", dict[int, tuple[float, float]]]:
	"""A skin with one 4x4 block (one CMPR block) per colour, and the UV centre of each colour's block."""
	uniq = sorted(set(int(c) for c in colors))
	rgb = hsl_to_rgb(np.array(uniq)) if uniq else np.zeros((0, 3), np.uint8)
	if len(uniq) > 256:	# a 64x64 skin holds 256 blocks: merge near colours
		from PIL import Image

		img = Image.fromarray(rgb[None]).quantize(256, method=Image.Quantize.MEDIANCUT)
		pal = np.array(img.getpalette()[:768], np.uint8).reshape(-1, 3)
		rgb = pal[np.asarray(img)[0]]
	cells, cell_of = np.unique(rgb, axis=0, return_inverse=True)
	cell_of = np.asarray(cell_of).reshape(-1)
	per_row = max(1, math.ceil(math.sqrt(len(cells))))
	side = 8
	while side < per_row * 4:
		side *= 2
	per_row = side // 4
	skin = np.full((side, side, 4), (128, 128, 128, 255), np.uint8)
	for i, c in enumerate(cells):
		cx, cy = (i % per_row) * 4, (i // per_row) * 4
		skin[cy:cy + 4, cx:cx + 4, :3] = c
	uv = {}
	for c, i in zip(uniq, cell_of):
		uv[c] = (((i % per_row) * 4 + 2) / side, ((i // per_row) * 4 + 2) / side)
	return skin, uv


def _texture_rgba(cache: Cache, tex: int, warnings: list[str]) -> "np.ndarray | None":
	if cache.legacy:
		return None
	sprite = tex
	raw = cache.file(IDX_TEXTURES, 0, tex)
	if raw is not None:
		try:
			sprite = texture_sprite(raw)
		except ValueError:
			pass
	data = cache.file(IDX_SPRITES, sprite)
	if data is None:
		return None
	try:
		frames = read_sprite(data, f"sprite {sprite}")
	except ValueError as exc:
		warnings.append(str(exc))
		return None
	return frames[0] if frames else None


def _assemble(name: str, rs: RsModel, anims: list[_Anim], cache: Cache, opts: "_m.ModelOptions",
		warnings: list[str]) -> "_m.ImportedModel":
	model = _m.ImportedModel(name, [], [], warnings)
	g = _to_gold(opts)
	k = float(np.cbrt(abs(np.linalg.det(g)))) or 1.0
	rot = g / k
	world = rs.vertices.astype(np.float64) @ g.T

	hidden = (rs.render_types & 3) == 2
	hidden |= rs.alphas >= 254	# alpha -1 / -2: not drawn
	if hidden.any():
		warnings.append(f"{int(hidden.sum())} hidden faces dropped")
	clear = (rs.alphas > 0) & ~hidden
	if clear.any():
		warnings.append(f"{int(clear.sum())} translucent faces drawn opaque (studio models have no per-face alpha)")
	keep = ~hidden & (rs.faces < len(world)).all(1) & (rs.faces >= 0).all(1)
	faces = rs.faces[keep]
	colors = rs.colors[keep]
	textures = rs.textures[keep].copy()
	uv_all = face_uvs(rs)[keep]
	flat = (rs.render_types[keep] & 1) == 1

	# textures that can be found become skins; the rest fall back to a grey palette colour
	tex_skin: dict[int, np.ndarray] = {}
	for t in sorted(set(int(t) for t in textures if t >= 0)):
		rgba = _texture_rgba(cache, t, warnings)
		if rgba is None:
			warnings.append(f"texture {t} not found; its faces use their colour")
			textures[textures == t] = -1
		else:
			tex_skin[t] = rgba

	# normals: flat faces use the face normal, smooth faces the average of their smooth neighbours
	fn = np.cross(world[faces[:, 1]] - world[faces[:, 0]], world[faces[:, 2]] - world[faces[:, 0]])
	vn = np.zeros_like(world)
	smooth = ~flat
	for c in range(3):
		np.add.at(vn, faces[smooth, c], fn[smooth])

	# bones: one per animated label group
	groups = rs.groups if rs.groups is not None and anims else None
	bone_of_group: dict[int, int] = {}
	centres: dict[int, np.ndarray] = {}
	if groups is not None:
		used = sorted(set(int(x) for x in groups))
		if len(used) + 1 > _sk.MAXSTUDIOBONES:
			warnings.append(f"{len(used)} vertex groups exceeds the studio bone limit; baked static")
			groups, anims = None, []
		else:
			model.bones = [_m.Bone("root", -1, np.zeros(6))]
			for lab in used:
				sel = groups == lab
				centres[lab] = rs.vertices[sel].astype(np.float64).mean(0)
				bone_of_group[lab] = len(model.bones)
				pos = centres[lab] @ g.T
				model.bones.append(_m.Bone(f"group{lab}", 0, np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0])))

	def corners(sel: np.ndarray):
		f = faces[sel]
		pos = world[f].reshape(-1, 3)
		nrm = np.where(flat[sel][:, None, None], fn[sel][:, None, :], vn[f]).reshape(-1, 3)
		lens = np.linalg.norm(nrm, axis=1, keepdims=True)
		nrm = np.where(lens > 1e-9, nrm / np.maximum(lens, 1e-9), [0.0, 0.0, 1.0])
		if groups is not None:
			labs = groups[f.reshape(-1)]
			bones = np.array([bone_of_group[int(x)] for x in labs], np.int32)
			origin = np.array([centres[int(x)] for x in labs]) @ g.T
			local = pos - origin
		else:
			bones = np.zeros(len(pos), np.int32)
			local = pos
		tris = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
		return local.astype(np.float32), nrm.astype(np.float32), tris, bones, pos.astype(np.float32)

	plain = textures < 0
	if plain.any():
		skin, uv = _palette(colors[plain])
		local, nrm, tris, bones, pos = corners(plain)
		uvs = np.repeat(np.array([uv[int(c)] for c in colors[plain]], np.float32), 3, axis=0)
		model.materials.append(_m.Material("palette", skin, False, "face colours"))
		model.meshes.append(_m.Submesh(0, local, nrm, uvs, tris, bones, pos))
	for t, rgba in tex_skin.items():
		sel = textures == t
		masked = bool((rgba[..., 3] < 128).any())
		skin = _m._fit_skin(rgba, opts.skin_max_edge).copy()
		skin[..., 3] = np.where(skin[..., 3] < 128, 0, 255) if masked else 255
		local, nrm, tris, bones, pos = corners(sel)
		uvs = uv_all[sel].reshape(-1, 2).astype(np.float32)
		model.materials.append(_m.Material(f"texture{t}", skin, masked, f"texture {t}"))
		model.meshes.append(_m.Submesh(len(model.materials) - 1, local, nrm, uvs, tris, bones, pos))
	if not model.meshes:
		raise ValueError(f"{name}: no visible faces")

	if groups is not None:
		used_names: set[str] = set()
		for an in anims:
			frames = _bake_sequence(an.seq, cache, rs, centres, bone_of_group, g, rot, opts.anim_fps, warnings)
			if frames is None:
				continue
			model.sequences.append(_m.Sequence(_m._unique(_m._seq_name(an.name), used_names), float(opts.anim_fps),
				frames, loop=True))
	elif anims and rs.groups is None:
		warnings.append("model has no vertex groups, so its sequences cannot move it; baked static")
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _rot_rs(dx: int, dy: int, dz: int) -> "np.ndarray":
	"""The client's frame rotation: about z (dz), then x (dx), then y (dy); 2048 steps a turn."""
	def ang(v):
		return (v & 255) * 8 * 2.0 * math.pi / 2048.0
	cz, sz = math.cos(ang(dz)), math.sin(ang(dz))
	cx, sx = math.cos(ang(dx)), math.sin(ang(dx))
	cy, sy = math.cos(ang(dy)), math.sin(ang(dy))
	rz = np.array([[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]])
	rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
	ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
	return ry @ rx @ rz


def group_transforms(frame: Frame, fmap: Framemap, rs: RsModel, centres: dict[int, "np.ndarray"]) -> dict[int, "np.ndarray"]:
	"""Apply one frame to the model's label groups: a 4x4 (RuneScape space) per group."""
	counts = {lab: int((rs.groups == lab).sum()) for lab in centres}
	mats = {lab: np.eye(4) for lab in centres}
	origin = np.zeros(3)
	for slot, dx, dy, dz in frame.ops:
		kind = fmap.types[slot]
		labs = [lab for lab in fmap.labels[slot] if lab in mats]
		if kind == 0:
			n = sum(counts[lab] for lab in labs)
			if n:
				s = sum(counts[lab] * (mats[lab][:3, :3] @ centres[lab] + mats[lab][:3, 3]) for lab in labs)
				origin = s / n + (dx, dy, dz)
			else:
				origin = np.array((dx, dy, dz), np.float64)
			continue
		if kind == 1:
			t = np.eye(4)
			t[:3, 3] = (dx, dy, dz)
		elif kind in (2, 3):
			lin = _rot_rs(dx, dy, dz) if kind == 2 else np.diag([dx / 128.0, dy / 128.0, dz / 128.0])
			t = np.eye(4)
			t[:3, :3] = lin
			t[:3, 3] = origin - lin @ origin
		else:	# 5: face transparency, not representable
			continue
		for lab in labs:
			mats[lab] = t @ mats[lab]
	return mats


def _bake_sequence(seq: SeqDef, cache: Cache, rs: RsModel, centres: dict[int, "np.ndarray"],
		bone_of_group: dict[int, int], g: "np.ndarray", rot: "np.ndarray", fps: int,
		warnings: list[str]) -> "np.ndarray | None":
	if seq.skeletal or not seq.frame_ids:
		warnings.append(f"sequence {seq.id} is skeletal (animaya) or empty; skipped")
		return None
	maps: dict[int, Framemap] = {}
	poses_per_frame: list[np.ndarray] = []
	scaled = False
	nb = len(bone_of_group) + 1
	for fid in seq.frame_ids:
		arch, fil = fid >> 16, fid & 0xFFFF
		data = cache.file(IDX_FRAMES, arch, fil)
		if data is None:
			warnings.append(f"sequence {seq.id}: frame {arch}:{fil} not found; skipped")
			return None
		fm_id = frame_framemap_id(data)
		if fm_id not in maps:
			fm_raw = cache.file(IDX_FRAMEMAPS, fm_id)
			if fm_raw is None:
				warnings.append(f"sequence {seq.id}: framemap {fm_id} not found; skipped")
				return None
			maps[fm_id] = read_framemap(fm_raw)
		fmap = maps[fm_id]
		mats = group_transforms(read_frame(data, fmap), fmap, rs, centres)
		pose = np.zeros((nb, 6))
		for lab, m in mats.items():
			u, sv, vt = np.linalg.svd(m[:3, :3])
			if np.abs(sv - 1.0).max() > 0.01:
				scaled = True
			r = u @ vt
			if np.linalg.det(r) < 0:
				u[:, -1] *= -1
				r = u @ vt
			gm = np.eye(4)
			gm[:3, :3] = rot @ r @ rot.T
			gm[:3, 3] = (m[:3, :3] @ centres[lab] + m[:3, 3]) @ g.T
			pose[bone_of_group[lab]] = _sk.pose_from_matrix(gm)
		poses_per_frame.append(pose)
	if scaled:
		warnings.append(f"sequence {seq.id}: scale steps kept their pivot but not their scale (studio bones are rigid)")
	# resample: frame lengths are client ticks; hold each frame for its length
	times = np.cumsum([0] + [max(1, n) for n in seq.frame_lengths]) * TICK
	total = times[-1]
	count = max(1, int(round(total * fps)))
	out = np.zeros((count, nb, 6))
	for i in range(count):
		t = i / fps
		k = int(np.searchsorted(times, t, side="right") - 1)
		out[i] = poses_per_frame[min(max(k, 0), len(poses_per_frame) - 1)]
	return out
