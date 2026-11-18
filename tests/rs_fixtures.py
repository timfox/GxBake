"""Writers for synthetic RuneScape cache data (no game data in the repo).

They follow the layouts gxbake/core/runescape.py documents, so a test can build a
cache with a few models, NPCs and sequences and read it back.
"""

from __future__ import annotations

import bz2
import gzip
import struct
from pathlib import Path

import numpy as np

SECTOR = 520


def _smart(v: int) -> bytes:
	if -64 <= v < 64:
		return bytes([v + 64])
	assert -16384 <= v < 16384, v
	return struct.pack(">H", v + 0xC000)


def _u16(v: int) -> bytes:
	return struct.pack(">H", v & 0xFFFF)


# -- store ------------------------------------------------------------------------


def container(data: bytes, compression: int = 2, revision: int | None = None) -> bytes:
	if compression == 0:
		out = bytes([0]) + struct.pack(">i", len(data)) + data
	else:
		packed = bz2.compress(data)[4:] if compression == 1 else gzip.compress(data)
		out = bytes([compression]) + struct.pack(">ii", len(packed), len(data)) + packed
	return out + (_u16(revision) if revision is not None else b"")


def pack_archive(files: list[bytes], chunks: int = 1) -> bytes:
	"""Several files in one archive, each split over `chunks` chunks."""
	if len(files) == 1:
		return files[0]
	pieces = []
	for f in files:
		cut = [len(f) * c // chunks for c in range(chunks + 1)]
		pieces.append([f[cut[c]:cut[c + 1]] for c in range(chunks)])
	body = b"".join(pieces[f][c] for c in range(chunks) for f in range(len(files)))
	table = b""
	for c in range(chunks):
		prev = 0
		for f in range(len(files)):
			n = len(pieces[f][c])
			table += struct.pack(">i", n - prev)
			prev = n
	return body + table + bytes([chunks])


def reference_table(archives: dict[int, list[int]], protocol: int = 6, named: bool = False) -> bytes:
	ids = sorted(archives)
	count = (lambda v: struct.pack(">I", v | 0x80000000) if v >= 0x8000 else _u16(v)) if protocol >= 7 else _u16
	out = bytes([protocol]) + (struct.pack(">i", 1) if protocol >= 6 else b"") + bytes([1 if named else 0])
	out += count(len(ids))
	last = 0
	for a in ids:
		out += count(a - last)
		last = a
	if named:
		out += b"".join(struct.pack(">i", a * 31) for a in ids)
	out += b"".join(struct.pack(">i", 0) for _ in ids)	# crc
	out += b"".join(struct.pack(">i", 1) for _ in ids)	# revision
	out += b"".join(count(len(archives[a])) for a in ids)
	for a in ids:
		last = 0
		for f in archives[a]:
			out += count(f - last)
			last = f
	if named:
		out += b"".join(struct.pack(">i", f) for a in ids for f in archives[a])
	return out


class _Store:
	def __init__(self):
		self.dat = bytearray(SECTOR)	# sector 0 is unused
		self.idx: dict[int, dict[int, tuple[int, int]]] = {}

	def put(self, index: int, archive: int, data: bytes, stored_index: int) -> None:
		big = archive > 0xFFFF
		head, room = (10, 510) if big else (8, 512)
		start = len(self.dat) // SECTOR
		parts = [data[i:i + room] for i in range(0, len(data), room)] or [b""]
		for p, chunk in enumerate(parts):
			sector = start + p
			nxt = 0 if p == len(parts) - 1 else sector + 1
			if big:
				h = struct.pack(">IH", archive, p) + nxt.to_bytes(3, "big") + bytes([stored_index])
			else:
				h = struct.pack(">HH", archive, p) + nxt.to_bytes(3, "big") + bytes([stored_index])
			self.dat += (h + chunk).ljust(SECTOR, b"\0")
		self.idx.setdefault(index, {})[archive] = (len(data), start)

	def write(self, folder: Path, dat_name: str) -> Path:
		folder.mkdir(parents=True, exist_ok=True)
		(folder / dat_name).write_bytes(bytes(self.dat))
		for index, entries in self.idx.items():
			n = max(entries) + 1
			raw = bytearray(6 * n)
			for a, (size, sector) in entries.items():
				raw[a * 6:a * 6 + 6] = size.to_bytes(3, "big") + sector.to_bytes(3, "big")
			(folder / f"main_file_cache.idx{index}").write_bytes(bytes(raw))
		return folder / dat_name


def js5_cache(folder: Path, indices: dict[int, dict[int, dict[int, bytes]]], compression: int = 2,
		protocol: int | None = None) -> Path:
	"""main_file_cache.dat2 + idx files: {index: {archive: {file: bytes}}}. Reference tables
	use protocol 6, or 7 when an index has ids above 65535 (as real caches do)."""
	st = _Store()
	for index, archives in sorted(indices.items()):
		big = max(archives) > 0xFFFF or any(max(f) > 0xFFFF for f in archives.values())
		proto = protocol or (7 if big else 6)
		for archive, files in sorted(archives.items()):
			ids = sorted(files)
			st.put(index, archive, container(pack_archive([files[f] for f in ids], chunks=2 if len(ids) > 2 else 1),
				compression, revision=1), index)
		table = reference_table({a: sorted(f) for a, f in archives.items()}, proto, named=index == 2)
		st.put(255, index, container(table, 0), 255)
	return st.write(folder, "main_file_cache.dat2")


def legacy_cache(folder: Path, models: dict[int, bytes]) -> Path:
	"""main_file_cache.dat + idx0..idx4 with gzipped models in store 1."""
	st = _Store()
	for mid, data in sorted(models.items()):
		st.put(1, mid, gzip.compress(data), 2)	# the legacy layout stores index + 1
	path = st.write(folder, "main_file_cache.dat")
	for i in (0, 2, 3, 4):
		(folder / f"main_file_cache.idx{i}").write_bytes(b"")
	return path


# -- models ---------------------------------------------------------------------------


def _vertex_streams(verts) -> tuple[bytes, bytes, bytes, bytes]:
	flags, xs, ys, zs = bytearray(), bytearray(), bytearray(), bytearray()
	last = np.zeros(3, int)
	for v in np.asarray(verts, int):
		d = v - last
		fl = 0
		for axis, buf in enumerate((xs, ys, zs)):
			if d[axis]:
				fl |= 1 << axis
				buf += _smart(int(d[axis]))
		flags.append(fl)
		last = v
	return bytes(flags), bytes(xs), bytes(ys), bytes(zs)


def _face_streams(faces) -> tuple[bytes, bytes]:
	"""Uses all four face encodings where they fit."""
	types, data = bytearray(), bytearray()
	a = b = c = last = 0
	for fa, fb, fc in np.asarray(faces, int):
		if (fa, fb) == (a, c):
			types.append(2)
			b = c
		elif (fa, fb) == (c, b):
			types.append(3)
			a = c
		elif (fa, fb) == (b, a):
			types.append(4)
			a, b = b, a
		else:
			types.append(1)
			data += _smart(fa - last) + _smart(fb - fa) + _smart(fc - fb)
			a, b, c = fa, fb, fc
			last = c
			continue
		data += _smart(fc - last)
		c = last = fc
	return bytes(types), bytes(data)


def model_old(verts, faces, colors, groups=None, render=None, textures=None, tex_coords=None, tex_tris=(),
		alphas=None, rigged: bool = False) -> bytes:
	"""The original encoding; rigged=True writes the 0xFFFE variant with empty skeletal weights.
	textures/tex_coords: per face, -1 for none (the texture id replaces the colour)."""
	nv, nf, nt = len(verts), len(faces), len(tex_tris)
	flags, xs, ys, zs = _vertex_streams(verts)
	ftypes, fidx = _face_streams(faces)
	textured = render is not None or textures is not None
	tex_flags = bytearray()
	cols = list(int(c) for c in colors)
	if textured:
		for i in range(nf):
			fl = int(render[i]) & 1 if render is not None else 0
			if textures is not None and textures[i] >= 0:
				fl |= 2 | (max(0, int(tex_coords[i])) << 2 if tex_coords is not None else 0)
				cols[i] = int(textures[i])
			tex_flags.append(fl)
	groups_b = bytes(int(g) for g in groups) if groups is not None else b""
	if rigged:
		groups_b += bytes(nv)	# no skeletal weights per vertex
	body = flags + ftypes + bytes(tex_flags) + groups_b
	body += bytes(int(a) for a in alphas) if alphas is not None else b""
	body += fidx + b"".join(_u16(c) for c in cols)
	body += b"".join(_u16(int(v)) for t in tex_tris for v in t)
	body += xs + ys + zs
	head = _u16(nv) + _u16(nf) + bytes([nt, 1 if textured else 0, 0, 1 if alphas is not None else 0, 0,
		1 if groups is not None else 0])
	if rigged:
		head += bytes([1])
	head += _u16(len(xs)) + _u16(len(ys)) + _u16(len(zs)) + _u16(len(fidx))
	if rigged:
		head += _u16(len(groups_b)) + b"\xff\xfe"
	return bytes(body + head)


def model_new(verts, faces, colors, groups=None, render=None, textures=None, tex_coords=None, tex_tris=(),
		rigged: bool = False) -> bytes:
	"""The 0xFFFF encoding (rigged=True: 0xFFFD with empty skeletal weights). Texture
	triangles are all plain (type 0)."""
	nv, nf, nt = len(verts), len(faces), len(tex_tris)
	flags, xs, ys, zs = _vertex_streams(verts)
	ftypes, fidx = _face_streams(faces)
	groups_b = bytes(int(g) for g in groups) if groups is not None else b""
	if rigged:
		groups_b += bytes(nv)
	ftex, coords = b"", bytearray()
	if textures is not None:
		ftex = b"".join(_u16(int(t) + 1) for t in textures)
		if nt:
			for i, t in enumerate(textures):
				if t >= 0:
					coords.append(int(tex_coords[i]) + 1 if tex_coords is not None else 0)
	body = bytes(nt) + flags
	body += bytes(int(r) & 0xFF for r in render) if render is not None else b""
	body += ftypes + groups_b + fidx + ftex + bytes(coords)
	body += b"".join(_u16(int(c)) for c in colors) + xs + ys + zs
	body += b"".join(_u16(int(v)) for t in tex_tris for v in t)
	body += bytes([0])	# no particle/billboard trailer
	head = _u16(nv) + _u16(nf) + bytes([nt, 1 if render is not None else 0, 0, 0, 0,
		1 if textures is not None else 0, 1 if groups is not None else 0])
	if rigged:
		head += bytes([1])
	head += _u16(len(xs)) + _u16(len(ys)) + _u16(len(zs)) + _u16(len(fidx)) + _u16(len(coords))
	if rigged:
		head += _u16(len(groups_b)) + b"\xff\xfd"
	else:
		head += b"\xff\xff"
	return bytes(body + head)


def box(center, half):
	"""Box in RuneScape axes (y down) with faces counter-clockwise seen from outside."""
	c, h = np.asarray(center, int), np.asarray(half, int)
	corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h + c
	tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
		[2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
	return corners, tris


# -- definitions ------------------------------------------------------------------------


def npc(name: str, models: list[int], stand: int = -1, walk: int = -1, recolor=None, width: int = 128,
		height: int = 128, extra: bytes = b"") -> bytes:
	out = bytes([1, len(models)]) + b"".join(_u16(m) for m in models)
	out += bytes([2]) + name.encode() + b"\0"
	out += bytes([30]) + b"Talk-to\0"
	if stand >= 0:
		out += bytes([13]) + _u16(stand)
	if walk >= 0:
		out += bytes([14]) + _u16(walk)
	if recolor:
		out += bytes([40, len(recolor)]) + b"".join(_u16(k) + _u16(v) for k, v in recolor.items())
	out += bytes([95]) + _u16(12)
	if width != 128:
		out += bytes([97]) + _u16(width)
	if height != 128:
		out += bytes([98]) + _u16(height)
	return out + extra + b"\0"


def sequence(frames: list[tuple[int, int, int]]) -> bytes:
	"""frames: (length in ticks, frame archive, frame file)."""
	out = bytes([1]) + _u16(len(frames))
	out += b"".join(_u16(n) for n, _a, _f in frames)
	out += b"".join(_u16(f) for _n, _a, f in frames)
	out += b"".join(_u16(a) for _n, a, _f in frames)
	return out + bytes([5, 1]) + b"\0"


def framemap(slots: list[tuple[int, list[int]]]) -> bytes:
	"""slots: (type, labels): 0 origin, 1 translate, 2 rotate, 3 scale, 5 alpha."""
	return bytes([len(slots)]) + bytes(t for t, _ in slots) + bytes(len(l) for _, l in slots) \
		+ b"".join(bytes(l) for _, l in slots)


def frame(framemap_id: int, nslots: int, values: dict[int, tuple[int, int, int]]) -> bytes:
	"""values: slot → (dx, dy, dz); missing axes write nothing when equal to the slot's default 0."""
	flags, data = bytearray(), bytearray()
	for i in range(nslots):
		if i not in values:
			flags.append(0)
			continue
		fl = 0
		for axis, v in enumerate(values[i]):
			if v is not None:
				fl |= 1 << axis
				data += _smart(v)
		flags.append(fl)
	return _u16(framemap_id) + bytes([nslots]) + bytes(flags) + bytes(data)


def sprite(rgba: "np.ndarray") -> bytes:
	"""One frame, palette-indexed (colour 0 = transparent), stored row by row."""
	h, w = rgba.shape[:2]
	colors = []
	idx = np.zeros((h, w), np.uint8)
	for y in range(h):
		for x in range(w):
			if rgba[y, x, 3] < 128:
				continue
			c = tuple(int(v) for v in rgba[y, x, :3])
			if c not in colors:
				colors.append(c)
			idx[y, x] = colors.index(c) + 1
	out = bytes([0]) + idx.tobytes()
	out += b"".join(bytes(c) for c in colors)
	out += _u16(w) + _u16(h) + bytes([len(colors)]) + _u16(0) + _u16(0) + _u16(w) + _u16(h) + _u16(1)
	return out


def texture(sprite_id: int, new_layout: bool = True) -> bytes:
	if new_layout:
		return _u16(sprite_id) + _u16(0x1234) + bytes([1, 0, 0])
	return _u16(0x1234) + bytes([1, 1]) + _u16(sprite_id) + struct.pack(">i", 0) + bytes([0, 0])
