"""Lump extraction for maps and the GameCube port's baked map files.

Ported from the per-game tools in the Xash3D GameCube ports, which each read a
part of a map's lumps for themselves:

* ``generate-gc-bsp-meta.py`` (HL1): the ``.gcmeta`` lump index (HOT/COLD tiers,
  32-byte alignment, face batches by texture). :func:`gcmeta` writes the same
  file, byte for byte.
* ``stage_bsp_mapmodels.py`` (Day of Defeat) and ``dod_bsp_prop_models`` in
  ``build-gamecube-disc.py``: the studio models a map's entity lump names.
  :func:`entity_refs` sorts every reference (models, sprites, sounds, WADs, sky,
  linked maps) the same way, from the entity lump only.
* ``bake_worlddl_goldsrc.py`` (HL1 and Lost Coast copies, and the older Day of
  Defeat one) and Lost Coast's ``bake_worlddl.py``: the ``media/wd_<map>.gcd``
  world display-list bake and its ``_lf.gcd`` light-face companion.
  :func:`read_world_dl` and :func:`read_lightfaces` read every version of them.

The differences between the projects' world bakes, which the reader handles:

=========== ========= ============ ============ ========== =============================
writer      version   chunk header vertex       material   extras
=========== ========= ============ ============ ========== =============================
Day of Def. 1         8 bytes      8 bytes      8 bytes    vertex colour RGB565, no pages
HL1 / LC    2, flag 1 16 bytes     15 bytes     12 bytes   32² RGB565 lightmap pages, x1
Lost Coast  2, flag 0 16 bytes     15 bytes     12 bytes   pages (x2), SKY1 block, env
=========== ========= ============ ============ ========== =============================

(The GoldSrc baker in the Lost Coast and HL1 trees is the same file; HL1's copy
also writes ``_lf.gcd``.)
"""

from __future__ import annotations

import json
import re
import struct
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# -- BSP lump tables ---------------------------------------------------------------------------

GOLDSRC_LUMPS = (
	"entities", "planes", "textures", "vertexes", "visibility", "nodes",
	"texinfo", "faces", "lighting", "clipnodes", "leafs", "marksurfaces",
	"edges", "surfedges", "models",
)
# What the port's map loader touches every frame (generate-gc-bsp-meta.py).
HOT = frozenset({"planes", "visibility", "nodes", "texinfo", "faces", "leafs", "marksurfaces", "models"})
GOLDSRC_STRIDE = {
	"planes": 20, "vertexes": 12, "nodes": 24, "texinfo": 40, "faces": 20, "clipnodes": 8,
	"leafs": 28, "marksurfaces": 2, "edges": 4, "surfedges": 4, "models": 64,
}

SOURCE_LUMPS = (
	"entities", "planes", "texdata", "vertexes", "visibility", "nodes", "texinfo", "faces",
	"lighting", "occlusion", "leafs", "faceids", "edges", "surfedges", "models", "worldlights",
	"leaffaces", "leafbrushes", "brushes", "brushsides", "areas", "areaportals", "portals",
	"clusters", "portalverts", "clusterportals", "dispinfo", "originalfaces", "physdisp",
	"physcollide", "vertnormals", "vertnormalindices", "disp_lightmap_alphas", "disp_verts",
	"disp_lightmap_sample_positions", "game_lump", "leafwaterdata", "primitives", "primverts",
	"primindices", "pakfile", "clipportalverts", "cubemaps", "texdata_string_data",
	"texdata_string_table", "overlays", "leafmindisttowater", "face_macro_texture_info",
	"disp_tris", "physcollidesurface", "wateroverlays", "leaf_ambient_index_hdr",
	"leaf_ambient_index", "lighting_hdr", "worldlights_hdr", "leaf_ambient_lighting_hdr",
	"leaf_ambient_lighting", "xzippakfile", "faces_hdr", "map_flags", "overlay_fades",
	"overlay_system_levels", "physlevel", "disp_multiblend",
)
SOURCE_STRIDE = {
	"planes": 20, "texdata": 32, "vertexes": 12, "nodes": 32, "texinfo": 72, "faces": 56,
	"edges": 4, "surfedges": 4, "models": 48, "leaffaces": 2, "leafbrushes": 2, "brushes": 12,
	"brushsides": 8, "dispinfo": 176, "disp_verts": 20, "faces_hdr": 56, "originalfaces": 56,
	"overlays": 352, "cubemaps": 16, "texdata_string_table": 4,
}

TEX_SPECIAL = 1


@dataclass
class LumpInfo:
	index: int
	name: str
	offset: int
	size: int
	stride: int = 0	# record size, 0 for byte blobs
	tier: str = ""	# HOT / COLD (GoldSrc and Quake maps)
	version: int = 0	# Source lump version
	compressed: bool = False	# Source LZMA lump

	@property
	def count(self) -> int:
		return self.size // self.stride if self.stride else 0

	@property
	def align32(self) -> int:
		return self.offset & 31


@dataclass
class BspLumps:
	path: str
	format: str	# "goldsrc" (v30), "quake" (v29), "source" (VBSP)
	version: int
	size: int
	crc32: int
	lumps: list[LumpInfo]
	issues: list[str] = field(default_factory=list)
	data: bytes = b""
	_source: object = field(default=None, repr=False)

	def raw(self, index: int) -> bytes:
		"""A lump's bytes (Source LZMA lumps decompressed)."""
		if self.format == "source":
			return self.source().raw(index)
		lump = self.lumps[index]
		return self.data[lump.offset:lump.offset + lump.size]

	def source(self):
		"""The parsed Source map (cached)."""
		if self._source is None:
			from . import source_bsp

			self._source = source_bsp.parse(self.data, self.path)
		return self._source

	def by_name(self, name: str) -> LumpInfo:
		return next(lump for lump in self.lumps if lump.name == name)


def read_bsp(data: bytes, path: str = "") -> BspLumps:
	"""The lump directory of a GoldSrc (v30), Quake (v29) or Source (VBSP) map, with layout problems."""
	if data[:4] == b"VBSP":
		return _read_source(data, path)
	if len(data) < 4 + 15 * 8:
		raise ValueError(f"BSP is too small: {path or len(data)}")
	version = struct.unpack_from("<i", data, 0)[0]
	if version not in (29, 30):
		raise ValueError(f"unsupported BSP version {version} (GoldSrc 30, Quake 29 and Source VBSP are read)")
	lumps: list[LumpInfo] = []
	for i, name in enumerate(GOLDSRC_LUMPS):
		off, size = struct.unpack_from("<ii", data, 4 + i * 8)
		lumps.append(LumpInfo(i, name, off, size, GOLDSRC_STRIDE.get(name, 0), "HOT" if name in HOT else "COLD"))
	out = BspLumps(path, "goldsrc" if version == 30 else "quake", version, len(data),
		zlib.crc32(data) & 0xFFFFFFFF, lumps, data=data)
	out.issues = _layout_issues(lumps, len(data))
	return out


def _read_source(data: bytes, path: str) -> BspLumps:
	from . import source_bsp

	bsp = source_bsp.parse(data, path)
	lumps = []
	for i, lump in enumerate(bsp.lumps):
		name = SOURCE_LUMPS[i] if i < len(SOURCE_LUMPS) else f"lump{i}"
		compressed = (lump.length >= 4 and data[lump.offset:lump.offset + 4] == b"LZMA")
		lumps.append(LumpInfo(i, name, lump.offset, lump.length,
			0 if compressed else SOURCE_STRIDE.get(name, 0), "", lump.version, compressed))
	out = BspLumps(path, "source", bsp.version, len(data), zlib.crc32(data) & 0xFFFFFFFF, lumps, data=data,
		_source=bsp)
	out.issues = _layout_issues(lumps, len(data))
	return out


def _layout_issues(lumps: list[LumpInfo], file_size: int) -> list[str]:
	issues = []
	ranges = []
	for lump in lumps:
		if lump.offset < 0 or lump.size < 0 or lump.offset + lump.size > file_size:
			issues.append(f"invalid {lump.name} lump: offset {lump.offset}, size {lump.size}, file {file_size}")
			continue
		if lump.size:
			ranges.append((lump.offset, lump.offset + lump.size, lump.name))
		if lump.stride and lump.size % lump.stride:
			issues.append(f"{lump.name} lump is {lump.size} bytes, not a multiple of its {lump.stride}-byte record")
	ranges.sort()
	for prev, cur in zip(ranges, ranges[1:]):
		if cur[0] < prev[1]:
			issues.append(f"overlapping BSP lumps: {prev[2]} and {cur[2]}")
	return issues


# -- .gcmeta (generate-gc-bsp-meta.py) ---------------------------------------------------------


def face_batches(bsp: BspLumps) -> dict[tuple[int, int], list[int]]:
	"""GoldSrc/Quake faces grouped by (miptex, texinfo flags), the order the renderer submits them."""
	faces, texinfo = bsp.by_name("faces"), bsp.by_name("texinfo")
	if faces.size % 20 or texinfo.size % 40:
		raise ValueError(f"invalid face/texinfo stride in {bsp.path}")
	data = bsp.data
	tex_count = texinfo.size // 40
	batches: dict[tuple[int, int], list[int]] = {}
	for fi in range(faces.size // 20):
		ti = struct.unpack_from("<h", data, faces.offset + fi * 20 + 10)[0]
		if ti < 0 or ti >= tex_count:
			raise ValueError(f"invalid texinfo index {ti} in face {fi}")
		miptex, flags = struct.unpack_from("<ii", data, texinfo.offset + ti * 40 + 32)
		batches.setdefault((miptex, flags), []).append(fi)
	return batches


def gcmeta(bsp: BspLumps) -> str:
	"""The port's ``media/<map>.gcmeta`` lump index, exactly as generate-gc-bsp-meta.py writes it."""
	if bsp.format == "source":
		raise ValueError("a .gcmeta indexes GoldSrc lumps; convert the Source map to BSP30 first")
	if bsp.issues:
		raise ValueError(bsp.issues[0] + f" in {bsp.path}")
	rows = [f"GCMETA 1 {bsp.version} {bsp.crc32:08x} {bsp.size}\n"]
	for lump in bsp.lumps:
		rows.append(f"{lump.index:02d} {lump.name} {lump.tier} {lump.offset} {lump.size} {lump.align32}\n")
	batches = face_batches(bsp)
	rows.append(f"BATCHES {len(batches)} {sum(len(f) for f in batches.values())}\n")
	for (miptex, flags), faces in sorted(batches.items()):
		rows.append("BATCH %d %d %d %d %s\n" % (miptex, flags, len(faces), faces[0], ",".join(map(str, faces))))
	return "".join(rows)


@dataclass
class GcMeta:
	version: int
	bsp_version: int
	crc32: int
	size: int
	lumps: list[LumpInfo]
	batches: list[tuple[int, int, list[int]]]	# miptex, flags, faces


def read_gcmeta(text: str) -> GcMeta:
	lines = [ln.split() for ln in text.splitlines() if ln.strip()]
	if not lines or lines[0][0] != "GCMETA":
		raise ValueError("not a .gcmeta file")
	_tag, ver, bsp_ver, crc, size = lines[0][:5]
	meta = GcMeta(int(ver), int(bsp_ver), int(crc, 16), int(size), [], [])
	for row in lines[1:]:
		if row[0] == "BATCH":
			faces = [int(x) for x in row[5].split(",")] if len(row) > 5 else []
			meta.batches.append((int(row[1]), int(row[2]), faces))
		elif row[0] != "BATCHES":
			idx, name, tier, off, sz = row[:5]
			meta.lumps.append(LumpInfo(int(idx), name, int(off), int(sz), GOLDSRC_STRIDE.get(name, 0), tier))
	return meta


# -- textures and entities ---------------------------------------------------------------------


@dataclass
class TextureInfo:
	index: int
	name: str
	width: int
	height: int
	embedded: bool	# pixels in the BSP (else from a WAD)
	faces: int = 0


def textures(bsp: BspLumps) -> list[TextureInfo]:
	"""The map's textures, with how many faces use each (GoldSrc/Quake miptex, or Source texdata)."""
	if bsp.format == "source":
		from . import source_bsp

		src = bsp.source()
		names = src.texture_names()
		out = [TextureInfo(i, n, int(td["width"]), int(td["height"]), False) for i, (n, td) in
			enumerate(zip(names, src.texdata))]
		faces, _light, _hdr = src.faces_and_lighting()
		ti = src.texinfo
		for f in faces:
			t = int(f["texinfo"])
			if 0 <= t < len(ti):
				d = int(ti[t]["texdata"])
				if 0 <= d < len(out):
					out[d].faces += 1
		return out
	lump = bsp.by_name("textures")
	data, base = bsp.data, lump.offset
	if lump.size < 4:
		return []
	count = struct.unpack_from("<i", data, base)[0]
	out = []
	for i in range(max(0, count)):
		o = struct.unpack_from("<i", data, base + 4 + i * 4)[0]
		if o < 0 or base + o + 40 > base + lump.size:
			out.append(TextureInfo(i, "", 0, 0, False))
			continue
		raw, w, h, m0 = struct.unpack_from("<16sIII", data, base + o)
		out.append(TextureInfo(i, raw.split(b"\0")[0].decode("latin-1"), w, h, bool(m0)))
	for (miptex, _flags), faces in face_batches(bsp).items():
		if 0 <= miptex < len(out):
			out[miptex].faces += len(faces)
	return out


def entities(bsp: BspLumps) -> list[dict[str, str]]:
	from . import source_bsp

	text = bsp.raw(0).split(b"\0", 1)[0].decode("latin-1", "replace")
	return source_bsp.parse_entities(text)


_KEY_KINDS = (
	(".mdl", "models"), (".spr", "sprites"), (".wav", "sounds"), (".mp3", "sounds"),
	(".vmt", "materials"), (".tga", "images"), (".bmp", "images"),
)


def entity_refs(ents: list[dict[str, str]]) -> dict:
	"""Every file the entity lump names, sorted by kind (the DoD mapmodel scan, for all kinds).

	Only the entity lump is read: the rest of the BSP can hold bytes that look
	like ``"model" "models/..."`` and would stage junk (stage_bsp_mapmodels.py).
	"""
	refs: dict[str, set[str]] = {k: set() for k in ("models", "sprites", "sounds", "materials", "images")}
	brush_models: set[str] = set()
	wads: list[str] = []
	maps: set[str] = set()
	classnames: dict[str, int] = {}
	sky = ""
	for i, ent in enumerate(ents):
		cls = ent.get("classname", "")
		classnames[cls] = classnames.get(cls, 0) + 1
		if i == 0 or cls == "worldspawn":
			for w in re.split(r"[;]", ent.get("wad", "")):
				base = w.replace("\\", "/").rsplit("/", 1)[-1].strip()
				if base and base.lower() not in (x.lower() for x in wads):
					wads.append(base)
			sky = ent.get("skyname", sky) or sky
		if cls in ("trigger_changelevel", "info_landmark") and ent.get("map"):
			maps.add(ent["map"].lower())
		for key, value in ent.items():
			v = value.strip().replace("\\", "/")
			low = v.lower()
			if key == "model" and low.startswith("*"):
				brush_models.add(low)
				continue
			for ext, kind in _KEY_KINDS:
				if low.endswith(ext):
					rel = low.lstrip("/")
					if kind == "sounds" and not rel.startswith("sound/"):
						rel = "sound/" + rel.lstrip("!*#")
					refs[kind].add(rel)
					break
	out = {k: sorted(v) for k, v in refs.items()}
	out["brush_models"] = sorted(brush_models, key=lambda s: int(s[1:]) if s[1:].isdigit() else 1 << 30)
	out["wads"] = wads
	out["sky"] = sky.lower()
	out["maps"] = sorted(maps)
	out["classnames"] = dict(sorted(classnames.items(), key=lambda kv: (-kv[1], kv[0])))
	return out


def static_prop_models(bsp: BspLumps) -> list[str]:
	"""Source maps: the models the static prop game lump places (they are not entities)."""
	try:
		props = bsp.source().static_props()
	except (ValueError, IndexError, struct.error):
		return []
	return sorted({p.model.lower().replace("\\", "/") for p in props if getattr(p, "model", "")})


def fnv1a32(s: str) -> int:
	"""The material key the GX world bakes store (lower-cased name)."""
	h = 2166136261
	for ch in s.lower():
		h ^= ord(ch) & 0xFF
		h = (h * 16777619) & 0xFFFFFFFF
	return h


# -- media/wd_<map>.gcd (GCwd world display lists) ---------------------------------------------

WD_MAGIC = b"GCwd"
WD_FLAG_LIGHT_X1 = 0x0001
MAT_FLAGS = {0x01: "water", 0x02: "nolight", 0x04: "envmap", 0x08: "basealpha", 0x10: "envnomask"}
LM_PAGE_BYTES = 2048	# 32x32 RGB565, GX 4x4 tiles
GX_PRIMS = {0x80: "quads", 0x90: "triangles", 0x98: "tristrip", 0xA0: "trifan", 0xA8: "lines",
	0xB0: "linestrip", 0xB8: "points"}


@dataclass
class WdMaterial:
	index: int
	hash: int
	hash2: int
	tevmat: int
	flags: int
	extra: int	# Lost Coast: envmap strength / cubemap; 0 elsewhere
	name: str = ""

	@property
	def flag_names(self) -> list[str]:
		return [n for bit, n in MAT_FLAGS.items() if self.flags & bit]


@dataclass
class WdBatch:
	material: int
	env: int	# Lost Coast: octahedral normal code of a reflective batch
	offset: int	# from chunk start
	size: int
	prims: int = 0
	verts: int = 0
	tris: int = 0


@dataclass
class WdChunk:
	index: int
	offset: int	# from ofs_chunks
	size: int
	mins: tuple[int, int, int]
	maxs: tuple[int, int, int]
	cluster: int
	positions: int = 0
	lm_page: int = -1	# -1: lit by vertex colour
	batches: list[WdBatch] = field(default_factory=list)

	@property
	def tris(self) -> int:
		return sum(b.tris for b in self.batches)


@dataclass
class WorldDL:
	path: str
	version: int
	flags: int
	clusters: int
	nodes: int
	planes: int
	offsets: dict[str, int]
	file_size: int
	max_chunk: int
	materials: list[WdMaterial]
	chunks: list[WdChunk]
	pvs_rows: int
	lm_offset: int
	lm_pages: int
	sky: dict | None = None
	data: bytes = b""

	@property
	def vertex_stride(self) -> int:
		return 8 if self.version < 2 else 15

	@property
	def resident_bytes(self) -> int:
		"""Kept in MEM1 for the map: header, materials, tree and planes (everything before the PVS dir)."""
		return self.offsets["pvsdir"]

	@property
	def chunk_bytes(self) -> int:
		return sum(c.size for c in self.chunks)

	@property
	def light_scale(self) -> int:
		return 1 if self.flags & WD_FLAG_LIGHT_X1 else 2

	def sections(self) -> dict[str, tuple[int, int]]:
		"""(offset, size) of each part of the file, in file order."""
		o = self.offsets
		order = [("header", 0, 64)]
		if self.sky:
			order.append(("sky", 64, o["mats"] - 64))
		ends = [("mats", o["mats"], o["nodes"]), ("nodes", o["nodes"], o["planes"]),
			("planes", o["planes"], o["chunkdir"]), ("chunkdir", o["chunkdir"], o["pvsdir"]),
			("pvsdir", o["pvsdir"], o["pvs"]), ("pvs", o["pvs"], o["chunks"]),
			("chunks", o["chunks"], o["chunks"] + self.chunk_bytes)]
		order += [(n, a, b - a) for n, a, b in ends]
		if self.lm_pages:
			order.append(("lightmaps", self.lm_offset, self.lm_pages * LM_PAGE_BYTES))
		return {n: (a, s) for n, a, s in order}

	def lightmap_page(self, page: int) -> "np.ndarray":
		"""One 32x32 page as H×W×3 RGB."""
		start = self.lm_offset + page * LM_PAGE_BYTES
		return rgb565_tiled_decode(self.data[start:start + LM_PAGE_BYTES], 32, 32)


def rgb565_tiled_decode(blob: bytes, w: int, h: int) -> "np.ndarray":
	"""GX RGB565 (4x4 tiles, big-endian) to H×W×3."""
	px = np.frombuffer(blob, ">u2").reshape(h // 4, w // 4, 4, 4).transpose(0, 2, 1, 3).reshape(h, w)
	r = ((px >> 11) & 31).astype(np.uint16)
	g = ((px >> 5) & 63).astype(np.uint16)
	b = (px & 31).astype(np.uint16)
	return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.uint8)


def is_world_dl(head: bytes) -> bool:
	return head[:4] == WD_MAGIC


def read_world_dl(data: bytes, path: str = "") -> WorldDL:
	"""Any version of the ports' ``wd_<map>.gcd`` (see the module docstring for the differences)."""
	if data[:4] != WD_MAGIC or len(data) < 64:
		raise ValueError("not a GCwd world display-list file")
	_m, version, flags, n_cl, n_ch, n_mat, n_node, n_plane, o_mats, o_nodes, o_planes, o_cd, o_pd, o_pvs, \
		o_chunks, fsize = struct.unpack_from(">4sHH13I", data, 0)
	max_chunk = struct.unpack_from(">I", data, 60)[0]
	if version not in (1, 2):
		raise ValueError(f"unsupported GCwd version {version}")
	offs = {"mats": o_mats, "nodes": o_nodes, "planes": o_planes, "chunkdir": o_cd, "pvsdir": o_pd,
		"pvs": o_pvs, "chunks": o_chunks}
	if not (64 <= o_mats <= o_nodes <= o_planes <= o_cd <= o_pd <= o_pvs <= o_chunks <= len(data)):
		raise ValueError("GCwd section offsets are out of order")
	if n_ch * 24 > o_pd - o_cd or n_cl * 4 > o_pvs - o_pd:
		raise ValueError("GCwd chunk or PVS directory is larger than its section")
	sky = None
	if o_mats >= 96 and data[64:68] == b"SKY1":
		_s, ox, oy, oz, scale, cl, fr, fg, fb, fen, fs, fe = struct.unpack_from(">4s4fi3BBff", data, 64)
		sky = {"origin": [ox, oy, oz], "scale": scale, "cluster": cl, "fog": [fr, fg, fb],
			"fog_enabled": bool(fen), "fog_start": fs, "fog_end": fe}
	mats = []
	mstride = 8 if version < 2 else 12
	for i in range(n_mat):
		p = o_mats + i * mstride
		if version < 2:
			h, tid, fl, ex = struct.unpack_from(">IBBH", data, p)
			h2 = 0
		else:
			h, h2, tid, fl, ex = struct.unpack_from(">IIBBH", data, p)
		mats.append(WdMaterial(i, h, h2, tid, fl, ex))
	wd = WorldDL(path, version, flags, n_cl, n_node, n_plane, offs, fsize, max_chunk, mats, [], n_cl, 0, 0, sky, data)
	stride = wd.vertex_stride
	for i in range(n_ch):
		run, size, *box, cl, _pad = struct.unpack_from(">II3h3hHH", data, o_cd + i * 24)
		ch = WdChunk(i, run, size, tuple(box[:3]), tuple(box[3:]), cl)
		base = o_chunks + run
		if base + size > len(data):
			raise ValueError(f"chunk {i} runs past the end of the file")
		if version < 2:
			ch.positions, nb, _pos = struct.unpack_from(">HHI", data, base)
			first = 8
		else:
			ch.positions, nb, _pos, lm = struct.unpack_from(">HHII", data, base)
			ch.lm_page = lm - 1
			first = 16
		for k in range(nb):
			m, env, dofs, dlen = struct.unpack_from(">HHII", data, base + first + 12 * k)
			b = WdBatch(m, env, dofs, dlen)
			_count_dl(data[base + dofs:base + dofs + dlen], stride, b)
			ch.batches.append(b)
		wd.chunks.append(ch)
	end = o_chunks + wd.chunk_bytes
	wd.lm_offset = end + (-end & 31)
	if wd.lm_offset < len(data):
		wd.lm_pages = (len(data) - wd.lm_offset) // LM_PAGE_BYTES
	return wd


def _count_dl(dl: bytes, stride: int, batch: WdBatch) -> None:
	i, n = 0, len(dl)
	while i < n:
		op = dl[i]
		if op == 0:	# GX_NOP padding
			i += 1
			continue
		if i + 3 > n:
			break
		prim = op & 0xF8
		count = struct.unpack_from(">H", dl, i + 1)[0]
		i += 3 + count * stride
		batch.prims += 1
		batch.verts += count
		if prim in (0x98, 0xA0):
			batch.tris += max(0, count - 2)
		elif prim == 0x90:
			batch.tris += count // 3
		elif prim == 0x80:
			batch.tris += count // 4 * 2


def name_materials(wd: WorldDL, names) -> int:
	"""Fill in material names from candidate texture names (their FNV-1a hashes); returns how many matched."""
	table: dict[int, str] = {}
	for n in names:
		if not n:
			continue
		low = n.lower().replace("\\", "/")
		for cand in (low, low.removeprefix("materials/").removesuffix(".vmt")):
			table.setdefault(fnv1a32(cand), n)
	hit = 0
	for m in wd.materials:
		m.name = table.get(m.hash, "")
		hit += bool(m.name)
	return hit


def names_from(path: Path) -> list[str]:
	"""Texture names to resolve material hashes: a BSP, WAD, .gct pack, or a text file of names."""
	data = path.read_bytes()
	if data[:4] in (b"WAD2", b"WAD3"):
		from . import goldsrc

		return [e.name for e in goldsrc.wad_entries(data)]
	if data[:4] == b"GCWT":
		from . import gct

		return [e.name for e in gct.read_gct(data)]
	if path.suffix.lower() == ".bsp":
		return [t.name for t in textures(read_bsp(data, str(path)))]
	if path.suffix.lower() == ".json":	# Lost Coast texture-names.json: material → pack name
		obj = json.loads(data.decode("utf-8"))
		return [*obj.values(), *obj.keys()] if isinstance(obj, dict) else [str(x) for x in obj]
	return [ln.strip() for ln in data.decode("latin-1").splitlines() if ln.strip()]


# -- media/wd_<map>_lf.gcd (GClf light faces) --------------------------------------------------

LF_MAGIC = b"GClf"
LF_FLAG_SKY, LF_FLAG_TILED = 1, 2
LF_NO_LIGHT = 0xFFFFFF


@dataclass
class LightFaces:
	path: str
	version: int
	faces: int
	records: list[dict]
	lighting_offset: int
	lighting_size: int
	data: bytes = b""

	@property
	def lighting(self) -> bytes:
		return self.data[self.lighting_offset:self.lighting_offset + self.lighting_size]


def read_lightfaces(data: bytes, path: str = "") -> LightFaces:
	"""HL1's ``_lf.gcd``: per-face texinfo vecs, extents and light offset for the GX light point."""
	if data[:4] != LF_MAGIC:
		raise ValueError("not a GClf light-face file")
	_m, ver, n, rec_ofs, lit_ofs, lit_len = struct.unpack_from(">4s5I", data, 0)
	recs = []
	for i in range(n):
		v = struct.unpack_from(">8f4iI4s", data, rec_ofs + i * 56)
		packed = v[12]
		lo = packed & LF_NO_LIGHT
		recs.append({"face": i, "vecs": [list(v[0:4]), list(v[4:8])], "texturemins": list(v[8:10]),
			"extents": list(v[10:12]), "lightofs": -1 if lo == LF_NO_LIGHT else lo,
			"sky": bool((packed >> 24) & LF_FLAG_SKY), "tiled": bool((packed >> 24) & LF_FLAG_TILED),
			"styles": list(v[13])})
	return LightFaces(path, ver, n, recs, lit_ofs, lit_len, data)


# -- one entry point ---------------------------------------------------------------------------


def kind_of(path: Path) -> str:
	"""bsp / worlddl / lightfaces / gcmeta, or '' for anything else."""
	try:
		with path.open("rb") as f:
			head = f.read(8)
	except OSError:
		return ""
	if head[:4] == WD_MAGIC:
		return "worlddl"
	if head[:4] == LF_MAGIC:
		return "lightfaces"
	if head[:6] == b"GCMETA":
		return "gcmeta"
	if head[:4] == b"VBSP" or (len(head) >= 4 and struct.unpack_from("<i", head)[0] in (29, 30)
			and path.suffix.lower() == ".bsp"):
		return "bsp"
	return ""


EXTENSIONS = (".bsp", ".gcd", ".gcmeta")


def find_inputs(paths) -> list[Path]:
	out: list[Path] = []
	for p in map(Path, paths):
		if p.is_dir():
			out += sorted(q for q in p.rglob("*") if q.suffix.lower() in EXTENSIONS and kind_of(q))
		else:
			out.append(p)
	return out


def _auto_names(path: Path) -> Path | None:
	"""Where the names of ``media/wd_<map>.gcd``'s materials are: the map's world texture pack
	(``media/wt_<map>.gct``, which every port bakes with the same names), its BSP, or Lost
	Coast's ``texture-names.json``."""
	stem = path.stem
	if not stem.startswith("wd_"):
		return None
	stem = stem[3:].removesuffix("_lf")
	media, root = path.parent, path.parent.parent
	for cand in (media / f"wt_{stem}.gct", media / f"pt_{stem}.gct", media / f"{stem}.bsp",
			root / "maps" / f"{stem}.bsp", media / "maps" / f"{stem}.bsp", root / "texture-names.json",
			media / "texture-names.json"):
		if cand.is_file():
			return cand
	return None


def describe(path: Path, names: Path | None = None) -> dict:
	"""A JSON-ready summary of any supported file."""
	kind = kind_of(path)
	data = path.read_bytes()
	if kind == "bsp":
		bsp = read_bsp(data, str(path))
		out = {"file": str(path), "kind": "bsp", "format": bsp.format, "version": bsp.version,
			"size": bsp.size, "crc32": f"{bsp.crc32:08x}", "issues": bsp.issues,
			"lumps": [_lump_row(lump) for lump in bsp.lumps]}
		if bsp.format != "source":
			out["hot_bytes"] = sum(lump.size for lump in bsp.lumps if lump.tier == "HOT")
			out["cold_bytes"] = sum(lump.size for lump in bsp.lumps if lump.tier == "COLD")
		try:
			ents = entities(bsp)
			out["entities"] = len(ents)
			out["refs"] = entity_refs(ents)
			if bsp.format == "source":
				out["refs"]["models"] = sorted(set(out["refs"]["models"]) | set(static_prop_models(bsp)))
		except (ValueError, IndexError, struct.error) as exc:
			out["issues"].append(f"entities: {exc}")
		try:
			tex = textures(bsp)
			out["textures"] = [asdict(t) for t in sorted(tex, key=lambda t: (-t.faces, t.name))]
			if bsp.format != "source" and not bsp.issues:
				b = face_batches(bsp)
				out["batches"] = len(b)
				out["faces"] = sum(len(f) for f in b.values())
		except (ValueError, IndexError, struct.error) as exc:
			out["issues"].append(f"textures: {exc}")
		return out
	if kind == "worlddl":
		wd = read_world_dl(data, str(path))
		names = names or _auto_names(path)
		if names:
			name_materials(wd, names_from(names))
		return _wd_summary(wd, names)
	if kind == "lightfaces":
		lf = read_lightfaces(data, str(path))
		return {"file": str(path), "kind": "lightfaces", "version": lf.version, "faces": lf.faces,
			"lit_faces": sum(r["lightofs"] >= 0 for r in lf.records),
			"sky_faces": sum(r["sky"] for r in lf.records), "lighting_bytes": lf.lighting_size, "size": len(data)}
	if kind == "gcmeta":
		meta = read_gcmeta(data.decode("ascii"))
		out = {"file": str(path), "kind": "gcmeta", "version": meta.version, "bsp_version": meta.bsp_version,
			"bsp_crc32": f"{meta.crc32:08x}", "bsp_size": meta.size,
			"lumps": [_lump_row(lump) for lump in meta.lumps], "batches": len(meta.batches),
			"faces": sum(len(f) for *_x, f in meta.batches), "issues": []}
		# The port trusts the index only for the BSP it was made from: check the one next to it.
		for cand in (path.with_suffix(".bsp"), path.parent.parent / "maps" / f"{path.stem}.bsp",
				path.parent / "maps" / f"{path.stem}.bsp"):
			if cand.is_file():
				raw = cand.read_bytes()
				out["bsp"] = str(cand)
				if len(raw) != meta.size or (zlib.crc32(raw) & 0xFFFFFFFF) != meta.crc32:
					out["issues"].append(f"stale: {cand.name} is no longer the BSP this index was made from "
						"(re-run gxbake lumps -o, or the disc build)")
				break
		return out
	raise ValueError(f"{path}: not a BSP, GCwd (.gcd), GClf (_lf.gcd) or .gcmeta file")


def _lump_row(lump: LumpInfo) -> dict:
	row = {"index": lump.index, "name": lump.name, "offset": lump.offset, "size": lump.size,
		"align32": lump.align32}
	if lump.stride:
		row["count"] = lump.count
	if lump.tier:
		row["tier"] = lump.tier
	if lump.version:
		row["version"] = lump.version
	if lump.compressed:
		row["compressed"] = True
	return row


def _wd_summary(wd: WorldDL, names: Path | None = None) -> dict:
	per_mat: dict[int, dict] = {}
	for ch in wd.chunks:
		for b in ch.batches:
			s = per_mat.setdefault(b.material, {"batches": 0, "chunks": set(), "tris": 0, "bytes": 0})
			s["batches"] += 1
			s["chunks"].add(ch.index)
			s["tris"] += b.tris
			s["bytes"] += b.size
	mats = []
	for m in wd.materials:
		s = per_mat.get(m.index, {"batches": 0, "chunks": set(), "tris": 0, "bytes": 0})
		mats.append({"index": m.index, "hash": f"{m.hash:08x}", "name": m.name, "hash2": f"{m.hash2:08x}",
			"tevmat": m.tevmat, "flags": m.flag_names, "extra": m.extra, "batches": s["batches"],
			"chunks": len(s["chunks"]), "tris": s["tris"], "bytes": s["bytes"]})
	mats.sort(key=lambda r: (-r["bytes"], r["index"]))
	chunks = [{"index": c.index, "cluster": c.cluster, "bytes": c.size, "positions": c.positions,
		"batches": len(c.batches), "tris": c.tris, "lm_page": c.lm_page, "mins": list(c.mins), "maxs": list(c.maxs)}
		for c in sorted(wd.chunks, key=lambda c: (-c.size, c.index))]
	return {
		"file": wd.path, "kind": "worlddl", "version": wd.version, "flags": wd.flags,
		"writer": _wd_writer(wd), "light_scale": wd.light_scale, "size": len(wd.data),
		"clusters": wd.clusters, "nodes": wd.nodes, "planes": wd.planes, "chunks": len(wd.chunks),
		"materials": len(wd.materials), "named_materials": sum(bool(m.name) for m in wd.materials),
		"names_from": str(names) if names else "", "tris": sum(c.tris for c in wd.chunks),
		"batches": sum(len(c.batches) for c in wd.chunks), "vertex_stride": wd.vertex_stride,
		"resident_bytes": wd.resident_bytes, "chunk_bytes": wd.chunk_bytes, "max_chunk": wd.max_chunk,
		"lightmap_pages": wd.lm_pages, "pages_used": sum(c.lm_page >= 0 for c in wd.chunks),
		"sky": wd.sky, "sections": {k: {"offset": a, "size": s} for k, (a, s) in wd.sections().items()},
		"material_table": mats, "chunk_table": chunks,
	}


def _wd_writer(wd: WorldDL) -> str:
	if wd.version < 2:
		return "GoldSrc bake v1 (Day of Defeat tools/bake_worlddl_goldsrc.py)"
	if wd.flags & WD_FLAG_LIGHT_X1:
		return "GoldSrc bake v2 (HL1 / Lost Coast scripts/bake_worlddl_goldsrc.py)"
	return "Source bake v2 (Lost Coast tools/bake_worlddl.py)"


# -- extraction --------------------------------------------------------------------------------


def _write_json(path: Path, obj) -> Path:
	path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
	return path


def extract(path: Path, out_dir: Path, *, decode_textures: bool = False, names: Path | None = None,
		raw: bool = True) -> list[Path]:
	"""Write a file's lumps and sorted tables into ``out_dir`` (a folder per input). Returns what was written."""
	from PIL import Image

	kind = kind_of(path)
	out_dir.mkdir(parents=True, exist_ok=True)
	written: list[Path] = []
	summary = describe(path, names)
	written.append(_write_json(out_dir / "summary.json", summary))
	data = path.read_bytes()
	if kind == "bsp":
		bsp = read_bsp(data, str(path))
		if raw:
			ld = out_dir / "lumps"
			ld.mkdir(exist_ok=True)
			for lump in bsp.lumps:
				if lump.size and lump not in _bad_lumps(bsp):
					p = ld / f"{lump.index:02d}_{lump.name}.bin"
					p.write_bytes(bsp.raw(lump.index))
					written.append(p)
		if "refs" in summary:
			ents = entities(bsp)
			p = out_dir / "entities.txt"
			p.write_text(bsp.raw(0).split(b"\0", 1)[0].decode("latin-1", "replace"), encoding="latin-1")
			written += [p, _write_json(out_dir / "entities.json", ents), _write_json(out_dir / "refs.json", summary["refs"])]
		if "textures" in summary:
			written.append(_write_json(out_dir / "textures.json", summary["textures"]))
		if bsp.format != "source" and not bsp.issues:
			p = out_dir / f"{path.stem}.gcmeta"
			p.write_text(gcmeta(bsp), encoding="ascii")
			written.append(p)
			tex = textures(bsp)
			rows = [{"miptex": m, "texture": tex[m].name if 0 <= m < len(tex) else "", "flags": f,
				"special": bool(f & TEX_SPECIAL), "faces": faces}
				for (m, f), faces in sorted(face_batches(bsp).items())]
			written.append(_write_json(out_dir / "batches.json", rows))
		if bsp.format == "source":
			pak = bsp.raw(40)
			if pak:
				p = out_dir / "pakfile.zip"
				p.write_bytes(pak)
				written.append(p)
		if decode_textures and bsp.format != "source":
			written += _decode_miptex(bsp, out_dir / "textures")
	elif kind == "worlddl":
		wd = read_world_dl(data, str(path))
		if raw:
			sd = out_dir / "sections"
			sd.mkdir(exist_ok=True)
			for name, (a, s) in wd.sections().items():
				p = sd / f"{name}.bin"
				p.write_bytes(data[a:a + s])
				written.append(p)
		written.append(_write_json(out_dir / "materials.json", summary["material_table"]))
		written.append(_write_json(out_dir / "chunks.json", summary["chunk_table"]))
		if wd.lm_pages:
			ld = out_dir / "lightmaps"
			ld.mkdir(exist_ok=True)
			pages = [wd.lightmap_page(i) for i in range(wd.lm_pages)]
			for i, px in enumerate(pages):
				p = ld / f"page_{i:03d}.png"
				Image.fromarray(px, "RGB").save(p)
				written.append(p)
			cols = min(16, len(pages))
			rows_n = -(-len(pages) // cols)
			sheet = np.zeros((rows_n * 32, cols * 32, 3), np.uint8)
			for i, px in enumerate(pages):
				r, c = divmod(i, cols)
				sheet[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = px
			p = ld / "atlas.png"
			Image.fromarray(sheet, "RGB").save(p)
			written.append(p)
	elif kind == "lightfaces":
		lf = read_lightfaces(data, str(path))
		written.append(_write_json(out_dir / "faces.json", lf.records))
		if raw:
			p = out_dir / "lighting.bin"
			p.write_bytes(lf.lighting)
			written.append(p)
	elif kind == "gcmeta":
		meta = read_gcmeta(data.decode("ascii"))
		written.append(_write_json(out_dir / "batches.json",
			[{"miptex": m, "flags": f, "faces": faces} for m, f, faces in meta.batches]))
	return written


def _bad_lumps(bsp: BspLumps) -> list[LumpInfo]:
	return [lump for lump in bsp.lumps if lump.offset < 0 or lump.size < 0 or lump.offset + lump.size > bsp.size]


def _decode_miptex(bsp: BspLumps, out_dir: Path) -> list[Path]:
	from PIL import Image

	from . import goldsrc

	out_dir.mkdir(parents=True, exist_ok=True)
	written = []
	palette = None
	if bsp.format == "quake":
		from . import quake

		palette = quake.quake_palette(Path(bsp.path).parent) if bsp.path else None
		palette = palette or quake.GREY_PALETTE
	lump = bsp.by_name("textures")
	for t in textures(bsp):
		if not t.embedded or not t.name:
			continue
		o = struct.unpack_from("<i", bsp.data, lump.offset + 4 + t.index * 4)[0]
		mip = goldsrc.read_miptex(bsp.data, lump.offset + o, palette)
		if mip is None:
			continue
		name, w, h, pixels, pal = mip
		rgba = goldsrc.miptex_rgba(w, h, pixels, pal, name.startswith("{"))
		safe = re.sub(r"[^A-Za-z0-9_.+{!*~-]", "_", t.name) or f"tex{t.index}"
		p = out_dir / f"{safe}.png"
		Image.fromarray(rgba, "RGBA").save(p)
		written.append(p)
	return written


def format_report(info: dict, limit: int = 20) -> str:
	"""Human-readable text for :func:`describe`'s result."""
	from .audit import human_bytes as hb

	lines = [f"{info['file']}"]
	k = info["kind"]
	if k == "bsp":
		lines.append(f"  {info['format']} BSP v{info['version']}, {hb(info['size'])}, crc32 {info['crc32']}")
		if "hot_bytes" in info:
			lines.append(f"  hot lumps {hb(info['hot_bytes'])}, cold {hb(info['cold_bytes'])}")
		lines.append("  #  lump                          offset       size  count  align  tier")
		for r in info["lumps"]:
			if not r["size"] and info["format"] == "source":
				continue
			tag = r.get("tier", "") + (" lzma" if r.get("compressed") else "")
			lines.append(f"  {r['index']:02d} {r['name']:<26} {r['offset']:>9} {r['size']:>10} "
				f"{r.get('count', ''):>6} {r['align32']:>6}  {tag}")
		if "batches" in info:
			lines.append(f"  faces {info['faces']} in {info['batches']} batches (by texture and flags)")
		if "textures" in info:
			tex = info["textures"]
			emb = sum(t["embedded"] for t in tex)
			ext = "from WADs" if info["format"] != "source" else "materials"
			lines.append(f"  textures {len(tex)} ({emb} embedded, {len(tex) - emb} {ext}); most used:")
			for t in tex[:limit]:
				lines.append(f"    {t['name'] or '-':<24} {t['width']:>4}x{t['height']:<4} faces {t['faces']}"
					+ ("  embedded" if t["embedded"] else ""))
		if "refs" in info:
			r = info["refs"]
			lines.append(f"  entities {info['entities']}, brush models {len(r['brush_models'])}, "
				f"sky '{r['sky'] or '-'}', wads {', '.join(r['wads']) or '-'}")
			for kind in ("models", "sprites", "sounds", "materials", "images", "maps"):
				if r[kind]:
					shown = r[kind][:limit]
					lines.append(f"  {kind} ({len(r[kind])}): " + ", ".join(shown)
						+ (" …" if len(r[kind]) > limit else ""))
	elif k == "worlddl":
		light = f"lightmap pages at x{info['light_scale']}" if info["version"] >= 2 else "vertex-colour light"
		lines.append(f"  GCwd v{info['version']}: {info['writer']}; {light}; {hb(info['size'])}")
		lines.append(f"  clusters {info['clusters']}, chunks {info['chunks']}, batches {info['batches']}, "
			f"tris {info['tris']}, materials {info['materials']}"
			+ (f" ({info['named_materials']} named from {Path(info['names_from']).name})" if info["names_from"] else ""))
		lines.append(f"  MEM1 resident {hb(info['resident_bytes'])}, ARAM chunks {hb(info['chunk_bytes'])} "
			f"(largest {hb(info['max_chunk'])}), lightmap pages {info['lightmap_pages']} ({info['pages_used']} bound)")
		if info["sky"]:
			s = info["sky"]
			lines.append(f"  3D sky: origin {s['origin']}, scale {s['scale']}, cluster {s['cluster']}")
		lines.append("  materials by display-list bytes:")
		for m in info["material_table"][:limit]:
			lines.append(f"    {m['index']:>4} {m['name'] or m['hash']:<26} {hb(m['bytes']):>9}  tris {m['tris']:>6}  "
				f"chunks {m['chunks']:>4}  {','.join(m['flags'])}")
		lines.append("  largest chunks:")
		for c in info["chunk_table"][:limit]:
			lines.append(f"    cluster {c['cluster']:>5} {hb(c['bytes']):>9}  tris {c['tris']:>6}  "
				f"batches {c['batches']:>3}  page {c['lm_page'] if c['lm_page'] >= 0 else '-'}")
	elif k == "lightfaces":
		lines.append(f"  GClf v{info['version']}: {info['faces']} faces ({info['lit_faces']} lit, "
			f"{info['sky_faces']} sky), lighting {hb(info['lighting_bytes'])}")
	elif k == "gcmeta":
		lines.append(f"  GCMETA v{info['version']} for a {hb(info['bsp_size'])} BSP v{info['bsp_version']} "
			f"(crc32 {info['bsp_crc32']}): {info['faces']} faces in {info['batches']} batches")
		for r in info["lumps"]:
			lines.append(f"  {r['index']:02d} {r['name']:<14} {r.get('tier', ''):<4} {r['offset']:>9} {r['size']:>9}")
	for issue in info.get("issues", []):
		lines.append(f"  issue: {issue}")
	return "\n".join(lines)
