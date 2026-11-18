"""Source engine maps (VBSP v19-21, and the v25 lump layout where it matches): read the lumps the
GoldSrc conversion (source_map.py) needs.

Layouts follow the Source SDK's bspfile.h as documented on the Valve
Developer Community wiki. LZMA-compressed lumps (Left 4 Dead 2, CS:GO, TF2
after 2010 for some lumps) are decompressed with Python's lzma module.
"""

from __future__ import annotations

import io
import lzma
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER_LUMPS = 64

L_ENTITIES, L_PLANES, L_TEXDATA, L_VERTEXES, L_VISIBILITY, L_NODES, L_TEXINFO, L_FACES = range(8)
L_LIGHTING, L_LEAFS, L_EDGES, L_SURFEDGES, L_MODELS = 8, 10, 12, 13, 14
L_LEAFFACES, L_LEAFBRUSHES, L_BRUSHES, L_BRUSHSIDES = 16, 17, 18, 19
L_DISPINFO, L_DISP_VERTS, L_GAME_LUMP, L_PAKFILE, L_OVERLAYS = 26, 33, 35, 40, 45
L_TEXDATA_STRING_DATA, L_TEXDATA_STRING_TABLE = 43, 44
L_LIGHTING_HDR, L_FACES_HDR = 53, 58

# contents (bspflags.h)
CONTENTS_SOLID = 0x1
CONTENTS_WINDOW = 0x2
CONTENTS_GRATE = 0x8
CONTENTS_SLIME = 0x10
CONTENTS_WATER = 0x20
CONTENTS_MOVEABLE = 0x4000
CONTENTS_PLAYERCLIP = 0x10000
CONTENTS_MONSTERCLIP = 0x20000
CONTENTS_DETAIL = 0x8000000
MASK_PLAYERSOLID = CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_GRATE | CONTENTS_MOVEABLE | CONTENTS_PLAYERCLIP

# surface flags
SURF_LIGHT, SURF_SKY2D, SURF_SKY, SURF_WARP, SURF_TRANS = 0x1, 0x2, 0x4, 0x8, 0x10
SURF_TRIGGER, SURF_NODRAW, SURF_HINT, SURF_SKIP, SURF_NOLIGHT, SURF_BUMPLIGHT = 0x40, 0x80, 0x100, 0x200, 0x400, 0x800

PLANE = np.dtype([("normal", "<f4", 3), ("dist", "<f4"), ("type", "<i4")])
TEXDATA = np.dtype([("reflectivity", "<f4", 3), ("name", "<i4"), ("width", "<i4"), ("height", "<i4"),
	("view_width", "<i4"), ("view_height", "<i4")])
NODE = np.dtype([("planenum", "<i4"), ("children", "<i4", 2), ("mins", "<i2", 3), ("maxs", "<i2", 3),
	("firstface", "<u2"), ("numfaces", "<u2"), ("area", "<i2"), ("pad", "<i2")])
TEXINFO = np.dtype([("vecs", "<f4", (2, 4)), ("lmvecs", "<f4", (2, 4)), ("flags", "<i4"), ("texdata", "<i4")])
FACE = np.dtype([("planenum", "<u2"), ("side", "u1"), ("onnode", "u1"), ("firstedge", "<i4"), ("numedges", "<i2"),
	("texinfo", "<i2"), ("dispinfo", "<i2"), ("fogvolume", "<i2"), ("styles", "u1", 4), ("lightofs", "<i4"),
	("area", "<f4"), ("lm_mins", "<i4", 2), ("lm_size", "<i4", 2), ("origface", "<i4"), ("numprims", "<u2"),
	("firstprim", "<u2"), ("smoothing", "<u4")])
LEAF_V1 = np.dtype([("contents", "<i4"), ("cluster", "<i2"), ("areaflags", "<i2"), ("mins", "<i2", 3),
	("maxs", "<i2", 3), ("firstleafface", "<u2"), ("numleaffaces", "<u2"), ("firstleafbrush", "<u2"),
	("numleafbrushes", "<u2"), ("water", "<i2"), ("pad", "<i2")])
LEAF_V0 = np.dtype(LEAF_V1.descr[:-1] + [("ambient", "u1", 24), ("pad", "<i2")])
MODEL = np.dtype([("mins", "<f4", 3), ("maxs", "<f4", 3), ("origin", "<f4", 3), ("headnode", "<i4"),
	("firstface", "<i4"), ("numfaces", "<i4")])
BRUSH = np.dtype([("firstside", "<i4"), ("numsides", "<i4"), ("contents", "<i4")])
BRUSHSIDE = np.dtype([("planenum", "<u2"), ("texinfo", "<i2"), ("dispinfo", "<i2"), ("bevel", "<i2")])
DISPINFO = np.dtype([("start", "<f4", 3), ("firstvert", "<i4"), ("firsttri", "<i4"), ("power", "<i4"),
	("mintess", "<i4"), ("smoothing", "<f4"), ("contents", "<i4"), ("mapface", "<u2"), ("pad", "<u2"),
	("lm_alpha", "<i4"), ("lm_samples", "<i4"), ("rest", "u1", 128)])
DISPVERT = np.dtype([("vec", "<f4", 3), ("dist", "<f4"), ("alpha", "<f4")])

OVERLAY = np.dtype([("id", "<i4"), ("texinfo", "<i2"), ("facecount", "<u2"), ("faces", "<i4", 64), ("u", "<f4", 2),
	("v", "<f4", 2), ("points", "<f4", (4, 3)), ("origin", "<f4", 3), ("normal", "<f4", 3)])

# Strata Source (Momentum Mod, version 25) widens indices to 32 bits and bounds to floats
FACE_V25 = np.dtype([("planenum", "<u4"), ("side", "u1"), ("onnode", "u1"), ("pad0", "<u2"), ("firstedge", "<i4"),
	("numedges", "<i4"), ("texinfo", "<i4"), ("dispinfo", "<i4"), ("fogvolume", "<i4"), ("styles", "u1", 4),
	("lightofs", "<i4"), ("area", "<f4"), ("lm_mins", "<i4", 2), ("lm_size", "<i4", 2), ("origface", "<i4"),
	("numprims", "<u4"), ("firstprim", "<u4"), ("smoothing", "<u4")])
LEAF_V25 = np.dtype([("contents", "<i4"), ("cluster", "<i4"), ("areaflags", "<u4"), ("mins", "<f4", 3),
	("maxs", "<f4", 3), ("firstleafface", "<u4"), ("numleaffaces", "<u4"), ("firstleafbrush", "<u4"),
	("numleafbrushes", "<u4"), ("water", "<i4")])
NODE_V25 = np.dtype([("planenum", "<i4"), ("children", "<i4", 2), ("mins", "<f4", 3), ("maxs", "<f4", 3),
	("firstface", "<u4"), ("numfaces", "<u4"), ("area", "<i2"), ("pad", "<i2")])
BRUSHSIDE_V25 = np.dtype([("planenum", "<u4"), ("texinfo", "<i4"), ("dispinfo", "<i4"), ("bevel", "i1"),
	("thin", "i1"), ("pad", "<i2")])
DISPINFO_V25 = np.dtype([("start", "<f4", 3), ("firstvert", "<i4"), ("firsttri", "<i4"), ("power", "<i4"),
	("mintess", "<i4"), ("smoothing", "<f4"), ("contents", "<i4"), ("mapface", "<u4"), ("lm_alpha", "<i4"),
	("lm_samples", "<i4"), ("rest", "u1", 184)])


@dataclass
class Lump:
	offset: int
	length: int
	version: int
	fourcc: int


@dataclass
class StaticProp:
	model: str
	origin: tuple[float, float, float]
	angles: tuple[float, float, float]
	solid: int
	skin: int


@dataclass
class SourceBSP:
	version: int
	data: bytes
	lumps: list[Lump]
	path: str = ""
	_cache: dict = field(default_factory=dict, repr=False)

	def raw(self, index: int) -> bytes:
		if index in self._cache:
			return self._cache[index]
		lump = self.lumps[index]
		b = self.data[lump.offset:lump.offset + lump.length]
		if b[:4] == b"LZMA":
			b = _lzma(b)
		self._cache[index] = b
		return b

	def array(self, index: int, dtype: np.dtype) -> "np.ndarray":
		b = self.raw(index)
		n = len(b) // dtype.itemsize
		return np.frombuffer(b[:n * dtype.itemsize], dtype)

	# -- the lumps by name --------------------------------------------------------

	@property
	def planes(self):
		return self.array(L_PLANES, PLANE)

	@property
	def vertexes(self):
		return self.array(L_VERTEXES, np.dtype(("<f4", 3))).astype(np.float64)

	@property
	def edges(self):
		return self.array(L_EDGES, np.dtype(("<u4" if self.wide else "<u2", 2)))

	@property
	def surfedges(self):
		return self.array(L_SURFEDGES, np.dtype("<i4"))

	@property
	def texinfo(self):
		return self.array(L_TEXINFO, TEXINFO)

	@property
	def texdata(self):
		return self.array(L_TEXDATA, TEXDATA)

	@property
	def nodes(self):
		return self.array(L_NODES, NODE_V25 if self.wide else NODE)

	@property
	def leafs(self):
		if self.wide:
			return self.array(L_LEAFS, LEAF_V25)
		return self.array(L_LEAFS, LEAF_V0 if self.leaf_v0 else LEAF_V1)

	@property
	def wide(self) -> bool:
		"""Strata Source's version 25 layout."""
		return self.version >= 25

	@property
	def face_dtype(self) -> np.dtype:
		return FACE_V25 if self.wide else FACE

	@property
	def leaf_v0(self) -> bool:
		lump = self.lumps[L_LEAFS]
		return self.version < 20 and lump.version == 0

	@property
	def models(self):
		return self.array(L_MODELS, MODEL)

	@property
	def leaffaces(self):
		return self.array(L_LEAFFACES, np.dtype("<u4" if self.wide else "<u2"))

	@property
	def leafbrushes(self):
		return self.array(L_LEAFBRUSHES, np.dtype("<u4" if self.wide else "<u2"))

	@property
	def brushes(self):
		return self.array(L_BRUSHES, BRUSH)

	@property
	def brushsides(self):
		return self.array(L_BRUSHSIDES, BRUSHSIDE_V25 if self.wide else BRUSHSIDE)

	@property
	def dispinfo(self):
		return self.array(L_DISPINFO, DISPINFO_V25 if self.wide else DISPINFO)

	@property
	def dispverts(self):
		return self.array(L_DISP_VERTS, DISPVERT)

	@property
	def overlays(self):
		"""info_overlay decals (none read from version 25, whose layout differs)."""
		if self.wide:
			return np.zeros(0, OVERLAY)
		return self.array(L_OVERLAYS, OVERLAY)

	def faces_and_lighting(self) -> tuple["np.ndarray", bytes, bool]:
		"""Faces and the lightmap lump they index: LDR when present, else HDR. bool: HDR."""
		ldr = self.raw(L_LIGHTING)
		if ldr:
			return self.array(L_FACES, self.face_dtype), ldr, False
		hdr = self.raw(L_LIGHTING_HDR)
		fd = self.face_dtype
		faces = self.array(L_FACES_HDR, fd) if self.lumps[L_FACES_HDR].length else self.array(L_FACES, fd)
		return faces, hdr, True

	def texture_names(self) -> list[str]:
		strdata = self.raw(L_TEXDATA_STRING_DATA)
		table = self.array(L_TEXDATA_STRING_TABLE, np.dtype("<i4"))
		out = []
		for td in self.texdata:
			i = int(td["name"])
			off = int(table[i]) if 0 <= i < len(table) else -1
			if off < 0 or off >= len(strdata):
				out.append("")
				continue
			end = strdata.find(b"\0", off)
			out.append(strdata[off:end if end >= 0 else None].decode("latin-1"))
		return out

	def entities_text(self) -> str:
		return self.raw(L_ENTITIES).split(b"\0", 1)[0].decode("latin-1", "replace")

	def visibility(self) -> tuple[int, list[bytes]]:
		"""(numclusters, decompressed PVS row per cluster)."""
		vis = self.raw(L_VISIBILITY)
		if len(vis) < 4:
			return 0, []
		n = struct.unpack_from("<i", vis, 0)[0]
		row = (n + 7) // 8
		out = []
		for c in range(n):
			ofs = struct.unpack_from("<i", vis, 4 + 8 * c)[0]
			out.append(decompress_vis(vis, ofs, row))
		return n, out

	def pakfile(self) -> "zipfile.ZipFile | None":
		raw = self.raw(L_PAKFILE)
		if len(raw) < 22:
			return None
		try:
			return zipfile.ZipFile(io.BytesIO(raw))
		except zipfile.BadZipFile:
			return None

	def static_props(self) -> list[StaticProp]:
		"""prop_static entries of the 'sprp' game lump (versions 4-11)."""
		gl = self.lumps[L_GAME_LUMP]
		raw = self.raw(L_GAME_LUMP)
		if len(raw) < 4:
			return []
		count = struct.unpack_from("<i", raw, 0)[0]
		for i in range(count):
			gid, _flags, ver, fileofs, filelen = struct.unpack_from("<4sHHii", raw, 4 + 16 * i)
			if gid != b"prps":
				continue
			b = self.data[fileofs:fileofs + filelen] if gl.fourcc == 0 or fileofs >= len(raw) else raw
			if b[:4] == b"LZMA":
				b = _lzma(b)
			try:
				return _read_sprp(b, ver)
			except (struct.error, ValueError):
				return []
		return []


def _lzma(b: bytes) -> bytes:
	"""Valve's LZMA lump header: 'LZMA', actual size, compressed size, 5 property bytes."""
	actual, _size = struct.unpack_from("<II", b, 4)
	props = b[12:17]
	alone = props + struct.pack("<Q", actual) + b[17:]
	return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(alone)[:actual]


def _read_sprp(b: bytes, ver: int) -> list[StaticProp]:
	p = 0
	n = struct.unpack_from("<i", b, p)[0]
	p += 4
	names = [b[p + 128 * i:p + 128 * (i + 1)].split(b"\0")[0].decode("latin-1") for i in range(n)]
	p += 128 * n
	n = struct.unpack_from("<i", b, p)[0]
	p += 4 + 2 * n	# leaf entries
	n = struct.unpack_from("<i", b, p)[0]
	p += 4
	if n <= 0:
		return []
	size = (len(b) - p) // n
	out = []
	for i in range(n):
		q = p + i * size
		ox, oy, oz, ax, ay, az, ptype = struct.unpack_from("<6fH", b, q)
		solid = b[q + 30]	# after FirstLeaf, LeafCount
		skin = struct.unpack_from("<i", b, q + 32)[0]
		if ptype < len(names):
			out.append(StaticProp(names[ptype], (ox, oy, oz), (ax, ay, az), solid, skin))
	return out


def decompress_vis(vis: bytes, ofs: int, row: int) -> bytes:
	"""Quake-style zero-run RLE."""
	out = bytearray()
	p = ofs
	while len(out) < row and p < len(vis):
		c = vis[p]
		if c:
			out.append(c)
			p += 1
		else:
			if p + 1 >= len(vis):
				break
			out += bytes(vis[p + 1])
			p += 2
	out += bytes(max(0, row - len(out)))
	return bytes(out[:row])


def is_source_bsp(head: bytes) -> bool:
	return head[:4] == b"VBSP" and len(head) >= 8 and 17 <= struct.unpack_from("<i", head, 4)[0] <= 29


def parse(data: bytes, path: Path | str = "") -> SourceBSP:
	if not is_source_bsp(data[:8]):
		raise ValueError(f"{Path(str(path)).name or 'data'}: not a Source BSP")
	version = struct.unpack_from("<i", data, 4)[0]
	lumps = []
	for i in range(HEADER_LUMPS):
		off, length, ver, fourcc = struct.unpack_from("<iiii", data, 8 + 16 * i)
		if off < 0 or off + length > len(data):
			off, length = 0, 0
		lumps.append(Lump(off, length, ver, fourcc))
	return SourceBSP(version, data, lumps, str(path))


def load(path: Path) -> SourceBSP:
	path = Path(path)
	return parse(path.read_bytes(), path)


def parse_entities(text: str) -> list[dict[str, str]]:
	"""The entity lump as a list of key/value dicts (later duplicate keys win, as in the engine)."""
	out: list[dict[str, str]] = []
	cur: dict[str, str] | None = None
	i, n = 0, len(text)
	key = None
	while i < n:
		c = text[i]
		if c == "{":
			cur, key = {}, None
			i += 1
		elif c == "}":
			if cur is not None:
				out.append(cur)
			cur = None
			i += 1
		elif c == '"':
			j = text.find('"', i + 1)
			if j < 0:
				break
			s = text[i + 1:j]
			i = j + 1
			if cur is None:
				continue
			if key is None:
				key = s
			else:
				cur[key] = s
				key = None
		else:
			i += 1
	return out


def format_entities(ents: list[dict[str, str]]) -> str:
	parts = []
	for e in ents:
		parts.append("{\n" + "".join(f'"{k}" "{v}"\n' for k, v in e.items()) + "}\n")
	return "".join(parts)
