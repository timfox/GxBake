"""Writing GoldSrc BSP v30 maps, and building their collision (clip) hulls from convex brushes.

Used by the map converters (source_map.py) that turn other engines' maps into
what the GameCube port's Mod_LoadBrushModel reads: the 15 classic lumps,
embedded miptex, RGB lightmaps, per-leaf compressed PVS and clipnodes for
hulls 1-3.

The clip hulls are a brush-BSP of the union of the brushes, each expanded by
the hull's half-size (the Minkowski sum with the player box, exact when the
brush carries its axial and edge bevel planes, as Source brushes do).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

VERSION = 30
CONTENTS_EMPTY, CONTENTS_SOLID, CONTENTS_WATER, CONTENTS_SLIME, CONTENTS_LAVA, CONTENTS_SKY = -1, -2, -3, -4, -5, -6
TEX_SPECIAL = 1

# lump order
(L_ENTITIES, L_PLANES, L_TEXTURES, L_VERTEXES, L_VISIBILITY, L_NODES, L_TEXINFO, L_FACES, L_LIGHTING,
	L_CLIPNODES, L_LEAFS, L_MARKSURFACES, L_EDGES, L_SURFEDGES, L_MODELS) = range(15)

# GoldSrc hull half-sizes (hull 1 standing, 2 large, 3 crouching)
HULL_SIZES = {1: (16.0, 16.0, 36.0), 2: (32.0, 32.0, 32.0), 3: (16.0, 16.0, 18.0)}

MAX_MAP_NODES = 32767
MAX_MAP_CLIPNODES = 32767
MAX_MAP_LEAFS = 32767
MAX_MAP_FACES = 65535
MAX_MAP_VERTS = 65535
MAX_MAP_EDGES = 65535 * 4	# edge indices are int32 in surfedges, but faces count edges in u16 runs


# -- planes -------------------------------------------------------------------------


class PlaneTable:
	"""Deduplicated planes, axial ones always facing +axis (the engine's fast path assumes it)."""

	def __init__(self):
		self.normals: list[tuple[float, float, float]] = []
		self.dists: list[float] = []
		self.types: list[int] = []
		self._index: dict[tuple, int] = {}

	def __len__(self) -> int:
		return len(self.dists)

	@staticmethod
	def _type(n) -> int:
		ax = [abs(c) for c in n]
		for i in range(3):
			if ax[i] > 0.99999:
				return i
		return 3 + int(np.argmax(ax))

	def add(self, normal, dist: float) -> tuple[int, bool]:
		"""(index, flipped): flipped means the stored plane faces the other way."""
		n = np.asarray(normal, np.float64)
		ln = float(np.linalg.norm(n))
		if ln == 0:
			raise ValueError("degenerate plane")
		n = n / ln
		d = float(dist) / ln
		t = self._type(n)
		flipped = False
		if t < 3:
			n = np.zeros(3)
			n[t] = 1.0 if normal[t] > 0 else -1.0
			if n[t] < 0:
				n, d, flipped = -n, -d, True
		else:
			# canonical: the largest component positive
			if n[t - 3] < 0:
				n, d, flipped = -n, -d, True
		key = (round(n[0], 5), round(n[1], 5), round(n[2], 5), round(d, 3))
		i = self._index.get(key)
		if i is None:
			i = len(self.dists)
			self._index[key] = i
			self.normals.append((float(n[0]), float(n[1]), float(n[2])))
			self.dists.append(d)
			self.types.append(t)
		return i, flipped

	def normal(self, i: int) -> "np.ndarray":
		return np.array(self.normals[i])

	def pack(self) -> bytes:
		return b"".join(struct.pack("<4fi", *n, d, t) for n, d, t in zip(self.normals, self.dists, self.types))


# -- windings ------------------------------------------------------------------------


def base_winding(normal: "np.ndarray", dist: float, size: float = 131072.0) -> "np.ndarray":
	n = np.asarray(normal, np.float64)
	ax = int(np.argmax(np.abs(n)))
	up = np.array([0.0, 0.0, 1.0]) if ax != 2 else np.array([1.0, 0.0, 0.0])
	up = up - n * np.dot(up, n)
	up /= np.linalg.norm(up)
	right = np.cross(up, n)
	org = n * dist
	up *= size
	right *= size
	return np.array([org - right + up, org + right + up, org + right - up, org - right - up])


def clip_winding(w: "np.ndarray", normal, dist: float, keep_front: bool = False, eps: float = 0.01):
	"""Sutherland-Hodgman: the part of w behind (n·x <= d) the plane, or in front with keep_front."""
	if w is None or len(w) == 0:
		return None
	d = w @ np.asarray(normal) - dist
	if keep_front:
		d = -d
	if (d <= eps).all():
		return w
	if (d >= -eps).all():
		return None
	out = []
	n = len(w)
	for i in range(n):
		a, b = w[i], w[(i + 1) % n]
		da, db = d[i], d[(i + 1) % n]
		if da <= eps:
			out.append(a)
		if (da < -eps and db > eps) or (da > eps and db < -eps):
			t = da / (da - db)
			out.append(a + (b - a) * t)
	return np.array(out) if len(out) >= 3 else None


# -- clip hulls -----------------------------------------------------------------------


@dataclass
class _Side:
	plane: int
	flip: bool	# side faces -stored normal
	winding: "np.ndarray"
	boundary: bool = False


@dataclass
class _Frag:
	sides: list[_Side]

	def points(self) -> "np.ndarray":
		return np.concatenate([s.winding for s in self.sides])


def expanded_brush(sides: list[tuple["np.ndarray", float]], extent) -> list[tuple["np.ndarray", float]]:
	"""A convex brush (outward normals, n·x <= d inside) grown by a box of half-size extent."""
	e = np.asarray(extent, np.float64)
	pts = brush_points(sides)
	if pts is None:
		return []
	out = [(np.asarray(n, np.float64), float(d) + float(np.abs(n) @ e)) for n, d in sides]
	lo, hi = pts.min(0), pts.max(0)
	for ax in range(3):
		for sign in (1.0, -1.0):
			n = np.zeros(3)
			n[ax] = sign
			if not any(np.allclose(n, s[0], atol=1e-5) for s in sides):
				out.append((n, (hi[ax] if sign > 0 else -lo[ax]) + e[ax]))
	return out


def brush_windings(sides: list[tuple["np.ndarray", float]]) -> list[tuple[int, "np.ndarray"]]:
	out = []
	for i, (n, d) in enumerate(sides):
		w = base_winding(n, d)
		for j, (n2, d2) in enumerate(sides):
			if i == j:
				continue
			w = clip_winding(w, n2, d2, eps=0.001)
			if w is None:
				break
		if w is not None:
			out.append((i, w))
	return out


def brush_points(sides) -> "np.ndarray | None":
	ws = brush_windings(sides)
	if len(ws) < 4:
		return None
	return np.concatenate([w for _i, w in ws])


class _HullBuilder:
	def __init__(self, planes: PlaneTable):
		self.planes = planes
		self.nodes: list[list[int]] = []	# [plane, child0, child1]

	def frag(self, sides: list[tuple["np.ndarray", float]]) -> _Frag | None:
		ws = brush_windings(sides)
		if len(ws) < 4:
			return None
		out = []
		for i, w in ws:
			n, d = sides[i]
			p, flip = self.planes.add(n, d)
			out.append(_Side(p, flip, w))
		return _Frag(out)

	def _split(self, f: _Frag, p: int, eps: float = 0.01):
		"""(front part, back part) of a fragment against stored plane p."""
		n = self.planes.normal(p)
		d = self.planes.dists[p]
		pts = f.points()
		dist = pts @ n - d
		if (dist >= -eps).all():
			return self._mark(f, p), None
		if (dist <= eps).all():
			return None, self._mark(f, p)
		front, back = [], []
		for s in f.sides:
			if s.plane == p:
				continue
			wf = clip_winding(s.winding, n, d, keep_front=True)
			wb = clip_winding(s.winding, n, d)
			if wf is not None:
				front.append(_Side(s.plane, s.flip, wf, s.boundary))
			if wb is not None:
				back.append(_Side(s.plane, s.flip, wb, s.boundary))
		cap = base_winding(n, d)
		for s in f.sides:
			sn = self.planes.normal(s.plane) * (-1 if s.flip else 1)
			sd = self.planes.dists[s.plane] * (-1 if s.flip else 1)
			cap = clip_winding(cap, sn, sd, eps=0.001)
			if cap is None:
				break
		if cap is None or len(front) < 3 or len(back) < 3:
			# numerically a sliver: keep it whole on the side holding more of it
			return (self._mark(f, p), None) if dist.sum() > 0 else (None, self._mark(f, p))
		# the cap faces -n for the front piece and +n for the back piece
		front.append(_Side(p, True, cap[::-1].copy(), True))
		back.append(_Side(p, False, cap.copy(), True))
		return _Frag(front), _Frag(back)

	@staticmethod
	def _mark(f: _Frag, p: int) -> _Frag:
		if any(s.plane == p and not s.boundary for s in f.sides):
			return _Frag([_Side(s.plane, s.flip, s.winding, s.boundary or s.plane == p) for s in f.sides])
		return f

	def _choose(self, frags: list[_Frag]) -> int | None:
		cands: dict[int, int] = {}
		for f in frags:
			for s in f.sides:
				if not s.boundary:
					cands[s.plane] = cands.get(s.plane, 0) + 1
		if not cands:
			return None
		order = sorted(cands, key=lambda p: (self.planes.types[p] >= 3, -cands[p]))[:24]
		if len(frags) == 1:
			return order[0]
		best, best_score = None, None
		pts = [f.points() for f in frags]
		for p in order:
			n = self.planes.normal(p)
			d = self.planes.dists[p]
			front = back = split = 0
			for q in pts:
				dist = q @ n - d
				if (dist >= -0.01).all():
					front += 1
				elif (dist <= 0.01).all():
					back += 1
				else:
					split += 1
			score = split * 5 + abs(front - back) + (2 if self.planes.types[p] >= 3 else 0)
			if best_score is None or score < best_score:
				best, best_score = p, score
		return best

	def build(self, frags: list[_Frag], depth: int = 0) -> int:
		"""Returns a node index (>= 0) or a contents value (< 0)."""
		if not frags:
			return CONTENTS_EMPTY
		if any(all(s.boundary for s in f.sides) for f in frags):
			return CONTENTS_SOLID
		if depth > 400:
			return CONTENTS_SOLID
		p = self._choose(frags)
		if p is None:
			return CONTENTS_SOLID
		fronts, backs = [], []
		for f in frags:
			a, b = self._split(f, p)
			if a is not None:
				fronts.append(a)
			if b is not None:
				backs.append(b)
		index = len(self.nodes)
		self.nodes.append([p, 0, 0])
		self.nodes[index][1] = self.build(fronts, depth + 1)
		self.nodes[index][2] = self.build(backs, depth + 1)
		return index


def build_clip_hull(brushes: list[list[tuple["np.ndarray", float]]], hull: int, planes: PlaneTable,
		clipnodes: list[list[int]]) -> int:
	"""Append one hull's clipnodes (global indices) for these brushes and return its head node.

	brushes: convex volumes as (outward normal, dist) lists. Every head is a
	real clipnode, so an empty hull still has one node (both sides empty).
	"""
	b = _HullBuilder(planes)
	frags = []
	for sides in brushes:
		grown = expanded_brush(sides, HULL_SIZES.get(hull, (0.0, 0.0, 0.0)))
		if grown:
			f = b.frag(grown)
			if f is not None:
				frags.append(f)
	head = b.build(frags)
	base = len(clipnodes)
	if head < 0:
		p, _f = planes.add((0.0, 0.0, 1.0), 0.0)
		clipnodes.append([p, head, head])
		return base
	for plane, c0, c1 in b.nodes:
		clipnodes.append([plane, c0 + base if c0 >= 0 else c0, c1 + base if c1 >= 0 else c1])
	return base + head


def point_contents_many(clipnodes: list[list[int]], planes: PlaneTable, head: int, points) -> "np.ndarray":
	"""hull_contents for many points at once: (n, 3) → (n,) contents."""
	p = np.asarray(points, np.float64).reshape(-1, 3)
	out = np.full(len(p), CONTENTS_EMPTY, np.int64)
	if head < 0:
		out[:] = head
		return out
	cn = np.asarray(clipnodes, np.int64).reshape(-1, 3)
	normals = np.asarray(planes.normals, np.float64).reshape(-1, 3)
	dists = np.asarray(planes.dists, np.float64)
	node = np.full(len(p), head, np.int64)
	live = np.arange(len(p))
	while len(live):
		nd = node[live]
		pl = cn[nd, 0]
		d = np.einsum("ij,ij->i", p[live], normals[pl]) - dists[pl]
		nxt = np.where(d >= 0, cn[nd, 1], cn[nd, 2])
		node[live] = nxt
		done = nxt < 0
		out[live[done]] = nxt[done]
		live = live[~done]
	return out


def hull_contents(clipnodes: list[list[int]], planes: PlaneTable, head: int, point) -> int:
	"""Point contents in a clip hull, as SV_HullPointContents walks it."""
	p = np.asarray(point, np.float64)
	n = head
	while n >= 0:
		plane, c0, c1 = clipnodes[n]
		d = float(planes.normal(plane) @ p - planes.dists[plane])
		n = c0 if d >= 0 else c1
	return n


# -- visibility -------------------------------------------------------------------------


def compress_vis(row: bytes) -> bytes:
	out = bytearray()
	i, n = 0, len(row)
	while i < n:
		c = row[i]
		if c:
			out.append(c)
			i += 1
			continue
		run = 0
		while i < n and row[i] == 0 and run < 255:
			run += 1
			i += 1
		out += bytes((0, run))
	return bytes(out)


def merge_vis_rows(rows: list, max_bits: int) -> list:
	"""Replace uncompressed PVS rows that differ by at most ``max_bits`` set
	bits with their union, so more rows compress to the same bytes and share
	one copy. A union only adds visible leaves, so nothing can disappear: the
	cost is a few more leaves reaching the frustum test. ``None`` rows (no
	visibility) are kept. Greedy, in row order."""
	if max_bits <= 0:
		return list(rows)
	reps: list[int] = []
	owner: dict[int, int] = {}
	width = next((len(r) for r in rows if r is not None), 0)
	for i, r in enumerate(rows):
		if r is None:
			continue
		v = int.from_bytes(r, "little")
		for j, u in enumerate(reps):
			if bin(u ^ v).count("1") <= max_bits:
				reps[j] = u | v
				owner[i] = j
				break
		else:
			owner[i] = len(reps)
			reps.append(v)
	return [None if r is None else reps[owner[i]].to_bytes(width, "little") for i, r in enumerate(rows)]


# -- the file -------------------------------------------------------------------------------


@dataclass
class Miptex:
	name: str
	width: int
	height: int
	mips: list[bytes] | None = None	# 4 levels of 8-bit indices; None = external (WAD)
	palette: bytes = b""

	def pack(self) -> bytes:
		name = self.name.encode("latin-1", "replace")[:15]
		head = struct.pack("<16sII", name, self.width, self.height)
		if not self.mips:
			return head + struct.pack("<4I", 0, 0, 0, 0)
		offs, pos = [], 40
		for m in self.mips:
			offs.append(pos)
			pos += len(m)
		pal = self.palette[:768].ljust(768, b"\0")
		return head + struct.pack("<4I", *offs) + b"".join(self.mips) + struct.pack("<H", 256) + pal + b"\0\0"


@dataclass
class Bsp30:
	entities: str = ""
	planes: PlaneTable = field(default_factory=PlaneTable)
	textures: list[Miptex] = field(default_factory=list)
	vertexes: list[tuple[float, float, float]] = field(default_factory=list)
	visibility: bytes = b""
	nodes: list[tuple] = field(default_factory=list)	# (plane, c0, c1, mins3, maxs3, firstface, numfaces)
	texinfo: list[tuple] = field(default_factory=list)	# (vecs 2x4, miptex, flags)
	faces: list[tuple] = field(default_factory=list)	# (plane, side, firstedge, numedges, texinfo, styles4, lightofs)
	lighting: bytearray = field(default_factory=bytearray)
	clipnodes: list[list[int]] = field(default_factory=list)
	leafs: list[tuple] = field(default_factory=list)	# (contents, visofs, mins3, maxs3, firstmark, nummark, ambient4)
	marksurfaces: list[int] = field(default_factory=list)
	edges: list[tuple[int, int]] = field(default_factory=lambda: [(0, 0)])
	surfedges: list[int] = field(default_factory=list)
	models: list[tuple] = field(default_factory=list)	# (mins3, maxs3, origin3, heads4, visleafs, firstface, numfaces)

	def check(self) -> list[str]:
		"""Limits of the BSP30 format (index widths), not of any one engine build."""
		out = []
		for name, n, cap in (("nodes", len(self.nodes), MAX_MAP_NODES), ("clipnodes", len(self.clipnodes), MAX_MAP_CLIPNODES),
				("leafs", len(self.leafs), MAX_MAP_LEAFS), ("faces", len(self.faces), MAX_MAP_FACES),
				("vertexes", len(self.vertexes), MAX_MAP_VERTS), ("edges", len(self.edges), 65535),
				("texinfo", len(self.texinfo), 32767), ("marksurfaces", len(self.marksurfaces), 65535)):
			if n > cap:
				out.append(f"{name}: {n} exceeds the BSP30 limit of {cap}")
		return out

	def write(self) -> bytes:
		problems = self.check()
		if problems:
			raise ValueError("; ".join(problems))
		lumps: list[bytes] = [b""] * 15
		lumps[0] = self.entities.encode("latin-1", "replace") + b"\0"
		lumps[1] = self.planes.pack()
		mips = [t.pack() for t in self.textures]
		offs, pos = [], 4 + 4 * len(mips)
		for m in mips:
			offs.append(pos)
			pos += len(m)
			pos += -pos % 4
		tex = bytearray(struct.pack("<i", len(mips)) + struct.pack(f"<{len(mips)}i", *offs))
		for m in mips:
			tex += m
			tex += bytes(-len(tex) % 4)
		lumps[2] = bytes(tex)
		lumps[3] = np.asarray(self.vertexes, "<f4").tobytes() if self.vertexes else b""
		lumps[4] = self.visibility
		lumps[5] = b"".join(struct.pack("<ihh3h3hHH", p, c0, c1, *mn, *mx, ff, nf)
			for p, c0, c1, mn, mx, ff, nf in self.nodes)
		lumps[6] = b"".join(struct.pack("<8fii", *np.asarray(v, np.float64).ravel(), m, f) for v, m, f in self.texinfo)
		lumps[7] = b"".join(struct.pack("<HHiHH4Bi", p, s, fe, ne, ti, *st, lo) for p, s, fe, ne, ti, st, lo in self.faces)
		lumps[8] = bytes(self.lighting)
		lumps[9] = b"".join(struct.pack("<ihh", p, c0, c1) for p, c0, c1 in self.clipnodes)
		lumps[10] = b"".join(struct.pack("<ii3h3hHH4B", c, v, *mn, *mx, fm, nm, *amb)
			for c, v, mn, mx, fm, nm, amb in self.leafs)
		lumps[11] = struct.pack(f"<{len(self.marksurfaces)}H", *self.marksurfaces)
		lumps[12] = b"".join(struct.pack("<HH", a, b) for a, b in self.edges)
		lumps[13] = struct.pack(f"<{len(self.surfedges)}i", *self.surfedges)
		lumps[14] = b"".join(struct.pack("<9f4iiii", *mn, *mx, *org, *heads, vl, ff, nf)
			for mn, mx, org, heads, vl, ff, nf in self.models)
		out = bytearray(struct.pack("<i", VERSION) + bytes(15 * 8))
		for i, b in enumerate(lumps):
			out += bytes(-len(out) % 4)
			struct.pack_into("<ii", out, 4 + 8 * i, len(out), len(b))
			out += b
		return bytes(out)


# -- reading back (tests and the inspector) ----------------------------------------------


def read_lumps(data: bytes) -> list[bytes]:
	if struct.unpack_from("<i", data, 0)[0] != VERSION:
		raise ValueError("not a BSP30 file")
	out = []
	for i in range(15):
		off, ln = struct.unpack_from("<ii", data, 4 + 8 * i)
		out.append(data[off:off + ln])
	return out
