"""Rectangle packing for lightmap atlases.

A GoldSrc map lights every face with its own small grid (one luxel per 16
units, so 2×2 up to about 17×17), and the port copies them into fixed-size
RGB565 atlas pages. How many pages a map needs depends on how well those
rectangles are packed, which is a 2D bin packing problem: NP-hard in general,
but good heuristics land within a few percent of the area bound.

Three packers:

* ``quake``: Quake's ``AllocBlock`` (below) with faces in map order, which is
  what GoldSrc-derived engines do.
* ``shelf``: rows of rectangles sorted by height; a new row when one is full.
  What a quick port loader does.
* ``skyline``: keeps the top edge of everything placed (one height per
  column, like Quake's ``AllocBlock``) and puts each rectangle where it
  rests lowest, then leftmost. Sorted by height first, which is most of
  the gain: tall items first leave a flat skyline for the short ones.

(MaxRects with Best Short Side Fit was tried too. On GoldSrc-shaped inputs,
many rectangles of 2–17 luxels, it packed worse than a sorted skyline and
ran 300× slower, so it is not here.)

All three return the same thing, so a loader can use whichever it likes.
``lower_bound`` is ceil(total area / page area) pages: no packer can beat it.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field


@dataclass
class Placement:
	item: int	# index into the input list
	page: int
	x: int
	y: int
	w: int
	h: int


@dataclass
class Packing:
	method: str
	page_w: int
	page_h: int
	placements: list[Placement] = field(default_factory=list)
	pages: int = 0

	@property
	def used(self) -> int:
		return sum(p.w * p.h for p in self.placements)

	@property
	def occupancy(self) -> float:
		return self.used / max(1, self.pages * self.page_w * self.page_h)


def lower_bound(sizes: list[tuple[int, int]], page_w: int, page_h: int) -> int:
	return math.ceil(sum(w * h for w, h in sizes) / (page_w * page_h)) if sizes else 0


def _check(sizes, page_w, page_h):
	for w, h in sizes:
		if w > page_w or h > page_h or w <= 0 or h <= 0:
			raise ValueError(f"{w}x{h} does not fit a {page_w}x{page_h} page")


# --- shelf -------------------------------------------------------------------------

def pack_shelf(sizes: list[tuple[int, int]], page_w: int, page_h: int) -> Packing:
	_check(sizes, page_w, page_h)
	out = Packing("shelf", page_w, page_h)
	order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
	page, x, y, row_h = 0, 0, 0, 0
	for i in order:
		w, h = sizes[i]
		if x + w > page_w:
			x, y, row_h = 0, y + row_h, 0
		if y + h > page_h:
			page, x, y, row_h = page + 1, 0, 0, 0
		out.placements.append(Placement(i, page, x, y, w, h))
		x += w
		row_h = max(row_h, h)
	out.pages = page + 1 if sizes else 0
	return out


# --- skyline -----------------------------------------------------------------------

def pack_skyline(sizes: list[tuple[int, int]], page_w: int, page_h: int, sort: bool = True) -> Packing:
	"""Bottom-left on a per-column height map, the way Quake's AllocBlock
	allocates lightmaps, but sorted tallest first and trying every page.
	With sort=False it is AllocBlock as the engines have it: faces in map
	order, the same search."""
	_check(sizes, page_w, page_h)
	out = Packing("skyline" if sort else "quake", page_w, page_h)
	heights: list[list[int]] = []
	order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0])) if sort else range(len(sizes))
	for i in order:
		w, h = sizes[i]
		placed = False
		for p, col in enumerate(heights):
			best_y, best_x = page_h, -1
			for x in range(page_w - w + 1):
				y = max(col[x:x + w])
				if y < best_y:
					best_y, best_x = y, x
			if best_x >= 0 and best_y + h <= page_h:
				for k in range(best_x, best_x + w):
					col[k] = best_y + h
				out.placements.append(Placement(i, p, best_x, best_y, w, h))
				placed = True
				break
		if not placed:
			col = [0] * page_w
			for k in range(w):
				col[k] = h
			heights.append(col)
			out.placements.append(Placement(i, len(heights) - 1, 0, 0, w, h))
	out.pages = len(heights)
	return out


PACKERS = {"quake": None, "shelf": pack_shelf, "skyline": pack_skyline}


def pack(sizes: list[tuple[int, int]], page_w: int, page_h: int, method: str = "skyline", pad: int = 0) -> Packing:
	"""Pack with `pad` texels of border on the right and bottom of each item
	(bilinear filtering needs one when neighbours differ)."""
	padded = [(w + pad, h + pad) for w, h in sizes]
	res = pack_skyline(padded, page_w, page_h, sort=False) if method == "quake" else PACKERS[method](padded, page_w, page_h)
	for p in res.placements:
		p.w -= pad
		p.h -= pad
	return res


def overlaps(res: Packing) -> bool:
	by_page: dict[int, list[Placement]] = {}
	for p in res.placements:
		by_page.setdefault(p.page, []).append(p)
	for items in by_page.values():
		for a in range(len(items)):
			for b in range(a + 1, len(items)):
				p, q = items[a], items[b]
				if p.x < q.x + q.w and q.x < p.x + p.w and p.y < q.y + q.h and q.y < p.y + p.h:
					return True
	return False


# --- lightmap rectangles from a GoldSrc BSP ----------------------------------------

def goldsrc_lightmap_sizes(data: bytes, luxel: int = 16) -> list[tuple[int, int]]:
	"""Luxel grid (w, h) of every lit face in a v30 BSP."""
	return [(w, h) for _f, w, h, _o in goldsrc_lightmaps(data, luxel)]


def goldsrc_lightmaps(data: bytes, luxel: int = 16) -> list[tuple[int, int, int, int]]:
	"""(face, w, h, lightofs) of every lit face in a v30 BSP: extents of the
	face's texture coordinates, floor/ceil to `luxel` units, plus one (the
	engine's CalcSurfaceExtents)."""
	def lump(i):
		off, ln = struct.unpack_from("<ii", data, 4 + i * 8)
		return data[off:off + ln]

	if struct.unpack_from("<i", data)[0] not in (29, 30):
		raise ValueError("not a GoldSrc BSP")
	verts = lump(3)
	texinfo = lump(6)
	faces = lump(7)
	edges = lump(12)
	surfedges = lump(13)
	out = []
	for k in range(len(faces) // 20):
		_plane, _side, first, num, ti, s0, _s1, _s2, _s3, lightofs = struct.unpack_from("<HHiHH4Bi", faces, k * 20)
		if lightofs < 0 or s0 == 255 or num < 3:
			continue
		sv = struct.unpack_from("<8f", texinfo, ti * 40)
		flags = struct.unpack_from("<i", texinfo, ti * 40 + 36)[0]
		if flags & 1:	# TEX_SPECIAL (sky, water): no lightmap
			continue
		mins, maxs = [math.inf, math.inf], [-math.inf, -math.inf]
		for e in range(first, first + num):
			se = struct.unpack_from("<i", surfedges, e * 4)[0]
			a, b = struct.unpack_from("<HH", edges, abs(se) * 4)
			v = a if se >= 0 else b
			x, y, z = struct.unpack_from("<3f", verts, v * 12)
			for j in range(2):
				t = x * sv[j * 4] + y * sv[j * 4 + 1] + z * sv[j * 4 + 2] + sv[j * 4 + 3]
				mins[j], maxs[j] = min(mins[j], t), max(maxs[j], t)
		w = math.ceil(maxs[0] / luxel) - math.floor(mins[0] / luxel) + 1
		h = math.ceil(maxs[1] / luxel) - math.floor(mins[1] / luxel) + 1
		out.append((k, w, h, lightofs))
	return out


@dataclass
class LightmapPlan:
	faces: int = 0
	luxels: int = 0	# style 0, as stored
	flat: int = 0	# faces whose whole lightmap is one colour within tolerance
	half: int = 0	# faces stored at half resolution
	luxels_after: int = 0
	pages_before: int = 0
	pages_after: int = 0
	worst_db: float = float("inf")
	sizes_after: list[tuple[int, int]] = field(default_factory=list)
	face_ids: list[int] = field(default_factory=list)
	sizes_before: list[tuple[int, int]] = field(default_factory=list)
	layout: "Packing | None" = None	# placement of sizes_after with the chosen packer

	def to_json(self) -> dict:
		"""What a world baker needs: per face its new grid and where it sits."""
		at = {p.item: p for p in self.layout.placements} if self.layout else {}
		return {"page": self.layout.page_w if self.layout else 0, "pages": self.pages_after,
			"packer": self.layout.method if self.layout else "", "faces": [
			{"face": f, "w": b[0], "h": b[1], "new_w": a[0], "new_h": a[1],
				**({"page": at[i].page, "x": at[i].x, "y": at[i].y} if i in at else {})}
			for i, (f, b, a) in enumerate(zip(self.face_ids, self.sizes_before, self.sizes_after))]}


def _psnr(a, b) -> float:
	import numpy as np

	d = np.mean((a.astype(np.float64) - b) ** 2)
	return float("inf") if d == 0 else 10 * math.log10(255 ** 2 / d)


def _upsample(small, w: int, h: int):
	"""Bilinear back to w×h with the corners pinned, the way a half-res
	lightmap would be sampled over the same face."""
	import numpy as np

	sh, sw = small.shape[:2]
	ys = np.linspace(0, sh - 1, h)
	xs = np.linspace(0, sw - 1, w)
	y0 = np.floor(ys).astype(int)
	x0 = np.floor(xs).astype(int)
	y1, x1 = np.minimum(y0 + 1, sh - 1), np.minimum(x0 + 1, sw - 1)
	fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
	a = small[y0][:, x0] * (1 - fx) + small[y0][:, x1] * fx
	b = small[y1][:, x0] * (1 - fx) + small[y1][:, x1] * fx
	return a * (1 - fy) + b * fy


def lightmap_plan(data: bytes, page: int = 128, min_db: float = 40.0, flat_tol: int = 2,
		method: str = "skyline", pad: int = 1, resize: bool = True) -> LightmapPlan:
	"""Pick a resolution per face by error, the way detail mip trim does for
	textures: a face whose luxels all sit within `flat_tol` of their mean is
	stored as 1×1; a face whose half-resolution grid (same span, corners
	kept) interpolates back within `min_db` PSNR is stored at half size; the
	rest stay. The port bakes lightmap UVs per vertex, so a face's grid size
	costs nothing at runtime. Then pack both sets: before with Quake's
	allocator in map order, after with the sorted skyline."""
	import numpy as np

	off, ln = struct.unpack_from("<ii", data, 4 + 8 * 8)
	light = np.frombuffer(data, np.uint8, ln, off)
	plan = LightmapPlan()
	before, after = [], []
	for face, w, h, lo in goldsrc_lightmaps(data):
		n = w * h * 3
		if lo + n > ln:
			continue
		lm = light[lo:lo + n].reshape(h, w, 3).astype(np.float64)
		plan.faces += 1
		plan.luxels += w * h
		before.append((w, h))
		plan.face_ids.append(face)
		if not resize:
			after.append((w, h))
			continue
		if np.abs(lm - lm.mean((0, 1))).max() <= flat_tol:
			plan.flat += 1
			after.append((1, 1))
			continue
		sw, sh = max(2, -(-w // 2)), max(2, -(-h // 2))
		if (sw, sh) != (w, h):
			small = _upsample(lm, sw, sh)	# point-sampled at the same span, corners kept
			db = _psnr(_upsample(np.round(small), w, h), lm)
			if db >= min_db:
				plan.half += 1
				plan.worst_db = min(plan.worst_db, db)
				after.append((sw, sh))
				continue
		after.append((w, h))
	plan.luxels_after = sum(w * h for w, h in after)
	plan.sizes_after = after
	plan.sizes_before = before
	if before:
		plan.pages_before = pack(before, page, page, "quake", pad=pad).pages
		plan.layout = pack(after, page, page, method, pad=pad)
		plan.pages_after = plan.layout.pages
	return plan


def compare(sizes: list[tuple[int, int]], page_w: int, page_h: int, pad: int = 0) -> dict[str, Packing]:
	return {m: pack(sizes, page_w, page_h, m, pad) for m in PACKERS}
