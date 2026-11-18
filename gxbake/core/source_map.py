"""Source engine maps (.bsp v19-21) → GoldSrc BSP v30 for the GameCube port.

Source keeps Quake's BSP design, so most of a map carries over directly:

- the render tree (nodes, leafs) becomes hull 0, leaf contents mapped to
  GoldSrc's (solid, water, slime, empty)
- faces keep their planes and edges; each is cut so its lightmap spans at
  most 16 luxels a side, as hlbsp does
- each GoldSrc luxel's world position is projected into the Source lightmap
  (its own lightmap vectors) and resampled, ColorRGBExp32 → 8-bit RGB through
  Source's own display curve, so a surface keeps the brightness it had in
  Source (up to full bright: GoldSrc lightmaps cannot overbrighten)
- Source textures (.vmt → .vtf, from the map's pakfile lump, the game folder
  or .vpk archives) are embedded as 8-bit miptex, resized so a texel covers
  about one unit (GoldSrc ties the lightmap grid to the texture: 16 texels)
- the cluster PVS is expanded to GoldSrc's per-leaf PVS
- displacements become triangle faces (at a reduced power), added to the
  leafs they touch, and thin wedge brushes for collision
- hulls 1-3 are built from the brushes (bsp30.build_clip_hull)
- entities pass through; team spawns become info_player_deathmatch, and
  prop_static entries become cycler_sprite entities naming a converted model

Not carried over: detail props, overlays and decals, cubemaps, water
reflections, HDR (the LDR lightmaps are used when the map has them),
areaportals and occluders. Detail brushes are solid in hulls 1-3 but not in
hull 0 (Source keeps them out of the render tree), so point traces pass them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bsp30, source_bsp as sb

SKIP_TOOLS = ("tools/toolsnodraw", "tools/toolsclip", "tools/toolsplayerclip", "tools/toolsnpcclip",
	"tools/toolstrigger", "tools/toolsinvisible", "tools/toolsareaportal", "tools/toolsoccluder",
	"tools/toolsblocklight", "tools/toolsskip", "tools/toolshint", "tools/toolsfog", "tools/toolsblockbullets",
	"tools/toolsblock_los", "tools/toolsinvisibleladder", "tools/toolsdotted", "tools/toolsorigin")
SKY_TOOLS = ("tools/toolsskybox", "tools/toolsskybox2d", "tools/toolsskyfog")
SPAWN_CLASSES = {"info_player_terrorist", "info_player_counterterrorist", "info_player_teamspawn",
	"info_player_combine", "info_player_rebel", "info_player_axis", "info_player_allies"}
SKY_SUFFIXES = ("rt", "bk", "lf", "ft", "up", "dn")


@dataclass
class MapOptions:
	max_texture: int = 256	# largest embedded miptex edge
	min_texture: int = 32
	texel_density: float = 1.0	# target texels per world unit (sets the lightmap grid: 16 texels a luxel)
	disp_power: int = 2	# displacement detail kept (2 = 5x5 vertices); 0..4
	light_gamma: float = 2.2	# Source's lightmap-to-screen gamma (overbright 2 in gamma space)
	light_scale: float = 1.0	# multiplies the Source light before the curve
	props: bool = True	# prop_static → cycler_sprite entities
	overlays: bool = True	# info_overlay decals become faces laid over the surface
	hulls: bool = True	# build clip hulls 1-3 (slow on big maps; off gives point-only collision)
	search: list[Path] = field(default_factory=list)	# extra game folders or _dir.vpk files
	# Source 2 maps only: they carry no lightmaps GxBake can read, so light is recomputed
	ambient: float = 0.25	# light everywhere, as a share of full bright
	shadows: bool = True	# trace shadows from the sun and lights through the collision
	max_collision: int = 4000	# collision pieces kept for the clip hulls (largest first)
	vis_merge: int = 8	# PVS rows differing by at most this many leaves share their union (0 = exact)


@dataclass
class MapResult:
	bsp: bytes
	warnings: list[str]
	sky: str | None
	sky_sides: dict[str, "np.ndarray"]
	props: list[str]
	stats: dict


# -- content lookup ---------------------------------------------------------------


class Content:
	"""Files by game path: the map's pakfile lump, then game folders, then .vpk archives."""

	def __init__(self, bsp: sb.SourceBSP | None, map_path: Path | None, search: list[Path]):
		self.pak = bsp.pakfile() if bsp is not None else None
		self.pak_names = {n.lower(): n for n in self.pak.namelist()} if self.pak else {}
		self.dirs: list[Path] = []
		self.vpks: list[Path] = []
		cands: list[Path] = []
		if map_path is not None:
			p = Path(map_path).resolve().parent
			cands += [p.parent if p.name.lower() == "maps" else p]
			for d in [p, *p.parents][:6]:
				if (d / "materials").is_dir() or any(d.glob("*_dir.vpk")):
					cands.append(d)
		cands += [Path(s) for s in search]
		for c in cands:
			if c.is_file() and c.name.lower().endswith(".vpk"):
				self.vpks.append(c)
			elif c.is_dir():
				if c not in self.dirs:
					self.dirs.append(c)
				self.vpks += [v for v in sorted(c.glob("*_dir.vpk")) if v not in self.vpks]
		self._vpk_index: dict[str, object] | None = None

	def _vpk(self) -> dict:
		if self._vpk_index is None:
			from . import archives

			self._vpk_index = {}
			for v in self.vpks:
				try:
					for m in archives.members(v):
						self._vpk_index.setdefault(m.name.lower(), m)
				except (OSError, ValueError):
					continue
		return self._vpk_index

	def read(self, rel: str) -> bytes | None:
		from .vtf import _ci_file

		rel = rel.replace("\\", "/").strip("/")
		low = rel.lower()
		if low in self.pak_names:
			return self.pak.read(self.pak_names[low])
		for d in self.dirs:
			f = _ci_file(d, rel)
			if f is not None:
				return f.read_bytes()
		m = self._vpk().get(low)
		return m.read() if m is not None else None

	def material(self, name: str, depth: int = 0) -> tuple["np.ndarray | None", str, dict]:
		"""(RGBA of $basetexture, shader, params) for a material name like 'brick/wall01'."""
		from .vtf import parse_vmt, read_vtf

		name = name.replace("\\", "/").strip("/").lower()
		raw = self.read(f"materials/{name}.vmt")
		shader, params = "", {}
		if raw is not None:
			shader, params = parse_vmt(raw.decode("latin-1", "replace"))
			if shader == "patch" and "include" in params and depth < 4:
				inc = params["include"].replace("\\", "/").lower()
				inc = inc[len("materials/"):] if inc.startswith("materials/") else inc
				inc = inc[:-4] if inc.endswith(".vmt") else inc
				rgba, shader, base = self.material(inc, depth + 1)
				merged = {**base, **{k: v for k, v in params.items() if k not in ("include",)}}
				if merged.get("$basetexture") == base.get("$basetexture") and rgba is not None:
					return rgba, shader, merged
				params = merged
		tex = params.get("$basetexture") or (name if raw is None else None)
		if not tex:
			return None, shader, params
		data = self.read("materials/" + tex.replace("\\", "/").strip("/").lower().removesuffix(".vtf") + ".vtf")
		if data is None:
			return None, shader, params
		try:
			return read_vtf(data), shader, params
		except (ValueError, KeyError, IndexError):
			return None, shader, params


# -- textures ----------------------------------------------------------------------------


def _pow2(v: float, lo: int, hi: int) -> int:
	v = max(lo, min(hi, v))
	return int(2 ** round(math.log2(max(1.0, v))))


def _short_name(name: str, prefix: str, used: set[str]) -> str:
	base = re.sub(r"[^a-z0-9_\-]", "_", name.lower().split("/")[-1]) or "tex"
	cand = (prefix + base)[:15]
	n = 1
	while cand in used:
		n += 1
		suffix = f"~{n}"
		cand = (prefix + base)[:15 - len(suffix)] + suffix
	used.add(cand)
	return cand


def make_miptex(name: str, rgba: "np.ndarray", masked: bool) -> bsp30.Miptex:
	from PIL import Image

	from .models import quantize_skin

	h, w = rgba.shape[:2]
	idx, pal = quantize_skin(rgba, masked)
	palette = np.frombuffer(pal, np.uint8).reshape(256, 3).astype(np.int32)
	levels = [idx]
	img = Image.fromarray(np.ascontiguousarray(rgba))
	for lvl in (1, 2, 3):
		sw, sh = max(1, w >> lvl), max(1, h >> lvl)
		small = np.asarray(img.resize((sw, sh), Image.Resampling.BOX)).astype(np.int32)
		flat = small[..., :3].reshape(-1, 1, 3)
		best = np.empty(len(flat), np.uint8)
		cand = palette[:255] if masked else palette
		for i in range(0, len(flat), 4096):
			d = ((flat[i:i + 4096] - cand[None]) ** 2).sum(2)
			best[i:i + 4096] = d.argmin(1)
		if masked:
			best[small[..., 3].reshape(-1) < 128] = 255
		levels.append(best.tobytes())
	return bsp30.Miptex(name, w, h, levels, pal)


# -- geometry helpers ------------------------------------------------------------------


def _face_points(fc, V, E, S) -> "np.ndarray":
	out = []
	for k in range(int(fc["numedges"])):
		e = int(S[int(fc["firstedge"]) + k])
		out.append(V[E[e][0]] if e >= 0 else V[E[-e][1]])
	return np.array(out)


def _tex_range(pts: "np.ndarray", vec: "np.ndarray") -> tuple[float, float]:
	v = pts.astype(np.float32) @ vec[:3].astype(np.float32) + np.float32(vec[3])
	return float(v.min()), float(v.max())


def _subdivide(poly: "np.ndarray", vecs: "np.ndarray", limit: int = 16) -> list["np.ndarray"]:
	"""Cut a polygon so neither texture axis spans more than `limit` luxels (16 texels each)."""
	out = []
	todo = [poly]
	while todo:
		p = todo.pop()
		done = True
		for ax in range(2):
			lo, hi = _tex_range(p, vecs[ax])
			if math.ceil(hi / 16) - math.floor(lo / 16) > limit:
				cut = (math.floor(lo / 16) + limit - 1) * 16
				n = vecs[ax][:3].astype(np.float64)
				ln = float(np.linalg.norm(n))
				d = (cut - vecs[ax][3]) / ln
				n = n / ln
				back = bsp30.clip_winding(p, n, d)
				front = bsp30.clip_winding(p, n, d, keep_front=True)
				if back is None or front is None:
					continue
				todo += [back, front]
				done = False
				break
		if done:
			out.append(p)
	return out


def _rgbexp(raw: bytes, ofs: int, count: int) -> "np.ndarray":
	a = np.frombuffer(raw, np.uint8, count * 4, ofs).reshape(count, 4)
	e = a[:, 3].astype(np.int8).astype(np.float32)
	return a[:, :3].astype(np.float32) * np.exp2(e)[:, None]


@dataclass
class _OutFace:
	pts: "np.ndarray"	# (k, 3) clockwise from the front
	normal: "np.ndarray"
	dist: float
	texinfo: int	# output texinfo
	src: int	# source face
	special: bool
	flat: "np.ndarray | None" = None	# displacement: undisplaced positions of pts
	lit: bool = True


# -- the converter -----------------------------------------------------------------------


def convert(path: Path, opts: MapOptions | None = None, log=lambda _m: None) -> MapResult:
	opts = opts or MapOptions()
	path = Path(path)
	src = sb.load(path)
	if not (19 <= src.version <= 21 or src.version == 25):
		raise ValueError(f"{path.name}: Source BSP version {src.version} is not supported (19-21 and 25 are)")
	warnings: list[str] = []
	content = Content(src, path, opts.search)
	planes_in = src.planes
	V, E, S = src.vertexes, src.edges, src.surfedges
	faces, lighting, hdr = src.faces_and_lighting()
	if hdr:
		warnings.append("map has HDR lighting only; lightmaps converted from HDR")
	texinfo = src.texinfo
	texdata = src.texdata
	tnames = [n.lower() for n in src.texture_names()]
	out = bsp30.Bsp30()
	P = out.planes

	# -- textures ---------------------------------------------------------------------
	def tex_name(ti: int) -> str:
		td = int(texinfo[ti]["texdata"])
		return tnames[td] if 0 <= td < len(tnames) else ""

	def is_skip(ti: int) -> bool:
		fl = int(texinfo[ti]["flags"])
		name = tex_name(ti)
		return bool(fl & (sb.SURF_NODRAW | sb.SURF_SKIP | sb.SURF_HINT | sb.SURF_TRIGGER)) or name in SKIP_TOOLS

	def is_sky(ti: int) -> bool:
		return bool(int(texinfo[ti]["flags"]) & (sb.SURF_SKY | sb.SURF_SKY2D)) or tex_name(ti) in SKY_TOOLS

	used_tex: dict[int, list[float]] = {}	# texdata → densities
	for fc in faces:
		ti = int(fc["texinfo"])
		if ti < 0 or is_skip(ti) or is_sky(ti):
			continue
		v = texinfo[ti]["vecs"]
		dens = (float(np.linalg.norm(v[0][:3])) + float(np.linalg.norm(v[1][:3]))) / 2
		used_tex.setdefault(int(texinfo[ti]["texdata"]), []).append(dens)
	# overlays: decal quads laid on faces; their textures join the set
	overlays: dict[int, list[tuple["np.ndarray", "np.ndarray", "np.ndarray", int]]] = {}	# face → (pts, uv, n, texdata)
	if opts.overlays:
		for ov in src.overlays:
			ti = int(ov["texinfo"])
			nface = int(ov["facecount"]) & 0x3FFF
			if not 0 <= ti < len(texinfo) or nface < 1:
				continue
			got = _overlay_quad(ov)
			if got is None:
				continue
			pts, uv, n = got
			td = int(texinfo[ti]["texdata"])
			f0 = int(ov["faces"][0])
			if not 0 <= f0 < len(faces) or int(faces[f0]["dispinfo"]) >= 0:
				continue
			overlays.setdefault(f0, []).append((pts, uv, n, td))
			w0 = int(texdata[td]["width"]) or 64
			extent = float(np.linalg.norm(pts[2] - pts[0])) / math.sqrt(2) or 64.0
			used_tex.setdefault(td, []).append(w0 / extent * max(1e-3, abs(float(ov["u"][1] - ov["u"][0]))))
	names_used: set[str] = {"sky"}
	miptex_of: dict[int, tuple[int, float, float]] = {}	# texdata → (miptex, s scale, t scale)
	missing = []
	for td, dens in sorted(used_tex.items()):
		name = tnames[td] if td < len(tnames) else f"tex{td}"
		rgba, shader, params = content.material(name)
		w0, h0 = int(texdata[td]["width"]) or 64, int(texdata[td]["height"]) or 64
		if rgba is None:
			missing.append(name)
			rgba = np.full((64, 64, 4), (128, 128, 128, 255), np.uint8)
			rgba[::8, :, :3] = 96
			rgba[:, ::8, :3] = 96
		masked = params.get("$alphatest", "0") not in ("0", "") or (
			params.get("$translucent", "0") not in ("0", "") and bool((rgba[..., 3] < 128).any()))
		water = "water" in shader or name.startswith("liquids/") or "water" in name.split("/")[-1]
		k = opts.texel_density / max(1e-6, float(np.median(dens)))
		tw = _pow2(w0 * k, min(opts.min_texture, w0), opts.max_texture)
		th = _pow2(h0 * k, min(opts.min_texture, h0), opts.max_texture)
		tw, th = max(16, tw), max(16, th)
		from PIL import Image

		img = np.asarray(Image.fromarray(np.ascontiguousarray(rgba)).resize((tw, th), Image.Resampling.LANCZOS))
		prefix = "{" if masked else "!" if water else ""
		mt = make_miptex(_short_name(name, prefix, names_used), img, masked)
		miptex_of[td] = (len(out.textures), tw / w0, th / h0)
		out.textures.append(mt)
	if missing:
		shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
		warnings.append(f"{len(missing)} material(s) not found, grey used: {shown}")
	sky_index = None

	def sky_miptex() -> int:
		nonlocal sky_index
		if sky_index is None:
			sky_index = len(out.textures)
			grey = np.full((16, 16, 4), (90, 120, 160, 255), np.uint8)
			out.textures.append(make_miptex("sky", grey, False))
		return sky_index

	texinfo_index: dict[tuple, int] = {}

	def out_texinfo(vecs: "np.ndarray", mip: int, flags: int) -> int:
		key = (mip, flags, tuple(np.round(np.asarray(vecs, np.float64).ravel(), 4)))
		i = texinfo_index.get(key)
		if i is None:
			i = len(out.texinfo)
			texinfo_index[key] = i
			out.texinfo.append((np.asarray(vecs, np.float64).reshape(2, 4), mip, flags))
		return i

	def scaled_vecs(ti: int) -> tuple["np.ndarray", int]:
		td = int(texinfo[ti]["texdata"])
		mip, ks, kt = miptex_of[td]
		v = texinfo[ti]["vecs"].astype(np.float64).copy()
		v[0] *= ks
		v[1] *= kt
		return v, mip

	# -- faces --------------------------------------------------------------------------
	dispinfo = src.dispinfo
	dispverts = src.dispverts
	out_faces: list[_OutFace] = []
	counts = np.zeros(len(faces) + 1, np.int64)
	disp_faces: list[tuple[int, int, int]] = []	# (source face, first out face, count)
	disp_collision: list[list[tuple["np.ndarray", float]]] = []
	disp_bases: list[tuple[int, "np.ndarray"]] = []	# (plane, centroid) of each displacement's base face
	disp_tris = 0
	overlay_count = 0
	for i, fc in enumerate(faces):
		ti = int(fc["texinfo"])
		start = len(out_faces)
		if ti < 0 or is_skip(ti):
			counts[i] = 0
			continue
		pts = _face_points(fc, V, E, S)
		pl = planes_in[int(fc["planenum"])]
		n = pl["normal"].astype(np.float64)
		d = float(pl["dist"])
		if is_sky(ti):
			v = texinfo[ti]["vecs"].astype(np.float64)
			oti = out_texinfo(v, sky_miptex(), bsp30.TEX_SPECIAL)
			out_faces.append(_OutFace(pts, n, d, oti, i, True, lit=False))
		elif int(fc["dispinfo"]) >= 0 and int(fc["dispinfo"]) < len(dispinfo) and len(pts) == 4:
			di = dispinfo[int(fc["dispinfo"])]
			disp_bases.append((int(fc["planenum"]), pts.mean(0)))
			vecs, mip = scaled_vecs(ti)
			tris, coll = _displacement(pts, di, dispverts, opts.disp_power, n)
			disp_collision += coll
			for tri_pts, tri_flat in tris:
				tn = np.cross(tri_pts[2] - tri_pts[0], tri_pts[1] - tri_pts[0])
				ln = float(np.linalg.norm(tn))
				if ln < 1e-6:
					continue
				tn /= ln
				tv = _tri_vecs(tri_pts, tri_flat, vecs, tn)
				oti = out_texinfo(tv, mip, 0)
				out_faces.append(_OutFace(tri_pts, tn, float(tn @ tri_pts[0]), oti, i, False, tri_flat))
				disp_tris += 1
			disp_faces.append((i, start, len(out_faces) - start))
		else:
			vecs, mip = scaled_vecs(ti)
			oti = out_texinfo(vecs, mip, 0)
			lit = int(fc["lightofs"]) >= 0 and not (int(texinfo[ti]["flags"]) & sb.SURF_NOLIGHT)
			for piece in _subdivide(pts, vecs):
				out_faces.append(_OutFace(piece, n, d, oti, i, False, lit=lit))
		for ov_pts, uv, on, td in overlays.get(i, ()):
			mip, ks, kt = miptex_of[td]
			tex = out.textures[mip]
			lifted = ov_pts + on * 0.5	# off the wall, so the port's depth test keeps it in front
			vv, good = _uv_vecs3(lifted[:3], uv[:3] * [tex.width, tex.height], on)
			if not good:
				continue
			oti = out_texinfo(vv, mip, 0)
			lit = int(fc["lightofs"]) >= 0
			for piece in _subdivide(lifted, vv):
				out_faces.append(_OutFace(piece, on, float(on @ piece[0]), oti, i, False, lit=lit))
			overlay_count += 1
		counts[i] = len(out_faces) - start
	rstart = np.concatenate([[0], np.cumsum(counts[:len(faces)])])
	log(f"  {len(out_faces)} faces ({disp_tris} from displacements), {len(out.textures)} textures")

	# -- vertexes, edges, faces, lighting ----------------------------------------------------
	vindex: dict[tuple, int] = {}
	eindex: dict[tuple[int, int], int] = {}

	def vid(p) -> int:
		key = (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))
		i = vindex.get(key)
		if i is None:
			i = len(out.vertexes)
			vindex[key] = i
			out.vertexes.append(key)
		return i

	def eid(a: int, b: int) -> int:
		if (b, a) in eindex:
			return -eindex[(b, a)]
		i = eindex.get((a, b))
		if i is None:
			i = len(out.edges)
			eindex[(a, b)] = i
			out.edges.append((a, b))
		return i

	light_stats = [0, 0]
	for of in out_faces:
		ids = [vid(p) for p in of.pts]
		ids = [x for k, x in enumerate(ids) if x != ids[k - 1]] if len(ids) > 3 else ids
		if len(ids) < 3:
			ids = ids + ids[:1] * (3 - len(ids))
		first = len(out.surfedges)
		for k in range(len(ids)):
			out.surfedges.append(eid(ids[k], ids[(k + 1) % len(ids)]))
		pidx, flipped = P.add(of.normal, of.dist)
		fc = faces[of.src]
		styles = (255, 255, 255, 255)
		lightofs = -1
		if not of.special and of.lit and int(fc["lightofs"]) >= 0 and len(lighting):
			vecs = out.texinfo[of.texinfo][0]
			samples = _relight(of, fc, texinfo[int(fc["texinfo"])], vecs, lighting, opts)
			if samples is not None:
				lightofs = len(out.lighting)
				styles = tuple(int(s) for s in fc["styles"][:len(samples)]) + (255,) * (4 - len(samples))
				for block in samples:
					out.lighting += block
				light_stats[0] += 1
			else:
				light_stats[1] += 1
		out.faces.append((pidx, 1 if flipped else 0, first, len(ids), of.texinfo, styles, lightofs))

	# -- nodes, leafs, marksurfaces, visibility ------------------------------------------------
	nodes = src.nodes
	leafs = src.leafs
	models = src.models
	leaffaces = src.leaffaces
	for nd in nodes:
		pl = planes_in[int(nd["planenum"])]
		pidx, flipped = P.add(pl["normal"].astype(np.float64), float(pl["dist"]))
		c0, c1 = int(nd["children"][0]), int(nd["children"][1])
		if flipped:
			c0, c1 = c1, c0
		ff, nf = int(nd["firstface"]), int(nd["numfaces"])
		a, b = int(rstart[ff]), int(rstart[min(ff + nf, len(faces))])
		out.nodes.append((pidx, c0, c1, _box(nd["mins"], -1), _box(nd["maxs"], 1), a, b - a))

	world_leafs = _leafs_under(nodes, int(models[0]["headnode"])) if len(models) else set()
	visleafs = max(world_leafs) if world_leafs else 0
	disp_leafs: dict[int, list[int]] = {}
	for srcf, first, cnt in disp_faces:
		for k in range(first, first + cnt):
			pts = out_faces[k].pts
			for lf in {_point_leaf(nodes, planes_in, int(models[0]["headnode"]), p) for p in [*pts, pts.mean(0)]}:
				if lf > 0 and int(leafs[lf]["contents"]) & sb.CONTENTS_SOLID == 0:
					disp_leafs.setdefault(lf, []).append(k)
	nclusters, pvs = src.visibility()
	cluster_leafs: dict[int, list[int]] = {}
	for li in range(1, visleafs + 1):
		c = int(leafs[li]["cluster"])
		if c >= 0:
			cluster_leafs.setdefault(c, []).append(li)
	vis = bytearray()
	vis_rows: dict[bytes, int] = {}
	row_bytes = (visleafs + 7) // 8
	cluster_row: dict[int, int] = {}
	# Every cluster's row in leaf bits first, so near-identical rows can be
	# merged into their union (bsp30.merge_vis_rows) before they are
	# compressed; identical compressed rows share one copy.
	rows: list[bytes | None] = []
	for c in range(nclusters):
		row = bytearray(row_bytes)
		bits = pvs[c]
		for oc in range(nclusters):
			if bits[oc >> 3] & (1 << (oc & 7)):
				for ol in cluster_leafs.get(oc, ()):
					row[(ol - 1) >> 3] |= 1 << ((ol - 1) & 7)
		rows.append(bytes(row))
	rows = bsp30.merge_vis_rows(rows, opts.vis_merge)
	for li, lf in enumerate(leafs):
		contents = int(lf["contents"])
		if contents & sb.CONTENTS_SOLID:
			gc = bsp30.CONTENTS_SOLID
		elif contents & sb.CONTENTS_SLIME:
			gc = bsp30.CONTENTS_SLIME
		elif contents & sb.CONTENTS_WATER:
			gc = bsp30.CONTENTS_WATER
		else:
			gc = bsp30.CONTENTS_EMPTY
		visofs = -1
		c = int(lf["cluster"])
		if nclusters and li in world_leafs and c >= 0 and c < nclusters:
			if c not in cluster_row:
				comp = bsp30.compress_vis(rows[c])
				if comp not in vis_rows:
					vis_rows[comp] = len(vis)
					vis += comp
				cluster_row[c] = vis_rows[comp]
			visofs = cluster_row[c]
		first = len(out.marksurfaces)
		if li > 0:
			for k in range(int(lf["numleaffaces"])):
				f = int(leaffaces[int(lf["firstleafface"]) + k])
				out.marksurfaces += range(int(rstart[f]), int(rstart[f + 1]))
			out.marksurfaces += disp_leafs.get(li, [])
		amb = (0, 0, 0, 0)
		out.leafs.append((gc, visofs, _box(lf["mins"], -1), _box(lf["maxs"], 1), first, len(out.marksurfaces) - first, amb))
	out.visibility = bytes(vis)

	# -- clip hulls and models ------------------------------------------------------------------
	brushes = src.brushes
	sides = src.brushsides
	leafbrushes = src.leafbrushes

	def brush_sides(bi: int) -> list[tuple["np.ndarray", float]]:
		br = brushes[bi]
		out_s = []
		for s in sides[int(br["firstside"]):int(br["firstside"]) + int(br["numsides"])]:
			pl = planes_in[int(s["planenum"])]
			out_s.append((pl["normal"].astype(np.float64), float(pl["dist"])))
		return out_s

	def disp_brush(bi: int) -> bool:
		"""The brush a displacement was made from: a side on the base face's plane, holding its centre.

		Source collides with the displacement surface instead, so these leave the hulls.
		"""
		if not disp_bases:
			return False
		br = brushes[bi]
		ss = sides[int(br["firstside"]):int(br["firstside"]) + int(br["numsides"])]
		pnums = {int(x["planenum"]) for x in ss}
		for plane, centre in disp_bases:
			if plane in pnums and all(
					float(planes_in[int(x["planenum"])]["normal"].astype(np.float64) @ centre)
					<= float(planes_in[int(x["planenum"])]["dist"]) + 0.5 for x in ss):
				return True
		return False

	hull_brush_count = 0
	for mi, md in enumerate(models):
		heads = [int(md["headnode"]), 0, 0, 0]
		ff, nf = int(md["firstface"]), int(md["numfaces"])
		a, b = int(rstart[ff]), int(rstart[min(ff + nf, len(faces))])
		if opts.hulls:
			bset = set()
			for lf in _leafs_under(nodes, int(md["headnode"])):
				L = leafs[lf]
				for k in range(int(L["numleafbrushes"])):
					bset.add(int(leafbrushes[int(L["firstleafbrush"]) + k]))
			hb = [brush_sides(bi) for bi in sorted(bset)
				if int(brushes[bi]["contents"]) & sb.MASK_PLAYERSOLID and not (mi == 0 and disp_brush(bi))]
			if mi == 0:
				hb += disp_collision
			hull_brush_count += len(hb)
			for h in (1, 2, 3):
				heads[h] = bsp30.build_clip_hull(hb, h, P, out.clipnodes)
		else:
			for h in (1, 2, 3):
				heads[h] = bsp30.build_clip_hull([], h, P, out.clipnodes)
		out.models.append((tuple(md["mins"]), tuple(md["maxs"]), tuple(md["origin"]), tuple(heads),
			visleafs if mi == 0 else 0, a, b - a))
	log(f"  {len(out.clipnodes)} clipnodes from {hull_brush_count} brushes")

	# -- entities --------------------------------------------------------------------------------
	ents = sb.parse_entities(src.entities_text())
	sky = None
	props: list[str] = []
	has_start = any(e.get("classname") == "info_player_start" for e in ents)
	first_spawn = None
	for e in ents:
		cls = e.get("classname", "")
		if cls == "worldspawn":
			sky = e.get("skyname") or None
			e["wad"] = ""
			for k in ("detailmaterial", "detailvbsp", "maxpropscreenwidth", "minpropscreenwidth"):
				e.pop(k, None)
		elif cls in SPAWN_CLASSES:
			e["classname"] = "info_player_deathmatch"
			first_spawn = first_spawn or e
	if not has_start and first_spawn is not None:
		ents.append({"classname": "info_player_start", "origin": first_spawn.get("origin", "0 0 0"),
			"angles": first_spawn.get("angles", "0 0 0")})
	if opts.props:
		for sp in src.static_props():
			stem = Path(sp.model).stem.lower()
			props.append(sp.model)
			ents.append({"classname": "cycler_sprite", "model": f"models/{stem}.mdl",
				"origin": " ".join(f"{c:g}" for c in sp.origin), "angles": " ".join(f"{c:g}" for c in sp.angles),
				"framerate": "0"})
	out.entities = sb.format_entities(ents)

	sky_sides: dict[str, np.ndarray] = {}
	if sky:
		for sfx in SKY_SUFFIXES:
			rgba, _sh, _p = content.material(f"skybox/{sky}{sfx}")
			if rgba is not None:
				sky_sides[sfx] = rgba
		if len(sky_sides) < 6:
			warnings.append(f"sky {sky}: {6 - len(sky_sides)} of 6 sides not found")
	problems = out.check()
	if problems:
		raise ValueError(f"{path.name}: " + "; ".join(problems))
	if light_stats[1]:
		warnings.append(f"{light_stats[1]} face(s) had no usable Source lightmap")
	stats = {"overlays": overlay_count, "faces": len(out.faces), "nodes": len(out.nodes), "leafs": len(out.leafs),
		"clipnodes": len(out.clipnodes), "textures": len(out.textures), "texinfo": len(out.texinfo),
		"lighting": len(out.lighting), "displacement triangles": disp_tris}
	return MapResult(out.write(), warnings, sky, sky_sides, sorted(set(props)), stats)


def _overlay_quad(ov) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"] | None:
	"""(corners clockwise from the front, their (u, v), normal) of one info_overlay."""
	n = ov["normal"].astype(np.float64)
	ln = float(np.linalg.norm(n))
	if ln < 1e-6:
		return None
	n /= ln
	p = ov["points"].astype(np.float64)
	bu = np.array([p[0, 2], p[1, 2], p[2, 2]])	# the engine keeps basis U in the points' z
	if float(np.linalg.norm(bu)) < 1e-6:
		bu = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
	bu /= float(np.linalg.norm(bu))
	bv = np.cross(n, bu)
	o = ov["origin"].astype(np.float64)
	pts = np.array([o + p[i, 0] * bu + p[i, 1] * bv for i in range(4)])
	u0, u1 = (float(x) for x in ov["u"])
	v0, v1 = (float(x) for x in ov["v"])
	uv = np.array([(u0, v0), (u0, v1), (u1, v1), (u1, v0)])
	if np.cross(pts[1] - pts[0], pts[2] - pts[0]) @ n > 0:
		pts, uv = pts[::-1].copy(), uv[::-1].copy()
	if float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))) < 1e-3:
		return None
	return pts, uv, n


def _uv_vecs3(pts: "np.ndarray", st: "np.ndarray", n: "np.ndarray") -> tuple["np.ndarray", bool]:
	from .s2map import _uv_vecs_many

	v, good = _uv_vecs_many(np.asarray(pts, np.float64)[None], np.asarray(st, np.float64)[None], np.asarray(n)[None])
	return v[0], bool(good[0])


def _box(v, direction: int) -> tuple[int, int, int]:
	"""Bounds as BSP30's int16 (floats from version 25 are rounded outward)."""
	a = np.asarray(v, np.float64)
	a = np.floor(a) if direction < 0 else np.ceil(a)
	return tuple(int(x) for x in np.clip(a, -32768, 32767))


# -- tree walks ----------------------------------------------------------------------------


def _leafs_under(nodes, head: int) -> set[int]:
	out: set[int] = set()
	stack = [head]
	while stack:
		n = stack.pop()
		if n < 0:
			out.add(-1 - n)
			continue
		stack += [int(nodes[n]["children"][0]), int(nodes[n]["children"][1])]
	return out


def _point_leaf(nodes, planes, head: int, p) -> int:
	n = head
	while n >= 0:
		nd = nodes[n]
		pl = planes[int(nd["planenum"])]
		d = float(pl["normal"].astype(np.float64) @ p - pl["dist"])
		n = int(nd["children"][0] if d >= 0 else nd["children"][1])
	return -1 - n


# -- displacements --------------------------------------------------------------------------


def _displacement(quad: "np.ndarray", di, dispverts, max_power: int, normal: "np.ndarray"):
	"""Triangles (displaced pts, flat pts), clockwise from the front, and collision wedges."""
	start = np.asarray(di["start"], np.float64)
	k0 = int(np.argmin(((quad - start) ** 2).sum(1)))
	a, b, c, d = (quad[(k0 + j) % 4] for j in range(4))
	power = int(di["power"])
	n = (1 << power) + 1
	dv = dispverts[int(di["firstvert"]):int(di["firstvert"]) + n * n]
	if len(dv) < n * n:
		return [], []
	idx = np.arange(n * n)
	t1 = (idx % n) / (n - 1)
	t2 = (idx // n) / (n - 1)
	ad = a[None] + (d - a)[None] * t1[:, None]
	bc = b[None] + (c - b)[None] * t1[:, None]
	flat = ad + (bc - ad) * t2[:, None]
	pos = flat + dv["vec"].astype(np.float64) * dv["dist"].astype(np.float64)[:, None]
	step = 1 << max(0, power - max(0, min(max_power, power)))
	keep = list(range(0, n, step))
	m = len(keep)

	def at(r: int, cc: int) -> int:
		return keep[r] * n + keep[cc]

	tris = []
	coll = []
	for r in range(m - 1):
		for cc in range(m - 1):
			q = (at(r, cc), at(r, cc + 1), at(r + 1, cc + 1), at(r + 1, cc))
			pair = ((q[0], q[1], q[2]), (q[0], q[2], q[3])) if (r + cc) % 2 == 0 else \
				((q[0], q[1], q[3]), (q[1], q[2], q[3]))
			for t in pair:
				tp, tf = pos[list(t)], flat[list(t)]
				# keep the base face's clockwise-from-front winding
				if np.cross(tp[1] - tp[0], tp[2] - tp[0]) @ normal > 0:
					tp, tf = tp[::-1].copy(), tf[::-1].copy()
				tris.append((tp, tf))
				coll.append(_wedge(tp, normal))
	return tris, [w for w in coll if w]


def _wedge(tri: "np.ndarray", up: "np.ndarray", depth: float = 12.0) -> list[tuple["np.ndarray", float]]:
	"""A thin prism under one displacement triangle, as outward planes."""
	n = np.cross(tri[2] - tri[0], tri[1] - tri[0])
	ln = float(np.linalg.norm(n))
	if ln < 1e-6:
		return []
	n /= ln
	bottom = tri - up * depth
	out = [(n, float(n @ tri[0])), (-up, float(-up @ bottom[0]))]
	centre = (tri.mean(0) + bottom.mean(0)) / 2
	for i in range(3):
		e = tri[(i + 1) % 3] - tri[i]
		sn = np.cross(e, up)
		ls = float(np.linalg.norm(sn))
		if ls < 1e-6:
			return []
		sn /= ls
		sd = float(sn @ tri[i])
		if sn @ centre > sd:
			sn, sd = -sn, -sd
		out.append((sn, sd))
	return out


def _tri_vecs(pts: "np.ndarray", flat: "np.ndarray", vecs: "np.ndarray", n: "np.ndarray") -> "np.ndarray":
	"""Texture vectors mapping a displaced triangle onto the UVs its undisplaced corners had."""
	st = flat @ vecs[:, :3].T + vecs[:, 3]	# (3, 2) wanted s, t
	plain = pts @ vecs[:, :3].T + vecs[:, 3]
	if np.abs(plain - st).max() < 0.25:
		return vecs
	m = np.zeros((4, 4))
	m[:3, :3] = pts
	m[:3, 3] = 1.0
	m[3, :3] = n
	out = np.zeros((2, 4))
	for ax in range(2):
		rhs = np.array([st[0, ax], st[1, ax], st[2, ax], 0.0])
		try:
			out[ax] = np.linalg.solve(m, rhs)
		except np.linalg.LinAlgError:
			return vecs
	return out


# -- lighting ----------------------------------------------------------------------------------


def _relight(of: _OutFace, fc, ti, vecs: "np.ndarray", lighting: bytes, opts: MapOptions):
	"""Per-style RGB blocks for one output face, sampled from its Source face's lightmap."""
	s_lo, s_hi = _tex_range(of.pts, vecs[0])
	t_lo, t_hi = _tex_range(of.pts, vecs[1])
	bs, bt = math.floor(s_lo / 16), math.floor(t_lo / 16)
	smax = math.ceil(s_hi / 16) - bs + 1
	tmax = math.ceil(t_hi / 16) - bt + 1
	if smax < 1 or tmax < 1 or smax > 64 or tmax > 64:
		return None
	ss, tt = np.meshgrid((bs + np.arange(smax)) * 16.0, (bt + np.arange(tmax)) * 16.0)
	m = np.array([vecs[0][:3], vecs[1][:3], of.normal], np.float64)
	try:
		inv = np.linalg.inv(m)
	except np.linalg.LinAlgError:
		return None
	rhs = np.stack([ss.ravel() - vecs[0][3], tt.ravel() - vecs[1][3], np.full(ss.size, of.dist)], 1)
	world = rhs @ inv.T	# (n, 3)
	if of.flat is not None:
		world = _to_flat(world, of.pts, of.flat)
	lm = ti["lmvecs"].astype(np.float64)
	w, h = int(fc["lm_size"][0]) + 1, int(fc["lm_size"][1]) + 1
	u = world @ lm[0][:3] + lm[0][3] - float(fc["lm_mins"][0])
	v = world @ lm[1][:3] + lm[1][3] - float(fc["lm_mins"][1])
	u = np.clip(u, 0, w - 1)
	v = np.clip(v, 0, h - 1)
	u0 = np.floor(u).astype(int)
	v0 = np.floor(v).astype(int)
	u1 = np.minimum(u0 + 1, w - 1)
	v1 = np.minimum(v0 + 1, h - 1)
	fu = (u - u0)[:, None]
	fv = (v - v0)[:, None]
	nstyles = sum(1 for s in fc["styles"] if s != 255)
	bump = bool(int(ti["flags"]) & sb.SURF_BUMPLIGHT)
	per = w * h * (4 if bump else 1)
	blocks = []
	base = int(fc["lightofs"])
	for st in range(max(1, nstyles)):
		ofs = base + st * per * 4
		if ofs + w * h * 4 > len(lighting):
			break
		grid = _rgbexp(lighting, ofs, w * h).reshape(h, w, 3)
		val = (grid[v0, u0] * (1 - fu) * (1 - fv) + grid[v0, u1] * fu * (1 - fv) +
			grid[v1, u0] * (1 - fu) * fv + grid[v1, u1] * fu * fv)
		# Source shows a texture at 2 * (L / 510) ** (1 / 2.2) of its brightness (gamma-space
		# overbright 2); GoldSrc shows it at about byte / 255 and cannot go over 1.
		val = 2.0 * np.clip(val * opts.light_scale / 510.0, 0, None) ** (1.0 / opts.light_gamma) * 255.0
		blocks.append(np.clip(val + 0.5, 0, 255).astype(np.uint8).tobytes())
	return blocks or None


def _to_flat(world: "np.ndarray", tri: "np.ndarray", flat: "np.ndarray") -> "np.ndarray":
	"""Barycentric coordinates on the displaced triangle, applied to its undisplaced corners."""
	a, b, c = tri
	v0, v1 = b - a, c - a
	d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
	den = d00 * d11 - d01 * d01
	if abs(den) < 1e-9:
		return np.repeat(flat.mean(0)[None], len(world), 0)
	p = world - a
	d20, d21 = p @ v0, p @ v1
	w1 = (d11 * d20 - d01 * d21) / den
	w2 = (d00 * d21 - d01 * d20) / den
	w0 = 1 - w1 - w2
	return w0[:, None] * flat[0] + w1[:, None] * flat[1] + w2[:, None] * flat[2]


def map_info(path: Path) -> dict:
	src = sb.load(path)
	faces, _l, hdr = src.faces_and_lighting()
	return {"Source BSP version": src.version, "Faces": len(faces), "Leafs": len(src.leafs),
		"Brushes": len(src.brushes), "Displacements": len(src.dispinfo), "Models": len(src.models),
		"Static props": len(src.static_props()), "Lighting": "HDR only" if hdr else "LDR"}
