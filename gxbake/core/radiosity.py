"""Radiosity / global-illumination lightmaps for GoldSrc (BSP v30) maps, baked for GX.

`gxbake gi map.bsp` relights a compiled map from its own geometry and
entities, the way HLRAD does, and writes the result two ways:

* **the map, relit** (`<map>.bsp`): the LIGHTING lump replaced and each
  face's light offset and styles updated. Face lightmap grids are exactly
  the engine's (CalcSurfaceExtents: one luxel per 16 texture units), so the
  port's existing world baker and lightmap atlas take it as they take any
  map. Nothing changes on the GameCube side.
* **native atlas pages** (`<map>_lm.tpl` + `<map>.gclm` + `<map>_lm.gxm`):
  every face's lightmap packed into a few power-of-two pages in whichever GX
  format holds it best under the lightmap window (CMPR, RGB565, or I4/I8 when
  the light is grey), with each face's rectangle aligned to the 4×4 block so
  a CMPR block never mixes two faces, a luxel of edge padding for bilinear
  filtering, and light stored as light ÷ 2 or ÷ 4 for an overbright TEV
  stage when the map is brighter than 1. `port/gc_lmap.c` reads the
  `.gclm` face table and turns a vertex's lightmap coordinates into atlas
  UVs.

What is lit:

* **direct light**: `light`, `light_spot` (cone and cone2) and the sun of
  `light_environment`, with shadow rays against the world (and, optionally,
  brush entities), inverse-square falloff;
* **emissive surfaces**: texture lights from a `lights.rad`-style file or
  the manifest (`~LIGHT1 255 255 200 800`), as area lights with their own
  shadows, and their own faces lit by their emission;
* **sky light**: every hemisphere ray that reaches a sky face brings back
  the sky's colour (`_diffuse_light` of `light_environment`, or a share of
  the sun's);
* **bounces**: radiosity between patches (face pieces of about `patch`
  units): each patch casts cosine-weighted rays, the hits become a sparse
  transfer matrix, and each bounce is one matrix product, so extra bounces
  cost almost nothing. Reflectance is each texture's average colour (from
  the BSP or its WADs);
* **ambient occlusion**: short rays per luxel darken the indirect and sky
  light in corners and contact areas (optional, `ao`).

Brightness follows HLRAD's conventions in spirit (point lights fall off with
1/d², the stored GoldSrc bytes are light^gamma with gamma 0.55), but it is
not HLRAD's code and the absolute scale is not byte-identical. `calibrate`
fits one global scale so the new lighting matches the map's existing lump in
the median, which is the way to keep a map's mood while adding bounce light.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import raytrace

LUXEL = 16
TEX_SPECIAL = 1
L_ENTITIES, L_PLANES, L_TEXTURES, L_VERTEXES, L_TEXINFO, L_FACES, L_LIGHTING, L_EDGES, L_SURFEDGES, L_MODELS = \
	0, 1, 2, 3, 6, 7, 8, 12, 13, 14

# light units: 1.0 = full bright (a GoldSrc lightmap byte of 255)
K_POINT = 40.0	# a brightness-200 white light 128 units away, head on, gives about 0.5
K_SUN = 1 / 255.0	# light_environment brightness 255 = full bright on a surface facing the sun
K_TEX = 0.5 / 255.0	# texture light exitance per unit of lights.rad intensity

GCLM_MAGIC = b"GCLM"
GCLM_VERSION = 1
GCLM_HEADER = 32
GCLM_FACE = 12
GCLM_NONE = 0xFFFF
GCLM_FLAG_PORT_TONE = 1	# pages hold 2·light^gamma (what the port's ×2 BSP-lightmap stage shows), ÷ scale
TONES = ("port", "linear")


@dataclass
class GiOptions:
	bounces: int = 3
	patch: float = 64.0	# radiosity patch size, world units (rounded to whole luxels)
	rays: int = 96	# hemisphere rays per patch for the transfer matrix
	gather: int = 0	# final-gather rays per luxel (0: interpolate the patches; sharper, slower when > 0)
	ao: float = 0.0	# 0..1: how much ambient occlusion darkens indirect and sky light
	ao_radius: float = 96.0
	ao_rays: int = 16
	light_scale: float = 1.0	# point and spot lights
	sun_scale: float = 1.0
	sky_scale: float = 1.0
	sky_default: float = 0.5	# sky colour as a share of the sun when light_environment has no _diffuse_light
	texlight_scale: float = 1.0
	texlight_step: int = 2	# one area-light sample per this many luxels of an emissive face
	cutoff: float = 1 / 4096	# lights are ignored where they would add less than this (light units)
	use_vis: bool = True	# cull lights with the map's PVS when it has one
	reflect: float = 1.0	# texture reflectance multiplier (HLRAD -texreflectscale)
	default_albedo: float = 0.5	# when a texture's pixels are not found (no WAD)
	max_albedo: float = 0.9
	ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)	# added everywhere, light units
	gamma: float = 0.55	# GoldSrc bytes = light^gamma (HLRAD -gamma)
	exposure: float = 0.0	# stops
	calibrate: bool = False	# scale to match the map's existing lighting (median)
	entity_shadows: bool = False	# brush entities (doors, func_wall) cast shadows too
	texlights: dict = field(default_factory=dict)	# texture name → (r, g, b, intensity)
	rad: list = field(default_factory=list)	# lights.rad files
	wads: list = field(default_factory=list)	# WAD files or folders for texture reflectance
	seed: int = 1
	log: object = None	# callable(str) for progress

	def say(self, msg: str) -> None:
		if callable(self.log):
			self.log(msg)


# -- the map --------------------------------------------------------------------------------------


@dataclass
class Face:
	index: int
	model: int
	verts: "np.ndarray"	# k×3
	normal: "np.ndarray"	# facing the lit side
	dist: float
	vecs: "np.ndarray"	# 2×4 texinfo s/t
	flags: int
	tex: str
	mins: tuple[int, int] = (0, 0)	# luxel units (texturemins / 16)
	w: int = 0
	h: int = 0
	lit: bool = False
	sky: bool = False

	@property
	def area(self) -> float:
		v = self.verts
		return 0.5 * float(np.linalg.norm(np.cross(v[1:-1] - v[0], v[2:] - v[0]).sum(0))) if len(v) >= 3 else 0.0


@dataclass
class Light:
	kind: str	# point | spot | sun | surface
	origin: "np.ndarray"
	color: "np.ndarray"	# linear, already × brightness × constant
	direction: "np.ndarray" = field(default_factory=lambda: np.zeros(3))	# spot axis / sun travel direction / surface normal
	cone: float = 0.0	# cos of the inner cone
	cone2: float = 0.0	# cos of the outer cone
	area: float = 0.0
	fade: float = 1.0
	name: str = ""


@dataclass
class Scene:
	faces: list[Face]
	entities: list[dict]
	textures: list[str]
	albedo: dict[str, "np.ndarray"]
	lumps: list[bytes]
	order: list[int]	# lump indices in file order
	warnings: list[str] = field(default_factory=list)


def _lumps(data: bytes) -> tuple[list[bytes], list[int]]:
	if len(data) < 4 + 15 * 8 or struct.unpack_from("<i", data)[0] != 30:
		raise ValueError("not a GoldSrc (v30) BSP")
	out, offs = [], []
	for i in range(15):
		off, ln = struct.unpack_from("<ii", data, 4 + 8 * i)
		if off < 0 or ln < 0 or off + ln > len(data):
			raise ValueError(f"lump {i} runs past the end of the file")
		out.append(data[off:off + ln])
		offs.append(off)
	return out, sorted(range(15), key=lambda i: (offs[i], i))


def _texture_names(lump: bytes) -> list[tuple[str, int]]:
	"""(name, offset in lump) per miptex index; offset -1 for a missing entry."""
	if len(lump) < 4:
		return []
	n = struct.unpack_from("<i", lump)[0]
	out = []
	for i in range(n):
		o = struct.unpack_from("<i", lump, 4 + 4 * i)[0]
		if o < 0 or o + 16 > len(lump):
			out.append(("", -1))
			continue
		out.append((lump[o:o + 16].split(b"\0")[0].decode("latin-1").lower(), o))
	return out


def _avg_linear(rgba: "np.ndarray") -> "np.ndarray":
	from .hdr import srgb_to_linear

	px = rgba[..., :3].reshape(-1, 3).astype(np.float32) / 255
	if rgba.shape[-1] > 3:
		vis = rgba[..., 3].reshape(-1) >= 128
		if vis.any():
			px = px[vis]
	return srgb_to_linear(px).mean(0) if len(px) else np.full(3, 0.5, np.float32)


def _wad_files(paths) -> list[Path]:
	out = []
	for p in paths:
		p = Path(p)
		if p.is_dir():
			out += sorted(q for q in p.rglob("*") if q.suffix.lower() == ".wad")
		elif p.is_file():
			out.append(p)
	return out


def _albedos(texlump: bytes, names: list[tuple[str, int]], wads: list[Path], warnings: list[str]) -> dict:
	from . import goldsrc

	out: dict[str, np.ndarray] = {}
	for name, o in names:
		if o < 0 or not name or name in out:
			continue
		mip = goldsrc.wt_read_miptex(texlump, o)
		if mip is not None:
			_n, w, h, px, pal = mip
			out[name] = _avg_linear(goldsrc.miptex_rgba(w, h, px, pal, name.startswith("{")))
	want = {n for n, _o in names if n and n not in out}
	for wad in wads:
		if not want:
			break
		try:
			data = wad.read_bytes()
			entries = goldsrc.wad_entries(data)
		except (OSError, ValueError, struct.error):
			continue
		for e in entries:
			n = e.name.lower()
			if n in want and not e.compressed:
				mip = goldsrc.wt_read_miptex(data, e.offset)
				if mip is not None:
					_n, w, h, px, pal = mip
					out[n] = _avg_linear(goldsrc.miptex_rgba(w, h, px, pal, n.startswith("{")))
					want.discard(n)
	missing = sorted(n for n in want if not n.startswith("sky") and n[:1] not in ("!", "*"))
	if missing:
		warnings.append(f"{len(missing)} texture(s) not in the BSP or the WADs given; they reflect as grey "
			f"({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''})")
	return out


def is_sky(name: str) -> bool:
	return name == "sky" or name.startswith("sky")


def load(data: bytes, opts: GiOptions | None = None) -> Scene:
	from .source_bsp import parse_entities

	opts = opts or GiOptions()
	lumps, order = _lumps(data)
	warnings: list[str] = []
	planes = np.frombuffer(lumps[L_PLANES], "<f4", len(lumps[L_PLANES]) // 20 * 5).reshape(-1, 5)
	verts = np.frombuffer(lumps[L_VERTEXES], "<f4", len(lumps[L_VERTEXES]) // 12 * 3).reshape(-1, 3).astype(np.float64)
	edges = np.frombuffer(lumps[L_EDGES], "<u2", len(lumps[L_EDGES]) // 4 * 2).reshape(-1, 2)
	surf = np.frombuffer(lumps[L_SURFEDGES], "<i4", len(lumps[L_SURFEDGES]) // 4)
	ti_raw = lumps[L_TEXINFO]
	ntex = len(ti_raw) // 40
	tvecs = np.frombuffer(ti_raw, "<f4", ntex * 10).reshape(-1, 10)[:, :8].astype(np.float64).reshape(-1, 2, 4)
	tmeta = np.frombuffer(ti_raw, "<i4", ntex * 10).reshape(-1, 10)[:, 8:]
	names = _texture_names(lumps[L_TEXTURES])
	ents = parse_entities(lumps[L_ENTITIES].split(b"\0", 1)[0].decode("latin-1", "replace"))
	model_of = {}
	mraw = lumps[L_MODELS]
	for m in range(len(mraw) // 64):
		first, num = struct.unpack_from("<ii", mraw, m * 64 + 56)
		for f in range(first, first + num):
			model_of.setdefault(f, m)
	wads = list(opts.wads)
	albedo = _albedos(lumps[L_TEXTURES], names, _wad_files(wads), warnings)
	faces = []
	fl = lumps[L_FACES]
	for k in range(len(fl) // 20):
		pnum, side, first, num, ti, *_st, _lo = struct.unpack_from("<HHiHH4Bi", fl, k * 20)
		if num < 3 or ti >= ntex or pnum >= len(planes) or first + num > len(surf):
			continue
		se = surf[first:first + num]
		idx = np.where(se >= 0, edges[np.abs(se), 0], edges[np.abs(se), 1])
		v = verts[idx]
		n = planes[pnum, :3].astype(np.float64)
		d = float(planes[pnum, 3])
		if side:
			n, d = -n, -d
		mi = int(tmeta[ti, 0])
		name = names[mi][0] if 0 <= mi < len(names) else ""
		f = Face(k, model_of.get(k, 0), v, n, d, tvecs[ti], int(tmeta[ti, 1]), name)
		f.sky = is_sky(name)
		f.lit = not (f.flags & TEX_SPECIAL) and not f.sky
		if f.lit:
			st = v @ f.vecs[:, :3].T + f.vecs[:, 3]
			lo = np.floor(st.min(0) / LUXEL).astype(int)
			hi = np.ceil(st.max(0) / LUXEL).astype(int)
			f.mins = (int(lo[0]), int(lo[1]))
			f.w, f.h = int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)
		faces.append(f)
	return Scene(faces, ents, [n for n, _o in names], albedo, lumps, order, warnings)


# -- lights ---------------------------------------------------------------------------------------


def _vec(text: str, n: int = 3) -> list[float]:
	try:
		v = [float(x) for x in text.split()]
	except ValueError:
		return []
	return v[:n] if len(v) >= n else v


def _direction(e: dict) -> "np.ndarray":
	"""HLRAD's light_spot / light_environment direction: yaw from angle(s), pitch from "pitch"
	(z = sin(pitch), so -90 points straight down), angle -1 up and -2 down."""
	ang = _vec(e.get("angles", "0 0 0")) or [0.0, 0.0, 0.0]
	angle = float(e.get("angle", ang[1] if len(ang) > 1 else 0) or 0)
	if angle == -1:
		return np.array([0.0, 0.0, 1.0])
	if angle == -2:
		return np.array([0.0, 0.0, -1.0])
	pitch = float(e.get("pitch", 0) or 0) or ang[0]
	y, p = math.radians(angle), math.radians(pitch)
	return np.array([math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), math.sin(p)])


def _light_value(e: dict, key: str = "_light", default=(255.0, 255.0, 255.0, 200.0)) -> tuple["np.ndarray", float]:
	"""(linear colour 0..1, brightness) from "_light" "r g b i", "r g b" or "i", or the "light" key."""
	v = _vec(e.get(key, ""), 4)
	if len(v) >= 4:
		col, b = np.array(v[:3]), v[3]
	elif len(v) == 3:
		col, b = np.array(v), max(v)
		col = col / max(1e-6, b) * 255
	elif len(v) == 1:
		col, b = np.array(default[:3]), v[0]
	elif e.get("light"):
		col, b = np.array(default[:3]), float(_vec(e["light"], 1)[0] if _vec(e["light"], 1) else default[3])
	else:
		col, b = np.array(default[:3]), default[3]
	from .hdr import srgb_to_linear

	return srgb_to_linear(np.clip(col, 0, 255) / 255).astype(np.float64), float(b)


def parse_rad(text: str) -> dict:
	"""lights.rad: one `TEXNAME r g b intensity` per line ('//' comments)."""
	out = {}
	for line in text.splitlines():
		line = line.split("//", 1)[0].strip()
		parts = line.split()
		if len(parts) < 4:
			continue
		try:
			vals = [float(x) for x in parts[1:5]]
		except ValueError:
			continue
		if len(vals) == 3:
			vals.append(max(vals))
		out[parts[0].lower()] = tuple(vals)
	return out


def texlights(opts: GiOptions) -> dict:
	tl: dict = {}
	for p in opts.rad:
		try:
			tl.update(parse_rad(Path(p).read_text(encoding="latin-1")))
		except OSError:
			pass
	for k, v in (opts.texlights or {}).items():
		vals = [float(x) for x in (v.split() if isinstance(v, str) else v)]
		if len(vals) == 3:
			vals.append(max(vals))
		tl[k.lower()] = tuple(vals[:4])
	return tl


def entity_lights(scene: Scene, opts: GiOptions) -> tuple[list[Light], "np.ndarray", list[str]]:
	"""Point, spot and sun lights, the sky colour, and notes (switchable styles are baked on)."""
	lights: list[Light] = []
	sky = np.zeros(3)
	notes = []
	styled = 0
	for e in scene.entities:
		cls = e.get("classname", "")
		if cls not in ("light", "light_spot", "light_environment"):
			continue
		if e.get("targetname") or e.get("style", "0") not in ("", "0"):
			styled += 1
		col, b = _light_value(e)
		if cls == "light_environment":
			d = _direction(e)
			lights.append(Light("sun", np.zeros(3), col * b * K_SUN * opts.sun_scale, d / max(1e-9, np.linalg.norm(d)),
				name=cls))
			if e.get("_diffuse_light"):
				sc, sb = _light_value(e, "_diffuse_light")
				sky = sc * sb * K_SUN * opts.sky_scale
			else:
				sky = col * b * K_SUN * opts.sky_default * opts.sky_scale
			continue
		org = np.array(_vec(e.get("origin", "0 0 0")) or [0, 0, 0], np.float64)
		fade = float(e.get("_fade", 1.0) or 1.0)
		if cls == "light_spot":
			d = _direction(e)
			inner = float(e.get("_cone", 30) or 30)
			outer = float(e.get("_cone2", max(inner, 45)) or max(inner, 45))
			outer = max(outer, inner)
			lights.append(Light("spot", org, col * b * K_POINT * opts.light_scale, d / max(1e-9, np.linalg.norm(d)),
				math.cos(math.radians(inner)), math.cos(math.radians(outer)), fade=fade, name=cls))
		else:
			lights.append(Light("point", org, col * b * K_POINT * opts.light_scale, fade=fade, name=cls))
	if styled:
		notes.append(f"{styled} switchable or styled light(s) are baked on in style 0 (light styles are not separated)")
	return lights, sky, notes


# -- luxels ---------------------------------------------------------------------------------------


@dataclass
class Grid:
	"""Every lit face's luxels, flattened: face f owns rows start[f]:start[f]+w*h."""
	faces: list[Face]
	start: "np.ndarray"
	pos: "np.ndarray"	# n×3 sample position (inside the face, 1 unit off it)
	normal: "np.ndarray"	# n×3
	face: "np.ndarray"	# n face index (into faces)
	inside: "np.ndarray"	# n bool: the luxel centre lies on the face (not pulled in)

	@property
	def n(self) -> int:
		return len(self.pos)


def _luxel_world(f: Face, u: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
	"""World points on f's plane with texture coords s = (mins_s + u)·16, t = (mins_t + v)·16."""
	m = np.vstack((f.vecs[0, :3], f.vecs[1, :3], f.normal))
	b = np.stack(((f.mins[0] + u) * LUXEL - f.vecs[0, 3], (f.mins[1] + v) * LUXEL - f.vecs[1, 3],
		np.full(u.shape, f.dist)), -1)
	return b @ np.linalg.inv(m).T


def _pull_inside(f: Face, p: "np.ndarray", nudge: float = 0.5) -> tuple["np.ndarray", "np.ndarray"]:
	"""Points outside the face polygon moved to its nearest point, then a little towards the centre."""
	v = f.verts
	c = v.mean(0)
	a, b = v, np.roll(v, -1, 0)
	en = np.cross(f.normal, b - a)
	en *= np.sign(np.einsum("ij,ij->i", en, c - a))[:, None]
	side = np.einsum("pej,ej->pe", p[:, None] - a[None], en)
	inside = (side >= -1e-3).all(1)
	if inside.all():
		return p, inside
	q = p[~inside]
	ab = b - a
	t = np.clip(np.einsum("pej,ej->pe", q[:, None] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
	near = a[None] + t[..., None] * ab[None]
	dd = ((near - q[:, None]) ** 2).sum(-1)
	best = near[np.arange(len(q)), dd.argmin(1)]
	to_c = c - best
	ln = np.linalg.norm(to_c, axis=1, keepdims=True)
	best = best + to_c / np.maximum(ln, 1e-9) * np.minimum(ln, nudge)
	out = p.copy()
	out[~inside] = best
	return out, inside


def luxel_grid(scene: Scene, offset: float = 1.0) -> Grid:
	lit = [f for f in scene.faces if f.lit]
	start = np.zeros(len(lit) + 1, np.int64)
	pos, nrm, fid, ins = [], [], [], []
	for i, f in enumerate(lit):
		vv, uu = np.mgrid[0:f.h, 0:f.w]
		p = _luxel_world(f, uu.ravel().astype(np.float64), vv.ravel().astype(np.float64))
		p, inside = _pull_inside(f, p)
		pos.append(p + f.normal * offset)
		nrm.append(np.broadcast_to(f.normal, p.shape))
		fid.append(np.full(len(p), i))
		ins.append(inside)
		start[i + 1] = start[i] + len(p)
	cat = (lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape))
	return Grid(lit, start, cat(pos, (0, 3)), cat(nrm, (0, 3)), cat(fid, (0,)).astype(np.int64), cat(ins, (0,)).astype(bool))


# -- the ray mesh ---------------------------------------------------------------------------------


@dataclass
class World:
	mesh: raytrace.Mesh
	tri_face: "np.ndarray"	# triangle → index into scene.faces
	sky_tri: "np.ndarray"	# bool per triangle
	has_sky: bool


def world_mesh(scene: Scene, opts: GiOptions) -> World:
	tris, owner = [], []
	for k, f in enumerate(scene.faces):
		if f.model != 0 and not opts.entity_shadows:
			continue
		if f.tex[:1] in ("!", "*") or f.tex in ("aaatrigger", "clip", "null", "origin"):
			continue
		v = f.verts
		for j in range(1, len(v) - 1):
			tris.append((v[0], v[j], v[j + 1]))
			owner.append(k)
	t = np.array(tris, np.float64).reshape(-1, 3, 3)
	owner_a = np.array(owner, np.int64)
	sky = np.array([scene.faces[k].sky for k in owner], bool)
	return World(raytrace.Mesh(t), owner_a, sky, bool(sky.any()))


# -- patches --------------------------------------------------------------------------------------


@dataclass
class Patches:
	pos: "np.ndarray"	# m×3
	normal: "np.ndarray"
	area: "np.ndarray"
	albedo: "np.ndarray"	# m×3
	emit: "np.ndarray"	# m×3 exitance of texture lights
	of_luxel: "np.ndarray"	# luxel → patch
	face_first: dict[int, int]	# scene face index → its first patch
	face_dims: dict[int, tuple[int, int, int]]	# scene face index → (patch luxels P, patches across, down)

	@property
	def m(self) -> int:
		return len(self.pos)


def make_patches(scene: Scene, grid: Grid, opts: GiOptions, tl: dict) -> Patches:
	P = max(1, int(round(opts.patch / LUXEL)))
	pos, nrm, area, alb, emit = [], [], [], [], []
	of_luxel = np.zeros(grid.n, np.int64)
	first, dims = {}, {}
	for i, f in enumerate(grid.faces):
		pw, ph = -(-f.w // P), -(-f.h // P)
		base = len(pos)
		first[f.index] = base
		dims[f.index] = (P, pw, ph)
		a, b = grid.start[i], grid.start[i + 1]
		vv, uu = np.mgrid[0:f.h, 0:f.w]
		pid = (vv // P) * pw + (uu // P)
		pid = pid.ravel()
		of_luxel[a:b] = base + pid
		cnt = np.bincount(pid, minlength=pw * ph).astype(np.float64)
		inside = np.bincount(pid, weights=grid.inside[a:b].astype(np.float64), minlength=pw * ph)
		sp = np.stack([np.bincount(pid, weights=grid.pos[a:b, c], minlength=pw * ph) for c in range(3)], -1)
		sp /= np.maximum(cnt, 1)[:, None]
		share = inside / max(1.0, inside.sum()) if inside.sum() else cnt / cnt.sum()
		col = scene.albedo.get(f.tex)
		col = np.full(3, opts.default_albedo) if col is None else np.asarray(col, np.float64)
		col = np.clip(col * opts.reflect, 0, opts.max_albedo)
		e = np.zeros(3)
		t = tl.get(f.tex)
		if t is not None:
			from .hdr import srgb_to_linear

			e = srgb_to_linear(np.clip(np.array(t[:3]), 0, 255) / 255).astype(np.float64) * t[3] * K_TEX * opts.texlight_scale
		for k in range(pw * ph):
			pos.append(sp[k])
			nrm.append(f.normal)
			area.append(f.area * share[k])
			alb.append(col)
			emit.append(e)
	z = (lambda xs, d: np.array(xs, np.float64).reshape(-1, d))
	return Patches(z(pos, 3), z(nrm, 3), np.array(area, np.float64), z(alb, 3), z(emit, 3), of_luxel, first, dims)


def hit_patches(scene: Scene, world: World, patches: Patches, hits: raytrace.Hits, orig, dirs) -> tuple["np.ndarray", "np.ndarray"]:
	"""Patch index of each hit (-1 for none or a face without patches) and whether it was sky."""
	n = len(hits.t)
	out = np.full(n, -1, np.int64)
	sky = np.zeros(n, bool)
	ok = hits.hit
	if not ok.any():
		return out, sky
	tri = hits.tri[ok]
	sky[ok] = world.sky_tri[tri]
	fidx = world.tri_face[tri]
	pts = orig[ok] + dirs[ok] * hits.t[ok][:, None]
	sel = np.flatnonzero(ok)
	for fi in np.unique(fidx):
		f = scene.faces[int(fi)]
		if f.index not in patches.face_first:
			continue
		m = fidx == fi
		P, pw, ph = patches.face_dims[f.index]
		st = pts[m] @ f.vecs[:, :3].T + f.vecs[:, 3]
		u = np.clip(np.floor(st[:, 0] / LUXEL - f.mins[0] + 0.5), 0, f.w - 1).astype(np.int64)
		v = np.clip(np.floor(st[:, 1] / LUXEL - f.mins[1] + 0.5), 0, f.h - 1).astype(np.int64)
		out[sel[m]] = patches.face_first[f.index] + (v // P) * pw + (u // P)
	return out, sky


# -- direct light ---------------------------------------------------------------------------------


def _shadow_pass(world: World, p: "np.ndarray", to: "np.ndarray", dist: "np.ndarray") -> "np.ndarray":
	"""True where the segment p → p + to·dist is clear."""
	if not len(p):
		return np.zeros(0, bool)
	return ~world.mesh.occluded(p, to, np.maximum(dist - 1.0, 0.0))


@dataclass
class Vis:
	"""The map's own PVS: which leaves each leaf can see (HLRAD lights only what a light's leaf sees)."""
	leaf_of_luxel: "np.ndarray"
	rows: dict
	planes: "np.ndarray"
	nodes: "np.ndarray"	# n×3: plane, child0, child1 (child < 0: leaf −child−1)
	head: int
	vis: bytes
	visofs: "np.ndarray"
	visleafs: int

	def leaf(self, pts: "np.ndarray") -> "np.ndarray":
		p = np.asarray(pts, np.float64).reshape(-1, 3)
		node = np.full(len(p), self.head, np.int64)
		out = np.zeros(len(p), np.int64)
		live = np.arange(len(p))
		guard = 0
		while len(live) and guard < 4096:
			nd = node[live]
			pl = self.planes[self.nodes[nd, 0]]
			d = np.einsum("ij,ij->i", p[live], pl[:, :3]) - pl[:, 3]
			nxt = np.where(d >= 0, self.nodes[nd, 1], self.nodes[nd, 2])
			node[live] = nxt
			done = nxt < 0
			out[live[done]] = -nxt[done] - 1
			live = live[~done]
			guard += 1
		return out

	def row(self, leaf: int) -> "np.ndarray":
		"""bool per leaf: visible from `leaf` (leaf 0, solid, and leaves without data see everything)."""
		if leaf in self.rows:
			return self.rows[leaf]
		n = len(self.visofs)
		ofs = int(self.visofs[leaf]) if 0 < leaf < n else -1
		if ofs < 0 or not self.vis:
			r = np.ones(n, bool)
		else:
			from .source_bsp import decompress_vis

			nbytes = (self.visleafs + 7) // 8
			bits = np.unpackbits(np.frombuffer(decompress_vis(self.vis, ofs, nbytes)[:nbytes], np.uint8),
				bitorder="little")[:self.visleafs]
			r = np.zeros(n, bool)
			r[1:1 + len(bits)] = bits.astype(bool)
			r[0] = True
			r[leaf] = True
		self.rows[leaf] = r
		return r


def map_vis(scene: Scene, grid: Grid) -> "Vis | None":
	lumps = scene.lumps
	nraw, lraw = lumps[5], lumps[10]
	if len(nraw) < 24 or len(lraw) < 56 or not lumps[4] or len(lumps[L_MODELS]) < 64:
		return None
	nn = len(nraw) // 24
	nodes = np.zeros((nn, 3), np.int64)
	for i in range(nn):
		pl, c0, c1 = struct.unpack_from("<ihh", nraw, i * 24)
		nodes[i] = (pl, c0, c1)
	nl = len(lraw) // 28
	visofs = np.array([struct.unpack_from("<i", lraw, i * 28 + 4)[0] for i in range(nl)], np.int64)
	planes = np.frombuffer(lumps[L_PLANES], "<f4", len(lumps[L_PLANES]) // 20 * 5).reshape(-1, 5)[:, :4].astype(np.float64)
	head = struct.unpack_from("<i", lumps[L_MODELS], 36)[0]
	visleafs = struct.unpack_from("<i", lumps[L_MODELS], 52)[0]
	if not (0 <= head < nn) or nodes[:, 0].max(initial=0) >= len(planes):
		return None
	v = Vis(np.zeros(0, np.int64), {}, planes, nodes, head, lumps[4], visofs, max(0, visleafs))
	v.leaf_of_luxel = v.leaf(grid.pos) if grid.n else np.zeros(0, np.int64)
	return v


def _face_bounds(grid: Grid) -> tuple["np.ndarray", "np.ndarray"]:
	c = np.zeros((len(grid.faces), 3))
	r = np.zeros(len(grid.faces))
	for i in range(len(grid.faces)):
		p = grid.pos[grid.start[i]:grid.start[i + 1]]
		c[i] = (p.min(0) + p.max(0)) / 2
		r[i] = np.linalg.norm(p - c[i], axis=1).max(initial=0)
	return c, r


def direct(grid: Grid, world: World, lights: list[Light], opts: GiOptions, vis: "Vis | None" = None,
		cutoff: float = 1 / 4096, chunk: int = 200000) -> "np.ndarray":
	"""Direct light per luxel with shadow rays. Each light only looks at faces in range (its light
	under `cutoff` beyond that), in front of it and, with a PVS, in leaves its own leaf can see."""
	out = np.zeros((grid.n, 3))
	if not grid.n:
		return out
	fc, fr = _face_bounds(grid)
	fn = np.array([f.normal for f in grid.faces]).reshape(-1, 3)
	counts = grid.start[1:] - grid.start[:-1]
	for L in lights:
		if L.kind == "sun":
			d = -L.direction
			cos = grid.normal @ d
			idx = np.flatnonzero(cos > 0)
			for a in range(0, len(idx), chunk):
				k = idx[a:a + chunk]
				hits = world.mesh.trace(grid.pos[k], np.broadcast_to(d, (len(k), 3)), 65536.0)
				vis_ = world.sky_tri[np.maximum(hits.tri, 0)] & hits.hit if world.has_sky else ~hits.hit
				out[k[vis_]] += cos[k[vis_], None] * L.color
			continue
		peak = float(L.color.max())
		reach = math.sqrt(peak * (L.area / math.pi if L.kind == "surface" else 1 / L.fade) / cutoff)
		rel = L.origin - fc
		ok = (np.linalg.norm(rel, axis=1) - fr <= reach) & (np.einsum("ij,ij->i", rel, fn) > -1.0)
		if not ok.any():
			continue
		sel = np.flatnonzero(ok)
		c = counts[sel]
		idx = np.repeat(grid.start[sel] - (np.cumsum(c) - c), c) + np.arange(int(c.sum()))
		if vis is not None:
			row = vis.row(int(vis.leaf(L.origin[None])[0]))
			idx = idx[row[np.minimum(vis.leaf_of_luxel[idx], len(row) - 1)]]
		dv = L.origin - grid.pos[idx]
		d2 = (dv * dv).sum(1)
		dist = np.sqrt(np.maximum(d2, 1e-9))
		to = dv / dist[:, None]
		cos = np.einsum("ij,ij->i", grid.normal[idx], to)
		if L.kind == "surface":
			cos_e = -(to @ L.direction)
			val = np.clip(cos, 0, None) * np.clip(cos_e, 0, None) * L.area / (math.pi * (d2 + L.area / math.pi))
		else:
			val = np.clip(cos, 0, None) / (np.maximum(d2, 1.0) * L.fade)
			if L.kind == "spot":
				sd = -(to @ L.direction)
				span = max(1e-6, L.cone - L.cone2)
				val *= np.clip((sd - L.cone2) / span, 0, 1)
		keep = np.flatnonzero(val * peak > cutoff)
		for a in range(0, len(keep), chunk):
			k = keep[a:a + chunk]
			vis_ = _shadow_pass(world, grid.pos[idx[k]], to[k], dist[k])
			out[idx[k[vis_]]] += val[k[vis_], None] * L.color
	return out


def surface_lights(scene: Scene, grid: Grid, patches: Patches, step: int = 2) -> list[Light]:
	"""Texture lights as small area lights: one every `step` luxels of each emissive face, on the face."""
	out = []
	for i, f in enumerate(grid.faces):
		base = patches.face_first[f.index]
		e = patches.emit[base]
		if e.max() <= 0:
			continue
		a = grid.start[i]
		vv, uu = np.mgrid[0:f.h, 0:f.w]
		pick = ((vv % step) == step // 2) | (f.h <= step)
		pick &= ((uu % step) == step // 2) | (f.w <= step)
		inside = grid.inside[a:a + f.w * f.h].reshape(f.h, f.w)
		sel = np.flatnonzero((pick & inside).ravel())
		if not len(sel):
			sel = np.flatnonzero(inside.ravel())[:1] if inside.any() else np.array([0])
		area = f.area / len(sel)
		for k in sel:
			out.append(Light("surface", grid.pos[a + k] - f.normal * 0.5, e.copy(), f.normal.copy(), area=area,
				name=f"texlight {f.tex}"))
	return out


# -- indirect -------------------------------------------------------------------------------------


@dataclass
class Transfer:
	src: "np.ndarray"	# receiving patch
	dst: "np.ndarray"	# patch seen
	w: "np.ndarray"	# share of the receiver's hemisphere
	sky: "np.ndarray"	# per receiving patch: share of rays reaching sky


def transfer(scene: Scene, world: World, patches: Patches, opts: GiOptions, rays: int,
		pts: "np.ndarray | None" = None, normals: "np.ndarray | None" = None, chunk_pts: int = 2048) -> Transfer:
	"""Cosine-weighted hemisphere rays from each point (patches by default): who sees whom."""
	pts = patches.pos if pts is None else pts
	normals = patches.normal if normals is None else normals
	local = raytrace.cosine_hemisphere(rays, opts.seed)
	src_l, dst_l = [], []
	sky = np.zeros(len(pts))
	for a in range(0, len(pts), chunk_pts):
		b = min(len(pts), a + chunk_pts)
		dirs = raytrace.hemisphere_rays(normals[a:b], local, opts.seed + a).reshape(-1, 3)
		o = np.repeat(pts[a:b], rays, 0)
		hits = world.mesh.trace(o, dirs, 65536.0)
		pid, s = hit_patches(scene, world, patches, hits, o, dirs)
		who = np.repeat(np.arange(a, b), rays)
		if not world.has_sky:
			s = ~hits.hit	# a leak or an open map: treat escaping rays as sky
		sky[a:b] = np.bincount(who[s] - a, minlength=b - a) / rays
		m = pid >= 0
		src_l.append(who[m])
		dst_l.append(pid[m])
	src = np.concatenate(src_l) if src_l else np.zeros(0, np.int64)
	dst = np.concatenate(dst_l) if dst_l else np.zeros(0, np.int64)
	if len(src):
		key = src * max(1, patches.m) + dst
		uk, cnt = np.unique(key, return_counts=True)
		src, dst, w = uk // max(1, patches.m), uk % max(1, patches.m), cnt / rays
	else:
		w = np.zeros(0)
	return Transfer(src, dst, w, sky)


def gather(tr: Transfer, out: "np.ndarray", n: int) -> "np.ndarray":
	return np.stack([np.bincount(tr.src, weights=tr.w * out[tr.dst, c], minlength=n) for c in range(3)], -1)


def ambient_occlusion(grid: Grid, world: World, opts: GiOptions, chunk_pts: int = 4096) -> "np.ndarray":
	if opts.ao <= 0 or opts.ao_rays <= 0 or not grid.n:
		return np.ones(grid.n)
	local = raytrace.cosine_hemisphere(opts.ao_rays, opts.seed + 7)
	occ = np.zeros(grid.n)
	for a in range(0, grid.n, chunk_pts):
		b = min(grid.n, a + chunk_pts)
		dirs = raytrace.hemisphere_rays(grid.normal[a:b], local, opts.seed + 99 + a).reshape(-1, 3)
		o = np.repeat(grid.pos[a:b], opts.ao_rays, 0)
		hits = world.mesh.trace(o, dirs, opts.ao_radius)
		# a sky face is open air, not an occluder
		blocked = hits.hit & ~world.sky_tri[np.maximum(hits.tri, 0)]
		# nearer hits count more (a soft falloff instead of a hard radius)
		wgt = np.where(blocked, 1 - hits.t / opts.ao_radius, 0.0)
		occ[a:b] = wgt.reshape(-1, opts.ao_rays).mean(1)
	return np.clip(1 - opts.ao * occ * 2, 0, 1)


def _interp_patch(grid: Grid, patches: Patches, val: "np.ndarray") -> "np.ndarray":
	"""Bilinear across each face's patch grid (patch centres), per luxel."""
	out = np.zeros((grid.n, val.shape[1]))
	for i, f in enumerate(grid.faces):
		a, b = grid.start[i], grid.start[i + 1]
		P, pw, ph = patches.face_dims[f.index]
		base = patches.face_first[f.index]
		g = val[base:base + pw * ph].reshape(ph, pw, -1)
		vv, uu = np.mgrid[0:f.h, 0:f.w]
		cu = np.clip((uu - (P - 1) / 2) / P, 0, pw - 1).ravel()
		cv = np.clip((vv - (P - 1) / 2) / P, 0, ph - 1).ravel()
		u0, v0 = np.floor(cu).astype(int), np.floor(cv).astype(int)
		u1, v1 = np.minimum(u0 + 1, pw - 1), np.minimum(v0 + 1, ph - 1)
		fu, fv = (cu - u0)[:, None], (cv - v0)[:, None]
		out[a:b] = (g[v0, u0] * (1 - fu) + g[v0, u1] * fu) * (1 - fv) + (g[v1, u0] * (1 - fu) + g[v1, u1] * fu) * fv
	return out


# -- the bake -------------------------------------------------------------------------------------


@dataclass
class Bake:
	scene: Scene
	grid: Grid
	light: "np.ndarray"	# n×3 linear light per luxel (1.0 = full bright)
	stats: dict
	direct: "np.ndarray | None" = None
	indirect: "np.ndarray | None" = None

	def face_light(self, i: int) -> "np.ndarray":
		f = self.grid.faces[i]
		a, b = self.grid.start[i], self.grid.start[i + 1]
		return self.light[a:b].reshape(f.h, f.w, 3)


def _existing(scene: Scene, grid: Grid) -> "np.ndarray | None":
	"""The map's own style-0 lighting per luxel (bytes), where the grid matches; NaN elsewhere."""
	lump = scene.lumps[L_LIGHTING]
	if not lump:
		return None
	fl = scene.lumps[L_FACES]
	out = np.full((grid.n, 3), np.nan)
	any_ = False
	raw = np.frombuffer(lump, np.uint8)
	for i, f in enumerate(grid.faces):
		lo = struct.unpack_from("<i", fl, f.index * 20 + 16)[0]
		st0 = fl[f.index * 20 + 12]
		n = f.w * f.h * 3
		if lo < 0 or st0 == 255 or lo + n > len(raw):
			continue
		out[grid.start[i]:grid.start[i + 1]] = raw[lo:lo + n].reshape(-1, 3)
		any_ = True
	return out if any_ else None


def bake(data: bytes, opts: GiOptions | None = None) -> Bake:
	opts = opts or GiOptions()
	t0 = time.time()
	scene = load(data, opts)
	tl = texlights(opts)
	grid = luxel_grid(scene)
	world = world_mesh(scene, opts)
	lights, sky, notes = entity_lights(scene, opts)
	patches = make_patches(scene, grid, opts, tl)
	slights = surface_lights(scene, grid, patches, max(1, int(opts.texlight_step))) if tl else []
	vis = map_vis(scene, grid) if opts.use_vis else None
	opts.say(f"ray caster: {raytrace.backend()}")
	opts.say(f"{len(grid.faces)} lit faces, {grid.n} luxels, {patches.m} patches, {world.mesh.count_tris} triangles, "
		f"{len(lights)} lights + {len(slights)} emissive samples")
	t1 = time.time()
	dl = direct(grid, world, lights + slights, opts, vis, opts.cutoff)
	opts.say(f"direct light: {time.time() - t1:.1f} s")
	# luxels of an emissive face show their own glow
	self_emit = patches.emit[patches.of_luxel]
	# patch irradiance from direct light: mean over its luxels
	cnt = np.bincount(patches.of_luxel, minlength=patches.m).astype(np.float64)
	pd = np.stack([np.bincount(patches.of_luxel, weights=dl[:, c], minlength=patches.m) for c in range(3)], -1)
	pd /= np.maximum(cnt, 1)[:, None]
	t2 = time.time()
	need_tr = opts.bounces > 0 or sky.max() > 0
	tr = transfer(scene, world, patches, opts, opts.rays) if need_tr and patches.m else None
	sky_in = (tr.sky[:, None] * sky[None]) if tr is not None else np.zeros((patches.m, 3))
	ind = np.zeros((patches.m, 3))
	bounce_energy = []
	for _b in range(opts.bounces if tr is not None else 0):
		out = patches.albedo * (pd + sky_in + ind)
		new = gather(tr, out, patches.m)
		bounce_energy.append(float((new * patches.area[:, None]).sum()))
		ind = new
	if tr is not None:
		opts.say(f"radiosity: {len(tr.w)} links, {opts.bounces} bounce(s), {time.time() - t2:.1f} s")
	t3 = time.time()
	if opts.gather > 0 and tr is not None:
		lt = transfer(scene, world, patches, opts, opts.gather, grid.pos, grid.normal)
		out = patches.albedo * (pd + sky_in + ind) if opts.bounces > 0 else np.zeros((patches.m, 3))
		indirect = gather(lt, out, grid.n) + lt.sky[:, None] * sky[None]
		opts.say(f"final gather: {opts.gather} rays per luxel, {time.time() - t3:.1f} s")
	else:
		indirect = _interp_patch(grid, patches, ind + sky_in) if patches.m else np.zeros((grid.n, 3))
	ao = ambient_occlusion(grid, world, opts)
	light = dl + self_emit + indirect * ao[:, None] + np.asarray(opts.ambient, np.float64)[None]
	light *= 2.0 ** opts.exposure
	stats = {"faces": len(grid.faces), "luxels": int(grid.n), "patches": int(patches.m),
		"triangles": int(world.mesh.count_tris), "lights": {k: sum(1 for L in lights if L.kind == k) for k in ("point", "spot", "sun")},
		"emissive_samples": len(slights), "texlights": sorted(t for t in tl if any(f.tex == t for f in scene.faces)),
		"sky": [round(float(x), 4) for x in sky], "bounces": opts.bounces, "bounce_energy": [round(x, 2) for x in bounce_energy],
		"ao": opts.ao, "gather": opts.gather, "raytracer": raytrace.backend(), "notes": notes + scene.warnings}
	if opts.calibrate:
		old = _existing(scene, grid)
		if old is None:
			stats["notes"].append("calibrate: the map has no lighting to match; left as baked")
		else:
			ol = (old / 255.0) ** (1 / opts.gamma)
			m = np.isfinite(ol).all(1) & (ol.max(1) > 0.02) & (light.max(1) > 1e-4)
			if m.sum() > 16:
				ratio = float(np.median(ol[m].mean(1) / light[m].mean(1)))
				light *= ratio
				stats["calibration"] = round(ratio, 4)
				opts.say(f"calibrated ×{ratio:.3f} to the map's existing lighting")
	stats["seconds"] = round(time.time() - t0, 2)
	return Bake(scene, grid, light, stats, dl, indirect)


# -- GoldSrc lighting lump ------------------------------------------------------------------------


def to_bytes(light: "np.ndarray", gamma: float) -> "np.ndarray":
	return np.clip(np.clip(light, 0, 1) ** gamma * 255 + 0.5, 0, 255).astype(np.uint8)


def relit_bsp(b: Bake, gamma: float | None = None) -> bytes:
	"""The map with its LIGHTING lump and face light offsets replaced (style 0 only)."""
	g = 0.55 if gamma is None else gamma
	sc = b.scene
	faces = bytearray(sc.lumps[L_FACES])
	lit = {f.index: i for i, f in enumerate(b.grid.faces)}
	light = bytearray()
	for k in range(len(faces) // 20):
		i = lit.get(k)
		if i is None:
			struct.pack_into("<4Bi", faces, k * 20 + 12, 255, 255, 255, 255, -1)
			continue
		struct.pack_into("<4Bi", faces, k * 20 + 12, 0, 255, 255, 255, len(light))
		light += to_bytes(b.face_light(i), g).tobytes()
	lumps = list(sc.lumps)
	lumps[L_FACES] = bytes(faces)
	lumps[L_LIGHTING] = bytes(light)
	out = bytearray(struct.pack("<i", 30) + bytes(15 * 8))
	for i in sc.order:
		out += bytes(-len(out) % 4)
		struct.pack_into("<ii", out, 4 + 8 * i, len(out), len(lumps[i]))
		out += lumps[i]
	return bytes(out)


# -- GX atlas -------------------------------------------------------------------------------------


@dataclass
class AtlasFace:
	face: int	# BSP face index
	w: int	# stored grid
	h: int
	src_w: int	# engine grid
	src_h: int
	page: int = -1
	x: int = 0	# first stored luxel (inside the padding)
	y: int = 0


@dataclass
class Atlas:
	pages: list["np.ndarray"]	# H×W×4 uint8 (light / scale)
	faces: list[AtlasFace]
	scale: int
	stats: dict
	sizes: list[tuple[int, int]] = field(default_factory=list)
	tone: str = "port"


def _psnr(a: "np.ndarray", b: "np.ndarray") -> float:
	d = float(np.mean((a.astype(np.float64) - b) ** 2))
	return float("inf") if d == 0 else 10 * math.log10(255 ** 2 / d)


def _resample(img: "np.ndarray", w: int, h: int) -> "np.ndarray":
	"""Bilinear with the corners pinned (how a w×h grid spans the same face)."""
	sh, sw = img.shape[:2]
	ys = np.linspace(0, sh - 1, h) if h > 1 else np.full(1, (sh - 1) / 2)
	xs = np.linspace(0, sw - 1, w) if w > 1 else np.full(1, (sw - 1) / 2)
	y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
	y1, x1 = np.minimum(y0 + 1, sh - 1), np.minimum(x0 + 1, sw - 1)
	fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
	a = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
	c = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
	return a * (1 - fy) + c * fy


def shrink_face(lm: "np.ndarray", flat_tol: float, min_db: float) -> "np.ndarray":
	"""1×1 when flat, half resolution when it interpolates back above min_db, else as is (8-bit domain)."""
	h, w = lm.shape[:2]
	f = lm.astype(np.float64)
	if np.abs(f - f.mean((0, 1))).max() <= flat_tol:
		return np.round(f.mean((0, 1)))[None, None].astype(np.uint8)
	sw, sh = max(2, -(-w // 2)), max(2, -(-h // 2))
	if (sw, sh) != (w, h):
		small = np.round(_resample(f, sw, sh))
		if _psnr(_resample(small, w, h), f) >= min_db:
			return np.clip(small, 0, 255).astype(np.uint8)
	return lm


def _align(n: int, a: int) -> int:
	return -(-n // a) * a


def skyline(sizes: list[tuple[int, int]], pw: int, ph: int, align: int = 1) -> list[tuple[int, int, int]]:
	"""(page, x, y) per item: bottom-left skyline, tallest first, every page tried, x on `align`.
	The numpy version of rectpack.pack_skyline (a sliding-window max per placement)."""
	from numpy.lib.stride_tricks import sliding_window_view

	order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
	cols: list[np.ndarray] = []
	out: list[tuple[int, int, int]] = [(0, 0, 0)] * len(sizes)
	for i in order:
		w, h = sizes[i]
		if w > pw or h > ph:
			raise ValueError(f"a {w}×{h} lightmap does not fit a {pw}×{ph} page")
		done = False
		for p, col in enumerate(cols):
			mx = sliding_window_view(col, w).max(1)[::align]
			x = int(np.argmin(mx))
			y = int(mx[x])
			if y + h <= ph:
				x *= align
				col[x:x + w] = y + h
				out[i] = (p, x, y)
				done = True
				break
		if not done:
			col = np.zeros(pw, np.int64)
			col[:w] = h
			cols.append(col)
			out[i] = (len(cols) - 1, 0, 0)
	return out


def atlas(b: Bake, page: int = 256, pad: int = 1, align: int = 4, overbright="auto", gamma: float = 1.0,
		shrink: bool = True, flat_tol: float = 2.0, min_db: float = 40.0, max_scale: int = 4, tone: str = "port",
		bsp_gamma: float = 0.55) -> Atlas:
	"""Every face's light in power-of-two pages, stored as value / scale for a TEV stage scaled ×scale.

	tone "port": value = 2·light^bsp_gamma, which is what the port shows for the relit BSP today
	(raw lightmap bytes × 2 in TEV), so both outputs look the same, and light above full bright is
	kept (scale 4) instead of clipping. tone "linear": value = light, stored ^(1/gamma)."""
	from . import hdr

	if tone not in TONES:
		raise ValueError(f"tone is one of {', '.join(TONES)}")
	light = np.maximum(b.light, 0)
	if tone == "port":
		light = 2.0 * light ** bsp_gamma
		gamma = 1.0
	if overbright == "auto":
		scale = hdr.pick_overbright(light) if len(light) else 1
		scale = min(scale, max_scale)
	else:
		scale = int(overbright)
	if scale not in (1, 2, 4):
		raise ValueError("overbright is 1, 2, 4 or auto")
	clipped = float((light.max(1) > scale).mean()) if len(light) else 0.0
	faces, blocks = [], []
	for i, f in enumerate(b.grid.faces):
		val = light[b.grid.start[i]:b.grid.start[i + 1]].reshape(f.h, f.w, 3)
		lm = (np.clip(val / scale, 0, 1) ** (1 / gamma) * 255 + 0.5).astype(np.uint8)
		st = shrink_face(lm, flat_tol, min_db) if shrink else lm
		faces.append(AtlasFace(f.index, st.shape[1], st.shape[0], f.w, f.h))
		blocks.append(st)
	sizes = [(_align(af.w + 2 * pad, align), _align(af.h + 2 * pad, align)) for af in faces]
	page = max(align, _align(page, align))
	place = skyline(sizes, page, page, align) if sizes else []
	npages = max((p for p, _x, _y in place), default=-1) + 1
	dims = [(page, page)] * npages
	# the last page: the smallest power-of-two rectangle that still holds what landed on it
	if npages:
		last = [i for i, (p, _x, _y) in enumerate(place) if p == npages - 1]
		best = None
		for lw in (page >> k for k in range(0, 8)):
			for lh in (page >> k for k in range(0, 8)):
				if lw < align or lh < align or (best and lw * lh >= best[0] * best[1]):
					continue
				if any(sizes[i][0] > lw or sizes[i][1] > lh for i in last):
					continue
				sub = skyline([sizes[i] for i in last], lw, lh, align)
				if all(p == 0 for p, _x, _y in sub):
					best = (lw, lh, sub)
		if best:
			lw, lh, sub = best
			for i, (_p, x, y) in zip(last, sub):
				place[i] = (npages - 1, x, y)
			dims[-1] = (lw, lh)
	pages = [np.zeros((h, w, 4), np.uint8) for w, h in dims]
	for pg in pages:
		pg[..., 3] = 255
	for af, st, sz, (p, x, y) in zip(faces, blocks, sizes, place):
		rw, rh = sz
		blk = np.pad(st, ((pad, rh - st.shape[0] - pad), (pad, rw - st.shape[1] - pad), (0, 0)), mode="edge")
		pages[p][y:y + rh, x:x + rw, :3] = blk
		af.page, af.x, af.y = p, x + pad, y + pad
	used = sum(af.w * af.h for af in faces)
	stats = {"tone": tone, "scale": scale, "tev_scale": f"GX_CS_SCALE_{scale}" if scale > 1 else "GX_CS_SCALE_1",
		"clipped": round(clipped, 5), "peak": round(float(light.max(initial=0)), 3), "gamma": gamma,
		"faces": len(faces), "luxels": int(sum(f.w * f.h for f in b.grid.faces)), "stored_luxels": int(used),
		"flat_faces": sum(1 for af in faces if af.w == 1 and af.h == 1),
		"shrunk_faces": sum(1 for af in faces if (af.w, af.h) != (af.src_w, af.src_h)),
		"pages": [{"w": w, "h": h} for w, h in dims], "pad": pad, "align": align,
		"occupancy": round(used / max(1, sum(w * h for w, h in dims)), 3)}
	return Atlas(pages, faces, scale, stats, dims, tone)


def gclm_bytes(at: Atlas, face_count: int, gamma: float = 1.0) -> bytes:
	"""The face table the port reads (port/gc_lmap.c), big-endian:

	header (32 B): 'GCLM', u16 version, u16 flags (1: port tone, see atlas()), u16 faces, u16 pages, u8 tev scale, u8 pad,
		u16 gamma×256 (stored = (light / scale)^(1/gamma)), u32 page table offset,
		u32 face table offset, 8 bytes reserved
	face (12 B), one per BSP face, indexed by face number: u16 page (0xFFFF none), u16 x, u16 y,
		u16 w, u16 h, u8 src_w, u8 src_h — x/y/w/h in luxels of the stored grid
	page (4 B): u16 width, u16 height (TPL image i is page i)"""
	faces_ofs = GCLM_HEADER
	pages_ofs = faces_ofs + face_count * GCLM_FACE
	out = bytearray(pages_ofs + 4 * len(at.sizes))
	flags = GCLM_FLAG_PORT_TONE if at.tone == "port" else 0
	struct.pack_into(">4sHHHHBBHII8x", out, 0, GCLM_MAGIC, GCLM_VERSION, flags, face_count, len(at.sizes),
		at.scale, 0, max(0, min(65535, round(gamma * 256))), pages_ofs, faces_ofs)
	for k in range(face_count):
		struct.pack_into(">HHHHHBB", out, faces_ofs + k * GCLM_FACE, GCLM_NONE, 0, 0, 0, 0, 0, 0)
	for af in at.faces:
		struct.pack_into(">HHHHHBB", out, faces_ofs + af.face * GCLM_FACE, af.page, af.x, af.y, af.w, af.h,
			min(255, af.src_w), min(255, af.src_h))
	for i, (w, h) in enumerate(at.sizes):
		struct.pack_into(">HH", out, pages_ofs + 4 * i, w, h)
	return bytes(out)


def read_gclm(data: bytes) -> dict:
	magic, ver, flags, nf, np_, scale, _p, g, pofs, fofs = struct.unpack_from(">4sHHHHBBHII", data, 0)
	if magic != GCLM_MAGIC:
		raise ValueError("not a .gclm file")
	faces = []
	for k in range(nf):
		p, x, y, w, h, sw, sh = struct.unpack_from(">HHHHHBB", data, fofs + k * GCLM_FACE)
		faces.append(None if p == GCLM_NONE else {"face": k, "page": p, "x": x, "y": y, "w": w, "h": h,
			"src_w": sw, "src_h": sh})
	pages = [struct.unpack_from(">HH", data, pofs + 4 * i) for i in range(np_)]
	return {"version": ver, "flags": flags, "scale": scale, "gamma": g / 256, "faces": faces, "pages": pages}


def atlas_uv(entry: dict, s: float, t: float, page_w: int, page_h: int) -> tuple[float, float]:
	"""Atlas UV for a point whose engine lightmap coords are (s, t) luxels from the face's
	texturemins (s = (dot(v, svec) + soff) / 16 − mins). What gc_lm_uv computes in C."""
	sw, sh = entry["src_w"], entry["src_h"]
	u = s * (entry["w"] - 1) / (sw - 1) if sw > 1 and entry["w"] > 1 else (entry["w"] - 1) / 2
	v = t * (entry["h"] - 1) / (sh - 1) if sh > 1 and entry["h"] > 1 else (entry["h"] - 1) / 2
	return (entry["x"] + u + 0.5) / page_w, (entry["y"] + v + 0.5) / page_h


def sample_atlas(page: "np.ndarray", u: float, v: float) -> "np.ndarray":
	"""Bilinear lookup like GX_LINEAR with clamp (texel centres at +0.5)."""
	h, w = page.shape[:2]
	x, y = u * w - 0.5, v * h - 0.5
	x0, y0 = int(math.floor(x)), int(math.floor(y))
	fx, fy = x - x0, y - y0
	c = lambda xx, yy: page[min(max(yy, 0), h - 1), min(max(xx, 0), w - 1)].astype(np.float64)
	return (c(x0, y0) * (1 - fx) + c(x0 + 1, y0) * fx) * (1 - fy) + (c(x0, y0 + 1) * (1 - fx) + c(x0 + 1, y0 + 1) * fx) * fy


# -- settings from a manifest or the command line ---------------------------------------------------

_FLOAT_KEYS = ("patch", "ao", "ao_radius", "light_scale", "sun_scale", "sky_scale", "sky_default", "texlight_scale",
	"reflect", "default_albedo", "max_albedo", "gamma", "exposure", "cutoff")
_INT_KEYS = ("bounces", "rays", "gather", "ao_rays", "texlight_step", "seed")
_BOOL_KEYS = ("calibrate", "entity_shadows", "use_vis")


def options_from(g: dict, root: Path | None = None, log=None) -> GiOptions:
	"""GiOptions from manifest keys ([[gi]] rule) or a plain dict; rad/wads paths relative to root."""
	o = GiOptions(log=log)
	for k in _FLOAT_KEYS:
		if g.get(k) is not None:
			setattr(o, k, float(g[k]))
	for k in _INT_KEYS:
		if g.get(k) is not None:
			setattr(o, k, int(g[k]))
	for k in _BOOL_KEYS:
		if g.get(k) is not None:
			setattr(o, k, bool(g[k]))
	if g.get("ambient") is not None:
		a = g["ambient"]
		a = [float(x) for x in (a.split() if isinstance(a, str) else (a if isinstance(a, (list, tuple)) else [a]))]
		o.ambient = tuple((a * 3)[:3]) if len(a) == 1 else tuple(a[:3])
	if g.get("texlights"):
		o.texlights = dict(g["texlights"])
	for key in ("rad", "wads"):
		v = g.get(key)
		if v:
			vals = [v] if isinstance(v, str) else list(v)
			setattr(o, key, [Path(x) if root is None or Path(x).is_absolute() else root / x for x in vals])
	return o


def atlas_settings(g: dict) -> dict:
	"""Atlas keys of a [[gi]] rule, with defaults."""
	from . import autotex

	fm = g.get("formats")
	if isinstance(fm, str):
		fm = [x.strip().upper() for x in fm.split(",") if x.strip()]
	fixed = str(g.get("format", "") or "").upper()
	if fixed in ("", "AUTO"):
		fixed = ""
	elif fixed not in autotex.ALL_FORMATS:
		raise ValueError(f"unknown GX format {fixed!r}")
	ob = g.get("overbright", "auto")
	return {"page": int(g.get("page", 256)), "formats": tuple(fm) if fm else autotex.ROLES["lightmap"], "fixed": fixed,
		"min_db": float(g.get("min_db", 40.0)), "overbright": ob if ob == "auto" else int(ob),
		"atlas_gamma": float(g.get("atlas_gamma", 1.0)), "tone": str(g.get("tone", "port")), "shrink": bool(g.get("shrink", True)),
		"pad": int(g.get("pad", 1)), "align": int(g.get("align", 0)), "write_bsp": bool(g.get("bsp", True)),
		"write_atlas": bool(g.get("atlas", True))}


def prepare(bsp: Path, opts: GiOptions, page: int = 256, formats=("CMPR", "RGB565", "I4", "I8"), fixed: str = "",
		min_db: float = 40.0, overbright="auto", atlas_gamma: float = 1.0, shrink: bool = True, pad: int = 1,
		align: int = 0, write_atlas: bool = True, name: str = "", tone: str = "port", **_unused) -> tuple[Bake, "Atlas | None", list]:
	"""Bake and lay out the atlas; returns the autotex jobs for its pages (placed by the caller)."""
	b = bake(Path(bsp).read_bytes(), opts)
	if not write_atlas or not b.grid.n:
		return b, None, []
	al = align or (4 if (fixed == "CMPR" if fixed else "CMPR" in formats) else 1)
	at = atlas(b, page, pad, al, overbright, atlas_gamma, shrink, min_db=min_db, tone=tone, bsp_gamma=opts.gamma)
	return b, at, atlas_jobs(at, name or Path(bsp).stem, tuple(formats), min_db, fixed)


# -- one call: bake and write ---------------------------------------------------------------------


@dataclass
class GiResult:
	bake: Bake
	atlas: "Atlas | None"
	placed: list
	files: list[tuple[Path, int, str]]
	report: dict


def atlas_jobs(at: Atlas, name: str, formats: tuple[str, ...], min_db: float, fixed: str = "") -> list:
	from . import autotex

	jobs = []
	for i, pg in enumerate(at.pages):
		h, w = pg.shape[:2]
		jobs.append(autotex.TexJob(f"{name}_lm{i}", pg, formats, min_db, max(w, h), max(w, h), False, True, "clamp",
			1.0, fixed, meta={"page": i, "map": "lightmap"}))
	return jobs


def write_outputs(res_bake: Bake, at: "Atlas | None", placed: list, out_dir: Path, name: str, *, write_bsp: bool,
		gamma_bsp: float, atlas_gamma: float, preview: bool) -> tuple[list[tuple[Path, int, str]], dict]:
	from . import tpl
	from .pbr import gxm_bytes

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	files: list[tuple[Path, int, str]] = []

	def save(p: Path, data: bytes, kind: str):
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_bytes(data)
		files.append((p, len(data), kind))

	if write_bsp:
		save(out_dir / f"{name}.bsp", relit_bsp(res_bake, gamma_bsp), "bsp")
	report = {"map": name, **res_bake.stats}
	if at is not None:
		save(out_dir / f"{name}_lm.tpl", tpl.write_tpl([p.image for p in placed]), "tpl")
		save(out_dir / f"{name}.gclm", gclm_bytes(at, len(res_bake.scene.faces), at.stats["gamma"]), "gclm")
		save(out_dir / f"{name}_lm.gxm", gxm_bytes("lightmap", ["lightmap"], tev_scale=at.scale), "gxm")
		from . import autotex

		report["atlas"] = {**at.stats, "textures": [autotex.summary(p) for p in placed],
			"bytes": sum(p.image.size for p in placed)}
		if preview:
			from PIL import Image

			for p in placed:
				f = out_dir / "preview" / f"{p.job.name}.png"
				f.parent.mkdir(parents=True, exist_ok=True)
				Image.fromarray(p.image.decode()).save(f)
				files.append((f, f.stat().st_size, "preview"))
	import json

	save(out_dir / f"{name}_gi.json", json.dumps(report, indent=1, default=float).encode(), "json")
	return files, report


def run(bsp: Path, out_dir: Path, opts: GiOptions | None = None, *, budget: int = 0, preview: bool = False,
		name: str = "", **settings) -> GiResult:
	"""Bake one map and write everything (settings: see atlas_settings / prepare)."""
	from . import autotex

	opts = opts or GiOptions()
	bsp = Path(bsp)
	name = name or bsp.stem
	write_bsp = settings.pop("write_bsp", True)
	b, at, jobs = prepare(bsp, opts, name=name, **settings)
	placed = []
	if jobs:
		rep = autotex.fit(jobs, budget)
		placed = rep.placed
		if budget and not rep.fits:
			b.stats["notes"].append(f"lightmap atlas is {rep.total} bytes, over the {budget}-byte budget")
	files, report = write_outputs(b, at, placed, out_dir, name, write_bsp=write_bsp, gamma_bsp=opts.gamma,
		atlas_gamma=settings.get("atlas_gamma", 1.0), preview=preview)
	return GiResult(b, at, placed, files, report)
