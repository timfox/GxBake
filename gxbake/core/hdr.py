"""HDR (OpenEXR, Radiance .hdr, PFM, float TIFF) down to what Flipper can show.

GX has no float textures, no sRGB decode and an 8-bit-per-channel TEV, so
every HDR source has to be squeezed into 8 bits before it reaches the disc.
What is right depends on the use:

* **texture / sky**: tone-map to display sRGB with one of `TONEMAPS`
  (`aces` matches the sky converter's long-standing default).
* **lightmap (overbright)**: a lightmap multiplies the albedo in TEV, and a
  TEV stage can scale its output by 2 or 4 (GX_CS_SCALE_2 / _4). Storing
  light / scale lets a lightmap brighten a surface up to 2× or 4× instead of
  clamping at 1, which is how GameCube games kept highlights from going flat.
  The value is stored linear (TEV multiplies raw numbers).
* **sphere map (environment reflections)**: GX texgen can generate texture
  coordinates from the view-space normal (GX_TG_MTX2x4 with a GX_TG_NRM
  source and a 0.5 scale/bias matrix), which looks up a sphere map: the
  whole environment as seen in a mirrored ball. `sphere_map` renders one from
  any sky source (panorama, cross, strip, faces, HDR), optionally blurred to
  a roughness so rough metals get soft reflections.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TONEMAPS = ("aces", "reinhard", "hable", "agx", "clip")
_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def read(path: Path) -> "np.ndarray":
	"""Linear float32 H×W×4 from any HDR source GxBake reads (8-bit images are linearised)."""
	from .imageformats import read_float

	hdr = read_float(Path(path))
	if hdr is None:
		from .images import load_rgba

		rgba = load_rgba(Path(path)).astype(np.float32) / 255
		rgba[..., :3] = srgb_to_linear(rgba[..., :3])
		return rgba
	hdr = np.nan_to_num(np.asarray(hdr, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	if hdr.ndim == 2:
		hdr = hdr[..., None]
	if hdr.shape[-1] == 1:
		hdr = np.concatenate([hdr] * 3, -1)
	if hdr.shape[-1] == 3:
		hdr = np.concatenate((hdr, np.ones(hdr.shape[:2] + (1,), np.float32)), -1)
	return hdr


def srgb_to_linear(x: "np.ndarray") -> "np.ndarray":
	x = np.asarray(x, np.float32)
	return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: "np.ndarray") -> "np.ndarray":
	x = np.clip(np.asarray(x, np.float32), 0, 1)
	return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def auto_exposure(rgb: "np.ndarray", key: float = 0.18) -> float:
	"""Scale that brings the log-average luminance to `key` (Reinhard 2002)."""
	lum = rgb[..., :3] @ _LUM
	g = float(np.exp(np.log(np.maximum(lum, 1e-6)).mean()))
	return key / max(g, 1e-6)


def _aces(x):
	return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)


def _hable_p(x):
	a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
	return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def _agx(x):
	"""AgX-style: log2 encode over [-12.47, 4.03] EV, then a fitted sigmoid (Troy Sobotka's
	curve, polynomial from the Blender 4 / three.js port). No look, no inset/outset matrices."""
	lo, hi = -12.47393, 4.026069
	v = np.clip((np.log2(np.maximum(x, 1e-10)) - lo) / (hi - lo), 0, 1)
	v2 = v * v
	v4 = v2 * v2
	y = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232
	return np.clip(y, 0, 1) ** 2.2	# the curve outputs display values; back to linear for the shared sRGB step


def tonemap_float(rgb: "np.ndarray", op: str = "aces", exposure: float = 0.0, auto: bool = True,
		white: float = 4.0) -> "np.ndarray":
	"""Linear HDR → linear display-referred [0, 1] float RGB."""
	if op not in TONEMAPS:
		raise ValueError(f"unknown tone map {op!r} (one of {', '.join(TONEMAPS)})")
	x = np.maximum(np.asarray(rgb[..., :3], np.float32), 0)
	scale = (auto_exposure(x) if auto else 1.0) * 2.0 ** exposure
	x = x * scale
	if op == "aces":
		y = _aces(x)
	elif op == "reinhard":	# extended Reinhard on luminance, `white` maps to 1
		lum = x @ _LUM
		lout = lum * (1 + lum / (white * white)) / (1 + lum)
		y = x * (lout / np.maximum(lum, 1e-6))[..., None]
	elif op == "hable":
		y = _hable_p(x * 2) / _hable_p(np.float32(white * 2.8))
	elif op == "agx":
		y = _agx(x)
	else:
		y = x
	return np.clip(y, 0, 1).astype(np.float32)


def tonemap(rgb: "np.ndarray", op: str = "aces", exposure: float = 0.0, auto: bool = True,
		white: float = 4.0) -> "np.ndarray":
	"""Linear HDR → sRGB uint8 RGB."""
	y = linear_to_srgb(tonemap_float(rgb, op, exposure, auto, white))
	return np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)


def to_rgba8(hdr: "np.ndarray", op: str = "aces", exposure: float = 0.0, auto: bool = True,
		white: float = 4.0) -> "np.ndarray":
	rgb = tonemap(hdr, op, exposure, auto, white)
	a = np.clip(hdr[..., 3] * 255 + 0.5, 0, 255).astype(np.uint8) if hdr.shape[-1] > 3 \
		else np.full(rgb.shape[:2], 255, np.uint8)
	return np.dstack([rgb, a])


# -- lightmaps --------------------------------------------------------------------------------------


def overbright(hdr: "np.ndarray", scale: int = 2, exposure: float = 0.0, gamma: float = 1.0) -> tuple["np.ndarray", dict]:
	"""Light values → uint8 RGBA holding light / scale, for a TEV stage with GX_CS_SCALE_<scale>.

	Returns the image and stats: how much of the light clipped even with the
	headroom, so a build can warn or pick a larger scale (`pick_overbright`)."""
	if scale not in (1, 2, 4):
		raise ValueError("TEV can scale by 1, 2 or 4")
	x = np.maximum(np.asarray(hdr[..., :3], np.float32), 0) * 2.0 ** exposure
	stored = x / scale
	clipped = float((stored.max(-1) > 1.0).mean())
	stored = np.clip(stored, 0, 1) ** (1 / gamma)
	rgb = np.clip(stored * 255 + 0.5, 0, 255).astype(np.uint8)
	return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), {
		"scale": scale, "clipped": round(clipped, 5), "peak": round(float(x.max(initial=0)), 4),
		"tev_scale": {1: "GX_CS_SCALE_1", 2: "GX_CS_SCALE_2", 4: "GX_CS_SCALE_4"}[scale], "gamma": gamma}


def pick_overbright(hdr: "np.ndarray", exposure: float = 0.0, clip_frac: float = 0.001) -> int:
	"""Smallest TEV scale (1, 2, 4) under which at most `clip_frac` of texels clip;
	smaller scales keep more precision in the dark, which is most of a lightmap."""
	x = np.maximum(np.asarray(hdr[..., :3], np.float32), 0).max(-1) * 2.0 ** exposure
	for s in (1, 2, 4):
		if float((x > s).mean()) <= clip_frac:
			return s
	return 4


# -- sphere maps ------------------------------------------------------------------------------------


def sphere_directions(size: int) -> tuple["np.ndarray", "np.ndarray"]:
	"""View-space reflection vector for each texel of a size×size sphere map, and the disc mask.

	GX sphere-map texgen: s = 0.5·nx + 0.5, t = −0.5·ny + 0.5 with n the view-space
	normal (camera looks down −Z, +Y up). The eye ray (0, 0, −1) reflected about n
	is r = (2·nz·nx, 2·nz·ny, 2·nz² − 1). Texels outside the disc take the rim's
	direction so bilinear filtering and mips never pull in black."""
	c = (np.arange(size) + 0.5) / size * 2 - 1
	nx, ny = np.meshgrid(c, -c)
	rr = nx * nx + ny * ny
	inside = rr <= 1
	k = np.where(inside, 1.0, 1 / np.sqrt(np.maximum(rr, 1e-12)))
	nx, ny = nx * k, ny * k
	nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
	r = np.stack((2 * nz * nx, 2 * nz * ny, 2 * nz * nz - 1), -1)
	return r.reshape(-1, 3), inside


def _cone_offsets(n: int, spread: float, seed: int = 7) -> "np.ndarray":
	"""n unit-ish tangent-plane offsets inside a cone of half-angle `spread` (radians), cosine-weighted."""
	rng = np.random.default_rng(seed)
	u = rng.random(n)
	phi = rng.random(n) * 2 * math.pi
	t = np.tan(spread) * np.sqrt(u)
	return np.stack((t * np.cos(phi), t * np.sin(phi)), -1)


def sphere_map(source, size: int = 128, yaw: float = 270.0, pitch: float = 0.0, roughness: float = 0.0,
		samples: int = 0, supersample: int = 2) -> "np.ndarray":
	"""Render a GX sphere map (uint8 RGBA, alpha 255) from a skybox.SkySource.

	The map is view dependent: it is what a mirror ball shows a camera facing
	engine `yaw` (0 = +X, 90 = +Y, 270 = −Y, Z up) tilted by `pitch` degrees.
	That is the usual GameCube compromise, a fixed environment that rotates with
	the camera. `roughness` (0–1) blurs it with a cosine-weighted cone of
	reflection directions, like a pre-filtered environment for a rough metal."""
	ss = max(1, supersample)
	n = size * ss
	r, _inside = sphere_directions(n)
	a, p = math.radians(yaw), math.radians(pitch)
	f = np.array([math.cos(a) * math.cos(p), math.sin(a) * math.cos(p), math.sin(p)])
	right = np.array([math.sin(a), -math.cos(a), 0.0])
	up = np.cross(right, f)
	# view space (x right, y up, z towards the viewer) → world
	world = r[:, :1] * right + r[:, 1:2] * up - r[:, 2:3] * f
	if roughness > 0:
		spread = min(math.pi / 2 * 0.95, roughness * roughness * math.pi / 2 + roughness * 0.35)
		k = samples or max(8, int(48 * roughness))
		offs = _cone_offsets(k, spread)
		helper = np.where(np.abs(world[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
		t1 = np.cross(world, helper)
		t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
		t2 = np.cross(world, t1)
		acc = np.zeros_like(world)
		for ox, oy in offs:
			acc += np.asarray(source.sample(world + ox * t1 + oy * t2), np.float64)
		rgb = acc / len(offs)
	else:
		rgb = np.asarray(source.sample(world), np.float64)
	rgb = rgb.reshape(size, ss, size, ss, 3).mean(axis=(1, 3))
	out = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
	return np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])


def sky_source(path: Path, op: str = "aces", exposure: float = 0.0, layout: str = "auto"):
	"""A skybox.SkySource for any panorama/cross/strip/faces, HDR tone-mapped with `op`."""
	from . import skybox

	opts = skybox.SkyOptions(layout=layout, exposure=exposure, tonemap=op)
	paths = [Path(p) for p in (path if isinstance(path, (list, tuple)) else [path])]
	return skybox.open_source(paths, opts)
