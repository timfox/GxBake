"""Panoramas and cubemaps to the GameCube port's six-sided sky.

The port's R_GXSkyLoad (ref/gx/r_gx_world.c) reads media/sky_<name><side>.cmp for
the sides rt, bk, lf, ft, up, dn, each exactly 128x128 GX_TF_CMPR with no mips
(8 KiB, all six in a 48 KiB BSS block), clamped, and R_GXDrawSkyBox draws them
with the st_to_vec table shared with ref/gl/gl_warp.c. In world space (Quake axes,
+Z up) each side faces, and its image's right and up edges point, along:

	side  faces  image right  image up
	rt    +X     -Y           +Z
	bk    +Y     +X           +Z
	lf    -X     +Y           +Z
	ft    -Y     -X           +Z
	up    +Z     -Y           -X
	dn    -Z     -Y           +X

Every source becomes a function from a view direction to a colour, and the six
sides are rendered from it by direction, so any layout lands in that orientation:

	equirect  a 2:1 latitude/longitude panorama (also Radiance .hdr, tone-mapped)
	cross     a 4:3 horizontal or 3:4 vertical cross (front in the middle)
	strip     six faces in a row or column, right left up down front back
	faces     six files: <name>rt/bk/lf/ft/up/dn (GoldSrc, used as they are),
	          <name>_right/left/front/back/up|top/down|bottom, or px/nx/py/ny/pz/nz
	photo     any other single image, wrapped round the horizon (mirrored to
	          close the seam) with its top and bottom rows faded to the zenith
	          and nadir

The source's "front" (the panorama centre, the cross's middle face) points at the
engine yaw given as `yaw` (degrees, 0 = +X/rt, 90 = +Y/bk, 270 = -Y/ft).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cmpr import cmpr_decode, gx_cmpr_encode

SKY_SIZE = 128	# R_GX_SKY_SIZE: the engine refuses any other length
SKY_BYTES = SKY_SIZE * SKY_SIZE // 2
SIDES = ("rt", "bk", "lf", "ft", "up", "dn")	# Xash r_skyBoxSuffix order
LAYOUTS = ("auto", "equirect", "cross", "strip", "faces", "photo")
DEFAULT_YAW = 270.0	# panorama centre → the ft side
NAME_MAX = 24	# r_gx_sky_name[32] and the 64-byte "media/sky_%s%s.cmp" path

_X, _Y, _Z = np.eye(3)

# side -> (normal, image right, image up), world space
ENGINE_FACES: dict[str, tuple] = {
	"rt": (_X, -_Y, _Z),
	"bk": (_Y, _X, _Z),
	"lf": (-_X, _Y, _Z),
	"ft": (-_Y, -_X, _Z),
	"up": (_Z, -_Y, -_X),
	"dn": (-_Z, -_Y, _X),
}

# Viewer-relative faces as cross and strip layouts draw them: (f, r, u) components.
_F, _R, _U = np.eye(3)
VIEW_FACES: dict[str, tuple] = {
	"front": (_F, _R, _U),
	"right": (_R, -_F, _U),
	"back": (-_F, -_R, _U),
	"left": (-_R, _F, _U),
	"up": (_U, _R, -_F),	# bottom edge meets the front's top
	"down": (-_U, _R, _F),	# top edge meets the front's bottom
}
VIEW_ORDER = ("right", "left", "up", "down", "front", "back")	# strip default (+X -X +Y -Y +Z -Z)
# (column, row) of each face in a horizontal cross; a vertical cross is checked separately.
_HCROSS = {"left": (0, 1), "front": (1, 1), "right": (2, 1), "back": (3, 1), "up": (1, 0), "down": (1, 2)}
_VCROSS = {"up": (1, 0), "left": (0, 1), "front": (1, 1), "right": (2, 1), "down": (1, 2), "back": (1, 3)}

_FACE_NAMES = {
	"right": "right", "left": "left", "front": "front", "back": "back", "up": "up", "top": "up", "down": "down", "bottom": "down",
	"px": "right", "nx": "left", "py": "up", "ny": "down", "pz": "front", "nz": "back",
	"posx": "right", "negx": "left", "posy": "up", "negy": "down", "posz": "front", "negz": "back",
}


@dataclass
class SkyOptions:
	layout: str = "auto"
	yaw: float = DEFAULT_YAW
	fov: float = 180.0	# photo layout: horizontal degrees the image covers
	exposure: float = 0.0	# .hdr: stops over the automatic exposure
	size: int = SKY_SIZE
	supersample: int = 4


@dataclass
class SkySource:
	"""Something that can be looked up by direction, plus how it was read."""
	layout: str
	name: str
	sample: object	# callable (N,3) world directions -> (N,3) float RGB 0..255
	notes: list[str] = field(default_factory=list)
	native: dict | None = None	# GoldSrc sides: side -> RGB, used without resampling


# -- loading -----------------------------------------------------------------


def load_rgb(path: Path, exposure: float = 0.0) -> "np.ndarray":
	"""Any image GxBake reads, or a Radiance .hdr tone-mapped, as uint8 RGB."""
	path = Path(path)
	if path.suffix.lower() == ".hdr":
		return tonemap(read_hdr(path), exposure)
	from .images import load_rgba

	return load_rgba(path)[..., :3]


def read_hdr(path: Path) -> "np.ndarray":
	"""Radiance RGBE (.hdr/.pic) → float32 linear RGB."""
	data = Path(path).read_bytes()
	if not (data.startswith(b"#?RADIANCE") or data.startswith(b"#?RGBE")):
		raise ValueError("not a Radiance .hdr file")
	pos = 0
	while True:	# header ends at an empty line
		end = data.index(b"\n", pos)
		line = data[pos:end].strip()
		pos = end + 1
		if line.startswith(b"FORMAT=") and line != b"FORMAT=32-bit_rle_rgbe":
			raise ValueError(f"unsupported .hdr {line.decode(errors='replace')}")
		if not line:
			break
	end = data.index(b"\n", pos)
	m = re.match(rb"([-+])Y (\d+) ([-+])X (\d+)", data[pos:end].strip())
	if not m:
		raise ValueError("unsupported .hdr orientation (only -Y H +X W)")
	h, w = int(m.group(2)), int(m.group(4))
	pos = end + 1
	out = np.empty((h, w, 4), np.uint8)
	buf = memoryview(data)
	for y in range(h):
		if 8 <= w < 0x8000 and data[pos] == 2 and data[pos + 1] == 2 and data[pos + 2] < 128:
			if (data[pos + 2] << 8 | data[pos + 3]) != w:
				raise ValueError("corrupt .hdr scanline")
			pos += 4
			for c in range(4):
				row = bytearray(w)
				x = 0
				while x < w:
					n = data[pos]
					pos += 1
					if n > 128:
						n -= 128
						row[x:x + n] = bytes((data[pos],)) * n
						pos += 1
					else:
						row[x:x + n] = buf[pos:pos + n]
						pos += n
					x += n
				out[y, :, c] = np.frombuffer(bytes(row[:w]), np.uint8)
		else:	# flat (old RLE is not supported, flat is)
			out[y] = np.frombuffer(data[pos:pos + w * 4], np.uint8).reshape(w, 4)
			pos += w * 4
	e = out[..., 3].astype(np.int32)
	scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
	return out[..., :3].astype(np.float32) * scale[..., None]


def tonemap(hdr: "np.ndarray", exposure: float = 0.0) -> "np.ndarray":
	"""Linear HDR → sRGB uint8: key-0.18 auto exposure, ACES fit, sRGB curve."""
	lum = hdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
	gmean = float(np.exp(np.log(np.maximum(lum, 1e-6)).mean()))
	x = hdr * (0.18 / max(gmean, 1e-6) * 2.0 ** exposure)
	x = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
	srgb = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
	return np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)


def _shrink(rgb: "np.ndarray", max_w: int) -> "np.ndarray":
	"""Pre-filter big sources so bilinear lookups do not alias."""
	h, w = rgb.shape[:2]
	if w <= max_w:
		return rgb
	from PIL import Image

	nh = max(1, round(h * max_w / w))
	return np.asarray(Image.fromarray(rgb).resize((max_w, nh), Image.Resampling.LANCZOS))


# -- layout detection ----------------------------------------------------------


def _is_cross(rgb: "np.ndarray", cells: dict) -> bool:
	"""The cells a cross leaves empty are flat (one colour, usually black/transparent)."""
	h, w = rgb.shape[:2]
	cols = max(c for c, _ in cells.values()) + 1
	rows = max(r for _, r in cells.values()) + 1
	cw, ch = w // cols, h // rows
	used = set(cells.values())
	for c in range(cols):
		for r in range(rows):
			if (c, r) in used:
				continue
			cell = rgb[r * ch + ch // 8:(r + 1) * ch - ch // 8, c * cw + cw // 8:(c + 1) * cw - cw // 8].astype(np.int16)
			if cell.size and int(np.ptp(cell, axis=(0, 1)).max()) > 8:
				return False
	return True


def detect_layout(rgb: "np.ndarray") -> str:
	h, w = rgb.shape[:2]
	if w == 2 * h:
		return "equirect"
	if w == 6 * h or h == 6 * w:
		return "strip"
	if (3 * w == 4 * h and _is_cross(rgb, _HCROSS)) or (4 * w == 3 * h and _is_cross(rgb, _VCROSS)):
		return "cross"
	return "photo"


def looks_like_panorama(path: Path) -> bool:
	"""Whether `convert` should treat a lone image as a sky rather than a texture.

	Radiance .hdr files always are; otherwise a 2:1 image at least 1024 wide, a
	6:1 strip of 128+ faces, or a cross with empty corners at least 512 wide.
	Smaller ones stay textures; `gxbake sky` takes anything."""
	path = Path(path)
	if path.suffix.lower() == ".hdr":
		return True
	try:
		from PIL import Image

		with Image.open(path) as img:
			w, h = img.size
	except Exception:	# noqa: BLE001 - VTF/WAL and broken files are not panoramas
		return False
	if w == 2 * h and w >= 1024:
		return True
	if (w == 6 * h and h >= 128) or (h == 6 * w and w >= 128):
		return True
	if (3 * w == 4 * h or 4 * w == 3 * h) and max(w, h) >= 512:
		return detect_layout(load_rgb(path)) == "cross"
	return False


def find_faces(path: Path) -> tuple[str, str, dict[str, Path]] | None:
	"""Six face files next to `path` (or in the folder `path`).

	Returns (name, scheme, {face: file}) where scheme is "goldsrc" (faces are
	rt..dn) or "view" (faces are right/left/front/back/up/down)."""
	from .images import IMAGE_EXTENSIONS

	path = Path(path)
	folder = path if path.is_dir() else path.parent
	files = [p for p in sorted(folder.iterdir()) if p.is_file() and (p.suffix.lower() in IMAGE_EXTENSIONS or p.suffix.lower() == ".hdr")]
	stems = [path.stem.lower()] if path.is_file() else [p.stem.lower() for p in files]
	for stem in stems:
		# Descriptive names first: "skyleft" also ends in "ft".
		for token, face in sorted(_FACE_NAMES.items(), key=lambda kv: -len(kv[0])):
			m = re.fullmatch(rf"(.*?)[_\-. ]?{token}", stem)
			if not m or not m.group(1):
				continue
			base = m.group(1)
			found: dict[str, Path] = {}
			for p in files:
				mm = re.fullmatch(rf"{re.escape(base)}[_\-. ]?([a-z]+)", p.stem.lower())
				if mm and mm.group(1) in _FACE_NAMES:
					found.setdefault(_FACE_NAMES[mm.group(1)], p)
			if len(found) == 6:
				return base.rstrip("_-. "), "view", found
		for sfx in SIDES:
			if stem.endswith(sfx) and len(stem) > len(sfx):
				base = stem[: -len(sfx)]
				found = {}
				for p in files:
					s = p.stem.lower()
					if s.startswith(base) and s[len(base):] in SIDES:
						found.setdefault(s[len(base):], p)
				if len(found) == 6:
					return base.rstrip("_-. "), "goldsrc", found
	return None


# -- direction lookups ---------------------------------------------------------


def view_basis(yaw: float) -> "np.ndarray":
	"""Rows F, R, U in world space for a viewer facing engine yaw `yaw`."""
	a = math.radians(yaw)
	f = np.array([math.cos(a), math.sin(a), 0.0])
	r = np.array([math.sin(a), -math.cos(a), 0.0])
	return np.stack([f, r, _Z])


def _bilinear(img: "np.ndarray", x: "np.ndarray", y: "np.ndarray", wrap_x: bool = False) -> "np.ndarray":
	"""Sample img at continuous texel coords (centre of texel 0 is 0.0)."""
	h, w = img.shape[:2]
	x0 = np.floor(x)
	y0 = np.floor(y)
	fx = (x - x0)[:, None]
	fy = (y - y0)[:, None]
	x0 = x0.astype(np.int64)
	y0 = y0.astype(np.int64)
	if wrap_x:
		xa, xb = x0 % w, (x0 + 1) % w
	else:
		xa, xb = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
	ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
	f = img.astype(np.float32)
	top = f[ya, xa] * (1 - fx) + f[ya, xb] * fx
	bot = f[yb, xa] * (1 - fx) + f[yb, xb] * fx
	return top * (1 - fy) + bot * fy


def equirect_sampler(rgb: "np.ndarray", basis: "np.ndarray"):
	h, w = rgb.shape[:2]

	def sample(d: "np.ndarray") -> "np.ndarray":
		v = d @ basis.T	# (f, r, u)
		lon = np.arctan2(v[:, 1], v[:, 0])	# right of centre is positive
		lat = np.arcsin(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
		x = (0.5 + lon / (2 * math.pi)) * w - 0.5
		y = (0.5 - lat / math.pi) * h - 0.5
		return _bilinear(rgb, x, y, wrap_x=True)

	return sample


def photo_sampler(rgb: "np.ndarray", basis: "np.ndarray", fov: float) -> tuple[object, float]:
	"""A flat image round the horizon, mirrored to fill 360°, faded to the poles."""
	h, w = rgb.shape[:2]
	pairs = max(1, round(360.0 / (2 * max(1.0, min(fov, 360.0)))))
	hfov = math.radians(360.0 / (2 * pairs))
	vhalf = min(hfov * h / w / 2, math.pi / 2)
	band = max(1, h // 16)
	top = rgb[:band].reshape(-1, 3).mean(0)
	bottom = rgb[-band:].reshape(-1, 3).mean(0)

	def sample(d: "np.ndarray") -> "np.ndarray":
		v = d @ basis.T
		lon = np.arctan2(v[:, 1], v[:, 0])
		lat = np.arcsin(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
		t = np.mod(lon / hfov + 0.5, 2.0)
		t = np.where(t > 1.0, 2.0 - t, t)	# mirror every other copy: no seam
		x = t * w - 0.5
		y = (0.5 - lat / (2 * vhalf)) * h - 0.5
		out = _bilinear(rgb, np.clip(x, 0, w - 1), np.clip(y, 0, h - 1))
		if vhalf < math.pi / 2:
			span = math.pi / 2 - vhalf
			up = np.clip((lat - vhalf) / span, 0, 1)[:, None]
			dn = np.clip((-lat - vhalf) / span, 0, 1)[:, None]
			out = out * (1 - up) + top * up
			out = out * (1 - dn) + bottom * dn
		return out

	return sample, math.degrees(hfov)


def cube_sampler(faces: dict[str, "np.ndarray"], frames: dict[str, tuple]):
	"""Look up six square images, each with (normal, right, up) in world space."""
	names = list(faces)
	normals = np.stack([frames[n][0] for n in names])

	def sample(d: "np.ndarray") -> "np.ndarray":
		pick = np.argmax(d @ normals.T, axis=1)
		out = np.zeros((len(d), 3), np.float32)
		for i, n in enumerate(names):
			sel = pick == i
			if not sel.any():
				continue
			nn, rr, uu = frames[n]
			dd = d[sel]
			dn = dd @ nn
			a = (dd @ rr) / dn
			b = (dd @ uu) / dn
			img = faces[n]
			h, w = img.shape[:2]
			out[sel] = _bilinear(img, (a + 1) / 2 * w - 0.5, (1 - b) / 2 * h - 0.5)
		return out

	return sample


def _view_frames(basis: "np.ndarray") -> dict[str, tuple]:
	return {k: tuple(v @ basis for v in vecs) for k, vecs in VIEW_FACES.items()}


def split_cross(rgb: "np.ndarray") -> dict[str, "np.ndarray"]:
	h, w = rgb.shape[:2]
	cells, cols = (_HCROSS, 4) if 3 * w == 4 * h else (_VCROSS, 3)
	n = w // cols
	out = {}
	for face, (c, r) in cells.items():
		cell = rgb[r * n:(r + 1) * n, c * n:(c + 1) * n]
		if cells is _VCROSS and face == "back":
			cell = cell[::-1, ::-1]	# a vertical cross hangs the back face upside down
		out[face] = cell
	return out


def split_strip(rgb: "np.ndarray", order=VIEW_ORDER) -> dict[str, "np.ndarray"]:
	h, w = rgb.shape[:2]
	n = min(w, h)
	if w > h:
		return {face: rgb[:, i * n:(i + 1) * n] for i, face in enumerate(order)}
	return {face: rgb[i * n:(i + 1) * n] for i, face in enumerate(order)}


def open_source(paths: list[Path], opts: SkyOptions | None = None, name: str | None = None) -> SkySource:
	"""Read one panorama/cross/strip/photo, or six face files, as a SkySource."""
	opts = opts or SkyOptions()
	paths = [Path(p) for p in paths]
	basis = view_basis(opts.yaw)
	layout = opts.layout
	if layout not in LAYOUTS:
		raise ValueError(f"unknown layout {layout!r} (one of {', '.join(LAYOUTS)})")
	notes: list[str] = []
	work = SKY_SIZE * max(1, opts.supersample)	# texels per 90° the renderer will ask for

	faces = None
	if len(paths) > 1 or paths[0].is_dir() or layout == "faces":
		if len(paths) == 6 and not any(p.is_dir() for p in paths):
			faces = _faces_from_list(paths)
		else:
			faces = find_faces(paths[0])
		if faces is None:
			raise ValueError("need six face images named <name>rt/bk/lf/ft/up/dn, <name>_right/left/front/back/up/down "
				"or <name>_px/nx/py/ny/pz/nz")
	elif layout == "auto":
		faces = find_faces(paths[0])

	if faces is not None:
		base, scheme, files = faces
		imgs = {k: _shrink(load_rgb(p, opts.exposure), work) for k, p in files.items()}
		sizes = {im.shape[:2] for im in imgs.values()}
		if len(sizes) > 1:
			notes.append(f"face sizes differ: {sorted(sizes)}")
		for k, im in imgs.items():
			if im.shape[0] != im.shape[1]:
				notes.append(f"{files[k].name} is not square ({im.shape[1]}x{im.shape[0]})")
		if scheme == "goldsrc":
			native = imgs if abs((opts.yaw - DEFAULT_YAW) % 360) < 1e-6 else None
			return SkySource("faces", name or base, cube_sampler(imgs, ENGINE_FACES), notes, native)
		return SkySource("faces", name or base, cube_sampler(imgs, _view_frames(basis)), notes)

	path = paths[0]
	rgb = load_rgb(path, opts.exposure)
	if layout == "auto":
		layout = detect_layout(rgb)
	stem = name or re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_") or "sky"
	h, w = rgb.shape[:2]
	if layout == "equirect":
		if w != 2 * h:
			notes.append(f"{w}x{h} is not 2:1; stretched as a full panorama")
		return SkySource(layout, stem, equirect_sampler(_shrink(rgb, 4 * work), basis), notes)
	if layout == "cross":
		if not (3 * w == 4 * h or 4 * w == 3 * h):
			raise ValueError(f"a cross is 4:3 or 3:4, not {w}x{h}")
		cells = {k: _shrink(v, work) for k, v in split_cross(rgb).items()}
		return SkySource(layout, stem, cube_sampler(cells, _view_frames(basis)), notes)
	if layout == "strip":
		if not (w == 6 * h or h == 6 * w):
			raise ValueError(f"a strip is 6:1 or 1:6, not {w}x{h}")
		cells = {k: _shrink(v, work) for k, v in split_strip(rgb).items()}
		return SkySource(layout, stem, cube_sampler(cells, _view_frames(basis)), notes)
	sample, used = photo_sampler(_shrink(rgb, 4 * work), basis, opts.fov)
	if abs(used - opts.fov) > 0.5:
		notes.append(f"photo widened to {used:g}° so its mirrored copies close the circle")
	return SkySource("photo", stem, sample, notes)


def _faces_from_list(paths: list[Path]) -> tuple[str, str, dict[str, Path]] | None:
	"""Six explicit files: by name if they carry face names, else in rt bk lf ft up dn order."""
	got = find_faces(paths[0])
	if got and set(map(Path.resolve, got[2].values())) == set(map(Path.resolve, paths)):
		return got
	return paths[0].stem.lower(), "goldsrc", dict(zip(SIDES, paths))


# -- rendering -----------------------------------------------------------------


def side_directions(side: str, n: int) -> "np.ndarray":
	"""World direction through the centre of each texel of an n×n engine side, row-major."""
	normal, right, up = ENGINE_FACES[side]
	c = (np.arange(n) + 0.5) / n * 2 - 1
	a, b = np.meshgrid(c, -c)	# b: +1 at the top row
	return (normal[None] + a.reshape(-1, 1) * right[None] + b.reshape(-1, 1) * up[None])


def render_side(source: SkySource, side: str, size: int = SKY_SIZE, supersample: int = 4) -> "np.ndarray":
	"""One engine side as uint8 RGB, `supersample`² lookups per texel box-filtered."""
	if source.native is not None:
		from PIL import Image

		return np.asarray(Image.fromarray(source.native[side]).resize((size, size), Image.Resampling.LANCZOS))
	ss = max(1, supersample)
	n = size * ss
	d = side_directions(side, n)
	rgb = source.sample(d).reshape(n, n, 3)
	rgb = rgb.reshape(size, ss, size, ss, 3).mean(axis=(1, 3))
	return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


@dataclass
class Skybox:
	name: str
	layout: str
	sides: dict[str, "np.ndarray"]	# rendered RGB, engine orientation
	cmpr: dict[str, bytes]
	notes: list[str]

	def decoded(self) -> dict[str, "np.ndarray"]:
		"""What the GameCube samples: each side's CMPR decoded back to RGBA."""
		return {k: cmpr_decode(v, *self.sides[k].shape[1::-1]) for k, v in self.cmpr.items()}

	def files(self) -> dict[str, bytes]:
		return {f"media/sky_{self.name}{s}.cmp": self.cmpr[s] for s in SIDES}


def check_name(name: str) -> list[str]:
	notes = []
	if name != name.lower():
		notes.append("the engine lowercases sky names; files are written lowercase")
	if len(name) > NAME_MAX:
		notes.append(f"sky name is {len(name)} characters; the engine keeps {NAME_MAX} at most")
	if not re.fullmatch(r"[a-z0-9_\-]+", name.lower()):
		notes.append("sky name should be letters, digits, '_' or '-' (it goes in worldspawn \"skyname\")")
	return notes


def build(paths: list[Path], opts: SkyOptions | None = None, name: str | None = None) -> Skybox:
	"""Read a source and render + CMPR-encode the six sides."""
	opts = opts or SkyOptions()
	src = open_source(paths, opts, name)
	notes = list(src.notes) + check_name(src.name)
	if opts.size != SKY_SIZE:
		notes.append(f"the port only loads {SKY_SIZE}x{SKY_SIZE} sides ({SKY_BYTES} bytes); {opts.size} will be refused")
	sides = {s: render_side(src, s, opts.size, opts.supersample) for s in SIDES}
	cmpr = {s: gx_cmpr_encode(sides[s]) for s in SIDES}
	return Skybox(src.name.lower()[:NAME_MAX], src.layout, sides, cmpr, notes)


def cross_preview(sides: dict[str, "np.ndarray"], yaw: float = DEFAULT_YAW) -> "np.ndarray":
	"""A horizontal-cross picture of engine sides, front (the yaw) in the middle."""
	imgs = {k: v[..., :3] for k, v in sides.items()}
	n = next(iter(imgs.values())).shape[0]
	sample = cube_sampler(imgs, ENGINE_FACES)
	frames = _view_frames(view_basis(yaw))
	out = np.zeros((3 * n, 4 * n, 3), np.uint8)
	c = (np.arange(n) + 0.5) / n * 2 - 1
	a, b = np.meshgrid(c, -c)
	for face, (col, row) in _HCROSS.items():
		nn, rr, uu = frames[face]
		d = nn[None] + a.reshape(-1, 1) * rr[None] + b.reshape(-1, 1) * uu[None]
		out[row * n:(row + 1) * n, col * n:(col + 1) * n] = np.clip(sample(d) + 0.5, 0, 255).astype(np.uint8).reshape(n, n, 3)
	return out
