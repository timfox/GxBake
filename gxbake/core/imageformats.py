"""The registry of every image format GxBake reads, and the loaders that are not
a dedicated game reader.

`FORMATS` lists each format with its extensions, group (for the app's file
filters and `gxbake formats`) and the optional package it needs, if any.
`load(path)` returns H×W×4 uint8 RGBA for any registered extension.

Optional packages (all in `pip install gxbake[images]`):

	pillow-heif        HEIC / HEIF
	pillow-jxl-plugin  JPEG XL (GPL-3.0, so it is its own extra: gxbake[jxl], and not in the executable)
	OpenEXR            OpenEXR .exr
	resvg-py           SVG (CairoSVG is used instead when it is installed)
	psd-tools          Photoshop layers composited (Pillow alone reads the saved composite)
	rawpy              camera RAW (DNG, CR2/CR3, NEF, ARW, ...)
	texture2ddecoder   ASTC, PVRTC, ATC and ETC2 punch-through, and faster ETC
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import texcodec as tc
from .texcodec import MissingDependency


@dataclass(frozen=True)
class Format:
	name: str
	group: str
	extensions: tuple[str, ...]
	loader: str	# "pillow", or "module:function" inside gxbake.core
	module: str = ""	# optional import this needs ("" = nothing beyond Pillow/numpy)
	package: str = ""	# pip name for that module
	note: str = ""
	decodes: bool = True	# False: recognised, but no decoder ships (Basis)


COMMON, HDR, GPU, CONSOLE, GAME, RETRO = ("Common images", "HDR, vector and pro formats", "GPU textures",
	"Console textures", "PC game textures", "Retro computer pictures")

FORMATS: list[Format] = [
	# -- through Pillow ---------------------------------------------------------------------------
	Format("PNG / APNG", COMMON, (".png", ".apng"), "pillow"),
	Format("JPEG", COMMON, (".jpg", ".jpeg", ".jpe", ".jfif"), "pillow"),
	Format("Targa", COMMON, (".tga", ".icb", ".vda", ".vst"), "pillow"),
	Format("Windows bitmap", COMMON, (".bmp", ".dib"), "pillow"),
	Format("GIF (first frame)", COMMON, (".gif",), "pillow"),
	Format("WebP", COMMON, (".webp",), "pillow"),
	Format("AVIF", COMMON, (".avif", ".avifs"), "pillow", note="needs Pillow built with libavif (the PyPI wheels are)"),
	Format("TIFF", COMMON, (".tif", ".tiff"), "pillow"),
	Format("QOI", COMMON, (".qoi",), "pillow"),
	Format("Netpbm", COMMON, (".ppm", ".pgm", ".pbm", ".pnm"), "pillow"),
	Format("Windows icon and cursor", COMMON, (".ico", ".cur"), "pillow"),
	Format("Apple icon", COMMON, (".icns",), "pillow"),
	Format("JPEG 2000", COMMON, (".jp2", ".j2k", ".j2c", ".jpc", ".jpf", ".jpx"), "pillow"),
	Format("Multi-picture JPEG", COMMON, (".mpo",), "pillow"),
	Format("X11 bitmap and pixmap", COMMON, (".xbm", ".xpm"), "pillow"),
	Format("Photoshop (saved composite)", COMMON, (".psd",), "imageformats:load_psd",
		note="psd-tools, when installed, composites the layers"),
	Format("DirectDraw Surface", GPU, (".dds",), "imageformats:load_dds"),
	Format("PCX / multi-page DCX", RETRO, (".pcx", ".dcx"), "pillow"),
	Format("SGI", COMMON, (".sgi", ".rgb", ".rgba", ".bw"), "pillow"),
	Format("Sun raster", RETRO, (".ras",), "pillow"),
	Format("Autodesk FLI/FLC animation (first frame)", RETRO, (".fli", ".flc"), "pillow"),
	Format("Kodak Photo CD", RETRO, (".pcd",), "pillow"),
	Format("Microsoft Paint", RETRO, (".msp",), "pillow"),
	Format("IFUNC / LabEye IM", RETRO, (".im",), "pillow"),
	Format("Pixar", HDR, (".pxr",), "pillow"),
	Format("FITS", HDR, (".fit", ".fits"), "pillow"),
	Format("GIMP brush", COMMON, (".gbr",), "pillow"),
	Format("Blizzard BLP1/BLP2", GAME, (".blp",), "pillow"),
	Format("Independence War 2 FTEX", GAME, (".ftc", ".ftu"), "pillow"),
	# -- optional plug-ins and libraries ------------------------------------------------------------
	Format("HEIC / HEIF", COMMON, (".heic", ".heif", ".hif"), "imageformats:load_heif", "pillow_heif", "pillow-heif"),
	Format("JPEG XL", COMMON, (".jxl",), "imageformats:load_jxl", "pillow_jxl", "pillow-jxl-plugin",
		note="pip install gxbake[jxl]; the plug-in is GPL-3.0"),
	Format("OpenEXR", HDR, (".exr",), "imageformats:load_exr", "OpenEXR", "OpenEXR"),
	Format("Radiance HDR", HDR, (".hdr", ".pic"), "imageformats:load_hdr"),
	Format("Portable float map", HDR, (".pfm",), "imageformats:load_pfm"),
	Format("SVG (rasterised)", HDR, (".svg", ".svgz"), "imageformats:load_svg", "resvg_py", "resvg-py"),
	Format("Photoshop large document", HDR, (".psb",), "imageformats:load_psd", "psd_tools", "psd-tools"),
	Format("Krita and OpenRaster", HDR, (".kra", ".ora"), "imageformats:load_zip_merged"),
	Format("Camera RAW", HDR, (".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".orf",
		".rw2", ".raf", ".pef", ".srw", ".3fr", ".erf", ".kdc", ".mrw", ".x3f", ".iiq", ".rwl"),
		"imageformats:load_raw", "rawpy", "rawpy"),
	# -- GPU texture containers ---------------------------------------------------------------------
	Format("Khronos KTX", GPU, (".ktx",), "gputex:load", note="ASTC/PVRTC/ATC payloads need texture2ddecoder"),
	Format("Khronos KTX 2", GPU, (".ktx2",), "gputex:load", note="BasisLZ/UASTC payloads are not supported"),
	Format("PowerVR / Dreamcast PVR", GPU, (".pvr",), "imageformats:load_pvr",
		note="PVRTC needs texture2ddecoder; a Dreamcast .pvr reads its .pvp palette"),
	Format("ASTC", GPU, (".astc",), "gputex:load", "texture2ddecoder", "texture2ddecoder"),
	Format("Basis Universal", GPU, (".basis",), "gputex:load", note="recognised; needs a Basis transcoder",
		decodes=False),
	# -- game engines already in GxBake ---------------------------------------------------------------
	Format("Valve Texture Format", GAME, (".vtf",), "images:_load_vtf"),
	Format("Source 2 texture", GAME, (".vtex_c",), "images:_load_vtex"),
	Format("Quake II wall texture", GAME, (".wal",), "quake:load_wal"),
	# -- consoles ---------------------------------------------------------------------------------------
	Format("GameCube/Wii TPL", CONSOLE, (".tpl",), "consoletex:load"),
	Format("GameCube/Wii BTI (J3D)", CONSOLE, (".bti",), "consoletex:load"),
	Format("Wii BRRES TEX0 (+ .plt0 palette)", CONSOLE, (".tex0",), "consoletex:load"),
	Format("Nintendo DS NCGR (+ .nclr palette)", CONSOLE, (".ncgr",), "consoletex:load"),
	Format("Nintendo 3DS BCLIM / BFLIM", CONSOLE, (".bclim", ".bflim"), "consoletex:load"),
	Format("PlayStation TIM", CONSOLE, (".tim",), "consoletex:load"),
	Format("PlayStation 2 TIM2", CONSOLE, (".tm2", ".tim2"), "consoletex:load"),
	Format("Xbox XPR0", CONSOLE, (".xpr",), "consoletex:load"),
	# -- PC games ---------------------------------------------------------------------------------------
	Format("Doom / Quake lump", GAME, (".lmp",), "retrotex:load", note="palette from PLAYPAL, an IWAD or gfx/palette.lmp"),
	Format("Build engine ART (Duke Nukem 3D, Blood)", GAME, (".art",), "retrotex:load", note="palette from PALETTE.DAT"),
	Format("Westwood SHP (C&C, Red Alert, Tiberian Sun)", GAME, (".shp",), "retrotex:load", note="palette from a .pal"),
	Format("Fallout FRM", GAME, (".frm", ".fr0", ".fr1", ".fr2", ".fr3", ".fr4", ".fr5"), "retrotex:load",
		note="palette from COLOR.PAL"),
	Format("Unreal Engine 1 texture package", GAME, (".utx",), "retrotex:load"),
	# -- retro computers ----------------------------------------------------------------------------------
	Format("Amiga IFF ILBM / Deluxe Paint LBM", RETRO, (".iff", ".lbm", ".ilbm"), "retrotex:load"),
	Format("Atari ST Degas", RETRO, (".pi1", ".pi2", ".pi3", ".pc1", ".pc2", ".pc3"), "retrotex:load"),
	Format("Commodore 64 Koala Painter", RETRO, (".koa", ".kla"), "retrotex:load"),
]

EXTENSIONS: dict[str, Format] = {}
for _f in FORMATS:
	for _e in _f.extensions:
		assert _e not in EXTENSIONS, f"{_e} registered twice"
		EXTENSIONS[_e] = _f


GROUP_ORDER = (COMMON, HDR, GPU, CONSOLE, GAME, RETRO)


def available(fmt: Format) -> bool:
	if not fmt.decodes:
		return False
	if fmt.module == "resvg_py":
		return tc.have("resvg_py") or tc.have("cairosvg")
	if fmt.name.startswith("AVIF"):
		from PIL import features

		return bool(features.check("avif"))
	return not fmt.module or tc.have(fmt.module)


def _call(spec: str, path: Path, warnings):
	import sys

	# plain imports (not importlib) so PyInstaller bundles every reader
	from . import consoletex, gputex, images, quake, retrotex

	mods = {"consoletex": consoletex, "gputex": gputex, "images": images, "quake": quake, "retrotex": retrotex,
		"imageformats": sys.modules[__name__]}
	mod, fn = spec.split(":")
	return getattr(mods[mod], fn)(path, warnings)


def load(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	"""Any registered image → H×W×4 uint8 RGBA."""
	path = Path(path)
	fmt = EXTENSIONS.get(path.suffix.lower())
	if fmt is None or fmt.loader == "pillow":
		return load_pillow(path, warnings)
	return _call(fmt.loader, path, warnings)


# -- Pillow ---------------------------------------------------------------------------------------


def pillow_rgba(img, warnings: list[str] | None = None) -> "np.ndarray":
	"""A PIL image → RGBA uint8, keeping 16-bit and float precision sensible."""
	n = getattr(img, "n_frames", 1)
	if n > 1 and warnings is not None and img.format not in ("MPO", "ICO", "ICNS", "DCX"):
		warnings.append(f"{n} frames; the first is converted")
	mode = img.mode
	if mode.startswith("I;16"):
		v = np.asarray(img).astype(np.uint16)
		g = (v >> 8).astype(np.uint8)
		return np.dstack([g, g, g, np.full_like(g, 255)])
	if mode == "I":
		v = np.asarray(img).astype(np.int64)
		hi = int(v.max()) if v.size else 0
		g = (np.clip(v, 0, None) >> (8 if hi > 255 and hi <= 65535 else 24 if hi > 65535 else 0)).astype(np.uint8)
		return np.dstack([g, g, g, np.full_like(g, 255)])
	if mode == "F":
		return tc.float_to_rgba8(np.asarray(img, np.float32), linear=False)
	return np.asarray(img.convert("RGBA")).copy()


def load_pillow(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	from PIL import Image

	if path.suffix.lower() in (".avif", ".avifs"):
		from PIL import features

		if not features.check("avif"):
			raise MissingDependency("AVIF images", "pillow-avif-plugin")
	with Image.open(path) as img:
		return pillow_rgba(img, warnings)


def load_heif(path: Path, warnings=None) -> "np.ndarray":
	ph = tc.require("pillow_heif", "HEIC/HEIF images", "pillow-heif")
	ph.register_heif_opener()
	return load_pillow(path, warnings)


def load_jxl(path: Path, warnings=None) -> "np.ndarray":
	from PIL import Image

	if ".jxl" not in Image.registered_extensions():
		tc.require("pillow_jxl", "JPEG XL images", "pillow-jxl-plugin")
	return load_pillow(path, warnings)


def load_psd(path: Path, warnings=None) -> "np.ndarray":
	if tc.have("psd_tools"):
		from psd_tools import PSDImage

		psd = PSDImage.open(path)
		img = psd.composite()
		if img is None:
			raise ValueError("Photoshop document has nothing visible")
		return pillow_rgba(img)
	if Path(path).suffix.lower() == ".psb":
		raise MissingDependency("Photoshop .psb documents", "psd-tools")
	return load_pillow(path, warnings)


def load_dds(path: Path, warnings=None) -> "np.ndarray":
	try:
		return load_pillow(path, warnings)
	except (OSError, ValueError, NotImplementedError, SyntaxError) as e:
		from .gputex import read_dds

		try:
			return read_dds(Path(path).read_bytes())
		except ValueError:
			raise e from None


def load_pvr(path: Path, warnings=None) -> "np.ndarray":
	with open(path, "rb") as f:
		head = f.read(4)
	if head in (b"GBIX", b"PVRT"):
		from .consoletex import load_dreamcast_pvr

		return load_dreamcast_pvr(path, warnings)
	from .gputex import load as gpu_load

	return gpu_load(path, warnings)


# -- HDR -----------------------------------------------------------------------------------------


def read_pfm(path: Path) -> "np.ndarray":
	"""Portable float map → float32 H×W×3 (or ×1), top row first."""
	data = Path(path).read_bytes()
	fields, pos = [], 0
	while len(fields) < 4:
		while data[pos:pos + 1].isspace():
			pos += 1
		end = pos
		while not data[end:end + 1].isspace():
			end += 1
		fields.append(data[pos:end])
		pos = end
	pos += 1
	kind, w, h, scale = fields[0], int(fields[1]), int(fields[2]), float(fields[3])
	if kind not in (b"PF", b"Pf"):
		raise ValueError("not a PFM file")
	c = 3 if kind == b"PF" else 1
	v = np.frombuffer(data, "<f4" if scale < 0 else ">f4", w * h * c, pos).reshape(h, w, c)
	return v[::-1].astype(np.float32)


def read_exr(path: Path) -> "np.ndarray":
	"""OpenEXR → float32 H×W×4 (first part, RGBA or Y channels)."""
	OpenEXR = tc.require("OpenEXR", "OpenEXR images")
	f = OpenEXR.File(str(path), separate_channels=True)
	ch = {k: np.array(v.pixels, np.float32) for k, v in f.channels().items()}

	def get(name):
		for k, v in ch.items():
			if k == name or k.upper() == name or k.endswith("." + name):
				return v
		return None

	r, g, b, a, y = (get(n) for n in "RGBAY")
	if r is None and y is not None:
		r = g = b = y
	if r is None:
		raise ValueError(f"EXR has no colour channels ({', '.join(ch) or 'none'})")
	g = r if g is None else g
	b = r if b is None else b
	a = np.ones_like(r) if a is None else a
	return np.dstack([r, g, b, a])


def read_float(path: Path) -> "np.ndarray | None":
	"""Linear float RGB(A) for HDR sources (for skies), or None for an 8-bit image."""
	ext = Path(path).suffix.lower()
	if ext in (".hdr", ".pic"):
		from .skybox import read_hdr

		return read_hdr(path)
	if ext == ".exr":
		return read_exr(path)
	if ext == ".pfm":
		return read_pfm(path)
	return None


def load_hdr(path: Path, warnings=None) -> "np.ndarray":
	from .skybox import read_hdr, tonemap

	rgb = tonemap(read_hdr(path))
	return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def load_exr(path: Path, warnings=None) -> "np.ndarray":
	return tc.float_to_rgba8(read_exr(path))


def load_pfm(path: Path, warnings=None) -> "np.ndarray":
	return tc.float_to_rgba8(read_pfm(path))


# -- vector, layered and camera files ----------------------------------------------------------------

SVG_LONG_SIDE = 1024	# rasterised so the long side is this; textures are shrunk later anyway


def load_svg(path: Path, warnings=None) -> "np.ndarray":
	import io

	from PIL import Image

	if tc.have("resvg_py"):
		import resvg_py

		png = bytes(resvg_py.svg_to_bytes(svg_path=str(path)))
		with Image.open(io.BytesIO(png)) as img:
			w, h = img.size
		zoom = SVG_LONG_SIDE / max(w, h, 1)
		if abs(zoom - 1) > 0.01:
			png = bytes(resvg_py.svg_to_bytes(svg_path=str(path), zoom=zoom))
	elif tc.have("cairosvg"):
		import gzip

		import cairosvg

		raw = Path(path).read_bytes()
		if raw[:2] == b"\x1f\x8b":
			raw = gzip.decompress(raw)
		png = cairosvg.svg2png(bytestring=raw, output_width=None)
		with Image.open(io.BytesIO(png)) as img:
			w, h = img.size
		scale = SVG_LONG_SIDE / max(w, h, 1)
		png = cairosvg.svg2png(bytestring=raw, scale=scale)
	else:
		raise MissingDependency("SVG images", "resvg-py")
	with Image.open(io.BytesIO(png)) as img:
		return pillow_rgba(img)


def load_zip_merged(path: Path, warnings=None) -> "np.ndarray":
	"""Krita .kra and OpenRaster .ora keep a flattened mergedimage.png."""
	import io
	import zipfile

	from PIL import Image

	with zipfile.ZipFile(path) as z:
		names = {n.lower(): n for n in z.namelist()}
		for want in ("mergedimage.png", "thumbnail.png", "preview.png"):
			if want in names:
				if want != "mergedimage.png" and warnings is not None:
					warnings.append(f"no mergedimage.png; using the smaller {want}")
				with Image.open(io.BytesIO(z.read(names[want]))) as img:
					return pillow_rgba(img)
	raise ValueError("no merged image inside the document")


def load_raw(path: Path, warnings=None) -> "np.ndarray":
	rawpy = tc.require("rawpy", "camera RAW files")
	with rawpy.imread(str(path)) as r:
		rgb = r.postprocess(use_camera_wb=True, output_bps=8)
	return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


# -- reporting -------------------------------------------------------------------------------------------


def report() -> list[dict]:
	"""One row per format: name, group, extensions, available, what to install."""
	rows = []
	for f in sorted(FORMATS, key=lambda f: GROUP_ORDER.index(f.group)):
		ok = available(f)
		rows.append({"name": f.name, "group": f.group, "extensions": list(f.extensions), "available": ok,
			"install": "" if ok or not f.package else f.package, "note": f.note})
	return rows


def groups() -> dict[str, list[str]]:
	"""Group → sorted extensions, for file dialogs."""
	out: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
	for f in FORMATS:
		out.setdefault(f.group, []).extend(f.extensions)
	return {g: sorted(set(e)) for g, e in out.items()}
