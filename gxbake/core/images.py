"""Loose images (PNG/TGA/BMP/JPG) and skyboxes to GX CMPR.

Skybox handling follows build_sky_cmpr_overlays in scripts/build-gamecube-disc.py:
six sides <name>{rt,bk,lf,ft,up,dn}, each resized to 128×128 and written as a
raw CMPR level (8 KiB) to media/sky_<name><side>.cmp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cmpr import cmpr_decode, gx_cmpr_encode
from .gct import GC_WT_FLAG_ALPHA, GctEntry, encode_mip_chain, wt_cache_key

GC_SKY_CMPR_SIZE = 128
GC_SKY_SUFFIXES = ("rt", "bk", "lf", "ft", "up", "dn")
IMAGE_EXTENSIONS = {".png", ".tga", ".bmp", ".jpg", ".jpeg", ".gif", ".webp", ".dds", ".tif", ".tiff", ".psd", ".pcx", ".qoi", ".ppm",
	".vtf",	# Source (vtf.py)
	".vtex_c",	# Source 2 (vtex.py)
	".wal",	# Quake II (quake.py)
	".blp", ".ftc", ".ftu",	# Blizzard BLP1/BLP2 and Fat textures, through Pillow
	".rgb", ".rgba", ".sgi", ".bw",	# SGI (Panda3D / Toontown textures), through Pillow
	".hdr"}	# Radiance HDR, tone-mapped (skybox.py)


def load_rgba(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	from PIL import Image

	ext = Path(path).suffix.lower()
	if ext == ".vtf":
		from .vtf import load_vtf

		return load_vtf(path)
	if ext == ".vtex_c":
		from . import vtex

		tex = vtex.load(path)
		if tex.cube and warnings is not None:
			warnings.append("cubemap: only the first face (+X) is packed")
		return tex.rgba(0)
	if ext == ".wal":
		from .quake import load_wal

		return load_wal(path, warnings)
	if ext == ".hdr":
		from .skybox import read_hdr, tonemap

		rgb = tonemap(read_hdr(path))
		return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
	with Image.open(path) as img:
		return np.asarray(img.convert("RGBA")).copy()


def has_transparency(rgba: "np.ndarray", threshold: int = 128) -> bool:
	return bool((rgba[..., 3] < threshold).any())


@dataclass
class ImageOptions:
	max_edge: int = 256	# clamp the long side; GC texture cache is 1 MiB
	mipmaps: bool = True
	alpha: str = "auto"	# auto | on | off  (CMPR has 1-bit alpha)
	alpha_threshold: int = 128
	resample: str = "lanczos"	# lanczos | bicubic | nearest
	cmpr_refine: bool = False	# least-squares CMPR endpoints (closer colours, same size)


def _target_size(w: int, h: int, max_edge: int) -> tuple[int, int]:
	tw = 1 << max(3, (w - 1).bit_length())
	th = 1 << max(3, (h - 1).bit_length())
	while max(tw, th) > max(8, max_edge):
		tw, th = max(8, tw // 2), max(8, th // 2)
	return tw, th


def encode_image(rgba: "np.ndarray", opts: ImageOptions, name: str = "") -> GctEntry:
	"""RGBA → power-of-two CMPR mip chain as a GCWT entry."""
	from PIL import Image

	h, w = rgba.shape[:2]
	tw, th = _target_size(w, h, opts.max_edge)
	use_alpha = opts.alpha == "on" or (opts.alpha == "auto" and has_transparency(rgba, opts.alpha_threshold))
	resample = {
		"lanczos": Image.Resampling.LANCZOS,
		"bicubic": Image.Resampling.BICUBIC,
		"nearest": Image.Resampling.NEAREST,
	}[opts.resample]
	rgb = rgba[..., :3]
	opaque = rgba[..., 3] >= opts.alpha_threshold if use_alpha else np.ones((h, w), bool)
	if (tw, th) != (w, h):
		rgb = np.asarray(Image.fromarray(rgb).resize((tw, th), resample))
		opaque = np.asarray(Image.fromarray(opaque).resize((tw, th), Image.Resampling.NEAREST))
	levels, body = encode_mip_chain(rgb, opaque, mipmaps=opts.mipmaps, refine=opts.cmpr_refine)
	tag = int(wt_cache_key(tw, th, rgb.tobytes(), b"", use_alpha)[:4], 16)
	return GctEntry(name[:15].lower(), tw, th, levels, GC_WT_FLAG_ALPHA if use_alpha else 0, tag, body)


def sky_side_cmpr(rgba: "np.ndarray", size: int = GC_SKY_CMPR_SIZE) -> bytes:
	from PIL import Image

	img = Image.fromarray(rgba[..., :3]).resize((size, size), Image.Resampling.LANCZOS)
	return gx_cmpr_encode(np.asarray(img))


def find_skybox_sides(path: Path) -> tuple[str, dict[str, Path]] | None:
	"""If `path` is one side of a skybox, return (name, {suffix: path}) for all six."""
	stem = path.stem.lower()
	for suffix in GC_SKY_SUFFIXES:
		if stem.endswith(suffix) and len(stem) > len(suffix):
			name = stem[: -len(suffix)]
			sides: dict[str, Path] = {}
			for sfx in GC_SKY_SUFFIXES:
				for cand in path.parent.iterdir():
					if cand.stem.lower() == name + sfx and cand.suffix.lower() in IMAGE_EXTENSIONS:
						sides[sfx] = cand
						break
			if len(sides) == 6:
				return name, sides
	return None


def decode_raw_cmp(data: bytes) -> "np.ndarray":
	"""Raw .cmp files carry no header; assume a square level (sky sides are 128²)."""
	texels = len(data) * 2
	side = int(round(texels ** 0.5))
	if side * side != texels:
		raise ValueError(f"cannot infer square size from {len(data)} bytes")
	return cmpr_decode(data, side, side)


def psnr(a: "np.ndarray", b: "np.ndarray") -> float:
	"""PSNR over RGB of two equal-size arrays, over texels opaque in both."""
	mask = np.ones(a.shape[:2], bool)
	if a.shape[-1] == 4:
		mask &= a[..., 3] >= 128
	if b.shape[-1] == 4:
		mask &= b[..., 3] >= 128
	a = a.astype(np.float64)
	b = b.astype(np.float64)
	if not mask.any():
		return float("inf")
	mse = ((a[..., :3] - b[..., :3]) ** 2)[mask].mean()
	return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
