"""Pixel decoding shared by the texture readers in gputex, consoletex and retrotex.

Everything here turns raw texel bytes into H×W×4 uint8 RGBA:

- `unpack_packed`: 8/16/32-bit packed pixels described as bit fields, any
  endianness (RGB565, RGB5A3-like, ARGB1555, ABGR1555, RGBA4444, ...);
- `unpack_bytes`: byte-per-channel layouts in any order (RGBA8, BGRA8, LA8, ...);
- `decode_blocks`: block compression (BC1–BC7 through Pillow's built-in "bcn"
  decoder; ETC1/ETC2/EAC through texture2ddecoder when it is installed, else
  etc.py; ASTC, PVRTC and ATC through texture2ddecoder only);
- swizzles (Morton/Z-order as on Xbox and 3DS, PowerVR "twiddling");
- palette lookups and sprite sheets;
- `float_to_rgba8` for HDR sources (EXR, PFM, float KTX/DDS, BC6H).

Optional dependencies raise `MissingDependency`, whose message says what to
install, so the app and CLI can show it as a per-file error.
"""

from __future__ import annotations

import numpy as np


class MissingDependency(RuntimeError):
	"""A decoder needs an optional package; the message says which and how to get it."""

	def __init__(self, what: str, package: str, extra: str | None = None):
		extra = extra or ("jxl" if package == "pillow-jxl-plugin" else "images")
		super().__init__(f"{what} needs the optional '{package}' package (pip install {package}, "
			f"or pip install gxbake[{extra}])")
		self.package = package


def require(module: str, what: str, package: str | None = None):
	"""Import an optional module or raise MissingDependency."""
	import importlib

	try:
		return importlib.import_module(module)
	except ImportError:
		raise MissingDependency(what, package or module) from None


def have(module: str) -> bool:
	import importlib.util

	try:
		return importlib.util.find_spec(module) is not None
	except (ImportError, ValueError):
		return False


# -- packed and byte layouts ---------------------------------------------------------


def _expand(v: "np.ndarray", bits: int) -> "np.ndarray":
	"""n-bit unsigned → 0..255 by bit replication (what GPUs do)."""
	if bits >= 8:
		return (v >> (bits - 8)).astype(np.uint8)
	if bits == 0:
		return np.zeros(v.shape, np.uint8)
	return ((v.astype(np.uint32) * 255 + ((1 << bits) - 1) // 2) // ((1 << bits) - 1)).astype(np.uint8)


def unpack_packed(data: bytes, w: int, h: int, fields: str, word: int = 16, big: bool = False,
		offset: int = 0) -> "np.ndarray":
	"""Packed pixels → RGBA. `fields` lists channels from the most significant bit,
	e.g. "R5G6B5", "A1R5G5B5", "A4R4G4B4", "R5G5B5A1", "X1B5G5R5". L is grey,
	X is ignored. Missing alpha is opaque."""
	import re

	spec = [(c, int(n)) for c, n in re.findall(r"([RGBALX])(\d+)", fields)]
	if sum(n for _, n in spec) != word:
		raise ValueError(f"{fields} is not {word} bits")
	n = w * h
	dt = {8: "u1", 16: "u2", 32: "u4"}[word]
	v = np.frombuffer(data, (">" if big else "<") + dt, n, offset).astype(np.uint32)
	out = np.zeros((n, 4), np.uint8)
	out[:, 3] = 255
	shift = word
	for c, bits in spec:
		shift -= bits
		if c == "X":
			continue
		ch = _expand((v >> shift) & ((1 << bits) - 1), bits)
		if c == "L":
			out[:, 0] = out[:, 1] = out[:, 2] = ch
		else:
			out[:, "RGBA".index(c)] = ch
	return out.reshape(h, w, 4)


def unpack_bytes(data: bytes, w: int, h: int, order: str, offset: int = 0) -> "np.ndarray":
	"""One byte per channel → RGBA. `order` is the byte order, e.g. "RGBA", "BGRA",
	"ARGB", "BGR", "LA", "L", "A", "R", "RG", "BGRX"."""
	k = len(order)
	px = np.frombuffer(data, np.uint8, w * h * k, offset).reshape(-1, k)
	out = np.zeros((w * h, 4), np.uint8)
	out[:, 3] = 255
	for i, c in enumerate(order):
		if c == "L":
			out[:, 0] = out[:, 1] = out[:, 2] = px[:, i]
		elif c in "RGBA":
			out[:, "RGBA".index(c)] = px[:, i]
	if order == "A":
		out[:, :3] = 255
	return out.reshape(h, w, 4)


def unpack_float(data: bytes, w: int, h: int, channels: str, dtype: str = "<f2", offset: int = 0) -> "np.ndarray":
	"""Float channels (half or single) → H×W×4 float32 RGBA (alpha 1 when absent)."""
	k = len(channels)
	v = np.frombuffer(data, dtype, w * h * k, offset).astype(np.float32).reshape(-1, k)
	out = np.zeros((w * h, 4), np.float32)
	out[:, 3] = 1.0
	for i, c in enumerate(channels):
		if c == "L":
			out[:, 0] = out[:, 1] = out[:, 2] = v[:, i]
		elif c in "RGBA":
			out[:, "RGBA".index(c)] = v[:, i]
	return out.reshape(h, w, 4)


def srgb_encode(lin: "np.ndarray") -> "np.ndarray":
	lin = np.clip(lin, 0, 1)
	return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def float_to_rgba8(rgba: "np.ndarray", linear: bool = True) -> "np.ndarray":
	"""Float RGB(A) → uint8 RGBA. Values within [0, 1] are only sRGB-encoded (when
	linear); brighter images are tone-mapped like Radiance .hdr skies."""
	rgba = np.nan_to_num(np.asarray(rgba, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	if rgba.ndim == 2:
		rgba = rgba[..., None]
	if rgba.shape[-1] == 1:
		rgba = np.concatenate([rgba] * 3, -1)
	if rgba.shape[-1] == 2:	# luminance + alpha
		rgba = np.concatenate([rgba[..., :1]] * 3 + [rgba[..., 1:]], -1)
	rgb = np.maximum(rgba[..., :3], 0)
	alpha = rgba[..., 3] if rgba.shape[-1] > 3 else np.ones(rgb.shape[:2], np.float32)
	if rgb.size and float(np.percentile(rgb, 99.5)) > 1.0:
		from .skybox import tonemap

		out8 = tonemap(rgb)
	else:
		out8 = np.clip((srgb_encode(rgb) if linear else rgb) * 255 + 0.5, 0, 255).astype(np.uint8)
	a8 = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
	return np.dstack([out8, a8])


# -- block compression -----------------------------------------------------------------

# bytes per 4×4 (or ASTC n×m) block
_BLOCK_BYTES = {"bc1": 8, "bc2": 16, "bc3": 16, "bc4": 8, "bc5": 16, "bc6h": 16, "bc6h_sf": 16, "bc7": 16,
	"etc1": 8, "etc2": 8, "etc2_a1": 8, "etc2_eac": 16, "r11": 8, "rg11": 16, "atc": 8, "atc_a": 16}


def block_dims(kind: str) -> tuple[int, int]:
	if kind.startswith("astc"):
		bw, bh = kind[4:].split("x")
		return int(bw), int(bh)
	return 4, 4


def compressed_size(kind: str, w: int, h: int) -> int:
	"""Bytes one level of `kind` occupies."""
	if kind == "pvrtc4":
		return max(w, 8) * max(h, 8) // 2
	if kind == "pvrtc2":
		return max(w, 16) * max(h, 8) // 4
	bw, bh = block_dims(kind)
	return -(-w // bw) * -(-h // bh) * (16 if kind.startswith("astc") else _BLOCK_BYTES[kind])


def is_compressed(kind: str) -> bool:
	return kind in _BLOCK_BYTES or kind.startswith("astc") or kind.startswith("pvrtc")


def _pillow_bcn(raw: bytes, w: int, h: int, n, mode: str) -> "np.ndarray":
	from PIL import Image

	pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
	need = (pw // 4) * (ph // 4) * (8 if n in (1, 4) else 16)
	if len(raw) < need:
		raise ValueError(f"truncated BC{n if isinstance(n, int) else n[0]} data ({len(raw)} of {need} bytes)")
	return np.asarray(Image.frombytes(mode, (pw, ph), raw[:need], "bcn", n))[:h, :w]


def _t2d(name: str, raw: bytes, w: int, h: int, *extra) -> "np.ndarray":
	t2d = require("texture2ddecoder", f"{name.upper()} textures")
	fn = getattr(t2d, "decode_" + name)
	bgra = np.frombuffer(fn(bytes(raw), w, h, *extra), np.uint8).reshape(h, w, 4)
	return bgra[..., [2, 1, 0, 3]].copy()


def decode_blocks(kind: str, raw: bytes, w: int, h: int, *, float_out: bool = False) -> "np.ndarray":
	"""Compressed level → RGBA uint8 (BC6H returns float32 RGBA when float_out)."""
	raw = bytes(raw)
	if kind == "bc1":
		return _pillow_bcn(raw, w, h, 1, "RGBA")
	if kind == "bc2":
		return _pillow_bcn(raw, w, h, 2, "RGBA")
	if kind == "bc3":
		return _pillow_bcn(raw, w, h, 3, "RGBA")
	if kind == "bc4":
		g = _pillow_bcn(raw, w, h, 4, "L")
		return np.dstack([g, g, g, np.full_like(g, 255)])
	if kind == "bc5":
		rg = _pillow_bcn(raw, w, h, 5, "RGB")
		return np.dstack([rg[..., 0], rg[..., 1], np.zeros_like(rg[..., 0]), np.full_like(rg[..., 0], 255)])
	if kind == "bc7":
		return _pillow_bcn(raw, w, h, 7, "RGBA")
	if kind in ("bc6h", "bc6h_sf"):
		# Pillow clamps BC6H to [0, 1] linear; good enough for textures, sRGB-encode it.
		lin = _pillow_bcn(raw, w, h, (6, "BC6HS" if kind == "bc6h_sf" else "BC6H"), "RGB").astype(np.float32) / 255
		rgba = np.dstack([lin, np.ones(lin.shape[:2], np.float32)])
		return rgba if float_out else float_to_rgba8(rgba)
	if kind in ("etc1", "etc2", "etc2_eac", "r11", "rg11", "etc2_a1"):
		if have("texture2ddecoder"):
			name = {"etc1": "etc1", "etc2": "etc2", "etc2_a1": "etc2a1", "etc2_eac": "etc2a8", "r11": "eacr",
				"rg11": "eacrg"}[kind]
			out = _t2d(name, raw, w, h)
			if kind == "r11":
				out[..., 1] = out[..., 2] = out[..., 0]
			return out
		if kind == "etc2_a1":
			raise MissingDependency("ETC2 punch-through alpha textures", "texture2ddecoder")
		from . import etc

		return etc.decode(raw, w, h, "etc2" if kind == "etc1" else kind)
	if kind.startswith("astc"):
		bw, bh = block_dims(kind)
		return _t2d("astc", raw, w, h, bw, bh)
	if kind in ("pvrtc2", "pvrtc4"):
		return _t2d("pvrtc", raw, w, h, kind == "pvrtc2")
	if kind == "atc":
		return _t2d("atc_rgb4", raw, w, h)
	if kind == "atc_a":
		return _t2d("atc_rgba8", raw, w, h)
	raise ValueError(f"unknown block format {kind}")


# -- swizzles ------------------------------------------------------------------------------


def _part1by1(v: "np.ndarray") -> "np.ndarray":
	v = v.astype(np.uint64) & 0xFFFF
	v = (v | (v << 8)) & 0x00FF00FF
	v = (v | (v << 4)) & 0x0F0F0F0F
	v = (v | (v << 2)) & 0x33333333
	v = (v | (v << 1)) & 0x55555555
	return v


def morton_index(w: int, h: int, x_first: bool = True) -> "np.ndarray":
	"""For each (y, x) of a w×h grid, its position in Morton (Z) order. Non-square
	grids interleave the shared low bits and append the rest of the longer side."""
	ys, xs = np.mgrid[0:h, 0:w]
	ys, xs = ys.astype(np.uint64), xs.astype(np.uint64)
	k = min(w, h).bit_length() - 1	# bits interleaved
	m = (1 << k) - 1
	lo = (_part1by1(xs & m) | (_part1by1(ys & m) << 1)) if x_first else (_part1by1(ys & m) | (_part1by1(xs & m) << 1))
	hi = (xs >> k) if w > h else (ys >> k)
	return (hi << (2 * k)) | lo


def unswizzle_morton(flat: "np.ndarray", w: int, h: int, x_first: bool = True) -> "np.ndarray":
	"""Texels stored in Morton order (Xbox swizzled, 3DS tiles) → row-major (h, w, ...)."""
	idx = morton_index(w, h, x_first)
	return flat[idx.reshape(-1).astype(np.intp)].reshape((h, w) + flat.shape[1:])


def twiddle_index(w: int, h: int) -> "np.ndarray":
	"""PowerVR (Dreamcast) twiddled order: Morton with y in the low bit."""
	return morton_index(w, h, x_first=False)


def untile(flat: "np.ndarray", w: int, h: int, tw: int, th: int) -> "np.ndarray":
	"""Row-major tiles of tw×th texels, tiles row-major → (h, w, ...) (w, h multiples of the tile)."""
	rest = flat.shape[1:]
	a = flat.reshape((h // th, w // tw, th, tw) + rest)
	return a.swapaxes(1, 2).reshape((h, w) + rest)


# -- palettes and sheets ---------------------------------------------------------------------


def palette_rgba(pal_rgb: bytes, *, six_bit: bool = False, transparent: int | None = None, count: int = 256) -> "np.ndarray":
	"""768-byte RGB palette (or 6-bit VGA DAC values) → count×4 RGBA."""
	p = np.zeros((count, 4), np.uint8)
	src = np.frombuffer(pal_rgb[:count * 3], np.uint8)
	src = src[: len(src) // 3 * 3].reshape(-1, 3)
	if six_bit or (src.size and src.max() < 64 and six_bit is None):
		src = (src.astype(np.uint16) * 255 // 63).astype(np.uint8)
	p[: len(src), :3] = src
	p[:, 3] = 255
	if transparent is not None:
		p[transparent, 3] = 0
	return p


GREY_RGBA = palette_rgba(bytes(np.repeat(np.arange(256, dtype=np.uint8), 3)))


def looks_six_bit(pal_rgb: bytes) -> bool:
	"""VGA DAC palettes (Build, Westwood, Fallout) only use 0..63."""
	return bool(pal_rgb) and max(pal_rgb[:768]) < 64


def sheet(frames: list["np.ndarray"], max_width: int = 1024, pad: int = 0) -> "np.ndarray":
	"""Lay RGBA frames out left to right in rows (a sprite sheet); transparent gaps."""
	frames = [f for f in frames if f.size]
	if not frames:
		raise ValueError("no frames with pixels")
	if len(frames) == 1:
		return frames[0]
	rows, row, x, rw = [], [], 0, 0
	for f in frames:
		fw = f.shape[1]
		if row and x + fw > max_width:
			rows.append(row)
			row, x = [], 0
		row.append(f)
		x += fw + pad
	rows.append(row)
	width = max(sum(f.shape[1] + pad for f in r) - pad for r in rows)
	height = sum(max(f.shape[0] for f in r) for r in rows) + pad * (len(rows) - 1)
	out = np.zeros((height, width, 4), np.uint8)
	y = 0
	for r in rows:
		x = 0
		for f in r:
			out[y:y + f.shape[0], x:x + f.shape[1]] = f
			x += f.shape[1] + pad
		y += max(f.shape[0] for f in r) + pad
	return out
