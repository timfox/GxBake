"""Every GX texture format Flipper samples, encoded and decoded, plus a picker.

The port uploads studio skins and lightmaps as 16-bit RGB565 / RGB5A3 and only
world textures as CMPR. Most of that colour precision is wasted: a GoldSrc skin
has at most 256 colours, so the same pixels as colour-indexed CI8 with an
RGB565 TLUT are bit-identical on screen at half the bytes, and CI4 (16 colours)
or the intensity formats (I4/I8/IA4/IA8) are a quarter to half again. Flipper
reads all of them natively, with no CPU work per draw.

Texel layouts (Yet Another GameCube Documentation, libogc gx.h):

	format   bpp  tile (w×h)  texel
	I4        4   8×8         intensity nibble, high nibble first
	IA4       8   8×4         alpha<<4 | intensity
	I8        8   8×4         intensity
	IA8      16   4×4         alpha byte, intensity byte
	RGB565   16   4×4         big-endian
	RGB5A3   16   4×4         1rrrrrgggggbbbbb opaque, or 0aaarrrrggggbbbb
	RGBA8    32   4×4         AR plane (32 B) then GB plane (32 B) per tile
	CI4       4   8×8         index nibble, high nibble first
	CI8       8   8×4         index
	CMPR      4   8×8         2×2 DXT1 blocks per tile (cmpr.py)

TLUTs (palettes) are 16-bit entries in IA8, RGB565 or RGB5A3, big-endian,
loaded with GX_InitTlutObj/GX_LoadTlut, and their entry count must be a
multiple of 16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cmpr import cmpr_decode, cmpr_refine, gx_cmpr_encode_blocks

# GXTexFmt values from libogc gx.h
GX_TF_I4, GX_TF_I8, GX_TF_IA4, GX_TF_IA8 = 0x0, 0x1, 0x2, 0x3
GX_TF_RGB565, GX_TF_RGB5A3, GX_TF_RGBA8 = 0x4, 0x5, 0x6
GX_TF_CI4, GX_TF_CI8, GX_TF_CMPR = 0x8, 0x9, 0xE
# GXTlutFmt
GX_TL_IA8, GX_TL_RGB565, GX_TL_RGB5A3 = 0, 1, 2

FORMATS = {
	"I4": (GX_TF_I4, 4, 8, 8), "I8": (GX_TF_I8, 8, 8, 4), "IA4": (GX_TF_IA4, 8, 8, 4),
	"IA8": (GX_TF_IA8, 16, 4, 4), "RGB565": (GX_TF_RGB565, 16, 4, 4), "RGB5A3": (GX_TF_RGB5A3, 16, 4, 4),
	"RGBA8": (GX_TF_RGBA8, 32, 4, 4), "CI4": (GX_TF_CI4, 4, 8, 8), "CI8": (GX_TF_CI8, 8, 8, 4),
	"CMPR": (GX_TF_CMPR, 4, 8, 8),
}
FORMAT_NAMES = {v[0]: k for k, v in FORMATS.items()}
TLUT_NAMES = {GX_TL_IA8: "IA8", GX_TL_RGB565: "RGB565", GX_TL_RGB5A3: "RGB5A3"}


def padded(w: int, h: int, fmt: str) -> tuple[int, int]:
	_, _, tw, th = FORMATS[fmt]
	return -(-w // tw) * tw, -(-h // th) * th


def tex_size(w: int, h: int, fmt: str) -> int:
	"""Bytes GX_GetTexBufferSize gives for one level (sides padded to whole tiles)."""
	pw, ph = padded(w, h, fmt)
	return pw * ph * FORMATS[fmt][1] // 8


# --- tiling -----------------------------------------------------------------

def _tile(a: "np.ndarray", tw: int, th: int) -> "np.ndarray":
	"""H×W(×C) array → texels in GX tile order (tiles row-major, texels row-major in a tile)."""
	h, w = a.shape[:2]
	rest = a.shape[2:]
	t = a.reshape(h // th, th, w // tw, tw, *rest).swapaxes(1, 2)
	return t.reshape(-1, *rest)


def _untile(flat: "np.ndarray", w: int, h: int, tw: int, th: int) -> "np.ndarray":
	rest = flat.shape[1:]
	t = flat.reshape(h // th, w // tw, th, tw, *rest).swapaxes(1, 2)
	return t.reshape(h, w, *rest)


def _pad(a: "np.ndarray", w: int, h: int) -> "np.ndarray":
	ph, pw = h - a.shape[0], w - a.shape[1]
	if ph or pw:
		a = np.pad(a, ((0, ph), (0, pw)) + ((0, 0),) * (a.ndim - 2), mode="edge")
	return a


def _nibbles(v: "np.ndarray") -> bytes:
	v = v.astype(np.uint8)
	return ((v[0::2] << 4) | (v[1::2] & 15)).astype(np.uint8).tobytes()


def _unnibble(data: bytes, n: int) -> "np.ndarray":
	b = np.frombuffer(data, np.uint8, n // 2)
	out = np.empty(n, np.uint8)
	out[0::2], out[1::2] = b >> 4, b & 15
	return out


# --- 16-bit colour words ----------------------------------------------------

def _q(v: "np.ndarray", bits: int) -> "np.ndarray":
	"""8-bit → n-bit, rounded."""
	return (v.astype(np.int32) * ((1 << bits) - 1) + 127) // 255


def _x(v: "np.ndarray", bits: int) -> "np.ndarray":
	"""n-bit → 8-bit the way Flipper expands (bit replication)."""
	v = v.astype(np.int32)
	return ((v << (8 - bits)) | (v >> (2 * bits - 8))) if bits >= 4 else v * 255 // ((1 << bits) - 1)


def rgb565(rgb: "np.ndarray") -> "np.ndarray":
	return ((_q(rgb[..., 0], 5) << 11) | (_q(rgb[..., 1], 6) << 5) | _q(rgb[..., 2], 5)).astype(np.uint16)


def from_rgb565(v: "np.ndarray") -> "np.ndarray":
	v = v.astype(np.int32)
	return np.stack((_x(v >> 11 & 31, 5), _x(v >> 5 & 63, 6), _x(v & 31, 5), np.full(v.shape, 255)), -1).astype(np.uint8)


def rgb5a3(rgba: "np.ndarray") -> "np.ndarray":
	r, g, b, a = (rgba[..., i] for i in range(4))
	opaque = (0x8000 | (_q(r, 5) << 10) | (_q(g, 5) << 5) | _q(b, 5))
	trans = (_q(a, 3) << 12) | (_q(r, 4) << 8) | (_q(g, 4) << 4) | _q(b, 4)
	return np.where(a >= 0xF0, opaque, trans).astype(np.uint16)


def from_rgb5a3(v: "np.ndarray") -> "np.ndarray":
	v = v.astype(np.int32)
	op = (v & 0x8000) != 0
	r = np.where(op, _x(v >> 10 & 31, 5), _x(v >> 8 & 15, 4))
	g = np.where(op, _x(v >> 5 & 31, 5), _x(v >> 4 & 15, 4))
	b = np.where(op, _x(v & 31, 5), _x(v & 15, 4))
	a = np.where(op, 255, _x(v >> 12 & 7, 3))
	return np.stack((r, g, b, a), -1).astype(np.uint8)


def ia8(ia: "np.ndarray") -> "np.ndarray":
	"""H×W×2 (intensity, alpha) → alpha<<8 | intensity, as GX stores IA8."""
	return ((ia[..., 1].astype(np.uint16) << 8) | ia[..., 0]).astype(np.uint16)


def from_ia8(v: "np.ndarray") -> "np.ndarray":
	i, a = (v & 255).astype(np.uint8), (v >> 8).astype(np.uint8)
	return np.stack((i, i, i, a), -1)


def tlut_words(rgba: "np.ndarray", tlut: int) -> "np.ndarray":
	if tlut == GX_TL_RGB565:
		return rgb565(rgba[..., :3])
	if tlut == GX_TL_RGB5A3:
		return rgb5a3(rgba)
	return ia8(np.stack((luma(rgba[..., :3]), rgba[..., 3]), -1))


def tlut_rgba(words: "np.ndarray", tlut: int) -> "np.ndarray":
	return {GX_TL_RGB565: from_rgb565, GX_TL_RGB5A3: from_rgb5a3, GX_TL_IA8: from_ia8}[tlut](words)


def luma(rgb: "np.ndarray") -> "np.ndarray":
	return np.clip((rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) + 0.5, 0, 255).astype(np.uint8)


# --- encode / decode --------------------------------------------------------

@dataclass
class GxTexture:
	fmt: str
	width: int
	height: int
	data: bytes
	tlut: bytes = b""	# big-endian 16-bit entries, padded to 16
	tlut_fmt: int = GX_TL_RGB565

	@property
	def size(self) -> int:
		return len(self.data) + len(self.tlut)

	def decode(self) -> "np.ndarray":
		return decode(self.fmt, self.width, self.height, self.data, self.tlut, self.tlut_fmt)


def encode(rgba: "np.ndarray", fmt: str) -> GxTexture:
	"""H×W×4 uint8 → one GX level in a direct (non-indexed) format.
	For CI4/CI8 use encode_indexed."""
	h, w = rgba.shape[:2]
	_, bpp, tw, th = FORMATS[fmt]
	pw, ph = padded(w, h, fmt)
	a = _pad(rgba, pw, ph)
	if fmt == "CMPR":
		opaque = a[..., 3] >= 128
		px = np.ascontiguousarray(a[..., :3])
		return GxTexture(fmt, w, h, cmpr_refine(px, opaque, gx_cmpr_encode_blocks(px, opaque)))
	t = _tile(a, tw, th)
	if fmt == "I4":
		data = _nibbles(_q(luma(t), 4))
	elif fmt == "I8":
		data = luma(t).tobytes()
	elif fmt == "IA4":
		data = ((_q(t[:, 3], 4) << 4) | _q(luma(t), 4)).astype(np.uint8).tobytes()
	elif fmt == "IA8":
		data = np.stack((t[:, 3], luma(t)), -1).astype(np.uint8).tobytes()
	elif fmt == "RGB565":
		data = rgb565(t[:, :3]).astype(">u2").tobytes()
	elif fmt == "RGB5A3":
		data = rgb5a3(t).astype(">u2").tobytes()
	elif fmt == "RGBA8":
		blocks = t.reshape(-1, 16, 4)
		ar = blocks[:, :, [3, 0]].reshape(-1, 32)
		gb = blocks[:, :, [1, 2]].reshape(-1, 32)
		data = np.concatenate((ar, gb), 1).astype(np.uint8).tobytes()
	else:
		raise ValueError(f"{fmt} is indexed or unknown; use encode_indexed")
	return GxTexture(fmt, w, h, data)


def encode_indexed(index: "np.ndarray", palette_rgba: "np.ndarray", tlut: int = GX_TL_RGB565,
		fmt: str | None = None) -> GxTexture:
	"""H×W indices + N×4 palette → CI4 (N ≤ 16) or CI8, TLUT padded to 16 entries."""
	n = len(palette_rgba)
	fmt = fmt or ("CI4" if n <= 16 else "CI8")
	if n > (16 if fmt == "CI4" else 256):
		raise ValueError(f"{n} colours do not fit {fmt}")
	h, w = index.shape
	_, _, tw, th = FORMATS[fmt]
	pw, ph = padded(w, h, fmt)
	t = _tile(_pad(index.astype(np.uint8), pw, ph), tw, th)
	data = _nibbles(t) if fmt == "CI4" else t.astype(np.uint8).tobytes()
	pal = np.zeros((-(-n // 16) * 16, 4), np.uint8)
	pal[:n] = palette_rgba
	return GxTexture(fmt, w, h, data, tlut_words(pal, tlut).astype(">u2").tobytes(), tlut)


def decode_indices(t: GxTexture) -> "np.ndarray":
	"""H×W TLUT indices of a CI4/CI8 texture (untiled)."""
	_, _, tw, th = FORMATS[t.fmt]
	pw, ph = padded(t.width, t.height, t.fmt)
	n = pw * ph
	idx = _unnibble(t.data, n) if t.fmt == "CI4" else np.frombuffer(t.data, np.uint8, n)
	tiles = idx.reshape(ph // th, pw // tw, th, tw).transpose(0, 2, 1, 3).reshape(ph, pw)
	return tiles[:t.height, :t.width]


def decode(fmt: str, w: int, h: int, data: bytes, tlut: bytes = b"", tlut_fmt: int = GX_TL_RGB565) -> "np.ndarray":
	"""One GX level → H×W×4 RGBA, as Flipper samples it."""
	_, bpp, tw, th = FORMATS[fmt]
	pw, ph = padded(w, h, fmt)
	n = pw * ph
	if fmt == "CMPR":
		out = cmpr_decode(data, pw, ph)
		if out.shape[-1] == 3:
			out = np.concatenate((out, np.full(out.shape[:2] + (1,), 255, np.uint8)), -1)
		return out[:h, :w]
	if fmt in ("CI4", "CI8"):
		idx = _unnibble(data, n) if fmt == "CI4" else np.frombuffer(data, np.uint8, n)
		pal = tlut_rgba(np.frombuffer(tlut, ">u2"), tlut_fmt)
		flat = pal[np.minimum(idx, len(pal) - 1)]
	elif fmt == "I4":
		i = _x(_unnibble(data, n), 4).astype(np.uint8)
		flat = np.stack((i, i, i, np.full(n, 255, np.uint8)), -1)
	elif fmt == "I8":
		i = np.frombuffer(data, np.uint8, n)
		flat = np.stack((i, i, i, np.full(n, 255, np.uint8)), -1)
	elif fmt == "IA4":
		b = np.frombuffer(data, np.uint8, n)
		i, a = _x(b & 15, 4).astype(np.uint8), _x(b >> 4, 4).astype(np.uint8)
		flat = np.stack((i, i, i, a), -1)
	elif fmt == "IA8":
		b = np.frombuffer(data, np.uint8, n * 2).reshape(n, 2)
		flat = np.stack((b[:, 1], b[:, 1], b[:, 1], b[:, 0]), -1)
	elif fmt == "RGB565":
		flat = from_rgb565(np.frombuffer(data, ">u2", n))
	elif fmt == "RGB5A3":
		flat = from_rgb5a3(np.frombuffer(data, ">u2", n))
	elif fmt == "RGBA8":
		b = np.frombuffer(data, np.uint8, n * 4).reshape(-1, 2, 16, 2)
		flat = np.stack((b[:, 0, :, 1], b[:, 1, :, 0], b[:, 1, :, 1], b[:, 0, :, 0]), -1).reshape(n, 4)
	else:
		raise ValueError(fmt)
	return _untile(flat, pw, ph, tw, th)[:h, :w]


# --- choosing ---------------------------------------------------------------

def psnr(a: "np.ndarray", b: "np.ndarray", mask: "np.ndarray | None" = None) -> float:
	"""PSNR over RGB (and alpha where it matters) in dB; inf when identical."""
	d = a.astype(np.float32) - b.astype(np.float32)
	if mask is not None:
		d = d[mask]
	mse = float((d ** 2).mean()) if d.size else 0.0
	return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


@dataclass
class Choice:
	tex: GxTexture
	psnr: float	# against the 16-bit format the port uses today
	baseline: int	# bytes of that 16-bit upload
	tried: dict[str, tuple[int, float]]	# format → (bytes, dB)


def _is_grey(rgb: "np.ndarray", tol: int = 3) -> bool:
	c = rgb.astype(np.int16)
	return int(np.abs(c[..., 0] - c[..., 1]).max(initial=0)) <= tol and int(np.abs(c[..., 1] - c[..., 2]).max(initial=0)) <= tol


_LUMA_W = np.sqrt(np.array([0.299, 0.587, 0.114]))


def quantize_ci4(index: "np.ndarray", pal: "np.ndarray", alpha: bool, iters: int = 12) -> "tuple[np.ndarray, np.ndarray] | None":
	"""16-colour palette for an indexed image by weighted k-means over the
	colours actually used (each weighted by its pixel count), seeded by
	weighted farthest-point picks so it is deterministic. A transparent
	colour keeps its own entry. Returns (new index, 16×4 palette)."""
	counts = np.bincount(index.reshape(-1), minlength=len(pal)).astype(np.float64)
	solid = np.flatnonzero((pal[:, 3] > 0) & (counts > 0)) if alpha else np.flatnonzero(counts > 0)
	k = 15 if alpha else 16
	if len(solid) <= k:
		return None
	x = pal[solid, :3].astype(np.float64) * _LUMA_W	# luma-weighted distance
	wts = counts[solid]
	cent = [x[np.argmax(wts)]]
	d = ((x - cent[0]) ** 2).sum(1)
	for _ in range(k - 1):
		cent.append(x[np.argmax(d * wts)])
		d = np.minimum(d, ((x - cent[-1]) ** 2).sum(1))
	c = np.array(cent)
	for _ in range(iters):
		lab = ((x[:, None] - c[None]) ** 2).sum(2).argmin(1)
		for j in range(k):
			m = lab == j
			if m.any():
				c[j] = (x[m] * wts[m, None]).sum(0) / wts[m].sum()
	lab = ((x[:, None] - c[None]) ** 2).sum(2).argmin(1)
	out_pal = np.zeros((16, 4), np.uint8)
	out_pal[:k, :3] = np.clip(np.round(c / _LUMA_W), 0, 255)
	out_pal[:k, 3] = 255
	lut = np.full(len(pal), 15, np.uint8)	# transparent entries → 15 (zeroed)
	lut[solid] = lab
	return lut[index], out_pal


def choose_indexed(index: "np.ndarray", palette_rgb: "np.ndarray", masked: bool, *,
		min_db: float = 40.0, allow_cmpr: bool = True) -> Choice:
	"""Smallest GX format for a palettised skin whose picture stays within
	`min_db` of the RGB565/RGB5A3 upload the port does today.

	Lossless candidates first (CI4/CI8 over the colours actually used, with
	the same 16-bit TLUT the port would have converted to), then lossy ones
	(CMPR, I4/I8/IA4 for grey skins) only if they clear `min_db`.
	"""
	h, w = index.shape
	used, remap = np.unique(index, return_inverse=True)
	remap = remap.reshape(h, w)
	pal = np.concatenate((palette_rgb[used], np.full((len(used), 1), 255, np.uint8)), 1)
	alpha = masked and 255 in used
	if alpha:
		pal[used == 255] = (0, 0, 0, 0)
	tlut = GX_TL_RGB5A3 if alpha else GX_TL_RGB565
	base_fmt = "RGB5A3" if alpha else "RGB565"
	rgba = np.concatenate((palette_rgb[index], np.full((h, w, 1), 255, np.uint8)), -1)
	if alpha:
		rgba[index == 255] = 0
	ref = encode(rgba, base_fmt).decode()
	baseline = tex_size(w, h, base_fmt)
	vis = ref[..., 3] > 0

	tried: dict[str, tuple[int, float]] = {}
	# Colours that differ in 8 bits but land on the same 16-bit TLUT word are
	# one colour on the GameCube: merging them is lossless and can drop CI8 to CI4.
	words, first, merged = np.unique(tlut_words(pal, tlut), return_index=True, return_inverse=True)
	cands: list[GxTexture] = [encode_indexed(merged.reshape(-1)[remap], pal[first], tlut)]
	if len(words) > 16:
		q = quantize_ci4(remap, pal, alpha)
		if q is not None:
			cands.append(encode_indexed(q[0], q[1], tlut, "CI4"))
	if allow_cmpr:
		cands.append(encode(rgba, "CMPR"))
	if _is_grey(palette_rgb[used]):
		cands += [encode(rgba, f) for f in (("IA4",) if alpha else ("I4", "I8"))]
	best: tuple[int, GxTexture, float] | None = None
	for t in cands:
		d = t.decode()
		db = psnr(d[..., :3], ref[..., :3], vis) if vis.any() else float("inf")
		if alpha:
			db = min(db, psnr(d[..., 3] > 127, ref[..., 3] > 127))
		tried[t.fmt] = (t.size, db)
		lossless = t.fmt in ("CI4", "CI8") and db == float("inf")
		ok = lossless or db >= min_db
		# Smaller wins; within 64 bytes (a TLUT's worth) the closer picture wins.
		if ok and (best is None or t.size < best[0] - 64 or (abs(t.size - best[0]) <= 64 and db > best[2])):
			best = (t.size, t, db)
	assert best is not None
	return Choice(best[1], best[2], baseline, tried)
