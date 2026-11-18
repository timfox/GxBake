"""GX CMPR (S3TC/DXT1, big-endian, 8x8 tiled) encode and decode.

The two encoders are ported unchanged from the Xash3D GameCube disc builder
(scripts/build-gamecube-disc.py: gx_cmpr_encode, gx_cmpr_encode_blocks) so
GxBake produces byte-identical output to the existing pipeline. The decoder
is new and exists so the GUI can show what Flipper will actually sample.
"""

from __future__ import annotations

import struct

import numpy as np


def gx_cmpr_encode(rgb: "np.ndarray") -> bytes:
	"""Encode an H×W×3 uint8 image (H, W multiples of 8) as GX CMPR (DXT1).

	Endpoints are the brightest and darkest pixel of each 4×4 block (enough
	for skies); blocks are laid out in 8×8 tiles of 2×2 blocks, big-endian.
	"""
	h, w, _ = rgb.shape
	px = rgb.astype(np.int32)
	# blocks[by, bx, y, x, c]
	blocks = px.reshape(h // 4, 4, w // 4, 4, 3).transpose(0, 2, 1, 3, 4)
	flat = blocks.reshape(h // 4, w // 4, 16, 3)
	lum = flat @ np.array([299, 587, 114])
	hi = np.take_along_axis(flat, lum.argmax(-1)[..., None, None], 2)[..., 0, :]
	lo = np.take_along_axis(flat, lum.argmin(-1)[..., None, None], 2)[..., 0, :]

	def to565(c):
		return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)

	def from565(v):
		r = (v >> 11) & 31
		g = (v >> 5) & 63
		b = v & 31
		return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), -1)

	c0, c1 = to565(hi), to565(lo)
	swap = c0 < c1
	c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
	e0, e1 = from565(c0), from565(c1)
	pal = np.stack((e0, e1, (2 * e0 + e1) // 3, (e0 + 2 * e1) // 3), 2)
	dist = ((flat[:, :, :, None, :] - pal[:, :, None, :, :]) ** 2).sum(-1)
	idx = dist.argmin(-1)
	idx = np.where((c0 == c1)[..., None], 0, idx)
	rows = idx.reshape(h // 4, w // 4, 4, 4)
	bits = (rows[..., 0] << 6) | (rows[..., 1] << 4) | (rows[..., 2] << 2) | rows[..., 3]
	out = bytearray()
	for ty in range(0, h // 4, 2):
		for tx in range(0, w // 4, 2):
			for by, bx in ((ty, tx), (ty, tx + 1), (ty + 1, tx), (ty + 1, tx + 1)):
				out += struct.pack(">HH", int(c0[by, bx]), int(c1[by, bx]))
				out += bytes(int(b) for b in bits[by, bx])
	return bytes(out)


def gx_cmpr_encode_blocks(rgb: "np.ndarray", opaque: "np.ndarray") -> bytes:
	"""GX CMPR (DXT1) with principal-axis endpoints, 1-bit alpha when a block
	has transparent pixels. rgb is H×W×3 uint8, opaque H×W bool, both multiples
	of 8; blocks are laid out in 8×8 tiles of 2×2 blocks, big-endian."""
	h, w = opaque.shape
	bh, bw = h // 4, w // 4
	flat = rgb.astype(np.float32).reshape(bh, 4, bw, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh, bw, 16, 3)
	mask = opaque.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(bh, bw, 16)
	weight = mask[..., None].astype(np.float32)
	count = np.maximum(weight.sum(2, keepdims=True), 1.0)
	mean = (flat * weight).sum(2, keepdims=True) / count
	diff = (flat - mean) * weight
	cov = np.einsum("...ki,...kj->...ij", diff, diff)
	axis = np.ones(cov.shape[:2] + (3,), np.float32)
	for _ in range(8):	# power iteration
		axis = np.einsum("...ij,...j->...i", cov, axis)
		axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-6)
	t = np.einsum("...ki,...i->...k", flat - mean, axis)
	lo_t = np.where(mask, t, np.inf).min(2)
	hi_t = np.where(mask, t, -np.inf).max(2)
	lo_t = np.where(np.isfinite(lo_t), lo_t, 0.0)
	hi_t = np.where(np.isfinite(hi_t), hi_t, 0.0)
	lo = np.clip(mean[:, :, 0] + axis * lo_t[..., None], 0, 255).astype(np.int32)
	hi = np.clip(mean[:, :, 0] + axis * hi_t[..., None], 0, 255).astype(np.int32)

	def to565(c):
		return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)

	def from565(v):
		r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
		return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), -1)

	a, b = to565(hi), to565(lo)
	has_alpha = ~mask.all(2)
	# c0 > c1 selects four colours, c0 <= c1 three plus transparent.
	four = np.where(a > b, a, b), np.where(a > b, b, a)
	c0 = np.where(has_alpha, four[1], four[0])
	c1 = np.where(has_alpha, four[0], four[1])
	e0, e1 = from565(c0), from565(c1)
	pal4 = np.stack((e0, e1, (2 * e0 + e1) // 3, (e0 + 2 * e1) // 3), 2)
	pal3 = np.stack((e0, e1, (e0 + e1) // 2, np.full_like(e0, 100000)), 2)
	pal = np.where(has_alpha[..., None, None], pal3, pal4).astype(np.float32)
	dist = ((flat[:, :, :, None, :] - pal[:, :, None, :, :]) ** 2).sum(-1)
	idx = dist.argmin(-1)
	idx = np.where(mask, idx, 3)
	idx = np.where(((c0 == c1) & ~has_alpha)[..., None], 0, idx)
	rows = idx.reshape(bh, bw, 4, 4)
	bits = ((rows[..., 0] << 6) | (rows[..., 1] << 4) | (rows[..., 2] << 2) | rows[..., 3]).astype(np.uint8)
	out = bytearray()
	for ty in range(0, bh, 2):
		for tx in range(0, bw, 2):
			for by, bx in ((ty, tx), (ty, tx + 1), (ty + 1, tx), (ty + 1, tx + 1)):
				out += struct.pack(">HH", int(c0[by, bx]), int(c1[by, bx]))
				out += bits[by, bx].tobytes()
	return bytes(out)


def _blocks_of(data: bytes, bh: int, bw: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
	"""(c0, c1, 16 indices) per block, undoing the 8×8 tile order."""
	raw = np.frombuffer(data, ">u2", bh * bw * 4).reshape(bh // 2, bw // 2, 2, 2, 4)
	raw = raw.transpose(0, 2, 1, 3, 4).reshape(bh, bw, 4)
	c0, c1 = raw[..., 0].astype(np.int32), raw[..., 1].astype(np.int32)
	bits = raw[..., 2:].astype(np.uint32)
	word = (bits[..., 0] << 16) | bits[..., 1]
	idx = np.stack([(word >> (30 - 2 * k)) & 3 for k in range(16)], -1).astype(np.int32)
	return c0, c1, idx


def _tile_blocks(c0: "np.ndarray", c1: "np.ndarray", idx: "np.ndarray") -> bytes:
	bh, bw = c0.shape
	word = np.zeros((bh, bw), np.uint32)
	for k in range(16):
		word |= idx[..., k].astype(np.uint32) << (30 - 2 * k)
	rows = np.stack((c0, c1, word >> 16, word & 0xFFFF), -1).astype(">u2")
	return rows.reshape(bh // 2, 2, bw // 2, 2, 4).transpose(0, 2, 1, 3, 4).tobytes()


def _from565(v):
	r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
	return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), -1)


def _to565(c):
	c = np.clip(np.round(c), 0, 255).astype(np.int32)
	return ((c[..., 0] * 31 + 127) // 255 << 11) | ((c[..., 1] * 63 + 127) // 255 << 5) | ((c[..., 2] * 31 + 127) // 255)


def _palette(c0, c1):
	e0, e1 = _from565(c0), _from565(c1)
	four = c0 > c1
	p2 = np.where(four[..., None], (2 * e0 + e1) // 3, (e0 + e1) // 2)
	p3 = np.where(four[..., None], (e0 + 2 * e1) // 3, 0)
	return np.stack((e0, e1, p2, p3), 2).astype(np.float32), four


def cmpr_refine(rgb: "np.ndarray", opaque: "np.ndarray", data: bytes, iters: int = 2) -> bytes:
	"""Least-squares endpoint refinement of CMPR blocks, same size.

	Each block's indices fix how much of each endpoint every pixel gets
	(weights 1, 0, ⅔, ⅓ in four-colour blocks; 1, 0, ½ in three-colour
	ones), so the endpoints that minimise squared error solve a 2×2 normal
	equation per channel. Quantise to RGB565, re-pick indices, and keep a
	block only when its error went down. Mode (alpha or not) never changes,
	so cut-outs stay exactly as they were."""
	h, w = opaque.shape
	bh, bw = h // 4, w // 4
	flat = rgb.astype(np.float32).reshape(bh, 4, bw, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh, bw, 16, 3)
	mask = opaque.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(bh, bw, 16)
	c0, c1, idx = _blocks_of(data, bh, bw)

	def err_of(c0, c1, idx):
		pal, four = _palette(c0, c1)
		got = np.take_along_axis(pal, idx[..., None].repeat(3, -1), 2)
		e = ((got - flat) ** 2).sum(-1)
		return np.where(mask, e, 0).sum(-1)

	best = err_of(c0, c1, idx)
	for _ in range(iters):
		four = c0 > c1
		wt = np.where(four[..., None], np.array([1, 0, 2 / 3, 1 / 3])[idx], np.array([1, 0, 0.5, 0])[idx])
		use = mask & (four[..., None] | (idx != 3))
		a = np.where(use, wt, 0.0)
		b = np.where(use, 1 - wt, 0.0)
		aa, bb, ab = (a * a).sum(-1), (b * b).sum(-1), (a * b).sum(-1)
		ax = (a[..., None] * flat).sum(2)
		bx = (b[..., None] * flat).sum(2)
		det = aa * bb - ab * ab
		ok = np.abs(det) > 1e-6
		sd = np.where(ok, det, 1.0)[..., None]
		e0 = (bb[..., None] * ax - ab[..., None] * bx) / sd
		e1 = (aa[..., None] * bx - ab[..., None] * ax) / sd
		n0, n1 = _to565(e0), _to565(e1)
		# keep the block's mode: c0 > c1 four colours, else three + transparent
		swap = np.where(four, n0 < n1, n0 > n1)
		n0, n1 = np.where(swap, n1, n0), np.where(swap, n0, n1)
		degenerate = four & (n0 == n1)	# would flip to three-colour mode
		pal, _ = _palette(n0, n1)
		d = ((flat[:, :, :, None, :] - pal[:, :, None, :, :]) ** 2).sum(-1)
		# three-colour blocks: index 3 is transparent, never a colour choice
		d[..., 3] = np.where(four[..., None], d[..., 3], np.inf)
		nidx = np.where(mask, d.argmin(-1), 3)
		nidx = np.where(four[..., None], np.where(mask, nidx, idx), nidx)
		nerr = err_of(n0, n1, nidx)
		better = ok & ~degenerate & (nerr < best)
		c0, c1 = np.where(better, n0, c0), np.where(better, n1, c1)
		idx = np.where(better[..., None], nidx, idx)
		best = np.where(better, nerr, best)
	return _tile_blocks(c0, c1, idx)


def cmpr_size(width: int, height: int) -> int:
	"""Bytes for one CMPR level (4 bits per texel, padded to 8×8 tiles)."""
	return ((width + 7) // 8) * ((height + 7) // 8) * 32


def cmpr_mip_chain_size(width: int, height: int, levels: int) -> int:
	total = 0
	for level in range(levels):
		total += cmpr_size(max(8, width >> level), max(8, height >> level))
	return total


def cmpr_decode(data: bytes, width: int, height: int) -> "np.ndarray":
	"""Decode one GX CMPR level to an H×W×4 uint8 RGBA array.

	Mirrors Flipper's sampling of S3TC: c0 > c1 gives four opaque colours,
	otherwise three colours plus transparent black.
	"""
	tw, th = (width + 7) // 8, (height + 7) // 8
	need = tw * th * 32
	if len(data) < need:
		raise ValueError(f"CMPR data too short: {len(data)} < {need} for {width}x{height}")
	raw = np.frombuffer(data[:need], np.uint8).reshape(th, tw, 2, 2, 8)
	c0 = (raw[..., 0].astype(np.uint16) << 8) | raw[..., 1]
	c1 = (raw[..., 2].astype(np.uint16) << 8) | raw[..., 3]

	def from565(v):
		v = v.astype(np.int32)
		r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
		return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), -1)

	e0, e1 = from565(c0), from565(c1)
	four = c0 > c1
	p2 = np.where(four[..., None], (2 * e0 + e1) // 3, (e0 + e1) // 2)
	p3 = np.where(four[..., None], (e0 + 2 * e1) // 3, 0)
	rgb = np.stack((e0, e1, p2, p3), -2)	# th,tw,2,2,4,3
	alpha = np.full(four.shape + (4,), 255, np.int32)
	alpha[..., 3] = np.where(four, 255, 0)
	pal = np.concatenate((rgb, alpha[..., None]), -1)	# th,tw,2,2,4,4
	bits = raw[..., 4:8].astype(np.int32)	# rows of 4 texels, MSB first
	shifts = np.array([6, 4, 2, 0])
	idx = (bits[..., :, None] >> shifts) & 3	# th,tw,2,2,4(row),4(col)
	blocks = th * tw * 4
	pal_flat = pal.reshape(blocks, 4, 4)
	idx_flat = idx.reshape(blocks, 16)
	texels = pal_flat[np.arange(blocks)[:, None], idx_flat].reshape(th, tw, 2, 2, 4, 4, 4)	# th,tw,by,bx,row,col,rgba
	img = texels.transpose(0, 2, 4, 1, 3, 5, 6).reshape(th * 8, tw * 8, 4)
	return img[:height, :width].astype(np.uint8)


def next_pow2(value: int, minimum: int = 8) -> int:
	return 1 << max(minimum.bit_length() - 1, (value - 1).bit_length())
