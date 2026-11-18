"""ETC2 / EAC texture blocks (the mobile formats Source 2 uses for some textures), decoded to RGBA8.

ETC2 RGB covers ETC1's individual and differential modes plus T, H and
planar; EAC carries alpha (ETC2_EAC) or one or two 11-bit channels (R11,
RG11), reduced to 8 bits here. Blocks are 4x4 texels, big-endian, with
pixel indices stored column by column. Follows the Khronos data format
specification.
"""

from __future__ import annotations

import numpy as np

_MODIFIERS = ((2, 8), (5, 17), (9, 29), (13, 42), (18, 60), (24, 80), (33, 106), (47, 183))
_DISTANCES = (3, 6, 11, 16, 23, 32, 41, 64)
_EAC = np.array([
	[-3, -6, -9, -15, 2, 5, 8, 14], [-3, -7, -10, -13, 2, 6, 9, 12], [-2, -5, -8, -13, 1, 4, 7, 12],
	[-2, -4, -6, -13, 1, 3, 5, 12], [-3, -6, -8, -12, 2, 5, 7, 11], [-3, -7, -9, -11, 2, 6, 8, 10],
	[-4, -7, -8, -11, 3, 6, 7, 10], [-3, -5, -8, -11, 2, 4, 7, 10], [-2, -6, -8, -10, 1, 5, 7, 9],
	[-2, -5, -8, -10, 1, 4, 7, 9], [-2, -4, -8, -10, 1, 3, 7, 9], [-2, -5, -7, -10, 1, 4, 6, 9],
	[-3, -4, -7, -10, 2, 3, 6, 9], [-1, -2, -3, -10, 0, 1, 2, 9], [-4, -6, -8, -9, 3, 5, 7, 8],
	[-3, -5, -7, -9, 2, 4, 6, 8]], np.int32)


def _x4(v: int) -> int:
	return v * 17


def _x5(v: int) -> int:
	return (v << 3) | (v >> 2)


def _sign3(v: int) -> int:
	return v - 8 if v & 4 else v


def _clamp(v):
	return np.clip(v, 0, 255)


def etc2_rgb_block(b: bytes) -> "np.ndarray":
	"""One 8-byte ETC2 RGB block → (4, 4, 3) uint8, indexed [y, x]."""
	hi = int.from_bytes(b[4:6], "big")
	lo = int.from_bytes(b[6:8], "big")
	idx = np.zeros((4, 4), np.int32)	# [y, x]
	for x in range(4):
		for y in range(4):
			k = x * 4 + y
			idx[y, x] = (((hi >> k) & 1) << 1) | ((lo >> k) & 1)
	b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
	diff = b3 & 2
	flip = b3 & 1
	if not diff:
		c1 = (_x4(b0 >> 4), _x4(b1 >> 4), _x4(b2 >> 4))
		c2 = (_x4(b0 & 15), _x4(b1 & 15), _x4(b2 & 15))
		return _etc1(idx, c1, c2, b3 >> 5, (b3 >> 2) & 7, flip)
	r, g, bl = b0 >> 3, b1 >> 3, b2 >> 3
	r2, g2, bl2 = r + _sign3(b0 & 7), g + _sign3(b1 & 7), bl + _sign3(b2 & 7)
	if not 0 <= r2 <= 31:	# T mode
		c1 = np.array([_x4((((b0 >> 3) & 3) << 2) | (b0 & 3)), _x4(b1 >> 4), _x4(b1 & 15)])
		c2 = np.array([_x4(b2 >> 4), _x4(b2 & 15), _x4(b3 >> 4)])
		d = _DISTANCES[(((b3 >> 2) & 3) << 1) | (b3 & 1)]
		paint = np.array([c1, _clamp(c2 + d), c2, _clamp(c2 - d)])
		return paint[idx].astype(np.uint8)
	if not 0 <= g2 <= 31:	# H mode
		r1 = (b0 >> 3) & 15
		g1 = ((b0 & 7) << 1) | ((b1 >> 4) & 1)
		bb1 = (b1 & 8) | ((b1 & 3) << 1) | (b2 >> 7)
		rr2 = (b2 >> 3) & 15
		gg2 = ((b2 & 7) << 1) | (b3 >> 7)
		bb2 = (b3 >> 3) & 15
		order = 1 if (r1 << 8 | g1 << 4 | bb1) >= (rr2 << 8 | gg2 << 4 | bb2) else 0
		d = _DISTANCES[(b3 & 4) | ((b3 & 1) << 1) | order]
		c1 = np.array([_x4(r1), _x4(g1), _x4(bb1)])
		c2 = np.array([_x4(rr2), _x4(gg2), _x4(bb2)])
		paint = np.array([_clamp(c1 + d), _clamp(c1 - d), _clamp(c2 + d), _clamp(c2 - d)])
		return paint[idx].astype(np.uint8)
	if not 0 <= bl2 <= 31:	# planar
		b4, b5, b6, b7 = b[4], b[5], b[6], b[7]
		ro = (b0 >> 1) & 63
		go = ((b0 & 1) << 6) | ((b1 >> 1) & 63)
		bo = ((b1 & 1) << 5) | (((b2 >> 3) & 3) << 3) | ((b2 & 3) << 1) | (b3 >> 7)
		rh = (((b3 >> 2) & 31) << 1) | (b3 & 1)
		gh = b4 >> 1
		bh = ((b4 & 1) << 5) | (b5 >> 3)
		rv = ((b5 & 7) << 3) | (b6 >> 5)
		gv = ((b6 & 31) << 2) | (b7 >> 6)
		bv = b7 & 63

		def x6(v):
			return (v << 2) | (v >> 4)

		def x7(v):
			return (v << 1) | (v >> 6)

		o = np.array([x6(ro), x7(go), x6(bo)])
		h = np.array([x6(rh), x7(gh), x6(bh)])
		v = np.array([x6(rv), x7(gv), x6(bv)])
		yy, xx = np.mgrid[0:4, 0:4]
		col = (xx[..., None] * (h - o) + yy[..., None] * (v - o) + 4 * o + 2) >> 2
		return _clamp(col).astype(np.uint8)
	c1 = (_x5(r), _x5(g), _x5(bl))
	c2 = (_x5(r2), _x5(g2), _x5(bl2))
	return _etc1(idx, c1, c2, b3 >> 5, (b3 >> 2) & 7, flip)


def _etc1(idx, c1, c2, t1: int, t2: int, flip: int) -> "np.ndarray":
	yy, xx = np.mgrid[0:4, 0:4]
	second = (yy >= 2) if flip else (xx >= 2)
	out = np.zeros((4, 4, 3), np.int32)
	for sub, (base, t) in enumerate(((c1, t1), (c2, t2))):
		a, bb = _MODIFIERS[t]
		mod = np.array([a, bb, -a, -bb])[idx]
		sel = second == bool(sub)
		out[sel] = _clamp(np.asarray(base)[None] + mod[sel][:, None])
	return out.astype(np.uint8)


def eac_block(b: bytes, eleven: bool = False) -> "np.ndarray":
	"""One 8-byte EAC block → (4, 4) uint8 [y, x] (alpha, or an 11-bit channel shifted to 8 bits)."""
	base, mult, table = b[0], b[1] >> 4, b[1] & 15
	bits = int.from_bytes(b[2:8], "big")
	out = np.zeros((4, 4), np.int32)
	for x in range(4):
		for y in range(4):
			k = x * 4 + y
			i = (bits >> (45 - 3 * k)) & 7
			m = int(_EAC[table, i])
			if eleven:
				v = base * 8 + 4 + (m * mult * 8 if mult else m)
				out[y, x] = (min(2047, max(0, v)) * 255 + 1023) // 2047
			else:
				out[y, x] = min(255, max(0, base + m * mult))
	return out.astype(np.uint8)


def decode(raw: bytes, w: int, h: int, kind: str) -> "np.ndarray":
	"""kind: 'etc2' (RGB), 'etc2_eac' (RGBA), 'r11' or 'rg11' → (h, w, 4) uint8."""
	size = {"etc2": 8, "etc2_eac": 16, "r11": 8, "rg11": 16}[kind]
	bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
	out = np.zeros((bh * 4, bw * 4, 4), np.uint8)
	out[..., 3] = 255
	p = 0
	for by in range(bh):
		for bx in range(bw):
			blk = raw[p:p + size]
			p += size
			if len(blk) < size:
				break
			tile = out[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4]
			if kind == "etc2":
				tile[..., :3] = etc2_rgb_block(blk)
			elif kind == "etc2_eac":
				tile[..., 3] = eac_block(blk[:8])
				tile[..., :3] = etc2_rgb_block(blk[8:])
			elif kind == "r11":
				tile[..., 0] = tile[..., 1] = tile[..., 2] = eac_block(blk, True)
			else:
				tile[..., 0] = eac_block(blk[:8], True)
				tile[..., 1] = eac_block(blk[8:], True)
				tile[..., 2] = 0
	return out[:h, :w]
