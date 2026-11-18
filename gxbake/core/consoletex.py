"""Console texture files as input: Nintendo GameCube/Wii (.tpl, .bti, .tex0),
DS (.ncgr + .nclr), 3DS (.bclim, .bflim), Sony PlayStation (.tim) and
PlayStation 2 (.tm2), Microsoft Xbox (.xpr) and Sega Dreamcast (.pvr).

Each reader returns the top level of the first image as H×W×4 uint8 RGBA and
appends a warning when it had to guess (no palette found, more images
skipped). Layouts follow the community documentation these formats are
known by: YAGCD and Dolphin's TextureDecoder for GX, the Custom Mario Kart
wiki for BRRES TEX0/PLT0, GBATEK and Tinke for the DS Nitro files, 3dbrew
for 3DS textures, the PSX spec (Martin Korth) for TIM, the TIM2 notes used
by Rainbow and Noesis, the Xbox XPR notes from the XBMC skin tools, and
PuyoTools for Dreamcast PVR.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import gxtex
from . import texcodec as tc

# -- GameCube / Wii (GX) -------------------------------------------------------------------

GX_TF_C14X2 = 0xA


def _warn(warnings, msg: str) -> None:
	if warnings is not None:
		warnings.append(msg)


def gx_decode(fmt: int, w: int, h: int, data: bytes, tlut: bytes = b"", tlut_fmt: int = gxtex.GX_TL_RGB565) -> "np.ndarray":
	"""Any GX texture format id (including CI14x2) → RGBA."""
	if fmt == GX_TF_C14X2:
		pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
		idx = np.frombuffer(data, ">u2", pw * ph) & 0x3FFF
		pal = gxtex.tlut_rgba(np.frombuffer(tlut, ">u2"), tlut_fmt) if tlut else tc.GREY_RGBA
		if not len(pal):
			pal = tc.GREY_RGBA
		flat = pal[np.minimum(idx, len(pal) - 1)]
		return tc.untile(flat, pw, ph, 4, 4)[:h, :w]
	name = gxtex.FORMAT_NAMES.get(fmt)
	if name is None:
		raise ValueError(f"unknown GX texture format 0x{fmt:X}")
	if name in ("CI4", "CI8") and not tlut:
		n = 16 if name == "CI4" else 256
		tlut = gxtex.tlut_words(tc.GREY_RGBA[:: 256 // n][:n], gxtex.GX_TL_RGB565).astype(">u2").tobytes()
		tlut_fmt = gxtex.GX_TL_RGB565
	need = gxtex.tex_size(w, h, name)
	if len(data) < need:
		raise ValueError(f"truncated {name} texture ({len(data)} of {need} bytes)")
	return gxtex.decode(name, w, h, data, tlut, tlut_fmt)


_TPL_MAGIC = 0x0020AF30


def read_tpl(data: bytes, warnings: list[str] | None = None) -> "np.ndarray":
	"""GameCube/Wii texture palette library (.tpl): the first image."""
	magic, count, table = struct.unpack_from(">III", data, 0)
	if magic != _TPL_MAGIC or not count:
		raise ValueError("not a TPL file")
	if count > 1:
		_warn(warnings, f"TPL holds {count} images; only the first is converted")
	img_off, pal_off = struct.unpack_from(">II", data, table)
	h, w, fmt, data_off = struct.unpack_from(">HHII", data, img_off)
	tlut, tlut_fmt = b"", gxtex.GX_TL_RGB565
	if pal_off:
		n, _unpacked, _pad, tlut_fmt, pal_data = struct.unpack_from(">HBBII", data, pal_off)
		tlut = data[pal_data: pal_data + n * 2]
	elif fmt in (8, 9, GX_TF_C14X2):
		_warn(warnings, "TPL colour-indexed image has no palette; greyscale used")
	return gx_decode(fmt, w, h, data[data_off:], tlut, tlut_fmt)


def read_bti(data: bytes, warnings: list[str] | None = None, base: int = 0) -> "np.ndarray":
	"""J3D binary texture image (.bti, also inside BMD/BDL TEX1): 32-byte header."""
	fmt, _alpha, w, h = struct.unpack_from(">BBHH", data, base)
	tlut_fmt, n_pal, pal_off = struct.unpack_from(">BHI", data, base + 9)
	img_off = struct.unpack_from(">I", data, base + 0x1C)[0]
	if not w or not h or fmt not in gxtex.FORMAT_NAMES and fmt != GX_TF_C14X2:
		raise ValueError("not a BTI texture")
	tlut = data[base + pal_off: base + pal_off + n_pal * 2] if n_pal else b""
	if fmt in (8, 9, GX_TF_C14X2) and not tlut:
		_warn(warnings, "BTI colour-indexed texture has no palette; greyscale used")
	return gx_decode(fmt, w, h, data[base + img_off:], tlut, tlut_fmt)


def read_plt0(data: bytes) -> tuple[bytes, int]:
	"""BRRES palette (.plt0) → (TLUT words, TLUT format)."""
	if data[:4] != b"PLT0":
		raise ValueError("not a PLT0 palette")
	off = struct.unpack_from(">I", data, 0x10)[0]
	fmt, n = struct.unpack_from(">IH", data, 0x18)
	return data[off: off + n * 2], fmt


def read_tex0(data: bytes, warnings: list[str] | None = None, palette: tuple[bytes, int] | None = None) -> "np.ndarray":
	"""BRRES texture (.tex0), as exported by BrawlBox and friends."""
	if data[:4] != b"TEX0":
		raise ValueError("not a TEX0 texture")
	off = struct.unpack_from(">I", data, 0x10)[0]
	ci = struct.unpack_from(">I", data, 0x18)[0]
	w, h, fmt = struct.unpack_from(">HHI", data, 0x1C)
	tlut, tlut_fmt = palette if palette else (b"", gxtex.GX_TL_RGB565)
	if (ci or fmt in (8, 9, GX_TF_C14X2)) and not tlut:
		_warn(warnings, "TEX0 is colour-indexed and no matching .plt0 was found; greyscale used")
	return gx_decode(fmt, w, h, data[off:], tlut, tlut_fmt)


def load_tex0(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	path = Path(path)
	pal = None
	for cand in path.parent.iterdir() if path.parent.is_dir() else ():
		if cand.stem.lower() == path.stem.lower() and cand.suffix.lower() == ".plt0":
			pal = read_plt0(cand.read_bytes())
			break
	return read_tex0(path.read_bytes(), warnings, pal)


# -- Nintendo DS -----------------------------------------------------------------------------


def _nitro_section(data: bytes, magic: bytes) -> int:
	hsize, nsec = struct.unpack_from("<HH", data, 12)
	pos = hsize
	for _ in range(nsec):
		if data[pos:pos + 4] == magic:
			return pos
		pos += struct.unpack_from("<I", data, pos + 4)[0]
	raise ValueError(f"no {magic.decode()} section")


def read_nclr(data: bytes) -> "np.ndarray":
	"""DS palette (.nclr, RLCN) → N×4 RGBA (BGR555)."""
	if data[:4] != b"RLCN":
		raise ValueError("not an NCLR palette")
	s = _nitro_section(data, b"TTLP")
	size, off = struct.unpack_from("<II", data, s + 0x10)
	words = np.frombuffer(data, "<u2", size // 2, s + 8 + off)
	return tc.unpack_packed(words.tobytes(), len(words), 1, "X1B5G5R5", 16)[0]


def read_ncgr(data: bytes, palette: "np.ndarray | None", warnings: list[str] | None = None) -> "np.ndarray":
	"""DS character graphics (.ncgr, RGCN): 8×8 tiles of 4 or 8 bpp, laid out as stored."""
	if data[:4] != b"RGCN":
		raise ValueError("not an NCGR file")
	s = _nitro_section(data, b"RAHC")
	th, tw, depth = struct.unpack_from("<HHI", data, s + 8)
	linear = struct.unpack_from("<I", data, s + 0x14)[0] & 0xFF
	size, off = struct.unpack_from("<II", data, s + 0x18)
	raw = np.frombuffer(data, np.uint8, size, s + 8 + off)
	bpp = 4 if depth == 3 else 8
	if bpp == 4:
		idx = np.empty(raw.size * 2, np.uint8)
		idx[0::2], idx[1::2] = raw & 15, raw >> 4	# low nibble is the left pixel
	else:
		idx = raw
	ntiles = idx.size // 64
	if th == 0xFFFF or tw == 0xFFFF or th * tw > ntiles or not th or not tw:
		tw = min(32, ntiles) or 1
		th = -(-ntiles // tw)
	w, h = tw * 8, th * 8
	idx = np.concatenate([idx, np.zeros(max(0, w * h - idx.size), np.uint8)])[: w * h]
	img = idx.reshape(h, w) if linear else tc.untile(idx, w, h, 8, 8)
	if palette is None:
		_warn(warnings, "no .nclr palette beside the .ncgr; greyscale used")
		palette = tc.GREY_RGBA[:: 16] if bpp == 4 else tc.GREY_RGBA
	elif bpp == 4 and len(palette) > 16:
		_warn(warnings, "4 bpp graphics use palette bank 0 (the screen map that picks banks is not read)")
	pal = np.zeros((256, 4), np.uint8)
	pal[: min(256, len(palette))] = palette[:256]
	pal[:, 3] = 255
	pal[0, 3] = 0	# colour 0 is transparent on the DS
	return pal[img]


def load_ncgr(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	path = Path(path)
	pal = None
	cands = sorted(p for p in path.parent.iterdir() if p.suffix.lower() == ".nclr")
	same = [p for p in cands if p.stem.lower() == path.stem.lower()]
	if same or len(cands) == 1:
		pal = read_nclr((same or cands)[0].read_bytes())
	elif cands:
		_warn(warnings, f"several .nclr palettes and none named {path.stem}.nclr; using {cands[0].name}")
		pal = read_nclr(cands[0].read_bytes())
	return read_ncgr(path.read_bytes(), pal, warnings)


# -- Nintendo 3DS ----------------------------------------------------------------------------

# format → (bits per pixel, decoder)
_CTR = {
	0: (8, ("bytes", "L")), 1: (8, ("bytes", "A")), 2: (8, ("packed", "L4A4", 8)), 3: (16, ("bytes", "AL")),
	4: (16, ("bytes", "GR")), 5: (16, ("packed", "R5G6B5", 16)), 6: (24, ("bytes", "BGR")),
	7: (16, ("packed", "R5G5B5A1", 16)), 8: (16, ("packed", "R4G4B4A4", 16)), 9: (32, ("bytes", "ABGR")),
	10: (4, "etc1"), 11: (8, "etc1a4"), 12: (4, ("nibble", "L")), 13: (4, ("nibble", "A")),
}


def _ctr_etc(data: bytes, w: int, h: int, alpha: bool) -> "np.ndarray":
	"""3DS ETC1(A4): 8×8 tiles of four 4×4 blocks in Z order, each block's 64 bits little-endian."""
	step = 16 if alpha else 8
	nb = (w // 4) * (h // 4)
	raw = np.frombuffer(data, np.uint8, nb * step).reshape(nb, step)
	colour = raw[:, -8:][:, ::-1]	# to the standard big-endian ETC1 block
	# block order in the file: tiles row-major, then TL TR BL BR inside a tile
	tiles_x, tiles_y = w // 8, h // 8
	order = np.empty(nb, np.intp)
	i = 0
	for ty in range(tiles_y):
		for tx in range(tiles_x):
			for by, bx in ((0, 0), (0, 1), (1, 0), (1, 1)):
				order[(ty * 2 + by) * (w // 4) + tx * 2 + bx] = i
				i += 1
	rgba = tc.decode_blocks("etc1", colour[order].tobytes(), w, h)
	if alpha:
		a64 = raw[:, :8].copy().view("<u8")[:, 0]	# 4 bits per texel, x-major inside the block
		a = np.empty((nb, 4, 4), np.uint8)
		for x in range(4):
			for y in range(4):
				a[:, y, x] = ((a64 >> np.uint64(4 * (x * 4 + y))) & np.uint64(15)).astype(np.uint8) * 17
		a = a[order]
		rgba[..., 3] = a.reshape(h // 4, w // 4, 4, 4).swapaxes(1, 2).reshape(h, w)
	return rgba


def ctr_decode(fmt: int, data: bytes, w: int, h: int) -> "np.ndarray":
	"""3DS (PICA200) texture, w and h multiples of 8 → RGBA, flipped to top-down."""
	bpp, how = _CTR[fmt]
	if how in ("etc1", "etc1a4"):
		return _ctr_etc(data, w, h, how == "etc1a4")[::-1].copy()
	n = w * h
	if how[0] == "nibble":
		b = np.frombuffer(data, np.uint8, n // 2)
		v = np.empty(n, np.uint8)
		v[0::2], v[1::2] = b & 15, b >> 4
		v = v * 17
		flat = np.zeros((n, 4), np.uint8)
		flat[:, 3] = 255
		if how[1] == "L":
			flat[:, 0] = flat[:, 1] = flat[:, 2] = v
		else:
			flat[:, :3] = 255
			flat[:, 3] = v
	elif how[0] == "bytes":
		flat = tc.unpack_bytes(data, n, 1, how[1])[0]
	else:
		flat = tc.unpack_packed(data, n, 1, how[1], how[2])[0]
	# texels: 8×8 tiles row-major; inside a tile, Morton order (x in the low bit)
	tiles = flat.reshape(-1, 64, 4)
	inner = tc.morton_index(8, 8).reshape(-1).astype(np.intp)
	tiles = tiles[:, inner]	# (tile, y*8+x)
	img = tc.untile(tiles.reshape(-1, 4), w, h, 8, 8)
	return img[::-1].copy()


def read_bclim(data: bytes, warnings: list[str] | None = None) -> "np.ndarray":
	"""3DS layout image (.bclim / 3DS .bflim): data first, CLIM/FLIM header and imag block at the end."""
	foot = len(data) - 0x28
	magic = data[foot:foot + 4]
	if magic not in (b"CLIM", b"FLIM"):
		raise ValueError("not a BCLIM/BFLIM image")
	if data[foot + 4:foot + 6] == b"\xfe\xff":
		raise ValueError("Wii U .bflim (GX2 tiled) is not supported, only 3DS")
	img = foot + 0x14
	if data[img:img + 4] != b"imag":
		raise ValueError("BCLIM: no imag block")
	w, h = struct.unpack_from("<HH", data, img + 8)
	if magic == b"CLIM":
		fmt = struct.unpack_from("<I", data, img + 12)[0]
		swizzle = 0
	else:
		fmt, swizzle = data[img + 14], data[img + 15]
	if fmt not in _CTR:
		raise ValueError(f"BCLIM: unknown format {fmt}")
	bpp = _CTR[fmt][0]
	# the texture is stored at power-of-two sides (at least 8); find the pair that matches the size
	pw, ph = tc_pow2(w), tc_pow2(h)
	if pw * ph * bpp // 8 > foot:
		pw, ph = -(-w // 8) * 8, -(-h // 8) * 8
	out = ctr_decode(fmt, data[:foot], pw, ph)
	if swizzle in (4, 8):	# rotated 90° by the tool that made it
		out = np.rot90(out, 1 if swizzle == 4 else -1).copy()
		w, h = h, w
	return out[:h, :w]


def tc_pow2(v: int) -> int:
	return 1 << max(3, (v - 1).bit_length())


# -- PlayStation ------------------------------------------------------------------------------


def _psx_colours(words: "np.ndarray") -> "np.ndarray":
	"""PS1 15-bit colours (bit 15 STP): 0x0000 is transparent, everything else opaque."""
	rgba = tc.unpack_packed(words.astype("<u2").tobytes(), len(words), 1, "X1B5G5R5", 16)[0]
	rgba[:, 3] = np.where(words == 0, 0, 255)
	return rgba


def read_tim(data: bytes, warnings: list[str] | None = None) -> "np.ndarray":
	"""PlayStation TIM: 4/8 bpp with CLUT, 15-bit direct or 24-bit."""
	magic, flags = struct.unpack_from("<II", data, 0)
	if magic != 0x10:
		raise ValueError("not a PlayStation TIM")
	mode = flags & 7
	pos = 8
	pal = None
	if flags & 8:
		blen, _x, _y, cw, ch = struct.unpack_from("<IHHHH", data, pos)
		if ch > 1:
			_warn(warnings, f"TIM has {ch} palettes; the first is used")
		pal = _psx_colours(np.frombuffer(data, "<u2", cw, pos + 12))
		pos += blen
	blen, _x, _y, w16, h = struct.unpack_from("<IHHHH", data, pos)
	px = data[pos + 12: pos + blen]
	if mode in (0, 1):
		if pal is None:
			_warn(warnings, "TIM has no CLUT; greyscale used")
			pal = tc.GREY_RGBA[:: 16] if mode == 0 else tc.GREY_RGBA
		if mode == 0:
			w = w16 * 4
			b = np.frombuffer(px, np.uint8, w * h // 2)
			idx = np.empty(w * h, np.uint8)
			idx[0::2], idx[1::2] = b & 15, b >> 4
		else:
			w = w16 * 2
			idx = np.frombuffer(px, np.uint8, w * h)
		return pal[np.minimum(idx, len(pal) - 1)].reshape(h, w, 4)
	if mode == 2:
		return _psx_colours(np.frombuffer(px, "<u2", w16 * h)).reshape(h, w16, 4)
	if mode == 3:
		w = w16 * 2 // 3
		rows = [px[y * w16 * 2: y * w16 * 2 + w * 3] for y in range(h)]
		return tc.unpack_bytes(b"".join(rows), w, h, "RGB")
	raise ValueError(f"TIM: unsupported mode {mode}")


def _ps2_alpha(rgba: "np.ndarray") -> "np.ndarray":
	"""GS alpha is 0..128 (0x80 = opaque)."""
	rgba[..., 3] = np.minimum(rgba[..., 3].astype(np.uint16) * 2, 255).astype(np.uint8)
	return rgba


def _tm2_colours(raw: bytes, n: int, kind: int) -> "np.ndarray":
	if kind == 1:
		out = tc.unpack_packed(raw, n, 1, "A1B5G5R5", 16)[0]
		return out
	if kind == 2:
		return tc.unpack_bytes(raw, n, 1, "RGB")[0]
	if kind == 3:
		return _ps2_alpha(tc.unpack_bytes(raw, n, 1, "RGBA")[0])
	raise ValueError(f"TIM2: unknown colour type {kind}")


def read_tm2(data: bytes, warnings: list[str] | None = None) -> "np.ndarray":
	"""PlayStation 2 TIM2 (.tm2): first picture."""
	if data[:4] != b"TIM2":
		raise ValueError("not a TIM2 file")
	_ver, align, count = struct.unpack_from("<BBH", data, 4)
	if count > 1:
		_warn(warnings, f"TIM2 has {count} pictures; only the first is converted")
	pos = 128 if align else 16
	total, clut_size, img_size, hsize, ncol, _pf, mips, clut_type, img_type, w, h = struct.unpack_from(
		"<IIIHHBBBBHH", data, pos)
	img = data[pos + hsize: pos + hsize + img_size]
	if img_type in (1, 2, 3):
		return _tm2_colours(img, w * h, img_type).reshape(h, w, 4)
	if img_type not in (4, 5):
		raise ValueError(f"TIM2: unsupported image type {img_type}")
	clut = data[pos + hsize + img_size: pos + hsize + img_size + clut_size]
	pal = _tm2_colours(clut, ncol, clut_type & 0x3F)
	if img_type == 5 and ncol == 256 and not clut_type & 0x80:	# CSM1: swap the middle 8-entry blocks of each 32
		i = np.arange(256)
		pal = pal[(i & ~0x18) | ((i & 0x08) << 1) | ((i & 0x10) >> 1)]
	if img_type == 4:
		b = np.frombuffer(img, np.uint8, w * h // 2)
		idx = np.empty(w * h, np.uint8)
		idx[0::2], idx[1::2] = b & 15, b >> 4
	else:
		idx = np.frombuffer(img, np.uint8, w * h)
	return pal[np.minimum(idx, len(pal) - 1)].reshape(h, w, 4)


# -- Xbox ------------------------------------------------------------------------------------

# D3DFORMAT (Xbox) → (bits, decoder, swizzled)
_XBOX = {
	0x00: (8, ("bytes", "L"), True), 0x01: (8, ("bytes", "A"), True), 0x02: (16, ("packed", "A1R5G5B5", 16), True),
	0x03: (16, ("packed", "X1R5G5B5", 16), True), 0x04: (16, ("packed", "A4R4G4B4", 16), True),
	0x05: (16, ("packed", "R5G6B5", 16), True), 0x06: (32, ("bytes", "BGRA"), True), 0x07: (32, ("bytes", "BGRX"), True),
	0x0B: (8, ("pal",), True), 0x0C: (4, "bc1", False), 0x0E: (8, "bc2", False), 0x0F: (8, "bc3", False),
	0x10: (16, ("packed", "A1R5G5B5", 16), False), 0x11: (16, ("packed", "R5G6B5", 16), False),
	0x12: (32, ("bytes", "BGRA"), False), 0x13: (8, ("bytes", "L"), False), 0x19: (8, ("bytes", "A"), True),
	0x1A: (16, ("bytes", "AL"), True), 0x1D: (16, ("packed", "A4R4G4B4", 16), False),
	0x1E: (32, ("bytes", "BGRX"), False), 0x1F: (8, ("bytes", "A"), False), 0x20: (16, ("bytes", "AL"), False),
	0x3A: (32, ("bytes", "RGBA"), True), 0x3F: (32, ("bytes", "RGBA"), False),
}


def read_xpr(data: bytes, warnings: list[str] | None = None) -> "np.ndarray":
	"""Xbox packed resource (XPR0): the first texture."""
	if data[:4] == b"XPR2":
		raise ValueError("Xbox 360 XPR2 (tiled, big-endian) is not supported, only original Xbox XPR0")
	if data[:4] != b"XPR0":
		raise ValueError("not an XPR0 file")
	_total, hsize = struct.unpack_from("<II", data, 4)
	pos = 12
	tex, pal = None, None
	textures = 0
	while pos + 4 <= hsize:
		common = struct.unpack_from("<I", data, pos)[0]
		if common == 0xFFFFFFFF:
			break
		rtype = (common >> 16) & 7
		if rtype == 4:	# texture: Common, Data, Lock, Format, Size
			textures += 1
			if tex is None:
				tex = struct.unpack_from("<IIIII", data, pos)
			pos += 20
		elif rtype == 3:	# palette: Common, Data, Lock
			if pal is None:
				pal = struct.unpack_from("<III", data, pos)
			pos += 12
		else:
			pos += 12 if rtype in (1, 2, 6) else 20
	if tex is None:
		raise ValueError("XPR has no textures")
	if textures > 1:
		_warn(warnings, f"XPR holds {textures} textures; only the first is converted")
	_c, doff, _l, fmt, size = tex
	xfmt = (fmt >> 8) & 0xFF
	if xfmt not in _XBOX:
		raise ValueError(f"XPR: unsupported Xbox texture format 0x{xfmt:02X}")
	bits, how, swizzled = _XBOX[xfmt]
	if size:
		w, h = (size & 0xFFF) + 1, ((size >> 12) & 0xFFF) + 1
		pitch = (((size >> 24) & 0xFF) + 1) * 64
	else:
		w, h = 1 << ((fmt >> 20) & 15), 1 << ((fmt >> 24) & 15)
		pitch = w * bits // 8
	px = data[hsize + doff:]
	if isinstance(how, str):
		return tc.decode_blocks(how, px, w, h)
	n = w * h
	if not swizzled and pitch != w * bits // 8:
		row = w * bits // 8
		px = b"".join(px[y * pitch: y * pitch + row] for y in range(h))
	if how[0] == "pal":
		idx = np.frombuffer(px, np.uint8, n)
		if pal is None:
			_warn(warnings, "XPR P8 texture without a palette; greyscale used")
			colours = tc.GREY_RGBA
		else:
			colours = tc.unpack_bytes(data[hsize + pal[1]: hsize + pal[1] + 1024], 256, 1, "BGRA")[0]
		flat = colours[idx]
	elif how[0] == "bytes":
		flat = tc.unpack_bytes(px, n, 1, how[1])[0]
	else:
		flat = tc.unpack_packed(px, n, 1, how[1], how[2])[0]
	if swizzled:
		return tc.unswizzle_morton(flat, w, h)
	return flat.reshape(h, w, 4)


# -- Dreamcast -------------------------------------------------------------------------------

_DC_PIXEL = {0: "A1R5G5B5", 1: "R5G6B5", 2: "A4R4G4B4"}


def _dc_untwiddle(flat: "np.ndarray", w: int, h: int) -> "np.ndarray":
	"""Twiddled texels (square, or a rectangle of square twiddled blocks) → (h, w, ...)."""
	s = min(w, h)
	blocks = []
	per = s * s
	for i in range(max(w, h) // s):
		blk = tc.unswizzle_morton(flat[i * per:(i + 1) * per], s, s, x_first=False)
		blocks.append(blk)
	return np.concatenate(blocks, axis=1 if w > h else 0)


def _dc_yuv422(words: "np.ndarray", w: int, h: int) -> "np.ndarray":
	v = words.reshape(h, w // 2, 2).astype(np.int32)
	u, y0 = v[..., 0] & 0xFF, v[..., 0] >> 8
	vv, y1 = v[..., 1] & 0xFF, v[..., 1] >> 8
	out = np.empty((h, w, 4), np.uint8)
	for k, y in ((0, y0), (1, y1)):
		r = y + 1.402 * (vv - 128)
		g = y - 0.344136 * (u - 128) - 0.714136 * (vv - 128)
		b = y + 1.772 * (u - 128)
		out[:, k::2, 0], out[:, k::2, 1], out[:, k::2, 2] = (np.clip(c, 0, 255) for c in (r, g, b))
	out[..., 3] = 255
	return out


def read_dreamcast_pvr(data: bytes, warnings: list[str] | None = None, palette: "np.ndarray | None" = None) -> "np.ndarray":
	"""Sega Dreamcast PVR (optional GBIX chunk, then PVRT)."""
	pos = 0
	if data[:4] == b"GBIX":
		pos = 8 + struct.unpack_from("<I", data, 4)[0]
	if data[pos:pos + 4] != b"PVRT":
		raise ValueError("not a Dreamcast PVR")
	size, pix, dfmt, _p, w, h = struct.unpack_from("<IBBHHH", data, pos + 4)
	body = data[pos + 16: pos + 8 + size]
	twiddled = dfmt in (1, 2, 5, 6, 0x0D, 0x12)
	mip = dfmt in (2, 4, 0x11, 0x12)
	if dfmt in (3, 4, 0x10, 0x11):	# VQ: codebook of 2×2 texel entries, then twiddled indices
		if dfmt in (0x10, 0x11):
			entries = (16 if w <= 16 else 32 if w <= 32 else 128 if w <= 64 else 256) if dfmt == 0x10 else \
				(16 if w <= 16 else 64 if w <= 32 else 256)
		else:
			entries = 256
		book = body[: entries * 8]
		idx = np.frombuffer(body, np.uint8, (w // 2) * (h // 2), len(body) - (w // 2) * (h // 2))
		if pix not in _DC_PIXEL:
			raise ValueError("Dreamcast VQ: unsupported pixel format")
		cb = tc.unpack_packed(book, entries * 4, 1, _DC_PIXEL[pix], 16)[0].reshape(entries, 2, 2, 4)
		cb = cb.transpose(0, 2, 1, 3)	# entries are twiddled: (x0,y0) (x0,y1) (x1,y0) (x1,y1)
		grid = tc.unswizzle_morton(np.minimum(idx, entries - 1), w // 2, h // 2, x_first=False)
		return cb[grid].transpose(0, 2, 1, 3, 4).reshape(h, w, 4)
	if dfmt in (5, 6):	# palettised, palette in a separate .pvp
		bpp = 4 if dfmt == 5 else 8
		n = w * h
		start = len(body) - n * bpp // 8 if mip else 0
		raw = np.frombuffer(body, np.uint8, n * bpp // 8, start)
		if bpp == 4:
			idx = np.empty(n, np.uint8)
			idx[0::2], idx[1::2] = raw & 15, raw >> 4
		else:
			idx = raw
		if palette is None:
			_warn(warnings, "Dreamcast paletted PVR without a .pvp palette; greyscale used")
			palette = tc.GREY_RGBA[:: 16] if bpp == 4 else tc.GREY_RGBA
		return _dc_untwiddle(palette[np.minimum(idx, len(palette) - 1)], w, h)
	n = w * h
	start = len(body) - n * 2 if mip else 0	# mip chains are stored smallest first
	words = np.frombuffer(body, "<u2", n, start)
	if pix == 3:
		if twiddled:
			raise ValueError("twiddled YUV422 Dreamcast textures are not supported")
		return _dc_yuv422(words, w, h)
	if pix not in _DC_PIXEL:
		raise ValueError(f"Dreamcast PVR: unsupported pixel format {pix}")
	flat = tc.unpack_packed(words.tobytes(), n, 1, _DC_PIXEL[pix], 16)[0]
	if twiddled:
		return _dc_untwiddle(flat, w, h)
	return flat.reshape(h, w, 4)


def read_pvp(data: bytes) -> "np.ndarray":
	"""Dreamcast palette (.pvp, PVPL)."""
	if data[:4] != b"PVPL":
		raise ValueError("not a PVPL palette")
	fmt = data[8]
	n = struct.unpack_from("<H", data, 14)[0]
	if fmt == 6:	# ARGB8888
		return tc.unpack_bytes(data[16:16 + n * 4], n, 1, "BGRA")[0]
	return tc.unpack_packed(data[16:16 + n * 2], n, 1, _DC_PIXEL.get(fmt, "A1R5G5B5"), 16)[0]


def load_dreamcast_pvr(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	path = Path(path)
	pvp = path.with_suffix(".pvp")
	pal = read_pvp(pvp.read_bytes()) if pvp.is_file() else None
	return read_dreamcast_pvr(path.read_bytes(), warnings, pal)


# -- dispatch --------------------------------------------------------------------------------


def load(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	path = Path(path)
	ext = path.suffix.lower()
	if ext == ".tex0":
		return load_tex0(path, warnings)
	if ext == ".ncgr":
		return load_ncgr(path, warnings)
	data = path.read_bytes()
	if ext == ".tpl":
		return read_tpl(data, warnings)
	if ext == ".bti":
		return read_bti(data, warnings)
	if ext in (".bclim", ".bflim"):
		return read_bclim(data, warnings)
	if ext == ".tim":
		return read_tim(data, warnings)
	if ext in (".tm2", ".tim2"):
		return read_tm2(data, warnings)
	if ext == ".xpr":
		return read_xpr(data, warnings)
	if ext == ".pvr":
		return load_dreamcast_pvr(path, warnings)
	raise ValueError(f"unrecognised console texture {ext}")
