"""GameCube texture containers written from GxTexture levels: TPL and BTI.

TPL (texture palette library) is the Dolphin SDK's own container, what
TEXPalette/TPLGet load, and holds any number of images, each with an
optional palette and a full mip chain. BTI is the single-texture header
J3D (Nintendo EAD's engine) uses, also stand-alone. Both are big-endian.

TPL (YAGCD, Dolphin's TextureDecoder, wiki.tockdom.com "TPL"):

	0x00  u32 magic 0x0020AF30, u32 image count, u32 table offset (0x0C)
	table: count × { u32 image header offset, u32 palette header offset (0 = none) }
	palette header (12 B): u16 entries, u8 unpacked, u8 pad, u32 TLUT format, u32 data offset
	image header (36 B): u16 height, u16 width, u32 format, u32 data offset,
		u32 wrap S, u32 wrap T, u32 min filter, u32 mag filter, f32 LOD bias,
		u8 edge LOD, u8 min LOD, u8 max LOD, u8 unpacked
	every data block 32-byte aligned; mip levels follow the top level back to
	back, each padded to whole tiles, as GX_InitTexObjLOD expects.

BTI (32-byte header, offsets relative to the header):

	u8 format, u8 alpha, u16 width, u16 height, u8 wrap S, u8 wrap T,
	u8 palettes on, u8 TLUT format, u16 entries, u32 palette offset,
	u8 mipmaps on, u8 edge LOD, u8 bias clamp, u8 max aniso, u8 min filter,
	u8 mag filter, s8 min LOD×8, s8 max LOD×8, u8 image count (levels),
	u8 pad, s16 LOD bias×100, u32 image offset
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from . import gxtex

TPL_MAGIC = 0x0020AF30
GX_CLAMP, GX_REPEAT, GX_MIRROR = 0, 1, 2
GX_NEAR, GX_LINEAR, GX_NEAR_MIP_NEAR, GX_LIN_MIP_NEAR, GX_NEAR_MIP_LIN, GX_LIN_MIP_LIN = range(6)
WRAP = {"clamp": GX_CLAMP, "repeat": GX_REPEAT, "mirror": GX_MIRROR}


@dataclass
class TplImage:
	"""One texture: its levels (largest first, all the same format) and how to sample it."""
	levels: list[gxtex.GxTexture]
	wrap_s: int = GX_REPEAT
	wrap_t: int = GX_REPEAT
	name: str = ""	# not stored in the file; for reports and .gxmat
	meta: dict = field(default_factory=dict)

	@property
	def fmt(self) -> str:
		return self.levels[0].fmt

	@property
	def width(self) -> int:
		return self.levels[0].width

	@property
	def height(self) -> int:
		return self.levels[0].height

	@property
	def tlut(self) -> bytes:
		return self.levels[0].tlut

	@property
	def data(self) -> bytes:
		return b"".join(lv.data for lv in self.levels)

	@property
	def size(self) -> int:
		"""Bytes the texture takes in memory: every level plus its TLUT."""
		return sum(len(lv.data) for lv in self.levels) + len(self.tlut)

	def decode(self, level: int = 0) -> "np.ndarray":
		return self.levels[level].decode()


def _align(n: int, a: int = 32) -> int:
	return -(-n // a) * a


def write_tpl(images: list[TplImage]) -> bytes:
	n = len(images)
	table = 0x0C
	pos = table + 8 * n
	pal_hdr: list[int] = []
	for im in images:
		pal_hdr.append(pos if im.tlut else 0)
		pos += 12 if im.tlut else 0
	img_hdr = []
	for _ in images:
		img_hdr.append(pos)
		pos += 36
	pos = _align(pos)
	pal_data, img_data = [], []
	for im in images:
		pal_data.append(pos if im.tlut else 0)
		pos = _align(pos + len(im.tlut)) if im.tlut else pos
		img_data.append(pos)
		pos = _align(pos + len(im.data))
	out = bytearray(pos)
	struct.pack_into(">III", out, 0, TPL_MAGIC, n, table)
	for i, im in enumerate(images):
		struct.pack_into(">II", out, table + 8 * i, img_hdr[i], pal_hdr[i])
		if im.tlut:
			struct.pack_into(">HBBII", out, pal_hdr[i], len(im.tlut) // 2, 0, 0, im.levels[0].tlut_fmt, pal_data[i])
			out[pal_data[i]:pal_data[i] + len(im.tlut)] = im.tlut
		mips = len(im.levels) > 1
		struct.pack_into(">HHIIIIIIfBBBB", out, img_hdr[i], im.height, im.width, gxtex.FORMATS[im.fmt][0], img_data[i],
			im.wrap_s, im.wrap_t, GX_LIN_MIP_LIN if mips else GX_LINEAR, GX_LINEAR, 0.0,
			0, 0, len(im.levels) - 1, 0)
		d = im.data
		out[img_data[i]:img_data[i] + len(d)] = d
	return bytes(out)


def read_tpl(data: bytes) -> list[TplImage]:
	"""Every image of a TPL with all its mip levels (consoletex.read_tpl decodes only the first)."""
	magic, n, table = struct.unpack_from(">III", data, 0)
	if magic != TPL_MAGIC:
		raise ValueError("not a TPL file")
	out = []
	for i in range(n):
		ih, ph = struct.unpack_from(">II", data, table + 8 * i)
		h, w, fid, off, ws, wt, _mn, _mg, _bias, _edge, _minl, maxl, _u = struct.unpack_from(">HHIIIIIIfBBBB", data, ih)
		fmt = gxtex.FORMAT_NAMES[fid]
		tlut, tfmt = b"", gxtex.GX_TL_RGB565
		if ph:
			cnt, _u2, _p, tfmt, po = struct.unpack_from(">HBBII", data, ph)
			tlut = data[po:po + cnt * 2]
		levels = []
		lw, lh = w, h
		for _ in range(maxl + 1):
			size = gxtex.tex_size(lw, lh, fmt)
			levels.append(gxtex.GxTexture(fmt, lw, lh, data[off:off + size], tlut, tfmt))
			off += size
			lw, lh = max(1, lw // 2), max(1, lh // 2)
		out.append(TplImage(levels, ws, wt))
	return out


def write_bti(im: TplImage) -> bytes:
	"""One texture as a stand-alone BTI: header, then TLUT, then the levels."""
	hdr = 32
	pal_off = hdr if im.tlut else 0
	img_off = _align(hdr + len(im.tlut)) if im.tlut else hdr
	mips = len(im.levels) > 1
	alpha = 1 if im.fmt in ("IA4", "IA8", "RGB5A3", "RGBA8") or im.levels[0].tlut_fmt != gxtex.GX_TL_RGB565 and im.tlut else 0
	head = struct.pack(">BBHHBBBBHIBBBBBBbbBBhI", gxtex.FORMATS[im.fmt][0], alpha, im.width, im.height,
		im.wrap_s, im.wrap_t, 1 if im.tlut else 0, im.levels[0].tlut_fmt if im.tlut else 0, len(im.tlut) // 2, pal_off,
		1 if mips else 0, 0, 0, 0, GX_LIN_MIP_LIN if mips else GX_LINEAR, GX_LINEAR, 0, (len(im.levels) - 1) * 8,
		len(im.levels), 0, 0, img_off)
	assert len(head) == hdr
	body = bytearray(head)
	body += im.tlut
	body += bytes(img_off - len(body))
	body += im.data
	return bytes(body)
