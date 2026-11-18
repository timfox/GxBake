"""The wider image-format coverage: each reader against a file built here from the format's layout."""

import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from gxbake.core import consoletex, gputex, gxtex, imageformats, images, retrotex
from gxbake.core import texcodec as tc
from gxbake.core.project import Kind
from gxbake.core.project import detect_kind as detect_kind_of


def _img(w=8, h=8):
	"""A recognisable RGBA test card: red/green gradients, blue checker, alpha hole top-left."""
	y, x = np.mgrid[0:h, 0:w]
	rgba = np.zeros((h, w, 4), np.uint8)
	rgba[..., 0] = x * 255 // max(1, w - 1)
	rgba[..., 1] = y * 255 // max(1, h - 1)
	rgba[..., 2] = ((x // 2 + y // 2) & 1) * 255
	rgba[..., 3] = 255
	rgba[0, 0, 3] = 0
	return rgba


def _close(a, b, tol):
	assert a.shape == b.shape, (a.shape, b.shape)
	assert np.abs(a.astype(int) - b.astype(int)).max() <= tol


# a BC1 block that is solid red (c0 = red, c1 = blue, every index 0)
BC1_RED = struct.pack("<HHI", 0xF800, 0x001F, 0)


# -- registry and wiring ------------------------------------------------------------------------


def test_registry_has_no_duplicates_and_feeds_image_extensions():
	seen = [e for f in imageformats.FORMATS for e in f.extensions]
	assert len(seen) == len(set(seen))
	assert set(seen) == images.IMAGE_EXTENSIONS
	for ext in (".heic", ".jxl", ".exr", ".svg", ".ktx2", ".tpl", ".tim", ".utx", ".shp", ".koa", ".vtf", ".wal"):
		assert ext in images.IMAGE_EXTENSIONS


def test_every_image_extension_is_detected_as_an_image(tmp_path):
	for ext in sorted(images.IMAGE_EXTENSIONS):
		p = tmp_path / f"x{ext}"
		p.write_bytes(b"\0" * 16)
		assert detect_kind_of(p) == Kind.IMAGE, ext


def test_report_lists_every_format():
	rows = imageformats.report()
	assert len(rows) == len(imageformats.FORMATS)
	assert all(r["available"] or r["install"] or r["note"] for r in rows)
	groups = imageformats.groups()
	assert set(groups) == {f.group for f in imageformats.FORMATS}


def test_missing_dependency_message(monkeypatch):
	import importlib

	real = importlib.import_module

	def fake(name, *a, **k):
		if name == "texture2ddecoder":
			raise ImportError(name)
		return real(name, *a, **k)

	monkeypatch.setattr(tc, "have", lambda m: m != "texture2ddecoder")
	monkeypatch.setattr(importlib, "import_module", fake)
	with pytest.raises(tc.MissingDependency, match="texture2ddecoder"):
		tc.decode_blocks("astc4x4", bytes(16), 4, 4)


# -- shared decoders ------------------------------------------------------------------------------


def test_packed_and_byte_layouts():
	words = np.array([0xF800, 0x07E0, 0x001F, 0xFFFF], "<u2").tobytes()
	out = tc.unpack_packed(words, 4, 1, "R5G6B5")
	assert out[0].tolist() == [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]]
	out = tc.unpack_packed(np.array([0x8000 | 31], ">u2").tobytes(), 1, 1, "A1R5G5B5", big=True)
	assert out[0, 0].tolist() == [0, 0, 255, 255]
	assert tc.unpack_bytes(bytes([1, 2, 3, 4]), 1, 1, "BGRA")[0, 0].tolist() == [3, 2, 1, 4]


def test_morton_unswizzle_roundtrip():
	for w, h in ((8, 8), (8, 2), (2, 8), (16, 4)):
		img = np.arange(w * h).reshape(h, w)
		idx = tc.morton_index(w, h).reshape(-1)
		flat = np.empty(w * h, int)
		flat[idx] = img.reshape(-1)
		assert (tc.unswizzle_morton(flat, w, h) == img).all()
		assert sorted(idx.tolist()) == list(range(w * h))


def test_block_decoders():
	assert tc.decode_blocks("bc1", BC1_RED, 4, 4)[..., :3].reshape(-1, 3).tolist() == [[255, 0, 0]] * 16
	assert tc.compressed_size("astc6x6", 13, 7) == 3 * 2 * 16
	assert tc.compressed_size("pvrtc4", 4, 4) == 32


def test_float_to_rgba8_tonemaps_only_bright_images():
	low = np.full((2, 2, 4), 0.5, np.float32)
	assert tc.float_to_rgba8(low)[0, 0, 0] == 188	# 0.5 linear in sRGB
	hi = np.full((2, 2, 3), 40.0, np.float32)
	assert tc.float_to_rgba8(hi).dtype == np.uint8


# -- GPU containers -----------------------------------------------------------------------------------


def _ktx1(gl_type, gl_format, internal, w, h, level):
	head = gputex.KTX1_MAGIC + struct.pack("<13I", 0x04030201, gl_type, 1, gl_format, internal, gl_format, w, h, 0, 0, 1, 1, 0)
	return head + struct.pack("<I", len(level)) + level


def test_ktx1_rgba8_and_bc1(tmp_path):
	src = _img(4, 4)
	p = tmp_path / "a.ktx"
	p.write_bytes(_ktx1(0x1401, 0x1908, 0x8058, 4, 4, src.tobytes()))
	assert (images.load_rgba(p) == src).all()
	p.write_bytes(_ktx1(0, 0, 0x83F0, 4, 4, BC1_RED))
	assert images.load_rgba(p)[2, 2, :3].tolist() == [255, 0, 0]


def test_ktx1_rgb_rows_are_padded(tmp_path):
	src = _img(3, 2)[..., :3]
	rows = b"".join(src[y].tobytes() + b"\0\0\0" for y in range(2))	# 9 bytes → 12 per row
	p = tmp_path / "a.ktx"
	p.write_bytes(_ktx1(0x1401, 0x1907, 0x8051, 3, 2, rows))
	assert (images.load_rgba(p)[..., :3] == src).all()


def _ktx2(vk, w, h, level, scheme=0, raw_len=None):
	head = gputex.KTX2_MAGIC + struct.pack("<9I", vk, 1, w, h, 0, 0, 1, 1, scheme)
	index = struct.pack("<4I2Q", 0, 0, 0, 0, 0, 0)
	off = len(head) + len(index) + 24
	return head + index + struct.pack("<3Q", off, len(level), raw_len or len(level)) + level


def test_ktx2_rgba8_bc1_and_zstd(tmp_path):
	src = _img(4, 4)
	p = tmp_path / "a.ktx2"
	p.write_bytes(_ktx2(37, 4, 4, src.tobytes()))
	assert (images.load_rgba(p) == src).all()
	p.write_bytes(_ktx2(131, 4, 4, BC1_RED))
	assert images.load_rgba(p)[0, 0, :3].tolist() == [255, 0, 0]
	from gxbake.core import valvecomp

	if valvecomp.zstd_available():
		import zstandard

		z = zstandard.ZstdCompressor().compress(src.tobytes())
		p.write_bytes(_ktx2(37, 4, 4, z, scheme=2, raw_len=64))
		assert (images.load_rgba(p) == src).all()
	p.write_bytes(_ktx2(0, 4, 4, b"x", scheme=1))
	with pytest.raises(tc.MissingDependency, match="Basis"):
		images.load_rgba(p)


def test_pvr3_and_legacy_pvr(tmp_path):
	src = _img(4, 4)
	fmt = ord("r") | ord("g") << 8 | ord("b") << 16 | ord("a") << 24 | (8 | 8 << 8 | 8 << 16 | 8 << 24) << 32
	p = tmp_path / "a.pvr"
	p.write_bytes(struct.pack("<IIQIIIIIIIII", 0x03525650, 0, fmt, 0, 0, 4, 4, 1, 1, 1, 1, 0) + src.tobytes())
	assert (images.load_rgba(p) == src).all()
	p.write_bytes(struct.pack("<IIQIIIIIIIII", 0x03525650, 0, 7, 0, 0, 4, 4, 1, 1, 1, 1, 0) + BC1_RED)
	assert images.load_rgba(p)[1, 1, :3].tolist() == [255, 0, 0]
	words = np.full(16, 0x07E0, "<u2").tobytes()	# legacy v2 RGB565 green
	head = struct.pack("<11I", 52, 4, 4, 0, 0x12, len(words), 16, 0, 0, 0, 0) + b"PVR!" + struct.pack("<I", 1)
	p.write_bytes(head + words)
	assert images.load_rgba(p)[0, 0].tolist() == [0, 255, 0, 255]


def test_astc_void_extent(tmp_path):
	pytest.importorskip("texture2ddecoder")
	block = bytes([0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]) + struct.pack("<4H", 0xFFFF, 0, 0xFFFF, 0xFFFF)
	p = tmp_path / "a.astc"
	p.write_bytes(struct.pack("<I", 0x5CA1AB13) + bytes([4, 4, 1]) + (8).to_bytes(3, "little") * 2 + (1).to_bytes(3, "little")
		+ block * 4)
	out = images.load_rgba(p)
	assert out.shape == (8, 8, 4) and out[3, 5].tolist() == [255, 0, 255, 255]


def test_dds_float_fallback(tmp_path):
	src = np.full((4, 4, 4), 0.25, np.float16)
	head = bytearray(148)
	head[:4] = b"DDS "
	struct.pack_into("<I", head, 4, 124)
	struct.pack_into("<II", head, 12, 4, 4)
	head[84:88] = b"DX10"
	struct.pack_into("<I", head, 128, 10)
	p = tmp_path / "a.dds"
	p.write_bytes(bytes(head) + src.tobytes())
	out = images.load_rgba(p)
	assert out[0, 0, 0] == 137 and out[0, 0, 3] == 64	# 0.25 linear → sRGB; alpha 0.25


# -- consoles ----------------------------------------------------------------------------------------


def test_tpl_rgb5a3_and_ci8(tmp_path):
	src = _img(8, 8)
	tex = gxtex.encode(src, "RGB5A3")
	body = struct.pack(">III", 0x0020AF30, 1, 12) + struct.pack(">II", 20, 0)
	img_hdr = struct.pack(">HHII", 8, 8, gxtex.GX_TF_RGB5A3, 64) + bytes(20)
	data = (body + img_hdr).ljust(64, b"\0") + tex.data
	p = tmp_path / "a.tpl"
	p.write_bytes(data)
	_close(images.load_rgba(p), gxtex.decode("RGB5A3", 8, 8, tex.data), 0)
	_close(images.load_rgba(p)[1:, 1:], src[1:, 1:], 16)
	# CI8 with an RGB565 palette
	pal = np.zeros((256, 4), np.uint8)
	pal[1] = (255, 0, 0, 255)
	idx = np.ones((8, 8), np.uint8)
	ci = gxtex.encode_indexed(idx, pal[:2], fmt="CI8")
	pal_hdr = struct.pack(">HBBII", len(ci.tlut) // 2, 0, 0, gxtex.GX_TL_RGB565, 128)
	body = struct.pack(">III", 0x0020AF30, 1, 12) + struct.pack(">II", 20, 52)
	img_hdr = struct.pack(">HHII", 8, 8, gxtex.GX_TF_CI8, 64) + bytes(20)
	data = (body + img_hdr + pal_hdr).ljust(64, b"\0") + ci.data
	data = data.ljust(128, b"\0") + ci.tlut
	p.write_bytes(data)
	assert images.load_rgba(p)[4, 4].tolist() == [255, 0, 0, 255]


def test_bti_and_tex0_with_plt0(tmp_path):
	src = _img(8, 8)
	tex = gxtex.encode(src, "RGBA8")
	hdr = struct.pack(">BBHHBBBBHIBBBBBBhHBB", gxtex.GX_TF_RGBA8, 1, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
	hdr = hdr.ljust(0x1C, b"\0") + struct.pack(">I", 0x20)
	(tmp_path / "a.bti").write_bytes(hdr + tex.data)
	assert (images.load_rgba(tmp_path / "a.bti") == src).all()
	# TEX0 CI4 + PLT0 IA8
	idx = np.full((8, 8), 3, np.uint8)
	pal = np.zeros((16, 4), np.uint8)
	pal[3] = (200, 200, 200, 255)
	ci = gxtex.encode_indexed(idx, pal, gxtex.GX_TL_IA8, fmt="CI4")
	tex0 = b"TEX0" + struct.pack(">IIiII", 0x40 + len(ci.data), 3, 0, 0x40, 0) + struct.pack(">IHHII", 1, 8, 8,
		gxtex.GX_TF_CI4, 1)
	(tmp_path / "b.tex0").write_bytes(tex0.ljust(0x40, b"\0") + ci.data)
	plt0 = b"PLT0" + struct.pack(">IIiII", 0x40 + len(ci.tlut), 3, 0, 0x40, 0) + struct.pack(">IH", gxtex.GX_TL_IA8, 16)
	(tmp_path / "B.PLT0").write_bytes(plt0.ljust(0x40, b"\0") + ci.tlut)
	assert images.load_rgba(tmp_path / "b.tex0")[0, 0].tolist() == [200, 200, 200, 255]


def _nitro(magic, section, body):
	sec = section + struct.pack("<I", 8 + len(body)) + body
	return magic + struct.pack("<HHIHH", 0xFEFF, 0x0101, 16 + len(sec), 16, 1) + sec


def test_ds_ncgr_with_nclr(tmp_path):
	colours = np.zeros(16, "<u2")
	colours[1] = 31	# red (BGR555)
	colours[2] = 31 << 10	# blue
	nclr = _nitro(b"RLCN", b"TTLP", struct.pack("<IIII", 3, 0, 32, 0x10) + colours.tobytes())
	# two 8×8 tiles side by side: tile 0 all colour 1, tile 1 all colour 2 (4 bpp, low nibble first)
	tiles = bytes([0x11] * 32 + [0x22] * 32)
	ncgr = _nitro(b"RGCN", b"RAHC", struct.pack("<HHIIIII", 1, 2, 3, 0, 0, len(tiles), 0x18) + tiles)
	(tmp_path / "hero.ncgr").write_bytes(ncgr)
	(tmp_path / "hero.nclr").write_bytes(nclr)
	out = images.load_rgba(tmp_path / "hero.ncgr")
	assert out.shape == (8, 16, 4)
	assert out[3, 3].tolist() == [255, 0, 0, 255] and out[3, 12].tolist() == [0, 0, 255, 255]


def _ctr_encode_rgba8(rgba):
	"""The 3DS layout written independently: bottom-up, 8×8 tiles, Morton inside, bytes ABGR."""
	h, w = rgba.shape[:2]
	flipped = rgba[::-1]
	out = bytearray(w * h * 4)
	for y in range(h):
		for x in range(w):
			tile = (y // 8) * (w // 8) + x // 8
			lx, ly = x & 7, y & 7
			m = 0
			for b in range(3):
				m |= ((lx >> b) & 1) << (2 * b) | ((ly >> b) & 1) << (2 * b + 1)
			o = (tile * 64 + m) * 4
			r, g, b_, a = flipped[y, x]
			out[o:o + 4] = bytes((a, b_, g, r))
	return bytes(out)


def test_3ds_bclim_rgba8(tmp_path):
	src = _img(16, 8)
	data = _ctr_encode_rgba8(src)
	foot = b"CLIM" + struct.pack("<HHIIHH", 0xFEFF, 0x14, 0x02020000, len(data) + 0x28, 1, 0)
	imag = b"imag" + struct.pack("<IHHII", 0x10, 16, 8, 9, len(data))
	(tmp_path / "a.bclim").write_bytes(data + foot + imag)
	assert (images.load_rgba(tmp_path / "a.bclim") == src).all()


def test_ps1_tim_8bpp_and_16bpp(tmp_path):
	clut = np.zeros(256, "<u2")
	clut[5] = 31 | (31 << 5)	# yellow
	clut_block = struct.pack("<IHHHH", 12 + 512, 0, 0, 256, 1) + clut.tobytes()
	pix = bytes([5] * 16 + [0] * 16)	# 4×8: top half colour 5, bottom colour 0 (transparent black)
	img_block = struct.pack("<IHHHH", 12 + len(pix), 0, 0, 2, 8) + pix
	(tmp_path / "a.tim").write_bytes(struct.pack("<II", 0x10, 9) + clut_block + img_block)
	out = images.load_rgba(tmp_path / "a.tim")
	assert out.shape == (8, 4, 4)
	assert out[0, 0].tolist() == [255, 255, 0, 255] and out[7, 3, 3] == 0
	words = np.full(4, 31 << 10, "<u2").tobytes()	# blue, 15-bit direct
	(tmp_path / "b.tim").write_bytes(struct.pack("<II", 0x10, 2) + struct.pack("<IHHHH", 20, 0, 0, 2, 2) + words)
	assert images.load_rgba(tmp_path / "b.tim")[1, 1].tolist() == [0, 0, 255, 255]


def _tm2(img_type, clut_type, w, h, img, clut=b"", ncol=0):
	hsize = 48
	pic = struct.pack("<IIIHHBBBBHH", hsize + len(img) + len(clut), len(clut), len(img), hsize, ncol, 0, 1, clut_type,
		img_type, w, h).ljust(hsize, b"\0")
	return b"TIM2" + struct.pack("<BBH", 4, 0, 1) + bytes(8) + pic + img + clut


def test_ps2_tim2_32bit_and_csm1_clut(tmp_path):
	img = bytes([255, 0, 0, 0x80] * 4)
	(tmp_path / "a.tm2").write_bytes(_tm2(3, 0, 2, 2, img))
	assert images.load_rgba(tmp_path / "a.tm2")[0, 0].tolist() == [255, 0, 0, 255]
	# CSM1 8-bit CLUT: logical colour 8 is stored at position 16
	clut = bytearray(256 * 4)
	clut[16 * 4:16 * 4 + 4] = bytes([0, 255, 0, 0x80])
	(tmp_path / "b.tm2").write_bytes(_tm2(5, 3, 2, 2, bytes([8] * 4), bytes(clut), 256))
	assert images.load_rgba(tmp_path / "b.tm2")[1, 0].tolist() == [0, 255, 0, 255]


def test_xbox_xpr_swizzled_argb(tmp_path):
	src = _img(4, 2)
	idx = tc.morton_index(4, 2).reshape(-1)
	flat = np.empty((8, 4), np.uint8)
	flat[idx] = src.reshape(-1, 4)
	px = flat[:, [2, 1, 0, 3]].tobytes()	# BGRA in memory
	fmt = (0x06 << 8) | (2 << 20) | (1 << 24)
	hdr = b"XPR0" + struct.pack("<II", 2048 + len(px), 2048) + struct.pack("<5I", 0x00040001, 0, 0, fmt, 0) + struct.pack("<I", 0xFFFFFFFF)
	(tmp_path / "a.xpr").write_bytes(hdr.ljust(2048, b"\xad") + px)
	assert (images.load_rgba(tmp_path / "a.xpr") == src).all()


def test_dreamcast_pvr_twiddled_rect_and_vq(tmp_path):
	src = _img(8, 8)
	words = gxtex.rgb565(src[..., :3]).astype("<u2")
	ref = gxtex.from_rgb565(words.astype(np.int32))
	idx = tc.twiddle_index(8, 8).reshape(-1)
	tw = np.empty(64, "<u2")
	tw[idx] = words.reshape(-1)

	def pvr(pix, dfmt, body, w=8, h=8):
		return b"GBIX" + struct.pack("<II", 8, 0) + bytes(4) + b"PVRT" + struct.pack("<IBBHHH", 8 + len(body), pix, dfmt, 0, w, h) + body

	(tmp_path / "a.pvr").write_bytes(pvr(1, 1, tw.tobytes()))
	assert (images.load_rgba(tmp_path / "a.pvr") == ref).all()
	(tmp_path / "b.pvr").write_bytes(pvr(1, 9, words.tobytes()))
	assert (images.load_rgba(tmp_path / "b.pvr") == ref).all()
	# VQ: codebook entry 0 = 2×2 of red, every index 0
	book = np.zeros(256 * 4, "<u2")
	book[:4] = 0xF800
	(tmp_path / "c.pvr").write_bytes(pvr(1, 3, book.tobytes() + bytes(16)))
	assert images.load_rgba(tmp_path / "c.pvr")[7, 7].tolist() == [255, 0, 0, 255]


# -- PC games -----------------------------------------------------------------------------------------


def _palette():
	pal = np.zeros((256, 3), np.uint8)
	pal[:, 0] = np.arange(256)
	pal[7] = (10, 20, 30)
	return pal.tobytes()


def test_doom_patch_flat_and_quake_qpic(tmp_path):
	(tmp_path / "PLAYPAL.lmp").write_bytes(_palette() * 14)
	# 2×3 patch: column 0 has a post of 2 pixels at row 1, column 1 empty
	cols = 8 + 8
	post = bytes([1, 2, 0, 7, 7, 0, 0xFF])
	patch = struct.pack("<hhhh", 2, 3, 0, 0) + struct.pack("<II", cols, cols + len(post)) + post + b"\xff"
	(tmp_path / "patch.lmp").write_bytes(patch)
	out = images.load_rgba(tmp_path / "patch.lmp")
	assert out.shape == (3, 2, 4)
	assert out[1, 0].tolist() == [10, 20, 30, 255] and out[0, 0, 3] == 0 and out[1, 1, 3] == 0
	(tmp_path / "flat.lmp").write_bytes(bytes([7]) * 4096)
	assert images.load_rgba(tmp_path / "flat.lmp")[63, 63].tolist() == [10, 20, 30, 255]
	assert images.load_rgba(tmp_path / "PLAYPAL.lmp").shape == (128, 128, 4)
	qp = tmp_path / "q" / "gfx"
	qp.mkdir(parents=True)
	(qp / "palette.lmp").write_bytes(_palette())
	(qp / "conchars.lmp").write_bytes(struct.pack("<II", 2, 1) + bytes([7, 255]))
	out = images.load_rgba(qp / "conchars.lmp")
	assert out[0, 0].tolist() == [10, 20, 30, 255] and out[0, 1, 3] == 0


def test_doom_palette_from_iwad(tmp_path):
	lump = _palette()
	wad = b"IWAD" + struct.pack("<ii", 1, 12 + len(lump)) + lump + struct.pack("<ii8s", 12, len(lump), b"PLAYPAL")
	(tmp_path / "doom2.wad").write_bytes(wad)
	(tmp_path / "flat.lmp").write_bytes(bytes([7]) * 4096)
	assert images.load_rgba(tmp_path / "flat.lmp")[0, 0].tolist() == [10, 20, 30, 255]


def test_build_art_sheet(tmp_path):
	pal = bytes(np.repeat(np.arange(256) % 64, 3).astype(np.uint8))
	(tmp_path / "PALETTE.DAT").write_bytes(pal + bytes(100))
	# two tiles: 2×3 and an empty one, then a 1×1
	xs, ys = [2, 0, 1], [3, 0, 1]
	pix = bytes([10, 11, 12, 20, 21, 22]) + bytes([255])
	art = struct.pack("<iiii", 1, 3, 0, 2) + struct.pack("<3h", *xs) + struct.pack("<3h", *ys) + bytes(12) + pix
	(tmp_path / "TILES000.ART").write_bytes(art)
	w = []
	out = images.load_rgba(tmp_path / "TILES000.ART", w)
	assert out.shape == (3, 4, 4)
	assert out[0, 1, 0] == 20 * 255 // 63	# column-major: x=1,y=0 is the 4th byte
	assert out[0, 3, 3] == 0	# index 255 is transparent
	assert any("2 tiles" in m for m in w)


def _lcw_literal(data: bytes) -> bytes:
	out = bytearray()
	for i in range(0, len(data), 63):
		chunk = data[i:i + 63]
		out += bytes([0x80 | len(chunk)]) + chunk
	return bytes(out) + b"\x80"


def test_westwood_shp_td_lcw_and_xor(tmp_path):
	(tmp_path / "temperat.pal").write_bytes(bytes(np.repeat(np.arange(256) % 64, 3).astype(np.uint8)))
	w, h = 4, 2
	f0 = bytes([1, 2, 3, 4, 5, 6, 7, 8])
	lcw = _lcw_literal(f0)
	xor = bytes([2, 0, 0]) + b"\x00"  # XOR-copy 2 bytes (0, 0)... then end marker below
	xor = bytes([0x02, 9, 9]) + bytes([0x80, 0, 0])	# xor the first two pixels with 9, then end
	base = 14 + 8 * 4
	offs = [(0x80 << 24 | base, 0), (0x40 << 24 | (base + len(lcw)), base), (base + len(lcw) + len(xor), 0),
		(0, 0)]
	head = struct.pack("<7H", 2, 0, 0, w, h, 16, 0) + b"".join(struct.pack("<II", a, b) for a, b in offs)
	(tmp_path / "unit.shp").write_bytes(head + lcw + xor)
	out = images.load_rgba(tmp_path / "unit.shp")
	assert out.shape == (2, 9, 4)	# two 4×2 frames with a 1-pixel gap
	assert out[0, 0, 0] == 1 * 255 // 63 and out[0, 5, 0] == (1 ^ 9) * 255 // 63 and out[0, 8, 0] == 4 * 255 // 63


def test_westwood_shp_tiberian_sun(tmp_path):
	frame = struct.pack("<4HIIII", 1, 1, 2, 1, 0, 0, 0, 8 + 24)
	(tmp_path / "a.shp").write_bytes(struct.pack("<4H", 0, 4, 3, 1) + frame + bytes([5, 6]))
	out = images.load_rgba(tmp_path / "a.shp")
	assert out.shape == (3, 4, 4) and out[1, 1, 0] == 5 and out[0, 0, 3] == 0


def test_fallout_frm(tmp_path):
	(tmp_path / "COLOR.PAL").write_bytes(bytes(np.repeat(np.arange(256) % 64, 3).astype(np.uint8)) + bytes(32768))
	frames = b"".join(struct.pack(">HHIhh", 2, 1, 2, 0, 0) + bytes([c, 0]) for c in (3, 4))
	head = struct.pack(">IHHH", 4, 10, 0, 2) + bytes(24) + struct.pack(">6I", 0, 0, 0, 0, 0, 0) + struct.pack(">I", len(frames))
	(tmp_path / "hero.frm").write_bytes(head + frames)
	out = images.load_rgba(tmp_path / "hero.frm")
	assert out.shape == (1, 5, 4) and out[0, 0, 0] == 3 * 255 // 63 and out[0, 1, 3] == 0


def _ci(v: int) -> bytes:
	"""Unreal compact index."""
	neg, v = v < 0, abs(v)
	out = bytearray([(0x80 if neg else 0) | (v & 0x3F) | (0x40 if v >= 0x40 else 0)])
	v >>= 6
	while v:
		out.append((v & 0x7F) | (0x80 if v >= 0x80 else 0))
		v >>= 7
	return bytes(out)


def test_unreal_utx_p8(tmp_path):
	names = ["None", "Format", "USize", "VSize", "Palette", "Texture", "Core", "Class", "Tex1", "Pal1"]
	N = {n: i for i, n in enumerate(names)}
	pal = np.zeros((256, 4), np.uint8)
	pal[1] = (0, 128, 255, 0)
	pal_obj = _ci(N["None"]) + _ci(256) + pal.tobytes()
	pix = bytes([1] * 16)
	tex_obj = (_ci(N["Format"]) + bytes([0x01, 0]) + _ci(N["Palette"]) + bytes([0x05]) + _ci(2)
		+ _ci(N["USize"]) + bytes([0x22]) + struct.pack("<i", 4) + _ci(N["None"])
		+ _ci(1) + struct.pack("<i", 0) + _ci(len(pix)) + pix + struct.pack("<ii", 4, 4) + bytes([2, 2]))
	name_tab = b"".join(_ci(len(n) + 1) + n.encode() + b"\0" + struct.pack("<I", 0) for n in names)
	imports = b"".join(_ci(N["Core"]) + _ci(N["Class"]) + struct.pack("<i", 0) + _ci(N[c]) for c in ("Texture", "Palette"))
	hdr_size = 36
	no = hdr_size
	io_ = no + len(name_tab)
	data_off = io_ + len(imports)
	tex_off, pal_off = data_off, data_off + len(tex_obj)
	exports = (_ci(-1) + _ci(0) + struct.pack("<i", 0) + _ci(N["Tex1"]) + struct.pack("<I", 0) + _ci(len(tex_obj)) + _ci(tex_off)
		+ _ci(-2) + _ci(0) + struct.pack("<i", 0) + _ci(N["Pal1"]) + struct.pack("<I", 0) + _ci(len(pal_obj)) + _ci(pal_off))
	eo = pal_off + len(pal_obj)
	head = struct.pack("<IHHIIIIIII", 0x9E2A83C1, 69, 0, 0, len(names), no, 2, eo, 2, io_)
	(tmp_path / "a.utx").write_bytes(head + name_tab + imports + tex_obj + pal_obj + exports)
	out = images.load_rgba(tmp_path / "a.utx")
	assert out.shape == (4, 4, 4) and out[2, 2].tolist() == [0, 128, 255, 255]


# -- retro computers ----------------------------------------------------------------------------------


def _byterun(data: bytes) -> bytes:
	out = bytearray()
	for i in range(0, len(data), 128):
		chunk = data[i:i + 128]
		out += bytes([len(chunk) - 1]) + chunk
	return bytes(out)


def _iff(form, bmhd, cmap, body, camg=None):
	chunks = b"BMHD" + struct.pack(">I", len(bmhd)) + bmhd + b"CMAP" + struct.pack(">I", len(cmap)) + cmap
	if camg is not None:
		chunks += b"CAMG" + struct.pack(">II", 4, camg)
	chunks += b"BODY" + struct.pack(">I", len(body)) + body + (b"\0" if len(body) & 1 else b"")
	return b"FORM" + struct.pack(">I", 4 + len(chunks)) + form + chunks


def test_iff_ilbm_and_pbm(tmp_path):
	w, h = 16, 2
	idx = np.zeros((h, w), np.uint8)
	idx[0, :8] = 5
	idx[1, 8:] = 10
	rows = b""
	for y in range(h):
		for k in range(4):
			rows += np.packbits((idx[y] >> k) & 1).tobytes()
	cmap = bytes(np.repeat(np.arange(16) * 16, 3).astype(np.uint8))
	bmhd = struct.pack(">HHhhBBBBHBBhh", w, h, 0, 0, 4, 0, 1, 0, 0, 1, 1, w, h)
	(tmp_path / "a.lbm").write_bytes(_iff(b"ILBM", bmhd, cmap, _byterun(rows)))
	out = images.load_rgba(tmp_path / "a.lbm")
	assert out[0, 0, 0] == 80 and out[1, 15, 0] == 160 and out[0, 15, 0] == 0
	bmhd = struct.pack(">HHhhBBBBHBBhh", w, h, 0, 0, 8, 2, 0, 0, 10, 1, 1, w, h)
	(tmp_path / "b.iff").write_bytes(_iff(b"PBM ", bmhd, cmap, idx.tobytes()))
	out = images.load_rgba(tmp_path / "b.iff")
	assert out[0, 0, 0] == 80 and out[1, 15, 3] == 0	# masking 2: colour 10 is transparent


def test_degas_and_koala(tmp_path):
	pal = [0] * 16
	pal[1] = 0x700	# red on an ST
	words = np.zeros((200, 20, 4), ">u2")
	words[:, :, 0] = 0xFFFF	# plane 0 set → colour 1 everywhere
	(tmp_path / "a.pi1").write_bytes(struct.pack(">H16H", 0, *pal) + words.tobytes())
	out = images.load_rgba(tmp_path / "a.pi1")
	assert out.shape == (200, 320, 4) and out[100, 100].tolist() == [238, 0, 0, 255]
	data = bytearray(10003)
	data[2:8002] = bytes([0b01010101]) * 8000	# every pixel takes the screen-RAM high nibble
	data[8002:9002] = bytes([0x20]) * 1000	# colour 2 (red)
	(tmp_path / "a.koa").write_bytes(bytes(data))
	out = images.load_rgba(tmp_path / "a.koa")
	assert out.shape == (200, 320, 4) and out[50, 50, :3].tolist() == [0x68, 0x37, 0x2B]


# -- modern and pro formats ------------------------------------------------------------------------


def test_pfm_and_hdr_float(tmp_path):
	rows = np.zeros((2, 3, 3), np.float32)
	rows[0] = 0.5	# bottom row first in the file
	(tmp_path / "a.pfm").write_bytes(b"PF\n3 2\n-1.0\n" + rows.astype("<f4").tobytes())
	out = images.load_rgba(tmp_path / "a.pfm")
	assert out[1, 0, 0] == 188 and out[0, 0, 0] == 0


def test_exr(tmp_path):
	OpenEXR = pytest.importorskip("OpenEXR")
	rgb = np.full((4, 8, 3), 0.5, np.float32)
	with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, {"RGB": rgb}) as f:
		f.write(str(tmp_path / "a.exr"))
	out = images.load_rgba(tmp_path / "a.exr")
	assert out.shape == (4, 8, 4) and out[0, 0, 0] == 188


def test_svg(tmp_path):
	if not (tc.have("resvg_py") or tc.have("cairosvg")):
		pytest.skip("no SVG rasteriser")
	(tmp_path / "a.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">'
		'<rect width="20" height="10" fill="#ff0000"/></svg>')
	out = images.load_rgba(tmp_path / "a.svg")
	assert out.shape[1] == imageformats.SVG_LONG_SIDE and out.shape[0] == imageformats.SVG_LONG_SIDE // 2
	assert out[10, 10].tolist() == [255, 0, 0, 255]


def test_krita_and_openraster(tmp_path):
	from PIL import Image

	buf = io.BytesIO()
	Image.new("RGBA", (6, 4), (1, 2, 3, 255)).save(buf, "PNG")
	for ext in (".kra", ".ora"):
		with zipfile.ZipFile(tmp_path / f"a{ext}", "w") as z:
			z.writestr("mimetype", "image/openraster")
			z.writestr("mergedimage.png", buf.getvalue())
		assert images.load_rgba(tmp_path / f"a{ext}")[0, 0].tolist() == [1, 2, 3, 255]


@pytest.mark.parametrize("ext,fmt,module", [
	(".heic", "HEIF", "pillow_heif"), (".jxl", "JXL", "pillow_jxl"), (".avif", "AVIF", None),
	(".jp2", "JPEG2000", None), (".qoi", "QOI", None), (".ico", "ICO", None), (".pgm", "PPM", None),
	(".icns", "ICNS", None),
])
def test_pillow_and_plugin_formats(tmp_path, ext, fmt, module):
	from PIL import Image, features

	if module:
		mod = pytest.importorskip(module)
		if module == "pillow_heif":
			mod.register_heif_opener()
	if fmt == "AVIF" and not features.check("avif"):
		pytest.skip("Pillow without AVIF")
	mode = "L" if ext == ".pgm" else "RGB"
	img = Image.new(mode, (32, 32), 200 if mode == "L" else (200, 40, 40))
	p = tmp_path / f"a{ext}"
	img.save(p, fmt, **({"quality": 95} if fmt in ("HEIF", "AVIF", "JXL") else {}))
	out = images.load_rgba(p)
	assert out.shape[2] == 4
	want = (200, 200, 200) if mode == "L" else (200, 40, 40)
	assert np.abs(out[16, 16, :3].astype(int) - want).max() <= 12


def test_sixteen_bit_png_is_scaled_not_clipped(tmp_path):
	from PIL import Image

	v = np.full((4, 4), 0x8000, np.uint16)
	Image.fromarray(v).save(tmp_path / "a.png")
	assert images.load_rgba(tmp_path / "a.png")[0, 0, 0] == 128


def test_convert_a_tpl_end_to_end(tmp_path):
	from gxbake.core.project import Project, Settings

	src = _img(16, 16)
	tex = gxtex.encode(src, "RGB565")
	body = struct.pack(">III", 0x0020AF30, 1, 12) + struct.pack(">II", 20, 0)
	img_hdr = struct.pack(">HHII", 16, 16, gxtex.GX_TF_RGB565, 64) + bytes(20)
	(tmp_path / "in").mkdir()
	(tmp_path / "in" / "brick.tpl").write_bytes((body + img_hdr).ljust(64, b"\0") + tex.data)
	proj = Project(Settings())
	assets = proj.add_path(tmp_path / "in")
	assert [a.kind for a in assets] == [Kind.IMAGE]
	proj.convert(assets, tmp_path / "out", log=lambda *a: None)
	assert list((tmp_path / "out" / "media").glob("*.gct"))


def test_gui_file_filter_has_each_entry_once():
	pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
	from gxbake.gui.main_window import FILE_FILTER

	parts = FILE_FILTER.split(";;")
	names = [p.split(" (")[0] for p in parts]
	assert len(names) == len(set(names))
	for p in parts:
		globs = p[p.index("(") + 1:-1].split()
		assert len(globs) == len(set(globs)), p
	assert "*.tpl" in FILE_FILTER and "*.heic" in FILE_FILTER and "*.koa" in FILE_FILTER


def test_cli_formats_and_decode(tmp_path, capsys):
	from gxbake.cli import main

	assert main(["formats"]) == 0
	out = capsys.readouterr().out
	assert "Console textures" in out and "PlayStation TIM" in out
	assert main(["formats", "--json"]) == 0
	import json

	assert len(json.loads(capsys.readouterr().out)) == len(imageformats.FORMATS)
	words = np.full(4, 31 << 10, "<u2").tobytes()
	(tmp_path / "b.tim").write_bytes(struct.pack("<II", 0x10, 2) + struct.pack("<IHHHH", 20, 0, 0, 2, 2) + words)
	(tmp_path / "bad.tpl").write_bytes(b"nope" * 8)
	assert main(["decode", str(tmp_path / "b.tim"), str(tmp_path / "bad.tpl"), "-o", str(tmp_path / "png")]) == 1
	assert (tmp_path / "png" / "b.tim.png").is_file()
