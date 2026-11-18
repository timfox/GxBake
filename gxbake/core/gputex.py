"""GPU texture containers: Khronos KTX 1 and KTX 2, PowerVR .pvr (v3 and the
legacy v2 header), raw .astc and Basis Universal.

Only the top mip of the first face / layer is read; GxBake builds its own mip
chain. Block formats decode through texcodec (BCn in Pillow, ETC in etc.py or
texture2ddecoder, ASTC/PVRTC/ATC in texture2ddecoder). KTX 2 Zstandard
supercompression uses valvecomp's zstd; BasisLZ/UASTC payloads need the
Basis Universal transcoder, which has no maintained Python binding yet, so
they raise a clear error.

References: KTX 1.1 and KTX 2.0 specifications (Khronos), PowerVR Container
Format specification (Imagination), the ASTC file header in astcenc.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import texcodec as tc

# -- format tables ---------------------------------------------------------------------

# OpenGL internal formats (KTX 1) → texcodec kind
_GL_COMPRESSED = {
	0x83F0: "bc1", 0x83F1: "bc1", 0x83F2: "bc2", 0x83F3: "bc3",
	0x8C4C: "bc1", 0x8C4D: "bc1", 0x8C4E: "bc2", 0x8C4F: "bc3",	# sRGB S3TC
	0x8DBB: "bc4", 0x8DBC: "bc4", 0x8DBD: "bc5", 0x8DBE: "bc5",
	0x8E8C: "bc7", 0x8E8D: "bc7", 0x8E8E: "bc6h_sf", 0x8E8F: "bc6h",
	0x8D64: "etc1", 0x9274: "etc2", 0x9275: "etc2", 0x9276: "etc2_a1", 0x9277: "etc2_a1",
	0x9278: "etc2_eac", 0x9279: "etc2_eac", 0x9270: "r11", 0x9271: "r11", 0x9272: "rg11", 0x9273: "rg11",
	0x8C00: "pvrtc4", 0x8C01: "pvrtc2", 0x8C02: "pvrtc4", 0x8C03: "pvrtc2",
	0x8C92: "atc", 0x8C93: "atc_a", 0x87EE: "atc_a",
}
_ASTC_DIMS = ("4x4", "5x4", "5x5", "6x5", "6x6", "8x5", "8x6", "8x8", "10x5", "10x6", "10x8", "10x10", "12x10", "12x12")
for _i, _d in enumerate(_ASTC_DIMS):
	_GL_COMPRESSED[0x93B0 + _i] = "astc" + _d	# linear
	_GL_COMPRESSED[0x93D0 + _i] = "astc" + _d	# sRGB

# glFormat (base, unsized) → byte order; glType 0x1401 UNSIGNED_BYTE, 0x140B HALF_FLOAT, 0x1406 FLOAT
_GL_BASE = {0x1908: "RGBA", 0x1907: "RGB", 0x80E1: "BGRA", 0x80E0: "BGR", 0x1909: "L", 0x190A: "LA", 0x1906: "A",
	0x1903: "R", 0x8227: "RG"}
_GL_PACKED = {0x8363: ("R5G6B5", "RGB"), 0x8033: ("R4G4B4A4", "RGBA"), 0x8034: ("R5G5B5A1", "RGBA")}

# Vulkan formats (KTX 2) → kind, or ("bytes", order) / ("float", channels, dtype) / ("packed", fields, bits)
_VK: dict[int, object] = {
	9: ("bytes", "R"), 15: ("bytes", "R"), 16: ("bytes", "RG"), 22: ("bytes", "RG"), 23: ("bytes", "RGB"),
	29: ("bytes", "RGB"), 30: ("bytes", "BGR"), 36: ("bytes", "BGR"), 37: ("bytes", "RGBA"), 43: ("bytes", "RGBA"),
	44: ("bytes", "BGRA"), 50: ("bytes", "BGRA"),
	2: ("packed", "R4G4B4A4", 16), 3: ("packed", "B4G4R4A4", 16), 4: ("packed", "R5G6B5", 16), 5: ("packed", "B5G6R5", 16),
	6: ("packed", "R5G5B5A1", 16), 7: ("packed", "B5G5R5A1", 16), 8: ("packed", "A1R5G5B5", 16),
	76: ("float", "R", "<f2"), 83: ("float", "RG", "<f2"), 90: ("float", "RGB", "<f2"), 97: ("float", "RGBA", "<f2"),
	100: ("float", "R", "<f4"), 103: ("float", "RG", "<f4"), 106: ("float", "RGB", "<f4"), 109: ("float", "RGBA", "<f4"),
	131: "bc1", 132: "bc1", 133: "bc1", 134: "bc1", 135: "bc2", 136: "bc2", 137: "bc3", 138: "bc3",
	139: "bc4", 140: "bc4", 141: "bc5", 142: "bc5", 143: "bc6h", 144: "bc6h_sf", 145: "bc7", 146: "bc7",
	147: "etc2", 148: "etc2", 149: "etc2_a1", 150: "etc2_a1", 151: "etc2_eac", 152: "etc2_eac",
	153: "r11", 154: "r11", 155: "rg11", 156: "rg11",
}
for _i, _d in enumerate(_ASTC_DIMS):
	_VK[157 + 2 * _i] = _VK[158 + 2 * _i] = "astc" + _d
for _i, _k in enumerate(("pvrtc2", "pvrtc4", "pvrtc2", "pvrtc4")):	# PVRTC1 2/4bpp, PVRTC2 2/4bpp (unorm)
	_VK[1000054000 + _i] = _k
	_VK[1000054004 + _i] = _k	# sRGB

# PVR v3 compressed pixel format ids
_PVR3_COMPRESSED = {0: "pvrtc2", 1: "pvrtc2", 2: "pvrtc4", 3: "pvrtc4", 6: "etc1", 7: "bc1", 8: "bc2", 9: "bc2",
	10: "bc3", 11: "bc3", 12: "bc4", 13: "bc5", 14: "bc6h", 15: "bc7", 22: "etc2", 23: "etc2_eac", 24: "etc2_a1",
	25: "r11", 26: "rg11"}
for _i, _d in enumerate(_ASTC_DIMS):
	_PVR3_COMPRESSED[27 + _i] = "astc" + _d


def _decode(kind, data: bytes, w: int, h: int, what: str) -> "np.ndarray":
	"""One level of any described format → RGBA uint8."""
	if isinstance(kind, str):
		need = tc.compressed_size(kind, w, h)
		if len(data) < need:
			raise ValueError(f"{what}: truncated {kind} level ({len(data)} of {need} bytes)")
		return tc.decode_blocks(kind, data[:need], w, h)
	tag = kind[0]
	if tag == "bytes":
		return tc.unpack_bytes(data, w, h, kind[1])
	if tag == "packed":
		return tc.unpack_packed(data, w, h, kind[1], kind[2])
	if tag == "float":
		return tc.float_to_rgba8(tc.unpack_float(data, w, h, kind[1], kind[2]))
	raise ValueError(f"{what}: unsupported format")


# -- KTX 1 ---------------------------------------------------------------------------------

KTX1_MAGIC = b"\xabKTX 11\xbb\r\n\x1a\n"
KTX2_MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"


def read_ktx1(data: bytes) -> "np.ndarray":
	if data[:12] != KTX1_MAGIC:
		raise ValueError("not a KTX 1 file")
	end = "<" if struct.unpack_from("<I", data, 12)[0] == 0x04030201 else ">"
	(gl_type, _type_size, gl_format, internal, _base, w, h, _depth, _arr, _faces, _mips, kv) = struct.unpack_from(
		end + "12I", data, 16)
	h = max(h, 1)
	pos = 64 + kv
	size = struct.unpack_from(end + "I", data, pos)[0]
	level = data[pos + 4: pos + 4 + size]
	if gl_type == 0:	# compressed
		kind = _GL_COMPRESSED.get(internal)
		if kind is None:
			raise ValueError(f"KTX: unsupported compressed format 0x{internal:04X}")
		return _decode(kind, level, w, h, "KTX")
	if gl_type in _GL_PACKED:
		fields, _ = _GL_PACKED[gl_type]
		if end == ">":
			return tc.unpack_packed(level, w, h, fields, 16, big=True)
		return _decode(("packed", fields, 16), level, w, h, "KTX")
	order = _GL_BASE.get(gl_format)
	if order is None:
		raise ValueError(f"KTX: unsupported glFormat 0x{gl_format:04X}")
	row = w * len(order) * {0x1401: 1, 0x140B: 2, 0x1406: 4}.get(gl_type, 1)
	stride = -(-row // 4) * 4	# KTX 1 rows are padded to 4 bytes
	if stride != row:
		level = b"".join(level[y * stride: y * stride + row] for y in range(h))
	if gl_type == 0x1401:
		return tc.unpack_bytes(level, w, h, order)
	if gl_type in (0x140B, 0x1406):
		return tc.float_to_rgba8(tc.unpack_float(level, w, h, order, end + ("f2" if gl_type == 0x140B else "f4")))
	raise ValueError(f"KTX: unsupported glType 0x{gl_type:04X}")


# -- KTX 2 ---------------------------------------------------------------------------------


def _basis_error(what: str) -> Exception:
	return tc.MissingDependency(f"{what} (Basis Universal)", "basisu transcoder")


def read_ktx2(data: bytes) -> "np.ndarray":
	if data[:12] != KTX2_MAGIC:
		raise ValueError("not a KTX 2 file")
	vk, _ts, w, h, _d, _layers, _faces, levels, scheme = struct.unpack_from("<9I", data, 12)
	h = max(h, 1)
	off, length, raw_len = struct.unpack_from("<3Q", data, 80)	# level 0 is first in the index
	level = data[off: off + length]
	if scheme == 1 or (vk == 0 and scheme != 2 and scheme != 3):
		raise _basis_error("KTX 2 BasisLZ/UASTC textures")
	if scheme == 2:
		from .valvecomp import zstd

		level = zstd(level, raw_len)
	elif scheme == 3:
		import zlib

		level = zlib.decompress(level)
	elif scheme:
		raise ValueError(f"KTX 2: unknown supercompression {scheme}")
	if vk == 0:
		raise _basis_error("KTX 2 UASTC textures")
	kind = _VK.get(vk)
	if kind is None:
		raise ValueError(f"KTX 2: unsupported VkFormat {vk}")
	return _decode(kind, level, w, h, "KTX 2")


# -- PowerVR -------------------------------------------------------------------------------


def _pvr3_uncompressed(fmt: int) -> object:
	"""PVR3 channel-order pixel formats: low dword names, high dword bit counts."""
	names = bytes((fmt >> (8 * i)) & 0xFF for i in range(4)).rstrip(b"\0").decode("latin-1").upper()
	bits = [(fmt >> (32 + 8 * i)) & 0xFF for i in range(len(names))]
	names = names.replace("I", "L")
	if all(b == 8 for b in bits):
		return ("bytes", names)
	total = sum(bits)
	if total in (8, 16, 32) and all(c in "RGBAL" for c in names):
		# little-endian words, the first named channel in the most significant bits (like GL 4_4_4_4)
		fields = "".join(f"{c}{b}" for c, b in zip(names, bits))
		return ("packed", fields, total)
	raise ValueError(f"PVR: unsupported channel layout {names} {bits}")


def read_pvr3(data: bytes) -> "np.ndarray":
	version, _flags, fmt, _cs, ctype, h, w, _d, _surf, _faces, _mips, meta = struct.unpack_from("<IIQIIIIIIIII", data, 0)
	if version != 0x03525650:
		raise ValueError("not a PVR v3 file")
	body = data[52 + meta:]
	if fmt >> 32 == 0:
		kind = _PVR3_COMPRESSED.get(fmt)
		if kind is None:
			raise ValueError(f"PVR: unsupported compressed format {fmt}")
	else:
		names = bytes((fmt >> (8 * i)) & 0xFF for i in range(4)).rstrip(b"\0").decode("latin-1").upper()
		bits = {(fmt >> (32 + 8 * i)) & 0xFF for i in range(len(names))}
		if ctype in (12, 13) and bits in ({16}, {32}):	# signed / unsigned float channels
			kind = ("float", names.replace("I", "L"), "<f2" if bits == {16} else "<f4")
		else:
			kind = _pvr3_uncompressed(fmt)
	return _decode(kind, body, w, h, "PVR")


_PVR2 = {0x10: ("packed", "A4R4G4B4", 16), 0x11: ("packed", "A1R5G5B5", 16), 0x12: ("packed", "R5G6B5", 16),
	0x13: ("packed", "X1R5G5B5", 16), 0x14: ("bytes", "RGB"), 0x15: ("bytes", "RGBA"), 0x16: ("bytes", "L"),
	0x17: ("bytes", "LA"), 0x18: "pvrtc2", 0x19: "pvrtc4", 0x1A: ("bytes", "BGRA"), 0x20: "bc1", 0x22: "bc2",
	0x24: "bc3", 0x36: "etc1", 0x00: ("packed", "A4R4G4B4", 16), 0x01: ("packed", "A1R5G5B5", 16),
	0x02: ("packed", "R5G6B5", 16), 0x0D: "pvrtc2", 0x0E: "pvrtc4"}


def read_pvr2(data: bytes) -> "np.ndarray":
	hsize, h, w, _mips, flags = struct.unpack_from("<5I", data, 0)
	if data[44:48] != b"PVR!":
		raise ValueError("not a legacy PVR file")
	kind = _PVR2.get(flags & 0xFF)
	if kind is None:
		raise ValueError(f"PVR v2: unsupported pixel type 0x{flags & 0xFF:02X}")
	out = _decode(kind, data[hsize:], w, h, "PVR")
	return out[::-1].copy() if flags & 0x10000 else out	# PVRTEX_VERTICAL_FLIP


def read_pvr(data: bytes) -> "np.ndarray":
	"""PowerVR v3/v2, or a Sega Dreamcast PVR (GBIX/PVRT), which shares the extension."""
	if data[:4] in (b"GBIX", b"PVRT"):
		from .consoletex import read_dreamcast_pvr

		return read_dreamcast_pvr(data)
	if data[:4] == b"PVR\x03":
		return read_pvr3(data)
	if data[44:48] == b"PVR!":
		return read_pvr2(data)
	raise ValueError("not a PowerVR or Dreamcast .pvr")


# -- ASTC and Basis ------------------------------------------------------------------------------


def read_astc(data: bytes) -> "np.ndarray":
	if struct.unpack_from("<I", data, 0)[0] != 0x5CA1AB13:
		raise ValueError("not an .astc file")
	bw, bh, bd = data[4], data[5], data[6]
	w = int.from_bytes(data[7:10], "little")
	h = int.from_bytes(data[10:13], "little")
	if bd != 1:
		raise ValueError("3D ASTC textures are not supported")
	return tc.decode_blocks(f"astc{bw}x{bh}", data[16:], w, h)


def read_basis(data: bytes) -> "np.ndarray":
	raise _basis_error(".basis textures")


def load(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	data = Path(path).read_bytes()
	ext = Path(path).suffix.lower()
	if data[:12] == KTX1_MAGIC:
		return read_ktx1(data)
	if data[:12] == KTX2_MAGIC:
		return read_ktx2(data)
	if ext == ".pvr":
		return read_pvr(data)
	if ext == ".astc":
		return read_astc(data)
	if ext == ".basis":
		return read_basis(data)
	raise ValueError(f"unrecognised {ext} texture")


# -- DDS fallback ----------------------------------------------------------------------------------
# Pillow reads most DDS files; this covers the DX10 formats it refuses (float, 16-bit, ETC/ASTC-in-DDS
# from some tools) for the top level of the first surface.

_DXGI: dict[int, object] = {
	2: ("float", "RGBA", "<f4"), 6: ("float", "RGB", "<f4"), 10: ("float", "RGBA", "<f2"), 16: ("float", "RG", "<f4"),
	34: ("float", "RG", "<f2"), 41: ("float", "R", "<f4"), 54: ("float", "R", "<f2"),
	28: ("bytes", "RGBA"), 29: ("bytes", "RGBA"), 87: ("bytes", "BGRA"), 91: ("bytes", "BGRA"),
	88: ("bytes", "BGRX"), 93: ("bytes", "BGRX"), 49: ("bytes", "RG"), 61: ("bytes", "R"), 65: ("bytes", "A"),
	85: ("packed", "R5G6B5", 16), 86: ("packed", "A1R5G5B5", 16), 115: ("packed", "A4R4G4B4", 16),
	71: "bc1", 72: "bc1", 74: "bc2", 75: "bc2", 77: "bc3", 78: "bc3", 80: "bc4", 81: "bc4", 83: "bc5", 84: "bc5",
	95: "bc6h", 96: "bc6h_sf", 98: "bc7", 99: "bc7",
}
for _i, _d in enumerate(_ASTC_DIMS):	# DXGI_FORMAT_ASTC_* (133 + 4·i, sRGB +1)
	_DXGI[134 + 4 * _i] = _DXGI[135 + 4 * _i] = "astc" + _d
_DDS_FOURCC = {b"DXT1": "bc1", b"DXT2": "bc2", b"DXT3": "bc2", b"DXT4": "bc3", b"DXT5": "bc3", b"ATI1": "bc4",
	b"BC4U": "bc4", b"ATI2": "bc5", b"BC5U": "bc5", b"ETC ": "etc1", b"ETC1": "etc1", b"ETC2": "etc2",
	b"ET2A": "etc2_eac", b"PTC2": "pvrtc2", b"PTC4": "pvrtc4"}
_DDS_D3DFMT = {113: ("float", "RGBA", "<f2"), 116: ("float", "RGBA", "<f4"), 111: ("float", "R", "<f2"),
	114: ("float", "R", "<f4"), 112: ("float", "RG", "<f2"), 115: ("float", "RG", "<f4")}


def read_dds(data: bytes) -> "np.ndarray":
	if data[:4] != b"DDS ":
		raise ValueError("not a DDS file")
	h, w = struct.unpack_from("<II", data, 12)
	fourcc = data[84:88]
	pos = 128
	if fourcc == b"DX10":
		dxgi = struct.unpack_from("<I", data, 128)[0]
		pos = 148
		kind = _DXGI.get(dxgi)
		if kind is None:
			raise ValueError(f"DDS: unsupported DXGI format {dxgi}")
	elif fourcc in _DDS_FOURCC:
		kind = _DDS_FOURCC[fourcc]
	elif struct.unpack_from("<I", fourcc)[0] in _DDS_D3DFMT:
		kind = _DDS_D3DFMT[struct.unpack_from("<I", fourcc)[0]]
	else:
		raise ValueError(f"DDS: unsupported pixel format {fourcc!r}")
	return _decode(kind, data[pos:], w, h, "DDS")
