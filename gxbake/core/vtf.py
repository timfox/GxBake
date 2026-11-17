"""Valve Texture Format (.vtf) and material (.vmt) reading for Source content.

A VTF is decoded to RGBA (largest mip, first frame, first face) and then goes
through the same CMPR path as any loose image, so it ends up in a GCWT pack,
or as sky .cmp sides when six of them form a skybox. Source models find their
skins through the .vmt files named by the model (see source_mdl.py).

Block-compressed formats (DXT1/3/5, ATI1N/ATI2N) are decoded here with numpy,
so nothing beyond Pillow and numpy is needed.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

# IMAGE_FORMAT enum from the Source SDK (bitmap/imageformat.h).
FMT_NAMES = {
	0: "RGBA8888", 1: "ABGR8888", 2: "RGB888", 3: "BGR888", 4: "RGB565", 5: "I8", 6: "IA88", 7: "P8",
	8: "A8", 9: "RGB888_BLUESCREEN", 10: "BGR888_BLUESCREEN", 11: "ARGB8888", 12: "BGRA8888", 13: "DXT1",
	14: "DXT3", 15: "DXT5", 16: "BGRX8888", 17: "BGR565", 18: "BGRX5551", 19: "BGRA4444",
	20: "DXT1_ONEBITALPHA", 21: "BGRA5551", 22: "UV88", 23: "UVWQ8888", 24: "RGBA16161616F",
	25: "RGBA16161616", 26: "UVLX8888", 34: "ATI2N", 35: "ATI1N",
}
_BYTES_PER_PIXEL = {0: 4, 1: 4, 2: 3, 3: 3, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 3, 10: 3, 11: 4, 12: 4, 16: 4,
	17: 2, 18: 2, 19: 2, 21: 2, 22: 2, 23: 4, 24: 8, 25: 8, 26: 4}
_BLOCK_BYTES = {13: 8, 20: 8, 14: 16, 15: 16, 34: 16, 35: 8}

TEXTUREFLAGS_ENVMAP = 0x4000
TEXTUREFLAGS_ONEBITALPHA = 0x1000
TEXTUREFLAGS_EIGHTBITALPHA = 0x2000


def level_bytes(fmt: int, w: int, h: int) -> int:
	if fmt in _BLOCK_BYTES:
		return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * _BLOCK_BYTES[fmt]
	if fmt not in _BYTES_PER_PIXEL:
		raise ValueError(f"unsupported VTF format {FMT_NAMES.get(fmt, fmt)}")
	return w * h * _BYTES_PER_PIXEL[fmt]


def vtf_header(data: bytes) -> dict:
	if len(data) < 64 or data[:4] != b"VTF\0":
		raise ValueError("not a VTF file")
	major, minor, header_size = struct.unpack_from("<III", data, 4)
	w, h, flags, frames, first_frame = struct.unpack_from("<HHIHH", data, 16)
	reflectivity = struct.unpack_from("<3f", data, 32)
	hi_fmt = struct.unpack_from("<i", data, 52)[0]
	mips = data[56]
	lo_fmt = struct.unpack_from("<i", data, 57)[0]
	lo_w, lo_h = data[61], data[62]
	depth = struct.unpack_from("<H", data, 63)[0] if (major, minor) >= (7, 2) else 1
	hi_off = None
	if (major, minor) >= (7, 3):
		nres = struct.unpack_from("<I", data, 68)[0]
		for i in range(min(nres, 32)):
			tag, rflags, off = struct.unpack_from("<3sBI", data, 80 + i * 8)
			if tag == b"\x30\0\0":
				hi_off = off
	if hi_off is None:
		lo = level_bytes(lo_fmt, lo_w, lo_h) if lo_fmt >= 0 and lo_w and lo_h else 0
		hi_off = header_size + lo
	faces = 1
	if flags & TEXTUREFLAGS_ENVMAP:
		faces = 7 if (major, minor) < (7, 5) and first_frame != 0xFFFF and minor >= 1 else 6
	return {"version": f"{major}.{minor}", "width": w, "height": h, "flags": flags, "frames": max(1, frames),
		"format": hi_fmt, "format_name": FMT_NAMES.get(hi_fmt, str(hi_fmt)), "mips": max(1, mips),
		"depth": max(1, depth), "faces": faces, "data_offset": hi_off, "reflectivity": reflectivity}


def read_vtf(data: bytes, frame: int = 0, face: int = 0, mip: int = 0) -> "np.ndarray":
	"""RGBA (H×W×4 uint8) of one mip/frame/face; mip 0 is the largest."""
	hd = vtf_header(data)
	fmt, w, h = hd["format"], hd["width"], hd["height"]
	frames, faces, mips = hd["frames"], hd["faces"], hd["mips"]
	frame, face, mip = min(frame, frames - 1), min(face, faces - 1), min(mip, mips - 1)

	def dims(level: int) -> tuple[int, int, int]:
		return max(1, w >> level), max(1, h >> level), max(1, hd["depth"] >> level)

	# Mips are stored smallest first; each holds frames × faces × slices.
	off = hd["data_offset"]
	for level in range(mips - 1, mip, -1):
		lw, lh, ld = dims(level)
		off += level_bytes(fmt, lw, lh) * frames * faces * ld
	lw, lh, ld = dims(mip)
	size = level_bytes(fmt, lw, lh)
	off += size * ld * (frame * faces + face)
	if off + size > len(data):
		raise ValueError("VTF is truncated")
	return decode_pixels(fmt, data[off:off + size], lw, lh)


def decode_pixels(fmt: int, raw: bytes, w: int, h: int) -> "np.ndarray":
	if fmt in (13, 20):
		return decode_dxt1(raw, w, h)
	if fmt == 14:
		return decode_dxt3(raw, w, h)
	if fmt == 15:
		return decode_dxt5(raw, w, h)
	if fmt == 35:
		a = _bc4(np.frombuffer(raw, np.uint8).reshape(-1, 8), w, h)
		return np.dstack([a, a, a, np.full_like(a, 255)])
	if fmt == 34:
		blocks = np.frombuffer(raw, np.uint8).reshape(-1, 16)
		x = _bc4(blocks[:, :8], w, h).astype(np.float32) / 127.5 - 1
		y = _bc4(blocks[:, 8:], w, h).astype(np.float32) / 127.5 - 1
		z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
		rgb = np.dstack([x, y, z]) * 127.5 + 127.5
		return np.dstack([rgb.clip(0, 255).astype(np.uint8), np.full((h, w), 255, np.uint8)])
	n = w * h
	b = np.frombuffer(raw, np.uint8)
	out = np.empty((n, 4), np.uint8)
	out[:, 3] = 255
	if fmt in (0, 1, 11, 12, 16, 23, 26):
		px = b[:n * 4].reshape(n, 4)
		order = {0: (0, 1, 2, 3), 1: (3, 2, 1, 0), 11: (1, 2, 3, 0), 12: (2, 1, 0, 3), 16: (2, 1, 0, 3),
			23: (0, 1, 2, 3), 26: (0, 1, 2, 3)}[fmt]
		out[:] = px[:, order]
		if fmt in (16, 26):
			out[:, 3] = 255
	elif fmt in (2, 3, 9, 10):
		px = b[:n * 3].reshape(n, 3)
		out[:, :3] = px if fmt in (2, 9) else px[:, ::-1]
		if fmt in (9, 10):	# pure blue is the transparency key
			out[:, 3] = np.where((out[:, 0] == 0) & (out[:, 1] == 0) & (out[:, 2] == 255), 0, 255)
	elif fmt in (4, 17, 18, 19, 21):
		v = b[:n * 2].view("<u2").astype(np.uint32)
		if fmt == 4:	# RGB565 with red in the low bits, as Valve stores it
			r, g, bl = v & 31, (v >> 5) & 63, v >> 11
			out[:, 0], out[:, 1], out[:, 2] = r * 255 // 31, g * 255 // 63, bl * 255 // 31
		elif fmt == 17:
			bl, g, r = v & 31, (v >> 5) & 63, v >> 11
			out[:, 0], out[:, 1], out[:, 2] = r * 255 // 31, g * 255 // 63, bl * 255 // 31
		elif fmt == 19:
			bl, g, r, a = v & 15, (v >> 4) & 15, (v >> 8) & 15, v >> 12
			out[:, 0], out[:, 1], out[:, 2], out[:, 3] = r * 17, g * 17, bl * 17, a * 17
		else:
			bl, g, r, a = v & 31, (v >> 5) & 31, (v >> 10) & 31, v >> 15
			out[:, 0], out[:, 1], out[:, 2] = r * 255 // 31, g * 255 // 31, bl * 255 // 31
			out[:, 3] = 255 if fmt == 18 else a * 255
	elif fmt in (5, 7):
		out[:, 0] = out[:, 1] = out[:, 2] = b[:n]
	elif fmt == 6:
		px = b[:n * 2].reshape(n, 2)
		out[:, 0] = out[:, 1] = out[:, 2] = px[:, 0]
		out[:, 3] = px[:, 1]
	elif fmt == 8:
		out[:, :3] = 255
		out[:, 3] = b[:n]
	elif fmt == 22:
		px = b[:n * 2].reshape(n, 2)
		out[:, :2] = px
		out[:, 2] = 255
	elif fmt in (24, 25):
		if fmt == 24:
			f = b[:n * 8].view("<f2").astype(np.float32).reshape(n, 4)
			f[:, :3] = f[:, :3] / (1 + f[:, :3])	# HDR → LDR (Reinhard)
		else:
			f = b[:n * 8].view("<u2").astype(np.float32).reshape(n, 4) / 65535.0
		out[:] = (f.clip(0, 1) * 255 + 0.5).astype(np.uint8)
	else:
		raise ValueError(f"unsupported VTF format {FMT_NAMES.get(fmt, fmt)}")
	return out.reshape(h, w, 4)


# -- block compression ------------------------------------------------------------


def _blocks_to_image(texels: "np.ndarray", w: int, h: int) -> "np.ndarray":
	"""(nblocks, 4, 4, C) in row-major block order → (h, w, C), cropped."""
	bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
	c = texels.shape[-1]
	img = texels[: bw * bh].reshape(bh, bw, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, c)
	return np.ascontiguousarray(img[:h, :w])


def _565(v: "np.ndarray") -> "np.ndarray":
	v = v.astype(np.int32)
	r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
	return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)


def _colour_block(blocks: "np.ndarray", four_colour: bool) -> tuple["np.ndarray", "np.ndarray"]:
	"""8-byte colour halves → ((n, 16, 3) RGB, (n, 16) punch-through mask)."""
	c0 = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
	c1 = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
	p0, p1 = _565(c0), _565(c1)
	opaque4 = four_colour | (c0 > c1)
	sel = opaque4[:, None]
	p2 = np.where(sel, (2 * p0 + p1) // 3, (p0 + p1) // 2)
	p3 = np.where(sel, (p0 + 2 * p1) // 3, 0)
	pal = np.stack([p0, p1, p2, p3], 1)	# (n, 4, 3)
	bits = blocks[:, 4:8].astype(np.uint32)
	word = bits[:, 0] | (bits[:, 1] << 8) | (bits[:, 2] << 16) | (bits[:, 3] << 24)
	idx = (word[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
	rgb = np.take_along_axis(pal, idx[..., None].astype(np.int64), 1)
	hole = (idx == 3) & ~opaque4[:, None]
	return rgb.astype(np.uint8), hole


def decode_dxt1(raw: bytes, w: int, h: int) -> "np.ndarray":
	blocks = np.frombuffer(raw, np.uint8).reshape(-1, 8)
	rgb, hole = _colour_block(blocks, False)
	a = np.where(hole, 0, 255).astype(np.uint8)
	return _blocks_to_image(np.dstack([rgb, a[..., None]]).reshape(-1, 4, 4, 4), w, h)


def decode_dxt3(raw: bytes, w: int, h: int) -> "np.ndarray":
	blocks = np.frombuffer(raw, np.uint8).reshape(-1, 16)
	rgb, _ = _colour_block(blocks[:, 8:], True)
	nib = np.stack([blocks[:, :8] & 15, blocks[:, :8] >> 4], -1).reshape(-1, 16)
	a = (nib * 17).astype(np.uint8)
	return _blocks_to_image(np.dstack([rgb, a[..., None]]).reshape(-1, 4, 4, 4), w, h)


def _bc4_values(blocks: "np.ndarray") -> "np.ndarray":
	"""8-byte BC4/DXT5-alpha blocks → (n, 16) uint8."""
	a0 = blocks[:, 0].astype(np.int32)
	a1 = blocks[:, 1].astype(np.int32)
	k = np.arange(1, 7)
	eight = (a0 > a1)[:, None]
	interp8 = ((7 - k) * a0[:, None] + k * a1[:, None]) // 7
	k5 = np.arange(1, 5)
	interp6 = ((5 - k5) * a0[:, None] + k5 * a1[:, None]) // 5
	six = np.concatenate([interp6, np.zeros((len(a0), 1), np.int32), np.full((len(a0), 1), 255, np.int32)], 1)
	pal = np.concatenate([a0[:, None], a1[:, None], np.where(eight, interp8, six)], 1)
	bits = np.zeros(len(blocks), np.uint64)
	for i in range(6):
		bits |= blocks[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
	idx = (bits[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))) & np.uint64(7)
	return np.take_along_axis(pal, idx.astype(np.int64), 1).astype(np.uint8)


def _bc4(blocks: "np.ndarray", w: int, h: int) -> "np.ndarray":
	return _blocks_to_image(_bc4_values(blocks).reshape(-1, 4, 4, 1), w, h)[..., 0]


def decode_dxt5(raw: bytes, w: int, h: int) -> "np.ndarray":
	blocks = np.frombuffer(raw, np.uint8).reshape(-1, 16)
	rgb, _ = _colour_block(blocks[:, 8:], True)
	a = _bc4_values(blocks[:, :8])
	return _blocks_to_image(np.dstack([rgb, a[..., None]]).reshape(-1, 4, 4, 4), w, h)


def load_vtf(path: Path) -> "np.ndarray":
	return read_vtf(Path(path).read_bytes())


def vtf_info(data: bytes) -> dict:
	hd = vtf_header(data)
	return {"VTF version": hd["version"], "Size": f"{hd['width']}x{hd['height']}", "Format": hd["format_name"],
		"Mip levels": hd["mips"], "Frames": hd["frames"], "Faces": hd["faces"]}


# -- materials ----------------------------------------------------------------------


def parse_vmt(text: str) -> tuple[str, dict[str, str]]:
	"""(shader, {lowercased key: value}) of a .vmt's top-level block (nested blocks flattened)."""
	text = re.sub(r"//[^\n]*", "", text)
	tokens = re.findall(r'"([^"]*)"|([{}])|([^\s{}"]+)', text)
	flat = [a or b or c for a, b, c in tokens]
	shader = flat[0].lower() if flat else ""
	params: dict[str, str] = {}
	i = 1
	while i < len(flat):
		tok = flat[i]
		if tok in "{}":
			i += 1
			continue
		if i + 1 < len(flat) and flat[i + 1] not in "{}":
			params.setdefault(tok.lower(), flat[i + 1])
			i += 2
		else:
			i += 1
	return shader, params


def find_materials_root(start: Path) -> Path | None:
	"""The nearest ancestor holding a materials/ folder (the Source gamedir)."""
	for d in [start, *start.parents]:
		try:
			if (d / "materials").is_dir():
				return d / "materials"
		except OSError:
			continue
	return None


def _ci_file(root: Path, rel: str) -> Path | None:
	"""Case-insensitive lookup of root/rel (Source paths are case-insensitive)."""
	p = root / rel
	if p.is_file():
		return p
	cur = root
	for part in Path(rel).parts:
		try:
			match = next((c for c in cur.iterdir() if c.name.lower() == part.lower()), None)
		except OSError:
			return None
		if match is None:
			return None
		cur = match
	return cur if cur.is_file() else None


def resolve_material(name: str, search_dirs: list[str], roots: list[Path]) -> tuple[Path | None, dict[str, str]]:
	"""Find <dir>/<name>.vmt under a materials root; return (base texture .vtf path, vmt params)."""
	name = name.replace("\\", "/").strip("/")
	for root in roots:
		for d in search_dirs or [""]:
			rel = (d.replace("\\", "/").strip("/") + "/" + name).strip("/")
			vmt = _ci_file(root, rel + ".vmt")
			params: dict[str, str] = {}
			if vmt is not None:
				try:
					shader, params = parse_vmt(vmt.read_text("latin-1"))
				except OSError:
					params = {}
				if shader == "patch" and "include" in params:
					inc = params["include"].replace("\\", "/")
					inc = inc[len("materials/"):] if inc.lower().startswith("materials/") else inc
					inc_vmt = _ci_file(root, inc)
					if inc_vmt is not None:
						_s, base = parse_vmt(inc_vmt.read_text("latin-1"))
						params = {**base, **params}
				base_tex = params.get("$basetexture")
				if base_tex:
					tex = _ci_file(root, base_tex.replace("\\", "/").strip("/").removesuffix(".vtf") + ".vtf")
					if tex is not None:
						return tex, params
			tex = _ci_file(root, rel + ".vtf")
			if tex is not None:
				return tex, params
	return None, {}
