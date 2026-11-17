"""GoldSrc sprites (.spr): lean HUD downsample and frame decode.

lean_hl_spr_bytes is ported unchanged from scripts/build-gamecube-disc.py
(G173/G174/G187 lean HUD). spr_frames is new and feeds the GUI preview.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SPR_TEXFORMATS = {0: "normal", 1: "additive", 2: "indexalpha", 3: "alphatest"}


def lean_hl_spr_bytes(src: Path, scale: int = 2) -> bytes | None:
	"""Nearest-neighbor downsample a single-frame HL SPR (G173/G174 lean HUD).

	G187: for SPR_ALPHATEST sheets, prefer any non-255 texel in the scale×scale
	block so thin crosshair strokes survive 128→64 (plain NN often empties them).
	"""
	try:
		d = src.read_bytes()
	except OSError:
		return None
	return lean_hl_spr_bytes_from(d, scale)


def lean_hl_spr_bytes_from(d: bytes, scale: int = 2) -> bytes | None:
	"""lean_hl_spr_bytes on sprite bytes (e.g. a Quake sprite converted to v2)."""
	import math

	if len(d) < 44 or d[:4] != b"IDSP":
		return None
	ver = struct.unpack_from("<i", d, 4)[0]
	if ver != 2:
		return None
	typ = struct.unpack_from("<i", d, 8)[0]
	tex = struct.unpack_from("<i", d, 12)[0]
	face = struct.unpack_from("<i", d, 32)[0]
	sync = struct.unpack_from("<i", d, 36)[0]
	ncolors = struct.unpack_from("<h", d, 40)[0]
	if ncolors <= 0 or ncolors > 256:
		return None
	pal_off = 42
	frame_off = pal_off + ncolors * 3
	if frame_off + 20 > len(d):
		return None
	ftype = struct.unpack_from("<i", d, frame_off)[0]
	if ftype != 0:
		return None
	ox, oy, w, h = struct.unpack_from("<iiii", d, frame_off + 4)
	pix_off = frame_off + 20
	if w <= 0 or h <= 0 or pix_off + w * h > len(d):
		return None
	if scale < 1 or (w % scale) or (h % scale):
		return None
	nw, nh = w // scale, h // scale
	pix = d[pix_off : pix_off + w * h]
	out_pix = bytearray(nw * nh)
	# texFormat 3 = SPR_ALPHATEST — index 255 is transparent.
	preserve_ink = tex == 3
	for y in range(nh):
		sy = y * scale
		for x in range(nw):
			sx = x * scale
			sample = pix[sy * w + sx]
			if preserve_ink and sample == 255:
				found = 255
				for dy in range(scale):
					row = (sy + dy) * w
					for dx in range(scale):
						p = pix[row + sx + dx]
						if p != 255:
							found = p
							break
					if found != 255:
						break
				sample = found
			out_pix[y * nw + x] = sample
	br = math.sqrt((nw * 0.5) ** 2 + (nh * 0.5) ** 2)
	out = bytearray()
	out += b"IDSP"
	out += struct.pack("<iii", 2, typ, tex)
	out += struct.pack("<f", br)
	out += struct.pack("<ii", nw, nh)
	out += struct.pack("<iii", 1, face, sync)
	out += struct.pack("<h", ncolors)
	out += d[pal_off:frame_off]
	out += struct.pack("<i", 0)
	out += struct.pack("<iiii", ox // scale, oy // scale, nw, nh)
	out += out_pix
	return bytes(out)



def spr_info(d: bytes) -> dict:
	if len(d) < 42 or d[:4] != b"IDSP":
		raise ValueError("not a sprite")
	ver, typ, tex = struct.unpack_from("<iii", d, 4)
	w, h, frames = struct.unpack_from("<iii", d, 20)
	return {"version": ver, "type": typ, "texformat": SPR_TEXFORMATS.get(tex, str(tex)),
		"width": w, "height": h, "frames": frames}


def spr_frames(d: bytes, limit: int = 64) -> list["np.ndarray"]:
	"""RGBA for each single frame of a v2 (Half-Life) sprite; group frames skipped."""
	if len(d) < 44 or d[:4] != b"IDSP" or struct.unpack_from("<i", d, 4)[0] != 2:
		return []
	tex = struct.unpack_from("<i", d, 12)[0]
	nframes = struct.unpack_from("<i", d, 28)[0]
	ncolors = struct.unpack_from("<h", d, 40)[0]
	if ncolors <= 0 or ncolors > 256:
		return []
	pal = np.zeros((256, 3), np.uint8)
	pal[:ncolors] = np.frombuffer(d, np.uint8, ncolors * 3, 42).reshape(ncolors, 3)
	pos = 42 + ncolors * 3
	out = []
	for _ in range(min(nframes, limit)):
		if pos + 20 > len(d):
			break
		ftype = struct.unpack_from("<i", d, pos)[0]
		if ftype != 0:
			break
		_ox, _oy, w, h = struct.unpack_from("<iiii", d, pos + 4)
		pos += 20
		if w <= 0 or h <= 0 or pos + w * h > len(d):
			break
		idx = np.frombuffer(d, np.uint8, w * h, pos).reshape(h, w)
		pos += w * h
		rgba = np.empty((h, w, 4), np.uint8)
		rgba[..., :3] = pal[idx]
		if tex == 3:
			rgba[..., 3] = np.where(idx == 255, 0, 255)
		elif tex == 2:
			rgba[..., :3] = pal[ncolors - 1]
			rgba[..., 3] = idx
		elif tex == 1:
			rgba[..., 3] = rgba[..., :3].max(-1)
		else:
			rgba[..., 3] = 255
		out.append(rgba)
	return out
