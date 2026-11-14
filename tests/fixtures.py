"""Builders for tiny synthetic GoldSrc files, so tests need no game data."""

from __future__ import annotations

import struct

import numpy as np


def palette(seed: int = 0) -> bytes:
	rng = np.random.default_rng(seed)
	pal = rng.integers(0, 256, (256, 3), dtype=np.uint8)
	pal[255] = (0, 0, 255)
	return pal.tobytes()


def miptex(name: str, w: int, h: int, seed: int = 0, embed: bool = True) -> bytes:
	rng = np.random.default_rng(seed)
	head = struct.pack("<16sII", name.encode()[:15], w, h)
	if not embed:
		return head + struct.pack("<4I", 0, 0, 0, 0)
	pix = rng.integers(0, 255, w * h, dtype=np.uint8)
	if name.startswith("{"):
		pix[: w * h // 4] = 255
	mips = [pix.tobytes()] + [bytes((w >> i) * (h >> i)) for i in (1, 2, 3)]
	offs, cur = [], 40
	for m in mips:
		offs.append(cur)
		cur += len(m)
	return head + struct.pack("<4I", *offs) + b"".join(mips) + struct.pack("<H", 256) + palette(seed) + b"\0\0"


def wad3(textures: dict[str, bytes]) -> bytes:
	body = bytearray(b"WAD3" + struct.pack("<ii", 0, 0))
	entries = []
	for name, data in textures.items():
		entries.append((len(body), len(data), name))
		body += data
	table = len(body)
	for pos, size, name in entries:
		body += struct.pack("<iiibbh16s", pos, size, size, 0x43, 0, 0, name.encode()[:15])
	struct.pack_into("<ii", body, 4, len(entries), table)
	return bytes(body)


def bsp30(entities: str, textures: list[bytes]) -> bytes:
	lumps = [b""] * 15
	lumps[0] = entities.encode() + b"\0"
	tex = bytearray(struct.pack("<i", len(textures)))
	offs, cur = [], 4 + 4 * len(textures)
	for t in textures:
		offs.append(cur)
		cur += len(t)
	tex += struct.pack(f"<{len(offs)}i", *offs) + b"".join(textures)
	lumps[2] = bytes(tex)
	out = bytearray(struct.pack("<i", 30) + bytes(15 * 8))
	for i, data in enumerate(lumps):
		struct.pack_into("<ii", out, 4 + i * 8, len(out), len(data))
		out += data
		out += bytes(-len(out) & 3)
	return bytes(out)


def studio(textures: list[tuple[str, int, int, int]], mesh_bytes: int = 400) -> bytes:
	"""IDST v10 with `mesh_bytes` of fake geometry then embedded skins."""
	hdr = bytearray(244)
	hdr[0:4] = b"IDST"
	struct.pack_into("<i", hdr, 4, 10)
	hdr[8:72] = b"models/test.mdl".ljust(64, b"\0")
	body = bytearray(hdr) + bytes(range(256)) * (mesh_bytes // 256) + bytes(mesh_bytes % 256)
	texindex = len(body)
	body += bytes(80 * len(textures))
	skinindex = len(body)
	body += struct.pack(f"<{len(textures)}h", *range(len(textures)))
	for i, (name, flags, w, h) in enumerate(textures):
		index = len(body)
		body += np.random.default_rng(i).integers(0, 256, w * h, dtype=np.uint8).tobytes() + palette(i)
		off = texindex + i * 80
		body[off:off + 64] = name.encode().ljust(64, b"\0")
		struct.pack_into("<iiii", body, off + 64, flags, w, h, index)
	struct.pack_into("<i", body, 72, len(body))
	struct.pack_into("<i", body, 180, len(textures))
	struct.pack_into("<i", body, 184, texindex)
	struct.pack_into("<i", body, 188, texindex + 80 * len(textures))
	struct.pack_into("<i", body, 192, len(textures))
	struct.pack_into("<i", body, 196, 1)
	struct.pack_into("<i", body, 200, skinindex)
	return bytes(body)


def sprite(w: int = 32, h: int = 32, texformat: int = 3) -> bytes:
	pix = np.full((h, w), 255, np.uint8)
	pix[h // 2, :] = 7	# a one-texel horizontal stroke
	pix[:, w // 2] = 9
	out = b"IDSP" + struct.pack("<iii", 2, 2, texformat) + struct.pack("<f", 1.0)
	out += struct.pack("<ii", w, h) + struct.pack("<iii", 1, 0, 0) + struct.pack("<h", 256) + palette(3)
	out += struct.pack("<i", 0) + struct.pack("<iiii", -w // 2, h // 2, w, h) + pix.tobytes()
	return out
