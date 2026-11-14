"""GoldSrc container readers: WAD2/WAD3 miptex, BSP v30 texture/entity lumps.

wt_read_miptex, wt_bsp_wad_basenames, wt_ingest_wad_file, wad_lump and
map_sky_name are ported from scripts/build-gamecube-disc.py. The list/inspect
helpers are new and feed the GUI asset tree.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def wt_read_miptex(data: bytes, offset: int = 0):
	"""A WAD3/BSP miptex: (name, w, h, indexed pixels, 768-byte palette), or None."""
	try:
		raw_name, w, h, m0 = struct.unpack_from("<16sIII", data, offset)
	except struct.error:
		return None
	if not m0 or not w or not h or w * h > 1 << 20:
		return None
	name = raw_name.split(b"\0")[0].decode("latin-1").lower()
	end = offset + m0 + w * h + (w * h) // 4 + (w * h) // 16 + (w * h) // 64
	if end + 2 > len(data):
		return None
	count = struct.unpack_from("<H", data, end)[0]
	if count != 256 or end + 2 + 768 > len(data):
		return None
	return name, w, h, data[offset + m0:offset + m0 + w * h], data[end + 2:end + 2 + 768]


def miptex_rgba(w: int, h: int, pixels: bytes, palette: bytes, alpha: bool) -> "np.ndarray":
	"""Indexed miptex to H×W×4 RGBA (index 255 transparent for '{' textures)."""
	pal = np.frombuffer(palette, np.uint8).reshape(256, 3)
	idx = np.frombuffer(pixels, np.uint8).reshape(h, w)
	rgba = np.empty((h, w, 4), np.uint8)
	rgba[..., :3] = pal[idx]
	rgba[..., 3] = np.where((idx == 255) & alpha, 0, 255)
	return rgba


@dataclass
class WadEntry:
	name: str
	kind: int
	offset: int
	disk_size: int
	size: int
	compressed: bool


def wad_entries(data: bytes) -> list[WadEntry]:
	if data[:4] not in (b"WAD3", b"WAD2"):
		raise ValueError("not a WAD2/WAD3 file")
	count, table = struct.unpack_from("<ii", data, 4)
	out = []
	for i in range(count):
		pos, disk, size, kind, comp, _pad, raw = struct.unpack_from("<iiibbh16s", data, table + i * 32)
		name = raw.split(b"\0")[0].decode("latin-1").lower()
		out.append(WadEntry(name, kind & 0xFF, pos, disk, size, bool(comp)))
	return out


def wt_ingest_wad_file(found: dict, wad: Path) -> int:
	"""Index every miptex (type 0x43) from one WAD; first entry wins. Returns new count."""
	try:
		data = wad.read_bytes()
	except OSError:
		return 0
	if data[:4] not in (b"WAD3", b"WAD2"):
		return 0
	count, table = struct.unpack_from("<ii", data, 4)
	added = 0
	for i in range(count):
		pos, disk, size, kind, comp, _pad, raw = struct.unpack_from("<iiibbh16s", data, table + i * 32)
		if kind != 0x43 or comp:
			continue
		name = raw.split(b"\0")[0].decode("latin-1").lower()
		if name not in found:
			found[name] = (data, pos)
			added += 1
	return added


def wad_miptex(path: Path) -> dict:
	"""name -> (data, offset) for every miptex in one WAD."""
	found: dict = {}
	wt_ingest_wad_file(found, path)
	return found


def wad_lump(wad: Path, name: str) -> bytes | None:
	"""Raw lump bytes from a WAD3 file, or None."""
	try:
		data = wad.read_bytes()
	except OSError:
		return None
	if data[:4] != b"WAD3":
		return None
	count, offset = struct.unpack_from("<ii", data, 4)
	for i in range(count):
		filepos, disksize, size, ltype, comp = struct.unpack_from("<iiibb", data, offset + i * 32)
		lname = data[offset + i * 32 + 16: offset + i * 32 + 32].split(b"\0")[0].decode("latin1")
		if lname.upper() == name.upper():
			return data[filepos: filepos + disksize]
	return None


# --- BSP v30 -----------------------------------------------------------------

BSP_LUMP_ENTITIES = 0
BSP_LUMP_TEXTURES = 2


def bsp_lump(data: bytes, index: int) -> tuple[int, int]:
	return struct.unpack_from("<ii", data, 4 + index * 8)


def bsp_entities_text(data: bytes, limit: int | None = None) -> str:
	off, length = bsp_lump(data, BSP_LUMP_ENTITIES)
	if limit is not None:
		length = min(length, limit)
	return data[off:off + length].decode("latin-1", "replace")


def wt_bsp_wad_basenames(bsp: Path | bytes) -> list[str]:
	"""Basenames from the BSP worldspawn "wad" key (order preserved)."""
	try:
		data = bsp if isinstance(bsp, (bytes, bytearray)) else bsp.read_bytes()
		ents = bsp_entities_text(data, 8192)
	except (OSError, struct.error):
		return []
	m = re.search(r'"wad"\s+"([^"]+)"', ents)
	if not m:
		return []
	names: list[str] = []
	seen: set[str] = set()
	for part in m.group(1).split(";"):
		part = part.strip().replace("\\", "/")
		if not part:
			continue
		name = part.split("/")[-1]
		key = name.lower()
		if not key.endswith(".wad") or key in seen:
			continue
		seen.add(key)
		names.append(name)
	return names


def map_sky_name(data: bytes) -> str | None:
	"""Worldspawn "skyname", or None."""
	try:
		text = bsp_entities_text(data, 4096)
	except struct.error:
		return None
	m = re.search(r'"skyname"\s+"([^"]+)"', text.split("}", 1)[0])
	return m.group(1).lower() if m else None


def bsp_miptex_offsets(data: bytes) -> list[tuple[str, int]]:
	"""(name, absolute offset) for each texture in the BSP texture lump, in order."""
	off, _size = bsp_lump(data, BSP_LUMP_TEXTURES)
	count = struct.unpack_from("<i", data, off)[0]
	out = []
	for o in struct.unpack_from(f"<{count}i", data, off + 4):
		if o < 0:
			continue
		raw = struct.unpack_from("<16s", data, off + o)[0]
		out.append((raw.split(b"\0")[0].decode("latin-1").lower(), off + o))
	return out


def is_world_diffuse(name: str) -> bool:
	"""Liquids ('!', '*') and sky keep the engine's own path, as in the disc builder."""
	return bool(name) and name[:1] not in ("!", "*") and not name.startswith("sky")
