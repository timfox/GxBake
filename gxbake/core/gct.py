"""GCWT texture packs (.gct): CMPR mip chains the engine reads in one go.

Layout (big-endian), as written by build_world_texture_packs in
scripts/build-gamecube-disc.py and read by engine/common/mod_bmodel.c:

	"GCWT" u32 count u32 blob_size
	count × entry { char name[16]; u16 w, h; u8 levels, flags; u16 tag; u32 offset }
	pad to 32 bytes, then the blob; every texture is 32-byte aligned in the blob.

wt_encode and the pack writer are ported from the disc builder, including the
cache key, so a GxBake cache directory is interchangeable with the builder's.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .cmpr import cmpr_decode, cmpr_refine, cmpr_size, gx_cmpr_encode_blocks
from .goldsrc import (
	bsp_miptex_offsets,
	is_world_diffuse,
	read_miptex,
	wt_bsp_wad_basenames,
	wt_ingest_wad_file,
	wt_read_miptex,
)

GC_WT_MAGIC = b"GCWT"
GC_WT_FLAG_ALPHA = 1
GC_WT_ENTRY = struct.Struct(">16sHHBBHI")	# name, w, h, levels, flags, content tag, offset


def _resize(arr: "np.ndarray", size: tuple[int, int], nearest: bool) -> "np.ndarray":
	from PIL import Image

	resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
	return np.asarray(Image.fromarray(arr).resize(size, resample))


def encode_mip_chain(
	rgb: "np.ndarray",
	opaque: "np.ndarray",
	*,
	mipmaps: bool = True,
	min_edge: int = 8,
	refine: bool = False,
) -> tuple[int, bytes]:
	"""CMPR levels for a power-of-two RGB image + coverage mask.

	This is the loop body of wt_encode: a coverage-weighted box filter so a
	transparent pixel's colour never bleeds into the edge of a cut-out.
	`refine` adds least-squares CMPR endpoints (cmpr_refine): same size,
	closer colours, different bytes from the disc builder's encoder.
	"""
	levels = bytearray()
	count = 0
	fcol = rgb.astype(np.float32)
	fcov = opaque.astype(np.float32)
	while True:
		px, op = np.clip(fcol + 0.5, 0, 255).astype(np.uint8), fcov >= 0.5
		lvl = gx_cmpr_encode_blocks(px, op)
		levels += cmpr_refine(px, op, lvl) if refine else lvl
		count += 1
		ch, cw = fcov.shape
		if not mipmaps or min(ch, cw) <= max(8, min_edge):
			break
		cover = fcov.reshape(ch // 2, 2, cw // 2, 2).sum((1, 3))
		sumc = (fcol * fcov[..., None]).reshape(ch // 2, 2, cw // 2, 2, 3).sum((1, 3))
		fcol = np.where(cover[..., None] > 0, sumc / np.maximum(cover[..., None], 1e-6), 0)
		fcov = cover / 4
	return count, bytes(levels)


def wt_encode(w: int, h: int, pixels: bytes, palette: bytes, alpha: bool,
		refine: bool = False) -> tuple[int, int, int, bytes]:
	"""Palette texture to a CMPR mip chain, power-of-two sides (GX_REPEAT
	needs them; UVs are normalised, so a resize does not move anything)."""
	pal = np.frombuffer(palette, np.uint8).reshape(256, 3)
	idx = np.frombuffer(pixels, np.uint8).reshape(h, w)
	rgb = pal[idx]
	opaque = idx != 255 if alpha else np.ones((h, w), bool)
	tw, th = 1 << max(3, (w - 1).bit_length()), 1 << max(3, (h - 1).bit_length())
	if (tw, th) != (w, h):
		rgb = _resize(rgb, (tw, th), nearest=alpha)
		opaque = _resize(opaque, (tw, th), nearest=True)
	count, body = encode_mip_chain(rgb, opaque, refine=refine)
	return tw, th, count, body


def wt_cache_key(w: int, h: int, pixels: bytes, palette: bytes, alpha: bool, refine: bool = False) -> str:
	return hashlib.md5(bytes((alpha,)) + struct.pack("<II", w, h) + pixels + palette + b"2"
		+ (b"r" if refine else b"")).hexdigest()


def wt_encode_cached(
	w: int, h: int, pixels: bytes, palette: bytes, alpha: bool, cache: Path | None, refine: bool = False
) -> tuple[str, int, int, int, bytes]:
	key = wt_cache_key(w, h, pixels, palette, alpha, refine)
	item = cache / f"{key}.bin" if cache else None
	if item is not None and item.is_file():
		cached = item.read_bytes()
		tw, th, levels = struct.unpack_from("<HHH", cached)
		return key, tw, th, levels, cached[6:]
	tw, th, levels, body = wt_encode(w, h, pixels, palette, alpha, refine)
	if item is not None:
		item.parent.mkdir(parents=True, exist_ok=True)
		item.write_bytes(struct.pack("<HHH", tw, th, levels) + body)
	return key, tw, th, levels, body


@dataclass
class GctEntry:
	name: str
	width: int
	height: int
	levels: int
	flags: int
	tag: int
	data: bytes = b""

	@property
	def alpha(self) -> bool:
		return bool(self.flags & GC_WT_FLAG_ALPHA)

	def level_size(self, level: int) -> int:
		return cmpr_size(max(8, self.width >> level), max(8, self.height >> level))

	def decode(self, level: int = 0) -> "np.ndarray":
		start = sum(self.level_size(i) for i in range(level))
		w, h = max(8, self.width >> level), max(8, self.height >> level)
		return cmpr_decode(self.data[start:start + self.level_size(level)], w, h)


def write_gct(entries: Iterable[GctEntry]) -> bytes:
	table_rows, blob = [], bytearray()
	entries = list(entries)
	for e in entries:
		table_rows.append(GC_WT_ENTRY.pack(
			e.name.encode("latin-1")[:16], e.width, e.height, e.levels, e.flags, e.tag & 0xFFFF, len(blob)))
		blob += e.data
		blob += bytes(-len(blob) & 31)
	head = GC_WT_MAGIC + struct.pack(">II", len(entries), len(blob))
	table = b"".join(table_rows)
	pad = bytes(-(len(head) + len(table)) & 31)
	return head + table + pad + bytes(blob)


def read_gct(data: bytes) -> list[GctEntry]:
	if data[:4] != GC_WT_MAGIC:
		raise ValueError("not a GCWT pack")
	count, blob_size = struct.unpack_from(">II", data, 4)
	first = 12 + count * GC_WT_ENTRY.size
	blob_start = first + (-first & 31)
	rows = []
	for i in range(count):
		raw, w, h, lv, fl, tag, of = GC_WT_ENTRY.unpack_from(data, 12 + i * GC_WT_ENTRY.size)
		rows.append((raw.split(b"\0")[0].decode("latin-1"), w, h, lv, fl, tag, of))
	out = []
	for name, w, h, lv, fl, tag, of in rows:
		e = GctEntry(name, w, h, lv, fl, tag)
		size = sum(e.level_size(i) for i in range(lv))
		e.data = data[blob_start + of: blob_start + of + size]
		out.append(e)
	return out


# --- world texture packs from BSP + WADs -------------------------------------

def wad_search_roots(gamedir: Path, extra: Iterable[Path] = ()) -> list[Path]:
	"""Gamedir, its siblings (mod + valve), user-chosen dirs, and the env vars
	the disc builder honours (DODGC_FULL_DOD, DODGC_WT_WAD_ROOT, HL_VALVE_DIR)."""
	roots: list[Path] = []
	seen: set[Path] = set()

	def add(path: Path | None) -> None:
		if path is None:
			return
		try:
			resolved = path.resolve()
		except OSError:
			return
		if not resolved.is_dir() or resolved in seen:
			return
		seen.add(resolved)
		roots.append(resolved)

	add(gamedir)
	for p in extra:
		add(Path(p))
	for name in (gamedir.name, "dod", "valve"):
		add(gamedir.parent / name)
	for env in ("DODGC_FULL_DOD", "DODGC_WT_WAD_ROOT", "HL_VALVE_DIR"):
		raw = os.environ.get(env, "").strip()
		if raw:
			add(Path(raw))
	return roots


def collect_wads(bsp_data: bytes, gamedir: Path, extra_dirs: Iterable[Path] = ()) -> tuple[dict, list[str]]:
	"""Every miptex a map may need, first WAD wins. Mirrors wt_read_wads."""
	found: dict = {}
	loaded: list[str] = []
	for wad in sorted(gamedir.glob("*.wad")):
		if wt_ingest_wad_file(found, wad):
			loaded.append(str(wad))
	wanted = {n.lower() for n in wt_bsp_wad_basenames(bsp_data)}
	wanted.update({"halflife.wad", "hlbasics.wad", "decals.wad"})
	have = {Path(p).name.lower() for p in loaded}
	for root in wad_search_roots(gamedir, extra_dirs):
		for wad in sorted(root.glob("*.wad")):
			key = wad.name.lower()
			if key in have:
				continue
			if key not in wanted and not key.startswith("dod_"):
				continue
			if wt_ingest_wad_file(found, wad):
				have.add(key)
				loaded.append(str(wad))
	return found, loaded


@dataclass
class WorldPackReport:
	entries: int = 0
	blob_bytes: int = 0
	missing: list[str] = field(default_factory=list)
	wads: list[str] = field(default_factory=list)
	notes: list[str] = field(default_factory=list)


def build_world_texture_pack(
	bsp: Path,
	*,
	gamedir: Path | None = None,
	wad_dirs: Iterable[Path] = (),
	cache: Path | None = None,
	progress: Callable[[int, int, str], None] | None = None,
	refine: bool = False,
) -> tuple[bytes | None, WorldPackReport, list[GctEntry]]:
	"""media/wt_<map>.gct for one map: each world texture as CMPR with mips."""
	data = bsp.read_bytes()
	if gamedir is None:
		gamedir = bsp.parent.parent if bsp.parent.name.lower() == "maps" else bsp.parent
	quake_pal = None
	if data[:4] == struct.pack("<i", 29):	# Quake BSP: miptex without palettes, WADs not referenced
		from .quake import GREY_PALETTE, quake_palette

		quake_pal = quake_palette(bsp)
		wads, loaded = {}, []
	else:
		wads, loaded = collect_wads(data, gamedir, wad_dirs)
	report = WorldPackReport(wads=loaded)
	if data[:4] == struct.pack("<i", 29) and quake_pal is None:
		report.notes.append("Quake palette (gfx/palette.lmp) not found; greyscale used")
		quake_pal = GREY_PALETTE
	names = bsp_miptex_offsets(data)
	todo = []
	seen = set()
	for name, off in names:
		if name in seen or not is_world_diffuse(name):
			continue
		seen.add(name)
		todo.append((name, off))
	entries: list[GctEntry] = []
	for i, (name, off) in enumerate(todo):
		if progress:
			progress(i, len(todo), name)
		mip = read_miptex(data, off, quake_pal)
		if mip is None and name in wads:
			mip = wt_read_miptex(*wads[name])
		if mip is None:
			report.missing.append(name)
			continue
		_, w, h, pixels, palette = mip
		alpha = name.startswith("{")
		key, tw, th, levels, body = wt_encode_cached(w, h, pixels, palette, alpha, cache, refine)
		entries.append(GctEntry(name, tw, th, levels, GC_WT_FLAG_ALPHA if alpha else 0, int(key[:4], 16), body))
	if progress:
		progress(len(todo), len(todo), "")
	if not entries:
		return None, report, entries
	packed = write_gct(entries)
	report.entries = len(entries)
	report.blob_bytes = len(packed)
	return packed, report, entries


def build_wad_texture_pack(
	wad: Path,
	*,
	cache: Path | None = None,
	include: set[str] | None = None,
	progress: Callable[[int, int, str], None] | None = None,
	palette: bytes | None = None,
	refine: bool = False,
) -> tuple[bytes | None, list[GctEntry]]:
	"""A GCWT pack holding every miptex in one WAD (or the `include` subset).

	Quake WAD2 miptex have no palette; `palette` (Quake's palette.lmp) colours them.
	"""
	found: dict = {}
	wt_ingest_wad_file(found, wad, (0x43, 0x44))
	names = [n for n in found if include is None or n in include]
	entries = []
	for i, name in enumerate(names):
		if progress:
			progress(i, len(names), name)
		mip = read_miptex(*found[name], palette)
		if mip is None:
			continue
		_, w, h, pixels, palette = mip
		alpha = name.startswith("{")
		key, tw, th, levels, body = wt_encode_cached(w, h, pixels, palette, alpha, cache, refine)
		entries.append(GctEntry(name, tw, th, levels, GC_WT_FLAG_ALPHA if alpha else 0, int(key[:4], 16), body))
	if include is None:	# gfx.wad / fonts.wad: HUD pictures and console fonts, index 255 see-through
		from .goldsrc import wad_pictures

		for name, pic in wad_pictures(wad).items():
			if name in found:
				continue
			alpha = pic.transparent
			key, tw, th, levels, body = wt_encode_cached(pic.width, pic.height, pic.pixels, pic.palette, alpha, cache, refine)
			entries.append(GctEntry(name, tw, th, levels, GC_WT_FLAG_ALPHA if alpha else 0, int(key[:4], 16), body))
	if progress:
		progress(len(names), len(names), "")
	return (write_gct(entries) if entries else None), entries
