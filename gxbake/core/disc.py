"""Disc usage of a staged gamedir: what it costs on a GameCube disc, what is
duplicated, and what no map refers to.

The port's disc builder exposes game data through ISO9660 (2048-byte sectors,
one extent per file) next to the boot FST, so every file costs whole sectors.
A GameCube disc holds 1,459,978,240 bytes.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import budget, goldsrc
from .gct import read_gct

SECTOR = 2048


def sectors(n: int) -> int:
	return (n + SECTOR - 1) // SECTOR * SECTOR


@dataclass
class DiscReport:
	gamedir: str
	files: int
	bytes: int
	on_disc: int	# sector-rounded
	capacity: int
	by_dir: list[tuple[str, int]]
	duplicates: list[dict] = field(default_factory=list)	# {"size", "wasted", "paths"}
	shared_textures: list[dict] = field(default_factory=list)	# {"name", "packs", "wasted"}
	unreferenced: list[tuple[str, int]] = field(default_factory=list)

	@property
	def wasted(self) -> int:
		return sum(d["wasted"] for d in self.duplicates)

	def to_json(self) -> dict:
		return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"wasted": self.wasted}


# Directories whose files maps refer to by name; everything else (gfx, menus,
# configs, HUD sprites listed in sprites/*.txt) is loaded by the engine or DLL.
_REFERENCED_DIRS = ("models", "gc_studio", "sound")


def _hash(path: Path) -> str:
	h = hashlib.sha1()
	with path.open("rb") as f:
		for block in iter(lambda: f.read(1 << 20), b""):
			h.update(block)
	return h.hexdigest()


def disc_report(gamedir: Path, include: list[str] | None = None, capacity: int = budget.DISC_CAPACITY) -> DiscReport:
	gamedir = Path(gamedir)
	files: list[tuple[str, Path, int]] = []
	for root, _dirs, names in os.walk(gamedir):
		for n in names:
			p = Path(root) / n
			files.append((p.relative_to(gamedir).as_posix(), p, p.stat().st_size))
	files.sort()
	by_dir: dict[str, int] = defaultdict(int)
	for rel, _p, size in files:
		by_dir[rel.split("/", 1)[0] if "/" in rel else "."] += sectors(size)

	# Exact duplicates: group by size first so only candidates are hashed.
	by_size: dict[int, list[tuple[str, Path]]] = defaultdict(list)
	for rel, p, size in files:
		if size:
			by_size[size].append((rel, p))
	dups = []
	for size, group in by_size.items():
		if len(group) < 2:
			continue
		by_hash: dict[str, list[str]] = defaultdict(list)
		for rel, p in group:
			by_hash[_hash(p)].append(rel)
		for paths in by_hash.values():
			if len(paths) > 1:
				dups.append({"size": size, "wasted": sectors(size) * (len(paths) - 1), "paths": paths})
	dups.sort(key=lambda d: -d["wasted"])

	# The same texture (same CMPR bytes) in several packs.
	tex: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
	for rel, p, _size in files:
		if rel.lower().endswith(".gct"):
			try:
				for e in read_gct(p.read_bytes()):
					tex[hashlib.sha1(e.data).hexdigest()].append((e.name, rel, len(e.data)))
			except (OSError, ValueError):
				continue
	shared = []
	for rows in tex.values():
		packs = sorted({r[1] for r in rows})
		if len(packs) > 1:
			shared.append({"name": rows[0][0], "packs": packs, "wasted": rows[0][2] * (len(packs) - 1)})
	shared.sort(key=lambda d: -d["wasted"])

	# Files in model/sound dirs that no map names.
	named: set[str] = set()
	maps_dir = gamedir / "maps"
	for bsp in sorted(maps_dir.glob("*.bsp")) if maps_dir.is_dir() else []:
		try:
			refs = budget.map_references(budget.parse_entities(goldsrc.bsp_entities_text(bsp.read_bytes())))
		except (OSError, ValueError):
			continue
		for m in refs.models:
			stem = Path(m).stem.lower()
			named |= {m.lower(), f"gc_studio/{stem}.mdl", f"gc_studio/{stem}t.mdl"}
		named |= {f"sound/{s.lower()}" for s in refs.sounds}
	for pattern in include or []:
		for p in gamedir.glob(pattern):
			rel = p.relative_to(gamedir).as_posix().lower()
			named.add(rel)
			if rel.startswith("models/"):
				stem = Path(rel).stem
				named |= {f"gc_studio/{stem}.mdl", f"gc_studio/{stem}t.mdl"}
	unref = [(rel, size) for rel, _p, size in files
		if rel.split("/", 1)[0].lower() in _REFERENCED_DIRS and rel.lower() not in named]
	unref.sort(key=lambda t: -t[1])

	total = sum(s for *_x, s in files)
	return DiscReport(str(gamedir), len(files), total, sum(sectors(s) for *_x, s in files), capacity,
		sorted(by_dir.items(), key=lambda t: -t[1]), dups, shared, unref)


def format_disc(rep: DiscReport, limit: int = 15) -> str:
	from .audit import human_bytes as hb

	out = [f"{rep.gamedir}: {rep.files} files, {hb(rep.bytes)} ({hb(rep.on_disc)} in 2 KiB sectors), "
		f"{rep.on_disc / rep.capacity:.1%} of a {hb(rep.capacity)} disc"]
	out.append("")
	out.append("by directory (on disc):")
	for name, size in rep.by_dir[:limit]:
		out.append(f"  {hb(size):>10}  {name}/" if name != "." else f"  {hb(size):>10}  (top level)")
	if rep.duplicates:
		out.append("")
		out.append(f"identical files: {len(rep.duplicates)} groups, {hb(rep.wasted)} could be saved")
		for d in rep.duplicates[:limit]:
			out.append(f"  {hb(d['wasted']):>10}  {' = '.join(d['paths'][:4])}" + (" …" if len(d["paths"]) > 4 else ""))
	if rep.shared_textures:
		wasted = sum(d["wasted"] for d in rep.shared_textures)
		out.append("")
		out.append(f"textures stored in more than one .gct: {len(rep.shared_textures)} ({hb(wasted)} repeated on disc; "
			"each map loads its own pack, so this only costs disc space)")
		for d in rep.shared_textures[:limit]:
			out.append(f"  {hb(d['wasted']):>10}  {d['name']} in {len(d['packs'])} packs")
	if rep.unreferenced:
		out.append("")
		out.append(f"not named by any map: {len(rep.unreferenced)} files, {hb(sum(s for _r, s in rep.unreferenced))} "
			"(the game DLL may still precache these; check before removing)")
		for rel, size in rep.unreferenced[:limit]:
			out.append(f"  {hb(size):>10}  {rel}")
	return "\n".join(out)
