"""Manage GCWT texture packs (.gct) without going back to the source assets:
list, extract to PNG, build from images, add or replace, remove and merge.

The engine reads at most 512 entries per pack and 15 characters of each name
(char name[16], NUL-terminated), and finds a world texture by name and content
tag, so names are kept lowercase and unique within a pack.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

from .gct import GctEntry, read_gct, write_gct
from .images import ImageOptions, encode_image, load_rgba

MAX_NAME = 15
MAX_ENTRIES = 512


@dataclass
class Row:
	name: str
	width: int
	height: int
	levels: int
	alpha: bool
	tag: int
	bytes: int


def rows(entries: list[GctEntry]) -> list[Row]:
	return [Row(e.name, e.width, e.height, e.levels, e.alpha, e.tag, len(e.data)) for e in entries]


def load(path: Path) -> list[GctEntry]:
	return read_gct(Path(path).read_bytes())


def save(path: Path, entries: list[GctEntry]) -> int:
	data = write_gct(entries)
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	Path(path).write_bytes(data)
	return len(data)


def matches(name: str, patterns: list[str] | None) -> bool:
	return not patterns or any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)


def extract(entries: list[GctEntry], out_dir: Path, patterns: list[str] | None = None,
		all_levels: bool = False) -> list[Path]:
	"""Decode entries to PNG: <name>.png, plus <name>_mip<N>.png with all_levels."""
	from PIL import Image

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	written = []
	for e in entries:
		if not matches(e.name, patterns):
			continue
		safe = "".join(c if c.isalnum() or c in "-_.+{!~" else "_" for c in e.name) or "texture"
		for level in range(e.levels if all_levels else 1):
			p = out_dir / (f"{safe}.png" if level == 0 else f"{safe}_mip{level}.png")
			Image.fromarray(e.decode(level)).save(p)
			written.append(p)
	return written


def entry_name(path: Path) -> str:
	return Path(path).stem.lower()


@dataclass
class EditReport:
	added: list[str] = field(default_factory=list)
	replaced: list[str] = field(default_factory=list)
	removed: list[str] = field(default_factory=list)
	warnings: list[str] = field(default_factory=list)


def _check(entries: list[GctEntry], rep: EditReport) -> None:
	if len(entries) > MAX_ENTRIES:
		rep.warnings.append(f"{len(entries)} entries; the engine reads at most {MAX_ENTRIES}")


def encode_files(paths: list[Path], opts: ImageOptions, rep: EditReport) -> list[GctEntry]:
	out = []
	for p in paths:
		name = entry_name(p)
		if len(name) > MAX_NAME:
			rep.warnings.append(f"{p.name}: name cut to {name[:MAX_NAME]!r} (15 characters max)")
		out.append(encode_image(load_rgba(Path(p)), opts, name))
	return out


def upsert(entries: list[GctEntry], new: list[GctEntry], rep: EditReport) -> list[GctEntry]:
	"""Add new entries; one with the same name replaces the old one in place."""
	work = list(entries)
	index = {e.name.lower(): i for i, e in enumerate(work)}
	for e in new:
		key = e.name.lower()
		if key in index:
			work[index[key]] = e
			rep.replaced.append(e.name)
		else:
			index[key] = len(work)
			work.append(e)
			rep.added.append(e.name)
	_check(work, rep)
	return work


def remove(entries: list[GctEntry], patterns: list[str], rep: EditReport) -> list[GctEntry]:
	keep = []
	for e in entries:
		if matches(e.name, patterns):
			rep.removed.append(e.name)
		else:
			keep.append(e)
	return keep


def merge(packs: list[list[GctEntry]], rep: EditReport) -> list[GctEntry]:
	"""Union of packs in order. Identical entries are kept once; a later pack's
	entry with the same name but different pixels replaces the earlier one."""
	out: list[GctEntry] = []
	index: dict[str, int] = {}
	for entries in packs:
		for e in entries:
			key = e.name.lower()
			if key not in index:
				index[key] = len(out)
				out.append(e)
				rep.added.append(e.name)
			elif out[index[key]].data != e.data or out[index[key]].tag != e.tag:
				out[index[key]] = e
				rep.replaced.append(e.name)
	_check(out, rep)
	return out
