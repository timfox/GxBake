"""Game archives: Quake/GoldSrc PACK (.pak), Source VPK (v1/v2) and PK3/ZIP.

Game content usually ships inside these, so `gxbake extract` unpacks them to
a folder that `gxbake convert` (or the app) can then take. The Quake palette
lookups in quake.py also read single files out of .pak archives.
"""

from __future__ import annotations

import fnmatch
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

ARCHIVE_EXTENSIONS = {".pak", ".vpk", ".pk3", ".pk4", ".zip"}
VPK_SIGNATURE = 0x55AA1234


@dataclass
class Member:
	name: str	# forward slashes, as stored
	size: int
	_read: object = None

	def read(self) -> bytes:
		return self._read()


def _pak_members(path: Path) -> list[Member]:
	with path.open("rb") as f:
		head = f.read(12)
		if head[:4] != b"PACK":
			raise ValueError(f"{path.name}: not a PACK file")
		dirofs, dirlen = struct.unpack_from("<ii", head, 4)
		f.seek(dirofs)
		table = f.read(dirlen)
	out = []
	for i in range(dirlen // 64):
		raw, pos, size = struct.unpack_from("<56sii", table, i * 64)
		name = raw.split(b"\0")[0].decode("latin-1")

		def read(pos=pos, size=size) -> bytes:
			with path.open("rb") as f:
				f.seek(pos)
				return f.read(size)

		out.append(Member(name, size, read))
	return out


def _vpk_members(path: Path) -> list[Member]:
	data = path.read_bytes()
	sig, version, tree_size = struct.unpack_from("<III", data, 0)
	if sig != VPK_SIGNATURE:
		raise ValueError(f"{path.name}: not a VPK directory file")
	header = 12 if version == 1 else 28
	pos = header
	data_base = header + tree_size	# archive index 0x7fff: data follows the tree in this file
	stem = path.stem[:-4] if path.stem.lower().endswith("_dir") else path.stem

	def cstr() -> str:
		nonlocal pos
		end = data.index(b"\0", pos)
		s = data[pos:end].decode("latin-1")
		pos = end + 1
		return s

	out = []
	while True:
		ext = cstr()
		if not ext:
			break
		while True:
			folder = cstr()
			if not folder:
				break
			while True:
				fname = cstr()
				if not fname:
					break
				_crc, preload, arch, off, length, _term = struct.unpack_from("<IHHIIH", data, pos)
				pos += 18
				pre = data[pos:pos + preload]
				pos += preload
				full = (f"{folder}/" if folder.strip() else "") + fname + (f".{ext}" if ext.strip() else "")

				def read(arch=arch, off=off, length=length, pre=pre) -> bytes:
					if not length:
						return pre
					if arch == 0x7FFF:
						return pre + data[data_base + off: data_base + off + length]
					chunk = path.with_name(f"{stem}_{arch:03d}.vpk")
					with chunk.open("rb") as f:
						f.seek(off)
						return pre + f.read(length)

				out.append(Member(full, preload + length, read))
	return out


def _zip_members(path: Path) -> list[Member]:
	zf = zipfile.ZipFile(path)
	return [Member(i.filename, i.file_size, (lambda n=i.filename: zf.read(n)))
		for i in zf.infolist() if not i.is_dir()]


def members(path: Path) -> list[Member]:
	path = Path(path)
	with path.open("rb") as f:
		head = f.read(4)
	if head == b"PACK":
		return _pak_members(path)
	if head == struct.pack("<I", VPK_SIGNATURE):
		return _vpk_members(path)
	if head[:2] == b"PK":
		return _zip_members(path)
	raise ValueError(f"{path.name}: not a PAK, VPK or PK3/ZIP archive")


def read_member(path: Path, name: str) -> bytes | None:
	"""One file from an archive by case-insensitive name, or None."""
	want = name.lower().replace("\\", "/")
	try:
		for m in members(path):
			if m.name.lower() == want:
				return m.read()
	except (OSError, ValueError, struct.error, zipfile.BadZipFile):
		return None
	return None


def extract(path: Path, out_dir: Path, patterns: list[str] | None = None, log=print) -> int:
	"""Unpack every member (or those matching a glob) under out_dir. Returns the count."""
	out_dir = Path(out_dir)
	count = 0
	for m in members(Path(path)):
		if patterns and not any(fnmatch.fnmatch(m.name.lower(), p.lower()) for p in patterns):
			continue
		rel = Path(*[p for p in m.name.replace("\\", "/").split("/") if p not in ("", ".", "..")])
		dest = out_dir / rel
		dest.parent.mkdir(parents=True, exist_ok=True)
		dest.write_bytes(m.read())
		count += 1
	log(f"extracted {count} files from {Path(path).name} to {out_dir}")
	return count
