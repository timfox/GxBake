"""Game archives: Quake/GoldSrc PACK (.pak), Source VPK (v1/v2), PK3/ZIP and Panda3D multifiles (.mf).

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

ARCHIVE_EXTENSIONS = {".pak", ".vpk", ".pk3", ".pk4", ".zip", ".mf"}
VPK_SIGNATURE = 0x55AA1234
MF_MAGIC = b"pmf\0\n\r"


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


# Multifile subfile flags (panda/src/express/multifile.h)
_MF_DELETED, _MF_INDEX_INVALID, _MF_DATA_INVALID = 0x1, 0x2, 0x4
_MF_COMPRESSED, _MF_ENCRYPTED, _MF_SIGNATURE = 0x8, 0x10, 0x20


def _mf_start(data: bytes) -> int:
	"""Offset of the pmf magic: a multifile may open with #! comment lines (Panda's header prefix)."""
	pos = 0
	while data[pos:pos + 1] == b"#":
		nl = data.find(b"\n", pos)
		if nl < 0:
			break
		pos = nl + 1
	return pos


def _mf_members(path: Path) -> list[Member]:
	"""Panda3D multifile (Toontown's phase_N.mf): an index of linked entries, names XOR 0xff,
	zlib-compressed subfiles. Encrypted subfiles need the game's password and are skipped."""
	import zlib

	data = path.read_bytes()
	base = _mf_start(data)
	if data[base:base + 6] != MF_MAGIC:
		raise ValueError(f"{path.name}: not a Panda3D multifile")
	major, minor, scale = struct.unpack_from("<hhI", data, base + 6)
	if major != 1:
		raise ValueError(f"{path.name}: multifile version {major}.{minor} is not supported")
	scale = scale or 1
	pos = base + 14 + (4 if minor >= 1 else 0)
	if scale > 1 and data[pos:pos + 4] == b"\0\0\0\0":	# index padded to the scale factor
		pos = -(-pos // scale) * scale
	out = []
	seen = set()
	while pos + 4 <= len(data) and pos not in seen:
		seen.add(pos)
		nxt, = struct.unpack_from("<I", data, pos)
		if nxt == 0:
			break
		start, length, flags = struct.unpack_from("<IIH", data, pos + 4)
		p = pos + 14
		orig = length
		if flags & (_MF_COMPRESSED | _MF_ENCRYPTED):
			orig, = struct.unpack_from("<I", data, p)
			p += 4
		if minor >= 1:
			p += 4	# timestamp
		nlen, = struct.unpack_from("<H", data, p)
		name = bytes(b ^ 0xFF for b in data[p + 2:p + 2 + nlen]).decode("utf-8", "replace")
		pos = nxt * scale	# offsets count from the start of the file, header prefix included
		if flags & (_MF_DELETED | _MF_INDEX_INVALID | _MF_DATA_INVALID | _MF_SIGNATURE):
			continue
		if flags & _MF_ENCRYPTED:
			out.append(Member(name, orig, (lambda n=name: _mf_encrypted(path, n))))
			continue
		off = start * scale

		def read(off=off, length=length, flags=flags) -> bytes:
			raw = data[off:off + length]
			return zlib.decompress(raw) if flags & _MF_COMPRESSED else raw

		out.append(Member(name, orig, read))
	return out


def _mf_encrypted(path: Path, name: str) -> bytes:
	raise ValueError(f"{path.name}: {name} is encrypted (the game's multifile password is needed)")


def members(path: Path) -> list[Member]:
	path = Path(path)
	with path.open("rb") as f:
		head = f.read(4)
		if head[:1] == b"#" or head[:3] == b"pmf":
			f.seek(0)
			start = f.read(4096)
			if start[_mf_start(start):_mf_start(start) + 6] == MF_MAGIC:
				return _mf_members(path)
	if head == b"PACK":
		return _pak_members(path)
	if head == struct.pack("<I", VPK_SIGNATURE):
		return _vpk_members(path)
	if head[:2] == b"PK":
		return _zip_members(path)
	raise ValueError(f"{path.name}: not a PAK, VPK, PK3/ZIP or Panda3D multifile")


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
	count = skipped = 0
	for m in members(Path(path)):
		if patterns and not any(fnmatch.fnmatch(m.name.lower(), p.lower()) for p in patterns):
			continue
		rel = Path(*[p for p in m.name.replace("\\", "/").split("/") if p not in ("", ".", "..")])
		try:
			body = m.read()
		except ValueError as exc:	# an encrypted multifile member
			log(f"skipped: {exc}")
			skipped += 1
			continue
		dest = out_dir / rel
		dest.parent.mkdir(parents=True, exist_ok=True)
		dest.write_bytes(body)
		count += 1
	log(f"extracted {count} files from {Path(path).name} to {out_dir}" + (f" ({skipped} encrypted, skipped)" if skipped else ""))
	return count
