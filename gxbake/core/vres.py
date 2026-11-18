"""Source 2 compiled resources (*_c files): the block container, NTRO introspection and external references.

Every Source 2 asset (vtex_c, vmat_c, vmdl_c, vmesh_c, vsnd_c, vmap_c,
vwrld_c, vwnod_c, vents_c, ...) shares one container: a header and a table of
4CC blocks (RERL, REDI/RED2, NTRO, DATA, CTRL, MDAT, VBIB, MBUF, ...). DATA is
KeyValues3 (kv3.py) on anything compiled since about 2017 and an NTRO
introspected struct before that; both come back as the same plain dict here,
so the format readers above this module do not care which one a file used.

Ported from ValveResourceFormat (MIT, see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import kv3

HEADER_VERSION = 12

# extension (without _c) -> resource kind
EXTENSIONS = {
	"vtex": "texture", "vmat": "material", "vmdl": "model", "vmesh": "mesh", "vsnd": "sound",
	"vmap": "map", "vwrld": "world", "vwnod": "worldnode", "vents": "entitylump", "vphys": "physics",
	"vanim": "animation", "vagrp": "animgroup", "vseq": "sequencegroup", "vmorf": "morph",
	"vpcf": "particle", "vsndevts": "soundevents", "vdata": "vdata", "vpost": "postprocess",
	"vnmskel": "nmskeleton", "vnmclip": "nmclip", "vrman": "manifest", "vsvg": "panorama",
	"vxml": "panorama", "vcss": "panorama", "vjs": "panorama", "vts": "panorama", "vpdi": "panorama",
	"vmix": "soundmix", "vsmart": "smartprop", "vcompmat": "compositematerial",
}


def resource_kind(path: Path | str) -> str | None:
	name = str(path).lower()
	if not name.endswith("_c"):
		return None
	ext = name.rsplit(".", 1)[-1][:-2]
	return EXTENSIONS.get(ext, "resource")


def looks_like_resource(data: bytes) -> bool:
	if len(data) < 16:
		return False
	size, hv, _ver, boff, bcount = struct.unpack_from("<IHHII", data, 0)
	return hv == HEADER_VERSION and boff == 8 and bcount < 64


def _offset_string(data: bytes, pos: int) -> str:
	"""A CResourceString: int32 offset relative to its own position."""
	rel = struct.unpack_from("<i", data, pos)[0]
	if rel == 0:
		return ""
	at = pos + rel
	end = data.index(b"\0", at)
	return data[at:end].decode("utf-8", "replace")


@dataclass
class Block:
	type: str
	offset: int
	size: int


@dataclass
class Resource:
	data: bytes
	file_size: int
	version: int
	blocks: list[Block]
	path: str = ""
	_cache: dict = field(default_factory=dict, repr=False)

	@property
	def kind(self) -> str | None:
		return resource_kind(self.path)

	def block(self, kind: str) -> Block | None:
		for b in self.blocks:
			if b.type == kind:
				return b
		return None

	def raw(self, b: "Block | str | int") -> bytes:
		if isinstance(b, str):
			b = self.block(b)
		elif isinstance(b, int):
			b = self.blocks[b]
		return self.data[b.offset:b.offset + b.size] if b else b""

	@property
	def data_end(self) -> int:
		"""Where pixel data starts in a texture: the end of the DATA block."""
		b = self.block("DATA")
		return b.offset + b.size if b else self.file_size

	def keyvalues(self, b: "Block | str | int" = "DATA", struct_name: str | None = None):
		"""Block contents as a dict (KV3 or NTRO); None when the block is missing."""
		if isinstance(b, str):
			b = self.block(b)
		elif isinstance(b, int):
			b = self.blocks[b]
		if b is None:
			return None
		key = (b.offset, struct_name)
		if key in self._cache:
			return self._cache[key]
		if b.size >= 4 and (kv3.is_kv3(self.data, b.offset) or self.data[b.offset:b.offset + 4] == b"<!--"):
			v = kv3.read(self.data, b.offset, b.offset + b.size)
		elif self.block("NTRO") is not None:
			v = Introspection(self).read(b, struct_name)
		else:
			v = None
		self._cache[key] = v
		return v

	def external_refs(self) -> dict[int, str]:
		if "rerl" not in self._cache:
			self._cache["rerl"] = _read_rerl(self)
		return self._cache["rerl"]

	def info(self) -> dict:
		return {"Resource version": self.version, "Blocks": " ".join(b.type for b in self.blocks)}


def parse(data: bytes, path: Path | str = "") -> Resource:
	if not looks_like_resource(data):
		raise ValueError(f"{Path(str(path)).name or 'data'}: not a Source 2 resource")
	file_size, _hv, version, boff, count = struct.unpack_from("<IHHII", data, 0)
	pos = 8 + boff
	blocks = []
	for _ in range(count):
		raw_type = data[pos:pos + 4]
		rel, size = struct.unpack_from("<II", data, pos + 4)
		blocks.append(Block(raw_type.decode("latin-1"), pos + 4 + rel, size))
		pos += 12
	return Resource(data, file_size, version, blocks, str(path))


def load(path: Path) -> Resource:
	path = Path(path)
	return parse(path.read_bytes(), path)


def _read_rerl(res: Resource) -> dict[int, str]:
	b = res.block("RERL")
	if b is None:
		return {}
	d = res.data
	off, count = struct.unpack_from("<II", d, b.offset)
	out = {}
	pos = b.offset + off
	for _ in range(count):
		rid = struct.unpack_from("<Q", d, pos)[0]
		out[rid] = _offset_string(d, pos + 8)
		pos += 16
	return out


# -- NTRO ---------------------------------------------------------------------

# SchemaFieldType
_T_STRUCT, _T_ENUM, _T_EXTREF = 1, 2, 3
_SCALARS = {
	4: ("<b", int), 5: ("<B", int), 6: ("<i", int), 7: ("<I", int), 8: ("<f", float), 9: ("<d", float),
	10: ("<b", int), 11: ("<B", int), 12: ("<h", int), 13: ("<H", int), 14: ("<i", int), 15: ("<I", int),
	16: ("<q", int), 17: ("<Q", int), 18: ("<f", float), 19: ("<d", float), 20: ("<f", float), 30: ("<?", bool),
}
_FLOATS = {21: 2, 22: 3, 23: 4, 24: 3, 25: 4, 26: 16, 27: 4, 33: 12, 36: 12, 40: 8, 42: 3, 43: 3, 44: 12, 45: 3, 46: 6}


@dataclass
class _Field:
	name: str
	count: int
	offset: int
	indirections: bytes
	type_data: int
	type: int


@dataclass
class _Struct:
	id: int
	name: str
	disk_size: int
	base: int
	flags: int
	fields: list[_Field]


class Introspection:
	"""Reads an NTRO-described DATA block into a dict, as ValveResourceFormat's NTRO does."""

	def __init__(self, res: Resource):
		self.res = res
		self.d = res.data
		self.structs: dict[int, _Struct] = {}
		self.order: list[_Struct] = []
		self.enums: dict[int, dict[int, str]] = {}
		b = res.block("NTRO")
		d = self.d
		pos = b.offset + 4
		off, count = struct.unpack_from("<iI", d, pos)
		p = pos + off
		for _ in range(count):
			_iv, sid = struct.unpack_from("<II", d, p)
			name = _offset_string(d, p + 8)
			_crc, _uv, disk, _align, base = struct.unpack_from("<IiHHI", d, p + 12)
			foff, fcount = struct.unpack_from("<iI", d, p + 28)
			fields = []
			fp = p + 28 + foff
			for _f in range(fcount):
				fname = _offset_string(d, fp)
				cnt, doff = struct.unpack_from("<hh", d, fp + 4)
				ioff, icount = struct.unpack_from("<iI", d, fp + 8)
				ind = d[fp + 8 + ioff:fp + 8 + ioff + icount] if icount else b""
				tdata, ftype = struct.unpack_from("<Ih", d, fp + 16)
				fields.append(_Field(fname, cnt, doff, bytes(ind), tdata, ftype))
				fp += 24
			flags = d[p + 36]
			st = _Struct(sid, name, disk, base, flags, fields)
			self.structs[sid] = st
			self.order.append(st)
			p += 40
		pos = b.offset + 12
		off, count = struct.unpack_from("<iI", d, pos)
		p = pos + off
		for _ in range(count):
			_iv, eid = struct.unpack_from("<II", d, p)
			voff, vcount = struct.unpack_from("<iI", d, p + 20)
			vals = {}
			vp = p + 20 + voff
			for _v in range(vcount):
				vals[struct.unpack_from("<i", d, vp + 4)[0]] = _offset_string(d, vp)
				vp += 8
			self.enums[eid] = vals
			p += 28

	def read(self, block: Block, struct_name: str | None = None) -> dict:
		st = None
		if struct_name:
			st = next((s for s in self.order if s.name == struct_name), None)
		if st is None:
			st = self.order[0]
		return self._struct(st, block.offset)

	def _struct(self, st: _Struct, at: int) -> dict:
		concrete = st
		out: dict = {}
		if st.flags & 1:
			concrete = self.structs.get(struct.unpack_from("<I", self.d, at)[0], st)
			out["_class"] = concrete.name
		s = concrete
		depth = 0
		while s is not None and depth < 16:
			for f in s.fields:
				if f.offset < 0:
					continue
				out[f.name] = self._field(f, at + f.offset)
			s = self.structs.get(s.base) if s.base else None
			depth += 1
		return out

	def _field(self, f: _Field, at: int):
		if f.type == 4 and f.count > 1 and not f.indirections:
			return _offset_string(self.d, at)
		if f.count > 1:
			return self._elements(f, 0, f.count, at)
		return self._indirect(f, 0, at)

	def _size(self, f: _Field) -> int:
		if f.type == _T_STRUCT:
			return self.structs[f.type_data].disk_size
		if f.type in _SCALARS:
			return struct.calcsize(_SCALARS[f.type][0])
		if f.type in _FLOATS:
			return 4 * _FLOATS[f.type]
		return {2: 4, 3: 8, 28: 4, 29: 16, 31: 4, 41: 8}.get(f.type, 4)

	def _indirect(self, f: _Field, level: int, at: int):
		if level == len(f.indirections):
			return self._value(f, at)
		kind = f.indirections[level]
		rel = struct.unpack_from("<i", self.d, at)[0]
		if kind == 3:	# resource pointer
			return None if rel == 0 else self._indirect(f, level + 1, at + rel)
		if kind == 4:	# resource array
			count = struct.unpack_from("<i", self.d, at + 4)[0]
			return self._elements(f, level + 1, count, at + rel)
		raise ValueError(f"NTRO: unsupported indirection {kind}")

	def _elements(self, f: _Field, level: int, count: int, at: int):
		if level == len(f.indirections) and f.type in (5, 11):
			return bytes(self.d[at:at + count])
		step = 4 if level < len(f.indirections) and f.indirections[level] == 3 else \
			8 if level < len(f.indirections) else self._size(f)
		return [self._indirect(f, level, at + i * step) for i in range(count)]

	def _value(self, f: _Field, at: int):
		d = self.d
		t = f.type
		if t == _T_STRUCT:
			return self._struct(self.structs[f.type_data], at)
		if t == _T_ENUM:
			v = struct.unpack_from("<i", d, at)[0]
			return self.enums.get(f.type_data, {}).get(v, v)
		if t in _SCALARS:
			fmt, cast = _SCALARS[t]
			return cast(struct.unpack_from(fmt, d, at)[0])
		if t in _FLOATS:
			return list(struct.unpack_from(f"<{_FLOATS[t]}f", d, at))
		if t in (_T_EXTREF, 41):
			rid = struct.unpack_from("<Q", d, at)[0]
			return self.res.external_refs().get(rid) if rid else None
		if t == 28:
			return list(d[at:at + 4])
		if t == 29:
			return bytes(d[at:at + 16])
		if t == 31:
			return _offset_string(d, at)
		raise ValueError(f"NTRO: unsupported field type {t} ({f.name})")
