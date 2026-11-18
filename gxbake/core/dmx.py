"""Valve Datamodel (DMX) files: binary encodings 1-5 and 9, and keyvalues2 text.

DMX is the container behind Source's model sources (.dmx from Blender Source
Tools or the SFM), particle systems (.pcf) and a few Source 2 tool files. Each
file is a graph of typed elements (DmeModel, DmeMesh, DmeParticleSystemDefinition...)
whose attributes hold scalars, vectors, strings, other elements or arrays of them.
Follows Datamodel.NET's reading of the encodings.
"""

from __future__ import annotations

import re
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_HEADER = re.compile(rb"<!--\s*dmx\s+encoding\s+(\S+)\s+(\d+)\s+format\s+(\S+)\s+(\d+)\s*-->")

# attribute types of binary 1-5 (9 adds uint64 and uint8); arrays follow
_TYPES = ["element", "int", "float", "bool", "string", "binary", "time", "color", "vector2", "vector3", "vector4",
	"qangle", "quaternion", "matrix"]
_TYPES9 = _TYPES + ["uint64", "uint8"]
_DIMS = {"vector2": 2, "vector3": 3, "vector4": 4, "qangle": 3, "quaternion": 4, "matrix": 16}


@dataclass(eq=False)
class Element:
	type: str
	name: str = ""
	id: str = ""
	attrs: dict = field(default_factory=dict)

	def __getitem__(self, key: str):
		return self.attrs[key]

	def get(self, key: str, default=None):
		return self.attrs.get(key, default)

	def __contains__(self, key: str) -> bool:
		return key in self.attrs

	def __repr__(self) -> str:
		return f"<{self.type} {self.name!r}>"


@dataclass
class Datamodel:
	encoding: str
	encoding_version: int
	format: str
	format_version: int
	root: Element | None
	elements: list[Element]

	def of_type(self, type_name: str) -> list[Element]:
		return [e for e in self.elements if e.type == type_name]


def header(data: bytes) -> tuple[str, int, str, int] | None:
	m = _HEADER.match(data.lstrip(b"\xef\xbb\xbf \t\r\n")[:200])
	if not m:
		return None
	return m.group(1).decode(), int(m.group(2)), m.group(3).decode(), int(m.group(4))


def is_dmx(head: bytes) -> bool:
	return header(head) is not None


def load(path: Path) -> Datamodel:
	return loads(Path(path).read_bytes())


def loads(data: bytes) -> Datamodel:
	hd = header(data)
	if hd is None:
		raise ValueError("not a DMX file (no <!-- dmx encoding ... --> header)")
	enc, ver, fmt, fver = hd
	if enc == "binary":
		root, elems = _Binary(data, ver).read()
	elif enc in ("keyvalues2", "keyvalues2_flat", "keyvalues2_noids"):
		root, elems = _read_kv2(data.decode("utf-8", "replace"))
	else:
		raise ValueError(f"DMX encoding {enc} {ver} is not supported")
	return Datamodel(enc, ver, fmt, fver, root, elems)


# -- binary ----------------------------------------------------------------------------------------


class _Binary:
	def __init__(self, data: bytes, version: int):
		if version not in (1, 2, 3, 4, 5, 9):
			raise ValueError(f"DMX binary encoding {version} is not supported")
		self.d = data
		self.v = version
		self.p = data.index(b"\0") + 1	# the header line is null-terminated
		self.types = _TYPES9 if version >= 9 else _TYPES
		self.strings: list[str] = []
		self.elements: list[Element] = []

	def _i32(self) -> int:
		v = struct.unpack_from("<i", self.d, self.p)[0]
		self.p += 4
		return v

	def _raw(self) -> str:
		end = self.d.index(b"\0", self.p)
		s = self.d[self.p:end].decode("utf-8", "replace")
		self.p = end + 1
		return s

	def _str(self) -> str:
		if self.v == 1:
			return self._raw()
		if self.v < 5:
			i = struct.unpack_from("<h", self.d, self.p)[0]
			self.p += 2
		else:
			i = self._i32()
		return self.strings[i] if 0 <= i < len(self.strings) else ""

	def read(self) -> tuple[Element | None, list[Element]]:
		if self.v >= 9:	# prefix elements: attributes only, kept nowhere
			for _ in range(self._i32()):
				for _ in range(self._i32()):
					self._raw()
					self._attribute()
		if self.v >= 2:
			if self.v < 4:
				n = struct.unpack_from("<h", self.d, self.p)[0]
				self.p += 2
			else:
				n = self._i32()
			self.strings = [self._raw() for _ in range(n)]
		count = self._i32()
		for _ in range(count):
			t = self._str()
			name = self._str() if self.v >= 4 else self._raw()
			gid = str(uuid.UUID(bytes_le=self.d[self.p:self.p + 16]))
			self.p += 16
			self.elements.append(Element(t, name, gid))
		self._stubs: dict[str, Element] = {}
		for e in self.elements:
			for _ in range(self._i32()):
				key = self._str()
				e.attrs[key] = self._attribute()
		return (self.elements[0] if self.elements else None), self.elements

	def _attribute(self):
		tid = self.d[self.p] - 1
		self.p += 1
		n = len(self.types)
		array = False
		if self.v >= 9 and tid >= 2 * n:
			tid -= 2 * n
			array = True
		elif tid >= n:
			tid -= n
			array = True
		if not 0 <= tid < n:
			raise ValueError(f"DMX: unknown attribute type {tid + 1}")
		t = self.types[tid]
		if t == "time" and self.v < 3:
			raise ValueError("DMX: ObjectID attributes are not supported")
		if not array:
			return self._value(t, self.v < 4)
		return [self._value(t, True) for _ in range(self._i32())]

	def _value(self, t: str, raw_string: bool):
		d = self.d
		if t == "element":
			i = self._i32()
			if i == -1:
				return None
			if i == -2:
				gid = self._raw()
				return self._stubs.setdefault(gid, Element("DmElement", "", gid))
			return self.elements[i] if 0 <= i < len(self.elements) else None
		if t == "int":
			return self._i32()
		if t == "float":
			v = struct.unpack_from("<f", d, self.p)[0]
			self.p += 4
			return v
		if t == "bool":
			self.p += 1
			return bool(d[self.p - 1])
		if t == "string":
			return self._raw() if raw_string else self._str()
		if t == "binary":
			n = self._i32()
			self.p += n
			return d[self.p - n:self.p]
		if t == "time":
			return self._i32() / 10000.0
		if t == "color":
			self.p += 4
			return tuple(d[self.p - 4:self.p])
		if t == "uint64":
			self.p += 8
			return struct.unpack_from("<Q", d, self.p - 8)[0]
		if t == "uint8":
			self.p += 1
			return d[self.p - 1]
		k = _DIMS[t]
		v = struct.unpack_from(f"<{k}f", d, self.p)
		self.p += 4 * k
		return v


# -- keyvalues2 ------------------------------------------------------------------------------------

_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}\[\],])|//[^\n]*|(\S+)')


def _tokens(text: str) -> list[str]:
	out: list[str] = []
	for m in _TOKEN.finditer(text):
		if m.group(1) is not None:
			out.append("\0" + m.group(1).replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n"))
		elif m.group(2) is not None:
			out.append(m.group(2))
		elif m.group(3) is not None:
			out.append("\0" + m.group(3))
	return out


def _kv2_value(t: str, s: str):
	if t == "int" or t == "uint8" or t == "uint64":
		return int(float(s)) if s.strip() else 0
	if t == "float":
		return float(s) if s.strip() else 0.0
	if t == "bool":
		return s.strip() not in ("0", "", "false")
	if t == "time":
		return float(s) if s.strip() else 0.0
	if t == "color":
		return tuple(int(x) for x in s.split()[:4])
	if t == "binary":
		return bytes.fromhex("".join(s.split()))
	if t in _DIMS:
		v = [float(x) for x in s.split()]
		return tuple(v + [0.0] * (_DIMS[t] - len(v)))[:_DIMS[t]]
	return s


def _read_kv2(text: str) -> tuple[Element | None, list[Element]]:
	toks = _tokens(text[text.index("-->") + 3:])
	pos = 0
	by_id: dict[str, Element] = {}
	elements: list[Element] = []
	refs: list[tuple[dict | list, object, str]] = []	# (container, key, id) resolved at the end

	def take() -> str:
		nonlocal pos
		t = toks[pos]
		pos += 1
		return t

	def element(type_name: str) -> Element:
		e = Element(type_name)
		elements.append(e)
		if take() != "{":
			raise ValueError("DMX keyvalues2: expected {")
		while toks[pos] != "}":
			key = take()[1:]
			t = take()[1:]
			if toks[pos] == "{":	# an inline element
				e.attrs[key] = element(t)
				continue
			if t.endswith("_array"):
				inner = t[:-6]
				if take() != "[":
					raise ValueError("DMX keyvalues2: expected [")
				items: list = []
				while toks[pos] != "]":
					if toks[pos] == ",":
						pos_skip()
						continue
					tok = take()[1:]
					if inner == "element":
						if toks[pos] == "{":
							items.append(element(tok))
						else:	# "element" "<id>"
							gid = take()[1:]
							items.append(None)
							refs.append((items, len(items) - 1, gid))
					else:
						items.append(_kv2_value(inner, tok))
				take()
				e.attrs[key] = items
				continue
			val = take()[1:]
			if key == "id" and t == "elementid":
				e.id = val
				by_id[val] = e
			elif key == "name" and t == "string":
				e.name = val
				e.attrs["name"] = val
			elif t == "element":
				e.attrs[key] = None
				if val:
					refs.append((e.attrs, key, val))
			else:
				e.attrs[key] = _kv2_value(t, val)
		take()
		return e

	def pos_skip() -> None:
		nonlocal pos
		pos += 1

	roots = []
	while pos < len(toks):
		roots.append(element(take()[1:]))
	for cont, key, gid in refs:
		cont[key] = by_id.get(gid)
	return (roots[0] if roots else None), elements
