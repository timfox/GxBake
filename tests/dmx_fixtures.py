"""DMX writers (binary 5 and keyvalues2) and a small skinned model, so the DMX readers are tested without game data.

Elements are written from N nodes whose attributes are (type, value) pairs, where the
type is a DMX attribute type name such as "vector3" or "element_array".
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field

_TYPES = ["element", "int", "float", "bool", "string", "binary", "time", "color", "vector2", "vector3", "vector4",
	"qangle", "quaternion", "matrix"]
_DIMS = {"vector2": 2, "vector3": 3, "vector4": 4, "qangle": 3, "quaternion": 4, "matrix": 16}


@dataclass(eq=False)
class N:
	type: str
	name: str = ""
	attrs: dict = field(default_factory=dict)
	id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _walk(root: N) -> list[N]:
	out: list[N] = []
	seen: set[int] = set()

	def visit(e):
		if e is None or id(e) in seen:
			return
		seen.add(id(e))
		out.append(e)
		for t, v in e.attrs.values():
			if t == "element":
				visit(v)
			elif t == "element_array":
				for x in v:
					visit(x)

	visit(root)
	return out


def binary(root: N, fmt: str = "model", fver: int = 18) -> bytes:
	"""Binary encoding 5: a string dictionary with int indices; array strings stay inline."""
	elems = _walk(root)
	index = {id(e): i for i, e in enumerate(elems)}
	strings: list[str] = []

	def s(x: str) -> int:
		if x not in strings:
			strings.append(x)
		return strings.index(x)

	body = bytearray(struct.pack("<i", len(elems)))
	for e in elems:
		body += struct.pack("<ii", s(e.type), s(e.name)) + uuid.UUID(e.id).bytes_le

	def value(t: str, v, in_array: bool) -> bytes:
		if t == "element":
			return struct.pack("<i", -1 if v is None else index[id(v)])
		if t == "int":
			return struct.pack("<i", v)
		if t == "float":
			return struct.pack("<f", v)
		if t == "bool":
			return bytes([1 if v else 0])
		if t == "string":
			return v.encode() + b"\0" if in_array else struct.pack("<i", s(v))
		if t == "binary":
			return struct.pack("<i", len(v)) + v
		if t == "time":
			return struct.pack("<i", round(v * 10000))
		if t == "color":
			return bytes(v)
		return struct.pack(f"<{_DIMS[t]}f", *v)

	for e in elems:
		body += struct.pack("<i", len(e.attrs))
		for k, (t, v) in e.attrs.items():
			body += struct.pack("<i", s(k))
			if t.endswith("_array"):
				inner = t[:-6]
				body += bytes([_TYPES.index(inner) + 1 + len(_TYPES)]) + struct.pack("<i", len(v))
				body += b"".join(value(inner, x, True) for x in v)
			else:
				body += bytes([_TYPES.index(t) + 1]) + value(t, v, False)
	head = f"<!-- dmx encoding binary 5 format {fmt} {fver} -->\n".encode() + b"\0"
	table = struct.pack("<i", len(strings)) + b"".join(x.encode() + b"\0" for x in strings)
	return head + table + bytes(body)


def kv2(root: N, fmt: str = "model", fver: int = 18) -> bytes:
	"""keyvalues2, with every element at the top level and referenced by id."""

	def text(t: str, v) -> str:
		if t == "element":
			return "" if v is None else v.id
		if t == "bool":
			return "1" if v else "0"
		if t in ("int", "string"):
			return str(v)
		if t in ("float", "time"):
			return repr(float(v))
		if t in ("color",):
			return " ".join(str(int(x)) for x in v)
		return " ".join(repr(float(x)) for x in v)

	lines = [f"<!-- dmx encoding keyvalues2 1 format {fmt} {fver} -->"]
	for e in _walk(root):
		lines += [f'"{e.type}"', "{", f'\t"id" "elementid" "{e.id}"', f'\t"name" "string" "{e.name}"']
		for k, (t, v) in e.attrs.items():
			if t.endswith("_array"):
				inner = t[:-6]
				lines.append(f'\t"{k}" "{t}"')
				lines.append("\t[")
				items = [f'\t\t"element" "{text(inner, x)}"' if inner == "element" else f'\t\t"{text(inner, x)}"'
					for x in v]
				lines.append(",\n".join(items))
				lines.append("\t]")
			else:
				lines.append(f'\t"{k}" "{t}" "{text(t, v)}"')
		lines.append("}")
	return ("\n".join(lines) + "\n").encode()


def transform(pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0)) -> N:
	return N("DmeTransform", "", {"position": ("vector3", pos), "orientation": ("quaternion", quat)})


def arm_model(animated: bool = True, up_axis: int = 3) -> N:
	"""A 16x16 column 64 units tall: the lower half follows joint "lower", the upper half joint "upper"
	(which sits at z = 32). With animated, the clip "wave" turns "upper" 90 degrees about +X over one second."""
	upper_tf = transform((0.0, 0.0, 32.0))
	upper = N("DmeJoint", "upper", {"transform": ("element", upper_tf), "children": ("element_array", [])})
	lower = N("DmeJoint", "lower", {"transform": ("element", transform()), "children": ("element_array", [upper])})
	pos, weights, joints = [], [], []
	for z, j in ((0.0, 0), (32.0, 0), (64.0, 1)):
		for x, y in ((-8, -8), (8, -8), (8, 8), (-8, 8)):
			pos.append((float(x), float(y), z))
			weights += [1.0, 0.0]
			joints += [j, 1 - j]
	# corners: one per position, uv from the corner's side and height
	uvs = [((i % 4) / 4.0, 1.0 - (i // 4) / 2.0) for i in range(12)]
	faces = []
	for ring in (0, 4):
		for k in range(4):
			a, b = ring + k, ring + (k + 1) % 4
			faces += [a, b, b + 4, a + 4, -1]	# counter-clockwise seen from outside
	material = N("DmeMaterial", "arm", {"mtlName": ("string", "models/test/arm")})
	vd = N("DmeVertexData", "bind", {
		"vertexFormat": ("string_array", ["positions", "textureCoordinates", "jointWeights", "jointIndices"]),
		"jointCount": ("int", 2),
		"positions": ("vector3_array", pos), "positionsIndices": ("int_array", list(range(12))),
		"textureCoordinates": ("vector2_array", uvs), "textureCoordinatesIndices": ("int_array", list(range(12))),
		"jointWeights": ("float_array", weights), "jointIndices": ("int_array", joints),
	})
	faceset = N("DmeFaceSet", "arm", {"material": ("element", material), "faces": ("int_array", faces)})
	mesh = N("DmeMesh", "arm", {"bindState": ("element", vd), "currentState": ("element", vd),
		"baseStates": ("element_array", [vd]), "faceSets": ("element_array", [faceset])})
	body = N("DmeDag", "body", {"transform": ("element", transform()), "shape": ("element", mesh),
		"children": ("element_array", [])})
	model_attrs = {"transform": ("element", transform()), "children": ("element_array", [lower, body]),
		"jointList": ("element_array", [lower, upper])}
	if up_axis != 3:
		model_attrs["axisSystem"] = ("element", N("DmeAxisSystem", "axisSystem", {"upAxis": ("int", up_axis),
			"forwardParity": ("int", 1), "coordSys": ("int", 0)}))
	model = N("DmeModel", "arm", model_attrs)
	root = {"skeleton": ("element", model), "model": ("element", model)}
	if animated:
		s = 0.70710678
		layer = N("DmeQuaternionLogLayer", "", {"times": ("time_array", [0.0, 1.0]),
			"values": ("quaternion_array", [(0.0, 0.0, 0.0, 1.0), (s, 0.0, 0.0, s)])})
		log = N("DmeQuaternionLog", "", {"layers": ("element_array", [layer])})
		ch = N("DmeChannel", "upper_o", {"toElement": ("element", upper_tf), "toAttribute": ("string", "orientation"),
			"log": ("element", log)})
		tf = N("DmeTimeFrame", "", {"start": ("time", 0.0), "duration": ("time", 1.0)})
		clip = N("DmeChannelsClip", "wave", {"channels": ("element_array", [ch]), "frameRate": ("int", 30),
			"timeFrame": ("element", tf)})
		root["animationList"] = ("element", N("DmeAnimationList", "anims", {"animations": ("element_array", [clip])}))
	return N("DmElement", "root", root)


def particles(material: str = "particle/test/spark.vmt") -> N:
	"""A .pcf root: "sparks" renders sprites with a material, "burst" only spawns it as a child."""
	render = N("DmeParticleOperator", "Render animated sprites", {
		"functionName": ("string", "render_animated_sprites"), "animation rate": ("float", 10.0)})
	sparks = N("DmeParticleSystemDefinition", "sparks", {
		"material": ("string", material), "max_particles": ("int", 32), "radius": ("float", 4.0),
		"color": ("color", (255, 200, 100, 255)),
		"renderers": ("element_array", [render]), "operators": ("element_array", []),
		"initializers": ("element_array", []), "emitters": ("element_array", []), "children": ("element_array", [])})
	child = N("DmeParticleChild", "sparks", {"child": ("element", sparks), "delay": ("float", 0.0)})
	burst = N("DmeParticleSystemDefinition", "burst", {
		"material": ("string", "vgui/white"), "max_particles": ("int", 1), "renderers": ("element_array", []),
		"children": ("element_array", [child])})
	return N("DmElement", "untitled", {"particleSystemDefinitions": ("element_array", [burst, sparks])})


def vtf_with_sheet(rgba, rects: list[tuple[float, float, float, float]]) -> bytes:
	"""A VTF 7.3 whose sheet resource has one sequence with a frame per (u0, v0, u1, v1) rectangle."""
	from .game_fixtures import vtf

	base = vtf(rgba, version=(7, 3))
	body = base[88:]
	sheet = struct.pack("<ii", 0, 1) + struct.pack("<iiif", 0, 0, len(rects), float(len(rects)))
	sheet += b"".join(struct.pack("<f4f", 1.0, *r) for r in rects)
	head = bytearray(base[:80]) + bytearray(16)
	struct.pack_into("<I", head, 12, 96)
	struct.pack_into("<I", head, 68, 2)
	struct.pack_into("<3sBI", head, 80, b"\x30\0\0", 0, 96)
	struct.pack_into("<3sBI", head, 88, b"\x10\0\0", 0, 96 + len(body))
	return bytes(head) + body + struct.pack("<I", len(sheet)) + sheet
