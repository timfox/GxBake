"""Generated Source 2 files: a KV3 writer, the resource container and tiny vtex/vmat/vsnd/vmdl.

Nothing here comes from game data; every file is built from the layouts the
readers in gxbake/core expect (which follow ValveResourceFormat).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

# KV3 node types (kv3.py)
_NULL, _BOOL, _INT64, _UINT64, _DOUBLE, _STRING, _BLOB, _ARRAY, _OBJECT, _ARRAY_TYPED = range(1, 11)
_INT32, _TRUE, _FALSE, _FLOAT, _UINT8 = 11, 13, 14, 19, 23


def kv3(obj) -> bytes:
	"""A KV3 v1 document (uncompressed) holding obj: each lane is written in read order."""
	types = bytearray()
	l1, l4, l8 = bytearray(), bytearray(), bytearray()
	strings: list[str] = []
	index: dict[str, int] = {}

	def sid(s: str) -> int:
		if s not in index:
			index[s] = len(strings)
			strings.append(s)
		return index[s]

	def value(v):
		nonlocal l1, l4, l8
		if v is None:
			types.append(_NULL)
		elif isinstance(v, bool):
			types.append(_TRUE if v else _FALSE)
		elif isinstance(v, int):
			types.append(_INT32)
			l4 += struct.pack("<i", v)
		elif isinstance(v, float):
			types.append(_DOUBLE)
			l8 += struct.pack("<d", v)
		elif isinstance(v, str):
			types.append(_STRING)
			l4 += struct.pack("<i", sid(v))
		elif isinstance(v, (bytes, bytearray)):
			types.append(_BLOB)
			l4 += struct.pack("<i", len(v))
			l1 += v
		elif isinstance(v, np.ndarray):
			flat = v.ravel()
			types.append(_ARRAY_TYPED)
			l4 += struct.pack("<i", len(flat))
			if v.dtype.kind == "f":
				types.append(_FLOAT)
				l4 += flat.astype("<f4").tobytes()
			else:
				types.append(_INT32)
				l4 += flat.astype("<i4").tobytes()
		elif isinstance(v, (list, tuple)):
			types.append(_ARRAY)
			l4 += struct.pack("<i", len(v))
			for x in v:
				value(x)
		elif isinstance(v, dict):
			types.append(_OBJECT)
			l4 += struct.pack("<i", len(v))
			for k, x in v.items():
				# reader order: the value's type byte, then the key index (lane 4), then the value
				l4 += struct.pack("<i", sid(k))
				value(x)
		else:
			raise TypeError(type(v))

	value(obj)
	head4 = struct.pack("<i", len(strings))
	body = bytearray(l1)
	body += bytes(-len(body) % 4)
	body += head4 + l4
	body += bytes(-len(body) % 8)
	body += l8
	body += bytes(-len(body) % 8)
	for s in strings:
		body += s.encode() + b"\0"
	body += types
	body += struct.pack("<I", 0xFFEEDD00)
	c1, c4, c8 = len(l1), 1 + len(l4) // 4, len(l8) // 8
	return struct.pack("<I16sI4i", 0x4B563301, bytes(16), 0, c1, c4, c8, len(body)) + bytes(body)


def kv3_legacy(obj) -> bytes:
	"""The same document under the old VKV3 header, uncompressed."""
	strings: list[str] = []
	index: dict[str, int] = {}

	def sid(s: str) -> int:
		if s not in index:
			index[s] = len(strings)
			strings.append(s)
		return index[s]

	def value(v) -> bytes:
		if v is None:
			return bytes([_NULL])
		if isinstance(v, bool):
			return bytes([_TRUE if v else _FALSE])
		if isinstance(v, int):
			return bytes([_INT32]) + struct.pack("<i", v)
		if isinstance(v, float):
			return bytes([_DOUBLE]) + struct.pack("<d", v)
		if isinstance(v, str):
			return bytes([_STRING]) + struct.pack("<i", sid(v))
		if isinstance(v, (bytes, bytearray)):
			return bytes([_BLOB]) + struct.pack("<i", len(v)) + bytes(v)
		if isinstance(v, np.ndarray):
			v = [float(x) for x in v.ravel()] if v.dtype.kind == "f" else [int(x) for x in v.ravel()]
		if isinstance(v, (list, tuple)):
			return bytes([_ARRAY]) + struct.pack("<i", len(v)) + b"".join(value(x) for x in v)
		if isinstance(v, dict):
			out = bytes([_OBJECT]) + struct.pack("<i", len(v))
			for k, x in v.items():
				out += struct.pack("<i", sid(k)) + value(x)
			return out
		raise TypeError(type(v))

	body = value(obj)
	table = struct.pack("<I", len(strings)) + b"".join(s.encode() + b"\0" for s in strings)
	return struct.pack("<I", 0x03564B56) + bytes(32) + table + body


def resource(blocks: list[tuple[str, bytes]], version: int = 0, stream: bytes = b"") -> bytes:
	"""The Source 2 block container; `stream` goes after the header's file size (sound data, pixels)."""
	head = 16 + 12 * len(blocks)
	pos = (head + 15) // 16 * 16
	table = bytearray()
	data = bytearray()
	for i, (kind, payload) in enumerate(blocks):
		at = pos + len(data)
		entry = 16 + 12 * i
		table += kind.encode("latin-1") + struct.pack("<II", at - (entry + 4), len(payload))
		data += payload
		if i + 1 < len(blocks):
			data += bytes(-len(data) % 16)	# the last block ends where pixel/sample data begins
	body = bytes(table) + bytes(pos - head) + bytes(data)
	size = 16 + len(body)
	return struct.pack("<IHHII", size, 12, version, 8, len(blocks)) + body + stream


# -- textures and materials -------------------------------------------------------


def vtex_rgba(rgba: "np.ndarray") -> bytes:
	h, w = rgba.shape[:2]
	data = struct.pack("<HH4f", 1, 0, 0.5, 0.5, 0.5, 0.5)
	data += struct.pack("<HHHBBI", w, h, 1, 4, 1, 0)	# RGBA8888, one mip
	data += struct.pack("<II", 8, 0)
	return resource([("DATA", data)], version=1, stream=rgba.astype(np.uint8).tobytes())


def vmat(color: str, alpha_test: bool = False) -> bytes:
	d = {"m_materialName": "materials/test.vmat", "m_shaderName": "complex.vfx",
		"m_textureParams": [{"m_name": "g_tColor", "m_pValue": color}],
		"m_intParams": [{"m_name": "F_ALPHA_TEST", "m_nValue": int(alpha_test)}],
		"m_floatParams": []}
	return resource([("DATA", kv3(d))], version=0)


# -- sounds -----------------------------------------------------------------------


def vsnd_pcm16(samples: "np.ndarray", rate: int = 44100, loop_start: int = -1, loop_end: int = 0) -> bytes:
	"""Resource version 4 sound with PCM16 mono samples."""
	pcm = samples.astype("<i2").tobytes()
	n = len(samples)
	data = struct.pack("<HBB", rate, 0, 1)
	data += struct.pack("<iIfI", loop_start, n, n / rate, 0)
	data += struct.pack("<IiI", 0, 0, len(pcm))
	data += bytes(12)
	data += struct.pack("<i", loop_end)
	return resource([("DATA", data)], version=4, stream=pcm)


# -- models -----------------------------------------------------------------------

_QS = 0.70710677 / 16384.0


def quat_word(v: float, sign: bool = False) -> int:
	if v >= 0:
		w = min(16383, int(round(v / _QS))) | 0x4000
	else:
		w = max(0, int(round(v / _QS)) + 16384)
	return w | (0x8000 if sign else 0)


def pack_quat(q) -> bytes:
	"""(x, y, z, w) with w the largest component → the six-byte form (no dropped-axis swap)."""
	x, y, z, w = q
	return struct.pack("<HHH", quat_word(x), quat_word(y), quat_word(z, w < 0))


def _segment(decoder: int, elements: list[int], payload: bytes) -> bytes:
	return struct.pack("<hhhh", decoder, 1, len(elements), 8 + 2 * len(elements) + len(payload)) + \
		struct.pack(f"<{len(elements)}h", *elements) + payload


def arm_vmdl(directory: Path, material: str = "materials/test.vmat") -> Path:
	"""A two-bone arm (root + forearm 10 units up Z) with one bending animation.

	Two quads: the lower one on the root, the upper one on the arm. Writes
	models/arm.vmdl_c under `directory` and returns its path.
	"""
	pos = np.array([
		[-2, 0, 0], [2, 0, 0], [2, 0, 9], [-2, 0, 9],	# root
		[-2, 0, 11], [2, 0, 11], [2, 0, 20], [-2, 0, 20],	# arm
	], np.float32)
	nrm = np.tile(np.array([0, -1, 0], np.float32), (8, 1))
	uv = np.array([[0, 1], [1, 1], [1, 0.5], [0, 0.5], [0, 0.5], [1, 0.5], [1, 0], [0, 0]], np.float32)
	bones = np.array([[0, 0, 0, 0]] * 4 + [[1, 0, 0, 0]] * 4, np.uint8)
	weights = np.array([[255, 0, 0, 0]] * 8, np.uint8)
	stride = 12 + 12 + 8 + 4 + 4
	vb = bytearray()
	for i in range(8):
		vb += pos[i].tobytes() + nrm[i].tobytes() + uv[i].tobytes() + bones[i].tobytes() + weights[i].tobytes()
	tris = np.array([[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]], np.uint16)
	layout = [
		{"m_pSemanticName": "POSITION", "m_nSemanticIndex": 0, "m_Format": 6, "m_nOffset": 0},
		{"m_pSemanticName": "NORMAL", "m_nSemanticIndex": 0, "m_Format": 6, "m_nOffset": 12},
		{"m_pSemanticName": "TEXCOORD", "m_nSemanticIndex": 0, "m_Format": 16, "m_nOffset": 24},
		{"m_pSemanticName": "BLENDINDICES", "m_nSemanticIndex": 0, "m_Format": 30, "m_nOffset": 32},
		{"m_pSemanticName": "BLENDWEIGHT", "m_nSemanticIndex": 0, "m_Format": 28, "m_nOffset": 36},
	]
	mesh = {"m_sceneObjects": [{"m_drawCalls": [{
		"m_nPrimitiveType": "RENDER_PRIM_TRIANGLES", "m_material": material,
		"m_vertexBuffers": [{"m_hBuffer": 0}], "m_indexBuffer": {"m_hBuffer": 0},
		"m_nStartIndex": 0, "m_nIndexCount": 12, "m_nBaseVertex": 0}]}]}
	data = {
		"m_name": "models/arm.vmdl",
		"m_modelSkeleton": {
			"m_boneName": ["root", "arm"],
			"m_nParent": np.array([-1, 0], np.int32),
			"m_bonePosParent": [np.array([0, 0, 0], np.float32), np.array([0, 0, 10], np.float32)],
			"m_boneRotParent": [np.array([0, 0, 0, 1], np.float32), np.array([0, 0, 0, 1], np.float32)],
		},
		"m_remappingTable": np.array([0, 1], np.int32),
		"m_remappingTableStarts": np.array([0], np.int32),
		"m_refMeshes": [],
	}
	# The arm swings 0°, 45°, 90° about +X; the root stays put (static quaternion).
	frames = []
	for deg in (0.0, 45.0, 90.0):
		h = math.radians(deg) / 2
		frames.append((math.sin(h), 0.0, 0.0, math.cos(h)))
	anim_rot = b"".join(pack_quat(q) for q in frames)	# one element per frame
	static_root = pack_quat((0.0, 0.0, 0.0, 1.0))
	arm_pos = struct.pack("<3f", 0.0, 0.0, 10.0)
	anim = {
		"m_animArray": [{"m_name": "@bend", "fps": 30.0, "m_flags": {"m_bLooping": True, "m_bDelta": False},
			"m_pData": {"m_nFrames": 3, "m_frameblockArray": [
				{"m_nStartFrame": 0, "m_nEndFrame": 2, "m_segmentIndexArray": np.array([0, 1, 2], np.int32)}]}},
			{"m_name": "lean_delta", "fps": 30.0, "m_flags": {"m_bLooping": False, "m_bDelta": True},
			"m_pData": {"m_nFrames": 1, "m_frameblockArray": []}}],
		"m_decoderArray": [{"m_szName": "CCompressedAnimQuaternion"}, {"m_szName": "CCompressedStaticQuaternion"},
			{"m_szName": "CCompressedStaticFullVector3"}],
		"m_segmentArray": [
			{"m_container": _segment(0, [1], anim_rot), "m_nLocalChannel": 0},
			{"m_container": _segment(1, [0], static_root), "m_nLocalChannel": 0},
			{"m_container": _segment(2, [1], arm_pos), "m_nLocalChannel": 1},
		],
	}
	group = {"m_decodeKey": {"m_dataChannelArray": [
		{"m_szChannelClass": "BoneChannel", "m_szVariableName": "Angle",
			"m_szElementNameArray": ["root", "arm"], "m_nElementIndexArray": np.array([0, 1], np.int32)},
		{"m_szChannelClass": "BoneChannel", "m_szVariableName": "Position",
			"m_szElementNameArray": ["arm"], "m_nElementIndexArray": np.array([1], np.int32)},
	]}}
	# block order: DATA 0, CTRL 1, MDAT 2, ANIM 3, AGRP 4
	ctrl = {
		"embedded_meshes": [{"m_Name": "arm", "m_nMeshIndex": 0, "m_nDataBlock": 2,
			"m_vertexBuffers": [{"m_nElementCount": 8, "m_nElementSizeInBytes": stride,
				"m_inputLayoutFields": layout, "m_pData": bytes(vb)}],
			"m_indexBuffers": [{"m_nElementCount": 12, "m_nElementSizeInBytes": 2,
				"m_inputLayoutFields": [], "m_pData": tris.astype("<u2").tobytes()}]}],
		"embedded_animation": {"group_data_block": 4, "anim_data_block": 3, "seqgroup_data_block": 0},
	}
	raw = resource([("DATA", kv3(data)), ("CTRL", kv3(ctrl)), ("MDAT", kv3(mesh)), ("ANIM", kv3(anim)),
		("AGRP", kv3(group))], version=1)
	out = Path(directory) / "models" / "arm.vmdl_c"
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_bytes(raw)
	return out
