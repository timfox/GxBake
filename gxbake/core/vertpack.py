"""GCSV: a studio model's vertices and normals as s16/s8 fixed point.

Studio models store every vertex and normal as three floats (12 bytes) in
bone space. Gekko's quantised paired-single loads (psq_l with a GQR) turn
s16 or s8 into floats for free, and GX can take s16 positions and s8
normals straight from a display list. So the port can keep 6 + 3 bytes
instead of 12 + 12 per vertex/normal pair with no runtime cost.

Each model gets one power-of-two scale: `frac` fractional bits, the most
that keeps every coordinate inside s16. That is exactly what a GQR
dequantise scale holds (value = int / 2**frac), so the port loads it with
one register setup per model. Normals use frac 6 (GX's s8 normal format
is 1.6 too).

	"GCSV" u32 version (1) u32 count u32 0
	count × { u16 bodypart; u16 model; u32 nverts; u32 nnorms; u8 pos_frac;
	          u8 nrm_frac; u16 0; u32 pos off; u32 nrm off }		(24 bytes)
	then per model: positions as big-endian s16 x,y,z; normals as s8 x,y,z;
	each array 32-byte aligned (DMA and GX friendly)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

GCSV_MAGIC = b"GCSV"
GCSV_ENTRY = struct.Struct(">HHIIBBHII")
NRM_FRAC = 6


@dataclass
class ModelVerts:
	bodypart: int
	model: int
	positions: "np.ndarray"	# N×3 float32, bone space
	normals: "np.ndarray"	# M×3 float32


def studio_model_verts(raw: bytes) -> list[ModelVerts]:
	"""Every model's vertex and normal arrays, in file order."""
	out = []
	nbody, body = struct.unpack_from("<ii", raw, 204)
	for bp in range(max(0, nbody)):
		pb = body + bp * 76
		nmodels, model_off = struct.unpack_from("<i", raw, pb + 64)[0], struct.unpack_from("<i", raw, pb + 72)[0]
		for m in range(max(0, nmodels)):
			mb = model_off + m * 112
			nv, _vi, vo, nn, _ni, no = struct.unpack_from("<6i", raw, mb + 80)
			pos = np.frombuffer(raw, "<f4", nv * 3, vo).reshape(-1, 3) if nv > 0 else np.zeros((0, 3), np.float32)
			nrm = np.frombuffer(raw, "<f4", nn * 3, no).reshape(-1, 3) if nn > 0 else np.zeros((0, 3), np.float32)
			out.append(ModelVerts(bp, m, pos, nrm))
	return out


def pos_frac(positions: "np.ndarray") -> int:
	"""Most fractional bits that keep every coordinate within s16 (0..15)."""
	peak = float(np.abs(positions).max()) if positions.size else 0.0
	if peak == 0:
		return 15
	return int(max(0, min(15, math.floor(math.log2(32767 / peak)))))


def quantise(positions: "np.ndarray", normals: "np.ndarray") -> tuple[int, "np.ndarray", "np.ndarray"]:
	frac = pos_frac(positions)
	p = np.clip(np.round(positions.astype(np.float64) * (1 << frac)), -32768, 32767).astype(">i2")
	n = np.clip(np.round(normals.astype(np.float64) * (1 << NRM_FRAC)), -128, 127).astype("i1")
	return frac, p, n


def write_vertex_pack(raw: bytes) -> bytes:
	models = studio_model_verts(raw)
	head = GCSV_MAGIC + struct.pack(">III", 1, len(models), 0)
	pos = len(head) + len(models) * GCSV_ENTRY.size
	pos += -pos & 31
	rows, blobs = [], []
	for mv in models:
		frac, p, n = quantise(mv.positions, mv.normals)
		pb, nb = p.tobytes(), n.tobytes()
		po = pos
		pos += len(pb) + (-len(pb) & 31)
		no = pos
		pos += len(nb) + (-len(nb) & 31)
		rows.append(GCSV_ENTRY.pack(mv.bodypart, mv.model, len(p), len(n), frac, NRM_FRAC, 0, po, no))
		blobs += [pb + bytes(-len(pb) & 31), nb + bytes(-len(nb) & 31)]
	body = head + b"".join(rows)
	return body + bytes(-len(body) & 31) + b"".join(blobs)


def read_vertex_pack(blob: bytes) -> list[ModelVerts]:
	"""Dequantised arrays, the floats the port's psq_l loads would produce."""
	if blob[:4] != GCSV_MAGIC:
		raise ValueError("not a GCSV vertex pack")
	_ver, count, _ = struct.unpack_from(">III", blob, 4)
	out = []
	for i in range(count):
		bp, m, nv, nn, pf, nf, _z, po, no = GCSV_ENTRY.unpack_from(blob, 16 + i * GCSV_ENTRY.size)
		p = np.frombuffer(blob, ">i2", nv * 3, po).reshape(-1, 3).astype(np.float32) / (1 << pf)
		n = np.frombuffer(blob, "i1", nn * 3, no).reshape(-1, 3).astype(np.float32) / (1 << nf)
		out.append(ModelVerts(bp, m, p, n))
	return out


def pack_error(raw: bytes) -> tuple[float, float]:
	"""(worst position error in model units, worst normal angle in degrees)."""
	worst_p, worst_n = 0.0, 0.0
	for a, b in zip(studio_model_verts(raw), read_vertex_pack(write_vertex_pack(raw))):
		if len(a.positions):
			worst_p = max(worst_p, float(np.abs(a.positions - b.positions).max()))
		if len(a.normals):
			na = a.normals / np.maximum(np.linalg.norm(a.normals, axis=1, keepdims=True), 1e-9)
			nb = b.normals / np.maximum(np.linalg.norm(b.normals, axis=1, keepdims=True), 1e-9)
			cos = np.clip((na * nb).sum(1), -1, 1)
			worst_n = max(worst_n, float(np.degrees(np.arccos(cos)).max()))
	return worst_p, worst_n
