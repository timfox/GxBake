"""Source 2 animations → studio sequences.

A model's animations live in its own blocks (CTRL "embedded_animation" names
the decode key, animation data and sequence blocks), in the .vagrp_c groups it
lists in m_refAnimGroups, or in .vanim_c files those groups name. Each
animation is a list of frame blocks; each block names segments, and a segment
is one compressed channel (Position, Angle or Scale) for a set of bones. The
decoders and the quaternion packing follow ValveResourceFormat (MIT, see
THIRD_PARTY_NOTICES.md).

Decoded values are bone-local and replace the bind pose for the bones a
segment covers, as in VRF's Frame. Delta (additive) animations are skipped,
like the Source 1 importer does.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from . import kv3, vres, vtex

_QSCALE = 0.70710677 / 16384.0


def _quat_order(a: float, c: float, e: float, w: float, s1: bool, s2: bool) -> tuple[float, float, float, float]:
	# Which slot the dropped (largest) component goes in depends on the two sign bits.
	if s1 and s2:
		return (c, e, w, a)
	if s1:
		return (e, w, a, c)
	if s2:
		return (w, a, c, e)
	return (a, c, e, w)


def _channel_kind(ch: dict) -> str | None:
	cls = str(ch.get("m_szChannelClass", ""))
	var = str(ch.get("m_szVariableName", ""))
	if cls in ("MorphChannel", "UserChannel"):
		return None
	return {"Position": "pos", "Angle": "rot", "Scale": "scale"}.get(var)


class _Segment:
	"""One decoded segment: values per (frame, wanted bone)."""

	def __init__(self, kind: str, decoder: str, elements: list[int], data: bytes, remap: list[int]):
		self.kind = kind
		self.decoder = decoder
		n = len(elements)
		self.n = n
		pos = {}
		for i in range(n - 1, -1, -1):
			if elements[i] >= 0:
				pos[elements[i]] = i
		self.bones: list[int] = []
		self.slots: list[int] = []
		for bone, el in enumerate(remap):
			if el >= 0 and el in pos:
				self.bones.append(bone)
				self.slots.append(pos[el])
		self.data = data
		self.static: np.ndarray | None = None
		self.frames: np.ndarray | None = None	# (nframes, n, k)
		self._decode()

	def _decode(self):
		d, n, name = self.data, self.n, self.decoder
		if n == 0:
			return
		if name in ("CCompressedStaticQuaternion", "CCompressedAnimQuaternion"):
			count = len(d) // 6
			q = np.array([read_quaternion(d, 6 * i) for i in range(count)], np.float64).reshape(-1, 4)
			if name == "CCompressedStaticQuaternion":
				self.static = q[:n]
			else:
				self.frames = q[:count // n * n].reshape(-1, n, 4)
		elif name == "CCompressedFullQuaternion":
			q = np.frombuffer(d[:len(d) // 16 * 16], "<f4").astype(np.float64).reshape(-1, 4)
			self.frames = q[:len(q) // n * n].reshape(-1, n, 4)
		elif name == "CCompressedStaticFullVector3":
			self.static = np.frombuffer(d[:n * 12], "<f4").astype(np.float64).reshape(n, 3)
		elif name == "CCompressedStaticVector3":
			self.static = np.frombuffer(d[:n * 6], "<f2").astype(np.float64).reshape(n, 3)
		elif name == "CCompressedFullVector3":
			v = np.frombuffer(d[:len(d) // 12 * 12], "<f4").astype(np.float64).reshape(-1, 3)
			self.frames = v[:len(v) // n * n].reshape(-1, n, 3)
		elif name == "CCompressedAnimVector3":
			v = np.frombuffer(d[:len(d) // 6 * 6], "<f2").astype(np.float64).reshape(-1, 3)
			self.frames = v[:len(v) // n * n].reshape(-1, n, 3)
		elif name == "CCompressedDeltaVector3":
			base = np.frombuffer(d[:n * 12], "<f4").astype(np.float64).reshape(n, 3)
			rest = d[n * 12:]
			delta = np.frombuffer(rest[:len(rest) // 6 * 6], "<f2").astype(np.float64).reshape(-1, 3)
			self.frames = delta[:len(delta) // n * n].reshape(-1, n, 3) + base[None]
		elif name == "CCompressedStaticFloat":
			self.static = np.frombuffer(d[:n * 4], "<f4").astype(np.float64).reshape(n, 1)
		elif name == "CCompressedFullFloat":
			v = np.frombuffer(d[:len(d) // 4 * 4], "<f4").astype(np.float64)
			self.frames = v[:len(v) // n * n].reshape(-1, n, 1)
		elif name in ("CCompressedStaticBool", "CCompressedFullBool"):
			return	# bools only drive data channels
		else:
			raise ValueError(f"unknown animation decoder {name}")

	def apply(self, local_frame: int, pos: np.ndarray, rot: np.ndarray, scl: np.ndarray):
		if self.static is not None:
			vals = self.static[self.slots]
		elif self.frames is not None and len(self.frames):
			vals = self.frames[min(local_frame, len(self.frames) - 1)][self.slots]
		else:
			return
		bones = self.bones
		if self.kind == "pos" and vals.shape[1] == 3:
			pos[bones] = vals
		elif self.kind == "rot" and vals.shape[1] == 4:
			rot[bones] = vals
		elif self.kind == "scale" and vals.shape[1] == 1:
			scl[bones] = vals[:, 0]


def read_quaternion(b: bytes, at: int = 0) -> tuple[float, float, float, float]:
	w1, w2, w3 = struct.unpack_from("<HHH", b, at)

	def comp(word: int) -> float:
		v = word & 0x3FFF
		return _QSCALE * v if word & 0x4000 else _QSCALE * (v - 16384)

	a, c, e = comp(w1), comp(w2), comp(w3)
	w = math.sqrt(max(0.0, 1.0 - a * a - c * c - e * e))
	if w3 & 0x8000:
		w = -w
	return _quat_order(a, c, e, w, bool(w1 & 0x8000), bool(w2 & 0x8000))


class SegmentTable:
	def __init__(self, anim_data: dict, decode_key: dict, bone_names: list[str]):
		decoders = [str(x.get("m_szName", "")) for x in anim_data.get("m_decoderArray") or []]
		lower = {}
		for i, n in enumerate(bone_names):
			lower.setdefault(n.lower(), i)
		channels = []
		for ch in decode_key.get("m_dataChannelArray") or []:
			kind = _channel_kind(ch)
			remap = [-1] * len(bone_names)
			if kind is not None:
				names = ch.get("m_szElementNameArray") or []
				idx = kv3.ints(ch.get("m_nElementIndexArray"))
				for name, el in zip(names, idx):
					b = lower.get(str(name).lower())
					if b is not None:
						remap[b] = int(el)
			channels.append((kind, remap))
		self.raw = anim_data.get("m_segmentArray") or []
		self.decoders = decoders
		self.channels = channels
		self._cache: dict[int, _Segment | None] = {}

	def get(self, index: int) -> "_Segment | None":
		if index in self._cache:
			return self._cache[index]
		seg = None
		s = self.raw[index]
		ci = int(s.get("m_nLocalChannel", -1))
		if 0 <= ci < len(self.channels) and self.channels[ci][0] is not None:
			kind, remap = self.channels[ci]
			c = s.get("m_container")
			c = bytes(c) if not isinstance(c, (bytes, bytearray)) else bytes(c)
			dec, _card, n, _total = struct.unpack_from("<hhhh", c, 0)
			end = 8 + 2 * n
			elements = list(struct.unpack_from(f"<{n}h", c, 8)) if n > 0 else []
			if 0 <= dec < len(self.decoders):
				seg = _Segment(kind, self.decoders[dec], elements, c[end:], remap)
		self._cache[index] = seg
		return seg


class Animation:
	def __init__(self, desc: dict, table: SegmentTable, name: str | None = None, loop: bool | None = None,
			delta: bool = False):
		flags = desc.get("m_flags") or {}
		self.name = name or str(desc.get("m_name", "anim"))
		self.fps = float(desc.get("fps", 30.0) or 30.0)
		self.loop = bool(flags.get("m_bLooping")) if loop is None else loop
		self.delta = delta or bool(flags.get("m_bDelta")) or bool(flags.get("m_bAnimGraphAdditive"))
		p = desc.get("m_pData")
		src = p if isinstance(p, dict) else desc
		self.frames = int((p or {}).get("m_nFrames", 0) if isinstance(p, dict) else desc.get("numframes", 0))
		self.blocks = [(int(b.get("m_nStartFrame", 0)), int(b.get("m_nEndFrame", 0)),
			kv3.ints(b.get("m_segmentIndexArray"))) for b in src.get("m_frameblockArray") or []]
		self.table = table

	def pose(self, frame: int, bind_pos: np.ndarray, bind_rot: np.ndarray):
		pos, rot = bind_pos.copy(), bind_rot.copy()
		scl = np.ones(len(pos))
		for start, end, segs in self.blocks:
			if start <= frame <= end:
				for si in segs:
					seg = self.table.get(int(si))
					if seg is not None:
						seg.apply(frame - start, pos, rot, scl)
		return pos, rot, scl


def _animations(anim_data: dict, decode_key: dict, seq_data: dict | None, bone_names: list[str]) -> list[Animation]:
	anims = anim_data.get("m_animArray") or []
	if not anims:
		return []
	table = SegmentTable(anim_data, decode_key, bone_names)
	if not seq_data:
		return [Animation(a, table) for a in anims]
	by_name = {str(a.get("m_name", "")).lower(): a for a in anims}
	names = [str(x) for x in seq_data.get("m_localSequenceNameArray") or []]
	out = []
	done: set[str] = set()
	for sd in seq_data.get("m_localS1SeqDescArray") or []:
		refs = kv3.ints((sd.get("m_fetch") or {}).get("m_localReferenceArray"))
		if not len(refs):
			continue
		r = int(refs[0])
		if not 0 <= r < len(names):
			continue
		desc = by_name.get(names[r].lower())
		if desc is None:
			continue
		f = sd.get("m_flags") or {}
		name = str(sd.get("m_sName", names[r]))
		done.add(name.lower())
		out.append(Animation(desc, table, name, bool(f.get("m_bLooping")), bool(f.get("m_bLegacyDelta"))))
	for a in anims:
		if str(a.get("m_name", "")).lower() not in done:
			out.append(Animation(a, table))
	return out


def _model_animations(res: vres.Resource, d: dict, bone_names: list[str], roots: list[Path],
		warnings: list[str]) -> list[Animation]:
	out: list[Animation] = []
	ctrl = res.keyvalues("CTRL") if res.block("CTRL") else None
	emb = ctrl.get("embedded_animation") if isinstance(ctrl, dict) else None
	if isinstance(emb, dict):
		def blk(key):
			i = int(emb.get(key, -1))
			return res.keyvalues(i) if 0 < i < len(res.blocks) else None
		group, data, seq = blk("group_data_block"), blk("anim_data_block"), blk("seqgroup_data_block")
		key = (group or {}).get("m_decodeKey") if isinstance(group, dict) else None
		if isinstance(data, dict) and isinstance(key, dict):
			out += _animations(data, key, seq if isinstance(seq, dict) else None, bone_names)
	for ref in d.get("m_refAnimGroups") or []:
		if not ref:
			continue
		gpath = vtex.find_compiled(str(ref), roots)
		if gpath is None:
			warnings.append(f"animation group {ref}: not found")
			continue
		try:
			gres = vres.load(gpath)
			gd = gres.keyvalues() or {}
			key = gd.get("m_decodeKey") or {}
			ab = gres.block("ANIM")
			if ab is not None:
				out += _animations(gres.keyvalues(ab) or {}, key, None, bone_names)
				continue
			for aref in gd.get("m_localHAnimArray") or []:
				if not aref:
					continue
				apath = vtex.find_compiled(str(aref), roots)
				if apath is None:
					warnings.append(f"animation {aref}: not found")
					continue
				out += _animations(vres.load(apath).keyvalues() or {}, key, None, bone_names)
		except (OSError, ValueError, KeyError, IndexError, struct.error) as exc:
			warnings.append(f"animation group {Path(str(ref)).name}: {exc}")
	return out


def model_sequences(res, path, d, sbones, roots, scale, opts, warnings) -> list:
	"""Studio sequences (models.Sequence) for a Source 2 model's skeleton."""
	from . import models as _m
	from . import skeleton as _sk
	from .vmdl import local_matrix

	names = [b.name for b in sbones]
	anims = _model_animations(res, d, names, roots, warnings)
	bind_pos = np.array([b.pos for b in sbones], np.float64).reshape(-1, 3)
	bind_rot = np.array([b.quat for b in sbones], np.float64).reshape(-1, 4)
	used: set[str] = set()
	out = []
	delta = failed = 0
	for a in anims:
		if a.delta:
			delta += 1
			continue
		n = max(1, a.frames)
		src_fps = a.fps if a.fps > 0 else 30.0
		fps = float(min(src_fps, opts.anim_fps)) if opts.anim_fps else src_fps
		count = max(1, int(round((n - 1) / src_fps * fps)) + 1) if n > 1 else 1
		picks = [min(n - 1, int(round(k * src_fps / fps))) for k in range(count)]
		frames = np.zeros((count, len(sbones), 6))
		try:
			for k, f in enumerate(picks):
				pos, rot, _scl = a.pose(f, bind_pos, bind_rot)
				for b in range(len(sbones)):
					frames[k, b] = _sk.pose_from_matrix(local_matrix(pos[b], rot[b], scale))
		except (ValueError, IndexError, struct.error) as exc:
			failed += 1
			if failed == 1:
				warnings.append(f"animation {a.name}: {exc}")
			continue
		out.append(_m.Sequence(_m._unique(_m._seq_name(a.name.lstrip("@")), used), fps, frames, loop=a.loop))
	if delta:
		warnings.append(f"{delta} delta (additive) animation(s) skipped")
	if failed > 1:
		warnings.append(f"{failed} animation(s) could not be decoded")
	return out
