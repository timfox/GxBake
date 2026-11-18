"""Make GoldSrc studio models smaller for the GameCube studio arena.

The port gives every map's studio models one 1 MiB MEM1 arena (c0a0 sits at
993 KiB), and the lean `gc_studio/<name>.mdl` it loads is the model minus its
skins: bones, sequences, animation and meshes. For NPCs and view models the
animation dominates, so this module works on that:

* studio_breakdown() says where a model's bytes go.
* slim_studio() rewrites the animation of sequences in the model file
  (seqgroup 0): keep every Nth frame (the engine interpolates between frames,
  fps is adjusted so each sequence lasts as long as before), or reduce
  sequences matching a pattern to a single frame. Sequences are never removed,
  so the game DLL's sequence indices and names stay valid. Events move with
  their frames. Everything after the animation is shifted and its offsets
  fixed up.
"""

from __future__ import annotations

import fnmatch
import math
import struct
from dataclasses import dataclass, field

HDR = 244
SEQ = 176
BONE = 112
BODYPART = 76
MODEL = 112
MESH = 20
TEXTURE = 80
EVENT = 76

# (count offset, index offset, element size) for header chunks.
_HEADER_CHUNKS = {
	"bones": (140, 144, BONE),
	"bonecontrollers": (148, 152, 24),
	"hitboxes": (156, 160, 32),
	"sequences": (164, 168, SEQ),
	"seqgroups": (172, 176, 104),
	"textures": (180, 184, TEXTURE),
	"bodyparts": (204, 208, BODYPART),
	"attachments": (212, 216, 88),
	"transitions": (236, 240, 1),
}
# Header fields that hold file offsets.
_HEADER_OFFSETS = (144, 152, 160, 168, 176, 184, 188, 200, 208, 216, 224, 232, 240)


def _i(raw, off: int) -> int:
	return struct.unpack_from("<i", raw, off)[0]


def _is_studio(raw: bytes) -> bool:
	return len(raw) >= HDR and raw[:4] == b"IDST" and _i(raw, 4) == 10


def _stream_len(raw: bytes, start: int, frames: int) -> int:
	"""Bytes in one mstudioanimvalue_t stream that covers `frames`."""
	p, k = start, 0
	while k < frames:
		valid, total = raw[p], raw[p + 1]
		if total == 0:
			break
		p += 2 * (valid + 1)
		k += total
	return p - start


def _expand(raw: bytes, start: int, frames: int) -> list[int]:
	"""Every frame's value of one stream (the engine's lookup, all at once)."""
	out: list[int] = []
	p = start
	while len(out) < frames:
		valid, total = raw[p], raw[p + 1]
		if total == 0 or valid == 0:
			break
		vals = struct.unpack_from(f"<{valid}h", raw, p + 2)
		out += vals
		out += [vals[-1]] * (total - valid)
		p += 2 * (valid + 1)
	out = out[:frames]
	return out + [out[-1] if out else 0] * (frames - len(out))


def _rle(q: list[int]) -> list[int]:
	from .skeleton import rle_encode

	return rle_encode(q)


@dataclass
class SeqInfo:
	index: int
	name: str
	fps: float
	frames: int
	blends: int
	group: int
	anim_off: int
	anim_bytes: int
	events: int


def sequences(raw: bytes) -> list[SeqInfo]:
	nbones = _i(raw, 140)
	nseq, seq_off = _i(raw, 164), _i(raw, 168)
	out = []
	for i in range(nseq):
		b = seq_off + i * SEQ
		name = raw[b:b + 32].split(b"\0")[0].decode("latin-1")
		fps = struct.unpack_from("<f", raw, b + 32)[0]
		frames, blends, anim, group = _i(raw, b + 56), _i(raw, b + 120), _i(raw, b + 124), _i(raw, b + 156)
		size = 0
		if group == 0 and anim > 0:
			end = anim + blends * nbones * 12
			for bl in range(blends):
				for bone in range(nbones):
					base = anim + (bl * nbones + bone) * 12
					for o in struct.unpack_from("<6H", raw, base):
						if o:
							end = max(end, base + o + _stream_len(raw, base + o, frames))
			size = end - anim
		out.append(SeqInfo(i, name, fps, frames, blends, group, anim, size, _i(raw, b + 48)))
	return out


def studio_breakdown(raw: bytes) -> dict[str, int]:
	"""Bytes per part of a studio model."""
	if not _is_studio(raw):
		raise ValueError("not a GoldSrc studio model (IDST v10)")
	d: dict[str, int] = {"header": HDR}
	for name, (cnt, idx, size) in _HEADER_CHUNKS.items():
		if name in ("textures", "transitions"):
			continue
		d[name] = _i(raw, cnt) * size
	seqs = sequences(raw)
	d["animation"] = sum(s.anim_bytes for s in seqs)
	d["events"] = sum(s.events for s in seqs) * EVENT
	mesh = 0
	nbody, body_off = _i(raw, 204), _i(raw, 208)
	for bp in range(nbody):
		nmodels, model_off = _i(raw, body_off + bp * BODYPART + 64), _i(raw, body_off + bp * BODYPART + 72)
		for m in range(nmodels):
			mb = model_off + m * MODEL
			nmesh, nverts, nnorms = _i(raw, mb + 72), _i(raw, mb + 80), _i(raw, mb + 92)
			mesh += MODEL + nmesh * MESH + nverts * 13 + nnorms * 13	# vec3 + bone byte
			mesh_off = _i(raw, mb + 76)
			for k in range(nmesh):
				tri = _i(raw, mesh_off + k * MESH + 4)
				p = tri
				while 0 < p + 2 <= len(raw):
					n = struct.unpack_from("<h", raw, p)[0]
					p += 2
					if n == 0:
						break
					p += abs(n) * 8
				mesh += p - tri
	d["meshes"] = mesh
	ntex, tex_off = _i(raw, 180), _i(raw, 184)
	d["skins"] = 0
	for t in range(ntex):
		_f, w, h, _idx = struct.unpack_from("<4i", raw, tex_off + t * TEXTURE + 64)
		d["skins"] += TEXTURE + w * h + 768
	d["skin tables"] = _i(raw, 192) * _i(raw, 196) * 2
	d["total"] = len(raw)
	d["other"] = len(raw) - sum(v for k, v in d.items() if k != "total")
	return d


@dataclass
class SlimResult:
	data: bytes
	before: int
	after: int
	changed: list[tuple[str, int, int]] = field(default_factory=list)	# name, old frames, new frames
	skipped: list[str] = field(default_factory=list)


def deadband(q: list[int], tol: int) -> list[int]:
	"""Hold each value until the curve moves more than `tol` away from it.

	Studio animation is stored as runs (mstudioanimvalue_t: a count of
	distinct values and a count of frames, the last value repeating), so a
	bone that jitters by a unit or two between frames costs two bytes a frame
	while a held value costs nothing. Holding within ±tol makes those runs
	long; no frame is ever further than tol from the original, and the engine
	reads the result like any other stream.
	"""
	if tol <= 0 or not q:
		return q
	out, held = [], q[0]
	for v in q:
		if abs(v - held) > tol:
			held = v
		out.append(held)
	if all(abs(v) <= tol for v in q):	# a stream that never leaves zero is dropped
		return [0] * len(q)
	return out


def slim_studio(raw: bytes, step: int = 1, stub: list[str] | None = None, keep: list[str] | None = None,
		min_frames: int = 2, tolerance: int = 0) -> SlimResult:
	"""Decimate (every `step`th frame) or stub (patterns → 1 frame) the
	animation of seqgroup-0 sequences, and hold values within `tolerance`
	(raw animation units, see deadband). `keep` patterns are left alone."""
	if not _is_studio(raw):
		raise ValueError("not a GoldSrc studio model (IDST v10)")
	stub, keep = stub or [], keep or []
	nbones = _i(raw, 140)
	seqs = [s for s in sequences(raw)]
	inline = [s for s in seqs if s.group == 0 and s.anim_off > 0 and s.anim_bytes > 0]
	res = SlimResult(raw, len(raw), len(raw))
	res.skipped = [s.name for s in seqs if s.group != 0]
	if not inline:
		return res
	start = min(s.anim_off for s in inline)
	end = max(s.anim_off + s.anim_bytes for s in inline)
	end += -end & 3

	# Refuse layouts where something else lives between the animation blocks.
	others = [_i(raw, o) for o in _HEADER_OFFSETS]
	seq_off = _i(raw, 168)
	for s in seqs:
		b = seq_off + s.index * SEQ
		others += [_i(raw, b + 52), _i(raw, b + 64)]
	if any(start <= o < end for o in others if o > 0) or start <= seq_off < end:
		raise ValueError("animation is interleaved with other data; this layout is not supported")

	def match(name: str, pats: list[str]) -> bool:
		return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in pats)

	blocks: list[tuple[SeqInfo, bytes, int, float]] = []	# seq, new block, new frames, new fps
	for s in sorted(inline, key=lambda s: s.anim_off):
		n = s.frames
		tol = 0 if match(s.name, keep) else tolerance
		if match(s.name, keep) or n <= min_frames:
			picks = list(range(n))
		elif match(s.name, stub):
			picks = [0]
		elif step > 1:
			count = max(min_frames, math.ceil((n - 1) / step) + 1)
			picks = sorted({min(n - 1, k * step) for k in range(count)} | {n - 1})
		else:
			picks = list(range(n))
		new_n = len(picks)
		head = bytearray(s.blends * nbones * 12)
		data = bytearray()
		for bl in range(s.blends):
			for bone in range(nbones):
				src = s.anim_off + (bl * nbones + bone) * 12
				hpos = (bl * nbones + bone) * 12
				offs = [0] * 6
				for j, o in enumerate(struct.unpack_from("<6H", raw, src)):
					if not o:
						continue
					full = _expand(raw, src + o, n)
					q = deadband([full[k] for k in picks], tol)
					if not any(q):
						continue
					rel = len(head) + len(data) - hpos
					if rel > 0xFFFF:
						raise ValueError(f"{s.name}: animation offsets overflow")
					offs[j] = rel
					stream = _rle(q)
					data += struct.pack(f"<{len(stream)}h", *stream)
				struct.pack_into("<6H", head, hpos, *offs)
		block = bytes(head + data)
		block += bytes(-len(block) & 3)
		fps = s.fps
		if new_n != n and new_n > 1 and n > 1:
			fps = s.fps * (new_n - 1) / (n - 1)	# same duration
		blocks.append((s, block, new_n, fps))
		if new_n != n:
			res.changed.append((s.name, n, new_n))

	if not res.changed and sum(len(b[1]) for b in blocks) >= end - start:
		return res	# nothing decimated and re-encoding (optimal runs, deadband) saved nothing

	anim = bytearray()
	new_off: dict[int, int] = {}
	for s, block, _n, _fps in blocks:
		new_off[s.index] = start + len(anim)
		anim += block
	delta = len(anim) - (end - start)
	out = bytearray(raw[:start]) + anim + bytearray(raw[end:])

	def fix(pos: int) -> None:
		"""Shift the offset stored at pos (in out) if it pointed past the old region."""
		v = _i(out, pos)
		if v >= end:
			struct.pack_into("<i", out, pos, v + delta)

	# Header offsets, then structures that contain offsets.
	for o in _HEADER_OFFSETS:
		fix(o)
	struct.pack_into("<i", out, 72, len(out))
	nseq, seq_base = _i(out, 164), _i(out, 168)
	frames_of = {s.index: (n, fps, s.frames) for s, _b, n, fps in blocks}
	for i in range(nseq):
		b = seq_base + i * SEQ
		fix(b + 52)	# eventindex
		fix(b + 64)	# pivotindex
		if i in new_off:
			struct.pack_into("<i", out, b + 124, new_off[i])
			new_n, fps, old_n = frames_of[i]
			struct.pack_into("<f", out, b + 32, fps)
			struct.pack_into("<i", out, b + 56, new_n)
			nev, ev = _i(out, b + 48), _i(out, b + 52)
			for e in range(nev):
				fr = _i(out, ev + e * EVENT)
				if old_n > 1 and new_n != old_n:
					struct.pack_into("<i", out, ev + e * EVENT, int(round(fr * (new_n - 1) / (old_n - 1))))
		elif _i(out, b + 156) == 0:
			fix(b + 124)
	nbody, body = _i(out, 204), _i(out, 208)
	for bp in range(nbody):
		pb = body + bp * BODYPART
		fix(pb + 72)
		nmodels, model_off = _i(out, pb + 64), _i(out, pb + 72)
		for m in range(nmodels):
			mb = model_off + m * MODEL
			for o in (76, 84, 88, 96, 100, 108):
				fix(mb + o)
			nmesh, mesh_off = _i(out, mb + 72), _i(out, mb + 76)
			for k in range(nmesh):
				fix(mesh_off + k * MESH + 4)	# triindex (normindex is an index, not an offset)
	ntex, tex = _i(out, 180), _i(out, 184)
	for t in range(ntex):
		fix(tex + t * TEXTURE + 76)
	res.data = bytes(out)
	res.after = len(out)
	return res
