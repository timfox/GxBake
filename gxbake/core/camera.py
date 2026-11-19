"""Camera paths → .gcam: compact, big-endian keyframe tracks that the GameCube port samples
and hands straight to GX.

Sources:
	.gltf / .glb      every camera node, per animation (parent rigs and dollies included),
	                  perspective yfov/znear/zfar/aspectRatio, KHR_animation_pointer yfov
	.dae / .fbx / …   through Assimp (gxbake[models]): the camera node's animation; FOV from the
	                  Collada <optics> block, otherwise --fov
	.json / .csv      hand-written or exported keys (see docs/camera.md, which has a Blender script)
	.bsp / .ent       GoldSrc trigger_camera + path_corner chains, as a straight-line approximation
	                  of the game's moving camera

Everything is converted to the engine's frame: Z-up world in GoldSrc units, camera looking
down +X with +Y to its left and +Z up (what AngleVectors gives for pitch/yaw/roll).

Each clip is sampled on a fixed grid (`rate` Hz, 60 by default), then every track
(position, rotation, vertical FOV) is cut down to the fewest keys whose curve stays inside
its tolerance. Each track's curve is the one port/gc_cam.c evaluates: linear, or "smooth"
(Catmull-Rom/Hermite with finite-difference tangents, rotations as normalised quaternion
components). Two keys at the same time are a cut: the curve jumps there, and tangents never
reach across it. Keys are stored as f32, or ("fixed") as s16 with a per-track binary point
and u16 times in ticks of 1/rate, which is about half the size.

File layout (big-endian, offsets from the start of the file, every block 4-aligned):

	header    16 B   "GCAM", u16 version (1), u16 clip count, u32 file size, u32 0
	clips     64 B each:
	          char name[32], u32 body offset, u32 body size, f32 duration (s), f32 near,
	          f32 far, f32 aspect (0 = the screen's), f32 tick rate (Hz),
	          u8 0, u8 flags (1 loop, 2 fixed), u8 track count, u8 0
	body      per track 16 B: u8 kind (0 position xyz, 1 rotation quaternion xyzw, 2 fov_y in
	          degrees), u8 components, u8 frac bits (fixed), u8 interp (0 linear, 1 smooth),
	          u16 keys, u16 0,
	          u32 times offset, u32 values offset (both from the body start);
	          then the times (u16 ticks or f32 seconds) and values (s16 or f32), each padded to 4
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"GCAM"
VERSION = 1
HEADER = 16
CLIP_ENTRY = 64
TRACK_ENTRY = 16
TRACK_POS, TRACK_ROT, TRACK_FOV = 0, 1, 2
INTERP = {"linear": 0, "smooth": 1}
INTERPS = ("auto", "smooth", "linear")
FLAG_LOOP, FLAG_FIXED = 1, 2
ENCODINGS = ("fixed", "f32")
AXES = ("auto", "engine", "gltf", "blender", "fbx")

EXTENSIONS_NATIVE = {".gltf", ".glb", ".json", ".csv", ".bsp", ".ent"}
EXTENSIONS_ASSIMP = {".dae", ".fbx", ".abc", ".3ds", ".x", ".ms3d", ".blend", ".ogex", ".usd", ".usda"}
EXTENSIONS = EXTENSIONS_NATIVE | EXTENSIONS_ASSIMP

# GoldSrc defaults: r_znear-like near plane, sv_zmax far plane, 90° horizontal at 4:3.
DEFAULT_NEAR = 4.0
DEFAULT_FAR = 4096.0
DEFAULT_FOV_Y = 73.739795	# = 90° horizontal at 4:3


def fov_x_to_y(fov_x: float, aspect: float = 4 / 3) -> float:
	return math.degrees(2 * math.atan(math.tan(math.radians(fov_x) / 2) / aspect))


def fov_y_to_x(fov_y: float, aspect: float = 4 / 3) -> float:
	return math.degrees(2 * math.atan(math.tan(math.radians(fov_y) / 2) * aspect))


# -- quaternions (x, y, z, w) and frames ------------------------------------------------------------


def quat_from_matrix(m: "np.ndarray") -> "np.ndarray":
	"""Rotation matrix (columns = local axes in world) → unit quaternion x, y, z, w."""
	m = np.asarray(m, np.float64)[:3, :3]
	tr = m[0, 0] + m[1, 1] + m[2, 2]
	if tr > 0:
		s = math.sqrt(tr + 1.0) * 2
		q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
	elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
		s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
		q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
	elif m[1, 1] > m[2, 2]:
		s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
		q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
	else:
		s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
		q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
	q = np.array(q)
	return q / np.linalg.norm(q)


def quat_to_matrix(q: "np.ndarray") -> "np.ndarray":
	x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
	return np.array([
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def angles_to_matrix(pitch: float, yaw: float, roll: float) -> "np.ndarray":
	"""GoldSrc angles (degrees; positive pitch looks down) → columns forward, left, up."""
	p, y, r = (math.radians(a) for a in (pitch, yaw, roll))
	sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
	fwd = np.array([cp * cy, cp * sy, -sp])
	right = np.array([-sr * sp * cy + cr * sy, -sr * sp * sy - cr * cy, -sr * cp])
	up = np.array([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp])
	return np.stack([fwd, -right, up], 1)


def matrix_to_angles(m: "np.ndarray") -> tuple[float, float, float]:
	"""Columns forward, left, up → GoldSrc pitch, yaw, roll in degrees (what gc_cam_angles returns)."""
	fwd, left, up = m[:, 0], m[:, 1], m[:, 2]
	pitch = -math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
	yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
	roll = math.degrees(math.atan2(left[2], up[2]))
	return pitch, yaw, roll


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> "np.ndarray":
	"""Engine-frame camera matrix looking from eye at target (no roll)."""
	f = np.asarray(target, np.float64) - np.asarray(eye, np.float64)
	n = np.linalg.norm(f)
	if n < 1e-9:
		return np.eye(3)
	f /= n
	left = np.cross(np.asarray(up, np.float64), f)
	if np.linalg.norm(left) < 1e-6:	# straight up or down
		left = np.cross([1.0, 0.0, 0.0] if abs(f[0]) < 0.9 else [0.0, 1.0, 0.0], f)
	left /= np.linalg.norm(left)
	return np.stack([f, left, np.cross(f, left)], 1)


# world axes of each frame → engine world (rotation only), and the camera's own axes:
# columns = where engine forward / left / up point in the source camera's local space.
_WORLD = {
	"engine": np.eye(3),
	"blender": np.eye(3),
	# Y-up (glTF, Maya, FBX, Collada as Assimp hands it) → Z-up, the inverse of Blender's glTF export
	"gltf": np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
}
_WORLD["fbx"] = _WORLD["gltf"]
_CAM = {
	"engine": np.eye(3),
	"gltf": np.array([[0.0, -1, 0], [0, 0, 1], [-1, 0, 0]]),	# looks down -Z, +Y up
	"fbx": np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),	# looks down +X, +Y up
}
_CAM["blender"] = _CAM["gltf"]


@dataclass
class Frame:
	"""How source coordinates map into the engine: axes, scale, a turn about Z, an offset."""
	world: str = "engine"
	camera: str = "engine"
	scale: float = 1.0
	yaw: float = 0.0
	offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

	def rotation(self) -> "np.ndarray":
		c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
		return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ _WORLD[self.world]

	def point(self, p) -> "np.ndarray":
		return self.rotation() @ (np.asarray(p, np.float64) * self.scale) + np.asarray(self.offset, np.float64)

	def orient(self, m) -> "np.ndarray":
		"""Source camera rotation (local axes as columns, in source world) → engine camera matrix."""
		r = self.rotation() @ np.asarray(m, np.float64)[:3, :3] @ _CAM[self.camera]
		u, _s, vt = np.linalg.svd(r)	# drop any scale/shear
		return u @ vt


# -- the camera path ---------------------------------------------------------------------------------


@dataclass
class CameraPath:
	"""A clip in engine space. Equal consecutive times mark a cut."""
	name: str
	times: "np.ndarray"	# (n,) seconds, non-decreasing
	pos: "np.ndarray"	# (n, 3)
	rot: "np.ndarray"	# (n, 4) x, y, z, w
	fov: "np.ndarray"	# (n,) vertical degrees
	near: float = DEFAULT_NEAR
	far: float = DEFAULT_FAR
	aspect: float = 0.0
	loop: bool = False
	source: str = ""
	notes: list[str] = field(default_factory=list)

	@property
	def duration(self) -> float:
		return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0


def _hemisphere(q: "np.ndarray") -> "np.ndarray":
	q = np.array(q, np.float64)
	for i in range(1, len(q)):
		if np.dot(q[i], q[i - 1]) < 0:
			q[i] = -q[i]
	return q


# -- evaluation (mirrors port/gc_cam.c) ----------------------------------------------------------------


def _tangents(t: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
	"""Finite-difference slopes per unit time; one-sided at the ends and next to a cut."""
	n = len(t)
	m = np.zeros_like(v)
	for i in range(n):
		prev = i > 0 and t[i - 1] < t[i]
		nxt = i + 1 < n and t[i + 1] > t[i]
		if prev and nxt:
			m[i] = (v[i + 1] - v[i - 1]) / (t[i + 1] - t[i - 1])
		elif nxt:
			m[i] = (v[i + 1] - v[i]) / (t[i + 1] - t[i])
		elif prev:
			m[i] = (v[i] - v[i - 1]) / (t[i] - t[i - 1])
	return m


def evaluate(t: "np.ndarray", v: "np.ndarray", at: "np.ndarray", interp: str = "smooth") -> "np.ndarray":
	"""Sample a key track (times t, values v (n, c)) at times `at`, exactly as gc_cam.c does: before
	the first key → the first value, at or after the last → the last; otherwise the span starting at
	the last key whose time is <= `at` (so at a cut the later key wins)."""
	t = np.asarray(t, np.float64)
	v = np.asarray(v, np.float64).reshape(len(t), -1)
	at = np.asarray(at, np.float64)
	if len(t) == 1:
		return np.repeat(v, len(at), 0)
	i = np.clip(np.searchsorted(t, at, side="right") - 1, 0, len(t) - 2)
	span = t[i + 1] - t[i]
	ok = span > 0
	u = np.where(ok, (at - t[i]) / np.where(ok, span, 1.0), 0.0)[:, None]
	if interp == "linear":
		out = v[i] + (v[i + 1] - v[i]) * u
	else:
		m = _tangents(t, v)
		u2, u3 = u * u, u * u * u
		dt = span[:, None]
		out = ((2 * u3 - 3 * u2 + 1) * v[i] + (u3 - 2 * u2 + u) * dt * m[i]
			+ (-2 * u3 + 3 * u2) * v[i + 1] + (u3 - u2) * dt * m[i + 1])
	out = np.where((at < t[0])[:, None], v[0], out)
	return np.where((at >= t[-1])[:, None], v[-1], out)


def _rot_error(q_eval: "np.ndarray", q_true: "np.ndarray") -> "np.ndarray":
	qe = q_eval / np.linalg.norm(q_eval, axis=1, keepdims=True)
	d = np.abs(np.sum(qe * q_true, axis=1))
	return np.degrees(2 * np.arccos(np.clip(d, -1.0, 1.0)))


def _error(kind: int, ev: "np.ndarray", truth: "np.ndarray") -> "np.ndarray":
	if kind == TRACK_ROT:
		return _rot_error(ev, truth)
	if kind == TRACK_POS:
		return np.linalg.norm(ev - truth, axis=1)
	return np.abs(ev - truth).max(axis=1)


# -- sampling and key reduction ------------------------------------------------------------------------


def resample(path: CameraPath, rate: float, interp: str = "smooth") -> CameraPath:
	"""Evaluate a (possibly sparse) path on a 1/rate grid. Each stretch between cuts is evaluated
	on its own, so nothing interpolates across a cut; the cut lands on its nearest tick, sampled
	twice (end of one shot, start of the next)."""
	t = np.asarray(path.times, np.float64) - float(path.times[0])
	rot = _hemisphere(path.rot)
	fov = np.asarray(path.fov, np.float64)[:, None]
	bounds = [0, *[i for i in range(1, len(t)) if t[i] == t[i - 1]], len(t)]
	parts: list[tuple] = []
	for a, b in zip(bounds, bounds[1:]):
		lo = int(round(t[a] * rate))
		hi = int(round((t[b] if b < len(t) else t[-1]) * rate))
		ts = np.arange(lo, hi + 1) / rate
		rr = evaluate(t[a:b], rot[a:b], ts, interp)
		parts.append((ts, evaluate(t[a:b], path.pos[a:b], ts, interp), rr / np.linalg.norm(rr, axis=1, keepdims=True),
			evaluate(t[a:b], fov[a:b], ts, interp)[:, 0]))
	return CameraPath(path.name, np.concatenate([p[0] for p in parts]), np.vstack([p[1] for p in parts]),
		_hemisphere(np.vstack([p[2] for p in parts])), np.concatenate([p[3] for p in parts]),
		path.near, path.far, path.aspect, path.loop, path.source, list(path.notes))


def reduce_track(t: "np.ndarray", v: "np.ndarray", kind: int, tol: float, interp: str,
		max_keys: int = 65535, seed: "list[int] | None" = None) -> "np.ndarray":
	"""Indices of the fewest samples (greedy refinement) whose curve stays within `tol` of every
	sample. `seed` samples are always kept (a sparse source's own keys)."""
	n = len(t)
	if n <= 2:
		return np.arange(n)
	keep = {0, n - 1, *(seed or [])}
	for i in range(1, n):
		if t[i] == t[i - 1]:	# both sides of a cut
			keep.update((i - 1, i))
	if kind == TRACK_ROT:
		v = v / np.linalg.norm(v, axis=1, keepdims=True)
	forced = set(keep)

	def worst(idx: "np.ndarray") -> "np.ndarray":
		err = _error(kind, evaluate(t[idx], v[idx], t, interp), v)
		err[idx] = 0
		return err

	while True:
		idx = np.array(sorted(keep))
		err = worst(idx)
		if err.max() <= tol or len(idx) >= min(n, max_keys):
			break
		# add the worst sample of every span that is out of tolerance
		seg = np.searchsorted(idx, np.arange(n), side="right") - 1
		add: dict[int, int] = {}
		for s in np.unique(seg[err > tol]):
			members = np.nonzero(seg == s)[0]
			add[int(s)] = int(members[np.argmax(err[members])])
		keep.update(add.values())
	# greedy refinement over-inserts: drop each key the curve can do without (one pass)
	for k in sorted(keep - forced, key=lambda k: -k):
		trial = np.array(sorted(keep - {k}))
		if worst(trial).max() <= tol:
			keep.discard(k)
	return np.array(sorted(keep))


# -- encoding -----------------------------------------------------------------------------------------


@dataclass
class Track:
	kind: int
	times: "np.ndarray"	# seconds
	values: "np.ndarray"	# (n, c)
	frac: int = 0
	interp: str = "smooth"


@dataclass
class Encoded:
	path: CameraPath
	fixed: bool
	rate: float
	tracks: list[Track]
	body: bytes
	max_error: dict[str, float]
	samples: int


def _frac_for(values: "np.ndarray") -> int:
	peak = float(np.abs(values).max()) if values.size else 0.0
	f = 15
	while f > 0 and round(peak * (1 << f)) > 32767:
		f -= 1
	if round(peak * (1 << f)) > 32767:
		raise ValueError(f"value {peak:.1f} does not fit s16 even with 0 fraction bits; use encoding f32")
	return f


def _pad4(b: bytearray) -> None:
	while len(b) % 4:
		b.append(0)


def _quantise(track: Track, fixed: bool, rate: float) -> tuple[bytes, bytes, Track]:
	"""(times bytes, values bytes, the track as the port will decode it)."""
	n, c = track.values.shape
	if fixed:
		ticks = np.round(track.times * rate).astype(np.int64)
		if ticks.max(initial=0) > 0xFFFF:
			raise ValueError(f"{track.times[-1]:.1f} s is {ticks.max()} ticks at {rate:g} Hz, more than u16 holds; "
				"lower the rate or use encoding f32")
		q = np.clip(np.round(track.values * (1 << track.frac)), -32768, 32767).astype(">i2")
		tb = ticks.astype(">u2").tobytes()
		vb = q.tobytes()
		dec = Track(track.kind, ticks / rate, q.astype(np.float64) / (1 << track.frac), track.frac, track.interp)
	else:
		t32 = track.times.astype(">f4")
		v32 = track.values.astype(">f4")
		tb, vb = t32.tobytes(), v32.tobytes()
		dec = Track(track.kind, t32.astype(np.float64), v32.astype(np.float64), 0, track.interp)
	return tb, vb, dec


def encode_clip(path: CameraPath, rate: float = 60.0, interp: str = "auto", encoding: str = "fixed",
		pos_tol: float = 0.25, rot_tol: float = 0.15, fov_tol: float = 0.05, dense: bool = False) -> Encoded:
	"""Resample (unless `dense`: already on the 1/rate grid), reduce and serialise one clip body.
	interp "auto" reduces each track both ways and keeps whichever needs fewer keys (a sparse
	source's own keys are then read as a smooth curve)."""
	if interp not in INTERPS:
		raise ValueError(f"interp is one of {', '.join(INTERPS)}")
	if encoding not in ENCODINGS:
		raise ValueError(f"encoding is one of {', '.join(ENCODINGS)}")
	fixed = encoding == "fixed"
	seed = None
	d = path
	t = d.times - d.times[0]
	if not dense:
		d = resample(path, rate, "smooth" if interp == "auto" else interp)
		t = d.times - d.times[0]
		src = np.round((np.asarray(path.times) - path.times[0]) * rate) / rate	# keep the source's own keys
		seed = sorted({int(np.searchsorted(t, x - 1e-9)) for x in src} & set(range(len(t))))
	left_limits = np.nonzero(np.diff(t) == 0)[0]
	sources = [(TRACK_POS, d.pos, pos_tol), (TRACK_ROT, d.rot, rot_tol), (TRACK_FOV, d.fov[:, None], fov_tol)]
	tracks, blobs = [], []
	max_err = {}
	for kind, vals, tol in sources:
		vals = np.asarray(vals, np.float64)
		frac = 14 if kind == TRACK_ROT else _frac_for(vals) if fixed else 0
		# leave room in the tolerance for quantisation
		q_step = (1.0 / (1 << frac)) if fixed else 0.0
		budget = tol
		if fixed and kind == TRACK_POS:
			budget = max(tol * 0.5, tol - q_step * 0.9)
		elif fixed and kind == TRACK_FOV:
			budget = max(tol * 0.5, tol - q_step * 0.5)
		best = None
		for mode in (("smooth", "linear") if interp == "auto" else (interp,)):
			idx = reduce_track(t, vals, kind, budget, mode, seed=seed if mode == "smooth" or interp != "auto" else None)
			tb, vb, dec = _quantise(Track(kind, t[idx], vals[idx], frac, mode), fixed, rate)
			if best is None or len(idx) < len(best[2].times):
				best = (tb, vb, dec)
		tb, vb, dec = best
		err = _error(kind, evaluate(dec.times, dec.values, t, dec.interp), vals)
		err[left_limits] = 0	# the last sample before a cut is only ever approached, never shown
		max_err[("pos", "rot", "fov")[kind]] = float(err.max()) if len(err) else 0.0
		tracks.append(dec)
		blobs.append((tb, vb))
	body = bytearray(TRACK_ENTRY * len(tracks))
	for k, (tr, (tb, vb)) in enumerate(zip(tracks, blobs)):
		t_off = len(body)
		body += tb
		_pad4(body)
		v_off = len(body)
		body += vb
		_pad4(body)
		struct.pack_into(">BBBBHHII", body, k * TRACK_ENTRY, tr.kind, tr.values.shape[1], tr.frac, INTERP[tr.interp],
			len(tr.times), 0, t_off, v_off)
	return Encoded(d, fixed, rate, tracks, bytes(body), max_err, len(t))


def clip_name(name: str) -> str:
	return (re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "camera")[:31]


def write_gcam(clips: list[Encoded]) -> bytes:
	names: set[str] = set()
	out = bytearray(HEADER + CLIP_ENTRY * len(clips))
	out[:4] = MAGIC
	for i, e in enumerate(clips):
		name = clip_name(e.path.name)
		base, k = name, 2
		while name in names:
			name = f"{base[:28]}_{k}"
			k += 1
		names.add(name)
		_pad4(out)
		off = len(out)
		out += e.body
		flags = (FLAG_LOOP if e.path.loop else 0) | (FLAG_FIXED if e.fixed else 0)
		struct.pack_into(">32sIIfffffBBBB", out, HEADER + i * CLIP_ENTRY, name.encode("ascii"), off, len(e.body),
			e.path.duration, e.path.near, e.path.far, e.path.aspect, e.rate, 0, flags, len(e.tracks), 0)
	_pad4(out)
	struct.pack_into(">4sHHII", out, 0, MAGIC, VERSION, len(clips), len(out), 0)
	return bytes(out)


@dataclass
class GcamClip:
	name: str
	duration: float
	near: float
	far: float
	aspect: float
	rate: float
	loop: bool
	fixed: bool
	tracks: dict[int, Track]
	bytes: int

	def sample(self, at) -> dict[str, "np.ndarray"]:
		"""Pose at times `at` (seconds), as gc_cam_sample computes it (loops wrap)."""
		at = np.atleast_1d(np.asarray(at, np.float64))
		if self.loop and self.duration > 0:
			at = np.mod(at, self.duration)
		names = {TRACK_POS: "pos", TRACK_ROT: "rot", TRACK_FOV: "fov"}
		out = {}
		for kind, tr in self.tracks.items():
			v = evaluate(tr.times, tr.values, at, tr.interp)
			if kind == TRACK_ROT:
				v = v / np.linalg.norm(v, axis=1, keepdims=True)
			out[names[kind]] = v[:, 0] if kind == TRACK_FOV else v
		return out


def read_gcam(data: bytes) -> list[GcamClip]:
	if len(data) < HEADER or data[:4] != MAGIC:
		raise ValueError("not a .gcam file")
	_m, ver, count, size, _r = struct.unpack_from(">4sHHII", data, 0)
	if ver != VERSION:
		raise ValueError(f".gcam version {ver} (this GxBake reads {VERSION})")
	if size > len(data) or HEADER + count * CLIP_ENTRY > len(data):
		raise ValueError(".gcam is truncated")
	clips = []
	for i in range(count):
		raw_name, off, blen, dur, near, far, aspect, rate, _r0, flags, ntr, _p = struct.unpack_from(
			">32sIIfffffBBBB", data, HEADER + i * CLIP_ENTRY)
		if off + blen > len(data):
			raise ValueError(".gcam clip runs past the end of the file")
		body = data[off:off + blen]
		fixed = bool(flags & FLAG_FIXED)
		tracks = {}
		for k in range(ntr):
			kind, comps, frac, interp, n, _b, t_off, v_off = struct.unpack_from(">BBBBHHII", body, k * TRACK_ENTRY)
			if t_off + n * (2 if fixed else 4) > blen or v_off + n * comps * (2 if fixed else 4) > blen:
				raise ValueError(f".gcam clip {i} track {k} runs past its body")
			if fixed:
				ticks = np.frombuffer(body, ">u2", n, t_off).astype(np.float64)
				vals = np.frombuffer(body, ">i2", n * comps, v_off).astype(np.float64) / (1 << frac)
				times = ticks / rate
			else:
				times = np.frombuffer(body, ">f4", n, t_off).astype(np.float64)
				vals = np.frombuffer(body, ">f4", n * comps, v_off).astype(np.float64)
			tracks[kind] = Track(kind, times, vals.reshape(n, comps), frac, "smooth" if interp == 1 else "linear")
		clips.append(GcamClip(raw_name.rstrip(b"\0").decode("ascii", "replace"), dur, near, far, aspect, rate,
			bool(flags & FLAG_LOOP), fixed, tracks, blen))
	return clips


# -- options ------------------------------------------------------------------------------------------


@dataclass
class CameraOptions:
	rate: float = 60.0	# sample grid (and fixed-point tick rate), Hz
	interp: str = "auto"	# auto (per track, whichever needs fewer keys) | linear | smooth
	encoding: str = "fixed"	# fixed (s16 + u16 ticks) | f32
	pos_tol: float = 0.25	# GoldSrc units
	rot_tol: float = 0.15	# degrees
	fov_tol: float = 0.05	# degrees
	axes: str = "auto"	# auto | engine | gltf | blender | fbx
	scale: float | None = None	# source units → GoldSrc units (None: 39.37 for glTF/metres, else 1)
	yaw: float = 0.0
	offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
	fov_y: float | None = None	# fallback/override vertical FOV in degrees
	near: float | None = None
	far: float | None = None
	aspect: float | None = None
	loop: bool | None = None
	node: str = ""	# Assimp: the camera node's name
	clips: tuple[str, ...] = ()	# only these clip names (fnmatch patterns)
	max_bytes: int = 0	# loosen tolerances until the clips fit (0 = no limit)


def _frame_for(path: Path, opts: CameraOptions, default: str) -> Frame:
	name = default if opts.axes == "auto" else opts.axes
	if name not in _WORLD:
		raise ValueError(f"axes is one of {', '.join(AXES)}")
	scale = opts.scale if opts.scale is not None else (39.37 if name in ("gltf", "fbx", "blender") else 1.0)
	return Frame(name, name, scale, opts.yaw, tuple(opts.offset))


def _finish(path: CameraPath, opts: CameraOptions) -> CameraPath:
	if opts.fov_y is not None:
		path.fov = np.full(len(path.times), float(opts.fov_y))
	if opts.near is not None:
		path.near = float(opts.near)
	if opts.far is not None:
		path.far = float(opts.far)
	if opts.aspect is not None:
		path.aspect = float(opts.aspect)
	if opts.loop is not None:
		path.loop = bool(opts.loop)
	if path.far <= path.near:
		raise ValueError(f"{path.name}: far ({path.far}) must be beyond near ({path.near})")
	return path


# -- JSON and CSV ---------------------------------------------------------------------------------------


def _key_rotation(k: dict, frame: Frame, pos: "np.ndarray") -> "np.ndarray":
	if "rot" in k or "quat" in k:
		q = np.asarray(k.get("rot", k.get("quat")), np.float64)
		return frame.orient(quat_to_matrix(q))
	if "matrix" in k:
		m = np.asarray(k["matrix"], np.float64).reshape(4, 4)
		return frame.orient(m[:3, :3])
	if "look_at" in k or "target" in k:
		return look_at(pos, frame.point(k.get("look_at", k.get("target"))))
	if "angles" in k or "euler" in k:
		a = k.get("angles", k.get("euler"))
		return frame.rotation() @ angles_to_matrix(*a) if frame.world != "engine" else angles_to_matrix(*a)
	raise ValueError("every key needs rot (x,y,z,w), matrix, angles (pitch,yaw,roll) or look_at")


def _keys_to_path(name: str, keys: list[dict], frame: Frame, meta: dict, source: str) -> CameraPath:
	if not keys:
		raise ValueError(f"{name}: no keys")
	fps = float(meta.get("fps", 0) or 0)
	rows = []
	last_rot = None
	last_fov = float(meta.get("fov_y", fov_x_to_y(meta["fov_x"], meta.get("aspect") or 4 / 3)) if "fov_x" in meta
		else meta.get("fov_y", meta.get("fov", DEFAULT_FOV_Y)))
	for i, k in enumerate(keys):
		t = float(k["t"]) if "t" in k else float(k["time"]) if "time" in k else float(k["frame"]) / fps if fps and "frame" in k else None
		if t is None:
			raise ValueError(f"{name}: key {i} has no t (seconds) or frame (with fps)")
		p = frame.point(k.get("pos", k.get("position", [0, 0, 0])))
		try:
			r = _key_rotation(k, frame, p)
		except ValueError:
			if last_rot is None:
				raise
			r = last_rot
		last_rot = r
		if "fov_x" in k:
			last_fov = fov_x_to_y(float(k["fov_x"]), float(meta.get("aspect") or 4 / 3))
		elif "lens" in k:
			last_fov = math.degrees(2 * math.atan(float(meta.get("sensor_height", 24.0)) / (2 * float(k["lens"]))))
		elif "fov_y" in k or "fov" in k:
			last_fov = float(k.get("fov_y", k.get("fov")))
		rows.append((t, bool(k.get("cut", False)), p, quat_from_matrix(r), last_fov))
	rows.sort(key=lambda r: r[0])
	times, pos, rot, fov = [], [], [], []
	for t, cut, p, q, f in rows:
		if cut and times and times[-1] < t:	# hold the previous shot until the cut
			times.append(t)
			pos.append(pos[-1])
			rot.append(rot[-1])
			fov.append(fov[-1])
		times.append(t)
		pos.append(p)
		rot.append(q)
		fov.append(f)
	near = float(meta["near"]) * frame.scale if "near" in meta else DEFAULT_NEAR
	far = float(meta["far"]) * frame.scale if "far" in meta else DEFAULT_FAR
	return CameraPath(name, np.array(times), np.array(pos), _hemisphere(np.array(rot)), np.array(fov, np.float64),
		near, far, float(meta.get("aspect", 0) or 0),
		bool(meta.get("loop", False)), source)


def load_json(path: Path, opts: CameraOptions) -> list[CameraPath]:
	doc = json.loads(Path(path).read_text(encoding="utf-8"))
	cams = doc.get("cameras", [doc]) if isinstance(doc, dict) else doc
	out = []
	for i, cam in enumerate(cams):
		meta = {**{k: v for k, v in doc.items() if k != "cameras"}, **cam} if isinstance(doc, dict) else cam
		ax = meta.get("axes", "engine")
		o = CameraOptions(**{**opts.__dict__, "axes": ax if opts.axes == "auto" else opts.axes,
			"scale": opts.scale if opts.scale is not None else float(meta["scale"]) if "scale" in meta else None})
		frame = _frame_for(path, o, "engine")
		name = cam.get("name") or (Path(path).stem if len(cams) == 1 else f"{Path(path).stem}_{i}")
		out.append(_keys_to_path(name, cam.get("keys", []), frame, meta, str(path)))
	return out


_CSV_ALIASES = {"time": "t", "seconds": "t", "px": "x", "py": "y", "pz": "z", "fov": "fov_y"}


def load_csv(path: Path, opts: CameraOptions) -> list[CameraPath]:
	"""Columns: t (or frame + --rate as fps), x y z, then qx qy qz qw | pitch yaw roll | tx ty tz,
	optionally fov_y / fov_x / lens, cut, name (one clip per name)."""
	text = Path(path).read_text(encoding="utf-8-sig")
	rows = list(csv.DictReader(io.StringIO(text)))
	if not rows:
		raise ValueError(f"{path}: no rows")
	frame = _frame_for(path, opts, "engine")
	clips: dict[str, list[dict]] = {}
	for r in rows:
		r = {_CSV_ALIASES.get(k.strip().lower(), k.strip().lower()): (v or "").strip() for k, v in r.items() if k}
		key: dict = {}
		if r.get("t"):
			key["t"] = float(r["t"])
		elif r.get("frame"):
			key["t"] = float(r["frame"]) / opts.rate
		key["pos"] = [float(r.get(c) or 0) for c in "xyz"]
		if r.get("qw"):
			key["rot"] = [float(r[c]) for c in ("qx", "qy", "qz", "qw")]
		elif r.get("yaw") or r.get("pitch"):
			key["angles"] = [float(r.get(c) or 0) for c in ("pitch", "yaw", "roll")]
		elif r.get("tx"):
			key["look_at"] = [float(r[c]) for c in ("tx", "ty", "tz")]
		for c in ("fov_y", "fov_x", "lens"):
			if r.get(c):
				key[c] = float(r[c])
		if r.get("cut", "").lower() in ("1", "true", "yes", "y"):
			key["cut"] = True
		clips.setdefault(r.get("name") or Path(path).stem, []).append(key)
	return [_keys_to_path(n, ks, frame, {}, str(path)) for n, ks in clips.items()]


# -- glTF ------------------------------------------------------------------------------------------


_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_CTYPES = {5120: ("b", 127.0), 5121: ("B", 255.0), 5122: ("h", 32767.0), 5123: ("H", 65535.0), 5125: ("I", 0), 5126: ("f", 0)}


def _gltf_buffers(doc: dict, bin_: bytes, base: Path) -> list[bytes]:
	import base64

	out = []
	for i, b in enumerate(doc.get("buffers", [])):
		uri = b.get("uri")
		if uri is None:
			out.append(bin_)
		elif uri.startswith("data:"):
			out.append(base64.b64decode(uri.split(",", 1)[1]))
		else:
			from urllib.parse import unquote

			out.append((base / unquote(uri)).read_bytes())
	return out


def _accessor(doc: dict, bufs: list[bytes], i: int) -> "np.ndarray":
	acc = doc["accessors"][i]
	n, comps = acc["count"], _COMPONENTS[acc["type"]]
	code, norm = _CTYPES[acc["componentType"]]
	out = np.zeros((n, comps))
	if "bufferView" in acc:
		bv = doc["bufferViews"][acc["bufferView"]]
		data = bufs[bv["buffer"]]
		start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
		size = struct.calcsize("<" + code)
		stride = bv.get("byteStride") or size * comps
		for k in range(n):
			out[k] = struct.unpack_from(f"<{comps}{code}", data, start + k * stride)
		if acc.get("normalized") or (norm and acc["componentType"] != 5126):
			out = np.maximum(out / norm, -1.0) if norm else out
	return out


def _trs_matrix(t, r, s) -> "np.ndarray":
	m = np.eye(4)
	m[:3, :3] = quat_to_matrix(r) * np.asarray(s, np.float64)
	m[:3, 3] = t
	return m


def _gltf_sampler(times: "np.ndarray", vals: "np.ndarray", interp: str, at: float, quat: bool) -> "np.ndarray":
	if interp == "CUBICSPLINE":
		vals = vals.reshape(len(times), 3, -1)
	if at <= times[0] or len(times) == 1:
		return vals[0, 1] if interp == "CUBICSPLINE" else vals[0]
	if at >= times[-1]:
		return vals[-1, 1] if interp == "CUBICSPLINE" else vals[-1]
	i = int(np.searchsorted(times, at, side="right")) - 1
	dt = times[i + 1] - times[i]
	u = (at - times[i]) / dt
	if interp == "STEP":
		return vals[i]
	if interp == "CUBICSPLINE":
		p0, m0, p1, m1 = vals[i, 1], vals[i, 2] * dt, vals[i + 1, 1], vals[i + 1, 0] * dt
		u2, u3 = u * u, u * u * u
		v = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1
		return v / np.linalg.norm(v) if quat else v
	a, b = vals[i], vals[i + 1]
	if quat:
		if np.dot(a, b) < 0:
			b = -b
		d = float(np.clip(np.dot(a, b), -1, 1))
		if d > 0.9995:
			v = a + (b - a) * u
			return v / np.linalg.norm(v)
		th = math.acos(d)
		return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
	return a + (b - a) * u


def load_gltf(path: Path, opts: CameraOptions) -> list[CameraPath]:
	from .pbr import _gltf_doc

	path = Path(path)
	doc, bin_ = _gltf_doc(path)
	bufs = _gltf_buffers(doc, bin_, path.parent)
	nodes = doc.get("nodes", [])
	cams = doc.get("cameras", [])
	parent = [-1] * len(nodes)
	for i, n in enumerate(nodes):
		for c in n.get("children", []):
			parent[c] = i
	rest = []
	for n in nodes:
		if "matrix" in n:
			m = np.array(n["matrix"], np.float64).reshape(4, 4).T
			t = m[:3, 3]
			s = np.linalg.norm(m[:3, :3], axis=0)
			r = quat_from_matrix(m[:3, :3] / np.where(s > 1e-12, s, 1))
			rest.append([t, r, s])
		else:
			rest.append([np.array(n.get("translation", [0, 0, 0]), np.float64),
				np.array(n.get("rotation", [0, 0, 0, 1]), np.float64), np.array(n.get("scale", [1, 1, 1]), np.float64)])
	frame = _frame_for(path, opts, "gltf")
	cam_nodes = [i for i, n in enumerate(nodes) if "camera" in n]
	if not cam_nodes:
		raise ValueError(f"{path.name}: no camera nodes")
	notes = []

	def chain(i):
		out = []
		while i >= 0:
			out.append(i)
			i = parent[i]
		return out

	def cam_info(ci):
		c = cams[ci]
		if c.get("type") == "orthographic":
			return None
		p = c.get("perspective", {})
		return (math.degrees(float(p.get("yfov", math.radians(DEFAULT_FOV_Y)))), float(p.get("znear", 0.01)) * frame.scale,
			float(p["zfar"]) * frame.scale if "zfar" in p else None, float(p.get("aspectRatio", 0) or 0))

	anims = doc.get("animations", [])
	out = []
	for cn in cam_nodes:
		info = cam_info(nodes[cn]["camera"])
		cname = nodes[cn].get("name") or doc["cameras"][nodes[cn]["camera"]].get("name") or f"camera{cn}"
		if info is None:
			notes.append(f"{cname}: orthographic cameras are not supported")
			continue
		fov0, near, far, aspect = info
		ch = set(chain(cn))
		clips = []
		for ai, an in enumerate(anims):
			samplers = an.get("samplers", [])
			chans = {}
			fov_ch = None
			for c in an.get("channels", []):
				tgt = c.get("target", {})
				s = samplers[c["sampler"]]
				times = _accessor(doc, bufs, s["input"])[:, 0]
				vals = _accessor(doc, bufs, s["output"])
				entry = (times, vals, s.get("interpolation", "LINEAR"))
				if tgt.get("node") in ch and tgt.get("path") in ("translation", "rotation", "scale"):
					chans[(tgt["node"], tgt["path"])] = entry
				ptr = tgt.get("extensions", {}).get("KHR_animation_pointer", {}).get("pointer", "")
				if tgt.get("path") == "pointer" and ptr == f"/cameras/{nodes[cn]['camera']}/perspective/yfov":
					fov_ch = entry
			if chans or fov_ch:
				clips.append((an.get("name") or f"anim{ai}", chans, fov_ch))
		if not clips:
			clips = [("static", {}, None)]
		for an_name, chans, fov_ch in clips:
			key_times = sorted({float(x) for e in [*chans.values(), *([fov_ch] if fov_ch else [])] for x in e[0]}) or [0.0]
			t0, t1 = key_times[0], key_times[-1]
			n = max(1, int(round((t1 - t0) * opts.rate)) + 1)
			grid = t0 + np.arange(n) / opts.rate
			grid[-1] = t1 if n > 1 else t0
			pos, rot, fov = [], [], []
			order = list(reversed(chain(cn)))
			for at in grid:
				m = np.eye(4)
				for nd in order:
					t, r, s = rest[nd]
					if (nd, "translation") in chans:
						tt, vv, ip = chans[(nd, "translation")]
						t = _gltf_sampler(tt, vv, ip, at, False)
					if (nd, "rotation") in chans:
						tt, vv, ip = chans[(nd, "rotation")]
						r = _gltf_sampler(tt, vv, ip, at, True)
					if (nd, "scale") in chans:
						tt, vv, ip = chans[(nd, "scale")]
						s = _gltf_sampler(tt, vv, ip, at, False)
					m = m @ _trs_matrix(t, r, s)
				pos.append(frame.point(m[:3, 3]))
				rot.append(quat_from_matrix(frame.orient(m[:3, :3])))
				fov.append(math.degrees(float(_gltf_sampler(fov_ch[0], fov_ch[1], fov_ch[2], at, False)[0])) if fov_ch else fov0)
			# STEP channels jump at their keys: mark those as cuts (a duplicate at the tick)
			times = list(grid - t0)
			name = an_name if len(cam_nodes) == 1 else f"{cname}_{an_name}"
			p = CameraPath(name, np.array(times), np.array(pos), _hemisphere(np.array(rot)), np.array(fov),
				near, far if far else DEFAULT_FAR, aspect, False, str(path), list(notes))
			if any(e[2] == "STEP" for e in chans.values()):
				p.notes.append("STEP channels are sampled at the grid; their jumps land on the nearest tick")
			out.append(p)
	if not out:
		raise ValueError(f"{path.name}: " + "; ".join(notes or ["no usable cameras"]))
	return out


# -- Assimp (FBX, Collada, ...) ---------------------------------------------------------------------------


def _collada_optics(path: Path) -> dict:
	"""yfov/xfov/aspect/znear/zfar of the first perspective camera in a .dae (Assimp hides them)."""
	import xml.etree.ElementTree as ET

	try:
		root = ET.parse(path).getroot()
	except (ET.ParseError, OSError):
		return {}
	out = {}
	for el in root.iter():
		if el.tag.split("}")[-1] == "perspective":
			for c in el:
				tag = c.tag.split("}")[-1]
				try:
					out[tag] = float((c.text or "").strip())
				except ValueError:
					pass
			break
	return out


def load_assimp(path: Path, opts: CameraOptions) -> list[CameraPath]:
	from . import models
	from . import skeleton as _sk

	ai = models._assimp()
	try:
		scene = ai.import_file(str(path), ai.Process_ValidateDataStructure)
	except Exception as exc:	# noqa: BLE001 - assimp raises bare RuntimeErrors
		raise ValueError(f"Assimp could not read {path.name}: {exc or 'no details'} (Assimp refuses scenes "
			"with no meshes: export the camera with any mesh, or as glTF)") from exc
	if scene is None:
		raise ValueError(f"Assimp could not read {path.name}")
	tree = models._Nodes(scene.root_node)
	ext = path.suffix.lower()
	frame = _frame_for(path, opts, "gltf")
	if opts.axes == "auto" and ext == ".fbx":
		frame.camera = "fbx"
	notes = []
	anims = list(getattr(scene, "animations", None) or [])
	if opts.node:
		if opts.node not in tree.by_name:
			raise ValueError(f"{path.name}: no node named {opts.node!r} (nodes: {', '.join(list(tree.by_name)[:20])})")
		cam = tree.by_name[opts.node]
	else:
		named = [i for i, n in enumerate(tree.nodes) if re.search(r"cam", n.name or "", re.I)]
		animated = {tree.by_name[c.node_name] for a in anims for c in a.channels if c.node_name in tree.by_name}
		pick = [i for i in named if i in animated] or named or sorted(animated)
		if not pick:
			raise ValueError(f"{path.name}: no node looks like a camera; name it with --node")
		cam = pick[0]
		if len(pick) > 1:
			notes.append(f"took node {tree.nodes[cam].name!r}; others: {', '.join(tree.nodes[i].name for i in pick[1:4])}")
	optics = _collada_optics(path) if ext == ".dae" else {}
	aspect = optics.get("aspect_ratio", 0.0)
	if "yfov" in optics:
		fov = optics["yfov"]
	elif "xfov" in optics:
		fov = fov_x_to_y(optics["xfov"], aspect or 4 / 3)
	else:
		fov = DEFAULT_FOV_Y
		if opts.fov_y is None:
			notes.append(f"Assimp does not pass camera FOV through for {ext}; using {DEFAULT_FOV_Y:.1f}° vertical (set --fov)")
	near = optics.get("znear", DEFAULT_NEAR / frame.scale) * frame.scale
	far = optics.get("zfar", DEFAULT_FAR / frame.scale) * frame.scale
	chain = []
	i = cam
	while i >= 0:
		chain.append(i)
		i = tree.parent[i]
	chain.reverse()
	out = []
	used = [(a.name or f"anim{k}", a) for k, a in enumerate(anims)
		if any(c.node_name in {tree.nodes[n].name for n in chain} for c in a.channels)] or [("static", None)]
	for an_name, an in used:
		chans = {}
		tps, dur = 25.0, 0.0
		if an is not None:
			tps = float(an.ticks_per_second or 0) or 25.0
			dur = float(an.duration or 0)
			for c in an.channels:
				if c.node_name in tree.by_name:
					chans[tree.by_name[c.node_name]] = _sk.Channel(c)
		secs = dur / tps
		n = max(1, int(round(secs * opts.rate)) + 1)
		pos, rot = [], []
		for k in range(n):
			tick = min(dur, k / opts.rate * tps)
			m = np.eye(4)
			for nd in chain:
				m = m @ (chans[nd].local(tick, tree.local[nd]) if nd in chans else tree.local[nd])
			pos.append(frame.point(m[:3, 3]))
			rot.append(quat_from_matrix(frame.orient(m[:3, :3])))
		name = re.sub(r"^.*\|", "", an_name)
		out.append(CameraPath(name, np.arange(n) / opts.rate, np.array(pos), _hemisphere(np.array(rot)),
			np.full(n, fov), near, far, aspect, False, str(path), list(notes)))
	return out


# -- GoldSrc trigger_camera / path_corner -------------------------------------------------------------


def _vec(s: str, default=(0.0, 0.0, 0.0)) -> "np.ndarray":
	try:
		v = [float(x) for x in s.split()]
		return np.array(v[:3] + [0.0] * (3 - len(v[:3])))
	except (ValueError, AttributeError):
		return np.array(default, np.float64)


def _bsp_model_centres(data: bytes) -> list["np.ndarray"]:
	from . import goldsrc

	off, size = goldsrc.bsp_lump(data, 14)
	out = []
	for k in range(size // 64):
		v = struct.unpack_from("<9f", data, off + k * 64)
		out.append((np.array(v[0:3]) + np.array(v[3:6])) / 2)
	return out


def load_goldsrc(path: Path, opts: CameraOptions) -> list[CameraPath]:
	"""trigger_camera entities: the camera travels its path_corner chain at each corner's speed
	(waiting where a corner says so) and looks at its target. The game's camera eases its speed
	and turn rate each frame; this is the straight-line version, so treat it as a starting point."""
	from . import goldsrc
	from .budget import parse_entities

	path = Path(path)
	data = path.read_bytes()
	if path.suffix.lower() == ".bsp":
		text = goldsrc.bsp_entities_text(data)
		centres = _bsp_model_centres(data)
	else:
		text, centres = data.decode("latin-1"), []
	ents = parse_entities(text)
	by_name: dict[str, list[dict]] = {}
	for e in ents:
		if e.get("targetname"):
			by_name.setdefault(e["targetname"], []).append(e)

	def where(e: dict) -> "np.ndarray":
		if "origin" in e:
			return _vec(e["origin"])
		m = e.get("model", "")
		if m.startswith("*") and m[1:].isdigit() and int(m[1:]) < len(centres):
			return centres[int(m[1:])]
		return np.zeros(3)

	out = []
	for k, cam in enumerate(e for e in ents if e.get("classname") == "trigger_camera"):
		name = cam.get("targetname") or f"camera{k}"
		notes = []
		hold = float(cam.get("wait", 10) or 10)
		speed0 = float(cam.get("speed", 0) or 0)
		tgt = by_name.get(cam.get("target", ""), [None])[0]
		spawn = int(cam.get("spawnflags", 0) or 0)
		if spawn & 1:
			notes.append("starts at the player's view (spawnflag 1); this path starts at the entity's origin")
		if spawn & 2:
			notes.append("follows the player (spawnflag 2); the player is not known here")
			tgt = None
		if tgt is not None and tgt.get("classname", "").startswith(("monster_", "func_train", "func_tracktrain")):
			notes.append(f"looks at {tgt.get('classname')}, which can move; its spawn position is used")
		look = where(tgt) if tgt is not None else None
		fixed_rot = angles_to_matrix(*_vec(cam.get("angles", "0 0 0")))
		# walk the corners
		pts = [(0.0, where(cam))]
		cur = by_name.get(cam.get("moveto", ""), [None])[0]
		t, speed, seen = 0.0, speed0, set()
		pos = where(cam)
		while cur is not None and id(cur) not in seen and t < hold:
			seen.add(id(cur))
			if float(cur.get("speed", 0) or 0):
				speed = float(cur["speed"])
			nxt_pos = where(cur)
			dist = float(np.linalg.norm(nxt_pos - pos))
			if speed <= 0:
				notes.append("no speed on the camera or its corners; it stays put")
				break
			t += dist / speed
			pts.append((t, nxt_pos))
			pos = nxt_pos
			wait = float(cur.get("wait", 0) or 0)
			if wait > 0:
				t += wait
				pts.append((t, pos))
			cur = by_name.get(cur.get("target", ""), [None])[0]
		if pts[-1][0] < hold:
			pts.append((hold, pts[-1][1]))
		times = np.array([p[0] for p in pts])
		keep = np.concatenate([[True], np.diff(times) > 1e-9])
		times = times[keep]
		P = np.array([p[1] for p in pts])[keep]
		n = max(2, int(round(times[-1] * opts.rate)) + 1)
		grid = np.minimum(np.arange(n) / opts.rate, times[-1])
		dense_p = evaluate(times, P, grid, "linear")
		rots = [quat_from_matrix(look_at(p, look) if look is not None else fixed_rot) for p in dense_p]
		cp = CameraPath(name, grid, dense_p, _hemisphere(np.array(rots)), np.full(n, DEFAULT_FOV_Y), DEFAULT_NEAR, DEFAULT_FAR,
			0.0, False, str(path), notes)
		cp.notes.append("straight lines at corner speeds; the game eases speed and turns over a few frames")
		out.append(cp)
	if not out:
		raise ValueError(f"{path.name}: no trigger_camera entities")
	return out


# -- front door ---------------------------------------------------------------------------------------


def load(path: Path, opts: CameraOptions | None = None) -> list[CameraPath]:
	import fnmatch

	opts = opts or CameraOptions()
	path = Path(path)
	ext = path.suffix.lower()
	if ext in (".gltf", ".glb"):
		clips = load_gltf(path, opts)
	elif ext == ".json":
		clips = load_json(path, opts)
	elif ext == ".csv":
		clips = load_csv(path, opts)
	elif ext in (".bsp", ".ent"):
		clips = load_goldsrc(path, opts)
	elif ext in EXTENSIONS_ASSIMP:
		clips = load_assimp(path, opts)
	else:
		raise ValueError(f"{path.name}: not a camera source ({', '.join(sorted(EXTENSIONS))})")
	if opts.clips:
		clips = [c for c in clips if any(fnmatch.fnmatch(c.name.lower(), p.lower()) for p in opts.clips)]
	return [_finish(c, opts) for c in clips]


def is_dense(path: CameraPath, rate: float) -> bool:
	"""Already sampled on the 1/rate grid (glTF, Assimp and GoldSrc loads are)."""
	t = path.times - path.times[0]
	return len(t) > 2 and np.allclose(np.diff(t), 1 / rate, rtol=0, atol=1e-6)


def encode_all(paths: list[CameraPath], opts: CameraOptions) -> tuple[bytes, list[Encoded]]:
	"""Encode clips into one .gcam; with max_bytes, scale the tolerances up until it fits."""
	scale = 1.0
	for _ in range(24):
		enc = [encode_clip(p, opts.rate, opts.interp, opts.encoding, opts.pos_tol * scale, opts.rot_tol * scale,
			opts.fov_tol * scale, dense=is_dense(p, opts.rate)) for p in paths]
		data = write_gcam(enc)
		if not opts.max_bytes or len(data) <= opts.max_bytes:
			if scale > 1:
				for e in enc:
					e.path.notes.append(f"tolerances ×{scale:.2f} to fit {opts.max_bytes} bytes")
			return data, enc
		scale *= 1.5
	raise ValueError(f"cameras do not fit {opts.max_bytes} bytes even with tolerances ×{scale / 1.5:.0f}")


def describe(enc: Encoded) -> dict:
	p = enc.path
	return {"name": clip_name(p.name), "duration": round(p.duration, 3), "samples": enc.samples,
		"keys": {("pos", "rot", "fov")[t.kind]: len(t.times) for t in enc.tracks},
		"max_error": {k: round(v, 4) for k, v in enc.max_error.items()},
		"bytes": len(enc.body) + CLIP_ENTRY, "near": p.near, "far": p.far, "fov_y": [round(float(p.fov.min()), 2),
		round(float(p.fov.max()), 2)], "interp": {("pos", "rot", "fov")[t.kind]: t.interp for t in enc.tracks}, "encoding": "fixed" if enc.fixed else "f32",
		"notes": p.notes, "source": p.source}


# -- preview --------------------------------------------------------------------------------------------


def preview_png(clip: GcamClip, dest: Path, size: int = 512) -> Path:
	"""Top-down (XY) and side (XZ) views of the decoded path with view directions, and FOV over time."""
	from PIL import Image, ImageDraw

	n = max(2, int(clip.duration * 30) + 1)
	at = np.linspace(0, clip.duration, n)
	s = clip.sample(at)
	pos, rot, fov = s["pos"], s["rot"], s["fov"]
	fwd = np.array([quat_to_matrix(q)[:, 0] for q in rot])
	lo, hi = pos.min(0), pos.max(0)
	span = float(max((hi - lo)[:2].max(), (hi - lo)[[0, 2]].max(), 1.0)) * 1.15
	mid = (lo + hi) / 2
	W, H = size * 2, size + 120
	im = Image.new("RGB", (W, H), (22, 24, 28))
	d = ImageDraw.Draw(im)

	def to(px, py, ox):
		return ox + (px / span + 0.5) * (size - 20) + 10, (0.5 - py / span) * (size - 20) + 10

	for ox, (a, b), label in ((0, (0, 1), "top (x, y)"), (size, (0, 2), "side (x, z)")):
		d.rectangle([ox + 4, 4, ox + size - 4, size - 4], outline=(60, 64, 72))
		d.text((ox + 10, 8), label, fill=(160, 166, 176))
		pts = [to(p[a] - mid[a], p[b] - mid[b], ox) for p in pos]
		for k in range(1, len(pts)):
			c = int(80 + 175 * k / len(pts))
			d.line([pts[k - 1], pts[k]], fill=(c, 140, 255 - c // 2), width=2)
		step = max(1, len(pos) // 24)
		for k in range(0, len(pos), step):
			x0, y0 = pts[k]
			x1, y1 = to(pos[k][a] - mid[a] + fwd[k][a] * span * 0.06, pos[k][b] - mid[b] + fwd[k][b] * span * 0.06, ox)
			d.line([(x0, y0), (x1, y1)], fill=(240, 200, 90), width=1)
		pos_keys = clip.tracks[TRACK_POS].values
		for p in pos_keys:
			x, y = to(p[a] - mid[a], p[b] - mid[b], ox)
			d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], outline=(255, 255, 255))
	# fov strip
	top = size + 10
	d.rectangle([4, top, W - 4, H - 4], outline=(60, 64, 72))
	f_lo, f_hi = float(fov.min()) - 1, float(fov.max()) + 1
	pts = [(10 + (W - 20) * k / (n - 1), H - 10 - (H - top - 20) * (f - f_lo) / (f_hi - f_lo)) for k, f in enumerate(fov)]
	d.line(pts, fill=(120, 220, 140), width=2)
	d.text((10, top + 4), f"{clip.name}: {clip.duration:.2f} s, fov_y {fov.min():.1f}-{fov.max():.1f} deg, "
		f"{len(clip.tracks[TRACK_POS].times)}/{len(clip.tracks[TRACK_ROT].times)}/{len(clip.tracks[TRACK_FOV].times)} "
		f"pos/rot/fov keys, {clip.bytes} B", fill=(200, 204, 212))
	dest = Path(dest)
	dest.parent.mkdir(parents=True, exist_ok=True)
	im.save(dest)
	return dest


# -- batch conversion (CLI and build rules) ---------------------------------------------------------


def options_from(d: dict) -> CameraOptions:
	"""CameraOptions from a manifest rule or CLI dict (unknown keys are ignored; sizes take 2K etc.)."""
	from .budget import parse_size

	o = CameraOptions()
	for k in ("rate", "pos_tol", "rot_tol", "fov_tol", "yaw"):
		if d.get(k) is not None:
			setattr(o, k, float(d[k]))
	for k in ("interp", "encoding", "axes", "node"):
		if d.get(k) is not None:
			setattr(o, k, str(d[k]))
	for k in ("scale", "near", "far", "aspect"):
		if d.get(k) is not None:
			setattr(o, k, float(d[k]))
	if d.get("loop") is not None:
		o.loop = bool(d["loop"])
	if d.get("offset") is not None:
		off = d["offset"]
		o.offset = tuple(float(x) for x in (off.split(",") if isinstance(off, str) else off))
	fy = d.get("fov_y", d.get("fov"))
	if fy is not None:
		o.fov_y = float(fy)
	elif d.get("fov_x") is not None:
		o.fov_y = fov_x_to_y(float(d["fov_x"]), float(d.get("aspect") or 4 / 3))
	clips = d.get("clips")
	if clips:
		o.clips = tuple(clips.split(",") if isinstance(clips, str) else clips)
	if d.get("max_bytes"):
		o.max_bytes = parse_size(str(d["max_bytes"]))
	if o.interp not in INTERPS:
		raise ValueError(f"interp is one of {', '.join(INTERPS)}")
	if o.encoding not in ENCODINGS:
		raise ValueError(f"encoding is one of {', '.join(ENCODINGS)}")
	return o


def output_name(src: Path) -> str:
	"""GoldSrc maps get cam_<map>.gcam (what `gxbake budget` looks for in media/); others <stem>.gcam."""
	stem = clip_name(Path(src).stem)
	return f"cam_{stem}.gcam" if Path(src).suffix.lower() in (".bsp", ".ent") else f"{stem}.gcam"


@dataclass
class Output:
	path: Path
	data: bytes
	clips: list[Encoded]
	inputs: list[Path]


def convert(files: list[Path], opts: CameraOptions, out: Path, combine: str | None = None) -> list[Output]:
	"""Load every file and encode one .gcam per file, or all clips into `combine`. Nothing is written."""
	loaded = [(f, load(f, opts)) for f in files]
	if combine:
		paths = [p for _f, ps in loaded for p in ps]
		if not paths:
			return []
		data, enc = encode_all(paths, opts)
		return [Output(out / combine, data, enc, [f for f, _ in loaded])]
	res = []
	used: set[str] = set()
	for f, ps in loaded:
		if ps:
			data, enc = encode_all(ps, opts)
			name = output_name(f)
			if name in used:	# intro.glb and intro.dae side by side
				name = f"{Path(name).stem}_{f.suffix.lower().lstrip('.')}.gcam"
			used.add(name)
			res.append(Output(out / name, data, enc, [f]))
	return res
