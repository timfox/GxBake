"""Bone and animation maths for baking Assimp scenes into GoldSrc studio models.

Studio bones are rigid: a position and Euler angles relative to the parent,
turned into a matrix as R = Rz(angles[2]) · Ry(angles[1]) · Rx(angles[0]) (the
engine's AngleQuaternion). Vertices are stored in their bone's space, so the
engine places them with the bone's world matrix alone (no inverse bind).

Animation values are the RLE-compressed mstudioanimvalue_t stream studiomdl
writes: per channel, value = bone.value[j] + stored * bone.scale[j].
"""

from __future__ import annotations

import math
import struct

import numpy as np

MAXSTUDIOBONES = 128
STUDIO_LOOPING = 0x0001


# -- matrices -----------------------------------------------------------------


def rigid(m: "np.ndarray") -> "np.ndarray":
	"""Nearest rotation + the translation of a 4x4 affine (drops scale and shear)."""
	u, _s, vt = np.linalg.svd(m[:3, :3])
	r = u @ vt
	if np.linalg.det(r) < 0:
		u[:, -1] *= -1
		r = u @ vt
	out = np.eye(4)
	out[:3, :3] = r
	out[:3, 3] = m[:3, 3]
	return out


def rigid_inverse(m: "np.ndarray") -> "np.ndarray":
	out = np.eye(4)
	rt = m[:3, :3].T
	out[:3, :3] = rt
	out[:3, 3] = -rt @ m[:3, 3]
	return out


def angles_from_matrix(r: "np.ndarray") -> tuple[float, float, float]:
	"""Inverse of angle_matrix: (roll about X, pitch about Y, yaw about Z) in radians."""
	sp = -float(np.clip(r[2, 0], -1.0, 1.0))
	pitch = math.asin(sp)
	if abs(sp) < 0.999999:
		roll = math.atan2(r[2, 1], r[2, 2])
		yaw = math.atan2(r[1, 0], r[0, 0])
	else:	# gimbal lock: fold roll into yaw
		roll = 0.0
		yaw = math.atan2(-r[0, 1], r[1, 1])
	return roll, pitch, yaw


def angle_matrix(a: "np.ndarray") -> "np.ndarray":
	sr, cr = math.sin(a[0]), math.cos(a[0])
	sp, cp = math.sin(a[1]), math.cos(a[1])
	sy, cy = math.sin(a[2]), math.cos(a[2])
	return np.array([
		[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
		[cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
		[-sp, sr * cp, cr * cp],
	])


def pose_from_matrix(m: "np.ndarray") -> "np.ndarray":
	"""Rigid 4x4 → studio (x, y, z, roll, pitch, yaw)."""
	return np.array([*m[:3, 3], *angles_from_matrix(m[:3, :3])])


def matrix_from_pose(p: "np.ndarray") -> "np.ndarray":
	out = np.eye(4)
	out[:3, :3] = angle_matrix(p[3:6])
	out[:3, 3] = p[:3]
	return out


def quat_matrix(q: "np.ndarray") -> "np.ndarray":
	"""(x, y, z, w) → 3x3."""
	x, y, z, w = q / (np.linalg.norm(q) or 1.0)
	return np.array([
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	])


def _slerp(a: "np.ndarray", b: "np.ndarray", t: float) -> "np.ndarray":
	d = float(np.dot(a, b))
	if d < 0:
		b, d = -b, -d
	if d > 0.9995:
		q = a + (b - a) * t
		return q / np.linalg.norm(q)
	th = math.acos(d)
	return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def sample_keys(times: "np.ndarray", values: "np.ndarray", t: float, quat: bool = False) -> "np.ndarray":
	"""Linear (or slerp) key interpolation, clamped at both ends."""
	if len(times) == 1 or t <= times[0]:
		return values[0]
	if t >= times[-1]:
		return values[-1]
	i = int(np.searchsorted(times, t, side="right")) - 1
	span = times[i + 1] - times[i]
	f = (t - times[i]) / span if span > 0 else 0.0
	if quat:
		return _slerp(values[i], values[i + 1], f)
	return values[i] + (values[i + 1] - values[i]) * f


class Channel:
	"""One node's keys from an assimp NodeAnim, as numpy arrays."""

	def __init__(self, ch):
		def arr(mv, n):
			if mv is None:
				return np.zeros((0, n))
			fmt = "d" if mv.format == "d" else "f"
			return np.frombuffer(mv, np.float64 if fmt == "d" else np.float32).astype(np.float64).reshape(-1, n)

		self.node = ch.node_name
		self.pt = arr(ch.position_key_times, 1)[:, 0]
		self.pv = arr(ch.position_key_values, 3)
		self.rt = arr(ch.rotation_key_times, 1)[:, 0]
		self.rv = arr(ch.rotation_key_values, 4)	# x, y, z, w
		self.st = arr(ch.scaling_key_times, 1)[:, 0]
		self.sv = arr(ch.scaling_key_values, 3)

	def local(self, tick: float, rest: "np.ndarray") -> "np.ndarray":
		"""Node-local matrix at `tick`; channels without keys keep the rest part."""
		r_t, r_r, r_s = decompose(rest)
		t = sample_keys(self.pt, self.pv, tick) if len(self.pt) else r_t
		r = quat_matrix(sample_keys(self.rt, self.rv, tick, quat=True)) if len(self.rt) else r_r
		s = sample_keys(self.st, self.sv, tick) if len(self.st) else r_s
		m = np.eye(4)
		m[:3, :3] = r * s
		m[:3, 3] = t
		return m


def decompose(m: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
	t = m[:3, 3].copy()
	s = np.linalg.norm(m[:3, :3], axis=0)
	r = m[:3, :3] / np.where(s > 1e-12, s, 1.0)
	return t, r, s


# -- animation value streams --------------------------------------------------


def rle_encode(q: list[int]) -> list[int]:
	"""mstudioanimvalue_t stream: (valid, total) headers, each chunk's
	trailing repeats dropped.

	The chunks are chosen to minimise the stream (dynamic programming over
	run boundaries), not cut greedily every 255 frames: a chunk only saves
	its trailing run, so [1 1 1 1 5 5 5 5] is cheaper as two chunks
	(1×4, 5×4: four shorts) than as one (six shorts). The engine walks
	chunks one after another, so any partition decodes the same.
	"""
	n = len(q)
	if not n:
		return []
	INF = 1 << 30
	best = [INF] * (n + 1)
	cut = [0] * (n + 1)
	best[n] = 0
	# A chunk only needs to end where a run ends (or at the 255-frame cap):
	# ending inside a run never saves anything.
	end_set = {k for k in range(1, n) if q[k] != q[k - 1]} | {n}
	run_start = [0] * n
	for k in range(1, n):
		run_start[k] = run_start[k - 1] if q[k] == q[k - 1] else k
	for i in range(n - 1, -1, -1):
		lim = min(n, i + 255)
		for j in range(i + 1, lim + 1):
			if j not in end_set and j != lim:
				continue
			valid = max(run_start[j - 1], i) - i + 1
			c = 1 + valid + best[j]
			if c < best[i]:
				best[i], cut[i] = c, j
	out: list[int] = []
	i = 0
	while i < n:
		j = cut[i]
		valid = max(run_start[j - 1], i) - i + 1
		out.append(struct.unpack("<h", bytes((valid, j - i)))[0])
		out += q[i:i + valid]
		i = j
	return out


def rle_decode(stream: "np.ndarray", frame: int) -> int:
	"""The engine's lookup (StudioCalcBonePosition/Rotation) for one frame."""
	p = 0
	k = frame
	while True:
		valid, total = struct.unpack("<BB", struct.pack("<h", int(stream[p])))
		if total > k:
			break
		k -= total
		p += valid + 1
	return int(stream[p + k + 1]) if valid > k else int(stream[p + valid])


def bone_scales(rest: "np.ndarray", sequences: list["np.ndarray"]) -> "np.ndarray":
	"""Per bone/channel quantisation step so every sequence fits in a short."""
	dev = np.zeros(rest.shape)
	for frames in sequences:
		dev = np.maximum(dev, np.abs(frames - rest[None]).max(0))
	return np.where(dev > 1e-9, dev / 32000.0, 1.0 / 32.0)


def encode_anim_block(frames: "np.ndarray", rest: "np.ndarray", scale: "np.ndarray") -> bytes:
	"""mstudioanim_t[numbones] followed by the value streams for one sequence."""
	nframes, nbones, _ = frames.shape
	head = bytearray(nbones * 12)
	data = bytearray()
	for b in range(nbones):
		offs = [0] * 6
		for j in range(6):
			q = np.rint((frames[:, b, j] - rest[b, j]) / scale[b, j]).astype(np.int64)
			if not q.any():
				continue	# offset 0: the channel stays at bone.value[j]
			q = np.clip(q, -32768, 32767)
			pos = len(head) + len(data)
			rel = pos - b * 12
			if rel > 0xFFFF:
				raise ValueError("animation data too large for 16-bit offsets; lower the animation frame rate")
			offs[j] = rel
			stream = rle_encode([int(x) for x in q])
			data += struct.pack(f"<{len(stream)}h", *stream)
		struct.pack_into("<6H", head, b * 12, *offs)
	out = bytes(head + data)
	return out + bytes(-len(out) % 4)


def decode_anim_frame(raw: bytes, anim_off: int, nbones: int, frame: int,
		values: "np.ndarray", scales: "np.ndarray") -> "np.ndarray":
	"""(nbones, 6) pose for one frame, the way the engine reads it."""
	out = values.copy()
	for b in range(nbones):
		base = anim_off + b * 12
		offs = struct.unpack_from("<6H", raw, base)
		for j, o in enumerate(offs):
			if o == 0:
				continue
			start = base + o
			n = (len(raw) - start) // 2
			stream = np.frombuffer(raw, np.int16, n, start)
			out[b, j] = values[b, j] + rle_decode(stream, frame) * scales[b, j]
	return out


def world_matrices(poses: "np.ndarray", parents: list[int]) -> list["np.ndarray"]:
	mats: list[np.ndarray] = []
	for b, p in enumerate(parents):
		m = matrix_from_pose(poses[b])
		mats.append(m if p < 0 else mats[p] @ m)
	return mats
