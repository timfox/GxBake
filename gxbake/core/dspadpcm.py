"""Nintendo DSP ADPCM (the GameCube's native sound format): encode, decode, .dsp files.

16-bit PCM is 2 bytes a sample; DSP ADPCM is 8 bytes per 14 samples (4.57
bits), so a sound takes 28 % of the space with a second-order predictor that
is usually transparent for game audio. The GameCube's DSP decodes it in
hardware (AX), and a software decoder is a dozen lines (decode_frames).

Frame: one header byte (predictor index << 4 | scale exponent), then 14 signed
nibbles. Each sample is

	s = clamp16(((nibble << exp) << 11) + 1024 + c1 * h1 + c2 * h2) >> 11)

with (c1, c2) one of the stream's 8 coefficient pairs (s16, 1.11 fixed
point) and h1, h2 the two previous decoded samples.

The .dsp file is the standard 96-byte big-endian header (as written by
Nintendo's DSPADPCM.exe and read by every GameCube tool) followed by the
frames. Addresses in it count nibbles, and a stream's first sample is at
nibble 2 because each frame's header byte takes two.

The encoder picks the 8 coefficient pairs by clustering per-frame
least-squares predictors (weighted by frame energy), then encodes each frame
with the pair and exponent that give the least squared error, closed loop so
the quantisation error never accumulates.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SAMPLES_PER_FRAME = 14
BYTES_PER_FRAME = 8
HEADER = struct.Struct(">IIIHHIII16hHHhhHhh22x")
assert HEADER.size == 96


def frame_count(samples: int) -> int:
	return -(-samples // SAMPLES_PER_FRAME)


def adpcm_bytes(samples: int) -> int:
	return frame_count(samples) * BYTES_PER_FRAME


def sample_to_nibble(i: int) -> int:
	return (i // SAMPLES_PER_FRAME) * 16 + 2 + i % SAMPLES_PER_FRAME


# --- coefficients -------------------------------------------------------------

def _frame_predictors(x: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
	"""Least-squares (c1, c2) per 14-sample frame, and each frame's weight."""
	n = frame_count(len(x))
	pad = np.zeros(n * SAMPLES_PER_FRAME + 2)
	pad[2:2 + len(x)] = x
	cur = np.lib.stride_tricks.sliding_window_view(pad[2:], SAMPLES_PER_FRAME)[::SAMPLES_PER_FRAME]
	h1 = np.lib.stride_tricks.sliding_window_view(pad[1:-1], SAMPLES_PER_FRAME)[::SAMPLES_PER_FRAME]
	h2 = np.lib.stride_tricks.sliding_window_view(pad[:-2], SAMPLES_PER_FRAME)[::SAMPLES_PER_FRAME]
	r11, r22, r12 = (h1 * h1).sum(1), (h2 * h2).sum(1), (h1 * h2).sum(1)
	r01, r02 = (cur * h1).sum(1), (cur * h2).sum(1)
	det = r11 * r22 - r12 * r12
	ok = np.abs(det) > 1e-3 * np.maximum(r11 * r22, 1)
	c1 = np.where(ok, (r01 * r22 - r02 * r12) / np.where(ok, det, 1), np.where(r11 > 0, r01 / np.maximum(r11, 1), 0))
	c2 = np.where(ok, (r02 * r11 - r01 * r12) / np.where(ok, det, 1), 0)
	coef = np.clip(np.stack((c1, c2), 1), -1.999, 1.999)
	return coef, np.log1p((cur * cur).sum(1))


# Pairs a stream can always fall back on: silence/DC, first and second order.
_SEED = np.array([[0, 0], [1, 0], [2, -1], [1.75, -0.85], [1.5, -0.6], [0.9, 0], [1.9, -0.95], [0.5, 0]])


def design_coefs(x: "np.ndarray", iters: int = 12) -> "np.ndarray":
	"""8 predictor pairs for a mono stream, as 1.11 fixed point s16."""
	cand, w = _frame_predictors(x.astype(np.float64))
	keep = w > 0
	cand, w = cand[keep], w[keep]
	cent = _SEED.astype(np.float64).copy()
	if len(cand) >= 8:
		for _ in range(iters):	# weighted k-means
			d = ((cand[:, None, :] - cent[None]) ** 2).sum(-1)
			lab = d.argmin(1)
			for k in range(8):
				m = lab == k
				if m.any():
					cent[k] = (cand[m] * w[m, None]).sum(0) / w[m].sum()
	return np.clip(np.round(cent * 2048), -32768, 32767).astype(np.int16)


# --- frames -----------------------------------------------------------------

def _open_loop_choice(pad: "np.ndarray", coefs: "np.ndarray") -> "np.ndarray":
	"""Per frame, the pair with the least open-loop residual energy (vectorised)."""
	x = pad.astype(np.float64)
	h1 = np.concatenate(([0.0], x[:-1]))
	h2 = np.concatenate(([0.0, 0.0], x[:-2]))
	c = coefs.astype(np.float64) / 2048
	res = x[None] - c[:, 0, None] * h1[None] - c[:, 1, None] * h2[None]
	return (res.reshape(8, -1, 14) ** 2).sum(2).argmin(0)


def _encode_frames(x: "np.ndarray", coefs: "np.ndarray", fast: bool = False) -> tuple[bytes, list[tuple[int, int, int]]]:
	"""Closed-loop encode; returns the frames and, per frame, (header, h1, h2) at its start.
	`fast` takes each frame's pair from an open-loop search and only searches
	the exponent closed loop (about 6x quicker, a fraction of a dB worse)."""
	n = frame_count(len(x))
	pad = np.zeros(n * SAMPLES_PER_FRAME, np.int64)
	pad[:len(x)] = x
	cs = [(int(a), int(b)) for a, b in coefs]
	pick = _open_loop_choice(pad, coefs).tolist() if fast else None
	out = bytearray()
	ctx: list[tuple[int, int, int]] = []
	h1 = h2 = 0
	for f in range(n):
		blk = pad[f * 14:f * 14 + 14].tolist()
		best = None
		for ci, (c1, c2) in ([(pick[f], cs[pick[f]])] if pick else enumerate(cs)):
			# open-loop residual tells the smallest exponent worth trying
			p1, p2, peak = h1, h2, 0
			for s in blk:
				r = s - ((c1 * p1 + c2 * p2 + 1024) >> 11)
				peak = max(peak, abs(r))
				p2, p1 = p1, s
			e0 = 0
			while e0 < 12 and (7 << e0) < peak:
				e0 += 1
			for e in range(max(0, e0 - 1), min(12, e0 + 1) + 1):
				p1, p2, err, nib = h1, h2, 0, []
				for s in blk:
					pred = c1 * p1 + c2 * p2 + 1024
					q = int(round(((s << 11) - pred + 1024) / (2048 << e)))
					q = -8 if q < -8 else 7 if q > 7 else q
					v = (((q << e) << 11) + pred) >> 11
					v = -32768 if v < -32768 else 32767 if v > 32767 else v
					err += (s - v) * (s - v)
					nib.append(q & 15)
					p2, p1 = p1, v
					if best is not None and err >= best[0]:
						break
				else:
					if best is None or err < best[0]:
						best = (err, ci, e, nib, p1, p2)
		_, ci, e, nib, n1, n2 = best
		head = (ci << 4) | e
		ctx.append((head, h1, h2))
		out.append(head)
		out += bytes((nib[k] << 4) | nib[k + 1] for k in range(0, 14, 2))
		h1, h2 = n1, n2
	return bytes(out), ctx


def decode_frames(data: bytes, coefs: "np.ndarray", samples: int, h1: int = 0, h2: int = 0) -> "np.ndarray":
	"""DSP ADPCM frames → s16 samples (what AX or a software mixer produces)."""
	out = np.empty(frame_count(samples) * 14, np.int16)
	cs = [(int(a), int(b)) for a, b in coefs]
	k = 0
	for f in range(frame_count(samples)):
		head = data[f * 8]
		c1, c2 = cs[head >> 4 & 7]
		e = head & 15
		for b in data[f * 8 + 1:f * 8 + 8]:
			for q in (b >> 4, b & 15):
				q = q - 16 if q >= 8 else q
				v = (((q << e) << 11) + 1024 + c1 * h1 + c2 * h2) >> 11
				v = -32768 if v < -32768 else 32767 if v > 32767 else v
				out[k] = v
				k += 1
				h2, h1 = h1, v
	return out[:samples]


# --- streams and .dsp files ---------------------------------------------------

@dataclass
class DspStream:
	samples: int
	rate: int
	coefs: "np.ndarray"	# 8×2 s16
	data: bytes
	loop_start: int | None = None	# sample index
	loop_end: int | None = None	# last sample played, inclusive
	loop_ctx: tuple[int, int, int] = (0, 0, 0)	# header, h1, h2 at the loop start's frame
	pred_scale: int = 0

	def to_dsp(self) -> bytes:
		loop = self.loop_start is not None
		ls = sample_to_nibble(self.loop_start) if loop else 2
		le = sample_to_nibble(self.loop_end if loop and self.loop_end is not None else self.samples - 1)
		head = HEADER.pack(self.samples, frame_count(self.samples) * 16, self.rate, int(loop), 0, ls, le, 2,
			*self.coefs.reshape(-1).tolist(), 0, self.pred_scale, 0, 0,
			self.loop_ctx[0] if loop else 0, self.loop_ctx[1] if loop else 0, self.loop_ctx[2] if loop else 0)
		return head + self.data

	def decode(self) -> "np.ndarray":
		return decode_frames(self.data, self.coefs, self.samples)


def encode(x: "np.ndarray", rate: int, loop_start: int | None = None, loop_end: int | None = None,
		fast: bool | None = None) -> DspStream:
	"""Mono s16 samples → a DSP ADPCM stream. Looping streams keep the
	decoder state at the loop start's frame in the header, as AX needs it."""
	x = np.asarray(x, np.int64)
	coefs = design_coefs(x)
	data, ctx = _encode_frames(x, coefs, len(x) > 10 * rate if fast is None else fast)
	st = DspStream(len(x), rate, coefs, data, loop_start, loop_end, pred_scale=data[0] if data else 0)
	if loop_start is not None and ctx:
		st.loop_ctx = ctx[loop_start // SAMPLES_PER_FRAME]
	return st


def read_dsp(blob: bytes) -> DspStream:
	f = HEADER.unpack_from(blob)
	samples, _nib, rate, loop, _fmt, ls, le, _cur = f[:8]
	coefs = np.array(f[8:24], np.int16).reshape(8, 2)
	def nib2s(n: int) -> int:
		return (n // 16) * 14 + max(0, n % 16 - 2)
	st = DspStream(samples, rate, coefs, blob[96:96 + adpcm_bytes(samples)],
		nib2s(ls) if loop else None, nib2s(le) if loop else None, (f[28], f[29], f[30]), f[25])
	return st


def snr_db(ref: "np.ndarray", test: "np.ndarray") -> float:
	r = ref.astype(np.float64)
	e = r - test.astype(np.float64)
	pe = float((e * e).sum())
	return float("inf") if pe == 0 else 10 * np.log10(max(float((r * r).sum()), 1e-9) / pe)
