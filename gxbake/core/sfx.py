"""Fit sound effects into the GameCube port's sound budget.

Sound_LoadWAV in the port (engine/client/soundlib/snd_wav.c) refuses any WAV
whose PCM data is over a per-file cap while map-load memory optimisation is
on: 20 KiB in gameplay, 48 KiB when the menu runs the server. Those sounds
never play. The MEM1 cache behind it (s_load.c) holds 128 KiB (320 KiB on the
menu route), backed by a 2 MiB ARAM store.

The loader only takes Microsoft PCM, 8 or 16 bit, mono or stereo, so ADPCM is
not an option yet. What does help, in the order fit_sound() tries them:

	1. cut the data after the loop end (the engine never plays it)
	2. stereo → mono (positional sfx are mixed mono anyway)
	3. 16 → 8 bit at the source rate (≤ 22050 Hz), with TPDF dither
	4. 22050 → 11025 Hz, 16 bit, then 8 bit
	5. optionally truncate, with a short fade, as a last resort

Loop points ("cue " chunk, and a CoolEdit "mark" loop length) are kept and
rescaled with the rate.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KiB = 1024
GAMEPLAY_MAX_FILE = 20 * KiB	# GC_SFX_MAX_FILE_BYTES
MENU_MAX_FILE = 48 * KiB	# GC_SFX_MENU_MAX_FILE_BYTES
GAMEPLAY_CACHE = 128 * KiB	# GC_SFX_BUDGET_BYTES
MENU_CACHE = 320 * KiB	# GC_SFX_MENU_BUDGET_BYTES
ARAM_STORE = 2 * 1024 * KiB	# GC_SND_ARAM_BYTES


@dataclass
class Sound:
	rate: int
	samples: "np.ndarray"	# float32, shape (frames, channels), -1..1
	width: int = 2	# bytes per sample in the source / output
	loop_start: int | None = None	# frame
	loop_end: int | None = None	# frame, exclusive

	@property
	def channels(self) -> int:
		return self.samples.shape[1]

	@property
	def frames(self) -> int:
		return self.samples.shape[0]

	@property
	def pcm_bytes(self) -> int:
		"""What Sound_LoadWAV compares against the cap."""
		return self.frames * self.channels * self.width

	@property
	def seconds(self) -> float:
		return self.frames / self.rate if self.rate else 0.0


def _chunks(data: bytes):
	if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
		raise ValueError("not a RIFF/WAVE file")
	pos = 12
	while pos + 8 <= len(data):
		cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
		yield cid, pos + 8, data[pos + 8:pos + 8 + size]
		pos += 8 + size + (size & 1)


def read_wav(data: bytes) -> Sound:
	fmt = body = cue = None
	mark_len = None
	for cid, _off, chunk in _chunks(data):
		if cid == b"fmt ":
			fmt = chunk
		elif cid == b"data":
			body = chunk
		elif cid == b"cue " and len(chunk) >= 28:
			cue = struct.unpack_from("<I", chunk, 24)[0]
		elif cid == b"LIST" and chunk[:4] == b"adtl" and b"mark" in chunk:
			# CoolEdit "ltxt" sub-chunk: id, sample length, purpose "mark"
			i = chunk.find(b"ltxt")
			if i >= 0 and chunk[i + 16:i + 20] == b"mark":
				mark_len = struct.unpack_from("<I", chunk, i + 12)[0]
	if fmt is None or body is None:
		raise ValueError("missing fmt or data chunk")
	tag, channels, rate = struct.unpack_from("<HHI", fmt, 0)
	bits = struct.unpack_from("<H", fmt, 14)[0]
	if tag == 0xFFFE and len(fmt) >= 26:	# WAVE_FORMAT_EXTENSIBLE: real tag in the GUID
		tag = struct.unpack_from("<H", fmt, 24)[0]
	if channels < 1:
		raise ValueError("no channels")
	width = bits // 8
	body = body[:len(body) - len(body) % (width * channels)]
	if tag == 1 and width == 1:
		x = (np.frombuffer(body, np.uint8).astype(np.float32) - 128) / 128
	elif tag == 1 and width == 2:
		x = np.frombuffer(body, "<i2").astype(np.float32) / 32768
	elif tag == 1 and width == 3:
		b = np.frombuffer(body, np.uint8).reshape(-1, 3).astype(np.int32)
		v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
		x = (np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)).astype(np.float32)
	elif tag == 1 and width == 4:
		x = (np.frombuffer(body, "<i4") / float(1 << 31)).astype(np.float32)
	elif tag == 3 and width in (4, 8):
		x = np.frombuffer(body, "<f4" if width == 4 else "<f8").astype(np.float32)
	else:
		raise ValueError(f"unsupported WAV encoding (format {tag}, {bits} bit)")
	x = x.reshape(-1, channels)
	snd = Sound(rate, x, min(width, 2) if tag == 1 else 2)
	if cue is not None and cue < snd.frames:
		snd.loop_start = cue
		if mark_len:
			snd.loop_end = min(snd.frames, cue + mark_len)
	return snd


def write_wav(snd: Sound) -> bytes:
	x = np.clip(snd.samples, -1, 1)
	if snd.width == 1:
		rng = np.random.default_rng(0)
		tpdf = (rng.random(x.shape) - rng.random(x.shape)) / 128	# ±1 LSB triangular dither
		pcm = np.clip(np.round((x + tpdf) * 127 + 128), 0, 255).astype(np.uint8).tobytes()
	else:
		pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes()
	ch = snd.channels
	fmt = struct.pack("<HHIIHH", 1, ch, snd.rate, snd.rate * ch * snd.width, ch * snd.width, snd.width * 8)
	chunks = [(b"fmt ", fmt)]
	if snd.loop_start is not None:
		# Same layout studiomdl-era tools write; the engine reads dwSampleOffset.
		chunks.append((b"cue ", struct.pack("<I6I", 1, 0, 0, 0x61746164, 0, 0, snd.loop_start)))
		if snd.loop_end is not None and snd.loop_end < snd.frames:
			ltxt = struct.pack("<4sIII4s", b"ltxt", 20, 0, snd.loop_end - snd.loop_start, b"mark") + bytes(8)
			chunks.append((b"LIST", b"adtl" + ltxt))
	chunks.append((b"data", pcm))
	body = b"".join(cid + struct.pack("<I", len(c)) + c + bytes(len(c) & 1) for cid, c in chunks)
	return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def resample(x: "np.ndarray", src: int, dst: int) -> "np.ndarray":
	"""Band-limited resample: windowed-sinc low-pass at the new Nyquist, then
	linear interpolation. Plenty for sound effects at 11/22 kHz."""
	if src == dst or len(x) == 0:
		return x
	if dst < src:
		cutoff = 0.5 * dst / src * 0.9
		taps = 63
		n = np.arange(taps) - taps // 2
		h = np.sinc(2 * cutoff * n) * np.hamming(taps)
		h /= h.sum()
		pad = np.pad(x, ((taps // 2, taps // 2), (0, 0)), mode="edge")
		x = np.stack([np.convolve(pad[:, c], h, mode="valid") for c in range(x.shape[1])], 1)
	n_out = max(1, int(round(len(x) * dst / src)))
	t = np.arange(n_out) * (src / dst)
	return np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)


def _with(snd: Sound, **kw) -> Sound:
	d = dict(rate=snd.rate, samples=snd.samples, width=snd.width, loop_start=snd.loop_start, loop_end=snd.loop_end)
	d.update(kw)
	return Sound(**d)


def _rerate(snd: Sound, rate: int) -> Sound:
	if rate == snd.rate:
		return snd
	scale = rate / snd.rate
	ls = None if snd.loop_start is None else int(round(snd.loop_start * scale))
	le = None if snd.loop_end is None else int(round(snd.loop_end * scale))
	return _with(snd, rate=rate, samples=resample(snd.samples, snd.rate, rate), loop_start=ls, loop_end=le)


@dataclass
class FitReport:
	before: int
	after: int
	steps: list[str] = field(default_factory=list)
	ok: bool = True
	truncated: bool = False


def fit_sound(snd: Sound, cap: int = GAMEPLAY_MAX_FILE, *, min_rate: int = 11025, keep_stereo: bool = False,
		allow_8bit: bool = True, truncate: bool = False, max_rate: int = 22050) -> tuple[Sound, FitReport]:
	"""The least lossy version of snd whose PCM data fits in cap bytes."""
	rep = FitReport(snd.pcm_bytes, snd.pcm_bytes)
	cur = snd
	if cur.loop_end is not None and cur.loop_end < cur.frames:
		cur = _with(cur, samples=cur.samples[:cur.loop_end])
		rep.steps.append("cut after loop end")
	if cur.width > 2:
		cur = _with(cur, width=2)
	if cur.rate > max_rate:
		target = max(r for r in (11025, 22050, max_rate) if r <= max_rate)
		cur = _rerate(cur, target)
		rep.steps.append(f"{snd.rate} → {target} Hz")
	if cur.channels > 1 and not keep_stereo and cur.pcm_bytes > cap:
		cur = _with(cur, samples=cur.samples.mean(1, keepdims=True))
		rep.steps.append("mono")

	ladder: list[tuple[int, int]] = []
	for rate in sorted({cur.rate, *(r for r in (22050, 11025, 8000) if r <= cur.rate and r >= min_rate)}, reverse=True):
		ladder.append((rate, 2))
		if allow_8bit:
			ladder.append((rate, 1))
	# Order by size, ties to the higher rate (8-bit 22 kHz and 16-bit 11 kHz cost the same; keep the treble).
	ladder.sort(key=lambda rw: (-(rw[0] * rw[1]), -rw[0]))
	chosen = None
	for rate, width in ladder:
		if cur.width < width:
			continue
		frames = int(round(cur.frames * rate / cur.rate))
		if frames * cur.channels * width <= cap:
			chosen = (rate, width)
			break
	if chosen is None:
		chosen = ladder[-1] if ladder else (cur.rate, cur.width)
	rate, width = chosen
	if rate != cur.rate:
		rep.steps.append(f"{cur.rate} → {rate} Hz")
		cur = _rerate(cur, rate)
	if width != cur.width:
		rep.steps.append("8 bit")
		cur = _with(cur, width=width)

	if cur.pcm_bytes > cap:
		if truncate:
			frames = cap // (cur.channels * cur.width)
			fade = min(frames, max(1, cur.rate // 100))	# 10 ms
			x = cur.samples[:frames].copy()
			x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
			ls = cur.loop_start if cur.loop_start is not None and cur.loop_start < frames else None
			cur = _with(cur, samples=x, loop_start=ls, loop_end=None)
			rep.steps.append(f"truncated to {frames / cur.rate:.2f} s")
			rep.truncated = True
		else:
			rep.ok = False
	rep.after = cur.pcm_bytes
	return cur, rep


def fit_wav_bytes(data: bytes, cap: int = GAMEPLAY_MAX_FILE, **kw) -> tuple[bytes, FitReport]:
	snd = read_wav(data)
	if snd.pcm_bytes <= cap and snd.width <= 2 and snd.channels <= 2 and snd.loop_end is None:
		return data, FitReport(snd.pcm_bytes, snd.pcm_bytes)
	out, rep = fit_sound(snd, cap, **kw)
	return write_wav(out), rep


def wav_pcm_bytes(path: Path) -> int | None:
	"""PCM bytes the engine would count, or None if it is not a readable WAV."""
	try:
		data = Path(path).read_bytes()
		fmt = body = None
		for cid, _off, chunk in _chunks(data):
			if cid == b"fmt ":
				fmt = chunk
			elif cid == b"data":
				body = len(chunk)
		return body if fmt is not None else None
	except (OSError, ValueError, struct.error):
		return None
