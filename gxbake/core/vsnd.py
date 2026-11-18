"""Source 2 sounds (.vsnd_c) → a PCM WAV the port's sound loader takes.

The sample data sits after the resource (at the file size the header gives),
as PCM, Microsoft ADPCM or MP3. PCM and ADPCM decode here; MP3 goes through
ffmpeg when it is installed. The header layout (resource versions 0-4 in DATA,
5 in CTRL) follows ValveResourceFormat (MIT, see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import vres


@dataclass
class VSnd:
	kind: str	# "pcm", "adpcm", "mp3"
	rate: int
	channels: int
	bits: int
	samples: int
	loop_start: int
	loop_end: int
	duration: float
	data: bytes
	adpcm_header: bytes = b""

	def info(self) -> dict:
		out = {"Format": self.kind.upper() + (f" {self.bits}-bit" if self.kind == "pcm" else ""),
			"Rate": f"{self.rate} Hz", "Channels": self.channels, "Duration": f"{self.duration:.2f} s"}
		if self.loop_start >= 0 and self.loop_end > self.loop_start:
			out["Loop"] = f"{self.loop_start}–{self.loop_end}"
		return out


def _bits(v: int, off: int, n: int) -> int:
	return (v >> off) & ((1 << n) - 1)


def parse(res: vres.Resource) -> VSnd:
	ctrl = res.keyvalues("CTRL") if res.block("CTRL") else None
	stream = res.data[res.file_size:]
	if isinstance(ctrl, dict) and ctrl.get("_class") in ("CVoiceContainerDefault", "CVoiceContainerEnvelope"):
		s = ctrl.get("m_vSound") or {}
		fmt = str(s.get("m_nFormat", "PCM16"))
		kind, bits = {"MP3": ("mp3", 0), "PCM8": ("pcm", 8), "PCM16": ("pcm", 16)}.get(fmt, ("pcm", 16))
		size = int(s.get("m_nStreamingSize", len(stream)))
		return VSnd(kind, int(s.get("m_nRate", 44100)), int(s.get("m_nChannels", 1)), bits,
			int(s.get("m_nSampleCount", 0)), int(s.get("m_nLoopStart", -1)), int(s.get("m_nLoopEnd", 0)),
			float(s.get("m_flDuration", 0.0)), stream[:size])
	b = res.block("DATA")
	if b is None or b.size == 0:
		raise ValueError("sound has no sample data (it only references other sounds)")
	d = res.data
	p = b.offset
	if res.version > 4:
		raise ValueError(f"unknown vsnd version {res.version}")
	if res.version >= 4:
		rate = struct.unpack_from("<H", d, p)[0]
		fmt, channels = d[p + 2], d[p + 3]
		p += 4
		kind, bits = {0: ("pcm", 16), 1: ("pcm", 8), 2: ("mp3", 0), 3: ("adpcm", 16)}.get(fmt, ("pcm", 16))
	else:
		packed = struct.unpack_from("<I", d, p)[0]
		p += 4
		typ, bits = _bits(packed, 0, 2), _bits(packed, 2, 5)
		wave_fmt = _bits(packed, 12, 2)
		channels = _bits(packed, 7, 2)
		rate = _bits(packed, 14, 16)
		if typ == 2:
			kind, bits = "mp3", 0
		elif wave_fmt == 2:
			kind, bits = "adpcm", 16
		else:
			kind, bits = "pcm", 8 if bits == 8 else 16
	loop_start, samples = struct.unpack_from("<iI", d, p)
	duration = struct.unpack_from("<f", d, p + 8)[0]
	header_at = p + 16
	header_off, header_size, stream_size = struct.unpack_from("<IiI", d, header_at)
	q = header_at + 12
	if res.version >= 1:
		q += 8
	if res.version >= 2:
		q += 4
	loop_end = struct.unpack_from("<i", d, q)[0] if res.version >= 4 else 0
	adpcm = bytes(d[header_at + header_off:header_at + header_off + header_size]) if header_size > 0 else b""
	if res.version < 3 and kind == "mp3":
		loop_start = -1
	return VSnd(kind, rate, max(1, channels), bits, samples, loop_start, loop_end, duration,
		stream[:stream_size], adpcm)


def load(path: Path) -> VSnd:
	return parse(vres.load(path))


# -- decoding ---------------------------------------------------------------------

_ADAPT = (230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230)
_COEF = ((256, 0), (512, -256), (0, 0), (192, 64), (240, 0), (460, -208), (392, -232))


def decode_ms_adpcm(data: bytes, channels: int, block_align: int, coefs=_COEF) -> "np.ndarray":
	"""Microsoft ADPCM → int16 samples (frames × channels)."""
	out: list[int] = []
	for start in range(0, len(data), block_align):
		blk = data[start:start + block_align]
		if len(blk) < 7 * channels:
			break
		pred = list(blk[:channels])
		p = channels
		delta = list(struct.unpack_from(f"<{channels}h", blk, p))
		p += 2 * channels
		s1 = list(struct.unpack_from(f"<{channels}h", blk, p))
		p += 2 * channels
		s2 = list(struct.unpack_from(f"<{channels}h", blk, p))
		p += 2 * channels
		frame = []
		for c in range(channels):
			frame.append(s2[c])
		out += frame
		out += s1
		co = [coefs[min(x, len(coefs) - 1)] for x in pred]
		ch = 0
		for byte in blk[p:]:
			for nib in (byte >> 4, byte & 15):
				signed = nib - 16 if nib & 8 else nib
				c1, c2 = co[ch]
				predict = (s1[ch] * c1 + s2[ch] * c2) >> 8
				val = max(-32768, min(32767, predict + signed * delta[ch]))
				out.append(val)
				s2[ch], s1[ch] = s1[ch], val
				delta[ch] = max(16, (_ADAPT[nib] * delta[ch]) >> 8)
				ch = (ch + 1) % channels
	n = len(out) // channels * channels
	return np.array(out[:n], np.int16).reshape(-1, channels)


def _wav(pcm: "np.ndarray", rate: int, bits: int = 16) -> bytes:
	ch = pcm.shape[1] if pcm.ndim == 2 else 1
	body = (pcm.astype(np.int16).astype("<i2").tobytes() if bits == 16 else pcm.astype(np.uint8).tobytes())
	align = ch * bits // 8
	fmt = struct.pack("<HHIIHH", 1, ch, rate, rate * align, align, bits)
	return b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + \
		b"data" + struct.pack("<I", len(body)) + body


def to_wav(snd: VSnd) -> bytes:
	"""A Microsoft PCM WAV (16-bit, or 8-bit when the source is 8-bit)."""
	if snd.kind == "pcm":
		if snd.bits == 8:
			pcm = np.frombuffer(snd.data[:len(snd.data) // snd.channels * snd.channels], np.uint8)
			return _wav(pcm.reshape(-1, snd.channels), snd.rate, 8)
		pcm = np.frombuffer(snd.data[:len(snd.data) // (2 * snd.channels) * 2 * snd.channels], "<i2")
		return _wav(pcm.reshape(-1, snd.channels), snd.rate)
	if snd.kind == "adpcm":
		h = snd.adpcm_header
		channels = snd.channels
		block_align = 512 * channels
		coefs = _COEF
		if len(h) >= 16:
			_tag, channels, _rate, _bps, block_align = struct.unpack_from("<HHIIH", h, 0)
			if len(h) >= 22:
				ncoef = struct.unpack_from("<H", h, 20)[0]
				if ncoef and len(h) >= 22 + 4 * ncoef:
					coefs = [struct.unpack_from("<hh", h, 22 + 4 * i) for i in range(ncoef)]
		pcm = decode_ms_adpcm(snd.data, max(1, channels), max(1, block_align), coefs)
		if snd.samples and len(pcm) > snd.samples:
			pcm = pcm[:snd.samples]
		return _wav(pcm, snd.rate)
	ffmpeg = shutil.which("ffmpeg")
	if ffmpeg is None:
		raise RuntimeError("this sound is MP3; ffmpeg is needed to decode it")
	with tempfile.TemporaryDirectory() as tmp:
		src = Path(tmp) / "in.mp3"
		src.write_bytes(snd.data)
		dst = Path(tmp) / "out.wav"
		subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-acodec", "pcm_s16le", str(dst)], check=True)
		return dst.read_bytes()


def to_port_wav(snd: VSnd, max_rate: int = 22050) -> bytes:
	"""to_wav at no more than max_rate (GoldSrc's 22050 Hz), with the loop as a cue point.

	The result is what sound/<name>.wav holds; `gxbake sfx` can shrink it
	further to the port's per-file cap.
	"""
	from . import sfx

	s = sfx.read_wav(to_wav(snd))
	if snd.loop_start >= 0 and (snd.kind != "mp3" or snd.loop_end > snd.loop_start):
		s.loop_start = min(snd.loop_start, s.frames - 1) if s.frames else None
		if snd.loop_end > snd.loop_start:
			s.loop_end = min(snd.loop_end, s.frames)
	if s.rate > max_rate:
		s = sfx._rerate(s, max_rate)
	return sfx.write_wav(s)
