"""THP movies: the GameCube's own video format, from any video FFmpeg can read.

THP is what Nintendo's SDK (and most commercial GameCube and Wii games) play
cut-scenes from: a frame-interleaved file of baseline JPEG pictures with an
optional DSP ADPCM stereo track. FFmpeg demuxes and decodes THP, so it is also
the check that what we write is well formed.

Layout (all big-endian, version 1.1):

	0x00 header        "THP\\0", version 0x00011000, max frame bytes, max audio
	                   samples per frame, fps (f32), frame count, first frame
	                   bytes, movie data bytes, component info offset, offset
	                   table offset (0: none), first frame offset, last frame offset
	0x30 components    count, 16 type bytes (0 video, 1 audio, 0xFF none), then
	                   video: width, height, format (0 progressive)
	                   audio: channels, rate, total samples, tracks (1)
	frames             32-byte aligned, back to back:
	                   next frame bytes, previous frame bytes, video bytes,
	                   [audio bytes], JPEG (padded to 4), [audio block], pad to 32

The audio block of a frame: bytes per channel, samples, 8 coefficient pairs
for each channel, each channel's two history samples at the block start, then
each channel's DSP ADPCM frames. Each video frame carries a whole number of
14-sample ADPCM frames, so the blocks are cut from one continuous encode and
every block decodes on its own.

JPEG pictures are baseline, 4:2:0, standard Huffman tables, no restart
markers and no APP/COM segments; sides are multiples of 16 (whole MCUs).
Unlike a .jpg, the scan is not byte-stuffed: a 0xFF in the entropy data is
not followed by 0x00 (the decoder knows the picture's length from the frame
header, so it never searches for markers). FFmpeg's THP decoder reads it that
way, and so does port/gc_thp.c.
"""

from __future__ import annotations

import io
import json
import math
import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import dspadpcm, media

MAGIC = b"THP\0"
VERSION = 0x00011000
HEADER = struct.Struct(">4sIIIfIIIIIII")
assert HEADER.size == 0x30
COMP_VIDEO, COMP_AUDIO, COMP_NONE = 0, 1, 0xFF
AUDIO_HEADER_BYTES = 8 + 2 * 32 + 2 * 4	# stereo: sizes, two coefficient tables, two histories
ALIGN = 32

MAX_WIDTH = 640	# the GameCube's EFB / video interface width
MAX_HEIGHT = 480
DEFAULT_BOX = (512, 448)	# fits 4:3 as 512x384 and 16:9 as 512x288
DEFAULT_MAX_FPS = 30.0
DEFAULT_QUALITY = 80
DEFAULT_AUDIO_RATE = 32000
AUDIO_RATES = (32000, 48000)	# the AI's two native rates


@dataclass
class ThpOptions:
	width: int = 0	# 0: fit the source inside box_width x box_height
	height: int = 0
	box_width: int = DEFAULT_BOX[0]
	box_height: int = DEFAULT_BOX[1]
	fps: float = 0.0	# 0: the source's, divided down to max_fps or below
	max_fps: float = DEFAULT_MAX_FPS
	quality: int = DEFAULT_QUALITY	# JPEG quality, 1-100
	min_quality: int = 10	# lowest quality a frame drops to under a size cap
	max_kbps: float = 0.0	# per-frame cap: video bytes per second x 8 / 1000 (0: none)
	max_bytes: int = 0	# whole-file cap; sets the per-frame cap from the length (0: none)
	audio: bool = True
	audio_rate: int = DEFAULT_AUDIO_RATE
	audio_track: int = 0
	start: float = 0.0	# seconds into the source
	duration: float = 0.0	# seconds (0: to the end)
	letterbox: bool = True	# with an explicit width/height: pad to keep the aspect instead of stretching


@dataclass
class SourceInfo:
	width: int
	height: int
	fps: float
	duration: float
	sar: float = 1.0
	audio_channels: int = 0
	audio_rate: int = 0
	video_codec: str = ""
	audio_codec: str = ""

	@property
	def display_aspect(self) -> float:
		return self.width * self.sar / max(self.height, 1)


@dataclass
class ThpInfo:
	version: int
	width: int
	height: int
	fps: float
	frames: int
	max_frame_bytes: int
	first_frame: int
	last_frame: int
	data_bytes: int
	audio_channels: int = 0
	audio_rate: int = 0
	audio_samples: int = 0
	max_audio_samples: int = 0

	@property
	def seconds(self) -> float:
		return self.frames / self.fps if self.fps else 0.0

	def as_dict(self) -> dict:
		return {"Frame size": f"{self.width}x{self.height}", "FPS": round(self.fps, 3), "Frames": self.frames,
			"Seconds": round(self.seconds, 2), "Largest frame": self.max_frame_bytes, "Video": "JPEG 4:2:0",
			"Audio": f"DSP ADPCM {self.audio_channels} ch {self.audio_rate} Hz" if self.audio_channels else "none"}


@dataclass
class ThpResult:
	path: Path
	bytes: int
	width: int
	height: int
	fps: float
	frames: int
	audio_rate: int = 0
	audio_bytes: int = 0
	video_bytes: int = 0
	quality_min: int = 0
	quality_max: int = 0
	over_cap: int = 0	# frames still over the cap at min_quality
	warnings: list[str] = field(default_factory=list)

	def as_dict(self) -> dict:
		return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


# --- probing and planning -----------------------------------------------------

def find_ffprobe(ffmpeg: str | None = None) -> str | None:
	import os
	import shutil

	ffmpeg = ffmpeg or media.find_ffmpeg()
	if ffmpeg:
		p = Path(ffmpeg)
		cand = p.with_name(p.name.replace("ffmpeg", "ffprobe"))
		if cand != p and cand.is_file():
			return str(cand)
	env = os.environ.get("GXBAKE_FFPROBE")
	if env and Path(env).is_file():
		return env
	return shutil.which("ffprobe")


def _rate(text: str | None) -> float:
	if not text or text in ("0/0", "N/A"):
		return 0.0
	if "/" in text or ":" in text:
		n, d = re.split("[/:]", text, maxsplit=1)
		return float(n) / float(d) if float(d) else 0.0
	return float(text)


def probe(source: Path, *, ffmpeg: str | None = None, audio_track: int = 0) -> SourceInfo:
	"""Size, frame rate, length and audio of a video, through ffprobe."""
	ffprobe = find_ffprobe(ffmpeg)
	if ffprobe is None:
		raise FileNotFoundError("ffprobe is required to read videos (it ships with ffmpeg)")
	res = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(source)],
		capture_output=True, text=True)
	if res.returncode != 0:
		raise ValueError(f"ffprobe cannot read {Path(source).name}: {res.stderr.strip()[-300:]}")
	d = json.loads(res.stdout or "{}")
	streams = d.get("streams", [])
	video = next((s for s in streams if s.get("codec_type") == "video"
		and not s.get("disposition", {}).get("attached_pic")), None)
	if video is None:
		raise ValueError(f"{Path(source).name} has no video stream")
	audios = [s for s in streams if s.get("codec_type") == "audio"]
	audio = audios[audio_track] if audio_track < len(audios) else None
	fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")) or 25.0
	if fps > 240:	# timebase-sized "rates" from VFR containers
		fps = _rate(video.get("r_frame_rate")) if _rate(video.get("r_frame_rate")) <= 240 else 30.0
	sar = _rate(video.get("sample_aspect_ratio")) or 1.0
	rot = str(video.get("tags", {}).get("rotate", "0"))
	for side in video.get("side_data_list", []):
		if "rotation" in side:	# display matrix (FFmpeg 5+); FFmpeg turns the frames upright itself
			rot = str(abs(int(float(side["rotation"]))) % 360)
	if rot.lstrip("-") in ("90", "270"):
		w, h = int(video["height"]), int(video["width"])
	else:
		w, h = int(video["width"]), int(video["height"])
	dur = float(video.get("duration") or d.get("format", {}).get("duration") or 0)
	return SourceInfo(w, h, fps, dur, sar,
		int(audio.get("channels", 0)) if audio else 0, int(audio.get("sample_rate", 0)) if audio else 0,
		video.get("codec_name", ""), audio.get("codec_name", "") if audio else "")


def _round16(v: float) -> int:
	return max(16, int(v) // 16 * 16)


def plan_size(src: SourceInfo, opts: ThpOptions) -> tuple[int, int]:
	"""Output size: explicit (rounded to 16) or the source fitted inside the box."""
	if opts.width and opts.height:
		w, h = _round16(opts.width), _round16(opts.height)
	else:
		bw, bh = min(opts.box_width or MAX_WIDTH, MAX_WIDTH), min(opts.box_height or MAX_HEIGHT, MAX_HEIGHT)
		aspect = src.display_aspect or 4 / 3
		w, h = bw, bw / aspect
		if h > bh:
			w, h = bh * aspect, bh
		w, h = min(w, src.width * src.sar if src.sar >= 1 else src.width), min(h, src.height)	# never upscale
		w, h = _round16(round(w)), _round16(round(h))
	return min(w, MAX_WIDTH), min(h, MAX_HEIGHT)


def plan_fps(src_fps: float, opts: ThpOptions) -> float:
	"""Explicit, or the source's divided by the smallest whole number that gets under max_fps
	(59.94 → 29.97, 60 → 30, 50 → 25), so frames are dropped evenly."""
	if opts.fps:
		return float(opts.fps)
	f = src_fps or 30.0
	cap = opts.max_fps or DEFAULT_MAX_FPS
	if f <= cap + 1e-6:
		return f
	return f / math.ceil(f / cap - 1e-6)


def audio_boundaries(frames: int, fps: float, rate: int) -> list[int]:
	"""Sample index where each frame's audio starts (and the end), on 14-sample ADPCM frames."""
	step = dspadpcm.SAMPLES_PER_FRAME
	return [int(round(k * rate / fps / step)) * step for k in range(frames + 1)]


# --- encoding -----------------------------------------------------------------

def _strip_jpeg(data: bytes) -> bytes:
	"""Drop APPn and COM segments (JFIF/EXIF headers): THP pictures carry only tables and the scan."""
	out = bytearray(data[:2])
	i = 2
	while i + 4 <= len(data) and data[i] == 0xFF:
		marker = data[i + 1]
		if marker == 0xDA:	# start of scan: header, entropy data, EOI
			seg = struct.unpack_from(">H", data, i + 2)[0]
			body = data[i + 2 + seg:]
			if body.endswith(b"\xff\xd9"):
				body = body[:-2]
			# THP scans are not byte-stuffed: a 0xFF in the entropy data is not followed by 0x00
			out += data[i:i + 2 + seg] + body.replace(b"\xff\x00", b"\xff") + b"\xff\xd9"
			return bytes(out)
		seg = struct.unpack_from(">H", data, i + 2)[0]
		if not (0xE0 <= marker <= 0xEF or marker == 0xFE):
			out += data[i:i + 2 + seg]
		i += 2 + seg
	return bytes(out) + data[i:]


def encode_jpeg(rgb: "np.ndarray", quality: int) -> bytes:
	from PIL import Image

	buf = io.BytesIO()
	Image.fromarray(rgb, "RGB").save(buf, "JPEG", quality=int(quality), subsampling=2,
		optimize=False, progressive=False)
	return _strip_jpeg(buf.getvalue())


def _fit_quality(rgb, quality: int, cap: int, floor: int, last: int | None) -> tuple[bytes, int, bool]:
	"""Highest quality ≤ `quality` whose picture fits `cap` bytes (binary search from the last frame's)."""
	jpg = encode_jpeg(rgb, quality)
	if not cap or len(jpg) <= cap:
		return jpg, quality, False
	lo, hi = floor, quality - 1
	best: tuple[bytes, int] | None = None
	guess = min(last, hi) if last else (lo + hi) // 2
	while lo <= hi:
		q = guess if guess is not None else (lo + hi) // 2
		guess = None
		j = encode_jpeg(rgb, q)
		if len(j) <= cap:
			best, lo = (j, q), q + 1
		else:
			hi = q - 1
	if best is None:
		return encode_jpeg(rgb, floor), floor, True
	return best[0], best[1], False


def _video_filter(src: SourceInfo, w: int, h: int, fps: float, opts: ThpOptions) -> str:
	vf = [f"fps={fps:.6f}"]
	if opts.width and opts.height and opts.letterbox:
		vf.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos")
		vf.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")
	else:
		vf.append(f"scale={w}:{h}:flags=lanczos")
	vf.append("setsar=1")
	vf.append("format=rgb24")
	return ",".join(vf)


def _trim_args(opts: ThpOptions) -> list[str]:
	args = []
	if opts.start:
		args += ["-ss", f"{opts.start:.3f}"]
	return args


def _read_audio(source: Path, rate: int, opts: ThpOptions, ffmpeg: str) -> "np.ndarray | None":
	cmd = [ffmpeg, "-v", "error", *_trim_args(opts), "-i", str(source)]
	if opts.duration:
		cmd += ["-t", f"{opts.duration:.3f}"]
	cmd += ["-map", f"0:a:{opts.audio_track}", "-vn", "-ar", str(rate), "-ac", "2", "-f", "s16le", "-acodec", "pcm_s16le", "-"]
	res = subprocess.run(cmd, capture_output=True)
	if res.returncode != 0:
		if b"matches no streams" in res.stderr:
			return None
		raise ValueError(f"ffmpeg could not decode the audio: {res.stderr.decode(errors='replace').strip()[-300:]}")
	if not res.stdout:
		return None
	return np.frombuffer(res.stdout[: len(res.stdout) // 4 * 4], "<i2").reshape(-1, 2)


@dataclass
class _Audio:
	rate: int
	coefs: list["np.ndarray"]	# per channel, 8x2 s16
	data: list[bytes]	# per channel, ADPCM frames
	ctx: list[list[tuple[int, int, int]]]	# per channel, per ADPCM frame: (header, h1, h2) at its start
	bounds: list[int]

	def block(self, k: int) -> bytes:
		a, b = self.bounds[k], self.bounds[k + 1]
		n = b - a
		fa, fb = a // dspadpcm.SAMPLES_PER_FRAME, b // dspadpcm.SAMPLES_PER_FRAME
		size = (fb - fa) * dspadpcm.BYTES_PER_FRAME
		out = bytearray(struct.pack(">II", size, n))
		for c in self.coefs:
			out += struct.pack(">16h", *c.reshape(-1).tolist())
		for ctx in self.ctx:
			_, h1, h2 = ctx[fa] if fa < len(ctx) else (0, 0, 0)
			out += struct.pack(">hh", h1, h2)
		for d in self.data:
			out += d[fa * 8:fb * 8]
		return bytes(out)


def _encode_audio(pcm: "np.ndarray", frames: int, fps: float, rate: int) -> _Audio:
	bounds = audio_boundaries(frames, fps, rate)
	total = bounds[-1]
	x = np.zeros((total, 2), np.int16)
	n = min(total, len(pcm))
	x[:n] = pcm[:n]
	coefs, data, ctx = [], [], []
	for c in range(2):
		ch = x[:, c].astype(np.int64)
		cf = dspadpcm.design_coefs(ch)
		blob, cx = dspadpcm._encode_frames(ch, cf, fast=len(ch) > 10 * rate)
		coefs.append(cf)
		data.append(blob)
		ctx.append(cx)
	return _Audio(rate, coefs, data, ctx, bounds)


def _pad(b: bytes | bytearray, n: int) -> bytes:
	return bytes(b) + bytes(-len(b) % n)


def write_thp(pictures: list[bytes], fps: float, width: int, height: int, audio: _Audio | None = None) -> bytes:
	"""Assemble a THP file from JPEG pictures (and an encoded audio track)."""
	if not pictures:
		raise ValueError("no frames")
	n = len(pictures)
	ncomp = 2 if audio else 1
	comp = struct.pack(">I", ncomp) + bytes([COMP_VIDEO] + ([COMP_AUDIO] if audio else []) + [COMP_NONE] * (16 - ncomp))
	comp += struct.pack(">III", width, height, 0)
	if audio:
		comp += struct.pack(">IIII", 2, audio.rate, audio.bounds[-1], 1)
	first = -(-(HEADER.size + len(comp)) // ALIGN) * ALIGN
	bodies, sizes = [], []
	for k, pic in enumerate(pictures):
		video = _pad(pic, 4)
		sound = audio.block(k) if audio else b""
		head_len = 8 + 4 * ncomp
		total = -(-(head_len + len(video) + len(sound)) // ALIGN) * ALIGN
		sizes.append(total)
		bodies.append((video, sound))
	out = bytearray()
	for k, (video, sound) in enumerate(bodies):
		nxt = sizes[k + 1] if k + 1 < n else sizes[0]	# wraps, so a looping player reads frame 0 next
		prv = sizes[k - 1] if k else sizes[-1]
		fr = struct.pack(">III", nxt, prv, len(video)) + (struct.pack(">I", len(sound)) if audio else b"")
		out += _pad(fr + video + sound, ALIGN)
	max_samples = max(b - a for a, b in zip(audio.bounds, audio.bounds[1:])) if audio else 0
	header = HEADER.pack(MAGIC, VERSION, max(sizes), max_samples, float(fps), n, sizes[0], len(out),
		HEADER.size, 0, first, first + sum(sizes[:-1]))
	return _pad(header + comp, ALIGN) + bytes(out)


def convert(
	source: Path,
	output: Path,
	opts: ThpOptions | None = None,
	*,
	ffmpeg: str | None = None,
	log: Callable[[str], None] = print,
	progress: Callable[[int, int], None] | None = None,
) -> ThpResult:
	"""Any video FFmpeg reads → a THP file."""
	opts = opts or ThpOptions()
	ffmpeg = ffmpeg or media.find_ffmpeg()
	if ffmpeg is None:
		raise FileNotFoundError("ffmpeg is required to convert videos (install it or set GXBAKE_FFMPEG)")
	if opts.audio_rate not in AUDIO_RATES:
		raise ValueError(f"audio rate must be one of {AUDIO_RATES}")
	src = probe(source, ffmpeg=ffmpeg, audio_track=opts.audio_track)
	w, h = plan_size(src, opts)
	fps = plan_fps(src.fps, opts)
	length = (opts.duration or max(src.duration - opts.start, 0.0)) or 0.0
	expect = max(1, int(round(length * fps))) if length else 0
	warnings: list[str] = []

	pcm = None
	if opts.audio and src.audio_channels:
		pcm = _read_audio(source, opts.audio_rate, opts, ffmpeg)
		if pcm is None:
			warnings.append("audio stream could not be decoded; writing video only")
	audio_rate = opts.audio_rate if pcm is not None else 0

	cap = 0
	if opts.max_kbps:
		cap = int(opts.max_kbps * 1000 / 8 / fps)
	if opts.max_bytes and expect:
		audio_est = int(length * audio_rate * 2 * 8 / 14) + expect * (AUDIO_HEADER_BYTES + 4) if audio_rate else 0
		per = (opts.max_bytes - audio_est - 0x60) // expect - 16 - 31	# frame header and alignment
		if per < 512:
			raise ValueError(f"{opts.max_bytes} bytes is too small for {expect} frames of {w}x{h}")
		cap = min(cap, per) if cap else per

	log(f"{Path(source).name}: {src.video_codec} {src.width}x{src.height} @ {src.fps:.3f} fps"
		+ (f", {src.audio_codec} {src.audio_channels} ch {src.audio_rate} Hz" if src.audio_channels else ", no audio")
		+ f" → THP {w}x{h} @ {fps:.3f} fps, JPEG q{opts.quality}"
		+ (f" (≤{cap} bytes a frame)" if cap else "")
		+ (f", DSP ADPCM {audio_rate} Hz stereo" if audio_rate else ""))

	cmd = [ffmpeg, "-v", "error", *_trim_args(opts), "-i", str(source)]
	if opts.duration:
		cmd += ["-t", f"{opts.duration:.3f}"]
	cmd += ["-map", "0:v:0", "-an", "-sn", "-vf", _video_filter(src, w, h, fps, opts), "-f", "rawvideo", "-"]
	frame_bytes = w * h * 3
	pictures: list[bytes] = []
	qs: list[int] = []
	over = 0
	last_q = None
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	try:
		while True:
			raw = proc.stdout.read(frame_bytes)
			if len(raw) < frame_bytes:
				break
			rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
			jpg, q, too_big = _fit_quality(rgb, opts.quality, cap, opts.min_quality, last_q)
			last_q = q if q < opts.quality else None
			over += too_big
			pictures.append(jpg)
			qs.append(q)
			if progress:
				progress(len(pictures), max(expect, len(pictures)))
	finally:
		proc.stdout.close()
		err = proc.stderr.read()
		proc.stderr.close()
		proc.wait()
	if proc.returncode != 0 and not pictures:
		raise ValueError(f"ffmpeg could not decode {Path(source).name}: {err.decode(errors='replace').strip()[-300:]}")
	if not pictures:
		raise ValueError(f"ffmpeg produced no frames for {Path(source).name}")
	if over:
		warnings.append(f"{over} frame(s) stay over the {cap}-byte cap at quality {opts.min_quality}")

	audio = _encode_audio(pcm, len(pictures), fps, audio_rate) if pcm is not None else None
	blob = write_thp(pictures, fps, w, h, audio)
	output = Path(output)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_bytes(blob)
	vbytes = sum(len(p) for p in pictures)
	abytes = sum(len(d) for d in audio.data) if audio else 0
	res = ThpResult(output, len(blob), w, h, fps, len(pictures), audio_rate, abytes, vbytes, min(qs), max(qs), over, warnings)
	secs = len(pictures) / fps
	log(f"Built THP {output.name}: {len(pictures)} frames, {secs:.1f} s, {len(blob) / 1024:.0f} KiB "
		f"({len(blob) * 8 / 1000 / max(secs, 1e-9):.0f} kbit/s), JPEG q{min(qs)}–{max(qs)}")
	for w_ in warnings:
		log(f"  warning: {w_}")
	return res


# --- reading ------------------------------------------------------------------

def read_info(data: bytes) -> ThpInfo:
	if data[:4] != MAGIC:
		raise ValueError("not a THP file")
	(_, ver, maxbuf, maxsamp, fps, frames, _first_size, data_size, comp_off, _offs, first, last) = HEADER.unpack_from(data)
	count = struct.unpack_from(">I", data, comp_off)[0]
	types = data[comp_off + 4:comp_off + 20]
	pos = comp_off + 20
	info = ThpInfo(ver, 0, 0, fps, frames, maxbuf, first, last, data_size, max_audio_samples=maxsamp)
	for t in types[:count]:
		if t == COMP_VIDEO:
			info.width, info.height = struct.unpack_from(">II", data, pos)
			pos += 12 if ver == VERSION else 8
		elif t == COMP_AUDIO:
			info.audio_channels, info.audio_rate, info.audio_samples = struct.unpack_from(">III", data, pos)
			pos += 16 if ver == VERSION else 12
	return info


def iter_frames(data: bytes):
	"""(video bytes, audio block or b"") for each frame."""
	info = read_info(data)
	has_audio = info.audio_channels > 0
	pos = info.first_frame
	size = struct.unpack_from(">I", data, 0x18)[0]	# first frame bytes
	for _ in range(info.frames):
		nxt, _prv, vsize = struct.unpack_from(">III", data, pos)
		asize = struct.unpack_from(">I", data, pos + 12)[0] if has_audio else 0
		body = pos + 12 + (4 if has_audio else 0)
		yield data[body:body + vsize], data[body + vsize:body + vsize + asize]
		pos += size
		size = nxt


def cues(data: bytes, passes: int = 1) -> list[tuple[int, int, int, int]]:
	"""When each frame's picture is due, as port/gc_thp.c's player gives it: (index, pass, due,
	until) in clock ticks, which are sample pairs of audio when the movie has sound (so the
	picture follows the sound) and milliseconds otherwise. `passes` > 1 plays it looped."""
	info = read_info(data)
	blocks = [a for _, a in iter_frames(data)]
	out, clock, k = [], 0, 0
	fps = float(info.fps)

	def ms(n: int) -> int:
		return int(n * 1000.0 / fps + 0.5) & 0xFFFFFFFF

	for p in range(passes):
		for i, a in enumerate(blocks):
			if info.audio_channels:
				n = struct.unpack_from(">I", a, 4)[0] if a else 0
				out.append((i, p, clock, (clock + n) & 0xFFFFFFFF))
				clock = (clock + n) & 0xFFFFFFFF
			else:
				out.append((i, p, ms(k), ms(k + 1)))
			k += 1
	return out


def decode_audio_block(block: bytes) -> "np.ndarray":
	"""One frame's audio block → s16 samples, shape (n, channels)."""
	size, n = struct.unpack_from(">II", block, 0)
	coefs = [np.array(struct.unpack_from(">16h", block, 8 + 32 * c), np.int16).reshape(8, 2) for c in range(2)]
	hist = [struct.unpack_from(">hh", block, 72 + 4 * c) for c in range(2)]
	data = block[AUDIO_HEADER_BYTES:]
	return np.stack([dspadpcm.decode_frames(data[c * size:(c + 1) * size], coefs[c], n, *hist[c]) for c in range(2)], 1)


def to_jpeg(picture: bytes) -> bytes:
	"""A THP picture as an ordinary .jpg (byte-stuffs the scan again), for Pillow or any JPEG reader."""
	i = 2
	while i + 4 <= len(picture) and picture[i] == 0xFF:
		seg = struct.unpack_from(">H", picture, i + 2)[0]
		if picture[i + 1] == 0xDA:
			body = picture[i + 2 + seg:].rstrip(b"\0")
			if body.endswith(b"\xff\xd9"):
				body = body[:-2]
			return picture[:i + 2 + seg] + body.replace(b"\xff", b"\xff\x00") + b"\xff\xd9"
		i += 2 + seg
	raise ValueError("no scan in picture")


def frame_rgba(data: bytes, index: int = 0) -> "np.ndarray":
	from PIL import Image

	for k, (video, _) in enumerate(iter_frames(data)):
		if k == index:
			with Image.open(io.BytesIO(to_jpeg(video))) as img:
				return np.asarray(img.convert("RGBA")).copy()
	raise IndexError(index)
