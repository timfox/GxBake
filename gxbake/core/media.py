"""Movies and audio: GCVID delta video and GCPCM 48 kHz big-endian PCM.

build_gcvid_companion and build_gcpcm_companion are ported from
scripts/build-gamecube-disc.py (used by convert-avi-to-gcvid.py). The only
changes: ffmpeg can be pointed at an explicit binary (bundled builds), the
RGB565 conversion is vectorised with numpy (same bytes, far faster), and
log lines go through a callback instead of print().
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

GCVID_MAGIC = b"GCV2"
GCVID_WIDTH = 320
GCVID_HEIGHT = 240
GCVID_FPS_NUM = 15
GCVID_FPS_DEN = 1
GCVID_TILE_SIZE = 8
GCVID_FLAG_STATIC_HOLD = 1 << 31
GCVID_FLAG_BGRA32 = 1 << 30
GCVID_KEYFRAME = 0
GCVID_DELTAFRAME = 1
GCVID_STILL_FRAME_INDEX = 80
GCPCM_MAGIC = b"GCPA"
GCPCM_RATE = 48000
GCPCM_CHANNELS = 2
GCPCM_WIDTH = 2

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm", ".mpg", ".mpeg", ".wmv", ".bik"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aif", ".aiff", ".voc", ".opus", ".m4a", ".wma"}


def find_ffmpeg() -> str | None:
	"""GXBAKE_FFMPEG, then an ffmpeg next to the executable, then PATH."""
	env = os.environ.get("GXBAKE_FFMPEG")
	if env and Path(env).is_file():
		return env
	import sys

	here = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
	for name in ("ffmpeg", "ffmpeg.exe"):
		cand = here / name
		if cand.is_file():
			return str(cand)
	return shutil.which("ffmpeg")


def _bgra_to_rgb565le(frame: bytes) -> bytes:
	px = np.frombuffer(frame, np.uint8).reshape(-1, 4).astype(np.uint16)
	b, g, r = px[:, 0], px[:, 1], px[:, 2]
	return (((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)).astype("<u2").tobytes()


def build_gcvid_companion(
	source_movie: Path,
	output_movie: Path,
	*,
	width: int = GCVID_WIDTH,
	height: int = GCVID_HEIGHT,
	fps_num: int = GCVID_FPS_NUM,
	fps_den: int = GCVID_FPS_DEN,
	still_frame_index: int = GCVID_STILL_FRAME_INDEX,
	rgb565: bool = False,
	ffmpeg: str | None = None,
	log: Callable[[str], None] = print,
) -> dict:
	ffmpeg = ffmpeg or find_ffmpeg()
	if ffmpeg is None:
		raise FileNotFoundError("ffmpeg is required to build .gcvid intro companions")

	raw_frames = subprocess.run(
		[
			ffmpeg,
			"-v", "error",
			"-i", str(source_movie),
			"-an",
			"-vf", f"fps={fps_num}/{fps_den},scale={width}:{height},format=bgra",
			"-pix_fmt", "bgra",
			"-f", "rawvideo",
			"-vsync", "cfr",
			"-",
		],
		capture_output=True,
		check=True,
	)
	frame_size = width * height * 4
	if len(raw_frames.stdout) % frame_size != 0:
		raise ValueError(f"ffmpeg raw frame size mismatch for {source_movie}")
	actual_frames = len(raw_frames.stdout) // frame_size
	if actual_frames <= 0:
		raise ValueError(f"ffmpeg produced no frames for {source_movie}")

	tile_size = GCVID_TILE_SIZE
	if width % GCVID_TILE_SIZE != 0 or height % GCVID_TILE_SIZE != 0:
		tile_size = 0

	frames = [
		raw_frames.stdout[i * frame_size:(i + 1) * frame_size]
		for i in range(actual_frames)
	]
	still_frame_index = min(still_frame_index, actual_frames - 1)
	offsets: list[int] = []
	packets = bytearray()
	previous: bytes | None = None
	tiles_x = width // tile_size if tile_size else 0
	tiles_y = height // tile_size if tile_size else 0

	def convert_frame(frame: bytes) -> bytes:
		return _bgra_to_rgb565le(frame) if rgb565 else frame

	def tile_bytes(frame: bytes, tile_x: int, tile_y: int) -> bytes:
		tile = bytearray()
		for row in range(tile_size):
			start = ((tile_y * tile_size + row) * width + tile_x * tile_size) * 2
			tile.extend(frame[start:start + tile_size * 2])
		return bytes(tile)

	for frame_index, source_frame in enumerate(frames):
		current = convert_frame(source_frame)
		offsets.append(len(packets))
		if frame_index == 0 or not tile_size:
			packets.append(GCVID_KEYFRAME)
			packets.extend(current)
		else:
			changed: list[tuple[int, bytes]] = []
			for tile_y in range(tiles_y):
				for tile_x in range(tiles_x):
					tile = tile_bytes(current, tile_x, tile_y)
					if tile != tile_bytes(previous, tile_x, tile_y):
						changed.append((tile_y * tiles_x + tile_x, tile))
			packets.append(GCVID_DELTAFRAME)
			packets.extend(struct.pack("<H", len(changed)))
			for tile_index, tile in changed:
				packets.extend(struct.pack("<H", tile_index))
				packets.extend(tile)
		previous = current

	header = bytearray()
	header.extend(GCVID_MAGIC)
	header.extend(struct.pack("<I", width))
	header.extend(struct.pack("<I", height))
	header.extend(struct.pack("<I", fps_num))
	header.extend(struct.pack("<I", fps_den))
	header.extend(struct.pack("<I", actual_frames))
	flags = tile_size
	if not rgb565:
		flags |= GCVID_FLAG_BGRA32
	header.extend(struct.pack("<I", flags))
	for offset in offsets:
		header.extend(struct.pack("<I", offset))

	output_movie.parent.mkdir(parents=True, exist_ok=True)
	output_movie.write_bytes(bytes(header) + bytes(packets))
	log(
		f"Built delta GCVID {output_movie.name} "
		f"({width}x{height}, duration {actual_frames} frames, "
		f"{'rgb565' if rgb565 else 'bgra32'})"
	)
	return {"frames": actual_frames, "bytes": len(header) + len(packets), "width": width, "height": height}


def build_gcpcm_companion(
	source_movie: Path,
	output_audio: Path,
	*,
	ffmpeg: str | None = None,
	log: Callable[[str], None] = print,
) -> bool:
	"""Convert movie audio to the GameCube mixer format: 48 kHz stereo s16le.

	PCM .wav files (the GoldSrc/Source/Quake sound format) convert without
	ffmpeg; everything else needs it.
	"""
	ffmpeg = ffmpeg or find_ffmpeg()
	if ffmpeg is None:
		pcm16 = wav_to_pcm48k(source_movie) if Path(source_movie).suffix.lower() == ".wav" else None
		if pcm16 is None:
			raise FileNotFoundError("ffmpeg is required to build .gcpcm intro companions")
		return _write_gcpcm(pcm16, output_audio, log)
	raw_audio = subprocess.run(
		[
			ffmpeg, "-v", "error", "-i", str(source_movie), "-map", "0:a:0",
			"-ar", str(GCPCM_RATE), "-ac", str(GCPCM_CHANNELS),
			"-f", "s16le", "-acodec", "pcm_s16le", "-",
		], capture_output=True,
	)
	if raw_audio.returncode != 0 and b"matches no streams" in raw_audio.stderr:
		log(f"Skipped native GCPCM {output_audio.name}: source has no audio stream")
		return False
	if raw_audio.returncode != 0:
		raise subprocess.CalledProcessError(raw_audio.returncode, raw_audio.args, raw_audio.stdout, raw_audio.stderr)
	if not raw_audio.stdout or len(raw_audio.stdout) % (GCPCM_CHANNELS * GCPCM_WIDTH):
		raise ValueError(f"ffmpeg produced invalid PCM audio for {source_movie}")
	return _write_gcpcm(raw_audio.stdout, output_audio, log)


def _write_gcpcm(pcm_s16le: bytes, output_audio: Path, log: Callable[[str], None]) -> bool:
	# The runtime passes this buffer directly to the native-endian mixer and
	# ASND's VOICE_STEREO_16BIT_BE path. Keep the metadata little-endian, but
	# store each signed-16 PCM sample in GameCube big-endian byte order.
	pcm = np.frombuffer(pcm_s16le, "<i2").astype(">i2").tobytes()
	header = GCPCM_MAGIC + struct.pack(
		"<IHHI", GCPCM_RATE, GCPCM_CHANNELS, GCPCM_WIDTH, len(pcm)
	)
	output_audio.parent.mkdir(parents=True, exist_ok=True)
	output_audio.write_bytes(header + pcm)
	log(f"Built native GCPCM {output_audio.name} ({len(pcm)} PCM bytes, 48 kHz stereo s16be/GameCube-native)")
	return True



def wav_to_pcm48k(path: Path) -> bytes | None:
	"""Decode a PCM WAV (8/16/24/32-bit, mono or stereo) to 48 kHz stereo s16le, or None.

	Resampling is linear, which is plenty for 11/22 kHz game sounds.
	"""
	import wave

	try:
		with wave.open(str(path), "rb") as w:
			ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
			raw = w.readframes(n)
	except (wave.Error, EOFError, OSError):
		return None	# compressed (ADPCM) or broken: leave it to ffmpeg
	if ch < 1 or rate <= 0:
		return None
	if width == 1:
		x = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128) / 128
	elif width == 2:
		x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
	elif width == 3:
		b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
		v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
		x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64) / (1 << 23)
	elif width == 4:
		x = np.frombuffer(raw, "<i4").astype(np.float64) / 2 ** 31
	else:
		return None
	x = x[: len(x) // ch * ch].reshape(-1, ch)
	stereo = np.repeat(x[:, :1], 2, 1) if ch == 1 else x[:, :2]
	if not len(stereo):
		return None
	out_n = max(1, int(round(len(stereo) * GCPCM_RATE / rate)))
	t = np.arange(out_n) * (rate / GCPCM_RATE)
	src = np.arange(len(stereo))
	res = np.stack([np.interp(t, src, stereo[:, c]) for c in range(2)], 1)
	return (res.clip(-1, 32767 / 32768) * 32768).round().astype("<i2").tobytes()


def video_still(
	source: Path,
	*,
	width: int | None = None,
	height: int | None = None,
	seconds: float = 1.0,
	ffmpeg: str | None = None,
) -> "np.ndarray | None":
	"""One RGBA frame for preview (scaled when width/height are given)."""
	ffmpeg = ffmpeg or find_ffmpeg()
	if ffmpeg is None:
		return None
	probe = ["-vf", f"scale={width}:{height}"] if width and height else []
	result = subprocess.run(
		[ffmpeg, "-v", "error", "-ss", str(seconds), "-i", str(source), "-frames:v", "1", *probe,
			"-f", "image2pipe", "-vcodec", "png", "-"],
		capture_output=True,
	)
	if result.returncode != 0 or not result.stdout:
		if seconds > 0:
			return video_still(source, width=width, height=height, seconds=0, ffmpeg=ffmpeg)
		return None
	import io

	from PIL import Image

	with Image.open(io.BytesIO(result.stdout)) as img:
		return np.asarray(img.convert("RGBA")).copy()


def rgb565_quantise(rgba: "np.ndarray") -> "np.ndarray":
	"""What an RGB565 GCVID frame looks like after the GameCube expands it."""
	out = rgba.copy()
	r = (rgba[..., 0] >> 3).astype(np.uint16)
	g = (rgba[..., 1] >> 2).astype(np.uint16)
	b = (rgba[..., 2] >> 3).astype(np.uint16)
	out[..., 0] = (r << 3) | (r >> 2)
	out[..., 1] = (g << 2) | (g >> 4)
	out[..., 2] = (b << 3) | (b >> 2)
	return out


def read_gcvid_header(data: bytes) -> dict:
	if data[:4] != GCVID_MAGIC:
		raise ValueError("not a GCVID stream")
	w, h, fn, fd, frames, flags = struct.unpack_from("<6I", data, 4)
	return {"width": w, "height": h, "fps": fn / max(fd, 1), "frames": frames,
		"tile": flags & 0xFFFF, "format": "bgra32" if flags & GCVID_FLAG_BGRA32 else "rgb565"}


def gcvid_first_frame(data: bytes) -> "np.ndarray":
	"""Decode the keyframe (frame 0) of a GCVID stream to RGBA."""
	info = read_gcvid_header(data)
	w, h, frames = info["width"], info["height"], info["frames"]
	start = 28 + frames * 4
	off = struct.unpack_from("<I", data, 28)[0]
	pos = start + off + 1
	if info["format"] == "rgb565":
		v = np.frombuffer(data, "<u2", w * h, pos).reshape(h, w).astype(np.int32)
		r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
		rgba = np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2), np.full_like(r, 255)), -1)
		return rgba.astype(np.uint8)
	px = np.frombuffer(data, np.uint8, w * h * 4, pos).reshape(h, w, 4)
	return np.stack((px[..., 2], px[..., 1], px[..., 0], np.full(px.shape[:2], 255, np.uint8)), -1)


def read_gcpcm_header(data: bytes) -> dict:
	if data[:4] != GCPCM_MAGIC:
		raise ValueError("not a GCPCM stream")
	rate, ch, width, size = struct.unpack_from("<IHHI", data, 4)
	return {"rate": rate, "channels": ch, "width": width, "bytes": size,
		"seconds": size / max(1, rate * ch * width)}
