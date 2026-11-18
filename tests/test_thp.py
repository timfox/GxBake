"""THP movies: the writer, the reader, FFmpeg's view of the files, and port/gc_thp.c."""

import io
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gxbake.core import dspadpcm, media, thp

FFMPEG = media.find_ffmpeg()
FFPROBE = thp.find_ffprobe()
PORT = Path(__file__).resolve().parent.parent / "port"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="no ffmpeg/ffprobe")


def _picture(w, h, seed=0, q=80):
	rng = np.random.default_rng(seed)
	yy, xx = np.mgrid[:h, :w]
	rgb = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx + yy + seed * 7) % 256)], -1).astype(np.uint8)
	rgb[rng.integers(0, h, 40), rng.integers(0, w, 40)] = 255	# some hard edges: 0xFF bytes in the scan
	return rgb, thp.encode_jpeg(rgb, q)


def _tone(n, rate=32000):
	t = np.arange(n) / rate
	left = 9000 * np.sin(2 * np.pi * 440 * t) + 2000 * np.sin(2 * np.pi * 3100 * t)
	right = 7000 * np.sin(2 * np.pi * 220 * t + 1)
	return np.stack([left, right], 1).astype(np.int16)


def _movie(tmp_path, frames=12, w=64, h=48, fps=24.0, audio=True):
	pics = [_picture(w, h, k)[1] for k in range(frames)]
	enc = None
	if audio:
		bounds = thp.audio_boundaries(frames, fps, 32000)
		enc = thp._encode_audio(_tone(bounds[-1]), frames, fps, 32000)
	blob = thp.write_thp(pics, fps, w, h, enc)
	path = tmp_path / "m.thp"
	path.write_bytes(blob)
	return path, pics, enc


def test_jpeg_pictures_are_unstuffed_baseline():
	_, jpg = _picture(64, 48)
	assert jpg[:4] == b"\xff\xd8\xff\xdb"	# no JFIF/APP header
	assert b"\xff\xc0" in jpg and b"\xff\xc2" not in jpg and b"\xff\xdd" not in jpg
	restuffed = thp.to_jpeg(jpg)
	from PIL import Image

	with Image.open(io.BytesIO(restuffed)) as img:
		assert img.size == (64, 48) and img.mode == "RGB"


def test_unstuff_restuff_round_trip():
	rgb, jpg = _picture(128, 64, 3)
	from PIL import Image

	buf = io.BytesIO()
	Image.fromarray(rgb).save(buf, "JPEG", quality=80, subsampling=2)
	stuffed = buf.getvalue()
	assert thp._strip_jpeg(stuffed) == jpg
	body = lambda b: b[b.index(b"\xff\xda"):]	# noqa: E731
	assert body(thp.to_jpeg(jpg)) == body(stuffed)


def test_container_layout(tmp_path):
	path, pics, enc = _movie(tmp_path)
	data = path.read_bytes()
	info = thp.read_info(data)
	assert (info.version, info.width, info.height, info.frames) == (thp.VERSION, 64, 48, 12)
	assert abs(info.fps - 24.0) < 1e-6
	assert (info.audio_channels, info.audio_rate, info.audio_samples) == (2, 32000, enc.bounds[-1])
	assert info.first_frame % 32 == 0
	frames = list(thp.iter_frames(data))
	assert len(frames) == 12
	pos, sizes = info.first_frame, []
	for k, (video, audio) in enumerate(frames):
		assert video.rstrip(b"\0") == pics[k]
		assert len(video) % 4 == 0
		nxt, prv = struct.unpack_from(">II", data, pos)
		sizes.append(nxt)
		assert pos % 32 == 0
		size = struct.unpack_from(">I", data, 0x18)[0] if k == 0 else sizes[k - 1]
		pos += size
	assert pos == len(data)
	assert struct.unpack_from(">I", data, 0x2C)[0] == len(data) - sizes[-2]	# last frame offset
	assert sizes[-1] == struct.unpack_from(">I", data, 0x18)[0]	# the last frame points back at frame 0
	assert info.max_frame_bytes == max(sizes)


def test_audio_blocks_decode_on_their_own(tmp_path):
	path, _, enc = _movie(tmp_path, frames=20, fps=29.97)
	data = path.read_bytes()
	blocks = [thp.decode_audio_block(a) for _, a in thp.iter_frames(data)]
	assert all(len(b) % 14 == 0 for b in blocks)
	whole = np.stack([dspadpcm.decode_frames(enc.data[c], enc.coefs[c], enc.bounds[-1]) for c in range(2)], 1)
	assert (np.concatenate(blocks) == whole).all()
	ref = _tone(enc.bounds[-1])
	assert dspadpcm.snr_db(ref[:, 0], whole[:, 0]) > 25


def test_audio_boundaries_keep_sync():
	b = thp.audio_boundaries(300, 29.97, 32000)
	assert all(x % 14 == 0 for x in b)
	assert abs(b[-1] - 300 / 29.97 * 32000) <= 7
	assert max(y - x for x, y in zip(b, b[1:])) - min(y - x for x, y in zip(b, b[1:])) <= 14


def test_no_audio_movie(tmp_path):
	path, _, _ = _movie(tmp_path, audio=False)
	data = path.read_bytes()
	info = thp.read_info(data)
	assert info.audio_channels == 0 and info.max_audio_samples == 0
	assert [a for _, a in thp.iter_frames(data)] == [b""] * 12
	assert thp.frame_rgba(data, 3).shape == (48, 64, 4)


def test_plan_size_and_fps():
	src = thp.SourceInfo(1920, 1080, 59.94, 10.0)
	assert thp.plan_size(src, thp.ThpOptions()) == (512, 288)
	assert thp.plan_size(thp.SourceInfo(640, 480, 30, 1), thp.ThpOptions()) == (512, 384)
	assert thp.plan_size(thp.SourceInfo(320, 240, 30, 1), thp.ThpOptions()) == (320, 240)	# no upscale
	assert thp.plan_size(thp.SourceInfo(720, 480, 30, 1, sar=8 / 9), thp.ThpOptions()) == (512, 384)	# DVD 4:3
	assert thp.plan_size(src, thp.ThpOptions(width=650, height=500)) == (640, 480)
	assert thp.plan_size(src, thp.ThpOptions(width=300, height=200)) == (288, 192)
	assert abs(thp.plan_fps(59.94, thp.ThpOptions()) - 29.97) < 1e-9
	assert thp.plan_fps(60, thp.ThpOptions()) == 30
	assert thp.plan_fps(50, thp.ThpOptions()) == 25
	assert thp.plan_fps(120, thp.ThpOptions()) == 30
	assert thp.plan_fps(23.976, thp.ThpOptions()) == 23.976
	assert thp.plan_fps(60, thp.ThpOptions(fps=12)) == 12


def test_quality_search_meets_cap():
	rgb, full = _picture(256, 192, 1, q=90)
	cap = len(full) // 3
	jpg, q, over = thp._fit_quality(rgb, 90, cap, 5, None)
	assert len(jpg) <= cap and not over and q < 90
	assert len(thp.encode_jpeg(rgb, q + 1)) > cap	# the highest quality that fits


@pytest.fixture(scope="module")
def source(tmp_path_factory):
	if not (FFMPEG and FFPROBE):
		pytest.skip("no ffmpeg")
	path = tmp_path_factory.mktemp("vid") / "src.mkv"
	subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=60:duration=1.5",
		"-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100:duration=1.5",
		"-c:v", "mpeg4", "-q:v", "3", "-c:a", "pcm_s16le", "-ac", "1", str(path)], check=True)
	return path


@needs_ffmpeg
def test_convert_and_ffmpeg_reads_it(tmp_path, source):
	out = tmp_path / "out.thp"
	res = thp.convert(source, out, log=lambda m: None)
	assert (res.width, res.height) == (480, 256) and 44 <= res.frames <= 46
	assert abs(res.fps - 30) < 1e-9 and res.audio_rate == 32000
	probe = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(out)], capture_output=True, text=True)
	import json

	streams = json.loads(probe.stdout)["streams"]
	assert [s["codec_name"] for s in streams] == ["thp", "adpcm_thp"]
	assert streams[1]["channels"] == 2 and int(streams[1]["sample_rate"]) == 32000
	dec = subprocess.run([FFMPEG, "-v", "error", "-i", str(out), "-map", "0:v", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
		capture_output=True)
	assert dec.returncode == 0 and not dec.stderr, dec.stderr[-400:]	# FFmpeg's THP decoder: no corrupt scans
	assert len(dec.stdout) == res.frames * res.width * res.height * 3
	mine = thp.frame_rgba(out.read_bytes(), 0)[..., :3].astype(int)
	theirs = np.frombuffer(dec.stdout, np.uint8, res.width * res.height * 3).reshape(res.height, res.width, 3).astype(int)
	assert np.abs(mine - theirs).mean() < 3	# same picture (chroma upsampling differs)


@needs_ffmpeg
def test_convert_caps(tmp_path, source):
	out = tmp_path / "cap.thp"
	res = thp.convert(source, out, thp.ThpOptions(max_bytes=120_000, box_width=320, box_height=240, audio=False),
		log=lambda m: None)
	assert out.stat().st_size <= 120_000 and res.quality_min < thp.DEFAULT_QUALITY
	res = thp.convert(source, tmp_path / "s.thp", thp.ThpOptions(width=256, height=256, fps=10, duration=1.0, audio_rate=48000),
		log=lambda m: None)
	assert (res.width, res.height, res.frames, res.audio_rate) == (256, 256, 10, 48000)


@needs_ffmpeg
def test_cli_and_project(tmp_path, source):
	from gxbake.cli import main
	from gxbake.core.project import Kind, Project, Settings

	assert main(["video", str(source), "-o", str(tmp_path / "cli"), "--max", "256x256", "--no-audio"]) == 0
	assert thp.read_info((tmp_path / "cli/src.thp").read_bytes()).width == 256
	proj = Project(Settings(video_format="both"))
	asset = proj.add_path(source)[0]
	assert asset.kind == Kind.VIDEO
	assert "THP" not in proj.preview(asset).info.get("Target", "")
	proj.settings.video_format = "thp"
	assert proj.preview(asset).info["Target"].startswith("THP 480x256")
	proj.settings.video_format = "both"
	written = {w.rel for w in proj.convert([asset], tmp_path / "staged", log=lambda m: None)}
	assert {"media/src.thp", "media/src.gcvid", "media/src.gcpcm"} <= written
	thp_asset = Project().add_path(tmp_path / "staged/media/src.thp")[0]
	assert thp_asset.kind == Kind.THP
	assert Project().info(thp_asset)["Frame size"] == "480x256"


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_reader(tmp_path):
	exe = tmp_path / "test_gc_thp"
	subprocess.run([CC, "-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", "-O2",
		str(PORT / "gc_thp.c"), str(PORT / "test_gc_thp.c"), "-o", str(exe)], check=True)
	w, h = 96, 64
	path, pics, enc = _movie(tmp_path, frames=9, w=w, h=h, fps=29.97)
	res = subprocess.run([exe, path, tmp_path / "y", tmp_path / "p", tmp_path / "t"], capture_output=True, text=True)
	assert res.returncode == 0, res
	assert res.stdout.split() == [str(w), str(h), "9", "29970", "2", "32000", str(enc.bounds[-1])]
	# audio: bit-exact with GxBake's decoder
	pcm = np.fromfile(tmp_path / "p", "<i2").reshape(-1, 2)
	blocks = np.concatenate([thp.decode_audio_block(a) for _, a in thp.iter_frames(path.read_bytes())])
	assert (pcm == blocks).all()
	# pictures: within 1 of libjpeg (Pillow) on the Y plane; the IDCT is libjpeg's
	from PIL import Image

	yuv = np.fromfile(tmp_path / "y", np.uint8).reshape(9, -1)
	for k, pic in enumerate(pics):
		with Image.open(io.BytesIO(thp.to_jpeg(pic))) as img:
			img.draft("YCbCr", img.size)
			ref = np.asarray(img.convert("YCbCr"))[..., 0].astype(int)
		ours = yuv[k, :w * h].reshape(h, w).astype(int)
		assert np.abs(ours - ref).max() <= 1
	# textures: tiled RGB565 and I8 from the planes, same maths as here
	tex = (tmp_path / "t").read_bytes()
	y0 = yuv[0, :w * h].reshape(h, w).astype(np.int64)
	cb = np.repeat(np.repeat(yuv[0, w * h:w * h * 5 // 4].reshape(h // 2, w // 2), 2, 0), 2, 1).astype(np.int64) - 128
	cr = np.repeat(np.repeat(yuv[0, w * h * 5 // 4:].reshape(h // 2, w // 2), 2, 0), 2, 1).astype(np.int64) - 128
	r = np.clip(y0 + ((91881 * cr + 32768) >> 16), 0, 255)
	g = np.clip(y0 + ((-22554 * cb - 46802 * cr + 32768) >> 16), 0, 255)
	b = np.clip(y0 + ((116130 * cb + 32768) >> 16), 0, 255)
	rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
	tiled = rgb565.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1)
	assert (np.frombuffer(tex, "<u2", w * h) == tiled).all()
	i8 = y0.reshape(h // 4, 4, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1)
	assert (np.frombuffer(tex, np.uint8, w * h, w * h * 2) == i8).all()


@pytest.mark.skipif(CC is None, reason="no C compiler")
@needs_ffmpeg
def test_c_reader_matches_ffmpeg(tmp_path, source):
	exe = tmp_path / "test_gc_thp"
	subprocess.run([CC, "-O2", str(PORT / "gc_thp.c"), str(PORT / "test_gc_thp.c"), "-o", str(exe)], check=True)
	out = tmp_path / "o.thp"
	thp.convert(source, out, thp.ThpOptions(duration=0.5), log=lambda m: None)
	subprocess.run([exe, out, tmp_path / "y", tmp_path / "p", tmp_path / "t"], check=True, capture_output=True)
	ff = subprocess.run([FFMPEG, "-v", "error", "-i", str(out), "-map", "0:v", "-f", "rawvideo", "-pix_fmt", "yuvj420p", "-"],
		capture_output=True, check=True).stdout
	ours = np.fromfile(tmp_path / "y", np.uint8).astype(int)
	theirs = np.frombuffer(ff, np.uint8).astype(int)
	assert len(ours) == len(theirs)
	assert np.abs(ours - theirs).max() <= 1 and (ours != theirs).mean() < 0.02
