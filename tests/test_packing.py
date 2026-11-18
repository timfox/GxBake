"""GX texture formats, DSP ADPCM, Yaz0, optimal animation runs and squeeze."""

import struct

import numpy as np
import pytest

from gxbake import cli
from gxbake.core import dspadpcm, gct, gxtex, models, sfx, skeleton, squeeze, studio, studio_slim, yaz0
from tests import fixtures
from tests.test_slim import animated_model


# --- GX texture formats ---------------------------------------------------------

def rgba_noise(h, w, seed=0, grey=False, alpha=False):
	rng = np.random.default_rng(seed)
	img = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
	if grey:
		img[..., 1] = img[..., 2] = img[..., 0]
	img[..., 3] = np.where(rng.random((h, w)) < 0.3, 0, 255) if alpha else 255
	return img


@pytest.mark.parametrize("fmt", ["I4", "I8", "IA4", "IA8", "RGB565", "RGB5A3", "RGBA8", "CMPR"])
def test_direct_formats_size_and_roundtrip(fmt):
	img = rgba_noise(12, 20, grey=fmt[0] == "I", alpha=True)
	t = gxtex.encode(img, fmt)
	assert len(t.data) == gxtex.tex_size(20, 12, fmt)
	d = t.decode()
	assert d.shape == (12, 20, 4)
	if fmt == "RGBA8":
		assert (d == img).all()
	if fmt in ("I8", "IA8"):
		assert np.abs(d[..., 0].astype(int) - img[..., 0]).max() <= 1
	if fmt == "RGB565":	# a 565 value survives a second trip exactly
		assert (gxtex.encode(d, fmt).data == t.data)


def test_i8_tile_layout():
	"""I8 tiles are 8×4 texels, row-major inside the tile, tiles row-major."""
	img = np.zeros((4, 16, 4), np.uint8)
	img[..., :3] = (np.arange(64).reshape(4, 16))[..., None]
	data = gxtex.encode(img, "I8").data
	assert list(data[:8]) == list(range(8))	# tile 0, row 0
	assert list(data[8:16]) == list(range(16, 24))	# tile 0, row 1
	assert list(data[32:40]) == list(range(8, 16))	# tile 1, row 0


def test_rgb5a3_both_modes():
	px = np.array([[[255, 0, 0, 255], [0, 255, 0, 96]]], np.uint8)
	w = gxtex.rgb5a3(px)
	assert w[0, 0] & 0x8000 and not w[0, 1] & 0x8000
	back = gxtex.from_rgb5a3(w)
	assert tuple(back[0, 0]) == (255, 0, 0, 255) and abs(int(back[0, 1, 3]) - 96) < 20


def test_indexed_is_lossless_and_half_of_rgb565():
	rng = np.random.default_rng(3)
	pal = rng.integers(0, 256, (256, 3), dtype=np.uint8)
	idx = rng.integers(0, 200, (64, 64)).astype(np.uint8)
	c = gxtex.choose_indexed(idx, pal, masked=False)
	assert c.tex.fmt == "CI8" and c.psnr == float("inf")
	assert c.baseline == 64 * 64 * 2 and c.tex.size == 64 * 64 + 208 * 2	# 200 colours → TLUT padded to 208
	ref = gxtex.encode(np.concatenate((pal[idx], np.full((64, 64, 1), 255, np.uint8)), -1), "RGB565").decode()
	assert (c.tex.decode() == ref).all()


def test_indexed_ci4_and_masked_tlut():
	pal = np.arange(768, dtype=np.uint8).reshape(256, 3)
	idx = np.tile(np.array([0, 3, 7, 255], np.uint8), (16, 4))
	c = gxtex.choose_indexed(idx, pal, masked=True, allow_cmpr=False)
	assert c.tex.fmt == "CI4" and c.tex.tlut_fmt == gxtex.GX_TL_RGB5A3
	d = c.tex.decode()
	assert (d[idx == 255, 3] == 0).all() and (d[idx != 255, 3] == 255).all()


def test_grey_skin_can_go_i4():
	pal = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1)
	idx = np.tile((np.arange(32) * 8).astype(np.uint8), (32, 1))	# a smooth grey ramp
	c = gxtex.choose_indexed(idx, pal, masked=False, min_db=20)
	assert c.tex.fmt in ("I4", "CMPR") and c.tex.size <= 32 * 32 // 2


# --- DSP ADPCM ------------------------------------------------------------------

def test_adpcm_tone_quality_and_size():
	r = 22050
	t = np.arange(r) / r
	x = (12000 * np.sin(2 * np.pi * 330 * t) + 3000 * np.sin(2 * np.pi * 1210 * t)).astype(np.int16)
	st = dspadpcm.encode(x, r)
	assert len(st.data) == dspadpcm.adpcm_bytes(r) and len(st.data) < len(x) * 2 * 0.3
	assert dspadpcm.snr_db(x, st.decode()) > 40


def test_adpcm_decoder_matches_reference_formula():
	"""One hand-built frame: predictor 1 = (2048, 0) repeats the last sample plus nibble << exp."""
	coefs = np.zeros((8, 2), np.int16)
	coefs[1] = (2048, 0)
	frame = bytes([0x12]) + bytes([0x1F, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00])	# pred 1, exp 2
	out = dspadpcm.decode_frames(frame, coefs, 4)
	assert list(out) == [4, 0, 28, 28]	# +1<<2, -1<<2, +7<<2, +0


def test_dsp_file_roundtrip_with_loop():
	x = (8000 * np.sin(np.arange(5000) / 9)).astype(np.int16)
	st = dspadpcm.encode(x, 32000, loop_start=1400, loop_end=4999)
	blob = st.to_dsp()
	assert len(blob) == 96 + dspadpcm.adpcm_bytes(5000)
	samples, nibbles, rate, loop = struct.unpack_from(">IIIH", blob)
	assert (samples, nibbles, rate, loop) == (5000, dspadpcm.frame_count(5000) * 16, 32000, 1)
	back = dspadpcm.read_dsp(blob)
	assert back.loop_start == 1400 and back.loop_end == 4999
	assert back.loop_ctx[0] == blob[96 + (1400 // 14) * 8]	# the loop frame's own header
	assert (back.decode() == st.decode()).all()


# --- Yaz0 -------------------------------------------------------------------------

@pytest.mark.parametrize("data", [b"", b"x", b"abc" * 3000, bytes(range(256)) * 40, bytes(70000)])
def test_yaz0_roundtrip(data):
	c = yaz0.compress(data)
	assert c[:4] == b"Yaz0" and struct.unpack_from(">I", c, 4)[0] == len(data)
	assert yaz0.decompress(c) == data


def test_yaz0_compresses_repeats():
	assert len(yaz0.compress(b"GXBAKE" * 4000)) < 400


# --- studio animation ---------------------------------------------------------------

def test_optimal_runs_beat_one_chunk():
	assert len(skeleton.rle_encode([1, 1, 1, 1, 5, 5, 5, 5])) == 4
	q = [0] * 3 + list(range(10)) + [9] * 20 + [3] * 40
	stream = skeleton.rle_encode(q)
	raw = struct.pack(f"<{len(stream)}h", *stream)
	assert studio_slim._expand(raw, 0, len(q)) == q
	for f in (0, 5, 30, 72):
		assert skeleton.rle_decode(np.array(stream, np.int16), f) == q[f]


def test_deadband_bounds_error():
	q = [100, 101, 99, 100, 140, 141, 139, 0, 1]
	d = studio_slim.deadband(q, 2)
	assert max(abs(a - b) for a, b in zip(q, d)) <= 2
	assert len(set(d)) < len(set(q))
	assert studio_slim.deadband([1, -2, 0, 2], 2) == [0, 0, 0, 0]


def test_slim_anim_tolerance_shrinks_and_keeps_frames():
	raw = animated_model()
	res = studio_slim.slim_studio(raw, tolerance=3000)
	assert res.after < len(raw) and not res.changed	# frame counts unchanged
	before = {s.name: s.frames for s in studio_slim.sequences(raw)}
	assert {s.name: s.frames for s in studio_slim.sequences(res.data)} == before
	assert studio.studio_info(res.data)["sequences"] == 3


# --- squeeze ---------------------------------------------------------------------

def flat_and_busy_pack() -> bytes:
	flat = np.full((64, 64, 3), (90, 120, 30), np.uint8)
	busy = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
	entries = []
	for name, rgb in (("flat", flat), ("busy", busy)):
		levels, body = gct.encode_mip_chain(rgb, np.ones((64, 64), bool))
		entries.append(gct.GctEntry(name, 64, 64, levels, 0, 1, body))
	return gct.write_gct(entries)


def test_detail_trim_drops_only_flat_mips():
	res = squeeze.trim_flat_mips(gct.read_gct(flat_and_busy_pack()), 40)
	sizes = {e.name: (e.width, e.height) for e in res.entries}
	assert sizes == {"flat": (8, 8), "busy": (64, 64)}
	assert set(res.trimmed) == {"flat"}


def test_skin_pack_roundtrip():
	raw = fixtures.studio([("skin", 32, 32, 0), ("mask", 16, 16, 0x40)])
	skins = squeeze.studio_skins(raw)
	assert [s.choice.tex.fmt for s in skins] and all(s.choice.tex.size < s.choice.baseline for s in skins)
	blob = squeeze.write_skin_pack(skins)
	back = squeeze.read_skin_pack(blob)
	for s, t in zip(skins, back):
		assert t.fmt == s.choice.tex.fmt and (t.decode() == s.choice.tex.decode()).all()
	assert all(len(t.tlut) % 32 == 0 for t in back)


def test_port_chromakey_rules():
	pal = np.array([[230, 20, 230], [10, 10, 230], [120, 120, 120], [5, 5, 5]], np.uint8)
	assert list(squeeze.port_chromakey(pal)) == [True, True, False, False]
	assert squeeze.port_chromakey(pal, hands=True)[3]


def test_squeeze_report_and_apply(tmp_path, capsys):
	root = tmp_path / "stage"
	(root / "media").mkdir(parents=True)
	(root / "gc_studio").mkdir()
	(root / "sound").mkdir()
	(root / "media/wt_test.gct").write_bytes(flat_and_busy_pack())
	(root / "gc_studio/guy.mdl").write_bytes(animated_model())
	snd = sfx.Sound(11025, (0.5 * np.sin(np.arange(4000) / 7)).astype(np.float32)[:, None], 2)
	(root / "sound/beep.wav").write_bytes(sfx.write_wav(snd))
	assert cli.main(["squeeze", str(root), "--no-yaz0"]) == 0
	text = capsys.readouterr().out
	for tech in ("detail mip trim", "optimal anim runs", "indexed skins", "DSP ADPCM sound", "s16 vertices"):
		assert tech in text
	out = tmp_path / "out"
	assert cli.main(["squeeze", str(root), "-o", str(out), "--emit", "--no-yaz0", "--anim-tol", "3"]) == 0
	assert (out / "media/wt_test.gct").stat().st_size < (root / "media/wt_test.gct").stat().st_size
	assert dspadpcm.read_dsp((out / "sound/beep.dsp").read_bytes()).samples == 4000
	assert squeeze.read_skin_pack((out / "gc_studio/guy.gsk").read_bytes())


def test_adpcm_and_yaz0_cli(tmp_path):
	wav = tmp_path / "tone.wav"
	wav.write_bytes(sfx.write_wav(sfx.Sound(22050, (0.3 * np.sin(np.arange(3000) / 5)).astype(np.float32)[:, None], 2)))
	assert cli.main(["adpcm", str(wav), "-o", str(tmp_path / "dsp")]) == 0
	assert (tmp_path / "dsp/tone.dsp").is_file()
	assert cli.main(["yaz0", str(wav)]) == 0
	assert cli.main(["yaz0", "-d", str(tmp_path / "tone.wav.yaz0"), "-o", str(tmp_path / "back")]) == 0
	assert (tmp_path / "back/tone.wav").read_bytes() == wav.read_bytes()


def test_squeeze_techniques_are_optional(tmp_path, capsys):
	root = tmp_path / "stage"
	(root / "media").mkdir(parents=True)
	(root / "gc_studio").mkdir()
	(root / "media/wt_test.gct").write_bytes(flat_and_busy_pack())
	(root / "gc_studio/guy.mdl").write_bytes(animated_model())
	assert cli.main(["squeeze", str(root), "--only", "skins,trim"]) == 0
	text = capsys.readouterr().out
	assert "indexed skins" in text and "detail mip trim" in text
	assert "optimal anim runs" not in text and "Yaz0" not in text and "s16 vertices" not in text
	out = tmp_path / "out"
	assert cli.main(["squeeze", str(root), "-o", str(out), "--emit", "--skip", "trim,yaz0,skins"]) == 0
	assert not (out / "media/wt_test.gct").exists() and not (out / "gc_studio/guy.gsk").exists()
	assert not list(out.rglob("*.yaz0"))
	with pytest.raises(SystemExit):
		cli.main(["squeeze", str(root), "--only", "bogus"])


def test_squeeze_presets(tmp_path, capsys):
	root = tmp_path / "stage"
	(root / "media").mkdir(parents=True)
	(root / "gc_studio").mkdir()
	(root / "media/wt_test.gct").write_bytes(flat_and_busy_pack())
	(root / "gc_studio/guy.mdl").write_bytes(animated_model())
	args = cli.build_parser().parse_args(["squeeze", str(root), "--preset", "max", "--anim-fit", "1"])
	opt = cli._squeeze_opts(args)
	assert (opt.min_db, opt.skin_db, opt.anim_tol, opt.anim_fit) == (34.0, 34.0, 2, 1.0)
	assert cli.main(["squeeze", str(root), "--no-yaz0"]) == 0	# balanced: keyframe fit at 0.5°
	assert "anim keyframe fit" in capsys.readouterr().out
	out = tmp_path / "lossless"
	assert cli.main(["squeeze", str(root), "-o", str(out), "--preset", "lossless", "--no-yaz0"]) == 0
	text = capsys.readouterr().out
	assert "anim keyframe fit" not in text and "anim deadband" not in text
	# Only bit-exact trims survive (the flat texture's top mip is exactly its next level magnified).
	trimmed = squeeze.trim_flat_mips(gct.read_gct((root / "media/wt_test.gct").read_bytes()), float("inf"))
	assert all(t[4] == float("inf") for t in trimmed.trimmed.values())
	m = out / "gc_studio/guy.mdl"
	if m.exists():	# lossless re-encode keeps every frame
		assert [s["frames"] for s in models.studio_sequences(m.read_bytes())] == [31, 20, 1]


def test_palette_collapses_in_tlut_domain():
	# 32 palette entries that differ only in bits RGB565 drops: 4 colours on the GameCube.
	pal = np.zeros((256, 3), np.uint8)
	for k in range(32):
		pal[k] = (np.array([0, 64, 128, 192])[k % 4] + k // 4) % 256
	index = (np.arange(64 * 64) % 32).reshape(64, 64).astype(np.uint8)
	ch = gxtex.choose_indexed(index, pal, False, allow_cmpr=False)
	assert ch.tex.fmt == "CI4" and ch.psnr == float("inf")


def test_kmeans_ci4_only_within_floor():
	rng = np.random.default_rng(3)
	pal = rng.integers(0, 256, (256, 3)).astype(np.uint8)
	# Six well-separated hues with small jitter: 60 colours that 16 entries hold nearly exactly.
	base = rng.integers(0, 256, (6, 3))
	pal[:60] = np.clip(base[np.arange(60) % 6] + rng.integers(-3, 4, (60, 3)), 0, 255)
	index = rng.integers(0, 60, (64, 64)).astype(np.uint8)
	ch = gxtex.choose_indexed(index, pal, False, min_db=35, allow_cmpr=False)
	assert ch.tex.fmt == "CI4" and 35 <= ch.psnr < float("inf")
	strict = gxtex.choose_indexed(index, pal, False, min_db=float("inf"), allow_cmpr=False)
	assert strict.tex.fmt == "CI8" and strict.psnr == float("inf")
	# Noise over 200 colours cannot hide in 16.
	noisy = rng.integers(0, 200, (64, 64)).astype(np.uint8)
	assert gxtex.choose_indexed(noisy, pal, False, min_db=35, allow_cmpr=False).tex.fmt == "CI8"


def test_skin_pack_shares_identical_blobs():
	pal = np.zeros((256, 3), np.uint8)
	pal[:4] = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
	a = (np.arange(32 * 32) % 4).reshape(32, 32).astype(np.uint8)
	b = a.T.copy()
	skins = [squeeze.SkinChoice(n, 32, 32, gxtex.choose_indexed(i, pal, False)) for n, i in (("a", a), ("b", b), ("c", a))]
	blob = squeeze.write_skin_pack(skins)
	single = len(squeeze.write_skin_pack(skins[:1]))
	back = squeeze.read_skin_pack(blob)
	assert [t.data for t in back] == [s.choice.tex.data for s in skins]
	assert all(t.tlut == back[0].tlut for t in back)
	# Three entries, but two texel blocks and one TLUT.
	rows = [squeeze.GCSK_ENTRY.unpack_from(blob, 16 + k * squeeze.GCSK_ENTRY.size) for k in range(3)]
	assert len({r[7] for r in rows}) == 2 and len({r[9] for r in rows}) == 1
	assert rows[0][7] == rows[2][7] and len(blob) < 3 * single


def test_dedup_finds_exact_and_near_copies(tmp_path):
	from gxbake.core import dedup

	rng = np.random.default_rng(5)
	base = rng.integers(0, 256, (32, 32, 4)).astype(np.uint8)
	base[..., 3] = 255
	soft = np.clip(base.astype(int) + 1, 0, 255).astype(np.uint8)	# one level brighter everywhere
	other = rng.integers(0, 256, (32, 32, 4)).astype(np.uint8)
	mk = lambda f, n, rgba, d: dedup.TexItem(f, n, 32, 32, 512, d, rgba)
	items = [mk("a.gct", "rock", base, "x"), mk("b.gct", "rock", base, "x"), mk("b.gct", "rock2", soft, "y"),
		mk("c.gct", "sky", other, "z")]
	rep = dedup.find_duplicates(items, min_db=40)
	assert len(rep.exact) == 1 and len(rep.exact[0].items) == 2 and rep.exact[0].wasted == 512
	assert len(rep.near) == 1 and {i.name for i in rep.near[0].items} == {"rock", "rock2"} and rep.near[0].psnr > 40
	assert "exact duplicate groups" in dedup.format_dedup(rep)
	assert dedup.dhash(base) == dedup.dhash(soft) and dedup.dhash(base) != dedup.dhash(other)


def test_squeeze_reports_duplicate_textures(tmp_path, capsys):
	root = tmp_path / "stage"
	(root / "media").mkdir(parents=True)
	pack = flat_and_busy_pack()
	(root / "media/wt_a.gct").write_bytes(pack)
	(root / "media/wt_b.gct").write_bytes(pack)
	assert cli.main(["squeeze", str(root), "--only", "dedup"]) == 0
	assert "duplicate textures" in capsys.readouterr().out


def test_cmpr_refine_same_size_closer_and_keeps_cutouts():
	from gxbake.core import cmpr

	rng = np.random.default_rng(9)
	y, x = np.mgrid[0:32, 0:32]
	rgb = np.stack((x * 8, y * 8, (x + y) * 4), -1).astype(np.float64) + rng.normal(0, 6, (32, 32, 3))
	rgb = np.clip(rgb, 0, 255).astype(np.uint8)
	op = np.ones((32, 32), bool)
	op[:8, :8] = (x[:8, :8] + y[:8, :8]) % 3 > 0	# some three-colour (cut-out) blocks
	base = cmpr.gx_cmpr_encode_blocks(rgb, op)
	ref = cmpr.cmpr_refine(rgb, op, base)
	assert len(ref) == len(base)
	a, b = cmpr.cmpr_decode(base, 32, 32), cmpr.cmpr_decode(ref, 32, 32)
	assert ((a[..., 3] > 0) == (b[..., 3] > 0)).all() and ((b[..., 3] > 0) == op).all()
	assert gxtex.psnr(b[..., :3], rgb, op) > gxtex.psnr(a[..., :3], rgb, op)
	# the block helpers round-trip the tiled layout
	c0, c1, idx = cmpr._blocks_of(base, 8, 8)
	assert cmpr._tile_blocks(c0, c1, idx) == base


def test_split_cmpr_roundtrip_and_helps_lz():
	pack = flat_and_busy_pack()
	sp = squeeze.split_cmpr(pack)
	assert len(sp) == len(pack) and sp != pack
	assert squeeze.split_cmpr(sp, inverse=True) == pack
	assert gct.read_gct(squeeze.split_cmpr(sp, inverse=True))[0].data == gct.read_gct(pack)[0].data
