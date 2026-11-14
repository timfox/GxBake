import importlib.util
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gxbake.core import cmpr, gct, goldsrc, images, media, sprite, studio
from gxbake.core.project import Kind, Project, Settings

from . import fixtures as fx


def test_cmpr_solid_roundtrip():
	rgb = np.zeros((16, 16, 3), np.uint8)
	rgb[:8, :8] = (255, 0, 0)
	rgb[8:, 8:] = (0, 255, 0)
	rgb[:8, 8:] = (40, 80, 120)
	data = cmpr.gx_cmpr_encode_blocks(rgb, np.ones((16, 16), bool))
	assert len(data) == cmpr.cmpr_size(16, 16) == 128
	out = cmpr.cmpr_decode(data, 16, 16)
	assert (out[..., 3] == 255).all()
	assert tuple(out[0, 0, :3]) == (255, 0, 0)
	assert tuple(out[15, 15, :3]) == (0, 255, 0)
	assert np.abs(out[0, 15, :3].astype(int) - (40, 80, 120)).max() <= 8


def test_cmpr_gradient_quality_and_alpha():
	y, x = np.mgrid[0:64, 0:64]
	rgb = np.stack((x * 4, y * 4, (x + y) * 2), -1).astype(np.uint8)
	opaque = np.ones((64, 64), bool)
	opaque[:8, :8] = False
	out = cmpr.cmpr_decode(cmpr.gx_cmpr_encode_blocks(rgb, opaque), 64, 64)
	assert (out[:8, :8, 3] == 0).all()
	assert (out[8:, 8:, 3] == 255).all()
	src = np.dstack((rgb, np.full((64, 64), 255, np.uint8)))
	assert images.psnr(src[8:, 8:], out[8:, 8:]) > 30


def test_sky_encoder_decodes():
	rgb = np.full((128, 128, 3), (10, 120, 250), np.uint8)
	out = cmpr.cmpr_decode(cmpr.gx_cmpr_encode(rgb), 128, 128)
	assert np.abs(out[..., :3].astype(int) - (10, 120, 250)).max() <= 8


UPSTREAM = os.environ.get("GXBAKE_UPSTREAM_SCRIPT")


@pytest.mark.skipif(not UPSTREAM, reason="set GXBAKE_UPSTREAM_SCRIPT to scripts/build-gamecube-disc.py")
def test_byte_identical_to_disc_builder():
	spec = importlib.util.spec_from_file_location("upstream_disc", UPSTREAM)
	up = importlib.util.module_from_spec(spec)
	import sys

	sys.modules[spec.name] = up
	spec.loader.exec_module(up)
	rng = np.random.default_rng(1)
	rgb = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
	opaque = rng.random((32, 64)) > 0.2
	assert up.gx_cmpr_encode(rgb) == cmpr.gx_cmpr_encode(rgb)
	assert up.gx_cmpr_encode_blocks(rgb, opaque) == cmpr.gx_cmpr_encode_blocks(rgb, opaque)
	mip = goldsrc.wt_read_miptex(fx.miptex("{fence", 48, 24, seed=4))
	_, w, h, pix, pal = mip
	assert up.wt_encode(w, h, pix, pal, True) == gct.wt_encode(w, h, pix, pal, True)


def test_gct_roundtrip():
	mip = goldsrc.wt_read_miptex(fx.miptex("brick", 64, 32))
	_, w, h, pix, pal = mip
	key, tw, th, levels, body = gct.wt_encode_cached(w, h, pix, pal, False, None)
	assert (tw, th) == (64, 32) and levels == 3	# 64x32, 32x16, 16x8
	assert len(body) == cmpr.cmpr_mip_chain_size(64, 32, 3)
	e = gct.GctEntry("brick", tw, th, levels, 0, int(key[:4], 16), body)
	packed = gct.write_gct([e, e])
	assert packed[:4] == b"GCWT"
	back = gct.read_gct(packed)
	assert [x.name for x in back] == ["brick", "brick"]
	assert back[1].data == body
	assert back[0].decode(0).shape == (32, 64, 4)
	assert back[0].decode(2).shape == (8, 16, 4)


def test_gct_cache_is_reused(tmp_path):
	mip = goldsrc.wt_read_miptex(fx.miptex("brick", 16, 16))
	_, w, h, pix, pal = mip
	first = gct.wt_encode_cached(w, h, pix, pal, False, tmp_path)
	assert len(list(tmp_path.glob("*.bin"))) == 1
	assert gct.wt_encode_cached(w, h, pix, pal, False, tmp_path) == first


def make_gamedir(tmp_path: Path) -> Path:
	game = tmp_path / "mod"
	(game / "maps").mkdir(parents=True)
	(game / "gfx/env").mkdir(parents=True)
	(game / "test.wad").write_bytes(fx.wad3({"ext_wall": fx.miptex("ext_wall", 32, 32, 2)}))
	ents = '{\n"classname" "worldspawn"\n"wad" "\\half-life\\mod\\test.wad;nothere.wad"\n"skyname" "blue"\n}\n'
	(game / "maps/demo.bsp").write_bytes(fx.bsp30(ents, [
		fx.miptex("floor", 16, 16, 1),
		fx.miptex("ext_wall", 32, 32, embed=False),
		fx.miptex("ghost", 16, 16, embed=False),
		fx.miptex("!water", 16, 16, 5),
		fx.miptex("sky", 16, 16, 6),
	]))
	from PIL import Image

	for sfx in images.GC_SKY_SUFFIXES:
		Image.new("RGB", (64, 64), (0, 0, 200)).save(game / f"gfx/env/blue{sfx}.tga")
	return game


def test_world_texture_pack_resolves_wads(tmp_path):
	game = make_gamedir(tmp_path)
	data, report, entries = gct.build_world_texture_pack(game / "maps/demo.bsp")
	assert sorted(e.name for e in entries) == ["ext_wall", "floor"]
	assert report.missing == ["ghost"]
	assert goldsrc.map_sky_name((game / "maps/demo.bsp").read_bytes()) == "blue"
	assert len(gct.read_gct(data)) == 2


def test_studio_lean_and_companion(tmp_path):
	p = tmp_path / "gun.mdl"
	p.write_bytes(fx.studio([("skin.bmp", 0x40, 256, 128), ("b.bmp", 0, 64, 64)], mesh_bytes=1000))
	lean = studio.lean_studio_mesh_bytes(p)
	assert lean is not None and len(lean) < p.stat().st_size
	assert struct.unpack_from("<i", lean, 180)[0] == 0
	comp = studio.lean_studio_tex_companion_bytes(p, 64)
	skins = studio.studio_textures(comp)
	assert [s[2].shape[:2] for s in skins] == [(32, 64), (64, 64)]
	assert skins[0][1] == 0x40


def test_sprite_lean_keeps_thin_strokes(tmp_path):
	p = tmp_path / "cross.spr"
	p.write_bytes(fx.sprite(32, 32, 3))
	lean = sprite.lean_hl_spr_bytes(p, 2)
	frame = sprite.spr_frames(lean)[0]
	assert frame.shape == (16, 16, 4)
	assert (frame[8, :, 3] == 255).all()	# stroke survived the 2x downsample


def test_project_convert_end_to_end(tmp_path):
	from PIL import Image

	game = make_gamedir(tmp_path)
	(game / "models").mkdir()
	(game / "models/gun.mdl").write_bytes(fx.studio([("skin.bmp", 0, 128, 128)]))
	(game / "sprites").mkdir()
	(game / "sprites/cross.spr").write_bytes(fx.sprite())
	rgba = np.zeros((100, 60, 4), np.uint8)
	rgba[..., 0] = 200
	rgba[:50, :, 3] = 255
	Image.fromarray(rgba).save(game / "logo.png")

	proj = Project(Settings(cache_dir=str(tmp_path / "cache")))
	proj.add_path(game)
	kinds = sorted(a.kind.value for a in proj.assets)
	assert kinds == sorted(k.value for k in (Kind.WAD, Kind.BSP, Kind.SKYBOX, Kind.MDL, Kind.SPR, Kind.IMAGE))
	for a in proj.assets:
		pv = proj.preview(a)
		assert pv.title
	out = tmp_path / "out"
	logs = []
	written = proj.convert(proj.assets, out, log=logs.append)
	rels = {w.rel for w in written}
	assert {"media/wt_demo.gct", "media/test.gct", "gc_studio/gun.mdl", "gc_studio/gunt.mdl",
		"sprites/cross.spr", "media/textures.gct", "media/sky_bluert.cmp"} <= rels
	assert all((out / r).is_file() for r in rels)
	tex = gct.read_gct((out / "media/textures.gct").read_bytes())
	assert tex[0].alpha and (tex[0].width, tex[0].height) == (64, 128)
	manifest = json.loads((out / "gxbake-manifest.json").read_text())
	assert manifest["total_bytes"] == sum(w.size for w in written)
	assert any("unresolved" in w for a in proj.assets for w in a.warnings)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_gcvid_and_gcpcm(tmp_path):
	src = tmp_path / "intro.avi"
	subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=1",
		"-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", "-c:v", "mjpeg", str(src)], check=True)
	info = media.build_gcvid_companion(src, tmp_path / "intro.gcvid", width=64, height=48, rgb565=True, log=lambda m: None)
	data = (tmp_path / "intro.gcvid").read_bytes()
	hdr = media.read_gcvid_header(data)
	assert hdr["format"] == "rgb565" and hdr["frames"] == info["frames"] >= 14
	assert media.gcvid_first_frame(data).shape == (48, 64, 4)
	assert media.build_gcpcm_companion(src, tmp_path / "intro.gcpcm", log=lambda m: None)
	pcm = media.read_gcpcm_header((tmp_path / "intro.gcpcm").read_bytes())
	assert pcm["rate"] == 48000 and 0.9 < pcm["seconds"] < 1.1
