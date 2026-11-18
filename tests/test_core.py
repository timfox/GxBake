import importlib.util
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gxbake.core import audit, cmpr, gct, goldsrc, images, media, sprite, studio
from gxbake.core.project import Kind, Project, Settings
from gxbake.core.project import detect_kind as detect_kind_of

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


def test_wad_pictures_and_fonts(tmp_path):
	wad = tmp_path / "gfx.wad"
	wad.write_bytes(fx.wad3({"wall": fx.miptex("wall", 32, 32, 1), "loading": fx.qpic(40, 24), "conchars": fx.font()},
		{"loading": 0x42, "conchars": 0x46}))
	pics = goldsrc.wad_pictures(wad)
	assert set(pics) == {"loading", "conchars"} and pics["loading"].transparent
	font = pics["conchars"]
	assert (font.width, font.height, font.row_height) == (256, 128, 8) and font.glyphs[17] == (8 * 256 + 16, 16)
	rgba = pics["loading"].rgba()
	assert rgba.shape == (24, 40, 4) and rgba[0, 0, 3] == 0 and rgba[5, 5, 3] == 255
	_, entries = gct.build_wad_texture_pack(wad)
	by = {e.name: e for e in entries}
	assert set(by) == {"wall", "loading", "conchars"} and by["loading"].flags & gct.GC_WT_FLAG_ALPHA
	assert not by["wall"].flags & gct.GC_WT_FLAG_ALPHA
	_, only = gct.build_wad_texture_pack(wad, include={"wall"})
	assert [e.name for e in only] == ["wall"]
	proj = Project(Settings())
	proj.add_path(wad)
	assert proj.children(proj.assets[0]) == ["conchars", "loading", "wall"]
	assert proj.info(proj.assets[0])["Pictures and fonts"] == 2
	assert proj.preview(proj.assets[0], "loading").result is not None


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


# -- Assimp model import --------------------------------------------------------

from gxbake.core import models  # noqa: E402

needs_assimp = pytest.mark.skipif(not models.available(), reason="assimp-py not installed")
needs_skeletal = pytest.mark.skipif(not models.skeletal_supported(), reason="assimp-py < 1.2 has no bone API")


def _signed_volume(raw: bytes) -> float:
	"""Volume of a closed studio mesh; > 0 when triangles are stored clockwise (outward)."""
	vol = 0.0
	for m in models.studio_meshes(raw):
		v = m["verts"].astype(np.float64)
		for a, b, c in m["tris"]:
			pa, pb, pc = v[a[0]], v[b[0]], v[c[0]]
			vol += np.dot(pa, np.cross(pc, pb)) / 6
	return vol


def test_strips_keep_every_triangle_and_winding():
	rng = np.random.default_rng(4)
	grid = np.arange(36).reshape(6, 6)
	tris = []
	for y in range(5):
		for x in range(5):
			a, b, c, d = grid[y, x], grid[y, x + 1], grid[y + 1, x + 1], grid[y + 1, x]
			tris += [(a, b, c), (a, c, d)]
	order = rng.permutation(len(tris))
	tris = [tuple(int(v) for v in tris[i]) for i in order]
	strips = models._strips(tris)
	got = []
	for s in strips:
		for k in range(len(s) - 2):
			t = (s[k], s[k + 1], s[k + 2]) if k % 2 == 0 else (s[k + 1], s[k], s[k + 2])
			got.append(t)

	def canon(t):
		i = t.index(min(t))
		return t[i:] + t[:i]	# rotation keeps winding

	assert sorted(map(canon, got)) == sorted(map(canon, tris))
	assert len(strips) < len(tris) / 2	# actually stripped, not one command per triangle


@needs_assimp
def test_obj_import_bakes_studio_model(tmp_path):
	obj = fx.obj_cube(tmp_path / "src")
	model = models.import_model(obj)
	assert sorted(m.name for m in model.materials) == ["red", "wood"]
	wood = next(m for m in model.materials if m.name == "wood")
	assert wood.rgba.shape[:2] == (64, 64)	# 40x48 found by basename despite the C:\ path, padded to 2^n
	assert model.num_triangles == 12
	lo, hi = model.bounds()
	assert np.allclose(lo, -1) and np.allclose(hi, 1)

	raw = models.build_studio_mdl(model)
	info = studio.studio_info(raw)
	assert info == {"name": "crate.mdl", "version": 10, "length": len(raw), "bones": 1, "sequences": 1,
		"textures": 2, "bodyparts": 1}
	meshes = models.studio_meshes(raw)
	assert sum(len(m["tris"]) for m in meshes) == 12
	assert _signed_volume(raw) == pytest.approx(8.0, rel=1e-4)	# closed, outward-facing, unit scale
	# texel-space s/t cover the whole skin
	st = [c[2:] for m in meshes if m["skin"] == model.materials.index(wood) for t in m["tris"] for c in t]
	assert {s for s, _ in st} == {0, 64} and {t for _, t in st} == {0, 64}
	# the source +Z face (red) ends up facing GoldSrc +X (forward)
	red = model.materials.index(next(m for m in model.materials if m.name == "red"))
	red_x = {round(float(m["verts"][c[0]][0]), 3) for m in meshes if m["skin"] == red for t in m["tris"] for c in t}
	assert red_x == {1.0}

	lean = studio.lean_studio_mesh_bytes_from(raw)
	comp = studio.lean_studio_tex_companion_bytes_from(raw, 32)
	assert struct.unpack_from("<i", lean, 180)[0] == 0 and len(lean) < len(raw)
	assert sum(len(m["tris"]) for m in models.studio_meshes(lean)) == 12	# geometry survives the cut
	assert sorted(r.shape[:2] for _n, _f, r in studio.studio_textures(comp)) == [(8, 8), (32, 32)]


@needs_assimp
def test_glb_embedded_masked_texture(tmp_path):
	p = tmp_path / "leaf.glb"
	fx.glb_quad(p)
	model = models.import_model(p, models.ModelOptions(scale=10))
	(mat,) = model.materials
	assert mat.texture == "*0" and mat.masked
	raw = models.build_studio_mdl(model)
	(_name, flags, rgba), = studio.studio_textures(raw)
	assert flags & models.STUDIO_NF_MASKED
	assert (rgba[:8, :8, 3] == 0).all() and (rgba[8:, 8:, 3] == 255).all()
	assert np.abs(rgba[12, 12, :3].astype(int) - (0, 200, 0)).max() <= 4
	lo, hi = model.bounds()
	assert np.allclose(lo, (0, -10, 0)) and np.allclose(hi, (0, 10, 20))	# Y-up quad → Z-up, scaled


@needs_assimp
def test_project_converts_assimp_model(tmp_path):
	obj = fx.obj_cube(tmp_path / "src")
	proj = Project()
	(asset,) = [a for a in proj.add_path(obj.parent) if a.kind == Kind.MODEL]
	assert sorted(proj.children(asset)) == ["red", "wood"]
	assert proj.info(asset)["Triangles"] == 12
	pv = proj.preview(asset)
	assert pv.source is not None and pv.result is not None
	assert (pv.result[..., :3] != pv.result[0, 0, :3]).any()	# something was drawn
	assert proj.preview(asset, "wood").result.shape[:2] == (64, 64)
	out = tmp_path / "out"
	proj.convert([asset], out, log=lambda _m: None)
	assert asset.status.startswith("Done"), asset.warnings
	for rel in ("models/crate.mdl", "gc_studio/crate.mdl", "gc_studio/cratet.mdl"):
		assert (out / rel).is_file(), rel
	assert [e for e in audit.validate_staged_tree(out) if "critical asset" not in e] == []


def test_rle_and_euler_roundtrip():
	from gxbake.core import skeleton as sk

	rng = np.random.default_rng(1)
	q = list(rng.integers(-500, 500, 300)) + [7] * 40 + [3]
	stream = np.array(sk.rle_encode([int(x) for x in q]), np.int16)
	assert [sk.rle_decode(stream, k) for k in range(len(q))] == q
	assert len(sk.rle_encode([5] * 200)) == 2	# one header + one value
	for _ in range(50):
		a = rng.uniform(-3, 3, 3)
		a[1] = rng.uniform(-1.5, 1.5)
		m = sk.angle_matrix(a)
		assert np.allclose(sk.angle_matrix(np.array(sk.angles_from_matrix(m))), m, atol=1e-9)


def _arm_expected(angle_deg: float) -> "np.ndarray":
	"""Rest box of fx.glb_arm with 'lower' rotated about source +Z at y=2, in GoldSrc axes."""
	pts = []
	a = np.radians(angle_deg)
	for y in (0.0, 1.0, 2.0, 3.0, 4.0):
		for x, z in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
			if y >= 2:
				x, y2 = x * np.cos(a) - (y - 2) * np.sin(a), x * np.sin(a) + (y - 2) * np.cos(a) + 2
			else:
				y2 = y
			pts.append((z, x, y2))	# Y-up → Z-up
	return np.array(pts)


def _same_points(a, b, tol=1e-3):
	a = sorted(map(tuple, np.round(a, 3)))
	b = sorted(map(tuple, np.round(b, 3)))
	return len(a) == len(b) and np.allclose(a, b, atol=tol)


@needs_skeletal
def test_skinned_animation_bakes_bones_and_sequences(tmp_path):
	p = tmp_path / "arm.glb"
	fx.glb_arm(p, 90.0)
	model = models.import_model(p, models.ModelOptions(anim_fps=10))
	assert [(b.name, b.parent) for b in model.bones][1:] == [("upper", 0), ("lower", 1)]
	(seq,) = model.sequences
	assert seq.name == "bend" and seq.frames.shape == (11, 3, 6)
	raw = models.build_studio_mdl(model)
	info = studio.studio_info(raw)
	assert info["bones"] == 3 and info["sequences"] == 1
	sq = models.studio_sequences(raw)[0]
	assert sq["frames"] == 11 and sq["flags"] & 1 and sq["fps"] == 10
	assert struct.unpack_from("<i", raw, 156)[0] == 2	# a hitbox for each bone with vertices
	assert _same_points(models.studio_posed_vertices(raw, 0, 0), _arm_expected(0))
	assert _same_points(models.studio_posed_vertices(raw, 0, 10), _arm_expected(90))
	assert _same_points(models.studio_posed_vertices(raw, 0, 5), _arm_expected(45))
	# the lean mesh keeps the skeleton and animation
	lean = studio.lean_studio_mesh_bytes_from(raw)
	assert _same_points(models.studio_posed_vertices(lean, 0, 10), _arm_expected(90))
	assert sq["bbmin"][1] < -1.9	# the sequence box covers the bent pose

	static = models.import_model(p, models.ModelOptions(static=True))
	assert len(static.bones) == 1 and not static.animated
	assert any("static" in w for w in static.warnings)


@needs_skeletal
def test_clip_files_become_sequences(tmp_path):
	fx.glb_arm(tmp_path / "arm.glb", 90.0, name="bend")
	fx.glb_arm(tmp_path / "arm@wave.glb", -45.0, name="Take 001")
	proj = Project()
	assets = proj.add_path(tmp_path)
	assert [a.path.name for a in assets] == ["arm.glb"]	# the clip is not its own asset
	model, raw = proj.imported_model(assets[0])
	assert [s.name for s in model.sequences] == ["bend", "wave"]
	last = model.sequences[1].frames.shape[0] - 1
	assert _same_points(models.studio_posed_vertices(raw, 1, last), _arm_expected(-45))
	kids = proj.children(assets[0])
	assert "▶ wave" in kids
	pv = proj.preview(assets[0], "▶ wave")
	assert pv.result is not None and pv.info["Frames"] == last + 1


@needs_assimp
def test_gltf_data_uri_texture(tmp_path):
	import base64
	import io as _io
	import json as _json

	from PIL import Image

	fx.glb_quad(tmp_path / "q.glb")
	raw = (tmp_path / "q.glb").read_bytes()
	jlen = struct.unpack_from("<I", raw, 12)[0]
	doc = _json.loads(raw[20:20 + jlen])
	binchunk = raw[20 + jlen + 8:]
	png = _io.BytesIO()
	Image.new("RGBA", (8, 8), (250, 120, 10, 255)).save(png, "PNG")
	doc["images"] = [{"uri": "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()}]
	doc["buffers"] = [{"byteLength": len(binchunk), "uri": "data:application/octet-stream;base64,"
		+ base64.b64encode(binchunk).decode()}]
	(tmp_path / "q.gltf").write_text(_json.dumps(doc))
	model = models.import_model(tmp_path / "q.gltf")
	(mat,) = model.materials
	assert tuple(mat.rgba[4, 4, :3]) == (250, 120, 10)


def test_more_image_formats_detected(tmp_path):
	from PIL import Image

	Image.new("RGB", (16, 16), (1, 2, 3)).save(tmp_path / "a.tif")
	assert detect_kind_of(tmp_path / "a.tif") == Kind.IMAGE
	assert images.load_rgba(tmp_path / "a.tif").shape == (16, 16, 4)
	for ext in (".dds", ".psd", ".md2", ".md5mesh", ".iqm"):
		assert detect_kind_of(tmp_path / f"x{ext}") in (Kind.IMAGE, Kind.MODEL)


def test_tri_commands_strips_and_fans_cover_every_triangle():
	rng = np.random.default_rng(9)
	# a grid plus a disc (fan territory) sharing nothing
	grid = np.arange(49).reshape(7, 7)
	tris = []
	for y in range(6):
		for x in range(6):
			a, b, c, d = grid[y, x], grid[y, x + 1], grid[y + 1, x + 1], grid[y + 1, x]
			tris += [(a, b, c), (a, c, d)]
	hub, rim = 100, list(range(101, 113))
	tris += [(hub, rim[i], rim[(i + 1) % 12]) for i in range(12)]
	tris = [tuple(int(v) for v in tris[i]) for i in rng.permutation(len(tris))]
	cmds = models._tri_commands(tris)
	got = []
	for fan, s in cmds:
		for k in range(len(s) - 2):
			if fan:
				got.append((s[0], s[k + 1], s[k + 2]))
			else:
				got.append((s[k], s[k + 1], s[k + 2]) if k % 2 == 0 else (s[k + 1], s[k], s[k + 2]))

	def canon(t):
		i = t.index(min(t))
		return t[i:] + t[:i]

	assert sorted(map(canon, got)) == sorted(map(canon, tris))
	assert any(fan for fan, _ in cmds)	# the disc becomes a fan
	assert sum(len(s) for _, s in cmds) < len(tris) * 1.6
