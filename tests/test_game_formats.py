"""Native Source / Quake / Quake II parsers and their GameCube outputs."""

import math
import struct

import numpy as np

from gxbake.core import archives, gct, media, models, quake, source_mdl, sprite, studio, vtf
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import fixtures as fx
from . import game_fixtures as gf


def _rgba_pattern(w=32, h=16):
	y, x = np.mgrid[0:h, 0:w]
	img = np.zeros((h, w, 4), np.uint8)
	img[..., 0] = x * 255 // (w - 1)
	img[..., 1] = y * 255 // (h - 1)
	img[..., 2] = 90
	img[..., 3] = np.where(x < 4, 0, 255)
	return img


# -- VTF -----------------------------------------------------------------------------


def test_vtf_bgra_and_versions_roundtrip():
	src = _rgba_pattern()
	for version in ((7, 2), (7, 3), (7, 5)):
		data = gf.vtf(src, "bgra", version)
		hd = vtf.vtf_header(data)
		assert hd["format_name"] == "BGRA8888" and hd["mips"] == 6
		assert (vtf.read_vtf(data) == src).all()
		assert vtf.read_vtf(data, mip=1).shape == (8, 16, 4)


def test_vtf_dxt1_decodes_block_colours():
	src = np.zeros((8, 8, 4), np.uint8)
	src[..., 3] = 255
	src[:4, :4, :3] = (248, 0, 0)
	src[4:, 4:, :3] = (0, 252, 0)
	out = vtf.read_vtf(gf.vtf(src, "dxt1"))
	assert tuple(out[0, 0]) == (255, 0, 0, 255)	# 565 expands with bit replication
	assert tuple(out[7, 7]) == (0, 255, 0, 255)
	assert tuple(out[0, 7, :3]) == (0, 0, 0)


def test_dxt5_alpha_and_bc4_interpolation():
	block = bytes([255, 0]) + bytes(6)	# alpha endpoints 255/0, every index 0 → 255
	colour = struct.pack("<HHI", 0xFFFF, 0, 0)
	out = vtf.decode_dxt5(block + colour, 4, 4)
	assert (out[..., 3] == 255).all() and (out[..., :3] == 255).all()


def test_vmt_resolution(tmp_path):
	mats = tmp_path / "materials"
	(mats / "models/props").mkdir(parents=True)
	(mats / "models/props/Bin.vtf").write_bytes(gf.vtf(_rgba_pattern(8, 8)))
	(mats / "models/props/bin_skin.vmt").write_text('VertexLitGeneric\n{\n\t$basetexture "models\\\\props\\\\Bin"\n\t$alphatest 1 // cut\n}\n')
	shader, params = vtf.parse_vmt((mats / "models/props/bin_skin.vmt").read_text())
	assert shader == "vertexlitgeneric" and params["$alphatest"] == "1"
	path, params = vtf.resolve_material("bin_skin", ["models\\props\\"], [mats])
	assert path is not None and path.name == "Bin.vtf"


def test_vtf_is_an_image_and_skybox_side(tmp_path):
	for side in ("rt", "bk", "lf", "ft", "up", "dn"):
		(tmp_path / f"sky_day{side}.vtf").write_bytes(gf.vtf(_rgba_pattern(16, 16)))
	(tmp_path / "decal.vtf").write_bytes(gf.vtf(_rgba_pattern()))
	proj = Project()
	proj.add_path(tmp_path)
	assert sorted(a.kind for a in proj.assets) == sorted([Kind.IMAGE, Kind.SKYBOX])
	out = tmp_path / "out"
	rels = {w.rel for w in proj.convert(proj.assets, out, log=lambda m: None)}
	assert "media/sky_sky_dayrt.cmp" in rels and "media/textures.gct" in rels
	assert gct.read_gct((out / "media/textures.gct").read_bytes())[0].alpha


# -- Source models -----------------------------------------------------------------------


def _signed_volume(mesh) -> float:
	p = mesh.world.astype(np.float64)
	t = mesh.tris.astype(np.int64)
	return float(np.einsum("ij,ij->i", p[t[:, 0]], np.cross(p[t[:, 1]], p[t[:, 2]])).sum())


def test_source_model_import(tmp_path):
	path = gf.source_model(tmp_path)
	assert detect_kind(path) == Kind.MODEL
	assert models.native_format(path) == "source"
	model = models.import_model(path)
	assert model.num_vertices == 6 and model.num_triangles == 4
	assert [b.name for b in model.bones[:2]] == ["root", "elbow"]
	assert model.materials[0].texture.endswith("crate.vtf")
	assert tuple(model.materials[0].rgba[0, 0, :3]) == (200, 0, 0)
	# Source's clockwise fronts become counter-clockwise: the strip faces -Y.
	m = model.meshes[0]
	p, t = m.world, m.tris[0]
	n = np.cross(p[t[1]] - p[t[0]], p[t[2]] - p[t[0]])
	assert n[1] < 0
	seq = model.sequences[0]
	assert seq.name == "bend" and seq.loop and len(seq.frames) == 2
	assert any("helper bones" in w for w in model.warnings)


def _posed(model, frame):
	from gxbake.core import skeleton as sk

	mats = np.array(sk.world_matrices(frame, [b.parent for b in model.bones]))
	out = []
	for sm in model.meshes:
		w = mats[sm.bones]
		out.append(np.einsum("nij,nj->ni", w[:, :3, :3], sm.positions) + w[:, :3, 3])
	return np.concatenate(out)


def test_source_animation_and_helper_bones(tmp_path):
	model = models.import_model(gf.source_model(tmp_path))
	rest = _posed(model, model.sequences[0].frames[0])
	bent = _posed(model, model.sequences[0].frames[1])
	world = np.concatenate([sm.world for sm in model.meshes])
	assert np.allclose(rest, world, atol=1e-3)
	top = bent[np.isclose(world[:, 2], 20)]
	# the elbow turns 90° about X at z=10: (x, 0, 20) → (x, -10, 10)
	assert np.allclose(top[:, 1:], [-10, 10], atol=0.05)
	# the 50/50 vertices follow a helper bone half way round the same joint
	mid = bent[np.isclose(world[:, 2], 15)]
	c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
	assert np.allclose(mid[:, 1:], [-5 * s, 10 + 5 * c], atol=0.05)
	mdl = models.build_studio_mdl(model)
	assert len(models.studio_bones(mdl)) == 3
	assert np.allclose(models.studio_posed_vertices(mdl, 0, 1)[:, 2].max(), 10 + 5 * c, atol=0.1)


def test_source_model_without_blend_bones(tmp_path):
	opts = models.ModelOptions(blend_bones=False)
	model = models.import_model(gf.source_model(tmp_path), opts)
	assert len(model.bones) == 2
	assert source_mdl.source_info(tmp_path / "models/arm.mdl")["Source MDL version"] == 48


def test_project_converts_source_model(tmp_path):
	gf.source_model(tmp_path)
	proj = Project(Settings())
	proj.add_path(tmp_path / "models")
	assert [a.kind for a in proj.assets] == [Kind.MODEL]	# .vvd/.vtx are read with the .mdl
	a = proj.assets[0]
	assert proj.info(a)["Source MDL version"] == 48
	rels = {w.rel for w in proj.convert(proj.assets, tmp_path / "out", log=lambda m: None)}
	assert {"models/arm.mdl", "gc_studio/arm.mdl", "gc_studio/armt.mdl"} <= rels


# -- Quake ------------------------------------------------------------------------------------


def _quake_dir(tmp_path, loose_palette=True):
	game = tmp_path / "quake/id1"
	(game / "progs").mkdir(parents=True)
	(game / "maps").mkdir()
	if loose_palette:
		(game / "gfx").mkdir()
		(game / "gfx/palette.lmp").write_bytes(gf.quake_palette())
	else:
		(game / "pak0.pak").write_bytes(gf.pak({"gfx/palette.lmp": gf.quake_palette()}))
	quake._palette_cache.clear()
	return game


def test_quake_alias_model(tmp_path):
	game = _quake_dir(tmp_path)
	p = game / "progs/tetra.mdl"
	p.write_bytes(gf.quake_mdl(frames=3))
	assert detect_kind(p) == Kind.MODEL
	model = models.import_model(p)
	assert model.num_triangles == 4
	assert _signed_volume(model.meshes[0]) > 0	# outward counter-clockwise
	assert tuple(model.materials[0].rgba[0, 0, :3]) == (200, 55, 77)
	assert any("3 vertex-animation frames" in w for w in model.warnings)
	mdl = models.build_studio_mdl(model)
	assert mdl[:4] == b"IDST"


def test_quake_palette_from_pak_and_bsp29(tmp_path):
	game = _quake_dir(tmp_path, loose_palette=False)
	bsp = game / "maps/e1m1.bsp"
	bsp.write_bytes(gf.bsp29([gf.quake_miptex("wall1", 16, 16, 10), gf.quake_miptex("*water", 16, 16), gf.quake_miptex("sky1", 16, 16)]))
	data, report, entries = gct.build_world_texture_pack(bsp)
	assert [e.name for e in entries] == ["wall1"] and not report.notes
	texel = entries[0].decode(0)[0, 0, :3]
	assert np.abs(texel.astype(int) - (10, 245, 77)).max() <= 8
	proj = Project()
	proj.add_path(bsp)
	a = proj.assets[0]
	assert proj.info(a)["BSP version"] == 29
	assert proj.preview(a, "wall1").result is not None and not a.warnings


def test_wad2_textures_use_quake_palette(tmp_path):
	game = _quake_dir(tmp_path)
	w = game / "textures.wad"
	w.write_bytes(gf.wad2({"metal1": gf.quake_miptex("metal1", 16, 16, 100)}))
	proj = Project()
	proj.add_path(w)
	a = proj.assets[0]
	assert proj.children(a) == ["metal1"]
	rels = {o.rel for o in proj.convert([a], tmp_path / "out", log=lambda m: None)}
	assert rels == {"media/textures.gct"}
	e = gct.read_gct((tmp_path / "out/media/textures.gct").read_bytes())[0]
	assert np.abs(e.decode(0)[0, 0, :3].astype(int) - (100, 155, 77)).max() <= 8


def test_missing_palette_falls_back_to_grey(tmp_path):
	(tmp_path / "lonely.wad").write_bytes(gf.wad2({"x": gf.quake_miptex("x", 8, 8, 60)}))
	quake._palette_cache.clear()
	proj = Project()
	proj.add_path(tmp_path / "lonely.wad")
	pv = proj.preview(proj.assets[0], "x")
	assert tuple(pv.source[0, 0, :3]) == (60, 60, 60)
	assert any("palette" in w for w in proj.assets[0].warnings)


def test_quake_sprite_becomes_hl_sprite(tmp_path):
	game = _quake_dir(tmp_path)
	(game / "progs/s_light.spr").write_bytes(gf.quake_sprite())
	hl = quake.hl_sprite_bytes(game / "progs/s_light.spr")
	info = sprite.spr_info(hl)
	assert info["version"] == 2 and info["texformat"] == "alphatest" and (info["width"], info["height"]) == (16, 16)
	frame = sprite.spr_frames(hl)[0]
	assert tuple(frame[8, 0]) == (40, 215, 77, 255) and frame[0, 0, 3] == 0
	proj = Project()
	proj.add_path(game / "progs/s_light.spr")
	rels = {o.rel for o in proj.convert(proj.assets, tmp_path / "out", log=lambda m: None)}
	assert rels == {"sprites/s_light.spr"}
	lean = (tmp_path / "out/sprites/s_light.spr").read_bytes()
	assert sprite.spr_info(lean)["width"] == 8


def test_quake2_sp2_and_wal(tmp_path):
	base = tmp_path / "quake2/baseq2"
	(base / "sprites").mkdir(parents=True)
	(base / "pics").mkdir()
	(base / "textures/e1u1").mkdir(parents=True)
	pal = gf.quake_palette()
	cm = np.zeros((4, 4), np.uint8)
	(base / "pics/colormap.pcx").write_bytes(gf.pcx8(cm, pal))
	idx = np.full((8, 8), 255, np.uint8)
	idx[2:6, 2:6] = 90
	(base / "sprites/flare_0.pcx").write_bytes(gf.pcx8(idx, pal))
	(base / "sprites/flare.sp2").write_bytes(gf.sp2([(8, 8, "sprites/flare_0.pcx")]))
	(base / "textures/e1u1/floor1_1.wal").write_bytes(gf.wal("e1u1/floor1_1", 16, 16, 30))
	quake._palette_cache.clear()
	assert detect_kind(base / "sprites/flare.sp2") == Kind.SPR
	hl = quake.hl_sprite_bytes(base / "sprites/flare.sp2")
	f = sprite.spr_frames(hl)[0]
	assert tuple(f[3, 3]) == (90, 165, 77, 255) and f[0, 0, 3] == 0
	assert detect_kind(base / "textures/e1u1/floor1_1.wal") == Kind.IMAGE
	from gxbake.core import images

	rgba = images.load_rgba(base / "textures/e1u1/floor1_1.wal")
	assert tuple(rgba[1, 1]) == (30, 225, 77, 255) and rgba[0, 0, 3] == 0


def test_goldsrc_mdl_still_detected(tmp_path):
	p = tmp_path / "gun.mdl"
	p.write_bytes(fx.studio([("skin.bmp", 0, 64, 64)]))
	assert detect_kind(p) == Kind.MDL
	(tmp_path / "gun01.mdl").write_bytes(b"IDSQ" + struct.pack("<i", 10) + bytes(64))
	assert detect_kind(tmp_path / "gun01.mdl") == Kind.UNKNOWN
	assert studio.studio_info(p.read_bytes())


# -- archives and audio -------------------------------------------------------------------------


def test_pak_and_vpk_extract(tmp_path):
	files = {"maps/start.bsp": b"BSP!", "gfx/palette.lmp": bytes(768), "readme.txt": b"hi"}
	(tmp_path / "pak0.pak").write_bytes(gf.pak(files))
	(tmp_path / "pak01_dir.vpk").write_bytes(gf.vpk(files))
	for name in ("pak0.pak", "pak01_dir.vpk"):
		out = tmp_path / ("x_" + name)
		assert archives.extract(tmp_path / name, out, log=lambda m: None) == 3
		assert (out / "maps/start.bsp").read_bytes() == b"BSP!"
		assert archives.extract(tmp_path / name, tmp_path / ("y_" + name), ["gfx/*"], log=lambda m: None) == 1
	from gxbake.cli import main

	assert main(["extract", str(tmp_path / "pak01_dir.vpk"), "-o", str(tmp_path / "cli"), "--match", "*.txt"]) == 0
	assert (tmp_path / "cli/readme.txt").read_bytes() == b"hi"


def test_wav_converts_without_ffmpeg(tmp_path, monkeypatch):
	monkeypatch.setattr(media, "find_ffmpeg", lambda: None)
	for width in (1, 2):
		src = tmp_path / f"beep{width}.wav"
		gf.wav_pcm(src, 22050, 0.5, width)
		out = tmp_path / f"beep{width}.gcpcm"
		assert media.build_gcpcm_companion(src, out, log=lambda m: None)
		data = out.read_bytes()
		hdr = media.read_gcpcm_header(data)
		assert hdr["rate"] == 48000 and abs(hdr["seconds"] - 0.5) < 0.01
		pcm = np.frombuffer(data[16:], ">i2").reshape(-1, 2)
		assert (pcm[:, 0] == pcm[:, 1]).all() and np.abs(pcm).max() > 1000
