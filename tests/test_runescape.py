"""RuneScape caches → studio MDL, on synthetic caches (no game data in the repo)."""

import json
import math

import numpy as np
import pytest

from gxbake import cli
from gxbake.core import models, runescape as rs, skeleton as sk, studio
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import rs_fixtures as rf

RED = (0 << 10) | (7 << 7) | 64	# hue 0, full saturation, mid lightness
BLUE = (43 << 10) | (7 << 7) | 64
WHITE = 127


def _guard():
	"""A two-box figure, y down: body (label 0) on the ground, a head (label 1) on top,
	and a nose sticking out to the south (-z) so the facing can be checked."""
	body, bt = rf.box((0, -100, 0), (40, 100, 20))
	head, ht = rf.box((0, -240, 0), (20, 40, 20))
	nose, nt = rf.box((0, -240, -40), (4, 4, 20))
	verts = np.concatenate([body, head, nose])
	faces = np.concatenate([bt, ht + 8, nt + 16])
	groups = [0] * 8 + [1] * 16
	colors = [RED] * 12 + [BLUE] * 12 + [WHITE] * 12
	return verts, faces, colors, groups


def _anim_cache(tmp_path, encoder=rf.model_old, **npc_kw):
	verts, faces, colors, groups = _guard()
	fmap = rf.framemap([(0, [1]), (2, [1]), (1, [1])])	# origin, rotate, translate for the head group
	frames = {
		0: rf.frame(3, 3, {}),	# rest
		1: rf.frame(3, 3, {1: (None, 64, None)}),	# head turns 64·8 = 512 steps (90°) about y
		2: rf.frame(3, 3, {2: (0, -50, None)}),	# head lifts 50 units (y is down)
	}
	return rf.js5_cache(tmp_path / "cache", {
		0: {7: frames},
		1: {3: {0: fmap}},
		2: {
			9: {
				1: rf.npc("Guard", [5], stand=10, walk=11, **npc_kw),
				2: rf.npc("null", [5]),
				3: rf.npc("Statue", [6]),
			},
			12: {10: rf.sequence([(25, 7, 0), (25, 7, 1)]), 11: rf.sequence([(10, 7, 2)])},
		},
		7: {5: {0: encoder(verts, faces, colors, groups=groups)}, 6: {0: rf.model_new(verts, faces, colors)}},
	})


# -- store ----------------------------------------------------------------------------


@pytest.mark.parametrize("compression", [0, 1, 2])
def test_containers_and_multi_file_archives(compression):
	files = [b"alpha" * 30, b"", b"gamma" * 7]
	packed = rf.pack_archive(files, chunks=3)
	assert rs.split_archive(rs.decompress_container(rf.container(packed, compression, 4)), 3) == files


@pytest.mark.parametrize("protocol", [5, 6, 7])
def test_reference_table_protocols(protocol):
	t = rs.read_reference_table(rf.reference_table({3: [0, 2, 9], 40000: [1]}, protocol, named=True))
	assert sorted(t.archives) == [3, 40000]
	assert t.archives[3].files == [0, 2, 9] and t.archives[40000].files == [1]


def test_js5_cache_reads_archives_files_and_big_ids(tmp_path):
	big = b"m" * 1500	# spans three sectors
	path = rf.js5_cache(tmp_path, {7: {70000: {0: big}, 1: {0: b"x"}}, 2: {9: {0: b"a", 5: b"bb", 6: b"ccc"}}})
	cache = rs.Cache(tmp_path)	# the folder works as well as the .dat2
	assert cache.path == path and not cache.legacy
	assert cache.indices() == [2, 7] and cache.model_ids() == [1, 70000]
	assert cache.model_data(70000) == big
	assert cache.files(2, 9) == {0: b"a", 5: b"bb", 6: b"ccc"}
	assert cache.file(2, 9, 5) == b"bb" and cache.file(2, 9, 7) is None
	assert cache.raw(7, 3) is None


def test_legacy_cache_gzipped_models(tmp_path):
	verts, faces, colors, _ = _guard()
	rf.legacy_cache(tmp_path, {12: rf.model_old(verts, faces, colors)})
	assert detect_kind(tmp_path / "main_file_cache.dat") == Kind.MODEL
	assert detect_kind(tmp_path / "main_file_cache.idx1") == Kind.UNKNOWN
	cache = rs.Cache(tmp_path / "main_file_cache.dat")
	assert cache.legacy and cache.model_ids() == [12]
	assert [e.key for e in rs.entries(cache.path)] == ["model:12"]
	model = rs.import_runescape(cache.path, models.ModelOptions(), "model:12")
	assert model.num_triangles == 36
	with pytest.raises(ValueError, match="legacy"):
		rs.import_runescape(cache.path, models.ModelOptions(), "npc:1")


# -- models -----------------------------------------------------------------------------


@pytest.mark.parametrize("encoder,rigged,name", [
	(rf.model_old, False, "old"), (rf.model_old, True, "rigged (0xFFFE)"),
	(rf.model_new, False, "new (0xFFFF)"), (rf.model_new, True, "rigged (0xFFFD)"),
])
def test_model_encodings_round_trip(encoder, rigged, name):
	verts, faces, colors, groups = _guard()
	render = [i % 2 for i in range(len(faces))]
	tex = [-1] * len(faces)
	tex[3], tex[30] = 7, 9
	coords = [-1] * len(faces)
	coords[3], coords[30] = 0, 1
	data = encoder(verts, faces, colors, groups=groups, render=render, textures=tex, tex_coords=coords,
		tex_tris=[(0, 1, 2), (16, 17, 18)], rigged=rigged)
	m = rs.read_model(data)
	assert m.encoding == name and m.rigged == rigged
	assert (m.vertices == verts).all() and (m.faces == faces).all()
	assert list(m.groups) == groups and list(m.render_types & 1) == render
	assert list(m.textures) == tex and m.tex_coords[3] == 0 and m.tex_coords[30] == 1
	assert m.tex_tris.tolist() == [[0, 1, 2], [16, 17, 18]]
	untextured = [c for c, t in zip(colors, tex) if t < 0]
	assert [int(c) for c, t in zip(m.colors, tex) if t < 0] == untextured


def test_face_stream_uses_every_compression_type():
	_v, faces, _c, _g = _guard()
	strip = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [2, 3, 5]])	# types 1, 2, 3, 4
	types, _ = rf._face_streams(strip)
	assert list(types) == [1, 2, 3, 4]
	verts = np.arange(18).reshape(6, 3)
	m = rs.read_model(rf.model_old(verts, strip, [WHITE] * 4))
	assert m.faces.tolist() == strip.tolist()


def test_hsl_palette_colours():
	rgb = rs.hsl_to_rgb(np.array([RED, BLUE, WHITE, 0]))
	assert rgb[0, 0] > 200 and rgb[0, 1] < 80 and rgb[0, 2] < 80
	assert rgb[1, 2] > 200 and rgb[1, 0] < 80
	assert rgb[2].min() > 240 and rgb[3].max() < 40


def test_texture_uvs_follow_the_texture_triangle():
	verts = np.array([[0, 0, 0], [128, 0, 0], [0, 0, 128], [64, 0, 64]])
	m = rs.read_model(rf.model_new(verts, [[0, 1, 2], [3, 1, 2]], [0, 0], textures=[4, 4], tex_coords=[0, 0],
		tex_tris=[(0, 1, 2)]))
	uv = rs.face_uvs(m)
	assert np.allclose(uv[0], [[0, 0], [1, 0], [0, 1]])
	assert np.allclose(uv[1], [[0.5, 0.5], [1, 0], [0, 1]])


# -- import ---------------------------------------------------------------------------------


def test_npc_import_axes_palette_recolour_and_scale(tmp_path):
	path = _anim_cache(tmp_path, recolor={BLUE: WHITE}, width=256)
	opts = models.ModelOptions(rs_scale=0.5, static=True)
	m = rs.import_runescape(path, opts, "npc:1")
	assert m.name == "Guard" and len(m.materials) == 1 and m.materials[0].name == "palette"
	lo, hi = m.bounds()
	# RuneScape y is down and models face -z; GoldSrc Z is up and models face +X.
	assert hi[2] == pytest.approx(280 * 0.5) and lo[2] == pytest.approx(0)
	assert hi[0] == pytest.approx(60 * 2 * 0.5)	# the nose reaches 60 units south → +X (width scale 2)
	assert hi[1] - lo[1] == pytest.approx(80 * 0.5 * 2)	# body width 80, doubled by the width scale
	skin = m.materials[0].rgba
	assert skin.shape == (8, 8, 4)	# two colours (blue recoloured to white): one 4x4 block each
	sm = m.meshes[0]
	cols = {tuple(skin[int(v * 8), int(u * 8), :3]) for u, v in sm.uvs}
	assert cols == {tuple(c) for c in rs.hsl_to_rgb(np.array([RED, WHITE]))}


def test_yaw_turns_the_model(tmp_path):
	path = _anim_cache(tmp_path)
	m = rs.import_runescape(path, models.ModelOptions(rs_scale=1.0, rs_yaw=90.0, static=True), "npc:3")
	lo, hi = m.bounds()
	assert hi[1] == pytest.approx(60)	# the nose now points +Y


def _world(model, seq, frame):
	poses = model.sequences[seq].frames[frame]
	mats = sk.world_matrices(poses, [b.parent for b in model.bones])
	out = []
	for sm in model.meshes:
		for p, b in zip(sm.positions, sm.bones):
			out.append(mats[b][:3, :3] @ p + mats[b][:3, 3])
	return np.array(out)


@pytest.mark.parametrize("encoder", [rf.model_old, rf.model_new])
def test_label_groups_become_bones_and_frames_become_poses(tmp_path, encoder):
	path = _anim_cache(tmp_path, encoder)
	opts = models.ModelOptions(rs_scale=1.0, anim_fps=10)
	m = rs.import_runescape(path, opts, "npc:1")
	assert [b.name for b in m.bones] == ["root", "group0", "group1"]
	assert [s.name for s in m.sequences] == ["idle", "walk"]
	idle, walk = m.sequences
	assert len(idle.frames) == 10	# 2 × 25 ticks of 20 ms at 10 fps
	rest = _world(m, 0, 0)
	assert np.allclose(rest, np.concatenate([sm.world for sm in m.meshes]), atol=1e-3)
	# frame 1 (from 0.5 s): head and nose turn 90° about the vertical through the group's centroid,
	# which the nose pulls 20 units forward
	turned = _world(m, 0, 6)
	head = np.concatenate([sm.bones for sm in m.meshes]) == 2	# label 1: head and nose
	assert np.allclose(turned[~head], rest[~head], atol=1e-3)
	centre = np.array([20.0, 0.0, 240.0])
	nose_tip = rest[head][:, 0].argmax()
	tip = turned[head][nose_tip]
	assert abs(tip[0] - centre[0]) <= 4 + 1e-3 and abs(abs(tip[1]) - 40) < 1e-3	# swung from +X to ±Y
	assert np.allclose(np.linalg.norm(turned[head] - centre, axis=1), np.linalg.norm(rest[head] - centre, axis=1), atol=1e-3)
	# walk: the head lifts 50 units (RuneScape -y)
	lifted = _world(m, 1, 0)
	assert np.allclose(lifted[head] - rest[head], [0, 0, 50], atol=1e-3)
	assert np.allclose(lifted[~head], rest[~head], atol=1e-3)


def test_studio_mdl_from_npc_decodes(tmp_path):
	path = _anim_cache(tmp_path)
	m = rs.import_runescape(path, models.ModelOptions(), "npc:1")
	mdl = models.build_studio_mdl(m, models.ModelOptions())
	info = studio.studio_info(mdl)
	assert info and studio.lean_studio_mesh_bytes_from(mdl)


def test_textured_faces_use_the_sprite(tmp_path):
	verts = np.array([[0, 0, 0], [128, 0, 0], [0, 0, 128], [128, 0, 128]])
	faces = [[0, 2, 1], [1, 2, 3]]
	rgba = np.zeros((4, 4, 4), np.uint8)
	rgba[:2] = (255, 255, 0, 255)	# top half yellow, bottom half transparent
	path = rf.js5_cache(tmp_path, {
		7: {1: {0: rf.model_new(verts, faces, [RED, RED], textures=[3, -1])}},
		8: {40: {0: rf.sprite(rgba)}},
		9: {0: {3: rf.texture(40), 4: rf.texture(40, new_layout=False)}},
	})
	m = rs.import_runescape(path, models.ModelOptions(), "model:1")
	names = [mat.name for mat in m.materials]
	assert names == ["palette", "texture3"]
	tex = m.materials[1]
	assert tex.masked and tex.rgba[0, 0, 3] == 255 and tex.rgba[-1, 0, 3] == 0
	cache = rs.Cache(path)
	assert rs.texture_sprite(cache.file(9, 0, 4)) == 40


def test_missing_texture_falls_back_to_colour(tmp_path):
	verts = np.array([[0, 0, 0], [128, 0, 0], [0, 0, 128]])
	path = rf.js5_cache(tmp_path, {7: {1: {0: rf.model_new(verts, [[0, 2, 1]], [RED], textures=[3])}}})
	m = rs.import_runescape(path, models.ModelOptions(), "model:1")
	assert [mat.name for mat in m.materials] == ["palette"]
	assert any("texture 3 not found" in w for w in m.warnings)


def test_npc_reader_stops_at_revision_dependent_opcodes():
	d = rs.read_npc(rf.npc("Hans", [1, 2], stand=8, extra=bytes([102, 1, 2, 3])), 4)
	assert d.name == "Hans" and d.models == [1, 2] and d.stand == 8 and d.partial


# -- project and CLI ------------------------------------------------------------------------


def test_project_lists_named_npcs_and_converts(tmp_path):
	path = _anim_cache(tmp_path)
	proj = Project(Settings())
	added = proj.add_path(tmp_path)
	assert [a.name for a in added] == ["main_file_cache.dat2 › npc:1 Guard", "main_file_cache.dat2 › npc:3 Statue"]
	assert proj.add_path(path) == []
	guard = added[0]
	assert proj.info(guard)["Source"] == "RuneScape cache, npc:1"
	out = tmp_path / "out"
	proj.convert(added, out, log=lambda _m: None)
	assert all(a.status.startswith("Done") for a in added), [a.warnings for a in added]
	names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.mdl"))
	assert "gc_studio/rs_npc1_guard.mdl" in names and "models/rs_npc3_statue.mdl" in names

	only = Project(Settings(rs_entries=["model:6"]))
	assert [a.stem for a in only.add_path(path)] == ["rs_model6"]


def test_cli_lists_and_converts(tmp_path, capsys):
	path = _anim_cache(tmp_path)
	assert cli.main(["rs", str(path), "--find", "gua"]) == 0
	text = capsys.readouterr().out
	assert "npc:1" in text and "Guard" in text and "Statue" not in text
	assert cli.main(["rs", str(path), "--json"]) == 0
	data = json.loads(capsys.readouterr().out)
	assert [n["entry"] for n in data["npcs"]] == ["npc:1", "npc:3"] and data["models"] == 2
	out = tmp_path / "out"
	assert cli.main(["convert", str(path), "-o", str(out), "--rs-entry", "npc:1", "--rs-scale", "0.5"]) == 0
	assert (out / "models" / "rs_npc1_guard.mdl").is_file()
	with pytest.raises(ValueError):
		rs.parse_entry("item:4")


def test_previews_colour_faces_from_the_palette(tmp_path):
	path = _anim_cache(tmp_path)
	m = rs.import_runescape(path, models.ModelOptions(), "npc:1")
	mdl = models.build_studio_mdl(m, models.ModelOptions())
	comp = studio.lean_studio_tex_companion_bytes_from(mdl, 64)
	for img in (models.preview_imported(m), models.preview_studio(mdl, comp)):
		px = img[..., :3].reshape(-1, 3).astype(int)
		assert ((px[:, 0] > 120) & (px[:, 2] < 60)).any()	# red body
		assert ((px[:, 2] > 120) & (px[:, 0] < 60)).any()	# blue head
