"""Final Fantasy VII PC field models and LGP archives, on synthetic files."""

import json

import numpy as np
import pytest

from gxbake import cli
from gxbake.core import archives, ff7, models, skeleton as sk, studio
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import ff7_fixtures as f7


def _field_model(folder, anims=True):
	"""A hip bone (length 2) carrying a grey cube, and a head bone on its end carrying a textured cube."""
	folder.mkdir(parents=True, exist_ok=True)
	v, n, f = f7.cube(0.5)
	(folder / "hipr.p").write_bytes(f7.p_model(v, f, n))
	uv = [(0.0, 0.0)] * 8
	(folder / "head.p").write_bytes(f7.p_model(v, f, n, groups=[(0, 12, 0, 0, 1, 0)], texcoords=uv))
	(folder / "HIPR.RSD").write_text(f7.rsd("HIPR"))
	(folder / "head.rsd").write_text(f7.rsd("HEAD", ["FACE"]))
	(folder / "face.tex").write_bytes(f7.tex(2, 2, [(0, 0, 0), (255, 0, 0)], [[1, 1], [1, 0]]))
	(folder / "aaaa.hrc").write_text(f7.hrc("sd_test", [("hip", "root", 2.0, ["HIPR"]), ("head", "hip", 1.0, ["HEAD"])]))
	if anims:
		# frame 0: the hip points up (alpha -90 turns -Z to -Y, which is up); frame 1: the head nods 90° about X
		(folder / "stand.a").write_bytes(f7.anim([((0, 0, 0), (0, 0, 0), [(-90, 0, 0), (0, 0, 0)])] * 2))
		(folder / "nod.a").write_bytes(f7.anim([((0, 0, 0), (0, 0, 0), [(-90, 0, 0), (0, 0, 0)]),
			((0, 0, 0), (0, 0, 0), [(-90, 0, 0), (90, 0, 0)])]))
		(folder / "other.a").write_bytes(f7.anim([((0, 0, 0), (0, 0, 0), [(0, 0, 0)] * 5)]))
		(folder / "test.ff7mdl").write_text(json.dumps({"skeleton": "AAAA.HRC", "animations": ["stand.a", "nod.a"]}))
	return folder


def test_p_model_reader():
	v, n, f = f7.cube(1.0)
	pm = ff7.read_p(f7.p_model(v, f, n, colors=[(10, 20, 30, 255)] * 8))
	assert pm.vertices.shape == (8, 3) and pm.polys.shape == (12, 3) and len(pm.groups) == 1
	assert tuple(pm.vcolors[0]) == (10, 20, 30, 255)
	assert pm.groups[0].poly_count == 12 and not pm.groups[0].textured
	with pytest.raises(ValueError):
		ff7.read_p(b"\0" * 200)


def test_tex_palette_and_colour_key():
	rgba = ff7.read_tex(f7.tex(2, 2, [(0, 0, 0), (255, 0, 0)], [[1, 0], [1, 1]]))
	assert rgba.shape == (2, 2, 4)
	assert tuple(rgba[0, 0]) == (255, 0, 0, 255) and rgba[0, 1, 3] == 0
	opaque = ff7.read_tex(f7.tex(2, 2, [(0, 0, 0), (255, 0, 0)], [[1, 0], [1, 1]], color_key=False))
	assert opaque[0, 1, 3] == 255


def test_hrc_rsd_and_animation_readers(tmp_path):
	bones = ff7.read_hrc(f7.hrc("x", [("hip", "root", 2.9662, ["ABJC"]), ("joint", "hip", 3.5, [])]))
	assert [(b.name, b.parent, b.resources) for b in bones] == [("hip", "root", ["ABJC"]), ("joint", "hip", [])]
	assert bones[0].length == pytest.approx(2.9662)
	r = ff7.read_rsd(f7.rsd("ACAB", ["ACAC", "ACAD"]))
	assert r.ply == "ACAB.PLY" and r.textures == ["ACAC.TIM", "ACAD.TIM"]
	frames = ff7.read_a(f7.anim([((1, 2, 3), (4, 5, 6), [(7, 8, 9)])] * 3))
	assert len(frames) == 3 and frames[0].rotations.tolist() == [[7, 8, 9]]
	assert frames[2].root_translation.tolist() == [4, 5, 6]


def test_rotation_order_is_y_x_z():
	r = ff7.rotation(np.array([90.0, 90.0, 0.0]))
	# Rx first (applied to the vector last): Ry(90)·Rx(90)·(0,0,1)
	assert np.allclose(r @ [0, 0, 1], [0, -1, 0], atol=1e-9)
	assert np.allclose(r @ [1, 0, 0], [0, 0, -1], atol=1e-9)


def test_manifest_poses_parts_and_bakes_animations(tmp_path):
	man = _field_model(tmp_path / "char") / "test.ff7mdl"
	m = ff7.import_ff7(man, models.ModelOptions(ff7_scale=10.0, anim_fps=30))
	assert [b.name for b in m.bones] == ["root", "hip", "head"]
	assert [b.parent for b in m.bones] == [-1, 0, 1]
	hip = [s for s in m.meshes if set(s.bones.tolist()) == {1}][0]
	head = [s for s in m.meshes if set(s.bones.tolist()) == {2}][0]
	assert np.allclose(hip.world.mean(0), 0, atol=1e-4)
	# the head part sits on the end of the 2-unit hip, which points up: GoldSrc +Z
	assert np.allclose(head.world.mean(0), (0, 0, 20.0), atol=1e-4)
	assert m.materials[head.material].rgba[0, 0, 0] == 255	# from face.tex
	assert m.materials[hip.material].name == "vertex_colours"
	# every triangle faces out, whatever way the file wound it
	for s in (hip, head):
		t = s.world[s.tris]
		nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
		assert (np.einsum("ni,ni->n", nrm, t.mean(1) - s.world.mean(0)) > 0).all()
	assert [s.name for s in m.sequences] == ["stand", "nod"]
	nod = m.sequences[1]
	assert nod.frames.shape == (2, 3, 6)
	w = sk.world_matrices(nod.frames[-1], [b.parent for b in m.bones])
	assert np.allclose(w[2][:3, 3], (0, 0, 20.0), atol=1e-4)	# the head pivots in place
	# the nod (+90 about X) undoes the hip's -90, so the head ends square with the world
	assert np.allclose(w[2][:3, :3], np.eye(3), atol=1e-6)
	w0 = sk.world_matrices(nod.frames[0], [b.parent for b in m.bones])
	assert not np.allclose(w0[2][:3, :3], np.eye(3), atol=1e-3)
	mdl = models.build_studio_mdl(m)
	assert mdl[:4] == b"IDST" and studio.lean_studio_mesh_bytes_from(mdl)


def test_hrc_without_animation_and_lone_p(tmp_path):
	folder = _field_model(tmp_path / "char", anims=False)
	m = ff7.import_ff7(folder / "aaaa.hrc", models.ModelOptions(ff7_scale=1.0))
	assert any("no animation" in w for w in m.warnings)
	head = [s for s in m.meshes if set(s.bones.tolist()) == {2}][0]
	# straight bones run along FF7 -Z, which is taken as forward (+X)
	assert np.allclose(head.world.mean(0), (2.0, 0, 0), atol=1e-4)
	lone = ff7.import_ff7(folder / "head.p", models.ModelOptions(ff7_scale=1.0))
	assert len(lone.bones) == 1 and lone.materials[0].texture == "head"


def test_yaw(tmp_path):
	folder = _field_model(tmp_path / "char", anims=False)
	m = ff7.import_ff7(folder / "aaaa.hrc", models.ModelOptions(ff7_scale=1.0, ff7_yaw=90.0))
	head = [s for s in m.meshes if set(s.bones.tolist()) == {2}][0]
	assert np.allclose(head.world.mean(0), (0, 2.0, 0), atol=1e-4)


def test_lgp_archive_lists_and_extracts(tmp_path):
	files = {"aaaa.hrc": b"skeleton", "hipr.p": b"\1\2\3"}
	arc = tmp_path / "char.lgp"
	arc.write_bytes(f7.lgp(files))
	mem = archives.members(arc)
	assert [(m.name, m.read()) for m in mem] == list(files.items())
	assert archives.read_member(arc, "HIPR.P") == b"\1\2\3"
	assert archives.extract(arc, tmp_path / "out", log=lambda *_: None) == 2
	assert (tmp_path / "out" / "aaaa.hrc").read_bytes() == b"skeleton"


def test_project_takes_skeletons_and_skips_their_parts(tmp_path):
	folder = _field_model(tmp_path / "char")
	assert detect_kind(folder / "hipr.p") == Kind.MODEL
	assert detect_kind(folder / "stand.a") == Kind.UNKNOWN
	proj = Project(Settings())
	added = proj.add_path(folder)
	assert sorted(a.path.name for a in added) == ["aaaa.hrc", "test.ff7mdl"]
	out = tmp_path / "out"
	proj.convert(added, out)
	assert all(a.status.startswith("Done") for a in added), [(a.name, a.status, a.warnings) for a in added]
	assert (out / "gc_studio" / "test.mdl").is_file() and (out / "models" / "aaaa.mdl").is_file()
	info = proj.info([a for a in added if a.path.suffix == ".ff7mdl"][0])
	assert info["Source"] == "Final Fantasy VII field model" and info["Bones"] == 3


def test_cli(tmp_path, capsys):
	folder = _field_model(tmp_path / "char")
	assert cli.main(["ff7", str(folder / "aaaa.hrc"), "--anims"]) == 0
	out = capsys.readouterr().out
	assert "head" in out and "2 animations with 2 bones: nod.a, stand.a" in out
	assert cli.main(["ff7", str(folder / "nod.a")]) == 0
	assert "frames: 2" in capsys.readouterr().out
	assert cli.main(["convert", str(folder / "test.ff7mdl"), "-o", str(tmp_path / "o"), "--ff7-scale", "3"]) == 0
	assert (tmp_path / "o" / "gc_studio" / "test.mdl").is_file()
