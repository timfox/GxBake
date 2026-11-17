"""ROSE Online formats → studio MDL, on synthetic files (no game data in the repo)."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from gxbake.core import models, rose, studio
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import fixtures as fx

ID = (1.0, 0.0, 0.0, 0.0)	# wxyz identity


def _close(a, b, tol=0.05):
	a = np.array(sorted(map(tuple, np.round(a, 2))))
	b = np.array(sorted(map(tuple, np.round(b, 2))))
	return a.shape == b.shape and np.allclose(a, b, atol=tol)


def _quat_x(deg):
	h = math.radians(deg) / 2
	return (math.cos(h), math.sin(h), 0.0, 0.0)


def _dds(path: Path, rgba) -> None:
	Image.fromarray(np.asarray(rgba, np.uint8)).save(path, "DDS")


def _arm(root: Path) -> None:
	"""3DDATA/NPC/ARM: a two-bone arm (root at the origin, elbow 1 m up) that bends 90° about X."""
	d = root / "3DDATA" / "NPC" / "arm"
	d.mkdir(parents=True)
	(d / "arm.zmd").write_bytes(fx.rose_zmd(
		[("root", -1, (0, 0, 0), ID), ("elbow", 0, (0, 0, 100), ID)],
		[("hand_dummy", 1, (0, 0, 50), ID)]))
	lower, lt = fx.box((0, 0, 0.5), (0.1, 0.1, 0.4))
	upper, ut = fx.box((0, 0, 1.5), (0.1, 0.1, 0.4))
	pos = np.concatenate([lower, upper])
	tris = np.concatenate([lt, ut + 8])
	joints = np.zeros((16, 4), int)
	joints[8:, 0] = 1	# local index 1 → ZMD bone 1 through the bone map
	weights = np.zeros((16, 4))
	weights[:, 0] = 1.0
	uvs = np.tile([[0.25, 0.25]], (16, 1))
	(d / "arm.zms").write_bytes(fx.rose_zms(pos, tris, uvs=uvs, joints=joints, weights=weights, bone_map=[0, 1]))
	_dds(d / "arm.dds", np.full((8, 8, 4), (200, 50, 50, 255)))
	knob, kt = fx.box((0, 0, 0), (0.05, 0.05, 0.05))
	(d / "knob.zms").write_bytes(fx.rose_zms(knob, kt, uvs=np.zeros((8, 2))))
	frames = 11
	(root / "3DDATA" / "MOTION").mkdir(parents=True)
	(root / "3DDATA" / "MOTION" / "arm_bend.zmo").write_bytes(fx.rose_zmo(10, frames, [
		(4, 1, [_quat_x(90 * f / (frames - 1)) for f in range(frames)]),
		(2, 1, [(0, 0, 100)] * frames),
	]))
	(root / "3DDATA" / "MOTION" / "arm_idle.zmo").write_bytes(fx.rose_zmo(10, 2, [(4, 1, [ID, ID])]))
	# PART_NPC.ZSC: object 0 = the skinned arm, object 1 = a knob on the hand dummy
	(root / "3DDATA" / "NPC" / "part_npc.zsc").write_bytes(fx.rose_zsc(
		["3DDATA\\NPC\\ARM\\ARM.ZMS", "3DDATA\\NPC\\ARM\\KNOB.ZMS"],
		[("3DDATA\\NPC\\ARM\\ARM.DDS", False, False)],
		[[{"mesh": 0, "material": 0}], [{"mesh": 1, "material": 0, "dummy": 0}]]))
	(root / "3DDATA" / "NPC" / "list_npc.chr").write_bytes(fx.rose_chr(
		["3DDATA\\NPC\\ARM\\ARM.ZMD"],
		["3DDATA\\MOTION\\ARM_IDLE.ZMO", "3DDATA\\MOTION\\ARM_BEND.ZMO"],
		[None, (0, "Arm Thing", [0], [(0, 0), (2, 1)])]))


def _bent(points, deg):
	"""Expected positions (cm) of the upper box after bending the elbow."""
	r = math.radians(deg)
	rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
	j = np.array([0, 0, 1.0])
	return ((np.asarray(points) - j) @ rx.T + j) * 100


def test_parsers_read_what_the_fixtures_write(tmp_path):
	_arm(tmp_path)
	d = tmp_path / "3DDATA" / "NPC" / "arm"
	mesh = rose.read_zms((d / "arm.zms").read_bytes())
	assert mesh.version == 8 and mesh.positions.shape == (16, 3) and mesh.tris.shape == (24, 3)
	assert mesh.skinned and mesh.joints[8:, 0].tolist() == [1] * 8 and mesh.uvs[0].tolist() == [0.25, 0.25]
	zmd = rose.read_zmd((d / "arm.zmd").read_bytes())
	assert [(b.name, b.parent) for b in zmd.bones] == [("root", -1), ("elbow", 0)]
	assert np.allclose(zmd.bones[1].position, [0, 0, 1.0]) and zmd.dummies[0].parent == 1
	zmo = rose.read_zmo((tmp_path / "3DDATA" / "MOTION" / "arm_bend.zmo").read_bytes())
	assert zmo.fps == 10 and zmo.num_frames == 11 and set(zmo.rotations) == {1}
	assert np.allclose(zmo.rotations[1][0], [0, 0, 0, 1]) and np.allclose(zmo.positions[1][3], [0, 0, 1.0])
	zsc = rose.read_zsc((tmp_path / "3DDATA" / "NPC" / "part_npc.zsc").read_bytes())
	assert len(zsc.objects) == 2 and zsc.objects[1].parts[0].dummy == 0
	chr_ = rose.read_chr((tmp_path / "3DDATA" / "NPC" / "list_npc.chr").read_bytes())
	assert [(c.index, c.name, c.objects) for c in chr_.characters] == [(1, "Arm Thing", [0])]


def test_zms_v6_centimetres_and_vertex_ids():
	import struct

	pos = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0]], np.float32)
	out = b"ZMS0006\0" + struct.pack("<I", 1 << 1) + bytes(24) + struct.pack("<I", 0) + struct.pack("<I", 3)
	out += b"".join(struct.pack("<I3f", i, *p) for i, p in enumerate(pos))
	out += struct.pack("<I", 1) + struct.pack("<4I", 0, 0, 1, 2) + struct.pack("<I", 0)
	mesh = rose.read_zms(out)
	assert np.allclose(mesh.positions[1], [1, 0, 0]) and mesh.tris.tolist() == [[0, 1, 2]]


def test_chr_character_bakes_skeleton_and_motions(tmp_path):
	_arm(tmp_path)
	chr_path = tmp_path / "3DDATA" / "NPC" / "list_npc.chr"
	assert [(e.index, e.label) for e in rose.entries(chr_path)] == [(1, "Arm Thing")]
	opts = models.ModelOptions(anim_fps=10, rose_scale=100.0, rose_yaw=0.0)
	model = rose.import_rose(chr_path, opts, entry=1)
	assert [(b.name, b.parent) for b in model.bones] == [("root", -1), ("elbow", 0), ("hand_dummy", 1)]
	assert [(s.name, s.loop) for s in model.sequences] == [("stop", True), ("attack", False)]
	assert model.sequences[1].frames.shape == (11, 3, 6)
	assert model.materials[0].texture.lower().endswith("arm.dds")
	assert tuple(model.materials[0].rgba[0, 0]) == (200, 50, 50, 255)
	raw = models.build_studio_mdl(model, opts)
	info = studio.studio_info(raw)
	assert info["bones"] == 3 and info["sequences"] == 2
	lower, _ = fx.box((0, 0, 0.5), (0.1, 0.1, 0.4))
	upper, _ = fx.box((0, 0, 1.5), (0.1, 0.1, 0.4))
	assert _close(models.studio_posed_vertices(raw, 1, 0), np.concatenate([lower * 100, _bent(upper, 0)]))
	assert _close(models.studio_posed_vertices(raw, 1, 10), np.concatenate([lower * 100, _bent(upper, 90)]))
	assert _close(models.studio_posed_vertices(raw, 1, 5), np.concatenate([lower * 100, _bent(upper, 45)]))
	# default yaw turns the model a quarter: ROSE -Y forward becomes GoldSrc +X
	turned = rose.import_rose(chr_path, models.ModelOptions(rose_scale=100.0), entry=1)
	lo, hi = turned.bounds()
	assert np.allclose(lo, [-10, -10, 10], atol=1e-3) and np.allclose(hi, [10, 10, 190], atol=1e-3)


def test_zsc_object_hierarchy_two_sided_and_masked(tmp_path):
	d = tmp_path / "3DDATA" / "JUNON" / "deco"
	d.mkdir(parents=True)
	quad = np.array([[-0.5, 0, 0], [0.5, 0, 0], [0.5, 0, 1], [-0.5, 0, 1]])
	(d / "leaf.zms").write_bytes(fx.rose_zms(quad, [[0, 1, 2], [0, 2, 3]], uvs=[[0, 1], [1, 1], [1, 0], [0, 0]]))
	trunk, tt = fx.box((0, 0, 0.5), (0.1, 0.1, 0.5))
	(d / "trunk.zms").write_bytes(fx.rose_zms(trunk, tt, uvs=np.zeros((8, 2))))
	leaf = np.full((8, 8, 4), (40, 160, 40, 255))
	leaf[:4, :4, 3] = 0
	_dds(d / "leaf.dds", leaf)
	_dds(d / "trunk.dds", np.full((8, 8, 4), (120, 80, 40, 30)))	# alpha is a spec map on an opaque material
	(tmp_path / "3DDATA" / "JUNON" / "list_deco.zsc").write_bytes(fx.rose_zsc(
		["3Ddata\\JUNON\\DECO\\TRUNK.ZMS", "3Ddata\\JUNON\\DECO\\LEAF.ZMS"],
		[("3Ddata\\JUNON\\DECO\\TRUNK.DDS", False, False), ("3Ddata\\JUNON\\DECO\\LEAF.DDS", True, True)],
		[[], [{"mesh": 0, "material": 0, "pos": (200, 0, 0)},
			{"mesh": 1, "material": 1, "pos": (0, 0, 100), "quat": (*_quat_x(0),), "parent": 1}]]))
	zsc_path = tmp_path / "3DDATA" / "JUNON" / "list_deco.zsc"
	assert [e.index for e in rose.entries(zsc_path)] == [1]
	model = rose.import_rose(zsc_path, models.ModelOptions(rose_scale=100.0, rose_yaw=0.0), entry=1)
	assert len(model.bones) == 1 and not model.animated and not model.warnings
	trunk_m, leaf_m = model.materials
	assert not trunk_m.masked and (trunk_m.rgba[..., 3] == 255).all()
	assert leaf_m.masked
	leaf_mesh = model.meshes[1]
	assert len(leaf_mesh.tris) == 4	# back faces added for the two-sided material
	assert np.allclose(leaf_mesh.world.min(0), [150, 0, 100]) and np.allclose(leaf_mesh.world.max(0), [250, 0, 200])
	raw = models.build_studio_mdl(model)
	assert studio.studio_info(raw)["textures"] == 2


def test_single_zms_finds_skeleton_texture_and_clip(tmp_path):
	_arm(tmp_path)
	d = tmp_path / "3DDATA" / "NPC" / "arm"
	(d / "arm@wave.zmo").write_bytes((tmp_path / "3DDATA" / "MOTION" / "arm_bend.zmo").read_bytes())
	model = rose.import_rose(d / "arm.zms", models.ModelOptions(anim_fps=5))
	assert len(model.bones) == 3 and [s.name for s in model.sequences] == ["wave"]
	static = rose.import_rose(d / "arm.zms", models.ModelOptions(static=True))
	assert len(static.bones) == 1 and any("static" in w for w in static.warnings)


def test_manifest_assembles_parts_and_motions(tmp_path):
	_arm(tmp_path)
	man = tmp_path / "arm.rosemdl"
	man.write_text(json.dumps({
		"name": "armguy", "skeleton": "3DDATA/NPC/ARM/ARM.ZMD",
		"parts": [{"zsc": "3DDATA/NPC/PART_NPC.ZSC", "object": 0}, {"mesh": "3DDATA/NPC/ARM/KNOB.ZMS", "bone": 1}],
		"motions": {"idle": "3DDATA/MOTION/ARM_IDLE.ZMO", "bend": "3DDATA/MOTION/ARM_BEND.ZMO"}}))
	model = rose.import_rose(man, models.ModelOptions(rose_scale=100.0, rose_yaw=0.0))
	assert model.name == "armguy" and [s.name for s in model.sequences] == ["idle", "bend"]
	knob = model.meshes[-1]
	assert set(knob.bones.tolist()) == {1} and np.allclose(knob.world.mean(0), [0, 0, 100], atol=1e-3)


def test_project_expands_lists_and_skips_parts(tmp_path):
	_arm(tmp_path)
	assert detect_kind(tmp_path / "x.ZMS") == Kind.MODEL and detect_kind(tmp_path / "x.zmo") == Kind.UNKNOWN
	proj = Project(Settings())
	added = proj.add_path(tmp_path)
	# the .chr's character only: its PART_*.ZSC, the loose .zms and the .dds are pulled in through it
	assert [a.name for a in added] == ["list_npc.chr › 1 Arm Thing"]
	assert proj.add_path(tmp_path / "3DDATA" / "NPC" / "list_npc.chr") == []
	asset = added[0]
	assert proj.children(asset)[-2:] == ["▶ stop", "▶ attack"]
	assert proj.info(asset)["Bones"] == 3
	out = tmp_path / "out"
	proj.convert(added, out, log=lambda _m: None)
	assert asset.status.startswith("Done"), asset.warnings
	names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.mdl"))
	assert names == ["gc_studio/npc_1_arm_thing.mdl", "gc_studio/npc_1_arm_thingt.mdl", "models/npc_1_arm_thing.mdl"]

	only = Project(Settings(rose_entries=[5]))
	assert only.add_path(tmp_path / "3DDATA" / "NPC" / "list_npc.chr") == []
