"""Final Fantasy XI DATs and sounds → studio MDL / WAV / GCPCM, on synthetic files."""

import json
import math
import struct

import numpy as np
import pytest

from gxbake import cli
from gxbake.core import ffxi, models, sfx, skeleton as sk, studio
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import ffxi_fixtures as ff

S = math.sqrt(0.5)
QI = (0.0, 0.0, 0.0, 1.0)


def _npc(flip=False, two_bone=False, anim=True):
	"""Root at the origin, a 'head' bone one unit up (FFXI is y-down) with a red cube on it,
	and a second cube half a unit out along +x (the way FFXI characters face)."""
	head_v, head_f = ff.box(0, 0, 0)	# bone-local
	nose_v, nose_f = ff.box(0.5, -1.0, 0, 0.05)	# root-local
	one = head_v + nose_v
	assign = [(ff.bone(1), 0)] * 8 + [(ff.bone(0), 0)] * 8
	two = []
	if two_bone:	# one vertex shared 30/70 between root (at y=-0.3) and head (at y=+0.7 local)
		two = [(0.0, -0.3 * 0.3, 0.0, 0.0, 0.7 * 0.7, 0.0, 0.3, 0.7, 0, 1, 0)]
		assign.append((ff.bone(0), ff.bone(1)))
	polys = ff.set_texture("tex_red") + ff.tri_list(head_f) + ff.set_texture("tex_blue") \
		+ ff.tri_list([tuple(i + 8 for i in f) for f in nose_f])
	if two_bone:
		polys += ff.set_texture("tex_red") + ff.tri_list([(16, 0, 1)])
	chunks = [
		ff.chunk("red ", ffxi.T_TEXTURE, ff.dxt_texture("tex_red", 8, 8, (255, 0, 0))),
		ff.chunk("blue", ffxi.T_TEXTURE, ff.dxt_texture("tex_blue", 4, 4, (0, 0, 255), alpha_nibble=8)),
		ff.chunk("skel", ffxi.T_SKELETON, ff.skeleton([(255, QI, (0, 0, 0)), (0, QI, (0, -1.0, 0))])),
		ff.chunk("body", ffxi.T_MESH, ff.mesh(one, assign, polys, two=two, flip=flip)),
	]
	if anim:	# head turns 90° about FFXI y over 3 frames, root bobs 0.5 up
		rot = [QI, (0, math.sin(math.pi / 8), 0, math.cos(math.pi / 8)), (0, S, 0, S)]
		chunks += [
			ff.chunk("idl0", ffxi.T_MOTION, ff.motion(3, 0.5, [{"bone": 1, "rot": rot}])),
			ff.chunk("idl1", ffxi.T_MOTION, ff.motion(3, 0.5, [{"bone": 0, "trans": [(0, 0, 0), (0, -0.25, 0), (0, -0.5, 0)]}])),
			ff.chunk("wlk0", ffxi.T_MOTION, ff.motion(2, 1.0, [{"bone": 0, "skip": True}, {"bone": 1, "rot": [QI, QI]}])),
		]
	return ff.dat(*chunks)


def test_chunk_chain_and_census(tmp_path):
	p = ff.write(tmp_path / "ROM" / "27" / "82.DAT", _npc())
	chunks = ffxi.read_chunks(p.read_bytes())
	assert [c.type for c in chunks] == [0x20, 0x20, 0x29, 0x2A, 0x2B, 0x2B, 0x2B]
	assert ffxi.chunk_census(p) == {0x20: 2, 0x29: 1, 0x2A: 1, 0x2B: 3}
	assert ffxi.is_model_dat(p)
	assert detect_kind(p) == Kind.MODEL
	assert ffxi.dat_stem(p) == "ffxi_rom_27_82"
	junk = ff.write(tmp_path / "junk.dat", b"hello world, not a chunk file at all....")
	assert not ffxi.is_model_dat(junk)
	assert detect_kind(junk) == Kind.UNKNOWN
	tex_only = ff.write(tmp_path / "t.dat", ff.chunk("red ", 0x20, ff.dxt_texture("x", 4, 4)))
	assert ffxi.chunk_census(tex_only) == {0x20: 1} and not ffxi.is_model_dat(tex_only)
	with pytest.raises(ValueError):
		ffxi.read_chunks(b"\xff\xfe\x00\x01" + b"\0" * 28)


def test_rom_tables_resolve_ids(tmp_path):
	root = tmp_path / "FINAL FANTASY XI"
	vt = bytearray(10)
	ft = bytearray(20)
	vt[3] = 1
	struct.pack_into("<H", ft, 6, (27 << 7) | 82)
	ff.write(root / "VTABLE.DAT", bytes(vt))
	ff.write(root / "FTABLE.DAT", bytes(ft))
	vt2, ft2 = bytearray(10), bytearray(20)
	vt2[5] = 2
	struct.pack_into("<H", ft2, 10, (4 << 7) | 1)
	ff.write(root / "ROM2" / "VTABLE2.DAT", bytes(vt2))
	ff.write(root / "ROM2" / "FTABLE2.DAT", bytes(ft2))
	ff.write(root / "ROM" / "27" / "82.DAT", _npc())
	assert ffxi.Rom.is_root(root)
	rom = ffxi.Rom(root)
	assert rom.locate(3) == (1, 27, 82)
	assert rom.locate(5) == (2, 4, 1)
	assert rom.locate(4) is None
	assert rom.path(3) == root / "ROM" / "27" / "82.DAT"
	assert rom.path(5).as_posix().endswith("ROM2/4/1.DAT")
	assert rom.count() == 2


def test_textures_dxt1_dxt3_and_palette():
	t = ffxi.read_texture(ff.dxt_texture("tex_red", 8, 4, (255, 0, 0)))
	assert t.name == "tex_red" and t.format == "DXT1" and t.rgba.shape == (4, 8, 4)
	assert tuple(t.rgba[2, 5]) == (255, 0, 0, 255)
	half = ffxi.read_texture(ff.dxt_texture("a", 4, 4, (0, 255, 0), alpha_nibble=2))
	full = ffxi.read_texture(ff.dxt_texture("a", 4, 4, (0, 255, 0), alpha_nibble=8))
	assert half.format == "DXT3" and half.rgba[0, 0, 3] == 64 and full.rgba[0, 0, 3] == 255
	px = np.array([[0, 1], [1, 0]])
	for at in (61, 64):	# both palette offsets in circulation are recognised
		p = ffxi.read_texture(ff.palette_texture("pal", 2, 2, [(10, 20, 30, 0x80), (200, 100, 50, 0x20)], px, at))
		assert tuple(p.rgba[0, 0]) == (10, 20, 30, 255), at
		assert tuple(p.rgba[0, 1]) == (200, 100, 50, 64), at
	with pytest.raises(ValueError):
		ffxi.read_texture(b"\x55" + b"\0" * 80)


def test_mesh_lists_strips_and_texture_changes():
	v, f = ff.box(0, 0, 0)
	polys = ff.set_texture("a") + ff.tri_list(f[:2]) + ff.set_texture("b") + ff.tri_strip([0, 1, 2, 3, 4])
	m = ffxi.read_mesh(ff.mesh(v, [(ff.bone(0), 0)] * 8, polys))
	assert len(m.pos1) == 8 and m.textures == ["a", "a", "b", "b", "b"]
	assert m.faces[2:].tolist() == [[0, 1, 2], [1, 3, 2], [2, 3, 4]]	# odd strip triangles swap
	cloth = ffxi.read_mesh(ff.mesh([x[:3] for x in v], [(0, 0)] * 8, ff.tri_list(f), cloth=True))
	assert cloth.cloth and cloth.normals is None and len(cloth.faces) == 12


def test_npc_axes_winding_textures_and_bones(tmp_path):
	p = ff.write(tmp_path / "npc.dat", _npc(anim=False))
	opts = models.ModelOptions(ffxi_scale=10.0)
	m = ffxi.import_ffxi(p, opts)
	assert len(m.bones) == 2 and m.bones[1].parent == 0
	# the head bone sits one unit up: FFXI -y → GoldSrc +Z
	assert np.allclose(m.bones[1].pose[:3], (0, 0, 10.0), atol=1e-5)
	red = [s for s in m.meshes if m.materials[s.material].name == "tex_red"][0]
	blue = [s for s in m.meshes if m.materials[s.material].name == "tex_blue"][0]
	assert np.allclose(red.world.mean(0), (0, 0, 10.0), atol=1e-4)
	assert set(red.bones.tolist()) == {1}
	assert np.allclose(red.positions.mean(0), 0, atol=1e-4)	# bone-local
	assert blue.world.mean(0)[0] == pytest.approx(5.0, abs=1e-4)	# +x stays forward
	assert tuple(m.materials[red.material].rgba[0, 0, :3]) == (255, 0, 0)
	# D3D-style clockwise cubes come out counter-clockwise from outside
	for s in (red, blue):
		t = s.world[s.tris]
		n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
		out = t.mean(1) - s.world.mean(0)
		assert (np.einsum("ni,ni->n", n, out) > 0).all()


def test_yaw_and_two_bone_vertices(tmp_path):
	p = ff.write(tmp_path / "npc.dat", _npc(two_bone=True, anim=False))
	m = ffxi.import_ffxi(p, models.ModelOptions(ffxi_scale=1.0, ffxi_yaw=90.0))
	blue = [s for s in m.meshes if m.materials[s.material].name == "tex_blue"][0]
	assert np.allclose(blue.world.mean(0)[:2], (0, 0.5), atol=1e-4)	# +x turned to +y
	red = [s for s in m.meshes if m.materials[s.material].name == "tex_red"]
	shared = np.concatenate([s.world for s in red])
	# 0.3·(root·(0,-0.3,0)) + 0.7·(head·(0,0.7,0)) = y -0.09 + 0.7·(-1 + 0.7) = -0.3 → Z 0.3
	assert np.isclose(shared[:, 2], 0.3, atol=1e-4).any()
	assert any("heavier bone" in w for w in m.warnings)


def test_flip_draws_the_mirror_half_on_the_mirror_bone(tmp_path):
	skel = ff.skeleton([(255, QI, (0, 0, 0)), (0, QI, (0, 0, 0.5)), (0, QI, (0, 0, -0.5))])
	v, f = ff.box(0.2, 0, 0)
	data = ff.dat(ff.chunk("skel", 0x29, skel),
		ff.chunk("arm ", 0x2A, ff.mesh(v, [(ff.bone(1, 2, mirror=1), 0)] * 8, ff.tri_list(f), flip=True)))
	m = ffxi.import_ffxi(ff.write(tmp_path / "a.dat", data), models.ModelOptions(ffxi_scale=1.0))
	assert len(m.meshes) == 2
	a, b = m.meshes
	assert set(a.bones.tolist()) == {1} and set(b.bones.tolist()) == {2}
	# bone 1 is at FFXI z +0.5 (GoldSrc +Y), its mirror at -0.5; the mirror flips local x
	assert np.allclose(a.world.mean(0), (0.2, 0.5, 0), atol=1e-4)
	assert np.allclose(b.world.mean(0), (-0.2, -0.5, 0), atol=1e-4)
	for s in (a, b):
		t = s.world[s.tris]
		n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
		assert (np.einsum("ni,ni->n", n, t.mean(1) - s.world.mean(0)) > 0).all()


def test_motions_group_and_bake(tmp_path):
	p = ff.write(tmp_path / "npc.dat", _npc())
	m = ffxi.import_ffxi(p, models.ModelOptions(ffxi_scale=1.0, anim_fps=15))
	assert [s.name for s in m.sequences] == ["idl", "wlk"]
	idl = m.sequences[0]
	# 3 keys at 30·0.5 = 15 fps: two intervals, so three frames at 15 fps
	assert idl.frames.shape == (3, 2, 6)
	assert idl.frames[2, 0, 2] == pytest.approx(0.5, abs=1e-5)	# root rose half a unit (FFXI -y)
	parents = [b.parent for b in m.bones]
	w = sk.world_matrices(idl.frames[2], parents)
	# a quarter turn about FFXI's y axis (down) is a quarter turn about GoldSrc -Z
	expect = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
	assert np.allclose(w[1][:3, :3], expect, atol=1e-5)
	assert np.allclose(w[1][:3, 3], (0, 0, 1.5), atol=1e-5)


def test_studio_mdl_decodes(tmp_path):
	p = ff.write(tmp_path / "npc.dat", _npc())
	m = ffxi.import_ffxi(p)
	mdl = models.build_studio_mdl(m)
	info = studio.studio_info(mdl)
	assert info and mdl[:4] == b"IDST"
	assert studio.lean_studio_mesh_bytes_from(mdl)


def test_equipment_without_skeleton_warns_and_manifest_supplies_it(tmp_path):
	v, f = ff.box(0, 0, 0)
	part = ff.dat(ff.chunk("tex ", 0x20, ff.dxt_texture("t", 4, 4)),
		ff.chunk("legs", 0x2A, ff.mesh(v, [(ff.bone(1), 0)] * 8, ff.set_texture("t") + ff.tri_list(f))))
	skel = ff.dat(ff.chunk("skel", 0x29, ff.skeleton([(255, QI, (0, 0, 0)), (0, QI, (0, -2.0, 0))])))
	mot = ff.dat(ff.chunk("run0", 0x2B, ff.motion(2, 1.0, [{"bone": 1, "trans": [(0, 0, 0), (0, -1, 0)]}])))
	pp = ff.write(tmp_path / "ROM" / "28" / "53.DAT", part)
	ff.write(tmp_path / "ROM" / "27" / "82.DAT", skel)
	ff.write(tmp_path / "ROM" / "32" / "13.DAT", mot)
	alone = ffxi.import_ffxi(pp)
	assert any("no skeleton" in w for w in alone.warnings)
	man = tmp_path / "hume.ffximdl"
	man.write_text(json.dumps({"skeleton": "ROM/27/82.DAT", "parts": ["ROM/28/53.DAT"], "motions": ["ROM/32/13.DAT"]}))
	m = ffxi.import_ffxi(man, models.ModelOptions(ffxi_scale=1.0))
	assert m.name == "hume" and len(m.bones) == 2
	assert np.allclose(m.meshes[0].world.mean(0), (0, 0, 2.0), atol=1e-4)
	assert [s.name for s in m.sequences] == ["run"]
	# the same through the option, as the CLI passes it
	m2 = ffxi.import_ffxi(pp, models.ModelOptions(ffxi_scale=1.0, ffxi_skeleton=str(tmp_path / "ROM/27/82.DAT")))
	assert np.allclose(m2.meshes[0].world.mean(0), (0, 0, 2.0), atol=1e-4)


# -- sounds ---------------------------------------------------------------------


def _block(header: int, nibbles: list[int]) -> bytes:
	out = bytearray([header])
	for i in range(0, len(nibbles), 2):
		out.append((nibbles[i] & 15) | ((nibbles[i + 1] & 15) << 4))
	return bytes(out)


def test_adpcm_decoder():
	# filter 0, scale nibble 0 → shift 12: nibble n is n·4096
	b0 = _block(0x00, [1, -1, 7, -8] + [0] * 12)
	# filter 1 (0xF0/256 of the last sample), shift 12 - 4 = 8
	b1 = _block(0x14, [1] + [0] * 15)
	info, pcm = ffxi.decode_sound(ff.spw([b0, b1]))
	assert info.kind == "effect" and info.rate == 22050 and info.channels == 1
	assert pcm[:4, 0].tolist() == [4096, -4096, 28672, -32768]
	assert pcm[4, 0] == 0	# filter 0 ignores history
	prev = 0
	expect = []
	for k in range(16):
		x = (1 << 8 if k == 0 else 0) + int(prev * 0xF0 / 256)
		expect.append(x)
		prev = x
	assert pcm[16:, 0].tolist() == expect


def test_sound_to_wav_and_stereo_music(tmp_path):
	data = ff.spw([_block(0x00, [2] * 16)] * 4, loop=1)
	wav, info = ffxi.sound_to_wav(data)
	snd = sfx.read_wav(wav)
	assert snd.rate == 22050 and snd.frames == 64 and snd.loop_start == 16
	left = _block(0x00, [1] * 16)
	right = _block(0x00, [-1] * 16)
	info, pcm = ffxi.decode_sound(ff.bgw([left + right] * 3))
	assert info.kind == "music" and pcm.shape == (48, 2)
	assert (pcm[:, 0] == 4096).all() and (pcm[:, 1] == -4096).all()


def test_project_converts_models_and_sounds(tmp_path):
	src = tmp_path / "ffxi"
	ff.write(src / "ROM" / "27" / "82.DAT", _npc())
	ff.write(src / "ROM" / "27" / "83.DAT", ff.chunk("text", 0x05, b"not a model"))
	ff.write(src / "sound" / "se001.spw", ff.spw([_block(0x00, [3] * 16)] * 8))
	ff.write(src / "music" / "music001.bgw", ff.bgw([_block(0, [1] * 16) * 2] * 8))
	proj = Project(Settings())
	added = proj.add_path(src)
	kinds = sorted((a.kind.name, a.stem) for a in added)
	assert kinds == [("AUDIO", "music001"), ("AUDIO", "se001"), ("MODEL", "ffxi_rom_27_82")]
	out = tmp_path / "out"
	proj.convert(added, out)
	assert all(a.status.startswith("Done") for a in added), [(a.name, a.status, a.warnings) for a in added]
	assert (out / "models" / "ffxi_rom_27_82.mdl").is_file()
	assert (out / "gc_studio" / "ffxi_rom_27_82.mdl").is_file()
	assert (out / "sound" / "ffxi" / "se001.wav").is_file()
	assert (out / "media" / "music001.gcpcm").is_file()
	info = proj.info(added[-1] if added[-1].kind == Kind.MODEL else [a for a in added if a.kind == Kind.MODEL][0])
	assert info["Source"] == "Final Fantasy XI DAT" and info["Bones"] == 2


def test_cli_describes_and_extracts(tmp_path, capsys):
	p = ff.write(tmp_path / "ROM" / "27" / "82.DAT", _npc())
	assert cli.main(["ffxi", str(p), "--textures", str(tmp_path / "png")]) == 0
	out = capsys.readouterr().out
	assert "texture 'tex_red': DXT1, 8x8" in out and "skeleton: 2 bones" in out and "motion idl: 2 chunk(s)" in out
	assert (tmp_path / "png" / "tex_red.png").is_file()
	s = ff.write(tmp_path / "se.spw", ff.spw([_block(0, [1] * 16)]))
	assert cli.main(["ffxi", str(s), "--wav", str(tmp_path / "se.wav")]) == 0
	assert sfx.read_wav((tmp_path / "se.wav").read_bytes()).frames == 16
	assert cli.main(["convert", str(p), "-o", str(tmp_path / "out"), "--ffxi-scale", "20"]) == 0
	assert (tmp_path / "out" / "gc_studio" / "ffxi_rom_27_82.mdl").is_file()


def _zone_dat(tmp_path):
	"""A tree prefab (a textured quad strip + a blank-named quad), a lone rock prefab, the sun (skipped),
	and placements: two trees, one turned 90° and doubled in size, and a name with no prefab."""
	pos, nrm, uv, idx = ff.quad_strip(1.0)
	tree = ff.mmb("tree", [[("bark", pos, nrm, uv, idx)], [("", pos, nrm, uv, [0, 1, 2, 3])]])
	rock = ff.mmb("rock", [[("bark", pos, nrm, uv, [0, 1, 2, 1, 3, 2])]], tri_list=True, seed=0x11)
	sun = ff.mmb("sunsphere", [[("bark", pos, nrm, uv, idx)]])
	placements = ff.mzb([("tree", (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
		("tree", (0.0, -5.0, 0.0), (0.0, np.pi / 2, 0.0), (2.0, 2.0, 2.0)),
		("ghost", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
	data = ff.dat(ff.chunk("bark", 0x20, ff.dxt_texture("bark", 4, 4, (0, 255, 0))),
		ff.chunk("tree", 0x2E, tree), ff.chunk("rock", 0x2E, rock), ff.chunk("sun", 0x2E, sun),
		ff.chunk("zone", 0x1C, placements))
	p = tmp_path / "ROM" / "1" / "20.DAT"
	p.parent.mkdir(parents=True)
	p.write_bytes(data)
	return p


def test_zone_unscramble_and_read(tmp_path):
	p = _zone_dat(tmp_path)
	assert ffxi.is_zone_dat(p) and not ffxi.is_model_dat(p)
	zone = ffxi.read_zone(p)
	assert sorted(zone.prefabs) == ["rock", "tree"]	# the sun is left to the sky
	assert [len(m.tris) for m in zone.prefabs["tree"]] == [2, 2]
	assert len(zone.prefabs["rock"][0].tris) == 2	# a triangle list
	assert zone.prefabs["tree"][1].texture == "bark"	# blank → the zone's main texture
	assert [i.name for i in zone.instances] == ["tree", "tree", "ghost"] and len(zone.placed()) == 2
	m = zone.instances[1].matrix
	assert np.allclose(m[:3, :3] @ [1, 0, 0], [0, 0, -2], atol=1e-6) and np.allclose(m[:3, 3], [0, -5, 0])
	info = ffxi.describe(p)["zone"]
	assert info["placed triangles"] == 8 and info["unmatched names"] == ["ghost"]


def test_zone_to_obj(tmp_path):
	p = _zone_dat(tmp_path)
	files = ffxi.zone_to_obj(p, tmp_path / "out", scale=10.0)
	obj = next(f for f in files if f.suffix == ".obj")
	assert obj.name == "ffxi_rom_1_20.obj" and (tmp_path / "out" / "textures" / "bark.png").is_file()
	v, faces = [], []
	for line in obj.read_text().splitlines():
		if line.startswith("v "):
			v.append([float(x) for x in line.split()[1:]])
		elif line.startswith("f "):
			faces.append([int(c.split("/")[0]) - 1 for c in line.split()[1:]])
	v, faces = np.array(v), np.array(faces)
	assert len(faces) == 8
	# FFXI x → GoldSrc x, FFXI -y (up) → GoldSrc z: the second tree stands 5 units up, scaled by 10
	assert np.isclose(v[:, 2].max(), 50.0) and np.isclose(v[:, 0].max(), 110.0)
	t = v[faces]
	n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
	assert (n[:, 2] > 0).all()	# every quad faces up, whatever way the strips wound it
	assert "map_Kd textures/bark.png" in (tmp_path / "out" / "ffxi_rom_1_20.mtl").read_text()
	props = ffxi.zone_to_obj(p, tmp_path / "props", prefabs=True)
	assert sorted(f.name for f in props if f.suffix == ".obj") == ["rock.obj", "tree.obj"]


def test_zone_cli(tmp_path, capsys):
	p = _zone_dat(tmp_path)
	assert cli.main(["ffxi", str(p), "--obj", str(tmp_path / "o")]) == 0
	out = capsys.readouterr().out
	assert "zone: 2 prefabs" in out and "2 of 3 placements" in out and "ghost" in out and "wrote 1 OBJ" in out
