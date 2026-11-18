"""Source 2 resources: KV3, the block container, textures, materials, sounds, models and animations."""

import math
import struct
import sys

import numpy as np
import pytest

from gxbake.core import kv3, kv3text, models, skeleton, studio, valvecomp, vanim, vmdl, vres, vsnd, vtex
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import s2_fixtures as s2


def _pattern(w=16, h=8):
	y, x = np.mgrid[0:h, 0:w]
	img = np.zeros((h, w, 4), np.uint8)
	img[..., 0] = x * 255 // (w - 1)
	img[..., 1] = y * 255 // (h - 1)
	img[..., 2] = 60
	img[..., 3] = 255
	return img


# -- KV3 and the container ------------------------------------------------------------


def test_kv3_binary_v1_and_legacy():
	doc = {"name": "x", "n": -3, "f": 0.25, "flags": [True, False, None], "blob": b"\x00\x01\xff",
		"ints": np.array([4, 5, 6], np.int32), "vec": np.array([1.0, 2.0, 3.0], np.float32), "sub": {"k": "v"}}
	got = kv3.read(s2.kv3(doc))
	assert got["name"] == "x" and got["n"] == -3 and got["f"] == 0.25
	assert got["flags"] == [True, False, None] and got["blob"] == b"\x00\x01\xff"
	assert list(got["ints"]) == [4, 5, 6] and np.allclose(got["vec"], [1, 2, 3])
	assert got["sub"] == {"k": "v"}
	old = kv3.read(s2.kv3_legacy(doc))
	assert old["name"] == "x" and old["ints"] == [4, 5, 6] and old["sub"] == {"k": "v"}


def test_kv3_text():
	text = '<!-- kv3 encoding:text:version{e21c7f3c-8a33-41c5-9977-a76d3a32aa0d} -->\n' \
		'{ a = 1 b = "two" c = [ 1.5, true, null ] d = resource:"x.vmat" e = #[01 ff] // note\n f = """multi\nline""" }'
	got = kv3text.parse(text)
	assert got == {"a": 1, "b": "two", "c": [1.5, True, None], "d": "x.vmat", "e": b"\x01\xff", "f": "multi\nline"}


def test_resource_blocks_and_kinds():
	raw = s2.resource([("DATA", s2.kv3({"a": 1})), ("CTRL", s2.kv3({"b": 2}))], version=3)
	assert vres.looks_like_resource(raw)
	res = vres.parse(raw, "x/y.vmdl_c")
	assert res.version == 3 and [b.type for b in res.blocks] == ["DATA", "CTRL"]
	assert res.keyvalues() == {"a": 1} and res.keyvalues("CTRL") == {"b": 2} and res.keyvalues(1) == {"b": 2}
	assert res.kind == "model" and vres.resource_kind("a.vsnd_c") == "sound" and vres.resource_kind("a.png") is None
	with pytest.raises(ValueError):
		vres.parse(b"\0" * 32)


def test_lz4_block_literals_and_match(monkeypatch):
	monkeypatch.setitem(sys.modules, "lz4.block", None)	# the pure-Python decoder
	# literal "abc", then a 9-byte match at offset 3, then the literal "x"
	block = bytes([0x35]) + b"abc" + b"\x03\x00" + bytes([0x10]) + b"x"
	assert valvecomp.lz4_block(block, 13) == b"abcabcabcabcx"


def test_meshopt_codecs_match_reference():
	mo = pytest.importorskip("meshoptimizer")
	rng = np.random.default_rng(1)
	verts = rng.integers(0, 255, (64, 16), dtype=np.uint8)
	enc = mo.encode_vertex_buffer(verts, 64, 16)
	assert valvecomp.meshopt_vertex(64, 16, bytes(enc)) == verts.tobytes()
	idx = np.array([0, 1, 2, 2, 1, 3, 3, 1, 4, 4, 5, 6] * 4, np.uint32)
	enc = mo.encode_index_buffer(idx, len(idx), 4)
	ref = np.frombuffer(bytes(mo.decode_index_buffer(len(idx), 4, enc)), "<u4")
	got = np.frombuffer(valvecomp.meshopt_index(len(idx), 4, bytes(enc)), "<u4")
	assert list(got) == list(ref)


# -- textures, materials, sounds ----------------------------------------------------


def test_vtex_rgba_and_project(tmp_path):
	src = _pattern()
	p = tmp_path / "materials" / "wall.vtex_c"
	p.parent.mkdir()
	p.write_bytes(s2.vtex_rgba(src))
	tex = vtex.load(p)
	assert tex.size == (16, 8) and tex.format_name == "RGBA8888" and not tex.cube
	assert (tex.rgba() == src).all()
	assert detect_kind(p) == Kind.IMAGE
	proj = Project(Settings())
	proj.add_path(p)
	asset = proj.assets[0]
	assert proj.info(asset)["Format"] == "RGBA8888"
	out = proj.convert([asset], tmp_path / "out", log=lambda _m: None)
	assert any(o.rel == "media/textures.gct" for o in out)


def test_vmat_colour_texture(tmp_path):
	p = tmp_path / "materials" / "test.vmat_c"
	p.parent.mkdir()
	p.write_bytes(s2.vmat("materials/wall.vtex", alpha_test=True))
	m = vtex.load_vmat(p)
	assert m.color_texture() == "materials/wall.vtex" and m.alpha_test and not m.translucent
	assert vtex.find_compiled("materials/test.vmat", [tmp_path]) == p
	assert vtex.find_compiled("MATERIALS/Test.vmat", [tmp_path]) == p	# case-insensitive fallback


def test_vsnd_pcm16_to_port_wav(tmp_path):
	t = np.arange(4410)
	samples = (np.sin(t * 2 * math.pi * 440 / 44100) * 12000).astype(np.int16)
	p = tmp_path / "sounds" / "tone.vsnd_c"
	p.parent.mkdir()
	p.write_bytes(s2.vsnd_pcm16(samples, 44100, loop_start=100, loop_end=4000))
	snd = vsnd.load(p)
	assert (snd.kind, snd.rate, snd.channels, snd.samples) == ("pcm", 44100, 1, 4410)
	wav = vsnd.to_wav(snd)
	assert wav[:4] == b"RIFF" and np.array_equal(np.frombuffer(wav[44:], "<i2"), samples)
	from gxbake.core import sfx

	port = sfx.read_wav(vsnd.to_port_wav(snd))
	assert port.rate == 22050 and port.loop_start == 50 and abs(port.frames - 2205) <= 1
	assert detect_kind(p) == Kind.AUDIO
	proj = Project(Settings())
	proj.add_path(p)
	asset = proj.assets[0]
	out = proj.convert([asset], tmp_path / "out", log=lambda _m: None)
	assert [o.rel for o in out] == ["sound/tone.wav"]


def test_ms_adpcm_decoder_first_samples():
	# one mono block: predictor 0, delta 16, samples 100 then 200, nibbles all zero
	blk = bytes([0]) + struct.pack("<hhh", 16, 200, 100) + bytes(4)
	pcm = vsnd.decode_ms_adpcm(blk, 1, len(blk))
	assert pcm[:2, 0].tolist() == [100, 200] and len(pcm) == 10


# -- models and animation -------------------------------------------------------------


def test_compressed_quaternion_round_trip():
	for q in [(0.0, 0.0, 0.0, 1.0), (0.3826834, 0.0, 0.0, 0.9238795), (-0.2, 0.1, -0.3, 0.9273618)]:
		got = vanim.read_quaternion(s2.pack_quat(q))
		assert np.allclose(got, q, atol=1e-3)
	# both sign flags: the stored words are (x, y, z) of (y, z, w, x) order, so w lands third
	words = struct.pack("<HHH", s2.quat_word(0.1) | 0x8000, s2.quat_word(0.2) | 0x8000, s2.quat_word(0.3))
	x, y, z, w = vanim.read_quaternion(words)
	assert np.allclose((x, y, w), (0.2, 0.3, 0.1), atol=1e-3) and abs(z - math.sqrt(1 - 0.14)) < 1e-3


def _arm(tmp_path):
	root = tmp_path / "game"
	p = s2.arm_vmdl(root)
	(root / "materials").mkdir()
	(root / "materials" / "test.vmat_c").write_bytes(s2.vmat("materials/skin.vtex"))
	(root / "materials" / "skin.vtex_c").write_bytes(s2.vtex_rgba(_pattern(8, 8)))
	return p


def test_vmdl_import_skeleton_mesh_and_material(tmp_path):
	p = _arm(tmp_path)
	m = vmdl.import_vmdl(p)
	assert [b.name for b in m.bones] == ["root", "arm"] and m.bones[1].parent == 0
	assert m.num_vertices == 8 and m.num_triangles == 4
	assert len(m.materials) == 1 and m.materials[0].texture.endswith("skin.vtex_c")
	verts = m.meshes[0]
	upper = verts.bones == 1
	assert upper.sum() == 4
	# vertices are stored in their bone's space: the arm's quad starts 1 unit above its joint
	assert np.allclose(sorted(set(np.round(verts.positions[upper][:, 2], 3))), [1.0, 10.0])
	assert not any("not found" in w for w in m.warnings)


def test_vmdl_animation_becomes_sequence(tmp_path):
	p = _arm(tmp_path)
	m = vmdl.import_vmdl(p)
	names = [s.name for s in m.sequences]
	assert names == ["bend"]	# the delta animation is skipped and the '@' prefix dropped
	assert any("delta" in w for w in m.warnings)
	sq = m.sequences[0]
	assert sq.loop and sq.fps == 15.0 and len(sq.frames) == 2	# 30 fps, 3 frames, resampled to 15
	first, last = (skeleton.matrix_from_pose(f[1]) for f in (sq.frames[0], sq.frames[-1]))
	assert np.allclose(first[:3, 3], [0, 0, 10]) and np.allclose(last[:3, 3], [0, 0, 10])
	assert np.allclose(first[:3, :3], np.eye(3), atol=1e-3)
	assert np.allclose(last[:3, :3], [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-2)	# 90° about +X


def test_vmdl_static_option_and_info(tmp_path):
	p = _arm(tmp_path)
	m = vmdl.import_vmdl(p, models.ModelOptions(static=True))
	assert len(m.bones) == 1 and m.sequences[0].name == "idle"
	info = vmdl.vmdl_info(p)
	assert info["Bones"] == 2 and info["Meshes"] == 1 and "CTRL" in info["Blocks"]


def test_project_converts_vmdl(tmp_path):
	p = _arm(tmp_path)
	assert detect_kind(p) == Kind.MODEL and models.native_format(p) == "source2"
	proj = Project(Settings())
	proj.add_path(p)
	asset = proj.assets[0]
	info = proj.info(asset)
	assert info["Sequences"].startswith("bend")
	out = {o.rel for o in proj.convert([asset], tmp_path / "out", log=lambda _m: None)}
	assert {"models/arm.mdl", "gc_studio/arm.mdl", "gc_studio/armt.mdl"} <= out
	raw = (tmp_path / "out" / "models" / "arm.mdl").read_bytes()
	assert studio.studio_info(raw)["sequences"] == 1 and studio.studio_info(raw)["bones"] == 2


def test_lone_vmesh(tmp_path):
	# a .vmesh_c keeps its draw calls in DATA and its buffers in a VBIB-less KV3
	root = tmp_path / "game"
	p = s2.arm_vmdl(root)
	res = vres.load(p)
	ctrl = res.keyvalues("CTRL")
	mesh = dict(res.keyvalues(2))
	mesh["m_vertexBuffers"] = ctrl["embedded_meshes"][0]["m_vertexBuffers"]
	mesh["m_indexBuffers"] = ctrl["embedded_meshes"][0]["m_indexBuffers"]
	vm = root / "models" / "arm_lod0.vmesh_c"
	vm.write_bytes(s2.resource([("DATA", s2.kv3(mesh))]))
	m = vmdl.import_vmdl(vm)
	assert len(m.bones) == 1 and m.num_triangles == 4
