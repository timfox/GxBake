"""DMX model sources and particle systems (.pcf, .vpcf_c), on generated files."""

import struct

import numpy as np
import pytest

from gxbake.core import dmx, dmx_model, models, particles, sprite
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import dmx_fixtures as fx
from . import s2_fixtures as s2
from .game_fixtures import vtf


@pytest.mark.parametrize("encode", [fx.binary, fx.kv2])
def test_read_both_encodings(tmp_path, encode):
	p = tmp_path / "arm.dmx"
	p.write_bytes(encode(fx.arm_model()))
	dm = dmx.load(p)
	assert (dm.format, dm.format_version) == ("model", 18) and dm.root.type == "DmElement"
	model = dm.root["model"]
	assert model is dm.root["skeleton"] and [j.name for j in model["jointList"]] == ["lower", "upper"]
	vd = dm.of_type("DmeVertexData")[0]
	assert vd["vertexFormat"][0] == "positions" and vd["positions"][8] == (-8.0, -8.0, 64.0)
	clip = dm.of_type("DmeChannelsClip")[0]
	assert clip["timeFrame"]["duration"] == pytest.approx(1.0)
	assert clip["channels"][0]["toElement"] is model["jointList"][1]["transform"]


def test_dmx_model_bones_skin_and_animation(tmp_path):
	p = tmp_path / "arm.dmx"
	p.write_bytes(fx.binary(fx.arm_model()))
	assert models.native_format(p) == "dmx" and detect_kind(p) == Kind.MODEL
	m = dmx_model.import_dmx(p)
	assert [(b.name, b.parent) for b in m.bones] == [("lower", -1), ("upper", 0)]
	assert np.allclose(m.bones[1].pose, [0, 0, 32, 0, 0, 0])
	assert m.num_triangles == 16 and m.meshes[0].bones.tolist() == [0] * 8 + [1] * 4
	# the top ring sits at the upper joint's height in its own space; uvs flip to a top-left origin
	assert np.allclose(m.meshes[0].positions[8:, 2], 32) and np.allclose(m.meshes[0].world[8:, 2], 64)
	assert np.allclose(m.meshes[0].uvs[8:, 1], 1.0)
	seq = m.sequences[0]
	assert seq.name == "wave" and np.allclose(seq.frames[-1, 1, 3], np.pi / 2, atol=1e-3)
	raw = models.build_studio_mdl(m)
	end = models.studio_posed_vertices(raw, 0, len(seq.frames) - 1)
	assert end[:, 1].min() == pytest.approx(-32, abs=0.5) and end[:, 2].max() == pytest.approx(40, abs=0.5)


def test_dmx_y_up_and_textures_beside_it(tmp_path):
	from PIL import Image

	p = tmp_path / "arm.dmx"
	p.write_bytes(fx.kv2(fx.arm_model(animated=False, up_axis=2)))
	Image.fromarray(np.full((16, 16, 4), (200, 20, 20, 255), np.uint8)).save(tmp_path / "arm.png")
	m = dmx_model.import_dmx(p)
	assert m.warnings == [] and m.materials[0].texture == "arm.png"
	lo, hi = m.bounds()
	# the column was 64 tall along Y-up's Y... here along source Z, which Y-up maps to GoldSrc +X
	assert hi[0] - lo[0] == pytest.approx(64)


def _material_tree(root):
	d = root / "materials" / "particle" / "test"
	d.mkdir(parents=True)
	img = np.zeros((32, 64, 4), np.uint8)
	img[8:24, 8:24] = (255, 160, 40, 255)	# frame 1: a square
	img[4:28, 36:60] = (40, 160, 255, 255)	# frame 2: a bigger square
	(d / "spark.vtf").write_bytes(fx.vtf_with_sheet(img, [(0, 0, 0.5, 1), (0.5, 0, 1, 1)]))
	(d / "spark.vmt").write_text('"SpriteCard"\n{\n\t"$basetexture" "particle/test/spark"\n\t"$additive" "1"\n}\n')
	y, x = np.mgrid[0:32, 0:32]
	smoke = np.zeros((32, 32, 4), np.uint8)
	smoke[..., :3] = 128
	smoke[..., 3] = np.clip(255 - np.hypot(x - 16, y - 16) * 16, 0, 255)
	(d / "smoke.vtf").write_bytes(vtf(smoke))
	(d / "smoke.vmt").write_text('"SpriteCard"\n{\n\t"$basetexture" "particle/test/smoke"\n}\n')


def test_pcf_systems_become_sprites(tmp_path):
	_material_tree(tmp_path)
	p = tmp_path / "particles" / "fx.pcf"
	p.parent.mkdir()
	p.write_bytes(fx.binary(fx.particles(), "pcf", 2))
	assert particles.is_particles(p) and detect_kind(p) == Kind.SPR
	sys_list = {s.name: s for s in particles.systems(p)}
	assert sys_list["burst"].children == ["sparks"] and not sys_list["burst"].renders
	assert sys_list["sparks"].renderers == ["render_animated_sprites"] and sys_list["sparks"].colour[:3] == (255, 200, 100)
	warnings = []
	out = particles.convert(p, warnings=warnings)
	assert list(out) == ["sparks"] and warnings == []
	raw = out["sparks"]
	typ, fmt = struct.unpack_from("<ii", raw, 8)
	assert raw[:4] == b"IDSP" and typ == particles.SPR_VP_PARALLEL and fmt == particles.SPR_ADDITIVE
	frames = sprite.spr_frames(raw)
	assert len(frames) == 2 and frames[0].shape[:2] == (32, 32)
	# the sheet's first frame is the orange square, the second the blue one
	assert frames[0][16, 16, 0] > 200 and frames[1][16, 16, 2] > 200 and frames[0][2, 2, :3].max() < 10


def test_soft_alpha_becomes_index_alpha(tmp_path):
	_material_tree(tmp_path)
	p = tmp_path / "smoke.pcf"
	p.write_bytes(fx.kv2(fx.particles("particle/test/smoke"), "pcf", 2))
	raw = particles.convert(p)["sparks"]
	assert struct.unpack_from("<i", raw, 12)[0] == particles.SPR_INDEXALPHA
	w, h = struct.unpack_from("<ii", raw, 20)
	pix = np.frombuffer(raw, np.uint8, w * h, len(raw) - w * h).reshape(h, w)
	assert pix[16, 16] > 240 and pix[0, 0] == 0	# alpha in the index


def test_vpcf_and_project(tmp_path):
	tex = tmp_path / "materials" / "particle" / "glow.vtex_c"
	tex.parent.mkdir(parents=True)
	tex.write_bytes(s2.vtex_rgba(np.full((16, 16, 4), (250, 250, 100, 255), np.uint8)))
	vp = tmp_path / "particles" / "glow.vpcf_c"
	vp.parent.mkdir()
	vp.write_bytes(s2.resource([("DATA", s2.kv3({"m_nMaxParticles": 8, "m_Renderers": [{
		"_class": "C_OP_RenderSprites", "m_nOutputBlendMode": "PARTICLE_OUTPUT_BLEND_MODE_ADD",
		"m_vecTexturesInput": [{"m_hTexture": "materials/particle/glow.vtex"}]}]}))]))
	assert detect_kind(vp) == Kind.SPR
	(s,) = particles.systems(vp)
	assert s.additive and s.material == "materials/particle/glow.vtex"
	proj = Project(Settings())
	proj.add_path(vp)
	asset = proj.assets[0]
	assert proj.children(asset) == ["glow"] and proj.info(asset)["Sprites"] == 1
	assert proj.preview(asset).source is not None
	out = {o.rel for o in proj.convert([asset], tmp_path / "out", log=lambda _m: None)}
	assert out == {"sprites/glow.spr"}


def test_project_converts_dmx_model(tmp_path):
	p = tmp_path / "arm.dmx"
	p.write_bytes(fx.binary(fx.arm_model()))
	proj = Project(Settings())
	proj.add_path(p)
	asset = proj.assets[0]
	assert proj.info(asset)["Source format"] == "DMX model 18 (binary 5)"
	out = {o.rel for o in proj.convert([asset], tmp_path / "out", log=lambda _m: None)}
	assert "models/arm.mdl" in out
