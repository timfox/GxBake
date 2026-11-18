"""USD import through FreeUSD."""

from __future__ import annotations

import numpy as np
import pytest

from gxbake.core import audit, models, studio, usd
from gxbake.core.project import Kind, Project, detect_kind

from . import usd_fixtures as ufx
from .test_core import _signed_volume

needs_freeusd = pytest.mark.skipif(not usd.available(), reason="FreeUSD not installed (pip install 'gxbake[usd]')")


# -- the layer rewrite (no FreeUSD needed) -------------------------------------------


def test_rewrite_strips_what_freeusd_cannot_parse():
	text, meta, prims = usd.rewrite_usda(ufx.CHAR_USDA)
	assert "uniform " not in text
	assert "half3[]" not in text and "float3[] scales" in text
	assert "elementSize" not in text and "interpolation" not in text
	# values survive, metadata blocks don't
	assert "int[] primvars:skel:jointIndices = [0, 1, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1, 2, 1, 2, 1]\n" in text
	assert "texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)]\n" in text
	# prim-level metadata (apiSchemas, references) is left alone
	assert 'prepend apiSchemas = ["SkelBindingAPI", "MaterialBindingAPI"]' in text
	assert meta[("/Char/Body", "primvars:skel:jointIndices")] == {"elementSize": "2", "interpolation": "vertex"}
	assert meta[("/Char/Body", "primvars:st")] == {"interpolation": "faceVarying"}
	assert "/Char/Body/HeadFaces" in prims and "/Char/Looks/Red/Tex" in prims and "/Char/Skel/Wave" in prims


def test_rewrite_single_line_metadata_and_tuples():
	src = ('def Mesh "M"\n{\n    double3 xformOp:translate = (1, 2, 3)\n'
		'    texCoord2f[] primvars:uv = [(0, 0), (1, 0), (0, 1)] ( interpolation = "faceVarying" )\n'
		'    custom uniform half h = 1\n}\n')
	text, meta, prims = usd.rewrite_usda(src)
	assert "double3 xformOp:translate = (1, 2, 3)\n" in text	# a tuple value isn't metadata
	assert "texCoord2f[] primvars:uv = [(0, 0), (1, 0), (0, 1)]\n" in text
	assert "float h = 1" in text
	assert meta == {("/M", "primvars:uv"): {"interpolation": "faceVarying"}}
	assert prims == ["/M"]


def test_usd_files_are_models(tmp_path):
	for ext in (".usd", ".usda", ".usdc", ".usdz"):
		p = tmp_path / f"x{ext}"
		p.write_bytes(b"#usda 1.0\n")
		assert detect_kind(p) == Kind.MODEL
	(tmp_path / "x@walk.usda").write_text("#usda 1.0\n")
	assert models.clip_base(tmp_path / "x@walk.usda") is not None
	assert [p.name for p in models.clip_files(tmp_path / "x.usda")] == ["x@walk.usda"]


# -- import -------------------------------------------------------------------------


@needs_freeusd
def test_skinned_character(tmp_path):
	model = usd.import_usd(ufx.write_char(tmp_path))
	assert model.warnings == []
	assert [b.name for b in model.bones] == ["root", "Hips", "Spine", "Head", "Box"]
	assert [b.parent for b in model.bones] == [-1, 0, 1, 2, 0]
	names = [m.name for m in model.materials]
	assert names == ["Green", "Red", "displayColor"]	# the guide mesh is skipped
	green, red, blue = model.materials
	assert tuple(green.rgba[0, 0]) == (30, 200, 60, 255) and green.texture == "tex/green.png"
	assert tuple(red.rgba[0, 0]) == (200, 30, 60, 255)
	assert tuple(blue.rgba[0, 0]) == (0, 0, 255, 255)
	# the GeomSubset's face (the head quad) went to Red; the rest stayed Green
	tris = {model.materials[m.material].name: len(m.tris) for m in model.meshes}
	assert tris == {"Green": 4, "Red": 2, "displayColor": 1}
	# centimetres, Y-up → inches, Z-up: 120 cm tall, 20 cm wide
	lo, hi = model.bounds()
	assert hi[2] == pytest.approx(120 / 2.54, abs=1e-3)
	assert lo[1] == pytest.approx(-10 / 2.54, abs=1e-3) and lo[2] == pytest.approx(0, abs=1e-6)
	# skinned points follow their heaviest joint
	body = next(m for m in model.meshes if model.materials[m.material].name == "Green")
	head = next(m for m in model.meshes if model.materials[m.material].name == "Red")
	assert set(body.bones.tolist()) == {1, 2, 3} and set(head.bones.tolist()) == {3}

	(seq,) = model.sequences
	assert seq.name == "wave" and seq.frames.shape == (16, 5, 6)	# 1 s at 15 fps
	# the spine turns 90° about the source Z axis, halfway through
	mid = seq.frames[np.argmin([abs(k / 15 * 24 - 12) for k in range(16)])]
	assert abs(mid[2, 5]) > 1.2 and abs(seq.frames[0, 2, 5]) < 1e-6
	# the prop rises 30 cm over the clip, as its own bone
	assert seq.frames[-1, 4, 2] - seq.frames[0, 4, 2] == pytest.approx(30 / 2.54, abs=1e-3)

	raw = models.build_studio_mdl(model)
	assert studio.studio_info(raw)["bones"] == 5
	rest = models.studio_posed_vertices(raw, 0, 0)
	assert rest[:, 2].max() == pytest.approx(120 / 2.54, abs=1e-2)


@needs_freeusd
def test_clip_files_add_sequences(tmp_path):
	model = usd.import_usd(ufx.write_char(tmp_path, clip=True))
	assert [s.name for s in model.sequences] == ["wave", "nod"]
	nod = model.sequences[1]
	assert nod.frames.shape == (16, 5, 6)	# 1 s at 15 fps
	assert np.abs(nod.frames[-1, 3, 3:] - nod.frames[0, 3, 3:]).max() > 1.5	# the head turns 90°
	assert np.allclose(nod.frames[:, 4], nod.frames[0, 4])	# the prop has no keys in the clip


@needs_freeusd
def test_static_bakes_one_bone(tmp_path):
	model = usd.import_usd(ufx.write_char(tmp_path), models.ModelOptions(static=True))
	assert len(model.bones) == 1 and [s.name for s in model.sequences] == ["idle"]
	assert "baked static: skeleton and animations ignored" in model.warnings


@needs_freeusd
def test_references_keep_their_materials(tmp_path):
	model = usd.import_usd(ufx.write_scene(tmp_path))
	assert model.warnings == []
	(mat,) = model.materials	# both instances share one skin
	assert mat.texture == "./tex/red.png" and tuple(mat.rgba[0, 0]) == (200, 30, 60, 255)
	assert len(model.meshes) == 2
	los = sorted(float(m.world[:, 0].min()) for m in model.meshes)
	assert los == pytest.approx([0.0, 6 / 0.0254], abs=1e-2)	# (3 m translate) x 2 scale, in inches


@needs_freeusd
@pytest.mark.parametrize("orientation", ["rightHanded", "leftHanded"])
def test_cube_winding_and_units(tmp_path, orientation):
	p = tmp_path / "cube.usda"
	p.write_text(ufx.cube_usda(orientation))
	model = usd.import_usd(p)
	lo, hi = model.bounds()
	assert np.allclose(lo, -1) and np.allclose(hi, 1)	# metersPerUnit 0.0254: one unit per inch
	raw = models.build_studio_mdl(model)
	assert _signed_volume(raw) == pytest.approx(8.0, rel=1e-4)	# closed and outward-facing either way
	assert tuple(model.materials[0].rgba[0, 0]) == (255, 0, 0, 255)


@needs_freeusd
def test_usdz_package(tmp_path):
	model = usd.import_usd(ufx.write_usdz(tmp_path / "char.usdz"))
	assert model.warnings == []
	assert [m.name for m in model.materials] == ["Green", "Red", "displayColor"]
	assert tuple(model.materials[0].rgba[0, 0]) == (30, 200, 60, 255)


@needs_freeusd
def test_binary_crate_says_how_to_convert(tmp_path):
	p = tmp_path / "x.usdc"
	p.write_bytes(b"PXR-USDC" + bytes(88))
	with pytest.raises(ValueError, match="usdcat"):
		usd.import_usd(p)


@needs_freeusd
def test_project_converts_usd(tmp_path):
	src = ufx.write_char(tmp_path / "src", clip=True)
	proj = Project()
	(asset,) = [a for a in proj.add_path(src.parent) if a.kind == Kind.MODEL]	# the clip isn't its own asset
	info = proj.info(asset)
	assert info["Source format"] == "USD stage (read with FreeUSD)" and info["Bones"] == 5
	out = tmp_path / "out"
	proj.convert([asset], out, log=lambda _m: None)
	assert asset.status.startswith("Done"), asset.warnings
	for rel in ("models/char.mdl", "gc_studio/char.mdl", "gc_studio/chart.mdl"):
		assert (out / rel).is_file(), rel
	assert [e for e in audit.validate_staged_tree(out) if "critical asset" not in e] == []


@needs_freeusd
def test_sublayers_and_payloads(tmp_path):
	tri = ('int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n'
		'    point3f[] points = [(0, 0, {z}), (1, 0, {z}), (0, 1, {z})]')
	(tmp_path / "sub.usda").write_text(f'#usda 1.0\n\ndef Mesh "Extra"\n{{\n    {tri.format(z=5)}\n}}\n')
	(tmp_path / "root.usda").write_text(
		'#usda 1.0\n(\n    subLayers = [\n        @./sub.usda@\n    ]\n    upAxis = "Z"\n    metersPerUnit = 0.0254\n)\n\n'
		f'def Mesh "Base"\n{{\n    {tri.format(z=0)}\n}}\n\n'
		'def Xform "Holder" (\n    prepend payload = @./sub.usda@</Extra>\n)\n{\n'
		'    double3 xformOp:translate = (0, 0, 10)\n    uniform token[] xformOpOrder = ["xformOp:translate"]\n}\n')
	model = usd.import_usd(tmp_path / "root.usda")
	assert sorted(float(m.world[:, 2].max()) for m in model.meshes) == [0.0, 5.0, 15.0]
