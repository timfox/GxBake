"""Panda3D (.egg/.bam/.pz/.mf) and Toontown-style layouts → studio MDL, on generated files only."""

import math
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gxbake import cli
from gxbake.core import archives, models, panda
from gxbake.core.project import Kind, Project, detect_kind

from . import fixtures as fx

K = 12.0	# default GoldSrc units per Panda unit


def _gold(p, yaw=-90.0, k=K):
	"""Panda Z-up point → where GxBake should put it (turned about Z, scaled)."""
	a = math.radians(yaw)
	r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
	return k * (np.asarray(p, float) @ r.T)


def _same(a, b, tol=0.05):
	a = np.array(sorted(map(tuple, np.round(np.asarray(a, float), 2))))
	b = np.array(sorted(map(tuple, np.round(np.asarray(b, float), 2))))
	return a.shape == b.shape and np.allclose(a, b, atol=tol)


ARM = """<CoordinateSystem> { Z-Up }
<Comment> { "generated for GxBake's tests" }
<Texture> skin {
  "phase_3/maps/arm_skin.jpg"
  <Scalar> alpha-file { "phase_3/maps/arm_skin_a.rgb" }
}
<Group> arm {
  <Dart> { 1 }
  <VertexPool> armpool {
    <Vertex> 0 { -0.1 0 0  <UV> { 0 0 } }
    <Vertex> 1 { 0.1 0 0   <UV> { 1 0 } }
    <Vertex> 2 { 0.1 0 1   <UV> { 1 1 } }
    <Vertex> 3 { -0.1 0 1  <UV> { 0 1 } }
    <Vertex> 4 { -0.1 0 2  <UV> { 0 1 } }
    <Vertex> 5 { 0.1 0 2   <UV> { 1 1 } }
    <Vertex> 6 { 0 0 2.5 <RGBA> { 1 0 0 1 } }
  }
  <Group> mesh {
    <Polygon> { <TRef> { skin } <VertexRef> { 0 1 2 3 <Ref> { armpool } } }
    <Polygon> { <TRef> { skin } <BFace> { 1 } <VertexRef> { 3 2 5 4 <Ref> { armpool } } }
    <Polygon> { <RGBA> { 0 1 0 1 } <VertexRef> { 4 5 6 <Ref> { armpool } } }
  }
  <Group> walls { <Collide> { Polyset descend } <Polygon> { <VertexRef> { 0 1 2 <Ref> { armpool } } } }
  <Joint> shoulder {
    <Transform> { <Matrix4> { 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1 } }
    <VertexRef> { 0 1 2 3 <Ref> { armpool } }
    <Joint> elbow {
      <Transform> { <Translate> { 0 0 1 } }
      <VertexRef> { 4 5 6 <Ref> { armpool } }
      <VertexRef> { 3 <Scalar> membership { 0.2 } <Ref> { armpool } }
    }
  }
}
"""


def _bend(frames=11, fps=10, cs="Z-Up"):
	"""The elbow pitches from 0 to 90 degrees (Panda pitch: about +X, nose up)."""
	p = " ".join(f"{90 * f / (frames - 1):g}" for f in range(frames))
	return f"""<CoordinateSystem> {{ {cs} }}
<Table> {{
  <Bundle> arm {{
    <Table> "<skeleton>" {{
      <Table> shoulder {{
        <Xfm$Anim_S$> xform {{ <Scalar> fps {{ {fps} }} }}
        <Table> elbow {{
          <Xfm$Anim_S$> xform {{
            <Scalar> fps {{ {fps} }}
            <Char*> order {{ srpht }}
            <S$Anim> p {{ <V> {{ {p} }} }}
            <S$Anim> z {{ <V> {{ 1 }} }}
          }}
        }}
      }}
    }}
  }}
}}
"""


def _toontown_tree(root: Path) -> Path:
	"""phase_3/models/char/<model>_1000.egg, its textures in phase_3/maps, its anim in phase_4."""
	char = root / "phase_3" / "models" / "char"
	char.mkdir(parents=True)
	maps = root / "phase_3" / "maps"
	maps.mkdir(parents=True)
	Image.new("RGB", (16, 16), (200, 100, 50)).save(maps / "arm_skin.jpg")
	alpha = Image.new("L", (16, 16), 255)
	alpha.paste(0, (0, 0, 8, 16))
	alpha.save(maps / "arm_skin_a.rgb", "SGI")
	model = char / "tt_a_chr_test_arm_1000.egg"
	model.write_text(ARM)
	(char / "tt_a_chr_test_arm_500.egg").write_text(ARM)	# another LOD: not an animation
	anims = root / "phase_4" / "models" / "char"
	anims.mkdir(parents=True)
	(anims / "tt_a_chr_test_arm_bend.egg").write_text(_bend())
	(root / "phase_4" / "models" / "char" / "unrelated_walk.egg").write_text(_bend())
	return model


def test_egg_syntax():
	root = panda.parse_egg('<Group> "a b" { <Scalar> x { 1 } 2 3 // comment\n /* block */ <V> { 4 } }')
	g = root.kids[0]
	assert (g.tag, g.name, g.vals) == ("group", "a b", ["2", "3"])
	assert g.scalar("x") == "1" and g.kid("v").floats() == [4.0]


def test_toontown_character_with_alpha_file_and_phase_anims(tmp_path):
	model = _toontown_tree(tmp_path)
	assert detect_kind(model) == Kind.MODEL
	assert panda.base_name("tt_a_chr_test_arm_1000") == "tt_a_chr_test_arm"
	assert panda.base_name("suitA-mod") == "suitA" and panda.base_name("panda-model") == "panda"
	assert [(lab, f.name) for lab, f in panda.animation_files(model)] == [("bend", "tt_a_chr_test_arm_bend.egg")]

	m = panda.import_panda(model, models.ModelOptions(anim_fps=10))
	assert [b.name for b in m.bones] == ["arm", "shoulder", "elbow"]
	assert [b.parent for b in m.bones] == [-1, 0, 1]
	assert [s.name for s in m.sequences] == ["bend"] and m.sequences[0].frames.shape == (11, 3, 6)
	skin = next(x for x in m.materials if x.name == "arm_skin")
	assert skin.masked and skin.rgba[0, 0, 3] == 0 and skin.rgba[0, -1, 3] == 255	# the _a.rgb mask
	assert any(x.name == "colours" for x in m.materials)	# untextured: one swatch skin
	# 4 textured triangles + 2 back faces of the <BFace> quad, 1 coloured; the <Collide> group is gone
	assert m.num_triangles == 4 + 2 + 1
	assert not any("not found" in w for w in m.warnings)

	mdl = models.build_studio_mdl(m, models.ModelOptions(anim_fps=10))
	rest = [(-0.1, 0, 0), (0.1, 0, 0), (0.1, 0, 1), (-0.1, 0, 1), (-0.1, 0, 2), (0.1, 0, 2), (0, 0, 2.5)]
	assert _same(models.studio_posed_vertices(mdl, 0, 0), _gold(rest))
	# last frame: pitch 90 about +X swings the forearm from +Z to -Y (Panda's back)
	bent = rest[:4] + [(-0.1, -1, 1), (0.1, -1, 1), (0, -1.5, 1)]
	assert _same(models.studio_posed_vertices(mdl, 0, 10), _gold(bent))


def test_project_scan_skips_maps_and_anim_files(tmp_path):
	_toontown_tree(tmp_path)
	proj = Project()
	added = proj.add_path(tmp_path)
	names = sorted(a.path.name for a in added)
	assert names == ["tt_a_chr_test_arm_1000.egg", "tt_a_chr_test_arm_500.egg"]	# no maps, no anims
	asset = next(a for a in added if a.path.name.endswith("_1000.egg"))
	info = proj.info(asset)
	assert info["Source"] == "Panda3D egg" and info["Animation files"] == "tt_a_chr_test_arm_bend.egg"
	assert info["Joints"] == "shoulder, elbow"
	assert "▶ bend" in proj.children(asset)
	out = tmp_path / "out"
	proj.convert([asset], out, log=lambda m: None)
	assert (out / "models" / "tt_a_chr_test_arm_1000.mdl").is_file()
	assert (out / "gc_studio" / "tt_a_chr_test_arm_1000.mdl").is_file()
	assert (out / "gc_studio" / "tt_a_chr_test_arm_1000t.mdl").is_file()


def test_y_up_anim_and_scale_options(tmp_path):
	"""An egg with no <CoordinateSystem> is Y-up, as Panda assumes; heading turns about its up axis."""
	model = tmp_path / "arm-model.egg"
	model.write_text(ARM.replace('"phase_3/maps/arm_skin.jpg"', '"missing.png"'))
	yup = _bend().replace("<CoordinateSystem> { Z-Up }", "").replace("<S$Anim> z { <V> { 1 } }", "<S$Anim> y { <V> { 1 } }")
	(tmp_path / "arm-bend.egg").write_text(yup)
	opts = models.ModelOptions(anim_fps=10, panda_scale=1.0, panda_yaw=0.0)
	m = panda.import_panda(model, opts)
	assert any("missing.png not found" in w for w in m.warnings)
	mdl = models.build_studio_mdl(m, opts)
	bent = [(-0.1, 0, 0), (0.1, 0, 0), (0.1, 0, 1), (-0.1, 0, 1), (-0.1, -1, 1), (0.1, -1, 1), (0, -1.5, 1)]
	assert _same(models.studio_posed_vertices(mdl, 0, 10), _gold(bent, 0.0, 1.0))


def test_static_lod_pz_and_flat_anim_table(tmp_path):
	egg = """<CoordinateSystem> { Z-Up }
<VertexPool> v {
  <Vertex> 0 { 0 0 0 } <Vertex> 1 { 1 0 0 } <Vertex> 2 { 0 1 0 }
  <Vertex> 3 { 0 0 5 } <Vertex> 4 { 1 0 5 } <Vertex> 5 { 0 1 5 } <Vertex> 6 { 1 1 5 }
}
<Group> lod {
  <Group> near { <SwitchCondition> { <Distance> { 50 0 <Vertex> { 0 0 0 } } }
    <Polygon> { <RGBA> { 0 0 1 1 } <VertexRef> { 3 4 6 5 <Ref> { v } } } }
  <Group> far { <SwitchCondition> { <Distance> { 500 50 <Vertex> { 0 0 0 } } }
    <Polygon> { <VertexRef> { 0 1 2 <Ref> { v } } } }
}
"""
	pz = tmp_path / "Crate.egg.pz"
	pz.write_bytes(zlib.compress(egg.encode()))
	assert detect_kind(pz) == Kind.MODEL
	m = panda.import_panda(pz)
	assert m.num_triangles == 2 and len(m.bones) == 1
	assert any("LOD" in w for w in m.warnings)
	assert _same(m.meshes[0].world, _gold([(0, 0, 5), (1, 0, 5), (1, 1, 5), (0, 1, 5)]))
	proj = Project()
	(asset,) = proj.add_path(pz)
	assert asset.stem == "crate"

	flat = panda.read_egg("""<CoordinateSystem> { Z-Up } <Table> { <Bundle> b { <Table> "<skeleton>" {
	  <Table> j { <Xfm$Anim> xform { <Scalar> fps { 5 } <Scalar> contents { hz } <V> { 0 1  90 2 } } } } } }""")
	(anim,) = flat.anims
	assert anim.fps == 5 and anim.num_frames == 2 and flat.animation_only
	m1 = anim.tables["j"][1]
	assert np.allclose(m1[:3, 3], (0, 0, 2)) and np.allclose(m1[:3, 0], (0, 1, 0))	# heading 90: +X → +Y


def test_pandactor_joins_parts_at_a_joint(tmp_path):
	legs = """<CoordinateSystem> { Z-Up }
<Group> legs { <Dart> { 1 }
  <VertexPool> p { <Vertex> 0 { 0 0 0 } <Vertex> 1 { 0.2 0 0 } <Vertex> 2 { 0 0 1 } }
  <Polygon> { <VertexRef> { 0 1 2 <Ref> { p } } }
  <Joint> hips { <Transform> { <Translate> { 0 0 1 } } <VertexRef> { 0 1 2 <Ref> { p } } }
}
"""
	torso = """<CoordinateSystem> { Z-Up }
<Group> torso { <Dart> { 1 }
  <VertexPool> p { <Vertex> 0 { 0 0 0 } <Vertex> 1 { 0.3 0 0 } <Vertex> 2 { 0 0 0.5 } }
  <Polygon> { <VertexRef> { 0 1 2 <Ref> { p } } }
  <Joint> spine { <VertexRef> { 0 1 2 <Ref> { p } } }
}
"""
	walk = """<CoordinateSystem> { Z-Up } <Table> { <Bundle> legs { <Table> "<skeleton>" {
  <Table> hips { <Xfm$Anim_S$> xform { <Scalar> fps { 10 } <S$Anim> z { <V> { 1 2 } } } } } } }"""
	(tmp_path / "legs-mod.egg").write_text(legs)
	(tmp_path / "legs-walk.egg").write_text(walk)
	(tmp_path / "torso-mod.egg").write_text(torso)
	actor = tmp_path / "toon.pandactor"
	actor.write_text('{"parts": [{"model": "legs-mod.egg"}, {"model": "torso-mod.egg", "attach": "hips"}]}')
	opts = models.ModelOptions(anim_fps=10, panda_scale=1.0, panda_yaw=0.0)
	m = panda.import_panda(actor, opts)
	assert [b.name for b in m.bones] == ["legs", "hips", "spine"] and m.bones[2].parent == 1
	mdl = models.build_studio_mdl(m, opts)
	tri_legs = [(0, 0, 0), (0.2, 0, 0), (0, 0, 1)]
	tri_torso = [(0, 0, 1), (0.3, 0, 1), (0, 0, 1.5)]	# the torso sits on the hips joint
	assert _same(models.studio_posed_vertices(mdl, 0, 0), tri_legs + tri_torso)
	up = [(x, y, z + 1) for x, y, z in tri_legs + tri_torso]	# hips rise 1: everything on them follows
	assert _same(models.studio_posed_vertices(mdl, 0, 1), up)
	proj = Project()
	(asset,) = proj.add_path(actor)
	assert proj.info(asset)["Source"] == "Panda3D multi-part actor"
	# "anims": null for a part drops its auto-found clip; the clip goes when no part has it
	actor.write_text('{"parts": [{"model": "legs-mod.egg"}, {"model": "torso-mod.egg", "attach": "hips"}],'
		' "anims": {"walk": [null]}}')
	assert [s.name for s in panda.import_panda(actor, opts).sequences] == ["idle"]


def test_multifile_extract(tmp_path, capsys):
	files = {"phase_3/models/char/a.egg": ARM.encode(), "phase_3/maps/a.jpg": b"\xff\xd8 not really"}
	for compress, prefix in ((False, b""), (True, b"#! /usr/bin/env packp3d\n#second line\n")):
		mf = tmp_path / ("c.mf" if compress else "p.mf")
		mf.write_bytes(fx.panda_multifile(files, compress=compress, prefix=prefix))
		assert {m.name: m.read() for m in archives.members(mf)} == files
	out = tmp_path / "out"
	assert cli.main(["extract", str(tmp_path / "c.mf"), "-o", str(out), "--match", "*.egg"]) == 0
	assert (out / "phase_3" / "models" / "char" / "a.egg").read_text() == ARM
	assert not (out / "phase_3" / "maps").exists()


def test_multifile_reads_in_panda3d(tmp_path):
	"""Our multifile writer (the test fixture) and reader agree with Panda3D's own Multifile."""
	core = pytest.importorskip("panda3d.core")
	mf_path = tmp_path / "x.mf"
	mf_path.write_bytes(fx.panda_multifile({"a/b.txt": b"hello" * 40, "c.egg": b"<Group> x { }"}, compress=True,
		prefix=b"#!/bin/sh\n"))
	mf = core.Multifile()
	assert mf.openRead(core.Filename.fromOsSpecific(str(mf_path)))
	assert {mf.getSubfileName(i): bytes(mf.readSubfile(i)) for i in range(mf.getNumSubfiles())} == {
		"a/b.txt": b"hello" * 40, "c.egg": b"<Group> x { }"}


HPR_MODEL = """<CoordinateSystem> { %s }
<Group> bot { <Dart> { 1 }
  <VertexPool> p {
    <Vertex> 0 { 0 0 0 } <Vertex> 1 { 1 0 0 } <Vertex> 2 { 0 1 0 }
    <Vertex> 3 { 0 0 1 } <Vertex> 4 { 1 0 1 } <Vertex> 5 { 0 1.5 1.2 } <Vertex> 6 { 0.3 0.2 2.0 }
  }
  <Polygon> { <VertexRef> { 0 1 2 <Ref> { p } } }
  <Polygon> { <VertexRef> { 3 4 5 <Ref> { p } } }
  <Polygon> { <VertexRef> { 4 5 6 <Ref> { p } } }
  <Joint> hips { <Transform> { <RotZ> { 10 } <Translate> { 0 0 0.5 } }
    <VertexRef> { 0 1 2 <Ref> { p } }
    <Joint> chest { <Transform> { <Rotate> { 20 1 1 0 } <Translate> { 0.1 0 0.5 } }
      <VertexRef> { 3 4 5 6 <Ref> { p } } } }
}
"""
HPR_ANIM = """<CoordinateSystem> { %s }
<Table> { <Bundle> bot { <Table> "<skeleton>" {
  <Table> hips { <Xfm$Anim_S$> xform { <Scalar> fps { 10 }
     <S$Anim> h { <V> { 0 20 40 } } <S$Anim> z { <V> { 0.5 0.7 0.9 } } <S$Anim> x { <V> { 0 0.2 0.4 } } }
    <Table> chest { <Xfm$Anim_S$> xform { <Scalar> fps { 10 }
       <S$Anim> h { <V> { 10 -30 50 } } <S$Anim> p { <V> { 5 25 -45 } } <S$Anim> r { <V> { 15 -35 60 } }
       <S$Anim> x { <V> { 0.1 } } <S$Anim> z { <V> { 0.5 } } } } } } } }
"""


@pytest.mark.parametrize("model_cs,anim_cs", [("Z-Up", "Z-Up"), ("Y-Up", "Y-Up"), ("Z-Up", "Y-Up")])
def test_poses_match_panda3d(tmp_path, model_cs, anim_cs):
	"""Every animated vertex lands where Panda3D's own skinning puts it, in .egg and through .bam."""
	core = pytest.importorskip("panda3d.core")
	egg_mod = pytest.importorskip("panda3d.egg")
	core.loadPrcFileData("", "notify-level error")
	mp, ap = tmp_path / "bot-model.egg", tmp_path / "bot-wave.egg"
	mp.write_text(HPR_MODEL % model_cs)
	ap.write_text(HPR_ANIM % anim_cs)
	model = core.NodePath(egg_mod.load_egg_file(core.Filename.fromOsSpecific(str(mp))))
	anim = core.NodePath(egg_mod.load_egg_file(core.Filename.fromOsSpecific(str(ap))))
	bams = tmp_path / "bam"
	bams.mkdir()
	model.writeBamFile(core.Filename.fromOsSpecific(str(bams / "bot-model.bam")))
	anim.writeBamFile(core.Filename.fromOsSpecific(str(bams / "bot-wave.bam")))
	char = model.find("**/+Character").node()
	bundle = char.getBundle(0)
	ctrl = bundle.bindAnim(anim.find("**/+AnimBundleNode").node().getBundle(), -1, core.PartSubset())
	opts = models.ModelOptions(anim_fps=10)
	for src in (mp, bams / "bot-model.bam"):
		mdl = models.build_studio_mdl(panda.import_panda(src, opts), opts)
		for f in range(ctrl.getNumFrames()):
			ctrl.pose(f)
			bundle.forceUpdate()
			pts = []
			for g in model.find("**/+GeomNode").node().getGeoms():
				vd = g.getVertexData().animateVertices(True, core.Thread.getCurrentThread())
				r = core.GeomVertexReader(vd, "vertex")
				while not r.isAtEnd():
					v = r.getData3()
					pts.append((v[0], v[1], v[2]))
			assert _same(models.studio_posed_vertices(mdl, 0, f), _gold(pts)), (src.name, f)
