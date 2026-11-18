"""gxbake lumps: map lump tables, the port's .gcmeta, entity references and the baked .gcd files."""

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from gxbake import cli
from gxbake.core import lumps, source_map
from gxbake.core.project import Kind, Project

from . import fixtures as fx
from . import lump_fixtures as lf
from . import s1_fixtures as s1

ENTS = (
	'{"classname" "worldspawn" "wad" "\\\\half-life\\\\valve\\\\halflife.wad;decals.wad" "skyname" "Desert"}\n'
	'{"classname" "func_door" "model" "*1"}\n'
	'{"classname" "env_model" "model" "models/Props/Crate.mdl"}\n'
	'{"classname" "env_sprite" "model" "sprites/glow01.spr"}\n'
	'{"classname" "ambient_generic" "message" "ambience/wind1.wav"}\n'
	'{"classname" "trigger_changelevel" "map" "c1a1" "landmark" "lm"}\n'
	'{"classname" "env_model" "model" "models/props/crate.mdl"}\n'
)


def _reference_gcmeta(data: bytes) -> str:
	"""The HL1 port's scripts/generate-gc-bsp-meta.py, verbatim in logic, to pin byte-for-byte parity."""
	names = ("entities", "planes", "textures", "vertexes", "visibility", "nodes", "texinfo", "faces",
		"lighting", "clipnodes", "leafs", "marksurfaces", "edges", "surfedges", "models")
	hot = {"planes", "visibility", "nodes", "texinfo", "faces", "leafs", "marksurfaces", "models"}
	version = struct.unpack_from("<i", data, 0)[0]
	rows = [f"GCMETA 1 {version} {zlib.crc32(data) & 0xffffffff:08x} {len(data)}\n"]
	for index, name in enumerate(names):
		offset, size = struct.unpack_from("<ii", data, 4 + index * 8)
		rows.append(f"{index:02d} {name} {'HOT' if name in hot else 'COLD'} {offset} {size} {offset & 31}\n")
	face_ofs, face_len = struct.unpack_from("<ii", data, 4 + 7 * 8)
	tex_ofs, _tex_len = struct.unpack_from("<ii", data, 4 + 6 * 8)
	batches: dict = {}
	for fi in range(face_len // 20):
		ti = struct.unpack_from("<h", data, face_ofs + fi * 20 + 10)[0]
		miptex, flags = struct.unpack_from("<ii", data, tex_ofs + ti * 40 + 32)
		batches.setdefault((miptex, flags), []).append(fi)
	rows.append(f"BATCHES {len(batches)} {face_len // 20}\n")
	for (miptex, flags), faces in sorted(batches.items()):
		rows.append("BATCH %d %d %d %d %s\n" % (miptex, flags, len(faces), faces[0], ",".join(map(str, faces))))
	return "".join(rows)


@pytest.fixture(scope="module")
def disp_map(tmp_path_factory) -> Path:
	"""A real BSP30 with 32 faces: the Source displacement fixture through the map converter."""
	d = tmp_path_factory.mktemp("lumpmap")
	r = source_map.convert(s1.write_map(d / "src", displacement=True), source_map.MapOptions(disp_power=2))
	p = d / "game" / "maps" / "disp.bsp"
	p.parent.mkdir(parents=True)
	p.write_bytes(r.bsp)
	return p


def test_lump_table_and_gcmeta_parity(disp_map):
	data = disp_map.read_bytes()
	bsp = lumps.read_bsp(data, str(disp_map))
	assert bsp.format == "goldsrc" and bsp.issues == []
	assert [x.name for x in bsp.lumps] == list(lumps.GOLDSRC_LUMPS)
	assert bsp.by_name("faces").count == 32 and bsp.by_name("faces").tier == "HOT"
	assert bsp.by_name("lighting").tier == "COLD"
	assert lumps.gcmeta(bsp) == _reference_gcmeta(data)
	meta = lumps.read_gcmeta(lumps.gcmeta(bsp))
	assert meta.crc32 == zlib.crc32(data) and sum(len(f) for *_x, f in meta.batches) == 32


def test_layout_problems_are_reported():
	data = bytearray(fx.bsp30('{"classname" "worldspawn"}', [fx.miptex("a", 16, 16)]))
	struct.pack_into("<ii", data, 4 + 3 * 8, 130, 40)	# vertexes now overlap the entities
	bsp = lumps.read_bsp(bytes(data))
	assert any("overlapping" in i for i in bsp.issues) and any("12-byte record" in i for i in bsp.issues)
	with pytest.raises(ValueError):
		lumps.gcmeta(bsp)


def test_entity_refs_sorted_by_kind():
	bsp = lumps.read_bsp(fx.bsp30(ENTS, [fx.miptex("floor", 16, 16), fx.miptex("wall", 16, 16, embed=False)]))
	refs = lumps.entity_refs(lumps.entities(bsp))
	assert refs["models"] == ["models/props/crate.mdl"]	# case-folded, once
	assert refs["sprites"] == ["sprites/glow01.spr"] and refs["sounds"] == ["sound/ambience/wind1.wav"]
	assert refs["brush_models"] == ["*1"] and refs["wads"] == ["halflife.wad", "decals.wad"]
	assert refs["sky"] == "desert" and refs["maps"] == ["c1a1"]
	assert list(refs["classnames"])[0] == "env_model"
	tex = lumps.textures(bsp)
	assert [(t.name, t.embedded) for t in tex] == [("floor", True), ("wall", False)]


def test_world_dl_versions_from_each_port():
	for version, flags, writer in ((1, 0, "Day of Defeat"), (2, 1, "HL1 / Lost Coast"), (2, 0, "Lost Coast")):
		wd = lumps.read_world_dl(lf.world_dl(version, flags, pages=2 if version > 1 else 0))
		assert wd.version == version and len(wd.chunks) == 2 and len(wd.materials) == 2
		assert wd.vertex_stride == (8 if version == 1 else 15)
		c0 = wd.chunks[0]
		assert [b.material for b in c0.batches] == [0, 1] and c0.batches[0].prims == 2 and c0.batches[0].tris == 4
		assert c0.tris == 8 and wd.chunks[1].tris == 2
		assert wd.lm_pages == (2 if version > 1 else 0)
		assert writer in lumps._wd_writer(wd)
	wd = lumps.read_world_dl(lf.world_dl(2, 0, pages=2, sky=True))
	assert wd.sky["scale"] == 16.0 and wd.sky["fog"] == [10, 20, 30] and "sky" in wd.sections()
	page = wd.lightmap_page(1)
	assert page.shape == (32, 32, 3) and (page[..., 0] == (16 << 3 | 16 >> 2)).all()


def test_world_dl_material_names(tmp_path):
	game = tmp_path / "game"
	(game / "media").mkdir(parents=True)
	(game / "maps").mkdir()
	(game / "maps" / "x.bsp").write_bytes(fx.bsp30('{"classname" "worldspawn"}',
		[fx.miptex("Floor", 16, 16), fx.miptex("wall", 16, 16)]))
	gcd = game / "media" / "wd_x.gcd"
	gcd.write_bytes(lf.world_dl(names=("floor", "wall")))
	info = lumps.describe(gcd)	# finds maps/x.bsp by itself
	assert info["named_materials"] == 2 and info["names_from"].endswith("x.bsp")
	assert {m["name"] for m in info["material_table"]} == {"Floor", "wall"}
	# Lost Coast: hashes are of the pack names in texture-names.json
	names = tmp_path / "texture-names.json"
	names.write_text(json.dumps({"nature/rock01": "rock01", "nature/sand": "sand"}))
	other = tmp_path / "wd_y.gcd"
	other.write_bytes(lf.world_dl(2, 0, names=("rock01", "sand")))
	assert lumps.describe(other, names)["named_materials"] == 2


def test_lightfaces():
	lfd = lumps.read_lightfaces(lf.lightfaces(3))
	assert lfd.faces == 3 and lfd.lighting == bytes(range(48))
	assert [r["lightofs"] for r in lfd.records] == [0, 12, -1] and lfd.records[2]["sky"]
	assert lfd.records[0]["extents"] == [32, 16]


def test_extract_bsp(disp_map, tmp_path):
	out = tmp_path / "x"
	written = lumps.extract(disp_map, out, decode_textures=True)
	names = {p.relative_to(out).as_posix() for p in written}
	assert "lumps/07_faces.bin" in names and "disp.gcmeta" in names and "textures/floor.png" in names
	assert (out / "lumps/07_faces.bin").stat().st_size == 640
	assert (out / "disp.gcmeta").read_text() == _reference_gcmeta(disp_map.read_bytes())
	assert json.loads((out / "refs.json").read_text())["models"] == ["models/crate01.mdl"]
	assert json.loads((out / "batches.json").read_text())[0]["texture"] == "floor"


def test_extract_world_dl(tmp_path):
	gcd = tmp_path / "wd_z.gcd"
	gcd.write_bytes(lf.world_dl(pages=2))
	out = tmp_path / "x"
	lumps.extract(gcd, out)
	assert (out / "lightmaps/page_001.png").is_file() and (out / "lightmaps/atlas.png").is_file()
	assert {p.stem for p in (out / "sections").iterdir()} >= {"header", "mats", "chunks", "lightmaps"}
	chunks = json.loads((out / "chunks.json").read_text())
	assert chunks[0]["bytes"] >= chunks[-1]["bytes"]	# largest first
	assert sum((out / "sections" / f"{k}.bin").stat().st_size for k in ("header", "mats", "nodes", "planes",
		"chunkdir", "pvsdir", "pvs", "chunks", "lightmaps")) <= gcd.stat().st_size


def test_source_map_lumps(tmp_path):
	p = s1.write_map(tmp_path)
	info = lumps.describe(p)
	assert info["format"] == "source" and info["issues"] == []
	assert info["textures"][0]["name"].lower() == "test/floor"
	assert "models/props/crate01.mdl" in info["refs"]["models"] or info["refs"]["models"]
	written = lumps.extract(p, tmp_path / "x", raw=False)
	assert (tmp_path / "x/pakfile.zip").is_file() and not (tmp_path / "x/lumps").exists()
	assert not any(w.suffix == ".gcmeta" for w in written)


def test_stale_gcmeta(disp_map, tmp_path):
	game = tmp_path / "game"
	(game / "maps").mkdir(parents=True)
	(game / "media").mkdir()
	bsp = game / "maps" / "disp.bsp"
	bsp.write_bytes(disp_map.read_bytes())
	meta = game / "media" / "disp.gcmeta"
	meta.write_text(lumps.gcmeta(lumps.read_bsp(bsp.read_bytes())))
	assert lumps.describe(meta)["issues"] == []
	bsp.write_bytes(bsp.read_bytes() + b"\0" * 4)
	assert "stale" in lumps.describe(meta)["issues"][0]


def test_cli_lumps(disp_map, tmp_path, capsys):
	media = tmp_path / "media"
	media.mkdir()
	(media / "wd_disp.gcd").write_bytes(lf.world_dl())
	(media / "wd_disp_lf.gcd").write_bytes(lf.lightfaces())
	assert cli.main(["lumps", str(disp_map), str(media), "-o", str(tmp_path / "out"), "--limit", "2"]) == 0
	text = capsys.readouterr().out
	assert "HOT" in text and "GCwd v2" in text and "GClf v1" in text
	assert {p.name for p in (tmp_path / "out").iterdir()} == {"disp_bsp", "wd_disp_gcd", "wd_disp_lf_gcd"}
	assert cli.main(["lumps", str(media / "wd_disp.gcd"), "--json"]) == 0
	assert json.loads(capsys.readouterr().out)["chunks"] == 2


def test_project_inspects_baked_maps(disp_map, tmp_path):
	gcd = tmp_path / "wd_q.gcd"
	gcd.write_bytes(lf.world_dl(pages=3))
	proj = Project()
	proj.add_path(gcd)
	proj.add_path(disp_map)
	a, b = proj.assets
	assert a.kind == Kind.GCMAP and "GCwd v2" in proj.info(a)["Bake"]
	assert proj.children(a) and "hot" in proj.info(b)["Lumps"]
	pv = proj.preview(a)
	assert pv.result.shape == (32, 96, 4) and np.all(pv.result[..., 3] == 255)
