"""Source 1 maps (VBSP v19-21) converted to the port's GoldSrc BSP30, on a generated map."""

import io
import struct
import zipfile

import numpy as np
import pytest

from gxbake.core import bsp30, source_bsp as sb, source_map
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import s1_fixtures as s1


def _hull(bsp: bytes, hull: int, point) -> int:
	"""Point contents in clip hull 1-3 of model 0, walked from the written lumps."""
	lumps = bsp30.read_lumps(bsp)
	planes = [struct.unpack_from("<4fi", lumps[bsp30.L_PLANES], i * 20) for i in range(len(lumps[bsp30.L_PLANES]) // 20)]
	clip = lumps[bsp30.L_CLIPNODES]
	n = struct.unpack_from("<4i", lumps[bsp30.L_MODELS], 36)[hull]
	p = np.asarray(point, np.float64)
	while n >= 0:
		pl, c0, c1 = struct.unpack_from("<ihh", clip, n * 8)
		nx, ny, nz, d, _t = planes[pl]
		n = c0 if nx * p[0] + ny * p[1] + nz * p[2] - d >= 0 else c1
	return n


def _entities(bsp: bytes) -> list[dict]:
	return sb.parse_entities(bsp30.read_lumps(bsp)[bsp30.L_ENTITIES].split(b"\0")[0].decode())


def test_parse_generated_vbsp():
	src = sb.parse(s1.source_bsp())
	assert src.version == 20 and sb.is_source_bsp(src.data[:8])
	assert src.texture_names() == ["TEST/FLOOR"]
	assert len(src.faces_and_lighting()[0]) == 1 and len(src.leafs) == 7 and len(src.brushes) == 1
	n, rows = src.visibility()
	assert n == 1 and rows == [b"\x01"]
	assert "materials/test/floor.vtf" in src.pakfile().namelist()
	props = src.static_props()
	assert [(p.model, p.origin) for p in props] == [("models/props/crate01.mdl", (64.0, 0.0, 0.0))]
	ents = sb.parse_entities(src.entities_text())
	assert ents[0]["skyname"] == "testsky" and ents[1]["classname"] == "info_player_teamspawn"
	with pytest.raises(ValueError):
		sb.parse(b"IBSP" + bytes(1100))


def test_vis_rle_round_trip():
	row = bytes([1, 0, 0, 0, 0, 255, 0, 3] + [0] * 300)
	comp = bsp30.compress_vis(row)
	assert len(comp) < len(row) and sb.decompress_vis(comp, 0, len(row)) == row


def test_plane_table_faces_positive_axis():
	pt = bsp30.PlaneTable()
	a, fa = pt.add(np.array([0.0, 0.0, -1.0]), 16.0)
	b, fb = pt.add(np.array([0.0, 0.0, 1.0]), -16.0)
	assert a == b and fa and not fb
	assert np.allclose(pt.normal(a), [0, 0, 1]) and pt.dists[a] == -16.0


def test_convert_slab(tmp_path):
	p = s1.write_map(tmp_path)
	r = source_map.convert(p)
	assert r.warnings == [] and r.stats["faces"] == 1 and r.stats["leafs"] == 7
	lumps = bsp30.read_lumps(r.bsp)
	assert struct.unpack_from("<i", r.bsp)[0] == 30
	# the texture came out of the pakfile at 1 texel per unit, as a paletted miptex
	assert struct.unpack_from("<i", lumps[bsp30.L_TEXTURES])[0] == 1
	ofs = struct.unpack_from("<i", lumps[bsp30.L_TEXTURES], 4)[0]
	name, w, h = struct.unpack_from("<16sII", lumps[bsp30.L_TEXTURES], ofs)
	assert name.split(b"\0")[0] == b"floor" and (w, h) == (64, 64)
	# 17x17 luxels, brighter towards +x as in the source lightmap
	light = np.frombuffer(lumps[bsp30.L_LIGHTING], np.uint8).reshape(17, 17, 3)
	assert light[:, -1, 0].mean() > light[:, 0, 0].mean() + 40
	# the slab is solid to a standing player up to its half height above the floor
	assert _hull(r.bsp, 1, (0, 0, 30)) == bsp30.CONTENTS_SOLID
	assert _hull(r.bsp, 1, (0, 0, 40)) == bsp30.CONTENTS_EMPTY
	assert _hull(r.bsp, 1, (150, 0, 0)) == bsp30.CONTENTS_EMPTY
	assert _hull(r.bsp, 3, (0, 0, 15)) == bsp30.CONTENTS_SOLID and _hull(r.bsp, 3, (0, 0, 20)) == bsp30.CONTENTS_EMPTY
	ents = _entities(r.bsp)
	world = ents[0]
	assert world["wad"] == "" and "detailmaterial" not in world
	classes = [e["classname"] for e in ents]
	assert "info_player_deathmatch" in classes and "info_player_start" in classes
	crate = next(e for e in ents if e["classname"] == "cycler_sprite")
	assert crate["model"] == "models/crate01.mdl" and crate["origin"] == "64 0 0"
	assert r.sky == "testsky" and sorted(r.sky_sides) == ["bk", "dn", "ft", "lf", "rt", "up"]


def test_convert_displacement(tmp_path):
	p = s1.write_map(tmp_path, displacement=True)
	r = source_map.convert(p, source_map.MapOptions(disp_power=2))
	assert r.stats["displacement triangles"] == 32 and r.stats["faces"] == 32
	lumps = bsp30.read_lumps(r.bsp)
	# the triangles reach the open leaf above the slab through its marksurfaces
	leaf1 = struct.unpack_from("<ii6hHH", lumps[bsp30.L_LEAFS], 28)
	assert leaf1[9] == 32	# once each
	# collision follows the bump, not the flat base brush
	assert _hull(r.bsp, 1, (0, 0, 60)) == bsp30.CONTENTS_SOLID
	assert _hull(r.bsp, 1, (0, 0, 75)) == bsp30.CONTENTS_EMPTY
	assert _hull(r.bsp, 1, (-120, -120, 40)) == bsp30.CONTENTS_EMPTY
	coarse = source_map.convert(p, source_map.MapOptions(disp_power=1))
	assert coarse.stats["displacement triangles"] == 8


def test_missing_content_warns(tmp_path):
	p = s1.write_map(tmp_path, pak=False, props=[])
	r = source_map.convert(p, source_map.MapOptions(props=False))
	assert any("material(s) not found" in w for w in r.warnings)
	assert any("sides not found" in w for w in r.warnings)
	assert not any(e["classname"] == "cycler_sprite" for e in _entities(r.bsp))


def test_material_search_dirs(tmp_path):
	game = tmp_path / "game"
	with zipfile.ZipFile(io.BytesIO(s1.pakfile())) as z:
		z.extractall(game)
	p = s1.write_map(tmp_path, pak=False)
	r = source_map.convert(p, source_map.MapOptions(search=[game]))
	assert not any("not found" in w for w in r.warnings)


def test_unsupported_version(tmp_path):
	raw = bytearray(s1.source_bsp())
	raw[4:8] = struct.pack("<i", 25)
	p = tmp_path / "v25.bsp"
	p.write_bytes(bytes(raw))
	with pytest.raises(ValueError, match="version 25"):
		source_map.convert(p)


def test_project_converts_source_map(tmp_path):
	p = s1.write_map(tmp_path)
	assert detect_kind(p) == Kind.BSP
	proj = Project(Settings())
	proj.add_path(p)
	asset = proj.assets[0]
	info = proj.info(asset)
	assert info["Source BSP version"] == 20 and info["Sky"] == "testsky"
	out = {o.rel for o in proj.convert([asset], tmp_path / "out", log=lambda _m: None)}
	assert {"maps/slab.bsp", "media/wt_slab.gct"} <= out
	assert sum(1 for o in out if o.startswith("media/sky_testsky")) == 6
	assert struct.unpack_from("<i", (tmp_path / "out" / "maps" / "slab.bsp").read_bytes())[0] == 30
