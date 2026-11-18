"""Source 2 maps (.vmap_c / world.vwrld_c) rebuilt as BSP30, on a generated map."""

import struct

import numpy as np

from gxbake.core import bsp30, s2map, source_bsp, vents, vres
from gxbake.core.project import Kind, Project, Settings, detect_kind

from . import s2_fixtures as s2
from . import s2map_fixtures as f


def _lumps(bsp: bytes) -> list[bytes]:
	return bsp30.read_lumps(bsp)


def _entities(bsp: bytes) -> list[dict]:
	return source_bsp.parse_entities(_lumps(bsp)[bsp30.L_ENTITIES].split(b"\0")[0].decode())


def _hull(bsp: bytes, model: int, hull: int, point) -> int:
	lumps = _lumps(bsp)
	pl = lumps[bsp30.L_PLANES]
	planes = [struct.unpack_from("<4fi", pl, i * 20) for i in range(len(pl) // 20)]
	clip = lumps[bsp30.L_CLIPNODES]
	n = struct.unpack_from("<4i", lumps[bsp30.L_MODELS], model * 64 + 36)[hull]
	while n >= 0:
		p, c0, c1 = struct.unpack_from("<ihh", clip, n * 8)
		nx, ny, nz, d, _t = planes[p]
		n = c0 if nx * point[0] + ny * point[1] + nz * point[2] - d >= 0 else c1
	return n


def test_murmur2_and_binary_entities():
	assert vents.murmur2("classname") == vents.murmur2("ClassName")
	assert vents._BY_HASH[vents.murmur2("origin")] == "origin"
	raw = f.binary_entity({"classname": "info_target", "origin": "1 2 3", "zzz_unknown": "x"})
	got = vents.entity_values({"m_keyValuesData": raw})
	assert got["classname"] == "info_target" and got["origin"] == "1 2 3"
	assert got[f"#{vents.murmur2('zzz_unknown'):08x}"] == "x"
	kv = vents.entity_values({"keyValues3Data": {"version": 1, "values": {"ClassName": "light_omni2",
		"color": [255, 128, 0, 255], "range": 256.0, "enabled": True}}})
	assert kv == {"classname": "light_omni2", "color": "255 128 0 255", "range": "256", "enabled": "1"}
	assert vents.entity_values({"keyValues3Data": {"version": 1, "values": {"origin": "0 0 0"}}}) is None


def test_entity_lump_children(tmp_path):
	child = {"m_name": "child", "m_childLumps": [], "m_entityKeyValues": [
		{"keyValues3Data": {"version": 1, "values": {"classname": "info_target"}}}]}
	parent = {"m_name": "parent", "m_childLumps": ["maps/x/entities/child.vents"], "m_entityKeyValues": [
		{"keyValues3Data": {"version": 1, "values": {"classname": "worldspawn"}}}]}
	d = tmp_path / "maps" / "x" / "entities"
	d.mkdir(parents=True)
	(d / "child.vents_c").write_bytes(s2.resource([("DATA", s2.kv3(child))]))
	(d / "parent.vents_c").write_bytes(s2.resource([("DATA", s2.kv3(parent))]))
	ents = vents.load_entities(d / "parent.vents_c", [tmp_path])
	assert [e["classname"] for e in ents] == ["worldspawn", "info_target"]


def test_merge_coplanar_triangles():
	# a 2x2 grid of squares, each two clockwise-from-above triangles: one convex polygon
	tris = []
	for x in (0, 1):
		for y in (0, 1):
			a, b, c, d = (x, y, 0), (x, y + 1, 0), (x + 1, y + 1, 0), (x + 1, y, 0)
			tris += [(a, b, c), (a, c, d)]
	polys = s2map.merge_coplanar(np.array(tris, float), np.array([0.0, 0.0, 1.0]))
	assert len(polys) == 1 and len(polys[0]) == 4
	assert np.cross(polys[0][1] - polys[0][0], polys[0][2] - polys[0][0])[2] < 0	# still clockwise
	# an L shape can't be one convex polygon
	l_shape = np.array(tris[:6], float)
	assert len(s2map.merge_coplanar(l_shape, np.array([0.0, 0.0, 1.0]))) == 2


def test_cube_sampler_faces():
	faces = [np.full((4, 4, 4), (i * 40, 0, 0, 255), np.uint8) for i in range(6)]
	sample = s2map.cube_face_sampler(faces)
	dirs = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
	assert sample(dirs)[:, 0].round().tolist() == [0, 40, 80, 120, 160, 200]


def test_convert_court(tmp_path):
	vmap = f.write_map(tmp_path)
	assert s2map.is_source2_map(vmap)
	r = s2map.convert(vmap)
	assert r.warnings == ["1 model(s) not found: maps/court/worldnodes/missing.vmdl"]
	assert r.stats["models"] == 2 and r.stats["lights"] == 2 and r.stats["collision pieces"] == 2
	lumps = _lumps(r.bsp)
	# the floor (twice: scene object and raised aggregate fragment), cut at 16 luxels
	verts = np.frombuffer(lumps[bsp30.L_VERTEXES], "<f4").reshape(-1, 3)
	assert {0.0, 128.0} <= set(np.round(verts[:, 2], 3).tolist())
	name, w, h = struct.unpack_from("<16sII", lumps[bsp30.L_TEXTURES], struct.unpack_from("<i", lumps[bsp30.L_TEXTURES], 4)[0])
	assert name.split(b"\0")[0] == b"floor" and (w, h) == (128, 128)	# 64 px per 128 units → 1 texel a unit
	faces = [struct.unpack_from("<HHiHH4Bi", lumps[bsp30.L_FACES], i * 20) for i in range(len(lumps[bsp30.L_FACES]) // 20)]
	assert all(fc[9] >= 0 for fc in faces)	# every face lit
	light = np.frombuffer(lumps[bsp30.L_LIGHTING], np.uint8).reshape(-1, 3)
	assert light.max() < 255 and light.min() > 0
	assert (light[:, 0].astype(int) - light[:, 1] > 20).any()	# the red omni light shows somewhere
	# collision: the slab (hull) and the wall (mesh prism); the trigger copy of the wall is dropped
	assert _hull(r.bsp, 0, 1, (0, 0, 30)) == bsp30.CONTENTS_SOLID
	assert _hull(r.bsp, 0, 1, (0, 0, 40)) == bsp30.CONTENTS_EMPTY
	assert _hull(r.bsp, 0, 1, (0, 245, 64)) == bsp30.CONTENTS_SOLID
	assert _hull(r.bsp, 0, 1, (0, 200, 64)) == bsp30.CONTENTS_EMPTY
	# the door is inline model 1, in world space, with its own hull
	ents = _entities(r.bsp)
	door = next(e for e in ents if e["classname"] == "func_door")
	assert door["model"] == "*1" and "origin" not in door and door["targetname"] == "door1"
	mins = struct.unpack_from("<3f", lumps[bsp30.L_MODELS], 64)
	assert np.allclose(mins, (124, -32, 0))
	assert _hull(r.bsp, 1, 1, (128, 0, 48)) == bsp30.CONTENTS_SOLID
	assert _hull(r.bsp, 1, 1, (128, 0, 200)) == bsp30.CONTENTS_EMPTY
	classes = [e["classname"] for e in ents]
	assert classes[0] == "worldspawn" and ents[0]["skyname"] == "court_sky" and ents[0]["wad"] == ""
	assert "info_player_deathmatch" in classes and "info_player_start" in classes and "env_sky" not in classes
	assert next(e for e in ents if e["classname"] == "cycler_sprite")["model"] == "models/crate.mdl"
	sun = next(e for e in ents if e["classname"] == "light_environment")
	assert sun["pitch"] == "-60" and sun["_light"].endswith(" 200")
	assert r.props == ["models/props/crate.vmdl"]
	# sky sides from the cubemap faces: rt +X, bk +Y, lf -X, ft -Y, up +Z, dn -Z
	assert {k: int(v[8, 8, 0]) for k, v in r.sky_sides.items()} == {"rt": 0, "lf": 40, "bk": 80, "ft": 120,
		"up": 160, "dn": 200}


def test_world_file_and_render_collision(tmp_path):
	vmap = f.write_map(tmp_path, physics=False, sky=False)
	world = vmap.parent / "court" / "world.vwrld_c"
	r = s2map.convert(world, s2map.MapOptions(shadows=False))
	assert any("no world physics" in w for w in r.warnings)
	assert r.sky is None and r.sky_sides == {}
	# the render floors stand in: standing on the raised one
	assert _hull(r.bsp, 0, 1, (0, 0, 128 + 30)) == bsp30.CONTENTS_SOLID
	assert _hull(r.bsp, 0, 1, (0, 0, 128 + 40)) == bsp30.CONTENTS_EMPTY


def test_project_converts_source2_map(tmp_path):
	vmap = f.write_map(tmp_path)
	assert detect_kind(vmap) == Kind.BSP
	proj = Project(Settings())
	proj.add_path(vmap)
	asset = proj.assets[0]
	info = proj.info(asset)
	assert info["Source 2 map"] == "court" and info["Sky"] == "court_sky"
	out = {o.rel for o in proj.convert([asset], tmp_path / "out", log=lambda _m: None)}
	assert {"maps/court.bsp", "media/wt_court.gct"} <= out
	assert sum(1 for o in out if o.startswith("media/sky_court_sky")) == 6
	assert vres.resource_kind(vmap) == "map"
