"""A tiny synthetic compiled Source 2 map, so the Source 2 map converter is tested without game data.

maps/court/world.vwrld_c names one world node and one entity lump. The node
places a floor model (a 512-unit square at z = 0, tiled by materials/floor)
twice: once as a plain scene object and once, raised 128 units, as an
aggregate fragment. world_physics.vphys_c holds the floor slab as a convex
hull and a wall along +Y as a triangle mesh. The entities are a spawn, a sun,
a point light, a sky, a prop and a door whose model lives under
maps/court/entities/.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import s2_fixtures as s2


def vtex_cube(faces: list["np.ndarray"]) -> bytes:
	h, w = faces[0].shape[:2]
	data = struct.pack("<HH4f", 1, 1 << 4, 0.5, 0.5, 0.5, 0.5)
	data += struct.pack("<HHHBBI", w, h, 1, 4, 1, 0)
	data += struct.pack("<II", 8, 0)
	return s2.resource([("DATA", data)], version=1, stream=b"".join(f.astype(np.uint8).tobytes() for f in faces))


def mesh_vmdl(pos, nrm, uv, tris, material: str, name: str = "m") -> bytes:
	"""A static .vmdl_c with one embedded mesh and one draw call."""
	pos, nrm, uv = (np.asarray(a, np.float32) for a in (pos, nrm, uv))
	tris = np.asarray(tris, np.uint16)
	stride = 32
	vb = b"".join(pos[i].tobytes() + nrm[i].tobytes() + uv[i].tobytes() for i in range(len(pos)))
	layout = [
		{"m_pSemanticName": "POSITION", "m_nSemanticIndex": 0, "m_Format": 6, "m_nOffset": 0},
		{"m_pSemanticName": "NORMAL", "m_nSemanticIndex": 0, "m_Format": 6, "m_nOffset": 12},
		{"m_pSemanticName": "TEXCOORD", "m_nSemanticIndex": 0, "m_Format": 16, "m_nOffset": 24},
	]
	mesh = {"m_sceneObjects": [{"m_drawCalls": [{
		"m_nPrimitiveType": "RENDER_PRIM_TRIANGLES", "m_material": material,
		"m_vertexBuffers": [{"m_hBuffer": 0}], "m_indexBuffer": {"m_hBuffer": 0},
		"m_nStartIndex": 0, "m_nIndexCount": int(tris.size), "m_nBaseVertex": 0}]}]}
	data = {"m_name": f"{name}.vmdl", "m_modelSkeleton": {"m_boneName": [], "m_nParent": np.zeros(0, np.int32),
		"m_bonePosParent": [], "m_boneRotParent": []}, "m_refMeshes": []}
	ctrl = {"embedded_meshes": [{"m_Name": name, "m_nMeshIndex": 0, "m_nDataBlock": 2,
		"m_vertexBuffers": [{"m_nElementCount": len(pos), "m_nElementSizeInBytes": stride,
			"m_inputLayoutFields": layout, "m_pData": vb}],
		"m_indexBuffers": [{"m_nElementCount": int(tris.size), "m_nElementSizeInBytes": 2,
			"m_inputLayoutFields": [], "m_pData": tris.astype("<u2").tobytes()}]}]}
	return s2.resource([("DATA", s2.kv3(data)), ("CTRL", s2.kv3(ctrl)), ("MDAT", s2.kv3(mesh))], version=1)


def quad(lo, hi, z: float):
	"""A square facing +Z, counter-clockwise from above, UVs tiling every 128 units."""
	pos = [(lo, lo, z), (hi, lo, z), (hi, hi, z), (lo, hi, z)]
	uv = [((x - lo) / 128, (hi - y) / 128) for x, y, _ in pos]
	return pos, [(0, 0, 1)] * 4, uv, [(0, 1, 2), (0, 2, 3)]


def box(lo, hi):
	"""An axis-aligned box, faces outward, counter-clockwise from outside."""
	lo, hi = np.asarray(lo, float), np.asarray(hi, float)
	pos, nrm, uv, tris = [], [], [], []
	for ax in range(3):
		for sign in (1, -1):
			u, v = [a for a in range(3) if a != ax]
			c = []
			for su, sv in ((0, 0), (1, 0), (1, 1), (0, 1)):
				p = np.where(sign > 0, hi, lo).astype(float).copy()
				p[u] = hi[u] if su else lo[u]
				p[v] = hi[v] if sv else lo[v]
				c.append(p)
			n = np.zeros(3)
			n[ax] = sign
			if np.cross(c[1] - c[0], c[2] - c[0]) @ n < 0:
				c = c[::-1]
			b = len(pos)
			pos += [tuple(x) for x in c]
			nrm += [tuple(n)] * 4
			uv += [(0, 0), (1, 0), (1, 1), (0, 1)]
			tris += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
	return pos, nrm, uv, tris


def binary_entity(values: dict[str, str]) -> bytes:
	from gxbake.core import vents

	out = struct.pack("<III", 1, len(values), 0)
	for k, v in values.items():
		out += struct.pack("<II", vents.murmur2(k), 0x1E) + v.encode() + b"\0"
	return out


def write_map(root: Path, physics: bool = True, sky: bool = True) -> Path:
	root = Path(root)
	(root / "materials").mkdir(parents=True, exist_ok=True)
	y, x = np.mgrid[0:64, 0:64]
	tex = np.zeros((64, 64, 4), np.uint8)
	tex[..., 0] = 80 + (x // 16 % 2) * 120
	tex[..., 1] = 120
	tex[..., 2] = 80 + (y // 16 % 2) * 120
	tex[..., 3] = 255
	(root / "materials" / "floor.vtex_c").write_bytes(s2.vtex_rgba(tex))
	(root / "materials" / "floor.vmat_c").write_bytes(s2.vmat("materials/floor.vtex"))
	sky_faces = [np.full((16, 16, 4), (40 * i, 100, 200, 255), np.uint8) for i in range(6)]
	(root / "materials" / "skybox").mkdir(exist_ok=True)
	(root / "materials" / "skybox" / "court_sky.vtex_c").write_bytes(vtex_cube(sky_faces))
	skymat = {"m_materialName": "materials/skybox/court_sky.vmat", "m_shaderName": "sky.vfx",
		"m_textureParams": [{"m_name": "g_tSkyTexture", "m_pValue": "materials/skybox/court_sky.vtex"}],
		"m_intParams": [], "m_floatParams": []}
	(root / "materials" / "skybox" / "court_sky.vmat_c").write_bytes(s2.resource([("DATA", s2.kv3(skymat))]))

	m = root / "maps" / "court"
	(m / "worldnodes").mkdir(parents=True, exist_ok=True)
	(m / "entities").mkdir(exist_ok=True)
	(m / "worldnodes" / "floor.vmdl_c").write_bytes(mesh_vmdl(*quad(-256, 256, 0), "materials/floor.vmat", "floor"))
	(m / "entities" / "door.vmdl_c").write_bytes(mesh_vmdl(*box((-4, -32, 0), (4, 32, 96)), "materials/floor.vmat", "door"))
	ident = [np.array([1, 0, 0, 0], np.float32), np.array([0, 1, 0, 0], np.float32), np.array([0, 0, 1, 0], np.float32)]
	raised = [np.array([1, 0, 0, 0], np.float32), np.array([0, 1, 0, 0], np.float32), np.array([0, 0, 1, 128], np.float32)]
	node = {
		"m_sceneObjects": [{"m_renderableModel": "maps/court/worldnodes/floor.vmdl", "m_vTransform": ident},
			{"m_renderableModel": "maps/court/worldnodes/missing.vmdl", "m_vTransform": ident}],
		"m_aggregateSceneObjects": [{"m_renderableModel": "maps/court/worldnodes/floor.vmdl",
			"m_aggregateMeshes": [{"m_nDrawCallIndex": 0, "m_bHasTransform": True, "m_nLODGroupMask": 1,
				"m_nLODSetupIndex": 0}],
			"m_fragmentTransforms": [raised]}],
	}
	(m / "worldnodes" / "node000.vwnod_c").write_bytes(s2.resource([("DATA", s2.kv3(node))]))
	ents = [
		{"keyValues3Data": {"version": 1, "values": {"classname": "worldspawn", "skyname": "ignored"}}},
		{"m_keyValuesData": binary_entity({"classname": "info_player_terrorist", "origin": "0 0 40",
			"angles": "0 90 0"})},
		{"keyValues3Data": {"version": 1, "values": {"classname": "light_environment", "angles": "60 30 0",
			"color": [255, 240, 220, 255], "brightness": 1.0}}},
		{"keyValues3Data": {"version": 1, "values": {"classname": "light_omni2", "origin": "100 100 64",
			"color": [255, 0, 0, 255], "brightness": 2.0, "range": 256.0}}},
		{"keyValues3Data": {"version": 1, "values": {"classname": "prop_dynamic", "origin": "50 0 0",
			"model": "models/props/crate.vmdl"}}},
		{"keyValues3Data": {"version": 1, "values": {"classname": "func_door", "origin": "128 0 0",
			"angles": "0 0 0", "model": "maps/court/entities/door.vmdl", "targetname": "door1"}}},
	]
	if sky:
		ents.append({"keyValues3Data": {"version": 1, "values": {"classname": "env_sky",
			"skyname": "materials/skybox/court_sky.vmat"}}})
	lump = {"m_name": "default_ents", "m_childLumps": [], "m_entityKeyValues": ents}
	(m / "entities" / "default_ents.vents_c").write_bytes(s2.resource([("DATA", s2.kv3(lump))]))
	if physics:
		slab = [((0, 0, 1), 0.0), ((0, 0, -1), 16.0), ((1, 0, 0), 256.0), ((-1, 0, 0), 256.0),
			((0, 1, 0), 256.0), ((0, -1, 0), 256.0)]
		wall_v = np.array([(-256, 256, 0), (256, 256, 0), (256, 256, 128), (-256, 256, 128)], np.float32)
		wall_t = np.array([(0, 2, 1), (0, 3, 2)], np.int32)	# facing -Y (into the court)
		phys = {"m_parts": [{"m_rnShape": {
			"m_hulls": [{"m_nCollisionAttributeIndex": 0, "m_Hull": {
				"m_Planes": [{"m_vNormal": list(n), "m_flOffset": d} for n, d in slab]}}],
			"m_meshes": [{"m_nCollisionAttributeIndex": 0, "m_Mesh": {
				"m_Vertices": wall_v.tobytes(), "m_Triangles": wall_t.tobytes()}},
				{"m_nCollisionAttributeIndex": 1, "m_Mesh": {
				"m_Vertices": wall_v.tobytes(), "m_Triangles": wall_t.tobytes()}}]}}],
			"m_collisionAttributes": [{"m_CollisionGroupString": "Default", "m_InteractAsStrings": ["solid"]},
				{"m_CollisionGroupString": "Trigger", "m_InteractAsStrings": ["trigger"]}]}
		(m / "world_physics.vphys_c").write_bytes(s2.resource([("DATA", s2.kv3(phys))]))
	world = {"m_worldNodes": [{"m_nParent": -1, "m_ChildNodeIndices": [],
		"m_worldNodePrefix": "maps\\court\\worldnodes\\node000"}],
		"m_entityLumps": ["maps/court/entities/default_ents.vents"]}
	(m / "world.vwrld_c").write_bytes(s2.resource([("DATA", s2.kv3(world))]))
	vmap = root / "maps" / "court.vmap_c"
	vmap.write_bytes(s2.resource([("DATA", s2.kv3({"m_name": "court"}))]))
	return vmap
