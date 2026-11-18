"""Source 2 maps (.vmap_c / world.vwrld_c) → the port's GoldSrc BSP30.

A compiled Source 2 map is a world (world.vwrld_c) of world nodes
(worldnodes/node*.vwnod_c), each a list of scene objects placing a model or
mesh with a 3x4 transform, plus entity lumps (entities/*.vents_c) and the
world's physics (world_physics.vphys_c or .vmdl_c). It has no BSP tree,
brushes or lightmaps in a form the port can use, so GxBake builds them:

- render geometry: every scene object's LOD 0 draw calls (aggregates too),
  in world space; tool materials are dropped. Coplanar neighbouring
  triangles that share a texture mapping are merged into convex polygons,
  then cut so no lightmap is wider than 16 luxels.
- textures: each material's colour texture, scaled to about one texel per
  world unit (at most max_texture) and embedded as an 8-bit miptex. Texture
  vectors are solved per polygon from the mesh UVs.
- tree: an axis-aligned k-d tree over the faces with empty leafs; every leaf
  sees every other (no PVS).
- collision: physics hull shapes become brushes, physics mesh shapes become
  thin prisms under each polygon; the render polygons stand in when the map
  has no physics. Clip hulls 1-3 are built from those, as for Source 1 maps.
- light: recomputed per luxel from light_environment (sun) and the point
  and spot lights, with shadows traced through the collision, plus ambient.
  Source 2's own lightmaps (BC6H texture arrays addressed by a second UV set)
  are not read.
- brush entities (models under maps/<map>/entities/) become inline models
  "*N" with their own faces and hulls; prop entities become cycler_sprite
  entities naming models/<name>.mdl; spawns and the sky are adapted.

Ported in part from ValveResourceFormat (MIT, see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bsp30, kv3, vents, vmdl, vres, vtex
from .source_map import MapOptions, MapResult, _pow2, _short_name, _subdivide, make_miptex

SPAWN_CLASSES = {"info_player_terrorist", "info_player_counterterrorist", "info_player_teamspawn",
	"info_player_combine", "info_player_rebel", "info_deathmatch_spawn"}
PROP_CLASSES = {"prop_static", "prop_dynamic", "prop_dynamic_override", "prop_physics", "prop_physics_override",
	"prop_physics_multiplayer", "prop_animinteractable", "prop_door_rotating"}
LIGHT_CLASSES = {"light_omni", "light_omni2", "light_spot", "light_barn", "light_rect", "light_capsule", "light_ortho",
	"light"}
SKIP_TAGS = ("trigger", "npcclip", "grenadeclip", "sky", "physics_only", "water")
LEAF_FACES = 48	# faces per k-d leaf
MAX_FACES = 60000
PRISM_DEPTH = 16.0


# -- reading the map ------------------------------------------------------------------


@dataclass
class DrawCall:
	material: str
	positions: "np.ndarray"	# (n, 3) model space
	normals: "np.ndarray"
	uvs: "np.ndarray"
	tris: "np.ndarray"	# (m, 3)


def model_drawcalls(path: Path, roots: list[Path], warnings: list[str]) -> list[DrawCall]:
	"""Every LOD 0 draw call of a .vmdl_c (default mesh group) or .vmesh_c, in draw-call order."""
	res = vres.load(path)
	if res.kind == "mesh":
		data = res.keyvalues() or {}
		vbs, ibs = vmdl._mesh_buffers(res, data, None)
		meshes = [vmdl.MeshData(path.stem, 0, data, vbs, ibs)]
	else:
		d = res.keyvalues() or {}
		meshes = [m for m in vmdl.model_meshes(res, roots, warnings) if vmdl._lod0(d, m.index) and vmdl._default_group(d, m.index)]
	out = []
	for mesh in meshes:
		for so in mesh.data.get("m_sceneObjects") or []:
			for dc in so.get("m_drawCalls") or []:
				mref = str(dc.get("m_material") or dc.get("m_pMaterial") or "")
				prim = dc.get("m_nPrimitiveType", "RENDER_PRIM_TRIANGLES")
				got = vmdl._draw_call(mesh, dc, None, 1) if prim in ("RENDER_PRIM_TRIANGLES", 5, 4) else None
				if got is None:
					out.append(DrawCall(mref, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3), np.int64)))
					continue
				pos, nrm, uv, tris, _top, _share = got
				out.append(DrawCall(mref, pos.astype(np.float64), nrm.astype(np.float64), uv.astype(np.float64), tris))
	return out


def _matrix34(rows) -> "np.ndarray":
	m = np.eye(4)
	a = np.asarray([kv3.floats(r, 4) for r in rows], np.float64)
	m[:3, :4] = a[:3]
	return m


def find_world(path: Path, roots: list[Path]) -> tuple[Path, str]:
	"""(world.vwrld_c, map name) for a .vmap_c or a world.vwrld_c."""
	path = Path(path)
	low = path.name.lower()
	if low.endswith(".vwrld_c"):
		return path, path.parent.name.lower()
	name = path.name.split(".")[0].lower()
	for cand in (path.parent / name / "world.vwrld_c", path.parent / "maps" / name / "world.vwrld_c"):
		if cand.is_file():
			return cand, name
	w = vtex.find_compiled(f"maps/{name}/world.vwrld", roots)
	if w is None:
		raise ValueError(f"{path.name}: maps/{name}/world.vwrld_c not found")
	return w, name


def is_source2_map(path: Path) -> bool:
	low = Path(path).name.lower()
	return low.endswith(".vwrld_c") or low.endswith(".vmap_c")


@dataclass
class _Instance:
	model: Path
	transform: "np.ndarray"
	fragments: list[tuple[int, "np.ndarray"]] | None = None	# aggregate: (draw call, transform)


def _world_instances(world: dict, roots: list[Path], warnings: list[str]) -> list[_Instance]:
	nodes = world.get("m_worldNodes") or []
	out: list[_Instance] = []
	missing: set[str] = set()
	for node in nodes:
		if node.get("m_ChildNodeIndices") is not None and len(node.get("m_ChildNodeIndices")) and len(nodes) > 1:
			continue	# a coarser LOD of its children
		prefix = str(node.get("m_worldNodePrefix", "")).replace("\\", "/")
		npath = vtex.find_compiled(prefix + ".vwnod", roots)
		if npath is None:
			warnings.append(f"world node {prefix} not found")
			continue
		nd = vres.load(npath).keyvalues() or {}
		for so in nd.get("m_sceneObjects") or []:
			ref = str(so.get("m_renderableModel") or so.get("m_renderable") or "")
			if not ref:
				continue
			mp = vtex.find_compiled(ref, roots)
			if mp is None:
				missing.add(ref)
				continue
			tf = so.get("m_vTransform")
			out.append(_Instance(mp, _matrix34(tf) if tf is not None else np.eye(4)))
		for agg in nd.get("m_aggregateSceneObjects") or []:
			ref = str(agg.get("m_renderableModel") or "")
			mp = vtex.find_compiled(ref, roots) if ref else None
			if mp is None:
				if ref:
					missing.add(ref)
				continue
			frags = agg.get("m_aggregateMeshes") or []
			tfs = agg.get("m_fragmentTransforms") or []
			combined: dict[int, int] = {}
			for f in frags:
				s = int(f.get("m_nLODSetupIndex", -1))
				combined[s] = combined.get(s, 0) | int(f.get("m_nLODGroupMask", 1))
			chosen = []
			ti = 0
			for f in frags:
				m = np.eye(4)
				if f.get("m_bHasTransform"):
					if ti < len(tfs):
						m = _matrix34(tfs[ti])
					ti += 1
				mask = int(f.get("m_nLODGroupMask", 1))
				comb = combined[int(f.get("m_nLODSetupIndex", -1))]
				lowest = comb & -comb
				if lowest and not mask & lowest:
					continue
				if "m_nDrawCallIndex" not in f:
					continue
				chosen.append((int(f["m_nDrawCallIndex"]), m))
			out.append(_Instance(mp, np.eye(4), chosen))
	if missing:
		shown = ", ".join(sorted(missing)[:4]) + (f" and {len(missing) - 4} more" if len(missing) > 4 else "")
		warnings.append(f"{len(missing)} model(s) not found: {shown}")
	return out


# -- materials --------------------------------------------------------------------------


@dataclass
class _Material:
	ref: str
	rgba: "np.ndarray"
	masked: bool
	water: bool
	tool: bool
	found: bool


def _load_material(ref: str, roots: list[Path]) -> _Material:
	low = ref.lower()
	if "tools/" in low or "toolsnodraw" in low or low.endswith("nodraw.vmat"):
		return _Material(ref, np.zeros((1, 1, 4), np.uint8), False, False, True, True)
	grey = np.full((64, 64, 4), (128, 128, 128, 255), np.uint8)
	mp = vtex.find_compiled(ref, roots) if ref else None
	if mp is None:
		return _Material(ref, grey, False, False, False, False)
	try:
		vm = vtex.load_vmat(mp)
	except (OSError, ValueError):
		return _Material(ref, grey, False, False, False, False)
	if vm.tool or "sky" in vm.shader.lower():
		return _Material(ref, grey, False, False, True, True)
	water = "water" in vm.shader.lower()
	tref = vm.color_texture()
	tp = vtex.find_compiled(tref, roots) if tref else None
	rgba = None
	if tp is not None:
		try:
			rgba = vtex.load_vtex(tp)
		except (OSError, ValueError, RuntimeError):
			rgba = None
	if rgba is None:
		return _Material(ref, grey, False, water, False, False)
	masked = vm.alpha_test and bool((rgba[..., 3] < 128).any())
	if not masked:
		rgba = rgba.copy()
		rgba[..., 3] = 255
	return _Material(ref, rgba, masked, water, False, True)


# -- polygons -----------------------------------------------------------------------------


def _uv_vecs_many(tris: "np.ndarray", st: "np.ndarray", n: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
	"""_uv_vecs for (m, 3, 3) triangles at once: ((m, 2, 4) vecs, (m,) solvable)."""
	m = np.zeros((len(tris), 4, 4))
	m[:, :3, :3] = tris
	m[:, :3, 3] = 1.0
	m[:, 3, :3] = n
	rhs = np.zeros((len(tris), 4, 2))
	rhs[:, :3] = st
	det = np.linalg.det(m)
	good = np.abs(det) > 1e-9
	out = np.zeros((len(tris), 2, 4))
	if good.any():
		sol = np.linalg.solve(m[good], rhs[good])	# (k, 4, 2)
		out[good] = np.transpose(sol, (0, 2, 1))
	good &= np.isfinite(out).all((1, 2)) & (np.abs(out).max((1, 2)) < 1e6)
	return out, good


def _convex_cw(poly: "np.ndarray", n: "np.ndarray") -> bool:
	"""Clockwise seen from the front of n and convex (collinear corners allowed)."""
	k = len(poly)
	for i in range(k):
		a, b, c = poly[i], poly[(i + 1) % k], poly[(i + 2) % k]
		if np.cross(b - a, c - b) @ n > 1e-6 * max(1.0, float(np.linalg.norm(b - a) * np.linalg.norm(c - b))):
			return False
	return True


def _drop_collinear(poly: "np.ndarray") -> "np.ndarray":
	keep = []
	k = len(poly)
	for i in range(k):
		a, b, c = poly[i - 1], poly[i], poly[(i + 1) % k]
		u, v = b - a, c - b
		if np.linalg.norm(np.cross(u, v)) > 1e-4 * max(1e-9, float(np.linalg.norm(u) * np.linalg.norm(v))):
			keep.append(b)
	return np.array(keep) if len(keep) >= 3 else poly


def _unspike(loop: list[int]) -> list[int]:
	"""Drop back-and-forth runs (... a b a ...) left where two polygons shared several edges."""
	changed = True
	while changed and len(loop) > 3:
		changed = False
		n = len(loop)
		for i in range(n):
			if loop[i - 1] == loop[(i + 1) % n]:
				drop = {i, (i + 1) % n}
				loop = [v for j, v in enumerate(loop) if j not in drop]
				changed = True
				break
	return loop


def merge_coplanar(tris: "np.ndarray", n: "np.ndarray", max_verts: int = 24) -> list["np.ndarray"]:
	"""Greedily merge clockwise triangles of one plane that share edges into convex polygons."""
	keyed = np.round(tris, 2)
	vid: dict[tuple, int] = {}
	verts: list = []
	polys: list[list[int] | None] = []
	for t, kt in zip(tris, keyed):
		ids = []
		for p, kp in zip(t, kt):
			key = tuple(kp)
			if key not in vid:
				vid[key] = len(verts)
				verts.append(p)
			ids.append(vid[key])
		if len(set(ids)) == 3:
			polys.append(ids)
	V = np.array(verts) if verts else np.zeros((0, 3))
	edges: dict[tuple[int, int], int] = {}
	for pi, p in enumerate(polys):
		for k in range(len(p)):
			edges[(p[k], p[(k + 1) % len(p)])] = pi
	changed = True
	while changed:
		changed = False
		for pi in range(len(polys)):
			p = polys[pi]
			if p is None:
				continue
			for k in range(len(p)):
				a, b = p[k], p[(k + 1) % len(p)]
				qi = edges.get((b, a))
				if qi is None or qi == pi or polys[qi] is None:
					continue
				q = polys[qi]
				j = q.index(b)
				if q[(j + 1) % len(q)] != a:
					continue
				# p: ... a b ...  q: ... b a ...  → p up to a, then q after a around to b, then p after b
				merged = p[:k + 1] + [q[(j + 2 + m) % len(q)] for m in range(len(q) - 2)] + p[k + 1:]
				merged = _unspike(merged)
				if len(merged) > max_verts or len(set(merged)) != len(merged):
					continue
				if not _convex_cw(V[merged], n):
					continue
				for m in range(len(p)):
					edges.pop((p[m], p[(m + 1) % len(p)]), None)
				for m in range(len(q)):
					edges.pop((q[m], q[(m + 1) % len(q)]), None)
				polys[pi] = merged
				polys[qi] = None
				for m in range(len(merged)):
					edges[(merged[m], merged[(m + 1) % len(merged)])] = pi
				changed = True
				break
	return [_drop_collinear(V[p]) for p in polys if p is not None]


def _plane_key(n: "np.ndarray", d: float) -> tuple:
	return (round(float(n[0]), 3), round(float(n[1]), 3), round(float(n[2]), 3), round(d, 1))


def _prism(poly: "np.ndarray", n: "np.ndarray", depth: float = PRISM_DEPTH) -> list[tuple["np.ndarray", float]]:
	"""A slab under a polygon (n its front), as outward planes."""
	d = float(n @ poly[0])
	out = [(n.copy(), d), (-n, -(d - depth))]
	centre = poly.mean(0) - n * depth / 2
	for i in range(len(poly)):
		e = poly[(i + 1) % len(poly)] - poly[i]
		sn = np.cross(e, n)
		ln = float(np.linalg.norm(sn))
		if ln < 1e-6:
			continue
		sn /= ln
		sd = float(sn @ poly[i])
		if sn @ centre > sd:
			sn, sd = -sn, -sd
		out.append((sn, sd))
	return out if len(out) >= 5 else []


# -- physics ------------------------------------------------------------------------------


def _blob_or_list(v, dtype, width: int) -> "np.ndarray":
	if v is None:
		return np.zeros((0, width))
	if isinstance(v, (bytes, bytearray)):
		a = np.frombuffer(bytes(v), dtype)
		return a[: len(a) // width * width].reshape(-1, width).astype(np.float64 if dtype != "<i4" else np.int64)
	if isinstance(v, np.ndarray):
		return v.astype(np.float64 if dtype != "<i4" else np.int64).reshape(-1, width)
	return np.asarray(v, np.float64 if dtype != "<i4" else np.int64).reshape(-1, width)


def physics_shapes(d: dict, warnings: list[str]) -> tuple[list[list[tuple["np.ndarray", float]]], list["np.ndarray"]]:
	"""(convex hull brushes, mesh triangles (m, 3, 3)) of a physics aggregate, solid to players."""
	attrs = d.get("m_collisionAttributes") or []

	def skipped(i: int) -> bool:
		if not 0 <= i < len(attrs):
			return False
		a = attrs[i] or {}
		tags = " ".join(str(x) for x in (a.get("m_InteractAsStrings") or []) + [a.get("m_CollisionGroupString", "")])
		return any(t in tags.lower() for t in SKIP_TAGS)

	binds = d.get("m_bindPose") or []
	brushes: list = []
	meshes: list = []
	for pi, part in enumerate(d.get("m_parts") or []):
		m = np.eye(4)
		if pi < len(binds) and binds[pi] is not None and len(kv3.floats(binds[pi])) == 12:
			m[:3, :4] = kv3.floats(binds[pi]).reshape(3, 4)
		shape = part.get("m_rnShape") or {}
		for h in shape.get("m_hulls") or []:
			if skipped(int(h.get("m_nCollisionAttributeIndex", -1))):
				continue
			hull = h.get("m_Hull") or {}
			planes = hull.get("m_Planes")
			if isinstance(planes, list) and planes and isinstance(planes[0], dict):
				pl = [(kv3.floats(p["m_vNormal"], 3), float(p["m_flOffset"])) for p in planes]
			else:
				arr = _blob_or_list(planes, "<f4", 4)
				pl = [(r[:3], float(r[3])) for r in arr]
			sides = []
			for nrm, off in pl:
				nw = m[:3, :3] @ nrm
				ln = float(np.linalg.norm(nw))
				if ln < 1e-9:
					continue
				nw /= ln
				sides.append((nw, off / ln + float(nw @ m[:3, 3])))
			if len(sides) >= 4:
				brushes.append(sides)
		for ms in shape.get("m_meshes") or []:
			if skipped(int(ms.get("m_nCollisionAttributeIndex", -1))):
				continue
			mesh = ms.get("m_Mesh") or {}
			verts = mesh.get("m_Vertices")
			if isinstance(verts, list) and verts and not isinstance(verts[0], (int, float)):
				V = np.asarray([kv3.floats(v, 3) for v in verts])
			else:
				V = _blob_or_list(verts, "<f4", 3)
			tri = mesh.get("m_Triangles")
			if isinstance(tri, list) and tri and isinstance(tri[0], dict):
				T = np.asarray([kv3.ints(t["m_nIndex"])[:3] for t in tri], np.int64)
			else:
				T = _blob_or_list(tri, "<i4", 3).astype(np.int64)
			if not len(T) or not len(V):
				continue
			T = T[(T >= 0).all(1) & (T < len(V)).all(1)]
			W = V @ m[:3, :3].T + m[:3, 3]
			meshes.append(W[T])
	return brushes, meshes


def _physics_of(path: Path, roots: list[Path]) -> dict | None:
	"""The physics aggregate of a .vphys_c, or the one a .vmdl_c embeds or references."""
	try:
		res = vres.load(path)
	except (OSError, ValueError):
		return None
	if res.kind == "physics":
		return res.keyvalues()
	b = res.block("PHYS")
	if b is not None:
		return res.keyvalues("PHYS")
	ctrl = res.keyvalues("CTRL") if res.block("CTRL") else None
	ep = (ctrl or {}).get("embedded_physics")
	if ep and "phys_data_block" in ep:
		return res.keyvalues(int(ep["phys_data_block"]))
	d = res.keyvalues() or {}
	for ref in d.get("m_refPhysicsData") or []:
		pp = vtex.find_compiled(str(ref), roots)
		if pp is not None:
			return _physics_of(pp, roots)
	return None


# -- k-d tree -------------------------------------------------------------------------------


@dataclass
class _Tree:
	nodes: list = field(default_factory=list)	# (plane, c0, c1, mins, maxs, firstface, numfaces)
	leafs: list = field(default_factory=list)	# (contents, visofs, mins, maxs, firstmark, nummark, amb)
	marksurfaces: list = field(default_factory=list)
	order: list = field(default_factory=list)	# output face order (indices into the input faces)


def _ibox(lo, hi) -> tuple[tuple, tuple]:
	lo = np.maximum(np.floor(lo) - 1, -32767).astype(int)
	hi = np.minimum(np.ceil(hi) + 1, 32767).astype(int)
	return tuple(int(x) for x in lo), tuple(int(x) for x in hi)


def build_tree(tree: _Tree, P: bsp30.PlaneTable, centres: "np.ndarray", lo: "np.ndarray", hi: "np.ndarray",
		faces: list[int], face_base: int, visofs: int) -> int:
	"""Append a k-d tree over `faces` (input indices); returns its head node. Faces are placed in
	tree.order so each node's faces (those of its leaf children) are contiguous; face_base is the
	output index of the first face this tree places."""

	def leaf(idx: list[int]) -> int:
		i = len(tree.leafs)
		first = len(tree.marksurfaces)
		tree.leafs.append(None)
		mins, maxs = _ibox(lo[idx].min(0), hi[idx].max(0)) if idx else ((0, 0, 0), (0, 0, 0))
		tree.leafs[i] = [bsp30.CONTENTS_EMPTY, visofs, mins, maxs, first, 0, (0, 0, 0, 0), idx]
		return -1 - i

	def rec(idx: list[int], depth: int) -> int:
		a = np.array(idx, np.int64)
		c = centres[a] if len(a) else np.zeros((0, 3))
		split = None
		if len(a) > LEAF_FACES and depth < 32:
			ext = c.max(0) - c.min(0)
			for ax in np.argsort(-ext):
				v = float(np.median(c[:, ax]))
				front = c[:, ax] >= v
				if front.all():
					v = float(c[:, ax].min() + ext[ax] / 2)
					front = c[:, ax] >= v
				if front.any() and not front.all():
					split = (int(ax), v, front)
					break
		ni = len(tree.nodes)
		tree.nodes.append(None)
		if split is None:
			# a lone leaf still needs a node: split below everything, the far side solid
			p, _f = P.add(np.array([0.0, 0.0, 1.0]), float(lo[a].min(0)[2] - 64) if len(a) else -65536.0)
			kid = leaf(idx)
			mins, maxs = tree.leafs[-1 - kid][2], tree.leafs[-1 - kid][3]
			tree.nodes[ni] = [p, kid, None, mins, maxs, [kid], 0]	# None: the solid leaf 0
			return ni
		ax, v, front = split
		n = np.zeros(3)
		n[ax] = 1.0
		p, _f = P.add(n, v)
		kids = []
		for part in (a[front], a[~front]):
			kids.append(rec([int(x) for x in part], depth + 1) if len(part) > LEAF_FACES else leaf([int(x) for x in part]))
		boxes = []
		for k in kids:
			if k >= 0:
				boxes.append((tree.nodes[k][3], tree.nodes[k][4]))
			else:
				boxes.append((tree.leafs[-1 - k][2], tree.leafs[-1 - k][3]))
		mins = tuple(int(min(b[0][i] for b in boxes)) for i in range(3))
		maxs = tuple(int(max(b[1][i] for b in boxes)) for i in range(3))
		tree.nodes[ni] = [p, kids[0], kids[1], mins, maxs, [k for k in kids if k < 0], 0]
		return ni

	head = rec(list(faces), 0)
	# lay faces out: each node owns the faces of its leaf children
	for ni in range(len(tree.nodes)):
		nd = tree.nodes[ni]
		if not isinstance(nd[5], list):
			continue
		first = face_base + len(tree.order)
		for kid in nd[5]:
			lf = tree.leafs[-1 - kid]
			idx = lf[7]
			out_first = face_base + len(tree.order)
			tree.order += idx
			lf[4] = len(tree.marksurfaces)
			lf[5] = len(idx)
			tree.marksurfaces += range(out_first, out_first + len(idx))
		nd[5] = first
		nd[6] = face_base + len(tree.order) - first
	return head


# -- light ----------------------------------------------------------------------------------


@dataclass
class _Light:
	origin: "np.ndarray | None"	# None: the sun
	direction: "np.ndarray | None"	# the way the light travels (sun, spots)
	color: "np.ndarray"	# linear, 1 = full bright
	range: float = 512.0
	cone: float = -2.0	# cos of the outer cone angle; -2 = no cone


def _angles_forward(angles: str) -> "np.ndarray":
	try:
		pitch, yaw, _roll = (float(x) for x in angles.split()[:3])
	except ValueError:
		pitch, yaw = 90.0, 0.0
	p, y = math.radians(pitch), math.radians(yaw)
	return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])


def _vec(s: str, n: int = 3, default: float = 0.0) -> "np.ndarray":
	parts = (s or "").split()
	out = np.full(n, default)
	for i in range(min(n, len(parts))):
		try:
			out[i] = float(parts[i])
		except ValueError:
			pass
	return out


def lights_from_entities(ents: list[dict[str, str]]) -> list[_Light]:
	out = []
	for e in ents:
		cls = e.get("classname", "")
		if cls not in LIGHT_CLASSES and cls != "light_environment":
			continue
		if e.get("enabled", "1") == "0" or e.get("startdisabled", e.get("start_disabled", "0")) == "1":
			continue
		col = _vec(e.get("color", "255 255 255"), 3, 255.0) / 255.0
		bright = float(_vec(e.get("brightness", "1"), 1, 1.0)[0])
		if cls == "light" and "_light" in e:
			lv = _vec(e["_light"], 4, 200.0)
			col, bright = lv[:3] / 255.0, lv[3] / 200.0
		if cls == "light_environment":
			out.append(_Light(None, _angles_forward(e.get("angles", "90 0 0")), col * bright))
			continue
		rng = float(_vec(e.get("range", "512"), 1, 512.0)[0]) or 512.0
		light = _Light(_vec(e.get("origin", "0 0 0")), None, col * bright, rng)
		if cls in ("light_spot", "light_barn"):
			light.direction = _angles_forward(e.get("angles", "0 0 0"))
			light.cone = math.cos(math.radians(float(_vec(e.get("outerconeangle", "45"), 1, 45.0)[0])))
		out.append(light)
	return out


def light_points(pts: "np.ndarray", nrm: "np.ndarray", lights: list[_Light], ambient: float,
		occluded=None, step: float = 8.0, sun_reach: float = 1536.0) -> "np.ndarray":
	"""Linear RGB light at each point (n, 3); occluded(points) → bool marks samples inside solids."""
	out = np.full((len(pts), 3), float(ambient))
	if not len(pts):
		return out
	for L in lights:
		if L.origin is None:
			to = -L.direction / max(1e-9, float(np.linalg.norm(L.direction)))
			ndl = nrm @ to
			lit = ndl > 0
			if occluded is not None and lit.any():
				idx = np.nonzero(lit)[0]
				base = pts[idx] + nrm[idx] * 1.0
				blocked = np.zeros(len(idx), bool)
				t = step
				while t <= sun_reach and not blocked.all():
					live = ~blocked
					blocked[live] |= occluded(base[live] + to * t)
					t += step
				lit[idx[blocked]] = False
			out += np.where(lit, ndl, 0.0)[:, None] * L.color[None]
			continue
		vec = L.origin[None] - pts
		dist = np.linalg.norm(vec, axis=1)
		near = (dist < L.range) & (dist > 1e-3)
		if not near.any():
			continue
		idx = np.nonzero(near)[0]
		to = vec[idx] / dist[idx, None]
		ndl = np.einsum("ij,ij->i", nrm[idx], to)
		keep = ndl > 0
		if L.cone > -1.5:
			keep &= (-to @ L.direction) >= L.cone
		idx, to, ndl = idx[keep], to[keep], ndl[keep]
		if occluded is not None and len(idx):
			base = pts[idx] + nrm[idx] * 1.0
			blocked = np.zeros(len(idx), bool)
			d = dist[idx]
			t = step
			while (t < d - step).any() and not blocked.all():
				live = ~blocked & (t < d - step)
				blocked[live] |= occluded(base[live] + to[live] * t)
				t += step
			idx, ndl = idx[~blocked], ndl[~blocked]
		att = (1.0 - dist[idx] / L.range) ** 2
		out[idx] += (ndl * att)[:, None] * L.color[None]
	return out


# -- the converter --------------------------------------------------------------------------


@dataclass
class _Poly:
	pts: "np.ndarray"	# clockwise from the front
	normal: "np.ndarray"
	dist: float
	vecs: "np.ndarray"
	mip: int


def _luxels(poly: _Poly) -> tuple["np.ndarray", int, int] | None:
	vecs = poly.vecs
	s = poly.pts @ vecs[0][:3] + vecs[0][3]
	t = poly.pts @ vecs[1][:3] + vecs[1][3]
	bs, bt = math.floor(s.min() / 16), math.floor(t.min() / 16)
	smax = math.ceil(s.max() / 16) - bs + 1
	tmax = math.ceil(t.max() / 16) - bt + 1
	if smax < 1 or tmax < 1 or smax > 64 or tmax > 64:
		return None
	ss, tt = np.meshgrid((bs + np.arange(smax)) * 16.0, (bt + np.arange(tmax)) * 16.0)
	m = np.array([vecs[0][:3], vecs[1][:3], poly.normal], np.float64)
	try:
		inv = np.linalg.inv(m)
	except np.linalg.LinAlgError:
		return None
	rhs = np.stack([ss.ravel() - vecs[0][3], tt.ravel() - vecs[1][3], np.full(ss.size, poly.dist)], 1)
	world = rhs @ inv.T
	# pull samples that fall off the polygon back inside it, so they don't sample behind walls
	centre = poly.pts.mean(0)
	world = centre + (world - centre) * 0.98
	return world, smax, tmax


class _Builder:
	def __init__(self, roots: list[Path], opts: MapOptions, warnings: list[str]):
		self.roots = roots
		self.opts = opts
		self.warnings = warnings
		self.models: dict[str, list[DrawCall]] = {}
		self.materials: dict[str, _Material] = {}
		self.out = bsp30.Bsp30()
		self.names: set[str] = {"sky"}
		self.mip_of: dict[str, int] = {}

	def drawcalls(self, path: Path) -> list[DrawCall]:
		key = str(path).lower()
		if key not in self.models:
			try:
				self.models[key] = model_drawcalls(path, self.roots, self.warnings)
			except (OSError, ValueError, KeyError, IndexError) as exc:
				self.warnings.append(f"{path.name}: {exc}")
				self.models[key] = []
		return self.models[key]

	def material(self, ref: str) -> _Material:
		if ref not in self.materials:
			self.materials[ref] = _load_material(ref, self.roots)
		return self.materials[ref]

	def triangles(self, inst: _Instance) -> list[tuple[str, "np.ndarray", "np.ndarray"]]:
		"""(material, world triangles (m, 3, 3) clockwise from the front, uvs (m, 3, 2))."""
		dcs = self.drawcalls(inst.model)
		jobs = [(dc, inst.transform) for dc in dcs] if inst.fragments is None else \
			[(dcs[i], inst.transform @ m) for i, m in inst.fragments if 0 <= i < len(dcs)]
		out = []
		for dc, m in jobs:
			if not len(dc.tris):
				continue
			pos = dc.positions @ m[:3, :3].T + m[:3, 3]
			nrm = dc.normals @ np.linalg.inv(m[:3, :3])
			t = dc.tris
			a, b, c = pos[t[:, 0]], pos[t[:, 1]], pos[t[:, 2]]
			ccw = np.einsum("ij,ij->i", np.cross(b - a, c - a), nrm[t].sum(1)) >= 0
			if np.linalg.det(m[:3, :3]) < 0:
				ccw = ~ccw
			# the port wants clockwise from the front: reverse the counter-clockwise ones
			t = np.where(ccw[:, None], t[:, ::-1], t)
			out.append((dc.material, pos[t], dc.uvs[t]))
		return out

	def miptex(self, mat: _Material, density: float) -> int:
		if mat.ref in self.mip_of:
			return self.mip_of[mat.ref]
		from PIL import Image

		h0, w0 = mat.rgba.shape[:2]
		k = self.opts.texel_density / max(1e-6, density)
		tw = max(16, _pow2(w0 * k, min(self.opts.min_texture, w0), self.opts.max_texture))
		th = max(16, _pow2(h0 * k, min(self.opts.min_texture, h0), self.opts.max_texture))
		img = np.asarray(Image.fromarray(np.ascontiguousarray(mat.rgba)).resize((tw, th), Image.Resampling.LANCZOS))
		prefix = "{" if mat.masked else "!" if mat.water else ""
		name = Path(mat.ref).stem or "tex"
		self.mip_of[mat.ref] = len(self.out.textures)
		self.out.textures.append(make_miptex(_short_name(name, prefix, self.names), img, mat.masked))
		return self.mip_of[mat.ref]

	def polygons(self, groups: list[tuple[str, "np.ndarray", "np.ndarray"]]) -> list[_Poly]:
		"""Merge, map and cut one model's triangles into faces."""
		by_mat: dict[str, list] = {}
		for ref, tris, uvs in groups:
			by_mat.setdefault(ref, []).append((tris, uvs))
		polys: list[_Poly] = []
		for ref, parts in by_mat.items():
			mat = self.material(ref)
			if mat.tool:
				continue
			tris = np.concatenate([p[0] for p in parts])
			uvs = np.concatenate([p[1] for p in parts])
			n = np.cross(tris[:, 2] - tris[:, 0], tris[:, 1] - tris[:, 0])	# clockwise → front normal
			area = np.linalg.norm(n, axis=1)
			ok = area > 1e-6
			tris, uvs, n, area = tris[ok], uvs[ok], n[ok] / area[ok, None], area[ok]
			if not len(tris):
				continue
			h0, w0 = mat.rgba.shape[:2]
			e1, e2 = uvs[:, 1] - uvs[:, 0], uvs[:, 2] - uvs[:, 0]
			uv_area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) * w0 * h0
			dens = np.sqrt(uv_area / area)
			dens = dens[np.isfinite(dens) & (dens > 1e-4)]
			mip = self.miptex(mat, float(np.median(dens)) if len(dens) else 1.0)
			tex = self.out.textures[mip]
			scale = np.array([tex.width, tex.height], np.float64)
			groups_: dict[tuple, list[int]] = {}
			vecs_of: dict[tuple, np.ndarray] = {}
			V, good = _uv_vecs_many(tris, uvs * scale, n)
			# wrap the offsets so equal mappings shifted by whole tiles group together
			V[:, 0, 3] %= tex.width
			V[:, 1, 3] %= tex.height
			d = np.einsum("ij,ij->i", n, tris[:, 0])
			keys = np.concatenate([np.round(n, 3), np.round(d, 1)[:, None], np.round(V.reshape(-1, 8), 2)], 1)
			for i in np.nonzero(good)[0]:
				key = tuple(keys[i].tolist())
				groups_.setdefault(key, []).append(int(i))
				vecs_of.setdefault(key, V[i])
			for key, idx in groups_.items():
				nn = n[idx[0]]
				vecs = vecs_of[key]
				for poly in merge_coplanar(tris[idx], nn):
					d = float(nn @ poly.mean(0))
					for piece in _subdivide(poly, vecs):
						if len(piece) >= 3:
							polys.append(_Poly(piece, nn, d, vecs, mip))
		return polys


def _bounds_planes(lo: "np.ndarray", hi: "np.ndarray") -> list[tuple["np.ndarray", float]]:
	out = []
	for ax in range(3):
		n = np.zeros(3)
		n[ax] = 1.0
		out.append((n, float(hi[ax])))
		out.append((-n, -float(lo[ax])))
	return out


def convert(path: Path, opts: MapOptions | None = None, log=lambda _m: None) -> MapResult:
	opts = opts or MapOptions()
	path = Path(path)
	roots = vtex.content_roots(path.parent) + [Path(p) for p in opts.search if Path(p).is_dir()]
	warnings: list[str] = []
	world_path, mapname = find_world(path, roots)
	if world_path.parent not in roots:
		roots += vtex.content_roots(world_path.parent)
	world = vres.load(world_path).keyvalues() or {}
	instances = _world_instances(world, roots, warnings)
	ents: list[dict[str, str]] = []
	for lump in world.get("m_entityLumps") or []:
		lp = vtex.find_compiled(str(lump), roots)
		if lp is None:
			warnings.append(f"entity lump {lump} not found")
			continue
		ents += vents.load_entities(lp, roots, warnings)
	B = _Builder(roots, opts, warnings)
	out = B.out
	P = out.planes

	# -- render geometry: world, then brush entities ------------------------------------------
	world_groups = []
	for inst in instances:
		world_groups += B.triangles(inst)
	world_polys = B.polygons(world_groups)
	brush_models: list[tuple[dict, list[_Poly], "np.ndarray", "np.ndarray", Path]] = []
	ent_prefix = f"maps/{mapname}/entities/"
	for e in ents:
		mref = e.get("model", "").replace("\\", "/").lower()
		if not mref.endswith(".vmdl") or not (mref.startswith(ent_prefix) or "/entities/" in mref):
			continue
		mp = vtex.find_compiled(mref, roots)
		if mp is None:
			warnings.append(f"brush entity model {mref} not found")
			continue
		m = np.eye(4)
		org = _vec(e.get("origin", "0 0 0"))
		ang = _vec(e.get("angles", "0 0 0"))
		m[:3, :3] = _rotation(ang)
		m[:3, 3] = org
		groups = B.triangles(_Instance(mp, m))
		allpts = [g[1].reshape(-1, 3) for g in groups]
		raw = [dc.positions @ m[:3, :3].T + m[:3, 3] for dc in B.drawcalls(mp) if len(dc.positions)]
		pts = np.concatenate(raw) if raw else (np.concatenate(allpts) if allpts else org[None])
		brush_models.append((e, B.polygons(groups), pts.min(0), pts.max(0), mp))
	if len(world_polys) > MAX_FACES:
		world_polys.sort(key=lambda p: -float(np.linalg.norm(np.cross(p.pts[1] - p.pts[0], p.pts[2] - p.pts[0]))))
		warnings.append(f"{len(world_polys)} faces exceeds {MAX_FACES}; the smallest {len(world_polys) - MAX_FACES} were dropped")
		world_polys = world_polys[:MAX_FACES]
	log(f"  {len(instances)} scene objects, {len(world_polys)} world faces, {len(brush_models)} brush entities, "
		f"{len(out.textures)} textures")
	missing = sorted(r for r, m in B.materials.items() if not m.found and not m.tool)
	if missing:
		shown = ", ".join(Path(r).stem for r in missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
		warnings.append(f"{len(missing)} material(s) without a colour texture, grey used: {shown}")

	# -- collision ---------------------------------------------------------------------------------
	world_brushes: list = []
	phys = None
	for cand in (f"maps/{mapname}/world_physics.vphys", f"maps/{mapname}/world_physics.vmdl"):
		pp = vtex.find_compiled(cand, roots)
		if pp is not None:
			phys = _physics_of(pp, roots)
			if phys is not None:
				break
	if phys is not None:
		hulls, meshes = physics_shapes(phys, warnings)
		world_brushes += hulls
		world_brushes += _mesh_prisms(meshes)
	else:
		warnings.append("no world physics found; collision built from the render faces")
		world_brushes += [b for p in world_polys if (b := _prism(p.pts, p.normal))]
	if len(world_brushes) > opts.max_collision:
		def size(b):
			pts = bsp30.brush_points(b)
			return 0.0 if pts is None else float(np.prod(np.maximum(pts.max(0) - pts.min(0), 1.0)))
		world_brushes.sort(key=size, reverse=True)
		warnings.append(f"{len(world_brushes)} collision pieces; kept the largest {opts.max_collision}")
		world_brushes = world_brushes[:opts.max_collision]

	# -- light ---------------------------------------------------------------------------------------
	lights = lights_from_entities(ents)
	if not lights:
		warnings.append("no lights found; the map is lit by ambient light only")
	occluded = None
	if opts.shadows and world_brushes and lights:
		sp = bsp30.PlaneTable()
		scn: list[list[int]] = []
		shead = bsp30.build_clip_hull(world_brushes, 0, sp, scn)

		def occluded(q, _cn=scn, _sp=sp, _h=shead):
			return bsp30.point_contents_many(_cn, _sp, _h, q) == bsp30.CONTENTS_SOLID
	all_polys = list(world_polys) + [p for bm in brush_models for p in bm[1]]
	grids = []
	pts_all, nrm_all = [], []
	for p in all_polys:
		g = _luxels(p)
		grids.append(g)
		if g is not None:
			pts_all.append(g[0])
			nrm_all.append(np.repeat(p.normal[None], len(g[0]), 0))
	if pts_all:
		lit = light_points(np.concatenate(pts_all), np.concatenate(nrm_all), lights, opts.ambient, occluded)
		# a soft shoulder instead of clipping: full sun plus ambient lands near 190, not 255
		lit = np.clip((1.0 - np.exp(-1.2 * lit * opts.light_scale)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
	else:
		lit = np.zeros((0, 3), np.uint8)

	# -- faces, tree, models ------------------------------------------------------------------------
	texinfo_index: dict[tuple, int] = {}

	def texinfo(vecs, mip: int) -> int:
		key = (mip, tuple(np.round(np.asarray(vecs).ravel(), 4)))
		i = texinfo_index.get(key)
		if i is None:
			i = texinfo_index[key] = len(out.texinfo)
			out.texinfo.append((np.asarray(vecs, np.float64).reshape(2, 4), mip, 0))
		return i

	vindex: dict[tuple, int] = {}
	eindex: dict[tuple[int, int], int] = {}

	def vid(p) -> int:
		key = (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))
		i = vindex.get(key)
		if i is None:
			i = vindex[key] = len(out.vertexes)
			out.vertexes.append(key)
		return i

	def eid(a: int, b: int) -> int:
		if (b, a) in eindex:
			return -eindex[(b, a)]
		i = eindex.get((a, b))
		if i is None:
			i = eindex[(a, b)] = len(out.edges)
			out.edges.append((a, b))
		return i

	light_ofs = {}
	pos = 0
	for k, g in enumerate(grids):
		if g is not None:
			light_ofs[k] = (pos, len(g[0]))
			pos += len(g[0])

	def emit_face(k: int):
		p = all_polys[k]
		ids = [vid(q) for q in p.pts]
		ids = [x for j, x in enumerate(ids) if x != ids[j - 1]]
		if len(ids) < 3:
			ids = (ids * 3)[:3]
		first = len(out.surfedges)
		for j in range(len(ids)):
			out.surfedges.append(eid(ids[j], ids[(j + 1) % len(ids)]))
		pidx, flipped = P.add(p.normal, p.dist)
		lo = -1
		styles = (255, 255, 255, 255)
		if k in light_ofs:
			a, n = light_ofs[k]
			lo = len(out.lighting)
			out.lighting += lit[a:a + n].tobytes()
			styles = (0, 255, 255, 255)
		out.faces.append((pidx, 1 if flipped else 0, first, len(ids), texinfo(p.vecs, p.mip), styles, lo))

	def poly_boxes(polys_idx):
		c = np.array([all_polys[k].pts.mean(0) for k in polys_idx]) if polys_idx else np.zeros((0, 3))
		lo = np.array([all_polys[k].pts.min(0) for k in polys_idx]) if polys_idx else np.zeros((0, 3))
		hi = np.array([all_polys[k].pts.max(0) for k in polys_idx]) if polys_idx else np.zeros((0, 3))
		return c, lo, hi

	tree = _Tree()
	tree.leafs.append([bsp30.CONTENTS_SOLID, -1, (0, 0, 0), (0, 0, 0), 0, 0, (0, 0, 0, 0), []])
	visleafs_holder = {}

	def add_model(face_idx: list[int], visofs: int, brushes, lo_box, hi_box, entity: bool):
		# local index arrays: build_tree indexes into these
		c, lo, hi = poly_boxes(face_idx)
		base = len(out.faces)
		local = _Tree(nodes=[], leafs=[], marksurfaces=[], order=[])
		head = build_tree(local, P, c, lo, hi, list(range(len(face_idx))), base, visofs)
		node_base = len(tree.nodes)
		leaf_base = len(tree.leafs)
		mark_base = len(out.marksurfaces)
		for nd in local.nodes:
			p, c0, c1, mins, maxs, ff, nf = nd
			c0 = c0 + node_base if c0 >= 0 else c0 - leaf_base
			c1 = -1 if c1 is None else (c1 + node_base if c1 >= 0 else c1 - leaf_base)
			tree.nodes.append((p, c0, c1, mins, maxs, ff, nf))
		for lf in local.leafs:
			tree.leafs.append((lf[0], lf[1], lf[2], lf[3], lf[4] + mark_base, lf[5], lf[6]))
		out.marksurfaces += local.marksurfaces
		for j in local.order:
			emit_face(face_idx[j])
		heads = [head + node_base, 0, 0, 0]
		for h in (1, 2, 3):
			heads[h] = bsp30.build_clip_hull(brushes if opts.hulls else [], h, P, out.clipnodes)
		if face_idx:
			allp = np.concatenate([all_polys[k].pts for k in face_idx])
			lo_box = np.minimum(lo_box, allp.min(0)) if lo_box is not None else allp.min(0)
			hi_box = np.maximum(hi_box, allp.max(0)) if hi_box is not None else allp.max(0)
		if lo_box is None:
			lo_box = hi_box = np.zeros(3)
		vl = len(local.leafs) if not entity else 0
		out.models.append((tuple(float(x) for x in lo_box), tuple(float(x) for x in hi_box), (0.0, 0.0, 0.0),
			tuple(heads), vl, base, len(out.faces) - base))
		return vl

	nworld = len(world_polys)
	visleafs = add_model(list(range(nworld)), 0, world_brushes, None, None, False)
	visleafs_holder["n"] = visleafs
	row = bytes([0xFF]) * ((visleafs + 7) // 8)
	if visleafs % 8:
		row = row[:-1] + bytes([(1 << (visleafs % 8)) - 1])
	out.visibility = bsp30.compress_vis(row)
	k = nworld
	for e, polys, lo_b, hi_b, mp in brush_models:
		idx = list(range(k, k + len(polys)))
		k += len(polys)
		brushes = []
		bphys = _physics_of(mp, roots) if opts.hulls else None
		if bphys is not None:
			h_, m_ = physics_shapes(bphys, warnings)
			# entity physics is in model space
			m = np.eye(4)
			m[:3, :3] = _rotation(_vec(e.get("angles", "0 0 0")))
			m[:3, 3] = _vec(e.get("origin", "0 0 0"))
			brushes = [_transform_brush(b, m) for b in h_] + [_transform_brush(b, m) for b in _mesh_prisms(m_)]
		elif polys:
			brushes = [b for p in polys if (b := _prism(p.pts, p.normal))]
		elif opts.hulls:
			brushes = [_bounds_planes(lo_b, hi_b)]
		cls = e.get("classname", "")
		if cls.startswith("trigger_") or cls in ("func_buyzone", "func_bomb_target", "func_hostage_rescue"):
			brushes = []	# touch volumes collide by bounds, not hulls
		e["model"] = f"*{len(out.models)}"
		add_model(idx, -1, brushes, lo_b, hi_b, True)
		e.pop("origin", None)
		e.pop("angles", None)
	out.nodes = [tuple(n) for n in tree.nodes]
	out.leafs = [tuple(lf[:7]) for lf in tree.leafs]

	# -- entities -----------------------------------------------------------------------------------
	sky, sky_sides = _sky(ents, roots, warnings)
	props: list[str] = []
	out_ents: list[dict[str, str]] = [{"classname": "worldspawn", "wad": "", "message": mapname}]
	if sky:
		out_ents[0]["skyname"] = sky
	has_start = any(e.get("classname") == "info_player_start" for e in ents)
	first_spawn = None
	for e in ents:
		cls = e.get("classname", "")
		e = {k: v for k, v in e.items() if not k.startswith("#") and len(v) < 1000}
		if cls == "worldspawn":
			for kk, vv in e.items():
				if kk not in ("classname", "skyname", "wad") and not kk.startswith("_"):
					out_ents[0].setdefault(kk, vv)
			continue
		if cls in SPAWN_CLASSES:
			e["classname"] = "info_player_deathmatch"
			first_spawn = first_spawn or e
		elif cls in PROP_CLASSES:
			if not opts.props:
				continue
			props.append(e.get("model", ""))
			e = {"classname": "cycler_sprite", "model": f"models/{Path(e.get('model', '')).stem.lower()}.mdl",
				"origin": e.get("origin", "0 0 0"), "angles": e.get("angles", "0 0 0"), "framerate": "0",
				**({"targetname": e["targetname"]} if "targetname" in e else {})}
		elif cls in LIGHT_CLASSES:
			lv = _vec(e.get("color", "255 255 255"), 3, 255.0)
			e = {"classname": "light", "origin": e.get("origin", "0 0 0"),
				"_light": f"{int(lv[0])} {int(lv[1])} {int(lv[2])} {int(200 * float(_vec(e.get('brightness', '1'), 1, 1)[0]))}"}
		elif cls == "light_environment":
			lv = _vec(e.get("color", "255 255 255"), 3, 255.0)
			pitch = float(_vec(e.get("angles", "90 0 0"))[0])
			e = {"classname": "light_environment", "angles": e.get("angles", "0 0 0"), "pitch": f"{-pitch:g}",
				"_light": f"{int(lv[0])} {int(lv[1])} {int(lv[2])} {int(200 * float(_vec(e.get('brightness', '1'), 1, 1)[0]))}"}
		elif cls == "env_sky":
			continue
		out_ents.append(e)
	if not has_start and first_spawn is not None:
		out_ents.append({"classname": "info_player_start", "origin": first_spawn.get("origin", "0 0 0"),
			"angles": first_spawn.get("angles", "0 0 0")})
	from .source_bsp import format_entities

	out.entities = format_entities(out_ents)
	problems = out.check()
	if problems:
		raise ValueError(f"{path.name}: " + "; ".join(problems))
	log(f"  {len(out.clipnodes)} clipnodes from {len(world_brushes)} collision pieces, {len(lights)} lights")
	stats = {"faces": len(out.faces), "nodes": len(out.nodes), "leafs": len(out.leafs),
		"clipnodes": len(out.clipnodes), "textures": len(out.textures), "texinfo": len(out.texinfo),
		"lighting": len(out.lighting), "models": len(out.models), "collision pieces": len(world_brushes),
		"lights": len(lights)}
	return MapResult(out.write(), warnings, sky, sky_sides, sorted(set(p for p in props if p)), stats)


def _rotation(angles: "np.ndarray") -> "np.ndarray":
	"""Source (pitch, yaw, roll) degrees → 3x3 rotation, as AngleMatrix builds it."""
	p, y, r = (math.radians(float(a)) for a in angles[:3])
	sp, cp = math.sin(p), math.cos(p)
	sy, cy = math.sin(y), math.cos(y)
	sr, cr = math.sin(r), math.cos(r)
	return np.array([
		[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
		[cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
		[-sp, sr * cp, cr * cp],
	])


def _transform_brush(b, m: "np.ndarray"):
	out = []
	for n, d in b:
		nw = m[:3, :3] @ n
		out.append((nw, d + float(nw @ m[:3, 3])))
	return out


def _mesh_prisms(meshes: list["np.ndarray"]) -> list:
	out = []
	for tris in meshes:
		# collision triangles are counter-clockwise from outside; the port's polygons clockwise
		cw = tris[:, ::-1]
		n = np.cross(cw[:, 2] - cw[:, 0], cw[:, 1] - cw[:, 0])
		ln = np.linalg.norm(n, axis=1)
		ok = ln > 1e-6
		cw, n = cw[ok], n[ok] / ln[ok, None]
		groups: dict[tuple, list[int]] = {}
		for i in range(len(cw)):
			groups.setdefault(_plane_key(n[i], float(n[i] @ cw[i][0])), []).append(i)
		for idx in groups.values():
			nn = n[idx[0]]
			for poly in merge_coplanar(cw[idx], nn):
				b = _prism(poly, nn)
				if b:
					out.append(b)
	return out


# -- sky --------------------------------------------------------------------------------------------


def cube_face_sampler(faces: list["np.ndarray"]):
	"""Direction (n, 3) → RGB from cube faces in +X -X +Y -Y +Z -Z order (GL face conventions), world axes."""
	fs = [f[..., :3].astype(np.float32) for f in faces]

	def sample(d: "np.ndarray") -> "np.ndarray":
		d = np.asarray(d, np.float64)
		ax = np.abs(d)
		out = np.zeros((len(d), 3), np.float32)
		major = np.argmax(ax, 1)
		for face in range(6):
			a = face // 2
			pos = face % 2 == 0
			sel = (major == a) & ((d[:, a] > 0) == pos)
			if not sel.any():
				continue
			x, y, z = d[sel, 0], d[sel, 1], d[sel, 2]
			ma = ax[sel, a]
			sc, tc = {
				0: (-z, -y), 1: (z, -y), 2: (x, z), 3: (x, -z), 4: (x, -y), 5: (-x, -y),
			}[face]
			img = fs[face]
			h, w = img.shape[:2]
			u = np.clip(((sc / ma + 1) / 2) * w - 0.5, 0, w - 1)
			v = np.clip(((tc / ma + 1) / 2) * h - 0.5, 0, h - 1)
			u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
			u1, v1 = np.minimum(u0 + 1, w - 1), np.minimum(v0 + 1, h - 1)
			fu, fv = (u - u0)[:, None], (v - v0)[:, None]
			out[sel] = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u1] * fu * (1 - fv) +
				img[v1, u0] * (1 - fu) * fv + img[v1, u1] * fu * fv)
		return out
	return sample


def cube_to_sides(tex: "vtex.VTex", size: int = 256) -> dict[str, "np.ndarray"]:
	"""A cubemap vtex rendered as the port's six sky sides (RGBA, engine orientation)."""
	from .skybox import SIDES, side_directions

	faces = [tex.rgba(face=f) for f in range(6)]
	sample = cube_face_sampler(faces)
	out = {}
	for side in SIDES:
		d = side_directions(side, size)
		rgb = np.clip(sample(d).reshape(size, size, 3) + 0.5, 0, 255).astype(np.uint8)
		out[side] = np.dstack([rgb, np.full((size, size), 255, np.uint8)])
	return out


def _sky(ents: list[dict[str, str]], roots: list[Path], warnings: list[str]) -> tuple[str | None, dict]:
	for e in ents:
		if e.get("classname") != "env_sky":
			continue
		ref = e.get("skyname", "")
		if not ref:
			continue
		name = Path(ref.replace("\\", "/")).stem.lower()
		for suffix in ("_hdr", "_ldr"):
			name = name.removesuffix(suffix)
		name = name[:20] or "sky"
		mp = vtex.find_compiled(ref, roots)
		if mp is None:
			warnings.append(f"sky material {ref} not found")
			return name, {}
		try:
			vm = vtex.load_vmat(mp)
		except (OSError, ValueError) as exc:
			warnings.append(f"sky material {ref}: {exc}")
			return name, {}
		tref = next((vm.textures[k] for k in ("g_tSkyTexture", "g_tSkyTextureLDR", "g_tSkyTextureHDR") if vm.textures.get(k)),
			None) or next((v for k, v in vm.textures.items() if "sky" in k.lower()), None)
		tp = vtex.find_compiled(tref, roots) if tref else None
		if tp is None:
			warnings.append(f"sky texture {tref or '(none)'} not found")
			return name, {}
		try:
			tex = vtex.load(tp)
			if not tex.cube:
				warnings.append(f"sky texture {Path(tref).name} is not a cubemap")
				return name, {}
			return name, cube_to_sides(tex)
		except (OSError, ValueError, RuntimeError) as exc:
			warnings.append(f"sky texture {Path(tref).name}: {exc}")
			return name, {}
	return None, {}


def map_info(path: Path) -> dict:
	roots = vtex.content_roots(Path(path).parent)
	try:
		world_path, name = find_world(Path(path), roots)
	except ValueError as exc:
		return {"Problem": str(exc)}
	roots += vtex.content_roots(world_path.parent)
	w = vres.load(world_path).keyvalues() or {}
	nodes = w.get("m_worldNodes") or []
	objects = 0
	for node in nodes:
		np_ = vtex.find_compiled(str(node.get("m_worldNodePrefix", "")).replace("\\", "/") + ".vwnod", roots)
		if np_ is not None:
			nd = vres.load(np_).keyvalues() or {}
			objects += len(nd.get("m_sceneObjects") or []) + len(nd.get("m_aggregateSceneObjects") or [])
	return {"Source 2 map": name, "World nodes": len(nodes), "Scene objects": objects,
		"Entity lumps": len(w.get("m_entityLumps") or [])}

