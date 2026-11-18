"""Source model sources in DMX (Blender Source Tools, the SFM, Source 2 ModelDoc exports) → models.ImportedModel.

A DmeModel holds a DAG of DmeDag and DmeJoint nodes, each with a DmeTransform
(position, orientation quaternion x y z w). Nodes with a "shape" carry a DmeMesh:
its bind DmeVertexData stores each field (positions, normals,
textureCoordinates, jointWeights/jointIndices, or Source 2's position$0,
normal$0, texcoord$0, blendweights$0/blendindices$0) as a value array plus a
per-corner index array, and its DmeFaceSets list polygons as corner indices
ended by -1, each with a DmeMaterial. Studio skinning is rigid, so each vertex
follows its heaviest joint. An animationList of DmeChannelsClips becomes
sequences, sampled from the clips' position and orientation logs. Flexes
(deltaStates) are not carried over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import dmx
from . import skeleton as _sk

_FIELDS = {
	"pos": ("positions", "position$0"),
	"norm": ("normals", "normal$0"),
	"uv": ("textureCoordinates", "texcoord$0"),
	"weight": ("jointWeights", "blendweights$0"),
	"index": ("jointIndices", "blendindices$0"),
}
_IMAGE_EXTS = (".vtf", ".tga", ".png", ".jpg", ".bmp", ".psd", ".dds")


def is_dmx_model(path: Path) -> bool:
	try:
		with Path(path).open("rb") as f:
			hd = dmx.header(f.read(200))
	except OSError:
		return False
	return hd is not None and hd[2] in ("model", "sfm_session", "dmx", "sfm")


def _transform(t) -> np.ndarray:
	m = np.eye(4)
	if t is None:
		return m
	m[:3, :3] = _sk.quat_matrix(np.asarray(t.get("orientation", (0, 0, 0, 1)), float))
	m[:3, 3] = t.get("position", (0, 0, 0))
	return m


def _field(vd, key: str):
	for name in _FIELDS[key]:
		if name in vd.attrs:
			return np.asarray(vd[name]), np.asarray(vd.get(name + "Indices", []), np.int64)
	return None, None


class _Space:
	"""Source scene space → GoldSrc model space: an axis swap for Y-up files, and the import scale."""

	def __init__(self, up_axis: int, scale: float):
		self.c = np.eye(3)
		if up_axis == 2:	# Y-up → Z-up (+Z forward becomes +X forward)
			self.c = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
		self.s = scale

	def matrix(self, m: np.ndarray) -> np.ndarray:
		out = np.eye(4)
		out[:3, :3] = self.c @ m[:3, :3] @ self.c.T
		out[:3, 3] = self.s * (self.c @ m[:3, 3])
		return out


def _skin(name: str, path: Path, roots: list[Path], opts, warnings: list[str]):
	from . import models as _m
	from .source_mdl import source_skin

	base = name.replace("\\", "/").removeprefix("materials/").rsplit(".", 1)[0] if name.lower().endswith(
		(".vmt", ".vmat")) else name.replace("\\", "/").removeprefix("materials/")
	if roots:
		found: list[str] = []
		mat = source_skin(base, [""], roots, opts, found)
		if not found:
			return mat
	stem = Path(base).name.lower()
	for d in (path.parent, path.parent / "materials", path.parent / "textures"):
		try:
			for f in d.iterdir():
				if f.stem.lower() == stem and f.suffix.lower() in _IMAGE_EXTS:
					from . import images

					rgba = images.load_rgba(f)
					rgba = _m._fit_skin(rgba, opts.skin_max_edge)
					return _m.Material(Path(base).name, rgba, bool((rgba[..., 3] < opts.alpha_threshold).any()), f.name)
		except OSError:
			continue
	return source_skin(base, [""], roots, opts, warnings)


def import_dmx(path: Path, opts=None):
	from . import models as _m
	from .vtf import find_materials_root

	opts = opts or _m.ModelOptions()
	path = Path(path)
	dm = dmx.load(path)
	root = dm.root
	if root is None:
		raise ValueError(f"{path.name}: empty DMX")
	model_el = root.get("model") or root.get("skeleton") or next(iter(dm.of_type("DmeModel")), None)
	if model_el is None:
		raise ValueError(f"{path.name}: no DmeModel in this DMX ({dm.format} {dm.format_version})")
	warnings: list[str] = []
	axis = model_el.get("axisSystem")
	space = _Space(int(axis.get("upAxis", 3)) if axis is not None else 3, float(opts.scale))

	# walk the DAG: world matrices, joints in parent-first order, meshes with their node's matrix
	joints: list = []	# (node, parent joint index, world bind, matrix from the parent joint to the node's parent)
	joint_of: dict[int, int] = {}
	meshes: list = []	# (DmeMesh, world, owning joint)
	seen: set[int] = set()

	def walk(node, parent_world: np.ndarray, parent_joint: int, since_joint: np.ndarray) -> None:
		if node is None or id(node) in seen:
			return
		seen.add(id(node))
		local = _transform(node.get("transform"))
		world = parent_world @ local
		pj, since = parent_joint, since_joint @ local
		if node.type == "DmeJoint":
			joint_of[id(node)] = len(joints)
			joints.append((node, parent_joint, world, since_joint))
			pj, since = len(joints) - 1, np.eye(4)
		shape = node.get("shape")
		if shape is not None and shape.type == "DmeMesh":
			meshes.append((shape, world, pj))
		for child in node.get("children") or []:
			walk(child, world, pj, since)

	base = _transform(model_el.get("transform"))
	for child in model_el.get("children") or []:
		walk(child, base, -1, base)
	joint_list = [j for j in (model_el.get("jointList") or []) if j is not None]
	list_to_bone = [joint_of.get(id(j), -1) for j in joint_list]

	static = bool(opts.static) or not joints
	if len(joints) > _sk.MAXSTUDIOBONES:
		warnings.append(f"{len(joints)} joints exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static instead")
		static = True
	world_bind = [space.matrix(w) for _, _, w, _ in joints]
	if static:
		bones = [_m.Bone("root", -1, np.zeros(6))]
	else:
		bones = []
		for i, (node, parent, _w, _since) in enumerate(joints):
			rel = world_bind[i] if parent < 0 else _sk.rigid_inverse(world_bind[parent]) @ world_bind[i]
			bones.append(_m.Bone((node.name or f"joint{i}")[:31], parent, _sk.pose_from_matrix(rel)))
	inv_bind = [_sk.rigid_inverse(w) for w in world_bind]

	roots = [r for r in (find_materials_root(path.parent),) if r is not None]
	materials: list = []
	mat_index: dict[str, int] = {}

	def material(fs) -> int:
		m = fs.get("material")
		name = (m.get("mtlName") if m is not None else None) or (m.name if m is not None else "") or "default"
		if name not in mat_index:
			mat_index[name] = len(materials)
			materials.append(_skin(name, path, roots, opts, warnings))
		return mat_index[name]

	model = _m.ImportedModel(path.stem, materials, [], warnings)
	model.bones = bones
	flips = total = 0
	pending = []
	snapped = 0
	for mesh, world, owner in meshes:
		vd = mesh.get("bindState") or mesh.get("currentState")
		if vd is None:
			continue
		pos, pos_i = _field(vd, "pos")
		if pos is None or not len(pos_i):
			continue
		nrm, nrm_i = _field(vd, "norm")
		uv, uv_i = _field(vd, "uv")
		wts, _ = _field(vd, "weight")
		jix, _ = _field(vd, "index")
		count = int(vd.get("jointCount", 0) or 0)
		w3 = world[:3, :3]
		pos_w = (pos.reshape(-1, 3) @ w3.T + world[:3, 3]) @ space.c.T * space.s
		nrm_w = (nrm.reshape(-1, 3) @ w3.T) @ space.c.T if nrm is not None and len(nrm) else None
		# heaviest joint per position
		vb = np.full(len(pos_w), max(owner, 0), np.int64)
		if not static and count and wts is not None and jix is not None and len(wts) >= len(pos_w) * count:
			w = wts[:len(pos_w) * count].reshape(-1, count)
			ji = jix[:len(pos_w) * count].reshape(-1, count).astype(np.int64)
			top = np.argmax(w, 1)
			pick = ji[np.arange(len(ji)), top]
			mapped = np.array([list_to_bone[k] if 0 <= k < len(list_to_bone) else -1 for k in pick], np.int64)
			vb = np.where(mapped >= 0, mapped, vb)
			snapped += int(((w > 0).sum(1) > 1).sum())
		for fs in mesh.get("faceSets") or []:
			faces = list(fs.get("faces") or [])
			tris: list[tuple[int, int, int]] = []
			poly: list[int] = []
			for c in faces + [-1]:
				if c < 0:
					tris += [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]
					poly = []
				else:
					poly.append(int(c))
			if not tris:
				continue
			corners = np.array(tris, np.int64)
			used, inv = np.unique(corners, return_inverse=True)
			if used.max() >= len(pos_i):
				warnings.append(f"{mesh.name}: face corners out of range; face set skipped")
				continue
			pi = pos_i[used]
			p = pos_w[pi]
			n = nrm_w[nrm_i[used]] if nrm_w is not None and len(nrm_i) > used.max() else np.zeros_like(p)
			t = uv.reshape(-1, 2)[uv_i[used]].astype(np.float64) if uv is not None and len(uv_i) > used.max() \
				else np.zeros((len(used), 2))
			t = np.stack([t[:, 0], 1.0 - t[:, 1]], 1)	# bottom-left origin → top-left
			local_tris = inv.reshape(-1, 3)
			a, b, c = p[local_tris[:, 0]], p[local_tris[:, 1]], p[local_tris[:, 2]]
			agree = np.einsum("ij,ij->i", np.cross(b - a, c - a), n[local_tris].sum(1))
			flips += int((agree < 0).sum())
			total += int((agree != 0).sum())
			pending.append((material(fs), p, n, t, local_tris, vb[pi]))
	if snapped and not static:
		warnings.append(f"{snapped} vertices had blended weights; each now follows its heaviest joint")
	if mesh_count := sum(1 for m, _, _ in meshes if m.get("deltaStates")):
		warnings.append(f"{mesh_count} mesh(es) have flexes (shape keys); only the base shape is kept")

	flip = flips * 2 > total
	for mat_i, p, n, t, tris, vbone in pending:
		if flip:
			tris = tris[:, [0, 2, 1]]
		if static:
			local, ln, vb_out = p, n, np.zeros(len(p), np.int32)
		else:
			ib = np.array([inv_bind[k] for k in vbone])
			local = np.einsum("nij,nj->ni", ib[:, :3, :3], p) + ib[:, :3, 3]
			ln = np.einsum("nij,nj->ni", ib[:, :3, :3], n)
			vb_out = vbone.astype(np.int32)
		lens = np.linalg.norm(ln, axis=1, keepdims=True)
		ln = np.where(lens > 1e-8, ln / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		model.meshes.append(_m.Submesh(mat_i, local.astype(np.float32), ln.astype(np.float32),
			t.astype(np.float32), tris.astype(np.uint32), vb_out, p.astype(np.float32)))
	if not model.meshes:
		raise ValueError(f"{path.name}: no meshes in this DMX")
	if not static:
		model.sequences = _sequences(root, joints, space, opts, warnings)
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _sequences(root, joints, space: _Space, opts, warnings: list[str]) -> list:
	"""DmeChannelsClips → studio sequences, sampling each joint's position and orientation logs."""
	from . import models as _m

	alist = root.get("animationList")
	clips = [c for c in (alist.get("animations") or []) if c is not None] if alist is not None else []
	by_transform = {id(node.get("transform")): i for i, (node, *_r) in enumerate(joints) if node.get("transform")}
	out = []
	used: set[str] = set()
	for clip in clips:
		tracks: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
		for ch in clip.get("channels") or []:
			if ch is None:
				continue
			target = ch.get("toElement")
			attr = ch.get("toAttribute")
			log = ch.get("log")
			if target is None or attr not in ("position", "orientation") or log is None:
				continue
			bone = by_transform.get(id(target))
			layers = log.get("layers") or []
			if bone is None or not layers or layers[0] is None:
				continue
			times = np.asarray(layers[0].get("times") or [], float)
			values = np.asarray(layers[0].get("values") or [], float)
			if len(times) and len(values) >= len(times):
				tracks[(bone, attr)] = (times, values[:len(times)])
		if not tracks:
			continue
		t0 = min(t[0] for t, _ in tracks.values())
		t1 = max(t[-1] for t, _ in tracks.values())
		tf = clip.get("timeFrame")
		if tf is not None and tf.get("duration"):
			t1 = t0 + float(tf["duration"])
		src_fps = float(clip.get("frameRate", 30) or 30)
		fps = min(src_fps, float(opts.anim_fps)) if opts.anim_fps else src_fps
		count = max(1, int(round((t1 - t0) * fps)) + 1)
		frames = np.zeros((count, len(joints), 6))
		for b, (node, parent, _w, since) in enumerate(joints):
			tr = node.get("transform")
			rest_p = np.asarray(tr.get("position", (0, 0, 0)) if tr is not None else (0, 0, 0), float)
			rest_q = np.asarray(tr.get("orientation", (0, 0, 0, 1)) if tr is not None else (0, 0, 0, 1), float)
			for k in range(count):
				t = t0 + k / fps
				p = _sk.sample_keys(*tracks[(b, "position")], t) if (b, "position") in tracks else rest_p
				q = _sk.sample_keys(*tracks[(b, "orientation")], t, quat=True) if (b, "orientation") in tracks \
					else rest_q
				local = np.eye(4)
				local[:3, :3] = _sk.quat_matrix(np.asarray(q, float))
				local[:3, 3] = p
				frames[k, b] = _sk.pose_from_matrix(space.matrix(since @ local))
		out.append(_m.Sequence(_m._unique(_m._seq_name(clip.name or "anim"), used), fps, frames, loop=True))
	if clips and not out:
		warnings.append("animationList: no channel drives a joint transform; no sequences")
	return out
