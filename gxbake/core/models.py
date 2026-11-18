"""Generic 3D models (OBJ, FBX, glTF, DAE, 3DS...) through Assimp → GoldSrc studio MDL.

The Xash3D GameCube port draws models as GoldSrc studio models, so an imported
mesh is baked into a static, single-bone studio MDL (version 10) with 8-bit
paletted skins. That MDL then goes through the same lean-mesh / skin-companion
pass as any other .mdl (see studio.py), so the outputs are the familiar
models/<name>.mdl, gc_studio/<name>.mdl and gc_studio/<name>t.mdl.

Assimp is reached through the assimp-py wheel (bundled native library, no
system install). It is an optional dependency: everything else in GxBake works
without it, and available() says whether it can be used.
"""

from __future__ import annotations

import io
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import skeleton as _sk

# Extensions sent to Assimp. .mdl stays with the GoldSrc studio path.
MODEL_EXTENSIONS = {
	# interchange
	".obj", ".fbx", ".gltf", ".glb", ".dae", ".3ds", ".ply", ".stl", ".x", ".x3d", ".amf", ".3mf", ".dxf", ".off",
	# DCC tools
	".blend", ".lwo", ".lws", ".lxo", ".ase", ".ac", ".cob", ".csm", ".sib", ".ndo", ".q3o", ".q3s", ".xgl", ".zgl",
	# game formats (.mdl stays with the GoldSrc studio path)
	".smd", ".md2", ".md3", ".mdc", ".md5mesh", ".ms3d", ".b3d", ".iqm", ".irr", ".irrmesh", ".ogex", ".hmp",
	".nff", ".ter",
}
# Animation-only files, used as <model>@<clip>.<ext> clips.
# Source 2 compiled models, read natively (vmdl.py)
SOURCE2_MODEL_EXTENSIONS = {".vmdl_c", ".vmesh_c"}
# Valve Datamodel model sources (Blender Source Tools, SFM), read natively (dmx_model.py)
DMX_MODEL_EXTENSIONS = {".dmx"}
CLIP_EXTENSIONS = MODEL_EXTENSIONS | {".bvh", ".md5anim"}

# GoldSrc / Xash studio limits (studio.h).
MAXSTUDIOVERTS = 2048
MAXSTUDIOTRIANGLES = 20000
MAXSTUDIOSKINS = 100
STUDIO_VERSION = 10
STUDIO_NF_MASKED = 0x40

_HDR = 244
_BONE = 112
_SEQ = 176
_SEQGROUP = 104
_BODYPART = 76
_MODEL = 112
_MESH = 20
_TEXTURE = 80

# aiTextureType_DIFFUSE and aiTextureType_BASE_COLOR (glTF PBR).
_DIFFUSE_TYPES = (1, 12)


def available() -> bool:
	try:
		import assimp_py	# noqa: F401
	except ImportError:
		return False
	return True


def skeletal_supported() -> bool:
	"""assimp-py 1.2+ exposes bones and animations; 1.1 (the last for Python 3.10/3.11) does not."""
	try:
		import assimp_py
	except ImportError:
		return False
	return hasattr(assimp_py, "Bone") and hasattr(assimp_py, "Animation")


def _assimp():
	try:
		import assimp_py
	except ImportError as exc:
		raise RuntimeError("Assimp is not installed: pip install assimp-py (or gxbake[models])") from exc
	return assimp_py


@dataclass
class ModelOptions:
	scale: float = 1.0	# source units → GoldSrc units (1 unit ≈ 1 inch; glTF is metres, try 39.37)
	up_axis: str = "y"	# y: Y-up source (glTF, OBJ, FBX via Assimp); z: already Z-up
	skin_max_edge: int = 256	# full-quality skin in models/<name>.mdl; the companion shrinks further
	alpha_threshold: int = 128
	anim_fps: int = 15	# sample rate for baked sequences
	static: bool = False	# ignore skeletons and animations: one root bone, bind pose
	clip_files: bool = True	# merge animations from <name>@<clip>.<ext> next to the model
	blend_bones: bool = True	# native Source import: helper bones stand in for two-bone weights
	rose_scale: float = 39.37	# ROSE Online models: GoldSrc units per metre (scale above is for Assimp only)
	rose_yaw: float = 90.0	# ROSE Online models face -Y; turn about Z so they face GoldSrc's +X
	rs_scale: float = 0.35	# RuneScape models: GoldSrc units per model unit (128 units = one tile)
	rs_yaw: float = 0.0	# RuneScape models: extra turn about Z (they are already turned to face +X)


@dataclass
class Material:
	name: str
	rgba: "np.ndarray"	# skin texels (power-of-two)
	masked: bool
	texture: str = ""	# where the texels came from


@dataclass
class Submesh:
	material: int
	positions: "np.ndarray"	# (n, 3) float32, in the space of each vertex's bone
	normals: "np.ndarray"	# (n, 3) float32, bone space
	uvs: "np.ndarray"	# (n, 2) float32, top-left origin
	tris: "np.ndarray"	# (m, 3) uint32, counter-clockwise like the source
	bones: "np.ndarray | None" = None	# (n,) studio bone per vertex; None = bone 0
	world: "np.ndarray | None" = None	# (n, 3) rest-pose model space (GoldSrc axes)

	def __post_init__(self):
		if self.bones is None:
			self.bones = np.zeros(len(self.positions), np.int32)
		if self.world is None:
			self.world = self.positions


@dataclass
class Bone:
	name: str
	parent: int
	pose: "np.ndarray"	# rest (x, y, z, roll, pitch, yaw) relative to the parent


@dataclass
class Sequence:
	name: str
	fps: float
	frames: "np.ndarray"	# (numframes, numbones, 6) poses
	loop: bool = True


@dataclass
class ImportedModel:
	name: str
	materials: list[Material]
	meshes: list[Submesh]
	warnings: list[str] = field(default_factory=list)
	bones: list[Bone] = field(default_factory=lambda: [Bone("root", -1, np.zeros(6))])
	sequences: list[Sequence] = field(default_factory=list)

	@property
	def num_vertices(self) -> int:
		return sum(len(m.positions) for m in self.meshes)

	@property
	def num_triangles(self) -> int:
		return sum(len(m.tris) for m in self.meshes)

	@property
	def animated(self) -> bool:
		return any(len(s.frames) > 1 for s in self.sequences)

	def bounds(self) -> tuple["np.ndarray", "np.ndarray"]:
		if not self.meshes:
			z = np.zeros(3, np.float32)
			return z, z
		pts = np.concatenate([m.world for m in self.meshes])
		return pts.min(0), pts.max(0)

	def rest_frames(self) -> "np.ndarray":
		return np.array([b.pose for b in self.bones])[None]


# -- import -------------------------------------------------------------------


def _flags(ai) -> int:
	return (
		ai.Process_Triangulate
		| ai.Process_JoinIdenticalVertices
		| ai.Process_GenSmoothNormals
		| ai.Process_SortByPType
		| ai.Process_FindDegenerates
		| ai.Process_RemoveRedundantMaterials
		| ai.Process_LimitBoneWeights
		| ai.Process_FlipUVs	# top-left texel origin, like studio s/t
		| ai.Process_ValidateDataStructure
	)


def _read_scene(ai, path: Path):
	try:
		scene = ai.import_file(str(path), _flags(ai))
	except FileNotFoundError:
		raise
	except Exception as exc:	# noqa: BLE001 - assimp raises bare RuntimeErrors with its log text
		raise ValueError(f"Assimp could not read {path.name}: {exc}") from exc
	if scene is None:
		raise ValueError(f"Assimp could not read {path.name}")
	return scene


class _Nodes:
	"""Flattened node tree: DFS order, parents, rest local matrices."""

	def __init__(self, root):
		self.nodes, self.parent, self.local = [], [], []
		self.by_name: dict[str, int] = {}
		stack = [(root, -1)]
		while stack:
			node, par = stack.pop()
			i = len(self.nodes)
			self.nodes.append(node)
			self.parent.append(par)
			self.local.append(np.array(node.transformation, np.float64))
			self.by_name.setdefault(node.name, i)
			for child in reversed(list(node.children or [])):
				stack.append((child, i))
		self.globals = []
		for i, p in enumerate(self.parent):
			self.globals.append(self.local[i] if p < 0 else self.globals[p] @ self.local[i])


def _axis_matrix(opts: ModelOptions) -> "np.ndarray":
	"""Source scene space → GoldSrc model space (axis swap and scale)."""
	m = np.zeros((4, 4))
	if opts.up_axis == "z":
		m[:3, :3] = np.eye(3)
	else:	# Y-up (+Z forward, +X left of the model) → GoldSrc (+X forward, +Y left, +Z up)
		m[0, 2] = m[1, 0] = m[2, 1] = 1.0
	m[:3, :3] *= opts.scale
	m[3, 3] = 1.0
	return m


def clip_base(path: Path) -> "Path | None":
	"""The model a <stem>@<clip>.<ext> clip belongs to, if it sits next to it."""
	path = Path(path)
	if "@" not in path.stem:
		return None
	base = path.stem.split("@", 1)[0].lower()
	try:
		for p in path.parent.iterdir():
			if p.is_file() and p.stem.lower() == base and p.suffix.lower() in MODEL_EXTENSIONS:
				return p
	except OSError:
		pass
	return None


def clip_files(path: Path) -> list[Path]:
	"""Unity-style animation clips next to the model: <stem>@<clip>.<ext>."""
	try:
		return sorted(p for p in path.parent.iterdir()
			if p.is_file() and p.stem.lower().startswith(path.stem.lower() + "@")
			and p.suffix.lower() in CLIP_EXTENSIONS)
	except OSError:
		return []


def native_format(path: Path) -> str | None:
	"""'source', 'source2', 'dmx' or 'quake' when GxBake reads the file itself instead of Assimp."""
	if Path(path).suffix.lower() in SOURCE2_MODEL_EXTENSIONS:
		return "source2"
	if Path(path).suffix.lower() == ".dmx":
		from .dmx_model import is_dmx_model

		return "dmx" if is_dmx_model(path) else None
	try:
		with Path(path).open("rb") as f:
			head = f.read(8)
	except OSError:
		return None
	from .quake import is_alias_mdl
	from .source_mdl import is_source_mdl

	if is_source_mdl(head):
		return "source"
	if is_alias_mdl(head):
		return "quake"
	return None


def import_model(path: Path, opts: ModelOptions | None = None) -> ImportedModel:
	"""Read a model into triangulated submeshes, bones and sequences.

	Source (.mdl v44-49), Source 2 (.vmdl_c, .vmesh_c), DMX model sources and
	Quake (.mdl IDPO) models are parsed natively;
	everything else goes through Assimp.
	"""
	opts = opts or ModelOptions()
	path = Path(path)
	native = native_format(path)
	if native == "source":
		from .source_mdl import import_source

		return import_source(path, opts)
	if native == "source2":
		from .vmdl import import_vmdl

		return import_vmdl(path, opts)
	if native == "dmx":
		from .dmx_model import import_dmx

		return import_dmx(path, opts)
	if native == "quake":
		from .quake import import_alias

		return import_alias(path, opts)
	ai = _assimp()
	scene = _read_scene(ai, path)
	if not scene.meshes:
		raise ValueError(f"{path.name}: no meshes")
	tree = _Nodes(scene.root_node)
	axis = _axis_matrix(opts)
	warnings: list[str] = []

	anims = [(a.name or f"anim{i}", a) for i, a in enumerate(getattr(scene, "animations", None) or [])]
	if not opts.static and opts.clip_files:
		for clip in clip_files(path):
			try:
				cs = _read_scene(ai, clip)
			except (ValueError, OSError) as exc:
				warnings.append(f"clip {clip.name}: {exc}")
				continue
			clip_anims = getattr(cs, "animations", None) or []
			label = clip.stem.split("@", 1)[1]
			for i, a in enumerate(clip_anims):
				anims.append((label if len(clip_anims) == 1 else f"{label}_{a.name or i}", a))
	skinned = any(getattr(m, "num_bones", 0) for m in scene.meshes)
	if not opts.static and not skeletal_supported():
		warnings.append("this assimp-py has no bone/animation API (needs 1.2+, Python 3.12+); baked static")
	skeletal = not opts.static and (skinned or bool(anims))

	# Which nodes become studio bones: skin joints, mesh owners, animated nodes, and their ancestors.
	mesh_nodes: list[tuple[int, int]] = []	# (node index, mesh index)
	for i, node in enumerate(tree.nodes):
		for mi in node.mesh_indices or ():
			mesh_nodes.append((i, mi))
	if skeletal:
		# a skinned mesh's own node only matters if some vertex has no weights
		want: set[int] = {i for i, mi in mesh_nodes if not getattr(scene.meshes[mi], "num_bones", 0)}
		for m in scene.meshes:
			for b in getattr(m, "bones", None) or []:
				if b.name in tree.by_name:
					want.add(tree.by_name[b.name])
		for _n, a in anims:
			for ch in a.channels:
				if ch.node_name in tree.by_name:
					want.add(tree.by_name[ch.node_name])
		for i in list(want):
			p = tree.parent[i]
			while p >= 0 and p not in want:
				want.add(p)
				p = tree.parent[p]
		order = [i for i in range(len(tree.nodes)) if i in want]	# DFS order: parents first
		if len(order) > _sk.MAXSTUDIOBONES:
			warnings.append(f"{len(order)} bones exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static instead")
			skeletal = False
	if not skeletal:
		order = []

	model = ImportedModel(path.stem, [], [], warnings)
	studio_of: dict[int, int] = {}	# node index → studio bone
	rigid_world: list[np.ndarray] = [np.eye(4)]
	if skeletal:
		model.bones, rigid_world = [], []
		used_names: set[str] = set()
		for n in order:
			studio_of[n] = len(model.bones)
			p = tree.parent[n]
			parent = studio_of.get(p, -1)
			w = _sk.rigid(axis @ tree.globals[n])
			local = w if parent < 0 else _sk.rigid_inverse(rigid_world[parent]) @ w
			name = _unique((tree.nodes[n].name or f"bone{len(model.bones)}")[:31], used_names)
			model.bones.append(Bone(name, parent, _sk.pose_from_matrix(local)))
			rigid_world.append(w)

	embedded = _embedded_images(path)
	mat_map: dict[int, int] = {}
	seen_skinned: set[int] = set()
	for node_i, mi in mesh_nodes:
		mesh = scene.meshes[mi]
		if mesh.num_faces == 0 or mesh.num_vertices == 0 or mesh.indices is None:
			continue
		idx = np.frombuffer(mesh.indices, np.uint32)
		if len(idx) % 3:
			continue	# points/lines left over by SortByPType
		mesh_bones = list(getattr(mesh, "bones", None) or []) if skeletal else []
		if mesh_bones:
			if mi in seen_skinned:
				continue	# skinned meshes ignore their node; place them once
			seen_skinned.add(mi)
		pos = np.frombuffer(mesh.vertices, np.float32).reshape(-1, 3).astype(np.float64)
		n = len(pos)
		nrm = (np.frombuffer(mesh.normals, np.float32).reshape(-1, 3).astype(np.float64)
			if mesh.normals is not None else np.tile([0.0, 1.0, 0.0], (n, 1)))
		uv = np.zeros((n, 2), np.float32)
		if mesh.texcoords and mesh.texcoords[0] is not None:
			t = np.frombuffer(mesh.texcoords[0], np.float32)
			comps = len(t) // n if n else 2
			uv = t.reshape(n, comps)[:, :2].copy()

		# Each vertex follows one bone: its heaviest joint, else the mesh's own node.
		owner = node_i
		while skeletal and owner >= 0 and owner not in studio_of:
			owner = tree.parent[owner]
		vbone = np.full(n, studio_of.get(owner, 0), np.int32)
		vmat = np.repeat((axis @ tree.globals[node_i])[None], n, 0)	# unweighted: rigid with the mesh node
		if mesh_bones:
			best = np.zeros(n)
			for b in mesh_bones:
				if b.name not in tree.by_name:
					warnings.append(f"skin joint {b.name} has no node; its vertices stay on the mesh node")
					continue
				ni = tree.by_name[b.name]
				ids = np.frombuffer(b.weight_vertex_ids, np.uint32) if b.num_weights else np.zeros(0, np.uint32)
				ws = np.frombuffer(b.weights, np.float32) if b.num_weights else np.zeros(0, np.float32)
				m = axis @ tree.globals[ni] @ np.array(b.offset_matrix, np.float64)
				take = ws > best[ids]
				sel = ids[take]
				best[sel] = ws[take]
				vbone[sel] = studio_of[ni]
				vmat[sel] = m
			if (best < 0.999).any() and (best > 0).any():
				warnings.append("blended skin weights are snapped to each vertex's heaviest bone (studio skinning is rigid)")
		world = np.einsum("nij,nj->ni", vmat[:, :3, :3], pos) + vmat[:, :3, 3]
		nmat = np.linalg.inv(vmat[:, :3, :3]).transpose(0, 2, 1)
		wn = np.einsum("nij,nj->ni", nmat, nrm)
		if skeletal:
			bw = np.array(rigid_world)[vbone]
			rel = world - bw[:, :3, 3]
			local = np.einsum("nji,nj->ni", bw[:, :3, :3], rel)
			ln = np.einsum("nji,nj->ni", bw[:, :3, :3], wn)
		else:
			local, ln = world, wn
		lens = np.linalg.norm(ln, axis=1, keepdims=True)
		ln = np.where(lens > 1e-8, ln / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		mat_i = mesh.material_index
		if mat_i not in mat_map:
			mat_map[mat_i] = len(model.materials)
			mdesc = scene.materials[mat_i] if 0 <= mat_i < len(scene.materials) else {}
			model.materials.append(_load_material(mdesc, path, embedded, opts, warnings))
		model.meshes.append(Submesh(mat_map[mat_i], local.astype(np.float32), ln.astype(np.float32), uv,
			idx.reshape(-1, 3).copy(), vbone, world.astype(np.float32)))
	if not model.meshes:
		raise ValueError(f"{path.name}: no triangle meshes")

	if skeletal:
		used: set[str] = set()
		for name, anim in anims:
			frames = _sample(anim, tree, order, [b.parent for b in model.bones], axis, opts.anim_fps)
			if frames is None:
				continue
			label = _unique(_seq_name(name), used)
			model.sequences.append(Sequence(label, float(opts.anim_fps), frames))
	if not model.sequences:
		model.sequences.append(Sequence("idle", 30.0, model.rest_frames(), loop=True))
	if opts.static:
		nanim = len(getattr(scene, "animations", None) or [])
		if nanim or skinned:
			warnings.append("baked static: skeleton and animations ignored")
	return model


def _unique(name: str, used: set[str]) -> str:
	base, n = name, 1
	while name.lower() in used:
		n += 1
		suffix = f"_{n}"
		name = base[: 31 - len(suffix)] + suffix
	used.add(name.lower())
	return name


def _seq_name(name: str) -> str:
	name = name.split("|")[-1]	# FBX "Armature|Walk"
	safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name.strip().lower())
	return (safe or "anim")[:31]


def _sample(anim, tree: _Nodes, order: list[int], parents: list[int], axis: "np.ndarray",
		fps: int) -> "np.ndarray | None":
	"""Bake one animation into (numframes, numbones, 6) studio poses."""
	tps = float(anim.ticks_per_second or 0) or 25.0
	duration = float(anim.duration or 0)
	chans = {}
	for ch in anim.channels or []:
		if ch.node_name in tree.by_name:
			chans[tree.by_name[ch.node_name]] = _sk.Channel(ch)
	if not chans:
		return None
	nframes = max(1, int(round(duration / tps * fps)) + 1)
	frames = np.zeros((nframes, len(order), 6))
	for k in range(nframes):
		tick = min(duration, k / fps * tps)
		world: list[np.ndarray] = []
		g: dict[int, np.ndarray] = {}
		for b, n in enumerate(order):
			local = chans[n].local(tick, tree.local[n]) if n in chans else tree.local[n]
			p = tree.parent[n]
			g[n] = (g[p] if p in g else (tree.globals[p] if p >= 0 else np.eye(4))) @ local
			w = _sk.rigid(axis @ g[n])
			world.append(w)
			rel = w if parents[b] < 0 else _sk.rigid_inverse(world[parents[b]]) @ w
			frames[k, b] = _sk.pose_from_matrix(rel)
	return frames


def _load_material(desc: dict, model_path: Path, embedded: list[bytes], opts: ModelOptions,
		warnings: list[str]) -> Material:
	name = str(desc.get("NAME") or "material")
	tex_ref = ""
	textures = desc.get("TEXTURES") or {}
	for t in _DIFFUSE_TYPES:
		if textures.get(t):
			tex_ref = textures[t][0]
			break
	if not tex_ref and desc.get("TEXTURE_BASE"):
		tex_ref = desc["TEXTURE_BASE"]
	rgba = None
	if tex_ref:
		rgba = _load_texture(tex_ref, model_path, embedded)
		if rgba is None:
			warnings.append(f"material {name}: texture {tex_ref} not found; using its colour")
	if rgba is None:
		col = desc.get("COLOR_BASE") or desc.get("COLOR_DIFFUSE") or [0.8, 0.8, 0.8]
		c = [int(round(max(0.0, min(1.0, float(x))) * 255)) for x in list(col)[:3]]
		alpha = 255 if float(desc.get("OPACITY", 1.0)) >= 0.5 else 0
		rgba = np.empty((8, 8, 4), np.uint8)
		rgba[...] = (*c, alpha)
	rgba = _fit_skin(rgba, opts.skin_max_edge)
	masked = bool((rgba[..., 3] < opts.alpha_threshold).any())
	return Material(name, rgba, masked, tex_ref)


def _open_rgba(data_or_path) -> "np.ndarray | None":
	from PIL import Image

	try:
		src = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else data_or_path
		with Image.open(src) as img:
			img.load()
			return np.asarray(img.convert("RGBA")).copy()
	except (OSError, ValueError, SyntaxError):
		return None


def _load_texture(ref: str, model_path: Path, embedded: list[bytes]) -> "np.ndarray | None":
	from urllib.parse import unquote

	from .images import IMAGE_EXTENSIONS

	if ref.startswith("*"):
		try:
			data = embedded[int(ref[1:])]
		except (ValueError, IndexError):
			return None
		return _open_rgba(data) if data else None
	if ref.startswith("data:"):
		return _open_rgba(_data_uri(ref))
	ref = unquote(ref.replace("\\", "/"))
	base = model_path.parent
	name = Path(ref).name
	cands = [base / ref, base / name, base / name.lower()]
	# Exporters often leave absolute paths from the artist's machine; try the
	# basename next to the model and in a textures/ folder.
	for d in ("textures", "Textures", "tex", "maps"):
		cands += [base / d / name, base.parent / d / name]
	for c in cands:
		try:
			if c.is_file():
				rgba = _open_rgba(c)
				if rgba is not None:
					return rgba
		except OSError:
			continue
	# Same name, other format (a .psd or .tif reference with a .png beside it).
	stem = Path(name).stem.lower()
	for d in (base, base / "textures", base.parent / "textures"):
		try:
			for c in sorted(d.iterdir()):
				if c.stem.lower() == stem and c.suffix.lower() in IMAGE_EXTENSIONS:
					rgba = _open_rgba(c)
					if rgba is not None:
						return rgba
		except OSError:
			continue
	return None


def _data_uri(uri: str) -> bytes:
	import base64
	from urllib.parse import unquote_to_bytes

	head, _, body = uri.partition(",")
	return base64.b64decode(body) if head.endswith(";base64") else unquote_to_bytes(body)


def _embedded_images(path: Path) -> list[bytes]:
	"""glTF images in document order (Assimp's "*N" refs): GLB chunks, data URIs, buffers."""
	ext = path.suffix.lower()
	if ext not in (".glb", ".gltf"):
		return []
	try:
		raw = path.read_bytes()
		doc, binchunk = None, b""
		if raw[:4] == b"glTF":
			off = 12
			while off + 8 <= len(raw):
				clen, ctype = struct.unpack_from("<I4s", raw, off)
				chunk = raw[off + 8: off + 8 + clen]
				if ctype == b"JSON":
					doc = json.loads(chunk)
				elif ctype == b"BIN\0":
					binchunk = chunk
				off += 8 + clen
		else:
			doc = json.loads(raw)
		if not doc:
			return []
		buffers: dict[int, bytes] = {}

		def buffer(i: int) -> bytes:
			if i not in buffers:
				b = doc["buffers"][i]
				uri = b.get("uri")
				if uri is None:
					buffers[i] = binchunk
				elif uri.startswith("data:"):
					buffers[i] = _data_uri(uri)
				else:
					from urllib.parse import unquote

					buffers[i] = (path.parent / unquote(uri)).read_bytes()
			return buffers[i]

		out = []
		for img in doc.get("images", []):
			try:
				if "bufferView" in img:
					bv = doc["bufferViews"][img["bufferView"]]
					s = bv.get("byteOffset", 0)
					out.append(buffer(bv["buffer"])[s: s + bv["byteLength"]])
				elif str(img.get("uri", "")).startswith("data:"):
					out.append(_data_uri(img["uri"]))
				else:
					from urllib.parse import unquote

					out.append((path.parent / unquote(img.get("uri", ""))).read_bytes())
			except (OSError, KeyError, IndexError, ValueError):
				out.append(b"")
		return out
	except (OSError, ValueError, KeyError, IndexError, struct.error):
		return []


def _fit_skin(rgba: "np.ndarray", max_edge: int) -> "np.ndarray":
	"""Power-of-two sides (GX_REPEAT and the companion's halving both want them)."""
	from PIL import Image

	h, w = rgba.shape[:2]
	tw = 1 << max(3, (w - 1).bit_length())
	th = 1 << max(3, (h - 1).bit_length())
	while max(tw, th) > max(8, max_edge):
		tw, th = max(8, tw // 2), max(8, th // 2)
	if (tw, th) == (w, h):
		return rgba
	return np.asarray(Image.fromarray(rgba).resize((tw, th), Image.Resampling.LANCZOS))


# -- studio MDL writer --------------------------------------------------------


def quantize_skin(rgba: "np.ndarray", masked: bool, alpha_threshold: int = 128) -> tuple[bytes, bytes]:
	"""RGBA → (indices, 768-byte palette). Masked skins reserve index 255 for holes."""
	from PIL import Image

	h, w = rgba.shape[:2]
	colours = 255 if masked else 256
	img = Image.fromarray(np.ascontiguousarray(rgba[..., :3]))
	q = img.quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
	idx = np.asarray(q, np.uint8).copy()
	pal = (q.getpalette() or [])[: colours * 3]
	pal = bytes(pal) + bytes(768 - len(pal))
	if masked:
		idx[rgba[..., 3] < alpha_threshold] = 255
		pal = pal[:765] + b"\x00\x00\xff"	# conventional blue key
	return idx.tobytes(), pal[:768]


def _strips(tris: list[tuple[int, int, int]]) -> list[list[int]]:
	"""Greedy GL triangle strips over corner ids, keeping each triangle's winding."""
	edge: dict[tuple[int, int], list[int]] = {}
	for t, (a, b, c) in enumerate(tris):
		for e in ((a, b), (b, c), (c, a)):
			edge.setdefault(e, []).append(t)
	used = [False] * len(tris)

	def take(x: int, y: int) -> tuple[int, int] | None:
		"""Unused triangle with directed edge x→y; returns (tri, third corner)."""
		for t in edge.get((x, y), ()):
			if not used[t]:
				a, b, c = tris[t]
				third = c if (a, b) == (x, y) else a if (b, c) == (x, y) else b
				return t, third
		return None

	def grow(seq: list[int], mark: bool) -> list[int]:
		strip = list(seq)
		claimed: list[int] = []
		while True:
			k = len(strip) - 2	# index of the triangle we'd add
			x, y = strip[-2], strip[-1]
			hit = take(x, y) if k % 2 == 0 else take(y, x)
			if hit is None or hit[0] in claimed:
				break
			claimed.append(hit[0])
			used[hit[0]] = True
			strip.append(hit[1])
		if not mark:
			for t in claimed:
				used[t] = False
		return strip

	out = []
	for t0, (a, b, c) in enumerate(tris):
		if used[t0]:
			continue
		used[t0] = True
		best = max(((a, b, c), (b, c, a), (c, a, b)), key=lambda s: len(grow(list(s), False)))
		out.append(grow(list(best), True))
	return out


def build_studio_mdl(model: ImportedModel, opts: ModelOptions | None = None) -> bytes:
	"""Bake an imported model into a GoldSrc studio MDL (v10): bones, sequences, meshes, skins."""
	opts = opts or ModelOptions()
	if len(model.materials) > MAXSTUDIOSKINS:
		raise ValueError(f"{len(model.materials)} materials; studio models allow {MAXSTUDIOSKINS}")

	# Positions are shared model-wide. Normals are laid out per mesh, because the
	# engine lights mesh j's normals as one contiguous run (mstudiomesh_t.numnorms).
	pos_index: dict[tuple, int] = {}
	positions: list[tuple[float, float, float]] = []
	normals: list[tuple[float, float, float]] = []
	vert_bone: list[int] = []
	norm_bone: list[int] = []
	meshes: list[tuple[int, int, int, list[list[tuple[int, int, int, int]]]]] = []	# (skin, numnorms, normstart, strips)
	ntris = 0
	for skin in range(len(model.materials)):
		subs = [m for m in model.meshes if m.material == skin]
		if not subs:
			continue
		mat = model.materials[skin]
		th, tw = mat.rgba.shape[:2]
		norm_index: dict[tuple, int] = {}
		norm_start = len(normals)
		corners: dict[tuple, int] = {}
		corner_list: list[tuple[int, int, int, int]] = []
		tris: list[tuple[int, int, int]] = []
		for sm in subs:
			cid = np.empty(len(sm.positions), np.int64)
			for v in range(len(sm.positions)):
				bone = int(sm.bones[v])
				p = tuple(np.round(sm.positions[v].astype(np.float64), 4))
				vi = pos_index.get((bone, p))
				if vi is None:
					vi = pos_index[(bone, p)] = len(positions)
					positions.append(p)
					vert_bone.append(bone)
				n = tuple(np.round(sm.normals[v].astype(np.float64), 3))
				ni = norm_index.get((bone, n))
				if ni is None:
					ni = norm_index[(bone, n)] = len(normals)
					normals.append(n)
					norm_bone.append(bone)
				s = int(np.clip(round(float(sm.uvs[v, 0]) * tw), -32768, 32767))
				t = int(np.clip(round(float(sm.uvs[v, 1]) * th), -32768, 32767))
				key = (vi, ni, s, t)
				c = corners.get(key)
				if c is None:
					c = corners[key] = len(corner_list)
					corner_list.append(key)
				cid[v] = c
			for a, b, c in sm.tris:
				ta, tb, tc = int(cid[a]), int(cid[b]), int(cid[c])
				if len({ta, tb, tc}) < 3:
					continue
				# studiomdl stores triangles reversed ("quake wants them in the reverse order").
				tris.append((ta, tc, tb))
		ntris += len(tris)
		strips = [[corner_list[c] for c in s] for s in _strips(tris)]
		meshes.append((skin, len(normals) - norm_start, norm_start, strips))

	if len(positions) > MAXSTUDIOVERTS:
		model.warnings.append(f"{len(positions)} vertices exceeds the studio limit of {MAXSTUDIOVERTS}; decimate the source")
	if len(normals) > MAXSTUDIOVERTS:
		model.warnings.append(f"{len(normals)} normals exceeds the studio limit of {MAXSTUDIOVERTS}")
	if ntris > MAXSTUDIOTRIANGLES:
		model.warnings.append(f"{ntris} triangles exceeds the studio limit of {MAXSTUDIOTRIANGLES}")

	nb = len(model.bones)
	parents = [b.parent for b in model.bones]
	rest = np.array([b.pose for b in model.bones], np.float64)
	seqs = model.sequences or [Sequence("idle", 30.0, rest[None])]
	pts = np.array(positions, np.float64).reshape(-1, 3)
	vb = np.array(vert_bone, np.int64)

	def posed(frame: "np.ndarray") -> "np.ndarray":
		mats = np.array(_sk.world_matrices(frame, parents))
		return np.einsum("nij,nj->ni", mats[vb, :3, :3], pts) + mats[vb, :3, 3] if len(pts) else pts

	def box(frames: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
		step = max(1, len(frames) // 16)	# a sample of frames is plenty for culling boxes
		w = np.concatenate([posed(f) for f in frames[::step]] + [posed(frames[-1])]) if len(pts) else np.zeros((1, 3))
		return w.min(0), w.max(0)

	bmin, bmax = box(rest[None])
	rest_world = posed(rest)
	radius = float(np.linalg.norm(rest_world, axis=1).max()) if len(pts) else 0.0
	scales = _sk.bone_scales(rest, [sq.frames for sq in seqs])

	out = bytearray(_HDR)
	name = (model.name.lower() + ".mdl").encode("latin-1", "replace")[:63]

	def align():
		out.extend(bytes((-len(out)) % 4))

	bone_off = len(out)
	for b, bone in enumerate(model.bones):
		out += struct.pack("<32sii6i6f6f", bone.name.encode("latin-1", "replace")[:31], bone.parent, 0,
			*([-1] * 6), *rest[b], *scales[b])

	# one hitbox per bone that owns vertices, so traces and damage work on animated models
	hit_off = len(out)
	nhit = 0
	for b in range(nb):
		sel = pts[vb == b]
		if len(sel):
			out += struct.pack("<ii3f3f", b, 0, *sel.min(0), *sel.max(0))
			nhit += 1

	anim_offs = []
	for sq in seqs:
		align()
		anim_offs.append(len(out))
		while True:
			try:
				out += _sk.encode_anim_block(sq.frames, rest, scales)
				break
			except ValueError:
				if len(sq.frames) <= 2:
					raise
				# too long for the 16-bit stream offsets: halve the frame rate of this one sequence
				sq.frames = sq.frames[::2]
				sq.fps = sq.fps / 2.0
				model.warnings.append(f"sequence {sq.name}: halved to {sq.fps:g} fps to fit studio offsets")
	align()
	seq_off = len(out)
	for sq, anim_off in zip(seqs, anim_offs):
		smin, smax = box(sq.frames)
		bmin, bmax = np.minimum(bmin, smin), np.maximum(bmax, smax)
		seq = bytearray(_SEQ)
		struct.pack_into("<32sfiiiiiiii", seq, 0, sq.name.encode("latin-1", "replace")[:31], float(sq.fps),
			_sk.STUDIO_LOOPING if sq.loop else 0, 0, 0, 0, 0, len(sq.frames), 0, 0)
		struct.pack_into("<ii", seq, 68, 0, 0)	# motiontype, motionbone
		struct.pack_into("<3f3f", seq, 96, *smin, *smax)	# bbmin, bbmax
		struct.pack_into("<ii", seq, 120, 1, anim_off)	# numblends, animindex
		struct.pack_into("<2f", seq, 144, 0.0, 1.0)	# blendend
		out += seq
	group_off = len(out)
	out += struct.pack("<32s64sii", b"default", b"", 0, 0)

	# body: one bodypart, one model
	bodypart_off = len(out)
	out += bytes(_BODYPART)
	model_off = len(out)
	out += bytes(_MODEL)
	vertinfo_off = len(out)
	out += bytes(vert_bone)
	align()
	norminfo_off = len(out)
	out += bytes(norm_bone)
	align()
	verts_off = len(out)
	for p in positions:
		out += struct.pack("<3f", *p)
	norms_off = len(out)
	for n in normals:
		out += struct.pack("<3f", *n)
	mesh_off = len(out)
	out += bytes(_MESH * len(meshes))
	for i, (skin, numnorms, normstart, strips) in enumerate(meshes):
		tri_off = len(out)
		count = 0
		for s in strips:
			out += struct.pack("<h", len(s))
			for vi, ni, ss, tt in s:
				out += struct.pack("<4h", vi, ni, ss, tt)
			count += len(s) - 2
		out += struct.pack("<h", 0)
		align()
		struct.pack_into("<5i", out, mesh_off + i * _MESH, count, tri_off, skin, numnorms, normstart)
	struct.pack_into("<64sii", out, bodypart_off, b"body", 1, 0)
	struct.pack_into("<i", out, bodypart_off + 72, model_off)
	struct.pack_into("<64sif10i", out, model_off, model.name.encode("latin-1", "replace")[:63], 0, radius,
		len(meshes), mesh_off, len(positions), vertinfo_off, verts_off, len(normals), norminfo_off, norms_off, 0, 0)

	# textures last, so lean_studio_mesh_bytes can cut them off
	tex_off = len(out)
	out += bytes(_TEXTURE * len(model.materials))
	skin_off = len(out)
	for i in range(len(model.materials)):
		out += struct.pack("<h", i)
	align()
	texdata_off = len(out)
	for i, mat in enumerate(model.materials):
		h, w = mat.rgba.shape[:2]
		pixels, pal = quantize_skin(mat.rgba, mat.masked, opts.alpha_threshold)
		tname = _texture_name(mat.name, i).encode("latin-1", "replace")[:63]
		struct.pack_into("<64siiii", out, tex_off + i * _TEXTURE, tname,
			STUDIO_NF_MASKED if mat.masked else 0, w, h, len(out))
		out += pixels + pal
	align()

	struct.pack_into("<4si64si", out, 0, b"IDST", STUDIO_VERSION, name, len(out))
	struct.pack_into("<3f3f3f3f3f", out, 76, 0.0, 0.0, 0.0, *bmin, *bmax, *bmin, *bmax)
	struct.pack_into("<i", out, 136, 0)	# flags
	struct.pack_into("<10i", out, 140, nb, bone_off, 0, 0, nhit, hit_off, len(seqs), seq_off, 1, group_off)
	struct.pack_into("<6i", out, 180, len(model.materials), tex_off, texdata_off,
		len(model.materials), 1, skin_off)
	struct.pack_into("<2i", out, 204, 1, bodypart_off)
	return bytes(out)


def _texture_name(name: str, i: int) -> str:
	safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name.lower()) or f"skin{i}"
	return f"{safe[:58]}.bmp"


# -- reading studio geometry back (tests, preview) ----------------------------


def studio_meshes(raw: bytes) -> list[dict]:
	"""Decode bodypart 0 / model 0 of a studio MDL into per-mesh triangle lists.

	Each mesh: {"skin": int, "tris": [((vi, ni, s, t), ...x3)], "verts": ndarray, "norms": ndarray}.
	Triangles are returned in the engine's stored (clockwise) order.
	"""
	if len(raw) < _HDR or raw[:4] not in (b"IDST", b"IDSQ"):
		raise ValueError("not a studio model")
	nbody, body_off = struct.unpack_from("<ii", raw, 204)
	if nbody <= 0:
		return []
	_nm, _base, model_off = struct.unpack_from("<iii", raw, body_off + 64)
	(nmesh, mesh_off, nverts, _vi, verts_off, nnorms, _ni, norms_off) = struct.unpack_from("<8i", raw, model_off + 72)
	(vertinfo_off,) = struct.unpack_from("<i", raw, model_off + 84)
	vertbone = np.frombuffer(raw, np.uint8, nverts, vertinfo_off).astype(np.int64)
	verts = np.frombuffer(raw, np.float32, nverts * 3, verts_off).reshape(-1, 3)
	norms = np.frombuffer(raw, np.float32, nnorms * 3, norms_off).reshape(-1, 3)
	out = []
	for i in range(nmesh):
		_ntri, tri_off, skin, _nn, _ns = struct.unpack_from("<5i", raw, mesh_off + i * _MESH)
		tris = []
		off = tri_off
		while True:
			(n,) = struct.unpack_from("<h", raw, off)
			off += 2
			if n == 0:
				break
			fan = n < 0
			n = abs(n)
			cmd = [struct.unpack_from("<4h", raw, off + j * 8) for j in range(n)]
			off += n * 8
			for k in range(n - 2):
				if fan:
					tris.append((cmd[0], cmd[k + 1], cmd[k + 2]))
				elif k % 2 == 0:
					tris.append((cmd[k], cmd[k + 1], cmd[k + 2]))
				else:
					tris.append((cmd[k + 1], cmd[k], cmd[k + 2]))
		out.append({"skin": skin, "tris": tris, "verts": verts, "norms": norms, "vertbone": vertbone})
	return out


def studio_bones(raw: bytes) -> list[dict]:
	nbones, bone_off = struct.unpack_from("<ii", raw, 140)
	out = []
	for b in range(nbones):
		f = struct.unpack_from("<32sii6i6f6f", raw, bone_off + b * _BONE)
		out.append({"name": f[0].split(b"\0")[0].decode("latin-1"), "parent": f[1],
			"value": np.array(f[9:15]), "scale": np.array(f[15:21])})
	return out


def studio_sequences(raw: bytes) -> list[dict]:
	nseq, seq_off = struct.unpack_from("<ii", raw, 164)
	out = []
	for i in range(nseq):
		base = seq_off + i * _SEQ
		label, fps, flags = struct.unpack_from("<32sfi", raw, base)
		(numframes,) = struct.unpack_from("<i", raw, base + 56)
		bb = struct.unpack_from("<3f3f", raw, base + 96)
		numblends, animindex = struct.unpack_from("<ii", raw, base + 120)
		out.append({"name": label.split(b"\0")[0].decode("latin-1"), "fps": fps, "flags": flags,
			"frames": numframes, "bbmin": np.array(bb[:3]), "bbmax": np.array(bb[3:]), "animindex": animindex})
	return out


def studio_pose(raw: bytes, seq: int = 0, frame: int = 0) -> list["np.ndarray"]:
	"""World matrix of every bone at one frame, decoded as the engine does."""
	bones = studio_bones(raw)
	seqs = studio_sequences(raw)
	values = np.array([b["value"] for b in bones])
	scales = np.array([b["scale"] for b in bones])
	poses = values
	if seqs:
		sq = seqs[min(seq, len(seqs) - 1)]
		f = max(0, min(frame, sq["frames"] - 1))
		poses = _sk.decode_anim_frame(raw, sq["animindex"], len(bones), f, values, scales)
	return _sk.world_matrices(poses, [b["parent"] for b in bones])


def studio_posed_vertices(raw: bytes, seq: int = 0, frame: int = 0) -> "np.ndarray":
	meshes = studio_meshes(raw)
	if not meshes:
		return np.zeros((0, 3))
	mats = np.array(studio_pose(raw, seq, frame))
	v = meshes[0]["verts"].astype(np.float64)
	vb = meshes[0]["vertbone"]
	return np.einsum("nij,nj->ni", mats[vb, :3, :3], v) + mats[vb, :3, 3]


# -- preview ------------------------------------------------------------------


def render_preview(tris_xyz: "np.ndarray", colours: "np.ndarray", size: int = 384) -> "np.ndarray":
	"""Flat-shaded three-quarter view (painter's algorithm), for the GUI panes.

	tris_xyz: (n, 3, 3) GoldSrc-space triangles; colours: (n, 3) uint8 base colour.
	"""
	from PIL import Image, ImageDraw

	img = Image.new("RGBA", (size, size), (40, 40, 46, 255))
	if len(tris_xyz) == 0:
		return np.asarray(img).copy()
	yaw, pitch = math.radians(-135.0), math.radians(25.0)
	# camera looks at the model's front-left from above
	cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
	fwd = np.array([cp * cy, cp * sy, -sp])
	right = np.array([sy, -cy, 0.0])
	up = np.cross(right, fwd)
	pts = tris_xyz.reshape(-1, 3).astype(np.float64)
	centre = (pts.min(0) + pts.max(0)) / 2
	p = pts - centre
	sx, syy, depth = p @ right, p @ up, p @ fwd
	extent = max(np.abs(sx).max(), np.abs(syy).max(), 1e-6)
	k = size * 0.45 / extent
	scr = np.stack((size / 2 + sx * k, size / 2 - syy * k), -1).reshape(-1, 3, 2)
	depth = depth.reshape(-1, 3).mean(1)
	# GoldSrc stores clockwise triangles, so the outward normal is (c-a)×(b-a).
	t = tris_xyz.astype(np.float64)
	n = np.cross(t[:, 2] - t[:, 0], t[:, 1] - t[:, 0])
	n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
	light = -fwd + np.array([0.0, 0.0, 0.6])
	light /= np.linalg.norm(light)
	facing = n @ -fwd
	shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
	draw = ImageDraw.Draw(img)
	for i in np.argsort(-depth):
		if facing[i] <= 0:
			continue
		c = tuple(int(x) for x in np.clip(colours[i] * shade[i], 0, 255)) + (255,)
		draw.polygon([tuple(v) for v in scr[i]], fill=c)
	return np.asarray(img).copy()


def _mean_colour(rgba: "np.ndarray") -> "np.ndarray":
	m = rgba[..., 3] >= 128
	px = rgba[..., :3][m] if m.any() else rgba[..., :3].reshape(-1, 3)
	return px.reshape(-1, 3).mean(0)


def _sample_colour(rgba: "np.ndarray", u: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
	"""Skin colour at UV fractions (top-left origin, wrapping), for per-triangle preview colours."""
	h, w = rgba.shape[:2]
	x = (np.floor(np.mod(u, 1.0) * w).astype(int)) % w
	y = (np.floor(np.mod(v, 1.0) * h).astype(int)) % h
	return rgba[y, x, :3].astype(np.float64)


def preview_imported(model: ImportedModel, size: int = 384) -> "np.ndarray":
	tris, cols = [], []
	for m in model.meshes:
		rgba = model.materials[m.material].rgba
		t = m.world[m.tris][:, [0, 2, 1]]	# to clockwise, like the studio output
		tris.append(t)
		uv = m.uvs[m.tris].mean(1) if len(m.uvs) else np.zeros((len(t), 2))
		col = _sample_colour(rgba, uv[:, 0], uv[:, 1])
		opaque = rgba[..., 3].min() >= 128
		cols.append(col if opaque else np.tile(_mean_colour(rgba), (len(t), 1)))
	if not tris:
		return render_preview(np.zeros((0, 3, 3)), np.zeros((0, 3)), size)
	return render_preview(np.concatenate(tris), np.concatenate(cols), size)


def preview_studio(raw: bytes, skins_raw: bytes | None = None, size: int = 384,
		seq: int = 0, frame: int = 0) -> "np.ndarray":
	"""Render the geometry the engine will load, coloured from the (companion) skins."""
	from .studio import studio_textures

	skins = studio_textures(skins_raw if skins_raw is not None else raw)
	sizes = [rgba.shape[:2] for _n, _f, rgba in studio_textures(raw)]	# s/t are in the full skin's texels
	skin_cols = [_mean_colour(rgba) for _n, _f, rgba in skins]
	tris, cols = [], []
	v = studio_posed_vertices(raw, seq, frame)
	for m in studio_meshes(raw):
		k = m["skin"]
		col = skin_cols[k] if k < len(skin_cols) else np.array([255.0, 255.0, 255.0])
		sample = k < len(skins) and k < len(sizes) and skins[k][2][..., 3].min() >= 128
		for tri in m["tris"]:
			tris.append(v[[c[0] for c in tri]])
			if sample:
				h, w = sizes[k]
				st = np.array([[c[2], c[3]] for c in tri], np.float64).mean(0)
				col = _sample_colour(skins[k][2], np.array([st[0] / w]), np.array([st[1] / h]))[0]
			cols.append(col)
	if not tris:
		return render_preview(np.zeros((0, 3, 3)), np.zeros((0, 3)), size)
	return render_preview(np.array(tris), np.array(cols), size)
