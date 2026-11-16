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

# Extensions sent to Assimp. .mdl stays with the GoldSrc studio path.
MODEL_EXTENSIONS = {
	".obj", ".fbx", ".gltf", ".glb", ".dae", ".3ds", ".ply", ".stl", ".x",
	".lwo", ".lws", ".ase", ".ms3d", ".off", ".blend", ".3mf", ".ac", ".smd",
}

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


@dataclass
class Material:
	name: str
	rgba: "np.ndarray"	# skin texels (power-of-two)
	masked: bool
	texture: str = ""	# where the texels came from


@dataclass
class Submesh:
	material: int
	positions: "np.ndarray"	# (n, 3) float32, GoldSrc axes, scaled
	normals: "np.ndarray"	# (n, 3) float32
	uvs: "np.ndarray"	# (n, 2) float32, top-left origin
	tris: "np.ndarray"	# (m, 3) uint32, counter-clockwise like the source


@dataclass
class ImportedModel:
	name: str
	materials: list[Material]
	meshes: list[Submesh]
	warnings: list[str] = field(default_factory=list)

	@property
	def num_vertices(self) -> int:
		return sum(len(m.positions) for m in self.meshes)

	@property
	def num_triangles(self) -> int:
		return sum(len(m.tris) for m in self.meshes)

	def bounds(self) -> tuple["np.ndarray", "np.ndarray"]:
		if not self.meshes:
			z = np.zeros(3, np.float32)
			return z, z
		pts = np.concatenate([m.positions for m in self.meshes])
		return pts.min(0), pts.max(0)


# -- import -------------------------------------------------------------------


def import_model(path: Path, opts: ModelOptions | None = None) -> ImportedModel:
	"""Read any Assimp-supported file into flattened, triangulated submeshes."""
	opts = opts or ModelOptions()
	ai = _assimp()
	path = Path(path)
	flags = (
		ai.Process_Triangulate
		| ai.Process_PreTransformVertices	# bake the node hierarchy; GoldSrc output is one static bone
		| ai.Process_JoinIdenticalVertices
		| ai.Process_GenSmoothNormals
		| ai.Process_SortByPType
		| ai.Process_FindDegenerates
		| ai.Process_RemoveRedundantMaterials
		| ai.Process_FlipUVs	# top-left texel origin, like studio s/t
		| ai.Process_ValidateDataStructure
	)
	try:
		scene = ai.import_file(str(path), flags)
	except FileNotFoundError:
		raise
	except Exception as exc:	# noqa: BLE001 - assimp raises bare RuntimeErrors with its log text
		raise ValueError(f"Assimp could not read {path.name}: {exc}") from exc
	if scene is None or not scene.meshes:
		raise ValueError(f"{path.name}: no meshes")

	model = ImportedModel(path.stem, [], [])
	embedded = _embedded_images(path)
	mat_map: dict[int, int] = {}
	for mesh in scene.meshes:
		if mesh.num_faces == 0 or mesh.num_vertices == 0 or mesh.indices is None:
			continue
		idx = np.frombuffer(mesh.indices, np.uint32)
		if len(idx) % 3:
			continue	# points/lines left over by SortByPType
		tris = idx.reshape(-1, 3)
		pos = np.frombuffer(mesh.vertices, np.float32).reshape(-1, 3).astype(np.float64)
		nrm = (np.frombuffer(mesh.normals, np.float32).reshape(-1, 3).astype(np.float64)
			if mesh.normals is not None else np.tile([0.0, 0.0, 1.0], (len(pos), 1)))
		uv = np.zeros((len(pos), 2), np.float32)
		if mesh.texcoords and mesh.texcoords[0] is not None:
			comps = mesh.num_uv_components[0] if mesh.num_uv_components else 2
			t = np.frombuffer(mesh.texcoords[0], np.float32)
			comps = len(t) // len(pos) if len(pos) else comps
			uv = t.reshape(len(pos), comps)[:, :2].copy()
		pos = _to_goldsrc(pos, opts.up_axis) * opts.scale
		nrm = _to_goldsrc(nrm, opts.up_axis)
		lens = np.linalg.norm(nrm, axis=1, keepdims=True)
		nrm = np.where(lens > 1e-8, nrm / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		mi = mesh.material_index
		if mi not in mat_map:
			mat_map[mi] = len(model.materials)
			mdesc = scene.materials[mi] if 0 <= mi < len(scene.materials) else {}
			model.materials.append(_load_material(mdesc, path, embedded, opts, model.warnings))
		model.meshes.append(Submesh(mat_map[mi], pos.astype(np.float32), nrm.astype(np.float32), uv, tris.copy()))
	if not model.meshes:
		raise ValueError(f"{path.name}: no triangle meshes")
	nanim = len(getattr(scene, "animations", None) or [])	# assimp-py 1.1 has no animation API
	if nanim:
		model.warnings.append(f"{nanim} animation(s) ignored; the model is baked in its bind pose")
	if any(getattr(m, "num_bones", 0) for m in scene.meshes):
		model.warnings.append("skinned mesh: bones are flattened into one static root bone")
	return model


def _to_goldsrc(v: "np.ndarray", up_axis: str) -> "np.ndarray":
	"""Y-up (+Z forward, +X left of the model) → GoldSrc (+X forward, +Y left, +Z up)."""
	if up_axis == "z":
		return v
	return v[:, [2, 0, 1]]


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


def _load_texture(ref: str, model_path: Path, embedded: list[bytes]) -> "np.ndarray | None":
	from PIL import Image

	if ref.startswith("*"):
		try:
			data = embedded[int(ref[1:])]
		except (ValueError, IndexError):
			return None
		with Image.open(io.BytesIO(data)) as img:
			return np.asarray(img.convert("RGBA")).copy()
	ref = ref.replace("\\", "/")
	base = model_path.parent
	cands = [base / ref, base / Path(ref).name]
	if ref.lower() != ref:
		cands.append(base / Path(ref).name.lower())
	# Exporters often leave absolute paths from the artist's machine; try the
	# basename next to the model and in a textures/ folder.
	cands += [base / "textures" / Path(ref).name, base.parent / "textures" / Path(ref).name]
	for c in cands:
		try:
			if c.is_file():
				with Image.open(c) as img:
					return np.asarray(img.convert("RGBA")).copy()
		except (OSError, ValueError):
			continue
	return None


def _embedded_images(path: Path) -> list[bytes]:
	"""Images packed in a .glb's binary chunk, in glTF image order ("*N" refs)."""
	if path.suffix.lower() != ".glb":
		return []
	try:
		raw = path.read_bytes()
		if raw[:4] != b"glTF":
			return []
		off = 12
		doc, binchunk = None, b""
		while off + 8 <= len(raw):
			clen, ctype = struct.unpack_from("<I4s", raw, off)
			chunk = raw[off + 8: off + 8 + clen]
			if ctype == b"JSON":
				doc = json.loads(chunk)
			elif ctype == b"BIN\0":
				binchunk = chunk
			off += 8 + clen
		if not doc:
			return []
		out = []
		for img in doc.get("images", []):
			bv = doc["bufferViews"][img["bufferView"]] if "bufferView" in img else None
			if bv is None:
				out.append(b"")
				continue
			s = bv.get("byteOffset", 0)
			out.append(binchunk[s: s + bv["byteLength"]])
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
	"""Bake an imported model into a static GoldSrc studio MDL (v10)."""
	opts = opts or ModelOptions()
	if len(model.materials) > MAXSTUDIOSKINS:
		raise ValueError(f"{len(model.materials)} materials; studio models allow {MAXSTUDIOSKINS}")

	# Positions are shared model-wide. Normals are laid out per mesh, because the
	# engine lights mesh j's normals as one contiguous run (mstudiomesh_t.numnorms).
	pos_index: dict[tuple, int] = {}
	positions: list[tuple[float, float, float]] = []
	normals: list[tuple[float, float, float]] = []
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
				p = tuple(np.round(sm.positions[v].astype(np.float64), 4))
				vi = pos_index.get(p)
				if vi is None:
					vi = pos_index[p] = len(positions)
					positions.append(p)
				n = tuple(np.round(sm.normals[v].astype(np.float64), 3))
				ni = norm_index.get(n)
				if ni is None:
					ni = norm_index[n] = len(normals)
					normals.append(n)
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

	pts = np.array(positions, np.float32).reshape(-1, 3)
	bmin = pts.min(0) if len(pts) else np.zeros(3, np.float32)
	bmax = pts.max(0) if len(pts) else np.zeros(3, np.float32)
	radius = float(np.linalg.norm(pts, axis=1).max()) if len(pts) else 0.0

	out = bytearray(_HDR)
	name = (model.name.lower() + ".mdl").encode("latin-1", "replace")[:63]

	def align():
		out.extend(bytes((-len(out)) % 4))

	# bones: one static root
	bone_off = len(out)
	out += struct.pack("<32sii6i6f6f", b"root", -1, 0, *([-1] * 6), *([0.0] * 6), *([1.0] * 6))

	# sequence 0 "idle", one frame, all-zero anim offsets (bone value[] holds the pose)
	anim_off = len(out)
	out += bytes(12)	# mstudioanim_t: 6 unsigned short offsets for the single bone
	align()
	seq_off = len(out)
	seq = bytearray(_SEQ)
	struct.pack_into("<32sfiiiiiiii", seq, 0, b"idle", 30.0, 0, 0, 0, 0, 0, 1, 0, 0)
	struct.pack_into("<ii", seq, 68, 0, 0)	# motiontype, motionbone
	struct.pack_into("<3f3f", seq, 96, *bmin, *bmax)	# bbmin, bbmax
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
	out += bytes(len(positions))
	align()
	norminfo_off = len(out)
	out += bytes(len(normals))
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
	struct.pack_into("<10i", out, 140, 1, bone_off, 0, 0, 0, 0, 1, seq_off, 1, group_off)
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
		out.append({"skin": skin, "tris": tris, "verts": verts, "norms": norms})
	return out


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


def preview_imported(model: ImportedModel, size: int = 384) -> "np.ndarray":
	tris, cols = [], []
	for m in model.meshes:
		col = _mean_colour(model.materials[m.material].rgba)
		t = m.positions[m.tris][:, [0, 2, 1]]	# to clockwise, like the studio output
		tris.append(t)
		cols.append(np.tile(col, (len(t), 1)))
	if not tris:
		return render_preview(np.zeros((0, 3, 3)), np.zeros((0, 3)), size)
	return render_preview(np.concatenate(tris), np.concatenate(cols), size)


def preview_studio(raw: bytes, skins_raw: bytes | None = None, size: int = 384) -> "np.ndarray":
	"""Render the geometry the engine will load, coloured from the (companion) skins."""
	from .studio import studio_textures

	skins = studio_textures(skins_raw if skins_raw is not None else raw)
	skin_cols = [_mean_colour(rgba) for _n, _f, rgba in skins]
	tris, cols = [], []
	for m in studio_meshes(raw):
		col = skin_cols[m["skin"]] if m["skin"] < len(skin_cols) else np.array([255.0, 255.0, 255.0])
		v = m["verts"]
		for tri in m["tris"]:
			tris.append(v[[c[0] for c in tri]])
			cols.append(col)
	if not tris:
		return render_preview(np.zeros((0, 3, 3)), np.zeros((0, 3)), size)
	return render_preview(np.array(tris), np.array(cols), size)
