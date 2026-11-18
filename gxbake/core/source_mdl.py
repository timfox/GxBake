"""Source engine studio models (.mdl v44–49 + .vvd + .dx90.vtx) → ImportedModel.

The GameCube port draws GoldSrc studio models, so a Source model is read
natively here and handed to models.build_studio_mdl like any Assimp import:
bones (rest pose from the bone table), the root LOD of each bodypart's
default model, family-0 skins resolved through .vmt → .vtf, and every
non-delta sequence whose animation is stored in the .mdl itself.

Source and GoldSrc share axes (+X forward, +Z up) and units, so no axis
swap happens; ModelOptions.scale still applies. Studio skinning is rigid, so
each vertex follows its heaviest bone, as with Assimp imports.

Not carried over: flexes (facial morphs), LODs past the root, bodygroups
other than the first model in each bodypart, IK rules, animations stored in
.ani blocks, and delta (gesture/layer) sequences.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from . import skeleton as _sk
from . import vtf as _vtf

MDL_VERSIONS = range(44, 50)

STUDIO_ANIM_RAWPOS = 0x01
STUDIO_ANIM_RAWROT = 0x02
STUDIO_ANIM_ANIMPOS = 0x04
STUDIO_ANIM_ANIMROT = 0x08
STUDIO_ANIM_DELTA = 0x10
STUDIO_ANIM_RAWROT2 = 0x20
STUDIO_LOOPING = 0x0001
STUDIO_DELTA = 0x0004

_BONE = 216
_TEXTURE = 64
_BODYPART = 16
_MODEL = 148
_MESH = 116
_ANIMDESC = 100
_SEQDESC = 212
_VVD_VERTEX = 48


def is_source_mdl(head: bytes) -> bool:
	return len(head) >= 8 and head[:4] == b"IDST" and struct.unpack_from("<i", head, 4)[0] in MDL_VERSIONS


def _cstr(data: bytes, off: int) -> str:
	end = data.find(b"\0", off)
	return data[off:end if end >= 0 else len(data)].decode("latin-1")


def companion_files(mdl: Path) -> tuple[Path | None, Path | None]:
	"""The .vvd and the best .vtx next to a Source .mdl."""
	vvd = None
	vtx = None
	try:
		siblings = {p.name.lower(): p for p in mdl.parent.iterdir()}
	except OSError:
		siblings = {}
	stem = mdl.stem.lower()
	vvd = siblings.get(stem + ".vvd")
	for suffix in (".dx90.vtx", ".vtx", ".dx80.vtx", ".sw.vtx", ".xbox.vtx"):
		if stem + suffix in siblings:
			vtx = siblings[stem + suffix]
			break
	return vvd, vtx


# -- .mdl ---------------------------------------------------------------------------


class _Header:
	def __init__(self, d: bytes):
		if not is_source_mdl(d):
			raise ValueError("not a Source studio model (IDST v44-49)")
		self.version = struct.unpack_from("<i", d, 4)[0]
		self.name = d[12:76].split(b"\0")[0].decode("latin-1")
		(self.numbones, self.boneindex, _nbc, _bci, _nhs, _hsi, self.numanim, self.animindex,
			self.numseq, self.seqindex, _alv, _ei, self.numtextures, self.textureindex,
			self.numcdtextures, self.cdtextureindex, self.numskinref, self.numskinfamilies, self.skinindex,
			self.numbodyparts, self.bodypartindex) = struct.unpack_from("<21i", d, 156)


class SourceBone:
	def __init__(self, d: bytes, off: int):
		self.name = _cstr(d, off + struct.unpack_from("<i", d, off)[0])
		self.parent = struct.unpack_from("<i", d, off + 4)[0]
		self.pos = np.array(struct.unpack_from("<3f", d, off + 32), np.float64)
		self.quat = np.array(struct.unpack_from("<4f", d, off + 44), np.float64)
		self.rot = np.array(struct.unpack_from("<3f", d, off + 60), np.float64)
		self.posscale = np.array(struct.unpack_from("<3f", d, off + 72), np.float64)
		self.rotscale = np.array(struct.unpack_from("<3f", d, off + 84), np.float64)
		m = np.array(struct.unpack_from("<12f", d, off + 96), np.float64).reshape(3, 4)
		self.pose_to_bone = np.vstack([m, [0, 0, 0, 1]])


def read_bones(d: bytes, h: _Header) -> list[SourceBone]:
	return [SourceBone(d, h.boneindex + i * _BONE) for i in range(h.numbones)]


def read_textures(d: bytes, h: _Header) -> tuple[list[str], list[str], list[int]]:
	"""(texture names, cdmaterials dirs, family-0 skin → texture index)."""
	names = []
	for i in range(h.numtextures):
		off = h.textureindex + i * _TEXTURE
		names.append(_cstr(d, off + struct.unpack_from("<i", d, off)[0]))
	cds = [_cstr(d, struct.unpack_from("<i", d, h.cdtextureindex + i * 4)[0]) for i in range(h.numcdtextures)]
	skins = list(range(h.numtextures))
	if h.numskinref and h.numskinfamilies:
		skins = list(struct.unpack_from(f"<{h.numskinref}h", d, h.skinindex))
	return names, cds, skins


def _models(d: bytes, h: _Header) -> list[tuple[int, int, int]]:
	"""(bodypart index, model offset, model index) for the first model of each bodypart."""
	out = []
	for bp in range(h.numbodyparts):
		off = h.bodypartindex + bp * _BODYPART
		nummodels, _base, modelindex = struct.unpack_from("<iii", d, off + 4)
		for mi in range(min(nummodels, 1)):
			out.append((bp, off + modelindex + mi * _MODEL, mi))
	return out


# -- .vvd ---------------------------------------------------------------------------


def read_vvd(data: bytes) -> dict[str, np.ndarray]:
	"""Root-LOD vertices (fixups applied): weights, bones, nbones, pos, normal, uv."""
	if data[:4] != b"IDSV":
		raise ValueError("not a VVD file")
	_ver, _ck, numlods = struct.unpack_from("<iii", data, 4)
	lod_verts = struct.unpack_from("<8i", data, 16)
	numfixups, fixup_start, vert_start = struct.unpack_from("<iii", data, 48)
	n_all = lod_verts[0]
	raw = np.frombuffer(data, np.uint8, n_all * _VVD_VERTEX, vert_start).reshape(n_all, _VVD_VERTEX)
	if numfixups:
		order = []
		for i in range(numfixups):
			lod, src, num = struct.unpack_from("<iii", data, fixup_start + i * 12)
			if lod >= 0:
				order.extend(range(src, src + num))
		raw = raw[np.array(order, np.int64)]
	rec = np.ascontiguousarray(raw)
	f = rec.view(np.uint8)
	return {
		"weights": f[:, 0:12].copy().view("<f4").reshape(-1, 3),
		"bones": f[:, 12:15].astype(np.int64),
		"nbones": f[:, 15].astype(np.int64),
		"pos": f[:, 16:28].copy().view("<f4").reshape(-1, 3).astype(np.float64),
		"normal": f[:, 28:40].copy().view("<f4").reshape(-1, 3).astype(np.float64),
		"uv": f[:, 40:48].copy().view("<f4").reshape(-1, 2).astype(np.float32),
	}


# -- .vtx ---------------------------------------------------------------------------


def read_vtx_lod0(data: bytes, mdl_version: int = 48) -> dict[tuple[int, int], list[list["np.ndarray"]]]:
	"""{(bodypart, model): [per mesh: list of (n, 3) arrays of mesh-relative vertex ids]} for LOD 0."""
	version, _cache, _mbs, _mbt, _mbv, _ck, _nlods, _mrl, nbp, bp_off = struct.unpack_from("<iiHHiiiiii", data, 0)
	if version != 7:
		raise ValueError(f"unsupported VTX version {version}")
	out: dict = {}

	def strip_group_size() -> int:
		# MDL v49 (L4D and later) adds numTopologyIndices/topologyOffset to each strip group.
		return 33 if mdl_version >= 49 else 25

	sgs = strip_group_size()
	for bp in range(nbp):
		bpo = bp_off + bp * 8
		nmodels, mo = struct.unpack_from("<ii", data, bpo)
		for m in range(nmodels):
			mod = bpo + mo + m * 8
			nlods, lo = struct.unpack_from("<ii", data, mod)
			if nlods < 1:
				continue
			lod = mod + lo
			nmeshes, meo, _sw = struct.unpack_from("<iif", data, lod)
			meshes = []
			for me in range(nmeshes):
				mesh = lod + meo + me * 9
				nsg, sgo, _flags = struct.unpack_from("<iiB", data, mesh)
				tris = []
				for g in range(nsg):
					sg = mesh + sgo + g * sgs
					nverts, vo, nidx, io, nstrips, so = struct.unpack_from("<6i", data, sg)
					orig = np.array([struct.unpack_from("<H", data, sg + vo + v * 9 + 4)[0] for v in range(nverts)],
						np.int64)
					idx = np.frombuffer(data, "<u2", nidx, sg + io).astype(np.int64)
					for s in range(nstrips):
						st = sg + so + s * 27
						s_nidx, s_io, _snv, _svo, _snb, s_flags = struct.unpack_from("<iiiihB", data, st)
						sel = idx[s_io:s_io + s_nidx]
						if s_flags & 2:	# triangle strip
							t = []
							for k in range(len(sel) - 2):
								a, b, c = sel[k], sel[k + 1], sel[k + 2]
								t.append((a, b, c) if k % 2 == 0 else (b, a, c))
							sel = np.array(t, np.int64).reshape(-1)
						if len(sel) >= 3:
							tris.append(orig[sel[: len(sel) // 3 * 3]].reshape(-1, 3))
				meshes.append(tris)
			out[(bp, m)] = meshes
	return out


# -- animation ------------------------------------------------------------------------


def _half(bits: int) -> float:
	return float(np.array([bits], np.uint16).view(np.float16)[0])


def _quat48(d: bytes, off: int) -> np.ndarray:
	x, y, zw = struct.unpack_from("<HHH", d, off)
	q = np.array([(x - 32768) / 32768.0, (y - 32768) / 32768.0, ((zw & 0x7FFF) - 16384) / 16384.0, 0.0])
	q[3] = math.sqrt(max(0.0, 1.0 - q[0] ** 2 - q[1] ** 2 - q[2] ** 2)) * (-1 if zw >> 15 else 1)
	return q


def _quat64(d: bytes, off: int) -> np.ndarray:
	v = struct.unpack_from("<Q", d, off)[0]
	x, y, z = v & 0x1FFFFF, (v >> 21) & 0x1FFFFF, (v >> 42) & 0x1FFFFF
	q = np.array([(x - 1048576) / 1048576.5, (y - 1048576) / 1048576.5, (z - 1048576) / 1048576.5, 0.0])
	q[3] = math.sqrt(max(0.0, 1.0 - q[0] ** 2 - q[1] ** 2 - q[2] ** 2)) * (-1 if v >> 63 else 1)
	return q


def _anim_value(d: bytes, off: int, frame: int) -> float:
	"""ExtractAnimValue: RLE stream of (valid, total) headers followed by `valid` shorts."""
	k = frame
	for _ in range(4096):
		valid, total = d[off], d[off + 1]
		if total == 0:
			return 0.0
		if total > k:
			i = k if valid > k else valid - 1
			return float(struct.unpack_from("<h", d, off + 2 + 2 * i)[0])
		k -= total
		off += 2 * (valid + 1)
	return 0.0


def euler_quat(a: np.ndarray) -> np.ndarray:
	"""Source AngleQuaternion(RadianEuler): rotation Rz(a[2])·Ry(a[1])·Rx(a[0]) as (x, y, z, w)."""
	sr, cr = math.sin(a[0] * 0.5), math.cos(a[0] * 0.5)
	sp, cp = math.sin(a[1] * 0.5), math.cos(a[1] * 0.5)
	sy, cy = math.sin(a[2] * 0.5), math.cos(a[2] * 0.5)
	return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
		cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def _decode_frame(d: bytes, anim_off: int, bones: list[SourceBone], frame: int) -> list[tuple[np.ndarray, np.ndarray]]:
	"""(pos, quat) per bone for one frame of one mstudioanim_t chain."""
	pose = [(b.pos.copy(), b.quat.copy()) for b in bones]
	off = anim_off
	for _ in range(len(bones) + 1):
		if off + 4 > len(d):
			break
		bone, flags, nxt = struct.unpack_from("<BBh", d, off)
		if bone >= len(bones):
			break
		b = bones[bone]
		data = off + 4
		pos, quat = b.pos.copy(), b.quat.copy()
		if flags & STUDIO_ANIM_RAWROT:
			quat = _quat48(d, data)
		elif flags & STUDIO_ANIM_RAWROT2:
			quat = _quat64(d, data)
		elif flags & STUDIO_ANIM_ANIMROT:
			ang = np.zeros(3)
			for j in range(3):
				o = struct.unpack_from("<h", d, data + 2 * j)[0]
				ang[j] = _anim_value(d, data + o, frame) * b.rotscale[j] if o > 0 else 0.0
			if not flags & STUDIO_ANIM_DELTA:
				ang += b.rot
			quat = euler_quat(ang)
		pdata = data + (6 if flags & STUDIO_ANIM_RAWROT else 0) + (8 if flags & STUDIO_ANIM_RAWROT2 else 0)
		if flags & STUDIO_ANIM_RAWPOS:
			pos = np.array([_half(v) for v in struct.unpack_from("<3H", d, pdata)])
		elif flags & STUDIO_ANIM_ANIMPOS:
			pv = data + (6 if flags & STUDIO_ANIM_ANIMROT else 0)
			pos = np.zeros(3)
			for j in range(3):
				o = struct.unpack_from("<h", d, pv + 2 * j)[0]
				pos[j] = _anim_value(d, pv + o, frame) * b.posscale[j] if o > 0 else 0.0
			if not flags & STUDIO_ANIM_DELTA:
				pos += b.pos
		pose[bone] = (pos, quat)
		if nxt == 0:
			break
		off += nxt
	return pose


class AnimDesc:
	def __init__(self, d: bytes, off: int):
		self.off = off
		self.name = _cstr(d, off + struct.unpack_from("<i", d, off + 4)[0])
		self.fps, self.flags, self.numframes = struct.unpack_from("<fii", d, off + 8)
		self.animblock, self.animindex = struct.unpack_from("<ii", d, off + 52)
		self.sectionindex, self.sectionframes = struct.unpack_from("<ii", d, off + 80)

	def frame_chain(self, d: bytes, frame: int) -> tuple[int, int] | None:
		"""(absolute offset of the mstudioanim_t chain, frame within it), or None if external."""
		if self.sectionframes:
			sec = frame // self.sectionframes
			block, index = struct.unpack_from("<ii", d, self.off + self.sectionindex + sec * 8)
			if block:
				return None
			return self.off + index, frame - sec * self.sectionframes
		if self.animblock:
			return None
		return self.off + self.animindex, frame


def read_sequences(d: bytes, h: _Header) -> list[tuple[str, int, int]]:
	"""(label, flags, first anim index) per sequence."""
	out = []
	for i in range(h.numseq):
		off = h.seqindex + i * _SEQDESC
		label = _cstr(d, off + struct.unpack_from("<i", d, off + 4)[0])
		flags = struct.unpack_from("<i", d, off + 12)[0]
		animindexindex = struct.unpack_from("<i", d, off + 60)[0]
		anim = struct.unpack_from("<h", d, off + animindexindex)[0]
		out.append((label, flags, anim))
	return out


# -- import -------------------------------------------------------------------------------


def _local_matrix(pos: np.ndarray, quat: np.ndarray, scale: float) -> np.ndarray:
	m = np.eye(4)
	m[:3, :3] = _sk.quat_matrix(quat)
	m[:3, 3] = pos * scale
	return m


def source_skin(name: str, search_dirs: list[str], roots: list[Path], opts, warnings: list[str]):
	"""A Source material (.vmt → $basetexture .vtf) as a models.Material; grey when it can't be found."""
	from . import models as _m

	vtf_path, params = _vtf.resolve_material(name, search_dirs, roots) if roots else (None, {})
	rgba = None
	if vtf_path is not None:
		try:
			rgba = _vtf.load_vtf(vtf_path)
		except (OSError, ValueError) as exc:
			warnings.append(f"material {name}: {vtf_path.name}: {exc}")
	if rgba is None:
		if not roots:
			warnings.append(f"material {name}: no materials/ folder above the model; using grey")
		else:
			warnings.append(f"material {name}: .vmt/.vtf not found under {roots[0]}; using grey")
		rgba = np.full((8, 8, 4), (180, 180, 180, 255), np.uint8)
	cutout = params.get("$alphatest", "0") not in ("0", "") or params.get("$translucent", "0") not in ("0", "")
	if not cutout:
		rgba = rgba.copy()
		rgba[..., 3] = 255	# Source packs spec/phong masks in alpha; only cut-outs keep it
	rgba = _m._fit_skin(rgba, opts.skin_max_edge)
	masked = bool((rgba[..., 3] < opts.alpha_threshold).any())
	return _m.Material(Path(name).name or name, rgba, masked,
		str(vtf_path.relative_to(roots[0].parent)) if vtf_path is not None and roots else "")


def import_source(path: Path, opts=None):
	"""Read a Source .mdl (+ .vvd/.vtx) into models.ImportedModel."""
	from . import models as _m

	opts = opts or _m.ModelOptions()
	path = Path(path)
	d = path.read_bytes()
	h = _Header(d)
	vvd_path, vtx_path = companion_files(path)
	if vvd_path is None or vtx_path is None:
		raise ValueError(f"{path.name}: needs {path.stem}.vvd and {path.stem}.dx90.vtx next to it")
	vvd = read_vvd(vvd_path.read_bytes())
	vtx = read_vtx_lod0(vtx_path.read_bytes(), h.version)
	warnings: list[str] = []
	scale = float(opts.scale)
	sbones = read_bones(d, h)
	if not sbones:
		raise ValueError(f"{path.name}: no bones")

	# Bones: Source rest pose (pos + quat relative to the parent) → studio poses.
	static = bool(opts.static)
	if len(sbones) > _sk.MAXSTUDIOBONES:
		warnings.append(f"{len(sbones)} bones exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static instead")
		static = True
	locals_ = [_local_matrix(b.pos, b.quat, scale) for b in sbones]
	world: list[np.ndarray] = []
	for i, b in enumerate(sbones):
		world.append(locals_[i] if b.parent < 0 else world[b.parent] @ locals_[i])
	if static:
		bones = [_m.Bone("root", -1, np.zeros(6))]
	else:
		bones = [_m.Bone(b.name[:31] or f"bone{i}", b.parent, _sk.pose_from_matrix(locals_[i]))
			for i, b in enumerate(sbones)]

	# Skins: family 0, resolved through .vmt files under the nearest materials/ folder.
	tex_names, cd_dirs, skin_ref = read_textures(d, h)
	roots = [r for r in (_vtf.find_materials_root(path.parent),) if r is not None]
	materials: list = []
	mat_of_tex: dict[int, int] = {}

	def material(tex: int):
		if tex in mat_of_tex:
			return mat_of_tex[tex]
		name = tex_names[tex] if 0 <= tex < len(tex_names) else f"skin{tex}"
		mat = source_skin(name, cd_dirs, roots, opts, warnings)
		mat_of_tex[tex] = len(materials)
		materials.append(mat)
		return mat_of_tex[tex]

	model = _m.ImportedModel(path.stem, materials, [], warnings)
	model.bones = bones
	world_arr = np.array(world)
	bind = np.array([b.pose_to_bone for b in sbones])	# model space → bone space at bind

	flips = 0
	total = 0
	pending = []
	for bp, model_off, mi in _models(d, h):
		nmeshes, meshindex, nverts, vertexindex = struct.unpack_from("<iiii", d, model_off + 72)
		first = vertexindex // _VVD_VERTEX
		vtx_meshes = vtx.get((bp, mi), [])
		for me in range(nmeshes):
			mesh_off = model_off + meshindex + me * _MESH
			mat, _modelindex, mnverts, voff = struct.unpack_from("<iiii", d, mesh_off)
			if me >= len(vtx_meshes) or not vtx_meshes[me]:
				continue
			tris = np.concatenate(vtx_meshes[me]) + first + voff
			if not len(tris) or tris.max() >= len(vvd["pos"]):
				warnings.append(f"bodypart {bp} mesh {me}: vertex ids out of range; skipped")
				continue
			used = np.unique(tris)
			remap = np.full(int(used.max()) + 1, -1, np.int64)
			remap[used] = np.arange(len(used))
			pos = vvd["pos"][used] * scale
			nrm = vvd["normal"][used]
			uv = vvd["uv"][used]
			# the two heaviest influences and the first one's share of them
			nb = np.clip(vvd["nbones"][used], 1, 3)
			w = np.where(np.arange(3)[None] < nb[:, None], vvd["weights"][used], 0.0)
			order = np.argsort(-w, 1)[:, :2]
			rows = np.arange(len(used))[:, None]
			top = np.clip(vvd["bones"][used][rows, order], 0, len(sbones) - 1)
			tw = w[rows, order]
			share = np.where(tw.sum(1) > 0, tw[:, 0] / np.maximum(tw.sum(1), 1e-9), 1.0)
			local_tris = remap[tris]
			# Source fronts are clockwise; build_studio_mdl wants counter-clockwise like Assimp.
			a, b, c = pos[local_tris[:, 0]], pos[local_tris[:, 1]], pos[local_tris[:, 2]]
			fn = np.cross(b - a, c - a)
			vn = nrm[local_tris].sum(1)
			agree = np.einsum("ij,ij->i", fn, vn)
			flips += int((agree < 0).sum())
			total += len(agree)
			pending.append([material(skin_ref[mat] if 0 <= mat < len(skin_ref) else mat),
				pos, nrm, uv, local_tris, top, share])

	if not static:
		model.sequences = _sequences(d, h, sbones, scale, opts, warnings)
	bind_s = bind.copy()
	bind_s[:, :3, 3] *= scale
	vbones = [p[5][:, 0].copy() for p in pending]
	if not static and getattr(opts, "blend_bones", True):
		_blend_bones(pending, vbones, sbones, model, world_arr, bind_s, warnings)

	flip = flips * 2 > total
	for (mat_i, pos, nrm, uv, tris, top, _share), vbone in zip(pending, vbones):
		if flip:
			tris = tris[:, [0, 2, 1]]
		if static:
			local, ln, vb_out = pos, nrm, np.zeros(len(pos), np.int32)
		else:
			# bone space through the bind of the heaviest Source bone (its poseToBone)
			pb = bind_s[top[:, 0]]
			local = np.einsum("nij,nj->ni", pb[:, :3, :3], pos) + pb[:, :3, 3]
			ln = np.einsum("nij,nj->ni", pb[:, :3, :3], nrm)
			vb_out = vbone.astype(np.int32)
		lens = np.linalg.norm(ln, axis=1, keepdims=True)
		ln = np.where(lens > 1e-8, ln / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		model.meshes.append(_m.Submesh(mat_i, local.astype(np.float32), ln.astype(np.float32), uv,
			tris.astype(np.uint32), vb_out, pos.astype(np.float32)))
	if not model.meshes:
		raise ValueError(f"{path.name}: no triangles in LOD 0")
	if h.numbodyparts and any(struct.unpack_from("<i", d, h.bodypartindex + i * _BODYPART + 4)[0] > 1
			for i in range(h.numbodyparts)):
		warnings.append("bodygroups: only the first model of each bodypart is converted")

	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _mat_quat(m: np.ndarray) -> np.ndarray:
	"""Rotation part of a 4x4 as a unit quaternion (w, x, y, z)."""
	r = m[:3, :3]
	tr = r[0, 0] + r[1, 1] + r[2, 2]
	if tr > 0:
		s = math.sqrt(tr + 1.0) * 2
		q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
	elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
		s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
		q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
	elif r[1, 1] > r[2, 2]:
		s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
		q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
	else:
		s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
		q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
	q = np.array(q)
	return q / np.linalg.norm(q)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	aw, ax, ay, az = a
	bw, bx, by, bz = b
	return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
		aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def dq_blend(ma: np.ndarray, mb: np.ndarray, wa: float) -> np.ndarray:
	"""Dual-quaternion blend of two rigid 4x4 transforms (weight wa on ma)."""
	ra, rb = _mat_quat(_sk.rigid(ma)), _mat_quat(_sk.rigid(mb))
	if np.dot(ra, rb) < 0:
		rb = -rb
	da = 0.5 * _qmul(np.array([0.0, *ma[:3, 3]]), ra)
	db = 0.5 * _qmul(np.array([0.0, *mb[:3, 3]]), rb)
	r = wa * ra + (1 - wa) * rb
	dual = wa * da + (1 - wa) * db
	n = np.linalg.norm(r)
	r, dual = r / n, dual / n
	t = 2 * _qmul(dual, r * np.array([1, -1, -1, -1]))[1:]
	out = np.eye(4)
	out[:3, :3] = _sk.quat_matrix(np.array([r[1], r[2], r[3], r[0]]))
	out[:3, 3] = t
	return out


BLEND_STEPS = (4, 2)	# weight resolutions tried for helper bones (quarters, then halves)


def _blend_bones(pending, vbones, sbones, model, world_rest, bind_s, warnings) -> None:
	"""Emulate two-bone blending with extra rigid bones.

	Studio skinning gives each vertex one bone. A vertex shared between bones A
	and B at weight w would tear at joints, so it gets a helper bone parented to
	A whose motion each frame is the dual-quaternion blend of w·Skin(A) and
	(1-w)·Skin(B), where Skin(X) = world(X) · poseToBone(X). A dual-quaternion
	blend stays rigid and turns about the shared joint, like the elbow it is. The helper rests where A rests, so
	the vertex keeps A's bone-space coordinates. Weights are quantised so a
	model needs few helpers; the finest resolution that fits the 128-bone
	limit wins, else vertices fall back to their heaviest bone.
	"""
	from . import models as _m

	nb = len(sbones)
	room = _sk.MAXSTUDIOBONES - nb
	keys_per_mesh = None
	for steps in BLEND_STEPS:
		trial = []
		uniq: set = set()
		for (_mi, _p, _n, _uv, _t, top, share) in pending:
			q = np.round(share * steps) / steps
			blended = (q < 1.0) & (top[:, 0] != top[:, 1])
			trial.append((q, blended))
			uniq.update(zip(top[blended, 0].tolist(), top[blended, 1].tolist(), q[blended].tolist()))
		if len(uniq) <= room:
			keys_per_mesh = (trial, sorted(uniq))
			break
	if keys_per_mesh is None:
		warnings.append("too many weighted joints for the 128-bone limit; blended vertices follow their heaviest bone")
		return
	trial, keys = keys_per_mesh
	if not keys:
		return
	index = {k: nb + i for i, k in enumerate(keys)}
	parents = [b.parent for b in sbones]
	for i, (a, b, q) in enumerate(keys):
		model.bones.append(_m.Bone(f"{sbones[a].name[:22]}~{i}"[:31], a, np.zeros(6)))
	for sq in model.sequences:
		extra = np.zeros((len(sq.frames), len(keys), 6))
		for f, frame in enumerate(sq.frames):
			wm = np.array(_sk.world_matrices(frame[:nb], parents))
			skin = wm @ bind_s
			for i, (a, b, q) in enumerate(keys):
				s_blend = dq_blend(skin[a], skin[b], q)
				extra[f, i] = _sk.pose_from_matrix(_sk.rigid_inverse(wm[a]) @ s_blend @ world_rest[a])
		sq.frames = np.concatenate([sq.frames, extra], 1)
	for vb, (_mi, _p, _n, _uv, _t, top, _s), (q, blended) in zip(vbones, pending, trial):
		for v in np.nonzero(blended)[0]:
			vb[v] = index[(int(top[v, 0]), int(top[v, 1]), float(q[v]))]
	warnings.append(f"{len(keys)} helper bones emulate blended skin weights (studio skinning is rigid)")


def _sequences(d: bytes, h: _Header, sbones: list[SourceBone], scale: float, opts, warnings: list[str]) -> list:
	from . import models as _m

	descs = [AnimDesc(d, h.animindex + i * _ANIMDESC) for i in range(h.numanim)]
	used: set[str] = set()
	out = []
	external = delta = 0
	for label, flags, anim in read_sequences(d, h):
		if not 0 <= anim < len(descs):
			continue
		desc = descs[anim]
		if flags & STUDIO_DELTA or desc.flags & STUDIO_DELTA:
			delta += 1
			continue
		n = max(1, desc.numframes)
		src_fps = desc.fps if desc.fps > 0 else 30.0
		fps = float(min(src_fps, opts.anim_fps)) if opts.anim_fps else src_fps
		count = max(1, int(round((n - 1) / src_fps * fps)) + 1) if n > 1 else 1
		picks = [min(n - 1, int(round(k * src_fps / fps))) for k in range(count)]
		frames = np.zeros((count, len(sbones), 6))
		ok = True
		for k, f in enumerate(picks):
			chain = desc.frame_chain(d, f)
			if chain is None:
				ok = False
				break
			pose = _decode_frame(d, chain[0], sbones, chain[1])
			for b, (p, q) in enumerate(pose):
				frames[k, b] = _sk.pose_from_matrix(_local_matrix(p, q, scale))
		if not ok:
			external += 1
			continue
		out.append(_m.Sequence(_m._unique(_m._seq_name(label.lstrip("@")), used), fps, frames,
			loop=bool(flags & STUDIO_LOOPING)))
	if external:
		warnings.append(f"{external} sequence(s) keep their animation in a .ani file; skipped")
	if delta:
		warnings.append(f"{delta} delta (gesture/layer) sequence(s) skipped")
	return out


def source_info(path: Path) -> dict:
	d = Path(path).read_bytes()
	h = _Header(d)
	vvd, vtx = companion_files(Path(path))
	return {"Source MDL version": h.version, "Source bones": h.numbones, "Source sequences": h.numseq,
		"Textures": h.numtextures, "VVD": vvd.name if vvd else "missing", "VTX": vtx.name if vtx else "missing"}
