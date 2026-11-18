"""Source 2 models (.vmdl_c, .vmesh_c) → models.ImportedModel → GoldSrc studio MDL.

A vmdl_c holds the skeleton (m_modelSkeleton) and either embeds its meshes
(CTRL "embedded_meshes": an MDAT draw-call block plus VBIB/MBUF or MVTX/MIDX
buffers) or names external .vmesh_c files (m_refMeshes). Vertex and index
buffers may be meshoptimizer and/or Zstandard compressed. Only LOD 0 and the
default mesh group (bodygroup choice) are converted, as with Source 1 models.

Skinning: vertices are stored in model space at bind pose. Each vertex goes to
its heaviest bone; two-bone weights get the same dual-quaternion helper bones
as Source 1 imports (source_mdl._blend_bones). Skins come from each draw call's
material: its colour texture is read from the compiled .vtex_c under the
content root (the folder holding materials/ and models/).

Embedded animations (ANIM/ASEQ/AGRP blocks and .vagrp_c groups) are decoded by
vanim.py. Buffer layouts follow ValveResourceFormat (MIT, see
THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import kv3, skeleton as _sk, valvecomp, vres, vtex

# DXGI_FORMAT values seen in Source 2 vertex layouts
_DXGI = {
	2: ("<f4", 4), 4: ("<i4", 4), 6: ("<f4", 3), 10: ("<f2", 4), 11: ("<u2", 4), 12: ("<u2", 4), 14: ("<i2", 4),
	16: ("<f4", 2), 28: ("u1", 4), 30: ("u1", 4), 31: ("i1", 4), 34: ("<f2", 2), 35: ("<u2", 2), 37: ("<i2", 2),
	38: ("<i2", 2), 41: ("<f4", 1), 42: ("<u4", 1),
}
_NORMALISED = {11: 65535.0, 28: 255.0, 31: 127.0, 35: 65535.0, 37: 32767.0}


@dataclass
class Attribute:
	name: str
	index: int
	format: int
	offset: int


@dataclass
class Buffer:
	count: int
	stride: int
	attributes: list[Attribute]
	data: bytes

	def attribute(self, name: str, index: int = 0) -> Attribute | None:
		for a in self.attributes:
			if a.name == name and a.index == index:
				return a
		return None

	def read(self, attr: Attribute) -> "np.ndarray":
		"""(count, components) float64 (normalised formats scaled to 0..1 / -1..1)."""
		if attr.format not in _DXGI:
			raise ValueError(f"{attr.name}: unsupported vertex format {attr.format}")
		dt, n = _DXGI[attr.format]
		item = np.dtype(dt).itemsize * n
		raw = np.frombuffer(self.data, np.uint8)[:self.count * self.stride].reshape(self.count, self.stride)
		cols = np.ascontiguousarray(raw[:, attr.offset:attr.offset + item])
		v = cols.view(dt).reshape(self.count, n).astype(np.float64)
		if attr.format in _NORMALISED:
			v = v / _NORMALISED[attr.format]
		return v

	def read_raw(self, attr: Attribute) -> "np.ndarray":
		dt, n = _DXGI[attr.format]
		item = np.dtype(dt).itemsize * n
		raw = np.frombuffer(self.data, np.uint8)[:self.count * self.stride].reshape(self.count, self.stride)
		return np.ascontiguousarray(raw[:, attr.offset:attr.offset + item]).view(dt).reshape(self.count, n)


def _decompress(count: int, stride: int, raw: bytes, vertex: bool, meshopt: bool, zstd: bool) -> bytes:
	want = count * stride
	if zstd:
		raw = valvecomp.zstd(raw) if raw else b""
	if not meshopt:
		if len(raw) < want:
			raise ValueError("vertex/index buffer is shorter than its element count")
		return bytes(raw[:want])
	if len(raw) == want and not zstd:
		return bytes(raw)
	if vertex:
		return valvecomp.meshopt_vertex(count, stride, raw)
	return valvecomp.meshopt_index(count, stride, raw)


def _read_vbib_block(res: vres.Resource, b: vres.Block) -> tuple[list[Buffer], list[Buffer]]:
	d = res.data
	p = b.offset
	voff, vcount, ioff, icount = struct.unpack_from("<IIII", d, p)

	def one(at: int, vertex: bool) -> Buffer:
		count, size = struct.unpack_from("<Ii", d, at)
		stride = size & 0x3FFFFFF
		meshopt = (size & 0x4000000) == 0
		zstd = (size & 0x8000000) != 0
		aoff, acount = struct.unpack_from("<II", d, at + 8)
		doff, total = struct.unpack_from("<Ii", d, at + 16)
		attrs = []
		ap = at + 8 + aoff
		for _ in range(acount):
			name = d[ap:ap + 32].split(b"\0")[0].decode("latin-1").upper()
			sidx, fmt, off = struct.unpack_from("<iII", d, ap + 32)
			attrs.append(Attribute(name, sidx, fmt, off))
			ap += 56
		raw = d[at + 16 + doff:at + 16 + doff + total]
		if count * stride > total:
			data = _decompress(count, stride, raw, vertex, meshopt, zstd)
		else:
			data = bytes(raw)
		return Buffer(count, stride, attrs, data)

	vbs = [one(p + voff + i * 24, True) for i in range(vcount)]
	ibs = [one(p + 8 + ioff + i * 24, False) for i in range(icount)]
	return vbs, ibs


def _buffer_from_kv(res: vres.Resource, kv: dict, vertex: bool) -> Buffer:
	count = int(kv.get("m_nElementCount", 0))
	stride = int(kv.get("m_nElementSizeInBytes", 0))
	attrs = []
	for f in kv.get("m_inputLayoutFields") or []:
		name = f.get("m_pSemanticName", "")
		if isinstance(name, (bytes, bytearray)):
			name = bytes(name).split(b"\0")[0].decode("latin-1")
		elif not isinstance(name, str):
			name = kv3.blob(name).split(b"\0")[0].decode("latin-1")
		attrs.append(Attribute(str(name).upper(), int(f.get("m_nSemanticIndex", 0)), int(f.get("m_Format", 0)),
			int(f.get("m_nOffset", 0))))
	if "m_pData" in kv and kv["m_pData"] is not None:
		raw = kv3.blob(kv["m_pData"])
		data = raw if len(raw) == count * stride else _decompress(count, stride, raw, vertex, True, False)
	else:
		blk = res.blocks[int(kv.get("m_nBlockIndex", -1))]
		raw = res.raw(blk)
		meshopt = bool(kv.get("m_bMeshoptCompressed"))
		zstd = bool(kv.get("m_bCompressedZSTD"))
		data = _decompress(count, stride, raw, vertex, meshopt, zstd) if (meshopt or zstd) else bytes(raw)
	return Buffer(count, stride, attrs, data)


@dataclass
class MeshData:
	name: str
	index: int	# mesh index in the model (LOD / mesh group / bone remap tables)
	data: dict	# PermRenderMeshData_t
	vbs: list[Buffer]
	ibs: list[Buffer]


def _mesh_buffers(res: vres.Resource, data: dict, vbib: vres.Block | None) -> tuple[list[Buffer], list[Buffer]]:
	if vbib is not None:
		return _read_vbib_block(res, vbib)
	for kind in ("VBIB", "MBUF"):
		b = res.block(kind)
		if b is not None:
			return _read_vbib_block(res, b)
	vbs = [_buffer_from_kv(res, v, True) for v in data.get("m_vertexBuffers") or []]
	ibs = [_buffer_from_kv(res, v, False) for v in data.get("m_indexBuffers") or []]
	return vbs, ibs


def model_meshes(res: vres.Resource, roots: list[Path], warnings: list[str]) -> list[MeshData]:
	out = []
	ctrl = res.keyvalues("CTRL") if res.block("CTRL") else None
	for em in (ctrl or {}).get("embedded_meshes") or []:
		if "vbib_block" in em:
			data = res.keyvalues(int(em["data_block"]))
			vbs, ibs = _read_vbib_block(res, res.blocks[int(em["vbib_block"])])
			out.append(MeshData(str(em.get("name", "")), int(em.get("mesh_index", len(out))), data, vbs, ibs))
		else:
			data = res.keyvalues(int(em["m_nDataBlock"]))
			vbs = [_buffer_from_kv(res, v, True) for v in em.get("m_vertexBuffers") or []]
			ibs = [_buffer_from_kv(res, v, False) for v in em.get("m_indexBuffers") or []]
			out.append(MeshData(str(em.get("m_Name", "")), int(em.get("m_nMeshIndex", len(out))), data, vbs, ibs))
	d = res.keyvalues() or {}
	for i, ref in enumerate(d.get("m_refMeshes") or []):
		if not ref:
			continue
		path = vtex.find_compiled(str(ref), roots)
		if path is None:
			warnings.append(f"mesh {ref} not found under the content root")
			continue
		mres = vres.load(path)
		data = mres.keyvalues() or {}
		vbs, ibs = _mesh_buffers(mres, data, None)
		out.append(MeshData(Path(str(ref)).stem, i, data, vbs, ibs))
	return out


# -- skeleton ---------------------------------------------------------------------


@dataclass
class S2Bone:
	name: str
	parent: int
	pos: "np.ndarray"
	quat: "np.ndarray"	# (x, y, z, w) as stored


def read_skeleton(d: dict) -> list[S2Bone]:
	sk = d.get("m_modelSkeleton") or {}
	names = list(sk.get("m_boneName") or [])
	parents = kv3.ints(sk.get("m_nParent"))
	pos = sk.get("m_bonePosParent") or []
	rot = sk.get("m_boneRotParent") or []
	out = []
	for i, n in enumerate(names):
		p = kv3.floats(pos[i], 3) if i < len(pos) else np.zeros(3)
		q = kv3.floats(rot[i], 4) if i < len(rot) else np.array([0, 0, 0, 1.0])
		out.append(S2Bone(str(n), int(parents[i]) if i < len(parents) else -1, p, q))
	return out


def quat_xyzw_matrix(q: "np.ndarray") -> "np.ndarray":
	return _sk.quat_matrix(np.asarray(q, np.float64))


def local_matrix(pos: "np.ndarray", quat_xyzw: "np.ndarray", scale: float = 1.0) -> "np.ndarray":
	m = np.eye(4)
	m[:3, :3] = quat_xyzw_matrix(quat_xyzw)
	m[:3, 3] = np.asarray(pos) * scale
	return m


# -- import -------------------------------------------------------------------------


def _lod0(d: dict, index: int) -> bool:
	masks = kv3.ints(d.get("m_refLODGroupMasks"))
	if index >= len(masks) or not len(masks):
		return True
	combined = int(np.bitwise_or.reduce(masks)) if len(masks) else 0
	lowest = (combined & -combined).bit_length() - 1 if combined else 0
	return masks[index] == 0 or bool(masks[index] & (1 << lowest))


def _default_group(d: dict, index: int) -> bool:
	masks = kv3.ints(d.get("m_refMeshGroupMasks"))
	default = d.get("m_nDefaultMeshGroupMask")
	if default is None or index >= len(masks):
		return True
	return bool(int(masks[index]) & int(default)) or int(default) == 0


@dataclass
class _Material:
	index: int
	tool: bool = False


def import_vmdl(path: Path, opts=None):
	"""Read a .vmdl_c (or a lone .vmesh_c) into models.ImportedModel."""
	from . import models as _m
	from . import source_mdl as _s1

	opts = opts or _m.ModelOptions()
	path = Path(path)
	res = vres.load(path)
	roots = vtex.content_roots(path.parent)
	warnings: list[str] = []
	scale = float(opts.scale)
	kind = res.kind
	if kind == "mesh":
		data = res.keyvalues() or {}
		vbs, ibs = _mesh_buffers(res, data, None)
		meshes = [MeshData(path.stem, 0, data, vbs, ibs)]
		d: dict = {}
	else:
		d = res.keyvalues() or {}
		meshes = [m for m in model_meshes(res, roots, warnings) if _lod0(d, m.index) and _default_group(d, m.index)]
	sbones = read_skeleton(d)
	static = bool(opts.static) or not sbones
	if len(sbones) > _sk.MAXSTUDIOBONES:
		warnings.append(f"{len(sbones)} bones exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static instead")
		static = True
	locals_ = [local_matrix(b.pos, b.quat, scale) for b in sbones]
	world: list[np.ndarray] = []
	for i, b in enumerate(sbones):
		world.append(locals_[i] if b.parent < 0 or b.parent >= i else world[b.parent] @ locals_[i])
	world_arr = np.array(world) if world else np.eye(4)[None]
	if static:
		bones = [_m.Bone("root", -1, np.zeros(6))]
	else:
		bones = [_m.Bone(b.name[:31] or f"bone{i}", b.parent if b.parent < i else -1, _sk.pose_from_matrix(locals_[i]))
			for i, b in enumerate(sbones)]
	remap_table = kv3.ints(d.get("m_remappingTable"))
	remap_starts = kv3.ints(d.get("m_remappingTableStarts"))

	materials: list = []
	mat_index: dict[str, int] = {}

	def material(ref: str) -> int:
		if ref in mat_index:
			return mat_index[ref]
		rgba = None
		masked_hint = False
		src = ""
		mpath = vtex.find_compiled(ref, roots) if ref else None
		if mpath is None:
			warnings.append(f"material {ref or '(none)'}: .vmat_c not found; using grey")
		else:
			try:
				vm = vtex.load_vmat(mpath)
				masked_hint = vm.alpha_test or vm.translucent
				tref = vm.color_texture()
				tpath = vtex.find_compiled(tref, roots) if tref else None
				if tpath is None:
					warnings.append(f"material {Path(ref).stem}: colour texture {tref or '(none)'} not found; using grey")
				else:
					rgba = vtex.load_vtex(tpath)
					src = str(tpath.relative_to(roots[0])) if roots else tpath.name
			except (OSError, ValueError, RuntimeError) as exc:
				warnings.append(f"material {Path(ref).stem}: {exc}")
		if rgba is None:
			rgba = np.full((8, 8, 4), (180, 180, 180, 255), np.uint8)
		if not masked_hint:
			rgba = rgba.copy()
			rgba[..., 3] = 255	# alpha holds masks unless the material cuts out
		rgba = _m._fit_skin(rgba, opts.skin_max_edge)
		masked = bool((rgba[..., 3] < opts.alpha_threshold).any())
		mat_index[ref] = len(materials)
		materials.append(_m.Material(Path(ref).stem or f"skin{len(materials)}", rgba, masked, src))
		return mat_index[ref]

	model = _m.ImportedModel(path.name.split(".")[0], materials, [], warnings)
	model.bones = bones
	pending = []
	skipped_tools = 0
	nb = max(1, len(sbones))
	for mesh in meshes:
		table = None
		if len(remap_starts) and mesh.index < len(remap_starts):
			start = int(remap_starts[mesh.index])
			end = int(remap_starts[mesh.index + 1]) if mesh.index + 1 < len(remap_starts) else len(remap_table)
			table = remap_table[start:end]
		for so in mesh.data.get("m_sceneObjects") or []:
			for dc in so.get("m_drawCalls") or []:
				prim = dc.get("m_nPrimitiveType", "RENDER_PRIM_TRIANGLES")
				if prim not in ("RENDER_PRIM_TRIANGLES", 5, 4):
					continue
				mref = str(dc.get("m_material") or dc.get("m_pMaterial") or "")
				low = mref.lower()
				if "tools/" in low or "toolsnodraw" in low or low.endswith("nodraw.vmat"):
					skipped_tools += 1
					continue
				got = _draw_call(mesh, dc, table, nb)
				if got is None:
					continue
				pos, nrm, uv, tris, top, share = got
				pos = pos * scale
				pending.append([material(mref), pos, nrm, uv, tris, top, share])
	if skipped_tools:
		warnings.append(f"{skipped_tools} tool-material draw call(s) skipped")
	if not pending:
		why = f" ({warnings[0]})" if warnings else ""
		raise ValueError(f"{path.name}: no triangles in LOD 0{why}")

	bind_s = np.array([_sk.rigid_inverse(w) for w in world_arr]) if not static else np.eye(4)[None]
	if not static:
		model.sequences = _sequences(res, path, d, sbones, roots, scale, opts, warnings)
	vbones = [p[5][:, 0].copy() for p in pending]
	if not static and getattr(opts, "blend_bones", True):
		_s1._blend_bones(pending, vbones, sbones, model, world_arr, bind_s, warnings)
	flips = total = 0
	for p in pending:
		_mi, pos, nrm, _uv, tris, _top, _share = p
		a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
		agree = np.einsum("ij,ij->i", np.cross(b - a, c - a), nrm[tris].sum(1))
		flips += int((agree < 0).sum())
		total += len(agree)
	flip = flips * 2 > total
	for (mat_i, pos, nrm, uv, tris, top, _share), vbone in zip(pending, vbones):
		if flip:
			tris = tris[:, [0, 2, 1]]
		if static:
			local, ln, vb_out = pos, nrm, np.zeros(len(pos), np.int32)
		else:
			pb = bind_s[top[:, 0]]
			local = np.einsum("nij,nj->ni", pb[:, :3, :3], pos) + pb[:, :3, 3]
			ln = np.einsum("nij,nj->ni", pb[:, :3, :3], nrm)
			vb_out = vbone.astype(np.int32)
		lens = np.linalg.norm(ln, axis=1, keepdims=True)
		ln = np.where(lens > 1e-8, ln / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		model.meshes.append(_m.Submesh(mat_i, local.astype(np.float32), ln.astype(np.float32),
			uv.astype(np.float32), tris.astype(np.uint32), vb_out, pos.astype(np.float32)))
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _normals(vb: Buffer, attr: Attribute) -> "np.ndarray":
	if attr.format == 6:
		return vb.read(attr)
	if attr.format == 42:	# CS2 packed normal/tangent frame
		v = vb.read_raw(attr)[:, 0].astype(np.int64)
		x = ((v >> 12) & 0x3FF) / 1023.0 * 2 - 1
		y = ((v >> 22) & 0x3FF) / 1023.0 * 2 - 1
		z = 1 - np.abs(x) - np.abs(y)
		neg = np.clip(-z, 0, 1)
		x = x + np.where(x >= 0, -neg, neg)
		y = y + np.where(y >= 0, -neg, neg)
		n = np.stack([x, y, z], 1)
		return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
	if attr.format == 28:	# version 1 compressed normal in bytes 0-1
		raw = vb.read_raw(attr)[:, :2].astype(np.float64)
		x, y = raw[:, 0] - 128.0, raw[:, 1] - 128.0
		zs = (x < 0).astype(np.float64)
		ts = (y < 0).astype(np.float64)
		zsign, tsign = -(2 * zs - 1), -(2 * ts - 1)
		x = x * zsign - zs - 64
		y = y * tsign - ts - 64
		xs, ys = (x < 0).astype(np.float64), (y < 0).astype(np.float64)
		xsign, ysign = -(2 * xs - 1), -(2 * ys - 1)
		x = (x * xsign - xs) / 63.0
		y = (y * ysign - ys) / 63.0
		z = 1.0 - x - y
		oolen = 1.0 / np.sqrt(x * x + y * y + z * z)
		return np.stack([x * oolen * xsign, y * oolen * ysign, z * oolen * zsign], 1)
	v = vb.read(attr)[:, :3]
	return v * 2 - 1 if attr.format in (28, 35, 11) else v


def _draw_call(mesh: MeshData, dc: dict, table, nbones: int):
	vrefs = dc.get("m_vertexBuffers") or []
	ib_ref = dc.get("m_indexBuffer") or {}
	if not vrefs or not mesh.ibs:
		return None
	streams = [mesh.vbs[int(v.get("m_hBuffer", 0))] for v in vrefs if int(v.get("m_hBuffer", 0)) < len(mesh.vbs)]
	if not streams:
		return None
	ib = mesh.ibs[int(ib_ref.get("m_hBuffer", 0))]
	base = int(dc.get("m_nBaseVertex", 0))
	start = int(dc.get("m_nStartIndex", 0))
	count = int(dc.get("m_nIndexCount", 0))
	idx = np.frombuffer(ib.data, "<u2" if ib.stride == 2 else "<u4")[start:start + count].astype(np.int64) + base
	if not len(idx):
		return None

	def find(name: str, index: int = 0):
		for s in streams:
			a = s.attribute(name, index)
			if a is not None:
				return s, a
		return None, None

	vb, pa = find("POSITION")
	if pa is None:
		return None
	positions = vb.read(pa)[:, :3]
	nvb, na = find("NORMAL")
	normals = _normals(nvb, na) if na is not None else np.tile([0.0, 0.0, 1.0], (len(positions), 1))
	tvb, ta = find("TEXCOORD")
	uvs = tvb.read(ta)[:, :2] if ta is not None else np.zeros((len(positions), 2))
	bvb, ba = find("BLENDINDICES")
	wvb, wa = find("BLENDWEIGHT")
	if wa is None:
		wvb, wa = find("BLENDWEIGHTS")
	n = len(positions)
	if ba is not None:
		if ba.format == 4:	# R32G32B32A32_SINT holds eight 16-bit joints
			raw = bvb.read_raw(ba).view("<u2").reshape(n, 8).astype(np.int64)
		elif ba.format == 12:	# R16G16B16A16_UINT holds eight 8-bit joints
			raw = bvb.read_raw(ba).view("u1").reshape(n, 8).astype(np.int64)
		else:
			raw = bvb.read_raw(ba).astype(np.int64)
		if ba.format in (38,):	# R16G16_SINT: two joints
			raw = np.concatenate([raw, raw[:, 1:2], raw[:, 1:2]], 1)
		if wa is not None:
			if wa.format == 11:	# R16G16B16A16_UNORM holds eight 8-bit weights
				w = wvb.read_raw(wa).view("u1").reshape(n, 8).astype(np.float64) / 255.0
			else:
				w = wvb.read(wa)
			if w.shape[1] < raw.shape[1]:
				w = np.concatenate([w, np.zeros((n, raw.shape[1] - w.shape[1]))], 1)
			w = w[:, :raw.shape[1]]
		else:
			w = np.zeros(raw.shape)
			w[:, 0] = 1.0
		if table is not None and len(table):
			raw = np.where(raw < len(table), table[np.clip(raw, 0, len(table) - 1)], 0)
		raw = np.clip(raw, 0, nbones - 1)
	else:
		raw = np.zeros((n, 2), np.int64)
		w = np.zeros((n, 2))
		w[:, 0] = 1.0
	used = np.unique(idx)
	used = used[used < n]
	remap = np.full(int(max(used.max(), idx.max())) + 1, -1, np.int64)
	remap[used] = np.arange(len(used))
	tris = remap[idx[: len(idx) // 3 * 3].reshape(-1, 3)]
	tris = tris[(tris >= 0).all(1)]
	order = np.argsort(-w[used], 1)[:, :2]
	rows = np.arange(len(used))[:, None]
	top = raw[used][rows, order]
	tw = w[used][rows, order]
	share = np.where(tw.sum(1) > 0, tw[:, 0] / np.maximum(tw.sum(1), 1e-9), 1.0)
	return positions[used], normals[used], uvs[used], tris, top, share


def _sequences(res, path, d, sbones, roots, scale, opts, warnings) -> list:
	try:
		from . import vanim
	except ImportError:
		return []
	try:
		return vanim.model_sequences(res, path, d, sbones, roots, scale, opts, warnings)
	except (ValueError, KeyError, IndexError, struct.error) as exc:
		warnings.append(f"animations not converted: {exc}")
		return []


def vmdl_info(path: Path) -> dict:
	res = vres.load(path)
	out = dict(res.info())
	if res.kind == "mesh":
		d = res.keyvalues() or {}
		out["Draw calls"] = sum(len(so.get("m_drawCalls") or []) for so in d.get("m_sceneObjects") or [])
		return out
	d = res.keyvalues() or {}
	out["Name"] = d.get("m_name", "")
	out["Bones"] = len((d.get("m_modelSkeleton") or {}).get("m_boneName") or [])
	ctrl = res.keyvalues("CTRL") if res.block("CTRL") else {}
	out["Meshes"] = len((ctrl or {}).get("embedded_meshes") or []) + len([m for m in d.get("m_refMeshes") or [] if m])
	return out
