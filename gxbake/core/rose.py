"""ROSE Online native models → GoldSrc studio MDL.

ROSE keeps its 3D data in its own little-endian binary formats:

	.ZMS	mesh (positions, normals, skin weights, UVs, triangles), versions 5-8
	.ZMD	skeleton (bones + dummy attachment points), versions 2-3
	.ZMO	motion (per-bone position / rotation tracks), version 2
	.ZSC	object list: each object is parts = mesh + material + transform
	.CHR	character list (LIST_NPC.CHR): skeleton + ZSC objects + motions per NPC
	.DDS	textures (DXT1/3/5, read through Pillow)

Assimp reads none of these, so they are parsed here and baked into the same
ImportedModel that models.py builds from Assimp scenes. From there it is the
usual path: build_studio_mdl, then the lean mesh and skin companion.

Units and axes: ZMS v7/8 positions are metres; ZMD, ZMO and ZSC positions and
ZMS v5/6 positions are centimetres. ROSE is Z-up like GoldSrc, so only a scale
(ModelOptions.rose_scale, GoldSrc units per metre) and a turn about Z
(ModelOptions.rose_yaw) are applied.

Paths inside ZSC/CHR files are VFS paths like "3DDATA\\NPC\\...\\X.ZMS". They are
resolved case-insensitively against the folder that holds 3DDATA, found by
walking up from the file being converted.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as _m
from . import skeleton as _sk

# Files that become assets. .zmd/.zmo/.dds are only read on behalf of these.
ROSE_EXTENSIONS = {".zms", ".zsc", ".chr", ".rosemdl"}
SUPPORT_EXTENSIONS = {".zmd", ".zmo"}
LIST_EXTENSIONS = {".zsc", ".chr"}	# one file, many models (entries)

# ZMS vertex format bits.
_POSITION, _NORMAL, _COLOR, _WEIGHT, _INDEX, _TANGENT = 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6
_UV = (1 << 7, 1 << 8, 1 << 9, 1 << 10)

# NPC motion slots in LIST_NPC.CHR, and whether the action loops.
NPC_MOTIONS = {
	0: ("stop", True), 1: ("walk", True), 2: ("attack", False), 3: ("hit", False), 4: ("die", False),
	5: ("run", True), 6: ("cast1", False), 7: ("skill1", False), 8: ("cast2", False), 9: ("skill2", False),
	10: ("etc", False),
}
_ONESHOT_WORDS = ("attack", "atk", "hit", "die", "dead", "death", "cast", "skill", "jump", "pickup", "sit_down")
_IDLE_WORDS = ("stop", "idle", "stand", "wait", "warning")


# -- binary reading -------------------------------------------------------------


class _Reader:
	def __init__(self, data: bytes, name: str = ""):
		self.data, self.pos, self.name = data, 0, name

	def need(self, n: int) -> None:
		if self.pos + n > len(self.data):
			raise ValueError(f"{self.name}: truncated at byte {self.pos}")

	def unpack(self, fmt: str):
		size = struct.calcsize("<" + fmt)
		self.need(size)
		out = struct.unpack_from("<" + fmt, self.data, self.pos)
		self.pos += size
		return out

	def u8(self) -> int:
		return self.unpack("B")[0]

	def u16(self) -> int:
		return self.unpack("H")[0]

	def u32(self) -> int:
		return self.unpack("I")[0]

	def f32(self) -> float:
		return self.unpack("f")[0]

	def array(self, dtype: str, count: int, width: int = 1) -> "np.ndarray":
		dt = np.dtype("<" + dtype)
		n = count * width
		self.need(n * dt.itemsize)
		out = np.frombuffer(self.data, dt, n, self.pos).copy()
		self.pos += n * dt.itemsize
		return out.reshape(count, width) if width > 1 else out

	def cstr(self) -> str:
		end = self.data.find(b"\0", self.pos)
		if end < 0:
			raise ValueError(f"{self.name}: unterminated string at byte {self.pos}")
		raw = self.data[self.pos:end]
		self.pos = end + 1
		return _text(raw)

	def fixed(self, n: int) -> str:
		self.need(n)
		raw = self.data[self.pos:self.pos + n]
		self.pos += n
		return _text(raw.split(b"\0", 1)[0])

	def skip(self, n: int) -> None:
		self.need(n)
		self.pos += n


def _text(raw: bytes) -> str:
	try:
		return raw.decode("ascii")
	except UnicodeDecodeError:
		return raw.decode("euc_kr", errors="replace")	# Korean names in the original client


def _quat_wxyz(r: _Reader) -> "np.ndarray":
	w, x, y, z = r.unpack("4f")
	return np.array([x, y, z, w], np.float64)	# stored (x, y, z, w) like skeleton.quat_matrix


# -- ZMS ---------------------------------------------------------------------------


@dataclass
class ZmsMesh:
	version: int
	positions: "np.ndarray"	# (n, 3) metres
	normals: "np.ndarray | None"
	uvs: "np.ndarray"	# (n, 2) first UV set, top-left origin
	tris: "np.ndarray"	# (m, 3)
	joints: "np.ndarray | None" = None	# (n, 4) ZMD bone ids
	weights: "np.ndarray | None" = None	# (n, 4)
	colors: "np.ndarray | None" = None
	uv_sets: int = 0

	@property
	def skinned(self) -> bool:
		return self.joints is not None


def read_zms(data: bytes, name: str = "zms") -> ZmsMesh:
	r = _Reader(data, name)
	magic = r.cstr()
	if magic not in ("ZMS0005", "ZMS0006", "ZMS0007", "ZMS0008"):
		raise ValueError(f"{name}: not a ZMS mesh (magic {magic!r})")
	version = int(magic[3:])
	fmt = r.u32()
	r.skip(24)	# bounding box
	old = version < 7

	if old:	# v5/6: counts are u32 and every vertex attribute carries a u32 vertex id
		nb = r.u32()
		bone_map = r.array("I", nb, 2)[:, 1].astype(np.int64) if nb else np.zeros(0, np.int64)
		nv = r.u32()

		def attr(width: int) -> "np.ndarray":
			raw = r.array("I", nv, 1 + width)
			return np.ascontiguousarray(raw[:, 1:]).view("<f4")
	else:
		nb = r.u16()
		bone_map = r.array("H", nb).astype(np.int64) if nb else np.zeros(0, np.int64)
		nv = r.u16()

		def attr(width: int) -> "np.ndarray":
			return r.array("f", nv, width)

	pos = attr(3).astype(np.float64) if fmt & _POSITION else np.zeros((nv, 3))
	if old:
		pos /= 100.0
	nrm = attr(3).astype(np.float64) if fmt & _NORMAL else None
	col = attr(4).astype(np.float64) if fmt & _COLOR else None
	joints = weights = None
	if fmt & _WEIGHT and fmt & _INDEX:
		if old:
			raw = r.array("I", nv, 9)
			weights = np.ascontiguousarray(raw[:, 1:5]).view("<f4").astype(np.float64)
			local = raw[:, 5:9].astype(np.int64)
		else:
			weights = np.empty((nv, 4))
			local = np.empty((nv, 4), np.int64)
			for i in range(nv):	# interleaved 4 floats + 4 shorts
				v = r.unpack("4f4H")
				weights[i], local[i] = v[:4], v[4:]
		if len(bone_map) == 0 or (local >= len(bone_map)).any():
			bad = local[weights > 0]
			if len(bone_map) == 0 or (bad >= len(bone_map)).any():
				raise ValueError(f"{name}: skin bone index out of range")
		joints = bone_map[np.clip(local, 0, max(0, len(bone_map) - 1))]
	if fmt & _TANGENT:
		attr(3)
	uv_sets = [attr(2).astype(np.float32) for bit in _UV if fmt & bit]
	uvs = uv_sets[0] if uv_sets else np.zeros((nv, 2), np.float32)

	if old:
		nt = r.u32()
		tris = r.array("I", nt, 4)[:, 1:].astype(np.int64)
	else:
		nt = r.u16()
		tris = r.array("H", nt, 3).astype(np.int64)
	if len(tris) and tris.max() >= nv:
		raise ValueError(f"{name}: triangle index out of range")
	return ZmsMesh(version, pos, nrm, uvs, tris.reshape(-1, 3), joints, weights, col, len(uv_sets))


# -- ZMD ---------------------------------------------------------------------------


@dataclass
class ZmdBone:
	name: str
	parent: int	# -1 for the root
	position: "np.ndarray"	# metres, relative to the parent
	rotation: "np.ndarray"	# (x, y, z, w), relative to the parent


@dataclass
class Zmd:
	bones: list[ZmdBone]
	dummies: list[ZmdBone]	# parent indexes into bones

	@property
	def all(self) -> list[ZmdBone]:
		return self.bones + self.dummies


def read_zmd(data: bytes, name: str = "zmd") -> Zmd:
	r = _Reader(data, name)
	magic = r.fixed(7)
	if magic not in ("ZMD0002", "ZMD0003"):
		raise ValueError(f"{name}: not a ZMD skeleton (magic {magic!r})")
	version = int(magic[3:])
	bones = []
	for i in range(r.u32()):
		parent = r.u32()
		bname = r.cstr()
		p = np.array(r.unpack("3f"), np.float64) / 100.0
		q = _quat_wxyz(r)
		bones.append(ZmdBone(bname or f"bone{i}", -1 if parent == i or i == 0 else int(parent), p, q))
	dummies = []
	for i in range(r.u32()):
		dname = r.cstr()
		parent = r.u32()
		p = np.array(r.unpack("3f"), np.float64) / 100.0
		q = _quat_wxyz(r) if version >= 3 else np.array([0.0, 0.0, 0.0, 1.0])
		dummies.append(ZmdBone(dname or f"dummy{i}", int(parent), p, q))
	if not bones:
		raise ValueError(f"{name}: skeleton has no bones")
	for b in bones + dummies:
		if b.parent >= len(bones):
			raise ValueError(f"{name}: bone {b.name} has parent {b.parent} out of range")
	return Zmd(bones, dummies)


# -- ZMO ---------------------------------------------------------------------------


@dataclass
class Zmo:
	fps: int
	num_frames: int
	positions: dict[int, "np.ndarray"] = field(default_factory=dict)	# bone → (frames, 3) metres
	rotations: dict[int, "np.ndarray"] = field(default_factory=dict)	# bone → (frames, 4) xyzw
	other_channels: int = 0	# UV/alpha/morph tracks we cannot bake

	@property
	def duration(self) -> float:
		return max(0, self.num_frames - 1) / max(1, self.fps)


_ZMO_WIDTH = {1: 0, 2: 3, 4: 4, 8: 3, 16: 1, 32: 2, 64: 2, 128: 2, 256: 2, 512: 1, 1024: 1}


def read_zmo(data: bytes, name: str = "zmo") -> Zmo:
	r = _Reader(data, name)
	magic = r.cstr()
	if magic != "ZMO0002":
		raise ValueError(f"{name}: not a ZMO motion (magic {magic!r})")
	fps, nframes, nch = r.u32(), r.u32(), r.u32()
	chans = []
	for _ in range(nch):
		kind, bone = r.u32(), r.u32()
		if kind not in _ZMO_WIDTH:
			raise ValueError(f"{name}: unknown channel type {kind}")
		chans.append((kind, bone))
	width = sum(_ZMO_WIDTH[k] for k, _b in chans)
	block = r.array("f", nframes, width).astype(np.float64) if width and nframes else np.zeros((nframes, 0))
	block = block.reshape(nframes, width)
	out = Zmo(max(1, fps), nframes)
	col = 0
	for kind, bone in chans:
		w = _ZMO_WIDTH[kind]
		vals = block[:, col:col + w]
		col += w
		if kind == 2:
			out.positions[bone] = vals / 100.0
		elif kind == 4:
			out.rotations[bone] = vals[:, [1, 2, 3, 0]]	# wxyz → xyzw
		elif kind != 1:
			out.other_channels += 1
	return out


# -- ZSC ---------------------------------------------------------------------------


@dataclass
class ZscMaterial:
	path: str
	is_skin: bool = False
	alpha_enabled: bool = False
	two_sided: bool = False
	alpha_test: bool = False
	alpha_ref: int = 128
	blend_add: bool = False
	alpha: float = 1.0


@dataclass
class ZscPart:
	mesh: int
	material: int
	position: "np.ndarray" = field(default_factory=lambda: np.zeros(3))	# metres
	rotation: "np.ndarray" = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
	scale: "np.ndarray" = field(default_factory=lambda: np.ones(3))
	bone: int | None = None
	dummy: int | None = None
	parent: int | None = None	# index of another part of the same object
	motion: str = ""	# per-part animation (windmills, flags...)

	def local_matrix(self) -> "np.ndarray":
		m = np.eye(4)
		q = self.rotation if np.linalg.norm(self.rotation) > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])
		m[:3, :3] = _sk.quat_matrix(q) * self.scale[None, :]
		m[:3, 3] = self.position
		return m


@dataclass
class ZscObject:
	parts: list[ZscPart]
	effects: int = 0


@dataclass
class Zsc:
	meshes: list[str]
	materials: list[ZscMaterial]
	effects: list[str]
	objects: list[ZscObject]


def _read_props(r: _Reader, part: ZscPart | None) -> None:
	"""The (id, size, payload) property list after a part or effect, ended by id 0."""
	while True:
		pid = r.u8()
		if pid == 0:
			return
		size = r.u8()
		start = r.pos
		if part is not None:
			if pid == 1:
				part.position = np.array(r.unpack("3f"), np.float64) / 100.0
			elif pid == 2:
				part.rotation = _quat_wxyz(r)
			elif pid == 3:
				part.scale = np.array(r.unpack("3f"), np.float64)
			elif pid == 5:
				part.bone = r.u16()
			elif pid == 6:
				part.dummy = r.u16()
			elif pid == 7:
				p = r.u16()
				part.parent = p - 1 if p else None
			elif pid == 30:
				part.motion = r.fixed(size)
		# Known payload sizes; the size byte is not reliable for every id.
		known = {1: 12, 2: 16, 3: 12, 4: 16, 5: 2, 6: 2, 7: 2, 29: 2, 31: 2, 32: 2}
		r.pos = start + (size if pid == 30 else known.get(pid, size))


def read_zsc(data: bytes, name: str = "zsc") -> Zsc:
	r = _Reader(data, name)
	meshes = [r.cstr() for _ in range(r.u16())]
	materials = []
	for _ in range(r.u16()):
		path = r.cstr()
		is_skin, alpha_on, two_sided, atest, aref, _zt, _zw, blend, _spec = r.unpack("9H")
		alpha = r.f32()
		r.skip(2 + 12)	# glow type + colour
		materials.append(ZscMaterial(path, bool(is_skin), bool(alpha_on), bool(two_sided), bool(atest),
			min(255, aref), blend == 1, alpha))
	effects = [r.cstr() for _ in range(r.u16())]
	objects = []
	for _ in range(r.u16()):
		r.skip(12)	# bounding sphere
		nparts = r.u16()
		if nparts == 0:
			objects.append(ZscObject([]))
			continue
		parts = []
		for _p in range(nparts):
			part = ZscPart(r.u16(), r.u16())
			_read_props(r, part)
			parts.append(part)
		neff = r.u16()
		for _e in range(neff):
			r.skip(4)	# effect id + type
			_read_props(r, None)
		r.skip(24)	# bounding box
		objects.append(ZscObject(parts, neff))
	return Zsc(meshes, materials, effects, objects)


# -- CHR ---------------------------------------------------------------------------


@dataclass
class ChrCharacter:
	index: int
	name: str
	skeleton: int
	objects: list[int]
	motions: list[tuple[int, int]]	# (motion slot, motion file index)


@dataclass
class Chr:
	skeletons: list[str]
	motions: list[str]
	effects: list[str]
	characters: list[ChrCharacter]	# enabled entries only


def read_chr(data: bytes, name: str = "chr") -> Chr:
	r = _Reader(data, name)
	skeletons = [r.cstr() for _ in range(r.u16())]
	motions = [r.cstr() for _ in range(r.u16())]
	effects = [r.cstr() for _ in range(r.u16())]
	chars = []
	for i in range(r.u16()):
		if r.u8() == 0:
			continue
		skel = r.u16()
		cname = r.cstr()
		objs = [r.u16() for _ in range(r.u16())]
		mots = [(r.u16(), r.u16()) for _ in range(r.u16())]
		for _ in range(r.u16()):
			r.skip(4)
		chars.append(ChrCharacter(i, cname, skel, objs, mots))
	return Chr(skeletons, motions, effects, chars)


# -- path resolution ---------------------------------------------------------------


class RoseFS:
	"""Case-insensitive lookup of VFS paths ("3DDATA\\NPC\\X.ZMS") on disk."""

	def __init__(self, near: Path):
		near = Path(near).resolve()
		self.near = near if near.is_dir() else near.parent
		self.root = self._find_root(self.near)
		self._dirs: dict[Path, dict[str, Path]] = {}

	@staticmethod
	def _find_root(d: Path) -> Path | None:
		for p in [d, *d.parents]:
			if p.name.lower() == "3ddata":
				return p.parent
			try:
				if any(c.name.lower() == "3ddata" and c.is_dir() for c in p.iterdir()):
					return p
			except OSError:
				continue
		return None

	def _listing(self, d: Path) -> dict[str, Path]:
		if d not in self._dirs:
			try:
				self._dirs[d] = {c.name.lower(): c for c in d.iterdir()}
			except OSError:
				self._dirs[d] = {}
		return self._dirs[d]

	def _walk(self, base: Path, parts: list[str]) -> Path | None:
		cur = base
		for part in parts:
			nxt = self._listing(cur).get(part.lower())
			if nxt is None:
				return None
			cur = nxt
		return cur if cur.is_file() else None

	def resolve(self, ref: str, relative_to: Path | None = None) -> Path | None:
		parts = [p for p in ref.replace("\\", "/").split("/") if p and p != "."]
		if not parts:
			return None
		if Path(ref).is_absolute():
			return Path(ref) if Path(ref).is_file() else None
		bases = [b for b in (self.root, relative_to, self.near) if b is not None]
		for base in bases:
			hit = self._walk(base, parts)
			if hit is not None:
				return hit
		for base in bases[1:]:	# loose copies: same file name next to the referencing file
			hit = self._walk(base, parts[-1:])
			if hit is not None:
				return hit
		return None


def _image_beside(path: Path) -> Path | None:
	from .images import IMAGE_EXTENSIONS

	exts = [".dds"] + sorted(IMAGE_EXTENSIONS - {".dds"})
	try:
		by = {c.name.lower(): c for c in path.parent.iterdir()}
	except OSError:
		return None
	for e in exts:
		hit = by.get(path.stem.lower() + e)
		if hit is not None:
			return hit
	return None


# -- model assembly ----------------------------------------------------------------


@dataclass
class _Part:
	name: str
	mesh: ZmsMesh
	texture: Path | None
	material: ZscMaterial
	transform: "np.ndarray"	# object space (metres), applied to unskinned meshes
	bone: int | None = None	# studio bone the transform is relative to (rigid attachment)


@dataclass
class _Motion:
	name: str
	zmo: Zmo
	loop: bool


@dataclass
class RoseEntry:
	"""One model inside a list file (a ZSC object or a CHR character)."""
	index: int
	label: str


def entries(path: Path) -> list[RoseEntry]:
	"""The models a .zsc or .chr holds; [] for single-model files."""
	path = Path(path)
	ext = path.suffix.lower()
	data = path.read_bytes()
	if ext == ".zsc":
		zsc = read_zsc(data, path.name)
		out = []
		for i, obj in enumerate(zsc.objects):
			if obj.parts:
				mesh = zsc.meshes[obj.parts[0].mesh] if obj.parts[0].mesh < len(zsc.meshes) else ""
				stem = Path(mesh.replace("\\", "/")).stem.lower() or f"object{i}"
				out.append(RoseEntry(i, stem))
		return out
	if ext == ".chr":
		chr_ = read_chr(data, path.name)
		return [RoseEntry(c.index, c.name) for c in chr_.characters if c.objects]
	return []


def entry_stem(path: Path, entry: RoseEntry) -> str:
	"""Output file stem for one entry: <list>_<index>_<label>, safe for the game's file system."""
	label = "".join(ch if ch.isalnum() else "_" for ch in entry.label.lower()).strip("_")[:24]
	base = Path(path).stem.lower()
	for prefix in ("list_", "part_"):
		if base.startswith(prefix):
			base = base[len(prefix):]
	return f"{base}_{entry.index}" + (f"_{label}" if label and label.isascii() else "")


def import_rose(path: Path, opts: "_m.ModelOptions | None" = None, entry: int | None = None) -> "_m.ImportedModel":
	"""Read a ROSE model (ZMS, a ZSC object, a CHR character or a .rosemdl manifest)."""
	opts = opts or _m.ModelOptions()
	path = Path(path)
	fs = RoseFS(path)
	warnings: list[str] = []
	ext = path.suffix.lower()
	name = path.stem
	skel: Zmd | None = None
	parts: list[_Part] = []
	motions: list[_Motion] = []

	if ext == ".zms":
		mesh = read_zms(path.read_bytes(), path.name)
		tex = _image_beside(path)
		if tex is None:
			warnings.append(f"no texture named {path.stem}.dds beside the mesh")
		parts.append(_Part(path.stem, mesh, tex, ZscMaterial(str(tex or "")), np.eye(4)))
		if mesh.skinned:
			zmd_path = _skeleton_beside(path)
			if zmd_path is None:
				warnings.append("skinned mesh but no .zmd skeleton beside it; baked static")
			else:
				skel = read_zmd(zmd_path.read_bytes(), zmd_path.name)
				motions = _motions_beside(path, warnings)
	elif ext == ".zsc":
		zsc = read_zsc(path.read_bytes(), path.name)
		index = entry if entry is not None else next((i for i, o in enumerate(zsc.objects) if o.parts), 0)
		if not 0 <= index < len(zsc.objects) or not zsc.objects[index].parts:
			raise ValueError(f"{path.name}: object {index} is empty or missing")
		name = f"{path.stem}_{index}"
		parts = _object_parts(zsc, index, fs, path, warnings)
	elif ext == ".chr":
		chr_ = read_chr(path.read_bytes(), path.name)
		chars = {c.index: c for c in chr_.characters if c.objects}
		if not chars:
			raise ValueError(f"{path.name}: no characters")
		ch = chars.get(entry if entry is not None else min(chars))
		if ch is None:
			raise ValueError(f"{path.name}: character {entry} is disabled or missing")
		name = ch.name or f"{path.stem}_{ch.index}"
		zsc_path = _part_list_for(path, fs)
		if zsc_path is None:
			raise ValueError(f"{path.name}: no PART_*.ZSC beside it to take the character's meshes from")
		zsc = read_zsc(zsc_path.read_bytes(), zsc_path.name)
		for oi in ch.objects:
			if 0 <= oi < len(zsc.objects):
				parts += _object_parts(zsc, oi, fs, path, warnings)
			else:
				warnings.append(f"object {oi} is not in {zsc_path.name}")
		if not opts.static and 0 <= ch.skeleton < len(chr_.skeletons):
			zp = fs.resolve(chr_.skeletons[ch.skeleton], path.parent)
			if zp is None:
				warnings.append(f"skeleton {chr_.skeletons[ch.skeleton]} not found; baked static")
			else:
				skel = read_zmd(zp.read_bytes(), zp.name)
				for slot, mi in sorted(ch.motions):
					if not 0 <= mi < len(chr_.motions):
						continue
					mp = fs.resolve(chr_.motions[mi], path.parent)
					label, loop = NPC_MOTIONS.get(slot, (f"motion{slot}", False))
					if mp is None:
						warnings.append(f"{label} motion {chr_.motions[mi]} not found")
						continue
					try:
						motions.append(_Motion(label, read_zmo(mp.read_bytes(), mp.name), loop))
					except ValueError as exc:
						warnings.append(str(exc))
	elif ext == ".rosemdl":
		name, skel, parts, motions = _manifest(path, fs, opts, warnings)
	else:
		raise ValueError(f"{path.name}: not a ROSE model")

	if not parts:
		why = "; ".join(w for w in warnings if "not found" in w)[:300]
		raise ValueError(f"{path.name}: nothing to convert" + (f" ({why})" if why else ""))
	if opts.static:
		if skel is not None or motions:
			warnings.append("baked static: skeleton and animations ignored")
		skel, motions = None, []
	return _assemble(name, skel, parts, motions, opts, warnings)


def _skeleton_beside(path: Path) -> Path | None:
	try:
		zmds = sorted(c for c in path.parent.iterdir() if c.suffix.lower() == ".zmd")
	except OSError:
		return None
	for z in zmds:
		if z.stem.lower() == path.stem.lower():
			return z
	return zmds[0] if len(zmds) == 1 else None


def _motions_beside(path: Path, warnings: list[str]) -> list[_Motion]:
	"""<stem>@<clip>.zmo clips, else every .zmo in the mesh's folder."""
	try:
		zmos = sorted(c for c in path.parent.iterdir() if c.suffix.lower() == ".zmo")
	except OSError:
		return []
	clips = [z for z in zmos if z.stem.lower().startswith(path.stem.lower() + "@")]
	out = []
	for z in clips or zmos:
		label = z.stem.split("@", 1)[1] if "@" in z.stem else z.stem
		try:
			out.append(_Motion(label, read_zmo(z.read_bytes(), z.name), _loops(label)))
		except ValueError as exc:
			warnings.append(str(exc))
	return _idle_first(out)


def _loops(label: str) -> bool:
	low = label.lower()
	return not any(w in low for w in _ONESHOT_WORDS)


def _idle_first(motions: list[_Motion]) -> list[_Motion]:
	"""GoldSrc plays sequence 0 by default, so an idle goes first when there is one."""
	for i, m in enumerate(motions):
		if any(w in m.name.lower() for w in _IDLE_WORDS):
			return [m] + motions[:i] + motions[i + 1:]
	return motions


def _part_list_for(chr_path: Path, fs: RoseFS) -> Path | None:
	"""LIST_NPC.CHR takes its meshes from PART_NPC.ZSC in the same folder."""
	try:
		zscs = {c.name.lower(): c for c in chr_path.parent.iterdir() if c.suffix.lower() == ".zsc"}
	except OSError:
		return None
	stem = chr_path.stem.lower()
	want = "part_" + stem[5:] + ".zsc" if stem.startswith("list_") else None
	if want and want in zscs:
		return zscs[want]
	parts = [p for n, p in sorted(zscs.items()) if n.startswith("part_")]
	if parts:
		return parts[0]
	return fs.resolve("3DDATA/NPC/PART_NPC.ZSC")


def _object_parts(zsc: Zsc, index: int, fs: RoseFS, near: Path, warnings: list[str]) -> list[_Part]:
	obj = zsc.objects[index]
	world: list[np.ndarray] = []
	out = []
	for pi, part in enumerate(obj.parts):
		local = part.local_matrix()
		parent = part.parent if part.parent is not None and part.parent < pi else None
		world.append(local if parent is None else world[parent] @ local)
		if part.motion:
			warnings.append(f"part {pi} has its own animation ({Path(part.motion.replace(chr(92), '/')).name}); baked at rest")
		if not 0 <= part.mesh < len(zsc.meshes):
			warnings.append(f"object {index} part {pi}: mesh {part.mesh} out of range")
			continue
		mref = zsc.meshes[part.mesh]
		mpath = fs.resolve(mref, near.parent)
		if mpath is None:
			warnings.append(f"mesh {mref} not found")
			continue
		try:
			mesh = read_zms(mpath.read_bytes(), mpath.name)
		except ValueError as exc:
			warnings.append(str(exc))
			continue
		mat = zsc.materials[part.material] if 0 <= part.material < len(zsc.materials) else ZscMaterial("")
		tex = fs.resolve(mat.path, near.parent) if mat.path else None
		if tex is None:
			tex = _image_beside(mpath)
		if tex is None:
			warnings.append(f"texture {mat.path or mpath.stem} not found; using grey")
		out.append(_Part(mpath.stem, mesh, tex, mat, world[pi], part.bone if part.dummy is None else ("dummy", part.dummy)))
	return out


def _manifest(path: Path, fs: RoseFS, opts: "_m.ModelOptions", warnings: list[str]):
	"""A .rosemdl JSON file that assembles a model by hand (e.g. a player avatar):

	{"name": "...", "skeleton": "3DDATA/AVATAR/MALE.ZMD",
	 "parts": [{"zsc": "3DDATA/AVATAR/LIST_MBODY.ZSC", "object": 1},
	           {"mesh": "x.zms", "texture": "x.dds", "bone": 4, "two_sided": false, "alpha_test": false}],
	 "motions": [{"name": "walk", "file": "3DDATA/MOTION/AVATAR/EMPTY_WALK_M1.ZMO", "loop": true}]}

	Paths are relative to the 3DDATA root or to the manifest.
	"""
	try:
		spec = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, ValueError) as exc:
		raise ValueError(f"{path.name}: bad manifest: {exc}") from exc

	def find(ref: str) -> Path:
		hit = fs.resolve(ref, path.parent)
		if hit is None:
			raise ValueError(f"{path.name}: {ref} not found")
		return hit

	skel = None
	if spec.get("skeleton") and not opts.static:
		zp = find(spec["skeleton"])
		skel = read_zmd(zp.read_bytes(), zp.name)
	parts: list[_Part] = []
	zsc_cache: dict[Path, Zsc] = {}
	for p in spec.get("parts", []):
		if "zsc" in p:
			zp = find(p["zsc"])
			if zp not in zsc_cache:
				zsc_cache[zp] = read_zsc(zp.read_bytes(), zp.name)
			zsc = zsc_cache[zp]
			oi = int(p.get("object", 0))
			if not 0 <= oi < len(zsc.objects):
				raise ValueError(f"{path.name}: {zp.name} has no object {oi}")
			parts += _object_parts(zsc, oi, fs, path, warnings)
		elif "mesh" in p:
			mp = find(p["mesh"])
			tex = find(p["texture"]) if p.get("texture") else _image_beside(mp)
			mat = ZscMaterial(str(tex or ""), two_sided=bool(p.get("two_sided")), alpha_test=bool(p.get("alpha_test")),
				alpha_enabled=bool(p.get("alpha")))
			bone = p.get("bone")
			parts.append(_Part(mp.stem, read_zms(mp.read_bytes(), mp.name), tex, mat, np.eye(4),
				int(bone) if bone is not None else None))
	motions = []
	mspec = spec.get("motions", [])
	if isinstance(mspec, dict):
		mspec = [{"name": k, "file": v} for k, v in mspec.items()]
	for m in mspec:
		mp = find(m["file"])
		label = m.get("name") or mp.stem
		motions.append(_Motion(label, read_zmo(mp.read_bytes(), mp.name), bool(m.get("loop", _loops(label)))))
	return spec.get("name") or path.stem, skel, parts, motions


# -- baking ------------------------------------------------------------------------


def _bone_order(zmd: Zmd) -> tuple[list[ZmdBone], list[int]]:
	"""Bones then dummies, parents first. Returns (ordered bones, ROSE index → studio index)."""
	src = zmd.all
	nb = len(zmd.bones)
	order: list[int] = []
	placed: set[int] = set()
	pending = list(range(len(src)))
	while pending:
		progress = False
		for i in list(pending):
			p = src[i].parent
			if i == 0 or p < 0 or p in placed or (i >= nb and p in placed):
				order.append(i)
				placed.add(i)
				pending.remove(i)
				progress = True
		if not progress:	# a cycle: hang the rest off the root
			for i in pending:
				src[i].parent = 0
			order += pending
			break
	remap = [0] * len(src)
	for s, i in enumerate(order):
		remap[i] = s
	return [src[i] for i in order], remap


def _to_gold(opts: "_m.ModelOptions") -> tuple["np.ndarray", float]:
	yaw = math.radians(opts.rose_yaw)
	c, s = math.cos(yaw), math.sin(yaw)
	rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
	return rot, float(opts.rose_scale)


def _gold(m: "np.ndarray", rot: "np.ndarray", k: float) -> "np.ndarray":
	"""ROSE rigid 4x4 (metres) → GoldSrc rigid 4x4 (scaled, turned)."""
	out = np.eye(4)
	out[:3, :3] = rot @ m[:3, :3]
	out[:3, 3] = k * (rot @ m[:3, 3])
	return out


def _worlds(local_pos: "np.ndarray", local_rot: "np.ndarray", parents: list[int]) -> list["np.ndarray"]:
	mats = []
	for b, p in enumerate(parents):
		m = np.eye(4)
		m[:3, :3] = _sk.quat_matrix(local_rot[b])
		m[:3, 3] = local_pos[b]
		mats.append(m if p < 0 else mats[p] @ m)
	return mats


def _assemble(name: str, zmd: Zmd | None, parts: list[_Part], motions: list[_Motion],
		opts: "_m.ModelOptions", warnings: list[str]) -> "_m.ImportedModel":
	rot, k = _to_gold(opts)
	model = _m.ImportedModel(name, [], [], warnings)

	if zmd is not None:
		ordered, remap = _bone_order(zmd)
		if len(ordered) > _sk.MAXSTUDIOBONES:
			warnings.append(f"{len(ordered)} bones exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static")
			zmd = None
	if zmd is not None:
		parents = [remap[b.parent] if b.parent >= 0 else -1 for b in ordered]
		parents[0] = -1
		bind_pos = np.array([b.position for b in ordered])
		bind_rot = np.array([b.rotation for b in ordered])
		bind_world = _worlds(bind_pos, bind_rot, parents)
		gold_world = [_gold(_sk.rigid(w), rot, k) for w in bind_world]
		model.bones = []
		used: set[str] = set()
		for s, b in enumerate(ordered):
			w = gold_world[s]
			local = w if parents[s] < 0 else _sk.rigid_inverse(gold_world[parents[s]]) @ w
			model.bones.append(_m.Bone(_m._unique(b.name[:31] or f"bone{s}", used), parents[s], _sk.pose_from_matrix(local)))
		nb = len(zmd.bones)
	else:
		remap, gold_world, nb = [0], [np.eye(4)], 0

	def studio_bone(ref) -> int:
		if zmd is None or ref is None:
			return 0
		if isinstance(ref, tuple):	# ("dummy", i)
			i = nb + ref[1]
			return remap[i] if i < len(remap) else 0
		return remap[ref] if 0 <= ref < nb else 0

	mat_keys: dict[tuple, int] = {}
	for part in parts:
		mesh = part.mesh
		n = len(mesh.positions)
		if n == 0 or len(mesh.tris) == 0:
			continue
		nrm = mesh.normals if mesh.normals is not None else _vertex_normals(mesh.positions, mesh.tris)
		if zmd is not None and mesh.skinned:
			heavy = np.argmax(mesh.weights, axis=1)
			rose_bone = mesh.joints[np.arange(n), heavy]
			if ((mesh.weights > 0.001).sum(1) > 1).any():
				msg = "blended skin weights are snapped to each vertex's heaviest bone (studio skinning is rigid)"
				if msg not in warnings:
					warnings.append(msg)
			rose_bone = np.where(rose_bone < len(remap), rose_bone, 0)
			vbone = np.array([remap[int(b)] for b in rose_bone], np.int32)
			pos_m, nrm_m = mesh.positions, nrm	# skinned vertices are already in bind (model) space
		else:
			bone = studio_bone(part.bone)
			vbone = np.full(n, bone, np.int32)
			t = part.transform
			if zmd is not None and part.bone is not None:
				i = (nb + part.bone[1]) if isinstance(part.bone, tuple) else part.bone
				t = bind_world[remap[i]] @ t if 0 <= i < len(remap) else t
			pos_m = mesh.positions @ t[:3, :3].T + t[:3, 3]
			nmat = np.linalg.inv(t[:3, :3]).T
			nrm_m = nrm @ nmat.T
		world = k * (pos_m @ rot.T)
		wn = nrm_m @ rot.T
		if zmd is not None:
			bw = np.array(gold_world)[vbone]
			local = np.einsum("nji,nj->ni", bw[:, :3, :3], world - bw[:, :3, 3])
			ln = np.einsum("nji,nj->ni", bw[:, :3, :3], wn)
		else:
			local, ln = world, wn
		lens = np.linalg.norm(ln, axis=1, keepdims=True)
		ln = np.where(lens > 1e-8, ln / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		tris = mesh.tris.astype(np.uint32)
		uvs = mesh.uvs.astype(np.float32)
		if part.material.two_sided:	# studio models are single-sided: add the back faces
			local = np.concatenate([local, local])
			ln = np.concatenate([ln, -ln])
			world = np.concatenate([world, world])
			uvs = np.concatenate([uvs, uvs])
			vbone = np.concatenate([vbone, vbone])
			tris = np.concatenate([tris, tris[:, ::-1] + n])
		key = (str(part.texture or ""), part.material.alpha_test, part.material.alpha_enabled, part.material.alpha_ref)
		if key not in mat_keys:
			mat_keys[key] = len(model.materials)
			model.materials.append(_material(part, opts, warnings))
		model.meshes.append(_m.Submesh(mat_keys[key], local.astype(np.float32), ln.astype(np.float32), uvs,
			tris, vbone, world.astype(np.float32)))
	if not model.meshes:
		raise ValueError(f"{name}: no triangles")

	if zmd is not None:
		used_seq: set[str] = set()
		for mo in motions:
			frames = _sample_motion(mo.zmo, ordered, remap, parents, bind_pos, bind_rot, rot, k, opts.anim_fps)
			if frames is None:
				warnings.append(f"motion {mo.name} does not drive this skeleton; skipped")
				continue
			if mo.zmo.other_channels:
				warnings.append(f"motion {mo.name}: {mo.zmo.other_channels} UV/alpha/morph tracks ignored")
			model.sequences.append(_m.Sequence(_m._unique(_m._seq_name(mo.name), used_seq), float(opts.anim_fps),
				frames, loop=mo.loop))
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _vertex_normals(pos: "np.ndarray", tris: "np.ndarray") -> "np.ndarray":
	fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
	out = np.zeros_like(pos)
	for c in range(3):
		np.add.at(out, tris[:, c], fn)
	return out


def _material(part: _Part, opts: "_m.ModelOptions", warnings: list[str]) -> "_m.Material":
	rgba = _m._open_rgba(part.texture) if part.texture else None
	if part.texture and rgba is None:
		warnings.append(f"could not read texture {part.texture.name}; using grey")
	if rgba is None:
		rgba = np.full((8, 8, 4), (200, 200, 200, 255), np.uint8)
	rgba = _m._fit_skin(rgba, opts.skin_max_edge).copy()
	mat = part.material
	cut = mat.alpha_ref if mat.alpha_test and mat.alpha_ref else opts.alpha_threshold
	masked = (mat.alpha_test or mat.alpha_enabled) and bool((rgba[..., 3] < cut).any())
	if masked:	# studio masks at the one threshold; bake this material's cut into the texels
		rgba[..., 3] = np.where(rgba[..., 3] < cut, 0, 255)
	else:	# DXT alpha is often a specular/glow map on opaque materials
		rgba[..., 3] = 255
	label = Path(str(part.texture)).stem if part.texture else part.name
	return _m.Material(label[:63] or part.name, rgba, masked, str(part.texture or ""))


def _sample_motion(zmo: Zmo, ordered: list[ZmdBone], remap: list[int], parents: list[int],
		bind_pos: "np.ndarray", bind_rot: "np.ndarray", rot: "np.ndarray", k: float, fps: int) -> "np.ndarray | None":
	"""Bake a ZMO into (frames, bones, 6) studio poses at the requested rate."""
	nb = len(ordered)
	pos_tracks = {remap[b]: v for b, v in zmo.positions.items() if b < len(remap)}
	rot_tracks = {remap[b]: v for b, v in zmo.rotations.items() if b < len(remap)}
	if not pos_tracks and not rot_tracks or zmo.num_frames == 0:
		return None
	nframes = max(1, int(round(zmo.duration * fps)) + 1)
	frames = np.zeros((nframes, nb, 6))
	last = zmo.num_frames - 1
	for f in range(nframes):
		src = min(last, f / fps * zmo.fps)
		i0 = int(math.floor(src))
		i1 = min(last, i0 + 1)
		t = src - i0
		lp = bind_pos.copy()
		lr = bind_rot.copy()
		for b, v in pos_tracks.items():
			lp[b] = v[i0] + (v[i1] - v[i0]) * t
		for b, v in rot_tracks.items():
			lr[b] = _sk._slerp(_unit(v[i0]), _unit(v[i1]), t)
		worlds = [_gold(_sk.rigid(w), rot, k) for w in _worlds(lp, lr, parents)]
		for b in range(nb):
			rel = worlds[b] if parents[b] < 0 else _sk.rigid_inverse(worlds[parents[b]]) @ worlds[b]
			frames[f, b] = _sk.pose_from_matrix(rel)
	return frames


def _unit(q: "np.ndarray") -> "np.ndarray":
	n = np.linalg.norm(q)
	return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
