"""Final Fantasy VII (PC) field and battle models → GoldSrc studio MDL.

The PC release keeps its 3D data in LGP archives (char.lgp holds the field
models; `gxbake extract char.lgp -o char` unpacks one). A field model is:

	<name>.hrc	text skeleton: bone name, parent name, length, and the RSD
			resources drawn on the bone
	<res>.rsd	text resource: PLY=<p file>, TEX[n]=<texture>.TIM (read as .TEX)
	<p>.p		binary polygon model: vertices, normals, UVs, vertex colours,
			triangles in groups, each group textured or not
	<tex>.tex	texture: palettised or direct colour, with an optional colour key
	<anim>.a	animation: per frame a root rotation + translation and one
			rotation per bone (degrees, applied Y, then X, then Z)

Which animations belong to which skeleton is not recorded in char.lgp (the
field scripts pick them), so a .ff7mdl manifest names them:

	{"skeleton": "aaaa.hrc", "animations": ["acfe.a", "bvjf.a"]}

A .hrc on its own converts in the pose of the first animation given, or with
every bone straight if there is none. A single .p converts as a static model.

Layout references (no code copied): the Qhimm wiki (FF7/P, LGP_format,
TEX_format, PSX/HRC, PSX/RSD) and Kimera's reader for the .a frames and how
bones chain: a bone draws its part, then moves -length along its own Z for its
children.

Battle models (battle.lgp: <base>aa skeleton, parts, textures, <base>da pack) are
read by import_battle; see the notes above it.

Axes: FF7 models are y-down like the PlayStation. They are taken to face -Z, so
(x, y, z) → (-z, x, -y) · ff7_scale, then a turn of ff7_yaw degrees about Z.
Like FFXI, the facing and unit size are not yet checked on real data.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import models as _m
from . import skeleton as _sk

FF7_EXTENSIONS = {".hrc", ".p", ".ff7mdl"}
DEFAULT_SCALE = 2.5	# GoldSrc units per FF7 field unit: a ~28-unit-tall character becomes ~70
FPS = 30.0	# field animations play at 30 frames a second
BATTLE_FPS = 15.0	# battle animations: one frame per battle tick (taken as 15 a second)
BATTLE_HEIGHT = 72.0	# auto battle scale: the rest pose stands this tall


def is_ff7(path: Path) -> bool:
	ext = Path(path).suffix.lower()
	if not ext:
		return is_battle_skeleton(path)
	if ext not in FF7_EXTENSIONS:
		return False
	if ext == ".p":
		try:
			with Path(path).open("rb") as f:
				head = f.read(8)
		except OSError:
			return False
		return len(head) == 8 and struct.unpack("<II", head) == (1, 1)
	return True


def _find(folder: Path, name: str) -> Path | None:
	"""Case-insensitive file lookup (LGP names are upper case on disc, lower case once extracted)."""
	p = folder / name
	if p.exists():
		return p
	try:
		low = name.lower()
		for c in folder.iterdir():
			if c.name.lower() == low:
				return c
	except OSError:
		pass
	return None


# -- P models -----------------------------------------------------------------------


@dataclass
class Group:
	poly_start: int
	poly_count: int
	vert_start: int
	tex_start: int
	textured: bool
	texture: int


@dataclass
class PModel:
	vertices: "np.ndarray"	# (n, 3)
	normals: "np.ndarray"	# (k, 3), may be empty
	texcoords: "np.ndarray"	# (t, 2)
	vcolors: "np.ndarray"	# (n, 4) RGBA
	pcolors: "np.ndarray"	# (p, 4) RGBA
	polys: "np.ndarray"	# (p, 3) group-relative vertex indices
	pnormals: "np.ndarray"	# (p, 3) normal indices
	groups: list[Group]


def read_p(data: bytes, name: str = "p") -> PModel:
	if len(data) < 128:
		raise ValueError(f"{name}: header truncated")
	h = struct.unpack_from("<16i", data, 0)
	if h[0] != 1 or h[1] != 1:
		raise ValueError(f"{name}: not a P model (version {h[0]})")
	nv, nn, nu1, nt, _nvc, ne, npoly, nu2, nu3, nh, ng = h[3], h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11], h[12], h[13]
	if min(nv, nn, nu1, nt, ne, npoly, nu2, nu3, nh, ng) < 0:
		raise ValueError(f"{name}: negative counts")
	pos = 128

	def take(size: int) -> int:
		nonlocal pos
		start = pos
		pos += size
		if pos > len(data):
			raise ValueError(f"{name}: truncated")
		return start

	verts = np.frombuffer(data, "<f4", nv * 3, take(12 * nv)).reshape(nv, 3).astype(np.float64)
	norms = np.frombuffer(data, "<f4", nn * 3, take(12 * nn)).reshape(nn, 3).astype(np.float64)
	take(12 * nu1)
	tex = np.frombuffer(data, "<f4", nt * 2, take(8 * nt)).reshape(nt, 2).astype(np.float64)
	# vertex colours: one per vertex (the header's colour count is not reliable)
	vc = np.frombuffer(data, np.uint8, nv * 4, take(4 * nv)).reshape(nv, 4)[:, [2, 1, 0, 3]]
	pc = np.frombuffer(data, np.uint8, npoly * 4, take(4 * npoly)).reshape(npoly, 4)[:, [2, 1, 0, 3]]
	take(4 * ne)
	raw = np.frombuffer(data, "<u2", npoly * 12, take(24 * npoly)).reshape(npoly, 12).astype(np.int64)
	take(24 * nu2)
	take(3 * nu3)
	take(100 * nh)
	groups = []
	at = take(56 * ng)
	for g in range(ng):
		v = struct.unpack_from("<14I", data, at + 56 * g)
		groups.append(Group(v[1], v[2], v[3], v[11], v[12] == 1, v[13]))
	return PModel(verts, norms, tex, vc, pc, raw[:, 1:4], raw[:, 4:7], groups)


# -- textures -----------------------------------------------------------------------


def read_tex(data: bytes, name: str = "tex") -> "np.ndarray":
	"""A .TEX as (h, w, 4) RGBA. The colour key makes pixel value 0 transparent."""
	if len(data) < 0xEC:
		raise ValueError(f"{name}: header truncated")
	u = lambda off: struct.unpack_from("<I", data, off)[0]	# noqa: E731
	if u(0) != 1:
		raise ValueError(f"{name}: not a TEX file")
	color_key = u(0x08)
	w, h = u(0x3C), u(0x40)
	paletted = u(0x4C) != 0
	pal_size = u(0x58)
	pal_colors = u(0x34) or u(0x5C) or pal_size
	bpp = u(0x68)
	if not (0 < w <= 4096 and 0 < h <= 4096) or bpp not in (1, 2, 3, 4):
		raise ValueError(f"{name}: bad size {w}x{h} or {bpp} bytes per pixel")
	at = 0xEC
	pal = None
	if paletted:
		pal = np.frombuffer(data, np.uint8, pal_size * 4, at).reshape(pal_size, 4)
		at += pal_size * 4
	need = w * h * bpp
	if at + need > len(data):
		raise ValueError(f"{name}: pixel data truncated")
	px = np.frombuffer(data, np.uint8, need, at).reshape(h, w, bpp)
	if bpp == 1:
		vals = px[..., 0].astype(np.int64)
	else:
		vals = np.zeros((h, w), np.int64)
		for b in range(bpp):
			vals |= px[..., b].astype(np.int64) << (8 * b)
	if pal is not None:
		first = pal[:max(1, min(pal_colors, len(pal)))]	# the first palette
		idx = np.minimum(vals, len(first) - 1)
		rgba = np.stack([first[:, 2], first[:, 1], first[:, 0], first[:, 3]], -1)[idx].astype(np.uint8)
		if not (first[:, 3] > 0).any():	# palettes with no alpha: opaque
			rgba[..., 3] = 255
	else:
		chans = []
		for c in range(4):
			bits, mask, shift = u(0x6C + 4 * c), u(0x7C + 4 * c), u(0x8C + 4 * c)
			if bits == 0 or mask == 0:
				chans.append(np.full((h, w), 255 if c == 3 else 0, np.uint8))
				continue
			v = (vals & mask) >> shift
			maxv = (1 << bits) - 1
			chans.append(np.round(v * 255.0 / maxv).astype(np.uint8))
		rgba = np.stack(chans, -1)
	if color_key:
		rgba = rgba.copy()
		rgba[vals == 0, 3] = 0
	return np.ascontiguousarray(rgba)


# -- skeleton and resources -----------------------------------------------------------


@dataclass
class HrcBone:
	name: str
	parent: str
	length: float
	resources: list[str]


def _lines(text: str) -> list[str]:
	out = []
	for line in text.splitlines():
		line = line.strip()
		if line and not line.startswith("#"):
			out.append(line)
	return out


def read_hrc(text: str, name: str = "hrc") -> list[HrcBone]:
	lines = [x for x in _lines(text) if not x.startswith(":")]
	if len(lines) % 4:
		raise ValueError(f"{name}: {len(lines)} lines is not a whole number of bones")
	bones = []
	for i in range(0, len(lines), 4):
		bname, parent, length, res = lines[i:i + 4]
		parts = res.split()
		try:
			n = int(parts[0])
			bones.append(HrcBone(bname, parent, float(length), parts[1:1 + n]))
		except (ValueError, IndexError) as exc:
			raise ValueError(f"{name}: bone {bname!r} is malformed") from exc
	return bones


@dataclass
class Rsd:
	ply: str
	textures: list[str]


def read_rsd(text: str, name: str = "rsd") -> Rsd:
	ply, textures = "", {}
	for line in _lines(text):
		key, _, val = line.partition("=")
		key = key.strip().upper()
		if key == "PLY":
			ply = val.strip()
		elif key.startswith("TEX["):
			try:
				textures[int(key[4:key.index("]")])] = val.strip()
			except ValueError:
				pass
	if not ply:
		raise ValueError(f"{name}: no PLY= line")
	return Rsd(ply, [textures[i] for i in sorted(textures)])


# -- animations ---------------------------------------------------------------------


@dataclass
class AFrame:
	root_rotation: "np.ndarray"	# (3,) degrees: alpha (X), beta (Y), gamma (Z)
	root_translation: "np.ndarray"	# (3,)
	rotations: "np.ndarray"	# (bones, 3) degrees


def read_a(data: bytes, name: str = "a") -> list[AFrame]:
	if len(data) < 36:
		raise ValueError(f"{name}: header truncated")
	version, frames, bones = struct.unpack_from("<iii", data, 0)
	if version != 1 or frames < 0 or not 0 <= bones <= 255:
		raise ValueError(f"{name}: not a field animation")
	size = 24 + 12 * bones
	frames = min(frames, (len(data) - 36) // size)
	out = []
	for f in range(frames):
		v = np.frombuffer(data, "<f4", size // 4, 36 + f * size).astype(np.float64)
		out.append(AFrame(v[0:3], v[3:6], v[6:].reshape(bones, 3)))
	return out


def rotation(deg: "np.ndarray") -> "np.ndarray":
	"""FF7 bone rotation: Ry(beta) · Rx(alpha) · Rz(gamma)."""
	a, b, g = np.radians(deg)
	ca, sa, cb, sb, cg, sg = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(g), math.sin(g)
	rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
	ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
	rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
	return ry @ rx @ rz


def bone_matrices(bones: list[HrcBone], frame: AFrame | None) -> tuple["np.ndarray", list["np.ndarray"]]:
	"""(root matrix, per-bone matrix where its part is drawn), in FF7 space."""
	root = np.eye(4)
	if frame is not None:
		t = frame.root_translation
		root[:3, 3] = (t[0], -t[1], t[2])
		root[:3, :3] = rotation(frame.root_rotation)
	ends: dict[str, np.ndarray] = {}
	out = []
	for i, b in enumerate(bones):
		parent_end = ends.get(b.parent.lower(), root)
		m = np.eye(4)
		if frame is not None and i < len(frame.rotations):
			m[:3, :3] = rotation(frame.rotations[i])
		w = parent_end @ m
		out.append(w)
		step = np.eye(4)
		step[2, 3] = -b.length
		ends[b.name.lower()] = w @ step
	return root, out


# -- battle models ------------------------------------------------------------------
#
# battle.lgp holds each battle model as files with no extension, all named from a
# two-letter base (Cloud is "rt"):
#
#	<base>aa	skeleton: header (bone count at 12, texture count at 24, body and
#			weapon animation counts at 28 and 40), then 12 bytes per bone
#			from 52: parent (-1 = root), length, has-part flag
#	<base>ac…	textures (.TEX), as many as the header says
#	<base>am…	one .P part per bone, in bone order (am…az, ba…bz, …)
#	<base>ck…	weapon models (not converted)
#	<base>da	animation pack: a count, then the body animations, then the
#			weapon animations, each a compressed bit stream
#
# Read the way Kimera reads them (FF7AASkeleton, FF7DAAnimationsPack*; no code
# copied) and the Qhimm wiki page Battle_Animation_(PC).


@dataclass
class BattleBone:
	parent: int	# -1 = root
	length: float
	has_part: bool


@dataclass
class BattleSkeleton:
	bones: list[BattleBone]
	textures: int
	body_anims: int
	weapon_anims: int


def is_battle_skeleton(path: Path) -> bool:
	"""An extracted battle.lgp skeleton: a four-letter name ending "aa" and a matching header."""
	path = Path(path)
	if path.suffix or len(path.name) != 4 or not path.name.lower().endswith("aa"):
		return False
	try:
		with path.open("rb") as f:
			head = f.read(52)
		size = path.stat().st_size
	except OSError:
		return False
	if len(head) < 52:
		return False
	bones = struct.unpack_from("<i", head, 12)[0]
	return 0 < bones <= 255 and size >= 52 + 12 * bones


def read_battle_skeleton(data: bytes, name: str = "aa") -> BattleSkeleton:
	if len(data) < 52:
		raise ValueError(f"{name}: header truncated")
	nb, = struct.unpack_from("<i", data, 12)
	tex, body = struct.unpack_from("<ii", data, 24)
	weapon, = struct.unpack_from("<i", data, 40)
	if nb == 0:
		raise ValueError(f"{name}: a battle location (no bones), not a character")
	if not 0 < nb <= 255 or len(data) < 52 + 12 * nb:
		raise ValueError(f"{name}: not a battle skeleton")
	bones = []
	for i in range(nb):
		parent, length, part = struct.unpack_from("<ifi", data, 52 + 12 * i)
		if parent >= i:
			raise ValueError(f"{name}: bone {i} has parent {parent}, which comes after it")
		bones.append(BattleBone(max(parent, -1), length, part != 0))
	return BattleSkeleton(bones, max(0, min(tex, 64)), max(0, min(body, 1024)), max(0, min(weapon, 1024)))


def battle_part_suffixes(count: int) -> list[str]:
	"""Part file suffixes for each bone: am, an … az, ba, bb …"""
	out, a, b = [], ord("a"), ord("m")
	for _ in range(count):
		out.append(chr(a) + chr(b))
		if b >= ord("z"):
			a, b = a + 1, ord("a")
		else:
			b += 1
	return out


class _Bits:
	"""An MSB-first bit stream."""

	def __init__(self, data: bytes):
		self.data = data
		self.pos = 0
		self.end = 8 * len(data)

	def unsigned(self, n: int) -> int:
		if self.pos + n > self.end:
			raise EOFError
		first, last = self.pos >> 3, (self.pos + n + 7) >> 3
		v = int.from_bytes(self.data[first:last], "big")
		v >>= 8 * (last - first) - (self.pos & 7) - n
		self.pos += n
		return v & ((1 << n) - 1)

	def signed(self, n: int) -> int:
		v = self.unsigned(n)
		return v - (1 << n) if v >> (n - 1) else v


def _rotation_delta(bits: _Bits, key: int) -> int:
	if not bits.unsigned(1):
		return 0
	kind = bits.unsigned(3)
	if kind == 0:
		v = -1	# the smallest step down
	elif kind == 7:
		v = bits.signed(12 - key)
	else:
		v = bits.signed(kind)
		v = v - (1 << (kind - 1)) if v < 0 else v + (1 << (kind - 1))
	return v << key


def read_da_animation(stream: bytes, key: int, slots: int) -> list[tuple["np.ndarray", "np.ndarray"]]:
	"""One body animation's bit stream → per frame (translation (3,), raw rotations (slots, 3)).
	Frames are read until the data runs out, as Kimera does: the counts in the headers are unreliable."""
	bits = _Bits(stream)
	out = []
	try:
		t = np.array([bits.signed(16) for _ in range(3)], np.int64)
		r = np.array([[bits.signed(12 - key) << key for _ in range(3)] for _ in range(slots)], np.int64).reshape(slots, 3)
	except EOFError:
		return out
	out.append((t, r))
	while True:
		try:
			nt = t.copy()
			for c in range(3):
				nt[c] += bits.signed(16 if bits.unsigned(1) else 7)
			nr = r.copy()
			for s in range(slots):
				for c in range(3):
					nr[s, c] += _rotation_delta(bits, key)
		except EOFError:
			break
		t, r = nt, nr
		out.append((t, r))
	return out


def read_da(data: bytes, bones: int, body_anims: int, name: str = "da") -> list[list[AFrame] | None]:
	"""A battle animation pack's body animations as frames for a skeleton of `bones` bones.
	Empty or unreadable slots come back as None so the rest keep their numbers."""
	if len(data) < 4:
		raise ValueError(f"{name}: truncated")
	slots = bones + 1 if bones > 1 else 1	# slot 0 is the root rotation
	at = 4
	out: list[list[AFrame] | None] = []
	for _ in range(body_anims):
		if at + 12 > len(data):
			break
		block, = struct.unpack_from("<I", data, at + 8)
		start = at + 12
		at = start + block
		if block < 11 or at > len(data):
			out.append(None)
			continue
		frames2, length, key = struct.unpack_from("<hHB", data, start)
		if frames2 == block - 5:	# no second frame counter (only some models)
			length, key = struct.unpack_from("<HB", data, start)
			body = data[start + 3:start + 3 + length]
		else:
			body = data[start + 5:start + 5 + length]
		if key not in (0, 2, 4):
			out.append(None)
			continue
		frames = []
		for t, r in read_da_animation(body, key, slots):
			deg = r.astype(np.float64) * (360.0 / 4096.0)
			rots = deg[1:] if slots > 1 else deg[:1]
			# the root moves by (x, y, z) in the model's own y-down space; bone_matrices negates y, so undo that here
			frames.append(AFrame(deg[0], np.array([t[0], -t[1], t[2]], np.float64), rots))
		out.append(frames or None)
	return out


def _battle_hrc(skel: BattleSkeleton) -> list[HrcBone]:
	"""Battle bones as HRC bones: a battle bone moves +length along Z, a field bone -length."""
	return [HrcBone(f"bone{i}", "root" if b.parent < 0 else f"bone{b.parent}", -b.length, [])
		for i, b in enumerate(skel.bones)]


def _battle_files(path: Path) -> tuple[str, Path]:
	path = Path(path)
	return path.name[:-2], path.parent


def import_battle(path: Path, opts: "_m.ModelOptions | None" = None) -> "_m.ImportedModel":
	"""<base>aa with its parts, textures and <base>da pack → ImportedModel."""
	opts = opts or _m.ModelOptions()
	path = Path(path)
	warnings: list[str] = []
	skel = read_battle_skeleton(path.read_bytes(), path.name)
	base, folder = _battle_files(path)
	textures = []
	for ti in range(skel.textures):
		tp = _find(folder, f"{base}a{chr(ord('c') + ti)}")
		try:
			textures.append(read_tex(tp.read_bytes(), tp.name) if tp else None)
		except ValueError as exc:
			warnings.append(str(exc))
			textures.append(None)
		if tp is None:
			warnings.append(f"texture {base}a{chr(ord('c') + ti)} not found")
	parts = []
	for i, (b, suf) in enumerate(zip(skel.bones, battle_part_suffixes(len(skel.bones)))):
		if not b.has_part:
			continue
		pp = _find(folder, base + suf)
		if pp is None:
			warnings.append(f"bone {i}: part {base}{suf} not found")
			continue
		try:
			parts.append(_Part(i + 1, read_p(pp.read_bytes(), pp.name), textures, pp.name))
		except ValueError as exc:
			warnings.append(str(exc))
	if not parts:
		raise ValueError(f"{path.name}: no parts found (are the {base}am… files in the same folder?)")
	anims: list[tuple[str, list[AFrame]]] = []
	da = _find(folder, f"{base}da")
	if da is None:
		warnings.append(f"no {base}da animation pack beside the skeleton; bones keep a straight rest pose")
	else:
		try:
			for i, frames in enumerate(read_da(da.read_bytes(), len(skel.bones), skel.body_anims, da.name)):
				if frames:
					anims.append((f"anim{i:02d}", frames))
		except ValueError as exc:
			warnings.append(str(exc))
		if not anims:
			warnings.append(f"{da.name}: no body animation could be read")
	bones = _battle_hrc(skel)
	scale = float(getattr(opts, "ff7_battle_scale", 0.0))
	if scale <= 0:	# auto: the rest pose stands BATTLE_HEIGHT units tall
		probe = _assemble(base, bones, parts, anims[:1], _m.ModelOptions(static=True), [], scale=1.0, yaw=0.0)
		pts = np.concatenate([m.world for m in probe.meshes])
		height = float(np.ptp(pts[:, 2]))
		scale = BATTLE_HEIGHT / height if height > 1e-6 else 0.05
		warnings.append(f"battle scale {scale:.4g} (auto: {height:.0f} FF7 units tall → {BATTLE_HEIGHT:g}); "
			"set --ff7-battle-scale to keep a set of models in proportion")
	return _assemble(base, bones, parts, anims, opts, warnings, fps=BATTLE_FPS, scale=scale)


# -- import -------------------------------------------------------------------------


@dataclass
class _Part:
	bone: int	# studio bone (0 = root)
	model: PModel
	textures: list["np.ndarray | None"]
	label: str


def import_ff7(path: Path, opts: "_m.ModelOptions | None" = None) -> "_m.ImportedModel":
	"""A field skeleton (.hrc), a manifest (.ff7mdl), a lone polygon model (.p) or a battle
	skeleton (<base>aa) → ImportedModel."""
	opts = opts or _m.ModelOptions()
	path = Path(path)
	if not path.suffix:
		return import_battle(path, opts)
	warnings: list[str] = []
	ext = path.suffix.lower()
	anims: list[tuple[str, list[AFrame]]] = []
	folder = path.parent
	if ext == ".ff7mdl":
		spec = json.loads(path.read_text(encoding="utf-8"))
		folder = path.parent / spec.get("folder", ".")
		hrc = _find(folder, spec["skeleton"])
		if hrc is None:
			raise ValueError(f"{spec['skeleton']}: not found next to {path.name}")
		for a in spec.get("animations", []):
			ap = _find(folder, a)
			if ap is None:
				warnings.append(f"animation {a} not found")
				continue
			try:
				anims.append((Path(a).stem, read_a(ap.read_bytes(), ap.name)))
			except ValueError as exc:
				warnings.append(str(exc))
		name = spec.get("name") or path.stem
		bones = read_hrc(hrc.read_text(encoding="latin-1"), hrc.name)
	elif ext == ".hrc":
		bones = read_hrc(path.read_text(encoding="latin-1"), path.name)
		name = path.stem.lower()
	else:
		pm = read_p(path.read_bytes(), path.name)
		return _assemble(path.stem.lower(), [], [_Part(0, pm, _textures_beside(path, pm, warnings), path.stem)], [],
			opts, warnings)
	parts: list[_Part] = []
	for i, b in enumerate(bones):
		for res in b.resources:
			rp = _find(folder, f"{res}.rsd")
			if rp is None:
				warnings.append(f"bone {b.name}: {res}.rsd not found")
				continue
			rsd = read_rsd(rp.read_text(encoding="latin-1"), rp.name)
			pp = _find(folder, f"{Path(rsd.ply).stem}.p")
			if pp is None:
				warnings.append(f"{rp.name}: {Path(rsd.ply).stem}.p not found")
				continue
			texs = []
			for t in rsd.textures:
				tp = _find(folder, f"{Path(t).stem}.tex")
				try:
					texs.append(read_tex(tp.read_bytes(), tp.name) if tp else None)
				except ValueError as exc:
					warnings.append(str(exc))
					texs.append(None)
				if tp is None:
					warnings.append(f"{rp.name}: texture {Path(t).stem}.tex not found")
			parts.append(_Part(i + 1, read_p(pp.read_bytes(), pp.name), texs, pp.stem))
	if not parts:
		raise ValueError(f"{name}: no parts found (are the .rsd/.p files in the same folder?)")
	if not anims:
		warnings.append("no animation given, so bones keep a straight rest pose; name animations in a .ff7mdl manifest")
	return _assemble(name, bones, parts, anims, opts, warnings)


def _textures_beside(path: Path, pm: PModel, warnings: list[str]) -> list:
	"""A lone .p: textures from a same-named .rsd if there is one."""
	for rp in path.parent.glob("*"):
		if rp.suffix.lower() != ".rsd":
			continue
		try:
			rsd = read_rsd(rp.read_text(encoding="latin-1"), rp.name)
		except (OSError, ValueError):
			continue
		if Path(rsd.ply).stem.lower() == path.stem.lower():
			out = []
			for t in rsd.textures:
				tp = _find(path.parent, f"{Path(t).stem}.tex")
				try:
					out.append(read_tex(tp.read_bytes(), tp.name) if tp else None)
				except ValueError as exc:
					warnings.append(str(exc))
					out.append(None)
			return out
	if any(g.textured for g in pm.groups):
		warnings.append("textured groups but no .rsd names their textures; drawn with vertex colours")
	return []


def _to_gold(opts: "_m.ModelOptions", scale: float | None = None, yaw: float | None = None) -> tuple["np.ndarray", float]:
	base = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])	# y-down, facing -Z → Z-up, +X
	a = math.radians(getattr(opts, "ff7_yaw", 0.0) if yaw is None else yaw)
	c, s = math.cos(a), math.sin(a)
	yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
	return yaw @ base, float(getattr(opts, "ff7_scale", DEFAULT_SCALE) if scale is None else scale)


def _gold(m: "np.ndarray", rot: "np.ndarray", k: float) -> "np.ndarray":
	out = np.eye(4)
	out[:3, :3] = rot @ m[:3, :3] @ rot.T
	out[:3, 3] = k * (rot @ m[:3, 3])
	return out


def _poses(bones: list[HrcBone], frame: AFrame | None, rot: "np.ndarray", k: float) -> tuple[list, "np.ndarray"]:
	root, mats = bone_matrices(bones, frame)
	worlds = [_gold(root, rot, k)] + [_gold(m, rot, k) for m in mats]
	parents = [-1] + [_parent(bones, i) for i in range(len(bones))]
	poses = np.zeros((len(worlds), 6))
	for i, w in enumerate(worlds):
		rel = w if parents[i] < 0 else _sk.rigid_inverse(worlds[parents[i]]) @ w
		poses[i] = _sk.pose_from_matrix(rel)
	return worlds, poses


def _parent(bones: list[HrcBone], i: int) -> int:
	"""Studio parent of HRC bone i (studio index i + 1); 0 is the root."""
	want = bones[i].parent.lower()
	for j in range(i - 1, -1, -1):
		if bones[j].name.lower() == want:
			return j + 1
	return 0


def _palette(colors: "np.ndarray") -> tuple["np.ndarray", dict[tuple, tuple[float, float]]]:
	"""One 4x4 block per colour in a skin, and each colour's UV."""
	uniq = sorted(set(map(tuple, colors.tolist())))
	if len(uniq) > 256:
		from PIL import Image

		img = Image.fromarray(np.array(uniq, np.uint8)[None, :, :3], "RGB").quantize(256)
		pal = np.array(img.getpalette()[:768], np.uint8).reshape(-1, 3)
		idx = np.array(img).reshape(-1)
		remap = {c: tuple(int(x) for x in pal[i]) + (255,) for c, i in zip(uniq, idx)}
		uniq = sorted(set(remap.values()))
	else:
		remap = {c: c for c in uniq}
	side = 4
	while (side // 4) ** 2 < len(uniq):
		side *= 2
	side = max(side, 8)
	skin = np.zeros((side, side, 4), np.uint8)
	skin[..., 3] = 255
	cells = side // 4
	where: dict[tuple, tuple[float, float]] = {}
	for i, c in enumerate(uniq):
		y, x = divmod(i, cells)
		skin[y * 4:y * 4 + 4, x * 4:x * 4 + 4, :3] = c[:3]
		where[c] = ((x * 4 + 2) / side, (y * 4 + 2) / side)
	return skin, {c: where[remap[c]] for c in remap}


def _assemble(name: str, bones: list[HrcBone], parts: list[_Part], anims: list[tuple[str, list[AFrame]]],
		opts: "_m.ModelOptions", warnings: list[str], fps: float = FPS, scale: float | None = None,
		yaw: float | None = None) -> "_m.ImportedModel":
	rot, k = _to_gold(opts, scale, yaw)
	model = _m.ImportedModel(name, [], [], warnings)
	if len(bones) + 1 > _sk.MAXSTUDIOBONES:
		raise ValueError(f"{len(bones)} bones exceeds the studio limit")
	rest = anims[0][1][0] if anims and anims[0][1] else None
	if rest is not None and len(rest.rotations) != len(bones):
		warnings.append(f"animation {anims[0][0]} has {len(rest.rotations)} bones, the skeleton {len(bones)}")
	worlds, poses = _poses(bones, rest, rot, k)
	model.bones = [_m.Bone("root", -1, poses[0])]
	used: set[str] = {"root"}
	for i, b in enumerate(bones):
		model.bones.append(_m.Bone(_m._unique(b.name[:31] or f"bone{i}", used), _parent(bones, i), poses[i + 1]))

	plain_faces: list[tuple] = []	# (part, bone, world corners, normals, rgba per face)
	tex_meshes: list = []
	for part in parts:
		pm = part.model
		w = worlds[part.bone]
		for gi, g in enumerate(pm.groups):
			end = min(len(pm.polys), g.poly_start + g.poly_count)
			if end <= g.poly_start:
				continue
			rel = pm.polys[g.poly_start:end]
			vidx = rel + g.vert_start
			ok = (vidx < len(pm.vertices)).all(1)
			if not ok.all():
				warnings.append(f"{part.label} group {gi}: {int((~ok).sum())} triangles with bad indices dropped")
			vidx, rel = vidx[ok], rel[ok]
			local = pm.vertices[vidx]	# (m, 3, 3) FF7 bone space
			world = k * (local @ rot.T)	# GoldSrc axes, still in the part's bone frame
			world = world @ w[:3, :3].T + w[:3, 3]
			fn = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
			nidx = pm.pnormals[g.poly_start:end][ok]
			if len(pm.normals) and (nidx < len(pm.normals)).all():
				vn = pm.normals[nidx] @ rot.T @ w[:3, :3].T
				score = np.einsum("mi,mi->m", fn, vn.sum(1))
				flip = score < 0
			else:
				vn = np.repeat(fn[:, None, :], 3, 1)
				flip = np.zeros(len(fn), bool)
			# FF7 winds faces both ways; turn each triangle to agree with its normals
			world[flip] = world[flip][:, [0, 2, 1]]
			vn[flip] = vn[flip][:, [0, 2, 1]]
			rel = rel.copy()
			rel[flip] = rel[flip][:, [0, 2, 1]]
			vidx = rel + g.vert_start
			tex = part.textures[g.texture] if g.textured and g.texture < len(part.textures) else None
			if g.textured and tex is None and part.textures:
				warnings.append(f"{part.label} group {gi}: texture {g.texture} missing; using vertex colours")
			if tex is not None:
				tidx = rel + g.tex_start
				good = (tidx < len(pm.texcoords)).all(1)
				uv = np.zeros((len(rel), 3, 2))
				uv[good] = pm.texcoords[tidx[good]]
				tex_meshes.append((part, gi, tex, world, vn, uv))
			else:
				cols = pm.vcolors[vidx].astype(np.int64).mean(1).round().astype(np.uint8)	# Gouraud → flat
				plain_faces.append((part, world, vn, cols))

	def add(mat: int, part: _Part, world: "np.ndarray", vn: "np.ndarray", uv: "np.ndarray") -> None:
		p = world.reshape(-1, 3)
		nn = vn.reshape(-1, 3)
		lens = np.linalg.norm(nn, axis=1, keepdims=True)
		nn = np.where(lens > 1e-9, nn / np.maximum(lens, 1e-9), [0.0, 0.0, 1.0])
		w = worlds[part.bone]
		local = (p - w[:3, 3]) @ w[:3, :3]
		ln = nn @ w[:3, :3]
		tris = np.arange(len(p), dtype=np.uint32).reshape(-1, 3)
		model.meshes.append(_m.Submesh(mat, local.astype(np.float32), ln.astype(np.float32),
			uv.reshape(-1, 2).astype(np.float32), tris, np.full(len(p), part.bone, np.int32), p.astype(np.float32)))

	if plain_faces:
		allc = np.concatenate([c for *_x, c in plain_faces])
		skin, where = _palette(allc)
		model.materials.append(_m.Material("vertex_colours", skin, False, "vertex colours"))
		for part, world, vn, cols in plain_faces:
			uv = np.repeat(np.array([where[tuple(c)] for c in cols.tolist()], np.float64)[:, None, :], 3, 1)
			add(0, part, world, vn, uv)
	seen: dict[int, int] = {}
	for part, gi, tex, world, vn, uv in tex_meshes:
		key = id(tex)
		if key not in seen:
			rgba = _m._fit_skin(tex, opts.skin_max_edge).copy()
			masked = bool((rgba[..., 3] < opts.alpha_threshold).any())
			rgba[..., 3] = np.where(rgba[..., 3] < opts.alpha_threshold, 0, 255) if masked else 255
			seen[key] = len(model.materials)
			model.materials.append(_m.Material(f"{part.label}_{gi}"[:63], rgba, masked, part.label))
		add(seen[key], part, world, vn, uv)
	if not model.meshes:
		raise ValueError(f"{name}: no triangles")

	if bones and not opts.static:
		used_seq: set[str] = set()
		for aname, frames in anims:
			if not frames:
				continue
			if len(frames[0].rotations) < len(bones):
				warnings.append(f"animation {aname}: {len(frames[0].rotations)} bones for a {len(bones)}-bone skeleton; skipped")
				continue
			src = len(frames)
			count = max(1, int(round((src - 1) / fps * opts.anim_fps)) + 1)
			out = np.zeros((count, len(bones) + 1, 6))
			for f in range(count):
				t = min(src - 1, f / opts.anim_fps * fps)
				i0 = int(math.floor(t))
				i1 = min(i0 + 1, src - 1)
				fr = _blend(frames[i0], frames[i1], t - i0)
				out[f] = _poses(bones, fr, rot, k)[1]
			model.sequences.append(_m.Sequence(_m._unique(_m._seq_name(aname), used_seq), float(opts.anim_fps), out))
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _blend(a: AFrame, b: AFrame, t: float) -> AFrame:
	"""Between two frames: angles take the short way round."""
	if t <= 1e-6:
		return a

	def lerp_deg(x, y):
		d = (y - x + 180.0) % 360.0 - 180.0
		return x + d * t

	return AFrame(lerp_deg(a.root_rotation, b.root_rotation), a.root_translation + (b.root_translation - a.root_translation) * t,
		lerp_deg(a.rotations, b.rotations))


def describe(path: Path) -> dict:
	"""What `gxbake ff7` prints."""
	path = Path(path)
	ext = path.suffix.lower()
	if ext == ".p":
		pm = read_p(path.read_bytes(), path.name)
		return {"path": str(path), "type": "polygon model", "vertices": len(pm.vertices), "triangles": len(pm.polys),
			"groups": len(pm.groups), "textured groups": sum(g.textured for g in pm.groups)}
	if ext == ".hrc":
		bones = read_hrc(path.read_text(encoding="latin-1"), path.name)
		return {"path": str(path), "type": "skeleton", "bones": [
			{"name": b.name, "parent": b.parent, "length": b.length, "resources": b.resources} for b in bones]}
	if ext == ".a":
		frames = read_a(path.read_bytes(), path.name)
		return {"path": str(path), "type": "animation", "frames": len(frames),
			"bones": len(frames[0].rotations) if frames else 0, "seconds": round(len(frames) / FPS, 2)}
	if not ext and is_battle_skeleton(path):
		skel = read_battle_skeleton(path.read_bytes(), path.name)
		base, folder = _battle_files(path)
		sufs = battle_part_suffixes(len(skel.bones))
		info = {"path": str(path), "type": "battle skeleton", "bones": [
			{"name": f"bone{i}", "parent": "root" if b.parent < 0 else f"bone{b.parent}", "length": b.length,
				"resources": [base + sufs[i]] if b.has_part else []} for i, b in enumerate(skel.bones)],
			"textures": skel.textures, "body animations": skel.body_anims, "weapon animations": skel.weapon_anims}
		da = _find(folder, f"{base}da")
		if da is not None:
			anims = read_da(da.read_bytes(), len(skel.bones), skel.body_anims, da.name)
			info["animation frames"] = [len(a) if a else 0 for a in anims]
		return info
	if ext == ".tex":
		rgba = read_tex(path.read_bytes(), path.name)
		return {"path": str(path), "type": "texture", "size": f"{rgba.shape[1]}x{rgba.shape[0]}",
			"transparent": bool((rgba[..., 3] < 128).any())}
	raise ValueError(f"{path.name}: not a .p, .hrc, .a or .tex file, or a battle skeleton (<base>aa)")


def animations_for(folder: Path, bones: int) -> list[Path]:
	"""Every .a in a folder whose bone count matches (to help write a manifest)."""
	out = []
	for p in sorted(Path(folder).iterdir()):
		if p.suffix.lower() != ".a":
			continue
		try:
			with p.open("rb") as f:
				head = f.read(12)
			if len(head) == 12 and struct.unpack("<iii", head)[0] == 1 and struct.unpack("<iii", head)[2] == bones:
				out.append(p)
		except OSError:
			continue
	return out
