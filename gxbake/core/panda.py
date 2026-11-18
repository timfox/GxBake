"""Panda3D models (.egg, .bam, .pz) and Toontown-style actors → GoldSrc studio MDL.

EGG is Panda3D's text scene format and is parsed here directly: vertex pools,
polygons, textures (including the separate `alpha-file` Toontown pairs a .jpg
with), materials, characters (<Dart> groups), joints with vertex membership and
<Bundle> animation tables. The result is the same ImportedModel the Assimp,
ROSE and Source importers produce, so it bakes into models/ + gc_studio/.

BAM is Panda3D's binary object stream. Its layout follows Panda's C++ classes
and changes with every engine version, so it is turned into EGG first by
Panda3D itself: the `panda3d` wheel when it is installed (pip install panda3d),
else a `bam2egg` program on PATH or named by GXBAKE_BAM2EGG.

Animations live in their own files (Panda actors load "model" + "anim" files).
Next to a model GxBake picks up the animation-only files that share its name
stem (panda-model.egg + panda-walk.egg, Toontown's tt_a_chr_..._1000.bam +
tt_a_chr_..._neutral.bam, suitA-mod.bam + suitA-walk.bam), also from the same
folder under the other phase_N trees of an extracted Toontown install.

A `.pandactor` JSON file joins a multi-part actor (Toontown toons are legs,
torso and head characters fastened at joints) into one studio model:

	{"parts": [{"model": "legs.bam"},
	           {"model": "torso.bam", "attach": "joint_hips"},
	           {"model": "head.bam", "attach": "def_head"}],
	 "anims": {"walk": ["legs-walk.bam", "torso-walk.bam"]}}
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as _m
from . import skeleton as _sk

PANDA_EXTENSIONS = {".egg", ".bam", ".pandactor"}
_LOD_RE = re.compile(r"([_-](\d+|lod\d*|hi|med|low|mod|model|zero))+$", re.I)
_COLLISION_TYPES = {"barrier", "trigger", "sphere", "tube", "floor", "polyset", "ground", "camera-collide",
	"barrier-no-mask", "invsphere", "trigger-sphere", "shadow", "dupefloor", "smooth-floors"}
_ONESHOT_WORDS = ("die", "death", "jump", "hit", "fall", "slip", "throw", "into", "outof", "sit",
	"teleport", "victory", "cringe", "conked", "pushbutton", "cast")
_COLOURS_MATERIAL = "colours"	# untextured polygons share one swatch skin


def is_panda(path: Path) -> bool:
	""".egg, .bam, their .pz-compressed forms, and .pandactor manifests."""
	name = Path(path).name.lower()
	if name.endswith(".pz"):
		name = name[:-3]
	return Path(name).suffix in PANDA_EXTENSIONS


def _kind(path: Path) -> str:
	name = Path(path).name.lower()
	if name.endswith(".pz"):
		name = name[:-3]
	return Path(name).suffix


def _stem(path: Path) -> str:
	name = Path(path).name
	if name.lower().endswith(".pz"):
		name = name[:-3]
	return Path(name).stem


# -- EGG syntax ---------------------------------------------------------------


@dataclass
class EggNode:
	tag: str	# lower case, without <>
	name: str
	kids: list["EggNode"] = field(default_factory=list)
	vals: list[str] = field(default_factory=list)

	def kid(self, tag: str, name: str | None = None) -> "EggNode | None":
		for k in self.kids:
			if k.tag == tag and (name is None or k.name.lower() == name):
				return k
		return None

	def all(self, tag: str) -> list["EggNode"]:
		return [k for k in self.kids if k.tag == tag]

	def floats(self) -> list[float]:
		out = []
		for v in self.vals:
			try:
				out.append(float(v))
			except ValueError:
				pass
		return out

	def scalar(self, name: str, default: str | None = None) -> str | None:
		for k in self.kids:
			if k.tag == "scalar" and k.name.lower() == name and k.vals:
				return k.vals[0]
		return default


_TOKEN = re.compile(r'\s+|//[^\n]*|/\*.*?\*/|"((?:\\.|[^"\\])*)"|<([^>]*)>|([{}])|([^\s{}"]+)', re.S)


def parse_egg(text: str) -> EggNode:
	"""EGG text → a tree of <Tag> name { values and children } entries."""
	root = EggNode("", "")
	stack = [root]
	tag: str | None = None
	name: str | None = None
	for m in _TOKEN.finditer(text):
		quoted, t, brace, word = m.group(1), m.group(2), m.group(3), m.group(4)
		if t is not None:
			tag, name = t.strip().lower(), None
		elif brace == "{":
			node = EggNode(tag or "", name or "")
			stack[-1].kids.append(node)
			stack.append(node)
			tag = name = None
		elif brace == "}":
			if len(stack) > 1:
				stack.pop()
			tag = name = None
		elif quoted is not None or word is not None:
			v = bytes(quoted, "utf-8").decode("unicode_escape", "replace") if quoted is not None and "\\" in quoted else (
				quoted if quoted is not None else word)
			if tag is not None and name is None:
				name = v
			else:
				stack[-1].vals.append(v)
	return root


# -- maths --------------------------------------------------------------------


def _rot(deg: float, axis) -> "np.ndarray":
	"""4x4 counter-clockwise rotation about an axis (column-vector convention)."""
	x, y, z = np.asarray(axis, float) / (np.linalg.norm(axis) or 1.0)
	a = math.radians(deg)
	c, s, t = math.cos(a), math.sin(a), 1 - math.cos(a)
	m = np.eye(4)
	m[:3, :3] = [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
		[t * x * y + s * z, t * y * y + c, t * y * z - s * x],
		[t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
	return m


def _translate(v) -> "np.ndarray":
	m = np.eye(4)
	m[:3, 3] = v
	return m


def _scale(v) -> "np.ndarray":
	m = np.eye(4)
	m[0, 0], m[1, 1], m[2, 2] = v
	return m


def egg_transform(node: "EggNode | None") -> "np.ndarray":
	"""An egg <Transform> (components applied in order, row-vector style) as a column 4x4."""
	m = np.eye(4)
	if node is None:
		return m
	for k in node.kids:
		v = k.floats()
		if k.tag == "matrix4" and len(v) >= 16:
			x = np.array(v[:16]).reshape(4, 4).T	# egg matrices are row-vector: translation in the last row
		elif k.tag == "translate" and len(v) >= 3:
			x = _translate(v[:3])
		elif k.tag == "rotate" and len(v) >= 4:
			x = _rot(v[0], v[1:4])
		elif k.tag in ("rotx", "roty", "rotz") and v:
			x = _rot(v[0], {"rotx": (1, 0, 0), "roty": (0, 1, 0), "rotz": (0, 0, 1)}[k.tag])
		elif k.tag == "scale" and v:
			x = _scale(v[:3] if len(v) >= 3 else [v[0]] * 3)
		else:
			continue
		m = x @ m
	return m


_STANDARD_ORDER = "srpht"	# Panda's default with temp-hpr-fix (roll, then pitch, then heading)


_AXES = {	# (up, right, forward): heading, pitch and roll axes in each coordinate system
	"z-up": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
	"y-up": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
}


def compose(values: dict[str, float], order: str = _STANDARD_ORDER, cs: str = "z-up") -> "np.ndarray":
	"""One frame of an <Xfm$Anim_S$> table (i j k scale, h p r degrees, x y z) → column 4x4,
	in the file's own coordinate system (heading turns about its up axis, and so on)."""
	up, right, fwd = _AXES["y-up" if cs.startswith("y-up") else "z-up"]
	m = np.eye(4)
	for c in order:
		if c == "s":
			x = _scale([values.get("i", 1.0), values.get("j", 1.0), values.get("k", 1.0)])
		elif c == "h":
			x = _rot(values.get("h", 0.0), up)
		elif c == "p":
			x = _rot(values.get("p", 0.0), right)
		elif c == "r":
			x = _rot(values.get("r", 0.0), fwd)
		elif c == "t":
			x = _translate([values.get("x", 0.0), values.get("y", 0.0), values.get("z", 0.0)])
		else:
			continue
		m = x @ m
	return m


# -- the egg file as data -----------------------------------------------------


@dataclass
class Pool:
	pos: dict[int, tuple] = field(default_factory=dict)
	normal: dict[int, tuple] = field(default_factory=dict)
	uv: dict[int, tuple] = field(default_factory=dict)
	rgba: dict[int, tuple] = field(default_factory=dict)


@dataclass
class Texture:
	file: str
	alpha_file: str = ""
	alpha_channel: int = 0


@dataclass
class Poly:
	pool: str
	verts: list[int]
	texture: str	# texture entry name ("" = untextured)
	colour: "tuple | None"	# polygon <RGBA> or material diffuse
	two_sided: bool
	joint: int	# enclosing <Joint> (rigid geometry), -1 = none


@dataclass
class Joint:
	name: str
	parent: int
	local: "np.ndarray"	# bind transform relative to the parent (column 4x4, egg space)
	refs: list[tuple[str, list[int], float]] = field(default_factory=list)	# (pool, vertices, membership)


@dataclass
class Anim:
	name: str	# the <Bundle> name: by Panda convention the character it drives
	fps: float
	tables: dict[str, "np.ndarray"]	# joint name → (frames, 4, 4) local matrices
	source: str = ""

	@property
	def num_frames(self) -> int:
		return max((len(t) for t in self.tables.values()), default=0)


@dataclass
class EggFile:
	name: str
	path: Path
	coordsys: str = "z-up"
	pools: dict[str, Pool] = field(default_factory=dict)
	textures: dict[str, Texture] = field(default_factory=dict)
	materials: dict[str, tuple] = field(default_factory=dict)
	polys: list[Poly] = field(default_factory=list)
	joints: list[Joint] = field(default_factory=list)
	character: str = ""
	anims: list[Anim] = field(default_factory=list)
	warnings: list[str] = field(default_factory=list)

	@property
	def animation_only(self) -> bool:
		return bool(self.anims) and not self.polys

	@property
	def joint_names(self) -> set[str]:
		return {j.name for j in self.joints}


def _first_node(root: EggNode, tags: set[str]):
	stack = [root]
	while stack:
		n = stack.pop()
		for k in n.kids:
			if k.tag in tags:
				yield k
			stack.append(k)


def read_egg(text: str, name: str = "egg", path: Path | None = None) -> EggFile:
	root = parse_egg(text)
	egg = EggFile(name, Path(path) if path else Path(name))
	egg.coordsys = "y-up"	# what Panda assumes when an egg does not say
	cs = root.kid("coordinatesystem")
	if cs is not None and cs.vals:
		egg.coordsys = cs.vals[0].lower()
		if egg.coordsys.endswith("left"):
			egg.warnings.append(f"left-handed coordinate system {cs.vals[0]} read as right-handed (mirrored)")

	# Pools, textures and materials can be referenced before or after they appear.
	for n in _first_node(root, {"vertexpool", "texture", "material"}):
		if n.tag == "vertexpool":
			egg.pools[n.name] = _read_pool(n)
		elif n.tag == "texture":
			if not n.vals:
				continue
			ch = n.scalar("alpha-file-channel", "0")
			egg.textures[n.name] = Texture(n.vals[0], n.scalar("alpha-file", "") or "", int(float(ch or 0)))
		else:
			col = [n.scalar(k) for k in ("diffr", "diffg", "diffb", "diffa")]
			if any(c is not None for c in col[:3]):
				egg.materials[n.name] = tuple(float(c) if c is not None else 1.0 for c in col)

	_walk(root, egg, -1)
	for n in _first_node(root, {"bundle"}):
		anim = _read_bundle(n, egg.coordsys)
		if anim.tables:
			anim.source = name
			egg.anims.append(anim)
	if egg.coordsys.startswith("y-up"):
		_to_zup(egg)
	return egg


_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _to_zup(egg: EggFile) -> None:
	"""Re-express a Y-up egg in Panda's Z-up space, as Panda's loader does."""
	c, ct = _YUP_TO_ZUP, _YUP_TO_ZUP.T
	r = c[:3, :3]
	for pool in egg.pools.values():
		pool.pos = {i: tuple(r @ np.array(v)) for i, v in pool.pos.items()}
		pool.normal = {i: tuple(r @ np.array(v)) for i, v in pool.normal.items()}
	for j in egg.joints:
		j.local = c @ j.local @ ct
	for a in egg.anims:
		a.tables = {n: c @ m @ ct for n, m in a.tables.items()}


def _read_pool(node: EggNode) -> Pool:
	pool = Pool()
	for v in node.kids:
		if v.tag != "vertex":
			continue
		try:
			i = int(v.name)
		except ValueError:
			continue
		xyz = v.floats()
		pool.pos[i] = tuple((xyz + [0.0, 0.0, 0.0])[:3])
		for k in v.kids:
			if k.tag == "normal":
				f = k.floats()
				if len(f) >= 3:
					pool.normal[i] = tuple(f[:3])
			elif k.tag == "uv" and i not in pool.uv:	# the first (default) UV set
				f = k.floats()
				if len(f) >= 2:
					pool.uv[i] = (f[0], f[1])
			elif k.tag == "rgba":
				f = k.floats()
				if len(f) >= 3:
					pool.rgba[i] = tuple((f + [1.0])[:4])
	return pool


def _is_collision(g: EggNode) -> bool:
	c = g.kid("collide")
	if c is not None and "keep" not in {v.lower() for v in c.vals}:
		return True
	for o in g.all("objecttype"):
		if any(v.lower() in _COLLISION_TYPES for v in o.vals):
			return True
	return False


def _lod_out(g: EggNode) -> "float | None":
	sc = g.kid("switchcondition")
	if sc is None:
		return None
	d = sc.kid("distance")
	f = d.floats() if d is not None else []
	return f[1] if len(f) >= 2 else 0.0


def _walk(node: EggNode, egg: EggFile, joint: int) -> None:
	# Of sibling LOD groups keep the nearest (highest detail) one.
	lods = [(o, k) for k in node.kids if k.tag in ("group", "instance") for o in [_lod_out(k)] if o is not None]
	skip = {id(k) for _o, k in sorted(lods, key=lambda t: t[0])[1:]}
	if skip and "lower LOD levels dropped (the nearest is kept)" not in egg.warnings:
		egg.warnings.append("lower LOD levels dropped (the nearest is kept)")
	for k in node.kids:
		if id(k) in skip:
			continue
		if k.tag in ("group", "instance"):
			if _is_collision(k):
				continue
			if k.tag == "instance":
				egg.warnings.append(f"<Instance> {k.name}: local vertex space not applied")
			dart = k.kid("dart")
			if dart is not None and (not dart.vals or dart.vals[0].lower() not in ("0", "none")) and not egg.character:
				egg.character = k.name
			_walk(k, egg, joint)
		elif k.tag == "joint":
			j = Joint(k.name or f"joint{len(egg.joints)}", joint, egg_transform(k.kid("transform")))
			for ref in k.all("vertexref"):
				pool = ref.kid("ref")
				if pool is None or not pool.vals:
					continue
				w = float(ref.scalar("membership", "1") or 1)
				j.refs.append((pool.vals[0], [int(float(x)) for x in ref.vals], w))
			egg.joints.append(j)
			_walk(k, egg, len(egg.joints) - 1)
		elif k.tag in ("polygon", "trianglefan", "trianglestrip"):
			_read_poly(k, egg, joint)


def _read_poly(k: EggNode, egg: EggFile, joint: int) -> None:
	ref = k.kid("vertexref")
	if ref is None:
		return
	pool = ref.kid("ref")
	verts = [int(float(x)) for x in ref.vals]
	if pool is None or not pool.vals or len(verts) < 3:
		return
	tref = k.kid("tref")
	tex = tref.vals[0] if tref is not None and tref.vals else ""
	colour = None
	c = k.kid("rgba")
	if c is not None and len(c.floats()) >= 3:
		colour = tuple((c.floats() + [1.0])[:4])
	mref = k.kid("mref")
	if colour is None and mref is not None and mref.vals and mref.vals[0] in egg.materials:
		colour = egg.materials[mref.vals[0]]
	bface = k.kid("bface")
	two = bface is not None and bool(bface.vals) and bface.vals[0] not in ("0",)
	if k.tag == "trianglestrip":	# split into a fan-friendly list of triangles below
		for i in range(len(verts) - 2):
			tri = verts[i:i + 3] if i % 2 == 0 else [verts[i + 1], verts[i], verts[i + 2]]
			egg.polys.append(Poly(pool.vals[0], tri, tex, colour, two, joint))
		return
	egg.polys.append(Poly(pool.vals[0], verts, tex, colour, two, joint))


def _read_bundle(bundle: EggNode, cs: str) -> Anim:
	anim = Anim(bundle.name, 0.0, {})
	skel = next((t for t in bundle.all("table") if t.name.lower() == "<skeleton>"), None)
	if skel is None:
		return anim
	fps: list[float] = []

	def visit(table: EggNode):
		for t in table.all("table"):
			xf = t.kid("xfm$anim_s$")
			if xf is not None:
				frames, f = _xfm_s(xf, cs)
				if frames is not None:
					anim.tables[t.name] = frames
					fps.append(f)
			xf = t.kid("xfm$anim")
			if xf is not None:
				frames, f = _xfm_flat(xf, cs)
				if frames is not None:
					anim.tables[t.name] = frames
					fps.append(f)
			visit(t)

	visit(skel)
	anim.fps = max(fps) if fps else 24.0
	return anim


def _xfm_s(xf: EggNode, cs: str) -> tuple["np.ndarray | None", float]:
	fps = float(xf.scalar("fps", "24") or 24)
	order = xf.kid("char*", "order")
	order = order.vals[0].lower() if order is not None and order.vals else _STANDARD_ORDER
	chans: dict[str, list[float]] = {}
	for s in xf.all("s$anim"):
		v = s.kid("v")
		vals = (v.floats() if v is not None else []) or s.floats()
		if vals:
			chans[s.name.lower()] = vals
	if not chans:
		return None, fps
	n = max(len(v) for v in chans.values())
	frames = np.empty((n, 4, 4))
	for f in range(n):
		frames[f] = compose({c: v[f] if f < len(v) else v[-1] for c, v in chans.items()}, order, cs)
	return frames, fps


def _xfm_flat(xf: EggNode, cs: str) -> tuple["np.ndarray | None", float]:
	"""The older <Xfm$Anim>: one interleaved <V> table with a `contents` string."""
	fps = float(xf.scalar("fps", "24") or 24)
	contents = (xf.scalar("contents", "ijkabchprxyz") or "").lower()
	order = (xf.scalar("order", _STANDARD_ORDER) or _STANDARD_ORDER).lower()
	v = xf.kid("v")
	vals = v.floats() if v is not None else []
	if not contents or len(vals) < len(contents):
		return None, fps
	n = len(vals) // len(contents)
	frames = np.empty((n, 4, 4))
	for f in range(n):
		row = vals[f * len(contents):(f + 1) * len(contents)]
		frames[f] = compose(dict(zip(contents, row)), order, cs)
	return frames, fps


# -- loading files ------------------------------------------------------------

_cache: dict = {}


def _read_bytes(path: Path) -> bytes:
	data = Path(path).read_bytes()
	if Path(path).name.lower().endswith(".pz"):
		try:
			data = zlib.decompressobj(47).decompress(data)	# zlib or gzip header
		except zlib.error as exc:
			raise ValueError(f"{Path(path).name}: not a Panda .pz file ({exc})") from exc
	return data


def bam_available() -> bool:
	"""Whether .bam files can be read (the panda3d wheel, or a bam2egg program)."""
	return _panda3d() is not None or _bam2egg_exe() is not None


def _panda3d():
	try:
		import importlib

		core = importlib.import_module("panda3d.core")	# not a static import: keep it out of frozen builds
		egg = importlib.import_module("panda3d.egg")
	except ImportError:
		return None
	return core, egg


def _bam2egg_exe() -> str | None:
	exe = os.environ.get("GXBAKE_BAM2EGG")
	if exe and Path(exe).is_file():
		return exe
	return shutil.which("bam2egg")


_prc_done = False


def bam_to_egg(path: Path) -> str:
	"""EGG text for a .bam (or .bam.pz), written by Panda3D itself."""
	global _prc_done
	path = Path(path)
	p3d = _panda3d()
	if p3d is not None:
		core, egg = p3d
		if not _prc_done:
			core.loadPrcFileData("gxbake", "notify-level error\ndefault-directnotify-level error\n"
				"textures-header-only 1\nmodel-cache-dir\n")
			_prc_done = True
		bf = core.BamFile()
		if not bf.openRead(core.Filename.fromOsSpecific(str(path))):
			raise ValueError(f"{path.name}: Panda3D {core.PandaSystem.getVersionString()} cannot open this BAM "
				f"(it reads BAM {bf.getReaderVersion() if hasattr(bf, 'getReaderVersion') else '6.x'})")
		node = bf.readNode()
		if node is None:
			raise ValueError(f"{path.name}: no scene in this BAM")
		data = egg.EggData()
		egg.save_egg_data(data, node)
		data.setCoordinateSystem(core.CS_zup_right)	# Panda's internal space
		with tempfile.TemporaryDirectory() as td:
			out = Path(td) / "out.egg"
			# writeEgg (not write) gathers the <Texture>/<Material> entries the polygons point at
			if not data.writeEgg(core.Filename.fromOsSpecific(str(out))):
				raise ValueError(f"{path.name}: Panda3D could not write it out as egg")
			return out.read_text("utf-8", "replace")
	exe = _bam2egg_exe()
	if exe is None:
		raise RuntimeError("reading .bam needs Panda3D: pip install panda3d (or gxbake[panda]), "
			"or put bam2egg on PATH / set GXBAKE_BAM2EGG")
	with tempfile.TemporaryDirectory() as td:
		out = Path(td) / "out.egg"
		r = subprocess.run([exe, "-o", str(out), str(path)], capture_output=True, text=True, timeout=300)
		if r.returncode != 0 or not out.is_file():
			raise ValueError(f"bam2egg failed on {path.name}: {(r.stderr or r.stdout).strip()[-300:]}")
		return out.read_text("utf-8", "replace")


def load(path: Path) -> EggFile:
	"""Parse an .egg/.bam (optionally .pz), cached per file and modification time."""
	path = Path(path)
	try:
		mtime = path.stat().st_mtime_ns
	except OSError:
		mtime = 0
	key = str(path.resolve())
	hit = _cache.get(key)
	if hit and hit[0] == mtime:
		return hit[1]
	kind = _kind(path)
	if kind == ".bam":
		text = bam_to_egg(path)
	elif kind == ".egg":
		text = _read_bytes(path).decode("utf-8", "replace")
	else:
		raise ValueError(f"{path.name}: not an .egg or .bam file")
	egg = read_egg(text, _stem(path), path)
	if len(_cache) > 256:
		_cache.clear()
	_cache[key] = (mtime, egg)
	return egg


def is_animation_only(path: Path) -> bool:
	"""An anim file (bundle tables, no geometry) that belongs with a model next to it."""
	if not is_panda(path) or _kind(path) == ".pandactor":
		return False
	if _kind(path) == ".egg":	# cheap check without a full parse
		try:
			text = _read_bytes(path).decode("utf-8", "replace")
		except (OSError, ValueError):
			return False
		low = text.lower()
		return "<bundle>" in low and "<polygon>" not in low and "<vertexpool>" not in low
	try:
		return load(path).animation_only
	except (OSError, ValueError, RuntimeError):
		return False


# -- finding animations -------------------------------------------------------


def base_name(stem: str) -> str:
	"""A model's family name: tt_a_chr_dgm_shorts_legs_1000 → tt_a_chr_dgm_shorts_legs, panda-model → panda."""
	b = _LOD_RE.sub("", stem)
	return b or stem


def _clip_label(anim_stem: str, base: str) -> str:
	rest = anim_stem[len(base):] if anim_stem.lower().startswith(base.lower()) else anim_stem
	return rest.lstrip("_-@. ") or anim_stem


def _search_dirs(path: Path) -> list[Path]:
	"""The model's folder, plus the same folder under every other phase_N tree (Toontown)."""
	d = Path(path).resolve().parent
	out = [d]
	parts = d.parts
	for i, p in enumerate(parts):
		if re.fullmatch(r"phase_[\w.]+", p, re.I):
			root = Path(*parts[:i]) if i else Path(parts[0])
			rel = Path(*parts[i + 1:]) if i + 1 < len(parts) else Path()
			try:
				for ph in sorted(root.iterdir()):
					if ph.is_dir() and re.fullmatch(r"phase_[\w.]+", ph.name, re.I) and (ph / rel) != d and (ph / rel).is_dir():
						out.append(ph / rel)
			except OSError:
				pass
			break
	return out


def animation_files(path: Path) -> list[tuple[str, Path]]:
	"""(clip label, file) for the animation-only files that go with a model."""
	path = Path(path).resolve()
	base = base_name(_stem(path)).lower()
	found: dict[str, Path] = {}
	for d in _search_dirs(path):
		try:
			files = sorted(d.iterdir())
		except OSError:
			continue
		for f in files:
			if not f.is_file() or f == path or not is_panda(f) or _kind(f) == ".pandactor":
				continue
			st = _stem(f)
			low = st.lower()
			if not (low.startswith(base + "_") or low.startswith(base + "-") or low.startswith(base + "@")
					or low.startswith(_stem(path).lower() + "@")):
				continue
			if base_name(st).lower() == base and st[len(base):].lstrip("_-").isdigit():
				continue	# another LOD of the same model
			if not is_animation_only(f):
				continue
			label = _clip_label(st, base if not low.startswith(_stem(path).lower() + "@") else _stem(path))
			found.setdefault(label.lower(), f)
	return sorted(((k, v) for k, v in found.items()), key=lambda t: t[0])


# -- assembly -----------------------------------------------------------------


@dataclass
class Part:
	egg: EggFile
	attach: str = ""	# joint (in an earlier part) this part's root hangs from
	anims: dict[str, Anim] = field(default_factory=dict)	# clip label → animation


def _to_gold(opts: "_m.ModelOptions") -> tuple["np.ndarray", float]:
	yaw = math.radians(opts.panda_yaw)
	c, s = math.cos(yaw), math.sin(yaw)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), float(opts.panda_scale)


def _gold(m: "np.ndarray", a: "np.ndarray", k: float) -> "np.ndarray":
	"""Egg-space 4x4 → rigid GoldSrc 4x4 (turned, scaled translation)."""
	r = _sk.rigid(m)
	out = np.eye(4)
	out[:3, :3] = a @ r[:3, :3]
	out[:3, 3] = k * (a @ r[:3, 3])
	return out


def _worlds(locals_: list["np.ndarray"], parents: list[int]) -> list["np.ndarray"]:
	out: list[np.ndarray] = []
	for i, p in enumerate(parents):
		out.append(locals_[i] if p < 0 else out[p] @ locals_[i])
	return out


def import_panda(path: Path, opts: "_m.ModelOptions | None" = None) -> "_m.ImportedModel":
	"""A Panda3D .egg/.bam (with its animation files) or a .pandactor manifest → ImportedModel."""
	opts = opts or _m.ModelOptions()
	path = Path(path)
	warnings: list[str] = []
	if _kind(path) == ".pandactor":
		parts = _manifest(path, opts, warnings)
		name = path.stem
	else:
		egg = load(path)
		if egg.animation_only:
			raise ValueError(f"{path.name}: an animation file with no geometry; open its model instead")
		part = Part(egg)
		if not opts.static:
			part.anims = {(a.name or f"anim{i}").lower(): a for i, a in enumerate(egg.anims)}
			if opts.clip_files:
				for label, f in animation_files(path):
					_add_anims(part, label, f, warnings)
		parts = [part]
		name = _stem(path)
	for p in parts:
		for w in p.egg.warnings:
			if w not in warnings:
				warnings.append(w)
	return _assemble(name, parts, opts, warnings)


def _add_anims(part: Part, label: str, f: Path, warnings: list[str]) -> None:
	try:
		a = load(f)
	except (OSError, ValueError, RuntimeError) as exc:
		warnings.append(f"animation {f.name}: {exc}")
		return
	for i, anim in enumerate(a.anims):
		if not set(anim.tables) & part.egg.joint_names:
			warnings.append(f"animation {f.name} drives none of {part.egg.name}'s joints; skipped")
			continue
		lab = label if len(a.anims) == 1 else f"{label}_{anim.name or i}"
		part.anims.setdefault(lab.lower(), anim)


def _manifest(path: Path, opts: "_m.ModelOptions", warnings: list[str]) -> list[Part]:
	try:
		spec = json.loads(Path(path).read_text("utf-8"))
	except (OSError, ValueError) as exc:
		raise ValueError(f"{path.name}: {exc}") from exc
	base = path.parent
	parts: list[Part] = []
	for i, ps in enumerate(spec.get("parts") or []):
		mp = base / ps["model"]
		egg = load(mp)
		part = Part(egg, str(ps.get("attach", "") or ""))
		if not opts.static and opts.clip_files and spec.get("auto_anims", True):
			for label, f in animation_files(mp):
				_add_anims(part, label, f, warnings)
		parts.append(part)
	if not parts:
		raise ValueError(f"{path.name}: no parts")
	if not opts.static:
		for label, files in (spec.get("anims") or {}).items():
			files = [files] if isinstance(files, str) else list(files)
			for i, f in enumerate(files[:len(parts)]):
				parts[i].anims.pop(label.lower(), None)	# null: this part keeps its rest pose in the clip
				if f is not None:
					_add_anims(parts[i], label, base / f, warnings)
	return parts


def _assemble(name: str, parts: list[Part], opts: "_m.ModelOptions", warnings: list[str]) -> "_m.ImportedModel":
	a, k = _to_gold(opts)
	model = _m.ImportedModel(name, [], [], warnings)

	# One skeleton over all parts: each part's joints, its roots under its attach joint.
	jparent: list[int] = []
	jlocal: list[np.ndarray] = []
	jnames: list[str] = []
	jpart: list[int] = []
	offsets: list[int] = []
	attach_world: list[np.ndarray] = []	# per part: bind world of its attach joint (Panda Z-up units)
	for pi, part in enumerate(parts):
		offsets.append(len(jnames))
		attach = -1
		if part.attach:
			attach = next((i for i, n in enumerate(jnames) if n.lower() == part.attach.lower()), -1)
			if attach < 0:
				warnings.append(f"part {part.egg.name}: attach joint {part.attach} not found; placed at the origin")
		worlds_so_far = _worlds(jlocal, jparent) if jlocal else []
		attach_world.append(worlds_so_far[attach] if attach >= 0 else np.eye(4))
		for j in part.egg.joints:
			jparent.append(offsets[pi] + j.parent if j.parent >= 0 else attach)
			jlocal.append(j.local)
			jnames.append(j.name)
			jpart.append(pi)
	skeletal = bool(jnames) and not opts.static
	if skeletal and len(jnames) + 1 > _sk.MAXSTUDIOBONES:
		warnings.append(f"{len(jnames)} joints exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static")
		skeletal = False
	if opts.static and jnames:
		warnings.append("baked static: skeleton and animations ignored")

	bind_world = _worlds(jlocal, jparent) if jnames else []
	# A part's joints are expressed in its own space; the attach point carries it into the parent part.
	gold_world: list[np.ndarray] = []
	if skeletal:
		# studio bone 0 is the character root (identity): unassigned vertices ride on it
		root_name = (parts[0].egg.character or "character")[:31]
		model.bones = [_m.Bone(root_name, -1, np.zeros(6))]
		gold_world = [np.eye(4)]
		used = {root_name.lower()}
		for i, n in enumerate(jnames):
			w = _gold(bind_world[i], a, k)
			p = jparent[i] + 1 if jparent[i] >= 0 else 0
			local = _sk.rigid_inverse(gold_world[p]) @ w
			model.bones.append(_m.Bone(_m._unique(n[:31] or f"joint{i}", used), p, _sk.pose_from_matrix(local)))
			gold_world.append(w)
		if any(abs(np.linalg.det(m[:3, :3]) - 1.0) > 0.05 for m in bind_world):
			warnings.append("scaled joints are baked rigid (studio bones carry no scale)")

	# Geometry: textured polygons by texture, untextured ones on one colour-swatch skin.
	groups: dict[tuple, dict] = {}
	swatches: dict[tuple, int] = {}
	for pi, part in enumerate(parts):
		egg = part.egg
		place = attach_world[pi]	# part space → model egg space at bind
		member: dict[tuple, tuple[float, int]] = {}
		if skeletal:
			for ji, j in enumerate(egg.joints):
				for pool, ids, w in j.refs:
					for v in ids:
						key = (pool, v)
						if key not in member or w > member[key][0]:
							member[key] = (w, offsets[pi] + ji)
		blended = False
		for poly in egg.polys:
			pool = egg.pools.get(poly.pool)
			if pool is None:
				continue
			tex = egg.textures.get(poly.texture) if poly.texture else None
			gkey = ("tex", poly.texture, pi) if tex is not None else ("col",)
			g = groups.setdefault(gkey, {"tex": tex, "egg": egg, "pos": [], "nrm": [], "uv": [], "bone": [], "tris": [], "index": {}})
			idx = []
			for v in poly.verts:
				if v not in pool.pos:
					break
				colour = None
				if tex is None:
					c = pool.rgba.get(v) or poly.colour or (1.0, 1.0, 1.0, 1.0)
					colour = tuple(int(round(max(0.0, min(1.0, x)) * 255)) for x in c[:3])
					swatches.setdefault(colour, len(swatches))
				vkey = (pi, poly.pool, v, colour)
				if vkey not in g["index"]:
					p = np.array(pool.pos[v], float)
					n = np.array(pool.normal[v], float) if v in pool.normal else np.full(3, np.nan)
					p = place[:3, :3] @ p + place[:3, 3]
					n = place[:3, :3] @ n
					bone = 0
					if skeletal:
						m = member.get((poly.pool, v))
						if m is not None:
							bone = m[1] + 1
							if m[0] < 0.999:
								blended = True
						elif poly.joint >= 0:
							bone = offsets[pi] + poly.joint + 1
					g["index"][vkey] = len(g["pos"])
					g["pos"].append(p)
					g["nrm"].append(n)
					uv = pool.uv.get(v, (0.0, 0.0)) if tex is not None else ("swatch", swatches[colour])
					g["uv"].append(uv)
					g["bone"].append(bone)
				idx.append(g["index"][vkey])
			else:
				for t in range(1, len(idx) - 1):	# fan: egg polygons are convex
					g["tris"].append((idx[0], idx[t], idx[t + 1]))
					if poly.two_sided:
						g["tris"].append((idx[0], idx[t + 1], idx[t], "back"))
		if blended:
			msg = "blended skin weights are snapped to each vertex's heaviest joint (studio skinning is rigid)"
			if msg not in warnings:
				warnings.append(msg)

	swatch_img, swatch_uv = _swatch_skin(swatches)
	for gkey, g in groups.items():
		if not g["tris"]:
			continue
		pos = np.array(g["pos"])
		nrm = np.array(g["nrm"])
		front = np.array([t[:3] for t in g["tris"] if len(t) == 3], np.int64).reshape(-1, 3)
		back = np.array([t[:3] for t in g["tris"] if len(t) == 4], np.int64).reshape(-1, 3)
		missing = np.isnan(nrm).any(1)
		if missing.any():
			fn = _face_normals(pos, front if len(front) else back)
			nrm[missing] = fn[missing]
		uv = np.array([swatch_uv[u[1]] if isinstance(u[0], str) else (u[0], 1.0 - u[1]) for u in g["uv"]], np.float32)
		bones = np.array(g["bone"], np.int32)
		n0 = len(pos)
		if len(back):	# studio models are single-sided: back faces get their own vertices
			pos = np.concatenate([pos, pos])
			nrm = np.concatenate([nrm, -nrm])
			uv = np.concatenate([uv, uv])
			bones = np.concatenate([bones, bones])
			back = back + n0
		tris = np.concatenate([front, back]).astype(np.uint32)
		world = k * (pos @ a.T)
		wn = nrm @ a.T
		if skeletal:
			bw = np.array(gold_world)[bones]
			local = np.einsum("nji,nj->ni", bw[:, :3, :3], world - bw[:, :3, 3])
			ln = np.einsum("nji,nj->ni", bw[:, :3, :3], wn)
		else:
			local, ln = world, wn
		lens = np.linalg.norm(ln, axis=1, keepdims=True)
		ln = np.where(lens > 1e-8, ln / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		if gkey[0] == "col":
			mat = _m.Material(_COLOURS_MATERIAL, swatch_img, False, "(polygon and vertex colours)")
		else:
			mat = _texture_material(gkey[1], g["tex"], g["egg"], opts, warnings)
		model.materials.append(mat)
		model.meshes.append(_m.Submesh(len(model.materials) - 1, local.astype(np.float32), ln.astype(np.float32), uv,
			tris, bones, world.astype(np.float32)))
	if not model.meshes:
		raise ValueError(f"{name}: no polygons")

	if skeletal:
		labels: list[str] = []
		for part in parts:
			for lab in part.anims:
				if lab not in labels:
					labels.append(lab)
		labels.sort(key=lambda s: (0 if any(w in s for w in ("neutral", "idle", "stand")) else 1, s))
		used_seq: set[str] = set()
		for lab in labels:
			frames, loop = _sample(lab, parts, jparent, jlocal, jpart, offsets, a, k, opts.anim_fps)
			if frames is None:
				continue
			model.sequences.append(_m.Sequence(_m._unique(_m._seq_name(lab), used_seq), float(opts.anim_fps), frames, loop))
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	return model


def _face_normals(pos: "np.ndarray", tris: "np.ndarray") -> "np.ndarray":
	out = np.zeros_like(pos)
	if len(tris):
		fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
		for c in range(3):
			np.add.at(out, tris[:, c], fn)
	lens = np.linalg.norm(out, axis=1, keepdims=True)
	return np.where(lens > 1e-12, out / np.maximum(lens, 1e-12), [0.0, 0.0, 1.0])


def _swatch_skin(swatches: dict[tuple, int]) -> tuple["np.ndarray", dict[int, tuple[float, float]]]:
	"""Untextured colours as 4x4 cells of one power-of-two skin; each colour's UV is its cell centre."""
	n = max(1, len(swatches))
	cols = 1 << max(1, math.ceil(math.log2(math.ceil(math.sqrt(n)))))
	rows = max(2, 1 << math.ceil(math.log2(math.ceil(n / cols))))
	cell = 4
	img = np.zeros((rows * cell, cols * cell, 4), np.uint8)
	img[..., 3] = 255
	uv: dict[int, tuple[float, float]] = {}
	for colour, i in swatches.items():
		r, c = divmod(i, cols)
		img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell, :3] = colour
		uv[i] = ((c + 0.5) / cols, (r + 0.5) / rows)
	return img, uv


def _texture_material(label: str, tex: Texture, egg: EggFile, opts: "_m.ModelOptions",
		warnings: list[str]) -> "_m.Material":
	rgba = None
	path = find_texture(tex.file, egg.path)
	if path is not None:
		rgba = _m._open_rgba(path)
	if rgba is None:
		warnings.append(f"texture {tex.file} not found; using grey")
		rgba = np.full((8, 8, 4), (200, 200, 200, 255), np.uint8)
	elif tex.alpha_file:
		ap = find_texture(tex.alpha_file, egg.path)
		alpha = _m._open_rgba(ap) if ap is not None else None
		if alpha is None:
			warnings.append(f"alpha file {tex.alpha_file} not found; {Path(tex.file).name} stays opaque")
		else:
			rgba = rgba.copy()
			rgba[..., 3] = _alpha_plane(alpha, tex.alpha_channel, rgba.shape[:2])
	rgba = _m._fit_skin(rgba, opts.skin_max_edge)
	masked = bool((rgba[..., 3] < opts.alpha_threshold).any())
	name = Path(tex.file.replace("\\", "/")).stem or label
	return _m.Material(name[:63], rgba, masked, str(path or tex.file))


def _alpha_plane(alpha: "np.ndarray", channel: int, shape: tuple[int, int]) -> "np.ndarray":
	from PIL import Image

	if 1 <= channel <= 4:
		plane = alpha[..., channel - 1]
	else:	# Panda's default: the alpha image's grey level
		plane = (alpha[..., :3].astype(np.uint32) @ np.array([299, 587, 114]) // 1000).astype(np.uint8)
	if plane.shape != shape:
		plane = np.asarray(Image.fromarray(plane).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
	return plane


def find_texture(ref: str, model_path: Path) -> Path | None:
	"""A texture path from an egg, looked up like Panda's model-path would: next to the model,
	then under its ancestors (Toontown eggs say phase_3/maps/x.jpg relative to the install root),
	dropping leading folders of absolute paths, case-insensitively, then any image with that stem."""
	from .images import IMAGE_EXTENSIONS

	if not ref:
		return None
	ref = ref.replace("\\", "/")
	comps = [c for c in ref.split("/") if c not in ("", ".")]
	tails = []
	for i in range(len(comps)):
		tail = comps[i:]
		if ".." in tail:
			continue
		tails.append(tail)
	start = Path(model_path).resolve().parent
	dirs = [start, *list(start.parents)[:6]]
	for tail in tails:
		for d in dirs:
			p = _ci_join(d, tail)
			if p is not None:
				return p
	# the same name in another image format (.rgb converted to .png, etc.)
	stem = Path(comps[-1]).stem.lower() if comps else ""
	for tail in tails:
		for d in dirs:
			folder = _ci_join(d, tail[:-1]) if len(tail) > 1 else d
			if folder is None:
				continue
			try:
				for c in sorted(folder.iterdir()):
					if c.stem.lower() == stem and c.suffix.lower() in IMAGE_EXTENSIONS and c.is_file():
						return c
			except OSError:
				continue
	return None


def _ci_join(base: Path, comps: list[str]) -> Path | None:
	p = base.joinpath(*comps) if comps else base
	try:
		if p.is_file():
			return p
	except OSError:
		return None
	cur = base
	for i, c in enumerate(comps):
		try:
			nxt = cur / c
			if not nxt.exists():
				low = c.lower()
				nxt = next((e for e in cur.iterdir() if e.name.lower() == low), None)
				if nxt is None:
					return None
		except OSError:
			return None
		cur = nxt
	return cur if cur.is_file() else None


def _sample(label: str, parts: list[Part], jparent: list[int], jlocal: list["np.ndarray"], jpart: list[int],
		offsets: list[int], a: "np.ndarray", k: float, fps: int) -> tuple["np.ndarray | None", bool]:
	"""Bake one clip (from every part that has it) into (frames, bones, 6) studio poses."""
	tracks: dict[int, np.ndarray] = {}
	duration = 0.0
	src_fps: dict[int, float] = {}
	for pi, part in enumerate(parts):
		anim = part.anims.get(label)
		if anim is None:
			continue
		names = {j.name: offsets[pi] + i for i, j in enumerate(part.egg.joints)}
		for jn, mats in anim.tables.items():
			if jn in names:
				tracks[names[jn]] = mats
				src_fps[names[jn]] = anim.fps or 24.0
		if anim.num_frames:
			duration = max(duration, (anim.num_frames - 1) / (anim.fps or 24.0))
	if not tracks:
		return None, True
	nframes = max(1, int(round(duration * fps)) + 1)
	nj = len(jparent)
	frames = np.zeros((nframes, nj + 1, 6))
	for f in range(nframes):
		t = f / fps
		locals_ = []
		for j in range(nj):
			if j in tracks:
				locals_.append(_frame(tracks[j], t * src_fps[j]))
			else:
				locals_.append(jlocal[j])
		worlds = [np.eye(4)] + [_gold(w, a, k) for w in _worlds(locals_, jparent)]
		for b in range(1, nj + 1):
			p = jparent[b - 1] + 1 if jparent[b - 1] >= 0 else 0
			frames[f, b] = _sk.pose_from_matrix(_sk.rigid_inverse(worlds[p]) @ worlds[b])
	return frames, not any(w in label for w in _ONESHOT_WORDS)


def _frame(mats: "np.ndarray", src: float) -> "np.ndarray":
	"""A joint's local matrix at a fractional source frame (blended between the two nearest)."""
	last = len(mats) - 1
	src = min(max(src, 0.0), last)
	i0 = int(math.floor(src))
	i1 = min(last, i0 + 1)
	t = src - i0
	if t < 1e-6 or i0 == i1:
		return mats[i0]
	r0, r1 = _sk.rigid(mats[i0]), _sk.rigid(mats[i1])
	q = _sk._slerp(_quat(r0[:3, :3]), _quat(r1[:3, :3]), t)
	out = np.eye(4)
	out[:3, :3] = _sk.quat_matrix(q)
	out[:3, 3] = r0[:3, 3] + (r1[:3, 3] - r0[:3, 3]) * t
	return out


def _quat(r: "np.ndarray") -> "np.ndarray":
	"""3x3 rotation → (x, y, z, w)."""
	tr = r[0, 0] + r[1, 1] + r[2, 2]
	if tr > 0:
		s = math.sqrt(tr + 1.0) * 2
		q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
	elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
		s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
		q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
	elif r[1, 1] > r[2, 2]:
		s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
		q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
	else:
		s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
		q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
	q = np.array(q)
	return q / (np.linalg.norm(q) or 1.0)
