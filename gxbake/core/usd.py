"""USD scenes (.usda, .usd, .usdc, .usdz) through FreeUSD → GoldSrc studio MDL.

FreeUSD (github.com/gopexllc/freeusd) is a clean-room OpenUSD implementation
with Python bindings. GxBake uses it to open and compose the stage
(sublayers, references, payloads), walk the prims, and read meshes,
UsdPreviewSurface materials, UsdSkel skeletons and animations. The result is
the same ImportedModel that models.py builds from Assimp scenes, so a USD file
ends up as the familiar models/<name>.mdl, gc_studio/<name>.mdl and
gc_studio/<name>t.mdl.

What is read:

- Every visible Mesh with render or default purpose, triangulated as fans.
  Normals come from primvars:normals or normals (else smooth normals are
  made); UVs from primvars:st (or st0, UVMap, uv, map1).
- Material bindings on the mesh, its ancestors, and materialBind GeomSubsets.
  A UsdPreviewSurface's diffuseColor texture becomes the skin, otherwise its
  diffuseColor or the mesh's displayColor.
- UsdSkel: each Skeleton's joints become studio bones, skinned points follow
  their heaviest joint (studio skinning is rigid), and the bound
  SkelAnimation is sampled into one sequence. Meshes moved by time-sampled
  xformOps get a bone of their own, so rigid animation carries over too.
  <name>@<clip>.usda files next to the model add more sequences.
- Stage metadata: upAxis decides the axis swap and metersPerUnit the scale
  (one GoldSrc unit is an inch); ModelOptions.scale multiplies on top.

FreeUSD gaps this module works around (FreeUSD 0.1.0, see docs/usd.md):

- USDA attributes with a metadata block (interpolation, elementSize) lose
  their values, and `uniform` / `custom` qualifiers end up inside attribute
  names. ASCII layers are rewritten into a temporary folder without them
  before FreeUSD reads them; the stripped interpolation and elementSize are
  kept here and used when the primvars are read.
- half-precision types (half3[], quath) don't parse; they're read as float.
- Arrays have no Python accessors, so their text form is parsed.
- Xformable.compute_local_to_world_transform multiplies in the wrong order;
  world matrices are composed here from local ones.
- Relationship targets inside a referenced layer keep that layer's paths
  (/Prop/Mat under /World/P0); they're re-rooted under the referencing prim.
- Prim type names are dropped, so meshes, skeletons and animations are found
  by the attributes they carry.
- Variant sets in the standard `variantSet "x" = { "a" { ... } }` form don't
  parse (FreeUSD reads its own variantSets = {...} metadata form only); flatten
  the variant selection before converting.
- Binary crate files (.usdc, and .usdz packages holding one) only open when
  they embed a USDA section. Convert them first: usdcat -o model.usda model.usdc
"""

from __future__ import annotations

import math
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as _m
from . import skeleton as _sk

USD_EXTENSIONS = _m.USD_EXTENSIONS

# Commit of gopexllc/freeusd that this importer was written and tested against.
FREEUSD_COMMIT = "f5bcd397a4d09941c1bdfeee28aa4efe93ed8db1"
FREEUSD_INSTALL = "pip install 'freeusd @ git+https://github.com/gopexllc/freeusd'  (or gxbake[usd])"

_CRATE_MAGIC = b"PXR-USDC"
_UV_NAMES = ("primvars:st", "primvars:st0", "primvars:UVMap", "primvars:uv", "primvars:UVW", "primvars:map1",
	"primvars:texcoord0", "primvars:TEXCOORD_0")
_INCH = 0.0254	# metres per GoldSrc unit


def available() -> bool:
	try:
		import freeusd.usd	# noqa: F401
	except ImportError:
		return False
	return True


def _freeusd():
	try:
		import freeusd.sdf as sdf
		import freeusd.tf as tf
		import freeusd.usd as usd
		import freeusd.usdGeom as geom
		import freeusd.usdShade as shade
		import freeusd.usdSkel as skel
	except ImportError as exc:
		raise RuntimeError(f"FreeUSD is not installed: {FREEUSD_INSTALL}") from exc
	return usd, sdf, tf, geom, shade, skel


def is_usd(path: Path) -> bool:
	return Path(path).suffix.lower() in USD_EXTENSIONS


# -- USDA rewrite for FreeUSD --------------------------------------------------

_DEF_RE = re.compile(r'^\s*(?:def|over|class)\b[^"]*"([^"]+)"')
_ATTR_RE = re.compile(r'^(\s*)((?:(?:uniform|custom|varying)\s+)+)?([A-Za-z_][\w\[\]]*)(\s+)([\w:.]+)(\s*=\s*)')
_HALF_RE = re.compile(r'\b(half|texCoord2h|texCoord3h|color3h|color4h|normal3h|point3h|vector3h|quath)(\d?)(\[\])?(?=\s)')
_HALF_MAP = {"half": "float", "texCoord2h": "texCoord2f", "texCoord3h": "texCoord3f", "color3h": "color3f",
	"color4h": "color4f", "normal3h": "normal3f", "point3h": "point3f", "vector3h": "vector3f", "quath": "quatf"}
_META_RE = re.compile(r'\b(interpolation|elementSize)\s*=\s*("?)([\w]+)\2')
_ASSET_RE = re.compile(r"@([^@]+)@")


def _scan_value(text: str, i: int) -> int:
	"""Index just past one USDA value starting at text[i] (brackets, strings and assets balanced)."""
	n = len(text)
	while i < n and text[i] in " \t":
		i += 1
	depth = 0
	while i < n:
		c = text[i]
		if c in "\"'":
			j = text.find(c, i + 1)
			i = n if j < 0 else j + 1
		elif c == "@":
			j = text.find("@", i + 1)
			i = n if j < 0 else j + 1
		elif c in "([{":
			depth += 1
			i += 1
		elif c in ")]}":
			if depth == 0:
				return i
			depth -= 1
			i += 1
			if depth == 0:
				return i
		elif depth == 0 and c in " \t\n":
			return i
		else:
			i += 1
	return i


def _matching_paren(text: str, i: int) -> int:
	"""text[i] is '('; index past its matching ')'."""
	depth = 0
	n = len(text)
	while i < n:
		c = text[i]
		if c == '"':
			j = text.find('"', i + 1)
			i = n if j < 0 else j + 1
			continue
		if c == "(":
			depth += 1
		elif c == ")":
			depth -= 1
			if depth == 0:
				return i + 1
		i += 1
	return n


def rewrite_usda(text: str) -> tuple[str, dict[tuple[str, str], dict[str, str]], list[str]]:
	"""USDA text FreeUSD can read, the attribute metadata stripped from it, and the layer's prim paths.

	The metadata map is keyed by (prim path in this layer, attribute name).
	"""
	out: list[str] = []
	meta: dict[tuple[str, str], dict[str, str]] = {}
	prims: list[str] = []
	stack: list[str | None] = []	# one entry per open '{': a prim name or None
	pending: str | None = None
	pos = 0
	n = len(text)
	while pos < n:
		end = text.find("\n", pos)
		end = n if end < 0 else end + 1
		line = text[pos:end]
		m = _ATTR_RE.match(line)
		if m and not line.lstrip().startswith(("rel ", "prepend ", "append ", "delete ", "add ", "reorder ")):
			# Attribute statement: drop qualifiers, map half types, cut a trailing metadata block.
			type_name = m.group(3)
			hm = _HALF_RE.match(type_name + " ")
			if hm:
				type_name = _HALF_MAP[hm.group(1)] + (hm.group(2) or "") + (hm.group(3) or "")
			head = f"{m.group(1)}{type_name}{m.group(4)}{m.group(5)}{m.group(6)}"
			vstart = pos + m.end()
			if text.startswith("{", vstart) or m.group(5).endswith(".timeSamples"):
				# timeSamples dictionary: keep as is (braces tracked below)
				out.append(head + line[m.end():])
				_track_braces(line[m.end():], stack, None)
				pos = end
				continue
			vend = _scan_value(text, vstart)
			value = text[vstart:vend]
			j = vend
			while j < n and text[j] in " \t":
				j += 1
			if j < n and text[j] == "(":
				mend = _matching_paren(text, j)
				block = text[j:mend]
				found = {k: v for k, _q, v in _META_RE.findall(block)}
				if found:
					path = "/" + "/".join(s for s in stack if s)
					meta[(path, m.group(5))] = found
				j = mend
			nl = text.find("\n", j)
			nl = n if nl < 0 else nl + 1
			out.append(head + value + text[j:nl].rstrip(" \t\r\n") + "\n")
			pos = nl
			continue
		dm = _DEF_RE.match(line)
		if dm:
			pending = dm.group(1)
		opened_name = pending if "{" in _strip_strings(line) else None
		if _track_braces(line, stack, opened_name):
			if opened_name:
				prims.append("/" + "/".join(x for x in stack if x))
			pending = None
		out.append(line)
		pos = end
	return "".join(out), meta, prims


def _strip_strings(s: str) -> str:
	return re.sub(r'"[^"]*"|@[^@]*@', "", s)


def _track_braces(line: str, stack: list, name: str | None) -> bool:
	"""Push/pop the brace stack for one line; returns True if a brace opened."""
	opened = False
	for c in _strip_strings(line):
		if c == "{":
			stack.append(name if not opened else None)
			opened = True
		elif c == "}" and stack:
			stack.pop()
	return opened


@dataclass
class _Layers:
	root: Path	# the file FreeUSD opens
	dirs: list[Path]	# folders textures may be relative to (original layer folders first)
	meta: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
	crate: bool = False
	# rewritten layer (posix path) → (its prim paths, defaultPrim, its attribute metadata)
	layer_prims: dict[str, tuple[list[str], str, dict]] = field(default_factory=dict)


def _is_crate(path: Path) -> bool:
	try:
		with path.open("rb") as f:
			return f.read(8) == _CRATE_MAGIC
	except OSError:
		return False


def _prepare(path: Path, tmp: Path) -> _Layers:
	"""Unpack a .usdz and rewrite the ASCII layer graph into tmp."""
	path = Path(path)
	src = path
	dirs = [path.parent]
	if path.suffix.lower() == ".usdz" or zipfile.is_zipfile(path) and path.suffix.lower() != ".usda":
		pkg = tmp / "usdz"
		with zipfile.ZipFile(path) as z:
			names = [i.filename for i in z.infolist() if not i.is_dir()]
			layers = [nm for nm in names if Path(nm).suffix.lower() in (".usd", ".usda", ".usdc")]
			if not layers:
				raise ValueError(f"{path.name}: no USD layer inside the package")
			for nm in names:
				dest = (pkg / nm).resolve()
				if not str(dest).startswith(str(pkg.resolve())):
					continue	# no writing outside the scratch folder
				dest.parent.mkdir(parents=True, exist_ok=True)
				dest.write_bytes(z.read(nm))
		src = pkg / layers[0]	# the package's first layer is its root
		dirs = [src.parent, pkg]
	if _is_crate(src):
		return _Layers(src, dirs, crate=True)
	mapping: dict[Path, Path] = {}
	meta: dict[tuple[str, str], dict[str, str]] = {}
	layer_prims: dict[str, tuple[list[str], str, dict]] = {}
	out = tmp / "layers"
	out.mkdir(parents=True, exist_ok=True)

	def visit(layer: Path) -> Path:
		layer = layer.resolve()
		if layer in mapping:
			return mapping[layer]
		dest = out / f"L{len(mapping)}_{layer.stem}.usda"
		mapping[layer] = dest
		if layer.parent not in dirs:
			dirs.append(layer.parent)
		text = layer.read_text("utf-8", errors="replace")
		if _is_crate(layer):
			return layer	# FreeUSD's own crate path
		dp = re.search(r'defaultPrim\s*=\s*"([^"]+)"', text[:4096])
		text, found, prims = rewrite_usda(text)
		layer_prims[dest.as_posix()] = (prims, dp.group(1) if dp else "", found)
		if layer == src.resolve():
			meta.update(found)

		def sub(mm: re.Match) -> str:
			ref = mm.group(1)
			target = (layer.parent / ref)
			if Path(ref).suffix.lower() in (".usd", ".usda", ".usdc") and target.is_file():
				return "@" + visit(target).as_posix() + "@"
			return mm.group(0)

		text = _ASSET_RE.sub(sub, text)
		dest.write_text(text, "utf-8")
		return dest

	root = visit(src)
	return _Layers(root, dirs, meta, layer_prims=layer_prims)


# -- reading values --------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan")


def _nums(value) -> "np.ndarray":
	if value is None:
		return np.zeros(0)
	s = str(value)
	if not s or s in ("None", "true", "false"):
		return np.zeros(0)
	return np.array([float(x) for x in _NUM_RE.findall(s)], np.float64)


class _Stage:
	"""The FreeUSD stage plus what GxBake needs on top of it."""

	def __init__(self, layers: _Layers):
		self.usd, self.sdf, self.tf, self.geom, self.shade, self.skel = _freeusd()
		try:
			self.stage = self.usd.Stage.open_from_root_file(str(layers.root))
		except ValueError as exc:
			if layers.crate:
				raise ValueError("binary USD (usdc) isn't decoded by FreeUSD yet; save the file as USDA "
					"(usdcat -o model.usda model.usdc) and convert that") from exc
			raise ValueError(f"FreeUSD could not read the stage: {exc}") from exc
		self.layers = layers
		self.prims: list = []
		self.stage.traverse_preorder(self.prims.append)
		self.by_path = {self.text(p.path()): p for p in self.prims}
		self._expand_references()
		self._local: dict[tuple[str, float], np.ndarray] = {}
		start, end = float(self.stage.start_time_code() or 0), float(self.stage.end_time_code() or 0)
		self.start, self.end = start, end
		self.tcps = float(self.stage.time_codes_per_second() or 0) or 24.0

	def _expand_references(self) -> None:
		"""Add prims FreeUSD composes but doesn't traverse: those below the first level of a reference.

		The referenced layer's own prim list says which paths to ask for.
		"""
		queue = [p for p in self.prims if p.has_references()]
		seen: set[str] = set()
		while queue:
			owner = queue.pop(0)
			opath = self.text(owner.path())
			if opath in seen:
				continue
			seen.add(opath)
			for ref in owner.get_references():
				asset = str(ref.asset_path)
				info = self.layers.layer_prims.get(Path(asset).as_posix()) if asset else None
				if not info:
					continue
				prims, default, lmeta = info
				rp = self.text(ref.prim_path) if ref.prim_path is not None else ""
				root = rp if rp and rp != "/" else ("/" + default if default else (prims[0] if prims else ""))
				if not root:
					continue
				for q in prims:
					if q != root and not q.startswith(root + "/"):
						continue
					cand = opath + q[len(root):]
					for (lp, attr), v in lmeta.items():
						if lp == q:
							self.layers.meta.setdefault((cand, attr), v)
					if cand in self.by_path:
						continue
					prim = self.stage.prim_at(self.sdf.Path.from_string(cand))
					if prim.is_valid():
						self.by_path[cand] = prim
						self.prims.append(prim)
						if prim.has_references():
							queue.append(prim)
		self.prims.sort(key=lambda p: self.text(p.path()))

	@staticmethod
	def text(path) -> str:
		return path.text() if hasattr(path, "text") else str(path)

	def T(self, name: str):
		return self.tf.Token(name)

	def attr(self, prim, name: str, time: float | None = None):
		if not prim.has_attribute(self.T(name)):
			return None
		v = prim.get_attribute(self.T(name), self.start if time is None else time)
		return None if v.is_empty() else v

	def nums(self, prim, name: str, time: float | None = None) -> "np.ndarray | None":
		v = self.attr(prim, name, time)
		return None if v is None else _nums(v)

	def token(self, prim, name: str) -> str | None:
		v = self.attr(prim, name)
		if v is None:
			return None
		t = v.as_token()
		s = (t.text() if hasattr(t, "text") else str(t)) if t is not None else (v.as_string() or str(v))
		return s.strip().strip('"')

	def tokens(self, prim, name: str) -> list[str]:
		v = self.attr(prim, name)
		if v is None:
			return []
		arr = v.as_token_array()
		if arr is not None:
			return [str(x) for x in arr]
		return re.findall(r'"([^"]*)"', str(v)) or [x.strip() for x in str(v).strip("[]").split(",") if x.strip()]

	def targets(self, prim, rel: str) -> list[str]:
		if not prim.has_relationship(self.T(rel)):
			return []
		owner = self.text(prim.path())
		return [self.remap(owner, self.text(p)) for p in prim.get_relationship_targets(self.T(rel))]

	def remap(self, owner: str, target: str) -> str:
		"""A relationship target authored inside a referenced layer, moved under the referencing prim.

		FreeUSD composes the reference but leaves its targets in the referenced
		layer's namespace (/Prop/Mat instead of /World/P0/Mat). Re-root the target
		under each ancestor of the owner until it names a prim that exists.
		"""
		if target in self.by_path or target.count("/") < 2:
			return target
		tail = "/" + target.split("/", 2)[2]
		p: str | None = owner
		while p and p != "/":
			cand = p + tail
			if cand in self.by_path:
				return cand
			p = self.parent_path(p)
		return target

	def meta(self, prim, name: str) -> dict[str, str]:
		return self.layers.meta.get((self.text(prim.path()), name), {})

	def parent_path(self, path: str) -> str | None:
		if path in ("/", ""):
			return None
		head = path.rsplit("/", 1)[0]
		return head or "/"

	def local(self, path: str, time: float) -> "np.ndarray":
		"""Column-vector local transform of a prim."""
		key = (path, time)
		if key not in self._local:
			prim = self.by_path.get(path)
			m = np.eye(4)
			if prim is not None and any(a.startswith("xformOp") for a in prim.list_attribute_names()):
				row = np.array(self.geom.Xformable(prim).compute_local_transform(time).as_list(), np.float64).reshape(4, 4)
				m = row.T
			self._local[key] = m
		return self._local[key]

	def world(self, path: str, time: float | None = None) -> "np.ndarray":
		"""Local-to-world, composed child-last (FreeUSD's own composes in the wrong order)."""
		t = self.start if time is None else time
		m = np.eye(4)
		p: str | None = path
		while p and p != "/":
			m = self.local(p, t) @ m
			p = self.parent_path(p)
		return m

	def xform_times(self, path: str) -> list[float]:
		"""Time samples on the xformOps of a prim and its ancestors."""
		times: set[float] = set()
		p: str | None = path
		while p and p != "/":
			prim = self.by_path.get(p)
			if prim is not None:
				for a in prim.list_attribute_names():
					if a.startswith("xformOp:"):
						times.update(float(x) for x in prim.list_attribute_sample_times(self.T(a)))
			p = self.parent_path(p)
		return sorted(times)

	def shown(self, prim) -> bool:
		if hasattr(prim, "is_active") and not prim.is_active():
			return False
		try:
			img = self.geom.Imageable(prim)
			if not img.compute_visibility(self.start):
				return False
			return str(img.compute_purpose(self.start)) in ("default", "render", "")
		except Exception:	# noqa: BLE001 - schema helpers raise on odd prims; treat as shown
			return True

	def timeline(self) -> tuple[float, float]:
		return self.start, self.end


# -- scene pieces -------------------------------------------------------------------


@dataclass
class _Skel:
	path: str
	joints: list[str]	# joint paths as authored
	parents: list[int]
	rest_local: list["np.ndarray"]	# joint-to-parent, column convention
	bind_world: list["np.ndarray"]	# skeleton-space bind transforms
	anim: str | None = None	# SkelAnimation prim path
	first_bone: int = 0	# studio bone of joint 0


def _matrices(vals: "np.ndarray | None") -> list["np.ndarray"]:
	if vals is None or len(vals) % 16:
		return []
	return [r.reshape(4, 4).T for r in vals.reshape(-1, 16)]	# USD rows → column convention


def _rest_space(sk: _Skel, locals_: list["np.ndarray"]) -> list["np.ndarray"]:
	out: list[np.ndarray] = []
	for j, p in enumerate(sk.parents):
		out.append(locals_[j] if p < 0 else out[p] @ locals_[j])
	return out


def _read_skeleton(st: _Stage, prim) -> _Skel | None:
	joints = st.tokens(prim, "joints")
	if not joints:
		return None
	path = st.text(prim.path())
	parents = []
	index = {j: i for i, j in enumerate(joints)}
	for j in joints:
		head = j.rsplit("/", 1)[0] if "/" in j else None
		parents.append(index.get(head, -1) if head else -1)
	bind = _matrices(st.nums(prim, "bindTransforms"))
	rest = _matrices(st.nums(prim, "restTransforms"))
	if len(bind) != len(joints):
		bind = []
	if len(rest) != len(joints):
		rest = []
	if not rest and bind:
		rest = [bind[j] if parents[j] < 0 else np.linalg.inv(bind[parents[j]]) @ bind[j] for j in range(len(joints))]
	if not rest:
		rest = [np.eye(4) for _ in joints]
	sk = _Skel(path, joints, parents, rest, [])
	sk.bind_world = bind or _rest_space(sk, rest)
	# The animation: on the skeleton, else inherited from an ancestor (usually the SkelRoot).
	p: str | None = path
	while p and p != "/":
		pr = st.by_path.get(p)
		if pr is not None:
			t = st.targets(pr, "skel:animationSource")
			if t:
				sk.anim = t[0]
				break
		p = st.parent_path(p)
	return sk


def _anim_locals(st: _Stage, sk: _Skel, anim_path: str | None, time: float) -> list["np.ndarray"]:
	"""Joint-local matrices at `time`: animated joints from the SkelAnimation, the rest from restTransforms."""
	locals_ = list(sk.rest_local)
	prim = st.by_path.get(anim_path) if anim_path else None
	if prim is None:
		return locals_
	names = st.tokens(prim, "joints")
	tr = st.nums(prim, "translations", time)
	ro = st.nums(prim, "rotations", time)
	sc = st.nums(prim, "scales", time)
	index = {j: i for i, j in enumerate(sk.joints)}
	short = {j.rsplit("/", 1)[-1]: i for i, j in enumerate(sk.joints)}
	for k, name in enumerate(names):
		j = index.get(name, short.get(name.rsplit("/", 1)[-1]))
		if j is None:
			continue
		t, r, s = _sk.decompose(sk.rest_local[j])
		if tr is not None and len(tr) >= 3 * (k + 1):
			t = tr[3 * k: 3 * k + 3]
		if ro is not None and len(ro) >= 4 * (k + 1):
			w, x, y, z = ro[4 * k: 4 * k + 4]
			r = _sk.quat_matrix(np.array([x, y, z, w]))
		if sc is not None and len(sc) >= 3 * (k + 1):
			s = sc[3 * k: 3 * k + 3]
		m = np.eye(4)
		m[:3, :3] = r * s
		m[:3, 3] = t
		locals_[j] = m
	return locals_


def _anim_times(st: _Stage, anim_path: str | None) -> list[float]:
	prim = st.by_path.get(anim_path) if anim_path else None
	if prim is None:
		return []
	times: set[float] = set()
	for a in ("translations", "rotations", "scales"):
		if prim.has_attribute(st.T(a)):
			times.update(float(x) for x in prim.list_attribute_sample_times(st.T(a)))
	return sorted(times)


def _is_mesh(st: _Stage, prim) -> bool:
	names = set(prim.list_attribute_names())
	return {"points", "faceVertexCounts", "faceVertexIndices"} <= names


def _is_skeleton(prim) -> bool:
	names = set(prim.list_attribute_names())
	return "joints" in names and ("bindTransforms" in names or "restTransforms" in names)


def _interp(meta: str | None, count: int, npoints: int, ncorners: int, nfaces: int) -> str | None:
	if meta in ("vertex", "varying"):
		return "vertex" if count >= npoints else None
	if meta == "faceVarying":
		return "faceVarying" if count >= ncorners else None
	if meta == "uniform":
		return "uniform" if count >= nfaces else None
	if meta == "constant":
		return "constant" if count >= 1 else None
	if count == ncorners and count != npoints:
		return "faceVarying"
	if count == npoints:
		return "vertex"
	if count == ncorners:
		return "faceVarying"
	if count == nfaces:
		return "uniform"
	if count == 1:
		return "constant"
	return None


def _primvar(st: _Stage, prim, name: str, comps: int, npoints: int, corner_point: "np.ndarray",
		face_of_corner: "np.ndarray") -> "np.ndarray | None":
	"""A primvar expanded to one value per face corner, or None."""
	vals = st.nums(prim, name)
	if vals is None or len(vals) < comps or len(vals) % comps:
		return None
	vals = vals.reshape(-1, comps)
	idx = st.nums(prim, name + ":indices")
	if idx is not None and len(idx):
		ii = idx.astype(np.int64)
		if ii.min() < 0 or ii.max() >= len(vals):
			return None
		vals = vals[ii]
	ncorners = len(corner_point)
	nfaces = int(face_of_corner.max()) + 1 if len(face_of_corner) else 0
	how = _interp(st.meta(prim, name).get("interpolation"), len(vals), npoints, ncorners, nfaces)
	if how == "faceVarying":
		return vals[:ncorners]
	if how == "vertex":
		return vals[corner_point]
	if how == "uniform":
		return vals[face_of_corner]
	if how == "constant":
		return np.repeat(vals[:1], ncorners, 0)
	return None


@dataclass
class _MeshData:
	path: str
	points: "np.ndarray"	# (n, 3) authored points
	tris: "np.ndarray"	# (m, 3) corner ids
	tri_face: "np.ndarray"	# (m,) source face per triangle
	corner_point: "np.ndarray"	# (c,) point per corner
	normals: "np.ndarray"	# (c, 3) per corner, mesh space
	uvs: "np.ndarray"	# (c, 2) top-left origin
	colour: "np.ndarray | None"	# displayColor, first value


def _read_mesh(st: _Stage, prim, warnings: list[str]) -> _MeshData | None:
	path = st.text(prim.path())
	pts = st.nums(prim, "points")
	counts = st.nums(prim, "faceVertexCounts")
	fvi = st.nums(prim, "faceVertexIndices")
	if pts is None or counts is None or fvi is None or len(pts) < 9 or len(pts) % 3:
		return None
	pts = pts.reshape(-1, 3)
	counts = counts.astype(np.int64)
	fvi = fvi.astype(np.int64)
	if counts.sum() != len(fvi) or (len(fvi) and (fvi.min() < 0 or fvi.max() >= len(pts))):
		warnings.append(f"{path}: face indices don't match its points; skipped")
		return None
	starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
	face_of_corner = np.repeat(np.arange(len(counts)), counts)
	tris, tri_face = [], []
	for f, (s, c) in enumerate(zip(starts, counts)):
		for k in range(1, int(c) - 1):
			tris.append((s, s + k, s + k + 1))
			tri_face.append(f)
	if not tris:
		return None
	tris_a = np.array(tris, np.int64)
	if (st.token(prim, "orientation") or "rightHanded") == "leftHanded":
		tris_a = tris_a[:, ::-1].copy()

	nrm = None
	for name in ("primvars:normals", "normals"):
		nrm = _primvar(st, prim, name, 3, len(pts), fvi, face_of_corner)
		if nrm is not None:
			break
	if nrm is None:	# smooth normals from area-weighted face normals
		a, b, c = pts[fvi[tris_a[:, 0]]], pts[fvi[tris_a[:, 1]]], pts[fvi[tris_a[:, 2]]]
		fn = np.cross(b - a, c - a)
		acc = np.zeros_like(pts)
		for k in range(3):
			np.add.at(acc, fvi[tris_a[:, k]], fn)
		lens = np.linalg.norm(acc, axis=1, keepdims=True)
		acc = np.where(lens > 1e-12, acc / np.maximum(lens, 1e-12), [0.0, 0.0, 1.0])
		nrm = acc[fvi]
	uv = None
	names = list(_UV_NAMES) + [a for a in prim.list_attribute_names()
		if a.startswith("primvars:") and not a.endswith(":indices") and a not in _UV_NAMES
		and any(k in a.lower() for k in ("uv", "st", "texcoord", "map"))
		and not a.startswith("primvars:skel:") and a != "primvars:displayColor"]
	for name in names:
		if prim.has_attribute(st.T(name)):
			uv = _primvar(st, prim, name, 2, len(pts), fvi, face_of_corner)
			if uv is not None:
				break
	if uv is None:
		uv = np.zeros((len(fvi), 2))
	else:
		uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])	# USD st is bottom-left; studio s/t top-left
	col = st.nums(prim, "primvars:displayColor")
	colour = col[:3] if col is not None and len(col) >= 3 else None
	return _MeshData(path, pts, tris_a, np.array(tri_face, np.int64), fvi, nrm, uv, colour)


def _material_path(st: _Stage, path: str) -> str | None:
	p: str | None = path
	while p and p != "/":
		prim = st.by_path.get(p)
		if prim is not None:
			for rel in ("material:binding", "material:binding:full", "material:binding:preview"):
				t = st.targets(prim, rel)
				if t:
					return t[0]
		p = st.parent_path(p)
	return None


def _subsets(st: _Stage, mesh_prim) -> list[tuple[str, "np.ndarray"]]:
	"""materialBind GeomSubsets: (material path, face indices)."""
	out = []
	for child in mesh_prim.children():
		names = set(child.list_attribute_names())
		if "indices" not in names:
			continue
		fam = st.token(child, "familyName")
		if fam not in (None, "", "materialBind"):
			continue
		mat = st.targets(child, "material:binding")
		idx = st.nums(child, "indices")
		if mat and idx is not None:
			out.append((mat[0], idx.astype(np.int64)))
	return out


def _load_material(st: _Stage, mat_path: str | None, colour: "np.ndarray | None", opts: _m.ModelOptions,
		warnings: list[str]) -> _m.Material:
	name = mat_path.rsplit("/", 1)[-1] if mat_path else "displayColor"
	tex_ref, rgb, opacity = "", None, 1.0
	if mat_path:
		try:
			mat = st.shade.Material.read_from_prim(st.stage, st.sdf.Path.from_string(mat_path))
			shader = mat.get_surface_shader_path() if mat else None
			surface = st.shade.PreviewSurface.read_from_prim(st.stage, shader) if shader is not None else None
		except Exception:	# noqa: BLE001 - FreeUSD raises on unresolvable paths
			surface = None
		if surface:
			tex_ref = surface.get_diffuse_texture_asset_path() or ""
			c = surface.get_diffuse_color()
			if c is not None:
				rgb = _nums(c)[:3]
			o = surface.get_opacity()
			if o is not None:
				opacity = float(o)
		elif mat_path not in st.by_path:
			warnings.append(f"material {mat_path} not found; using the mesh colour")
		if not tex_ref and mat_path in st.by_path:
			tex_ref, c = _network_fallback(st, mat_path)
			if rgb is None and c is not None:
				rgb = c
	rgba = None
	if tex_ref:
		ref = tex_ref.replace("<UDIM>", "1001")
		for d in st.layers.dirs:
			rgba = _m._load_texture(ref, d / "_", [])
			if rgba is not None:
				break
		if rgba is None:
			warnings.append(f"material {name}: texture {tex_ref} not found; using its colour")
	if rgba is None:
		col = rgb if rgb is not None and len(rgb) == 3 else colour if colour is not None else np.array([0.8, 0.8, 0.8])
		c = [int(round(max(0.0, min(1.0, float(x))) * 255)) for x in list(col)[:3]]
		rgba = np.empty((8, 8, 4), np.uint8)
		rgba[...] = (*c, 255 if opacity >= 0.5 else 0)
	rgba = _m._fit_skin(rgba, opts.skin_max_edge)
	masked = bool((rgba[..., 3] < opts.alpha_threshold).any())
	return _m.Material(name[:63], rgba, masked, tex_ref)


_NOT_DIFFUSE = ("normal", "nrm", "rough", "metal", "spec", "occl", "_ao", "opac", "emis", "height", "disp", "bump",
	"mask", "orm", "arm")
_DIFFUSE_HINTS = ("diffuse", "albedo", "basecolor", "base_color", "color", "colour", "diff", "_col", "_d.")


def _network_fallback(st: _Stage, mat_path: str) -> tuple[str, "np.ndarray | None"]:
	"""The material's colour texture and diffuseColor when FreeUSD lost the shader connections.

	Connections authored inside a referenced layer don't survive FreeUSD's
	composition, so look at the shaders under the material directly: the
	UsdUVTexture that looks like a colour map, and the preview surface's
	unconnected diffuseColor.
	"""
	textures: list[tuple[int, str]] = []
	colour = None
	for p, prim in st.by_path.items():
		if not p.startswith(mat_path + "/"):
			continue
		sid = st.token(prim, "info:id") or ""
		if sid == "UsdPreviewSurface" and colour is None:
			c = st.nums(prim, "inputs:diffuseColor")
			if c is not None and len(c) >= 3:
				colour = c[:3]
		f = st.attr(prim, "inputs:file")
		if f is None:
			continue
		ref = str(f.as_string() or f).strip().strip("@")
		if not ref:
			continue
		hint = (p.rsplit("/", 1)[-1] + " " + Path(ref).name).lower()
		score = 2 if any(h in hint for h in _DIFFUSE_HINTS) else 1
		if any(h in hint for h in _NOT_DIFFUSE):
			score = 0
		textures.append((score, ref))
	textures.sort(key=lambda x: -x[0])
	if textures and (textures[0][0] > 0 or len(textures) == 1):
		return textures[0][1], colour
	return "", colour


def _skin_weights(st: _Stage, prim, npoints: int, sk: _Skel) -> "np.ndarray | None":
	"""Each point's heaviest joint (skeleton joint index), or None when unbound."""
	ji = st.nums(prim, "primvars:skel:jointIndices")
	jw = st.nums(prim, "primvars:skel:jointWeights")
	if ji is None or not len(ji):
		return None
	meta = st.meta(prim, "primvars:skel:jointIndices")
	if jw is None or len(jw) != len(ji):
		jw = np.ones(len(ji))
	es = int(meta.get("elementSize", 0) or 0)
	if es <= 0:
		es = len(ji) // npoints if npoints and len(ji) % npoints == 0 else len(ji)
	es = max(1, es)
	ji = ji.astype(np.int64).reshape(-1, es)
	jw = jw.reshape(-1, es)
	best = ji[np.arange(len(ji)), jw.argmax(1)]
	if len(best) == 1 or meta.get("interpolation") == "constant":
		best = np.repeat(best[:1], npoints)
	if len(best) != npoints:
		return None
	# Mesh-level skel:joints reorders the indices into its own joint list.
	order = st.tokens(prim, "skel:joints")
	if order:
		index = {j: i for i, j in enumerate(sk.joints)}
		remap = np.array([index.get(j, 0) for j in order], np.int64)
		best = remap[np.clip(best, 0, len(remap) - 1)]
	return np.clip(best, 0, len(sk.joints) - 1)


def _skeleton_for(st: _Stage, path: str, skels: dict[str, _Skel]) -> _Skel | None:
	p: str | None = path
	while p and p != "/":
		prim = st.by_path.get(p)
		if prim is not None:
			t = st.targets(prim, "skel:skeleton")
			if t:
				return skels.get(t[0])
		p = st.parent_path(p)
	return None


def _seq_label(name: str) -> str:
	return _m._seq_name(name)


def _frame_times(start: float, end: float, tcps: float, fps: int) -> list[float]:
	n = max(1, int(round((end - start) / tcps * fps)) + 1)
	return [min(end, start + k / fps * tcps) for k in range(n)]


# -- import ---------------------------------------------------------------------------


def import_usd(path: Path, opts: _m.ModelOptions | None = None) -> _m.ImportedModel:
	"""Read a USD stage into triangulated submeshes, bones and sequences."""
	opts = opts or _m.ModelOptions()
	path = Path(path)
	with tempfile.TemporaryDirectory(prefix="gxbake-usd-") as tmp:
		layers = _prepare(path, Path(tmp))
		st = _Stage(layers)
		return _build(st, path, opts, Path(tmp))


def _axis(st: _Stage, opts: _m.ModelOptions) -> "np.ndarray":
	up = str(st.stage.up_axis() or "Y").strip().upper()
	mpu = float(st.stage.meters_per_unit() or 0) or 0.01
	return _m._axis_matrix(_m.ModelOptions(scale=opts.scale * mpu / _INCH, up_axis="z" if up == "Z" else "y"))


def _build(st: _Stage, path: Path, opts: _m.ModelOptions, tmp: Path) -> _m.ImportedModel:
	warnings: list[str] = []
	axis = _axis(st, opts)

	skels: dict[str, _Skel] = {}
	meshes: list[tuple] = []
	for prim in st.prims:
		if _is_skeleton(prim):
			sk = _read_skeleton(st, prim)
			if sk:
				skels[sk.path] = sk
		elif _is_mesh(st, prim):
			if not st.shown(prim):
				continue
			md = _read_mesh(st, prim, warnings)
			if md is not None:
				meshes.append((prim, md))
	if not meshes:
		raise ValueError(f"{path.name}: no meshes FreeUSD could read")

	# Bones: a root, each skeleton's joints, and one per mesh that moves on its own.
	skeletal = not opts.static
	bound: dict[str, tuple[_Skel, np.ndarray]] = {}
	if skeletal:
		for prim, md in meshes:
			sk = _skeleton_for(st, md.path, skels)
			if sk is not None:
				w = _skin_weights(st, prim, len(md.points), sk)
				if w is not None:
					bound[md.path] = (sk, w)
	used_skels = list(dict.fromkeys(sk.path for sk, _w in bound.values()))
	start, end = st.timeline()
	moving = [md.path for _p, md in meshes if md.path not in bound and len(st.xform_times(md.path)) > 1] if skeletal else []
	nbones = 1 + sum(len(skels[s].joints) for s in used_skels) + len(moving)
	if nbones > _sk.MAXSTUDIOBONES:
		warnings.append(f"{nbones} bones exceeds the studio limit of {_sk.MAXSTUDIOBONES}; baked static instead")
		skeletal, bound, used_skels, moving = False, {}, [], []
	if opts.static and (skels or any(len(st.xform_times(md.path)) > 1 for _p, md in meshes)):
		warnings.append("baked static: skeleton and animations ignored")

	model = _m.ImportedModel(path.stem, [], [], warnings)
	used_names: set[str] = {"root"}
	rest_world: list[np.ndarray] = [np.eye(4)]	# studio bone → rigid model-space rest matrix
	bone_of_mesh: dict[str, int] = {}

	def add_bone(name: str, parent: int, world: "np.ndarray") -> int:
		w = _sk.rigid(world)
		local = w if parent < 0 else _sk.rigid_inverse(rest_world[parent]) @ w
		model.bones.append(_m.Bone(_m._unique(name[:31] or f"bone{len(model.bones)}", used_names), parent,
			_sk.pose_from_matrix(local)))
		rest_world.append(w)
		return len(model.bones) - 1

	skel_rest: dict[str, list[np.ndarray]] = {}
	for sp in used_skels:
		sk = skels[sp]
		sk.first_bone = len(model.bones)
		rest = _rest_space(sk, sk.rest_local)
		skel_rest[sp] = rest
		sw = st.world(sp)
		for j, jp in enumerate(sk.joints):
			parent = sk.first_bone + sk.parents[j] if sk.parents[j] >= 0 else 0
			add_bone(jp.rsplit("/", 1)[-1], parent, axis @ sw @ rest[j])
	for mp in moving:
		bone_of_mesh[mp] = add_bone(mp.rsplit("/", 1)[-1], 0, axis @ st.world(mp))

	# Meshes: one submesh per (mesh, material), vertices in their bone's space.
	mat_index: dict[tuple, int] = {}
	for prim, md in meshes:
		face_mat: dict[int, str] = {}
		for mat_path, faces in _subsets(st, prim):
			for f in faces:
				face_mat[int(f)] = mat_path
		default_mat = _material_path(st, md.path)
		corner_world = np.zeros((len(md.corner_point), 3))
		corner_nrm = np.zeros((len(md.corner_point), 3))
		corner_bone = np.zeros(len(md.corner_point), np.int64)
		flip = False
		if md.path in bound:
			sk, jw = bound[md.path]
			gb = _matrices(st.nums(prim, "primvars:skel:geomBindTransform"))
			geom_bind = gb[0] if gb else np.eye(4)
			sw = st.world(sk.path)
			rest = skel_rest[sk.path]
			mats = np.array([axis @ sw @ rest[j] @ np.linalg.inv(sk.bind_world[j]) @ geom_bind
				for j in range(len(sk.joints))])
			pj = jw[md.corner_point]
			m = mats[pj]
			p = md.points[md.corner_point]
			corner_world = np.einsum("nij,nj->ni", m[:, :3, :3], p) + m[:, :3, 3]
			nm = np.linalg.inv(m[:, :3, :3]).transpose(0, 2, 1)
			corner_nrm = np.einsum("nij,nj->ni", nm, md.normals)
			corner_bone = sk.first_bone + pj
			flip = np.linalg.det((axis @ sw @ geom_bind)[:3, :3]) < 0
		else:
			m = axis @ st.world(md.path)
			p = md.points[md.corner_point]
			corner_world = p @ m[:3, :3].T + m[:3, 3]
			corner_nrm = md.normals @ np.linalg.inv(m[:3, :3])
			corner_bone[:] = bone_of_mesh.get(md.path, 0)
			flip = np.linalg.det(m[:3, :3]) < 0
		# into bone space
		bw = np.array(rest_world)[corner_bone]
		rel = corner_world - bw[:, :3, 3]
		local = np.einsum("nji,nj->ni", bw[:, :3, :3], rel)
		ln = np.einsum("nji,nj->ni", bw[:, :3, :3], corner_nrm)
		lens = np.linalg.norm(ln, axis=1, keepdims=True)
		ln = np.where(lens > 1e-8, ln / np.maximum(lens, 1e-8), [0.0, 0.0, 1.0])
		tris = md.tris[:, ::-1] if flip else md.tris
		groups: dict[str | None, list[int]] = {}
		for t, f in enumerate(md.tri_face):
			groups.setdefault(face_mat.get(int(f), default_mat), []).append(t)
		for mat_path, tri_ids in groups.items():
			key = (mat_path,) if mat_path else ("colour", tuple(np.round(md.colour, 3)) if md.colour is not None else None)
			if key not in mat_index:
				mat = _load_material(st, mat_path, md.colour, opts, warnings)
				same = (mat.name, mat.texture, mat.rgba.shape, hash(mat.rgba.tobytes()))
				if same not in mat_index:
					mat_index[same] = len(model.materials)
					model.materials.append(mat)
				mat_index[key] = mat_index[same]
			sel = tris[tri_ids]
			used, inverse = np.unique(sel.ravel(), return_inverse=True)
			model.meshes.append(_m.Submesh(mat_index[key], local[used].astype(np.float32), ln[used].astype(np.float32),
				md.uvs[used].astype(np.float32), inverse.reshape(-1, 3).astype(np.uint32),
				corner_bone[used].astype(np.int32), corner_world[used].astype(np.float32)))
	if len(model.materials) > _m.MAXSTUDIOSKINS:
		warnings.append(f"{len(model.materials)} materials; studio models allow {_m.MAXSTUDIOSKINS}")

	# Sequences: the stage timeline, then <name>@<clip>.usd* files.
	if skeletal and len(model.bones) > 1:
		used: set[str] = set()
		parents = [b.parent for b in model.bones]
		frames = _sample(st, used_skels, skels, moving, axis, parents, opts.anim_fps)
		if frames is not None:
			anim_names = [skels[s].anim.rsplit("/", 1)[-1] for s in used_skels if skels[s].anim]
			label = _m._unique(_seq_label(anim_names[0] if anim_names else "anim"), used)
			model.sequences.append(_m.Sequence(label, float(opts.anim_fps), frames))
		if opts.clip_files:
			fixed = {p: st.world(p) for p in list(used_skels) + moving}
			for clip in _m.clip_files(path):
				try:
					with tempfile.TemporaryDirectory(prefix="gxbake-usdclip-") as ctmp:
						cst = _Stage(_prepare(clip, Path(ctmp)))
						cf = _sample_clip(cst, used_skels, skels, moving, axis, parents, opts.anim_fps, fixed)
				except (ValueError, OSError) as exc:
					warnings.append(f"clip {clip.name}: {exc}")
					continue
				if cf is None:
					warnings.append(f"clip {clip.name}: no animation for this model's joints or prims")
					continue
				label = _m._unique(_seq_label(clip.stem.split("@", 1)[1]), used)
				model.sequences.append(_m.Sequence(label, float(opts.anim_fps), cf))
	if not model.sequences:
		model.sequences.append(_m.Sequence("idle", 30.0, model.rest_frames(), loop=True))
	if not model.meshes:
		raise ValueError(f"{path.name}: no triangle meshes")
	return model


def _sample(st: _Stage, used_skels: list[str], skels: dict[str, _Skel], moving: list[str], axis: "np.ndarray",
		parents: list[int], fps: int, anims: dict[str, str | None] | None = None,
		fixed: dict[str, "np.ndarray"] | None = None) -> "np.ndarray | None":
	"""Bake the stage timeline into (numframes, numbones, 6) studio poses; None if nothing moves.

	`fixed` holds world matrices for prims this stage doesn't have (a clip file
	that only animates the skeleton).
	"""
	anims = anims if anims is not None else {s: skels[s].anim for s in used_skels}
	fixed = fixed or {}
	times: set[float] = set()
	for s in used_skels:
		times.update(_anim_times(st, anims.get(s)))
	for mp in moving:
		if mp in st.by_path:
			times.update(st.xform_times(mp))
	if len(times) < 2:
		return None
	start, end = st.timeline()
	if end <= start:
		start, end = min(times), max(times)

	def world(p: str, t: float) -> "np.ndarray":
		return st.world(p, t) if p in st.by_path or p not in fixed else fixed[p]

	ft = _frame_times(start, end, st.tcps, fps)
	frames = np.zeros((len(ft), len(parents), 6))
	for k, t in enumerate(ft):
		bones: list[np.ndarray] = [np.eye(4)]
		for s in used_skels:
			sk = skels[s]
			sw = world(sk.path, t)
			ws = _rest_space(sk, _anim_locals(st, sk, anims.get(s), t))
			for j in range(len(sk.joints)):
				bones.append(_sk.rigid(axis @ sw @ ws[j]))
		for mp in moving:
			bones.append(_sk.rigid(axis @ world(mp, t)))
		for b, w in enumerate(bones):
			rel = w if parents[b] < 0 else _sk.rigid_inverse(bones[parents[b]]) @ w
			frames[k, b] = _sk.pose_from_matrix(rel)
	return frames


def _sample_clip(cst: _Stage, used_skels: list[str], skels: dict[str, _Skel], moving: list[str],
		axis: "np.ndarray", parents: list[int], fps: int, fixed: dict[str, "np.ndarray"]) -> "np.ndarray | None":
	"""A clip file: its SkelAnimations matched to our skeletons by joint names, prims by path."""
	anim_prims = [p for p in cst.prims if "joints" in p.list_attribute_names()
		and any(a in p.list_attribute_names() for a in ("translations", "rotations"))]
	anims: dict[str, str | None] = {}
	for s in used_skels:
		joints = set(skels[s].joints) | {j.rsplit("/", 1)[-1] for j in skels[s].joints}
		best, score = None, 0
		for ap in anim_prims:
			names = cst.tokens(ap, "joints")
			hit = sum(1 for nm in names if nm in joints or nm.rsplit("/", 1)[-1] in joints)
			if hit > score:
				best, score = cst.text(ap.path()), hit
		anims[s] = best
	if not any(anims.values()) and not any(mp in cst.by_path for mp in moving):
		return None
	return _sample(cst, used_skels, skels, moving, axis, parents, fps, anims, fixed)


def describe(path: Path) -> dict:
	"""What FreeUSD sees in a stage, for `gxbake inspect`."""
	with tempfile.TemporaryDirectory(prefix="gxbake-usd-") as tmp:
		layers = _prepare(Path(path), Path(tmp))
		st = _Stage(layers)
		meshes = [st.text(p.path()) for p in st.prims if _is_mesh(st, p)]
		skels = [st.text(p.path()) for p in st.prims if _is_skeleton(p)]
		return {
			"up_axis": str(st.stage.up_axis() or "Y"),
			"meters_per_unit": float(st.stage.meters_per_unit() or 0.01),
			"time_codes": (st.start, st.end, st.tcps),
			"prims": len(st.prims),
			"meshes": meshes,
			"skeletons": skels,
			"layers_rewritten": not layers.crate,
		}
