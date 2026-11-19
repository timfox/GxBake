"""Batch and manifest-driven builds: many sources in, GameCube files out, under budgets.

Every converter here turns its inputs into *units*. A unit is a set of
autotex jobs (textures whose format and size are still open) plus a writer
that saves them once they are placed. All the units that share a budget are
placed together by `autotex.fit`, so a texture pool's bytes go where they
buy the most picture, across texture folders, PBR materials, HDR lightmaps
and sphere maps alike. Skies are fixed (128² CMPR, the port's only size) and
are only counted.

	gxbake build                      # reads ./gxbake.toml (or .json)
	gxbake build assets.toml --force  # ignore the incremental cache
	gxbake build --init               # write a commented example manifest

Manifest (TOML or JSON), paths relative to the manifest:

	[build]
	output = "build/gc"     # every rule's `output` is under this
	budget = "448K"         # shared by every rule without its own budget (0 = none)
	min_db = 38             # default quality floor
	jobs = 4                # worker processes for encoding

	[[textures]]            # any images → auto-picked GX formats
	inputs = ["art/ui/**/*.png"]
	role = "ui"             # auto | color | world | ui | mask | normal | lightmap
	container = "tpl"       # tpl (one per image) | bti | gct (one CMPR pack) | combined tpl via `combine`
	combine = "ui.tpl"      # optional: every image in one TPL plus a .h of indices

	[[pbr]]                 # material sets or glTF/GLB → baked maps + .gxmat
	inputs = ["materials"]
	preset = "lit"          # prelit | lit | emboss | indirect
	env = "sky/studio.hdr"  # sphere map for the indirect preset

	[[hdr]]                 # EXR/HDR/PFM → tone-mapped textures or overbright lightmaps
	inputs = ["lightmaps/*.exr"]
	mode = "lightmap"       # texture | lightmap
	overbright = "auto"     # 1 | 2 | 4 | auto (smallest TEV scale that does not clip)

	[[sphere]]              # panoramas/cubemaps → GX sphere maps for reflections
	inputs = ["env/*.hdr"]
	size = 128
	roughness = 0.2

	[[sky]]                 # → media/sky_<name><side>.cmp
	inputs = ["sky/canyon.hdr"]
	tonemap = "aces"

	[[camera]]              # glTF/DAE/FBX/JSON/CSV/BSP camera paths → .gcam keyframe tracks
	inputs = ["cutscenes/*.glb"]
	output = "media"
	combine = "cam_c1a0.gcam"   # every clip in one file (default: one .gcam per source)
	pos_tol = 0.25          # GoldSrc units; rot_tol (degrees), fov_tol (degrees)
	max_bytes = "4K"        # loosen the tolerances until the file fits
	[[gi]]                  # GoldSrc BSPs → radiosity lightmaps (docs/gi.md)
	inputs = ["maps/*.bsp"]
	bounces = 3             # light bounces between surfaces
	rad = "lights.rad"      # texture lights
	ao = 0.5                # ambient occlusion on indirect and sky light
	budget = "800K"         # per map; each map is its own group

Lightmap atlases are per map (the port loads one map's at a time), so every
BSP of a [[gi]] rule is placed against its own budget, default 800 KiB.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import autotex, gct, tpl
from .budget import parse_size

STATE_FILE = ".gxbake-build.json"
CONTAINERS = ("tpl", "bti", "gct")
RULE_KINDS = ("textures", "pbr", "hdr", "sphere", "sky", "camera", "gi")
GI_BUDGET = "800K"	# the port's lightmap atlas window (vid_gamecube.c GC_PackCapLightmapAtlas)


@dataclass
class Written:
	path: Path
	bytes: int
	kind: str


@dataclass
class Unit:
	kind: str
	name: str
	jobs: list[autotex.TexJob]
	write: Callable[[list[autotex.Placed]], list[Written]]
	inputs: list[Path] = field(default_factory=list)
	fixed_bytes: int = 0	# memory the unit takes that no budget step can change (skies)
	notes: list[str] = field(default_factory=list)
	info: dict = field(default_factory=dict)


@dataclass
class UnitResult:
	unit: Unit
	placed: list[autotex.Placed]
	written: list[Written]


@dataclass
class GroupResult:
	budget: int
	total: int
	fits: bool
	units: list[UnitResult]
	skipped: bool = False


# -- shared helpers ----------------------------------------------------------------------------------


def safe_name(text: str, n: int = 64) -> str:
	return (re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "tex")[:n]


def expand(inputs: list[str], root: Path, exts: set[str] | None = None, recursive: bool = True) -> list[tuple[Path, Path]]:
	"""(file, base) pairs for paths, folders and globs; `base` keeps sub-folders in the output."""
	from .imageformats import EXTENSIONS

	exts = exts if exts is not None else set(EXTENSIONS)
	out: list[tuple[Path, Path]] = []
	seen: set[Path] = set()
	for raw in inputs:
		p = Path(raw)
		p = p if p.is_absolute() else root / p
		if any(c in raw for c in "*?["):
			hits = [Path(h) for h in sorted(glob.glob(str(p), recursive=True))]
			base = Path(str(p).split("*")[0].split("?")[0].split("[")[0])
			base = base if base.is_dir() else base.parent
			pairs = [(h, base) for h in hits if h.is_file()]
		elif p.is_dir():
			it = p.rglob("*") if recursive else p.iterdir()
			pairs = [(f, p) for f in sorted(it) if f.is_file()]
		else:
			pairs = [(p, p.parent)]
		for f, b in pairs:
			if f.suffix.lower() in exts and f.resolve() not in seen:
				seen.add(f.resolve())
				out.append((f, b))
	return out


def _rel_stem(f: Path, base: Path) -> Path:
	try:
		rel = f.relative_to(base)
	except ValueError:
		rel = Path(f.name)
	return rel.parent / safe_name(rel.stem)


def _preview(placed: list[autotex.Placed], folder: Path | None) -> list[Written]:
	if folder is None:
		return []
	from PIL import Image

	folder.mkdir(parents=True, exist_ok=True)
	out = []
	for p in placed:
		f = folder / f"{safe_name(p.job.name)}.png"
		Image.fromarray(p.image.decode()).save(f)
		out.append(Written(f, f.stat().st_size, "preview"))
	return out


def _save(path: Path, data: bytes, kind: str) -> Written:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(data)
	return Written(path, len(data), kind)


def gct_entry(p: autotex.Placed) -> gct.GctEntry:
	"""A placed CMPR texture as a GCWT pack entry (the port's world-texture format)."""
	im = p.image
	if im.fmt != "CMPR":
		raise ValueError(f"{p.job.name}: .gct holds CMPR only, got {im.fmt}")
	body = im.data
	alpha = bool((p.job.rgba[..., 3] < 128).any())
	tag = int(hashlib.md5(body).hexdigest()[:4], 16)
	return gct.GctEntry(safe_name(p.job.name, 15), im.width, im.height, len(im.levels),
		gct.GC_WT_FLAG_ALPHA if alpha else 0, tag, body)


def write_images(placed: list[autotex.Placed], out_stem: Path, container: str) -> list[Written]:
	"""One TPL holding every image (in order), or one BTI per image."""
	if container == "tpl":
		return [_save(out_stem.with_suffix(".tpl"), tpl.write_tpl([p.image for p in placed]), "tpl")]
	if container == "bti":
		if len(placed) == 1:
			return [_save(out_stem.with_suffix(".bti"), tpl.write_bti(placed[0].image), "bti")]
		return [_save(out_stem.parent / f"{out_stem.name}_{p.job.meta.get('map', i)}.bti", tpl.write_bti(p.image), "bti")
			for i, p in enumerate(placed)]
	raise ValueError(f"unknown container {container!r}")


def c_header(names: list[str], guard: str) -> str:
	g = safe_name(guard).upper()
	lines = [f"/* generated by gxbake: TPL image indices */", f"#ifndef {g}_H", f"#define {g}_H", ""]
	lines += [f"#define {g}_{safe_name(n).upper()} {i}" for i, n in enumerate(names)]
	lines += [f"#define {g}_COUNT {len(names)}", "", f"#endif", ""]
	return "\n".join(lines)


# -- rules → units -----------------------------------------------------------------------------------


@dataclass
class Rule:
	kind: str
	inputs: list[str]
	output: str = ""
	options: dict = field(default_factory=dict)

	def get(self, key: str, default=None):
		return self.options.get(key, default)


def _job_opts(rule: Rule, defaults: dict) -> dict:
	g = {**defaults, **rule.options}
	fm = g.get("formats")
	if isinstance(fm, str):
		fm = [x.strip().upper() for x in fm.split(",") if x.strip()]
	role = g.get("role", "auto")
	if role not in autotex.ROLES:
		raise ValueError(f"unknown role {role!r} (one of {', '.join(autotex.ROLES)})")
	return {"formats": tuple(fm) if fm else autotex.ROLES[role], "min_db": float(g.get("min_db", 38.0)),
		"max_edge": int(g.get("max_edge", 256)), "min_edge": int(g.get("min_edge", 8)),
		"mipmaps": bool(g.get("mipmaps", True)), "pow2": bool(g.get("pow2", True)),
		"wrap": g.get("wrap", "repeat"), "weight": float(g.get("weight", 1.0))}


def texture_units(rule: Rule, root: Path, out: Path, defaults: dict, preview: Path | None = None) -> list[Unit]:
	from .images import load_rgba

	o = _job_opts(rule, defaults)
	container = rule.get("container", defaults.get("container", "tpl"))
	if container not in CONTAINERS:
		raise ValueError(f"unknown container {container!r} (one of {', '.join(CONTAINERS)})")
	if container == "gct":
		o["formats"] = ("CMPR",)
	files = expand(rule.inputs, root, recursive=bool(rule.get("recursive", True)))
	exclude = rule.get("exclude", [])
	files = [(f, b) for f, b in files if not any(fnmatch.fnmatch(f.name, x) or fnmatch.fnmatch(str(f), x) for x in exclude)]
	jobs = []
	for f, b in files:
		warn: list[str] = []
		jobs.append(autotex.TexJob(str(_rel_stem(f, b)).replace("\\", "/"), load_rgba(f, warn), **o,
			meta={"source": str(f), "warnings": warn}))
	combine = rule.get("combine")
	if container == "gct" or combine:
		name = combine or "textures.gct"
		path = out / name

		def write_all(placed, path=path, container=container):
			if container == "gct":
				data = gct.write_gct([gct_entry(p) for p in placed])
				res = [_save(path.with_suffix(".gct"), data, "gct")]
			else:
				res = [_save(path.with_suffix(".tpl"), tpl.write_tpl([p.image for p in placed]), "tpl")]
				res.append(_save(path.with_suffix(".h"), c_header([p.job.name for p in placed], path.stem).encode(), "header"))
			return res + _preview(placed, preview)

		return [Unit("textures", path.name, jobs, write_all, [f for f, _ in files])] if jobs else []
	units = []
	for job, (f, b) in zip(jobs, files):
		stem = out / job.name

		def write_one(placed, stem=stem):
			return write_images(placed, stem, container) + _preview(placed, preview)

		units.append(Unit("textures", job.name, [job], write_one, [f]))
	return units


def _env_sphere(rule_env, root: Path, size: int, roughness: float, opts: dict) -> "np.ndarray | None":
	if not rule_env:
		return None
	from . import hdr

	p = Path(rule_env)
	src = hdr.sky_source(p if p.is_absolute() else root / p, opts.get("tonemap", "aces"), float(opts.get("exposure", 0.0)))
	return hdr.sphere_map(src, size, float(opts.get("yaw", 270.0)), float(opts.get("pitch", 0.0)), roughness)


def pbr_units(rule: Rule, root: Path, out: Path, defaults: dict, preview: Path | None = None) -> list[Unit]:
	from . import pbr

	g = {**defaults, **rule.options}
	preset = g.get("preset", "lit")
	paths: list[Path] = []
	for raw in rule.inputs:
		p = Path(raw)
		p = p if p.is_absolute() else root / p
		paths += [Path(h) for h in sorted(glob.glob(str(p), recursive=True))] if any(c in raw for c in "*?[") else [p]
	mats = []
	for p in paths:
		if p.suffix.lower() in (".gltf", ".glb"):
			mats += pbr.load_gltf(p, int(g.get("work_edge", 1024)))
	loose = [p for p in paths if p.suffix.lower() not in (".gltf", ".glb")]
	if loose:
		mats += [pbr.load_set(n, c, int(g.get("work_edge", 1024))) for n, c in pbr.find_sets(loose).items()]
	container = g.get("container", "tpl")
	if container == "gct":
		raise ValueError("pbr rules write TPL or BTI; use gct_base = \"name.gct\" to also pack the base maps for the port")
	light = g.get("light")
	if isinstance(light, str):
		light = tuple(float(x) for x in light.split(","))
	env_rough = g.get("env_roughness", "auto")
	units = []
	for mat in mats:
		env = None
		if preset == "indirect" and g.get("env"):
			r = float(np.median(mat.maps["roughness"])) if env_rough == "auto" and "roughness" in mat.maps else \
				0.0 if env_rough == "auto" else float(env_rough)
			env = _env_sphere(g["env"], root, int(g.get("env_size", 128)), r, g)
		bo = pbr.BakeOptions(preset=preset, max_edge=int(g.get("max_edge", 256)), min_db=float(g.get("min_db", 36.0)),
			mipmaps=bool(g.get("mipmaps", True)), detail=float(g.get("detail", 1.0)),
			ao_strength=float(g.get("ao_strength", 1.0)), spec_strength=float(g.get("spec_strength", 0.6)),
			ambient=float(g.get("ambient", 0.55)), env=env, env_name=str(g.get("env", "")),
			ao_from_height=bool(g.get("ao_from_height", True)))
		if light:
			bo.light = tuple(light)
		baked = pbr.bake(mat, bo)
		stem = out / safe_name(mat.name)

		def write_mat(placed, baked=baked, stem=stem, container=container):
			res = write_images(placed, stem, container)
			desc = {"material": baked.material.name, "preset": baked.preset,
				"container": res[0].path.name if container == "tpl" else [r.path.name for r in res],
				"maps": [{"index": i, **autotex.summary(p), "map": p.job.meta.get("map")} for i, p in enumerate(placed)],
				"tev": baked.tev, "sources": baked.material.sources, "notes": baked.notes}
			res.append(_save(stem.with_suffix(".gxmat.json"), json.dumps(desc, indent=1).encode(), "gxmat"))
			if container == "tpl":
				res.append(_save(stem.with_suffix(".gxm"), pbr.gxm_bytes(baked.preset,
					[p.job.meta.get("map") for p in placed], baked.tev["alpha"], baked.material.alpha_cutoff), "gxm"))
			return res + _preview(placed, preview)

		units.append(Unit("pbr", mat.name, baked.jobs, write_mat, [], notes=baked.notes,
			info={"preset": preset, "tev_stages": baked.tev["tev_stages"]}))
	gb = g.get("gct_base")
	if gb and units:
		bases = [u for u in units]
		path = out / gb

		def write_gct(placed, path=path):
			return [_save(path.with_suffix(".gct"), gct.write_gct([gct_entry(p) for p in placed]), "gct")]

		jobs = []
		for u in bases:
			j = u.jobs[0]
			jobs.append(autotex.TexJob(safe_name(u.name, 15), j.rgba, ("CMPR",), j.min_db, j.max_edge, j.min_edge,
				j.mipmaps, True, "repeat", j.weight, meta={"material": u.name, "map": "base"}))
		units.append(Unit("pbr", path.name, jobs, write_gct))
	return units


def hdr_units(rule: Rule, root: Path, out: Path, defaults: dict, preview: Path | None = None) -> list[Unit]:
	from . import hdr

	g = {**defaults, **rule.options}
	mode = g.get("mode", "texture")
	if mode not in ("texture", "lightmap"):
		raise ValueError("hdr mode is texture or lightmap (sky and sphere maps have their own rules)")
	o = _job_opts(rule, {**defaults, "role": g.get("role", "lightmap" if mode == "lightmap" else "auto")})
	files = expand(rule.inputs, root)
	container = g.get("container", "tpl")
	units = []
	for f, b in files:
		x = hdr.read(f)
		meta: dict = {"source": f.name, "mode": mode}
		if mode == "lightmap":
			ob = g.get("overbright", "auto")
			scale = hdr.pick_overbright(x, float(g.get("exposure", 0.0))) if ob == "auto" else int(ob)
			rgba, stats = hdr.overbright(x, scale, float(g.get("exposure", 0.0)), float(g.get("gamma", 1.0)))
			meta.update(stats)
		else:
			rgba = hdr.to_rgba8(x, g.get("tonemap", "aces"), float(g.get("exposure", 0.0)),
				bool(g.get("auto_exposure", True)), float(g.get("white", 4.0)))
			meta["tonemap"] = g.get("tonemap", "aces")
		name = str(_rel_stem(f, b)).replace("\\", "/")
		job = autotex.TexJob(name, rgba, **o, meta=meta)
		stem = out / name

		def write_one(placed, stem=stem, meta=meta, container=container):
			res = write_images(placed, stem, container)
			if meta.get("mode") == "lightmap":
				from .pbr import gxm_bytes

				res.append(_save(stem.with_suffix(".lightmap.json"),
					json.dumps({**meta, **autotex.summary(placed[0])}, indent=1).encode(), "json"))
				if container == "tpl":
					res.append(_save(stem.with_suffix(".gxm"), gxm_bytes("lightmap", ["lightmap"],
						tev_scale=int(meta["scale"])), "gxm"))
			return res + _preview(placed, preview)

		units.append(Unit("hdr", name, [job], write_one, [f], info=meta))
	return units


def sphere_units(rule: Rule, root: Path, out: Path, defaults: dict, preview: Path | None = None) -> list[Unit]:
	from . import hdr

	g = {**defaults, **rule.options}
	size = int(g.get("size", 128))
	o = _job_opts(rule, {**defaults, "role": g.get("role", "color"), "mipmaps": g.get("mipmaps", False)})
	o["max_edge"] = size
	o["wrap"] = "clamp"
	container = g.get("container", "tpl")
	units = []
	for f, b in expand(rule.inputs, root):
		src = hdr.sky_source(f, g.get("tonemap", "aces"), float(g.get("exposure", 0.0)), g.get("layout", "auto"))
		rgba = hdr.sphere_map(src, size, float(g.get("yaw", 270.0)), float(g.get("pitch", 0.0)),
			float(g.get("roughness", 0.0)))
		name = str(_rel_stem(f, b)).replace("\\", "/")
		stem = out / name

		def write_one(placed, stem=stem, container=container):
			return write_images(placed, stem, container) + _preview(placed, preview)

		units.append(Unit("sphere", name, [autotex.TexJob(name, rgba, **o, meta={"source": str(f)})], write_one, [f]))
	return units


def sky_units(rule: Rule, root: Path, out: Path, defaults: dict, preview: Path | None = None) -> list[Unit]:
	from . import skybox

	g = {**defaults, **rule.options}
	units = []
	groups: list[list[Path]] = []
	for raw in rule.inputs:
		p = Path(raw)
		p = p if p.is_absolute() else root / p
		if any(c in raw for c in "*?["):
			groups += [[Path(h)] for h in sorted(glob.glob(str(p), recursive=True))]
		else:
			groups.append([p])
	if g.get("faces"):
		groups = [[root / x if not Path(x).is_absolute() else Path(x) for x in rule.inputs]]
	for paths in groups:
		opts = skybox.SkyOptions(layout=g.get("layout", "auto"), yaw=float(g.get("yaw", skybox.DEFAULT_YAW)),
			fov=float(g.get("fov", 180.0)), exposure=float(g.get("exposure", 0.0)), tonemap=g.get("tonemap", "aces"))
		name = g.get("name") if len(groups) == 1 else None

		def write_sky(placed, paths=paths, opts=opts, name=name):
			sky = skybox.build(paths, opts, name)
			return [_save(out / rel, data, "sky") for rel, data in sky.files().items()]

		units.append(Unit("sky", str(paths[0].stem), [], write_sky, paths, fixed_bytes=6 * skybox.SKY_BYTES,
			notes=["skies live in the port's 48 KiB sky block, outside the texture pool"]))
	return units


def camera_units(rule: Rule, root: Path, out: Path, defaults: dict, preview: Path | None = None) -> list[Unit]:
	from . import camera

	g = {**rule.options}
	opts = camera.options_from(g)
	files = [f for f, _b in expand(rule.inputs, root, camera.EXTENSIONS)]
	units = []
	for o in camera.convert(files, opts, out, g.get("combine")):

		def write_cam(placed, o=o):
			res = [_save(o.path, o.data, "gcam")]
			if preview is not None:
				for clip in camera.read_gcam(o.data):
					f = camera.preview_png(clip, preview / f"{o.path.stem}_{clip.name}.png")
					res.append(Written(f, f.stat().st_size, "preview"))
			return res

		notes = [f"{camera.clip_name(e.path.name)}: {n}" for e in o.clips for n in e.path.notes]
		units.append(Unit("camera", o.path.name, [], write_cam, o.inputs, fixed_bytes=len(o.data), notes=notes,
			info={"clips": [camera.describe(e) for e in o.clips]}))
def gi_units(rule: Rule, root: Path, out: Path, defaults: dict, preview: Path | None = None, log=None) -> list[Unit]:
	"""Radiosity lightmaps: each BSP is baked here, its atlas pages become jobs, and the writer
	saves the relit BSP, the TPL pages, the .gclm face table and the .gxm."""
	from . import radiosity

	g = {**{k: v for k, v in defaults.items() if k == "min_db"}, **rule.options}
	g.setdefault("min_db", 40.0)
	settings = radiosity.atlas_settings(g)
	units = []
	for f, _b in expand(rule.inputs, root, {".bsp"}):
		opts = radiosity.options_from(g, root, log)
		name = safe_name(f.stem)
		b, at, jobs = radiosity.prepare(f, opts, name=name, **settings)

		def write_gi(placed, b=b, at=at, name=name):
			files, report = radiosity.write_outputs(b, at, placed, out, name, write_bsp=settings["write_bsp"],
				gamma_bsp=opts.gamma, atlas_gamma=settings["atlas_gamma"], preview=preview is not None)
			return [Written(p, n, k) for p, n, k in files]

		notes = list(b.stats["notes"])
		if at is not None:
			notes.append(f"{len(at.sizes)} page(s) {', '.join(f'{w}x{h}' for w, h in at.sizes)}, light stored ÷{at.scale} "
				f"(TEV ×{at.scale}), {at.stats['stored_luxels']} of {at.stats['luxels']} luxels kept")
		units.append(Unit("gi", name, jobs, write_gi, [f], notes=notes, info=b.stats))
	return units


BUILDERS = {"textures": texture_units, "pbr": pbr_units, "hdr": hdr_units, "sphere": sphere_units, "sky": sky_units,
	"gi": gi_units}


# -- placing and writing ------------------------------------------------------------------------------


def run_units(units: list[Unit], budget: int = 0, workers: int = 1) -> GroupResult:
	jobs = [j for u in units for j in u.jobs]
	rep = autotex.fit(jobs, budget, workers=workers) if jobs else autotex.FitReport([], budget, 0, True, 0)
	it = iter(rep.placed)
	results = []
	for u in units:
		placed = [next(it) for _ in u.jobs]
		results.append(UnitResult(u, placed, u.write(placed)))
	return GroupResult(budget, rep.total, rep.fits, results)


# -- manifests ----------------------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
	path = Path(path)
	text = path.read_text(encoding="utf-8")
	if path.suffix.lower() == ".json":
		return json.loads(text)
	try:
		import tomllib
	except ImportError:	# Python 3.10
		try:
			import tomli as tomllib	# type: ignore
		except ImportError as e:
			raise RuntimeError("TOML manifests need Python 3.11+ or `pip install tomli`; or write the manifest as JSON") from e
	return tomllib.loads(text)


def find_manifest(start: Path) -> Path | None:
	start = Path(start)
	if start.is_file():
		return start
	for name in ("gxbake.toml", "gxbake.json"):
		if (start / name).is_file():
			return start / name
	return None


def parse_rules(doc: dict) -> tuple[dict, list[Rule]]:
	settings = dict(doc.get("build", {}))
	unknown = set(doc) - {"build", *RULE_KINDS}
	if unknown:
		raise ValueError(f"unknown manifest section(s): {', '.join(sorted(unknown))} "
			f"(expected [build] and [[{']], [['.join(RULE_KINDS)}]])")
	rules = []
	for kind in RULE_KINDS:
		for entry in doc.get(kind, []):
			entry = dict(entry)
			inputs = entry.pop("inputs", entry.pop("input", []))
			if isinstance(inputs, str):
				inputs = [inputs]
			if not inputs:
				raise ValueError(f"a [[{kind}]] rule has no inputs")
			rules.append(Rule(kind, list(inputs), entry.pop("output", ""), entry))
	return settings, rules


def _fingerprint(rule: Rule, root: Path, settings: dict) -> str:
	from .. import __version__

	h = hashlib.sha256(json.dumps([__version__, rule.kind, rule.inputs, rule.output, rule.options,
		{k: v for k, v in settings.items() if k not in ("jobs",)}], sort_keys=True, default=str).encode())
	files: list[Path] = []
	extra = []
	for key in ("env", "rad", "wads"):
		v = rule.options.get(key)
		if v:
			extra += [v] if isinstance(v, str) else list(v)
	for raw in rule.inputs + extra:
		p = Path(raw)
		p = p if p.is_absolute() else root / p
		if any(c in raw for c in "*?["):
			files += [Path(x) for x in glob.glob(str(p), recursive=True)]
		elif p.is_dir():
			files += [f for f in p.rglob("*") if f.is_file()]
		else:
			files.append(p)
	for f in sorted(set(files)):
		try:
			st = f.stat()
			h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
		except OSError:
			h.update(f"{f}:missing".encode())
	return h.hexdigest()


@dataclass
class BuildReport:
	manifest: Path
	output: Path
	groups: dict[str, GroupResult]
	fits: bool

	def to_json(self) -> dict:
		out = {"manifest": str(self.manifest), "output": str(self.output), "fits": self.fits, "groups": {}}
		for name, g in self.groups.items():
			out["groups"][name] = {"budget": g.budget, "total": g.total, "fits": g.fits, "skipped": g.skipped,
				"units": [{"kind": r.unit.kind, "name": r.unit.name, "textures": [autotex.summary(p) for p in r.placed],
					"files": [str(w.path) for w in r.written], "notes": r.unit.notes, "fixed_bytes": r.unit.fixed_bytes}
					for r in g.units]}
		return out


def build(manifest: Path, force: bool = False, dry_run: bool = False, workers: int | None = None,
		log: Callable[[str], None] = lambda s: None) -> BuildReport:
	"""Run a manifest. Rules sharing a budget are placed together; a group whose inputs and
	settings have not changed since the last build is skipped (unless `force`)."""
	manifest = Path(manifest).resolve()
	root = manifest.parent
	settings, rules = parse_rules(load_manifest(manifest))
	out = Path(settings.get("output", "build/gc"))
	out = out if out.is_absolute() else root / out
	preview = settings.get("preview")
	preview = (out / preview) if preview else None
	workers = workers or int(settings.get("jobs", 1))
	defaults = {k: v for k, v in settings.items() if k in ("min_db", "max_edge", "min_edge", "mipmaps", "container", "role")}
	shared = parse_size(str(settings.get("budget", 0)))
	state_path = out / STATE_FILE
	try:
		state = json.loads(state_path.read_text())
	except (OSError, ValueError):
		state = {}
	groups: dict[str, list[Rule]] = {}
	for i, r in enumerate(rules):
		if r.kind == "gi":	# every map has the lightmap window to itself
			for f, _b in expand(r.inputs, root, {".bsp"}):
				one = Rule("gi", [str(f)], r.output, {**r.options, "budget": r.get("budget", GI_BUDGET)})
				groups.setdefault(f"gi#{i}:{f.stem}", []).append(one)
			continue
		key = f"{r.kind}#{i}" if r.get("budget") is not None else "shared"
		groups.setdefault(key, []).append(r)
	results: dict[str, GroupResult] = {}
	new_state = dict(state)
	for key, rs in groups.items():
		budget = shared if key == "shared" else parse_size(str(rs[0].get("budget")))
		fp = hashlib.sha256("".join(_fingerprint(r, root, settings) for r in rs).encode()).hexdigest()
		prev = state.get(key, {})
		if not force and prev.get("fingerprint") == fp and all(Path(f).exists() for f in prev.get("files", [])):
			log(f"{key}: up to date ({len(prev.get('files', []))} files)")
			results[key] = GroupResult(budget, prev.get("total", 0), prev.get("fits", True), [], skipped=True)
			continue
		units: list[Unit] = []
		for r in rs:
			dest = out / r.output if r.output else out
			if r.kind == "gi" and dry_run:	# baking is the slow part: only name what would be baked
				units += [Unit("gi", Path(x).stem, [], lambda placed: [], [Path(x)]) for x in r.inputs]
				continue
			if r.kind == "gi":
				units += gi_units(r, root, dest, defaults, preview, log)
				continue
			units += BUILDERS[r.kind](r, root, dest, defaults, preview)
		log(f"{key}: {len(units)} unit(s), {sum(len(u.jobs) for u in units)} texture(s)"
			+ (f", budget {budget} bytes" if budget else ""))
		if dry_run:
			results[key] = GroupResult(budget, 0, True, [UnitResult(u, [], []) for u in units])
			continue
		g = run_units(units, budget, workers)
		results[key] = g
		files = [str(w.path) for r in g.units for w in r.written]
		new_state[key] = {"fingerprint": fp, "files": files, "total": g.total, "fits": g.fits}
		log(f"{key}: {g.total} bytes" + (f" of {budget}" if budget else "") + ("" if g.fits else "  OVER BUDGET"))
	if not dry_run:
		out.mkdir(parents=True, exist_ok=True)
		state_path.write_text(json.dumps(new_state, indent=1))
	return BuildReport(manifest, out, results, all(g.fits for g in results.values()))


EXAMPLE_MANIFEST = '''# gxbake build manifest: `gxbake build` in this folder runs it.
# Paths are relative to this file. Every rule writes under [build] output.

[build]
output = "build/gc"
budget = "448K"      # textures from every rule without its own budget share this (0 = no limit)
min_db = 38          # quality floor before the budget starts trading it away
max_edge = 256
jobs = 4
# preview = "preview"  # decoded PNGs of what the GameCube will show

[[textures]]
inputs = ["textures/**/*.png"]
role = "auto"        # auto | color | world | ui | mask | normal | lightmap
container = "tpl"    # tpl | bti | gct

# [[textures]]
# inputs = ["ui"]
# role = "ui"
# mipmaps = false
# combine = "ui.tpl"   # one TPL + ui.h with an index per image
# budget = "128K"      # its own budget instead of the shared one

# [[pbr]]
# inputs = ["materials", "props/crate.glb"]
# preset = "lit"       # prelit | lit | emboss | indirect
# env = "sky/studio.hdr"
# gct_base = "materials.gct"   # also pack every base map for the port's world-texture loader

# [[hdr]]
# inputs = ["lightmaps/*.exr"]
# mode = "lightmap"    # lightmap (overbright, TEV scale 2/4) | texture (tone-mapped)
# overbright = "auto"

# [[sphere]]
# inputs = ["env/*.hdr"]
# size = 128
# roughness = 0.15

# [[sky]]
# inputs = ["sky/canyon.hdr"]
# tonemap = "aces"     # aces | reinhard | hable | agx | clip

# [[gi]]               # radiosity lightmaps: relit .bsp + GX atlas pages (.tpl/.gclm/.gxm)
# inputs = ["maps/*.bsp"]
# bounces = 3
# rad = "lights.rad"   # texture lights: TEXNAME r g b intensity
# wads = ["wads"]      # texture colours for bounce light
# ao = 0.5
# budget = "800K"      # per map (the port's lightmap window)
# texlights = { "~light1" = "255 255 220 800" }
'''
