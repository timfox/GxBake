"""Command-line front end: the same pipeline as the GUI, for scripts and CI.

	gxbake convert <files or folders...> -o OUT [options]
	gxbake inspect <file>
	gxbake validate <staged gamedir>
	gxbake budget <staged gamedir> [--map NAME] [--json]
	gxbake fit <pack.gct> [-o OUT] [--pool 448K]
	gxbake disc <staged gamedir> [--json]
	gxbake slim <model.mdl> [-o OUT] [--step 2 | --fit-deg 1] [--stub 'death*'] [--anim-tol 2]
	gxbake squeeze <staged gamedir> [-o OUT] [--emit] [--preset lossless|balanced|max] [--json]
	gxbake adpcm <wav|gcpcm...> -o OUT
	gxbake yaz0 <files...> [-d] [-o OUT]
	gxbake sfx <wavs or folders...> -o OUT [--menu] [--truncate]
	gxbake extract <archive.pak|_dir.vpk|.pk3> -o OUT [--match GLOB]
	gxbake sky <panorama|cross|strip|six faces> -o OUT [--name NAME] [--yaw 270]
	gxbake rs <main_file_cache.dat2|.dat> [--find NAME] [--json]
	gxbake ff7 <.hrc|.p|.a|.tex|<base>aa> [--anims] [--json]
	gxbake ffxi <DAT|.spw|.bgw|FINAL FANTASY XI folder> [--id N] [--textures DIR] [--wav OUT] [--obj DIR [--prefabs]] [--json]
	gxbake gui [files...]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from . import __version__
import json

from .core import audit, budget, disc, sfx, skybox
from .core.project import CONVERTIBLE, Project, Settings


def _convert(args) -> int:
	s = Settings(cache_dir=args.cache or "", wad_dirs=args.wad_dir or [], source_dirs=args.game_dir or [])
	s.map_hulls = not args.no_map_hulls
	s.map_props = not args.no_map_props
	s.map_texture_max = args.map_texture_max
	s.map_disp_power = args.map_disp_power
	s.map_shadows = not args.no_map_shadows
	s.image.max_edge = args.max_edge
	s.image.mipmaps = not args.no_mipmaps
	s.image.alpha = args.alpha
	s.pack_images = not args.no_pack
	s.pack_name = args.pack_name
	s.raw_cmp = args.raw_cmp
	s.mdl_max_edge = args.mdl_max_edge
	s.mdl_anim_step = args.mdl_anim_step
	s.spr_scale = args.spr_scale
	s.model_scale = args.model_scale
	s.model_up_axis = args.model_up
	s.model_skin_max_edge = args.model_skin_max_edge
	s.model_write_full = not args.no_model_full
	s.model_static = args.model_static
	s.model_anim_fps = args.model_fps
	s.model_clip_files = not args.no_model_clips
	s.wt_fit = args.fit_world
	s.wt_pool = budget.parse_size(args.wt_pool)
	s.model_blend_bones = not args.no_blend_bones
	s.rose_scale = args.rose_scale
	s.rose_yaw = args.rose_yaw
	s.rose_entries = args.rose_entry or []
	s.panda_scale = args.panda_scale
	s.panda_yaw = args.panda_yaw
	s.rs_scale = args.rs_scale
	s.rs_yaw = args.rs_yaw
	s.rs_entries = args.rs_entry or []
	s.ffxi_scale = args.ffxi_scale
	s.ffxi_yaw = args.ffxi_yaw
	s.ffxi_skeleton = args.ffxi_skeleton or ""
	s.ffxi_music = not args.ffxi_music_wav
	s.ff7_scale = args.ff7_scale
	s.ff7_yaw = args.ff7_yaw
	s.ff7_battle_scale = args.ff7_battle_scale
	s.video_width, s.video_height, s.video_fps = args.video_width, args.video_height, args.video_fps
	s.sky_yaw, s.sky_fov, s.sky_exposure = args.sky_yaw, args.sky_fov, args.sky_exposure
	s.detect_panoramas = not args.no_sky_detect
	proj = Project(s)
	for p in args.paths:
		proj.add_path(Path(p))
	todo = [a for a in proj.assets if a.kind in CONVERTIBLE]
	if not todo:
		print("nothing to convert", file=sys.stderr)
		return 1
	written = proj.convert(todo, Path(args.output))
	total = sum(w.size for w in written)
	print(f"wrote {len(written)} files, {audit.human_bytes(total)} "
		f"({total / audit.MAIN_RAM:.1%} of GameCube main RAM) to {args.output}")
	return 1 if any(a.status == "Failed" for a in todo) else 0


def _inspect(args) -> int:
	proj = Project()
	proj.add_path(Path(args.path))
	for a in proj.assets:
		print(f"== {a.name}")
		for k, v in proj.info(a).items():
			print(f"  {k}: {v}")
		kids = proj.children(a)
		if kids:
			print(f"  contents ({len(kids)}): {', '.join(kids[:40])}{' …' if len(kids) > 40 else ''}")
		for w in a.warnings:
			print(f"  warning: {w}")
	return 0


def _rs(args) -> int:
	from .core import runescape

	try:
		cache = runescape.Cache(Path(args.path))
		info = runescape.describe(cache)
		found = runescape.npcs(cache, args.find or "")
	except (OSError, ValueError) as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 1
	rows = [{"entry": f"npc:{n.id}", "name": n.name, "models": n.models, "stand": n.stand, "walk": n.walk}
		for n in found if n.models and (args.all or (n.name and n.name.lower() != "null"))]
	if args.json:
		print(json.dumps({**info, "npcs": rows}, indent=2))
		return 0
	print(f"{info['path']}: {info['layout']} cache, {info['models']} models")
	print("  indices: " + ", ".join(f"{i} ({n})" for i, n in info["indices"].items()))
	if cache.legacy:
		print("  legacy caches: convert models with --rs-entry model:<id>")
	for r in rows[:args.limit] if args.limit else rows:
		anims = ", ".join(f"{k} {r[k]}" for k in ("stand", "walk") if r[k] >= 0)
		print(f"  {r['entry']:<11} {r['name'] or '-':<28} models {','.join(map(str, r['models']))}"
			+ (f"  ({anims})" if anims else ""))
	if args.limit and len(rows) > args.limit:
		print(f"  … {len(rows) - args.limit} more (--limit 0 lists all)")
	return 0


def _ffxi(args) -> int:
	from .core import ffxi

	path = Path(args.path)
	try:
		if path.is_dir() or args.id is not None:
			root = path if path.is_dir() else path.parent
			rom = ffxi.Rom(root)
			if args.id is None:
				info = {"root": str(root), "roms": [n for n, _v, _f in rom.tables], "files": rom.count()}
				if args.json:
					print(json.dumps(info, indent=2))
				else:
					print(f"{root}: ROM tables {', '.join(map(str, info['roms']))}, {info['files']} files")
					print("  look one up with --id N")
				return 0
			found = rom.path(args.id)
			if found is None:
				print(f"error: file id {args.id} is not in the tables", file=sys.stderr)
				return 1
			print(f"id {args.id} → {found}")
			path = found
		if path.suffix.lower() in ffxi.AUDIO_EXTENSIONS:
			data = path.read_bytes()
			h = ffxi.read_sound_header(data)
			fmt = {0: "ADPCM", 1: "PCM", 3: "ATRAC3"}.get(h.format, str(h.format))
			if args.json:
				print(json.dumps(h.__dict__, indent=2))
			else:
				loop = f", loops from block {h.loop_start}" if h.loop_start >= 0 else ""
				print(f"{path.name}: {h.kind}, {fmt}, {h.channels} ch, {h.rate} Hz, id {h.id}{loop}")
			if args.wav:
				wav, _ = ffxi.sound_to_wav(data)
				Path(args.wav).parent.mkdir(parents=True, exist_ok=True)
				Path(args.wav).write_bytes(wav)
				print(f"wrote {args.wav}")
			return 0
		info = ffxi.describe(path)
	except (OSError, ValueError) as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 1
	if args.json:
		print(json.dumps(info, indent=2))
	else:
		print(f"{path}: " + ", ".join(f"{n} {k}" for k, n in info["chunks"].items()))
		for t in info["textures"]:
			print(f"  texture {t['name']!r}: {t['format']}, {t['size']}")
		if info["bones"]:
			print(f"  skeleton: {info['bones']} bones")
		for m in info["meshes"]:
			print(f"  mesh {m['name']!r}: {m['vertices']} vertices, {m['triangles']} triangles"
				+ (", mirrored" if m["mirrored"] else ""))
		for m in info["motions"]:
			print(f"  motion {m['name']}: {m['chunks']} chunk(s), {m['frames']} frames at {m['fps']:g} fps")
		z = info.get("zone")
		if z:
			print(f"  zone: {z['prefabs']} prefabs ({z['prefab triangles']} triangles), {z['placed']} of {z['placements']}"
				f" placements drawn ({z['placed triangles']} triangles)")
			if z["unmatched names"]:
				print(f"  placements naming no prefab here: {', '.join(z['unmatched names'])}")
		for p in info["problems"]:
			print(f"  warning: {p}")
	if args.textures:
		for p in ffxi.extract_textures(path, Path(args.textures)):
			print(f"wrote {p}")
	if args.obj:
		try:
			files = ffxi.zone_to_obj(path, Path(args.obj), args.scale, args.yaw, prefabs=args.prefabs)
		except (OSError, ValueError) as exc:
			print(f"error: {exc}", file=sys.stderr)
			return 1
		objs = [p for p in files if p.suffix == ".obj"]
		print(f"wrote {len(objs)} OBJ file(s) and {len(files) - 2 * len(objs)} texture(s) to {args.obj}")
	return 0


def _ff7(args) -> int:
	from .core import ff7

	path = Path(args.path)
	try:
		info = ff7.describe(path)
	except (OSError, ValueError, struct.error) as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 1
	matches = []
	if args.anims and info["type"] == "skeleton":
		matches = [p.name for p in ff7.animations_for(path.parent, len(info["bones"]))]
		info["animations"] = matches
	if args.json:
		print(json.dumps(info, indent=2))
		return 0
	print(f"{path}: {info['type']}")
	if info["type"] in ("skeleton", "battle skeleton"):
		for b in info["bones"]:
			res = f"  parts {', '.join(b['resources'])}" if b["resources"] else ""
			print(f"  {b['name']:<12} ← {b['parent']:<12} length {b['length']:g}{res}")
		for k, v in info.items():
			if k not in ("path", "type", "bones", "animations"):
				print(f"  {k}: {v}")
		if args.anims:
			print(f"  {len(matches)} animations with {len(info['bones'])} bones: {', '.join(matches[:30])}"
				+ (" …" if len(matches) > 30 else ""))
	else:
		for k, v in info.items():
			if k not in ("path", "type"):
				print(f"  {k}: {v}")
	return 0


def _validate(args) -> int:
	errors = audit.validate_staged_tree(Path(args.path))
	for e in errors:
		print(e)
	print(f"{len(errors)} issue(s)")
	return 1 if errors else 0


def _budgets(args) -> budget.Budgets:
	b = budget.Budgets()
	if getattr(args, "pool", None):
		b.wt_pool = budget.parse_size(args.pool)
	if getattr(args, "split", None):
		b.wt_split = args.split
	return b.override(args.budget or [])


def _budget(args) -> int:
	rep = budget.gamedir_budget(Path(args.path), _budgets(args), args.map, args.include)
	if args.json:
		print(json.dumps(rep.to_json(), indent=2))
	else:
		print(budget.format_report(rep, args.verbose))
	worst = rep.status
	return 1 if worst == budget.OVER or (args.strict and worst == budget.WARN) else 0


def _fit(args) -> int:
	src = Path(args.path)
	b = _budgets(args)
	data, res = budget.fit_pack_bytes(src.read_bytes(), b, args.max_edge)
	hb = audit.human_bytes

	def line(f: budget.PackFit) -> str:
		split = f"split {f.split}" if f.split else "does not fit"
		return f"{hb(f.data_bytes)} in ARAM, {hb(f.table_bytes + f.resident)} resident in MEM1, {split}"

	print(f"before: {line(res.before)}")
	print(f"after:  {line(res.after)}")
	for name, (ow, oh, nw, nh) in sorted(res.trimmed.items(), key=lambda t: -t[1][0] * t[1][1]):
		print(f"  {name:<16} {ow}x{oh} -> {nw}x{nh}")
	if not res.ok:
		print(f"still over budget: {res.reason}", file=sys.stderr)
	if args.output:
		Path(args.output).write_bytes(data)
		print(f"wrote {args.output} ({hb(len(data))})")
	elif res.trimmed:
		print("dry run; pass -o to write the fitted pack")
	return 0 if res.ok else 1


def _sfx(args) -> int:
	cap = budget.parse_size(args.cap) if args.cap else (sfx.MENU_MAX_FILE if args.menu else sfx.GAMEPLAY_MAX_FILE)
	jobs: list[tuple[Path, Path]] = []
	for p in map(Path, args.paths):
		if p.is_dir():
			jobs += [(f, f.relative_to(p)) for f in sorted(p.rglob("*")) if f.suffix.lower() == ".wav"]
		else:
			jobs.append((p, Path(p.name)))
	if not jobs:
		print("no .wav files", file=sys.stderr)
		return 1
	if not args.output and not args.in_place:
		print("dry run; pass -o OUT or --in-place to write", file=sys.stderr)
	hb = audit.human_bytes
	refused = before = after = 0
	for src, rel in jobs:
		try:
			data, rep = sfx.fit_wav_bytes(src.read_bytes(), cap, min_rate=args.min_rate,
				keep_stereo=args.keep_stereo, allow_8bit=not args.no_8bit, truncate=args.truncate)
		except ValueError as exc:
			print(f"  skip  {rel}: {exc}")
			refused += 1
			continue
		before += rep.before
		after += rep.after
		state = "ok" if rep.ok else "OVER"
		refused += not rep.ok
		if rep.steps or not rep.ok or args.verbose:
			print(f"  {state:>4}  {rel}: {hb(rep.before)} -> {hb(rep.after)}"
				+ (f" ({', '.join(rep.steps)})" if rep.steps else ""))
		dest = src if args.in_place else (Path(args.output) / rel if args.output else None)
		if dest is not None:
			dest.parent.mkdir(parents=True, exist_ok=True)
			dest.write_bytes(data)
	print(f"{len(jobs)} sounds, {hb(before)} -> {hb(after)} of PCM; cap {hb(cap)} per file"
		+ (f"; {refused} still over (try --truncate)" if refused else ""))
	return 1 if refused else 0


def _disc(args) -> int:
	rep = disc.disc_report(Path(args.path), args.include)
	if args.json:
		print(json.dumps(rep.to_json(), indent=2))
	else:
		print(disc.format_disc(rep, args.limit))
	return 1 if rep.on_disc > rep.capacity else 0


def _slim(args) -> int:
	from .core import studio_slim

	hb = audit.human_bytes
	src = Path(args.path)
	raw = src.read_bytes()
	try:
		before = studio_slim.studio_breakdown(raw)
		res = studio_slim.slim_studio(raw, args.step, args.stub, args.keep, tolerance=args.anim_tol,
			fit_deg=args.fit_deg, fit_units=args.fit_units, max_step=args.max_step)
	except ValueError as exc:
		print(f"{src}: {exc}", file=sys.stderr)
		return 1
	after = studio_slim.studio_breakdown(res.data)
	skins = before["skins"] + before["skin tables"]
	print(f"{src.name}: {hb(len(raw))}, lean mesh (what gc_studio/ loads) {hb(len(raw) - skins)}")
	for k, v in sorted(before.items(), key=lambda t: -t[1]):
		if k == "total" or not v:
			continue
		delta = f"  -> {hb(after[k])}" if after.get(k) != v else ""
		print(f"  {k:<16} {hb(v):>10}{delta}")
	for name, old, new in res.changed:
		print(f"  {name:<16} {old} -> {new} frames")
	if res.skipped:
		print(f"  {len(res.skipped)} sequences live in external seqgroup files and were left alone")
	if args.output:
		Path(args.output).write_bytes(res.data)
		print(f"wrote {args.output} ({hb(len(res.data))}, saved {hb(len(raw) - len(res.data))})")
	elif res.changed or res.after < len(raw):
		print(f"dry run ({hb(len(raw))} -> {hb(res.after)}); pass -o to write")
	return 0


def _add_budget_args(p) -> None:
	p.add_argument("--pool", help="world texture MEM1 pool (default 448K, GC_WT_POOL_BYTES)")
	p.add_argument("--split", type=int, choices=(64, 32, 16, 8),
		help="split dimension the pool must manage (default 64; lower accepts softer textures)")
	p.add_argument("--budget", action="append", metavar="NAME=SIZE",
		help="override any budget, e.g. studio_arena=1.5M (repeatable)")
def _extract(args) -> int:
	from .core import archives

	total = 0
	for p in args.archives:
		total += archives.extract(Path(p), Path(args.output), args.match)
	return 0 if total else 1


def _sky(args) -> int:
	opts = skybox.SkyOptions(layout=args.layout, yaw=args.yaw, fov=args.fov, exposure=args.exposure, size=args.size)
	paths = [Path(p) for p in args.inputs]
	try:
		sky = skybox.build(paths, opts, args.name)
	except (OSError, ValueError) as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 1
	out = Path(args.output)
	for rel, data in sky.files().items():
		dest = out / rel
		dest.parent.mkdir(parents=True, exist_ok=True)
		dest.write_bytes(data)
	from PIL import Image

	if args.env:
		src = skybox.open_source(paths, opts, sky.name)
		for side in skybox.SIDES:
			dest = out / "gfx/env" / f"{sky.name}{side}.tga"
			dest.parent.mkdir(parents=True, exist_ok=True)
			Image.fromarray(skybox.render_side(src, side, args.env, opts.supersample)).save(dest)
	if args.preview:
		Image.fromarray(skybox.cross_preview(sky.decoded(), args.yaw)).save(args.preview)
	total = sum(map(len, sky.cmpr.values()))
	print(f"sky {sky.name}: {sky.layout} → media/sky_{sky.name}{{{','.join(skybox.SIDES)}}}.cmp, "
		f"6 × {sky.sides['rt'].shape[0]}² CMPR, {audit.human_bytes(total)}")
	if args.env:
		print(f"  also gfx/env/{sky.name}<side>.tga at {args.env}²")
	if args.preview:
		print(f"  preview (decoded, front in the middle): {args.preview}")
	for n in sky.notes:
		print(f"  note: {n}", file=sys.stderr)
	print(f'  maps use it with worldspawn "skyname" "{sky.name}"')
	return 0


def _squeeze_opts(args):
	from .core import squeeze

	keys = set(squeeze.TECHNIQUES)
	for spec, keep in ((args.only, True), (args.skip, False)):
		for item in spec or []:
			for k in filter(None, (x.strip().lower() for x in item.split(","))):
				if k not in squeeze.TECHNIQUES:
					raise SystemExit(f"unknown technique '{k}'; known: {', '.join(squeeze.TECHNIQUES)}")
	if args.only:
		keys = {k.strip().lower() for item in args.only for k in item.split(",") if k.strip()}
	for item in args.skip or []:
		keys -= {k.strip().lower() for k in item.split(",")}
	vals = dict(squeeze.PRESETS[args.preset])
	for k in ("min_db", "skin_db", "anim_tol", "anim_fit"):
		if getattr(args, k) is not None:
			vals[k] = getattr(args, k)
	return squeeze.SqueezeOptions(**vals, yaz0=not args.no_yaz0, techniques=keys)


def _squeeze(args) -> int:
	from .core import squeeze

	root = Path(args.path)
	opt = _squeeze_opts(args)
	rep = squeeze.squeeze(root, opt, args.match)
	if args.json:
		print(json.dumps(rep.to_json(), indent=2))
	else:
		print(squeeze.format_report(rep, args.verbose))
	if args.output:
		written = squeeze.apply(root, Path(args.output), opt, args.emit, args.match)
		print(f"wrote {len(written)} files to {args.output}")
	return 0


def _adpcm(args) -> int:
	from .core import dspadpcm, squeeze

	out = Path(args.output)
	out.mkdir(parents=True, exist_ok=True)
	rc = 0
	for p in map(Path, args.paths):
		data = p.read_bytes()
		if data[:4] == b"GCPA":
			rate, pcm = squeeze.read_gcpcm(data)
			loop = (None, None)
		else:
			try:
				snd = sfx.read_wav(data)
			except ValueError as exc:
				print(f"{p.name}: {exc}", file=sys.stderr)
				rc = 1
				continue
			rate, pcm = snd.rate, squeeze._pcm16(snd)
			loop = (snd.loop_start, None if snd.loop_end is None else snd.loop_end - 1)
		ch = pcm.shape[1]
		for c in range(ch):
			st = dspadpcm.encode(pcm[:, c], rate, *loop)
			name = p.stem + (f".{'LR'[c]}" if ch == 2 else f".{c}" if ch > 2 else "") + ".dsp"
			(out / name).write_bytes(st.to_dsp())
			db = dspadpcm.snr_db(pcm[:, c], st.decode())
			print(f"{name}: {audit.human_bytes(pcm.shape[0] * 2)} PCM -> {audit.human_bytes(len(st.data) + 96)}, "
				f"{rate} Hz, SNR {db:.1f} dB{', loops' if loop[0] is not None else ''}")
	return rc


def _yaz0(args) -> int:
	from .core import yaz0

	for p in map(Path, args.paths):
		data = p.read_bytes()
		if args.decompress:
			blob = yaz0.decompress(data)
			dest = Path(args.output) / p.name.removesuffix(".yaz0") if args.output else p.with_name(p.name.removesuffix(".yaz0"))
		else:
			blob = yaz0.compress(data)
			dest = (Path(args.output) / p.name if args.output else p).with_name(p.name + ".yaz0")
		dest.parent.mkdir(parents=True, exist_ok=True)
		dest.write_bytes(blob)
		print(f"{dest}: {audit.human_bytes(len(data))} -> {audit.human_bytes(len(blob))}")
	return 0


def _image_opts(args):
	from .core.images import ImageOptions

	return ImageOptions(max_edge=args.max_edge, mipmaps=not args.no_mipmaps, alpha=args.alpha)


def _pack_report(rep) -> None:
	for label, names in (("added", rep.added), ("replaced", rep.replaced), ("removed", rep.removed)):
		if names:
			print(f"{label} {len(names)}: {', '.join(names[:12])}{' …' if len(names) > 12 else ''}")
	for w in rep.warnings:
		print(f"warning: {w}", file=sys.stderr)


def _pack_list(args) -> int:
	from .core import packtool

	entries = packtool.load(Path(args.pack))
	rs = [r for r in packtool.rows(entries) if packtool.matches(r.name, args.name)]
	if args.json:
		print(json.dumps([r.__dict__ for r in rs], indent=2))
		return 0
	hb = audit.human_bytes
	print(f"{'name':<16} {'size':>9} {'mips':>4} alpha {'tag':>6} {'bytes':>10}")
	for r in sorted(rs, key=lambda r: -r.bytes) if args.sort == "size" else rs:
		print(f"{r.name:<16} {f'{r.width}x{r.height}':>9} {r.levels:>4} {'yes' if r.alpha else '  -':>5} {r.tag:>6x} {hb(r.bytes):>10}")
	print(f"{len(rs)} textures, {hb(sum(r.bytes for r in rs))} of CMPR")
	return 0


def _pack_extract(args) -> int:
	from .core import packtool

	written = packtool.extract(packtool.load(Path(args.pack)), Path(args.output), args.name, args.all_levels)
	print(f"wrote {len(written)} PNGs to {args.output}")
	return 0 if written else 1


def _image_files(paths: list[str]) -> list[Path]:
	from .core.images import IMAGE_EXTENSIONS

	out = []
	for p in map(Path, paths):
		if p.is_dir():
			out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
		else:
			out.append(p)
	return out


def _pack_build(args) -> int:
	from .core import packtool

	rep = packtool.EditReport()
	files = _image_files(args.images)
	if not files:
		print("no images", file=sys.stderr)
		return 1
	entries = packtool.upsert([], packtool.encode_files(files, _image_opts(args), rep), rep)
	size = packtool.save(Path(args.output), entries)
	_pack_report(rep)
	print(f"wrote {args.output}: {len(entries)} textures, {audit.human_bytes(size)}")
	return 0


def _pack_add(args) -> int:
	from .core import packtool

	rep = packtool.EditReport()
	entries = packtool.upsert(packtool.load(Path(args.pack)),
		packtool.encode_files(_image_files(args.images), _image_opts(args), rep), rep)
	out = Path(args.output or args.pack)
	size = packtool.save(out, entries)
	_pack_report(rep)
	print(f"wrote {out}: {len(entries)} textures, {audit.human_bytes(size)}")
	return 0


def _pack_remove(args) -> int:
	from .core import packtool

	rep = packtool.EditReport()
	entries = packtool.remove(packtool.load(Path(args.pack)), args.names, rep)
	if not rep.removed:
		print("nothing matched", file=sys.stderr)
		return 1
	out = Path(args.output or args.pack)
	size = packtool.save(out, entries)
	_pack_report(rep)
	print(f"wrote {out}: {len(entries)} textures, {audit.human_bytes(size)}")
	return 0


def _pack_merge(args) -> int:
	from .core import packtool

	rep = packtool.EditReport()
	entries = packtool.merge([packtool.load(Path(p)) for p in args.packs], rep)
	size = packtool.save(Path(args.output), entries)
	if rep.replaced:
		print(f"replaced by a later pack: {', '.join(rep.replaced[:12])}")
	for w in rep.warnings:
		print(f"warning: {w}", file=sys.stderr)
	print(f"wrote {args.output}: {len(entries)} textures, {audit.human_bytes(size)}")
	return 0


def _add_image_opts(p) -> None:
	p.add_argument("--max-edge", type=int, default=256, help="largest texture side (power of two)")
	p.add_argument("--no-mipmaps", action="store_true", help="write level 0 only")
	p.add_argument("--alpha", choices=("auto", "on", "off"), default="auto",
		help="CMPR 1-bit alpha: auto uses it when the image has transparency")


def _add_pack_parser(sub) -> None:
	pk = sub.add_parser("pack", help="list, extract, build and edit .gct texture packs",
		description="Manage GCWT texture packs (.gct) directly. Entry names are the image file names, "
		"lowercased and cut to 15 characters; the engine reads at most 512 entries per pack.")
	ps = pk.add_subparsers(dest="pack_cmd", metavar="action", required=True)

	a = ps.add_parser("list", help="the textures in a pack with their sizes")
	a.add_argument("pack", help=".gct file")
	a.add_argument("--name", action="append", metavar="GLOB", help="only matching names (repeatable)")
	a.add_argument("--sort", choices=("pack", "size"), default="pack", help="pack order or largest first")
	a.add_argument("--json", action="store_true", help="print the list as JSON")
	a.set_defaults(fn=_pack_list)

	a = ps.add_parser("extract", help="decode textures to PNG, as the GameCube samples them")
	a.add_argument("pack", help=".gct file")
	a.add_argument("-o", "--output", required=True, help="folder for the PNGs")
	a.add_argument("--name", action="append", metavar="GLOB", help="only matching names (repeatable)")
	a.add_argument("--all-levels", action="store_true", help="also write every mip level as <name>_mip<N>.png")
	a.set_defaults(fn=_pack_extract)

	a = ps.add_parser("build", help="encode images into a new pack")
	a.add_argument("images", nargs="+", help="image files or folders")
	a.add_argument("-o", "--output", required=True, help=".gct to write")
	_add_image_opts(a)
	a.set_defaults(fn=_pack_build)

	a = ps.add_parser("add", help="add images to a pack, replacing entries with the same name")
	a.add_argument("pack", help=".gct file")
	a.add_argument("images", nargs="+", help="image files or folders")
	a.add_argument("-o", "--output", help="write here instead of updating the pack in place")
	_add_image_opts(a)
	a.set_defaults(fn=_pack_add)

	a = ps.add_parser("remove", help="drop entries by name")
	a.add_argument("pack", help=".gct file")
	a.add_argument("names", nargs="+", metavar="GLOB", help="names or globs to remove")
	a.add_argument("-o", "--output", help="write here instead of updating the pack in place")
	a.set_defaults(fn=_pack_remove)

	a = ps.add_parser("merge", help="combine packs; a later pack wins on a name clash")
	a.add_argument("packs", nargs="+", help=".gct files, in priority order (last wins)")
	a.add_argument("-o", "--output", required=True, help=".gct to write")
	a.set_defaults(fn=_pack_merge)


def build_parser() -> argparse.ArgumentParser:
	"""Every subcommand and option. docs/cli.md is generated from this (python -m gxbake.clidoc)."""
	from .core import squeeze

	parser = argparse.ArgumentParser(prog="gxbake", description="Convert and optimise assets for the GameCube GX renderer.")
	parser.add_argument("--version", action="version", version=f"GxBake {__version__}")
	sub = parser.add_subparsers(dest="cmd", metavar="command")

	c = sub.add_parser("convert", help="convert files or folders")
	c.add_argument("paths", nargs="+", help="files or folders (a gamedir, WADs, BSPs, models, images, audio, video)")
	c.add_argument("-o", "--output", required=True, help="staged output folder")
	c.add_argument("--max-edge", type=int, default=256, help="largest texture side for loose images (power of two)")
	c.add_argument("--no-mipmaps", action="store_true", help="write level 0 only")
	c.add_argument("--alpha", choices=("auto", "on", "off"), default="auto", help="CMPR 1-bit alpha: auto uses it when the image has transparency")
	c.add_argument("--no-pack", action="store_true", help="one .gct per loose image")
	c.add_argument("--pack-name", default="textures", help="loose images go to media/<name>.gct")
	c.add_argument("--raw-cmp", action="store_true", help="also write media/<name>.cmp (level 0) per loose image")
	c.add_argument("--mdl-max-edge", type=int, default=64, help="largest skin side in gc_studio/<name>t.mdl")
	c.add_argument("--mdl-anim-step", type=int, default=1,
		help="keep every Nth animation frame in gc_studio/<name>.mdl (seqgroup 0 only)")
	c.add_argument("--spr-scale", type=int, default=2, help="sprite downscale factor")
	c.add_argument("--model-scale", type=float, default=1.0, help="3D model import: source units → GoldSrc units (USD: multiplies the stage's metersPerUnit scale)")
	c.add_argument("--model-up", choices=("y", "z"), default="y", help="3D model import: source up axis (USD files use their own upAxis)")
	c.add_argument("--model-skin-max-edge", type=int, default=256, help="3D model import: largest skin side in models/<name>.mdl")
	c.add_argument("--no-model-full", action="store_true", help="skip models/<name>.mdl for imported 3D models")
	c.add_argument("--model-static", action="store_true", help="3D model import: ignore skeletons and animations")
	c.add_argument("--model-fps", type=int, default=15, help="3D model import: animation sample rate")
	c.add_argument("--no-model-clips", action="store_true", help="don't merge <name>@<clip>.<ext> animation files")
	c.add_argument("--no-blend-bones", action="store_true",
		help="Source models: snap blended vertices to one bone instead of adding helper bones")
	c.add_argument("--rose-scale", type=float, default=39.37, help="ROSE Online models: GoldSrc units per metre")
	c.add_argument("--rose-yaw", type=float, default=90.0, help="ROSE Online models: degrees to turn about Z (90 faces them +X)")
	c.add_argument("--rose-entry", type=int, action="append",
		help="ROSE .zsc/.chr: convert only this object/character index (repeatable; default all)")
	c.add_argument("--panda-scale", type=float, default=12.0,
		help="Panda3D/Toontown models: GoldSrc units per Panda unit (12: a Toontown foot)")
	c.add_argument("--panda-yaw", type=float, default=-90.0,
		help="Panda3D models: degrees to turn about Z (-90 turns Panda's +Y forward to GoldSrc's +X)")
	c.add_argument("--rs-scale", type=float, default=0.35,
		help="RuneScape caches: GoldSrc units per model unit (128 units = one tile)")
	c.add_argument("--rs-yaw", type=float, default=0.0, help="RuneScape caches: extra degrees to turn about Z")
	c.add_argument("--rs-entry", action="append", metavar="KIND:ID",
		help="RuneScape caches: convert npc:<id> or model:<id> (repeatable; default every named NPC)")
	c.add_argument("--ffxi-scale", type=float, default=39.37, help="Final Fantasy XI models: GoldSrc units per FFXI unit")
	c.add_argument("--ffxi-yaw", type=float, default=0.0, help="Final Fantasy XI models: extra degrees to turn about Z")
	c.add_argument("--ffxi-skeleton", metavar="DAT",
		help="Final Fantasy XI: race skeleton DAT for equipment and face DATs that have none of their own")
	c.add_argument("--ffxi-music-wav", action="store_true",
		help="Final Fantasy XI: write .bgw music as WAVs under sound/ffxi/ instead of media/<name>.gcpcm")
	c.add_argument("--ff7-scale", type=float, default=2.5, help="Final Fantasy VII field models: GoldSrc units per FF7 unit")
	c.add_argument("--ff7-yaw", type=float, default=0.0, help="Final Fantasy VII models: extra degrees to turn about Z")
	c.add_argument("--ff7-battle-scale", type=float, default=0.0,
		help="Final Fantasy VII battle models: GoldSrc units per FF7 unit (0 = fit each model to 72 units tall)")
	c.add_argument("--video-width", type=int, default=320, help="GCVID frame width")
	c.add_argument("--video-height", type=int, default=240, help="GCVID frame height")
	c.add_argument("--video-fps", type=int, default=15, help="GCVID frame rate")
	c.add_argument("--fit-world", action="store_true",
		help="trim world texture packs' top mips until they fit the engine's MEM1 pool")
	c.add_argument("--wt-pool", default="448K", help="pool size for --fit-world")
	c.add_argument("--wad-dir", action="append", help="extra WAD search folder (repeatable)")
	c.add_argument("--game-dir", action="append", metavar="DIR_OR_VPK",
		help="Source maps: game folder or _dir.vpk to find materials in (repeatable; the map's own pakfile and "
		"the folder above maps/ are searched first)")
	c.add_argument("--map-texture-max", type=int, default=256, help="Source maps: largest embedded texture side")
	c.add_argument("--map-disp-power", type=int, default=2, choices=range(0, 5),
		help="Source maps: displacement detail kept (2 = 5x5 vertices per displacement)")
	c.add_argument("--no-map-hulls", action="store_true",
		help="Source maps: skip building collision hulls 1-3 (faster; players then only collide with hull 0)")
	c.add_argument("--no-map-shadows", action="store_true",
		help="Source 2 maps: light without tracing shadows (faster)")
	c.add_argument("--no-map-props", action="store_true", help="Source maps: don't turn prop_static into cycler_sprite entities")
	c.add_argument("--cache", help="CMPR encode cache folder")
	c.add_argument("--no-sky-detect", action="store_true",
		help="keep big 2:1 images, crosses, strips and .hdr files as textures instead of making skies of them")
	c.add_argument("--sky-yaw", type=float, default=skybox.DEFAULT_YAW,
		help="panorama skies: engine yaw the image centre faces (0 rt, 90 bk, 180 lf, 270 ft)")
	c.add_argument("--sky-fov", type=float, default=180.0, help="flat photos as skies: horizontal degrees the photo covers")
	c.add_argument("--sky-exposure", type=float, default=0.0, help=".hdr panoramas: stops brighter (+) or darker (-) than auto")
	c.set_defaults(fn=_convert)

	i = sub.add_parser("inspect", help="print what GxBake sees in a file or folder")
	i.add_argument("path", help="file or folder")
	i.set_defaults(fn=_inspect)

	r = sub.add_parser("rs", help="list the NPCs and models in a RuneScape cache")
	r.add_argument("path", help="main_file_cache.dat2 (JS5: OSRS / RS2) or main_file_cache.dat (2004-era), or its folder")
	r.add_argument("--find", help="only NPCs whose name contains this")
	r.add_argument("--all", action="store_true", help="include NPCs named 'null' (variants the client swaps in)")
	r.add_argument("--limit", type=int, default=50, help="NPCs to print (0 = all)")
	r.add_argument("--json", action="store_true", help="print everything as JSON")
	r.set_defaults(fn=_rs)

	f7 = sub.add_parser("ff7", help="look inside Final Fantasy VII field and battle models (.hrc, .p, .a, .tex, <base>aa)")
	f7.add_argument("path", help="a skeleton (.hrc), polygon model (.p), animation (.a), texture (.tex) or battle skeleton (<base>aa)")
	f7.add_argument("--anims", action="store_true", help="for a .hrc: list the .a files beside it with the same bone count")
	f7.add_argument("--json", action="store_true", help="print as JSON")
	f7.set_defaults(fn=_ff7)

	ff = sub.add_parser("ffxi", help="look inside Final Fantasy XI DATs, sounds and file tables")
	ff.add_argument("path", help="a ROM DAT, a .spw/.bgw sound, or the FINAL FANTASY XI folder (with --id)")
	ff.add_argument("--id", type=int, help="resolve this file id through VTABLE/FTABLE and describe that DAT")
	ff.add_argument("--textures", metavar="DIR", help="write the DAT's textures as PNGs here")
	ff.add_argument("--wav", metavar="OUT", help="decode a .spw/.bgw to this WAV")
	ff.add_argument("--obj", metavar="DIR",
		help="zone DATs: write the placed zone as OBJ + MTL + PNG textures here (GoldSrc axes and units)")
	ff.add_argument("--prefabs", action="store_true", help="with --obj: one OBJ per zone prefab instead of the placed zone")
	ff.add_argument("--scale", type=float, default=39.37, help="with --obj: GoldSrc units per FFXI unit")
	ff.add_argument("--yaw", type=float, default=0.0, help="with --obj: extra degrees to turn about Z")
	ff.add_argument("--json", action="store_true", help="print as JSON")
	ff.set_defaults(fn=_ffxi)

	v = sub.add_parser("validate", help="check a staged gamedir like the disc builder does")
	v.add_argument("path", help="staged gamedir")
	v.set_defaults(fn=_validate)

	bd = sub.add_parser("budget", help="per-map memory report against the GameCube port's windows")
	bd.add_argument("path", help="staged gamedir (the output of convert)")
	bd.add_argument("--map", action="append", help="only this map (repeatable)")
	bd.add_argument("--include", action="append", metavar="GLOB",
		help="models every map loads, e.g. 'models/v_*.mdl' (repeatable)")
	bd.add_argument("--json", action="store_true", help="print the report as JSON")
	bd.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
	bd.add_argument("-v", "--verbose", action="store_true", help="list the biggest items on every line")
	_add_budget_args(bd)
	bd.set_defaults(fn=_budget)

	f = sub.add_parser("fit", help="trim a .gct's top mips until it fits the world texture budget")
	f.add_argument("path", help="world texture pack (.gct)")
	f.add_argument("-o", "--output", help="write the fitted pack here (otherwise a dry run)")
	f.add_argument("--max-edge", type=int, help="also cap every texture to this size first")
	_add_budget_args(f)
	f.set_defaults(fn=_fit)

	x = sub.add_parser("sfx", help="shrink WAVs under the port's per-file sound cap (20 KiB gameplay, 48 KiB menu)")
	x.add_argument("paths", nargs="+", help=".wav files or folders")
	x.add_argument("-o", "--output", help="write fitted WAVs here, keeping relative paths")
	x.add_argument("--in-place", action="store_true", help="overwrite the source files")
	x.add_argument("--menu", action="store_true", help="use the 48 KiB menu-route cap")
	x.add_argument("--cap", help="explicit cap, e.g. 16K")
	x.add_argument("--min-rate", type=int, default=11025, help="lowest sample rate to fall back to")
	x.add_argument("--keep-stereo", action="store_true", help="never fold stereo to mono")
	x.add_argument("--no-8bit", action="store_true", help="never drop to 8-bit samples")
	x.add_argument("--truncate", action="store_true", help="cut (with a 10 ms fade) what still does not fit")
	x.add_argument("-v", "--verbose", action="store_true", help="also list sounds that already fit")
	x.set_defaults(fn=_sfx)

	dk = sub.add_parser("disc", help="disc usage: sector cost, duplicate files and textures, files no map names")
	dk.add_argument("path", help="staged gamedir")
	dk.add_argument("--include", action="append", metavar="GLOB", help="count these as referenced (repeatable)")
	dk.add_argument("--limit", type=int, default=15, help="rows per section")
	dk.add_argument("--json", action="store_true", help="print the report as JSON")
	dk.set_defaults(fn=_disc)

	x = sub.add_parser("extract", help="unpack a Quake/GoldSrc .pak, Source .vpk, .pk3 or Panda3D .mf archive")
	x.add_argument("archives", nargs="+", help=".pak, .vpk, .pk3 or .mf (e.g. Toontown's phase_3.mf) files")
	x.add_argument("-o", "--output", required=True, help="folder to extract into")
	x.add_argument("--match", action="append", help="only members matching this glob, e.g. 'models/*' (repeatable)")
	x.set_defaults(fn=_extract)

	sl = sub.add_parser("slim", help="show where a studio model's bytes go and thin its animation")
	sl.add_argument("path", help="GoldSrc studio model (.mdl)")
	sl.add_argument("-o", "--output", help="write the slimmed model here (otherwise a dry run)")
	sl.add_argument("--step", type=int, default=1, help="keep every Nth animation frame (fps adjusted, same duration)")
	sl.add_argument("--stub", action="append", metavar="PATTERN", help="reduce matching sequences to one frame")
	sl.add_argument("--keep", action="append", metavar="PATTERN", help="never touch matching sequences")
	sl.add_argument("--anim-tol", type=int, default=0, metavar="UNITS",
		help="hold animation values within ±UNITS so runs get longer (lossy, bounded; 0 = lossless re-encode only)")
	sl.add_argument("--fit-deg", type=float, default=0, metavar="DEG",
		help="resample each sequence to the fewest evenly spaced keys that stay within DEG degrees of every bone")
	sl.add_argument("--fit-units", type=float, metavar="UNITS", help="position tolerance for --fit-deg (default DEG / 10)")
	sl.add_argument("--max-step", type=float, default=8, help="--fit-deg keeps keys at most this many frames apart")
	sl.set_defaults(fn=_slim)

	sq = sub.add_parser("squeeze", help="measure smarter packing on a staged tree, and write what the port reads today")
	sq.add_argument("path", help="staged gamedir (or one file)")
	sq.add_argument("-o", "--output", help="write the results here, same relative paths (otherwise a report only)")
	sq.add_argument("--emit", action="store_true",
		help="with -o, also write the formats that need port support: .gsk skins, .dsp sound, .yaz0")
	sq.add_argument("--preset", choices=sorted(squeeze.PRESETS), default="balanced",
		help="starting quality: lossless (bit-exact only), balanced (default: ≥40 dB, ≤0.5° animation) or max "
		"(≥34 dB, ≤2°); the options below override it")
	sq.add_argument("--min-db", type=float,
		help="drop a world texture's top mip when the next level magnified is within this PSNR (balanced: 40)")
	sq.add_argument("--anim-tol", type=int, metavar="UNITS", help="studio animation deadband (0 = lossless)")
	sq.add_argument("--anim-fit", type=float, metavar="DEG",
		help="resample studio animation to the fewest keys within DEG degrees (0 = off; balanced: 0.5)")
	sq.add_argument("--skin-db", type=float, help="lossy skin formats must stay within this PSNR of RGB565 (balanced: 40)")
	sq.add_argument("--no-yaz0", action="store_true", help="skip the (slow) Yaz0 estimate (same as --skip yaz0)")
	sq.add_argument("--only", action="append", metavar="LIST",
		help="only these techniques, comma-separated: trim, anim, skins, adpcm, verts, lightmaps, yaz0")
	sq.add_argument("--skip", action="append", metavar="LIST", help="leave these techniques out (same names as --only)")
	sq.add_argument("--match", action="append", metavar="GLOB", help="only these relative paths (repeatable)")
	sq.add_argument("--json", action="store_true", help="print every row as JSON")
	sq.add_argument("-v", "--verbose", action="store_true", help="list every file")
	sq.set_defaults(fn=_squeeze)

	ad = sub.add_parser("adpcm", help="encode WAV or .gcpcm audio as GameCube DSP ADPCM (.dsp, one per channel)")
	ad.add_argument("paths", nargs="+", help=".wav or .gcpcm files")
	ad.add_argument("-o", "--output", required=True, help="folder for the .dsp files")
	ad.set_defaults(fn=_adpcm)

	yz = sub.add_parser("yaz0", help="Yaz0-compress files (Nintendo LZ), or decompress with -d")
	yz.add_argument("paths", nargs="+", help="files")
	yz.add_argument("-d", "--decompress", action="store_true", help="decompress .yaz0 files")
	yz.add_argument("-o", "--output", help="folder to write into (default: next to the input)")
	yz.set_defaults(fn=_yaz0)

	sk = sub.add_parser("sky", help="turn a panorama, cubemap cross/strip, photo or six faces into the port's sky sides")
	sk.add_argument("inputs", nargs="+",
		help="one image (equirectangular panorama, .hdr, 4:3/3:4 cross, 6:1 strip or a photo), "
		"six face images, or a folder holding them")
	sk.add_argument("-o", "--output", required=True, help="staged gamedir; sides go to media/sky_<name><side>.cmp")
	sk.add_argument("--name", help="sky name (worldspawn \"skyname\"); default from the file name")
	sk.add_argument("--layout", choices=skybox.LAYOUTS, default="auto", help="how to read the input; auto guesses from its shape")
	sk.add_argument("--yaw", type=float, default=skybox.DEFAULT_YAW,
		help="engine yaw the image centre (or the cross's front) faces: 0 rt, 90 bk, 180 lf, 270 ft")
	sk.add_argument("--fov", type=float, default=180.0,
		help="photo layout: horizontal degrees the photo covers; mirrored copies fill the rest")
	sk.add_argument("--exposure", type=float, default=0.0, help=".hdr input: stops brighter (+) or darker (-) than auto")
	sk.add_argument("--size", type=int, default=skybox.SKY_SIZE, help="side size; the port loads only 128")
	sk.add_argument("--env", type=int, metavar="SIZE",
		help="also write gfx/env/<name><side>.tga at SIZE (PC builds, the disc builder)")
	sk.add_argument("--preview", metavar="PNG", help="write the decoded sides as a cross to check seams and orientation")
	sk.set_defaults(fn=_sky)

	_add_pack_parser(sub)

	g = sub.add_parser("gui", help="open the desktop app")
	g.add_argument("paths", nargs="*", help="files or folders to open")
	return parser


def subcommands() -> list[str]:
	parser = build_parser()
	sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
	return list(sub.choices)


def main(argv: list[str] | None = None) -> int:
	argv = sys.argv[1:] if argv is None else argv
	parser = build_parser()
	if argv and not argv[0].startswith("-") and argv[0] not in subcommands():
		argv = ["gui", *argv]	# "Open with GxBake" / drag onto the executable
	args = parser.parse_args(argv)
	if args.cmd in (None, "gui"):
		from .gui.app import main as gui_main

		return gui_main([sys.argv[0], *getattr(args, "paths", [])])
	return args.fn(args)


if __name__ == "__main__":
	raise SystemExit(main())
