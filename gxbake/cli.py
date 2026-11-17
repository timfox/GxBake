"""Command-line front end: the same pipeline as the GUI, for scripts and CI.

	gxbake convert <files or folders...> -o OUT [options]
	gxbake inspect <file>
	gxbake validate <staged gamedir>
	gxbake budget <staged gamedir> [--map NAME] [--json]
	gxbake fit <pack.gct> [-o OUT] [--pool 448K]
	gxbake disc <staged gamedir> [--json]
	gxbake sfx <wavs or folders...> -o OUT [--menu] [--truncate]
	gxbake extract <archive.pak|_dir.vpk|.pk3> -o OUT [--match GLOB]
	gxbake gui [files...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
import json

from .core import audit, budget, disc, sfx
from .core.project import CONVERTIBLE, Project, Settings


def _convert(args) -> int:
	s = Settings(cache_dir=args.cache or "", wad_dirs=args.wad_dir or [])
	s.image.max_edge = args.max_edge
	s.image.mipmaps = not args.no_mipmaps
	s.image.alpha = args.alpha
	s.pack_images = not args.no_pack
	s.pack_name = args.pack_name
	s.raw_cmp = args.raw_cmp
	s.mdl_max_edge = args.mdl_max_edge
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
	s.video_width, s.video_height, s.video_fps = args.video_width, args.video_height, args.video_fps
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


def main(argv: list[str] | None = None) -> int:
	argv = sys.argv[1:] if argv is None else argv
	if argv and not argv[0].startswith("-") and argv[0] not in ("convert", "inspect", "validate", "budget", "fit", "sfx", "disc", "gui"):
	if argv and not argv[0].startswith("-") and argv[0] not in ("convert", "inspect", "validate", "extract", "gui"):
		argv = ["gui", *argv]	# "Open with GxBake" / drag onto the executable
	parser = argparse.ArgumentParser(prog="gxbake", description="Convert and optimise assets for the GameCube GX renderer.")
	parser.add_argument("--version", action="version", version=f"GxBake {__version__}")
	sub = parser.add_subparsers(dest="cmd")

	c = sub.add_parser("convert", help="convert files or folders")
	c.add_argument("paths", nargs="+")
	c.add_argument("-o", "--output", required=True)
	c.add_argument("--max-edge", type=int, default=256)
	c.add_argument("--no-mipmaps", action="store_true")
	c.add_argument("--alpha", choices=("auto", "on", "off"), default="auto")
	c.add_argument("--no-pack", action="store_true", help="one .gct per loose image")
	c.add_argument("--pack-name", default="textures")
	c.add_argument("--raw-cmp", action="store_true")
	c.add_argument("--mdl-max-edge", type=int, default=64)
	c.add_argument("--spr-scale", type=int, default=2)
	c.add_argument("--model-scale", type=float, default=1.0, help="3D model import: source units → GoldSrc units")
	c.add_argument("--model-up", choices=("y", "z"), default="y", help="3D model import: source up axis")
	c.add_argument("--model-skin-max-edge", type=int, default=256)
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
	c.add_argument("--video-width", type=int, default=320)
	c.add_argument("--video-height", type=int, default=240)
	c.add_argument("--video-fps", type=int, default=15)
	c.add_argument("--fit-world", action="store_true",
		help="trim world texture packs' top mips until they fit the engine's MEM1 pool")
	c.add_argument("--wt-pool", default="448K", help="pool size for --fit-world")
	c.add_argument("--wad-dir", action="append", help="extra WAD search folder (repeatable)")
	c.add_argument("--cache", help="CMPR encode cache folder")
	c.set_defaults(fn=_convert)

	i = sub.add_parser("inspect", help="print what GxBake sees in a file or folder")
	i.add_argument("path")
	i.set_defaults(fn=_inspect)

	v = sub.add_parser("validate", help="check a staged gamedir like the disc builder does")
	v.add_argument("path")
	v.set_defaults(fn=_validate)

	bd = sub.add_parser("budget", help="per-map memory report against the GameCube port's windows")
	bd.add_argument("path", help="staged gamedir (the output of convert)")
	bd.add_argument("--map", action="append", help="only this map (repeatable)")
	bd.add_argument("--include", action="append", metavar="GLOB",
		help="models every map loads, e.g. 'models/v_*.mdl' (repeatable)")
	bd.add_argument("--json", action="store_true")
	bd.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
	bd.add_argument("-v", "--verbose", action="store_true", help="list the biggest items on every line")
	_add_budget_args(bd)
	bd.set_defaults(fn=_budget)

	f = sub.add_parser("fit", help="trim a .gct's top mips until it fits the world texture budget")
	f.add_argument("path")
	f.add_argument("-o", "--output", help="write the fitted pack here (otherwise a dry run)")
	f.add_argument("--max-edge", type=int, help="also cap every texture to this size first")
	_add_budget_args(f)
	f.set_defaults(fn=_fit)

	x = sub.add_parser("sfx", help="shrink WAVs under the port's per-file sound cap (20 KiB gameplay, 48 KiB menu)")
	x.add_argument("paths", nargs="+")
	x.add_argument("-o", "--output", help="write fitted WAVs here, keeping relative paths")
	x.add_argument("--in-place", action="store_true")
	x.add_argument("--menu", action="store_true", help="use the 48 KiB menu-route cap")
	x.add_argument("--cap", help="explicit cap, e.g. 16K")
	x.add_argument("--min-rate", type=int, default=11025, help="lowest sample rate to fall back to")
	x.add_argument("--keep-stereo", action="store_true")
	x.add_argument("--no-8bit", action="store_true")
	x.add_argument("--truncate", action="store_true", help="cut (with a 10 ms fade) what still does not fit")
	x.add_argument("-v", "--verbose", action="store_true", help="also list sounds that already fit")
	x.set_defaults(fn=_sfx)

	dk = sub.add_parser("disc", help="disc usage: sector cost, duplicate files and textures, files no map names")
	dk.add_argument("path")
	dk.add_argument("--include", action="append", metavar="GLOB", help="count these as referenced (repeatable)")
	dk.add_argument("--limit", type=int, default=15, help="rows per section")
	dk.add_argument("--json", action="store_true")
	dk.set_defaults(fn=_disc)
	x = sub.add_parser("extract", help="unpack a Quake/GoldSrc .pak, Source .vpk or .pk3 archive")
	x.add_argument("archives", nargs="+")
	x.add_argument("-o", "--output", required=True)
	x.add_argument("--match", action="append", help="only members matching this glob, e.g. 'models/*' (repeatable)")
	x.set_defaults(fn=_extract)

	g = sub.add_parser("gui", help="open the desktop app")
	g.add_argument("paths", nargs="*")

	args = parser.parse_args(argv)
	if args.cmd in (None, "gui"):
		from .gui.app import main as gui_main

		return gui_main([sys.argv[0], *getattr(args, "paths", [])])
	return args.fn(args)


if __name__ == "__main__":
	raise SystemExit(main())
