"""Command-line front end: the same pipeline as the GUI, for scripts and CI.

	gxbake convert <files or folders...> -o OUT [options]
	gxbake inspect <file>
	gxbake validate <staged gamedir>
	gxbake extract <archive.pak|_dir.vpk|.pk3> -o OUT [--match GLOB]
	gxbake gui [files...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core import audit
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


def _extract(args) -> int:
	from .core import archives

	total = 0
	for p in args.archives:
		total += archives.extract(Path(p), Path(args.output), args.match)
	return 0 if total else 1


def main(argv: list[str] | None = None) -> int:
	argv = sys.argv[1:] if argv is None else argv
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
	c.add_argument("--wad-dir", action="append", help="extra WAD search folder (repeatable)")
	c.add_argument("--cache", help="CMPR encode cache folder")
	c.set_defaults(fn=_convert)

	i = sub.add_parser("inspect", help="print what GxBake sees in a file or folder")
	i.add_argument("path")
	i.set_defaults(fn=_inspect)

	v = sub.add_parser("validate", help="check a staged gamedir like the disc builder does")
	v.add_argument("path")
	v.set_defaults(fn=_validate)

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
