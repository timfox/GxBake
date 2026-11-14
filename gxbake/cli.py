"""Command-line front end: the same pipeline as the GUI, for scripts and CI.

	gxbake convert <files or folders...> -o OUT [options]
	gxbake inspect <file>
	gxbake validate <staged gamedir>
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


def main(argv: list[str] | None = None) -> int:
	argv = sys.argv[1:] if argv is None else argv
	if argv and not argv[0].startswith("-") and argv[0] not in ("convert", "inspect", "validate", "gui"):
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

	g = sub.add_parser("gui", help="open the desktop app")
	g.add_argument("paths", nargs="*")

	args = parser.parse_args(argv)
	if args.cmd in (None, "gui"):
		from .gui.app import main as gui_main

		return gui_main([sys.argv[0], *getattr(args, "paths", [])])
	return args.fn(args)


if __name__ == "__main__":
	raise SystemExit(main())
