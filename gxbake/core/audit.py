"""GameCube budgets and source-tree checks.

The constants and validate_staged_tree() mirror the checks in
scripts/build-gamecube-disc.py (validate_staged_retail_assets): strict
lowercase directories, unsupported extensions and the 10 MiB per-asset cap.
"""

from __future__ import annotations

import os
from pathlib import Path

MAIN_RAM = 24 * 1024 * 1024	# 1T-SRAM
ARAM = 16 * 1024 * 1024
TEXTURE_CACHE = 1024 * 1024	# TMEM; one map's resident textures want to fit here
MAX_ASSET_SIZE = 10 * 1024 * 1024
PACK_EXTENSIONS = {".bsp", ".wad", ".pak", ".pk3"}

UNSUPPORTED_EXTENSIONS = {
	".wmv", ".mpg", ".mpeg",	# video: unsupported or too heavy
	".mp3", ".ogg",	# audio: streaming not yet stable on GC
}

STRICT_CASE_DIRS = ("maps", "models", "sprites", "decals", "sound", "gfx", "lang")

CRITICAL_ASSETS = (
	"liblist.gam",
	"delta.lst",
	"gfx.wad",
	"gfx/palette.lmp",
	"gfx/conback.lmp",
	"gfx/colormap.lmp",
	"valve.rc",
	"default.cfg",
	"config.cfg",
)


def in_strict_case_dir(rel_root: str) -> bool:
	if not rel_root or rel_root == ".":
		return False
	return any(rel_root == strict or rel_root.startswith(f"{strict}/") for strict in STRICT_CASE_DIRS)


def file_warnings(path: Path, rel_path: str | None = None) -> list[str]:
	"""Per-file warnings shown in the asset list."""
	out = []
	ext = path.suffix.lower()
	try:
		size = path.stat().st_size
	except OSError:
		return ["cannot stat file"]
	if ext in UNSUPPORTED_EXTENSIONS:
		out.append(f"'{ext}' is not loadable on GameCube as-is; convert it")
	if size > MAX_ASSET_SIZE and ext not in PACK_EXTENSIONS:
		out.append(f"{size / 1048576:.1f} MiB exceeds the {MAX_ASSET_SIZE // 1048576} MiB per-asset cap")
	if rel_path:
		parent = os.path.dirname(rel_path)
		if in_strict_case_dir(parent) and path.name != path.name.lower():
			out.append("mixed-case name inside a case-strict directory")
	return out


def validate_staged_tree(data_path: Path) -> list[str]:
	"""Validate a staged gamedir (what the disc builder checks before mastering)."""
	errors = []
	if not data_path.is_dir():
		return [f"Data directory not found: {data_path}"]
	for asset in CRITICAL_ASSETS:
		if not (data_path / asset).is_file():
			errors.append(f"MISSING: critical asset '{asset}' is not present.")
	for root, _dirs, files in os.walk(data_path):
		rel_root = os.path.relpath(root, data_path)
		if rel_root == ".":
			rel_root = ""
		rel_root = rel_root.replace("\\", "/")
		for filename in files:
			rel_path = f"{rel_root}/{filename}" if rel_root else filename
			for w in file_warnings(Path(root) / filename, rel_path):
				errors.append(f"{rel_path}: {w}")
	return errors


def human_bytes(n: int) -> str:
	for unit in ("B", "KiB", "MiB", "GiB"):
		if abs(n) < 1024 or unit == "GiB":
			return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
		n /= 1024
	return f"{n} B"
