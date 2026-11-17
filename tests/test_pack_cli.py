import json

import numpy as np
from PIL import Image

from gxbake import cli, clidoc
from gxbake.core import gct, packtool


def png(path, w=32, h=32, seed=0, alpha=False):
	rng = np.random.default_rng(seed)
	rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
	rgba[..., 3] = 255
	if alpha:
		rgba[: h // 2, :, 3] = 0
	Image.fromarray(rgba).save(path)
	return path


def test_pack_build_list_extract_add_remove_merge(tmp_path, capsys):
	src = tmp_path / "src"
	src.mkdir()
	png(src / "Wall01.png", 64, 64, 1)
	png(src / "{fence.png", 32, 32, 2, alpha=True)
	png(src / "a_really_long_texture_name.png", 16, 16, 3)
	pack = tmp_path / "out.gct"
	assert cli.main(["pack", "build", str(src), "-o", str(pack), "--max-edge", "32"]) == 0
	entries = packtool.load(pack)
	names = {e.name: e for e in entries}
	assert set(names) == {"wall01", "{fence", "a_really_long_t"}
	assert (names["wall01"].width, names["wall01"].height) == (32, 32)	# capped
	assert names["{fence"].alpha and not names["wall01"].alpha
	assert "15 characters" in capsys.readouterr().err

	assert cli.main(["pack", "list", str(pack), "--json"]) == 0
	rows = json.loads(capsys.readouterr().out)
	assert {r["name"] for r in rows} == set(names)
	assert cli.main(["pack", "list", str(pack), "--sort", "size"]) == 0
	assert "3 textures" in capsys.readouterr().out

	out = tmp_path / "png"
	assert cli.main(["pack", "extract", str(pack), "-o", str(out), "--all-levels", "--name", "wall*"]) == 0
	files = sorted(p.name for p in out.iterdir())
	assert files[0] == "wall01.png" and "wall01_mip1.png" in files
	assert np.array_equal(np.asarray(Image.open(out / "wall01.png")), names["wall01"].decode(0))
	capsys.readouterr()

	png(tmp_path / "wall01.png", 16, 16, 9)
	png(tmp_path / "new.png", 8, 8, 4)
	assert cli.main(["pack", "add", str(pack), str(tmp_path / "wall01.png"), str(tmp_path / "new.png")]) == 0
	text = capsys.readouterr().out
	assert "replaced 1: wall01" in text and "added 1: new" in text
	after = {e.name: e for e in packtool.load(pack)}
	assert after["wall01"].width == 16 and [e.name for e in packtool.load(pack)][0] == "wall01"	# replaced in place

	assert cli.main(["pack", "remove", str(pack), "a_*", "-o", str(tmp_path / "less.gct")]) == 0
	assert {e.name for e in packtool.load(tmp_path / "less.gct")} == {"wall01", "{fence", "new"}
	assert cli.main(["pack", "remove", str(pack), "nomatch"]) == 1
	capsys.readouterr()

	other = tmp_path / "other.gct"
	assert cli.main(["pack", "build", str(png(tmp_path / "new.png", 8, 8, 5)), "-o", str(other)]) == 0
	merged = tmp_path / "merged.gct"
	assert cli.main(["pack", "merge", str(pack), str(tmp_path / "less.gct"), str(other), "-o", str(merged)]) == 0
	m = packtool.load(merged)
	assert len(m) == 4	# identical entries kept once
	assert "replaced by a later pack: new" in capsys.readouterr().out
	assert next(e for e in m if e.name == "new").data == packtool.load(other)[0].data


def test_pack_entry_limit_warning():
	rep = packtool.EditReport()
	e = gct.GctEntry("x", 8, 8, 1, 0, 0, bytes(32))
	packtool.upsert([], [gct.GctEntry(f"t{i}", 8, 8, 1, 0, 0, bytes(32)) for i in range(513)] + [e], rep)
	assert any("512" in w for w in rep.warnings)


def test_subcommands_come_from_the_parser():
	cmds = cli.subcommands()
	assert {"convert", "budget", "fit", "sfx", "disc", "slim", "pack", "extract", "gui"} <= set(cmds)


def test_cli_doc_is_current():
	assert clidoc.main(["--check"]) == 0, "run: python -m gxbake.clidoc"
	text = clidoc.render()
	for cmd in cli.subcommands():
		assert f"## `gxbake {cmd}`" in text
	assert "`studio_arena`" in text
