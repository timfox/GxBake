"""port/gc_mat.c on the host: TPL reading and decoding match GxBake's encoder texel for texel,
.gxm materials parse, and gc_mat_setup issues the GX calls each preset needs (against a
recording stub of <ogc/gx.h>, so this checks the sequence, not the picture on hardware)."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gxbake.core import autotex, pbr, tpl

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
STUB = ROOT / "tests" / "c_stub"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler")
FLAGS = ["-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", "-O2"]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
	d = tmp_path_factory.mktemp("matbin")
	plain, gx = d / "test_gc_mat", d / "test_gc_mat_gx"
	subprocess.run([CC, *FLAGS, str(PORT / "gc_mat.c"), str(PORT / "test_gc_mat.c"), "-o", str(plain)], check=True)
	subprocess.run([CC, *FLAGS, "-DGC_MAT_GX", f"-I{STUB}", str(PORT / "gc_mat.c"), str(PORT / "test_gc_mat.c"),
		str(STUB / "gx_stub.c"), "-o", str(gx)], check=True)
	return plain, gx


def picture(w, h, seed=3):
	rng = np.random.default_rng(seed)
	y, x = np.mgrid[0:h, 0:w]
	img = np.zeros((h, w, 4), np.uint8)
	img[..., 0] = x * 255 // max(1, w - 1)
	img[..., 1] = y * 255 // max(1, h - 1)
	img[..., 2] = rng.integers(0, 256, (h, w))
	img[..., 3] = np.where((x + y) % 5 == 0, 0, np.where(x % 3 == 0, 128, 255))
	return img


@pytest.mark.parametrize("fmt", ["I4", "I8", "IA4", "IA8", "RGB565", "RGB5A3", "RGBA8", "CMPR", "CI4", "CI8"])
def test_c_tpl_decodes_like_python(harness, tmp_path, fmt):
	src = picture(24, 20)	# not a multiple of the tile size: padding must be skipped
	sizes = autotex.mip_sizes(24, 20, True, 2)
	levels = autotex.encode_levels(autotex.mip_chain(src, sizes), fmt)
	extra = tpl.TplImage([autotex.encode_levels([picture(8, 8, 9)], "RGB565")[0]], tpl.GX_CLAMP, tpl.GX_CLAMP)
	(tmp_path / "t.tpl").write_bytes(tpl.write_tpl([tpl.TplImage(levels), extra]))
	res = subprocess.run([harness[0], "tpl", tmp_path / "t.tpl", tmp_path / "out"], check=True,
		capture_output=True, text=True)
	rows = [list(map(int, r.split())) for r in res.stdout.splitlines()]
	from gxbake.core import gxtex

	assert rows[0][:4] == [gxtex.FORMATS[fmt][0], 24, 20, len(levels)]
	assert rows[1] == [4, 8, 8, 1, 0, 0, 0]
	got = np.fromfile(tmp_path / "out", np.uint8)
	pos = 0
	for lv in [*levels, extra.levels[0]]:
		n = lv.width * lv.height * 4
		assert np.array_equal(got[pos:pos + n].reshape(lv.height, lv.width, 4), lv.decode()), (fmt, lv.width)
		pos += n
	assert pos == len(got)


def test_c_tpl_rejects_truncated(harness, tmp_path):
	data = tpl.write_tpl([tpl.TplImage([autotex.encode_levels([picture(16, 16)], "RGBA8")[0]])])
	(tmp_path / "t.tpl").write_bytes(data[:-100])
	assert subprocess.run([harness[0], "tpl", tmp_path / "t.tpl", tmp_path / "o"]).returncode == 1


def test_c_gxm_and_sphere_matrix(harness, tmp_path):
	(tmp_path / "m.gxm").write_bytes(pbr.gxm_bytes("indirect", ["base", "spec", "normal", "env"], "mask", 0.5))
	res = subprocess.run([harness[0], "gxm", tmp_path / "m.gxm"], check=True, capture_output=True, text=True)
	head, roles = res.stdout.splitlines()
	assert head.split() == ["3", "1", "128", "1", "0"]
	assert roles.split() == ["0", "1", "-1", "-1", "2", "3", "-1"]
	res = subprocess.run([harness[0], "sphere"], check=True, capture_output=True, text=True)
	m = np.array([list(map(float, r.split())) for r in res.stdout.splitlines()])
	n = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
	assert np.allclose(m[0, :3], 0.5 * n[0]) and np.allclose(m[1, :3], -0.5 * n[1]) and np.allclose(m[:2, 3], 0.5)


def setup_calls(harness, tmp_path, preset, maps, *extra, **kw):
	(tmp_path / "m.gxm").write_bytes(pbr.gxm_bytes(preset, maps, **kw))
	res = subprocess.run([harness[1], "setup", tmp_path / "m.gxm", *extra], check=True, capture_output=True, text=True)
	lines = res.stdout.splitlines()
	return lines, int(lines[-1].split()[1])


@pytest.mark.parametrize("preset,maps,stages", [
	("prelit", ["base"], 1),
	("lit", ["base", "spec"], 2),
	("lit", ["base", "spec", "emit"], 3),
	("emboss", ["base", "spec", "height"], 5),
	("indirect", ["base", "spec", "normal", "env"], 4),
	("indirect", ["base", "normal", "env"], 2),
])
def test_c_setup_stage_counts(harness, tmp_path, preset, maps, stages):
	lines, n = setup_calls(harness, tmp_path, preset, maps)
	assert n == stages and f"SetNumTevStages {stages}" in lines
	assert sum(1 for line in lines if line.startswith("LoadTexObj")) == len(maps)
	if preset == "emboss":
		assert any(line.startswith("SetTexCoordGen 1 2 112") for line in lines)	# GX_TG_BUMP0 from texcoord 0
		assert "SetTevColorOp 1 1 1 0 1 1" in lines	# SUB, ADDHALF into C0
	if preset == "indirect":
		assert "SetNumIndStages 1" in lines and any(line.startswith("SetTevIndirect") for line in lines)
		assert any(line.startswith("SetTexCoordGen 1 1 101 30") for line in lines)	# GX_TG_NRM via GX_TEXMTX0
	if "spec" in maps:
		assert "SetNumChans 2" in lines


def test_c_setup_lightmap_scale_and_alpha(harness, tmp_path):
	lines, n = setup_calls(harness, tmp_path, "lightmap", ["lightmap"], tev_scale=4)
	assert n == 2 and "SetNumChans 0" in lines and "SetTevColorOp 1 0 0 2 1 0" in lines	# ×4
	lines, _ = setup_calls(harness, tmp_path, "prelit", ["base"], "unlit", alpha_mode="mask", cutoff=0.5)
	assert "SetNumChans 0" in lines and "SetAlphaCompare 6 128 0 7 0" in lines and "SetZCompLoc 0" in lines
	lines, _ = setup_calls(harness, tmp_path, "lit", ["base"], alpha_mode="blend")
	assert "SetBlendMode 1 4 5 0" in lines


def test_build_writes_gxm(tmp_path):
	from tests.test_pbr_hdr import make_project
	from gxbake.core import build

	man = make_project(tmp_path)
	build.build(man)
	m = (tmp_path / "out" / "bricks.gxm").read_bytes()
	assert m[:4] == b"GXM1" and m[4] == 1 and m[8] == 2	# lit, base + spec
	lm = (tmp_path / "out" / "room.gxm").read_bytes()
	assert lm[4] == 4 and lm[7] == 2	# lightmap, TEV ×2


def test_cli_tpl_and_watch_once(tmp_path, capsys):
	from gxbake import cli
	from tests.test_pbr_hdr import make_project

	man = make_project(tmp_path)
	assert cli.main(["build", str(man), "--watch", "--once"]) == 0
	capsys.readouterr()
	out = tmp_path / "out"
	assert cli.main(["tpl", str(out / "bricks.tpl"), str(out / "bricks.gxm"), "--extract", str(tmp_path / "png")]) == 0
	text = capsys.readouterr().out
	assert "TPL, 2 image(s)" in text and "preset lit" in text and "base=#0" in text
	assert (tmp_path / "png" / "bricks_0.png").is_file()
