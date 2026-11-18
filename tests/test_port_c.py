"""port/gc_pack.c, compiled for the host, decodes GxBake's output bit-exactly."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gxbake.core import dspadpcm, squeeze, yaz0
from tests import fixtures

PORT = Path(__file__).resolve().parent.parent / "port"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler")


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
	exe = tmp_path_factory.mktemp("cbin") / "test_gc_pack"
	subprocess.run([CC, "-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", "-O2",
		str(PORT / "gc_pack.c"), str(PORT / "test_gc_pack.c"), "-o", str(exe)], check=True)
	return exe


@pytest.mark.parametrize("data", [b"", b"q", b"abcabcabc" * 900, bytes(range(256)) * 30, bytes(5000)])
def test_c_yaz0(harness, tmp_path, data):
	src, out = tmp_path / "in.yaz0", tmp_path / "out"
	src.write_bytes(yaz0.compress(data))
	subprocess.run([harness, "yaz0", src, out], check=True)
	assert out.read_bytes() == data


def test_c_yaz0_rejects_corrupt(harness, tmp_path):
	blob = bytearray(yaz0.compress(b"hello hello hello hello"))
	blob[4:8] = (10_000).to_bytes(4, "big")	# claims more than the stream holds
	(tmp_path / "bad").write_bytes(bytes(blob))
	assert subprocess.run([harness, "yaz0", tmp_path / "bad", tmp_path / "o"]).returncode == 1


def test_c_dsp_matches_python(harness, tmp_path):
	rng = np.random.default_rng(5)
	x = (9000 * np.sin(np.arange(7001) / 11) + rng.normal(0, 900, 7001)).astype(np.int16)
	st = dspadpcm.encode(x, 22050, loop_start=700, loop_end=7000)
	(tmp_path / "a.dsp").write_bytes(st.to_dsp())
	res = subprocess.run([harness, "dsp", tmp_path / "a.dsp", tmp_path / "a.raw"], check=True, capture_output=True, text=True)
	assert res.stdout.split() == ["7001", "22050", "1", "700", "7000", str(st.loop_ctx[0])]
	assert (np.fromfile(tmp_path / "a.raw", "<i2") == st.decode()).all()


def test_c_gsk_table(harness, tmp_path):
	raw = fixtures.studio([("skin", 32, 32, 0), ("mask", 16, 16, 0x40)])
	skins = squeeze.studio_skins(raw)
	blob = squeeze.write_skin_pack(skins)
	(tmp_path / "m.gsk").write_bytes(blob)
	res = subprocess.run([harness, "gsk", tmp_path / "m.gsk"], check=True, capture_output=True, text=True)
	rows = [list(map(int, line.split())) for line in res.stdout.splitlines()]
	assert len(rows) == len(skins)
	for i, (row, s) in enumerate(zip(rows, skins)):
		t = s.choice.tex
		assert row[:5] == [i, squeeze.gxtex.FORMATS[t.fmt][0], t.tlut_fmt, t.width, t.height]
		assert row[7:9] == [len(t.data), len(t.tlut)] and row[9] % 32 == 0
		assert blob[row[9]:row[9] + row[7]] == t.data
