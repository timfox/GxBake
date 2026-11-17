import json
import struct

import numpy as np
import pytest

from gxbake import cli
from gxbake.core import budget, cmpr, gct

from . import fixtures as fx


def entry(name: str, w: int, h: int, mips: bool = True) -> gct.GctEntry:
	levels, size = 0, 0
	while True:
		size += cmpr.cmpr_size(max(8, w >> levels), max(8, h >> levels))
		levels += 1
		if not mips or min(w >> (levels - 1), h >> (levels - 1)) <= 8:
			break
	rng = np.random.default_rng(len(name))
	return gct.GctEntry(name, w, h, levels, 0, 0, rng.integers(0, 256, size, dtype=np.uint8).tobytes())


def test_parse_size():
	assert budget.parse_size("448K") == 448 * 1024
	assert budget.parse_size("1.5M") == 1536 * 1024
	assert budget.parse_size("6MiB") == 6 * 1024 * 1024
	assert budget.parse_size("2048") == 2048
	with pytest.raises(ValueError):
		budget.parse_size("lots")
	b = budget.Budgets().override(["studio-arena=2M"])
	assert b.studio_arena == 2 * 1024 * 1024
	with pytest.raises(ValueError):
		budget.Budgets().override(["nope=1"])


def test_simulate_matches_engine_split_rule():
	# Small textures stay whole; 128² textures lose level 0 (128*128/2 bytes) from MEM1.
	pack = gct.write_gct([entry("a", 32, 32), entry("b", 128, 128), entry("c", 128, 64)])
	fit = budget.simulate_world_pack(pack)
	assert fit.split == 64
	first = (12 + 3 * 28 + 31) & ~31
	assert fit.table_bytes == first
	chains = [len(entry(n, w, h).data) for n, w, h in (("a", 32, 32), ("b", 128, 128), ("c", 128, 64))]
	assert fit.data_bytes == sum(chains)
	assert fit.resident == chains[0] + chains[1] - 128 * 128 // 2 + chains[2] - 128 * 64 // 2
	assert fit.hot == 448 * 1024 - first - fit.resident


def test_simulate_softer_split_under_pressure():
	# 64² textures are whole at split 64 but split at 32: a tight pool picks 32.
	entries = [entry(f"t{i}", 64, 64) for i in range(100)]
	pack = gct.write_gct(entries)
	b = budget.Budgets(wt_pool=200 * 1024)
	fit = budget.simulate_world_pack(pack, b)
	assert fit.resident_by_split[64] > fit.resident_by_split[32]
	assert fit.split == 32


def test_drop_top_mip_is_level_one():
	y, x = np.mgrid[0:64, 0:64]
	rgb = np.stack((x * 4, y * 4, (x + y) * 2), -1).astype(np.uint8)
	levels, data = gct.encode_mip_chain(rgb, np.ones((64, 64), bool))
	e = gct.GctEntry("grad", 64, 64, levels, 0, 7, data)
	d = budget.drop_top_mip(e)
	assert (d.width, d.height, d.levels, d.tag) == (32, 32, levels - 1, 7)
	assert (d.decode(0) == e.decode(1)).all()
	with pytest.raises(ValueError):
		budget.drop_top_mip(entry("flat", 64, 64, mips=False))


def test_fit_trims_largest_first_until_it_fits():
	entries = [entry(f"big{i}", 256, 256) for i in range(40)] + [entry("small", 32, 32)]
	before = budget.simulate_world_pack(gct.write_gct(entries))
	assert before.split is None
	res = budget.fit_world_pack(entries)
	assert res.ok and res.after.split == 64
	assert "small" not in res.trimmed
	assert all(v[2] < 256 for k, v in res.trimmed.items())
	again = budget.simulate_world_pack(gct.write_gct(res.entries))
	assert again.split == 64 and again.data_bytes == res.after.data_bytes


def test_fit_reports_untrimmable():
	entries = [entry(f"flat{i}", 256, 256, mips=False) for i in range(20)]
	res = budget.fit_world_pack(entries)
	assert not res.ok and "mip" in res.reason


def test_fit_max_edge():
	res = budget.fit_world_pack([entry("a", 512, 256), entry("b", 64, 64)], max_edge=128)
	assert res.trimmed == {"a": (512, 256, 128, 64)}


ENTITIES = """{
"classname" "worldspawn"
"skyname" "desert"
}
{
"classname" "monster_barney"
"origin" "0 0 0"
}
{
"classname" "ambient_generic"
"message" "ambience/Wind.wav"
}
{
"classname" "env_sprite"
"model" "sprites/glow.spr"
}
{
"classname" "func_door"
"model" "*1"
}
{
"classname" "cycler"
"model" "models/missing.mdl"
}
{
"classname" "scripted_sentence"
"sentence" "!BA_HELLO"
}
"""


def staged(tmp_path, barney_size: int):
	g = tmp_path / "valve"
	(g / "maps").mkdir(parents=True)
	(g / "maps" / "c0a0.bsp").write_bytes(fx.bsp30(ENTITIES, []))
	(g / "gc_studio").mkdir()
	(g / "gc_studio" / "barney.mdl").write_bytes(bytes(barney_size))
	(g / "gc_studio" / "barneyt.mdl").write_bytes(bytes(1000))
	(g / "sound" / "ambience").mkdir(parents=True)
	(g / "sound" / "ambience" / "wind.wav").write_bytes(bytes(5000))
	(g / "sprites").mkdir()
	(g / "sprites" / "glow.spr").write_bytes(bytes(300))
	(g / "media").mkdir()
	(g / "media" / "wt_c0a0.gct").write_bytes(gct.write_gct([entry("wall", 128, 128)]))
	for sfx in ("rt", "bk", "lf", "ft", "up", "dn"):
		(g / "media" / f"sky_desert{sfx}.cmp").write_bytes(bytes(8192))
	return g


def test_entity_references():
	refs = budget.map_references(budget.parse_entities(ENTITIES))
	assert refs.models == ["models/barney.mdl", "models/missing.mdl"]
	assert refs.sounds == ["ambience/Wind.wav"]
	assert refs.sprites == ["sprites/glow.spr"]


def test_map_budget(tmp_path):
	g = staged(tmp_path, 20_000)
	rep = budget.map_budget(g, g / "maps" / "c0a0.bsp")
	lines = {ln.pool: ln for ln in rep.lines}
	assert lines["World textures (MEM1 pool)"].status == budget.OK
	assert lines["Studio models (MEM1 arena)"].used == 21_000
	assert lines["Entity sounds"].used == 5000	# matched case-insensitively
	assert lines["Sky"].used == 6 * 8192
	assert rep.missing == ["models/missing.mdl"]
	assert rep.status == budget.WARN

	g2 = staged(tmp_path / "b", 2 * 1024 * 1024)
	rep = budget.map_budget(g2, g2 / "maps" / "c0a0.bsp")
	assert rep.status == budget.OVER


def test_budget_and_fit_cli(tmp_path, capsys):
	g = staged(tmp_path, 2 * 1024 * 1024)
	assert cli.main(["budget", str(g)]) == 1
	text = capsys.readouterr().out
	assert "OVER" in text and "barney" in text
	assert cli.main(["budget", str(g), "--budget", "studio_arena=4M"]) == 0
	capsys.readouterr()
	assert cli.main(["budget", str(g), "--budget", "studio_arena=4M", "--strict"]) == 1
	capsys.readouterr()
	cli.main(["budget", str(g), "--json", "--budget", "studio_arena=4M"])
	data = json.loads(capsys.readouterr().out)
	assert data["maps"][0]["map"] == "c0a0"

	pack = tmp_path / "wt_big.gct"
	pack.write_bytes(gct.write_gct([entry(f"big{i}", 256, 256) for i in range(40)]))
	out = tmp_path / "fitted.gct"
	assert cli.main(["fit", str(pack)]) == 0	# dry run
	assert not out.exists()
	assert cli.main(["fit", str(pack), "-o", str(out)]) == 0
	assert budget.simulate_world_pack(out.read_bytes()).split == 64


# --- sound effects --------------------------------------------------------------

from gxbake.core import sfx  # noqa: E402


def tone(seconds: float, rate: int = 22050, channels: int = 1, width: int = 2, loop=None) -> bytes:
	t = np.arange(int(seconds * rate)) / rate
	x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None].repeat(channels, 1)
	ls, le = loop if loop else (None, None)
	return sfx.write_wav(sfx.Sound(rate, x, width, ls, le))


def test_wav_roundtrip_and_loop_points():
	data = tone(0.5, loop=(1000, 5000))
	snd = sfx.read_wav(data)
	assert (snd.rate, snd.channels, snd.width, snd.loop_start, snd.loop_end) == (22050, 1, 2, 1000, 5000)
	# Same offsets the engine reads: dwSampleOffset at cue+32, "mark" at LIST+28.
	cue = data.index(b"cue ")
	assert struct.unpack_from("<I", data, cue + 32)[0] == 1000
	lst = data.index(b"LIST")
	assert data[lst + 28:lst + 32] == b"mark" and struct.unpack_from("<I", data, lst + 24)[0] == 4000


def test_fit_sound_ladder():
	# 1 s mono 16-bit 22 kHz = 44 KiB → 8-bit 11 kHz = ~10.8 KiB fits 20 KiB? 8-bit 22k (21.5 KiB) does not.
	data = tone(1.0)
	out, rep = sfx.fit_wav_bytes(data, 20 * 1024)
	snd = sfx.read_wav(out)
	assert rep.ok and snd.pcm_bytes <= 20 * 1024
	assert (snd.rate, snd.width) == (11025, 2) or (snd.rate, snd.width) == (11025, 1)
	# Under the 48 KiB menu cap it stays 22 kHz 16 bit.
	out, rep = sfx.fit_wav_bytes(data, 48 * 1024)
	assert out == data and rep.steps == []
	# Stereo is folded first.
	out, rep = sfx.fit_wav_bytes(tone(0.4, channels=2), 20 * 1024)
	assert "mono" in rep.steps and sfx.read_wav(out).channels == 1


def test_fit_sound_keeps_tone_and_loop():
	data = tone(1.0, rate=44100, loop=(22050, 44100))
	out, rep = sfx.fit_wav_bytes(data, 20 * 1024)
	snd = sfx.read_wav(out)
	assert snd.loop_start == snd.rate // 2
	spec = np.abs(np.fft.rfft(snd.samples[:, 0]))
	peak = np.argmax(spec) * snd.rate / len(snd.samples)
	assert abs(peak - 440) < 5


def test_fit_sound_cut_after_loop_end_and_truncate():
	out, rep = sfx.fit_wav_bytes(tone(1.0, loop=(0, 4000)), 20 * 1024)
	assert rep.steps[0] == "cut after loop end" and rep.after == 8000
	out, rep = sfx.fit_wav_bytes(tone(5.0), 20 * 1024)
	assert not rep.ok
	out, rep = sfx.fit_wav_bytes(tone(5.0), 20 * 1024, truncate=True)
	assert rep.ok and rep.truncated and sfx.read_wav(out).pcm_bytes <= 20 * 1024


def test_sfx_cli_and_budget_cap(tmp_path, capsys):
	g = staged(tmp_path, 1000)
	(g / "sound" / "ambience" / "wind.wav").write_bytes(tone(1.5))	# 65 KiB PCM: over the cap
	rep = budget.map_budget(g, g / "maps" / "c0a0.bsp")
	snd = {ln.pool: ln for ln in rep.lines}["Entity sounds"]
	assert snd.status == budget.OVER and "never play" in snd.detail

	assert cli.main(["sfx", str(g / "sound")]) == 0	# dry run fits at 11 kHz 8 bit
	assert cli.main(["sfx", str(g / "sound"), "--in-place"]) == 0
	capsys.readouterr()
	rep = budget.map_budget(g, g / "maps" / "c0a0.bsp")
	assert {ln.pool: ln for ln in rep.lines}["Entity sounds"].status == budget.OK


# --- disc report -----------------------------------------------------------------

from gxbake.core import disc  # noqa: E402


def test_disc_report(tmp_path, capsys):
	g = staged(tmp_path, 3000)
	(g / "gc_studio" / "copy.mdl").write_bytes(bytes(3000))	# same bytes as barney.mdl
	(g / "sound" / "unused.wav").write_bytes(b"x" * 10)
	(g / "media" / "wt_c1a0.gct").write_bytes(gct.write_gct([entry("wall", 128, 128), entry("floor", 64, 64)]))
	rep = disc.disc_report(g)
	assert rep.on_disc % 2048 == 0 and rep.on_disc >= rep.bytes
	assert any(set(d["paths"]) == {"gc_studio/barney.mdl", "gc_studio/copy.mdl"} for d in rep.duplicates)
	assert [d["name"] for d in rep.shared_textures] == ["wall"]
	unref = [r for r, _s in rep.unreferenced]
	assert "gc_studio/copy.mdl" in unref and "sound/unused.wav" in unref
	assert "gc_studio/barney.mdl" not in unref and "sound/ambience/wind.wav" not in unref
	assert cli.main(["disc", str(g)]) == 0
	assert "identical files" in capsys.readouterr().out
	assert cli.main(["disc", str(g), "--json"]) == 0
	assert json.loads(capsys.readouterr().out)["files"] == rep.files
