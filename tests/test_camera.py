"""Camera paths → .gcam: importers, key reduction under tolerances, the file format, the CLI and
build rule, and port/gc_cam.c sampling exactly what GxBake's reader does (compiled on the host)."""

import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gxbake import cli
from gxbake.core import budget, build, camera

from . import camera_fixtures as cf
from . import fixtures

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
STUB = ROOT / "tests" / "c_stub"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
FLAGS = ["-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", "-O2"]


def fwd(q):
	return camera.quat_to_matrix(q)[:, 0]


def angle_between(a, b):
	return math.degrees(math.acos(float(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def write(tmp_path, name, data):
	p = tmp_path / name
	p.write_bytes(data) if isinstance(data, bytes) else p.write_text(data)
	return p


# -- maths --------------------------------------------------------------------------------------------


def test_angles_round_trip_and_frames():
	for a in [(0, 0, 0), (30, 45, 0), (-20, 170, 10), (60, -90, -30)]:
		m = camera.angles_to_matrix(*a)
		assert np.allclose(m.T @ m, np.eye(3))
		assert np.allclose(camera.matrix_to_angles(m), a, atol=1e-9)
		assert np.allclose(camera.matrix_to_angles(camera.quat_to_matrix(camera.quat_from_matrix(m))), a, atol=1e-6)
	# a glTF camera with no rotation looks down -Z (Y-up), which is +Y in the engine
	f = camera.Frame("gltf", "gltf")
	assert np.allclose(f.orient(np.eye(3))[:, 0], [0, 1, 0])
	assert np.allclose(f.point([1, 2, 3]), [1, -3, 2])
	# FBX cameras look down +X
	assert np.allclose(camera.Frame("fbx", "fbx").orient(np.eye(3))[:, 0], [1, 0, 0])
	assert camera.fov_x_to_y(90) == pytest.approx(camera.DEFAULT_FOV_Y, abs=1e-5)
	assert camera.fov_y_to_x(camera.fov_x_to_y(100, 16 / 9), 16 / 9) == pytest.approx(100)


def test_reduction_stays_within_tolerance():
	t = np.arange(301) / 60
	pos = np.stack([200 * np.sin(t), 150 * np.cos(0.7 * t), 20 * t], 1)
	for interp in ("smooth", "linear"):
		idx = camera.reduce_track(t, pos, camera.TRACK_POS, 0.1, interp)
		err = np.linalg.norm(camera.evaluate(t[idx], pos[idx], t, interp) - pos, axis=1)
		assert err.max() <= 0.1
		assert len(idx) < (60 if interp == "smooth" else 120)	# of 301 samples
	smooth = camera.reduce_track(t, pos, camera.TRACK_POS, 0.1, "smooth")
	linear = camera.reduce_track(t, pos, camera.TRACK_POS, 0.1, "linear")
	assert len(smooth) < len(linear)	# a smooth path wants the smooth curve
	# a straight line needs only its ends
	line = np.stack([t * 100, t * 0, t * 0], 1)
	assert list(camera.reduce_track(t, line, camera.TRACK_POS, 0.01, "linear")) == [0, 300]


# -- sources ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("container", ["gltf", "glb"])
def test_gltf_dolly_camera(tmp_path, container):
	doc = cf.dolly_gltf()
	src = write(tmp_path, f"shot.{container}", json.dumps(doc) if container == "gltf" else cf.glb(doc))
	(clip,) = camera.load(src)
	assert clip.name == "flyby"
	assert clip.near == pytest.approx(0.1 * 39.37) and clip.far == pytest.approx(100 * 39.37)
	assert clip.aspect == pytest.approx(1.5)
	data, (enc,) = camera.encode_all([clip], camera.CameraOptions())
	assert enc.max_error["pos"] <= 0.25 and enc.max_error["rot"] <= 0.15 and enc.max_error["fov"] <= 0.05
	(g,) = camera.read_gcam(data)
	assert g.duration == pytest.approx(2.0)
	for t in (0.0, 0.37, 1.0, 1.61, 2.0):
		s = g.sample(t)
		p, f = cf.dolly_expected(t)
		# the dolly's 30 keys/s are straight segments of a circle: allow their chord error too
		assert np.linalg.norm(s["pos"][0] - p) < 0.35 + 3 * 39.37 * (1 - math.cos(1 / 60)), t
		assert angle_between(fwd(s["rot"][0]), f) < 0.2, t
		assert s["fov"][0] == pytest.approx(40 + 10 * t, abs=0.05)
	assert len(data) < 400	# 121 samples × (3 + 4 + 1) floats would be 3.8 KB


def test_gltf_without_animation_is_one_static_clip(tmp_path):
	doc = cf.dolly_gltf()
	del doc["animations"]
	(clip,) = camera.load(write(tmp_path, "still.gltf", json.dumps(doc)))
	assert clip.name == "static" and len(clip.times) == 1
	data, _ = camera.encode_all([clip], camera.CameraOptions())
	(g,) = camera.read_gcam(data)
	assert g.duration == 0
	assert np.allclose(g.sample(5.0)["pos"][0], [0, 0, 39.37], atol=0.01)	# the camera sits 1 m up its dolly


def test_json_keys_cut_look_at_and_fov(tmp_path):
	doc = {"fps": 24, "near": 2, "far": 8000, "cameras": [{"name": "Intro Shot", "keys": [
		{"t": 0, "pos": [0, 0, 64], "look_at": [500, 0, 64], "fov_x": 90},
		{"t": 1.5, "pos": [300, 100, 64], "look_at": [500, 0, 64]},
		{"frame": 72, "pos": [400, 0, 100], "angles": [15, 90, 0], "fov": 50},
		{"t": 3, "cut": True, "pos": [-800, 0, 0], "angles": [0, 180, 0]},
		{"t": 5, "pos": [-900, 200, 0], "angles": [0, 270, 0]}]}]}
	(clip,) = camera.load(write(tmp_path, "keys.json", json.dumps(doc)))
	assert clip.near == 2 and clip.far == 8000
	data, (enc,) = camera.encode_all([clip], camera.CameraOptions())
	(g,) = camera.read_gcam(data)
	assert g.name == "intro_shot"
	s = g.sample([0.0, 2.999, 3.0, 5.0])
	assert np.allclose(s["pos"][0], [0, 0, 64], atol=0.01)
	assert angle_between(fwd(s["rot"][0]), [1, 0, 0]) < 0.1
	assert s["fov"][0] == pytest.approx(camera.DEFAULT_FOV_Y, abs=0.01)
	# the first shot holds its last key (t = 3 s from frame 72 at 24 fps) up to the cut, then jumps
	assert np.allclose(s["pos"][1], [400, 0, 100], atol=0.5)
	assert np.allclose(camera.matrix_to_angles(camera.quat_to_matrix(s["rot"][1])), [15, 90, 0], atol=0.2)
	assert np.allclose(s["pos"][2], [-800, 0, 0], atol=0.01)
	assert s["fov"][2] == pytest.approx(50, abs=0.05)
	assert np.allclose(camera.matrix_to_angles(camera.quat_to_matrix(s["rot"][3]))[:2], [0, -90], atol=0.1)
	# both sides of the cut are keys in every track
	for tr in g.tracks.values():
		assert (np.diff(tr.times) == 0).sum() == 1


def test_json_blender_axes(tmp_path):
	"""What docs/camera.md's Blender script writes: Z-up metres, the camera looking down its -Z."""
	h = math.sqrt(0.5)
	doc = {"axes": "blender", "fps": 24, "near": 0.1, "far": 100, "cameras": [{"name": "Camera", "keys": [
		{"frame": 0, "pos": [1, 2, 3], "rot": [0, 0, 0, 1], "fov_y": 40},
		{"frame": 24, "pos": [1, 2, 3], "rot": [h, 0, 0, h], "fov_y": 40}]}]}
	(c,) = camera.load(write(tmp_path, "blender.json", json.dumps(doc)))
	assert np.allclose(c.pos[0], np.array([1, 2, 3]) * 39.37)
	assert c.near == pytest.approx(3.937) and c.far == pytest.approx(3937)
	assert np.allclose(fwd(c.rot[0]), [0, 0, -1])	# unrotated: looking straight down
	assert np.allclose(fwd(c.rot[1]), [0, 1, 0])	# tilted up 90° about X: looking along +Y
	assert np.allclose(camera.quat_to_matrix(c.rot[1])[:, 2], [0, 0, 1])	# and level


def test_csv_clips_and_columns(tmp_path):
	text = "name,t,x,y,z,pitch,yaw,roll,fov_x\n" + "\n".join(
		f"a,{t},{t * 100},0,64,0,{t * 30},0,90" for t in range(5)) + "\n" + "\n".join(
		f"b,{t},0,{t * 50},0,10,0,0,75" for t in range(3)) + "\n"
	clips = camera.load(write(tmp_path, "path.csv", text))
	assert [c.name for c in clips] == ["a", "b"]
	data, encs = camera.encode_all(clips, camera.CameraOptions(encoding="f32"))
	a, b = camera.read_gcam(data)
	assert np.allclose(a.sample(2.0)["pos"][0], [200, 0, 64], atol=1e-3)
	assert camera.matrix_to_angles(camera.quat_to_matrix(a.sample(2.0)["rot"][0]))[1] == pytest.approx(60, abs=0.01)
	assert camera.fov_y_to_x(b.sample(1.0)["fov"][0]) == pytest.approx(75, abs=0.01)
	assert not a.fixed and len(a.tracks[camera.TRACK_POS].times) == 2	# a straight line at constant speed


def test_goldsrc_trigger_camera(tmp_path):
	clips = camera.load(write(tmp_path, "c1a0.ent", cf.goldsrc_entities()))
	(c,) = clips
	assert c.name == "intro_cam" and c.duration == pytest.approx(8.0)
	data, _ = camera.encode_all(clips, camera.CameraOptions())
	(g,) = camera.read_gcam(data)
	s = g.sample([0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0])
	want = [[0, 0, 64], [100, 0, 64], [200, 0, 64], [200, 0, 64], [200, 0, 64], [200, 150, 64], [200, 300, 64], [200, 300, 64]]
	assert np.allclose(s["pos"], want, atol=0.3)
	for p, q in zip(s["pos"], s["rot"]):	# always looking at the info_target
		assert angle_between(fwd(q), np.array([100, 100, 0]) - p) < 0.2
	# from a BSP the output is named for the map, which is what budget looks for
	bsp = write(tmp_path, "c1a0.bsp", fixtures.bsp30(cf.goldsrc_entities(), []))
	assert camera.output_name(bsp) == "cam_c1a0.gcam"
	assert camera.load(bsp)[0].name == "intro_cam"


def test_assimp_collada(tmp_path):
	pytest.importorskip("assimp_py")
	(c,) = camera.load(write(tmp_path, "move.dae", cf.collada()))
	assert c.fov[0] == pytest.approx(50)
	assert c.far == pytest.approx(200 * 39.37)
	data, (enc,) = camera.encode_all([c], camera.CameraOptions())
	(g,) = camera.read_gcam(data)
	s = g.sample([0.0, 0.5, 1.0])
	assert np.allclose(s["pos"], [[0, -5 * 39.37, 39.37], [2 * 39.37, -5 * 39.37, 39.37], [4 * 39.37, -5 * 39.37, 39.37]], atol=0.3)
	assert angle_between(fwd(s["rot"][0]), [0, 1, 0]) < 0.01
	assert len(g.tracks[camera.TRACK_POS].times) == 2


# -- encoding and limits ------------------------------------------------------------------------------


def _wiggle(seconds=6.0):
	t = np.arange(int(seconds * 60) + 1) / 60
	pos = np.stack([300 * np.cos(t), 300 * np.sin(t), 80 + 10 * np.sin(3 * t)], 1)
	rot = np.array([camera.quat_from_matrix(camera.angles_to_matrix(5 * math.sin(2 * x), math.degrees(x) + 90, 0)) for x in t])
	return camera.CameraPath("orbit", t, pos, camera._hemisphere(rot), 60 + 10 * np.sin(t))


def test_fixed_is_smaller_and_both_meet_tolerance():
	p = _wiggle()
	sizes = {}
	for enc in camera.ENCODINGS:
		data, (e,) = camera.encode_all([p], camera.CameraOptions(encoding=enc))
		assert e.max_error["pos"] <= 0.25 and e.max_error["rot"] <= 0.15 and e.max_error["fov"] <= 0.05, enc
		sizes[enc] = len(data)
		(g,) = camera.read_gcam(data)
		s = g.sample(p.times)
		assert np.linalg.norm(s["pos"] - p.pos, axis=1).max() <= 0.25
	assert sizes["fixed"] < sizes["f32"] * 0.65


def test_max_bytes_loosens_tolerances():
	p = _wiggle()
	full, _ = camera.encode_all([p], camera.CameraOptions())
	small, (e,) = camera.encode_all([p], camera.CameraOptions(max_bytes=len(full) // 2))
	assert len(small) <= len(full) // 2
	assert any("tolerances" in n for n in e.path.notes)
	with pytest.raises(ValueError):
		camera.encode_all([p], camera.CameraOptions(max_bytes=100))


def test_fixed_rejects_too_long_and_loops_wrap():
	p = _wiggle(2.0)
	p.loop = True
	data, _ = camera.encode_all([p], camera.CameraOptions())
	(g,) = camera.read_gcam(data)
	assert g.loop
	assert np.allclose(g.sample(0.5)["pos"], g.sample(2.5)["pos"])
	long = camera.CameraPath("long", np.array([0.0, 2000.0]), np.zeros((2, 3)), np.array([[0, 0, 0, 1.0]] * 2), np.full(2, 70.0))
	with pytest.raises(ValueError, match="u16"):
		camera.encode_all([long], camera.CameraOptions())
	camera.encode_all([long], camera.CameraOptions(encoding="f32"))


def test_read_gcam_rejects_bad_files():
	data, _ = camera.encode_all([_wiggle(1.0)], camera.CameraOptions())
	with pytest.raises(ValueError):
		camera.read_gcam(b"NOPE" + data[4:])
	with pytest.raises(ValueError):
		camera.read_gcam(data[:40])
	assert struct.unpack_from(">I", data, 8)[0] == len(data)


# -- CLI, build and budget ------------------------------------------------------------------------------


def test_camera_cli(tmp_path, capsys):
	src = write(tmp_path, "shot.gltf", json.dumps(cf.dolly_gltf()))
	write(tmp_path, "c1a0.ent", cf.goldsrc_entities())
	out = tmp_path / "out"
	assert cli.main(["camera", str(src), str(tmp_path / "c1a0.ent"), "-o", str(out), "--preview", str(tmp_path / "pv")]) == 0
	assert (out / "shot.gcam").is_file() and (out / "cam_c1a0.gcam").is_file()
	assert (tmp_path / "pv" / "shot_flyby.png").is_file()
	capsys.readouterr()
	assert cli.main(["camera", "--inspect", str(out / "shot.gcam"), "--json", "--samples", str(tmp_path / "s")]) == 0
	rows = json.loads(capsys.readouterr().out)
	assert rows[0]["name"] == "flyby" and rows[0]["tracks"]["fov"]["keys"] == 2
	lines = (tmp_path / "s" / "shot_flyby.csv").read_text().splitlines()
	assert lines[0] == "t,x,y,z,pitch,yaw,roll,fov_y" and len(lines) == 62
	assert cli.main(["camera", str(src), "-o", str(out), "--combine", "all.gcam", "--fov-x", "100", "--encoding", "f32"]) == 0
	(g,) = camera.read_gcam((out / "all.gcam").read_bytes())
	assert camera.fov_y_to_x(g.sample(1.0)["fov"][0]) == pytest.approx(100, abs=1e-3)
	assert cli.main(["camera", str(src), "--list"]) == 0
	assert cli.main(["camera", str(tmp_path / "nothing.xyz"), "-o", str(out)]) == 1


def test_build_rule_and_budget_line(tmp_path):
	(tmp_path / "cuts").mkdir()
	write(tmp_path, "cuts/a.gltf", json.dumps(cf.dolly_gltf()))
	(tmp_path / "gxbake.toml").write_text(
		'[build]\noutput = "game"\n\n[[camera]]\ninputs = ["cuts/*.gltf"]\noutput = "media"\n'
		'combine = "cam_c0a0.gcam"\nmax_bytes = "4K"\n')
	rep = build.build(tmp_path / "gxbake.toml")
	(g,) = rep.groups.values()
	(u,) = g.units
	assert u.unit.kind == "camera" and u.unit.fixed_bytes == (tmp_path / "game/media/cam_c0a0.gcam").stat().st_size
	assert build.build(tmp_path / "gxbake.toml").groups["shared"].skipped	# incremental
	# a staged gamedir with the map: budget reports the camera file
	game = tmp_path / "game"
	(game / "maps").mkdir()
	(game / "maps" / "c0a0.bsp").write_bytes(fixtures.bsp30('{\n"classname" "worldspawn"\n}\n', []))
	lines = {ln.pool: ln for ln in budget.map_budget(game, game / "maps" / "c0a0.bsp").lines}
	assert lines["Camera paths"].used == u.unit.fixed_bytes
	assert lines["Camera paths"].detail == "1 clips"


# -- port/gc_cam.c ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
	if CC is None:
		pytest.skip("no C compiler")
	d = tmp_path_factory.mktemp("cambin")
	plain, gx = d / "test_gc_cam", d / "test_gc_cam_gx"
	subprocess.run([CC, *FLAGS, str(PORT / "gc_cam.c"), str(PORT / "test_gc_cam.c"), "-o", str(plain), "-lm"], check=True)
	subprocess.run([CC, *FLAGS, "-DGC_CAM_GX", f"-I{STUB}", str(PORT / "gc_cam.c"), str(PORT / "test_gc_cam.c"),
		str(STUB / "gx_stub.c"), "-o", str(gx), "-lm"], check=True)
	return plain, gx


def _c_sample(exe, path, name, times):
	res = subprocess.run([exe, "sample", path, name, *[f"{t:.6f}" for t in times]], check=True, capture_output=True, text=True)
	return np.array([list(map(float, r.split())) for r in res.stdout.splitlines()])


@pytest.mark.parametrize("encoding", camera.ENCODINGS)
@pytest.mark.parametrize("interp", ["smooth", "linear"])
def test_c_samples_like_python(harness, tmp_path, encoding, interp):
	clips = [_wiggle(4.0)] + camera.load(write(tmp_path, "keys.json", json.dumps({"cameras": [{"name": "Cuts", "keys": [
		{"t": 0, "pos": [0, 0, 0], "angles": [0, 0, 0]}, {"t": 1, "pos": [100, 0, 0], "angles": [0, 40, 0]},
		{"t": 1, "cut": True, "pos": [0, 500, 0], "angles": [20, 180, 5]}, {"t": 2.5, "pos": [50, 400, 30], "angles": [0, 220, 0]}]}]})))
	data, _ = camera.encode_all(clips, camera.CameraOptions(encoding=encoding, interp=interp))
	f = write(tmp_path, "c.gcam", data)
	listing = subprocess.run([harness[0], "list", f], check=True, capture_output=True, text=True).stdout.splitlines()
	assert [r.split()[1] for r in listing] == ["orbit", "cuts"]
	for g in camera.read_gcam(data):
		times = list(np.linspace(-0.5, g.duration + 0.5, 97)) + [0.999, 1.0005]
		c = _c_sample(harness[0], f, g.name.upper(), times)	# names match case-insensitively
		s = g.sample(times)
		assert np.allclose(c[:, 0:3], s["pos"], atol=2e-3 * max(1, np.abs(s["pos"]).max() / 100))
		assert np.allclose(np.abs(np.sum(c[:, 3:7] * s["rot"], 1)), 1, atol=1e-5)
		assert np.allclose(c[:, 7], s["fov"], atol=1e-3)
		ang = np.array([camera.matrix_to_angles(camera.quat_to_matrix(q)) for q in s["rot"]])
		d = np.abs((c[:, 8:11] - ang + 180) % 360 - 180)
		assert d[:, 0].max() < 0.01 and d[:, 1][np.abs(ang[:, 0]) < 85].max() < 0.01
		assert np.allclose(c[:, 11], [camera.fov_y_to_x(x) for x in s["fov"]], atol=1e-3)


def test_c_view_and_projection(harness, tmp_path):
	data, _ = camera.encode_all([_wiggle(2.0)], camera.CameraOptions(encoding="f32"))
	f = write(tmp_path, "c.gcam", data)
	(g,) = camera.read_gcam(data)
	res = subprocess.run([harness[0], "mtx", f, "orbit", "0.7", "1.7777"], check=True, capture_output=True, text=True)
	m = np.array([list(map(float, r.split())) for r in res.stdout.splitlines()])
	view, proj = m[:3], m[3:]
	s = g.sample(0.7)
	r = camera.quat_to_matrix(s["rot"][0])
	# a point straight ahead of the camera lands on -Z in GX camera space; left of it on -X
	ahead = s["pos"][0] + r[:, 0] * 100
	left = s["pos"][0] + r[:, 1] * 10
	assert np.allclose(view @ np.append(ahead, 1), [0, 0, -100], atol=1e-2)
	assert np.allclose(view @ np.append(left, 1), [-10, 0, 0], atol=1e-2)
	cot = 1 / math.tan(math.radians(s["fov"][0]) / 2)
	n, fz = g.near, g.far
	assert np.allclose(proj, [[cot / 1.7777, 0, 0, 0], [0, cot, 0, 0], [0, 0, -n / (fz - n), -fz * n / (fz - n)], [0, 0, -1, 0]], atol=1e-4)
	res = subprocess.run([harness[1], "load", f, "orbit", "0.7", "1.7777"], check=True, capture_output=True, text=True)
	assert res.stdout.splitlines()[0].startswith("LoadProjectionMtx 0 ")


def test_c_rejects_bad_files(harness, tmp_path):
	data, _ = camera.encode_all([_wiggle(1.0)], camera.CameraOptions())
	f = write(tmp_path, "t.gcam", data[:-8])
	assert subprocess.run([harness[0], "list", f]).returncode == 1
	bad = bytearray(data)
	struct.pack_into(">I", bad, camera.HEADER + 36, len(data))	# body size past the end
	f = write(tmp_path, "b.gcam", bytes(bad))
	assert subprocess.run([harness[0], "list", f]).returncode == 1
	f = write(tmp_path, "ok.gcam", data)
	assert subprocess.run([harness[0], "sample", f, "missing", "0"]).returncode == 1
