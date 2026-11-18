import numpy as np
import pytest

from gxbake import cli
from gxbake.core import models, studio, studio_slim


def animated_model() -> bytes:
	tri = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], np.float32)
	mesh = models.Submesh(0, tri, np.tile([0, 0, 1], (4, 1)).astype(np.float32),
		np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32), np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
		bones=np.array([0, 1, 2, 2], np.int32))
	rgba = np.zeros((16, 16, 4), np.uint8)
	rgba[..., 0] = np.arange(16)[None] * 16
	rgba[..., 3] = 255
	bones = [models.Bone("root", -1, np.zeros(6)), models.Bone("arm", 0, np.array([0, 0, 10, 0, 0, 0.0])),
		models.Bone("hand", 1, np.array([10, 0, 0, 0, 0, 0.0]))]
	t = np.linspace(0, 2 * np.pi, 31)
	walk = np.zeros((31, 3, 6))
	walk[:] = [b.pose for b in bones]
	walk[:, 1, 5] = np.sin(t) * 0.8
	walk[:, 2, 0] = 10 + np.cos(t) * 2
	death = np.zeros((20, 3, 6))
	death[:] = [b.pose for b in bones]
	death[:, 0, 2] = np.linspace(0, -30, 20)
	model = models.ImportedModel("guy", [models.Material("skin", rgba, False)], [mesh], bones=bones,
		sequences=[models.Sequence("walk", 30, walk), models.Sequence("death1", 15, death),
			models.Sequence("idle", 10, walk[:1])])
	return models.build_studio_mdl(model)


def test_breakdown_adds_up():
	raw = animated_model()
	d = studio_slim.studio_breakdown(raw)
	assert d["total"] == len(raw)
	assert d["animation"] > 0 and d["meshes"] > 0 and d["skins"] >= 16 * 16 + 768
	assert d["other"] >= 0


def test_decimate_keeps_poses_and_duration():
	raw = animated_model()
	res = studio_slim.slim_studio(raw, step=2)
	assert res.after < res.before
	assert ("walk", 31, 16) in res.changed and ("death1", 20, 11) in res.changed
	seqs = {s["name"]: s for s in models.studio_sequences(res.data)}
	old = {s["name"]: s for s in models.studio_sequences(raw)}
	assert seqs["walk"]["frames"] == 16
	assert seqs["walk"]["fps"] == pytest.approx(30 * 15 / 30)	# 30 intervals → 15: same 1 s
	assert seqs["idle"]["frames"] == 1
	for k in range(15):
		a = models.studio_pose(raw, 0, 2 * k)
		b = models.studio_pose(res.data, 0, k)
		assert all(np.allclose(x, y, atol=1e-4) for x, y in zip(a, b))
	assert all(np.allclose(x, y) for x, y in zip(models.studio_pose(raw, 1, 19), models.studio_pose(res.data, 1, 10)))
	# Geometry, skins and the lean/companion split survive the relocation.
	assert models.studio_meshes(res.data)[0]["tris"] == models.studio_meshes(raw)[0]["tris"]
	assert [n for n, *_ in studio.studio_textures(res.data)] == [n for n, *_ in studio.studio_textures(raw)]
	assert (studio.studio_textures(res.data)[0][2] == studio.studio_textures(raw)[0][2]).all()
	assert studio.lean_studio_mesh_bytes_from(res.data) is not None
	assert len(old) == len(seqs)


def test_stub_and_keep():
	raw = animated_model()
	res = studio_slim.slim_studio(raw, step=3, stub=["death*"], keep=["walk"])
	assert res.changed == [("death1", 20, 1)]
	seqs = models.studio_sequences(res.data)
	assert [s["frames"] for s in seqs] == [31, 1, 1]
	assert all(np.allclose(x, y) for x, y in zip(models.studio_pose(raw, 1, 0), models.studio_pose(res.data, 1, 0)))


def test_slim_cli(tmp_path, capsys):
	src = tmp_path / "guy.mdl"
	src.write_bytes(animated_model())
	assert cli.main(["slim", str(src)]) == 0
	assert "animation" in capsys.readouterr().out
	out = tmp_path / "out.mdl"
	assert cli.main(["slim", str(src), "--step", "2", "-o", str(out)]) == 0
	assert out.stat().st_size < src.stat().st_size


def _local_poses(raw, seq, frame):
	from gxbake.core import skeleton

	bones = models.studio_bones(raw)
	sq = models.studio_sequences(raw)[seq]
	return skeleton.decode_anim_frame(raw, sq["animindex"], len(bones), frame,
		np.array([b["value"] for b in bones]), np.array([b["scale"] for b in bones]))


@pytest.mark.parametrize("deg", [0.5, 1, 3])
def test_fit_deg_resamples_within_tolerance(deg):
	raw = animated_model()
	res = studio_slim.slim_studio(raw, fit_deg=deg)
	changed = {name: (n, m) for name, n, m in res.changed}
	assert changed["death1"] == (20, 4)	# linear motion: as few keys as max_step allows
	assert "walk" in changed and res.after < res.before
	for seq, name in ((0, "walk"), (1, "death1")):
		n, m = changed[name]
		for f in range(n):
			p = f * (m - 1) / (n - 1)
			i = min(int(p), m - 2)
			a, b = _local_poses(res.data, seq, i), _local_poses(res.data, seq, i + 1)
			got = a + (b - a) * (p - i)
			want = _local_poses(raw, seq, f)
			assert np.abs(got[:, :3] - want[:, :3]).max() <= deg / 10 + 0.01
			assert np.degrees(np.abs(got[:, 3:] - want[:, 3:]).max()) <= deg + 0.05


def test_fit_frames_and_auto_step_pick_fewest_keys():
	line = [float(k) for k in range(20)]
	assert studio_slim.fit_frames([(line, 0.01, None)], max_step=8) == 4
	assert studio_slim.auto_step([(line, 0.01, None)], max_step=8) == 1	# 19 frames: no step divides evenly
	assert studio_slim.auto_step([(line[:17], 0.01, None)], max_step=8) == 8
	# Angles wrap: a steady spin through ±π is still a straight line.
	spin = [((k * 0.5 + np.pi) % (2 * np.pi)) - np.pi for k in range(17)]
	assert studio_slim.fit_frames([(spin, 1e-6, 2 * np.pi)], max_step=2) == 9
