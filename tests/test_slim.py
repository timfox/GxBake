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
