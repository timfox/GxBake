"""Panorama / cubemap → GameCube sky sides (gxbake.core.skybox, `gxbake sky`)."""

import math
import struct

import numpy as np
import pytest
from PIL import Image

from gxbake import cli
from gxbake.core import images, skybox
from gxbake.core.project import SKY_NET, Kind, Project, Settings

# R_GXDrawSkyBox in ref/gx/r_gx_world.c (and ref/gl/gl_warp.c), copied as data.
ST_TO_VEC = ((3, -1, 2), (-3, 1, 2), (1, 3, 2), (-1, -3, 2), (-2, -1, 3), (2, -1, -3))
SKY_TEX_ORDER = (0, 2, 1, 3, 4, 5)	# axis -> index into rt bk lf ft up dn


def engine_dir(side: str, s: float, t: float) -> np.ndarray:
	"""World direction the engine draws at texcoord (s, t) of `side`, straight from its tables."""
	axis = SKY_TEX_ORDER.index(skybox.SIDES.index(side))
	cx, cy = 2 * s - 1, 1 - 2 * t	# inverse of ((cx+1)/2, 1-(cy+1)/2)
	b = (cx, cy, 1.0)
	return np.array([(-b[-k - 1] if k < 0 else b[k - 1]) for k in ST_TO_VEC[axis]])


def direction_panorama(w=2048):
	"""2:1 panorama whose red = longitude, green = latitude (centre = lon 0)."""
	h = w // 2
	yy, xx = np.mgrid[0:h, 0:w]
	lon = (xx + 0.5) / w * 360 - 180
	lat = 90 - (yy + 0.5) / h * 180
	img = np.zeros((h, w, 3), np.uint8)
	img[..., 0] = ((lon + 180) / 360 * 255).round()
	img[..., 1] = ((lat + 90) / 180 * 255).round()
	img[..., 2] = (((np.floor(lon / 30) + np.floor(lat / 30)) % 2) * 255)
	return img


def lon_lat(d: np.ndarray, yaw=skybox.DEFAULT_YAW) -> tuple[float, float]:
	f, r, u = skybox.view_basis(yaw) @ (d / np.linalg.norm(d))
	return math.degrees(math.atan2(r, f)), math.degrees(math.asin(u))


def test_engine_faces_match_the_renderer_tables():
	for side, (n, right, up) in skybox.ENGINE_FACES.items():
		for s, t in ((0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.25, 0.9)):
			ours = n + (2 * s - 1) * right + (1 - 2 * t) * up
			assert np.allclose(ours, engine_dir(side, s, t)), side
	# side_directions walks texel centres row by row, top row first.
	d = skybox.side_directions("ft", 4)
	assert np.allclose(d[0], engine_dir("ft", 0.125, 0.125))
	assert np.allclose(d[-1], engine_dir("ft", 0.875, 0.875))


def test_equirect_lands_on_the_right_sides(tmp_path):
	Image.fromarray(direction_panorama()).save(tmp_path / "beach.png")
	sky = skybox.build([tmp_path / "beach.png"])
	assert (sky.name, sky.layout) == ("beach", "equirect")
	assert all(len(v) == skybox.SKY_BYTES == 8192 for v in sky.cmpr.values())
	for side in ("rt", "bk", "lf", "ft"):
		px = sky.sides[side][40, 64].astype(float)	# a little above the horizon, centre column
		lon, lat = lon_lat(engine_dir(side, 64.5 / 128, 40.5 / 128))
		assert abs(px[0] / 255 * 360 - 180 - lon) < 4 or abs(lon) > 170, side
		assert abs(px[1] / 255 * 180 - 90 - lat) < 3, side
	assert sky.sides["up"][64, 64, 1] > 245 and sky.sides["dn"][64, 64, 1] < 10
	# Default yaw: panorama centre on ft; turning 90° puts it on rt.
	assert abs(int(sky.sides["ft"][64, 64, 0]) - 128) <= 2
	turned = skybox.build([tmp_path / "beach.png"], skybox.SkyOptions(yaw=0))
	assert np.abs(turned.sides["rt"].astype(int) - sky.sides["ft"].astype(int)).mean() < 2


def test_sides_meet_without_seams(tmp_path):
	Image.fromarray(direction_panorama()).save(tmp_path / "p.png")
	sky = skybox.build([tmp_path / "p.png"])
	# Neighbours along the horizon: the engine's right edge of one side is the left edge of the next.
	n = 128
	for a in skybox.SIDES[:4]:
		for b in skybox.SIDES[:4]:
			if np.allclose(engine_dir(a, 1, 0.5), engine_dir(b, 0, 0.5)):
				diff = np.abs(sky.sides[a][:, -1].astype(int) - sky.sides[b][:, 0].astype(int))[16:-16, :2]
				assert diff.mean() < 4, (a, b)
				break
		else:
			pytest.fail(f"no side to the right of {a}")
	assert n == sky.sides["ft"].shape[0]


def test_cross_and_strip_round_trip(tmp_path):
	Image.fromarray(direction_panorama(1024)).save(tmp_path / "p.png")
	ref = skybox.build([tmp_path / "p.png"])
	cross = skybox.cross_preview(ref.sides)
	Image.fromarray(cross).save(tmp_path / "h.png")
	got = skybox.build([tmp_path / "h.png"])
	assert got.layout == "cross"
	for s in skybox.SIDES:
		assert images.psnr(got.sides[s], ref.sides[s]) > 30, s

	faces = skybox.split_cross(cross)
	n = faces["front"].shape[0]
	vert = np.zeros((4 * n, 3 * n, 3), np.uint8)
	for face, (c, r) in skybox._VCROSS.items():
		f = faces[face][::-1, ::-1] if face == "back" else faces[face]
		vert[r * n:(r + 1) * n, c * n:(c + 1) * n] = f
	strip = np.concatenate([faces[f] for f in skybox.VIEW_ORDER], axis=1)
	Image.fromarray(vert).save(tmp_path / "v.png")
	Image.fromarray(strip).save(tmp_path / "s.png")
	for name, layout in (("v.png", "cross"), ("s.png", "strip")):
		got = skybox.build([tmp_path / name])
		assert got.layout == layout
		for s in skybox.SIDES:
			assert images.psnr(got.sides[s], ref.sides[s]) > 30, (name, s)

	# The same faces as six named files.
	for face, img in faces.items():
		Image.fromarray(img).save(tmp_path / f"cube_{face}.png")
	got = skybox.build([tmp_path / "cube_left.png"])
	assert (got.name, got.layout) == ("cube", "faces")
	for s in skybox.SIDES:
		assert images.psnr(got.sides[s], ref.sides[s]) > 30, s


def test_goldsrc_faces_are_used_as_they_are(tmp_path):
	rng = np.random.default_rng(3)
	for s in skybox.SIDES:
		Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(tmp_path / f"dusk{s}.tga")
	sky = skybox.build([tmp_path / "duskft.tga"])
	assert sky.name == "dusk"
	want = Image.open(tmp_path / "duskbk.tga").convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
	assert np.array_equal(sky.sides["bk"], np.asarray(want))
	assert sky.cmpr["bk"] == images.sky_side_cmpr(images.load_rgba(tmp_path / "duskbk.tga"))


def test_photo_wraps_without_a_seam(tmp_path):
	img = np.zeros((300, 800, 3), np.uint8)
	img[:, :, 0] = np.linspace(0, 255, 800)[None]	# red ramps left → right
	img[:150, :, 2] = 255	# blue sky over a black ground
	Image.fromarray(img).save(tmp_path / "vista.jpg")
	sky = skybox.build([tmp_path / "vista.jpg"])
	assert sky.layout == "photo"
	ft, bk = sky.sides["ft"], sky.sides["bk"]
	assert abs(int(ft[60, 64, 0]) - 128) < 12	# photo centre in front
	assert abs(int(bk[60, 64, 0]) - 128) < 12	# behind: the mirrored copy's centre
	assert sky.sides["rt"][60, 64, 0] < 12 and sky.sides["lf"][60, 64, 0] > 243	# the photo's edges, left and right
	assert sky.sides["up"][64, 64, 2] > 200 and sky.sides["dn"][64, 64, 2] < 40


def write_hdr(path, rgb: np.ndarray, rle: bool):
	h, w = rgb.shape[:2]
	m = rgb.max(axis=2)
	e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, 0)
	scale = np.where(m > 1e-32, 256.0 / np.exp2(e), 0)
	rgbe = np.zeros((h, w, 4), np.uint8)
	rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255)
	rgbe[..., 3] = np.where(m > 1e-32, e + 128, 0)
	out = bytearray(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode())
	for y in range(h):
		if not rle:
			out += rgbe[y].tobytes()
			continue
		out += bytes((2, 2, w >> 8, w & 255))
		for c in range(4):
			row = rgbe[y, :, c]
			x = 0
			while x < w:	# one run, then literals
				n = 1
				while x + n < w and n < 127 and row[x + n] == row[x]:
					n += 1
				if n > 2:
					out += bytes((128 + n, row[x]))
				else:
					n = min(w - x, 128)
					out += bytes((n,)) + row[x:x + n].tobytes()
				x += n
	path.write_bytes(bytes(out))


@pytest.mark.parametrize("rle", [False, True])
def test_hdr_reader(tmp_path, rle):
	rng = np.random.default_rng(1)
	rgb = rng.uniform(0, 20, (16, 64, 3)).astype(np.float32)
	rgb[:, 10:30] = 3.0	# runs for the RLE path
	write_hdr(tmp_path / "x.hdr", rgb, rle)
	got = skybox.read_hdr(tmp_path / "x.hdr")
	assert got.shape == rgb.shape
	assert np.abs(got - rgb).max() <= rgb.max() / 128	# 8-bit shared-exponent mantissas
	ldr = skybox.tonemap(got)
	assert ldr.dtype == np.uint8 and ldr.shape == rgb.shape


def test_sky_command(tmp_path, capsys):
	rgb = np.ones((256, 512, 3), np.float32)
	rgb[:128] *= 8.0	# bright sky, dark ground
	write_hdr(tmp_path / "Field.hdr", rgb, rle=True)
	out = tmp_path / "out"
	assert cli.main(["sky", str(tmp_path / "Field.hdr"), "-o", str(out), "--env", "64",
		"--preview", str(tmp_path / "prev.png")]) == 0
	text = capsys.readouterr().out
	assert '"skyname" "field"' in text and "48.0 KiB" in text
	for s in skybox.SIDES:
		assert (out / f"media/sky_field{s}.cmp").stat().st_size == 8192
		assert Image.open(out / f"gfx/env/field{s}.tga").size == (64, 64)
	assert Image.open(tmp_path / "prev.png").size == (512, 384)
	up = images.decode_raw_cmp((out / "media/sky_fieldup.cmp").read_bytes())
	dn = images.decode_raw_cmp((out / "media/sky_fielddn.cmp").read_bytes())
	assert up[..., :3].mean() > dn[..., :3].mean() + 40
	assert cli.main(["sky", str(tmp_path / "nothing.png"), "-o", str(out)]) == 1


def test_project_detects_panoramas(tmp_path):
	Image.fromarray(direction_panorama(1024)).save(tmp_path / "noon.png")
	Image.fromarray(np.zeros((128, 256, 3), np.uint8)).save(tmp_path / "trim.png")	# small 2:1: a texture
	proj = Project(Settings(pack_images=False))
	proj.add_path(tmp_path)
	kinds = {a.path.name: a.kind for a in proj.assets}
	assert kinds == {"noon.png": Kind.SKYBOX, "trim.png": Kind.IMAGE}
	sky = next(a for a in proj.assets if a.kind == Kind.SKYBOX)
	assert sky.name == "sky: noon"
	assert proj.children(sky) == [*skybox.SIDES, SKY_NET]
	pv = proj.preview(sky, SKY_NET)
	assert pv.result.shape[:2] == (384, 512) and pv.info["Layout"] == "equirect"
	assert proj.preview(sky, "up").result.shape[:2] == (128, 128)
	written = proj.convert(proj.assets, tmp_path / "out")
	rels = {w.rel for w in written}
	assert {f"media/sky_noon{s}.cmp" for s in skybox.SIDES} <= rels
	assert "media/trim.gct" in rels

	off = Project(Settings(detect_panoramas=False))
	off.add_path(tmp_path / "noon.png")
	assert off.assets[0].kind == Kind.IMAGE
	assert off.add_skybox([tmp_path / "noon.png"], name="Noon2")[0].extra["name"] == "noon2"
