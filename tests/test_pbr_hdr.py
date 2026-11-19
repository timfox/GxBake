"""TPL/BTI writers, automatic GX formats under budgets, HDR tone maps/lightmaps/sphere maps,
PBR material baking and manifest builds (`gxbake auto|pbr|hdr|build`)."""

import base64
import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from gxbake import cli
from gxbake.core import autotex, build, consoletex, gct, gxtex, hdr, pbr, skybox, tpl


def gradient(w=64, h=64, alpha=None):
	y, x = np.mgrid[0:h, 0:w]
	img = np.zeros((h, w, 4), np.uint8)
	img[..., 0] = x * 255 // max(1, w - 1)
	img[..., 1] = y * 255 // max(1, h - 1)
	img[..., 2] = 90
	img[..., 3] = 255 if alpha is None else alpha(x, y)
	return img


def noise(w=64, h=64, seed=0):
	img = np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)
	img[..., 3] = 255
	return img


# -- containers ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["I4", "I8", "IA4", "IA8", "RGB565", "RGB5A3", "RGBA8", "CMPR", "CI4", "CI8"])
def test_tpl_roundtrip_with_mips(fmt):
	src = gradient(32, 16, alpha=lambda x, y: np.where(x > 8, 255, 0))
	sizes = autotex.mip_sizes(32, 16, True, 4)
	levels = autotex.encode_levels(autotex.mip_chain(src, sizes), fmt)
	im = tpl.TplImage(levels)
	data = tpl.write_tpl([im, tpl.TplImage([gxtex.encode(gradient(8, 8), "RGB565")])])
	back = tpl.read_tpl(data)
	assert len(back) == 2 and back[0].fmt == fmt and len(back[0].levels) == len(sizes)
	for a, b in zip(back[0].levels, levels):
		assert (a.width, a.height, a.data) == (b.width, b.height, b.data)
	# the existing reader (gxbake decode, the app) sees the same top level
	assert np.array_equal(consoletex.read_tpl(data), levels[0].decode())
	# every data block is 32-byte aligned
	_, _, table = struct.unpack_from(">III", data, 0)
	ih, _ = struct.unpack_from(">II", data, table)
	assert struct.unpack_from(">I", data, ih + 8)[0] % 32 == 0


def test_bti_roundtrip_through_reader():
	idx, pal = autotex._quantize(gradient(16, 16), 16)
	t = gxtex.encode_indexed(idx, pal, gxtex.GX_TL_RGB565, "CI4")
	data = tpl.write_bti(tpl.TplImage([t]))
	assert np.array_equal(consoletex.read_bti(data), t.decode())


# -- automatic formats -----------------------------------------------------------------------------


def test_choose_picks_by_content():
	grey = gradient()
	grey[..., :3] = grey[..., :1]
	assert autotex.choose(autotex.TexJob("g", grey)).option.fmt in ("I4", "I8", "CMPR")
	ui = autotex.choose(autotex.TexJob("u", gradient(alpha=lambda x, y: (x * 4) % 256), formats=autotex.ROLES["ui"]))
	assert ui.option.fmt != "CMPR"
	flat = np.zeros((32, 32, 4), np.uint8)
	flat[..., 3] = 255
	flat[:16] = (200, 30, 30, 255)
	p = autotex.choose(autotex.TexJob("f", flat))
	assert p.option.db >= 38 and p.option.fmt in ("CMPR", "CI4")	# two colours: 4 bpp either way
	# a texture never grows past max_edge and keeps power-of-two sides
	big = autotex.choose(autotex.TexJob("b", gradient(300, 120), max_edge=128))
	assert (big.option.width, big.option.height) == (128, 64)


def test_fit_meets_budget_and_spends_it_where_cheapest():
	jobs = [autotex.TexJob("smooth", gradient(128, 128)), autotex.TexJob("busy", noise(128, 128))]
	free = autotex.fit(jobs)
	assert free.fits and all(p.option.width == 128 for p in free.placed)
	budget = free.total // 3
	r = autotex.fit(jobs, budget)
	assert r.fits and r.total <= budget
	by = {p.job.name: p for p in r.placed}
	# shrinking a smooth gradient costs far less picture than shrinking noise
	assert by["smooth"].option.width < by["busy"].option.width or by["smooth"].option.bytes < by["busy"].option.bytes
	assert autotex.fit(jobs, 40).fits is False	# two 8×8 CMPR levels are 64 bytes


# -- HDR ----------------------------------------------------------------------------------------------


def test_tonemaps_are_monotone_and_aces_matches_the_sky_converter():
	ramp = np.geomspace(1e-3, 1e3, 64, dtype=np.float32)
	rgb = np.repeat(ramp[None, :, None], 3, -1)
	for op in hdr.TONEMAPS:
		y = hdr.tonemap(rgb, op, auto=False)[0, :, 1].astype(int)
		assert (np.diff(y) >= 0).all() and y[0] < 40 and y[-1] > 200, op
	assert np.array_equal(skybox.tonemap(rgb), hdr.tonemap(rgb, "aces"))


def test_overbright_lightmap_scale():
	light = np.full((8, 8, 3), 0.5, np.float32)
	assert hdr.pick_overbright(light) == 1
	light[0, 0] = 3.0
	light[0, 1] = 3.0
	assert hdr.pick_overbright(light, clip_frac=0.0) == 4
	img, st = hdr.overbright(light, 4)
	assert st["tev_scale"] == "GX_CS_SCALE_4" and st["clipped"] == 0
	assert img[2, 2, 0] == round(0.5 / 4 * 255) and img[0, 0, 0] == round(3 / 4 * 255)


class _AxisSource:
	"""Red where a direction points along +X, green +Y, blue +Z (world)."""

	def sample(self, d):
		d = np.asarray(d, np.float64)
		d = d / np.linalg.norm(d, axis=1, keepdims=True)
		return np.clip(d, 0, 1) * 255


def test_sphere_map_orientation():
	m = hdr.sphere_map(_AxisSource(), 64, yaw=0.0)	# camera faces +X
	c = m[32, 32]	# centre reflects straight back at the camera: −X → black
	assert c[:3].max() < 40
	top = m[2, 32]	# near the top rim the reflection points behind and up... the ring is "forward"
	rim = hdr.sphere_directions(64)[0].reshape(64, 64, 3)[0, 32]
	assert rim[2] < 0	# the very edge reflects away from the viewer (forward, +X)
	assert top[0] > 100	# so it shows +X
	assert m[16, 32, 2] > m[48, 32, 2]	# upper half sees +Z (sky) more than the lower half


# -- PBR ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,expect", [
	("Bricks_BaseColor.png", ("bricks", "albedo")),
	("rock_wall_diff_2k.jpg", ("rock_wall", "albedo")),
	("rock_wall_nor_gl_2k.png", ("rock_wall", "normal")),
	("Metal01_Normal_DX.png", ("metal01", "normal_dx")),
	("crate_OcclusionRoughnessMetallic.png", ("crate", "orm")),
	("panel_MaskMap.tga", ("panel", "maskmap")),
	("tile_rough.png", ("tile", "roughness")),
	("photo.png", None),
])
def test_classify(name, expect):
	assert pbr.classify(name) == expect


def _height(n=64):
	y, x = np.mgrid[0:n, 0:n] / n
	return (0.5 + 0.5 * np.sin(x * 2 * np.pi * 3) * np.cos(y * 2 * np.pi * 2)).astype(np.float32)


def _normals_from_height(h, k=6.0):
	d_row, d_col = np.gradient(h)
	n = np.dstack([-d_col * k, d_row * k, np.ones_like(h)])	# +Y up: dh/drow = +ny/nz
	return n / np.linalg.norm(n, axis=-1, keepdims=True)


def test_height_from_normals_recovers_the_surface():
	h = _height()
	rec = pbr.height_from_normals(_normals_from_height(h))
	assert np.corrcoef(h.ravel(), rec.ravel())[0, 1] > 0.97


def write_material(folder, name="Bricks", metal=0, ao=200):
	folder.mkdir(parents=True, exist_ok=True)
	n = 64
	Image.fromarray(gradient(n, n)[..., :3]).save(folder / f"{name}_BaseColor.png")
	nm = (_normals_from_height(_height(n)) * 0.5 + 0.5) * 255
	Image.fromarray(nm.astype(np.uint8)).save(folder / f"{name}_Normal.png")
	orm = np.zeros((n, n, 3), np.uint8)
	orm[..., 0], orm[..., 1], orm[..., 2] = ao, 60, metal
	Image.fromarray(orm).save(folder / f"{name}_OcclusionRoughnessMetallic.png")
	Image.fromarray(gradient(n, n)[..., :3]).save(folder / "lonely_d.png")	# a lone map is not a set


def test_find_sets_and_load(tmp_path):
	write_material(tmp_path)
	sets = pbr.find_sets([tmp_path])
	assert list(sets) == ["bricks"] and set(sets["bricks"]) == {"albedo", "normal", "orm"}
	mat = pbr.load_material(tmp_path)[0]
	assert set(mat.maps) >= {"albedo", "normal", "ao", "roughness", "metallic"}
	assert abs(float(mat.maps["ao"].mean()) - 200 / 255) < 1e-3


@pytest.mark.parametrize("preset,maps", [
	("prelit", ["base"]), ("lit", ["base", "spec"]), ("emboss", ["base", "spec", "height"]),
	("indirect", ["base", "spec", "normal", "env"]),
])
def test_bake_presets(tmp_path, preset, maps):
	write_material(tmp_path)
	mat = pbr.load_material(tmp_path)[0]
	env = hdr.sphere_map(_AxisSource(), 32) if preset == "indirect" else None
	b = pbr.bake(mat, pbr.BakeOptions(preset=preset, env=env))
	assert list(b.images) == maps
	assert b.tev["tev_stages"] >= 1 and set(b.tev["texmaps"]) == set(maps)
	placed = autotex.fit(b.jobs).placed
	if "normal" in maps:
		assert placed[maps.index("normal")].image.fmt == "IA8"


def test_ao_darkens_lit_base(tmp_path):
	write_material(tmp_path / "a", ao=255)
	write_material(tmp_path / "b", ao=100)
	opts = pbr.BakeOptions(preset="lit", detail=0.0)
	bright = pbr.bake(pbr.load_material(tmp_path / "a")[0], opts).images["base"]
	dark = pbr.bake(pbr.load_material(tmp_path / "b")[0], opts).images["base"]
	assert dark[..., :3].mean() < bright[..., :3].mean() * 0.8


def _png_bytes(img):
	b = io.BytesIO()
	Image.fromarray(img).save(b, "PNG")
	return b.getvalue()


def _gltf_doc(uri_or_view):
	return {"asset": {"version": "2.0"}, "images": [uri_or_view], "textures": [{"source": 0}],
		"materials": [{"name": "Shiny Metal", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
			"metallicFactor": 1.0, "roughnessFactor": 0.2, "baseColorFactor": [1, 0.5, 0.5, 1]},
			"alphaMode": "MASK", "alphaCutoff": 0.3}]}


def test_gltf_and_glb(tmp_path):
	png = _png_bytes(gradient(16, 16)[..., :3])
	doc = _gltf_doc({"uri": "data:image/png;base64," + base64.b64encode(png).decode()})
	(tmp_path / "m.gltf").write_text(json.dumps(doc))
	m = pbr.load_material(tmp_path / "m.gltf")[0]
	assert m.name == "shiny_metal" and m.alpha_mode == "mask"
	assert float(m.maps["metallic"].mean()) == 1.0 and abs(float(m.maps["roughness"].mean()) - 0.2) < 1e-6
	assert m.maps["albedo"][..., 1].mean() < m.maps["albedo"][..., 0].mean()	# green factor 0.5
	# the same material in a GLB, image in the binary chunk
	blob = png + b"\0" * (-len(png) % 4)
	doc = _gltf_doc({"bufferView": 0, "mimeType": "image/png"})
	doc["buffers"] = [{"byteLength": len(blob)}]
	doc["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(png)}]
	js = json.dumps(doc).encode()
	js += b" " * (-len(js) % 4)
	glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)) \
		+ struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob
	(tmp_path / "m.glb").write_bytes(glb)
	g = pbr.load_material(tmp_path / "m.glb")[0]
	assert np.allclose(g.maps["albedo"], m.maps["albedo"])


# -- builds and the CLI -----------------------------------------------------------------------------


def make_project(root):
	(root / "tex" / "sub").mkdir(parents=True)
	Image.fromarray(noise(64, 64)).save(root / "tex" / "busy.png")
	Image.fromarray(gradient(64, 64)).save(root / "tex" / "sub" / "smooth.png")
	write_material(root / "mats")
	(root / "mats" / "lonely_d.png").unlink()
	lm = np.zeros((16, 16, 3), np.float32)
	lm[..., :] = np.linspace(0, 1.8, 16, dtype=np.float32)[None, :, None]
	pfm = b"PF\n16 16\n-1.0\n" + lm[::-1].astype("<f4").tobytes()
	(root / "lm").mkdir()
	(root / "lm" / "room.pfm").write_bytes(pfm)
	doc = {"build": {"output": "out", "budget": "24K", "min_db": 36},
		"textures": [{"inputs": ["tex"]}],
		"pbr": [{"inputs": ["mats"], "preset": "lit", "gct_base": "mats.gct"}],
		"hdr": [{"inputs": ["lm/*.pfm"], "mode": "lightmap", "budget": "8K"}]}
	(root / "gxbake.json").write_text(json.dumps(doc))
	return root / "gxbake.json"


def test_manifest_build_budget_and_incremental(tmp_path):
	man = make_project(tmp_path)
	rep = build.build(man)
	out = tmp_path / "out"
	assert rep.fits and rep.groups["shared"].total <= 24 * 1024
	for f in ("busy.tpl", "sub/smooth.tpl", "bricks.tpl", "bricks.gxmat.json", "mats.gct", "room.tpl", "room.lightmap.json"):
		assert (out / f).is_file(), f
	assert json.loads((out / "room.lightmap.json").read_text())["scale"] == 2
	assert [e.name for e in gct.read_gct((out / "mats.gct").read_bytes())] == ["bricks"]
	gx = json.loads((out / "bricks.gxmat.json").read_text())
	assert gx["preset"] == "lit" and gx["maps"][0]["map"] == "base"
	again = build.build(man)
	assert all(g.skipped for g in again.groups.values())
	(tmp_path / "tex" / "busy.png").write_bytes((tmp_path / "tex" / "busy.png").read_bytes())	# touch
	import os
	os.utime(tmp_path / "tex" / "busy.png", ns=(1, 1))
	third = build.build(man)
	assert not third.groups["shared"].skipped and third.groups["hdr#2"].skipped


def test_cli_auto_pbr_hdr_build(tmp_path, capsys):
	make_project(tmp_path)
	out = tmp_path / "o"
	assert cli.main(["auto", str(tmp_path / "tex"), "-o", str(out), "--combine", "all.tpl", "--json"]) == 0
	res = json.loads(capsys.readouterr().out)
	assert len(res["units"][0]["textures"]) == 2 and (out / "all.h").is_file()
	assert len(tpl.read_tpl((out / "all.tpl").read_bytes())) == 2
	assert cli.main(["auto", str(tmp_path / "tex" / "busy.png"), "-o", str(out), "--container", "bti", "--role", "world"]) == 0
	assert consoletex.read_bti((out / "busy.bti").read_bytes()).shape == (64, 64, 4)
	assert cli.main(["pbr", str(tmp_path / "mats"), "--list", "-o", str(out)]) == 0
	assert "bricks:" in capsys.readouterr().out
	assert cli.main(["pbr", str(tmp_path / "mats"), "-o", str(out), "--preset", "emboss"]) == 0
	assert [i.fmt for i in tpl.read_tpl((out / "bricks.tpl").read_bytes())][2] in ("I4", "I8")
	assert cli.main(["hdr", str(tmp_path / "lm" / "room.pfm"), "-o", str(out), "--mode", "sphere", "--size", "32"]) == 0
	assert tpl.read_tpl((out / "room.tpl").read_bytes())[0].width == 32
	assert cli.main(["auto", str(tmp_path / "tex"), "-o", str(out), "--budget", "40"]) == 2	# cannot fit
	capsys.readouterr()
	man = tmp_path / "new" / "gxbake.toml"
	man.parent.mkdir()
	assert cli.main(["build", "--init", str(man)]) == 0
	assert build.parse_rules(build.load_manifest(man))[1][0].kind == "textures"
