"""Radiosity lightmaps (gxbake gi): the ray caster, the light transport on small maps built in code,
the relit BSP, the GX atlas and its .gclm face table (Python and port/gc_lmap.c), build manifests.

The maps are synthetic boxes: these check that shadows, bounces, sky, texture lights, AO and the
PVS behave and that the files are consistent, not that a real Half-Life map looks like HLRAD's."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gxbake.core import bsp30, build, gxtex, radiosity as R, raytrace, rectpack, tpl
from tests.gi_fixtures import MapBuilder, room

ROOT = Path(__file__).resolve().parent.parent
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def face_index(b, tex, nz=None):
	for i, f in enumerate(b.grid.faces):
		if f.tex == tex and (nz is None or round(f.normal[2]) == nz):
			return i
	raise KeyError(tex)


def luxel_at(b, i, x, y):
	"""Light of face i's luxel nearest world (x, y)."""
	a, e = b.grid.start[i], b.grid.start[i + 1]
	p = b.grid.pos[a:e]
	k = int(np.argmin((p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2))
	return b.light[a + k]


# -- ray casting ----------------------------------------------------------------------------------


@pytest.mark.parametrize("native", [False, True])
def test_bvh_matches_brute_force(native):
	if native and raytrace.native() is None:
		pytest.skip(raytrace.backend())
	rng = np.random.default_rng(4)
	tris = rng.random((300, 1, 3)) * 50 + rng.normal(size=(300, 3, 3)) * 3
	mesh = raytrace.Mesh(tris)
	o = rng.random((9000, 3)) * 50	# > 4096 rays: the C path splits them over threads
	d = rng.normal(size=(9000, 3))
	d /= np.linalg.norm(d, axis=1, keepdims=True)
	h = mesh.trace(o, d, use_native=native)
	assert (h.tri == mesh.trace(o, d, use_native=not native).tri).all()
	o, d, h = o[:400], d[:400], raytrace.Hits(h.t[:400], h.tri[:400])
	v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
	for i in range(len(o)):
		p = np.cross(d[i], e2)
		det = (e1 * p).sum(1)
		s = o[i] - v0
		u = (s * p).sum(1) / det
		q = np.cross(s, e1)
		v = (q * d[i]).sum(1) / det
		t = (e2 * q).sum(1) / det
		ok = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
		want = int(np.argmin(np.where(ok, t, np.inf))) if ok.any() else -1
		assert h.tri[i] == want
		if want >= 0:
			assert h.t[i] == pytest.approx(t[want])
	assert (mesh.trace(o, d, 1e9, any_hit=True, use_native=native).hit == (h.tri >= 0)).all()
	short = mesh.trace(o, d, 0.5, use_native=native)
	assert ((short.tri < 0) | (short.t <= 0.5)).all()


def test_cosine_hemisphere_is_unit_and_upward():
	d = raytrace.cosine_hemisphere(64, 3)
	assert np.allclose(np.linalg.norm(d, axis=1), 1)
	assert (d[:, 2] >= 0).all()
	assert d[:, 2].mean() == pytest.approx(2 / 3, abs=0.05)	# E[cos] of a cosine-weighted hemisphere
	n = np.array([[1.0, 0, 0], [0, -1.0, 0]])
	w = raytrace.hemisphere_rays(n, d)
	assert (np.einsum("pkj,pj->pk", w, n) >= -1e-9).all()


# -- the map and its luxels -----------------------------------------------------------------------


def test_luxel_grid_matches_engine_extents():
	data = room()
	sc = R.load(data)
	b = R.bake(data, R.GiOptions(bounces=0))
	relit = R.relit_bsp(b)
	engine = {f: (w, h) for f, w, h, _o in rectpack.goldsrc_lightmaps(relit)}
	assert engine == {f.index: (f.w, f.h) for f in sc.faces if f.lit}
	# every luxel sample sits on its face's lit side, one unit off the plane
	for i, f in enumerate(b.grid.faces):
		p = b.grid.pos[b.grid.start[i]:b.grid.start[i + 1]]
		assert np.allclose(p @ f.normal - f.dist, 1.0, atol=1e-6)


def test_point_light_casts_a_shadow_and_bounces_fill_it():
	data = room("point")
	b0 = R.bake(data, R.GiOptions(bounces=0))
	b3 = R.bake(data, R.GiOptions(bounces=3))
	fl = face_index(b0, "floor")
	# light at x=128: the floor at x=360 behind the pillar (224..288) is in shadow, x=360 off to the side is not
	behind = luxel_at(b0, fl, 360, 256).mean()
	side = luxel_at(b0, fl, 360, 80).mean()
	assert behind < 0.01 < side
	assert luxel_at(b3, fl, 360, 256).mean() > behind + 0.01	# bounce light reaches the shadow
	assert b3.light.mean() > b0.light.mean()
	e = b3.stats["bounce_energy"]
	assert e[0] > 0 and e[0] <= e[1] <= e[2] and e[2] - e[1] < e[1] - e[0]	# converging


def test_spot_light_lights_only_its_cone_and_the_ceiling_by_bounce():
	data = room("spot_down", pillar=False)
	b0 = R.bake(data, R.GiOptions(bounces=0))
	b2 = R.bake(data, R.GiOptions(bounces=2))
	fl = face_index(b0, "floor")
	ce = face_index(b0, "wall", -1)
	assert luxel_at(b0, fl, 128, 256).mean() > 0.05	# under the light
	assert luxel_at(b0, fl, 450, 450).mean() == 0	# outside the 40° cone
	assert b0.face_light(ce).max() == 0
	assert b2.face_light(ce).mean() > 0
	# the floor is red: so is the light it bounces onto the ceiling
	c = b2.face_light(ce).reshape(-1, 3).mean(0)
	assert c[0] > 2 * c[2]


def test_sky_light_and_sun():
	data = room(light="", sky=True)
	b = R.bake(data, R.GiOptions(bounces=0))
	fl = face_index(b, "floor")
	assert b.stats["lights"]["sun"] == 1
	lit = b.face_light(fl).reshape(-1, 3)
	assert lit.max() > 0.3
	# the pillar shades the floor from the sun; that floor is lit by the blue sky only
	sun = np.array(R._direction({"angles": "-60 30 0", "pitch": "-60"}))
	assert sun[2] < 0
	tip = np.array([256, 256, 160.0])
	shadow = tip + sun * (160 / -sun[2])	# where the pillar top's shadow lands
	mid = (tip[:2] + shadow[:2]) / 2
	sh = luxel_at(b, fl, *mid)
	assert sh.mean() < 0.5 * lit.mean(1).max()
	assert sh[2] > sh[0]	# blue sky


def test_texture_lights_are_area_lights():
	data = room(light="", texlight=True, pillar=False)
	dark = R.bake(data, R.GiOptions(bounces=0))
	assert dark.light.max() == 0
	b = R.bake(data, R.GiOptions(bounces=1, texlights={"~light": "255 255 230 800"}))
	assert b.stats["emissive_samples"] > 4
	fl = face_index(b, "floor")
	under = luxel_at(b, fl, 256, 256).mean()
	corner = luxel_at(b, fl, 20, 20).mean()
	assert under > 2 * corner > 0
	# the light's own face shows its glow
	assert b.face_light(face_index(b, "~light")).min() > 0.3
	# the same light from a lights.rad file
	rad = R.parse_rad("// comment\n~LIGHT 255 255 230 800\nbad line\n")
	assert rad == {"~light": (255.0, 255.0, 230.0, 800.0)}


def test_ambient_occlusion_darkens_corners_not_open_floor():
	data = room("point", pillar=True)
	a0 = R.bake(data, R.GiOptions(bounces=2))
	a1 = R.bake(data, R.GiOptions(bounces=2, ao=1.0))
	fl = face_index(a0, "floor")
	open_ = luxel_at(a0, fl, 120, 420).mean(), luxel_at(a1, fl, 120, 420).mean()
	nook = luxel_at(a0, fl, 216, 256).mean(), luxel_at(a1, fl, 216, 256).mean()	# beside the pillar
	assert open_[1] == pytest.approx(open_[0], rel=0.05)
	assert nook[1] < nook[0]


def test_pvs_culls_lights_in_unseen_leaves():
	"""One node splits the room at x = 256 into leaves 1 and 2, each seeing only itself:
	a light in leaf 1 must not reach leaf 2 even though nothing blocks it."""
	m = MapBuilder()
	m.box((0, 0, 0), (512, 256, 128), "wall", (128, 128, 128), inward=True)
	b = m.b
	p, flip = b.planes.add((1.0, 0, 0), 256.0)
	assert not flip
	b.nodes.append((p, -2, -3, (0, 0, 0), (512, 256, 128), 0, 0))	# front (x ≥ 256) → leaf 1
	rows = [bytes([0b01]), bytes([0b10])]
	vis = bytearray()
	ofs = []
	for r in rows:
		ofs.append(len(vis))
		vis += bsp30.compress_vis(r)
	b.visibility = bytes(vis)
	b.leafs = [(-2, -1, (0, 0, 0), (0, 0, 0), 0, 0, (0, 0, 0, 0)),
		(-1, ofs[0], (256, 0, 0), (512, 256, 128), 0, 0, (0, 0, 0, 0)),
		(-1, ofs[1], (0, 0, 0), (256, 256, 128), 0, 0, (0, 0, 0, 0))]
	data = m.finish('{\n"classname" "worldspawn"\n}\n{\n"classname" "light"\n"origin" "400 128 64"\n"_light" "255 255 255 200"\n}\n')
	data = bytearray(data)
	struct.pack_into("<i", data, struct.unpack_from("<i", data, 4 + 14 * 8)[0] + 52, 2)	# models[0].visleafs
	data = bytes(data)
	culled = R.bake(data, R.GiOptions(bounces=0))
	open_ = R.bake(data, R.GiOptions(bounces=0, use_vis=False))
	fl = face_index(culled, "wall", 1)
	assert luxel_at(open_, fl, 100, 128).mean() > 0
	assert luxel_at(culled, fl, 100, 128).mean() == 0
	assert luxel_at(culled, fl, 400, 128).mean() == pytest.approx(luxel_at(open_, fl, 400, 128).mean())


# -- outputs --------------------------------------------------------------------------------------


def test_relit_bsp_lighting_lump():
	data = room()
	b = R.bake(data, R.GiOptions(bounces=1))
	out = R.relit_bsp(b)
	lumps = bsp30.read_lumps(out)
	old = bsp30.read_lumps(data)
	for i in range(15):
		if i not in (R.L_FACES, R.L_LIGHTING):
			assert lumps[i] == old[i]
	light = np.frombuffer(lumps[R.L_LIGHTING], np.uint8)
	assert len(light) == 3 * b.grid.n
	for i, f in enumerate(b.grid.faces):
		st = struct.unpack_from("<4Bi", lumps[R.L_FACES], f.index * 20 + 12)
		assert st[:4] == (0, 255, 255, 255)
		got = light[st[4]:st[4] + f.w * f.h * 3].reshape(f.h, f.w, 3)
		assert (got == R.to_bytes(b.face_light(i), 0.55)).all()


def test_calibrate_matches_existing_lighting():
	data = room()
	ref = R.relit_bsp(R.bake(data, R.GiOptions(bounces=1)))
	b = R.bake(ref, R.GiOptions(bounces=1, light_scale=3.0, calibrate=True))
	assert b.stats["calibration"] == pytest.approx(1 / 3, rel=0.08)


def _decode_page(placed, i):
	return placed[i].image.decode()[..., :3].astype(np.float64)


def test_atlas_layout_alignment_and_content(tmp_path):
	res = R.run(_write(tmp_path, room(texlight=True)), tmp_path / "out", R.GiOptions(bounces=1,
		texlights={"~light": "255 255 230 800"}), fixed="RGB565", shrink=False, page=128)
	at = res.atlas
	assert at.scale in (1, 2, 4)
	assert all(p.image.fmt == "RGB565" for p in res.placed)
	# rectangles (with padding) do not overlap
	rects = {}
	for af in at.faces:
		rects.setdefault(af.page, []).append((af.x - 1, af.y - 1, af.w + 2, af.h + 2))
	for rs in rects.values():
		for a in range(len(rs)):
			for c in range(a + 1, len(rs)):
				x0, y0, w0, h0 = rs[a]
				x1, y1, w1, h1 = rs[c]
				assert not (x0 < x1 + w1 and x1 < x0 + w0 and y0 < y1 + h1 and y1 < y0 + h0)
	# each face's texels hold its light (port tone: 2·light^0.55 / scale), RGB565 accuracy
	pages = [_decode_page(res.placed, i) for i in range(len(res.placed))]
	for i, af in enumerate(at.faces):
		want = np.clip(2 * np.maximum(res.bake.face_light(i), 0) ** 0.55 / at.scale, 0, 1) * 255
		got = pages[af.page][af.y:af.y + af.h, af.x:af.x + af.w]
		assert np.abs(got - want).max() <= 9
		# the padding repeats the edge
		assert np.abs(pages[af.page][af.y - 1, af.x:af.x + af.w] - got[0]).max() <= 1


def test_atlas_cmpr_blocks_never_straddle_faces(tmp_path):
	res = R.run(_write(tmp_path, room()), tmp_path / "out", R.GiOptions(bounces=0), fixed="CMPR", page=64)
	assert all(p.image.fmt == "CMPR" for p in res.placed)
	for af in res.atlas.faces:
		assert (af.x - 1) % 4 == 0 and (af.y - 1) % 4 == 0
	for w, h in res.atlas.sizes:
		assert w & (w - 1) == 0 and h & (h - 1) == 0	# power-of-two pages


def test_atlas_shrinks_flat_faces_and_pages_hold_them(tmp_path):
	b = R.bake(room(light="", sky=True), R.GiOptions(bounces=0))
	full = R.atlas(b, 256, shrink=False)
	small = R.atlas(b, 256, shrink=True)
	assert small.stats["stored_luxels"] < full.stats["stored_luxels"]
	assert sum(w * h for w, h in small.sizes) <= sum(w * h for w, h in full.sizes)
	assert small.stats["shrunk_faces"] > 0


def test_auto_format_picks_intensity_for_grey_light(tmp_path):
	# a grey room lit by a white light gives grey lightmaps: I8/I4 hold them in half the bytes of RGB565
	res = R.run(_write(tmp_path, room(floor=(160, 160, 160))), tmp_path / "out", R.GiOptions(bounces=1),
		formats=("RGB565", "I4", "I8"), tone="linear")
	assert {p.image.fmt for p in res.placed} <= {"I4", "I8"}


def test_overbright_scale_tracks_brightness(tmp_path):
	data = room()
	dim = R.atlas(R.bake(data, R.GiOptions(bounces=0, light_scale=0.05)), 256)
	hot = R.atlas(R.bake(data, R.GiOptions(bounces=0, light_scale=20)), 256)
	assert dim.scale <= 2 and hot.scale == 4
	lin = R.atlas(R.bake(data, R.GiOptions(bounces=0, light_scale=0.05)), 256, tone="linear")
	assert lin.scale == 1


def _write(tmp_path, data, name="m.bsp"):
	p = tmp_path / name
	p.write_bytes(data)
	return p


def test_gclm_round_trip_and_uv(tmp_path):
	res = R.run(_write(tmp_path, room()), tmp_path / "out", R.GiOptions(bounces=0), shrink=True)
	g = R.read_gclm((tmp_path / "out" / "m.gclm").read_bytes())
	assert g["scale"] == res.atlas.scale and g["flags"] == R.GCLM_FLAG_PORT_TONE
	assert len(g["faces"]) == len(res.bake.scene.faces)
	assert g["pages"] == [tuple(s) for s in res.atlas.sizes]
	lit = {af.face for af in res.atlas.faces}
	for k, e in enumerate(g["faces"]):
		assert (e is not None) == (k in lit)
	# the UV of the face's first and last luxel land on the stored grid's corner texel centres
	e = next(e for e in g["faces"] if e and e["w"] > 1 and e["h"] > 1)
	pw, ph = g["pages"][e["page"]]
	u, v = R.atlas_uv(e, 0, 0, pw, ph)
	assert (u * pw, v * ph) == pytest.approx((e["x"] + 0.5, e["y"] + 0.5))
	u, v = R.atlas_uv(e, e["src_w"] - 1, e["src_h"] - 1, pw, ph)
	assert (u * pw, v * ph) == pytest.approx((e["x"] + e["w"] - 0.5, e["y"] + e["h"] - 0.5))
	# TPL pages and the .gxm
	images = tpl.read_tpl((tmp_path / "out" / "m_lm.tpl").read_bytes())
	assert [(im.width, im.height) for im in images] == [tuple(s) for s in res.atlas.sizes]
	gxm = (tmp_path / "out" / "m_lm.gxm").read_bytes()
	assert gxm[:4] == b"GXM1" and gxm[7] == res.atlas.scale
	report = json.loads((tmp_path / "out" / "m_gi.json").read_text())
	assert report["atlas"]["bytes"] == sum(p.image.size for p in res.placed)


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_port_reader_matches_python(tmp_path):
	exe = tmp_path / "test_gc_lmap"
	subprocess.run([CC, "-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", "-O2", str(ROOT / "port" / "gc_lmap.c"),
		str(ROOT / "port" / "test_gc_lmap.c"), "-o", str(exe)], check=True)
	R.run(_write(tmp_path, room(texlight=True)), tmp_path / "out", R.GiOptions(bounces=0,
		texlights={"~light": "255 255 230 800"}), page=64)
	path = tmp_path / "out" / "m.gclm"
	g = R.read_gclm(path.read_bytes())
	out = subprocess.run([str(exe), str(path)], check=True, capture_output=True, text=True).stdout.splitlines()
	head = out[0].split()
	assert [int(x) for x in head[:5]] == [g["version"], g["flags"], len(g["faces"]), len(g["pages"]), g["scale"]]
	assert [tuple(map(int, ln.split()[1:])) for ln in out[1:1 + len(g["pages"])]] == g["pages"]
	for ln, e in zip(out[1 + len(g["pages"]):], g["faces"]):
		parts = ln.split()
		if e is None:
			assert parts[1] == "-"
		else:
			assert [int(x) for x in parts[1:]] == [e["page"], e["x"], e["y"], e["w"], e["h"], e["src_w"], e["src_h"]]
	for k, e in enumerate(g["faces"]):
		if e is None:
			continue
		for s, t in ((0, 0), (e["src_w"] - 1, 0), (1.25, 0.5), (e["src_w"] / 2, e["src_h"] - 1)):
			res = subprocess.run([str(exe), str(path), str(k), str(s), str(t)], check=True, capture_output=True, text=True)
			u, v = map(float, res.stdout.split())
			pw, ph = g["pages"][e["page"]]
			assert (u, v) == pytest.approx(R.atlas_uv(e, s, t, pw, ph), abs=1e-5)
	bad = tmp_path / "bad.gclm"
	bad.write_bytes(b"GCLM" + bytes(28))
	assert subprocess.run([str(exe), str(bad)], capture_output=True, text=True).stdout.strip() == "bad"


def test_build_manifest_bakes_each_map_on_its_own(tmp_path):
	(tmp_path / "maps").mkdir()
	(tmp_path / "maps" / "a.bsp").write_bytes(room())
	(tmp_path / "maps" / "b.bsp").write_bytes(room(light="", sky=True))
	(tmp_path / "lights.rad").write_text("~light 255 255 230 800\n")
	(tmp_path / "gxbake.toml").write_text('[build]\noutput = "out"\n\n[[gi]]\ninputs = ["maps/*.bsp"]\n'
		'rad = "lights.rad"\nbounces = 1\nrays = 32\nbudget = "64K"\n')
	rep = build.build(tmp_path / "gxbake.toml")
	assert set(rep.groups) == {"gi#0:a", "gi#0:b"}
	for name in ("a", "b"):
		for suffix in (".bsp", "_lm.tpl", ".gclm", "_lm.gxm", "_gi.json"):
			assert (tmp_path / "out" / f"{name}{suffix}").is_file()
		g = rep.groups[f"gi#0:{name}"]
		assert g.budget == 64 * 1024 and g.fits
	again = build.build(tmp_path / "gxbake.toml")
	assert all(g.skipped for g in again.groups.values())
	(tmp_path / "lights.rad").write_text("~light 255 255 230 900\n")
	third = build.build(tmp_path / "gxbake.toml")
	assert not any(g.skipped for g in third.groups.values())	# the .rad is an input
	dry = build.build(tmp_path / "gxbake.toml", force=True, dry_run=True)
	assert all(not r.placed for g in dry.groups.values() for r in g.units)


def test_cli_gi(tmp_path, capsys):
	from gxbake import cli

	p = _write(tmp_path, room(texlight=True))
	code = cli.main(["gi", str(p), "-o", str(tmp_path / "o"), "--bounces", "1", "--rays", "32",
		"--texlight", "~light=255 255 230 800", "--ao", "0.5", "--budget", "32K", "--preview"])
	assert code == 0
	out = capsys.readouterr().out
	assert "atlas:" in out and "m.gclm" in out
	assert (tmp_path / "o" / "preview").is_dir()
	code = cli.main(["gi", str(p), "-o", str(tmp_path / "j"), "--bounces", "0", "--no-atlas", "--json"])
	assert code == 0
	rep = json.loads(capsys.readouterr().out)
	assert "atlas" not in rep and (tmp_path / "j" / "m.bsp").is_file()
