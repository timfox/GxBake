"""A tiny synthetic Source BSP (v20): one floor slab under an open sky, so the map converter is tested without game data.

The slab spans x, y in [-128, 128] and z in [-16, 0]. Its top face is lit by a 17x17 LDR
lightmap and textured with test/floor, which the pakfile carries as a VMT and VTF along
with the six sides of the sky. With displacement=True the top face becomes a power-2
displacement with a 32-unit bump in its middle.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import numpy as np

from gxbake.core import source_bsp as sb

from .game_fixtures import vtf

# (normal, dist) of the slab's six sides; the node tree and the brush use the same order
SLAB_PLANES = [((0, 0, 1), 0.0), ((0, 0, -1), 16.0), ((1, 0, 0), 128.0), ((-1, 0, 0), 128.0),
	((0, 1, 0), 128.0), ((0, -1, 0), 128.0)]
SPAWN = (0.0, 64.0, 40.0)


def _floor_texture() -> "np.ndarray":
	y, x = np.mgrid[0:64, 0:64]
	img = np.zeros((64, 64, 4), np.uint8)
	img[..., 0] = 60 + (x // 8 % 2) * 120
	img[..., 1] = 90
	img[..., 2] = 60 + (y // 8 % 2) * 120
	img[..., 3] = 255
	return img


def pakfile(sky: str = "testsky") -> bytes:
	buf = io.BytesIO()
	with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
		z.writestr("materials/test/floor.vmt", '"LightmappedGeneric"\n{\n\t"$basetexture" "test/floor"\n}\n')
		z.writestr("materials/test/floor.vtf", vtf(_floor_texture()))
		for i, sfx in enumerate(("rt", "bk", "lf", "ft", "up", "dn")):
			img = np.full((32, 32, 4), (40 * i, 100, 200, 255), np.uint8)
			z.writestr(f"materials/skybox/{sky}{sfx}.vmt", f'"UnlitGeneric"\n{{\n\t"$basetexture" "skybox/{sky}{sfx}"\n}}\n')
			z.writestr(f"materials/skybox/{sky}{sfx}.vtf", vtf(img))
	return buf.getvalue()


def _sprp(props: list[tuple[str, tuple[float, float, float]]]) -> bytes:
	names = sorted({m for m, _ in props})
	out = struct.pack("<i", len(names)) + b"".join(n.encode().ljust(128, b"\0") for n in names)
	out += struct.pack("<i", 0)	# leaf entries
	out += struct.pack("<i", len(props))
	for model, origin in props:
		e = struct.pack("<6fHHHBBi", *origin, 0, 90, 0, names.index(model), 0, 0, 6, 0, 0)
		out += e.ljust(64, b"\0")
	return out


def source_bsp(displacement: bool = False, pak: bool = True, sky: str = "testsky",
		props: list[tuple[str, tuple[float, float, float]]] | None = None) -> bytes:
	props = [("models/props/crate01.mdl", (64.0, 0.0, 0.0))] if props is None else props
	lumps: dict[int, bytes] = {}

	planes = np.zeros(len(SLAB_PLANES), sb.PLANE)
	for i, (n, d) in enumerate(SLAB_PLANES):
		planes[i] = (n, d, int(np.argmax(np.abs(n))))
	lumps[sb.L_PLANES] = planes.tobytes()

	# the top face, clockwise seen from above
	quad = np.array([(-128, -128, 0), (-128, 128, 0), (128, 128, 0), (128, -128, 0)], np.float32)
	lumps[sb.L_VERTEXES] = quad.tobytes()
	lumps[sb.L_EDGES] = np.array([(0, 0), (0, 1), (1, 2), (2, 3), (3, 0)], "<u2").tobytes()
	lumps[sb.L_SURFEDGES] = np.array([1, 2, 3, 4], "<i4").tobytes()

	ti = np.zeros(1, sb.TEXINFO)
	ti[0]["vecs"] = [(1, 0, 0, 0), (0, -1, 0, 0)]
	ti[0]["lmvecs"] = [(1 / 16, 0, 0, 0), (0, -1 / 16, 0, 0)]
	lumps[sb.L_TEXINFO] = ti.tobytes()
	td = np.zeros(1, sb.TEXDATA)
	td[0] = ((0.5, 0.5, 0.5), 0, 64, 64, 64, 64)
	lumps[sb.L_TEXDATA] = td.tobytes()
	lumps[sb.L_TEXDATA_STRING_DATA] = b"TEST/FLOOR\0"
	lumps[sb.L_TEXDATA_STRING_TABLE] = struct.pack("<i", 0)

	# 17x17 luxels, brighter towards +x; ColorRGBExp32 with exponent 0
	u = np.tile(np.arange(17), 17)
	light = np.zeros((17 * 17, 4), np.uint8)
	light[:, 0] = 40 + u * 8
	light[:, 1] = 40 + u * 8
	light[:, 2] = 40 + u * 8
	lumps[sb.L_LIGHTING] = light.tobytes()

	face = np.zeros(1, sb.FACE)
	face[0]["planenum"] = 0
	face[0]["onnode"] = 1
	face[0]["numedges"] = 4
	face[0]["dispinfo"] = 0 if displacement else -1
	face[0]["fogvolume"] = -1
	face[0]["styles"] = (0, 255, 255, 255)
	face[0]["area"] = 256 * 256
	face[0]["lm_mins"] = (-8, -8)
	face[0]["lm_size"] = (16, 16)
	face[0]["origface"] = 0
	lumps[sb.L_FACES] = face.tobytes()

	if displacement:
		di = np.zeros(1, sb.DISPINFO)
		di[0]["start"] = (-128, -128, 0)
		di[0]["power"] = 2
		di[0]["contents"] = sb.CONTENTS_SOLID
		di[0]["mapface"] = 0
		lumps[sb.L_DISPINFO] = di.tobytes()
		dv = np.zeros(25, sb.DISPVERT)
		dv["vec"] = (0, 0, 1)
		dv["alpha"] = 0
		dv["dist"][12] = 32.0	# the middle vertex
		lumps[sb.L_DISP_VERTS] = dv.tobytes()

	# node k splits on plane k: in front is open air (leaf k + 1), behind is the next node;
	# behind the last one is the slab itself (leaf 0, solid)
	nodes = np.zeros(6, sb.NODE)
	for k in range(6):
		nodes[k]["planenum"] = k
		nodes[k]["children"] = (-(k + 2), k + 1 if k < 5 else -1)
		nodes[k]["mins"] = (-4096, -4096, -4096)
		nodes[k]["maxs"] = (4096, 4096, 4096)
	nodes[0]["numfaces"] = 1
	lumps[sb.L_NODES] = nodes.tobytes()

	leafs = np.zeros(7, sb.LEAF_V1)
	leafs["cluster"] = 0
	leafs["mins"] = (-4096, -4096, -4096)
	leafs["maxs"] = (4096, 4096, 4096)
	leafs["water"] = -1
	leafs[0]["contents"] = sb.CONTENTS_SOLID
	leafs[0]["cluster"] = -1
	leafs[0]["mins"] = (-128, -128, -16)
	leafs[0]["maxs"] = (128, 128, 0)
	leafs[0]["numleafbrushes"] = 1
	if not displacement:	# Source keeps displacement faces out of the leaf face lists
		leafs[1]["numleaffaces"] = 1
	lumps[sb.L_LEAFS] = leafs.tobytes()
	lumps[sb.L_LEAFFACES] = struct.pack("<H", 0)
	lumps[sb.L_LEAFBRUSHES] = struct.pack("<H", 0)
	lumps[sb.L_BRUSHES] = struct.pack("<iii", 0, 6, sb.CONTENTS_SOLID)
	bs = np.zeros(6, sb.BRUSHSIDE)
	bs["planenum"] = np.arange(6)
	bs["dispinfo"] = -1
	lumps[sb.L_BRUSHSIDES] = bs.tobytes()

	model = np.zeros(1, sb.MODEL)
	model[0] = ((-128, -128, -16), (128, 128, 0), (0, 0, 0), 0, 0, 1)
	lumps[sb.L_MODELS] = model.tobytes()

	# one cluster that sees itself
	lumps[sb.L_VISIBILITY] = struct.pack("<iii", 1, 12, 13) + b"\x01\x01"

	ents = [{"classname": "worldspawn", "skyname": sky, "detailmaterial": "detail/x"},
		{"classname": "info_player_teamspawn", "origin": " ".join(f"{c:g}" for c in SPAWN), "angles": "0 90 0"},
		{"classname": "light", "origin": "0 0 128", "_light": "255 255 255 200"}]
	lumps[sb.L_ENTITIES] = sb.format_entities(ents).encode() + b"\0"
	if pak:
		lumps[sb.L_PAKFILE] = pakfile(sky)

	# lay out the file; the game lump's sprp data follows its own directory, by absolute offset
	sprp = _sprp(props) if props else b""
	game_len = 4 + (16 + len(sprp) if sprp else 0)
	header = 8 + 16 * sb.HEADER_LUMPS + 4
	order = sorted(lumps) + [sb.L_GAME_LUMP]
	offsets: dict[int, int] = {}
	pos = header
	for i in order:
		offsets[i] = pos
		pos += (game_len if i == sb.L_GAME_LUMP else len(lumps[i]))
		pos = (pos + 3) & ~3
	if sprp:
		ofs = offsets[sb.L_GAME_LUMP] + 20
		lumps[sb.L_GAME_LUMP] = struct.pack("<i4sHHii", 1, b"prps", 0, 10, ofs, len(sprp)) + sprp
	else:
		lumps[sb.L_GAME_LUMP] = struct.pack("<i", 0)
	out = bytearray(pos)
	out[0:8] = b"VBSP" + struct.pack("<i", 20)
	for i in order:
		struct.pack_into("<iiii", out, 8 + 16 * i, offsets[i], len(lumps[i]), 1 if i == sb.L_LEAFS else 0, 0)
		out[offsets[i]:offsets[i] + len(lumps[i])] = lumps[i]
	struct.pack_into("<i", out, header - 4, 1)	# map revision
	return bytes(out)


def write_map(directory: Path, name: str = "slab", **kw) -> Path:
	p = Path(directory) / "maps" / f"{name}.bsp"
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_bytes(source_bsp(**kw))
	return p
