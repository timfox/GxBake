"""Quake and Quake II formats → GoldSrc/GameCube equivalents.

* Quake alias models (.mdl, "IDPO" v6) → ImportedModel → studio MDL. Alias
  models animate by moving vertices, which studio models cannot do, so the
  first frame is converted and the frame count is reported.
* Quake BSP v29 and WAD2 miptex: no palette is stored with the texture, so the
  Quake palette (gfx/palette.lmp, loose or in a .pak) is used for the GCWT pack.
* Quake sprites (.spr v1) and Quake II sprites (.sp2) → Half-Life v2 sprites,
  which the port draws and the lean HUD pass shrinks.
* Quake II wall textures (.wal) → RGBA with the Quake II palette
  (pics/colormap.pcx), then the usual CMPR path.

The palettes are game data and are not shipped here; they are found next to
the content. Without one, a greyscale ramp is used and a warning says so.
"""

from __future__ import annotations

import io
import math
import struct
from pathlib import Path

import numpy as np

from . import archives

GREY_PALETTE = bytes(np.repeat(np.arange(256, dtype=np.uint8), 3))

SPR_ALPHATEST = 3
SPR_VP_PARALLEL = 2


# -- palettes -------------------------------------------------------------------------


def _ancestors(start: Path, depth: int = 5) -> list[Path]:
	start = Path(start)
	start = start if start.is_dir() else start.parent
	return [start, *list(start.parents)[:depth]]


def _find_in_tree(start: Path, rel: str, subdirs: tuple[str, ...]) -> bytes | None:
	"""rel loose under an ancestor (or ancestor/<subdir>), else inside a .pak there."""
	for d in _ancestors(start):
		for base in (d, *(d / s for s in subdirs)):
			p = base / rel
			try:
				if p.is_file():
					return p.read_bytes()
			except OSError:
				continue
	for d in _ancestors(start):
		for base in (d, *(d / s for s in subdirs)):
			try:
				paks = sorted(base.glob("*.pak")) + sorted(base.glob("*.PAK"))
			except OSError:
				continue
			for pak in paks:
				data = archives.read_member(pak, rel)
				if data is not None:
					return data
	return None


_palette_cache: dict = {}


def quake_palette(start: Path) -> bytes | None:
	"""Quake's gfx/palette.lmp (768 bytes) near `start`, or None."""
	key = ("q1", str(Path(start).resolve().parent))
	if key not in _palette_cache:
		data = _find_in_tree(start, "gfx/palette.lmp", ("id1",))
		_palette_cache[key] = data[:768] if data and len(data) >= 768 else None
	return _palette_cache[key]


def quake2_palette(start: Path) -> bytes | None:
	"""Quake II's palette from pics/colormap.pcx near `start`, or None."""
	key = ("q2", str(Path(start).resolve().parent))
	if key not in _palette_cache:
		data = _find_in_tree(start, "pics/colormap.pcx", ("baseq2",))
		_palette_cache[key] = pcx_palette(data) if data else None
	return _palette_cache[key]


def pcx_palette(data: bytes) -> bytes | None:
	if len(data) > 769 and data[-769] == 0x0C:
		return data[-768:]
	return None


# -- miptex without a palette (BSP v29, WAD2) ---------------------------------------------


def quake_miptex(data: bytes, offset: int, palette: bytes):
	"""(name, w, h, indexed pixels, palette) like goldsrc.wt_read_miptex, using `palette`."""
	try:
		raw_name, w, h, m0 = struct.unpack_from("<16sIII", data, offset)
	except struct.error:
		return None
	if not m0 or not w or not h or w * h > 1 << 20 or offset + m0 + w * h > len(data):
		return None
	name = raw_name.split(b"\0")[0].decode("latin-1").lower()
	return name, w, h, data[offset + m0: offset + m0 + w * h], palette


def bsp_version(data: bytes) -> int:
	return struct.unpack_from("<i", data, 0)[0] if len(data) >= 4 else -1


# -- Quake II .wal ----------------------------------------------------------------------------


def read_wal(data: bytes, palette: bytes) -> "np.ndarray":
	name, w, h = struct.unpack_from("<32sII", data, 0)
	off = struct.unpack_from("<I", data, 40)[0]
	if not w or not h or off + w * h > len(data):
		raise ValueError("truncated WAL texture")
	idx = np.frombuffer(data, np.uint8, w * h, off).reshape(h, w)
	pal = np.frombuffer(palette, np.uint8).reshape(256, 3)
	rgba = np.empty((h, w, 4), np.uint8)
	rgba[..., :3] = pal[idx]
	rgba[..., 3] = np.where(idx == 255, 0, 255)	# 255 is see-through in Quake II's palette
	return rgba


def load_wal(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	pal = quake2_palette(path)
	if pal is None:
		if warnings is not None:
			warnings.append("Quake II palette (pics/colormap.pcx) not found; greyscale used")
		pal = GREY_PALETTE
	return read_wal(Path(path).read_bytes(), pal)


# -- sprites ------------------------------------------------------------------------------------


def _hl_header(typ: int, tex: int, radius: float, w: int, h: int, nframes: int, beam: float, sync: int,
		palette: bytes) -> bytes:
	return (b"IDSP" + struct.pack("<iiifiiifi", 2, typ, tex, radius, w, h, nframes, beam, sync)
		+ struct.pack("<h", 256) + palette[:768])


def quake_spr_to_hl(data: bytes, palette: bytes) -> bytes:
	"""Quake sprite (IDSP v1) → Half-Life sprite (v2): same frames, plus palette and texFormat."""
	if data[:4] != b"IDSP" or struct.unpack_from("<i", data, 4)[0] != 1:
		raise ValueError("not a Quake (v1) sprite")
	typ, radius, w, h, nframes, beam, sync = struct.unpack_from("<ifiiifi", data, 8)
	return _hl_header(typ, SPR_ALPHATEST, radius, w, h, nframes, beam, sync, palette) + data[36:]


def quake2_sp2_to_hl(path: Path, warnings: list[str] | None = None) -> bytes:
	"""Quake II sprite (IDS2) → Half-Life sprite: each frame's PCX becomes a v2 frame."""
	from PIL import Image

	path = Path(path)
	data = path.read_bytes()
	if data[:4] != b"IDS2":
		raise ValueError("not a Quake II sprite")
	_ver, nframes = struct.unpack_from("<ii", data, 4)
	frames = []
	palette = None
	for i in range(nframes):
		fw, fh, ox, oy, raw = struct.unpack_from("<iiii64s", data, 12 + i * 80)
		name = raw.split(b"\0")[0].decode("latin-1")
		pcx = _find_in_tree(path, name, ("baseq2",)) or _find_in_tree(path, Path(name).name, ())
		if pcx is None:
			raise ValueError(f"frame {i}: {name} not found")
		with Image.open(io.BytesIO(pcx)) as img:
			if img.mode != "P":
				raise ValueError(f"frame {i}: {name} is not an 8-bit PCX")
			if palette is None:
				palette = bytes(img.getpalette()[:768]).ljust(768, b"\0")
			idx = np.asarray(img, np.uint8)
		frames.append((-ox, oy, idx))
	if palette is None:
		raise ValueError("sprite has no frames")
	w = max(f[2].shape[1] for f in frames)
	h = max(f[2].shape[0] for f in frames)
	out = bytearray(_hl_header(SPR_VP_PARALLEL, SPR_ALPHATEST, math.hypot(w / 2, h / 2), w, h, len(frames), 0.0, 0,
		palette))
	for ox, oy, idx in frames:
		out += struct.pack("<iiiii", 0, ox, oy, idx.shape[1], idx.shape[0]) + idx.tobytes()
	return bytes(out)


def hl_sprite_bytes(path: Path, warnings: list[str] | None = None) -> bytes:
	"""Any supported sprite as Half-Life v2 sprite bytes (v2 input is returned unchanged)."""
	path = Path(path)
	data = path.read_bytes()
	if data[:4] == b"IDS2":
		return quake2_sp2_to_hl(path, warnings)
	if data[:4] == b"IDSP" and struct.unpack_from("<i", data, 4)[0] == 1:
		pal = quake_palette(path)
		if pal is None:
			if warnings is not None:
				warnings.append("Quake palette (gfx/palette.lmp) not found; greyscale used")
			pal = GREY_PALETTE
		return quake_spr_to_hl(data, pal)
	return data


# -- alias models ------------------------------------------------------------------------------------


def is_alias_mdl(head: bytes) -> bool:
	return head[:4] == b"IDPO" and len(head) >= 8 and struct.unpack_from("<i", head, 4)[0] == 6


def read_alias(data: bytes) -> dict:
	"""Header, first skin (indices), st verts, triangles and every frame's vertices."""
	if not is_alias_mdl(data):
		raise ValueError("not a Quake alias model (IDPO v6)")
	scale = np.array(struct.unpack_from("<3f", data, 8))
	origin = np.array(struct.unpack_from("<3f", data, 20))
	numskins, sw, sh, numverts, numtris, numframes, _sync, flags = struct.unpack_from("<8i", data, 48)
	pos = 84
	skin = None
	for _ in range(numskins):
		group = struct.unpack_from("<i", data, pos)[0]
		pos += 4
		if group == 0:
			if skin is None:
				skin = data[pos:pos + sw * sh]
			pos += sw * sh
		else:
			n = struct.unpack_from("<i", data, pos)[0]
			pos += 4 + 4 * n
			if skin is None:
				skin = data[pos:pos + sw * sh]
			pos += n * sw * sh
	st = np.frombuffer(data, "<i4", numverts * 3, pos).reshape(numverts, 3)
	pos += numverts * 12
	tri = np.frombuffer(data, "<i4", numtris * 4, pos).reshape(numtris, 4)
	pos += numtris * 16
	frames = []
	names = []
	for _ in range(numframes):
		typ = struct.unpack_from("<i", data, pos)[0]
		pos += 4
		count = 1
		if typ != 0:
			count = struct.unpack_from("<i", data, pos)[0]
			pos += 4 + 8 + 4 * count
		for _k in range(count):
			name = data[pos + 8:pos + 24].split(b"\0")[0].decode("latin-1")
			pos += 24
			v = np.frombuffer(data, np.uint8, numverts * 4, pos).reshape(numverts, 4)
			pos += numverts * 4
			frames.append(v[:, :3].astype(np.float64) * scale + origin)
			names.append(name)
	return {"skin": skin, "skin_size": (sw, sh), "st": st, "tris": tri, "frames": frames, "frame_names": names,
		"flags": flags, "numskins": numskins}


def import_alias(path: Path, opts=None):
	"""Quake alias model → models.ImportedModel (first frame, first skin)."""
	from . import models as _m

	opts = opts or _m.ModelOptions()
	path = Path(path)
	al = read_alias(path.read_bytes())
	warnings: list[str] = []
	sw, sh = al["skin_size"]
	pal = quake_palette(path)
	if pal is None:
		warnings.append("Quake palette (gfx/palette.lmp) not found; greyscale skin")
		pal = GREY_PALETTE
	if al["skin"] is not None and len(al["skin"]) == sw * sh:
		idx = np.frombuffer(al["skin"], np.uint8).reshape(sh, sw)
		rgba = np.empty((sh, sw, 4), np.uint8)
		rgba[..., :3] = np.frombuffer(pal, np.uint8).reshape(256, 3)[idx]
		rgba[..., 3] = 255
	else:
		rgba = np.full((8, 8, 4), 180, np.uint8)
	if len(al["frames"]) > 1:
		warnings.append(f"{len(al['frames'])} vertex-animation frames: studio models can't morph, "
			f"so frame 0 ({al['frame_names'][0] or 'unnamed'}) is converted")
	verts = al["frames"][0] * float(opts.scale)
	st, tris = al["st"], al["tris"]
	# Back-facing triangles shift on-seam vertices by half the skin width.
	keys: dict[tuple[int, int], int] = {}
	out_pos, out_uv, faces = [], [], []
	for front, a, b, c in tris:
		face = []
		for v in (a, b, c):
			shift = int(st[v, 0] != 0 and not front)
			k = (int(v), shift)
			if k not in keys:
				keys[k] = len(out_pos)
				out_pos.append(verts[v])
				s = st[v, 1] + (sw // 2 if shift else 0)
				out_uv.append(((s + 0.5) / sw, (st[v, 2] + 0.5) / sh))
			face.append(keys[k])
		faces.append(face)
	pos = np.array(out_pos, np.float64)
	f = np.array(faces, np.int64)
	# Quake fronts are clockwise; build_studio_mdl wants counter-clockwise. Check by volume.
	cen = pos.mean(0)
	vol = np.einsum("ij,ij->i", pos[f[:, 0]] - cen, np.cross(pos[f[:, 1]] - cen, pos[f[:, 2]] - cen)).sum()
	if vol < 0:
		f = f[:, [0, 2, 1]]
	fn = np.cross(pos[f[:, 1]] - pos[f[:, 0]], pos[f[:, 2]] - pos[f[:, 0]])
	nrm = np.zeros_like(pos)
	for k in range(3):
		np.add.at(nrm, f[:, k], fn)
	lens = np.linalg.norm(nrm, axis=1, keepdims=True)
	nrm = np.where(lens > 1e-9, nrm / np.maximum(lens, 1e-9), [0.0, 0.0, 1.0])
	skin = _m._fit_skin(rgba, opts.skin_max_edge)
	model = _m.ImportedModel(path.stem, [_m.Material(f"{path.stem}_skin", skin, False, "skin 0")], [], warnings)
	model.meshes.append(_m.Submesh(0, pos.astype(np.float32), nrm.astype(np.float32),
		np.array(out_uv, np.float32), f.astype(np.uint32)))
	model.sequences.append(_m.Sequence("idle", 10.0, model.rest_frames(), loop=True))
	return model
