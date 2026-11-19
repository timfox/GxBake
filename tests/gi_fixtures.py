"""Small GoldSrc maps built in code for the lightmap baker's tests: a closed room (or one open to
the sky) with a pillar, lights, a texture light and a sky option. Only the lumps the baker reads
are real (no nodes, leafs or visibility)."""

from __future__ import annotations

import numpy as np

from gxbake.core import bsp30

# Quake's texture axes per dominant normal (s, t)
_AXES = {0: ((0, 1, 0), (0, 0, -1)), 1: ((1, 0, 0), (0, 0, -1)), 2: ((1, 0, 0), (0, -1, 0))}


def _miptex(name: str, rgb) -> bsp30.Miptex:
	pal = bytes(rgb) + bytes(765)
	mips = [bytes((16 >> k) * (16 >> k)) for k in range(4)]	# every pixel palette index 0
	return bsp30.Miptex(name, 16, 16, mips, pal)


class MapBuilder:
	def __init__(self):
		self.b = bsp30.Bsp30()
		self.texids: dict[str, int] = {}
		self.tinfo: dict[tuple, int] = {}

	def texture(self, name: str, rgb=(128, 128, 128)) -> int:
		if name not in self.texids:
			self.texids[name] = len(self.b.textures)
			self.b.textures.append(_miptex(name, rgb))
		return self.texids[name]

	def _texinfo(self, normal, tex: int, flags: int) -> int:
		ax = int(np.argmax(np.abs(normal)))
		s, t = _AXES[ax]
		key = (ax, tex, flags)
		if key not in self.tinfo:
			self.tinfo[key] = len(self.b.texinfo)
			self.b.texinfo.append(([[*s, 0.0], [*t, 0.0]], tex, flags))
		return self.tinfo[key]

	def quad(self, pts, tex: str, rgb=(128, 128, 128), flags: int = 0):
		"""A convex polygon, front side by the right-hand rule (counter-clockwise seen from the front)."""
		pts = np.asarray(pts, np.float64)
		n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
		n /= np.linalg.norm(n)
		# GoldSrc winds faces clockwise seen from the front; the baker does not care, but keep it
		pidx, flipped = self.b.planes.add(n, float(n @ pts[0]))
		first = len(self.b.surfedges)
		for i in range(len(pts)):
			a, c = pts[i], pts[(i + 1) % len(pts)]
			ia = self._vert(a)
			ic = self._vert(c)
			self.b.edges.append((ia, ic))
			self.b.surfedges.append(len(self.b.edges) - 1)
		ti = self._texinfo(n, self.texture(tex, rgb), flags)
		self.b.faces.append((pidx, 1 if flipped else 0, first, len(pts), ti, (255, 255, 255, 255), -1))

	def _vert(self, p) -> int:
		self.b.vertexes.append(tuple(float(x) for x in p))
		return len(self.b.vertexes) - 1

	def box(self, lo, hi, tex: str, rgb=(128, 128, 128), inward: bool = False, skip=(), textures=None):
		"""Six faces of an axis-aligned box. skip: face names to leave out (e.g. "top")."""
		x0, y0, z0 = lo
		x1, y1, z1 = hi
		f = {
			"bottom": [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],
			"top": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
			"west": [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
			"east": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
			"south": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
			"north": [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],
		}
		for name, pts in f.items():
			if name in skip:
				continue
			tname, trgb = (textures or {}).get(name, (tex, rgb))
			self.quad(pts[::-1] if inward else pts, tname, trgb, 1 if tname.startswith("sky") else 0)

	def finish(self, entities: str) -> bytes:
		n = len(self.b.faces)
		vs = np.array(self.b.vertexes)
		self.b.models.append((tuple(vs.min(0)), tuple(vs.max(0)), (0.0, 0.0, 0.0), (0, 0, 0, 0), 0, 0, n))
		self.b.entities = entities
		return self.b.write()


def room(light: str = "point", pillar: bool = True, sky: bool = False, texlight: bool = False,
		size=(512, 512, 256), wall=(160, 160, 160), floor=(200, 60, 60)) -> bytes:
	"""A closed room (floor red, walls grey), optionally a pillar, its ceiling sky, or a light panel."""
	m = MapBuilder()
	x, y, z = size
	tex = {"bottom": ("floor", floor)}
	if sky:
		tex["top"] = ("sky", (0, 0, 0))
	m.box((0, 0, 0), size, "wall", wall, inward=True, textures=tex)
	if pillar:
		m.box((224, 224, 0), (288, 288, 160), "pillar", (180, 180, 180))
	if texlight:
		m.quad([(192, 192, z - 1), (320, 192, z - 1), (320, 320, z - 1), (192, 320, z - 1)][::-1], "~light", (255, 255, 230))
	ents = ['{\n"classname" "worldspawn"\n"wad" "\\\\half-life\\\\valve\\\\halflife.wad"\n}\n']
	if light == "point":
		ents.append('{\n"classname" "light"\n"origin" "128 256 200"\n"_light" "255 255 255 300"\n}\n')
	elif light == "spot_down":
		ents.append('{\n"classname" "light_spot"\n"origin" "128 256 240"\n"pitch" "-90"\n"_cone" "20"\n"_cone2" "40"\n'
			'"_light" "255 255 255 400"\n}\n')
	if sky:
		ents.append('{\n"classname" "light_environment"\n"angles" "-60 30 0"\n"pitch" "-60"\n"_light" "255 240 220 200"\n'
			'"_diffuse_light" "120 160 255 120"\n}\n')
	return m.finish("".join(ents))
