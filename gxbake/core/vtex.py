"""Source 2 textures (.vtex_c) and materials (.vmat_c).

A vtex_c is decoded to RGBA (largest mip, first face/slice) and then takes the
same CMPR path as any loose image. Block formats go through Pillow's BCn
decoder (BC1, BC3, BC4, BC5, BC7); uncompressed, float and raw JPEG/PNG/WebP
textures are handled here. HDR formats are tone mapped to 8 bits so a Source 2
sky cubemap can become the port's six sky sides.

A vmat_c names its textures in m_textureParams; material_texture() picks the
colour map the GameCube can use (g_tColor, g_tColorA, ...).

Header layout and mip ordering follow ValveResourceFormat (MIT, see
THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import valvecomp, vres

FORMATS = {
	1: "DXT1", 2: "DXT5", 3: "I8", 4: "RGBA8888", 5: "R16", 6: "RG1616", 7: "RGBA16161616", 8: "R16F",
	9: "RG1616F", 10: "RGBA16161616F", 11: "R32F", 12: "RG3232F", 13: "RGB323232F", 14: "RGBA32323232F",
	15: "JPEG_RGBA8888", 16: "PNG_RGBA8888", 17: "JPEG_DXT5", 18: "PNG_DXT5", 19: "BC6H", 20: "BC7", 21: "ATI2N",
	22: "IA88", 23: "ETC2", 24: "ETC2_EAC", 25: "R11_EAC", 26: "RG11_EAC", 27: "ATI1N", 28: "BGRA8888",
	29: "WEBP_RGBA8888", 30: "WEBP_DXT5", 31: "R32_UINT", 32: "A8", 33: "R8_UNORM",
}
_BLOCK = {1: 8, 2: 16, 19: 16, 20: 16, 21: 16, 23: 8, 24: 16, 25: 8, 26: 16, 27: 8}
_PIXEL = {3: 1, 4: 4, 5: 2, 6: 4, 7: 8, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 12, 14: 16, 22: 2, 28: 4, 31: 4,
	32: 1, 33: 1}
_RAW_IMAGE = {15, 16, 17, 18, 29, 30}
_LEGACY_FORMATS = {0: 4, 5: 3, 13: 1, 15: 2, 79: 6}	# old (version 0) ImageFormat -> VTexFormat

FLAG_CUBE = 1 << 4
FLAG_VOLUME = 1 << 5
FLAG_ARRAY = 1 << 6


@dataclass
class VTex:
	res: vres.Resource
	width: int
	height: int
	depth: int
	format: int
	mips: int
	flags: int
	display: tuple[int, int] = (0, 0)
	mip_compression: int = 0
	compressed_mips: list[int] | None = None
	codec: set[str] = field(default_factory=set)

	@property
	def format_name(self) -> str:
		return FORMATS.get(self.format, f"format {self.format}")

	@property
	def cube(self) -> bool:
		return bool(self.flags & FLAG_CUBE)

	@property
	def size(self) -> tuple[int, int]:
		dw, dh = self.display
		if dw > 0 and dh > 0 and dw <= self.width and dh <= self.height:
			return dw, dh
		return self.width, self.height

	def info(self) -> dict:
		w, h = self.size
		out = {"Size": f"{w}×{h}", "Format": self.format_name, "Mips": self.mips}
		if self.cube:
			out["Cubemap"] = "yes"
		if self.depth > 1:
			out["Depth"] = self.depth
		if self.codec:
			out["Encoding"] = ", ".join(sorted(self.codec))
		return out

	def _level_dims(self, level: int) -> tuple[int, int, int]:
		w = max(1, self.width >> level)
		h = max(1, self.height >> level)
		d = self.depth if not self.flags & FLAG_VOLUME else max(1, self.depth >> level)
		if self.cube:
			d *= 6
		return w, h, d

	def level_bytes(self, level: int) -> int:
		w, h, d = self._level_dims(level)
		if self.format in _BLOCK:
			bw = max(4, (w + 3) // 4 * 4)
			bh = max(4, (h + 3) // 4 * 4)
			if 1 < d < 4:
				d = 4
			return (bw * bh // 16) * d * _BLOCK[self.format]
		return w * h * d * _PIXEL.get(self.format, 1)

	def read_level(self, level: int = 0) -> bytes:
		"""Raw (decompressed) bytes of one mip level, all faces and slices."""
		pos = self.res.data_end
		for j in range(self.mips - 1, level, -1):	# smallest mips are stored first
			size = self.level_bytes(j)
			if self.compressed_mips is not None:
				size = min(size, self.compressed_mips[j])
			pos += size
		size = self.level_bytes(level)
		data = self.res.data
		if self.compressed_mips is None or self.mip_compression == 0 or self.compressed_mips[level] >= size:
			return bytes(data[pos:pos + size])
		csize = self.compressed_mips[level]
		if self.mip_compression == 1:
			return valvecomp.lz4_block(data[pos:pos + csize], size)
		return valvecomp.zstd(data[pos:pos + csize], size)[:size]

	def rgba(self, face: int = 0, level: int = 0) -> "np.ndarray":
		"""H×W×4 uint8 of one face (cubemaps: +X -X +Y -Y +Z -Z) of one mip."""
		if self.format in _RAW_IMAGE:
			from PIL import Image

			raw = self.res.data[self.res.data_end:]
			return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA")).copy()
		level = min(level, max(0, self.mips - 1))
		raw = self.read_level(level)
		w, h, d = self._level_dims(level)
		per = len(raw) // max(1, d)
		raw = raw[face * per:(face + 1) * per] if self.cube else raw[:per]
		img = decode(self.format, raw, w, h)
		dw, dh = self.size
		if level == 0 and (dw, dh) != (w, h):
			img = img[:dh, :dw]
		return _convert(img, self.codec, self.format, self.cube)


def _codec(res: vres.Resource) -> set[str]:
	info = res.raw("RED2") or res.raw("REDI")
	out = set()
	if b"Image YCoCg Conversion" in info:
		out.add("YCoCg")
	if b"Mip HemiOctNormal" in info or b"HemiOctIsoRoughness" in info or b"HemiOctAnisoRoughness" in info:
		out.add("HemiOct")
	if b"Image NormalizeNormals" in info:
		out.add("normal map")
	return out


def parse(res: vres.Resource) -> VTex:
	b = res.block("DATA")
	if b is None:
		raise ValueError("texture has no DATA block")
	d = res.data
	p = b.offset
	if res.version == 0:
		w, h, depth = struct.unpack_from("<HHH", d, p)
		fmt = _LEGACY_FORMATS.get(struct.unpack_from("<b", d, p + 6)[0], 0)
		mips = d[p + 7]
		flags = struct.unpack_from("<H", d, p + 14)[0] >> 4
		return VTex(res, w, h, depth, fmt, mips, flags, codec=_codec(res))
	version, flags = struct.unpack_from("<HH", d, p)
	if version != 1:
		raise ValueError(f"unknown vtex version {version}")
	w, h, depth = struct.unpack_from("<HHH", d, p + 20)
	fmt, mips = d[p + 26], d[p + 27]
	extra_off, extra_count = struct.unpack_from("<II", d, p + 32)
	tex = VTex(res, w, h, max(1, depth), fmt, max(1, mips), flags, codec=_codec(res))
	q = p + 32 + extra_off
	for _ in range(extra_count):
		etype, eoff, esize = struct.unpack_from("<III", d, q)
		at = q + 4 + eoff
		if etype == 3 and esize >= 8:	# METADATA: display rect
			dw, dh = struct.unpack_from("<HH", d, at + 2)
			tex.display = (dw, dh)
		elif etype == 4:	# COMPRESSED_MIP_SIZE
			method, moff, mcount = struct.unpack_from("<III", d, at)
			tex.mip_compression = method
			tex.compressed_mips = list(struct.unpack_from(f"<{mcount}i", d, at + 4 + moff))
		q += 12
	return tex


def load(path: Path) -> VTex:
	return parse(vres.load(path))


def load_vtex(path: Path, face: int = 0) -> "np.ndarray":
	return load(path).rgba(face)


# -- pixel decoders ------------------------------------------------------------


def _bcn(raw: bytes, w: int, h: int, n, mode: str) -> "np.ndarray":
	from PIL import Image

	bw, bh = max(4, (w + 3) // 4 * 4), max(4, (h + 3) // 4 * 4)
	img = Image.frombytes(mode, (bw, bh), raw, "bcn", n)
	return np.asarray(img)[:h, :w]


def _half_to_ldr(v: "np.ndarray") -> "np.ndarray":
	"""Linear HDR → 8-bit sRGB with a simple exposure (the brightest 1% maps to white)."""
	v = np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	v = np.maximum(v, 0)
	peak = float(np.percentile(v, 99)) if v.size else 1.0
	v = v / max(peak, 1e-6)
	v = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(np.minimum(v, 1.0), 1 / 2.4) - 0.055)
	return np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8)


def decode(fmt: int, raw: bytes, w: int, h: int) -> "np.ndarray":
	n = w * h
	if fmt == 1:
		return _bcn(raw, w, h, 1, "RGBA")
	if fmt == 2:
		return _bcn(raw, w, h, 3, "RGBA")
	if fmt == 20:
		return _bcn(raw, w, h, 7, "RGBA")
	if fmt == 19:	# BC6H: Pillow decodes it clamped to [0, 1] linear; show it in sRGB like the other formats
		lin = _bcn(raw, w, h, (6, "BC6H"), "RGB").astype(np.float32) / 255.0
		srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
		return _expand(np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8))
	if fmt in (23, 24, 25, 26):
		from . import etc

		return etc.decode(raw, w, h, {23: "etc2", 24: "etc2_eac", 25: "r11", 26: "rg11"}[fmt])
	if fmt == 27:
		g = _bcn(raw, w, h, 4, "L")
		return np.dstack([g, g, g, np.full_like(g, 255)])
	if fmt == 21:
		rg = _bcn(raw, w, h, 5, "RGB")
		return np.dstack([rg[..., 0], rg[..., 1], np.zeros_like(rg[..., 0]), np.full_like(rg[..., 0], 255)])
	if fmt == 4:
		return np.frombuffer(raw, np.uint8, n * 4).reshape(h, w, 4).copy()
	if fmt == 28:
		return np.frombuffer(raw, np.uint8, n * 4).reshape(h, w, 4)[..., [2, 1, 0, 3]].copy()
	if fmt in (3, 33):
		g = np.frombuffer(raw, np.uint8, n).reshape(h, w)
		return np.dstack([g, g, g, np.full_like(g, 255)])
	if fmt == 32:
		a = np.frombuffer(raw, np.uint8, n).reshape(h, w)
		return np.dstack([np.full_like(a, 255)] * 3 + [a])
	if fmt == 22:
		ia = np.frombuffer(raw, np.uint8, n * 2).reshape(h, w, 2)
		return np.dstack([ia[..., 0]] * 3 + [ia[..., 1]])
	if fmt in (5, 6, 7):
		ch = {5: 1, 6: 2, 7: 4}[fmt]
		v = np.frombuffer(raw, "<u2", n * ch).reshape(h, w, ch)
		v8 = (v >> 8).astype(np.uint8)
		return _expand(v8)
	if fmt in (8, 9, 10, 11, 12, 13, 14):
		dt, ch = {8: ("<f2", 1), 9: ("<f2", 2), 10: ("<f2", 4), 11: ("<f4", 1), 12: ("<f4", 2),
			13: ("<f4", 3), 14: ("<f4", 4)}[fmt]
		v = np.frombuffer(raw, dt, n * ch).reshape(h, w, ch).astype(np.float32)
		rgb = _half_to_ldr(v[..., :min(ch, 3)])
		out = _expand(rgb)
		if ch == 4:
			out[..., 3] = np.clip(v[..., 3] * 255 + 0.5, 0, 255).astype(np.uint8)
		return out
	raise ValueError(f"vtex format {FORMATS.get(fmt, fmt)} is not supported")


def _expand(v: "np.ndarray") -> "np.ndarray":
	h, w = v.shape[:2]
	ch = v.shape[2] if v.ndim == 3 else 1
	v = v.reshape(h, w, ch)
	out = np.zeros((h, w, 4), np.uint8)
	out[..., 3] = 255
	if ch == 1:
		out[..., :3] = v
	else:
		out[..., :min(ch, 4)] = v[..., :4]
	return out


def _convert(img: "np.ndarray", codec: set[str], fmt: int, cube: bool) -> "np.ndarray":
	if "YCoCg" in codec:
		c = img.astype(np.float32) / 255.0
		if cube:
			c[..., :3] = np.where(c[..., :3] <= 0.04045, c[..., :3] / 12.92, ((c[..., :3] + 0.055) / 1.055) ** 2.4)
		scale = c[..., 2] * (255.0 / 8.0) + 1.0
		co = (c[..., 0] - 128 / 255) / scale
		cg = (c[..., 1] - 128 / 255) / scale
		y = c[..., 3]
		rgb = np.stack([y + co - cg, y + cg, y - co - cg], -1)
		if cube:
			rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(np.clip(rgb, 0, 1), 1 / 2.4) - 0.055)
		out = np.empty_like(img)
		out[..., :3] = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
		out[..., 3] = 255
		return out
	return img


# -- materials -----------------------------------------------------------------

# Colour inputs in the order a GameCube skin prefers them.
COLOR_PARAMS = ("g_tColor", "g_tColor1", "g_tColorA", "g_tColorB", "g_tColor2", "g_tBaseColor", "g_tAlbedo",
	"g_tLayer1Color", "TextureColor", "g_tMasks", "g_tSelfIllumMask")


@dataclass
class VMat:
	name: str
	shader: str
	textures: dict[str, str]
	ints: dict[str, int]
	floats: dict[str, float]

	@property
	def translucent(self) -> bool:
		return bool(self.ints.get("F_TRANSLUCENT") or self.ints.get("F_BLEND_MODE"))

	@property
	def alpha_test(self) -> bool:
		return bool(self.ints.get("F_ALPHA_TEST"))

	@property
	def tool(self) -> bool:
		n = (self.name + " " + self.shader).lower()
		return "tools" in n and ("toolsnodraw" in n or "tools_" in n or "/tools/" in n) or "nodraw" in n

	def color_texture(self) -> str | None:
		for k in COLOR_PARAMS:
			if self.textures.get(k):
				return self.textures[k]
		for k, v in self.textures.items():
			if "color" in k.lower() and v:
				return v
		return None


def parse_vmat(res: vres.Resource) -> VMat:
	d = res.keyvalues() or {}

	def table(key: str, value: str) -> dict:
		out = {}
		for item in d.get(key) or []:
			if isinstance(item, dict):
				out[str(item.get("m_name", ""))] = item.get(value)
		return out

	return VMat(
		str(d.get("m_materialName", "")), str(d.get("m_shaderName", "")),
		{k: str(v) for k, v in table("m_textureParams", "m_pValue").items() if v},
		{k: int(v) for k, v in table("m_intParams", "m_nValue").items() if v is not None},
		{k: float(v) for k, v in table("m_floatParams", "m_flValue").items() if v is not None},
	)


def load_vmat(path: Path) -> VMat:
	return parse_vmat(vres.load(path))


# -- finding compiled files ------------------------------------------------------


def content_roots(start: Path, depth: int = 8) -> list[Path]:
	"""Folders above `start` that could be a Source 2 game/addon root (hold materials/ or models/)."""
	out = []
	p = Path(start).resolve()
	for _ in range(depth):
		if any((p / sub).is_dir() for sub in ("materials", "models", "maps", "sounds")):
			out.append(p)
		if p.parent == p:
			break
		p = p.parent
	return out


def find_compiled(ref: str, roots: list[Path]) -> Path | None:
	"""A resource reference ('materials/x/y.vmat') → the compiled file under one of `roots`."""
	if not ref:
		return None
	rel = ref.replace("\\", "/").lstrip("/")
	if not rel.endswith("_c"):
		rel += "_c"
	for root in roots:
		cand = root / rel
		if cand.is_file():
			return cand
		# case-insensitive fallback (dumps from Windows VPKs on case-sensitive disks)
		cur = root
		ok = True
		for part in rel.split("/"):
			try:
				hit = next((c for c in cur.iterdir() if c.name.lower() == part.lower()), None)
			except OSError:
				hit = None
			if hit is None:
				ok = False
				break
			cur = hit
		if ok and cur.is_file():
			return cur
	return None
