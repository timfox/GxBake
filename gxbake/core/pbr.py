"""PBR material sets baked down to what a GX TEV pipeline can draw.

Flipper has no pixel shaders: per-pixel lighting, roughness and metalness do
not exist on a GameCube. What it does have is up to 8 textures, 16 TEV
stages of fixed-function combine, per-vertex diffuse and specular lighting
channels, emboss bump (GX_TG_BUMP*) and indirect texturing. So a PBR set is
baked into a small number of plain textures plus a recipe for the TEV:

	preset    textures                         TEV stages   what it looks like
	prelit    base                             1            lighting, AO, gloss highlights and glow painted in
	lit       base [+ spec] [+ emit]           1-3          AO and normal detail in the albedo; engine lights it
	emboss    lit + height (I4/I8)             3-5          + GX emboss bump towards one light
	indirect  lit + normal (IA8) [+ env]       3-5          + normal-perturbed sphere-map reflections

`load_material` reads loose maps (Substance, Quixel, Poly Haven, Unity and
Unreal naming, packed ORM/ARM/MRA/RMA/MaskMap channels) or a glTF/GLB
material; `bake` turns it into autotex jobs (formats picked automatically,
or under a budget) and a `.gxmat` JSON that describes the TEV setup.

The maths, in short: albedo is converted to linear, multiplied by AO (from
the map, or estimated from the height map when there is none), by a
normal-map "detail" term that depends only on how far each normal tilts
(lit) or also on a key light (prelit), and for prelit gets a normalised
Blinn-Phong highlight from roughness and an F0 from metalness. The spec
mask is gloss² × (¼ for dielectrics … 1 for metals): what to multiply the
GX specular channel or the sphere map by.
"""

from __future__ import annotations

import base64
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import autotex
from .hdr import linear_to_srgb, srgb_to_linear

PRESETS = ("prelit", "lit", "emboss", "indirect")
CHANNELS = ("albedo", "normal", "roughness", "metallic", "ao", "height", "emissive", "opacity", "specular", "gloss")

# filename token(s) → channel; checked longest first on the end of the stem
_TOKENS: dict[str, str] = {}
for _ch, _names in {
	"albedo": "albedo basecolor basecolour base_color base_colour diffuse diff color colour col alb base d",
	"normal": "normal normalgl normal_gl normalopengl normal_opengl nor_gl norgl nrm nor norm nml nm n",
	"normal_dx": "normaldx normal_dx normaldirectx normal_directx nor_dx nordx",
	"roughness": "roughness rough rgh r",
	"metallic": "metallic metalness metal met m",
	"ao": "ao ambientocclusion ambient_occlusion occlusion occ",
	"height": "height displacement disp bump heightmap h",
	"emissive": "emissive emission emit glow illum selfillum e",
	"opacity": "opacity alpha transparency cutout mask_alpha",
	"specular": "specular spec s",
	"gloss": "gloss glossiness smoothness",
	"orm": "orm arm occlusionroughnessmetallic occlusion_roughness_metallic",
	"rma": "rma roughnessmetallicao",
	"mra": "mra metallicroughnessao",
	"metallicroughness": "metallicroughness metallic_roughness metalroughness mr",
	"maskmap": "maskmap mask_map",
	"metallicsmoothness": "metallicsmoothness metallic_smoothness",
}.items():
	for _n in _names.split():
		_TOKENS[_n] = _ch

# packed map → {channel: (component index, invert)}
PACKED = {
	"orm": {"ao": (0, False), "roughness": (1, False), "metallic": (2, False)},
	"rma": {"roughness": (0, False), "metallic": (1, False), "ao": (2, False)},
	"mra": {"metallic": (0, False), "roughness": (1, False), "ao": (2, False)},
	"metallicroughness": {"roughness": (1, False), "metallic": (2, False)},	# glTF
	"maskmap": {"metallic": (0, False), "ao": (1, False), "roughness": (3, True)},	# Unity HDRP
	"metallicsmoothness": {"metallic": (0, False), "roughness": (3, True)},	# Unity standard
}


@dataclass
class Material:
	name: str
	maps: dict[str, "np.ndarray"] = field(default_factory=dict)	# float32 [0,1]; albedo/emissive/specular H×W×3, others H×W
	alpha: "np.ndarray | None" = None	# H×W float32 coverage
	sources: dict[str, str] = field(default_factory=dict)
	normal_y_down: bool = False	# DirectX normal maps (green = down)
	normal_scale: float = 1.0
	ao_strength: float = 1.0
	alpha_mode: str = "auto"	# auto | opaque | mask | blend
	alpha_cutoff: float = 0.5
	factors: dict = field(default_factory=dict)	# baseColor/metallic/roughness/emissive factors (glTF)
	notes: list[str] = field(default_factory=list)

	@property
	def size(self) -> tuple[int, int]:
		for m in self.maps.values():
			return m.shape[1], m.shape[0]
		return (0, 0)


# -- finding material sets ----------------------------------------------------------------------------


def _split(stem: str) -> list[str]:
	s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stem)
	return [t for t in re.split(r"[_\-. ]+", s.lower()) if t]


def classify(path: Path) -> tuple[str, str] | None:
	"""(material base name, channel) from a file name, or None if no channel token ends it."""
	toks = _split(Path(path).stem)
	# Poly Haven puts the resolution last ("rock_diff_2k"); Quixel puts it in the middle
	while toks and re.fullmatch(r"\d+k|\d+px|\d{3,4}|lod\d", toks[-1]):
		toks.pop()
	for n in (3, 2, 1):
		if len(toks) > n or (len(toks) == n and n > 1):
			key = "".join(toks[-n:])
			key_us = "_".join(toks[-n:])
			ch = _TOKENS.get(key) or _TOKENS.get(key_us)
			if ch and len(toks) > n:
				return "_".join(toks[:-n]), ch
	return None


def find_sets(paths: list[Path], recursive: bool = True) -> dict[str, dict[str, Path]]:
	"""Group image files into material sets by base name: {name: {channel: path}}.
	A set needs an albedo (or packed ORM with something to colour) to count."""
	from .imageformats import EXTENSIONS

	files: list[Path] = []
	for p in map(Path, paths):
		if p.is_dir():
			it = p.rglob("*") if recursive else p.iterdir()
			files += sorted(f for f in it if f.is_file() and f.suffix.lower() in EXTENSIONS)
		elif p.suffix.lower() in EXTENSIONS:
			files.append(p)
	sets: dict[tuple[Path, str], dict[str, Path]] = {}
	for f in files:
		c = classify(f)
		if c is None:
			continue
		sets.setdefault((f.parent, c[0]), {}).setdefault(c[1], f)
	out: dict[str, dict[str, Path]] = {}
	for (parent, name), chans in sorted(sets.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
		if "albedo" not in chans and not ({"specular", "emissive"} & set(chans)):
			continue
		# a lone "_d"/"_col" picture is just a texture, not a material
		if len(chans) < 2:
			continue
		key = name
		n = 2
		while key in out:
			key = f"{name}_{n}"
			n += 1
		out[key] = chans
	return out


# -- loading ------------------------------------------------------------------------------------------


def _load_float(path_or_bytes, max_edge: int) -> "np.ndarray":
	"""H×W×4 float32 [0,1] (HDR sources are clamped, colour stays as stored)."""
	from PIL import Image

	if isinstance(path_or_bytes, (bytes, bytearray)):
		import io

		from .imageformats import pillow_rgba

		rgba = pillow_rgba(Image.open(io.BytesIO(path_or_bytes)))
		f = rgba.astype(np.float32) / 255
	else:
		p = Path(path_or_bytes)
		from .imageformats import read_float

		hdr = read_float(p) if p.suffix.lower() in (".exr", ".hdr", ".pic", ".pfm") else None
		if hdr is not None:	# float maps hold linear values: data maps as they are, colour maps marked
			from .hdr import read

			f = np.clip(read(p), 0, 1)
			f = np.dstack([f, np.ones(f.shape[:2], np.float32)])[..., :5]
			f[..., 4] = 1	# a fifth plane flags "linear" for _assign
		else:
			from .images import load_rgba

			f = load_rgba(p).astype(np.float32) / 255
	h, w = f.shape[:2]
	if max(w, h) > max_edge:
		k = max_edge / max(w, h)
		nw, nh = max(1, round(w * k)), max(1, round(h * k))
		f = np.dstack([np.asarray(Image.fromarray(np.ascontiguousarray(f[..., c])).resize((nw, nh), Image.Resampling.BOX))
			for c in range(f.shape[2])])
	return f.astype(np.float32)


def _fit(a: "np.ndarray", w: int, h: int) -> "np.ndarray":
	from PIL import Image

	if a.shape[1] == w and a.shape[0] == h:
		return a
	if a.ndim == 2:
		return np.asarray(Image.fromarray(a).resize((w, h), Image.Resampling.BILINEAR))
	return np.dstack([np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c])).resize((w, h), Image.Resampling.BILINEAR))
		for c in range(a.shape[2])])


def _lum(rgb: "np.ndarray") -> "np.ndarray":
	return rgb[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _assign(mat: Material, ch: str, f: "np.ndarray", src: str) -> None:
	linear = f.shape[-1] > 4
	f = f[..., :4]
	if linear and ch in ("albedo", "emissive", "specular"):
		f = f.copy()
		f[..., :3] = linear_to_srgb(f[..., :3])
	if ch in PACKED:
		for sub, (i, inv) in PACKED[ch].items():
			if sub not in mat.maps:
				v = f[..., i]
				mat.maps[sub] = 1 - v if inv else v
				mat.sources[sub] = f"{src} [{'RGBA'[i]}{' inverted' if inv else ''}]"
		return
	if ch == "normal_dx":
		mat.normal_y_down = True
		ch = "normal"
	if ch in ("albedo", "emissive", "specular", "normal"):
		mat.maps[ch] = f[..., :3]
		if ch == "albedo" and float(f[..., 3].min()) < 0.999:
			mat.alpha = f[..., 3]
	elif ch == "opacity":
		mat.alpha = _lum(f) if float(f[..., 3].min()) > 0.999 else f[..., 3]
	else:
		mat.maps[ch] = _lum(f) if ch != "gloss" or float(f[..., 3].min()) > 0.999 else f[..., 3]
	mat.sources[ch] = src


def _unify(mat: Material) -> Material:
	sizes = [m.shape[:2] for m in mat.maps.values()]
	if mat.alpha is not None:
		sizes.append(mat.alpha.shape[:2])
	if not sizes:
		raise ValueError(f"material {mat.name!r} has no maps")
	h, w = max(sizes, key=lambda s: s[0] * s[1])
	if len(set(sizes)) > 1:
		mat.notes.append(f"maps differ in size; all scaled to {w}x{h}")
	mat.maps = {k: _fit(v, w, h) for k, v in mat.maps.items()}
	if mat.alpha is not None:
		mat.alpha = _fit(mat.alpha, w, h)
	if "gloss" in mat.maps and "roughness" not in mat.maps:
		mat.maps["roughness"] = 1 - mat.maps.pop("gloss")
	return mat


def load_set(name: str, chans: dict[str, Path], max_edge: int = 1024) -> Material:
	mat = Material(name)
	for ch, p in sorted(chans.items(), key=lambda kv: kv[0] not in PACKED):	# packed first, loose maps override
		f = _load_float(p, max_edge)
		if ch in PACKED:
			_assign(mat, ch, f, p.name)
	for ch, p in chans.items():
		if ch not in PACKED:
			sub = ch if ch != "normal_dx" else "normal"
			mat.maps.pop(sub, None)
			_assign(mat, ch, _load_float(p, max_edge), p.name)
	return _unify(mat)


# -- glTF -----------------------------------------------------------------------------------------


def _gltf_doc(path: Path) -> tuple[dict, bytes]:
	data = Path(path).read_bytes()
	if data[:4] == b"glTF":
		_magic, _ver, length = struct.unpack_from("<III", data, 0)
		pos, doc, bin_ = 12, None, b""
		while pos < length:
			clen, ctype = struct.unpack_from("<II", data, pos)
			chunk = data[pos + 8:pos + 8 + clen]
			if ctype == 0x4E4F534A:
				doc = json.loads(chunk)
			elif ctype == 0x004E4942:
				bin_ = chunk
			pos += 8 + clen
		if doc is None:
			raise ValueError("GLB without a JSON chunk")
		return doc, bin_
	return json.loads(data), b""


def _gltf_image(doc: dict, bin_: bytes, base: Path, tex_index: int, cache: dict):
	if tex_index in cache:
		return cache[tex_index]
	tex = doc["textures"][tex_index]
	src = tex.get("source")
	for ext in ("KHR_texture_basisu", "EXT_texture_webp", "MSFT_texture_dds"):
		src = tex.get("extensions", {}).get(ext, {}).get("source", src)
	img = doc["images"][src]
	if "uri" in img:
		uri = img["uri"]
		if uri.startswith("data:"):
			payload = base64.b64decode(uri.split(",", 1)[1])
			label = f"image {src} (embedded)"
		else:
			from urllib.parse import unquote

			payload = base / unquote(uri)
			label = Path(unquote(uri)).name
	else:
		bv = doc["bufferViews"][img["bufferView"]]
		if bv.get("buffer", 0) != 0 or not bin_:
			raise ValueError("glTF image in an external buffer is not supported")
		off = bv.get("byteOffset", 0)
		payload = bin_[off:off + bv["byteLength"]]
		label = f"image {src} ({img.get('name') or img.get('mimeType', 'embedded')})"
	cache[tex_index] = (payload, label)
	return cache[tex_index]


def load_gltf(path: Path, max_edge: int = 1024) -> list[Material]:
	"""Every material of a .gltf/.glb with its textures and factors."""
	path = Path(path)
	doc, bin_ = _gltf_doc(path)
	cache: dict = {}
	out = []
	for i, m in enumerate(doc.get("materials", [])):
		name = re.sub(r"[^a-z0-9_]+", "_", (m.get("name") or f"{path.stem}_{i}").lower()).strip("_") or f"material{i}"
		mat = Material(name)
		pbr = m.get("pbrMetallicRoughness", {})
		sg = m.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness")

		def tex(info, ch, mat=mat):
			if not info or "index" not in info:
				return
			payload, label = _gltf_image(doc, bin_, path.parent, info["index"], cache)
			_assign(mat, ch, _load_float(payload, max_edge), label)

		if sg:
			tex(sg.get("diffuseTexture"), "albedo")
			if sg.get("specularGlossinessTexture"):
				payload, label = _gltf_image(doc, bin_, path.parent, sg["specularGlossinessTexture"]["index"], cache)
				f = _load_float(payload, max_edge)
				mat.maps["specular"], mat.maps["roughness"] = f[..., :3], 1 - f[..., 3]
				mat.sources["specular"] = mat.sources["roughness"] = label
			mat.factors["baseColor"] = sg.get("diffuseFactor", [1, 1, 1, 1])
			mat.factors["roughness"] = 1 - sg.get("glossinessFactor", 1.0)
		else:
			tex(pbr.get("baseColorTexture"), "albedo")
			tex(pbr.get("metallicRoughnessTexture"), "metallicroughness")
			mat.factors["baseColor"] = pbr.get("baseColorFactor", [1, 1, 1, 1])
			mat.factors["metallic"] = pbr.get("metallicFactor", 1.0)
			mat.factors["roughness"] = pbr.get("roughnessFactor", 1.0)
		tex(m.get("normalTexture"), "normal")
		occ = m.get("occlusionTexture")
		if occ and "index" in occ:
			payload, label = _gltf_image(doc, bin_, path.parent, occ["index"], cache)
			mat.maps["ao"] = _load_float(payload, max_edge)[..., 0]	# glTF: occlusion is the R channel
			mat.sources["ao"] = label + " [R]"
			mat.ao_strength = occ.get("strength", 1.0)
		tex(m.get("emissiveTexture"), "emissive")
		mat.factors["emissive"] = m.get("emissiveFactor", [0, 0, 0])
		if m.get("normalTexture"):
			mat.normal_scale = m["normalTexture"].get("scale", 1.0)
		mat.alpha_mode = {"OPAQUE": "opaque", "MASK": "mask", "BLEND": "blend"}.get(m.get("alphaMode", "OPAQUE"), "opaque")
		mat.alpha_cutoff = m.get("alphaCutoff", 0.5)
		if not mat.maps:	# factors only: a flat 4×4 colour
			mat.maps["albedo"] = np.ones((4, 4, 3), np.float32)
			mat.notes.append("no textures; flat colour from the factors")
		out.append(_apply_factors(_unify(mat)))
	return out


def _apply_factors(mat: Material) -> Material:
	f = mat.factors
	h, w = mat.size[1], mat.size[0]
	bc = f.get("baseColor")
	if bc is not None:
		if "albedo" not in mat.maps:
			mat.maps["albedo"] = np.ones((h, w, 3), np.float32)
		# factors are linear; the texture is sRGB
		lin = srgb_to_linear(mat.maps["albedo"]) * np.array(bc[:3], np.float32)
		mat.maps["albedo"] = linear_to_srgb(lin)
		if len(bc) > 3 and bc[3] < 1:
			mat.alpha = (mat.alpha if mat.alpha is not None else np.ones((h, w), np.float32)) * bc[3]
	for ch in ("metallic", "roughness"):
		if ch in f:
			mat.maps[ch] = (mat.maps[ch] if ch in mat.maps else np.ones((h, w), np.float32)) * float(f[ch])
	e = f.get("emissive")
	if e is not None and any(e):
		base = mat.maps.get("emissive", np.ones((h, w, 3), np.float32))
		mat.maps["emissive"] = linear_to_srgb(srgb_to_linear(base) * np.array(e[:3], np.float32))
	elif "emissive" in mat.maps and e is not None and not any(e):
		mat.maps.pop("emissive")	# glTF: a texture with a zero factor emits nothing
	return mat


def load_material(path_or_set, max_edge: int = 1024) -> list[Material]:
	"""A .gltf/.glb (every material), or a folder/list of loose maps (every set found)."""
	if isinstance(path_or_set, (str, Path)) and Path(path_or_set).suffix.lower() in (".gltf", ".glb"):
		return load_gltf(Path(path_or_set), max_edge)
	paths = path_or_set if isinstance(path_or_set, (list, tuple)) else [path_or_set]
	return [load_set(n, c, max_edge) for n, c in find_sets(list(paths)).items()]


# -- image maths ----------------------------------------------------------------------------------


def _freqs(h: int, w: int) -> tuple["np.ndarray", "np.ndarray"]:
	wy = 2 * np.pi * np.fft.fftfreq(h)[:, None]
	wx = 2 * np.pi * np.fft.fftfreq(w)[None, :]
	return wx, wy


def blur(a: "np.ndarray", sigma: float) -> "np.ndarray":
	"""Gaussian blur that wraps (textures tile)."""
	if sigma <= 0:
		return a
	h, w = a.shape
	wx, wy = _freqs(h, w)
	g = np.exp(-(wx ** 2 + wy ** 2) * sigma * sigma / 2)
	return np.real(np.fft.ifft2(np.fft.fft2(a) * g)).astype(np.float32)


def decode_normals(rgb: "np.ndarray", y_down: bool = False, scale: float = 1.0) -> "np.ndarray":
	n = rgb.astype(np.float32) * 2 - 1
	if y_down:
		n[..., 1] = -n[..., 1]
	n[..., :2] *= scale
	n[..., 2] = np.maximum(n[..., 2], 1e-3)
	return n / np.linalg.norm(n, axis=-1, keepdims=True)


def height_from_normals(n: "np.ndarray") -> "np.ndarray":
	"""Integrate a tangent-space normal map to a height field (Frankot–Chellappa, periodic),
	normalised to [0, 1]. +Y is up in the image, so dh/drow = +ny/nz."""
	nz = np.maximum(n[..., 2], 0.05)
	p = -n[..., 0] / nz	# dh/dx
	q = n[..., 1] / nz	# dh/drow
	h, w = p.shape
	wx, wy = _freqs(h, w)
	den = wx ** 2 + wy ** 2
	den[0, 0] = 1
	z = (-1j * wx * np.fft.fft2(p) - 1j * wy * np.fft.fft2(q)) / den
	z[0, 0] = 0
	hgt = np.real(np.fft.ifft2(z))
	lo, hi = np.percentile(hgt, (0.5, 99.5))
	return np.clip((hgt - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def ao_from_height(hgt: "np.ndarray", strength: float = 1.0) -> "np.ndarray":
	"""Cheap cavity AO: how far each texel sits below its blurred surroundings at three scales."""
	h, w = hgt.shape
	base = max(1.0, min(h, w) / 256)
	occ = np.zeros_like(hgt)
	for s, wt in ((2, 0.5), (6, 0.3), (16, 0.2)):
		occ += wt * np.maximum(0, blur(hgt, s * base) - hgt)
	occ /= max(float(np.percentile(occ, 99)), 1e-4)
	return np.clip(1 - 0.6 * strength * occ, 0, 1).astype(np.float32)


# -- baking ---------------------------------------------------------------------------------------


@dataclass
class BakeOptions:
	preset: str = "lit"
	light: tuple[float, float, float] = (-0.45, 0.55, 0.70)	# tangent-space key light for prelit/emboss (upper left)
	ambient: float = 0.55	# share of hemisphere (ambient) light in the prelit/lit detail term
	detail: float = 1.0	# how strongly normal-map detail is painted into the base
	ao_strength: float = 1.0
	ao_from_height: bool = True	# estimate AO from height (or integrated normals) when there is no AO map
	spec_strength: float = 0.6	# prelit highlight strength
	emissive_in_base: bool | None = None	# None: only for prelit
	max_edge: int = 256	# GX texture size cap (per map)
	min_db: float = 36.0
	mipmaps: bool = True
	env: "np.ndarray | None" = None	# a sphere map (RGBA uint8) for the indirect preset
	env_name: str = ""
	spec_threshold: float = 0.06	# skip the spec map when the mask's 99th percentile is below this
	formats: dict = field(default_factory=dict)	# map name → allowed formats override


@dataclass
class Baked:
	material: Material
	preset: str
	images: dict[str, "np.ndarray"]	# map name → RGBA uint8, in TPL order
	jobs: list[autotex.TexJob]
	tev: dict
	notes: list[str]


def _u8(x: "np.ndarray") -> "np.ndarray":
	return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def _rgba(rgb: "np.ndarray", a: "np.ndarray | None" = None) -> "np.ndarray":
	if rgb.ndim == 2:
		rgb = np.repeat(rgb[..., None], 3, -1)
	a8 = np.full(rgb.shape[:2], 255, np.uint8) if a is None else _u8(a)
	return np.dstack([_u8(rgb), a8])


def bake(mat: Material, opts: BakeOptions | None = None) -> Baked:
	opts = opts or BakeOptions()
	if opts.preset not in PRESETS:
		raise ValueError(f"unknown preset {opts.preset!r} (one of {', '.join(PRESETS)})")
	w, h = mat.size
	notes = list(mat.notes)
	m = mat.maps
	alb = m.get("albedo", np.full((h, w, 3), 0.8, np.float32))
	A = srgb_to_linear(alb)
	rough = np.clip(m.get("roughness", np.full((h, w), 0.6, np.float32)), 0.02, 1)
	metal = np.clip(m.get("metallic", np.zeros((h, w), np.float32)), 0, 1)
	if "specular" in m and "metallic" not in m:	# spec/gloss workflow: F0 given directly
		F0 = srgb_to_linear(m["specular"])
		metal = np.clip((_lum(F0) - 0.04) / 0.5, 0, 1)
	else:
		F0 = 0.04 * (1 - metal[..., None]) + A * metal[..., None]
	N = decode_normals(m["normal"], mat.normal_y_down, mat.normal_scale) if "normal" in m else None
	hgt = m.get("height")
	need_height = opts.preset == "emboss" or (opts.ao_from_height and "ao" not in m)
	if hgt is None and N is not None and need_height:
		hgt = height_from_normals(N)
		if opts.preset == "emboss":
			notes.append("height integrated from the normal map")
	if "ao" in m:
		ao = 1 - (1 - m["ao"]) * mat.ao_strength * opts.ao_strength
	elif opts.ao_from_height and hgt is not None:
		ao = ao_from_height(hgt, opts.ao_strength)
		notes.append("AO estimated from " + ("the height map" if "height" in m else "the normal map"))
	else:
		ao = np.ones((h, w), np.float32)
	ao = np.clip(ao, 0, 1)

	# detail: hemisphere term (normal tilt), plus key light for prelit; 1.0 on a flat normal
	L = np.array(opts.light, np.float32)
	L /= np.linalg.norm(L)
	if N is not None:
		amb = 0.5 + 0.5 * N[..., 2]
		if opts.preset == "prelit":
			key = np.maximum(0, N @ L)
			D = (opts.ambient * amb + (1 - opts.ambient) * key) / (opts.ambient + (1 - opts.ambient) * L[2])
		else:
			D = amb
		D = 1 + (D - 1) * opts.detail
	else:
		D = np.ones((h, w), np.float32)

	gloss = 1 - rough
	spec_mask = np.clip(gloss * gloss * (0.25 + 0.75 * metal), 0, 1)
	emi = m.get("emissive")
	emi_in_base = opts.emissive_in_base if opts.emissive_in_base is not None else opts.preset == "prelit"

	if opts.preset == "prelit":
		diffuse = A * (1 - metal[..., None]) * D[..., None]
		metal_env = F0 * metal[..., None] * (0.35 + 0.65 * D[..., None])	# metals: a dim, even environment
		base = (diffuse + metal_env) * ao[..., None]
		if N is not None:
			Hv = L + np.array([0, 0, 1], np.float32)
			Hv /= np.linalg.norm(Hv)
			ndh = np.maximum(0, N @ Hv)
			ndl = np.maximum(0, N @ L)
		else:
			ndh = np.full((h, w), float(((L + [0, 0, 1]) / np.linalg.norm(L + [0, 0, 1]))[2]))
			ndl = np.full((h, w), float(L[2]))
		alpha = rough * rough
		n_exp = np.clip(2 / np.maximum(alpha * alpha, 1e-4) - 2, 1, 2048)
		spec = (n_exp + 8) / (8 * math.pi) * ndh ** n_exp * ndl
		base = base + F0 * (spec * ao * opts.spec_strength)[..., None] * (1 - opts.ambient)
	else:
		base = A * (ao * D)[..., None]
	if emi is not None and emi_in_base:
		base = base + srgb_to_linear(emi)
	base_srgb = linear_to_srgb(base)

	alpha_mode = mat.alpha_mode
	a = mat.alpha
	if alpha_mode == "auto":
		alpha_mode = "opaque" if a is None or float(a.min()) > 0.99 else \
			"mask" if float(((a < 0.04) | (a > 0.96)).mean()) > 0.98 else "blend"
	if alpha_mode == "opaque":
		a = None
	elif alpha_mode == "mask" and a is not None:
		a = (a >= mat.alpha_cutoff).astype(np.float32)

	images: dict[str, np.ndarray] = {"base": _rgba(base_srgb, a)}
	roles = {"base": "auto"}
	if opts.preset != "prelit":
		if float(np.percentile(spec_mask, 99)) >= opts.spec_threshold:
			images["spec"] = _rgba(spec_mask)
			roles["spec"] = "mask"
		else:
			notes.append("no spec map: the material is matte")
		if emi is not None and not emi_in_base and float(emi.max()) > 0.02:
			images["emit"] = _rgba(emi)
			roles["emit"] = "auto"
	if opts.preset == "emboss":
		if hgt is None:
			notes.append("no height or normal map: emboss preset has nothing to bump; height map skipped")
		else:
			images["height"] = _rgba(hgt)
			roles["height"] = "mask"
	if opts.preset == "indirect":
		if N is None:
			notes.append("no normal map: indirect preset has nothing to perturb; normal map skipped")
		else:
			nrm = np.zeros((h, w, 4), np.uint8)
			nrm[..., :3] = _u8(N[..., 0] * 0.5 + 0.5)[..., None]
			nrm[..., 3] = _u8(N[..., 1] * 0.5 + 0.5)
			images["normal"] = nrm
			roles["normal"] = "normal"
		if opts.env is not None:
			images["env"] = opts.env
			roles["env"] = "color"

	jobs = []
	for k, img in images.items():
		role = roles[k]
		fmts = tuple(opts.formats.get(k, autotex.ROLES[role]))
		jobs.append(autotex.TexJob(
			f"{mat.name}_{k}"[:31], img, fmts,
			max(opts.min_db, 40.0) if k == "normal" else opts.min_db - 6 if role == "mask" else opts.min_db,
			opts.max_edge if k != "env" else img.shape[1], mipmaps=opts.mipmaps and k != "env",
			wrap="clamp" if k == "env" else "repeat",
			fixed="IA8" if k == "normal" and fmts == autotex.ROLES["normal"] else "",
			meta={"material": mat.name, "map": k}))
	tev = tev_recipe(opts.preset, list(images), alpha_mode, mat.alpha_cutoff)
	return Baked(mat, opts.preset, images, jobs, tev, notes)


def tev_recipe(preset: str, maps: list[str], alpha_mode: str, cutoff: float) -> dict:
	"""How to draw the baked maps with GX (libogc names). TEXMAPn follows the TPL image order.
	The recipes are the standard fixed-function set-ups; check them against your renderer."""
	tm = {k: f"GX_TEXMAP{i}" for i, k in enumerate(maps)}
	st: list[str] = []
	chans = ["GX_COLOR0A0: diffuse lighting (GX_SetChanCtrl, GX_AF_SPOT/NONE, material colour white)"]
	if preset == "prelit":
		st.append(f"stage0: TEXC({tm['base']}) × RASC (GX_MODULATE) — or GX_REPLACE for an unlit, fully baked look")
	else:
		st.append(f"stage0: TEXC({tm['base']}) × RASC(GX_COLOR0A0) (GX_MODULATE)")
	if "spec" in maps:
		chans.append("GX_COLOR1A1: specular (GX_SetChanCtrl with GX_AF_SPEC, GX_DF_NONE; GX_InitSpecularDir per light)")
		st.append(f"stage{len(st)}: CPREV + TEXC({tm['spec']}) × RASC(GX_COLOR1A1) (a=ZERO b=RASC c=TEXC d=CPREV)")
	if "height" in maps:
		st.append(f"stage{len(st)}: emboss — texcoord1 = GX_TG_BUMP0 of texcoord0 (light 0); "
			f"TEXC({tm['height']} @ tc0) − TEXC({tm['height']} @ tc1), GX_TEV_SUB with bias/scale, then × CPREV")
	if "normal" in maps:
		target = tm.get("env", "your sphere or light map")
		st.append(f"indirect: GX_SetIndTexOrder(GX_INDTEXSTAGE0, texcoord0, {tm['normal']}); X in I, Y in A, "
			"biased 0.5 (GX_ITB_ST, GX_ITF_8); GX_SetIndTexMatrix scale ≈ sphere-map radius")
		st.append(f"stage{len(st)}: CPREV + TEXC({target} @ GX_TG_NRM sphere texgen, warped by the indirect stage)"
			+ (f" × TEXC({tm['spec']}) mask" if "spec" in maps else ""))
	elif "env" in maps:
		st.append(f"stage{len(st)}: CPREV + TEXC({tm['env']} @ GX_TG_NRM sphere texgen)"
			+ (f" × TEXC({tm['spec']})" if "spec" in maps else ""))
	if "emit" in maps:
		st.append(f"stage{len(st)}: CPREV + TEXC({tm['emit']}) (glow ignores lighting)")
	blend = {"opaque": "GX_BM_NONE", "mask": f"GX_SetAlphaCompare(GX_GEQUAL, {int(cutoff * 255)}) with z-update",
		"blend": "GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA; sort back to front"}[alpha_mode]
	return {"texmaps": tm, "channels": chans, "stages": st, "tev_stages": sum(1 for s in st if s.startswith("stage")),
		"alpha": alpha_mode, "blend": blend}
