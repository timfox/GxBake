"""Particle systems (Source 1 .pcf, Source 2 .vpcf_c) → Half-Life sprites the port can draw.

GoldSrc has no particle system, so what carries over is the look of each system:
the texture its sprite renderer draws, as a parallel-facing sprite. Source 1
systems are DmeParticleSystemDefinitions in a DMX file; their material is a
spritecard .vmt whose $basetexture may hold a sprite sheet (a VTF resource of
sequences of frame rectangles), and the first sequence becomes the sprite's
frames. A multi-frame VTF gives one frame per VTF frame instead. Source 2
systems name the texture in their renderer (m_hTexture or m_vecTexturesInput).

Additive materials become additive sprites. Others become alpha-test sprites
when their alpha is a cut-out, or index-alpha sprites (one colour, alpha from
the index) when it is soft, the closest GoldSrc has to smoke.
Emitters, operators and children's timing are not converted;
the inspector lists them.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dmx

PARTICLE_EXTENSIONS = {".pcf", ".vpcf_c"}
SPR_NORMAL, SPR_ADDITIVE, SPR_INDEXALPHA, SPR_ALPHATEST = 0, 1, 2, 3
SPR_VP_PARALLEL = 2
_SHEET = b"\x10\0\0"
MAX_FRAMES = 64


@dataclass
class ParticleSystem:
	name: str
	material: str	# Source 1: material path; Source 2: texture resource
	additive: bool = False
	renderers: list[str] = field(default_factory=list)
	children: list[str] = field(default_factory=list)
	max_particles: int = 0
	radius: float = 0.0
	colour: tuple[int, int, int, int] = (255, 255, 255, 255)

	@property
	def renders(self) -> bool:
		return bool(self.material and self.renderers)


def is_particles(path: Path) -> bool:
	p = Path(path)
	if p.name.lower().endswith(".vpcf_c"):
		return True
	if p.suffix.lower() != ".pcf":
		return False
	try:
		with p.open("rb") as f:
			hd = dmx.header(f.read(200))
	except OSError:
		return False
	return hd is not None


def systems(path: Path) -> list[ParticleSystem]:
	path = Path(path)
	if path.name.lower().endswith(".vpcf_c"):
		return [_vpcf(path)]
	dm = dmx.load(path)
	defs = (dm.root.get("particleSystemDefinitions") if dm.root is not None else None) or \
		dm.of_type("DmeParticleSystemDefinition")
	out = []
	for d in defs:
		if d is None:
			continue
		rend = [r for r in (d.get("renderers") or []) if r is not None]
		children = []
		for c in d.get("children") or []:
			ref = c.get("child") if c is not None else None
			if ref is not None:
				children.append(ref.name)
		col = d.get("color", (255, 255, 255, 255))
		out.append(ParticleSystem(d.name, str(d.get("material", "") or ""), False,
			[str(r.get("functionName", r.name)) for r in rend], children, int(d.get("max_particles", 0) or 0),
			float(d.get("radius", 0) or 0), tuple(int(x) for x in col) if isinstance(col, tuple) else (255,) * 4))
	return out


def _vpcf(path: Path) -> ParticleSystem:
	from . import vres

	d = vres.load(path).keyvalues() or {}
	tex = ""
	additive = False
	names = []
	for r in d.get("m_Renderers") or []:
		if not isinstance(r, dict):
			continue
		names.append(str(r.get("_class", "")))
		t = r.get("m_hTexture") or ""
		for inp in r.get("m_vecTexturesInput") or []:
			if isinstance(inp, dict) and inp.get("m_hTexture"):
				t = t or inp["m_hTexture"]
		tex = tex or str(t)
		blend = str(r.get("m_nOutputBlendMode", ""))
		additive = additive or bool(r.get("m_bAdditive")) or blend.endswith("ADD")
	children = [str(c.get("m_ChildRef", "")) for c in d.get("m_Children") or [] if isinstance(c, dict)]
	col = d.get("m_ConstantColor")
	col = [255, 255, 255, 255] if col is None else [int(x) for x in list(col)[:4]]
	return ParticleSystem(path.name[:-len(".vpcf_c")], tex, additive, names, children,
		int(d.get("m_nMaxParticles", 0) or 0), float(d.get("m_flConstantRadius", 0) or 0),
		tuple(int(x) for x in list(col)[:4]))


# -- textures --------------------------------------------------------------------------------------


def vtf_resource(data: bytes, tag: bytes) -> bytes | None:
	"""The data of one VTF 7.3+ resource (e.g. the sprite sheet), or None."""
	major, minor = struct.unpack_from("<II", data, 4)
	if (major, minor) < (7, 3):
		return None
	n = struct.unpack_from("<I", data, 68)[0]
	for i in range(min(n, 32)):
		t, flags, off = struct.unpack_from("<3sBI", data, 80 + i * 8)
		if t == tag and not flags & 2 and off + 4 <= len(data):
			size = struct.unpack_from("<I", data, off)[0]
			return data[off + 4:off + 4 + size]
	return None


def sheet_frames(sheet: bytes) -> list[tuple[float, float, float, float]]:
	"""The first sequence of a sprite sheet resource as (u0, v0, u1, v1) rectangles."""
	version, nseq = struct.unpack_from("<ii", sheet, 0)
	images = 4 if version >= 1 else 1
	p = 8
	best: tuple[int, list] | None = None
	for _ in range(nseq):
		seq, _clamp, nframes, _total = struct.unpack_from("<iiif", sheet, p)
		p += 16
		rects = []
		for _ in range(nframes):
			p += 4	# duration
			rects.append(struct.unpack_from("<4f", sheet, p))
			p += 16 * images
		if best is None or seq < best[0]:
			best = (seq, rects)
	return best[1] if best else []


def _source1_frames(material: str, roots: list[Path], warnings: list[str]) -> tuple[list[np.ndarray], bool]:
	from . import vtf

	name = material.replace("\\", "/").removeprefix("materials/").removesuffix(".vmt")
	tex, params = vtf.resolve_material(name, [""], roots)
	if tex is None:
		warnings.append(f"{material}: not found under {roots[0] if roots else 'any materials/ folder'}")
		return [], False
	additive = params.get("$additive", "0") not in ("0", "")
	data = tex.read_bytes()
	hd = vtf.vtf_header(data)
	sheet = vtf_resource(data, _SHEET)
	if sheet:
		img = vtf.read_vtf(data)
		h, w = img.shape[:2]
		frames = []
		for u0, v0, u1, v1 in sheet_frames(sheet)[:MAX_FRAMES]:
			x0, x1 = round(u0 * w), max(round(u0 * w) + 1, round(u1 * w))
			y0, y1 = round(v0 * h), max(round(v0 * h) + 1, round(v1 * h))
			frames.append(img[y0:y1, x0:x1])
		if frames:
			return frames, additive
	return [vtf.read_vtf(data, frame=i) for i in range(min(hd["frames"], MAX_FRAMES))], additive


def _source2_frames(texture: str, roots: list[Path], warnings: list[str]) -> list[np.ndarray]:
	from . import vtex

	p = vtex.find_compiled(texture, roots)
	if p is None:
		warnings.append(f"{texture}: not found")
		return []
	return [vtex.load(p).rgba()]


# -- sprite writing --------------------------------------------------------------------------------


def _fit(img: np.ndarray, max_edge: int) -> np.ndarray:
	from PIL import Image

	h, w = img.shape[:2]
	k = max(h, w) / max_edge
	if k <= 1:
		return img
	return np.asarray(Image.fromarray(img).resize((max(1, round(w / k)), max(1, round(h / k))),
		Image.Resampling.LANCZOS))


def sprite_bytes(frames: list[np.ndarray], additive: bool, max_edge: int = 128) -> tuple[bytes, int]:
	"""RGBA frames → a Half-Life v2 parallel sprite sharing one palette; returns (bytes, texture format)."""
	from .models import quantize_skin
	from .quake import _hl_header

	frames = [_fit(np.ascontiguousarray(f), max_edge) for f in frames]
	alpha = np.concatenate([f[..., 3].ravel() for f in frames])
	if additive:
		fmt = SPR_ADDITIVE
	elif (alpha < 255).any() and ((alpha > 16) & (alpha < 240)).mean() > 0.05:
		fmt = SPR_INDEXALPHA
	else:
		fmt = SPR_ALPHATEST
	w = max(f.shape[1] for f in frames)
	h = max(f.shape[0] for f in frames)
	if fmt == SPR_INDEXALPHA:
		weights = alpha.astype(np.float64)
		rgb = np.concatenate([f[..., :3].reshape(-1, 3) for f in frames]).astype(np.float64)
		colour = (rgb * weights[:, None]).sum(0) / max(weights.sum(), 1.0)
		palette = bytes(np.clip(np.round(colour), 0, 255).astype(np.uint8).tolist()) * 256
		indices = [f[..., 3].astype(np.uint8) for f in frames]
	else:
		# quantise every frame together (stacked), so they share the sprite's palette
		stack = np.zeros((sum(f.shape[0] for f in frames), w, 4), np.uint8)
		y = 0
		for f in frames:
			src = f.copy()
			if additive:	# premultiplied: black adds nothing
				src[..., :3] = (src[..., :3].astype(np.uint16) * src[..., 3:] // 255).astype(np.uint8)
				src[..., 3] = 255
			stack[y:y + f.shape[0], :f.shape[1]] = src
			y += f.shape[0]
		idx, palette = quantize_skin(stack, fmt == SPR_ALPHATEST)
		flat = np.frombuffer(idx, np.uint8).reshape(stack.shape[:2])
		indices, y = [], 0
		for f in frames:
			indices.append(flat[y:y + f.shape[0], :f.shape[1]])
			y += f.shape[0]
	out = bytearray(_hl_header(SPR_VP_PARALLEL, fmt, math.hypot(w / 2, h / 2), w, h, len(frames), 0.0, 0, palette))
	for ix in indices:
		fh, fw = ix.shape
		out += struct.pack("<iiiii", 0, -(fw // 2), fh // 2, fw, fh) + np.ascontiguousarray(ix).tobytes()
	return bytes(out), fmt


def _roots(path: Path, extra: list[Path] | None) -> tuple[list[Path], list[Path]]:
	from . import vtex
	from .vtf import find_materials_root

	s1 = [r for r in (find_materials_root(path.parent),) if r is not None]
	s1 += [Path(e) / "materials" for e in extra or [] if (Path(e) / "materials").is_dir()]
	s2 = vtex.content_roots(path.parent) + [Path(e) for e in extra or []]
	return s1, s2


def convert(path: Path, search: list[Path] | None = None, max_edge: int = 128,
		warnings: list[str] | None = None) -> dict[str, bytes]:
	"""system name → Half-Life sprite bytes, for every system that renders a texture."""
	path = Path(path)
	warnings = warnings if warnings is not None else []
	s1, s2 = _roots(path, search)
	out: dict[str, bytes] = {}
	cache: dict[str, bytes] = {}
	for sysdef in systems(path):
		if not sysdef.renders:
			continue
		key = sysdef.material.lower()
		if key not in cache:
			if path.name.lower().endswith(".vpcf_c"):
				frames, additive = _source2_frames(sysdef.material, s2, warnings), sysdef.additive
			else:
				frames, additive = _source1_frames(sysdef.material, s1, warnings)
			cache[key] = sprite_bytes(frames, additive, max_edge)[0] if frames else b""
		if cache[key]:
			out[sysdef.name] = cache[key]
	return out


def info(path: Path) -> dict:
	sys_list = systems(path)
	d: dict = {"Particle systems": len(sys_list), "With a sprite renderer": sum(1 for s in sys_list if s.renders)}
	mats = sorted({s.material for s in sys_list if s.renders})
	if mats:
		d["Materials"] = ", ".join(mats[:8]) + (f" (+{len(mats) - 8})" if len(mats) > 8 else "")
	d["Not converted"] = "emitters, operators, initializers and child timing (GoldSrc has no particle system)"
	return d
