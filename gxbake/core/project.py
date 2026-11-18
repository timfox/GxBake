"""Asset model shared by the GUI and the CLI: detect, inspect, preview, convert.

Every conversion calls the ported pipeline functions in this package; this
module only decides which one applies to a file and where its output goes.
"""

from __future__ import annotations

import json
import re
import struct
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from . import audit, budget, gct, goldsrc, images, media, models, panda, quake, rose, skybox, sprite, studio, studio_slim
from . import audit, budget, gct, goldsrc, images, media, models, pandaquake, rose, runescape, skybox, sprite, studio, studio_slim
from .cmpr import cmpr_decode

Log = Callable[[str], None]
Progress = Callable[[int, int, str], None]


SEQ_PREFIX = "▶ "	# tree children of a 3D model that are sequences, not materials
SKY_NET = "all sides"	# tree child of a panorama sky: the six sides as a cross


class Kind(str, Enum):
	IMAGE = "Image"
	SKYBOX = "Skybox"
	WAD = "WAD textures"
	BSP = "BSP map"
	MDL = "Studio model"
	MODEL = "3D model"
	SPR = "Sprite"
	VIDEO = "Video"
	AUDIO = "Audio"
	GCT = "GX texture pack"
	CMP = "GX CMPR (raw)"
	GCVID = "GCVID video"
	GCPCM = "GCPCM audio"
	UNKNOWN = "Unsupported"


CONVERTIBLE = {Kind.IMAGE, Kind.SKYBOX, Kind.WAD, Kind.BSP, Kind.MDL, Kind.MODEL, Kind.SPR, Kind.VIDEO, Kind.AUDIO}


def _head(path: Path, n: int = 8) -> bytes:
	try:
		with path.open("rb") as f:
			return f.read(n)
	except OSError:
		return b""


def detect_kind(path: Path) -> Kind:
	ext = path.suffix.lower()
	if panda.is_panda(path):	# .egg/.bam (also .egg.pz/.bam.pz) and .pandactor
		return Kind.MODEL
	if runescape.is_cache(path):
		return Kind.MODEL	# a RuneScape cache: one asset per NPC or model
	if ext == ".mdl":
		# GoldSrc studio (IDST v10), Source studio (IDST v44-49) and Quake alias (IDPO) share .mdl.
		head = _head(path)
		if models.native_format(path):
			return Kind.MODEL
		if head[:4] == b"IDSQ":
			return Kind.UNKNOWN	# a GoldSrc sequence group (<name>01.mdl), read with its model
		return Kind.MDL
	if ext == ".sp2":
		return Kind.SPR
	if ext in images.IMAGE_EXTENSIONS:
		return Kind.IMAGE
	if ext in media.VIDEO_EXTENSIONS:
		return Kind.VIDEO
	if ext in media.AUDIO_EXTENSIONS:
		return Kind.AUDIO
	if ext in models.MODEL_EXTENSIONS or ext in rose.ROSE_EXTENSIONS:
		return Kind.MODEL
	return {
		".wad": Kind.WAD,
		".bsp": Kind.BSP,
		".mdl": Kind.MDL,
		".spr": Kind.SPR,
		".gct": Kind.GCT,
		".cmp": Kind.CMP,
		".gcvid": Kind.GCVID,
		".gcpcm": Kind.GCPCM,
	}.get(ext, Kind.UNKNOWN)


@dataclass
class Settings:
	image: images.ImageOptions = field(default_factory=images.ImageOptions)
	pack_images: bool = True	# loose images → one media/<pack_name>.gct
	pack_name: str = "textures"
	raw_cmp: bool = False	# also write level-0 .cmp per loose image
	sky_size: int = images.GC_SKY_CMPR_SIZE
	sky_yaw: float = skybox.DEFAULT_YAW	# panoramas: engine yaw the image centre faces (270 = ft)
	sky_fov: float = 180.0	# flat photos used as skies: horizontal degrees they cover
	sky_exposure: float = 0.0	# .hdr panoramas: stops over auto exposure
	detect_panoramas: bool = True	# big 2:1 images, .hdr, crosses and strips become skies
	mdl_lean_mesh: bool = True
	mdl_skin_companion: bool = True
	mdl_max_edge: int = 64
	mdl_anim_step: int = 1	# keep every Nth animation frame in gc_studio/<name>.mdl
	model_scale: float = 1.0	# Assimp import: source units → GoldSrc units
	model_up_axis: str = "y"
	model_skin_max_edge: int = 256
	model_write_full: bool = True	# also write the full studio MDL to models/<name>.mdl
	model_anim_fps: int = 15
	model_static: bool = False	# ignore skeletons/animations
	model_clip_files: bool = True	# merge <name>@<clip>.<ext> animations
	model_blend_bones: bool = True	# Source models: helper bones for two-bone weights
	rose_scale: float = 39.37	# ROSE Online: GoldSrc units per metre
	rose_yaw: float = 90.0	# ROSE Online: turn about Z (ROSE models face -Y, GoldSrc +X)
	rose_entries: list[int] = field(default_factory=list)	# only these ZSC objects / CHR characters ([] = all)
	panda_scale: float = 12.0	# Panda3D/Toontown: GoldSrc units per Panda unit (a foot in Toontown)
	panda_yaw: float = -90.0	# Panda3D: turn about Z (Panda models face +Y, GoldSrc +X)
	rs_scale: float = runescape.DEFAULT_SCALE	# RuneScape: GoldSrc units per model unit (128 = one tile)
	rs_yaw: float = 0.0	# RuneScape: extra turn about Z (models already face GoldSrc's +X)
	rs_entries: list[str] = field(default_factory=list)	# RuneScape caches: npc:<id> / model:<id> ([] = every named NPC)
	spr_scale: int = 2
	video_width: int = media.GCVID_WIDTH
	video_height: int = media.GCVID_HEIGHT
	video_fps: int = media.GCVID_FPS_NUM
	video_still_frame: int = media.GCVID_STILL_FRAME_INDEX
	video_audio: bool = True
	wt_fit: bool = False	# trim world packs' top mips until they fit the engine's MEM1 pool
	wt_pool: int = 448 * 1024	# that pool (GC_WT_POOL_BYTES)
	wad_dirs: list[str] = field(default_factory=list)
	cache_dir: str = ""

	def sky_options(self, layout: str = "auto") -> skybox.SkyOptions:
		return skybox.SkyOptions(layout=layout, yaw=self.sky_yaw, fov=self.sky_fov, exposure=self.sky_exposure, size=self.sky_size)

	def cache(self) -> Path | None:
		return Path(self.cache_dir) if self.cache_dir else None

	def model_options(self) -> models.ModelOptions:
		return models.ModelOptions(self.model_scale, self.model_up_axis, self.model_skin_max_edge,
			self.image.alpha_threshold, self.model_anim_fps, self.model_static, self.model_clip_files,
			self.model_blend_bones, self.rose_scale, self.rose_yaw, self.panda_scale, self.panda_yaw)
			self.model_blend_bones, self.rose_scale, self.rose_yaw, self.rs_scale, self.rs_yaw)

	def to_json(self) -> dict:
		d = asdict(self)
		return d

	@classmethod
	def from_json(cls, d: dict) -> "Settings":
		d = dict(d)
		img = images.ImageOptions(**d.pop("image", {}))
		known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
		return cls(image=img, **known)


@dataclass
class Preview:
	"""One thing to look at: the source texels and what the GameCube will show."""
	title: str
	source: "np.ndarray | None" = None
	result: "np.ndarray | None" = None
	info: dict = field(default_factory=dict)


@dataclass
class OutputFile:
	rel: str
	size: int


@dataclass
class Asset:
	path: Path
	kind: Kind
	extra: dict = field(default_factory=dict)	# e.g. skybox sides
	status: str = "Ready"
	warnings: list[str] = field(default_factory=list)
	outputs: list[OutputFile] = field(default_factory=list)

	@property
	def name(self) -> str:
		if self.kind == Kind.SKYBOX:
			return f"sky: {self.extra['name']}"
		if "entry" in self.extra:
			return f"{self.path.name} › {self.extra['entry']} {self.extra.get('label', '')}".rstrip()
		return self.path.name

	@property
	def entry(self) -> "int | str | None":
		"""Which object/character of a ROSE list file (.zsc/.chr), or which npc:/model: of a RuneScape cache."""
		return self.extra.get("entry")

	@property
	def stem(self) -> str:
		"""Output file stem."""
		return self.extra.get("stem") or self.path.stem.lower()

	@property
	def is_rose(self) -> bool:
		return self.path.suffix.lower() in rose.ROSE_EXTENSIONS

	@property
	def is_panda(self) -> bool:
		return panda.is_panda(self.path)
	def is_runescape(self) -> bool:
		return runescape.is_cache(self.path)

	@property
	def size(self) -> int:
		try:
			if self.kind == Kind.SKYBOX:
				return sum(Path(p).stat().st_size for p in self.sky_files)
			return self.path.stat().st_size
		except OSError:
			return 0

	@property
	def sky_files(self) -> list[Path]:
		"""A skybox's source files: six GoldSrc sides, or a panorama/cross/strip/six faces."""
		if "source" in self.extra:
			return [Path(p) for p in self.extra["source"]]
		return [Path(p) for p in self.extra["sides"].values()]

	@property
	def output_size(self) -> int:
		return sum(o.size for o in self.outputs)


class Project:
	def __init__(self, settings: Settings | None = None):
		self.settings = settings or Settings()
		self.assets: list[Asset] = []
		self._model_cache: dict = {}
		self._sky_cache: dict = {}

	# -- loading ---------------------------------------------------------------

	def add_path(self, path: Path) -> list[Asset]:
		path = Path(path)
		if path.is_dir():
			files = [c for c in sorted(path.rglob("*")) if c.is_file() and detect_kind(c) != Kind.UNKNOWN]
			added = []
			for child in _skip_panda_parts(_skip_rose_parts(files)):
				added += self._add_file(child, root=path)
			return added
		return self._add_file(path)

	def _add_file(self, path: Path, root: Path | None = None) -> list[Asset]:
		path = path.resolve()
		if path.suffix.lower() in rose.LIST_EXTENSIONS:
			return self._add_rose_list(path)
		if runescape.is_cache(path):
			return self._add_rs_cache(path)
		if any(a.path == path for a in self.assets):
			return []
		kind = detect_kind(path)
		if kind == Kind.MODEL and models.clip_base(path) is not None:
			return []	# an animation clip; it is merged into its base model
		if kind == Kind.MODEL and panda.is_panda(path):
			if panda.is_animation_only(path):
				return []	# a Panda3D anim file; merged into the model it belongs to
			if path.name.lower().endswith(".pz"):
				asset = Asset(path, kind, {"stem": panda._stem(path).lower()})
				self.assets.append(asset)
				return [asset]
		if kind == Kind.IMAGE:
			sky = images.find_skybox_sides(path)
			if sky is not None:
				name, sides = sky
				if any(a.kind == Kind.SKYBOX and a.extra.get("name") == name for a in self.assets):
					return []
				asset = Asset(path, Kind.SKYBOX, {"name": name, "sides": {k: str(v) for k, v in sides.items()}})
				self.assets.append(asset)
				return [asset]
			faces = skybox.find_faces(path)
			if faces is not None and faces[1] == "view":
				return self.add_skybox(list(faces[2].values()), "faces", faces[0])
			if self.settings.detect_panoramas and skybox.looks_like_panorama(path):
				return self.add_skybox([path])
		rel = path.relative_to(root.resolve()).as_posix() if root else None
		asset = Asset(path, kind, warnings=audit.file_warnings(path, rel))
		self.assets.append(asset)
		return [asset]

	def add_skybox(self, paths: list[Path], layout: str = "auto", name: str | None = None) -> list[Asset]:
		"""A sky made from a panorama, cross, strip, photo or six face files (see core/skybox.py)."""
		paths = [Path(p).resolve() for p in paths]
		if any(a.kind == Kind.SKYBOX and set(a.sky_files) & set(paths) for a in self.assets):
			return []
		if name is None:
			found = skybox.find_faces(paths[0]) if layout in ("auto", "faces") and len(paths) == 1 else None
			name = found[0] if found and len(paths) > 1 else re.sub(r"[^a-z0-9_]+", "_", paths[0].stem.lower()).strip("_") or "sky"
		asset = Asset(paths[0], Kind.SKYBOX, {"name": name.lower(), "source": [str(p) for p in paths], "layout": layout})
		asset.warnings += skybox.check_name(name)
		self.assets.append(asset)
		return [asset]

	def sky(self, asset: Asset) -> skybox.Skybox:
		"""Render and encode a panorama-style skybox, cached per file and sky settings."""
		opts = self.settings.sky_options(asset.extra.get("layout", "auto"))
		files = asset.sky_files
		key = (tuple((str(p), p.stat().st_mtime_ns) for p in files), asset.extra["name"], opts.layout, opts.yaw,
			opts.fov, opts.exposure, opts.size)
		hit = self._sky_cache.get(id(asset))
		if hit and hit[0] == key:
			return hit[1]
		sky = skybox.build(files, opts, asset.extra["name"])
		for n in sky.notes:
			if n not in asset.warnings:
				asset.warnings.append(n)
		self._sky_cache[id(asset)] = (key, sky)
		return sky

	def _add_rose_list(self, path: Path) -> list[Asset]:
		"""One asset per object of a .zsc or character of a .chr."""
		have = {a.entry for a in self.assets if a.path == path}
		try:
			found = rose.entries(path)
		except (OSError, ValueError) as exc:
			if None in have:
				return []
			asset = Asset(path, Kind.MODEL, status="Failed", warnings=[f"could not read: {exc}"])
			self.assets.append(asset)
			return [asset]
		want = set(self.settings.rose_entries)
		added = []
		for e in found:
			if e.index in have or (want and e.index not in want):
				continue
			asset = Asset(path, Kind.MODEL, {"entry": e.index, "label": e.label, "stem": rose.entry_stem(path, e)})
			self.assets.append(asset)
			added.append(asset)
		return added

	def _add_rs_cache(self, path: Path) -> list[Asset]:
		"""One asset per NPC (or model) of a RuneScape cache."""
		have = {a.entry for a in self.assets if a.path == path}
		try:
			found = runescape.entries(path, self.settings.rs_entries)
		except (OSError, ValueError) as exc:
			if None in have:
				return []
			asset = Asset(path, Kind.MODEL, status="Failed", warnings=[f"could not read: {exc}"])
			self.assets.append(asset)
			return [asset]
		added = []
		for e in found:
			if e.key in have:
				continue
			asset = Asset(path, Kind.MODEL, {"entry": e.key, "label": e.label, "stem": runescape.entry_stem(e)})
			self.assets.append(asset)
			added.append(asset)
		return added

	def remove(self, asset: Asset) -> None:
		self.assets.remove(asset)

	# -- inspection ------------------------------------------------------------

	def children(self, asset: Asset) -> list[str]:
		"""Sub-items shown under an asset in the tree (textures, frames...)."""
		try:
			if asset.kind == Kind.WAD:
				return sorted(goldsrc.wad_miptex(asset.path))
			if asset.kind == Kind.BSP:
				data = asset.path.read_bytes()
				return list(dict.fromkeys(n for n, _ in goldsrc.bsp_miptex_offsets(data) if goldsrc.is_world_diffuse(n)))
			if asset.kind == Kind.GCT:
				return [e.name for e in gct.read_gct(asset.path.read_bytes())]
			if asset.kind == Kind.MDL:
				return [n for n, _f, _r in studio.studio_textures(asset.path.read_bytes())]
			if asset.kind == Kind.MODEL:
				model = self.imported_model(asset)[0]
				kids = [m.name for m in model.materials]
				if model.animated or len(model.bones) > 1:
					kids += [SEQ_PREFIX + sq.name for sq in model.sequences]
				return kids
			if asset.kind == Kind.SKYBOX:
				return list(images.GC_SKY_SUFFIXES) + ([SKY_NET] if "source" in asset.extra else [])
		except (OSError, ValueError, Exception) as exc:	# noqa: BLE001 - malformed input should not crash the UI
			asset.warnings.append(f"inspect failed: {exc}")
		return []

	def info(self, asset: Asset) -> dict:
		d: dict = {"Type": asset.kind.value, "Path": str(asset.path), "Size": audit.human_bytes(asset.size)}
		try:
			if asset.kind == Kind.MDL:
				d.update(studio.studio_info(asset.path.read_bytes()))
			elif asset.kind == Kind.MODEL:
				native = models.native_format(asset.path)
				if native == "source":
					from . import source_mdl

					d.update(source_mdl.source_info(asset.path))
				elif native == "quake":
					d["Source format"] = "Quake alias model (IDPO v6)"
				model, mdl = self.imported_model(asset)
				lo, hi = model.bounds()
				d["Meshes"] = len(model.meshes)
				d["Materials"] = len(model.materials)
				d["Vertices"] = model.num_vertices
				d["Triangles"] = model.num_triangles
				d["Bounds"] = " × ".join(f"{x:.1f}" for x in hi - lo) + " units"
				d["Bones"] = len(model.bones)
				d["Sequences"] = ", ".join(f"{sq.name} ({len(sq.frames)})" for sq in model.sequences)
				d["Studio MDL"] = audit.human_bytes(len(mdl))
				if asset.is_rose:
					d["Source"] = "ROSE Online" + (f", entry {asset.entry}" if asset.entry is not None else "")
				if asset.is_panda:
					kind = panda._kind(asset.path)
					d["Source"] = {".egg": "Panda3D egg", ".bam": "Panda3D bam", ".pandactor": "Panda3D multi-part actor"}[kind]
					d["Joints"] = ", ".join(b.name for b in model.bones[1:]) or "-"
					if kind != ".pandactor":
						anims = panda.animation_files(asset.path) if self.settings.model_clip_files else []
						d["Animation files"] = ", ".join(f.name for _l, f in anims) or "-"
				elif asset.is_runescape:
					d["Source"] = "RuneScape cache" + (f", {asset.entry}" if asset.entry is not None else "")
			elif asset.kind == Kind.SPR:
				raw = asset.path.read_bytes()
				if raw[:4] == b"IDS2" or (raw[:4] == b"IDSP" and struct.unpack_from("<i", raw, 4)[0] == 1):
					d["Source format"] = "Quake II sprite" if raw[:4] == b"IDS2" else "Quake sprite (v1)"
					raw = quake.hl_sprite_bytes(asset.path, asset.warnings)
				d.update(sprite.spr_info(raw))
			elif asset.kind == Kind.IMAGE and asset.path.suffix.lower() == ".vtf":
				from . import vtf

				d.update(vtf.vtf_info(asset.path.read_bytes()))
			elif asset.kind == Kind.BSP:
				data = asset.path.read_bytes()
				d["BSP version"] = quake.bsp_version(data)
				d["Sky"] = goldsrc.map_sky_name(data) or "-"
				d["WADs"] = ", ".join(goldsrc.wt_bsp_wad_basenames(data)) or "-"
			elif asset.kind == Kind.GCT:
				entries = gct.read_gct(asset.path.read_bytes())
				d["Textures"] = len(entries)
				d["Alpha textures"] = sum(e.alpha for e in entries)
			elif asset.kind == Kind.GCVID:
				d.update(media.read_gcvid_header(asset.path.read_bytes()[:64]))
			elif asset.kind == Kind.GCPCM:
				d.update(media.read_gcpcm_header(asset.path.read_bytes()[:16]))
			elif asset.kind == Kind.WAD:
				d["Textures"] = len(goldsrc.wad_miptex(asset.path))
		except Exception as exc:	# noqa: BLE001
			d["Error"] = str(exc)
		return d

	def preview(self, asset: Asset, child: str | None = None) -> Preview:
		"""Source texels next to the decoded GX result for one asset/sub-item."""
		s = self.settings
		k = asset.kind
		if k == Kind.IMAGE:
			src = images.load_rgba(asset.path, asset.warnings)
			entry = images.encode_image(src, s.image, asset.path.stem)
			return _texture_preview(asset.name, src, entry)
		if k == Kind.SKYBOX and "source" in asset.extra:
			sky = self.sky(asset)
			info = {"Layout": sky.layout, "Sky name": sky.name, "Yaw": f"{s.sky_yaw:g}° (centre → "
				f"{skybox.SIDES[round(s.sky_yaw / 90) % 4]})", "Files": f"media/sky_{sky.name}{{rt,bk,lf,ft,up,dn}}.cmp"}
			dec = sky.decoded()
			if child in (None, SKY_NET):
				src = skybox.cross_preview(sky.sides, s.sky_yaw)
				res = skybox.cross_preview(dec, s.sky_yaw)
				info.update(_stats(src, res, sum(map(len, sky.cmpr.values())), "6 × CMPR, no mips"))
				info["View"] = "cross, front in the middle"
				return Preview(f"sky {sky.name}", src, res, info)
			src = sky.sides[child]
			res = dec[child]
			info.update(_stats(src, res, len(sky.cmpr[child]), "CMPR, no mips"))
			return Preview(f"{sky.name}{child}", src, res, info)
		if k == Kind.SKYBOX:
			side = child or "ft"
			src = images.load_rgba(Path(asset.extra["sides"][side]))
			data = images.sky_side_cmpr(src, s.sky_size)
			res = cmpr_decode(data, s.sky_size, s.sky_size)
			return Preview(f"{asset.extra['name']}{side}", src, res, _stats(src, res, len(data), "CMPR, no mips"))
		if k in (Kind.WAD, Kind.BSP):
			names = self.children(asset)
			name = child or (names[0] if names else None)
			if name is None:
				return Preview(asset.name, info={"Note": "no textures"})
			mip = self._find_miptex(asset, name)
			if mip is None:
				return Preview(name, info={"Status": "missing: not embedded and not found in any WAD search dir"})
			_, w, h, pixels, palette = mip
			alpha = name.startswith("{")
			src = goldsrc.miptex_rgba(w, h, pixels, palette, alpha)
			key, tw, th, levels, body = gct.wt_encode_cached(w, h, pixels, palette, alpha, s.cache())
			entry = gct.GctEntry(name, tw, th, levels, gct.GC_WT_FLAG_ALPHA if alpha else 0, int(key[:4], 16), body)
			return _texture_preview(name, src, entry)
		if k == Kind.GCT:
			entries = gct.read_gct(asset.path.read_bytes())
			e = next((x for x in entries if x.name == child), entries[0] if entries else None)
			if e is None:
				return Preview(asset.name)
			res = e.decode(0)
			return Preview(e.name, None, res, {"Size": f"{e.width}x{e.height}", "Mip levels": e.levels,
				"Alpha": "1-bit" if e.alpha else "none", "Bytes": audit.human_bytes(len(e.data)), "Tag": f"{e.tag:04x}"})
		if k == Kind.CMP:
			data = asset.path.read_bytes()
			res = images.decode_raw_cmp(data)
			return Preview(asset.name, None, res, {"Size": f"{res.shape[1]}x{res.shape[0]}", "Bytes": audit.human_bytes(len(data))})
		if k == Kind.MDL:
			raw = asset.path.read_bytes()
			skins = studio.studio_textures(raw)
			if not skins:
				return Preview(asset.name, info={"Note": "no embedded skins (already lean, or uses a T.mdl)"})
			idx = next((i for i, (n, _f, _r) in enumerate(skins) if n == child), 0)
			name, _flags, src = skins[idx]
			comp = studio.lean_studio_tex_companion_bytes(asset.path, s.mdl_max_edge)
			res = studio.studio_textures(comp)[idx][2] if comp else None
			info = {"Source": f"{src.shape[1]}x{src.shape[0]}"}
			if res is not None:
				info["Companion"] = f"{res.shape[1]}x{res.shape[0]}"
				info["Skin companion bytes"] = audit.human_bytes(len(comp))
			lean = studio.lean_studio_mesh_bytes(asset.path)
			if lean:
				info["Lean mesh bytes"] = f"{audit.human_bytes(len(lean))} (from {audit.human_bytes(len(raw))})"
			return Preview(name, src, res, info)
		if k == Kind.MODEL:
			return self._model_preview(asset, child)
		if k == Kind.SPR:
			raw = quake.hl_sprite_bytes(asset.path, asset.warnings)
			frames = sprite.spr_frames(raw)
			src = frames[0] if frames else None
			lean = sprite.lean_hl_spr_bytes_from(raw, s.spr_scale)
			res_frames = sprite.spr_frames(lean) if lean else []
			info = {"Bytes": audit.human_bytes(len(raw))}
			if lean:
				info["Lean bytes"] = audit.human_bytes(len(lean))
			else:
				info["Lean"] = "not applicable (multi-frame, or size not divisible by scale)"
			return Preview(asset.name, src, res_frames[0] if res_frames else None, info)
		if k == Kind.VIDEO:
			src = media.video_still(asset.path)
			small = media.video_still(asset.path, width=s.video_width, height=s.video_height)
			res = media.rgb565_quantise(small) if small is not None else None
			info = {"Target": f"{s.video_width}x{s.video_height} @ {s.video_fps} fps RGB565 delta"}
			if src is None:
				info["Note"] = "ffmpeg not found; install it or set GXBAKE_FFMPEG"
			return Preview(asset.name, src, res, info)
		if k == Kind.GCVID:
			data = asset.path.read_bytes()
			return Preview(asset.name, None, media.gcvid_first_frame(data), media.read_gcvid_header(data))
		return Preview(asset.name, info=self.info(asset))

	def imported_model(self, asset: Asset) -> "tuple[models.ImportedModel, bytes]":
		"""Assimp import + baked studio MDL, cached per file and model settings."""
		opts = self.settings.model_options()
		try:
			mtime = asset.path.stat().st_mtime_ns
		except OSError:
			mtime = 0
		clips = tuple((p.name, p.stat().st_mtime_ns) for p in models.clip_files(asset.path)) if opts.clip_files else ()
		key = (asset.path, mtime, clips, opts.scale, opts.up_axis, opts.skin_max_edge, opts.alpha_threshold,
			opts.anim_fps, opts.static, opts.blend_bones)
		hit = self._model_cache.get(asset.path)
		if asset.is_rose:
			clips: tuple = (opts.rose_scale, opts.rose_yaw)
		elif asset.is_panda:
			clips = (opts.panda_scale, opts.panda_yaw, opts.clip_files, *(
				(f.name, f.stat().st_mtime_ns) for _l, f in (panda.animation_files(asset.path) if opts.clip_files else [])))
		elif asset.is_runescape:
			clips = (opts.rs_scale, opts.rs_yaw)
		else:
			clips = tuple((p.name, p.stat().st_mtime_ns) for p in models.clip_files(asset.path)) if opts.clip_files else ()
		key = (asset.path, asset.entry, mtime, clips, opts.scale, opts.up_axis, opts.skin_max_edge,
			opts.alpha_threshold, opts.anim_fps, opts.static)
		ck = (asset.path, asset.entry)
		hit = self._model_cache.get(ck)
		if hit and hit[0] == key:
			return hit[1], hit[2]
		if asset.is_rose:
			model = rose.import_rose(asset.path, opts, asset.entry)
		elif asset.is_panda:
			model = panda.import_panda(asset.path, opts)
		elif asset.is_runescape:
			model = runescape.import_runescape(asset.path, opts, asset.entry)
		else:
			model = models.import_model(asset.path, opts)
		mdl = models.build_studio_mdl(model, opts)
		for w in model.warnings:
			if w not in asset.warnings:
				asset.warnings.append(w)
		self._model_cache[ck] = (key, model, mdl)
		return model, mdl

	def _model_preview(self, asset: Asset, child: str | None) -> Preview:
		s = self.settings
		model, mdl = self.imported_model(asset)
		comp = studio.lean_studio_tex_companion_bytes_from(mdl, s.mdl_max_edge)
		lean = studio.lean_studio_mesh_bytes_from(mdl)
		info = {"Vertices": model.num_vertices, "Triangles": model.num_triangles,
			"Studio MDL": audit.human_bytes(len(mdl))}
		if lean:
			info["Lean mesh bytes"] = audit.human_bytes(len(lean))
		if comp:
			info["Skin companion bytes"] = audit.human_bytes(len(comp))
		info["Bones"] = len(model.bones)
		names = [m.name for m in model.materials]
		if child and child.startswith(SEQ_PREFIX):
			seqs = [sq.name for sq in model.sequences]
			i = seqs.index(child[len(SEQ_PREFIX):]) if child[len(SEQ_PREFIX):] in seqs else 0
			sq = model.sequences[i]
			frame = (len(sq.frames) - 1) // 2
			info = {"Frames": len(sq.frames), "Frame rate": f"{sq.fps:g} fps",
				"Length": f"{(len(sq.frames) - 1) / sq.fps:.2f} s", "Looping": "yes" if sq.loop else "no",
				"Shown": f"source rest pose / frame {frame} as the engine decodes it", "Studio MDL": audit.human_bytes(len(mdl))}
			return Preview(child, models.preview_imported(model), models.preview_studio(mdl, comp, seq=i, frame=frame), info)
		if child in names:
			i = names.index(child)
			mat = model.materials[i]
			res = studio.studio_textures(comp)[i][2] if comp else None
			info = {"Texture": mat.texture or "(material colour)", "Skin": f"{mat.rgba.shape[1]}x{mat.rgba.shape[0]}",
				"Masked": "yes (index 255)" if mat.masked else "no"}
			if res is not None:
				info["Companion"] = f"{res.shape[1]}x{res.shape[0]}"
			return Preview(child, mat.rgba, res, info)
		return Preview(asset.name, models.preview_imported(model), models.preview_studio(mdl, comp), info)

	def _quake_palette(self, asset: Asset) -> bytes:
		pal = quake.quake_palette(asset.path)
		if pal is None:
			msg = "Quake palette (gfx/palette.lmp) not found; greyscale used"
			if msg not in asset.warnings:
				asset.warnings.append(msg)
			pal = quake.GREY_PALETTE
		return pal

	def _needs_quake_palette(self, asset: Asset) -> bool:
		head = _head(asset.path, 4)
		return head == b"WAD2" or (asset.kind == Kind.BSP and head == struct.pack("<i", 29))

	def _find_miptex(self, asset: Asset, name: str):
		pal = self._quake_palette(asset) if self._needs_quake_palette(asset) else None
		if asset.kind == Kind.WAD:
			found = goldsrc.wad_miptex(asset.path)
			return goldsrc.read_miptex(*found[name], pal) if name in found else None
		data = asset.path.read_bytes()
		for n, off in goldsrc.bsp_miptex_offsets(data):
			if n == name:
				mip = goldsrc.read_miptex(data, off, pal)
				if mip is not None:
					return mip
				break
		gamedir = asset.path.parent.parent if asset.path.parent.name.lower() == "maps" else asset.path.parent
		wads, _ = gct.collect_wads(data, gamedir, [Path(p) for p in self.settings.wad_dirs])
		return goldsrc.wt_read_miptex(*wads[name]) if name in wads else None

	# -- conversion ------------------------------------------------------------

	def convert(
		self,
		assets: list[Asset],
		out_dir: Path,
		log: Log = print,
		progress: Progress | None = None,
	) -> list[OutputFile]:
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)
		s = self.settings
		written: list[OutputFile] = []
		pack: list[gct.GctEntry] = []
		pack_assets: list[Asset] = []

		def emit(asset: Asset, rel: str, data: bytes) -> None:
			dest = out_dir / rel
			dest.parent.mkdir(parents=True, exist_ok=True)
			dest.write_bytes(data)
			of = OutputFile(rel, len(data))
			asset.outputs.append(of)
			written.append(of)
			log(f"  wrote {rel} ({audit.human_bytes(len(data))})")

		def emit_file(asset: Asset, rel: str) -> None:
			dest = out_dir / rel
			if dest.is_file():
				of = OutputFile(rel, dest.stat().st_size)
				asset.outputs.append(of)
				written.append(of)

		todo = [a for a in assets if a.kind in CONVERTIBLE]
		for i, asset in enumerate(todo):
			if progress:
				progress(i, len(todo), asset.name)
			asset.outputs = []
			asset.status = "Converting"
			log(f"{asset.name} [{asset.kind.value}]")
			t0 = time.monotonic()
			try:
				stem = asset.stem
				if asset.kind == Kind.IMAGE:
					src = images.load_rgba(asset.path, asset.warnings)
					entry = images.encode_image(src, s.image, asset.path.stem)
					if s.pack_images:
						pack.append(entry)
						pack_assets.append(asset)
					else:
						emit(asset, f"media/{stem}.gct", gct.write_gct([entry]))
					if s.raw_cmp:
						emit(asset, f"media/{stem}.cmp", entry.data[:entry.level_size(0)])
				elif asset.kind == Kind.SKYBOX and "source" in asset.extra:
					sky = self.sky(asset)
					log(f"  {sky.layout} → 6 sides, yaw {s.sky_yaw:g}°")
					for n in sky.notes:
						log("  note: " + n)
					for rel, data in sky.files().items():
						emit(asset, rel, data)
				elif asset.kind == Kind.SKYBOX:
					name = asset.extra["name"]
					for sfx in images.GC_SKY_SUFFIXES:
						src = images.load_rgba(Path(asset.extra["sides"][sfx]))
						emit(asset, f"media/sky_{name}{sfx}.cmp", images.sky_side_cmpr(src, s.sky_size))
				elif asset.kind == Kind.WAD:
					pal = self._quake_palette(asset) if self._needs_quake_palette(asset) else None
					data, entries = gct.build_wad_texture_pack(asset.path, cache=s.cache(), palette=pal)
					if data:
						emit(asset, f"media/{stem}.gct", data)
				elif asset.kind == Kind.BSP:
					data, report, _entries = gct.build_world_texture_pack(
						asset.path, wad_dirs=[Path(p) for p in s.wad_dirs], cache=s.cache())
					log(f"  {len(report.wads)} WADs searched, {report.entries} textures packed")
					for note in report.notes:
						asset.warnings.append(note)
						log("  warning: " + note)
					if report.missing:
						msg = f"{len(report.missing)} textures unresolved: {', '.join(report.missing[:8])}"
						asset.warnings.append(msg)
						log("  warning: " + msg)
					if data and s.wt_fit:
						data, res = budget.fit_pack_bytes(data, budget.Budgets(wt_pool=s.wt_pool))
						if res.trimmed:
							log(f"  fitted to the {s.wt_pool // 1024} KiB pool: {len(res.trimmed)} textures halved or more, "
								f"{res.before.data_bytes // 1024} → {res.after.data_bytes // 1024} KiB")
						if not res.ok:
							asset.warnings.append(f"world pack still over budget: {res.reason}")
					if data:
						emit(asset, f"media/wt_{stem}.gct", data)
					self._bsp_sky(asset, emit)
				elif asset.kind == Kind.MDL:
					if s.mdl_lean_mesh:
						lean = studio.lean_studio_mesh_bytes(asset.path)
						if lean and s.mdl_anim_step > 1:
							try:
								slim = studio_slim.slim_studio(lean, s.mdl_anim_step)
								if slim.changed:
									log(f"  animation: every {s.mdl_anim_step}th frame, {len(lean) // 1024} → {len(slim.data) // 1024} KiB")
								lean = slim.data
							except ValueError as exc:
								log(f"  animation left as is: {exc}")
						if lean:
							emit(asset, f"gc_studio/{stem}.mdl", lean)
					if s.mdl_skin_companion:
						comp = studio.lean_studio_tex_companion_bytes(asset.path, s.mdl_max_edge)
						if comp:
							emit(asset, f"gc_studio/{stem}t.mdl", comp)
						else:
							log("  no embedded skins; companion skipped")
				elif asset.kind == Kind.MODEL:
					model, mdl = self.imported_model(asset)
					log(f"  {model.num_vertices} vertices, {model.num_triangles} triangles, {len(model.materials)} skins")
					for w in model.warnings:
						log("  warning: " + w)
					if s.model_write_full:
						emit(asset, f"models/{stem}.mdl", mdl)
					if s.mdl_lean_mesh:
						lean = studio.lean_studio_mesh_bytes_from(mdl)
						if lean:
							emit(asset, f"gc_studio/{stem}.mdl", lean)
					if s.mdl_skin_companion:
						comp = studio.lean_studio_tex_companion_bytes_from(mdl, s.mdl_max_edge)
						if comp:
							emit(asset, f"gc_studio/{stem}t.mdl", comp)
				elif asset.kind == Kind.SPR:
					raw = quake.hl_sprite_bytes(asset.path, asset.warnings)
					if raw[:8] != asset.path.read_bytes()[:8]:
						log("  converted to a Half-Life (v2) sprite")
					lean = sprite.lean_hl_spr_bytes_from(raw, s.spr_scale)
					if lean is None:
						log("  sprite not leanable (multi-frame or size not divisible); copied as-is")
						lean = raw
					emit(asset, f"sprites/{asset.path.stem.lower()}.spr", lean)
				elif asset.kind == Kind.VIDEO:
					rel = f"media/{stem}.gcvid"
					media.build_gcvid_companion(
						asset.path, out_dir / rel,
						width=s.video_width, height=s.video_height,
						fps_num=s.video_fps, fps_den=1,
						still_frame_index=s.video_still_frame, rgb565=True, log=log)
					emit_file(asset, rel)
					if s.video_audio and media.build_gcpcm_companion(asset.path, out_dir / f"media/{stem}.gcpcm", log=log):
						emit_file(asset, f"media/{stem}.gcpcm")
				elif asset.kind == Kind.AUDIO:
					rel = f"media/{stem}.gcpcm"
					if media.build_gcpcm_companion(asset.path, out_dir / rel, log=log):
						emit_file(asset, rel)
				asset.status = f"Done ({audit.human_bytes(asset.output_size)})"
				log(f"  done in {time.monotonic() - t0:.1f}s")
			except Exception as exc:	# noqa: BLE001 - report and keep going
				asset.status = "Failed"
				asset.warnings.append(str(exc))
				log(f"  error: {exc}")
		if pack:
			rel = f"media/{s.pack_name or 'textures'}.gct"
			data = gct.write_gct(pack)
			dest = out_dir / rel
			dest.parent.mkdir(parents=True, exist_ok=True)
			dest.write_bytes(data)
			of = OutputFile(rel, len(data))
			written.append(of)
			for a in pack_assets:
				a.outputs.append(OutputFile(rel, len(data) // len(pack_assets)))
				a.status = f"Packed into {rel}"
			log(f"packed {len(pack)} images into {rel} ({audit.human_bytes(len(data))})")
		if progress:
			progress(len(todo), len(todo), "")
		self.write_manifest(out_dir, written)
		return written

	def _bsp_sky(self, asset: Asset, emit) -> None:
		data = asset.path.read_bytes()
		name = goldsrc.map_sky_name(data)
		if not name:
			return
		gamedir = asset.path.parent.parent if asset.path.parent.name.lower() == "maps" else asset.path.parent
		sides = {}
		for sfx in images.GC_SKY_SUFFIXES:
			for ext in ("tga", "bmp", "png"):
				p = gamedir / "gfx/env" / f"{name}{sfx}.{ext}"
				if p.is_file():
					sides[sfx] = p
					break
		if len(sides) != 6:
			return
		for sfx, p in sides.items():
			emit(asset, f"media/sky_{name}{sfx}.cmp", images.sky_side_cmpr(images.load_rgba(p), self.settings.sky_size))

	def write_manifest(self, out_dir: Path, written: list[OutputFile]) -> None:
		total = sum(o.size for o in written)
		manifest = {
			"tool": "GxBake",
			"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
			"settings": self.settings.to_json(),
			"total_bytes": total,
			"main_ram_fraction": round(total / audit.MAIN_RAM, 4),
			"outputs": [asdict(o) for o in written],
			"assets": [
				{"source": str(a.path), "kind": a.kind.value, "status": a.status,
					"warnings": a.warnings, "outputs": [asdict(o) for o in a.outputs]}
				for a in self.assets
			],
		}
		(out_dir / "gxbake-manifest.json").write_text(json.dumps(manifest, indent=2))


def _skip_rose_parts(files: list[Path]) -> list[Path]:
	"""A folder scan of ROSE data keeps the list files (.zsc/.chr) and drops what they
	reference: loose .zms meshes and .dds textures, and the PART_*.ZSC beside a .chr."""
	lists = [f for f in files if f.suffix.lower() in rose.LIST_EXTENSIONS]
	if not lists:
		return files
	chr_dirs = {f.parent for f in lists if f.suffix.lower() == ".chr"}
	out = []
	for f in files:
		ext = f.suffix.lower()
		if ext in (".zms", ".dds"):
			continue
		if ext == ".zsc" and f.parent in chr_dirs and f.name.lower().startswith("part_"):
			continue
		out.append(f)
	return out


def _skip_panda_parts(files: list[Path]) -> list[Path]:
	"""A folder scan with Panda3D models (an extracted Toontown phase tree) drops the textures
	their eggs/bams point at: images in maps/ folders and the *_a.rgb alpha masks."""
	if not any(panda.is_panda(f) for f in files):
		return files
	out = []
	for f in files:
		if f.suffix.lower() in images.IMAGE_EXTENSIONS and not panda.is_panda(f):
			if any(p.lower() == "maps" for p in f.parent.parts) or f.stem.lower().endswith("_a"):
				continue
		out.append(f)
	return out


def _stats(src: "np.ndarray | None", res: "np.ndarray | None", nbytes: int, fmt: str) -> dict:
	d = {"Format": fmt, "GX bytes": audit.human_bytes(nbytes)}
	if src is not None:
		d["Source"] = f"{src.shape[1]}x{src.shape[0]}"
	if res is not None:
		d["Result"] = f"{res.shape[1]}x{res.shape[0]}"
	if src is not None and res is not None:
		from PIL import Image

		ref = src if src.shape[:2] == res.shape[:2] else np.asarray(
			Image.fromarray(src).resize((res.shape[1], res.shape[0]), Image.Resampling.LANCZOS))
		d["PSNR"] = f"{images.psnr(ref, res):.1f} dB"
		d["Source RGBA bytes"] = audit.human_bytes(src.shape[0] * src.shape[1] * 4)
	return d


def _texture_preview(title: str, src: "np.ndarray", entry: gct.GctEntry) -> Preview:
	res = entry.decode(0)
	info = _stats(src, res, len(entry.data), f"CMPR, {entry.levels} mip level{'s' * (entry.levels > 1)}")
	info["Alpha"] = "1-bit" if entry.alpha else "none"
	return Preview(title, src, res, info)
