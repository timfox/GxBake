"""Memory budgets for the Xash3D GameCube port, per map.

The numbers are the engine's own windows, not the raw hardware sizes. Main RAM
is 24 MiB and ARAM 16 MiB, but by the time a map loads, world textures get a
448 KiB MEM1 pool, studio models a 1 MiB arena, sound a 256 KiB cache, and
ARAM is carved into tagged stores (docs/GAMECUBE_ARAM_STREAMING_DESIGN.md and
engine/common/mod_bmodel.c in the port). A map whose assets overflow one of
those windows still loads, but degrades: textures draw from low mips, models
are refused by name, sounds are refused once the cache is full.

simulate_world_pack() mirrors Mod_GCWorldPackSplitDim, so it predicts the
split the engine will pick for a .gct. fit_world_pack() trims top mip levels
(no re-encode, so it is fast and adds no new artefacts) until a pack fits.
map_budget() puts it together for one BSP in a staged gamedir.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import audit, goldsrc, sfx
from .gct import GC_WT_ENTRY, GC_WT_MAGIC, GctEntry, read_gct, write_gct

KiB = 1024
MiB = 1024 * 1024

# Nintendo GameCube optical disc: 712,880 sectors of 2048 bytes.
DISC_CAPACITY = 1_459_978_240

BSP_LUMP_LIGHTING = 8


@dataclass
class Budgets:
	"""Every limit map_budget() checks, in bytes unless noted."""
	main_ram: int = audit.MAIN_RAM
	aram: int = audit.ARAM
	disc: int = DISC_CAPACITY
	wt_pool: int = 448 * KiB	# GC_WT_POOL_BYTES: table + resident low mips + hot ring
	wt_min_hot: int = 32 * KiB	# GC_WT_MIN_HOT_BYTES
	wt_split: int = 64	# GC_WT_SPLIT_DIM: the split the pool should manage without going soft
	wt_aram: int = 6 * MiB	# GC_WT_ARAM_BYTES ("wtstore")
	wt_max_textures: int = 512	# GC_WT_MAX_TEXTURES
	studio_arena: int = 1 * MiB	# Mod_GCStudioBssAlloc arena, per map
	sfx_cache: int = 128 * KiB	# GC_SFX_BUDGET_BYTES (320K when the menu runs the server)
	sfx_max_file: int = 20 * KiB	# GC_SFX_MAX_FILE_BYTES: larger WAVs are refused (48K on the menu route)
	sfx_aram: int = 2 * MiB	# "sfx" ARAM store
	lightmap_atlas: int = 800 * KiB

	def override(self, specs: list[str]) -> "Budgets":
		"""Apply "name=size" overrides, e.g. wt_pool=640K or studio_arena=1.5M."""
		known = {f.name for f in fields(self)}
		for spec in specs:
			name, sep, value = spec.partition("=")
			name = name.strip().replace("-", "_")
			if not sep or name not in known:
				raise ValueError(f"unknown budget '{spec}'; known: {', '.join(sorted(known))}")
			setattr(self, name, parse_size(value))
		return self


# What each Budgets field is, for --budget NAME=SIZE and docs/cli.md.
BUDGET_HELP = {
	"main_ram": "GameCube main RAM (MEM1)",
	"aram": "auxiliary RAM (ARAM)",
	"disc": "usable bytes on a GameCube disc",
	"wt_pool": "world texture MEM1 pool: table, resident low mips and hot ring (GC_WT_POOL_BYTES)",
	"wt_min_hot": "smallest hot ring the split rule accepts (GC_WT_MIN_HOT_BYTES)",
	"wt_split": "split dimension a pack must fit at to count as ok (a number, not a size)",
	"wt_aram": "world texture ARAM store (GC_WT_ARAM_BYTES)",
	"wt_max_textures": "entries per world pack the engine reads (a count)",
	"studio_arena": "studio model MEM1 arena per map",
	"sfx_cache": "MEM1 sound cache (GC_SFX_BUDGET_BYTES; 320K on the menu route)",
	"sfx_max_file": "largest WAV PCM the loader accepts (GC_SFX_MAX_FILE_BYTES; 48K on the menu route)",
	"sfx_aram": "ARAM sound store",
	"lightmap_atlas": "lightmap atlas",
}


def parse_size(text: str) -> int:
	"""'448K', '1.5M', '6MiB', '2048' → bytes."""
	m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kmg]?)(i?b)?\s*", text, re.IGNORECASE)
	if not m:
		raise ValueError(f"not a size: {text!r}")
	scale = {"": 1, "k": KiB, "m": MiB, "g": 1024 * MiB}[m.group(2).lower()]
	return int(float(m.group(1)) * scale)


# --- world texture packs ------------------------------------------------------

@dataclass
class PackFit:
	count: int
	table_bytes: int	# header + table, 32-aligned ("first" in the engine)
	data_bytes: int	# the whole blob: what goes to the ARAM store
	max_chain: int
	split: int | None	# split dimension the engine picks, None when even 8 overflows
	resident: int	# low-mip bytes held in MEM1 at that split
	hot: int	# what is left for the hot ring
	resident_by_split: dict[int, int] = field(default_factory=dict)

	def fits(self, pool_split: int) -> bool:
		return self.split is not None and self.split >= pool_split


def _rows(data: bytes) -> tuple[list[tuple[str, int, int, int, int]], int, int]:
	"""(name, w, h, levels, offset) for each entry, data_bytes, first."""
	if data[:4] != GC_WT_MAGIC:
		raise ValueError("not a GCWT pack")
	count, data_bytes = struct.unpack_from(">II", data, 4)
	rows = []
	for i in range(count):
		raw, w, h, lv, _fl, _tag, of = GC_WT_ENTRY.unpack_from(data, 12 + i * GC_WT_ENTRY.size)
		rows.append((raw.split(b"\0")[0].decode("latin-1"), w, h, lv, of))
	first = (12 + count * GC_WT_ENTRY.size + 31) & ~31
	return rows, data_bytes, first


def _simulate(dims: list[tuple[int, int, int, int]], data_bytes: int, first: int, pool: int,
		min_hot: int) -> PackFit:
	"""dims: (w, h, levels, chain bytes incl. padding) per entry, in pack order."""
	max_chain = max((b for *_x, b in dims), default=0)
	by_split: dict[int, int] = {}
	chosen = None
	for cand in (64, 32, 16, 8):
		resident = 0
		for w, h, lv, b in dims:
			split = lv > 1 and (w > cand or h > cand)
			resident += b - (w * h // 2 if split else 0)
		by_split[cand] = resident
		if chosen is None and first <= pool and resident <= pool - first \
				and max(max_chain, min_hot) <= pool - first - resident:
			chosen = cand
	resident = by_split[chosen if chosen is not None else 8]
	return PackFit(len(dims), first, data_bytes, max_chain, chosen, resident,
		pool - first - resident, by_split)


def simulate_world_pack(data: bytes, budgets: Budgets | None = None) -> PackFit:
	"""Which split Mod_GCWorldPackSplitDim picks for this pack and pool."""
	b = budgets or Budgets()
	rows, data_bytes, first = _rows(data)
	dims = []
	for i, (_n, w, h, lv, of) in enumerate(rows):
		end = rows[i + 1][4] if i + 1 < len(rows) else data_bytes
		dims.append((w, h, lv, end - of))
	return _simulate(dims, data_bytes, first, b.wt_pool, b.wt_min_hot)


def _entries_fit(entries: list[GctEntry], b: Budgets) -> PackFit:
	first = (12 + len(entries) * GC_WT_ENTRY.size + 31) & ~31
	dims = [(e.width, e.height, e.levels, len(e.data) + (-len(e.data) & 31)) for e in entries]
	return _simulate(dims, sum(d[3] for d in dims), first, b.wt_pool, b.wt_min_hot)


def drop_top_mip(e: GctEntry) -> GctEntry:
	"""The same texture one level smaller: level 1 becomes level 0. UVs are
	normalised, so nothing on the map moves."""
	if e.levels < 2:
		raise ValueError(f"{e.name} has no mip below level 0")
	return GctEntry(e.name, max(8, e.width >> 1), max(8, e.height >> 1), e.levels - 1, e.flags, e.tag,
		e.data[e.level_size(0):])


@dataclass
class FitResult:
	entries: list[GctEntry]
	before: PackFit
	after: PackFit
	trimmed: dict[str, tuple[int, int, int, int]]	# name → (old w, old h, new w, new h)
	ok: bool
	reason: str = ""


def fit_world_pack(entries: list[GctEntry], budgets: Budgets | None = None,
		max_edge: int | None = None) -> FitResult:
	"""Drop top mip levels, largest textures first, until the pack fits the
	MEM1 pool at budgets.wt_split and the ARAM store.

	Largest-first costs the least detail per byte saved: halving a 256² texture
	frees 32 KiB, and it was the texture most likely to be minified anyway.
	"""
	b = budgets or Budgets()
	work = list(entries)
	original = {e.name: (e.width, e.height) for e in entries}
	before = _entries_fit(work, b)
	if max_edge:
		for i, e in enumerate(work):
			while max(e.width, e.height) > max_edge and e.levels > 1:
				e = drop_top_mip(e)
			work[i] = e

	def over(fit: PackFit) -> bool:
		return not fit.fits(b.wt_split) or fit.data_bytes > b.wt_aram

	fit = _entries_fit(work, b)
	while over(fit):
		cands = [i for i, e in enumerate(work) if e.levels > 1 and max(e.width, e.height) > 8]
		if not cands:
			break
		i = max(cands, key=lambda k: (work[k].width * work[k].height, len(work[k].data)))
		work[i] = drop_top_mip(work[i])
		fit = _entries_fit(work, b)
	trimmed = {e.name: (*original[e.name], e.width, e.height)
		for e in work if (e.width, e.height) != original[e.name]}
	reason = ""
	if over(fit):
		reason = "textures without mip chains cannot be trimmed" if any(e.levels < 2 for e in work) \
			else "every texture is already at its smallest mip"
	if len(work) > b.wt_max_textures:
		reason = f"{len(work)} textures; the engine reads at most {b.wt_max_textures}"
	return FitResult(work, before, fit, trimmed, not over(fit) and len(work) <= b.wt_max_textures, reason)


# --- what a map pulls in --------------------------------------------------------

# Models the Half-Life game DLL precaches for a classname without a "model"
# key. Not exhaustive: anything missing can be added with --include.
CLASSNAME_MODELS = {
	"monster_barney": ["models/barney.mdl"],
	"monster_scientist": ["models/scientist.mdl"],
	"monster_sitting_scientist": ["models/scientist.mdl"],
	"monster_zombie": ["models/zombie.mdl"],
	"monster_headcrab": ["models/headcrab.mdl"],
	"monster_babycrab": ["models/baby_headcrab.mdl"],
	"monster_houndeye": ["models/houndeye.mdl"],
	"monster_bullchicken": ["models/bullsquid.mdl"],
	"monster_human_grunt": ["models/hgrunt.mdl"],
	"monster_human_assassin": ["models/hassassin.mdl"],
	"monster_alien_slave": ["models/islave.mdl"],
	"monster_alien_grunt": ["models/agrunt.mdl"],
	"monster_alien_controller": ["models/controller.mdl"],
	"monster_gman": ["models/gman.mdl"],
	"monster_barnacle": ["models/barnacle.mdl"],
	"monster_ichthyosaur": ["models/icky.mdl"],
	"monster_leech": ["models/leech.mdl"],
	"monster_cockroach": ["models/roach.mdl"],
	"monster_snark": ["models/w_squeak.mdl"],
	"monster_tentacle": ["models/tentacle2.mdl"],
	"monster_gargantua": ["models/garg.mdl"],
	"monster_bigmomma": ["models/big_mom.mdl"],
	"monster_osprey": ["models/osprey.mdl"],
	"monster_apache": ["models/apache.mdl"],
	"monster_turret": ["models/turret.mdl"],
	"monster_miniturret": ["models/miniturret.mdl"],
	"monster_sentry": ["models/sentry.mdl"],
	"item_healthkit": ["models/w_medkit.mdl"],
	"item_battery": ["models/w_battery.mdl"],
	"item_suit": ["models/w_suit.mdl"],
	"item_longjump": ["models/w_longjump.mdl"],
	"item_security": ["models/w_security.mdl"],
	"weapon_crowbar": ["models/w_crowbar.mdl", "models/v_crowbar.mdl", "models/p_crowbar.mdl"],
	"weapon_9mmhandgun": ["models/w_9mmhandgun.mdl", "models/v_9mmhandgun.mdl", "models/p_9mmhandgun.mdl"],
	"weapon_glock": ["models/w_9mmhandgun.mdl", "models/v_9mmhandgun.mdl", "models/p_9mmhandgun.mdl"],
	"weapon_357": ["models/w_357.mdl", "models/v_357.mdl", "models/p_357.mdl"],
	"weapon_9mmAR": ["models/w_9mmAR.mdl", "models/v_9mmAR.mdl", "models/p_9mmAR.mdl"],
	"weapon_mp5": ["models/w_9mmAR.mdl", "models/v_9mmAR.mdl", "models/p_9mmAR.mdl"],
	"weapon_shotgun": ["models/w_shotgun.mdl", "models/v_shotgun.mdl", "models/p_shotgun.mdl"],
	"weapon_crossbow": ["models/w_crossbow.mdl", "models/v_crossbow.mdl", "models/p_crossbow.mdl"],
	"weapon_rpg": ["models/w_rpg.mdl", "models/v_rpg.mdl", "models/p_rpg.mdl"],
	"weapon_gauss": ["models/w_gauss.mdl", "models/v_gauss.mdl", "models/p_gauss.mdl"],
	"weapon_egon": ["models/w_egon.mdl", "models/v_egon.mdl", "models/p_egon.mdl"],
	"weapon_hornetgun": ["models/w_hgun.mdl", "models/v_hgun.mdl", "models/p_hgun.mdl"],
	"weapon_handgrenade": ["models/w_grenade.mdl", "models/v_grenade.mdl", "models/p_grenade.mdl"],
	"weapon_satchel": ["models/w_satchel.mdl", "models/v_satchel.mdl", "models/p_satchel.mdl"],
	"weapon_tripmine": ["models/v_tripmine.mdl", "models/p_tripmine.mdl"],
	"weapon_snark": ["models/w_sqknest.mdl", "models/v_squeak.mdl", "models/p_squeak.mdl"],
}

_KV = re.compile(r'"([^"]*)"\s+"([^"]*)"')


def parse_entities(text: str) -> list[dict[str, str]]:
	ents = []
	for block in re.findall(r"\{([^{}]*)\}", text):
		ents.append({k.lower(): v for k, v in _KV.findall(block)})
	return ents


@dataclass
class MapRefs:
	models: list[str] = field(default_factory=list)	# gamedir-relative paths, lowercase
	sprites: list[str] = field(default_factory=list)
	sounds: list[str] = field(default_factory=list)	# relative to sound/


def map_references(entities: list[dict[str, str]]) -> MapRefs:
	refs = MapRefs()
	seen: set[str] = set()

	def add(bucket: list[str], value: str) -> None:
		v = value.strip().replace("\\", "/").lstrip("*!")
		if v and v.lower() not in seen:
			seen.add(v.lower())
			bucket.append(v)

	for ent in entities:
		for model in CLASSNAME_MODELS.get(ent.get("classname", ""), []):
			add(refs.models, model)
		for key, value in ent.items():
			low = value.lower()
			if low.startswith("*") and key == "model":
				continue	# brush model
			if low.endswith(".mdl"):
				add(refs.models, value)
			elif low.endswith(".spr"):
				add(refs.sprites, value)
			elif low.endswith(".wav") and not value.startswith("!"):
				add(refs.sounds, value)
	return refs


# --- the report -----------------------------------------------------------------

OK, WARN, OVER, INFO = "ok", "warn", "over", "info"


@dataclass
class Line:
	pool: str
	used: int
	limit: int | None
	status: str
	detail: str = ""
	items: list[tuple[str, int]] = field(default_factory=list)	# biggest contributors


@dataclass
class MapBudget:
	map: str
	lines: list[Line] = field(default_factory=list)
	missing: list[str] = field(default_factory=list)	# referenced but not staged

	@property
	def status(self) -> str:
		states = {ln.status for ln in self.lines}
		return OVER if OVER in states else WARN if WARN in states or self.missing else OK

	def to_json(self) -> dict:
		d = asdict(self)
		d["status"] = self.status
		return d


def _ci_file(root: Path, rel: str) -> Path | None:
	"""Find rel under root, matching case-insensitively (staged trees are
	lowercase; entity strings often are not)."""
	direct = root / rel
	if direct.is_file():
		return direct
	cur = root
	for part in Path(rel).parts:
		if not cur.is_dir():
			return None
		match = next((c for c in cur.iterdir() if c.name.lower() == part.lower()), None)
		if match is None:
			return None
		cur = match
	return cur if cur.is_file() else None


def _status(used: int, limit: int, warn_at: float = 0.9) -> str:
	return OVER if used > limit else WARN if used > limit * warn_at else OK


def _top(items: list[tuple[str, int]], n: int = 5) -> list[tuple[str, int]]:
	return sorted(items, key=lambda t: -t[1])[:n]


def map_budget(gamedir: Path, bsp: Path, budgets: Budgets | None = None,
		include: list[str] | None = None) -> MapBudget:
	"""Check one map of a staged gamedir against the engine's windows."""
	b = budgets or Budgets()
	gamedir = Path(gamedir)
	stem = bsp.stem.lower()
	out = MapBudget(stem)
	data = bsp.read_bytes()

	# World textures: the MEM1 pool and the ARAM store.
	pack = _ci_file(gamedir, f"media/wt_{stem}.gct")
	if pack is None:
		out.lines.append(Line("World textures (MEM1 pool)", 0, b.wt_pool, WARN,
			f"media/wt_{stem}.gct not staged; the map falls back to WAD textures"))
	else:
		raw = pack.read_bytes()
		fit = simulate_world_pack(raw, b)
		entries = [(e.name, len(e.data)) for e in read_gct(raw)]
		if fit.split is None:
			st, msg = OVER, ("does not fit even at split 8; the rest streams from ARAM with "
				"placeholder binds. Run: gxbake fit " + pack.relative_to(gamedir).as_posix())
		elif fit.split < b.wt_split:
			st, msg = WARN, (f"fits only at split {fit.split} (textures over {fit.split}px draw from "
				f"low mips until streamed); at split {b.wt_split} it needs "
				f"{audit.human_bytes(fit.table_bytes + fit.resident_by_split[b.wt_split] + max(fit.max_chain, b.wt_min_hot))}")
		else:
			st, msg = OK, f"split {fit.split}, {audit.human_bytes(fit.hot)} hot ring"
		used = fit.table_bytes + fit.resident + max(fit.max_chain, b.wt_min_hot)
		out.lines.append(Line("World textures (MEM1 pool)", used, b.wt_pool, st,
			f"{fit.count} textures; " + msg, _top(entries)))
		st = _status(fit.data_bytes, b.wt_aram)
		if fit.count > b.wt_max_textures:
			st = OVER
		out.lines.append(Line("World textures (ARAM store)", fit.data_bytes, b.wt_aram, st,
			f"{fit.count} of {b.wt_max_textures} entries"))

	refs = map_references(parse_entities(goldsrc.bsp_entities_text(data)))
	for pattern in include or []:
		for p in sorted(gamedir.glob(pattern)):
			rel = p.relative_to(gamedir).as_posix()
			if rel.lower() not in {m.lower() for m in refs.models}:
				refs.models.append(rel)

	# Studio models: the lean mesh and skin companion the engine loads.
	items = []
	for model in refs.models:
		name = Path(model).stem.lower()
		lean = _ci_file(gamedir, f"gc_studio/{name}.mdl")
		skins = _ci_file(gamedir, f"gc_studio/{name}t.mdl")
		if lean is None:
			full = _ci_file(gamedir, model)
			if full is None:
				out.missing.append(model)
				continue
			items.append((f"{model} (not converted)", full.stat().st_size))
			continue
		items.append((name, lean.stat().st_size + (skins.stat().st_size if skins else 0)))
	used = sum(s for _n, s in items)
	out.lines.append(Line("Studio models (MEM1 arena)", used, b.studio_arena, _status(used, b.studio_arena),
		f"{len(items)} models" + ("; unconverted ones counted at full size" if any("not converted" in n for n, _ in items) else ""),
		_top(items)))

	# Sound: per-file cap, then the MEM1 cache, backed by the ARAM store.
	items, too_big = [], []
	for snd in refs.sounds:
		f = _ci_file(gamedir, f"sound/{snd}")
		if f is None:
			out.missing.append(f"sound/{snd}")
			continue
		pcm = sfx.wav_pcm_bytes(f)
		size = pcm if pcm is not None else f.stat().st_size
		items.append((snd, size))
		if size > b.sfx_max_file:
			too_big.append(snd)
	used = sum(s for _n, s in items)
	st = OVER if used > b.sfx_aram or too_big else WARN if used > b.sfx_cache else OK
	detail = f"{len(items)} sounds named by entities (PCM bytes)"
	if too_big:
		detail += (f"; {len(too_big)} over the {audit.human_bytes(b.sfx_max_file)} per-file cap never play: "
			f"{', '.join(too_big[:5])}. Run: gxbake sfx")
	if used > b.sfx_cache:
		detail += f"; more than the {audit.human_bytes(b.sfx_cache)} MEM1 cache, so they cycle through ARAM"
	out.lines.append(Line("Entity sounds", used, b.sfx_aram, st, detail, _top(items)))

	# Sprites (counted, no dedicated window yet).
	items = []
	for spr in refs.sprites:
		f = _ci_file(gamedir, spr)
		if f is None:
			out.missing.append(spr)
			continue
		items.append((spr, f.stat().st_size))
	if items:
		out.lines.append(Line("Sprites", sum(s for _n, s in items), None, INFO, f"{len(items)} sprites", _top(items)))

	# Lightmaps: the raw lump is an upper bound for the style-0 atlas.
	_off, light = goldsrc.bsp_lump(data, BSP_LUMP_LIGHTING)
	out.lines.append(Line("Lightmaps (atlas, estimate)", light, b.lightmap_atlas, _status(light, b.lightmap_atlas),
		"raw LUMP_LIGHTING, every style; the atlas holds style 0, so this is an upper bound"))

	# Sky and the BSP itself, for scale.
	sky = goldsrc.map_sky_name(data)
	if sky:
		sides = [_ci_file(gamedir, f"media/sky_{sky}{sfx}.cmp") for sfx in ("rt", "bk", "lf", "ft", "up", "dn")]
		size = sum(p.stat().st_size for p in sides if p)
		if not all(sides):
			out.missing.append(f"media/sky_{sky}*.cmp")
		out.lines.append(Line("Sky", size, None, INFO, sky))
	cam = _ci_file(gamedir, f"media/cam_{stem}.gcam")
	if cam is not None:
		from .camera import read_gcam

		raw = cam.read_bytes()
		try:
			clips = read_gcam(raw)
			out.lines.append(Line("Camera paths", len(raw), None, INFO, f"{len(clips)} clips",
				_top([(c.name, c.bytes) for c in clips])))
		except ValueError as exc:
			out.lines.append(Line("Camera paths", len(raw), None, WARN, f"media/cam_{stem}.gcam: {exc}"))
	out.lines.append(Line("BSP file", len(data), None, INFO, "lumps are read into the heap at map load"))
	return out


@dataclass
class GamedirBudget:
	gamedir: str
	total: int
	disc: int
	maps: list[MapBudget]
	largest: list[tuple[str, int]]

	@property
	def status(self) -> str:
		states = {m.status for m in self.maps}
		if self.total > self.disc:
			return OVER
		return OVER if OVER in states else WARN if WARN in states else OK

	def to_json(self) -> dict:
		return {"gamedir": self.gamedir, "status": self.status, "total": self.total, "disc": self.disc,
			"largest": self.largest, "maps": [m.to_json() for m in self.maps]}


def gamedir_budget(gamedir: Path, budgets: Budgets | None = None, maps: list[str] | None = None,
		include: list[str] | None = None) -> GamedirBudget:
	b = budgets or Budgets()
	gamedir = Path(gamedir)
	files = []
	for root, _dirs, names in os.walk(gamedir):
		for n in names:
			p = Path(root) / n
			files.append((p.relative_to(gamedir).as_posix(), p.stat().st_size))
	bsps = sorted((gamedir / "maps").glob("*.bsp")) if (gamedir / "maps").is_dir() else []
	if maps:
		wanted = {m.lower().removesuffix(".bsp") for m in maps}
		bsps = [p for p in bsps if p.stem.lower() in wanted]
	reports = [map_budget(gamedir, p, b, include) for p in bsps]
	return GamedirBudget(str(gamedir), sum(s for _n, s in files), b.disc, reports, _top(files, 10))


def format_report(rep: GamedirBudget, verbose: bool = False) -> str:
	hb = audit.human_bytes
	mark = {OK: "  ok", WARN: "WARN", OVER: "OVER", INFO: "    "}
	out = [f"{rep.gamedir}: {hb(rep.total)} staged, {rep.total / rep.disc:.1%} of a GameCube disc"]
	for m in rep.maps:
		out.append("")
		out.append(f"== {m.map} [{m.status.upper()}]")
		for ln in m.lines:
			lim = f" / {hb(ln.limit)} ({ln.used / ln.limit:.0%})" if ln.limit else ""
			out.append(f"  {mark[ln.status]}  {ln.pool:<28} {hb(ln.used):>10}{lim}")
			if ln.detail:
				out.append(f"        {ln.detail}")
			if verbose or ln.status in (WARN, OVER):
				for name, size in ln.items:
					out.append(f"          {hb(size):>10}  {name}")
		if m.missing:
			out.append(f"  WARN  {len(m.missing)} referenced but not staged: {', '.join(m.missing[:8])}"
				+ (" …" if len(m.missing) > 8 else ""))
	if not rep.maps:
		out.append("no maps/*.bsp found")
	return "\n".join(out)


def fit_pack_bytes(raw: bytes, budgets: Budgets | None = None, max_edge: int | None = None) -> tuple[bytes, FitResult]:
	"""fit_world_pack() on a serialized .gct; returns the new pack and the result."""
	res = fit_world_pack(read_gct(raw), budgets, max_edge)
	return write_gct(res.entries), res
