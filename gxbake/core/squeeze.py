"""Smarter packing for the GameCube port: measure it, and apply what ships today.

`gxbake squeeze <staged gamedir>` runs every packing technique below on the
real files and reports bytes before and after, per memory window. Each
technique is marked:

* **ships** — the port reads the result unchanged, so `squeeze -o` writes it.
* **port** — needs a loader change in the port (separate repo). GxBake can
  already write the data (`--emit`), and docs/packing.md says what the port
  must add.

Techniques:

=====================  =====  ================================================
detail mip trim         ships  drop a world texture's top mip when it adds less
                               than `min_db` of detail over the bilinear
                               upsample of the next level (flat and blurry
                               textures shrink 4x, sharp ones are untouched)
optimal anim runs       ships  re-encode studio animation with the smallest
                               mstudioanimvalue_t chunking (lossless)
anim deadband           ships  hold animation values within ±tol units so
                               runs get long (bounded error, opt-in)
indexed skins           port   studio skins as CI4/CI8 + 16-bit TLUT (or CMPR,
                               I4/I8 for grey) instead of RGB565/RGB5A3
DSP ADPCM sound         port   sound effects and .gcpcm music as DSP ADPCM,
                               the GameCube's hardware sound format (3.5x)
s16 vertices            port   studio positions as s16 and normals as s8,
                               dequantised for free by Gekko's psq_l loads
u8 texture coords       port   strip/fan commands whose S/T all fit 0..255
                               sent as u8 in a second GX vertex format
                               (lossless: same step; from Lost Coast props)
CMPR lightmaps          port   static lightmap pages as CMPR, not RGB565 (4x)
Yaz0 on disc            port   LZ the files the port reads whole
=====================  =====  ================================================
"""

from __future__ import annotations

import fnmatch
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audit, dspadpcm, gxtex, yaz0
from .budget import Budgets, _entries_fit, drop_top_mip
from .gct import GctEntry, read_gct, write_gct

KiB = 1024


@dataclass
class Saving:
	technique: str
	ships: bool
	window: str	# which memory the bytes come out of
	path: str
	before: int
	after: int
	quality: str = ""
	note: str = ""
	db: float | None = None	# worst PSNR/SNR when the technique is lossy and measured

	@property
	def saved(self) -> int:
		return self.before - self.after


# Technique keys for SqueezeOptions.techniques / --only / --skip.
TECHNIQUES = {
	"trim": "detail mip trim (world textures)",
	"anim": "optimal anim runs and deadband (studio animation)",
	"skins": "indexed skins (.gsk)",
	"adpcm": "DSP ADPCM sound and music (.dsp)",
	"verts": "s16 vertices (estimate)",
	"lightmaps": "CMPR lightmaps (estimate)",
	"yaz0": "Yaz0 (.yaz0)",
}


@dataclass
class SqueezeOptions:
	min_db: float = 40.0	# detail trim: drop a top mip whose detail is under this
	anim_tol: int = 0	# deadband in raw animation units; 0 = lossless only
	skin_db: float = 40.0	# lossy skin formats must stay this close to today's RGB565
	adpcm_preview_s: float = 20.0	# seconds of long streams to encode for the SNR figure
	yaz0_sample: int = 512 * KiB	# compress this much of a big file and extrapolate
	yaz0: bool = True
	techniques: set[str] = field(default_factory=lambda: set(TECHNIQUES))

	def on(self, key: str) -> bool:
		return key in self.techniques and (key != "yaz0" or self.yaz0)


# --- world textures: detail mip trim ----------------------------------------------

def upsample2(img: "np.ndarray") -> "np.ndarray":
	"""2x bilinear with texel centres and wrap-around (GX_REPEAT), the way
	Flipper magnifies the next mip level."""
	a = img.astype(np.float32)

	def axis(x, ax):
		prev, nxt = np.roll(x, 1, ax), np.roll(x, -1, ax)
		even, odd = 0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * nxt
		out = np.stack((even, odd), ax + 1)
		shape = list(x.shape)
		shape[ax] *= 2
		return out.reshape(shape)

	return axis(axis(a, 0), 1)


def top_mip_detail(e: GctEntry, max_mask_change: float = 0.002) -> float:
	"""PSNR (dB) of level 0 against level 1 magnified: how much level 0 adds.
	For 1-bit alpha textures the cut-out must also survive: if more than
	`max_mask_change` of the texels flip between see-through and solid, the
	answer is 0 (keep the mip)."""
	l0 = e.decode(0).astype(np.float32)
	l1 = e.decode(1).astype(np.float32)
	up = upsample2(l1)[:l0.shape[0], :l0.shape[1]]
	if up.shape != l0.shape:	# a side already at the 8-texel floor
		return 0.0
	if not e.alpha:
		return gxtex.psnr(up[..., :3], l0[..., :3])
	m0, m1 = l0[..., 3] > 127, up[..., 3] > 127
	if float((m0 != m1).mean()) > max_mask_change:
		return 0.0
	vis = m0 & m1
	return gxtex.psnr(up[..., :3], l0[..., :3], vis) if vis.any() else float("inf")


@dataclass
class TrimResult:
	entries: list[GctEntry]
	trimmed: dict[str, tuple[int, int, int, int, float]]	# name → (w, h, new w, new h, worst dB)


def trim_flat_mips(entries: list[GctEntry], min_db: float = 40.0, min_edge: int = 8) -> TrimResult:
	"""Drop top mips that a bilinear upsample of the next level reproduces
	within `min_db`. Repeats, so a solid-colour 256² texture ends at 8².
	Alpha-tested textures are left alone (their cut-out edge is the detail)."""
	out, trimmed = [], {}
	for e in entries:
		w, h, worst = e.width, e.height, float("inf")
		while e.levels > 1 and min(e.width, e.height) > min_edge:
			db = top_mip_detail(e)
			if db < min_db:
				break
			worst = min(worst, db)
			e = drop_top_mip(e)
		if (e.width, e.height) != (w, h):
			trimmed[e.name] = (w, h, e.width, e.height, worst)
		out.append(e)
	return TrimResult(out, trimmed)


# --- studio skins: indexed formats -----------------------------------------------

def port_chromakey(pal: "np.ndarray", hands: bool = False) -> "np.ndarray":
	"""Palette entries the port keys out of view-model skins
	(R_GXSkinBuildTiled in ref/gx/r_gx_skins.c). Applying the rule to the
	TLUT is exact: it depends only on the colour."""
	r, g, b = (pal[:, i].astype(np.int32) for i in range(3))
	key = (r > 200) & (b > 180) & (g + 40 < r) & (g + 40 < b)
	key |= (r < 40) & (g < 40) & (b > 200)
	key |= (r > 220) & (g < 40) & (b > 220)
	if hands:
		key |= (r < 14) & (g < 14) & (b < 14) & (r + g + b < 28)
	return key


@dataclass
class SkinChoice:
	name: str
	width: int
	height: int
	choice: gxtex.Choice


def studio_skins(raw: bytes, *, min_db: float = 40.0, viewmodel: bool = False) -> list[SkinChoice]:
	"""Pick a GX format for every skin in a studio model (or *t.mdl companion)."""
	out: list[SkinChoice] = []
	if len(raw) < 244 or raw[:4] not in (b"IDST", b"IDSQ"):
		return out
	ntex, tex_off = struct.unpack_from("<ii", raw, 180)
	for i in range(max(0, ntex)):
		o = tex_off + i * 80
		if o + 80 > len(raw):
			break
		name = raw[o:o + 64].split(b"\0")[0].decode("latin-1")
		flags, w, h, idx = struct.unpack_from("<4i", raw, o + 64)
		if w <= 0 or h <= 0 or idx <= 0 or idx + w * h + 768 > len(raw):
			continue
		index = np.frombuffer(raw, np.uint8, w * h, idx).reshape(h, w)
		pal = np.frombuffer(raw, np.uint8, 768, idx + w * h).reshape(256, 3).copy()
		masked = bool(flags & 0x40)
		if viewmodel:
			key = port_chromakey(pal, hands="hand" in name.lower())
			if key[np.unique(index)].any():
				# fold every keyed colour onto index 255, the masked index
				index = np.where(key[index], 255, index).astype(np.uint8)
				masked = True
		out.append(SkinChoice(name, w, h, gxtex.choose_indexed(index, pal, masked, min_db=min_db)))
	return out


GCSK_MAGIC = b"GCSK"
GCSK_ENTRY = struct.Struct(">HBBHHHHIIII8x")	# skin, fmt, tlut fmt, w, h, tlut entries, flags, tex off, tex bytes, tlut off, tlut bytes


def write_skin_pack(skins: list[SkinChoice]) -> bytes:
	"""GCSK: pre-tiled GX texels and TLUTs for a model's skins, in skin order.

		"GCSK" u32 count u32 version (1) u32 0
		count × { u16 skin; u8 GXTexFmt; u8 GXTlutFmt; u16 w, h; u16 tlut entries;
		          u16 flags (1 = 1-bit alpha); u32 tex off, tex bytes, tlut off, tlut bytes; 8 × 0 }
		then texels and TLUTs, each 32-byte aligned (GX_InitTexObj, GX_LoadTlut)
	"""
	head = GCSK_MAGIC + struct.pack(">III", len(skins), 1, 0)
	table_end = len(head) + len(skins) * GCSK_ENTRY.size
	pos = table_end + (-table_end & 31)
	rows, blobs = [], []
	for i, s in enumerate(skins):
		t = s.choice.tex
		tex_off = pos
		pos += len(t.data) + (-len(t.data) & 31)
		tlut_off = pos if t.tlut else 0
		pos += len(t.tlut) + (-len(t.tlut) & 31)
		alpha = t.fmt in ("RGB5A3", "IA4", "IA8", "CMPR") or t.tlut_fmt == gxtex.GX_TL_RGB5A3 and bool(t.tlut)
		rows.append(GCSK_ENTRY.pack(i, gxtex.FORMATS[t.fmt][0], t.tlut_fmt, t.width, t.height, len(t.tlut) // 2,
			int(alpha), tex_off, len(t.data), tlut_off, len(t.tlut)))
		blobs += [t.data + bytes(-len(t.data) & 31), t.tlut + bytes(-len(t.tlut) & 31)]
	body = head + b"".join(rows)
	return body + bytes(-len(body) & 31) + b"".join(blobs)


def read_skin_pack(blob: bytes) -> list[gxtex.GxTexture]:
	if blob[:4] != GCSK_MAGIC:
		raise ValueError("not a GCSK skin pack")
	count = struct.unpack_from(">I", blob, 4)[0]
	out = []
	for i in range(count):
		_s, fmt, tl, w, h, _n, _fl, to, tb, lo, lb = GCSK_ENTRY.unpack_from(blob, 16 + i * GCSK_ENTRY.size)
		out.append(gxtex.GxTexture(gxtex.FORMAT_NAMES[fmt], w, h, blob[to:to + tb], blob[lo:lo + lb] if lb else b"", tl))
	return out


# --- studio meshes --------------------------------------------------------------

def studio_vertex_counts(raw: bytes) -> tuple[int, int]:
	"""(vertices, normals) over every body part and model."""
	nv = nn = 0
	nbody, body = struct.unpack_from("<ii", raw, 204)
	for bp in range(max(0, nbody)):
		nmodels, model_off = struct.unpack_from("<i", raw, body + bp * 76 + 64)[0], struct.unpack_from("<i", raw, body + bp * 76 + 72)[0]
		for m in range(max(0, nmodels)):
			mb = model_off + m * 112
			nv += struct.unpack_from("<i", raw, mb + 80)[0]
			nn += struct.unpack_from("<i", raw, mb + 92)[0]
	return nv, nn


def studio_texcoord_counts(raw: bytes) -> tuple[int, int]:
	"""(vertices in the triangle commands, vertices of commands whose S and T
	all fit 0..255). GoldSrc stores S/T as whole texels in s16, so a command
	that stays inside 0..255 can go to the GPU as u8 in a second GX vertex
	format with no loss, 2 bytes a vertex less; Lost Coast's props took 15%
	off their display lists this way."""
	total = fit = 0
	nbody, body = struct.unpack_from("<ii", raw, 204)
	for bp in range(max(0, nbody)):
		nmodels, model_off = struct.unpack_from("<i", raw, body + bp * 76 + 64)[0], struct.unpack_from("<i", raw, body + bp * 76 + 72)[0]
		for m in range(max(0, nmodels)):
			mb = model_off + m * 112
			nmesh, mesh_off = struct.unpack_from("<ii", raw, mb + 72)
			for k in range(max(0, nmesh)):
				_ntris, tri = struct.unpack_from("<ii", raw, mesh_off + k * 20)
				while 0 <= tri < len(raw) - 2:
					n = abs(struct.unpack_from("<h", raw, tri)[0])
					tri += 2
					if not n:
						break
					st = struct.unpack_from("<%dh" % (4 * n), raw, tri)
					tri += 8 * n
					total += n
					if all(0 <= v <= 255 for i, v in enumerate(st) if i % 4 >= 2):
						fit += n
	return total, fit


# --- sound -------------------------------------------------------------------

def _pcm16(snd) -> "np.ndarray":
	return np.clip(np.round(snd.samples * 32767), -32768, 32767).astype(np.int16)


def adpcm_for(pcm: "np.ndarray", rate: int, preview_s: float | None = None) -> tuple[int, float]:
	"""(ADPCM bytes for every channel, worst channel SNR in dB) of frames × channels s16."""
	frames, ch = pcm.shape
	size = ch * dspadpcm.adpcm_bytes(frames)
	n = frames if preview_s is None else min(frames, int(rate * preview_s))
	worst = float("inf")
	for c in range(ch):
		x = pcm[:n, c]
		if not np.any(x):
			continue
		st = dspadpcm.encode(x, rate)
		worst = min(worst, dspadpcm.snr_db(x, st.decode()))
	return size, worst


def read_gcpcm(data: bytes) -> tuple[int, "np.ndarray"]:
	rate, ch, width, size = struct.unpack_from("<IHHI", data, 4)
	pcm = np.frombuffer(data, ">i2", size // 2, 16).reshape(-1, ch)
	return rate, pcm


# --- the report --------------------------------------------------------------------

def _yaz0_estimate(data: bytes, sample: int) -> int:
	if len(data) <= sample:
		return len(yaz0.compress(data))
	step = len(data) // 4
	parts = [data[k * step:k * step + sample // 4] for k in range(4)]
	ratio = sum(len(yaz0.compress(p)) for p in parts) / sum(map(len, parts))
	return int(len(data) * ratio)


def _fmt_db(db: float) -> str:
	return "lossless" if db == float("inf") else f"{db:.1f} dB"


def analyse_file(path: Path, rel: str, opt: SqueezeOptions) -> list[Saving]:
	"""Every technique that applies to one staged file."""
	out: list[Saving] = []
	data = path.read_bytes()
	low = rel.lower()
	ext = path.suffix.lower()
	if ext == ".gct" and data[:4] == b"GCWT" and opt.on("trim"):
		entries = read_gct(data)
		res = trim_flat_mips(entries, opt.min_db)
		new = write_gct(res.entries)
		b0, b1 = _entries_fit(entries, Budgets()), _entries_fit(res.entries, Budgets())
		worst = min((t[4] for t in res.trimmed.values()), default=float("inf"))
		out.append(Saving("detail mip trim", True, "ARAM wtstore + MEM1 pool", rel, len(data), len(new),
			f"≥{opt.min_db:.0f} dB" if res.trimmed else "",
			f"{len(res.trimmed)}/{len(entries)} textures; MEM1 low mips {audit.human_bytes(b0.resident)} → "
			f"{audit.human_bytes(b1.resident)}", worst if res.trimmed else None))
	elif ext == ".mdl" and data[:4] == b"IDST":
		from .studio_slim import slim_studio, studio_breakdown

		is_tex = struct.unpack_from("<i", data, 204)[0] == 0 and struct.unpack_from("<i", data, 180)[0] > 0
		if not is_tex and opt.on("anim"):
			try:
				lossless = slim_studio(data)
				anim = studio_breakdown(data).get("animation", 0)
				out.append(Saving("optimal anim runs", True, "MEM1 studio arena", rel, len(data), lossless.after,
					"lossless", f"animation {audit.human_bytes(anim)}"))
				if opt.anim_tol > 0:
					tol = slim_studio(data, tolerance=opt.anim_tol)
					out.append(Saving("anim deadband", True, "MEM1 studio arena", rel, lossless.after, tol.after,
						f"±{opt.anim_tol} units"))
			except (ValueError, struct.error) as exc:
				out.append(Saving("optimal anim runs", True, "MEM1 studio arena", rel, len(data), len(data), "", str(exc)))
		nv, nn = studio_vertex_counts(data) if not is_tex and opt.on("verts") else (0, 0)
		if nv:
			out.append(Saving("s16 vertices", False, "MEM1 studio arena", rel, nv * 12 + nn * 12, nv * 6 + nn * 3,
				"≤1/65536 of the model's extent", f"{nv} positions, {nn} normals"))
			try:
				tv, fit = studio_texcoord_counts(data)
			except struct.error:
				tv = fit = 0
			if tv:
				out.append(Saving("u8 texture coords", False, "MEM1 studio arena", rel, tv * 4, tv * 4 - fit * 2,
					"lossless", f"{fit} of {tv} strip vertices fit u8"))
		skins = studio_skins(data, min_db=opt.skin_db, viewmodel=Path(rel).name.lower().startswith("v_")) \
			if opt.on("skins") else []
		if skins:
			before = sum(s.choice.baseline for s in skins)
			after = sum(s.choice.tex.size for s in skins)
			fmts = sorted({s.choice.tex.fmt for s in skins})
			worst = min(s.choice.psnr for s in skins)
			out.append(Saving("indexed skins", False, "MEM1 skin window / ARAM", rel, before, after, _fmt_db(worst),
				f"{len(skins)} skins as {'/'.join(fmts)} (today RGB565/RGB5A3)", worst))
	elif ext == ".wav" and opt.on("adpcm"):
		from . import sfx

		try:
			snd = sfx.read_wav(data)
		except ValueError:
			return out
		pcm = _pcm16(snd)
		before = snd.frames * snd.channels * snd.width
		size, db = adpcm_for(pcm, snd.rate)
		out.append(Saving("DSP ADPCM sound", False, "MEM1 sfx cache + ARAM sfx", rel, before, size,
			_fmt_db(db), f"{snd.rate} Hz {snd.width * 8}-bit", db))
	elif ext == ".gcpcm" and data[:4] == b"GCPA" and opt.on("adpcm"):
		rate, pcm = read_gcpcm(data)
		size, db = adpcm_for(pcm, rate, opt.adpcm_preview_s)
		out.append(Saving("DSP ADPCM sound", False, "disc stream bandwidth", rel, pcm.nbytes, size, _fmt_db(db),
			f"{len(pcm) / rate:.0f} s at {rate} Hz; SNR of the first {opt.adpcm_preview_s:.0f} s", db))
	elif ext == ".bsp" and len(data) > 124 and opt.on("lightmaps"):
		ofs, ln = struct.unpack_from("<ii", data, 4 + 8 * 8)
		if struct.unpack_from("<i", data, 0)[0] in (29, 30) and 0 < ln and ofs + ln <= len(data):
			rgb565 = ln * 2 // 3	# 24-bit samples → 16-bit atlas texels
			out.append(Saving("CMPR lightmaps", False, "MEM1 lightmap atlas", rel, rgb565, rgb565 // 4,
				"lossy (DXT1 on smooth gradients)", "estimate from the lighting lump; styled pages stay RGB565"))
	if opt.on("yaz0") and ext in (".gct", ".mdl", ".bsp", ".spr", ".cmp") and len(data) > 4 * KiB:
		out.append(Saving("Yaz0 on disc", False, "disc + ARAM cache", rel, len(data), _yaz0_estimate(data, opt.yaz0_sample),
			"lossless"))
	return out


def staged_files(root: Path) -> list[tuple[Path, str]]:
	if root.is_file():
		return [(root, root.name)]
	return [(p, p.relative_to(root).as_posix()) for p in sorted(root.rglob("*")) if p.is_file()]


@dataclass
class Report:
	rows: list[Saving] = field(default_factory=list)

	def by_technique(self) -> dict[str, list[Saving]]:
		out: dict[str, list[Saving]] = {}
		for r in self.rows:
			out.setdefault(r.technique, []).append(r)
		return out

	def to_json(self) -> list[dict]:
		return [asdict(r) | {"saved": r.saved} for r in self.rows]


def squeeze(root: Path, opt: SqueezeOptions | None = None, match: list[str] | None = None) -> Report:
	opt = opt or SqueezeOptions()
	rep = Report()
	for path, rel in staged_files(root):
		if not _matches(rel, match):
			continue
		rep.rows += analyse_file(path, rel, opt)
	return rep


def format_report(rep: Report, verbose: bool = False) -> str:
	hb = audit.human_bytes
	lines = [f"{'technique':<20} {'needs':<6} {'files':>5} {'before':>11} {'after':>11} {'saved':>7}  quality / window"]
	for tech, rows in rep.by_technique().items():
		b, a = sum(r.before for r in rows), sum(r.after for r in rows)
		dbs = [r.db for r in rows if r.db is not None]
		quals = sorted({r.quality for r in rows if r.quality})
		if dbs and len(quals) > 1:
			q = f"median {_fmt_db(float(np.median(dbs)))}, worst {_fmt_db(min(dbs))}"
		else:
			q = quals[0] if quals else ""
		pct = f"{100 * (b - a) / b:.0f} %" if b else "-"
		lines.append(f"{tech:<20} {'-' if rows[0].ships else 'port':<6} {len(rows):>5} {hb(b):>11} {hb(a):>11} {pct:>7}  "
			f"{q}{'; ' if q else ''}{rows[0].window}")
		if verbose:
			for r in sorted(rows, key=lambda r: -r.saved):
				lines.append(f"    {r.path:<40} {hb(r.before):>11} {hb(r.after):>11}  {r.quality}  {r.note}")
	lines.append("needs: '-' = the port reads it today (squeeze -o writes it); 'port' = needs a loader change "
		"(docs/packing.md), --emit writes the data")
	return "\n".join(lines)


# --- writing -------------------------------------------------------------------

def _matches(rel: str, match: list[str] | None) -> bool:
	return not match or any(fnmatch.fnmatch(rel.lower(), m.lower()) for m in match)


def apply(root: Path, out: Path, opt: SqueezeOptions | None = None, emit: bool = False,
		match: list[str] | None = None) -> list[str]:
	"""Write what ships today into `out` (same relative paths), and with
	`emit` the port-side formats next to them (.dsp, .gsk, .yaz0)."""
	opt = opt or SqueezeOptions()
	written = []
	for path, rel in staged_files(root):
		if not _matches(rel, match):
			continue
		data = path.read_bytes()
		ext = path.suffix.lower()
		new = None
		if ext == ".gct" and data[:4] == b"GCWT" and opt.on("trim"):
			res = trim_flat_mips(read_gct(data), opt.min_db)
			if res.trimmed:
				new = write_gct(res.entries)
		elif ext == ".mdl" and data[:4] == b"IDST" and struct.unpack_from("<i", data, 204)[0] > 0 and opt.on("anim"):
			from .studio_slim import slim_studio

			try:
				r = slim_studio(data, tolerance=opt.anim_tol)
				new = r.data if r.after < len(data) else None
			except ValueError:
				new = None
		if new is not None:
			dest = out / rel
			dest.parent.mkdir(parents=True, exist_ok=True)
			dest.write_bytes(new)
			written.append(rel)
		if not emit:
			continue
		extra: dict[str, bytes] = {}
		if ext == ".mdl" and data[:4] == b"IDST" and opt.on("skins"):
			skins = studio_skins(data, min_db=opt.skin_db, viewmodel=Path(rel).name.lower().startswith("v_"))
			if skins:
				extra[str(Path(rel).with_suffix(".gsk"))] = write_skin_pack(skins)
		elif ext == ".wav" and opt.on("adpcm"):
			from . import sfx

			try:
				snd = sfx.read_wav(data)
			except ValueError:
				snd = None
			if snd is not None:
				pcm = _pcm16(snd)
				for c in range(snd.channels):
					st = dspadpcm.encode(pcm[:, c], snd.rate, snd.loop_start,
						None if snd.loop_end is None else snd.loop_end - 1)
					suffix = ".dsp" if snd.channels == 1 else f".{'LR'[c] if snd.channels == 2 else c}.dsp"
					extra[str(Path(rel).with_suffix(suffix))] = st.to_dsp()
		elif ext == ".gcpcm" and data[:4] == b"GCPA" and opt.on("adpcm"):
			rate, pcm = read_gcpcm(data)
			for c in range(pcm.shape[1]):
				st = dspadpcm.encode(pcm[:, c], rate)
				extra[str(Path(rel).with_suffix(f".{'LR'[c] if pcm.shape[1] == 2 else c}.dsp"))] = st.to_dsp()
		if opt.on("yaz0") and ext in (".gct", ".mdl", ".bsp", ".spr", ".cmp"):
			extra[rel + ".yaz0"] = yaz0.compress(new if new is not None else data)
		for r, blob in extra.items():
			dest = out / r
			dest.parent.mkdir(parents=True, exist_ok=True)
			dest.write_bytes(blob)
			written.append(r)
	return written
