"""Pick GX texture formats and sizes automatically, alone or under a byte budget.

`choose()` encodes one image in every GX format that suits its content
(grey or colour, opaque, cut-out or graded alpha) and keeps the smallest
whose picture stays above `min_db` PSNR against the 8-bit source; RGBA8 is
the lossless fallback. `fit()` does the same for a whole set of textures
that must share a byte budget (a texture pool, a memory window): it starts
every texture at its `choose()` pick and then, while the set is over
budget, takes whichever single step (a smaller format or half the size)
costs the least picture per byte saved, until it fits.

Errors are measured at the texture's full target size (a halved texture is
decoded, scaled back up bilinearly, then compared), so trading resolution
against format is a fair fight. Mip chains are counted in the bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gxtex
from .tpl import GX_CLAMP, GX_REPEAT, TplImage

ALL_FORMATS = ("I4", "I8", "IA4", "IA8", "CI4", "CI8", "CMPR", "RGB565", "RGB5A3", "RGBA8")

# Named format sets for common roles. "auto" lets content decide.
ROLES: dict[str, tuple[str, ...]] = {
	"auto": ALL_FORMATS,
	"color": ("CI4", "CI8", "CMPR", "RGB565", "RGB5A3", "RGBA8"),
	"world": ("CMPR",),	# what the port's GCWT world packs hold
	"ui": ("I4", "I8", "IA4", "IA8", "CI4", "CI8", "RGB5A3", "RGBA8"),	# no CMPR smear on text and edges
	"mask": ("I4", "I8"),	# spec/gloss/AO/height: one channel
	"normal": ("IA8", "RGB565", "RGBA8"),	# X/Y kept apart; IA8 holds X in I, Y in A
	"lightmap": ("CMPR", "RGB565", "I4", "I8"),
}


@dataclass
class TexJob:
	"""One texture to place: the source image and what it may become."""
	name: str
	rgba: "np.ndarray"	# H×W×4 uint8, any size
	formats: tuple[str, ...] = ALL_FORMATS
	min_db: float = 38.0
	max_edge: int = 256
	min_edge: int = 8
	mipmaps: bool = True
	pow2: bool = True
	wrap: str = "repeat"	# repeat | clamp
	weight: float = 1.0	# how much this texture's error counts against the others
	fixed: str = ""	# force one format (still resized to fit)
	meta: dict = field(default_factory=dict)


@dataclass
class Option:
	fmt: str
	width: int
	height: int
	bytes: int
	mse: float	# against the source at full target size

	@property
	def db(self) -> float:
		return float("inf") if self.mse <= 0 else 10 * np.log10(255 ** 2 / self.mse)


@dataclass
class Placed:
	job: TexJob
	option: Option
	image: TplImage
	tried: list[Option]
	under: bool	# below the job's min_db (only when a budget forced it)


# -- content ----------------------------------------------------------------------------------------


def analyse(rgba: "np.ndarray") -> dict:
	a = rgba[..., 3]
	alpha = "opaque" if a.min() >= 250 else "binary" if bool(((a < 8) | (a > 247)).all()) else "graded"
	c = rgba[..., :3].astype(np.int16)
	vis = a >= 8 if alpha != "opaque" else np.ones(a.shape, bool)
	grey = bool(vis.any()) and int(np.abs(c[..., 0] - c[..., 1])[vis].max()) <= 3 \
		and int(np.abs(c[..., 1] - c[..., 2])[vis].max()) <= 3
	return {"alpha": alpha, "grey": grey or not vis.any()}


def candidates(rgba: "np.ndarray", allowed: tuple[str, ...] = ALL_FORMATS) -> list[str]:
	"""The formats worth trying for this picture, from the allowed set."""
	info = analyse(rgba)
	alpha, grey = info["alpha"], info["grey"]
	if grey:
		pick = {"opaque": ("I4", "I8", "CI4", "CMPR", "IA8"),
			"binary": ("IA4", "IA8", "CI4", "CI8", "CMPR", "RGB5A3"),
			"graded": ("IA4", "IA8", "CI4", "CI8", "RGB5A3", "RGBA8")}[alpha]
	else:
		pick = {"opaque": ("CMPR", "CI4", "CI8", "RGB565", "RGBA8"),
			"binary": ("CMPR", "CI4", "CI8", "RGB5A3", "RGBA8"),
			"graded": ("CI4", "CI8", "RGB5A3", "RGBA8")}[alpha]
	out = [f for f in pick if f in allowed]
	if not out:	# the role asked for formats content alone would not pick (e.g. a mask of a colour image)
		out = list(allowed)
	return out


# -- resizing and mips ------------------------------------------------------------------------------


def _resize(rgba: "np.ndarray", w: int, h: int, smooth: bool = True) -> "np.ndarray":
	from PIL import Image

	if rgba.shape[1] == w and rgba.shape[0] == h:
		return rgba
	r = Image.Resampling.LANCZOS if smooth else Image.Resampling.BILINEAR
	return np.asarray(Image.fromarray(np.ascontiguousarray(rgba), "RGBA").resize((w, h), r))


def target_size(w: int, h: int, max_edge: int, pow2: bool = True, min_edge: int = 8) -> tuple[int, int]:
	if pow2:
		tw = 1 << max(0, (w - 1).bit_length())
		th = 1 << max(0, (h - 1).bit_length())
		# round to the nearer power of two rather than always up: 300 → 256, 400 → 512
		if tw > w and tw - w > w - tw // 2:
			tw //= 2
		if th > h and th - h > h - th // 2:
			th //= 2
	else:
		tw, th = w, h
	while max(tw, th) > max_edge:
		tw, th = max(1, tw // 2), max(1, th // 2)
	return max(min(min_edge, w), tw, 1), max(min(min_edge, h), th, 1)


def mip_sizes(w: int, h: int, mipmaps: bool, min_edge: int = 8) -> list[tuple[int, int]]:
	out = [(w, h)]
	while mipmaps and min(w, h) > max(1, min_edge) and w > 1 and h > 1:
		w, h = w // 2, h // 2
		out.append((w, h))
	return out


def chain_bytes(w: int, h: int, fmt: str, mipmaps: bool, min_edge: int = 8, tlut_entries: int = 0) -> int:
	n = sum(gxtex.tex_size(lw, lh, fmt) for lw, lh in mip_sizes(w, h, mipmaps, min_edge))
	if fmt in ("CI4", "CI8"):
		n += 2 * (tlut_entries or (16 if fmt == "CI4" else 256))
	return n


def mip_chain(rgba: "np.ndarray", sizes: list[tuple[int, int]]) -> list["np.ndarray"]:
	"""Each level box-filtered from the one above with premultiplied alpha, so a
	cut-out's transparent texels never bleed colour into its edge."""
	levels = [rgba]
	f = rgba.astype(np.float32)
	for w, h in sizes[1:]:
		ph, pw = f.shape[:2]
		if (pw, ph) == (w * 2, h * 2):
			a = f[..., 3:4] / 255
			pre = np.concatenate((f[..., :3] * a, f[..., 3:4]), -1)
			pre = pre.reshape(h, 2, w, 2, 4).mean((1, 3))
			al = pre[..., 3:4] / 255
			rgb = np.where(al > 0, pre[..., :3] / np.maximum(al, 1e-6), 0)
			f = np.concatenate((rgb, pre[..., 3:4]), -1)
		else:
			f = _resize(np.clip(f + 0.5, 0, 255).astype(np.uint8), w, h).astype(np.float32)
		levels.append(np.clip(f + 0.5, 0, 255).astype(np.uint8))
	return levels


# -- encoding ---------------------------------------------------------------------------------------


def _quantize(rgba: "np.ndarray", colours: int) -> tuple["np.ndarray", "np.ndarray"]:
	"""Index + palette (N×4) with Pillow: median cut for opaque, octree with alpha."""
	from PIL import Image

	alpha = bool((rgba[..., 3] < 250).any())
	if alpha:
		q = Image.fromarray(np.ascontiguousarray(rgba), "RGBA").quantize(colours, method=Image.Quantize.FASTOCTREE)
		pal = np.array(q.getpalette(rawmode="RGBA") or [], np.uint8).reshape(-1, 4)
	else:
		q = Image.fromarray(np.ascontiguousarray(rgba[..., :3])).quantize(colours, method=Image.Quantize.MEDIANCUT)
		pal = np.array(q.getpalette() or [], np.uint8).reshape(-1, 3)
		pal = np.concatenate((pal, np.full((len(pal), 1), 255, np.uint8)), 1)
	idx = np.asarray(q, np.uint8)
	n = max(1, int(idx.max()) + 1)
	return idx, pal[:n]


def _map_to(rgba: "np.ndarray", pal: "np.ndarray") -> "np.ndarray":
	"""Nearest palette entry for each texel (lower mip levels reuse the top's palette)."""
	px = rgba.reshape(-1, 4).astype(np.int32)
	best = np.zeros(len(px), np.uint8)
	bd = np.full(len(px), np.iinfo(np.int64).max, np.int64)
	for i, c in enumerate(pal.astype(np.int32)):
		d = ((px - c) ** 2).sum(1)
		m = d < bd
		bd[m], best[m] = d[m], i
	return best.reshape(rgba.shape[:2])


def encode_levels(levels: list["np.ndarray"], fmt: str) -> list[gxtex.GxTexture]:
	"""Every level in one format; colour-indexed levels share the top level's palette."""
	if fmt not in ("CI4", "CI8"):
		return [gxtex.encode(lv, fmt) for lv in levels]
	top = levels[0]
	idx, pal = _quantize(top, 16 if fmt == "CI4" else 256)
	alpha = bool((pal[:, 3] < 255).any())
	tl = gxtex.GX_TL_RGB5A3 if alpha else gxtex.GX_TL_RGB565
	out = [gxtex.encode_indexed(idx, pal, tl, fmt)]
	for lv in levels[1:]:
		out.append(gxtex.encode_indexed(_map_to(lv, pal), pal, tl, fmt))
	return out


def error(src: "np.ndarray", dec: "np.ndarray") -> float:
	"""MSE over visible RGB plus alpha (alpha counted as a fourth channel when it varies)."""
	if dec.shape[:2] != src.shape[:2]:
		dec = _resize(dec, src.shape[1], src.shape[0], smooth=False)
	s = src.astype(np.float32)
	d = dec.astype(np.float32)
	vis = s[..., 3] >= 128
	rgb = float(((s[..., :3] - d[..., :3]) ** 2)[vis].mean()) if vis.any() else 0.0
	if s[..., 3].min() >= 250:
		return rgb
	a = float(((s[..., 3] - d[..., 3]) ** 2).mean())
	return (3 * rgb + a) / 4


def _options(job: TexJob, full: "np.ndarray", sizes_only_full: bool = False) -> list[Option]:
	th, tw = full.shape[:2]
	fmts = (job.fixed,) if job.fixed else tuple(candidates(full, job.formats))
	out: list[Option] = []
	w, h = tw, th
	while True:
		img = _resize(full, w, h)
		for f in fmts:
			t = encode_levels([img], f)[0]
			tl = len(t.tlut) // 2
			out.append(Option(f, w, h, chain_bytes(w, h, f, job.mipmaps, job.min_edge, tl), error(full, t.decode())))
		if sizes_only_full or w // 2 < max(1, job.min_edge) or h // 2 < max(1, job.min_edge) or w <= 1 or h <= 1:
			break
		w, h = w // 2, h // 2
	return out


def _pareto(opts: list[Option]) -> list[Option]:
	"""Largest first, each next option strictly smaller and strictly worse."""
	best: list[Option] = []
	for o in sorted(opts, key=lambda o: (o.bytes, o.mse)):
		if not best or o.mse < best[-1].mse:
			best.append(o)
	return best[::-1]


def _first_ok(front: list[Option], min_db: float) -> int:
	"""Index in a Pareto front (largest first) of the smallest option above min_db."""
	ok = [i for i, o in enumerate(front) if o.db >= min_db]
	return ok[-1] if ok else 0


def _build(job: TexJob, full: "np.ndarray", o: Option) -> TplImage:
	img = _resize(full, o.width, o.height)
	levels = encode_levels(mip_chain(img, mip_sizes(o.width, o.height, job.mipmaps, job.min_edge)), o.fmt)
	npot = o.width & (o.width - 1) or o.height & (o.height - 1)
	wrap = GX_CLAMP if job.wrap == "clamp" or npot else GX_REPEAT
	return TplImage(levels, wrap, wrap, job.name, dict(job.meta))


def _full(job: TexJob) -> "np.ndarray":
	h, w = job.rgba.shape[:2]
	tw, th = target_size(w, h, job.max_edge, job.pow2, job.min_edge)
	if not job.pow2 and job.mipmaps:
		tw, th = target_size(w, h, job.max_edge, True, job.min_edge)
	return _resize(np.ascontiguousarray(job.rgba), tw, th)


def choose(job: TexJob) -> Placed:
	"""Smallest format at full target size whose picture clears job.min_db."""
	full = _full(job)
	opts = _options(job, full, sizes_only_full=True)
	front = _pareto(opts)
	o = front[_first_ok(front, job.min_db)]
	return Placed(job, o, _build(job, full, o), opts, bool(o.db < job.min_db))


@dataclass
class FitReport:
	placed: list[Placed]
	budget: int
	total: int
	fits: bool
	steps: int


def _options_task(args) -> list[Option]:
	return _options(*args)


def _map(fn, items: list, workers: int) -> list:
	if workers > 1 and len(items) > 1:
		from concurrent.futures import ProcessPoolExecutor

		try:
			with ProcessPoolExecutor(min(workers, len(items))) as ex:
				return list(ex.map(fn, items))
		except (OSError, RuntimeError):	# no process support (frozen app, sandbox): run here
			pass
	return [fn(x) for x in items]


def _build_task(args) -> TplImage:
	return _build(*args)


def fit(jobs: list[TexJob], budget: int = 0, workers: int = 1) -> FitReport:
	"""Place every job; with a budget (bytes, 0 = none), trade quality for bytes until they fit.
	`workers` > 1 encodes the candidates in that many processes."""
	fulls = [_full(j) for j in jobs]
	tried = _map(_options_task, [(j, f, budget <= 0) for j, f in zip(jobs, fulls)], workers)
	fronts = []
	for j, f, opts in zip(jobs, fulls, tried):
		# start where choose() would (smallest format clearing min_db at full size); only the
		# budget may push a texture below that, and only towards smaller, worse options
		full = [o for o in opts if (o.width, o.height) == (f.shape[1], f.shape[0])]
		ff = _pareto(full)
		start = ff[_first_ok(ff, j.min_db)]
		fronts.append(_pareto([o for o in opts if o.bytes <= start.bytes]))
	pos = [0] * len(jobs)
	total = sum(fr[p].bytes for fr, p in zip(fronts, pos))
	steps = 0
	while budget > 0 and total > budget:
		best, bi = None, -1
		for i, (fr, p) in enumerate(zip(fronts, pos)):
			if p + 1 >= len(fr):
				continue
			a, b = fr[p], fr[p + 1]
			cost = jobs[i].weight * (b.mse - a.mse) / max(1, a.bytes - b.bytes)
			if best is None or cost < best:
				best, bi = cost, i
		if bi < 0:
			break
		total -= fronts[bi][pos[bi]].bytes - fronts[bi][pos[bi] + 1].bytes
		pos[bi] += 1
		steps += 1
	# the last step can overshoot: give bytes back where they buy the most picture
	while budget > 0 and total <= budget:
		best, bi = None, -1
		for i, (fr, p) in enumerate(zip(fronts, pos)):
			if p <= 0:
				continue
			a, b = fr[p], fr[p - 1]
			if total + b.bytes - a.bytes > budget:
				continue
			gain = jobs[i].weight * (a.mse - b.mse) / max(1, b.bytes - a.bytes)
			if best is None or gain > best:
				best, bi = gain, i
		if bi < 0:
			break
		total += fronts[bi][pos[bi] - 1].bytes - fronts[bi][pos[bi]].bytes
		pos[bi] -= 1
		steps += 1
	chosen = [fr[p] for fr, p in zip(fronts, pos)]
	images = _map(_build_task, list(zip(jobs, fulls, chosen)), workers)
	placed = [Placed(j, o, im, t, bool(o.db < j.min_db)) for j, o, im, t in zip(jobs, chosen, images, tried)]
	total = sum(pl.image.size for pl in placed)
	return FitReport(placed, budget, total, budget <= 0 or total <= budget, steps)


def summary(p: Placed) -> dict:
	o = p.option
	return {"name": p.job.name, "format": o.fmt, "width": o.width, "height": o.height,
		"levels": len(p.image.levels), "bytes": p.image.size, "psnr": round(float(o.db), 2) if o.db != float("inf") else None,
		"under_min_db": bool(p.under)}
