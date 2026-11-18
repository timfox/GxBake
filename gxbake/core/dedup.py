"""Find textures stored more than once: byte-identical, or close enough to
share by a perceptual hash.

GoldSrc content repeats itself. The same WAD texture lands in every map's
world pack, and models ship copies of the same skin under other names. Two
measures:

* exact: the GX data (CMPR blocks, or the 16-bit picture of a skin) hashes
  the same, so one copy can stand for all of them with no change at all;
* near: a 64-bit difference hash (dHash: is each pixel brighter than its right
  neighbour on a 9×8 grey thumbnail) within a few bits, then confirmed by PSNR
  on the full pictures (same size only) before it is called a duplicate.

Near matches below the PSNR floor are listed as candidates for a person to
look at, never merged.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import gxtex


@dataclass
class TexItem:
	file: str	# staged relative path
	name: str	# texture or skin name inside it
	width: int
	height: int
	size: int	# bytes this copy costs where it lives
	digest: str	# hash of the bytes the GameCube samples
	rgba: "np.ndarray | None" = None	# level 0, for the perceptual check

	@property
	def label(self) -> str:
		return f"{self.file}:{self.name}"


@dataclass
class DupGroup:
	items: list[TexItem]
	exact: bool
	psnr: float = float("inf")

	@property
	def wasted(self) -> int:
		"""Bytes saved if one copy (the biggest) stood for the rest."""
		return sum(i.size for i in self.items) - max(i.size for i in self.items)

	@property
	def same_file(self) -> bool:
		return len({i.file for i in self.items}) == 1


@dataclass
class DedupReport:
	exact: list[DupGroup] = field(default_factory=list)
	near: list[DupGroup] = field(default_factory=list)
	candidates: list[DupGroup] = field(default_factory=list)	# looked alike but failed the PSNR floor
	items: int = 0


def dhash(rgba: "np.ndarray") -> int:
	"""64-bit difference hash of an H×W×3/4 picture."""
	g = rgba[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
	h, w = g.shape
	# box-average each thumbnail cell so the hash ignores texel noise
	yb = np.linspace(0, h, 9).astype(int)
	xb = np.linspace(0, w, 10).astype(int)
	thumb = np.array([[g[yb[r]:max(yb[r + 1], yb[r] + 1), xb[c]:max(xb[c + 1], xb[c] + 1)].mean()
		for c in range(9)] for r in range(8)])
	bits = (thumb[:, 1:] > thumb[:, :-1]).reshape(-1)
	return int(sum(1 << i for i, b in enumerate(bits) if b))


def _popcount(x: int) -> int:
	return bin(x).count("1")


def gct_items(rel: str, data: bytes) -> list[TexItem]:
	from .gct import read_gct

	out = []
	for e in read_gct(data):
		out.append(TexItem(rel, e.name, e.width, e.height, len(e.data),
			hashlib.sha1(struct.pack(">HHH", e.width, e.height, e.levels) + e.data).hexdigest(), e.decode(0)))
	return out


def studio_items(rel: str, raw: bytes) -> list[TexItem]:
	"""Skins as the port uploads them (RGB565, RGB5A3 when masked)."""
	out = []
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
		pal = np.frombuffer(raw, np.uint8, 768, idx + w * h).reshape(256, 3)
		rgba = np.concatenate((pal[index], np.full((h, w, 1), 255, np.uint8)), -1)
		masked = bool(flags & 0x40)
		if masked:
			rgba[index == 255] = 0
		words = gxtex.tlut_words(rgba.reshape(-1, 4), gxtex.GX_TL_RGB5A3 if masked else gxtex.GX_TL_RGB565)
		digest = hashlib.sha1(struct.pack(">HHB", w, h, masked) + words.astype(">u2").tobytes()).hexdigest()
		out.append(TexItem(rel, name, w, h, w * h + 768, digest, rgba))
	return out


def find_duplicates(items: list[TexItem], *, min_db: float = 40.0, max_bits: int = 4) -> DedupReport:
	rep = DedupReport(items=len(items))
	by_digest: dict[str, list[TexItem]] = {}
	for it in items:
		by_digest.setdefault(it.digest, []).append(it)
	singles = []
	for group in by_digest.values():
		if len(group) > 1:
			rep.exact.append(DupGroup(group, True))
		singles.append(group[0])
	# Near duplicates among one representative per exact group.
	hashes = [(dhash(it.rgba), it) for it in singles if it.rgba is not None]
	used: set[int] = set()
	for a in range(len(hashes)):
		if a in used:
			continue
		ha, ia = hashes[a]
		group, worst, cand = [ia], float("inf"), []
		for b in range(a + 1, len(hashes)):
			hb, ib = hashes[b]
			if b in used or _popcount(ha ^ hb) > max_bits:
				continue
			if (ia.width, ia.height) == (ib.width, ib.height):
				db = gxtex.psnr(ia.rgba[..., :3], ib.rgba[..., :3])
				if db >= min_db:
					group.append(ib)
					worst = min(worst, db)
					used.add(b)
					continue
			cand.append(ib)
		if len(group) > 1:
			rep.near.append(DupGroup(group, False, worst))
		if cand:
			rep.candidates.append(DupGroup([ia] + cand, False, float("nan")))
	return rep


def scan(root: Path, files: list[tuple[Path, str]], min_db: float = 40.0) -> DedupReport:
	items: list[TexItem] = []
	for path, rel in files:
		ext = path.suffix.lower()
		try:
			if ext == ".gct":
				data = path.read_bytes()
				if data[:4] == b"GCWT":
					items += gct_items(rel, data)
			elif ext == ".mdl":
				items += studio_items(rel, path.read_bytes())
		except (ValueError, struct.error):
			continue
	return find_duplicates(items, min_db=min_db)


def format_dedup(rep: DedupReport, limit: int = 10) -> str:
	from .audit import human_bytes as hb

	ex = sum(g.wasted for g in rep.exact)
	nr = sum(g.wasted for g in rep.near)
	inpack = sum(g.wasted for g in rep.exact + rep.near if g.same_file)
	lines = [f"{rep.items} textures: {len(rep.exact)} exact duplicate groups ({hb(ex)} repeated), "
		f"{len(rep.near)} near groups ({hb(nr)}); {hb(inpack)} of it inside a single file"]
	for g in sorted(rep.exact + rep.near, key=lambda g: -g.wasted)[:limit]:
		q = "identical" if g.exact else f"≥{g.psnr:.1f} dB"
		lines.append(f"  {hb(g.wasted):>9}  {q:<10} " + ", ".join(i.label for i in g.items[:4])
			+ (f" +{len(g.items) - 4}" if len(g.items) > 4 else ""))
	if rep.candidates:
		lines.append(f"  {len(rep.candidates)} look-alike groups below the PSNR floor (check by eye):")
		for g in rep.candidates[:limit]:
			lines.append("    " + ", ".join(i.label for i in g.items[:4]))
	return "\n".join(lines)
