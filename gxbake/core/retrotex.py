"""PC and retro game picture formats as input.

- id Software: Doom/Heretic/Hexen lumps (`.lmp`: patches, 64×64 flats,
  PLAYPAL, COLORMAP) and Quake lumps (`.lmp`: qpics, palette.lmp, colormap);
- Build engine `.art` tile files (Duke Nukem 3D, Blood, Shadow Warrior);
- Westwood `.shp` sprites (Tiberian Dawn / Red Alert, and Tiberian Sun / RA2);
- Fallout 1/2 `.frm` sprites;
- Unreal Engine 1 `.utx` texture packages (the first texture);
- Electronic Arts IFF `.iff` / `.lbm` (ILBM and PBM, EHB and HAM);
- Atari ST Degas (`.pi1`–`.pi3`, `.pc1`–`.pc3`) and C64 Koala (`.koa`).

Palettes are looked for beside the file and in the folders above it (Doom
PLAYPAL in a lump or an IWAD, Build PALETTE.DAT, Fallout COLOR.PAL, a
Westwood .pal, Quake gfx/palette.lmp). When none is found the picture is
decoded in greyscale and a warning says so. Multi-frame sprites become a
sprite sheet (frames left to right in rows).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import texcodec as tc


def _warn(warnings, msg: str) -> None:
	if warnings is not None:
		warnings.append(msg)


def _ancestors(start: Path, depth: int = 4) -> list[Path]:
	start = Path(start).resolve()
	out = [start.parent]
	for _ in range(depth):
		if out[-1].parent == out[-1]:
			break
		out.append(out[-1].parent)
	return out


def find_near(start: Path, names: tuple[str, ...], depth: int = 4) -> Path | None:
	"""A file named one of `names` (case-insensitive) beside `start` or in a folder above it."""
	want = [n.lower() for n in names]
	for d in _ancestors(start, depth):
		try:
			files = {p.name.lower(): p for p in d.iterdir() if p.is_file()}
		except OSError:
			continue
		for n in want:
			if n in files:
				return files[n]
	return None


# -- Doom / Quake lumps ----------------------------------------------------------------------


def wad_lump(wad: bytes, name: str) -> bytes | None:
	"""A lump from a Doom IWAD/PWAD by name."""
	if wad[:4] not in (b"IWAD", b"PWAD"):
		return None
	n, diroff = struct.unpack_from("<ii", wad, 4)
	want = name.upper().encode()
	for i in range(n):
		off, size, raw = struct.unpack_from("<ii8s", wad, diroff + 16 * i)
		if raw.rstrip(b"\0").upper() == want:
			return wad[off:off + size]
	return None


def doom_palette(start: Path) -> bytes | None:
	p = find_near(start, ("playpal.lmp", "playpal", "playpal.pal"))
	if p is not None and p.stat().st_size >= 768:
		return p.read_bytes()[:768]
	for d in _ancestors(start, 3):
		try:
			wads = sorted(f for f in d.iterdir() if f.suffix.lower() == ".wad" and f.is_file())
		except OSError:
			continue
		for wad in wads[:8]:
			with open(wad, "rb") as f:
				if f.read(4) != b"IWAD":
					continue
			lump = wad_lump(wad.read_bytes(), "PLAYPAL")
			if lump and len(lump) >= 768:
				return lump[:768]
	return None


def _swatch(pal: "np.ndarray") -> "np.ndarray":
	"""A palette as a 16×16 grid of 8×8 swatches."""
	cells = pal[:256].reshape(16, 16, 4)
	return np.repeat(np.repeat(cells, 8, 0), 8, 1)


def read_doom_patch(data: bytes, pal: "np.ndarray") -> "np.ndarray":
	"""Doom column-post picture: transparent where no post covers."""
	w, h, _left, _top = struct.unpack_from("<hhhh", data, 0)
	if not (0 < w <= 4096 and 0 < h <= 4096) or 8 + 4 * w > len(data):
		raise ValueError("not a Doom patch")
	cols = struct.unpack_from(f"<{w}I", data, 8)
	out = np.zeros((h, w, 4), np.uint8)
	for x, off in enumerate(cols):
		if off >= len(data):
			raise ValueError("not a Doom patch")
		top = -1
		while off < len(data) and data[off] != 0xFF:
			delta, n = data[off], data[off + 1]
			top = top + delta if delta <= top else delta	# DeePsea tall patches
			px = np.frombuffer(data, np.uint8, n, off + 3)
			y0, y1 = top, min(h, top + n)
			if y1 > y0:
				out[y0:y1, x] = pal[px[: y1 - y0]]
				out[y0:y1, x, 3] = 255
			off += n + 4
	return out


def _is_doom_patch(data: bytes) -> bool:
	if len(data) < 12:
		return False
	w, h = struct.unpack_from("<hh", data, 0)
	if not (0 < w <= 4096 and 0 < h <= 4096) or 8 + 4 * w > len(data):
		return False
	cols = struct.unpack_from(f"<{w}I", data, 8)
	return all(8 + 4 * w <= c < len(data) for c in cols)


def load_lmp(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	"""A Doom or Quake .lmp: qpic, patch, flat, palette or colormap."""
	from .quake import quake_palette

	path = Path(path)
	data = path.read_bytes()
	n = len(data)
	if n >= 8:
		w, h = struct.unpack_from("<II", data, 0)
		if 0 < w <= 4096 and 0 < h <= 4096 and 8 + w * h == n:	# Quake qpic (gfx.wad pics, conback)
			pal = quake_palette(path)
			if pal is None:
				pal = doom_palette(path)
			if pal is None:
				_warn(warnings, "Quake palette (gfx/palette.lmp) not found; greyscale used")
			rgba = tc.palette_rgba(pal, transparent=255) if pal else tc.GREY_RGBA.copy()
			if pal is None:
				rgba[255, 3] = 0
			return rgba[np.frombuffer(data, np.uint8, w * h, 8)].reshape(h, w, 4)
	if n in (768, 768 * 14):	# Quake palette.lmp / Doom PLAYPAL
		return _swatch(tc.palette_rgba(data[:768]))
	pal_bytes = doom_palette(path) or quake_palette(path)
	if pal_bytes is None:
		_warn(warnings, "no palette found (PLAYPAL lump, an IWAD, or Quake's gfx/palette.lmp); greyscale used")
		pal = tc.GREY_RGBA
	else:
		pal = tc.palette_rgba(pal_bytes)
	if _is_doom_patch(data):
		return read_doom_patch(data, pal)
	if n in (4096, 8192, 16384) or (n == 4160):	# flats (Heretic/Hexen have 64×128 and 128×128)
		w = 128 if n == 16384 else 64
		h = n // w if n != 4160 else 65
		return pal[np.frombuffer(data, np.uint8, w * h)].reshape(h, w, 4)
	if n in (8704, 16385, 34 * 256):	# Doom COLORMAP, Quake colormap.lmp
		rows = n // 256
		return pal[np.frombuffer(data, np.uint8, rows * 256)].reshape(rows, 256, 4)
	raise ValueError("unrecognised .lmp lump (not a qpic, patch, flat, palette or colormap)")


# -- Build engine ART --------------------------------------------------------------------------


def build_palette(start: Path) -> bytes | None:
	p = find_near(start, ("palette.dat",))
	return p.read_bytes()[:768] if p is not None else None


def read_art(data: bytes, pal: "np.ndarray", warnings: list[str] | None = None) -> "np.ndarray":
	version, _num, start, end = struct.unpack_from("<iiii", data, 0)
	if version != 1 or end < start or end - start > 9999:
		raise ValueError("not a Build engine ART file")
	n = end - start + 1
	xs = np.frombuffer(data, "<i2", n, 16)
	ys = np.frombuffer(data, "<i2", n, 16 + 2 * n)
	pos = 16 + 8 * n
	frames = []
	for w, h in zip(xs.tolist(), ys.tolist()):
		if w <= 0 or h <= 0:
			continue
		idx = np.frombuffer(data, np.uint8, w * h, pos).reshape(w, h).T	# stored column by column
		pos += w * h
		frames.append(pal[idx])
	if len(frames) > 1:
		_warn(warnings, f"ART file: {len(frames)} tiles laid out as one sheet")
	return tc.sheet(frames, pad=1)


def load_art(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	raw = build_palette(path)
	if raw is None:
		_warn(warnings, "PALETTE.DAT not found; greyscale used")
	pal = tc.palette_rgba(raw, six_bit=True, transparent=255) if raw else tc.GREY_RGBA.copy()
	pal[255, 3] = 0
	return read_art(Path(path).read_bytes(), pal, warnings)


# -- Westwood SHP ------------------------------------------------------------------------------


def lcw_decode(src: bytes, size: int) -> bytearray:
	"""Westwood LCW ("format 80") decompression."""
	dst = bytearray()
	i = 0
	n = len(src)
	while i < n and len(dst) < size:
		c = src[i]
		i += 1
		if not c & 0x80:	# 0cccpppp pppppppp: copy from `p` bytes back
			count = (c >> 4) + 3
			back = ((c & 0x0F) << 8) | src[i]
			i += 1
			p = len(dst) - back
			for k in range(count):
				dst.append(dst[p + k])
		elif not c & 0x40:	# 10cccccc: literal run
			count = c & 0x3F
			if not count:
				break
			dst += src[i:i + count]
			i += count
		else:
			count = c & 0x3F
			if count == 0x3E:	# fill
				count = src[i] | src[i + 1] << 8
				dst += bytes((src[i + 2],)) * count
				i += 3
				continue
			if count == 0x3F:
				count = src[i] | src[i + 1] << 8
				i += 2
			else:
				count += 3
			p = src[i] | src[i + 1] << 8	# absolute position in the output
			i += 2
			for k in range(count):
				dst.append(dst[p + k])
	dst += bytes(max(0, size - len(dst)))
	return dst[:size]


def xor_delta_apply(dst: bytearray, src: bytes) -> None:
	"""Westwood XOR delta ("format 40") applied in place."""
	i = p = 0
	n = len(src)
	while i < n:
		c = src[i]
		i += 1
		if not c & 0x80:
			if c == 0:
				count, v = src[i], src[i + 1]
				i += 2
				for k in range(count):
					dst[p + k] ^= v
				p += count
			else:
				for k in range(c):
					dst[p + k] ^= src[i + k]
				i += c
				p += c
			continue
		count = c & 0x7F
		if count:
			p += count
			continue
		count = src[i] | src[i + 1] << 8
		i += 2
		if not count & 0x8000:
			if not count:
				break
			p += count
		elif count & 0x4000:
			count &= 0x3FFF
			v = src[i]
			i += 1
			for k in range(count):
				dst[p + k] ^= v
			p += count
		else:
			count &= 0x3FFF
			for k in range(count):
				dst[p + k] ^= src[i + k]
			i += count
			p += count


def read_shp_td(data: bytes) -> list["np.ndarray"]:
	"""Tiberian Dawn / Red Alert SHP → index frames."""
	n, _x, _y, w, h, _largest, _flags = struct.unpack_from("<7H", data, 0)
	if not n or not w or not h:
		raise ValueError("not a Westwood SHP")
	offs = [struct.unpack_from("<II", data, 14 + 8 * i) for i in range(n + 1)]
	frames: list[bytearray] = []
	by_offset: dict[int, bytearray] = {}
	size = w * h
	for i in range(n):
		a, b = offs[i]
		fmt, off = a >> 24, a & 0xFFFFFF
		end = offs[i + 1][0] & 0xFFFFFF or len(data)
		chunk = data[off:end]
		if fmt == 0x80:
			frame = lcw_decode(chunk, size)
		elif fmt == 0x40:
			base = by_offset.get(b & 0xFFFFFF)
			frame = bytearray(base) if base is not None else bytearray(size)
			xor_delta_apply(frame, chunk)
		elif fmt == 0x20:
			frame = bytearray(frames[-1]) if frames else bytearray(size)
			xor_delta_apply(frame, chunk)
		else:
			raise ValueError(f"Westwood SHP: unknown frame format 0x{fmt:02X}")
		by_offset[off] = frame
		frames.append(frame)
	return [np.frombuffer(bytes(f), np.uint8).reshape(h, w) for f in frames]


def read_shp_ts(data: bytes) -> list["np.ndarray"]:
	"""Tiberian Sun / Red Alert 2 SHP → index frames on the full canvas."""
	zero, w, h, n = struct.unpack_from("<4H", data, 0)
	if zero or not n or not w or not h:
		raise ValueError("not a Tiberian Sun SHP")
	frames = []
	for i in range(n):
		fx, fy, fw, fh, flags, _col, _res, off = struct.unpack_from("<4HIIII", data, 8 + 24 * i)
		canvas = np.zeros((h, w), np.uint8)
		if fw and fh and off:
			if flags & 2:	# rows of [u16 length][0, count = zeros | literal bytes]
				rows = []
				p = off
				for _ in range(fh):
					ln = struct.unpack_from("<H", data, p)[0]
					row = bytearray()
					q, end = p + 2, p + ln
					while q < end and len(row) < fw:
						if data[q] == 0:
							row += bytes(data[q + 1])
							q += 2
						else:
							row.append(data[q])
							q += 1
					rows.append(bytes(row[:fw]).ljust(fw, b"\0"))
					p = end
				pix = np.frombuffer(b"".join(rows), np.uint8).reshape(fh, fw)
			else:
				pix = np.frombuffer(data, np.uint8, fw * fh, off).reshape(fh, fw)
			canvas[fy:fy + fh, fx:fx + fw] = pix[: h - fy, : w - fx]
		frames.append(canvas)
	return frames


def westwood_palette(start: Path) -> bytes | None:
	p = find_near(start, ("temperat.pal", "unittem.pal", "unittem.pal", "temperate.pal", "palette.pal", "snow.pal"))
	if p is None:
		for d in _ancestors(start, 1):
			pals = sorted(f for f in d.iterdir() if f.suffix.lower() == ".pal" and f.stat().st_size == 768)
			if pals:
				p = pals[0]
				break
	return p.read_bytes()[:768] if p is not None else None


def load_shp(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	data = Path(path).read_bytes()
	frames = read_shp_ts(data) if data[:2] == b"\0\0" else read_shp_td(data)
	raw = westwood_palette(path)
	if raw is None:
		_warn(warnings, "no Westwood .pal found beside the .shp; greyscale used")
		pal = tc.GREY_RGBA.copy()
	else:
		pal = tc.palette_rgba(raw, six_bit=tc.looks_six_bit(raw))
	pal[0, 3] = 0
	if len(frames) > 1:
		_warn(warnings, f"SHP: {len(frames)} frames laid out as one sheet")
	return tc.sheet([pal[f] for f in frames], pad=1)


# -- Fallout FRM -------------------------------------------------------------------------------


def fallout_palette(start: Path) -> bytes | None:
	p = find_near(start, ("color.pal",))
	return p.read_bytes()[:768] if p is not None else None


def read_frm(data: bytes, pal: "np.ndarray", warnings: list[str] | None = None) -> "np.ndarray":
	_ver, _fps, _action, per_dir = struct.unpack_from(">IHHH", data, 0)
	dir_offs = struct.unpack_from(">6I", data, 34)
	if not 0 < per_dir < 1000:
		raise ValueError("not a Fallout FRM")
	pos = 62 + dir_offs[0]
	frames = []
	for _ in range(per_dir):
		w, h, size = struct.unpack_from(">HHI", data, pos)
		frames.append(pal[np.frombuffer(data, np.uint8, w * h, pos + 12).reshape(h, w)])
		pos += 12 + size
	dirs = len({o for o in dir_offs})
	if dirs > 1:
		_warn(warnings, f"FRM has {dirs} directions; only the first is converted")
	if per_dir > 1:
		_warn(warnings, f"FRM: {per_dir} frames laid out as one sheet")
	return tc.sheet(frames, pad=1)


def load_frm(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	raw = fallout_palette(path)
	if raw is None:
		_warn(warnings, "Fallout COLOR.PAL not found; greyscale used")
		pal = tc.GREY_RGBA.copy()
	else:
		pal = tc.palette_rgba(raw, six_bit=True)
	pal[0, 3] = 0
	return read_frm(Path(path).read_bytes(), pal, warnings)


# -- Unreal Engine 1 UTX -----------------------------------------------------------------------


def _compact(d: bytes, p: int) -> tuple[int, int]:
	b0 = d[p]
	p += 1
	v = b0 & 0x3F
	if b0 & 0x40:
		shift = 6
		while True:
			b = d[p]
			p += 1
			v |= (b & 0x7F) << shift
			shift += 7
			if not b & 0x80 or shift > 27:
				break
	return (-v if b0 & 0x80 else v), p


class UnrealPackage:
	"""Just enough of an Unreal Engine 1/2 package to find textures and palettes."""

	def __init__(self, data: bytes):
		magic, self.version, _lic, _flags, nn, no, ne, eo, ni, io = struct.unpack_from("<IHHIIIIIII", data, 0)
		if magic != 0x9E2A83C1:
			raise ValueError("not an Unreal package")
		self.data = data
		self.names = []
		p = no
		for _ in range(nn):
			if self.version >= 64:
				ln, p = _compact(data, p)
				s = data[p:p + ln].rstrip(b"\0")
				p += ln
			else:
				e = data.index(b"\0", p)
				s, p = data[p:e], e + 1
			self.names.append(s.decode("latin-1"))
			p += 4
		self.imports = []
		p = io
		for _ in range(ni):
			_cp, p = _compact(data, p)
			cn, p = _compact(data, p)
			p += 4
			on, p = _compact(data, p)
			self.imports.append((self.names[cn], self.names[on]))
		self.exports = []
		p = eo
		for _ in range(ne):
			cls, p = _compact(data, p)
			_sup, p = _compact(data, p)
			p += 4
			name, p = _compact(data, p)
			p += 4
			size, p = _compact(data, p)
			off = 0
			if size > 0:
				off, p = _compact(data, p)
			self.exports.append((self.class_name(cls), self.names[name], off, size))

	def class_name(self, ref: int) -> str:
		if ref < 0:
			return self.imports[-ref - 1][1]
		if ref > 0:
			return self.names[0] if False else "export"
		return "Class"

	def properties(self, p: int) -> tuple[dict, int]:
		d = self.data
		props: dict = {}
		while True:
			ni, p = _compact(d, p)
			name = self.names[ni]
			if name == "None":
				return props, p
			info = d[p]
			p += 1
			typ, sc, arr = info & 0x0F, (info >> 4) & 7, info & 0x80
			if typ == 10:
				_s, p = _compact(d, p)
			size = {0: 1, 1: 2, 2: 4, 3: 12, 4: 16}.get(sc)
			if sc == 5:
				size, p = d[p], p + 1
			elif sc == 6:
				size, p = struct.unpack_from("<H", d, p)[0], p + 2
			elif sc == 7:
				size, p = struct.unpack_from("<I", d, p)[0], p + 4
			if typ == 3:	# bool: the value is the array bit, no payload
				props[name] = bool(arr)
				continue
			if arr:
				b = d[p]
				p += 1 if b < 0x80 else 2 if b & 0xC0 == 0x80 else 4
			val = d[p:p + size]
			p += size
			if typ == 1:
				props[name] = val[0]
			elif typ == 2:
				props[name] = struct.unpack_from("<i", val)[0]
			elif typ == 5:
				props[name] = _compact(val, 0)[0]
			else:
				props[name] = val

	def palette(self, ref: int) -> "np.ndarray | None":
		if ref <= 0 or ref > len(self.exports):
			return None
		cls, _name, off, _size = self.exports[ref - 1]
		_props, p = self.properties(off)
		n, p = _compact(self.data, p)
		rgba = np.frombuffer(self.data, np.uint8, n * 4, p).reshape(n, 4).copy()
		rgba[:, 3] = 255
		return rgba


def read_utx(data: bytes, warnings: list[str] | None = None) -> "np.ndarray":
	pkg = UnrealPackage(data)
	texs = [e for e in pkg.exports if e[0] in ("Texture", "ScriptedTexture") and e[3] > 0]
	if not texs:
		raise ValueError("no textures in this Unreal package")
	if len(texs) > 1:
		_warn(warnings, f"package holds {len(texs)} textures; only the first ({texs[0][1]}) is converted")
	if pkg.version > 69:
		_warn(warnings, f"Unreal package version {pkg.version} (Unreal Engine 2) is read on a best-effort basis")
	_cls, _name, off, _size = texs[0]
	props, p = pkg.properties(off)
	d = pkg.data
	nmips, p = _compact(d, p)
	if nmips <= 0:
		raise ValueError("texture has no mipmaps")
	if pkg.version >= 63:
		p += 4	# lazy array skip offset
	ln, p = _compact(d, p)
	pix = d[p:p + ln]
	p += ln
	w, h = struct.unpack_from("<ii", d, p)
	fmt = props.get("Format", 0)
	if fmt == 0:	# P8
		pal = pkg.palette(props.get("Palette", 0))
		if pal is None:
			_warn(warnings, "texture palette not found; greyscale used")
			pal = tc.GREY_RGBA.copy()
		pal = np.concatenate([pal, np.zeros((max(0, 256 - len(pal)), 4), np.uint8)])[:256]
		if props.get("bMasked"):
			pal[0, 3] = 0
		return pal[np.frombuffer(pix, np.uint8, w * h)].reshape(h, w, 4)
	if fmt == 3:
		return tc.decode_blocks("bc1", pix, w, h)
	if fmt == 5:
		return tc.unpack_bytes(pix, w, h, "BGRA")
	if fmt == 4:
		return tc.unpack_bytes(pix, w, h, "BGR")
	raise ValueError(f"Unreal texture format {fmt} is not supported")


# -- IFF ILBM / PBM ------------------------------------------------------------------------------


def byterun1(src: bytes, size: int, pos: int = 0) -> tuple[bytes, int]:
	out = bytearray()
	n = len(src)
	while len(out) < size and pos < n:
		c = src[pos]
		pos += 1
		if c < 128:
			out += src[pos:pos + c + 1]
			pos += c + 1
		elif c > 128:
			out += bytes((src[pos],)) * (257 - c)
			pos += 1
	return bytes(out[:size]).ljust(size, b"\0"), pos


def read_iff(data: bytes, warnings: list[str] | None = None) -> "np.ndarray":
	if data[:4] != b"FORM" or data[8:12] not in (b"ILBM", b"PBM "):
		raise ValueError("not an IFF ILBM/PBM picture")
	chunky = data[8:12] == b"PBM "
	chunks: dict[bytes, bytes] = {}
	p = 12
	while p + 8 <= len(data):
		cid, ln = data[p:p + 4], struct.unpack_from(">I", data, p + 4)[0]
		chunks.setdefault(cid, data[p + 8:p + 8 + ln])
		p += 8 + ln + (ln & 1)
	if b"BMHD" not in chunks or b"BODY" not in chunks:
		raise ValueError("IFF picture without BMHD or BODY")
	w, h, _x, _y, planes, masking, comp, _pad, trans = struct.unpack_from(">HHhhBBBBH", chunks[b"BMHD"])
	camg = struct.unpack_from(">I", chunks[b"CAMG"])[0] if len(chunks.get(b"CAMG", b"")) >= 4 else 0
	body = chunks[b"BODY"]
	cmap = chunks.get(b"CMAP", b"")
	if chunky:
		row = w + (w & 1)
		raw = byterun1(body, row * h)[0] if comp else body[: row * h].ljust(row * h, b"\0")
		idx = np.frombuffer(raw, np.uint8).reshape(h, row)[:, :w]
		mask = None
	else:
		rb = (w + 15) // 16 * 2
		nplanes = planes + (1 if masking == 1 else 0)
		raw = byterun1(body, rb * nplanes * h)[0] if comp else body[: rb * nplanes * h].ljust(rb * nplanes * h, b"\0")
		bits = np.unpackbits(np.frombuffer(raw, np.uint8).reshape(h, nplanes, rb), axis=2)[:, :, :w]
		idx = np.zeros((h, w), np.uint32)
		for k in range(planes):
			idx |= bits[:, k].astype(np.uint32) << k
		mask = bits[:, planes] if masking == 1 else None
	if not chunky and planes in (24, 32):
		r, g, b = (idx >> 0) & 255, (idx >> 8) & 255, (idx >> 16) & 255
		a = (idx >> 24) & 255 if planes == 32 else np.full_like(idx, 255)
		out = np.dstack([r, g, b, a]).astype(np.uint8)
	else:
		pal = tc.palette_rgba(cmap) if cmap else tc.GREY_RGBA.copy()
		if not cmap:
			_warn(warnings, "IFF picture has no CMAP; greyscale used")
		ncol = len(cmap) // 3
		if camg & 0x80 and ncol <= 32:	# Extra Half-Brite: colours 32..63 are half of 0..31
			pal[32:64, :3] = pal[:32, :3] // 2
		if camg & 0x800 and planes in (6, 8):	# HAM6 / HAM8
			out = _ham(idx, pal, planes)
		else:
			out = pal[np.minimum(idx, 255)]
	if mask is not None:
		out[..., 3] = mask * 255
	elif masking == 2 and not (planes in (24, 32)):
		out[idx == trans, 3] = 0
	return out


def _ham(idx: "np.ndarray", pal: "np.ndarray", planes: int) -> "np.ndarray":
	vb = planes - 2
	h, w = idx.shape
	out = np.zeros((h, w, 4), np.uint8)
	out[..., 3] = 255
	for y in range(h):
		r, g, b = (int(c) for c in pal[0, :3])
		for x in range(w):
			v = int(idx[y, x])
			ctl, val = v >> vb, v & ((1 << vb) - 1)
			c8 = (val << (8 - vb)) | (val >> (2 * vb - 8) if vb >= 4 else 0)
			if ctl == 0:
				r, g, b = (int(c) for c in pal[val, :3])
			elif ctl == 1:
				b = c8
			elif ctl == 2:
				r = c8
			else:
				g = c8
			out[y, x, :3] = r, g, b
	return out


# -- Atari ST Degas, C64 Koala -----------------------------------------------------------------------


def read_degas(data: bytes) -> "np.ndarray":
	res = struct.unpack_from(">H", data, 0)[0]
	compressed = bool(res & 0x8000)
	res &= 3
	w, h, planes = {0: (320, 200, 4), 1: (640, 200, 2), 2: (640, 400, 1)}[res]
	words = np.frombuffer(data, ">u2", 16, 2).astype(np.int32)

	def ste(v):	# STE: the low bit of each 4-bit channel is stored in bit 3
		v = ((v & 7) << 1) | ((v >> 3) & 1)
		return (v * 17).astype(np.uint8)

	pal = np.zeros((16, 4), np.uint8)
	pal[:, 0], pal[:, 1], pal[:, 2] = ste(words >> 8 & 15), ste(words >> 4 & 15), ste(words & 15)
	pal[:, 3] = 255
	if res == 2:	# monochrome: colour 0 is white
		pal[0, :3], pal[1, :3] = 255, 0
	rb = w // 8	# bytes per plane per row
	if compressed:	# each row: each plane's bytes, PackBits
		raw, pos = bytearray(), 34
		for _ in range(h):
			row, pos = byterun1(data, rb * planes, pos)
			raw += row
		bits = np.unpackbits(np.frombuffer(bytes(raw), np.uint8).reshape(h, planes, rb * 8 // 8), axis=2)
	else:	# interleaved 16-pixel words: plane0 plane1 ... per group
		raw = np.frombuffer(data, np.uint8, 32000, 34).reshape(h, w // 16, planes, 2)
		bits = np.unpackbits(raw.transpose(0, 2, 1, 3).reshape(h, planes, w // 8), axis=2)
	idx = np.zeros((h, w), np.uint8)
	for k in range(planes):
		idx |= bits[:, k, :w] << k
	return pal[idx]


C64_PALETTE = np.array([
	(0x00, 0x00, 0x00), (0xFF, 0xFF, 0xFF), (0x68, 0x37, 0x2B), (0x70, 0xA4, 0xB2), (0x6F, 0x3D, 0x86),
	(0x58, 0x8D, 0x43), (0x35, 0x28, 0x79), (0xB8, 0xC7, 0x6F), (0x6F, 0x4F, 0x25), (0x43, 0x39, 0x00),
	(0x9A, 0x67, 0x59), (0x44, 0x44, 0x44), (0x6C, 0x6C, 0x6C), (0x9A, 0xD2, 0x84), (0x6C, 0x5E, 0xB5),
	(0x95, 0x95, 0x95)], np.uint8)


def read_koala(data: bytes) -> "np.ndarray":
	"""C64 Koala Painter multicolour bitmap (160×200 double-wide → 320×200)."""
	if len(data) == 10003:
		data = data[2:]
	if len(data) < 10001:
		raise ValueError("not a Koala picture")
	bitmap = np.frombuffer(data, np.uint8, 8000, 0).reshape(25, 40, 8)
	screen = np.frombuffer(data, np.uint8, 1000, 8000).reshape(25, 40)
	colour = np.frombuffer(data, np.uint8, 1000, 9000).reshape(25, 40)
	bg = data[10000] & 15
	pairs = np.stack([(bitmap >> s) & 3 for s in (6, 4, 2, 0)], -1)	# 25,40,8,4
	choices = np.stack([np.full((25, 40), bg), screen >> 4, screen & 15, colour & 15], -1)	# 25,40,4
	ci = np.take_along_axis(choices[:, :, None, :], pairs.astype(np.intp), -1)	# 25,40,8,4
	img = ci.transpose(0, 2, 1, 3).reshape(200, 160)
	rgb = C64_PALETTE[img]
	rgb = np.repeat(rgb, 2, axis=1)
	return np.dstack([rgb, np.full((200, 320), 255, np.uint8)])


# -- dispatch ------------------------------------------------------------------------------------


def load(path: Path, warnings: list[str] | None = None) -> "np.ndarray":
	path = Path(path)
	ext = path.suffix.lower()
	if ext == ".lmp":
		return load_lmp(path, warnings)
	if ext == ".art":
		return load_art(path, warnings)
	if ext == ".shp":
		return load_shp(path, warnings)
	if ext == ".frm" or ext[:3] == ".fr" and ext[3:].isdigit():
		return load_frm(path, warnings)
	data = path.read_bytes()
	if ext in (".utx",):
		return read_utx(data, warnings)
	if ext in (".iff", ".lbm", ".ilbm"):
		return read_iff(data, warnings)
	if ext in (".pi1", ".pi2", ".pi3", ".pc1", ".pc2", ".pc3"):
		return read_degas(data)
	if ext in (".koa", ".kla"):
		return read_koala(data)
	raise ValueError(f"unrecognised retro picture {ext}")
