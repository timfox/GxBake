"""GoldSrc studio models (.mdl): lean mesh + downsampled skin companion.

lean_studio_mesh_bytes and lean_studio_tex_companion_bytes are ported
unchanged from scripts/build-gamecube-disc.py (G289 tip-safe studio BSS).
studio_textures is new and feeds the GUI preview.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def lean_studio_mesh_bytes(src: Path) -> bytes | None:
	"""G289: truncate MDL before texture blobs (numtextures=0) for tip-safe BSS.

	Keeps bones/meshes/seqs; Flipper binds white soft skins under quality 0.
	Handgun drops ~134 KiB → ~60 KiB; crowbar ~47 → ~19 KiB.
	"""
	try:
		raw = src.read_bytes()
	except OSError:
		return None
	if len(raw) < 192 or raw[:4] not in (b"IDST", b"IDSQ"):
		return None
	# studiohdr_t: textureindex @184, texturedataindex @188, numtextures @180
	numtextures = struct.unpack_from("<i", raw, 180)[0]
	textureindex = struct.unpack_from("<i", raw, 184)[0]
	texturedataindex = struct.unpack_from("<i", raw, 188)[0]
	length = struct.unpack_from("<i", raw, 72)[0]
	if textureindex <= 244 or textureindex > len(raw):
		# No usable texture cursor — keep whole file if small enough.
		if length <= 0 or length > len(raw):
			length = len(raw)
		return bytes(raw[:length])
	# Cut at texture chunk; clear texture counts so loaders skip skin blobs.
	cut = textureindex
	if cut > len(raw):
		cut = len(raw)
	out = bytearray(raw[:cut])
	struct.pack_into("<i", out, 72, cut)  # length
	struct.pack_into("<i", out, 180, 0)  # numtextures
	struct.pack_into("<i", out, 184, 0)  # textureindex
	struct.pack_into("<i", out, 188, 0)  # texturedataindex
	# skinref tables often live after textures; zero if past cut
	numskinref = struct.unpack_from("<i", out, 192)[0] if len(out) >= 196 else 0
	skinindex = struct.unpack_from("<i", out, 200)[0] if len(out) >= 204 else 0
	if skinindex >= cut:
		struct.pack_into("<i", out, 192, 0)
		struct.pack_into("<i", out, 196, 0)
		struct.pack_into("<i", out, 200, 0)
	_ = (numtextures, texturedataindex, numskinref)
	return bytes(out)


def lean_studio_tex_companion_bytes(src: Path, max_edge: int = 64) -> bytes | None:
	"""Build a tip-safe *t.mdl skin atlas from an embedded-texture studio MDL.

	CapFaces DoD rifles are 300–600 KiB fully textured — too big for the 184 KiB
	studio BSS and the ~400 KiB tip heap after densify. Mesh stays lean; this
	companion downsamples indexed skins so gc_studio/<stem>t.mdl fits the
	132 KiB studio tex scratch and Mod_GCAttachStudioSkins can upload.
	"""
	try:
		raw = src.read_bytes()
	except OSError:
		return None
	if len(raw) < 244 or raw[:4] not in (b"IDST", b"IDSQ"):
		return None
	numtextures = struct.unpack_from("<i", raw, 180)[0]
	textureindex = struct.unpack_from("<i", raw, 184)[0]
	numskinref = struct.unpack_from("<i", raw, 192)[0]
	numskinfamilies = struct.unpack_from("<i", raw, 196)[0]
	skinindex = struct.unpack_from("<i", raw, 200)[0]
	if numtextures <= 0 or textureindex <= 0 or textureindex + numtextures * 80 > len(raw):
		return None

	# mstudiotexture_t is 80 bytes; pixels are width*height + 768 RGB palette.
	tex_hdrs: list[tuple[bytes, int, int, int, bytes, bytes]] = []
	for i in range(numtextures):
		off = textureindex + i * 80
		name = raw[off : off + 64]
		flags, width, height, index = struct.unpack_from("<iiii", raw, off + 64)
		if width <= 0 or height <= 0 or index < 0 or index + width * height + 768 > len(raw):
			return None
		pixels = raw[index : index + width * height]
		palette = raw[index + width * height : index + width * height + 768]
		tex_hdrs.append((name, flags, width, height, pixels, palette))

	skin_bytes = b""
	if numskinref > 0 and numskinfamilies > 0 and skinindex > 0:
		skin_size = numskinref * numskinfamilies * 2
		if skinindex + skin_size <= len(raw):
			skin_bytes = raw[skinindex : skinindex + skin_size]
		else:
			numskinref = 0
			numskinfamilies = 0
	else:
		numskinref = 0
		numskinfamilies = 0

	# Header + texture structs + skinrefs, then tightly packed downsampled blobs.
	hdr_size = 244  # sizeof(studiohdr_t) on GoldSrc
	tex_table = hdr_size
	skin_table = tex_table + numtextures * 80
	pix_cursor = skin_table + len(skin_bytes)
	out = bytearray(pix_cursor)
	out[0:4] = b"IDST"
	struct.pack_into("<i", out, 4, 10)  # STUDIO_VERSION
	out[8:72] = raw[8:72]  # model name

	for i, (name, flags, width, height, pixels, palette) in enumerate(tex_hdrs):
		nw, nh = width, height
		while nw > max_edge or nh > max_edge:
			nw = max(1, nw // 2)
			nh = max(1, nh // 2)
		if nw == width and nh == height:
			new_pixels = pixels
		else:
			new_pixels = bytearray(nw * nh)
			for y in range(nh):
				sy = min(height - 1, (y * height) // nh)
				for x in range(nw):
					sx = min(width - 1, (x * width) // nw)
					new_pixels[y * nw + x] = pixels[sy * width + sx]
			new_pixels = bytes(new_pixels)
		blob = new_pixels + palette
		off = tex_table + i * 80
		out[off : off + 64] = name
		# Drop chrome/additive; keep MASKED (0x40) so VM chromakey still floods.
		keep_flags = flags & 0x40
		struct.pack_into("<iiii", out, off + 64, keep_flags, nw, nh, pix_cursor)
		out.extend(blob)
		pix_cursor += len(blob)

	if skin_bytes:
		out[skin_table : skin_table + len(skin_bytes)] = skin_bytes

	struct.pack_into("<i", out, 72, len(out))  # length
	struct.pack_into("<i", out, 180, numtextures)
	struct.pack_into("<i", out, 184, tex_table)  # textureindex
	struct.pack_into("<i", out, 188, skin_table + len(skin_bytes))  # texturedataindex
	struct.pack_into("<i", out, 192, numskinref)
	struct.pack_into("<i", out, 196, numskinfamilies)
	struct.pack_into("<i", out, 200, skin_table if skin_bytes else 0)
	return bytes(out)



def studio_info(raw: bytes) -> dict:
	"""Header summary for the inspector."""
	if len(raw) < 244 or raw[:4] not in (b"IDST", b"IDSQ"):
		raise ValueError("not a GoldSrc studio model")
	name = raw[8:72].split(b"\0")[0].decode("latin-1")
	fields = {
		"name": name,
		"version": struct.unpack_from("<i", raw, 4)[0],
		"length": struct.unpack_from("<i", raw, 72)[0],
		"bones": struct.unpack_from("<i", raw, 140)[0],
		"sequences": struct.unpack_from("<i", raw, 164)[0],
		"textures": struct.unpack_from("<i", raw, 180)[0],
		"bodyparts": struct.unpack_from("<i", raw, 204)[0],
	}
	return fields


def studio_textures(raw: bytes) -> list[tuple[str, int, "np.ndarray"]]:
	"""(name, flags, RGBA) for each embedded skin; masked skins key index 255."""
	if len(raw) < 244 or raw[:4] not in (b"IDST", b"IDSQ"):
		return []
	numtextures = struct.unpack_from("<i", raw, 180)[0]
	textureindex = struct.unpack_from("<i", raw, 184)[0]
	out = []
	if numtextures <= 0 or textureindex <= 0 or textureindex + numtextures * 80 > len(raw):
		return out
	for i in range(numtextures):
		off = textureindex + i * 80
		name = raw[off:off + 64].split(b"\0")[0].decode("latin-1")
		flags, width, height, index = struct.unpack_from("<iiii", raw, off + 64)
		if width <= 0 or height <= 0 or index < 0 or index + width * height + 768 > len(raw):
			continue
		idx = np.frombuffer(raw, np.uint8, width * height, index).reshape(height, width)
		pal = np.frombuffer(raw, np.uint8, 768, index + width * height).reshape(256, 3)
		rgba = np.empty((height, width, 4), np.uint8)
		rgba[..., :3] = pal[idx]
		rgba[..., 3] = np.where((idx == 255) & bool(flags & 0x40), 0, 255)
		out.append((name, flags, rgba))
	return out
