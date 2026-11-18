"""Writers for the GameCube ports' baked map files, as the ports' bake scripts lay them out.

GCwd v1 is Day of Defeat's tools/bake_worlddl_goldsrc.py, v2 with flag 1 the HL1 / Lost Coast
scripts/bake_worlddl_goldsrc.py, v2 with flag 0 and a SKY1 block Lost Coast's tools/bake_worlddl.py.
"""

from __future__ import annotations

import struct


def _pad32(b: bytearray) -> None:
	b += b"\0" * ((-len(b)) % 32)


def _chunk(version: int, mats: list[int], fans: int, lm_page: int) -> bytes:
	"""One chunk: four positions and one 4-vertex fan per material, repeated `fans` times."""
	stride_fmt = ">HHhh" if version < 2 else ">Hbbb4BhhBB"
	head = 8 if version < 2 else 16
	out = bytearray(head + 12 * len(mats))
	_pad32(out)
	pos_ofs = len(out)
	for p in ((0, 0, 0), (64, 0, 0), (64, 64, 0), (0, 64, 0)):
		out += struct.pack(">hhh", *p)
	_pad32(out)
	for i, m in enumerate(mats):
		dl = bytearray()
		for _ in range(fans):
			dl += struct.pack(">BH", 0xA0 | 6, 4)
			for v in range(4):
				if version < 2:
					dl += struct.pack(stride_fmt, v, 0xFFFF, v * 64, 0)
				else:
					dl += struct.pack(stride_fmt, v, 0, 0, 64, 200, 200, 200, 255, v * 64, 0, 8, 8)
		_pad32(dl)
		struct.pack_into(">HHII", out, head + 12 * i, m, 0, len(out), len(dl))
		out += dl
	if version < 2:
		struct.pack_into(">HHI", out, 0, 4, len(mats), pos_ofs)
	else:
		struct.pack_into(">HHII", out, 0, 4, len(mats), pos_ofs, lm_page + 1)
	return bytes(out)


def world_dl(version: int = 2, flags: int = 1, names=("floor", "wall"), chunks=((0, (0, 1), 2), (3, (1,), 1)),
		pages: int = 1, sky: bool = False, hash_fn=None) -> bytes:
	"""chunks: (cluster, material indices, fans per material). Chunk k gets lightmap page k when k < pages."""
	from gxbake.core.lumps import fnv1a32

	hash_fn = hash_fn or fnv1a32
	clusters = 4
	body = bytearray(64)
	if sky:
		body += struct.pack(">4s4fi3BBff", b"SKY1", 1.0, 2.0, 3.0, 16.0, 2, 10, 20, 30, 1, 100.0, 900.0)
		_pad32(body)
	ofs_mats = len(body)
	for n in names:
		if version < 2:
			body += struct.pack(">IBBH", hash_fn(n), 0, 0, 0)
		else:
			body += struct.pack(">IIBBH", hash_fn(n), 0, 1, 0x04 if n.startswith("env") else 0, 0)
	_pad32(body)
	ofs_nodes = len(body)
	body += struct.pack(">Hhh", 0, -2, -3)	# one node: clusters 0 and 1
	_pad32(body)
	ofs_planes = len(body)
	body += struct.pack(">ffff", 0.0, 0.0, 1.0, 0.0)
	_pad32(body)
	blobs = [_chunk(version, list(m), fans, k if k < pages and version >= 2 else -1)
		for k, (_cl, m, fans) in enumerate(chunks)]
	ofs_chunkdir = len(body)
	run = 0
	for (cl, _m, _f), blob in zip(chunks, blobs):
		body += struct.pack(">II3h3hHH", run, len(blob), 0, 0, 0, 64, 64, 0, cl, 0)
		run += len(blob)
	_pad32(body)
	ofs_pvsdir = len(body)
	for c in range(clusters):
		body += struct.pack(">I", 0 if c else 0xFFFFFFFF)
	_pad32(body)
	ofs_pvs = len(body)
	body += b"\x0f"
	_pad32(body)
	ofs_chunks = len(body)
	for blob in blobs:
		body += blob
	_pad32(body)
	if version >= 2:
		for p in range(pages):
			# page p: solid colour, red ramps with p (RGB565, tiled; a solid page is the same tiled or not)
			px = ((min(31, 8 * (p + 1))) << 11) | (32 << 5) | 4
			body += struct.pack(">H", px) * 1024
	struct.pack_into(">4sHH13I", body, 0, b"GCwd", version, flags, clusters, len(blobs), len(names), 1, 1,
		ofs_mats, ofs_nodes, ofs_planes, ofs_chunkdir, ofs_pvsdir, ofs_pvs, ofs_chunks, len(body))
	struct.pack_into(">I", body, 60, max(len(b) for b in blobs))
	return bytes(body)


def lightfaces(faces: int = 3, lighting: bytes = bytes(range(48))) -> bytes:
	"""HL1 scripts/bake_worlddl_goldsrc.py bake_lightfaces layout."""
	recs = bytearray()
	for i in range(faces):
		flags = 1 if i == faces - 1 else 0	# last face is sky
		lo = 0xFFFFFF if flags else i * 12
		recs += struct.pack(">8f4iI4s", 1, 0, 0, 0, 0, 1, 0, 0, -16, -16, 32, 16, (flags << 24) | lo, b"\0\xff\xff\xff")
	body = bytearray(32) + recs
	_pad32(body)
	lit = len(body)
	body += lighting
	_pad32(body)
	struct.pack_into(">4s5I", body, 0, b"GClf", 1, faces, 32, lit, len(lighting))
	return bytes(body)
