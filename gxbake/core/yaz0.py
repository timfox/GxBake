"""Yaz0, Nintendo's own LZ77 (every first-party GameCube game uses it).

A 16-byte header ("Yaz0", u32 decompressed size, 8 zero bytes), then groups of
one flag byte and eight items, high bit first: 1 = copy a literal byte,
0 = a back reference. A back reference is two bytes NR RR: distance RRR + 1
(up to 4096), length N + 2 (3–17); N = 0 means a third byte holds length - 0x12
(18–273).

The decoder is about thirty lines of C and runs at tens of MB/s on Gekko,
faster than the disc delivers, so a compressed file costs nothing to read and
saves disc space, disc bandwidth and, if kept compressed, ARAM.
"""

from __future__ import annotations

import struct

MAGIC = b"Yaz0"
WINDOW = 4096
MAX_LEN = 0x111
MIN_LEN = 3


def compress(data: bytes, level: int = 32) -> bytes:
	"""Greedy LZ with one-step lazy matching over hash chains of 3-byte
	prefixes; `level` caps how many earlier positions are tried per byte."""
	n = len(data)
	out = bytearray(MAGIC + struct.pack(">I", n) + bytes(8))
	head: dict[bytes, list[int]] = {}

	def find(i: int) -> tuple[int, int]:
		if i + MIN_LEN > n:
			return 0, 0
		chain = head.get(data[i:i + 3])
		if not chain:
			return 0, 0
		best_len, best_pos = 0, 0
		limit = min(MAX_LEN, n - i)
		lo = i - WINDOW
		tried = 0
		for p in reversed(chain):
			if p < lo or tried >= level:
				break
			tried += 1
			if best_len and data[p + best_len] != data[i + best_len]:
				continue
			k = 3
			while k < limit and data[p + k] == data[i + k]:
				k += 1
			if k > best_len:
				best_len, best_pos = k, p
				if k == limit:
					break
		return best_len, best_pos

	def insert(i: int) -> None:
		if i + 3 <= n:
			chain = head.setdefault(data[i:i + 3], [])
			chain.append(i)
			if len(chain) > 4 * level:
				del chain[:len(chain) - 2 * level]

	i = 0
	while i < n:
		flag_at = len(out)
		out.append(0)
		flags = 0
		for bit in range(8):
			if i >= n:
				break
			length, pos = find(i)
			if length >= MIN_LEN:
				insert(i)
				nlen, _ = find(i + 1)
				if nlen > length + 1:	# lazy: a literal now buys a longer match next
					length = 0
				else:
					dist = i - pos - 1
					if length >= 0x12:
						out += bytes((dist >> 8, dist & 255, length - 0x12))
					else:
						out += bytes((((length - 2) << 4) | (dist >> 8), dist & 255))
					for k in range(1, length):
						insert(i + k)
					i += length
					continue
			else:
				insert(i)
			flags |= 0x80 >> bit
			out.append(data[i])
			i += 1
		out[flag_at] = flags
	return bytes(out)


def decompress(blob: bytes) -> bytes:
	if blob[:4] != MAGIC:
		raise ValueError("not Yaz0")
	size = struct.unpack_from(">I", blob, 4)[0]
	out = bytearray()
	p = 16
	while len(out) < size:
		flags = blob[p]
		p += 1
		for bit in range(8):
			if len(out) >= size:
				break
			if flags & (0x80 >> bit):
				out.append(blob[p])
				p += 1
				continue
			b1, b2 = blob[p], blob[p + 1]
			p += 2
			dist = ((b1 & 15) << 8 | b2) + 1
			length = b1 >> 4
			if length == 0:
				length = blob[p] + 0x12
				p += 1
			else:
				length += 2
			start = len(out) - dist
			for k in range(length):
				out.append(out[start + k])
	return bytes(out)
