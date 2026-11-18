"""Source 2 entity lumps (.vents_c): each entity's key/values as plain strings.

An entity is stored either as KV3 ("keyValues3Data": {version 1, values {...}})
or, before that, as a small binary record: version 1, then (murmur2 hash, type,
value) per key and (hash, name, type, value) per attribute. The hash is
MurmurHash2 of the lowercased key with seed 0x31415926; keys whose name isn't in
KNOWN_KEYS come back as "#<hash in hex>".

Ported from ValveResourceFormat (MIT, see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import vres

SEED = 0x31415926

# the keys a GoldSrc entity can use, and the ones the map converter reads
KNOWN_KEYS = """
classname targetname target parentname origin angles scales model skin body spawnflags rendercolor renderamt
rendermode renderfx disableshadows solid health speed wait delay message netname killtarget team teamnum
skyname skyname_lod lightgroup color brightness brightness_units range attenuation0 attenuation1 attenuation2
falloff innerconeangle outerconeangle _light _lightHDR _lightscaleHDR _ambient _ambientHDR _cone _inner_cone
_constant_attn _linear_attn _quadratic_attn _fifty_percent_distance _zero_percent_distance pitch yaw roll
directlight indirectlight bouncescale castshadows shadowtexturewidth shadowtextureheight bakelight
ambient_color ambient_occlusion skycolor skyintensity skytexture skytexturescale skyambientbounce
sunlightminbrightness lower_hemisphere_is_black enabled startdisabled start_disabled fademindist fademaxdist
fadescale minfadedist maxfadedist lightmapstatic world_layer_name layername hammeruniqueid
uniform_scale scale modelscale vscripts thinkfunction sound soundname message_noise radius volume
fogcolor fogcolor2 fogstart fogend fogmaxdensity fogenable fogblend fogdir fogexponent
landmark map changelevel nextmap ignoreuncaughterrors maxs mins boxmins boxmaxs
point0 point1 point2 starttouch filtername damage damagetype dmg lip movedir movedistance
locked_sound unlocked_sound noise1 noise2 effects friction gravity mass physdamagescale
""".split()


def murmur2(key: str, seed: int = SEED) -> int:
	data = key.lower().encode("utf-8")
	m = 0x5BD1E995
	h = (seed ^ len(data)) & 0xFFFFFFFF
	n = len(data) // 4
	for (k,) in struct.iter_unpack("<I", data[:n * 4]):
		k = (k * m) & 0xFFFFFFFF
		k ^= k >> 24
		k = (k * m) & 0xFFFFFFFF
		h = ((h * m) & 0xFFFFFFFF) ^ k
	tail = data[n * 4:]
	if len(tail) == 3:
		h ^= tail[2] << 16
	if len(tail) >= 2:
		h ^= tail[1] << 8
	if tail:
		h ^= tail[0]
		h = (h * m) & 0xFFFFFFFF
	h ^= h >> 13
	h = (h * m) & 0xFFFFFFFF
	h ^= h >> 15
	return h


_BY_HASH = {murmur2(k): k.lower() for k in KNOWN_KEYS}


def _fmt(v) -> str:
	if isinstance(v, bool):
		return "1" if v else "0"
	if isinstance(v, float):
		return f"{v:g}"
	if isinstance(v, (list, tuple, np.ndarray)):
		return " ".join(_fmt(x.item() if isinstance(x, np.generic) else x) for x in v)
	if isinstance(v, bytes):
		return v.hex()
	return str(v)


def _read_binary(b: bytes) -> dict[str, str]:
	ver, nkeys, nattrs = struct.unpack_from("<III", b, 0)
	if ver != 1:
		raise ValueError(f"entity data version {ver} is not supported")
	p = 12
	out: dict[str, str] = {}

	def value() -> str:
		nonlocal p
		t = struct.unpack_from("<I", b, p)[0]
		p += 4
		if t == 0x06:	# bool
			v = "1" if b[p] else "0"
			p += 1
		elif t == 0x01:	# float
			v = _fmt(struct.unpack_from("<f", b, p)[0])
			p += 4
		elif t == 0x22:	# float64
			v = _fmt(struct.unpack_from("<d", b, p)[0])
			p += 8
		elif t == 0x09:	# Color32
			v = " ".join(str(c) for c in b[p:p + 4])
			p += 4
		elif t in (0x05, 0x25):	# int, uint
			v = str(struct.unpack_from("<i" if t == 0x05 else "<I", b, p)[0])
			p += 4
		elif t in (0x1A, 0x21):	# int64, uint64
			v = str(struct.unpack_from("<q" if t == 0x1A else "<Q", b, p)[0])
			p += 8
		elif t in (0x03, 0x27):	# Vector, QAngle
			v = _fmt(struct.unpack_from("<3f", b, p))
			p += 12
		elif t == 0x1E:	# CString
			end = b.index(b"\0", p)
			v = b[p:end].decode("utf-8", "replace")
			p = end + 1
		else:
			raise ValueError(f"entity field type {t:#x} is not supported")
		return v

	for _ in range(nkeys):
		h = struct.unpack_from("<I", b, p)[0]
		p += 4
		v = value()
		out[_BY_HASH.get(h, f"#{h:08x}")] = v
	for _ in range(nattrs):	# named attributes; the converter has no use for them
		p += 4
		p = b.index(b"\0", p) + 1
		value()
	return out


def entity_values(kv: dict) -> dict[str, str] | None:
	"""One m_entityKeyValues entry as {lowercase key: string value}, or None without a classname."""
	if "keyValues3Data" in kv:
		d = kv["keyValues3Data"] or {}
		if int(d.get("version", 1)) != 1:
			raise ValueError(f"entity data version {d.get('version')} is not supported")
		out = {str(k).lower(): _fmt(v) for k, v in (d.get("values") or {}).items()}
	else:
		raw = kv.get("m_keyValuesData")
		raw = bytes(raw) if not isinstance(raw, bytes) else raw
		out = _read_binary(raw) if raw else {}
	return out if "classname" in out else None


def read_lump(res: vres.Resource) -> tuple[list[dict[str, str]], list[str]]:
	"""(entities, child lump paths) of one entity lump."""
	d = res.keyvalues() or {}
	ents = []
	for kv in d.get("m_entityKeyValues") or []:
		e = entity_values(kv)
		if e is not None:
			ents.append(e)
	return ents, [str(c) for c in d.get("m_childLumps") or []]


def load_entities(path: Path, roots: list[Path], warnings: list[str] | None = None) -> list[dict[str, str]]:
	"""Entities of a lump and every child lump it names, depth first."""
	from .vtex import find_compiled

	out: list[dict[str, str]] = []
	seen: set[str] = set()
	todo = [Path(path)]
	while todo:
		p = todo.pop(0)
		key = str(p).lower()
		if key in seen:
			continue
		seen.add(key)
		try:
			ents, children = read_lump(vres.load(p))
		except (OSError, ValueError) as exc:
			if warnings is not None:
				warnings.append(f"entity lump {p.name}: {exc}")
			continue
		out += ents
		for c in children:
			cp = find_compiled(c, roots)
			if cp is None:
				if warnings is not None:
					warnings.append(f"entity lump {c} not found")
				continue
			todo.append(cp)
	return out
