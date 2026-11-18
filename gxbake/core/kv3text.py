"""KeyValues3 text (the <!-- kv3 ... --> files and the few text DATA blocks in compiled resources)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"""
	(?P<ws>\s+|//[^\n]*|/\*.*?\*/) |
	(?P<ml>\"\"\"\n?(?P<mlbody>.*?)\n?\"\"\") |
	(?P<str>"(?P<sbody>(?:[^"\\]|\\.)*)") |
	(?P<blob>\#\[(?P<hex>[0-9A-Fa-f\s]*)\]) |
	(?P<punct>[{}\[\]=,]) |
	(?P<flag>[A-Za-z_][\w]*:(?=["\w])) |
	(?P<word>[^\s{}\[\]=,"]+)
""", re.S | re.X)


def _tokens(text: str):
	pos = 0
	n = len(text)
	while pos < n:
		m = _TOKEN.match(text, pos)
		if not m:
			raise ValueError(f"KV3 text: unexpected character at {pos}")
		pos = m.end()
		kind = m.lastgroup
		if kind == "ws":
			continue
		if kind == "ml":
			yield "str", m.group("mlbody")
		elif kind == "str":
			yield "str", bytes(m.group("sbody"), "utf-8").decode("unicode_escape", "replace") if "\\" in m.group("sbody") else m.group("sbody")
		elif kind == "blob":
			yield "blob", bytes.fromhex("".join(m.group("hex").split()))
		elif kind == "flag":
			continue	# resource:, soundevent:, panorama: ... the value follows
		else:
			yield kind, m.group(kind)


def _word(w: str):
	lw = w.lower()
	if lw == "true":
		return True
	if lw == "false":
		return False
	if lw == "null":
		return None
	try:
		return int(w)
	except ValueError:
		pass
	try:
		return float(w)
	except ValueError:
		return w


def parse(text: str):
	"""Parse KV3 text; the leading <!-- kv3 ... --> header is optional."""
	text = re.sub(r"^\s*<!--.*?-->", "", text, count=1, flags=re.S)
	toks = list(_tokens(text))
	pos = 0

	def value():
		nonlocal pos
		kind, v = toks[pos]
		pos += 1
		if kind == "punct" and v == "{":
			out = {}
			while toks[pos] != ("punct", "}"):
				kk, key = toks[pos]
				pos += 1
				if toks[pos] != ("punct", "="):
					raise ValueError(f"KV3 text: expected '=' after {key!r}")
				pos += 1
				out[key] = value()
				if toks[pos] == ("punct", ","):
					pos += 1
			pos += 1
			return out
		if kind == "punct" and v == "[":
			out = []
			while toks[pos] != ("punct", "]"):
				out.append(value())
				if toks[pos] == ("punct", ","):
					pos += 1
			pos += 1
			return out
		if kind in ("str", "blob"):
			return v
		if kind == "word":
			return _word(v)
		raise ValueError(f"KV3 text: unexpected {v!r}")

	return value()
