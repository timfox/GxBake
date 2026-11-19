"""Ray casting against triangle soups, in numpy, for the lightmap baker.

A bounding-volume hierarchy (median split on the longest axis of the
triangle centroids, a few triangles per leaf) is walked for thousands of
rays at once: every step tests all live (ray, node) pairs against their
boxes, expands inner nodes into their two children and tests leaf triangles
with Möller–Trumbore. There is no per-ray Python loop, so a map's worth of
shadow and bounce rays runs at numpy speed (a few hundred thousand rays a
second on one core, more for short rays).

Triangles are two-sided. Each carries the index of the face it came from, so
a hit can be turned back into a lightmap luxel or a radiosity patch.

When a C compiler is on the PATH (or in $CC), `_rt.c` is compiled once into a
small shared library in the user cache folder and the same BVH is walked
there instead, one ray at a time front to back, on every core (ctypes drops
the GIL). That is 50-100× the numpy walker. GXBAKE_NO_CC=1 turns it off;
without a compiler (the packaged app) the numpy walker is used.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-6
_LIB: "ctypes.CDLL | None" = None
_LIB_TRIED = False
_LIB_NOTE = ""


def _cache_dir() -> Path:
	base = os.environ.get("XDG_CACHE_HOME") or (os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else "") \
		or str(Path.home() / ".cache")
	return Path(base) / "gxbake"


def native() -> "ctypes.CDLL | None":
	"""The compiled ray caster, building it on first use; None when there is no compiler."""
	global _LIB, _LIB_TRIED, _LIB_NOTE
	if _LIB_TRIED:
		return _LIB
	_LIB_TRIED = True
	if os.environ.get("GXBAKE_NO_CC"):
		_LIB_NOTE = "GXBAKE_NO_CC is set"
		return None
	src = Path(__file__).with_name("_rt.c")
	cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
	if not src.is_file() or not cc:
		_LIB_NOTE = "no C compiler" if src.is_file() else "_rt.c is not packaged"
		return None
	code = src.read_bytes()
	ext = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
	tag = hashlib.sha256(code + sys.platform.encode() + str(ctypes.sizeof(ctypes.c_void_p)).encode()).hexdigest()[:16]
	lib = _cache_dir() / f"rt-{tag}{ext}"
	try:
		if not lib.is_file():
			lib.parent.mkdir(parents=True, exist_ok=True)
			with tempfile.TemporaryDirectory() as td:
				tmp = Path(td) / f"rt{ext}"
				subprocess.run([cc, "-O3", "-std=c99", "-shared", "-fPIC", str(src), "-o", str(tmp), "-lm"],
					check=True, capture_output=True, timeout=120)
				shutil.move(str(tmp), str(lib))
		dll = ctypes.CDLL(str(lib))
	except (OSError, subprocess.SubprocessError) as exc:
		_LIB_NOTE = f"could not build the C ray caster ({exc.__class__.__name__})"
		return None
	f = dll.gx_rt_trace
	dp = ctypes.POINTER(ctypes.c_double)
	ip = ctypes.POINTER(ctypes.c_int64)
	f.argtypes = [dp, dp, dp, dp, dp, ip, ip, ip, ip, ctypes.c_int64, dp, dp, dp, ctypes.c_int64, ctypes.c_int, dp, ip]
	f.restype = ctypes.c_int
	_LIB = dll
	_LIB_NOTE = f"C ray caster ({lib.name})"
	return _LIB


def backend() -> str:
	native()
	return "native" if _LIB is not None else f"numpy ({_LIB_NOTE})"


def _ptr(a: "np.ndarray", kind):
	return a.ctypes.data_as(ctypes.POINTER(kind))


@dataclass
class Hits:
	t: "np.ndarray"	# distance along the (unit) direction; tmax where nothing was hit
	tri: "np.ndarray"	# triangle index, -1 for a miss

	@property
	def hit(self) -> "np.ndarray":
		return self.tri >= 0


class Mesh:
	"""Triangles (N×3×3) with a BVH over them."""

	def __init__(self, tris: "np.ndarray", leaf: int = 6):
		tris = np.asarray(tris, np.float64).reshape(-1, 3, 3)
		self.count_tris = len(tris)
		lo_n, hi_n, left, right, start, count = [], [], [], [], [], []
		order: list[np.ndarray] = []
		placed = 0
		if len(tris):
			cent = tris.mean(1)
			tlo, thi = tris.min(1), tris.max(1)
			stack = [(np.arange(len(tris)), -1, 0)]	# (indices, parent, which child)
			while stack:
				idx, parent, side = stack.pop()
				me = len(lo_n)
				lo_n.append(tlo[idx].min(0))
				hi_n.append(thi[idx].max(0))
				left.append(-1)
				right.append(-1)
				start.append(0)
				count.append(0)
				if parent >= 0:
					(left if side == 0 else right)[parent] = me
				c = cent[idx]
				ext = c.max(0) - c.min(0)
				if len(idx) <= leaf or ext.max() <= 0:
					start[me] = placed
					count[me] = len(idx)
					order.append(idx)
					placed += len(idx)
					continue
				ax = int(np.argmax(ext))
				half = len(idx) // 2
				part = np.argpartition(c[:, ax], half)
				stack.append((idx[part[half:]], me, 1))
				stack.append((idx[part[:half]], me, 0))
		self.order = np.concatenate(order) if order else np.zeros(0, np.int64)
		t = tris[self.order]
		self.v0 = t[:, 0]
		self.e1 = t[:, 1] - t[:, 0]
		self.e2 = t[:, 2] - t[:, 0]
		self.lo = np.array(lo_n, np.float64).reshape(-1, 3)
		self.hi = np.array(hi_n, np.float64).reshape(-1, 3)
		self.left = np.array(left, np.int64)
		self.right = np.array(right, np.int64)
		self.start = np.array(start, np.int64)
		self.count = np.array(count, np.int64)

	@property
	def nodes(self) -> int:
		return len(self.lo)

	def _tri_t(self, o: "np.ndarray", d: "np.ndarray", k: "np.ndarray") -> "np.ndarray":
		"""Möller–Trumbore, many rays against one triangle each: distance or inf."""
		e1, e2 = self.e1[k], self.e2[k]
		p = np.cross(d, e2)
		det = np.einsum("ij,ij->i", e1, p)
		ok = np.abs(det) > 1e-12
		inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
		s = o - self.v0[k]
		u = np.einsum("ij,ij->i", s, p) * inv
		q = np.cross(s, e1)
		v = np.einsum("ij,ij->i", d, q) * inv
		t = np.einsum("ij,ij->i", e2, q) * inv
		ok &= (u >= -EPS) & (v >= -EPS) & (u + v <= 1 + EPS) & (t > EPS)
		return np.where(ok, t, np.inf)

	def _trace_chunk(self, o, d, tmax, any_hit: bool):
		n = len(o)
		best_t = tmax.astype(np.float64).copy()
		best = np.full(n, -1, np.int64)
		if not self.nodes or not n:
			return best_t, best
		with np.errstate(divide="ignore"):
			inv = 1.0 / np.where(d == 0, 1e-300, d)
		r = np.arange(n)
		nd = np.zeros(n, np.int64)
		while r.size:
			o_r, iv = o[r], inv[r]
			t1 = (self.lo[nd] - o_r) * iv
			t2 = (self.hi[nd] - o_r) * iv
			tn = np.minimum(t1, t2).max(1)
			tf = np.maximum(t1, t2).min(1)
			keep = (tf >= np.maximum(tn, 0)) & (tn <= best_t[r])
			if any_hit:
				keep &= best[r] < 0
			r, nd = r[keep], nd[keep]
			if not r.size:
				break
			cnt = self.count[nd]
			leaf = cnt > 0
			if leaf.any():
				lr, ln, c = r[leaf], nd[leaf], cnt[leaf]
				rr = np.repeat(lr, c)
				first = np.cumsum(c) - c
				kk = np.repeat(self.start[ln] - first, c) + np.arange(int(c.sum()))
				t = self._tri_t(o[rr], d[rr], kk)
				ok = t < best_t[rr]
				if ok.any():
					rr, kk, t = rr[ok], kk[ok], t[ok]
					np.minimum.at(best_t, rr, t)
					win = t <= best_t[rr]
					best[rr[win]] = kk[win]
			inner = ~leaf
			ri, ni = r[inner], nd[inner]
			r = np.concatenate((ri, ri))
			nd = np.concatenate((self.left[ni], self.right[ni]))
		return best_t, best

	def _native(self, o, d, tm, any_hit: bool) -> "Hits | None":
		lib = native()
		if lib is None or not len(o):
			return None
		if not hasattr(self, "_c"):
			self._c = [np.ascontiguousarray(a) for a in (self.v0, self.e1, self.e2, self.lo, self.hi)] + \
				[np.ascontiguousarray(a, np.int64) for a in (self.left, self.right, self.start, self.count)]
		v0, e1, e2, lo, hi, le, ri, st, cn = self._c
		o = np.ascontiguousarray(o)
		d = np.ascontiguousarray(d)
		tm = np.ascontiguousarray(tm)
		out_t = np.empty(len(o))
		out_k = np.empty(len(o), np.int64)
		dp, ip = ctypes.c_double, ctypes.c_int64
		threads = max(1, min(os.cpu_count() or 1, len(o) // 4096))
		bounds = np.linspace(0, len(o), threads + 1).astype(np.int64)

		def run(i):
			a, b = int(bounds[i]), int(bounds[i + 1])
			if b <= a:
				return 0
			return lib.gx_rt_trace(_ptr(v0, dp), _ptr(e1, dp), _ptr(e2, dp), _ptr(lo, dp), _ptr(hi, dp), _ptr(le, ip),
				_ptr(ri, ip), _ptr(st, ip), _ptr(cn, ip), self.nodes, _ptr(o[a:], dp), _ptr(d[a:], dp), _ptr(tm[a:], dp),
				b - a, int(any_hit), _ptr(out_t[a:], dp), _ptr(out_k[a:], ip))

		if threads == 1:
			codes = [run(0)]
		else:
			with ThreadPoolExecutor(threads) as ex:
				codes = list(ex.map(run, range(threads)))
		if any(codes):
			return None
		tri = np.where(out_k >= 0, self.order[np.maximum(out_k, 0)] if len(self.order) else -1, -1)
		return Hits(out_t, tri)

	def trace(self, orig, dirs, tmax=1e9, any_hit: bool = False, chunk: int = 16384, use_native: bool = True) -> Hits:
		"""Nearest hit (or, with any_hit, some hit) of each ray within tmax. dirs need not be unit;
		t is in units of |dir|."""
		o = np.asarray(orig, np.float64).reshape(-1, 3)
		d = np.asarray(dirs, np.float64).reshape(-1, 3)
		tm = np.broadcast_to(np.asarray(tmax, np.float64), (len(o),))
		if use_native:
			h = self._native(o, d, tm, any_hit)
			if h is not None:
				return h
		ts = np.empty(len(o))
		tri = np.empty(len(o), np.int64)
		for a in range(0, len(o), chunk):
			b = min(len(o), a + chunk)
			ts[a:b], k = self._trace_chunk(o[a:b], d[a:b], tm[a:b], any_hit)
			tri[a:b] = np.where(k >= 0, self.order[np.maximum(k, 0)] if len(self.order) else -1, -1)
		return Hits(ts, tri)

	def occluded(self, orig, dirs, tmax) -> "np.ndarray":
		return self.trace(orig, dirs, tmax, any_hit=True).hit


def cosine_hemisphere(n: int, seed: int = 1) -> "np.ndarray":
	"""n directions about +Z, cosine-weighted, stratified on a jittered grid (fewer rays for the same noise)."""
	rng = np.random.default_rng(seed)
	k = max(1, int(np.ceil(np.sqrt(n))))
	cells = np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2)
	cells = cells[rng.permutation(len(cells))[:n]]
	u = (cells + rng.random(cells.shape)) / k
	r = np.sqrt(u[:, 0])
	phi = 2 * np.pi * u[:, 1]
	return np.stack((r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0, 1 - u[:, 0]))), -1)


def basis(normals: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
	"""Two tangents for each unit normal (an orthonormal frame with the normal as Z)."""
	nrm = np.asarray(normals, np.float64).reshape(-1, 3)
	helper = np.where(np.abs(nrm[:, 2:3]) < 0.9, np.array([[0.0, 0, 1]]), np.array([[1.0, 0, 0]]))
	t1 = np.cross(helper, nrm)
	t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
	t2 = np.cross(nrm, t1)
	return t1, t2


def hemisphere_rays(normals: "np.ndarray", local: "np.ndarray", rotate_seed: int = 0) -> "np.ndarray":
	"""(points × samples × 3) world directions: the local cosine set turned into each normal's frame,
	each point's set spun by its own random angle so neighbouring luxels do not share a pattern."""
	nrm = np.asarray(normals, np.float64).reshape(-1, 3)
	t1, t2 = basis(nrm)
	ang = np.random.default_rng(rotate_seed).random(len(nrm)) * 2 * np.pi
	c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
	a1 = c * t1 + s * t2
	a2 = -s * t1 + c * t2
	return (local[None, :, 0:1] * a1[:, None] + local[None, :, 1:2] * a2[:, None] + local[None, :, 2:3] * nrm[:, None])
