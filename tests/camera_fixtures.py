"""Generated camera sources for the .gcam tests: glTF (with a parent dolly, a yfov pointer
animation and a STEP channel), Collada for the Assimp route, and GoldSrc entity text."""

from __future__ import annotations

import base64
import json
import math
import struct

import numpy as np


def _axis_quat(axis, deg):
	a = np.asarray(axis, float) / np.linalg.norm(axis)
	h = math.radians(deg) / 2
	return [*(a * math.sin(h)), math.cos(h)]


def dolly_gltf(seconds: float = 2.0, keys_per_s: int = 30, with_fov: bool = True) -> dict:
	"""A camera on a dolly: the dolly moves on a circle of radius 3 m (Y-up), the camera (child,
	offset 1 m up) pans about Y from 0 to 90°, and yfov goes 40° → 60° (KHR_animation_pointer)."""
	n = int(seconds * keys_per_s) + 1
	times = np.linspace(0, seconds, n)
	trans = np.stack([3 * np.cos(times), np.zeros(n), 3 * np.sin(times)], 1)
	rots = np.array([_axis_quat([0, 1, 0], 90 * t / seconds) for t in times])
	fov_t = np.array([0.0, seconds])
	fov_v = np.radians([40.0, 60.0])
	blobs = [times.astype("<f4").tobytes(), trans.astype("<f4").tobytes(), rots.astype("<f4").tobytes(),
		fov_t.astype("<f4").tobytes(), fov_v.astype("<f4").tobytes()]
	views, accs, off, buf = [], [], 0, b""
	types = ["SCALAR", "VEC3", "VEC4", "SCALAR", "SCALAR"]
	counts = [n, n, n, 2, 2]
	for b, ty, c in zip(blobs, types, counts):
		views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b)})
		acc = {"bufferView": len(views) - 1, "componentType": 5126, "count": c, "type": ty}
		if ty == "SCALAR" and len(accs) in (0, 3):
			acc["min"], acc["max"] = [float(np.frombuffer(b, "<f4").min())], [float(np.frombuffer(b, "<f4").max())]
		accs.append(acc)
		buf += b
		off += len(b)
	channels = [{"sampler": 0, "target": {"node": 0, "path": "translation"}},
		{"sampler": 1, "target": {"node": 1, "path": "rotation"}}]
	samplers = [{"input": 0, "output": 1, "interpolation": "LINEAR"}, {"input": 0, "output": 2, "interpolation": "LINEAR"}]
	if with_fov:
		channels.append({"sampler": 2, "target": {"path": "pointer",
			"extensions": {"KHR_animation_pointer": {"pointer": "/cameras/0/perspective/yfov"}}}})
		samplers.append({"input": 3, "output": 4, "interpolation": "LINEAR"})
	return {
		"asset": {"version": "2.0"},
		"extensionsUsed": ["KHR_animation_pointer"] if with_fov else [],
		"scene": 0, "scenes": [{"nodes": [0]}],
		"nodes": [{"name": "dolly", "children": [1]}, {"name": "shotcam", "camera": 0, "translation": [0, 1, 0]}],
		"cameras": [{"type": "perspective", "perspective": {"yfov": math.radians(40), "znear": 0.1, "zfar": 100.0,
			"aspectRatio": 1.5}}],
		"animations": [{"name": "flyby", "channels": channels, "samplers": samplers}],
		"buffers": [{"byteLength": len(buf), "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode()}],
		"bufferViews": views, "accessors": accs,
	}


def dolly_expected(t: float, seconds: float = 2.0, scale: float = 39.37):
	"""Engine-space camera position and forward vector for dolly_gltf at time t (independent maths)."""
	dolly = np.array([3 * math.cos(t), 0, 3 * math.sin(t)])
	cam_y = dolly + np.array([0, 1, 0])	# the dolly does not rotate
	yaw = math.radians(90 * t / seconds)
	# glTF camera looks down -Z; rotating about +Y by yaw turns -Z towards -X
	fwd_y = np.array([-math.sin(yaw), 0, -math.cos(yaw)])
	to_engine = lambda v: np.array([v[0], -v[2], v[1]])
	return to_engine(cam_y) * scale, to_engine(fwd_y)


def collada(seconds: float = 1.0) -> str:
	"""A Collada camera node moving along +X (Y-up) for one second, yfov 50°. It carries one
	triangle because Assimp refuses a scene with no meshes."""
	times = " ".join(f"{t:g}" for t in (0.0, seconds))
	mats = []
	for x in (0.0, 4.0):
		m = [1, 0, 0, x, 0, 1, 0, 1, 0, 0, 1, 5, 0, 0, 0, 1]
		mats.append(" ".join(f"{v:g}" for v in m))
	return f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Y_UP</up_axis></asset>
  <library_cameras>
    <camera id="cam-lib" name="cam">
      <optics><technique_common><perspective>
        <yfov>50</yfov><aspect_ratio>1.333333</aspect_ratio><znear>0.1</znear><zfar>200</zfar>
      </perspective></technique_common></optics>
    </camera>
  </library_cameras>
  <library_geometries><geometry id="tri" name="tri"><mesh>
    <source id="tri-p"><float_array id="tri-pa" count="9">0 0 0 1 0 0 0 1 0</float_array>
      <technique_common><accessor source="#tri-pa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <vertices id="tri-v"><input semantic="POSITION" source="#tri-p"/></vertices>
    <triangles count="1"><input semantic="VERTEX" source="#tri-v" offset="0"/><p>0 1 2</p></triangles>
  </mesh></geometry></library_geometries>
  <library_animations>
    <animation id="move">
      <source id="move-in"><float_array id="move-in-a" count="2">{times}</float_array>
        <technique_common><accessor source="#move-in-a" count="2" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
      <source id="move-out"><float_array id="move-out-a" count="32">{mats[0]} {mats[1]}</float_array>
        <technique_common><accessor source="#move-out-a" count="2" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
      <source id="move-interp"><Name_array id="move-interp-a" count="2">LINEAR LINEAR</Name_array>
        <technique_common><accessor source="#move-interp-a" count="2" stride="1"><param name="INTERPOLATION" type="name"/></accessor></technique_common></source>
      <sampler id="move-s"><input semantic="INPUT" source="#move-in"/><input semantic="OUTPUT" source="#move-out"/>
        <input semantic="INTERPOLATION" source="#move-interp"/></sampler>
      <channel source="#move-s" target="CameraNode/matrix"/>
    </animation>
  </library_animations>
  <library_visual_scenes>
    <visual_scene id="scene">
      <node id="Ground" name="Ground"><instance_geometry url="#tri"/></node>
      <node id="CameraNode" name="CameraNode">
        <matrix sid="matrix">1 0 0 0 0 1 0 1 0 0 1 5 0 0 0 1</matrix>
        <instance_camera url="#cam-lib"/>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
"""


def goldsrc_entities() -> str:
	"""A trigger_camera that moves through two corners (waiting at the first) looking at a target."""
	return """{
"classname" "worldspawn"
}
{
"classname" "trigger_camera"
"targetname" "intro_cam"
"origin" "0 0 64"
"target" "look_here"
"moveto" "c1"
"speed" "100"
"wait" "8"
}
{
"classname" "path_corner"
"targetname" "c1"
"origin" "200 0 64"
"wait" "1"
"target" "c2"
}
{
"classname" "path_corner"
"targetname" "c2"
"origin" "200 300 64"
"speed" "150"
}
{
"classname" "info_target"
"targetname" "look_here"
"origin" "100 100 0"
}
"""


def glb(doc: dict) -> bytes:
	"""The same document as a GLB with its buffer in the BIN chunk."""
	doc = json.loads(json.dumps(doc))
	uri = doc["buffers"][0].pop("uri")
	bin_ = base64.b64decode(uri.split(",", 1)[1])
	js = json.dumps(doc).encode()
	js += b" " * (-len(js) % 4)
	bin_ += b"\0" * (-len(bin_) % 4)
	total = 12 + 8 + len(js) + 8 + len(bin_)
	return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
		+ struct.pack("<II", len(bin_), 0x004E4942) + bin_)
