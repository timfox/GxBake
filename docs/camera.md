# Camera paths (`.gcam`)

`gxbake camera` turns camera animation from other tools into `.gcam`: a small big-endian file of keyframe tracks that the port samples with [`port/gc_cam.c`](../port/gc_cam.c) and hands to GX as a view matrix, a projection matrix, or GoldSrc view angles plus `fov_x`.

```sh
gxbake camera cutscenes/intro.glb -o staged/valve/media --preview check
gxbake camera maps/c1a0.bsp -o staged/valve/media            # trigger_cameras → media/cam_c1a0.gcam
gxbake camera shots/*.json --combine cam_c1a0.gcam -o staged/valve/media --max-bytes 4K
gxbake camera --inspect staged/valve/media/cam_c1a0.gcam --samples check   # what the port will see, as CSV
```

In a `gxbake.toml`:

```toml
[[camera]]
inputs = ["cutscenes/*.glb"]
output = "media"
combine = "cam_c1a0.gcam"
pos_tol = 0.25
max_bytes = "4K"
```

`gxbake budget` lists `media/cam_<map>.gcam` under each map (as information: the port has no camera window to check against).

## Sources

| Source | What is read |
| --- | --- |
| glTF / GLB | Every perspective camera node, once per animation that moves it or any of its parents (rigs and dollies work). `yfov`, `znear`, `zfar`, `aspectRatio`. An animated `yfov` through `KHR_animation_pointer`. LINEAR, STEP and CUBICSPLINE samplers. A camera with no animation gives one still clip called `static`. Orthographic cameras are skipped. |
| Collada `.dae`, FBX and other Assimp formats | The camera node's animation (the animated node whose name contains "cam", or `--node`). Assimp's Python binding does not pass camera lenses through, so FOV, near and far come from the `.dae`'s `<optics>` block, or `--fov`/`--near`/`--far` for everything else. Assimp refuses a file with no meshes at all, so a camera-only export needs any mesh in it (or use glTF). |
| JSON | Hand-written or exported keys (below). |
| CSV | One row per key (below). |
| GoldSrc `.bsp` / `.ent` | Each `trigger_camera`: it travels its `path_corner` chain at the corners' speeds, waits where a corner has `wait`, and looks at its `target` (a brush entity's centre comes from the BSP's models). This is an approximation: the game eases the speed and the turn rate every frame, and a moving target is taken at its spawn position. The port already plays these in-engine; the converter is for previewing them, or for reusing a level's camera for an attract mode or menu background. |

Everything is converted into the engine's frame: Z up, GoldSrc units, the camera looking down +X with +Y to its left and +Z up (what `AngleVectors` gives). `--axes` says what the source uses, and `auto` picks by file type:

| `--axes` | World | Camera looks down | Default scale |
| --- | --- | --- | --- |
| `engine` | Z up (GoldSrc, Source, Hammer, TrenchBroom) | +X, +Z up | 1 |
| `gltf` | Y up (glTF, Collada and most Assimp formats) | −Z, +Y up | 39.37 (metres → units) |
| `blender` | Z up | −Z, +Y up | 39.37 |
| `fbx` | Y up | +X, +Y up | 39.37 |

Y-up becomes Z-up the way Blender's glTF exporter undoes it: (x, y, z) → (x, −z, y). A level built in Blender and exported both ways therefore lines up. `--scale`, `--yaw` (about Z) and `--offset x,y,z` move the whole path afterwards.

### JSON

```json
{
  "axes": "engine",
  "fps": 30,
  "near": 4, "far": 4096,
  "cameras": [
    {"name": "intro", "loop": false, "keys": [
      {"t": 0.0, "pos": [0, 0, 64], "look_at": [500, 0, 64], "fov_x": 90},
      {"t": 2.5, "pos": [300, 100, 64], "angles": [10, 30, 0]},
      {"frame": 120, "pos": [400, 0, 100], "rot": [0, 0, 0.383, 0.924], "fov": 50},
      {"t": 4.0, "cut": true, "pos": [-800, 0, 0], "angles": [0, 180, 0]}
    ]}
  ]
}
```

- Time is `t` (seconds), or `frame` with `fps`.
- Orientation is `rot` (quaternion x, y, z, w in the source's axes), `matrix` (4×4, row-major), `angles` (GoldSrc pitch, yaw, roll in degrees; positive pitch looks down) or `look_at` (a point). A key without one keeps the previous orientation.
- FOV is `fov` / `fov_y` (vertical degrees), `fov_x` (horizontal at `aspect`, 4:3 when absent: HL's `default_fov`) or `lens` (mm, with `sensor_height`, default 24). A key without one keeps the previous FOV. The default is 90° horizontal at 4:3 (73.74° vertical).
- `cut: true` makes the camera jump to this key: the previous shot holds its last key until then.
- `near`, `far` and `scale` can sit on the document or on one camera. Sparse keys are joined by the smooth curve (below), and every key you wrote is kept.

### CSV

A header row, then one row per key. Columns: `t` (or `frame`, divided by `--rate`), `x`, `y`, `z`; then `qx qy qz qw`, or `pitch yaw roll`, or `tx ty tz` (a look-at point); optionally `fov_y` / `fov_x` / `lens`, `cut`, and `name` (rows with different names become separate clips).

### Blender

Run this in Blender's Python console or Text Editor with the scene's camera animated. It writes `camera.json` next to the .blend, sampled every frame, and `gxbake camera camera.json` reads it with `axes: blender` and metres.

```python
import bpy, json, math

scene = bpy.context.scene
cam = scene.camera
r = scene.render
aspect = (r.resolution_x * r.pixel_aspect_x) / (r.resolution_y * r.pixel_aspect_y)
keys = []
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    m = cam.matrix_world
    q = m.to_quaternion()
    d = cam.data
    fit = d.sensor_fit if d.sensor_fit != "AUTO" else ("HORIZONTAL" if aspect >= 1 else "VERTICAL")
    fov_y = 2 * math.atan(math.tan(d.angle_x / 2) / aspect) if fit == "HORIZONTAL" else d.angle_y
    keys.append({"frame": f - scene.frame_start, "pos": list(m.translation),
                 "rot": [q.x, q.y, q.z, q.w], "fov_y": math.degrees(fov_y)})
doc = {"axes": "blender", "fps": r.fps / r.fps_base, "near": cam.data.clip_start, "far": cam.data.clip_end,
       "cameras": [{"name": cam.name, "keys": keys}]}
with open(bpy.path.abspath("//camera.json"), "w") as fh:
    json.dump(doc, fh)
```

Exporting the camera as glTF from Blender works too (`File → Export → glTF`, with *Cameras* ticked); FOV animation needs Blender 4.2+ with the animation-pointer option.

## How a clip is made small

1. The clip is sampled on a fixed grid, `--rate` Hz (60 by default). A cut lands on its nearest tick and is sampled twice: the end of one shot and the start of the next.
2. Each track (position, rotation, vertical FOV) is cut down to the fewest keys whose curve stays within its tolerance of every sample: `--pos-tol` GoldSrc units (0.25), `--rot-tol` degrees (0.15), `--fov-tol` degrees (0.05). Keys are added where the error is worst until everything is within tolerance, then each key the curve can do without is removed again.
3. With `--interp auto` (the default), each track is tried both as a linear and as a smooth curve, and the one with fewer keys is kept. Straight dolly moves end up linear, orbits and hand-animated moves smooth.
4. `--encoding fixed` (the default) stores values as s16 with a per-track binary point (positions to 1/8 unit while they stay inside ±4095, quaternions to 1/16384, FOV to 1/256° or finer; the binary point is picked per track from its largest value) and times as u16 ticks of 1/rate. That is about half the size of `f32`, and the tolerance already leaves room for the rounding. A fixed-point clip can be up to 65535 ticks long (18 minutes at 60 Hz).
5. `--max-bytes` raises all three tolerances in ×1.5 steps until each file fits.

The command prints each clip's key counts, which curve each track got, and the largest error measured against the samples after rounding. For example, a 2-second glTF dolly orbit sampled at 121 points becomes 17 position, 4 rotation and 2 FOV keys (316 bytes, against 3.8 KB of raw floats), with a largest error of 0.11 units and 0.08°.

### The curves

Both `gxbake` and `gc_cam.c` evaluate exactly this:

- Before the first key: the first value. At or after the last: the last value. A looping clip wraps time first.
- Otherwise, take the last key at or before `t` (at a cut, that is the later key) and the next one.
- Linear: straight interpolation.
- Smooth: cubic Hermite, where the slope at each key is `(next − previous) / (t_next − t_previous)`. At the ends, and next to a cut, it uses the one neighbour it has. On evenly spaced keys this is Catmull-Rom.
- Rotations interpolate as quaternion components (kept in one hemisphere), then normalise.

## File layout

Big-endian. Offsets are from the start of the file; every block is 4-aligned.

| Offset | Size | Header |
| --- | --- | --- |
| 0 | 4 | `GCAM` |
| 4 | 2 | version (1) |
| 6 | 2 | clip count |
| 8 | 4 | file size |
| 12 | 4 | 0 |

Clip table, 64 bytes per clip, straight after the header:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 32 | name, NUL-padded (lowercase, `[a-z0-9_]`) |
| 32 | 4 | body offset |
| 36 | 4 | body size |
| 40 | 4 | duration, f32 seconds |
| 44 | 4 | near, f32 |
| 48 | 4 | far, f32 |
| 52 | 4 | aspect, f32 (0 = the screen's) |
| 56 | 4 | tick rate, f32 Hz |
| 60 | 1 | 0 |
| 61 | 1 | flags: 1 loop, 2 fixed |
| 62 | 1 | track count |
| 63 | 1 | 0 |

A body starts with 16 bytes per track: `u8 kind` (0 position xyz, 1 rotation quaternion xyzw, 2 vertical FOV in degrees), `u8 components`, `u8 frac` (fixed point: value = s16 / 2^frac), `u8 interp` (0 linear, 1 smooth), `u16 keys`, `u16 0`, `u32 times offset`, `u32 values offset` (both from the body start). Then come the times (u16 ticks or f32 seconds) and the values (s16 or f32, components per key), each padded to 4. A reader skips track kinds it does not know.

## In the port

`port/gc_cam.h` / `gc_cam.c` are plain C89 with no allocation. They read the file in place:

```c
#include "gc_cam.h"

gc_cam_clip_t clip;
gc_cam_pose_t pose;
float angles[3];

if( gc_cam_clip( blob, len, gc_cam_find( blob, len, "intro" ), &clip ) == 0 )
{
	gc_cam_sample( &clip, cl.time - start_time, &pose );

	/* the engine's own view set-up: */
	VectorCopy( pose.origin, refdef->vieworg );
	gc_cam_angles( &pose, angles );       /* pitch, yaw, roll for refdef->viewangles */
	refdef->fov_x = gc_cam_fov_x( pose.fov_y, 4.0f / 3.0f );

	/* or straight to GX (GC_CAM_GX): projection loaded, view matrix returned */
	/* gc_cam_load( &pose, 640.0f / 480.0f, view ); */
}
```

See [port/README.md](../port/README.md#camera-paths-gcam) for where it could hook in.

## What is and is not checked

- The tests build every source from generated files: a glTF/GLB with a camera on an animated dolly and an animated `yfov`, a Collada camera read through Assimp, JSON with cuts and look-at keys, CSV, Blender-style JSON, and a GoldSrc entity lump. They check positions and view directions against independent maths.
- `gc_cam.c` is compiled on the host with `-std=c89 -pedantic -Werror` and samples the same positions, rotations, FOV and angles as GxBake's reader, for fixed and f32, linear and smooth, with cuts and outside the clip's range. Its view and projection matrices are checked against the GX conventions and `guPerspective`'s formula. `gc_cam_load` is checked only against a recording stub of GX.
- Not yet tried: real FBX files (FBX cameras are assumed to look down +X, as Assimp's FBX importer sets them up), real Blender, Maya or Source Filmmaker exports, real HL1 maps' cameras next to the game's own playback, and anything on a GameCube or in Dolphin. Nothing in the port calls `gc_cam.c` yet.
- Source Filmmaker sessions, Source `point_viewcontrol` / `path_track` and `.vcd` choreography are not read.
