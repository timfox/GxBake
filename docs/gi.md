# Radiosity lightmaps (`gxbake gi`)

`gxbake gi` relights a compiled GoldSrc map (BSP v30) with global illumination and writes the result two ways: back into the map, and as GameCube-native lightmap pages.

```sh
gxbake gi c1a0.bsp -o build/gi                                  # bake with 3 bounces
gxbake gi c1a0.bsp -o build/gi --rad lights.rad --wad valve/    # texture lights, real texture colours
gxbake gi c1a0.bsp -o build/gi --calibrate --ao 0.5             # keep the map's brightness, add contact shadows
gxbake gi maps/*.bsp -o build/gi --budget 800K --preview        # atlas under the port's lightmap window
```

Every option is in the [command line reference](cli.md#gxbake-gi).

## What is lit

| Light | Where it comes from | How |
| --- | --- | --- |
| Point lights | `light` entities: `_light` "r g b brightness" (or "r g b", or the `light` key), `_fade` | 1/d² falloff, shadow ray per luxel |
| Spot lights | `light_spot`: `_cone` (full), `_cone2` (edge), `pitch` / `angles` / `angle` (-1 up, -2 down) | as point lights, faded between the cones |
| Sun | `light_environment`: `_light`, `pitch`, `angles` | parallel light; a luxel is lit when its ray toward the sun reaches a sky face |
| Sky | `_diffuse_light` of `light_environment`, else half the sun's colour (`sky_default`) | every hemisphere ray that reaches a sky face brings back the sky colour |
| Texture lights | `--rad lights.rad` (`TEXNAME r g b intensity` per line), `--texlight "~light1=255 255 220 800"`, or `texlights = {…}` in a manifest | each emissive face becomes small area lights (one every two luxels) with shadows, and its own luxels show its glow |
| Bounces | everything above, reflected off surfaces | radiosity between patches, below |
| Ambient occlusion | `--ao 0..1`, `--ao-radius` | short rays per luxel darken the indirect and sky light in corners and contact areas |

Direct light (point, spot, sun, texture lights) is computed per luxel. Each light only looks at faces it could reach (its light would add less than `cutoff`, 1/4096 of full bright, beyond that), that face it, and, when the map has visibility data, that lie in leaves its own leaf can see (the map's PVS, as HLRAD does).

### Bounces

Faces are cut into patches of about `--patch` units (64, so 4×4 luxels). Each patch casts `--rays` cosine-weighted hemisphere rays (96). Where they land becomes a sparse transfer matrix: patch *i* sees patch *j* in a given share of its hemisphere, and the sky in another share. Each bounce is then one sparse product: what a patch receives is the light leaving everything it sees, and what it sends on is that times its reflectance. Extra bounces cost almost nothing once the matrix exists. Luxels take the bounce light of their face's patches by bilinear interpolation, or with `--gather N` cast their own N rays (sharper contact, slower).

Reflectance is each texture's average colour, linearised from sRGB, times `--reflect` and capped at 0.9. Textures embedded in the BSP are read directly; external ones come from `--wad` files or folders. A texture that is found nowhere reflects as grey (0.5) and the report says which.

Water (`!`, `*`), `aaatrigger`, `clip`, `null` and `origin` faces do not block light. Brush entities are lit but do not cast shadows unless `--entity-shadows` is given (HLRAD's default is the same: only the world shadows). Switchable lights (a `targetname` or a `style`) are baked on into style 0; separate light styles are not written.

## Brightness

Light is computed in linear units where 1.0 is full bright. The constants follow HLRAD's conventions in spirit but are not HLRAD's code:

- a point light gives `colour × brightness × 40 × cos / d²` (a brightness-200 white light 128 units away, head on, gives about 0.5);
- a `light_environment` of brightness 255 gives 1.0 on a surface facing it;
- a texture light of intensity *I* leaves its surface with `colour × I × 0.5 / 255`.

`--light-scale`, `--sun-scale`, `--sky-scale`, `--texlight-scale` and `--exposure` adjust them. `--calibrate` fits one overall scale so that the new lighting matches the map's existing lighting lump in the median (over luxels where both are lit), which keeps a map's mood while adding bounce light and sky. Absolute brightness on a real map is the first thing to check.

## Outputs

For `map.bsp`, in the output folder:

| File | What | Who reads it |
| --- | --- | --- |
| `map.bsp` | the map with its LIGHTING lump replaced (style 0) and each face's light offset and styles updated; every other lump is unchanged | the port, as it is today |
| `map_lm.tpl` | the atlas pages; page *i* is TPL image *i* | `gc_tpl_image` / `gc_tpl_bind` (port/gc_mat.c) |
| `map.gclm` | per BSP face: page, position and size of its lightmap in the atlas | `port/gc_lmap.c` |
| `map_lm.gxm` | the TEV scale for the lightmap stage (the `lightmap` preset) | `gc_mat_parse` |
| `map_gi.json` | the report: counts, timings, bounce energy, atlas formats, sizes and PSNR, notes | you |
| `preview/` | (`--preview`) each page decoded the way GX samples it | you |

### The relit BSP

Face lightmap grids are exactly the engine's (CalcSurfaceExtents: one luxel per 16 texture units, extents + 1), so nothing changes for the port or its world baker: `Mod_GCBakeMenuLightmapAtlas` and the CapFaces atlas pack the new bytes as they packed the old ones. The bytes are `light^0.55 × 255` (HLRAD's `-gamma`, `--gamma`). The port's world stage multiplies raw BSP lightmaps by 2 in TEV (`GX_CS_SCALE_2` in `r_gx_world.c`), so light above about 0.28 is already shown overbright and light above 1.0 clips.

### The GX atlas

The atlas is built for the way Flipper samples:

- **Pages are powers of two** (`--page`, 256 by default), clamped, with linear filtering and no mips (lightmaps are magnified, never minified). The last page shrinks to the smallest power-of-two rectangle that still holds what landed on it.
- **Each face's rectangle has a luxel of padding** that repeats its edge, so bilinear filtering never reads a neighbour.
- **Rectangles start on a 4×4 block** when CMPR is allowed, so a CMPR block never mixes the colours of two faces.
- **Faces are stored only as large as they need to be** (`--no-shrink` turns this off): a face whose lightmap is flat within ±2 becomes one texel, and one whose half-resolution grid (same span, corners kept) interpolates back above `--min-db` (40 dB) is stored at half size. `gc_lmap_uv` maps the engine's coordinates onto what was stored.
- **Packing** is a bottom-left skyline, tallest first, every page tried.
- **Each page gets the smallest GX format** whose picture stays above `--min-db` against the 8-bit page: I4 or I8 when the light is grey (white lights on grey walls, the usual Half-Life case: a quarter or half the bytes of RGB565), else CMPR, else RGB565 (`--formats`, or `--format` to force one). With `--budget` (800K is the port's lightmap window), pages step down to smaller formats where that costs the least picture until the map fits.
- **Overbright.** A page holds value ÷ *scale* and the lightmap TEV stage multiplies by *scale* (1, 2 or 4, `GX_CS_SCALE_n`). `--overbright auto` picks the smallest scale under which at most 0.1 % of luxels clip; smaller scales keep more precision in the dark.
- **Tone.** With `--tone port` (the default) the value is `2 × light^0.55`: exactly what the port shows today for the relit BSP (byte × 2), so scale 2 gives the same picture as the current stage, and scale 4 keeps highlights the BSP had to clip. `--tone linear` stores the light itself (`(light / scale)^(1 / atlas-gamma)`), which is right for a renderer that multiplies linear light.

### `.gclm` layout

Big-endian. `port/gc_lmap.h` has the same description.

```
header (32 B)  "GCLM", u16 version (1), u16 flags (1 = port tone), u16 faces, u16 pages,
               u8 tev scale, u8 pad, u16 gamma × 256, u32 page table offset,
               u32 face table offset, 8 bytes reserved
face table     one 12-byte entry per BSP face, in face order:
               u16 page (0xFFFF = none), u16 x, u16 y, u16 w, u16 h, u8 engine w, u8 engine h
               (x, y: first stored texel, inside the padding)
page table     pages × { u16 width, u16 height }
```

## In a manifest

```toml
[[gi]]
inputs = ["maps/*.bsp"]
output = "maps"         # under [build] output
bounces = 3
rad = "lights.rad"
wads = ["wads"]
ao = 0.5
calibrate = true
budget = "800K"         # per map: the default
texlights = { "~light1" = "255 255 220 800" }
```

Keys are the command's options with underscores (`light_scale`, `ao_radius`, `atlas_gamma`, …), plus `bsp = false` / `atlas = false` to skip an output. Each map is its own budget group (the port has one map's lightmaps in memory at a time), so maps never compete for bytes. A map is baked again only when it, its `.rad`, its WADs or the rule change; `--dry-run` lists the maps without baking.

## Speed

Rays are cast against a bounding-volume hierarchy of the map's triangles. When a C compiler is on the PATH (or in `$CC`), the inner loop (`gxbake/core/_rt.c`) is compiled once into the user cache folder and runs on every core; otherwise numpy does the same job about 100 times slower. On a synthetic 3072-unit open map with 150 boxes, 31 lights and a sun (190,000 luxels, 14,000 patches, 0.5 AO) a bake takes about 6 seconds with the compiled caster on 4 cores; with numpy, direct light alone took 2 minutes. `GXBAKE_NO_CC=1` forces numpy. The packaged app has no compiler, so it uses numpy.

## Port side

`port/gc_lmap.c` (C89, host-tested against GxBake's Python reader and UV maths):

```c
gc_lmap_t lm;
gc_lmap_face_t f;
uint16_t pw, ph;
float u, v;

gc_lmap_parse( &lm, gclm_bytes, gclm_size );
if( gc_lmap_face( &lm, face_index, &f ) && !gc_lmap_page_size( &lm, f.page, &pw, &ph ))
{
	/* s, t: the engine's lightmap coordinates of a vertex, in luxels from the texture mins */
	gc_lmap_uv( &f, pw, ph, s, t, &u, &v );	/* -> GX_VA_TEX1 */
}
```

Bind page `f.page` of `map_lm.tpl` with `gc_tpl_bind` on the lightmap texmap and set the lightmap stage's scale from `lm.tev_scale` (`gc_mat_lightmap_stage`). Faces are best drawn grouped by page, so a page is loaded once. Where it hooks into the HL1 port (the atlas is built in `mod_bmodel.c` and bound by `R_GXBindLightmapAtlas` in `r_gx_world.c`) is in [port/README.md](../port/README.md#radiosity-lightmap-atlas-gclm).

## What is not checked yet

- **No real map has been baked.** Every test map is a box built in code; the shadows, bounces, sky, texture lights, AO, PVS culling and file layouts are tested on those. Brightness constants, spot and sun directions and `light_environment` keys on real Half-Life maps, and HLRAD look-alike, are unverified. Run with `--calibrate` first.
- **Nothing has run in the port, on hardware or in Dolphin.** The relit BSP uses the existing path and should need nothing. The atlas path needs the port to load the TPL and `.gclm`, set lightmap UVs from `gc_lmap_uv` and use the `.gxm` scale; `gc_lmap.c` is only host-tested.
- Not done: phong (smooth) shading across faces, light styles, opaque or translucent brush entities beyond on/off shadows, `_falloff` modes other than inverse square, and studio model lighting (a light grid).
