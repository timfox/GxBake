# GxBake

A desktop tool (and CLI) for converting assets to native GameCube GX renderer formats, previewing exactly what Flipper will sample, and exporting a staged `media/` tree.

GxBake wraps the GX pipeline from the Xash3D GameCube port (`scripts/build-gamecube-disc.py`). The encoders are ported unchanged, so the output is byte-identical to the disc builder's (a test checks this).

## What it converts

| Source | Output | Pipeline function |
| --- | --- | --- |
| PNG / TGA / BMP / JPG / DDS / TIFF / PSD / PCX / QOI, Source `.vtf`, Quake II `.wal`, Blizzard `.blp` | `media/<pack>.gct`: CMPR, power-of-two, mip chain, 1-bit alpha | `gx_cmpr_encode_blocks`, `wt_encode` mip loop |
| Skybox (`<name>rt/bk/lf/ft/up/dn`) | `media/sky_<name><side>.cmp`: 128² CMPR, 8 KiB per side | `gx_cmpr_encode`, `build_sky_cmpr_overlays` |
| Panoramas (equirectangular, Radiance `.hdr`), cubemap crosses and strips, six named faces, flat photos, Source 2 cubemap `.vtex_c` | the same six `media/sky_<name><side>.cmp`, rendered in the engine's orientation | `skybox.build` |
| WAD3 (miptex, plus the qpic pictures and console fonts of gfx.wad and fonts.wad, index 255 see-through), Quake WAD2 | `media/<wad>.gct` | `wt_read_miptex`, `wt_encode` |
| BSP v30, Quake BSP v29 | `media/wt_<map>.gct` (world textures resolved through the map's WADs, or embedded with the Quake palette) and its sky | `build_world_texture_packs`, `wt_read_wads` |
| Source BSP v19–21 and 25 (`VBSP`) | `maps/<map>.bsp` rebuilt as BSP30, plus its `media/wt_<map>.gct` and `media/sky_<sky>*.cmp` | `source_map.convert` |
| Source 2 map (`<map>.vmap_c` or `maps/<map>/world.vwrld_c`) | the same `maps/<map>.bsp`, world pack and sky | `s2map.convert` |
| Studio `.mdl` | `gc_studio/<name>.mdl` (lean mesh) and `gc_studio/<name>t.mdl` (downsampled skins) | `lean_studio_mesh_bytes`, `lean_studio_tex_companion_bytes` |
| 3D models via Assimp (OBJ, FBX, glTF/GLB, DAE, 3DS, Blend, MD2/MD3/MD5, IQM, B3D, SMD, MS3D and ~25 more) | `models/<name>.mdl` (studio MDL v10 with bones, sequences and paletted skins), then the same `gc_studio/<name>.mdl` + `<name>t.mdl` as a studio model | `models.import_model`, `models.build_studio_mdl`, then the two functions above |
| USD stages via FreeUSD (`.usda`, `.usd`, `.usdz`; skinned meshes, UsdPreviewSurface textures, UsdSkel animation) | the same `models/` and `gc_studio/` outputs as an Assimp model | `usd.import_usd`, then `models.build_studio_mdl` |
| Source `.mdl` (v44–49) + `.vvd` + `.vtx`, Quake `.mdl` | the same `models/` and `gc_studio/` outputs as an Assimp model | `source_mdl.import_source`, `quake.import_alias`, then `models.build_studio_mdl` |
| Source 2 `.vmdl_c` / `.vmesh_c` (with their `.vmat_c`, `.vtex_c` and embedded or `.vagrp_c` animations) | the same `models/` and `gc_studio/` outputs | `vmdl.import_vmdl`, `vanim.model_sequences`, then `models.build_studio_mdl` |
| DMX model sources (`.dmx`, binary 1–5 and 9 or keyvalues2, from Blender Source Tools, the SFM or ModelDoc), with their joints, skin weights and `animationList` clips | the same `models/` and `gc_studio/` outputs (each vertex follows its heaviest joint; flexes are dropped) | `dmx_model.import_dmx`, then `models.build_studio_mdl` |
| Source 2 `.vtex_c` | an entry in `media/<pack>.gct`, like any image | `vtex.load`, then as an image |
| Source 2 `.vsnd_c` (PCM, MS-ADPCM, MP3) | `sound/<name>.wav`: Microsoft PCM at up to 22050 Hz, loop kept as a cue point | `vsnd.to_port_wav` |
| Sprite `.spr` (Half-Life, Quake), Quake II `.sp2` | `sprites/<name>.spr`: a Half-Life sprite, downsampled, keeping thin alpha-test strokes | `quake.hl_sprite_bytes`, `lean_hl_spr_bytes` |
| Particle systems: Source `.pcf`, Source 2 `.vpcf_c` | `sprites/<system>.spr` for each system that renders a texture: a parallel sprite from its spritecard (the sprite sheet's first sequence, or the VTF's frames), additive when the material is, index-alpha for soft smoke, alpha-test otherwise. Emitters and operators have no GoldSrc equivalent | `particles.convert` |
| ROSE Online `.zms` meshes, `.zsc` object lists, `.chr` character lists (with `.zmd` skeletons, `.zmo` motions, `.dds` textures) | the same `models/` + `gc_studio/` studio models, one per object or character | `rose.import_rose`, then as above |
| Panda3D `.egg`, `.bam` (and `.pz`), Toontown actors and props (with their animation files, `.jpg` + `_a.rgb` textures); `.pandactor` multi-part actors | the same `models/` + `gc_studio/` studio models | `panda.import_panda`, then as above |
| RuneScape caches: OSRS / RS2 JS5 (`main_file_cache.dat2` + `.idx*`) and 2004-era (`main_file_cache.dat`) | the same `models/` + `gc_studio/` studio models, one per NPC (or model), with face colours baked into a palette skin and label-group animations as bones | `runescape.import_runescape`, then as above |
| Final Fantasy XI (PC): ROM `.DAT` models (NPCs, monsters, equipment, weapons), `.ffximdl` manifests, `.spw` / `.bgw` sounds, zone DATs (MZB/MMB) | studio models with skeleton and motions; sound effects as fitted `sound/ffxi/<name>.wav`, music as `media/<name>.gcpcm`; zones as OBJ + PNG (`gxbake ffxi --obj`) for a map editor | `ffxi.import_ffxi`, `ffxi.sound_to_wav`, `ffxi.zone_to_obj` |
| Final Fantasy VII (PC): field models (`.hrc` + `.rsd` + `.p` + `.tex`, `.a` animations), `.ff7mdl` manifests, lone `.p` models, battle models (`<base>aa` skeletons with their parts, textures and `<base>da` animation packs); `.lgp` archives through `gxbake extract` | studio models, one per skeleton | `ff7.import_ff7` |
| Video (AVI, MP4…) | `media/<name>.gcvid` (RGB565 delta) and `.gcpcm` | `build_gcvid_companion`, `build_gcpcm_companion` |
| Audio (WAV, MP3, OGG, FLAC, AIFF, VOC, Opus…) | `media/<name>.gcpcm` (48 kHz stereo, s16 big-endian); PCM WAV needs no ffmpeg | `build_gcpcm_companion` |

It also opens the outputs (`.gct`, `.cmp`, `.gcvid`, `.gcpcm`) to inspect them. Every run writes `gxbake-manifest.json` with the settings, every output and its size, and the share of GameCube main RAM it uses.

## 3D model import (Assimp)

Any format [Assimp](https://github.com/assimp/assimp) reads is baked into a GoldSrc studio model, because that is what the GameCube port draws. GxBake triangulates the mesh, converts Y-up sources to GoldSrc's Z-up (+Z forward becomes +X forward), builds triangle strips, and turns each material's diffuse/base-colour texture into an 8-bit paletted skin (power-of-two, cut-out alpha as masked index 255).

**Skeletons and animation.** When the file has a skin or animations, the node hierarchy becomes studio bones: skin joints, the nodes that own meshes, animated nodes, and their parents. Studio skinning is rigid, so each vertex follows its heaviest joint (the inspector says when blended weights were snapped). Every animation is sampled at **Animation fps** into a looping sequence, stored as the RLE value streams studiomdl writes. Each bone with vertices gets a hitbox, and each sequence gets a bounding box that covers its motion. Node animation without a skin (doors, turrets) works the same way. Tick **Bake static** to turn this off and get one root bone in the bind pose.

**Clip files.** Animations in files named `<model>@<clip>.<ext>` next to the model (the Unity convention, e.g. `soldier@run.fbx`, or `.bvh` / `.md5anim`) are added as sequences named after the clip, matched to bones by name. These files are not listed as assets of their own.

**Textures** are found next to the model, by basename when the exporter wrote an absolute path, in `textures/`, `tex/` or `maps/`, under another extension when the referenced `.psd`/`.tif` has a `.png` beside it, or inside the file (`.glb` chunks, glTF data URIs and buffers). A material with no texture becomes a solid skin of its colour.

Settings: **Scale** (glTF and most FBX are in metres, so try 39.37), **Source up axis**, **Import skin max**, **Animation fps**, **Bake static**, **Merge clip files**, and whether to also write the full `models/<name>.mdl`. The preview renders the imported mesh next to the geometry decoded back from the lean MDL. Expand the model to compare each material with its companion skin, or pick a `▶` sequence to see its middle frame posed the way the engine decodes it. The inspector warns past the studio limits (128 bones, 2048 vertices, 20000 triangles, 100 skins).

Not carried over: morph/vertex animation (MD2/MD3 import their first frame), blended skin weights beyond the heaviest bone, root motion, and multiple skin families.

Assimp comes from the `assimp-py` wheel, which bundles the native library, so there is nothing to install separately. Skeletons and animations need assimp-py 1.2, which ships for Python 3.12 and newer; on 3.10 and 3.11 models import static. It is part of the `gui` and `dev` extras, or `pip install -e ".[models]"` for the CLI alone.

## Memory budgets

`gxbake budget <staged gamedir>` checks every map against the windows the GameCube port actually gives each kind of data at map load, not just the 24 MiB / 16 MiB totals:

| Window | Budget | What GxBake counts |
| --- | --- | --- |
| World textures, MEM1 pool | 448 KiB (`GC_WT_POOL_BYTES`) | the `.gct` table, every texture's resident low mips and the hot ring, using the engine's own split rule (`Mod_GCWorldPackSplitDim`) |
| World textures, ARAM store | 6 MiB, 512 entries | the whole `media/wt_<map>.gct` blob |
| Studio models, MEM1 arena | 1 MiB per map | `gc_studio/<name>.mdl` + `<name>t.mdl` for every model the entities name, plus the Half-Life classname defaults (`monster_barney` → `barney.mdl`) and anything given with `--include 'models/v_*.mdl'` |
| Entity sounds | 20 KiB per file, 128 KiB MEM1 cache, 2 MiB ARAM | the PCM bytes of `.wav` files the entities name; one over the per-file cap is refused and never plays |
| Lightmaps | 800 KiB atlas | the BSP lighting lump (an upper bound) |

It also lists sprites, the sky and the BSP size, anything referenced but not staged, and the whole tree against the 1.36 GiB disc. A map is `WARN` when the texture pool only fits at a softer split (big textures draw from low mips until they stream in) and `OVER` when something will be refused. The exit code is 1 when anything is over (`--strict` adds warnings), so it can gate CI. `--json` prints the same report as data, and `--budget studio_arena=1.5M` (repeatable), `--pool` and `--split` change the limits.

`gxbake fit media/wt_c1a0.gct -o media/wt_c1a0.gct` makes a world pack fit: it drops the top mip level of the largest textures until the pack fits the pool at split 64 and the ARAM store. Level 1 is already encoded, so there is no re-encode and no new artefacts. Without `-o` it is a dry run that lists what it would halve. `gxbake convert --fit-world` does the same while converting BSPs.

### Sound effects

`gxbake sfx sound/ -o staged/valve/sound` shrinks WAVs under the port's per-file cap: `Sound_LoadWAV` refuses PCM data over 20 KiB in gameplay (48 KiB with `--menu`, when the menu runs the server). It only accepts 8/16-bit Microsoft PCM, so it tries, least lossy first: cutting data after the loop end (never played), stereo to mono, then the smallest step down the 22 kHz/11 kHz × 16/8-bit ladder that fits (8-bit gets TPDF dither, resampling is band-limited). Loop points are kept and rescaled. Anything that still does not fit is reported, or cut with a 10 ms fade with `--truncate`. `--in-place`, `--cap`, `--min-rate`, `--keep-stereo` and `--no-8bit` adjust it, and without `-o` it is a dry run.

### Texture packs

`gxbake pack` edits `.gct` packs directly, without going back to the source assets: `list` (with sizes, `--sort size`), `extract` to PNG exactly as the GameCube samples it (`--all-levels` for every mip), `build` a pack from images, `add` images (an entry of the same name is replaced in place), `remove` entries by name or glob, and `merge` packs (a later pack wins on a name clash). Names are the image file names, lowercased and cut to the engine's 15 characters, and GxBake warns past the engine's 512 entries.

### Smarter packing

`gxbake squeeze <staged gamedir>` measures optional packing techniques on the real files: detail-aware mip trimming, optimal and deadband animation runs, indexed (CI4/CI8 + TLUT) skins, DSP ADPCM sound, s16 vertices, CMPR lightmaps and Yaz0. It prints bytes before and after per memory window. `-o OUT` writes the ones the port reads today, `--emit` also writes the formats that need port support, and `--only`/`--skip` pick techniques. See [docs/packing.md](docs/packing.md) for the numbers on sample assets, the ranking and what the port must add.

### Disc usage

`gxbake disc <staged gamedir>` shows what the tree costs on a disc: every file rounded up to 2 KiB ISO9660 sectors (the disc builder exposes game data through ISO9660), the share of the 1.36 GiB disc, and a breakdown by directory. It also lists byte-identical files, textures stored in more than one `.gct` (same CMPR bytes; each map loads its own pack, so this only costs disc), and files under `models/`, `gc_studio/` and `sound/` that no map names. The game DLL may still precache those, so treat that list as a place to look, and pass `--include` for models you know are needed. `--json` gives the same data.
### Studio models

`gxbake slim models/barney.mdl` shows where a studio model's bytes go (animation, meshes, bones, sequences, skins) and how big the lean `gc_studio/` mesh is, since that is what fills the 1 MiB studio arena. For NPCs and view models the animation is usually the largest part, so:

- `--step 2` keeps every second frame of each sequence and scales its fps, so it lasts as long as before (the engine interpolates between frames). Events move with their frames.
- `--stub 'death*'` reduces matching sequences to their first frame, and `--keep 'idle*'` protects others.

Sequences are never removed, so the game DLL's sequence numbers and names stay valid. Only animation stored in the model itself is changed; sequences in external `<name>01.mdl` seqgroup files are left alone. Without `-o` it is a dry run. `gxbake convert --mdl-anim-step 2` applies the frame step to every `gc_studio/<name>.mdl` it writes.

## Skies from panoramas

`gxbake sky` (or **Add Skybox…** in the app) turns one image, or six, into the six sides the port's `R_GXSkyLoad` reads: `media/sky_<name>{rt,bk,lf,ft,up,dn}.cmp`, each exactly 128×128 CMPR with no mips (8 KiB; the engine refuses any other length and keeps all six in a 48 KiB block). Maps use the sky through worldspawn `"skyname" "<name>"`.

```sh
gxbake sky sunset.hdr -o staged/valve                       # equirectangular panorama
gxbake sky alps_cross.png -o staged/valve --name alps        # 4:3 or 3:4 cross
gxbake sky cube/ -o staged/valve --preview alps.png          # six faces in a folder
gxbake sky vista.jpg -o staged/valve --fov 120 --env 512     # a flat photo, plus gfx/env TGAs
```

The input layout is guessed from its shape, or set with `--layout`:

| Layout | Input |
| --- | --- |
| `equirect` | a 2:1 latitude/longitude panorama; `.hdr` files are tone-mapped (auto exposure, `--exposure` adds stops) |
| `cross` | a 4:3 horizontal or 3:4 vertical cross, front face in the middle |
| `strip` | six square faces in a row or column, in the order right, left, up, down, front, back (+X −X +Y −Y +Z −Z) |
| `faces` | six files: `<name>rt/bk/lf/ft/up/dn` (GoldSrc sides, used as they are), `<name>_right/left/front/back/up/down` (or `top`/`bottom`), or `<name>_px/nx/py/ny/pz/nz` |
| `photo` | any other single image: it covers `--fov` degrees of the horizon (180 by default), mirrored copies fill the rest so there is no seam, and its top and bottom rows fade into the zenith and nadir |

Every source is resampled by direction, so each side comes out the way `R_GXDrawSkyBox` draws it (the same tables as `gl_warp.c`; a test checks against them). GoldSrc's names are not where you would guess: facing `ft` (−Y), `rt` (+X) is on your left and `lf` on your right, and `up`/`dn` are turned 90°. The front of the source (the panorama's centre) faces `--yaw`, in the engine's yaw angles: 270 (`ft`) by default, 0 `rt`, 90 `bk`, 180 `lf`. Each side is rendered from 4×4 lookups per texel, and sources larger than that are pre-filtered, so a 16K panorama does not alias.

`--preview` writes the decoded sides as a cross, front in the middle, to check seams and orientation; `--env SIZE` also writes `gfx/env/<name><side>.tga` for PC builds and the disc builder. When `convert` or the app is given a 2:1 image at least 1024 wide, a `.hdr`, a 6:1 strip or a cross with empty corners, it makes a sky of it too (`--no-sky-detect`, or the **Skies from panoramas** settings, turn that off; smaller images stay textures). Expand a sky in the app to see each side or **all sides** as a cross.

## Source, Quake and Quake II formats

These are read by GxBake itself (no Assimp), and each becomes what the GameCube port loads.

**Source models** (`.mdl` v44–49, with the `.vvd` and `.dx90.vtx` beside it). `.mdl` files are told apart by their header: GoldSrc studio models (IDST v10) keep the lean/companion path, Source and Quake models are imported. A Source model brings its bones, the root LOD of the first model in each bodypart, family-0 skins and every sequence whose animation lives in the `.mdl`. Skins are found through the `.vmt` files named by the model, under the nearest `materials/` folder (`$cdmaterials`, `$basetexture`, `patch` materials); `$alphatest`/`$translucent` keep the alpha as cut-outs, otherwise the spec/phong mask in alpha is dropped. Source and GoldSrc share axes and units, so the source up axis is ignored.

Studio skinning is rigid, so a vertex weighted between two bones would tear at the joint. GxBake adds **helper bones**: each one is parented to the vertex's main bone and follows the dual-quaternion blend of the two bones at that weight (quarters, or halves if quarters don't fit under 128 bones), so elbows and shoulders bend. They make the animation block larger; untick **Helper bones for blended weights** (or pass `--no-blend-bones`) to snap vertices to their heaviest bone instead. Not carried over: flexes, other LODs and bodygroups, IK rules, animations stored in `.ani` files, and delta (gesture) sequences.

**VTF textures** (7.0–7.5): DXT1/3/5, ATI1N/ATI2N, the 8888/888/565/5551/4444 orders, I8/IA88/A8 and 16-bit float (tone-mapped). The largest mip of the first frame and face becomes a GCWT entry, and six `<name>rt/bk/lf/ft/up/dn.vtf` make a skybox.

**Source maps** (`.bsp`, VBSP versions 19 to 21 and 25, as in Half-Life 2, Counter-Strike: Source, Team Fortress 2 and most mods) are rebuilt as the port's BSP30, so they load like any GoldSrc map:

- Faces keep Source's own BSP tree (nodes, leafs, PVS); each face is split so no lightmap is wider than 16 luxels, and tool faces (nodraw, clip, trigger, hint and so on) are dropped. Sky faces are drawn by the port's sky, whose six sides come from `materials/skybox/<skyname>rt…dn`.
- Displacements become triangles (at most 4 levels of detail; `--map-disp-power` lowers it) and are added to the leafs they sit in.
- Lightmaps are resampled from Source's lightmap of each face (LDR when the map has it, else HDR), with every light style, and mapped from Source's display curve to GoldSrc's.
- Each material's `$basetexture` is looked up in the map's pakfile, then the folders given with `--game-dir`, then the `*_dir.vpk` files in them. It is scaled to about one texel per world unit (at most 256 pixels by default, `--map-texture-max`) and quantized to an 8-bit miptex embedded in the BSP. `$alphatest` textures become `{` (masked) and water becomes `!`.
- Clip hulls 1 to 3 are built from the player-solid brushes, expanded to each hull's size with bevels, and the displacements get thin collision wedges under their surface. `--no-map-hulls` skips this for a quick look.
- Entities pass through. Team spawns become `info_player_deathmatch` (and an `info_player_start` is added), and each `prop_static` becomes a `cycler_sprite` naming `models/<name>.mdl`, which you convert from the Source model (`--no-map-props` leaves them out).

- Overlays (`info_overlay`) become decal faces laid half a unit above the face they sit on, lit from that face's lightmap.

Strata Source's version 25 (Momentum Mod), with its 32-bit indices and float bounds, is read too (its overlays are not yet).

Not carried over: detail props (sprite grass), decals placed by `infodecal`, cubemaps, water reflections, areaportals and occluders. Detail brushes collide in hulls 1 to 3 but not in hull 0, because Source keeps them out of the render tree. Big maps can exceed BSP30's 32767 clipnodes, which the converter reports.

## Source 2 assets

GxBake reads Source 2's compiled resources (the `_c` files inside a Source 2 `.vpk`, which `gxbake extract` unpacks) without any Valve tools. The container, KeyValues3 (binary versions 1 to 5, the older VKV3 header and the text form) and the older NTRO-described blocks are all read; LZ4 and Valve's block compression are decoded in Python, and Zstandard needs Python 3.14 or `pip install gxbake[valve]`.

- **Models** (`.vmdl_c`, or a lone `.vmesh_c`): the skeleton, LOD 0 of the default mesh group (embedded meshes or `.vmesh_c` files next to it), meshoptimizer and Zstandard vertex/index buffers, packed normals, and up to eight weights per vertex reduced to the two heaviest, which get helper bones like Source 1 models. Each draw call's `.vmat_c` is followed to its colour texture (`g_tColor` and friends); tool materials are skipped. Animations come from the model's own blocks, from the `.vagrp_c` groups it lists and the `.vanim_c` files they name, decoded segment by segment (every compressed decoder VRF knows) and sampled at **Animation fps**. Delta (additive) animations are skipped, and a sequence too long for studio's 16-bit animation offsets is halved in frame rate until it fits. Resource paths are looked up under the nearest folder above the file that holds `materials/`, `models/`, `maps/` or `sounds/`, case-insensitively.
- **Textures** (`.vtex_c`): DXT1/DXT5, BC4/BC5 (ATI1N/ATI2N), BC6H (through Pillow, clamped to LDR), BC7, ETC2, ETC2_EAC, R11 and RG11 EAC, the 8-bit orders, I8/IA88/A8, 16- and 32-bit float (tone-mapped), and textures stored as PNG, JPEG or WebP; YCoCg-encoded colour and compressed mips are handled. The top mip goes into the pack; a cubemap becomes a sky instead (its six faces rendered into the port's sides).
- **Sounds** (`.vsnd_c`): 8/16-bit PCM and Microsoft ADPCM decode directly, MP3 through `ffmpeg`. The result is resampled to 22050 Hz at most and written as `sound/<name>.wav`; `gxbake sfx` then fits it under the port's per-file cap.

- **Maps** (`<map>.vmap_c`, or `maps/<map>/world.vwrld_c` with its `worldnodes/` and `entities/`) become BSP30 like Source 1 maps, but a Source 2 map has no BSP tree, brushes or lightmaps the port can use, so GxBake builds them:
  - Every scene object and aggregate fragment of the world nodes is placed in world space (LOD 0, tool materials dropped). Coplanar neighbouring triangles with the same texture mapping are merged into convex faces and cut at 16 luxels; texture vectors come from the mesh UVs, with textures scaled to about a texel per unit (`--map-texture-max`).
  - The tree is an axis-aligned split of the faces into empty leafs, all visible from each other.
  - Collision comes from `world_physics` (convex hulls become brushes, triangle meshes become thin slabs under each face; trigger and NPC-clip shapes are skipped), or from the render faces when there is none. At most 4000 pieces are kept, largest first.
  - Light is recomputed per luxel from `light_environment`, the point and spot lights and an ambient term, with shadows traced through the collision (`--no-map-shadows` skips the tracing). Source 2's own lightmaps are not read.
  - Entity lumps (KeyValues3 and the older hashed binary records) are read. Brush entities whose model lives under `maps/<map>/entities/` become inline models `*N` with their own faces and hulls (triggers keep only their bounds), props become `cycler_sprite` entities naming `models/<name>.mdl`, spawns are adapted as for Source 1, and the `env_sky` cubemap is rendered into the port's six sky sides.

Not converted yet: Source 2 lightmaps and light probes, water volumes, decals and overlays, physics props, morphs, cloth, bodygroups other than the default, and animation-graph (AG2) clips. Readers follow [ValveResourceFormat](https://github.com/ValveResourceFormat/ValveResourceFormat); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Quake**: alias models (`.mdl`, IDPO v6) convert their first frame and first skin, since studio models can't morph vertices (the inspector says how many frames there were). BSP v29 world textures and WAD2 miptex carry no palette, so GxBake reads `gfx/palette.lmp` from the game folder or from a `pak*.pak` there. Quake sprites (v1) become Half-Life sprites with that palette and index 255 transparent.

**Quake II**: `.wal` textures use the palette in `pics/colormap.pcx`, and `.sp2` sprites become Half-Life sprites built from their PCX frames.

Without the palette files, a greyscale ramp is used and the asset warns. Neither palette is shipped with GxBake, since both are game data.

**Archives.** Game data usually sits in `.pak`, `.vpk`, `.pk3` or (Panda3D) `.mf` files. `gxbake extract pak0.pak -o unpacked` unpacks one (`--match 'models/*'` to pick), then convert the folder. For Source, pass the `_dir.vpk`; its numbered `_000.vpk` chunks are read from beside it.
## ROSE Online models

GxBake reads ROSE Online's own formats directly (Assimp knows none of them): ZMS meshes, ZMD skeletons, ZMO motions, ZSC object lists and LIST_NPC.CHR character lists, plus DDS textures through Pillow. Point it at an extracted client tree (the folder that holds `3DDATA`) or at single files:

- **`LIST_NPC.CHR`**: one asset per character. Its meshes come from `PART_NPC.ZSC` next to it, its skeleton and motions from the paths in the CHR, so each NPC or monster becomes an animated studio model with sequences named after its motion slots (`stop`, `walk`, `attack`, `hit`, `die`, `run`…; idle and walk loop, the rest play once).
- **`LIST_*.ZSC`**: one asset per object (map decorations, buildings, items), with every part placed by its ZSC transform.
- **`.zms`**: a single mesh. A same-named texture next to it is used; a skinned mesh picks up the `.zmd` beside it and the `.zmo` motions in its folder (or `<mesh>@<clip>.zmo` clips).
- **`.rosemdl`**: a small JSON file for models nothing lists, such as a player avatar assembled from body, arm, foot and face parts: `{"skeleton": "3DDATA/AVATAR/MALE.ZMD", "parts": [{"zsc": "3DDATA/AVATAR/LIST_MBODY.ZSC", "object": 1}, {"mesh": "x.zms", "texture": "x.dds", "bone": 4}], "motions": {"walk": "3DDATA/MOTION/AVATAR/EMPTY_WALK_M1.ZMO"}}`.

Paths inside ZSC/CHR files are matched case-insensitively against the `3DDATA` root. Adding a whole folder lists the CHR characters and ZSC objects and leaves out the loose meshes, textures and `PART_*.ZSC` they draw from. Output names are `<list>_<index>_<name>`, e.g. `npc_33_deck16.mdl`. `--rose-entry N` (repeatable) limits a CLI run to some entries.

ROSE is Z-up like GoldSrc. Settings: **Units per metre** (39.37 keeps real size) and **Turn** (ROSE models face -Y; 90° makes them face GoldSrc's +X). Skinning is rigid (each vertex follows its heaviest bone), double-sided materials get back faces, and alpha-tested materials become masked skins. Not converted yet: per-part ZSC animations, UV/alpha ZMO tracks, effects, and maps; see [docs/rose-online.md](docs/rose-online.md).

## Panda3D and Toontown models

GxBake reads Panda3D's formats itself, so Toontown's actors and props convert like any other model. See [docs/panda3d.md](docs/panda3d.md) for the details.

- **`.egg`** (Panda's text format) is parsed directly: vertex pools, polygons, textures, materials, characters, joints with vertex membership, and animation tables. `.egg.pz` (zlib) works too.
- **`.bam`** (Panda's binary format, which is what Toontown ships) changes with every engine version, so Panda3D itself turns it into egg first: `pip install panda3d` (or `gxbake[panda]`), or a `bam2egg` program on `PATH` or named by `GXBAKE_BAM2EGG`. The standalone executable does not bundle Panda3D, so it needs `bam2egg`.
- **`.mf`** multifiles (Toontown's `phase_3.mf`, `phase_4.mf`…) unpack with `gxbake extract phase_3.mf -o tt`. Encrypted members are skipped with a message.

Animations sit in their own files, as Panda actors load them. Next to a model, GxBake adds the animation-only files that share its name: `panda-model.egg` takes `panda-walk.egg`, Toontown's `tt_a_chr_dgm_shorts_legs_1000.bam` takes `tt_a_chr_dgm_shorts_legs_neutral.bam`, `suitA-mod.bam` takes `suitA-walk.bam`, and `<model>@<clip>` works as for Assimp. In an extracted Toontown tree it also looks in the same folder under the other `phase_N` folders, because a model's animations are spread across phases. Animation files are not listed as assets of their own, and neither are the textures in `maps/` folders when a folder scan finds Panda models.

Textures are found the way Panda's model-path finds them: Toontown eggs say `phase_3/maps/x.jpg` relative to the install root, so GxBake looks next to the model and in each folder above it, case-insensitively, and under another image extension if the file was converted. A texture's `alpha-file` (Toontown pairs `x.jpg` with `x_a.rgb`) becomes the skin's alpha, and SGI `.rgb` images also convert on their own.

A **`.pandactor`** JSON file joins a multi-part actor, such as a toon's legs, torso and head, into one studio model. Each part's root joints hang from a joint of an earlier part, and every part's animations are found as above (or listed under `anims`). The file and joint names below show the shape; check them against your own data:

```json
{"parts": [{"model": "phase_3/models/char/tt_a_chr_dgm_shorts_legs_1000.bam"},
           {"model": "phase_3/models/char/tt_a_chr_dgm_shorts_torso_1000.bam", "attach": "joint_hips"},
           {"model": "phase_3/models/char/dogM_heads_1000.bam", "attach": "def_head"}]}
```

Settings: **Units per Panda unit** (12 by default, since Toontown's unit is a foot and GoldSrc's is an inch) and **Turn** (Panda models face +Y; −90° makes them face GoldSrc's +X). Lower LODs (`<SwitchCondition>` groups) and collision geometry (`<Collide>`, collision `<ObjectType>`s) are left out. Skinning is rigid, double-sided (`<BFace>`) polygons get back faces, and untextured polygons share one small skin of their colours. Not converted yet: Toontown's `.dna` street and building layouts, morph (blend shape) animation, texture transforms and multitexturing beyond the first texture.
## RuneScape caches

GxBake reads RuneScape's client cache in place; nothing needs extracting first. Point it at `main_file_cache.dat2` (or the folder holding it and its `.idx` files):

- **JS5 caches** (Old School RuneScape, and RS2 from 2006 on): one asset per named NPC. Each NPC's models are merged, its recolours, retextures and width/height scale applied, and its stand, walk and run sequences baked as `idle`, `walk` and `run`.
- **2004-era caches** (`main_file_cache.dat` with `idx0`–`idx4`): models only, one asset per model id.
- `--rs-entry npc:<id>` or `--rs-entry model:<id>` (repeatable) picks what to convert instead of every NPC. `gxbake rs <cache> --find guard` lists NPC ids, names, models and sequences.

All four model encodings are read (the original one and the 0xFFFF, 0xFFFE and 0xFFFD ones). RuneScape models have a 16-bit HSL colour per face instead of a texture, so the colours go into a small palette skin with one 4x4 block per colour (one CMPR block, so the GameCube's compression keeps them exact) and every face samples its block. Textured faces get the texture's sprite as their own skin. Output names are `rs_npc<id>_<name>` or `rs_model<id>`.

Animations move groups of vertices, not bones, so every vertex group gets a studio bone and each frame's rotate and translate steps become bone poses. Settings: **Units per model unit** (0.35 makes a ~200-unit player about 70 units tall; 128 units are one tile) and **Turn**. Not converted yet: items, objects (locs) and maps, skeletal ("animaya") animations of newer OSRS models, translucent faces (drawn opaque) and texture animation. See [docs/runescape.md](docs/runescape.md).

## Final Fantasy XI and VII

Both read the PC releases. Nothing from either game is in the repository; the tests build their own files. See [docs/final-fantasy.md](docs/final-fantasy.md) for the formats and what is still guessed.

**FFXI.** Point GxBake at a `ROM*/<dir>/<file>.DAT` (or a whole ROM folder: only DATs with a mesh chunk are listed). NPC and monster DATs carry their own skeleton, textures and motions and convert on their own. Player equipment, faces and hair are drawn on the race skeleton from another DAT: pass it with `--ffxi-skeleton`, or write a `.ffximdl` manifest naming the skeleton, the parts and the motion DATs (paths, or `id:<file id>` with `"root"` set to the FINAL FANTASY XI folder). Textures (DXT1/DXT3 and 256-colour) become skins, mirrored meshes get their second half on the mirror bones, two-bone vertices follow the heavier bone, and motion chunks with the same name prefix (upper and lower body) are joined into one sequence. Output names are `ffxi_<rom>_<dir>_<file>`, so the many `1.DAT`s don't collide. `.spw` effects decode to WAV and go through the same fitting as `gxbake sfx`; `.bgw` music becomes `media/<name>.gcpcm` (`--ffxi-music-wav` writes WAVs instead). `gxbake ffxi <DAT>` lists a DAT's chunks, textures, meshes and motions (`--textures DIR` writes the textures as PNG); `gxbake ffxi "<FINAL FANTASY XI>" --id 1234` resolves a file id through VTABLE/FTABLE. Settings: **Units per FFXI unit** (39.37, taking a unit as a metre) and **Turn**. Zones: `gxbake ffxi <zone DAT> --obj DIR` writes the placed zone as one OBJ with an MTL and PNG textures, in GoldSrc axes and units, to bring into a map editor and compile as a BSP (a whole zone is far over the studio arena, and GxBake has no BSP compiler); `--prefabs` writes each prefab (trees, buildings, rocks) as its own OBJ, which GxBake then converts to studio props like any OBJ. Not converted: zone collision and lighting, cloth physics, particle effects, and the ATRAC3 sounds of the PS2/360 releases.

**FFVII.** Unpack `char.lgp` with `gxbake extract char.lgp -o char`, then convert a skeleton. An `.hrc` names its bones and the `.rsd` resources on each; an `.rsd` names the `.p` polygon model and its `.tex` textures. The game picks animations in its field scripts, so name them in a `.ff7mdl` manifest (`{"skeleton": "aaaa.hrc", "animations": ["acfe.a"]}`); the first one also sets the rest pose. `gxbake ff7 aaaa.hrc --anims` lists the `.a` files beside it with the right bone count. Untextured groups keep their vertex colours in a palette skin, and triangles are turned to agree with their normals, since FF7 winds them both ways. Adding the extracted folder lists the skeletons and manifests and leaves out the `.p` parts. Battle models come from `battle.lgp` (`gxbake extract battle.lgp -o battle`): add a `<base>aa` skeleton (Cloud is `rtaa`) and its parts (`rtam`, `rtan`, …), textures (`rtac`, …) and body animations from the `rtda` pack come with it, one sequence per pack slot (`anim00`, `anim01`, …). Settings: **Units per FF7 unit** (2.5, field), **Battle units per FF7 unit** (auto fits each battle model to 72 units tall; set a number to keep a cast in proportion) and **Turn**. Not converted: battle weapons and weapon animations, battle locations, the world map and battle scenes.

## The app

- Drag files or a whole gamedir onto the window, or use **Add Files…** / **Add Folder…**.
- Expand a WAD, BSP, model, texture pack or skybox to preview each texture in it.
- **Add Skybox…** makes the six sky sides from a panorama, cubemap cross or strip, photo, or six face images (see [Skies from panoramas](#skies-from-panoramas)).
- The middle panes show the source next to the decoded GameCube result, with PSNR and byte counts in the **Inspector** tab.
## USD import (FreeUSD)

`.usda`, `.usd` and `.usdz` files are read with [FreeUSD](https://github.com/gopexllc/freeusd), a clean-room OpenUSD implementation, and baked into the same studio model as an Assimp import. GxBake opens and composes the stage (sublayers, references, payloads), then reads every visible mesh with its normals and UVs, the UsdPreviewSurface colour texture of each bound material (GeomSubsets included), UsdSkel skeletons with the skinned points on their heaviest joint, the SkelAnimation on the stage timeline, and xformOp animation of rigid meshes. `<model>@<clip>.usda` files beside the model add sequences. The stage's `upAxis` and `metersPerUnit` set the axis swap and scale, so a centimetre Y-up character arrives at its real size in inches; **Scale** multiplies on top.

FreeUSD builds a C++ extension, so it is an optional extra: `pip install -e ".[usd]"` (needs CMake and a C++17 compiler; the first build takes several minutes). Binary `.usdc` layers, and `.usdz` packages that hold one, aren't decoded by FreeUSD yet: save them as USDA first (`usdcat -o model.usda model.usdc`). [docs/usd.md](docs/usd.md) lists what is read and the FreeUSD gaps GxBake works around.

- The **Settings** tab holds the optimisation knobs: max texture size, mipmaps, alpha mode and cutoff, resize filter, packing, sky size, model skin size, sprite scale, GCVID size and frame rate, extra WAD search folders, and an encode cache. Changes re-render the preview.
- **Convert Selected** or **Convert All** writes into the output folder. The status bar shows the total against the 24 MiB of main RAM.

## Run from source

```sh
pip install -e ".[gui]"
gxbake            # opens the app
gxbake convert path/to/valve -o out --cache ~/.cache/gxbake
gxbake inspect path/to/file.gct
gxbake validate staged/valve   # same checks as the disc builder
gxbake budget staged/valve     # per-map memory report
gxbake extract hl2_misc_dir.vpk -o unpacked --match 'models/*' --match 'materials/*'
gxbake pack list staged/valve/media/wt_c1a0.gct --sort size
gxbake sky sunset.hdr -o staged/valve --name sunset
```

Every command and option is listed in the [command line reference](docs/cli.md), and [the porting workflow](docs/workflow.md) shows how they fit together, from unpacking a game to a staged tree that fits the GameCube's memory. The reference is generated from the CLI itself: after changing an option, run `python -m gxbake.clidoc` (a test fails until you do).

Video and audio need `ffmpeg` on `PATH`, next to the executable, or set with `GXBAKE_FFMPEG`.

## Standalone executable

```sh
pip install -e ".[dev]"
pyinstaller packaging/gxbake.spec    # → dist/GxBake (GxBake.exe on Windows, GxBake.app on macOS)
```

The executable opens the app when run with no arguments or with file paths. With a subcommand (`convert`, `inspect`, `validate`, `budget`, `fit`, `sfx`, `disc`, `slim`, `pack`, `sky`, `extract`) it runs as the CLI. The `build` GitHub workflow runs the tests and builds the executable for Linux, Windows and macOS on every push.

## Layout

- `gxbake/core/`: the pipeline as a library, with no Qt dependency (`budget`, `sfx`, `disc`, `studio_slim`, `packtool`, `skybox`, `cmpr`, `gct`, `goldsrc`, `images`, `studio`, `models`, `skeleton`, `sprite`, `media`, `audit`, `project`, `particles`), the native readers (`source_mdl`, `vtf`, `quake`, `archives`, `rose`, `panda`, `runescape`, `ffxi`, `ff7`, `dmx`, `dmx_model`, `usd` for FreeUSD), the map converters (`source_bsp`, `source_map`, `bsp30`), and for Source 2 `vres`, `vents`, `s2map`, `kv3`, `kv3text`, `valvecomp`, `vtex`, `vsnd`, `vmdl`, `vanim`.
- `gxbake/gui/`: the PySide6 front end.
- `gxbake/cli.py`: the command line.
- `tests/`: builds synthetic WAD, BSP, MDL and SPR files (and Source, Quake and Quake II ones in `game_fixtures.py`, a Source map in `s1_fixtures.py`, DMX models and particle systems in `dmx_fixtures.py`, Source 2 resources in `s2_fixtures.py` and a Source 2 map in `s2map_fixtures.py`, USD stages in `usd_fixtures.py`, RuneScape caches in `rs_fixtures.py`, Final Fantasy files in `ffxi_fixtures.py` and `ff7_fixtures.py`), so no game data is needed. Set `GXBAKE_UPSTREAM_SCRIPT=/path/to/scripts/build-gamecube-disc.py` to also check byte-for-byte parity with the disc builder.
