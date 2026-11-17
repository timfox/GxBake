# GxBake

A desktop tool (and CLI) for converting assets to native GameCube GX renderer formats, previewing exactly what Flipper will sample, and exporting a staged `media/` tree.

GxBake wraps the GX pipeline from the Xash3D GameCube port (`scripts/build-gamecube-disc.py`). The encoders are ported unchanged, so the output is byte-identical to the disc builder's (a test checks this).

## What it converts

| Source | Output | Pipeline function |
| --- | --- | --- |
| PNG / TGA / BMP / JPG / DDS / TIFF / PSD / PCX / QOI, Source `.vtf`, Quake II `.wal`, Blizzard `.blp` | `media/<pack>.gct`: CMPR, power-of-two, mip chain, 1-bit alpha | `gx_cmpr_encode_blocks`, `wt_encode` mip loop |
| Skybox (`<name>rt/bk/lf/ft/up/dn`) | `media/sky_<name><side>.cmp`: 128² CMPR, 8 KiB per side | `gx_cmpr_encode`, `build_sky_cmpr_overlays` |
| WAD3, Quake WAD2 | `media/<wad>.gct` | `wt_read_miptex`, `wt_encode` |
| BSP v30, Quake BSP v29 | `media/wt_<map>.gct` (world textures resolved through the map's WADs, or embedded with the Quake palette) and its sky | `build_world_texture_packs`, `wt_read_wads` |
| Studio `.mdl` | `gc_studio/<name>.mdl` (lean mesh) and `gc_studio/<name>t.mdl` (downsampled skins) | `lean_studio_mesh_bytes`, `lean_studio_tex_companion_bytes` |
| 3D models via Assimp (OBJ, FBX, glTF/GLB, DAE, 3DS, Blend, MD2/MD3/MD5, IQM, B3D, SMD, MS3D and ~25 more) | `models/<name>.mdl` (studio MDL v10 with bones, sequences and paletted skins), then the same `gc_studio/<name>.mdl` + `<name>t.mdl` as a studio model | `models.import_model`, `models.build_studio_mdl`, then the two functions above |
| Source `.mdl` (v44–49) + `.vvd` + `.vtx`, Quake `.mdl` | the same `models/` and `gc_studio/` outputs as an Assimp model | `source_mdl.import_source`, `quake.import_alias`, then `models.build_studio_mdl` |
| Sprite `.spr` (Half-Life, Quake), Quake II `.sp2` | `sprites/<name>.spr`: a Half-Life sprite, downsampled, keeping thin alpha-test strokes | `quake.hl_sprite_bytes`, `lean_hl_spr_bytes` |
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

## Source, Quake and Quake II formats

These are read by GxBake itself (no Assimp), and each becomes what the GameCube port loads.

**Source models** (`.mdl` v44–49, with the `.vvd` and `.dx90.vtx` beside it). `.mdl` files are told apart by their header: GoldSrc studio models (IDST v10) keep the lean/companion path, Source and Quake models are imported. A Source model brings its bones, the root LOD of the first model in each bodypart, family-0 skins and every sequence whose animation lives in the `.mdl`. Skins are found through the `.vmt` files named by the model, under the nearest `materials/` folder (`$cdmaterials`, `$basetexture`, `patch` materials); `$alphatest`/`$translucent` keep the alpha as cut-outs, otherwise the spec/phong mask in alpha is dropped. Source and GoldSrc share axes and units, so the source up axis is ignored.

Studio skinning is rigid, so a vertex weighted between two bones would tear at the joint. GxBake adds **helper bones**: each one is parented to the vertex's main bone and follows the dual-quaternion blend of the two bones at that weight (quarters, or halves if quarters don't fit under 128 bones), so elbows and shoulders bend. They make the animation block larger; untick **Helper bones for blended weights** (or pass `--no-blend-bones`) to snap vertices to their heaviest bone instead. Not carried over: flexes, other LODs and bodygroups, IK rules, animations stored in `.ani` files, and delta (gesture) sequences.

**VTF textures** (7.0–7.5): DXT1/3/5, ATI1N/ATI2N, the 8888/888/565/5551/4444 orders, I8/IA88/A8 and 16-bit float (tone-mapped). The largest mip of the first frame and face becomes a GCWT entry, and six `<name>rt/bk/lf/ft/up/dn.vtf` make a skybox.

**Quake**: alias models (`.mdl`, IDPO v6) convert their first frame and first skin, since studio models can't morph vertices (the inspector says how many frames there were). BSP v29 world textures and WAD2 miptex carry no palette, so GxBake reads `gfx/palette.lmp` from the game folder or from a `pak*.pak` there. Quake sprites (v1) become Half-Life sprites with that palette and index 255 transparent.

**Quake II**: `.wal` textures use the palette in `pics/colormap.pcx`, and `.sp2` sprites become Half-Life sprites built from their PCX frames.

Without the palette files, a greyscale ramp is used and the asset warns. Neither palette is shipped with GxBake, since both are game data.

**Archives.** Game data usually sits in `.pak`, `.vpk` or `.pk3` files. `gxbake extract pak0.pak -o unpacked` unpacks one (`--match 'models/*'` to pick), then convert the folder. For Source, pass the `_dir.vpk`; its numbered `_000.vpk` chunks are read from beside it.

## The app

- Drag files or a whole gamedir onto the window, or use **Add Files…** / **Add Folder…**.
- Expand a WAD, BSP, model, texture pack or skybox to preview each texture in it.
- The middle panes show the source next to the decoded GameCube result, with PSNR and byte counts in the **Inspector** tab.
- The **Settings** tab holds the optimisation knobs: max texture size, mipmaps, alpha mode and cutoff, resize filter, packing, sky size, model skin size, sprite scale, GCVID size and frame rate, extra WAD search folders, and an encode cache. Changes re-render the preview.
- **Convert Selected** or **Convert All** writes into the output folder. The status bar shows the total against the 24 MiB of main RAM.

## Run from source

```sh
pip install -e ".[gui]"
gxbake            # opens the app
gxbake convert path/to/valve -o out --cache ~/.cache/gxbake
gxbake inspect path/to/file.gct
gxbake validate staged/valve   # same checks as the disc builder
gxbake extract hl2_misc_dir.vpk -o unpacked --match 'models/*' --match 'materials/*'
```

Video and audio need `ffmpeg` on `PATH`, next to the executable, or set with `GXBAKE_FFMPEG`.

## Standalone executable

```sh
pip install -e ".[dev]"
pyinstaller packaging/gxbake.spec    # → dist/GxBake (GxBake.exe on Windows, GxBake.app on macOS)
```

The executable opens the app when run with no arguments or with file paths. With a subcommand (`convert`, `inspect`, `validate`) it runs as the CLI. The `build` GitHub workflow runs the tests and builds the executable for Linux, Windows and macOS on every push.

## Layout

- `gxbake/core/`: the pipeline as a library, with no Qt dependency (`cmpr`, `gct`, `goldsrc`, `images`, `studio`, `models`, `skeleton`, `sprite`, `media`, `audit`, `project`, and the native readers `source_mdl`, `vtf`, `quake`, `archives`).
- `gxbake/gui/`: the PySide6 front end.
- `gxbake/cli.py`: the command line.
- `tests/`: builds synthetic WAD, BSP, MDL and SPR files (and Source, Quake and Quake II ones in `game_fixtures.py`), so no game data is needed. Set `GXBAKE_UPSTREAM_SCRIPT=/path/to/scripts/build-gamecube-disc.py` to also check byte-for-byte parity with the disc builder.
