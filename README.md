# GxBake

A desktop tool (and CLI) for converting assets to native GameCube GX renderer formats, previewing exactly what Flipper will sample, and exporting a staged `media/` tree.

GxBake wraps the GX pipeline from the Xash3D GameCube port (`scripts/build-gamecube-disc.py`). The encoders are ported unchanged, so the output is byte-identical to the disc builder's (a test checks this).

## What it converts

| Source | Output | Pipeline function |
| --- | --- | --- |
| PNG / TGA / BMP / JPG / DDS / TIFF / PSD / PCX / QOI | `media/<pack>.gct`: CMPR, power-of-two, mip chain, 1-bit alpha | `gx_cmpr_encode_blocks`, `wt_encode` mip loop |
| Skybox (`<name>rt/bk/lf/ft/up/dn`) | `media/sky_<name><side>.cmp`: 128² CMPR, 8 KiB per side | `gx_cmpr_encode`, `build_sky_cmpr_overlays` |
| WAD2/WAD3 | `media/<wad>.gct` | `wt_read_miptex`, `wt_encode` |
| BSP v30 | `media/wt_<map>.gct` (world textures resolved through the map's WADs) and its sky | `build_world_texture_packs`, `wt_read_wads` |
| Studio `.mdl` | `gc_studio/<name>.mdl` (lean mesh) and `gc_studio/<name>t.mdl` (downsampled skins) | `lean_studio_mesh_bytes`, `lean_studio_tex_companion_bytes` |
| 3D models via Assimp (OBJ, FBX, glTF/GLB, DAE, 3DS, Blend, MD2/MD3/MD5, IQM, B3D, SMD, MS3D and ~25 more) | `models/<name>.mdl` (studio MDL v10 with bones, sequences and paletted skins), then the same `gc_studio/<name>.mdl` + `<name>t.mdl` as a studio model | `models.import_model`, `models.build_studio_mdl`, then the two functions above |
| Sprite `.spr` | `sprites/<name>.spr`, downsampled, keeping thin alpha-test strokes | `lean_hl_spr_bytes` |
| Video (AVI, MP4…) | `media/<name>.gcvid` (RGB565 delta) and `.gcpcm` | `build_gcvid_companion`, `build_gcpcm_companion` |
| Audio (WAV, MP3, OGG) | `media/<name>.gcpcm` (48 kHz stereo, s16 big-endian) | `build_gcpcm_companion` |

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

### Disc usage

`gxbake disc <staged gamedir>` shows what the tree costs on a disc: every file rounded up to 2 KiB ISO9660 sectors (the disc builder exposes game data through ISO9660), the share of the 1.36 GiB disc, and a breakdown by directory. It also lists byte-identical files, textures stored in more than one `.gct` (same CMPR bytes; each map loads its own pack, so this only costs disc), and files under `models/`, `gc_studio/` and `sound/` that no map names. The game DLL may still precache those, so treat that list as a place to look, and pass `--include` for models you know are needed. `--json` gives the same data.

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
gxbake budget staged/valve     # per-map memory report
```

Video and audio need `ffmpeg` on `PATH`, next to the executable, or set with `GXBAKE_FFMPEG`.

## Standalone executable

```sh
pip install -e ".[dev]"
pyinstaller packaging/gxbake.spec    # → dist/GxBake (GxBake.exe on Windows, GxBake.app on macOS)
```

The executable opens the app when run with no arguments or with file paths. With a subcommand (`convert`, `inspect`, `validate`, `budget`, `fit`, `sfx`, `disc`) it runs as the CLI. The `build` GitHub workflow runs the tests and builds the executable for Linux, Windows and macOS on every push.

## Layout

- `gxbake/core/`: the pipeline as a library, with no Qt dependency (`budget`, `sfx`, `disc`, `cmpr`, `gct`, `goldsrc`, `images`, `studio`, `models`, `skeleton`, `sprite`, `media`, `audit`, `project`).
- `gxbake/gui/`: the PySide6 front end.
- `gxbake/cli.py`: the command line.
- `tests/`: builds synthetic WAD, BSP, MDL and SPR files, so no game data is needed. Set `GXBAKE_UPSTREAM_SCRIPT=/path/to/scripts/build-gamecube-disc.py` to also check byte-for-byte parity with the disc builder.
