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
| ROSE Online `.zms` meshes, `.zsc` object lists, `.chr` character lists (with `.zmd` skeletons, `.zmo` motions, `.dds` textures) | the same `models/` + `gc_studio/` studio models, one per object or character | `rose.import_rose`, then as above |
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

## ROSE Online models

GxBake reads ROSE Online's own formats directly (Assimp knows none of them): ZMS meshes, ZMD skeletons, ZMO motions, ZSC object lists and LIST_NPC.CHR character lists, plus DDS textures through Pillow. Point it at an extracted client tree (the folder that holds `3DDATA`) or at single files:

- **`LIST_NPC.CHR`**: one asset per character. Its meshes come from `PART_NPC.ZSC` next to it, its skeleton and motions from the paths in the CHR, so each NPC or monster becomes an animated studio model with sequences named after its motion slots (`stop`, `walk`, `attack`, `hit`, `die`, `run`…; idle and walk loop, the rest play once).
- **`LIST_*.ZSC`**: one asset per object (map decorations, buildings, items), with every part placed by its ZSC transform.
- **`.zms`**: a single mesh. A same-named texture next to it is used; a skinned mesh picks up the `.zmd` beside it and the `.zmo` motions in its folder (or `<mesh>@<clip>.zmo` clips).
- **`.rosemdl`**: a small JSON file for models nothing lists, such as a player avatar assembled from body, arm, foot and face parts: `{"skeleton": "3DDATA/AVATAR/MALE.ZMD", "parts": [{"zsc": "3DDATA/AVATAR/LIST_MBODY.ZSC", "object": 1}, {"mesh": "x.zms", "texture": "x.dds", "bone": 4}], "motions": {"walk": "3DDATA/MOTION/AVATAR/EMPTY_WALK_M1.ZMO"}}`.

Paths inside ZSC/CHR files are matched case-insensitively against the `3DDATA` root. Adding a whole folder lists the CHR characters and ZSC objects and leaves out the loose meshes, textures and `PART_*.ZSC` they draw from. Output names are `<list>_<index>_<name>`, e.g. `npc_33_deck16.mdl`. `--rose-entry N` (repeatable) limits a CLI run to some entries.

ROSE is Z-up like GoldSrc. Settings: **Units per metre** (39.37 keeps real size) and **Turn** (ROSE models face -Y; 90° makes them face GoldSrc's +X). Skinning is rigid (each vertex follows its heaviest bone), double-sided materials get back faces, and alpha-tested materials become masked skins. Not converted yet: per-part ZSC animations, UV/alpha ZMO tracks, effects, and maps; see [docs/rose-online.md](docs/rose-online.md).

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
```

Video and audio need `ffmpeg` on `PATH`, next to the executable, or set with `GXBAKE_FFMPEG`.

## Standalone executable

```sh
pip install -e ".[dev]"
pyinstaller packaging/gxbake.spec    # → dist/GxBake (GxBake.exe on Windows, GxBake.app on macOS)
```

The executable opens the app when run with no arguments or with file paths. With a subcommand (`convert`, `inspect`, `validate`) it runs as the CLI. The `build` GitHub workflow runs the tests and builds the executable for Linux, Windows and macOS on every push.

## Layout

- `gxbake/core/`: the pipeline as a library, with no Qt dependency (`cmpr`, `gct`, `goldsrc`, `images`, `studio`, `models`, `skeleton`, `sprite`, `media`, `audit`, `project`).
- `gxbake/gui/`: the PySide6 front end.
- `gxbake/cli.py`: the command line.
- `tests/`: builds synthetic WAD, BSP, MDL and SPR files, so no game data is needed. Set `GXBAKE_UPSTREAM_SCRIPT=/path/to/scripts/build-gamecube-disc.py` to also check byte-for-byte parity with the disc builder.
