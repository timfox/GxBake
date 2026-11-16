# GxBake

A desktop tool (and CLI) for converting assets to native GameCube GX renderer formats, previewing exactly what Flipper will sample, and exporting a staged `media/` tree.

GxBake wraps the GX pipeline from the Xash3D GameCube port (`scripts/build-gamecube-disc.py`). The encoders are ported unchanged, so the output is byte-identical to the disc builder's (a test checks this).

## What it converts

| Source | Output | Pipeline function |
| --- | --- | --- |
| PNG / TGA / BMP / JPG | `media/<pack>.gct`: CMPR, power-of-two, mip chain, 1-bit alpha | `gx_cmpr_encode_blocks`, `wt_encode` mip loop |
| Skybox (`<name>rt/bk/lf/ft/up/dn`) | `media/sky_<name><side>.cmp`: 128² CMPR, 8 KiB per side | `gx_cmpr_encode`, `build_sky_cmpr_overlays` |
| WAD2/WAD3 | `media/<wad>.gct` | `wt_read_miptex`, `wt_encode` |
| BSP v30 | `media/wt_<map>.gct` (world textures resolved through the map's WADs) and its sky | `build_world_texture_packs`, `wt_read_wads` |
| Studio `.mdl` | `gc_studio/<name>.mdl` (lean mesh) and `gc_studio/<name>t.mdl` (downsampled skins) | `lean_studio_mesh_bytes`, `lean_studio_tex_companion_bytes` |
| 3D models via Assimp (OBJ, FBX, glTF/GLB, DAE, 3DS, PLY, STL, X, LWO, Blend…) | `models/<name>.mdl` (static studio MDL v10, paletted skins), then the same `gc_studio/<name>.mdl` + `<name>t.mdl` as a studio model | `models.import_model`, `models.build_studio_mdl`, then the two functions above |
| Sprite `.spr` | `sprites/<name>.spr`, downsampled, keeping thin alpha-test strokes | `lean_hl_spr_bytes` |
| Video (AVI, MP4…) | `media/<name>.gcvid` (RGB565 delta) and `.gcpcm` | `build_gcvid_companion`, `build_gcpcm_companion` |
| Audio (WAV, MP3, OGG) | `media/<name>.gcpcm` (48 kHz stereo, s16 big-endian) | `build_gcpcm_companion` |

It also opens the outputs (`.gct`, `.cmp`, `.gcvid`, `.gcpcm`) to inspect them. Every run writes `gxbake-manifest.json` with the settings, every output and its size, and the share of GameCube main RAM it uses.

## 3D model import (Assimp)

Any format [Assimp](https://github.com/assimp/assimp) reads is baked into a GoldSrc studio model, because that is what the GameCube port draws. GxBake triangulates the mesh, flattens the node hierarchy into one static root bone, converts Y-up sources to GoldSrc's Z-up (+Z forward becomes +X forward), builds triangle strips, and turns each material's diffuse/base-colour texture into an 8-bit paletted skin (power-of-two, cut-out alpha as masked index 255). Textures are found next to the model, by basename when the exporter wrote an absolute path, or inside a `.glb`. A material with no texture becomes a solid skin of its colour.

Settings: **Scale** (source units to GoldSrc units; glTF and most FBX are in metres, so try 39.37), **Source up axis**, **Import skin max**, and whether to also write the full `models/<name>.mdl`. The preview renders the imported mesh next to the geometry decoded back from the lean MDL; expand the model to compare each material's texture with its companion skin. Animations and bone weights are not carried over yet, and the inspector warns about anything past the studio limits (2048 vertices, 20000 triangles, 100 skins).

Assimp comes from the `assimp-py` wheel, which bundles the native library, so there is nothing to install separately. It is part of the `gui` and `dev` extras, or `pip install -e ".[models]"` for the CLI alone.

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

- `gxbake/core/`: the pipeline as a library, with no Qt dependency (`cmpr`, `gct`, `goldsrc`, `images`, `studio`, `models`, `sprite`, `media`, `audit`, `project`).
- `gxbake/gui/`: the PySide6 front end.
- `gxbake/cli.py`: the command line.
- `tests/`: builds synthetic WAD, BSP, MDL and SPR files, so no game data is needed. Set `GXBAKE_UPSTREAM_SCRIPT=/path/to/scripts/build-gamecube-disc.py` to also check byte-for-byte parity with the disc builder.
