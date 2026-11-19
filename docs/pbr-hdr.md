# PBR materials, HDR and automated builds

Four commands share one engine (`gxbake/core/autotex.py`): every texture they make is encoded in each GX format that suits it, and the smallest one that stays above a PSNR floor (`--min-db`) wins. Give them a byte budget and they trade quality for bytes across *all* their textures at once, taking whichever step (a smaller format, or half the size) costs the least picture per byte saved.

| Command | In | Out |
| --- | --- | --- |
| `gxbake auto` | any image GxBake reads (`gxbake formats`) | `.tpl` per image, one combined `.tpl` + `.h`, `.bti`, or a CMPR `.gct` pack |
| `gxbake pbr` | PBR material sets (loose maps) or `.gltf`/`.glb` | `<material>.tpl` + `<material>.gxmat.json` (the TEV recipe) |
| `gxbake hdr` | `.exr`, `.hdr`, `.pfm` (and 8-bit images) | tone-mapped textures, overbright lightmaps, GX sphere maps |
| `gxbake build` | a `gxbake.toml`/`.json` manifest | all of the above plus skies, incremental, under budgets |

## Output containers

* **TPL**: the Dolphin SDK's texture library (`TPLGet`/`TEXGet`), any number of images, each with its palette and full mip chain, 32-byte aligned. `gxbake/core/tpl.py` writes and reads them; `consoletex.read_tpl` (used by `gxbake decode` and the app) reads them back too.
* **BTI**: J3D's single-texture header (one file per image).
* **GCT**: the port's GCWT world-texture pack. CMPR only, so `--container gct` limits the formats to CMPR.

The port today loads CMPR `.gct` world packs, `.cmp` skies and its own studio/sprite formats; it has no TPL loader yet. TPL/BTI are what the SDK and most GameCube tools load, and the format of every texture is in the file header.

## GX formats and roles

`--role` (or `role =` in a manifest) narrows the formats tried:

| Role | Formats | For |
| --- | --- | --- |
| `auto` | content decides: grey → I4/I8/IA4/IA8, cut-out → CMPR/CI/RGB5A3, graded alpha → CI/RGB5A3/RGBA8, colour → CMPR/CI4/CI8/RGB565 | most textures |
| `color` | CI4, CI8, CMPR, RGB565, RGB5A3, RGBA8 | colour that must stay colour |
| `world` | CMPR | what the port's world packs hold |
| `ui` | everything except CMPR | text, HUD, sprites: no 4×4 block smear |
| `mask` | I4, I8 | spec, gloss, AO, height |
| `normal` | IA8 (X in I, Y in A), RGB565, RGBA8 | normal maps for indirect bump |
| `lightmap` | CMPR, RGB565, I4, I8 | lightmaps |

CI4/CI8 palettes are quantised by Pillow (median cut, octree with alpha) into an RGB565 TLUT, or RGB5A3 when alpha varies; lower mip levels reuse the top level's palette. Mips are premultiplied box filters, so cut-out edges do not bleed. Non-power-of-two sizes are rounded to the nearer power of two unless `--npot` (which forces clamp and no repeat).

## PBR materials (`gxbake pbr`)

Flipper has no pixel shaders, so a PBR set is baked into plain textures plus a TEV recipe.

```sh
gxbake pbr materials/ -o build/mats                        # every material set found, preset lit
gxbake pbr crate.glb -o build/mats --preset prelit         # glTF/GLB materials, everything painted in
gxbake pbr metal/ -o build/mats --preset indirect --env studio.exr
gxbake pbr materials/ --list -o x                          # what was found and which file feeds each map
gxbake pbr materials/ -o build/mats --gct-base mats.gct    # also pack the base maps for the port
```

**Finding maps.** Files are grouped by base name and the channel token at the end of the name (case and `_`/`-`/camelCase do not matter; a trailing resolution like `_2k` is ignored): `albedo basecolor diffuse diff color col d`, `normal nor nrm n` (`normal_dx`, `nor_dx`: DirectX green-down, flipped), `roughness rough r`, `metallic metalness metal m`, `ao occlusion`, `height displacement disp bump`, `emissive emission glow`, `opacity alpha`, `specular spec`, `gloss glossiness smoothness`. Packed maps are split: `orm`/`arm`/`OcclusionRoughnessMetallic` (R AO, G rough, B metal), `rma`, `mra`, glTF `metallicRoughness` (G rough, B metal), Unity `MaskMap` (R metal, G AO, A smoothness) and `MetallicSmoothness`. A set needs at least two maps. glTF/GLB materials are read with their factors, alpha mode and cut-off, embedded or external images, and `KHR_materials_pbrSpecularGlossiness`.

**Presets.**

| Preset | Textures | TEV stages | Look |
| --- | --- | --- | --- |
| `prelit` | base | 1 | lighting from a key light (`--light x,y,z`, tangent space), AO, gloss highlights and glow painted into one texture |
| `lit` (default) | base, spec mask, glow | 1–3 | AO and normal detail in the albedo; the engine's GX lights do the rest; the spec mask scales GX's specular channel |
| `emboss` | lit + height (I4/I8) | 3–5 | + GX emboss bump (`GX_TG_BUMP0`) towards light 0 |
| `indirect` | lit + normal (IA8) + sphere map | 3–5 | + normal-perturbed sphere-map reflections |

What the bake does: albedo goes to linear, is multiplied by AO (from the map, or estimated from the height map, or from the normal map integrated to a height field when there is no height) and by a detail term from the normal map (how far each normal tilts; for `prelit` also N·L to the key light), normalised so a flat surface keeps its albedo. `prelit` adds a normalised Blinn-Phong highlight with an exponent from roughness and F0 from metalness. The spec mask is gloss² × (¼ dielectric … 1 metal). Normal maps for `indirect` are stored X in intensity and Y in alpha of IA8.

**`.gxm`** is the same material in 12 + 4 bytes per map for the port: preset, alpha mode and cut-off, TEV scale and the TPL index of each role. `port/gc_mat.c` reads it and sets up GX for the preset (`gc_mat_setup`); see [port/README.md](../port/README.md#materials-and-tpl-textures-tpl-gxm). `gxbake hdr --mode lightmap` writes one too, with the overbright scale.

**`.gxmat.json`** lists each map (TPL index, format, size, bytes, PSNR), the material's alpha mode with the blend/alpha-compare set-up, and the TEV stages in libogc terms (`GX_TEXMAPn` follows the TPL order). `port/gc_mat.c` implements them (host-tested against a recording GX stub); none has run on hardware or in Dolphin yet, so check the emboss and indirect ones there first.

## HDR (`gxbake hdr`, `gxbake sky --tonemap`)

```sh
gxbake hdr shots/*.exr -o build/tex --tonemap agx                 # tone-mapped textures
gxbake hdr bake/lightmaps/ -o build/lm --mode lightmap             # overbright lightmaps
gxbake hdr studio.exr -o build/env --mode sphere --roughness 0.2   # reflection sphere map
gxbake sky canyon.exr -o staged/valve --tonemap hable              # HDR sky with another curve
```

* **Tone maps**: `aces` (Narkowicz fit, the sky converter's long-standing default, unchanged), `reinhard` (extended, `--white`), `hable` (Uncharted 2), `agx` (log2 encode + AgX sigmoid fit), `clip`. Auto exposure brings the log-average luminance to 0.18; `--exposure` adds stops; `--no-auto-exposure` uses the values as they are.
* **Overbright lightmaps**: a TEV stage can scale its result by 2 or 4 (`GX_CS_SCALE_2`/`_4`). Storing light / scale lets a lightmap brighten a surface past 1.0 instead of clamping. `--overbright auto` picks the smallest scale that clips at most 0.1 % of texels (smaller scales keep more precision in the dark). Values are stored linear (`--gamma` to change); `<name>.lightmap.json` records the scale, peak and clipped fraction. The port's lightmap path is RGB565 with no TEV scale today, so using these needs that stage set up.
* **Sphere maps**: GX texgen can make texture coordinates from the view-space normal (`GX_TG_MTX2x4`, `GX_TG_NRM`, a 0.5 scale/bias matrix). `--mode sphere` renders that map from any panorama, cross, strip or faces (HDR tone-mapped first), as seen by a camera facing `--yaw` (engine angles, 270 = −Y), pitched by `--pitch`; `--roughness` blurs it with a cosine-weighted cone for rough metals. Texels outside the disc carry the rim's colour so filtering never pulls in black.

## Manifests (`gxbake build`)

`gxbake build --init` writes a commented `gxbake.toml`. Rules: `[[textures]]`, `[[pbr]]`, `[[hdr]]`, `[[sphere]]`, `[[sky]]`, `[[camera]]` (see [camera.md](camera.md)); every option of the matching command is a key (underscores for dashes). Paths are relative to the manifest; `inputs` take files, folders (recursive, sub-folders kept in the output) and globs (`**` works); `exclude` takes name globs.

* `[build] budget` is shared by every rule that has no `budget` of its own; a rule with one is placed alone. Skies are counted separately (they live in the port's 48 KiB sky block).
* `[build] jobs` encodes in that many processes.
* `[build] preview = "preview"` writes every texture decoded as the GameCube samples it.
* A group whose inputs (size and mtime), options and GxBake version have not changed is skipped; `--force` rebuilds. The state is `output/.gxbake-build.json`.
* `--json` prints the report; the exit code is 2 when a budget cannot be met even at the smallest options, so CI can fail on it.

* `gxbake build --watch` keeps running and rebuilds whatever changed (polling every `--interval` seconds; editing the manifest rebuilds everything).
* `gxbake tpl out/*.tpl out/*.gxm [--extract DIR]` lists what was written (formats, sizes, mips, TLUTs, material roles) and decodes it to PNG.

TOML needs Python 3.11+ (or `pip install tomli`); JSON manifests work everywhere.
