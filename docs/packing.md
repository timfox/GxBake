# Smarter packing for the GameCube port

`gxbake squeeze <staged gamedir>` measures every technique below on the real files and prints bytes before and after, per memory window. `-o OUT` writes the ones the port already reads. `--emit` also writes the formats that need a loader change in the port (`.gsk` skins, `.dsp` sound, `.yaz0`), so the port side can be built and tested against real data.

Every technique is optional. `--only skins,adpcm` runs just those, `--skip yaz0,trim` leaves some out. The names are `trim`, `anim`, `skins`, `adpcm`, `verts`, `lightmaps` and `yaz0`. Lossy settings are opt-in thresholds: `--min-db` for the mip trim, `--skin-db` for lossy skin formats and `--anim-tol` for the animation deadband (0, the default, is lossless). The standalone tools are `gxbake adpcm`, `gxbake yaz0` and `gxbake slim --anim-tol`.

The one change that is not optional is lossless: studio animation is now always written with the optimal run encoding, which the engine reads unchanged.

## Where the bytes go today

What the port does with each kind of data, read from its loaders (`ref/gx/r_gx_skins.c`, `r_image.c`, `r_gx_world.c`, `vid_gamecube.c`, `snddma_gamecube.c`, `mod_studio.c`):

| Data | Stored as today | Bits per texel / sample |
| --- | --- | --- |
| World textures | CMPR mip chains (`.gct`) | 4, already the smallest GX format |
| Studio skins | 8-bit indexed on disc, expanded to RGB565 / RGB5A3 at upload | 16 |
| Lightmaps | RGB565 atlas, 800 KiB | 16 |
| Sound effects | 8/16-bit PCM, 20 KiB per file cap | 8–16 |
| Music, cinematics audio | `.gcpcm`, 48 kHz stereo s16 | 16 (192 KB/s off the disc) |
| Studio vertices | float vec3 + bone byte | 104 bits per vertex |
| Studio animation | RLE s16 streams | 16 per changing value |

The biggest waste is the 16-bit places: skins, lightmaps and sound all hold data that needs far fewer bits.

## Results on sample assets

Real, freely licensed assets, not synthetic ones: Kenney's CC0 tiles, sound effects and music (from the `arcade` package) and Panda3D's own `panda-model` + `panda-walk4` and environment textures (BSD). The numbers are from `squeeze` and a bench script over the staged output of `gxbake convert`.

| Technique | Needs port change | Before | After | Saved | Quality |
| --- | --- | ---: | ---: | ---: | --- |
| Indexed skins (CI8 + RGB565 TLUT), panda 256² skin | yes | 128.0 KiB | 64.5 KiB | **50 %** | bit-identical to today's RGB565 |
| Indexed skins, panda 64² companion (`t.mdl`) | yes | 8.0 KiB | 4.4 KiB | 45 % | bit-identical |
| s16 positions + s8 normals, panda (901 vertices) | yes | 21.1 KiB | 7.9 KiB | **62 %** | max error 0.05 units over a 6,500-unit model (1/130,000) |
| DSP ADPCM music, 172 s `.gcpcm` at 48 kHz | yes | 31.6 MiB | 9.0 MiB | **71 %** | 51.6 dB SNR (transparent); 192 → 55 KB/s off the disc |
| DSP ADPCM music at 32 kHz | yes | 31.6 MiB | 6.0 MiB | 81 % | 45.7 dB SNR |
| DSP ADPCM sound effects, 55 Kenney sounds | yes | 782.7 KiB (fitted PCM) | 313.6 KiB | **60 %** | median 28 dB (see below) |
| Yaz0, photo texture pack (12 textures) | yes | 365.5 KiB | 230.9 KiB | 37 % | lossless |
| Yaz0, flat-art texture pack | yes | 2.0 MiB | 550 KiB | 73 % | lossless |
| Yaz0, lean studio mesh | yes | 58.7 KiB | 43.0 KiB | 27 % | lossless |
| Detail mip trim, flat-art pack (151 textures) | **no** | 1.64 MiB | 1.52 MiB | 7 % | every trimmed level ≥ 40 dB; MEM1 low mips 417 → 389 KiB, and the pack goes from not fitting the 448 KiB pool to fitting at split 32 |
| Detail mip trim, photo pack | **no** | 365.5 KiB | 362.9 KiB | 1 % | photos have real detail, so it leaves them alone |
| Optimal animation runs, panda walk | **no** | 6.7 KiB | 6.5 KiB | 3 % | lossless |
| Strips + fans (new stripifier), panda 1,798 triangles | **no** | 22.8 KiB | 20.0 KiB | 12 % of triangle commands | lossless, same triangles and winding |
| Keyframe fit, synthetic sine walk + linear death, at 0.5° / 1° / 2° | **no** | 282 B (31 + 20 frames) | 234 / 190 / 170 B | 17 / 33 / 40 % of animation | every bone within the stated angle (and a tenth of it in units) at the original timing |
| k-means CI4 skin, panda 256² (`--preset max`) | yes | 128.0 KiB | 32.0 KiB | **75 %** | 34.5 dB (CMPR at the same size: 33.0 dB) |
| k-means CI4 skin, panda 64² companion | yes | 8.0 KiB | 2.0 KiB | 75 % | 34.9 dB (CMPR: 28.6 dB) |
| Yaz0 + CMPR split, flat-art pack | yes | 2.0 MiB | 461 KiB | **77 %** | lossless; plain Yaz0 on the same pack: 805 KiB, so 43 % smaller |
| Deflate (LZ + Huffman) instead of Yaz0, 4 packs and models | yes | 2.2 MiB | 487 KiB | 78 % | lossless; Yaz0: 943 KiB |
| Duplicate textures, flat-art pack | yes | 21.3 KiB | 10.7 KiB | 50 % of the pair | ≥ 59 dB look-alike in the same pack |
| Lightmap size by error, GxBake-lit Source 2 test map (24 faces; synthetic light, no real HL map to hand) | yes | 2,981 luxels | 1,022 luxels | 66 % | 14 flat faces → 1×1 within ±2; 4 at half resolution, worst 48 dB |
| Vertex order by first use, panda | **no** | 2,131 cache-line misses | 1,934 | 9 % fewer misses (simulated 2 KiB LRU, 32-byte lines) | lossless; Yaz0 and deflate 0.1–0.4 % smaller |
| Atlas packing, sorted skyline vs Quake's allocator, 600–3,000 GoldSrc-sized faces | yes | n pages | n pages | 0 % | both already reach the area lower bound |
| CMPR endpoint refinement, 158 textures | **no** (`--cmpr-refine`) | same size | same size | 0 % | +0.3 dB median, +0.4 mean, up to +2.4 dB |

What the sound numbers mean: Kenney's effects are synthetic square-wave chiptune, the worst case for any 4-bit ADPCM (22.6 dB on `laser4`). Natural recordings code far better, as the music shows (51.6 dB). Against today's fitter, which drops long sounds to 8-bit at 11 kHz to get under the 20 KiB cap, ADPCM at 22 kHz fits every one of the 55 sounds under the cap with twice the bandwidth.

So against Tim's target of about half the memory with no visible or audible difference: skins (50 %, lossless), vertices (62 %, sub-pixel), music (71 %, transparent) and sound memory (60–72 %) get there. World textures are already CMPR, so they only get a few percent without loss. Yaz0 roughly halves disc and ARAM-cache bytes for free.

## Port-side code

[`port/gc_pack.c`](../port/gc_pack.c) holds drop-in C readers for `.gsk`, `.dsp`, `.gsv` and Yaz0. They are host-tested bit-exact against GxBake's encoders. [`port/README.md`](../port/README.md) shows where each one hooks into the port.

## Ranking

Ranked by memory saved per unit of port work. "Runtime cost" is the cost on Gekko/Flipper per frame.

| # | Technique | Window | Saves | Port work | Runtime cost |
| --- | --- | --- | --- | --- | --- |
| 1 | **Indexed skins** (`.gsk`): CI8/CI4 + TLUT, CMPR or I4/I8 when within `--skin-db` | 384 KiB skin overflow, 192 KiB skin window, ARAM | 50 % lossless, up to 75 % | Small: `GX_InitTlutObj` + `GX_LoadTlut` + `GX_InitTexObjCI`, and read texels instead of building RGB565 | Less than today: no per-pixel conversion or tiling on the CPU. TLUT loads are 512 B |
| 2 | **DSP ADPCM sound** (`.dsp`) | 128 KiB MEM1 sfx cache, 2 MiB ARAM sfx, disc bandwidth | 72 % vs 16-bit | Medium: decode in `S_` loader (12 lines of C, see `dspadpcm.decode_frames`), or keep ADPCM resident and decode in the mixer for the full 3.5× | Decode is a few integer ops per sample; the DSP can also do it in hardware through AX |
| 3 | **ADPCM `.gcpcm`** | disc bandwidth shared with texture streaming | 71 % | Small: the stream reader decodes chunks before `ASND_AddVoice` | Same as above |
| 4 | **s16 vertices, s8 normals** | 1 MiB studio arena | 62 % of vertex arrays | Medium: `R_StudioDrawPoints` reads them through `psq_l` with a GQR scale | **Free**: Gekko's quantised paired-single loads convert s16/s8 to float in the load itself. Display lists can also hand s16 straight to GX with `GX_SetVtxAttrFmt(..., GX_POS_XYZ, GX_S16, frac)` |
| 5 | **Yaz0 on disc** | disc space and read time; ARAM asset cache if kept compressed | 27–73 % | Small: a 30-line decoder in the file loader | Decoding runs faster than the disc delivers, so reads get quicker |
| 6 | **CMPR lightmaps** for static pages | 800 KiB atlas | 75 % (estimate) | Larger: encode at map load or bake per page; styled pages stay RGB565 | One-off encode at load |
| 7 | **Detail mip trim** | MEM1 pool + ARAM wtstore | 1–7 % here, more on blurry or upscaled WAD textures | None | None |
| 8 | **Optimal animation runs**, **deadband** | studio arena | 3 % lossless here; more on hand-keyed animation with holds | None | None |

## Techniques in detail

### Indexed skins (`gxtex.py`, `squeeze.studio_skins`, `.gsk`)

GoldSrc skins have at most 256 colours, and the port converts them to RGB565 anyway. Stored as CI8 with the palette converted once to an RGB565 TLUT, Flipper samples exactly the same colours at half the bytes. A skin with 16 or fewer colours used goes CI4 (a quarter). Masked skins use an RGB5A3 TLUT with index 255 transparent, and for `v_*.mdl` view models the port's chromakey rules (`R_GXSkinBuildTiled`) are applied to the palette, which is exact because they only look at the colour. Lossy candidates (CMPR, and I4/I8/IA4 for grey skins) are taken only when they stay within `--skin-db` (40 dB) of today's picture; the panda skin's CMPR came to 31 dB, so it stayed CI8.

`gxtex.py` also encodes and decodes every other GX format (I4, I8, IA4, IA8, RGB565, RGB5A3, RGBA8, CI4, CI8, CMPR) with the exact tile layouts, for future use (HUD art, fonts, lightmaps).

`.gsk` layout (big-endian, every blob 32-byte aligned for `GX_InitTexObj` and `GX_LoadTlut`):

```
"GCSK" u32 count u32 version (1) u32 0
count × { u16 skin; u8 GXTexFmt; u8 GXTlutFmt; u16 w, h; u16 tlut entries;
          u16 flags (1 = alpha); u32 tex off, tex bytes, tlut off, tlut bytes; 8 × 0 }
texels and TLUTs
```

Port side: in `Mod_GCAttachStudioSkins`, look for `gc_studio/<name>.gsk` before the `t.mdl`. For each entry, `GX_InitTlutObj(&tlut, pal, fmt, n)` + `GX_LoadTlut(&tlut, GX_TLUT0 + k)` and `GX_InitTexObjCI(&obj, texels, w, h, fmt, GX_REPEAT, GX_REPEAT, GX_FALSE, GX_TLUT0 + k)`. Up to 16 TLUTs of 256 entries fit in TMEM at once; skins of one model can share one when their palettes match.

### DSP ADPCM (`dspadpcm.py`, `gxbake adpcm`)

The standard `.dsp` file (96-byte header, as written by Nintendo's DSPADPCM tool), one per channel, with loop points and the loop frame's decoder state. The encoder picks 8 predictor pairs by weighted k-means over per-frame least-squares predictors, then encodes each 14-sample frame closed loop, trying every pair and the exponents around the residual's peak. Decode, for the port:

```c
for each 8-byte frame: ps = *p++; c1 = coef[ps >> 4][0]; c2 = coef[ps >> 4][1]; e = ps & 15;
  for 14 nibbles n (high first, signed): s = clamp16((((n << e) << 11) + 1024 + c1 * h1 + c2 * h2) >> 11); h2 = h1; h1 = s;
```

### s16 vertices

A studio vertex is a float vec3 (12 bytes) and a normal another 12. Positions as s16 take 6 bytes and normals as s8 take 3. Gekko has quantised paired-single loads (`psq_l` with a GQR set to s16 and a scale), so the CPU skinning loop reads them as floats at no extra cost.

`squeeze --emit` writes `gc_studio/<model>.gsv` (`vertpack.py`). Each model gets one power-of-two scale: the most fractional bits that keep every coordinate inside s16, which is exactly what a GQR dequantise scale holds. The error is therefore at most half a step, under 1/32,767 of the model's largest coordinate. Normals are s8 with 6 fractional bits, GX's own s8 normal format, so they are within about 1°. `port/gc_pack.c` has `gc_gsv_parse`, `gc_gsv_entry` and a portable `gc_gsv_dequant`, which is host-tested to produce the same floats as the Python reader. On the console, set GQR scale = `pos_frac` and `psq_l` the arrays directly.

### Yaz0 (`yaz0.py`, `gxbake yaz0`)

Nintendo's LZ77 (16-byte header, flag bytes, 4 KiB window, matches up to 273 bytes). The encoder does hash-chain matching with a one-step lazy match.

### Detail mip trim (ships today)

For each world texture, level 1 is magnified the way Flipper does (bilinear, texel centres, wrapping like `GX_REPEAT`) and compared with level 0. If the difference is under the threshold (40 dB PSNR, an RMS error of about 2.5 out of 255), level 0 adds nothing visible and is dropped. This repeats, so a solid colour ends at 8×8. 1-bit alpha textures also need their cut-out to survive (at most 0.2 % of texels may flip). This is the same operation as `gxbake fit`, which the port already reads: UVs are normalised, so nothing moves.

### Optimal animation runs and deadband (ship today)

A studio animation stream is a list of chunks, each holding some values and repeating the last one to the end of the chunk. GxBake used to cut chunks only at 255 frames, so a hold in the middle of a chunk cost a value every frame. The encoder now picks the cheapest chunking by dynamic programming over run boundaries. This is lossless, the engine reads it unchanged, and every model GxBake builds gets it. `--anim-tol N` (on `slim` and `squeeze`) also holds a value until the curve moves more than N raw units away, which makes runs longer with a bounded error. It helps hand-keyed animation more than the sampled walk measured above.

## The maths behind the newer options

### Triangle strips and fans (`models._tri_commands`, ships today)

GoldSrc triangle commands cost 2 bytes plus 8 per vertex, so the number to minimise is vertex references per triangle (3.0 for lone triangles, approaching 1.0 for long runs). The old greedy stripper started anywhere and only made strips. The new one is the SGI/STRIPE heuristic: start at a triangle with the fewest free neighbours (the edge of the remaining mesh, so it does not strand islands), try all three rotations as both a strip and a fan, and keep whichever covers most triangles. Fans matter on caps, poles and eyes, where one vertex is shared by a ring of triangles: a fan of n triangles is n + 2 references, where strips through the same ring break into pieces. On the panda this goes to 1.37 references per triangle with 79 fans among 336 commands, 12 % fewer triangle-command bytes. Every model GxBake builds gets it; the engine reads negative counts as fans already.

### Animation keyframe fit (`slim --fit-deg`, `squeeze --anim-fit`, ships today)

A studio sequence stores every frame at a fixed rate, and the engine interpolates between neighbouring frames (slerp for rotations). That makes the stored frames a piecewise-linear curve with evenly spaced knots, so the question is: what is the fewest evenly spaced knots m such that playing them back over the same duration stays within tolerance at every original frame? For each m from the smallest allowed (`--max-step`, default 8 frames apart) upward, each bone channel is resampled at m points (between original frames when (n − 1) is not a multiple of (m − 1)), played back, and checked against the original: positions in model units, rotations in degrees, the short way round. The first m that passes for every channel wins; fps is scaled so the duration and events stay put. Linear motion (a fall, a slide) drops to the minimum; smooth cycles drop by half at about 1°. The panda walk does not shrink at 0.5°: three of its bones jump by thousands of units for single frames (the source animation has them), and no fit can hold a one-frame spike, which is the correct answer.

### Skin palettes (`gxtex.choose_indexed`)

Three things, in order:

1. **Merge in the 16-bit domain (lossless).** A GoldSrc skin has 8-bit-per-channel palette entries, but the GameCube samples RGB565. Entries that differ only in the bits RGB565 drops are the same colour on the console, so they are merged before counting. A skin with 40 palette entries that land on 16 RGB565 words becomes CI4, a quarter of today's bytes, with nothing changed.
2. **Weighted k-means CI4 (lossy, under the floor).** If more than 16 colours remain, the used colours are clustered into 16 (15 plus transparent for masked skins), each colour weighted by its pixel count, with distances weighted by luma (0.299, 0.587, 0.114) because the eye is most sensitive to green and brightness. Seeds are weighted farthest-point picks, so the result is deterministic. The candidate competes with CMPR at the same size. On the panda skin it wins (34.5 dB vs 33.0), and on the 64² companion by a lot (34.9 vs 28.6), because a palette keeps exact flat colours where CMPR has to blend within each 4×4 block. At the default 40 dB floor neither passes, so `balanced` keeps CI8; `--preset max` (34 dB) takes CI4 for 75 %.
3. **Shared blobs.** `.gsk` stores identical texels or TLUTs once and points several entries at them. The port reader already follows offsets, so nothing changes there.
4. **One TLUT per model (`squeeze.share_tlut`).** When a model's CI8 skins together use at most 256 distinct 16-bit colours, they are re-indexed against one shared TLUT. This is lossless: every texel keeps its exact word. The pack then stores one TLUT instead of one per skin, and the port can keep a single TLUT loaded while it draws the whole model.

When two candidates are within 64 bytes of each other (one TLUT), the closer picture wins.

### Duplicate textures (`dedup.py`, `squeeze --only dedup`)

Exact duplicates are found by hashing what GX actually samples (CMPR blocks, or a skin as its RGB565/RGB5A3 words), not the source bytes, so two WAD textures that differ only in bits the console drops count as one. Near duplicates use a 64-bit difference hash: shrink to a 9×8 grey thumbnail by box averaging, and set a bit wherever a cell is brighter than its right neighbour. Pictures within 4 bits of each other are compared at full size, and are reported as duplicates only if they clear the PSNR floor; look-alikes that fail it are listed for a person to check. The report separates duplicates inside one file (they cost MEM1 when that map or model loads) from ones across files (they only cost disc, since one map loads at a time). Sharing them needs the port's pack loader to accept two entries with the same offset.

### CMPR endpoint refinement (`cmpr.cmpr_refine`, `--cmpr-refine`)

Given the indices a block already uses, each pixel is a known blend of the two endpoints (weights 1, 0, ⅔, ⅓, or 1, 0, ½ in three-colour blocks). The endpoints that minimise squared error therefore solve a 2×2 linear system per channel. The refiner solves it, rounds to RGB565, re-picks indices, repeats twice, and keeps a block only if its error dropped. Size is unchanged and a block never changes mode, so cut-outs stay exactly where they were. Over 158 sample textures it gains 0.3 dB median and up to 2.4 dB. It is on for new GX skins, and opt-in for world textures and images (`convert --cmpr-refine`, `pack --cmpr-refine`) so default output stays byte-identical to the disc builder's encoder.

### Sorting bytes by meaning before compression (`squeeze.split_cmpr`)

A CMPR block is 4 bytes of colour endpoints followed by 4 bytes of 2-bit indices. Neighbouring blocks often share endpoints, but in the normal layout those repeats are 8 bytes apart with index noise between them, so an LZ coder sees short matches. Moving all endpoint words first and all index words after (a transpose, undone with one loop after decompressing) gives Yaz0 long runs to match. On the flat-art pack Yaz0 goes from 805 KiB to 461 KiB, 43 % smaller, at no runtime cost beyond the un-split.

### Entropy coding vs Yaz0

Yaz0 is LZ with no entropy coding stage, so every literal costs a full byte. Measured on the staged samples, deflate (LZ + Huffman) comes out 10–25 % smaller than Yaz0 on every file type, and LZMA smaller again. On already-dense data Yaz0 is worse than nothing: the 8-bit Kenney sound effects grow by up to 12 % under Yaz0, while deflate still saves a little, so `squeeze` only offers Yaz0 for packs, models, BSPs and sprites. A small inflater (tinf or uzlib, about 2 KiB of code) would be worth it if disc space or ARAM caching becomes the limit; the `deflate on disc` row in `squeeze` shows what it would save.

### Lightmap atlas packing (`rectpack.py`): a measured non-win

Packing face lightmaps into atlas pages is 2D bin packing. `rectpack` has Quake's `AllocBlock` (a per-column skyline, faces in map order), a height-sorted shelf, and a height-sorted skyline, and a lower bound of ceil(total area / page area). On GoldSrc-shaped inputs (hundreds to thousands of rectangles 2–17 luxels a side, 128² pages, 1-luxel border) all three hit the lower bound: the pieces are so small next to the page that only the last page has slack. MaxRects (Best Short Side Fit) was also tried and did worse in a strip and ran about 300× slower. So the atlas is not where lightmap memory goes; the next section is.

### Lightmap resolution by error (`gxbake lightmaps`, `squeeze` row "lightmap size by error")

The same idea as detail mip trim, applied per face. A face whose luxels all sit within ±2 of their mean is one colour, so it can be 1×1. Otherwise the face is resampled to half resolution over the same span (corners kept), interpolated back, and kept at half size when the result is within the PSNR floor (40 dB by default). The port bakes lightmap UVs per vertex in its world display lists, so a face's grid size costs nothing at runtime: only the baker (`bake_worlddl_goldsrc.py` in the port tree) needs to scale those UVs. On the one lit map available here, a GxBake-lit Source 2 test map, it removes 66 % of the luxels. That light is smooth and synthetic; real Half-Life maps have hard shadows, so expect less. Run `squeeze` on a real staged `maps/` folder to measure it.

`gxbake lightmaps <map.bsp>` does both steps as options. `--packer quake|shelf|skyline` picks the packer, `--page`, `--pad`, `--min-db` and `--flat-tol` tune it, and `--no-resize` only repacks. `--plan OUT.json` writes, for every lit face, its old and new grid size and its page, x and y. A world baker can take that file to place lightmaps and scale UVs.

### Vertex order by first use (`models._first_use_order`, ships today)

The engine transforms every studio vertex into an array, and the triangle commands then read it. GxBake now numbers vertices (and each mesh's normals) in the order the commands first touch them, so the reads walk mostly forward. In a simulated 2 KiB LRU cache with 32-byte lines (Gekko's line size), the panda's command stream misses 9 % less. The model is the same: same triangles, same vertices, different numbering. The index stream also compresses slightly better (0.1–0.4 %).

### Presets (`squeeze --preset`)

| Preset | Detail trim | Lossy skins | Animation |
| --- | --- | --- | --- |
| `lossless` | only bit-exact levels | never | runs re-encoded only |
| `balanced` (default) | ≥ 40 dB | ≥ 40 dB | keyframe fit ≤ 0.5° |
| `max` | ≥ 34 dB | ≥ 34 dB | keyframe fit ≤ 2°, deadband ±2 |

Any explicit option (`--min-db`, `--skin-db`, `--anim-tol`, `--anim-fit`) overrides the preset.

## Other ideas, not built yet

- **Lightmaps**: GoldSrc lightmaps are mostly low-saturation. I8 or IA4 intensity with a per-face tint in TEV would be 50–75 % smaller than RGB565 and could be close to invisible; this needs a BSP sample to measure.
- **Detail split for world textures**: a half-resolution CMPR base times a shared, tiling I4 detail texture in TEV. That is 4× less for the base, at the cost of a second texture stage.
- **u8 texture coordinates** (`squeeze` reports it under `verts`): GoldSrc S/T are whole texels in s16, so a strip or fan whose coordinates all stay in 0..255 can go to the GPU as u8 in a second GX vertex format (`GX_SetVtxAttrFmt(GX_VTXFMT1, GX_VA_TEX0, GX_TEX_ST, GX_U8, 0)` and opcode `| 1`), with no loss. Lost Coast's props use exactly this (pack v8): 87 % of their vertices qualified and the display lists shrank 15 % with a pixel-identical picture. Streamed lists must pass the command's own format bits to `GX_Begin`.
- **Shared TLUTs across a map's NPCs**, by quantising the union to 256 colours when it is larger. Within one model, the lossless case is done.
- **Vertex cache order** (Forsyth/Tipsify) for indexed display lists, if the port moves studio meshes to `GX_INDEX16` display lists. Today's software transform path gets first-use order instead (above).
- **Bit-packed BSP/collision**: plane indices and node children in the narrowest integer that holds them, plane sharing between brushes.
- **GX index formats**: `GX_INDEX8` vertex indices in display lists when a mesh has under 256 vertices.
