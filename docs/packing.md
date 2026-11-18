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

What the sound numbers mean: Kenney's effects are synthetic square-wave chiptune, the worst case for any 4-bit ADPCM (22.6 dB on `laser4`). Natural recordings code far better, as the music shows (51.6 dB). Against today's fitter, which drops long sounds to 8-bit at 11 kHz to get under the 20 KiB cap, ADPCM at 22 kHz fits every one of the 55 sounds under the cap with twice the bandwidth.

So against Tim's target of about half the memory with no visible or audible difference: skins (50 %, lossless), vertices (62 %, sub-pixel), music (71 %, transparent) and sound memory (60–72 %) get there. World textures are already CMPR, so they only get a few percent without loss. Yaz0 roughly halves disc and ARAM-cache bytes for free.

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

A studio vertex is a float vec3 (12 bytes) and a normal another 12. Quantised to s16 against the model's bounds, positions take 6 bytes with an error of 1/65,536 of the model's size; normals as s8 take 3. Gekko has quantised paired-single loads (`psq_l` with a GQR set to s16 and a scale), so the CPU skinning loop reads them as floats at no extra cost. GxBake measures this today; writing the format waits for the port's reader.

### Yaz0 (`yaz0.py`, `gxbake yaz0`)

Nintendo's LZ77 (16-byte header, flag bytes, 4 KiB window, matches up to 273 bytes). The encoder does hash-chain matching with a one-step lazy match.

### Detail mip trim (ships today)

For each world texture, level 1 is magnified the way Flipper does (bilinear, texel centres, wrapping like `GX_REPEAT`) and compared with level 0. If the difference is under the threshold (40 dB PSNR, an RMS error of about 2.5 out of 255), level 0 adds nothing visible and is dropped. This repeats, so a solid colour ends at 8×8. 1-bit alpha textures also need their cut-out to survive (at most 0.2 % of texels may flip). This is the same operation as `gxbake fit`, which the port already reads: UVs are normalised, so nothing moves.

### Optimal animation runs and deadband (ship today)

A studio animation stream is a list of chunks, each holding some values and repeating the last one to the end of the chunk. GxBake used to cut chunks only at 255 frames, so a hold in the middle of a chunk cost a value every frame. The encoder now picks the cheapest chunking by dynamic programming over run boundaries. This is lossless, the engine reads it unchanged, and every model GxBake builds gets it. `--anim-tol N` (on `slim` and `squeeze`) also holds a value until the curve moves more than N raw units away, which makes runs longer with a bounded error. It helps hand-keyed animation more than the sampled walk measured above.

## Other ideas, not built yet

- **Lightmaps**: GoldSrc lightmaps are mostly low-saturation. I8 or IA4 intensity with a per-face tint in TEV would be 50–75 % smaller than RGB565 and could be close to invisible; this needs a BSP sample to measure.
- **Detail split for world textures**: a half-resolution CMPR base times a shared, tiling I4 detail texture in TEV. That is 4× less for the base, at the cost of a second texture stage.
- **Shared TLUTs**: skins of one model, or the NPCs of one map, drawn from one 256-colour palette, so TMEM holds one TLUT for all of them.
- **u8 texture coordinates** (`squeeze` reports it under `verts`): GoldSrc S/T are whole texels in s16, so a strip or fan whose coordinates all stay in 0..255 can go to the GPU as u8 in a second GX vertex format (`GX_SetVtxAttrFmt(GX_VTXFMT1, GX_VA_TEX0, GX_TEX_ST, GX_U8, 0)` and opcode `| 1`), with no loss. Lost Coast's props use exactly this (pack v8): 87 % of their vertices qualified and the display lists shrank 15 % with a pixel-identical picture. Streamed lists must pass the command's own format bits to `GX_Begin`.
- **GX index formats**: `GX_INDEX8` vertex indices in display lists when a mesh has under 256 vertices.
