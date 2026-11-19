# Port-side readers for GxBake's packed formats

`gc_pack.h` / `gc_pack.c` are drop-in C for the Xash3D GameCube port (or any GameCube program) that read what `gxbake squeeze --emit`, `gxbake adpcm` and `gxbake yaz0` write. They are plain C89 with byte-wise big-endian reads and bounds checks. GxBake's tests (`tests/test_port_c.py`) compile them on the host with `-Werror` and check that they decode GxBake's output bit-exactly. The GX calls (`gc_gsk_bind`) sit behind `GC_PACK_GX`.

Copy both files into `engine/platform/gamecube/`, add `gc_pack.c` to the GameCube sources in `engine/wscript`, and define `GC_PACK_GX` for the renderer build.

`gc_thp.h` / `gc_thp.c` are the same kind of drop-in for THP movies from `gxbake video` (see [THP movies](#thp-movies-thp) below). Add `gc_thp.c` to the sources the same way; it needs nothing else.

`gc_lmap.h` / `gc_lmap.c` read the `.gclm` face table that `gxbake gi` writes next to a map's radiosity lightmap pages (`<map>_lm.tpl`), and map a vertex's lightmap coordinates into a page. See [Radiosity lightmap atlas](#radiosity-lightmap-atlas-gclm) below.

`gc_mat.h` / `gc_mat.c` read the TPL textures that `gxbake auto`, `gxbake pbr` and `gxbake hdr` write, and the small `.gxm` material file next to them, and (behind `GC_MAT_GX`) set GX up to draw each preset. See [Materials and TPL textures](#materials-and-tpl-textures-tpl-gxm) below.

## Where each one hooks in

The `.gsk` and `.dsp` readers are wired into the HL1 port on a local branch (`gxbake-packing`, 2026-11-19), at the hooks below. That build compiles and boots in Dolphin, but neither path has run in a game yet. The other hooks are from reading the source.

### Indexed skins (`.gsk`): half the skin memory, same pixels

On HL1 `master` the hook is `Mod_StudioLoadTextures` / `R_StudioLoadTexture` in `ref/gx/r_studio.c`, beside the baked-CMPR route (`R_GCBakedCmprSkin`, `STUDIO_NF_GC_CMPR`). Older trees had `Mod_GCAttachStudioSkins` in `engine/common/mod_studio.c`. The HL1 branch:

- reads only the 32-byte entry table from the `.gsk`, using `gc_gsk_entry`'s layout;
- copies each skin's texels and TLUT into the existing studio skin window (`R_GCBakedGskSkin`, `r_image.c`);
- binds them in `R_GXNativeTexture` (`r_gx_world.c`), rotating TLUTs through `GX_TLUT12`–`15`.

`-gcnogsk` turns this off. The steps in general:

1. Try `gc_studio/<name>.gsk` and load it 32-byte aligned (`memalign(32, …)`), or just its table and then each skin on demand. Then call `DCFlushRange` on the bytes GX will read.
2. For each entry from `gc_gsk_entry`, the texels and the TLUT are already GX-tiled. Point the skin's texture object at them with `gc_gsk_bind` instead of building RGB565 in `R_StudioLoadTexture` / `R_GXSkinBuildTiled`. This removes that per-pixel CPU loop.
3. A CI skin needs its TLUT loaded in a TLUT slot (`GX_TLUT0`–`GX_TLUT15`) while it draws. The simplest way is to give each resident skin a slot round-robin and re-call `GX_LoadTlut` when a skin is bound (512 bytes).
4. If the `.gsk` is missing, the current RGB565 path is unchanged.

For `v_*.mdl` view models, GxBake has already applied `R_GXSkinBuildTiled`'s chromakey colours as transparent palette entries, so the alpha matches today's.

### DSP ADPCM sound (`.dsp`): 3.5× less sound memory

The simplest version decodes at load. On HL1 this is `FS_LoadSound` (`snd_main.c`). One ASND voice is fed by the software mixer (`snddma_gamecube.c`), so the decoded PCM goes into the usual sound cache. Watch out: most HL1 effects are 8-bit WAVs, and decoding `.dsp` to s16 doubles their MEM1 size. Sounds over the per-file cap are sent to ARAM by their `.wav` size before `FS_LoadSound` runs, so they never see the `.dsp`. Decode-at-load therefore only pays for 16-bit sounds, and the in-mixer version below is the real win. When `sound/<name>.dsp` exists, decode it once into the PCM buffer the mixer already uses: `gc_dsp_parse`, then `gc_dsp_decode(&dsp, 0, dsp.samples, &st, pcm)`. Loop points come from `dsp.loop_start` / `dsp.loop_end`. This saves disc space and lets the ARAM sfx store hold 3.5× more, but the MEM1 cache still holds PCM.

The full win keeps the `.dsp` bytes in the MEM1 cache and the ARAM store, and decodes in the mixer as it pulls samples. `gc_dsp_decode` streams: keep a `gc_dsp_state_t` per channel and call it with consecutive ranges. When a looping sound wraps, seed the state from `dsp.loop_h1` / `loop_h2` at `loop_start`. With that, the 128 KiB cache and the 20 KiB per-file cap hold 3.5× more audio.

### Music (`.gcpcm` → `.L.dsp` / `.R.dsp`)

In `engine/client/avi/avi_gc.c`, where the GCPCM stream is read in chunks, read the two `.dsp` files instead. Decode each chunk with `gc_dsp_decode` into the s16 stereo buffer that `ASND_AddVoice` takes. Disc traffic for music drops from 192 KB/s to 55 KB/s.

### s16 vertices (`.gsv`)

Studio vertices are transformed on the CPU in `R_StudioDrawPoints` (`ref/gx/r_studio.c`): `Matrix3x4_VectorTransform` into `g_studio.verts`. Load `gc_studio/<name>.gsv` next to the model, and point each model's vertex and normal source at its `gc_gsv_entry` arrays. Then set a GQR with type s16 and scale `pos_frac` (and s8/`nrm_frac` for normals), and read them with `psq_l` in the transform loop. Before that is written, `gc_gsv_dequant` gives the same floats, so a first version can dequantise into the existing float arrays at load. That saves nothing, but it proves the data. The display-list routes (`R_GXActorDraw` in `r_gx_actor.c`, `R_GXStudioDLTryDraw` in `r_gx_studiodl.c`) can take s16 positions directly with `GX_SetVtxAttrFmt(…, GX_POS_XYZ, GX_S16, pos_frac)`.

### Lightmap sizes (`gxbake lightmaps --plan`)

The atlas is RGB565, built in `engine/common/mod_bmodel.c` (`Mod_GCBakeMenuLightmapAtlas`, `GC_GetLiveLightmapAtlas`). It is capped in `GC_PackCapLightmapAtlas` (`vid_gamecube.c`) and bound by `R_GXBindLightmapAtlas` (`r_gx_world.c`). The plan's per-face `new_w`/`new_h`/`page`/`x`/`y` replace the allocator's own placement. A face's lightmap UVs scale by `(new − 1) / (old − 1)` per axis. A 1×1 face samples a single texel.

### Radiosity lightmap atlas (`.gclm`)

`gxbake gi` ([docs/gi.md](../docs/gi.md)) writes two things for a map. The relit `<map>.bsp` needs nothing new: its LIGHTING lump has the same per-face grids, so `Mod_GCBakeMenuLightmapAtlas` / the CapFaces atlas and the `×2` lightmap stage in `r_gx_world.c` show it as they show any map.

The GX-native pages replace the RGB565 atlas the port builds at load. Not wired in yet; the steps:

1. At map load, read `maps/<map>.gclm` and `maps/<map>_lm.tpl` (32-byte aligned, `DCFlushRange`) instead of running `Mod_GCBakeMenuLightmapAtlas`. `gc_lmap_parse`, then `gc_tpl_parse`. Skip the LIGHTING lump: the pages are the light.
2. Where the world baker or `R_GXBindLightmapAtlas` sets each face's lightmap texcoords, use `gc_lmap_face` and `gc_lmap_uv` with the engine's per-vertex `s`, `t` (luxels from `lightmapmins`). A face can be stored smaller than its grid (1×1 when flat); `gc_lmap_uv` handles that.
3. Bind page `f.page` with `gc_tpl_bind` on `GX_TEXMAP1` (pages are 1–8 per map, so group faces by page and switch rarely), and set the lightmap stage's scale from `lm.tev_scale` instead of the fixed `GX_CS_SCALE_2`. With the default "port" tone, scale 2 looks the same as today.
4. Pages are CMPR, RGB565, I4 or I8. An I4/I8 page reads as grey in TEVC, which is what a white-lit map needs.

Faces with no entry (`page == 0xFFFF`) are sky, water and other special faces; they keep their current path.

### Yaz0 (`*.yaz0`)

Any whole-file read can try `<path>.yaz0` first: `gc_yaz0_size`, then allocate, then `gc_yaz0_decode`. Good places are `Mod_GCLoadStudioFile` and the `.gct` pack load. The ARAM asset cache can store the compressed bytes and decode on the way back to MEM1.

### THP movies (`.thp`)

`gxbake video` writes THP, the SDK's movie format: one baseline JPEG picture and one stereo DSP ADPCM block per frame ([docs/video.md](../docs/video.md)). `gc_thp.c` reads it without GX calls and is host-tested: pictures match FFmpeg's THP decoder within ±1, audio is bit-exact with GxBake's encoder, textures are checked texel by texel.

A player in `engine/client/avi/avi_gc.c`, next to the GCVID path, when `media/<name>.thp` exists:

1. Read the first `GC_THP_HEADER_BYTES` and call `gc_thp_parse`. Allocate two frame buffers of `max_frame_bytes` (32-byte aligned), the planes (`width*height*3/2` bytes), an RGB565 texture (`width*height*2`, 32-byte aligned) and an audio buffer of `max_audio_samples * 4` bytes. At 512×288 that is about 0.55 MiB plus the two frame buffers.
2. Frame k is at `first_frame` with size `first_frame_bytes`; after that each frame's `next_bytes` gives the next size and the offset is the running sum. Read frame k+1 asynchronously (DVD) into the other buffer while frame k is shown.
3. `gc_thp_frame` splits the frame. `gc_thp_audio` decodes the sound block into interleaved s16; queue it on an ASND voice (`VOICE_STEREO_16BIT`, `audio_rate`), e.g. by double-buffering with `ASND_AddVoice`. Use the audio as the clock: show frame k when `k / fps` seconds of sound have played, and skip pictures (not sound) if decoding falls behind.
4. `gc_thp_jpeg` decodes the picture into the planes, `gc_thp_rgb565_tiled` writes the texture, then `DCFlushRange` it, `GX_InvalidateTexAll`, `GX_InitTexObj(..., GX_TF_RGB565, GX_CLAMP, GX_CLAMP, GX_FALSE)` and draw the screen quad the GCVID path draws today.
5. When decoding time per frame is known, the CPU colour conversion can be dropped: upload Y, Cb and Cr with `gc_thp_i8_tiled` as three `GX_TF_I8` textures (the chroma ones at half size, sampled with linear filtering) and do the YCbCr→RGB matrix in three or four TEV stages with konstant colours. That removes a per-pixel loop and a 16-bit write per pixel.

`gc_thp_jpeg` keeps its ~6 KiB of Huffman and quantisation tables in a static, so it is not reentrant: decode on one thread. It reads only what `gxbake video` writes (and what Nintendo's THPConv wrote): 4:2:0, baseline, no restart markers.

### Materials and TPL textures (`.tpl`, `.gxm`)

`gc_tpl_parse` / `gc_tpl_image` read any TPL (every GX format, palettes, mip chains). `gc_tpl_bind` makes the `GXTexObj` (and loads the TLUT for CI4/CI8), with trilinear filtering when the image has mips. `gc_tex_decode` decodes any GX format on the CPU, which is how the host tests check it texel for texel against GxBake's encoder.

A `.gxm` (12 + 4 bytes per map) says which preset a material is for, which TPL image holds each role (base, spec, glow, height, normal, env, lightmap), its alpha mode and cut-off, and a lightmap's TEV scale. `gc_mat_setup` then sets the texgens, lighting channels, TEV stages, alpha compare and blend:

| Preset | Stages | What GX does |
| --- | --- | --- |
| `prelit` | 1 | base × diffuse light (`unlit`: the texture alone, no channels) |
| `lit` | 1–3 | base × COLOR0A0, + spec mask × COLOR1A1 (`GX_AF_SPEC`), + glow |
| `emboss` | 3–5 | C0 = height@uv − height@`GX_TG_BUMPn` + ½, then base × light × 2·C0 |
| `indirect` | 2–5 | lit, + sphere map (`GX_TG_NRM` through `GX_TEXMTX0`) × spec mask, its lookup bent by the IA8 normal map in indirect stage 0 |
| `lightmap` | 2 | surface texture × lightmap (texcoord 1) × 2 or 4 (`GX_CS_SCALE_2/4`) |

For the sphere map, load `GX_TEXMTX0` each time the normal matrix changes: `gc_mat_sphere_texmtx( nrm, m ); GX_LoadTexMtxImm( m, GX_TEXMTX0, GX_MTX2x4 );`. The vertex format needs normals (`GX_VA_NRM`) for it, and a second UV set (`GX_VA_TEX1`) for lightmaps.

Hooks (not yet checked against the port's current source, and nothing here has run on hardware or in Dolphin; the host tests check the GX call sequence against a recording stub):

1. **Overbright lightmaps.** Where the world renderer combines a surface texture with its lightmap, replace the multiply stage with `gc_mat_lightmap_stage( stage, GX_TEXCOORD1, lightmap_map, scale )`, `scale` from the lightmap's `.gxm` (or 2 for every lightmap baked with `gxbake hdr --mode lightmap --overbright 2`). The lightmap atlas has to hold light / scale, which is what `gxbake hdr --mode lightmap` writes; today's lightmaps (light in 0..1) would come out 2× or 4× too bright.
2. **PBR materials on models and props.** Load `<name>.tpl` and `<name>.gxm` 32-byte aligned, `DCFlushRange` them, `gc_tpl_bind` each image into `bind.tex[role]` (roles from `mat.image[]`), set `bind.light_mask` to the lights the model uses, and call `gc_mat_setup` before drawing the mesh. Restore the renderer's own TEV state afterwards (the call changes texgens, channels, stage count, alpha compare and blend).
3. **Plain TPL textures** (`gxbake auto`, `--combine ui.tpl` writes a `.h` of image indices): `gc_tpl_image` + `gc_tpl_bind`, then draw as any other texture.

Two things to verify first on hardware: which components of an IA8 texel the indirect unit reads for s and t (the code assumes alpha → s, colour → t and swaps them back in the matrix; flip the matrix if reflections bend the wrong way), and the emboss sign (if bumps look inverted, swap the two height lookups).
