# Port-side readers for GxBake's packed formats

`gc_pack.h` / `gc_pack.c` are drop-in C for the Xash3D GameCube port (or any GameCube program) that read what `gxbake squeeze --emit`, `gxbake adpcm` and `gxbake yaz0` write. They are plain C89 with byte-wise big-endian reads and bounds checks. GxBake's tests (`tests/test_port_c.py`) compile them on the host with `-Werror` and check that they decode GxBake's output bit-exactly. The GX calls (`gc_gsk_bind`) sit behind `GC_PACK_GX`.

Copy both files into `engine/platform/gamecube/`, add `gc_pack.c` to the GameCube sources in `engine/wscript`, and define `GC_PACK_GX` for the renderer build.

`gc_thp.h` / `gc_thp.c` are the same kind of drop-in for THP movies from `gxbake video` (see [THP movies](#thp-movies-thp) below). Add `gc_thp.c` to the sources the same way; it needs nothing else.

## Where each one hooks in

These hooks are based on the port's source as of 2026-11-18. None of them has been built or run on hardware.

### Indexed skins (`.gsk`): half the skin memory, same pixels

In `Mod_GCAttachStudioSkins` (`engine/common/mod_studio.c`), before `Mod_GCLoadStudioTexFile`:

1. Try `gc_studio/<name>.gsk` and load it 32-byte aligned (`memalign(32, …)`). Then call `DCFlushRange` on it.
2. For each entry from `gc_gsk_entry`, the texels and the TLUT are already GX-tiled. Point the skin's texture object at them with `gc_gsk_bind` instead of building RGB565 in `R_StudioLoadTexture` / `R_GXSkinBuildTiled`. This removes that per-pixel CPU loop.
3. A CI skin needs its TLUT loaded in a TLUT slot (`GX_TLUT0`–`GX_TLUT15`) while it draws. The simplest way is to give each resident skin a slot round-robin and re-call `GX_LoadTlut` when a skin is bound (512 bytes).
4. If the `.gsk` is missing, the current RGB565 path is unchanged.

For `v_*.mdl` view models, GxBake has already applied `R_GXSkinBuildTiled`'s chromakey colours as transparent palette entries, so the alpha matches today's.

### DSP ADPCM sound (`.dsp`): 3.5× less sound memory

The simplest version decodes at load. In `Sound_LoadWAV`'s caller (`S_LoadSound`), when `sound/<name>.dsp` exists, decode it once into the PCM buffer the mixer already uses: `gc_dsp_parse`, then `gc_dsp_decode(&dsp, 0, dsp.samples, &st, pcm)`. Loop points come from `dsp.loop_start` / `dsp.loop_end`. This saves disc space and lets the ARAM sfx store hold 3.5× more, but the MEM1 cache still holds PCM.

The full win keeps the `.dsp` bytes in the MEM1 cache and the ARAM store, and decodes in the mixer as it pulls samples. `gc_dsp_decode` streams: keep a `gc_dsp_state_t` per channel and call it with consecutive ranges. When a looping sound wraps, seed the state from `dsp.loop_h1` / `loop_h2` at `loop_start`. With that, the 128 KiB cache and the 20 KiB per-file cap hold 3.5× more audio.

### Music (`.gcpcm` → `.L.dsp` / `.R.dsp`)

In `engine/client/avi/avi_gc.c`, where the GCPCM stream is read in chunks, read the two `.dsp` files instead. Decode each chunk with `gc_dsp_decode` into the s16 stereo buffer that `ASND_AddVoice` takes. Disc traffic for music drops from 192 KB/s to 55 KB/s.

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
