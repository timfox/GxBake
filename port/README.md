# Port-side readers for GxBake's packed formats

`gc_pack.h` / `gc_pack.c` are drop-in C for the Xash3D GameCube port (or any GameCube program) that read what `gxbake squeeze --emit`, `gxbake adpcm` and `gxbake yaz0` write. They are plain C89 with byte-wise big-endian reads and bounds checks. GxBake's tests (`tests/test_port_c.py`) compile them on the host with `-Werror` and check that they decode GxBake's output bit-exactly. The GX calls (`gc_gsk_bind`) sit behind `GC_PACK_GX`.

Copy both files into `engine/platform/gamecube/`, add `gc_pack.c` to the GameCube sources in `engine/wscript`, and define `GC_PACK_GX` for the renderer build.

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
