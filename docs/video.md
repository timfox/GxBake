# Video: THP movies

`gxbake video` turns any video FFmpeg can read into a **THP** file, the movie format of Nintendo's GameCube SDK and the one most commercial GameCube and Wii games play their cut-scenes from. The GUI does the same with **Convert Video…** on the toolbar, and **Convert** makes THP from videos when **Convert videos to** is `thp` or `both`.

```sh
gxbake video intro.mkv -o staged/valve/media/intro.thp
gxbake video movies/*.mp4 -o staged/valve/media --max 448x336 --max-size 24M
gxbake video logo.bik -o logo.thp --size 640x480 --fps 30 --no-audio
gxbake inspect staged/valve/media/intro.thp
```

## What goes in

Anything FFmpeg decodes: H.264/H.265/AV1/VP8/VP9/MPEG-1/2/4, WMV, Theora, ProRes, DV and so on, in MP4, MKV, MOV, AVI, WebM, MPEG-PS/TS, FLV, OGG, MXF and the rest. That includes game movie formats FFmpeg reads: Bink (`.bik`, `.bk2`), Smacker, RoQ, Interplay MVE, Westwood VQA, PlayStation STR, Xbox XMV, and THP itself (to re-encode at another size). `gxbake video` takes any file; the extensions in `media.VIDEO_EXTENSIONS` are the ones **Add Folder…** and `gxbake convert` pick up on their own. Rotated phone videos and non-square pixels (DVD) are handled.

## What comes out

| | |
| --- | --- |
| Picture | baseline JPEG, 4:2:0, standard Huffman tables, one per frame (every frame is a keyframe, so seeking and looping are free) |
| Size | the source fitted inside `--max` (default 512×448, so 4:3 is 512×384 and 16:9 is 512×288), never upscaled, sides rounded down to multiples of 16, at most 640×480. `--size WxH` sets it exactly, letterboxed unless `--stretch` |
| Frame rate | the source's, divided by the smallest whole number that gets it to `--max-fps` (30): 59.94 → 29.97, 60 → 30, 50 → 25, 24 stays 24. `--fps` sets it |
| Sound | stereo DSP ADPCM at 32000 Hz (`--audio-rate 48000` for the other native rate), 4.57 bits a sample: 36 KB/s. Mono sources are written as two equal channels, because the THP audio block is stereo |
| Budget | `--quality` (80) sets the JPEG quality. `--max-kbps` caps every frame: frames over it are re-encoded at the highest quality that fits, down to `--min-quality`. `--max-size 20M` caps the whole file and sets the per-frame cap from the length |
| Part of a video | `--start 12.5 --duration 30` |

The log line and `--json` give the frame count, bytes, kbit/s and the range of qualities the cap needed. On FFmpeg's `testsrc2` pattern (very busy) 512×288 at q80 is about 3.5 Mbit/s; real footage is usually well under that. Disc reads are around 2 MB/s, so stay under roughly 8 Mbit/s for streaming from disc with headroom.

## The format

All big-endian, version 1.1:

- **Header** (0x30 bytes): `THP\0`, version `0x00011000`, largest frame in bytes, most audio samples in one frame, fps (f32), frame count, first frame's size, total frame bytes, component info offset, frame offset table offset (0: none), first and last frame offsets.
- **Components**: a count, 16 type bytes (0 video, 1 audio, 0xFF unused), then video `width, height, 0` and audio `channels, rate, total samples, 1`.
- **Frames**, 32-byte aligned and back to back: `next frame size, previous frame size, picture size, audio size`, the picture (padded to 4 bytes), the audio block, padding to 32. The last frame's "next" is frame 0's size, so a looping player never needs the header again.
- **Audio block**: bytes per channel, samples, the 8 predictor pairs of each channel, each channel's two history samples at the block start, then each channel's ADPCM frames. Every block holds a whole number of 14-sample ADPCM frames and starts with its own history, so any frame's sound decodes on its own (the blocks are cut from one continuous encode, so playing them in order is seamless).
- **Pictures are not byte-stuffed.** In a .jpg every 0xFF in the compressed data is followed by a 0x00; in THP it is not (the frame header gives the picture's length, so the decoder never scans for markers). GxBake strips the stuffing when writing; `thp.to_jpeg` puts it back to open a frame in an image viewer. FFmpeg's THP decoder reads the files without errors, and the tests check that.

FFmpeg plays the result (`ffplay intro.thp`), which is the quickest check. Its THP audio decoder rounds differently from the GameCube's DSP, so low tones can sound slightly off in FFmpeg only; GxBake's decoder and `port/gc_thp.c` match the hardware's rounding.

## Playing it in the port

The SDK's THP player is Nintendo's, so the port needs its own. `port/gc_thp.c` / `gc_thp.h` have the parts that are not GX calls, in plain C89, host-tested against FFmpeg (pictures within ±1, audio bit-exact with GxBake's encoder):

- `gc_thp_parse` and `gc_thp_frame`: the header and one frame.
- `gc_thp_audio`: an audio block to interleaved stereo s16, which `ASND_SetVoice(..., VOICE_STEREO_16BIT, rate, ...)` plays as is.
- `gc_thp_jpeg`: a picture to Y, Cb and Cr planes (accurate integer IDCT, 9-bit Huffman lookup, no allocation).
- `gc_thp_rgb565_tiled`: the planes to a `GX_TF_RGB565` texture, the same kind of texture the GCVID player draws today.
- `gc_thp_i8_tiled`: one plane to a `GX_TF_I8` texture, for a later path that converts YUV to RGB in TEV instead of on the CPU.

See [port/README.md](../port/README.md#thp-movies-thp) for where it hooks into `avi_gc.c`. None of it has run on hardware yet. The SDK decoder reached 640×480 at 30 fps with paired-single and locked-cache tricks; plain C will be slower, so start at 448×336 or smaller and raise it once the decode time per frame is measured on the console.

## GCVID or THP?

`gxbake convert` still makes `.gcvid` + `.gcpcm` by default, because that is what the port plays today. GCVID stores RGB565 tiles with an 8×8 delta, so it costs almost no CPU to play, but every changed tile is stored raw: a fully changing 320×240 frame is 150 KiB. On FFmpeg's `testsrc2` pattern at the same 320×240 and 15 fps, GCVID is 2.2 MB for 4 seconds and THP (q80) 0.54 MB, 4× smaller; footage with more motion and detail widens the gap, a static picture narrows it. THP also carries its own sound, at the price of a JPEG decode per frame. Set **Convert videos to** (`--video-format`) to `thp` once the port has a THP player, or `both` while it is being written.
