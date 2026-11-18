# RuneScape cache formats

What `gxbake/core/runescape.py` reads, how it maps onto GoldSrc studio models, and what is left. The layouts follow the community's documentation of the client cache (RuneLite's `cache` module and OpenRS2 are the usual references). Everything is big-endian.

## Store

| Layout | Files | Notes |
| --- | --- | --- |
| JS5 (OSRS; RS2 from 2006) | `main_file_cache.dat2`, `main_file_cache.idx0`…`idxN`, `main_file_cache.idx255` | Each idx entry is 6 bytes: size (u24), first sector (u24). Sectors are 520 bytes with an 8-byte header (archive u16, part u16, next sector u24, index u8), or 10 bytes (archive u32) for archive ids above 65535. |
| Legacy (2004-era RS2) | `main_file_cache.dat`, `idx0`…`idx4` | The same sectors, but the header stores the store number plus one. Store 1 holds the models, each gzipped. Configs live in "jag" archives in store 0, which GxBake doesn't read yet. |

JS5 archives are **containers**: compression (0 none, 1 bzip2 without its `BZh1` header, 2 gzip; 3 is RS3's LZMA and is refused), length (i32), then for compressed ones the uncompressed length (i32), the payload, and an optional trailing revision. XTEA-encrypted archives (map locations) are not read.

**Reference tables** (store 255, one archive per index): protocol 5–7 (7 uses "big smart" counts), flags (1 named, 4 sized), archive ids as deltas, then name hashes, CRCs, sizes, revisions, file counts and file ids. A multi-file archive ends with a chunk table: chunk count in the last byte, before it per chunk a delta-coded size per file.

## Indices used (OSRS numbering)

| Index | Contents | Read |
| --- | --- | --- |
| 0 | animation frames, grouped in archives | yes |
| 1 | framemaps ("skeletons" / bases) | yes |
| 2 | configs; archive 9 NPCs, 12 sequences (6 objects, 10 items are not read yet) | NPCs, sequences |
| 7 | models, one per archive | yes |
| 8 | sprites | for textures |
| 9 | textures (archive 0, one file per texture) | sprite id only |

## Models

The last two bytes select the encoding:

| Tail | Encoding | Header (at the end) |
| --- | --- | --- |
| anything else | original | 18 bytes: vertex/face counts, texture-triangle count, flags (textured, priority, alpha, face groups, vertex groups), x/y/z/face data lengths |
| `FF FF` | new | 23 bytes: plus texture types first in the file, 16-bit face texture ids and a texture-coordinate stream |
| `FF FE` | original + skeletal weights | 23 bytes: plus a "rigged" flag and the vertex-group section length |
| `FF FD` | new + skeletal weights | 26 bytes: likewise |

Vertices are delta-coded signed "smarts" per axis, gated by a flag byte. Faces use four compressions (1: three new indices; 2/3/4: reuse two of the previous face's indices and read one). In the original encoding a textured face stores its texture id in the colour slot and its texture triangle in the top six bits of the render-type byte. Skeletal weights (newer OSRS) are read past and ignored.

Texture coordinates come from **texture triangles** (P, M, N vertex indices): U runs along P→M and V along P→N, measured in the triangle's plane, exactly as the client does it. Faces whose texture triangle is a cylinder/cube/sphere mapping (RS2 HD era) fall back to the face's own corners.

## Mapping onto studio models

- **Axes.** Model space is x east, y down, z north, and a model with orientation 0 faces south. GoldSrc is X forward, Y left, Z up: `(x, y, z) → (-z, x, -y)`, times **Units per model unit** (default 0.35; 128 units are one tile), then **Turn** about Z. Face winding is kept: the client's normals are `(b - a) × (c - a)`, outward in this right-handed space.
- **Colours.** Face colours are 16-bit HSL (6 bits hue, 3 saturation, 7 lightness) and are converted with the client's formula and its default brightness (0.7). Each distinct colour gets a 4x4 block in a palette skin, which is exactly one CMPR block, so CMPR keeps every colour; past 256 colours near ones are merged so the skin stays 64x64. Flat-shaded faces get face normals, smooth faces averaged ones.
- **Hidden and translucent faces.** Render type 2 and alpha 254/255 faces are dropped. Other translucent faces are drawn opaque, since studio models have no per-face alpha.
- **NPCs.** Opcodes up to 101, plus 114–117, are read: models (1 or 61), name, size, stand/walk/run sequences, options, recolours (40), retextures (41), width/height scale (97/98). Opcode 102 (head icons) and the later ones changed size between revisions, so reading stops there; nothing GxBake uses comes after them.
- **Animation.** A frame names a framemap and lists operations on its slots: 0 sets the origin (the centroid of the slot's vertex groups plus an offset), 1 translates, 2 rotates (about z, then x, then y, in 2048ths of a turn), 3 scales by n/128 about the origin, 5 changes face alpha. Every group therefore moves rigidly, so each vertex group becomes a studio bone (parented to a root) and each frame's accumulated group transform becomes that bone's pose. Scale keeps its pivot but not its size; alpha steps are ignored. Frames are held for their length in 20 ms ticks and sampled at the **Animation fps** setting.

## Tested

Only against synthetic caches written by `tests/rs_fixtures.py` (all four model encodings, every face compression, bzip2/gzip/plain containers, protocols 5–7, archive ids above 65535, multi-chunk archives, legacy gzip models, NPCs, sequences, frames, sprites and both texture definition layouts). No real cache has been run through it yet, so the default scale, facing and winding still need a look on real data, and nothing has been run in the Xash3D GameCube engine.

## Not done yet

- Items (configs archive 10: inventory and worn models) and objects / locs (archive 6), which have their own model-type tables.
- Maps (index 5: terrain heights, overlays/underlays and locs; locs need XTEA keys), which would follow the same BSP route described in [rose-online.md](rose-online.md#maps-not-converted-yet).
- Skeletal ("animaya") sequences used by newer OSRS models.
- Legacy configs (NPC and sequence definitions inside the `config` jag archive) and legacy animation frames.
- Texture animation and the client's per-model lighting (ambient/contrast).
