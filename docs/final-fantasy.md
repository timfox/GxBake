# Final Fantasy formats

What `gxbake/core/ffxi.py` and `gxbake/core/ff7.py` read, how they map onto GoldSrc studio models, and what is still guessed. No game data is in the repository: the tests write their own files (`tests/ffxi_fixtures.py`, `tests/ff7_fixtures.py`), and nothing here has been run on real game files yet.

References, read for layout only (no code copied):

- **POLUtils** (Windower, Apache-2.0): FFXI file tables, image headers, the SeWave/BGMStream headers and the ADPCM codec.
- **Vanalytics** `src/Vanalytics.Web/src/lib/ffxi-dat` (Soverance, MIT): the FFXI chunk chain and the skeleton, mesh, motion and texture chunks, and how its viewer orients and plays them.
- **Qhimm wiki** (now the ff7-flat-wiki): FF7 `P`, `LGP_format`, `TEX_format`, `PSX/HRC`, `PSX/RSD`.
- **Kimera** (field model editor): the `.a` frame layout and how a field skeleton chains its bones.

## Final Fantasy XI (PC)

### Install layout

| Path | What |
| --- | --- |
| `VTABLE.DAT`, `FTABLE.DAT` | base game tables: one byte per file id (1 = in `ROM`), one uint16 per file id packing `dir << 7 | file` |
| `ROMn/VTABLEn.DAT`, `ROMn/FTABLEn.DAT` | expansion tables (n = 2…9): a byte equal to n means "in `ROMn`" |
| `ROM*/<dir>/<file>.DAT` | the data |
| `sound*/win/se/**.spw`, `sound*/win/music/data/**.bgw` | sound effects and music |

`ffxi.Rom(root).path(id)` resolves an id (case-insensitive on Linux).

### Chunk files

A model DAT is a chain of chunks, each with a 16-byte header: 4-character name, uint32 (`type = v & 0x7F`, `length = ((v >> 7) & 0x7FFFF) * 16`, header included), 8 unused bytes. The chain must tile the file exactly, which is how GxBake tells an FFXI DAT from any other `.dat` (it walks only the headers, so scanning a ROM folder stays cheap). Only DATs with a mesh chunk become model assets.

| Type | Chunk | Read |
| --- | --- | --- |
| 0x20 | texture | flag 0xA1/0x81: 16-byte name, a 40-byte BITMAPINFOHEADER, then a reversed FourCC (`1TXD`, `3TXD`), size and block count, then DXT data. 0xB1/0x91: the same header, then a 256-entry BGRA palette and 8-bit pixels |
| 0x29 | skeleton | uint16, bone count (int16), 30 bytes per bone: parent (255 = root), leaf flag, quaternion xyzw, translation |
| 0x2A | mesh | see below |
| 0x2B | motion | version, element count, frame count, speed (float); 84 bytes per element: bone, 4 rotation key indices + defaults, 3 translation + defaults, 3 scale + defaults; key index 0 means "use the default", a negative first index means "not animated". Indices count floats from just after the 10-byte header. Plays at 30·speed fps |
| 0x1C, 0x2E | zone MZB / MMB | no |

**Meshes.** A 64-byte header with offsets in 16-bit words from the chunk body: polygon commands, bone table, weight counts (single-bone and two-bone vertex counts), bone assignments (two words per vertex: low/high bone, each 7 bits left, 7 bits mirror bone, 2 bits mirror axis), and vertices. Type bit 0x80 means the assignments index the bone table; type 1 is cloth (no normals). Single-bone vertices are position + normal (24 bytes, 12 for cloth); two-bone vertices interleave both bones (x1 x2 y1 y2 z1 z2 w1 w2, then normals; 56 or 32 bytes), with each position already scaled by its weight. Polygon commands: `0x8000` texture change (16-byte name), `0x8010` material state (46 bytes, skipped), `0x0054` triangle list (30 bytes per triangle with UVs), `0x5453` strip (a first triangle, then 10 bytes per corner), `0x4353`/`0x0043` skipped, `0xFFFF` end. A non-zero flip word draws a second, mirrored copy bound to the mirror bones.

**How it becomes a studio model.** Bones keep the skeleton's hierarchy. Every vertex is put in bind pose (two-bone vertices blended), then bound to its heavier bone, since studio skinning is rigid. The mirrored half is bound to the mirror bones with its winding reversed. Each mesh is wound counter-clockwise from outside by checking its triangles against the stored normals (D3D data is clockwise; the check also keeps mirrored halves right). Motion keys follow Vanalytics' tested playback: a key's rotation is applied before the bone's bind rotation (`q = key · bind`) and its translation is added to the bind translation. Scale keys are dropped (with a warning).

**Guessed, to check on real data:**

- The axes: FFXI is y-down, and characters are taken to face +X (from how Vanalytics turns them to face its camera). `--ffxi-yaw` turns them.
- The unit: one FFXI unit is taken as a metre (39.37 GoldSrc units).
- DXT3 alpha: POLUtils treats nibble 8 and up as opaque (FFXI's half-range alpha); Vanalytics scales the nibble to 0–255. GxBake follows POLUtils.
- DXT1 punch-through black is drawn opaque, as Vanalytics does (FFXI blends by material).
- The palette of a 0xB1 texture starts at byte 61 (POLUtils) or 64 (Vanalytics). GxBake takes whichever start gives FFXI-looking alpha bytes (0…0x80). Rows are read top-down.
- Motion chunks are grouped into one sequence when they are consecutive, share the first three characters of their name, the frame count and the speed, and drive different bones.

### Sounds

`.spw` starts `SeWave\0\0`: size, format, id, sample blocks, loop start, rate (two halves, added), 4 bytes, channels, block size, 4 bytes; data at 0x30. `.bgw` starts `BGMStream\0\0\0` with format and size swapped and no trailing word. Format 0 is ADPCM: per channel per block, a header byte (low nibble: shift `12 - n`; high nibble: one of five predictor filters) then `block_size / 2` bytes of 4-bit samples, low nibble first, channels one after the other. Format 1 is raw PCM. Format 3 (ATRAC3) is refused.

Effects become 16-bit WAVs with a `cue ` loop point, then go through `sfx.fit_wav_bytes` (the engine's 20 KiB per-file cap). Music becomes 48 kHz GCPCM.

## Final Fantasy VII (PC)

### LGP archives

12-byte creator (`\0\0SQUARESOFT`, or `FICEDULA-LGP` for patches), file count, then 27-byte entries: 20-byte name, data offset, a check byte and a duplicate-name id. A 3600-byte lookup table follows, then (only when some names repeat) a folder table: uint16 count, then groups of uint16 count × (128-byte folder, uint16 entry index). Each file's data starts with its own 20-byte name and a uint32 length. `gxbake extract` reads them like PAK/VPK/ZIP; repeated names without a folder table get `~2`, `~3` suffixes.

### Field models

| File | Read |
| --- | --- |
| `.hrc` | text: header lines starting `:`, then 4 lines per bone: name, parent name (`root` for the first), length, and `<n> <rsd names…>` |
| `.rsd` | text: `PLY=` (the `.p` with the same stem), `TEX[i]=` (`.TIM` names; the PC files are `.TEX`) |
| `.p` | 128-byte header (16 counts, then 64 bytes of runtime data); vertices, normals, an unknown float3 block, UVs, one BGRA colour per vertex, one per polygon, edges, 24-byte polygons (vertex, normal and edge indices), two unknown blocks, 100-byte render states, 56-byte groups (polygon and vertex start/count, UV start, textured flag and texture number). Vertex and UV indices are relative to their group; normal indices are not |
| `.tex` | 0xEC-byte header (size, palette flag, palette size, bytes per pixel, channel masks and shifts, colour key flag), BGRA palette, pixels. With the colour key on, pixel value 0 is transparent |
| `.a` | version 1, frame count, bone count, rotation order bytes, 20 bytes runtime; per frame a root rotation, root translation and a rotation per bone, all floats, angles in degrees |

**Posing.** Following Kimera: the root is translated by (x, -y, z) and rotated; then each bone, in file order, starts at the end of its parent, rotates by `Ry(β)·Rx(α)·Rz(γ)`, draws its parts, and moves `-length` along its own Z for its children. Studio bone 0 is the root and HRC bones follow it. Without an animation, every rotation is zero and the skeleton lies straight. Animations are resampled from 30 fps to **Animation fps**, interpolating angles the short way round.

**Materials.** Textured groups use their `.tex`; untextured ones average their vertex colours per face into a palette skin (one 4x4 block per colour). FF7 data winds triangles both ways, so each triangle is turned to agree with its normals (or left as it is when the file has none, like battle models).

**Guessed, to check on real data:** the facing (-Z taken as forward, as on the PlayStation; `--ff7-yaw` turns it), the scale (2.5 makes a field character about 70 units tall), and the handedness (a mirrored character would show as left and right swapped).

### Battle models

`battle.lgp` keeps each model as extensionless files named from a two-letter base (`rt` is Cloud):

| File | Read |
| --- | --- |
| `<base>aa` | skeleton: bone count (int32 at 12), texture count (24), body animation count (28), weapon animation count (40); from byte 52, 12 bytes per bone: parent (-1 = root), length (float), has-part flag. A bone count of 0 means a battle location (not read) |
| `<base>ac`, `ad`, … | `.tex` textures, as many as the header says; group texture numbers index them |
| `<base>am` … `az`, `ba`, … | one `.p` part per bone, in bone order (a bone without a part still takes a name) |
| `<base>ck` … | weapon models (not read) |
| `<base>da` | animation pack: uint32 count, then the body animations, then the weapon animations |

Each pack block is 12 bytes (bone count + 1, frame count, block length; both counts are unreliable), then, when the length is 11 or more, a mini header (int16 frame count, uint16 stream length, key byte) and the bit stream. A block whose first int16 equals `length - 5` has no frame count (3-byte mini header). Shorter blocks are empty slots and keep their number. The stream is read MSB first and, as Kimera does, frame by frame until it runs out:

- First frame: 3 × int16 root offsets, then for the root rotation and each bone 3 × `(12 - key)`-bit signed values, shifted left by `key`.
- Later frames: per offset, 1 bit picks a 7- or 16-bit signed delta. Per rotation value, a 0 bit means no change; otherwise 3 bits: 0 is `-1 << key`, 7 is a full `(12 - key)`-bit value, and 1–6 read that many bits `t`, which become `t - 2^(n-1)` when negative or `t + 2^(n-1)` otherwise, then `<< key`.
- Deltas add to the previous frame. 4096 is a full turn.

**Posing.** As in Kimera's battle viewer: translate the root by the frame's offsets (no y flip, since the parts are y-down too), rotate by slot 0, then each bone starts at the end of its parent, rotates `Ry·Rx·Rz` by its slot, draws its part and moves `+length` along Z (lengths are stored negative). A one-bone model uses slot 0 for both. Animations are resampled from 15 fps.

**Guessed, to check on real data:** the frame rate (15 fps, one frame per battle tick), the auto scale (each model fit to 72 units tall unless `--ff7-battle-scale` is set; the auto factor is printed as a warning), and the winding: battle parts have no normals, so triangles keep the file's order.

**Not read:** battle weapons and weapon animations, battle locations, the world map, battle scenes, and the PSX formats.
