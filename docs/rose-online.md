# ROSE Online formats

What `gxbake/core/rose.py` reads, how it maps onto GoldSrc studio models, and what converting ROSE maps would take.

All formats are little-endian. Strings are NUL-terminated (EUC-KR in the original client; paths are ASCII). Paths inside files are VFS paths such as `3Ddata\NPC\animal\snail\body01_010.ZMS`, case-insensitive.

## Models (converted)

| Format | Contents | Notes |
| --- | --- | --- |
| ZMS (`ZMS0005`–`ZMS0008`) | format bits, bounding box, bone map, vertex streams (position, normal, colour, 4 weights + 4 bone indices, tangent, up to 4 UV sets), triangles, per-material face counts, strips | v7/8 use u16 counts and metres. v5/6 use u32 counts, prefix every vertex attribute and triangle with a u32 id, and store centimetres. Skin indices go through the bone map to ZMD bones. |
| ZMD (`ZMD0002`/`ZMD0003`) | bones (parent, name, position, rotation wxyz), then dummies (name, parent bone, position, rotation in v3) | Centimetres, rotations relative to the parent. Bone 0 is its own parent. Dummies are attachment points (weapons, effects); GxBake adds them as studio bones. |
| ZMO (`ZMO0002`) | fps, frame count, channels (type, bone), then frames × channels of values | Position (cm) and rotation (wxyz) tracks are baked. Normal/alpha/UV/texture/scale tracks are counted and skipped. `EZMO`/`3ZMO` trailers (frame events) are ignored. |
| ZSC | mesh paths, materials (texture path + flags: skin, alpha blend, two-sided, alpha test + ref, z flags, blend mode, specular, alpha, glow), effect paths, objects | Each object is parts with id/size/value properties: 1 position (cm), 2 rotation (wxyz), 3 scale, 5 bone, 6 dummy, 7 parent part (1-based), 30 per-part motion path. |
| CHR | skeleton paths, motion paths, effect paths, characters (enabled flag, skeleton, name, ZSC object ids, (motion slot, motion file) pairs, effects) | Motion slots used for sequence names: 0 stop, 1 walk, 2 attack, 3 hit, 4 die, 5 run, 6/8 cast, 7/9 skill, 10 etc. |
| DDS | DXT1/3/5 textures | Read by Pillow. Alpha is kept only when the material alpha-tests or blends; otherwise it is often a specular map and is dropped. |

### Mapping onto studio models

- ROSE and GoldSrc are both Z-up. Positions are scaled by `rose_scale` (GoldSrc units per metre, default 39.37) and turned `rose_yaw` degrees about Z (default 90: ROSE models face -Y, studio models face +X). Checked on `MALE.ZMD`: toes point forward and the left hand ends up on +Y.
- Studio skinning is rigid, so each vertex follows its heaviest weight. Blended weights produce a warning.
- Parts with a bone or dummy property are attached rigidly to it; skinned parts ignore their part transform.
- Studio models are single-sided, so two-sided materials get a reversed copy of their triangles.
- Motions are resampled to the **Animation fps** setting with slerp between ZMO frames. A motion that drives no bone of the skeleton is skipped.

### Tested on real data

Tested against files from a JDT01 client export: `LIST_NPC.CHR` (649 skeleton paths, 3,700 enabled characters), `PART_NPC.ZSC` (2,678 objects), `LIST_MBODY.ZSC`, `LIST_DECO_JDT.ZSC`, and the meshes, skeletons and motions listed below all parse. The snail NPC (`LIST_NPC.CHR` entry 33, 30 bones + 6 dummies) and the Junon `tree011` decoration convert, and their decoded lean MDLs match the source geometry. `EMPTY_STOP1/WALK/RUN_M1.ZMO` were baked on `MALE.ZMD`. That export has no NPC motion files, so NPC animation has only been checked with synthetic motions in the tests. Nothing has been run in the Xash3D GameCube engine yet.

## Avatars

Player characters are not in any CHR. The client assembles them at runtime from `MALE.ZMD`/`FEMALE.ZMD`, the part lists in `3DDATA/AVATAR/LIST_M*.ZSC` / `LIST_W*.ZSC` (body, arms, feet, face, hair, cap), and item tables (STB) that pick an object per slot. Until the STB tables are read, a `.rosemdl` manifest lists the parts and motions by hand (see the README).

## Maps (not converted yet)

A ROSE zone is:

- `.ZON` (one per zone): block table; zone info (grid per patch, grid size), event positions, the tile texture list, and tiles (two texture layers, offsets, blend flag, rotation).
- Per 16×16 block, named `<x>_<y>.*` next to the ZON:
  - `.HIM`: heightmap, `width × height` floats (typically 65×65 per block), after an 8-byte header.
  - `.TIL`: tile index per patch (16×16), each 3 skipped bytes + u32 into the ZON's tile list.
  - `.IFO`: placed objects in typed blocks: deco objects and buildings (`CNST`) referencing the zone's `LIST_DECO_*.ZSC` / `LIST_CNST_*.ZSC` with position, rotation and scale; animated objects; NPCs; monster spawns; water planes; warps; effects; sounds; event objects.
  - `.LIT`: per-object lightmap references; `PLANELIGHTINGMAP.DDS` lightmaps for the terrain.

The Xash3D GameCube port draws worlds as GoldSrc BSP (v30), so converting a map would mean:

1. **Terrain**: triangulate each block's 65×65 heightmap (2 triangles per cell, about 8k per block), and texture it from the ZON tiles (blended two-layer tiles need baking into one texture per patch or per block, since BSP faces take a single texture). The terrain lightmap can feed the BSP lightmap lump.
2. **Objects**: place the ZSC objects from the IFO files. Either bake them into the BSP as brush-less faces (only possible by writing the BSP directly, as GoldSrc compilers expect convex brushes) or keep each one as a studio model placed by an entity (`env_model`/`cycler_sprite`-style), which reuses this converter and keeps the BSP small.
3. **Collision/visibility**: GoldSrc needs hulls and a VIS tree. A heightfield cannot be expressed as brushes cheaply, so the realistic route is generating a `.map` with one brush per terrain quad (or per merged patch) and running the existing GoldSrc compile tools, or writing a simplified BSP with a single leaf and clip hulls from the heightfield.
4. **Entities**: NPCs, spawns, warps and sounds from the IFO become entities in the BSP's entity lump.

Of these, terrain + objects as studio models is the most direct first step; a zone like JDT01 is many blocks, so the result would also need splitting to fit the GameCube's memory. The HIM/TIL/IFO/ZON files are in the JDT01 export, so they can be tested when this work starts.
