# Porting a game's assets to the GameCube with GxBake

This walks through the commands in the order a port uses them, from the original
game files to a staged tree that fits the Xash3D GameCube port. Every option is
listed in the [command line reference](cli.md).

The GameCube has 24 MiB of main RAM, 16 MiB of ARAM and a 1.36 GiB disc. Those
totals matter less than the windows the port gives each kind of data while a map
loads. GxBake checks against those windows:

| Window | Size | Checked by |
| --- | --- | --- |
| World textures, MEM1 pool | 448 KiB | `budget`, fixed by `fit` |
| World textures, ARAM store | 6 MiB, 512 entries | `budget`, fixed by `fit` |
| Studio models, MEM1 arena | 1 MiB per map | `budget`, reduced by `slim` |
| One sound effect | 20 KiB of PCM (48 KiB on the menu route) | `budget`, fixed by `sfx` |
| Sound cache / ARAM store | 128 KiB / 2 MiB | `budget` |
| Lightmap atlas | 800 KiB | `budget` |
| Disc | 1,459,978,240 bytes | `budget`, `disc` |

## 1. Get the files out of their archives

Quake and GoldSrc `.pak`, Source `.vpk` and `.pk3` archives unpack with `extract`:

```sh
gxbake extract valve/pak0.pak -o src/valve
gxbake extract hl2/hl2_misc_dir.vpk -o src/hl2 --match 'models/*' --match 'materials/*'
```

## 2. Look before converting

`inspect` prints what GxBake reads from a file or folder: texture lists, model
bones and sequences, a map's sky and WADs, and warnings such as mixed-case names
in case-strict folders.

```sh
gxbake inspect src/valve/maps/c1a0.bsp
gxbake inspect src/valve/models/barney.mdl
```

## 3. Convert

`convert` takes a whole gamedir or single files and writes the staged tree:
`media/wt_<map>.gct` world texture packs and skies, `gc_studio/` lean models and
skin companions, downsampled sprites, imported 3D models, and video and audio
companions.

```sh
gxbake convert src/valve -o staged/valve --cache ~/.cache/gxbake \
	--fit-world --mdl-anim-step 2
```

- `--fit-world` trims each map's texture pack until it fits the MEM1 pool (see step 5).
- `--mdl-anim-step 2` halves the animation frames in every `gc_studio` model (see step 6).
- `--cache` keeps encoded CMPR between runs, so a second run only encodes what changed.

New models can come from any format Assimp reads, and USD stages (`.usda`, `.usdz`)
come in through FreeUSD once the `usd` extra is installed; see [USD import](usd.md).

### Skies

A map's sky comes from `gfx/env/<skyname><side>.tga` when the BSP is converted.
To give a map a new sky, or to bring one in from a panorama, make the sides with
`sky` and set the map's worldspawn `"skyname"` to the same name:

```sh
gxbake sky art/sunset_4k.hdr -o staged/valve --name sunset --preview sunset.png
```

It reads equirectangular panoramas (including `.hdr`), cubemap crosses and strips,
six face images and flat photos, and writes the six 8 KiB sides. `--yaw` turns
the sky (270, the default, puts the panorama's centre on the `ft` side). The six
sides live in a fixed 48 KiB block, so a sky never counts against a map's texture
pool.

## 4. Check the budget

```sh
gxbake budget staged/valve --include 'models/v_*.mdl' --include models/player.mdl
```

Each map gets a line per window, `ok`, `WARN` or `OVER`, with the largest items
listed under any line that is not ok. `--include` adds models the game code loads
on every map but no map entity names (view models, the player). The command exits
with 1 when anything is `OVER` (add `--strict` to fail on `WARN` too), so it can
gate CI:

```sh
gxbake budget staged/valve --strict --json > budget.json
```

## 5. Fix world textures

A map's pack is `WARN` when it fits the pool only at a softer split (textures over
32 or 16 pixels draw from low mips until they stream in) and `OVER` when it does
not fit at all. `fit` drops the top mip level of the largest textures until it
fits, with no re-encode:

```sh
gxbake fit staged/valve/media/wt_c1a0.gct                     # dry run: what would be halved
gxbake fit staged/valve/media/wt_c1a0.gct -o staged/valve/media/wt_c1a0.gct
```

To look at or hand-edit a pack, use `pack`:

```sh
gxbake pack list staged/valve/media/wt_c1a0.gct --sort size   # biggest textures first
gxbake pack extract staged/valve/media/wt_c1a0.gct -o review/c1a0
gxbake pack add staged/valve/media/wt_c1a0.gct fixes/crate01.png --max-edge 64
gxbake pack remove staged/valve/media/wt_c1a0.gct 'debug*'
```

`pack extract` decodes what the GameCube will sample, so a texture that looks
wrong there looks wrong on hardware. A replacement image added with `pack add`
takes over the entry of the same name.

## 6. Fit studio models in the arena

When the studio line is over, find out what the models are made of:

```sh
gxbake slim staged/valve/models/scientist.mdl
```

For NPCs, animation is usually the largest part. `--step 2` keeps every second
frame and adjusts the frame rate so each sequence lasts as long as before.
`--stub` reduces sequences a port does not need (cutscene-only, deaths) to one
frame without removing them, so the game code's sequence numbers stay valid:

```sh
gxbake slim src/valve/models/scientist.mdl --step 2 --stub 'death*' -o src/valve/models/scientist.mdl
```

Run `convert` again afterwards so the `gc_studio` mesh is rebuilt from the slimmed model.

## 7. Fit sound effects

A WAV over the per-file cap never plays. `budget` names the ones a map uses;
`sfx` shrinks them, trying mono, 11 kHz and 8-bit in order of least loss, and
keeps loop points:

```sh
gxbake sfx src/valve/sound -o staged/valve/sound            # gameplay cap, 20 KiB
gxbake sfx src/valve/sound/ambience --menu --in-place       # menu route, 48 KiB
```

What still does not fit is reported; `--truncate` cuts it with a short fade.

## 8. Check the disc and the tree

```sh
gxbake disc staged/valve        # sector cost by folder, duplicates, unused models and sounds
gxbake validate staged/valve    # the disc builder's own checks
```

`disc` lists files no map names under `models/`, `gc_studio/` and `sound/`. The
game code may still load them, so check before deleting anything.

## Putting it in a script

```sh
#!/bin/sh
set -e
gxbake convert src/valve -o staged/valve --cache ~/.cache/gxbake --fit-world --mdl-anim-step 2
gxbake sfx staged/valve/sound --in-place
gxbake validate staged/valve
gxbake budget staged/valve --include 'models/v_*.mdl' --include models/player.mdl
gxbake disc staged/valve
```

Each step exits with a non-zero status on failure, so `set -e` stops at the first
problem.
