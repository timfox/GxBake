# Panda3D and Toontown assets

GxBake converts Panda3D models, actors and animations (Toontown's included) into the
GoldSrc studio models the Xash3D GameCube port draws: `models/<name>.mdl`, then the lean
`gc_studio/<name>.mdl` and its `<name>t.mdl` skin companion. The code is
`gxbake/core/panda.py`; multifiles are in `gxbake/core/archives.py`.

## From a Toontown install to studio models

```sh
gxbake extract toontown/phase_*.mf -o src/toontown          # Panda multifiles → folders
gxbake convert src/toontown/phase_3/models/char -o staged     # every actor in the folder
gxbake budget staged                                          # does it fit the 1 MiB studio arena?
```

A folder scan lists the models and leaves out their animation files and the images in
`maps/` folders, which are pulled in through the models. Encrypted multifile members
(the newer fan servers encrypt theirs) need the game's password and are skipped.

## Formats

| File | How it is read |
| --- | --- |
| `.egg`, `.egg.pz` | parsed by GxBake: `<VertexPool>`, `<Polygon>`/`<TriangleStrip>`, `<Texture>` (with `alpha-file`), `<Material>`, `<Group>` + `<Dart>` (a character), `<Joint>` + `<VertexRef>` membership, `<Bundle>` animation tables (`<Xfm$Anim_S$>` and the older `<Xfm$Anim>`) |
| `.bam`, `.bam.pz` | converted to egg by Panda3D (`pip install panda3d`) or by `bam2egg` on `PATH` / `GXBAKE_BAM2EGG`, then read as above. Which BAM versions open depends on that Panda3D |
| `.mf` | `gxbake extract`: Panda multifile v1.x, header prefix lines, scale factor, zlib-compressed members |
| `.rgb`, `.sgi` | SGI images through Pillow (as textures and as loose images) |
| `.pandactor` | GxBake's JSON for multi-part actors (below) |

## Space, scale and facing

Eggs without a `<CoordinateSystem>` are Y-up, as Panda assumes; Y-up files are turned into
Panda's Z-up space the way Panda's loader does it (heading turns about the file's own up
axis). Panda is Z-up like GoldSrc, but its models face +Y, so the default **Turn** of −90°
makes them face GoldSrc's +X. Toontown measures in feet, so the default **Units per Panda
unit** of 12 keeps GoldSrc's inch scale.

The tests pose generated actors with Panda3D's own skinning and check that every vertex
of every frame lands in the same place in the studio model, for Z-up, Y-up and mixed files
and through `.bam`.

## Animations

Panda actors keep each animation in its own file. For a model, GxBake takes the
animation-only files whose names start with the model's family name, which is the stem
without a trailing LOD or model tag (`_1000`, `_500`, `-mod`, `-model`, `-zero`, `_hi`…):

- `panda-model.egg` + `panda-walk4.egg` → sequence `walk4`
- `tt_a_chr_dgm_shorts_legs_1000.bam` + `tt_a_chr_dgm_shorts_legs_neutral.bam` → `neutral`
- `suitA-mod.bam` + `suitA-walk.bam` → `walk`

In an extracted Toontown tree it also searches the same folder under every other
`phase_N` folder. An animation that drives none of the model's joints is skipped with a
warning. Sequences are sampled at **Animation fps**; neutral/idle ones come first, so the
engine's default sequence is a standing pose, and names like `die`, `jump` or `victory`
play once while the rest loop.

## Multi-part actors (`.pandactor`)

Toontown builds a toon from separate legs, torso and head characters and fastens each to
a joint of the one before. A `.pandactor` file does the same:

```json
{
  "parts": [
    {"model": "legs_1000.bam"},
    {"model": "torso_1000.bam", "attach": "joint_hips"},
    {"model": "head_1000.bam", "attach": "def_head"}
  ],
  "anims": {"wave": [null, "torso-wave.bam"]}
}
```

Paths are relative to the `.pandactor` file. Each part's root joints become children of
the `attach` joint, so the part follows it in every sequence. Animations are found per part
as above (set `"auto_anims": false` to turn that off); `anims` adds or replaces a clip per
part, in part order, with `null` for parts that keep their rest pose. A clip any part has
becomes one sequence, and parts without it stay in their rest pose.

The joint names above are an example; the inspector lists each part's joints under **Joints**.

## Not converted

- Toontown's `.dna` / `.pdna` street, interior and building layouts (maps)
- morph (blend shape) animation, UV scrolling and other non-joint tables
- texture transforms, second and later textures of a multitextured polygon
- blended skin weights: each vertex follows its heaviest joint (studio skinning is rigid)
- joint scale: studio bones carry rotation and translation only
