# USD import

GxBake reads USD stages with [FreeUSD](https://github.com/gopexllc/freeusd), a
clean-room implementation of the OpenUSD file formats with Python bindings, and
bakes them into the GoldSrc studio models the GameCube port draws. A USD file goes
through the same steps as an Assimp import: `models/<name>.mdl` with bones,
sequences and paletted skins, then the lean `gc_studio/<name>.mdl` and the skin
companion `gc_studio/<name>t.mdl`.

```sh
pip install -e ".[usd]"     # builds FreeUSD's C++ extension: needs CMake and a C++17 compiler
gxbake inspect hero.usda
gxbake convert hero.usda -o staged/valve
```

`.usd`, `.usda`, `.usdc` and `.usdz` files show up as models in the app and the
CLI. The extra pins FreeUSD to the commit GxBake was tested against
(`usd.FREEUSD_COMMIT`); the first install compiles it, which takes several minutes.

## What is read

| USD | Becomes |
| --- | --- |
| Every `Mesh` that is visible and has `default` or `render` purpose | triangles (polygons are fanned); guide and proxy meshes are skipped |
| `primvars:normals` or `normals` | studio normals; smooth normals are made when there are none |
| `primvars:st` (or `st0`, `UVMap`, `uv`, `map1`, `TEXCOORD_0`), with `:indices` | skin coordinates, flipped to the top-left origin studio models use |
| `material:binding` on the mesh or an ancestor, and `materialBind` GeomSubsets | one skin per material; triangles split by subset |
| `UsdPreviewSurface` `diffuseColor` texture (`UsdUVTexture` `inputs:file`) | the skin texels (power-of-two, cut-out alpha as masked index 255) |
| `diffuseColor` value, or the mesh's `displayColor` | a solid-colour skin when there is no texture |
| `Skeleton` joints, `restTransforms`, `bindTransforms` | studio bones under a `root` bone |
| `primvars:skel:jointIndices` / `jointWeights` (any `elementSize`), `skel:joints`, `geomBindTransform` | each point follows its heaviest joint (studio skinning is rigid) |
| `SkelAnimation` bound through `skel:animationSource` | one looping sequence over the stage's start and end time codes, sampled at **Animation fps** |
| Time-sampled xformOps on a mesh or its parents | a bone for that mesh, animated in the same sequence |
| `<model>@<clip>.usda` beside the model | another sequence, named after the clip; its SkelAnimation is matched to the skeleton by joint names |
| `upAxis`, `metersPerUnit` | the axis swap (Y-up sources turn so +Z forward becomes +X forward) and the scale: one GoldSrc unit is an inch, and **Scale** multiplies on top |
| `timeCodesPerSecond` | sequence timing |

Composition goes through FreeUSD: sublayers, references and payloads work. Textures
are looked up next to the layer that names them, then in `textures/`, `tex/` and
`maps/` folders, by basename and under other extensions, the same way as for Assimp
imports. A `.usdz` is unpacked to a temporary folder, and its first layer is the root.

Not carried over: blend shapes, blended skin weights beyond the heaviest joint,
point instancers, subdivision (the control cage is used), lights, cameras and
physics. For materials other than UsdPreviewSurface (MaterialX, for example),
GxBake takes the image file under the material that looks like a colour map, if
there is one, and otherwise the mesh colour.

## FreeUSD gaps and what GxBake does about them

These hold for FreeUSD 0.1.0 at commit `f5bcd39`. GxBake works around the first
group so that files written by Pixar's USD library, Blender, Houdini or Omniverse
import correctly; the second group needs fixes in FreeUSD itself.

Worked around in `gxbake/core/usd.py`:

- **Attribute metadata blocks.** An attribute followed by `( interpolation = ... )`
  or `( elementSize = ... )` loses its value, and nearly every primvar a DCC writes
  has one. ASCII layers are rewritten into a temporary folder before FreeUSD opens
  them, with the blocks cut out; GxBake keeps the interpolation and element size it
  cut and applies them when it reads the primvars (and infers them from the value
  count when a layer it couldn't map, such as a referenced one, doesn't say).
- **`uniform` / `custom` qualifiers** end up inside attribute names
  (`token subdivisionScheme`), so the rewrite drops them.
- **Half-precision types** (`half3[]`, `quath`, ...) don't parse; the rewrite reads
  them as float.
- **Array values** have no Python accessors; GxBake parses their text form.
- **`Xformable.compute_local_to_world_transform`** multiplies parent and child in
  the wrong order (a parent's translation gets the child's scale). World matrices
  are composed from `compute_local_transform` instead.
- **Prim type names** are parsed and dropped, so there is no way to ask whether a
  prim is a Mesh. Meshes, skeletons, animations and subsets are recognised by the
  attributes they carry.
- **Referenced prims** below the first level are composed (`prim_at` finds them)
  but not traversed, and relationship targets and shader connections inside a
  referenced layer keep that layer's paths. GxBake asks for the referenced layer's
  prims by path, re-roots relationship targets under the referencing prim, and,
  when a material's connections are gone, takes the colour texture from the
  `UsdUVTexture` under it.

Not worked around:

- **Binary crate files** (`.usdc`, and `.usdz` packages holding one, which is what
  most tools write) only open when they embed a USDA section. Save them as USDA
  with `usdcat -o model.usda model.usdc` (from Pixar's `usd-core`), or export USDA
  from the DCC.
- **Variant sets** in the standard `variantSet "shape" = { "big" { ... } }` form
  don't parse; FreeUSD reads its own `variantSets = { ... }` metadata form only.
  Flatten the variant selection before converting.
- **Single-line prim bodies** (`def Cube "Box" { double size = 1 }`) don't parse.
- **Implicit surfaces** (`Cube`, `Sphere`, `Cylinder`...) have no points, so they
  are not converted; with type names dropped they can't be told apart either.

## Licence

FreeUSD's `LICENSE` file is MIT, but its `pyproject.toml` declares
`GPL-3.0-or-later` and its engine docs talk about GPL-compatible engines. Until
that is settled upstream, FreeUSD stays an optional extra: it is not part of the
`dev` or `gui` extras, so the PyInstaller executable doesn't bundle it.
