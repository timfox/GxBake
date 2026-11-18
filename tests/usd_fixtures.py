"""USD test scenes, written the way Pixar's USD library writes USDA.

The text below matches what usd-core 0.26 saves (uniform qualifiers,
attribute metadata blocks, half3 scales, GeomSubsets), so the tests cover the
layer rewrite as well as the import.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

CHAR_USDA = '''#usda 1.0
(
    defaultPrim = "Char"
    endTimeCode = 24
    metersPerUnit = 0.01
    startTimeCode = 0
    timeCodesPerSecond = 24
    upAxis = "Y"
)

def SkelRoot "Char" (
    prepend apiSchemas = ["SkelBindingAPI"]
)
{
    rel skel:animationSource = </Char/Skel/Wave>

    def Skeleton "Skel"
    {
        uniform matrix4d[] bindTransforms = [( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) ), ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 50, 0, 1) ), ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 100, 0, 1) )]
        uniform token[] joints = ["Hips", "Hips/Spine", "Hips/Spine/Head"]
        uniform matrix4d[] restTransforms = [( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) ), ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 50, 0, 1) ), ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 50, 0, 1) )]

        def SkelAnimation "Wave"
        {
            uniform token[] joints = ["Hips/Spine", "Hips"]
            quatf[] rotations.timeSamples = {
                0: [(1, 0, 0, 0), (1, 0, 0, 0)],
                12: [(0.70710677, 0, 0, 0.70710677), (1, 0, 0, 0)],
                24: [(1, 0, 0, 0), (1, 0, 0, 0)],
            }
            half3[] scales = [(1, 1, 1), (1, 1, 1)]
            float3[] translations.timeSamples = {
                0: [(0, 50, 0), (0, 0, 0)],
                24: [(0, 50, 0), (0, 0, 0)],
            }
        }
    }

    def Mesh "Body" (
        prepend apiSchemas = ["SkelBindingAPI", "MaterialBindingAPI"]
    )
    {
        float3[] extent = [(-10, 0, 0), (10, 120, 0)]
        int[] faceVertexCounts = [4, 4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 3, 2, 4, 5, 5, 4, 6, 7]
        rel material:binding = </Char/Looks/Green>
        point3f[] points = [(-10, 0, 0), (10, 0, 0), (10, 50, 0), (-10, 50, 0), (10, 100, 0), (-10, 100, 0), (10, 120, 0), (-10, 120, 0)]
        normal3f[] primvars:normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        matrix4d primvars:skel:geomBindTransform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
        int[] primvars:skel:jointIndices = [0, 1, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1, 2, 1, 2, 1] (
            elementSize = 2
            interpolation = "vertex"
        )
        float[] primvars:skel:jointWeights = [1, 0, 1, 0, 0.8, 0.2, 0.8, 0.2, 0.9, 0.1, 0.9, 0.1, 1, 0, 1, 0] (
            elementSize = 2
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "faceVarying"
        )
        int[] primvars:st:indices = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        rel skel:skeleton = </Char/Skel>

        def GeomSubset "HeadFaces" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [2]
            rel material:binding = </Char/Looks/Red>
        }
    }

    def "Looks"
    {
        def Material "Green"
        {
            token outputs:surface.connect = </Char/Looks/Green/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Char/Looks/Green/Tex.outputs:rgb>
                token outputs:surface
            }

            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @tex/green.png@
                float3 outputs:rgb
            }
        }

        def Material "Red"
        {
            token outputs:surface.connect = </Char/Looks/Red/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Char/Looks/Red/Tex.outputs:rgb>
                token outputs:surface
            }

            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @tex/red.png@
                float3 outputs:rgb
            }
        }
    }

    def Xform "Prop"
    {
        double3 xformOp:translate.timeSamples = {
            0: (40, 0, 0),
            24: (40, 30, 0),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Mesh "Box"
        {
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]
            color3f[] primvars:displayColor = [(0, 0, 1)]
        }
    }

    def Mesh "Guide"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        uniform token purpose = "guide"
    }
}
'''

# A clip with only the skeleton's animation: <name>@<clip>.usda.
CLIP_USDA = '''#usda 1.0
(
    endTimeCode = 10
    startTimeCode = 0
    timeCodesPerSecond = 10
)

def SkelAnimation "Nod"
{
    uniform token[] joints = ["Hips/Spine/Head"]
    quatf[] rotations.timeSamples = {
        0: [(1, 0, 0, 0)],
        10: [(0.70710677, 0.70710677, 0, 0)],
    }
}
'''

PROP_USDA = '''#usda 1.0
(
    defaultPrim = "Prop"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Prop"
{
    def Mesh "Geo" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </Prop/Mat>
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
    }

    def Material "Mat"
    {
        token outputs:surface.connect = </Prop/Mat/S.outputs:surface>

        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Prop/Mat/T.outputs:rgb>
            token outputs:surface
        }

        def Shader "T"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./tex/red.png@
            float3 outputs:rgb
        }
    }
}
'''

SCENE_USDA = '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    float3 xformOp:scale = (2, 2, 2)
    uniform token[] xformOpOrder = ["xformOp:scale"]

    def Xform "P0" (
        prepend references = @./parts/prop.usda@
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Xform "P1" (
        prepend references = @./parts/prop.usda@
    )
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
'''


def cube_usda(orientation: str = "rightHanded") -> str:
	"""A closed unit cube (-1..1), Z-up, metres, faces wound for `orientation`."""
	pts = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
	faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]	# CCW from outside
	if orientation == "leftHanded":
		faces = [f[::-1] for f in faces]
	fvi = ", ".join(str(i) for f in faces for i in f)
	p = ", ".join(f"({x}, {y}, {z})" for x, y, z in pts)
	return f'''#usda 1.0
(
    metersPerUnit = 0.0254
    upAxis = "Z"
)

def Mesh "Cube"
{{
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [{fvi}]
    uniform token orientation = "{orientation}"
    point3f[] points = [{p}]
    color3f[] primvars:displayColor = [(1, 0, 0)] (
        interpolation = "constant"
    )
    uniform token subdivisionScheme = "none"
}}
'''


def _png(path: Path, rgb) -> None:
	from PIL import Image

	path.parent.mkdir(parents=True, exist_ok=True)
	rgba = np.zeros((32, 32, 4), np.uint8)
	rgba[...] = (*rgb, 255)
	Image.fromarray(rgba).save(path)


def write_char(directory: Path, clip: bool = False) -> Path:
	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)
	_png(directory / "tex" / "green.png", (30, 200, 60))
	_png(directory / "tex" / "red.png", (200, 30, 60))
	(directory / "char.usda").write_text(CHAR_USDA)
	if clip:
		(directory / "char@nod.usda").write_text(CLIP_USDA)
	return directory / "char.usda"


def write_scene(directory: Path) -> Path:
	directory = Path(directory)
	_png(directory / "parts" / "tex" / "red.png", (200, 30, 60))
	(directory / "parts" / "prop.usda").write_text(PROP_USDA)
	(directory / "scene.usda").write_text(SCENE_USDA)
	return directory / "scene.usda"


def write_usdz(path: Path) -> Path:
	"""A .usdz package: the character layer first, its textures beside it."""
	import io

	from PIL import Image

	path = Path(path)
	with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
		z.writestr("char.usda", CHAR_USDA.replace("@tex/", "@0/"))
		for name, rgb in (("green", (30, 200, 60)), ("red", (200, 30, 60))):
			buf = io.BytesIO()
			rgba = np.zeros((32, 32, 4), np.uint8)
			rgba[...] = (*rgb, 255)
			Image.fromarray(rgba).save(buf, "PNG")
			z.writestr(f"0/{name}.png", buf.getvalue())
	return path
