/*
gc_mat.h - TPL textures and GxBake material set-ups (.gxm) for the Xash3D GC port,
or any libogc program.

`gxbake auto`, `gxbake pbr` and `gxbake hdr` write textures as TPL (the Dolphin
SDK's texture library: any GX format, palettes, mip chains). `gxbake pbr` and
`gxbake hdr --mode lightmap` also write a .gxm: a few bytes saying which preset
the textures are for and how to blend them. This file reads both and, behind
GC_MAT_GX, sets up GX to draw them:

  gc_tpl_parse, gc_tpl_image    images of a .tpl (format, size, levels, TLUT)
  gc_tex_size                   bytes of one level in a GX format
  gc_tex_decode                 any GX format -> RGBA8 on the CPU (tools, tests)
  gc_mat_parse                  a .gxm -> gc_mat_t
  gc_mat_sphere_texmtx          the sphere-map texture matrix for GX_TG_NRM texgen
  gc_tpl_bind        (GX)       GXTexObj (+ TLUT) for a TPL image, mips included
  gc_mat_setup       (GX)       texgens, channels, TEV stages, alpha and blend for a preset
  gc_mat_lightmap_stage (GX)    one TEV stage: CPREV x lightmap x 2 or 4 (overbright)

Plain C89 plus <stdint.h>, no allocation, byte-wise big-endian reads, bounds
checked. GxBake's tests compile it on the host (-Werror): the readers and the
decoder against GxBake's own encoder, and the GX set-up against a recording
stub of <ogc/gx.h>, so the call sequence is checked but it has not been run on
hardware or in Dolphin.

.gxm layout (big-endian, 12 + 4 x maps bytes):
  "GXM1", u8 preset, u8 alpha mode, u8 alpha ref, u8 tev scale,
  u8 map count, u8 pad, u16 flags,
  maps x { u8 role, u8 tpl image index, u16 pad }
*/
#ifndef GC_MAT_H
#define GC_MAT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- TPL ------------------------------------------------------------------ */

typedef struct
{
	uint8_t  fmt;		/* GXTexFmt: I4 0, I8 1, IA4 2, IA8 3, RGB565 4, RGB5A3 5, RGBA8 6, CI4 8, CI8 9, CMPR 14 */
	uint16_t width, height;
	uint8_t  levels;	/* 1 = no mips */
	uint8_t  wrap_s, wrap_t;	/* GX_CLAMP 0, GX_REPEAT 1, GX_MIRROR 2 */
	const uint8_t *texels;	/* every level back to back; 32-byte aligned if the file is */
	uint32_t texel_bytes;
	const uint8_t *tlut;	/* NULL unless CI4/CI8 */
	uint16_t tlut_entries;
	uint8_t  tlut_fmt;	/* GX_TL_IA8 0, GX_TL_RGB565 1, GX_TL_RGB5A3 2 */
} gc_tpl_image_t;

/* Images in a .tpl, or -1 if it is not one. */
int gc_tpl_parse( const uint8_t *blob, size_t len );
/* Fill *img for image i; 0 on success, -1 if out of range, truncated or an unknown format. */
int gc_tpl_image( const uint8_t *blob, size_t len, int i, gc_tpl_image_t *img );
/* Bytes of one level (sides padded to whole tiles), 0 for an unknown format. */
uint32_t gc_tex_size( uint8_t fmt, uint32_t width, uint32_t height );
/* Decode one level to RGBA8 (width x height x 4). 0 on success. */
int gc_tex_decode( const gc_tpl_image_t *img, int level, uint8_t *rgba );

/* --- .gxm materials ------------------------------------------------------ */

enum { GC_MAT_PRELIT, GC_MAT_LIT, GC_MAT_EMBOSS, GC_MAT_INDIRECT, GC_MAT_LIGHTMAP };
enum { GC_ALPHA_OPAQUE, GC_ALPHA_MASK, GC_ALPHA_BLEND };
enum { GC_ROLE_BASE, GC_ROLE_SPEC, GC_ROLE_EMIT, GC_ROLE_HEIGHT, GC_ROLE_NORMAL, GC_ROLE_ENV, GC_ROLE_LIGHTMAP,
	GC_MAT_ROLES };
#define GC_MAT_NONE 0xFF

typedef struct
{
	uint8_t  preset;	/* GC_MAT_* */
	uint8_t  alpha;		/* GC_ALPHA_* */
	uint8_t  alpha_ref;	/* GC_ALPHA_MASK: alpha >= ref passes */
	uint8_t  tev_scale;	/* 1, 2 or 4: lightmaps stored at light / scale */
	uint16_t flags;
	uint8_t  image[GC_MAT_ROLES];	/* role -> TPL image index, GC_MAT_NONE when absent */
} gc_mat_t;

int gc_mat_parse( const uint8_t *blob, size_t len, gc_mat_t *mat );

/* Sphere-map texture matrix: s = 0.5 nx + 0.5, t = -0.5 ny + 0.5 of the
 * view-space normal. Pass the normal matrix the vertices are drawn with
 * (3x4, as GX_LoadNrmMtxImm takes; the translation column is ignored);
 * load the result with GX_LoadTexMtxImm( out, GX_TEXMTX0, GX_MTX2x4 ). */
void gc_mat_sphere_texmtx( float nrm[3][4], float out[3][4] );

#ifdef GC_MAT_GX
#include <ogc/gx.h>

/* Texture object for a TPL image with its mip chain and wrap. CI images
 * also load their TLUT into tlut_name. 0 on success. Call DCFlushRange on
 * the TPL once after loading it (32-byte aligned). */
int gc_tpl_bind( const gc_tpl_image_t *img, GXTexObj *obj, GXTlutObj *tlut, u32 tlut_name );

typedef struct
{
	GXTexObj *tex[GC_MAT_ROLES];	/* bound objects by role; NULL where the material has none.
					 * GC_MAT_LIGHTMAP takes the surface's own texture as tex[GC_ROLE_BASE]. */
	u8  light_mask;			/* GX_LIGHT0 | ...: lights for the colour channels */
	u8  bump_light;			/* emboss: the light GX_TG_BUMPn shifts towards (0..7) */
	f32 indirect_strength;		/* indirect: how far normals bend the reflection (0.5 is a start) */
	u8  unlit;			/* prelit: 1 = ignore vertex lighting (texture as is) */
} gc_mat_bind_t;

/* Program GX for the material: texture maps from GX_TEXMAP0 in role order,
 * texcoord 0 = the mesh UVs (GX_TG_TEX0), texcoord 1 = bump (emboss),
 * lightmap UVs (GX_TG_TEX1) or the sphere map (GX_TG_NRM through GX_TEXMTX0,
 * see gc_mat_sphere_texmtx), channels, TEV stages, alpha compare and blend.
 * Returns the number of TEV stages used, or -1 if a needed texture is missing. */
int gc_mat_setup( const gc_mat_t *mat, const gc_mat_bind_t *bind );

/* One TEV stage: CPREV x TEXC(texmap @ texcoord), scaled by tev_scale (1, 2, 4). */
void gc_mat_lightmap_stage( u8 stage, u8 texcoord, u32 texmap, u8 tev_scale );
#endif

#ifdef __cplusplus
}
#endif
#endif
