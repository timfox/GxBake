/*
gc_lmap.h - GxBake radiosity lightmap atlases (.gclm + TPL pages) for the Xash3D GC port,
or any libogc program.

`gxbake gi map.bsp` bakes a map's lightmaps (bounce light, sky, texture lights, AO) and,
next to the relit BSP, writes them as GX-native pages:

  <map>_lm.tpl    page i is TPL image i (CMPR, RGB565, I4 or I8, picked per page)
  <map>.gclm      where each BSP face's lightmap sits in those pages (this file reads it)
  <map>_lm.gxm    the TEV scale (read with gc_mat_parse; the .gclm repeats it)

Pages hold value / scale, and the lightmap TEV stage multiplies by scale
(gc_mat_lightmap_stage, GX_CS_SCALE_1/2/4). With the default "port" tone the value is
what the port shows today for raw BSP lightmaps (byte / 255 x 2), so scale 2 gives the
same picture as the existing x2 stage, and scale 4 keeps light that the BSP had to clip.

A face's rectangle is padded by a luxel on each side (bilinear filtering never reads a
neighbour) and starts on a 4x4 block, so a CMPR block never mixes two faces. Faces may
be stored smaller than the engine's grid (flat ones as 1x1, smooth ones at half size);
gc_lmap_uv maps the engine's lightmap coordinates onto whatever was stored.

Plain C89 plus <stdint.h>, no allocation, byte-wise big-endian reads, bounds checked.
Host-tested by GxBake (tests/test_gi.py) against its Python reference; not yet run in
the port, on hardware or in Dolphin.

.gclm layout (big-endian):
  header (32 B): "GCLM", u16 version (1), u16 flags (1 = port tone), u16 faces, u16 pages,
                 u8 tev scale, u8 pad, u16 gamma x 256, u32 page table offset,
                 u32 face table offset, 8 bytes reserved
  face table: one 12-byte entry per BSP face, in face order:
                 u16 page (0xFFFF = no lightmap), u16 x, u16 y, u16 w, u16 h,
                 u8 engine grid w, u8 engine grid h      (x, y: first stored luxel)
  page table: pages x { u16 width, u16 height }
*/
#ifndef GC_LMAP_H
#define GC_LMAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GC_LMAP_NONE 0xFFFFu
#define GC_LMAP_FLAG_PORT_TONE 1u

typedef struct
{
	const uint8_t *data;
	size_t size;
	uint16_t version, flags, faces, pages;
	uint8_t tev_scale;	/* 1, 2 or 4: GX_CS_SCALE_1 / _2 / _4 on the lightmap stage */
	float gamma;	/* pages hold (value / scale)^(1/gamma); 1 = linear */
	uint32_t page_ofs, face_ofs;
} gc_lmap_t;

typedef struct
{
	uint16_t page, x, y, w, h;	/* stored rectangle (without its padding), in texels */
	uint8_t src_w, src_h;	/* the engine's grid: extents / 16 + 1 */
} gc_lmap_face_t;

/* 0 on success, -1 if the blob is not a .gclm or its tables run past the end. */
int gc_lmap_parse( gc_lmap_t *m, const void *data, size_t size );

/* 1 and *out filled when face has a lightmap in the atlas, 0 when it has none (sky, water,
   special) or face is out of range. */
int gc_lmap_face( const gc_lmap_t *m, uint32_t face, gc_lmap_face_t *out );

/* 0 and the page's size, or -1 when page is out of range. */
int gc_lmap_page_size( const gc_lmap_t *m, uint32_t page, uint16_t *w, uint16_t *h );

/* Atlas UV of a point on the face. s and t are the engine's lightmap coordinates in luxels
   from the face's texture mins, as Xash computes them per vertex:
     s = ( DotProduct( v, info->lmvecs[0] ) + info->lmvecs[0][3] - info->lightmapmins[0] ) / 16
   (0 .. src_w - 1 across the face). page_w / page_h from gc_lmap_page_size. */
void gc_lmap_uv( const gc_lmap_face_t *f, uint16_t page_w, uint16_t page_h, float s, float t, float *u, float *v );

#ifdef __cplusplus
}
#endif

#endif
