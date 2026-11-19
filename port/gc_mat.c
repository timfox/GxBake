/*
gc_mat.c - TPL textures and GxBake material set-ups. See gc_mat.h.
*/
#include "gc_mat.h"

static uint32_t rd32( const uint8_t *p ) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static uint16_t rd16( const uint8_t *p ) { return (uint16_t)((p[0] << 8) | p[1]); }

/* --- formats ---------------------------------------------------------------- */

/* bits per texel and tile size, by GXTexFmt; 0 bpp = unknown */
static void fmt_info( uint8_t fmt, uint32_t *bpp, uint32_t *tw, uint32_t *th )
{
	*bpp = 0; *tw = 8; *th = 8;
	switch( fmt )
	{
	case 0: case 8: case 14: *bpp = 4; break;			/* I4, CI4, CMPR: 8x8 */
	case 1: case 2: case 9: *bpp = 8; *th = 4; break;		/* I8, IA4, CI8: 8x4 */
	case 3: case 4: case 5: *bpp = 16; *tw = 4; *th = 4; break;	/* IA8, RGB565, RGB5A3: 4x4 */
	case 6: *bpp = 32; *tw = 4; *th = 4; break;		/* RGBA8: 4x4, two planes */
	}
}

uint32_t gc_tex_size( uint8_t fmt, uint32_t width, uint32_t height )
{
	uint32_t bpp, tw, th;

	fmt_info( fmt, &bpp, &tw, &th );
	if( !bpp )
		return 0;
	return ((width + tw - 1) / tw * tw) * ((height + th - 1) / th * th) * bpp / 8;
}

/* --- TPL ---------------------------------------------------------------------- */

#define TPL_MAGIC 0x0020AF30u

int gc_tpl_parse( const uint8_t *blob, size_t len )
{
	uint32_t count, table;

	if( len < 12 || rd32( blob ) != TPL_MAGIC )
		return -1;
	count = rd32( blob + 4 );
	table = rd32( blob + 8 );
	if( count > 1024 || table > len || (size_t)count * 8 > len - table )
		return -1;
	return (int)count;
}

int gc_tpl_image( const uint8_t *blob, size_t len, int i, gc_tpl_image_t *img )
{
	const uint8_t *h;
	uint32_t ih, ph, off, w, ht, total, k;
	int n = gc_tpl_parse( blob, len );

	if( n < 0 || i < 0 || i >= n )
		return -1;
	ih = rd32( blob + rd32( blob + 8 ) + 8 * (uint32_t)i );
	ph = rd32( blob + rd32( blob + 8 ) + 8 * (uint32_t)i + 4 );
	if( ih > len || len - ih < 36 )
		return -1;
	h = blob + ih;
	img->height = rd16( h );
	img->width = rd16( h + 2 );
	img->fmt = (uint8_t)rd32( h + 4 );
	off = rd32( h + 8 );
	img->wrap_s = (uint8_t)rd32( h + 12 );
	img->wrap_t = (uint8_t)rd32( h + 16 );
	img->levels = (uint8_t)(h[34] + 1);	/* max LOD + 1 */
	if( !img->width || !img->height || !gc_tex_size( img->fmt, 1, 1 ))
		return -1;
	total = 0;
	w = img->width;
	ht = img->height;
	for( k = 0; k < img->levels; k++ )
	{
		total += gc_tex_size( img->fmt, w, ht );
		w = w > 1 ? w / 2 : 1;
		ht = ht > 1 ? ht / 2 : 1;
	}
	if( off > len || len - off < total )
		return -1;
	img->texels = blob + off;
	img->texel_bytes = total;
	img->tlut = NULL;
	img->tlut_entries = 0;
	img->tlut_fmt = 1;
	if( img->fmt == 8 || img->fmt == 9 )
	{
		uint32_t po;

		if( !ph || ph > len || len - ph < 12 )
			return -1;
		img->tlut_entries = rd16( blob + ph );
		img->tlut_fmt = (uint8_t)rd32( blob + ph + 4 );
		po = rd32( blob + ph + 8 );
		if( po > len || len - po < (uint32_t)img->tlut_entries * 2 || img->tlut_fmt > 2 )
			return -1;
		img->tlut = blob + po;
	}
	return 0;
}

/* --- decoding --------------------------------------------------------------- */

static uint8_t x3( uint32_t v ) { return (uint8_t)(v * 255 / 7); }
static uint8_t x4( uint32_t v ) { return (uint8_t)((v << 4) | v); }
static uint8_t x5( uint32_t v ) { return (uint8_t)((v << 3) | (v >> 2)); }
static uint8_t x6( uint32_t v ) { return (uint8_t)((v << 2) | (v >> 4)); }

static void word_rgba( uint16_t v, uint8_t fmt, uint8_t *o )	/* fmt: 0 IA8, 1 RGB565, 2 RGB5A3 */
{
	if( fmt == 1 )
	{
		o[0] = x5( v >> 11 & 31 ); o[1] = x6( v >> 5 & 63 ); o[2] = x5( v & 31 ); o[3] = 255;
	}
	else if( fmt == 2 )
	{
		if( v & 0x8000 )
		{
			o[0] = x5( v >> 10 & 31 ); o[1] = x5( v >> 5 & 31 ); o[2] = x5( v & 31 ); o[3] = 255;
		}
		else
		{
			o[0] = x4( v >> 8 & 15 ); o[1] = x4( v >> 4 & 15 ); o[2] = x4( v & 15 ); o[3] = x3( v >> 12 & 7 );
		}
	}
	else
	{
		o[0] = o[1] = o[2] = (uint8_t)(v & 255); o[3] = (uint8_t)(v >> 8);
	}
}

static void cmpr_block( const uint8_t *b, uint8_t *rgba, uint32_t w, uint32_t h, uint32_t x0, uint32_t y0 )
{
	uint8_t pal[4][4];
	uint16_t c0 = rd16( b ), c1 = rd16( b + 2 );
	uint32_t x, y, k;

	word_rgba( c0, 1, pal[0] );
	word_rgba( c1, 1, pal[1] );
	for( k = 0; k < 3; k++ )
	{
		if( c0 > c1 )
		{
			pal[2][k] = (uint8_t)((2 * pal[0][k] + pal[1][k]) / 3);
			pal[3][k] = (uint8_t)((pal[0][k] + 2 * pal[1][k]) / 3);
		}
		else
		{
			pal[2][k] = (uint8_t)((pal[0][k] + pal[1][k]) / 2);
			pal[3][k] = 0;
		}
	}
	pal[2][3] = 255;
	pal[3][3] = c0 > c1 ? 255 : 0;
	for( y = 0; y < 4; y++ )
		for( x = 0; x < 4; x++ )
		{
			uint32_t px = x0 + x, py = y0 + y, idx = (uint32_t)(b[4 + y] >> (6 - 2 * x)) & 3;

			if( px < w && py < h )
				for( k = 0; k < 4; k++ )
					rgba[(py * w + px) * 4 + k] = pal[idx][k];
		}
}

int gc_tex_decode( const gc_tpl_image_t *img, int level, uint8_t *rgba )
{
	const uint8_t *p = img->texels;
	uint32_t w = img->width, h = img->height, bpp, tw, th, tx, ty, x, y, k, pw, ph, n = 0;

	if( level < 0 || level >= img->levels )
		return -1;
	for( k = 0; k < (uint32_t)level; k++ )
	{
		p += gc_tex_size( img->fmt, w, h );
		w = w > 1 ? w / 2 : 1;
		h = h > 1 ? h / 2 : 1;
	}
	fmt_info( img->fmt, &bpp, &tw, &th );
	if( !bpp || ((img->fmt == 8 || img->fmt == 9) && !img->tlut))
		return -1;
	pw = (w + tw - 1) / tw * tw;
	ph = (h + th - 1) / th * th;
	if( img->fmt == 14 )
	{
		for( ty = 0; ty < ph; ty += 8 )
			for( tx = 0; tx < pw; tx += 8 )
				for( k = 0; k < 4; k++, p += 8 )
					cmpr_block( p, rgba, w, h, tx + (k & 1) * 4, ty + (k >> 1) * 4 );
		return 0;
	}
	for( ty = 0; ty < ph; ty += th )
		for( tx = 0; tx < pw; tx += tw )
		{
			if( img->fmt == 6 )	/* RGBA8: 16 AR pairs, then 16 GB pairs */
			{
				for( k = 0; k < 16; k++ )
				{
					x = tx + (k & 3); y = ty + (k >> 2);
					if( x < w && y < h )
					{
						uint8_t *o = rgba + (y * w + x) * 4;
						o[3] = p[2 * k]; o[0] = p[2 * k + 1]; o[1] = p[32 + 2 * k]; o[2] = p[32 + 2 * k + 1];
					}
				}
				p += 64;
				continue;
			}
			for( y = ty; y < ty + th; y++ )
				for( x = tx; x < tx + tw; x++, n++ )
				{
					uint8_t c[4];
					uint32_t v;

					switch( img->fmt )
					{
					case 0: v = (p[n >> 1] >> ((n & 1) ? 0 : 4)) & 15; c[0] = c[1] = c[2] = x4( v ); c[3] = 255; break;
					case 1: c[0] = c[1] = c[2] = p[n]; c[3] = 255; break;
					case 2: c[0] = c[1] = c[2] = x4( p[n] & 15 ); c[3] = x4( p[n] >> 4 ); break;
					case 3: c[0] = c[1] = c[2] = p[2 * n + 1]; c[3] = p[2 * n]; break;
					case 4: word_rgba( rd16( p + 2 * n ), 1, c ); break;
					case 5: word_rgba( rd16( p + 2 * n ), 2, c ); break;
					default:	/* CI4, CI8 */
						v = img->fmt == 8 ? (uint32_t)(p[n >> 1] >> ((n & 1) ? 0 : 4)) & 15 : p[n];
						if( v >= img->tlut_entries )
							v = img->tlut_entries ? img->tlut_entries - 1u : 0;
						word_rgba( img->tlut_entries ? rd16( img->tlut + 2 * v ) : 0, img->tlut_fmt, c );
					}
					if( x < w && y < h )
						for( k = 0; k < 4; k++ )
							rgba[(y * w + x) * 4 + k] = c[k];
				}
		}
	return 0;
}

/* --- .gxm --------------------------------------------------------------------- */

int gc_mat_parse( const uint8_t *blob, size_t len, gc_mat_t *mat )
{
	uint32_t n, k;

	if( len < 12 || blob[0] != 'G' || blob[1] != 'X' || blob[2] != 'M' || blob[3] != '1' )
		return -1;
	n = blob[8];
	if( len < 12 + 4 * (size_t)n || blob[4] > GC_MAT_LIGHTMAP || blob[5] > GC_ALPHA_BLEND )
		return -1;
	mat->preset = blob[4];
	mat->alpha = blob[5];
	mat->alpha_ref = blob[6];
	mat->tev_scale = (uint8_t)(blob[7] == 2 || blob[7] == 4 ? blob[7] : 1);
	mat->flags = rd16( blob + 10 );
	for( k = 0; k < GC_MAT_ROLES; k++ )
		mat->image[k] = GC_MAT_NONE;
	for( k = 0; k < n; k++ )
		if( blob[12 + 4 * k] < GC_MAT_ROLES )
			mat->image[blob[12 + 4 * k]] = blob[13 + 4 * k];
	return 0;
}

void gc_mat_sphere_texmtx( float nrm[3][4], float out[3][4] )
{
	int c;

	for( c = 0; c < 3; c++ )
	{
		out[0][c] = 0.5f * nrm[0][c];
		out[1][c] = -0.5f * nrm[1][c];
		out[2][c] = 0.0f;
	}
	out[0][3] = 0.5f;
	out[1][3] = 0.5f;
	out[2][3] = 1.0f;
}

/* --- GX ------------------------------------------------------------------------- */

#ifdef GC_MAT_GX

int gc_tpl_bind( const gc_tpl_image_t *img, GXTexObj *obj, GXTlutObj *tlut, u32 tlut_name )
{
	u8 mip = img->levels > 1 ? GX_TRUE : GX_FALSE;

	if( img->fmt == GX_TF_CI4 || img->fmt == GX_TF_CI8 )
	{
		if( !img->tlut )
			return -1;
		GX_InitTlutObj( tlut, (void *)img->tlut, img->tlut_fmt, img->tlut_entries );
		GX_LoadTlut( tlut, tlut_name );
		GX_InitTexObjCI( obj, (void *)img->texels, img->width, img->height, img->fmt, img->wrap_s, img->wrap_t, mip, tlut_name );
	}
	else
		GX_InitTexObj( obj, (void *)img->texels, img->width, img->height, img->fmt, img->wrap_s, img->wrap_t, mip );
	GX_InitTexObjLOD( obj, mip ? GX_LIN_MIP_LIN : GX_LINEAR, GX_LINEAR, 0.0f, (f32)(img->levels - 1), 0.0f,
		GX_FALSE, GX_FALSE, GX_ANISO_1 );
	return 0;
}

static u8 tev_scale( u8 s ) { return s == 4 ? GX_CS_SCALE_4 : s == 2 ? GX_CS_SCALE_2 : GX_CS_SCALE_1; }

/* colour = d + (1-c)*a + c*b (op ADD); alpha passes APREV unless set otherwise */
static void stage_c( u8 st, u8 a, u8 b, u8 c, u8 d, u8 scale, u8 reg )
{
	GX_SetTevColorIn( st, a, b, c, d );
	GX_SetTevColorOp( st, GX_TEV_ADD, GX_TB_ZERO, scale, GX_TRUE, reg );
	GX_SetTevAlphaIn( st, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_APREV );
	GX_SetTevAlphaOp( st, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV );
	GX_SetTevDirect( st );
}

void gc_mat_lightmap_stage( u8 stage, u8 texcoord, u32 texmap, u8 scale )
{
	GX_SetTevOrder( stage, texcoord, texmap, GX_COLORNULL );
	stage_c( stage, GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO, tev_scale( scale ), GX_TEVPREV );
}

int gc_mat_setup( const gc_mat_t *mat, const gc_mat_bind_t *bind )
{
	u32 map[GC_MAT_ROLES];
	u8 st = 0, gens = 1, r, next = 0, spec_reg = 0;
	int spec = 0;

	for( r = 0; r < GC_MAT_ROLES; r++ )
	{
		map[r] = GX_TEXMAP_NULL;
		if( bind->tex[r] && (mat->image[r] != GC_MAT_NONE || (r == GC_ROLE_BASE && mat->preset == GC_MAT_LIGHTMAP)))
		{
			map[r] = GX_TEXMAP0 + next++;
			GX_LoadTexObj( bind->tex[r], (u8)map[r] );
		}
		else if( mat->image[r] != GC_MAT_NONE )
			return -1;
	}
	if( map[GC_ROLE_BASE] == GX_TEXMAP_NULL )
		return -1;
	spec = map[GC_ROLE_SPEC] != GX_TEXMAP_NULL && mat->preset != GC_MAT_PRELIT && mat->preset != GC_MAT_LIGHTMAP;

	GX_SetTexCoordGen( GX_TEXCOORD0, GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY );
	GX_SetNumIndStages( 0 );

	/* lighting channels: COLOR0A0 diffuse, COLOR1A1 specular */
	if( mat->preset == GC_MAT_LIGHTMAP || (mat->preset == GC_MAT_PRELIT && bind->unlit))
		GX_SetNumChans( 0 );
	else
	{
		GX_SetChanCtrl( GX_COLOR0A0, GX_ENABLE, GX_SRC_REG, GX_SRC_REG, bind->light_mask, GX_DF_CLAMP, GX_AF_NONE );
		if( spec )
			GX_SetChanCtrl( GX_COLOR1A1, GX_ENABLE, GX_SRC_REG, GX_SRC_REG, bind->light_mask, GX_DF_NONE, GX_AF_SPEC );
		GX_SetNumChans( spec ? 2 : 1 );
	}

	if( mat->preset == GC_MAT_LIGHTMAP )
	{
		if( map[GC_ROLE_LIGHTMAP] == GX_TEXMAP_NULL )
			return -1;
		GX_SetTexCoordGen( GX_TEXCOORD1, GX_TG_MTX2x4, GX_TG_TEX1, GX_IDENTITY );
		gens = 2;
		GX_SetTevOrder( st, GX_TEXCOORD0, map[GC_ROLE_BASE], GX_COLORNULL );
		stage_c( st, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, GX_CS_SCALE_1, GX_TEVPREV );
		GX_SetTevAlphaIn( st, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_TEXA );
		st++;
		gc_mat_lightmap_stage( st++, GX_TEXCOORD1, map[GC_ROLE_LIGHTMAP], mat->tev_scale );
	}
	else
	{
		/* emboss first: C0 = (height@uv - height@uv shifted towards the light) + 0.5, x2 */
		if( mat->preset == GC_MAT_EMBOSS && map[GC_ROLE_HEIGHT] != GX_TEXMAP_NULL )
		{
			GX_SetTexCoordGen( GX_TEXCOORD1, GX_TG_BUMP0 + (bind->bump_light & 7), GX_TG_TEXCOORD0, GX_IDENTITY );
			gens = 2;
			GX_SetTevOrder( st, GX_TEXCOORD0, map[GC_ROLE_HEIGHT], GX_COLORNULL );
			stage_c( st, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, GX_CS_SCALE_1, GX_TEVREG0 );
			st++;
			GX_SetTevOrder( st, GX_TEXCOORD1, map[GC_ROLE_HEIGHT], GX_COLORNULL );
			GX_SetTevColorIn( st, GX_CC_TEXC, GX_CC_ZERO, GX_CC_ZERO, GX_CC_C0 );
			GX_SetTevColorOp( st, GX_TEV_SUB, GX_TB_ADDHALF, GX_CS_SCALE_1, GX_TRUE, GX_TEVREG0 );
			GX_SetTevAlphaIn( st, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_APREV );
			GX_SetTevAlphaOp( st, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV );
			GX_SetTevDirect( st );
			st++;
		}
		/* base x diffuse light (or the texture alone when unlit) */
		GX_SetTevOrder( st, GX_TEXCOORD0, map[GC_ROLE_BASE], bind->unlit && mat->preset == GC_MAT_PRELIT ? GX_COLORNULL : GX_COLOR0A0 );
		if( bind->unlit && mat->preset == GC_MAT_PRELIT )
		{
			stage_c( st, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, GX_CS_SCALE_1, GX_TEVPREV );
			GX_SetTevAlphaIn( st, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_TEXA );
		}
		else
		{
			stage_c( st, GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO, GX_CS_SCALE_1, GX_TEVPREV );
			GX_SetTevAlphaIn( st, GX_CA_ZERO, GX_CA_TEXA, GX_CA_RASA, GX_CA_ZERO );
		}
		st++;
		if( mat->preset == GC_MAT_EMBOSS && map[GC_ROLE_HEIGHT] != GX_TEXMAP_NULL )
		{
			GX_SetTevOrder( st, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL );
			stage_c( st, GX_CC_ZERO, GX_CC_CPREV, GX_CC_C0, GX_CC_ZERO, GX_CS_SCALE_2, GX_TEVPREV );	/* 0.5 = flat */
			st++;
		}
		if( spec )	/* CPREV + mask x specular light; the mask also goes to C1 for the reflection */
		{
			GX_SetTevOrder( st, GX_TEXCOORD0, map[GC_ROLE_SPEC], GX_COLOR1A1 );
			stage_c( st, GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_CPREV, GX_CS_SCALE_1, GX_TEVPREV );
			st++;
			if( map[GC_ROLE_ENV] != GX_TEXMAP_NULL )
			{
				GX_SetTevOrder( st, GX_TEXCOORD0, map[GC_ROLE_SPEC], GX_COLORNULL );
				stage_c( st, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, GX_CS_SCALE_1, GX_TEVREG1 );
				st++;
				spec_reg = 1;
			}
		}
		if( map[GC_ROLE_ENV] != GX_TEXMAP_NULL )	/* sphere map from the view-space normal, bent by the normal map */
		{
			u8 tc = (u8)(GX_TEXCOORD0 + gens);

			GX_SetTexCoordGen( tc, GX_TG_MTX2x4, GX_TG_NRM, GX_TEXMTX0 );
			gens++;
			GX_SetTevOrder( st, tc, map[GC_ROLE_ENV], GX_COLORNULL );
			if( spec_reg )
				stage_c( st, GX_CC_ZERO, GX_CC_TEXC, GX_CC_C1, GX_CC_CPREV, GX_CS_SCALE_1, GX_TEVPREV );
			else
			{
				GX_SetTevKColorSel( st, GX_TEV_KCSEL_1_4 );
				stage_c( st, GX_CC_ZERO, GX_CC_TEXC, GX_CC_KONST, GX_CC_CPREV, GX_CS_SCALE_1, GX_TEVPREV );
			}
			if( mat->preset == GC_MAT_INDIRECT && map[GC_ROLE_NORMAL] != GX_TEXMAP_NULL )
			{
				/* IA8 normal: X in I, Y in A. Indirect lookups put A in s and the colour (B) in t,
				 * so the matrix swaps them back: s' = k X, t' = -k Y (t grows downwards). */
				f32 m[2][3];
				f32 k = bind->indirect_strength > 0 ? bind->indirect_strength : 0.5f;
				s8 e = 0;

				while( k >= 1.0f && e < 10 ) { k *= 0.5f; e++; }
				m[0][0] = 0.0f; m[0][1] = k;    m[0][2] = 0.0f;
				m[1][0] = -k;   m[1][1] = 0.0f; m[1][2] = 0.0f;
				GX_SetNumIndStages( 1 );
				GX_SetIndTexOrder( GX_INDTEXSTAGE0, GX_TEXCOORD0, (u8)map[GC_ROLE_NORMAL] );
				GX_SetIndTexCoordScale( GX_INDTEXSTAGE0, GX_ITS_1, GX_ITS_1 );
				GX_SetIndTexMatrix( GX_ITM_0, m, e );
				GX_SetTevIndirect( st, GX_INDTEXSTAGE0, GX_ITF_8, GX_ITB_ST, GX_ITM_0, GX_ITW_OFF, GX_ITW_OFF,
					GX_FALSE, GX_FALSE, GX_ITBA_OFF );
			}
			st++;
		}
		if( map[GC_ROLE_EMIT] != GX_TEXMAP_NULL )	/* glow ignores lighting */
		{
			GX_SetTevOrder( st, GX_TEXCOORD0, map[GC_ROLE_EMIT], GX_COLORNULL );
			stage_c( st, GX_CC_TEXC, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV, GX_CS_SCALE_1, GX_TEVPREV );
			st++;
		}
	}
	GX_SetNumTexGens( gens );
	GX_SetNumTevStages( st );

	if( mat->alpha == GC_ALPHA_MASK )
	{
		GX_SetAlphaCompare( GX_GEQUAL, mat->alpha_ref, GX_AOP_AND, GX_ALWAYS, 0 );
		GX_SetZCompLoc( GX_FALSE );	/* z after texturing, so cut-out texels do not write depth */
		GX_SetBlendMode( GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR );
	}
	else
	{
		GX_SetAlphaCompare( GX_ALWAYS, 0, GX_AOP_AND, GX_ALWAYS, 0 );
		GX_SetZCompLoc( GX_TRUE );
		if( mat->alpha == GC_ALPHA_BLEND )
			GX_SetBlendMode( GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_CLEAR );
		else
			GX_SetBlendMode( GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR );
	}
	return st;
}

#endif
