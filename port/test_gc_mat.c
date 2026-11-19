/* Host harness for gc_mat.c, run by GxBake's tests.
 *   test_gc_mat tpl FILE OUT     prints "fmt w h levels wrap_s wrap_t tlut_entries" per image and writes
 *                                every level of every image decoded to RGBA8, back to back, to OUT
 *   test_gc_mat gxm FILE         prints "preset alpha ref scale flags" and the image index of each role
 *   test_gc_mat sphere           prints gc_mat_sphere_texmtx of a rotation
 *   test_gc_mat setup FILE [unlit]  (built with -DGC_MAT_GX against a recording stub) runs gc_mat_setup
 *                                with a texture object (id = role + 1) for every role the .gxm has, plus a
 *                                base for lightmaps, and prints the GX calls and "stages N"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gc_mat.h"

static uint8_t *slurp( const char *path, size_t *len )
{
	FILE *f = fopen( path, "rb" );
	uint8_t *buf;
	long n;

	if( !f ) return NULL;
	fseek( f, 0, SEEK_END ); n = ftell( f ); fseek( f, 0, SEEK_SET );
	buf = (uint8_t *)malloc( n ? (size_t)n : 1 );
	if( buf && fread( buf, 1, (size_t)n, f ) != (size_t)n ) { free( buf ); buf = NULL; }
	fclose( f );
	*len = (size_t)n;
	return buf;
}

int main( int argc, char **argv )
{
	size_t len;
	uint8_t *in;

	if( argc >= 2 && !strcmp( argv[1], "sphere" ))
	{
		float n[3][4] = { { 0, 1, 0, 5 }, { -1, 0, 0, 6 }, { 0, 0, 1, 7 } }, o[3][4];
		int r, c;

		gc_mat_sphere_texmtx( n, o );
		for( r = 0; r < 3; r++ )
			for( c = 0; c < 4; c++ )
				printf( "%g%c", o[r][c], c == 3 ? '\n' : ' ' );
		return 0;
	}
	if( argc < 3 || !(in = slurp( argv[2], &len )))
		return 2;
	if( !strcmp( argv[1], "tpl" ) && argc == 4 )
	{
		FILE *out = fopen( argv[3], "wb" );
		int n = gc_tpl_parse( in, len ), i, l;

		if( n < 0 || !out ) return 1;
		for( i = 0; i < n; i++ )
		{
			gc_tpl_image_t img;
			uint32_t w, h;

			if( gc_tpl_image( in, len, i, &img )) return 1;
			printf( "%d %d %d %d %d %d %d\n", img.fmt, img.width, img.height, img.levels, img.wrap_s, img.wrap_t, img.tlut_entries );
			w = img.width; h = img.height;
			for( l = 0; l < img.levels; l++ )
			{
				uint8_t *rgba = (uint8_t *)malloc( w * h * 4 );

				if( !rgba || gc_tex_decode( &img, l, rgba )) return 1;
				fwrite( rgba, 1, w * h * 4, out );
				free( rgba );
				w = w > 1 ? w / 2 : 1; h = h > 1 ? h / 2 : 1;
			}
		}
		fclose( out );
		return 0;
	}
	if( !strcmp( argv[1], "gxm" ))
	{
		gc_mat_t m;
		int r;

		if( gc_mat_parse( in, len, &m )) return 1;
		printf( "%d %d %d %d %d\n", m.preset, m.alpha, m.alpha_ref, m.tev_scale, m.flags );
		for( r = 0; r < GC_MAT_ROLES; r++ )
			printf( "%d%c", m.image[r] == GC_MAT_NONE ? -1 : m.image[r], r == GC_MAT_ROLES - 1 ? '\n' : ' ' );
		return 0;
	}
#ifdef GC_MAT_GX
	if( !strcmp( argv[1], "setup" ))
	{
		gc_mat_t m;
		gc_mat_bind_t b;
		GXTexObj objs[GC_MAT_ROLES];
		int r;

		if( gc_mat_parse( in, len, &m )) return 1;
		memset( &b, 0, sizeof( b ));
		for( r = 0; r < GC_MAT_ROLES; r++ )
		{
			objs[r].id = (u32)r + 1;
			if( m.image[r] != GC_MAT_NONE || (r == GC_ROLE_BASE && m.preset == GC_MAT_LIGHTMAP))
				b.tex[r] = &objs[r];
		}
		b.light_mask = 1;
		b.unlit = argc > 3 && !strcmp( argv[3], "unlit" );
		printf( "stages %d\n", gc_mat_setup( &m, &b ));
		return 0;
	}
#endif
	return 2;
}
