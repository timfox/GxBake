/* Host harness for gc_lmap.c, run by GxBake's tests.
 *   test_gc_lmap FILE                prints "version flags faces pages scale gamma", one "page w h" per
 *                                    page, then one line per face: "face page x y w h src_w src_h" or "face -"
 *   test_gc_lmap FILE FACE S T       prints the atlas "u v" of luxel coords (S, T) on FACE
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gc_lmap.h"

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
	gc_lmap_t m;
	gc_lmap_face_t f;
	uint32_t i;
	uint16_t w, h;

	if( argc < 2 || !(in = slurp( argv[1], &len )))
		return 2;
	if( gc_lmap_parse( &m, in, len ))
	{
		printf( "bad\n" );
		return 1;
	}
	if( argc == 5 )
	{
		float u, v;

		if( !gc_lmap_face( &m, (uint32_t)atoi( argv[2] ), &f ) || gc_lmap_page_size( &m, f.page, &w, &h ))
			return 1;
		gc_lmap_uv( &f, w, h, (float)atof( argv[3] ), (float)atof( argv[4] ), &u, &v );
		printf( "%.6f %.6f\n", u, v );
		return 0;
	}
	printf( "%d %d %d %d %d %g\n", m.version, m.flags, m.faces, m.pages, m.tev_scale, m.gamma );
	for( i = 0; i < m.pages; i++ )
	{
		if( gc_lmap_page_size( &m, i, &w, &h )) return 1;
		printf( "page %d %d\n", w, h );
	}
	for( i = 0; i < m.faces; i++ )
	{
		if( gc_lmap_face( &m, i, &f ))
			printf( "%u %d %d %d %d %d %d %d\n", (unsigned)i, f.page, f.x, f.y, f.w, f.h, f.src_w, f.src_h );
		else
			printf( "%u -\n", (unsigned)i );
	}
	free( in );
	return 0;
}
