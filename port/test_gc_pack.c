/* Host harness for gc_pack.c: GxBake's tests run it and compare with Python.
 *   test_gc_pack yaz0 IN OUT     decompress
 *   test_gc_pack dsp IN OUT      decode to raw s16 little-endian
 *   test_gc_pack gsk IN          print one line per skin: skin fmt tlutfmt w h entries flags texbytes tlutbytes
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gc_pack.h"

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

	if( argc < 3 || !(in = slurp( argv[2], &len )))
		return 2;
	if( !strcmp( argv[1], "yaz0" ) && argc == 4 )
	{
		uint32_t size = gc_yaz0_size( in, len );
		uint8_t *out = (uint8_t *)malloc( size ? size : 1 );
		FILE *f;
		if( gc_yaz0_decode( in, len, out, size ) != size ) return 1;
		f = fopen( argv[3], "wb" ); fwrite( out, 1, size, f ); fclose( f );
		return 0;
	}
	if( !strcmp( argv[1], "dsp" ) && argc == 4 )
	{
		gc_dsp_t dsp;
		gc_dsp_state_t st = { 0, 0 };
		int16_t *out;
		uint32_t i, half;
		FILE *f;
		if( gc_dsp_parse( in, len, &dsp )) return 1;
		out = (int16_t *)malloc( (dsp.samples + 1) * sizeof( int16_t ));
		/* two calls, split mid-frame, to exercise streaming */
		half = dsp.samples / 2;
		if( gc_dsp_decode( &dsp, 0, half, &st, out ) != half ) return 1;
		if( gc_dsp_decode( &dsp, half, dsp.samples - half, &st, out + half ) != dsp.samples - half ) return 1;
		f = fopen( argv[3], "wb" );
		for( i = 0; i < dsp.samples; i++ ) { uint8_t b[2]; b[0] = (uint8_t)(out[i] & 255); b[1] = (uint8_t)((uint16_t)out[i] >> 8); fwrite( b, 1, 2, f ); }
		fclose( f );
		printf( "%u %u %u %u %u %u\n", dsp.samples, dsp.rate, dsp.loop, dsp.loop_start, dsp.loop_end, dsp.loop_ps );
		return 0;
	}
	if( !strcmp( argv[1], "gsk" ))
	{
		int n = gc_gsk_parse( in, len ), i;
		if( n < 0 ) return 1;
		for( i = 0; i < n; i++ )
		{
			gc_gsk_entry_t e;
			if( gc_gsk_entry( in, len, i, &e )) return 1;
			printf( "%u %u %u %u %u %u %u %u %u %lu\n", e.skin, e.fmt, e.tlut_fmt, e.width, e.height, e.tlut_entries,
				e.flags, e.texel_bytes, e.tlut_bytes, (unsigned long)(e.texels - in) );
		}
		return 0;
	}
	return 2;
}
