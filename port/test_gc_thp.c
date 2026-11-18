/* Host harness for gc_thp.c: GxBake's tests run it and compare with FFmpeg and Python.
 *   test_gc_thp FILE YUV PCM TEX
 * prints "width height frames fps*1000 channels rate samples", then writes every
 * picture as yuv420p planes to YUV, every audio block as interleaved s16 (host
 * order) to PCM, and frame 0 as a tiled RGB565 texture followed by its Y plane
 * as a tiled I8 texture to TEX. Exit 1 on any decode error.
 */
#include <stdio.h>
#include <stdlib.h>
#include "gc_thp.h"

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
	size_t len, pos, size, planes;
	uint8_t *in, *yuv;
	int16_t *pcm;
	uint16_t *rgb;
	uint8_t *i8;
	gc_thp_t thp;
	uint32_t k;
	FILE *fy, *fp, *ft;

	if( argc != 5 || !(in = slurp( argv[1], &len )))
		return 2;
	if( gc_thp_parse( in, len, &thp ))
		return 1;
	printf( "%u %u %u %d %u %u %u\n", (unsigned)thp.width, (unsigned)thp.height, (unsigned)thp.frames,
		(int)(thp.fps * 1000 + 0.5f), (unsigned)thp.audio_channels, (unsigned)thp.audio_rate, (unsigned)thp.audio_samples );
	planes = thp.width * thp.height;
	yuv = (uint8_t *)malloc( planes * 3 / 2 );
	pcm = (int16_t *)malloc( (thp.max_audio_samples + 1) * 4 );
	rgb = (uint16_t *)malloc( planes * 2 );
	i8 = (uint8_t *)malloc( planes );
	fy = fopen( argv[2], "wb" ); fp = fopen( argv[3], "wb" ); ft = fopen( argv[4], "wb" );
	pos = thp.first_frame;
	size = thp.first_frame_bytes;
	for( k = 0; k < thp.frames; k++ )
	{
		gc_thp_frame_t fr;

		if( pos > len || size > len - pos || gc_thp_frame( &thp, in + pos, size, &fr ))
			return 1;
		if( gc_thp_jpeg( fr.video, fr.video_bytes, (int)thp.width, (int)thp.height,
			yuv, yuv + planes, yuv + planes + planes / 4 ))
			return 1;
		fwrite( yuv, 1, planes * 3 / 2, fy );
		if( fr.audio )
		{
			uint32_t n = gc_thp_audio( fr.audio, fr.audio_bytes, pcm, thp.max_audio_samples );
			if( !n ) return 1;
			fwrite( pcm, 4, n, fp );
		}
		if( k == 0 )
		{
			gc_thp_rgb565_tiled( yuv, yuv + planes, yuv + planes + planes / 4, (int)thp.width, (int)thp.height, rgb );
			gc_thp_i8_tiled( yuv, (int)thp.width, (int)thp.height, i8 );
			fwrite( rgb, 2, planes, ft );
			fwrite( i8, 1, planes, ft );
		}
		pos += size;
		size = fr.next_bytes;
	}
	fclose( fy ); fclose( fp ); fclose( ft );
	return 0;
}
