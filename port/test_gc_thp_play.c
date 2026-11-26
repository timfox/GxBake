/* Host harness for gc_thp.c's player: plays a movie against a simulated clock.
 *   test_gc_thp_play FILE LOOP TAKES COST PCM
 * LOOP 0/1; stops after TAKES frames (0 = until the movie ends); each picture shown costs
 * COST clock ticks of decode time, during which the clock runs on. Prints the clock rate,
 * then one line per frame: "S|K index pass due until now" (shown or skipped, and the
 * clock when it was decided). Every audio block is queued as it is read: interleaved
 * s16 (host order) to PCM. A WAIT jumps the clock to the cue's due time. Exit 1 on error.
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
	size_t len;
	uint8_t *in;
	int16_t *pcm;
	gc_thp_t thp;
	gc_thp_player_t pl;
	uint32_t off, bytes, now = 0, cost, takes, taken = 0;
	FILE *fp;

	if( argc != 6 || !(in = slurp( argv[1], &len )))
		return 2;
	takes = (uint32_t)atol( argv[3] );
	cost = (uint32_t)atol( argv[4] );
	if( gc_thp_parse( in, len, &thp ) || gc_thp_player_init( &pl, &thp, atoi( argv[2] )))
		return 1;
	printf( "%u\n", (unsigned)gc_thp_player_rate( &pl ));
	pcm = (int16_t *)malloc( (thp.max_audio_samples + 1) * 4 );
	fp = fopen( argv[5], "wb" );
	while( (!takes || taken < takes) && gc_thp_player_next( &pl, &off, &bytes ))
	{
		gc_thp_cue_t cue;
		int d;

		if( off > len || bytes > len - off || gc_thp_player_take( &pl, in + off, bytes, &cue ))
			return 1;
		taken++;
		if( cue.fr.audio )
		{
			uint32_t n = gc_thp_audio( cue.fr.audio, cue.fr.audio_bytes, pcm, thp.max_audio_samples );
			if( n != cue.samples ) return 1;
			fwrite( pcm, 4, n, fp );
		}
		while( (d = gc_thp_player_decide( &cue, now )) == GC_THP_WAIT )
			now = cue.due;
		printf( "%c %u %u %u %u %u\n", d == GC_THP_SHOW ? 'S' : 'K', (unsigned)cue.index, (unsigned)cue.pass,
			(unsigned)cue.due, (unsigned)cue.until, (unsigned)now );
		if( d == GC_THP_SHOW )
			now += cost;
	}
	fclose( fp );
	free( pcm );
	free( in );
	return 0;
}
