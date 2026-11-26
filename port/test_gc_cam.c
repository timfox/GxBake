/* Host harness for gc_cam.c, run by GxBake's tests.
 *   test_gc_cam list FILE                 prints "index name duration near far aspect loop fixed keys_pos keys_rot keys_fov"
 *   test_gc_cam sample FILE NAME T...     prints, per time: origin[3] quat[4] fov_y angles[3] fov_x(4:3)
 *   test_gc_cam mtx FILE NAME T ASPECT    prints the 3x4 view matrix, then the 4x4 projection
 *   test_gc_cam play FILE NAME START T...  plays the clip from clock START, then per clock time T
 *                                         prints "1 t origin[3] angles[3] fov_x(4:3)" or "0" once it has ended
 *   test_gc_cam load FILE NAME T ASPECT   (built with -DGC_CAM_GX against a recording stub) runs gc_cam_load
 * Exit status 1 when the file is rejected or the clip is not found.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gc_cam.h"

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
	gc_cam_clip_t clip;
	gc_cam_pose_t pose;
	int n, i, idx;

	if( argc < 3 || !(in = slurp( argv[2], &len )))
		return 2;
	n = gc_cam_count( in, len );
	if( n < 0 )
		return 1;
	if( !strcmp( argv[1], "list" ))
	{
		for( i = 0; i < n; i++ )
		{
			if( gc_cam_clip( in, len, i, &clip ))
				return 1;
			printf( "%d %s %.6f %.6f %.6f %.6f %d %d %d %d %d\n", i, clip.name, clip.duration, clip.near_z, clip.far_z,
				clip.aspect, clip.loop, clip.fixed, clip.track[0].keys, clip.track[1].keys, clip.track[2].keys );
		}
		return 0;
	}
	if( !strcmp( argv[1], "play" ) && argc >= 5 )
	{
		gc_cam_player_t pl;
		gc_cam_view_t v;

		if( gc_cam_play( &pl, in, len, argv[3], atof( argv[4] )))
			return 1;
		for( i = 5; i < argc; i++ )
		{
			if( !gc_cam_update( &pl, atof( argv[i] ), 4.0f / 3.0f, &v ))
			{
				printf( "0\n" );
				continue;
			}
			printf( "1 %.6f %.6f %.6f %.6f %.5f %.5f %.5f %.6f\n", v.t, v.vieworg[0], v.vieworg[1], v.vieworg[2],
				v.viewangles[0], v.viewangles[1], v.viewangles[2], v.fov_x );
		}
		return 0;
	}
	if( argc < 5 )
		return 2;
	idx = gc_cam_find( in, len, argv[3] );
	if( idx < 0 || gc_cam_clip( in, len, idx, &clip ))
		return 1;
	if( !strcmp( argv[1], "sample" ))
	{
		for( i = 4; i < argc; i++ )
		{
			float a[3];

			gc_cam_sample( &clip, (float)atof( argv[i] ), &pose );
			gc_cam_angles( &pose, a );
			printf( "%.6f %.6f %.6f %.7f %.7f %.7f %.7f %.6f %.5f %.5f %.5f %.6f\n", pose.origin[0], pose.origin[1], pose.origin[2],
				pose.quat[0], pose.quat[1], pose.quat[2], pose.quat[3], pose.fov_y, a[0], a[1], a[2],
				gc_cam_fov_x( pose.fov_y, 4.0f / 3.0f ));
		}
		return 0;
	}
	if( !strcmp( argv[1], "mtx" ) && argc >= 6 )
	{
		float v[3][4], p[4][4];
		int r;

		gc_cam_sample( &clip, (float)atof( argv[4] ), &pose );
		gc_cam_view_mtx( &pose, v );
		gc_cam_proj_mtx( &pose, (float)atof( argv[5] ), p );
		for( r = 0; r < 3; r++ )
			printf( "%.7f %.7f %.7f %.5f\n", v[r][0], v[r][1], v[r][2], v[r][3] );
		for( r = 0; r < 4; r++ )
			printf( "%.7f %.7f %.7f %.7f\n", p[r][0], p[r][1], p[r][2], p[r][3] );
		return 0;
	}
#ifdef GC_CAM_GX
	if( !strcmp( argv[1], "load" ) && argc >= 6 )
	{
		float v[3][4];

		gc_cam_sample( &clip, (float)atof( argv[4] ), &pose );
		gc_cam_load( &pose, (float)atof( argv[5] ), v );
		printf( "view %.5f\n", v[0][3] );
		return 0;
	}
#endif
	return 2;
}
