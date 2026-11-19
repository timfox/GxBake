/* _rt.c - the ray caster's inner loop in C, for gxbake/core/raytrace.py.
 *
 * raytrace.py builds the BVH in numpy and, when a C compiler is around, compiles this file
 * once into a small shared library (ctypes, no Python headers needed) and hands it the
 * arrays. Without a compiler the numpy walker does the same job, slower.
 *
 * Plain C99, no allocation besides a fixed stack, no globals: safe to call from threads.
 */
#include <math.h>
#include <stdint.h>

#define STACK 128

static int box( const double *lo, const double *hi, const double o[3], const double inv[3], double tmax, double *tn_out )
{
	double tn = 0.0, tf = tmax;
	int k;

	for( k = 0; k < 3; k++ )
	{
		double a = ( lo[k] - o[k] ) * inv[k], b = ( hi[k] - o[k] ) * inv[k];

		if( a > b ) { double t = a; a = b; b = t; }
		if( a > tn ) tn = a;
		if( b < tf ) tf = b;
		if( tn > tf ) return 0;
	}
	*tn_out = tn;
	return 1;
}

static double tri( const double *v0, const double *e1, const double *e2, const double o[3], const double d[3] )
{
	double p[3], s[3], q[3], det, inv, u, v, t;

	p[0] = d[1] * e2[2] - d[2] * e2[1];
	p[1] = d[2] * e2[0] - d[0] * e2[2];
	p[2] = d[0] * e2[1] - d[1] * e2[0];
	det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
	if( fabs( det ) <= 1e-12 ) return INFINITY;
	inv = 1.0 / det;
	s[0] = o[0] - v0[0]; s[1] = o[1] - v0[1]; s[2] = o[2] - v0[2];
	u = ( s[0] * p[0] + s[1] * p[1] + s[2] * p[2] ) * inv;
	if( u < -1e-6 || u > 1.0 + 1e-6 ) return INFINITY;
	q[0] = s[1] * e1[2] - s[2] * e1[1];
	q[1] = s[2] * e1[0] - s[0] * e1[2];
	q[2] = s[0] * e1[1] - s[1] * e1[0];
	v = ( d[0] * q[0] + d[1] * q[1] + d[2] * q[2] ) * inv;
	if( v < -1e-6 || u + v > 1.0 + 1e-6 ) return INFINITY;
	t = ( e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2] ) * inv;
	return t > 1e-6 ? t : INFINITY;
}

/* Nearest (or, with any_hit, first found) triangle along each ray within tmax[i].
 * Triangles and nodes are as raytrace.Mesh stores them (triangles in leaf order).
 * out_tri gets the leaf-order triangle index or -1; out_t the distance (tmax on a miss). */
int gx_rt_trace( const double *v0, const double *e1, const double *e2,
	const double *lo, const double *hi, const int64_t *left, const int64_t *right,
	const int64_t *start, const int64_t *count, int64_t nnodes,
	const double *orig, const double *dir, const double *tmax, int64_t nray, int any_hit,
	double *out_t, int64_t *out_tri )
{
	int64_t r;

	if( nnodes <= 0 )
	{
		for( r = 0; r < nray; r++ ) { out_t[r] = tmax[r]; out_tri[r] = -1; }
		return 0;
	}
	for( r = 0; r < nray; r++ )
	{
		const double *o = orig + 3 * r, *d = dir + 3 * r;
		double inv[3], best = tmax[r], tn;
		int64_t hit = -1, stack[STACK];
		int sp = 0, k;

		for( k = 0; k < 3; k++ )
			inv[k] = d[k] != 0.0 ? 1.0 / d[k] : ( d[k] < 0 ? -1e300 : 1e300 );
		if( box( lo, hi, o, inv, best, &tn ))
			stack[sp++] = 0;
		while( sp > 0 )
		{
			int64_t n = stack[--sp];

			if( !box( lo + 3 * n, hi + 3 * n, o, inv, best, &tn ))
				continue;
			if( count[n] > 0 )
			{
				int64_t i, end = start[n] + count[n];

				for( i = start[n]; i < end; i++ )
				{
					double t = tri( v0 + 3 * i, e1 + 3 * i, e2 + 3 * i, o, d );

					if( t < best ) { best = t; hit = i; }
				}
				if( any_hit && hit >= 0 )
					break;
			}
			else if( sp + 2 <= STACK )
			{
				/* nearer child last, so it is popped first */
				int64_t a = left[n], b = right[n];
				double ta, tb;
				int ha = box( lo + 3 * a, hi + 3 * a, o, inv, best, &ta );
				int hb = box( lo + 3 * b, hi + 3 * b, o, inv, best, &tb );

				if( ha && hb )
				{
					if( ta < tb ) { stack[sp++] = b; stack[sp++] = a; }
					else { stack[sp++] = a; stack[sp++] = b; }
				}
				else if( ha ) stack[sp++] = a;
				else if( hb ) stack[sp++] = b;
			}
			else
				return -1;	/* deeper than STACK: the caller falls back to numpy */
		}
		out_t[r] = best;
		out_tri[r] = hit;
	}
	return 0;
}
