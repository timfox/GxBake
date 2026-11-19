/* gc_lmap.c - see gc_lmap.h */
#include "gc_lmap.h"

static uint32_t rd32( const uint8_t *p ) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static uint16_t rd16( const uint8_t *p ) { return (uint16_t)((p[0] << 8) | p[1]); }

int gc_lmap_parse( gc_lmap_t *m, const void *data, size_t size )
{
	const uint8_t *b = (const uint8_t *)data;

	if( !m || !b || size < 32 || b[0] != 'G' || b[1] != 'C' || b[2] != 'L' || b[3] != 'M' )
		return -1;
	m->data = b;
	m->size = size;
	m->version = rd16( b + 4 );
	m->flags = rd16( b + 6 );
	m->faces = rd16( b + 8 );
	m->pages = rd16( b + 10 );
	m->tev_scale = b[12];
	m->gamma = (float)rd16( b + 14 ) / 256.0f;
	m->page_ofs = rd32( b + 16 );
	m->face_ofs = rd32( b + 20 );
	if( m->version != 1 || m->gamma <= 0.0f )
		return -1;
	if( m->tev_scale != 1 && m->tev_scale != 2 && m->tev_scale != 4 )
		return -1;
	if( m->face_ofs > size || (size - m->face_ofs) / 12u < m->faces )
		return -1;
	if( m->page_ofs > size || (size - m->page_ofs) / 4u < m->pages )
		return -1;
	return 0;
}

int gc_lmap_face( const gc_lmap_t *m, uint32_t face, gc_lmap_face_t *out )
{
	const uint8_t *e;

	if( !m || face >= m->faces )
		return 0;
	e = m->data + m->face_ofs + 12u * face;
	out->page = rd16( e );
	if( out->page == GC_LMAP_NONE || out->page >= m->pages )
		return 0;
	out->x = rd16( e + 2 );
	out->y = rd16( e + 4 );
	out->w = rd16( e + 6 );
	out->h = rd16( e + 8 );
	out->src_w = e[10];
	out->src_h = e[11];
	return out->w && out->h;
}

int gc_lmap_page_size( const gc_lmap_t *m, uint32_t page, uint16_t *w, uint16_t *h )
{
	const uint8_t *e;

	if( !m || page >= m->pages )
		return -1;
	e = m->data + m->page_ofs + 4u * page;
	*w = rd16( e );
	*h = rd16( e + 2 );
	return 0;
}

static float axis( float s, uint16_t stored, uint8_t src )
{
	if( stored <= 1 || src <= 1 )
		return ( (float)stored - 1.0f ) * 0.5f;	/* one texel (or a 1-wide grid): its centre */
	return s * (float)( stored - 1 ) / (float)( src - 1 );
}

void gc_lmap_uv( const gc_lmap_face_t *f, uint16_t page_w, uint16_t page_h, float s, float t, float *u, float *v )
{
	/* texel centres sit at +0.5 (GX_LINEAR) */
	*u = ( (float)f->x + axis( s, f->w, f->src_w ) + 0.5f ) / (float)page_w;
	*v = ( (float)f->y + axis( t, f->h, f->src_h ) + 0.5f ) / (float)page_h;
}
