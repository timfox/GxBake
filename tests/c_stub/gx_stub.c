/* Recording GX stub: prints each call, for tests/test_port_c.py. */
#include <stdio.h>
#include "ogc/gx.h"
void GX_InitTlutObj( GXTlutObj *o, void *l, u8 f, u16 n ) { (void)o; (void)l; printf( "InitTlutObj %d %d\n", f, n ); }
void GX_LoadTlut( GXTlutObj *o, u32 n ) { (void)o; printf( "LoadTlut %u\n", (unsigned)n ); }
void GX_InitTexObj( GXTexObj *o, void *i, u16 w, u16 h, u8 f, u8 ws, u8 wt, u8 m ) { (void)o; (void)i; printf( "InitTexObj %d %d %d %d %d %d\n", w, h, f, ws, wt, m ); }
void GX_InitTexObjCI( GXTexObj *o, void *i, u16 w, u16 h, u8 f, u8 ws, u8 wt, u8 m, u32 t ) { (void)o; (void)i; printf( "InitTexObjCI %d %d %d %d %d %d %u\n", w, h, f, ws, wt, m, (unsigned)t ); }
void GX_InitTexObjLOD( GXTexObj *o, u8 a, u8 b, f32 c, f32 d, f32 e, u8 f, u8 g, u8 h ) { (void)o; printf( "InitTexObjLOD %d %d %g %g %g %d %d %d\n", a, b, c, d, e, f, g, h ); }
void GX_LoadTexObj( GXTexObj *o, u8 m ) { printf( "LoadTexObj %u %d\n", (unsigned)o->id, m ); }
void GX_SetTexCoordGen( u16 tc, u32 t, u32 s, u32 m ) { printf( "SetTexCoordGen %d %u %u %u\n", tc, (unsigned)t, (unsigned)s, (unsigned)m ); }
void GX_SetNumTexGens( u32 n ) { printf( "SetNumTexGens %u\n", (unsigned)n ); }
void GX_SetNumChans( u8 n ) { printf( "SetNumChans %d\n", n ); }
void GX_SetChanCtrl( s32 c, u8 e, u8 a, u8 m, u8 l, u8 d, u8 t ) { printf( "SetChanCtrl %d %d %d %d %d %d %d\n", (int)c, e, a, m, l, d, t ); }
void GX_SetNumTevStages( u8 n ) { printf( "SetNumTevStages %d\n", n ); }
void GX_SetTevOrder( u8 s, u8 t, u32 m, u8 c ) { printf( "SetTevOrder %d %d %u %d\n", s, t, (unsigned)m, c ); }
void GX_SetTevColorIn( u8 s, u8 a, u8 b, u8 c, u8 d ) { printf( "SetTevColorIn %d %d %d %d %d\n", s, a, b, c, d ); }
void GX_SetTevColorOp( u8 s, u8 o, u8 b, u8 sc, u8 c, u8 r ) { printf( "SetTevColorOp %d %d %d %d %d %d\n", s, o, b, sc, c, r ); }
void GX_SetTevAlphaIn( u8 s, u8 a, u8 b, u8 c, u8 d ) { printf( "SetTevAlphaIn %d %d %d %d %d\n", s, a, b, c, d ); }
void GX_SetTevAlphaOp( u8 s, u8 o, u8 b, u8 sc, u8 c, u8 r ) { printf( "SetTevAlphaOp %d %d %d %d %d %d\n", s, o, b, sc, c, r ); }
void GX_SetTevKColorSel( u8 s, u8 k ) { printf( "SetTevKColorSel %d %d\n", s, k ); }
void GX_SetTevDirect( u8 s ) { printf( "SetTevDirect %d\n", s ); }
void GX_SetNumIndStages( u8 n ) { printf( "SetNumIndStages %d\n", n ); }
void GX_SetIndTexOrder( u8 i, u8 t, u8 m ) { printf( "SetIndTexOrder %d %d %d\n", i, t, m ); }
void GX_SetIndTexCoordScale( u8 i, u8 s, u8 t ) { printf( "SetIndTexCoordScale %d %d %d\n", i, s, t ); }
void GX_SetIndTexMatrix( u8 x, f32 m[2][3], s8 e ) { printf( "SetIndTexMatrix %d %g %g %g %g %g %g %d\n", x, m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], e ); }
void GX_SetTevIndirect( u8 s, u8 i, u8 f, u8 b, u8 m, u8 ws, u8 wt, u8 ap, u8 u, u8 a ) { printf( "SetTevIndirect %d %d %d %d %d %d %d %d %d %d\n", s, i, f, b, m, ws, wt, ap, u, a ); }
void GX_SetAlphaCompare( u8 c0, u8 r0, u8 o, u8 c1, u8 r1 ) { printf( "SetAlphaCompare %d %d %d %d %d\n", c0, r0, o, c1, r1 ); }
void GX_SetZCompLoc( u8 b ) { printf( "SetZCompLoc %d\n", b ); }
void GX_SetBlendMode( u8 t, u8 s, u8 d, u8 o ) { printf( "SetBlendMode %d %d %d %d\n", t, s, d, o ); }
void GX_LoadProjectionMtx( f32 m[4][4], u8 t ) { printf( "LoadProjectionMtx %d %g %g %g %g %g\n", t, m[0][0], m[1][1], m[2][2], m[2][3], m[3][2] ); }
