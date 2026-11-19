/* A recording stand-in for libogc's <ogc/gx.h>, only for host tests of port/gc_mat.c:
 * every GX call prints its name and arguments, one per line. Names and signatures
 * follow libogc; the constant values here are arbitrary (except where the code
 * relies on order, e.g. GX_TG_BUMP0..7 and GX_TEXMAP0..7 being consecutive). */
#ifndef STUB_OGC_GX_H
#define STUB_OGC_GX_H
#include <stdint.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef int8_t s8; typedef int32_t s32; typedef float f32;
typedef struct { u32 id; } GXTexObj;
typedef struct { u32 id; } GXTlutObj;
enum {
	GX_FALSE = 0, GX_TRUE = 1, GX_DISABLE = 0, GX_ENABLE = 1,
	GX_TF_CI4 = 8, GX_TF_CI8 = 9,
	GX_CLAMP = 0, GX_REPEAT = 1,
	GX_NEAR = 0, GX_LINEAR = 1, GX_LIN_MIP_LIN = 5, GX_ANISO_1 = 0,
	GX_TEXCOORD0 = 0, GX_TEXCOORD1 = 1, GX_TEXCOORDNULL = 0xff,
	GX_TEXMAP0 = 0, GX_TEXMAP_NULL = 0xff,
	GX_COLOR0A0 = 4, GX_COLOR1A1 = 5, GX_COLORNULL = 0xff,
	GX_TG_MTX2x4 = 1, GX_TG_BUMP0 = 2,
	GX_TG_NRM = 101, GX_TG_TEX0 = 104, GX_TG_TEX1 = 105, GX_TG_TEXCOORD0 = 112,
	GX_IDENTITY = 60, GX_TEXMTX0 = 30,
	GX_CC_CPREV = 0, GX_CC_C0 = 2, GX_CC_C1 = 4, GX_CC_TEXC = 8, GX_CC_RASC = 10, GX_CC_KONST = 14, GX_CC_ZERO = 15,
	GX_CA_APREV = 0, GX_CA_TEXA = 4, GX_CA_RASA = 5, GX_CA_ZERO = 7,
	GX_TEV_ADD = 0, GX_TEV_SUB = 1, GX_TB_ZERO = 0, GX_TB_ADDHALF = 1,
	GX_CS_SCALE_1 = 0, GX_CS_SCALE_2 = 1, GX_CS_SCALE_4 = 2,
	GX_TEVPREV = 0, GX_TEVREG0 = 1, GX_TEVREG1 = 2,
	GX_SRC_REG = 0, GX_DF_NONE = 0, GX_DF_CLAMP = 2, GX_AF_SPEC = 0, GX_AF_NONE = 2,
	GX_ALWAYS = 7, GX_GEQUAL = 6, GX_AOP_AND = 0,
	GX_BM_NONE = 0, GX_BM_BLEND = 1, GX_BL_ZERO = 0, GX_BL_ONE = 1, GX_BL_SRCALPHA = 4, GX_BL_INVSRCALPHA = 5, GX_LO_CLEAR = 0,
	GX_INDTEXSTAGE0 = 0, GX_ITS_1 = 0, GX_ITM_0 = 1, GX_ITF_8 = 0, GX_ITB_ST = 3, GX_ITW_OFF = 0, GX_ITBA_OFF = 0,
	GX_TEV_KCSEL_1_4 = 6
};
void GX_InitTlutObj( GXTlutObj *obj, void *lut, u8 fmt, u16 entries );
void GX_LoadTlut( GXTlutObj *obj, u32 name );
void GX_InitTexObj( GXTexObj *obj, void *img, u16 w, u16 h, u8 fmt, u8 ws, u8 wt, u8 mip );
void GX_InitTexObjCI( GXTexObj *obj, void *img, u16 w, u16 h, u8 fmt, u8 ws, u8 wt, u8 mip, u32 tlut );
void GX_InitTexObjLOD( GXTexObj *obj, u8 minf, u8 magf, f32 minlod, f32 maxlod, f32 bias, u8 clamp, u8 edge, u8 aniso );
void GX_LoadTexObj( GXTexObj *obj, u8 map );
void GX_SetTexCoordGen( u16 tc, u32 type, u32 src, u32 mtx );
void GX_SetNumTexGens( u32 n );
void GX_SetNumChans( u8 n );
void GX_SetChanCtrl( s32 chan, u8 enable, u8 amb, u8 mat, u8 lights, u8 diff, u8 attn );
void GX_SetNumTevStages( u8 n );
void GX_SetTevOrder( u8 st, u8 tc, u32 map, u8 color );
void GX_SetTevColorIn( u8 st, u8 a, u8 b, u8 c, u8 d );
void GX_SetTevColorOp( u8 st, u8 op, u8 bias, u8 scale, u8 clamp, u8 reg );
void GX_SetTevAlphaIn( u8 st, u8 a, u8 b, u8 c, u8 d );
void GX_SetTevAlphaOp( u8 st, u8 op, u8 bias, u8 scale, u8 clamp, u8 reg );
void GX_SetTevKColorSel( u8 st, u8 sel );
void GX_SetTevDirect( u8 st );
void GX_SetNumIndStages( u8 n );
void GX_SetIndTexOrder( u8 ind, u8 tc, u8 map );
void GX_SetIndTexCoordScale( u8 ind, u8 s, u8 t );
void GX_SetIndTexMatrix( u8 mtx, f32 m[2][3], s8 e );
void GX_SetTevIndirect( u8 st, u8 ind, u8 fmt, u8 bias, u8 mtx, u8 ws, u8 wt, u8 addprev, u8 utclod, u8 a );
void GX_SetAlphaCompare( u8 c0, u8 r0, u8 op, u8 c1, u8 r1 );
void GX_SetZCompLoc( u8 before );
void GX_SetBlendMode( u8 type, u8 src, u8 dst, u8 op );
#endif
