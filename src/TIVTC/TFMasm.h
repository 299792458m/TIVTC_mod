/*
**                    TIVTC v1.0.6 for Avisynth 2.6 interface
**
**   TIVTC includes a field matching filter (TFM) and a decimation
**   filter (TDecimate) which can be used together to achieve an
**   IVTC or for other uses. TIVTC currently supports YV12 and
**   YUY2 colorspaces.
**
**   Copyright (C) 2004-2008 Kevin Stone, additional work (C) 2017 pinterf
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef __TFMASM_H__
#define __TFMASM_H__

#include <windows.h>
#include <xmmintrin.h>
#include <emmintrin.h>	//SSE2
//#include <pmmintrin.h>	//SSE3
//#include <tmmintrin.h>	//SSSE3
//#include <smmintrin.h>	//SSE4.1
#include <nmmintrin.h>	//SSE4.2
#include <immintrin.h>	//AVX/AVX2/FMA
#include "internal.h"

  void checkSceneChangeYUY2_2_SSE2(const unsigned char *prvp, const unsigned char *srcp,
    const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
    int nxt_pitch, unsigned long &diffp, unsigned long &diffn);
  void checkSceneChangeYUY2_1_SSE2(const unsigned char *prvp, const unsigned char *srcp,
    int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp);
  void checkSceneChangeYV12_1_SSE2(const unsigned char *prvp, const unsigned char *srcp,
    int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp);
  void checkSceneChangeYV12_2_SSE2(const unsigned char *prvp, const unsigned char *srcp,
    const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
    int nxt_pitch, unsigned long &diffp, unsigned long &diffn);

  template<bool aligned>
  void check_combing_SSE2(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb,
    __m128i thresh6w);
  
  template<bool aligned>
  void check_combing_SSE2_Luma(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb,
    __m128i thresh6w);
  
  template<bool aligned>
  void check_combing_SSE2_M1(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int dst_pitch, __m128i thresh);
  
  template<bool aligned>
  void check_combing_SSE2_Luma_M1(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int dst_pitch, __m128i thresh);
  
  void buildABSDiffMask_SSE2(const unsigned char *prvp, const unsigned char *nxtp,
    unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height);
  
  void buildABSDiffMask2_SSE2(const unsigned char *prvp, const unsigned char *nxtp,
    unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height);
  
  void compute_sum_8x8_sse2(const unsigned char *srcp, int pitch, int &sum);

  template<bool aligned>
  void compute_sum_8x16_sse2_luma(const unsigned char *srcp, int pitch, int &sum);

  extern __m128i compareFieldsSlowCal0_SSSE3(int ebx, __m128i readmsk, unsigned char *t_mapp, unsigned char *t_mapn);

  extern void compareFieldsSlowCal1_SSSE3(int ebx, __m128i eax, __m128i readmsk,
	  const unsigned char *t_prvpf, const unsigned char*t_prvnf,
	  const unsigned char *t_curpf, const unsigned char*t_curf, const unsigned char*t_curnf,
	  const unsigned char *t_nxtpf, const unsigned char*t_nxtnf,
	  unsigned long  &accumPc, unsigned long  &accumNc, unsigned long  &accumPm, unsigned long  &accumNm, unsigned long  &accumPml, unsigned long  &accumNml);

  extern void compareFieldsSlowCal2_SSE41(int ebx, __m128i eax, __m128i readmsk,int sft,
	  const unsigned char *t_prvf0, const unsigned char*t_prvf1, const unsigned char*t_prvf2,
	  const unsigned char *t_curf0, const unsigned char*t_curf1,
	  const unsigned char *t_nxtf0, const unsigned char*t_nxtf1, const unsigned char*t_nxtf2,
	  unsigned long  &accumPc, unsigned long  &accumNc, unsigned long  &accumPm, unsigned long  &accumNm, unsigned long  &accumPml, unsigned long  &accumNml);

  void buildDiffMapPlaneYV12_SSE42(const unsigned char *tbuffer,
	  unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int Height, int Width, int tpitch);

#ifdef ALLOW_MMX
  void checkSceneChangeYUY2_1_ISSE(const unsigned char *prvp, const unsigned char *srcp,
    int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp);
  void checkSceneChangeYUY2_2_ISSE(const unsigned char *prvp, const unsigned char *srcp,
    const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
    int nxt_pitch, unsigned long &diffp, unsigned long &diffn);
  void checkSceneChangeYV12_1_ISSE(const unsigned char *prvp, const unsigned char *srcp,
    int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp);
  void checkSceneChangeYV12_2_ISSE(const unsigned char *prvp, const unsigned char *srcp,
    const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
    int nxt_pitch, unsigned long &diffp, unsigned long &diffn);
  void checkSceneChangeYUY2_1_MMX(const unsigned char *prvp, const unsigned char *srcp,
    int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp);
  void checkSceneChangeYUY2_2_MMX(const unsigned char *prvp, const unsigned char *srcp,
    const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
    int nxt_pitch, unsigned long &diffp, unsigned long &diffn);
  void checkSceneChangeYV12_1_MMX(const unsigned char *prvp, const unsigned char *srcp,
    int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp);
  void checkSceneChangeYV12_2_MMX(const unsigned char *prvp, const unsigned char *srcp,
    const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
    int nxt_pitch, unsigned long &diffp, unsigned long &diffn);
  void check_combing_MMX(const unsigned char *srcp, unsigned char *dstp, int width,
    int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb,
    __int64 thresh6w);
  void check_combing_MMX_Luma(const unsigned char *srcp, unsigned char *dstp, int width,
    int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb,
    __int64 thresh6w);
  void check_combing_iSSE(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb,
    __int64 thresh6w);
  void check_combing_iSSE_Luma(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb,
    __int64 thresh6w);
  void check_combing_MMX_M1(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int dst_pitch, __int64 thresh);
  void check_combing_MMX_Luma_M1(const unsigned char *srcp, unsigned char *dstp,
    int width, int height, int src_pitch, int dst_pitch, __int64 thresh);
  void buildABSDiffMask_MMX(const unsigned char *prvp, const unsigned char *nxtp,
    unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height);
  void buildABSDiffMask2_MMX(const unsigned char *prvp, const unsigned char *nxtp,
    unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height);
  // these are mmx because of the block size
  void compute_sum_8x8_mmx(const unsigned char *srcp, int pitch, int &sum);
  void compute_sum_8x8_isse(const unsigned char *srcp, int pitch, int &sum);
  void compute_sum_8x16_mmx_luma(const unsigned char *srcp, int pitch, int &sum);
  void compute_sum_8x16_isse_luma(const unsigned char *srcp, int pitch, int &sum);
#endif

#endif // TFMASM_H__
