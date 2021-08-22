/*
**                    TIVTC for AviSynth 2.6 interface
**
**   TIVTC includes a field matching filter (TFM) and a decimation
**   filter (TDecimate) which can be used together to achieve an
**   IVTC or for other uses. TIVTC currently supports 8 bit planar YUV and
**   YUY2 colorspaces.
**
**   Copyright (C) 2004-2008 Kevin Stone, additional work (C) 2020 pinterf
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

#include "internal.h"
#include <nmmintrin.h>	//SSE4.2
#include <immintrin.h>	//AVX/AVX2/FMA

void checkSceneChangeYUY2_1_SSE2(const uint8_t* prvp, const uint8_t* srcp,
  int height, int width, int prv_pitch, int src_pitch, uint64_t& diffp);
void checkSceneChangeYUY2_2_SSE2(const uint8_t* prvp, const uint8_t* srcp,
  const uint8_t* nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, uint64_t& diffp, uint64_t& diffn);

void checkSceneChangePlanar_1_SSE2(const uint8_t* prvp, const uint8_t* srcp,
  int height, int width, int prv_pitch, int src_pitch, uint64_t& diffp);
void checkSceneChangePlanar_2_SSE2(const uint8_t* prvp, const uint8_t* srcp,
  const uint8_t* nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, uint64_t& diffp, uint64_t& diffn);


extern __m128i compareFieldsSlowCal0_SSSE3(int ebx, __m128i readmsk, uint8_t* t_mapp, uint8_t* t_mapn);

extern void compareFieldsSlowCal1_SSSE3(int ebx, __m128i eax, __m128i readmsk,
	const uint8_t* t_prvpf, const uint8_t* t_prvnf,
	const uint8_t* t_curpf, const uint8_t* t_curf, const uint8_t* t_curnf,
	const uint8_t* t_nxtpf, const uint8_t* t_nxtnf,
	uint64_t& accumPc, uint64_t& accumNc, uint64_t& accumPm, uint64_t& accumNm, uint64_t& accumPml, uint64_t& accumNml);

extern void compareFieldsSlowCal2_SSE41(int ebx, __m128i eax, __m128i readmsk, int sft,
	const uint8_t* t_prvf0, const uint8_t* t_prvf1, const uint8_t* t_prvf2,
	const uint8_t* t_curf0, const uint8_t* t_curf1,
	const uint8_t* t_nxtf0, const uint8_t* t_nxtf1, const uint8_t* t_nxtf2,
	uint64_t& accumPc, uint64_t& accumNc, uint64_t& accumPm, uint64_t& accumNm, uint64_t& accumPml, uint64_t& accumNml);


#endif // TFMASM_H__
