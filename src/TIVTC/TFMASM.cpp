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

#include "TFMasm.h"
#include "emmintrin.h"

void checkSceneChangePlanar_1_SSE2(const uint8_t *prvp, const uint8_t *srcp,
  int height, int width, int prv_pitch, int src_pitch, uint64_t &diffp)
{
  __m128i sum = _mm_setzero_si128();
  while (height--) {
    for (int x = 0; x < width; x += 16)
    {
      __m128i src1 = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
      __m128i src2 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x));
      __m128i sad = _mm_sad_epu8(src1, src2);
      sum = _mm_add_epi32(sum, sad);
    }
    prvp += prv_pitch;
    srcp += src_pitch;
  }
  __m128i res = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  diffp = _mm_cvtsi128_si32(res);
}


void checkSceneChangePlanar_2_SSE2(const uint8_t *prvp, const uint8_t *srcp,
  const uint8_t *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, uint64_t &diffp, uint64_t &diffn)
{
  __m128i sump = _mm_setzero_si128();
  __m128i sumn = _mm_setzero_si128();
  while (height--) {
    for (int x = 0; x < width; x += 16)
    {
      __m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
      __m128i src_curr = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x));
      __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i *>(nxtp + x));
      __m128i sadp = _mm_sad_epu8(src_prev, src_curr);
      __m128i sadn = _mm_sad_epu8(src_next, src_curr);
      sump = _mm_add_epi32(sump, sadp);
      sumn = _mm_add_epi32(sumn, sadn);
    }
    prvp += prv_pitch;
    srcp += src_pitch;
    nxtp += nxt_pitch;
  }
  __m128i resp = _mm_add_epi32(sump, _mm_srli_si128(sump, 8));
  diffp = _mm_cvtsi128_si32(resp);
  __m128i resn = _mm_add_epi32(sumn, _mm_srli_si128(sumn, 8));
  diffn = _mm_cvtsi128_si32(resn);
}


void checkSceneChangeYUY2_1_SSE2(const uint8_t *prvp, const uint8_t *srcp,
  int height, int width, int prv_pitch, int src_pitch, uint64_t &diffp)
{
  __m128i sum = _mm_setzero_si128();
  __m128i lumaMask = _mm_set1_epi16(0x00FF);
  while (height--) {
    for (int x = 0; x < width; x += 16)
    {
      __m128i src1 = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
      __m128i src2 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x));
      src1 = _mm_and_si128(src1, lumaMask);
      src2 = _mm_and_si128(src2, lumaMask);
      __m128i sad = _mm_sad_epu8(src1, src2);
      sum = _mm_add_epi32(sum, sad);
    }
    prvp += prv_pitch;
    srcp += src_pitch;
  }
  __m128i res = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  diffp = _mm_cvtsi128_si32(res);
}


void checkSceneChangeYUY2_2_SSE2(const uint8_t *prvp, const uint8_t *srcp,
  const uint8_t *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, uint64_t &diffp, uint64_t &diffn)
{
  __m128i sump = _mm_setzero_si128();
  __m128i sumn = _mm_setzero_si128();
  __m128i lumaMask = _mm_set1_epi16(0x00FF);
  while (height--) {
    for (int x = 0; x < width; x += 16)
    {
      __m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
      __m128i src_curr = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x));
      __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i *>(nxtp + x));
      src_prev = _mm_and_si128(src_prev, lumaMask);
      src_curr = _mm_and_si128(src_curr, lumaMask);
      src_next = _mm_and_si128(src_next, lumaMask);
      __m128i sadp = _mm_sad_epu8(src_prev, src_curr);
      __m128i sadn = _mm_sad_epu8(src_next, src_curr);
      sump = _mm_add_epi32(sump, sadp);
      sumn = _mm_add_epi32(sumn, sadn);
    }
    prvp += prv_pitch;
    srcp += src_pitch;
    nxtp += nxt_pitch;
  }
  __m128i resp = _mm_add_epi32(sump, _mm_srli_si128(sump, 8));
  diffp = _mm_cvtsi128_si32(resp);
  __m128i resn = _mm_add_epi32(sumn, _mm_srli_si128(sumn, 8));
  diffn = _mm_cvtsi128_si32(resn);
}


AVS_FORCEINLINE __m128i compareFieldsSlowCal0_SSSE3(int ebx, __m128i readmsk, uint8_t* t_mapp, uint8_t* t_mapn)
{
    //eax = (t_mapp[ebx] << 3) + t_mapn[ebx];	 // diff from prev asm block (at buildDiffMapPlane2): <<3 instead of <<2
    auto temp = _mm_loadu_si128((__m128i*)(t_mapp + ebx));	//128bit境界以外にアクセスするが大丈夫？ ＋配列の順番注意
    temp = _mm_shuffle_epi8(temp, readmsk);		//_mm_cvtepu8_epi16(temp);		//incl==1では下位8個を16bitに展開、==2では2byte毎
    __m128i eax = _mm_slli_epi16(temp, 3);		//ssse3 to do shift and mul... 8bit命令はない
    temp = _mm_loadu_si128((__m128i*)(t_mapn + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    eax = _mm_add_epi16(eax, temp);

    return eax;
}


AVS_FORCEINLINE void compareFieldsSlowCal1_SSSE3(int ebx, __m128i eax, __m128i readmsk,
    const uint8_t* t_prvpf, const uint8_t* t_prvnf,
    const uint8_t* t_curpf, const uint8_t* t_curf, const uint8_t* t_curnf,
    const uint8_t* t_nxtpf, const uint8_t* t_nxtnf,
    uint64_t& accumPc, uint64_t& accumNc, uint64_t& accumPm, uint64_t& accumNm, uint64_t& accumPml, uint64_t& accumNml)
{
    __m128i temp,temp2, a_curr, a_prev, a_next, diff_p_c, diff_n_c, eaxmsk;
    __m128i zero = _mm_setzero_si128();

    //a_curr = t_curpf[ebx] + (t_curf[ebx] << 2) + t_curnf[ebx];
    temp = _mm_loadu_si128((__m128i*)(t_curpf + ebx));
    a_curr = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_loadu_si128((__m128i*)(t_curf + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_slli_epi16(temp, 2);
    a_curr = _mm_add_epi16(a_curr, temp);
    temp = _mm_loadu_si128((__m128i*)(t_curnf + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    a_curr = _mm_add_epi16(a_curr, temp);


    //a_prev = 3 * (t_prvpf[ebx] + t_prvnf[ebx]);
    temp = _mm_loadu_si128((__m128i*)(t_prvpf + ebx));
    a_prev = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_loadu_si128((__m128i*)(t_prvnf + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    a_prev = _mm_add_epi16(a_prev, temp);
    temp = _mm_set1_epi16(3);
    a_prev = _mm_mullo_epi16(a_prev, temp);


    //a_next = 3 * (t_nxtpf[ebx] + t_nxtnf[ebx]);
    temp = _mm_loadu_si128((__m128i*)(t_nxtpf + ebx));
    a_next = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_loadu_si128((__m128i*)(t_nxtnf + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    a_next = _mm_add_epi16(a_next, temp);
    temp = _mm_set1_epi16(3);
    a_next = _mm_mullo_epi16(a_next, temp);


    //diff_p_c = abs(a_prev - a_curr);
    temp = _mm_sub_epi16(a_prev, a_curr);
    diff_p_c = _mm_abs_epi16(temp);	//ssse3

    //diff_n_c = abs(a_next - a_curr);
    temp = _mm_sub_epi16(a_next, a_curr);
    diff_n_c = _mm_abs_epi16(temp);

    //if ((eax & 9) != 0){
    //	if (diff_p_c > 23) {accumPc  += diff_p_c;}
    //	if (diff_n_c > 23) {accumNc  += diff_n_c;}
    //}
    temp = _mm_and_si128(eax, _mm_set1_epi16(9));
    temp = _mm_cmpeq_epi16(temp, zero);		//(eax&9 == 0) ? -1 : 0
    //auto eaxmsk = _mm_add_epi16(temp, _mm_set1_epi16(1));	//(eax&9 == 0) ? 0 : 1
    eaxmsk = _mm_cmpeq_epi16(temp, zero);	//(eax&9 != 0) ? -1 : 0

    temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(23));	//(diffpc>23) ? -1 : 0
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffpc>23) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_p_c);					//eaxmsk∧(diffpc>23) ? diffpc : 0
    //temp = _mm_hadd_epi16(temp, zero);						//sum ssse3 水平方向は遅い
    //temp = _mm_hadd_epi16(temp, zero);
    //temp = _mm_hadd_epi16(temp, zero);
    temp2 = _mm_srli_si128(temp, 8);
    temp  = _mm_add_epi16(temp, temp2);      //....7+3,6+2,5+1,4+0 = ....3,2,1,0 + ....7,6,5,4
    temp2 = _mm_srli_si128(temp, 4);        //......      7+3,6+2
    temp  = _mm_add_epi16(temp, temp2);      //......  7+5+3+1,6+4+2+0
    temp2 = _mm_srli_si128(temp, 2);        //.......         7+5+3+1
    temp  = _mm_add_epi16(temp, temp2);      //......  7+5+3+1+6+4+2+0
    temp  = _mm_cvtepu16_epi32(temp);       //sse4.1 上位にはごみが残ってるので ダイナミックレンジは8bit*8個なので16bitで十分 符号なしでよい
    accumPc += _mm_cvtsi128_si32(temp);

    temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(23));	//(diffnc>23) ? -1 : 0
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eax9∧(diffnc>23) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_n_c);					//eax9∧(diffnc>23) ? diffnc : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumNc += _mm_cvtsi128_si32(temp);

    //if ((eax & 18) != 0){
    //	if (diff_p_c > 42) {accumPm  += diff_p_c;}
    //	if (diff_n_c > 42) {accumNm  += diff_n_c;}
    //}
    temp = _mm_and_si128(eax, _mm_set1_epi16(18));
    temp = _mm_cmpeq_epi16(temp, zero);		//(eax&18 == 0) ? -1 : 0
    eaxmsk = _mm_cmpeq_epi16(temp, zero);	//(eax&18 != 0) ? -1 : 0

    temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));	//(diffpc>42) ? ffff : 0
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumPm += _mm_cvtsi128_si32(temp);

    temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));	//(diffnc>42) ? ffff : 0
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffnc>42) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_n_c);					//eaxmsk∧(diffnc>42) ? diff_n_c : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumNm += _mm_cvtsi128_si32(temp);


    //if ((eax & 36) != 0){
    //	if (diff_p_c > 42) {accumPml += diff_p_c;}
    //	if (diff_n_c > 42) {accumNml += diff_n_c;}
    //}
    temp = _mm_and_si128(eax, _mm_set1_epi16(36));
    temp = _mm_cmpeq_epi16(temp, zero);		//(eax&36 == 0) ? -1 : 0
    eaxmsk = _mm_cmpeq_epi16(temp, zero);	//(eax&36 != 0) ? -1 : 0

    temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));	//(diffpc>42) ? ffff : 0
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumPml += _mm_cvtsi128_si32(temp);

    temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));	//(diffnc>42) ? ffff : 0
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffnc>42)で1、else0
    temp = _mm_mullo_epi16(temp, diff_n_c);					//eaxmsk∧(diffnc>42)でdiffnc、else0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumNml += _mm_cvtsi128_si32(temp);

    return;
}


AVS_FORCEINLINE void compareFieldsSlowCal2_SSE41(int ebx, __m128i eax, __m128i readmsk, int sft,
    const uint8_t* t_prvf0, const uint8_t* t_prvf1, const uint8_t* t_prvf2,
    const uint8_t* t_curf0, const uint8_t* t_curf1,
    const uint8_t* t_nxtf0, const uint8_t* t_nxtf1, const uint8_t* t_nxtf2,
    uint64_t& accumPc, uint64_t& accumNc, uint64_t& accumPm, uint64_t& accumNm, uint64_t& accumPml, uint64_t& accumNml)
{
    __m128i temp,temp2, a_curr, a_prev, a_next, diff_p_c, diff_n_c, eaxmsk;
    __m128i zero = _mm_setzero_si128();

    // additional difference from TFM 1144

    //if ((eax & 56) == 0) continue; //field0
    //if ((eax &  7) == 0) continue; //field1
    temp = _mm_set1_epi16(7 << sft);
    if (_mm_testz_si128(eax, temp)) { return; }	//sse4.1

    //	a_curr = 3 * (t_curpf[ebx] + t_curf[ebx]);	//field0
    //	a_curr = 3 * (t_curf[ebx] + t_curnf[ebx]);	//field1
    temp = _mm_loadu_si128((__m128i*)(t_curf0 + ebx));
    a_curr = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_loadu_si128((__m128i*)(t_curf1 + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    a_curr = _mm_add_epi16(a_curr, temp);
    temp = _mm_set1_epi16(3);
    a_curr = _mm_mullo_epi16(a_curr, temp);


    //	a_prev = t_prvppf[ebx] + (t_prvpf[ebx] << 2) + t_prvnf[ebx];	//field0
    //	a_prev = t_prvpf[ebx] + (t_prvnf[ebx] << 2) + t_prvnnf[ebx];	//field1
    temp = _mm_loadu_si128((__m128i*)(t_prvf0 + ebx));
    a_prev = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_loadu_si128((__m128i*)(t_prvf1 + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_slli_epi16(temp, 2);
    a_prev = _mm_add_epi16(a_prev, temp);
    temp = _mm_loadu_si128((__m128i*)(t_prvf2 + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    a_prev = _mm_add_epi16(a_prev, temp);


    //	a_next = t_nxtppf[ebx] + (t_nxtpf[ebx] << 2) + t_nxtnf[ebx];	//field0
    //	a_next = t_nxtpf[ebx] + (t_nxtnf[ebx] << 2) + t_nxtnnf[ebx];	//field1
    temp = _mm_loadu_si128((__m128i*)(t_nxtf0 + ebx));
    a_next = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_loadu_si128((__m128i*)(t_nxtf1 + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    temp = _mm_slli_epi16(temp, 2);
    a_next = _mm_add_epi16(a_next, temp);
    temp = _mm_loadu_si128((__m128i*)(t_nxtf2 + ebx));
    temp = _mm_shuffle_epi8(temp, readmsk);
    a_next = _mm_add_epi16(a_next, temp);

    //	diff_p_c = abs(a_prev - a_curr);
    temp = _mm_sub_epi16(a_prev, a_curr);
    diff_p_c = _mm_abs_epi16(temp);

    //	diff_n_c = abs(a_next - a_curr);
    temp = _mm_sub_epi16(a_next, a_curr);
    diff_n_c = _mm_abs_epi16(temp);

    //	if ((eax & 8) != 0){	//field0:8 field1:1
    //		if (diff_p_c > 23) {accumPc  += diff_p_c;}
    //		if (diff_n_c > 23) {accumNc  += diff_n_c;}
    //	}
    temp = _mm_and_si128(eax, _mm_set1_epi16(1 << sft));		//
    temp = _mm_cmpeq_epi16(temp, zero);						//(eax&8 == 0) ? -1 : 0
    eaxmsk = _mm_cmpeq_epi16(temp, zero);					//(eax&8 == 0) ? 0 : -1

    temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(23));
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffpc>23) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_p_c);					//eaxmsk∧(diffpc>23) ? diffpc : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumPc += _mm_cvtsi128_si32(temp);

    temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(23));
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffnc>23) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_n_c);					//eaxmsk∧(diffnc>23) ? diffnc : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumNc += _mm_cvtsi128_si32(temp);

    //	if ((eax & 16) != 0){	//field0:16 field1:2
    //		if (diff_p_c > 42) {accumPm  += diff_p_c;}
    //		if (diff_n_c > 42) {accumNm  += diff_n_c;}
    //	}
    temp = _mm_and_si128(eax, _mm_set1_epi16(2 << sft));
    temp = _mm_cmpeq_epi16(temp, zero);						//(eax&16 == 0) ? -1 : 0
    eaxmsk = _mm_cmpeq_epi16(temp, zero);					//(eax&16 == 0) ? 0 : -1

    temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumPm += _mm_cvtsi128_si32(temp);

    temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffnc>42) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_n_c);					//eaxmsk∧(diffnc>42) ? diff_n_c : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumNm += _mm_cvtsi128_si32(temp);

    //	if ((eax & 32) != 0){	//field0:32 field1:4
    //		if (diff_p_c > 42) {accumPml += diff_p_c;}
    //		if (diff_n_c > 42) {accumNml += diff_n_c;}
    //	}
    temp = _mm_and_si128(eax, _mm_set1_epi16(4 << sft));		//
    temp = _mm_cmpeq_epi16(temp, zero);						//(eax&32 == 0) ? -1 : 0
    eaxmsk = _mm_cmpeq_epi16(temp, zero);					//(eax&32 == 0) ? 0 : -1

    temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumPml += _mm_cvtsi128_si32(temp);

    temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));
    temp = _mm_mullo_epi16(temp, eaxmsk);					//eaxmsk∧(diffnc>42) ? 1 : 0
    temp = _mm_mullo_epi16(temp, diff_n_c);					//eaxmsk∧(diffnc>42) ? diff_n_c : 0
    temp2 = _mm_srli_si128(temp, 8);        //sum
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 4);
    temp  = _mm_add_epi16(temp, temp2);
    temp2 = _mm_srli_si128(temp, 2);
    temp  = _mm_add_epi16(temp, temp2);
    temp  = _mm_cvtepu16_epi32(temp);
    accumNml += _mm_cvtsi128_si32(temp);
}

