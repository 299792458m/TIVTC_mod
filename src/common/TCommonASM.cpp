/*
**   Helper methods for TIVTC and TDeint
**
**
**   Copyright (C) 2004-2007 Kevin Stone, additional work (C) 2020 pinterf
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

#include "TCommonASM.h"
#include <immintrin.h>
#include <algorithm>
#include <intrin.h>

void absDiff_SSE2(const uint8_t *srcp1, const uint8_t *srcp2,
  uint8_t *dstp, int src1_pitch, int src2_pitch, int dst_pitch, int width,
  int height, int mthresh1, int mthresh2)
{
  // for non-YUY2, mthresh1 and 2 are the same
  mthresh1 = std::min(std::max(255 - mthresh1, 0), 255);
  mthresh2 = std::min(std::max(255 - mthresh2, 0), 255);

  auto onesMask = _mm_set1_epi8(1);
  auto sthresh = _mm_set1_epi16((mthresh2 << 8) + mthresh1);
  auto all_ff = _mm_set1_epi8(-1);
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; x += 16)
    {
      auto src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp1 + x));
      auto src2 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp2 + x));
      auto diff12 = _mm_subs_epu8(src1, src2);
      auto diff21 = _mm_subs_epu8(src2, src1);
      auto diff = _mm_or_si128(diff12, diff21);
      auto addedsthresh = _mm_adds_epu8(diff, sthresh);
      auto cmpresult = _mm_cmpeq_epi8(addedsthresh, all_ff);
      auto res = _mm_xor_si128(cmpresult, all_ff);
      auto tmp = _mm_and_si128(res, onesMask);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), tmp);
      /*
      if (abs(srcp1[x] - srcp2[x]) < mthresh1) dstp[x] = 1;
      else dstp[x] = 0;
      ++x;
      if (abs(srcp1[x] - srcp2[x]) < mthresh2) dstp[x] = 1;
      else dstp[x] = 0;
      */
    }
    srcp1 += src1_pitch;
    srcp2 += src2_pitch;
    dstp += dst_pitch;
  }

}

// fills target byte buffer with 1 where absdiff is less that threshold, 0 otherwise
void absDiff_c(const uint8_t* srcp1, const uint8_t* srcp2,
  uint8_t* dstp, int src1_pitch, int src2_pitch, int dst_pitch, int width,
  int height, int mthresh1, int mthresh2)
{
  // for non-YUY2 mthresh1 and 2 are the same
  // dstp is a simple 1-byte format buffer (no high bit depth content)
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      if (abs(srcp1[x] - srcp2[x]) < mthresh1) dstp[x] = 1;
      else dstp[x] = 0;
      ++x; // next planar pixel or YUY2 chroma
      if (abs(srcp1[x] - srcp2[x]) < mthresh2) dstp[x] = 1;
      else dstp[x] = 0;
    }
    srcp1 += src1_pitch;
    srcp2 += src2_pitch;
    dstp += dst_pitch;
  }
}

void absDiff_uint16_c(const uint8_t* srcp1, const uint8_t* srcp2,
  uint8_t* dstp, int src1_pitch, int src2_pitch, int dst_pitch, int width,
  int height, int mthresh)
{
  // dstp is a simple 1-byte format buffer (no high bit depth content)
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      if (abs(reinterpret_cast<const uint16_t *>(srcp1)[x] - reinterpret_cast<const uint16_t*>(srcp2)[x]) < mthresh)
        dstp[x] = 1;
      else
        dstp[x] = 0;
    }
    srcp1 += src1_pitch;
    srcp2 += src2_pitch;
    dstp += dst_pitch;
  }
}

// different path if not mod16, but only for remaining 8 bytes
template<typename pixel_t>
void buildABSDiffMask_SSE2(const uint8_t* prvp, const uint8_t* nxtp,
  uint8_t* dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int rowsize,
  int height)
{
  __m128i diffpn, diffnp;

  if (!(rowsize & 15)) // exact mod16
  {
    while (height--) {
      for (int x = 0; x < rowsize; x += 16)
      {
        auto src_prev = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x));
        auto src_next = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x));
        if constexpr (sizeof(pixel_t) == 1) {
          diffpn = _mm_subs_epu8(src_prev, src_next);
          diffnp = _mm_subs_epu8(src_next, src_prev);
        }
        else {
          diffpn = _mm_subs_epu16(src_prev, src_next);
          diffnp = _mm_subs_epu16(src_next, src_prev);
        }
        __m128i diff = _mm_or_si128(diffpn, diffnp);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), diff);
      }
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
  else {
    rowsize -= 8; // last chunk is 8 bytes instead of 16
    while (height--) {
      int x;
      for (x = 0; x < rowsize; x += 16)
      {
        __m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x));
        __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x));
        if constexpr (sizeof(pixel_t) == 1) {
          diffpn = _mm_subs_epu8(src_prev, src_next);
          diffnp = _mm_subs_epu8(src_next, src_prev);
        }
        else {
          diffpn = _mm_subs_epu16(src_prev, src_next);
          diffnp = _mm_subs_epu16(src_next, src_prev);
        }
        __m128i diff = _mm_or_si128(diffpn, diffnp);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), diff);
      }
      // remaining half block
      __m128i src_prev = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(prvp + x));
      __m128i src_next = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(nxtp + x));
      if constexpr (sizeof(pixel_t) == 1) {
        diffpn = _mm_subs_epu8(src_prev, src_next);
        diffnp = _mm_subs_epu8(src_next, src_prev);
      }
      else {
        diffpn = _mm_subs_epu16(src_prev, src_next);
        diffnp = _mm_subs_epu16(src_next, src_prev);
      }
      __m128i diff = _mm_or_si128(diffpn, diffnp);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), diff);
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
}

template<typename pixel_t>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif 
void buildABSDiffMask_AVX2(const uint8_t* prvp, const uint8_t* nxtp, uint8_t* dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height)
{
    while (height--) {
        for (int x = 0; x < width; x += 32)
        {
            auto src_prev = _mm256_load_si256(reinterpret_cast<const __m256i*>(prvp + x));
            auto src_next = _mm256_load_si256(reinterpret_cast<const __m256i*>(nxtp + x));

            __m256i diffpn, diffnp;
            if constexpr (sizeof(pixel_t) == 1) {
                diffpn = _mm256_subs_epu8(src_prev, src_next);
                diffnp = _mm256_subs_epu8(src_next, src_prev);
            }
            else {
                diffpn = _mm256_subs_epu16(src_prev, src_next);
                diffnp = _mm256_subs_epu16(src_next, src_prev);
            }
            auto diff = _mm256_or_si256(diffpn, diffnp);

            //auto diffmax = _mm256_max_epu8(src_prev, src_next);	//����Ȃ̂�����
            //auto diffmin = _mm256_min_epu8(src_prev, src_next);
            //auto diff = _mm256_subs_epu8(diffmax, diffmin);
            _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), diff);
        }
        prvp += prv_pitch;
        nxtp += nxt_pitch;
        dstp += dst_pitch;
    }
    _mm256_zeroupper();
}

template<typename pixel_t, bool YUY2_LumaOnly>
void buildABSDiffMask_c(const uint8_t* prvp, const uint8_t* nxtp,
  uint8_t* dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height)
{
  if (width <= 0)
    return;

  if constexpr (YUY2_LumaOnly) {
    // 8 bit only
    // C version is quicker if dealing with every second (luma) pixel
    // SSE2: no luma-nonluma difference because with omitting chroma is slower.
    // YUY2: YUYVYUYV... skip U and V
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; x += 4)
      {
        dstp[x + 0] = abs(prvp[x + 0] - nxtp[x + 0]);
        dstp[x + 2] = abs(prvp[x + 2] - nxtp[x + 2]);
      }
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
  else {
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; x++)
      {
        reinterpret_cast<pixel_t *>(dstp)[x] = abs(reinterpret_cast<const pixel_t*>(prvp)[x] - reinterpret_cast<const pixel_t*>(nxtp)[x]);
      }
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
}

template<typename pixel_t>
void do_buildABSDiffMask(const uint8_t* prvp, const uint8_t* nxtp, uint8_t* tbuffer,
  int prv_pitch, int nxt_pitch, int tpitch, int width, int height, bool YUY2_LumaOnly, int cpuFlags)
{
  if (cpuFlags & CPUF_SSE2 && width >= 8)
  {
    const int rowsize = width * sizeof(pixel_t);
    const int rowsizemod8 = rowsize / 8 * 8;
    // SSE2 is not YUY2 chroma-ignore template, it's quicker if not skipping each YUY2 chroma
    if ((cpuFlags & CPUF_AVX2) && (width >= 32) &&!((intptr_t(prvp) | intptr_t(nxtp) | prv_pitch | nxt_pitch | tpitch | rowsizemod8) & 31))
        buildABSDiffMask_AVX2<pixel_t>(prvp, nxtp, tbuffer, prv_pitch, nxt_pitch, tpitch, rowsizemod8, height);
    else
        buildABSDiffMask_SSE2<pixel_t>(prvp, nxtp, tbuffer, prv_pitch, nxt_pitch, tpitch, rowsizemod8, height);

    if(YUY2_LumaOnly)
      buildABSDiffMask_c<pixel_t, true>(
        prvp + rowsizemod8, 
        nxtp + rowsizemod8, 
        tbuffer + rowsizemod8, 
        prv_pitch, nxt_pitch, tpitch, 
        width - rowsizemod8 / sizeof(pixel_t), 
        height);
    else
      buildABSDiffMask_c<pixel_t, false>(
        prvp + rowsizemod8,
        nxtp + rowsizemod8,
        tbuffer + rowsizemod8,
        prv_pitch, nxt_pitch, tpitch,
        width - rowsizemod8 / sizeof(pixel_t),
        height);
  }
  else {
    if (YUY2_LumaOnly)
      buildABSDiffMask_c<pixel_t, true>(prvp, nxtp, tbuffer, prv_pitch, nxt_pitch, tpitch, width, height);
    else
      buildABSDiffMask_c<pixel_t, false>(prvp, nxtp, tbuffer, prv_pitch, nxt_pitch, tpitch, width, height);
  }
}
// instantiate
template void do_buildABSDiffMask<uint8_t>(const uint8_t* prvp, const uint8_t* nxtp, uint8_t* tbuffer,
  int prv_pitch, int nxt_pitch, int tpitch, int width, int height, bool YUY2_LumaOnly, int cpuFlags);
template void do_buildABSDiffMask<uint16_t>(const uint8_t* prvp, const uint8_t* nxtp, uint8_t* tbuffer,
  int prv_pitch, int nxt_pitch, int tpitch, int width, int height, bool YUY2_LumaOnly, int cpuFlags);


template<typename pixel_t>
void do_buildABSDiffMask2(const uint8_t* prvp, const uint8_t* nxtp, uint8_t* dstp,
  int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height, bool YUY2_LumaOnly, int cpuFlags, int bits_per_pixel)
{
  if ((cpuFlags & CPUF_SSE2) && width >= 8) // yes, width and not row_size
  {
    int mod8Width = width / 8 * 8;
    if constexpr(sizeof(pixel_t) == 8)
      buildABSDiffMask2_uint8_SSE2(prvp, nxtp, dstp, prv_pitch, nxt_pitch, dst_pitch, mod8Width, height);
    else
      buildABSDiffMask2_uint16_SSE2(prvp, nxtp, dstp, prv_pitch, nxt_pitch, dst_pitch, mod8Width, height, bits_per_pixel);
    if (YUY2_LumaOnly)
      buildABSDiffMask2_c<pixel_t, true>(
        prvp + mod8Width * sizeof(pixel_t),
        nxtp + mod8Width * sizeof(pixel_t),
        dstp + mod8Width,
        prv_pitch, nxt_pitch, dst_pitch, width - mod8Width, height, bits_per_pixel);
    else
      buildABSDiffMask2_c<pixel_t, false>(
        prvp + mod8Width * sizeof(pixel_t), 
        nxtp + mod8Width * sizeof(pixel_t),
        dstp + mod8Width, // dstp is really 8 bits
        prv_pitch, nxt_pitch, dst_pitch, width - mod8Width, height, bits_per_pixel);
  }
  else {
    if (YUY2_LumaOnly)
      buildABSDiffMask2_c<pixel_t, true>(prvp, nxtp, dstp, prv_pitch, nxt_pitch, dst_pitch, width, height, bits_per_pixel);
    else
      buildABSDiffMask2_c<pixel_t, false>(prvp, nxtp, dstp, prv_pitch, nxt_pitch, dst_pitch, width, height, bits_per_pixel);
  }
}
// instantiate
template void do_buildABSDiffMask2<uint8_t>(const uint8_t* prvp, const uint8_t* nxtp, uint8_t* dstp,
  int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height, bool YUY2_LumaOnly, int cpuFlags, int bits_per_pixel);
template void do_buildABSDiffMask2<uint16_t>(const uint8_t* prvp, const uint8_t* nxtp, uint8_t* dstp,
  int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height, bool YUY2_LumaOnly, int cpuFlags, int bits_per_pixel);

// Finally this is common for TFM and TDeint, planar and YUY2 (luma, luma+chroma))
// This C code replaces some thousand line of copy pasted original inline asm lines
// (plus handles 10+bits)

// distance of neighboring pixels:
// 1 for planar any
// 2 for YUY2 luma
// 4 for YUY2 chroma
template<typename pixel_t, int bits_per_pixel, int DIST>
static AVS_FORCEINLINE void AnalyzeOnePixel(uint8_t* dstp,
  const pixel_t* dppp, const pixel_t* dpp,
  const pixel_t* dp,
  const pixel_t* dpn, const pixel_t* dpnn,
  int& x, int& y, int& Width, int& Height)
{
  constexpr int Const3 = 3 << (bits_per_pixel - 8);
  constexpr int Const19 = 19 << (bits_per_pixel - 8);

  if (dp[x] <= Const3)
    return;

  if (dp[x - DIST] <= Const3 && dp[x + DIST] <= Const3 &&
    dpp[x - DIST] <= Const3 && dpp[x] <= Const3 && dpp[x + DIST] <= Const3 &&
    dpn[x - DIST] <= Const3 && dpn[x] <= Const3 && dpn[x + DIST] <= Const3)
    return;

  dstp[x]++;

  if (dp[x] <= Const19)
    return;

  int edi = 0;
  int lower = 0;
  int upper = 0;

  if (dpp[x - DIST] > Const19) edi++;
  if (dpp[x] > Const19) edi++;
  if (dpp[x + DIST] > Const19) edi++;

  if (edi != 0) upper = 1;

  if (dp[x - DIST] > Const19) edi++;
  if (dp[x + DIST] > Const19) edi++;

  int esi = edi;

  if (dpn[x - DIST] > Const19) edi++;
  if (dpn[x] > Const19) edi++;
  if (dpn[x + DIST] > Const19) edi++;

  if (edi <= 2)
    return;

  int count = edi;
  if (count != esi) {
    lower = 1;
    if (upper != 0) {
      dstp[x] += 2;
      return;
    }
  }

  int lower2 = 0;
  int upper2 = 0;

  int startx, stopx;

  constexpr bool YUY2_chroma = (DIST == 4);

  if (YUY2_chroma) {
    const int firstchroma = (x & 2) + 1;
    startx = x - 4 * 4 < firstchroma ? firstchroma : x - 4 * 4;
    stopx = x + 4 * 4 + 2 > Width ? Width : x + 4 * 4 + 2;
  }
  else {
    startx = x < 4 * DIST ? 0 : x - 4 * DIST;
    stopx = x + 4 * DIST + DIST > Width ? Width : x + 4 * DIST + DIST;
  }

  if (y != 2) {
    for (int esi = startx; esi < stopx; esi += DIST) {
      if (dppp[esi] > Const19) {
        upper2 = 1;
        break;
      }
    }
  }

  for (int esi = startx; esi < stopx; esi += DIST)
  {
    if (dpp[esi] > Const19)
      upper = 1;
    if (dpn[esi] > Const19)
      lower = 1;
    if (upper != 0 && lower != 0)
      break;
  }

  if (y != Height - 4) {
    for (int esi = startx; esi < stopx; esi += DIST)
    {
      if (dpnn[esi] > Const19) {
        lower2 = 1;
        break;
      }
    }
  }

  if (upper == 0) {
    if (lower == 0 || lower2 == 0) {
      if (count > 4)
        dstp[x] += 4;
    }
    else {
      dstp[x] += 2;
    }
  }
  else {
    if (lower != 0 || upper2 != 0) {
      dstp[x] += 2;
    }
    else {
      if (count > 4)
        dstp[x] += 4;
    }
  }
}

static inline void AnalyzeOnePixel_AVX2_881(uint8_t* dstp,
    const uint8_t* dppp, const uint8_t* dpp,
    const uint8_t* dp,
    const uint8_t* dpn, const uint8_t* dpnn,
    int& x, int& y, int& Width, int& Height)
{
    unsigned int dp_d, dpp_d, dpn_d;
    unsigned int tmpi;

    dp_d = *(unsigned int*)(dp + x - 1);				//-1,0,1,2��4������Ă��� ���̕�(�g�����O)�ɒu����x�̈ˑ��֌W�����܂��؂蕪�����Ȃ��H�̂��A�N�Z�X���x��

    //��pixel��-3�����ŃX�L�b�v
    //if (_mm_testz_si128(temp, _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-4)))
    if ((dp_d & 0x0000FC00) == 0)
    {
        auto temp = _mm256_loadu_si256((__m256i*)(dp + x));	//x=1����n�܂�̂��C�ɂȂ邪�A���x�I�ɂ͂��܂�ς��Ȃ����ۂ�
        if (_mm256_testz_si256(temp, _mm256_set1_epi8(-4))) {	//sse4.1
            //x += 31;return; //�A���X�L�b�v�Ȃ��̎�
            do {	//�A���X�L�b�v�p �����ŋǏ������邩����for�ł�邩�����ŉ��Z�͕ς��Ȃ��̂ł��܂���ʂȂ��H
                x += 32;
                temp = _mm256_loadu_si256((__m256i*)(dp + x));
            } while ((_mm256_testz_si256(temp, _mm256_set1_epi8(-4))) && (x < Width));
            //x--;return;	//continue����++����邩��
        }

        temp = _mm256_and_si256(temp, _mm256_set1_epi8(-4));	// subs(0.5cpi)��������(0.33cpi)
        temp = _mm256_cmpeq_epi8(temp, _mm256_setzero_si256());	// <=3 ? ff : 0
        tmpi = _mm256_movemask_epi8(temp);	// temp[] <=3 ? 1 : 0 �e�o�C�g�̍ŏ�ʃr�b�g�𒊏o

        //���̎��_�� <=3 ? 0 : 1

        //tzcnt�̏ꍇ(BMI���g�����AVX2)
        //tmpi = ~tmpi;						// temp[] <=3 ? 0 : 1  si128�ł͏��16bit�ɂ�1������ andnot���g���������̂�����
        //tmpi = _mm_tzcnt_32(tmpi);	        //tzcnt�̏ꍇ(BMI) bsf��荂��
        //x += tmpi -1; return;	//continue��(for��post)��+1�����̂�-1

        //popcnt�ő�ց�SSE42
        tmpi = _mm_popcnt_u32(tmpi & (~tmpi - 1));	//������Ƃ����x�����E�E�ESSE���͂�����
        //x += tmpi - 1; return;                //continue��(for��post)��+1�����̂�-1
        x += tmpi;
        dp_d = *(unsigned int*)(dp + x - 1);	//continue���Ȃ��ꍇ

        //bsf�̏ꍇ POPCNT���g���Ȃ��Ȃ炱���Ȃ�
        //long unsigned int tmpl;			//32bit���w�肵�Ȃ���bsf������������E�E�E
        //tmpi = ~tmpi;						// temp[] <=3 ? 0 : 1
        //_BitScanForward(&tmpl,tmpi);	//�g�ݍ��݊֐��g�p 1��������Ȃ��ꍇ�͕Ԃ�l��0�ɂȂ�(�������Ă邪)
        //x += tmpl-1;					//continue��(for��post)��+1�����̂�-1
        //continue;

        _mm256_zeroupper();
    }

    //8�ߖT�����ׂ�<=3�ŃX�L�b�v
    //if (dp[x - 1] <= 3 && dp[x + 1] <= 3 &&
    //dpp[x - 1] <= 3 && dpp[x] <= 3 && dpp[x + 1] <= 3 &&
    //dpn[x - 1] <= 3 && dpn[x] <= 3 && dpn[x + 1] <= 3) continue;
    dpp_d = *(unsigned int*)(dpp + x - 1);
    dpn_d = *(unsigned int*)(dpn + x - 1);		//�͈�(Width)�𒴂��ăA�N�Z�X����̂Œ���

    if (((dp_d & 0x00FC00FC) | (dpp_d & 0x00FCFCFC) | (dpn_d & 0x00FCFCFC)) == 0)	return;
    //if ( _mm_testz_si128(temp, _mm_set_epi32(0,0x00FCFCFC,0x00FC00FC,0x00FCFCFC)) ) {continue;}	//dpp[x]�͔��肵�Ȃ��̂Œ���

    dstp[x] = 1;	//+=1

    //�J�����g��dp<=19�ŃX�L�b�v
    //if (dp[x] <= 19) continue;
    if ((dp_d & 0x0000ff00) <= 0x00001300) return;

    //8�ߖT��dp<=19�ŃX�L�b�v
    auto temp = _mm_set_epi32(0, dpn_d, dp_d, dpp_d);	//dpn,dp.dpp�̏�
    temp = _mm_subs_epu8(temp, _mm_set1_epi8(19));		//unsigned��temp[]-19  ���ɂȂ�ꍇ��0 ==19�ȉ���0
    temp = _mm_cmpeq_epi8(temp, _mm_setzero_si128());	//temp[] <=19 ? ff : 0 
    //temp = _mm_min_epu8(temp, _mm_set1_epi8(127));	//�{����epu8�Ŕ�r���������A���߂��Ȃ��̂ŁA128�ȏ��127�Ɋۂ߂�H
    //temp = _mm_sub_epi8(temp, _mm_set1_epi8(20));		//signed��temp[]-20  ��(�ŏ�ʃr�b�g������)==19�ȉ� �ł�128�ȏ゠��ꍇ�Ɏg���Ȃ�
    int ans = _mm_movemask_epi8(temp);	// temp[] <=19 ? 1 : 0 �e�o�C�g�̍ŏ�ʃr�b�g�𒊏o
    ans = ~ans;								// temp[] > 19 ? 1 : 0 ��ʂ�1�ɂȂ�̂Œ���
    ans = ans & 0x0757;						//�s�v����(dp[x],[x+2])�𖳎�

    int count = _mm_popcnt_u32(ans);			//bit���J�E���g sse4.2�v
    if (count <= 2) return;

    bool upper = 0;
    bool upper2 = 0;
    bool lower = 0;
    bool lower2 = 0;

    upper = ((ans & 0x0007) != 0) ? 1 : 0;	//dpp_d�ŉ��Z
    lower = ((ans & 0x0700) != 0) ? 1 : 0;	//dpn_d�ŉ��Z

    if (upper && lower) {					//dpp/dpn�ŗ������Z
        dstp[x] = 3;	//+=2
        return;
    }
    //�ȍ~upper|lower�̕Е��̂݃t���O����

    int start = x - 4;				// b11
    if (start < 0) start = 0;

    int end = x + 5;				//  p3
    if (end > Width) end = Width;	//

    __m128i  msk;
    int size = (end - start);
    //msk.m128i_u64[1] = (size > 8) ? ( ( (long long)1 << (8*(size -8)) ) -1 ) : 0;	//msk�̏�� size��MAX9byte�Ȃ̂ŁA����0��ff�݂̂ɂ����Ȃ�Ȃ�
    //msk.m128i_u64[1] = (size > 8) ? 255 : 0;	//msk�̏�� size��MAX9byte�Ȃ̂ŁA����0��ff�݂̂ɂ����Ȃ�Ȃ�
    //msk.m128i_u64[0] = ( ( (long long)1 << (8*size) ) -1 );
    //64bit�̉��Z�������聫�̕��������C��
    switch (size) {
    case 0:				msk = _mm_set_epi32(0, 0, 0, 0);				break;
    case 1:				msk = _mm_set_epi32(0, 0, 0, 0xff);			    break;
    case 2:				msk = _mm_set_epi32(0, 0, 0, 0xffff);				break;
    case 3:				msk = _mm_set_epi32(0, 0, 0, 0xffffff);				break;
    case 4:				msk = _mm_set_epi32(0, 0, 0, 0xffffffff);				break;
    case 5:				msk = _mm_set_epi32(0, 0, 0xff, 0xffffffff);				break;
    case 6:				msk = _mm_set_epi32(0, 0, 0xffff, 0xffffffff);				break;
    case 7:				msk = _mm_set_epi32(0, 0, 0xffffff, 0xffffffff);				break;
    case 8:				msk = _mm_set_epi32(0, 0, 0xffffffff, 0xffffffff);				break;
    default:case 9:		msk = _mm_set_epi32(0, 0xff, 0xffffffff, 0xffffffff);				break;
    }

    if (y != 2) {	// >
        temp = _mm_loadu_si128((__m128i*)(dppp + start));
        temp = _mm_min_epu8(temp, _mm_set1_epi8(127));	//�{����epu8�Ŕ�r���������A���߂��Ȃ��̂ŁA128�ȏ��127�Ɋۂ߂�H
        temp = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
        upper2 = (_mm_testz_si128(temp, msk)) ? 0 : 1;	//==0��1�Ȃ̂ŁA���] sete/setne�Ƃ������߂Ŕ��]�ł��邩�火�̏��׍H�s�v�H
        //temp = _mm_subs_epu8( temp, _mm_set1_epi8(19) );	//temp[]-19  ���ɂȂ�ꍇ��0 ==19�ȉ���0
        //temp = _mm_cmpeq_epi8( temp, _mm_setzero_si128() );	//temp[] <=19 ? ff : 0    0�ɂȂ�==19�ȉ�
        //upper2= _mm_testz_si128(temp,msk);
    }
    if (upper) {
        temp = _mm_loadu_si128((__m128i*)(dpn + start));
        temp = _mm_min_epu8(temp, _mm_set1_epi8(127));
        temp = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
        lower = (_mm_testz_si128(temp, msk)) ? 0 : 1;
    }
    else {	//lower==1
        temp = _mm_loadu_si128((__m128i*)(dpp + start));
        temp = _mm_min_epu8(temp, _mm_set1_epi8(127));
        temp = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
        upper = (_mm_testz_si128(temp, msk)) ? 0 : 1;
    }
    if (y != Height - 4) {	// <
        temp = _mm_loadu_si128((__m128i*)(dpnn + start));
        temp = _mm_min_epu8(temp, _mm_set1_epi8(127));
        temp = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
        lower2 = (_mm_testz_si128(temp, msk)) ? 0 : 1;
    }

    //p13 :
    if (upper == 0) {
        if (lower == 0 || lower2 == 0) {	// p17:
            if (count > 4) {
                dstp[x] = 5;	//+=4
            }
        }
        else {
            dstp[x] = 3;	// p18	+=2
        }
    }
    else {	//upper=1
        if (lower != 0 || upper2 != 0) {
            dstp[x] = 3;		//+=2
        }
        else {
            if (count > 4) {
                dstp[x] = 5;	//+=4
            }
        }
    }
}

// Common TDeint and TFM version
template<typename pixel_t, int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.2")))
#endif
void AnalyzeDiffMask_Planar(uint8_t* dstp, int dst_pitch, uint8_t* tbuffer8, int tpitch, int Width, int Height, int cpuFlags)
{
  tpitch /= sizeof(pixel_t);
  const pixel_t* tbuffer = reinterpret_cast<const pixel_t*>(tbuffer8);
  const pixel_t* dppp = tbuffer - tpitch;
  const pixel_t* dpp = tbuffer;
  const pixel_t* dp = tbuffer + tpitch;
  const pixel_t* dpn = tbuffer + tpitch * 2;
  const pixel_t* dpnn = tbuffer + tpitch * 3;

  if (cpuFlags & CPUF_AVX2) {
    for (int y = 2; y < Height - 2; y += 2) {
        for (int x = 1; x < Width - 1; x++) {
            if constexpr (sizeof(pixel_t) == 1)
                AnalyzeOnePixel_AVX2_881(dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height);
            else
                AnalyzeOnePixel<pixel_t, bits_per_pixel, 1>(dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height);
        }
        dppp += tpitch;
        dpp += tpitch;
        dp += tpitch;
        dpn += tpitch;
        dpnn += tpitch;
        dstp += dst_pitch;
    }
  }
  else {
      for (int y = 2; y < Height - 2; y += 2) {
          for (int x = 1; x < Width - 1; x++) {
              AnalyzeOnePixel<pixel_t, bits_per_pixel, 1>(dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height);
          }
          dppp += tpitch;
          dpp += tpitch;
          dp += tpitch;
          dpn += tpitch;
          dpnn += tpitch;
          dstp += dst_pitch;
      }
  }
}
// instantiate
template void AnalyzeDiffMask_Planar<uint8_t,8>(uint8_t* dstp, int dst_pitch, uint8_t* tbuffer8, int tpitch, int Width, int Height, int cpuFlags);
template void AnalyzeDiffMask_Planar<uint16_t, 10>(uint8_t* dstp, int dst_pitch, uint8_t* tbuffer8, int tpitch, int Width, int Height, int cpuFlags);
template void AnalyzeDiffMask_Planar<uint16_t, 12>(uint8_t* dstp, int dst_pitch, uint8_t* tbuffer8, int tpitch, int Width, int Height, int cpuFlags);
template void AnalyzeDiffMask_Planar<uint16_t, 14>(uint8_t* dstp, int dst_pitch, uint8_t* tbuffer8, int tpitch, int Width, int Height, int cpuFlags);
template void AnalyzeDiffMask_Planar<uint16_t, 16>(uint8_t* dstp, int dst_pitch, uint8_t* tbuffer8, int tpitch, int Width, int Height, int cpuFlags);

// TDeint and TFM version
void AnalyzeDiffMask_YUY2(uint8_t* dstp, int dst_pitch, uint8_t* tbuffer, int tpitch, int Width, int Height, bool mChroma)
{
  // YUY2 we won't touch it if it works. No hbd here
  const uint8_t* dppp = tbuffer - tpitch;
  const uint8_t* dpp = tbuffer;
  const uint8_t* dp = tbuffer + tpitch;
  const uint8_t* dpn = tbuffer + tpitch * 2;
  const uint8_t* dpnn = tbuffer + tpitch * 3;
  // reconstructed from inline 700+ lines asm by pinterf

  if (mChroma) // TFM YUY2's mChroma bool parameter
  {
    for (int y = 2; y < Height - 2; y += 2) {
      // small difference from planar: x starts from 2 instead of 1; ends at width-2 instead of width-1
      // [YUYV]YUYVYUYVYUYV...YUYV[YUYV]
      for (int x = 4; x < Width - 4; x += 1)
      {
        AnalyzeOnePixel<uint8_t, 8, 2>(dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height);
        // skip to chroma
        x++;
        AnalyzeOnePixel<uint8_t, 8, 4>(dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height);

      }
      dppp += tpitch;
      dpp += tpitch;
      dp += tpitch;
      dpn += tpitch;
      dpnn += tpitch;
      dstp += dst_pitch;
    }
  }
  else {
    // no YUY2 chroma, LumaOnly
    for (int y = 2; y < Height - 2; y += 2) {
      for (int x = 4; x < Width - 4; x += 2)
      {
        AnalyzeOnePixel<uint8_t, 8, 2>(dstp, dppp, dpp, dp, dpn, dpnn, x, y, Width, Height);
      }
      dppp += tpitch;
      dpp += tpitch;
      dp += tpitch;
      dpn += tpitch;
      dpnn += tpitch;
      dstp += dst_pitch;
    }
  }
}

// HBD ready
template<typename pixel_t, bool YUY2_LumaOnly>
void buildABSDiffMask2_c(const uint8_t* prvp, const uint8_t* nxtp,
  uint8_t* dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width, int height, int bits_per_pixel)
{
  if (width <= 0)
    return;

  constexpr int inc = YUY2_LumaOnly ? 2 : 1;
  const int Const19 = 19 << (bits_per_pixel - 8);
  const int Const3 = 3 << (bits_per_pixel - 8);
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; x += inc)
    {
      const int diff = abs(reinterpret_cast<const pixel_t *>(prvp)[x] - reinterpret_cast<const pixel_t*>(nxtp)[x]);
      if (diff > Const19) dstp[x] = 3;
      else if (diff > Const3) dstp[x] = 1;
      else dstp[x] = 0;
    }
    prvp += prv_pitch;
    nxtp += nxt_pitch;
    dstp += dst_pitch;
  }
}


static AVS_FORCEINLINE __m128i _MM_CMPLE_EPU16(__m128i x, __m128i y)
{
  // Returns 0xFFFF where x <= y:
  return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}

void buildABSDiffMask2_uint8_SSE2(const uint8_t* prvp, const uint8_t* nxtp,
  uint8_t* dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width,
  int height)
{
  auto onesMask = _mm_set1_epi8(0x01); // byte target!
  auto twosMask = _mm_set1_epi8(0x02);
  auto all_ff = _mm_set1_epi8(-1);
  // C version: 19 and 3
  // 255 - 1 - 19 = 235
  // 255 - 1 - 3 = 251

  // diff > 19 => diff - 19 > 0 => 
  // diff - 19 >= 1 => diff - 19 - 1 +255 >= 255 =>
  // add_satutare(diff, 255 - 19 - 1) == 255
  const int Const251 = 255 - 1 - 3;
  const int Const235 = 255 - 1 - 19;

  auto Compare251 = _mm_set1_epi8((char)Const251);
  auto Compare235 = _mm_set1_epi8((char)Const235);

  if (!(width & 15)) // exact mod16
  {
    while (height--) {
      for (int x = 0; x < width; x += 16)
      {
        __m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x));
        __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x));
        __m128i diffpn = _mm_subs_epu8(src_prev, src_next);
        __m128i diffnp = _mm_subs_epu8(src_next, src_prev);
        __m128i diff = _mm_or_si128(diffpn, diffnp);
        /*
        const int diff = abs(prvp[x] - nxtp[x]);
        if (diff > 19) dstp[x] |= 2; // 2 + 1
        if (diff > 3) dstp[x] |= 1;
        else dstp[x] = 0;

        */
        __m128i added251 = _mm_adds_epu8(diff, Compare251);
        __m128i added235 = _mm_adds_epu8(diff, Compare235);
        auto cmp251 = _mm_cmpeq_epi8(added251, all_ff);
        auto cmp235 = _mm_cmpeq_epi8(added235, all_ff);
        // target is byte buffer
        __m128i tmp1 = _mm_and_si128(cmp251, onesMask);
        __m128i tmp2 = _mm_and_si128(cmp235, twosMask);
        __m128i tmp = _mm_or_si128(tmp1, tmp2);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), tmp);
      }
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
  else {
    width -= 8; // last chunk is 8 bytes instead of 16
    while (height--) {
      int x; // intentionally not in 'for'
      for (x = 0; x < width; x += 16)
      {
        __m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x));
        __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x));
        __m128i diffpn = _mm_subs_epu8(src_prev, src_next);
        __m128i diffnp = _mm_subs_epu8(src_next, src_prev);
        __m128i diff = _mm_or_si128(diffpn, diffnp);
        /*
        const int diff = abs(prvp[x] - nxtp[x]);
        if (diff > 19) dstp[x] |= 2; // 2 + 1
        if (diff > 3) dstp[x] |= 1;
        else dstp[x] = 0;
        */
        __m128i added251 = _mm_adds_epu8(diff, Compare251);
        __m128i added235 = _mm_adds_epu8(diff, Compare235);
        auto cmp251 = _mm_cmpeq_epi8(added251, all_ff);
        auto cmp235 = _mm_cmpeq_epi8(added235, all_ff);
        // target is byte buffer
        __m128i tmp1 = _mm_and_si128(cmp251, onesMask);
        __m128i tmp2 = _mm_and_si128(cmp235, twosMask);
        __m128i tmp = _mm_or_si128(tmp1, tmp2);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), tmp);
      }
      // rest 8 bytes
      __m128i src_prev = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(prvp + x));
      __m128i src_next = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(nxtp + x));
      __m128i diffpn = _mm_subs_epu8(src_prev, src_next);
      __m128i diffnp = _mm_subs_epu8(src_next, src_prev);
      __m128i diff = _mm_or_si128(diffpn, diffnp);
      __m128i added251 = _mm_adds_epu8(diff, Compare251);
      __m128i added235 = _mm_adds_epu8(diff, Compare235);
      auto cmp251 = _mm_cmpeq_epi8(added251, all_ff);
      auto cmp235 = _mm_cmpeq_epi8(added235, all_ff);
      __m128i tmp1 = _mm_and_si128(cmp251, onesMask);
      __m128i tmp2 = _mm_and_si128(cmp235, twosMask);
      __m128i tmp = _mm_or_si128(tmp1, tmp2);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), tmp);

      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
}

void buildABSDiffMask2_uint16_SSE2(const uint8_t* prvp, const uint8_t* nxtp,
  uint8_t* dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width,
  int height, int bits_per_pixel)
{
  auto onesMask = _mm_set1_epi8(0x01); // byte target!
  auto twosMask = _mm_set1_epi8(0x02);
  // C version: 19 and 3

  const int Const19plus1 = (19 << (bits_per_pixel - 8)) + 1;
  const int Const3plus1 = (3 << (bits_per_pixel - 8)) + 1;

  auto Compare19plus1 = _mm_set1_epi16((short)Const19plus1);
  auto Compare3plus1 = _mm_set1_epi16((short)Const3plus1);

  if (!(width & 15)) // exact mod16
  {
    while (height--) {
      for (int x = 0; x < width; x += 16)
      {
        // 16 byte result needs 32 byte source (16 x uint16_t pixels)

        /*
        const int diff = abs(prvp[x] - nxtp[x]);
        if (diff > Const19) dstp[x] |= 2; // 2 + 1
        if (diff > Const3) dstp[x] |= 1;
        else dstp[x] = 0;

        if (diff > 19) ==> diff >= 19+1
        if (diff > 3) ==> diff >= 3+1
        */

        auto src_prev_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x * 2));
        auto src_next_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x * 2));
        auto diffpn_lo = _mm_subs_epu16(src_prev_lo, src_next_lo);
        auto diffnp_lo = _mm_subs_epu16(src_next_lo, src_prev_lo);
        auto diff_lo = _mm_or_si128(diffpn_lo, diffnp_lo);

        auto cmp19_lo = _MM_CMPLE_EPU16(Compare19plus1, diff_lo); // FFFF where 20 <= diff (19 < diff)
        auto cmp3_lo = _MM_CMPLE_EPU16(Compare3plus1, diff_lo); // FFFF where 4 <= diff (3 < diff)

        auto src_prev_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x * 2 + 16));
        auto src_next_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x * 2 + 16));
        auto diffpn_hi = _mm_subs_epu16(src_prev_hi, src_next_hi);
        auto diffnp_hi = _mm_subs_epu16(src_next_hi, src_prev_hi);
        auto diff_hi = _mm_or_si128(diffpn_hi, diffnp_hi);

        auto cmp19_hi = _MM_CMPLE_EPU16(Compare19plus1, diff_hi); // FFFF where 20 <= diff (19 < diff)
        auto cmp3_hi = _MM_CMPLE_EPU16(Compare3plus1, diff_hi); // FFFF where 4 <= diff (3 < diff)

        // make bytes from wordBools
        auto cmp251 = _mm_packus_epi16(cmp3_lo, cmp3_hi);
        auto cmp235 = _mm_packus_epi16(cmp19_lo, cmp19_hi);

        // target is byte buffer!
        auto tmp1 = _mm_and_si128(cmp251, onesMask);
        auto  tmp2 = _mm_and_si128(cmp235, twosMask);
        auto  tmp = _mm_or_si128(tmp1, tmp2);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), tmp);
      }
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
  else {
    width -= 8; // last chunk is 8 bytes instead of 16
    while (height--) {
      int x; // intentionally not in 'for'
      for (x = 0; x < width; x += 16)
      {
        auto src_prev_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x * 2));
        auto src_next_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x * 2));
        auto diffpn_lo = _mm_subs_epu16(src_prev_lo, src_next_lo);
        auto diffnp_lo = _mm_subs_epu16(src_next_lo, src_prev_lo);
        auto diff_lo = _mm_or_si128(diffpn_lo, diffnp_lo);

        auto cmp19_lo = _MM_CMPLE_EPU16(Compare19plus1, diff_lo); // FFFF where 20 <= diff (19 < diff)
        auto cmp3_lo = _MM_CMPLE_EPU16(Compare3plus1, diff_lo); // FFFF where 4 <= diff (3 < diff)

        auto src_prev_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x * 2 + 16));
        auto src_next_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x * 2 + 16));
        auto diffpn_hi = _mm_subs_epu16(src_prev_hi, src_next_hi);
        auto diffnp_hi = _mm_subs_epu16(src_next_hi, src_prev_hi);
        auto diff_hi = _mm_or_si128(diffpn_hi, diffnp_hi);

        auto cmp19_hi = _MM_CMPLE_EPU16(Compare19plus1, diff_hi); // FFFF where 20 <= diff (19 < diff)
        auto cmp3_hi = _MM_CMPLE_EPU16(Compare3plus1, diff_hi); // FFFF where 4 <= diff (3 < diff)

        // make bytes from wordBools
        auto cmp251 = _mm_packus_epi16(cmp3_lo, cmp3_hi);
        auto cmp235 = _mm_packus_epi16(cmp19_lo, cmp19_hi);

        // target is byte buffer!
        auto tmp1 = _mm_and_si128(cmp251, onesMask);
        auto  tmp2 = _mm_and_si128(cmp235, twosMask);
        auto  tmp = _mm_or_si128(tmp1, tmp2);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), tmp);
      }
      // rest 8 pixels
      auto src_prev_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(prvp + x * 2));
      auto src_next_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(nxtp + x * 2));
      auto diffpn_lo = _mm_subs_epu16(src_prev_lo, src_next_lo);
      auto diffnp_lo = _mm_subs_epu16(src_next_lo, src_prev_lo);
      auto diff_lo = _mm_or_si128(diffpn_lo, diffnp_lo);

      auto cmp19_lo = _MM_CMPLE_EPU16(Compare19plus1, diff_lo); // FFFF where 20 <= diff (19 < diff)
      auto cmp3_lo = _MM_CMPLE_EPU16(Compare3plus1, diff_lo); // FFFF where 4 <= diff (3 < diff)

      // make bytes from wordBools
      auto cmp251 = _mm_packus_epi16(cmp3_lo, cmp3_lo); // 8 bytes valid only
      auto cmp235 = _mm_packus_epi16(cmp19_lo, cmp19_lo);

      // target is byte buffer!
      auto tmp1 = _mm_and_si128(cmp251, onesMask);
      auto  tmp2 = _mm_and_si128(cmp235, twosMask);
      auto  tmp = _mm_or_si128(tmp1, tmp2);
      // store 8 bytes
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), tmp);

      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
}

template<typename pixel_t, bool YUY2_LumaOnly>
void check_combing_c(const pixel_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, int cthresh)
{
  // cthresh is scaled to actual bit depth
  const pixel_t* srcppp = srcp - src_pitch * 2;
  const pixel_t* srcpp = srcp - src_pitch;
  const pixel_t* srcpn = srcp + src_pitch;
  const pixel_t* srcpnn = srcp + src_pitch * 2;

  int increment;
  if constexpr (YUY2_LumaOnly)
    increment = 2;
  else
    increment = 1; // planar, YUY2 luma + chroma

  const int cthresh6 = cthresh * 6;
  // no luma masking
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; x += increment)
    {
      const int sFirst = srcp[x] - srcpp[x];
      const int sSecond = srcp[x] - srcpn[x];
      if ((sFirst > cthresh && sSecond > cthresh) || (sFirst < -cthresh && sSecond < -cthresh))
      {
        if (abs(srcppp[x] + (srcp[x] << 2) + srcpnn[x] - (3 * (srcpp[x] + srcpn[x]))) > cthresh6)
          cmkp[x] = 0xFF;
      }
    }
    srcppp += src_pitch;
    srcpp += src_pitch;
    srcp += src_pitch;
    srcpn += src_pitch;
    srcpnn += src_pitch;
    cmkp += cmk_pitch;
  }
}
// instantiate
template void check_combing_c<uint8_t, false>(const uint8_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, int cthresh);
template void check_combing_c<uint8_t, true>(const uint8_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, int cthresh);
template void check_combing_c<uint16_t, false>(const uint16_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, int cthresh);

template<typename pixel_t, bool YUY2_LumaOnly, typename safeint_t>
void check_combing_c_Metric1(const pixel_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, safeint_t cthreshsq)
{
  // cthresh is scaled to actual bit depth
  const pixel_t* srcpp = srcp - src_pitch;
  const pixel_t* srcpn = srcp + src_pitch;

  int increment;
  if constexpr (YUY2_LumaOnly)
    increment = 2;
  else
    increment = 1; // planar, YUY2 luma + chroma

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      if ((safeint_t)(srcp[x] - srcpp[x]) * (srcp[x] - srcpn[x]) > cthreshsq)
        cmkp[x] = 0xFF;
    }
    srcpp += src_pitch;
    srcp += src_pitch;
    srcpn += src_pitch;
    cmkp += cmk_pitch;
  }
}
// instantiate
template void check_combing_c_Metric1<uint8_t, false, int>(const uint8_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, int cthreshsq);
template void check_combing_c_Metric1<uint8_t, true, int>(const uint8_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, int cthreshsq);
template void check_combing_c_Metric1<uint16_t, false, int64_t>(const uint16_t* srcp, uint8_t* cmkp, int width, int height, int src_pitch, int cmk_pitch, int64_t cthreshsq);


void check_combing_AVX2(const uint8_t* srcp, uint8_t* dstp, int width,
    int height, int src_pitch, int dst_pitch, int cthresh)
{
    unsigned int cthresht = std::min(std::max(255 - cthresh - 1, 0), 255);
    auto threshb = _mm256_set1_epi8(cthresht);
    unsigned int cthresh6t = std::min(std::max(65535 - cthresh * 6 - 1, 0), 65535);
    auto thresh6w = _mm256_set1_epi16(cthresh6t);
    auto zero = _mm256_setzero_si256();
    auto all_ff = _mm256_set1_epi8(-1);

    while (height--) {
        for (int x = 0; x < width; x += 32) {
            auto prev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcp - src_pitch + x));
            auto curr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcp + x));
            auto next = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcp + src_pitch + x));
            auto diff_curr_next = _mm256_subs_epu8(curr, next);
            auto diff_next_curr = _mm256_subs_epu8(next, curr);
            auto diff_curr_prev = _mm256_subs_epu8(curr, prev);
            auto diff_prev_curr = _mm256_subs_epu8(prev, curr);
            // max(min(p-s,n-s), min(s-n,s-p))
            auto xmm2_max = _mm256_max_epu8(_mm256_min_epu8(diff_prev_curr, diff_next_curr), _mm256_min_epu8(diff_curr_next, diff_curr_prev));
            auto xmm2_cmp = _mm256_cmpeq_epi8(_mm256_adds_epu8(xmm2_max, threshb), all_ff);
            //if constexpr (with_luma_mask) { // YUY2 luma mask
            //    __m128i lumaMask = _mm_set1_epi16(0x00FF);
            //    xmm2_cmp = _mm_and_si128(xmm2_cmp, lumaMask);
            //}
            auto res_part1 = xmm2_cmp;
            if (!_mm256_test_all_zeros(xmm2_cmp, xmm2_cmp)) {
                // output2
                // compute 3*(p+n)
                auto next_lo = _mm256_unpacklo_epi8(next, zero);
                auto prev_lo = _mm256_unpacklo_epi8(prev, zero);
                auto next_hi = _mm256_unpackhi_epi8(next, zero);
                auto prev_hi = _mm256_unpackhi_epi8(prev, zero);
                auto three = _mm256_set1_epi16(3);
                auto mul_lo = _mm256_mullo_epi16(_mm256_adds_epu16(next_lo, prev_lo), three);
                auto mul_hi = _mm256_mullo_epi16(_mm256_adds_epu16(next_hi, prev_hi), three);

                // compute (pp+c*4+nn)
                auto prevprev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcp - src_pitch * 2 + x));
                auto prevprev_lo = _mm256_unpacklo_epi8(prevprev, zero);
                auto prevprev_hi = _mm256_unpackhi_epi8(prevprev, zero);
                auto curr_lo = _mm256_unpacklo_epi8(curr, zero);
                auto curr_hi = _mm256_unpackhi_epi8(curr, zero);
                auto sum2_lo = _mm256_adds_epu16(_mm256_slli_epi16(curr_lo, 2), prevprev_lo); // pp + c*4
                auto sum2_hi = _mm256_adds_epu16(_mm256_slli_epi16(curr_hi, 2), prevprev_hi); // pp + c*4

                auto nextnext = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcp + src_pitch * 2 + x));
                auto nextnext_lo = _mm256_unpacklo_epi8(nextnext, zero);
                auto nextnext_hi = _mm256_unpackhi_epi8(nextnext, zero);
                auto sum3_lo = _mm256_adds_epu16(sum2_lo, nextnext_lo);
                auto sum3_hi = _mm256_adds_epu16(sum2_hi, nextnext_hi);

                // working with sum3=(pp+c*4+nn)   and  mul=3*(p+n)
                auto diff_sum3lo_mullo = _mm256_subs_epu16(sum3_lo, mul_lo);
                auto diff_mullo_sum3lo = _mm256_subs_epu16(mul_lo, sum3_lo);
                auto diff_sum3hi_mulhi = _mm256_subs_epu16(sum3_hi, mul_hi);
                auto diff_mulhi_sum3hi = _mm256_subs_epu16(mul_hi, sum3_hi);
                // abs( (pp+c*4+nn) - mul=3*(p+n) )
                auto max_lo = _mm256_max_epi16(diff_sum3lo_mullo, diff_mullo_sum3lo);
                auto max_hi = _mm256_max_epi16(diff_sum3hi_mulhi, diff_mulhi_sum3hi);
                // abs( (pp+c*4+nn) - mul=3*(p+n) ) + thresh6w
                auto lo_thresh6w_added = _mm256_adds_epu16(max_lo, thresh6w);
                auto hi_thresh6w_added = _mm256_adds_epu16(max_hi, thresh6w);
                // maximum reached?
                auto cmp_lo = _mm256_cmpeq_epi16(lo_thresh6w_added, all_ff);
                auto cmp_hi = _mm256_cmpeq_epi16(hi_thresh6w_added, all_ff);

                auto res_lo = _mm256_srli_epi16(cmp_lo, 8);
                auto res_hi = _mm256_srli_epi16(cmp_hi, 8);
                auto res_part2 = _mm256_packus_epi16(res_lo, res_hi);

                auto res = _mm256_and_si256(res_part1, res_part2);
                _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), res);
            }
        }
        srcp += src_pitch;
        dstp += dst_pitch;
    }
    _mm256_zeroupper();
}


template<bool with_luma_mask>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif 
static void check_combing_SSE4_generic(const uint8_t *srcp, uint8_t *dstp, int width,
  int height, int src_pitch, int dst_pitch, int cthresh)
{
  unsigned int cthresht = std::min(std::max(255 - cthresh - 1, 0), 255);
  auto threshb = _mm_set1_epi8(cthresht);
  unsigned int cthresh6t = std::min(std::max(65535 - cthresh * 6 - 1, 0), 65535);
  auto thresh6w = _mm_set1_epi16(cthresh6t);
  auto zero = _mm_setzero_si128();

  __m128i all_ff = _mm_set1_epi8(-1);
  while (height--) {
    for (int x = 0; x < width; x += 16) {
      auto next = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + src_pitch + x));
      auto curr = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x));
      auto prev = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp - src_pitch + x));
      auto diff_curr_next = _mm_subs_epu8(curr, next);
      auto diff_next_curr = _mm_subs_epu8(next, curr);
      auto diff_curr_prev = _mm_subs_epu8(curr, prev);
      auto diff_prev_curr = _mm_subs_epu8(prev, curr);
      // max(min(p-s,n-s), min(s-n,s-p))
      auto xmm2_max = _mm_max_epu8(_mm_min_epu8(diff_prev_curr, diff_next_curr), _mm_min_epu8(diff_curr_next, diff_curr_prev));
      auto xmm2_cmp = _mm_cmpeq_epi8(_mm_adds_epu8(xmm2_max, threshb), all_ff);
      if constexpr (with_luma_mask) { // YUY2 luma mask
        __m128i lumaMask = _mm_set1_epi16(0x00FF);
        xmm2_cmp = _mm_and_si128(xmm2_cmp, lumaMask);
      }
      auto res_part1 = xmm2_cmp;
//      bool cmpres_is_allzero;
//#ifdef _M_X64
//      cmpres_is_allzero = (_mm_cvtsi128_si64(xmm2_cmp) | _mm_cvtsi128_si64(_mm_srli_si128(xmm2_cmp, 8))) == 0; // _si64: only at x64 platform
//#else
//      cmpres_is_allzero = (_mm_cvtsi128_si32(xmm2_cmp) |
//          _mm_cvtsi128_si32(_mm_srli_si128(xmm2_cmp, 4)) |
//          _mm_cvtsi128_si32(_mm_srli_si128(xmm2_cmp, 8)) |
//          _mm_cvtsi128_si32(_mm_srli_si128(xmm2_cmp, 12))
//          ) == 0;
//#endif
      //cmpres_is_allzero = _mm_testz_si128(xmm2_cmp, xmm2_cmp);	//needs sse4.1(Penryn or after)
      if (!_mm_test_all_zeros(xmm2_cmp, xmm2_cmp)) {
          // output2
          // compute 3*(p+n)
          auto next_lo = _mm_unpacklo_epi8(next, zero);
          auto prev_lo = _mm_unpacklo_epi8(prev, zero);
          auto next_hi = _mm_unpackhi_epi8(next, zero);
          auto prev_hi = _mm_unpackhi_epi8(prev, zero);
          __m128i three = _mm_set1_epi16(3);
          auto mul_lo = _mm_mullo_epi16(_mm_adds_epu16(next_lo, prev_lo), three);
          auto mul_hi = _mm_mullo_epi16(_mm_adds_epu16(next_hi, prev_hi), three);

          // compute (pp+c*4+nn)
          auto prevprev = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp - src_pitch * 2 + x));
          auto prevprev_lo = _mm_unpacklo_epi8(prevprev, zero);
          auto prevprev_hi = _mm_unpackhi_epi8(prevprev, zero);
          auto curr_lo = _mm_unpacklo_epi8(curr, zero);
          auto curr_hi = _mm_unpackhi_epi8(curr, zero);
          auto sum2_lo = _mm_adds_epu16(_mm_slli_epi16(curr_lo, 2), prevprev_lo); // pp + c*4
          auto sum2_hi = _mm_adds_epu16(_mm_slli_epi16(curr_hi, 2), prevprev_hi); // pp + c*4

          auto nextnext = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + src_pitch * 2 + x));
          auto nextnext_lo = _mm_unpacklo_epi8(nextnext, zero);
          auto nextnext_hi = _mm_unpackhi_epi8(nextnext, zero);
          auto sum3_lo = _mm_adds_epu16(sum2_lo, nextnext_lo);
          auto sum3_hi = _mm_adds_epu16(sum2_hi, nextnext_hi);

          // working with sum3=(pp+c*4+nn)   and  mul=3*(p+n)
          auto diff_sum3lo_mullo = _mm_subs_epu16(sum3_lo, mul_lo);
          auto diff_mullo_sum3lo = _mm_subs_epu16(mul_lo, sum3_lo);
          auto diff_sum3hi_mulhi = _mm_subs_epu16(sum3_hi, mul_hi);
          auto diff_mulhi_sum3hi = _mm_subs_epu16(mul_hi, sum3_hi);
          // abs( (pp+c*4+nn) - mul=3*(p+n) )
          auto max_lo = _mm_max_epi16(diff_sum3lo_mullo, diff_mullo_sum3lo);
          auto max_hi = _mm_max_epi16(diff_sum3hi_mulhi, diff_mulhi_sum3hi);
          // abs( (pp+c*4+nn) - mul=3*(p+n) ) + thresh6w
          auto lo_thresh6w_added = _mm_adds_epu16(max_lo, thresh6w);
          auto hi_thresh6w_added = _mm_adds_epu16(max_hi, thresh6w);
          // maximum reached?
          auto cmp_lo = _mm_cmpeq_epi16(lo_thresh6w_added, all_ff);
          auto cmp_hi = _mm_cmpeq_epi16(hi_thresh6w_added, all_ff);

          auto res_lo = _mm_srli_epi16(cmp_lo, 8);
          auto res_hi = _mm_srli_epi16(cmp_hi, 8);
          auto res_part2 = _mm_packus_epi16(res_lo, res_hi);

          auto res = _mm_and_si128(res_part1, res_part2);
          _mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), res);
        }
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}


void check_combing_SSE4(const uint8_t *srcp, uint8_t *dstp, int width, int height, int src_pitch, int dst_pitch, int cthresh)
{
  // no luma masking
  check_combing_SSE4_generic<false>(srcp, dstp, width, height, src_pitch, dst_pitch, cthresh);
}

void check_combing_YUY2LumaOnly_SSE4(const uint8_t *srcp, uint8_t *dstp, int width, int height, int src_pitch, int dst_pitch, int cthresh)
{
  // with luma masking
  check_combing_SSE4_generic<true>(srcp, dstp, width, height, src_pitch, dst_pitch, cthresh);
}


#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif 
void check_combing_uint16_SSE4(const uint16_t* srcp, uint8_t* dstp, int width, int height, int src_pitch, int dst_pitch, int cthresh)
{
  // src_pitch ok for the 16 bit pointer
/*
  const int sFirst = srcp[x] - srcpp[x];
  const int sSecond = srcp[x] - srcpn[x];
  if ((sFirst > cthresh && sSecond > cthresh) || (sFirst < -cthresh && sSecond < -cthresh))
  {
    if (abs(srcppp[x] + (srcp[x] << 2) + srcpnn[x] - (3 * (srcpp[x] + srcpn[x]))) > cthresh6)
      cmkp[x] = 0xFF;
  }
*/
  unsigned int cthresht = std::min(std::max(65535 - cthresh - 1, 0), 65535);
  auto thresh = _mm_set1_epi16(cthresht); // cmp by adds and check saturation

  auto thresh6 = _mm_set1_epi32(cthresh * 6);

  __m128i all_ff = _mm_set1_epi8(-1);
  while (height--) {
    // sets 8 mask byte by 8x uint16_t pixels
    for (int x = 0; x < width; x += 16 / sizeof(uint16_t)) {
      auto next = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + src_pitch + x));
      auto curr = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      auto prev = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp - src_pitch + x));
      auto diff_curr_next = _mm_subs_epu16(curr, next);
      auto diff_next_curr = _mm_subs_epu16(next, curr);
      auto diff_curr_prev = _mm_subs_epu16(curr, prev);
      auto diff_prev_curr = _mm_subs_epu16(prev, curr);
      // max(min(p-s,n-s), min(s-n,s-p))
      // instead of abs
      auto xmm2_max = _mm_max_epu16(_mm_min_epu16(diff_prev_curr, diff_next_curr), _mm_min_epu16(diff_curr_next, diff_curr_prev));
      auto xmm2_cmp = _mm_cmpeq_epi16(_mm_adds_epu16(xmm2_max, thresh), all_ff);

      auto res_part1 = xmm2_cmp;
      bool cmpres_is_allzero;
      cmpres_is_allzero = _mm_testz_si128(xmm2_cmp, xmm2_cmp);	//needs sse4.1(Penryn or after)
      if (!cmpres_is_allzero) {
        // output2
        auto zero = _mm_setzero_si128();
        // compute 3*(p+n)
        auto next_lo = _mm_unpacklo_epi16(next, zero);
        auto prev_lo = _mm_unpacklo_epi16(prev, zero);
        auto next_hi = _mm_unpackhi_epi16(next, zero);
        auto prev_hi = _mm_unpackhi_epi16(prev, zero);
        __m128i three = _mm_set1_epi32(3);
        auto mul_lo = _mm_mullo_epi32(_mm_add_epi32(next_lo, prev_lo), three);
        auto mul_hi = _mm_mullo_epi32(_mm_add_epi32(next_hi, prev_hi), three);

        // compute (pp+c*4+nn)
        auto prevprev = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp - src_pitch * 2 + x));
        auto prevprev_lo = _mm_unpacklo_epi16(prevprev, zero);
        auto prevprev_hi = _mm_unpackhi_epi16(prevprev, zero);
        auto curr_lo = _mm_unpacklo_epi16(curr, zero);
        auto curr_hi = _mm_unpackhi_epi16(curr, zero);
        auto sum2_lo = _mm_add_epi32(_mm_slli_epi32(curr_lo, 2), prevprev_lo); // pp + c*4
        auto sum2_hi = _mm_add_epi32(_mm_slli_epi32(curr_hi, 2), prevprev_hi); // pp + c*4

/*        if (abs(srcppp[x] + (srcp[x] << 2) + srcpnn[x] - (3 * (srcpp[x] + srcpn[x]))) > cthresh6)
          cmkp[x] = 0xFF;
          */
        auto nextnext = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + src_pitch * 2 + x));
        auto nextnext_lo = _mm_unpacklo_epi16(nextnext, zero);
        auto nextnext_hi = _mm_unpackhi_epi16(nextnext, zero);
        auto sum3_lo = _mm_add_epi32(sum2_lo, nextnext_lo);
        auto sum3_hi = _mm_add_epi32(sum2_hi, nextnext_hi);

        // working with sum3=(pp+c*4+nn)   and  mul=3*(p+n)
        auto diff_sum3lo_mullo = _mm_sub_epi32(sum3_lo, mul_lo);
        auto diff_sum3hi_mulhi = _mm_sub_epi32(sum3_hi, mul_hi);
        // abs( (pp+c*4+nn) - mul=3*(p+n) )
        auto abs_lo = _mm_abs_epi32(diff_sum3lo_mullo);
        auto abs_hi = _mm_abs_epi32(diff_sum3hi_mulhi);
        // abs( (pp+c*4+nn) - mul=3*(p+n) ) > thresh6 ??
        auto cmp_lo = _mm_cmpgt_epi32(abs_lo, thresh6);
        auto cmp_hi = _mm_cmpgt_epi32(abs_hi, thresh6);

        auto res_part2 = _mm_packs_epi32(cmp_lo, cmp_hi);

        auto res = _mm_and_si128(res_part1, res_part2);
        // mask is 8 bits
        res = _mm_packs_epi16(res, res);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), res);
      }
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}


void check_combing_SSE2_Metric1(const uint8_t *srcp, uint8_t *dstp,
  int width, int height, int src_pitch, int dst_pitch, int cthreshsq)
{
  __m128i thresh = _mm_set1_epi32(cthreshsq);
  __m128i zero = _mm_setzero_si128();
  __m128i lumaMask = _mm_set1_epi16(0x00FF);

  while (height--) {
    for (int x = 0; x < width; x += 16) {
      auto next = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + src_pitch + x));
      auto curr = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x));
      auto prev = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp - src_pitch + x));

      auto prev_lo = _mm_unpacklo_epi8(prev, zero);
      auto prev_hi = _mm_unpackhi_epi8(prev, zero);
      auto curr_lo = _mm_unpacklo_epi8(curr, zero);
      auto curr_hi = _mm_unpackhi_epi8(curr, zero);
      auto next_lo = _mm_unpacklo_epi8(next, zero);
      auto next_hi = _mm_unpackhi_epi8(next, zero);

      auto diff_prev_curr_lo = _mm_subs_epi16(prev_lo, curr_lo);
      auto diff_next_curr_lo = _mm_subs_epi16(next_lo, curr_lo);
      auto diff_prev_curr_hi = _mm_subs_epi16(prev_hi, curr_hi);
      auto diff_next_curr_hi = _mm_subs_epi16(next_hi, curr_hi);

      // -- lo
      auto diff_prev_curr_lo_lo = _mm_unpacklo_epi16(diff_prev_curr_lo, zero);
      auto diff_prev_curr_lo_hi = _mm_unpackhi_epi16(diff_prev_curr_lo, zero);
      auto diff_next_curr_lo_lo = _mm_unpacklo_epi16(diff_next_curr_lo, zero);
      auto diff_next_curr_lo_hi = _mm_unpackhi_epi16(diff_next_curr_lo, zero);

      auto res_lo_lo = _mm_madd_epi16(diff_prev_curr_lo_lo, diff_next_curr_lo_lo);
      auto res_lo_hi = _mm_madd_epi16(diff_prev_curr_lo_hi, diff_next_curr_lo_hi);

      // -- hi
      auto diff_prev_curr_hi_lo = _mm_unpacklo_epi16(diff_prev_curr_hi, zero);
      auto diff_prev_curr_hi_hi = _mm_unpackhi_epi16(diff_prev_curr_hi, zero);
      auto diff_next_curr_hi_lo = _mm_unpacklo_epi16(diff_next_curr_hi, zero);
      auto diff_next_curr_hi_hi = _mm_unpackhi_epi16(diff_next_curr_hi, zero);

      auto res_hi_lo = _mm_madd_epi16(diff_prev_curr_hi_lo, diff_next_curr_hi_lo);
      auto res_hi_hi = _mm_madd_epi16(diff_prev_curr_hi_hi, diff_next_curr_hi_hi);

      auto cmp_lo_lo = _mm_cmpgt_epi32(res_lo_lo, thresh);
      auto cmp_lo_hi = _mm_cmpgt_epi32(res_lo_hi, thresh);
      auto cmp_hi_lo = _mm_cmpgt_epi32(res_hi_lo, thresh);
      auto cmp_hi_hi = _mm_cmpgt_epi32(res_hi_hi, thresh);

      auto cmp_lo = _mm_packs_epi32(cmp_lo_lo, cmp_lo_hi);
      auto cmp_hi = _mm_packs_epi32(cmp_hi_lo, cmp_hi_hi);
      auto cmp_lo_masked = _mm_and_si128(cmp_lo, lumaMask);
      auto cmp_hi_masked = _mm_and_si128(cmp_hi, lumaMask);

      auto res = _mm_packus_epi16(cmp_lo_masked, cmp_hi_masked);
      _mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), res);
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }

}


void check_combing_SSE2_Luma_Metric1(const uint8_t *srcp, uint8_t *dstp,
  int width, int height, int src_pitch, int dst_pitch, int cthreshsq)
{
  __m128i thresh = _mm_set1_epi32(cthreshsq);
  __m128i lumaMask = _mm_set1_epi16(0x00FF);
  __m128i zero = _mm_setzero_si128();
  while (height--) {
    for (int x = 0; x < width; x += 16) {
      auto next = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + src_pitch + x));
      auto curr = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x));
      auto prev = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp - src_pitch + x));
      
      next = _mm_and_si128(next, lumaMask);
      curr = _mm_and_si128(curr, lumaMask);
      prev = _mm_and_si128(prev, lumaMask);

      auto diff_prev_curr = _mm_subs_epi16(prev, curr);
      auto diff_next_curr = _mm_subs_epi16(next, curr);

      auto diff_prev_curr_lo = _mm_unpacklo_epi16(diff_prev_curr, zero);
      auto diff_prev_curr_hi = _mm_unpackhi_epi16(diff_prev_curr, zero);
      auto diff_next_curr_lo = _mm_unpacklo_epi16(diff_next_curr, zero);
      auto diff_next_curr_hi = _mm_unpackhi_epi16(diff_next_curr, zero);

      auto res_lo = _mm_madd_epi16(diff_prev_curr_lo, diff_next_curr_lo);
      auto res_hi = _mm_madd_epi16(diff_prev_curr_hi, diff_next_curr_hi);

      auto cmp_lo = _mm_cmpgt_epi32(res_lo, thresh);
      auto cmp_hi = _mm_cmpgt_epi32(res_hi, thresh);

      auto cmp = _mm_packs_epi32(cmp_lo, cmp_hi);
      auto cmp_masked = _mm_and_si128(cmp, lumaMask);

      _mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), cmp_masked);
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

template<int blockSizeY>
void compute_sum_8xN_sse2(const uint8_t *srcp, int pitch, int &sum)
{
  // sums masks
  // if (cmkppT[x + v] == 0xFF && cmkpT[x + v] == 0xFF && cmkpnT[x + v] == 0xFF) sum++;
  // scrp is prev
  auto onesMask = _mm_set1_epi8(1);
  auto all_ff = _mm_set1_epi8(-1);
  auto prev = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp));
  auto curr = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp + pitch));
  auto summa = _mm_setzero_si128();
  srcp += pitch * 2; // points to next
  // unroll 2
  for (int i = 0; i < blockSizeY / 2; i++) { // 4x2=8
    /*
    p  #
    c  # #
    n  # #
    nn   #
    */
    auto next = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp));
    auto nextnext = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp + pitch));

    auto anded_common = _mm_and_si128(curr, next);
    auto with_prev = _mm_and_si128(prev, anded_common);
    auto with_nextnext = _mm_and_si128(anded_common, nextnext);

    // these were missing from the original assembler code (== 0xFF)
    with_prev = _mm_cmpeq_epi8(with_prev, all_ff);
    with_nextnext = _mm_cmpeq_epi8(with_nextnext, all_ff);

    with_prev = _mm_and_si128(with_prev, onesMask);
    with_nextnext = _mm_and_si128(with_nextnext, onesMask);

    prev = next;
    curr = nextnext;

    summa = _mm_adds_epu8(summa, with_prev);
    summa = _mm_adds_epu8(summa, with_nextnext);
    srcp += pitch * 2;
  }
  // now we have to sum up lower 8 bytes
  // in sse2, we use sad
  auto zero = _mm_setzero_si128();
  auto tmpsum = _mm_sad_epu8(summa, zero);  // sum(lo 8 bytes)(needed) / sum(hi 8 bytes)(not needed)
  sum = _mm_cvtsi128_si32(tmpsum);
}

// instantiate for 8x8
template void compute_sum_8xN_sse2<8>(const uint8_t* srcp, int pitch, int& sum);

// YUY2 luma only case
void compute_sum_16x8_sse2_luma(const uint8_t *srcp, int pitch, int &sum)
{
  // sums masks
  // if (cmkppT[x + v] == 0xFF && cmkpT[x + v] == 0xFF && cmkpnT[x + v] == 0xFF) sum++;
  // scrp is prev
  auto onesMask = _mm_set1_epi16(0x0001); // ones where luma
  auto all_ff = _mm_set1_epi8(-1);
  auto prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp));
  auto curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + pitch));
  auto summa = _mm_setzero_si128();
  srcp += pitch * 2; // points to next
  for (int i = 0; i < 4; i++) { // 4x2=8
    /*
    p  #
    c  # #
    n  # #
    nn   #
    */
    auto next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp));
    auto nextnext = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + pitch));

    auto anded_common = _mm_and_si128(curr, next);
    auto with_prev = _mm_and_si128(prev, anded_common);
    auto with_nextnext = _mm_and_si128(anded_common, nextnext);

    // these were missing from the original assembler code (== 0xFF)
    with_prev = _mm_cmpeq_epi8(with_prev, all_ff);
    with_nextnext = _mm_cmpeq_epi8(with_nextnext, all_ff);

    with_prev = _mm_and_si128(with_prev, onesMask);
    with_nextnext = _mm_and_si128(with_nextnext, onesMask);

    prev = next;
    curr = nextnext;

    summa = _mm_adds_epu8(summa, with_prev);
    summa = _mm_adds_epu8(summa, with_nextnext);
    srcp += pitch * 2;
  }

  // now we have to sum up lower and upper 8 bytes
  // in sse2, we use sad
  auto zero = _mm_setzero_si128();
  auto tmpsum = _mm_sad_epu8(summa, zero);  // sum(lo 8 bytes) / sum(hi 8 bytes)
  tmpsum = _mm_add_epi32(tmpsum, _mm_srli_si128(tmpsum, 8)); // lo + hi
  sum = _mm_cvtsi128_si32(tmpsum);
}

void copyFrame(PVideoFrame& dst, PVideoFrame& src, const VideoInfo& vi, IScriptEnvironment* env)
{
  const int planes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  // bit depth independent
  const int np = vi.IsYUY2() || vi.IsY() ? 1 : 3;
  for (int b = 0; b < np; ++b)
  {
    const int plane = planes[b];
    env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane),
      src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
  }
}

// fast blend routine for 50:50 case
template<typename pixel_t>
void blend_5050_SSE2(uint8_t* dstp, const uint8_t* srcp1, const uint8_t* srcp2, int width, int height, int dst_pitch, int src1_pitch, int src2_pitch)
{
  while (height--) {
    for (int x = 0; x < width * sizeof(pixel_t); x += 16) {
      auto src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp1 + x));
      auto src2 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp2 + x));
      if constexpr (sizeof(pixel_t) == 1)
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), _mm_avg_epu8(src1, src2));
      else
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), _mm_avg_epu16(src1, src2));
    }
    dstp += dst_pitch;
    srcp1 += src1_pitch;
    srcp2 += src2_pitch;
  }
}
// instantiate
template void blend_5050_SSE2<uint8_t>(uint8_t* dstp, const uint8_t* srcp1, const uint8_t* srcp2, int width, int height, int dst_pitch, int src1_pitch, int src2_pitch);
template void blend_5050_SSE2<uint16_t>(uint8_t* dstp, const uint8_t* srcp1, const uint8_t* srcp2, int width, int height, int dst_pitch, int src1_pitch, int src2_pitch);

template<typename pixel_t>
void blend_5050_c(uint8_t* dstp, const uint8_t* srcp1, const uint8_t* srcp2, int width, int height, int dst_pitch, int src1_pitch, int src2_pitch)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
      reinterpret_cast<pixel_t*>(dstp)[x] = (reinterpret_cast<const pixel_t*>(srcp1)[x] + reinterpret_cast<const pixel_t*>(srcp2)[x] + 1) >> 1;
    srcp1 += src1_pitch;
    srcp2 += src2_pitch;
    dstp += dst_pitch;
  }
}

// instantiate
template void blend_5050_c<uint8_t>(uint8_t* dstp, const uint8_t* srcp1, const uint8_t* srcp2, int width, int height, int dst_pitch, int src1_pitch, int src2_pitch);
template void blend_5050_c<uint16_t>(uint8_t* dstp, const uint8_t* srcp1, const uint8_t* srcp2, int width, int height, int dst_pitch, int src1_pitch, int src2_pitch);

// like HandleChromaCombing in TDeinterlace
// used by isCombedTIVTC as well
// mask only, no hbd needed
template<int planarType>
void do_FillCombedPlanarUpdateCmaskByUV(uint8_t* cmkp, uint8_t* cmkpU, uint8_t* cmkpV, int Width, int Height, ptrdiff_t cmk_pitch, ptrdiff_t cmk_pitchUV)
{
  // 420 only
  uint8_t* cmkpn = cmkp + cmk_pitch;
  uint8_t* cmkpp = cmkp - cmk_pitch;
  uint8_t* cmkpnn = cmkpn + cmk_pitch;

  uint8_t* cmkppU = cmkpU - cmk_pitchUV;
  uint8_t* cmkpnU = cmkpU + cmk_pitchUV;

  uint8_t* cmkppV = cmkpV - cmk_pitchUV;
  uint8_t* cmkpnV = cmkpV + cmk_pitchUV;
  for (int y = 1; y < Height - 1; ++y)
  {
    if (planarType == 420) {
      cmkp += cmk_pitch * 2;
      cmkpn += cmk_pitch * 2;
      cmkpp += cmk_pitch * 2;
      cmkpnn += cmk_pitch * 2;
    }
    else {
      cmkp += cmk_pitch;
    }
    cmkppV += cmk_pitchUV;
    cmkpV += cmk_pitchUV;
    cmkpnV += cmk_pitchUV;
    cmkppU += cmk_pitchUV;
    cmkpU += cmk_pitchUV;
    cmkpnU += cmk_pitchUV;
    for (int x = 1; x < Width - 1; ++x)
    {
      if (
        (cmkpV[x] == 0xFF &&
          (cmkpV[x - 1] == 0xFF || cmkpV[x + 1] == 0xFF ||
            cmkppV[x - 1] == 0xFF || cmkppV[x] == 0xFF || cmkppV[x + 1] == 0xFF ||
            cmkpnV[x - 1] == 0xFF || cmkpnV[x] == 0xFF || cmkpnV[x + 1] == 0xFF
            )
          ) ||
        (cmkpU[x] == 0xFF &&
          (cmkpU[x - 1] == 0xFF || cmkpU[x + 1] == 0xFF ||
            cmkppU[x - 1] == 0xFF || cmkppU[x] == 0xFF || cmkppU[x + 1] == 0xFF ||
            cmkpnU[x - 1] == 0xFF || cmkpnU[x] == 0xFF || cmkpnU[x + 1] == 0xFF
            )
          )
        )
      {
        if (planarType == 420) {
          ((uint16_t*)cmkp)[x] = (uint16_t)0xFFFF;
          ((uint16_t*)cmkpn)[x] = (uint16_t)0xFFFF;
          if (y & 1)
            ((uint16_t*)cmkpp)[x] = (uint16_t)0xFFFF;
          else
            ((uint16_t*)cmkpnn)[x] = (uint16_t)0xFFFF;
        }
        else if (planarType == 422) {
          ((uint16_t*)cmkp)[x] = (uint16_t)0xFFFF;
        }
        else if (planarType == 444) {
          cmkp[x] = 0xFF;
        }
        else if (planarType == 411) {
          ((uint32_t*)cmkp)[x] = (uint32_t)0xFFFFFFFF;
        }
      }
    }
  }
}

template void do_FillCombedPlanarUpdateCmaskByUV<411>(uint8_t* cmkp, uint8_t* cmkpU, uint8_t* cmkpV, int Width, int Height, ptrdiff_t cmk_pitch, ptrdiff_t cmk_pitchUV);
template void do_FillCombedPlanarUpdateCmaskByUV<420>(uint8_t* cmkp, uint8_t* cmkpU, uint8_t* cmkpV, int Width, int Height, ptrdiff_t cmk_pitch, ptrdiff_t cmk_pitchUV);
template void do_FillCombedPlanarUpdateCmaskByUV<422>(uint8_t* cmkp, uint8_t* cmkpU, uint8_t* cmkpV, int Width, int Height, ptrdiff_t cmk_pitch, ptrdiff_t cmk_pitchUV);
template void do_FillCombedPlanarUpdateCmaskByUV<444>(uint8_t* cmkp, uint8_t* cmkpU, uint8_t* cmkpV, int Width, int Height, ptrdiff_t cmk_pitch, ptrdiff_t cmk_pitchUV);


void mymemset(void* ptr, int v, size_t len)
{
    const size_t simdsize = 16;
    size_t loff = ((intptr_t)ptr) % simdsize;
    size_t lsimd = (len - loff) / simdsize;
    size_t lrem = len - lsimd * simdsize - loff;
    char* p = (char*)ptr;
    char  c = (char)v;

    if (loff) {
        //memset(p, c, loff);
        for (size_t i = 0; i < loff; i++) {p[i] = c;}
        p += loff;
    }

    //AVX=simdsize = 32
    //auto csimd = _mm256_set1_epi8(c); 
    //for (size_t i = 0; i < lsimd; i++) {
    //    _mm256_storeu_si256((__m256i*)p, csimd);
    //    p += simdsize;
    //}
    //_mm256_zeroupper();

    auto csimd = _mm_set1_epi8(c);
    for (size_t i = 0; i < lsimd; i++) {
        _mm_storeu_si128((__m128i*)p, csimd);
        p += simdsize;
    }

    if (lrem) {
        //memset(p, c, lrem);
        for (size_t i = 0; i < lrem; i++) { p[i] = c; }   //����������memset�ɒu����������
    }
}