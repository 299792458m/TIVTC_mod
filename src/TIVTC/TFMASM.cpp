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

#include "TFMasm.h"
#include "emmintrin.h"

#if !defined(USE_INTR) || defined(ALLOW_MMX)
__declspec(align(16)) const __int64 onesMask[2] = { 0x0101010101010101, 0x0101010101010101 };
__declspec(align(16)) const __int64 onesMaskLuma[2] = { 0x0001000100010001, 0x0001000100010001 };
__declspec(align(16)) const __int64 twosMask[2] = { 0x0202020202020202, 0x0202020202020202 };
__declspec(align(16)) const __int64 mask251[2] = { 0xFBFBFBFBFBFBFBFB, 0xFBFBFBFBFBFBFBFB };
__declspec(align(16)) const __int64 mask235[2] = { 0xEBEBEBEBEBEBEBEB, 0xEBEBEBEBEBEBEBEB };
__declspec(align(16)) const __int64 lumaMask[2] = { 0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF };
__declspec(align(16)) const __int64 threeMask[2] = { 0x0003000300030003, 0x0003000300030003 };
__declspec(align(16)) const __int64 ffMask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
#endif

void checkSceneChangeYV12_1_SSE2(const unsigned char *prvp, const unsigned char *srcp,
  int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp)
{
#ifdef USE_INTR
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
#else
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov esi, prv_pitch
    mov ebx, src_pitch
    mov edx, width
    pxor xmm7, xmm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx]
      psadbw xmm0, [edi + ecx]
      add ecx, 16
      paddd xmm7, xmm0
      cmp ecx, edx
      jl xloop
      add eax, esi
      add edi, ebx
      dec height
      jnz yloop
      mov eax, diffp
      movdqa xmm6, xmm7
      psrldq xmm7, 8
      paddd xmm6, xmm7
      movd[eax], xmm6
  }
#endif
}

#ifdef ALLOW_MMX
void checkSceneChangeYV12_1_ISSE(const unsigned char *prvp, const unsigned char *srcp,
  int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov esi, prv_pitch
    mov ebx, src_pitch
    mov edx, width
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      psadbw mm0, [edi + ecx]
      psadbw mm1, [edi + ecx + 8]
      add ecx, 16
      paddd mm6, mm0
      paddd mm7, mm1
      cmp ecx, edx
      jl xloop
      add eax, esi
      add edi, ebx
      dec height
      jnz yloop
      paddd mm6, mm7
      mov eax, diffp
      movd[eax], mm6
      emms
  }
}
#endif

#ifdef ALLOW_MMX
void checkSceneChangeYV12_1_MMX(const unsigned char *prvp, const unsigned char *srcp,
  int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov esi, prv_pitch
    mov ebx, src_pitch
    mov edx, width
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      movq mm2, [edi + ecx]
      movq mm3, [edi + ecx + 8]
      movq mm4, mm0
      movq mm5, mm1
      psubusb mm0, mm2
      psubusb mm1, mm3
      psubusb mm2, mm4
      psubusb mm3, mm5
      por mm0, mm2
      por mm1, mm3
      pxor mm4, mm4
      pxor mm5, mm5
      movq mm2, mm0
      movq mm3, mm1
      punpcklbw mm0, mm4
      punpcklbw mm1, mm5
      punpckhbw mm2, mm4
      punpckhbw mm3, mm5
      paddw mm0, mm1
      paddw mm2, mm3
      paddw mm0, mm2
      movq mm1, mm0
      punpcklwd mm0, mm4
      punpckhwd mm1, mm5
      add ecx, 16
      paddd mm6, mm0
      paddd mm7, mm1
      cmp ecx, edx
      jl xloop
      add eax, esi
      add edi, ebx
      dec height
      jnz yloop
      paddd mm6, mm7
      movq mm5, mm6
      psrlq mm6, 32
      paddd mm5, mm6
      mov eax, diffp
      movd[eax], mm5
      emms
  }
}
#endif

void checkSceneChangeYV12_2_SSE2(const unsigned char *prvp, const unsigned char *srcp,
  const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, unsigned long &diffp, unsigned long &diffn)
{
#ifdef USE_INTR
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
#else
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov ebx, nxtp
    mov edx, width
    mov esi, height
    pxor xmm6, xmm6
    pxor xmm7, xmm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx]
      movdqa xmm1, [edi + ecx]
      movdqa xmm2, [ebx + ecx]
      psadbw xmm0, xmm1
      add ecx, 16
      psadbw xmm2, xmm1
      paddd xmm6, xmm0
      paddd xmm7, xmm2
      cmp ecx, edx
      jl xloop
      add eax, prv_pitch
      add edi, src_pitch
      add ebx, nxt_pitch
      dec esi
      jnz yloop
      mov eax, diffp
      mov ebx, diffn
      movdqa xmm4, xmm6
      movdqa xmm5, xmm7
      psrldq xmm6, 8
      psrldq xmm7, 8
      paddd xmm4, xmm6
      paddd xmm5, xmm7
      movd[eax], xmm4
      movd[ebx], xmm5
  }
#endif
}

#ifdef ALLOW_MMX
void checkSceneChangeYV12_2_ISSE(const unsigned char *prvp, const unsigned char *srcp,
  const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, unsigned long &diffp, unsigned long &diffn)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov ebx, nxtp
    mov edx, width
    mov esi, height
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      movq mm2, [edi + ecx]
      movq mm3, [edi + ecx + 8]
      movq mm4, [ebx + ecx]
      movq mm5, [ebx + ecx + 8]
      psadbw mm0, mm2
      psadbw mm5, mm3
      psadbw mm4, mm2
      psadbw mm1, mm3
      paddd mm6, mm0
      paddd mm7, mm4
      add ecx, 16
      paddd mm6, mm1
      paddd mm7, mm5
      cmp ecx, edx
      jl xloop
      add eax, prv_pitch
      add edi, src_pitch
      add ebx, nxt_pitch
      dec esi
      jnz yloop
      mov eax, diffp
      mov ebx, diffn
      movd[eax], mm6
      movd[ebx], mm7
      emms
  }
}
#endif

#ifdef ALLOW_MMX
void checkSceneChangeYV12_2_MMX(const unsigned char *prvp, const unsigned char *srcp,
  const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, unsigned long &diffp, unsigned long &diffn)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov ebx, nxtp
    mov edx, width
    mov esi, height
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      movq mm2, [edi + ecx]
      movq mm3, [edi + ecx + 8]
      movq mm4, mm2
      movq mm5, mm3
      psubusb mm2, mm0
      psubusb mm3, mm1
      psubusb mm0, mm4
      psubusb mm1, mm5
      por mm0, mm2
      por mm1, mm3
      movq mm2, mm0
      pxor mm3, mm3
      punpcklbw mm0, mm3
      punpckhbw mm2, mm3
      paddw mm0, mm2
      movq mm2, mm1
      punpcklbw mm1, mm3
      punpckhbw mm2, mm3
      paddw mm1, mm2
      paddw mm0, mm1
      movq mm1, mm0
      punpcklwd mm0, mm3
      punpckhwd mm1, mm3
      paddd mm6, mm0
      paddd mm6, mm1
      movq mm0, [ebx + ecx]
      movq mm1, [ebx + ecx + 8]
      movq mm2, mm4
      movq mm3, mm5
      psubusb mm2, mm0
      psubusb mm3, mm1
      psubusb mm0, mm4
      psubusb mm1, mm5
      por mm0, mm2
      por mm1, mm3
      movq mm2, mm0
      pxor mm3, mm3
      punpcklbw mm0, mm3
      punpckhbw mm2, mm3
      paddw mm0, mm2
      movq mm2, mm1
      punpcklbw mm1, mm3
      punpckhbw mm2, mm3
      paddw mm1, mm2
      paddw mm0, mm1
      movq mm1, mm0
      punpcklwd mm0, mm3
      punpckhwd mm1, mm3
      add ecx, 16
      paddd mm7, mm0
      paddd mm7, mm1
      cmp ecx, edx
      jl xloop
      add eax, prv_pitch
      add edi, src_pitch
      add ebx, nxt_pitch
      dec esi
      jnz yloop
      mov eax, diffp
      mov ebx, diffn
      movq mm4, mm6
      movq mm5, mm7
      psrlq mm6, 32
      psrlq mm7, 32
      paddd mm4, mm6
      paddd mm5, mm7
      movd[eax], mm4
      movd[ebx], mm5
      emms
  }
}
#endif

void checkSceneChangeYUY2_1_SSE2(const unsigned char *prvp, const unsigned char *srcp,
  int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp)
{
#ifdef USE_INTR
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
#else
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov esi, prv_pitch
    mov ebx, src_pitch
    mov edx, width
    movdqa xmm5, lumaMask
    movdqa xmm6, lumaMask
    pxor xmm7, xmm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx]
      movdqa xmm1, [edi + ecx]
      pand xmm0, xmm5
      pand xmm1, xmm6
      psadbw xmm0, xmm1
      add ecx, 16
      paddd xmm7, xmm0
      cmp ecx, edx
      jl xloop
      add eax, esi
      add edi, ebx
      dec height
      jnz yloop
      mov eax, diffp
      movdqa xmm6, xmm7
      psrldq xmm7, 8
      paddd xmm6, xmm7
      movd[eax], xmm6
  }
#endif
}

#ifdef ALLOW_MMX
void checkSceneChangeYUY2_1_ISSE(const unsigned char *prvp, const unsigned char *srcp,
  int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov esi, prv_pitch
    mov ebx, src_pitch
    mov edx, width
    movq mm4, lumaMask
    movq mm5, lumaMask
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      movq mm2, [edi + ecx]
      movq mm3, [edi + ecx + 8]
      pand mm0, mm4
      pand mm1, mm5
      pand mm2, mm4
      pand mm3, mm5
      psadbw mm0, mm2
      psadbw mm1, mm3
      add ecx, 16
      paddd mm6, mm0
      paddd mm7, mm1
      cmp ecx, edx
      jl xloop
      add eax, esi
      add edi, ebx
      dec height
      jnz yloop
      paddd mm6, mm7
      mov eax, diffp
      movd[eax], mm6
      emms
  }
}
#endif

#ifdef ALLOW_MMX
void checkSceneChangeYUY2_1_MMX(const unsigned char *prvp, const unsigned char *srcp,
  int height, int width, int prv_pitch, int src_pitch, unsigned long &diffp)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov esi, prv_pitch
    mov ebx, src_pitch
    mov edx, width
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      movq mm2, [edi + ecx]
      movq mm3, [edi + ecx + 8]
      movq mm4, mm0
      movq mm5, mm1
      psubusb mm0, mm2
      psubusb mm1, mm3
      psubusb mm2, mm4
      psubusb mm3, mm5
      por mm0, mm2
      por mm1, mm3
      pand mm0, lumaMask
      pand mm1, lumaMask
      pxor mm4, mm4
      paddw mm0, mm1
      pxor mm5, mm5
      movq mm1, mm0
      punpcklwd mm0, mm4
      punpckhwd mm1, mm5
      add ecx, 16
      paddd mm6, mm0
      paddd mm7, mm1
      cmp ecx, edx
      jl xloop
      add eax, esi
      add edi, ebx
      dec height
      jnz yloop
      paddd mm6, mm7
      movq mm5, mm6
      psrlq mm6, 32
      paddd mm5, mm6
      mov eax, diffp
      movd[eax], mm5
      emms
  }
}
#endif

void checkSceneChangeYUY2_2_SSE2(const unsigned char *prvp, const unsigned char *srcp,
  const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, unsigned long &diffp, unsigned long &diffn)
{
#ifdef USE_INTR
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
#else
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov ebx, nxtp
    mov edx, width
    mov esi, height
    movdqa xmm4, lumaMask
    movdqa xmm5, lumaMask
    pxor xmm6, xmm6
    pxor xmm7, xmm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx]
      movdqa xmm1, [edi + ecx]
      movdqa xmm2, [ebx + ecx]
      pand xmm0, xmm4
      pand xmm1, xmm5
      pand xmm2, xmm4
      psadbw xmm0, xmm1
      add ecx, 16
      psadbw xmm2, xmm1
      paddd xmm6, xmm0
      paddd xmm7, xmm2
      cmp ecx, edx
      jl xloop
      add eax, prv_pitch
      add edi, src_pitch
      add ebx, nxt_pitch
      dec esi
      jnz yloop
      mov eax, diffp
      mov ebx, diffn
      movdqa xmm4, xmm6
      movdqa xmm5, xmm7
      psrldq xmm6, 8
      psrldq xmm7, 8
      paddd xmm4, xmm6
      paddd xmm5, xmm7
      movd[eax], xmm4
      movd[ebx], xmm5
  }
#endif
}

#ifdef ALLOW_MMX
void checkSceneChangeYUY2_2_ISSE(const unsigned char *prvp, const unsigned char *srcp,
  const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, unsigned long &diffp, unsigned long &diffn)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov ebx, nxtp
    mov edx, width
    mov esi, height
    movq mm4, lumaMask
    movq mm5, lumaMask
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      movq mm2, [edi + ecx]
      movq mm3, [edi + ecx + 8]
      pand mm0, mm4
      pand mm1, mm5
      pand mm2, mm4
      pand mm3, mm5
      psadbw mm0, mm2
      psadbw mm1, mm3
      paddd mm6, mm0
      paddd mm6, mm1
      movq mm0, [ebx + ecx]
      movq mm1, [ebx + ecx + 8]
      pand mm0, mm4
      pand mm1, mm5
      psadbw mm0, mm2
      psadbw mm1, mm3
      add ecx, 16
      paddd mm7, mm0
      paddd mm7, mm1
      cmp ecx, edx
      jl xloop
      add eax, prv_pitch
      add edi, src_pitch
      add ebx, nxt_pitch
      dec esi
      jnz yloop
      mov eax, diffp
      mov ebx, diffn
      movd[eax], mm6
      movd[ebx], mm7
      emms
  }
}
#endif

#ifdef ALLOW_MMX
void checkSceneChangeYUY2_2_MMX(const unsigned char *prvp, const unsigned char *srcp,
  const unsigned char *nxtp, int height, int width, int prv_pitch, int src_pitch,
  int nxt_pitch, unsigned long &diffp, unsigned long &diffn)
{
  __asm
  {
    mov eax, prvp
    mov edi, srcp
    mov ebx, nxtp
    mov edx, width
    mov esi, height
    pxor mm6, mm6
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [eax + ecx + 8]
      movq mm2, [edi + ecx]
      movq mm3, [edi + ecx + 8]
      movq mm4, mm2
      movq mm5, mm3
      psubusb mm2, mm0
      psubusb mm3, mm1
      psubusb mm0, mm4
      psubusb mm1, mm5
      por mm2, mm0
      por mm3, mm1
      pand mm2, lumaMask
      pand mm3, lumaMask
      paddw mm2, mm3
      pxor mm0, mm0
      movq mm3, mm2
      pxor mm1, mm1
      punpcklwd mm2, mm0
      punpckhwd mm3, mm1
      paddd mm6, mm2
      paddd mm6, mm3
      movq mm0, [ebx + ecx]
      movq mm1, [ebx + ecx + 8]
      movq mm2, mm4
      movq mm3, mm5
      psubusb mm4, mm0
      psubusb mm5, mm1
      psubusb mm0, mm2
      psubusb mm1, mm3
      por mm0, mm4
      por mm1, mm5
      pand mm0, lumaMask
      pand mm1, lumaMask
      paddw mm0, mm1
      pxor mm2, mm2
      movq mm1, mm0
      pxor mm3, mm3
      punpcklwd mm0, mm2
      punpckhwd mm1, mm3
      add ecx, 16
      paddd mm7, mm0
      paddd mm7, mm1
      cmp ecx, edx
      jl xloop
      add eax, prv_pitch
      add edi, src_pitch
      add ebx, nxt_pitch
      dec esi
      jnz yloop
      mov eax, diffp
      mov ebx, diffn
      movq mm4, mm6
      movq mm5, mm7
      psrlq mm6, 32
      psrlq mm7, 32
      paddd mm4, mm6
      paddd mm5, mm7
      movd[eax], mm4
      movd[ebx], mm5
      emms
  }
}
#endif

// aligned. 
// different path if not mod16, but only for remaining 8 bytes
void buildABSDiffMask_SSE2(const unsigned char *prvp, const unsigned char *nxtp,
  unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width,
  int height)
{
#ifdef USE_INTR
	if (!(width & 15))
	  {
		while (height--) {
		 for (int x = 0; x < width; x += 16)
		  {
			__m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
			__m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i *>(nxtp + x));
			__m128i diffpn = _mm_subs_epu8(src_prev, src_next);	//負だと0
			__m128i diffnp = _mm_subs_epu8(src_next, src_prev);	//負だと0
			__m128i diff = _mm_or_si128(diffpn, diffnp);		//正の結果だけ残る
			_mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), diff);
		  }
		  prvp += prv_pitch;
		  nxtp += nxt_pitch;
		  dstp += dst_pitch;
		}
	  }
	  else {
		width -= 8; // last chunk is 8 bytes instead of 16
		while (height--) {
		  int x;
		  for (x = 0; x < width; x += 16)
		  {
			__m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
			__m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i *>(nxtp + x));
			__m128i diffpn = _mm_subs_epu8(src_prev, src_next);
			__m128i diffnp = _mm_subs_epu8(src_next, src_prev);
			__m128i diff = _mm_or_si128(diffpn, diffnp);
			_mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), diff);
		  }
		  __m128i src_prev = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(prvp + x));
		  __m128i src_next = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(nxtp + x));
		  __m128i diffpn = _mm_subs_epu8(src_prev, src_next);
		  __m128i diffnp = _mm_subs_epu8(src_next, src_prev);
		  __m128i diff = _mm_or_si128(diffpn, diffnp);
		  _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp + x), diff);
		  prvp += prv_pitch;
		  nxtp += nxt_pitch;
		  dstp += dst_pitch;
		}
	  }
#else
  if (!(width & 15))
  {
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      yloop :
      xor ecx, ecx
        align 16
        xloop :
        movdqa xmm0, [eax + ecx]
        movdqa xmm1, [ebx + ecx]
        movdqa xmm2, xmm0
        psubusb xmm0, xmm1
        psubusb xmm1, xmm2
        por xmm0, xmm1
        movdqa[edx + ecx], xmm0
        add ecx, 16
        cmp ecx, edi
        jl xloop
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop
    }
  }
  else
  {
    width -= 8;
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      yloop2 :
      xor ecx, ecx
        align 16
        xloop2 :
        movdqa xmm0, [eax + ecx]
        movdqa xmm1, [ebx + ecx]
        movdqa xmm2, xmm0
        psubusb xmm0, xmm1
        psubusb xmm1, xmm2
        por xmm0, xmm1
        movdqa[edx + ecx], xmm0
        add ecx, 16
        cmp ecx, edi
        jl xloop2
        movq xmm0, qword ptr[eax + ecx]
        movq xmm1, qword ptr[ebx + ecx]
        movq xmm2, xmm0
        psubusb xmm0, xmm1
        psubusb xmm1, xmm2
        por xmm0, xmm1
        movq qword ptr[edx + ecx], xmm0
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop2
    }
  }
#endif
}

void buildABSDiffMask_AVX2(const unsigned char *prvp, const unsigned char *nxtp,
	unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width,
	int height)
{
	while (height--) {
		for (int x = 0; x < width; x += 32)
		{
			auto src_prev = _mm256_load_si256(reinterpret_cast<const __m256i *>(prvp + x));
			auto src_next = _mm256_load_si256(reinterpret_cast<const __m256i *>(nxtp + x));
			//auto diffpn = _mm256_subs_epu8(src_prev, src_next);
			//auto diffnp = _mm256_subs_epu8(src_next, src_prev);
			//auto diff   = _mm256_or_si256(diffpn, diffnp);
			auto diffmax = _mm256_max_epu8(src_prev, src_next);	//こんなのもあり
			auto diffmin = _mm256_min_epu8(src_prev, src_next);
			auto diff = _mm256_subs_epu8(diffmax, diffmin);
			_mm256_stream_si256(reinterpret_cast<__m256i *>(dstp + x), diff);
		}
		prvp += prv_pitch;
		nxtp += nxt_pitch;
		dstp += dst_pitch;
	}
	_mm256_zeroupper();
}
#ifdef ALLOW_MMX
void buildABSDiffMask_MMX(const unsigned char *prvp, const unsigned char *nxtp,
  unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width,
  int height)
{
  if (!(width & 15))
  {
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      yloop :
      xor ecx, ecx
        align 16
        xloop :
        movq mm0, [eax + ecx]
        movq mm1, [eax + ecx + 8]
        movq mm2, [ebx + ecx]
        movq mm3, [ebx + ecx + 8]
        movq mm4, mm0
        movq mm5, mm1
        psubusb mm4, mm2
        psubusb mm5, mm3
        psubusb mm2, mm0
        psubusb mm3, mm1
        por mm2, mm4
        por mm3, mm5
        movq[edx + ecx], mm2
        movq[edx + ecx + 8], mm3
        add ecx, 16
        cmp ecx, edi
        jl xloop
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop
        emms
    }
  }
  else
  {
    width -= 8;
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      yloop2 :
      xor ecx, ecx
        align 16
        xloop2 :
        movq mm0, [eax + ecx]
        movq mm1, [eax + ecx + 8]
        movq mm2, [ebx + ecx]
        movq mm3, [ebx + ecx + 8]
        movq mm4, mm0
        movq mm5, mm1
        psubusb mm4, mm2
        psubusb mm5, mm3
        psubusb mm2, mm0
        psubusb mm3, mm1
        por mm2, mm4
        por mm3, mm5
        movq[edx + ecx], mm2
        movq[edx + ecx + 8], mm3
        add ecx, 16
        cmp ecx, edi
        jl xloop2
        movq mm0, [eax + ecx]
        movq mm1, [ebx + ecx]
        movq mm2, mm0
        psubusb mm2, mm1
        psubusb mm1, mm0
        por mm1, mm2
        movq[edx + ecx], mm1
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop2
        emms
    }
  }
}
#endif

void buildABSDiffMask2_SSE2(const unsigned char *prvp, const unsigned char *nxtp,
  unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width,
  int height)
{
#ifdef USE_INTR
  __m128i onesMask = _mm_set1_epi8(1);
  __m128i twosMask = _mm_set1_epi8(2);
  __m128i all_ff = _mm_set1_epi8(-1);
  __m128i mask251 = _mm_set1_epi8(-5); // 1111 1011
  __m128i mask235 = _mm_set1_epi8(-21); // 1110 1011

  if (!(width & 15))
  {
    while (height--) {
      for (int x = 0; x < width; x += 16)
      {
        __m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
        __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i *>(nxtp + x));
        __m128i diffpn = _mm_subs_epu8(src_prev, src_next);
        __m128i diffnp = _mm_subs_epu8(src_next, src_prev);
        __m128i diff = _mm_or_si128(diffpn, diffnp);
        __m128i added251 = _mm_adds_epu8(diff, mask251);
        __m128i added235 = _mm_adds_epu8(diff, mask235);
        __m128i cmp251 = _mm_cmpeq_epi8(added251, all_ff);
        __m128i cmp235 = _mm_cmpeq_epi8(added235, all_ff);
        __m128i tmp1 = _mm_and_si128(cmp251, onesMask);
        __m128i tmp2 = _mm_and_si128(cmp235, twosMask);
        __m128i tmp = _mm_or_si128(tmp1, tmp2);
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), tmp);
      }
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
  else {
    width -= 8; // last chunk is 8 bytes instead of 16
    while (height--) {
      int x;
      for (x = 0; x < width; x += 16)
      {
        __m128i src_prev = _mm_load_si128(reinterpret_cast<const __m128i *>(prvp + x));
        __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i *>(nxtp + x));
        __m128i diffpn = _mm_subs_epu8(src_prev, src_next);
        __m128i diffnp = _mm_subs_epu8(src_next, src_prev);
        __m128i diff = _mm_or_si128(diffpn, diffnp);
        __m128i added251 = _mm_adds_epu8(diff, mask251);
        __m128i added235 = _mm_adds_epu8(diff, mask235);
        __m128i cmp251 = _mm_cmpeq_epi8(added251, all_ff);
        __m128i cmp235 = _mm_cmpeq_epi8(added235, all_ff);
        __m128i tmp1 = _mm_and_si128(cmp251, onesMask);
        __m128i tmp2 = _mm_and_si128(cmp235, twosMask);
        __m128i tmp = _mm_or_si128(tmp1, tmp2);
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), tmp);
      }
		  __m128i src_prev = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(prvp + x));
		  __m128i src_next = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(nxtp + x));
		  __m128i diffpn = _mm_subs_epu8(src_prev, src_next);
		  __m128i diffnp = _mm_subs_epu8(src_next, src_prev);
		  __m128i diff = _mm_or_si128(diffpn, diffnp);
		  __m128i added251 = _mm_adds_epu8(diff, mask251);
		  __m128i added235 = _mm_adds_epu8(diff, mask235);
		  __m128i cmp251 = _mm_cmpeq_epi8(added251, all_ff);
		  __m128i cmp235 = _mm_cmpeq_epi8(added235, all_ff);
		  __m128i tmp1 = _mm_and_si128(cmp251, onesMask);
		  __m128i tmp2 = _mm_and_si128(cmp235, twosMask);
		  __m128i tmp = _mm_or_si128(tmp1, tmp2);
		  _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp + x), tmp);
      prvp += prv_pitch;
      nxtp += nxt_pitch;
      dstp += dst_pitch;
    }
  }
#else
  if (!(width & 15))
  {
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      movdqa xmm3, onesMask
      movdqa xmm4, twosMask
      pcmpeqb xmm5, xmm5
      movdqa xmm6, mask251
      movdqa xmm7, mask235
      yloop :
      xor ecx, ecx
        align 16
        xloop :
        movdqa xmm0, [eax + ecx]
        movdqa xmm1, [ebx + ecx]
        movdqa xmm2, xmm0
        psubusb xmm0, xmm1
        psubusb xmm1, xmm2
        por xmm0, xmm1
        movdqa xmm1, xmm0
        paddusb xmm0, xmm6
        paddusb xmm1, xmm7
        pcmpeqb xmm0, xmm5
        pcmpeqb xmm1, xmm5
        pand xmm0, xmm3
        pand xmm1, xmm4
        por xmm0, xmm1
        movdqa[edx + ecx], xmm0
        add ecx, 16
        cmp ecx, edi
        jl xloop
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop
    }
  }
  else
  {
    width -= 8;
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      movdqa xmm3, onesMask
      movdqa xmm4, twosMask
      pcmpeqb xmm5, xmm5
      movdqa xmm6, mask251
      movdqa xmm7, mask235
      yloop2 :
      xor ecx, ecx
        align 16
        xloop2 :
        movdqa xmm0, [eax + ecx]
        movdqa xmm1, [ebx + ecx]
        movdqa xmm2, xmm0
        psubusb xmm0, xmm1
        psubusb xmm1, xmm2
        por xmm0, xmm1
        movdqa xmm1, xmm0
        paddusb xmm0, xmm6
        paddusb xmm1, xmm7
        pcmpeqb xmm0, xmm5
        pcmpeqb xmm1, xmm5
        pand xmm0, xmm3
        pand xmm1, xmm4
        por xmm0, xmm1
        movdqa[edx + ecx], xmm0
        add ecx, 16
        cmp ecx, edi
        jl xloop2
        movq xmm0, qword ptr[eax + ecx]
        movq xmm1, qword ptr[ebx + ecx]
        movq xmm2, xmm0
        psubusb xmm0, xmm1
        psubusb xmm1, xmm2
        por xmm0, xmm1
        movdqa xmm1, xmm0
        paddusb xmm0, xmm6
        paddusb xmm1, xmm7
        pcmpeqb xmm0, xmm5
        pcmpeqb xmm1, xmm5
        pand xmm0, xmm3
        pand xmm1, xmm4
        por xmm0, xmm1
        movq qword ptr[edx + ecx], xmm0
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop2
    }
  }
#endif
}

#ifdef ALLOW_MMX
void buildABSDiffMask2_MMX(const unsigned char *prvp, const unsigned char *nxtp,
  unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int width,
  int height)
{
  if (!(width & 15))
  {
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      movq mm6, mask251
      movq mm7, mask235
      yloop :
      xor ecx, ecx
        align 16
        xloop :
        movq mm0, [eax + ecx]
        movq mm1, [eax + ecx + 8]
        movq mm2, [ebx + ecx]
        movq mm3, [ebx + ecx + 8]
        movq mm4, mm0
        movq mm5, mm1
        psubusb mm4, mm2
        psubusb mm5, mm3
        psubusb mm2, mm0
        psubusb mm3, mm1
        por mm2, mm4
        por mm3, mm5
        pcmpeqb mm0, mm0
        pcmpeqb mm1, mm1
        movq mm4, mm2
        movq mm5, mm3
        paddusb mm4, mm6
        paddusb mm2, mm7
        paddusb mm5, mm6
        paddusb mm3, mm7
        pcmpeqb mm4, mm0
        pcmpeqb mm2, mm1
        pcmpeqb mm5, mm0
        pcmpeqb mm3, mm1
        pand mm4, onesMask
        pand mm5, onesMask
        pand mm2, twosMask
        pand mm3, twosMask
        por mm2, mm4
        por mm3, mm5
        movq[edx + ecx], mm2
        movq[edx + ecx + 8], mm3
        add ecx, 16
        cmp ecx, edi
        jl xloop
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop
        emms
    }
  }
  else
  {
    width -= 8;
    __asm
    {
      mov eax, prvp
      mov ebx, nxtp
      mov edx, dstp
      mov edi, width
      mov esi, height
      movq mm6, mask251
      movq mm7, mask235
      yloop2 :
      xor ecx, ecx
        align 16
        xloop2 :
        movq mm0, [eax + ecx]
        movq mm1, [eax + ecx + 8]
        movq mm2, [ebx + ecx]
        movq mm3, [ebx + ecx + 8]
        movq mm4, mm0
        movq mm5, mm1
        psubusb mm4, mm2
        psubusb mm5, mm3
        psubusb mm2, mm0
        psubusb mm3, mm1
        por mm2, mm4
        por mm3, mm5
        pcmpeqb mm0, mm0
        pcmpeqb mm1, mm1
        movq mm4, mm2
        movq mm5, mm3
        paddusb mm4, mm6
        paddusb mm2, mm7
        paddusb mm5, mm6
        paddusb mm3, mm7
        pcmpeqb mm4, mm0
        pcmpeqb mm2, mm1
        pcmpeqb mm5, mm0
        pcmpeqb mm3, mm1
        pand mm4, onesMask
        pand mm5, onesMask
        pand mm2, twosMask
        pand mm3, twosMask
        por mm2, mm4
        por mm3, mm5
        movq[edx + ecx], mm2
        movq[edx + ecx + 8], mm3
        add ecx, 16
        cmp ecx, edi
        jl xloop2
        movq mm0, [eax + ecx]
        movq mm1, [ebx + ecx]
        movq mm2, mm0
        psubusb mm2, mm1
        psubusb mm1, mm0
        por mm1, mm2
        pcmpeqb mm0, mm0
        movq mm2, mm1
        paddusb mm1, mm6
        paddusb mm2, mm7
        pcmpeqb mm1, mm0
        pcmpeqb mm2, mm0
        pand mm1, onesMask
        pand mm2, twosMask
        por mm1, mm2
        movq[edx + ecx], mm1
        add eax, prv_pitch
        add ebx, nxt_pitch
        add edx, dst_pitch
        dec esi
        jnz yloop2
        emms
    }
  }
}
#endif

template<bool aligned, bool with_luma_mask>
static void check_combing_SSE41_generic_simd(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb, __m128i thresh6w)
{
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
      if (with_luma_mask) {
        __m128i lumaMask = _mm_set1_epi16(0x00FF);
        xmm2_cmp = _mm_and_si128(xmm2_cmp, lumaMask);
      }
      auto res_part1 = xmm2_cmp;
      bool cmpres_is_allzero;

	  cmpres_is_allzero = _mm_testz_si128(xmm2_cmp, xmm2_cmp);	//needs sse4.1(Penryn or after)

		if (!cmpres_is_allzero) {
			// output2
			auto zero = _mm_setzero_si128();
			// compute 3*(p+n)
			auto next_lo = _mm_unpacklo_epi8(next, zero);
			auto prev_lo = _mm_unpacklo_epi8(prev, zero);
			auto next_hi = _mm_unpackhi_epi8(next, zero);
			auto prev_hi = _mm_unpackhi_epi8(prev, zero);
			__m128i threeMask = _mm_set1_epi16(3);
			auto mul_lo = _mm_mullo_epi16(_mm_adds_epu16(next_lo, prev_lo), threeMask);
			auto mul_hi = _mm_mullo_epi16(_mm_adds_epu16(next_hi, prev_hi), threeMask);

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


// instantiate
template void check_combing_SSE2<false>(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb, __m128i thresh6w);
template void check_combing_SSE2<true>(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb, __m128i thresh6w);

// todo: this one needs an unaligned version, too
// src_pitch2: src_pitch*2 for inline asm speed reasons
template<bool aligned>
void check_combing_SSE2(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb, __m128i thresh6w)
{
#ifdef USE_INTR
  // no luma masking
  check_combing_SSE41_generic_simd<aligned, false>(srcp, dstp, width, height, src_pitch, src_pitch2, dst_pitch, threshb, thresh6w);
#else
  __asm
  {
    mov eax, srcp
    mov edx, dstp
    mov edi, src_pitch
    add eax, edi
    movdqa xmm6, threshb
    pcmpeqb xmm7, xmm7 // Full FF
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx]	// next
      sub eax, edi
      movdqa xmm4, [eax + ecx]	// srcp
      movdqa xmm5, xmm0		// cpy next
      sub eax, edi
      movdqa xmm1, [eax + ecx]	// prev
      movdqa xmm2, xmm4		// cpy srcp
      movdqa xmm3, xmm1		// cpy prev
      psubusb xmm5, xmm2		// next-srcp
      psubusb xmm3, xmm4		// prev-srcp
      psubusb xmm2, xmm0		// srcp-next
      psubusb xmm4, xmm1		// srcp-prev 
      pminub xmm3, xmm5
      pminub xmm2, xmm4
      pmaxub xmm2, xmm3

      paddusb xmm2, xmm6
      pcmpeqb xmm2, xmm7

      movdqa xmm3, xmm2
      psrldq xmm2, 4
      movd edi, xmm3
      movd esi, xmm2
      or edi, esi
      jnz output2

      movdqa xmm4, xmm2
      psrldq xmm2, 4
      psrldq xmm4, 8
      movd edi, xmm2
      movd esi, xmm4
      or edi, esi
      jnz output2

      mov edi, src_pitch
      add ecx, 16
      lea eax, [eax + edi * 2]
      cmp ecx, width
      jl xloop

      add eax, edi
      add edx, dst_pitch
      dec height
      jnz yloop

      jmp end

      output2 :
    mov esi, src_pitch2
      mov edi, src_pitch
      pxor xmm7, xmm7
      movdqa xmm2, xmm0
      movdqa xmm4, xmm1
      punpcklbw xmm2, xmm7
      punpckhbw xmm0, xmm7
      sub eax, edi
      punpcklbw xmm4, xmm7
      punpckhbw xmm1, xmm7
      paddusw xmm2, xmm4
      paddusw xmm0, xmm1
      pmullw xmm2, threeMask	// 3*(p+n)
      pmullw xmm0, threeMask	// 3*(p+n)
      movdqa xmm1, [eax + ecx]
      movdqa xmm4, xmm1
      punpcklbw xmm4, xmm7
      add eax, esi
      punpckhbw xmm1, xmm7
      movdqa xmm5, [eax + ecx]
      movdqa xmm6, xmm5
      punpcklbw xmm6, xmm7
      punpckhbw xmm5, xmm7
      psllw xmm6, 2
      add eax, esi
      psllw xmm5, 2
      paddusw xmm4, xmm6
      paddusw xmm1, xmm5
      movdqa xmm5, [eax + ecx]
      movdqa xmm6, xmm5
      punpcklbw xmm6, xmm7  // PF fix 170418: was: punpcklbw xmm6, mm7
      punpckhbw xmm5, xmm7  // PF fix 170418: was: punpckhbw xmm5, mm7
      paddusw xmm4, xmm6			// (pp+c*4+nn)
      paddusw xmm1, xmm5			// (pp+c*4+nn)
      movdqa xmm6, xmm4
      movdqa xmm5, xmm1
      psubusw xmm6, xmm2
      psubusw xmm5, xmm0
      psubusw xmm2, xmm4
      psubusw xmm0, xmm1
      pcmpeqb xmm7, xmm7
      pmaxsw xmm2, xmm6
      pmaxsw xmm0, xmm5
      paddusw xmm2, thresh6w
      paddusw xmm0, thresh6w
      pcmpeqw xmm2, xmm7
      pcmpeqw xmm0, xmm7
      sub eax, edi
      psrlw xmm2, 8
      movdqa xmm6, threshb
      psrlw xmm0, 8
      packuswb xmm2, xmm0
      pand xmm3, xmm2
      movdqa[edx + ecx], xmm3
      add ecx, 16
      cmp ecx, width
      jl xloop
      
      add eax, edi
      add edx, dst_pitch

      dec height
      jnz yloop
      end :
  }
#endif
}

#ifdef ALLOW_MMX
void check_combing_iSSE(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb, __int64 thresh6w)
{
  __asm
  {
    mov eax, srcp
    mov ebx, dstp
    mov edx, width
    mov edi, src_pitch
    add eax, edi
    movq mm6, threshb
    pcmpeqb mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]	// next
      sub eax, edi
      movq mm4, [eax + ecx]	// srcp
      movq mm5, mm0		// cpy next
      sub eax, edi
      movq mm1, [eax + ecx]	// prev
      movq mm2, mm4		// cpy srcp
      movq mm3, mm1		// cpy prev
      psubusb mm5, mm2		// next-srcp
      psubusb mm3, mm4		// prev-srcp
      psubusb mm2, mm0		// srcp-next
      psubusb mm4, mm1		// srcp-prev 
      pminub mm3, mm5
      pminub mm2, mm4
      pmaxub mm2, mm3
      paddusb mm2, mm6
      pcmpeqb mm2, mm7
      movq mm3, mm2
      psrlq mm2, 32
      movd edi, mm3
      movd esi, mm2
      or edi, esi
      jnz output2
      mov edi, src_pitch
      add ecx, 8
      lea eax, [eax + edi * 2]
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      jmp end
      output2 :
    mov esi, src_pitch2
      mov edi, src_pitch
      pxor mm7, mm7
      movq mm2, mm0
      movq mm4, mm1
      punpcklbw mm2, mm7
      punpckhbw mm0, mm7
      sub eax, edi
      punpcklbw mm4, mm7
      punpckhbw mm1, mm7
      paddusw mm2, mm4
      paddusw mm0, mm1
      pmullw mm2, threeMask	// 3*(p+n)
      pmullw mm0, threeMask	// 3*(p+n)
      movq mm1, [eax + ecx]
      movq mm4, mm1
      punpcklbw mm4, mm7
      add eax, esi
      punpckhbw mm1, mm7
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      psllw mm6, 2
      add eax, esi
      psllw mm5, 2
      paddusw mm4, mm6
      paddusw mm1, mm5
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      paddusw mm4, mm6			// (pp+c*4+nn)
      paddusw mm1, mm5			// (pp+c*4+nn)
      movq mm6, mm4
      movq mm5, mm1
      psubusw mm6, mm2
      psubusw mm5, mm0
      psubusw mm2, mm4
      psubusw mm0, mm1
      pcmpeqb mm7, mm7
      pmaxsw mm2, mm6
      pmaxsw mm0, mm5
      paddusw mm2, thresh6w
      paddusw mm0, thresh6w
      pcmpeqw mm2, mm7
      pcmpeqw mm0, mm7
      sub eax, edi
      psrlw mm2, 8
      movq mm6, threshb
      psrlw mm0, 8
      packuswb mm2, mm0
      pand mm3, mm2
      movq[ebx + ecx], mm3
      add ecx, 8
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      end :
    emms
  }
}
#endif

#ifdef ALLOW_MMX
void check_combing_MMX(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb, __int64 thresh6w)
{
  __asm
  {
    mov eax, srcp
    mov ebx, dstp
    mov edx, width
    mov edi, src_pitch
    add eax, edi
    movq mm6, threshb
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]	// next
      sub eax, edi
      movq mm4, [eax + ecx]	// srcp
      movq mm5, mm0		// cpy next
      sub eax, edi
      movq mm1, [eax + ecx]	// prev
      movq mm2, mm4		// cpy srcp
      movq mm3, mm1		// cpy prev
      psubusb mm5, mm2		// next-srcp
      psubusb mm3, mm4		// prev-srcp
      psubusb mm2, mm0		// srcp-next
      psubusb mm4, mm1		// srcp-prev
      //pminub mm3,mm5
      //pminub mm2,mm4
      movq mm0, mm3
      movq mm1, mm2
      psubusb mm0, mm5
      psubusb mm1, mm4
      pcmpeqb mm0, mm7
      pcmpeqb mm1, mm7
      pand mm3, mm0
      pand mm2, mm1
      pxor mm0, ffMask
      pxor mm1, ffMask
      pand mm5, mm0
      pand mm4, mm1
      por mm3, mm5			// min(mm3,mm5)
      por mm2, mm4			// min(mm2,mm4)
      //pmaxub mm2,mm3
      movq mm0, mm2
      psubusb mm0, mm3
      pcmpeqb mm0, mm7
      pand mm3, mm0
      pxor mm0, ffMask
      pand mm2, mm0
      por mm2, mm3			// max(mm2,mm3)
      paddusb mm2, mm6
      pcmpeqb mm2, ffMask
      movq mm3, mm2
      psrlq mm2, 32
      movd edi, mm3
      movd esi, mm2
      or edi, esi
      jnz output2
      mov edi, src_pitch
      add ecx, 8
      lea eax, [eax + edi * 2]
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      jmp end
      output2 :
    movq mm1, [eax + ecx]	// prev
      mov esi, src_pitch2
      mov edi, src_pitch
      add eax, esi
      movq mm0, [eax + ecx]	// next
      movq mm2, mm0
      movq mm4, mm1
      sub eax, esi
      punpcklbw mm2, mm7
      punpckhbw mm0, mm7
      sub eax, edi
      punpcklbw mm4, mm7
      punpckhbw mm1, mm7
      paddusw mm2, mm4
      paddusw mm0, mm1
      pmullw mm2, threeMask	// 3*(p+n)
      pmullw mm0, threeMask	// 3*(p+n)
      movq mm1, [eax + ecx]
      movq mm4, mm1
      punpcklbw mm4, mm7
      add eax, esi
      punpckhbw mm1, mm7
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      psllw mm6, 2
      add eax, esi
      psllw mm5, 2
      paddusw mm4, mm6
      paddusw mm1, mm5
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      paddusw mm4, mm6			// (pp+c*4+nn)
      paddusw mm1, mm5			// (pp+c*4+nn)
      movq mm6, mm4
      movq mm5, mm1
      psubusw mm6, mm2
      psubusw mm5, mm0
      psubusw mm2, mm4
      psubusw mm0, mm1
      //pmaxsw mm2,mm6
      //pmaxsw mm0,mm5
      movq mm1, mm2
      movq mm4, mm0
      pcmpgtw mm1, mm6
      pcmpgtw mm4, mm5
      pand mm2, mm1
      pand mm0, mm4
      pxor mm1, ffMask
      pxor mm4, ffMask
      pand mm6, mm1
      pand mm5, mm4
      por mm2, mm6				// max(mm2,mm6)
      por mm0, mm5				// max(mm0,mm5)
      paddusw mm2, thresh6w
      paddusw mm0, thresh6w
      pcmpeqw mm2, ffMask
      pcmpeqw mm0, ffMask
      sub eax, edi
      psrlw mm2, 8
      movq mm6, threshb
      psrlw mm0, 8
      packuswb mm2, mm0
      pand mm3, mm2
      movq[ebx + ecx], mm3
      add ecx, 8
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      end :
    emms
  }
}
#endif

template<bool aligned>
void check_combing_SSE2_Luma(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb, __m128i thresh6w)
{
#ifdef USE_INTR
  // with luma masking
  check_combing_SSE41_generic_simd<aligned, true>(srcp, dstp, width, height, src_pitch, src_pitch2, dst_pitch, threshb, thresh6w);
#else
  __asm
  {
    mov eax, srcp
    mov edx, dstp
    mov edi, src_pitch
    add eax, edi
    movdqa xmm6, threshb
    pcmpeqb xmm7, xmm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx]	// next
      sub eax, edi
      movdqa xmm4, [eax + ecx]	// srcp
      movdqa xmm5, xmm0		// cpy next
      sub eax, edi
      movdqa xmm1, [eax + ecx]	// prev
      movdqa xmm2, xmm4		// cpy srcp
      movdqa xmm3, xmm1		// cpy prev
      psubusb xmm5, xmm2		// next-srcp
      psubusb xmm3, xmm4		// prev-srcp
      psubusb xmm2, xmm0		// srcp-next
      psubusb xmm4, xmm1		// srcp-prev 
      pminub xmm3, xmm5
      pminub xmm2, xmm4
      pmaxub xmm2, xmm3
      paddusb xmm2, xmm6
      pcmpeqb xmm2, xmm7
      pand xmm2, lumaMask

      movdqa xmm3, xmm2
      psrldq xmm2, 4
      movd edi, xmm3
      movd esi, xmm2
      or edi, esi
      jnz output2
      movdqa xmm4, xmm2
      psrldq xmm2, 4
      psrldq xmm4, 8
      movd edi, xmm2
      movd esi, xmm4
      or edi, esi
      jnz output2
      mov edi, src_pitch
      add ecx, 16
      lea eax, [eax + edi * 2]
      cmp ecx, width
      jl xloop
      add eax, edi
      add edx, dst_pitch
      dec height
      jnz yloop
      jmp end
      output2 :
    mov esi, src_pitch2
      mov edi, src_pitch
      pxor xmm7, xmm7 // PF Fix: 170418 was: pxor xmm7, mm7. Used in YUY2
      movdqa xmm2, xmm0
      movdqa xmm4, xmm1
      punpcklbw xmm2, xmm7
      punpckhbw xmm0, xmm7
      sub eax, edi
      punpcklbw xmm4, xmm7
      punpckhbw xmm1, xmm7
      paddusw xmm2, xmm4
      paddusw xmm0, xmm1
      pmullw xmm2, threeMask	// 3*(p+n)
      pmullw xmm0, threeMask	// 3*(p+n)
      movdqa xmm1, [eax + ecx]
      movdqa xmm4, xmm1
      punpcklbw xmm4, xmm7
      add eax, esi
      punpckhbw xmm1, xmm7
      movdqa xmm5, [eax + ecx]
      movdqa xmm6, xmm5
      punpcklbw xmm6, xmm7
      punpckhbw xmm5, xmm7
      psllw xmm6, 2
      add eax, esi
      psllw xmm5, 2
      paddusw xmm4, xmm6
      paddusw xmm1, xmm5
      movdqa xmm5, [eax + ecx]
      movdqa xmm6, xmm5
      punpcklbw xmm6, xmm7
      punpckhbw xmm5, xmm7
      paddusw xmm4, xmm6			// (pp+c*4+nn)
      paddusw xmm1, xmm5			// (pp+c*4+nn)
      movdqa xmm6, xmm4
      movdqa xmm5, xmm1
      psubusw xmm6, xmm2
      psubusw xmm5, xmm0
      psubusw xmm2, xmm4
      psubusw xmm0, xmm1
      pcmpeqb xmm7, xmm7
      pmaxsw xmm2, xmm6
      pmaxsw xmm0, xmm5
      paddusw xmm2, thresh6w
      paddusw xmm0, thresh6w
      pcmpeqw xmm2, xmm7
      pcmpeqw xmm0, xmm7
      sub eax, edi
      psrlw xmm2, 8
      movdqa xmm6, threshb
      psrlw xmm0, 8
      packuswb xmm2, xmm0
      pand xmm3, xmm2
      movdqa[edx + ecx], xmm3
      add ecx, 16
      cmp ecx, width
      jl xloop
      add eax, edi
      add edx, dst_pitch
      dec height
      jnz yloop
      end :
  }
#endif
}

// instantiate
template void check_combing_SSE2_Luma<false>(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb, __m128i thresh6w);
template void check_combing_SSE2_Luma<true>(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __m128i threshb, __m128i thresh6w);


#ifdef ALLOW_MMX
void check_combing_iSSE_Luma(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb, __int64 thresh6w)
{
  __asm
  {
    mov eax, srcp
    mov ebx, dstp
    mov edx, width
    mov edi, src_pitch
    add eax, edi
    movq mm6, threshb
    pcmpeqb mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]	// next
      sub eax, edi
      movq mm4, [eax + ecx]	// srcp
      movq mm5, mm0		// cpy next
      sub eax, edi
      movq mm1, [eax + ecx]	// prev
      movq mm2, mm4		// cpy srcp
      movq mm3, mm1		// cpy prev
      psubusb mm5, mm2		// next-srcp
      psubusb mm3, mm4		// prev-srcp
      psubusb mm2, mm0		// srcp-next
      psubusb mm4, mm1		// srcp-prev 
      pminub mm3, mm5
      pminub mm2, mm4
      pmaxub mm2, mm3
      paddusb mm2, mm6
      pcmpeqb mm2, mm7
      pand mm2, lumaMask
      movq mm3, mm2
      psrlq mm2, 32
      movd edi, mm3
      movd esi, mm2
      or edi, esi
      jnz output2
      mov edi, src_pitch
      add ecx, 8
      lea eax, [eax + edi * 2]
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      jmp end
      output2 :
    mov esi, src_pitch2
      mov edi, src_pitch
      pxor mm7, mm7
      movq mm2, mm0
      movq mm4, mm1
      punpcklbw mm2, mm7
      punpckhbw mm0, mm7
      sub eax, edi
      punpcklbw mm4, mm7
      punpckhbw mm1, mm7
      paddusw mm2, mm4
      paddusw mm0, mm1
      pmullw mm2, threeMask	// 3*(p+n)
      pmullw mm0, threeMask	// 3*(p+n)
      movq mm1, [eax + ecx]
      movq mm4, mm1
      punpcklbw mm4, mm7
      add eax, esi
      punpckhbw mm1, mm7
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      psllw mm6, 2
      add eax, esi
      psllw mm5, 2
      paddusw mm4, mm6
      paddusw mm1, mm5
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      paddusw mm4, mm6			// (pp+c*4+nn)
      paddusw mm1, mm5			// (pp+c*4+nn)
      movq mm6, mm4
      movq mm5, mm1
      psubusw mm6, mm2
      psubusw mm5, mm0
      psubusw mm2, mm4
      psubusw mm0, mm1
      pcmpeqb mm7, mm7
      pmaxsw mm2, mm6
      pmaxsw mm0, mm5
      paddusw mm2, thresh6w
      paddusw mm0, thresh6w
      pcmpeqw mm2, mm7
      pcmpeqw mm0, mm7
      sub eax, edi
      psrlw mm2, 8
      movq mm6, threshb
      psrlw mm0, 8
      packuswb mm2, mm0
      pand mm3, mm2
      movq[ebx + ecx], mm3
      add ecx, 8
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      end :
    emms
  }
}
#endif

#ifdef ALLOW_MMX
void check_combing_MMX_Luma(const unsigned char *srcp, unsigned char *dstp, int width,
  int height, int src_pitch, int src_pitch2, int dst_pitch, __int64 threshb, __int64 thresh6w)
{
  __asm
  {
    mov eax, srcp
    mov ebx, dstp
    mov edx, width
    mov edi, src_pitch
    add eax, edi
    movq mm6, threshb
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]	// next
      sub eax, edi
      movq mm4, [eax + ecx]	// srcp
      movq mm5, mm0		// cpy next
      sub eax, edi
      movq mm1, [eax + ecx]	// prev
      movq mm2, mm4		// cpy srcp
      movq mm3, mm1		// cpy prev
      psubusb mm5, mm2		// next-srcp
      psubusb mm3, mm4		// prev-srcp
      psubusb mm2, mm0		// srcp-next
      psubusb mm4, mm1		// srcp-prev
      //pminub mm3,mm5
      //pminub mm2,mm4
      movq mm0, mm3
      movq mm1, mm2
      psubusb mm0, mm5
      psubusb mm1, mm4
      pcmpeqb mm0, mm7
      pcmpeqb mm1, mm7
      pand mm3, mm0
      pand mm2, mm1
      pxor mm0, ffMask
      pxor mm1, ffMask
      pand mm5, mm0
      pand mm4, mm1
      por mm3, mm5			// min(mm3,mm5)
      por mm2, mm4			// min(mm2,mm4)
      //pmaxub mm2,mm3
      movq mm0, mm2
      psubusb mm0, mm3
      pcmpeqb mm0, mm7
      pand mm3, mm0
      pxor mm0, ffMask
      pand mm2, mm0
      por mm2, mm3			// max(mm2,mm3)
      paddusb mm2, mm6
      pcmpeqb mm2, ffMask
      pand mm2, lumaMask
      movq mm3, mm2
      psrlq mm2, 32
      movd edi, mm3
      movd esi, mm2
      or edi, esi
      jnz output2
      mov edi, src_pitch
      add ecx, 8
      lea eax, [eax + edi * 2]
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      jmp end
      output2 :
    movq mm1, [eax + ecx]	// prev
      mov esi, src_pitch2
      mov edi, src_pitch
      add eax, esi
      movq mm0, [eax + ecx]	// next
      movq mm2, mm0
      movq mm4, mm1
      sub eax, esi
      punpcklbw mm2, mm7
      punpckhbw mm0, mm7
      sub eax, edi
      punpcklbw mm4, mm7
      punpckhbw mm1, mm7
      paddusw mm2, mm4
      paddusw mm0, mm1
      pmullw mm2, threeMask	// 3*(p+n)
      pmullw mm0, threeMask	// 3*(p+n)
      movq mm1, [eax + ecx]
      movq mm4, mm1
      punpcklbw mm4, mm7
      add eax, esi
      punpckhbw mm1, mm7
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      psllw mm6, 2
      add eax, esi
      psllw mm5, 2
      paddusw mm4, mm6
      paddusw mm1, mm5
      movq mm5, [eax + ecx]
      movq mm6, mm5
      punpcklbw mm6, mm7
      punpckhbw mm5, mm7
      paddusw mm4, mm6			// (pp+c*4+nn)
      paddusw mm1, mm5			// (pp+c*4+nn)
      movq mm6, mm4
      movq mm5, mm1
      psubusw mm6, mm2
      psubusw mm5, mm0
      psubusw mm2, mm4
      psubusw mm0, mm1
      //pmaxsw mm2,mm6
      //pmaxsw mm0,mm5
      movq mm1, mm2
      movq mm4, mm0
      pcmpgtw mm1, mm6
      pcmpgtw mm4, mm5
      pand mm2, mm1
      pand mm0, mm4
      pxor mm1, ffMask
      pxor mm4, ffMask
      pand mm6, mm1
      pand mm5, mm4
      por mm2, mm6				// max(mm2,mm6)
      por mm0, mm5				// max(mm0,mm5)
      paddusw mm2, thresh6w
      paddusw mm0, thresh6w
      pcmpeqw mm2, ffMask
      pcmpeqw mm0, ffMask
      sub eax, edi
      psrlw mm2, 8
      movq mm6, threshb
      psrlw mm0, 8
      packuswb mm2, mm0
      pand mm3, mm2
      movq[ebx + ecx], mm3
      add ecx, 8
      cmp ecx, edx
      jl xloop
      add eax, edi
      add ebx, dst_pitch
      dec height
      jnz yloop
      end :
    emms
  }
}
#endif

#ifdef ALLOW_MMX
void check_combing_MMX_M1(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __int64 thresh)
{
  __asm
  {
    mov ebx, srcp
    mov edi, dstp
    mov eax, ebx
    mov esi, ebx
    sub eax, src_pitch
    add esi, src_pitch
    mov edx, width
    movq mm6, thresh
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [ebx + ecx]
      movq mm2, [esi + ecx]
      movq mm3, mm0
      movq mm4, mm1
      movq mm5, mm2
      punpcklbw mm0, mm7
      punpcklbw mm1, mm7
      punpcklbw mm2, mm7
      punpckhbw mm3, mm7
      punpckhbw mm4, mm7
      punpckhbw mm5, mm7
      psubsw mm0, mm1
      psubsw mm2, mm1
      psubsw mm3, mm4
      psubsw mm5, mm4
      movq mm1, mm0
      movq mm4, mm2
      punpcklwd mm0, mm7
      punpckhwd mm1, mm7
      punpcklwd mm2, mm7
      punpckhwd mm4, mm7
      pmaddwd mm0, mm2
      pmaddwd mm1, mm4
      movq mm2, mm3
      movq mm4, mm5
      punpcklwd mm2, mm7
      punpckhwd mm3, mm7
      punpcklwd mm4, mm7
      punpckhwd mm5, mm7
      pmaddwd mm2, mm4
      pmaddwd mm3, mm5
      pcmpgtd mm0, mm6
      pcmpgtd mm1, mm6
      pcmpgtd mm2, mm6
      pcmpgtd mm3, mm6
      packssdw mm0, mm1
      packssdw mm2, mm3
      pand mm0, lumaMask
      pand mm2, lumaMask
      packuswb mm0, mm2
      movq[edi + ecx], mm0
      add ecx, 8
      cmp ecx, edx
      jl xloop
      add eax, src_pitch
      add ebx, src_pitch
      add esi, src_pitch
      add edi, dst_pitch
      dec height
      jnz yloop
      emms
  }
}
#endif

template<bool aligned>
void check_combing_SSE2_M1(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __m128i thresh)
{
#ifdef USE_INTR
  __m128i all_ff = _mm_set1_epi8(-1);
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

#else
  __asm
  {
    movdqa xmm6, thresh
    mov edx, srcp
    mov edi, dstp
    mov eax, edx
    mov esi, edx
    sub eax, src_pitch
    add esi, src_pitch
    pxor xmm7, xmm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx] // prev
      movdqa xmm1, [edx + ecx] // curr
      movdqa xmm2, [esi + ecx] // next
      movdqa xmm3, xmm0
      movdqa xmm4, xmm1
      movdqa xmm5, xmm2
      punpcklbw xmm0, xmm7 // prev_lo
      punpcklbw xmm1, xmm7 // curr_lo
      punpcklbw xmm2, xmm7 // next_lo
      punpckhbw xmm3, xmm7 // prev_hi
      punpckhbw xmm4, xmm7 // curr_hi
      punpckhbw xmm5, xmm7 // next_hi
      psubsw xmm0, xmm1 // diff_prev_curr_lo = prev_lo - curr_lo
      psubsw xmm2, xmm1 // diff_next_curr_lo = next_lo - curr_lo
      psubsw xmm3, xmm4 // diff_prev_curr_hi = prev_hi - curr_hi
      psubsw xmm5, xmm4 // diff_next_curr_hi = next_hi - curr_hi

      movdqa xmm1, xmm0
      movdqa xmm4, xmm2
      punpcklwd xmm0, xmm7 // diff_prev_curr_lo_lo
      punpckhwd xmm1, xmm7 // diff_prev_curr_lo_hi
      punpcklwd xmm2, xmm7 // diff_next_curr_lo_lo
      punpckhwd xmm4, xmm7 // diff_next_curr_lo_hi
      pmaddwd xmm0, xmm2   // res_lo_lo = _mm_madd_epi16(diff_prev_curr_lo_lo,diff_next_curr_lo_lo)
      pmaddwd xmm1, xmm4   // res_lo_hi = _mm_madd_epi16(diff_prev_curr_lo_hi,diff_next_curr_lo_hi)

      movdqa xmm2, xmm3   
      movdqa xmm4, xmm5
      punpcklwd xmm2, xmm7 // diff_prev_curr_hi_lo
      punpckhwd xmm3, xmm7 // diff_prev_curr_hi_hi
      punpcklwd xmm4, xmm7 // diff_next_curr_hi_lo
      punpckhwd xmm5, xmm7 // diff_next_curr_hi_hi
      pmaddwd xmm2, xmm4   // res_hi_lo = _mm_madd_epi16(diff_prev_curr_hi_lo,diff_next_curr_hi_lo)
      pmaddwd xmm3, xmm5   // res_hi_hi = _mm_madd_epi16(diff_prev_curr_hi_hi,diff_next_curr_hi_hi)

      pcmpgtd xmm0, xmm6   // cmp_lo_lo = mm_cmp_epi32(res_lo_lo,thresh)
      pcmpgtd xmm1, xmm6   // cmp_lo_hi = mm_cmp_epi32(res_lo_hi,thresh)
      pcmpgtd xmm2, xmm6   // cmp_hi_lo = mm_cmp_epi32(res_hi_lo,thresh)
      pcmpgtd xmm3, xmm6   // cmp_hi_hi = mm_cmp_epi32(res_hi_hi,thresh)

      packssdw xmm0, xmm1  // cmp_lo = mm_packs_epi32(cmp_lo_lo,cmp_lo_hi);
      packssdw xmm2, xmm3  // cmp_hi = mm_packs_epi32(cmp_hi_lo,cmp_hi_hi);
      pand xmm0, lumaMask  // cmp_lo_masked = _mm_and_si128(cmp_lo, lumaMask)
      pand xmm2, lumaMask  // cmp_hi_masked = _mm_and_si128(cmp_hi, lumaMask)
      packuswb xmm0, xmm2  // res = _mm_packus_epi16(cmp_lo_masked, cmp_hi_masked)
      movdqa[edi + ecx], xmm0
      add ecx, 16
      cmp ecx, width
      jl xloop
      add eax, src_pitch
      add edx, src_pitch
      add esi, src_pitch
      add edi, dst_pitch
      dec height
      jnz yloop
  }
#endif
}

// instantiate
template void check_combing_SSE2_M1<false>(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __m128i thresh);
template void check_combing_SSE2_M1<true>(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __m128i thresh);


#ifdef ALLOW_MMX
void check_combing_MMX_Luma_M1(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __int64 thresh)
{
  __asm
  {
    mov ebx, srcp
    mov edi, dstp
    mov eax, ebx
    mov esi, ebx
    sub eax, src_pitch
    add esi, src_pitch
    mov edx, width
    movq mm5, lumaMask
    movq mm6, thresh
    pxor mm7, mm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movq mm0, [eax + ecx]
      movq mm1, [ebx + ecx]
      movq mm2, [esi + ecx]
      pand mm0, mm5
      pand mm1, mm5
      pand mm2, mm5
      psubsw mm0, mm1
      psubsw mm2, mm1
      movq mm1, mm0
      movq mm4, mm2
      punpcklwd mm0, mm7
      punpckhwd mm1, mm7
      punpcklwd mm2, mm7
      punpckhwd mm4, mm7
      pmaddwd mm0, mm2
      pmaddwd mm1, mm4
      pcmpgtd mm0, mm6
      pcmpgtd mm1, mm6
      packssdw mm0, mm1
      pand mm0, mm5
      movq[edi + ecx], mm0
      add ecx, 8
      cmp ecx, edx
      jl xloop
      add eax, src_pitch
      add ebx, src_pitch
      add esi, src_pitch
      add edi, dst_pitch
      dec height
      jnz yloop
      emms
  }
}
#endif


template<bool aligned>
void check_combing_SSE2_Luma_M1(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __m128i thresh)
{
#ifdef USE_INTR
  __m128i lumaMask = _mm_set1_epi16(0x00FF);
  __m128i all_ff = _mm_set1_epi8(-1);
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
#else
  __asm
  {
    movdqa xmm6, thresh
    mov edx, srcp
    mov edi, dstp
    mov eax, edx
    mov esi, edx
    sub eax, src_pitch
    add esi, src_pitch
    movdqa xmm5, lumaMask
    pxor xmm7, xmm7
    yloop :
    xor ecx, ecx
      align 16
      xloop :
      movdqa xmm0, [eax + ecx]
      movdqa xmm1, [edx + ecx]
      movdqa xmm2, [esi + ecx]
      pand xmm0, xmm5
      pand xmm1, xmm5
      pand xmm2, xmm5
      psubsw xmm0, xmm1
      psubsw xmm2, xmm1
      movdqa xmm1, xmm0
      movdqa xmm4, xmm2
      punpcklwd xmm0, xmm7
      punpckhwd xmm1, xmm7
      punpcklwd xmm2, xmm7
      punpckhwd xmm4, xmm7
      pmaddwd xmm0, xmm2
      pmaddwd xmm1, xmm4
      pcmpgtd xmm0, xmm6
      pcmpgtd xmm1, xmm6
      packssdw xmm0, xmm1
      pand xmm0, xmm5
      movdqa[edi + ecx], xmm0
      add ecx, 16
      cmp ecx, width
      jl xloop
      add eax, src_pitch
      add edx, src_pitch
      add esi, src_pitch
      add edi, dst_pitch
      dec height
      jnz yloop
  }
#endif
}

// instantiate
template void check_combing_SSE2_Luma_M1<false>(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __m128i thresh);
template void check_combing_SSE2_Luma_M1<true>(const unsigned char *srcp, unsigned char *dstp,
  int width, int height, int src_pitch, int dst_pitch, __m128i thresh);


// There are no emms instructions at the end of these compute_sum 
// mmx/isse routines because it is called at the end of the routine 
// that calls these individual functions.

// no alignment needed for 8 bytes
void compute_sum_8x8_sse2(const unsigned char *srcp, int pitch, int &sum)
{
  // sums masks
  // if (cmkppT[x + v] == 0xFF && cmkpT[x + v] == 0xFF && cmkpnT[x + v] == 0xFF) sum++;
  __m128i onesMask = _mm_set1_epi8(1);
  __m128i prev0 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp));
  __m128i prev1_currMinus1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp + pitch));
  __m128i zero = _mm_setzero_si128();
  __m128i summa = _mm_setzero_si128();
  srcp += pitch * 2;
  //for (int i = 0; i < 4; i++) { // 4x2=8
  {
	__m128i curr0_prev2 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp));
    __m128i curr1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp + pitch));

    __m128i prev_anded = _mm_and_si128(prev0, prev1_currMinus1);
    prev_anded = _mm_and_si128(prev_anded, curr0_prev2); // prev0 prev1 prev2
    prev_anded = _mm_and_si128(prev_anded, onesMask);

    __m128i curr_anded = _mm_and_si128(curr0_prev2, curr1);
    curr_anded = _mm_and_si128(curr_anded, prev1_currMinus1); // currMinus1 curr0 curr1
    curr_anded = _mm_and_si128(curr_anded, onesMask);

    prev0 = curr0_prev2;
    prev1_currMinus1 = curr1;

    summa = _mm_adds_epu8(summa, prev_anded);
    summa = _mm_adds_epu8(summa, curr_anded);
    srcp += pitch * 2;

	curr0_prev2 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp));
	curr1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp + pitch));

	prev_anded = _mm_and_si128(prev0, prev1_currMinus1);
	prev_anded = _mm_and_si128(prev_anded, curr0_prev2); // prev0 prev1 prev2
	prev_anded = _mm_and_si128(prev_anded, onesMask);

	curr_anded = _mm_and_si128(curr0_prev2, curr1);
	curr_anded = _mm_and_si128(curr_anded, prev1_currMinus1); // currMinus1 curr0 curr1
	curr_anded = _mm_and_si128(curr_anded, onesMask);

	prev0 = curr0_prev2;
	prev1_currMinus1 = curr1;

	summa = _mm_adds_epu8(summa, prev_anded);
	summa = _mm_adds_epu8(summa, curr_anded);
	srcp += pitch * 2;

	curr0_prev2 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp));
	curr1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp + pitch));

	prev_anded = _mm_and_si128(prev0, prev1_currMinus1);
	prev_anded = _mm_and_si128(prev_anded, curr0_prev2); // prev0 prev1 prev2
	prev_anded = _mm_and_si128(prev_anded, onesMask);

	curr_anded = _mm_and_si128(curr0_prev2, curr1);
	curr_anded = _mm_and_si128(curr_anded, prev1_currMinus1); // currMinus1 curr0 curr1
	curr_anded = _mm_and_si128(curr_anded, onesMask);

	prev0 = curr0_prev2;
	prev1_currMinus1 = curr1;

	summa = _mm_adds_epu8(summa, prev_anded);
	summa = _mm_adds_epu8(summa, curr_anded);
	srcp += pitch * 2;

	curr0_prev2 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp));
	curr1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp + pitch));

	prev_anded = _mm_and_si128(prev0, prev1_currMinus1);
	prev_anded = _mm_and_si128(prev_anded, curr0_prev2); // prev0 prev1 prev2
	prev_anded = _mm_and_si128(prev_anded, onesMask);

	curr_anded = _mm_and_si128(curr0_prev2, curr1);
	curr_anded = _mm_and_si128(curr_anded, prev1_currMinus1); // currMinus1 curr0 curr1
	curr_anded = _mm_and_si128(curr_anded, onesMask);

	prev0 = curr0_prev2;
	prev1_currMinus1 = curr1;

	summa = _mm_adds_epu8(summa, prev_anded);
	summa = _mm_adds_epu8(summa, curr_anded);
	srcp += pitch * 2;

  }
  // now we have to sum up lower 8 bytes
  // in sse2, we use sad
  __m128i tmpsum = _mm_sad_epu8(summa, zero);  // sum(lo 8 bytes)(needed) / sum(hi 8 bytes)(not needed)
  sum = _mm_cvtsi128_si32(tmpsum);
}

#pragma warning(push)
#pragma warning(disable:4799)	// disable no emms warning message

#ifdef ALLOW_MMX
void compute_sum_8x8_mmx(const unsigned char *srcp, int pitch, int &sum)
{
  __asm
  {
    mov eax, srcp
    mov edi, pitch
    mov ecx, 4
    movq mm0, [eax]
    movq mm1, [eax + edi]
    movq mm5, onesMask
    lea eax, [eax + edi * 2]
    pxor mm6, mm6
    pxor mm7, mm7
    align 16
    loopy:
    movq mm2, [eax]
      movq mm3, [eax + edi]
      movq mm4, mm2
      pand mm0, mm1
      pand mm4, mm3
      pand mm0, mm2
      pand mm4, mm1
      pand mm0, mm5
      pand mm4, mm5
      paddusb mm7, mm0
      lea eax, [eax + edi * 2]
      movq mm0, mm2
      movq mm1, mm3
      paddusb mm7, mm4
      dec ecx
      jnz loopy
      movq mm0, mm7
      mov eax, sum
      punpcklbw mm7, mm6
      punpckhbw mm0, mm6
      paddusw mm7, mm0
      movq mm0, mm7
      punpcklwd mm7, mm6
      punpckhwd mm0, mm6
      paddd mm7, mm0
      movq mm0, mm7
      psrlq mm7, 32
      paddd mm0, mm7
      movd[eax], mm0
  }
}
#endif

#ifdef ALLOW_MMX
void compute_sum_8x8_isse(const unsigned char *srcp, int pitch, int &sum)
{
  __asm
  {
    mov eax, srcp
    mov edi, pitch
    mov ecx, 4
    movq mm0, [eax]
    movq mm1, [eax + edi]
    movq mm5, onesMask
    lea eax, [eax + edi * 2]
    pxor mm6, mm6
    pxor mm7, mm7
    align 16
    loopy:
    movq mm2, [eax]
      movq mm3, [eax + edi]
      movq mm4, mm2
      pand mm0, mm1
      pand mm4, mm3
      pand mm0, mm2
      pand mm4, mm1
      pand mm0, mm5
      pand mm4, mm5
      paddusb mm7, mm0
      lea eax, [eax + edi * 2]
      movq mm0, mm2
      movq mm1, mm3
      paddusb mm7, mm4
      dec ecx
      jnz loopy
      mov eax, sum
      psadbw mm7, mm6
      movd[eax], mm7
  }
}
#endif

#ifdef ALLOW_MMX
void compute_sum_8x16_mmx_luma(const unsigned char *srcp, int pitch, int &sum)
{
  __asm
  {
    mov eax, srcp
    mov edi, pitch
    mov ecx, 4
    xor edx, edx
    movq mm0, [eax]
    movq mm1, [eax + edi]
    movq mm5, onesMaskLuma
    lea eax, [eax + edi * 2]
    pxor mm6, mm6
    pxor mm7, mm7
    jmp xskip
    loopx :
    mov eax, srcp
      mov ecx, 4
      add eax, 8
      inc edx
      movq mm0, [eax]
      movq mm1, [eax + edi]
      lea eax, [eax + edi * 2]
      xskip:
    align 16
      loopy :
      movq mm2, [eax]
      movq mm3, [eax + edi]
      movq mm4, mm2
      pand mm0, mm1
      pand mm4, mm3
      pand mm0, mm2
      pand mm4, mm1
      pand mm0, mm5
      pand mm4, mm5
      paddusb mm7, mm0
      lea eax, [eax + edi * 2]
      movq mm0, mm2
      movq mm1, mm3
      paddusb mm7, mm4
      dec ecx
      jnz loopy
      or edx, edx
      jz loopx
      movq mm0, mm7
      mov eax, sum
      punpcklwd mm7, mm6
      punpckhwd mm0, mm6
      paddd mm7, mm0
      movq mm0, mm7
      psrlq mm7, 32
      paddd mm0, mm7
      movd[eax], mm0
  }
}
#endif

#ifdef ALLOW_MMX
void compute_sum_8x16_isse_luma(const unsigned char *srcp, int pitch, int &sum)
{
  __asm
  {
    mov eax, srcp
    mov edi, pitch
    mov ecx, 4
    xor edx, edx
    movq mm0, [eax]
    movq mm1, [eax + edi]
    movq mm5, onesMaskLuma
    lea eax, [eax + edi * 2]
    pxor mm6, mm6
    pxor mm7, mm7
    jmp xskip
    loopx :
    mov eax, srcp
      add ecx, 4
      add eax, 8
      inc edx
      movq mm0, [eax]
      movq mm1, [eax + edi]
      lea eax, [eax + edi * 2]
      xskip:
    align 16
      loopy :
      movq mm2, [eax]
      movq mm3, [eax + edi]
      movq mm4, mm2
      pand mm0, mm1
      pand mm4, mm3
      pand mm0, mm2
      pand mm4, mm1
      pand mm0, mm5
      pand mm4, mm5
      paddusb mm7, mm0
      lea eax, [eax + edi * 2]
      movq mm0, mm2
      movq mm1, mm3
      paddusb mm7, mm4
      dec ecx
      jnz loopy
      or edx, edx
      jz loopx
      mov eax, sum
      psadbw mm7, mm6
      movd[eax], mm7
  }
}
#endif


template<bool aligned>
void compute_sum_8x16_sse2_luma(const unsigned char *srcp, int pitch, int &sum)
{
#ifdef USE_INTR
  // sums masks
  // if (cmkppT[x + v] == 0xFF && cmkpT[x + v] == 0xFF && cmkpnT[x + v] == 0xFF) sum++;
  __m128i onesMask = _mm_set1_epi16(1); // onesMaskLuma Word(1)
  __m128i prev0, prev1_currMinus1;
  if (aligned) {
    prev0 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp));
    prev1_currMinus1 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + pitch));
  }
  else {
    prev0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp));
    prev1_currMinus1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + pitch));
  }
  __m128i zero = _mm_setzero_si128();
  __m128i summa = _mm_setzero_si128();
  srcp += pitch * 2;
  for (int i = 0; i < 4; i++) { // 4x2=8
    __m128i curr0_prev2, curr1;
    if (aligned) {
      curr0_prev2 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp));
      curr1 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + pitch));
    }
    else {
      curr0_prev2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp));
      curr1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + pitch));
    }

    __m128i prev_anded = _mm_and_si128(prev0, prev1_currMinus1);
    prev_anded = _mm_and_si128(prev_anded, curr0_prev2); // prev0 prev1 prev2
    prev_anded = _mm_and_si128(prev_anded, onesMask);

    __m128i curr_anded = _mm_and_si128(curr0_prev2, curr1);
    curr_anded = _mm_and_si128(curr_anded, prev1_currMinus1); // currMinus1 curr0 curr1
    curr_anded = _mm_and_si128(curr_anded, onesMask);

    prev0 = curr0_prev2;
    prev1_currMinus1 = curr1;

    summa = _mm_adds_epu8(summa, prev_anded);
    summa = _mm_adds_epu8(summa, curr_anded);
    srcp += pitch * 2;
  }
  // now we have to sum up lower 8 bytes
  // in sse2, we use sad
  __m128i tmpsum = _mm_sad_epu8(summa, zero);  // sum(lo 8 bytes) / sum(hi 8 bytes)
  tmpsum = _mm_add_epi32(tmpsum, _mm_srli_si128(tmpsum, 8)); // lo + hi
  sum = _mm_cvtsi128_si32(tmpsum);
#else
  __asm
  {
    mov eax, srcp
    mov edi, pitch
    mov ecx, 4
    movdqa xmm0, [eax]
    movdqa xmm1, [eax + edi]
    movdqa xmm5, onesMaskLuma
    lea eax, [eax + edi * 2]
    pxor xmm6, xmm6
    pxor xmm7, xmm7
    align 16
    loopy:
    movdqa xmm2, [eax]
      movdqa xmm3, [eax + edi]
      movdqa xmm4, xmm2
      pand xmm0, xmm1
      pand xmm4, xmm3
      pand xmm0, xmm2
      pand xmm4, xmm1
      pand xmm0, xmm5
      pand xmm4, xmm5
      paddusb xmm7, xmm0
      lea eax, [eax + edi * 2]
      movdqa xmm0, xmm2
      movdqa xmm1, xmm3
      paddusb xmm7, xmm4
      dec ecx
      jnz loopy

      mov eax, sum
      psadbw xmm7, xmm6
      movdqa xmm4, xmm7
      psrldq xmm7, 8
      paddq xmm4, xmm7
      movd[eax], xmm4
  }
#endif
}

// instantiate
template void compute_sum_8x16_sse2_luma<false>(const unsigned char *srcp, int pitch, int &sum);
template void compute_sum_8x16_sse2_luma<true>(const unsigned char *srcp, int pitch, int &sum);


__forceinline __m128i compareFieldsSlowCal0_SSSE3(int ebx, __m128i readmsk, unsigned char *t_mapp, unsigned char *t_mapn)
{
	//eax = (t_mapp[ebx] << 3) + t_mapn[ebx];	 // diff from prev asm block (at buildDiffMapPlane2): <<3 instead of <<2
	auto temp = _mm_loadu_si128((__m128i *)(t_mapp + ebx));	//128bit境界以外にアクセスするが大丈夫？ ＋配列の順番注意
	temp = _mm_shuffle_epi8(temp, readmsk);		//_mm_cvtepu8_epi16(temp);		//incl==1では下位8個を16bitに展開、==2では2byte毎
	__m128i eax = _mm_slli_epi16(temp,3);		//ssse3 to do shift and mul... 8bit命令はない
	temp = _mm_loadu_si128((__m128i *)(t_mapn + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	eax = _mm_add_epi16(eax,temp);

	return eax;
}

__forceinline void compareFieldsSlowCal1_SSSE3(int ebx, __m128i eax, __m128i readmsk,
	const unsigned char *t_prvpf, const unsigned char*t_prvnf,
	const unsigned char *t_curpf, const unsigned char*t_curf, const unsigned char*t_curnf,
	const unsigned char *t_nxtpf, const unsigned char*t_nxtnf,
	unsigned long  &accumPc, unsigned long  &accumNc, unsigned long  &accumPm, unsigned long  &accumNm, unsigned long  &accumPml, unsigned long  &accumNml)
{
	__m128i temp,a_curr,a_prev,a_next,diff_p_c,diff_n_c,eaxmsk;
	__m128i zero = _mm_setzero_si128();

	//a_curr = t_curpf[ebx] + (t_curf[ebx] << 2) + t_curnf[ebx];
	temp = _mm_loadu_si128((__m128i *)(t_curpf + ebx));
	a_curr = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_loadu_si128((__m128i *)(t_curf + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_slli_epi16(temp,2);
	a_curr = _mm_add_epi16(a_curr,temp);
	temp = _mm_loadu_si128((__m128i *)(t_curnf + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	a_curr = _mm_add_epi16(a_curr,temp);


	//a_prev = 3 * (t_prvpf[ebx] + t_prvnf[ebx]);
	temp = _mm_loadu_si128((__m128i *)(t_prvpf + ebx));
	a_prev = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_loadu_si128((__m128i *)(t_prvnf + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	a_prev = _mm_add_epi16(a_prev,temp);
	temp = _mm_set1_epi16(3);
	a_prev = _mm_mullo_epi16(a_prev,temp);


	//a_next = 3 * (t_nxtpf[ebx] + t_nxtnf[ebx]);
	temp = _mm_loadu_si128((__m128i *)(t_nxtpf + ebx));
	a_next = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_loadu_si128((__m128i *)(t_nxtnf + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	a_next = _mm_add_epi16(a_next,temp);
	temp = _mm_set1_epi16(3);
	a_next = _mm_mullo_epi16(a_next,temp);


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
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffpc>23) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_p_c);					//eaxmsk∧(diffpc>23) ? diffpc : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum ssse3
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumPc += _mm_cvtsi128_si32(temp);	

	temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(23));	//(diffnc>23) ? -1 : 0
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eax9∧(diffnc>23) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_n_c);					//eax9∧(diffnc>23) ? diffnc : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumNc += _mm_cvtsi128_si32(temp);	


	//if ((eax & 18) != 0){
	//	if (diff_p_c > 42) {accumPm  += diff_p_c;}
	//	if (diff_n_c > 42) {accumNm  += diff_n_c;}
	//}
	temp = _mm_and_si128(eax, _mm_set1_epi16(18));
	temp = _mm_cmpeq_epi16(temp, zero);		//(eax&18 == 0) ? -1 : 0
	eaxmsk = _mm_cmpeq_epi16(temp, zero);	//(eax&18 != 0) ? -1 : 0

	temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));	//(diffpc>42) ? ffff : 0
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumPm += _mm_cvtsi128_si32(temp);	

	temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));	//(diffnc>42) ? ffff : 0
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffnc>42) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_n_c);					//eaxmsk∧(diffnc>42) ? diff_n_c : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumNm += _mm_cvtsi128_si32(temp);


	//if ((eax & 36) != 0){
	//	if (diff_p_c > 42) {accumPml += diff_p_c;}
	//	if (diff_n_c > 42) {accumNml += diff_n_c;}
	//}
	temp = _mm_and_si128(eax, _mm_set1_epi16(36));
	temp = _mm_cmpeq_epi16(temp, zero);		//(eax&36 == 0) ? -1 : 0
	eaxmsk = _mm_cmpeq_epi16(temp, zero);	//(eax&36 != 0) ? -1 : 0

	temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));	//(diffpc>42) ? ffff : 0
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumPml += _mm_cvtsi128_si32(temp);

	temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));	//(diffnc>42) ? ffff : 0
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffnc>42)で1、else0
	temp = _mm_mullo_epi16(temp,diff_n_c);					//eaxmsk∧(diffnc>42)でdiffnc、else0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumNml += _mm_cvtsi128_si32(temp);

	return;
}


__forceinline void compareFieldsSlowCal2_SSE41(int ebx, __m128i eax, __m128i readmsk,int sft,
	const unsigned char *t_prvf0, const unsigned char*t_prvf1, const unsigned char*t_prvf2,
	const unsigned char *t_curf0, const unsigned char*t_curf1,
	const unsigned char *t_nxtf0, const unsigned char*t_nxtf1, const unsigned char*t_nxtf2,
	unsigned long  &accumPc, unsigned long  &accumNc, unsigned long  &accumPm, unsigned long  &accumNm, unsigned long  &accumPml, unsigned long  &accumNml)
{
	__m128i temp,a_curr,a_prev,a_next,diff_p_c,diff_n_c,eaxmsk;
	__m128i zero = _mm_setzero_si128();

	// additional difference from TFM 1144

	//if ((eax & 56) == 0) continue; //field0
	//if ((eax &  7) == 0) continue; //field1
	temp = _mm_set1_epi16(7<<sft);
	if (_mm_testz_si128(eax,temp)){return;}	//sse4.1

	//	a_curr = 3 * (t_curpf[ebx] + t_curf[ebx]);	//field0
	//	a_curr = 3 * (t_curf[ebx] + t_curnf[ebx]);	//field1
	temp = _mm_loadu_si128((__m128i *)(t_curf0 + ebx));
	a_curr = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_loadu_si128((__m128i *)(t_curf1 + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	a_curr = _mm_add_epi16(a_curr,temp);
	temp = _mm_set1_epi16(3);
	a_curr = _mm_mullo_epi16(a_curr,temp);


	//	a_prev = t_prvppf[ebx] + (t_prvpf[ebx] << 2) + t_prvnf[ebx];	//field0
	//	a_prev = t_prvpf[ebx] + (t_prvnf[ebx] << 2) + t_prvnnf[ebx];	//field1
	temp = _mm_loadu_si128((__m128i *)(t_prvf0 + ebx));
	a_prev = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_loadu_si128((__m128i *)(t_prvf1 + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_slli_epi16(temp,2);
	a_prev = _mm_add_epi16(a_prev,temp);
	temp = _mm_loadu_si128((__m128i *)(t_prvf2 + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	a_prev = _mm_add_epi16(a_prev,temp);


	//	a_next = t_nxtppf[ebx] + (t_nxtpf[ebx] << 2) + t_nxtnf[ebx];	//field0
	//	a_next = t_nxtpf[ebx] + (t_nxtnf[ebx] << 2) + t_nxtnnf[ebx];	//field1
	temp = _mm_loadu_si128((__m128i *)(t_nxtf0 + ebx));
	a_next = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_loadu_si128((__m128i *)(t_nxtf1 + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	temp = _mm_slli_epi16(temp,2);
	a_next = _mm_add_epi16(a_next,temp);
	temp = _mm_loadu_si128((__m128i *)(t_nxtf2 + ebx));
	temp = _mm_shuffle_epi8(temp, readmsk);
	a_next = _mm_add_epi16(a_next,temp);

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
	temp = _mm_and_si128(eax, _mm_set1_epi16(1<<sft));		//
	temp = _mm_cmpeq_epi16(temp, zero);						//(eax&8 == 0) ? -1 : 0
	eaxmsk = _mm_cmpeq_epi16(temp, zero);					//(eax&8 == 0) ? 0 : -1

	temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(23));	
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffpc>23) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_p_c);					//eaxmsk∧(diffpc>23) ? diffpc : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumPc += _mm_cvtsi128_si32(temp);	

	temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(23));
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffnc>23) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_n_c);					//eaxmsk∧(diffnc>23) ? diffnc : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumNc += _mm_cvtsi128_si32(temp);	

	//	if ((eax & 16) != 0){	//field0:16 field1:2
	//		if (diff_p_c > 42) {accumPm  += diff_p_c;}
	//		if (diff_n_c > 42) {accumNm  += diff_n_c;}
	//	}
	temp = _mm_and_si128(eax, _mm_set1_epi16(2<<sft));
	temp = _mm_cmpeq_epi16(temp, zero);						//(eax&16 == 0) ? -1 : 0
	eaxmsk = _mm_cmpeq_epi16(temp, zero);					//(eax&16 == 0) ? 0 : -1

	temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));	
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumPm += _mm_cvtsi128_si32(temp);

	temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));	
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffnc>42) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_n_c);					//eaxmsk∧(diffnc>42) ? diff_n_c : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumNm += _mm_cvtsi128_si32(temp);

	//	if ((eax & 32) != 0){	//field0:32 field1:4
	//		if (diff_p_c > 42) {accumPml += diff_p_c;}
	//		if (diff_n_c > 42) {accumNml += diff_n_c;}
	//	}
	temp = _mm_and_si128(eax, _mm_set1_epi16(4<<sft));		//
	temp = _mm_cmpeq_epi16(temp, zero);						//(eax&32 == 0) ? -1 : 0
	eaxmsk = _mm_cmpeq_epi16(temp, zero);					//(eax&32 == 0) ? 0 : -1

	temp = _mm_cmpgt_epi16(diff_p_c, _mm_set1_epi16(42));	
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffpc>42) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_p_c);					//eaxmsk∧(diffpc>42) ? diff_p_c : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumPml += _mm_cvtsi128_si32(temp);

	temp = _mm_cmpgt_epi16(diff_n_c, _mm_set1_epi16(42));
	temp = _mm_mullo_epi16(temp,eaxmsk);					//eaxmsk∧(diffnc>42) ? 1 : 0
	temp = _mm_mullo_epi16(temp,diff_n_c);					//eaxmsk∧(diffnc>42) ? diff_n_c : 0
	temp = _mm_hadd_epi16(temp,zero);						//sum
	temp = _mm_hadd_epi16(temp,zero);
	temp = _mm_hadd_epi16(temp,zero);
	accumNml += _mm_cvtsi128_si32(temp);
}

void buildDiffMapPlaneYV12_SSE42(const unsigned char *tbuffer,
	unsigned char *dstp, int prv_pitch, int nxt_pitch, int dst_pitch, int Height,
	int Width, int tpitch)
{
	const unsigned char *dppp = tbuffer - tpitch;
	const unsigned char *dpp  = tbuffer;
	const unsigned char *dp   = tbuffer + tpitch;
	const unsigned char *dpn  = tbuffer + tpitch * 2;
	const unsigned char *dpnn = tbuffer + tpitch * 3;

	for (int y = 2; y < Height - 2; y += 2) {

		for (int x=1; x < Width - 1; x++){

			unsigned int dp_d,dpp_d,dpn_d;
			unsigned int tmpi;

			dp_d  = *(unsigned int *)(dp + x -1);				//-1,0,1,2の4個分入れておく 下の方(使う直前)に置くとxの依存関係がうまく切り分けられない？のかアクセスが遅い

			//現pixelが-3未満でスキップ
			//if (_mm_testz_si128(temp, _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-4)))
			if( (dp_d & 0x0000FC00) == 0)
			{
				auto temp = _mm_loadu_si128((__m128i *)(dp + x));	//x=1から始まるのが気になるが、速度的にはあまり変わらないっぽい
				if (_mm_testz_si128(temp, _mm_set1_epi8(-4))) {	//sse4.1
					do{	//連続スキップ用 ここで局所化するか次のforでやるかだけで演算は変わらないのであまり効果なし？
						x += 16;
						temp = _mm_loadu_si128((__m128i *)(dp + x));
					}
					while ((_mm_testz_si128(temp, _mm_set1_epi8(-4))) && (x < Width));
					x--;	//continue時に++されるから
					continue;
				}

				temp = _mm_and_si128( temp, _mm_set1_epi8( -4 ) );	// subs(0.5cpi)よりも速い(0.33cpi)
				temp = _mm_cmpeq_epi8( temp, _mm_setzero_si128() );	// <=3 ? ff : 0
				tmpi = _mm_movemask_epi8( temp );	// temp[] <=3 ? 1 : 0 各バイトの最上位ビットを抽出
				tmpi= ~tmpi;						// temp[] <=3 ? 0 : 1  上位16bitには1が入る andnotを使うよりも何故か速い

				//この時点で <=3 ? 0 : 1

				//tzcntの場合(BMIが使える≒AVX2)
				//tmpi = _tzcnt_u32(tmpi);	//tzcntの場合(BMI) bsfより高速
				//x += tmpi-1;continue;		//continue後(forのpost)に+1されるので-1

				//popcntで代替≒SSE42
				tmpi = _mm_popcnt_u32( ~tmpi & (tmpi-1));	//ちょっとだけ遅いが・・・SSE時はこちら
				x += tmpi-1;continue;		//continue後(forのpost)に+1されるので-1
				//x += tmpi;				//continueしない場合

				//bsfの場合 POPCNTも使えないならこうなる
				//long unsigned int tmpl;			//32bitを指定しないとbsfが文句を言う・・・
				//_BitScanForward(&tmpl,tmpi);	//組み込み関数使用 1が見つからない場合は返り値が0になる(無視してるが)
				//x += tmpl-1;					//continue後(forのpost)に+1されるので-1
				//continue;
			}

			//8近傍がすべて<=3でスキップ
			//if (dp[x - 1] <= 3 && dp[x + 1] <= 3 &&
			//dpp[x - 1] <= 3 && dpp[x] <= 3 && dpp[x + 1] <= 3 &&
			//dpn[x - 1] <= 3 && dpn[x] <= 3 && dpn[x + 1] <= 3) continue;
			dpp_d = *(unsigned int *)(dp + x - 1 - tpitch);
			dpn_d = *(unsigned int *)(dp + x - 1 + tpitch);		//範囲(Width)を超えてアクセスするので注意

			if (((dp_d & 0x00FC00FC) == 0) && ((dpp_d & 0x00FCFCFC) == 0) && ((dpn_d & 0x00FCFCFC) == 0))	continue;
			//if ( _mm_testz_si128(temp, _mm_set_epi32(0,0x00FCFCFC,0x00FC00FC,0x00FCFCFC)) ) {continue;}	//dpp[x]は判定しないので注意

			dstp[x]=1;	//+=1

			//カレントのdp<=19でスキップ
			//if (dp[x] <= 19) continue;
			if( (dp_d &0x0000ff00) <= 0x00001300 ) continue;

			//8近傍のdp<=19でスキップ
			auto temp = _mm_set_epi32(0, dpn_d, dp_d, dpp_d);	//dpn,dp.dppの順
			temp = _mm_subs_epu8(temp, _mm_set1_epi8(19) );		//unsignedでtemp[]-19  負になる場合は0 ==19以下は0
			temp = _mm_cmpeq_epi8( temp, _mm_setzero_si128() );	//temp[] <=19 ? ff : 0 
			//temp = _mm_min_epu8(temp, _mm_set1_epi8(127));	//本当はepu8で比較したいが、命令がないので、128以上は127に丸める？
			//temp = _mm_sub_epi8(temp, _mm_set1_epi8(20));		//signedでtemp[]-20  負(最上位ビットが立つ)==19以下 でも128以上ある場合に使えない
			int ans = _mm_movemask_epi8( temp );	// temp[] <=19 ? 1 : 0 各バイトの最上位ビットを抽出
			ans =~ans;								// temp[] > 19 ? 1 : 0 上位が1になるので注意
			ans = ans & 0x0757;						//不要部分(dp[x],[x+2])を無視

			int count = _mm_popcnt_u32(ans);			//bitをカウント sse4.2要
			if (count <= 2) continue;

			bool upper  = 0;
			bool upper2 = 0;
			bool lower  = 0;
			bool lower2 = 0;

			upper = ((ans & 0x0007) != 0) ? 1 : 0;	//dpp_dで加算
			lower = ((ans & 0x0700) != 0) ? 1 : 0;	//dpn_dで加算

			if (upper && lower) {					//dpp/dpnで両方加算
				dstp[x] = 3;	//+=2
				continue;
			}
			//以降upper|lowerの片方のみフラグあり

			int start = x - 4;				// b11
			if (start < 0) start = 0;

			int end = x + 5;				//  p3
			if (end > Width) end = Width;	//

			////p4 :
			//if (y > 2) {	//!=
			//	for (int ii=start; ii<end; ii++){
			//		if (dppp[ii] > 19) {
			//			upper2 = 1;  // ??? check others copied from here! todo change upper to upper2 if not upper 2 like in YUY2 part
			//			break;
			//		}
			//	}
			//}

			//if(upper){
			//	for (int ii=start; ii<end; ii++){
			//		if (dpn[ii] > 19){
			//			lower = 1;
			//			break;
			//		}
			//	}
			//}
			//else{	//lower==1
			//	for (int ii=start; ii<end; ii++){
			//		if (dpp[ii] > 19){
			//			upper = 1;
			//			break;
			//		}
			//	}
			//}

			//if (y < Height - 4) {	//!=
			//	for (int ii=start; ii<end; ii++){
			//		if (dpnn[ii] > 19) {
			//			lower2 = 1;
			//			break;
			//		}
			//	}
			//}

			__m128i  msk;
			int size = (end - start);
			//msk.m128i_u64[1] = (size > 8) ? ( ( (long long)1 << (8*(size -8)) ) -1 ) : 0;	//mskの上位 sizeはMAX9byteなので、実は0かffのみにしかならない
			//msk.m128i_u64[1] = (size > 8) ? 255 : 0;	//mskの上位 sizeはMAX9byteなので、実は0かffのみにしかならない
			//msk.m128i_u64[0] = ( ( (long long)1 << (8*size) ) -1 );
			//64bitの演算をするより↓の方が早い気が
			switch(size){
			case 0:				msk = _mm_set_epi32(0,0,0,0);				break;
			case 1:				msk = _mm_set_epi32(0,0,0,0xff);			break;
			case 2:				msk = _mm_set_epi32(0,0,0,0xffff);				break;
			case 3:				msk = _mm_set_epi32(0,0,0,0xffffff);				break;
			case 4:				msk = _mm_set_epi32(0,0,0,0xffffffff);				break;
			case 5:				msk = _mm_set_epi32(0,0,0xff,0xffffffff);				break;
			case 6:				msk = _mm_set_epi32(0,0,0xffff,0xffffffff);				break;
			case 7:				msk = _mm_set_epi32(0,0,0xffffff,0xffffffff);				break;
			case 8:				msk = _mm_set_epi32(0,0,0xffffffff,0xffffffff);				break;
			default:case 9:		msk = _mm_set_epi32(0,0xff,0xffffffff,0xffffffff);				break;
			}

			if (y != 2) {	// >
				temp  = _mm_loadu_si128((__m128i *)(dppp + start));
				temp  = _mm_min_epu8(temp,_mm_set1_epi8(127));	//本当はepu8で比較したいが、命令がないので、128以上は127に丸める？
				temp  = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
				upper2= (_mm_testz_si128(temp,msk)) ? 0 : 1;	//==0で1なので、反転 sete/setneという命令で反転できるから↓の小細工不要？
				//temp = _mm_subs_epu8( temp, _mm_set1_epi8(19) );	//temp[]-19  負になる場合は0 ==19以下は0
				//temp = _mm_cmpeq_epi8( temp, _mm_setzero_si128() );	//temp[] <=19 ? ff : 0    0になる==19以下
				//upper2= _mm_testz_si128(temp,msk);
			}
			if(upper){
				temp   = _mm_loadu_si128((__m128i *)(dpn + start));
				temp  = _mm_min_epu8(temp,_mm_set1_epi8(127));	
				temp  = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
				lower = (_mm_testz_si128(temp,msk)) ? 0 : 1;
			}
			else{	//lower==1
				temp   = _mm_loadu_si128((__m128i *)(dpp + start));
				temp  = _mm_min_epu8(temp,_mm_set1_epi8(127));	
				temp  = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
				upper = (_mm_testz_si128(temp,msk)) ? 0 : 1;
			}
			if (y != Height - 4) {	// <
				temp  = _mm_loadu_si128((__m128i *)(dpnn + start));
				temp  = _mm_min_epu8(temp,_mm_set1_epi8(127));	
				temp  = _mm_cmpgt_epi8(temp, _mm_set1_epi8(19));
				lower2= (_mm_testz_si128(temp,msk)) ? 0 : 1;
			}



			//p13 :
			if (upper == 0) {
				if (lower == 0 || lower2 == 0) {	// p17:
					if (count > 4){
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
					if (count > 4){
						dstp[x] = 5;	//+=4
					}
				}
			}
		}
		dppp += tpitch;
		dpp += tpitch;
		dp += tpitch;
		dpn += tpitch;
		dpnn += tpitch;
		dstp += dst_pitch;
	}
}

#pragma warning(pop)	// reenable no emms warning