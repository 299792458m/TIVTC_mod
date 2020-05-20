#ifndef __FIXEDFONTS_H__
#define __FIXEDFONTS_H__

typedef struct {
  const char* const fontname;
  const char* const fontname_internal;
  int charcount;
  int width;
  int height;
  bool bold;
} FixedFont_info_t;

constexpr int PREDEFINED_FONT_COUNT = 2; // 2xinfo_h

#endif  // __FIXEDFONTS_H__
