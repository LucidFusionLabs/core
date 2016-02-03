/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "lfapp/lfapp.h"
#include "lfapp/string.h"

#ifdef LFL_ICONV
#include <iconv.h>
#endif

#ifdef LFL_REGEX
#include "regexp.h"
#endif

extern "C" {
#ifdef LFL_SREGEX
#include "sregex.h"
#endif
};

#define StringPrintfImpl(ret, fmt, vsprintf, offset) \
  (ret)->resize(offset + 4096); \
  va_list ap, ap2; \
  va_start(ap, fmt); \
  va_copy(ap2, ap); \
  int len = -1; \
  for (int i=0; len < 0 || len >= (ret)->size()-1; ++i) { \
    if (i) { \
      va_copy(ap, ap2); \
      (ret)->resize((ret)->size() * 2); \
    } \
    len = vsprintf(&(*ret)[0] + offset, (ret)->size(), fmt, ap); \
    va_end(ap); \
  } \
  va_end(ap2); \
  (ret)->resize(offset + len);

extern "C" void LFAppLog(int level, const char *file, int line, const char *fmt, ...) {
  string message;
  StringPrintfImpl(&message, fmt, vsnprintf, 0);
  LFL::app->Log(level, file, line, message);
}

namespace LFL {
::std::ostream& operator<<(::std::ostream& os, const point &x) { return os << x.DebugString(); }
::std::ostream& operator<<(::std::ostream& os, const Box   &x) { return os << x.DebugString(); }

Printable::Printable(const pair<int, int> &x) : string(StrCat("pair(", x.first, ", ", x.second, ")")) {}
Printable::Printable(const vector<string> &x) : string(StrCat("{", Vec<string>::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<double> &x) : string(StrCat("{", Vec<double>::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<float>  &x) : string(StrCat("{", Vec<float> ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<int>    &x) : string(StrCat("{", Vec<int>   ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const Color          &x) : string(x.DebugString()) {}
Printable::Printable(const String16       &x) : string(String::ToUTF8(x)) {}
Printable::Printable(const void           *x) : string(StringPrintf("%p", x)) {}

#ifdef  LFL_UNICODE_DEBUG
#define UnicodeDebug(...) ERROR(__VA_ARGS__)
#else
#define UnicodeDebug(...)
#endif

#ifdef LFL_ICONV
template <class X, class Y>
int String::Convert(const StringPieceT<X> &in, basic_string<Y> *out, const char *from, const char *to) {
  iconv_t cd = iconv_open(to, from);
  if (cd < 0) { ERROR("failed convert ", from, " to ", to); out->clear(); return 0; }

  out->resize(in.len*4/sizeof(Y)+4);
  char *inp = (char*)in.buf, *top = (char*)out->data();
  size_t in_remaining = in.len*sizeof(X), to_remaining = out->size()*sizeof(Y);
  if (iconv(cd, &inp, &in_remaining, &top, &to_remaining) == -1)
  { /* ERROR("failed convert ", from, " to ", to); */ iconv_close(cd); out->clear(); return 0; }
  out->resize(out->size() - to_remaining/sizeof(Y));
  iconv_close(cd);

  return in.len - in_remaining/sizeof(X);
}
#else /* LFL_ICONV */
template <class X, class Y>
int String::Convert(const StringPieceT<X> &in, basic_string<Y> *out, const char *from, const char *to) {
  if (!strcmp(from, to)) { String::Copy(in, out); return in.len; }
#ifdef WIN32
  if (!strcmp(from, "UTF-16LE") && !strcmp(to, "UTF-8")) {
    out->resize(WideCharToMultiByte(CP_UTF8, 0, (wchar_t*)in.data(), in.size(), NULL, 0, NULL, NULL));
    WideCharToMultiByte(CP_UTF8, 0, (wchar_t*)in.data(), in.size(), (char*)&(*out)[0], out->size(), NULL, NULL);
    return in.len;
  }
#endif
  ONCE(ERROR("conversion from ", from, " to ", to, " not supported.  copying.  #define LFL_ICONV"));
  String::Copy(in, out);
  return in.len;    
}
#endif /* LFL_ICONV */

template int String::Convert<char,     char    >(const StringPiece  &, string  *, const char*, const char*);
template int String::Convert<char,     char16_t>(const StringPiece  &, String16*, const char*, const char*);
template int String::Convert<char16_t, char    >(const String16Piece&, string  *, const char*, const char*);
template int String::Convert<char16_t, char16_t>(const String16Piece&, String16*, const char*, const char*);

String16 String::ToUTF16(const StringPiece &text, int *consumed) {
  int input = text.Length(), output = 0, c_bytes, c;
  String16 ret;
  ret.resize(input);
  const char *b = text.data(), *p = b;
  for (; !text.Done(p); p += c_bytes, output++) {
    ret[output] = UTF8::ReadGlyph(text, p, &c_bytes);
    if (!c_bytes) break;
  }
  CHECK_LE(output, input);
  if (consumed) *consumed = p - b;
  ret.resize(output);
  return ret;
}

String16 UTF16::WriteGlyph(int codepoint) { return String16(1, codepoint); }
int UTF16::ReadGlyph(const String16Piece &s, const char16_t *p, int *len, bool eof) { *len=1; return *p; }
int UTF8 ::ReadGlyph(const StringPiece   &s, const char     *p, int *len, bool eof) {
  *len = 1;
  unsigned char c0 = *(const unsigned char *)p;
  if ((c0 & (1<<7)) == 0) return c0; // ascii
  if ((c0 & (1<<6)) == 0) { UnicodeDebug("unexpected continuation byte"); return c0; }
  for ((*len)++; *len < 4; (*len)++) {
    if (s.Done(p + *len - 1)) { UnicodeDebug("unexpected end of string"); *len=eof; return c0; }
    if ((c0 & (1<<(7 - *len))) == 0) break;
  }

  int ret = 0;
  if      (*len == 2) ret = c0 & 0x1f;
  else if (*len == 3) ret = c0 & 0x0f;
  else if (*len == 4) ret = c0 & 0x07;
  else { UnicodeDebug("invalid len ", *len); *len=1; return c0; }

  for (int i = *len; i > 1; i--) {
    unsigned char c = *(const unsigned char *)++p;
    if ((c & 0xc0) != 0x80) { UnicodeDebug("unexpected non-continuation byte"); *len=1; return c0; }
    ret = (ret << 6) | (c & 0x3f);
  }
  return ret;
}
string UTF8::WriteGlyph(int codepoint) {
#if 1
  string out;
  char16_t in[] = { (char16_t)codepoint, 0 };
  String::Convert(String16Piece(in, 1), &out, "UTF-16LE", "UTF-8");
  return out;
#else
#endif
}

int isfileslash(int c) { return c == LocalFile::Slash; }
int MatchingParens(int c1, int c2) { return (c1 == '(' && c2 == ')') || (c1 == '[' && c2 == ']') || (c1 == '<' && c2 == '>'); }
float my_atof(const char *v) { return v ? ::atof(v) : 0; }
int atoi(const char *v) { return v ? ::atoi(v) : 0; }
int atoi(const char16_t *v) {
  const char16_t *p;
  if (!v) return 0; int ret = 0;
  if (!(p = FindChar(v, notspace))) return 0;
  bool neg = *p == '-';
  for (p += neg; *p >= '0' && *p <= '9'; p++) ret = ret*10 + *p - '0';
  return neg ? -ret : ret;
}

unsigned fnv32(const void *buf, unsigned len, unsigned hval) {
  if (!len) len = strlen((const char *)buf);
  unsigned char *bp = (unsigned char *)buf, *be = bp + len;
  while (bp < be) {
    hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
    hval ^= (unsigned)*bp++;
  }
  return hval;
}

unsigned long long fnv64(const void *buf, unsigned len, unsigned long long hval) {
  if (!len) len = strlen((const char *)buf);
  unsigned char *bp = (unsigned char *)buf, *be = bp + len;
  while (bp < be) {
    hval += (hval<<1) + (hval<<4) + (hval<<5) + (hval<<7) + (hval<<8) + (hval<<40);
    hval ^= (unsigned long long)*bp++;
  }
  return hval;
}

int sprint(char *out, int len, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = vsnprintf(out,len,fmt,ap);
  if (ret > len) ret = len;
  va_end(ap);
  return ret;
}

void StringAppendf(string *out, const char *fmt, ...) {
  int offset = out->size();
  StringPrintfImpl(out, fmt, vsnprintf, offset);
}
void StringAppendf(String16 *uc_out, const char *fmt, ...) {
  string outs, *out = &outs;
  StringPrintfImpl(out, fmt, vsnprintf, 0);
  String::Append(outs, uc_out);
}

string StringPrintf(const char *fmt, ...) {
  string ret;
  StringPrintfImpl(&ret, fmt, vsnprintf, 0);
  return ret;
}
String16 String16Printf(const char *fmt, ...) {
  string ret;
  StringPrintfImpl(&ret, fmt, vsnprintf, 0);
  return String::ToUTF16(ret);
}
basic_string<wchar_t> WStringPrintf(const wchar_t *fmt, ...) {
  basic_string<wchar_t> ret;
  StringPrintfImpl(&ret, fmt, vswprintf, 0);
  return ret;
}

#define StrCatInit(s) string out; out.resize(s); Serializable::MutableStream o(&out[0], out.size());
#define StrCatAdd(x) memcpy(o.Get(x.size()), x.data(), x.size())
#define StrCatReturn() CHECK_EQ(o.error, 0); return out;
string StrCat(const Printable &x1, const Printable &x2) { StrCatInit(x1.size()+x2.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3) { StrCatInit(x1.size()+x2.size()+x3.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13, const Printable &x14) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()+x14.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrCatAdd(x14); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13, const Printable &x14, const Printable &x15) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()+x14.size()+x15.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrCatAdd(x14); StrCatAdd(x15); StrCatReturn(); }

#define StrAppendInit(s); out->resize(out->size()+(s)); Serializable::MutableStream o(&(*out)[0]+out->size()-(s), (s));
#define StrAppendReturn() CHECK_EQ(o.error, 0)
void StrAppend(string *out, const Printable &x1) { (*out) += x1; }
void StrAppend(string *out, const Printable &x1, const Printable &x2) { StrAppendInit(x1.size()+x2.size()); StrCatAdd(x1); StrCatAdd(x2); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3) { StrAppendInit(x1.size()+x2.size()+x3.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrAppendReturn(); }

void StrAppend(String16 *out, const Printable &x1) { String::Append(StrCat(x1), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2) { String::Append(StrCat(x1,x2), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3) { String::Append(StrCat(x1,x2,x3), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4) { String::Append(StrCat(x1,x2,x3,x4), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5) { String::Append(StrCat(x1,x2,x3,x4,x5), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6) { String::Append(StrCat(x1,x2,x3,x4,x5,x6), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13), out); }

template <class X, class Y> bool PrefixMatch(const X* in, const Y* pref, int cs) {
  while (*in && *pref &&
         ((cs && *in == *pref) || 
          (!cs && ::tolower(*in) == ::tolower(*pref)))) { in++; pref++; }
  return !*pref;
}
bool PrefixMatch(const char     *in, const string   &pref, int cs) { return PrefixMatch<char,     char>    (in,         pref.c_str(), cs); }
bool PrefixMatch(const string   &in, const char     *pref, int cs) { return PrefixMatch<char,     char>    (in.c_str(), pref,         cs); }
bool PrefixMatch(const string   &in, const string   &pref, int cs) { return PrefixMatch<char,     char>    (in.c_str(), pref.c_str(), cs); }
bool PrefixMatch(const String16 &in, const String16 &pref, int cs) { return PrefixMatch<char16_t, char16_t>(in.c_str(), pref.c_str(), cs); }
bool PrefixMatch(const String16 &in, const char     *pref, int cs) { return PrefixMatch<char16_t, char>    (in.c_str(), pref, cs); }
bool PrefixMatch(const char     *in, const char     *pref, int cs) { return PrefixMatch<char,     char>    (in, pref, cs); }
bool PrefixMatch(const char16_t *in, const char16_t *pref, int cs) { return PrefixMatch<char16_t, char16_t>(in, pref, cs); }

template <class X, class Y>
bool SuffixMatch(const X *in, int inlen, const Y *pref, int preflen, int cs) {
  if (inlen < preflen) return 0;
  const X *in_suffix = in + inlen - preflen;
  for (in += inlen-1, pref += preflen-1;
       in >= in_suffix &&
       ((cs && *in == *pref) ||
        (!cs && ::tolower(*in) == ::tolower(*pref))); in--, pref--) {}
  return in < in_suffix;
}
bool SuffixMatch(const char16_t *in, const char16_t *pref, int cs) { return SuffixMatch(String16(in), String16(pref), cs); }
bool SuffixMatch(const char     *in, const char     *pref, int cs) { return SuffixMatch(string(in),   string(pref),   cs); }
bool SuffixMatch(const char     *in, const string   &pref, int cs) { return SuffixMatch(string(in),   pref,           cs); }
bool SuffixMatch(const string   &in, const char     *pref, int cs) { return SuffixMatch(in,           string(pref),   cs); }
bool SuffixMatch(const string   &in, const string   &pref, int cs) { return SuffixMatch<char,     char    >(in.data(), in.size(), pref.data(), pref.size(), cs); }
bool SuffixMatch(const String16 &in, const String16 &pref, int cs) { return SuffixMatch<char16_t, char16_t>(in.data(), in.size(), pref.data(), pref.size(), cs); }
bool SuffixMatch(const String16 &in, const string   &pref, int cs) { return SuffixMatch<char16_t, char    >(in.data(), in.size(), pref.data(), pref.size(), cs); }

template <class X, class Y>
bool StringEquals(const X *s1, const Y *s2, int cs) {
  while (*s1 && *s2 &&
         ((cs && *s1 == *s2) ||
          (!cs && ::tolower(*s1) == ::tolower(*s2)))) { s1++; s2++; }
  return !*s1 && !*s2;
}
bool StringEquals(const String16 &s1, const String16 &s2, int cs) { return s1.size() == s2.size() && StringEquals(s1.c_str(), s2.c_str(), cs); }
bool StringEquals(const string   &s1, const string   &s2, int cs) { return s1.size() == s2.size() && StringEquals(s1.c_str(), s2.c_str(), cs); }
bool StringEquals(const string   &s1, const char     *s2, int cs) { return                           StringEquals(s1.c_str(), s2,         cs); }
bool StringEquals(const char     *s1, const string   &s2, int cs) { return                           StringEquals(s1,         s2.c_str(), cs); }
bool StringEquals(const char     *s1, const char     *s2, int cs) { return cs ? !strcmp(s1, s2) : !strcasecmp(s1, s2); }
bool StringEquals(const char16_t *s1, const char16_t *s2, int cs) { return StringEquals<char16_t, char16_t>(s1, s2, cs); }
bool StringEquals(const String16 &s1, const char     *s2, int cs) { return StringEquals<char16_t, char    >(s1.c_str(), s2, cs); }

bool StringEmptyOrEquals(const string   &cmp, const string   &ref, int cs) { return cmp.empty() || StringEquals(cmp, ref, cs); }
bool StringEmptyOrEquals(const String16 &cmp, const String16 &ref, int cs) { return cmp.empty() || StringEquals<char16_t, char16_t>(cmp.c_str(), ref.c_str(), cs); }
bool StringEmptyOrEquals(const String16 &cmp, const string   &ref, int cs) { return cmp.empty() || StringEquals<char16_t, char    >(cmp.c_str(), ref.c_str(), cs); }
bool StringEmptyOrEquals(const string   &cmp, const string   &ref1, const string   &ref2, int cs) { return cmp.empty() || StringEquals                    (cmp,         ref1,         cs) || StringEquals                    (cmp,         ref2,         cs); }
bool StringEmptyOrEquals(const String16 &cmp, const String16 &ref1, const String16 &ref2, int cs) { return cmp.empty() || StringEquals<char16_t, char16_t>(cmp.c_str(), ref1.c_str(), cs) || StringEquals<char16_t, char16_t>(cmp.c_str(), ref2.c_str(), cs); }
bool StringEmptyOrEquals(const String16 &cmp, const string   &ref1, const string   &ref2, int cs) { return cmp.empty() || StringEquals<char16_t, char    >(cmp.c_str(), ref1.c_str(), cs) || StringEquals<char16_t, char    >(cmp.c_str(), ref2.c_str(), cs); }

#define FindCharLoopImpl(ischar_p, deref_p) \
  if (!in_quote && (ischar_p)) { ret=p; break; } \
  if (isquotec && isquotec(deref_p)) in_quote = !in_quote;

#define FindCharImpl(type, ischar_p, deref_p, check_p) \
  const type *ret=0, *p=text; \
  bool have_len = len >= 0, in_quote = false; \
  if (have_len) { for (const type *e = p+len; p != e;  ++p) { FindCharLoopImpl(ischar_p, deref_p); } } \
  else          { for (;                      check_p; ++p) { FindCharLoopImpl(ischar_p, deref_p); } } \
  if (outlen) *outlen = ret ? ret-text : p-text; \
  return ret;

template <class X> const X *FindChar(const X *text, int ischar, int (*isquotec)(int), int len, int *outlen) {
  FindCharImpl(X, (ischar == *p), *p, *p);
}
template <class X>       X *FindChar(      X *text, int ischar, int (*isquotec)(int), int len, int *outlen) { return (X*)FindChar((const X *)text, ischar, isquotec, len, outlen); }
template <class X> const X *FindChar(const X *text, int ischar, int len, int *outlen) { return FindChar(text, ischar, 0, len, outlen); }
template <class X>       X *FindChar(      X *text, int ischar, int len, int *outlen) { return (X*)FindChar((const X *)text, ischar, len, outlen); }

template <class X> const X *FindChar(const X *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) {
  FindCharImpl(X, ischar(*p), *p, *p);
}
template <class X>       X *FindChar(      X *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) { return (X*)FindChar((const X *)text, ischar, isquotec, len, outlen); }
template <class X> const X *FindChar(const X *text, int (*ischar)(int), int len, int *outlen) { return FindChar(text, ischar, 0, len, outlen); }
template <class X>       X *FindChar(      X *text, int (*ischar)(int), int len, int *outlen) { return (X*)FindChar((const X *)text, ischar, len, outlen); }

template <> const DrawableBox *FindChar(const DrawableBox *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) {
  FindCharImpl(DrawableBox, ischar(p->Id()), p->Id(), 1);
}
template <>       DrawableBox *FindChar(      DrawableBox *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) { return (DrawableBox*)FindChar((const DrawableBox *)text, ischar, isquotec, len, outlen); }
template <> const DrawableBox *FindChar(const DrawableBox *text, int (*ischar)(int), int len, int *outlen) { return FindChar(text, ischar, 0, len, outlen); }
template <>       DrawableBox *FindChar(      DrawableBox *text, int (*ischar)(int), int len, int *outlen) { return (DrawableBox*)FindChar((const DrawableBox *)text, ischar, len, outlen); }

template       char*        FindChar<char    >   (      char*,        int (*)(int), int, int*);
template const char*        FindChar<char    >   (const char*,        int (*)(int), int, int*);
template       char16_t*    FindChar<char16_t>   (      char16_t*,    int (*)(int), int, int*);
template const char16_t*    FindChar<char16_t>   (const char16_t*,    int (*)(int), int, int*);
template       DrawableBox* FindChar<DrawableBox>(      DrawableBox*, int (*)(int), int, int*);
template const DrawableBox* FindChar<DrawableBox>(const DrawableBox*, int (*)(int), int, int*);

#define LengthCharImpl(in, len, p, deref_p, check_p) \
  if (len >= 0) while (p-in < len &&  ischar(deref_p)) p++; \
  else          while (check_p    &&  ischar(deref_p)) p++;

#define LengthNotCharImpl(in, len, p, deref_p, check_p) \
  if (len >= 0) while (p-in < len && !ischar(deref_p)) p++; \
  else          while (check_p    && !ischar(deref_p)) p++;

#define LengthCharFunctionImpl(type, deref_p, check_p) \
  const type *p = in; \
  LengthCharImpl(in, len, p, deref_p, check_p); \
  return p - in;

template <class X> int LengthChar(const X *in, int (*ischar)(int), int len) {
  LengthCharFunctionImpl(X, *p, *p);
}
template <> int LengthChar(const DrawableBox *in, int (*ischar)(int), int len) {
  LengthCharFunctionImpl(DrawableBox, p->Id(), 1);
}
template int LengthChar(const char*,        int(*)(int), int);
template int LengthChar(const char16_t*,    int(*)(int), int);
template int LengthChar(const DrawableBox*, int(*)(int), int);

#define RLengthCharFunctionImpl(type, deref_p) \
  if (len <= 0) return 0; \
  const type *p = in, *e = in - len; \
  for (; p != e; --p) if (!ischar(deref_p)) break; \
  return in - p;

template <class X> int RLengthChar(const X *in, int (*ischar)(int), int len) {
  RLengthCharFunctionImpl(X, *p);
}
template <> int RLengthChar(const DrawableBox *in, int (*ischar)(int), int len) {
  RLengthCharFunctionImpl(DrawableBox, p->Id());
}
template int RLengthChar(const char*,        int(*)(int), int);
template int RLengthChar(const char16_t*,    int(*)(int), int);
template int RLengthChar(const DrawableBox*, int(*)(int), int);

int Split(const StringPiece &in, int (*ischar)(int), string *left) {
  const char *p = in.buf;
  LengthNotCharImpl(in.buf, in.len, p, *p, *p);
  left->assign(in.buf, p-in.buf);
  return 1;
}

int Split(const StringPiece &in, int (*ischar)(int), string *left, string *right) {
  const char *p = in.buf;
  LengthNotCharImpl(in.buf, in.len, p, *p, *p);
  left->assign(in.buf, p-in.buf);
  if (in.Done(p)) { *right=""; return 1; }

  LengthCharImpl(in.buf, in.len, p, *p, *p);
  if (in.Done(p)) { *right=""; return 1; }
  right->assign(p);
  return 2;
}

void Join(string *out, const vector<string> &in) { return StrAppend(out, in, 0, in.size()); }
void Join(string *out, const vector<string> &in, int inB, int inE) {
  int size = 0;        for (int i = inB; i < inE; i++) size += in[i].size();
  StrAppendInit(size); for (int i = inB; i < inE; i++) { StrCatAdd(in[i]); } StrAppendReturn();
}
string Join(const vector<string> &strs, const string &separator) {
  string ret;
  for (vector<string>::const_iterator i = strs.begin(); i != strs.end(); i++) StrAppend(&ret, ret.size()?separator:"", *i);
  return ret;
}
string Join(const vector<string> &strs, const string &separator, int beg_ind, int end_ind) {
  string ret;
  for (int i = beg_ind; i < strs.size() && i < end_ind; i++) StrAppend(&ret, ret.size()?separator:"", strs[i]);
  return ret;
}

string strip(const char *s, int (*stripchar1)(int), int (*stripchar2)(int)) {
  string ret;
  for (/**/; *s; s++) if ((!stripchar1 || !stripchar1(*s)) && (!stripchar2 || !stripchar2(*s))) ret += *s;
  return ret;
}

string togrep(const char *s, int (*grepchar1)(int), int (*grepchar2)(int)) {
  string ret;
  for (/**/; *s; s++) if ((!grepchar1 || grepchar1(*s)) || (!grepchar2 || grepchar2(*s))) ret += *s;
  return ret;
}

template <class X>
basic_string<X> toconvert(const X *s, int (*tochar)(int), int (*ischar)(int)) {
  basic_string<X> input = s;
  for (int i=0; i<input.size(); i++)
    if (!ischar || ischar(input[i]))
      input[i] = tochar(input[i]);

  return input;
}
string toconvert  (const string   &s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<char>    (s.c_str(), tochar, ischar); }
string toconvert  (const char     *s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<char>    (s,         tochar, ischar); }
String16 toconvert(const String16 &s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<char16_t>(s.c_str(), tochar, ischar); }
String16 toconvert(const char16_t *s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<char16_t>(s,         tochar, ischar); }

string   toupper(const char     *s) { return toconvert(s        , ::toupper, isalpha); }
string   toupper(const string   &s) { return toconvert(s.c_str(), ::toupper, isalpha); }
String16 toupper(const char16_t *s) { return toconvert(s        , ::toupper, isalpha); }
String16 toupper(const String16 &s) { return toconvert(s.c_str(), ::toupper, isalpha); }

string   tolower(const char     *s) { return toconvert(s        , ::tolower, isalpha); }
string   tolower(const string   &s) { return toconvert(s.c_str(), ::tolower, isalpha); }
String16 tolower(const char16_t *s) { return toconvert(s        , ::tolower, isalpha); }
String16 tolower(const String16 &s) { return toconvert(s.c_str(), ::tolower, isalpha); }

string   ReplaceEmpty (const string   &in, const string   &replace_with) { return in.empty() ? replace_with : in; }
String16 ReplaceEmpty (const String16 &in, const String16 &replace_with) { return in.empty() ? replace_with : in; }
String16 ReplaceEmpty (const String16 &in, const string   &replace_with) { return in.empty() ? String::ToUTF16(replace_with) : in; }

string ReplaceNewlines(const string   &in, const string   &replace_with) {
  string ret;
  for (const char *p = in.data(); p-in.data() < in.size(); p++) {
    if (*p == '\r' && *(p+1) == '\n') { ret += replace_with; p++; }
    else if (*p == '\n') ret += replace_with;
    else ret += string(p, 1);
  }
  return ret;
}

bool ReplaceString(string *text, const string &needle, const string &replace) {
  int pos = text->find(needle);
  if (pos == string::npos) return false;
  text->erase(pos, needle.size());
  text->insert(pos, replace);
  return true;
}

template <class X> string CHexEscape(const basic_string<X> &text) {
  string ret;
  ret.reserve(text.size()*4);
  for (typename make_unsigned<X>::type c : text) StringAppendf(&ret, "\\x%02x", c);
  return ret;
}

template <class X> string CHexEscapeNonAscii(const basic_string<X> &text) {
  string ret;
  ret.reserve(text.size()*4);
  for (typename make_unsigned<X>::type c : text)
    if (isascii(c)) ret += c;
    else StringAppendf(&ret, "\\x%02x", c);
  return ret;
}

template string CHexEscape        (const string   &);
template string CHexEscape        (const String16 &);
template string CHexEscapeNonAscii(const string   &);
template string CHexEscapeNonAscii(const String16 &);

string FirstMatchCSV(const StringPiece &haystack, const StringPiece &needle, int (*ischar)(int)) {
  unordered_set<string> h_map;
  StringWordIter h_words(haystack, ischar);
  StringWordIter i_words(needle,   ischar);
  for (string w = IterNextString(&h_words); !h_words.Done(); w = IterNextString(&h_words)) h_map.insert(w);
  for (string w = IterNextString(&i_words); !i_words.Done(); w = IterNextString(&i_words)) if (Contains(h_map, w)) return w;
  return "";
}

bool ParseKV(const string &t, string *k_out, string *v_out, int equal_char) {
  auto p = FindChar(t.c_str(), equal_char);
  if (!p) return false;
  ptrdiff_t p_offset = p - t.c_str();
  StringWordIter k_words(t.c_str(), p_offset), v_words(p+1, t.size()-p_offset-1);
  *k_out = IterNextString(&k_words);
  *v_out = IterRemainingString(&v_words);
  return k_out->size();
}

string UpdateKVLine(const string &haystack, const string &key, const string &val, int equal_char) {
  bool found = 0;
  StringLineIter lines(haystack);
  string ret, k, v, update = StrCat(key, string(1, equal_char), val);
  for (string line = IterNextString(&lines); !lines.Done(); line = IterNextString(&lines)) {
    if (ParseKV(line, &k, &v, ':') && k == key && (found=1)) StrAppend(&ret, update, "\n");
    else                                                     StrAppend(&ret, line,   "\n");
  }
  return found ? ret : StrCat(ret, update, "\n");
}

template <class X, bool chomp> const X *NextLine(const StringPieceT<X> &text, bool final, int *outlen) {
  const X *ret=0, *p = text.buf;
  for (/**/; !text.Done(p); ++p) { 
    if (*p == '\n') { ret = p+1; break; }
  }
  if (!ret) { if (outlen) *outlen = p - text.buf; return final ? text.buf : 0; }
  if (outlen) {
    int ol = ret-text.buf-1;
    if (chomp && ret-2 >= text.buf && *(ret-2) == '\r') ol--;
    *outlen = ol;
  }
  return ret;
}
const char     *NextLine   (const StringPiece   &text, bool final, int *outlen) { return NextLine<char,     true >(text, final, outlen); }
const char16_t *NextLine   (const String16Piece &text, bool final, int *outlen) { return NextLine<char16_t, true >(text, final, outlen); }
const char     *NextLineRaw(const StringPiece   &text, bool final, int *outlen) { return NextLine<char,     false>(text, final, outlen); }
const char16_t *NextLineRaw(const String16Piece &text, bool final, int *outlen) { return NextLine<char16_t, false>(text, final, outlen); }

const char *NextProto(const StringPiece &text, bool final, int *outlen) {
  if (text.len < ProtoHeader::size) return 0;
  ProtoHeader hdr(text.buf);
  if (ProtoHeader::size + hdr.len > text.len) return 0;
  *outlen = hdr.len;
  return text.buf + ProtoHeader::size + hdr.len;
}

template <class X> 
StringWordIterT<X>::StringWordIterT(const X *B, int S, int (*delim)(int), int (*quote)(int), int F)
  : in(B, S), IsSpace(delim ? delim : isspace), IsQuote(quote), flag(F) {
  if (!in.buf || !in.len) next_offset  = -1;
  else                    next_offset += LengthChar(in.buf + cur_offset, IsSpace, in.Remaining(cur_offset));
}

template <class X> const X *StringWordIterT<X>::Next() {
  cur_offset = next_offset;
  if (cur_offset < 0) return 0;
  const X *word = in.buf + cur_offset, *next = FindChar(word, IsSpace, IsQuote, in.Remaining(cur_offset), &cur_len);
  if ((next_offset = next ? next - in.buf : -1) >= 0) {
    next_offset += LengthChar(in.buf + next_offset, IsSpace, in.Remaining(next_offset));
    if (in.Done(in.buf + next_offset)) next_offset = -1;
  }
  return cur_len ? word : 0;
}

template <class X> const X *StringLineIterT<X>::Next() {
  first = false;
  for (cur_offset = next_offset; cur_offset >= 0; cur_offset = next_offset) {
    const X *line = in.buf + cur_offset, *next = NextLine(StringPieceT<X>(line, in.Remaining(cur_offset)), false, &cur_len);
    next_offset = next ? next - in.buf : -1;
    if (cur_len) cur_len -= ChompNewlineLength(line, cur_len);
    if (cur_len || ((flag & Flag::BlankLines) && next)) return line;
  }
  return 0;
}

template struct StringWordIterT<char>;
template struct StringLineIterT<char>;
template struct StringWordIterT<char16_t>;
template struct StringLineIterT<char16_t>;
template struct StringWordIterT<DrawableBox>;

const char *IterWordIter::Next() {
  if (!iter) return 0;
  const char *w = word.in.buf ? word.Next() : 0;
  while (!w) {
    first_count++;
    const char *line = iter->Next();
    if (!line) return 0;
    word = StringWordIter(line, iter->CurrentLength(), word.IsSpace);
    w = word.Next();
  }
  return w;
}    

template <class X> int IsNewline(const X *line) {
  if (!*line) return 0;
  if (*line == '\n') return 1;
  if (*line == '\r' && *(line+1) == '\n') return 2;
  return 0;
}
template int IsNewline(const char *line);

template <class X> int ChompNewline(X *line, int len) {
  int ret = 0;
  if (line[len-1] == '\n') { line[len-1] = 0; ret++; }
  if (line[len-2] == '\r') { line[len-2] = 0; ret++; }
  return ret;
}
template int ChompNewline(char *line, int len);

template <class X> int ChompNewlineLength(const X *line, int len) {
  int ret = 0;
  if (line[len-1] == '\n') ret++;
  if (line[len-2] == '\r') ret++;
  return ret;
}

const char *Default(const char *in, const char *default_in) { return (in && in[0]) ? in : default_in; }

const char *ParseProtocol(const char *url, string *protO) {
  static const int hdr_size = 3;
  static const char hdr[] = "://";
  const char *prot_end = strstr(url, hdr), *prot, *host;
  if (prot_end) { prot = url; host = prot_end + hdr_size; }
  else          { prot = 0;   host = url;                 }
  while (prot && *prot && isspace(*prot)) prot++;
  while (host && *host && isspace(*host)) host++;
  if (protO) protO->assign(prot ? prot : "", prot ? prot_end-prot : 0);
  return host;
}

const char *BaseName(const StringPiece &path, int *outlen) {
  const char *ret = path.buf;
  int len = path.Length();
  for (const char *p = path.buf+len-1; p > path.buf; --p) if (isfileslash(*p)) { ret=p+1; break;}
  if (outlen) {
    int namelen = len - (ret-path.buf), baselen;
    FindChar(ret, isdot, namelen, &baselen);
    *outlen = baselen ? baselen : namelen;
  }
  return ret;
}

int BaseDir(const char *path, const char *cmp) {
  int l1=strlen(path), l2=strlen(cmp), slash=0, s1=-1, s2=0;
  if (!l1 || l2 > l1) return 0;
  if (*(path+l1-1) == '/') return 0;
  for (const char *p=path+l1-1; p>=path; p--) if (isfileslash(*p)) {
    slash++; /* count bacwards */
    if (slash == 1) s2 = p-path;
    if (slash == 2) { s1 = p-path; break; }
  }
  if (slash < 1 || s2-s1-1 != l2) return 0;
  return !strncasecmp(&path[s1]+1, cmp, l2);
}

template <class X> int DirNameLen(const StringPieceT<X> &path, bool include_slash) {
  int len = path.Length();
  const X *start = path.buf + len - 1, *slash = 0;
  for (const X *p = start; p > path.buf; --p) if (isfileslash(*p)) { slash=p; break; }
  return !slash ? 0 : len - (start - slash + !include_slash);
}
int DirNameLen(const StringPiece   &text, bool include_slash) { return DirNameLen<char>    (text, include_slash); }
int DirNameLen(const String16Piece &text, bool include_slash) { return DirNameLen<char16_t>(text, include_slash); }

Base64::Base64() : encoding_table("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"), decoding_table(256, 0) {
  mod_table[0]=0; mod_table[1]=2; mod_table[2]=1;
  for (int i = 0; i < 64; i++) decoding_table[(unsigned char)encoding_table[i]] = i;
}

string Base64::Encode(const char *in, size_t input_length) {
  const unsigned char *data = (const unsigned char *) in;
  string encoded_data(4 * ((input_length + 2) / 3), 0);
  for (int i = 0, j = 0; i < input_length;) {
    unsigned octet_a = i < input_length ? data[i++] : 0;
    unsigned octet_b = i < input_length ? data[i++] : 0;
    unsigned octet_c = i < input_length ? data[i++] : 0;
    unsigned triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;
    encoded_data[j++] = encoding_table[(triple >> 3 * 6) & 0x3F];
    encoded_data[j++] = encoding_table[(triple >> 2 * 6) & 0x3F];
    encoded_data[j++] = encoding_table[(triple >> 1 * 6) & 0x3F];
    encoded_data[j++] = encoding_table[(triple >> 0 * 6) & 0x3F];
  }
  for (int i = 0; i < mod_table[input_length % 3]; i++) encoded_data[encoded_data.size() - 1 - i] = '=';
  return encoded_data;
}

string Base64::Decode(const char *data, size_t input_length) {
  CHECK_EQ(input_length % 4, 0);
  string decoded_data(input_length / 4 * 3, 0);
  if (data[input_length - 1] == '=') decoded_data.erase(decoded_data.size()-1);
  if (data[input_length - 2] == '=') decoded_data.erase(decoded_data.size()-1);
  for (int i = 0, j = 0; i < input_length;) {
    unsigned sextet_a = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
    unsigned sextet_b = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
    unsigned sextet_c = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
    unsigned sextet_d = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
    unsigned triple = (sextet_a << 3 * 6) + (sextet_b << 2 * 6) + (sextet_c << 1 * 6) + (sextet_d << 0 * 6);
    if (j < decoded_data.size()) decoded_data[j++] = (triple >> 2 * 8) & 0xFF;
    if (j < decoded_data.size()) decoded_data[j++] = (triple >> 1 * 8) & 0xFF;
    if (j < decoded_data.size()) decoded_data[j++] = (triple >> 0 * 8) & 0xFF;
  }
  return decoded_data;
}

#ifdef LFL_REGEX
Regex::~Regex() { re_free((regexp*)impl); }
Regex::Regex(const string &patternstr) {
  regexp* compiled = 0;
  if (!re_comp(&compiled, patternstr.c_str())) impl = compiled;
}
int Regex::Match(const string &text, vector<Regex::Result> *out) {
  if (!impl) return -1;
  regexp* compiled = (regexp*)impl;
  vector<regmatch> matches(re_nsubexp(compiled));
  int retval = re_exec(compiled, text.c_str(), matches.size(), &matches[0]);
  if (retval < 1) return retval;
  if (out) for (auto i : matches) out->emplace_back(i.begin, i.end);
  return 1;
}
#else
Regex::~Regex() {}
Regex::Regex(const string &patternstr) {}
int Regex::Match(const string &text, vector<Regex::Result> *out) { ERROR("regex not implemented"); return 0; }
#endif

#ifdef LFL_SREGEX
StreamRegex::~StreamRegex() {
  if (ppool) sre_destroy_pool((sre_pool_t*)ppool);
  if (cpool) sre_destroy_pool((sre_pool_t*)cpool);
}
StreamRegex::StreamRegex(const string &patternstr) : ppool(sre_create_pool(1024)), cpool(sre_create_pool(1024)) {
  sre_uint_t ncaps;
  sre_int_t err_offset = -1;
  sre_regex_t *re = sre_regex_parse((sre_pool_t*)cpool, (sre_char *)patternstr.c_str(), &ncaps, 0, &err_offset);
  prog = sre_regex_compile((sre_pool_t*)ppool, re);
  sre_reset_pool((sre_pool_t*)cpool);
  res.resize(2*(ncaps+1));
  ctx = sre_vm_pike_create_ctx((sre_pool_t*)cpool, (sre_program_t*)prog, &res[0], res.size()*sizeof(sre_int_t));
}
int StreamRegex::Match(const string &text, vector<Regex::Result> *out, bool eof) {
  int offset = last_end + since_last_end;
  sre_int_t rc = sre_vm_pike_exec((sre_vm_pike_ctx_t*)ctx, (sre_char*)text.data(), text.size(), eof, NULL);
  if (rc >= 0) {
    since_last_end = 0;
    for (int i = 0, l = res.size(); i < l; i += 2) 
      out->emplace_back(res[i] - offset, (last_end = res[i+1]) - offset);
  } else since_last_end += text.size();
  return 1;
}
#else
StreamRegex::~StreamRegex() {}
StreamRegex::StreamRegex(const string &patternstr) {}
int StreamRegex::Match(const string &text, vector<Regex::Result> *out, bool eof) { return 0; }
#endif

void NextRecordReader::Init(File *F, int fo) { return Init(bind(&File::Read, F, _1, _2), fo); }

const char *NextRecordReader::ReadNextRecord(int *offsetOut, int *nextoffsetOut, NextRecordCB nextcb) {
  const char *next, *text; int left; bool read_short = false;
  if (buf_dirty) buf_offset = buf.size();
  for (;;) {
    left = buf.size() - buf_offset;
    text = buf.data() + buf_offset;
    if (!buf_dirty && left>0 && (next = nextcb(StringPiece(text, left), read_short, &record_len))) {

      if (offsetOut) *offsetOut = file_offset - buf.size() + buf_offset;
      if (nextoffsetOut) *nextoffsetOut = file_offset - buf.size() + (next - buf.data());

      record_offset = buf_offset;
      buf_offset = next-buf.data();
      return text;
    }
    if (read_short) {
      buf_offset = -1;
      return 0;
    }

    buf.erase(0, buf_offset);
    int buf_filled = buf.size();
    buf.resize(buf.size() < 4096 ? 4096 : buf.size()*2);
    int len = read_cb((char*)buf.data()+buf_filled, buf.size()-buf_filled);
    if (len > 0) file_offset += len;
    read_short = len < buf.size()-buf_filled;
    buf.resize(max(len,0) + buf_filled);
    buf_dirty = false;
    buf_offset = 0;
  }
}

const char *NextRecordReader::NextLine(int *offset, int *nextoffset) {
  const char *nl;
  if (!(nl = ReadNextRecord(offset, nextoffset, LFL::NextLine))) return 0;
  if (nl) buf[record_offset + record_len] = 0;
  return nl;
}

const char *NextRecordReader::NextLineRaw(int *offset, int *nextoffset) {
  const char *nl;
  if (!(nl = ReadNextRecord(offset, nextoffset, LFL::NextLineRaw))) return 0;
  if (nl) buf[record_offset + record_len] = 0;
  return nl;
}

const char *NextRecordReader::NextChunk(int *offset, int *nextoffset) {
  const char *nc;
  if (!(nc = ReadNextRecord(offset, nextoffset, LFL::NextChunk<4096>))) return 0;
  if (nc) buf[record_offset + record_len] = 0;
  return nc;
}

const char *NextRecordReader::NextProto(int *offset, int *nextoffset, ProtoHeader *bhout) {
  const char *np;
  if (!(np = ReadNextRecord(offset, nextoffset, LFL::NextProto))) return 0;
  if (bhout) *bhout = ProtoHeader(np);
  return np + ProtoHeader::size;
}

}; // namespace LFL
