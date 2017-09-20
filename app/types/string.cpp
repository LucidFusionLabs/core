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

#include "core/app/types/string.h"
#include "core/app/flow.h"

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
  LFL::string message;
  StringPrintfImpl(&message, fmt, vsnprintf, 0);
  LFL::Logger::Log(level, file, line, message);
}

extern "C" void LFAppDebug(const char *file, int line, const char *fmt, ...) {
  LFL::string message;
  StringPrintfImpl(&message, fmt, vsnprintf, 0);
  LFL::Logger::WriteDebugLine(message.c_str(), file, line);
}

namespace LFL {
::std::ostream& operator<<(::std::ostream& os, const point &x) { return os << x.DebugString(); }
::std::ostream& operator<<(::std::ostream& os, const Box   &x) { return os << x.DebugString(); }

Printable::Printable(const pair<int,int>         &x) : string(StrCat("(", x.first, ", ", x.second, ")")) {}
Printable::Printable(const pair<string,string>   &x) : string(StrCat("(", x.first, ", ", x.second, ")")) {}
Printable::Printable(const vector<pair<int,int>> &x) : string(StrCat("{", Vec<pair<int,int>>::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const StringPairVec         &x) : string(StrCat("{", Vec<StringPair>   ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<string>        &x) : string(StrCat("{", Vec<string>       ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<double>        &x) : string(StrCat("{", Vec<double>       ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<float>         &x) : string(StrCat("{", Vec<float>        ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<int>           &x) : string(StrCat("{", Vec<int>          ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<uint16_t>      &x) : string(StrCat("{", Vec<uint16_t>     ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const Color                 &x) : string(x.DebugString()) {}
Printable::Printable(const String16              &x) : string(String::ToUTF8(x)) {}
Printable::Printable(const Void                  &x) : string(StringPrintf("%p", x)) {}
Printable::Printable(const void                  *x) : string(StringPrintf("%p", x)) {}
point Scannable::Scan(const point &p, const char *v) { StringWordIter w(StringPiece::Unbounded(v), isint2<' ', ','>); return point(atoi(w.NextString()), atoi(w.NextString())); }
Box   Scannable::Scan(const Box   &b, const char *v) { StringWordIter w(StringPiece::Unbounded(v), isint2<' ', ','>); return Box  (atoi(w.NextString()), atoi(w.NextString()), atoi(w.NextString()), atoi(w.NextString())); }
Color Scannable::Scan(const Color &b, const char *v) { return Color(v); }

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

#ifdef LFL_UNICODE_DEBUG
#define UnicodeDebug(...) ERROR(__VA_ARGS__)
#else
#define UnicodeDebug(...)
#endif

int UTF8::ReadGlyph(const StringPiece &s, const char *p, int *len, bool eof) {
  *len = 1;
  unsigned char c0 = *MakeUnsigned(p);
  if ((c0 & (1<<7)) == 0) return c0; // ascii
  if ((c0 & (1<<6)) == 0) { UnicodeDebug("unexpected continuation byte"); return Unicode::replacement_char; }
  if (c0 == 0xc0 || c0 == 0xc1 || (c0 >= 0xf5 && c0 <= 0xff)) { UnicodeDebug("overlong byte"); return Unicode::replacement_char; } 
  for ((*len)++; *len < 4; (*len)++) {
    if (s.Done(p + *len - 1)) { UnicodeDebug("unexpected end of string"); *len=eof; return Unicode::replacement_char; }
    if ((c0 & (1<<(7 - *len))) == 0) break;
  }

  int ret = 0;
  if      (*len == 2) ret = c0 & 0x1f;
  else if (*len == 3) ret = c0 & 0x0f;
  else if (*len == 4) ret = c0 & 0x07;
  else { UnicodeDebug("invalid len ", *len); *len=1; return Unicode::replacement_char; }

  for (int i = *len; i > 1; i--) {
    unsigned char c = *MakeUnsigned(++p);
    if ((c & 0xc0) != 0x80) { UnicodeDebug("unexpected non-continuation byte"); *len=1; return Unicode::replacement_char; }
    ret = (ret << 6) | (c & 0x3f);
  }
  return ret;
}

string UTF8::WriteGlyph(int codepoint) {
#if 0
  string out;
  char16_t in[] = { char16_t(codepoint), 0 };
  String::Convert(String16Piece(in, 1), &out, "UTF-16LE", "UTF-8");
  return out;
  replacement_char = 0xFFFD;
#else
  int len=0;
  if      (codepoint <= 0x7F)     return string(1, codepoint & 0x7F);
  else if (codepoint <= 0x7FF)    len = 1;
  else if (codepoint <= 0xFFFF)   len = 2;
  else if (codepoint <= 0x10FFFF) len = 3;
  else                            return "\xef\xbf\xbd"; 

  char buf[3];
  for (int i = 0; i != len; ++i) {
    buf[len - i] = 0x80 | (codepoint & 0x3F);
    codepoint >>= 6;
  }
  buf[0] = (0x1E << (6 - len)) | (codepoint & (0x3F >> len));
  return string(buf, len);
#endif
}

int UTF16::ReadGlyph(const String16Piece &s, const char16_t *p, int *len, bool eof) { *len=1; return *p; }
String16 UTF16::WriteGlyph(int codepoint) { return String16(1, codepoint); }

int isfileslash (int c) { return c == LocalFile::Slash; }
int IsOpenParen (int c) { return c == '(' || c == '[' || c == '<' || c == '{'; }
int IsCloseParen(int c) { return c == ')' || c == ']' || c == '>' || c == '}'; }
int MatchingParens(int c1, int c2) { return (c1 == '(' && c2 == ')') || (c1 == '[' && c2 == ']') || (c1 == '<' && c2 == '>') || (c1 == '{' && c2 == '}'); }
float my_atof(const char *v) { return v ? atof(v) : 0; }
int   my_atoi(const char *v) { return v ? atoi(v) : 0; }
int atoi(const char16_t *v) {
  const char16_t *p;
  if (!v) return 0; int ret = 0;
  if (!(p = FindChar(v, notspace))) return 0;
  bool neg = *p == '-';
  for (p += neg; *p >= '0' && *p <= '9'; p++) ret = ret*10 + *p - '0';
  return neg ? -ret : ret;
}

uint32_t fnv32(const void *buf, unsigned len, uint32_t hval) {
  if (!len) len = strlen(static_cast<const char*>(buf));
  const unsigned char *bp = static_cast<const unsigned char*>(buf), *be = bp + len;
  while (bp < be) {
    hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
    hval ^= uint32_t(*bp++);
  }
  return hval;
}

uint64_t fnv64(const void *buf, unsigned len, uint64_t hval) {
  if (!len) len = strlen(static_cast<const char*>(buf));
  const unsigned char *bp = static_cast<const unsigned char*>(buf), *be = bp + len;
  while (bp < be) {
    hval += (hval<<1) + (hval<<4) + (hval<<5) + (hval<<7) + (hval<<8) + (hval<<40);
    hval ^= uint64_t(*bp++);
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
template <class X>       X *FindChar(      X *text, int ischar, int (*isquotec)(int), int len, int *outlen) { return const_cast<X*>(FindChar(const_cast<const X*>(text), ischar, isquotec, len, outlen)); }
template <class X> const X *FindChar(const X *text, int ischar, int len, int *outlen) { return FindChar(text, ischar, 0, len, outlen); }
template <class X>       X *FindChar(      X *text, int ischar, int len, int *outlen) { return const_cast<X*>(FindChar(const_cast<const X*>(text), ischar, len, outlen)); }

template <class X> const X *FindChar(const X *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) {
  FindCharImpl(X, ischar(*p), *p, *p);
}
template <class X>       X *FindChar(      X *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) { return const_cast<X*>(FindChar(const_cast<const X*>(text), ischar, isquotec, len, outlen)); }
template <class X> const X *FindChar(const X *text, int (*ischar)(int), int len, int *outlen) { return FindChar(text, ischar, 0, len, outlen); }
template <class X>       X *FindChar(      X *text, int (*ischar)(int), int len, int *outlen) { return const_cast<X*>(FindChar(const_cast<const X*>(text), ischar, len, outlen)); }

template       char*        FindChar<char    >   (      char*,        int (*)(int), int, int*);
template const char*        FindChar<char    >   (const char*,        int (*)(int), int, int*);
template       char16_t*    FindChar<char16_t>   (      char16_t*,    int (*)(int), int, int*);
template const char16_t*    FindChar<char16_t>   (const char16_t*,    int (*)(int), int, int*);

#define LengthCharImpl(in, len, p, deref_p, check_p) \
  if (len >= 0) while (p-in < len &&  ischar(deref_p)) p++; \
  else          while (check_p    &&  ischar(deref_p)) p++;

#define LengthNotCharImpl(in, len, p, deref_p, check_p) \
  if (len >= 0) while (p-in < len && !ischar(deref_p)) p++; \
  else          while (check_p    && !ischar(deref_p)) p++;

template <class X> int LengthChar(const X *in, int (*ischar)(int), int len) {
  const X *p = in;
  LengthCharImpl(in, len, p, *p, *p);
  return p - in;
}
template int LengthChar(const char*,        int(*)(int), int);
template int LengthChar(const char16_t*,    int(*)(int), int);
template int LengthChar(const DrawableBox*, int(*)(int), int);

template <class X> int RLengthChar(const X *in, int (*ischar)(int), int len) {
  if (len <= 0) return 0;
  const X *p = in, *e = in - len;
  for (; p != e; --p) if (!ischar(*p)) break;
  return in - p;
}
template int RLengthChar(const char*,        int(*)(int), int);
template int RLengthChar(const char16_t*,    int(*)(int), int);
template int RLengthChar(const DrawableBox*, int(*)(int), int);

template <class X> bool ContainsChar(const X *in, int (*ischar)(int), int len) {
  const X *p = in;
  if (len >= 0) while (p-in < len) { if (ischar(*p)) return true; p++; }
  else          while (*p)         { if (ischar(*p)) return true; p++; }
  return false;
}
template bool ContainsChar(const char*,        int(*)(int), int);
template bool ContainsChar(const char16_t*,    int(*)(int), int);
template bool ContainsChar(const DrawableBox*, int(*)(int), int);

template <class X> const X *FindString(const StringPieceT<X> &haystack, const StringPieceT<X> &needle, bool case_sensitive) {
  const X *it = case_sensitive ?
    std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) :
    std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                [](X ch1, X ch2) { return std::toupper(ch1) == std::toupper(ch2); });
  return it == haystack.end() ? nullptr : it;
}
template const char     *FindString(const StringPiece   &haystack, const StringPiece   &needle, bool case_sensitive);
template const char16_t *FindString(const String16Piece &haystack, const String16Piece &needle, bool case_sensitive);

template <class X> int FindStringOffset(const StringPieceT<X> &haystack, const StringPieceT<X> &needle, bool case_sensitive) {
  const X *found = FindString<X>(haystack, needle, case_sensitive);
  return found ? found - haystack.begin() : -1;
}
template int FindStringOffset(const StringPiece   &haystack, const StringPiece   &needle, bool case_sensitive);
template int FindStringOffset(const String16Piece &haystack, const String16Piece &needle, bool case_sensitive);

template <class X> PieceIndex FindStringIndex(const StringPieceT<X> &haystack, const StringPieceT<X> &needle, bool case_sensitive) {
  const X *found = FindString<X>(haystack, needle, case_sensitive);
  if (!found) return PieceIndex();
  return PieceIndex(found - haystack.begin(), needle.size());
}
template PieceIndex FindStringIndex(const StringPiece   &haystack, const StringPiece   &needle, bool case_sensitive);
template PieceIndex FindStringIndex(const String16Piece &haystack, const String16Piece &needle, bool case_sensitive);

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

vector<string> Split(const string &text, char sep) {
  vector<string> tokens;
  size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != string::npos) {
    tokens.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  tokens.push_back(text.substr(start));
  return tokens;
}

void Join(string *out, const vector<string> &in) { return StrAppend(out, in, 0, in.size()); }
void Join(string *out, const vector<string> &in, int inB, int inE) {
  int size = 0;        for (int i = inB; i < inE; i++) size += in[i].size();
  StrAppendInit(size); for (int i = inB; i < inE; i++) { StrCatAdd(in[i]); } StrAppendReturn();
}
void Join(vector<string> *out, const vector<string> &in, const string &separator, bool left_or_right) {
  if (left_or_right) {
    for (int ii=0, oi=0; oi<out->size(); oi++) {
      if (ii >= in.size()) (*out)[oi] = "";
      else (*out)[oi] = (oi < out->size()-1) ? in[ii++] : Join(in, " ", ii, in.size());
    }
  } else {
    for (int ii=in.size()-1, oi=out->size()-1; oi>=0; oi--) {
      if (ii < 0) (*out)[oi] = "";
      else (*out)[oi] = (oi > 0) ? in[ii--] : Join(in, " ", 0, ii+1);
    }
  }
}
string Join(const vector<const char *> &strs, const string &separator) {
  string ret;
  for (auto &i : strs) StrAppend(&ret, ret.size()?separator:"", i);
  return ret;
}
string Join(const vector<string> &strs, const string &separator) {
  string ret;
  for (auto &i : strs) StrAppend(&ret, ret.size()?separator:"", i);
  return ret;
}
string Join(const vector<string> &strs, const string &separator, int beg_ind, int end_ind) {
  string ret;
  for (int i = beg_ind; i < strs.size() && i < end_ind; i++) StrAppend(&ret, ret.size()?separator:"", strs[i]);
  return ret;
}
string Join(const char* const* strs, const string &separator) {
  auto end = strs;
  while(*end) end++;
  return Join(strs, separator, 0, end-strs);
}
string Join(const char* const* strs, const string &separator, int beg_ind, int end_ind) {
  string ret;
  for (int i = beg_ind; i != end_ind; ++i) StrAppend(&ret, ret.size()?separator:"", strs[i]);
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

bool RemoveTrailing(string *text, int (*ischar)(int)) {
  int l = RLengthChar(text->data() + text->size() - 1, ischar, text->size());
  if (l) text->erase(text->size() - l, l);
  return l;
}

template <class X> string HexEscape(const StringPieceT<X> &text, const string &delim) {
  string ret;
  ret.reserve(text.size()*4);
  for (const X *p = text.data(); !text.Done(p); ++p) {
    auto c = *reinterpret_cast<const typename make_unsigned<X>::type*>(p);
    ret += delim;
    StringAppendf(&ret, "%02x", c);
  }
  return ret;
}

template <class X> string HexEscapeNonAscii(const StringPieceT<X> &text, const string &delim) {
  string ret;
  ret.reserve(text.size()*4);
  for (const X *p = text.data(); !text.Done(p); ++p) {
    auto c = *reinterpret_cast<const typename make_unsigned<X>::type*>(p);
    if (isascii(c)) ret += c;
    else {
      ret += delim;
      StringAppendf(&ret, "%02x", c);
    }
  }
  return ret;
}

template <class X> string JSONEscape(const StringPieceT<X> &text) {
  string ret;
  ret.reserve(text.size()*2);
  for (const X *p = text.data(); !text.Done(p); ++p) {
    auto c = *reinterpret_cast<const typename make_unsigned<X>::type*>(p);
    switch (c) {
      case '"':  StrAppend(&ret, "\\\""); break;
      case '\\': StrAppend(&ret, "\\\\"); break;
      case '\b': StrAppend(&ret, "\\b");  break;
      case '\f': StrAppend(&ret, "\\f");  break;
      case '\n': StrAppend(&ret, "\\n");  break;
      case '\r': StrAppend(&ret, "\\r");  break;
      case '\t': StrAppend(&ret, "\\t");  break;
      default:
        if (c <= 0x1f || c >= 0x7f) StringAppendf(&ret, "\\u%04x", c);
        else                        ret += c;
        break;
    }
  }
  return ret;
}

template string HexEscape        (const StringPiece  &, const string&);
template string HexEscape        (const String16Piece&, const string&);
template string HexEscapeNonAscii(const StringPiece  &, const string&);
template string HexEscapeNonAscii(const String16Piece&, const string&);
template string JSONEscape       (const StringPiece  &);
template string JSONEscape       (const String16Piece&);

string FirstMatchCSV(const StringPiece &haystack, const StringPiece &needle, int (*ischar)(int)) {
  unordered_set<string> h_map;
  StringWordIter h_words(haystack, ischar);
  StringWordIter i_words(needle,   ischar);
  for (string w = h_words.NextString(); !h_words.Done(); w = h_words.NextString()) h_map.insert(w);
  for (string w = i_words.NextString(); !i_words.Done(); w = i_words.NextString()) if (Contains(h_map, w)) return w;
  return "";
}

bool ParseKV(const string &t, string *k_out, string *v_out, int equal_char) {
  auto p = FindChar(t.c_str(), equal_char);
  if (!p) return false;
  ptrdiff_t p_offset = p - t.c_str();
  StringWordIter k_words(t.c_str(), p_offset), v_words(p+1, t.size()-p_offset-1);
  *k_out = k_words.NextString();
  *v_out = v_words.RemainingString();
  return k_out->size();
}

string UpdateKVLine(const string &haystack, const string &key, const string &val, int equal_char) {
  bool found = 0;
  StringLineIter lines(haystack);
  string ret, k, v, update = StrCat(key, string(1, equal_char), val);
  for (string line = lines.NextString(); !lines.Done(); line = lines.NextString()) {
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
  if (!ret) { 
    if (outlen) *outlen = p - text.buf;
    return final ? text.buf : 0;
  }
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

const char *NextContainerFileEntry(const StringPiece &text, bool final, int *outlen) {
  if (text.len < ContainerFileHeader::size) return 0;
  ContainerFileHeader hdr(text.buf);
  if (ContainerFileHeader::size + hdr.len > text.len) return 0;
  *outlen = hdr.len;
  return text.buf + ContainerFileHeader::size + hdr.len;
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
    const X *line = in.buf + cur_offset, *next = chomp ?
      NextLine   (StringPieceT<X>(line, in.Remaining(cur_offset)), false, &cur_len) :
      NextLineRaw(StringPieceT<X>(line, in.Remaining(cur_offset)), false, &cur_len);
    next_offset = next ? next - in.buf : -1;
    if (cur_len || (blanks && next)) return line;
  }
  return 0;
}

template struct StringWordIterT<char>;
template struct StringLineIterT<char>;
template struct StringWordIterT<char16_t>;
template struct StringLineIterT<char16_t>;
template struct StringWordIterT<DrawableBox>;

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
  if (len > 0 && line[len-1] == '\n') ret++;
  if (len > 1 && line[len-2] == '\r') ret++;
  return ret;
}
template int ChompNewlineLength(const char *line, int len);

const char *IncrementNewline(const char *in) {
  if (*in == '\r') in++;
  if (*in == '\n') in++;
  return in;
}

const char *DecrementNewline(const char *in) {
  if (*in == '\n') in--;
  if (*in == '\r') in--;
  return in;
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
  for (int i = 0; i < 64; i++) decoding_table[uint8_t(encoding_table[i])] = i;
}

string Base64::Encode(const char *in, size_t input_length) const {
  const unsigned char *data = MakeUnsigned(in);
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

string Base64::Decode(const char *data, size_t input_length) const {
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

bool Regex::Result::operator<(const Regex::Result &x) const { SortImpl2(begin, x.begin, end, x.end); }

Regex::Result RegexMatcher::MatchNext() {
  Regex::Result m = regex->MatchOne(StringPiece(iter, text.end() - iter));
  if (!m) return m;
  m += iter - text.begin();
  iter = text.begin() + m.end;
  return m;
}

int RegexMatcher::MatchAll(vector<Regex::Result> *out) {
  out->clear();
  for (auto r = MatchNext(); !!r; r = MatchNext()) out->push_back(r);
  return out->size();
}

bool RegexMatcher::ConvertToLineNumberColumnCoordinates
(const StringPiece &text, const vector<Regex::Result> &matches, vector<pair<int, int>> *out) {
  out->clear();
  int line_number = 0;
  auto match_i = matches.begin(), match_e = matches.end();
  StringLineIter lines(text, StringLineIter::Flag::BlankLines | StringLineIter::Flag::Raw);
  for (const char *l = lines.Next(); l && match_i != match_e; l = lines.Next(), line_number++) {
    size_t offset_beg = lines.CurrentOffset(), offset_end = offset_beg + lines.CurrentLength();
    for(; match_i != match_e && offset_beg <= match_i->begin && match_i->begin <= offset_end; ++match_i)
      out->emplace_back(line_number, match_i->begin - offset_beg);
  }
  return match_i == match_e;
}

pair<int,int> RegexLineMatcher::MatchNext() {
  for (;;) {
    if (!current_line) {
      current_line_number++;
      current_line = lines.Next();
      if (!current_line) return pair<int, int>(-1, -1);
      current_match = RegexMatcher(regex, StringPiece(current_line, lines.CurrentLength()));
    }
    Regex::Result r = current_match.MatchNext();
    if (!!r) return pair<int, int>(current_line_number, r.begin);
    current_line = nullptr;
  }
}

int RegexLineMatcher::MatchAll(vector<pair<int,int>> *out) {
  out->clear();
  for (auto r = MatchNext(); !Null(r); r = MatchNext()) out->push_back(r);
  return out->size();
}

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
    int len = read_cb(&buf[0] + buf_filled, buf.size() - buf_filled);
    if (len > 0) file_offset += len;
    read_short = len < buf.size() - buf_filled;
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

const char *NextRecordReader::NextContainerFileEntry(int *offset, int *nextoffset, ContainerFileHeader *bhout) {
  const char *np;
  if (!(np = ReadNextRecord(offset, nextoffset, LFL::NextContainerFileEntry))) return 0;
  if (bhout) *bhout = ContainerFileHeader(np);
  return np + ContainerFileHeader::size;
}

void NextRecordDispatcher::AddData(const StringPiece &b, bool final) {
  buf.append(b.data(), b.size());
  StringPiece remaining(buf), record;
  while ((remaining.buf = next_cb((record.buf = remaining.buf), final, &record.len))) {
    remaining.len = buf.size() - (remaining.buf - buf.data());
    cb(record.str());
  }
  buf.erase(0, buf.size() - remaining.len);
};

template <class X>
void TokenProcessor<X>::Init(const ArrayPiece<X> &text, int o, const ArrayPiece<X> &V, int Erase, CB &&C) {
  CHECK_LE((x = o), text.len);
  LoadV(V);
  cb = C;
  ni = x + ((erase = Erase) ? erase : 0);
  nw = ni<text.len && !isspace(text[ni ]);
  pw = x >0        && !isspace(text[x-1]);
  pi = x - pw;
  if ((pw && nw) || (pw && sw)) FindPrev(text);
  if ((pw && nw) || (nw && ew)) FindNext(text);
  FindBoundaryConditions(text, &lbw, &lew);
}

template <class X> void TokenProcessor<X>::ProcessUpdate(const ArrayPiece<X> &text) {
  int tokens = 0, vl = v.size();
  if (!vl) return;

  StringWordIterT<X> word(v.buf, v.len, isspace, 0);
  for (const X *w = word.Next(); w; w = word.Next(), tokens++) {
    int start_offset = w - v.buf, end_offset = start_offset + word.cur_len;
    bool first = start_offset == 0, last = end_offset == v.len;
    if (first && last && pw && nw) cb(pi, ni-pi+1,                             erase ? -1 : 1);
    else if (first && pw)          cb(pi, x+end_offset-pi,                     erase ? -2 : 2);
    else if (last && nw)           cb(x+start_offset, ni-x-start_offset+1,     erase ? -3 : 3);
    else                           cb(x+start_offset, end_offset-start_offset, erase ? -4 : 4);
  }
  if ((!tokens || overwrite) && vl) {
    if (pw && !sw && osw) { FindPrev(text); cb(pi, x-pi,        erase ? -5 : 5); }
    if (nw && !ew && oew) { FindNext(text); cb(x+vl, ni-x-vl+1, erase ? -6 : 6); }
  }
}

template <class X> void TokenProcessor<X>::ProcessResult() {
  if      (pw && nw) cb(pi, ni - pi + 1, erase ? 7 : -7);
  else if (pw && sw) cb(pi, x  - pi,     erase ? 8 : -8);
  else if (nw && ew) cb(x,  ni - x + 1,  erase ? 9 : -9);
}

template <class X>
void TokenProcessor<X>::FindBoundaryConditions(const ArrayPiece<X> &v, bool *sw, bool *ew) {
  *sw = v.size() && !isspace(v.front());
  *ew = v.size() && !isspace(v.back ());
}

template struct TokenProcessor<char>;
template struct TokenProcessor<char16_t>;
template struct TokenProcessor<DrawableBox>;

string Serializable::ToString() const { string ret; ToString(&ret); return ret; }

void Serializable::ToString(string *out) const {
  out->resize(Size());
  return ToString(&(*out)[0], out->size());
}

void Serializable::ToString(char *buf, int len) const {
  MutableStream os(buf, len);
  Out(&os);
}

void SerializableProto::Header::Out(Serializable::Stream *o) const { o->Htons( id); o->Htons( seq); }
void SerializableProto::Header::In(const Serializable::Stream *i)  { i->Ntohs(&id); i->Ntohs(&seq); }

string SerializableProto::ToString(unsigned short seq) const { string ret; ToString(&ret, seq); return ret; }

void SerializableProto::ToString(string *out, unsigned short seq) const {
  out->resize(Header::size + Size());
  return ToString(&(*out)[0], out->size(), seq);
}

void SerializableProto::ToString(char *buf, int len, unsigned short seq) const {
  MutableStream os(buf, len);
  Header hdr = { uint16_t(Id), seq };
  hdr.Out(&os);
  Out(&os);
}

}; // namespace LFL
