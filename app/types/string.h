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

#ifndef LFL_CORE_APP_TYPES_STRING_H__
#define LFL_CORE_APP_TYPES_STRING_H__

#ifdef WIN32
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
static __forceinline int ffsl(long x) {
  unsigned long i;
  if (_BitScanForward(&i, x)) return (i + 1);
  return (0);
}
static __forceinline int ffs(int x) { return ffsl(x); }
#endif

namespace LFL {
struct Bit {
  static int Count(unsigned long long n) { int c=0; for (; n; c++) n &= (n-1); return c; }
  static void Indices(unsigned long long in, int *o) { 
    for (int b, n = in & 0xffffffff; n; /**/) { b = ffs(n)-1; n ^= (1<<b); *o++ = b;    }
    for (int b, n = in >> 32       ; n; /**/) { b = ffs(n)-1; n ^= (1<<b); *o++ = b+32; }
    *o++ = -1;
  }
};

struct BitString {
  static int Clear(      unsigned char *b, int bucket) {          b[bucket/8] &= ~(1 << (bucket % 8)); return 1; }
  static int Set  (      unsigned char *b, int bucket) {          b[bucket/8] |=  (1 << (bucket % 8)); return 1; }
  static int Get  (const unsigned char *b, int bucket) { return   b[bucket/8] &   (1 << (bucket % 8));           }
  static int Not  (const unsigned char *b, int bucket) { return !(b[bucket/8] &   (1 << (bucket % 8)));          }
  static int Get  (      unsigned char *b, int bucket) { return Get(reinterpret_cast<const unsigned char*>(b), bucket); }
  static int Not  (      unsigned char *b, int bucket) { return Not(reinterpret_cast<const unsigned char*>(b), bucket); }
  static int Set  (               char *b, int bucket) { return Set(reinterpret_cast<      unsigned char*>(b), bucket); }
  static int Get  (const          char *b, int bucket) { return Get(reinterpret_cast<const unsigned char*>(b), bucket); }
  static int Get  (               char *b, int bucket) { return Get(reinterpret_cast<      unsigned char*>(b), bucket); }
  static int Clear(               char *b, int bucket) { return Clear(reinterpret_cast<    unsigned char*>(b), bucket); }

  static int FirstSet  (const unsigned char *b, int l) { for (auto i=b, e=b+l; i != e; ++i) { auto c = *i;     if (c != 0)   return (i-b)  *8 + ffs( c)-1; } return -1; }
  static int FirstClear(const unsigned char *b, int l) { for (auto i=b, e=b+l; i != e; ++i) { auto c = *i;     if (c != 255) return (i-b)  *8 + ffs(~c)-1; } return -1; }
  static int LastClear (const unsigned char *b, int l) { for (auto i=b+l;      i != b; --i) { auto c = *(i-1); if (c != 255) return (i-b-1)*8 + ffs(~c)-1; } return -1; }
  static int LastClear (const          char *b, int l) { return LastClear (reinterpret_cast<const unsigned char*>(b), l); }
  static int FirstClear(const          char *b, int l) { return FirstClear(reinterpret_cast<const unsigned char*>(b), l); }
};

struct Unicode {
  static const unsigned char non_breaking_space = 0xA0;
  static const unsigned short zero_width_non_breaking_space = 0xFEFF, replacement_char = 0xFFFD;
};

typedef basic_string<char16_t> String16;
template <class X> string ToString(const X& x) { std::stringstream in; in << x; return in.str(); }
string StringPrintf(const char *fmt, ...);

struct Printable : public string {
  Printable(const void *x);
  Printable(const basic_string<char16_t> &x);
  Printable(const string &x) : string(x) {}
  Printable(const char *x) : string(x) {}
  Printable(      char *x) : string(x) {}
  Printable(const bool &x) : string(ToString(x)) {}
  Printable(const int  &x) : string(ToString(x)) {}
  Printable(const long &x) : string(ToString(x)) {}
  Printable(const unsigned char *x) : string(reinterpret_cast<const char*>(x)) {}
  Printable(const char &x) : string(ToString(x)) {}
  Printable(const char16_t &x) : string(ToString(x)) {}
  Printable(const short &x) : string(ToString(x)) {}
  Printable(const float &x) : string(ToString(x)) {}
  Printable(const double &x) : string(ToString(x)) {}
  Printable(const unsigned &x) : string(ToString(x)) {}
  Printable(const long long &x) : string(ToString(x)) {}
  Printable(const unsigned char &x) : string(ToString(x)) {}
  Printable(const unsigned short &x) : string(ToString(x)) {}
  Printable(const unsigned long &x) : string(ToString(x)) {}
  Printable(const unsigned long long &x) : string(ToString(x)) {}
  Printable(const pair<int, int> &x);
  Printable(const vector<string> &x);
  Printable(const vector<double> &x);
  Printable(const vector<float> &x);
  Printable(const vector<int> &x);
  Printable(const Color &x);
  template <size_t N> Printable(const char (&x)[N]) : string(x) {}
  template <class X> Printable(const X& x) : string(StringPrintf("%s(%p)", typeid(X).name(), &x)) {}
};

struct Scannable {
  static bool     Scan(const bool&,     const char     *v) { return *v ? atoi(v) : true; }
  static int      Scan(const int&,      const char     *v) { return atoi(v); }
  static unsigned Scan(const unsigned&, const char     *v) { return atoi(v); }
  static float    Scan(const float&,    const char     *v) { return atof(v); }
  static double   Scan(const double&,   const char     *v) { return atof(v); }
  static string   Scan(const string&,   const char     *v) { return string(v); }
  static String16 Scan(const String16&, const char16_t *v) { return String16(v); }
};

struct PieceIndex {
  int offset, len;
  PieceIndex(int o=-1, int l=0) : offset(o), len(l) {}
};

template <class X> struct ArrayPiece {
  typedef       X*       iterator;
  typedef const X* const_iterator;
  const X *buf;
  int len;
  virtual ~ArrayPiece() {}
  ArrayPiece()                  : buf(0), len(0) {}
  ArrayPiece(const X *b, int l) : buf(b), len(l) {}
  ArrayPiece(const X *b, const PieceIndex &i) : buf(i.offset < 0 ? 0 : &b[i.offset]), len(i.len) {}
  ArrayPiece(const vector<X> &b) : buf(b.data()), len(b.size()) {}
  const X& operator[](int i) const { return buf[i]; }
  const X& front() const { return buf[0]; }
  const X& back() const { return buf[len-1]; }
  void clear() { buf=0; len=0; }
  bool null() const { return !buf; }
  bool empty() const { return !buf || len <= 0; }
  bool has_size() const { return len >= 0; }
  int size() const { return max(0, len); }
  void assign(const X *b, int l) { buf=b; len=l; }
  const X *data() const { return buf; }
  const void *ByteData() const { return buf; }
  int Bytes() const { return size() * sizeof(X); }
  int Remaining(int offset) { return has_size() ? len - offset : -1; }
  const_iterator begin() const { return buf; }
  const_iterator rbegin() const { return buf+len-1; }
  const_iterator end() const { return buf+len; }
};

template <class X> struct StringPieceT : public ArrayPiece<X> {
  virtual ~StringPieceT() {}
  StringPieceT() {}
  StringPieceT(const basic_string<X> &s) : ArrayPiece<X>(s.data(), s.size())  {}
  StringPieceT(const X *b, int l)        : ArrayPiece<X>(b,        l)         {}
  StringPieceT(const X *b)               : ArrayPiece<X>(b,        Length(b)) {}
  basic_string<X> str() const {
    if (this->buf && this->len < 0) return this->buf;
    return this->buf ? basic_string<X>(this->buf, this->len) : basic_string<X>();
  }
  bool Done(const X* p) const { return (this->len >= 0 && p >= this->buf + this->len) || *p == X(); }
  int Length() const { return this->len >= 0 ? this->len : Length(this->buf); }
  static StringPieceT<X> Unbounded (const X *b) { return StringPieceT<X>(b, -1); }
  static StringPieceT<X> FromString(const X *b) { return StringPieceT<X>(b, b?Length(b):0); }
  static StringPieceT<X> FromRemaining(const basic_string<X> &s, int r) { return StringPieceT<X>(s.data()+r, s.size()-r); }
  static size_t Length(const X *b) { const X *p = b; while (*p) p++; return p - b; }
  static const X *Blank() { static X x[1] = {0}; return x; }
  static const X *Space() { static X x[2] = {' ',0}; return x; }
  static const X *NullSpelled() { static X x[7] = {'<','N','U','L','L','>',0}; return x; }
};
typedef StringPieceT<char> StringPiece;
typedef StringPieceT<char16_t> String16Piece;
template <class X> StringPieceT<X> MakeUnbounded(const typename enable_if<is_signed<X>::value, X                              >::type *b) { return StringPieceT<X>::Unbounded(b); }
template <class X> StringPieceT<X> MakeUnbounded(const typename enable_if<is_signed<X>::value, typename make_unsigned<X>::type>::type *b) { return StringPieceT<X>::Unbounded(reinterpret_cast<const X*>(b)); }

struct StringBuffer {
  string data;
  StringPiece buf;
  StringBuffer(int s=0) { if (s) Resize(s); }
  int size() const { return buf.len; }
  int Capacity() const { return data.size(); }
  int Remaining() const { return data.size() - buf.len; }
  const char *begin() const { return buf.buf; }
  const char *end() const { return buf.buf + buf.len; };
  char *begin() { return &data[0]; }
  char *end() { return &data[0] + buf.len; }
  void Clear() { buf.len=0; }
  void Resize(int n) { data.resize(n); buf.buf=data.data(); }
  void EnsureAdditional(int n) { int s=data.size(), f=1; while(buf.len+n > s*f) f*=2; if (f>1) Resize(s*f); }
  void EnsureZeroTerminated() { if (buf.len < data.size()) data[buf.len] = 0; }
  void Add(const void *x, int l) { EnsureAdditional(l); memcpy(&data[0] + buf.len, x, l); buf.len += l; }
  void Added(int l) { /*CHECK_LE(buf.len+l, data.size());*/ buf.len += l; }
  void Flush(int l) { if (l && l != buf.len) memmove(&data[0], &data[0]+l, buf.len-l); buf.len -= l; }
};

struct String {
  template          <class Y> static void Copy(const string          &in, basic_string<Y> *out, int offset=0) { return Copy<char,    Y>(in, out, offset); }
  template          <class Y> static void Copy(const String16        &in, basic_string<Y> *out, int offset=0) { return Copy<char16_t,Y>(in, out, offset); }
  template <class X, class Y> static void Copy(const StringPieceT<X> &in, basic_string<Y> *out, int offset=0) {
    out->resize(offset + in.len);
    Y* o = &(*out)[offset];
    for (const X *i = in.buf; !in.Done(i); /**/) *o++ = *i++;
  }
  template          <class Y> static void Append(const string          &in, basic_string<Y> *out) { return Append<char,    Y>(in, out); }
  template          <class Y> static void Append(const String16        &in, basic_string<Y> *out) { return Append<char16_t,Y>(in, out); }
  template <class X, class Y> static void Append(const StringPieceT<X> &in, basic_string<Y> *out) {
    Copy(in.data(), out, out->size());
  }
  template          <class Y> static int Convert(const string          &in, basic_string<Y> *out, const char *fe, const char *te) { return Convert<char,     Y>(in, out, fe, te); }
  template          <class Y> static int Convert(const String16        &in, basic_string<Y> *out, const char *fe, const char *te) { return Convert<char16_t, Y>(in, out, fe, te); }
  template <class X, class Y> static int Convert(const StringPieceT<X> &in, basic_string<Y> *out,
                                                 const char *from_encoding, const char *to_encoding);

  static string   ToAscii(const StringPiece    &s, int *lo=0) { if (lo) *lo=s.size(); return s.str(); }
  static string   ToAscii(const String16Piece  &s, int *lo=0) { string v; int l=Convert(s, &v, "UCS-16LE", "US-ASCII"); if (lo) *lo=l; return v; }
  static string   ToUTF8 (const String16Piece  &s, int *lo=0) { string v; int l=Convert(s, &v, "UTF-16LE", "UTF-8");    if (lo) *lo=l; return v; }
  static string   ToUTF8 (const StringPiece    &s, int *lo=0) { if (lo) *lo=s.size(); return s.str(); }
  static String16 ToUTF16(const String16Piece  &s, int *lo=0) { if (lo) *lo=s.size(); return s.str(); }
  static String16 ToUTF16(const StringPiece    &s, int *lo=0);
};

struct UTF8 {
  static string WriteGlyph(int codepoint);
  static int ReadGlyph(const StringPiece   &s, const char  *p, int *l, bool eof=0);
};
struct UTF16 {
  static String16 WriteGlyph(int codepoint);
  static int ReadGlyph(const String16Piece &s, const char16_t *p, int *l, bool eof=0);
};

template <class X> struct UTF {};
template <> struct UTF<char> {
  static string WriteGlyph(int codepoint) { return UTF8::WriteGlyph(codepoint); }
  static int ReadGlyph(const StringPiece   &s, const char     *p, int *l, bool eof=0) { return UTF8::ReadGlyph(s, p, l, eof); }
  static int ReadGlyph(const String16Piece &s, const char16_t *p, int *l, bool eof=0) { FATALf("%s", "no such thing as 16bit UTF-8"); }
};
template <> struct UTF<char16_t> {
  static String16 WriteGlyph(int codepoint) { return UTF16::WriteGlyph(codepoint); }
  static int ReadGlyph(const String16Piece &s, const char16_t *p, int *l, bool eof=0) { return UTF16::ReadGlyph(s, p, l, eof); }
  static int ReadGlyph(const StringPiece   &s, const char     *p, int *l, bool eof=0) { FATALf("%s", "no such thing as 8bit UTF-16"); }
};

template <class X> struct StringIterT {
  typedef X type;
  virtual ~StringIterT() {}
  virtual void Reset() {}
  virtual bool Done() const = 0;
  virtual const X *Next() = 0;
  virtual const X *Begin() const = 0;
  virtual const X *Current() const = 0;
  virtual int CurrentOffset() const = 0;
  virtual int CurrentLength() const = 0;
  virtual int TotalLength() const = 0;

  template <class Y=X> typename enable_if<is_pod<Y>::value, basic_string<type>>::type NextString() {
    if (const type *b = Next()) return basic_string<type>(b, CurrentLength());
    else                        return basic_string<type>();
  }
  template <class Y=X> typename enable_if<is_pod<Y>::value, basic_string<type>>::type RemainingString() {
    int total = TotalLength(), offset = CurrentOffset();
    if (total >= 0) return basic_string<type>(Begin() + offset, total - offset);
    else            return basic_string<type>(Begin() + offset);
  }
  template <class Y=X, class Z> typename enable_if<is_pod<Y>::value, void>::type ScanN(Z *out, int N) {
    for (int i=0; i<N; ++i) {
      auto v = Next(); 
      out[i] = v ? Scannable::Scan(Z(), basic_string<type>(v, CurrentLength()).data()) : 0;
    }
  }
};
typedef StringIterT<char> StringIter;

template <class X> struct StringWordIterT : public StringIterT<X> {
  StringPieceT<X> in;
  int cur_len=0, cur_offset=0, next_offset=0, (*IsSpace)(int)=0, (*IsQuote)(int)=0, flag=0; 
  StringWordIterT() {}
  StringWordIterT(const X *buf, int len,    int (*IsSpace)(int)=0, int(*IsQuote)(int)=0, int Flag=0);
  StringWordIterT(const StringPieceT<X> &b, int (*IsSpace)(int)=0, int(*IsQuote)(int)=0, int Flag=0) :
    StringWordIterT(b.buf, b.len, IsSpace, IsQuote, Flag) {}
  bool Done() const { return cur_offset < 0; }
  const X *Next();
  const X *Begin() const { return in.buf; }
  const X *Current() const { return in.buf + cur_offset; }
  int CurrentOffset() const { return cur_offset; }
  int CurrentLength() const { return cur_len; }
  int TotalLength() const { return in.len; }
};
typedef StringWordIterT<char>     StringWordIter;
typedef StringWordIterT<char16_t> StringWord16Iter;

template <class X> struct StringLineIterT : public StringIterT<X> {
  struct Flag { enum { BlankLines=1 }; };
  StringPieceT<X> in;
  basic_string<X> buf;
  int cur_len=0, cur_offset=0, next_offset=0, flag=0;
  bool first=0;
  StringLineIterT(const StringPieceT<X> &B, int F=0) : in(B), flag(F), first(1) {}
  StringLineIterT() : cur_offset(-1) {}
  bool Done() const { return cur_offset < 0; }
  const X *Next();
  const X *Begin() const { return in.buf; }
  const X *Current() const { return in.buf + cur_offset; }
  int CurrentOffset() const { return cur_offset; }
  int CurrentLength() const { return cur_len; }
  int TotalLength() const { return in.len; }
};
typedef StringLineIterT<char>     StringLineIter;
typedef StringLineIterT<char16_t> StringLine16Iter;

struct IterWordIter : public StringIter {
  StringIter *iter;
  StringWordIter word;
  int first_count=0;
  bool own_iter=0;
  ~IterWordIter() { if (own_iter) delete iter; }
  IterWordIter(StringIter *i, bool owner=0) : iter(i), own_iter(owner) {};
  void Reset() { if (iter) iter->Reset(); first_count=0; }
  bool Done() const { return iter->Done(); }
  const char *Next();
  const char *Begin() const { return iter->Begin(); }
  const char *Current() const { return word.Current(); }
  int CurrentOffset() const { return iter->CurrentOffset() + word.CurrentOffset(); }
  int CurrentLength() const { return word.CurrentLength(); }
  int TotalLength() const { return iter->TotalLength(); }
};

template <int F, int T>                 int tochar (int i) { return i == F ? T :  i; }
template <int F, int T, int F2, int T2> int tochar2(int i) { return i == F ? T : (i == F2 ? T2 : i); }

template <int V>          int                 isint (int N) { return N == V; }
template <int V1, int V2> int                 isint2(int N) { return (N == V1) || (N == V2); }
template <int V1, int V2, int V3>         int isint3(int N) { return (N == V1) || (N == V2) || (N == V3); }
template <int V1, int V2, int V3, int V4> int isint4(int N) { return (N == V1) || (N == V2) || (N == V3) || (N == V4); }

#undef isspace
#undef isascii
#undef isalpha
#undef isalnum
#undef isupper
#undef islower
#undef isdigit
#undef isnumber
#undef ispunct
inline int isspace(int c) { return std::iswspace(c); }
inline int isascii(int c) { return c >= 32 && c < 128; }
inline int isalpha(int c) { return std::iswalpha(c); }
inline int isalnum(int c) { return std::iswalnum(c); }
inline int isupper(int c) { return std::iswupper(c); }
inline int islower(int c) { return std::iswlower(c); }
inline int ispunct(int c) { return std::iswpunct(c); }
inline int isdot(int c) { return c == '.'; }
inline int iscomma(int c) { return c == ','; }
inline int isand(int c) { return c == '&'; }
inline int isdquote(int c) { return c == '"'; }
inline int issquote(int c) { return c == '\''; }
inline int istick(int c) { return c == '`'; }
inline int isdigit(int c) { return (c >= '0' && c <= '9'); }
inline int isnumber(int c) { return isdigit(c) || c == '.'; }
inline int isquote(int c) { return isdquote(c) || issquote(c) || istick(c); }
inline int notspace(int c) { return !isspace(c); }
inline int notalpha(int c) { return !isalpha(c); }
inline int notalnum(int c) { return !isalnum(c); }
inline int notnum(int c) { return !isnumber(c); }
inline int notcomma(int c) { return !iscomma(c); }
inline int notdot(int c) { return !isdot(c); }

int isfileslash(int c);
int MatchingParens(int c1, int c2);
int atoi(const char     *v);
int atoi(const char16_t *v);
float my_atof(const char *v);
inline double atof(const string &v) { return ::atof(v.c_str()); }
inline int    atoi(const string &v) { return ::atoi(v.c_str()); }
uint32_t fnv32(const void *buf, unsigned len=0, uint32_t hval=0);
uint64_t fnv64(const void *buf, unsigned len=0, uint64_t hval=0);

String16 String16Printf(const char *fmt, ...);
basic_string<wchar_t> WStringPrintf(const wchar_t *fmt, ...);
void StringAppendf(string *out, const char *fmt, ...);
void StringAppendf(String16 *out, const char *fmt, ...);
int sprint(char *out, int len, const char *fmt, ...);

inline string StrCat(const Printable &x1) { return x1; }
inline String16 Str16Cat(const Printable &x1) { return String::ToUTF16(x1); }
string StrCat(const Printable &x1, const Printable &x2);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13, const Printable &x14);
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13, const Printable &x14, const Printable &x15);

void StrAppend(string *out, const Printable &x1);
void StrAppend(string *out, const Printable &x1, const Printable &x2);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12);
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13);

void StrAppend(String16 *out, const Printable &x1);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12);
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13);

bool PrefixMatch(const char16_t *in, const char16_t *pref, int case_sensitive=true);
bool PrefixMatch(const char     *in, const char     *pref, int case_sensitive=true);
bool PrefixMatch(const char     *in, const string   &pref, int case_sensitive=true);
bool PrefixMatch(const string   &in, const char     *pref, int case_sensitive=true);
bool PrefixMatch(const string   &in, const string   &pref, int case_sensitive=true);
bool PrefixMatch(const String16 &in, const String16 &pref, int case_sensitive=true);
bool PrefixMatch(const String16 &in, const char     *pref, int case_sensitive=true);

bool SuffixMatch(const char16_t *in, const char16_t *pref, int case_sensitive=true);
bool SuffixMatch(const char     *in, const char     *pref, int case_sensitive=true);
bool SuffixMatch(const char     *in, const string   &pref, int case_sensitive=true);
bool SuffixMatch(const string   &in, const char     *pref, int case_sensitive=true);
bool SuffixMatch(const string   &in, const string   &pref, int case_sensitive=true);
bool SuffixMatch(const String16 &in, const string   &pref, int case_sensitive=true);
bool SuffixMatch(const String16 &in, const String16 &pref, int case_sensitive=true);

bool StringEquals(const String16 &s1, const String16 &s2, int case_sensitive=false);
bool StringEquals(const String16 &s1, const char     *s2, int case_sensitive=false);
bool StringEquals(const string   &s1, const string   &s2, int case_sensitive=false);
bool StringEquals(const char     *s1, const string   &s2, int case_sensitive=false);
bool StringEquals(const string   &s1, const char     *s2, int case_sensitive=false);
bool StringEquals(const char     *s1, const char     *s2, int case_sensitive=false);
bool StringEquals(const char16_t *s1, const char16_t *s2, int case_sensitive=false);

bool StringEmptyOrEquals(const string   &in, const string   &ref, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const String16 &ref, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const string   &ref, int case_sensitive=false);
bool StringEmptyOrEquals(const string   &in, const string   &ref1, const string   &ref2, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const String16 &ref1, const String16 &ref2, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const string   &ref1, const string   &ref2, int case_sensitive=false);

template <class X>       X *FindChar(      X *text, int c,                                   int len=-1, int *outlen=0);
template <class X> const X *FindChar(const X *text, int c,                                   int len=-1, int *outlen=0);
template <class X>       X *FindChar(      X *text, int c,              int (*isquote)(int), int len=-1, int *outlen=0);
template <class X> const X *FindChar(const X *text, int c,              int (*isquote)(int), int len=-1, int *outlen=0);
template <class X>       X *FindChar(      X *text, int (*ischar)(int),                      int len=-1, int *outlen=0);
template <class X> const X *FindChar(const X *text, int (*ischar)(int),                      int len=-1, int *outlen=0);
template <class X>       X *FindChar(      X *text, int (*ischar)(int), int (*isquote)(int), int len=-1, int *outlen=0);
template <class X> const X *FindChar(const X *text, int (*ischar)(int), int (*isquote)(int), int len=-1, int *outlen=0);
template <class X> int    LengthChar(const X* text, int (*ischar)(int), int len=-1);
template <class X> int   RLengthChar(const X* text, int (*ischar)(int), int len);

template <class X, class Y> int Split(const StringPieceT<X> &in, int (*ischar)(int), int (*isquote)(int), vector<Y> *out) {
  out->clear();
  if (!in.buf) return 0;
  StringWordIterT<X> words(in, ischar, isquote);
  for (string word = words.NextString(); !words.Done(); word = words.NextString())
    out->push_back(Scannable::Scan(Y(), word.c_str()));
  return out->size();
}
template <class X> int Split(const string   &in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<char,     X>(StringPiece(in),              ischar, isquote, out); }
template <class X> int Split(const string   &in, int (*ischar)(int),                      vector<X> *out) { return Split<char,     X>(StringPiece(in),              ischar, NULL,    out); }
template <class X> int Split(const char     *in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<char,     X>(StringPiece::Unbounded(in),   ischar, isquote, out); }
template <class X> int Split(const char     *in, int (*ischar)(int),                      vector<X> *out) { return Split<char,     X>(StringPiece::Unbounded(in),   ischar, NULL,    out); }
template <class X> int Split(const String16 &in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<char16_t, X>(String16Piece(in),            ischar, isquote, out); }
template <class X> int Split(const String16 &in, int (*ischar)(int),                      vector<X> *out) { return Split<char16_t, X>(String16Piece(in),            ischar, NULL,    out); }
template <class X> int Split(const char16_t *in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<char16_t, X>(String16Piece::Unbounded(in), ischar, isquote, out); }
template <class X> int Split(const char16_t *in, int (*ischar)(int),                      vector<X> *out) { return Split<char16_t, X>(String16Piece::Unbounded(in), ischar, NULL,    out); }

template <class X> int Split(const StringPiece &in, int (*ischar)(int), int (*isquote)(int), set<X> *out) {
  out->clear();
  if (!in.buf) return 0;
  StringWordIter words(in, ischar, isquote);
  for (string word = words.NextString(); !words.Done(); word = words.NextString())
    out->insert(Scannable::Scan(X(), word.c_str()));
  return out->size();
}
template <class X> int Split(const char   *in, int (*ischar)(int),                      set<X> *out) { return Split(StringPiece::Unbounded(in), in, ischar, NULL,    out); }
template <class X> int Split(const string &in, int (*ischar)(int), int (*isquote)(int), set<X> *out) { return Split(StringPiece(in),                ischar, isquote, out); }
template <class X> int Split(const string &in, int (*ischar)(int),                      set<X> *out) { return Split(StringPiece(in),                ischar, NULL,    out); }

int           Split(const StringPiece &in, int (*ischar)(int), string *left);
int           Split(const StringPiece &in, int (*ischar)(int), string *left, string *right);
inline int    Split(const char        *in, int (*ischar)(int), string *left, string *right) { return Split(StringPiece::Unbounded(in), ischar, left, right); }
inline int    Split(const string      &in, int (*ischar)(int), string *left, string *right) { return Split(StringPiece(in),            ischar, left, right); }
inline string Split(const StringPiece &in, int (*ischar)(int)) { string ret; Split(in, ischar, &ret); return ret; }

void Join(string *out, const vector<string> &in);
void Join(string *out, const vector<string> &in, int inB, int inE);
string Join(const vector<const char *> &strs, const string &separator);
string Join(const vector<string> &strs, const string &separator);
string Join(const vector<string> &strs, const string &separator, int beg_ind, int end_ind);
string strip(const char *s, int (*stripchar)(int), int (*stripchar2)(int)=0);
string togrep(const char *s, int (*grepchar)(int), int (*grepchar2)(int)=0);
string   toconvert(const char     *text, int (*tochar)(int), int (*ischar)(int)=0);
string   toconvert(const string   &text, int (*tochar)(int), int (*ischar)(int)=0);
String16 toconvert(const char16_t *text, int (*tochar)(int), int (*ischar)(int)=0);
String16 toconvert(const String16 &text, int (*tochar)(int), int (*ischar)(int)=0);
string   toupper(const char     *text);
string   toupper(const string   &text);
String16 toupper(const char16_t *text);
String16 toupper(const String16 &text);
string   tolower(const char     *text);
string   tolower(const string   &text);
String16 tolower(const char16_t *text);
String16 tolower(const String16 &text);
string   ReplaceEmpty (const string   &in, const string   &replace_with);
String16 ReplaceEmpty (const String16 &in, const string   &replace_with);
String16 ReplaceEmpty (const String16 &in, const String16 &replace_with);
string ReplaceNewlines(const string   &in, const string   &replace_with);
bool ReplaceString(string *text, const string &needle, const string &replace);

template <class X> string CHexEscape        (const basic_string<X> &text);
template <class X> string CHexEscapeNonAscii(const basic_string<X> &text);

template <class... Args> void StrAppendCSV(string *out, Args&&... args) { StrAppend(out, out->size() ? "," : "", forward<Args>(args)...); }
string FirstMatchCSV(const StringPiece &haystack, const StringPiece &needle, int (*ischar)(int) = iscomma);

bool ParseKV(const string &t, string *k_out, string *v_out, int equal_char='=');
string UpdateKVLine(const string &haystack, const string &key, const string &val, int equal_char='=');

const char     *NextLine   (const StringPiece   &text, bool final=0, int *outlen=0);
const char16_t *NextLine   (const String16Piece &text, bool final=0, int *outlen=0);
const char     *NextLineRaw(const StringPiece   &text, bool final=0, int *outlen=0);
const char16_t *NextLineRaw(const String16Piece &text, bool final=0, int *outlen=0);
const char     *NextProto  (const StringPiece   &text, bool final=0, int *outlen=0);
template <int S> const char *NextChunk(const StringPiece &text, bool final=0, int *outlen=0) {
  int add = final ? max(text.len, 0) : S;
  if (text.len < add) return 0;
  *outlen = add;
  return text.buf + add;
}

template <class X> const X *BlankNull(const X *x) { return x ? x : StringPieceT<X>::Blank(); }
template <class X> const X *SpellNull(const X *x) { return x ? x : StringPieceT<X>::NullSpelled(); }
template <class X> void AccumulateAsciiDigit(X *v, unsigned char c) { *v = *v * 10 + (c - '0'); }
template <class X> int IsNewline(const X *str);
template <class X> int ChompNewline(X *str, int len);
template <class X> int ChompNewlineLength(const X *str, int len);
const char *Default(const char *x, const char *default_x);
const char *ParseProtocol(const char *url, string *protO);
const char *BaseName(const StringPiece   &text, int *outlen=0);
int BaseDir(const char *path, const char *cmp);
int DirNameLen(const StringPiece   &text, bool include_slash=false);
int DirNameLen(const String16Piece &text, bool include_slash=false);

struct Base64 {
  string encoding_table, decoding_table;
  int mod_table[3];
  Base64();

  string Encode(const char *in,   size_t input_length);
  string Decode(const char *data, size_t input_length);
};

struct Regex {
  struct Result {
    int begin, end;
    Result(int B=0, int E=0) : begin(B), end(E) {}
    string Text(const string &t) const { return t.substr(begin, end - begin); }
    float FloatVal(const string &t) const { return atof(Text(t).c_str()); }
    void operator+=(const Result &v) { begin += v.begin; end += v.end; }
    void operator-=(const Result &v) { begin -= v.begin; end -= v.end; }
  };
  void *impl=0;
  ~Regex();
  Regex() {}
  Regex(const string &pattern);
  int Match(const string &text, vector<Result> *out);
};

struct StreamRegex {
  void *prog=0, *ctx=0, *ppool=0, *cpool=0;
  int last_end=0, since_last_end=0;
  vector<intptr_t> res;
  ~StreamRegex();
  StreamRegex(const string &pattern);
  int Match(const string &text, vector<Regex::Result> *out, bool eof=0);
};

struct NextRecordReader {
  typedef const char* (*NextRecordCB)(const StringPiece&, bool, int *);
  typedef function<int(void*, size_t)> ReadCB;
  ReadCB read_cb;
  string buf;
  bool buf_dirty;
  int buf_offset, file_offset, record_offset, record_len;
  NextRecordReader(File *f, int fo=0) { Init(f, fo); }
  NextRecordReader(const ReadCB &cb, int fo=0) { Init(cb, fo); }

  void Init(File *f, int fo=0);
  void Init(const ReadCB &cb, int fo=0) { read_cb=cb; Reset(); file_offset=fo; }
  void Reset() { buf.clear(); buf_dirty = 0; buf_offset = file_offset = record_offset = record_len = 0; }
  void AddFileOffset(int v) { file_offset += v; buf_dirty = 1; }
  void SetFileOffset(int v) { file_offset  = v; buf_dirty = 1; }
  const char *ReadNextRecord(int *offset, int *nextoffset, NextRecordCB cb);
  const char *NextLine   (int *offset=0, int *nextoffset=0);
  const char *NextLineRaw(int *offset=0, int *nextoffset=0);
  const char *NextChunk  (int *offset=0, int *nextoffset=0);
  const char *NextProto  (int *offset=0, int *nextoffset=0, ProtoHeader *phout=0);
};

struct NextRecordDispatcher {
  typedef const char* (*NextCB)(const StringPiece&, bool, int*);
  StringCB cb;
  string buf;
  NextCB next_cb;
  NextRecordDispatcher(const StringCB &C=StringCB(), NextCB NC=&LFL::NextLine) : cb(C), next_cb(NC) {}
  void AddData(const StringPiece &b, bool final=0);
};

template <class X> struct TokenProcessor {
  typedef function<void(int,int,int)> CB;
  bool sw=0, ew=0, pw=0, nw=0, overwrite=0, osw=1, oew=1;
  bool lbw=0, lew=0, nlbw=0, nlew=0;
  int x=0, erase=0, pi=0, ni=0;
  ArrayPiece<X> v;
  CB cb;
  void Init(const ArrayPiece<X>&, int o, const ArrayPiece<X>&, int Erase, CB&&);
  void LoadV(const ArrayPiece<X> &V) { FindBoundaryConditions((v=V), &sw, &ew); }
  void FindPrev(const ArrayPiece<X> &t) { while (pi > 0       && !isspace(t[pi-1])) pi--; }
  void FindNext(const ArrayPiece<X> &t) { while (ni < t.len-1 && !isspace(t[ni+1])) ni++; }
  void PrepareOverwrite(const ArrayPiece<X> &V) { osw=sw; oew=ew; LoadV(V); erase=0; overwrite=1; }
  void ProcessUpdate(const ArrayPiece<X>&);
  void ProcessResult();
  void SetNewLineBoundaryConditions(bool sw, bool ew) { nlbw=sw; nlew=ew; }
  static void FindBoundaryConditions(const ArrayPiece<X> &v, bool *sw, bool *ew);
};

struct Serializable {
  template <class X> static void ReadType (X    *addr, const void *v) { if (addr) memcpy(addr,  v, sizeof(X)); }
  template <class X> static void WriteType(void *addr, const X    &v) { if (addr) memcpy(addr, &v, sizeof(X)); }

  struct Stream {
    char *buf;
    size_t size;
    mutable int offset=0;
    mutable bool error=0;
    Stream(char *B, size_t S) : buf(B), size(S) {}

    int Len() const { return size; }
    int Pos() const { return offset; };
    int Remaining() const { return size - offset; }
    int Result() const { return error ? -1 : 0; }
    const char *Start() const { return buf; }
    const char *End() const { return buf + size; }
    const char *Get(int len=0) const {
      const char *ret = buf + offset;
      if ((offset += len) > size) { error=1; return 0; }
      return ret;
    }
    virtual char *End()          { FATAL(Void(this), ": ConstStream write"); return 0; }
    virtual char *Get(int len=0) { FATAL(Void(this), ": ConstStream write"); return 0; }

    template <class X>
    void AString (const ArrayPiece<X> &in) { auto v = Get(in.Bytes()+sizeof(int)); if (v) { memcpy(v+4, in.ByteData(), in.Bytes()); WriteType<int>(v, htonl(in.Bytes())); } }
    void BString (const StringPiece   &in) { auto v = Get(in.size ()+sizeof(int)); if (v) { memcpy(v+4, in.data(),     in.size());  WriteType<int>(v, htonl(in.size ())); } }
    void NTString(const StringPiece   &in) { auto v = Get(in.size ()+1);           if (v) { memcpy(v,   in.data(),     in.size());  v[in.size()]=0; } }
    void String  (const StringPiece   &in) { auto v = Get(in.size ());             if (v) { memcpy(v,   in.data(),     in.size());                  } }

    void Write8 (const unsigned char  &v) { WriteType(Get(sizeof(v)), v); }
    void Write8 (const          char  &v) { WriteType(Get(sizeof(v)), v); }
    void Write16(const unsigned short &v) { WriteType(Get(sizeof(v)), v); }
    void Write16(const          short &v) { WriteType(Get(sizeof(v)), v); }
    void Write32(const unsigned int   &v) { WriteType(Get(sizeof(v)), v); }
    void Write32(const          int   &v) { WriteType(Get(sizeof(v)), v); }
    void Write32(const unsigned long  &v) { WriteType(Get(sizeof(v)), v); }
    void Write32(const          long  &v) { WriteType(Get(sizeof(v)), v); }

    void Htons  (const unsigned short &v) { WriteType(Get(sizeof(v)), htons(v)); }
    void Ntohs  (const unsigned short &v) { WriteType(Get(sizeof(v)), ntohs(v)); }
    void Htons  (const          short &v) { WriteType(Get(sizeof(v)), htons(v)); }
    void Ntohs  (const          short &v) { WriteType(Get(sizeof(v)), ntohs(v)); }
    void Htonl  (const unsigned int   &v) { WriteType(Get(sizeof(v)), htonl(v)); }
    void Ntohl  (const unsigned int   &v) { WriteType(Get(sizeof(v)), ntohl(v)); }
    void Htonl  (const          int   &v) { WriteType(Get(sizeof(v)), htonl(v)); }
    void Ntohl  (const          int   &v) { WriteType(Get(sizeof(v)), ntohl(v)); }

    void Htons(unsigned short *out) const { Read16(out); *out = htons(*out); }
    void Ntohs(unsigned short *out) const { Read16(out); *out = ntohs(*out); }
    void Htons(         short *out) const { Read16(out); *out = htons(*out); }
    void Ntohs(         short *out) const { Read16(out); *out = ntohs(*out); }
    void Htonl(unsigned int   *out) const { Read32(out); *out = htonl(*out); }
    void Ntohl(unsigned int   *out) const { Read32(out); *out = ntohl(*out); }
    void Htonl(         int   *out) const { Read32(out); *out = htonl(*out); }
    void Ntohl(         int   *out) const { Read32(out); *out = ntohl(*out); }
    
    void Read8 (unsigned char   *out) const { ReadType(out, Get(sizeof(*out))); }
    void Read8 (         char   *out) const { ReadType(out, Get(sizeof(*out))); }
    void Read16(unsigned short  *out) const { ReadType(out, Get(sizeof(*out))); }
    void Read16(         short  *out) const { ReadType(out, Get(sizeof(*out))); }
    void Read32(unsigned int    *out) const { ReadType(out, Get(sizeof(*out))); }
    void Read32(         int    *out) const { ReadType(out, Get(sizeof(*out))); }
    void Read32(unsigned long   *out) const { ReadType(out, Get(sizeof(*out))); }
    void Read32(         long   *out) const { ReadType(out, Get(sizeof(*out))); }
    void ReadString(StringPiece *out) const { Ntohl(&out->len); out->buf = Get(out->len); }
    void ReadString(string *out) const { int len=0; Ntohl(&len); *out = string(Get(len), len); }

    template <class X> void ReadUnalignedArray(ArrayPiece<X> *out) const
    { int l=0; Ntohl(&l); out->assign(reinterpret_cast<const X*>(Get(l)), l/sizeof(X)); }
  };

  struct ConstStream : public Stream {
    ConstStream(const char *B, int S) : Stream(const_cast<char*>(B), S) {}
  };

  struct MutableStream : public Stream {
    MutableStream(char *B, int S) : Stream(B, S) {}
    char *End() { return buf + size; }
    char *Get(int len=0) {
      char *ret = buf + offset;
      if ((offset += len) > size) { error=1; return 0; }
      return ret;
    }
  };

  struct Header {
    static const int size = 4;
    unsigned short id, seq;

    void Out(Stream *o) const;
    void In(const Stream *i);
  };

  int Id;
  Serializable(int ID) : Id(ID) {}
  virtual ~Serializable() {}

  virtual int Size() const = 0;
  virtual int HeaderSize() const = 0;
  virtual int In(const Stream *i) = 0;
  virtual void Out(Stream *o) const = 0;

  virtual string ToString() const;
  virtual string ToString(unsigned short seq) const;
  virtual void ToString(string *out) const;
  virtual void ToString(string *out, unsigned short seq) const;
  virtual void ToString(char *buf, int len) const;
  virtual void ToString(char *buf, int len, unsigned short seq) const;

  bool HdrCheck(int content_len) { return content_len >= Header::size + HeaderSize(); }
  bool    Check(int content_len) { return content_len >= Header::size +       Size(); }
  bool HdrCheck(const Stream *is) { return HdrCheck(is->Len()); }
  bool    Check(const Stream *is) { return    Check(is->Len()); }
  int      Read(const Stream *is) { if (!HdrCheck(is)) return -1; return In(is); }
};

}; // namespace LFL
#endif // LFL_CORE_APP_TYPES_STRING_H__
