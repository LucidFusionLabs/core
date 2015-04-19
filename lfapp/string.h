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

#ifndef __LFL_LFAPP_STRING_H__
#define __LFL_LFAPP_STRING_H__

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
    static const unsigned short replacement_char = 0xFFFD;
};
    
typedef basic_string<short> String16;
template <class X> string ToString(const X& x) { std::stringstream in; in << x; return in.str(); }
string StringPrintf(const char *fmt, ...);

struct Printable : public string {
    Printable(const void *x);
    Printable(const basic_string<short> &x);
    Printable(const string &x) : string(x) {}
    Printable(const char *x) : string(x) {}
    Printable(      char *x) : string(x) {}
    Printable(const bool &x) : string(ToString(x)) {}
    Printable(const int  &x) : string(ToString(x)) {}
    Printable(const long &x) : string(ToString(x)) {}
    Printable(const unsigned char *x) : string((char*)x) {}
    Printable(const char &x) : string(ToString(x)) {}
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
    static bool     Scan(const bool&,     const char  *v) { return *v ? atoi(v) : true; }
    static int      Scan(const int&,      const char  *v) { return atoi(v); }
    static unsigned Scan(const unsigned&, const char  *v) { return atoi(v); }
    static float    Scan(const float&,    const char  *v) { return atof(v); }
    static double   Scan(const double&,   const char  *v) { return atof(v); }
    static string   Scan(const string&,   const char  *v) { return string(v); }
    static String16 Scan(const String16&, const short *v) { return String16(v); }
};

template <class X> struct ArrayPiece {
    typedef       X*       iterator;
    typedef const X* const_iterator;
    const X *buf; int len;
    ArrayPiece()                  : buf(0), len(0) {}
    ArrayPiece(const X *b, int l) : buf(b), len(l) {}
    const X& operator[](int i) const { return buf[i]; }
    const X& back() const { return buf[len-1]; }
    void clear() { buf=0; len=0; }
    bool null() const { return !buf; }
    bool empty() const { return !buf || len <= 0; }
    bool has_size() const { return len >= 0; }
    int size() const { return max(0, len); }
    void assign(const X *b, int l) { buf=b; len=l; }
    const X *data() const { return buf; }
    const_iterator begin() const { return buf; }
    const_iterator end() const { return buf+len; }
};

template <class X> struct StringPieceT : public ArrayPiece<X> {
    StringPieceT() {}
    StringPieceT(const basic_string<X> &s) : ArrayPiece<X>(s.data(), s.size())  {}
    StringPieceT(const X *b, int l)        : ArrayPiece<X>(b,        l)         {}
    StringPieceT(const X *b)               : ArrayPiece<X>(b,        Length(b)) {}
    basic_string<X> str() const {
        if (this->buf && this->len < 0) return this->buf;
        return this->buf ? basic_string<X>(this->buf, this->len) : basic_string<X>();
    }
    bool Done(const X* p) const { return (this->len >= 0 && p >= this->buf + this->len) || !*p; }
    int Length() const { return this->len >= 0 ? this->len : Length(this->buf); }
    static StringPieceT<X> Unbounded (const X *b) { return StringPieceT<X>(b, -1); }
    static StringPieceT<X> FromString(const X *b) { return StringPieceT<X>(b, b?Length(b):0); }
    static size_t Length(const X *b) { const X *p = b; while (*p) p++; return p - b; }
    static const X *Blank() { static X x[1] = {0}; return x; }
    static const X *Space() { static X x[2] = {' ',0}; return x; }
    static const X *NullSpelled() { static X x[7] = {'<','N','U','L','L','>',0}; return x; }
};
typedef StringPieceT<char> StringPiece;
typedef StringPieceT<short> String16Piece;

struct String {
    template          <class Y> static void Copy(const string          &in, basic_string<Y> *out, int offset=0) { return Copy<char, Y>(in, out, offset); }
    template          <class Y> static void Copy(const String16        &in, basic_string<Y> *out, int offset=0) { return Copy<short,Y>(in, out, offset); }
    template <class X, class Y> static void Copy(const StringPieceT<X> &in, basic_string<Y> *out, int offset=0) {
        out->resize(offset + in.len);
        Y* o = &(*out)[offset];
        for (const X *i = in.buf; !in.Done(i); /**/) *o++ = *i++;
    }
    template          <class Y> static void Append(const string          &in, basic_string<Y> *out) { return Append<char, Y>(in, out); }
    template          <class Y> static void Append(const String16        &in, basic_string<Y> *out) { return Append<short,Y>(in, out); }
    template <class X, class Y> static void Append(const StringPieceT<X> &in, basic_string<Y> *out) {
        Copy(in.data(), out, out->size());
    }
    template          <class Y> static int Convert(const string          &in, basic_string<Y> *out, const char *fe, const char *te) { return Convert<char,  Y>(in, out, fe, te); }
    template          <class Y> static int Convert(const String16        &in, basic_string<Y> *out, const char *fe, const char *te) { return Convert<short, Y>(in, out, fe, te); }
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
    static int ReadGlyph(const String16Piece &s, const short *p, int *l, bool eof=0);
};

template <class X> struct UTF {};
template <> struct UTF<char> {
    static string WriteGlyph(int codepoint) { return UTF8::WriteGlyph(codepoint); }
    static int ReadGlyph(const StringPiece   &s, const char  *p, int *l, bool eof=0) { return UTF8::ReadGlyph(s, p, l, eof); }
    static int ReadGlyph(const String16Piece &s, const short *p, int *l, bool eof=0) { FATALf("%s", "no such thing as 16bit UTF-8"); }
};
template <> struct UTF<short> {
    static String16 WriteGlyph(int codepoint) { return UTF16::WriteGlyph(codepoint); }
    static int ReadGlyph(const String16Piece &s, const short *p, int *l, bool eof=0) { return UTF16::ReadGlyph(s, p, l, eof); }
    static int ReadGlyph(const StringPiece   &s, const char  *p, int *l, bool eof=0) { FATALf("%s", "no such thing as 8bit UTF-16"); }
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
};
typedef StringIterT<char> StringIter;

template <class X> basic_string<typename X::type> IterNextString(X *iter) {
    const typename X::type *w = iter->Next();
    return w ? basic_string<typename X::type>(w, iter->CurrentLength()) : basic_string<typename X::type>();
}

template <class X> basic_string<typename X::type> IterRemainingString(X *iter) {
    basic_string<typename X::type> ret;
    int total = iter->TotalLength(), offset = iter->CurrentOffset();
    if (total >= 0) ret.assign(iter->Begin() + offset, total - offset);
    else            ret.assign(iter->Begin() + offset);
    return ret;
}

template <class X, class Y> void IterScanN(X *iter, Y *out, int N) {
    for (int i=0; i<N; ++i) {
        auto v = iter->Next(); 
        out[i] = v ? Scannable::Scan(Y(), string(v, iter->CurrentLength()).c_str()) : 0;
    }
}

template <class X> struct StringWordIterT : public StringIterT<X> {
    const X *in=0;
    int size=0, cur_len=0, cur_offset=0, next_offset=0, (*IsSpace)(int)=0, (*IsQuote)(int)=0, flag=0; 
    StringWordIterT() {}
    StringWordIterT(const X *buf, int len,    int (*IsSpace)(int)=0, int(*IsQuote)(int)=0, int Flag=0);
    StringWordIterT(const StringPieceT<X> &b, int (*IsSpace)(int)=0, int(*IsQuote)(int)=0, int Flag=0) :
        StringWordIterT(b.buf, b.len, IsSpace, IsQuote, Flag) {}
    bool Done() const { return cur_offset < 0; }
    const X *Next();
    const X *Begin() const { return in; }
    const X *Current() const { return in + cur_offset; }
    int CurrentOffset() const { return cur_offset; }
    int CurrentLength() const { return cur_len; }
    int TotalLength() const { return size; }
};
typedef StringWordIterT<char>  StringWordIter;
typedef StringWordIterT<short> StringWord16Iter;

template <class X> struct StringLineIterT : public StringIterT<X> {
    struct Flag { enum { BlankLines=1 }; };
    const X *in;
    basic_string<X> buf;
    int size=0, cur_len=0, cur_offset=0, next_offset=0, flag=0; bool first=0;
    StringLineIterT(const StringPieceT<X> &B, int F=0) : in(B.buf), size(B.len), flag(F), first(1) {}
    StringLineIterT() : cur_offset(-1) {}
    bool Done() const { return cur_offset < 0; }
    const X *Next();
    const X *Begin() const { return in; }
    const X *Current() const { return in + cur_offset; }
    int CurrentOffset() const { return cur_offset; }
    int CurrentLength() const { return cur_len; }
    int TotalLength() const { return size; }
};
typedef StringLineIterT<char>  StringLineIter;
typedef StringLineIterT<short> StringLine16Iter;

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

template <int V>          int                 isint (int N) { return N == V; }
template <int V1, int V2> int                 isint2(int N) { return (N == V1) || (N == V2); }
template <int V1, int V2, int V3>         int isint3(int N) { return (N == V1) || (N == V2) || (N == V3); }
template <int V1, int V2, int V3, int V4> int isint4(int N) { return (N == V1) || (N == V2) || (N == V3) || (N == V4); }
int IsAscii(int c);
int isfileslash(int c);
int isdot(int c);
int iscomma(int c);
int isand(int c);
int isdquote(int c);
int issquote(int c);
int istick(int c);
int isdig(int c);
int isnum(int c);
int isquote(int c);
int notspace(int c);
int notalpha(int c);
int notalnum(int c);
int notnum(int c);
int notcomma(int c);
int notdot(int c);
float my_atof(const char *v);
inline double atof(const string &v) { return ::atof(v.c_str()); }
inline int    atoi(const string &v) { return ::atoi(v.c_str()); }

int atoi(const char  *v);
int atoi(const short *v);
template <int F, int T>                 int tochar (int i) { return i == F ? T :  i; }
template <int F, int T, int F2, int T2> int tochar2(int i) { return i == F ? T : (i == F2 ? T2 : i); }

string WStringPrintf(const wchar_t *fmt, ...);
String16 String16Printf(const char *fmt, ...);
void StringAppendf(string *out, const char *fmt, ...);
void StringAppendf(String16 *out, const char *fmt, ...);
int sprint(char *out, int len, const char *fmt, ...);

inline string StrCat(const Printable &x1) { return x1; }
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

bool PrefixMatch(const short    *in, const short    *pref, int case_sensitive=true);
bool PrefixMatch(const char     *in, const char     *pref, int case_sensitive=true);
bool PrefixMatch(const char     *in, const string   &pref, int case_sensitive=true);
bool PrefixMatch(const string   &in, const char     *pref, int case_sensitive=true);
bool PrefixMatch(const string   &in, const string   &pref, int case_sensitive=true);
bool PrefixMatch(const String16 &in, const String16 &pref, int case_sensitive=true);
bool PrefixMatch(const String16 &in, const char     *pref, int case_sensitive=true);

bool SuffixMatch(const short    *in, const short    *pref, int case_sensitive=true);
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
bool StringEquals(const short    *s1, const short    *s2, int case_sensitive=false);

bool StringEmptyOrEquals(const string   &in, const string   &ref, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const String16 &ref, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const string   &ref, int case_sensitive=false);
bool StringEmptyOrEquals(const string   &in, const string   &ref1, const string   &ref2, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const String16 &ref1, const String16 &ref2, int case_sensitive=false);
bool StringEmptyOrEquals(const String16 &in, const string   &ref1, const string   &ref2, int case_sensitive=false);
bool StringReplace(string *text, const string &needle, const string &replace);

template <class X, class Y> int Split(const X *in, int (*ischar)(int), int (*isquote)(int), vector<Y> *out) {
    out->clear();
    if (!in) return 0;
    StringWordIterT<X> words(in, ischar, isquote);
    for (string word = IterNextString(&words); !words.Done(); word = IterNextString(&words))
        out->push_back(Scannable::Scan(Y(), word.c_str()));
    return out->size();
}
template <class X> int Split(const string   &in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<char,  X>(in.c_str(), ischar, isquote, out); }
template <class X> int Split(const string   &in, int (*ischar)(int),                      vector<X> *out) { return Split<char,  X>(in.c_str(), ischar, NULL,    out); }
template <class X> int Split(const char     *in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<char,  X>(in, ischar, isquote, out); }
template <class X> int Split(const char     *in, int (*ischar)(int),                      vector<X> *out) { return Split<char,  X>(in, ischar, NULL,    out); }
template <class X> int Split(const String16 &in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<short, X>(in.c_str(), ischar, isquote, out); }
template <class X> int Split(const String16 &in, int (*ischar)(int),                      vector<X> *out) { return Split<short, X>(in.c_str(), ischar, NULL,    out); }
template <class X> int Split(const short    *in, int (*ischar)(int), int (*isquote)(int), vector<X> *out) { return Split<short, X>(in, ischar, isquote, out); }
template <class X> int Split(const short    *in, int (*ischar)(int),                      vector<X> *out) { return Split<short, X>(in, ischar, NULL,    out); }

template <class X> int Split(const char   *in, int (*ischar)(int), int (*isquote)(int), set<X> *out) {
    out->clear(); if (!in) return 0;
    StringWordIter words(in, ischar, isquote);
    for (string word = IterNextString(&words); !words.Done(); word = IterNextString(&words))
        out->insert(Scannable::Scan(X(), word.c_str()));
    return out->size();
}
template <class X> int Split(const char   *in, int (*ischar)(int),                      set<X> *out) { return Split(in, ischar, NULL, out); }
template <class X> int Split(const string &in, int (*ischar)(int), int (*isquote)(int), set<X> *out) { return Split(in.c_str(), ischar, isquote, out); }
template <class X> int Split(const string &in, int (*ischar)(int),                      set<X> *out) { return Split(in, ischar, NULL, out); }

int Split(const char   *in, int (*ischar)(int), string *left, string *right);
int Split(const string &in, int (*ischar)(int), string *left, string *right);
void Join(string *out, const vector<string> &in);
void Join(string *out, const vector<string> &in, int inB, int inE);
string Join(const vector<string> &strs, const string &separator);
string Join(const vector<string> &strs, const string &separator, int beg_ind, int end_ind);
string strip(const char *s, int (*stripchar)(int), int (*stripchar2)(int)=0);
string togrep(const char *s, int (*grepchar)(int), int (*grepchar2)(int)=0);
string   toconvert(const char     *text, int (*tochar)(int), int (*ischar)(int)=0);
string   toconvert(const string   &text, int (*tochar)(int), int (*ischar)(int)=0);
String16 toconvert(const short    *text, int (*tochar)(int), int (*ischar)(int)=0);
String16 toconvert(const String16 &text, int (*tochar)(int), int (*ischar)(int)=0);
string   toupper(const char     *text);
string   toupper(const string   &text);
String16 toupper(const short    *text);
String16 toupper(const String16 &text);
string   tolower(const char     *text);
string   tolower(const string   &text);
String16 tolower(const short    *text);
String16 tolower(const String16 &text);
string   ReplaceEmpty (const string   &in, const string   &replace_with);
String16 ReplaceEmpty (const String16 &in, const string   &replace_with);
String16 ReplaceEmpty (const String16 &in, const String16 &replace_with);
string ReplaceNewlines(const string   &in, const string   &replace_with);
template <class X> string CHexEscape        (const basic_string<X> &text);
template <class X> string CHexEscapeNonAscii(const basic_string<X> &text);

const char  *NextLine   (const StringPiece   &text, bool final=0, int *outlen=0);
const short *NextLine   (const String16Piece &text, bool final=0, int *outlen=0);
const char  *NextLineRaw(const StringPiece   &text, bool final=0, int *outlen=0);
const short *NextLineRaw(const String16Piece &text, bool final=0, int *outlen=0);
const char  *NextProto  (const StringPiece   &text, bool final=0, int *outlen=0);
template <int S> const char *NextChunk(const StringPiece &text, bool final=0, int *outlen=0) {
    int add = final ? max(text.len, 0) : S;
    if (text.len < add) return 0;
    *outlen = add;
    return text.buf + add;
}
template <class X>       X *NextChar(      X *text, int (*ischar)(int),                      int len=-1, int *outlen=0);
template <class X> const X *NextChar(const X *text, int (*ischar)(int),                      int len=-1, int *outlen=0);
template <class X>       X *NextChar(      X *text, int (*ischar)(int), int (*isquote)(int), int len=-1, int *outlen=0);
template <class X> const X *NextChar(const X *text, int (*ischar)(int), int (*isquote)(int), int len=-1, int *outlen=0);
template <class X> int  LengthChar(const StringPieceT<X> &text, int (*ischar)(int));
template <class X> int RLengthChar(const StringPieceT<X> &text, int (*ischar)(int));
template <class X> int SkipChar(int (*ischar)(int), const X* in, int len=0);

unsigned           fnv32(const void *buf, unsigned len=0, unsigned           hval=0);
unsigned long long fnv64(const void *buf, unsigned len=0, unsigned long long hval=0);

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

struct Regex {
    struct Result {
        int begin, end;
        Result(int B=0, int E=0) : begin(B), end(E) {}
        string Text(const string &t) const { return t.substr(begin, end - begin); }
        float FloatVal(const string &t) const { return atof(Text(t).c_str()); }
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

struct Base64 {
    string encoding_table, decoding_table;
    int mod_table[3];
    Base64();

    string Encode(const char *in,   size_t input_length);
    string Decode(const char *data, size_t input_length);
};

}; // namespace LFL
#endif // __LFL_LFAPP_STRING_H__
