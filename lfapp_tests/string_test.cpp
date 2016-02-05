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

#include "gtest/gtest.h"
#include "lfapp/lfapp.h"

namespace LFL {

TEST(StringTest, Bit) {
  EXPECT_EQ(0, Bit::Count(0));
  EXPECT_EQ(1, Bit::Count(1));
  EXPECT_EQ(1, Bit::Count(2));
  EXPECT_EQ(8, Bit::Count(255));

  int bit_indices[65];
  Bit::Indices(1LL<<34 | 1<<30 | 1<<14, bit_indices);
  EXPECT_EQ(14, bit_indices[0]);
  EXPECT_EQ(30, bit_indices[1]);
  EXPECT_EQ(34, bit_indices[2]);
  EXPECT_EQ(-1, bit_indices[3]);
}

TEST(StringTest, BitString) {
  { unsigned char bits[] = { 0,   0,   0,   0,   0,   0   }; EXPECT_EQ(-1, BitString::FirstSet(bits, sizeof(bits))); }
  { unsigned char bits[] = { 0,   0,   0,   0,   0,   1   }; EXPECT_EQ(40, BitString::FirstSet(bits, sizeof(bits))); }
  { unsigned char bits[] = { 0,   0,   0,   0,   0,   2   }; EXPECT_EQ(41, BitString::FirstSet(bits, sizeof(bits))); }

  { unsigned char bits[] = { 255, 255, 255, 255, 255, 255 }; EXPECT_EQ(-1, BitString::LastClear (bits, sizeof(bits))); }
  { unsigned char bits[] = { 255, 255, 255, 255, 255, 255 }; EXPECT_EQ(-1, BitString::FirstClear(bits, sizeof(bits))); }
  { unsigned char bits[] = { 255, 255, 255, 255, 255, 254 }; EXPECT_EQ(40, BitString::FirstClear(bits, sizeof(bits))); }
  { unsigned char bits[] = { 255, 255, 255, 255, 255, 253 }; EXPECT_EQ(41, BitString::FirstClear(bits, sizeof(bits))); }

  {
    unsigned char bits[] = { 255, 255, 255, 255, 255, 0 };
    EXPECT_EQ(40, BitString::FirstClear(bits, sizeof(bits)));
    BitString::Set(bits, 40);
    EXPECT_EQ(41, BitString::FirstClear(bits, sizeof(bits)));
    BitString::Set(bits, 41);
    EXPECT_EQ(42, BitString::FirstClear(bits, sizeof(bits)));
    BitString::Clear(bits, 41);
    EXPECT_EQ(41, BitString::FirstClear(bits, sizeof(bits)));
    BitString::Clear(bits, 11);
    EXPECT_EQ(11, BitString::FirstClear(bits, sizeof(bits)));
    BitString::Clear(bits, 7);
    EXPECT_EQ(7,  BitString::FirstClear(bits, sizeof(bits)));
    EXPECT_EQ(127, bits[0]);
  }
}

TEST(StringTest, UTF) {
  unsigned char utf8[] = { 0xE9, 0xAB, 0x98, 0 }; int gl=0;
  EXPECT_EQ(0x9AD8, UTF8::ReadGlyph((char*)utf8, (char*)utf8, &gl));
  EXPECT_EQ(3,      gl);
}

TEST(StringTest, WordIter) {
  //          01234567890123456789012345678901234567890
  string b = "aaaaaaaa bbbbbb ccccc  dd         eeeee ffffff          ggggggg     hhhhhhhhhhhh  kkkk";
  const char *word;
  {
    StringWordIter words(StringPiece(b.data(), 40), isspace, 0);
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "aaaaaaaa", words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "bbbbbb",   words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "ccccc",    words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "dd",       words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "eeeee",    words.cur_len));
    EXPECT_EQ(nullptr, ((word = words.Next())));
  }
  {
    StringWordIter words(StringPiece(b.data(), 39), isspace, 0);
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "aaaaaaaa", words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "bbbbbb",   words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "ccccc",    words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "dd",       words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "eeeee",    words.cur_len));
    EXPECT_EQ(nullptr, ((word = words.Next())));
  }
  {
    StringWordIter words(StringPiece(b.data(), 38), isspace, 0);
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "aaaaaaaa", words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "bbbbbb",   words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "ccccc",    words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "dd",       words.cur_len));
    EXPECT_NE(nullptr, ((word = words.Next()))); EXPECT_EQ(0, strncmp(word, "eeee",     words.cur_len));
    EXPECT_EQ(nullptr, ((word = words.Next())));
  }
  {
    StringWordIter words("");
    EXPECT_EQ("", words.NextString());
    EXPECT_EQ(true, words.Done());
  }
}

TEST(IterTest, LineIter) {
  string b = "1 2 3\n\n4 5 6";
  StringLineIter line(b);
  EXPECT_EQ("1 2 3", line.NextString());
  EXPECT_EQ("4 5 6", line.NextString());
}

TEST(StringTest, StringAppendf) {
#define ToUTF16(x) String::ToUTF16(x)
#define ToUTF8(x) String::ToUTF8(x)
  string ret; String16 reu;
  StringAppendf(&ret, "y0y0\n");
  StringAppendf(&reu, "y0y0\n");
  EXPECT_EQ("y0y0\n",        ret);
  EXPECT_EQ("y0y0\n", ToUTF8(reu));
  StringAppendf(&ret, "h0h0\n");
  StringAppendf(&reu, "h0h0\n");
  EXPECT_EQ("y0y0\nh0h0\n",        ret);
  EXPECT_EQ("y0y0\nh0h0\n", ToUTF8(reu));

  string   a="aX",b="bZ",c="cD",d="dG",e="eX",f="fG",z;
  String16 A=String::ToUTF16("aX"),B=String::ToUTF16("bZ"),C=String::ToUTF16("cD"),D=String::ToUTF16("dG"),E=String::ToUTF16("eX"),F=String::ToUTF16("fG"),Z;
  EXPECT_EQ((a+b),                           StrCat(a,b));
  EXPECT_EQ((a+b+c),                         StrCat(a,b,c));
  EXPECT_EQ((a+b+c+d),                       StrCat(a,b,c,d));
  EXPECT_EQ((a+b+c+d+e),                     StrCat(a,b,c,d,e));
  EXPECT_EQ((a+b+c+d+e+f),                   StrCat(a,b,c,d,e,f));
  EXPECT_EQ((a+b+c+d+e+f+b),                 StrCat(a,b,c,d,e,f,b));
  EXPECT_EQ((a+b+c+d+e+f+b+e),               StrCat(a,b,c,d,e,f,b,e));
  EXPECT_EQ((a+b+c+d+e+f+b+e+b),             StrCat(a,b,c,d,e,f,b,e,b));
  EXPECT_EQ((a+b+c+d+e+f+b+e+b+a),           StrCat(a,b,c,d,e,f,b,e,b,a));
  EXPECT_EQ((a+b+c+d+e+f+b+e+b+a+f),         StrCat(a,b,c,d,e,f,b,e,b,a,f));
  EXPECT_EQ((a+b+c+d+e+f+b+e+b+a+f+c),       StrCat(a,b,c,d,e,f,b,e,b,a,f,c));
  EXPECT_EQ((a+b+c+d+e+f+b+e+b+a+f+c+d),     StrCat(a,b,c,d,e,f,b,e,b,a,f,c,d));
  EXPECT_EQ((a+b+c+d+e+f+b+e+b+a+f+c+d+a),   StrCat(a,b,c,d,e,f,b,e,b,a,f,c,d,a));
  EXPECT_EQ((a+b+c+d+e+f+b+e+b+a+f+c+d+a+c), StrCat(a,b,c,d,e,f,b,e,b,a,f,c,d,a,c));

  z=ret; StrAppend(&z, a);                         EXPECT_EQ((ret+a), z);
  Z=reu; StrAppend(&Z, A);                         EXPECT_EQ((reu+A), Z);
  Z=reu; StrAppend(&Z, a);                         EXPECT_EQ((ret+a), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b);                       EXPECT_EQ((ret+a+b), z);
  Z=reu; StrAppend(&Z, A,B);                       EXPECT_EQ((reu+A+B), Z);
  Z=reu; StrAppend(&Z, a,b);                       EXPECT_EQ((ret+a+b), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c);                     EXPECT_EQ((ret+a+b+c), z);
  Z=reu; StrAppend(&Z, A,B,C);                     EXPECT_EQ((reu+A+B+C), Z);
  Z=reu; StrAppend(&Z, a,b,c);                     EXPECT_EQ((ret+a+b+c), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d);                   EXPECT_EQ((ret+a+b+c+d), z);
  Z=reu; StrAppend(&Z, A,B,C,D);                   EXPECT_EQ((reu+A+B+C+D), Z);
  Z=reu; StrAppend(&Z, a,b,c,d);                   EXPECT_EQ((ret+a+b+c+d), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e);                 EXPECT_EQ((ret+a+b+c+d+e), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E);                 EXPECT_EQ((reu+A+B+C+D+E), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e);                 EXPECT_EQ((ret+a+b+c+d+e), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f);               EXPECT_EQ((ret+a+b+c+d+e+f), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F);               EXPECT_EQ((reu+A+B+C+D+E+F), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f);               EXPECT_EQ((ret+a+b+c+d+e+f), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f,b);             EXPECT_EQ((ret+a+b+c+d+e+f+b), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F,B);             EXPECT_EQ((reu+A+B+C+D+E+F+B), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f,b);             EXPECT_EQ((ret+a+b+c+d+e+f+b), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f,b,e);           EXPECT_EQ((ret+a+b+c+d+e+f+b+e), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F,B,E);           EXPECT_EQ((reu+A+B+C+D+E+F+B+E), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f,b,e);           EXPECT_EQ((ret+a+b+c+d+e+f+b+e), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f,b,e,a);         EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F,B,E,A);         EXPECT_EQ((reu+A+B+C+D+E+F+B+E+A), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f,b,e,a);         EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f,b,e,a,c);       EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F,B,E,A,C);       EXPECT_EQ((reu+A+B+C+D+E+F+B+E+A+C), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f,b,e,a,c);       EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f,b,e,a,c,b);     EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c+b), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F,B,E,A,C,B);     EXPECT_EQ((reu+A+B+C+D+E+F+B+E+A+C+B), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f,b,e,a,c,b);     EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c+b), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f,b,e,a,c,b,d);   EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c+b+d), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F,B,E,A,C,B,D);   EXPECT_EQ((reu+A+B+C+D+E+F+B+E+A+C+B+D), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f,b,e,a,c,b,d);   EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c+b+d), String::ToUTF8(Z));
  z=ret; StrAppend(&z, a,b,c,d,e,f,b,e,a,c,b,d,a); EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c+b+d+a), z);
  Z=reu; StrAppend(&Z, A,B,C,D,E,F,B,E,A,C,B,D,A); EXPECT_EQ((reu+A+B+C+D+E+F+B+E+A+C+B+D+A), Z);
  Z=reu; StrAppend(&Z, a,b,c,d,e,f,b,e,a,c,b,d,a); EXPECT_EQ((ret+a+b+c+d+e+f+b+e+a+c+b+d+a), String::ToUTF8(Z));
}

TEST(StringTest, StringMatch) {
  EXPECT_TRUE (PrefixMatch(        "y0y0",          "y0"));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"),         "y0"));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"), ToUTF16("y0")));

  EXPECT_FALSE(PrefixMatch(        "y0y0",          "0y"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"),         "0y"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"), ToUTF16("0y")));

  EXPECT_TRUE (PrefixMatch(        "y0y0",          "y0y0"));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"),         "y0y0"));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"), ToUTF16("y0y0")));

  EXPECT_FALSE(PrefixMatch(        "y0y0",          "0y0y0y"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"),         "0y0y0y"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"), ToUTF16("0y0y0y")));

  EXPECT_FALSE(PrefixMatch(        "y0y0",          "Y0"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"),         "Y0"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0")));

  EXPECT_TRUE (PrefixMatch(        "y0y0",          "Y0",  false));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"),         "Y0",  false));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0"), false));

  EXPECT_FALSE(PrefixMatch(        "y0y0",          "Y0Y0"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"),         "Y0Y0"));
  EXPECT_FALSE(PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0")));

  EXPECT_TRUE (PrefixMatch(        "y0y0",          "Y0Y0",  false));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"),         "Y0Y0",  false));
  EXPECT_TRUE (PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0"), false));

  EXPECT_TRUE (SuffixMatch(        "y0y0",          "y0"));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"),         "y0"));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"), ToUTF16("y0")));

  EXPECT_FALSE(SuffixMatch(        "y0y0",          "0y"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"),         "0y"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"), ToUTF16("0y")));

  EXPECT_TRUE (SuffixMatch(        "y0y0",          "y0y0"));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"),         "y0y0"));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"), ToUTF16("y0y0")));

  EXPECT_FALSE(SuffixMatch(        "y0y0",          "0y0y0y"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"),         "0y0y0y"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"), ToUTF16("0y0y0y")));

  EXPECT_FALSE(SuffixMatch(        "y0y0",          "Y0"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"),         "Y0"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0")));

  EXPECT_TRUE (SuffixMatch(        "y0y0",          "Y0",  false));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"),         "Y0",  false));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0"), false));

  EXPECT_FALSE(SuffixMatch(        "y0y0",          "Y0Y0"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"),         "Y0Y0"));
  EXPECT_FALSE(SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0")));

  EXPECT_TRUE (SuffixMatch(        "y0y0",          "Y0Y0",  false));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"),         "Y0Y0",  false));
  EXPECT_TRUE (SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0"), false));

  EXPECT_TRUE (StringEquals(        "g0G0",          "G0g0",  false));
  EXPECT_TRUE (StringEquals(ToUTF16("g0G0"),         "G0g0",  false));
  EXPECT_TRUE (StringEquals(ToUTF16("g0G0"), ToUTF16("G0g0"), false));

  EXPECT_TRUE (StringEquals(        "g0G0",          "g0G0",  true));
  EXPECT_TRUE (StringEquals(ToUTF16("g0G0"),         "g0G0",  true));
  EXPECT_TRUE (StringEquals(ToUTF16("g0G0"), ToUTF16("g0G0"), true));

  EXPECT_FALSE(StringEquals(        "g0G0",          "G0g0",  true));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0"),         "G0g0",  true));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0"), ToUTF16("G0g0"), true));

  EXPECT_FALSE(StringEquals(        "g0G0",          "G0g0z",  false));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0"),         "G0g0z",  false));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0"), ToUTF16("G0g0z"), false));

  EXPECT_FALSE(StringEquals(        "g0G0z",          "G0g0",  false));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0z"),         "G0g0",  false));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0z"), ToUTF16("G0g0"), false));

  EXPECT_FALSE(StringEquals(        "a0G0",          "G0g0z",  false));
  EXPECT_FALSE(StringEquals(ToUTF16("a0G0"),         "G0g0z",  false));
  EXPECT_FALSE(StringEquals(ToUTF16("a0G0"), ToUTF16("G0g0z"), false));

  EXPECT_FALSE(StringEquals(        "g0G0",          "g0G0z",  true));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0"),         "g0G0z",  true));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0"), ToUTF16("g0G0z"), true));

  EXPECT_FALSE(StringEquals(        "g0G0z",          "g0G0",  true));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0z"),         "g0G0",  true));
  EXPECT_FALSE(StringEquals(ToUTF16("g0G0z"), ToUTF16("g0G0"), true));

  string a1 = "foo bar baz bat";
  EXPECT_TRUE (ReplaceString(&a1, "bar baz", "raz"));
  EXPECT_FALSE(ReplaceString(&a1, "bar baz", "raz"));
  EXPECT_EQ(a1, "foo raz bat");
}

TEST(StringTest, Split) {
  string l, r, l1="aaaaa", r1="bbbb", x1="zzqrrrz", z1=", enzoop";

  Split(StrCat(x1, " "), isspace, &l);
  EXPECT_EQ(x1, l);

  Split(StrCat(l1, " \t\r\n", r1).c_str(), isspace, &l, &r);
  EXPECT_EQ(l1, l); EXPECT_EQ(r1, r);

  vector<string> v;
  Split(StrCat(l1, ", ", r1, ", ", ", ", x1).c_str(), isint2<',', ' '>, &v);
  CHECK_EQ(3, v.size());
  EXPECT_EQ(l1, v[0]);
  EXPECT_EQ(r1, v[1]);
  EXPECT_EQ(x1, v[2]);

  v.clear();
  Split(StrCat(l1, ", \"", r1, z1, "\", ", x1).c_str(), isint2<',', ' '>, isint<'"'>, &v);
  CHECK_EQ(3, v.size());
  EXPECT_EQ(l1, v[0]);
  EXPECT_EQ(x1, v[2]);
  EXPECT_EQ(StrCat("\"", r1, z1, "\""), v[1]);

  vector<int> vi;
  Split("1,  34  ,  29,   48", isint2<',', ' '>, &vi);
  CHECK_EQ(4,   vi.size());
  EXPECT_EQ(1,  vi[0]);
  EXPECT_EQ(34, vi[1]);
  EXPECT_EQ(29, vi[2]);
  EXPECT_EQ(48, vi[3]);

  vector<string> dotv;
  Split("com.", isdot, &dotv);
  EXPECT_EQ(1, dotv.size());
  EXPECT_EQ("com", dotv[0]);
}

TEST(StringTest, CHexEscape) {
  EXPECT_EQ("a\\x08a", CHexEscapeNonAscii(string("a\x08""a")));
}

TEST(StringTest, RLengthChar) {
  char b1[] = " ", b2[] = "  ", b3[] = "aaa   ";
  EXPECT_EQ(strlen(b1), RLengthChar(b1+strlen(b1)-1, isspace, strlen(b1)));
  EXPECT_EQ(strlen(b2), RLengthChar(b2+strlen(b2)-1, isspace, strlen(b2)));
  EXPECT_EQ(3,          RLengthChar(b3+strlen(b3)-1, isspace, strlen(b3)));
  EXPECT_EQ(2,          RLengthChar(b3+strlen(b3)-1, isspace, 2         ));
}

TEST(StringTest, FNVHash) {
  char text1[] = "The quick brown fox jumped over the lazy dog.";
  EXPECT_EQ(583960891,             fnv32(text1));
  EXPECT_EQ(6903599980961175675LL, fnv64(text1));
}

TEST(StringTest, BaseName) {
  int len;
  EXPECT_EQ(string(BaseName("/s/foo.bar",     &len)), "foo.bar"); EXPECT_EQ(3, len);
  EXPECT_EQ(string(BaseName("aa/bb/s/foobar", &len)), "foobar");  EXPECT_EQ(6, len);

  EXPECT_EQ(0, DirNameLen("foo.bar"));
  EXPECT_EQ(2, DirNameLen("/s/foo.bar"));
  EXPECT_EQ(3, DirNameLen("/s/foo.bar", 1));

  char foo[] = "aa/bb/s/foobar";
  EXPECT_EQ(7, DirNameLen(foo));
  EXPECT_EQ(8, DirNameLen(foo, 1));
  EXPECT_EQ(string("aa/bb/s"),  string(foo, DirNameLen(foo)));
  EXPECT_EQ(string("aa/bb/s/"), string(foo, DirNameLen(foo, 1)));

  string bar = "aa/bb/s/foobar", baz;
  EXPECT_EQ(7, DirNameLen(bar));
  EXPECT_EQ(8, DirNameLen(bar, 1));
  EXPECT_EQ(string("aa/bb/s"),  string(bar.c_str(), DirNameLen(bar)));
  EXPECT_EQ(string("aa/bb/s/"), string(bar.c_str(), DirNameLen(bar, 1)));

  baz.assign(bar, 0, DirNameLen(bar, 1));
  EXPECT_EQ(string("aa/bb/s/"), baz);

  baz.assign(bar, 0, DirNameLen("baz", 1));
  EXPECT_TRUE (baz.empty());

  EXPECT_TRUE (BaseDir(   "wav/foo.bar", "wav"));
  EXPECT_FALSE(BaseDir(   "wav/foo.bar", "Mav"));
  EXPECT_FALSE(BaseDir(   "wav/foo.bar", "wa" ));
  EXPECT_FALSE(BaseDir(   "wav/foo.bar",  "av"));
  EXPECT_TRUE (BaseDir(  "/wav/foo.bar", "wav"));
  EXPECT_FALSE(BaseDir(  "/wav/foo.bar", "wa" ));
  EXPECT_FALSE(BaseDir(  "/wav/foo.bar",  "av"));
  EXPECT_FALSE(BaseDir(  "/wav/foo.bar", "Mav"));
  EXPECT_TRUE (BaseDir("/s/wav/foo.bar", "wav"));
  EXPECT_FALSE(BaseDir("/s/wav/foo.bar", "Mav"));
  EXPECT_FALSE(BaseDir("/s/wav/foo.bar", "wa" ));
  EXPECT_FALSE(BaseDir("/s/wav/foo.bar",  "av"));
}

TEST(RegexTest, URL) {
  Regex url_matcher("https?://");
  vector<Regex::Result> matches;

  string in = "aa http://foo bb";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(in));
  matches.clear();

  in = "aa https://foo bb";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("https://", matches[0].Text(in));
  matches.clear();
}

TEST(RegexTest, StreamURL) {
  StreamRegex url_matcher("https?://");
  vector<Regex::Result> matches;

  string in = "aa http://foo bb";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(in));
  matches.clear();

  in = "aa https://foo bb xxxxxxxxxxx ";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("https://", matches[0].Text(in));
  matches.clear();

  in = " nothing of note here ";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(0, matches.size());
  matches.clear();

  in = "aa zzzzzzzzzzzz ddddddddddd https://foo qqqqq bb";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("https://", matches[0].Text(in));
  matches.clear();

  in = "drzzz http://coo rrrz";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(in));
  matches.clear();

  in = " and um htt";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(0, matches.size());

  in = "p://doo ddd ";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ(-3, matches[0].begin); EXPECT_EQ(4, matches[0].end);
}

TEST(RegexTest, AhoCorasickURL) {
  AhoCorasickFSM<char> url_matcher({"http://", "https://"});
  vector<Regex::Result> matches;

  string in = "aa http://foo bb";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(in));
  matches.clear();

  in = "aa https://foo bb xxxxxxxxxxx ";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("https://", matches[0].Text(in));
  matches.clear();

  in = " nothing of note here ";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(0, matches.size());
  matches.clear();

  in = "aa zzzzzzzzzzzz ddddddddddd https://foo qqqqq bb";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("https://", matches[0].Text(in));
  matches.clear();

  in = "drzzz http://coo rrrz";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(in));
  matches.clear();

  in = " and um htt";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(0, matches.size());

  in = "p://doo ddd ";
  url_matcher.Match(in, &matches);
  EXPECT_EQ(1, matches.size()); EXPECT_EQ(-3, matches[0].begin); EXPECT_EQ(4, matches[0].end);
}

TEST(RegexTest, StringMatcherURL) {
  AhoCorasickFSM<char> url_fsm({"http://", "https://"});
  StringMatcher<char> url_matcher(&url_fsm);

  string in = "aa http://foo bb";
  StringMatcher<char>::iterator mi = url_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("aa http://", string(mi.b, mi.nb));
  EXPECT_EQ(0, mi.MatchBegin()); EXPECT_EQ(0, mi.Matching()); EXPECT_EQ(0, mi.MatchEnd());
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("foo", string(mi.b, mi.nb));
  EXPECT_EQ(1, mi.MatchBegin()); EXPECT_EQ(1, mi.Matching()); EXPECT_EQ(1, mi.MatchEnd());
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ(" bb", string(mi.b, mi.nb));
  EXPECT_EQ(0, mi.MatchBegin()); EXPECT_EQ(0, mi.Matching()); EXPECT_EQ(0, mi.MatchEnd());
  ++mi;
  EXPECT_EQ(mi.b, mi.e);

  in = "aa https://foo bb xxxxxxxxxxx ";
  mi = url_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("aa https://", string(mi.b, mi.nb));
  EXPECT_EQ(0, mi.MatchBegin()); EXPECT_EQ(0, mi.Matching()); EXPECT_EQ(0, mi.MatchEnd());
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("foo", string(mi.b, mi.nb));
  EXPECT_EQ(1, mi.MatchBegin()); EXPECT_EQ(1, mi.Matching()); EXPECT_EQ(1, mi.MatchEnd());
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ(" bb xxxxxxxxxxx ", string(mi.b, mi.nb));
  EXPECT_EQ(0, mi.MatchBegin()); EXPECT_EQ(0, mi.Matching()); EXPECT_EQ(0, mi.MatchEnd());
  ++mi;
  EXPECT_EQ(mi.b, mi.e);

  in = " nothing of note here ";
  mi = url_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ(in, string(mi.b, mi.nb));
  ++mi;
  EXPECT_EQ(mi.b, mi.e);

  in = " and um htt";
  mi = url_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ(in, string(mi.b, mi.nb));
  ++mi;
  EXPECT_EQ(mi.b, mi.e);

  in = "p://doo ddd ";
  mi = url_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("p://", string(mi.b, mi.nb));
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("doo", string(mi.b, mi.nb));
  EXPECT_EQ(1, mi.MatchBegin()); EXPECT_EQ(1, mi.Matching()); EXPECT_EQ(1, mi.MatchEnd());
  EXPECT_EQ(-3, url_matcher.match.begin);
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ(" ddd ", string(mi.b, mi.nb));
  EXPECT_EQ(0, mi.MatchBegin()); EXPECT_EQ(0, mi.Matching()); EXPECT_EQ(0, mi.MatchEnd());
  ++mi;
  EXPECT_EQ(mi.b, mi.e);

  AhoCorasickFSM<char> line_fsm({"\n<12>"});
  StringMatcher<char> line_matcher(&line_fsm);
  line_matcher.match_end_condition = &isint<'\n'>;

  in = "y0y0y0\n<12";
  mi = line_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("y0y0y0\n<12", string(mi.b, mi.nb));
  ++mi;
  EXPECT_EQ(mi.b, mi.e);

  in = ">line1\n<12";
  mi = line_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ(">", string(mi.b, mi.nb));
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("line1", string(mi.b, mi.nb));
  EXPECT_EQ(1, mi.MatchBegin()); EXPECT_EQ(1, mi.Matching()); EXPECT_EQ(1, mi.MatchEnd());
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("\n<12", string(mi.b, mi.nb));
  ++mi;
  EXPECT_EQ(mi.b, mi.e);

  in = ">line2\n<12";
  mi = line_matcher.Begin(in);
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ(">", string(mi.b, mi.nb));
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("line2", string(mi.b, mi.nb));
  EXPECT_EQ(1, mi.MatchBegin()); EXPECT_EQ(1, mi.Matching()); EXPECT_EQ(1, mi.MatchEnd());
  ++mi;
  EXPECT_NE(mi.b, mi.e);
  EXPECT_EQ("\n<12", string(mi.b, mi.nb));
  ++mi;
  EXPECT_EQ(mi.b, mi.e);
}

TEST(RegexTest, AhoCorasick) {
  AhoCorasickFSM<char> matcher({"he", "hers", "his", "she"});
  string in = "ushers";
  vector<Regex::Result> matches;
  matcher.Match(in, &matches);
  EXPECT_EQ(3, matches.size()); 
  EXPECT_EQ("she",  matches[0].Text(in));
  EXPECT_EQ("he",   matches[1].Text(in));
  EXPECT_EQ("hers", matches[2].Text(in));
}

}; // namespace LFL
