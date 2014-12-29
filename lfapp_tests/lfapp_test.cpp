/*
 * $Id: lfapp_test.cpp 1335 2014-12-02 04:13:46Z justin $
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
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"

using namespace LFL;

TEST(StringTest, FNVHash) {
    char text1[] = "The quick brown fox jumped over the lazy dog.";
    EXPECT_EQ(583960891,             fnv32(text1));
    EXPECT_EQ(6903599980961175675LL, fnv64(text1));
}

TEST(StringTest, rlengthchar) {
    char b1[] = " ", b2[] = "  ", b3[] = "aaa   ";
    EXPECT_EQ(strlen(b1), rlengthchar(b1+strlen(b1)-1, isspace, strlen(b1)));
    EXPECT_EQ(strlen(b2), rlengthchar(b2+strlen(b2)-1, isspace, strlen(b2)));
    EXPECT_EQ(3,          rlengthchar(b3+strlen(b3)-1, isspace, strlen(b3)));
    EXPECT_EQ(2,          rlengthchar(b3+strlen(b3)-1, isspace, 2));
}

TEST(StringTest, UTF) {
    unsigned char utf8[] = { 0xE9, 0xAB, 0x98, 0 }; int gl=0;
    EXPECT_EQ(0x9AD8, UTF8::ReadGlyph((char*)utf8, (char*)utf8, &gl));
    EXPECT_EQ(3,      gl);
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
    EXPECT_EQ(true,  PrefixMatch(        "y0y0",          "y0"));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"),         "y0"));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"), ToUTF16("y0")));

    EXPECT_EQ(false, PrefixMatch(        "y0y0",          "0y"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"),         "0y"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"), ToUTF16("0y")));

    EXPECT_EQ(true,  PrefixMatch(        "y0y0",          "y0y0"));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"),         "y0y0"));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"), ToUTF16("y0y0")));

    EXPECT_EQ(false, PrefixMatch(        "y0y0",          "0y0y0y"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"),         "0y0y0y"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"), ToUTF16("0y0y0y")));

    EXPECT_EQ(false, PrefixMatch(        "y0y0",          "Y0"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"),         "Y0"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0")));

    EXPECT_EQ(true,  PrefixMatch(        "y0y0",          "Y0",  false));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"),         "Y0",  false));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0"), false));

    EXPECT_EQ(false, PrefixMatch(        "y0y0",          "Y0Y0"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"),         "Y0Y0"));
    EXPECT_EQ(false, PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0")));

    EXPECT_EQ(true,  PrefixMatch(        "y0y0",          "Y0Y0",  false));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"),         "Y0Y0",  false));
    EXPECT_EQ(true,  PrefixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0"), false));

    EXPECT_EQ(true,  SuffixMatch(        "y0y0",          "y0"));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"),         "y0"));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"), ToUTF16("y0")));

    EXPECT_EQ(false, SuffixMatch(        "y0y0",          "0y"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"),         "0y"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"), ToUTF16("0y")));

    EXPECT_EQ(true,  SuffixMatch(        "y0y0",          "y0y0"));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"),         "y0y0"));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"), ToUTF16("y0y0")));

    EXPECT_EQ(false, SuffixMatch(        "y0y0",          "0y0y0y"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"),         "0y0y0y"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"), ToUTF16("0y0y0y")));

    EXPECT_EQ(false, SuffixMatch(        "y0y0",          "Y0"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"),         "Y0"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0")));

    EXPECT_EQ(true,  SuffixMatch(        "y0y0",          "Y0",  false));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"),         "Y0",  false));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0"), false));

    EXPECT_EQ(false, SuffixMatch(        "y0y0",          "Y0Y0"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"),         "Y0Y0"));
    EXPECT_EQ(false, SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0")));

    EXPECT_EQ(true,  SuffixMatch(        "y0y0",          "Y0Y0",  false));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"),         "Y0Y0",  false));
    EXPECT_EQ(true,  SuffixMatch(ToUTF16("y0y0"), ToUTF16("Y0Y0"), false));

    EXPECT_EQ(true,  StringEquals(        "g0G0",          "G0g0",  false));
    EXPECT_EQ(true,  StringEquals(ToUTF16("g0G0"),         "G0g0",  false));
    EXPECT_EQ(true,  StringEquals(ToUTF16("g0G0"), ToUTF16("G0g0"), false));

    EXPECT_EQ(true,  StringEquals(        "g0G0",          "g0G0",  true));
    EXPECT_EQ(true,  StringEquals(ToUTF16("g0G0"),         "g0G0",  true));
    EXPECT_EQ(true,  StringEquals(ToUTF16("g0G0"), ToUTF16("g0G0"), true));

    EXPECT_EQ(false, StringEquals(        "g0G0",          "G0g0",  true));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0"),         "G0g0",  true));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0"), ToUTF16("G0g0"), true));

    EXPECT_EQ(false, StringEquals(        "g0G0",          "G0g0z",  false));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0"),         "G0g0z",  false));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0"), ToUTF16("G0g0z"), false));

    EXPECT_EQ(false, StringEquals(        "g0G0z",          "G0g0",  false));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0z"),         "G0g0",  false));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0z"), ToUTF16("G0g0"), false));

    EXPECT_EQ(false, StringEquals(        "a0G0",          "G0g0z",  false));
    EXPECT_EQ(false, StringEquals(ToUTF16("a0G0"),         "G0g0z",  false));
    EXPECT_EQ(false, StringEquals(ToUTF16("a0G0"), ToUTF16("G0g0z"), false));

    EXPECT_EQ(false, StringEquals(        "g0G0",          "g0G0z",  true));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0"),         "g0G0z",  true));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0"), ToUTF16("g0G0z"), true));

    EXPECT_EQ(false, StringEquals(        "g0G0z",          "g0G0",  true));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0z"),         "g0G0",  true));
    EXPECT_EQ(false, StringEquals(ToUTF16("g0G0z"), ToUTF16("g0G0"), true));

    string a1 = "foo bar baz bat";
    EXPECT_EQ(true,  StringReplace(&a1, "bar baz", "raz"));
    EXPECT_EQ(false, StringReplace(&a1, "bar baz", "raz"));
    EXPECT_EQ(a1, "foo raz bat");
}

TEST(StringTest, Split) {
    string l, r, l1="aaaaa", r1="bbbb", x1="zzqrrrz", z1=", enzoop";
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
}

TEST(StringTest, basename) {
    int len;
    EXPECT_EQ(string(basename("/s/foo.bar",     0, &len)), "foo.bar"); EXPECT_EQ(3, len);
    EXPECT_EQ(string(basename("aa/bb/s/foobar", 0, &len)), "foobar");  EXPECT_EQ(6, len);

    EXPECT_EQ(0, dirnamelen("foo.bar"));
    EXPECT_EQ(2, dirnamelen("/s/foo.bar"));
    EXPECT_EQ(3, dirnamelen("/s/foo.bar", 0, 1));

    char foo[] = "aa/bb/s/foobar";
    EXPECT_EQ(7, dirnamelen(foo));
    EXPECT_EQ(8, dirnamelen(foo, 0, 1));
    EXPECT_EQ(string("aa/bb/s"),  string(foo, dirnamelen(foo)));
    EXPECT_EQ(string("aa/bb/s/"), string(foo, dirnamelen(foo, 0, 1)));

    string bar = "aa/bb/s/foobar", baz;
    EXPECT_EQ(7, dirnamelen(bar.c_str()));
    EXPECT_EQ(8, dirnamelen(bar.c_str(), 0, 1));
    EXPECT_EQ(string("aa/bb/s"),  string(bar.c_str(), dirnamelen(bar.c_str())));
    EXPECT_EQ(string("aa/bb/s/"), string(bar.c_str(), dirnamelen(bar.c_str(), 0, 1)));

    baz.assign(bar, 0, dirnamelen(bar.c_str(), 0, 1));
    EXPECT_EQ(string("aa/bb/s/"), baz);

    baz.assign(bar, 0, dirnamelen("baz", 0, 1));
    EXPECT_EQ(true, baz.empty());

    EXPECT_EQ(true,  basedir(   "wav/foo.bar", "wav"));
    EXPECT_EQ(false, basedir(   "wav/foo.bar", "Mav"));
    EXPECT_EQ(false, basedir(   "wav/foo.bar", "wa" ));
    EXPECT_EQ(false, basedir(   "wav/foo.bar",  "av"));
    EXPECT_EQ(true,  basedir(  "/wav/foo.bar", "wav"));
    EXPECT_EQ(false, basedir(  "/wav/foo.bar", "wa" ));
    EXPECT_EQ(false, basedir(  "/wav/foo.bar",  "av"));
    EXPECT_EQ(false, basedir(  "/wav/foo.bar", "Mav"));
    EXPECT_EQ(true,  basedir("/s/wav/foo.bar", "wav"));
    EXPECT_EQ(false, basedir("/s/wav/foo.bar", "Mav"));
    EXPECT_EQ(false, basedir("/s/wav/foo.bar", "wa" ));
    EXPECT_EQ(false, basedir("/s/wav/foo.bar",  "av"));
}

TEST(IterTest, LineIter) {
    string b = "1 2 3\n4 5 6";
    StringLineIter line(b.data(), b.size());
    EXPECT_EQ(0, strcmp(BlankNull(line.Next()), "1 2 3"));
    EXPECT_EQ(0, strcmp(BlankNull(line.Next()), "4 5 6"));
}

TEST(BufferFileTest, Read) {
    string b = "1 2 3\n4 5 6", z = "7 8 9\n9 8 7\n7 8 9";
    BufferFile bf(b.data(), b.size());
    EXPECT_EQ(b.size(), bf.Size());
    EXPECT_EQ(0, strcmp(BlankNull(bf.NextLine()), "1 2 3"));
    EXPECT_EQ(0, strcmp(BlankNull(bf.NextLine()), "4 5 6"));
    EXPECT_EQ(b, bf.Contents());
    bf.Write(z.data(), z.size());
    EXPECT_EQ(z.size(), bf.Size());
    EXPECT_EQ(z, bf.Contents());
}

TEST(LocalFileTest, Read) {
    {
        string fn = "../fv/assets/MenuAtlas1,0,0,0,0,000.png", contents = LocalFile::FileContents(fn), buf;
        INFO("Read ", fn, " ", contents.size(), " bytes");
        LocalFile f(fn, "r");
        for (const char *line = f.NextChunk(); line; line = f.NextChunk()) buf.append(line, f.nr.record_len);
        EXPECT_EQ(contents, buf);
    }
    {
        string fn = "../lfapp/lfapp.cpp", contents = LocalFile::FileContents(fn), buf;
        INFO("Read ", fn, " ", contents.size(), " bytes");
        if (contents.back() != '\n') contents.append("\n");
        LocalFile f(fn, "r");
        for (const char *line = f.NextLineRaw(); line; line = f.NextLineRaw())
            buf += string(line, f.nr.record_len) + "\n";
        EXPECT_EQ(contents, buf);
    }
}
