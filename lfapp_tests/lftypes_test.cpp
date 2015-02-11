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
TEST(UtilTest, Move) {
    //               01234567890123456789012345
    { char text[] = "abcdefghijklmnopqrstuvwxyz";
        //               |  |       |
        Move<char>(&text[4], &text[7], 9);
        EXPECT_EQ(0, strcmp(text, "abcdhijklmnopnopqrstuvwxyz"));
    }
    { char text[] = "abcdefghijklmnopqrstuvwxyz";
        //               |  |       |
        Move<char>(&text[7], &text[4], 9);
        EXPECT_EQ(0, strcmp(text, "abcdefgefghijklmqrstuvwxyz"));
    }
}

TEST(ArrayTest, Segment) {
    {
        struct ZZSeg1 { int x, y; ZZSeg1(int X=0, int Y=0) : x(X), y(Y) {} };
        vector<ZZSeg1> vec(100);
        for (int i=0; i<100; i++) {
            if      (i < 33) { vec[i].x = i+1; vec[i].y = 3; }
            else if (i < 67) { vec[i].x = i-1; vec[i].y = 9; }
            else             { vec[i].x = i;   vec[i].y = 7; }
        }
        int count = 0;
        for (ArrayMemberSegmentIter<ZZSeg1, int, &ZZSeg1::y> iter(vec); !iter.Done(); iter.Increment(), count++) {
            if      (count == 0) { EXPECT_EQ(0, iter.i); EXPECT_EQ(    33, iter.Length()); EXPECT_EQ(&vec[0],  iter.Data()); }
            else if (count == 1) { EXPECT_EQ(1, iter.i); EXPECT_EQ( 67-33, iter.Length()); EXPECT_EQ(&vec[33], iter.Data()); }
            else if (count == 2) { EXPECT_EQ(2, iter.i); EXPECT_EQ(100-67, iter.Length()); EXPECT_EQ(&vec[67], iter.Data()); }
            else                 { EXPECT_EQ(0, 1); }
        }
        EXPECT_EQ(count, 3);
    }
    {
        vector<int> vec(100);
        for (int i=0; i<100; i++) {
            if      (i < 33) vec[i] = 3;
            else if (i < 67) vec[i] = 9;
            else             vec[i] = 7;
        }
        int count = 0;
        for (ArraySegmentIter<int> iter(vec); !iter.Done(); iter.Increment(), count++) {
            if      (count == 0) { EXPECT_EQ(0, iter.i); EXPECT_EQ(    33, iter.Length()); EXPECT_EQ(&vec[0],  iter.Data()); }
            else if (count == 1) { EXPECT_EQ(1, iter.i); EXPECT_EQ( 67-33, iter.Length()); EXPECT_EQ(&vec[33], iter.Data()); }
            else if (count == 2) { EXPECT_EQ(2, iter.i); EXPECT_EQ(100-67, iter.Length()); EXPECT_EQ(&vec[67], iter.Data()); }
            else                 { EXPECT_EQ(0, 1); }
        }
        EXPECT_EQ(3, count);
    }
}

TEST(ArrayTest, FlattenedArrayVals) {
    typedef pair<int, int> AT_FAV_Iter;
    vector<int> v = { 1, 3, 1, 2, 1, 3, 2 };
    FlattenedArrayValues<vector<int>> at_fav(&v, v.size(), (int(*)(vector<int>*,int))&IndexOrDefault<int>);
    AT_FAV_Iter i1, i2, last = at_fav.LastIter();
    EXPECT_EQ(AT_FAV_Iter(0, 0), i1);
    EXPECT_EQ(AT_FAV_Iter(v.size()-1, v.back()-1), last);

    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(1, 0), i1); EXPECT_EQ(1, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(1, 1), i1); EXPECT_EQ(1, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(1, 2), i1); EXPECT_EQ(1, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(2, 0), i1); EXPECT_EQ(1, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 2); EXPECT_EQ(AT_FAV_Iter(3, 1), i1); EXPECT_EQ(2, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 3); EXPECT_EQ(AT_FAV_Iter(5, 1), i1); EXPECT_EQ(3, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(5, 2), i1); EXPECT_EQ(1, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(6, 0), i1); EXPECT_EQ(1, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(6, 1), i1); EXPECT_EQ(1, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 1); EXPECT_EQ(AT_FAV_Iter(6, 1), i1); EXPECT_EQ(0, at_fav.Distance(i1, i2)); i2=i1;
    at_fav.AdvanceIter(&i1, 9); EXPECT_EQ(AT_FAV_Iter(6, 1), i1); EXPECT_EQ(0, at_fav.Distance(i1, i2)); i2=i1;

    int sum = 0; for (auto i : v) sum += i;
    i1 = AT_FAV_Iter();
    i2 = last;
    for (int i=1; i<=sum; i++) {
        EXPECT_EQ(sum-i, at_fav.Distance(i1, i2));
        EXPECT_EQ(sum-i, at_fav.Distance(i2, i1));
        at_fav.AdvanceIter(&i1, 1);
    }
}

TEST(BitTest, Bit) {
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

TEST(BitTest, BitField) {
    { unsigned char bitfield[] = { 0,   0,   0,   0,   0,   0   }; EXPECT_EQ(-1, BitField::FirstSet(bitfield, sizeof(bitfield))); }
    { unsigned char bitfield[] = { 0,   0,   0,   0,   0,   1   }; EXPECT_EQ(40, BitField::FirstSet(bitfield, sizeof(bitfield))); }
    { unsigned char bitfield[] = { 0,   0,   0,   0,   0,   2   }; EXPECT_EQ(41, BitField::FirstSet(bitfield, sizeof(bitfield))); }

    { unsigned char bitfield[] = { 255, 255, 255, 255, 255, 255 }; EXPECT_EQ(-1, BitField::LastClear (bitfield, sizeof(bitfield))); }
    { unsigned char bitfield[] = { 255, 255, 255, 255, 255, 255 }; EXPECT_EQ(-1, BitField::FirstClear(bitfield, sizeof(bitfield))); }
    { unsigned char bitfield[] = { 255, 255, 255, 255, 255, 254 }; EXPECT_EQ(40, BitField::FirstClear(bitfield, sizeof(bitfield))); }
    { unsigned char bitfield[] = { 255, 255, 255, 255, 255, 253 }; EXPECT_EQ(41, BitField::FirstClear(bitfield, sizeof(bitfield))); }

    {
        unsigned char bitfield[] = { 255, 255, 255, 255, 255, 0 };
        EXPECT_EQ(40, BitField::FirstClear(bitfield, sizeof(bitfield)));
        BitField::Set(bitfield, 40);
        EXPECT_EQ(41, BitField::FirstClear(bitfield, sizeof(bitfield)));
        BitField::Set(bitfield, 41);
        EXPECT_EQ(42, BitField::FirstClear(bitfield, sizeof(bitfield)));
        BitField::Clear(bitfield, 41);
        EXPECT_EQ(41, BitField::FirstClear(bitfield, sizeof(bitfield)));
        BitField::Clear(bitfield, 11);
        EXPECT_EQ(11, BitField::FirstClear(bitfield, sizeof(bitfield)));
        BitField::Clear(bitfield, 7);
        EXPECT_EQ(7,  BitField::FirstClear(bitfield, sizeof(bitfield)));
        EXPECT_EQ(127, bitfield[0]);
    }
}
}; // namespace LFL
