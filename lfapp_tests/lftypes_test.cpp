/*
 * $Id: lfapp.cpp 1309 2014-10-10 19:20:55Z justin $
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

using namespace LFL;

TEST(util, MoveTest) {
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
