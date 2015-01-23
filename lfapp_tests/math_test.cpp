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

namespace LFL {
TEST(MathTest, util) {
    EXPECT_EQ(true,  Equal(Decimals( 10.23456),  .23456));
    EXPECT_EQ(true,  Equal(Decimals(-10.23456), -.23456));
    EXPECT_EQ(false, IsPowerOfTwo(0));
    EXPECT_EQ(true,  IsPowerOfTwo(1));
    EXPECT_EQ(true,  IsPowerOfTwo(2));
    EXPECT_EQ(false, IsPowerOfTwo(3));
    { short str[] = { ' ', ' ', '-', '3', '7', '4', 'a', 0 }; EXPECT_EQ(-374, atoi(str)); }
}

TEST(MathTest, matrix) {
    const int init_M=3, init_N=4;
    Matrix m(init_M, init_N, 2.1);
    EXPECT_EQ(init_M, m.M);
    EXPECT_EQ(init_N, m.N);
    int count = 0, max_I = -1, max_J = -1;
    MatrixIter(&m) {
        max_I = max(max_I, i);
        max_J = max(max_J, j);
        EXPECT_EQ(2.1, m.row(i)[j]);
        count++;
    }
    EXPECT_EQ(init_M * init_N, count);
    EXPECT_EQ(init_M-1, max_I);
    EXPECT_EQ(init_N-1, max_J);

    m.AddRows(1);
    EXPECT_EQ(init_M+1, m.M);
    count = 0; MatrixIter(&m) {
        if (i == init_M) EXPECT_EQ(0,   m.row(i)[j]);
        else             EXPECT_EQ(2.1, m.row(i)[j]);
        count++;
    }
    EXPECT_EQ((init_M+1) * init_N, count);

    m.AddCols(1);
    EXPECT_EQ(init_N+1, m.N);
    count = 0; MatrixIter(&m) {
        if (i == init_M || j == init_N) EXPECT_EQ(0,   m.row(i)[j]);
        else                            EXPECT_EQ(2.1, m.row(i)[j]);
        count++;
    }
    EXPECT_EQ((init_M+1) * (init_N+1), count);

    m.AddCols(1, true);
    EXPECT_EQ(init_N+2, m.N);
    count = 0; MatrixIter(&m) {
        if (i == init_M || j == init_N+1 || !j) EXPECT_EQ(0,   m.row(i)[j]);
        else                                    EXPECT_EQ(2.1, m.row(i)[j]);
        count++;
    }
    EXPECT_EQ((init_M+1) * (init_N+2), count);

    m.AddRows(1, true);
    EXPECT_EQ(init_M+2, m.M);
    count = 0; MatrixIter(&m) {
        if (i == init_M+1 || j == init_N+1 || !j || !i) EXPECT_EQ(0,   m.row(i)[j]);
        else                                            EXPECT_EQ(2.1, m.row(i)[j]);
        count++;
    }
    EXPECT_EQ((init_M+2) * (init_N+2), count);

    { // mult
        Matrix A(2, 3), B(3, 2), C(2, 2);
        { double *r = A.row(0); r[0] = 0; r[1] = -1; r[2] = 2; }
        { double *r = A.row(1); r[0] = 4; r[1] = 11; r[2] = 2; }

        { double *r = B.row(0); r[0] = 3; r[1] = -1; }
        { double *r = B.row(1); r[0] = 1; r[1] =  2; }
        { double *r = B.row(2); r[0] = 6; r[1] =  1; }

        EXPECT_EQ(&C, Matrix::Mult(&A, &B, &C));
        { double *r = C.row(0); EXPECT_EQ(11, r[0]); EXPECT_EQ( 0, r[1]); }
        { double *r = C.row(1); EXPECT_EQ(35, r[0]); EXPECT_EQ(20, r[1]); }
    }

    { // convolve
        Matrix A(10, 10, 1), B(3, 3, 1), C(A.M, A.N);
        EXPECT_EQ(&C, Matrix::Convolve(&A, &B, &C));
        MatrixIter(&C) {
            bool border_i = (i == 0 || i == C.M-1), border_j = (j == 0 || j == C.N-1);
            if      (border_i && border_j) EXPECT_EQ(4, C.row(i)[j]);
            else if (border_i || border_j) EXPECT_EQ(6, C.row(i)[j]);
            else                           EXPECT_EQ(9, C.row(i)[j]);
        }
    }

    { // invert
        Matrix A(2, 2), Ainv(2, 2);
        { double *r = A.row(0); r[0] = 4; r[1] = 7; }
        { double *r = A.row(1); r[0] = 2; r[1] = 6; }

        Invert(&A, &Ainv);
        { double *r = Ainv.row(0); EXPECT_EQ(1, Equal( 0.6, r[0])); EXPECT_EQ(1, Equal(-0.7, r[1])); }
        { double *r = Ainv.row(1); EXPECT_EQ(1, Equal(-0.2, r[0])); EXPECT_EQ(1, Equal( 0.4, r[1])); }
    }
}
}; // namespace LFL
