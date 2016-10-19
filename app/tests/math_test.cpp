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

namespace LFL {
TEST(MathTest, Util) {
  EXPECT_FLOAT_EQ(Decimals( 10.23456),  .23456);
  EXPECT_FLOAT_EQ(Decimals(-10.23456), -.23456);
  EXPECT_FALSE(IsPowerOfTwo(0));
  EXPECT_TRUE (IsPowerOfTwo(1));
  EXPECT_TRUE (IsPowerOfTwo(2));
  EXPECT_FALSE(IsPowerOfTwo(3));
  EXPECT_EQ(10,    WhichLog2(1024));
  EXPECT_EQ(2048,  NextPowerOfTwo(1026));
  { char16_t str[] = { ' ', ' ', '-', '3', '7', '4', 'a', 0 }; EXPECT_EQ(-374, atoi(str)); }
  EXPECT_EQ( 1,  RoundDown( 1.2));
  EXPECT_EQ( 2,  RoundUp  ( 1.2));
  EXPECT_EQ(-2,  RoundUp  (-1.2));
  EXPECT_EQ(-1,  RoundDown(-1.2));
  EXPECT_EQ(0,  PrevMultipleOfN(15, 16));
  EXPECT_EQ(16, PrevMultipleOfN(16, 16));
  EXPECT_EQ(16, NextMultipleOfN(16, 16));
  EXPECT_EQ(32, NextMultipleOfN(17, 16));
}

TEST(MatrixTest, ChangeDimensions) {
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
}

TEST(MatrixTest, Multipy) {
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

TEST(MatrixTest, Convolve) {
  Matrix A(10, 10, 1), B(3, 3, 1), C(A.M, A.N);
  EXPECT_EQ(&C, Matrix::Convolve(&A, &B, &C));
  MatrixIter(&C) {
    bool border_i = (i == 0 || i == C.M-1), border_j = (j == 0 || j == C.N-1);
    if      (border_i && border_j) EXPECT_EQ(4, C.row(i)[j]);
    else if (border_i || border_j) EXPECT_EQ(6, C.row(i)[j]);
    else                           EXPECT_EQ(9, C.row(i)[j]);
  }
}

TEST(MatrixTest, Invert) {
  const float e = 1e-6;
  {
    Matrix A(2, 2), Ainv(2, 2);
    { double *r = A.row(0); r[0] = 4; r[1] = 7; }
    { double *r = A.row(1); r[0] = 2; r[1] = 6; }

    Invert(&A, &Ainv);
    { double *r = Ainv.row(0); EXPECT_NEAR( 0.6, r[0], e); EXPECT_NEAR(-0.7, r[1], e); }
    { double *r = Ainv.row(1); EXPECT_NEAR(-0.2, r[0], e); EXPECT_NEAR( 0.4, r[1], e); }
  }
  {
    Matrix A(4, 4), Ainv(4, 4);
    { double *r = A.row(0); r[0] = 2; r[1] = 3; r[2] = 1; r[3] = 9; }
    { double *r = A.row(1); r[0] = 7; r[1] = 4; r[2] = 1; r[3] = 3; }
    { double *r = A.row(2); r[0] = 8; r[1] = 5; r[2] = 2; r[3] = 0; }
    { double *r = A.row(3); r[0] = 1; r[1] = 7; r[2] = 3; r[3] = 1; }

    float d, av[] = { 2, 3, 1, 9,
                      7, 4, 1, 3,
                      8, 5, 2, 0,
                      1, 7, 3, 1 };
    m44 a(av), ai;
    EXPECT_EQ(true, m44::Invert(a, &ai, &d));
    EXPECT_NEAR(222, d, e);
    { const v4 &r = ai[0];     EXPECT_NEAR( 6  /d, r[0], e); EXPECT_NEAR(-10 /d, r[1], e); EXPECT_NEAR( 38 /d, r[2], e); EXPECT_NEAR(-24/d, r[3], e); }
    { const v4 &r = ai[1];     EXPECT_NEAR(-72 /d, r[0], e); EXPECT_NEAR( 194/d, r[1], e); EXPECT_NEAR(-160/d, r[2], e); EXPECT_NEAR( 66/d, r[3], e); }
    { const v4 &r = ai[2];     EXPECT_NEAR( 156/d, r[0], e); EXPECT_NEAR(-445/d, r[1], e); EXPECT_NEAR( 359/d, r[2], e); EXPECT_NEAR(-69/d, r[3], e); }
    { const v4 &r = ai[3];     EXPECT_NEAR( 30 /d, r[0], e); EXPECT_NEAR(-13 /d, r[1], e); EXPECT_NEAR( 5  /d, r[2], e); EXPECT_NEAR(-9 /d, r[3], e); }

    Invert(&A, &Ainv);
    { double *r = Ainv.row(0); EXPECT_NEAR( 6  /d, r[0], e); EXPECT_NEAR(-10 /d, r[1], e); EXPECT_NEAR( 38 /d, r[2], e); EXPECT_NEAR(-24/d, r[3], e); }
    { double *r = Ainv.row(1); EXPECT_NEAR(-72 /d, r[0], e); EXPECT_NEAR( 194/d, r[1], e); EXPECT_NEAR(-160/d, r[2], e); EXPECT_NEAR( 66/d, r[3], e); }
    { double *r = Ainv.row(2); EXPECT_NEAR( 156/d, r[0], e); EXPECT_NEAR(-445/d, r[1], e); EXPECT_NEAR( 359/d, r[2], e); EXPECT_NEAR(-69/d, r[3], e); }
    { double *r = Ainv.row(3); EXPECT_NEAR( 30 /d, r[0], e); EXPECT_NEAR(-13 /d, r[1], e); EXPECT_NEAR( 5  /d, r[2], e); EXPECT_NEAR(-9 /d, r[3], e); }
  }
}

TEST(MatrixTest, Ortho) {
  int w=640, h=480;
  m44 ortho = m44::Ortho(0, w, 0, h, 0, 100);
  v4 a(0,0,0,1), b(w, h, 0, 1), aa=ortho.Transform(a), bb=ortho.Transform(b);
  EXPECT_EQ(v4(-1,-1,-1,1), aa);
  EXPECT_EQ(v4( 1, 1,-1,1), bb);
}

}; // namespace LFL
