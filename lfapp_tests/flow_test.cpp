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
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "lfapp/dom.h"
#include "lfapp/flow.h"

namespace LFL {

TEST(FlowTest, FloatContainer) {
  FloatContainer container;
  LFL::DOM::Text A(0), B(0), C(0), D(0);
  Box ab, bb, cb, db;

  // width=0 wrap each left
  container.Clear();
  container.AddFloat(-20, 20, 20, 0, &A, &ab);
  EXPECT_EQ(Box(0, -20, 20, 20), ab);
  container.AddFloat(-20, 20, 20, 0, &B, &bb);
  EXPECT_EQ(Box(0, -40, 20, 20), bb);
  EXPECT_EQ(40, container.FloatHeight());

  // 3 in a horizontal line, then wrap 1 left
  container.Clear();
  container.w = 70;
  container.AddFloat(-20, 20, 20, 0, &A, &ab);
  EXPECT_EQ(Box(0, -20, 20, 20), ab);
  container.AddFloat(-20, 20, 20, 0, &B, &bb);
  EXPECT_EQ(Box(20, -20, 20, 20), bb);
  container.AddFloat(-20, 20, 20, 0, &C, &cb);
  EXPECT_EQ(Box(40, -20, 20, 20), cb);
  EXPECT_EQ(10, container.CenterFloatWidth(-20, 20));
  container.AddFloat(-20, 20, 20, 0, &D, &db);
  EXPECT_EQ(Box(0, -40, 20, 20), db);
  EXPECT_EQ(40, container.FloatHeight());

  // 3 in a horizontal line, then wrap 1 right
  container.Clear();
  container.w = 70;
  container.AddFloat(-20, 20, 20, 1, &A, &ab);
  EXPECT_EQ(Box(50, -20, 20, 20), ab);
  container.AddFloat(-20, 20, 20, 1, &B, &bb);
  EXPECT_EQ(Box(30, -20, 20, 20), bb);
  container.AddFloat(-20, 20, 20, 1, &C, &cb);
  EXPECT_EQ(Box(10, -20, 20, 20), cb);
  EXPECT_EQ(10, container.CenterFloatWidth(-20, 20));
  container.AddFloat(-20, 20, 20, 1, &D, &db);
  EXPECT_EQ(Box(50, -40, 20, 20), db);
  EXPECT_EQ(40, container.FloatHeight());

  // 1 right, 1 left, 1 right, wrap 1 left
  container.Clear();
  container.w = 70;
  container.AddFloat(-20, 20, 20, 1, &A, &ab);
  EXPECT_EQ(Box(50, -20, 20, 20), ab);
  container.AddFloat(-20, 20, 20, 0, &B, &bb);
  EXPECT_EQ(Box(0, -20, 20, 20), bb);
  container.AddFloat(-20, 20, 20, 1, &C, &cb);
  EXPECT_EQ(Box(30, -20, 20, 20), cb);
  EXPECT_EQ(10, container.CenterFloatWidth(-20, 20));
  container.AddFloat(-20, 20, 20, 0, &D, &db);
  EXPECT_EQ(Box(0, -40, 20, 20), db);
  EXPECT_EQ(40, container.FloatHeight());
  EXPECT_EQ(59, container.ClearFloats( -1, 20, 1, 1));
  EXPECT_EQ(40, container.ClearFloats(-20, 20, 1, 1));

  FloatContainer parent;
  parent.SetDimension(point(200, 200));
  container.SetPosition(point(7, -47));
  container.h = 40;
  EXPECT_EQ(4, container.AddFloatsToParent(&parent));
  EXPECT_EQ(container.float_right.size(), parent.float_right.size());
  EXPECT_EQ(container.float_left .size(), parent.float_left .size());
  EXPECT_EQ(Box(57, -27, 20, 20), parent.float_right[0]);
  EXPECT_EQ(Box(37, -27, 20, 20), parent.float_right[1]);
  EXPECT_EQ(Box( 7, -27, 20, 20), parent.float_left [0]);
  EXPECT_EQ(Box( 7, -47, 20, 20), parent.float_left [1]);

  FloatContainer child;
  child.SetDimension(container.Dimension());
  child.SetPosition(container.Position());
  EXPECT_EQ(4, child.InheritFloats(&parent));
  EXPECT_EQ(parent.float_right.size(), child.float_right.size());
  EXPECT_EQ(parent.float_left .size(), child.float_left .size());
  EXPECT_EQ(container.float_right[0], child.float_right[0]);
  EXPECT_EQ(container.float_right[1], child.float_right[1]);
  EXPECT_EQ(container.float_left [0], child.float_left [0]);
  EXPECT_EQ(container.float_left [1], child.float_left [1]);
}

TEST(FlowTest, Layout) {
  Box dim(128, 128), box;
  Flow flow(&dim);
  int bw=20, bh=20;
  for (int i=0; i<10; i++) {
    flow.SetMinimumAscent(bh);
    flow.AppendBox(bw, bh, &box);
    EXPECT_EQ(bw, box.w);
    EXPECT_EQ(bh, box.h);
    if (i < 6) {
      EXPECT_EQ(bw*i, box.x);
      EXPECT_EQ(-bh,  box.y);
    } else {
      EXPECT_EQ(bw*(i-6), box.x);
      EXPECT_EQ(-bh*2,    box.y);
    }
  }
}

}; // namespace LFL
