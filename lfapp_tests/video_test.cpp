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

TEST(VideoTest, Color) {
    float h, s, v;
    Color r(1.0, 0.0, 0.0, 1.0);
    r.ToHSV(&h, &s, &v);
    EXPECT_EQ(255,                     r.R());
    EXPECT_EQ(Color::red,              r);
    EXPECT_EQ(Color(r.HexString()),    r);
    EXPECT_EQ(Color::FromHSV(h, s, v), r);

    Color g(0.0, 1.0, 0.0, 1.0);
    g.ToHSV(&h, &s, &v);
    EXPECT_EQ(255,                     g.G());
    EXPECT_EQ(Color::green,            g);
    EXPECT_EQ(Color(g.HexString()),    g);
    EXPECT_EQ(Color::FromHSV(h, s, v), g);

    Color b(0.0, 0.0, 1.0, 1.0);
    b.ToHSV(&h, &s, &v);
    EXPECT_EQ(255,                     b.B());
    EXPECT_EQ(Color::blue,             b);
    EXPECT_EQ(Color(b.HexString()),    b);
    EXPECT_EQ(Color::FromHSV(h, s, v), b);

    Color w(1.0, 1.0, 1.0);
    w.ToHSV(&h, &s, &v);
    EXPECT_EQ(Color::white,            w);
    EXPECT_EQ(Color(w.HexString()),    w);
    EXPECT_EQ(Color::FromHSV(h, s, v), w);

    Color c1(93, 125, 58);
    c1.ToHSV(&h, &s, &v);
    EXPECT_EQ(Color(c1.HexString()),   c1);
    EXPECT_EQ(Color::FromHSV(h, s, v), c1);

    Color  c2("FF7400");
    EXPECT_EQ("FF7400", c2.HexString());
    c2.ToHSV(&h, &s, &v);
    EXPECT_EQ("FF7400", Color::FromHSV(h, s, v).HexString());
    h += 180;
    EXPECT_EQ("008BFF", Color::FromHSV(h, s, v).HexString());
}

TEST(VideoTest, Box) {
    Box box = screen->Box();
    EXPECT_EQ(0, box.x);
    EXPECT_EQ(0, box.y);
    EXPECT_TRUE(box.w != 0);
    EXPECT_TRUE(box.h != 0);
    EXPECT_EQ(box, screen->Box(1.0, 1.0));
    EXPECT_EQ(box, screen->Box().center(screen->Box(1.0, 1.0)));

    box = Box(640, 480);               EXPECT_EQ(Box(0,  0,  640, 480), box);
    box = Box::DelBorder(box, 40, 80); EXPECT_EQ(Box(20, 40, 600, 400), box);
    box = Box::AddBorder(box, 40, 80); EXPECT_EQ(Box(0,  0,  640, 480), box);
}

}; // namespace LFL
