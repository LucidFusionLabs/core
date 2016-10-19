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
#include "core/app/ipc.h"

namespace LFL {
const int MultiProcessPaintResource::DrawBox::Type;
const int MultiProcessPaintResource::DrawBackground::Type;

TEST(WireTest, MultiProcessPaintResource) {
  MultiProcessPaintResourceBuilder builder;
  builder.Add(MultiProcessPaintResource::DrawBox(       Box(3, 9,  11, 33), 17));
  builder.Add(MultiProcessPaintResource::DrawBackground(Box(4, 10, 12, 34)));

  MultiProcessPaintResource::Iterator iter(builder.data.buf);
  /**/         EXPECT_EQ(MultiProcessPaintResource::DrawBox::Type,           iter.type);
  /**/         EXPECT_EQ(iter.Get<MultiProcessPaintResource::DrawBox>()->b,  Box(3, 9, 11, 33));
  /**/         EXPECT_EQ(iter.Get<MultiProcessPaintResource::DrawBox>()->id, 17);

  iter.Next(); EXPECT_EQ(MultiProcessPaintResource::DrawBackground::Type,    iter.type);
  /**/         EXPECT_EQ(iter.Get<MultiProcessPaintResource::DrawBox>()->b,  Box(4, 10, 12, 34));

  iter.Next(); EXPECT_EQ(0,                                                  iter.type);
}

}; // namespace LFL
