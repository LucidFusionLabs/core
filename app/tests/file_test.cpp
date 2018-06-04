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

TEST(FileTest, BufferFile) {
  string b = "1 2 3\n4 5 6", z = "7 8 9\n9 8 7\n7 8 9";
  BufferFile bf(b);
  NextRecordReader nr(&bf);
  EXPECT_EQ(b.size(), bf.Size());
  EXPECT_EQ(0, strcmp(BlankNull(nr.NextLine()), "1 2 3"));
  EXPECT_EQ(0, strcmp(BlankNull(nr.NextLine()), "4 5 6"));
  EXPECT_EQ(b, bf.Contents());
  bf.Write(z.data(), z.size());
  EXPECT_EQ(z.size(), bf.Size());
  EXPECT_EQ(z, bf.Contents());
}

TEST(FileTest, LocalFileRead) {
  {
    string fn = "../../../../core/app/assets/MenuAtlas,0,255,255,255,0.0000.png", contents = LocalFile(fn, "r").Contents(), buf;
    INFO("Read ", fn, " ", contents.size(), " bytes");
    LocalFile f(fn, "r");
    NextRecordReader nr(&f);
    for (const char *line = nr.NextChunk(); line; line = nr.NextChunk()) buf.append(line, nr.record_len);
    EXPECT_EQ(contents, buf);
  }
  {
    string fn = "../../../../core/app/app.cpp", contents = LocalFile(fn, "r").Contents(), buf;
    INFO("Read ", fn, " ", contents.size(), " bytes");
    if (contents.back() != '\n') contents.append("\n");
    LocalFile f(fn, "r");
    NextRecordReader nr(&f);
    for (const char *line = nr.NextLineRaw(); line; line = nr.NextLineRaw())
      buf += string(line, nr.record_len) + "\n";
    EXPECT_EQ(contents, buf);

    buf.clear();
    StringLineIter lines(contents, StringLineIter::Flag::BlankLines | StringLineIter::Flag::Raw);
    for (const char *line = lines.Next(); line; line = lines.Next())
      buf += string(line, lines.CurrentLength()) + "\n";
    EXPECT_EQ(contents, buf);
  }
}

}; // namespace LFL
