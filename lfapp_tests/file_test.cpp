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

TEST(FileTest, BufferFile) {
    string b = "1 2 3\n4 5 6", z = "7 8 9\n9 8 7\n7 8 9";
    BufferFile bf(b);
    EXPECT_EQ(b.size(), bf.Size());
    EXPECT_EQ(0, strcmp(BlankNull(bf.NextLine()), "1 2 3"));
    EXPECT_EQ(0, strcmp(BlankNull(bf.NextLine()), "4 5 6"));
    EXPECT_EQ(b, bf.Contents());
    bf.Write(z.data(), z.size());
    EXPECT_EQ(z.size(), bf.Size());
    EXPECT_EQ(z, bf.Contents());
}

TEST(FileTest, LocalFileRead) {
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

}; // namespace LFL
