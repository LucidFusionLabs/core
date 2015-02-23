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

GTEST_API_ int main(int argc, const char **argv) {
    testing::InitGoogleTest(&argc, (char**)argv);
    LFL::FLAGS_default_font = LFL::FakeFont::Filename();
    CHECK_EQ(LFL::app->Create(argc, argv, __FILE__), 0);
    return RUN_ALL_TESTS();
}

namespace LFL {
DEFINE_int(size, 1024*1024, "Test size"); 

struct MyEnvironment : public ::testing::Environment {
    string test1 = "yeah fun http://url and whatever ";
    virtual ~MyEnvironment() {}
    virtual void TearDown() { INFO(Singleton<PerformanceTimers>::Get()->DebugString()); }
    virtual void SetUp() {}
};

MyEnvironment* const my_env = (MyEnvironment*)::testing::AddGlobalTestEnvironment(new MyEnvironment);

TEST(RegexTest, StrstrURL) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    int tid = timers->Create("StrstrURL");
    timers->AccumulateTo(tid);
    for (int i=0; i<FLAGS_size; ++i) {
        const char *match = strstr(my_env->test1.c_str(), "http://");
        EXPECT_NE(nullptr, match);
        EXPECT_EQ(0, strncmp(match, "http://", 7));
    }
    timers->AccumulateTo(0);
}

TEST(RegexTest, RegexpURL) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    int tid = timers->Create("RegexpURL");

    vector<Regex::Result> matches;
    Regex url_matcher("https?://");
    timers->AccumulateTo(tid);
    for (int i=0; i<FLAGS_size; ++i) {
        url_matcher.Match(my_env->test1, &matches);
        EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(my_env->test1));
        matches.clear();
    }
    timers->AccumulateTo(0);
}

TEST(RegexTest, RegexStreamURL) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    int tid = timers->Create("StreamURL");

    vector<Regex::Result> matches;
    StreamRegex url_matcher("https?://");
    timers->AccumulateTo(tid);
    for (int i=0; i<FLAGS_size; ++i) {
        url_matcher.Match(my_env->test1, &matches);
        EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(my_env->test1));
        matches.clear();
    }
    timers->AccumulateTo(0);
}

}; // namespace LFL
