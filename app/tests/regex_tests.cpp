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
#include "core/app/types/trie.h"

namespace LFL {
DEFINE_int(size, 1024*1024, "Test size"); 

extern "C" LFApp *MyAppCreate(int argc, const char* const* argv) {
  FLAGS_font = FakeFontEngine::Filename();
  auto app = make_unique<Application>(argc, argv).release();
  app->focused = app->framework->ConstructWindow(app).release();
  testing::InitGoogleTest(&argc, const_cast<char**>(argv));
  return app;
}

extern "C" int MyAppMain(LFApp *a) {
  auto app = static_cast<Application*>(a);
  CHECK_EQ(0, app->Create(__FILE__));
  return RUN_ALL_TESTS();
}

struct MyEnvironment : public ::testing::Environment {
  string prefix1="yeah fun ", prot1="http://", url1="url", suffix1=" and whatever ";
  string test1 = prefix1 + prot1 + url1 + suffix1;
  virtual ~MyEnvironment() {}
  virtual void TearDown() { INFO(Singleton<PerformanceTimers>::Get()->DebugString()); }
  virtual void SetUp() {}
};

MyEnvironment* const my_env = dynamic_cast<MyEnvironment*>(::testing::AddGlobalTestEnvironment(new MyEnvironment));

TEST(RegexTest, StrstrURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Set();
  int tid = timers->Create("StrstrURL");
  timers->AccumulateTo(tid);
  for (int i=0; i<FLAGS_size; ++i) {
    const char *match = strstr(my_env->test1.c_str(), "http://");
    EXPECT_NE(nullptr, match);
    EXPECT_EQ(0, strncmp(match, "http://", 7));
    match = strstr(my_env->test1.c_str(), "https://");
    EXPECT_EQ(nullptr, match);
  }
  timers->AccumulateTo(0);
}

TEST(RegexTest, RegexpURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Set();
  int tid = timers->Create("RegexpURL");

  Regex::Result match;
  Regex url_matcher("(https?://)");
  timers->AccumulateTo(tid);
  for (int i=0; i<FLAGS_size; ++i) {
    match = url_matcher.MatchOne(my_env->test1);
    EXPECT_EQ(true, !!match); if (!!match) EXPECT_EQ("http://", match.Text(my_env->test1));
  }
  timers->AccumulateTo(0);
}

#ifdef LFL_SREGEX
TEST(RegexTest, StreamRegexURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
  int tid = timers->Create("StreamRegexURL");

  vector<Regex::Result> matches;
  StreamRegex url_matcher("https?://");
  timers->AccumulateTo(tid);
  for (int i=0; i<FLAGS_size; ++i) {
    url_matcher.Match(my_env->test1, &matches);
    EXPECT_EQ(1, matches.size()); if (matches.size()) EXPECT_EQ("http://", matches[0].Text(my_env->test1));
    matches.clear();
  }
  timers->AccumulateTo(0);
}
#endif

TEST(RegexTest, AhoCorasickURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Set();
  int tid = timers->Create("AhoCorasickURL");

  vector<Regex::Result> matches;
  AhoCorasickFSM<char> url_matcher({ "http://", "https://" });
  timers->AccumulateTo(tid);
  for (int i=0; i<FLAGS_size; ++i) {
    url_matcher.Match(my_env->test1, &matches);
    EXPECT_EQ(1, matches.size()); if (matches.size()) EXPECT_EQ("http://", matches[0].Text(my_env->test1));
    matches.clear();
  }
  timers->AccumulateTo(0);
}

TEST(RegexTest, AhoCorasickMatcherURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Set();
  int tid = timers->Create("AhoCorasickMatcherURL");

  AhoCorasickFSM<char> url_fsm({ "http://", "https://" });
  StringMatcher<char> matcher(&url_fsm);
  timers->AccumulateTo(tid);
  string r1 = my_env->prefix1 + my_env->prot1;
  for (int i=0; i<FLAGS_size; ++i) {
    StringMatcher<char>::iterator chunk = matcher.Begin(my_env->test1);
    EXPECT_NE(chunk.b, chunk.e);
    // EXPECT_EQ(r1, string(chunk.b, chunk.nb));
    ++chunk;
    EXPECT_NE(chunk.b, chunk.e);
    EXPECT_EQ(my_env->url1, string(chunk.b, chunk.nb));
    ++chunk;
    EXPECT_NE(chunk.b, chunk.e);
    // EXPECT_EQ(my_env->suffix1, string(chunk.b, chunk.nb));
    ++chunk;
    EXPECT_EQ(chunk.e, chunk.b);
  }
  timers->AccumulateTo(0);
}

}; // namespace LFL
