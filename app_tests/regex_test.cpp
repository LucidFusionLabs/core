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
#include "core/app/app.h"
#include "core/app/types/trie.h"

extern "C" void MyAppCreate() {
  LFL::FLAGS_default_font = LFL::FakeFontEngine::Filename();
  LFL::app = new LFL::Application();
  LFL::screen = new LFL::Window();
}

extern "C" int MyAppMain(int argc, const char* const* argv) {
  testing::InitGoogleTest(&argc, const_cast<char**>(argv));
  if (!LFL::app) MyAppCreate();
  CHECK_EQ(0, LFL::app->Create(argc, argv, __FILE__));
  return RUN_ALL_TESTS();
}

namespace LFL {
DEFINE_int(size, 1024*1024, "Test size"); 

struct MyEnvironment : public ::testing::Environment {
  string prefix1="yeah fun ", prot1="http://", url1="url", suffix1=" and whatever ";
  string test1 = prefix1 + prot1 + url1 + suffix1;
  virtual ~MyEnvironment() {}
  virtual void TearDown() { INFO(Singleton<PerformanceTimers>::Get()->DebugString()); }
  virtual void SetUp() {}
};

MyEnvironment* const my_env = dynamic_cast<MyEnvironment*>(::testing::AddGlobalTestEnvironment(new MyEnvironment));

TEST(RegexTest, StrstrURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
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

TEST(RegexTest, StreamRegexURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
  int tid = timers->Create("StreamRegexURL");

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

TEST(RegexTest, AhoCorasickURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
  int tid = timers->Create("AhoCorasickURL");

  vector<Regex::Result> matches;
  AhoCorasickFSM<char> url_matcher({ "http://", "https://" });
  timers->AccumulateTo(tid);
  for (int i=0; i<FLAGS_size; ++i) {
    url_matcher.Match(my_env->test1, &matches);
    EXPECT_EQ(1, matches.size()); EXPECT_EQ("http://", matches[0].Text(my_env->test1));
    matches.clear();
  }
  timers->AccumulateTo(0);
}

TEST(RegexTest, AhoCorasickMatcherURL) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
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
