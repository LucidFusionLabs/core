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
Application *app = nullptr;

extern "C" LFApp *MyAppCreate(int argc, const char* const* argv) {
  FLAGS_enable_video = true;
  FLAGS_font = FakeFontEngine::Filename();
  app = CreateApplication(argc, argv).release();
  app->focused = CreateWindow(app).release();
  testing::InitGoogleTest(&argc, const_cast<char**>(argv));
  return app;
}

extern "C" int MyAppMain() {
  CHECK_EQ(0, app->Create(__FILE__));
  CHECK_EQ(0, app->Init());
  return RUN_ALL_TESTS();
}

class MyEnvironment : public ::testing::Environment {
  public:
    virtual ~MyEnvironment() {}
    virtual void TearDown() {}
    virtual void SetUp() {}
};

::testing::Environment* const my_env = ::testing::AddGlobalTestEnvironment(new MyEnvironment);
}; // namespace LFL
