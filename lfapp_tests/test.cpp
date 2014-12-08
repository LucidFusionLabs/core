/*
 * $Id: lfapp.cpp 1309 2014-10-10 19:20:55Z justin $
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

using namespace LFL;

class MyEnvironment : public ::testing::Environment {
  public:
    virtual ~MyEnvironment() {}
    virtual void TearDown() {}
    virtual void SetUp() { 
        FLAGS_default_font = FakeFont::Filename();
        const char *av[] = { "testargv0", 0 };
        CHECK_EQ(app->Create(1, av, __FILE__), 0);
    }
};

::testing::Environment* const my_env = ::testing::AddGlobalTestEnvironment(new MyEnvironment);

