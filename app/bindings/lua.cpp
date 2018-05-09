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

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
};

namespace LFL {
struct MyLuaContext : public LuaContext {
  lua_State *L;
  ~MyLuaContext() { lua_close(L); }
  MyLuaContext() : L(luaL_newstate()) {
    luaopen_base(L);
    luaopen_table(L);
    luaopen_io(L);
    luaopen_string(L);
    luaopen_math(L);
  }

  string Execute(const string &s) {
    if (luaL_loadbuffer(L, s.data(), s.size(), "MyLuaExec")) { ERROR("luaL_loadstring ", lua_tostring(L, -1)); return ""; }
    if (lua_pcall(L, 0, LUA_MULTRET, 0))                     { ERROR("lua_pcall ",       lua_tostring(L, -1)); return ""; }
    return "";
  }
};

unique_ptr<LuaContext> LuaContext::Create() { return make_unique<MyLuaContext>(); }

}; // namespace LFL
