# $Id$

lfl_add_target(app_null_js STATIC_LIBRARY SOURCES null_js.cpp)

if(LFL_LUA)
  lfl_add_target(app_lua STATIC_LIBRARY SOURCES lua.cpp
                 INCLUDE_DIRECTORIES ${LUA_INCLUDE} LINK_LIBRARIES ${LUA_LIB})
endif()

if(LFL_V8JS)
  lfl_add_target(app_v8_js STATIC_LIBRARY SOURCES v8_js.cpp
                 INCLUDE_DIRECTORIES ${V8JS_INCLUDE} LINK_LIBRARIES ${V8JS_LIB})
  add_dependencies(app_v8_js v8)
endif()

