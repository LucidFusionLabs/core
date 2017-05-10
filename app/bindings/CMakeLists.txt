# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_js STATIC_LIBRARY SOURCES null_js.cpp)
lfl_add_target(app_null_tu STATIC_LIBRARY SOURCES null_tu.cpp)

if(LFL_CUDA)
  lfl_add_target(app_cuda STATIC_LIBRARY SOURCES cuda.cpp
                 INCLUDE_DIRECTORIES ${CUDA_INCLUDE} LINK_LIBRARIES ${CUDA_LIB})
endif()

if(LFL_LUA)
  lfl_add_target(app_lua STATIC_LIBRARY SOURCES lua.cpp
                 INCLUDE_DIRECTORIES ${LUA_INCLUDE} LINK_LIBRARIES ${LUA_LIB})
endif()

if(LFL_V8JS)
  lfl_add_target(app_v8_js STATIC_LIBRARY SOURCES v8_js.cpp
                 INCLUDE_DIRECTORIES ${V8JS_INCLUDE} LINK_LIBRARIES ${V8JS_LIB})
endif()

