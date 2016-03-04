# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

include_directories(${LFL_SOURCE_DIR})
add_library(app_null_js STATIC null_js.cpp)
add_library(app_null_tu STATIC null_tu.cpp)

if(LFL_V8JS)
  add_library(app_v8_js STATIC v8_js.cpp)
  target_include_directories(app_v8_js PUBLIC ${V8JS_INCLUDE})
  target_link_libraries(app_v8_js PUBLIC ${V8JS_LIB})
endif()

if(LFL_LIBCLANG)
  add_library(app_clang_tu STATIC clang_tu.cpp)
  target_include_directories(app_clang_tu PUBLIC ${LIBCLANG_INCLUDE})
  target_link_libraries(app_clang_tu PUBLIC ${LIBCLANG_LIB})
endif()
