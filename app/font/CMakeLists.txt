# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_SOURCE_DIR})
lfl_add_library(app_null_ttf STATIC null_ttf.cpp)

if(LFL_FREETYPE)
  lfl_add_library(app_freetype_ttf STATIC freetype_ttf.cpp)
  target_include_directories(app_freetype_ttf PUBLIC ${FREETYPE_INCLUDE})
  target_link_libraries(app_freetype_ttf PUBLIC ${FREETYPE_LIB})
endif()
