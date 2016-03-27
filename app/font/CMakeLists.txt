# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR}/..)

set(FONT_SOURCE atlas_font.cpp ipc_font.cpp)
if(LFL_APPLE)
  set(FONT_SOURCE ${FONT_SOURCE} coretext_font.cpp)
elseif(LFL_WINDOWS)
  set(FONT_SOURCE ${FONT_SOURCE} gdi_font.cpp)
endif()
lfl_add_library(app_font STATIC ${FONT_SOURCE})
lfl_add_library(app_null_ttf STATIC null_ttf.cpp)

if(LFL_FREETYPE)
  lfl_add_library(app_freetype_ttf STATIC freetype_ttf.cpp)
  target_include_directories(app_freetype_ttf PUBLIC ${FREETYPE_INCLUDE})
  target_link_libraries(app_freetype_ttf PUBLIC ${FREETYPE_LIB})
endif()
