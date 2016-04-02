# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

set(FONT_SOURCE atlas_font.cpp ipc_font.cpp)
if(LFL_APPLE)
  set(FONT_SOURCE ${FONT_SOURCE} coretext_font.cpp)
elseif(LFL_WINDOWS)
  set(FONT_SOURCE ${FONT_SOURCE} gdi_font.cpp)
endif()
lfl_add_target(app_font STATIC_LIBRARY SOURCES ${FONT_SOURCE}
               INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/..)
lfl_add_target(app_null_ttf STATIC_LIBRARY SOURCES null_ttf.cpp)

if(LFL_FREETYPE)
  lfl_add_target(app_freetype_ttf STATIC_LIBRARY SOURCES freetype_ttf.cpp
                 INCLUDE_DIRECTORIES ${FREETYPE_INCLUDE} LINK_LIBRARIES ${FREETYPE_LIB})
endif()
