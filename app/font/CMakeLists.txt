# $Id$

if(LFL_FREETYPE)
  lfl_add_target(app_freetype_ttf STATIC_LIBRARY SOURCES freetype_ttf.cpp
                 INCLUDE_DIRECTORIES ${FREETYPE_INCLUDE} LINK_LIBRARIES ${FREETYPE_LIB})
endif()

set(FONT_SOURCE atlas_font.cpp ipc_font.cpp)
set(FONT_LINK_LIBRARIES lfl_app)
if(LFL_APPLE)
  set(FONT_SOURCE ${FONT_SOURCE} coretext_font.cpp)
elseif(LFL_WINDOWS)
  set(FONT_SOURCE ${FONT_SOURCE} gdi_font.cpp)
elseif(LFL_ANDROID AND LFL_FREETYPE)
  set(FONT_SOURCE ${FONT_SOURCE} android_font.cpp)
  set(FONT_LINK_LIBRARIES ${FONT_LINK_LIBRARIES} app_freetype_ttf)
elseif(LFL_LINUX AND LFL_FREETYPE)
  set(FONT_SOURCE ${FONT_SOURCE} fc_font.cpp)
  set(FONT_LINK_LIBRARIES ${FONT_LINK_LIBRARIES} app_freetype_ttf -lfontconfig)
endif()
lfl_add_target(app_font STATIC_LIBRARY SOURCES ${FONT_SOURCE} LINK_LIBRARIES ${FONT_LINK_LIBRARIES})
lfl_add_target(app_null_ttf STATIC_LIBRARY SOURCES null_ttf.cpp)
