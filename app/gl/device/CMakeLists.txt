# $Id$

lfl_add_target(app_null_graphics STATIC_LIBRARY SOURCES null_graphics.cpp)

set(OPENGL_DEFS)
set(OPENGL_INCS ${OPENGL_INCLUDE})
set(OPENGL_LIBS ${OPENGL_LIB})
if(LFL_GLES1)
  set(OPENGL_DEFS ${OPENGL_DEFS} -DLFL_GLES1)
endif()
if(LFL_GLES2)
  set(OPENGL_DEFS ${OPENGL_DEFS} -DLFL_GLES2)
endif()

if(LFL_OPENGL)
  set(DEFS ${OPENGL_DEFS})
  set(INCS ${OPENGL_INCS})
  set(LIBS ${OPENGL_LIBS})
  if(LFL_GLEW)
    set(DEFS ${DEFS} ${GLEW_DEF})
    set(INCS ${INCS} ${GLEW_INCLUDE})
    set(LIBS ${LIBS} ${GLEW_LIB})
  endif()

  lfl_add_target(app_opengl_graphics STATIC_LIBRARY SOURCES opengl_graphics.cpp
                 COMPILE_DEFINITIONS ${DEFS} INCLUDE_DIRECTORIES ${INCS} LINK_LIBRARIES ${LIBS})
endif()

if(LFL_QT)
  lfl_enable_qt()
  lfl_add_target(app_qt_graphics STATIC_LIBRARY SOURCES opengl_graphics.cpp
                 COMPILE_DEFINITIONS -DLFL_QTGL ${OPENGL_DEFS}
                 INCLUDE_DIRECTORIES ${QT_INCLUDE} ${OPENGL_INCS} LINK_LIBRARIES ${OPENGL_LIBS})
endif()

if(LFL_DIRECTX)
  lfl_add_target(app_directx_graphics STATIC_LIBRARY SOURCES directx_graphics.cpp
                 INCLUDE_DIRECTORIES ${DIRECTX_INCLUDE} LINK_LIBRARIES ${DIRECTX_LIB})
endif()

if(LFL_BGFX)
  lfl_add_target(app_bgfx_graphics STATIC_LIBRARY SOURCES bgfx_graphics.cpp
                 INCLUDE_DIRECTORIES ${BGFX_INCLUDE} LINK_LIBRARIES ${BGFX_LIB} "-framework Metal")
  add_dependencies(app_bgfx_graphics bgfx)
endif()

