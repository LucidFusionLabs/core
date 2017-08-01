# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

set(LFL_GLES2 1)

set(defs)
set(incs ${OPENGL_INCLUDE})
set(libs ${OPENGL_LIB})
if(LFL_GLES1)
  set(defs ${defs} -DLFL_GLES1)
endif()
if(LFL_GLES2)
  set(defs ${defs} -DLFL_GLES2)
endif()

set(qt_defs ${defs})
set(qt_incs ${incs})
set(qt_libs ${libs})

if(LFL_GLEW)
  set(defs ${defs} -DLFL_GLEW)
  set(incs ${incs} ${GLEW_INCLUDE})
  set(libs ${libs} ${GLEW_LIB})
  if(LFL_WINDOWS)
    set(defs ${defs} -DGLEW_STATIC)
  endif()
endif()

lfl_add_target(app_null_graphics STATIC_LIBRARY SOURCES null_graphics.cpp)
lfl_add_target(app_opengl_graphics STATIC_LIBRARY SOURCES opengl_graphics.cpp
               COMPILE_DEFINITIONS ${defs} INCLUDE_DIRECTORIES ${incs} LINK_LIBRARIES ${libs})

if(LFL_QT)
  lfl_enable_qt()
  lfl_add_target(app_qt_graphics STATIC_LIBRARY SOURCES opengl_graphics.cpp
                 COMPILE_DEFINITIONS -DLFL_QTGL ${qt_defs}
                 INCLUDE_DIRECTORIES ${QT_INCLUDE} ${qt_incs} LINK_LIBRARIES ${qt_libs})
endif()
