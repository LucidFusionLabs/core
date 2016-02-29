# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

set(LFL_GLES2 1)
include_directories(${LFL_SOURCE_DIR})
add_library(app_null_graphics STATIC null_graphics.cpp)

add_library(app_opengl_graphics STATIC opengl_graphics.cpp)
if(LFL_GLES2)
  target_compile_definitions(app_opengl_graphics PUBLIC -DLFL_GLES2)
endif()
if(LFL_GLEW)
  target_include_directories(app_opengl_graphics PUBLIC ${OPENGL_INCLUDE} ${GLEW_INCLUDE})
  target_link_libraries(app_opengl_graphics PUBLIC ${OPENGL_LIB} ${GLEW_LIB})
endif()

if(LFL_QT)
  lfl_enable_qt()
  add_library(app_qt_graphics STATIC opengl_graphics.cpp)
  target_compile_definitions(app_qt_graphics PUBLIC -DLFL_QTGL)
  target_include_directories(app_qt_graphics PUBLIC ${QT_INCLUDE})
  if(LFL_GLES2)
    target_compile_definitions(app_qt_graphics PUBLIC -DLFL_GLES2)
  endif()
endif()
