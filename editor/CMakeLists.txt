# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_PNG 1)

if(LFL_ANDROID)
 set(LFL_GLES2 1)
elseif(LFL_IPHONE)
 set(LFL_GLES2 1)
else(LFL_ANDROID)
 set(LFL_GLOG 1)
 set(LFL_GLFWINPUT 1)
 set(LFL_GLFWVIDEO 1)
endif(LFL_ANDROID)

lfl_project(editor)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

if(LFL_ANDROID)
add_library(editor editor.cpp)
else(LFL_ANDROID)
add_executable(editor editor.cpp)
target_link_libraries(editor ${LFAPP_LIB})
endif(LFL_ANDROID)

