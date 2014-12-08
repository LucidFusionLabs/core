# $Id: CMakeLists.txt 1334 2014-11-28 09:14:21Z justin $

set(LFL_PNG 1)
set(LFL_CLING 1)
set(LFL_OPENCV 1)

if(LFL_ANDROID)
 set(LFL_GLES2 1)
elseif(LFL_IPHONE)
 set(LFL_GLES2 1)
else(LFL_ANDROID)
 set(LFL_GLOG 1)
 set(LFL_GLFWINPUT 1)
 set(LFL_GLFWVIDEO 1)
endif(LFL_ANDROID)

lfl_project(calculator)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

if(LFL_ANDROID)
add_library(calculator calculator.cpp)
else(LFL_ANDROID)
add_executable(calculator calculator.cpp ${SDL_MAIN})
target_link_libraries(calculator ${LFAPP_LIB})
endif(LFL_ANDROID)

