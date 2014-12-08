# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_EASTL 1)
set(LFL_GEOIP 1)
set(LFL_PCAP 1)
set(LFL_PNG 1)

if(LFL_ANDROID)
 set(LFL_GLES2 1)
elseif(LFL_IPHONE)
 set(LFL_GLES2 1)
else(LFL_ANDROID)
 set(LFL_FFMPEG 1)
 set(LFL_GLFWINPUT 1)
 set(LFL_GLFWVIDEO 1)
 set(LFL_GLOG 1)
 set(LFL_LAME 1)
 set(LFL_PORTAUDIO 1)
 set(LFL_X264 1)
endif(LFL_ANDROID)

lfl_project(cb)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

if(LFL_ANDROID)
add_library(cb cb.cpp)
else(LFL_ANDROID)
add_executable(cb cb.cpp ${SDL_MAIN})
target_link_libraries(cb ${LFAPP_LIB})
endif(LFL_ANDROID)

