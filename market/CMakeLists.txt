# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_GLES2 1)
set(LFL_PROTOBUF 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
if(NOT LFL_HEADLESS)
 set(LFL_FFMPEG 1)
 set(LGL_GLEW 1)
 set(LFL_GLFWINPUT 1)
 set(LFL_GLFWVIDEO 1)
 set(LFL_PORTAUDIO 1)
 if(NOT WIN32)
  set(LFL_GLOG 1)
  set(LFL_LAME 1)
  set(LFL_X264 1)
 endif(NOT WIN32)
endif(NOT LFL_HEADLESS)
endif(LFL_ANDROID)

lfl_project(market)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HDR market.proto)

if(LFL_ANDROID)
add_library(market market.cpp)
else(LFL_ANDROID)
add_executable(market market.cpp ${SDL_MAIN} ${PROTO_SRC} ../crawler/crawler.pb.cc)
target_link_libraries(market ${LFAPP_LIB})
endif(LFL_ANDROID)

