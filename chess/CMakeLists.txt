# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_PNG 1)
set(LFL_GLES2 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
    set(LFL_FFMPEG 1)
    set(LFL_GLFWINPUT 1)
    set(LFL_GLFWVIDEO 1)
    set(LFL_GLOG 1)
    set(LFL_LAME 1)
    set(LFL_PORTAUDIO 1)
    set(LFL_X264 1)
endif(LFL_ANDROID)

lfl_project(chess)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

if(LFL_ANDROID)
    add_library(chess chess.cpp)
else(LFL_ANDROID)
    add_executable(chess chess.cpp ${SDL_MAIN})
    target_link_libraries(chess ${LFAPP_LIB})
    lfapp_post_build_start(chess LChess)
endif(LFL_ANDROID)

include_directories(${GTEST_INCLUDE})
add_executable(chess_tests chess_tests.cpp)
target_link_libraries(chess_tests ${LFAPP_LIB} ${GTEST_LIB})
