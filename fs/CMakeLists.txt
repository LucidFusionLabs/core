# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_EASTL 1)
set(LFL_FFMPEG 1)
set(LFL_GLOG 1)
set(LFL_HEADLESS 1)
set(LFL_LAME 1)
set(LFL_X264 1)

lfl_project(fs)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

add_executable(fs fs.cpp ${LFL_RECOGNITION_SOURCE})

target_link_libraries(fs ${LFAPP_LIB})
lfapp_post_build_start(fs FusionServer)

