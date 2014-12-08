# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_EASTL 1)
set(LFL_GLOG 1)
set(LFL_HEADLESS 1)

lfl_project(mapreduce)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

# mapreduce
add_executable(mapreduce mapreduce.cpp ${SDL_MAIN})
target_link_libraries(mapreduce ${LFAPP_LIB})
lfapp_post_build(mapreduce mapreduce)

