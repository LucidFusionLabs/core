# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_EASTL 1)
set(LFL_HEADLESS 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
 set(LFL_GLOG 1)
endif(LFL_ANDROID)

lfl_project(master)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

if(LFL_ANDROID)
add_library(master master.cpp)
else(LFL_ANDROID)
add_executable(master master.cpp ${SDL_MAIN})
target_link_libraries(master ${LFAPP_LIB})
endif(LFL_ANDROID)

