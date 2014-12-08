# $Id: CMakeLists.txt 1336 2014-12-08 09:29:59Z justin $

set(LFL_EASTL 1)
set(LFL_HEADLESS 1)
set(LFL_OPENSSL 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
 set(LFL_GLOG 1)
endif(LFL_ANDROID)

lfl_project(senators)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

add_executable(senators senators.cpp ${SDL_MAIN})
target_link_libraries(senators ${LFAPP_LIB})
lfapp_post_build(senators senators)

