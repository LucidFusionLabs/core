# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_EASTL 1)
set(LFL_HEADLESS 1)
set(LFL_OPENSSL 1)
set(LFL_EPOLL 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
 set(LFL_GLOG 1)
endif(LFL_ANDROID)

lfl_project(blaster)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

add_executable(blaster blaster.cpp ${SDL_MAIN})
target_link_libraries(blaster ${LFAPP_LIB})

add_executable(resolver resolver.cpp ${SDL_MAIN})
target_link_libraries(resolver ${LFAPP_LIB})

add_executable(tester tester.cpp ${SDL_MAIN})
target_link_libraries(tester ${LFAPP_LIB})

add_executable(receiver receiver.cpp ${SDL_MAIN})
target_link_libraries(receiver ${LFAPP_LIB})

