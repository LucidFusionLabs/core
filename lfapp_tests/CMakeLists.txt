# $Id: CMakeLists.txt 1334 2014-11-28 09:14:21Z justin $

set(LFL_TEST 1)
set(LFL_GLOG 1)
set(LFL_GTEST 1)
set(LFL_HEADLESS 1)
set(LFL_OPENCV 1)
set(LFL_LIBCSS 1)
set(LFL_REGEX 1)
set(LFL_SREGEX 1)
set(LFL_ICONV 1)
set(LFL_PNG 1)
set(LFL_JUDY 1)
set(LFL_OPENSSL 1)

lfl_project(lfapp_test)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})
include_directories(${GTEST_INCLUDE})

add_executable(tests test.cpp assets_test.cpp gui_test.cpp lfapp_test.cpp lftypes_test.cpp file_test.cpp string_test.cpp dom_test.cpp flow_test.cpp math_test.cpp video_test.cpp)
target_link_libraries(tests ${LFAPP_LIB} ${GTEST_LIB})

add_executable(datastructure_test datastructure_test.cpp)
target_link_libraries(datastructure_test ${LFAPP_LIB} ${GTEST_LIB})

add_executable(regex_test regex_test.cpp)
target_link_libraries(regex_test ${LFAPP_LIB} ${GTEST_LIB})
