# $Id: CMakeLists.txt 1334 2014-11-28 09:14:21Z justin $

set(LFL_GLOG 1)
set(LFL_GTEST 1)
set(LFL_HEADLESS 1)
set(LFL_OPENCV 1)
set(LFL_LIBCSS 1)
set(LFL_REGEX 1)
set(LFL_SREGEX 1)
set(LFL_ICONV 1)
set(LFL_PNG 1)
set(LFL_OPENSSL 1)

lfl_project(app_test)
add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})
include_directories(${GTEST_INCLUDE})

add_executable(tests test.cpp assets_test.cpp gui_test.cpp lfapp_test.cpp lftypes_test.cpp file_test.cpp string_test.cpp dom_test.cpp flow_test.cpp math_test.cpp wire_test.cpp video_test.cpp)
target_link_libraries(tests ${LFL_APP_LIB} ${GTEST_LIB} app_null_platform app_null_graphics app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader app_simple_resampler app_iconv_convert app_regexp_regex app_libsregex_sregex app_libpng_png app_null_jpeg app_null_gif)

add_executable(datastructure_test datastructure_test.cpp)
target_link_libraries(datastructure_test ${LFL_APP_LIB} ${GTEST_LIB} app_null_platform app_null_graphics app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader app_simple_resampler app_iconv_convert app_regexp_regex app_libsregex_sregex app_libpng_png app_null_jpeg app_null_gif)

add_executable(regex_test regex_test.cpp)
target_link_libraries(regex_test ${LFL_APP_LIB} ${GTEST_LIB} app_null_platform app_null_graphics app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader app_simple_resampler app_iconv_convert app_regexp_regex app_libsregex_sregex app_libpng_png app_null_jpeg app_null_gif)
