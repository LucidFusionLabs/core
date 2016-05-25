# $Id: CMakeLists.txt 1334 2014-11-28 09:14:21Z justin $

lfl_project(app_test)
lfl_add_target(tests EXECUTABLE
               INCLUDE_DIRECTORIES ${GTEST_INCLUDE} ${LIBCSS_INCLUDE}
               COMPILE_DEFINITIONS -DLFL_LIBCSS -DLFL_PNG
               SOURCES tests.cpp assets_test.cpp gui_test.cpp app_test.cpp types_test.cpp file_test.cpp
               string_test.cpp dom_test.cpp flow_test.cpp math_test.cpp wire_test.cpp video_test.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} ${GTEST_LIB} app_null_framework app_null_graphics
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
               app_libpng_png app_null_jpeg app_null_gif app_null_ogg app_libcss_css app_null_ttf
               app_null_ssl app_null_js app_null_tu ${LFL_APP_OS})

add_custom_command(TARGET tests POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E make_directory lib
                   COMMAND cp ${OPENCV_LIB_FILES} lib)

lfl_add_target(datastructure_tests EXECUTABLE
               INCLUDE_DIRECTORIES "${GTEST_INCLUDE}"
               SOURCES datastructure_tests.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} ${GTEST_LIB} app_null_framework app_null_graphics
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
               app_libpng_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
               app_null_ssl app_null_js app_null_tu ${LFL_APP_OS})

lfl_add_target(regex_tests EXECUTABLE
               INCLUDE_DIRECTORIES ${GTEST_INCLUDE}
               SOURCES regex_tests.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} ${GTEST_LIB} app_null_framework app_null_graphics
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
               app_libpng_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
               app_null_ssl app_null_js app_null_tu ${LFL_APP_OS})

lfl_add_target(gl_tests EXECUTABLE SOURCES gl_tests.cpp INCLUDE_DIRECTORIES ${GTEST_INCLUDE}
               LINK_LIBRARIES ${LFL_APP_LIB} ${GTEST_LIB} ${LFL_APP_FRAMEWORK} ${LFL_APP_GRAPHICS}
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
               app_libpng_png app_null_jpeg app_null_gif app_null_ogg app_libcss_css app_null_ttf
               app_null_ssl app_null_js app_null_tu ${LFL_APP_OS})
set(LFL_APP_LIB_FILES ${OPENCV_LIB_FILES})
lfl_post_build_start(gl_tests gl_tests GLTestsApp)

add_test(NAME app_tests COMMAND tests)
