# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

lfl_add_target(app_null_css STATIC_LIBRARY SOURCES null_css.cpp)
lfl_add_target(app_browser STATIC_LIBRARY SOURCES browser.cpp)

if(LFL_LIBCSS)
  lfl_add_target(app_libcss_css STATIC_LIBRARY SOURCES libcss_css.cpp
                 INCLUDE_DIRECTORIES ${LIBCSS_INCLUDE} ${LFL_CORE_BINARY_DIR}/app 
                 LINK_LIBRARIES ${LIBCSS_LIB})
  set(CMAKE_AUTOMOC)
endif()

lfl_project(crawler)
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HDR crawler.proto)
add_executable(crawler crawler.cpp ${PROTO_SRC})
target_include_directories(crawler PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(crawler PUBLIC ${LFL_APP_LIB} app_null_framework app_null_graphics
                      app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
                      app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
                      app_null_png app_null_jpeg app_null_gif app_null_ogg app_libcss_css app_null_ttf
                      app_openssl_ssl app_null_js app_null_tu app_protobuf_proto ${LFL_APP_OS})
lfl_post_build_start(crawler crawler crawler)
