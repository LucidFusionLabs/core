# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_HEADLESS 1)
set(LFL_PROTOBUF 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
 set(LFL_GLOG 1)
endif(LFL_ANDROID)

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR})
lfl_project(crawler)

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HDR crawler.proto)

# crawler
add_executable(crawler crawler.cpp ${PROTO_SRC})
target_link_libraries(crawler ${LFL_APP_LIB} app_null_framework app_null_graphics
                      app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
                      app_simple_resampler app_iconv_convert app_regexp_regex app_null_sregex
                      app_null_png app_null_jpeg app_null_gif app_libcss_css app_null_ttf app_openssl_ssl
                      app_null_js app_null_tu app_null_os app_protobuf_proto)
lfl_post_build_start(crawler crawler crawler)
