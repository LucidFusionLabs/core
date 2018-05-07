# $Id$

if(LFL_MOBILE)
  return()
endif()

lfl_add_target(cluster EXECUTABLE SOURCES cluster.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} app_null_framework app_null_graphics app_null_toolkit
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
               app_libpng_png app_null_jpeg app_null_gif app_null_ogg app_libcss_css app_null_ttf
               app_null_ssl app_null_js app_null_tu app_null_crashreporting ${LFL_APP_OS})
