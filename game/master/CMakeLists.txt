# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

if(LFL_MOBILE)
  return()
endif()

lfl_project(master)
lfl_add_target(master EXECUTABLE SOURCES master.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} app_null_framework app_null_graphics app_null_toolkit
               app_null_audio app_null_camera app_null_matrix app_null_fft app_simple_loader
               app_simple_resampler ${LFL_APP_CONVERT} app_null_regex app_null_sregex
               app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
               app_null_ssl app_null_js app_null_tu app_null_crashreporting ${LFL_APP_OS})
