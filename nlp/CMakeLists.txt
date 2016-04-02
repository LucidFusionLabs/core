# $Id: CMakeLists.txt 1336 2014-12-08 09:29:59Z justin $

lfl_project(nlp)
lfl_add_target(nlp STATIC_LIBRARY SOURCES nlp.cpp)

lfl_add_target(nlp-trainer EXECUTABLE SOURCES trainer.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} app_null_framework app_null_graphics
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
               app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
               app_null_ssl app_null_js app_null_tu nlp ${LFL_APP_OS})
lfl_post_build_start(nlp-trainer nlp-trainer NLPTrainer)
