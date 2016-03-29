# $Id: CMakeLists.txt 1336 2014-12-08 09:29:59Z justin $

lfl_project(nlp)
add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})

# nlp
lfl_add_library(nlp nlp.cpp)

# trainer
lfl_add_executable(nlp-trainer trainer.cpp)
target_link_libraries(nlp-trainer ${LFL_APP_LIB} app_null_framework app_null_graphics
                      app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
                      app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex
                      app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
                      app_null_ssl app_null_js app_null_tu app_null_os nlp)
lfl_post_build_start(nlp-trainer nlp-trainer NLPTrainer)

