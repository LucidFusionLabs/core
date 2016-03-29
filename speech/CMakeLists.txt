# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

lfl_project(speech)
add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})

# recognition
add_library(recognition STATIC speech.cpp recognition.cpp voice.cpp)

# trainer
lfl_add_executable(speech-trainer trainer.cpp)
target_link_libraries(speech-trainer ${LFL_APP_LIB} app_null_framework app_null_graphics
                      app_null_audio app_null_camera app_opencv_matrix app_ffmpeg_fft app_simple_loader
                      app_ffmpeg_resampler app_iconv_convert app_stdregex_regex app_null_sregex app_libarchive_archive
                      app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
                      app_null_ssl app_null_js app_null_tu app_null_os recognition)
lfl_post_build_start(speech-trainer speech-trainer SpeechTrainer)

# mfcp
lfl_add_executable(mfcp mfcp.cpp)
target_link_libraries(mfcp ${LFL_APP_LIB} app_null_framework app_null_graphics
                      app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
                      app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex app_libarchive_archive
                      app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
                      app_null_ssl app_null_js app_null_tu app_null_os recognition)

# vpcat
lfl_add_executable(vpcat vpcat.cpp)
target_link_libraries(vpcat ${LFL_APP_LIB} app_null_framework app_null_graphics
                      app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
                      app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex app_libarchive_archive
                      app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
                      app_null_ssl app_null_js app_null_tu app_null_os recognition)
