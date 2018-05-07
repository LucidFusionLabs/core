# $Id$

lfl_project(speech)

lfl_add_target(recognition STATIC_LIBRARY SOURCES speech.cpp recognition.cpp voice.cpp)

if(LFL_MOBILE)
  return()
endif()

lfl_add_target(speech-trainer EXECUTABLE SOURCES trainer.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} app_null_framework app_null_graphics app_null_toolkit
               app_null_audio app_null_camera app_opencv_matrix app_ffmpeg_fft app_simple_loader
               app_ffmpeg_resampler app_iconv_convert app_stdregex_regex app_null_sregex app_libarchive_archive
               app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
               app_null_ssl app_null_js app_null_tu app_null_crashreporting recognition ${LFL_APP_OS})
lfl_post_build_start(speech-trainer speech-trainer SpeechTrainer)

lfl_add_target(mfcp EXECUTABLE SOURCES mfcp.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} app_null_framework app_null_graphics app_null_toolkit
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex app_libarchive_archive
               app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
               app_null_ssl app_null_js app_null_tu app_null_crashreporting recognition ${LFL_APP_OS})

lfl_add_target(vpcat EXECUTABLE SOURCES vpcat.cpp
               LINK_LIBRARIES ${LFL_APP_LIB} app_null_framework app_null_graphics app_null_toolkit
               app_null_audio app_null_camera app_opencv_matrix app_null_fft app_simple_loader
               app_simple_resampler app_iconv_convert app_stdregex_regex app_null_sregex app_libarchive_archive
               app_null_png app_null_jpeg app_null_gif app_null_ogg app_null_css app_null_ttf
               app_null_ssl app_null_js app_null_tu app_null_crashreporting recognition ${LFL_APP_OS})
