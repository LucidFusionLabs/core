# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

lfl_project($BINNAME)
lfl_add_target($BINNAME $BINNAME.cpp $BINNAME.rc)
target_compile_definitions($BINNAME PUBLIC ${LFL_APP_DEF})
target_include_directories($BINNAME PUBLIC ${LFL_APP_INCLUDE})
target_link_libraries($BINNAME ${LFL_APP_LIB} ${LFL_APP_FRAMEWORK} ${LFL_APP_GRAPHICS}
                      app_null_audio app_null_camera app_null_matrix app_null_fft
                      app_simple_resampler app_simple_loader ${LFL_APP_CONVERT} app_libpng_png
                      app_null_jpeg app_null_gif app_null_ogg app_null_css ${LFL_APP_FONT}
                      app_null_ssl app_null_js app_null_tu app_null_crypto app_null_dh ${LFL_APP_OS})
set(LFL_APP_ASSETS ${LFL_APP_ASSETS} ${LFL_SOURCE_DIR}/core/app/assets/Nobile.*)
lfl_post_build_start($BINNAME $BINNAME $PKGNAME)
