# $Id$

lfl_add_target(app_simple_loader STATIC_LIBRARY SOURCES simple_loader.cpp)
lfl_add_target(app_simple_resampler STATIC_LIBRARY SOURCES simple_resampler.cpp)
lfl_add_target(app_null_png STATIC_LIBRARY SOURCES null_png.cpp)
lfl_add_target(app_null_jpeg STATIC_LIBRARY SOURCES null_jpeg.cpp)
lfl_add_target(app_null_gif STATIC_LIBRARY SOURCES null_gif.cpp)
lfl_add_target(app_null_archive STATIC_LIBRARY SOURCES null_archive.cpp)
lfl_add_target(app_null_proto STATIC_LIBRARY SOURCES null_proto.cpp)
lfl_add_target(app_null_ogg STATIC_LIBRARY SOURCES null_ogg.cpp)
lfl_add_target(app_stb_ogg STATIC_LIBRARY SOURCES stb_ogg.cpp)

if(LFL_PROTOBUF)
  add_library(app_protobuf_proto protobuf_proto.cpp)
  target_compile_definitions(app_protobuf_proto PUBLIC ${LFL_APP_DEF})
  target_include_directories(app_protobuf_proto PUBLIC ${LFL_APP_INCLUDE} ${PROTOBUF_INCLUDE_DIR})
  target_link_libraries(app_protobuf_proto PUBLIC ${PROTOBUF_LIBRARY})
  add_dependencies(app_protobuf_proto protoc)
endif()

if(LFL_PNG)
  lfl_add_target(app_libpng_png STATIC_LIBRARY SOURCES libpng_png.cpp
                 INCLUDE_DIRECTORIES ${PNG_INCLUDE} LINK_LIBRARIES ${PNG_LIB})
  if(LFL_EMSCRIPTEN)
    target_compile_options(app_libpng_png PUBLIC -s USE_LIBPNG=1)
  endif()
endif()

if(LFL_GIF)
  lfl_add_target(app_giflib_gif STATIC_LIBRARY SOURCES giflib_gif.cpp
                 INCLUDE_DIRECTORIES ${GIF_INCLUDE} LINK_LIBRARIES ${GIF_LIB})
endif()

if(LFL_JPEG)
  lfl_add_target(app_libjpeg_jpeg STATIC_LIBRARY SOURCES libjpeg_jpeg.cpp LINK_LIBRARIES ${JPEG_LIB})
endif()

if(LFL_VORBIS)
  lfl_add_target(app_libvorbis_ogg STATIC_LIBRARY SOURCES libvorbis_ogg.cpp
                 INCLUDE_DIRECTORIES ${OGG_INCLUDE} ${VORBIS_INCLUDE}
                 LINK_LIBRARIES ${OGG_LIB} ${VORBIS_LIB})
  if(LFL_EMSCRIPTEN)
    target_compile_options(app_libvorbis_ogg PUBLIC -s USE_OGG=1 USE_VORBIS=1)
  endif()
endif()

if(LFL_LIBARCHIVE)
  lfl_add_target(app_libarchive_archive STATIC_LIBRARY SOURCES libarchive_archive.cpp
                 INCLUDE_DIRECTORIES ${ARCHIVE_INCLUDE} LINK_LIBRARIES ${ARCHIVE_LIB})
endif()

if(LFL_FFMPEG)
  lfl_add_target(app_ffmpeg_loader STATIC_LIBRARY SOURCES ffmpeg_loader.cpp
                 INCLUDE_DIRECTORIES ${FFMPEG_INCLUDE} LINK_LIBRARIES ${FFMPEG_LIB})
  lfl_add_target(app_ffmpeg_resampler STATIC_LIBRARY SOURCES ffmpeg_resampler.cpp
                 INCLUDE_DIRECTORIES ${FFMPEG_INCLUDE} LINK_LIBRARIES ${FFMPEG_LIB})
endif()

if(LFL_EMSCRIPTEN)
  lfl_add_target(app_sdl_loader STATIC_LIBRARY SOURCES sdl_loader.cpp
                 INCLUDE_DIRECTORIES ${SDL_INCLUDE} LINK_LIBRARIES ${SDL_LIB})
endif()
