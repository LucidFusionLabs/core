# $Id: CMakeLists.txt 1325 2014-10-29 06:08:31Z justin $

FOREACH(flag CMAKE_CXX_FLAGS_DEBUG)
  STRING(REPLACE "-Wold-style-cast" "" "${flag}" "${${flag}}")
ENDFOREACH()

if(LFL_WINDOWS)
  # make blank unistd.h
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/fake_unistd/unistd.h)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fake_unistd)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fake_unistd/unistd.h "")
  endif()
endif()

# optionally hijack OpenCV/3rdparty/*/CMakeLists.txt
macro(ocv_install_target)
endmacro()
macro(ocv_warnings_disable)
endmacro()
macro(ocv_include_directories)
  include_directories(${ARGN})
endmacro()
macro(ocv_list_filterout lst regex)
  foreach(item ${${lst}})
    if(item MATCHES "${regex}")
      list(REMOVE_ITEM ${lst} "${item}")
    endif()
  endforeach()
endmacro()

# boost
set(BOOST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/boost PARENT_SCOPE)

# zlib
if(LFL_WINDOWS)
  set(3P_LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/zlib)
  set(ZLIB_LIBRARY zlib)
  add_subdirectory(OpenCV/3rdparty/zlib zlib)
endif()

# png
if(LFL_PNG AND NOT LFL_EMSCRIPTEN)
  if(LFL_WINDOWS)
    if(FALSE)
      ExternalProject_Add(libpng SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libpng
                          PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libpng LIST_SEPARATOR ::
                          CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/libpng
                          -DZLIB_INCLUDE_DIR:STRING=${CMAKE_CURRENT_BINARY_DIR}/zlib::${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/zlib
                          -DZLIB_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/zlib/${CMAKE_BUILD_TYPE}/zlib.lib)
      add_windows_dependency(libpng zlib)
      add_library(libpng16 IMPORTED STATIC GLOBAL)
      add_dependencies(libpng16 libpng)
      set_property(TARGET libpng16 PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libpng/lib/libpng16_staticd.lib)
    else()
      if(LFL_WINDOWS)
        set(ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/zlib ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/zlib)
        set(ZLIB_LIBRARIES zlib)
      endif()
      if(LFL_ANDROID)
        set(ENABLE_NEON ON)
      endif()
      set(3P_LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/libpng)
      set(PNG_LIBRARY libpng16)
      add_subdirectory(OpenCV/3rdparty/libpng libpng)
      set_property(TARGET libpng16 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/libpng)
    endif()
    set(PNG_LIB libpng16 PARENT_SCOPE)
  elseif(LFL_LINUX)
    set(PNG_LIB -lpng PARENT_SCOPE)
  else()
    ExternalProject_Add(libpng SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libpng
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libpng
                        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/libpng
                        -DCMAKE_TOOLCHAIN_FILE:string=${CMAKE_TOOLCHAIN_FILE}
                        -DPNG_SHARED=FALSE -DPNG_TESTS=FALSE)
    add_library(libpng16 IMPORTED STATIC GLOBAL)
    add_dependencies(libpng16 libpng)
    set_property(TARGET libpng16 PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libpng/lib/libpng16.a)
    add_interface_include_directory(libpng16 ${CMAKE_CURRENT_BINARY_DIR}/libpng/include/libpng16)
    set(PNG_LIB libpng16 PARENT_SCOPE)
  endif()
endif()

# jpeg
if(LFL_JPEG)
  if(WIN32 OR LFL_IOS)
    set(3P_LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/libjpeg)
    set(JPEG_LIBRARY libjpeg)
    add_subdirectory(OpenCV/3rdparty/libjpeg libjpeg)
    target_include_directories(libjpeg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/libjpeg)
    set(JPEG_LIB libjpeg PARENT_SCOPE)
  else()
    set(JPEG_CONFIGURE_ENV ${CONFIGURE_ENV})
    if(LFL_OSX)
      set(JPEG_CONFIGURE_ENV ${JPEG_CONFIGURE_ENV} NASM=/opt/local/bin/nasm)
    endif()
    ExternalProject_Add(libjpeg-turbo LOG_CONFIGURE ON LOG_BUILD ON
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo
                        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${JPEG_CONFIGURE_ENV}
                        <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> ${CONFIGURE_OPTIONS}
                        DOWNLOAD_COMMAND rm -rf libjpeg-turbo
                        COMMAND cp -R ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo libjpeg-turbo
                        COMMAND ${CMAKE_COMMAND} -E chdir libjpeg-turbo autoreconf -fi)
    add_library(libjpeg IMPORTED STATIC GLOBAL)
    add_dependencies(libjpeg libjpeg-turbo)
    set_property(TARGET libjpeg PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a)
    add_interface_include_directory(libjpeg ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/include)
    set(JPEG_LIB libjpeg PARENT_SCOPE)
  endif()
endif()

# gif
if(LFL_GIF)
  add_library(libgif STATIC giflib/lib/dgif_lib.c giflib/lib/egif_lib.c giflib/lib/gif_err.c
              giflib/lib/gif_font.c giflib/lib/gif_hash.c giflib/lib/gifalloc.c giflib/lib/quantize.c
              giflib/lib/openbsd-reallocarray.c)
  if(WIN32)
    target_include_directories(libgif PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/fake_unistd)
  endif()
  target_include_directories(libgif PUBLIC giflib/lib)
  set(GIF_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/giflib/lib PARENT_SCOPE)
  set(GIF_LIB libgif PARENT_SCOPE)
endif()

# freetype
if(LFL_FREETYPE)
  if(LFL_LINUX)
    set(FREETYPE_INCLUDE /usr/include/freetype2 PARENT_SCOPE)
    set(FREETYPE_LIB -lfreetype PARENT_SCOPE)
  elseif(LFL_WINDOWS)
    INCLUDE_EXTERNAL_MSPROJECT(freetype ${CMAKE_CURRENT_SOURCE_DIR}/freetype/builds/win32/vc2008/freetype.vcxproj)
    set(FREETYPE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/freetype/include PARENT_SCOPE)
    set(FREETYPE_LIB freetype239_D.lib PARENT_SCOPE)
  else()
    set(FREETYPE_CMAKE_DEF)
    if(LFL_ANDROID)
      set(FREETYPE_CMAKE_DEF ${FREETYPE_CMAKE_DEF} -DWITH_BZip2=OFF)
    endif()
    ExternalProject_Add(freetype SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/freetype
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/freetype
                        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/freetype
                        -DCMAKE_TOOLCHAIN_FILE:string=${CMAKE_TOOLCHAIN_FILE} ${FREETYPE_CMAKE_DEF})
    add_library(libfreetype IMPORTED STATIC GLOBAL)
    add_dependencies(libfreetype freetype)
    set_property(TARGET libfreetype PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/freetype/lib/libfreetype.a)
    set(FREETYPE_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/freetype/include/freetype2 PARENT_SCOPE)
    set(FREETYPE_LIB libfreetype PARENT_SCOPE)
  endif()
endif()

# harfbuzz
# ./configure --enable-static --with-coretext=yes --prefix=`pwd`/bin
if(LFL_HARFBUZZ)
  set(HARFBUZZ_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/harfbuzz/bin/include PARENT_SCOPE)
  set(HARFBUZZ_LIB ${CMAKE_CURRENT_BINARY_DIR}/harfbuzz/src/.libs/libharfbuzz.a PARENT_SCOPE)
endif()

# libcss
if(LFL_LIBCSS)
  add_subdirectory(libcss)
  set(LIBCSS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libcss/libcss/include
      ${CMAKE_CURRENT_SOURCE_DIR}/libcss/libwapcaplet/include PARENT_SCOPE)
  set(LIBCSS_LIB libcss PARENT_SCOPE)
endif()

# protobuf
if(LFL_PROTOBUF)
  option(protobuf_BUILD_TESTS OFF)
  add_subdirectory(protobuf/cmake protobuf)
  set(PROTOBUF_LIBRARY libprotobuf PARENT_SCOPE)
endif()

# flatbuffers
if(LFL_FLATBUFFERS)
  set(CMAKE_CXX_FLAGS_ORIG ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
  add_subdirectory(flatbuffers)
  set(FLATBUFFERS_LIB flatbuffers PARENT_SCOPE)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_ORIG})
endif()

# capnproto
if(LFL_CAPNPROTO)
  ExternalProject_Add(capnproto SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/capnproto/c++
                                PREFIX ${CMAKE_CURRENT_BINARY_DIR}/capnproto
                                CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/capnproto)
endif()

# glog
if(LFL_GLOG)
  set(GLOG_CMAKE_ARGS)
  if(LFL_XCODE)
    set(GLOG_CMAKE_ARGS ${GLOG_CMAKE_ARGS} -DHAVE_UINT16_T=TRUE)
  endif()
  ExternalProject_Add(googlelog LOG_CONFIGURE ON LOG_BUILD ON
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glog
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/glog
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/glog
                      -DCMAKE_TOOLCHAIN_FILE:string=${CMAKE_TOOLCHAIN_FILE} ${GLOG_CMAKE_ARGS})
  add_library(libglog IMPORTED STATIC GLOBAL)
  add_dependencies(libglog googlelog)
  get_static_library_name(_lib glog)
  set_property(TARGET libglog PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/glog/lib/${_lib})
  set(GLOG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/glog/include PARENT_SCOPE)
  set(GLOG_LIB libglog PARENT_SCOPE)
endif()

# gtest
if(LFL_GTEST)
  ExternalProject_Add(googletest LOG_CONFIGURE ON LOG_BUILD ON
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/googletest
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/googletest
                      -DCMAKE_TOOLCHAIN_FILE:string=${CMAKE_TOOLCHAIN_FILE})
  add_library(libgtest IMPORTED STATIC GLOBAL)
  add_dependencies(libgtest googletest)
  get_static_library_name(_lib gtest)
  set_property(TARGET libgtest PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/googletest/lib/${_lib})
  set(GTEST_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/googletest/include PARENT_SCOPE)
  set(GTEST_LIB libgtest PARENT_SCOPE)
endif()

# tcmalloc
if(LFL_TCMALLOC)
  ExternalProject_Add(gperftools LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gperftools
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gperftools
                      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-minimal)
  add_library(libtcmalloc_minimal IMPORTED STATIC GLOBAL)
  add_dependencies(libtcmalloc_minimal gperftools)
  set_property(TARGET libtcmalloc_minimal PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/gperftools/lib/libtcmalloc_minimal.a)
  set(TCMALLOC_LIB libtcmalloc_minimal
      -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free PARENT_SCOPE)
endif()

# fabric
if(LFL_FABRIC)
  set(FABRIC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/fabric-ios/Fabric.framework/Headers
      ${CMAKE_CURRENT_SOURCE_DIR}/fabric-ios/Crashlytics.framework/Headers PARENT_SCOPE)
  set(FABRIC_LIB "-F${CMAKE_CURRENT_SOURCE_DIR}/fabric-ios -framework Fabric -framework Crashlytics" PARENT_SCOPE)
endif()

# crittercism
if(LFL_CRITTERCISM)
  set(CRITTERCISM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/crittercism/iOS/Crittercism.framework/Headers PARENT_SCOPE)
  set(CRITTERCISM_LIB "-F${CMAKE_CURRENT_SOURCE_DIR}/crittercism/iOS -framework Crittercism" PARENT_SCOPE)
endif()

# HockeyApp
if(LFL_HOCKEYAPP)
  if(LFL_HOCKEYAPP_CRASHREPORTING_ONLY)
    set(HOCKEYAPP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/HockeyApp/HockeySDK-iOS/HockeySDKCrashOnly/HockeySDK.framework/Headers PARENT_SCOPE)
    set(HOCKEYAPP_LIB "-F${CMAKE_CURRENT_SOURCE_DIR}/HockeyApp/HockeySDK-iOS/HockeySDKCrashOnly -framework HockeySDK" PARENT_SCOPE)
  else()
    set(HOCKEYAPP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/HockeyApp/HockeySDK-iOS/HockeySDK.embeddedframework/HockeySDK.framework/Headers PARENT_SCOPE)
    set(HOCKEYAPP_LIB "-F${CMAKE_CURRENT_SOURCE_DIR}/HockeyApp/HockeySDK-iOS/HockeySDK.embeddedframework -framework HockeySDK" PARENT_SCOPE)
  endif()
endif()

# AdMob
if(LFL_IOS)
  set(ADMOB_OPTIONS -F${CMAKE_CURRENT_SOURCE_DIR}/AdMob-ios PARENT_SCOPE)
  set(ADMOB_LIBS -F${CMAKE_CURRENT_SOURCE_DIR}/AdMob-ios PARENT_SCOPE)
endif()

# openssl
if(LFL_OPENSSL)
  if(LFL_WINDOWS)
    set(OPENSSL_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/openssl/include)
    set(OPENSSL_LIB ${CMAKE_CURRENT_BINARY_DIR}/openssl/lib/ssleay32.lib ${CMAKE_CURRENT_BINARY_DIR}/openssl/lib/libeay32.lib)
  elseif(LFL_ANDROID)
    set(OPENSSL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/openssl_prebuilt_for_android/include)
    set(OPENSSL_CRYPTO_LIB ${CMAKE_CURRENT_SOURCE_DIR}/openssl_prebuilt_for_android/armeabi-v7a/libcrypto.a)
    set(OPENSSL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/openssl_prebuilt_for_android/armeabi-v7a/libssl.a ${OPENSSL_CRYPTO_LIB})
  elseif(LFL_IOS)
    set(OPENSSL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/include)
    set(OPENSSL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/lib/libssl.a ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/lib/libcrypto.a)
  elseif(LFL_APPLE)
    set(OPENSSL_CONFIGURE_ENV ${CONFIGURE_ENV} PATH=$ENV{PATH}:/opt/X11/bin)
    ExternalProject_Add(openssl PREFIX openssl LOG_CONFIGURE ON LOG_BUILD ON BUILD_IN_SOURCE ON
                        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${OPENSSL_CONFIGURE_ENV}
                        <SOURCE_DIR>/Configure darwin64-x86_64-cc shared enable-ec_nistp_64_gcc_128 no-ssl2 no-ssl3 no-comp $ENV{CFLAGS} --prefix=<INSTALL_DIR>
                        COMMAND ${CMAKE_COMMAND} -E env ${OPENSSL_CONFIGURE_ENV} make depend
                        DOWNLOAD_COMMAND rm -rf openssl
                        COMMAND cp -R ${CMAKE_CURRENT_SOURCE_DIR}/openssl openssl
                        INSTALL_DIR openssl/install)
    ExternalProject_Get_Property(openssl INSTALL_DIR)

    add_library(libssl IMPORTED SHARED GLOBAL)
    add_dependencies(libssl openssl)
    set_property(TARGET libssl PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/lib/libssl.a)

    add_library(libcrypto IMPORTED SHARED GLOBAL)
    add_dependencies(libcrypto openssl)
    set_property(TARGET libcrypto PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/lib/libcrypto.a)

    set(OPENSSL_INCLUDE ${INSTALL_DIR}/include)
    set(OPENSSL_LIB libssl libcrypto)
  else()
    set(OPENSSL_INCLUDE)
    set(OPENSSL_LIB -lssl -lcrypto)
    set(OPENSSL_CRYPTO_LIB -lcrypto)
  endif()
  set(OPENSSL_INCLUDE ${OPENSSL_INCLUDE} PARENT_SCOPE)
  set(OPENSSL_LIB ${OPENSSL_LIB} PARENT_SCOPE)
endif()

# ref10_extract
if(LFL_REF10EXTRACT)
  add_subdirectory(ref10_extract)
  set(REF10EXTRACT_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ref10_extract
      ${CMAKE_CURRENT_SOURCE_DIR}/ref10_extract/ed25519/nacl_includes PARENT_SCOPE)
  set(REF10EXTRACT_LIB libcurve25519 libed25519 PARENT_SCOPE)
endif()

# mosh
if(LFL_MOSH)
  set(CONFIG_OPTS ${CONFIGURE_OPTIONS})
  set(CONFIG_ENV ${CONFIGURE_ENV} PROTOC=${PROTOBUF_PROTOC_EXECUTABLE} protobuf_CFLAGS=-I${PROTOBUF_INCLUDE_DIR})
  if(LFL_IOS OR LFL_APPLE)
    set(CONFIG_OPTS --with-crypto-library=apple-common-crypto)
  endif()
  ExternalProject_Add(mosh PREFIX mosh LOG_CONFIGURE ON LOG_BUILD ON BUILD_IN_SOURCE ON
                      CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh
                      COMMAND ${CMAKE_COMMAND} -E env ${CONFIG_ENV} protobuf_LIBS=$<TARGET_FILE:libprotobuf>
                      <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> ${CONFIG_OPTS}
                      DOWNLOAD_COMMAND rm -rf mosh COMMAND cp -R ${CMAKE_CURRENT_SOURCE_DIR}/mosh mosh)
  add_library(libmosh IMPORTED STATIC GLOBAL)
  add_dependencies(libmosh mosh)
  add_dependencies(mosh libprotobuf)
  set_property(TARGET libmosh PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ogg/lib/libogg.a)
endif()

# box2d
if(LFL_BOX2D)
  option(BOX2D_BUILD_EXAMPLES OFF)
  add_subdirectory(Box2D/Box2D)
  set(BOX2D_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/Box2D/Box2D PARENT_SCOPE)
  set(BOX2D_LIB Box2D PARENT_SCOPE)
endif()

# sqlite
if(LFL_SQLITE)
  if(LFL_IOS)
    set(SQLITE_INCLUDE PARENT_SCOPE)
    set(SQLITE_LIB -lsqlite3 PARENT_SCOPE)
  else()
    set(BUILD_SHELL ON)
    add_subdirectory(sqlite-amalgamation)
    set(SQLITE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/sqlite-amalgamation PARENT_SCOPE)
    set(SQLITE_LIB sqlite3 PARENT_SCOPE)
  endif()
endif()

# sqlcipher
if(LFL_SQLCIPHER)
  set(CONFIG_ENV ${CONFIGURE_ENV})
  list_append_kv(CONFIG_ENV CFLAGS "-DSQLITE_HAS_CODEC -DSQLITE_ENABLE_FTS3")
  set(CONFIG_OPTS ${CONFIGURE_OPTIONS} --enable-load-extension --enable-tempstore=yes --disable-tcl
      --disable-readline --disable-editline)
  if(LFL_IOS OR LFL_OSX)
    set(CONFIG_OPTS ${CONFIG_OPTS} --with-crypto-lib=commoncrypto)
    if(LFL_IOS_SIM)
      set(CONFIG_OPTS ${CONFIG_OPTS} --host=i386)
      list_append_kv(CONFIG_ENV LDFLAGS "/System/Library/Frameworks/Security.framework/Versions/Current/Security /System/Library/Frameworks/CoreFoundation.framework/Versions/Current/CoreFoundation")
    endif()
  else()
    if(OPENSSL_INCLUDE)
      list_append_kv(CONFIG_ENV CFLAGS "-I${OPENSSL_INCLUDE}")
    endif()
    list_append_kv(CONFIG_ENV LIBS "${OPENSSL_CRYPTO_LIB}")
  endif()

  ExternalProject_Add(sqlcipher LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/sqlcipher
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sqlcipher
                      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${CONFIG_ENV}
                      <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> ${CONFIG_OPTS}
                      BUILD_COMMAND env -i ${CMAKE_COMMAND} -E env HOME=$ENV{HOME} PATH=$ENV{PATH} ${CONFIG_ENV}
                      make libsqlcipher.la)
  add_library(libsqlcipher IMPORTED SHARED GLOBAL)
  add_dependencies(libsqlcipher sqlcipher)
  set_property(TARGET libsqlcipher PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/sqlcipher/lib/libsqlcipher.a)
  set(SQLCIPHER_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/sqlcipher/include PARENT_SCOPE)
  set(SQLCIPHER_LIB libsqlcipher PARENT_SCOPE)
  set(SQLCIPHER_DEF -DSQLITE_HAS_CODEC -DSQLITE_TEMP_STORE=2 PARENT_SCOPE)
endif()

# assimp
if(LFL_ASSIMP)
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory(assimp)
  set(ASSIMP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include PARENT_SCOPE)
  set(ASSIMP_LIB assimp PARENT_SCOPE)
endif()
    
# lame
if(LFL_LAME AND NOT LFL_WINDOWS)
  ExternalProject_Add(lame LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/lame
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lame
                      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>)
  add_library(libmp3lame IMPORTED SHARED GLOBAL)
  add_dependencies(libmp3lame lame)
  get_shared_library_name(SHARED_LIBRARY_NAME libmp3lame 0)
  set_property(TARGET libmp3lame PROPERTY IMPORTED_LOCATION
               ${CMAKE_CURRENT_BINARY_DIR}/lame/lib/${SHARED_LIBRARY_NAME})
  set(LAME_LIB libmp3lame)
  set(LAME_LIB ${LAME_LIB} PARENT_SCOPE)
endif()

# x264
if(LFL_X264 AND NOT LFL_WINDOWS)
  set(X264_CONFIGURE_ENV ${CONFIGURE_ENV})
  if(LFL_XCODE)
    set(X264_CONFIGURE_ENV ${X264_CONFIGURE_ENV} PATH=$ENV{PATH}:/opt/local/bin)
  endif()
  ExternalProject_Add(x264 LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/x264
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/x264
                      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${X264_CONFIGURE_ENV}
                      <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --extra-cflags=-w --enable-shared
                      BUILD_COMMAND ${CMAKE_COMMAND} -E env ${X264_CONFIGURE_ENV} make)
  add_library(libx264 IMPORTED SHARED GLOBAL)
  add_dependencies(libx264 x264)
  get_shared_library_name(SHARED_LIBRARY_NAME libx264 138)
  set_property(TARGET libx264 PROPERTY IMPORTED_LOCATION
               ${CMAKE_CURRENT_BINARY_DIR}/x264/lib/${SHARED_LIBRARY_NAME})
  set(X264_LIB libx264)
  set(X264_LIB ${X264_LIB} PARENT_SCOPE)
endif()

# ffmpeg
if(LFL_FFMPEG)
  if(WIN32)
    set(FFMPEG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-for-Windows-prebuilt/include PARENT_SCOPE)
    set(FFMPEG_LIB ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-for-Windows-prebuilt/lib/avformat.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-for-Windows-prebuilt/lib/avcodec.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-for-Windows-prebuilt/lib/avutil.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-for-Windows-prebuilt/lib/swscale.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-for-Windows-prebuilt/lib/swresample.lib
        ${LAME_LIB} ${X264_LIB} PARENT_SCOPE)
  else()
    if(LFL_ANDROID)
      set(CONFIGURE_OPTIONS --enable-cross-compile --enable-pic --disable-debug --disable-stripping
          --disable-bzlib --cross-prefix=${ANDROIDROOT}/bin/arm-linux-androideabi-
          --sysroot=${ANDROIDROOT}/sysroot --arch=arm5te --target-os=linux --disable-avdevice
          --enable-shared)
    else()
      set(CONFIGURE_OPTIONS --enable-gpl --enable-libx264 --enable-libmp3lame --disable-openssl --disable-securetransport
          --extra-libs=-L${CMAKE_CURRENT_BINARY_DIR}/lame/lib
          --extra-libs=-L${CMAKE_CURRENT_BINARY_DIR}/x264/lib
          --extra-cflags=-I${CMAKE_CURRENT_BINARY_DIR}/lame/include
          --extra-cflags=-I${CMAKE_CURRENT_BINARY_DIR}/x264/include --disable-sse --disable-outdev=sdl)
    endif()

    set(FFMPEG_CONFIGURE_ENV ${CONFIGURE_ENV})
    if(LFL_OSX)
      set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --disable-decoder=vp9 --disable-iconv)
      set(FFMPEG_CONFIGURE_ENV ${FFMPEG_CONFIGURE_ENV} PATH=/opt/local/bin:$ENV{PATH})
    endif()

    ExternalProject_Add(ffmpeg LOG_CONFIGURE ON LOG_BUILD ON
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg
                        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${FFMPEG_CONFIGURE_ENV}
                        <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> ${CONFIGURE_OPTIONS}
                        BUILD_COMMAND ${CMAKE_COMMAND} -E env ${FFMPEG_CONFIGURE_ENV} make)
    add_dependencies(ffmpeg ${LAME_LIB} ${X264_LIB})
    add_library(libavdevice IMPORTED STATIC GLOBAL)
    add_library(libavfilter IMPORTED STATIC GLOBAL)
    add_library(libavformat IMPORTED STATIC GLOBAL)
    add_library(libavcodec IMPORTED STATIC GLOBAL)
    add_library(libavutil IMPORTED STATIC GLOBAL)
    add_library(libswscale IMPORTED STATIC GLOBAL)
    add_library(libswresample IMPORTED STATIC GLOBAL)
    add_library(libpostproc IMPORTED STATIC GLOBAL)
    add_dependencies(libavdevice ffmpeg)
    add_dependencies(libavfilter ffmpeg)
    add_dependencies(libavformat ffmpeg)
    add_dependencies(libavcodec ffmpeg)
    add_dependencies(libavutil ffmpeg)
    add_dependencies(libswscale ffmpeg)
    add_dependencies(libswresample ffmpeg)
    add_dependencies(libpostproc ffmpeg)
    set_property(TARGET libavdevice PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libavdevice.a)
    set_property(TARGET libavfilter PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libavfilter.a)
    set_property(TARGET libavformat PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libavformat.a)
    set_property(TARGET libavcodec PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libavcodec.a)
    set_property(TARGET libavutil PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libavutil.a)
    set_property(TARGET libswscale PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libswscale.a)
    set_property(TARGET libswresample PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libswresample.a)
    set_property(TARGET libpostproc PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/lib/libpostproc.a)

    set(FFMPEG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/include PARENT_SCOPE)
    set(FFMPEG_LIB libavdevice libavfilter libavformat libavcodec libavutil libswscale libswresample
        libpostproc ${LAME_LIB} ${X264_LIB})
    if(LFL_APPLE)
      set(FFMPEG_LIB ${FFMPEG_LIB} "-framework AudioToolbox -framework VideoToolbox -framework CoreMedia -framework VideoDecodeAcceleration")
    endif()
        
    set(FFMPEG_LIB ${FFMPEG_LIB} PARENT_SCOPE)
    add_shared_library(FFMPEG_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/lame/lib/libmp3lame 0)
    add_shared_library(FFMPEG_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/x264/lib/libx264 138)
    set(FFMPEG_LIB_FILES ${FFMPEG_LIB_FILES} PARENT_SCOPE)
  endif()
endif()

# OGG
if(LFL_OGG)
  ExternalProject_Add(ogg LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ogg
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ogg
                      CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh --prefix=<INSTALL_DIR>)
  add_library(libogg IMPORTED STATIC GLOBAL)
  add_dependencies(libogg ogg)
  set_property(TARGET libogg PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ogg/lib/libogg.a)
  set(OGG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/ogg/include PARENT_SCOPE)
  set(OGG_LIB libogg PARENT_SCOPE)
endif()

# vorbis
if(LFL_VORBIS)
  set(OGG_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/ogg/include)
  set(OGG_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/ogg/lib/libogg.a)
  add_subdirectory(vorbis)
  set(VORBIS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/vorbis/include PARENT_SCOPE)
  set(VORBIS_LIB vorbis PARENT_SCOPE)
endif()

# CUDA
if(LFL_CUDA)
  INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../CMake/cuda/FindCUDA.cmake)
  set(CUDA_INCLUDE ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
  set(CUDA_LIB ${CUDA_LIBRARIES} PARENT_SCOPE)
endif()

# OpenGL
if(LFL_ANDROID)
  set(OPENGL_LIB "-lGLESv2 -lGLESv1_CM" PARENT_SCOPE)
elseif(LFL_IOS)
elseif(LFL_EMSCRIPTEN)
  INCLUDE(${EMSCRIPTEN_ROOT_PATH}/cmake/Modules/FindOpenGL.cmake)
  set(OPENGL_INCLUDE ${OPENGL_INCLUDE_DIR} PARENT_SCOPE)
  set(OPENGL_LIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} PARENT_SCOPE)
else()
  INCLUDE(${CMAKE_ROOT}/Modules/FindOpenGL.cmake)
  set(OPENGL_INCLUDE ${OPENGL_INCLUDE_DIR} PARENT_SCOPE)
  set(OPENGL_LIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} PARENT_SCOPE)
endif()

# GLEW
if(LFL_GLEW)
  add_subdirectory(glew)
  set(GLEW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glew/include PARENT_SCOPE)
  set(GLEW_LIB glew PARENT_SCOPE)
endif()

# wxWidgets
# ./configure --disable-shared --enable-unicode --with-cocoa --with-macosx-version-min=10.7 --with-macosx-sdk=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk CXXFLAGS="-std=c++0x -stdlib=libc++" CPPFLAGS="-stdlib=libc++" LIBS=-lc++
if(LFL_WXWIDGETS)
  set(ENV{wxWidgets_ROOT_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/wxWidgets")
  set(wxWidgets_CONFIG_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/wxWidgets/wx-config)
  find_package(wxWidgets COMPONENTS core base gl REQUIRED)
  include(${wxWidgets_USE_FILE})
  set(WXWIDGETS_DEF "")
  foreach(d ${wxWidgets_DEFINITIONS})
    list(APPEND WXWIDGETS_DEF "-D${d}")
  endforeach()
  set(WXWIDGETS_DEF "${WXWIDGETS_DEF}" PARENT_SCOPE)
  set(WXWIDGETS_INCLUDE "${wxWidgets_INCLUDE_DIRS}" PARENT_SCOPE)
  set(WXWIDGETS_LIB "${wxWidgets_LIBRARIES}" PARENT_SCOPE)
endif()

# GLFW
if(LFL_GLFW)
  add_subdirectory(glfw)
  set(GLFW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include PARENT_SCOPE)
  set(GLFW_LIB glfw "-framework IOKit" PARENT_SCOPE)
endif()

# SDL
if(LFL_SDL AND NOT LFL_EMSCRIPTEN)
  add_subdirectory(SDL)
  set(SDL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/SDL/include PARENT_SCOPE)
  set(SDL_LIB SDL2-static "-framework Carbon -framework IOKit -framework forcefeedback -framework CoreAudio -framework AudioUnit" PARENT_SCOPE)
endif()

# portaudio
if(LFL_PORTAUDIO)
  if(WIN32)
    INCLUDE_EXTERNAL_MSPROJECT(portaudio ${CMAKE_CURRENT_SOURCE_DIR}/portaudio/build/msvc/portaudio.vcxproj)
    set(PORTAUDIO_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/portaudio/include PARENT_SCOPE)
    set(PORTAUDIO_LIB portaudio_x86.lib PARENT_SCOPE)
  else()
    ExternalProject_Add(portaudio LOG_CONFIGURE ON LOG_BUILD ON
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/portaudio
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/portaudio
                        CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --disable-mac-universal)
    add_library(libportaudio IMPORTED SHARED GLOBAL)
    add_dependencies(libportaudio portaudio)
    get_shared_library_name(SHARED_LIBRARY_NAME libportaudio 2)
    set_property(TARGET libportaudio PROPERTY IMPORTED_LOCATION
                 ${CMAKE_CURRENT_BINARY_DIR}/portaudio/lib/${SHARED_LIBRARY_NAME})

    set(PORTAUDIO_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/portaudio/include PARENT_SCOPE)
    set(PORTAUDIO_LIB libportaudio PARENT_SCOPE)
    set(PORTAUDIO_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/portaudio/lib/${SHARED_LIBRARY_NAME} PARENT_SCOPE)
  endif()
endif()

# OpenCV
if(LFL_OPENCV)
  ExternalProject_Add(OpenCV SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/OpenCV
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/OpenCV
                      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DWITH_IPP=OFF -DBUILD_opencv_videoio=OFF)

  add_library(libopencv_core IMPORTED SHARED GLOBAL)
  add_library(libopencv_imgproc IMPORTED SHARED GLOBAL)
  add_dependencies(libopencv_core OpenCV)
  add_dependencies(libopencv_imgproc OpenCV)
  get_shared_library_name(SHARED_LIBRARY_NAME libopencv_core 3.1)
  set_property(TARGET libopencv_core PROPERTY IMPORTED_LOCATION
               ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib/${SHARED_LIBRARY_NAME})
  get_shared_library_name(SHARED_LIBRARY_NAME libopencv_imgproc 3.1)
  set_property(TARGET libopencv_imgproc PROPERTY IMPORTED_LOCATION
               ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib/${SHARED_LIBRARY_NAME})

  set(OPENCV_LIB libopencv_core libopencv_imgproc PARENT_SCOPE)
  set(OPENCV_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/include PARENT_SCOPE)
  add_shared_library(OPENCV_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib/libopencv_core 3.1)
  add_shared_library(OPENCV_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib/libopencv_imgproc 3.1)
  set(OPENCV_LIB_FILES ${OPENCV_LIB_FILES} PARENT_SCOPE)
endif()

# libarchive
if(LFL_LIBARCHIVE)
  set(ENABLE_ICONV FALSE)
  set(ENABLE_CAT FALSE)
  set(ENABLE_TAR FALSE)
  set(ENABLE_CPIO FALSE)
  add_subdirectory(libarchive)
  set(ARCHIVE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libarchive PARENT_SCOPE)
  if(WIN32)
    set(ARCHIVE_LIB libarchive.lib PARENT_SCOPE)
  else()
    set(ARCHIVE_LIB archive_static PARENT_SCOPE)
  endif()
endif()

# re2
if(LFL_RE2)
  ExternalProject_Add(re2 LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/re2
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/re2
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/re2
                      INSTALL_COMMAND "")
  add_library(libre2 IMPORTED STATIC GLOBAL)
  add_dependencies(libre2 re2)
  set_property(TARGET libre2 PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/re2/src/re2-build/libre2.a)
  set(RE2_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/re2 PARENT_SCOPE)
  set(RE2_LIB libre2 PARENT_SCOPE)
endif()

# sregex
if(LFL_SREGEX)
  add_subdirectory(sregex)
  set(SREGEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/sregex/src/sregex PARENT_SCOPE)
  set(SREGEX_LIB sregex PARENT_SCOPE)
endif()

# judy
if(LFL_JUDY)
  add_subdirectory(judy)
  set(JUDY_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/judy/src PARENT_SCOPE)
  set(JUDY_LIB ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyL/.libs/libJudyL.a
      ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyL/.libs/libprev.a
      ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyL/.libs/libnext.a
      ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudySL/.libs/libJudySL.a
      ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyCommon/.libs/libJudyMalloc.a PARENT_SCOPE)
endif()

# libclang
if(LFL_LIBCLANG)
  get_unversioned_shared_library_name(CLANG_LIBRARY ${LLVM_DIR}/lib/libclang)
  if(LFL_APPLE)
    set(LLVM_SYSTEM_LIBRARIES -lncurses)
  endif()
  set(LIBCLANG_INCLUDE ${LLVM_DIR}/include PARENT_SCOPE)
  set(LIBCLANG_LIB ${CLANG_LIBRARY} ${LLVM_DIR}/lib/libLLVMX86Disassembler.a
      ${LLVM_DIR}/lib/libLLVMX86AsmParser.a ${LLVM_DIR}/lib/libLLVMX86CodeGen.a
      ${LLVM_DIR}/lib/libLLVMSelectionDAG.a ${LLVM_DIR}/lib/libLLVMAsmPrinter.a
      ${LLVM_DIR}/lib/libLLVMX86Desc.a ${LLVM_DIR}/lib/libLLVMMCDisassembler.a
      ${LLVM_DIR}/lib/libLLVMX86Info.a ${LLVM_DIR}/lib/libLLVMX86AsmPrinter.a
      ${LLVM_DIR}/lib/libLLVMX86Utils.a ${LLVM_DIR}/lib/libLLVMInterpreter.a
      ${LLVM_DIR}/lib/libLLVMCodeGen.a ${LLVM_DIR}/lib/libLLVMScalarOpts.a
      ${LLVM_DIR}/lib/libLLVMInstCombine.a ${LLVM_DIR}/lib/libLLVMInstrumentation.a
      ${LLVM_DIR}/lib/libLLVMProfileData.a ${LLVM_DIR}/lib/libLLVMTransformUtils.a
      ${LLVM_DIR}/lib/libLLVMBitWriter.a ${LLVM_DIR}/lib/libLLVMExecutionEngine.a
      ${LLVM_DIR}/lib/libLLVMTarget.a ${LLVM_DIR}/lib/libLLVMAnalysis.a
      ${LLVM_DIR}/lib/libLLVMRuntimeDyld.a ${LLVM_DIR}/lib/libLLVMObject.a
      ${LLVM_DIR}/lib/libLLVMMCParser.a ${LLVM_DIR}/lib/libLLVMBitReader.a
      ${LLVM_DIR}/lib/libLLVMMC.a ${LLVM_DIR}/lib/libLLVMMCJIT.a
      ${LLVM_DIR}/lib/libLLVMCore.a ${LLVM_DIR}/lib/libLLVMSupport.a
      ${LLVM_SYSTEM_LIBRARIES} -rdynamic PARENT_SCOPE)
endif()

# libcling
if(LFL_LIBCLING)
  set(LIBCLING_INCLUDE ${CLING_LLVM_DIR}/include PARENT_SCOPE)
  set(LIBCLING_LIB ${CLING_LLVM_DIR}/lib/libclingMetaProcessor.a
      ${CLING_LLVM_DIR}/lib/libclingInterpreter.a
      ${CLING_LLVM_DIR}/lib/libclingUtils.a ${CLING_LLVM_DIR}/lib/libclangFrontend.a
      ${CLING_LLVM_DIR}/lib/libclangSerialization.a ${CLING_LLVM_DIR}/lib/libclangDriver.a
      ${CLING_LLVM_DIR}/lib/libclangCodeGen.a ${CLING_LLVM_DIR}/lib/libclangParse.a
      ${CLING_LLVM_DIR}/lib/libclangSema.a ${CLING_LLVM_DIR}/lib/libclangEdit.a
      ${CLING_LLVM_DIR}/lib/libclangAnalysis.a ${CLING_LLVM_DIR}/lib/libclangAST.a
      ${CLING_LLVM_DIR}/lib/libclangLex.a ${CLING_LLVM_DIR}/lib/libclangBasic.a
      ${CLING_LLVM_DIR}/lib/libLLVMLTO.a ${CLING_LLVM_DIR}/lib/libLLVMObjCARCOpts.a
      ${CLING_LLVM_DIR}/lib/libLLVMLinker.a ${CLING_LLVM_DIR}/lib/libLLVMipo.a
      ${CLING_LLVM_DIR}/lib/libLLVMVectorize.a ${CLING_LLVM_DIR}/lib/libLLVMBitWriter.a
      ${CLING_LLVM_DIR}/lib/libLLVMTableGen.a ${CLING_LLVM_DIR}/lib/libLLVMDebugInfo.a
      ${CLING_LLVM_DIR}/lib/libLLVMOption.a ${CLING_LLVM_DIR}/lib/libLLVMX86Disassembler.a
      ${CLING_LLVM_DIR}/lib/libLLVMX86AsmParser.a ${CLING_LLVM_DIR}/lib/libLLVMX86CodeGen.a
      ${CLING_LLVM_DIR}/lib/libLLVMSelectionDAG.a ${CLING_LLVM_DIR}/lib/libLLVMAsmPrinter.a
      ${CLING_LLVM_DIR}/lib/libLLVMX86Desc.a ${CLING_LLVM_DIR}/lib/libLLVMMCDisassembler.a
      ${CLING_LLVM_DIR}/lib/libLLVMX86Info.a ${CLING_LLVM_DIR}/lib/libLLVMX86AsmPrinter.a
      ${CLING_LLVM_DIR}/lib/libLLVMX86Utils.a ${CLING_LLVM_DIR}/lib/libLLVMJIT.a
      ${CLING_LLVM_DIR}/lib/libLLVMIRReader.a ${CLING_LLVM_DIR}/lib/libLLVMAsmParser.a
      ${CLING_LLVM_DIR}/lib/libLLVMLineEditor.a ${CLING_LLVM_DIR}/lib/libLLVMMCAnalysis.a
      ${CLING_LLVM_DIR}/lib/libLLVMInstrumentation.a ${CLING_LLVM_DIR}/lib/libLLVMInterpreter.a
      ${CLING_LLVM_DIR}/lib/libLLVMCodeGen.a ${CLING_LLVM_DIR}/lib/libLLVMScalarOpts.a
      ${CLING_LLVM_DIR}/lib/libLLVMInstCombine.a ${CLING_LLVM_DIR}/lib/libLLVMTransformUtils.a
      ${CLING_LLVM_DIR}/lib/libLLVMipa.a ${CLING_LLVM_DIR}/lib/libLLVMAnalysis.a
      ${CLING_LLVM_DIR}/lib/libLLVMProfileData.a ${CLING_LLVM_DIR}/lib/libLLVMMCJIT.a
      ${CLING_LLVM_DIR}/lib/libLLVMTarget.a ${CLING_LLVM_DIR}/lib/libLLVMRuntimeDyld.a
      ${CLING_LLVM_DIR}/lib/libLLVMObject.a ${CLING_LLVM_DIR}/lib/libLLVMMCParser.a
      ${CLING_LLVM_DIR}/lib/libLLVMBitReader.a ${CLING_LLVM_DIR}/lib/libLLVMExecutionEngine.a
      ${CLING_LLVM_DIR}/lib/libLLVMMC.a ${CLING_LLVM_DIR}/lib/libLLVMCore.a
      ${CLING_LLVM_DIR}/lib/libLLVMSupport.a -lncurses PARENT_SCOPE)
endif()

# bullet
if(LFL_BULLET)
  add_subdirectory(bullet)
  set(BULLET_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/bullet/src PARENT_SCOPE)
  set(BULLET_LIB BulletDynamics BulletCollision LinearMath PARENT_SCOPE)
endif()

# open dynamics engine
if(LFL_ODE)
  add_subdirectory(ode)
  set(ODE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ode/include PARENT_SCOPE)
  set(ODE_LIB ${CMAKE_CURRENT_BINARY_DIR}/ode/ode/src/.libs/libode.a PARENT_SCOPE)
endif()

# berkelium
if(LFL_BERKELIUM)
  set(BERKELIUM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/berkelium/osx/include PARENT_SCOPE)
  set(BERKELIUM_LIB ${CMAKE_CURRENT_BINARY_DIR}/berkelium/osx/lib/liblibberkelium.dylib PARENT_SCOPE)
endif()

# lua
if(LFL_LUA)
  add_subdirectory(lua)
  set(LUA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/lua/src PARENT_SCOPE)
  set(LUA_LIB ${CMAKE_CURRENT_BINARY_DIR}/lua/src/liblua.a PARENT_SCOPE)
endif()

# v8 js
if(LFL_V8JS)
  set(V8JS_INCLUDE ${V8_DIR}/include PARENT_SCOPE)
  set(V8JS_LIB ${V8_DIR}/native/libv8_libplatform.a ${V8_DIR}/native/libv8_libbase.a
      ${V8_DIR}/native/libv8_base.a ${V8_DIR}/native/libv8_snapshot.a ${V8_DIR}/native/libicuuc.a
      ${V8_DIR}/native/libicui18n.a ${V8_DIR}/native/libicudata.a PARENT_SCOPE)
endif()

# tinyjs
if(LFL_TINYJS)
  set(TINYJS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/TinyJS/TinyJS.o PARENT_SCOPE)
endif()

# jsoncpp
if(LFL_JSONCPP)
  ExternalProject_Add(jsoncpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/jsoncpp)
  add_library(libjsoncpp IMPORTED STATIC GLOBAL)
  add_dependencies(libjsoncpp jsoncpp)
  set_property(TARGET libjsoncpp PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp/lib/libjsoncpp.a)
  set(JSONCPP_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp/include PARENT_SCOPE)
  set(JSONCPP_LIB libjsoncpp PARENT_SCOPE)
endif()

# pcap
if(LFL_PCAP)
  ExternalProject_Add(pcap LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libppcap
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libpcap
                      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --disable-universal)
  add_library(libpcap IMPORTED STATIC GLOBAL)
  add_dependencies(libpcap pcap)
  set_property(TARGET libpcap PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libpcap/lib/libpcap.a)
  set(PCAP_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/libpcap/include PARENT_SCOPE)
  set(PCAP_LIB libpcap PARENT_SCOPE)
endif()

# GeoIP
if(LFL_GEOIP)
  if(WIN32)
    #nmake -f Makefile.vc
  else()
    ExternalProject_Add(geoip LOG_CONFIGURE ON LOG_BUILD ON
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/geoip-api-c
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/geoip-api-c
                        CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>)
    add_library(libGeoIP IMPORTED STATIC GLOBAL)
    add_dependencies(libGeoIP geoip)
    set_property(TARGET libGeoIP PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/geoip-aip-c/libGeoIP/lib/libGeoIP.a)
    set(GEOIP_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/geoip-api-c/libGeoIP/include PARENT_SCOPE)
    set(GEOIP_LIB libGeoIP PARENT_SCOPE)
  endif()
endif()

# libuv
if(LFL_LIBUV)
  ExternalProject_Add(uv LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libuv
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libuv
                      CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh --prefix=<INSTALL_DIR>
                      COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>)
  add_library(libuv IMPORTED STATIC GLOBAL)
  add_dependencies(libuv uv)
  set_property(TARGET libuv PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libuv/lib/libuv.a)
  set(LIBUV_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/libuv/include PARENT_SCOPE)
  set(LIBUV_LIB libuv PARENT_SCOPE)
endif()

# cmake-daemon
if(LFL_CMAKEDAEMON)
  ExternalProject_Add(cmake-daemon LOG_CONFIGURE ON LOG_BUILD ON
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cmake-daemon
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake-daemon
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/cmake-daemon
                      -DLIBUV_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/libuv/include
                      -DLIBUV_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/libuv/lib/libuv.a
                      -DDTL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/dtl)
  add_dependencies(cmake-daemon libuv)
  add_executable(cmakedaemon IMPORTED GLOBAL)
  add_dependencies(cmakedaemon cmake-daemon)
  set_property(TARGET cmakedaemon PROPERTY IMPORTED_LOCATION
               ${CMAKE_CURRENT_BINARY_DIR}/cmake-daemon/bin/cmake)
endif()

if(LFL_STOCKFISH)
  if(LFL64)
    set(BUILD_ARCH x86-64)
  else()
    set(BUILD_ARCH x86-32)
  endif()
  if(LFL_OSX OR LFL_IOS)
    set(BUILD_ARGS ARCH=${BUILD_ARCH} COMP=clang)
  else()
    set(BUILD_ARGS ARCH=${BUILD_ARCH} COMP=gcc)
  endif()
  ExternalProject_Add(Stockfish LOG_BUILD ON BUILD_IN_SOURCE ON CONFIGURE_COMMAND "" INSTALL_COMMAND ""
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/Stockfish
                      DOWNLOAD_COMMAND rm -rf Stockfish
                      COMMAND cp -R ${CMAKE_CURRENT_SOURCE_DIR}/Stockfish/src Stockfish
                      BUILD_COMMAND make build ${BUILD_ARGS})
  ExternalProject_Get_Property(Stockfish SOURCE_DIR)
  set(STOCKFISH_BIN ${SOURCE_DIR}/stockfish PARENT_SCOPE)
endif()
