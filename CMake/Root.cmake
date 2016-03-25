# $Id: lfapp.h 1335 2014-12-02 04:13:46Z justin $
# Copyright (C) 2009 Lucid Fusion Labs

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

CMAKE_POLICY(SET CMP0004 OLD)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(LFL_DEBUG 1)
endif()

if(LFL_IPHONESIM)
  set(LFL_IPHONE 1)
  set(IPHONESIM "-Simulator")
endif()

if(LFL_IPHONE OR LFL_ANDROID)
  set(LFL_MOBILE 1)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(LFL_APPLE 1)
endif()

if(LFL_EMSCRIPTEN)
  set(LFL_APP_OS app_null_os)
  set(LFL_APP_FRAMEWORK app_sdl_framework)
  set(LFL_APP_GRAPHICS app_opengl_graphics)
  set(LFL_APP_AUDIO app_null_audio)
  set(LFL_APP_CAMERA app_null_camera)
  set(LFL_APP_CONVERT app_null_convert)
  set(LFL_APP_CRYPTO app_null_crypto)
  set(LFL_APP_FONT app_null_ttf)
  set(LFL_APP_GAME_LOADER app_simple_resampler app_simple_loader app_libpng_png app_null_jpeg app_null_gif)

elseif(LFL_ANDROID)
  set(LFL_APP_OS app_android_os)
  set(LFL_APP_FRAMEWORK app_android_framework)
  set(LFL_APP_GRAPHICS app_opengl_graphics)
  set(LFL_APP_AUDIO app_android_audio)
  set(LFL_APP_CAMERA app_null_camera)
  set(LFL_APP_CONVERT app_null_convert)
  set(LFL_APP_CRYPTO app_openssl_crypto)
  set(LFL_APP_FONT app_null_ttf)
  set(LFL_APP_GAME_LOADER app_simple_resampler app_simple_loader app_libpng_png app_libjpeg_jpeg app_null_gif)

elseif(LFL_IPHONE)
  set(LFL_APP_OS app_iphone_os)
  set(LFL_APP_FRAMEWORK app_iphone_framework)
  set(LFL_APP_GRAPHICS app_opengl_graphics)
  set(LFL_APP_AUDIO app_iphone_audio)
  set(LFL_APP_CAMERA app_avcapture_camera)
  set(LFL_APP_CONVERT app_iconv_convert)
  set(LFL_APP_CRYPTO app_commoncrypto_crypto)
  set(LFL_APP_FONT app_null_ttf)
  set(LFL_APP_GAME_LOADER app_simple_resampler app_simple_loader app_libpng_png app_libjpeg_jpeg app_null_gif)

elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(LFL_OSX 1)
  set(LFL_APP_OS app_osx_os)
  set(LFL_APP_FRAMEWORK app_osx_framework)
  set(LFL_APP_GRAPHICS app_opengl_graphics)
  set(LFL_APP_AUDIO app_portaudio_audio)
  set(LFL_APP_CAMERA app_qtkit_camera)
  set(LFL_APP_CONVERT app_iconv_convert)
  set(LFL_APP_CRYPTO app_commoncrypto_crypto)
  set(LFL_APP_FONT app_freetype_ttf)
  set(LFL_APP_GAME_LOADER app_ffmpeg_resampler app_ffmpeg_loader app_libpng_png app_libjpeg_jpeg app_null_gif)

elseif(WIN32 OR WIN64)
  set(LFL_WINDOWS 1)
  set(LFL_APP_OS app_windows_os)
  set(LFL_APP_FRAMEWORK app_windows_framework)
  set(LFL_APP_GRAPHICS app_opengl_graphics)
  set(LFL_APP_AUDIO app_portaudio_audio)
  set(LFL_APP_CAMERA app_directshow_camera)
  set(LFL_APP_CONVERT app_null_convert)
  set(LFL_APP_CRYPTO app_openssl_crypto)
  set(LFL_APP_FONT app_freetype_ttf)
  set(LFL_APP_GAME_LOADER app_ffmpeg_resampler app_ffmpeg_loader app_libpng_png app_libjpeg_jpeg app_null_gif)

elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(LFL_LINUX 1)
  set(LFL_APP_OS app_linux_os)
  set(LFL_APP_FRAMEWORK app_x11_framework)
  set(LFL_APP_GRAPHICS app_opengl_graphics)
  set(LFL_APP_AUDIO app_portaudio_audio)
  set(LFL_APP_CAMERA app_ffmpeg_camera)
  set(LFL_APP_CONVERT app_iconv_convert)
  set(LFL_APP_CRYPTO app_openssl_crypto)
  set(LFL_APP_FONT app_freetype_ttf)
  set(LFL_APP_GAME_LOADER app_ffmpeg_resampler app_ffmpeg_loader app_libpng_png app_libjpeg_jpeg app_null_gif)

endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(LFL64 1)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(LFL32 1)
else()
  message(FATAL_ERROR "Pointer size ${CMAKE_SIZEOF_VOID_P}")
endif()

include(ExternalProject)
include(BundleUtilities)

list(APPEND CMAKE_MODULE_PATH ${LFL_SOURCE_DIR}/core/CMake)
include(PrecompiledHeader)
include(Autoconf)
include(util)

if(WIN32)
  link_directories("")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SAFESEH:NO")
  FOREACH(flag CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG_INIT
    CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_DEBUG  CMAKE_CXX_FLAGS_DEBUG_INIT)
    STRING(REPLACE "/MD"  "/MT"  "${flag}" "${${flag}}")
    STRING(REPLACE "/MDd" "/MTd" "${flag}" "${${flag}}")
    SET("${flag}" "${${flag}} /EHsc")
  ENDFOREACH()
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wno-deprecated-declarations") # -Wold-style-cast")
  if(LFL_USE_LIBCPP)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -stdlib=libc++")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -stdlib=libc++")
    #set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++")
    #set(CMAKE_SHARED_LINKER_FLAGS "-stdlib=libc++")
    #set(CMAKE_MODULE_LINKER_FLAGS "-stdlib=libc++")
  endif()
endif()

add_definitions(-D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS)

# imports
add_subdirectory(${LFL_SOURCE_DIR}/core/imports)

# macros
macro(lfl_project _name)
  project(${_name})
  set(LFL_PROJECT ${_name})
  set(LFL_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(LFL_PROJECT_BINDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro(lfl_project)

macro(lfl_add_library _name)
  add_library(${_name} ${ARGN})
  add_dependencies(${_name} lfl_app)
  target_use_precompiled_header(${_name} core/app/app.h FORCEINCLUDE)
endmacro()

macro(lfl_add_executable _name)
  add_executable(${_name} ${ARGN})
  add_dependencies(${_name} lfl_app)
  target_use_precompiled_header(${_name} core/app/app.h FORCEINCLUDE)
endmacro()

# post build macros
include(Package)

# proto macros
if(LFL_PROTOBUF)
  include(FindProtoBuf)
else()
  macro(PROTOBUF_GENERATE_CPP _s _h _f)
  endmacro()
endif()

if(LFL_FLATBUFFERS)
  set(FLATBUFFERS_INCLUDE_DIR ${LFL_SOURCE_DIR}/core/imports/flatbuffers/include)
  set(FLATBUFFERS_FLATC_EXECUTABLE ${LFL_CORE_BINARY_DIR}/imports/flatbuffers/flatc)
  include(${LFL_SOURCE_DIR}/core/imports/flatbuffers/CMake/FindFlatBuffers.cmake)
endif()

if(LFL_CAPNPROTO)
  set(CAPNP_LIB_KJ          ${LFL_CORE_BINARY_DIR}/imports/capnproto/lib/libkj.a)
  set(CAPNP_LIB_KJ-ASYNC    ${LFL_CORE_BINARY_DIR}/imports/capnproto/lib/libkj-async.a)
  set(CAPNP_LIB_CAPNP       ${LFL_CORE_BINARY_DIR}/imports/capnproto/lib/libcapnp.a)
  set(CAPNP_LIB_CAPNP-RPC   ${LFL_CORE_BINARY_DIR}/imports/capnproto/lib/libcapnp-rpc.a)
  set(CAPNP_EXECUTABLE      ${LFL_CORE_BINARY_DIR}/imports/capnproto/bin/capnp)
  set(CAPNPC_CXX_EXECUTABLE ${LFL_CORE_BINARY_DIR}/imports/capnproto/bin/capnpc-c++)
  set(CAPNP_INCLUDE_DIRS    ${LFL_CORE_BINARY_DIR}/imports/capnproto/include)
  include(FindCapnProto)
endif(LFL_CAPNPROTO)

# platform macros
macro(lfl_enable_qt)
  find_package(Qt5OpenGL REQUIRED)
  find_package(Qt5WebKit REQUIRED)
  find_package(Qt5WebKitWidgets REQUIRED)
  foreach(_current ${Qt5WebKitWidgets_COMPILE_DEFINITIONS})
    set(QT_DEF ${QT_DEF} "-D${_current}")
  endforeach()
  foreach(_current ${Qt5WebKit_COMPILE_DEFINITIONS})
    set(QT_DEF ${QT_DEF} "-D${_current}")
  endforeach()
  foreach(_current ${Qt5OpenGL_COMPILE_DEFINITIONS})
    set(QT_DEF ${QT_DEF} "-D${_current}")
  endforeach()
  set(QT_INCLUDE ${Qt5WebKitWidgets_INCLUDE_DIRS} ${Qt5WebKit_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})
  set(QT_LIB ${Qt5WebKitWidgets_LIBRARIES} ${Qt5WebKit_LIBRARIES} ${Qt5OpenGL_LIBRARIES})
endmacro()

# app
add_subdirectory(${LFL_SOURCE_DIR}/core/app)

# app unit tests
add_subdirectory(${LFL_SOURCE_DIR}/core/app_tests)

# web
add_subdirectory(${LFL_SOURCE_DIR}/core/web)

# game
add_subdirectory(${LFL_SOURCE_DIR}/core/game)

# nlp
add_subdirectory(${LFL_SOURCE_DIR}/core/nlp)

# speech
add_subdirectory(${LFL_SOURCE_DIR}/core/speech)
