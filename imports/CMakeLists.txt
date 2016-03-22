# $Id: CMakeLists.txt 1325 2014-10-29 06:08:31Z justin $

FOREACH(flag CMAKE_CXX_FLAGS_DEBUG)
  STRING(REPLACE "-Wold-style-cast" "" "${flag}" "${${flag}}")
ENDFOREACH()

# Windows
if(LFL_WINDOWS)
  INCLUDE_EXTERNAL_MSPROJECT(zlib ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/zlib/zlib.vcxproj)
  set(PLATFORM_LIB Ws2_32.lib Dnsapi.lib opengl32.lib glu32.lib strmiids.lib zlib.lib)
  set(PLATFORM_DEF "-DGLEW_STATIC")
endif()

# Mac
if(LFL_OSX)
  set(OPENGL_INCLUDE ${OPENGL_INCLUDE_DIR}/Headers ${CMAKE_CURRENT_SOURCE_DIR}/GL PARENT_SCOPE)
  set(PLATFORM_LIB "-framework Cocoa -framework CoreVideo -framework QTKit -lz -lbz2 -liconv")
  if(LFL_SDL)
    set(PLATFORM_LIB "${PLATFORM_LIB} -framework Carbon -framework IOKit -framework forcefeedback -framework CoreAudio -framework AudioUnit")
  endif()
  if(LFL_GLFW)
    set(PLATFORM_LIB "${PLATFORM_LIB} -framework IOKit")
  endif()
endif()

# Linux
if(LFL_LINUX)
  set(HAVE_SDLCONFIG 1)
  set(CUDA_LIB /usr/local/cuda/lib64/libcudart.so PARENT_SCOPE)
  set(PLATFORM_LIB "-lutil -lz -lpng -lpthread -ldl -lbz2")
endif()

# Android
if(LFL_ANDROID)
  set(PLATFORM_LIB "-llog -lz")
endif()

# iPhone 
if(LFL_IPHONE)
  set(PLATFORM_LIB "-liconv -lz -framework Foundation -framework CoreText -framework CoreAudio -framework CoreGraphics -framework OpenGLES -framework UIKit -framework QuartzCore -framework AVFoundation -framework GLKit -framework Security")
  if(NOT LFL_IPHONESIM)
    set(PLATFORM_LIB "${PLATFORM_LIB} -framework CoreVideo -framework CoreMedia")
  endif()
endif()

# export platform
set(PLATFORM_DEF ${PLATFORM_DEF} PARENT_SCOPE)
set(PLATFORM_LIB ${PLATFORM_LIB} PARENT_SCOPE)
set(PLATFORM_INCLUDE ${PLATFORM_INCLUDE} PARENT_SCOPE)

# boost
set(BOOST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/boost PARENT_SCOPE)

# png
if(LFL_PNG)
  add_subdirectory(libpng)
  set(PNG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/libpng ${CMAKE_CURRENT_SOURCE_DIR}/libpng PARENT_SCOPE)
  set(PNG_LIB png PARENT_SCOPE)
endif()

# jpeg
if(LFL_JPEG)
  if(WIN32)
    INCLUDE_EXTERNAL_MSPROJECT(libjpeg ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/libjpeg/libjpeg.vcxproj)
    set(JPEG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/include PARENT_SCOPE)
    set(JPEG_LIB libjpeg.lib PARENT_SCOPE)
    set(JPEG_DEP libjpeg PARENT_SCOPE)
  else()
    add_subdirectory(libjpeg-turbo)
    set(JPEG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo PARENT_SCOPE)
    set(JPEG_LIB ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/.libs/libturbojpeg.a PARENT_SCOPE)
  endif()
endif()

# gif
if(LFL_GIF)
  add_subdirectory(giflib)
  set(GIF_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/giflib/lib PARENT_SCOPE)
  set(GIF_LIB libgif PARENT_SCOPE)
endif()

# freetype
if(LFL_FREETYPE)
  add_subdirectory(freetype)
  set(FREETYPE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/freetype/include PARENT_SCOPE)
  if(WIN32)
    set(FREETYPE_LIB freetype239_D.lib PARENT_SCOPE)
  else()
    set(FREETYPE_LIB ${CMAKE_CURRENT_BINARY_DIR}/freetype/.libs/libfreetype.a PARENT_SCOPE)
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
  set(LIBCSS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libcss/libcss/include ${CMAKE_CURRENT_SOURCE_DIR}/libcss/libwapcaplet/include PARENT_SCOPE)
  set(LIBCSS_LIB libcss PARENT_SCOPE)
endif()

# protobuf
if(LFL_PROTOBUF)
  add_subdirectory(protobuf)
  if(WIN32)
    set(PROTOBUF_LIBRARY libprotobuf-lite.lib PARENT_SCOPE)
  else()
    set(PROTOBUF_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/.libs/libprotobuf.a PARENT_SCOPE)
  endif()
  set(PROTOBUF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src PARENT_SCOPE)
  set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protoc PARENT_SCOPE)
endif()

# flatbuffers
if(LFL_FLATBUFFERS)
  add_subdirectory(flatbuffers)
  set(FLATBUFFERS_LIB flatbuffers PARENT_SCOPE)
endif()

# capnproto
if(LFL_CAPNPROTO)
  ExternalProject_Add(capnproto SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/capnproto/c++
                                PREFIX ${CMAKE_CURRENT_BINARY_DIR}/capnproto
                                CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/capnproto)
endif()

# glog
if(LFL_GLOG)
  add_subdirectory(glog)
  set(GLOG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glog/src/glog PARENT_SCOPE)
  set(GLOG_LIB ${CMAKE_CURRENT_BINARY_DIR}/glog/.libs/libglog.a PARENT_SCOPE)
endif()

# gtest
if(LFL_GTEST)
  add_subdirectory(protobuf/gtest)
  set(GTEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/gtest/include PARENT_SCOPE)
  set(GTEST_LIB ${CMAKE_CURRENT_BINARY_DIR}/protobuf/gtest/lib/.libs/libgtest.a PARENT_SCOPE)
endif()

# tcmalloc
if(LFL_TCMALLOC)
  add_subdirectory(gperftools)
  set(TCMALLOC_LIB ${CMAKE_CURRENT_BINARY_DIR}/gperftools/.libs/libtcmalloc_minimal.a -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free PARENT_SCOPE)
endif()

# openssl
if(LFL_OPENSSL)
  if(LFL_WINDOWS)
    set(OPENSSL_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/openssl/include PARENT_SCOPE)
    set(OPENSSL_LIB ${CMAKE_CURRENT_BINARY_DIR}/openssl/lib/ssleay32.lib ${CMAKE_CURRENT_BINARY_DIR}/openssl/lib/libeay32.lib PARENT_SCOPE)
  elseif(LFL_ANDROID)
    set(OPENSSL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/OpenSSL-for-Android-Prebuilt/openssl-1.0.2/include PARENT_SCOPE)
    set(OPENSSL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/OpenSSL-for-Android-Prebuilt/openssl-1.0.2/armeabi-v7a/lib/libssl.a ${CMAKE_CURRENT_SOURCE_DIR}/OpenSSL-for-Android-Prebuilt/openssl-1.0.2/armeabi-v7a/lib/libcrypto.a PARENT_SCOPE)
  elseif(LFL_IPHONE)
    set(OPENSSL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/include PARENT_SCOPE)
    set(OPENSSL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/lib/libssl.a ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/lib/libcrypto.a PARENT_SCOPE)
  else()
    set(OPENSSL_LIB -lssl -lcrypto PARENT_SCOPE)
  endif()
endif()

# box2d
if(LFL_BOX2D)
  add_subdirectory(Box2D)
  set(BOX2D_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/Box2D PARENT_SCOPE)
  set(BOX2D_LIB Box2D PARENT_SCOPE)
endif()

# assimp
if(LFL_ASSIMP)
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory(assimp)
  set(ASSIMP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include PARENT_SCOPE)
  set(ASSIMP_LIB assimp PARENT_SCOPE)
endif()
    
# lame
if(LFL_LAME)    
  add_subdirectory(lame)
  set(LAME_LIB -L${CMAKE_CURRENT_BINARY_DIR}/lame/libmp3lame/.libs -lmp3lame)
  set(LAME_LIB ${LAME_LIB} PARENT_SCOPE)
endif()

# x264
if(LFL_X264)
  add_subdirectory(x264)
  set(X264_LIB -L${CMAKE_CURRENT_BINARY_DIR}/x264 -lx264)
  set(X264_LIB ${X264_LIB} PARENT_SCOPE)
endif()

# ffmpeg
if(LFL_FFMPEG)
  add_subdirectory(ffmpeg)
  if(WIN32)
    set(FFMPEG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/include PARENT_SCOPE)
    set(FFMPEG_LIB ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/avformat.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/avcodec.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/avutil.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/swscale.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/swresample.lib
        ${LAME_LIB} ${X264_LIB} PARENT_SCOPE)
  else()
    set(FFMPEG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg PARENT_SCOPE)
    set(FFMPEG_LIB ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavdevice/libavdevice.a
        ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavfilter/libavfilter.a
        ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavformat/libavformat.a
        ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavcodec/libavcodec.a
        ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavutil/libavutil.a
        ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libswscale/libswscale.a
        ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libswresample/libswresample.a
        ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libpostproc/libpostproc.a
        ${LAME_LIB} ${X264_LIB} PARENT_SCOPE)
    add_shared_library(FFMPEG_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/lame/libmp3lame/.libs/libmp3lame 0)
    add_shared_library(FFMPEG_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/x264/libx264 142)
    set(FFMPEG_LIB_FILES ${FFMPEG_LIB_FILES} PARENT_SCOPE)
  endif()
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
elseif(LFL_IPHONE)
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
  set(GLFW_LIB glfw PARENT_SCOPE)
endif()

# SDL
if(LFL_SDL)
  add_subdirectory(SDL)
  set(SDL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/SDL/include PARENT_SCOPE)
  set(SDL_LIB ${CMAKE_CURRENT_BINARY_DIR}/SDL/build/.libs/libSDL2.a PARENT_SCOPE)
endif()

# portaudio
if(LFL_PORTAUDIO)
  add_subdirectory(portaudio)
  set(PORTAUDIO_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/portaudio/include PARENT_SCOPE)
  if(WIN32)
    set(PORTAUDIO_LIB portaudio_x86.lib PARENT_SCOPE)
  else()
    set(PORTAUDIO_LIB -lportaudio -L${CMAKE_CURRENT_BINARY_DIR}/portaudio/lib/.libs PARENT_SCOPE)
    add_shared_library(PORTAUDIO_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/portaudio/lib/.libs/libportaudio 2)
    set(PORTAUDIO_LIB_FILES ${PORTAUDIO_LIB_FILES} PARENT_SCOPE)
  endif()
endif()

# OpenCV
if(LFL_OPENCV)
  set(OPENCV_DEP OpenCV PARENT_SCOPE)
  set(OPENCV_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/include PARENT_SCOPE)
  if(WIN32)
    INCLUDE_EXTERNAL_MSPROJECT(OpenCV ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/src/cv/cv.vcxproj)
    INCLUDE_EXTERNAL_MSPROJECT(cxcore ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/src/cxcore/cxcore.vcxproj)
    INCLUDE_EXTERNAL_MSPROJECT(flann ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/flann/flann.vcxproj)
    INCLUDE_EXTERNAL_MSPROJECT(opencv_lapack ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/lapack/opencv_lapack.vcxproj)
    set(OPENCV_LIB cv.lib cxcore.lib flannd.lib opencv_lapackd.lib PARENT_SCOPE)
  else()
    ExternalProject_Add(OpenCV SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV
                               PREFIX ${CMAKE_CURRENT_BINARY_DIR}/OpenCV
                               CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/OpenCV)
    set(OPENCV_LIB -L${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib -lcv -lcxcore PARENT_SCOPE)
    add_shared_library(OPENCV_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib/libcxcore 2.1)
    add_shared_library(OPENCV_LIB_FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib/libcv 2.1)
    set(OPENCV_LIB_FILES ${OPENCV_LIB_FILES} PARENT_SCOPE)
  endif()
endif()

# libarchive
if(LFL_LIBARCHIVE)
  add_subdirectory(libarchive)
  set(ARCHIVE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libarchive PARENT_SCOPE)
  if(WIN32)
    set(ARCHIVE_LIB libarchive.lib PARENT_SCOPE)
  else()
    set(ARCHIVE_LIB archive_static PARENT_SCOPE)
  endif()
endif()

# regex
if(LFL_REGEX)
  add_subdirectory(regex)
  set(REGEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/regex PARENT_SCOPE)
  set(REGEX_LIB regex PARENT_SCOPE)
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
  set(CLANG_LIB ${LLVM_DIR}/lib/libclang.dylib)
  set(LLVM_LIB ${LLVM_DIR}/lib/libLLVMAnalysis.a ${LLVM_DIR}/lib/libLLVMAsmParser.a
      ${LLVM_DIR}/lib/libLLVMAsmPrinter.a ${LLVM_DIR}/lib/libLLVMBitReader.a
      ${LLVM_DIR}/lib/libLLVMBitWriter.a ${LLVM_DIR}/lib/libLLVMCodeGen.a
      ${LLVM_DIR}/lib/libLLVMCore.a ${LLVM_DIR}/lib/libLLVMExecutionEngine.a
      ${LLVM_DIR}/lib/libLLVMInstCombine.a ${LLVM_DIR}/lib/libLLVMInstrumentation.a
      ${LLVM_DIR}/lib/libLLVMInterpreter.a ${LLVM_DIR}/lib/libLLVMipa.a
      ${LLVM_DIR}/lib/libLLVMipo.a ${LLVM_DIR}/lib/libLLVMIRReader.a
      ${LLVM_DIR}/lib/libLLVMLineEditor.a ${LLVM_DIR}/lib/libLLVMLinker.a
      ${LLVM_DIR}/lib/libLLVMLTO.a ${LLVM_DIR}/lib/libLLVMMC.a
      ${LLVM_DIR}/lib/libLLVMMCDisassembler.a ${LLVM_DIR}/lib/libLLVMMCJIT.a
      ${LLVM_DIR}/lib/libLLVMMCParser.a ${LLVM_DIR}/lib/libLLVMObjCARCOpts.a
      ${LLVM_DIR}/lib/libLLVMObject.a ${LLVM_DIR}/lib/libLLVMOption.a
      ${LLVM_DIR}/lib/libLLVMProfileData.a ${LLVM_DIR}/lib/libLLVMRuntimeDyld.a
      ${LLVM_DIR}/lib/libLLVMScalarOpts.a ${LLVM_DIR}/lib/libLLVMSelectionDAG.a
      ${LLVM_DIR}/lib/libLLVMSupport.a ${LLVM_DIR}/lib/libLLVMTableGen.a
      ${LLVM_DIR}/lib/libLLVMTarget.a ${LLVM_DIR}/lib/libLLVMTransformUtils.a
      ${LLVM_DIR}/lib/libLLVMVectorize.a ${LLVM_DIR}/lib/libLLVMX86AsmParser.a
      ${LLVM_DIR}/lib/libLLVMX86AsmPrinter.a ${LLVM_DIR}/lib/libLLVMX86CodeGen.a
      ${LLVM_DIR}/lib/libLLVMX86Desc.a ${LLVM_DIR}/lib/libLLVMX86Disassembler.a
      ${LLVM_DIR}/lib/libLLVMX86Info.a ${LLVM_DIR}/lib/libLLVMX86Utils.a)
  set(LIBCLANG_INCLUDE ${LLVM_DIR}/include PARENT_SCOPE)
  set(LIBCLANG_LIB ${CLANG_LIB} ${LLVM_LIB} -lcurses PARENT_SCOPE)
endif()

# libcling
if(LFL_LIBCLING)
  set(LIBCLING_INCLUDE ${CLING_LLVM_DIR}/include PARENT_SCOPE)
  set(LIBCLING_LIB ${CLING_LLVM_DIR}/lib/libclingMetaProcessor.a ${CLING_LLVM_DIR}/lib/libclingInterpreter.a
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
  set(JSONCPP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp/include PARENT_SCOPE)
  set(JSONCPP_LIB jsoncpp_lib_static PARENT_SCOPE)
  add_subdirectory(jsoncpp)
endif()

# pcap
if(LFL_PCAP)
  add_subdirectory(libpcap)
  set(PCAP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libpcap PARENT_SCOPE)
  set(PCAP_LIB ${CMAKE_CURRENT_BINARY_DIR}/libpcap/libpcap.a PARENT_SCOPE)
endif()

# GeoIP
if(LFL_GEOIP)
  add_subdirectory(libGeoIP)
  set(GEOIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libGeoIP/libGeoIP PARENT_SCOPE)
  set(GEOIP_LIB ${CMAKE_CURRENT_BINARY_DIR}/libGeoIP/libGeoIP/.libs/libGeoIP.a PARENT_SCOPE)
endif()
