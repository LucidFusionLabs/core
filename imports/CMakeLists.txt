# $Id: CMakeLists.txt 1325 2014-10-29 06:08:31Z justin $

# Windows
if(WIN32)
    INCLUDE_EXTERNAL_MSPROJECT(zlib ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/zlib/zlib.vcxproj)
    set(PLATFORM_LIB Ws2_32.lib Dnsapi.lib opengl32.lib glu32.lib strmiids.lib zlib.lib)
    set(PLATFORM_DEF "-DGLEW_STATIC")
endif(WIN32)

# Mac
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(OPENGL_INCLUDE ${OPENGL_INCLUDE_DIR}/Headers ${CMAKE_CURRENT_SOURCE_DIR}/GL PARENT_SCOPE)
    set(PLATFORM_LIB "-framework Cocoa -framework CoreVideo -framework QTKit -lz -lbz2 -liconv")
    if(LFL_SDLINPUT)
        set(PLATFORM_LIB "${PLATFORM_LIB} -framework Carbon -framework IOKit -framework forcefeedback -framework CoreAudio -framework AudioUnit")
    endif(LFL_SDLINPUT)
    if(LFL_GLFWINPUT)
        set(PLATFORM_LIB "${PLATFORM_LIB} -framework IOKit")
    endif(LFL_GLFWINPUT)
endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")

# Linux
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(HAVE_SDLCONFIG 1)
    set(CUDA_LIB /usr/local/cuda/lib64/libcudart.so PARENT_SCOPE)
    set(PLATFORM_LIB "-lutil -lz -lpng -lpthread")
    if(NOT LFL_HEADLESS)
        set(PLATFORM_LIB "${PLATFORM_LIB} -lXi -lXv -lXext -lX11 -lXxf86vm -lXrandr -lXcursor -lbz2") # -lasound")
    endif()
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

# iPhone 
if(LFL_IPHONE)
    set(PLATFORM_LIB "-liconv -lz -framework Foundation -framework CoreText -framework CoreAudio -framework CoreGraphics -framework OpenGLES -framework UIKit -framework QuartzCore -framework AVFoundation -framework GLKit -framework Security")
    if(NOT LFL_IPHONESIM)
        set(PLATFORM_LIB "${PLATFORM_LIB} -framework CoreVideo -framework CoreMedia")
    endif()
endif(LFL_IPHONE)

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
endif(LFL_PNG)

# jpeg
if(LFL_JPEG)
    if(WIN32)
        INCLUDE_EXTERNAL_MSPROJECT(libjpeg ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/libjpeg/libjpeg.vcxproj)
        set(JPEG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/include PARENT_SCOPE)
        set(JPEG_LIB libjpeg.lib PARENT_SCOPE)
    else(WIN32)
        add_subdirectory(libjpeg-turbo)
        set(JPEG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo PARENT_SCOPE)
        set(JPEG_LIB ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/.libs/libturbojpeg.a PARENT_SCOPE)
    endif(WIN32)
endif(LFL_JPEG)

# gif
if(LFL_GIF)
    add_subdirectory(giflib)
    set(GIF_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/giflib/lib PARENT_SCOPE)
    set(GIF_LIB libgif PARENT_SCOPE)
endif(LFL_GIF)

# freetype
if(LFL_FREETYPE)
    add_subdirectory(freetype)
    set(FREETYPE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/freetype/include PARENT_SCOPE)
    if(WIN32)
        set(FREETYPE_LIB freetype239_D.lib PARENT_SCOPE)
    else(WIN32)
        set(FREETYPE_LIB ${CMAKE_CURRENT_BINARY_DIR}/freetype/.libs/libfreetype.a PARENT_SCOPE)
    endif(WIN32)
endif(LFL_FREETYPE)

# harfbuzz
# ./configure --enable-static --with-coretext=yes --prefix=`pwd`/bin
if(LFL_HARFBUZZ)
    set(HARFBUZZ_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/harfbuzz/bin/include PARENT_SCOPE)
    set(HARFBUZZ_LIB ${CMAKE_CURRENT_BINARY_DIR}/harfbuzz/src/.libs/libharfbuzz.a PARENT_SCOPE)
endif(LFL_HARFBUZZ)

# libcss
if(LFL_LIBCSS)
    add_subdirectory(libcss)
    set(LIBCSS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libcss/libcss/include ${CMAKE_CURRENT_SOURCE_DIR}/libcss/libwapcaplet/include PARENT_SCOPE)
    set(LIBCSS_LIB libcss PARENT_SCOPE)
endif(LFL_LIBCSS)

# protobuf
if(LFL_PROTOBUF)
    add_subdirectory(protobuf)
    if(WIN32)
        set(PROTOBUF_LIBRARY libprotobuf-lite.lib PARENT_SCOPE)
    else(WIN32)
        set(PROTOBUF_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/.libs/libprotobuf.a PARENT_SCOPE)
    endif(WIN32)
    set(PROTOBUF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src PARENT_SCOPE)
    set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protoc PARENT_SCOPE)
endif(LFL_PROTOBUF)

# glog
if(LFL_GLOG)
    add_subdirectory(glog)
    set(GLOG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glog/src/glog PARENT_SCOPE)
    set(GLOG_LIB ${CMAKE_CURRENT_BINARY_DIR}/glog/.libs/libglog.a PARENT_SCOPE)
endif(LFL_GLOG)

# gtest
if(LFL_GTEST)
    add_subdirectory(protobuf/gtest)
    set(GTEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/gtest/include PARENT_SCOPE)
    set(GTEST_LIB ${CMAKE_CURRENT_BINARY_DIR}/protobuf/gtest/lib/.libs/libgtest.a PARENT_SCOPE)
endif(LFL_GTEST)

# tcmalloc
if(LFL_TCMALLOC)
    add_subdirectory(gperftools)
    set(TCMALLOC_LIB ${CMAKE_CURRENT_BINARY_DIR}/gperftools/.libs/libtcmalloc_minimal.a -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free PARENT_SCOPE)
endif(LFL_TCMALLOC)

# openssl
if(LFL_OPENSSL)
    if(WIN32)
        set(OPENSSL_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/openssl/include PARENT_SCOPE)
        set(OPENSSL_LIB ${CMAKE_CURRENT_BINARY_DIR}/openssl/lib/ssleay32.lib ${CMAKE_CURRENT_BINARY_DIR}/openssl/lib/libeay32.lib PARENT_SCOPE)
    elseif(LFL_IPHONE)
        set(OPENSSL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/include PARENT_SCOPE)
        set(OPENSSL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/lib/libssl.a ${CMAKE_CURRENT_SOURCE_DIR}/ios-openssl/lib/libcrypto.a PARENT_SCOPE)
    else()
        set(OPENSSL_LIB -lssl -lcrypto PARENT_SCOPE)
    endif()
endif(LFL_OPENSSL)

# box2d
if(LFL_BOX2D)
    add_subdirectory(Box2D)
    set(BOX2D_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/Box2D PARENT_SCOPE)
    set(BOX2D_LIB Box2D PARENT_SCOPE)
endif(LFL_BOX2D)

# assimp
if(LFL_ASSIMP)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(assimp)
    set(ASSIMP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include PARENT_SCOPE)
    set(ASSIMP_LIB assimp PARENT_SCOPE)
endif(LFL_ASSIMP)

# ffmpeg
if(LFL_FFMPEG)
    if(NOT LFL_ANDROID AND NOT WIN32)
        # lame
        add_subdirectory(lame)
        set(LAME_LIB -L${CMAKE_CURRENT_BINARY_DIR}/lame/libmp3lame/.libs -lmp3lame PARENT_SCOPE)
    
        # x264
        add_subdirectory(x264)
        set(X264_LIB -L${CMAKE_CURRENT_BINARY_DIR}/x264 -lx264 PARENT_SCOPE)
    endif(NOT LFL_ANDROID AND NOT WIN32)

    add_subdirectory(ffmpeg)
    if(WIN32)
        set(FFMPEG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/include PARENT_SCOPE)
        set(FFMPEG_LIB ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/avformat.lib ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/avcodec.lib ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/avutil.lib ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/swscale.lib ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/w32/lib/swresample.lib PARENT_SCOPE)
    else(WIN32)
        set(FFMPEG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg PARENT_SCOPE)
        set(FFMPEG_LIB ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavdevice/libavdevice.a ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavfilter/libavfilter.a ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavformat/libavformat.a ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavcodec/libavcodec.a ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libavutil/libavutil.a ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libswscale/libswscale.a ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libswresample/libswresample.a ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/libpostproc/libpostproc.a PARENT_SCOPE)
    endif(WIN32)
endif(LFL_FFMPEG)

# CUDA
if(LFL_CUDA)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../CMake/cuda/FindCUDA.cmake)
    set(CUDA_INCLUDE ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
    set(CUDA_LIB ${CUDA_LIBRARIES} PARENT_SCOPE)
endif(LFL_CUDA)

# OpenGL
if(LFL_ANDROID)
elseif(LFL_IPHONE)
elseif(NOT LFL_HEADLESS)
    INCLUDE(${CMAKE_ROOT}/Modules/FindOpenGL.cmake)
    set(OPENGL_INCLUDE ${OPENGL_INCLUDE_DIR} PARENT_SCOPE)
    set(OPENGL_LIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} PARENT_SCOPE)
endif()

# GLEW
if(LFL_GLEW)
    add_subdirectory(glew)
    set(GLEW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glew/include PARENT_SCOPE)
    set(GLEW_LIB glew PARENT_SCOPE)
endif(LFL_GLEW)

# GLFW
if(LFL_GLFWINPUT)
    add_subdirectory(glfw)
    set(GLFW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include PARENT_SCOPE)
    set(GLFW_LIB glfw PARENT_SCOPE)
endif(LFL_GLFWINPUT)

# SDL
if(LFL_SDLINPUT)
    add_subdirectory(SDL)
    set(SDL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/SDL/include PARENT_SCOPE)
    set(SDL_LIB ${CMAKE_CURRENT_BINARY_DIR}/SDL/build/.libs/libSDL2.a PARENT_SCOPE)
endif(LFL_SDLINPUT)

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
    endforeach(d)
    set(WXWIDGETS_DEF "${WXWIDGETS_DEF}" PARENT_SCOPE)
    set(WXWIDGETS_INCLUDE "${wxWidgets_INCLUDE_DIRS}" PARENT_SCOPE)
    set(WXWIDGETS_LIB "${wxWidgets_LIBRARIES}" PARENT_SCOPE)
endif(LFL_WXWIDGETS)

# portaudio
if(LFL_PORTAUDIO)
    add_subdirectory(portaudio)
    set(PORTAUDIO_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/portaudio/include PARENT_SCOPE)
    if(WIN32)
        set(PORTAUDIO_LIB portaudio_x86.lib PARENT_SCOPE)
    else(WIN32)
        set(PORTAUDIO_LIB -lportaudio -L${CMAKE_CURRENT_BINARY_DIR}/portaudio/lib/.libs PARENT_SCOPE)
    endif(WIN32)
endif(LFL_PORTAUDIO)

# OpenCV
if(LFL_OPENCV)
    set(OPENCV_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/include PARENT_SCOPE)
    if(WIN32)
        INCLUDE_EXTERNAL_MSPROJECT(OpenCV ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/src/cv/cv.vcxproj)
        INCLUDE_EXTERNAL_MSPROJECT(cxcore ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/src/cxcore/cxcore.vcxproj)
        INCLUDE_EXTERNAL_MSPROJECT(flann ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/flann/flann.vcxproj)
        INCLUDE_EXTERNAL_MSPROJECT(opencv_lapack ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV/3rdparty/lapack/opencv_lapack.vcxproj)
    set(OPENCV_LIB cv.lib cxcore.lib flannd.lib opencv_lapackd.lib PARENT_SCOPE)
    else(WIN32)
        ExternalProject_Add(OpenCV SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OpenCV
                                   PREFIX ${CMAKE_CURRENT_BINARY_DIR}/OpenCV
                               CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/OpenCV)
        set(OPENCV_LIB -L${CMAKE_CURRENT_BINARY_DIR}/OpenCV/lib -lcv -lcxcore PARENT_SCOPE)
    endif(WIN32)
endif(LFL_OPENCV)

# libarchive
if(LFL_LIBARCHIVE)
    add_subdirectory(libarchive)
    set(ARCHIVE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libarchive PARENT_SCOPE)
    if(WIN32)
        set(ARCHIVE_LIB libarchive.lib PARENT_SCOPE)
    else(WIN32)
        set(ARCHIVE_LIB archive_static PARENT_SCOPE)
    endif(WIN32)
endif(LFL_LIBARCHIVE)

# regex
if(LFL_REGEX)
    add_subdirectory(regex)
    set(REGEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/regex PARENT_SCOPE)
    set(REGEX_LIB regex PARENT_SCOPE)
endif(LFL_REGEX)

# sregex
if(LFL_SREGEX)
    add_subdirectory(sregex)
    set(SREGEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/sregex/src/sregex PARENT_SCOPE)
    set(SREGEX_LIB sregex PARENT_SCOPE)
endif(LFL_SREGEX)

# judy
if(LFL_JUDY)
    add_subdirectory(judy)
    set(JUDY_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/judy/src PARENT_SCOPE)
    set(JUDY_LIB ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyL/.libs/libJudyL.a ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyL/.libs/libprev.a ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyL/.libs/libnext.a ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudySL/.libs/libJudySL.a ${CMAKE_CURRENT_BINARY_DIR}/judy/src/JudyCommon/.libs/libJudyMalloc.a PARENT_SCOPE)
endif(LFL_JUDY)

# cling
if(LFL_CLING)
    set(CLING_INCLUDE $ENV{HOME}/cling/include PARENT_SCOPE)
    set(CLING_LIB $ENV{HOME}/cling/lib/libclang.a $ENV{HOME}/cling/lib/libclangAnalysis.a $ENV{HOME}/cling/lib/libclangARCMigrate.a $ENV{HOME}/cling/lib/libclangAST.a $ENV{HOME}/cling/lib/libclangASTMatchers.a $ENV{HOME}/cling/lib/libclangBasic.a $ENV{HOME}/cling/lib/libclangCodeGen.a $ENV{HOME}/cling/lib/libclangDriver.a $ENV{HOME}/cling/lib/libclangDynamicASTMatchers.a $ENV{HOME}/cling/lib/libclangEdit.a $ENV{HOME}/cling/lib/libclangFormat.a $ENV{HOME}/cling/lib/libclangFrontend.a $ENV{HOME}/cling/lib/libclangFrontendTool.a $ENV{HOME}/cling/lib/libclangIndex.a $ENV{HOME}/cling/lib/libclangLex.a $ENV{HOME}/cling/lib/libclangParse.a $ENV{HOME}/cling/lib/libclangRewrite.a $ENV{HOME}/cling/lib/libclangRewriteFrontend.a $ENV{HOME}/cling/lib/libclangSema.a $ENV{HOME}/cling/lib/libclangSerialization.a $ENV{HOME}/cling/lib/libclangStaticAnalyzerCheckers.a $ENV{HOME}/cling/lib/libclangStaticAnalyzerCore.a $ENV{HOME}/cling/lib/libclangStaticAnalyzerFrontend.a $ENV{HOME}/cling/lib/libclangTooling.a $ENV{HOME}/cling/lib/libclingInterpreter.a $ENV{HOME}/cling/lib/libclingMetaProcessor.a $ENV{HOME}/cling/lib/libclingUITextInput.a $ENV{HOME}/cling/lib/libclingUserInterface.a $ENV{HOME}/cling/lib/libclingUtils.a $ENV{HOME}/cling/lib/libLLVMAnalysis.a $ENV{HOME}/cling/lib/libLLVMAsmParser.a $ENV{HOME}/cling/lib/libLLVMAsmPrinter.a $ENV{HOME}/cling/lib/libLLVMBitReader.a $ENV{HOME}/cling/lib/libLLVMBitWriter.a $ENV{HOME}/cling/lib/libLLVMCodeGen.a $ENV{HOME}/cling/lib/libLLVMCore.a $ENV{HOME}/cling/lib/libLLVMDebugInfo.a $ENV{HOME}/cling/lib/libLLVMExecutionEngine.a $ENV{HOME}/cling/lib/libLLVMInstCombine.a $ENV{HOME}/cling/lib/libLLVMInstrumentation.a $ENV{HOME}/cling/lib/libLLVMInterpreter.a $ENV{HOME}/cling/lib/libLLVMipa.a $ENV{HOME}/cling/lib/libLLVMipo.a $ENV{HOME}/cling/lib/libLLVMIRReader.a $ENV{HOME}/cling/lib/libLLVMJIT.a $ENV{HOME}/cling/lib/libLLVMLineEditor.a $ENV{HOME}/cling/lib/libLLVMLinker.a $ENV{HOME}/cling/lib/libLLVMLTO.a $ENV{HOME}/cling/lib/libLLVMMC.a $ENV{HOME}/cling/lib/libLLVMMCAnalysis.a $ENV{HOME}/cling/lib/libLLVMMCDisassembler.a $ENV{HOME}/cling/lib/libLLVMMCJIT.a $ENV{HOME}/cling/lib/libLLVMMCParser.a $ENV{HOME}/cling/lib/libLLVMObjCARCOpts.a $ENV{HOME}/cling/lib/libLLVMObject.a $ENV{HOME}/cling/lib/libLLVMOption.a $ENV{HOME}/cling/lib/libLLVMProfileData.a $ENV{HOME}/cling/lib/libLLVMRuntimeDyld.a $ENV{HOME}/cling/lib/libLLVMScalarOpts.a $ENV{HOME}/cling/lib/libLLVMSelectionDAG.a $ENV{HOME}/cling/lib/libLLVMSupport.a $ENV{HOME}/cling/lib/libLLVMTableGen.a $ENV{HOME}/cling/lib/libLLVMTarget.a $ENV{HOME}/cling/lib/libLLVMTransformUtils.a $ENV{HOME}/cling/lib/libLLVMVectorize.a $ENV{HOME}/cling/lib/libLLVMX86AsmParser.a $ENV{HOME}/cling/lib/libLLVMX86AsmPrinter.a $ENV{HOME}/cling/lib/libLLVMX86CodeGen.a $ENV{HOME}/cling/lib/libLLVMX86Desc.a $ENV{HOME}/cling/lib/libLLVMX86Disassembler.a $ENV{HOME}/cling/lib/libLLVMX86Info.a $ENV{HOME}/cling/lib/libLLVMX86Utils.a $ENV{HOME}/cling/lib/libLTO.a -lcurses PARENT_SCOPE) 
endif(LFL_CLING)

# bullet
if(LFL_BULLET)
    add_subdirectory(bullet)
    set(BULLET_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/bullet/src PARENT_SCOPE)
    set(BULLET_LIB BulletDynamics BulletCollision LinearMath PARENT_SCOPE)
endif(LFL_BULLET)

# open dynamics engine
if(LFL_ODE)
    add_subdirectory(ode)
    set(ODE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ode/include PARENT_SCOPE)
    set(ODE_LIB ${CMAKE_CURRENT_BINARY_DIR}/ode/ode/src/.libs/libode.a PARENT_SCOPE)
endif(LFL_ODE)

# berkelium
if(LFL_BERKELIUM)
    set(BERKELIUM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/berkelium/osx/include PARENT_SCOPE)
    set(BERKELIUM_LIB ${CMAKE_CURRENT_BINARY_DIR}/berkelium/osx/lib/liblibberkelium.dylib PARENT_SCOPE)
endif(LFL_BERKELIUM)

# lua
if(LFL_LUA)
    add_subdirectory(lua)
    set(LUA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/lua/src PARENT_SCOPE)
    set(LUA_LIB ${CMAKE_CURRENT_BINARY_DIR}/lua/src/liblua.a PARENT_SCOPE)
endif(LFL_LUA)

# v8 js
if(LFL_V8JS)
    set(V8JS_INCLUDE $ENV{HOME}/v8/include PARENT_SCOPE)
    set(V8JS_LIB $ENV{HOME}/v8/native/libv8_libplatform.a $ENV{HOME}/v8/native/libv8_libbase.a $ENV{HOME}/v8/native/libv8_base.a $ENV{HOME}/v8/native/libv8_snapshot.a $ENV{HOME}/v8/native/libicuuc.a $ENV{HOME}/v8/native/libicui18n.a $ENV{HOME}/v8/native/libicudata.a PARENT_SCOPE)
endif(LFL_V8JS)

# tinyjs
if(LFL_TINYJS)
    set(TINYJS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/TinyJS/TinyJS.o PARENT_SCOPE)
endif(LFL_TINYJS)

# pcap
if(LFL_PCAP)
    add_subdirectory(libpcap)
    set(PCAP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libpcap PARENT_SCOPE)
    set(PCAP_LIB ${CMAKE_CURRENT_BINARY_DIR}/libpcap/libpcap.a PARENT_SCOPE)
endif(LFL_PCAP)

# GeoIP
if(LFL_GEOIP)
    add_subdirectory(libGeoIP)
    set(GEOIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/libGeoIP/libGeoIP PARENT_SCOPE)
    set(GEOIP_LIB ${CMAKE_CURRENT_BINARY_DIR}/libGeoIP/libGeoIP/.libs/libGeoIP.a PARENT_SCOPE)
endif(LFL_GEOIP)
