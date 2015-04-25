# $Id: CMakeLists.txt 1318 2014-10-23 06:13:52Z justin $

# Lucid Fusion Labs Root Make File
cmake_minimum_required(VERSION 3.0)
CMAKE_POLICY(SET CMP0004 OLD)
# MESSAGE(FATAL_ERROR "Debug message")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif(NOT CMAKE_BUILD_TYPE)

# http://www.cmake.org/cmake/resources/software.html
# http://atrpms.net/dist/el5/cmake/
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/Autoconf.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/util.cmake)
set(LFL_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
link_directories("")
set(CMAKE_EXE_LINKER_FLAGS "/SAFESEH:NO /NODEFAULTLIB:LIBCMTD")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else(WIN32)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11") # -stdlib=libc++")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wno-deprecated-declarations") # -stdlib=libc++")
#set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++")
#set(CMAKE_SHARED_LINKER_FLAGS "-stdlib=libc++")
#set(CMAKE_MODULE_LINKER_FLAGS "-stdlib=libc++")
endif(WIN32)
add_definitions("-D__STDC_CONSTANT_MACROS")

set(LFL_PROTOBUF 1)
#set(LFL_HEADLESS 1)
#set(LFL_CUDA 1)

# args
if(LFL_IPHONESIM)
 set(LFL_IPHONE 1)
 set(IPHONESIM "-Simulator")
endif(LFL_IPHONESIM)

# configure
if(LFL_IPHONE)
 set(LFL_PNG 1)
 set(LFL_GLES2 1)
 set(LFL_AUDIOUNIT 1)
 set(LFL_BOX2D 1)
 set(PLATFORM_LIB "-lz -framework Foundation -framework CoreAudio -framework CoreGraphics -framework OpenGLES -framework UIKit -framework QuartzCore -framework AVFoundation")

 if(IPHONESIM)
  set(IPHONEROOT "/Developer/Platforms/iPhoneSimulator.platform/Developer")
  set(IPHONESDK "${IPHONEROOT}/SDKs/iPhoneSimulator4.2.sdk")
  add_definitions("-isysroot ${IPHONESDK} -arch i686 -mmacosx-version-min=10.5 -D__IPHONE_OS_VERSION_MIN_REQUIRED=30200")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -isysroot ${IPHONESDK} -arch i386 -mmacosx-version-min=10.5 -F${IPHONESDK}/System/Library/Frameworks")
 else(IPHONESIM)
  set(IPHONEROOT "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/")
  set(IPHONESDK "${IPHONEROOT}/SDKs/iPhoneOS6.1.sdk")
  add_definitions("-isysroot ${IPHONESDK} -arch armv7 -miphoneos-version-min=3.2 -D__IPHONE_OS_VERSION_MIN_REQUIRED=30200")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -isysroot ${IPHONESDK} -arch armv7 -miphoneos-version-min=3.2 -F${IPHONESDK}/System/Library/Frameworks")
  set(PLATFORM_LIB "${PLATFORM_LIB} -framework CoreVideo -framework CoreMedia")
 endif(IPHONESIM)
 
 set(CMAKE_C_COMPILER "${IPHONEROOT}/usr/bin/gcc")
 set(CMAKE_CXX_COMPILER "${IPHONEROOT}/usr/bin/g++")
 include_directories(${IPHONESDK}/System/Library/Frameworks/OpenGLES.framework/Headers/ ${IPHONESDK}/System/Library/Frameworks/Foundation.framework/Headers/ ${IPHONESDK}/System/Library/Frameworks/UIKit.framework/Headers/)

elseif(LFL_ANDROID)
 set(ANDROIDNDK "$ENV{HOME}/android-ndk-r9")
 set(ANDROIDROOT "$ENV{HOME}/android-toolchain")
 set(CMAKE_C_COMPILER ${ANDROIDROOT}/bin/arm-linux-androideabi-gcc)
 set(CMAKE_CXX_COMPILER ${ANDROIDROOT}/bin/arm-linux-androideabi-g++)
 set(CMAKE_AR ${ANDROIDROOT}/bin/arm-linux-androideabi-ar)
 set(CMAKE_RANLIB ${ANDROIDROOT}/bin/arm-linux-androideabi-ranlib)
 set(PLATFORM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/imports/OpenCV/3rdparty/include)
 add_definitions("-I${ANDROIDNDK}/sources/cxx-stl/gnu-libstdc++/4.6/include/ -I${ANDROIDNDK}/sources/cxx-stl/gnu-libstdc++/4.6/libs/armeabi/include/")
 set(LFL_GLES2 1)
 set(LFL_FREETYPE 1)
 set(LFL_BOX2D 1)
 set(LFL_PNG 1)

else(LFL_IPHONE)
 set(LFL_GLOG 1)
 set(LFL_GTEST 1)
 set(LFL_BOX2D 1)
 set(LFL_JUDY 1)
 set(LFL_LIBARCHIVE 1)
 set(LFL_REGEX 1)
 set(LFL_SREGEX 1)
 set(LFL_PNG 1)
 set(LFL_JPEG 1)
 set(LFL_GIF 1)
 set(LFL_GEOIP 1)
 set(LFL_OPENSSL 1)
 set(LFL_PCAP 1)
 set(LFL_CLING 1)
 if(NOT LFL_HEADLESS)
  set(LFL_GLEW 1)
  set(LFL_QT 1)
  #set(LFL_WXWIDGETS 1)
  set(LFL_GLFWVIDEO 1)
  set(LFL_GLFWINPUT 1)
  set(LFL_SDLVIDEO 1)
  set(LFL_SDLINPUT 1)
  set(LFL_FREETYPE 1)
  set(LFL_HARFBUZZ 1)
  set(LFL_LIBCSS 1)
  set(LFL_CAMERA 1)
  set(LFL_OPENCV 1)
  set(LFL_FFMPEG 1)
  set(LFL_X264 1)
  set(LFL_LAME 1)
  set(LFL_LUA 1)
  set(LFL_V8JS 1)
  #set(LFL_BERKELIUM 1)
  set(LFL_PORTAUDIO 1)
 endif(NOT LFL_HEADLESS)
endif(LFL_IPHONE)

# imports
add_subdirectory(imports)

# macro includes
if(LFL_PROTOBUF)
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/FindProtoBuf.cmake)
endif(LFL_PROTOBUF)

# macros
macro(lfl_project _name)
set(LFL_PROJECT ${_name})
set(LFL_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LFAPP_LIB_TYPE STATIC)
if(LFL_QT)
 set(CMAKE_AUTOMOC ON)
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
endif(LFL_QT)
endmacro(lfl_project)

macro(lfl_clear_projects)
set(LFAPP_DEF)
set(LFAPP_INCLUDE)
set(LFAPP_LIB)
set(LFL_AUDIOUNIT)
set(LFL_BERKELIUM)
set(LFL_BOX2D)
set(LFL_CAMERA)
set(LFL_CLING)
set(LFL_FFMPEG)
set(LFL_FREETYPE)
set(LFL_GEOIP)
set(LFL_GIF)
set(LFL_GLOG)
set(LFL_GLES2)
set(LFL_GLEW)
set(LFL_GLFWVIDEO)
set(LFL_GLFWINPUT)
set(LFL_GTEST)
set(LFL_HARFBUZZ)
set(LFL_ICONV)
set(LFL_JPEG)
set(LFL_JUDY)
set(LFL_LAME)
set(LFL_LFLINPUT)
set(LFL_LFLVIDEO)
set(LFL_LIBCSS)
set(LFL_LIBARCHIVE)
set(LFL_LUA)
set(LFL_OPENCV)
set(LFL_OPENSSL)
set(LFL_OSXINPUT)
set(LFL_OSXVIDEO)
set(LFL_PCAP)
set(LFL_PNG)
set(LFL_PROTOBUF)
set(LFL_PORTAUDIO)
set(LFL_QT)
set(LFL_REGEX)
set(LFL_SDLINPUT)
set(LFL_SDLVIDEO)
set(LFL_SREGEX)
set(LFL_V8JS)
set(LFL_WXWIDGETS)
set(LFL_X264)
endmacro(lfl_clear_projects)
lfl_clear_projects()

if(LFL_CUDA)
add_subdirectory(lfcuda)
endif(LFL_CUDA)

# lfapp unit tests
add_subdirectory(lfapp_tests)

# crystal bawl
add_subdirectory(cb)

# cluster
add_subdirectory(cluster)

# crawler
add_subdirectory(crawler)

# map/reduce
add_subdirectory(mapreduce)

# nlp
add_subdirectory(nlp)

# speech
add_subdirectory(speech)

# fusion server
add_subdirectory(fs)

# fusion viewer
add_subdirectory(fv)

# calculator
add_subdirectory(calculator)

# terminal
add_subdirectory(term)

# editor
add_subdirectory(editor)

# web browser
add_subdirectory(browser)

# chess
add_subdirectory(chess)

# spaceball
add_subdirectory(spaceball)

# image util
add_subdirectory(image)

# master server list
add_subdirectory(master)

# mta contract
add_subdirectory(blaster)

# market app
add_subdirectory(market)

# q3 clone
add_subdirectory(quake)

if(LFL_ANDROID)
else(LFL_ANDROID)
add_subdirectory(senators)
endif(LFL_ANDROID)

# add more apps here
# add_subdirectory(my_first_app)

