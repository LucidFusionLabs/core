set(LFL_ANDROID 1)
set(LFL_ANDROID_SDK "$ENV{HOME}/android-sdk-macosx")
set(LFL_ANDROID_NDK "$ENV{HOME}/android-ndk-r10d")
set(LFL_ANDROID_ROOT "$ENV{HOME}/android-toolchain")
set(LFL_GRADLE_BIN "$ENV{HOME}/gradle-2.4/bin/gradle")

#include(core/CMake/LFLOS.cmake)
if(NOT LFL_OS)
  if(APPLE)
    set(LFL_OS osx)
  elseif(UNIX)
    set(LFL_OS linux)
  elseif(WIN32 OR WIN64)
    set(LFL_OS win32)
  else()
    MESSAGE(FATAL_ERROR "unknown OS")
  endif()
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER ${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-gcc CACHE PATH "C compiler")
set(CMAKE_CXX_COMPILER ${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-g++ CACHE PATH "C++ compiler")
set(CMAKE_AR ${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-ar CACHE PATH "archive")
set(CMAKE_RANLIB ${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-ranlib CACHE PATH "ranlib")
set(CMAKE_LINKER ${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-ld CACHE PATH "linker")

set(ENV{CC} "${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-gcc")
set(ENV{CXX} "${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-g++")
set(ENV{CPP} "${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-cpp")
set(ENV{CXXCPP} "${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-cpp")
set(ENV{AR} "${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-ar")
set(ENV{RANLIB} "${LFL_ANDROID_ROOT}/bin/arm-linux-androideabi-ranlib")
set(ENV{CFLAGS} "-isysroot ${LFL_ANDROID_ROOT}/sysroot")
set(ENV{CXXFLAGS} "-isysroot ${LFL_ANDROID_ROOT}/sysroot")
set(ENV{LDFLAGS} "-isysroot ${LFL_ANDROID_ROOT}/sysroot")
set(CONFIGURE_OPTIONS "--host=arm")
set(CONFIGURE_ENV CC=$ENV{CC} CXX=$ENV{CXX} CPP=$ENV{CPP} CXXCPP=$ENV{CXXCPP} AR=$ENV{AR} RANLIB=$ENV{RANLIB}
    CFLAGS=$ENV{CFLAGS} CXXFLAGS=$ENV{CXXFLAGS} LDFLAGS=$ENV{LDFLAGS})

set(M_LIBRARY ${LFL_ANDROID_ROOT}/sysroot/usr/lib/libm.so)
set(ZLIB_INCLUDE_DIR ${LFL_ANDROID_ROOT}/sysroot/usr/include)
set(ZLIB_LIBRARY ${LFL_ANDROID_ROOT}/sysroot/usr/lib/libz.so)
