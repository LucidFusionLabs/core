set(LFL_APPLE_DEVELOPER "/Applications/Xcode.app/Contents/Developer")
set(LFL_IOS_ROOT "${LFL_APPLE_DEVELOPER}/Platforms/iPhoneOS.platform/Developer")
set(LFL_IOS_SDK "${LFL_IOS_ROOT}/SDKs/iPhoneOS.sdk")
set(LFL_IOS TRUE)
set(LFL_USE_LIBCPP ON)

include(CMakeForceCompiler)
CMAKE_FORCE_C_COMPILER(/usr/bin/clang Apple)
CMAKE_FORCE_CXX_COMPILER(/usr/bin/clang++ Apple)
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_AR ar CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_OSX_SYSROOT ${LFL_IOS_SDK} CACHE PATH "iOS sysroot")
set(CMAKE_OSX_DEPLOYMENT_TARGET "" CACHE STRING "iOS deploy" FORCE)
set(CMAKE_OSX_ARCHITECTURES armv7 CACHE string "iOS arch")
set(CMAKE_SYSTEM_FRAMEWORK_PATH ${LFL_IOS_SDK}/System/Library/Frameworks)
set(CMAKE_SIZEOF_VOID_P 4)

if(CMAKE_GENERATOR MATCHES Xcode)
  set(CMAKE_MACOSX_BUNDLE YES)
  set(IOS_VERSION_MIN)
  set(IOS_VERSION_MIN_FULL) 
else()
  set(IOS_VERSION_MIN      "-miphoneos-version-min=5.0")
  set(IOS_VERSION_MIN_FULL "-miphoneos-version-min=5.0 -D__IPHONE_OS_VERSION_MIN_REQUIRED=50000") 
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -isysroot ${LFL_IOS_SDK} -F${LFL_IOS_SDK}/System/Library/Frameworks ${IOS_VERSION_MIN}")
add_definitions("-isysroot ${LFL_IOS_SDK} ${IOS_VERSION_MIN_FULL}")
include_directories(${LFL_IOS_SDK}/System/Library/Frameworks/OpenGLES.framework/Headers/
                    ${LFL_IOS_SDK}/System/Library/Frameworks/Foundation.framework/Headers/
                    ${LFL_IOS_SDK}/System/Library/Frameworks/UIKit.framework/Headers/)

set(ENV{CC} "/usr/bin/clang")
set(ENV{CXX} "/usr/bin/clang++")
set(ENV{CODESIGN_ALLOCATE} "${LFL_IOS_ROOT}/usr/bin/codesign_allocate")
set(ENV{CFLAGS}   "-arch armv7 -isysroot ${LFL_IOS_SDK} ${IOS_VERSION_MIN}")
set(ENV{CXXFLAGS} "-arch armv7 -isysroot ${LFL_IOS_SDK} ${IOS_VERSION_MIN}")
set(ENV{LDFLAGS}  "-arch armv7 -isysroot ${LFL_IOS_SDK} ${IOS_VERSION_MIN}")
set(ENV{AR} "${LFL_IOS_ROOT}/usr/bin/ar")

set(CONFIGURE_OPTIONS "--host=arm-apple-darwin")
set(CONFIGURE_ENV CC=$ENV{CC} CXX=$ENV{CXX} CPP=$ENV{CPP} CXXCPP=$ENV{CXXCPP} AR=$ENV{AR} RANLIB=$ENV{RANLIB}
    CFLAGS=$ENV{CFLAGS} CXXFLAGS=$ENV{CXXFLAGS} LDFLAGS=$ENV{LDFLAGS})

set(M_LIBRARY ${LFL_IOS_SDK}/usr/lib/libm.tbd)
set(ZLIB_INCLUDE_DIR ${LFL_IOS_SDK}/usr/include)
set(ZLIB_LIBRARY ${LFL_IOS_SDK}/usr/lib/libz.tbd)