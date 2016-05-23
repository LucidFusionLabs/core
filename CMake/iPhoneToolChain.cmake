set(LFL_IPHONE 1)

set(LFL_USE_LIBCPP ON)
set(IPHONEROOT "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer")
set(IPHONESDK "${IPHONEROOT}/SDKs/iPhoneOS8.3.sdk")

include(CMakeForceCompiler)
CMAKE_FORCE_C_COMPILER(/usr/bin/clang Apple)
CMAKE_FORCE_CXX_COMPILER(/usr/bin/clang++ Apple)
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_AR ar CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_OSX_SYSROOT ${IPHONESDK} CACHE PATH "iOS sysroot")
set(CMAKE_OSX_DEPLOYMENT_TARGET "" CACHE STRING "iOS deploy" FORCE)
set(CMAKE_SYSTEM_FRAMEWORK_PATH ${IPHONESDK}/System/Library/Frameworks)
set(CMAKE_SIZEOF_VOID_P 4)
set(CMAKE_OSX_ARCHITECTURES armv7 CACHE string "iOS arch")

add_definitions("-isysroot ${IPHONESDK} -miphoneos-version-min=5.0 -D__IPHONE_OS_VERSION_MIN_REQUIRED=50000")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -isysroot ${IPHONESDK} -miphoneos-version-min=5.0 -F${IPHONESDK}/System/Library/Frameworks")

include_directories(${IPHONESDK}/System/Library/Frameworks/OpenGLES.framework/Headers/
                    ${IPHONESDK}/System/Library/Frameworks/Foundation.framework/Headers/
                    ${IPHONESDK}/System/Library/Frameworks/UIKit.framework/Headers/)

set(ENV{CC} "/usr/bin/clang")
set(ENV{CXX} "/usr/bin/clang++")
set(ENV{CODESIGN_ALLOCATE} "${IPHONEROOT}/usr/bin/codesign_allocate")
set(ENV{CFLAGS}   "-arch armv7 -miphoneos-version-min=5.0 -isysroot ${IPHONESDK}")
set(ENV{CXXFLAGS} "-arch armv7 -miphoneos-version-min=5.0 -isysroot ${IPHONESDK}")
set(ENV{LDFLAGS}  "-arch armv7 -miphoneos-version-min=5.0 -isysroot ${IPHONESDK}")
set(ENV{AR} "${IPHONEROOT}/usr/bin/ar")
set(CONFIGURE_OPTIONS "--host=arm-apple-darwin")

set(CONFIGURE_ENV CC=$ENV{CC} CXX=$ENV{CXX} CPP=$ENV{CPP} CXXCPP=$ENV{CXXCPP} AR=$ENV{AR} RANLIB=$ENV{RANLIB}
    CFLAGS=$ENV{CFLAGS} CXXFLAGS=$ENV{CXXFLAGS} LDFLAGS=$ENV{LDFLAGS})

set(M_LIBRARY ${IPHONESDK}/usr/lib/libm.dylib)
set(ZLIB_INCLUDE_DIR ${IPHONESDK}/usr/include)
set(ZLIB_LIBRARY ${IPHONESDK}/usr/lib/libz.dylib)
