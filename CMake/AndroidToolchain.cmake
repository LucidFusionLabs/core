set(LFL_ANDROID 1)
set(CMAKE_SYSTEM_NAME Linux)
set(ANDROIDNDK "$ENV{HOME}/android-ndk-r10d")
set(ANDROIDROOT "$ENV{HOME}/android-toolchain")
set(CMAKE_C_COMPILER ${ANDROIDROOT}/bin/arm-linux-androideabi-gcc CACHE PATH "C compiler")
set(CMAKE_CXX_COMPILER ${ANDROIDROOT}/bin/arm-linux-androideabi-g++ CACHE PATH "C++ compiler")
set(CMAKE_AR ${ANDROIDROOT}/bin/arm-linux-androideabi-ar CACHE PATH "archive")
set(CMAKE_RANLIB ${ANDROIDROOT}/bin/arm-linux-androideabi-ranlib CACHE PATH "ranlib")
set(CMAKE_LINKER ${ANDROIDROOT}/bin/arm-linux-androideabi-ld CACHE PATH "linker")
add_definitions("-isysroot ${ANDROIDNDK}/platforms/android-8/arch-arm")

macro(lfl_export_toolchain)
set(ENV{CC} "${ANDROIDROOT}/bin/arm-linux-androideabi-gcc")
set(ENV{CXX} "${ANDROIDROOT}/bin/arm-linux-androideabi-g++")
set(ENV{CPP} "${ANDROIDROOT}/bin/arm-linux-androideabi-cpp")
set(ENV{CXXCPP} "${ANDROIDROOT}/bin/arm-linux-androideabi-cpp")
set(ENV{AR} "${ANDROIDROOT}/bin/arm-linux-androideabi-ar")
set(ENV{RANLIB} "${ANDROIDROOT}/bin/arm-linux-androideabi-ranlib")
set(ENV{CFLAGS} "-isysroot ${ANDROIDROOT}/sysroot")
set(ENV{CXXFLAGS} "-isysroot ${ANDROIDROOT}/sysroot")
set(ENV{LDFLAGS} "-isysroot ${ANDROIDROOT}/sysroot")
set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} -host arm --enable-static=yes --enable-shared=no)
endmacro(lfl_export_toolchain)
