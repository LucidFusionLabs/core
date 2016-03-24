set(LLVM_DIR $ENV{HOME}/llvm)

include(CMakeForceCompiler)
CMAKE_FORCE_C_COMPILER(${LLVM_DIR}/bin/clang Clang)
CMAKE_FORCE_CXX_COMPILER(${LLVM_DIR}/bin/clang++ Clang)
set(CMAKE_AR ${LLVM_DIR}/bin/llvm-ar CACHE PATH "archive")
set(CMAKE_RANLIB ${LLVM_DIR}/bin/llvm-ranlib CACHE PATH "ranlib")
set(CMAKE_LINKER ${LLVM_DIR}/bin/llvm-ld CACHE PATH "linker")
set(CMAKE_SIZEOF_VOID_P 8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LFL_PRECOMPILED_HEADERS ON)
set(LFL_USE_LIBCPP ON)

set(ENV{CC} "${LLVM_DIR}/bin/clang")
set(ENV{CXX} "${LLVM_DIR}/bin/clang++")
set(ENV{CPP} "${LLVM_DIR}/bin/clang -E")
set(ENV{CXXCPP} "${LLVM_DIR}/bin/clang++ -E")
set(ENV{AR} "${LLVM_DIR}/bin/llvm-ar")
set(ENV{RANLIB} "${LLVM_DIR}/bin/llvm-ranlib")
set(ENV{CXXFLAGS} "-stdlib=libc++")
