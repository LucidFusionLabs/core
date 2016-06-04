set(LFL_USE_LIBCPP ON)
set(LFL_EMSCRIPTEN 1)
set(LFL_EMSCRIPTEN_ROOT $ENV{HOME}/emsdk_portable/emscripten/1.35.0)

include(${LFL_EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ENV{CC} "${CMAKE_C_COMPILER}")
set(ENV{CXX} "${CMAKE_CXX_COMPILER}")
set(ENV{CPP} "${CMAKE_C_COMPILER} -E")
set(ENV{CXXCPP} "${CMAKE_CXX_COMPILER} -E")
set(ENV{AR} "${CMAKE_AR}")
set(ENV{RANLIB} "${CMAKE_RANLIB}")
set(ENV{CXXFLAGS} "-stdlib=libc++")
