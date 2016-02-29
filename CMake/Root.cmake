# Lucid Fusion Labs Root Make File
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

if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND NOT LFL_IPHONE)
  set(LFL_OSX 1)
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
include(${LFL_SOURCE_DIR}/core/CMake/Autoconf.cmake)
include(${LFL_SOURCE_DIR}/core/CMake/util.cmake)

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
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11") # -stdlib=libc++")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wno-deprecated-declarations") # -Wold-style-cast") # -stdlib=libc++")
  #set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++")
  #set(CMAKE_SHARED_LINKER_FLAGS "-stdlib=libc++")
  #set(CMAKE_MODULE_LINKER_FLAGS "-stdlib=libc++")
  if(LFL_IPHONE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -stdlib=libc++")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -stdlib=libc++")
  endif()
endif()

add_definitions(-D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS)

# imports
add_subdirectory(${LFL_SOURCE_DIR}/core/imports)

# macros
include(${LFL_SOURCE_DIR}/core/CMake/Package.cmake)

if(LFL_PROTOBUF)
  include(${LFL_SOURCE_DIR}/core/CMake/FindProtoBuf.cmake)
endif()

if(LFL_FLATBUFFERS)
  set(FLATBUFFERS_INCLUDE_DIR ${LFL_SOURCE_DIR}/core/imports/flatbuffers/include)
  set(FLATBUFFERS_FLATC_EXECUTABLE ${LFL_BINARY_DIR}/core/imports/flatbuffers/flatc)
  include(${LFL_SOURCE_DIR}/core/imports/flatbuffers/CMake/FindFlatBuffers.cmake)
endif()

if(LFL_CAPNPROTO)
  set(CAPNP_LIB_KJ          ${LFL_BINARY_DIR}/core/imports/capnproto/lib/libkj.a)
  set(CAPNP_LIB_KJ-ASYNC    ${LFL_BINARY_DIR}/core/imports/capnproto/lib/libkj-async.a)
  set(CAPNP_LIB_CAPNP       ${LFL_BINARY_DIR}/core/imports/capnproto/lib/libcapnp.a)
  set(CAPNP_LIB_CAPNP-RPC   ${LFL_BINARY_DIR}/core/imports/capnproto/lib/libcapnp-rpc.a)
  set(CAPNP_EXECUTABLE      ${LFL_BINARY_DIR}/core/imports/capnproto/bin/capnp)
  set(CAPNPC_CXX_EXECUTABLE ${LFL_BINARY_DIR}/core/imports/capnproto/bin/capnpc-c++)
  set(CAPNP_INCLUDE_DIRS    ${LFL_BINARY_DIR}/core/imports/capnproto/include)
  include(${LFL_SOURCE_DIR}/core/imports/capnproto/c++/cmake/FindCapnProto.cmake)
endif(LFL_CAPNPROTO)

macro(lfl_project _name)
  project(${_name})
  set(LFL_PROJECT ${_name})
  set(LFL_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(LFL_PROJECT_BINDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro(lfl_project)

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

if(WIN32)
  macro(add_external_dep _lib)
  endmacro()
else()
  macro(add_external_dep _lib)
    add_dependencies(${_lib} ${ARGN})
  endmacro()
endif()

# app
add_subdirectory(${LFL_SOURCE_DIR}/core/app)

# app unit tests
add_subdirectory(${LFL_SOURCE_DIR}/core/app_tests)

# web
add_subdirectory(${LFL_SOURCE_DIR}/core/web)

# nlp
add_subdirectory(${LFL_SOURCE_DIR}/core/nlp)

# speech
add_subdirectory(${LFL_SOURCE_DIR}/core/speech)
