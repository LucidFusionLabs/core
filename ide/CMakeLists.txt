# $Id$

lfl_add_target(app_ide STATIC_LIBRARY SOURCES syntax.cpp)
lfl_add_target(app_null_tu STATIC_LIBRARY SOURCES null_tu.cpp)

if(LFL_LIBCLANG)
  lfl_add_target(app_clang_tu STATIC_LIBRARY SOURCES clang_tu.cpp
                 INCLUDE_DIRECTORIES ${LIBCLANG_INCLUDE} LINK_LIBRARIES ${LIBCLANG_LIB})
  add_dependencies(app_clang_tu llvm)
endif()

if(LFL_JSONCPP)
  lfl_add_target(app_cmake_daemon STATIC_LIBRARY SOURCES cmake.cpp
                 INCLUDE_DIRECTORIES ${JSONCPP_INCLUDE} LINK_LIBRARIES ${JSONCPP_LIB})
  add_dependencies(app_cmake_daemon cmakedaemon)
endif()
