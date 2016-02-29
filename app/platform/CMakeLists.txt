# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

include_directories(${LFL_SOURCE_DIR})
add_library(app_null_platform STATIC null_platform.cpp)

if(LFL_OSX)
  add_library(app_osx_platform STATIC osx_platform.mm)
  target_compile_options(app_osx_platform PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
endif()

if(LFL_QT)
  lfl_enable_qt()
  set(CMAKE_AUTOMOC ON)
  add_library(app_qt_platform STATIC qt_platform.cpp)
  target_include_directories(app_qt_platform PUBLIC ${QT_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(app_qt_platform PUBLIC ${QT_LIB})
  set(CMAKE_AUTOMOC)
endif()
