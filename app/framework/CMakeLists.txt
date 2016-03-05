# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})
add_library(app_null_os STATIC null_os.cpp)
add_library(app_null_framework STATIC null_framework.cpp)

if(LFL_OSX)
  add_library(app_osx_os STATIC osx_os.mm)
  target_compile_options(app_osx_os PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)

  add_library(app_osx_framework STATIC osx_framework.mm)
  target_compile_options(app_osx_framework PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
endif()

if(LFL_LINUX)
  add_library(app_linux_os STATIC linux_os.cpp)
  target_link_libraries(app_linux_os ${PLATFORM_LIB})

  add_library(app_x11_framework STATIC x11_framework.cpp)
  target_link_libraries(app_x11_framework -lXi -lXv -lXext -lX11 -lXxf86vm -lXrandr -lXcursor -lXt)
endif()

if(LFL_QT)
  lfl_enable_qt()
  set(CMAKE_AUTOMOC ON)
  add_library(app_qt_framework STATIC qt_framework.cpp)
  target_include_directories(app_qt_framework PUBLIC ${QT_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(app_qt_framework PUBLIC ${QT_LIB})
  set(CMAKE_AUTOMOC)
endif()

if(LFL_SDL)
  add_library(app_sdl_framework STATIC sdl_framework.cpp)
  target_include_directories(app_sdl_framework PUBLIC ${SDL_INCLUDE})
  target_link_libraries(app_sdl_framework PUBLIC ${SDL_LIB})
endif()
