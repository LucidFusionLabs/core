# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_framework STATIC_LIBRARY SOURCES null_framework.cpp)
lfl_add_target(app_null_os STATIC_LIBRARY SOURCES null_os.cpp)
lfl_add_target(app_null_crashreporting STATIC_LIBRARY SOURCES null_crashreporting.cpp)

if(LFL_ANDROID)
  lfl_add_target(app_android_os STATIC_LIBRARY SOURCES android_os.cpp LINK_LIBRARIES -llog -lz)
  lfl_add_target(app_android_framework STATIC_LIBRARY SOURCES android_framework.cpp)
endif()

if(LFL_IOS)
  add_library(app_ios_os STATIC ios_os.mm apple_common.mm)
  target_include_directories(app_ios_os PUBLIC ${LFL_APP_INCLUDE})
  target_compile_definitions(app_ios_os PUBLIC ${LFL_APP_DEF})
  target_compile_options(app_ios_os PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
  target_link_libraries(app_ios_os -liconv -lz "-framework Foundation" "-framework CoreText"
                        "-framework CoreGraphics" "-framework OpenGLES" "-framework UIKit"
                        "-framework QuartzCore" "-framework GLKit" "-framework Security"
                        "-framework AVFoundation")
  if(NOT LFL_IOS_SIM)
    target_link_libraries(app_ios_os "-framework CoreVideo" "-framework CoreMedia")
  endif()

  add_library(app_ios_framework STATIC ios_framework.mm)
  target_include_directories(app_ios_framework PUBLIC ${LFL_APP_INCLUDE})
  target_compile_definitions(app_ios_framework PUBLIC ${LFL_APP_DEF})
  target_compile_options(app_ios_framework PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
endif()

if(LFL_OSX)
  add_library(app_osx_os STATIC osx_os.mm apple_common.mm)
  target_include_directories(app_osx_os PUBLIC ${LFL_APP_INCLUDE})
  target_compile_definitions(app_osx_os PUBLIC ${LFL_APP_DEF})
  target_compile_options(app_osx_os PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
  target_link_libraries(app_osx_os "-framework Cocoa" "-framework CoreVideo" "-framework QTKit"
                        -lz -lbz2 -liconv)

  add_library(app_osx_framework STATIC osx_framework.mm)
  target_include_directories(app_osx_framework PUBLIC ${LFL_APP_INCLUDE})
  target_compile_definitions(app_osx_framework PUBLIC ${LFL_APP_DEF})
  target_compile_options(app_osx_framework PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
endif()

if(LFL_CRITTERCISM AND (LFL_IOS OR LFL_OSX))
  lfl_add_target(app_crittercism_crashreporting STATIC_LIBRARY SOURCES crittercism_crashreporting.mm
                 COMPILE_OPTIONS -fobjc-abi-version=2 -fobjc-legacy-dispatch
                 INCLUDE_DIRECTORIES ${CRITTERCISM_INCLUDE}
                 LINK_LIBRARIES ${CRITTERCISM_LIB} "-framework SystemConfiguration")
endif()

if(LFL_HOCKEYAPP AND LFL_IOS)
  lfl_add_target(app_hockeyapp_crashreporting STATIC_LIBRARY SOURCES hockeyapp_crashreporting.mm
                 COMPILE_OPTIONS -fobjc-abi-version=2 -fobjc-legacy-dispatch
                 INCLUDE_DIRECTORIES ${HOCKEYAPP_INCLUDE}
                 LINK_LIBRARIES ${HOCKEYAPP_LIB} "-framework SystemConfiguration")
endif()

if(LFL_LINUX)
  lfl_add_target(app_linux_os STATIC_LIBRARY SOURCES linux_os.cpp
                 LINK_LIBRARIES -lutil -lz -lpng -lpthread -ldl -lbz2)
  lfl_add_target(app_x11_framework STATIC_LIBRARY SOURCES x11_framework.cpp
                 LINK_LIBRARIES -lXi -lXv -lXext -lX11 -lXxf86vm -lXrandr -lXcursor -lXt)
endif()

if(LFL_WINDOWS)
  lfl_add_target(app_windows_os STATIC_LIBRARY SOURCES windows_os.cpp
                 LINK_LIBRARIES Ws2_32.lib Dnsapi.lib opengl32.lib glu32.lib strmiids.lib zlib)
  lfl_add_target(app_windows_framework STATIC_LIBRARY SOURCES windows_framework.cpp)
endif()

if(LFL_QT)
  lfl_enable_qt()
  set(CMAKE_AUTOMOC ON)
  lfl_add_target(app_qt_framework STATIC_LIBRARY SOURCES qt_framework.cpp
                 INCLUDE_DIRECTORIES ${QT_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} LINK_LIBRARIES ${QT_LIB})
  set(CMAKE_AUTOMOC OFF)
endif()

if(LFL_WXWIDGETS)
  lfl_add_target(app_wxwidgets_framework STATIC_LIBRARY SOURCES wxwidgets_framework.cpp
                 INCLUDE_DIRECTORIES ${WXWIDGETS_INCLUDE} LINK_LIBRARIES ${WXWIDGETS_LIB})
endif()

if(LFL_GLFW)
  lfl_add_target(app_glfw_framework STATIC_LIBRARY SOURCES glfw_framework.cpp
                 INCLUDE_DIRECTORIES ${GLFW_INCLUDE} LINK_LIBRARIES ${GLFW_LIB})
endif()

if(LFL_SDL)
  lfl_add_target(app_sdl_framework STATIC_LIBRARY SOURCES sdl_framework.cpp
                 INCLUDE_DIRECTORIES ${SDL_INCLUDE} LINK_LIBRARIES ${SDL_LIB})
endif()
