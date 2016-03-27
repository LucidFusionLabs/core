# $Id: CMakeLists.txt 1327 2014-11-03 23:26:43Z justin $

set(LFL_APP_CFLAGS)
set(LFL_APP_DEF ${PLATFORM_DEF})
set(LFL_APP_INCLUDE ${LFL_SOURCE_DIR} ${PLATFORM_INCLUDE})
set(LFL_APP_LIB lfl_app app_font)

macro(add_app_def _def)
  if(${_def})
    set(LFL_APP_DEF ${LFL_APP_DEF} -D${_def})
  endif(${_def})
endmacro()

macro(add_app_lib _def _inc _lib)
  if(${_def})
    set(LFL_APP_DEF ${LFL_APP_DEF} -D${_def})
    set(LFL_APP_INCLUDE ${LFL_APP_INCLUDE} ${_inc})
    set(LFL_APP_LIB ${LFL_APP_LIB} ${_lib})
  endif(${_def})
endmacro()

add_app_def(LFL32)
add_app_def(LFL64)
add_app_def(LFL_DEBUG)
add_app_def(LFL_APPLE)
add_app_def(LFL_LINUX)
add_app_def(LFL_WINDOWS)
add_app_def(LFL_MOBILE)
add_app_def(LFL_ANDROID)
add_app_def(LFL_IPHONE)
add_app_def(LFL_IPHONESIM)
add_app_def(LFL_EMSCRIPTEN)

add_app_lib(LFL_GLOG "${GLOG_INCLUDE}" "${GLOG_LIB}")
add_app_lib(LFL_FLATBUFFERS "${FLATBUFFERS_INCLUDE_DIR}" "${FLATBUFFERS_LIB}")

if(LFL_QT)
  set(LFL_APP_DEF ${LFL_APP_DEF} ${QT_DEF})
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(LFL_APP_CFLAGS ${LFL_APP_CFLAGS} -fPIC)
  endif()
endif()

if(LFL_FLATBUFFERS)
  FLATBUFFERS_GENERATE_C_HEADERS(FLATBUFFERS_SRC ipc.fbs)
endif()

add_subdirectory(framework)
add_subdirectory(types)
add_subdirectory(math)
add_subdirectory(graphics)
add_subdirectory(font)
add_subdirectory(audio)
add_subdirectory(loader)
add_subdirectory(net)
add_subdirectory(crypto)
add_subdirectory(browser)
add_subdirectory(bindings)
add_subdirectory(camera)
add_subdirectory(cuda)

add_library(lfl_app ${LFL_APP_TYPE} gui.cpp browser.cpp font.cpp flow.cpp app.cpp types/types.cpp
            types/string.cpp types/time.cpp "file.cpp" assets.cpp loader.cpp audio.cpp video.cpp input.cpp
            math.cpp ipc.cpp network.cpp net/http.cpp net/ssh.cpp net/smtp.cpp net/resolver.cpp
            camera.cpp scene.cpp shell.cpp crypto.cpp bindings.cpp
            ${PROTO_SRC} ${CAPNP_SRC} ${FLATBUFFERS_SRC_OUTPUTS})

target_compile_options(lfl_app PUBLIC ${LFL_APP_CFLAGS})
target_compile_definitions(lfl_app PUBLIC ${LFL_APP_DEF})
target_include_directories(lfl_app PUBLIC ${LFL_APP_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR})

add_precompiled_header(core/app/app.h c++-header lfl_app)
target_use_precompiled_header(lfl_app core/app/app.h FORCEINCLUDE)

if(LFL_FLATBUFFERS)
  add_dependencies(lfl_app flatbuffers flatc ${FLATBUFFERS_SRC})
endif()

if(LFL_APP_TYPE MATCHES "^SHARED$")
  target_link_libraries(lfl_app ${LFL_APP_LIB})
  set(LFL_APP_LIB lfl_app)
endif()

set(LFL_APP_CFLAGS ${LFL_APP_CFLAGS} PARENT_SCOPE)
set(LFL_APP_DEF ${LFL_APP_DEF} PARENT_SCOPE)
set(LFL_APP_INCLUDE ${LFL_APP_INCLUDE} PARENT_SCOPE)
set(LFL_APP_LIB ${LFL_APP_LIB} PARENT_SCOPE)
set(LFL_APP_ASSETS ${LFL_SOURCE_DIR}/core/app/shaders/default.*
    ${LFL_SOURCE_DIR}/core/app/assets/MenuAtlas*
    ${LFL_SOURCE_DIR}/core/app/assets/MobileAtlas* PARENT_SCOPE)

