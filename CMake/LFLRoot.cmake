# Lucid Fusion Labs Root Make File
CMAKE_POLICY(SET CMP0004 OLD)
# MESSAGE(FATAL_ERROR "Debug message")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif(NOT CMAKE_BUILD_TYPE)

if(LFL_CORE_ONLY)
   set(CORE_SUBDIR .)
else(LFL_CORE_ONLY)
   set(CORE_SUBDIR core)
endif(LFL_CORE_ONLY)

include(ExternalProject)
include(BundleUtilities)
include(${CMAKE_CURRENT_SOURCE_DIR}/${CORE_SUBDIR}/CMake/Autoconf.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/${CORE_SUBDIR}/CMake/util.cmake)
set(LFL_CORE ${CMAKE_CURRENT_SOURCE_DIR}/${CORE_SUBDIR})
set(LFL_COREBIN ${CMAKE_CURRENT_BINARY_DIR}/${CORE_SUBDIR})

if(WIN32)
    link_directories("")
    set(CMAKE_EXE_LINKER_FLAGS "/SAFESEH:NO /NODEFAULTLIB:LIBCMTD")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11") # -stdlib=libc++")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wno-deprecated-declarations") # -stdlib=libc++")
    #set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++")
    #set(CMAKE_SHARED_LINKER_FLAGS "-stdlib=libc++")
    #set(CMAKE_MODULE_LINKER_FLAGS "-stdlib=libc++")
    if(LFL_IPHONE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -stdlib=libc++")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -stdlib=libc++")
    endif(LFL_IPHONE)
endif(WIN32)
add_definitions("-D__STDC_CONSTANT_MACROS")

# args
if(LFL_IPHONESIM)
    set(LFL_IPHONE 1)
    set(IPHONESIM "-Simulator")
endif(LFL_IPHONESIM)

# imports
add_subdirectory(${CORE_SUBDIR}/imports)

# macro includes
if(LFL_PROTOBUF)
    include(${CMAKE_CURRENT_SOURCE_DIR}/${CORE_SUBDIR}/CMake/FindProtoBuf.cmake)
endif(LFL_PROTOBUF)

# macros
macro(lfl_project _name)
    set(LFL_PROJECT ${_name})
    set(LFL_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(LFAPP_LIB_TYPE STATIC)
    if(LFL_QT)
        set(CMAKE_AUTOMOC ON)
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
    endif(LFL_QT)
endmacro(lfl_project)

macro(lfl_clear_projects)
    set(LFAPP_DEF)
    set(LFAPP_INCLUDE)
    set(LFAPP_LIB)
    set(LFL_ASSIMP)
    set(LFL_AUDIOUNIT)
    set(LFL_BERKELIUM)
    set(LFL_BOX2D)
    set(LFL_CAMERA)
    set(LFL_CLING)
    set(LFL_FFMPEG)
    set(LFL_FREETYPE)
    set(LFL_GEOIP)
    set(LFL_GIF)
    set(LFL_GLOG)
    set(LFL_GLES2)
    set(LFL_GLEW)
    set(LFL_GLFWVIDEO)
    set(LFL_GLFWINPUT)
    set(LFL_GTEST)
    set(LFL_HARFBUZZ)
    set(LFL_ICONV)
    set(LFL_JPEG)
    set(LFL_JUDY)
    set(LFL_LAME)
    set(LFL_LFLINPUT)
    set(LFL_LFLVIDEO)
    set(LFL_LIBCSS)
    set(LFL_LIBARCHIVE)
    set(LFL_LUA)
    set(LFL_OPENCV)
    set(LFL_OPENSSL)
    set(LFL_OSXINPUT)
    set(LFL_OSXVIDEO)
    set(LFL_PCAP)
    set(LFL_PNG)
    set(LFL_PROTOBUF)
    set(LFL_PORTAUDIO)
    set(LFL_QT)
    set(LFL_REGEX)
    set(LFL_SDLINPUT)
    set(LFL_SDLVIDEO)
    set(LFL_SREGEX)
    set(LFL_V8JS)
    set(LFL_WXWIDGETS)
    set(LFL_X264)
endmacro(lfl_clear_projects)
lfl_clear_projects()

# cuda
if(LFL_CUDA)
    add_subdirectory(${CORE_SUBDIR}/lfcuda)
endif(LFL_CUDA)

# lfapp unit tests
add_subdirectory(${CORE_SUBDIR}/lfapp_tests)

# crawler
add_subdirectory(${CORE_SUBDIR}/crawler)

# nlp
add_subdirectory(${CORE_SUBDIR}/nlp)

# speech
add_subdirectory(${CORE_SUBDIR}/speech)