# $Id: CMakeLists.txt 1327 2014-11-03 23:26:43Z justin $

macro(lfapp_def _def)
    if(${_def})
    set(LFAPP_DEF ${LFAPP_DEF} -D${_def})
    endif(${_def})
endmacro(lfapp_def)

macro(lfapp_lib _def _inc _lib)
    if(${_def})
        set(LFAPP_DEF ${LFAPP_DEF} -D${_def})
        set(LFAPP_LIB ${LFAPP_LIB} ${_lib})
        set(LFAPP_INCLUDE ${LFAPP_INCLUDE} ${_inc})
    endif(${_def})
endmacro(lfapp_lib)

# init
set(LFAPP_INCLUDE ${LFL_CORE} ${LFAPP_INCLUDE} ${PLATFORM_INCLUDE} ${OPENGL_INCLUDE})
set(LFAPP_LIB ${LFL_PROJECT}_lfapp ${OPENGL_LIB})
set(LFAPP_DEF -DLFL_BUILD ${PLATFORM_DEF} ${WXWIDGETS_DEF})

if(LFL_LFLINPUT)
if(LFL_IPHONE)
    set(LFL_IPHONEINPUT 1)
elseif(LFL_ANDROID)
    set(LFL_ANDROIDINPUT 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(LFL_OSXINPUT 1)
endif(LFL_IPHONE)
endif(LFL_LFLINPUT)

if(LFL_LFLVIDEO)
if(LFL_IPHONE)
    set(LFL_IPHONEVIDEO 1)
    set(LFL_IPHONEINPUT 1)
elseif(LFL_ANDROID)
    set(LFL_ANDROIDVIDEO 1)
    set(LFL_ANDROIDINPUT 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(LFL_OSXVIDEO 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(LFL_LINUXVIDEO 1)
endif(LFL_IPHONE)
endif(LFL_LFLVIDEO)

lfapp_def(LFL_HEADLESS)
lfapp_def(LFL_TEST)
lfapp_def(LFL_GLES2)
lfapp_def(LFL_CAMERA)
lfapp_def(LFL_GLFWVIDEO)
lfapp_def(LFL_SDLAUDIO)
lfapp_def(LFL_SDLVIDEO)
lfapp_def(LFL_OSXVIDEO)
lfapp_def(LFL_OSXINPUT)
lfapp_def(LFL_IPHONE)
lfapp_def(LFL_IPHONESIM)
lfapp_def(LFL_IPHONEVIDEO)
lfapp_def(LFL_IPHONEINPUT)
lfapp_def(LFL_ANDROID)
lfapp_def(LFL_ANDROIDVIDEO)
lfapp_def(LFL_ANDROIDINPUT)
lfapp_def(LFL_LINUXVIDEO)
lfapp_def(LFL_OPENSL)
lfapp_def(LFL_EPOLL)
lfapp_def(LFL_IPC)

if(LFL_QT)
    set(LFAPP_DEF ${LFAPP_DEF} ${QT_DEF})
endif(LFL_QT)

lfapp_lib(LFL_BOOST "${BOOST_INCLUDE}" "${BOOST_LIB}")
lfapp_lib(LFL_QT "${QT_INCLUDE}" "${QT_LIB}")
lfapp_lib(LFL_WXWIDGETS "${WXWIDGETS_INCLUDE}" "${WXWIDGETS_LIB}")
lfapp_lib(LFL_GLEW "${GLEW_INCLUDE}" "${GLEW_LIB}")
lfapp_lib(LFL_GLFWINPUT "${GLFW_INCLUDE}" "${GLFW_LIB}")
lfapp_lib(LFL_SDLINPUT "${SDL_INCLUDE}" "${SDL_LIB}")
lfapp_lib(LFL_LIBARCHIVE "${ARCHIVE_INCLUDE}" "${ARCHIVE_LIB}")
lfapp_lib(LFL_FREETYPE "${FREETYPE_INCLUDE}" "${FREETYPE_LIB}")
lfapp_lib(LFL_ASSIMP "${ASSIMP_INCLUDE}" "${ASSIMP_LIB}")
lfapp_lib(LFL_HARFBUZZ "${HARFBUZZ_INCLUDE}" "${HARFBUZZ_LIB}")
lfapp_lib(LFL_LIBCSS "${LIBCSS_INCLUDE}" "${LIBCSS_LIB}")
lfapp_lib(LFL_FFMPEG "${FFMPEG_INCLUDE}" "${FFMPEG_LIB}")
lfapp_lib(LFL_PORTAUDIO "${PORTAUDIO_INCLUDE}" "${PORTAUDIO_LIB}")
lfapp_lib(LFL_AUDIOQUEUE "" "-framework AudioToolbox")
lfapp_lib(LFL_AUDIOUNIT "" "-framework AudioToolbox")
lfapp_lib(LFL_X264 "" "${X264_LIB}")
lfapp_lib(LFL_LAME "" "${LAME_LIB}")
lfapp_lib(LFL_OPENCV "${OPENCV_INCLUDE}" "${OPENCV_LIB}")
lfapp_lib(LFL_REGEX "${REGEX_INCLUDE}" "${REGEX_LIB}")
lfapp_lib(LFL_SREGEX "${SREGEX_INCLUDE}" "${SREGEX_LIB}")
lfapp_lib(LFL_JUDY "${JUDY_INCLUDE}" "${JUDY_LIB}")
lfapp_lib(LFL_BULLET "${BULLET_INCLUDE}" "${BULLET_LIB}")
lfapp_lib(LFL_BOX2D "${BOX2D_INCLUDE}" "${BOX2D_LIB}")
lfapp_lib(LFL_ODE "${ODE_INCLUDE}" "${ODE_LIB}")
lfapp_lib(LFL_BERKELIUM "${BERKELIUM_INCLUDE}" "${BERKELIUM_LIB}")
lfapp_lib(LFL_OPENSSL "" "-lssl -lcrypto")
lfapp_lib(LFL_CUDA "${CUDA_INCLUDE}" "${CUDA_LIB} lfcuda")
lfapp_lib(LFL_PNG "${PNG_INCLUDE}" "${PNG_LIB}")
lfapp_lib(LFL_JPEG "${JPEG_INCLUDE}" "${JPEG_LIB}")
lfapp_lib(LFL_GIF "${GIF_INCLUDE}" "${GIF_LIB}")
lfapp_lib(LFL_ICONV "${ICONV_INCLUDE}" "${ICONV_LIB}")
lfapp_lib(LFL_PROTOBUF "${PROTOBUF_INCLUDE_DIR}" "${PROTOBUF_LIBRARY}")
lfapp_lib(LFL_GLOG "${GLOG_INCLUDE}" "${GLOG_LIB}")
lfapp_lib(LFL_TCMALLOC "${TCMALLOC_INCLUDE}" "${TCMALLOC_LIB}")
lfapp_lib(LFL_PCAP "${PCAP_INCLUDE}" "${PCAP_LIB}")
lfapp_lib(LFL_GEOIP "${GEOIP_INCLUDE}" "${GEOIP_LIB}")
lfapp_lib(LFL_TINYJS "${TINYJS_INCLUDE}" "${TINYJS_LIB}")
lfapp_lib(LFL_V8JS "${V8JS_INCLUDE}" "${V8JS_LIB}")
lfapp_lib(LFL_LUA "${LUA_INCLUDE}" "${LUA_LIB}")
lfapp_lib(LFL_CLING "${CLING_INCLUDE}" "${CLING_LIB}")

set(LFAPP_LIB ${LFAPP_LIB} ${PLATFORM_LIB})

foreach(_current ${LFAPP_INCLUDE})
    set(LFAPP_INC ${LFAPP_INC} "-I${_current}")
endforeach()

# mac
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    if(NOT LFL_ANDROID)
        add_subdirectory(lfobjc)
        set(LFAPP_LIB ${LFAPP_LIB} ${LFL_PROJECT}_lfobjc)
    endif(NOT LFL_ANDROID)
    if(LFL_CAMERA AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(LFAPP_LIB ${LFAPP_LIB} ${LFL_PROJECT}_lfobjc "-framework QTKit -framework CoreVideo")
    endif(LFL_CAMERA AND CMAKE_SIZEOF_VOID_P EQUAL 8)
endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")

# proto
# PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HDR game.proto)

add_library(${LFL_PROJECT}_lfapp ${LFAPP_LIB_TYPE} gui.cpp font.cpp flow.cpp lfapp.cpp lftypes.cpp string.cpp time.cpp "file.cpp" assets.cpp audio.cpp video.cpp input.cpp math.cpp ipc.cpp network.cpp wire.cpp resolver.cpp camera.cpp scene.cpp ${PROTO_SRC})
target_include_directories(${LFL_PROJECT}_lfapp PUBLIC ${LFAPP_INCLUDE} ${LFL_PROJECT_DIR}/${LFL_PROJECT}_lfapp_obj)
target_compile_definitions(${LFL_PROJECT}_lfapp PUBLIC ${LFAPP_DEF})
if(LFAPP_LIB_TYPE MATCHES "^SHARED$")
    target_link_libraries(${LFL_PROJECT}_lfapp ${LFAPP_LIB})
    set(LFAPP_LIB ${LFL_PROJECT}_lfapp)
endif(LFAPP_LIB_TYPE MATCHES "^SHARED$")

# lfapp deps
macro(lfapp_dep _def _lib)
    if(${_def})
        add_dependencies(${LFL_PROJECT}_lfapp ${_lib})
    endif(${_def})
endmacro(lfapp_dep)
lfapp_dep(LFL_BOX2D Box2D)
lfapp_dep(LFL_OPENCV OpenCV)
lfapp_dep(LFL_PNG png)

# post build step
if(LFL_IPHONE)
    macro(lfapp_add_target target)
        add_executable(${target} ${ARGN})
        target_link_libraries(${target} ${LFAPP_LIB})
    endmacro(lfapp_add_target)
    macro(lfapp_post_build_start target pkgname)
        set(bin i${pkgname}.app/Contents/MacOS/${target}) 
        set(info_plist ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist)
        add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND rm -rf   i${pkgname}.app
            COMMAND mkdir -p i${pkgname}.app/Contents
            COMMAND cp  ${info_plist} i${pkgname}.app/Contents/Info.plist
            COMMAND cat ${info_plist} | grep -A1 CFBundlePackageType | tail -1 | cut -f2 -d\\> | cut -f1 -d \\< | tr -d '\\n' | tee    i${pkgname}.app/Contents/PkgInfo
            COMMAND cat ${info_plist} | grep -A1 CFBundleSignature   | tail -1 | cut -f2 -d\\> | cut -f1 -d \\< | tr -d '\\n' | tee -a i${pkgname}.app/Contents/PkgInfo
            COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/assets i${pkgname}.app
            COMMAND cp ${LFL_CORE}/lfapp/*.glsl i${pkgname}.app/assets
            COMMAND cp ${target} i${pkgname}.app
            COMMAND chmod +x ${target} i${pkgname}.app/${target}
            COMMAND mkdir -p i${pkgname}.app/_CodeSignature
            COMMAND touch i${pkgname}.app/_CodeSignature/CodeResources
            COMMAND ln -s _CodeSignature/CodeResources i${pkgname}.app/CodeResources
            COMMAND codesign -f -s "iPhone Developer" i${pkgname}.app)
        add_custom_target(${target}_pkg WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND rm -rf   Payload
            COMMAND mkdir -p Payload
            COMMAND cp -rp i${pkgname}.app Payload
            COMMAND zip -r i${pkgname}.ipa Payload)
    endmacro(lfapp_post_build_start)
    macro(lfapp_post_build_copy_bin target pkgname)
    endmacro(lfapp_post_build_copy_bin)
elseif(LFL_ANDROID)
    macro(lfapp_add_target target)
        add_library(${target} ${ARGN})
        target_link_libraries(${target} ${LFL_PROJECT}_lfapp)
        set(${target}_lfapp_obj ${CMAKE_CURRENT_BINARY_DIR}/${LFL_PROJECT}_lfapp_obj/lib${LFL_PROJECT}_lfapp.a)
        set(${target}_deps ${LFL_CORE}/lfapp/lfjni/lfjni.cpp ${${target}_lfapp_obj})
        set_source_files_properties(${${target}_lfapp_obj} PROPERTIES HEADER_FILE_ONLY TRUE)
        set_source_files_properties(${LFL_CORE}/lfapp/lfjni/lfjni.cpp PROPERTIES HEADER_FILE_ONLY TRUE)
        set_source_files_properties(${ARGV1} PROPERTIES OBJECT_DEPENDS "${${target}_deps}")
    endmacro(lfapp_add_target)
    macro(lfapp_post_build_start target pkgname)
        add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android/jni
            COMMAND ${ANDROIDNDK}/ndk-build
            COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android/res/raw
            COMMAND cp ${LFL_CORE}/lfapp/*.glsl ${CMAKE_CURRENT_SOURCE_DIR}/assets
            COMMAND if [ -f ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.wav ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.wav ../res/raw\; fi
            COMMAND if [ -f ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.mp3 ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.mp3 ../res/raw\; fi)
        add_custom_target(${target}_pkg_debug WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
            COMMAND "ANDROID_HOME=${ANDROIDSDK}" ${GRADLEBIN} assembleDebug)
        add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
            COMMAND "ANDROID_HOME=${ANDROIDSDK}" ${GRADLEBIN} uninstallDebug
            COMMAND "ANDROID_HOME=${ANDROIDSDK}" ${GRADLEBIN}   installDebug
            COMMAND ${ANDROIDSDK}/platform-tools/adb shell am start -n `${ANDROIDSDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep package | cut -d\\' -f2`/`${ANDROIDSDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep launchable-activity | cut -d\\' -f2`
            COMMAND ${ANDROIDSDK}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)
        add_custom_target(${target}_debug_start WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
            COMMAND ${ANDROIDSDK}/platform-tools/adb shell am start -n `${ANDROIDSDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep package | cut -d\\' -f2`/`${ANDROIDSDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep launchable-activity | cut -d\\' -f2`
            COMMAND ${ANDROIDSDK}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)
    endmacro(lfapp_post_build_start)
    macro(lfapp_post_build_copy_bin target pkgname)
    endmacro(lfapp_post_build_copy_bin)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    macro(lfapp_add_target target)
        add_executable(${target} ${ARGN})
        target_link_libraries(${target} ${LFAPP_LIB})
    endmacro(lfapp_add_target)
    macro(lfapp_post_build_start target pkgname)
        set(pa_lib ../imports/portaudio/lib/.libs/libportaudio.so.2)
        set(mp3_lib ../imports/lame/libmp3lame/.libs/libmp3lame.so.0)
        set(x264_lib ../imports/x264/libx264.so.142)
        set(cv_lib ../imports/OpenCV/lib/libcv.so.2.1)
        set(cx_lib ../imports/OpenCV/lib/libcxcore.so.2.1)
        add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND rm -rf   ${pkgname}
            COMMAND mkdir -p ${pkgname}
            COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/assets ${pkgname}
            COMMAND cp ${LFL_CORE}/lfapp/*.glsl ${pkgname}/assets
            COMMAND cp ${target} ${pkgname}
            COMMAND if [ -f ${pa_lib}   ]; then cp ${pa_lib}   ${pkgname}\; fi
            COMMAND if [ -f ${mp3_lib}  ]; then cp ${mp3_lib}  ${pkgname}\; fi
            COMMAND if [ -f ${x264_lib} ]; then cp ${x264_lib} ${pkgname}\; fi
            COMMAND if [ -f ${cv_lib}   ]; then cp ${cv_lib}   ${pkgname}\; fi
            COMMAND if [ -f ${cx_lib}   ]; then cp ${cx_lib}   ${pkgname}\; fi)
    endmacro(lfapp_post_build_start)
    macro(lfapp_post_build_copy_bin target pkgname)
    endmacro(lfapp_post_build_copy_bin)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    macro(lfapp_add_target target)
        add_executable(${target} ${ARGN})
        target_link_libraries(${target} ${LFAPP_LIB})
    endmacro(lfapp_add_target)
    macro(osx_lib defname varname filename)
        set(${varname} ${filename})
        if(${${defname}})
            set(if_${varname} 1)
        endif(${${defname}})
    endmacro(osx_lib)
    macro(osx_libs)
        osx_lib(LFL_PORTAUDIO pa_lib ../imports/portaudio/lib/.libs/libportaudio.2.dylib)
        osx_lib(LFL_FFMPEG mp3_lib ../imports/lame/libmp3lame/.libs/libmp3lame.0.dylib)
        osx_lib(LFL_OPENCV cx_lib ../imports/OpenCV/lib/libcxcore.2.1.dylib)
        osx_lib(LFL_OPENCV cv_lib ../imports/OpenCV/lib/libcv.2.1.dylib)
    endmacro(osx_libs)
    macro(lfapp_post_build_start target pkgname)
        osx_libs()
        set(bin ${pkgname}.app/Contents/MacOS/${target}) 
        set(lib ${pkgname}.app/Contents/Libraries) 
        set(res ${pkgname}.app/Contents/Resources) 
        set(info_plist ${CMAKE_CURRENT_SOURCE_DIR}/mac-Info.plist)
        add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND rm -rf   ${pkgname}.app
            COMMAND mkdir -p ${pkgname}.app/Contents/MacOS
            COMMAND mkdir -p ${pkgname}.app/Contents/Resources
            COMMAND mkdir -p ${pkgname}.app/Contents/Frameworks
            COMMAND mkdir -p ${pkgname}.app/Contents/Libraries
            COMMAND cp  ${info_plist} ${pkgname}.app/Contents/Info.plist
            COMMAND cat ${info_plist} | grep -A1 CFBundlePackageType | tail -1 | cut -f2 -d\\> | cut -f1 -d \\< | tr -d '\\n' | tee    ${pkgname}.app/Contents/PkgInfo
            COMMAND cat ${info_plist} | grep -A1 CFBundleSignature   | tail -1 | cut -f2 -d\\> | cut -f1 -d \\< | tr -d '\\n' | tee -a ${pkgname}.app/Contents/PkgInfo
            COMMAND cp  ${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.icns ${res}
            COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/assets ${res}
            COMMAND cp ${LFL_CORE}/lfapp/*.glsl ${res}/assets
            COMMAND cp ${target} ${pkgname}.app/Contents/MacOS
            COMMAND if [ ${if_pa_lib}  ]; then cp ${pa_lib}  ${lib}\; fi
            COMMAND if [ ${if_mp3_lib} ]; then cp ${mp3_lib} ${lib}\; fi
            COMMAND if [ ${if_cx_lib}  ]; then cp ${cx_lib}  ${lib}\; fi
            COMMAND if [ ${if_cv_lib}  ]; then cp ${cv_lib}  ${lib}\; fi
            COMMAND if [ ${if_pa_lib}  ]; then install_name_tool -change /usr/local/lib/libportaudio.2.dylib @loader_path/../Libraries/libportaudio.2.dylib ${bin}\; fi
            COMMAND if [ ${if_mp3_lib} ]; then install_name_tool -change /usr/local/lib/libmp3lame.0.dylib @loader_path/../Libraries/libmp3lame.0.dylib ${bin}\; fi
            COMMAND if [ ${if_cv_lib}  ]; then install_name_tool -change libcxcore.2.1.dylib @loader_path/../Libraries/libcxcore.2.1.dylib ${lib}/libcv.2.1.dylib\; fi
            COMMAND if [ ${if_cx_lib}  ]; then install_name_tool -change libcxcore.2.1.dylib @loader_path/../Libraries/libcxcore.2.1.dylib ${bin}\; fi
            COMMAND if [ ${if_cv_lib}  ]; then install_name_tool -change libcv.2.1.dylib @loader_path/../Libraries/libcv.2.1.dylib ${bin}\; fi)
        add_custom_target(${target}_pkg WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND if [ -d /Volumes/${pkgname} ]; then umount /Volumes/${pkgname}\; fi
            COMMAND rm -rf ${pkgname}.dmg ${pkgname}.sparseimage
            COMMAND hdiutil create -size 60m -type SPARSE -fs HFS+ -volname ${pkgname} -attach ${pkgname}.sparseimage
            COMMAND bless --folder /Volumes/${pkgname} --openfolder /Volumes/${pkgname}
            COMMAND cp -r ${pkgname}.app /Volumes/${pkgname}/
            COMMAND ln -s /Applications /Volumes/${pkgname}/.
            COMMAND hdiutil eject /Volumes/${pkgname}
            COMMAND hdiutil convert ${pkgname}.sparseimage -format UDBZ -o ${pkgname}.dmg)
        add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
            COMMAND ${pkgname}.app/Contents/MacOS/${target})
        add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
            COMMAND lldb -f ${pkgname}.app/Contents/MacOS/${target} -o run)
    endmacro(lfapp_post_build_start)
    macro(lfapp_post_build_copy_bin target pkgname)
        osx_libs()
        set(bin ${pkgname}.app/Contents/MacOS/${target})
        add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND cp ${target} ${pkgname}.app/Contents/MacOS
            COMMAND if [ ${if_pa_lib}  ]; then install_name_tool -change /usr/local/lib/libportaudio.2.dylib @loader_path/../Libraries/libportaudio.2.dylib ${bin}\; fi
            COMMAND if [ ${if_mp3_lib} ]; then install_name_tool -change /usr/local/lib/libmp3lame.0.dylib @loader_path/../Libraries/libmp3lame.0.dylib ${bin}\; fi
            COMMAND if [ ${if_cx_lib}  ]; then install_name_tool -change libcxcore.2.1.dylib @loader_path/../Libraries/libcxcore.2.1.dylib ${bin}\; fi
            COMMAND if [ ${if_cv_lib}  ]; then install_name_tool -change libcv.2.1.dylib @loader_path/../Libraries/libcv.2.1.dylib ${bin}\; fi)
    endmacro(lfapp_post_build_copy_bin)
else()
    macro(lfapp_post_build_start target pkgname)
    endmacro(lfapp_post_build_start)
    macro(lfapp_post_build_copy_bin target pkgname)
    endmacro(lfapp_post_build_copy_bin)
endif()

# export
set(LFAPP_INCLUDE ${LFAPP_INCLUDE} PARENT_SCOPE)
set(LFAPP_LIB ${LFAPP_LIB} PARENT_SCOPE)
set(LFAPP_DEF ${LFAPP_DEF} PARENT_SCOPE)
