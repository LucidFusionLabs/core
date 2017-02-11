# $Id: lfapp.h 1335 2014-12-02 04:13:46Z justin $
# Copyright (C) 2009 Lucid Fusion Labs

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

if(LFL_EMSCRIPTEN)
  function(lfl_post_build_copy_asset_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_copy_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_start target binname pkgname)
  endfunction()

  function(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${target}.html)
    set_target_properties(${target} PROPERTIES LINK_FLAGS
      "--embed-file assets -s USE_SDL=2 -s USE_LIBPNG=1 -s USE_ZLIB=1 -s TOTAL_MEMORY=20971520")

    add_custom_command(TARGET ${target} PRE_LINK WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND rm -rf assets
      COMMAND mkdir assets
      COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/assets/* assets
      COMMAND cp ${LFL_APP_ASSET_FILES} assets)
  endfunction()

elseif(LFL_ANDROID)
  function(lfl_post_build_copy_asset_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_copy_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_start target binname pkgname)
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android/jni
      COMMAND ${LFL_ANDROID_NDK}/ndk-build
      COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android/res/raw
      COMMAND cp ${LFL_APP_ASSET_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/assets
      COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/drawable-\*\; do dbn=`basename $$d`\; if [ -d ../res/$$dbn ]; then cp $$d/\* ../res/$$dbn\; fi\; done
      COMMAND if [ $$\(find ${CMAKE_CURRENT_SOURCE_DIR}/assets -name "*.wav" | wc -l\) != "0" ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.wav ../res/raw\; fi
      COMMAND if [ $$\(find ${CMAKE_CURRENT_SOURCE_DIR}/assets -name "*.mp3" | wc -l\) != "0" ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.mp3 ../res/raw\; fi
      COMMAND if [ $$\(find ${CMAKE_CURRENT_SOURCE_DIR}/assets -name "*.ogg" | wc -l\) != "0" ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.ogg ../res/raw\; fi)
    
    add_custom_target(${target}_release WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
      COMMAND "ANDROID_HOME=${LFL_ANDROID_SDK}" ${LFL_GRADLE_BIN} uninstallRelease
      COMMAND "ANDROID_HOME=${LFL_ANDROID_SDK}" ${LFL_GRADLE_BIN} assembleRelease
      COMMAND ${LFL_ANDROID_SDK}/platform-tools/adb install ./build/outputs/apk/${target}-android-release.apk
      COMMAND ${LFL_ANDROID_SDK}/platform-tools/adb shell am start -n `${LFL_ANDROID_SDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-release.apk | grep package | cut -d\\' -f2`/`${LFL_ANDROID_SDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-release.apk | grep launchable-activity | cut -d\\' -f2`
      COMMAND ${LFL_ANDROID_SDK}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)

    add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
      COMMAND "ANDROID_HOME=${LFL_ANDROID_SDK}" ${LFL_GRADLE_BIN} uninstallDebug
      COMMAND "ANDROID_HOME=${LFL_ANDROID_SDK}" ${LFL_GRADLE_BIN}   installDebug
      COMMAND ${LFL_ANDROID_SDK}/platform-tools/adb shell am start -n `${LFL_ANDROID_SDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep package | cut -d\\' -f2`/`${LFL_ANDROID_SDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep launchable-activity | cut -d\\' -f2`
      COMMAND ${LFL_ANDROID_SDK}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)

    add_custom_target(${target}_debug_start WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
      COMMAND ${LFL_ANDROID_SDK}/platform-tools/adb shell am start -n `${LFL_ANDROID_SDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep package | cut -d\\' -f2`/`${LFL_ANDROID_SDK}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep launchable-activity | cut -d\\' -f2`
      COMMAND ${LFL_ANDROID_SDK}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)

    add_custom_target(${target}_help WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND echo "Symbolicate with: ${LFL_ANDROID_NDK}/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/bin/arm-linux-androideabi-addr2line -C -f -e libapp.so 008bd340")
  endfunction()

  function(lfl_add_package target)
    lfl_add_target(${target} SHARED_LIBRARY ${ARGN})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME app)
  endfunction()

elseif(LFL_IOS)
  function(lfl_post_build_copy_asset_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_copy_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_start target binname pkgname)
    set(bin i${pkgname}.app/Contents/MacOS/${target}) 
    set(info_plist ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist)
    set(entitlements_plist ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Entitlements.plist)

    # For the Info.plist files, Xcode inserts BuildMachineOSBuild, DTCompiler, DTPlatformBuild,
    # DTPlatformVersion, DTSDKBuild, DTSDKName, DTXcode, DTXcodeBuild keys.
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND rm -rf ${pkgname}.dSYM i${pkgname}.app 
      COMMAND mkdir  i${pkgname}.app
      COMMAND cp ${info_plist} i${pkgname}.app/Info.plist
      COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/assets i${pkgname}.app
      COMMAND cp ${LFL_APP_ASSET_FILES} i${pkgname}.app/assets
      COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/Images/* i${pkgname}.app
      COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/drawable-\*\;                do if [ -d $$d ]; then cp -R $$d i${pkgname}.app\; fi\; done
      COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/\*.lproj\;  do if [ -d $$d ]; then cp -R $$d i${pkgname}.app\; fi\; done
      COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/\*.bundle\; do if [ -d $$d ]; then cp -R $$d i${pkgname}.app\; fi\; done
      COMMAND for f in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/Resources/\*\; do o=`basename $$f | sed s/xib$$/nib/`\; ${LFL_APPLE_DEVELOPER}/usr/bin/ibtool --warnings --errors --notices --compile ${CMAKE_CURRENT_BINARY_DIR}/i${pkgname}.app/$$o $$f\; done
      COMMAND dsymutil ${target} -o ${pkgname}.dSYM
      COMMAND cp ${target} i${pkgname}.app
      COMMAND if ! [ ${LFL_IOS_SIM} ]\; then codesign -f -s \"${LFL_IOS_CERT}\" --entitlements ${entitlements_plist} i${pkgname}.app\; fi)

    add_custom_target(${target}_pkg WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND rm -rf i${pkgname}.ipa
      COMMAND /usr/bin/xcrun -sdk iphoneos PackageApplication -v ${CMAKE_CURRENT_BINARY_DIR}/i${pkgname}.app
      -o ${CMAKE_CURRENT_BINARY_DIR}/i${pkgname}.ipa --sign \"${LFL_IOS_CERT}\" --embed ${LFL_IOS_PROVISION})

    add_custom_target(${target}_help WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND echo "Symbolicate with: atos -arch armv7 -o ./i${pkgname}.app/${target} -l 0xcc000 0x006cf99f")

    if(LFL_IOS_SIM)
      add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND if pgrep -f Simulator.app\; then echo\; else nohup /Applications/Xcode.app/Contents/Developer/Applications/Simulator.app/Contents/MacOS/Simulator & sleep 5\; fi
        COMMAND xcrun simctl install booted i${pkgname}.app || { tail -1 $ENV{HOME}/Library/Logs/CoreSimulator/CoreSimulator.log && false\; }
        COMMAND xcrun simctl launch booted `cat ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist | grep BundleIdentifier -A1 | tail -1 | cut -f2 -d\\> | cut -f1 -d \\<`
        COMMAND touch   `find $ENV{HOME}/Library/Developer/CoreSimulator/Devices/\\`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)\\`/data/Containers/Bundle/Application -name i${pkgname}.app`/${pkgname}.txt
        COMMAND tail -f `find $ENV{HOME}/Library/Developer/CoreSimulator/Devices/\\`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)\\`/data/Containers/Bundle/Application -name i${pkgname}.app`/${pkgname}.txt | tee debug.txt)

      add_custom_target(${target}_run_syslog WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND if pgrep iOS\ Simulator\; then echo\; else nohup /Applications/Xcode.app/Contents/Developer/Applications/iOS\ Simulator.app/Contents/MacOS/iOS\ Simulator & sleep 5\; fi
        COMMAND xcrun simctl install booted i${pkgname}.app || { tail -1 $ENV{HOME}/Library/Logs/CoreSimulator/CoreSimulator.log && false\; }
        COMMAND xcrun simctl launch booted `cat ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist | grep BundleIdentifier -A1 | tail -1 | cut -f2 -d\\> | cut -f1 -d \\<`
        COMMAND echo tail -f ~/Library/Logs/CoreSimulator/`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)`/system.log
        COMMAND      tail -f ~/Library/Logs/CoreSimulator/`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)`/system.log)

      add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND if pgrep iOS\ Simulator\; then echo\; else nohup /Applications/Xcode.app/Contents/Developer/Applications/iOS\ Simulator.app/Contents/MacOS/iOS\ Simulator & sleep 5\; fi
        COMMAND xcrun simctl install booted i${pkgname}.app || { tail -1 $ENV{HOME}/Library/Logs/CoreSimulator/CoreSimulator.log && false\; }
        COMMAND find $ENV{HOME}/Library/Developer/CoreSimulator/Devices/`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)`/data/Containers/Bundle/Application -name i${pkgname}.app
        COMMAND xcrun simctl launch booted `cat ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist | grep BundleIdentifier -A1 | tail -1 | cut -f2 -d\\> | cut -f1 -d \\<`
        COMMAND lldb -n ${target} -o cont)

    else()
      add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND ios-deploy --bundle i${pkgname}.app)
      add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND ios-deploy --debug --bundle i${pkgname}.app)
    endif()
  endfunction()

  function(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endfunction()

elseif(LFL_OSX)
  function(lfl_post_build_copy_asset_bin target binname pkgname)
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND cp ${binname} ${pkgname}.app/Contents/Resources/assets)
  endfunction()

  function(lfl_post_build_copy_bin target binname pkgname)
    set(bin ${pkgname}.app/Contents/MacOS/${target})

    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND cp ${target} ${pkgname}.app/Contents/MacOS
      COMMAND install_name_tool -change /usr/local/lib/libportaudio.2.dylib @loader_path/../Libraries/libportaudio.2.dylib ${bin}
      COMMAND install_name_tool -change /usr/local/lib/libmp3lame.0.dylib @loader_path/../Libraries/libmp3lame.0.dylib ${bin}
      COMMAND install_name_tool -change lib/libopencv_core.3.1.dylib @loader_path/../Libraries/libopencv_core.3.1.dylib ${bin}
      COMMAND install_name_tool -change lib/libopencv_imgproc.3.1.dylib @loader_path/../Libraries/libopencv_imgproc.3.1.dylib ${bin}
      COMMAND codesign -f -s \"${LFL_OSX_CERT}\" ${pkgname}.app/Contents/MacOS/${target})

    add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND ${pkgname}.app/Contents/MacOS/${target})

    add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND lldb -f ${pkgname}.app/Contents/MacOS/${target} -o run)
  endfunction()

  function(lfl_post_build_start target binname pkgname)
    set(bin ${pkgname}.app/Contents/MacOS/${target}) 
    set(lib ${pkgname}.app/Contents/Libraries) 
    set(res ${pkgname}.app/Contents/Resources) 
    set(info_plist ${CMAKE_CURRENT_SOURCE_DIR}/mac-Info.plist)
    if(LFL_APP_LIB_FILES)
      set(copy_lfl_app_lib_files 1)
    endif()
    if(LFL_APP_ASSET_FILES)
      set(copy_lfl_app_asset_files 1)
    endif()
    if(LFL_APP_ASSET_DIRS)
      set(copy_lfl_app_asset_dirs 1)
    endif()

    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND rm -rf   ${pkgname}.app
      COMMAND mkdir -p ${pkgname}.app/Contents/MacOS
      COMMAND mkdir -p ${pkgname}.app/Contents/Resources
      COMMAND mkdir -p ${pkgname}.app/Contents/Frameworks
      COMMAND mkdir -p ${pkgname}.app/Contents/Libraries
      COMMAND if [ -f ${info_plist} ]; then cp  ${info_plist} ${pkgname}.app/Contents/Info.plist\; fi
      COMMAND if [ -f ${info_plist} ]; then cat ${info_plist} | grep -A1 CFBundlePackageType | tail -1 | cut -f2 -d\\> | cut -f1 -d \\< | tr -d '\\n' | tee    ${pkgname}.app/Contents/PkgInfo\; fi
      COMMAND if [ -f ${info_plist} ]; then cat ${info_plist} | grep -A1 CFBundleSignature   | tail -1 | cut -f2 -d\\> | cut -f1 -d \\< | tr -d '\\n' | tee -a ${pkgname}.app/Contents/PkgInfo\; fi
      COMMAND if [ -f ${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.icns ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.icns ${res}\; fi
      COMMAND if [ -d ${CMAKE_CURRENT_SOURCE_DIR}/assets ]; then cp -r ${CMAKE_CURRENT_SOURCE_DIR}/assets ${res}\; else mkdir ${res}/assets\; fi
      COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/drawable-\*\; do if [ -d $$d ]; then cp -R $$d ${res}\; fi\; done
      COMMAND if [ ${copy_lfl_app_asset_files} ]; then cp ${LFL_APP_ASSET_FILES} ${res}/assets\; fi
      COMMAND cp ${target} ${pkgname}.app/Contents/MacOS
      COMMAND if [ ${copy_lfl_app_lib_files} ]; then cp ${LFL_APP_LIB_FILES} ${lib}\; fi
      COMMAND if [ ${copy_lfl_app_asset_dirs} ]; then for d in ${LFL_APP_ASSET_DIRS}\; do cp -R $$d ${res}/assets\; done\; fi
      COMMAND install_name_tool -change /usr/local/lib/libportaudio.2.dylib @loader_path/../Libraries/libportaudio.2.dylib ${bin}
      COMMAND install_name_tool -change /usr/local/lib/libmp3lame.0.dylib @loader_path/../Libraries/libmp3lame.0.dylib ${bin}
      COMMAND if [ -f ${lib}/libopencv_imgproc.3.1.dylib ]; then install_name_tool -change lib/libopencv_core.3.1.dylib @loader_path/../Libraries/libopencv_core.3.1.dylib ${lib}/libopencv_imgproc.3.1.dylib\; fi
      COMMAND install_name_tool -change lib/libopencv_core.3.1.dylib @loader_path/../Libraries/libopencv_core.3.1.dylib ${bin}
      COMMAND install_name_tool -change lib/libopencv_imgproc.3.1.dylib @loader_path/../Libraries/libopencv_imgproc.3.1.dylib ${bin})

    add_custom_target(${target}_pkg WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND codesign -f -s \"${LFL_OSX_CERT}\" ${pkgname}.app/Contents/MacOS/${binname}
      COMMAND if [ -d /Volumes/${pkgname} ]; then umount /Volumes/${pkgname}\; fi
      COMMAND rm -rf ${pkgname}.dmg ${pkgname}.sparseimage
      COMMAND hdiutil create -size 60m -type SPARSE -fs HFS+ -volname ${pkgname} -attach ${pkgname}.sparseimage
      COMMAND bless --folder /Volumes/${pkgname} --openfolder /Volumes/${pkgname}
      COMMAND cp -r ${pkgname}.app /Volumes/${pkgname}/
      COMMAND ln -s /Applications /Volumes/${pkgname}/.
      COMMAND hdiutil eject /Volumes/${pkgname}
      COMMAND hdiutil convert ${pkgname}.sparseimage -format UDBZ -o ${pkgname}.dmg
      COMMAND codesign -f -s \"${LFL_OSX_CERT}\" ${pkgname}.dmg)

    add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND ${pkgname}.app/Contents/MacOS/${target})

    add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND lldb -f ${pkgname}.app/Contents/MacOS/${target} -o run)

    if(LFL_ADD_BITCODE_TARGETS AND TARGET ${target}_designer)
      lfl_post_build_copy_bin(${target}_designer ${target}_designer ${pkgname})
    endif()
  endfunction()

  function(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endfunction()

elseif(LFL_WINDOWS)
  function(lfl_post_build_copy_asset_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_copy_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_start target binname pkgname)
  endfunction()

  function(lfl_add_package target)
    link_directories(${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    lfl_add_target(${target} EXECUTABLE WIN32 ${ARGN})
    add_dependencies(${target} zlib)
    if(LFL_JPEG)
      add_dependencies(${target} libjpeg)
    endif()
  endfunction()

elseif(LFL_LINUX)
  function(lfl_post_build_copy_asset_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_copy_bin target binname pkgname)
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND cp ${binname} ${pkgname})
  endfunction()

  function(lfl_post_build_start target binname pkgname)
    if(LFL_APP_LIB_FILES)
      set(copy_lfl_app_lib_files 1)
    endif()

    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND rm -rf ${pkgname}
      COMMAND mkdir -p ${pkgname}
      COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/assets ${pkgname}
      COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/drawable-*\; do if [ -d $$d ]; then cp -R $$d ${pkgname}\; fi\; done
      COMMAND cp ${LFL_APP_ASSET_FILES} ${pkgname}/assets
      COMMAND cp ${target} ${pkgname}
      COMMAND if [ ${copy_lfl_app_lib_files} ]; then cp ${LFL_APP_LIB_FILES} ${pkgname}\; fi)

    add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND ${pkgname}/${binname})
  endfunction()

  function(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endfunction()

else()
  function(lfl_post_build_copy_asset_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_copy_bin target binname pkgname)
  endfunction()

  function(lfl_post_build_start target binname pkgname)
  endfunction()

  function(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endfunction()
endif()
