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
  function(lfl_post_build_copy_asset_bin target dest_target)
  endfunction()

  function(lfl_post_build_copy_bin target dest_target)
  endfunction()

  function(lfl_post_build_start target)
  endfunction()

  macro(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${target}.html)
    set_target_properties(${target} PROPERTIES LINK_FLAGS
      "--embed-file assets -s USE_SDL=2 -s USE_LIBPNG=1 -s USE_ZLIB=1 -s TOTAL_MEMORY=20971520")

    add_custom_command(TARGET ${target} PRE_LINK WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND rm -rf assets
      COMMAND mkdir assets
      COMMAND cp ${${target}_ASSET_FILES} assets)
  endmacro()

elseif(LFL_ANDROID)
  function(lfl_post_build_copy_asset_bin target dest_target)
  endfunction()

  function(lfl_post_build_copy_bin target dest_target)
  endfunction()

  function(lfl_post_build_start target)
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android
      COMMAND mkdir -p assets
      COMMAND mkdir -p res/raw
      COMMAND cp ${${target}_ASSET_FILES} assets
      COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/drawable-*\; do dbn=`basename $$d`\; if [ -d res/$$dbn ]; then cp $$d/* res/$$dbn\; fi\; done
      COMMAND if [ $$\(find ${CMAKE_CURRENT_SOURCE_DIR}/assets -name "*.wav" | wc -l\) != "0" ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.wav res/raw\; fi
      COMMAND if [ $$\(find ${CMAKE_CURRENT_SOURCE_DIR}/assets -name "*.mp3" | wc -l\) != "0" ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.mp3 res/raw\; fi
      COMMAND if [ $$\(find ${CMAKE_CURRENT_SOURCE_DIR}/assets -name "*.ogg" | wc -l\) != "0" ]; then cp ${CMAKE_CURRENT_SOURCE_DIR}/assets/*.ogg res/raw\; fi)

    add_custom_target(${target}_release WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
      COMMAND "JAVA_HOME=${JAVA_HOME}" "ANDROID_HOME=${ANDROID_HOME}" ${GRADLE_HOME}/bin/gradle uninstallRelease
      COMMAND "JAVA_HOME=${JAVA_HOME}" "ANDROID_HOME=${ANDROID_HOME}" ${GRADLE_HOME}/bin/gradle assembleRelease
      COMMAND ${ANDROID_HOME}/platform-tools/adb install ./build/outputs/apk/${target}-android-release.apk
      COMMAND ${ANDROID_HOME}/platform-tools/adb shell am start -n `${ANDROID_HOME}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-release.apk | grep package | cut -d\\' -f2`/`${ANDROID_HOME}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-release.apk | grep launchable-activity | cut -d\\' -f2`
      COMMAND ${ANDROID_HOME}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)

    add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
      COMMAND "JAVA_HOME=${JAVA_HOME}" "ANDROID_HOME=${ANDROID_HOME}" ${GRADLE_HOME}/bin/gradle uninstallDebug
      COMMAND "JAVA_HOME=${JAVA_HOME}" "ANDROID_HOME=${ANDROID_HOME}" ${GRADLE_HOME}/bin/gradle   installDebug
      COMMAND ${ANDROID_HOME}/platform-tools/adb shell am start -n `${ANDROID_HOME}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep package | cut -d\\' -f2`/`${ANDROID_HOME}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep launchable-activity | cut -d\\' -f2`
      COMMAND ${ANDROID_HOME}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)

    add_custom_target(${target}_debug_start WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target}-android DEPENDS ${target}
      COMMAND ${ANDROID_HOME}/platform-tools/adb shell am start -n `${ANDROID_HOME}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep package | cut -d\\' -f2`/`${ANDROID_HOME}/build-tools/19.1.0/aapt dump badging ./build/outputs/apk/${target}-android-debug.apk | grep launchable-activity | cut -d\\' -f2`
      COMMAND ${ANDROID_HOME}/platform-tools/adb logcat | tee ${CMAKE_CURRENT_BINARY_DIR}/debug.txt)

    add_custom_target(${target}_help WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND echo "Symbolicate with: ${ANDROID_NDK_HOME}/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/bin/arm-linux-androideabi-addr2line -C -f -e libnative-lib.so 008bd340")
  endfunction()

  macro(lfl_add_package target)
    lfl_add_target(${target} SHARED_LIBRARY ${ARGN})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME native-lib)
  endmacro()

elseif(LFL_IOS)
  function(lfl_post_build_copy_asset_bin target dest_target)
  endfunction()

  function(lfl_post_build_copy_bin target dest_target)
  endfunction()

  function(lfl_post_build_start target)
  string(REPLACE ";" " " IOS_CERT "${LFL_IOS_CERT}")
  set_target_properties(${target} PROPERTIES
                        MACOSX_BUNDLE TRUE
                        XCODE_ATTRIBUTE_SKIP_INSTALL NO
                        XCODE_ATTRIBUTE_ENABLE_BITCODE FALSE
                        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "${IOS_CERT}"
                        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${LFL_IOS_TEAM}"
                        XCODE_ATTRIBUTE_PROVISIONING_PROFILE_SPECIFIER "${LFL_IOS_PROVISION_NAME}")

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist)
      set_target_properties(${target} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist)
    endif()

    if(LFL_XCODE)
      add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/BundleRoot/* "\${BUILT_PRODUCTS_DIR}/\${PRODUCT_NAME}.app"
        COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/\*.lproj\;  do if [ -d $$d ]; then cp -R $$d "\${BUILT_PRODUCTS_DIR}/\${PRODUCT_NAME}.app" \; fi\; done
        COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/\*.bundle\; do if [ -d $$d ]; then cp -R $$d "\${BUILT_PRODUCTS_DIR}/\${PRODUCT_NAME}.app" \; fi\; done
        COMMAND for f in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/Resources/\*\; do o=`basename $$f | sed s/xib$$/nib/`\; ${LFL_APPLE_DEVELOPER}/usr/bin/ibtool --warnings --errors --notices --compile "\${BUILT_PRODUCTS_DIR}/\${PRODUCT_NAME}.app/\$\$o" $$f\; done)
    else()
      set(should_sign)
      if(NOT LFL_IOS_SIM)
        set(should_sign 1)
      endif()
      add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND rm -rf ${target}.dSYM
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/BundleRoot/* $<TARGET_FILE_DIR:${target}>
        COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/\*.lproj\;  do if [ -d $$d ]; then cp -R $$d $<TARGET_FILE_DIR:${target}>\; fi\; done
        COMMAND for d in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/\*.bundle\; do if [ -d $$d ]; then cp -R $$d $<TARGET_FILE_DIR:${target}>\; fi\; done
        COMMAND for f in ${CMAKE_CURRENT_SOURCE_DIR}/${target}-iphone/Resources/\*\; do o=`basename $$f | sed s/xib$$/nib/`\; ${LFL_APPLE_DEVELOPER}/usr/bin/ibtool --warnings --errors --notices --compile $<TARGET_FILE_DIR:${target}>/$$o $$f\; done
        COMMAND dsymutil $<TARGET_FILE:${target}> -o ${target}.dSYM
        COMMAND if [ ${should_sign} ]\; then codesign -f -s \"${LFL_IOS_CERT}\"
        --entitlements ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Entitlements.plist $<TARGET_FILE_DIR:${target}>\; fi)
    endif()

    add_custom_target(${target}_pkg WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND rm -rf i${target}.ipa
      COMMAND /usr/bin/xcrun -sdk iphoneos PackageApplication -v $<TARGET_FILE_DIR:${target}>
      -o ${CMAKE_CURRENT_BINARY_DIR}/i${target}.ipa --sign \"${LFL_IOS_CERT}\" --embed ${LFL_IOS_PROVISION})

    add_custom_target(${target}_help WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND echo "Symbolicate with: atos -arch armv7 -o $<TARGET_FILE:${target}> -l 0xcc000 0x006cf99f")

    if(LFL_IOS_SIM)
      add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND if pgrep -f Simulator.app\; then echo\; else nohup /Applications/Xcode.app/Contents/Developer/Applications/Simulator.app/Contents/MacOS/Simulator & sleep 5\; fi
        COMMAND xcrun simctl install booted $<TARGET_FILE_DIR:${target}> || { tail -1 $ENV{HOME}/Library/Logs/CoreSimulator/CoreSimulator.log && false\; }
        COMMAND xcrun simctl launch booted `cat ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist | grep BundleIdentifier -A1 | tail -1 | cut -f2 -d\\> | cut -f1 -d \\<`
        COMMAND touch   `find $ENV{HOME}/Library/Developer/CoreSimulator/Devices/\\`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)\\`/data/Containers/Bundle/Application -name ${target}.app`/${target}.txt
        COMMAND tail -f `find $ENV{HOME}/Library/Developer/CoreSimulator/Devices/\\`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)\\`/data/Containers/Bundle/Application -name ${target}.app`/${target}.txt | tee debug.txt)

      add_custom_target(${target}_run_syslog WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND if pgrep iOS\ Simulator\; then echo\; else nohup /Applications/Xcode.app/Contents/Developer/Applications/iOS\ Simulator.app/Contents/MacOS/iOS\ Simulator & sleep 5\; fi
        COMMAND xcrun simctl install booted $<TARGET_FILE_DIR:${target}> || { tail -1 $ENV{HOME}/Library/Logs/CoreSimulator/CoreSimulator.log && false\; }
        COMMAND xcrun simctl launch booted `cat ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist | grep BundleIdentifier -A1 | tail -1 | cut -f2 -d\\> | cut -f1 -d \\<`
        COMMAND echo tail -f ~/Library/Logs/CoreSimulator/`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)`/system.log
        COMMAND      tail -f ~/Library/Logs/CoreSimulator/`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)`/system.log)

      add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND if pgrep iOS\ Simulator\; then echo\; else nohup /Applications/Xcode.app/Contents/Developer/Applications/iOS\ Simulator.app/Contents/MacOS/iOS\ Simulator & sleep 5\; fi
        COMMAND xcrun simctl install booted $<TARGET_FILE_DIR:${target}> || { tail -1 $ENV{HOME}/Library/Logs/CoreSimulator/CoreSimulator.log && false\; }
        COMMAND find $ENV{HOME}/Library/Developer/CoreSimulator/Devices/`xcrun simctl list | grep Booted | head -1 | cut -f2 -d\\\( -f2 | cut -f1 -d\\\)`/data/Containers/Bundle/Application -name ${target}.app
        COMMAND xcrun simctl launch booted `cat ${CMAKE_CURRENT_SOURCE_DIR}/iphone-Info.plist | grep BundleIdentifier -A1 | tail -1 | cut -f2 -d\\> | cut -f1 -d \\<`
        COMMAND lldb -n ${target} -o cont)

    else()
      add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND ios-deploy --bundle $<TARGET_FILE_DIR:${target}> 
        COMMAND zip deployed-`date +\"%Y-%m-%d_%H_%M_%S\"`-${target}.zip -r $<TARGET_FILE_DIR:${target}>)
      add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
        COMMAND ios-deploy --debug --bundle $<TARGET_FILE_DIR:${target}>)
    endif()
  endfunction()

  macro(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endmacro()

elseif(LFL_OSX)
  function(lfl_post_build_copy_asset_bin target dest_target)
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND cp $<TARGET_FILE:${target}> $<TARGET_FILE_DIR:${dest_target}>/../Resources/assets)
  endfunction()

  function(lfl_post_build_copy_bin target dest_target)
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND cp $<TARGET_FILE:${target}> $<TARGET_FILE_DIR:${dest_target}>/${target}
      COMMAND install_name_tool -change /usr/local/lib/libportaudio.2.dylib @loader_path/../Libraries/libportaudio.2.dylib $<TARGET_FILE_DIR:${dest_target}>/${target}
      COMMAND install_name_tool -change /usr/local/lib/libmp3lame.0.dylib @loader_path/../Libraries/libmp3lame.0.dylib $<TARGET_FILE_DIR:${dest_target}>/${target}
      COMMAND install_name_tool -change lib/libopencv_core.3.1.dylib @loader_path/../Libraries/libopencv_core.3.1.dylib $<TARGET_FILE_DIR:${dest_target}>/${target}
      COMMAND install_name_tool -change lib/libopencv_imgproc.3.1.dylib @loader_path/../Libraries/libopencv_imgproc.3.1.dylib $<TARGET_FILE_DIR:${dest_target}>/${target}
      COMMAND codesign -f -s \"${LFL_OSX_CERT}\" $<TARGET_FILE_DIR:${dest_target}>/${target})

    add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND $<TARGET_FILE_DIR:${dest_target}>/${target})

    add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND lldb -f $<TARGET_FILE_DIR:${dest_target}>/${target} -o run)
  endfunction()

  function(lfl_post_build_start target)
    set_target_properties(${target} PROPERTIES MACOSX_BUNDLE TRUE)
    set_target_properties(${target} PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME ${target})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mac-Info.plist)
      set_target_properties(${target} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/mac-Info.plist)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.icns)
      set_target_properties(${target} PROPERTIES MACOSX_BUNDLE_ICON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.icns)
    endif()

    if(${target}_LIB_FILES)
      set(copy_lfl_app_lib_files 1)
    endif()

    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND mkdir -p $<TARGET_FILE_DIR:${target}>/../Libraries
      COMMAND if [ ${copy_lfl_app_lib_files} ]; then cp ${${target}_LIB_FILES} $<TARGET_FILE_DIR:${target}>/../Libraries\; fi
      COMMAND install_name_tool -change /usr/local/lib/libportaudio.2.dylib @loader_path/../Libraries/libportaudio.2.dylib $<TARGET_FILE:${target}> 
      COMMAND install_name_tool -change /usr/local/lib/libmp3lame.0.dylib @loader_path/../Libraries/libmp3lame.0.dylib $<TARGET_FILE:${target}> 
      COMMAND if [ -f $<TARGET_FILE_DIR:${target}>/../Libraries/libopencv_imgproc.3.1.dylib ]; then install_name_tool -change lib/libopencv_core.3.1.dylib @loader_path/../Libraries/libopencv_core.3.1.dylib $<TARGET_FILE_DIR:${target}>/../Libraries/libopencv_imgproc.3.1.dylib\; fi
      COMMAND install_name_tool -change lib/libopencv_core.3.1.dylib @loader_path/../Libraries/libopencv_core.3.1.dylib $<TARGET_FILE:${target}> 
      COMMAND install_name_tool -change lib/libopencv_imgproc.3.1.dylib @loader_path/../Libraries/libopencv_imgproc.3.1.dylib $<TARGET_FILE:${target}> )

    add_custom_target(${target}_pkg WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND codesign -f -s \"${LFL_OSX_CERT}\" $<TARGET_FILE:${target}>
      COMMAND if [ -d /Volumes/${target} ]; then umount /Volumes/${target}\; fi
      COMMAND rm -rf ${target}.dmg ${target}.sparseimage
      COMMAND hdiutil create -size 60m -type SPARSE -fs HFS+ -volname ${target} -attach ${target}.sparseimage
      COMMAND bless --folder /Volumes/${target} --openfolder /Volumes/${target}
      COMMAND cp -r ${target}.app /Volumes/${target}/
      COMMAND ln -s /Applications /Volumes/${target}/.
      COMMAND hdiutil eject /Volumes/${target}
      COMMAND hdiutil convert ${target}.sparseimage -format UDBZ -o ${target}.dmg
      COMMAND codesign -f -s \"${LFL_OSX_CERT}\" ${target}.dmg)

    add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND $<TARGET_FILE:${target}>)

    add_custom_target(${target}_debug WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND lldb -f $<TARGET_FILE:${target}> -o run)

    if(LFL_ADD_BITCODE_TARGETS AND TARGET ${target}_designer)
      lfl_post_build_copy_bin(${target}_designer ${target}_designer ${target})
    endif()
  endfunction()

  macro(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endmacro()

elseif(LFL_WINDOWS)
  function(lfl_post_build_copy_asset_bin target dest_target)
  endfunction()

  function(lfl_post_build_copy_bin target dest_target)
  endfunction()

  function(lfl_post_build_start target)
  endfunction()

  macro(lfl_add_package target)
    link_directories(${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    lfl_add_target(${target} EXECUTABLE WIN32 ${ARGN})
    add_dependencies(${target} zlib)
    if(LFL_JPEG)
      add_dependencies(${target} libjpeg)
    endif()
  endmacro()

elseif(LFL_LINUX)
  function(lfl_post_build_copy_asset_bin target dest_target)
  endfunction()

  function(lfl_post_build_copy_bin target dest_target)
    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND cp $<TARGET_FILE:${target}> ${dest_target}.app/${target})
  endfunction()

  function(lfl_post_build_start target)
    if(${target}_LIB_FILES)
      set(copy_lfl_app_lib_files 1)
    endif()

    add_custom_command(TARGET ${target} POST_BUILD WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND rm -rf ${target}.app
      COMMAND mkdir -p ${target}.app/assets
      COMMAND cp $<TARGET_FILE:${target}> ${target}.app/${target}
      COMMAND cp ${${target}_ASSET_FILES} ${target}.app/assets
      COMMAND if [ ${copy_lfl_app_lib_files} ]; then cp ${${target}_LIB_FILES} ${target}.app\; fi)

    add_custom_target(${target}_run WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${target}
      COMMAND ${target}.app/${target})
  endfunction()

  macro(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endmacro()

else()
  function(lfl_post_build_copy_asset_bin target dest_target)
  endfunction()

  function(lfl_post_build_copy_bin target dest_target)
  endfunction()

  function(lfl_post_build_start target dest_target)
  endfunction()

  macro(lfl_add_package target)
    lfl_add_target(${target} EXECUTABLE ${ARGN})
  endmacro()
endif()
