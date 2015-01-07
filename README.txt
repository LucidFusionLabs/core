http://lucidfusionlabs.com/svn/lfl/README.txt
=============================================

OVERVIEW
--------

The "lfapp" API primarly consists of the: Application, Window, and Scene
classes, plus the 7 Modules: Audio, Video, Input, Assets, Network, Camera,
and CUDA.

Projects include:

- term:         LTerminal, a modern terminal
- editor:       LEditor, a text editor and IDE
- browser:      LBrowser, a full HTML4/CSS2 web browser with V8 javascript
- image:        LImage, an image and 3D-model manipulation utility
- fs:           Fusion Server, a speech and image recognition server
- fv:           Fusion Viewer, a speech and image recognition client
- market:       Financial data visualization and automated trading code
- spaceball:    Spaceball Future, a multiplayer 3d game
- cb:           Crystal Bawl, a geopacket visualization screensaver
- chess:        LChess, a magic bitboard chess engine and FICS client
- quake:        LQuake, a quake clone
- senators:     IRC bots with NLP capabilties

The following applies to any app derived from lfl/new_app_template.
See lfl/new_app_template/README to quick start your next app.


BUILDING
--------

svn co http://lucidfusionlabs.com/svn/lfl

LFL builds easily for Windows, Linux, Mac OSX, iPhone and Android.

The greatest stumbling block is missing a package dependency in lfl/imports.
Examine the cmake output carefully for errors, such as missing the yasm
assembler required to build x264.

* Replace "FusionViewer" and "fv" with "YourPackage" and "YourApp" to
build other apps.


BUILDING Windows
----------------

* use CMake 3.0.2

        select c:\lfl for source and binaries
        Configure
        uncheck USE_MSVC_RUNTIME_LIBRARY_DLL
        Generate

        start Visual Studio Command Prompt
        cd lfl\imports\judy\src
        build.bat

* use Visual Studio C++ 2013 Express
* Tools > Options > Text Editor > All Languages > Tabs > Insert Spaces

        c:\lfl\Project.sln
        Build FV

        cd c:\lfl\fv
        copy ..\debug\*.dll debug
        copy ..\lfapp\*.glsl assets
        copy ..\imports\berkelium\w32\bin\* Debug
        copy ..\imports\ffmpeg\w32\dll\*.dll Debug [overwrite:All]
        [Run]

        [Right click] fv.nsi > Compile NSIS Script

* Windows installer fvinst.exe results


BUILDING Linux
--------------

* if yasm < 1 install http://www.tortall.net/projects/yasm/releases/yasm-1.2.0.tar.gz

        cd lfl
        ./imports/build.sh
        touch crawler/crawler.pb.h
        touch crawler/crawler.pb.cc
        cmake .
        cd fv && make

        ./pkg/lin.sh
        export LD_LIBRARY_PATH=./FusionViewer
        ./FusionViewer/fv

        tar cvfz FusionViewer.tgz FusionViewer

* Linux package FusionViewer.tgz results


BUILDING Mac
------------

* http://www.cmake.org/files/v3.0/cmake-3.0.2-Darwin64-universal.dmg
* For C++ Interpreter setup ~/cling following http://root.cern.ch/drupal/content/cling-build-instructions
* For V8 Javascript setup ~/v8 following https://developers.google.com/v8/build then:

        sudo port install yasm
        export CXX="clang++ -std=c++11 -stdlib=libc++"
        export CC=clang
        export CPP="clang -E"
        export LINK="clang++ -std=c++11 -stdlib=libc++"
        export CXX_host=clang++
        export CC_host=clang
        export CPP_host="clang -E"
        export LINK_host=clang++
        export GYP_DEFINES="clang=1 mac_deployment_target=10.8"
        make native; cp -R out ~/v8; cp -R include ~/v8

* Minimum of XCode 6 required

        cd lfl
        ./imports/build.sh
        touch crawler/crawler.pb.h
        touch crawler/crawler.pb.cc
        cmake .
        make

        cd lfl/fv
        ./pkg/macprep.sh
        ./FusionViewer.app/Contents/MacOS/fv

        ./pkg/macpkg.sh

* OSX installer FusionViewer.dmg results


BUILDING iPhone Device
----------------------

        cd lfl
        cmake -D LFL_IPHONE=1 .
        make

        cd fv
        cp -R assets fv-iphone
        cp lfapp/*.glsl fv-iphone/assets
        ./pkg/iphoneprep.sh
        ./pkg/iphonepkg.sh

        open fv-iphone/fv-iphone.xcodeproj

        [Change configuration to Device]
        [Build and run]
        cp fv fv-iphone/build/Debug-iphoneos/fv-iphone.app/fv-iphone
        cp skorp ~/Library//Developer/Xcode/DerivedData/skorp-iphone-cwokylhxlztdqwhdhxqzpqiemvoz/Build/Products/Debug-iphoneos/skorp-iphone.app/skorp-iphone
        [Build and run]

* iPhone Installer iFusionViewer.ipa results


BUILDING iPhone Simulator
-------------------------

        cd lfl
        cmake -D LFL_IPHONESIM=1 .
        make

        cd fv
        cp -R assets fv-iphone
        ./pkg/iphoneprep.sh
        ./pkg/iphonepkg.sh

        open fv-iphone/fv-iphone.xcodeproj
        [Change configuration to Simulator]
        [Build and run]
        cp fv fv-iphone/build/Debug-iphonesimulator/fv-iphone.app/fv-iphone
        [Build and run]

* iPhone Installer iFusionViewer.ipa results


BUILDING Android
----------------

* http://www.oracle.com/technetwork/java/javase/downloads/java-se-jdk-7-download-432154.html
* http://www.eclipse.org/downloads/packages/eclipse-classic-37/indigor

* Android SDK http://developer.android.com/sdk/index.html
* Android NDK http://developer.android.com/sdk/ndk/index.html
* ADT Plugin http://developer.android.com/sdk/eclipse-adt.html#installing

        $HOME/android-sdk-linux_x86/tools/android
        [Install Platform android-13 + Google Play Services]
        [New Virtual Device]

        $HOME/android-ndk-r8b/build/tools/make-standalone-toolchain.sh \
        --platform=android-8 --install-dir=$HOME/android-toolchain

        cd lfl
        ** Modify ANDROIDROOT in CMakeLists.txt
        cmake -D LFL_ANDROID=1 .
        make

        cd lfl/fv/fv-android/jni
        ../../pkg/androidprebuild.sh
        rm ~/lfl-android/lfl/lfapp/lfjava/lfjava
        cd src && ln -s ~/android-ndk-r9/sources/cxx-stl/gnu-libstdc++ && cd ..
        vim lfapp/Android.mk # Change to: libskorp_lfapp.a
        ndk-build

        cd lfl/fv/fv-android/assets
        cp -R ../../assets .
        cp ../../lfapp/*.glsl assets

        cd lfl/fv/fv-android
        vi local.properties
        ant debug

        Eclipse > [Eclipse|Window] > Preferences > Android > SDK Location
        Eclipse > File > New > Other > Android Project > From Existing Source > fv-android (Name: FusionViwer, Target 2.2)
        FusionViewer > Refresh
        FusionViewer > Debug as Android Application

* Android Installer bin/FusionViewer.apk results

* Setup eclipse_keystore

        FusionViewer > Android Tools > Export Signed Application Package

* Signed Android Installer results


