![lfl](assets/lfl.png)

## Overview

[![Build Status](https://travis-ci.org/koldfuzor/lfl.svg?branch=master)](https://travis-ci.org/koldfuzor/lfl)

The API primarly consists of the: `Application`, `Window`, and `Scene` classes,
plus the 7 modules: `Audio`, `Video`, `Input`, `Assets`, `Network`, `Camera`,
and `CUDA`.

The key implementation files are:
[lfapp/lfapp.h](lfapp/lfapp.h)
[lfapp/lfapp.cpp](lfapp/lfapp.cpp)

## Projects

* **term**:         LTerminal, a modern terminal
* **editor**:       LEditor, a text editor and IDE
* **browser**:      LBrowser, a HTML4/CSS2 web browser with V8 javascript
* **image**:        LImage, an image and 3D-model manipulation utility
* **fs**:           Fusion Server, a speech and image recognition server
* **fv**:           Fusion Viewer, a speech and image recognition client
* **market**:       Financial data visualization and automated trading code
* **spaceball**:    Spaceball Future, a multiplayer 3d game
* **cb**:           Crystal Bawl, a geopacket visualization screensaver
* **chess**:        LChess, a magic bitboard chess engine and FICS client
* **quake**:        LQuake, a quake clone
* **senators**:     IRC bots with NLP capabilties

The following build procedures apply to any app cloned from [new_app_template](new_app_template).
See [new_app_template/README.txt](new_app_template/README.txt) to quick start your next app.

## Building

`git clone https://github.com/koldfuzor/lfl.git`

LFL builds easily for Windows, Linux, Mac OSX, iPhone and Android.

* Replace "LTerminal" and "lterm" with "YourPackage" and "YourApp" to build other apps.

### Windows

* use CMake 3.0.2

        [select c:\lfl for source and binaries]
        [Configure]
        [uncheck USE_MSVC_RUNTIME_LIBRARY_DLL]
        [Generate]

        start Visual Studio Command Prompt
        cd lfl\imports\judy\src
        build.bat

* use Visual Studio C++ 2013 Express
* Tools > Options > Text Editor > All Languages > Tabs > Insert Spaces

        c:\lfl\Project.sln
        [Build LTerminal]

        cd c:\lfl\term
        copy ..\debug\*.dll debug
        copy ..\lfapp\*.glsl assets
        copy ..\imports\berkelium\w32\bin\* Debug
        copy ..\imports\ffmpeg\w32\dll\*.dll Debug [overwrite:All]
        [Run]

        [Right click] term.nsi > Compile NSIS Script

* Windows installer lterminst.exe results

### Linux

* See [.travis.yml](.travis.yml) for package dependencies

        cd lfl
        mkdir linux && cd linux
        cmake ..

        cd term
        make -j4
        ./pkg/lin.sh
        export LD_LIBRARY_PATH=./LTerminal
        ./LTerminal/lterm

        tar cvfz LTerminal.tgz LTerminal

* Linux package LTerminal.tgz results

### OSX

* http://www.cmake.org/files/v3.0/cmake-3.0.2-Darwin64-universal.dmg
* Minimum of XCode 6 required, nasm & yasm from macports

        cd lfl
        mkdir osx && cd osx
        cmake ..

        cd term
        make -j4
        ./pkg/macprep.sh
        ./LTerminal.app/Contents/MacOS/lterm

        ./pkg/macpkg.sh

* OSX installer LTerminal.dmg results
* For C++ Interpreter setup ~/cling following http://root.cern.ch/drupal/content/cling-build-instructions
* For V8 Javascript setup ~/v8 following https://developers.google.com/v8/build then:

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

### iOS Device

        cd lfl
        mkdir iphone && cd iphone
        cmake -DCMAKE_TOOLCHAIN_FILE=../CMake/iPhoneToolchain.cmake ..

        cd term
        make -j4
        cp -R assets term-iphone
        cp lfapp/*.glsl term-iphone/assets
        ./pkg/iphoneprep.sh
        ./pkg/iphonepkg.sh

        open term-iphone/term-iphone.xcodeproj

        [Change configuration to Device]
        [Build and run]
        cp lterm term-iphone/build/Debug-iphoneos/term-iphone.app/term-iphone
        cp skorp ~/Library/Developer/Xcode/DerivedData/skorp-iphone-cwokylhxlztdqwhdhxqzpqiemvoz/Build/Products/Debug-iphoneos/skorp-iphone.app/skorp-iphone
        [Build and run]

* iPhone Installer iLTerminal.ipa results

### iOS Simulator

        cd lfl
        mkdir iphonesim && cd iphonesim
        cmake -DCMAKE_TOOLCHAIN_FILE=../CMake/iPhoneToolchain.cmake -DLFL_IPHONESIM=1 ..

        cd term
        make -j4
        cp -R assets term-iphone
        ./pkg/iphoneprep.sh
        ./pkg/iphonepkg.sh

        open term-iphone/term-iphone.xcodeproj
        [Change configuration to Simulator]
        [Build and run]
        cp lterm term-iphone/build/Debug-iphonesimulator/term-iphone.app/term-iphone
        [Build and run]

* iPhone Installer iLTerminal.ipa results

### Android

* http://www.oracle.com/technetwork/java/javase/downloads/java-se-jdk-7-download-432154.html
* http://www.eclipse.org/downloads/packages/eclipse-classic-37/indigor

* Android SDK http://developer.android.com/sdk/index.html
* Android NDK http://developer.android.com/sdk/ndk/index.html
* ADT Plugin http://developer.android.com/sdk/eclipse-adt.html#installing

        $HOME/android-sdk-linux_x86/tools/android
        [Install Platform android-13 + Google Play Services]
        [New Virtual Device]

        $HOME/android-ndk-r10d/build/tools/make-standalone-toolchain.sh \
        --platform=android-8 --toolchain=arm-linux-androideabi-4.8 --install-dir=$HOME/android-toolchain

        cd lfl
        mkdir android && cd android
        ** Modify ANDROIDROOT in ../CMake/AndroidToolchain.cmake
        cmake -DCMAKE_TOOLCHAIN_FILE=../CMake/AndroidToolchain.cmake ..

        cd term
        make -j4
        cd term-android/jni
        ndk-build

        cd lfl/term/term-android/assets
        cp -R ../../assets .
        cp ../../lfapp/*.glsl assets
        cp ../../assets/*.wav ../../assets/*.mp3 ../res/raw/

        cd lfl/term/term-android
        vi local.properties
        ant debug

        Eclipse > [Eclipse|Window] > Preferences > Android > SDK Location
        Eclipse > File > New > Other > Android Project > From Existing Source > term-android (Name: LTerminal, Target 2.2)
        LTerminal > Refresh
        LTerminal > Debug as Android Application

* Android Installer bin/LTerminal.apk results

* Setup eclipse_keystore

        LTerminal > Android Tools > Export Signed Application Package

* Signed Android Installer results

## Contributing

1. Fork this repo and make changes in your own fork.
2. Commit your changes and push to your fork `git push origin master`
3. Create a new pull request and submit it back to the project.

