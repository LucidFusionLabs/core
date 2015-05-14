![lfl](assets/lfl.png)

## Overview

[![Build Status](https://travis-ci.org/lucidfusionlabs/core.svg?branch=master)](https://travis-ci.org/lucidfusionlabs/core)

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

`git clone https://github.com/lucidfusionlabs/lfl.git`
`git submodule update --init --recursive`

LFL builds easily for Windows, Linux, Mac OSX, iPhone and Android.

* Replace "LTerminal" and "lterm" with "YourPackage" and "YourApp" to build other apps.

### Windows

* use CMake 3.0.2

        [select c:\lfl for source and binaries]
        [Configure]
        [uncheck USE_MSVC_RUNTIME_LIBRARY_DLL]
        [Generate]

        start Visual Studio Command Prompt
        cd lfl\core\imports\judy\src
        build.bat

* use Visual Studio C++ 2013 Express
* Tools > Options > Text Editor > All Languages > Tabs > Insert Spaces

        c:\lfl\term\Project.sln
        [Build LTerminal]

        cd c:\lfl\term
        copy ..\core\lfapp\*.glsl assets
        copy ..\core\imports\ffmpeg\w32\dll\*.dll Debug [overwrite:All]
        [Run]

        [Right click] term.nsi > Compile NSIS Script

* Windows installer lterminst.exe results

### Linux

* See [.travis.yml](.travis.yml) for package dependencies

        cd lfl
        mkdir linux && cd linux
        cmake ..

        cd term
        make lterm_run
        make lterm_pkg

* Linux package LTerminal.tgz results

### OSX

* http://www.cmake.org/files/v3.0/cmake-3.0.2-Darwin64-universal.dmg
* Minimum of XCode 6 required, nasm & yasm from macports

        cd lfl
        mkdir osx && cd osx
        cmake ..

        cd term
        make lterm_run
        make lterm_pkg

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
        ** Modify IPHONEROOT in ../core/CMake/iPhoneToolchain.cmake
        cmake -DCMAKE_TOOLCHAIN_FILE=../core/CMake/iPhoneToolchain.cmake ..

        cd term
        make lterm_run
        make lterm_pkg

* iPhone Installer iLTerminal.ipa results

### Android

* http://www.oracle.com/technetwork/java/javase/downloads/java-se-jdk-7-download-432154.html
* http://www.eclipse.org/downloads/packages/eclipse-classic-37/indigor

* Android SDK http://developer.android.com/sdk/index.html
* Android NDK http://developer.android.com/sdk/ndk/index.html
* ADT Plugin http://developer.android.com/sdk/eclipse-adt.html#installing

        $HOME/android-sdk-linux_x86/tools/android
        [Install Platform android-20 + Google Play Services]
        [New Virtual Device]
        $HOME/android-ndk-r10d/build/tools/make-standalone-toolchain.sh \
        --platform=android-8 --toolchain=arm-linux-androideabi-4.8 --install-dir=$HOME/android-toolchain

        cd lfl
        mkdir android && cd android
        ** Modify ANDROIDROOT in ../core/CMake/AndroidToolchain.cmake
        cmake -DCMAKE_TOOLCHAIN_FILE=../core/CMake/AndroidToolchain.cmake ..

        cd term
        make lterm_debug
        make lterm_pkg

* Android Installer build/bin/LTerminal.apk results

* Setup eclipse_keystore

        LTerminal > Android Tools > Export Signed Application Package

* Signed Android Installer results

## Contributing

1. Fork this repo and make changes in your own fork.
2. Commit your changes and push to your fork `git push origin master`
3. Create a new pull request and submit it back to the project.

