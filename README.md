![lfl](assets/lfl.png)

## Overview

[![Build Status](https://travis-ci.org/LucidFusionLabs/core.svg?branch=master)](https://travis-ci.org/LucidFusionLabs/core)

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
* **fs**:           Fusion Sensor, a speech and image recognition client/server
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

* Use CMake 3.2.3

        [select c:\lfl for source code]
        [select c:\lfl\win32 for binaries]
        [Configure]
        [select Visual Studio 2015 generator]
        [uncheck USE_MSVC_RUNTIME_LIBRARY_DLL]
        [Generate]

* Install ActivePerl

        From Visual Studio Command Prompt:
        cd lfl\core\imports\openssl
        perl Configure VC-WIN32 --prefix=C:\lfl\win32\core\imports\openssl
        ms\do_ms
        nmake -f ms\nt.mak 
        nmake -f ms\nt.mak install

* Use Visual Studio C++ 2015
* Tools > Options > Text Editor > All Languages > Tabs > Insert Spaces

        c:\lfl\win32\term\lterm.sln
        [Build lterm]

        cd c:\lfl\win32\term
        mkdir assets
        copy ..\..\term\assets\* assets
        copy ..\..\core\lfapp\*.glsl assets
        Optionally copy ..\..\core\imports\ffmpeg\w32\dll\*.dll Debug [overwrite:All]
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

* http://www.cmake.org/files/v3.2/cmake-3.2.3-Darwin64-universal.dmg
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

* Install node from brew or macports and: npm install -g ios-deploy
* Check logs with: idevicesyslog

        cd lfl
        mkdir iphone && cd iphone
        ** Modify IPHONEROOT in ../core/CMake/iPhoneToolchain.cmake
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../core/CMake/iPhoneToolchain.cmake ..

        cd term
        make lterm_run
        make lterm_pkg

* iPhone Installer iLTerminal.ipa results

### Android

* Install Android Developer Console, Android SDK (android-20 + PlayServices),
  Android NDK, and Gradle

* $HOME/android-ndk-r10d/build/tools/make-standalone-toolchain.sh \
  --platform=android-9 --toolchain=arm-linux-androideabi-4.8 --install-dir=$HOME/android-toolchain

        cd lfl
        mkdir android && cd android
        ** Modify ANDROIDROOT in ../core/CMake/AndroidToolchain.cmake
        cmake -DCMAKE_TOOLCHAIN_FILE=../core/CMake/AndroidToolchain.cmake ..

        cd term
        make lterm_debug
        make lterm_pkg

## Contributing

1. Fork this repo and make changes in your own fork.
2. Commit your changes and push to your fork `git push origin master`
3. Create a new pull request and submit it back to the project.

## License

* GPL
* Unlimited commercial license available for 5$/year.

