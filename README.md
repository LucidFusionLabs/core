![lfl](www/lfl/lfl.png)

## Overview

[![Build Status](https://travis-ci.org/LucidFusionLabs/core.svg?branch=master)](https://travis-ci.org/LucidFusionLabs/core)

The core API consists of the: `Application`, `Window` classes,
the `Audio`, `Video`, `Input`, `Network`, `Camera`, `CUDA` Modules,
and the `Fonts`, `GUI`, `AssetLoader`, `Scene` subsystems. 

The key implementation files are:
[app/app.h](app/app.h)
[app/app.cpp](app/app.cpp)

## Projects

* **[term](http://lucidfusionlabs.com/terminal)**:          LTerminal, a modern terminal
* **[editor](http://lucidfusionlabs.com/editor)**:          LEditor, a text editor and IDE
* **[browser](http://lucidfusionlabs.com/browser)**:        LBrowser, a HTML4/CSS2 web browser with V8 javascript
* **[chess](http://lucidfusionlabs.com/chess)**:            LChess, a magic bitboard chess engine and FICS client
* **[quake](http://github.com/LucidFusionLabs/quake)**:     LQuake, a quake clone
* **[spaceball](http://spaceballfuture.com)**:              Spaceball Future, a multiplayer 3d game
* **[fs](http://lucidfusionlabs.com/fs)**:                  Fusion Sensor, a speech and image recognition client/server

The following build procedures apply to any app cloned from [new_app_template](new_app_template).
Replace "LTerminal" and "lterm" with "YourPackage" and "YourApp" to build other apps.
See [new_app_template/README.txt](new_app_template/README.txt) to quick start your next app.


## Checkout

`git clone --recursive https://github.com/lucidfusionlabs/lfl.git`


## Get bleeding edge

`cd core; git checkout master; git pull origin master; cd ..`

`cd term; git checkout master; git pull origin master; cd ..`


### Build Windows

* Use CMake 3.2.3

        [select c:\lfl for source code]
        [select c:\lfl\win32 for binaries]
        [Configure]
        [select Visual Studio 2015 generator]
        [uncheck USE_MSVC_RUNTIME_LIBRARY_DLL]
        [set CMAKE_GENERATOR_TOOLSET v140_xp]
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
        copy ..\..\core\app\*.glsl assets
        If using ffmpeg, copy ..\..\core\imports\ffmpeg\w32\dll\*.dll Debug [overwrite:All]
        [Run]

        [Right click] term.nsi > Compile NSIS Script

* Windows installer lterminst.exe results

### Build Linux

* Using Ubuntu 15.04 [VirtualBox Image](http://virtualboxes.org/images/ubuntu)

        sudo apt-get install screen vim git cmake yasm xorg-dev autotools-dev \
        automake flex bison libssl-dev libbz2-dev qtbase5-dev libqt5webkit5-dev

* Also see [.travis.yml](.travis.yml) for package dependencies

        cd lfl
        mkdir linux && cd linux
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../core/CMake/LinuxToolchain.cmake ..

        cd term
        make lterm_run
        make lterm_pkg

* Linux package LTerminal.tgz results

### Build OSX

* http://www.cmake.org/files/v3.2/cmake-3.2.3-Darwin-universal.dmg
* Minimum of XCode 6 required, nasm & yasm (from macports or brew)

        cd lfl
        mkdir osx && cd osx
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../core/CMake/OSXToolchain.cmake ..

        cd term
        make lterm_run
        make lterm_pkg

* OSX installer LTerminal.dmg results
* For libclang build llvm following http://clang.llvm.org/get_started.html
(including libcxx and using -DCMAKE_BUILD_TYPE=Release)
then cmake -DCMAKE_INSTALL_PREFIX=~/llvm -P cmake_install.cmake

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

### Build iOS

* Install node from brew or macports and: npm install -g ios-deploy
* Check logs with: idevicesyslog

        cd lfl
        mkdir ios && cd ios
        ** Modify LFL_IOS_ROOT in ../core/CMake/iOSToolchain.cmake
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../core/CMake/iOSToolchain.cmake ..

        cd term
        make lterm_run
        make lterm_pkg

* iOS Installer iLTerminal.ipa results

### Build Android

* Install Android Developer Console, Android SDK (android-20 + PlayServices),
  Android NDK, and Gradle

* $HOME/android-ndk-r10d/build/tools/make-standalone-toolchain.sh \
  --platform=android-9 --toolchain=arm-linux-androideabi-4.8 --install-dir=$HOME/android-toolchain

        cd lfl
        mkdir android && cd android
        ** Modify LFL_ANDROID_ROOT in ../core/CMake/AndroidToolchain.cmake
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../core/CMake/AndroidToolchain.cmake ..

        cd term
        make lterm_debug
        make lterm_pkg

## Contribute

1. Fork this repo and make changes in your own fork.
2. Commit your changes and push to your fork `git push origin master`
3. Create a new pull request and submit it back to the project.

* Code style should generally follow [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html)

## License

* GPL
* Unlimited commercial licenses available for $25 per platform per year.

