## Overview

[![Build Status](https://travis-ci.org/LucidFusionLabs/core.svg?branch=master)](https://travis-ci.org/LucidFusionLabs/core)

The core API consists of the: `Application`, and `Window` classes,
the `Audio`, `Video`, `Input`, `SocketServices`, `Camera`, and `CUDA` Modules,
the `Framework`, `Fonts`, `GUI`, `Loader`, `Scene`, `IPC`, and `Crypto` subsystems,
and runs on OS X, Windows, Linux, iOS, and Android.

The key implementation files are: 
[app/app.h](app/app.h)            
[app/app.cpp](app/app.cpp)        
                                  
## Projects                       

* **[LTerminal](http://lucidfusionlabs.com/LTerminal)**:             modern terminal
* **[TepidFusion](http://lucidfusionlabs.com/TepidFusion)**:         text editor and IDE
* **[TinyBrowser](http://lucidfusionlabs.com/TinyBrowser)**:         HTML4/CSS2 web browser with V8 javascript
* **[AncientChess](http://lucidfusionlabs.com/AncientChess)**:       magic bitboard chess engine and FICS client
* **[FusionSensor](http://lucidfusionlabs.com/FusionSensor)**:       speech and image recognition client/server
* **[SpaceballFuture](http://lucidfusionlabs.com/SpaceballFuture)**: multiplayer 3d game

The following build procedures apply to any app cloned from [new_app_template](new_app_template).
Replace "LTerminal" with "YourApp" to build other apps.
See [new_app_template/README.txt](new_app_template/README.txt) to quick start your next app.


## Checkout

`git clone --recursive https://github.com/lucidfusionlabs/lfl.git`


## Get bleeding edge

`cd core; git checkout master; git pull origin master; cd ..`

`cd LTerminal; git checkout master; git pull origin master; cd ..`


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

        c:\lfl\win32\LTerminal\LTerminal.sln
        [Build LTerminal]

        cd c:\lfl\win32\LTerminal
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

        cd LTerminal
        make LTerminal_run
        make LTerminal_pkg

* Linux package LTerminal.tgz results

### Build OSX

* https://cmake.org/files/v3.6/cmake-3.6.3-Darwin-x86_64.dmg
* Minimum of XCode 6 required, nasm & yasm (from macports or brew)
* Make sure /usr/bin is before /opt/local/bin in PATH

        cd lfl
        mkdir osx && cd osx
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../core/CMake/OSXToolchain.cmake ..

        cd LTerminal
        make LTerminal_run
        make LTerminal_pkg

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

        cd LTerminal
        make LTerminal_run
        make LTerminal_pkg

* iOS Installer iLTerminal.ipa results

### Build Android

* Install Android Developer Console, Android SDK (android-23 + PlayServices),
  Android NDK, and Gradle

* $HOME/android-ndk-r13b/build/tools/make_standalone_toolchain.py \
  --arch arm --api 9 --install-dir $HOME/android-toolchain

        cd lfl
        mkdir android && cd android
        ** Modify LFL_ANDROID_ROOT in ../core/CMake/AndroidToolchain.cmake
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../core/CMake/AndroidToolchain.cmake ..

        cd LTerminal
        make LTerminal_release

## Contribute

1. Fork this repo and make changes in your own fork.
2. Commit your changes and push to your fork `git push origin master`
3. Create a new pull request and submit it back to the project.

* Code style should generally follow [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html)

## License

* GPL
* Unlimited commercial licenses available for $25 per platform per year.

