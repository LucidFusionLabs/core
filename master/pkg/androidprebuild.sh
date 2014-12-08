#!/bin/bash
# $Id: androidprebuild.sh 1247 2011-11-19 01:48:49Z justin $

if [ ! -d ../../pkg ]; then echo "Run $0 from lflpub/your_app_here/your_app_here-android/jni"; exit 1; fi

JNI=`pwd`
ANDROIDPROJ=`dirname $JNI`
APPPATH=`dirname $ANDROIDPROJ`
APPDIR=`basename $APPPATH`

ln -s ../../../../lfapp/lfjni/lfjni.cpp src/lfjni.cpp
ln -s ../../../../../lfapp/lfjava ../src/com/lucidfusionlabs/lfjava

ln -s ../../../imports/OpenCV/3rdparty/libpng libpng
ln -s ../../../imports/EASTL EASTL
ln -s ../../../imports/freetype freetype
ln -s ../../../lfapp/ lfapp
ln -s ../../../$APPDIR/ $APPDIR
