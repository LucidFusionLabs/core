#!/bin/bash
# $Id: macprepbin.sh 1071 2011-09-10 18:21:18Z justin $

if [ $# -ne 1 ]; then echo "Usage $0 <binary>"; exit 1; fi

IMPORTS=`pwd`/../imports

install_name_tool -change /usr/local/lib/libSDL-1.3.0.dylib @loader_path/../Libraries/libSDL-1.3.0.dylib $1
install_name_tool -change /usr/local/lib/libportaudio.2.dylib @loader_path/../Libraries/libportaudio.2.dylib $1
install_name_tool -change /usr/local/lib/libpng12.0.dylib @loader_path/../Libraries/libpng12.0.dylib $1
install_name_tool -change /usr/local/lib/libjpeg.62.dylib @loader_path/../Libraries/libjpeg.62.dylib $1
install_name_tool -change /usr/local/cuda/lib/libcuda.dylib @loader_path/../Libraries/libcuda.dylib $1
install_name_tool -change @rpath/libcudart.dylib @loader_path/../Libraries/libcudart.dylib $1
install_name_tool -change /usr/local/lib/libmp3lame.0.dylib @loader_path/../Libraries/libmp3lame.0.dylib $1
install_name_tool -change $IMPORTS/OpenCV/lib/libcxcore.2.1.dylib @loader_path/../Libraries/libcxcore.2.1.dylib $1
install_name_tool -change $IMPORTS/OpenCV/lib/libcv.2.1.dylib @loader_path/../Libraries/libcv.2.1.dylib $1
install_name_tool -change $IMPORTS/OpenCV/lib/libhighgui.2.1.dylib  @loader_path/../Libraries/libhighgui.2.1.dylib $1

