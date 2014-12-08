#!/bin/bash
# $Id: lin.sh 1071 2011-09-10 18:21:18Z justin $

if [ ! -d ./pkg ]; then echo "Run $0 from lflpub/your_app_here"; exit 1; fi

PKGNAME=`head -1 pkg/PkgName.txt`
BINNAME=`head -1 pkg/BinName.txt`

rm -rf $PKGNAME
mkdir $PKGNAME

for f in `cat pkg/BinName.txt `; do
    cp $f $PKGNAME
done

cp README.txt $PKGNAME
cp GPL.txt $PKGNAME

mkdir $PKGNAME/assets
cp assets/* $PKGNAME/assets
cp ../lfapp/*.glsl $PKGNAME/assets

cp ../imports/portaudio/lib/.libs/libportaudio.so.2 $PKGNAME
cp ../imports/lame/libmp3lame/.libs/libmp3lame.so.0 $PKGNAME
cp ../imports/x264/libx264.so.142 $PKGNAME
cp ../imports/OpenCV/lib/libcv.so.2.1 $PKGNAME
cp ../imports/OpenCV/lib/libcxcore.so.2.1 $PKGNAME

cp ../imports/ffmpeg/ffplay $PKGNAME
cp ../imports/ffmpeg/ffmpeg $PKGNAME

