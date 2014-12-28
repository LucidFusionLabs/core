#!/bin/bash
# $Id: macprep.sh 1091 2011-09-13 02:14:44Z justin $

if [ ! -d ./pkg ]; then echo "Run $0 from lflpub/your_app_here"; exit 1; fi

PKGNAME=`head -1 pkg/PkgName.txt`
BINNAME=`head -1 pkg/BinName.txt`

LFLPUBAPP=`pwd`
LFLPUB=`dirname $LFLPUBAPP`

rm -rf $PKGNAME.app
mkdir -p $PKGNAME.app/Contents/MacOS
mkdir -p $PKGNAME.app/Contents/Resources
mkdir -p $PKGNAME.app/Contents/Frameworks
mkdir -p $PKGNAME.app/Contents/Libraries

cat pkg/mac-Info.plist | pkg/pkgsedpipe.sh pkg > $PKGNAME.app/Contents/Info.plist
pkg/macPkgInfo.sh pkg/mac-Info.plist > $PKGNAME.app/Contents/PkgInfo

for f in `cat pkg/BinName.txt `; do
    cp $f $PKGNAME.app/Contents/MacOS/
    pkg/macprepbin.sh $PKGNAME.app/Contents/MacOS/$f
done

cat > $PKGNAME.app/Contents/MacOS/$BINNAME.sh <<EOF
#!/usr/bin/env sh
BASEDIR=\`dirname "\$0"\`
"\$BASEDIR/$BINNAME" >& /tmp/$PKGNAME.log
EOF
chmod +x $PKGNAME.app/Contents/MacOS/$BINNAME.sh

cp assets/icon.icns $PKGNAME.app/Contents/Resources/
cp -r assets $PKGNAME.app/Contents/Resources/
rm -rf $PKGNAME.app/Contents/Resources/assets/.svn
cp ../lfapp/*.glsl $PKGNAME.app/Contents/Resources/assets

cp ../imports/SDL/build/.libs/libSDL-1.3.0.dylib $PKGNAME.app/Contents/Libraries/
cp ../imports/portaudio/lib/.libs/libportaudio.2.dylib $PKGNAME.app/Contents/Libraries/
cp /usr/local/lib/libpng12.0.dylib $PKGNAME.app/Contents/Libraries/
cp /usr/local/lib/libjpeg.62.dylib $PKGNAME.app/Contents/Libraries/
cp ../imports/lame/libmp3lame/.libs/libmp3lame.0.dylib $PKGNAME.app/Contents/Libraries/

cp /usr/local/cuda/lib/libcuda.dylib $PKGNAME.app/Contents/Libraries/
cp /usr/local/cuda/lib/libcudart.dylib $PKGNAME.app/Contents/Libraries/
cp /usr/local/cuda/lib/libtlshook.dylib $PKGNAME.app/Contents/Libraries/
install_name_tool -change @rpath/libtlshook.dylib @loader_path/../Libraries/libtlshook.dylib $PKGNAME.app/Contents/Libraries/libcudart.dylib

cp ../imports/OpenCV/lib/libcxcore.2.1.dylib $PKGNAME.app/Contents/Libraries/
cp ../imports/OpenCV/lib/libcv.2.1.dylib $PKGNAME.app/Contents/Libraries/
install_name_tool -change $LFLPUB/imports/OpenCV/lib/libcxcore.2.1.dylib @loader_path/../Libraries/libcxcore.2.1.dylib $PKGNAME.app/Contents/Libraries/libcv.2.1.dylib

cp ../imports/OpenCV/lib/libhighgui.2.1.dylib $PKGNAME.app/Contents/Libraries/
install_name_tool -change $LFLPUB/imports/OpenCV/lib/libcxcore.2.1.dylib @loader_path/../Libraries/libcxcore.2.1.dylib $PKGNAME.app/Contents/Libraries/libhighgui.2.1.dylib
install_name_tool -change $LFLPUB/imports/OpenCV/lib/libcv.2.1.dylib @loader_path/../Libraries/libcv.2.1.dylib $PKGNAME.app/Contents/Libraries/libhighgui.2.1.dylib

install_name_tool -change $LFLPUB/imports/OpenCV/lib/libcxcore.2.1.dylib @loader_path/../Libraries/libcxcore.2.1.dylib $PKGNAME.app/Contents/MacOS/$BINNAME
install_name_tool -change $LFLPUB/imports/OpenCV/lib/libcv.2.1.dylib @loader_path/../Libraries/libcv.2.1.dylib $PKGNAME.app/Contents/MacOS/$BINNAME
