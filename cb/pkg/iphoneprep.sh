#!/bin/bash
# $Id: iphoneprep.sh 1124 2011-09-18 04:46:34Z justin $

if [ ! -d ./pkg ]; then echo "Run $0 from lflpub/your_app_here"; exit 1; fi

PKGNAME=`head -1 pkg/PkgName.txt`
BINNAME=`head -1 pkg/BinName.txt`

rm -rf i$PKGNAME.app
mkdir -p i$PKGNAME.app/Contents

# manifest
cat pkg/iphone-Info.plist | pkg/pkgsedpipe.sh pkg > i$PKGNAME.app/Contents/Info.plist
pkg/macPkgInfo.sh pkg/iphone-Info.plist > i$PKGNAME.app/PkgInfo

# executable
for f in `cat pkg/BinName.txt `; do
    cp $f i$PKGNAME.app/
    chmod +x i$PKGNAME.app/$f
done

#cp assets/icon.icns $PKGNAME.app/Contents/Resources/
cp -r assets i$PKGNAME.app/
rm -rf i$PKGNAME.app/assets/.svn

# symlink for codesign resource
mkdir -p i$PKGNAME.app/_CodeSignature
touch i$PKGNAME.app/_CodeSignature/CodeResources
ln -s _CodeSignature/CodeResources i$PKGNAME.app/CodeResources

### TODO: copy embedded.mobileprovision

#export CODESIGN_ALLOCATE=/Developer/Platforms/iPhoneSimulator.platform/Developer/usr/bin/codesign_allocate
export CODESIGN_ALLOCATE=/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/codesign_allocate
codesign -f -s "iPhone Developer" i$PKGNAME.app

