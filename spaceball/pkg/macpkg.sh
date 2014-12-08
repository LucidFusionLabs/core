#!/bin/bash
# $Id: macpkg.sh 1071 2011-09-10 18:21:18Z justin $

if [ ! -d ./pkg ]; then echo "Run $0 from lflpub/your_app_here"; exit 1; fi

PKGNAME=`head -1 pkg/PkgName.txt`
BINNAME=`head -1 pkg/BinName.txt`

umount /Volumes/$PKGNAME
rm -rf $PKGNAME.dmg $PKGNAME.sparseimage

hdiutil create -size 60m -type SPARSE -fs HFS+ -volname $PKGNAME -attach $PKGNAME.sparseimage

bless --folder /Volumes/$PKGNAME --openfolder /Volumes/$PKGNAME
cp -r $PKGNAME.app /Volumes/$PKGNAME/
ln -s /Applications /Volumes/$PKGNAME/.

hdiutil eject /Volumes/$PKGNAME

hdiutil convert $PKGNAME.sparseimage -format UDBZ -o $PKGNAME.dmg

