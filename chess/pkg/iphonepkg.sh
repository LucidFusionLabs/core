#!/bin/bash
# $Id: iphonepkg.sh 1071 2011-09-10 18:21:18Z justin $

if [ ! -d ./pkg ]; then echo "Run $0 from lflpub/your_app_here"; exit 1; fi

PKGNAME=`cat pkg/PkgName.txt`
APPDIR="i$PKGNAME.app"

rm -rf Payload
mkdir -p Payload/
cp -rp "$APPDIR" "Payload/"

IPA=`echo $APPDIR | sed "s/\.app/\.ipa/"`
zip -r $IPA Payload/

