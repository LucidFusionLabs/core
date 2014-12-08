#!/bin/bash
# $Id: macPkgInfo.sh 1070 2011-09-10 18:20:20Z justin $

if [ $# -ne 1 ]; then echo "Usage $0 <Info.plist>"; exit 1; fi

INPUT=$1
PACKAGETYPE=`cat $1 | grep -A1 CFBundlePackageType | tail -1 | cut -f2 -d\> | cut -f1 -d \<`
SIGNATURE=`cat $1 | grep -A1 CFBundleSignature | tail -1 | cut -f2 -d\> | cut -f1 -d \<`

echo -n $PACKAGETYPE$SIGNATURE

