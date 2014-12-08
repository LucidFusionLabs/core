#!/bin/bash
# $Id: pkgsedpipe.sh 1091 2011-09-13 02:14:44Z justin $

if [ $# -ne 1 ]; then echo "Usage $0 <pkgdir>"; exit 1; fi

PKGDIR=$1
PKGNAME=`head -1 $PKGDIR/PkgName.txt`
BINNAME=`head -1 $PKGDIR/BinName.txt`
ORGNAME=`head -1 $PKGDIR/OrgName.txt`

sed -e "s/\$PKGNAME/$PKGNAME/g" -e "s/\$BINNAME/$BINNAME/g" -e "s/\$ORGNAME/$ORGNAME/g"

