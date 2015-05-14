#!/bin/bash
# $Id: clone.sh 1264 2011-11-24 08:43:37Z justin $

if [ $# -ne 3 ]; then echo "Usage: $0 <org> <package> <binary>"; echo "Eg: $0 com.lucidfusionlabs FusionViewer fv"; exit 1; fi

ORGNAME=$1
PKGNAME=$2
BINNAME=$3

TEMPLATEDIR=`dirname $0`
if [ "$TEMPLATEDIR" = "" -o "$TEMPLATEDIR" = "." ]; then echo "Can't run from new_app_template directory"; exit 1; fi

TEMPLATEFILES=`find $TEMPLATEDIR/* | grep -v "/\." | grep -v "/clone.sh" | grep -v "/pkg/" | grep -v "/assets/" | grep -v "README.txt"`

echo "core/new_app_template/clone.sh: Cloning new app"
echo "Domain: $ORGNAME"
echo "Package: $PKGNAME"
echo "Directory/Binary: $BINNAME"

mkdir $BINNAME || { echo "CLONE FAILED: mkdir $BINNAME"; exit 1; }

cp -R $TEMPLATEDIR/assets $BINNAME

for f in $TEMPLATEFILES; do
    t=`echo $f | sed -e s,"$TEMPLATEDIR","$BINNAME", -e s,"new_app_template","$BINNAME",g`

    let len=${#f}-4
    suffix=${f:$len}

    if [ `basename $f` = "assets" ]; then
        true
    elif [ -d $f ]; then
        mkdir $t
    elif [ "$suffix" = ".png" ] || [ "$suffix" = ".dll" ]; then
        cp $f $t
    else
        cat $f | sed -e "s/\$PKGNAME/$PKGNAME/g" -e "s/\$BINNAME/$BINNAME/g" -e "s/\$ORGNAME/$ORGNAME/g" > $t
    fi
done

echo "core/new_app_template/clone.sh: Successfully cloned new_app_template $BINNAME"

