#!/bin/bash
# $Id: start.sh 317 2011-03-05 02:14:22Z justin $
# Copyright (C) 2011 Lucid Fusion Labs

aflag=
bflag=
while getopts 'ab:?' OPTION
do case $OPTION in
a)    aflag=1
      ;;
b)    bflag=1
      bval="$OPTARG"
      ;;
?)    printf "Usage: %s: [-a] [-b value] args\n" $(basename $0) >&2
      exit 2
      ;;
esac
done

shift $(($OPTIND - 1))
remaining="$*"

if [ "$aflag" ]
then printf "Option -a specified\n"
fi

if [ "$bflag" ]
then printf 'Option -b "%s" specified\n' "$bval"
fi

printf "Remaining arguments are: %s\n" "$remaining"

