#!/bin/bash
set -e
mkdir -p assets
mkdir -p res/raw
mkdir -p res/drawable-xhdpi
cp ../assets/* $@ assets
for d in ../drawable-*; do dbn=`basename $d`; if [ -d res/$dbn ]; then cp $d/*.png res/$dbn; cp $d/android/*.png res/$dbn; fi; done
if [ $(find ../assets -name "*.wav" | wc -l) != "0" ]; then cp ../assets/*.wav res/raw; fi
if [ $(find ../assets -name "*.mp3" | wc -l) != "0" ]; then cp ../assets/*.mp3 res/raw; fi
if [ $(find ../assets -name "*.ogg" | wc -l) != "0" ]; then cp ../assets/*.ogg res/raw; fi
