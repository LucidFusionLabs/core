#!/bin/sh
if [ $# -ne 1 ]; then echo "Usage: $0 <output-prefix> e.g. $0 ~/lfl/spaceball/assets/sbmaps"; exit 1; fi
cp ./Image.app/Contents/Resources/assets/png_atlas00.png $1,0,0,0,0,000.png
cp ./Image.app/Contents/Resources/assets/png_atlas.0000.glyphs.matrix $1,0,0,0,0,0.0000.glyphs.matrix
