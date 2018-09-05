#!/bin/bash

for f in $2; do
  FPW=`sips -g pixelWidth  $f | tail -1 | awk '{print $2}'`
  FPH=`sips -g pixelHeight $f | tail -1 | awk '{print $2}'`
  echo $f width=$FPW height=$FPH
  convert $1 -filter lanczos2 -format png -quality 100 -resize ${FPW}x${FPH}\! "$f"
done

