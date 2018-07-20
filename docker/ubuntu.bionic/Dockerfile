FROM ubuntu:bionic
LABEL maintainer="LucidFusionLabs@gmail.com"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      yasm automake libtool libalut-dev libglu1-mesa-dev xorg-dev \
      autotools-dev cmake cmake-data make g++ libjpeg-dev libbz2-dev libssl-dev \
      libqt5opengl5 libqt5opengl5-dev && \
    apt-get clean

ENV INSTALL_COMMND apt install -yqV *.deb
