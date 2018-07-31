FROM ubuntu:xenial
LABEL maintainer="support@lucidfusionlabs.com"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      automake autotools-dev make cmake cmake-data g++ libtool yasm \
      libalut-dev libglu1-mesa-dev xorg-dev libjpeg-dev libbz2-dev libssl-dev \
      lsb-release ccache && \
    apt-get clean

ENV PACKAGE_INSTALL apt install -yqV
ENV PACKAGE_SUFFIX deb
