FROM ubuntu:bionic
LABEL maintainer="support@lucidfusionlabs.com"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      automake autotools-dev make cmake cmake-data g++ libtool yasm \
      libalut-dev libglu1-mesa-dev xorg-dev libjpeg-dev libbz2-dev libssl-dev libqt5opengl5-dev \
      lsb-release ccache && \
    apt-get clean

ENV INSTALL_COMMND apt install -yqV *.deb
