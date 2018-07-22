FROM centos:7.4.1708
LABEL maintainer="support@lucidfusionlabs.com"

RUN yum install -y epel-release && \
    yum install -y centos-release-scl && \
    yum install -y llvm-toolset-7 gcc-c++ cmake3 openssl-devel bzip2-devel \
    libpng-devel libjpeg-devel freetype-devel fontconfig-devel mesa-libGL-devel mesa-libGLU-devel \
    libXrandr-devel libXi-devel libXv-devel libXt-devel libXcursor-devel \
    redhat-lsb rpm-build && \
    ln -s /usr/bin/cmake3 /usr/bin/cmake

ENV PACKAGE_INSTALL yum -y install
ENV PACKAGE_SUFFIX rpm
