#!/bin/bash

if [ ! -d "$HOME/.grpc" ]; then
    git clone https://github.com/grpc/grpc.git
    cd grpc
    git checkout d2c7d4dea492b9a86a53555aabdbfa90c2b01730   # v1.15.0
    git submodule update --init
    cd third_party/protobuf
    git checkout 48cb18e5c419ddd23d9badcfe4e9df7bde1979b2   # v3.6.1
    cd ../..
    make
    cd ..
    rm -Rf ~/.grpc
    mv grpc ~/.grpc
    echo "Done installing grpc v1.15.0"
else
    echo "grpc is already installed in ~/.grpc"
fi

echo ""
echo "Add the following lines to your .bashrc"
echo "-------------------------------------------"
echo "# gRPC"
echo 'export PATH=$PATH:$HOME/.grpc/bins/opt:$HOME/.grpc/bins/opt/protobuf'
echo 'export CPATH=$CPATH:$HOME/.grpc/include:$HOME/.grpc/third_party/protobuf/src'
echo 'export LIBRARY_PATH=$HOME/.grpc/libs/opt:$HOME/.grpc/libs/opt/protobuf'
echo 'export PKG_CONFIG_PATH=$HOME/.grpc/libs/opt/pkgconfig:$HOME/.grpc/third_party/protobuf'
echo 'export LD_LIBRARY_PATH=$HOME/.grpc/libs/opt'
echo "-------------------------------------------"

echo "If you have an error, run the following:"
echo "apt-get install -y build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev"
echo ""
echo "If you want to reinstall, remove the ~/.grpc directory"
