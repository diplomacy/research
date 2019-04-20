#!/bin/bash

if [ ! -d "$HOME/.protobuf" ]; then
    wget https://github.com/google/protobuf/archive/v3.6.1.zip
    unzip v3.6.1.zip
    rm -Rf v3.6.1.zip
    cd protobuf-3.6.1/
    ./autogen.sh
    ./configure --prefix=$HOME/.protobuf
    make
    make install
    cd ..
    rm -Rf protobuf-3.6.1
    echo "Done installing protobuf 3.6.1"
else
    echo "Protobuf is already installed in ~/.protobuf"
fi

echo ""
echo "Add the following lines to your .bashrc"
echo "-------------------------------------------"
echo "# Protobuf"
echo 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp'
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.protobuf/lib'
echo 'export LIBRARY_PATH=$LIBRARY_PATH:$HOME/.protobuf/lib'
echo 'export CPATH=$HOME/.protobuf/include:$CPATH'
echo 'export PROTOC=~/.protobuf/bin/protoc'
echo 'alias protoc=~/.protobuf/bin/protoc'
echo "-------------------------------------------"

echo "If you have an error, run the following:"
echo "apt-get install -y dh-autoreconf autoconf automake libtool curl make g++ unzip"
echo ""
echo "If you want to reinstall, remove the ~/.protobuf directory"
