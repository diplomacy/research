#!/bin/bash

# Installing Singularity v3.2.0
export VERSION=v3.2.0
sudo apt-get update -y
sudo apt-get install -y build-essential libssl-dev uuid-dev libgpgme11-dev libseccomp-dev pkg-config squashfs-tools

# Installing GO 1.12.5
export GO_VERSION=1.12.5 OS=linux ARCH=amd64
wget -nv https://dl.google.com/go/go$GO_VERSION.$OS-$ARCH.tar.gz
sudo tar -C /usr/local -xzf go$GO_VERSION.$OS-$ARCH.tar.gz
rm -f go$GO_VERSION.$OS-$ARCH.tar.gz
export GOPATH=$HOME/.go
export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin
mkdir -p $GOPATH
go get github.com/golang/dep/cmd/dep

# Building from source
mkdir -p $GOPATH/src/github.com/sylabs
cd $GOPATH/src/github.com/sylabs
git clone https://github.com/sylabs/singularity.git
cd singularity
git checkout $VERSION
./mconfig -p /usr/local
cd ./builddir
make
sudo make install
echo "Done installing singularity v3.2.0"

echo ""
echo "Add the following lines to your .bashrc"
echo "-------------------------------------------"
echo "# Singularity"
echo 'export GOPATH=$HOME/.go'
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin'
echo "-------------------------------------------"