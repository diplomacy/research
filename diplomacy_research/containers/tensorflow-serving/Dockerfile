FROM ubuntu:18.04

# To build and push:
# sudo -s
# apt-get update -y
# apt-get install -y docker.io
# echo '{ "experimental": true }' >> /etc/docker/daemon.json
# service docker restart
# docker build --squash -t pcpaquette/tensorflow-serving:20190226 -t pcpaquette/tensorflow-serving:latest .
# docker login
# docker push pcpaquette/tensorflow-serving

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
ENV PYTHONIOENCODING=utf-8
ENV LANG=en_CA.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PATH=/data/env3.7/bin:$PATH

ENV PYTHON_BIN_PATH=/data/env3.7/bin/python
ENV PYTHON_LIB_PATH=/data/env3.7/lib/python3.7/site-packages
ENV TF_NEED_JEMALLOC=1
ENV TF_NEED_IGNITE=1
ENV TF_NEED_AWS=1
ENV TF_NEED_GCP=1
ENV TF_NEED_HDFS=1
ENV TF_NEED_S3=1
ENV TF_NEED_KAFKA=0
ENV TF_ENABLE_XLA=0
ENV TF_NEED_OPENCL=0
ENV TF_NEED_OPENCL_SYCL=0
ENV TF_NEED_ROCM=0
ENV TF_NEED_CUDA=0
ENV TF_DOWNLOAD_CLANG=0
ENV TF_NEED_MPI=0
ENV MPI_HOME="/usr/local/openmpi"
ENV CC_OPT_FLAGS="-march=native"
ENV TF_SET_ANDROID_WORKSPACE=0
ENV GCC_HOST_COMPILER_PATH="/usr/bin/gcc"

ENV BAZEL_VERSION=0.20.0

USER root
RUN FILE_VERSION=2019-02-26 && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing dependencies and Python " && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
    apt-get install -y \
        apt-transport-https \
        autoconf \
        automake \
        build-essential \
        bzip2 \
        ca-certificates \
        clang \
        curl \
        dh-autoreconf \
        g++ \
        gcc \
        git \
        gnupg \
        libc++-dev \
        libcurl4-openssl-dev \
        libfreetype6-dev \
        libgflags-dev \
        libgtest-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        locales \
        make \
        mlocate \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        pkg-config \
        software-properties-common \
        swig \
        unzip \
        virtualenv \
        wget \
        zip \
        zlib1g-dev && \
    locale-gen en_US.UTF-8 && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing Protobuf v3.6.1 " && \
    echo "------------------------------------------------------" && \
    wget https://github.com/google/protobuf/archive/v3.6.1.zip && \
    unzip v3.6.1.zip && \
    rm -Rf v3.6.1.zip && \
    cd protobuf-3.6.1/ && \
    ./autogen.sh && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -Rf protobuf-3.6.1 && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing grpcio v1.15.0 " && \
    echo "------------------------------------------------------" && \
    git clone https://github.com/grpc/grpc.git && \
    cd grpc && \
    echo "Using gRPC v1.15.0... (d2c7d4d)" && \
    git checkout d2c7d4dea492b9a86a53555aabdbfa90c2b01730 && \
    git submodule update --init && \
    cd third_party/protobuf && \
    echo "Using Protobuf v3.6.1... (48cb18e)" && \
    git checkout 48cb18e5c419ddd23d9badcfe4e9df7bde1979b2 && \
    cd ../.. && \
    make && \
    make install && \
    cd .. && \
    rm -Rf grpc && \
    \
    echo "------------------------------------------------------" && \
    echo "Setting up Bazel" && \
    echo "------------------------------------------------------" && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    mkdir -p /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing Python 3.7" && \
    echo "------------------------------------------------------" && \
    mkdir -p /data && \
    cd /data && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/Miniconda3-py37.sh -O miniconda.sh && \
    bash ./miniconda.sh -b -p /data/env3.7 && \
    export PATH="/data/env3.7/bin:$PATH" && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing PIP dependencies" && \
    echo "------------------------------------------------------" && \
    pip install pip six wheel mock requests && \
    pip install 'protobuf==3.6.1' && \
    pip install 'grpcio==1.15.0' && \
    pip install 'grpcio-tools==1.15.0' && \
    pip install 'h5py==2.8.0' && \
    pip install 'keras_applications==1.0.6' --no-deps && \
    pip install 'keras_preprocessing==1.0.5' --no-deps && \
    pip install 'numpy>=1.15,<1.16' && \
    \
    echo "------------------------------------------------------" && \
    echo "Building bazel" && \
    echo "------------------------------------------------------" && \
    cd /bazel && \
    git clone https://github.com/tensorflow/serving && \
    cd serving && \
    echo "# Serving v1.13.0" && \
    git checkout f16e77783927353fca89dbb411fc01cbd3d42bda && \
    git submodule update --init && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/tensorflow/patch/serving-v1.13-20190226.patch && \
    git apply serving-v1.13-20190226.patch && \
    \
    echo "------------------------------------------------------" && \
    echo "Selecting Tensorflow Version" && \
    echo "------------------------------------------------------" && \
    echo "# TensorFlow v1.13.1" && \
    export TF_REVISION=6612da89516247503f03ef76e974b51a434fb52e && \
    rm -Rf tensorflow/ && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout "$TF_REVISION" && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/tensorflow/patch/tensorflow-v1.13-20190226.patch && \
    git apply tensorflow-v1.13-20190226.patch && \
    ./configure && \
    cd ../ && \
    cp tensorflow/.tf_configure.bazelrc ./ && \
    echo "" >> .bazelrc && \
    echo "import /bazel/serving/tensorflow/.tf_configure.bazelrc" >> .bazelrc && \
    \
    echo "------------------------------------------------------" && \
    echo "Building TF Serving" && \
    echo "------------------------------------------------------" && \
    bazel build \
        --color=yes --curses=yes \
        --action_env TF_REVISION="$TF_REVISION" \
        --output_filter=DONT_MATCH_ANYTHING \
        --verbose_failures \
        --spawn_strategy=standalone \
        tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/ && \
    bazel clean --expunge && \
    \
    cd /data && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/serving.zip -O serving.zip && \
    unzip serving.zip && \
    rm -f serving.zip && \
    chmod -R 777 /data/serving/ && \
    \
    echo "------------------------------------------------------" && \
    echo "Cleaning up" && \
    echo "------------------------------------------------------" && \
    mkdir -p /work_dir && \
    chmod -R 777 /work_dir && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -Rf /bazel && \
    rm -Rf /tmp/pip_build_root && \
    rm -Rf ~/.cache/bazel
