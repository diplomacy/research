FROM ubuntu:18.04

# To build and push:
# sudo -s
# apt-get update -y
# apt-get install -y docker.io
# echo '{ "experimental": true }' >> /etc/docker/daemon.json
# service docker restart
# docker build --squash -t pcpaquette/ubuntu-cuda10:20190226 .
# docker login
# docker push pcpaquette/ubuntu-cuda10

ENV PYTHONIOENCODING=utf-8
ENV LANG=en_CA.UTF-8

ENV PYTHON_BIN_PATH=/data/env3.7/bin/python
ENV PYTHON_LIB_PATH=/data/env3.7/lib/python3.7/site-packages
ENV TF_NEED_JEMALLOC=1
ENV TF_NEED_IGNITE=1
ENV TF_NEED_AWS=1
ENV TF_NEED_GCP=1
ENV TF_NEED_HDFS=1
ENV TF_NEED_S3=1
ENV TF_NEED_KAFKA=1
ENV TF_ENABLE_XLA=1
ENV TF_NEED_OPENCL=0
ENV TF_NEED_OPENCL_SYCL=0
ENV TF_NEED_ROCM=0
ENV TF_NEED_CUDA=1
ENV TF_CUDA_VERSION="10.0"
ENV CUDA_TOOLKIT_PATH="/usr/local/cuda"
ENV TF_CUDNN_VERSION="7"
ENV CUDNN_INSTALL_PATH="/usr/local/cuda"
ENV TF_NEED_TENSORRT=0
ENV TF_NCCL_VERSION="2"
ENV NCCL_INSTALL_PATH=/usr/lib/nccl/lib
ENV NCCL_HDR_PATH=/usr/lib/nccl/include
ENV TF_CUDA_COMPUTE_CAPABILITIES="3.5,3.7,5.2,6.0,6.1,7.0"
ENV TF_CUDA_CLANG=0
ENV TF_DOWNLOAD_CLANG=0
ENV GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
ENV TF_NEED_MPI=0
ENV MPI_HOME="/usr/local/openmpi"
ENV CC_OPT_FLAGS="--copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2"
ENV TF_SET_ANDROID_WORKSPACE=0

# Versions
ENV NCCL_VERSION=2.4.2
ENV CUDA_VERSION=10.0.130
ENV CUDNN_VERSION=7.4.1.5
ENV NVIDIA_DRIVER=410
ENV BAZEL_VERSION=0.20.0
ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1

USER root
RUN FILE_VERSION=2019-02-26 && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing dependencies and Python (psmisc -> fuser)  " && \
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
        gfortran \
        git \
        gnupg \
        libatlas-base-dev \
        libblas-dev \
        libc++-dev \
        libcupti-dev \
        libcurl4-openssl-dev \
        libfreetype6-dev \
        libgflags-dev \
        libgtest-dev \
        liblapack-dev \
        libncurses5-dev \
        libopenblas-dev \
        libpng-dev \
        libtool \
        libxft-dev \
        libzmq3-dev \
        locales \
        make \
        mlocate \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        pkg-config \
        psmisc \
        software-properties-common \
        swig \
        unzip \
        virtualenv \
        wget \
        zip \
        zlib1g-dev && \
    locale-gen en_US.UTF-8 && \
    \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "------------------------------------------------------" && \
    echo "Installing CUDA 10.0.130" && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get -y update && \
    apt-get install -y \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-toolkit-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-cublas-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION && \
    ln -s cuda-10.0 /usr/local/cuda && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing NCCL 2.4" && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
    apt-get -y install libnccl2=${NCCL_VERSION}-1+cuda10.0 libnccl-dev=${NCCL_VERSION}-1+cuda10.0 && \
    mkdir -p ${NCCL_INSTALL_PATH} && \
    mkdir -p ${NCCL_HDR_PATH} && \
    ln -s /usr/include/nccl.h ${NCCL_HDR_PATH}/nccl.h && \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so ${NCCL_INSTALL_PATH}/libnccl.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.${TF_NCCL_VERSION} ${NCCL_INSTALL_PATH}/libnccl.so.${TF_NCCL_VERSION} && \
    ldconfig && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing CuDNN 7.4.1" && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
    apt-get -y install libcudnn7=${CUDNN_VERSION}-1+cuda10.0 libcudnn7-dev=${CUDNN_VERSION}-1+cuda10.0 && \
    mkdir /usr/lib/x86_64-linux-gnu/include/ && \
    ln -s /usr/lib/x86_64-linux-gnu/include/cudnn.h /usr/lib/x86_64-linux-gnu/include/cudnn.h && \
    ln -s /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.${TF_CUDNN_VERSION} /usr/local/cuda/lib64/libcudnn.so.${TF_CUDNN_VERSION} && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing VERBS" && \
    echo "------------------------------------------------------" && \
    apt-get install -y alien && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/libibverbs-13-7.el7.x86_64.rpm && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/libibumad-13-7.el7.x86_64.rpm && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/libibverbs-utils-13-7.el7.x86_64.rpm && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/librdmacm-13-7.el7.x86_64.rpm && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/libibcm-13-7.el7.x86_64.rpm && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/ibacm-13-7.el7.x86_64.rpm && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/libnl3-3.2.28-4.el7.x86_64.rpm && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/verbs/rdma-core-devel-13-7.el7.x86_64.rpm && \
    alien -i libibverbs-13-7.el7.x86_64.rpm && \
    alien -i libibumad-13-7.el7.x86_64.rpm && \
    alien -i libibverbs-utils-13-7.el7.x86_64.rpm && \
    alien -i librdmacm-13-7.el7.x86_64.rpm && \
    alien -i libibcm-13-7.el7.x86_64.rpm && \
    alien -i ibacm-13-7.el7.x86_64.rpm && \
    alien -i libnl3-3.2.28-4.el7.x86_64.rpm && \
    alien -i rdma-core-devel-13-7.el7.x86_64.rpm && \
    rm libibverbs-13-7.el7.x86_64.rpm && \
    rm libibumad-13-7.el7.x86_64.rpm && \
    rm libibverbs-utils-13-7.el7.x86_64.rpm && \
    rm librdmacm-13-7.el7.x86_64.rpm && \
    rm libibcm-13-7.el7.x86_64.rpm && \
    rm ibacm-13-7.el7.x86_64.rpm && \
    rm libnl3-3.2.28-4.el7.x86_64.rpm && \
    rm rdma-core-devel-13-7.el7.x86_64.rpm && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 && \
    \
    find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm -f /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "------------------------------------------------------" && \
    echo "Installing Protobuf 3.6.1" && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
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
    echo "Installing gRPC v1.15.0" && \
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
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "------------------------------------------------------" && \
    echo "Installing OpenMPI 3.0.0" && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
    wget -nv https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar -xzf openmpi-3.0.0.tar.gz && \
    cd openmpi-3.0.0 && \
    ./configure --prefix=/usr/local/openmpi --enable-orterun-prefix-by-default --enable-mpirun-prefix-by-default --enable-static --enable-shared --with-cuda --with-verbs && \
    make all install && \
    cd .. && \
    rm -Rf openmpi-3.0.0 && \
    rm openmpi-3.0.0.tar.gz && \
    \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "------------------------------------------------------" && \
    echo "Installing Redis 5.0" && \
    echo "------------------------------------------------------" && \
    add-apt-repository -y ppa:chris-lea/redis-server && \
    apt-get -y update && \
    apt-get install -y redis-server && \
    echo vm.overcommit_memory = 1 >> /etc/sysctl.conf && \
    sysctl vm.overcommit_memory=1 && \
    \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# Tensorflow v1.13.1
ENV TF_REVISION="6612da89516247503f03ef76e974b51a434fb52e"
RUN echo "------------------------------------------------------" && \
    echo "Compiling Tensorflow" && \
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
    apt-get install -y nvidia-driver-$NVIDIA_DRIVER && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    export CUDA_HOME=/usr/local/cuda && \
    export CUDA_ROOT=$CUDA_HOME && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib64 && \
    \
    mkdir -p /data && \
    cd /data && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/Miniconda3-py37.sh -O miniconda.sh && \
    bash ./miniconda.sh -b -p /data/env3.7 && \
    export PATH="/data/env3.7/bin:$PATH" && \
    pip install pip six wheel mock requests && \
    pip install 'protobuf==3.6.1' && \
    pip install 'grpcio==1.15.0' && \
    pip install 'grpcio-tools==1.15.0' && \
    pip install 'h5py==2.8.0' && \
    pip install 'keras_applications==1.0.6' --no-deps && \
    pip install 'keras_preprocessing==1.0.5' --no-deps && \
    pip install 'numpy>=1.15,<1.16' && \
    \
    cd /bazel && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout $TF_REVISION && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/tensorflow/patch/tensorflow-v1.13-20190226.patch && \
    git apply tensorflow-v1.13-20190226.patch && \
    ./configure && \
    bazel build \
        --verbose_failures -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 \
        --config=cuda --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
        -k //tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg && \
    cp /tmp/tensorflow_pkg/*.whl /data && \
    cd .. && \
    rm -Rf tensorflow/ && \
    \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# Tensorflow Serving 1.13.1
ENV TF_REVISION_SERVING="f16e77783927353fca89dbb411fc01cbd3d42bda"
RUN echo "------------------------------------------------------" && \
    echo "Compiling Tensorflow Serving" && \
    echo "------------------------------------------------------" && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    export PATH="/data/env3.7/bin:$PATH" && \
    export CUDA_HOME=/usr/local/cuda && \
    export CUDA_ROOT=$CUDA_HOME && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib64 && \
    \
    mkdir -p /bazel && \
    cd /bazel && \
    git clone https://github.com/tensorflow/serving && \
    cd serving && \
    echo "# Serving v1.13.0" && \
    git checkout $TF_REVISION_SERVING && \
    git submodule update --init && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/tensorflow/patch/serving-v1.13-20190226.patch && \
    git apply serving-v1.13-20190226.patch && \
    \
    rm -Rf tensorflow/ && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout $TF_REVISION && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/tensorflow/patch/tensorflow-v1.13-20190226.patch && \
    git apply tensorflow-v1.13-20190226.patch && \
    ./configure && \
    cd ../ && \
    cp tensorflow/.tf_configure.bazelrc ./ && \
    echo "" >> .bazelrc && \
    echo "import /bazel/serving/tensorflow/.tf_configure.bazelrc" >> .bazelrc && \
    \
    bazel build \
        --color=yes --curses=yes --config=cuda --copt="-fPIC" \
        --action_env TF_REVISION="$TF_REVISION" --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
        --output_filter=DONT_MATCH_ANYTHING \
        --verbose_failures -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 \
        --spawn_strategy=standalone --crosstool_top=@local_config_cuda//crosstool:toolchain \
        tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/ && \
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel clean --expunge && \
    \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "------------------------------------------------------" && \
    echo "Final Cleanup" && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
    apt-get install -y sudo && \
    apt-get purge -y nvidia-driver-$NVIDIA_DRIVER && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -Rf /bazel && \
    rm -Rf /tmp/tensorflow_pkg && \
    rm -Rf /tmp/pip_build_root && \
    rm -Rf ~/.cache/bazel && \
    rm -Rf /data/env*
