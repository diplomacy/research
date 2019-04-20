#!/usr/bin/env bash
TF_ARCHIVE="https://github.com/tensorflow/tensorflow/archive/6612da89516247503f03ef76e974b51a434fb52e.zip"          # Tensorflow v1.13.1
TF_SERVING_ARCHIVE="https://github.com/tensorflow/serving/archive/f16e77783927353fca89dbb411fc01cbd3d42bda.zip"     # Serving v1.13.0
PROTO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../proto/" && pwd )"

# Downloading Tensorflow
rm -Rf $PROTO_DIR/temp/
rm -Rf $PROTO_DIR/diplomacy_tensorflow/
wget $TF_ARCHIVE -O tf.zip
unzip tf.zip -d $PROTO_DIR/temp/
mv $PROTO_DIR/temp/*/tensorflow $PROTO_DIR/diplomacy_tensorflow
find $PROTO_DIR/diplomacy_tensorflow/ ! -name '*.proto' -type f -delete
find $PROTO_DIR/diplomacy_tensorflow/ -name '*test*.proto' -type f -delete
find $PROTO_DIR/diplomacy_tensorflow/ -type d -empty -delete
find $PROTO_DIR/diplomacy_tensorflow/ -type f -exec sed -i 's@package tensorflow@package diplomacy.tensorflow@g' {} +
find $PROTO_DIR/diplomacy_tensorflow/ -type f -exec sed -i 's@import "tensorflow@import "diplomacy_tensorflow@g' {} +
find $PROTO_DIR/diplomacy_tensorflow/ -type f -exec sed -i 's@ tensorflow.tf2xla.@ diplomacy.tensorflow.tf2xla.@g' {} +

# Downloading Tensorflow Serving
rm -Rf $PROTO_DIR/temp/
rm -Rf $PROTO_DIR/tensorflow_serving/
wget $TF_SERVING_ARCHIVE -O tf_serving.zip
unzip tf_serving.zip -d $PROTO_DIR/temp/
mv $PROTO_DIR/temp/*/tensorflow_serving $PROTO_DIR/
find $PROTO_DIR/tensorflow_serving/ ! -name '*.proto' -type f -delete
find $PROTO_DIR/tensorflow_serving/ -name '*test*.proto' -type f -delete
find $PROTO_DIR/tensorflow_serving/ -type d -empty -delete
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@import "tensorflow/@import "diplomacy_tensorflow/@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ SignatureDef>@ diplomacy.tensorflow.SignatureDef>@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ tensorflow.Example@ diplomacy.tensorflow.Example@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ TensorProto@ diplomacy.tensorflow.TensorProto@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ NamedTensorProto@ diplomacy.tensorflow.NamedTensorProto@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ RunOptions@ diplomacy.tensorflow.RunOptions@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ RunMetadata@ diplomacy.tensorflow.RunMetadata@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ ConfigProto@ diplomacy.tensorflow.ConfigProto@g' {} +
find $PROTO_DIR/tensorflow_serving/ -type f -exec sed -i 's@ error.Code@ diplomacy.tensorflow.error.Code@g' {} +

# Cleaning up
rm -Rf $PROTO_DIR/temp/
rm ./*.zip
