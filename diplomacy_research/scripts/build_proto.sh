#!/usr/bin/env bash
## USAGE: ./build_proto.sh <path_to_protoc>

# Making sure we are using protoc 3.6.1
# -----------------------------------
if ${PROTOC:=protoc} --version | grep 'libprotoc 3.6.1'; then
    echo "Protoc 3.6.1 detected."
else
    echo "You need to use protoc 3.6.1. Update the PROTOC variable to point to the right protoc."
    exit 1
fi

# Finding all protofiles and generating them
echo "-----------------------------------------"
echo "Generating"
PROTO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../proto/" && pwd )"
find $PROTO_DIR -type f -iname "*.proto" -print0 | while IFS= read -r -d $'\0' line; do
    ${PROTOC:=protoc} --proto_path=$PROTO_DIR --cpp_out=$PROTO_DIR --python_out=$PROTO_DIR "$line"
    echo "   * $line"
done

# Compiling gRPC classes
# -----------------------------------
pip install -q 'grpcio-tools==1.15.0'
PROTOC_CPP=$(which grpc_cpp_plugin)
PROTOC_PYTHON=$(which grpc_python_plugin)
PROTOC_PATH="$( cd "$( dirname "${PROTOC_CPP}" )/../../" && pwd )"
python -m grpc.tools.protoc --proto_path=$PROTO_DIR --proto_path=$PROTOC_PATH --plugin=protoc-gen-grpc=$PROTOC_PYTHON --plugin=protoc-gen-cpp=$PROTOC_CPP --cpp_out=$PROTO_DIR --python_out=$PROTO_DIR --grpc_python_out=$PROTO_DIR $PROTO_DIR/tensorflow_serving/apis/*_service.proto

# Done
# -----------------------------------
echo "Don't forget to run 'pip install -e .' to build the Cython extensions."
echo "Done."
