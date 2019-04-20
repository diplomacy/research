#!/usr/bin/env bash

# Validating number of arguments
if [ "$#" -ne 5 ]; then
    echo "Expected 5 arguments"
    echo "Syntax: ./run_deploy.sh <REPO> <PR_NUMBER> <BRANCH> <COMMIT_ID> <GITHUB_TOKEN>"
    echo "Use PR_NUMBER=0 if not a PR"
    exit 1
fi

REPO=$1
PR_NUMBER=$2
BRANCH=$3
COMMIT_ID=$4
GITHUB_TOKEN=$5
BUILD_DIR=$(pwd)

# Exiting if we are not building master or dev, or if we are in a pull request
if [[ "$BRANCH" != "master" && "$BRANCH" != "dev" && "$BRANCH" != "build" && "$BRANCH" != "build_cpu" && "$BRANCH" != "web" && "$BRANCH" != "webdip" ]]; then exit; fi
if [ "$PR_NUMBER" != "0" ]; then exit; fi

# Setting variables
mkdir -p $HOME/.container
export COMMIT_ID=${COMMIT_ID:0:7}
echo "export REPO=${REPO}" > $HOME/.container/build_args
echo "export BRANCH=${BRANCH}" >> $HOME/.container/build_args
echo "export COMMIT_ID=${COMMIT_ID}" >> $HOME/.container/build_args
echo "export GITHUB_TOKEN=${GITHUB_TOKEN}" >> $HOME/.container/build_args
cp diplomacy_research/containers/research/redis.conf $HOME/.container/redis.conf

# Installing Google Cloud
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get -y update && sudo apt-get install -y google-cloud-sdk

# ------ Building web container for web.diplomacy.ai ------
if [[ "$BRANCH" == "web" ]]; then
    sudo apt-get install -y docker.io
    cd ${BUILD_DIR}/diplomacy_research/containers/deploy-web/
    sudo docker build -t gcr.io/ppaquette-diplomacy/web-deploy:${COMMIT_ID} --build-arg GITHUB_TOKEN=$GITHUB_TOKEN --build-arg REPO=$REPO --build-arg COMMIT_ID=$COMMIT_ID .
    if [ $? -ne 0 ]; then
        echo "Return code was not zero. The build command failed. Aborting."
        exit 1
    fi
    sudo gcloud auth configure-docker --quiet
    sudo gcloud docker -- push gcr.io/ppaquette-diplomacy/web-deploy:${COMMIT_ID}

    # Deploying
    echo "Deploying docker to web.diplomacy.ai"
    sudo chmod -R 777 ~/.config/gcloud/
    gcloud beta compute instances update-container demo-server --zone northamerica-northeast1-a --container-image=gcr.io/ppaquette-diplomacy/web-deploy:${COMMIT_ID}
    echo "Done deploying gcr.io/ppaquette-diplomacy/web-deploy:${COMMIT_ID} to web.diplomacy.ai"

# ------ Building webdip container for webdip.diplomacy.ai ------
elif [[ "$BRANCH" == "webdip" ]]; then
    sudo apt-get install -y docker.io
    cd ${BUILD_DIR}/diplomacy_research/containers/deploy-webdip/
    sudo docker build -t gcr.io/ppaquette-diplomacy/webdip-deploy:${COMMIT_ID} --build-arg GITHUB_TOKEN=$GITHUB_TOKEN --build-arg REPO=$REPO --build-arg COMMIT_ID=$COMMIT_ID .
    if [ $? -ne 0 ]; then
        echo "Return code was not zero. The build command failed. Aborting."
        exit 1
    fi
    sudo gcloud auth configure-docker --quiet
    sudo gcloud docker -- push gcr.io/ppaquette-diplomacy/webdip-deploy:${COMMIT_ID}

    # Deploying
    echo "Deploying docker to webdip.diplomacy.ai"
    sudo chmod -R 777 ~/.config/gcloud/
    gcloud beta compute instances update-container webdip-server --zone europe-west4-a --container-image=gcr.io/ppaquette-diplomacy/webdip-deploy:${COMMIT_ID}
    echo "Done deploying gcr.io/ppaquette-diplomacy/webdip-deploy:${COMMIT_ID} to webdip.diplomacy.ai"

# ------ Building Singularity container to deploy ------
else
    # Installing Singularity
    cd ${BUILD_DIR}
    ./run_build_singularity.sh

    # Building RESEARCH container
    if [ "$BRANCH" == "build_cpu" ]; then export SUFFIX=".cpu"; else export SUFFIX=""; fi
    export BUILD_RESEARCH="${BUILD_DIR}/build/research/"
    export IMG_RESEARCH="${BUILD_DIR}/build/research/$(TZ='America/Montreal' date +"%Y%m%d_%H%M%S")_${BRANCH}_${COMMIT_ID}.sif${SUFFIX}"
    mkdir -p $BUILD_RESEARCH
    sudo mkdir -p /data/
    sudo wget -nv https://storage.googleapis.com/ppaquette-diplomacy/containers/ubuntu-cuda10/ubuntu-cuda10-20190226.sif -O /data/ubuntu-cuda10-20190226.sif

    sudo singularity build $IMG_RESEARCH ${BUILD_DIR}/diplomacy_research/containers/research/Singularity
    if [ $? -ne 0 ]; then
        echo "Return code was not zero. The build command failed. Aborting."
        exit 1
    fi

    # Uploading
    cd ${BUILD_DIR}
    sudo rm -f /etc/boto.cfg
    sudo rm -Rf ${BUILD_DIR}/build/lib.*
    sudo rm -Rf ${BUILD_DIR}/build/temp.*
    gsutil cp -R ${BUILD_DIR}/build/* gs://ppaquette-research/containers/
fi
